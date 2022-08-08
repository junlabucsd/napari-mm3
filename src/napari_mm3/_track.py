import multiprocessing
import napari
import matplotlib.pyplot as plt
import yaml
import numpy as np
import six
import pickle
import os

from skimage import io
from magicgui import magic_factory
from pathlib import Path
from skimage.measure import regionprops

# TODO: IMO, lineage tracking should be TOTALLY separate from general tracking. Otherwise it gets confusing.
from ._function import (
    information,
    load_specs,
    Cell,
    load_stack,
    load_time_table,
    find_complete_cells,
    plot_lineage_images,
    find_cells_of_birth_label,
)


# functions for checking if a cell has divided or not
# this function should also take the variable t to
# weight the allowed changes by the difference in time as well
def check_growth_by_region(params, cell, region):
    """Checks to see if it makes sense
    to grow a cell by a particular region"""
    # load parameters for checking
    max_growth_length = params["track"]["max_growth_length"]
    min_growth_length = params["track"]["min_growth_length"]
    max_growth_area = params["track"]["max_growth_area"]
    min_growth_area = params["track"]["min_growth_area"]

    # check if length is not too much longer
    if cell.lengths[-1] * max_growth_length < region.major_axis_length:
        return False

    # check if it is not too short (cell should not shrink really)
    if cell.lengths[-1] * min_growth_length > region.major_axis_length:
        return False

    # check if area is not too great
    if cell.areas[-1] * max_growth_area < region.area:
        return False

    # check if area is not too small
    if cell.lengths[-1] * min_growth_area > region.area:
        return False

    # # check if y position of region is within
    # # the quarter positions of the bounding box
    # lower_quarter = cell.bboxes[-1][0] + (region.major_axis_length / 4)
    # upper_quarter = cell.bboxes[-1][2] - (region.major_axis_length / 4)
    # if lower_quarter > region.centroid[0] or upper_quarter < region.centroid[0]:
    #     return False

    # check if y position of region is within the bounding box of previous region
    lower_bound = cell.bboxes[-1][0]
    upper_bound = cell.bboxes[-1][2]
    if lower_bound > region.centroid[0] or upper_bound < region.centroid[0]:
        return False

    # return true if you get this far
    return True


# see if a cell has reasonably divided
def check_division(params, cell, region1, region2):
    """Checks to see if it makes sense to divide a
    cell into two new cells based on two regions.

    Return 0 if nothing should happend and regions ignored
    Return 1 if cell should grow by region 1
    Return 2 if cell should grow by region 2
    Return 3 if cell should divide into the regions."""

    # load in parameters
    max_growth_length = params["track"]["max_growth_length"]
    min_growth_length = params["track"]["min_growth_length"]

    # see if either region just could be continued growth,
    # if that is the case then just return
    # these shouldn't return true if the cells are divided
    # as they would be too small
    if check_growth_by_region(params, cell, region1):
        return 1

    if check_growth_by_region(params, cell, region2):
        return 2

    # make sure combined size of daughters is not too big
    combined_size = region1.major_axis_length + region2.major_axis_length
    # check if length is not too much longer
    if cell.lengths[-1] * max_growth_length < combined_size:
        return 0
    # and not too small
    if cell.lengths[-1] * min_growth_length > combined_size:
        return 0

    # centroids of regions should be in the upper and lower half of the
    # of the mother's bounding box, respectively
    # top region within top half of mother bounding box
    if (
        cell.bboxes[-1][0] > region1.centroid[0]
        or cell.centroids[-1][0] < region1.centroid[0]
    ):
        return 0
    # bottom region with bottom half of mother bounding box
    if (
        cell.centroids[-1][0] > region2.centroid[0]
        or cell.bboxes[-1][2] < region2.centroid[0]
    ):
        return 0

    # if you got this far then divide the mother
    return 3


# take info and make string for cell id
def create_cell_id(region, t, peak, fov, experiment_name=None):
    """Make a unique cell id string for a new cell"""
    # cell_id = ['f', str(fov), 'p', str(peak), 't', str(t), 'r', str(region.label)]
    if experiment_name is None:
        cell_id = [
            "f",
            "%02d" % fov,
            "p",
            "%04d" % peak,
            "t",
            "%04d" % t,
            "r",
            "%02d" % region.label,
        ]
        cell_id = "".join(cell_id)
    else:
        cell_id = "{}f{:0=2}p{:0=4}t{:0=4}r{:0=2}".format(
            experiment_name, fov, peak, t, region.label
        )
    return cell_id


# Creates lineage for a single channel
def make_lineage_chnl_stack(params, fov_and_peak_id):
    """
    Create the lineage for a set of segmented images for one channel. Start by making the regions in the first time points potenial cells.
    Go forward in time and map regions in the timepoint to the potential cells in previous time points, building the life of a cell.
    Used basic checks such as the regions should overlap, and grow by a little and not shrink too much. If regions do not link back in time, discard them.
    If two regions map to one previous region, check if it is a sensible division event.

    Parameters
    ----------
    fov_and_peak_ids : tuple.
        (fov_id, peak_id)

    Returns
    -------
    Cells : dict
        A dictionary of all the cells from this lineage, divided and undivided

    """

    # load in parameters
    # if leaf regions see no action for longer than this, drop them
    lost_cell_time = params["track"]["lost_cell_time"]
    # only cells with y positions below this value will recieve the honor of becoming new
    # cells, unless they are daughters of current cells
    new_cell_y_cutoff = params["track"]["new_cell_y_cutoff"]
    # only regions with labels less than or equal to this value will be considered to start cells
    new_cell_region_cutoff = params["track"]["new_cell_region_cutoff"]

    # get the specific ids from the tuple
    fov_id, peak_id = fov_and_peak_id

    # TODO: Get this to be passed in?
    time_table = load_time_table(params["ana_dir"])
    # start time is the first time point for this series of TIFFs.
    start_time_index = min(time_table[fov_id].keys())

    information("Creating lineage for FOV %d, channel %d." % (fov_id, peak_id))

    # load segmented data
    image_data_seg = load_stack(
        params, fov_id, peak_id, color=params["track"]["seg_img"]
    )
    # image_data_seg = load_stack(params, fov_id, peak_id, color='seg')

    # Calculate all data for all time points.
    # this list will be length of the number of time points
    regions_by_time = [
        regionprops(label_image=timepoint) for timepoint in image_data_seg
    ]  # removed coordinates='xy'

    # Set up data structures.
    Cells = {}  # Dict that holds all the cell objects, divided and undivided
    cell_leaves = []  # cell ids of the current leaves of the growing lineage tree

    # go through regions by timepoint and build lineages
    # timepoints start with the index of the first image
    for t, regions in enumerate(regions_by_time, start=start_time_index):
        # if there are cell leaves who are still waiting to be linked, but
        # too much time has passed, remove them.
        for leaf_id in cell_leaves:
            if t - Cells[leaf_id].times[-1] > lost_cell_time:
                cell_leaves.remove(leaf_id)

        # make all the regions leaves if there are no current leaves
        if not cell_leaves:
            for region in regions:
                if (
                    region.centroid[0] < new_cell_y_cutoff
                    and region.label <= new_cell_region_cutoff
                ):
                    # Create cell and put in cell dictionary
                    cell_id = create_cell_id(region, t, peak_id, fov_id)
                    Cells[cell_id] = Cell(
                        time_table, cell_id, region, t, parent_id=None
                    )

                    # add thes id to list of current leaves
                    cell_leaves.append(cell_id)

        # Determine if the regions are children of current leaves
        else:
            ### create mapping between regions and leaves
            leaf_region_map = {}
            leaf_region_map = {leaf_id: [] for leaf_id in cell_leaves}

            # get the last y position of current leaves and create tuple with the id
            current_leaf_positions = [
                (leaf_id, Cells[leaf_id].centroids[-1][0]) for leaf_id in cell_leaves
            ]

            # go through regions, they will come off in Y position order
            for r, region in enumerate(regions):
                # create tuple which is cell_id of closest leaf, distance
                current_closest = (None, float("inf"))

                # check this region against all positions of all current leaf regions,
                # find the closest one in y.
                for leaf in current_leaf_positions:
                    # calculate distance between region and leaf
                    y_dist_region_to_leaf = abs(region.centroid[0] - leaf[1])

                    # if the distance is closer than before, update
                    if y_dist_region_to_leaf < current_closest[1]:
                        current_closest = (leaf[0], y_dist_region_to_leaf)

                # update map with the closest region
                leaf_region_map[current_closest[0]].append((r, y_dist_region_to_leaf))

            # go through the current leaf regions.
            # limit by the closest two current regions if there are three regions to the leaf
            for leaf_id, region_links in six.iteritems(leaf_region_map):
                if len(region_links) > 2:
                    closest_two_regions = sorted(region_links, key=lambda x: x[1])[:2]
                    # but sort by region order so top region is first
                    closest_two_regions = sorted(
                        closest_two_regions, key=lambda x: x[0]
                    )
                    # replace value in dictionary
                    leaf_region_map[leaf_id] = closest_two_regions

                    # for the discarded regions, put them as new leaves
                    # if they are near the closed end of the channel
                    discarded_regions = sorted(region_links, key=lambda x: x[1])[2:]
                    for discarded_region in discarded_regions:
                        region = regions[discarded_region[0]]
                        if (
                            region.centroid[0] < new_cell_y_cutoff
                            and region.label <= new_cell_region_cutoff
                        ):
                            cell_id = create_cell_id(region, t, peak_id, fov_id)
                            Cells[cell_id] = Cell(
                                time_table, cell_id, region, t, parent_id=None
                            )
                            cell_leaves.append(cell_id)  # add to leaves
                        else:
                            # since the regions are ordered, none of the remaining will pass
                            break

            ### iterate over the leaves, looking to see what regions connect to them.
            for leaf_id, region_links in six.iteritems(leaf_region_map):

                # if there is just one suggested descendant,
                # see if it checks out and append the data
                if len(region_links) == 1:
                    region = regions[
                        region_links[0][0]
                    ]  # grab the region from the list

                    # check if the pairing makes sense based on size and position
                    # this function returns true if things are okay
                    if check_growth_by_region(params, Cells[leaf_id], region):
                        # grow the cell by the region in this case
                        Cells[leaf_id].grow(time_table, region, t)

                # there may be two daughters, or maybe there is just one child and a new cell
                elif len(region_links) == 2:
                    # grab these two daughters
                    region1 = regions[region_links[0][0]]
                    region2 = regions[region_links[1][0]]

                    # check_division returns 3 if cell divided,
                    # 1 if first region is just the cell growing and the second is trash
                    # 2 if the second region is the cell, and the first is trash
                    # or 0 if it cannot be determined.
                    check_division_result = check_division(
                        params, Cells[leaf_id], region1, region2
                    )

                    if check_division_result == 3:
                        # create two new cells and divide the mother
                        daughter1_id = create_cell_id(region1, t, peak_id, fov_id)
                        daughter2_id = create_cell_id(region2, t, peak_id, fov_id)
                        Cells[daughter1_id] = Cell(
                            time_table, daughter1_id, region1, t, parent_id=leaf_id
                        )
                        Cells[daughter2_id] = Cell(
                            time_table, daughter2_id, region2, t, parent_id=leaf_id
                        )
                        Cells[leaf_id].divide(
                            Cells[daughter1_id], Cells[daughter2_id], t
                        )

                        # remove mother from current leaves
                        cell_leaves.remove(leaf_id)

                        # add the daughter ids to list of current leaves if they pass cutoffs
                        if (
                            region1.centroid[0] < new_cell_y_cutoff
                            and region1.label <= new_cell_region_cutoff
                        ):
                            cell_leaves.append(daughter1_id)

                        if (
                            region2.centroid[0] < new_cell_y_cutoff
                            and region2.label <= new_cell_region_cutoff
                        ):
                            cell_leaves.append(daughter2_id)

                    # 1 means that daughter 1 is just a continuation of the mother
                    # The other region should be a leaf it passes the requirements
                    elif check_division_result == 1:
                        Cells[leaf_id].grow(time_table, region1, t)

                        if (
                            region2.centroid[0] < new_cell_y_cutoff
                            and region2.label <= new_cell_region_cutoff
                        ):
                            cell_id = create_cell_id(region2, t, peak_id, fov_id)
                            Cells[cell_id] = Cell(
                                time_table, cell_id, region2, t, parent_id=None
                            )
                            cell_leaves.append(cell_id)  # add to leaves

                    # ditto for 2
                    elif check_division_result == 2:
                        Cells[leaf_id].grow(time_table, region2, t)

                        if (
                            region1.centroid[0] < new_cell_y_cutoff
                            and region1.label <= new_cell_region_cutoff
                        ):
                            cell_id = create_cell_id(region1, t, peak_id, fov_id)
                            Cells[cell_id] = Cell(
                                time_table, cell_id, region1, t, parent_id=None
                            )
                            cell_leaves.append(cell_id)  # add to leaves

    # return the dictionary with all the cells
    return Cells


# finds lineages for all peaks in a fov
def make_lineages_fov(params, fov_id, specs):
    """
    For a given fov, create the lineages from the segmented images.

    Called by
    mm3_Segment.py

    Calls
    mm3.make_lineage_chnl_stack
    """
    ana_peak_ids = []  # channels to be analyzed
    for peak_id, spec in six.iteritems(specs[fov_id]):
        if spec == 1:  # 1 means analyze
            ana_peak_ids.append(peak_id)
    ana_peak_ids = sorted(ana_peak_ids)  # sort for repeatability

    information(
        "Creating lineage for FOV %d with %d channels." % (fov_id, len(ana_peak_ids))
    )

    # just break if there are no peaks to analize
    if not ana_peak_ids:
        # returning empty dictionary will add nothing to current cells dictionary
        return {}

    # This is a list of tuples (fov_id, peak_id) to send to the Pool command
    fov_and_peak_ids_list = [(fov_id, peak_id) for peak_id in ana_peak_ids]

    # # set up multiprocessing pool. will complete pool before going on
    # pool = Pool(processes=params['num_analyzers'])

    # # create the lineages for each peak individually
    # # the output is a list of dictionaries
    # lineages = pool.map(make_lineage_chnl_stack, fov_and_peak_ids_list, chunksize=8)

    # pool.close() # tells the process nothing more will be added.
    # pool.join() # blocks script until everything has been processed and workers exit

    # This is the non-parallelized version (useful for debug)
    lineages = []
    for fov_and_peak_ids in fov_and_peak_ids_list:
        lineages.append(make_lineage_chnl_stack(params, fov_and_peak_ids))

    # combine all dictionaries into one dictionary
    Cells = {}  # create dictionary to hold all information
    for cell_dict in lineages:  # for all the other dictionaries in the list
        Cells.update(cell_dict)  # updates Cells with the entries in cell_dict

    return Cells


def Lineage(params):
    # plotting lineage trees for complete cells
    # load specs file
    with open(os.path.join(params["ana_dir"], "specs.yaml"), "r") as specs_file:
        specs = yaml.safe_load(specs_file)
    with open(os.path.join(params["cell_dir"], "all_cells.pkl"), "rb") as cell_file:
        Cells = pickle.load(cell_file)
    with open(
        os.path.join(params["cell_dir"], "complete_cells.pkl"), "rb"
    ) as cell_file:
        Cells2 = pickle.load(cell_file)
        Cells2 = find_cells_of_birth_label(Cells2, label_num=[1, 2])

    lin_dir = os.path.join(
        params["experiment_directory"], params["analysis_directory"], "lineages"
    )
    if not os.path.exists(lin_dir):
        os.makedirs(lin_dir)

    for fov_id in sorted(specs.keys()):
        fov_id_d = fov_id
        # determine which peaks are to be analyzed (those which have been subtracted)
        for peak_id, spec in six.iteritems(specs[fov_id]):
            if (
                spec == 1
            ):  # 0 means it should be used for empty, -1 is ignore, 1 is analyzed
                peak_id_d = peak_id
                sample_img = load_stack(params, fov_id, peak_id)
                peak_len = np.shape(sample_img)[0]

                break
        break

    viewer = napari.current_viewer()
    viewer.layers.clear()
    # need this to avoid vispy bug for some reason
    # related to https://github.com/napari/napari/issues/2584
    viewer.add_image(np.zeros((1, 1)))
    viewer.layers.clear()

    fig, ax = plot_lineage_images(
        params, Cells, fov_id_d, peak_id_d, Cells2, bgcolor=params["phase_plane"]
    )
    lin_filename = params["experiment_name"] + "_xy%03d_p%04d_lin.png" % (
        fov_id,
        peak_id,
    )
    lin_filepath = os.path.join(lin_dir, lin_filename)
    fig.savefig(lin_filepath, dpi=75)
    plt.close(fig)

    img = io.imread(lin_filepath)
    viewer = napari.current_viewer()
    imgs = []
    # get height of image

    # get the length of each peak
    for i in range(0, len(img[0]), int(len(img[0]) / peak_len) + 1):
        crop_img = img[:, i : i + 300, :]
        if len(crop_img[0]) == 300:
            imgs.append(crop_img)

    img_stack = np.stack(imgs, axis=0)

    viewer.add_image(img_stack, name=lin_filename)

    information("Completed Plotting")


def Track_Cells(params):
    # Load the project parameters file
    information("Loading experiment parameters.")
    p = params

    if p["FOV"]:
        if "-" in p["FOV"]:
            user_spec_fovs = range(
                int(p["FOV"].split("-")[0]), int(p["FOV"].split("-")[1]) + 1
            )
        else:
            user_spec_fovs = [int(val) for val in p["FOV"].split(",")]
    else:
        user_spec_fovs = []

    information("Using {} threads for multiprocessing.".format(p["num_analyzers"]))

    information("Using {} images for tracking.".format(p["track"]["seg_img"]))

    # create segmenteation and cell data folder if they don't exist
    if not os.path.exists(p["seg_dir"]) and p["output"] == "TIFF":
        os.makedirs(p["seg_dir"])
    if not os.path.exists(p["cell_dir"]):
        os.makedirs(p["cell_dir"])

    # load specs file
    specs = load_specs(params)

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    ### Create cell lineages from segmented images
    information("Creating cell lineages using standard algorithm.")

    # Load time table, which goes into params
    # time_table = load_time_table(params)

    # This dictionary holds information for all cells
    Cells = {}

    # do lineage creation per fov, so pooling can be done by peak
    for fov_id in fov_id_list:
        # update will add the output from make_lineages_function, which is a
        # dict of Cell entries, into Cells
        Cells.update(make_lineages_fov(params, fov_id, specs))

    information("Finished lineage creation.")

    ### Now prune and save the data.
    information("Curating and saving cell data.")

    # this returns only cells with a parent and daughters
    Complete_Cells = find_complete_cells(Cells)

    ### save the cell data. Use the script mm3_OutputData for additional outputs.
    # All cell data (includes incomplete cells)
    with open(p["cell_dir"] + "/all_cells.pkl", "wb") as cell_file:
        pickle.dump(Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    # Just the complete cells, those with mother and daugther
    # This is a dictionary of cell objects.
    with open(os.path.join(p["cell_dir"], "complete_cells.pkl"), "wb") as cell_file:
        pickle.dump(Complete_Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    information("Finished curating and saving cell data.")


def track_update_params(
    experiment_name,
    experiment_directory,
    image_directory,
    analysis_directory,
    FOV,
    phase_plane,
    pxl2um,
    lost_cell_time,
    new_cell_y_cutoff,
    new_cell_region_cutoff,
    max_growth_length,
    min_growth_length,
    seg_img,
):
    params = dict()
    params["experiment_name"] = experiment_name
    params["experiment_directory"] = experiment_directory
    params["image_directory"] = image_directory
    params["analysis_directory"] = analysis_directory
    params["FOV"] = FOV
    params["phase_plane"] = phase_plane
    params["pxl2um"] = pxl2um
    params["output"] = "TIFF"
    params["num_analyzers"] = multiprocessing.cpu_count()
    params["track"] = dict()
    params["track"]["lost_cell_time"] = lost_cell_time
    params["track"]["new_cell_y_cutoff"] = new_cell_y_cutoff
    params["track"]["new_cell_region_cutoff"] = new_cell_region_cutoff
    params["track"]["max_growth_length"] = max_growth_length
    params["track"]["min_growth_length"] = min_growth_length
    params["track"]["max_growth_area"] = max_growth_length
    params["track"]["min_growth_area"] = min_growth_length
    if seg_img == "Otsu":
        params["track"]["seg_img"] = "seg_otsu"
    elif seg_img == "U-net":
        params["track"]["seg_img"] = "seg_unet"

    # useful folder shorthands for opening files
    params["TIFF_dir"] = os.path.join(
        params["experiment_directory"], params["image_directory"]
    )
    params["ana_dir"] = os.path.join(
        params["experiment_directory"], params["analysis_directory"]
    )
    params["hdf5_dir"] = os.path.join(params["ana_dir"], "hdf5")
    params["chnl_dir"] = os.path.join(params["ana_dir"], "channels")
    params["empty_dir"] = os.path.join(params["ana_dir"], "empties")
    params["sub_dir"] = os.path.join(params["ana_dir"], "subtracted")
    params["seg_dir"] = os.path.join(params["ana_dir"], "segmented")
    params["cell_dir"] = os.path.join(params["ana_dir"], "cell_data")
    params["track_dir"] = os.path.join(params["ana_dir"], "tracking")

    return params


@magic_factory(
    seg_img={
        "choices": ["Otsu", "U-net"],
        "tooltip": "Segmentation mechanism that was used.",
    },
    working_directory={
        "mode": "d",
        "tooltip": "Directory within which all your data and analyses will be located.",
    },
    phase_plane={"choices": ["c1", "c2", "c3"], "tooltip": "Phase contrast plane"},
    output_prefix={"tooltip": "Optional. Prefix for output files"},
    image_directory={
        "tooltip": "Required. Location (within working directory) for the input images. 'working directory/TIFF/' by default."
    },
    analysis_directory={
        "tooltip": "Required. Location (within working directory) for outputting analysis. 'working directory/analysis/' by default."
    },
    FOV_range={
        "tooltip": "Optional. Range of FOVs to include. By default, all will be processed. E.g. '1-9' or '2,3,6-8'."
    },
    pxl2um={"tooltip": "Micrometers per pixel ('PiXel To Micrometer)"},
)
def Track(
    working_directory: Path = Path(),
    output_prefix: str = "",
    image_directory: str = "TIFF/",
    analysis_directory: str = "analysis/",
    FOV_range: str = "1",
    pxl2um: float = 0.11,
    phase_plane="c1",
    lost_cell_time: int = 3,
    new_cell_y_cutoff: int = 150,
    new_cell_region_cutoff: float = 4,
    max_growth_length: float = 1.5,
    min_growth_length: float = 0.7,
    seg_img="Otsu",
):
    """Performs Mother Machine Analysis"""
    params = track_update_params(
        output_prefix,
        working_directory,
        image_directory,
        analysis_directory,
        FOV_range,
        phase_plane,
        pxl2um,
        lost_cell_time,
        new_cell_y_cutoff,
        new_cell_region_cutoff,
        max_growth_length,
        min_growth_length,
        seg_img,
    )
    Track_Cells(params)
    Lineage(params)
