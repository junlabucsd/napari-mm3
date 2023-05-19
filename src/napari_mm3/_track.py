import multiprocessing
from multiprocessing import Pool
import napari
import matplotlib.pyplot as plt
import yaml
import numpy as np
import six
import pickle
import os
import matplotlib as mpl
import seaborn as sns
import matplotlib.patches as mpatches

from skimage import io
from skimage.measure import regionprops
from napari.utils import progress

from ._deriving_widgets import (
    MM3Container,
    PlanePicker,
    FOVChooser,
    load_specs,
    information,
    load_stack_params,
    warning,
)
from magicgui.widgets import FloatSpinBox, SpinBox, ComboBox, CheckBox, PushButton

from .utils import (
    Cell,
    find_complete_cells,
    find_cells_of_birth_label,
    find_cells_of_fov_and_peak,
    write_cells_to_json,
)

# load the time table
def load_time_table(ana_dir):
    """Add the time table dictionary to the params global dictionary.
    This is so it can be used during Cell creation.
    """

    # try first for yaml, then for pkl
    try:
        with open(ana_dir / "time_table.yaml", "rb") as time_table_file:
            return yaml.safe_load(time_table_file)
    except:
        with open(ana_dir / "time_table.pkl", "rb") as time_table_file:
            return pickle.load(time_table_file)


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
    image_data_seg = load_stack_params(
        params, fov_id, peak_id, postfix=params["track"]["seg_img"]
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
                        params["pxl2um"], time_table, cell_id, region, t, parent_id=None
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
                                params["pxl2um"],
                                time_table,
                                cell_id,
                                region,
                                t,
                                parent_id=None,
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
                            params["pxl2um"],
                            time_table,
                            daughter1_id,
                            region1,
                            t,
                            parent_id=leaf_id,
                        )
                        Cells[daughter2_id] = Cell(
                            params["pxl2um"],
                            time_table,
                            daughter2_id,
                            region2,
                            t,
                            parent_id=leaf_id,
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
                                params["pxl2um"],
                                time_table,
                                cell_id,
                                region2,
                                t,
                                parent_id=None,
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
                                params["pxl2um"],
                                time_table,
                                cell_id,
                                region1,
                                t,
                                parent_id=None,
                            )
                            cell_leaves.append(cell_id)  # add to leaves

    ## plot kymograph with lineage overlay & save it out
    make_lineage_plot(params, fov_id, peak_id, Cells)
    
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

    ## This is a list of tuples (fov_id, peak_id) to send to the Pool command
    fov_and_peak_ids_list = [(fov_id, peak_id) for peak_id in ana_peak_ids]

    # # set up multiprocessing pool. will complete pool before going on
    # pool = Pool(processes=params['num_analyzers'])

    # # create the lineages for each peak individually
    # # the output is a list of dictionaries
    # lineages = pool.map(make_lineage_chnl_stack, params, fov_and_peak_ids_list, chunksize=8)

    # pool.close() # tells the process nothing more will be added.
    # pool.join() # blocks script until everything has been processed and workers exit

    # # This is the non-parallelized version (useful for debug)
    lineages = []
    for fov_and_peak_ids in progress(fov_and_peak_ids_list):
        lineages.append(make_lineage_chnl_stack(params, fov_and_peak_ids))

    # combine all dictionaries into one dictionary
    Cells = {}  # create dictionary to hold all information
    for cell_dict in lineages:  # for all the other dictionaries in the list
        Cells.update(cell_dict)  # updates Cells with the entries in cell_dict

    return Cells


def make_lineage_plot(params, fov_id, peak_id, Cells):
    """Produces a lineage image for the first valid FOV containing cells"""
    # plotting lineage trees for complete cells

    lin_dir = params["ana_dir"] / "lineages"
    if not os.path.exists(lin_dir):
        os.makedirs(lin_dir)

    fig, ax = plot_lineage_images(
        params, Cells, fov_id, peak_id, bgcolor=params["phase_plane"]
    )
    lin_filename = f'{params["experiment_name"]}_{fov_id}_{peak_id}.tif'
    lin_filepath = lin_dir / lin_filename
    fig.savefig(lin_filepath, dpi=75)
    plt.close(fig)


def load_lineage_image(params, fov_id, peak_id):
    lin_dir = params["ana_dir"] / "lineages"

    lin_filename = f'{params["experiment_name"]}_{fov_id}_{peak_id}.tif'
    lin_filepath = lin_dir / lin_filename

    img = io.imread(lin_filepath)
    imgs = []

    crop = min(2000, len(img[0]))
    for i in range(0, len(img[0]), int(len(img[0]) / 50) + 1):
        crop_img = img[:, i : i + crop, :]
        if len(crop_img[0]) == crop:
            imgs.append(crop_img)

    img_stack = np.stack(imgs, axis=0)

    return img_stack


def plot_lineage_images(
    params,
    Cells,
    fov_id,
    peak_id,
    Cells2=None,
    bgcolor="c1",
    fgcolor="seg",
    plot_tracks=True,
    trim_time=False,
    time_set=(0, 100),
    t_adj=1,
):
    """
    Plot linages over images across time points for one FOV/peak.
    Parameters
    ----------
    bgcolor : Designation of background to use. Subtracted images look best if you have them.
    fgcolor : Designation of foreground to use. This should be a segmented image.
    Cells2 : second set of linages to overlay. Useful for comparing lineage output.
    plot_tracks : bool
        If to plot cell traces or not.
    t_adj : int
        adjust time indexing for differences between t index of image and image number
    """

    # filter cells
    Cells = find_cells_of_fov_and_peak(Cells, fov_id, peak_id)

    # load subtracted and segmented data
    image_data_bg = load_stack_params(params, fov_id, peak_id, postfix=bgcolor)

    if fgcolor:
        image_data_seg = load_stack_params(params, fov_id, peak_id, postfix=fgcolor)

    if trim_time:
        image_data_bg = image_data_bg[time_set[0] : time_set[1]]
        if fgcolor:
            image_data_seg = image_data_seg[time_set[0] : time_set[1]]

    n_imgs = image_data_bg.shape[0]
    image_indicies = range(n_imgs)

    if fgcolor:
        # calculate the regions across the segmented images
        regions_by_time = [regionprops(timepoint) for timepoint in image_data_seg]

        # Color map for good label colors
        vmin = 0.5  # values under this color go to black
        vmax = 100  # max y value
        cmap = mpl.colors.ListedColormap(sns.husl_palette(vmax, h=0.5, l=0.8, s=1))
        cmap.set_under(color="black")

    # Trying to get the image size down
    figxsize = image_data_bg.shape[2] * n_imgs / 100.0
    figysize = image_data_bg.shape[1] / 100.0

    # plot the images in a series
    fig, axes = plt.subplots(
        ncols=n_imgs,
        nrows=1,
        figsize=(figxsize, figysize),
        facecolor="black",
        edgecolor="black",
    )
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    transFigure = fig.transFigure.inverted()

    # change settings for each axis
    ax = axes.flat  # same as axes.ravel()
    for a in ax:
        a.set_axis_off()
        a.set_aspect("equal")
        ttl = a.title
        ttl.set_position([0.5, 0.05])

    for i in image_indicies:
        ax[i].imshow(image_data_bg[i], cmap=plt.cm.gray, aspect="equal")

        if fgcolor:
            # make a new version of the segmented image where the
            # regions are relabeled by their y centroid position.
            # scale it so it falls within 100.
            seg_relabeled = image_data_seg[i].copy().astype(float)
            for region in regions_by_time[i]:
                rescaled_color_index = (
                    region.centroid[0] / image_data_seg.shape[1] * vmax
                )
                seg_relabeled[seg_relabeled == region.label] = (
                    int(rescaled_color_index) - 0.1
                )  # subtract small value to make it so there is not overlabeling
            ax[i].imshow(seg_relabeled, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)

        ax[i].set_title(str(i + t_adj), color="white")

    # Annotate each cell with information
    if plot_tracks:
        for cell_id in Cells:
            for n, t in enumerate(Cells[cell_id].times):
                t -= t_adj  # adjust for special indexing

                # don't look at time points out of the interval
                if trim_time:
                    if t < time_set[0] or t >= time_set[1] - 1:
                        break

                x = Cells[cell_id].centroids[n][1]
                y = Cells[cell_id].centroids[n][0]

                # add a circle at the centroid for every point in this cell's life
                circle = mpatches.Circle(
                    xy=(x, y), radius=2, color="white", lw=0, alpha=0.5
                )
                ax[t].add_patch(circle)

                # draw connecting lines between the centroids of cells in same lineage
                try:
                    if n < len(Cells[cell_id].times) - 1:
                        # coordinates of the next centroid
                        x_next = Cells[cell_id].centroids[n + 1][1]
                        y_next = Cells[cell_id].centroids[n + 1][0]
                        t_next = (
                            Cells[cell_id].times[n + 1] - t_adj
                        )  # adjust for special indexing

                        # get coordinates for the whole figure
                        coord1 = transFigure.transform(
                            ax[t].transData.transform([x, y])
                        )
                        coord2 = transFigure.transform(
                            ax[t_next].transData.transform([x_next, y_next])
                        )

                        # create line
                        line = mpl.lines.Line2D(
                            (coord1[0], coord2[0]),
                            (coord1[1], coord2[1]),
                            transform=fig.transFigure,
                            color="white",
                            lw=1,
                            alpha=0.5,
                        )

                        # add it to plot
                        fig.lines.append(line)
                except:
                    pass

                # draw connecting between mother and daughters
                try:
                    if n == len(Cells[cell_id].times) - 1 and Cells[cell_id].daughters:
                        # daughter ids
                        d1_id = Cells[cell_id].daughters[0]
                        d2_id = Cells[cell_id].daughters[1]

                        # both daughters should have been born at the same time.
                        t_next = Cells[d1_id].times[0] - t_adj

                        # coordinates of the two daughters
                        x_d1 = Cells[d1_id].centroids[0][1]
                        y_d1 = Cells[d1_id].centroids[0][0]
                        x_d2 = Cells[d2_id].centroids[0][1]
                        y_d2 = Cells[d2_id].centroids[0][0]

                        # get coordinates for the whole figure
                        coord1 = transFigure.transform(
                            ax[t].transData.transform([x, y])
                        )
                        coordd1 = transFigure.transform(
                            ax[t_next].transData.transform([x_d1, y_d1])
                        )
                        coordd2 = transFigure.transform(
                            ax[t_next].transData.transform([x_d2, y_d2])
                        )

                        # create line and add it to plot for both
                        for coord in [coordd1, coordd2]:
                            line = mpl.lines.Line2D(
                                (coord1[0], coord[0]),
                                (coord1[1], coord[1]),
                                transform=fig.transFigure,
                                color="white",
                                lw=1,
                                alpha=0.5,
                                ls="dashed",
                            )
                            # add it to plot
                            fig.lines.append(line)
                except:
                    pass

        # this is for plotting the traces from a second set of cells
        if Cells2 and plot_tracks:
            Cells2 = find_cells_of_fov_and_peak(Cells2, fov_id, peak_id)
            for cell_id in Cells2:
                for n, t in enumerate(Cells2[cell_id].times):
                    t -= t_adj

                    # don't look at time points out of the interval
                    if trim_time:
                        if t < time_set[0] or t >= time_set[1] - 1:
                            break

                    x = Cells2[cell_id].centroids[n][1]
                    y = Cells2[cell_id].centroids[n][0]

                    # add a circle at the centroid for every point in this cell's life
                    circle = mpatches.Circle(
                        xy=(x, y), radius=2, color="yellow", lw=0, alpha=0.25
                    )
                    ax[t].add_patch(circle)

                    # draw connecting lines between the centroids of cells in same lineage
                    try:
                        if n < len(Cells2[cell_id].times) - 1:
                            # coordinates of the next centroid
                            x_next = Cells2[cell_id].centroids[n + 1][1]
                            y_next = Cells2[cell_id].centroids[n + 1][0]
                            t_next = Cells2[cell_id].times[n + 1] - t_adj

                            # get coordinates for the whole figure
                            coord1 = transFigure.transform(
                                ax[t].transData.transform([x, y])
                            )
                            coord2 = transFigure.transform(
                                ax[t_next].transData.transform([x_next, y_next])
                            )

                            # create line
                            line = mpl.lines.Line2D(
                                (coord1[0], coord2[0]),
                                (coord1[1], coord2[1]),
                                transform=fig.transFigure,
                                color="yellow",
                                lw=1,
                                alpha=0.25,
                            )

                            # add it to plot
                            fig.lines.append(line)
                    except:
                        pass

                    # draw connecting between mother and daughters
                    try:
                        if (
                            n == len(Cells2[cell_id].times) - 1
                            and Cells2[cell_id].daughters
                        ):
                            # daughter ids
                            d1_id = Cells2[cell_id].daughters[0]
                            d2_id = Cells2[cell_id].daughters[1]

                            # both daughters should have been born at the same time.
                            t_next = Cells2[d1_id].times[0] - t_adj

                            # coordinates of the two daughters
                            x_d1 = Cells2[d1_id].centroids[0][1]
                            y_d1 = Cells2[d1_id].centroids[0][0]
                            x_d2 = Cells2[d2_id].centroids[0][1]
                            y_d2 = Cells2[d2_id].centroids[0][0]

                            # get coordinates for the whole figure
                            coord1 = transFigure.transform(
                                ax[t].transData.transform([x, y])
                            )
                            coordd1 = transFigure.transform(
                                ax[t_next].transData.transform([x_d1, y_d1])
                            )
                            coordd2 = transFigure.transform(
                                ax[t_next].transData.transform([x_d2, y_d2])
                            )

                            # create line and add it to plot for both
                            for coord in [coordd1, coordd2]:
                                line = mpl.lines.Line2D(
                                    (coord1[0], coord[0]),
                                    (coord1[1], coord[1]),
                                    transform=fig.transFigure,
                                    color="yellow",
                                    lw=1,
                                    alpha=0.25,
                                    ls="dashed",
                                )
                                # add it to plot
                                fig.lines.append(line)
                    except:
                        pass

    return fig, ax


def Track_Cells(params):
    # Load the project parameters file
    information("Loading experiment parameters.")
    p = params

    user_spec_fovs = params["FOV"]

    information("Using {} threads for multiprocessing.".format(p["num_analyzers"]))

    information("Using {} images for tracking.".format(p["track"]["seg_img"]))

    # create segmenteation and cell data folder if they don't exist
    if not os.path.exists(p["seg_dir"]) and p["output"] == "TIFF":
        os.makedirs(p["seg_dir"])
    if not os.path.exists(p["cell_dir"]):
        os.makedirs(p["cell_dir"])

    # load specs file
    specs = load_specs(params["ana_dir"])

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

    ### save the cell data
    # All cell data (includes incomplete cells)
    with open(p["cell_dir"] / "all_cells.pkl", "wb") as cell_file:
        pickle.dump(Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    ### save to .json
    write_cells_to_json(Cells, p["cell_dir"] / "all_cells.json")

    # Just the complete cells, those with mother and daughter
    # This is a dictionary of cell objects.
    with open(p["cell_dir"] / "complete_cells.pkl", "wb") as cell_file:
        pickle.dump(Complete_Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    ### save to .json
    write_cells_to_json(Complete_Cells, p["cell_dir"] / "complete_cells.json")

    information("Finished curating and saving cell data.")


class Track(MM3Container):
    def create_widgets(self):
        self.viewer.text_overlay.visible = False
        """Overriding method. Widget constructor. See _deriving_widgets.MM3Container for more details."""
        self.fov_widget = FOVChooser(self.valid_fovs)
        self.pxl2um_widget = FloatSpinBox(
            label="um per pixel",
            min=0.0,
            max=2.0,
            step=0.001,
            value=0.11,
            tooltip="Micrometers per pixel",
        )
        self.phase_plane_widget = PlanePicker(
            self.valid_planes,
            label="phase plane",
            tooltip="The phase plane in the experiment.",
        )
        self.lost_cell_time_widget = SpinBox(
            label="lost cell time",
            min=1,
            max=10000,
            value=3,
            tooltip="Number of frames after which a cell is dropped if no new regions "
            "connect to it",
        )
        self.new_cell_y_cutoff_widget = SpinBox(
            label="new cell y cutoff",
            min=1,
            max=20000,
            value=150,
            tooltip="regions only less than this value down the channel from the"
            "closed end will be considered to start potential new cells."
            "Does not apply to daughters. unit is pixels",
        )
        self.new_cell_region_cutoff_widget = FloatSpinBox(
            label="new cell region cutoff",
            value=4,
            min=0,
            max=1000,
            step=0.1,
            tooltip="only regions with labels less than or equal to this value will "
            "be considered to start potential new cells. Does not apply to daughters",
        )
        self.max_growth_length_widget = FloatSpinBox(
            label="max growth length (ratio)",
            value=1.3,
            min=0,
            max=20,
            tooltip="Maximum increase in length allowed when linked new region to "
            "existing potential cell. Unit is ratio.",
        )
        self.min_growth_length_widget = FloatSpinBox(
            label="min growth length (ratio)",
            value=0.8,
            min=0,
            max=20,
            tooltip="Minimum change in length allowed when linked new region to "
            "existing potential cell. Unit is ratio.",
        )
        self.max_growth_area_widget = FloatSpinBox(
            label="max growth area (ratio)",
            value=1.3,
            min=0,
            max=20,
            tooltip="Maximum change in area allowed when linked new region to "
            "existing potential cell. Unit is ratio.",
        )
        self.min_growth_area_widget = FloatSpinBox(
            label="min growth area (ratio)",
            value=0.8,
            min=0,
            max=20,
            tooltip="Minimum change in area allowed when linked new region to existing potential cell. Unit is ratio.",
        )
        self.segmentation_method_widget = ComboBox(
            label="segmentation method", choices=["Otsu", "U-net"]
        )

        self.run_widget = PushButton(text="Construct lineages")
        self.display_widget = PushButton(text="Display results")
        self.set_display_fovs_widget = FOVChooser(
            self.valid_fovs, custom_label="Display results from FOVs "
        )

        self.fov_widget.connect_callback(self.set_fovs)
        self.pxl2um_widget.changed.connect(self.set_pxl2um)
        self.phase_plane_widget.changed.connect(self.set_phase_plane)
        self.lost_cell_time_widget.changed.connect(self.set_lost_cell_time)
        self.new_cell_y_cutoff_widget.changed.connect(self.set_new_cell_y_cutoff)
        self.new_cell_region_cutoff_widget.changed.connect(
            self.set_new_cell_region_cutoff
        )
        self.max_growth_length_widget.changed.connect(self.set_max_growth_length)
        self.min_growth_length_widget.changed.connect(self.set_min_growth_length)
        self.max_growth_area_widget.changed.connect(self.set_max_growth_area)
        self.min_growth_area_widget.changed.connect(self.set_min_growth_area)
        self.segmentation_method_widget.changed.connect(self.set_segmentation_method)

        self.run_widget.clicked.connect(self.run)
        self.display_widget.clicked.connect(self.display_fovs)
        self.set_display_fovs_widget.connect_callback(self.set_display_fovs)

        self.append(self.fov_widget)
        self.append(self.pxl2um_widget)
        self.append(self.phase_plane_widget)
        self.append(self.lost_cell_time_widget)
        self.append(self.new_cell_y_cutoff_widget)
        self.append(self.new_cell_region_cutoff_widget)
        self.append(self.max_growth_length_widget)
        self.append(self.min_growth_length_widget)
        self.append(self.max_growth_area_widget)
        self.append(self.min_growth_area_widget)
        self.append(self.segmentation_method_widget)
        self.append(self.run_widget)
        self.append(self.display_widget)
        self.append(self.set_display_fovs_widget)

        self.set_fovs(self.valid_fovs)
        self.set_pxl2um()
        self.set_phase_plane()
        self.set_lost_cell_time()
        self.set_new_cell_y_cutoff()
        self.set_new_cell_region_cutoff()
        self.set_max_growth_length()
        self.set_min_growth_length()
        self.set_max_growth_area()
        self.set_min_growth_area()
        self.set_segmentation_method()
        self.set_display_fovs(self.valid_fovs)

    def set_params(self):
        self.params = dict()
        self.params["experiment_name"] = self.experiment_name
        self.params["FOV"] = self.fovs
        self.params["phase_plane"] = self.phase_plane
        self.params["pxl2um"] = self.pxl2um
        self.params["output"] = "TIFF"
        self.params["num_analyzers"] = multiprocessing.cpu_count()
        self.params["track"] = dict()
        self.params["track"]["lost_cell_time"] = self.lost_cell_time
        self.params["track"]["new_cell_y_cutoff"] = self.new_cell_y_cutoff
        self.params["track"]["new_cell_region_cutoff"] = self.new_cell_region_cutoff
        self.params["track"]["max_growth_length"] = self.max_growth_length
        self.params["track"]["min_growth_length"] = self.min_growth_length
        self.params["track"]["max_growth_area"] = self.max_growth_area
        self.params["track"]["min_growth_area"] = self.min_growth_area
        if self.segmentation_method == "Otsu":
            self.params["track"]["seg_img"] = "seg_otsu"
        elif self.segmentation_method == "U-net":
            self.params["track"]["seg_img"] = "seg_unet"

        # useful folder shorthands for opening files
        self.params["TIFF_dir"] = self.TIFF_folder
        self.params["ana_dir"] = self.analysis_folder
        self.params["hdf5_dir"] = self.params["ana_dir"] / "hdf5"
        self.params["chnl_dir"] = self.params["ana_dir"] / "channels"
        self.params["empty_dir"] = self.params["ana_dir"] / "empties"
        self.params["sub_dir"] = self.params["ana_dir"] / "subtracted"
        self.params["seg_dir"] = self.params["ana_dir"] / "segmented"
        self.params["cell_dir"] = self.params["ana_dir"] / "cell_data"
        self.params["track_dir"] = self.params["ana_dir"] / "tracking"

    def run(self):
        """Overriding method. Performs Mother Machine Analysis"""
        self.set_params()

        self.viewer.window._status_bar._toggle_activity_dock(True)
        viewer = napari.current_viewer()
        viewer.layers.clear()
        # need this to avoid vispy bug for some reason
        # related to https://github.com/napari/napari/issues/2584
        viewer.add_image(np.zeros((1, 1)))
        viewer.layers.clear()
        Track_Cells(self.params)

    def display_fovs(self):
        viewer = napari.current_viewer()
        viewer.layers.clear()

        self.set_params()
        specs = load_specs(self.params["ana_dir"])

        for fov_id in self.fovs_to_display:
            ana_peak_ids = []  # channels to be analyzed
            for peak_id, spec in six.iteritems(specs[int(fov_id)]):
                if spec == 1:  # 1 means analyze
                    ana_peak_ids.append(peak_id)
            ana_peak_ids = sorted(ana_peak_ids)  # sort for repeatability

            for peak_id in ana_peak_ids:
                try:
                    img_stack = load_lineage_image(self.params, fov_id, peak_id)
                    lin_filename = (
                        f'{self.params["experiment_name"]}_{fov_id}_{peak_id}.tif'
                    )

                    viewer.grid.enabled = True
                    viewer.grid.shape = (-1, 1)

                    viewer.add_image(img_stack, name=lin_filename)
                except:
                    warning(
                        f"No lineage found for FOV {fov_id}, peak {peak_id}. Run lineage construction first!"
                    )

    def set_fovs(self, fovs):
        self.fovs = fovs

    def set_pxl2um(self):
        self.pxl2um = self.pxl2um_widget.value

    def set_phase_plane(self):
        self.phase_plane = self.phase_plane_widget.value

    def set_lost_cell_time(self):
        self.lost_cell_time = self.lost_cell_time_widget.value

    def set_new_cell_y_cutoff(self):
        self.new_cell_y_cutoff = self.new_cell_y_cutoff_widget.value

    def set_new_cell_region_cutoff(self):
        self.new_cell_region_cutoff = self.new_cell_region_cutoff_widget.value

    def set_max_growth_length(self):
        self.max_growth_length = self.max_growth_length_widget.value

    def set_max_growth_length(self):
        self.max_growth_length = self.max_growth_length_widget.value

    def set_min_growth_length(self):
        self.min_growth_length = self.min_growth_length_widget.value

    def set_max_growth_area(self):
        self.max_growth_area = self.max_growth_area_widget.value

    def set_min_growth_area(self):
        self.min_growth_area = self.min_growth_area_widget.value

    def set_segmentation_method(self):
        self.segmentation_method = self.segmentation_method_widget.value

    def set_display_fovs(self, fovs):
        self.fovs_to_display = fovs
