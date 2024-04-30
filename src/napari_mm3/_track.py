import multiprocessing
from multiprocessing import Pool
from pathlib import Path
import napari
import yaml
import six
import pickle
import os
from typing import Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from skimage import io
from skimage.measure import regionprops
from napari.utils import progress
from magicgui.widgets import FloatSpinBox, SpinBox, ComboBox, PushButton

from ._deriving_widgets import (
    MM3Container,
    PlanePicker,
    FOVChooser,
    load_specs,
    information,
    SegmentationMode,
    load_seg_stack,
    load_unmodified_stack,
    warning,
)
from .utils import (
    Cell,
    find_complete_cells,
    find_cells_of_fov_and_peak,
    write_cells_to_json,
)

# load the time table
def load_time_table(ana_dir: Path) -> dict:
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
def check_growth_by_region(params: dict, cell: Cell, region) -> bool:
    """Checks to see if it makes sense
    to grow a cell by a particular region

    Parameters
    ----------
    params: dict
        dictionary of parameters
    cell: Cell object
        Cell object currently tracked
    region: RegionProps object
        regionprops object containing area attributes

    Returns
    -------
    bool
        True if it makes sense to grow the cell
    """
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

    # check if y position of region is within the bounding box of previous region
    lower_bound = cell.bboxes[-1][0]
    upper_bound = cell.bboxes[-1][2]
    if lower_bound > region.centroid[0] or upper_bound < region.centroid[0]:
        return False

    # return true if you get this far
    return True


# see if a cell has reasonably divided
def check_division(params: dict, cell: Cell, region1, region2) -> int:
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
def create_cell_id(
    region, t: int, peak: int, fov: int, experiment_name: str = None
) -> str:
    """Make a unique cell id string for a new cell
    Parameters
    ----------
    region: regionprops object
        region to initialize cell from
    t: int
        time
    peak: int
        peak id
    fov: int
        fov id
    experiment_name: str
        experiment label

    Returns
    -------
    cell_id: str
        string for cell ID
    """
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


def update_leaf_regions(
    regions: list,
    current_leaf_positions: list[Tuple[str, int]],
    leaf_region_map: dict[str, Tuple[int, int]],
) -> dict[str, Tuple[int, int]]:
    """
    Loop through regions from current time step and match them to existing leaves

    Parameters
    ----------
    regions: list
        list of RegionProps objects
    current_leaf_positions: list[Tuple[leaf_id, position]]
        list of (leaf_id, cell centroid) for current leaves
    leaf_region_map: dict[str,Tuple[int,int]]
        dict whose keys are leaves (cell IDs) and values (region number, region location)

    Returns
    ------
    leaf_region_map: dict[str,Tuple[int,int]]
        updated leaf region map
    """
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

    return leaf_region_map


def get_two_closest_regions(region_links: list) -> list:
    """
    Retrieve two regions closest to closed end of the channel.

    Parameters
    ----------
    region_links: list
        list of all linked regions

    Returns
    -------
    closest two regions: list
        two regions closest to closed end of channel.
    """
    closest_two_regions = sorted(region_links, key=lambda x: x[1])[:2]
    # but sort by region order so top region is first
    closest_two_regions = sorted(closest_two_regions, key=lambda x: x[0])
    return closest_two_regions


def handle_discarded_regions(
    cell_leaves: list[str],
    region_links: list,
    regions,
    new_cell_y_cutoff: int,
    new_cell_region_cutoff: int,
    cells: dict[str, Cell],
    pxl2um: float,
    time_table: dict,
    t: int,
    peak_id: int,
    fov_id: int,
) -> Tuple[list, dict]:
    """
    Process third+ regions down from closed end of channel. They will either be discarded or made into new cells.

    Parameters
    ----------
    cell_leaves: list[str]
        list of current cell leaves
    region_links: list
        regions from current time step, sorted by y position
    regions: list
        list of RegionProps objects
    new_cell_y_cutoff: int
        y position cutoff for new cells
    new_cell_region_cutoff: int
        region label cutoff for new cells
    cells: dict[str, Cell]
        dictionary of Cell objects
    pxl2um: float
        conversion factor from pixels to microns
    time_table: dict
        dict of time points
    t: int
        current time step
    peak_id: int
        current channel (trap) id
    fov_id: int
        current fov

    Returns
    -------
    cell_leaves: list[str]
        updated list of leaves by cell id
    cells: dict[str, Cell]
        updated dict of Cell objects
    """
    discarded_regions = sorted(region_links, key=lambda x: x[1])[2:]
    for discarded_region in discarded_regions:
        region = regions[discarded_region[0]]
        if (
            region.centroid[0] < new_cell_y_cutoff
            and region.label <= new_cell_region_cutoff
        ):
            cell_id = create_cell_id(region, t, peak_id, fov_id)
            cells[cell_id] = Cell(
                pxl2um,
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
    return cell_leaves, cells


def divide_cell(
    region1,
    region2,
    t: int,
    peak_id: int,
    fov_id: int,
    cells: dict[str, Cell],
    params: dict,
    time_table: dict,
    leaf_id: str,
) -> Tuple[str, str, dict]:
    """
    Create two new cells and divide the mother

    Parameters
    ----------
    region1: RegionProperties object
        first region
    region2: RegionProperties object
        second region
    t: int
        current time step
    peak_id: int
        current peak (trap) id
    fov_id: int
        current FOV id
    cells: dict[str, Cell]
        dictionary of cell objects
    params: dict
        dictionary of parameters
    time_table: dict
        dictionary of time points
    leaf_id: str
        cell id of current leaf

    Returns
    -------
    daughter1_id: str
        cell id of 1st daughter
    daughter2_id: str
        cell id of 2nd daughter
    cells: dict[str, Cell]
        updated dictionary of Cell objects
    """

    daughter1_id = create_cell_id(region1, t, peak_id, fov_id)
    daughter2_id = create_cell_id(region2, t, peak_id, fov_id)
    cells[daughter1_id] = Cell(
        params["pxl2um"],
        time_table,
        daughter1_id,
        region1,
        t,
        parent_id=leaf_id,
    )
    cells[daughter2_id] = Cell(
        params["pxl2um"],
        time_table,
        daughter2_id,
        region2,
        t,
        parent_id=leaf_id,
    )
    cells[leaf_id].divide(cells[daughter1_id], cells[daughter2_id], t)

    return daughter1_id, daughter2_id, cells


def add_leaf_daughter(
    region, cell_leaves: list[str], id: str, y_cutoff: int, region_cutoff: int
) -> list:
    """
    Add new leaf to tree if it clears thresholds

    Parameters
    ----------
    region: RegionProps object
        candidate region
    cell_leaves: list[str]
        list of cell leaves
    id: str
        candidate cell id
    y_cutoff: int
        max distance from closed end of channel to allow new cells
    region_cutoff: int
        max region (labeled ascending from closed end of channel)

    Returns
    -------
    cell_leaves: list[str]
        updated list of cell leaves
    """
    # add the daughter ids to list of current leaves if they pass cutoffs
    if region.centroid[0] < y_cutoff and region.label <= region_cutoff:
        cell_leaves.append(id)

    return cell_leaves


def add_leaf_orphan(
    region,
    cell_leaves: list[str],
    cells: dict[str, Cell],
    y_cutoff: int,
    region_cutoff: int,
    t: int,
    peak_id: int,
    fov_id: int,
    pxl2um: float,
    time_table: dict,
) -> Tuple[list, dict]:
    """
    Add new leaf if it clears thresholds.

    Parameters
    ----------
    region: regionprops object
        candidation region for new cell
    cell_leaves: list[str]
        list of current leaves
    cells: dict[str, Cell]
        dict of Cell objects
    y_cutoff: int
        max distance from closed end of channel to allow new cells
    region_cutoff: int
        max region (labeled ascending from closed end of channel)
    t: int
        time
    peak: int
        peak id
    fov: int
        fov id
    pxl2um: float
        pixel to micron conversion
    time_table: dict
        dictionary of time points

    Returns
    -------
    cell_leaves: list[str]
        updated leaves
    cells: dict[str, Cell]
        updated dict of Cell objects

    """
    if region.centroid[0] < y_cutoff and region.label <= region_cutoff:
        cell_id = create_cell_id(region, t, peak_id, fov_id)
        cells[cell_id] = Cell(
            pxl2um,
            time_table,
            cell_id,
            region,
            t,
            parent_id=None,
        )
        cell_leaves.append(cell_id)  # add to leaves
    return cell_leaves, cells


def handle_two_regions(
    region1,
    region2,
    cells: dict[str, Cell],
    cell_leaves: list[str],
    params: dict,
    pxl2um: float,
    leaf_id: int,
    t: int,
    peak_id: int,
    fov_id: int,
    time_table: dict,
    y_cutoff: int,
    region_cutoff: int,
):
    """
    Classify the two regions as either a divided cell (two daughters), or one growing cell and one trash.

    Parameters
    ----------
    region1: RegionProps object
        first region
    region2: RegionProps object
        second region
    cells: dict
        dictionary of Cell objects
    params: dict
        parameter dictionary
    leaf_id: int
        id of current tracked cell
    t: int
        time step
    peak_id: int
        peak (trap) number
    fov_id: int
        current fov
    time_table: dict
        dictionary of time points
    y_cutoff: int
        y position threshold for new cells
    region_cutoff: int
        region label cutoff for new cells

    Returns
    -------
    cell_leaves: list[str]
        list of cell leaves
    cells: dict
        updated dicitonary of"""

    # check_division returns 3 if cell divided,
    # 1 if first region is just the cell growing and the second is trash
    # 2 if the second region is the cell, and the first is trash
    # or 0 if it cannot be determined.
    check_division_result = check_division(params, cells[leaf_id], region1, region2)

    if check_division_result == 3:
        daughter1_id, daughter2_id, cells = divide_cell(
            region1, region2, t, peak_id, fov_id, cells, params, time_table, leaf_id
        )
        # remove mother from current leaves
        cell_leaves.remove(leaf_id)

        # add the daughter ids to list of current leaves if they pass cutoffs
        cell_leaves = add_leaf_daughter(
            region1, cell_leaves, daughter1_id, y_cutoff, region_cutoff
        )

        cell_leaves = add_leaf_daughter(
            region2, cell_leaves, daughter2_id, y_cutoff, region_cutoff
        )

    # 1 means that daughter 1 is just a continuation of the mother
    # The other region should be a leaf it passes the requirements
    elif check_division_result == 1:
        cells[leaf_id].grow(time_table, region1, t)

        cell_leaves, cells = add_leaf_orphan(
            region2,
            cell_leaves,
            cells,
            y_cutoff,
            region_cutoff,
            t,
            peak_id,
            fov_id,
            pxl2um,
            time_table,
        )

    # ditto for 2
    elif check_division_result == 2:
        cells[leaf_id].grow(time_table, region2, t)

        cell_leaves, cells = add_leaf_orphan(
            region1,
            cell_leaves,
            cells,
            y_cutoff,
            region_cutoff,
            t,
            peak_id,
            fov_id,
            pxl2um,
            time_table,
        )

    return cell_leaves, cells


def prune_leaves(
    cell_leaves: list[str], cells: dict[str, Cell], lost_cell_time: int, t: int
) -> Tuple[list, dict]:
    """
    Remove leaves for cells that have been lost for more than lost_cell_time

    Parameters
    ----------
    cell_leaves: list[str]
        list of current cell leaves
    cells: dict[str, Cell]
        dictionary of all Cell objects
    lost_cell_time: int
        number of time steps after which to drop lost cells
    t: int
        current time step

    Returns
    -------
    cell_leaves: list[str]
        updated list of cell leaves
    cells: dict[str,Cell]
        updated dictionary of Cell objects
    """
    for leaf_id in cell_leaves:
        if t - cells[leaf_id].times[-1] > lost_cell_time:
            cell_leaves.remove(leaf_id)
    return cell_leaves, cells


def update_region_links(
    cell_leaves: list[str],
    cells: dict[str, Cell],
    leaf_region_map: dict[str, Tuple[int, int]],
    regions: list,
    params: dict,
    pxl2um: float,
    time_table: dict,
    t: int,
    peak_id: int,
    fov_id: int,
    y_cutoff: int,
    region_cutoff: int,
):
    """
    Loop over current leaves and connect them to descendants

    Parameters
    ----------
    cell_leaves: list[str]
        currently tracked cell_ids
    cells: dict[str, Cell]
        dictionary of Cell objects
    leaf_region_map: dict[str,Tuple[int,int]]
        dictionary with keys = cell id, values = (region number, region centroid)
    regions: list
        list of RegionProps objects
    params: dict
        dictionary of parameters
    pxl2um: float
        pixel to uM conversion factor
    time_table: dict
        dictionary of time points
    t: int
        current time step
    peak_id: int
        current peak (trap) id
    fov_id: int
        current fov id

    Returns
    -------
    cell_leaves: list[str]
        list of current leaves labeled by cell id
    cells: dict[str, Cell]
        updated dictionary of Cell objects
    """
    ### iterate over the leaves, looking to see what regions connect to them.
    for leaf_id, region_links in six.iteritems(leaf_region_map):
        # if there is just one suggested descendant,
        # see if it checks out and append the data
        if len(region_links) == 1:
            region = regions[region_links[0][0]]  # grab the region from the list

            # check if the pairing makes sense based on size and position
            # this function returns true if things are okay
            if check_growth_by_region(params, cells[leaf_id], region):
                # grow the cell by the region in this case
                cells[leaf_id].grow(time_table, region, t)

        # there may be two daughters, or maybe there is just one child and a new cell
        elif len(region_links) == 2:
            # grab these two daughters
            region1 = regions[region_links[0][0]]
            region2 = regions[region_links[1][0]]
            cell_leaves, cells = handle_two_regions(
                region1,
                region2,
                cells,
                cell_leaves,
                params,
                pxl2um,
                leaf_id,
                t,
                peak_id,
                fov_id,
                time_table,
                y_cutoff,
                region_cutoff,
            )

    return cell_leaves, cells


def make_leaf_region_map(
    regions: list,
    pxl2um: float,
    params: dict,
    time_table: dict,
    cell_leaves: list[str],
    cells: dict[str, Cell],
    new_cell_y_cutoff: int,
    new_cell_region_cutoff: int,
    t: int,
    peak_id: int,
    fov_id: int,
) -> Tuple[list, dict[str, Cell]]:
    """
    Map regions in current time point onto previously tracked cells

    Parameters
    ----------
    regions: list
        regions from current time point
    params: dict
        dictionary of parameters
    time_table: dict
        dictionary of time points

    Returns
    -------
    cell_leaves: list[str]
        list of tree leaves
    cells: dict
        dictionary of cell objects
    """
    ### create mapping between regions and leaves
    leaf_region_map = {}
    leaf_region_map = {leaf_id: [] for leaf_id in cell_leaves}

    # get the last y position of current leaves and create tuple with the id
    current_leaf_positions = [
        (leaf_id, cells[leaf_id].centroids[-1][0]) for leaf_id in cell_leaves
    ]

    leaf_region_map = update_leaf_regions(
        regions, current_leaf_positions, leaf_region_map
    )

    # go through the current leaf regions.
    # limit by the closest two current regions if there are three regions to the leaf
    for leaf_id, region_links in six.iteritems(leaf_region_map):
        if len(region_links) > 2:

            leaf_region_map[leaf_id] = get_two_closest_regions(region_links)

            # for the discarded regions, put them as new leaves
            # if they are near the closed end of the channel
            cell_leaves, cells = handle_discarded_regions(
                cell_leaves,
                region_links,
                regions,
                new_cell_y_cutoff,
                new_cell_region_cutoff,
                cells,
                pxl2um,
                time_table,
                t,
                peak_id,
                fov_id,
            )

    cell_leaves, cells = update_region_links(
        cell_leaves,
        cells,
        leaf_region_map,
        regions,
        params,
        pxl2um,
        time_table,
        t,
        peak_id,
        fov_id,
        new_cell_y_cutoff,
        new_cell_region_cutoff,
    )

    return cell_leaves, cells


# Creates lineage for a single channel
def make_lineage_chnl_stack(params: dict, fov_and_peak_id: tuple) -> dict:
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
    cells : dict
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

    pxl2um = params["pxl2um"]

    # get the specific ids from the tuple
    fov_id, peak_id = fov_and_peak_id

    # TODO: Get this to be passed in?
    time_table = load_time_table(params["ana_dir"])
    # start time is the first time point for this series of TIFFs.
    start_time_index = min(time_table[fov_id].keys())

    information("Creating lineage for FOV %d, channel %d." % (fov_id, peak_id))

    seg_mode = (
        SegmentationMode.UNET
        if params["track"]["seg_img"] == "seg_unet"
        else SegmentationMode.OTSU
    )
    image_data_seg = load_seg_stack(
        ana_dir=params["ana_dir"],
        experiment_name=params["experiment_name"],
        fov_id=fov_id,
        peak_id=peak_id,
        seg_mode=seg_mode,
    )

    # Calculate all data for all time points.
    # this list will be length of the number of time points
    regions_by_time = [
        regionprops(label_image=timepoint) for timepoint in image_data_seg
    ]  # removed coordinates='xy'

    # Set up data structures.
    cells = {}  # Dict that holds all the cell objects, divided and undivided
    cell_leaves = []  # cell ids of the current leaves of the growing lineage tree

    # go through regions by timepoint and build lineages
    # timepoints start with the index of the first image
    for t, regions in enumerate(regions_by_time, start=start_time_index):
        # if there are cell leaves who are still waiting to be linked, but
        # too much time has passed, remove them.
        cell_leaves, cells = prune_leaves(cell_leaves, cells, lost_cell_time, t)

        # make all the regions leaves if there are no current leaves
        if not cell_leaves:
            for region in regions:
                cell_leaves, cells = add_leaf_orphan(
                    region,
                    cell_leaves,
                    cells,
                    new_cell_y_cutoff,
                    new_cell_region_cutoff,
                    t,
                    peak_id,
                    fov_id,
                    pxl2um,
                    time_table,
                )

        # Determine if the regions are children of current leaves
        else:
            cell_leaves, cells = make_leaf_region_map(
                regions,
                pxl2um,
                params,
                time_table,
                cell_leaves,
                cells,
                new_cell_y_cutoff,
                new_cell_region_cutoff,
                t,
                peak_id,
                fov_id,
            )

    ## plot kymograph with lineage overlay & save it out
    make_lineage_plot(params, fov_id, peak_id, cells, start_time_index)

    # return the dictionary with all the cells
    return cells


# finds lineages for all peaks in a fov
def make_lineages_fov(params: dict, fov_id: int, specs: dict) -> dict:
    """
    For a given fov, create the lineages from the segmented images.

    Parameters
    ---------
    params: dict
        dictionary of parameters
    fov_id: int
        fov to analyze
    specs: dict
        dictionary of fov and peak ids and their classifications 

    Returns
    -------
    cells: dict
        dictionary of Cell objects

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
    cells = {}  # create dictionary to hold all information
    for cell_dict in lineages:  # for all the other dictionaries in the list
        cells.update(cell_dict)  # updates cells with the entries in cell_dict

    return cells


def make_lineage_plot(
    params: dict,
    fov_id: int,
    peak_id: int,
    cells: dict[str, Cell],
    start_time_index: int,
):
    """Produces a lineage image for the first valid FOV containing cells

    Parameters
    ----------
    params: dict
        parameters dictionary
    fov_id: int
        current FOV
    peak_id: int
        current peak (trap)
    cells: dict[str, Cell]
        dict of Cell objects
    start_time_index: int
        time offset from time_table

    Returns
    -------
    None
    """
    # plotting lineage trees for complete cells

    lin_dir = params["ana_dir"] / "lineages"
    if not os.path.exists(lin_dir):
        os.makedirs(lin_dir)

    fig, ax = plot_lineage_images(
        params,
        cells,
        fov_id,
        peak_id,
        bgcolor=params["phase_plane"],
        t_adj=start_time_index,
    )
    lin_filename = f'{params["experiment_name"]}_{fov_id}_{peak_id}.tif'
    lin_filepath = lin_dir / lin_filename
    try:
        fig.savefig(lin_filepath, dpi=75)
    # sometimes image size is too large for matplotlib renderer
    except ValueError:
        warning("Image size may be too large for matplotlib")
    plt.close(fig)


def load_lineage_image(params: dict, fov_id: int, peak_id: int):
    """Loads a lineage image for the given FOV and peak

    Parameters
    ----------
    params : dict
        Dictionary of parameters
    fov_id : int
        FOV ID
    peak_id : int
        Peak ID

    Returns
    -------
    img_stack : np.ndarray
        Image stack of lineage images
    """
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


def connect_centroids(
    fig: plt.Figure,
    ax: plt.Axes,
    cell: Cell,
    n: int,
    t: int,
    t_adj: int,
    x: int,
    y: int,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw lines linking cell centroids in time

    Parameters
    ----------
    fig: plt.Figure
        figure object
    ax: plt.Axes
        axis object
    cell: Cell
        current cell object
    n: int
        cell time index relative to birth
    t: int
        time step (relative to experiment start)
    x: int
        centroid x position
    y: int
        centroid y position

    Returns:
    fig: plt.Figure
        updated figure
    ax: plt.Axes
        updated axis
    """
    # coordinates of the next centroid
    x_next = cell.centroids[n + 1][1]
    y_next = cell.centroids[n + 1][0]
    t_next = cell.times[n + 1] - t_adj  # adjust for special indexing

    transFigure = fig.transFigure.inverted()

    # get coordinates for the whole figure
    coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
    coord2 = transFigure.transform(ax[t_next].transData.transform([x_next, y_next]))

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
    return fig, ax


def connect_mother_daughter(
    fig: plt.Figure,
    ax: plt.Axes,
    cells: dict[str, Cell],
    cell_id: str,
    t: int,
    t_adj: int,
    x: int,
    y: int,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw lines linking mother to its daughters

    Parameters
    ----------
    fig: plt.Figure
        figure object
    ax: plt.Axes
        axis object
    cells: dict[str, Cell]
        dictionary of Cell objects
    cell_id: str
        current mother cell id
    t: int
        time step (relative to experiment start)
    t_adj: int
        time offset from time_table
    x: int
        centroid x position
    y: int
        centroid y position

    Returns
    -------
    fig: plt.Figure
        updated figure
    ax: plt.Axes
        updated axis
    """
    # daughter ids
    d1_id = cells[cell_id].daughters[0]
    d2_id = cells[cell_id].daughters[1]

    # both daughters should have been born at the same time.
    t_next = cells[d1_id].times[0] - t_adj

    # coordinates of the two daughters
    x_d1 = cells[d1_id].centroids[0][1]
    y_d1 = cells[d1_id].centroids[0][0]
    x_d2 = cells[d2_id].centroids[0][1]
    y_d2 = cells[d2_id].centroids[0][0]

    transFigure = fig.transFigure.inverted()

    # get coordinates for the whole figure
    coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
    coordd1 = transFigure.transform(ax[t_next].transData.transform([x_d1, y_d1]))
    coordd2 = transFigure.transform(ax[t_next].transData.transform([x_d2, y_d2]))

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
    return fig, ax


def plot_tracks(
    cells: dict[str, Cell], t_adj: int, fig: plt.Figure, ax: plt.Axes
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw lines linking tracked cells across time

    Parameters
    ----------
    cells: dict[str, Cell]
        dictionary of Cell objects
    t_adj: int
        time offset
    fig: plt.Figure
        figure object
    ax: plt.Axes
        axis object

    Returns
    -------
    fig: plt.Figure
        updated figure object
    ax: plt.Axes
        updated axis object
    """
    for cell_id in cells:
        for n, t in enumerate(cells[cell_id].times):
            t -= t_adj  # adjust for special indexing

            x = cells[cell_id].centroids[n][1]
            y = cells[cell_id].centroids[n][0]

            # add a circle at the centroid for every point in this cell's life
            circle = mpatches.Circle(
                xy=(x, y), radius=2, color="white", lw=0, alpha=0.5
            )
            ax[t].add_patch(circle)

            # draw connecting lines between the centroids of cells in same lineage
            try:
                if n < len(cells[cell_id].times) - 1:
                    fig, ax = connect_centroids(
                        fig, ax, cells[cell_id], n, t, t_adj, x, y
                    )
            except:
                pass

            # draw connecting between mother and daughters
            try:
                if n == len(cells[cell_id].times) - 1 and cells[cell_id].daughters:
                    fig, ax = connect_mother_daughter(
                        fig, ax, cells, cell_id, t, t_adj, x, y
                    )
            except:
                pass
    return fig, ax


def plot_regions(seg_data: np.ndarray, regions: list, ax: plt.Axes, cmap: str, vmin: int = 0.5, vmax: int = 100) -> plt.Axes:
    """
    Plot segmented cells from one peak & time step

    Parameters
    ----------
    seg_data: np.ndarray
        segmentation labels
    regions: list
        list of regionprops objects
    ax: plt.Axes
        current axis

    Returns
    -------
    ax: plt.Axes
        updated axis
    """
    # make a new version of the segmented image where the
    # regions are relabeled by their y centroid position.
    # scale it so it falls within 100.
    seg_relabeled = seg_data.copy().astype(float)
    for region in regions:
        rescaled_color_index = region.centroid[0] / seg_data.shape[0] * vmax
        seg_relabeled[seg_relabeled == region.label] = (
            int(rescaled_color_index) - 0.1
        )  # subtract small value to make it so there is not overlabeling
    ax.imshow(seg_relabeled, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
    return ax


def plot_cells(
    image_data_bg: np.ndarray, image_data_seg: np.ndarray, fgcolor: bool, t_adj: int
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot phase imaging data overlaid with segmented cells

    Parameters
    ----------
    image_data_bg: np.ndarray
        phase contrast images
    image_data_seg: np.ndarray
        segmented images
    fgcolor: bool
        whether to plot segmented images
    t_adj: int
        time offset from time_table

    Returns
    -------
    fig: plt.Figure
        matplotlib figure
    ax: plt.Axes
        matplotlib axis
    """
    n_imgs = image_data_bg.shape[0]
    image_indices = range(n_imgs)

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

    # change settings for each axis
    ax = axes.flat  # same as axes.ravel()
    for a in ax:
        a.set_axis_off()
        a.set_aspect("equal")
        ttl = a.title
        ttl.set_position([0.5, 0.05])

    for i in image_indices:
        ax[i].imshow(image_data_bg[i], cmap=plt.cm.gray, aspect="equal")

        if fgcolor:
            # calculate the regions across the segmented images
            regions_by_time = [regionprops(timepoint) for timepoint in image_data_seg]
            # Color map for good label colors
            cmap = mpl.colors.ListedColormap(sns.husl_palette(n_colors=100, h=0.5, l=0.8, s=1))
            cmap.set_under(color="black")
            ax[i] = plot_regions(image_data_seg[i], regions_by_time[i], ax[i], cmap)

        ax[i].set_title(str(i + t_adj), color="white")

    return fig, ax


def plot_lineage_images(
    params,
    cells,
    fov_id,
    peak_id,
    bgcolor="c1",
    fgcolor="seg",
    t_adj=1,
):
    """
    Plot linages over images across time points for one FOV/peak.
    Parameters
    ----------
    bgcolor : Designation of background to use. Subtracted images look best if you have them.
    fgcolor : Designation of foreground to use. This should be a segmented image.
    t_adj : int
        adjust time indexing for differences between t index of image and image number
    """

    # filter cells
    cells = find_cells_of_fov_and_peak(cells, fov_id, peak_id)

    # load subtracted and segmented data
    image_data_bg = load_unmodified_stack(
        params["ana_dir"], params["experiment_name"], fov_id, peak_id, postfix=bgcolor
    )

    if fgcolor:
        # image_data_seg = load_stack_params(params, fov_id, peak_id, postfix=fgcolor)
        seg_mode = (
            SegmentationMode.OTSU
            if params["track"]["seg_img"] == "seg_otsu"
            else SegmentationMode.UNET
        )
        image_data_seg = load_seg_stack(
            ana_dir=params["ana_dir"],
            experiment_name=params["experiment_name"],
            fov_id=fov_id,
            peak_id=peak_id,
            seg_mode=seg_mode,
        )

    fig, ax = plot_cells(image_data_bg, image_data_seg, fgcolor, t_adj)

    # Annotate each cell with information
    fig, ax = plot_tracks(cells, t_adj, fig, ax)

    return fig, ax


def track_cells(params):
    """Track cells in a set of FOVs.
    Parameters
    ----------
    params : dict
        Dictionary containing all parameters for the analysis.
    """
    # Load the project parameters file
    information("Loading experiment parameters.")
    p = params

    user_spec_fovs = params["FOV"]

    information("Using {} threads for multiprocessing.".format(p["num_analyzers"]))

    information("Using {} images for tracking.".format(p["track"]["seg_img"]))

    # create segmenteation and cell data folder if they don't exist
    if not os.path.exists(p["seg_dir"]):
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
    cells = {}

    # do lineage creation per fov, so pooling can be done by peak
    for fov_id in fov_id_list:
        # update will add the output from make_lineages_function, which is a
        # dict of Cell entries, into cells
        cells.update(make_lineages_fov(params, fov_id, specs))
    information("Finished lineage creation.")

    ### Now prune and save the data.
    information("Curating and saving cell data.")

    # this returns only cells with a parent and daughters
    complete_cells = find_complete_cells(cells)

    ### save the cell data
    # All cell data (includes incomplete cells)
    with open(p["cell_dir"] / "all_cells.pkl", "wb") as cell_file:
        pickle.dump(cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    ### save to .json
    write_cells_to_json(cells, p["cell_dir"] / "all_cells.json")

    # Just the complete cells, those with mother and daughter
    # This is a dictionary of cell objects.
    with open(p["cell_dir"] / "complete_cells.pkl", "wb") as cell_file:
        pickle.dump(complete_cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    ### save to .json
    write_cells_to_json(complete_cells, p["cell_dir"] / "complete_cells.json")

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
            step=1,
            tooltip="only regions with labels less than or equal to this value will "
            "be considered to start potential new cells. Does not apply to daughters",
        )
        self.max_growth_length_widget = FloatSpinBox(
            label="max growth length (ratio)",
            value=1.3,
            min=0,
            max=20,
            step=0.1,
            tooltip="Maximum increase in length allowed when linked new region to "
            "existing potential cell. Unit is ratio.",
        )
        self.min_growth_length_widget = FloatSpinBox(
            label="min growth length (ratio)",
            value=0.8,
            min=0,
            max=20,
            step=0.1,
            tooltip="Minimum change in length allowed when linked new region to "
            "existing potential cell. Unit is ratio.",
        )
        self.max_growth_area_widget = FloatSpinBox(
            label="max growth area (ratio)",
            value=1.3,
            min=0,
            max=20,
            step=0.1,
            tooltip="Maximum change in area allowed when linked new region to "
            "existing potential cell. Unit is ratio.",
        )
        self.min_growth_area_widget = FloatSpinBox(
            label="min growth area (ratio)",
            value=0.8,
            min=0,
            max=20,
            step=0.1,
            tooltip="Minimum change in area allowed when linked new region to existing potential cell. Unit is ratio.",
        )
        self.segmentation_method_widget = ComboBox(
            label="segmentation method", choices=["Otsu", "U-net"]
        )

        self.run_widget.text = "Construct lineages"
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
        track_cells(self.params)

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
