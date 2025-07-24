import argparse
import multiprocessing
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import napari
import numpy as np
import seaborn as sns
import six
from magicgui.widgets import ComboBox, FloatSpinBox, PushButton, SpinBox
from napari.utils import progress
from skimage import io
from skimage.measure import regionprops

from ._deriving_widgets import (
    FOVChooser,
    MM3Container,
    PlanePicker,
    get_valid_fovs_folder,
    get_valid_times,
    information,
    load_specs,
    load_tiff,
    load_time_table,
    range_string_to_indices,
    warning,
)
from .utils import (
    TIFF_FILE_FORMAT_PEAK,
    Cell,
    find_complete_cells,
    write_cells_to_json,
)


class CellTracker:
    """
    Class to track cells through time.
    """

    def __init__(
        self,
        fmt_cell_id: str,
        y_cutoff: int,
        region_cutoff: int,
        lost_cell_time: int,
        time_table: dict,
        pxl2um: float,
        max_growth_length: float,
        min_growth_length: float,
        max_growth_area: float,
        min_growth_area: float,
    ):
        self.fmt_cell_id = fmt_cell_id
        self.cell_leaves: list[str] = []
        self.cells: dict[str, Cell] = {}
        self.y_cutoff = y_cutoff
        self.region_cutoff = region_cutoff
        self.lost_cell_time = lost_cell_time
        self.time_table = time_table
        self.pxl2um = pxl2um
        self.max_growth_length = max_growth_length
        self.min_growth_length = min_growth_length
        self.max_growth_area = max_growth_area
        self.min_growth_area = min_growth_area

    def add_cell(self, region, t, parent_id=None) -> str:
        """Adds a cell to the graph and returns its id."""
        cell_id = self.fmt_cell_id.format(region.label, t)
        self.cells[cell_id] = Cell(
            self.pxl2um,
            self.time_table,
            cell_id,
            region,
            t,
            parent_id=parent_id,
        )
        return cell_id

    def prune_leaves(self, t: int):
        """
        Remove leaves for cells that have been lost for more than lost_cell_time
        """
        for leaf_id in self.cell_leaves:
            if t - self.cells[leaf_id].times[-1] > self.lost_cell_time:
                self.cell_leaves.remove(leaf_id)

    def add_leaf_orphan(
        self,
        region,
        t: int,
    ):
        """
        Add new leaf if it clears thresholds.
        """
        if region.centroid[0] < self.y_cutoff and region.label <= self.region_cutoff:
            cell_id = self.add_cell(region, t, parent_id=None)
            self.cell_leaves.append(cell_id)  # add to leaves

    def divide_cell(
        self,
        region1,
        region2,
        t: int,
        leaf_id: str,
    ) -> Tuple[str, str, dict]:
        """
        Create two new cells and divide the mother
        """
        # This logic's a bit weird, but it's preserved from original mm3.
        # If we're not adding this to leaves, why add it to the cells?
        daughter1_id = self.add_cell(region1, t, parent_id=leaf_id)
        daughter2_id = self.add_cell(region2, t, parent_id=leaf_id)
        self.cells[leaf_id].divide(
            self.cells[daughter1_id], self.cells[daughter2_id], t
        )

        self.cell_leaves.remove(leaf_id)
        if region1.centroid[0] < self.y_cutoff and region1.label <= self.region_cutoff:
            self.cell_leaves.append(daughter1_id)
        if region2.centroid[0] < self.y_cutoff and region2.label <= self.region_cutoff:
            self.cell_leaves.append(daughter2_id)

        return daughter1_id, daughter2_id, self.cells

    def update_region_links(
        self,
        leaf_region_map: dict[str, list[tuple[int, float]]],
        regions: list,
        t: int,
    ):
        """
        Loop over current leaves and connect them to descendants
        """
        for leaf_id, region_links in six.iteritems(leaf_region_map):
            if len(region_links) == 1:
                region = regions[region_links[0][0]]
                if self.check_growth_by_region(
                    self.cells[leaf_id],
                    region,
                ):
                    self.cells[leaf_id].grow(self.time_table, region, t)
            elif len(region_links) == 2:
                region1 = regions[region_links[0][0]]
                region2 = regions[region_links[1][0]]
                self.handle_two_regions(
                    region1,
                    region2,
                    leaf_id,
                    t,
                )

    def handle_two_regions(
        self,
        region1,
        region2,
        leaf_id: str,
        t: int,
    ):
        """
        Classify the two regions as either a divided cell (two daughters),
        or one growing cell and one trash.
        """

        if self.check_growth_by_region(self.cells[leaf_id], region1):
            self.cells[leaf_id].grow(self.time_table, region1, t)
            self.add_leaf_orphan(
                region2,
                t,
            )
        elif self.check_growth_by_region(self.cells[leaf_id], region2):
            self.cells[leaf_id].grow(self.time_table, region2, t)
            self.add_leaf_orphan(
                region1,
                t,
            )
        elif self.check_division(
            self.cells[leaf_id],
            region1,
            region2,
        ):
            daughter1_id, daughter2_id, self.cells = self.divide_cell(
                region1,
                region2,
                t,
                leaf_id,
            )

    def link_regions_to_previous_cells(
        self,
        regions: list,
        t: int,
    ):
        """
        Map regions in current time point onto previously tracked cells
        """
        leaf_region_map = {leaf_id: [] for leaf_id in self.cell_leaves}

        current_leaf_positions = [
            (leaf_id, self.cells[leaf_id].centroids[-1][0])
            for leaf_id in self.cell_leaves
        ]

        leaf_region_map = self.update_leaf_regions(
            regions, current_leaf_positions, leaf_region_map
        )

        for leaf_id, region_links in six.iteritems(leaf_region_map):
            if len(region_links) > 2:
                leaf_region_map[leaf_id] = self.get_two_closest_regions(region_links)
                self.handle_discarded_regions(
                    region_links,
                    regions,
                    t,
                )

        self.update_region_links(
            leaf_region_map,
            regions,
            t,
        )

    def handle_discarded_regions(
        self,
        region_links: list,
        regions: list,
        t: int,
    ):
        """
        Process third+ regions down from closed end of channel.
        They will either be discarded or made into new cells.
        """
        discarded_regions = sorted(region_links, key=lambda x: x[1])[2:]
        for discarded_region in discarded_regions:
            region = regions[discarded_region[0]]
            if (
                region.centroid[0] < self.y_cutoff
                and region.label <= self.region_cutoff
            ):
                cell_id = self.add_cell(region, t, parent_id=None)
                self.cell_leaves.append(cell_id)
            else:
                break

    def check_growth_by_region(
        self,
        cell: Cell,
        region,
    ) -> bool:
        """Checks to see if it makes sense
        to grow a cell by a particular region

        Returns
        -------
        bool
            True if it makes sense to grow the cell
        """
        # check if length is not too much longer
        if cell.lengths[-1] * self.max_growth_length < region.major_axis_length:
            return False

        # check if it is not too short (cell should not shrink really)
        if cell.lengths[-1] * self.min_growth_length > region.major_axis_length:
            return False

        # check if area is not too great
        if cell.areas[-1] * self.max_growth_area < region.area:
            return False

        # check if area is not too small
        if cell.areas[-1] * self.min_growth_area > region.area:
            return False

        # check if y position of region is within the bounding box of previous region
        lower_bound = cell.bboxes[-1][0]
        upper_bound = cell.bboxes[-1][2]
        if lower_bound > region.centroid[0] or upper_bound < region.centroid[0]:
            return False

        # return true if you get this far
        return True

    def check_division(
        self,
        cell: Cell,
        region1,
        region2,
    ) -> bool:
        """Checks to see if it makes sense to divide a
        cell into two new cells based on two regions.

        Return False if nothing should happend and regions ignored
        Return True if cell should divide into the regions."""

        # make sure combined size of daughters is not too big
        combined_size = region1.major_axis_length + region2.major_axis_length
        max_size = cell.lengths[-1] * self.max_growth_length
        min_size = cell.lengths[-1] * self.min_growth_length

        if max_size < combined_size:
            return False
        if min_size > combined_size:
            return False

        # centroids of regions should be in the upper and lower half of the
        # of the mother's bounding box
        # top region within top half of mother bounding box
        cell_bottom, cell_top = cell.bboxes[-1][0], cell.bboxes[-1][2]
        cell_center = cell.centroids[-1][0]
        if cell_bottom > region1.centroid[0]:
            return False
        if cell_center < region1.centroid[0]:
            return False
        # bottom region with bottom half of mother bounding box
        if cell_center > region2.centroid[0]:
            return False
        if cell_top < region2.centroid[0]:
            return False

        return True

    def update_leaf_regions(
        self,
        regions: list,
        current_leaf_positions: list[tuple[str, float]],
        leaf_region_map: dict[str, list[tuple[int, float]]],
    ) -> dict[str, list[tuple[int, float]]]:
        """
        Loop through regions from current time step and match them to existing leaves

        Parameters
        ----------
        regions: list
            list of RegionProps objects
        current_leaf_positions: list[tuple[leaf_id, position]]
            list of (leaf_id, cell centroid) for current leaves
        leaf_region_map: dict[str,list[tuple[int,float]]]
            dict whose keys are leaves (cell IDs) and list of values (region number, region location)

        Returns
        ------
        leaf_region_map: dict[str,Tuple[int,int]]
            updated leaf region map
        """
        # go through regions, they will come off in Y position order
        for r, region in enumerate(regions):
            # create tuple which is cell_id of closest leaf, distance
            current_closest = (str(""), float("inf"))

            # check this region against all positions of all current leaf regions,
            # find the closest one in y.
            for leaf in current_leaf_positions:
                # calculate distance between region and leaf
                y_dist_region_to_leaf: float = abs(region.centroid[0] - leaf[1])

                # if the distance is closer than before, update
                if y_dist_region_to_leaf < current_closest[1]:
                    current_closest = (leaf[0], y_dist_region_to_leaf)

            # update map with the closest region
            leaf_region_map[current_closest[0]].append((r, y_dist_region_to_leaf))

        return leaf_region_map

    def get_two_closest_regions(self, region_links: list) -> list:
        """
        Retrieve two regions closest to closed end of the channel.
        """
        closest_two_regions = sorted(region_links, key=lambda x: x[1])[:2]
        # but sort by region order so top region is first
        closest_two_regions = sorted(closest_two_regions, key=lambda x: x[0])
        return closest_two_regions


# Creates lineage for a single channel
def make_lineage_chnl_stack(
    ana_dir: Path,
    experiment_name: str,
    fov_and_peak_id: tuple,
    lost_cell_time: int,
    new_cell_y_cutoff: int,
    new_cell_region_cutoff: int,
    max_growth_length: float,
    min_growth_length: float,
    max_growth_area: float,
    min_growth_area: float,
    pxl2um: float,
    seg_img: str,
    phase_plane: str,
) -> dict:
    """
    Create the lineage for a set of segmented images for one channel.
    Start by making the regions in the first time points potential cells.
    Go forward in time and map regions in the timepoint to the potential
    cells in previous time points, building the life of a cell.
    Used basic checks such as the regions should overlap, and grow by a little and
    not shrink too much. If regions do not link back in time, discard them.
    If two regions map to one previous region, check if it is a sensible division event.

    Returns
    -------
    cells : dict
        A dictionary of all the cells from this lineage, divided and undivided
    """

    # get the specific ids from the tuple
    fov_id, peak_id = fov_and_peak_id

    # TODO: Get this to be passed in?
    time_table = load_time_table(ana_dir)
    # start time is the first time point for this series of TIFFs.
    start_time_index = min(time_table[fov_id].keys())

    information("Creating lineage for FOV %d, channel %d." % (fov_id, peak_id))

    img_filename = TIFF_FILE_FORMAT_PEAK % (experiment_name, fov_id, peak_id, seg_img)
    image_data_seg = load_tiff(ana_dir / "segmented" / img_filename)

    # Calculate all data for all time points.
    # this list will be length of the number of time points
    regions_by_time = [
        regionprops(label_image=timepoint) for timepoint in image_data_seg
    ]  # removed coordinates='xy'

    # TODO: move to global constant.
    cell_id_format = f"{experiment_name}f{fov_id:0=2}p{peak_id:0=4}t{{:0=4}}r{{:0=2}}"
    tracker = CellTracker(
        cell_id_format,
        new_cell_y_cutoff,
        new_cell_region_cutoff,
        lost_cell_time,
        time_table,
        pxl2um,
        max_growth_length,
        min_growth_length,
        max_growth_area,
        min_growth_area,
    )

    for t, regions in enumerate(regions_by_time, start=start_time_index):
        tracker.prune_leaves(t)
        if not tracker.cell_leaves:
            for region in regions:
                tracker.add_leaf_orphan(
                    region,
                    t,
                )
        else:
            tracker.link_regions_to_previous_cells(
                regions,
                t,
            )

    cells = tracker.cells

    ## plot kymograph with lineage overlay & save it out
    plotter = LineagePlotter(
        ana_dir,
        experiment_name,
        fov_id,
        peak_id,
        cells,
        start_time_index,
        phase_plane,
        seg_img,
    )
    plotter.make_lineage_plot()

    # return the dictionary with all the cells
    return cells


# finds lineages for all peaks in a fov
def make_lineages_fov(
    ana_dir: Path,
    experiment_name: str,
    fov_id: int,
    specs: dict,
    lost_cell_time: int,
    new_cell_y_cutoff: int,
    new_cell_region_cutoff: int,
    max_growth_length: float,
    min_growth_length: float,
    max_growth_area: float,
    min_growth_area: float,
    pxl2um: float,
    seg_img: str,
    phase_plane: str,
) -> dict:
    """
    For a given fov, create the lineages from the segmented images.

    Returns
    -------
    cells : dict
        Dictionary of Cell objects
    """
    ana_peak_ids = []  # channels to be analyzed
    for peak_id, spec in six.iteritems(specs[fov_id]):
        if spec == 1:  # 1 means analyze
            ana_peak_ids.append(peak_id)
    ana_peak_ids = sorted(ana_peak_ids)  # sort for repeatability

    information(
        "Creating lineage for FOV %d with %d channels." % (fov_id, len(ana_peak_ids))
    )

    # just break if there are no peaks to analyze
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
        lineages.append(
            make_lineage_chnl_stack(
                ana_dir,
                experiment_name,
                fov_and_peak_ids,
                lost_cell_time,
                new_cell_y_cutoff,
                new_cell_region_cutoff,
                max_growth_length,
                min_growth_length,
                max_growth_area,
                min_growth_area,
                pxl2um,
                seg_img,
                phase_plane,
            )
        )

    # combine all dictionaries into one dictionary
    cells = {}  # create dictionary to hold all information
    for cell_dict in lineages:  # for all the other dictionaries in the list
        cells.update(cell_dict)  # updates cells with the entries in cell_dict

    return cells


def load_lineage_image(ana_dir, experiment_name, fov_id, peak_id):
    """Loads a lineage image for the given FOV and peak

    Returns
    -------
    img_stack : np.ndarray
        Image stack of lineage images
    """
    lin_dir = ana_dir / "lineages"

    lin_filename = f"{experiment_name}_{fov_id}_{peak_id}.tif"
    lin_filepath = lin_dir / lin_filename

    img = io.imread(lin_filepath)
    imgs = []

    crop = min(2000, len(img[0]))
    for i in range(0, len(img[0]), int(len(img[0]) / 50) + 1):
        crop_img = img[:, i : i + crop, :]
        if len(crop_img[0]) == crop:
            imgs.append(crop_img)

    return np.stack(imgs, axis=0)


class LineagePlotter:
    def __init__(
        self,
        ana_dir: Path,
        experiment_name: str,
        fov_id: int,
        peak_id: int,
        cells: dict,
        start_time_index: int,
        phase_plane: str,
        seg_mode: str,
    ):
        self.ana_dir = ana_dir
        self.experiment_name = experiment_name
        self.fov_id = fov_id
        self.peak_id = peak_id
        self.cells = cells
        self.start_time_index = start_time_index
        self.phase_plane = phase_plane
        self.seg_mode = seg_mode

    def make_lineage_plot(self):
        """
        Produces a lineage image for the first valid FOV containing cells
        """

        lin_dir = self.ana_dir / "lineages"
        if not os.path.exists(lin_dir):
            os.makedirs(lin_dir)

        fig, ax = self.plot_lineage_images(
            t_adj=self.start_time_index,
        )
        lin_filename = f"{self.experiment_name}_{self.fov_id}_{self.peak_id}.tif"
        lin_filepath = lin_dir / lin_filename
        try:
            fig.savefig(lin_filepath, dpi=75)
        except ValueError:
            warning("Image size may be too large for matplotlib")
        plt.close(fig)

    def connect_centroids(
        self,
        fig: plt.Figure,
        ax: np.ndarray[plt.Axes],
        cell: Cell,
        n: int,
        t: int,
        t_adj: int,
        x: int,
        y: int,
    ) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
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
        x_next = cell.centroids[n + 1][1]
        y_next = cell.centroids[n + 1][0]
        t_next = cell.times[n + 1] - t_adj

        transFigure = fig.transFigure.inverted()

        coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
        coord2 = transFigure.transform(ax[t_next].transData.transform([x_next, y_next]))

        line = mpl.lines.Line2D(
            (coord1[0], coord2[0]),
            (coord1[1], coord2[1]),
            transform=fig.transFigure,
            color="white",
            lw=1,
            alpha=0.5,
        )

        fig.lines.append(line)
        return fig, ax

    def connect_mother_daughter(
        self,
        fig: plt.Figure,
        ax: np.ndarray[plt.Axes],
        cell_id: str,
        t: int,
        t_adj: int,
        x: int,
        y: int,
    ) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
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
        d1_id = self.cells[cell_id].daughters[0]
        d2_id = self.cells[cell_id].daughters[1]

        t_next = self.cells[d1_id].times[0] - t_adj

        x_d1 = self.cells[d1_id].centroids[0][1]
        y_d1 = self.cells[d1_id].centroids[0][0]
        x_d2 = self.cells[d2_id].centroids[0][1]
        y_d2 = self.cells[d2_id].centroids[0][0]

        transFigure = fig.transFigure.inverted()

        coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
        coordd1 = transFigure.transform(ax[t_next].transData.transform([x_d1, y_d1]))
        coordd2 = transFigure.transform(ax[t_next].transData.transform([x_d2, y_d2]))

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
            fig.lines.append(line)
        return fig, ax

    def plot_tracks(
        self, t_adj: int, fig: plt.Figure, ax: np.ndarray[plt.Axes]
    ) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """
        Draw lines linking tracked cells across time

        Parameters
        ----------
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
        for cell_id in self.cells:
            for n, t in enumerate(self.cells[cell_id].times):
                t -= t_adj

                x = self.cells[cell_id].centroids[n][1]
                y = self.cells[cell_id].centroids[n][0]

                circle = mpatches.Circle(
                    xy=(x, y), radius=2, color="white", lw=0, alpha=0.5
                )
                ax[t].add_patch(circle)

                try:
                    if n < len(self.cells[cell_id].times) - 1:
                        fig, ax = self.connect_centroids(
                            fig, ax, self.cells[cell_id], n, t, t_adj, x, y
                        )
                except Exception as e:
                    warning(f"Error connecting centroids: {e}")
                    pass

                try:
                    if (
                        n == len(self.cells[cell_id].times) - 1
                        and self.cells[cell_id].daughters
                    ):
                        fig, ax = self.connect_mother_daughter(
                            fig, ax, cell_id, t, t_adj, x, y
                        )
                except Exception as e:
                    warning(f"Error connecting mother-daughter: {e}")
                    pass
        return fig, ax

    def plot_regions(
        self,
        seg_data: np.ndarray,
        regions: list,
        ax: plt.Axes,
        cmap: str,
        vmin: float = 0.5,
        vmax: float = 100,
    ) -> plt.Axes:
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
        seg_relabeled = seg_data.copy().astype(float)
        for region in regions:
            rescaled_color_index = region.centroid[0] / seg_data.shape[0] * vmax
            seg_relabeled[seg_relabeled == region.label] = (
                int(rescaled_color_index) - 0.1
            )
        ax.imshow(seg_relabeled, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
        return ax

    def plot_cells(
        self,
        image_data_bg: np.ndarray,
        image_data_seg: np.ndarray,
        t_adj: int,
    ) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """
        Plot phase imaging data overlaid with segmented cells

        Parameters
        ----------
        image_data_bg: np.ndarray
            phase contrast images
        image_data_seg: np.ndarray
            segmented images
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

        figxsize = image_data_bg.shape[2] * n_imgs / 100.0
        figysize = image_data_bg.shape[1] / 100.0

        fig, axes = plt.subplots(
            ncols=n_imgs,
            nrows=1,
            figsize=(figxsize, figysize),
            facecolor="black",
            edgecolor="black",
        )
        fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

        ax = axes.flat
        for a in ax:
            a.set_axis_off()
            a.set_aspect("equal")
            ttl = a.title
            ttl.set_position([0.5, 0.05])

        for i in image_indices:
            ax[i].imshow(image_data_bg[i], cmap=plt.cm.gray, aspect="equal")

            regions_by_time = [regionprops(timepoint) for timepoint in image_data_seg]
            cmap = mpl.colors.ListedColormap(
                sns.husl_palette(n_colors=100, h=0.5, l=0.8, s=1)
            )
            cmap.set_under(color="black")
            ax[i] = self.plot_regions(
                image_data_seg[i], regions_by_time[i], ax[i], cmap
            )

            ax[i].set_title(str(i), color="white")

        return fig, ax

    def plot_lineage_images(
        self,
        t_adj: int = 1,
    ) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """
        Plot linages over images across time points for one FOV/peak.
        """
        # switch postfix to c1/c2/c3 auto??
        img_filename = TIFF_FILE_FORMAT_PEAK % (
            self.experiment_name,
            self.fov_id,
            self.peak_id,
            self.phase_plane,
        )
        image_data_bg = load_tiff(self.ana_dir / "channels" / img_filename)

        img_filename = TIFF_FILE_FORMAT_PEAK % (
            self.experiment_name,
            self.fov_id,
            self.peak_id,
            self.seg_mode,
        )
        image_data_seg = load_tiff(self.ana_dir / "segmented" / img_filename)

        fig, ax = self.plot_cells(
            image_data_bg,
            image_data_seg,
            t_adj,
        )

        fig, ax = self.plot_tracks(t_adj, fig, ax)

        return fig, ax


def track_cells(
    experiment_name: str,
    fovs: list[int],
    phase_plane: str,
    pxl2um: float,
    num_analyzers: int,
    lost_cell_time: int,
    new_cell_y_cutoff: int,
    new_cell_region_cutoff: int,
    max_growth_length: float,  # Max allowed growth length ratio.
    min_growth_length: float,  # Minimum allowed growth length ratio.
    max_growth_area: float,  # Max allowed growth area ratio.
    min_growth_area: float,  # Min allowed growth area ratio.
    seg_img: str,
    ana_dir: Path,
    seg_dir: Path,
    cell_dir: Path,
):
    """Track cells in a set of FOVs"""
    # Load the project parameters file
    information("Loading experiment parameters.")
    information("Using {} threads for multiprocessing.".format(num_analyzers))
    information("Using {} images for tracking.".format(seg_img))

    # create segmentation and cell data folder if they don't exist
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
    if not os.path.exists(cell_dir):
        os.makedirs(cell_dir)

    # load specs file
    specs = load_specs(ana_dir)

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in fovs]

    ### Create cell lineages from segmented images
    information("Creating cell lineages using standard algorithm.")

    # This dictionary holds information for all cells
    cells = {}

    # do lineage creation per fov, so pooling can be done by peak
    for fov_id in fov_id_list:
        # update will add the output from make_lineages_function, which is a
        # dict of Cell entries, into cells
        cells.update(
            make_lineages_fov(
                ana_dir,
                experiment_name,
                fov_id,
                specs,
                lost_cell_time,
                new_cell_y_cutoff,
                new_cell_region_cutoff,
                max_growth_length,
                min_growth_length,
                max_growth_area,
                min_growth_area,
                pxl2um,
                seg_img,
                phase_plane,
            )
        )
    information("Finished lineage creation.")

    ### Now prune and save the data.
    information("Curating and saving cell data.")

    # this returns only cells with a parent and daughters
    complete_cells = find_complete_cells(cells)

    ### save the cell data
    # All cell data (includes incomplete cells)
    with open(cell_dir / "all_cells.pkl", "wb") as cell_file:
        pickle.dump(cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    ### save to .json
    write_cells_to_json(cells, cell_dir / "all_cells.json")

    # Just the complete cells, those with mother and daughter
    # This is a dictionary of cell objects.
    with open(cell_dir / "complete_cells.pkl", "wb") as cell_file:
        pickle.dump(complete_cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    ### save to .json
    write_cells_to_json(complete_cells, cell_dir / "complete_cells.json")

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

    def run(self):
        """Overriding method. Performs Mother Machine Analysis"""
        self.viewer.window._status_bar._toggle_activity_dock(True)
        viewer = napari.current_viewer()
        viewer.layers.clear()
        # need this to avoid vispy bug for some reason
        # related to https://github.com/napari/napari/issues/2584
        viewer.add_image(np.zeros((1, 1)))
        viewer.layers.clear()

        track_cells(
            experiment_name=self.experiment_name,
            fovs=self.fovs,
            phase_plane=self.phase_plane,
            pxl2um=self.pxl2um,
            num_analyzers=multiprocessing.cpu_count(),
            lost_cell_time=self.lost_cell_time,
            new_cell_y_cutoff=self.new_cell_y_cutoff,
            new_cell_region_cutoff=self.new_cell_region_cutoff,
            max_growth_length=self.max_growth_length,
            min_growth_length=self.min_growth_length,
            max_growth_area=self.max_growth_area,
            min_growth_area=self.min_growth_area,
            seg_img="seg_otsu" if self.segmentation_method == "Otsu" else "seg_unet",
            ana_dir=self.analysis_folder,
            seg_dir=self.analysis_folder / "segmented",
            cell_dir=self.analysis_folder / "cell_data",
        )

    def display_fovs(self):
        viewer = napari.current_viewer()
        viewer.layers.clear()

        specs = load_specs(self.analysis_folder)

        for fov_id in self.fovs_to_display:
            ana_peak_ids = []  # channels to be analyzed
            for peak_id, spec in six.iteritems(specs[int(fov_id)]):
                if spec == 1:  # 1 means analyze
                    ana_peak_ids.append(peak_id)
            ana_peak_ids = sorted(ana_peak_ids)  # sort for repeatability

            for peak_id in ana_peak_ids:
                try:
                    img_stack = load_lineage_image(
                        self.analysis_folder, self.experiment_name, fov_id, peak_id
                    )
                    lin_filename = f"{self.experiment_name}_{fov_id}_{peak_id}.tif"

                    viewer.grid.enabled = True
                    viewer.grid.shape = (-1, 1)

                    viewer.add_image(img_stack, name=lin_filename)
                except Exception as e:
                    warning(
                        f"{e}. No lineage found for FOV {fov_id}, peak {peak_id}. Run lineage construction first!"
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


if __name__ == "__main__":
    experiment_name = ""
    cur_dir = Path(".")
    analysis_folder = cur_dir / "analysis"
    end_time = get_valid_times(cur_dir / "TIFF")
    all_fovs = get_valid_fovs_folder(cur_dir / "TIFF")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_time", help="1-indexed time to start at", default=1, type=int
    )
    parser.add_argument(
        "--end_time",
        help="1-indexed time to end at (exclusive)",
        default=end_time,
        type=int,
    )
    parser.add_argument("--fovs", help="Which FOVs to include?", default="", type=str)
    p = parser.parse_args()

    if p.fovs == "":
        fovs = all_fovs
    else:
        fovs = range_string_to_indices(p.fovs)
        for fov in fovs:
            if fov not in all_fovs:
                raise ValueError("Some FOVs are out of range for your nd2 file.")

    if (p.start_time < 0) or (p.end_time > end_time) or (p.start_time > p.end_time):
        raise ValueError("Times out of range")

    track_cells(
        experiment_name=experiment_name,
        fovs=fovs,
        phase_plane="c1",
        pxl2um=0.11,
        num_analyzers=multiprocessing.cpu_count(),
        lost_cell_time=3,
        new_cell_y_cutoff=150,
        new_cell_region_cutoff=4,
        max_growth_length=1.3,
        min_growth_length=0.8,
        max_growth_area=1.3,
        min_growth_area=0.8,
        seg_img="seg_otsu",
        ana_dir=analysis_folder,
        seg_dir=analysis_folder / "segmented",
        cell_dir=analysis_folder / "cell_data",
    )
