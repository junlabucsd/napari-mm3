from collections import namedtuple
from dataclasses import dataclass
from multiprocessing import Pool
import six
from typing import NamedTuple, Tuple, Union


from .utils import (
    Cell,
)

# in between two frames, a cell can:
# grow. divide. die/disappear.
# we are effectively representing a graph here.
# let's represent this as as a dictionary:
# the keys uniquely identify a region (time, idx)
# (namedtuple seems appropriate)
# the values represent connections -- as per the current code:
#    1 child => grow
#    2 children => divide
#    if we want to be really careful, we can actually label each edge...
#    but currently this is not worth it.


@dataclass
class Divide:
    daughter1: str
    daughter2: str


@dataclass
class Grow:
    successor: str


@dataclass
class Die:
    pass


CellEvent = Union[Divide, Grow, Die]
CellNode = namedtuple("CellNode", ["region_label", "time"])


class CellGraph:
    cell_graph: dict = {}
    leaves: set = set()

    # All methods must guarantee that there are never any leaves present
    # in cell_graphs.

    def divide_cell(self, parent, daughter1, daughter2):
        self.cell_graph[parent] = Divide(daughter1, daughter2)
        self.leaves.remove(parent)
        self.leaves.add(daughter1)
        self.leaves.add(daughter2)

    def grow_cell(self, parent, child):
        self.cell_graph[parent] = Grow(child)
        self.leaves.remove(parent)
        self.leaves.add(child)

    def kill_cell(self, parent):
        self.cell_graph[parent] = Die()

    def add_orphan(self, orphan):
        self.leaves.add(orphan)

    def prune_leaves(self, t, lost_cell_time):
        """
        Kills leaves older than lost_cell_time
        """
        for leaf in self.leaves:
            if t - leaf.time > lost_cell_time:
                self.leaves.remove(leaf)
                self.cell_graph[leaf] = Die()


def handle_discarded_regions(
    cell_graph: CellGraph,
    region_links: list,
    regions: list,
    y_cutoff,
    region_cutoff,
    t: int,
):
    """
    Process third+ regions down from closed end of channel.
    They will either be discarded or made into new cells.
    """
    discarded_regions = sorted(region_links, key=lambda x: x[1])[2:]
    for discarded_region in discarded_regions:
        region = regions[discarded_region[0]]
        if region.centroid[0] < y_cutoff and region.label <= region_cutoff:
            cell = CellNode(region.label, t)
            cell_graph.add_orphan(cell)
        else:
            break


def position_to_closest_region(positions, regions):
    """
    Loop through regions from current time step and match them to existing leaves
    """
    out = {i: [] for i, _ in enumerate(positions)}
    # go through regions, they will come off in Y position order
    for r, region in enumerate(regions):
        # create tuple which is cell_id of closest leaf, distance
        closest_position = 0
        closest_distance = float("inf")
        for i, p in enumerate(positions):
            # calculate distance between region and leaf
            y_distance = abs(region.centroid[0] - p)

            # if the distance is closer than before, update
            if y_distance < closest_distance:
                closest_position, closest_distance = i, y_distance

        # update map with the closest region
        out[closest_position].append((r, closest_distance))

    return out


def get_two_closest_regions(region_links: list) -> list:
    """
    Retrieve two regions closest to closed end of the channel.
    """
    closest_two_regions = sorted(region_links, key=lambda x: x[1])[:2]
    # but sort by region order so top region is first
    closest_two_regions = sorted(closest_two_regions, key=lambda x: x[0])
    return closest_two_regions


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

def update_region_links(
    leaf_region_map,
    regions: list,
    t: int,
):
    """
    Loop over current leaves and connect them to descendants
    """
    for leaf_id, region_links in six.iteritems(leaf_region_map):
        if len(region_links) == 1:
            region = regions[region_links[0][0]]
            if check_growth_by_region(
                cells[leaf_id],
                region,
            ):
                cells[leaf_id].grow(time_table, region, t)
        elif len(region_links) == 2:
            region1 = regions[region_links[0][0]]
            region2 = regions[region_links[1][0]]
            handle_two_regions(
                region1,
                region2,
                leaf_id,
                t,
            )

def link_regions_to_previous_cells(cell_graph: CellGraph, regions: list, t: int):
    """
    Map regions in current time point onto previously tracked cells
    """

    current_leaf_positions = [
        (leaf, regions[leaf].centroids[-1][0]) for leaf in cell_graph.leaves
    ]

    leaf_region_map = position_to_closest_region(
        regions=regions, positions=current_leaf_positions
    )

    for leaf_id, region_links in six.iteritems(leaf_region_map):
        if len(region_links) > 2:
            leaf_region_map[leaf_id] = get_two_closest_regions(region_links)
            handle_discarded_regions(
                cell_graph,
                region_links,
                regions,
                y_cutoff,
                region_cutoff,
                t,
            )

    update_region_links(
        leaf_region_map,
        regions,
        t,
    )




def update_cell_graph(regions, cell_graph: CellGraph, t, lost_cell_time):
    # assuming this function has been run up to t-1, update cell graph.
    # for t, regions in enumerate(regions_by_time):
    cell_graph.prune_leaves(t, lost_cell_time)
    if not cell_graph.leaves:
        for region in regions:
            cell_node = CellNode(region, t)
            cell_graph.add_orphan(cell_node)
    else:
        cell_graph.link_regions_to_previous_cells(
            regions,
            t,
        )


class CellTracker:
    """
    Class to track cells through time.
    """

    def __init__(
        self,
        fov_id: int,
        peak_id: int,
        y_cutoff: int,
        region_cutoff: int,
        lost_cell_time: int,
        time_table: dict,
        pxl2um: float,
        max_growth_length: float,
        min_growth_length: float,
        max_growth_area: float,
        min_growth_area: float,
        experiment_name: str,
    ):
        self.cell_leaves: list[str] = []
        self.cells: dict[str, Cell] = {}
        self.fov_id = fov_id
        self.peak_id = peak_id
        self.y_cutoff = y_cutoff
        self.region_cutoff = region_cutoff
        self.lost_cell_time = lost_cell_time
        self.time_table = time_table
        self.pxl2um = pxl2um
        self.max_growth_length = max_growth_length
        self.min_growth_length = min_growth_length
        self.max_growth_area = max_growth_area
        self.min_growth_area = min_growth_area
        self.experiment_name = experiment_name

    # implemented
    # def prune_leaves(self, t: int):
    #     """
    #     Remove leaves for cells that have been lost for more than lost_cell_time
    #     """
    #     for leaf_id in self.cell_leaves:
    #         if t - self.cells[leaf_id].times[-1] > self.lost_cell_time:
    #             self.cell_leaves.remove(leaf_id)

    def add_leaf_orphan(
        self,
        region,
        t: int,
    ):
        """
        Add new leaf if it clears thresholds.
        """
        if region.centroid[0] < self.y_cutoff and region.label <= self.region_cutoff:
            cell_id = self.create_cell_id(
                region,
                t,
            )
            self.add_cell(cell_id, region, t, parent_id=None)
            self.cell_leaves.append(cell_id)  # add to leaves

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
            self.cell_leaves.remove(leaf_id)
            self.add_leaf_daughter(region1, daughter1_id)
            self.add_leaf_daughter(region2, daughter2_id)

    def add_leaf_daughter(self, region, id: str):
        """
        Add new leaf to tree if it clears thresholds
        """
        if region.centroid[0] < self.y_cutoff and region.label <= self.region_cutoff:
            self.cell_leaves.append(id)

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
                cell_id = self.create_cell_id(
                    region,
                    t,
                )
                self.add_cell(cell_id, region, t, parent_id=None)
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

    def create_cell_id(
        self,
        region,
        t: int,
    ) -> str:
        """
        Make a unique cell id string for a new cell
        Returns
        -------
        cell_id: str
            string for cell ID
        """
        if self.experiment_name is None:
            cell_id = "f%02dp%04dt%04dr%02d" % (
                self.fov_id,
                self.peak_id,
                t,
                region.label,
            )
        else:
            cell_id = "{}f{:0=2}p{:0=4}t{:0=4}r{:0=2}".format(
                self.experiment_name, self.fov_id, self.peak_id, t, region.label
            )
        return cell_id

    def add_cell(self, cell_id, region, t, parent_id=None):
        self.cells[cell_id] = Cell(
            self.pxl2um,
            self.time_table,
            cell_id,
            region,
            t,
            parent_id=parent_id,
        )

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
        daughter1_id = self.create_cell_id(
            region1,
            t,
        )
        daughter2_id = self.create_cell_id(
            region2,
            t,
        )

        self.add_cell(daughter1_id, region1, t, parent_id=leaf_id)
        self.add_cell(daughter2_id, region2, t, parent_id=leaf_id)

        self.cells[leaf_id].divide(
            self.cells[daughter1_id], self.cells[daughter2_id], t
        )

        return daughter1_id, daughter2_id, self.cells
