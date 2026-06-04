import multiprocessing as mp
from collections import namedtuple
from concurrent import futures as cf
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import six
from magicgui.widgets import Container, PushButton
from napari import Viewer
from napari.utils import progress
from skimage import io
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties

from ._deriving_widgets import (
    FOVList,
    MM3Container2,
    get_valid_fovs_folder,
    get_valid_planes,
    information,
    load_specs,
    load_tiff,
    load_timetable,
)
from .utils import (
    TIFF_FORMAT_PEAK,
    Cell,
    construct_cell,
    find_complete_cells,
    write_cells_to_json,
)


@dataclass
class RunParams:
    fovs: FOVList
    phase_plane: str
    pxl2um: Annotated[
        float,
        {"min": 0.0, "max": 2.0, "step": 0.001, "tooltip": "Micrometers per pixel"},
    ] = 0.11
    num_analyzers: int = mp.cpu_count()
    growth_ratio: Annotated[
        float,
        {
            "tooltip": "growth ratio of cells from frame-to-frame. For unbiased centroid matching (eg, in an antibiotic switch or growth arrest experiments), set this to 1."
        },
    ] = 1.0
    lost_cell_time: Annotated[
        int,
        {
            "min": 0,
            "max": 20,
            "tooltip": "Number of frames after which a cell is dropped if no new regions connect to it.",
        },
    ] = 3
    new_cell_y_cutoff: Annotated[
        int,
        {
            "min": 1,
            "tooltip": "How far from the top (pixels) to consider cells for starting new lineages",
        },
    ] = 350
    new_cell_region_cutoff: Annotated[
        int,
        {
            "min": 0,
            "tooltip": "how many cells from the top to consider for starting new lineages",
        },
    ] = 4
    max_growth_length: Annotated[
        float,
        {
            "min": 0.0,
            "max": 20.0,
            "step": 0.1,
            "tooltip": "max change in length allowed when linking regions. Ratio.",
        },
    ] = 1.3  # Max allowed growth length ratio.
    min_growth_length: Annotated[
        float,
        {
            "min": 0.0,
            "step": 0.1,
            "tooltip": "min change in length allowed when linking regions. Ratio.",
        },
    ] = 0.8
    max_growth_area: Annotated[
        float,
        {
            "min": 0.0,
            "step": 0.1,
            "tooltip": "max change in length allowed when linking regions. Ratio.",
        },
    ] = 1.3
    min_growth_area: Annotated[
        float,
        {
            "min": 0.0,
            "step": 0.1,
            "tooltip": "min change in length allowed when linking regions. Ratio.",
        },
    ] = 0.8


@dataclass
class TrackParams:
    lost_cell_time: int
    y_cutoff: int
    region_cutoff: int
    growth_ratio: float

    min_growth_length: float
    max_growth_length: float
    min_growth_area: float
    max_growth_area: float

    def __init__(self, run_params: RunParams):
        self.lost_cell_time = run_params.lost_cell_time
        self.y_cutoff = run_params.new_cell_y_cutoff
        self.region_cutoff = run_params.new_cell_region_cutoff
        self.growth_ratio = run_params.growth_ratio
        self.min_growth_area = run_params.min_growth_area
        self.min_growth_length = run_params.min_growth_length
        self.max_growth_area = run_params.max_growth_area
        self.max_growth_length = run_params.max_growth_length


CellId = namedtuple("CellId", ["t_idx", "region_id"])
DivisionEdgeSingle = namedtuple("DivisionEdgeSingle", ["daughter"])
DivisionEdge = namedtuple("DivisionEdge", ["daughter1", "daughter2"])
GrowthEdge = namedtuple("GrowthEdge", ["cell_id"])
Edge = GrowthEdge | DivisionEdge | DivisionEdgeSingle


@dataclass
class CellGraph:
    orphans: set[CellId]
    cell_ids: set[CellId]
    leaves: set[CellId]
    edges: dict[CellId, Edge]
    regions: dict[CellId, RegionProperties]

    def __init__(self):
        self.regions = {}
        self.cell_ids: set[CellId] = set()
        self.leaves: set[CellId] = set()
        self.edges: dict[CellId, Edge] = {}
        self.orphans: set[CellId] = set()

    def add_leaf(self, t, region, orphan=True):
        cell_id = CellId(t, region_id=region.label)
        self.leaves.add(cell_id)
        self.regions[cell_id] = region
        if orphan:
            self.orphans.add(cell_id)
        return cell_id

    def remove_leaf(self, cell_id: CellId):
        self.leaves.remove(cell_id)

    def get_leaf_regions(self):
        for leaf in self.leaves:
            yield self.regions[leaf]

    def connect_leaf_to_region(self, parent_leaf_id: CellId, t, region):
        child_leaf_id = self.add_leaf(t, region, orphan=False)
        self.remove_leaf(parent_leaf_id)
        self.edges[parent_leaf_id] = GrowthEdge(child_leaf_id)

    def divide_cell(
        self,
        cell_id: CellId,
        t,
        region1: Optional[RegionProperties] = None,
        region2: Optional[RegionProperties] = None,
    ):
        self.leaves.remove(cell_id)

        if region1 and region2:
            daughter1 = self.add_leaf(t, region1, orphan=False)
            daughter2 = self.add_leaf(t, region2, orphan=False)
            edge = DivisionEdge(daughter1=daughter1, daughter2=daughter2)
        elif region1:
            daughter1 = self.add_leaf(t, region1, orphan=False)
            edge = DivisionEdgeSingle(daughter=daughter1)
        elif region2:
            daughter2 = self.add_leaf(t, region2, orphan=False)
            edge = DivisionEdgeSingle(daughter=daughter2)
        else:
            return

        self.edges[cell_id] = edge


def trace_cell_growths(cell_graph: CellGraph, cell_id: CellId) -> list[CellId]:
    if cell_id in cell_graph.edges.keys():
        edge = cell_graph.edges[cell_id]
        if isinstance(edge, GrowthEdge):
            return [cell_id] + trace_cell_growths(cell_graph, edge.cell_id)
    return [cell_id]


def trace_cell_daughters(cell_graph: CellGraph, cell_id: CellId) -> list[CellId]:
    if cell_id in cell_graph.edges:
        edge = cell_graph.edges[cell_id]
        if isinstance(edge, DivisionEdge):
            return [edge.daughter1, edge.daughter2]
        if isinstance(edge, DivisionEdgeSingle):
            return [
                edge.daughter,
            ]
    return []


def prune_leaves(cell_graph: CellGraph, t: int, lost_cell_time: int):
    """
    Remove any leaves that haven't had children for too long.
    """
    real_t = t  # timetable[t]
    initial_leaves = cell_graph.leaves.copy()
    for cell_id in initial_leaves:
        cell_real_t = cell_id.t_idx  # timetable[cell_id.t_idx]
        if real_t - cell_real_t > lost_cell_time:
            cell_graph.remove_leaf(cell_id)


def add_regions(
    graph: CellGraph, track_params: TrackParams, t_idx, regions: list[RegionProperties]
):
    for r in regions:
        if (
            r.centroid[0] < track_params.y_cutoff
            and r.label <= track_params.region_cutoff
        ):
            graph.add_leaf(t_idx, r)
        else:
            break


def check_growth(
    track_params: TrackParams,
    y0,
    old_cell_region: RegionProperties,
    new_cell_region: RegionProperties,
) -> bool:
    """
    Checks to see if it makes sense
    to grow a cell (represented by a region) by another region

    Returns
    -------
    bool
        True if it makes sense to grow the cell
    """
    length1, area1 = old_cell_region.axis_major_length, old_cell_region.area
    length1, area1 = (
        length1 * track_params.growth_ratio,
        area1 * track_params.growth_ratio,
    )
    length2, area2 = new_cell_region.axis_major_length, new_cell_region.area
    if length1 * track_params.max_growth_length < length2:
        return False

    if length1 * track_params.min_growth_length > length2:
        return False

    if area1 * track_params.max_growth_area < area2:
        return False

    if area1 * track_params.min_growth_area > area2:
        return False

    # check if y position of region is within the bounding box of previous region
    min_row = old_cell_region.bbox[0]
    max_row = old_cell_region.bbox[2] - 1  # upper range not inclusive in regionprops
    if (
        new_cell_region.centroid[0] < y0 + (min_row - y0) * track_params.growth_ratio
    ) or (
        new_cell_region.centroid[0] > y0 + (max_row - y0) * track_params.growth_ratio
    ):
        return False

    # return true if you get this far
    return True


def check_division(
    track_params: TrackParams,
    y0,
    parent_region: RegionProperties,
    region1: RegionProperties,
    region2: RegionProperties,
) -> bool:
    """
    Checks:
     * that the daughters are of a reasonable size,
     * upper daughter's center is in the upper half of our cell, and lower daughter is in the lower half of our cell.

    Return False if nothing should happend and regions ignored
    Return True if cell should divide into the regions.
    """
    r1, r2 = (region1, region2) if region1.label < region2.label else (region2, region1)

    # make sure combined size of daughters is not too big
    combined_size = region1.axis_major_length + region2.axis_major_length

    parent_length = parent_region.axis_major_length
    max_size = (
        parent_length * track_params.max_growth_length * track_params.growth_ratio
    )
    min_size = (
        parent_length * track_params.min_growth_length * track_params.growth_ratio
    )

    if (max_size < combined_size) or (combined_size < min_size):
        return False

    cell_bottom, cell_top = (
        y0 + (parent_region.bbox[0] - y0) * track_params.growth_ratio,
        y0 + (parent_region.bbox[2] - y0) * track_params.growth_ratio,
    )
    cell_center = y0 + (parent_region.centroid[0] - y0) * track_params.growth_ratio
    # centroid of region 1 must be in the bottom half of the parent cell
    if (cell_center < r1.centroid[0]) or (r1.centroid[0] < cell_bottom):
        return False
    # centroid of region 2 must be in the top half of the parent cell
    if (cell_center > r2.centroid[0]) or (cell_top < r2.centroid[0]):
        return False

    return True


def link_new_regions(
    track_params: TrackParams,
    graph: CellGraph,
    regions: list,
    t: int,
):
    """
    Main iterative tracking function. Links regions in current time point onto previously tracked cells.
    """
    if len(regions) == 0:
        return

    region_ys = [r.bbox[0] for r in regions]
    y0 = min(region_ys)
    leaf_ys = np.array([graph.regions[leaf_id].centroid[0] for leaf_id in graph.leaves])
    leaf_ys_shifted = y0 + (leaf_ys - y0) * track_params.growth_ratio

    # Map each leaf to a set of regions that are closest to it.
    leaf_region_map = {leaf_id: [] for leaf_id in graph.leaves}
    leaf_region_dist_map = {leaf_id: [] for leaf_id in graph.leaves}
    for r_id, region in enumerate(regions):
        closest_leaf_id = None
        closest_leaf_distance = float("inf")

        for leaf_id, leaf_centroid in zip(graph.leaves, leaf_ys_shifted):
            y_dist_region_to_leaf = abs(region.centroid[0] - leaf_centroid)

            if y_dist_region_to_leaf < closest_leaf_distance:
                closest_leaf_id = leaf_id
                closest_leaf_distance = y_dist_region_to_leaf

        assert isinstance(closest_leaf_id, CellId)
        # leaf_region_map[closest_leaf_id].append((r_id, y_dist_region_to_leaf))
        leaf_region_map[closest_leaf_id].append(region)
        leaf_region_dist_map[closest_leaf_id].append(y_dist_region_to_leaf)

    # given the above map, check for reasonable cell events.
    for leaf_id in leaf_region_map.keys():
        leaf_region = graph.regions[leaf_id]
        closest_regions_to_leaf = leaf_region_map[leaf_id]
        region_to_leaf_distances = leaf_region_dist_map[leaf_id]
        if len(closest_regions_to_leaf) < 1:
            continue
        if len(closest_regions_to_leaf) == 1:
            first_region = closest_regions_to_leaf[0]
            if check_growth(track_params, y0, leaf_region, first_region):
                graph.connect_leaf_to_region(leaf_id, t, first_region)
            else:
                add_regions(graph, track_params, t, [first_region])
            continue
        if len(closest_regions_to_leaf) >= 2:
            # note: tuples are compared element-by-element, which makes this work.
            sorted_regions = [
                r
                for _, r in sorted(
                    zip(region_to_leaf_distances, closest_regions_to_leaf)
                )
            ]
            rest_of_rs = []
            if check_growth(track_params, y0, leaf_region, sorted_regions[0]):
                graph.connect_leaf_to_region(leaf_id, t, sorted_regions[0])
                rest_of_rs = sorted_regions[1:]
            elif check_growth(track_params, y0, leaf_region, sorted_regions[1]):
                graph.connect_leaf_to_region(leaf_id, t, sorted_regions[1])
                rest_of_rs = [sorted_regions[0]] + sorted_regions[2:]
            elif check_division(
                track_params,
                y0,
                leaf_region,
                sorted_regions[0],
                sorted_regions[1],
            ):
                r1, r2 = sorted_regions[0], sorted_regions[1]
                rest_of_rs = sorted_regions[2:]
                if (r1.centroid[0] >= track_params.y_cutoff) or (
                    r1.label > track_params.region_cutoff
                ):
                    r1 = None

                if (r2.centroid[0] >= track_params.y_cutoff) or (
                    r2.label > track_params.region_cutoff
                ):
                    r2 = None

                graph.divide_cell(leaf_id, t, r1, r2)

            add_regions(graph, track_params, t, rest_of_rs)


def cell_lineage_from_graph(
    target_cell_id: CellId,
    cell_graph: CellGraph,
    pxl2um,
    cell_id_fmt: str,
    timetable,
    parent_id: str | None = None,
) -> dict[str, Cell]:
    """
    Given a lineage graph converts it to something more usable.
    """
    cell_str_id = cell_id_fmt.format(target_cell_id.t_idx, target_cell_id.region_id)
    cell_growths = trace_cell_growths(cell_graph, target_cell_id)
    cell_daughters = trace_cell_daughters(cell_graph, cell_id=cell_growths[-1])

    cell_daughter_str_ids = []
    cell_daughter_objects = []

    # trace daughters and add them to the graph.
    out_dict: dict[str, Cell] = {}
    if cell_daughters:
        x = 0
        for daughter_id in cell_daughters:
            out_dict = out_dict | cell_lineage_from_graph(
                daughter_id,
                cell_graph,
                pxl2um,
                cell_id_fmt,
                timetable,
                parent_id=cell_str_id,
            )
            daughter_str_id = cell_id_fmt.format(
                daughter_id.t_idx, daughter_id.region_id
            )
            cell_daughter_str_ids.append(daughter_str_id)
            cell_daughter_objects.append(out_dict[daughter_str_id])
            x += 1

    region_list = [cell_graph.regions[cell_id] for cell_id in cell_growths]
    t_idxs = [cell_id.t_idx for cell_id in cell_growths]
    # make a cell object for our boy.
    out_dict[cell_str_id] = construct_cell(
        cell_str_id,
        pxl2um,
        timetable,
        region_list,
        t_idxs,
        parent_id=parent_id,
        daughters=cell_daughter_objects,
    )

    return out_dict


def count_complete_cells(
    cell_graph: CellGraph, cell_id: CellId | None = None, has_parent=False
):
    # recursion end cases
    if cell_id is None:
        cnt = 0
        for cell_id in cell_graph.orphans:
            cnt += count_complete_cells(cell_graph, cell_id, has_parent=False)
        return cnt

    if cell_id in cell_graph.leaves:
        return 0

    # continue recursing
    if cell_id not in cell_graph.edges:
        # print(f"cell_id {cell_id}")
        return 0

    edge = cell_graph.edges[cell_id]
    if isinstance(edge, GrowthEdge):
        return count_complete_cells(cell_graph, edge.cell_id, has_parent=has_parent)
    elif isinstance(edge, DivisionEdge):
        # print(f"division! {edge}, {has_parent=}")
        return (
            has_parent
            + count_complete_cells(cell_graph, cell_id=edge.daughter1, has_parent=True)
            + count_complete_cells(cell_graph, cell_id=edge.daughter2, has_parent=True)
        )

    return 0


def cell_graph_to_dict(
    cell_graph: CellGraph,
    pxl2um: float,
    cell_id_fmt: str,
    timetable,
) -> dict[str, Cell]:
    cells = {}

    for cell_id in cell_graph.orphans:
        cells = cells | cell_lineage_from_graph(
            cell_id,
            cell_graph,
            pxl2um,
            cell_id_fmt,
            timetable,
        )

    return cells


# Creates lineage for a single channel
def make_channel_lineages(
    timetable: Path,
    segmented_dir: Path,
    experiment_name: str,
    track_params: TrackParams,
    pxl2um: float,
    seg_img: str,
    fov_and_peak_id: tuple,
) -> CellGraph:
    """
    Returns
    -------
    cells : dict
        A dictionary of all the cells from this lineage, divided and undivided
    """

    # get the specific ids from the tuple
    fov_id, peak_id = fov_and_peak_id

    # time_table = load_timetable(timetable)
    start_time_index = (
        0  # should be derived from segmented_dir  # min(time_table[fov_id].keys())
    )

    information("Creating lineage for FOV %d, channel %d." % (fov_id, peak_id))

    img_filename = TIFF_FORMAT_PEAK % (experiment_name, fov_id, peak_id, seg_img)
    try:
        image_data_seg = load_tiff(segmented_dir / img_filename)
    except FileNotFoundError:
        return CellGraph()

    # Calculate all data for all time points.
    regions_by_time = [
        regionprops(label_image=timepoint) for timepoint in image_data_seg
    ]
    # print(regions_by_time)

    # generate the graph in the form of a (uni-directional) linked list.
    graph = CellGraph()
    for t, regions in enumerate(regions_by_time, start=start_time_index):
        prune_leaves(graph, t, track_params.lost_cell_time)
        if not graph.leaves:
            add_regions(graph, track_params, t, regions)
        else:
            link_new_regions(track_params, graph, regions, t)
    # print(graph)
    # print(graph)
    return graph


def make_channel_lineage_old(
    timetable: Path,
    segmented_dir: Path,
    experiment_name: str,
    track_params: TrackParams,
    pxl2um: float,
    seg_img: str,
    fov_and_peak_id: tuple,
):
    print("tracing lineage")
    graph = make_channel_lineages(
        timetable,
        segmented_dir,
        experiment_name,
        track_params,
        pxl2um,
        seg_img,
        fov_and_peak_id,
    )

    fov_id, peak_id = fov_and_peak_id
    try:
        time_table = load_timetable(timetable)
        time_table[fov_id] = {t_idx: val for t_idx, val in time_table[fov_id].items()}
    except FileNotFoundError:
        time_table = {}
        stupid_table = {i: i for i in range(1000)}  # TODO: Make this sensible.
        time_table[fov_id] = stupid_table
    cell_id_fmt = f"{experiment_name}f{fov_id:0=2}p{peak_id:0=4}t{{:0=4}}r{{:0=2}}"

    # NOTE: This is the bottleneck. It is also the only part of the code I haven't touched.
    print("converting lineage")
    out_dict = cell_graph_to_dict(graph, pxl2um, cell_id_fmt, time_table)
    return out_dict


# finds lineages for all peaks in a fov


def make_lineages_fov(
    timetable_path: Path,
    lineages_path: Path,
    segmented_path: Path,
    experiment_name: str,
    fov_id: int,
    specs: dict,
    track_params: TrackParams,
    pxl2um: float,
    seg_img: str,
    phase_plane: str,
    num_analyzers: int,
    debug: bool = True,
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

    fov_and_peak_ids_list = [(fov_id, peak_id) for peak_id in ana_peak_ids]
    # use up all parameters EXCEPT fov_and_peak_ids
    temp_worker = partial(
        make_channel_lineage_old,
        timetable_path,
        segmented_path,
        experiment_name,
        track_params,
        pxl2um,
        seg_img,
    )

    cells = {}  # create dictionary to hold all information
    if debug:
        # # This is the non-parallelized version (useful for debug)
        for fov_and_peak_id in progress(fov_and_peak_ids_list):
            cell_dict = temp_worker(fov_and_peak_id)

            cells = cells | cell_dict  # updates cells with the entries in cell_dict
    else:
        # # set up multiprocessing pool. will complete pool before going on
        with cf.ThreadPoolExecutor(max_workers=num_analyzers) as executor:
            it = executor.map(temp_worker, fov_and_peak_ids_list)
            for fov, cell_dict in progress(
                zip(fov_and_peak_ids_list, it),
                total=len(fov_and_peak_ids_list),
                desc=f"Processing FOV {fov_id}",
            ):
                cells.update(cell_dict)  # updates cells with the entries in cell_dict
                print(f"Finished {fov}", end="\r")
            print("Finished analysis of all FOVs.")
    return cells


def load_lineage_image(lin_dir, experiment_name, fov_id, peak_id):
    """Loads a lineage image for the given FOV and peak

    Returns
    -------
    img_stack : np.ndarray
        Image stack of lineage images
    """

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


@dataclass
class InPaths:
    """
    1. check folders for existence, fetch FOVs & times & planes
        -> upon failure, simply show a list of inputs + update button.
    """

    timetable_path: Path = Path("./analysis/timetable.json")
    specs_path: Path = Path("./analysis/specs.yaml")
    segmented_dir: Annotated[Path, {"mode": "d"}] = Path("./analysis/segmented")
    channels_dir: Annotated[Path, {"mode": "d"}] = Path("./analysis/channels")
    seg_img: Annotated[str, {"choices": ["seg_unet", "seg_otsu"]}] = "seg_otsu"


@dataclass
class OutPaths:
    cell_dir: Annotated[Path, {"mode": "d"}] = Path("./analysis/cell_data")
    lineages_path: Annotated[Path, {"mode": "d"}] = Path("./analysis/lineages")
    experiment_name: str = ""


def gen_default_run_params(in_files: InPaths):
    try:
        all_fovs = get_valid_fovs_folder(in_files.segmented_dir)
        # get the brightest channel as the default phase plane!
        # TODO: Bad! In the long run i'd like to eliminate this.
        channels = get_valid_planes(in_files.channels_dir)
        # move this into runparams somehow!
        params = RunParams(
            phase_plane=channels[0],
            fovs=FOVList(all_fovs),
        )
        params.__annotations__["phase_plane"] = Annotated[str, {"choices": channels}]
        return params
    except FileNotFoundError:
        raise FileNotFoundError("TIFF folder not found")
    except ValueError:
        raise ValueError(
            "Invalid filenames. Make sure that timestamps are denoted as t[0-9]* and FOVs as xy[0-9]*"
        )


def track_cells(in_paths: InPaths, run_params: RunParams, out_paths: OutPaths):
    """Track cells in a set of FOVs"""

    # Load the project parameters file
    information("Loading experiment parameters.")
    information(f"Using {run_params.num_analyzers} threads for multiprocessing.")
    information(f"Using {in_paths.seg_img} images for tracking.")

    # create segmentation and cell data folder if they don't exist
    if not out_paths.cell_dir.exists():
        out_paths.cell_dir.mkdir()

    # load specs file
    specs = load_specs(in_paths.specs_path)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])
    fov_id_list[:] = [fov for fov in fov_id_list if fov in run_params.fovs]

    ### Create cell lineages from segmented images
    information("Creating cell lineages using standard algorithm.")

    # This dictionary holds information for all cells
    cells = {}

    # do lineage creation per fov, so pooling can be done by peak
    for fov_id in fov_id_list:
        # update will add the output from make_lineages_function, which is a
        # dict of Cell entries, into cells
        cells = cells | make_lineages_fov(
            in_paths.timetable_path,
            out_paths.lineages_path,
            in_paths.segmented_dir,
            out_paths.experiment_name,
            fov_id,
            specs,
            TrackParams(run_params),
            run_params.pxl2um,
            in_paths.seg_img,
            run_params.phase_plane,
            run_params.num_analyzers,
        )
    information("Finished lineage creation.")

    # write_cells_to_json(cells, "test.json")
    # save the cell data
    # a cell is 'complete' if it has a daughter and a mother.
    write_cells_to_json(cells, out_paths.cell_dir / "all_cells.json")

    complete_cells = find_complete_cells(cells)
    if complete_cells is not None:
        write_cells_to_json(complete_cells, out_paths.cell_dir / "complete_cells.json")

    # information("Finished curating and saving cell data.")


class Track(MM3Container2):
    def __init__(self, viewer: Viewer):
        super().__init__(viewer)
        self.viewer = viewer

        self.in_paths = InPaths()
        try:
            self.run_params = gen_default_run_params(self.in_paths)
            self.out_paths = OutPaths()
            self.initialized = True
            self.regen_widgets()

        except (FileNotFoundError, ValueError) as e:
            print(e)
            self.initialized = False
            self.regen_widgets()

    def regen_widgets(self):
        super().regen_widgets()

        self.cur_fov_idx = 0
        self.increment_fov_widget = PushButton(text="next fov")
        self.decrement_fov_widget = PushButton(text="prev fov")
        self.increment_fov_widget.clicked.connect(self.increment_fov)
        self.increment_fov_widget.clicked.connect(self.render_preview)
        self.decrement_fov_widget.clicked.connect(self.decrement_fov)
        self.decrement_fov_widget.clicked.connect(self.render_preview)

        self.change_cur_fov_widget = Container(
            widgets=[self.decrement_fov_widget, self.increment_fov_widget],
            layout="horizontal",
        )
        self.append(self.change_cur_fov_widget)

        self.cur_peak_idx = 0
        self.increment_peak_widget = PushButton(text="next peak")
        self.decrement_peak_widget = PushButton(text="prev peak")
        self.increment_peak_widget.clicked.connect(self.increment_peak)
        self.increment_peak_widget.clicked.connect(self.render_preview)
        self.decrement_peak_widget.clicked.connect(self.decrement_peak)
        self.decrement_peak_widget.clicked.connect(self.render_preview)

        self.change_cur_peak_widget = Container(
            widgets=[self.decrement_peak_widget, self.increment_peak_widget],
            layout="horizontal",
        )
        self.append(self.change_cur_peak_widget)

        self.preview_widget = PushButton(label="generate preview")
        self.append(self.preview_widget)
        self.preview_widget.changed.connect(self.render_preview)

        self.render_preview()

    def run(self):
        track_cells(self.in_paths, self.run_params, self.out_paths)

    def increment_fov(self):
        self.cur_fov_idx = min(self.cur_fov_idx + 1, len(self.run_params.fovs) - 1)
        self.cur_peak_idx = 0

    def decrement_fov(self):
        self.cur_fov_idx = max(self.cur_fov_idx - 1, 0)
        self.cur_peak_idx = 0

    def increment_peak(self):
        valid_fov = self.run_params.fovs[self.cur_fov_idx]
        specs = load_specs(self.in_paths.specs_path)
        total_peaks = len(
            [key for key in specs[valid_fov] if specs[valid_fov][key] == 1]
        )
        self.cur_peak_idx = min(self.cur_peak_idx + 1, total_peaks - 1)

    def decrement_peak(self):
        self.cur_peak_idx = max(self.cur_peak_idx - 1, 0)

    def get_cur_fov_and_peak(self):
        specs = load_specs(self.in_paths.specs_path)
        cur_fov = self.run_params.fovs[self.cur_fov_idx]
        cur_peak = [key for key in specs[cur_fov] if specs[cur_fov][key] == 1][
            self.cur_peak_idx
        ]
        return cur_fov, cur_peak

    def load_cur_pc(self):
        cur_fov, cur_peak = self.get_cur_fov_and_peak()
        fname = TIFF_FORMAT_PEAK % ("", cur_fov, cur_peak, "c1")
        return load_tiff(self.in_paths.channels_dir / fname)

    def load_cur_seg(self):
        cur_fov, cur_peak = self.get_cur_fov_and_peak()
        fname = TIFF_FORMAT_PEAK % ("", cur_fov, cur_peak, self.in_paths.seg_img)
        seg = load_tiff(self.in_paths.segmented_dir / fname)
        return seg

    def render_preview(self):
        valid_fov, valid_peak = self.get_cur_fov_and_peak()

        track_params = TrackParams(self.run_params)
        # # there should really be a way of doing this sans side effects...
        pc = self.load_cur_pc()
        width = pc.shape[-1]
        seg = self.load_cur_seg()

        pc = np.hstack(pc)
        seg = np.hstack(seg)

        self.viewer.layers.clear()

        self.viewer.add_image(
            pc, name=TIFF_FORMAT_PEAK % ("", valid_fov, valid_peak, "c1")
        )
        self.viewer.add_labels(
            seg,
            name=TIFF_FORMAT_PEAK % ("", valid_fov, valid_peak, self.in_paths.seg_img),
        )

        # Track cells in the currently visible peak
        cell_graph = make_channel_lineages(
            self.in_paths.timetable_path,
            self.in_paths.segmented_dir,
            self.out_paths.experiment_name,
            track_params,
            self.run_params.pxl2um,
            self.in_paths.seg_img,  # self.in_paths.seg_img,
            (valid_fov, valid_peak),
        )
        print(f"Complete cells: {count_complete_cells(cell_graph)}")
        self.plot_cell_graph(cell_graph, t_spacing=width)

    def plot_cell_graph(self, cell_graph: CellGraph, t_spacing: int):
        cell_pts, growth_vecs, edge_vecs = [], [], []
        for cell1_id, edge in cell_graph.edges.items():
            initial_cell = cell_graph.regions[cell1_id]
            v0_row, v0_col = initial_cell.centroid
            v0_col = v0_col + (cell1_id.t_idx) * t_spacing
            cell_pts.append([v0_row, v0_col])
            if isinstance(edge, GrowthEdge):
                cell2_id = edge.cell_id
                cell2_region = cell_graph.regions[cell2_id]
                v1_row, v1_col = cell2_region.centroid
                v1_col = v1_col + (cell2_id.t_idx) * t_spacing
                dcol = v1_col - v0_col
                drow = v1_row - v0_row
                vec = np.array(
                    [
                        [v0_row, v0_col],
                        [drow, dcol],
                    ]
                )
                growth_vecs.append(vec)
            elif isinstance(edge, DivisionEdge):
                cell2_id = edge.daughter1
                cell2_region = cell_graph.regions[cell2_id]
                v1_row, v1_col = cell2_region.centroid
                v1_col = v1_col + (cell2_id.t_idx) * t_spacing
                dcol = v1_col - v0_col
                drow = v1_row - v0_row
                vec1 = np.array(
                    [
                        [v0_row, v0_col],
                        [drow, dcol],
                    ]
                )
                cell3_id = edge.daughter2
                cell3_region = cell_graph.regions[cell3_id]
                v2_row, v2_col = cell3_region.centroid
                v2_col = v2_col + (cell3_id.t_idx) * t_spacing
                dcol = v2_col - v0_col
                drow = v2_row - v0_row
                vec2 = np.array(
                    [
                        [v0_row, v0_col],
                        [drow, dcol],
                    ]
                )
                edge_vecs.append(vec1)
                edge_vecs.append(vec2)

        self.viewer.add_points(
            data=cell_pts, face_color="yellow", size=10, name="cell centroids"
        )
        self.viewer.add_vectors(
            growth_vecs, edge_width=3, vector_style="line", name="growths"
        )
        self.viewer.add_vectors(
            edge_vecs,
            edge_width=3,
            vector_style="line",
            edge_color="orange",
            name="cell divisions",
        )


def trace_lineage(cells: dict, cell_id: str, ngens=3) -> list[str]:
    """
    Get a cell's lineage, in chronological order
    """
    cell = cells[cell_id]
    if cell.daughters is None:
        return [cell.id]

    daughter1_lineage = trace_lineage(cells, cell.daughters[0], ngens=ngens - 1)
    daughter2_lineage = trace_lineage(cells, cell.daughters[1], ngens=ngens - 1)

    final_lineage = [cell_id] + daughter1_lineage + daughter2_lineage
    return sorted(final_lineage)


def get_lineage_times(cells: dict[str, Cell], lineage: list[str]) -> list[int]:
    """
    Return a sorted list of every time associated with a set of cells.
    """
    all_times = []
    for cell_id in lineage:
        cell_times = cells[cell_id].times
        all_times += cell_times
    # all_times = list(set(all_times))
    return sorted(set(all_times))


def screenspace_centroids(cell: Cell, img_width):
    centroids = np.array(cell.centroids_px)
    offsets = np.arange(0, len(centroids) * img_width, img_width)

    xs = centroids[:, 1] + offsets
    ys = centroids[:, 0]
    return xs, ys


def plot_lineage(cells: dict, cell_lineage: list[str], img_stack: np.ndarray):
    width = img_stack.shape[-1]
    t0 = cells[cell_lineage[0]].times[0]
    lineage_times = get_lineage_times(cells, cell_lineage)
    # cell_lineage = trace_lineage(cells, p)
    # pc_stack = img_stack[..., 0, :, :]
    pc_stack = img_stack[lineage_times, :, :]

    pc_stack = np.hstack(pc_stack)  # ty: ignore

    plt.imshow(pc_stack)
    for cell_id in cell_lineage:
        cell = cells[cell_id]
        t_offset = cell.times[0] - t0
        xs, ys = screenspace_centroids(cell, width)
        plt.scatter(
            xs + t_offset * width,
            ys,
            edgecolors="yellow",
            facecolors="none",
        )

    plt.show()


if __name__ == "__main__":
    in_paths = InPaths()
    run_params = gen_default_run_params(in_paths)
    out_paths = OutPaths()
    track_cells(in_paths, run_params, out_paths)
