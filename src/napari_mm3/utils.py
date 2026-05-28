from __future__ import division, print_function

import json
import pickle
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import FunctionType
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio
import six
import tifffile as tiff
import yaml
from scipy import ndimage as ndi
from skimage.measure._regionprops import RegionProperties

TIFF_FORMAT_PEAK = "%s_xy%03d_p%04d_%s.tif"
TIFF_FORMAT_NO_PEAK = "%s_xy%03d_%s.tif"
TIFF_FORMAT_PEAK_NO_SUFFIX = "%s_xy%03d_p%04d.tif"


@dataclass
class Cell:
    """
    All (most?) useful properties of a cell.

    All units are in pixels or dicrete indices (for times), unless specified otherwise.
    """

    id: str
    fov: int
    peak: int
    birth_label: int
    parent: str | None
    times: list[int]
    abs_times_s: list[float]
    birth_time: int
    labels: int
    bboxes_px: list[list[list[int]]]
    areas_px: list[int]
    orientations_rad: list[float]
    centroids_px: list[list[int]]
    lengths_um: list[float]
    widths_um: list[float]
    areas_um2: list[float]
    volumes_um3: list[float]

    # only for complete cells
    daughters: None | list[str]
    division_time: None | int
    abs_division_time_s: None | float
    growth_rate: None | float
    sb: None | float
    sd: None | float
    delta: None | float
    septum_position: None | float
    division_length_px: None | float
    division_width_px: None | float
    division_length_um: None | float
    division_width_um: None | float

    complete: bool

    # only for fluorescent cells.
    total_fluorescence: list[float] | None = None

    @property
    def times_w_div(self):
        return self.times + [self.division_time]

    @property
    def abs_times_w_div(self):
        return self.abs_times_s + [self.abs_division_time_s]

    @property
    def tau(self):
        return (self.abs_times_s[-1] - self.abs_times_s[0]) / 60.0


Cells = dict[str, Cell]


def construct_cell(
    cell_id: str,
    pxl2um: float,
    time_table: dict[int, dict[int, float]],
    regions: list[RegionProperties],
    times: list[int],
    parent_id: str | None,
    daughters: list[Cell] = [],
):
    """
    Could be done with a class method but it's not worth it.
    """
    fov = int(cell_id.split("f")[1].split("p")[0])

    def orient(region: RegionProperties):
        if region.orientation > 0:
            return -(np.pi / 2 - region.orientation)
        return np.pi / 2 + region.orientation

    length_and_widths = np.array([feretdiameter(r) for r in regions])
    lengths = length_and_widths[:, 0]
    lengths_um = lengths * pxl2um
    widths = length_and_widths[:, 1]
    widths_um = widths * pxl2um

    # calculate cell volume as cylinder plus hemispherical ends (sphere). Unit is px^3
    volumes = [
        (lengths - widths) * np.pi * (widths / 2) ** 2
        + (4 / 3) * np.pi * (widths / 2) ** 3
    ]

    areas_px = np.array([r.area for r in regions])
    areas_um2 = areas_px * pxl2um**2
    volumes_um3 = np.array(volumes) * pxl2um**3

    abs_times_s = [time_table[fov][t] for t in times]
    complete = False

    (
        sb,
        sd,
        delta,
        division_time,
        abs_division_time_s,
        division_length_px,
        division_length_um,
        division_width_px,
        division_width_um,
        growth_rate,
        septum_position,
    ) = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    if len(daughters) == 2:
        """
        Divide the cell and update stats.
        daugther1 and daugther2 are instances of the Cell class.
        daughter1 is the daugther closer to the closed end.
        """
        complete = True
        daughter1, daughter2 = daughters
        sb = lengths[0] * pxl2um
        # needs to be refactored
        sd = daughter1.lengths_um[0] + daughter2.lengths_um[0]
        delta = sd - sb
        division_time = daughter1.birth_time
        abs_division_time_s = time_table[fov][division_time]
        division_length_px = daughter1.lengths_um[0] + daughter2.lengths_um[0]
        division_length_um = pxl2um * division_length_px
        division_width_px = (daughter1.widths_um[0] + daughter2.widths_um[0]) / 2
        division_width_um = division_width_px * pxl2um

        # calculate the average growth rate
        try:
            times_ = np.float64((np.array(abs_times_s) - abs_times_s[0]) / 60.0)
            log_lengths = np.float64(np.log(lengths * pxl2um + [division_length_px]))
            p = np.polyfit(times_, log_lengths, 1)  # this wants float64
            growth_rate = p[0] * 60.0  # convert to hours

        except:
            growth_rate = np.float64("NaN")
            print("Elongation rate calculate failed for {}.".format(id))

        septum_position = daughter1.lengths_um[0] / (
            daughter2.lengths_um[0] + daughter2.lengths_um[0]
        )

    return Cell(
        id=cell_id,
        parent=parent_id,
        fov=fov,
        peak=int(cell_id.split("p")[1].split("t")[0]),
        birth_label=regions[0].label,
        times=times,
        abs_times_s=[time_table[fov][t] for t in times],
        birth_time=times[0],
        labels=[r.label for r in regions],  # ty: ignore
        bboxes_px=[r.bbox for r in regions],
        areas_px=[r.area for r in regions],
        orientations_rad=[orient(r) for r in regions],
        centroids_px=[r.centroid for r in regions],
        lengths_um=lengths * pxl2um,
        widths_um=widths * pxl2um,
        areas_um2=areas_um2,
        volumes_um3=volumes_um3,
        daughters=[daughter.id for daughter in daughters],
        # only for complete cells
        division_time=division_time,
        abs_division_time_s=abs_division_time_s,
        growth_rate=growth_rate,
        delta=delta,
        sb=sb,
        sd=sd,
        septum_position=septum_position,
        division_length_px=division_length_px,
        division_width_px=division_width_px,
        division_length_um=division_length_um,
        division_width_um=division_width_um,
        complete=complete,
    )


def compute_cell_df(cells: Cells) -> pd.DataFrame:
    cols = ["fov", "peak", "birth_time", "complete", "birth_region", "cell_id"]
    df = pd.DataFrame({c: [] for c in cols})
    for cell in cells.values():
        for c in cols:
            df[c].append(getattr(cell, c))

    df = df.sort_values(by=["fov", "peak", "birth_time", "birth_label"])
    return df


def dataframe_cell_stat(
    cells: Cells,
    fn: Callable[[Cell], Any],
    celldf: pd.DataFrame,
    out_name: str | None = None,
) -> list[Any]:
    """
    Computes a statistic for every cell, and writes it to a dataframe
    Inputs:
        cells: the cells to analyze
        fn: the statistic you want to compute
        celldf: if you want to save the stat to a dataframe, the dataframe to use
        out_name (optional): if saving the stat to a dataframe and you want a custom column name instead of the function name.
    Returns:
        values list[Any]
    """
    values = []
    if out_name is None:
        if isinstance(fn, FunctionType):
            out_name = fn.__name__
        else:
            raise ValueError(
                "If adding a new column to a datasheet and using a Callable without __name__ "
                "(likely a lambda expression), you must supply a column name."
            )

    for row in celldf:
        cell_id = row["cell_id"]
        cur_cell = cells[cell_id]
        stat = fn(cur_cell)
        values.append(stat)

    celldf[out_name] = values

    return values


def load_specs(path: Path) -> dict:
    """
    Load specs file which indicates which channels should be analyzed,
    used as empties, or ignored.
    """
    if path.suffix == ".yaml":
        with (path).open("r") as specs_file:
            specs = yaml.safe_load(specs_file)
        return specs
    try:
        with (path / "specs.yaml").open("r") as specs_file:
            specs = yaml.safe_load(specs_file)
    except FileNotFoundError:
        try:
            with (path / "specs.pkl").open("rb") as specs_file:
                specs = pickle.load(specs_file)
        except ValueError:
            print("Could not load specs file.")

    return specs


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def write_cells_to_json(cells: dict[str, Cell], path_out):
    json_out = {}
    for cell_id, cell in cells.items():
        json_out[cell_id] = vars(cell)
        try:
            json_out[cell_id].pop("time_table")
        except:
            pass
    with open(path_out, "w") as fout:
        json.dump(json_out, fout, sort_keys=True, indent=2, cls=NpEncoder)


def write_cells_to_matlab(cells, path_out):
    # TODO: make this run when we want.
    for cell_id in cells:
        cell = cells[cell_id]
        new_cell = {k: v for k, v in vars(cell).items() if v is not None}
        new_cell = cell_from_dict(new_cell)
        cells[cell_id] = new_cell
    with open(path_out, "wb") as f:
        sio.savemat(f, cells)


def cell_from_dict(in_dict):
    """Helper function for json deserialization.
    Turns a dictionary from a serialized cell object to a 'cell' object.
    """
    cell = Cell(**in_dict)
    for key, val in in_dict.items():
        vars(cell)[key] = val
    return cell


def place_in_cell(cell: Cell, x, y, t):
    """Translates from screen-space to in-cell coordinates."""
    # check if our cell exists at the current timestamp.
    if not (t in cell.times):
        return None, None, None

    cell_time = cell.times.index(t)
    bbox = cell.bboxes_px[cell_time]
    # check that the point is inside the cell's bounding box.
    if not ((bbox[0] < y) & (y < bbox[2]) & (bbox[1] < x) & (x < bbox[3])):
        return None, None, None

    centroid = cell.centroids_px[cell_time]
    orientation = cell.orientations_rad[cell_time]
    dx = x - centroid[1]
    dy = y - centroid[0]
    if orientation < 0:
        orientation = np.pi + orientation
    disp_y = dy * np.sin(orientation) - dx * np.cos(orientation)
    disp_x = dy * np.cos(orientation) + dx * np.sin(orientation)

    return disp_y, disp_x, cell_time


### Cell class and related functions


def read_cells_from_json(path_in: Path) -> dict[str, Cell]:
    with open(path_in, "r") as fin:
        json_loaded = json.load(fin)
    cells_new = {}
    for cell_id, cell in json_loaded.items():
        cells_new[cell_id] = cell_from_dict(cell)
    return Cells(cells_new)


# obtains cell length and width of the cell using the feret diameter
def feretdiameter(region):
    """
    feretdiameter calculates the length and width of the binary region shape. The cell orientation
    from the ellipsoid is used to find the major and minor axis of the cell.
    See https://en.wikipedia.org/wiki/Feret_diameter.

    Parameters
    ----------
    region : skimage.measure._regionprops._RegionProperties
        regionprops object of the binary region of the cell

    Returns
    -------
    length : float
        length of the cell
    width : float
        width of the cell
    """

    # y: along vertical axis of the image; x: along horizontal axis of the image;
    # calculate the relative centroid in the bounding box (non-rotated)
    # print(region.centroid)
    y0, x0 = region.centroid
    y0 = y0 - np.int16(region.bbox[0]) + 1
    x0 = x0 - np.int16(region.bbox[1]) + 1

    # orientation is now measured in RC coordinates - quick fix to convert
    # back to xy
    if region.orientation > 0:
        ori1 = -np.pi / 2 + region.orientation
    else:
        ori1 = np.pi / 2 + region.orientation
    cosorient = np.cos(ori1)
    sinorient = np.sin(ori1)

    amp_param = (
        1.2  # amplifying number to make sure the axis is longer than actual cell length
    )

    # coordinates relative to bounding box
    # r_coords = region.coords - [np.int16(region.bbox[0]), np.int16(region.bbox[1])]

    # limit to perimeter coords. pixels are relative to bounding box
    region_binimg = np.pad(
        region.image, 1, "constant"
    )  # pad region binary image by 1 to avoid boundary non-zero pixels
    distance_image = ndi.distance_transform_edt(region_binimg)
    r_coords = np.where(distance_image == 1)
    r_coords = list(zip(r_coords[0], r_coords[1]))

    # coordinates are already sorted by y. partion into top and bottom to search faster later
    # if orientation > 0, L1 is closer to top of image (lower Y coord)
    if (ori1) > 0:
        L1_coords = r_coords[: int(np.round(len(r_coords) / 4))]
        L2_coords = r_coords[int(np.round(len(r_coords) / 4)) :]
    else:
        L1_coords = r_coords[int(np.round(len(r_coords) / 4)) :]
        L2_coords = r_coords[: int(np.round(len(r_coords) / 4))]

    #####################
    # calculte cell length
    L1_pt = np.zeros((2, 1))
    L2_pt = np.zeros((2, 1))

    # define the two end points of the the long axis line
    # one pole.
    L1_pt[1] = x0 + cosorient * 0.5 * region.major_axis_length * amp_param
    L1_pt[0] = y0 - sinorient * 0.5 * region.major_axis_length * amp_param

    # the other pole.
    L2_pt[1] = x0 - cosorient * 0.5 * region.major_axis_length * amp_param
    L2_pt[0] = y0 + sinorient * 0.5 * region.major_axis_length * amp_param

    # calculate the minimal distance between the points at both ends of 3 lines
    # aka calcule the closest coordiante in the region to each of the above points.
    # pt_L1 = r_coords[np.argmin([np.sqrt(np.power(Pt[0]-L1_pt[0],2) + np.power(Pt[1]-L1_pt[1],2)) for Pt in r_coords])]
    # pt_L2 = r_coords[np.argmin([np.sqrt(np.power(Pt[0]-L2_pt[0],2) + np.power(Pt[1]-L2_pt[1],2)) for Pt in r_coords])]

    try:
        pt_L1 = L1_coords[
            np.argmin(
                [
                    np.sqrt(
                        np.power(Pt[0] - L1_pt[0], 2) + np.power(Pt[1] - L1_pt[1], 2)
                    )
                    for Pt in L1_coords
                ]
            )
        ]
        pt_L2 = L2_coords[
            np.argmin(
                [
                    np.sqrt(
                        np.power(Pt[0] - L2_pt[0], 2) + np.power(Pt[1] - L2_pt[1], 2)
                    )
                    for Pt in L2_coords
                ]
            )
        ]
        length = np.sqrt(
            np.power(pt_L1[0] - pt_L2[0], 2) + np.power(pt_L1[1] - pt_L2[1], 2)
        )
    except:
        length = None

    #####################
    # calculate cell width
    # draw 2 parallel lines along the short axis line spaced by 0.8*quarter of length = 0.4, to avoid  in midcell

    # limit to points in each half
    W_coords = []
    if (ori1) > 0:
        W_coords.append(
            r_coords[: int(np.round(len(r_coords) / 2))]
        )  # note the /2 here instead of /4
        W_coords.append(r_coords[int(np.round(len(r_coords) / 2)) :])
    else:
        W_coords.append(r_coords[int(np.round(len(r_coords) / 2)) :])
        W_coords.append(r_coords[: int(np.round(len(r_coords) / 2))])

    # starting points
    x1 = x0 + cosorient * 0.5 * length * 0.4
    y1 = y0 - sinorient * 0.5 * length * 0.4
    x2 = x0 - cosorient * 0.5 * length * 0.4
    y2 = y0 + sinorient * 0.5 * length * 0.4
    W1_pts = np.zeros((2, 2))
    W2_pts = np.zeros((2, 2))

    # now find the ends of the lines
    # one side
    W1_pts[0, 1] = x1 - sinorient * 0.5 * region.minor_axis_length * amp_param
    W1_pts[0, 0] = y1 - cosorient * 0.5 * region.minor_axis_length * amp_param
    W1_pts[1, 1] = x2 - sinorient * 0.5 * region.minor_axis_length * amp_param
    W1_pts[1, 0] = y2 - cosorient * 0.5 * region.minor_axis_length * amp_param

    # the other side
    W2_pts[0, 1] = x1 + sinorient * 0.5 * region.minor_axis_length * amp_param
    W2_pts[0, 0] = y1 + cosorient * 0.5 * region.minor_axis_length * amp_param
    W2_pts[1, 1] = x2 + sinorient * 0.5 * region.minor_axis_length * amp_param
    W2_pts[1, 0] = y2 + cosorient * 0.5 * region.minor_axis_length * amp_param

    # calculate the minimal distance between the points at both ends of 3 lines
    pt_W1 = np.zeros((2, 2))
    pt_W2 = np.zeros((2, 2))
    d_W = np.zeros((2, 1))
    i = 0
    for W1_pt, W2_pt in zip(W1_pts, W2_pts):
        # # find the points closest to the guide points
        # pt_W1[i,0], pt_W1[i,1] = r_coords[np.argmin([np.sqrt(np.power(Pt[0]-W1_pt[0],2) + np.power(Pt[1]-W1_pt[1],2)) for Pt in r_coords])]
        # pt_W2[i,0], pt_W2[i,1] = r_coords[np.argmin([np.sqrt(np.power(Pt[0]-W2_pt[0],2) + np.power(Pt[1]-W2_pt[1],2)) for Pt in r_coords])]

        # find the points closest to the guide points
        pt_W1[i, 0], pt_W1[i, 1] = W_coords[i][
            np.argmin(
                [
                    np.sqrt(
                        np.power(Pt[0] - W1_pt[0], 2) + np.power(Pt[1] - W1_pt[1], 2)
                    )
                    for Pt in W_coords[i]
                ]
            )
        ]
        pt_W2[i, 0], pt_W2[i, 1] = W_coords[i][
            np.argmin(
                [
                    np.sqrt(
                        np.power(Pt[0] - W2_pt[0], 2) + np.power(Pt[1] - W2_pt[1], 2)
                    )
                    for Pt in W_coords[i]
                ]
            )
        ]

        # calculate the actual width
        d_W[i] = np.sqrt(
            np.power(pt_W1[i, 0] - pt_W2[i, 0], 2)
            + np.power(pt_W1[i, 1] - pt_W2[i, 1], 2)
        )
        i += 1

    # take the average of the two at quarter positions
    width = np.mean([d_W[0], d_W[1]])
    return length, width


# take info and make string for cell id
def create_focus_id(region, t, peak, fov, experiment_name=None):
    """Make a unique focus id string for a new focus"""
    if experiment_name is None:
        focus_id = "f{:0=2}p{:0=4}t{:0=4}r{:0=2}".format(fov, peak, t, region.label)
    else:
        focus_id = "{}f{:0=2}p{:0=4}t{:0=4}r{:0=2}".format(
            experiment_name, fov, peak, t, region.label
        )
    return focus_id


# function for a growing cell, used to calculate growth rate
def cell_growth_func(t, sb, elong_rate):
    """
    Assumes you have taken log of the data.
    It also allows the size at birth to be a free parameter, rather than fixed
    at the actual size at birth (but still uses that as a guess)
    Assumes natural log, not base 2 (though I think that makes less sense)

    old form: sb*2**(alpha*t)
    """
    return sb + elong_rate * t


def filter_cells(cells: Cells, filt_func: Callable[[Cell], bool]) -> Cells:
    new_dict = {}
    for cell_id, cell_obj in cells.items():
        if filt_func(cell_obj):
            new_dict[cell_id] = cell_obj
    return new_dict


def map_cells(cells: Cells, map_func: Callable[[Cell], Any]) -> dict[str, Any]:
    new_dict: dict[str, Any] = {}
    for cell_id, cell_obj in cells.items():
        new_dict[cell_id] = map_func(cell_obj)
    return new_dict


# find cells with both a mother and two daughters
def find_complete_cells(cells: Cells) -> Cells:
    def is_complete_cell(cell: Cell):
        return bool(cell.daughters and cell.parent)

    return filter_cells(cells, is_complete_cell)


# finds cells whose birth label is 1
def find_mother_cells(cells) -> Cells:
    """Return only cells whose starting region label is 1."""

    def is_mother(cell):
        return cell.birth_label == 1

    return filter_cells(cells, is_mother)


def find_cells_of_birth_label(cells, label_num: (int | list[int]) = 1) -> Cells:
    def cell_of_birth_label(cell: Cell):
        if isinstance(label_num, int):
            return cell.birth_label == label_num
        return cell.birth_label in label_num

    return filter_cells(cells, cell_of_birth_label)


def organize_cells_by_channel_new(
    cells: dict[str, Cell],
) -> dict[int, dict[int, Cells]]:
    """
    Returns a nested dictionary where the keys are first
    the fov_id and then the peak_id (similar to specs),
    and the final value is a dictiary of cell objects that go in that
    specific channel, in the same format as normal {cell_id : Cell, ...}
    """

    # make a nested dictionary that holds lists of cells for one fov/peak
    cells_by_fov_peak = {}
    for cell_id, cell in cells.items():
        if cell.fov in cells_by_fov_peak:
            if cell.peak in cells_by_fov_peak[cell.fov]:
                cells_by_fov_peak[cell.fov][cell.peak][cell_id] = cell
            else:
                cells_by_fov_peak[cell.fov][cell.peak] = {cell_id: cell}
        else:
            cells_by_fov_peak[cell.fov] = {cell.peak: {cell_id: cell}}

    return cells_by_fov_peak


def organize_cells_by_channel(cells, specs) -> dict:
    """
    Returns a nested dictionary where the keys are first
    the fov_id and then the peak_id (similar to specs),
    and the final value is a dictiary of cell objects that go in that
    specific channel, in the same format as normal {cell_id : Cell, ...}
    """

    # make a nested dictionary that holds lists of cells for one fov/peak
    cells_by_peak = {}
    for fov_id in specs.keys():
        cells_by_peak[fov_id] = {}
        for peak_id, spec in specs[fov_id].items():
            # only make a space for channels that are analyized
            if spec == 1:
                cells_by_peak[fov_id][peak_id] = {}

    # organize the cells
    for cell_id, cell in cells.items():
        cells_by_peak[cell.fov][cell.peak][cell_id] = cell

    # remove peaks and that do not contain cells
    remove_fovs = []
    for fov_id, peaks in cells_by_peak.items():
        remove_peaks = []
        for peak_id in peaks.keys():
            if not peaks[peak_id]:
                remove_peaks.append(peak_id)

        for peak_id in remove_peaks:
            peaks.pop(peak_id)

        if not cells_by_peak[fov_id]:
            remove_fovs.append(fov_id)

    for fov_id in remove_fovs:
        cells_by_peak.pop(fov_id)

    return cells_by_peak


def gen_label_to_cell_mapping(cells, specs) -> dict:
    """
    Generates a map from (fov, peak, time, label) -> cell_id
    """
    fov_cells = organize_cells_by_channel(cells, specs)
    cell_map = {}
    for fov_id, peaks in fov_cells.items():
        cell_map[fov_id] = {}
        for peak_id, peak in peaks.items():
            new_peak = {}
            for cell_id, cell in peak.items():
                for time_idx, time in enumerate(cell.times):
                    if time not in new_peak:
                        new_peak[time] = {}
                    new_peak[time][cell.labels[time_idx]] = cell.id
            cell_map[fov_id][peak_id] = new_peak
    return cell_map


def infer_cell_id(cells: Cells, xloc, yloc, t):
    """
    Given screen-space coordinates and timestamp of a point, this finds the
    cell that point belongs to.
    Returns:
      cell_id: id of the cell the point is inside of.
      disp_y, disp_x: The cell-space displacement of the point.
      cell_time: the cell-time of the point
    """
    cell: Cell
    for cell_id, cell in cells.items():
        disp_y, disp_x, cell_time = cell.place_in_cell(xloc, yloc, t)
        if disp_y is None:
            continue
        return cell_id, disp_y, disp_x, cell_time
    return None, None, None, None


def find_screenspace_foci(cells: Cells):
    """
    From a list of cells, gets the absolute screen-space position of all foci.
    Returns:
      x_pts: List of foci x-positions
      y_pts: List of foci y-positions.
      times (int): List of times associated with above foci.
    """
    x_pts = []
    y_pts = []
    times = []
    cell: Cell
    for cell_id, cell in cells.items():
        # get conversion from cell to 'real' time
        for i, time in enumerate(cell.times):
            orientation = cell.orientations_rad[i]
            centroid = cell.centroids_px[i]
            x_locs = cell.disp_w[i]
            y_locs = cell.disp_l[i]

            data_x_locs = []
            data_y_locs = []
            for x, y in zip(x_locs, y_locs):
                # convert from cell to pixel space.
                if orientation < 0:
                    orientation = np.pi + orientation

                dy = y * np.sin(orientation) + x * np.cos(orientation)
                dx = -y * np.cos(orientation) + x * np.sin(orientation)

                xloc = dx + centroid[1]
                yloc = dy + centroid[0]
                data_x_locs.append(xloc)
                data_y_locs.append(yloc)

            x_pts.extend(data_x_locs)
            y_pts.extend(data_y_locs)
            times.extend(len(data_x_locs) * [time])

    return np.array(x_pts), np.array(y_pts), np.array(times)


def find_all_cell_intensities_helper(
    cells,
    intensity_directory,
    seg_directory,
    channel_name: str,
    apply_background_correction=False,
):
    fov_cells = organize_cells_by_channel_new(cells)
    # iterate over each fov in specs
    for fov_id, fov_peaks in fov_cells.items():
        print(f"Computing fluorescence for FOV {fov_id}")
        # iterate over each peak in fov
        for peak_id, peak_cells in fov_peaks.items():
            # Load fluorescent images and segmented images for this channel
            fl_stack_name = TIFF_FORMAT_PEAK % (
                "",
                fov_id,
                peak_id,
                channel_name,
            )
            fl_stack: np.ndarray = tiff.imread(
                intensity_directory / fl_stack_name
            )  # ty: ignore
            corrected_stack = np.zeros(fl_stack.shape)
            seg_stack_name = TIFF_FORMAT_PEAK % (
                "",
                fov_id,
                peak_id,
                "seg_otsu",
            )
            seg_stack: np.ndarray = tiff.imread(
                seg_directory / seg_stack_name
            )  # ty: ignore

            # evaluate whether each cell is in this fov/peak combination
            for _, cell in peak_cells.items():
                cell_times = cell.times
                cell_labels = cell.labels
                total_fluorescences = []

                # loop through cell's times
                for i, t in enumerate(cell_times):
                    frame = t - 1
                    cell_label = cell_labels[i]
                    cell_mask = seg_stack[frame, :, :] == cell_label

                    total_fluorescence = np.sum(fl_stack[frame, cell_mask])
                    total_fluorescences.append(total_fluorescence)
                total_fluorescences = np.array(total_fluorescences)
                if cell.total_fluorescence is None:
                    cell.total_fluorescence = {channel_name: total_fluorescences}
                else:
                    cell.total_fluorescence[channel_name] = total_fluorescences


def find_all_cell_intensities(
    cells,
    intensity_directory,
    seg_directory,
    channel_names: list[str] | str = "sub_c2",
    apply_background_correction=False,
):
    """
    Finds fluorescenct information for cells. All the cells in cells
    should be from one fov/peak.
    """
    if isinstance(channel_names, list):
        for cname in channel_names:
            find_all_cell_intensities_helper(
                cells,
                intensity_directory,
                seg_directory,
                cname,
                apply_background_correction=apply_background_correction,
            )
        return
    find_all_cell_intensities_helper(
        cells,
        intensity_directory,
        seg_directory,
        channel_names,
        apply_background_correction=apply_background_correction,
    )
    # The cell objects in the original dictionary will be updated,
    # no need to return anything specifically.


def find_cells_of_fov_and_peak(cells, fov_id, peak_id) -> Cells:
    """Return only cells from a specific fov/peak
    Parameters
    ----------
    fov_id : int corresponding to FOV
    peak_id : int correstonging to peak
    """

    fcells = {}  # f is for filtered

    for cell_id in cells:
        if cells[cell_id].fov == fov_id and cells[cell_id].peak == peak_id:
            fcells[cell_id] = cells[cell_id]

    return fcells


def find_last_daughter(cell, Cells):
    """Finds the last daughter in a lineage starting with a earlier cell.
    Helper function for find_continuous_lineages"""

    if cell.daughters[0] in Cells:
        cell = Cells[cell.daughters[0]]
        cell = find_last_daughter(cell, Cells)
        return cell

    return cell


def find_cells_born_after(cells, born_after=None):
    """
    Returns Cells dictionary of cells with a birth_time after the value specified
    """

    if born_after is None:
        return cells

    return {
        cell_id: cell
        for cell_id, cell in six.iteritems(cells)
        if cell.birth_time >= born_after
    }


def lineages_to_dict(lineages):
    """Converts the lineage structure of cells organized by peak back
    to a dictionary of cells. Useful for filtering but then using the
    dictionary based plotting functions"""

    Cells = {}

    for fov, peaks in six.iteritems(lineages):
        for peak, cells in six.iteritems(peaks):
            Cells.update(cells)

    return Cells


def find_continuous_lineages(cells, specs, t1=0, t2=1000):
    """
    Uses a recursive function to only return cells that have continuous
    lineages between two time points. Takes a "lineage" form of Cells and
    returns a dictionary of the same format. Good for plotting
    with saw_tooth_plot()
    t1 : int
        First cell in lineage must be born before this time point
    t2 : int
        Last cell in lineage must be born after this time point
    """

    Lineages = organize_cells_by_channel(cells, specs)

    # This is a mirror of the lineages dictionary, just for the continuous cells
    Continuous_Lineages = {}

    for fov, peaks in six.iteritems(Lineages):
        # print("fov = {:d}".format(fov))
        # Create a dictionary to hold this FOV
        Continuous_Lineages[fov] = {}

        for peak, cells in six.iteritems(peaks):
            # print("{:<4s}peak = {:d}".format("",peak))
            # sort the cells by time in a list for this peak
            cells_sorted = [(cell_id, cell) for cell_id, cell in six.iteritems(cells)]
            cells_sorted = sorted(cells_sorted, key=lambda x: x[1].birth_time)

            # Sometimes there are not any cells for the channel even if it was to be analyzed
            if not cells_sorted:
                continue

            # look through list to find the cell born immediately before t1
            # and divides after t1, but not after t2
            for i, cell_data in enumerate(cells_sorted):
                cell_id, cell = cell_data
                if cell.birth_time < t1 and t1 <= cell.division_time < t2:
                    first_cell_index = i
                    break

            # filter cell_sorted or skip if you got to the end of the list
            if i == len(cells_sorted) - 1:
                continue
            else:
                cells_sorted = cells_sorted[i:]

            # get the first cell and it's last contiguous daughter
            first_cell = cells_sorted[0][1]
            last_daughter = find_last_daughter(first_cell, cells)

            # check to the daughter makes the second cut off
            if last_daughter.birth_time > t2:
                # print(fov, peak, 'Made it')

                # now retrieve only those cells within the two times
                # use the function to easily return in dictionary format
                cells_cont = find_cells_born_after(cells, born_after=t1)
                # Cells_cont = find_cells_born_before(Cells_cont, born_before=t2)

                # append the first cell which was filtered out in the above step
                cells_cont[first_cell.id] = first_cell

                # and add it to the big dictionary
                Continuous_Lineages[fov][peak] = cells_cont

        # remove keys that do not have any lineages
        if not Continuous_Lineages[fov]:
            Continuous_Lineages.pop(fov)

    cells = lineages_to_dict(Continuous_Lineages)  # revert back to return

    return cells
