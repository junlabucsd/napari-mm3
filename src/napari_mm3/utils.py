from __future__ import division, print_function

import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as sps
import seaborn as sns
import tifffile as tiff
import yaml
from matplotlib.axes import Axes
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
    pxl2um: float
    birth_label: int
    parent_id: str | None
    time_idxs: list[int]
    abs_times_s: list[float]
    birth_time_idx: int
    labels: list[int]
    bboxes_px: list[list[list[int]]]
    areas_px: list[int]
    orientations_rad: list[float]
    centroids_px: list[list[int]]
    lengths_px: list[float]
    widths_px: list[float]
    areas_px2: list[float]
    volumes_px3: list[float]

    # only for complete cells
    daughter_ids: None | list[str]
    division_time_idx: None | int
    abs_division_time_s: None | float
    elong_rate_per_hr: None | float
    sb_px: None | float
    sd_px: None | float
    delta: None | float
    septum_position_px: None | float
    division_length_px: None | float
    division_width_px: None | float

    complete: bool

    # only for fluorescent cells.
    total_fluorescence: dict[str, np.ndarray] | None = None

    # only for multiplexed experiments
    strain: str | None = None

    @property
    def times_w_div(self):
        return self.time_idxs + [self.division_time_idx]

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
    times = [t + 1 for t in times]

    def orient(region: RegionProperties):
        if region.orientation > 0:
            return -(np.pi / 2 - region.orientation)
        return np.pi / 2 + region.orientation

    length_and_widths = np.array([feretdiameter(r) for r in regions])
    lengths, widths = length_and_widths[:, 0], length_and_widths[:, 1]

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
        sd = daughter1.lengths_px[0] + daughter2.lengths_px[0]
        delta = sd - sb
        division_time = daughter1.birth_time_idx
        abs_division_time_s = time_table[fov][division_time]
        division_length_px = daughter1.lengths_px[0] + daughter2.lengths_px[0]
        division_width_px = (daughter1.widths_px[0] + daughter2.widths_px[0]) / 2
        septum_position = daughter1.lengths_px[0] / (
            daughter2.lengths_px[0] + daughter2.lengths_px[0]
        )

        elong_rate = None
        if len(times) > 1:
            # calculate the average growth rate
            try:
                rel_times_hr = (
                    (np.array(abs_times_s) - abs_times_s[0]) / 60.0 / 60.0
                ).astype(np.float64)
                log_lengths = (np.log(lengths * pxl2um + [division_length_px])).astype(
                    np.float64
                )

                p = np.polyfit(rel_times_hr, log_lengths, 1)  # this wants float64
                elong_rate = p[0]  # convert to hours
            except ValueError:
                print(f"Elongation rate calculate failed for {id}.")

    return Cell(
        id=cell_id,
        parent_id=parent_id,
        fov=fov,
        pxl2um=pxl2um,
        peak=int(cell_id.split("p")[1].split("t")[0]),
        birth_label=regions[0].label,
        time_idxs=times,
        abs_times_s=[time_table[fov][t] for t in times],
        birth_time_idx=times[0],
        labels=[r.label for r in regions],
        bboxes_px=[r.bbox for r in regions],
        areas_px=[r.area for r in regions],
        orientations_rad=[orient(r) for r in regions],
        centroids_px=[r.centroid for r in regions],
        lengths_px=lengths * pxl2um,
        widths_px=widths * pxl2um,
        areas_px2=areas_um2,
        volumes_px3=volumes_um3,
        daughter_ids=[daughter.id for daughter in daughters],
        # only for complete cells
        division_time_idx=division_time if complete else None,
        abs_division_time_s=abs_division_time_s if complete else None,
        elong_rate_per_hr=elong_rate if complete else None,
        delta=delta if complete else None,
        sb_px=sb if complete else None,
        sd_px=sd if complete else None,
        septum_position_px=septum_position if complete else None,
        division_length_px=division_length_px if complete else None,
        division_width_px=division_width_px if complete else None,
        complete=complete if complete else False,
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
    def default(self, obj):  # ty: ignore This is fine, actually, ty is just confused :D
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
        except KeyError:
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


# %% Random utility function for parsing range strings
def range_string_to_indices(range_string):
    """Convert a range string to a list of indices."""
    print(f"'{range_string}'")
    try:
        range_string = range_string.replace(" ", "")
        split = range_string.split(",")
        indices = []
        for items in split:
            # If it's a range
            if "-" in items:
                limits = list(map(int, items.split("-")))
                if len(limits) == 2:
                    # Make it an inclusive range, as users would expect
                    limits[1] += 1
                    indices += list(range(limits[0], limits[1]))
            # If it's a single item.
            else:
                indices += [int(items)]
        print("Index range string valid!")
        return indices
    except:  # noqa: E722
        raise ValueError(
            "Index range string invalid. Returning empty range until a new string is specified."
        )


# %% Convenience functions for interaction and foci..
def place_in_cell(cell: Cell, x, y, t):
    """Translates from screen-space to in-cell coordinates."""
    # check if our cell exists at the current timestamp.
    if t not in cell.time_idxs:
        return None, None, None

    cell_time = cell.time_idxs.index(t)
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
        disp_y, disp_x, cell_time = place_in_cell(cell, xloc, yloc, t)
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
        for i, time in enumerate(cell.time_idxs):
            orientation = cell.orientations_rad[i]
            centroid = cell.centroids_px[i]

            x_locs = cell.disp_w[i]  # ty: ignore Only exists after foci tracking
            y_locs = cell.disp_l[i]  # ty: ignore Only exists after foci tracking.

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


# %% Cell class and related functions


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
    except ValueError:
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
        return bool(cell.daughter_ids and cell.parent_id)

    return filter_cells(cells, is_complete_cell)


# finds cells whose birth label is 1
def find_mother_cells(cells) -> Cells:
    """Return only cells whose starting region label is 1."""

    def is_mother(cell):
        return cell.birth_label == 1

    return filter_cells(cells, is_mother)


def find_cells_born_after(cells: Cells, born_after):
    def f(cell: Cell):
        return cell.birth_time_idx > born_after

    return filter_cells(cells, f)


def find_cells_of_fov_and_peak(cells, fov_id, peak_id) -> Cells:
    def f(cell: Cell):
        return (cell.fov == fov_id) and (cell.peak == peak_id)

    return filter_cells(cells, f)


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


# %%
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
            fl_stack: np.ndarray = tiff.imread(intensity_directory / fl_stack_name)  # ty: ignore
            try:
                seg_stack_name = TIFF_FORMAT_PEAK % (
                    "",
                    fov_id,
                    peak_id,
                    "seg_unet",
                )
                seg_stack: np.ndarray = tiff.imread(seg_directory / seg_stack_name)  # ty: ignore
            except FileNotFoundError:
                seg_stack_name = TIFF_FORMAT_PEAK % (
                    "",
                    fov_id,
                    peak_id,
                    "seg_otsu",
                )
                seg_stack: np.ndarray = tiff.imread(seg_directory / seg_stack_name)  # ty: ignore

            for _, cell in peak_cells.items():
                cell_times = cell.time_idxs
                cell_labels = cell.labels
                total_fluorescences = []

                for i, t in enumerate(cell_times):
                    frame, cell_label = t - 1, cell_labels[i]
                    # can't do this via numpy because sometimes the cell_label changes (eg, if a higher cell divides)
                    cell_mask = seg_stack[frame, :, :] == cell_label

                    total_fluorescence = np.sum(fl_stack[frame, cell_mask])
                    total_fluorescences.append(total_fluorescence)

                total_fluorescences = np.array(total_fluorescences)

                if cell.total_fluorescence is None:
                    cell.total_fluorescence = {channel_name: total_fluorescences}
                else:
                    cell.total_fluorescence[channel_name] = total_fluorescences


def find_all_cell_intensities(
    cells: Cells,
    intensity_directory,
    seg_directory,
    channel_names: list[str] | str = "sub_c2",
    apply_background_correction=False,
):
    """
    Finds fluorescenct information for cells. All the cells in cells
    should be from one fov/peak.

    Cells dict will be modified and returned.
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

    return cells


def find_last_daughter(cell, cells: Cells):
    """Finds the last daughter in a lineage starting with a earlier cell.
    Helper function for find_continuous_lineages"""

    if cell.daughters[0] in cells:
        cell = cells[cell.daughters[0]]
        cell = find_last_daughter(cell, cells)
        return cell

    return cell


def lineages_to_dict(lineages):
    """Converts the lineage structure of cells organized by peak back
    to a dictionary of cells. Useful for filtering but then using the
    dictionary based plotting functions"""

    cells = {}

    for fov, peaks in lineages.items():
        for peak, cells in peaks.items():
            cells.update(cells)

    return cells


def find_continuous_lineages(cells: Cells, specs, t1=0, t2=1000):
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

    lineages = organize_cells_by_channel(cells, specs)

    # This is a mirror of the lineages dictionary, just for the continuous cells
    continuous_lineages = {}

    for fov, peaks in lineages.items():
        # Create a dictionary to hold this FOV
        continuous_lineages[fov] = {}

        for peak, cells in peaks.items():
            # sort the cells by time in a list for this peak
            cells_sorted = [(cell_id, cell) for cell_id, cell in cells.items()]
            cells_sorted = sorted(cells_sorted, key=lambda x: x[1].birth_time_idx)

            # Sometimes there are not any cells for the channel even if it was to be analyzed
            if not cells_sorted:
                continue

            # look through list to find the cell born immediately before t1
            # and divides after t1, but not after t2
            for i, cell_data in enumerate(cells_sorted):
                cell_id, cell = cell_data
                if cell.birth_time_idx < t1 and t1 <= cell.division_time_idx < t2:
                    # first_cell_index = i
                    break

            # filter cell_sorted or skip if you got to the end of the list
            if i == len(cells_sorted) - 1:
                continue
            else:
                cells_sorted = cells_sorted[i:]

            first_cell = cells_sorted[0][1]
            last_daughter = find_last_daughter(first_cell, cells)

            # check to the daughter makes the second cut off
            if last_daughter.birth_time > t2:
                # now retrieve only those cells within the two times
                # use the function to easily return in dictionary format
                cells_cont = find_cells_born_after(cells, born_after=t1)

                # append the first cell which was filtered out in the above step
                cells_cont[first_cell.id] = first_cell

                # and add it to the big dictionary
                continuous_lineages[fov][peak] = cells_cont

        # remove keys that do not have any lineages
        if not continuous_lineages[fov]:
            continuous_lineages.pop(fov)

    return continuous_lineages


def cells2df(
    cells_dict: dict[str, Cell],
    columns=[
        "fov",
        "peak",
        "birth_label",
        "birth_time_idx",
        "division_time_s",
        "sb_px",
        "sd_px",
        "width_px",
        "delta",
        "elong_rate_per_hr",
        "septum_position_px",
    ],
):
    """
    Take cell data (a dicionary of Cell objects) and return a dataframe.
    """

    # Make dataframe for plotting variables
    cells_df = pd.DataFrame(
        cells_dict, index="id"
    ).transpose()  # must be transposed so data is in columns
    cells_df = cells_df.sort_values(by=["fov", "peak", "birth_time", "birth_label"])
    cells_df = cells_df[columns].apply(pd.to_numeric)

    return cells_df


def cells2dict(cells: Cells):
    """
    Take a dictionary of Cells and returns a dictionary of dictionaries
    """

    cells_dict = {cell_id: vars(cell) for cell_id, cell in cells.items()}

    return cells_dict


### Filtering functions ############################################################################
def find_cells_of_fov(cells, FOVs: list | int = []):
    """Return only cells from certain FOVs.

    FOVs : int or list of ints
    """

    fCells = {}  # f is for filtered

    if isinstance(FOVs, int):
        FOVs = [FOVs]

    fCells = {
        cell_id: cell_tmp for cell_id, cell_tmp in cells.items() if cell_tmp.fov in FOVs
    }

    return fCells


def find_cells_born_before(cells, born_before=None):
    """
    Returns Cells dictionary of cells with a birth_time before the value specified
    """

    if born_before is None:
        return cells

    fCells = {
        cell_id: Cell
        for cell_id, Cell in cells.items()
        if Cell.birth_time <= born_before
    }

    return fCells


def filter_by_stat(cells, center_stat="mean", std_distance=3):
    """
    Filters a dictionary of Cells by ensuring all of the 6 major parameters are
    within some number of standard deviations away from either the mean or median
    """

    # Calculate stats.
    Cells_df = cells2df(cells2dict(cells))
    stats_columns = ["sb", "sd", "delta", "elong_rate", "tau", "septum_position"]
    cell_stats = Cells_df[stats_columns].describe()

    # set low and high bounds for each stat attribute
    bounds = {}
    for label in stats_columns:
        low_bound = (
            cell_stats[label][center_stat] - std_distance * cell_stats[label]["std"]
        )
        high_bound = (
            cell_stats[label][center_stat] + std_distance * cell_stats[label]["std"]
        )
        bounds[label] = {"low": low_bound, "high": high_bound}

    # add filtered cells to dict
    fCells = {}  # dict to hold filtered cells

    for cell_id, Cell in cells.items():
        benchmark = 0  # this needs to equal 6, so it passes all tests

        for label in stats_columns:
            attribute = getattr(Cell, label)  # current value of this attribute for cell
            if attribute > bounds[label]["low"] and attribute < bounds[label]["high"]:
                benchmark += 1

        if benchmark == 6:
            fCells[cell_id] = cells[cell_id]

    return fCells


def binned_stat(x, y, statistic="mean", bin_edges="sturges", binmin=None):
    """Calculate binned mean or median on X. Returns plotting variables

    bin_edges : int or list/array
        If int, this is the number of bins. If it is a list it defines the bin edges.

    """

    # define range for bins
    data_mean = x.mean()
    data_std = x.std()
    bin_range = (data_mean - 3 * data_std, data_mean + 3 * data_std)

    # gives better bin edges. If a defined sequence is passed it will use that.
    bin_edges = np.histogram_bin_edges(x, bins=bin_edges, range=bin_range)

    # calculate mean
    bin_result = sps.binned_statistic(x, y, statistic=statistic, bins=bin_edges)
    bin_means, bin_edges, bin_n = bin_result
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    # calculate error at each bin (standard error)
    bin_error_result = sps.binned_statistic(x, y, statistic=np.std, bins=bin_edges)
    bin_stds, _, _ = bin_error_result

    # if using median, multiply this number by 1.253. Holds for large samples only
    if statistic == "median":
        bin_stds = bin_stds * 1.253

    bin_count_results = sps.binned_statistic(x, y, statistic="count", bins=bin_edges)
    bin_counts, _, _ = bin_count_results

    bin_errors = np.divide(bin_stds, np.sqrt(bin_counts))

    # remove bins with not enought datapoints
    if binmin:
        delete_me = []
        for i, points in enumerate(bin_counts):
            if points < binmin:
                delete_me.append(i)
        delete_me = tuple(delete_me)
        bin_centers = np.delete(bin_centers, delete_me)
        bin_means = np.delete(bin_means, delete_me)
        bin_errors = np.delete(bin_errors, delete_me)

        # only keep locations where there is data
        bin_centers = bin_centers[~np.isnan(bin_means)]
        bin_means = bin_means[~np.isnan(bin_means)]
        bin_errors = bin_errors[~np.isnan(bin_means)]

    return bin_centers, bin_means, bin_errors


# %% Plotting functions
def plot_against_birth_time(cells: dict[str, Cell], fn):
    t_to_v = {}
    for cell_id, cell in cells.items():
        stat = fn(cell)
        if stat is None:
            continue
        if cell.birth_time_idx not in t_to_v:
            t_to_v[cell.birth_time_idx] = [stat]
            continue
        t_to_v[cell.birth_time_idx].append(stat)

    xs = np.array(sorted(t for t in t_to_v))
    ys = [np.nanmean(t_to_v[a]) for a in xs]

    return xs, np.array(ys)


def adder_plot(cells: dict[str, Cell], crop: slice = None):
    """
    A common control plot.
    Plots size at birth, size at division, and the difference between the two.
    For good, physiological data, all three should be constant over the relevant range.
    """
    fig, ax = plt.subplots(
        nrows=3, figsize=(6, 6), sharex="col", layout="constrained", sharey="col"
    )

    fig.suptitle("Cell length follows adder.", fontweight="semibold")

    xs, ys = plot_against_birth_time(
        cells,
        lambda cell: cell.division_length_px,
    )
    ax[0].scatter(xs[crop], ys[crop], marker=".")
    ax[0].set_ylim(bottom=0)
    ax[0].set_ylabel("$S_d$")

    xs, ys = plot_against_birth_time(
        cells,
        lambda cell: cell.lengths_px[0],
    )
    ax[1].scatter(xs[crop], ys[crop], marker=".")
    ax[1].set_ylim(bottom=0)
    ax[1].set_ylabel("$S_b$")

    xs, ys = plot_against_birth_time(
        cells,
        lambda cell: (
            (cell.division_length_px - cell.lengths_px[0])
            if cell.division_length_px
            else None
        ),
    )
    ax[2].scatter(xs[crop], ys[crop], marker=".")
    ax[2].set_ylim(bottom=0)
    ax[2].set_ylabel("$S_b  - S_d$")
    ax[-1].set_xlabel("Birth time")

    plt.show()


def plot_channel_traces(
    Cells,
    time_int=1.0,
    fl_plane="c2",
    alt_time="birth",
    fl_int=1.0,
    plot_fl=False,
    plot_foci=False,
    plot_pole=False,
    pxl2um=1.0,
    xlims=None,
    foci_size=100,
):
    """Plot a cell lineage with profile information. Plots cells at their Y location in the growth channel.

    Parameters
    ----------
    Cells : dict of Cell objects
        All the cells should come from a single peak.
    time_int : int or float
        Used to adjust the X axis to plot in hours
    alt_time : float or 'birth'
        Adjusts all time by this value. 'birth' adjust the time so first birth time is at zero.
    fl_plane : str
        Plane from which to get florescent data
    plot_fl : boolean
        Flag to plot florescent line profile.
    plot_foci : boolean
        Flag to plot foci or not.
    plot_pole : boolean
        If true, plot different colors for cells with different pole ages.
    plx2um : float
        Conversion factor between pixels and microns.
    xlims : [float, float]
        Manually set xlims. If None then set automatically.
    """

    time_int = float(time_int)
    fl_int = float(fl_int)

    color = "b"  # overwritten if plot_pole == True

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(8, 3))
    ax = [axes]

    # turn it into a list to fidn first time
    lin = [(cell_id, cell) for cell_id, cell in Cells.items()]
    lin = sorted(lin, key=lambda x: x[1].birth_time)

    # align time to first birth or shift time
    if alt_time is None:
        alt_time = 0
    elif alt_time == "birth":
        alt_time = lin[0][1].birth_time * time_int / 60.0

    # determine last time for xlims
    if (xlims is None) or (xlims[1] is None):
        if alt_time == "birth" or alt_time == 0:
            first_time = 0
        else:  # adjust for negative birth times
            first_time = (lin[0][1].times[0] - 10) * time_int / 60.0 - alt_time
        last_time = (lin[-1][1].times[-1] + 10) * time_int / 60.0 - alt_time
        xlims = (first_time, last_time)

    # adjust scatter marker size so colors touch but do not overlap
    # uses size of figure in inches, with the dpi (ppi) to convert to points.
    # scatter marker size is points squared.
    bbox = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = float(bbox.width)
    # print(fig.dpi, width, xlims[1], xlims[0],  time_int)
    # print(((fig.dpi * width) / ((xlims[1] - xlims[0]) * 60.0 / time_int)))
    # print((fig.dpi * width) / ((xlims[1] - xlims[0]) * 60.0))
    # print((fig.dpi * width) / ((xlims[1] - xlims[0]) * 60.0 / time_int)**2)
    scat_s = (
        (fig.dpi * width) / ((xlims[1] - xlims[0]) * 60.0 / time_int) * fl_int
    ) ** 2
    # print(time_int)
    # print(scat_s)

    # Choose colormap. Need to add alpha to color map and normalization
    # green/c2
    if plot_fl:
        max_c2_int = 0
        min_c2_int = float("inf")
        for cell_id, cell in lin:
            for profile_t in getattr(cell, "fl_profiles_" + fl_plane):
                if max(profile_t) > max_c2_int:
                    max_c2_int = max(profile_t)
                if min(profile_t) < min_c2_int:
                    min_c2_int = min(profile_t)
        cmap_c2 = plt.cm.Greens  # ty: ignore
        color_norm_c2 = mpl.colors.Normalize(vmin=min_c2_int, vmax=max_c2_int)

    for cell_id, cell in Cells.items():
        # if this is a complete cell plot till division with a line at the end
        cell_times = np.array(cell.times) * time_int / 60.0 - alt_time
        cell_yposs = np.array([y for y, x in cell.centroids]) * pxl2um
        cell_halflengths = np.array(cell.lengths) / 2.0 * pxl2um
        ytop = cell_yposs + cell_halflengths
        ybot = cell_yposs - cell_halflengths

        if plot_pole:
            if cell.poleage:
                color_choices = sns.hls_palette(4)
                if cell.poleage == (1000, 0):
                    color = color_choices[0]
                elif cell.poleage == (0, 1) and cell.birth_label <= 2:
                    color = color_choices[1]
                elif cell.poleage == (1, 0) and cell.birth_label <= 3:
                    color = color_choices[2]
                elif cell.poleage == (0, 2):
                    color = color_choices[3]
                # elif cell.poleage == (2, 0):
                #     color = color_choices[4]
                else:
                    color = "k"
            elif cell.poleage is None:
                color = "k"

        # plot two lines for top and bottom of cell
        ax[0].plot(cell_times, ybot, cell_times, ytop, color=color, alpha=0.75, lw=1)
        # ax[0].fill_between(cell_times, ybot, ytop,
        #                    color=color, lw=0.5, alpha=1)

        # plot lines for birth and division
        ax[0].plot(
            [cell_times[0], cell_times[0]],
            [ybot[0], ytop[0]],
            color=color,
            alpha=0.75,
            lw=1,
        )
        ax[0].plot(
            [cell_times[-1], cell_times[-1]],
            [ybot[-1], ytop[-1]],
            color=color,
            alpha=0.75,
            lw=1,
        )

        # plot fluorescence line profile
        if plot_fl:
            for i, t in enumerate(cell_times):
                if cell.times[i] % fl_int == 1:
                    fl_x = (
                        np.ones(len(getattr(cell, "fl_profiles_" + fl_plane)[i])) * t
                    )  # times
                    fl_ymin = cell_yposs[i] - (
                        len(getattr(cell, "fl_profiles_" + fl_plane)[i]) / 2 * pxl2um
                    )
                    fl_ymax = fl_ymin + (
                        len(getattr(cell, "fl_profiles_" + fl_plane)[i]) * pxl2um
                    )
                    fl_y = np.linspace(
                        fl_ymin,
                        fl_ymax,
                        len(getattr(cell, "fl_profiles_" + fl_plane)[i]),
                    )
                    fl_z = getattr(cell, "fl_profiles_" + fl_plane)[i]
                    ax[0].scatter(
                        fl_x,
                        fl_y,
                        c=fl_z,
                        cmap=cmap_c2,
                        marker="s",
                        s=scat_s,
                        norm=color_norm_c2,
                        rasterized=True,
                    )

        # plot foci
        if plot_foci:
            for i, t in enumerate(cell_times):
                if cell.times[i] % fl_int == 1:
                    for j, foci_y in enumerate(cell.disp_l[i]):
                        foci_y_pos = cell_yposs[i] + (foci_y * pxl2um)
                        ax[0].scatter(
                            t,
                            foci_y_pos,
                            s=cell.foci_h[i][j] / foci_size,
                            linewidth=0.5,
                            edgecolors="k",
                            facecolors="none",
                            alpha=0.5,
                            rasterized=False,
                        )

    ax[0].set_xlabel("time (hours)")
    ax[0].set_xlim(xlims)
    #     ax[0].set_ylabel('position ' + pnames['um'])
    ax[0].set_ylim(bottom=0)
    #     ax[0].set_yticklabels([0,2,4,6,8,10])
    sns.despine()
    plt.tight_layout()


def plot_moving_avg(df, time_mark, column, window, ax, label=None):
    time_df = df[[time_mark, column]].apply(pd.to_numeric)
    xlims = (time_df[time_mark].min(), time_df[time_mark].max())  # x lims for bins
    # xlims = x_extents
    bin_mean, bin_edges, bin_n = sps.binned_statistic(
        time_df[time_mark],
        time_df[column],
        statistic="mean",
        bins=np.arange(xlims[0] - 1, xlims[1] + 1, window),
    )
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    ax.plot(bin_centers, bin_mean, lw=1, alpha=1, label=label)


def line_hist(data, bins=None, density=True):
    if bins is None:
        bin_vals, bin_edges = np.histogram(data, density=density)
    else:
        bin_vals, bin_edges = np.histogram(data, density=density, bins=bins)
    bin_steps = np.diff(bin_edges) / 2.0
    bin_centers = bin_edges[:-1] + bin_steps
    # add zeros to the next points outside this so plot line always goes down
    bin_centers = np.insert(bin_centers, 0, bin_centers[0] - bin_steps[0])
    bin_centers = np.append(bin_centers, bin_centers[-1] + bin_steps[-1])
    bin_vals = np.insert(bin_vals, 0, 0)
    bin_vals = np.append(bin_vals, 0)
    return (bin_centers, bin_vals)


def plot_line_hist(
    data,
    bins=None,
    density=True,
    ax: Axes = None,
    skipna=False,
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots()

    if skipna:
        data = np.array(data)
        data = data[data != np.array(None)]

    bin_centers, bin_vals = line_hist(data, bins=bins, density=density)

    ax.plot(bin_centers, bin_vals, **kwargs)
    ax.set_ylim(0, np.max(bin_vals) * 1.3)
    ax.spines["left"].set_visible(False)
    if density:
        ax.set_yticks([])

    return bin_centers, bin_vals


def plot_distributions(df, columns, labels=None, titles=None):
    """
    Plot distributions of cell cycle parameters

    Parameters
    ----------
    df : pandas dataframe
        dataframe containing cell cycle parameters to plot
    columns : list
        list of column names to plot
    labels : list
        list of labels for each column
    titles : list
        list of titles for each column

    Returns
    -------
    fig : matplotlib figure
        figure containing plots
    axes : matplotlib axes
        axes containing plots
    """

    fig, axes = plt.subplots(1, 6, figsize=(12, 3))
    ax = np.ravel(axes)

    if not labels:
        labels = [
            "Birth length ($\\mu$M)",
            "Division length ($\\mu$M)",
            "$\\Delta$ ($\\mu$M)",
            "Elongation rate (1/hr)",
            "$\\tau$ (minutes)",
            "Septum position",
        ]

    titles = ["S$_{B}$", "S$_{D}$", "$\\Delta$", "$\\lambda$", "$\\tau$", "L$_{1/2}$"]

    for i, c in enumerate(columns):
        mu1 = df[c].mean()
        cv1 = df[c].std() / df[c].mean()

        ax[i].set_title(titles[i], fontsize=14)
        plot_line_hist(
            df[c], color="C0", lw=1, label=f"$\\mu$ = {mu1:2.2f}\nCV = {cv1:2.2f}"
        )
        ax[i].set_xlabel(labels[i], fontsize=12)

    plt.tight_layout()


def plot_hex_time(Cells_df, time_mark="birth_time", x_extents=None, bin_extents=None):
    """
    Plots cell parameters over time using a hex scatter plot and a moving average
    """

    # lists for plotting and formatting
    columns = ["sb", "elong_rate", "sd", "tau", "delta", "septum_position"]
    titles = [
        "Length at Birth",
        "Elongation Rate",
        "Length at Division",
        "Generation Time",
        "Delta",
        "Septum Position",
    ]
    ylabels = ["$\\mu$m", "$\\lambda$", "$\\mu$m", "min", "$\\mu$m", "daughter/mother"]

    # create figure, going to apply graphs to each axis sequentially
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=[8, 8], squeeze=False)

    ax = np.ravel(axes)
    # binning parameters, should be arguments
    binmin = 3  # minimum bin size to display
    bingrid = (20, 10)  # how many bins to have in the x and y directions
    moving_window = 10  # window to calculate moving stat

    # bining parameters for each data type
    # bin_extent in within which bounds should bins go. (left, right, bottom, top)
    if x_extents is None:
        x_extents = (Cells_df["birth_time"].min(), Cells_df["birth_time"].max())

    if bin_extents is None:
        bin_extents = [
            (x_extents[0], x_extents[1], 0, 4),
            (x_extents[0], x_extents[1], 0, 1.5),
            (x_extents[0], x_extents[1], 0, 8),
            (x_extents[0], x_extents[1], 0, 140),
            (x_extents[0], x_extents[1], 0, 4),
            (x_extents[0], x_extents[1], 0, 1),
            (x_extents[0], x_extents[1], 0, 100),
            (x_extents[0], x_extents[1], 0, 80),
            (x_extents[0], x_extents[1], 0, 2),
        ]

    # Now plot the filtered data
    for i, column in enumerate(columns):
        # get out just the data to be plot for one subplot
        time_df = Cells_df[[time_mark, column]].apply(pd.to_numeric)
        time_df.sort_values(by=time_mark, inplace=True)

        # plot the hex scatter plot
        p = ax[i].hexbin(
            time_df[time_mark], time_df[column], mincnt=binmin, gridsize=bingrid
        )

        # graph moving average
        # xlims = (time_df['birth_time'].min(), time_df['birth_time'].max()) # x lims for bins
        xlims = x_extents
        try:
            bin_mean, bin_edges, bin_n = sps.binned_statistic(
                time_df[time_mark],
                time_df[column],
                statistic="mean",
                bins=np.arange(xlims[0] - 1, xlims[1] + 1, moving_window),
            )
            bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
            ax[i].plot(bin_centers, bin_mean, lw=4, alpha=0.8, color="yellow")

        except:
            pass

        # formatting
        ax[i].set_title(titles[i])
        ax[i].set_ylabel(ylabels[i])

        p.set_cmap(cmap=plt.cm.Blues)  # ty: ignore

    ax[5].set_xlabel("%s [frame]" % time_mark)
    ax[4].set_xlabel("%s [frame]" % time_mark)

    plt.tight_layout()

    return fig, ax
