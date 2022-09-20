from __future__ import print_function, division
import re
import datetime
import h5py
import numpy as np
import os

try:
    import cPickle as pickle
except:
    import pickle
import re
from scipy import ndimage as ndi
from skimage import filters, morphology
from skimage.filters import median

import sys
import time
import warnings
import yaml
import tifffile as tiff

import seaborn as sns

sns.set(style="ticks", color_codes=True)
sns.set_palette("deep")

### functions ###########################################################
# alert the user what is up

# print a warning
def warning(*objs):
    print(time.strftime("%H:%M:%S WARNING:", time.localtime()), *objs, file=sys.stderr)


def information(*objs):
    print(time.strftime("%H:%M:%S", time.localtime()), *objs, file=sys.stdout)


def get_plane(filepath):
    pattern = r"(c\d+).tif"
    res = re.search(pattern, filepath, re.IGNORECASE)
    if res != None:
        return res.group(1)
    else:
        return None


def get_fov(filepath):
    pattern = r"xy(\d+)\w*.tif"
    res = re.search(pattern, filepath, re.IGNORECASE)
    if res != None:
        return int(res.group(1))
    else:
        return None


def get_time(filepath):
    pattern = r"t(\d+)xy\w+.tif"
    res = re.search(pattern, filepath, re.IGNORECASE)
    if res != None:
        return np.int_(res.group(1))
    else:
        return None


# loads and image stack from TIFF or HDF5 using mm3 conventions
def load_stack(params, fov_id, peak_id, color="c1", image_return_number=None):
    """
    Loads an image stack.

    Supports reading TIFF stacks or HDF5 files.

    Parameters
    ----------
    fov_id : int
        The FOV id
    peak_id : int
        The peak (channel) id. Dummy None value incase color='empty'
    color : str
        The image stack type to return. Can be:
        c1 : phase stack
        cN : where n is an integer for arbitrary color channel
        sub : subtracted images
        seg : segmented images
        empty : get the empty channel for this fov, slightly different

    Returns
    -------
    image_stack : np.ndarray
        The image stack through time. Shape is (t, y, x)
    """

    # things are slightly different for empty channels
    if "empty" in color:
        if params["output"] == "TIFF":
            img_filename = params["experiment_name"] + "_xy%03d_%s.tif" % (
                fov_id,
                color,
            )

            with tiff.TiffFile(os.path.join(params["empty_dir"], img_filename)) as tif:
                img_stack = tif.asarray()

        if params["output"] == "HDF5":
            with h5py.File(
                os.path.join(params["hdf5_dir"], "xy%03d.hdf5" % fov_id), "r"
            ) as h5f:
                img_stack = h5f[color][:]

        return img_stack

    # load normal images for either TIFF or HDF5
    if params["output"] == "TIFF":
        if color[0] == "c":
            img_dir = params["chnl_dir"]
            img_filename = params["experiment_name"] + "_xy%03d_p%04d_%s.tif" % (
                fov_id,
                peak_id,
                color,
            )
        elif "sub" in color:
            img_dir = params["sub_dir"]
            img_filename = params["experiment_name"] + "_xy%03d_p%04d_%s.tif" % (
                fov_id,
                peak_id,
                color,
            )
        elif "foci" in color:
            img_dir = params["foci_seg_dir"]
            img_filename = params["experiment_name"] + "_xy%03d_p%04d_%s.tif" % (
                fov_id,
                peak_id,
                color,
            )
        elif "seg" in color:
            last = "seg_otsu"
            if "seg_img" in params.keys():
                last = params["seg_img"]
            if "track" in params.keys():
                last = params["track"]["seg_img"]

            img_dir = params["seg_dir"]
            img_filename = params["experiment_name"] + "_xy%03d_p%04d_%s.tif" % (
                fov_id,
                peak_id,
                last,
            )
        else:
            img_filename = params["experiment_name"] + "_xy%03d_p%04d_%s.tif" % (
                fov_id,
                peak_id,
                color,
            )

        with tiff.TiffFile(os.path.join(img_dir, img_filename)) as tif:
            img_stack = tif.asarray()

    if params["output"] == "HDF5":
        with h5py.File(
            os.path.join(params["hdf5_dir"], "xy%03d.hdf5" % fov_id), "r"
        ) as h5f:
            # normal naming
            # need to use [:] to get a copy, else it references the closed hdf5 dataset
            img_stack = h5f["channel_%04d/p%04d_%s" % (peak_id, peak_id, color)][:]

    return img_stack


# load the time table and add it to the global params
def load_time_table(ana_dir):
    """Add the time table dictionary to the params global dictionary.
    This is so it can be used during Cell creation.
    """

    # try first for yaml, then for pkl
    try:
        with open(os.path.join(ana_dir, "time_table.yaml"), "rb") as time_table_file:
            return yaml.safe_load(time_table_file)
    except:
        with open(os.path.join(ana_dir, "time_table.pkl"), "rb") as time_table_file:
            return pickle.load(time_table_file)


# function for loading the specs file
def load_specs(params):
    """Load specs file which indicates which channels should be analyzed, used as empties, or ignored."""

    try:
        with open(os.path.join(params["ana_dir"], "specs.yaml"), "r") as specs_file:
            specs = yaml.safe_load(specs_file)
    except:
        try:
            with open(os.path.join(params["ana_dir"], "specs.pkl"), "rb") as specs_file:
                specs = pickle.load(specs_file)
        except ValueError:
            warning("Could not load specs file.")

    return specs


### Cell class and related functions

# this is the object that holds all information for a cell
class Cell:
    """
    The Cell class is one cell that has been born. It is not neccesarily a cell that
    has divided.
    """

    # initialize (birth) the cell
    def __init__(self, params, time_table, cell_id, region, t, parent_id=None):
        """The cell must be given a unique cell_id and passed the region
        information from the segmentation

        Parameters
        __________

        cell_id : str
            cell_id is a string in the form fXpXtXrX
            f is 3 digit FOV number
            p is 4 digit peak number
            t is 4 digit time point at time of birth
            r is region label for that segmentation
            Use the function create_cell_id to do return a proper string.

        region : region properties object
            Information about the labeled region from
            skimage.measure.regionprops()

        parent_id : str
            id of the parent if there is one.
        """

        # create all the attributes
        # id
        self.id = cell_id

        self.params = params
        self.time_table = time_table

        # identification convenience
        self.fov = int(cell_id.split("f")[1].split("p")[0])
        self.peak = int(cell_id.split("p")[1].split("t")[0])
        self.birth_label = int(cell_id.split("r")[1])

        # parent id may be none
        self.parent = parent_id

        # daughters is updated when cell divides
        # if this is none then the cell did not divide
        self.daughters = None

        # birth and division time
        self.birth_time = t
        self.division_time = None  # filled out if cell divides

        # the following information is on a per timepoint basis
        self.times = [t]
        self.abs_times = [time_table[self.fov][t]]  # elapsed time in seconds
        self.labels = [region.label]
        self.bboxes = [region.bbox]
        self.areas = [region.area]

        # calculating cell length and width by using Feret Diamter. These values are in pixels
        length_tmp, width_tmp = feretdiameter(region)
        if length_tmp == None:
            warning("feretdiameter() failed for " + self.id + " at t=" + str(t) + ".")
        self.lengths = [length_tmp]
        self.widths = [width_tmp]

        # calculate cell volume as cylinder plus hemispherical ends (sphere). Unit is px^3
        self.volumes = [
            (length_tmp - width_tmp) * np.pi * (width_tmp / 2) ** 2
            + (4 / 3) * np.pi * (width_tmp / 2) ** 3
        ]

        # angle of the fit elipsoid and centroid location
        if region.orientation > 0:
            self.orientations = [-(np.pi / 2 - region.orientation)]
        else:
            self.orientations = [np.pi / 2 + region.orientation]

        self.centroids = [region.centroid]

        # these are special datatype, as they include information from the daugthers for division
        # computed upon division
        self.times_w_div = None
        self.lengths_w_div = None
        self.widths_w_div = None

        # this information is the "production" information that
        # we want to extract at the end. Some of this is for convenience.
        # This is only filled out if a cell divides.
        self.sb = None  # in um
        self.sd = None  # this should be combined lengths of daughters, in um
        self.delta = None
        self.tau = None
        self.elong_rate = None
        self.septum_position = None
        self.width = None
        self.death = None

    def grow(self, time_table, region, t):
        """Append data from a region to this cell.
        use cell.times[-1] to get most current value"""

        self.times.append(t)
        # TODO: Switch time_table to be passed in directly.
        self.abs_times.append(time_table[self.fov][t])
        self.labels.append(region.label)
        self.bboxes.append(region.bbox)
        self.areas.append(region.area)

        # calculating cell length and width by using Feret Diamter
        length_tmp, width_tmp = feretdiameter(region)
        if length_tmp == None:
            warning("feretdiameter() failed for " + self.id + " at t=" + str(t) + ".")
        self.lengths.append(length_tmp)
        self.widths.append(width_tmp)
        self.volumes.append(
            (length_tmp - width_tmp) * np.pi * (width_tmp / 2) ** 2
            + (4 / 3) * np.pi * (width_tmp / 2) ** 3
        )

        if region.orientation > 0:
            ori = -(np.pi / 2 - region.orientation)
        else:
            ori = np.pi / 2 + region.orientation

        self.orientations.append(ori)
        self.centroids.append(region.centroid)

    def die(self, region, t):
        """
        Annotate cell as dying from current t to next t.
        """
        self.death = t

    def divide(self, daughter1, daughter2, t):
        """Divide the cell and update stats.
        daugther1 and daugther2 are instances of the Cell class.
        daughter1 is the daugther closer to the closed end."""

        # put the daugther ids into the cell
        self.daughters = [daughter1.id, daughter2.id]

        # give this guy a division time
        self.division_time = daughter1.birth_time

        # update times
        self.times_w_div = self.times + [self.division_time]
        self.abs_times.append(self.time_table[self.fov][self.division_time])

        # flesh out the stats for this cell
        # size at birth
        self.sb = self.lengths[0] * self.params["pxl2um"]

        # force the division length to be the combined lengths of the daughters
        self.sd = (daughter1.lengths[0] + daughter2.lengths[0]) * self.params["pxl2um"]

        # delta is here for convenience
        self.delta = self.sd - self.sb

        # generation time. Use more accurate times and convert to minutes
        self.tau = np.float64((self.abs_times[-1] - self.abs_times[0]) / 60.0)

        # include the data points from the daughters
        self.lengths_w_div = [l * self.params["pxl2um"] for l in self.lengths] + [
            self.sd
        ]
        self.widths_w_div = [w * self.params["pxl2um"] for w in self.widths] + [
            ((daughter1.widths[0] + daughter2.widths[0]) / 2) * self.params["pxl2um"]
        ]

        # volumes for all timepoints, in um^3
        self.volumes_w_div = []
        for i in range(len(self.lengths_w_div)):
            self.volumes_w_div.append(
                (self.lengths_w_div[i] - self.widths_w_div[i])
                * np.pi
                * (self.widths_w_div[i] / 2) ** 2
                + (4 / 3) * np.pi * (self.widths_w_div[i] / 2) ** 3
            )

        # calculate elongation rate.

        try:
            times = np.float64((np.array(self.abs_times) - self.abs_times[0]) / 60.0)
            log_lengths = np.float64(np.log(self.lengths_w_div))
            p = np.polyfit(times, log_lengths, 1)  # this wants float64
            self.elong_rate = p[0] * 60.0  # convert to hours

        except:
            self.elong_rate = np.float64("NaN")
            warning("Elongation rate calculate failed for {}.".format(self.id))

        # calculate the septum position as a number between 0 and 1
        # which indicates the size of daughter closer to the closed end
        # compared to the total size
        self.septum_position = daughter1.lengths[0] / (
            daughter1.lengths[0] + daughter2.lengths[0]
        )

        # calculate single width over cell's life
        self.width = np.mean(self.widths_w_div)

        # convert data to smaller floats. No need for float64
        # see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
        convert_to = "float16"  # numpy datatype to convert to

        self.sb = self.sb.astype(convert_to)
        self.sd = self.sd.astype(convert_to)
        self.delta = self.delta.astype(convert_to)
        self.elong_rate = self.elong_rate.astype(convert_to)
        self.tau = self.tau.astype(convert_to)
        self.septum_position = self.septum_position.astype(convert_to)
        self.width = self.width.astype(convert_to)

        self.lengths = [length.astype(convert_to) for length in self.lengths]
        self.lengths_w_div = [
            length.astype(convert_to) for length in self.lengths_w_div
        ]
        self.widths = [width.astype(convert_to) for width in self.widths]
        self.widths_w_div = [width.astype(convert_to) for width in self.widths_w_div]
        self.volumes = [vol.astype(convert_to) for vol in self.volumes]
        self.volumes_w_div = [vol.astype(convert_to) for vol in self.volumes_w_div]
        # note the float16 is hardcoded here
        self.orientations = [
            np.float16(orientation) for orientation in self.orientations
        ]
        self.centroids = [
            (y.astype(convert_to), x.astype(convert_to)) for y, x in self.centroids
        ]

    def print_info(self):
        """prints information about the cell"""
        print("id = %s" % self.id)
        print("times = {}".format(", ".join("{}".format(t) for t in self.times)))
        print(
            "lengths = {}".format(", ".join("{:.2f}".format(l) for l in self.lengths))
        )


# obtains cell length and width of the cell using the feret diameter
def feretdiameter(region):
    """
    feretdiameter calculates the length and width of the binary region shape. The cell orientation
    from the ellipsoid is used to find the major and minor axis of the cell.
    See https://en.wikipedia.org/wiki/Feret_diameter.
    """

    # y: along vertical axis of the image; x: along horizontal axis of the image;
    # calculate the relative centroid in the bounding box (non-rotated)
    # print(region.centroid)
    y0, x0 = region.centroid
    y0 = y0 - np.int16(region.bbox[0]) + 1
    x0 = x0 - np.int16(region.bbox[1]) + 1

    ## orientation is now measured in RC coordinates - quick fix to convert
    ## back to xy
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


### functions for pruning a dictionary of cells
# find cells with both a mother and two daughters
def find_complete_cells(Cells):
    """Go through a dictionary of cells and return another dictionary
    that contains just those with a parent and daughters"""

    Complete_Cells = {}

    for cell_id in Cells:
        if Cells[cell_id].daughters and Cells[cell_id].parent:
            Complete_Cells[cell_id] = Cells[cell_id]

    return Complete_Cells


# finds cells whose birth label is 1
def find_mother_cells(Cells):
    """Return only cells whose starting region label is 1."""

    Mother_Cells = {}

    for cell_id in Cells:
        if Cells[cell_id].birth_label == 1:
            Mother_Cells[cell_id] = Cells[cell_id]

    return Mother_Cells


def filter_cells(Cells, attr, val, idx=None, debug=False):
    """Return only cells whose designated attribute equals "val"."""

    Filtered_Cells = {}

    for cell_id, cell in Cells.items():

        at_val = getattr(cell, attr)
        if debug:
            print(at_val)
            print("Times: ", cell.times)
        if idx is not None:
            at_val = at_val[idx]
        if at_val == val:
            Filtered_Cells[cell_id] = cell

    return Filtered_Cells


def filter_cells_containing_val_in_attr(Cells, attr, val):
    """Return only cells that have val in list attribute, attr."""

    Filtered_Cells = {}

    for cell_id, cell in Cells.items():

        at_list = getattr(cell, attr)
        if val in at_list:
            Filtered_Cells[cell_id] = cell

    return Filtered_Cells


def find_all_cell_intensities(
    Cells,
    params,
    specs,
    time_table,
    channel_name="sub_c2",
    apply_background_correction=True,
):
    """
    Finds fluorescenct information for cells. All the cells in Cells
    should be from one fov/peak.
    """

    # iterate over each fov in specs
    for fov_id, fov_peaks in specs.items():

        # iterate over each peak in fov
        for peak_id, peak_value in fov_peaks.items():

            # if peak_id's value is not 1, go to next peak
            if peak_value != 1:
                continue

            print(
                "Quantifying channel {} fluorescence in cells in fov {}, peak {}.".format(
                    channel_name, fov_id, peak_id
                )
            )
            # Load fluorescent images and segmented images for this channel
            fl_stack = load_stack(params, fov_id, peak_id, color=channel_name)
            corrected_stack = np.zeros(fl_stack.shape)

            for frame in range(fl_stack.shape[0]):
                # median filter will be applied to every image
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    median_filtered = median(
                        fl_stack[frame, ...], selem=morphology.disk(1)
                    )

                # subtract the gaussian-filtered image from true image to correct
                #   uneven background fluorescence
                if apply_background_correction:
                    blurred = filters.gaussian(
                        median_filtered, sigma=10, preserve_range=True
                    )
                    corrected_stack[frame, :, :] = median_filtered - blurred
                else:
                    corrected_stack[frame, :, :] = median_filtered

            seg_stack = load_stack(params, fov_id, peak_id, color="seg_unet")

            # evaluate whether each cell is in this fov/peak combination
            for cell_id, cell in Cells.items():

                cell_fov = cell.fov
                if cell_fov != fov_id:
                    continue

                cell_peak = cell.peak
                if cell_peak != peak_id:
                    continue

                cell_times = cell.times
                cell_labels = cell.labels
                cell.area_mean_fluorescence[channel_name] = []
                cell.volume_mean_fluorescence[channel_name] = []
                cell.total_fluorescence[channel_name] = []

                # loop through cell's times
                for i, t in enumerate(cell_times):
                    frame = t - 1
                    cell_label = cell_labels[i]

                    total_fluor = np.sum(
                        corrected_stack[frame, seg_stack[frame, :, :] == cell_label]
                    )

                    cell.area_mean_fluorescence[channel_name].append(
                        total_fluor / cell.areas[i]
                    )
                    cell.volume_mean_fluorescence[channel_name].append(
                        total_fluor / cell.volumes[i]
                    )
                    cell.total_fluorescence[channel_name].append(total_fluor)

    # The cell objects in the original dictionary will be updated,
    # no need to return anything specifically.
    return


def find_cells_of_fov_and_peak(Cells, fov_id, peak_id):
    """Return only cells from a specific fov/peak
    Parameters
    ----------
    fov_id : int corresponding to FOV
    peak_id : int correstonging to peak
    """

    fCells = {}  # f is for filtered

    for cell_id in Cells:
        if Cells[cell_id].fov == fov_id and Cells[cell_id].peak == peak_id:
            fCells[cell_id] = Cells[cell_id]

    return fCells


def find_cells_of_birth_label(Cells, label_num=1):
    """Return only cells whose starting region label is given.
    If no birth_label is given, returns the mother cells.
    label_num can also be a list to include cells of many birth labels
    """

    fCells = {}  # f is for filtered

    if type(label_num) is int:
        label_num = [label_num]

    for cell_id in Cells:
        if Cells[cell_id].birth_label in label_num:
            fCells[cell_id] = Cells[cell_id]

    return fCells
