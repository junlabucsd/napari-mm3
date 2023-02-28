from __future__ import print_function

# import modules
import sys
import os
from pathlib import Path
import numpy as np
from napari.utils import progress

import pickle
import multiprocessing
from multiprocessing import Pool
from skimage.feature import blob_log  # used for foci finding
from scipy.optimize import leastsq  # fitting 2d gaussian


from .utils import organize_cells_by_channel

from ._deriving_widgets import (
    MM3Container,
    PlanePicker,
    FOVChooser,
    load_specs,
    load_time_table,
    information,
    load_stack_params,
)
from magicgui.widgets import SpinBox, ComboBox, FileEdit, FloatSpinBox, PushButton

# returnes a 2D gaussian function
def gaussian(height, center_x, center_y, width):
    """Returns a gaussian function with the given parameters. It is a circular gaussian.
    width is 2*sigma x or y
    """
    # return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
    return lambda x, y: height * np.exp(
        -(((center_x - x) / width) ** 2 + ((center_y - y) / width) ** 2) / 2
    )


# finds best fit for 2d gaussian using functin above
def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit
    if params are not provided, they are calculated from the moments
    params should be (height, x, y, width_x, width_y)"""
    gparams = moments(data)  # create guess parameters.
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = leastsq(errorfunction, gparams)
    return p


# moments of a 2D gaussian
def moments(data):
    """
    Returns (height, x, y, width_x, width_y)
    The (circular) gaussian parameters of a 2D distribution by calculating its moments.
    width_x and width_y are 2*sigma x and sigma y of the guassian.
    """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width = float(np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum()))
    row = data[int(x), :]
    # width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width


# find foci using a difference of gaussians method
def foci_analysis(
    fov_id, peak_id, Cells, params, seg_method, time_table, preview=False
):
    """Find foci in cells using a fluorescent image channel.
    This function works on a single peak and all the cells therein."""

    # make directory for foci debug
    # foci_dir = os.path.join(params['ana_dir'], 'overlay/')
    # if not os.path.exists(foci_dir):
    #     os.makedirs(foci_dir)

    # Import segmented and fluorescenct images
    image_data_seg = load_stack_params(
        params, fov_id, peak_id, postfix="seg_{}".format(seg_method)
    )

    image_data_FL = load_stack_params(
        params, fov_id, peak_id, postfix="sub_{}".format(params["foci_plane"])
    )

    # determine absolute time index
    times_all = []
    for fov, times in time_table.items():
        times_all = np.append(times_all, list(times.keys()))
    times_all = np.unique(times_all)
    times_all = np.sort(times_all)
    times_all = np.array(times_all, np.int_)
    t0 = times_all[0]  # first time index

    points = []
    radii = []
    times_p = []

    for cell_id, cell in progress(Cells.items()):

        information("Extracting foci information for %s." % (cell_id))
        # declare lists holding information about foci.
        disp_l = []
        disp_w = []
        foci_h = []
        # foci_stack = np.zeros((np.size(cell.times),
        #                        image_data_seg[0,:,:].shape[0], image_data_seg[0,:,:].shape[1]))

        # Go through each time point of this cell
        for t in cell.times:
            # retrieve this timepoint and images.
            image_data_temp = image_data_FL[t - t0, :, :]
            image_data_temp_seg = image_data_seg[t - t0, :, :]

            # find foci as long as there is information in the fluorescent image
            if np.sum(image_data_temp) != 0:
                if preview:
                    disp_l_tmp, disp_w_tmp, foci_h_tmp, points_t, radii_t = foci_lap(
                        image_data_temp_seg, image_data_temp, cell, t, params, preview
                    )
                    points.extend(points_t)
                    radii.extend(radii_t)
                    times_p.extend([t] * len(points_t))

                else:
                    disp_l_tmp, disp_w_tmp, foci_h_tmp = foci_lap(
                        image_data_temp_seg, image_data_temp, cell, t, params
                    )

                disp_l.append(disp_l_tmp)
                disp_w.append(disp_w_tmp)
                foci_h.append(foci_h_tmp)

            # if there is no information, append an empty list.
            # Should this be NaN?
            else:
                disp_l.append([])
                disp_w.append([])
                foci_h.append([])
                # foci_stack[i] = image_data_temp_seg

        # add information to the cell (will replace old data)
        cell.disp_l = disp_l
        cell.disp_w = disp_w
        cell.foci_h = foci_h

    return points, radii, times_p


# foci pool (for parallel analysis)
def foci_analysis_pool(fov_id, peak_id, Cells, params, seg_method, time_table):
    """Find foci in cells using a fluorescent image channel.
    This function works on a single peak and all the cells therein."""

    # Import segmented and fluorescenct images
    image_data_seg = load_stack_params(params, fov_id, peak_id, postfix=seg_method)
    image_data_FL = load_stack_params(
        params, fov_id, peak_id, postfix="sub_{}".format(params["foci_plane"])
    )

    # Load time table to determine first image index.
    times_all = np.array(np.sort(time_table[fov_id].keys()), np.int_)
    t0 = times_all[0]  # first time index
    tN = times_all[-1]  # last time index

    # call foci_cell for each cell object
    pool = Pool(processes=params["num_analyzers"])
    [
        pool.apply_async(
            foci_cell(cell_id, cell, t0, image_data_seg, image_data_FL, params)
        )
        for cell_id, cell in Cells.items()
    ]
    pool.close()
    pool.join()


# parralel function for each cell
def foci_cell(cell_id, cell, t0, image_data_seg, image_data_FL, params):
    """find foci in a cell, single instance to be called by the foci_analysis_pool for parallel processing."""
    disp_l = []
    disp_w = []
    foci_h = []

    # Go through each time point of this cell
    for t in cell.times:
        # retrieve this timepoint and images.
        image_data_temp = image_data_FL[t - t0, :, :]
        image_data_temp_seg = image_data_seg[t - t0, :, :]

        # find foci as long as there is information in the fluorescent image
        if np.sum(image_data_temp) != 0:
            disp_l_tmp, disp_w_tmp, foci_h_tmp = foci_lap(
                image_data_temp_seg, image_data_temp, cell, t, params
            )

            disp_l.append(disp_l_tmp)
            disp_w.append(disp_w_tmp)
            foci_h.append(foci_h_tmp)

        # if there is no information, append an empty list.
        # Should this be NaN?
        else:
            disp_l.append(np.nan)
            disp_w.append(np.nan)
            foci_h.append(np.nan)

    # add information to the cell (will replace old data)
    cell.disp_l = disp_l
    cell.disp_w = disp_w
    cell.foci_h = foci_h


# actual worker function for foci detection
def foci_lap(img, img_foci, cell, t, params, preview=False):
    """foci_lap finds foci using a laplacian convolution then fits a 2D
    Gaussian.

    The returned information are the parameters of this Gaussian.
    All the information is returned in the form of np.arrays which are the
    length of the number of found foci across all cells in the image.

    Parameters
    ----------
    img : 2D np.array
        phase contrast or bright field image. Only used for debug
    img_foci : 2D np.array
        fluorescent image with foci.
    cell : cell object
    t : int
        time point to which the images correspond

    Returns
    -------
    disp_l : 1D np.array
        displacement on long axis, in px, of a foci from the center of the cell
    disp_w : 1D np.array
        displacement on short axis, in px, of a foci from the center of the cell
    foci_h : 1D np.array
        Foci "height." Sum of the intensity of the gaussian fitting area.
    """

    # pull out useful information for just this time point
    i = cell.times.index(
        t
    )  # find position of the time point in lists (time points may be missing)
    bbox = cell.bboxes[i]
    orientation = cell.orientations[i]
    centroid = cell.centroids[i]
    region = cell.labels[i]

    # declare arrays which will hold foci data
    disp_l = []  # displacement in length of foci from cell center
    disp_w = []  # displacement in width of foci from cell center
    foci_h = []  # foci total amount (from raw image)

    # define parameters for foci finding
    minsig = params["foci_log_minsig"]
    maxsig = params["foci_log_maxsig"]
    thresh = params["foci_log_thresh"]
    peak_med_ratio = params["foci_log_peak_med_ratio"]

    # calculate median cell intensity. Used to filter foci
    img_foci_masked = np.copy(img_foci).astype(float)
    # correction for difference between segmentation image mask and fluorescence channel by padding on the rightmost column(s)
    if np.shape(img) != np.shape(img_foci_masked):
        delta_col = np.shape(img)[1] - np.shape(img_foci_masked)[1]
        img_foci_masked = np.pad(img_foci_masked, ((0, 0), (0, delta_col)), "edge")
    img_foci_masked[img != region] = np.nan
    cell_fl_median = np.nanmedian(img_foci_masked)
    cell_fl_mean = np.nanmean(img_foci_masked)

    img_foci_masked[img != region] = 0

    # find blobs using difference of gaussian
    over_lap = (
        0.95  # if two blobs overlap by more than this fraction, smaller blob is cut
    )
    numsig = (
        maxsig - minsig + 1
    )  # number of division to consider between min ang max sig
    blobs = blob_log(
        img_foci_masked,
        min_sigma=minsig,
        max_sigma=maxsig,
        overlap=over_lap,
        num_sigma=numsig,
        threshold=thresh,
    )

    # these will hold information about foci position temporarily
    x_blob, y_blob, r_blob = [], [], []
    x_gaus, y_gaus, w_gaus = [], [], []

    # loop through each potential foci
    for blob in blobs:
        yloc, xloc, sig = blob  # x location, y location, and sigma of gaus
        xloc = int(np.around(xloc))  # switch to int for slicing images
        yloc = int(np.around(yloc))
        radius = int(
            np.ceil(np.sqrt(2) * sig)
        )  # will be used to slice out area around foci

        # ensure blob is inside the bounding box
        # this might be better to check if (xloc, yloc) is in regions.coords
        if (
            yloc > np.int16(bbox[0])
            and yloc < np.int16(bbox[2])
            and xloc > np.int16(bbox[1])
            and xloc < np.int16(bbox[3])
        ):

            x_blob.append(xloc)  # for plotting
            y_blob.append(yloc)  # for plotting
            r_blob.append(radius)

            # cut out a small image from original image to fit gaussian
            gfit_area = img_foci[
                yloc - radius : yloc + radius, xloc - radius : xloc + radius
            ]
            # gfit_area_0 = img_foci[max(0, yloc-1*radius):min(img_foci.shape[0], yloc+1*radius),
            #                        max(0, xloc-1*radius):min(img_foci.shape[1], xloc+1*radius)]
            gfit_area_fixed = img_foci[
                yloc - maxsig : yloc + maxsig, xloc - maxsig : xloc + maxsig
            ]

            # fit gaussian to proposed foci in small box
            p = fitgaussian(gfit_area)
            (peak_fit, x_fit, y_fit, w_fit) = p

            # print('peak', peak_fit)
            if x_fit <= 0 or x_fit >= radius * 2 or y_fit <= 0 or y_fit >= radius * 2:
                continue
            elif peak_fit / cell_fl_median < peak_med_ratio:
                continue
            else:
                # find x and y position relative to the whole image (convert from small box)
                x_rel = int(xloc - radius + x_fit)
                y_rel = int(yloc - radius + y_fit)
                x_gaus = np.append(x_gaus, x_rel)  # for plotting
                y_gaus = np.append(y_gaus, y_rel)  # for plotting
                w_gaus = np.append(w_gaus, w_fit)  # for plotting

                # calculate distance of foci from middle of cell (scikit image)
                if orientation < 0:
                    orientation = np.pi + orientation
                disp_y = (y_rel - centroid[0]) * np.sin(orientation) - (
                    x_rel - centroid[1]
                ) * np.cos(orientation)
                disp_x = (y_rel - centroid[0]) * np.cos(orientation) + (
                    x_rel - centroid[1]
                ) * np.sin(orientation)

                # append foci information to the list
                disp_l = np.append(disp_l, disp_y)
                disp_w = np.append(disp_w, disp_x)
                foci_h = np.append(foci_h, np.sum(gfit_area_fixed))

    if preview:
        return disp_l, disp_w, foci_h, y_blob, r_blob

    return disp_l, disp_w, foci_h


def kymograph(fov_id, peak_id, params):

    sub_stack_fl = load_stack_params(
        params, fov_id, peak_id, postfix="sub_" + params["foci_plane"]
    )
    fl_proj = np.transpose(np.max(sub_stack_fl, axis=2))

    return fl_proj


def update_cell_foci(cells, foci):
    """Updates cells' .foci attribute in-place using information
    in foci dictionary
    """
    for focus_id, focus in foci.items():
        for cell in focus.cells:

            cell_id = cell.id
            cells[cell_id].foci[focus_id] = focus


def foci(params, fl_plane, seg_method, cell_file_path):

    with open(cell_file_path, "rb") as cell_file:
        Cells = pickle.load(cell_file)

    # load specs file
    specs = load_specs(params["ana_dir"])

    # load time table. Puts in params dictionary
    time_table = load_time_table(params["ana_dir"])

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    ### foci analysis
    information("Starting foci analysis.")

    # create dictionary which organizes cells by fov and peak_id
    Cells_by_peak = organize_cells_by_channel(Cells, specs)
    # for each set of cells in one fov/peak, find the foci
    for fov_id in fov_id_list:
        if not fov_id in Cells_by_peak:
            continue

        for peak_id, Cells_of_peak in Cells_by_peak[fov_id].items():
            if len(Cells_of_peak) == 0:
                continue
            information("running foci analysis")

            points, radii, times = foci_analysis(
                fov_id, peak_id, Cells_of_peak, params, seg_method, time_table
            )

    # Output data to both dictionary and the .mat format used by the GUI
    cell_filename = os.path.basename(cell_file_path)
    with open(
        os.path.join(params["cell_dir"], cell_filename[:-4] + "_foci.pkl"), "wb"
    ) as cell_file:
        pickle.dump(Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    information("Finished foci analysis.")


class Foci(MM3Container):
    def create_widgets(self):
        self.cellfile_widget = FileEdit(
            label="cell_file",
            value=Path("./analysis/cell_data/complete_cells.pkl"),
            tooltip="Cell file to be analyzed",
        )

        self.plane_widget = PlanePicker(
            self.valid_planes,
            label="analysis plane",
            tooltip="Fluoresence plane that you would like to analyze",
        )
        self.segmentation_method_widget = ComboBox(
            label="segmentation method", choices=["Otsu", "U-net"]
        )
        self.fov_widget = FOVChooser(self.valid_fovs)

        # minimum and maximum sigma of laplacian to convolve in pixels.
        # Scales with minimum foci width to detect as 2*sqrt(2)*minsig
        self.log_minsig_widget = SpinBox(
            value=2,
            label="LoG min sigma",
            tooltip="min sigma of laplacian to convolve in pixels",
            min=1,
        )
        self.log_maxsig_widget = SpinBox(
            value=3,
            label="LoG max sigma",
            tooltip="max sigma of laplacian to convolve in pixels",
            min=1,
        )
        # absolute threshold laplacian must reach to record potential foci. Keep low to detect dimmer spots.
        self.log_thresh_widget = FloatSpinBox(
            value=0.001,
            step=0.001,
            label="LoG threshold",
            tooltip="absolute threshold laplacian must reach to record potential foci",
        )  # default: 0.002;
        # foci peaks must be this many times greater than median cell intensity.
        # Think signal to noise ratio
        self.log_peak_ratio_widget = FloatSpinBox(
            value=1.5,
            step=0.1,
            label="SNR",
            tooltip="Minimum foci peak to background ratio for detection",
        )

        self.preview_widget = PushButton(label="generate preview", value=False)

        self.set_plane()
        self.set_segmentation_method()
        self.set_cellfile()
        self.set_fovs(self.valid_fovs)
        self.set_log_maxsig()
        self.set_log_minsig()
        self.set_log_thresh()
        self.set_log_peak_ratio()

        self.plane_widget.changed.connect(self.set_plane)
        self.segmentation_method_widget.changed.connect(self.set_segmentation_method)
        self.cellfile_widget.changed.connect(self.set_cellfile)
        self.fov_widget.connect_callback(self.set_fovs)
        self.log_minsig_widget.changed.connect(self.set_log_minsig)
        self.log_maxsig_widget.changed.connect(self.set_log_maxsig)
        self.log_thresh_widget.changed.connect(self.set_log_thresh)
        self.log_peak_ratio_widget.changed.connect(self.set_log_peak_ratio)

        self.viewer.window._status_bar._toggle_activity_dock(True)

        self.preview_widget.clicked.connect(self.render_preview)

        self.append(self.plane_widget)
        self.append(self.segmentation_method_widget)
        self.append(self.cellfile_widget)
        self.append(self.fov_widget)
        self.append(self.log_minsig_widget)
        self.append(self.log_maxsig_widget)
        self.append(self.log_thresh_widget)
        self.append(self.log_peak_ratio_widget)
        self.append(self.preview_widget)

        # self.render_preview()

    def set_params(self):
        # These have been wittled down to bare minimum.
        self.params = {
            "experiment_name": self.experiment_name,
            "ana_dir": self.analysis_folder,
            "FOV": self.fovs,
            "foci_plane": self.fl_plane,
            "cell_file": self.cellfile,
            "foci_log_minsig": self.log_minsig,
            "foci_log_maxsig": self.log_maxsig,
            "foci_log_thresh": self.log_thresh,
            "foci_log_peak_med_ratio": self.log_peak_ratio,
            "num_analyzers": multiprocessing.cpu_count(),
            "output": "TIFF",
            "chnl_dir": self.analysis_folder / "channels",
            "seg_dir": self.analysis_folder / "segmented",
            "cell_dir": self.analysis_folder / "cell_data",
            "sub_dir": self.analysis_folder / "subtracted",
        }

    def run(self):
        self.set_params()
        foci(self.params, self.fl_plane, self.segmentation_method, str(self.cellfile))

    def set_plane(self):
        self.fl_plane = self.plane_widget.value

    def set_segmentation_method(self):
        self.segmentation_method = self.segmentation_method_widget.value

    def set_cellfile(self):
        self.cellfile = self.cellfile_widget.value

    def set_fovs(self, fovs):
        self.fovs = list(set(fovs))

    def set_log_peak_ratio(self):
        self.log_peak_ratio = self.log_peak_ratio_widget.value

    def set_log_minsig(self):
        self.log_minsig = self.log_minsig_widget.value

    def set_log_maxsig(self):
        self.log_maxsig = self.log_maxsig_widget.value

    def set_log_thresh(self):
        self.log_thresh = self.log_thresh_widget.value

    def render_preview(self):
        self.viewer.layers.clear()
        self.set_params()
        # TODO: Add ability to change these to other FOVs
        valid_fov = self.valid_fovs[0]
        specs = load_specs(self.params["ana_dir"])
        # Find first cell-containing peak
        valid_peak = [key for key in specs[valid_fov] if specs[valid_fov][key] == 1][0]
        ## pull out first fov & peak id with cells

        kymo = kymograph(valid_fov, valid_peak, self.params)
        self.viewer.add_image(kymo)

        with open(self.cellfile, "rb") as cell_file:
            Cells = pickle.load(cell_file)

        time_table = load_time_table(self.params["ana_dir"])

        Cells_by_peak = organize_cells_by_channel(Cells, specs)[valid_fov][valid_peak]

        y_pos, radii, times = foci_analysis(
            valid_fov,
            valid_peak,
            Cells_by_peak,
            self.params,
            self.segmentation_method,
            time_table,
            preview=True,
        )

        points = np.stack((y_pos, times)).transpose()

        self.viewer.add_points(
            data=points,
            size=np.array(radii) / 5,
            face_color="orange",
            edge_color="white",
        )

        # usually want to stretch the image a bit to make it more legible
        self.viewer.layers[-1].scale = [1, 2]
        self.viewer.layers[-2].scale = [1, 2]
