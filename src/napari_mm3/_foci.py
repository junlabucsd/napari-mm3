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
from napari import Viewer


from .utils import organize_cells_by_channel, write_cells_to_json

from ._deriving_widgets import (
    MM3Container,
    PlanePicker,
    FOVChooser,
    SegmentationMode,
    load_specs,
    load_time_table,
    information,
    load_seg_stack,
    load_subtracted_stack,
    InteractivePeakChooser,
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
def foci_analysis(fov_id, peak_id, Cells, ana_dir, params, seg_method, time_table):
    """Find foci in cells using a fluorescent image channel.
    This function works on a single peak and all the cells therein.

    Parameters
    ----------
    fov_id : int
        The field of view to be analyzed.
    peak_id : int
        The peak to be analyzed.
    Cells : dict
        A dictionary of cells to be analyzed.
    params : dict
        A dictionary of parameters.
    seg_method : str
        The segmentation method used (Otsu or U-Net).
    time_table : dict
        A dictionary of time points.
    preview : bool, optional
        Set to true if we are displaying the results in the napari viewer. The default is False.

    Returns
    -------
    points : list
        A list of foci locations (used only for display).
    radii : list
        A list of radii for foci visualization (proportional to intensity). Used for display.
    times_p : list
        A list of times for detected foci. Used for display.

    """
    # Import segmented and fluorescenct images
    seg_mode = SegmentationMode.OTSU if seg_method == "Otsu" else SegmentationMode.UNET
    image_data_seg = load_seg_stack(
        ana_dir=ana_dir,
        experiment_name=params["experiment_name"],
        fov_id=fov_id,
        peak_id=peak_id,
        seg_mode=seg_mode,
    )

    postfix = f'sub_{params["foci_plane"]}'
    image_data_FL = load_subtracted_stack(
        ana_dir, params["experiment_name"], fov_id, peak_id, postfix
    )

    # determine absolute time index
    times_all = []
    for _, times in time_table.items():
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
        # Go through each time point of this cell
        for t in cell.times:
            # retrieve this timepoint and images.
            image_data_temp = image_data_FL[t - t0, :, :]
            image_data_temp_seg = image_data_seg[t - t0, :, :]

            # find foci as long as there is information in the fluorescent image
            if np.sum(image_data_temp) != 0:
                minsig = params["foci_log_minsig"]
                maxsig = params["foci_log_maxsig"]
                thresh = params["foci_log_thresh"]
                peak_med_ratio = params["foci_log_peak_med_ratio"]

                disp_l_tmp, disp_w_tmp, foci_h_tmp = foci_lap(
                    image_data_temp_seg,
                    image_data_temp,
                    cell,
                    t,
                    minsig,
                    maxsig,
                    thresh,
                    peak_med_ratio,
                    preview=False,
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


# find foci using a difference of gaussians method.
# the idea of this one is to be run on a single preview
def foci_preview(fov_id, peak_id, Cells, ana_dir, params, seg_method, time_table):
    """Find foci in cells using a fluorescent image channel.
    This function works on a single peak and all the cells therein.

    Parameters
    ----------
    fov_id : int
        The field of view to be analyzed.
    peak_id : int
        The peak to be analyzed.
    Cells : dict
        A dictionary of cells to be analyzed.
    params : dict
        A dictionary of parameters.
    seg_method : str
        The segmentation method used (Otsu or U-Net).
    time_table : dict
        A dictionary of time points.
    preview : bool, optional
        Set to true if we are displaying the results in the napari viewer. The default is False.

    Returns
    -------
    points : list
        A list of foci locations (used only for display).
    radii : list
        A list of radii for foci visualization (proportional to intensity). Used for display.
    times_p : list
        A list of times for detected foci. Used for display.

    """
    # Import segmented and fluorescent images
    seg_mode = SegmentationMode.OTSU if seg_method == "Otsu" else SegmentationMode.UNET
    image_data_seg = load_seg_stack(
        ana_dir=ana_dir,
        experiment_name=params["experiment_name"],
        fov_id=fov_id,
        peak_id=peak_id,
        seg_mode=seg_mode,
    )

    postfix = f'sub_{params["foci_plane"]}'
    image_data_FL = load_subtracted_stack(
        ana_dir, params["experiment_name"], fov_id, peak_id, postfix
    )

    # determine absolute time index
    times_all = []
    for fov, times in time_table.items():
        times_all = np.append(times_all, list(times.keys()))
    times_all = np.unique(times_all)
    times_all = np.sort(times_all)
    times_all = np.array(times_all, np.int_)
    t0 = times_all[0]  # first time index

    x_blob = []
    y_blob = []
    radii = []
    times_p = []

    for cell_id, cell in progress(Cells.items()):

        information("Extracting foci information for %s." % (cell_id))
        # declare lists holding information about foci.
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
                minsig = params["foci_log_minsig"]
                maxsig = params["foci_log_maxsig"]
                thresh = params["foci_log_thresh"]
                peak_med_ratio = params["foci_log_peak_med_ratio"]

                (
                    disp_l_tmp,
                    disp_w_tmp,
                    foci_h_tmp,
                    x_blob_tmp,
                    y_blob_tmp,
                    r_blob_tmp,
                ) = foci_lap(
                    image_data_temp_seg,
                    image_data_temp,
                    cell,
                    t,
                    minsig,
                    maxsig,
                    thresh,
                    peak_med_ratio,
                    preview=True,
                )

                x_blob.extend(x_blob_tmp)
                y_blob.extend(y_blob_tmp)
                radii.extend(r_blob_tmp)
                times_p.extend([t] * len(x_blob_tmp))
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

    return x_blob, y_blob, radii, times_p


# foci pool (for parallel analysis)
def foci_analysis_pool(fov_id, peak_id, Cells, ana_dir, params, seg_method, time_table):
    """Find foci in cells using a fluorescent image channel.
    This function works on a single peak and all the cells therein.
    Parameters
    ----------
    fov_id : int
        Field of view ID.
    peak_id : int
        Peak ID.
    Cells : dict
        Dictionary of cell objects.
    params : dict
        Dictionary of parameters.
    seg_method : str
        Segmentation method.
    time_table : dict
        Dictionary of time points.
    Returns
    -------
    None"""

    # Import segmented and fluorescenct images
    seg_mode = SegmentationMode.OTSU if seg_method == "Otsu" else SegmentationMode.UNET
    image_data_seg = load_seg_stack(
        ana_dir=ana_dir,
        experiment_name=params["experiment_name"],
        fov_id=fov_id,
        peak_id=peak_id,
        seg_mode=seg_mode,
    )
    postfix = f'sub_{params["foci_plane"]}'
    image_data_FL = load_subtracted_stack(
        ana_dir, params["experiment_name"], fov_id, peak_id, postfix
    )

    # Load time table to determine first image index.
    times_all = np.array(np.sort(time_table[fov_id].keys()), np.int_)
    t0 = times_all[0]  # first time index
    tN = times_all[-1]  # last time index

    # call foci_cell for each cell object
    pool = Pool(processes=params["num_analyzers"])
    [
        pool.apply_async(foci_cell(cell, t0, image_data_seg, image_data_FL, params))
        for cell_id, cell in Cells.items()
    ]
    pool.close()
    pool.join()


# parralel function for each cell
def foci_cell(cell, t0, image_data_seg, image_data_FL, params):
    """find foci in a cell, single instance to be called by the foci_analysis_pool for parallel processing.
    Parameters
    ----------
    cell : Cell object
        Cell object to be analyzed.
    t0 : int
        First time index.
    image_data_seg : numpy array
        Segmented image data.
    image_data_FL : numpy array
        Fluorescent image data.
    params : dict
        Dictionary of parameters.
    """
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

            minsig = params["foci_log_minsig"]
            maxsig = params["foci_log_maxsig"]
            thresh = params["foci_log_thresh"]
            peak_med_ratio = params["foci_log_peak_med_ratio"]

            disp_l_tmp, disp_w_tmp, foci_h_tmp = foci_lap(
                image_data_temp_seg,
                image_data_temp,
                cell,
                t,
                minsig,
                maxsig,
                thresh,
                peak_med_ratio,
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


def find_blobs_in_cell(img, img_foci, cell, t, maxsig, minsig, thresh):
    # pull out useful information for just this time point
    # find position of the time point in lists (time points may be missing):
    i = cell.times.index(t)
    region = cell.labels[i]
    # calculate median cell intensity. Used to filter foci
    img_foci_masked = np.copy(img_foci).astype(float)
    # correction for difference between segmentation image mask and fluorescence channel by padding on the rightmost column(s)
    if np.shape(img) != np.shape(img_foci_masked):
        delta_col = np.shape(img)[1] - np.shape(img_foci_masked)[1]
        img_foci_masked = np.pad(img_foci_masked, ((0, 0), (0, delta_col)), "edge")
    img_foci_masked[img != region] = np.nan
    cell_fl_median = np.nanmedian(img_foci_masked)

    img_foci_masked[img != region] = 0

    # find blobs using difference of gaussian
    # if two blobs overlap by more than this fraction, smaller blob is cut:
    over_lap = 0.95
    # number of division to consider between min ang max sig:
    numsig = maxsig - minsig + 1
    blobs = blob_log(
        img_foci_masked,
        min_sigma=minsig,
        max_sigma=maxsig,
        overlap=over_lap,
        num_sigma=numsig,
        threshold=thresh,
    )

    return blobs, cell_fl_median


def filter_blobs(blobs, img, bbox, threshold):
    """
    Filters blobs!

    Filters used:
        * bbox: if the blob is outside the bbox, it is discarded.
        * peak < threshold (threshold = cell_fl_median * peak_med_ratio)

    Parameters
    ----------
    blobs: 2D np.array, 3xN
        all of your blobs
    img: 2D np.array
        image to be used to check that the brightness is above the given threshold
    threshold:
        filtering threshold
    """
    new_blobs = []
    x_fits, y_fits, w_fits = [], [], []
    # loop through each potential foci
    for blob in blobs:
        yloc, xloc, sig = blob  # x location, y location, and sigma of gaus
        xloc = int(np.around(xloc))  # switch to int for slicing images
        yloc = int(np.around(yloc))
        radius = int(np.ceil(np.sqrt(2) * sig))  # use to slice area around foci

        # ensure blob is inside the bounding box
        # this might be better to check if (xloc, yloc) is in regions.coords
        if (bbox[0] < yloc) & (yloc < bbox[2]) & (bbox[1] < xloc) & (xloc < bbox[3]):
            # cut out a small image from original image to fit gaussian
            gfit_area = img[
                yloc - radius : yloc + radius, xloc - radius : xloc + radius
            ]

            # fit gaussian to proposed foci in small box
            (peak_fit, x_fit, y_fit, w_fit) = fitgaussian(gfit_area)

            if (
                (x_fit <= 0)
                or (x_fit >= radius * 2)
                or (y_fit <= 0)
                or (y_fit >= radius * 2)
            ):
                continue
            elif peak_fit < threshold:
                continue
            else:
                new_blobs.append(blob)
                x_fits.append(x_fit)
                y_fits.append(y_fit)
                w_fits.append(w_fit)
    return new_blobs, x_fits, y_fits, w_fits


# actual worker function for foci detection
def foci_lap(
    img, img_foci, cell, t, minsig, maxsig, thresh, peak_med_ratio, preview=False
):
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

    blobs, cell_fl_median = find_blobs_in_cell(
        img, img_foci, cell, t, maxsig, minsig, thresh
    )
    # these will hold information about foci position temporarily
    i = cell.times.index(t)
    bbox = np.int16(cell.bboxes[i])
    cell_orientation = cell.orientations[i]
    centroid = cell.centroids[i]

    blobs, x_fits, y_fits, w_fits = filter_blobs(
        blobs, img_foci, bbox, threshold=cell_fl_median * peak_med_ratio
    )
    if blobs == []:
        if preview:
            return [], [], [], [], [], []
        return [], [], []
    blobs, x_fits, y_fits, w_fits = (
        np.array(blobs),
        np.array(x_fits),
        np.array(y_fits),
        np.array(w_fits),
    )

    # vectorized blob stat assembly!
    y_blob = np.int16(np.around(blobs[:, 0]))
    x_blob = np.int16(np.around(blobs[:, 1]))
    r_blob = np.int16(np.ceil(np.sqrt(2) * blobs[:, 2]))

    x_gaus = x_blob - r_blob + x_fits
    y_gaus = y_blob - r_blob + y_fits

    # calculate distance of foci from middle of cell (scikit image)
    if cell_orientation < 0:
        cell_orientation = np.pi + cell_orientation

    disp_y = (y_gaus - centroid[0]) * np.sin(cell_orientation) - (
        x_gaus - centroid[1]
    ) * np.cos(cell_orientation)
    disp_x = (y_gaus - centroid[0]) * np.cos(cell_orientation) + (
        x_gaus - centroid[1]
    ) * np.sin(cell_orientation)

    y_lo_lim, x_lo_lim = y_blob - maxsig, x_blob - maxsig
    y_hi_lim, x_hi_lim = y_blob + maxsig, x_blob + maxsig

    # loop through each potential foci
    foci_h = []  # foci total amount (from raw image)
    for y_lo, x_lo, y_hi, x_hi in zip(y_lo_lim, x_lo_lim, y_hi_lim, x_hi_lim):
        gfit_area_fixed = img_foci[y_lo:y_hi, x_lo:x_hi]
        foci_h = np.append(foci_h, np.sum(gfit_area_fixed))

    if preview:
        return disp_y, disp_x, foci_h, x_blob, y_blob, r_blob

    return disp_y, disp_x, foci_h


def ultra_kymograph(ana_dir, experiment_name, foci_plane, fov_id, peak_id, n_steps=50):
    postfix = f"sub_{foci_plane}"
    sub_stack_fl = load_subtracted_stack(
        ana_dir, experiment_name, fov_id, peak_id, postfix
    )
    ultra_kymo = []
    for i in range(sub_stack_fl.shape[0] - n_steps):
        sub_stack_fl2 = sub_stack_fl[i : i + n_steps, :, :]
        mini_kymo = np.hstack(sub_stack_fl2)
        ultra_kymo.append(mini_kymo)
    return np.array(ultra_kymo)


def update_cell_foci(cells, foci):
    """Updates cells' .foci attribute in-place using information
    in foci dictionary
    """
    for focus_id, focus in foci.items():
        for cell in focus.cells:

            cell_id = cell.id
            cells[cell_id].foci[focus_id] = focus


def foci(ana_dir, params, fl_plane, seg_method, cell_file_path):
    """
    Main function for foci analysis. Loads cells, finds foci, and saves out the results.
    Parameters
    ----------
    params : dict
        Dictionary of parameters
    fl_plane : str
        Name of fluorescence plane to use for foci analysis
    seg_method : str
        Name of segmentation method to use for foci analysis
    cell_file_path : str
        Path to cell file

    Returns
    -------
    None"""

    with open(cell_file_path, "rb") as cell_file:
        Cells = pickle.load(cell_file)
    specs = load_specs(ana_dir)
    time_table = load_time_table(ana_dir)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])
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
                fov_id, peak_id, Cells_of_peak, ana_dir, params, seg_method, time_table
            )

    # Output data to both dictionary and the .mat format used by the GUI

    write_cells_to_json(Cells, params["cell_dir"] / "all_cells_foci.json")
    # TODO readd MAT format export
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

        self.specs = load_specs(self.analysis_folder)

        self.n_steps = 40

        self.set_plane()
        self.set_segmentation_method()
        self.set_cellfile()
        self.set_fovs(self.valid_fovs)
        self.set_log_maxsig()
        self.set_log_minsig()
        self.set_log_thresh()
        self.set_log_peak_ratio()
        self.generate_fovs_to_peaks()
        self.peak_switch_widget = InteractivePeakChooser(
            self.valid_fovs, self.fov_to_peaks
        )
        self.set_peak_and_fov()

        self.plane_widget.changed.connect(self.set_plane)
        self.segmentation_method_widget.changed.connect(self.set_segmentation_method)
        self.cellfile_widget.changed.connect(self.set_cellfile)
        self.log_minsig_widget.changed.connect(self.set_log_minsig)
        self.log_maxsig_widget.changed.connect(self.set_log_maxsig)
        self.log_thresh_widget.changed.connect(self.set_log_thresh)
        self.log_peak_ratio_widget.changed.connect(self.set_log_peak_ratio)
        self.viewer.window._status_bar._toggle_activity_dock(True)
        self.preview_widget.clicked.connect(self.render_preview)
        self.peak_switch_widget.connect(self.set_peak_and_fov)

        self.append(self.plane_widget)
        self.append(self.segmentation_method_widget)
        self.append(self.cellfile_widget)
        self.append(self.log_minsig_widget)
        self.append(self.log_maxsig_widget)
        self.append(self.log_thresh_widget)
        self.append(self.log_peak_ratio_widget)
        self.append(self.preview_widget)
        self.append(self.peak_switch_widget)

    def generate_fovs_to_peaks(self):
        with open(self.cellfile, "rb") as cell_file:
            Cells = pickle.load(cell_file)
        self.fov_to_peaks = {}
        for valid_fov in self.valid_fovs:
            valid_peaks = organize_cells_by_channel(Cells, self.specs)[valid_fov].keys()
            self.fov_to_peaks[valid_fov] = valid_peaks

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
        foci(
            self.analysis_folder,
            self.params,
            self.fl_plane,
            self.segmentation_method,
            str(self.cellfile),
        )

    def set_peak_and_fov(self):
        self.preview_peak = self.peak_switch_widget.cur_peak
        self.preview_fov = self.peak_switch_widget.cur_fov

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
        """
        Previews foci in a pseudo-kymograph.
        """

        self.viewer.layers.clear()
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()
        self.set_params()
        # TODO: Add a check for number of steps
        n_steps = self.n_steps

        # load images
        # TODO: remove params dependency here.

        kymos = []
        ## pull out first fov & peak id with cells
        for plane in self.valid_planes:
            kymo = ultra_kymograph(
                self.analysis_folder,
                self.experiment_name,
                plane,
                self.preview_fov,
                self.preview_peak,
                n_steps=n_steps,
            )
            kymos.append(kymo)

        kymos = np.array(kymos)

        self.viewer.add_image(np.array(kymos))
        img_width = kymos[0].shape[2] // n_steps
        self.img_width = img_width
        with open(self.cellfile, "rb") as cell_file:
            Cells = pickle.load(cell_file)
        time_table = load_time_table(self.analysis_folder)
        Cells_by_peak = organize_cells_by_channel(Cells, self.specs)[self.preview_fov][
            self.preview_peak
        ]

        x_blob, y_blob, radii, times = foci_preview(
            self.preview_fov,
            self.preview_peak,
            Cells_by_peak,
            self.analysis_folder,
            self.params,
            self.segmentation_method,
            time_table,
        )
        self.x_blob, self.y_blob, self.radii, self.times = (
            np.array(x_blob),
            np.array(y_blob),
            np.array(radii),
            np.array(times),
        )

        self.draw_points()
        # Add this here because napari doesn't like it if you access this without having any images in the canvas.
        self.viewer.dims.events.current_step.connect(self.draw_points)

        self.viewer.layers[-1].scale = [1, 0.5]
        self.viewer.layers[-2].scale = [1, 0.5]
        self.viewer.layers["Image"].reset_contrast_limits()
        self.viewer.reset_view()
        self.viewer.layers[-1].selected_data = []

    # draws foci within the current frame. doing it as a nested function is much more straightforward
    # than passing around a bunch of parameters, so I am doing it this way.
    def draw_points(self):
        try:
            self.viewer.layers["Image"].reset_contrast_limits()
        except Exception as e:
            pass
        cur_frame = self.viewer.dims.current_step[1]
        times_in_range_mask = (cur_frame + 1 <= self.times) & (
            self.times < cur_frame + self.n_steps + 1
        )
        times_in_range = self.times[times_in_range_mask]
        relevant_x = self.x_blob[times_in_range_mask]
        relevant_x_with_offset = (
            np.array(relevant_x)
            + (np.array(times_in_range) - cur_frame - 1) * self.img_width
        )
        relevant_y = self.y_blob[times_in_range_mask]
        points = np.stack((relevant_y, relevant_x_with_offset)).transpose()

        # if the points layer exists => update it.
        # if the points layer does not exist => create it.
        try:
            self.viewer.layers["points"].data = points
            self.viewer.layers["points"].size = np.array(
                self.radii[times_in_range_mask]
            )
        except Exception as e:
            self.points = self.viewer.add_points(
                data=points,
                size=1.5 * np.array(self.radii[times_in_range_mask]),
                name="points",
                face_color="orange",
                edge_color="white",
            )
