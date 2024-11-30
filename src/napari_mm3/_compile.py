import multiprocessing
import re
import os
import yaml
import six
import pickle
import sys
import traceback
import tifffile as tiff
import numpy as np
import json
from typing import Union

from scipy import ndimage as ndi
from skimage.feature import match_template
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint
from scipy.signal import find_peaks_cwt
from magicgui.widgets import FloatSpinBox, SpinBox, PushButton, ComboBox, CheckBox
from napari import Viewer
from napari.utils import progress
from ._deriving_widgets import (
    MM3Container,
    FOVChooser,
    TimeRangeSelector,
    PlanePicker,
    information,
    warning,
    load_unmodified_stack,
)


#### Helpful utility functions.


def get_plane(filepath: str) -> Union[str, None]:
    """Extracts the plane / channel number (e.g. phase fluorescence etc.) from a tiff file name.
    It is used to sort the tiff files into the correct order for processing.
    """
    pattern = r"(c\d+).tif"
    res = re.search(pattern, filepath, re.IGNORECASE)
    if res != None:
        return res.group(1)
    else:
        return None


def get_fov(filepath: str) -> Union[int, None]:
    """Extracts the fov number from a tiff file name."""
    pattern = r"xy(\d+)\w*.tif"
    res = re.search(pattern, filepath, re.IGNORECASE)
    if res != None:
        return int(res.group(1))
    else:
        return None


def get_time(filepath: str) -> Union[np.int_, None]:
    """Extracts the time point from a tiff file name."""
    pattern = r"t(\d+)\w*.tif"
    res = re.search(pattern, filepath, re.IGNORECASE)
    if res != None:
        return np.int_(res.group(1))
    else:
        return None


def stack_channels(found_files: np.ndarray, TIFF_dir: Path) -> None:
    """Check if phase and fluorescence channels are separate tiffs.
    If so, restack them as one tiff file and save out to TIFF_dir.
    Move the original files to a new directory original_TIFF.

    Parameters
    ---------
    found_files: ndarray of filepaths for each imaging plane
    TIFF_dir: Path
        Directory containing the TIFF files.

    Returns
    ---------
    None
    """

    found_files = np.transpose(found_files)

    for files in found_files:
        information("Merging files")
        print(*files, sep="\n")
        ims = [tiff.imread(f) for f in files]
        im_out = np.stack(ims, axis=0)

        # need to insert regex here to catch variants
        name_out = re.sub("c\d+", "", str(files[0]), flags=re.IGNORECASE)
        # 'minisblack' necessary to ensure that it interprets image as black/white.
        tiff.imwrite(name_out, im_out, photometric="minisblack")

        old_tiff_path = TIFF_dir.parent / "original_TIFF"
        if not old_tiff_path.exists():
            os.makedirs(old_tiff_path)

        for f in files:
            os.rename(f, old_tiff_path / Path(f).name)

    return


### Functions for working with TIFF metadata ###


# get params is the major function which processes raw TIFF images
def get_tif_params(
    TIFF_dir: Path,
    TIFF_source: str,
    image_filename: str,
    channel_width: int,
    channel_separation: int,
    channel_width_pad: int,
    channel_detection_snr: float,
    phase_plane: str,
    planes: list[str],
    image_orientation: str,
    find_channels: bool = True,
) -> dict:
    """This is a damn important function for getting the information
    out of an image. It loads a tiff file, pulls out the image data, and the metadata,
    including the location of the channels if flagged.

    it returns a dictionary like this for each image:

    'filename': image_filename,
    'fov' : image_metadata['fov'], # fov id
    't' : image_metadata['t'], # time point
    'jdn' : image_metadata['jdn'], # absolute julian time
    'plane_names' : image_metadata['plane_names'] # list of plane names
    'channels': cp_dict, # dictionary of channel locations, in the case of Unet-based channel segmentation, it's a dictionary of channel labels

    Called by
    compile

    Calls
    extract_metadata
    find_channels
    """

    try:
        with tiff.TiffFile(TIFF_dir / image_filename) as tif:
            image_data = tif.asarray()
            if TIFF_source == "nd2":
                image_metadata = get_tif_metadata_nd2(tif)
            elif TIFF_source == "BioFormats / other TIFF":
                image_metadata = get_tif_metadata_filename(tif)

            if image_metadata["planes"] is None:
                image_metadata["planes"] = planes

            if find_channels:
                # fix the image orientation and get the number of planes

                image_data = fix_orientation(image_orientation, phase_plane, image_data)

                # if the image data has more than 1 plane restrict image_data to phase,
                # which should have highest mean pixel data
                if len(image_data.shape) > 2:
                    # ph_index = np.argmax([np.mean(image_data[ci]) for ci in range(image_data.shape[0])])
                    ph_index = int(phase_plane[1:]) - 1
                    image_data = image_data[ph_index]

                # get shape of single plane
                img_shape = [image_data.shape[0], image_data.shape[1]]

                # find channels on the processed image
                chnl_loc_dict = find_channel_locs(
                    channel_width,
                    channel_separation,
                    channel_width_pad,
                    channel_detection_snr,
                    image_data,
                )

        information("Analyzed %s" % image_filename)

        return {
            "filepath": TIFF_dir / image_filename,
            "fov": image_metadata["fov"],
            "t": image_metadata["t"],
            "jd": image_metadata["jd"],
            "planes": image_metadata["planes"],
            "shape": img_shape,
            "channels": chnl_loc_dict,
        }

    except:
        warning(f"Failed get_params for {image_filename}")
        information(sys.exc_info()[0])
        information(sys.exc_info()[1])
        information(traceback.print_tb(sys.exc_info()[2]))
        return {
            "filepath": TIFF_dir / image_filename,
            "analyze_success": False,
        }


def get_tif_params_loop(
    num_analyzers: int,
    TIFF_dir: Path,
    found_files: list,
    TIFF_source: str,
    channel_width: int,
    channel_separation: int,
    channel_width_pad: int,
    channel_detection_snr: float,
    phase_plane: str,
    image_orientation: str,
) -> dict:
    """Loop over found files and extract image parameters.

    Parameters
    ----------
    num_analyzers: int
        Number of analyzers to use for multiprocessing.
    TIFF_dir: Path
        Directory containing the TIFF files.
    found_files: list
        List of tiff files to analyze.
    TIFF_source: str
        Source of the TIFF files.
    channel_width: int
        Width of the channels.
    channel_separation: int
        Separation between channels.
    channel_width_pad: int
        Padding for the channel width.
    channel_detection_snr: float
        Signal-to-noise ratio for channel detection.
    phase_plane: str
        Phase plane channel identifier.
    image_orientation: str
        Orientation of the image.

    Returns
    --------
    analyzed_imgs: dict
        Dictionary of image metadata.
    """

    analyzed_imgs = {}
    pool = Pool(num_analyzers)

    for fn in found_files:
        analyzed_imgs[fn] = pool.apply_async(
            get_tif_params,
            args=(
                TIFF_dir,
                TIFF_source,
                fn,
                channel_width,
                channel_separation,
                channel_width_pad,
                channel_detection_snr,
                phase_plane,
                [],
                image_orientation,
                True,
            ),
        )

    information("Waiting for image analysis pool to be finished.")

    pool.close()
    pool.join()

    information("Image analysis pool finished, getting results.")

    for fn in analyzed_imgs.keys():
        result = analyzed_imgs[fn]
        if result.successful():
            analyzed_imgs[fn] = result.get()
        else:
            analyzed_imgs[fn] = {"analyze_success": False}

    return analyzed_imgs


def get_tif_metadata_nd2(tif: tiff.TiffFile) -> dict:
    """This function pulls out the metadata from a tif file and returns it as a dictionary.
    This if tiff files as exported by the mm3 function nd2ToTiff. All the metdata
    is found in that script and saved in json format to the tiff, so it is simply extracted here

    Paramters
    ---------
    tif: tiff.TiffFile
        TIFF file object from which data will be extracted

    Returns
    -------
    idata: dict
        dictionary of values:
            'fov': int,
            't' : int,
            'jdn' (float)
            'planes' (list of strings)

    Called by
    mm3_Compile.get_tif_params

    """
    # get the first page of the tiff and pull out image description
    # this dictionary should be in the above form

    for tag in tif.pages[0].tags:
        if tag.name == "ImageDescription":
            idata = tag.value
            break

    idata = json.loads(idata)
    return idata


def get_tif_metadata_filename(tif: tiff.TiffFile) -> dict:
    """This function pulls out the metadata from a tif filename and returns it as a dictionary.
    This just gets the tiff metadata from the filename and is a backup option when the known format of the metadata is not known.

    Parameters
    ---------
    tif: tiff.TiffFile
        TIFF file object from which data will be extracted

    Returns
    -------
    idata: dict
        dictionary of values:
            'fov': int,
            't' : int,
            'jdn' (float)

    Called by
    get_tif_params

    """
    idata = {
        "fov": get_fov(tif.filename),  # fov id
        "t": get_time(tif.filename),  # time point
        "jd": -1 * 0.0,  # absolute julian time
        "planes": get_plane(tif.filename),
    }

    return idata


### Functions for dealing with cross-correlations, which are used to determine empty/full channels ###
# calculate cross correlation between pixels in channel stack
def channel_xcorr(
    alignment_pad: int,
    ana_dir: Path,
    experiment_name: str,
    phase_plane: str,
    fov_id: int,
    peak_id: int,
) -> list:
    """
    Function calculates the cross correlation of images in a
    stack to the first image in the stack. The output is an
    array that is the length of the stack with the best cross
    correlation between that image and the first image.

    The very first value should be 1.

    Parameters
    ----------
    alignment_pad: int
        Padding for channel alignment
    ana_dir: Path
        Analysis directory
    experiment_name: str
        Name of the experiment
    phase_plane: str
        Phase contrast channel identifier
    fov_id: int
        fov to analyze
    peak_id:
        peak (trap) to analyze

    Returns
    -------
    xcorr_array: list
        array of cross correlations over time
    """

    pad_size = alignment_pad
    number_of_images = 20

    image_data = load_unmodified_stack(
        ana_dir, experiment_name, fov_id, peak_id, phase_plane
    )

    if image_data.shape[0] > number_of_images:
        spacing = int(image_data.shape[0] / number_of_images)
        image_data = image_data[::spacing, :, :]
        if image_data.shape[0] > number_of_images:
            image_data = image_data[:number_of_images, :, :]

    first_img = np.pad(image_data[0, :, :], pad_size, mode="reflect")

    xcorr_array = []
    for img in image_data:
        xcorr_array.append(np.max(match_template(first_img, img)))

    return xcorr_array


def compute_xcorr(
    channel_masks: dict,
    user_spec_fovs: list,
    num_analyzers: int,
    alignment_pad: int,
    ana_dir: Path,
    experiment_name: str,
    phase_plane: str,
) -> None:
    """Loop over FOVs and compute time autocorrelation for each channel
    Calls channel_xcorr

    Parameters
    ----------
    channel_masks: list
        Trap locations relative to FOV
    user_spec_fovs:
        FOVs to analyzed
    num_analyzers: int
        Number of analyzers to use for multiprocessing
    alignment_pad: int
        Padding for alignment
    ana_dir: Path
        Analysis directory
    experiment_name: str
        Name of the experiment
    phase_plane: str
        Phase contrast channel identifier

    Returns
    ---------
    None
    """

    crosscorrs: dict[int, dict[int, dict[str, Union[float, list[float]]]]] = {}

    for fov_id in progress(user_spec_fovs):
        information("Calculating cross correlations for FOV %d." % fov_id)
        crosscorrs[fov_id] = {}
        pool = Pool(num_analyzers)

        for peak_id in sorted(channel_masks[fov_id].keys()):
            information("Calculating cross correlations for peak %d." % peak_id)
            # linear loop
            # crosscorrs[fov_id][peak_id] = channel_xcorr(params, fov_id, peak_id)
            crosscorrs[fov_id][peak_id] = pool.apply_async(
                channel_xcorr,
                args=(
                    alignment_pad,
                    ana_dir,
                    experiment_name,
                    phase_plane,
                    fov_id,
                    peak_id,
                ),
            )

        information("Waiting for cross correlation pool to finish for FOV %d." % fov_id)

        pool.close()  # tells the process nothing more will be added.
        pool.join()  # blocks script until everything has been processed and workers exit

        information("Finished cross correlations for FOV %d." % fov_id)

    for fov_id, peaks in six.iteritems(crosscorrs):
        for peak_id, result in six.iteritems(peaks):
            if result.successful():
                crosscorrs[fov_id][peak_id] = {
                    "ccs": result.get(),
                    "cc_avg": np.average(result.get()),
                }

            else:
                crosscorrs[fov_id][peak_id] = False

    information("Writing cross correlations file.")
    with open(os.path.join(ana_dir, "crosscorrs.pkl"), "wb") as xcorrs_file:
        pickle.dump(crosscorrs, xcorrs_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(ana_dir, "crosscorrs.txt"), "w") as xcorrs_file:
        pprint(crosscorrs, stream=xcorrs_file)
    information("Wrote cross correlations files.")


### functions about trimming, padding, and manipulating images
# cuts out channels from the image
def cut_slice(image_data: np.ndarray, channel_loc: list) -> np.ndarray:
    """Takes an image and cuts out the channel based on the slice location
    slice location is the list with the peak information, in the form
    [[y1, y2],[x1, x2]]. Returns the channel slice as a numpy array.
    The numpy array will be a stack if there are multiple planes.

    if you want to slice all the channels from a picture with the channel_masks
    dictionary use a loop like this:

    for channel_loc in channel_masks[fov_id]: # fov_id is the fov of the image
        channel_slice = cut_slice[image_pixel_data, channel_loc]
        then do something with the slice

    NOTE: this function will try to determine what the shape of your
    image is and slice accordingly. It expects the images are in the order
    [t, x, y, c]. It assumes images with three dimensions are [x, y, c] not
    [t, x, y].

    Parameters
    ----------
    image_data: np.ndarray
        image to be sliced
    channel_loc: list
        nested list of slice locations [[y1, y2],[x1, x2]]

    Returns
    -------
    channel_slice: np.ndarray
        sliced channel
    """

    # case where image is in form [x, y]
    if len(image_data.shape) == 2:
        # make slice object
        channel_slicer = np.s_[
            channel_loc[0][0] : channel_loc[0][1], channel_loc[1][0] : channel_loc[1][1]
        ]

    # case where image is in form [x, y, c]
    elif len(image_data.shape) == 3:
        channel_slicer = np.s_[
            channel_loc[0][0] : channel_loc[0][1],
            channel_loc[1][0] : channel_loc[1][1],
            :,
        ]

    # case where image in form [t, x , y, c]
    elif len(image_data.shape) == 4:
        channel_slicer = np.s_[
            :,
            channel_loc[0][0] : channel_loc[0][1],
            channel_loc[1][0] : channel_loc[1][1],
            :,
        ]
    else:
        warning(
            f"Image shape not recognized. Expected 2, 3, or 4 dimensions, found {image_data.ndim} dimensions with shape {image_data.shape}."
        )

    # slice based on appropriate slicer object.
    channel_slice = image_data[channel_slicer]

    # pad y of channel if slice happened to be outside of image
    y_difference = (channel_loc[0][1] - channel_loc[0][0]) - channel_slice.shape[1]
    if y_difference > 0:
        paddings = [[0, 0], [0, y_difference], [0, 0], [0, 0]]  # t  # y  # x  # c
        channel_slice = np.pad(channel_slice, paddings, mode="edge")

    return channel_slice


# slice_and_write cuts up the image files one at a time and writes them out to tiff stacks
def tiff_stack_slice_and_write(
    ana_dir: Path,
    images_to_write: list,
    channel_masks: dict,
    analyzed_imgs: dict,
    phase_plane: str,
    image_orientation: str,
    channel_dir: Path,
    experiment_name: str,
) -> None:
    """Writes out 4D stacks of TIFF images per channel.
    Loads all tiffs from and FOV into memory and then slices all time points at once.

    Parameters
    ----------
    ana_dir: Path
        analysis directory
    images_to_write: list
        list of images to write
    channel_masks: dict
        dictionary of channel masks
    analyzed_imgs: dict
        dictionary of analyzed images
    phase_plane: str
        phase contrast channel identifier
    image_orientation: str
        image orientation
    channel_dir: Path
        directory to save the stacked and sliced channels
    experiment_name: str
        name of the experiment

    Returns
    -------
    None

    """

    # make an array of images and then concatenate them into one big stack
    image_fov_stack = []

    # go through list of images and get the file path
    for n, image in enumerate(images_to_write):
        # analyzed_imgs dictionary will be found in main scope. [0] is the key, [1] is jd
        image_params = analyzed_imgs[image[0]]
        information("Loading %s." % image_params["filepath"].name)

        if n == 1:
            # declare identification variables for saving using first image
            fov_id = image_params["fov"]

        # load the tif and store it in array
        with tiff.TiffFile(image_params["filepath"]) as tif:
            image_data = tif.asarray()

        # channel finding was also done on images after orientation was fixed
        image_data = fix_orientation(image_orientation, phase_plane, image_data)

        # add additional axis if the image is flat
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, 0)

        # change axis so it goes Y, X, Plane
        image_data = np.rollaxis(image_data, 0, 3)

        # add it to list. The images should be in time order
        image_fov_stack.append(image_data)

    # concatenate the list into one big ass stack
    image_fov_array = np.stack(image_fov_stack, axis=0)

    # cut out the channels as per channel masks for this fov
    for peak, channel_loc in six.iteritems(channel_masks[fov_id]):
        information("Slicing and saving channel peak %d." % peak)

        # channel masks should only contain ints, but you can use this for hard fix
        # for i in range(len(channel_loc)):
        #     for j in range(len(channel_loc[i])):
        #         channel_loc[i][j] = int(channel_loc[i][j])

        # slice out channel.
        # The function should recognize the shape length as 4 and cut all time points
        channel_stack = cut_slice(image_fov_array, channel_loc)

        # save a different time stack for all colors
        for color_index in range(channel_stack.shape[3]):
            # save stack
            # this is the filename for the channel
            channel_filename = channel_dir / (
                f"{experiment_name}_xy{fov_id:03d}_p{peak:04d}_c{color_index + 1}.tif"
            )
            tiff.imwrite(
                channel_filename,
                channel_stack[:, :, :, color_index],
                compression=("zlib", 4),
            )

    return


def make_consensus_mask(
    fov: int,
    analyzed_imgs: dict,
    image_rows: int,
    image_cols: int,
    crop_wp: int,
    chan_lp: int,
) -> np.ndarray:
    """
    Generate consensus channel mask for a given fov.

    Parameters
    ----------
    fov: int
        fov to analyze
    analyzed_imgs: dict
        image data
    image_rows: int
        image height
    image_cols: int
        image width
    crop_wp: int
        channel width padding
    crop_lp: int
        channel_width padding

    Returns
    -------
    consensus_mask: np.ndarray
    """

    consensus_mask = np.zeros([image_rows, image_cols])  # mask for labeling

    # bring up information for each image
    for img_k in analyzed_imgs.keys():
        img_v = analyzed_imgs[img_k]
        # skip this one if it is not of the current fov
        if img_v["fov"] != fov:
            continue

        # for each channel in each image make a single mask
        img_chnl_mask = np.zeros([image_rows, image_cols])

        # and add the channel mask to it
        for chnl_peak, peak_ends in six.iteritems(img_v["channels"]):
            # pull out the peak location and top and bottom location
            # and expand by padding (more padding done later for width)
            x1 = max(chnl_peak - crop_wp, 0)
            x2 = min(chnl_peak + crop_wp, image_cols)
            y1 = max(peak_ends["closed_end_px"] - chan_lp, 0)
            y2 = min(peak_ends["open_end_px"] + chan_lp, image_rows)

            # add it to the mask for this image
            img_chnl_mask[y1:y2, x1:x2] = 1

        # add it to the consensus mask
        consensus_mask += img_chnl_mask

    # Normalize consensus mask between 0 and 1.
    consensus_mask = consensus_mask.astype("float32") / float(np.amax(consensus_mask))

    # threshhold and homogenize each channel mask within the mask, label them
    # label when value is above 0.1 (so 90% occupancy), transpose.
    # the [0] is for the array ([1] is the number of regions)
    # It transposes and then transposes again so regions are labeled left to right
    # clear border it to make sure the channels are off the edge
    consensus_mask = ndi.label(consensus_mask)[0]

    return consensus_mask


def update_channel_masks(
    max_len: int,
    max_wid: int,
    binary_core: np.ndarray,
    image_cols: int,
    channel_masks_1fov: dict,
) -> tuple[int, int, dict]:
    """
    Add channel locations to dict. Update max channel length/width.

    Parameters
    ----------
    max_len: int
        maximum channel length
    max_wid: int
        maximum channel width
    binary_core: np.ndarray
        binary mask selecting current label
    image_cols: int
        image width
    channel_masks_1fov: dict
        dictionary of channel masks

    Returns
    -------
    max_len: int
        updated maximum channel length
    max_wid: int
        updated maximum channel width
    channel_masks_1fov: dict
        updated dictionary of channel masks
    """

    # clean up the rough edges
    poscols = np.any(binary_core, axis=0)  # column positions where true (any)
    posrows = np.any(binary_core, axis=1)  # row positions where true (any)

    # channel_id given by horizontal position
    # this is important. later updates to the positions will have to check
    # if their channels contain this median value to match up
    channel_id = int(np.median(np.where(poscols)[0]))

    # store the edge locations of the channel mask in the dictionary. Will be ints
    min_row = np.min(np.where(posrows)[0])
    max_row = np.max(np.where(posrows)[0])
    min_col = np.min(np.where(poscols)[0])
    max_col = np.max(np.where(poscols)[0])

    # if the min/max cols are within the image bounds,
    # add the mask, as 4 points, to the dictionary
    if min_col > 0 and max_col < image_cols:
        channel_masks_1fov[channel_id] = [
            [min_row, max_row],
            [min_col, max_col],
        ]

        # find the largest channel width and height while you go round
        max_len = int(max(max_len, max_row - min_row))
        max_wid = int(max(max_wid, max_col - min_col))

    return max_len, max_wid, channel_masks_1fov


def adjust_channel_mask(
    chnl_mask: list, cm_copy: list, max_len: int, max_wid: int, image_cols: int
) -> list:
    """
    Expand channel mask to match maximum length and width.

    Parameters
    ----------
    chnl_mask: list
        indices of channel mask corners
    cm_copy: list
        consensus channel mask
    max_len: int
        maximum channel length
    max_width: int
        maximum channel width
    image_cols: int
        image width in pixels

    Returns
    -------
    cm_copy: list
        updated consensus channel mask
    """
    # just add length to the open end (bottom of image, low column)
    if chnl_mask[0][1] - chnl_mask[0][0] != max_len:
        cm_copy[0][1] = chnl_mask[0][0] + max_len
    # enlarge widths around the middle, but make sure you don't get floats
    if chnl_mask[1][1] - chnl_mask[1][0] != max_wid:
        wid_diff = max_wid - (chnl_mask[1][1] - chnl_mask[1][0])
        if wid_diff % 2 == 0:
            cm_copy[1][0] = max(chnl_mask[1][0] - wid_diff / 2, 0)
            cm_copy[1][1] = min(chnl_mask[1][1] + wid_diff / 2, image_cols - 1)
        else:
            cm_copy[1][0] = max(chnl_mask[1][0] - (wid_diff - 1) / 2, 0)
            cm_copy[1][1] = min(chnl_mask[1][1] + (wid_diff + 1) / 2, image_cols - 1)

    cm_copy = [list(map(int, i)) for i in cm_copy]  # make sure they are ints

    return cm_copy


# make masks from initial set of images (same images as clusters)
def make_masks(
    analyzed_imgs: dict,
    t_start: int,
    t_end: int,
    ana_dir: Path,
    channel_width: int,
    channel_width_pad: int,
    channel_length_pad: int,
) -> dict:
    """
    Make masks goes through the channel locations in the image metadata and builds a consensus
    Mask for each image per fov, which it returns as dictionary named channel_masks.
    The keys in this dictionary are fov id, and the values is another dictionary. This dict's keys are channel locations (peaks) and the values is a [2][2] array:
    [[minrow, maxrow],[mincol, maxcol]] of pixel locations designating the corner of each mask
    for each channel on the whole image

    One important consequence of these function is that the channel ids and the size of the
    channel slices are decided now. Updates to mask must coordinate with these values.

    Parameters
    ----------
    analyzed_imgs : dict
        image information created by get_params
    t_start : int
        start time for analysis
    t_end : int
        end time for analysis
    ana_dir : Path
        directory to save the masks
    channel_width : int
        width of the channels
    channel_width_pad : int
        padding for the channel width
    channel_length_pad : int
        padding for the channel length

    Returns
    -------
    channel_masks : dict
        dictionary of consensus channel masks.

    Called By
    compile

    Calls
    make_consensus_mask
    get_max_chnl_dim
    adjust_channel_mask
    """
    information("Determining initial channel masks...")

    # only calculate channels masks from images before t_end in case it is specified
    if t_start:
        analyzed_imgs = {
            fn: i_metadata
            for fn, i_metadata in six.iteritems(analyzed_imgs)
            if i_metadata["t"] >= t_start
        }
    if t_end:
        analyzed_imgs = {
            fn: i_metadata
            for fn, i_metadata in six.iteritems(analyzed_imgs)
            if i_metadata["t"] <= t_end
        }

    # declare temp variables from parameters.
    crop_wp = int(channel_width_pad + channel_width / 2)
    chan_lp = int(channel_length_pad)

    # get the size of the images (hope they are the same)
    for img_k in analyzed_imgs.keys():
        img_v = analyzed_imgs[img_k]
        image_rows = img_v["shape"][0]  # x pixels
        image_cols = img_v["shape"][1]  # y pixels
        break  # just need one. using iteritems mean the whole dict doesn't load

    # get the fov ids
    fovs = []
    for img_k in analyzed_imgs.keys():
        img_v = analyzed_imgs[img_k]
        if img_v["fov"] not in fovs:
            fovs.append(img_v["fov"])

    # max width and length across all fovs. channels will get expanded by these values
    # this important for later updates to the masks, which should be the same
    max_len = 0
    max_wid = 0

    # intialize dictionary
    channel_masks = {}

    # for each fov make a channel_mask dictionary from consensus mask
    for fov in fovs:

        consensus_mask = make_consensus_mask(
            fov, analyzed_imgs, image_rows, image_cols, crop_wp, chan_lp
        )

        # initialize dict which holds channel masks {peak : [[y1, y2],[x1,x2]],...}
        channel_masks_1fov: dict[int, list[list[float]]] = {}

        # go through each label
        for label in np.unique(consensus_mask):
            if label == 0:  # label zero is the background
                continue

            binary_core = consensus_mask == label
            max_len, max_wid, channel_masks_1fov = update_channel_masks(
                max_len, max_wid, binary_core, image_cols, channel_masks_1fov
            )

        # add channel_mask dictionary to the fov dictionary, use copy to play it safe
        channel_masks[fov] = channel_masks_1fov.copy()

    # update all channel masks to be the max size
    cm_copy = channel_masks.copy()

    for fov, peaks in six.iteritems(channel_masks):
        for peak, chnl_mask in six.iteritems(peaks):
            cm_copy[fov][peak] = adjust_channel_mask(
                chnl_mask, cm_copy[fov][peak], max_len, max_wid, image_cols
            )

    # save the channel mask dictionary to a yaml and a text file
    with open(os.path.join(ana_dir, "channel_masks.txt"), "w") as cmask_file:
        pprint(cm_copy, stream=cmask_file)
    with open(os.path.join(ana_dir, "channel_masks.yaml"), "w") as cmask_file:
        yaml.dump(data=cm_copy, stream=cmask_file, default_flow_style=False, tags=None)

    information("Channel masks saved.")

    return cm_copy


# function for loading the channel masks
def load_channel_masks(ana_dir: Path) -> dict:
    """Load channel masks dictionary. Should be .yaml but try pickle too.

    Parameters
    ----------
    ana_dir: Path
        analysis directory path

    Returns
    -------
    channel_masks: dict
        dictionary of channel masks
    """
    information("Loading channel masks dictionary.")

    # try loading from .yaml before .pkl
    try:
        information("Path:", os.path.join(ana_dir, "channel_masks.yaml"))
        with open(os.path.join(ana_dir, "channel_masks.yaml"), "r") as cmask_file:
            channel_masks = yaml.safe_load(cmask_file)
    except:
        warning("Could not load channel masks dictionary from .yaml.")

        try:
            information("Path:", os.path.join(ana_dir, "channel_masks.pkl"))
            with open(os.path.join(ana_dir, "channel_masks.pkl"), "rb") as cmask_file:
                channel_masks = pickle.load(cmask_file)
        except ValueError:
            warning("Could not load channel masks dictionary from .pkl.")

    return channel_masks


# make a lookup time table for converting nominal time to elapsed time in seconds
def make_time_table(
    analyzed_imgs: dict, use_jd: bool, seconds_per_time_index: int, ana_dir: Path
) -> dict:
    """
    Loops through the analyzed images and uses the jd time in the metadata to find the elapsed
    time in seconds that each picture was taken. This is later used for more accurate elongation
    rate calculation.

    Parameters
    ---------
    analyzed_imgs : dict
        The output of get_tif_params.
    use_jd : boolean
        If set to True, 'jd' time will be used from the image metadata to use to create time table. Otherwise the 't' index will be used, and the parameter 'seconds_per_time_index' will be used to convert to seconds.
    seconds_per_time_index : int
        Time interval in seconds between consecutive imaging rounds.
    ana_dir : Path
        Directory where the time table will be saved.

    Returns
    -------
    time_table : dict
        Look up dictionary with keys for the FOV and then the time point.
    """
    information("Making time table...")

    # initialize
    time_table: dict[int, dict[int, int]] = {}

    first_time = float("inf")

    # need to go through the data once to find the first time
    for iname, idata in six.iteritems(analyzed_imgs):
        if use_jd:
            try:
                if idata["jd"] < first_time:
                    first_time = idata["jd"]
            except:
                if idata["t"] < first_time:
                    first_time = idata["t"]
        else:
            if idata["t"] < first_time:
                first_time = idata["t"]

        # init dictionary for specific times per FOV
        if idata["fov"] not in time_table:
            time_table[idata["fov"]] = {}

    for iname, idata in six.iteritems(analyzed_imgs):
        if use_jd:
            # convert jd time to elapsed time in seconds
            try:
                t_in_seconds = np.around(
                    (idata["jd"] - first_time) * 24 * 60 * 60, decimals=0
                ).astype("uint32")
            except:
                information(
                    "Failed to extract time from metadata. Using user-specified interval."
                )
                t_in_seconds = np.around(
                    (idata["t"] - first_time) * seconds_per_time_index,
                    decimals=0,
                ).astype("uint32")
        else:
            t_in_seconds = np.around(
                (idata["t"] - first_time) * seconds_per_time_index, decimals=0
            ).astype("uint32")

        time_table[int(idata["fov"])][int(idata["t"])] = int(t_in_seconds)

    with open(os.path.join(ana_dir, "time_table.yaml"), "w") as time_table_file:
        yaml.dump(
            data=time_table, stream=time_table_file, default_flow_style=False, tags=None
        )
    information("Time table saved.")

    return time_table


# finds the location of channels in a tif
def find_channel_locs(
    channel_width: int,
    channel_separation: int,
    channel_width_pad: int,
    channel_detection_snr: float,
    image_data: np.ndarray,
) -> dict:
    """Finds the location of channels from a phase contrast image. The channels are returned in
    a dictionary where the key is the x position of the channel in pixel and the value is a
    dictionary with the open and closed end in pixels in y.

    Parameters
    ----------
    channel_width : int
        The width of the channel in pixels.
    channel_separation : int
        The separation between channels in pixels.
    channel_width_pad : int
        The padding to add to the channel width.
    channel_detection_snr : float
        The signal to noise ratio for peak detection.
    image_data : np.ndarray
        The image data.

    Returns
    -------
    chnl_loc_dict : dict
        Dictionary with the channel locations.

    Called by
    get_tif_params
    """

    # declare temp variables from parameters.
    crop_wp = int(channel_width_pad + channel_width / 2)

    # Detect peaks in the x projection (i.e. find the channels)
    projection_x = image_data.sum(axis=0).astype(np.int32)
    peaks = find_peaks_cwt(
        projection_x,
        np.arange(channel_width - 5, channel_width + 5),
        min_snr=channel_detection_snr,
    )

    # If the left-most peak position is within half of a channel separation,
    # discard the channel from the list.
    if peaks[0] < (channel_separation / 2):
        peaks = peaks[1:]
    # If the difference between the right-most peak position and the right edge
    # of the image is less than half of a channel separation, discard the channel.
    if image_data.shape[1] - peaks[-1] < (channel_separation / 2):
        peaks = peaks[:-1]

    # Find the average channel ends for the y-projected image
    projection_y = image_data.sum(axis=1)
    # find derivative, must use int32 because it was unsigned 16b before.
    proj_y_d = np.diff(projection_y.astype(np.int32))
    # use the top third to look for closed end, is pixel location of highest deriv
    onethirdpoint_y = int(projection_y.shape[0] / 3.0)
    default_closed_end_px = proj_y_d[:onethirdpoint_y].argmax()
    # use bottom third to look for open end, pixel location of lowest deriv
    twothirdpoint_y = int(projection_y.shape[0] * 2.0 / 3.0)
    default_open_end_px = twothirdpoint_y + proj_y_d[twothirdpoint_y:].argmin()
    default_length = default_open_end_px - default_closed_end_px

    # go through peaks and assign information
    chnl_loc_dict = {}

    for peak in peaks:
        # set defaults
        chnl_loc_dict[peak] = {
            "closed_end_px": default_closed_end_px,
            "open_end_px": default_open_end_px,
        }
        # redo the previous y projection finding with just this channel
        channel_slice = image_data[:, peak - crop_wp : peak + crop_wp]
        slice_projection_y = channel_slice.sum(axis=1)
        slice_proj_y_d = np.diff(slice_projection_y.astype(np.int32))
        slice_closed_end_px = slice_proj_y_d[:onethirdpoint_y].argmax()
        slice_open_end_px = twothirdpoint_y + slice_proj_y_d[twothirdpoint_y:].argmin()
        slice_length = slice_open_end_px - slice_closed_end_px

        # check if these values make sense. If so, use them. If not, use default
        # make sure lenght is not 30 pixels bigger or smaller than default
        # *** This 15 should probably be a parameter or at least changed to a fraction.
        if slice_length + 15 < default_length or slice_length - 15 > default_length:
            continue
        # make sure ends are greater than 15 pixels from image edge
        if slice_closed_end_px < 15 or slice_open_end_px > image_data.shape[0] - 15:
            continue

        # if you made it to this point then update the entry
        chnl_loc_dict[peak] = {
            "closed_end_px": slice_closed_end_px,
            "open_end_px": slice_open_end_px,
        }

    return chnl_loc_dict


# define function for flipping the images on an FOV by FOV basis
def fix_orientation(
    image_orientation: str, phase_plane: str, image_data: np.ndarray
) -> np.ndarray:
    """
    Fix the orientation. The standard direction for channels to open to is down.

    Parameters
    ----------
    image_orientation : str
        The desired image orientation ('auto', 'up', 'down').
    phase_plane : str
        The phase plane channel identifier.
    image_data : np.ndarray
        The image data to be oriented.

    Returns
    -------
    np.ndarray
        The oriented image data.
    """

    image_data = np.squeeze(
        image_data
    )  # remove singleton dimensions to standardize shape

    # if this is just a phase image give in an extra layer so rest of code is fine
    flat = False  # flag for if the image is flat or multiple levels
    if len(image_data.shape) == 2:
        image_data = np.expand_dims(image_data, 0)
        flat = True

    # setting image_orientation to 'auto' will use autodetection
    if image_orientation == "auto":
        # use 'phase_plane' to find the phase plane in image_data, assuming c1, c2, c3... naming scheme here.
        try:
            ph_channel = int(re.search("[0-9]", phase_plane).group(0)) - 1
        except:
            # Pick the plane to analyze with the highest mean px value (should be phase)
            ph_channel = np.argmax(
                [np.mean(image_data[ci]) for ci in range(image_data.shape[0])]
            )

        # flip based on the index of the highest average row value
        # this should be closer to the opening
        if (
            np.argmax(image_data[ph_channel].mean(axis=1))
            < image_data[ph_channel].shape[0] / 2
        ):
            image_data = image_data[:, ::-1, :]
        else:
            pass  # no need to do anything

    # flip if up is chosen
    elif image_orientation == "up":
        return image_data[:, ::-1, :]

    # do not flip the images if "down" is the specified image orientation
    elif image_orientation == "down":
        pass

    if flat:
        image_data = image_data[0]  # just return that first layer

    return image_data


def filter_files(
    found_files: list, t_start: int, t_end: int, user_spec_fovs: list
) -> list:
    """Filter images for analysis based on user specified start / end time and FOVs

    Parameters
    ------------
    found_files: list
        list of files in TIFF directory.
    t_start: int
        start time for analysis.
    t_end: int
        end time for analysis.

    Returns
    ---------
    found_files: list
        filtered files for analysis.

    """

    found_files = [filepath.name for filepath in found_files]  # remove pre-path
    found_files = sorted(found_files)  # should sort by timepoint

    # keep images starting at this timepoint
    if t_start is not None:
        information("Removing images before time {}".format(t_start))
        # go through list and find first place where timepoint is equivalent to t_start
        for n, ifile in enumerate(found_files):
            string = re.compile(
                "t{:0=3}xy|t{:0=4}xy".format(t_start, t_start), re.IGNORECASE
            )  # account for 3 and 4 digit
            # if re.search == True then a match was found
            if re.search(string, ifile):
                # cut off every file name prior to this one and quit the loop
                found_files = found_files[n:]
                break

    # remove images after this timepoint
    if t_end is not None:
        information("Removing images after time {}".format(t_end))
        # go through list and find first place where timepoint is equivalent to t_end
        for n, ifile in enumerate(found_files):
            string = re.compile(
                "t%03dxy|t%04dxy" % (t_end, t_end), re.IGNORECASE
            )  # account for 3 and 4 digit
            if re.search(string, ifile):
                found_files = found_files[:n]
                break

    # if user has specified only certain FOVs, filter for those
    if len(user_spec_fovs) > 0:
        information("Filtering TIFFs by FOV.")
        fitered_files = []
        for fov_id in user_spec_fovs:
            fov_string = re.compile("xy%02d|xy%03d" % (fov_id, fov_id), re.IGNORECASE)
            fitered_files += [
                ifile for ifile in found_files if re.search(fov_string, ifile)
            ]

        found_files = fitered_files[:]

    return found_files


def slice_channels(
    channel_masks: dict,
    analyzed_imgs: dict,
    user_spec_fovs: list,
    ana_dir: Path,
    phase_plane: str,
    image_orientation: str,
    chnl_dir: Path,
    experiment_name: str,
):
    """Loops over FOVs and slices individual traps for analysis.

    Parameters:
    ------------
    channel_masks: dict
        dictionary of trap positions for slicing.

    analyzed_imgs
        dictionary of image filenames and parameters

    user_spec_fovs
        FOVs to analyze

    ana_dir: Path
        analysis directory path

    phase_plane: str
        phase plane channel identifier"""

    # do it by FOV. Not set up for multiprocessing
    for fov, peaks in six.iteritems(channel_masks):

        # skip fov if not in the group
        if user_spec_fovs and fov not in user_spec_fovs:
            continue

        information("Loading images for FOV %03d." % fov)

        # get filenames just for this fov along with the julian date of acquisition
        send_to_write = [
            [k, v["t"]] for k, v in six.iteritems(analyzed_imgs) if v["fov"] == fov
        ]

        # sort the filenames by jdn
        send_to_write = progress(sorted(send_to_write, key=lambda time: time[1]))

        # This is for loading the whole raw tiff stack and then slicing through it
        tiff_stack_slice_and_write(
            ana_dir,
            send_to_write,
            channel_masks,
            analyzed_imgs,
            phase_plane,
            image_orientation,
            chnl_dir,
            experiment_name,
        )

    information("Channel slices saved.")


def compile(
    TIFF_dir: Path,
    num_analyzers: int,
    ana_dir: Path,
    t_start: int,
    t_end: int,
    image_orientation: str,
    channel_width: int,
    channel_separation: int,
    channel_detection_snr: float,
    channel_length_pad: int,
    channel_width_pad: int,
    alignment_pad: int,
    do_metadata: bool,
    do_time_table: bool,
    do_channel_masks: bool,
    do_slicing: bool,
    do_crosscorrs: bool,
    experiment_name: str,
    phase_plane: str,
    FOV: list,
    TIFF_source: str,
    use_jd: bool,
    chnl_dir: Path,
    seconds_per_time_index: int,
) -> None:
    """
    Compile function for the MM3 analysis pipeline. This function is the main entry point for the analysis pipeline.

    Parameters
    ----------
    TIFF_dir : Path
        Path to the directory containing the TIFF files.
    num_analyzers : int
        Number of threads to use for multiprocessing.
    ana_dir : Path
        Path to the directory where the analysis files will be saved.
    t_start : int
        Start time for analysis.
    t_end : int
        End time for analysis.
    image_orientation : str
        Orientation of the images ('auto', 'up', 'down').
    channel_width : int
        Width of the channels in pixels.
    channel_separation : int
        Separation between channels in pixels.
    channel_detection_snr : float
        Signal to noise ratio for peak detection.
    channel_length_pad : int
        Padding for the channel length.
    channel_width_pad : int
        Padding for the channel width.
    alignment_pad : int
        Padding for alignment.
    do_metadata : bool
        If True, the metadata will be loaded from the analysis directory.
    do_time_table : bool
        If True, the time table will be created.
    do_channel_masks : bool
        If True, the channel masks will be created.
    do_slicing : bool
        If True, the channels will be sliced.
    do_crosscorrs : bool
        If True, the cross-correlations will be computed.
    experiment_name : str
        Name of the experiment.
    phase_plane : str
        Phase plane channel identifier.
    FOV : list
        List of FOVs to analyze.
    TIFF_source : str
        Source of the TIFF files ('BioFormats / other TIFF').
    use_jd : bool
        If True, the Julian date will be used for the time table.
    chnl_dir : Path
        Path to the directory where the channel files will be saved.
    seconds_per_time_index : int
        Time interval in seconds between consecutive imaging rounds.

    Returns
    -------
    None
    """
    information("Loading experiment parameters.")
    user_spec_fovs = FOV
    information("Using {} threads for multiprocessing.".format(num_analyzers))

    if not os.path.exists(ana_dir):
        os.makedirs(ana_dir)

    if not os.path.exists(chnl_dir):
        os.makedirs(chnl_dir)

    if TIFF_source == "BioFormats / other TIFF":
        information("Checking if imaging channels are separated")
        found_files = list(TIFF_dir.glob("*.tif"))
        found_files = sorted(found_files)

        files_list = []
        i = 0
        while True:
            c_string = re.compile(f"c(0)*{i}[._]", re.IGNORECASE)
            matched_files = [f for f in found_files if re.search(c_string, f.name)]
            if matched_files:
                files_list.append(matched_files)
                i += 1
            elif i == 0:
                # continue in case channels indexed from 1
                i += 1
            else:
                break

        files_array = np.array(files_list).squeeze()
        if files_array.ndim > 1:
            information("Merging TIFFs across channel")
            stack_channels(files_array, TIFF_dir)

        else:
            pass

    if not do_metadata:
        information("Loading image parameters dictionary.")
        with open(os.path.join(ana_dir, "TIFF_metadata.pkl"), "rb") as tiff_metadata:
            analyzed_imgs = pickle.load(tiff_metadata)
    else:
        information("Finding image parameters.")
        found_files = TIFF_dir.glob("*.tif")
        found_files = filter_files(found_files, t_start, t_end, user_spec_fovs)
        if len(found_files) > 0:
            analyzed_imgs = get_tif_params_loop(
                num_analyzers,
                TIFF_dir,
                found_files,
                TIFF_source,
                channel_width,
                channel_separation,
                channel_width_pad,
                channel_detection_snr,
                phase_plane,
                image_orientation,
            )
        else:
            information("No files found.")
            return

        information("Saving metadata from analyzed images...")
        with open(os.path.join(ana_dir, "TIFF_metadata.pkl"), "wb") as tiff_metadata:
            pickle.dump(analyzed_imgs, tiff_metadata, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(ana_dir, "TIFF_metadata.txt"), "w") as tiff_metadata:
            pprint(analyzed_imgs, stream=tiff_metadata)
        information("Saved metadata from analyzed images.")

    if do_time_table:
        time_table = make_time_table(
            analyzed_imgs, use_jd, seconds_per_time_index, ana_dir
        )

    if do_channel_masks:
        channel_masks = make_masks(
            analyzed_imgs,
            t_start,
            t_end,
            ana_dir,
            channel_width,
            channel_width_pad,
            channel_length_pad,
        )
    else:
        channel_masks = load_channel_masks(ana_dir)

    if do_slicing:
        information("Saving channel slices.")

        slice_channels(
            channel_masks,
            analyzed_imgs,
            user_spec_fovs,
            ana_dir,
            phase_plane,
            image_orientation,
            chnl_dir,
            experiment_name,
        )

    if do_crosscorrs:
        compute_xcorr(
            channel_masks,
            user_spec_fovs,
            num_analyzers,
            alignment_pad,
            ana_dir,
            experiment_name,
            phase_plane,
        )


def load_fov(
    image_directory: Path, fov_id: int, filter_str: str = ""
) -> Union[np.ndarray, None]:
    """
    Load a single FOV from a directory of TIFF files.

    Parameters
    ----------
    image_directory : Path
        Path to the directory containing the TIFF files.
    fov_id : int
        FOV ID to load.
    filter_str : str
        Filter string to apply to the filenames.

    Returns
    -------
    np.ndarray
        Array of images for the specified FOV.
    """

    information("getting files")
    found_files_paths = image_directory.glob("*.tif")
    file_string = re.compile(f"xy0*{fov_id}\w*.tif", re.IGNORECASE)
    found_files = [f.name for f in found_files_paths if re.search(file_string, f.name)]
    if filter_str:
        found_files = [
            f for f in found_files if re.search(filter_str, f, re.IGNORECASE)
        ]

    information("sorting files")
    found_files = sorted(found_files)  # should sort by timepoint

    if len(found_files) == 0:
        information("No data found for FOV " + str(fov_id))
        return None

    image_fov_stack = []

    information("Loading files")
    for img_filename in found_files:
        with tiff.TiffFile(image_directory / img_filename) as tif:
            image_fov_stack.append(tif.asarray())

    information("numpying files")
    return np.array(image_fov_stack, dtype=object)


class Compile(MM3Container):
    def __init__(self, napari_viewer: Viewer):
        super().__init__(napari_viewer=napari_viewer, validate_folders=False)

    def create_widgets(self):
        """Override method. Serves as the widget constructor. See MM3Container for more details."""
        self.viewer.text_overlay.visible = False

        self.fov_widget = FOVChooser(self.valid_fovs)
        # TODO: Auto-infer?
        self.image_source_widget = ComboBox(
            label="image source",
            choices=["nd2", "BioFormats / other TIFF"],
        )
        self.split_channels_widget = CheckBox(
            label="separate image plane files",
            tooltip="Check this box if you have separate tiffs for phase / fluorescence channels. Used for display only.",
        )
        self.phase_plane_widget = PlanePicker(
            self.valid_planes, label="phase plane channel"
        )
        self.time_range_widget = TimeRangeSelector(self.valid_times)
        self.seconds_per_frame_widget = SpinBox(
            value=150,
            label="seconds per frame",
            tooltip="Required if TIFF source is not .nd2. Time interval in seconds "
            + "between consecutive imaging rounds.",
            min=1,
            max=60 * 60 * 24,
        )

        self.channel_orientation_widget = ComboBox(
            label="trap orientation", choices=["auto", "up", "down"]
        )

        self.channel_width_widget = SpinBox(
            value=10,
            label="channel width",
            tooltip="Required. Approx. width of traps in pixels.",
            min=1,
            max=10000,
        )
        self.channel_separation_widget = SpinBox(
            value=45,
            label="channel separation",
            tooltip="Required. Center-to-center distance between traps in pixels.",
            min=1,
            max=10000,
        )
        self.xcorr_threshold_widget = FloatSpinBox(
            label="cross correlation threshold",
            value=0.97,
            tooltip="Recommended. Threshold for designating channels as full /empty"
            + "based on time correlation of trap intensity profile. "
            + "Traps above threshold will be set as empty.",
            step=0.01,
            min=0,
            max=1,
        )

        self.inspect_widget = PushButton(text="visualize all FOVs (from .nd2)")

        self.fov_widget.connect_callback(self.set_fovs)
        self.image_source_widget.changed.connect(self.set_image_source)
        self.split_channels_widget.changed.connect(self.set_split_channels)
        self.phase_plane_widget.changed.connect(self.set_phase_plane)
        self.time_range_widget.changed.connect(self.set_range)
        self.seconds_per_frame_widget.changed.connect(self.set_seconds_per_frame)
        self.channel_width_widget.changed.connect(self.set_channel_width)
        self.channel_separation_widget.changed.connect(self.set_channel_separation)
        self.inspect_widget.clicked.connect(self.display_all_fovs)
        self.channel_orientation_widget.changed.connect(self.set_channel_orientation)

        self.append(self.fov_widget)
        self.append(self.image_source_widget)
        self.append(self.split_channels_widget)
        self.append(self.phase_plane_widget)
        self.append(self.time_range_widget)
        self.append(self.seconds_per_frame_widget)
        self.append(self.channel_orientation_widget)
        self.append(self.channel_width_widget)
        self.append(self.channel_separation_widget)
        self.append(self.inspect_widget)

        self.set_image_source()
        self.set_split_channels()
        self.set_phase_plane()
        self.set_fovs(self.valid_fovs)
        self.set_range()
        self.set_seconds_per_frame()
        self.set_channel_width()
        self.set_channel_separation()
        self.set_channel_orientation()

        self.display_single_fov()

    def run(self):
        """Overriding method. Performs Mother Machine Analysis"""
        self.viewer.window._status_bar._toggle_activity_dock(True)

        compile(
            TIFF_dir=self.TIFF_folder,
            num_analyzers=multiprocessing.cpu_count(),
            ana_dir=self.analysis_folder,
            t_start=self.time_range[0],
            t_end=self.time_range[1] + 1,
            image_orientation=self.channel_orientation,
            channel_width=self.channel_width,
            channel_separation=self.channel_separation,
            channel_detection_snr=1,
            channel_length_pad=10,
            channel_width_pad=10,
            alignment_pad=10,
            do_metadata=True,
            do_time_table=True,
            do_channel_masks=True,
            do_slicing=True,
            do_crosscorrs=True,
            experiment_name=self.experiment_name,
            phase_plane=self.phase_plane,
            FOV=self.fovs,
            TIFF_source=self.image_source,
            use_jd=False,  # disabling for now because of bug with Elements output formatting
            chnl_dir=self.analysis_folder / "channels",
            seconds_per_time_index=self.seconds_per_frame,
        )
        information("Finished.")

    def display_single_fov(self):
        self.viewer.layers.clear()
        self.viewer.text_overlay.visible = False
        if self.split_channels:
            image_fov_stack = load_fov(
                self.TIFF_folder, min(self.valid_fovs), filter_str="c*01"
            )
        else:
            image_fov_stack = load_fov(self.TIFF_folder, min(self.valid_fovs))
        image_fov_stack = np.squeeze(
            image_fov_stack
        )  ## remove axes of length 1 for napari viewer
        images = self.viewer.add_image(image_fov_stack.astype(np.float32))
        self.viewer.dims.current_step = (0, 0)
        images.reset_contrast_limits()
        # images.gamma = 0.5

    def display_all_fovs(self):
        pass
        # viewer = self.viewer
        # viewer.layers.clear()
        # viewer.grid.enabled = True

        # filepath = Path(".")
        # nd2file = list(filepath.glob("*.nd2"))[0]

        # if not nd2file:
        #     warning(
        #         f"Could not find .nd2 file to display in directory {filepath.resolve()}"
        #     )
        #     return

        # with nd2reader.reader.ND2Reader(str(nd2file)) as ndx:
        #     sizes = ndx.sizes

        #     if "t" not in sizes:
        #         sizes["t"] = 1
        #     if "z" not in sizes:
        #         sizes["z"] = 1
        #     if "c" not in sizes:
        #         sizes["c"] = 1
        #     ndx.bundle_axes = "zcyx"
        #     ndx.iter_axes = "t"
        #     n = len(ndx)

        #     shape = (
        #         sizes["t"],
        #         sizes["z"],
        #         sizes["v"],
        #         sizes["c"],
        #         sizes["y"],
        #         sizes["x"],
        #     )
        #     image = np.zeros(shape, dtype=np.float32)

        #     for i in range(n):
        #         image[i] = ndx.get_frame(i)

        # image = np.squeeze(image)

        # viewer.add_image(image, channel_axis=1, colormap="gray")
        # viewer.grid.shape = (-1, 3)

        # viewer.dims.current_step = (0, 0)
        # viewer.layers.link_layers()  ## allows user to set contrast limits for all FOVs at once

    def set_image_source(self):
        self.image_source = self.image_source_widget.value

    def set_phase_plane(self):
        self.phase_plane = self.phase_plane_widget.value

    # NOTE! This is different from the other functions in that it requires a parameter.
    def set_fovs(self, new_fovs):
        self.fovs = list(
            set(new_fovs)
        )  # set(new_fovs).intersection(set(self.valid_fovs))

    def set_range(self):
        self.time_range = (
            self.time_range_widget.start.value,
            self.time_range_widget.stop.value,
        )

    def set_seconds_per_frame(self):
        self.seconds_per_frame = self.seconds_per_frame_widget.value

    def set_channel_width(self):
        self.channel_width = self.channel_width_widget.value

    def set_channel_separation(self):
        self.channel_separation = self.channel_separation_widget.value

    def set_channel_orientation(self):
        self.channel_orientation = self.channel_orientation_widget.value

    def set_split_channels(self):
        self.split_channels = self.split_channels_widget.value
        self.display_single_fov()
