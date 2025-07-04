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
from typing import Union, Optional

from scipy import ndimage as ndi
from skimage.feature import match_template
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint
from scipy.ndimage import rotate
from scipy.signal import find_peaks_cwt
from magicgui.widgets import (
    SpinBox,
    PushButton,
    ComboBox,
    CheckBox,
    Slider,
)
from napari import Viewer
from napari.utils import progress
from ._deriving_widgets import (
    MM3Container,
    FOVChooser,
    TimeRangeSelector,
    PlanePicker,
    information,
    warning,
    load_tiff,
)

from .utils import TIFF_FILE_FORMAT_PEAK


## IMAGE ORDER: FOV, phase plane, time, y, x. THIS IS NON-NEGOTIABLE


#### Helpful utility functions.
def get_plane(filepath: str) -> Optional[str]:
    """Extracts the plane / channel number (e.g. phase fluorescence etc.) from a tiff file name.
    It is used to sort the tiff files into the correct order for processing.
    """
    pattern = r"(c\d+).tif"
    res = re.search(pattern, filepath, re.IGNORECASE)
    if res:
        return res.group(1)
    return None


def get_fov(filepath: str) -> Optional[int]:
    """Extracts the fov number from a tiff file name."""
    pattern = r"xy(\d+)\w*.tif"
    res = re.search(pattern, filepath, re.IGNORECASE)
    if res != None:
        return int(res.group(1))
    return None


def get_time(filepath: str) -> Optional[int]:
    """Extracts the time point from a tiff file name."""
    pattern = r"t(\d+)\w*.tif"
    res = re.search(pattern, filepath, re.IGNORECASE)
    if res != None:
        return int(res.group(1))
    return None


def merge_split_channels(TIFF_dir: Path) -> None:
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

    files_array = np.array(files_list).squeeze()  # type:ignore
    if files_array.ndim > 1:
        information("Merging TIFFs across channel")
        stack_channels(files_array, TIFF_dir)

    else:
        pass


def stack_channels(found_files: np.ndarray, TIFF_dir: Path) -> None:
    """Check if phase and fluorescence channels are separate tiffs.
    If so, restack them as one tiff file and save out to TIFF_dir.
    Move the original files to a new directory original_TIFF.

    Parameters
    ---------
    found_files: ndarray of filepaths for each imaging plane
    TIFF_dir: Path
        Directory containing the TIFF files.

    """

    for files in found_files:
        information("Merging files")
        print(*files, sep="\n")
        ims = [tiff.imread(f) for f in files]
        im_out = np.stack(ims, axis=0)

        # need to insert regex here to catch variants
        name_out = re.sub(r"c\d+", "", str(files[0]), flags=re.IGNORECASE)
        # 'minisblack' necessary to ensure that it interprets image as black/white.
        tiff.imwrite(name_out, im_out, photometric="minisblack")

        old_tiff_path = TIFF_dir.parent / "original_TIFF"
        if not old_tiff_path.exists():
            os.makedirs(old_tiff_path)

        for f in files:
            os.rename(f, old_tiff_path / Path(f).name)


def fix_rotation(angle: float, image_data: np.ndarray) -> np.ndarray:
    print("rotating")
    if angle == 0:
        return image_data

    if len(image_data.shape) == 2:
        image_data = np.expand_dims(image_data, 0)

    return rotate(image_data, angle, axes=(2, 1))


# define function for flipping the images on an FOV by FOV basis
def fix_orientation(
    image_data: np.ndarray, phase_idx: int, image_orientation: str
) -> np.ndarray:
    """
    Fix the orientation. The standard direction for channels to open to is down.
    """

    image_data = np.squeeze(image_data)  # remove singleton dimensions

    # if this is just a phase image give in an extra layer so rest of code is fine
    flat = False
    if len(image_data.shape) == 2:
        image_data = np.expand_dims(image_data, 0)
        flat = True

    if image_orientation == "up":
        return image_data[:, ::-1, :]
    elif image_orientation == "down":
        pass
    elif image_orientation == "auto":
        # flip based on the index of the highest average row value
        brightest_row = np.argmax(image_data[phase_idx].mean(axis=1))
        midline = image_data[phase_idx].shape[0] / 2
        if brightest_row < midline:
            image_data = image_data[:, ::-1, :]
        else:
            pass

    # just return that first layer if it's that single phase layer.
    if flat:
        image_data = image_data[0]

    return image_data


def find_phase_idx(image_data: np.ndarray, phase_plane: str):
    # use 'phase_plane' to find the phase plane in image_data, assuming c1, c2, c3... naming scheme here.
    try:
        return int(re.search("[0-9]", phase_plane).group(0)) - 1  # type:ignore
    except:
        # Pick the plane to analyze with the highest mean px value (should be phase)
        average_channel_brightness = np.mean(image_data, axis=range(1, image_data.ndim))
        return np.argmax(average_channel_brightness)


# finds the location of channels in a tif
def find_channel_locs(
    phase_data: np.ndarray,
    channel_width_pad: int,
    channel_width: int,
    channel_detection_snr: float,
    channel_separation: int,
) -> dict:
    """Finds the location of channels from a phase contrast image. The channels are returned in
    a dictionary where the key is the x position of the channel in pixel and the value is a
    dictionary with the open and closed end in pixels in y.

    chnl_loc_dict : dict
        Dictionary with the channel locations.
    """

    crop_wp = int(channel_width_pad + channel_width / 2)
    projection_x = phase_data.sum(axis=0).astype(np.int32)
    peaks = find_peaks_cwt(
        projection_x,
        np.arange(channel_width - 5, channel_width + 5),
        min_snr=channel_detection_snr,  # type: ignore
    )

    # if the first or last peaks are close to the margins, ignore them.
    image_height, image_width = phase_data.shape[0], phase_data.shape[1]
    if peaks[0] < (channel_separation / 2):
        peaks = peaks[1:]
    if image_width - peaks[-1] < (channel_separation / 2):
        peaks = peaks[:-1]

    # assume the closed end is in the upper third and open in
    # the lower third, respectively. Find them.
    projection_y = phase_data.sum(axis=1)
    proj_diff = np.diff(projection_y.astype(np.int32))
    onethirdpoint = int(image_height / 3.0)
    twothirdpoint = int(image_height * 2.0 / 3.0)
    default_closed_end = proj_diff[:onethirdpoint].argmax()
    default_open_end = twothirdpoint + proj_diff[twothirdpoint:].argmin()
    default_length = default_open_end - default_closed_end

    chnl_loc_dict = {}
    for peak in peaks:
        chnl_loc_dict[peak] = {
            "closed_end_px": default_closed_end,
            "open_end_px": default_open_end,
        }
        channel_slice = phase_data[:, peak - crop_wp : peak + crop_wp]
        slice_projection_y = channel_slice.sum(axis=1)
        proj_diff = np.diff(slice_projection_y.astype(np.int32))
        slice_closed_end = proj_diff[:onethirdpoint].argmax()
        slice_open_end = twothirdpoint + proj_diff[twothirdpoint:].argmin()
        slice_length = slice_open_end - slice_closed_end

        image_height = phase_data.shape[0]
        if (
            abs(slice_length - default_length) <= 15
            and 15 <= slice_closed_end <= image_height - 15
            and 15 <= slice_open_end <= image_height - 15
        ):
            chnl_loc_dict[peak] = {
                "closed_end_px": slice_closed_end,
                "open_end_px": slice_open_end,
            }

    return chnl_loc_dict


def get_tif_metadata_filename(tif: tiff.TiffFile) -> dict:
    return {
        "fov": get_fov(tif.filename),  # fov id
        "t": get_time(tif.filename),  # time point
        "jd": -1 * 0.0,  # absolute julian time
        "planes": get_plane(tif.filename),
    }


def get_tif_metadata_nd2(tif: tiff.TiffFile) -> dict:
    """
    Run when TIFF files originated with an nd2.
    All the metdata is found in Nd2ToTIFF and saved in json format to the tiff; it is simply extracted here

    Returns
    -------
    idata: dict
        dictionary of values:
            'fov': int,
            't' : int,
            'planes' (list of strings)

    """
    # get the first page of the tiff and pull out image description
    for tag in tif.pages[0].tags:
        if tag.name == "ImageDescription":
            idata = tag.value
            break

    idata = json.loads(idata)
    return idata


def analyze_image(
    tif,
    TIFF_source: str,
    phase_plane: str,
    image_orientation: str,
    image_rotation: float,
    channel_width_pad: int,
    channel_width: int,
    channel_detection_snr: float,
    channel_separation: int,
):
    if TIFF_source == "nd2":
        image_metadata = get_tif_metadata_nd2(tif)
    elif TIFF_source == "BioFormats / other TIFF":
        image_metadata = get_tif_metadata_filename(tif)

    # fix the image orientation and get the number of planes
    image_data = tif.asarray()
    phase_idx = int(find_phase_idx(image_data, phase_plane))
    image_data = fix_rotation(image_rotation, image_data)
    image_data = fix_orientation(image_data, phase_idx, image_orientation)

    img_shape = [image_data.shape[0], image_data.shape[1]]  # type: ignore

    image_metadata = {
        "fov": image_metadata["fov"],
        "t": image_metadata["t"],
        "planes": image_metadata["planes"],
        "shape": img_shape,
    }

    # if the image data has more than 1 plane restrict image_data to phase,
    # which should have highest mean pixel data
    if len(image_data.shape) > 2:
        ph_index = int(phase_plane[1:]) - 1
        phase_data = image_data[ph_index]

    # find channels on the processed image
    image_metadata["chnl_loc_dict"] = find_channel_locs(
        phase_data,
        channel_width_pad,
        channel_width,
        channel_detection_snr,
        channel_separation,
    )

    return image_metadata


def get_tif_params(
    image_path: Path,
    TIFF_source: str,
    phase_plane: str,
    image_orientation: str,
    image_rotation: float,
    channel_width_pad: int,
    channel_width: int,
    channel_detection_snr: float,
    channel_separation: int,
) -> dict:
    """Loads a tiff file, pulls out the image data, and the metadata,
    including the location of the channels if flagged.

    it returns a dictionary like this for each image:

    'filename': image_filename,
    'fov' : image_metadata['fov'], # fov id
    't' : image_metadata['t'], # time point
    'plane_names' : image_metadata['plane_names'] # list of plane names
    'channels': cp_dict, # dictionary of channel locations, in the case of
                Unet-based channel segmentation, it's a dictionary of channel
                labels
    """

    try:
        with tiff.TiffFile(image_path) as tif:
            tif_params = analyze_image(
                tif,
                TIFF_source,
                phase_plane,
                image_orientation,
                image_rotation,
                channel_width_pad,
                channel_width,
                channel_detection_snr,
                channel_separation,
            )

            information("Analyzed %s" % image_path.name)
            tif_params["filepath"] = image_path

            return tif_params
    except:
        warning(f"Failed get_params for {image_path.name}")
        information(sys.exc_info()[0])
        information(sys.exc_info()[1])
        information(traceback.print_tb(sys.exc_info()[2]))
        return {
            "filepath": image_path,
            "analyze_success": False,
        }


def get_tif_params_loop(
    found_files: list,
    num_analyzers: int,
    TIFF_source: str,
    phase_plane: str,
    image_orientation: str,
    image_rotation: float,
    channel_width_pad: int,
    channel_width: int,
    channel_detection_snr: float,
    channel_separation: int,
) -> dict:
    """Loop over found files and extract image parameters.

    Parameters
    ----------
    num_analyzers: int
        Number of analyzers to use for multiprocessing.
    found_files: list
        List of tiff files to analyze.

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
                fn,
                TIFF_source,
                phase_plane,
                image_orientation,
                image_rotation,
                channel_width_pad,
                channel_width,
                channel_detection_snr,
                channel_separation,
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


def channel_xcorr(img_path: Path, pad_size) -> list:
    """
    Function calculates the cross correlation of images in a
    stack to the first image in the stack. 
    The very first value should be all 1s.


    Returns
    -------
    xcorr_array: list
        array of cross correlations over time
    """

    number_of_images = 20
    # switch postfix to c1/c2/c3 auto??
    image_data = load_tiff(img_path)

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
    analysis_dir: Path,
    experiment_name: str,
    phase_plane: str,
    channel_masks: dict,  # Trap locations relative to FOV
    user_spec_fovs: list,  # user specified fovs
    num_analyzers: int,
    alignment_pad: int,
) -> dict:
    """
    Compute time autocorrelation for each channel
    """

    crosscorrs = {}

    for fov_id in progress(user_spec_fovs):
        information("Calculating cross correlations for FOV %d." % fov_id)
        crosscorrs[fov_id] = {}
        pool = Pool(num_analyzers)

        for peak_id in sorted(channel_masks[fov_id].keys()):
            information("Calculating cross correlations for peak %d." % peak_id)
            # currently broken:
            # channel_xcorr(ana_dir, experiment_name, fov_id: int, peak_id: int, phase_plane, pad_size) -> list:
            img_filename = TIFF_FILE_FORMAT_PEAK % (
                experiment_name,
                fov_id,
                peak_id,
                phase_plane,
            )
            img_path = analysis_dir / "channels" / img_filename
            crosscorrs[fov_id][peak_id] = pool.apply_async(  # type:ignore
                channel_xcorr, args=(img_path, alignment_pad)
            )

        information("Waiting for cross correlation pool to finish for FOV %d." % fov_id)

        pool.close()  # tells the process nothing more will be added.
        pool.join()  # blocks script until everything has been processed and workers exit

        information("Finished cross correlations for FOV %d." % fov_id)

    for fov_id, peaks in crosscorrs.items():
        for peak_id, result in peaks.items():
            if result.successful():  # type:ignore
                crosscorrs[fov_id][peak_id] = {
                    "ccs": result.get(),  # type: ignore
                    "cc_avg": np.average(result.get()),  # type:ignore
                }

            else:
                crosscorrs[fov_id][peak_id] = False  # type:ignore

    information("Writing cross correlations file.")
    with open(analysis_dir / "crosscorrs.pkl", "wb") as xcorrs_file:
        pickle.dump(crosscorrs, xcorrs_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return crosscorrs


def load_channel_masks(ana_dir: Path) -> dict:
    """Load channel masks dictionary. Should be .yaml but try pickle too.

    Returns
    -------
    channel_masks: dict
        dictionary of channel masks
    """
    information("Loading channel masks dictionary.")

    # try loading from .yaml before .pkl
    try:
        information("Path:", ana_dir / "channel_masks.yaml")
        with (ana_dir / "channel_masks.yaml").open("r") as cmask_file:
            channel_masks = yaml.safe_load(cmask_file)
    except:
        warning("Could not load channel masks dictionary from .yaml.")

        try:
            information("Path:", ana_dir / "channel_masks.pkl")
            with (ana_dir / "channel_masks.pkl").open("rb") as cmask_file:
                channel_masks = pickle.load(cmask_file)
        except ValueError:
            warning("Could not load channel masks dictionary from .pkl.")

    return channel_masks


def make_consensus_mask(
    analyzed_imgs,
    fov: int,
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

        img_chnl_mask = np.zeros([image_rows, image_cols])

        # and add the channel mask to it
        for chnl_peak, peak_ends in img_v["channels"].items():
            # expand by padding (more padding done later for width)
            x1 = max(chnl_peak - crop_wp, 0)
            x2 = min(chnl_peak + crop_wp, image_cols)
            y1 = max(peak_ends["closed_end_px"] - chan_lp, 0)
            y2 = min(peak_ends["open_end_px"] + chan_lp, image_rows)

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
    consensus_mask = ndi.label(consensus_mask)[0]  # type: ignore

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
    chnl_mask: list,
    cm_copy: list,
    max_len: int,
    max_wid: int,
    image_cols: int,
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


def make_masks(
    analyzed_imgs: dict,
    channel_width_pad: int = 0,
    channel_width: int = 0,
    channel_length_pad: int = 0,
) -> dict:
    """
    Make masks goes through the channel locations in the image metadata and builds a consensus
    Mask for each image per fov, which it returns as dictionary named channel_masks.
    The keys in this dictionary are fov id, and the values is another dictionary.
    This dict's keys are channel locations (peaks) and the values is a [2][2] array:
    [[minrow, maxrow],[mincol, maxcol]] of pixel locations designating the corner of each mask
    for each channel on the whole image

    One important consequence of these function is that the channel ids and the size of the
    channel slices are decided now. Updates to mask must coordinate with these values.

    Returns
    -------
    channel_masks : dict
        dictionary of consensus channel masks.

    """
    information("Determining initial channel masks...")

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
            analyzed_imgs, fov, image_rows, image_cols, crop_wp, chan_lp
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

    information("Channel masks saved.")
    return cm_copy


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
            channel_loc[0][0] : channel_loc[0][1],
            channel_loc[1][0] : channel_loc[1][1],
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
            f"Image shape not recognized. Expected 2, 3, or 4 dimensions, "
            "found {image_data.ndim} dimensions with shape {image_data.shape}."
        )

    # slice based on appropriate slicer object.
    channel_slice = image_data[channel_slicer]

    # pad y of channel if slice happened to be outside of image
    y_difference = (channel_loc[0][1] - channel_loc[0][0]) - channel_slice.shape[1]
    if y_difference > 0:
        paddings = [[0, 0], [0, y_difference], [0, 0], [0, 0]]  # t  # y  # x  # c
        channel_slice = np.pad(channel_slice, paddings, mode="edge")

    return channel_slice


def tiff_stack_slice_and_write(
    images_to_write: list,
    channel_masks_fov: dict,
    experiment_name: str,
    channel_dir: Path,
    analyzed_imgs: dict,
    phase_plane: str,
    image_orientation: str,
    image_rotation: float,
) -> None:
    """Writes out 4D stacks of TIFF images per channel.
    Loads all tiffs from and FOV into memory and then slices all time points at once.

    Parameters
    ----------
    images_to_write: list
        list of images to write

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
        phase_idx = int(find_phase_idx(image_data, phase_plane))
        image_data = fix_orientation(image_data, phase_idx, image_orientation)
        image_data = fix_rotation(image_rotation, image_data)
        # add additional axis if the image is flat
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, 0)
        # change axis so it goes Y, X, Plane
        image_data = np.rollaxis(image_data, 0, 3)
        image_fov_stack.append(image_data)

    # concatenate the list into one big ass stack
    image_fov_array = np.stack(image_fov_stack, axis=0)

    # cut out the channels as per channel masks for this fov
    for peak, channel_loc in six.iteritems(channel_masks_fov):
        information("Slicing and saving channel peak %d." % peak)

        # slice out channel.
        # The function should recognize the shape length as 4 and cut all time points
        channel_stack = cut_slice(image_fov_array, channel_loc)

        for color_index in range(channel_stack.shape[3]):
            # save stack
            # this is the filename for the channel
            channel_filename = channel_dir / (
                f"{experiment_name}_xy{fov_id:03d}_p{peak:04d}_c{color_index + 1}.tif"
            )
            tiff.imwrite(
                channel_filename,
                channel_stack[:, :, :, color_index],
                compression=("zlib", 4),  # type: ignore
            )


def slice_channels(
    channel_masks: dict,
    experiment_name: str,
    channel_dir: Path,
    analyzed_imgs: dict,
    phase_plane: str,
    image_orientation: str,
    image_rotation: float,
):
    """Loops over FOVs and slices individual traps for analysis.

    Parameters:
    ------------
    user_spec_fovs
        FOVs to analyze
    """

    # do it by FOV. Not set up for multiprocessing
    for fov in channel_masks.keys():
        information("Loading images for FOV %03d." % fov)

        # get filenames just for this fov along with the julian date of acquisition
        send_to_write = [
            [k, v["t"]] for k, v in analyzed_imgs.items() if v["fov"] == fov
        ]

        # sort the filenames by time
        send_to_write = sorted(send_to_write, key=lambda time: time[1])

        # This is for loading the whole raw tiff stack and then slicing through it
        tiff_stack_slice_and_write(
            send_to_write,  # type:ignore
            channel_masks[fov],
            experiment_name,
            channel_dir,
            analyzed_imgs,
            phase_plane,
            image_orientation,
            image_rotation,
        )

    information("Channel slices saved.")


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
    found_files = sorted(found_files)  # should sort by timepoint

    # keep images starting at this timepoint
    if t_start:
        information("Removing images before time {}".format(t_start))
        # go through list and find first place where timepoint is equivalent to t_start
        for n, fpath in enumerate(found_files):
            string = re.compile(
                "t{:0=3}xy|t{:0=4}xy".format(t_start, t_start), re.IGNORECASE
            )  # account for 3 and 4 digit
            # if re.search == True then a match was found
            if re.search(string, fpath.name):
                # cut off every file name prior to this one and quit the loop
                found_files = found_files[n:]
                break

    # remove images after this timepoint
    if t_end:
        information("Removing images after time {}".format(t_end))
        # go through list and find first place where timepoint is equivalent to t_end
        for n, fpath in enumerate(found_files):
            string = re.compile(
                "t%03dxy|t%04dxy" % (t_end, t_end), re.IGNORECASE
            )  # account for 3 and 4 digit
            if re.search(string, fpath.name):
                found_files = found_files[:n]
                break

    # if user has specified only certain FOVs, filter for those
    if len(user_spec_fovs) > 0:
        information("Filtering TIFFs by FOV.")
        fitered_files = []
        for fov_id in user_spec_fovs:
            fov_string = re.compile("xy%02d|xy%03d" % (fov_id, fov_id), re.IGNORECASE)
            fitered_files += [
                fpath for fpath in found_files if re.search(fov_string, fpath.name)
            ]

        found_files = fitered_files[:]

    return found_files


def load_fov(
    image_directory: Path, fov_id: int, filter_str: str = ""
) -> Union[np.ndarray, None]:
    """
    Load a single FOV from a directory of TIFF files.
    """

    information("getting files")
    found_files_paths = image_directory.glob("*.tif")
    file_string = re.compile(rf"xy0*{fov_id}\w*.tif", re.IGNORECASE)
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


def compile(
    TIFF_dir: Path,
    num_analyzers: int,
    analysis_dir: Path,
    t_start: int,
    t_end: int,
    image_orientation: str,  # orientation \in ('auto', 'up', 'down')
    image_rotation: float,
    channel_width: int,
    channel_separation: int,
    channel_detection_snr: float,
    channel_length_pad: int,
    channel_width_pad: int,
    alignment_pad: int,
    do_metadata: bool,
    do_channel_masks: bool,
    do_slicing: bool,
    do_crosscorrs: bool,
    experiment_name: str,
    phase_plane: str,
    FOVs: list,
    TIFF_source: str,  # one of {'nd2', 'BioFormats / other TIFF'}
    chnl_dir: Path,
) -> None:
    """
    Compile function for the MM3 analysis pipeline. This function is the main entry point for the analysis pipeline.
    """
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    if not os.path.exists(chnl_dir):
        os.makedirs(chnl_dir)

    if TIFF_source == "BioFormats / other TIFF":
        merge_split_channels(TIFF_dir)

    if do_metadata:
        information("Finding image parameters.")
        found_files = list(TIFF_dir.glob("*.tif"))
        found_files = filter_files(found_files, t_start, t_end, FOVs)
        if len(found_files) == 0:
            return
        img_params = get_tif_params_loop(
            found_files,
            num_analyzers,
            TIFF_source,
            phase_plane,
            image_orientation,
            image_rotation,
            channel_width_pad,
            channel_width,
            channel_detection_snr,
            channel_separation,
        )
        information("Saving image parameters.")
        with open(analysis_dir / "TIFF_metadata.txt", "w") as tiff_metadata:
            pprint(img_params, stream=tiff_metadata)
    else:
        information("Loading image parameters dictionary.")
        # switch this to reading from the txt file.
        with open(analysis_dir / "TIFF_metadata.pkl", "rb") as tiff_metadata:
            analyzed_imgs_unfiltered = pickle.load(tiff_metadata)

        # now, remove all irrelevant images by fov and timestamp.
        img_params = {}
        for k, v in analyzed_imgs_unfiltered.items():
            if not (v["FOV"] in FOVs):
                continue
            if not (t_start <= v["t"] <= t_end):
                continue
            img_params[k] = v

    if do_channel_masks:
        channel_masks = make_masks(
            img_params, channel_width_pad, channel_width, channel_length_pad
        )
        # save the channel mask dictionary to a yaml and a text file
        with open(analysis_dir / "channel_masks.txt", "w") as cmask_file:
            pprint(channel_masks, stream=cmask_file)
    else:
        channel_masks = load_channel_masks(analysis_dir)

    if do_slicing:
        slice_channels(
            channel_masks,
            experiment_name,
            chnl_dir,
            img_params,
            phase_plane,
            image_orientation,
            image_rotation,
        )

    if do_crosscorrs:
        compute_xcorr(
            analysis_dir,
            experiment_name,
            phase_plane,
            channel_masks,
            FOVs,
            num_analyzers,
            alignment_pad,
        )


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
        self.image_rotation_widget = Slider(
            label="image rotation", value=0, min=-90, max=90, step=1
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
        self.image_rotation_widget.changed.connect(self.set_image_rotation)

        self.append(self.fov_widget)
        self.append(self.image_source_widget)
        self.append(self.split_channels_widget)
        self.append(self.phase_plane_widget)
        self.append(self.time_range_widget)
        self.append(self.seconds_per_frame_widget)
        self.append(self.channel_orientation_widget)
        self.append(self.image_rotation_widget)
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
        self.set_image_rotation()

        self.display_single_fov()

    def run(self):
        """Overriding method. Performs Mother Machine Analysis"""
        self.viewer.window._status_bar._toggle_activity_dock(True)

        compile(
            TIFF_dir=self.TIFF_folder,
            num_analyzers=multiprocessing.cpu_count(),
            analysis_dir=self.analysis_folder,
            t_start=self.time_range[0],
            t_end=self.time_range[1] + 1,
            image_orientation=self.channel_orientation,
            image_rotation=self.image_rotation,
            channel_width=self.channel_width,
            channel_separation=self.channel_separation,
            channel_detection_snr=1,
            channel_length_pad=10,
            channel_width_pad=10,
            alignment_pad=10,
            do_metadata=True,
            do_channel_masks=True,
            do_slicing=True,
            do_crosscorrs=False,
            experiment_name=self.experiment_name,
            phase_plane=self.phase_plane,
            FOVs=self.fovs,
            TIFF_source=self.image_source,
            chnl_dir=self.analysis_folder / "channels",
        )
        information("Finished.")

    def display_single_fov(self):
        self.viewer.layers.clear()
        self.viewer.text_overlay.visible = False
        if self.split_channels:
            image_fov_stack = load_fov(
                self.TIFF_folder, min(self.valid_fovs), filter_str="c0*1"
            )
        else:
            image_fov_stack = load_fov(self.TIFF_folder, min(self.valid_fovs))
        image_fov_stack = np.squeeze(image_fov_stack)  # type:ignore
        images = self.viewer.add_image(image_fov_stack.astype(np.float32))
        self.viewer.dims.current_step = (0, 0)
        images.reset_contrast_limits()  # type:ignore
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

    def set_image_rotation(self):
        self.image_rotation = self.image_rotation_widget.value
        print(self.image_rotation)
