"""
Splits image TIFFs into trap TIFFs.
Can be run headless with parameters; if left unspecified it will analyze the full time range and all FOVs,
and use the default UI parameters.
"""

import argparse
import multiprocessing
import os
import pickle
import re
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated, Optional, Union, overload

import numpy as np
import six
import tifffile as tiff
import yaml
from magicgui.widgets import Slider
from napari import Viewer
from napari.qt.threading import thread_worker
from napari.utils import progress
from scipy import ndimage as ndi
from scipy.ndimage import rotate
from scipy.signal import find_peaks_cwt
from skimage.feature import match_template

from ._deriving_widgets import (
    FOVList,
    MM3Container2,
    get_valid_fovs_folder,
    get_valid_planes,
    get_valid_times,
    information,
    load_tiff,
    range_string_to_indices,
    warning,
)
from .utils import TIFF_FILE_FORMAT_PEAK

## Refactor:
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
    if res is not None:
        return int(res.group(1))
    return None


def get_time(filepath: str) -> Optional[int]:
    """Extracts the time point from a tiff file name."""
    pattern = r"t(\d+)\w*.tif"
    res = re.search(pattern, filepath, re.IGNORECASE)
    if res is not None:
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
    # need to add support for no channels.
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
    except:  # noqa: E722
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
        crosscorrs[fov_id] = {}
        pool = Pool(num_analyzers)

        for peak_id in sorted(channel_masks[fov_id].keys()):
            print(
                f"Calculating cross correlations for fov {fov_id + 1} and peak {peak_id}",
                end="\r",
            )
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

    with open(analysis_dir / "crosscorrs.pkl", "wb") as xcorrs_file:
        pickle.dump(crosscorrs, xcorrs_file, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(analysis_dir / "crosscorrs.pkl", "wb") as xcorrs_file:
    #     json.dump(crosscorrs, xcorrs_file)

    return crosscorrs


def load_channel_masks(ana_dir: Path) -> dict:
    """
    Load channel masks dictionary. Should be .yaml but try pickle too.
    """
    information("Loading channel masks dictionary.")

    # try loading from .yaml before .pkl
    try:
        information("Path:", ana_dir / "channel_masks.yaml")
        with (ana_dir / "channel_masks.yaml").open("r") as cmask_file:
            channel_masks = yaml.safe_load(cmask_file)
    except:  # noqa: E722
        warning("Could not load channel masks dictionary from .yaml.")

        try:
            information("Path:", ana_dir / "channel_masks.pkl")
            with (ana_dir / "channel_masks.pkl").open("rb") as cmask_file:
                channel_masks = pickle.load(cmask_file)
        except ValueError:
            warning("Could not load channel masks dictionary from .pkl.")

    return channel_masks


def make_consensus_mask(
    phase_imgs: np.ndarray,
    chann_array: list[dict],  # time series of channel dicts
    crop_wp: int,  # channel with padding
    chan_lp: int,  # channel length padding
) -> np.ndarray:
    """
    Generate consensus channel mask for a given fov.
    Assume timestamps are pre-selected.

    Returns
    -------
    consensus_mask: np.ndarray
    """
    times, img_rows, img_cols = phase_imgs.shape
    consensus_mask = np.zeros([img_rows, img_cols])  # mask for labeling

    # bring up information for each image
    for cur_t_img in phase_imgs:
        img_chnl_mask = np.zeros([img_rows, img_cols])

        # and add the channel mask to it
        for t_chnls in chann_array:
            for chnl_peak, peak_ends in t_chnls.items():
                # expand by padding (more padding done later for width)
                x1 = max(chnl_peak - crop_wp, 0)
                x2 = min(chnl_peak + crop_wp, img_cols)
                y1 = max(peak_ends["closed_end_px"] - chan_lp, 0)
                y2 = min(peak_ends["open_end_px"] + chan_lp, img_rows)

                img_chnl_mask[y1:y2, x1:x2] = 1

            # add it to the consensus mask
            consensus_mask += img_chnl_mask

    # Normalize consensus mask between 0 and 1.
    consensus_mask = consensus_mask.astype("float32") / float(np.amax(consensus_mask))

    # label when value is above 0.1 (so 90% occupancy), transpose.
    # the [0] is for the array ([1] is the number of regions)
    # It transposes and then transposes again so regions are labeled left to right
    # clear border it to make sure the channels are off the edge
    consensus_mask = ndi.label(consensus_mask)[0]  # type: ignore

    return consensus_mask


def update_channel_masks(
    max_ch_l: int,
    max_ch_w: int,
    label_mask: np.ndarray,
    img_w: int,
    ch_masks: dict,  # dictionary of channel masks (per FOV)
) -> tuple[int, int, dict]:
    """
    Returns
    -------
    max_ch_l: int
        maximum channel length
    max_ch_w: int
        maximum channel width
    ch_masks: dict
        updated dictionary of channel masks for the FOV
    """

    # clean up the rough edges
    poscols = np.any(label_mask, axis=0)  # column positions where true (any)
    posrows = np.any(label_mask, axis=1)  # row positions where true (any)

    # channel_id given by horizontal position
    # later updates to the positions will have to check
    # if their channels contain this median value to match up
    channel_id = int(np.median(np.where(poscols)[0]))

    # store the edge locations of the channel mask in the dictionary. Will be ints
    min_row = np.min(np.where(posrows)[0])
    max_row = np.max(np.where(posrows)[0])
    min_col = np.min(np.where(poscols)[0])
    max_col = np.max(np.where(poscols)[0])

    if (0 < min_col) and (max_col < img_w):
        ch_masks[channel_id] = [
            [min_row, max_row],
            [min_col, max_col],
        ]

        # find the largest channel width and height while you go round
        max_ch_l = int(max(max_ch_l, max_row - min_row))
        max_ch_w = int(max(max_ch_w, max_col - min_col))

    return max_ch_l, max_ch_w, ch_masks


def adjust_channel_mask(
    chnl_mask_corners: list,  # indices of mask corners
    consensus_chnl_mask: list,
    max_len: int,
    max_wid: int,
    image_cols: int,
) -> list:
    """
    Expand channel mask to match maximum length and width.

    Returns
    -------
    ch_mask_copy: list
        updated consensus channel mask
    """
    # just add length to the open end (bottom of image, low column)
    if chnl_mask_corners[0][1] - chnl_mask_corners[0][0] != max_len:
        consensus_chnl_mask[0][1] = chnl_mask_corners[0][0] + max_len
    # enlarge widths around the middle, but make sure you don't get floats
    if chnl_mask_corners[1][1] - chnl_mask_corners[1][0] != max_wid:
        wid_diff = max_wid - (chnl_mask_corners[1][1] - chnl_mask_corners[1][0])
        if wid_diff % 2 == 0:
            consensus_chnl_mask[1][0] = max(chnl_mask_corners[1][0] - wid_diff / 2, 0)
            consensus_chnl_mask[1][1] = min(
                chnl_mask_corners[1][1] + wid_diff / 2, image_cols - 1
            )
        else:
            consensus_chnl_mask[1][0] = max(
                chnl_mask_corners[1][0] - (wid_diff - 1) / 2, 0
            )
            consensus_chnl_mask[1][1] = min(
                chnl_mask_corners[1][1] + (wid_diff + 1) / 2, image_cols - 1
            )

    consensus_chnl_mask = [
        list(map(int, i)) for i in consensus_chnl_mask
    ]  # make sure they are ints

    return consensus_chnl_mask


def make_masks(
    phase_image: np.ndarray,
    chnl_timeseries: list[dict],
    channel_width_pad: int = 0,
    channel_width: int = 0,
    channel_length_pad: int = 0,
) -> tuple[int, int, dict]:
    """
    The output dictionary's keys are peak numbers, and the values are channel locations in format
    [[minrow, maxrow], [mincol, maxcol]]
    One important consequence of these function is that the channel ids and the size of the
    channel slices are decided now. Updates to mask must coordinate with these values.

    Returns
    -------
    channel_masks : dict
        dictionary of consensus channel masks.

    """
    # declare temp variables from parameters.
    crop_wp = int(channel_width_pad + channel_width / 2)
    chan_lp = int(channel_length_pad)

    times, image_rows, image_cols = phase_image.shape

    # max width and length across all fovs. channels will get expanded by these values
    # this important for later updates to the masks, which should be the same
    max_len, max_wid = 0, 0

    consensus_mask = make_consensus_mask(phase_image, chnl_timeseries, crop_wp, chan_lp)

    # initialize dict which holds channel masks {peak : [[y1, y2],[x1,x2]],...}
    channel_masks = {}

    # go through each label
    for label in np.unique(consensus_mask):
        if label == 0:  # label zero is the background
            continue

        label_mask = consensus_mask == label
        max_len, max_wid, channel_masks = update_channel_masks(
            max_len, max_wid, label_mask, image_cols, channel_masks
        )

    # add channel_mask dictionary to the fov dictionary, use copy to play it safe
    return max_len, max_wid, channel_masks.copy()


def tiff_stack_slice_and_write(
    images_to_write: list,
    fov_id: int,
    channel_masks_fov: dict,
    experiment_name: str,
    channel_dir: Path,
) -> None:
    """Writes out 4D stacks of TIFF images per channel.
    Loads all tiffs from and FOV into memory and then slices all time points at once.

    Parameters
    ----------
    images_to_write: list
        list of images to write
    """
    # cut out the channels as per channel masks for this fov
    for peak, channel_corners in six.iteritems(channel_masks_fov):
        # slice out channel.
        # The function should recognize the shape length as 4 and cut all time points
        images_to_write = np.array(images_to_write)
        channel_stack = images_to_write[
            ...,
            channel_corners[0][0] : channel_corners[0][1],
            channel_corners[1][0] : channel_corners[1][1],
        ]
        # # pad y of channel if slice happened to be outside of image NOT SURE WHAT THIS IS FOR
        # y_difference = (channel_loc[0][1] - channel_loc[0][0]) - channel_slice.shape[-2]
        # if y_difference > 0:
        #     paddings = [[0, 0], [0, 0], [0, y_difference], [0, 0]]  # t  # y  # x  # c
        #     channel_slice = np.pad(channel_slice, paddings, mode="edge")

        channel_stack = channel_stack.squeeze()
        for color_index in range(channel_stack.shape[1]):
            # save stack
            # this is the filename for the channel
            channel_filename = channel_dir / (
                f"{experiment_name}_xy{fov_id:03d}_p{peak:04d}_c{color_index + 1}.tif"
            )
            tiff.imwrite(
                channel_filename,
                channel_stack[:, color_index, :, :],
                compression=("zlib", 4),  # type: ignore
            )


@thread_worker
def load_fov(
    image_directory: Path, fov_id: int, filter_str: str = ""
) -> Union[np.ndarray, None]:
    """
    Load a single FOV from a directory of TIFF files.
    """

    information("getting files")
    found_files_paths = list(image_directory.glob("*.tif"))
    get_fov_regex = re.compile(r"xy(\d+)", re.IGNORECASE)
    fovs = list(
        int(get_fov_regex.findall(filename.name)[0]) for filename in found_files_paths
    )
    found_files = zip(found_files_paths, fovs)
    found_files = [fpath.name for fpath, fov in found_files if fov == fov_id]
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
    return np.squeeze(np.array(image_fov_stack, dtype=np.int32))


def load_fov_quick(
    image_directory: Path, fov_id: int, filter_str: str = ""
) -> Union[np.ndarray, None]:
    """
    Load a single FOV from a directory of TIFF files.
    """

    information("getting files")
    found_files_paths = list(image_directory.glob("*.tif"))
    get_fov_regex = re.compile(r"xy(\d+)", re.IGNORECASE)
    fovs = list(
        int(get_fov_regex.findall(filename.name)[0]) for filename in found_files_paths
    )
    found_files = zip(found_files_paths, fovs)
    found_files = [fpath.name for fpath, fov in found_files if fov == fov_id]
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
    with tiff.TiffFile(image_directory / found_files[0]) as tif:
        image_fov_stack = tif.asarray()

    information("numpying files")
    return np.squeeze(np.array(image_fov_stack, dtype=np.int32))


class Orientation(Enum):
    auto = 1
    up = 2
    down = 3


@dataclass
class InPaths:
    """
    1. check folders for existence, fetch FOVs & times & planes
        -> upon failure, simply show a list of inputs + update button.
    """

    TIFF_dir: Annotated[Path, {"mode": "d"}] = Path(".") / "TIFF"


@dataclass
class OutPaths:
    experiment_name: str = ""
    channel_dir: Annotated[Path, {"mode": "d"}] = Path("./analysis/channels")
    analysis_dir: Annotated[Path, {"mode": "d"}] = Path("./analysis")


@dataclass
class RunParams:
    t_start: int
    t_end: int
    FOVs: FOVList
    phase_plane: str
    num_analyzers: int = multiprocessing.cpu_count()
    image_orientation: Orientation = Orientation.auto
    image_rotation: Annotated[int, {"max": 90, "min": -90, "widget_type": Slider}] = 0
    channel_width: int = 10
    channel_separation: int = 45
    channel_detection_snr: float = 1.0
    channel_length_pad: int = 10
    channel_width_pad: int = 10
    alignment_pad: int = 10
    TIFF_source: str = "nd2"  # one of {'nd2', 'BioFormats / other TIFF'}


def gen_default_run_params(in_files: InPaths):
    try:
        t_start, t_end = get_valid_times(in_files.TIFF_dir)
        all_fovs = get_valid_fovs_folder(in_files.TIFF_dir)
        # get the brightest channel as the default phase plane!
        channels = get_valid_planes(in_files.TIFF_dir)
        # move this into runparams somehow!
        params = RunParams(
            phase_plane=channels[0],
            t_start=t_start,
            t_end=t_end,
            FOVs=all_fovs,
        )
        params.__annotations__["phase_plane"] = Annotated[str, {"choices": channels}]
        return params
    except FileNotFoundError:
        raise FileNotFoundError("TIFF folder not found")
    except ValueError:
        raise ValueError(
            "Invalid filenames. Make sure that timestamps are denoted as t[0-9]* and FOVs as xy[0-9]*"
        )


def compile(in_paths: InPaths, p: RunParams, out_paths: OutPaths) -> None:
    """
    Finds channels, and slices them up into individual tiffs.
    """

    if not out_paths.analysis_dir.exists():
        out_paths.analysis_dir.mkdir()
    if not out_paths.channel_dir.exists():
        out_paths.channel_dir.mkdir()
    if p.TIFF_source == "BioFormats / other TIFF":
        merge_split_channels(p.TIFF_dir)

    # load in filtered FOVs and time ranges.
    information("Finding image parameters.")
    found_files = list(in_paths.TIFF_dir.glob("*.tif"))
    if len(found_files) == 0:
        return
    print(found_files)
    fov_to_files = {}
    for ff in found_files:
        fov, time = get_fov(ff.name), get_time(ff.name)
        if (not time) or (not fov):
            raise Exception()
        if (p.t_start <= time <= p.t_end) and (fov in p.FOVs):
            if fov in fov_to_files:
                fov_to_files[fov].append(ff)
            else:
                fov_to_files[fov] = [ff]

    all_channels = {}  # index in by fov, peak #
    for fov, paths in fov_to_files.items():
        print(f"analyzing FOV {fov + 1}", end="\n")
        chnl_timeseries = []
        img_timeseries = []
        sorted(paths)  # sort by timestamps
        for path in paths:
            time = get_time(ff.name)
            with tiff.TiffFile(path) as tif:
                image_data = tif.asarray().squeeze()
            # TODO: move this out of the loop
            phase_idx = int(find_phase_idx(image_data, p.phase_plane))

            if p.image_rotation != 0:
                image_data = fix_rotation(p.image_rotation, image_data)
            image_data = fix_orientation(
                image_data, phase_idx, p.image_orientation.name
            )
            phase_image = image_data[phase_idx]

            channel_locs = find_channel_locs(
                phase_image,
                p.channel_width_pad,
                p.channel_width,
                p.channel_detection_snr,
                p.channel_separation,
            )

            chnl_timeseries.append(channel_locs)
            img_timeseries.append(image_data)

        print(f"making masks for FOV {fov + 1}", end="\n")
        img_timeseries = np.array(img_timeseries)
        # maybe clean this up in a sec
        max_len, max_wid, channel_masks = make_masks(
            img_timeseries[:, phase_idx, :, :],
            chnl_timeseries,
            p.channel_width_pad,
            p.channel_width,
            p.channel_length_pad,
        )

        print(f"adjusting masks for FOV {fov + 1}", end="\n")
        all_channels[fov] = {}
        for peak, mask in channel_masks.items():
            img_width = phase_image.shape[1]
            all_channels[fov][peak] = adjust_channel_mask(
                mask, channel_masks[peak], max_len, max_wid, img_width
            )

        print(f"slicing channels for FOV {fov + 1}", end="\n")
        tiff_stack_slice_and_write(
            img_timeseries,
            fov,
            all_channels[fov],
            out_paths.experiment_name,
            out_paths.channel_dir,
        )
        print(f"finished analyzing FOV {fov+1}")

    compute_xcorr(
        out_paths.analysis_dir,
        out_paths.experiment_name,
        p.phase_plane,
        all_channels,
        p.FOVs,
        p.num_analyzers,
        p.alignment_pad,
    )

    with open(out_paths.analysis_dir / "channel_masks.yaml", "w") as cmask_file:
        yaml.dump(
            data=all_channels,
            stream=cmask_file,
            default_flow_style=False,
            tags=None,
        )


class Compile(MM3Container2):
    """
    the pipeline is as follows.
      1. check folders for existence, fetch FOVs & times & planes
          -> upon failure, simply show a list of inputs + update button.
      2. create a 'params' object whose params are valuemapped to widgets contained in our main list (a la guiclass)
          -> input directories, again, have a special status.
      3.
    """

    def __init__(self, viewer: Viewer):
        super().__init__()
        self.viewer = viewer

        self.in_paths = InPaths()
        try:
            self.run_params = gen_default_run_params(self.in_paths)
            self.out_paths = OutPaths()
            self.initialized = True
            self.display_fov_full()
            self.regen_widgets()
        except FileNotFoundError | ValueError:
            self.initialized = False
            self.regen_widgets()

    def run(self):
        print(self.run_params)
        compile(self.in_paths, self.run_params, self.out_paths)

    def display_fov_full(self):
        self.viewer.text_overlay.visible = False
        self.viewer.layers.clear()
        low_fov = self.run_params.FOVs[0]
        image_fov_worker = load_fov(self.in_paths.TIFF_dir, low_fov)
        image_fov_worker.returned.connect(lambda _: self.viewer.layers.clear())
        image_fov_worker.returned.connect(self.viewer.add_image)
        image_fov_worker.returned.connect(
            lambda _: setattr(self.viewer.dims, "current_step", (0, 0))
        )
        image_fov_worker.returned.connect(
            lambda _: self.viewer.layers[0].reset_contrast_limits()
        )
        image_fov_worker.returned.connect(
            lambda _: setattr(self.viewer.layers[0], "gamma", 0.5)
        )
        image_fov_worker.start()
        print(self.viewer.layers)


if __name__ == "__main__":
    cur_dir = Path(".")
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

    compile(
        TIFF_dir=cur_dir / "TIFF",
        num_analyzers=30,
        analysis_dir=cur_dir / "analysis",
        t_start=p.start_time - 1,
        t_end=p.end_time - 1,
        image_orientation=Orientation.auto,
        image_rotation=0,
        channel_width=10,
        channel_separation=45,
        channel_detection_snr=1,
        channel_length_pad=10,
        channel_width_pad=10,
        alignment_pad=10,
        experiment_name="",
        phase_plane="c1",
        FOVs=fovs,
        TIFF_source="nd2",
    )
