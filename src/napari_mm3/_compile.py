"""
Splits image TIFFs into trap TIFFs.
Can be run headless with parameters; if left unspecified it will analyze the full time range and all FOVs,
and use the default UI parameters.
"""

import argparse
import concurrent.futures as cf
import json
import multiprocessing
import pickle
import re
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated, Optional, Union

import numpy as np
import pandas as pd
import six
import tifffile as tiff
import yaml
from napari import Viewer
from napari.utils import progress
from scipy import ndimage as ndi
from scipy.signal import find_peaks_cwt
from skimage.feature import match_template

from ._deriving_widgets import (
    FOVList,
    MM3Container2,
    get_valid_fovs_folder,
    get_valid_planes_old,
    get_valid_times,
    information,
    load_tiff,
)
from .utils import TIFF_FILE_FORMAT_PEAK


#### Helpful utility functions.
def get_plane(filepath: str) -> Optional[str]:
    """
    Extracts the plane / channel number (e.g. phase fluorescence etc.) from a tiff file name.
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


def find_phase_idx(image_data: np.ndarray, phase_plane: str):
    if image_data.shape[0] == 1:
        return 0
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
    trench_length: int,
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
    default_open_end = twothirdpoint + proj_diff[twothirdpoint:].argmin()
    default_closed_end = (
        proj_diff[:onethirdpoint].argmax()
        if trench_length < 0
        else default_open_end - trench_length
    )
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
        slice_open_end = twothirdpoint + proj_diff[twothirdpoint:].argmin()
        slice_closed_end = (
            proj_diff[:onethirdpoint].argmax()
            if trench_length < 0
            else slice_open_end - trench_length
        )
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

    #    pickle.dump(crosscorrs, xcorrs_file, protocol=pickle.HIGHEST_PROTOCOL)
    json_crosscorrs = {}
    for fov, fov_val in crosscorrs.items():
        json_crosscorrs[str(fov)] = {}
        for peak, peak_val in fov_val.items():
            json_crosscorrs[str(fov)][str(peak)] = {}
            json_crosscorrs[str(fov)][str(peak)]["cc_avg"] = float(peak_val["cc_avg"])
            json_crosscorrs[str(fov)][str(peak)]["ccs"] = list(
                map(float, peak_val["ccs"])
            )

    with open(analysis_dir / "crosscorrs.json", "w") as xcorrs_file:
        json.dump(json_crosscorrs, xcorrs_file, indent=4)

    with open(analysis_dir / "crosscorrs.pkl", "wb") as xcorrs_file:
        pickle.dump(crosscorrs, xcorrs_file, protocol=pickle.HIGHEST_PROTOCOL)

    return crosscorrs


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


def find_mask_bounding_box(
    label_mask: np.ndarray,
    img_w: int,
    ch_masks: dict,  # dictionary of channel masks (one per FOV)
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
    masks = ndi.find_objects(label_mask)
    # generate bounding box for the channel mask.
    min_row, max_row = masks[0][0].start, masks[0][0].stop
    min_col, max_col = masks[0][1].start, masks[0][1].stop

    # channel_id (aka peak_id) given by the middle position
    channel_id = int(np.median([min_col, max_col]))

    if (0 < min_col) and (max_col < img_w):
        ch_masks[channel_id] = [
            [min_row, max_row],
            [min_col, max_col],
        ]

    return ch_masks


def make_masks(
    phase_image: np.ndarray,
    chnl_timeseries: list[dict],
    channel_width_pad: int = 0,
    channel_width: int = 0,
    channel_length_pad: int = 0,
) -> tuple[int, int, dict]:
    """
    Returns
    -------
    channel_masks : dict[channel_id: int, channel_corners: list[[y1, y2], [x1, x2]]]
        dictionary of consensus channel masks.
        channels uniquely identified by their median x position.

    """
    # declare temp variables from parameters.
    crop_wp = int(channel_width_pad + channel_width / 2)
    chan_lp = int(channel_length_pad)

    times, image_rows, image_cols = phase_image.shape

    consensus_mask = make_consensus_mask(phase_image, chnl_timeseries, crop_wp, chan_lp)

    channel_masks = {}
    for label in np.unique(consensus_mask):
        if label == 0:  # label zero is the background
            continue

        label_mask = consensus_mask == label
        channel_masks = find_mask_bounding_box(label_mask, image_cols, channel_masks)
    # add channel_mask dictionary to the fov dictionary, use copy to play it safe
    return channel_masks.copy()


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

        # channel_stack = channel_stack.squeeze()
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
    trench_length: int = -1
    channel_width: int = 10
    channel_separation: int = 45
    channel_detection_snr: float = 1.0
    channel_length_pad: int = 10
    channel_width_pad: int = 10
    alignment_pad: int = 10
    TIFF_source: Annotated[str, {"choices": ["nd2", "BioFormats / other TIFF"]}] = "nd2"


def gen_default_run_params(in_files: InPaths):
    try:
        t_start, t_end = get_valid_times(in_files.TIFF_dir)
        all_fovs = get_valid_fovs_folder(in_files.TIFF_dir)
        # get the brightest channel as the default phase plane!
        channels = get_valid_planes_old(in_files.TIFF_dir)
        # move this into runparams somehow!
        params = RunParams(
            phase_plane=channels[0],
            t_start=t_start,
            t_end=t_end,
            FOVs=FOVList(all_fovs),
        )
        params.__annotations__["phase_plane"] = Annotated[str, {"choices": channels}]
        return params
    except ValueError as e:
        raise FileNotFoundError("TIFF folder not found or not valid")


def generate_tif_dataframe(tiff_folder: Path) -> pd.DataFrame:
    """
    Generate a dataframe with columns for fov, time and file path for each TIFF in the folder.
    Sorted by fov, then time.
    """
    found_files = list(tiff_folder.glob("*.tif"))
    data = []
    for f in found_files:
        fov = get_fov(f.name)
        time = get_time(f.name)
        if (fov is not None) and (time is not None):
            data.append({"fov": fov, "time": time, "path": f})

    df = pd.DataFrame(data)
    df.sort_values(by=["fov", "time"], inplace=True)
    return df


def worker(
    df_files: pd.DataFrame,
    p: RunParams,
    out_paths: OutPaths,
    fov: int,
):
    image_series = df_files[df_files["fov"] == fov]
    if len(image_series) == 0:
        print(f"No TIFF files found for FOV {fov}.")
        return

    image_series = image_series[
        (p.t_start <= image_series["time"]) & (image_series["time"] <= p.t_end)
    ]

    print(f"analyzing FOV {fov}", end="\n")
    chnl_timeseries = []
    img_timeseries = []
    print(f"Finding channel locations for FOV {fov}")
    for row in image_series.itertuples(index=False):
        fov, time, path = row

        with tiff.TiffFile(path) as tif:
            image_data = tif.asarray()

        # TODO: move this out of the loop
        phase_idx = int(find_phase_idx(image_data, p.phase_plane))

        phase_image = image_data[phase_idx]

        channel_locs = find_channel_locs(
            phase_image,
            p.channel_width_pad,
            p.channel_width,
            p.channel_detection_snr,
            p.channel_separation,
            p.trench_length,
        )

        chnl_timeseries.append(channel_locs)
        img_timeseries.append(image_data)

    print(f"making masks for FOV {fov}", end="\n")
    img_timeseries = np.array(img_timeseries)
    # maybe clean this up in a sec
    channel_masks = make_masks(
        img_timeseries[:, phase_idx, :, :],
        chnl_timeseries,
        p.channel_width_pad,
        p.channel_width,
        p.channel_length_pad,
    )

    print(f"slicing channels for FOV {fov}", end="\n")
    tiff_stack_slice_and_write(
        img_timeseries,
        fov,
        channel_masks,
        out_paths.experiment_name,
        out_paths.channel_dir,
    )

    print(f"finished analyzing FOV {fov}")
    return channel_masks


def compile(in_paths: InPaths, p: RunParams, out_paths: OutPaths) -> None:
    """
    Given some TIFFs, identify any traps in the phase contrast channel and write out new TIFFs for each trap and channel.
    """
    out_paths.analysis_dir.mkdir(exist_ok=True)
    out_paths.channel_dir.mkdir(exist_ok=True)

    df_files = generate_tif_dataframe(in_paths.TIFF_dir)
    all_channels = {}

    temp_worker = partial(worker, df_files, p, out_paths)
    with cf.ThreadPoolExecutor(max_workers=p.num_analyzers) as executor:
        it = executor.map(temp_worker, p.FOVs)
        for fov, chans in progress(
            zip(p.FOVs, it), total=len(p.FOVs), desc="Processing FOVs"
        ):
            all_channels[fov] = chans
            print(f"finished {fov + 1}")
        print("Finished analysis of all FOVs.")

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
    def __init__(self, viewer: Viewer):
        super().__init__(viewer)
        self.viewer = viewer

        self.in_paths = InPaths()
        self.regen_run_params()

    def regen_run_params(self):
        try:
            self.run_params = gen_default_run_params(self.in_paths)
            self.out_paths = OutPaths()
            self.initialized = True
            self.display_fov_full()
        except FileNotFoundError as e:
            self.initialized = False

        self.regen_widgets()

    def run(self):
        compile(self.in_paths, self.run_params, self.out_paths)

    def update_view(self):
        self.display_fov_full()

    def display_fov_full(self):
        self.viewer.text_overlay.visible = False
        self.viewer.layers.clear()
        low_fov = self.run_params.FOVs[0]
        image_fov = load_fov(self.in_paths.TIFF_dir, low_fov)

        self.viewer.layers.clear()
        self.viewer.add_image(
            image_fov,
            contrast_limits=[0, np.percentile(image_fov, 99.9)],
            name="image_fov",
        )
        self.viewer.dims.set_current_step(0, 0)
        self.viewer.layers["image_fov"].gamma = 0.5

        loc = find_channel_locs(
            image_fov[0],
            self.run_params.channel_width_pad,
            self.run_params.channel_width,
            self.run_params.channel_detection_snr,
            self.run_params.channel_separation,
            self.run_params.trench_length,
        )
        cur_image_masks = make_masks(
            image_fov[0][np.newaxis, ...],
            [loc],
            self.run_params.channel_width_pad,
            self.run_params.channel_width,
            self.run_params.channel_length_pad,
        )
        print(cur_image_masks)
        render_shapes = []
        for peak, bbox in cur_image_masks.items():
            min_row, max_row = bbox[0]
            min_col, max_col = bbox[1]
            rect = np.array(
                [
                    [min_row, min_col],
                    [min_row, max_col],
                    [max_row, max_col],
                    [max_row, min_col],
                ]
            )
            render_shapes.append(rect)
        self.viewer.add_shapes(
            render_shapes, face_color="transparent", edge_color="red", edge_width=3
        )


if __name__ == "__main__":
    """
    Example usage:
    python -m napari_mm3._compile -t-start 0 --t-end 50 --fov-list 1-5 --phase-plane c1 --channel-width 12 
    """

    parser = argparse.ArgumentParser(
        description="Compile and segment image TIFFs into individual channel traps"
    )

    # Input/Output parameters
    parser.add_argument(
        "--tiff-dir",
        type=Path,
        default=Path(".") / "TIFF",
        help="Directory containing TIFF files (default: ./TIFF)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="",
        help="Name of the experiment (default: empty string)",
    )
    parser.add_argument(
        "--channel-dir",
        type=Path,
        default=Path("./analysis/channels"),
        help="Output directory for channel TIFFs (default: ./analysis/channels)",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("./analysis"),
        help="Analysis output directory (default: ./analysis)",
    )

    # Time and FOV parameters
    parser.add_argument(
        "--t-start",
        type=int,
        default=None,
        help="Starting time point (default: from image data)",
    )
    parser.add_argument(
        "--t-end",
        type=int,
        default=None,
        help="Ending time point (default: from image data)",
    )
    parser.add_argument(
        "--fov-list",
        type=str,
        default=None,
        help="Field of view indices to process (e.g., '1,3,5' or '1-5,10', default: all FOVs)",
    )

    # Channel detection parameters
    parser.add_argument(
        "--phase-plane",
        type=str,
        default=None,
        help="Phase plane channel (e.g., 'c1', default: brightest channel)",
    )
    parser.add_argument(
        "--channel-width",
        type=int,
        default=10,
        help="Channel width in pixels (default: 10)",
    )
    parser.add_argument(
        "--channel-separation",
        type=int,
        default=45,
        help="Minimum channel separation in pixels (default: 45)",
    )
    parser.add_argument(
        "--channel-detection-snr",
        type=float,
        default=1.0,
        help="Signal-to-noise ratio for channel detection (default: 1.0)",
    )
    parser.add_argument(
        "--channel-width-pad",
        type=int,
        default=10,
        help="Width padding for channel detection (default: 10)",
    )
    parser.add_argument(
        "--channel-length-pad",
        type=int,
        default=10,
        help="Length padding for channel masks (default: 10)",
    )
    parser.add_argument(
        "--trench-length",
        type=int,
        default=-1,
        help="Trench length in pixels (-1 to auto-detect, default: -1)",
    )

    # Processing parameters
    parser.add_argument(
        "--num-analyzers",
        type=int,
        default=multiprocessing.cpu_count(),
        help=f"Number of parallel analyzers (default: {multiprocessing.cpu_count()})",
    )
    parser.add_argument(
        "--alignment-pad",
        type=int,
        default=10,
        help="Padding for alignment calculations (default: 10)",
    )
    parser.add_argument(
        "--tiff-source",
        type=str,
        choices=["nd2", "BioFormats / other TIFF"],
        default="nd2",
        help="Source format of TIFF files (default: nd2)",
    )

    args = parser.parse_args()

    # Create InPaths
    in_paths = InPaths(TIFF_dir=args.tiff_dir)

    # Get defaults from gen_default_run_params
    try:
        default_params = gen_default_run_params(in_paths)
    except FileNotFoundError:
        raise FileNotFoundError("TIFF folder not found or not valid")

    # Override defaults with command-line arguments
    t_start = args.t_start if args.t_start is not None else default_params.t_start
    t_end = args.t_end if args.t_end is not None else default_params.t_end
    phase_plane = (
        args.phase_plane if args.phase_plane is not None else default_params.phase_plane
    )

    if args.fov_list is not None:
        fov_list = FOVList(args.fov_list)
    else:
        fov_list = default_params.FOVs

    # Create RunParams with all parameters
    run_params = RunParams(
        t_start=t_start,
        t_end=t_end,
        FOVs=fov_list,
        phase_plane=phase_plane,
        num_analyzers=args.num_analyzers,
        trench_length=args.trench_length,
        channel_width=args.channel_width,
        channel_separation=args.channel_separation,
        channel_detection_snr=args.channel_detection_snr,
        channel_length_pad=args.channel_length_pad,
        channel_width_pad=args.channel_width_pad,
        alignment_pad=args.alignment_pad,
        TIFF_source=args.tiff_source,
    )

    # Create OutPaths
    out_paths = OutPaths(
        experiment_name=args.experiment_name,
        channel_dir=args.channel_dir,
        analysis_dir=args.analysis_dir,
    )

    compile(in_paths, run_params, out_paths)
