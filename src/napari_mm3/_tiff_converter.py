import argparse
import copy
import datetime
import json
import multiprocessing
import os
import re
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import napari
import nd2
import numpy as np
import tifffile as tiff
from magicgui.widgets import Container, FileEdit, FloatSpinBox, PushButton
from napari import Viewer
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
from tifffile import imread, imwrite
from tqdm import tqdm

from ._deriving_widgets import (
    FOVList,
    MM3Container2,
    information,
)

# === User Parameters ===
# DATA_DIR = "/Users/haochen/Documents/Data/20250521_SJ2624_Mgly_mm/TIFF_0ara_channel1610_1735"  # e.g., "./tif_images"
# OUTPUT_STACK_DIR = "/Users/haochen/Documents/Data/20250521_SJ2624_Mgly_mm/TIFF_0ara_channel1610_1735_drift_corrected_stacks"
# OUTPUT_FRAME_DIR = "/Users/haochen/Documents/Data/20250521_SJ2624_Mgly_mm/TIFF_0ara_channel1610_1735_drift_corrected"
# UPSAMPLE_FACTOR = 5  # Reduce from 10 to 5 for faster run
# COMPRESSION = "zlib"  # or "lzma" for stronger compression
# CROP = True  # Crop edges after registration
# MAX_WORKERS = multiprocessing.cpu_count()  # Limit for multiprocessing
#
# os.makedirs(OUTPUT_STACK_DIR, exist_ok=True)
# os.makedirs(OUTPUT_FRAME_DIR, exist_ok=True)
#
#
# # === Define Per-FOV Processing Function ===
#
#
# def process_fov(
#     xy_idx,
#     frames,
#     data_dir,
#     output_dir_stack,
#     output_dir_frames,
#     upsample_factor=10,
#     crop=True,
# ):
#     frames.sort()
#     raw_stack = []
#
#     for _, _, fname, _ in frames:
#         img = imread(os.path.join(data_dir, fname))  # [2, H, W]
#         raw_stack.append(img)
#
#     raw_stack = np.stack(raw_stack)  # [T, 2, H, W]
#     T, C, H, W = raw_stack.shape
#     corrected_stack = np.empty_like(raw_stack)
#     shifts = []
#
#     ref = raw_stack[T // 2, 0]  # Reference frame from channel 0
#
#     for i in range(T):
#         shift_est, _, _ = phase_cross_correlation(
#             ref, raw_stack[i, 0], upsample_factor=upsample_factor
#         )
#         shifts.append(shift_est)
#         for ch in range(C):
#             corrected_stack[i, ch] = shift(
#                 raw_stack[i, ch], shift=shift_est, mode="constant", cval=0
#             )
#
#     if crop:
#         shifts = np.array(shifts)
#         dy = shifts[:, 0]
#         dx = shifts[:, 1]
#
#         top_crop = int(np.ceil(np.max(dy)))
#         bottom_crop = int(np.ceil(-np.min(dy)))
#         left_crop = int(np.ceil(np.max(dx)))
#         right_crop = int(np.ceil(-np.min(dx)))
#
#         crop_top = top_crop
#         crop_bottom = H - bottom_crop
#         crop_left = left_crop
#         crop_right = W - right_crop
#
#         corrected_stack = corrected_stack[
#             :, :, crop_top:crop_bottom, crop_left:crop_right
#         ]
#
#     # === Save multi-page stacks per channel ===
#     for ch in range(C):
#         stack_file = os.path.join(output_dir_stack, f"xy{xy_idx}_ch{ch}.tif")
#         imwrite(
#             stack_file,
#             corrected_stack[:, ch],
#             dtype=raw_stack.dtype,
#             compression="zlib",
#             photometric="minisblack",
#         )
#
#     # === Save individual dual-channel frames ===
#     for i, (_, t_str, _, base_name) in enumerate(frames):
#         frame_file = os.path.join(
#             output_dir_frames, f"{base_name}_t{t_str}xy{xy_idx}.tif"
#         )
#         imwrite(
#             frame_file, corrected_stack[i], dtype=raw_stack.dtype, compression="zlib"
#         )
#
#     print(f"[DONE] xy{xy_idx} ({T} timepoints)")
#
#
# # === Main Function ===
# def main():
#     # Parse all TIFFs and group by FOV
#     file_pattern = re.compile(r"(.*)_t(\d{4})xy(\d{2})\.tif$")
#     fov_files = defaultdict(list)
#
#     for fname in os.listdir(DATA_DIR):
#         if not fname.lower().endswith(".tif"):
#             continue
#         match = file_pattern.match(fname)
#         if match:
#             base, t_str, xy_idx = match.groups()
#             fov_files[xy_idx].append((int(t_str), t_str, fname, base))
#
#     print(f"Found {len(fov_files)} FOVs. Starting parallel processing...")
#
#     # Run parallel processing
#     with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = []
#         for xy_idx, frames in fov_files.items():
#             futures.append(
#                 executor.submit(
#                     process_fov,
#                     xy_idx,
#                     frames,
#                     DATA_DIR,
#                     OUTPUT_STACK_DIR,
#                     OUTPUT_FRAME_DIR,
#                     UPSAMPLE_FACTOR,
#                     COMPRESSION,
#                     CROP,
#                 )
#             )
#
#         # Ensure all complete and capture exceptions
#         for f in tqdm(futures):
#             f.result()
#
#
# if __name__ == "__main__":
#     main()
#


def get_nd2_fovs(data_path):
    with nd2.ND2File(str(data_path)) as nd2f:
        # allow for 1 fov with no FOV axis:
        try:
            return nd2f.sizes["P"]
        except KeyError:
            return 1


def get_nd2_times(data_path):
    with nd2.ND2File(str(data_path)) as nd2f:
        try:
            return nd2f.sizes["T"]
        except KeyError:
            return 1


def nd2_iter(nd2f: nd2.ND2File, time_range_ids, fov_list_ids):
    """
    Iterates over the contents of an ND2File.
    If available, use fov_list/time_range. If not,
    do the whole iamge.

    params:
        nd2f: The ND2Reader object to use.

    returns:
        t: the time-index of the returned frame.
        fov: the fov-index of the returned frame.
        image_data: the image at the given time/fov.
    """
    try:
        nd2_time_range = set(range(0, nd2f.sizes["T"]))
    except KeyError:
        nd2_time_range = set([1])

    # if only 1 fov, then just yield the single FOV.
    if "P" not in nd2f.sizes:
        im = nd2f.asarray()
        for t in nd2_time_range:
            if t not in time_range_ids:
                continue
            image_data = im[t]
            yield t, 0, image_data
        return

    nd2_fov_list = set(range(0, nd2f.sizes["P"]))
    if fov_list_ids != []:
        valid_fovs = set(fov_list_ids).intersection(nd2_fov_list)
    else:
        valid_fovs = nd2_fov_list

    if valid_fovs != set(fov_list_ids):
        information("The following FOVs were not in the nd2, and thus were omitted:")
        information(set(fov_list_ids) - valid_fovs)

    for fov in valid_fovs:
        im = nd2f.asarray(fov)
        for t in nd2_time_range:
            if t not in time_range_ids:
                continue
            image_data = im[t]
            yield t, fov, image_data


def write_timetable(nd2f: nd2.ND2File, path: Path):
    timetable = {}
    for event in nd2f.events():
        if "P Index" not in event:
            continue
        timestamp = float(int(event["Time [s]"]))
        fov_idx = int(event["P Index"])
        t_idx = int(event["T Index"])
        if fov_idx not in timetable:
            timetable[fov_idx] = {t_idx: timestamp}
        else:
            timetable[fov_idx][t_idx] = timestamp
    with path.open("w") as f:
        json.dump(timetable, f, indent=4)


@dataclass
class InPaths:
    nd2_file: Path

    def __init__(self):
        nd2files = list(Path(".").glob("*.nd2"))
        self.nd2_file = nd2files[0]


@dataclass
class RunParams:
    image_start: int
    image_end: int
    fov_list: FOVList
    vertical_crop_lower: float = 0.0
    vertical_crop_upper: float = 1.0
    horizontal_crop_lower: float = 0.0
    horizontal_crop_upper: float = 1.0


@dataclass
class OutPaths:
    tiff_folder: Annotated[Path, {"mode": "d"}] = Path("./TIFF/")


# TODO: Make the imaging consistent!
def gen_default_run_params(in_paths: InPaths):
    end_time = get_nd2_times(in_paths.nd2_file)
    total_fovs = get_nd2_fovs(in_paths.nd2_file)
    return RunParams(
        image_start=1,
        image_end=end_time,
        fov_list=FOVList(list(range(1, total_fovs + 1))),
    )


def nd2ToTIFF(
    in_paths: InPaths,
    run_params: RunParams,
    out_paths: OutPaths,
):
    """
    This script converts a Nikon Elements .nd2 file to individual TIFF files per time point.
    Multiple color planes are stacked in each time point to make a multipage TIFF.

    params:
        data_path: Path to the experimental data
        tif_dir: Where to put the TIFFs when we are done.
        tif_filename: A prefix for the output tifs
        vertical_crop: [ymin, ymax]. Percentage crop. Optional.
        tworow_crop: [[y1_min, y1_max], [y2_min, y2_max]]. Used for cropping if you have multiple rows; currently only two are supported.
        FOVs: String specifying a range (or a single number) of FOVs to convert to nd2
        image_start, image_end: Image range that we want to turn into TIFFs (inclusive)
    """
    # set up image folders if they do not already exist
    if not os.path.exists(out_paths.tiff_folder):
        os.makedirs(out_paths.tiff_folder)

    if not os.path.exists(Path(".") / "analysis"):
        os.makedirs(Path(".") / "analysis")

    nd2file = in_paths.nd2_file
    file_prefix = os.path.split(os.path.splitext(nd2file)[0])[1]
    information(f"Extracting {file_prefix} ...")
    with nd2.ND2File(str(nd2file)) as nd2f:
        # load in the time table.
        # TODO: Add analysis
        write_timetable(nd2f, Path(".") / "analysis" / "timetable.json")
        starttime = nd2f.text_info["date"]
        starttime = datetime.datetime.strptime(starttime, "%m/%d/%Y %I:%M:%S %p")

        try:
            planes = nd2f.sizes["C"]
        except KeyError:
            planes = 1

        # Extraction range is the time points that will be taken out.
        time_range_ids = range(run_params.image_start - 1, run_params.image_end)
        fov_list_ids = [fov_id - 1 for fov_id in run_params.fov_list]
        for t_id, fov_id, image_data in nd2_iter(
            nd2f, time_range_ids=time_range_ids, fov_list_ids=fov_list_ids
        ):
            # timepoint and fov output name (1 indexed rather than 0 indexed)
            try:
                milliseconds = copy.deepcopy(nd2f.events()[t_id * 2]["Time [s]"])
                acq_days = milliseconds / 60.0 / 60.0 / 24.0
                acq_time = starttime.timestamp() + acq_days
            except IndexError:
                acq_time = None

            # add extra axis to make below slicing simpler. removed automatically if only one color
            if len(image_data.shape) <= 3:
                image_data = np.expand_dims(image_data, axis=0)
            # in case one channel, one fov.
            if len(image_data.shape) == 3:
                image_data = np.expand_dims(image_data, axis=0)

            # for just a simple crop
            if (
                run_params.vertical_crop_lower != 0.0
                or run_params.vertical_crop_upper != 1.0
            ):
                _, nc, H, W = image_data.shape
                ## convert from xy to row-column coordinates for numpy slicing
                yhi = int((1 - run_params.vertical_crop_lower) * H)
                ylo = int((1 - run_params.vertical_crop_upper) * H)
                image_data = image_data[:, ylo:yhi, :]

            if (
                run_params.horizontal_crop_lower != 0.0
                or run_params.horizontal_crop_upper != 1.0
            ):
                _, nc, H, W = image_data.shape
                xlo = int(run_params.horizontal_crop_lower * W)
                xhi = int(run_params.horizontal_crop_upper * W)
                image_data = image_data[:, :, xlo:xhi]

            # make dictionary which will be the metdata for this TIFF
            metadata_json = json.dumps(
                {
                    "fov": fov_id + 1,
                    "t": t_id + 1,
                    "jd": acq_time,
                    "planes": planes,
                }
            )
            tif_filename = f"{file_prefix}_t{t_id + 1:04d}xy{fov_id + 1:02d}.tif"
            information("Saving %s." % tif_filename)
            tiff.imwrite(
                out_paths.tiff_folder / tif_filename,
                data=image_data,
                description=metadata_json,
                compression="zlib",
                photometric="minisblack",
            )


class TIFFExport(MM3Container2):
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
            # self.render_nd2()
            self.regen_widgets()
        except FileNotFoundError | ValueError:
            self.initialized = False
            self.regen_widgets()

    def run(self):
        print(self.run_params)
        nd2ToTIFF(self.in_paths, self.run_params, self.out_paths)

    # TODO: Fix this one up.
    # def render_nd2(self):
    #     viewer = self.viewer
    #     viewer.layers.clear()
    #     viewer.grid.enabled = True

    #     nd2file = self.in_paths.nd2_file

    #     with nd2.ND2File(str(nd2file)) as ndx:
    #         sizes = ndx.sizes

    #         if "T" not in sizes:
    #             sizes["T"] = 1
    #         if "P" not in sizes:
    #             sizes["P"] = 1
    #         if "C" not in sizes:
    #             sizes["C"] = 1
    #         ndx.bundle_axes = "zcyx"
    #         ndx.iter_axes = "t"
    #         n = len(ndx)

    #         shape = (
    #             sizes["t"],
    #             sizes["z"],
    #             sizes["v"],
    #             sizes["c"],
    #             sizes["y"],
    #             sizes["x"],
    #         )
    #         image = np.zeros(shape, dtype=np.float32)

    #         for i in range(n):
    #             image[i] = ndx.get_frame(i)

    #     image = np.squeeze(image)

    #     viewer.add_image(image, channel_axis=1, colormap="gray")
    #     viewer.grid.shape = (-1, 3)
    #     viewer.dims.current_step = (0, 0)
    #     viewer.layers.link_layers()  ## allows user to set contrast limits for all FOVs at once


if __name__ == "__main__":
    in_files = InPaths()
    run_params: RunParams = gen_default_run_params(in_files)
    out_paths = OutPaths()

    nd2ToTIFF(in_files, run_params, out_paths)
