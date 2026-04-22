import argparse
import concurrent.futures as cf
import datetime
import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Annotated

import nd2
import numpy as np
import psutil
import tifffile as tiff
from magicgui.widgets import PushButton, SpinBox
from napari import Viewer
from napari.utils import progress
from scipy.ndimage import rotate, shift
from skimage.registration import phase_cross_correlation

from ._deriving_widgets import (
    FOVList,
    MM3Container2,
)

BLACK = np.array([0.0, 0.0, 0.0])


def parse_datetime_flexible(date_string: str) -> datetime.datetime:
    """
    Parse a datetime string, trying multiple common formats.

    Args:
        date_string: The datetime string to parse

    Returns:
        A datetime object

    Raises:
        ValueError: If none of the known formats match
    """
    # Normalize whitespace (collapse multiple spaces into one)
    normalized = re.sub(r"\s+", " ", date_string.strip())

    formats = [
        "%m/%d/%Y %I:%M:%S %p",  # Original format: 12/10/2025 6:58:17 PM
        "%Y/%m/%d %H:%M:%S",  # 2025/12/10 18:58:17
        "%Y-%m-%d %H:%M:%S",  # 2025-12-10 18:58:17
        "%m/%d/%Y %H:%M:%S",  # 12/10/2025 18:58:17
        "%d/%m/%Y %H:%M:%S",  # 10/12/2025 18:58:17
        "%Y/%m/%d %I:%M:%S %p",  # 2025/12/10 6:58:17 PM
    ]

    for fmt in formats:
        try:
            return datetime.datetime.strptime(normalized, fmt)
        except ValueError:
            continue

    raise ValueError(
        f"Could not parse datetime '{date_string}' with any known format. "
        f"Tried formats: {formats}"
    )


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


def write_timetable(nd2f: nd2.ND2File, path: Path):
    timetable = {}
    for event in nd2f.events():
        if "P Index" not in event:
            continue
        timestamp = float(int(event["Time [s]"]))
        fov_idx = int(event["P Index"]) + 1
        try:
            t_idx = event["T Index"] + 1
        except KeyError:
            t_idx = 0
        t_idx = int(t_idx)
        if fov_idx not in timetable:
            timetable[fov_idx] = {t_idx: timestamp}
        else:
            timetable[fov_idx][t_idx] = timestamp
    with path.open("w") as f:
        json.dump(timetable, f, indent=4)


def fix_rotation(angle: float, image_data: np.ndarray) -> np.ndarray:
    # need to add support for no channels.
    if angle == 0:
        return image_data

    image_data = np.asarray(image_data, dtype=np.float32)
    return rotate(image_data, float(angle), axes=(-1, -2), order=1)


def register_shifts(
    raw_stack,
    ref=None,
    upsample_factor=2,
):
    """
    Generate shifts for drift correction. Register against middle frame.
    Seems to work pretty well; depending on context last or first frame might work better.
    """
    print("Stabilizing")
    raw_stack = np.stack(raw_stack)  # [T, N, H, W]

    T = raw_stack.shape[0]
    if ref is None:
        ref = raw_stack[T // 2, 0]  # Reference frame from channel 0

    shifts_f = []
    for i in range(0, T):
        shift_est, _, _ = phase_cross_correlation(
            ref, raw_stack[i, 0], upsample_factor=upsample_factor
        )
        shifts_f.append(shift_est)

    # shifts_b = np.zeros((T, 2))
    # for i in range(T - 2, -1, -1):
    #     inc, _, _ = phase_cross_correlation(
    #         raw_stack[i + 1, 0],
    #         raw_stack[i, 0],
    #         upsample_factor=upsample_factor,
    #     )
    #     shifts_b[i] = shifts_b[i + 1] + inc
    #     shifts_b -= shifts_b[0]

    # w = np.linspace(1, 0, T)[:, None]
    # shifts = w * shifts_f + (1 - w) * shifts_b
    return shifts_f


def stabilize_fov(
    raw_stack,
    shifts=None,
    upsample_factor=2,
):
    """
    Drift correction.
    Note: If you want easy performance improvements, simply cache 'shifts'.

    If people report issues, would be good to get this to use less memory.
    But, a typical raw stack is at most ~1GB, so this is a reasonable expectation.
    """
    print("Stabilizing")
    # remove fov axis if it exists, since this works on one FOV at a time.
    raw_stack = raw_stack.squeeze()
    if raw_stack.ndim == 3:
        raw_stack = raw_stack[
            :, np.newaxis, ...
        ]  # add channel axis if it doesn't exist
    # final shape is [T, C, H, W]
    print(raw_stack.shape)
    if shifts is None:
        shifts = register_shifts(raw_stack, upsample_factor=upsample_factor)

    T = raw_stack.shape[0]
    C = raw_stack.shape[1]
    corrected_stack = np.empty_like(raw_stack)

    for i in range(T):
        for ch in range(C):
            corrected_stack[i, ch] = shift(
                raw_stack[i, ch], shift=shifts[i], mode="constant", cval=0
            )

    return corrected_stack


@dataclass
class InPaths:
    nd2_file: Path

    def __init__(self):
        nd2files = list(Path(".").glob("*.nd2"))
        self.nd2_file = nd2files[0]


class FlipImage(Enum):
    auto = 1
    yes = 2
    no = 3


@dataclass
class RunParams:
    image_start: int
    image_end: int
    fov_list: FOVList
    stabilize: bool
    flip_image: FlipImage = FlipImage.auto
    rotate: Annotated[float, {"min": -90, "max": 90}] = 0
    vertical_crop_lower: float = 0.0
    vertical_crop_upper: float = 1.0
    horizontal_crop_lower: float = 0.0
    horizontal_crop_upper: float = 1.0


@dataclass
class OutPaths:
    tiff_folder: Annotated[Path, {"mode": "d"}] = Path("./TIFF/")
    timetable: Annotated[Path, {"mode": "d"}] = Path("./analysis/timetable.json")


# TODO: Make the imaging consistent!
def gen_default_run_params(in_paths: InPaths):
    end_time = get_nd2_times(in_paths.nd2_file)
    total_fovs = get_nd2_fovs(in_paths.nd2_file)
    return RunParams(
        stabilize=False,
        image_start=1,
        image_end=end_time,
        fov_list=FOVList(list(range(1, total_fovs + 1))),
    )


def load_fov(in_paths: InPaths, fov_idx):
    with nd2.ND2File(str(in_paths.nd2_file)) as nd2f:
        arr = nd2f.asarray(fov_idx - 1)

    return arr


def pipeline(image_data, run_params: RunParams):
    if run_params.rotate != 0:
        image_data = fix_rotation(run_params.rotate, image_data)

    # add extra axis to make below slicing simpler. removed automatically if only one color.
    # sanity check: what do if it's 1 channel/1 time?
    if len(image_data.shape) < 3:
        image_data = np.expand_dims(image_data, axis=0)

    # for just a simple crop
    if (run_params.vertical_crop_lower > 0.0) or (run_params.vertical_crop_upper < 1.0):
        H, W = image_data.shape[-2], image_data.shape[-1]
        ## convert from xy to row-column coordinates for numpy slicing
        yhi = int((1 - run_params.vertical_crop_lower) * H)
        ylo = int((1 - run_params.vertical_crop_upper) * H)
        image_data = image_data[..., ylo:yhi, :]

    if (run_params.horizontal_crop_lower != 0.0) or (
        run_params.horizontal_crop_upper != 1.0
    ):
        H, W = image_data.shape[-2], image_data.shape[-1]
        xlo = int(run_params.horizontal_crop_lower * W)
        xhi = int(run_params.horizontal_crop_upper * W)
        image_data = image_data[..., :, xlo:xhi]

    if run_params.stabilize:
        image_data = stabilize_fov(image_data)

    image_data = fix_orientation(image_data, 0, run_params.flip_image)

    return image_data


def worker(
    nd2f,
    time_range_ids,
    planes,
    file_prefix,
    run_params: RunParams,
    out_paths: OutPaths,
    fov_id: int,
):
    print(f"starting analysis of FOV {fov_id + 1}")
    image_data = nd2f.asarray(fov_id)

    if "T" not in nd2f.sizes:
        image_data = image_data[np.newaxis, ...]

    if "P" not in nd2f.sizes:
        image_data = image_data[:, np.newaxis, ...]

    image_data = image_data[time_range_ids, ...]
    image_data = pipeline(image_data, run_params)

    print(image_data.shape)
    print(f"writing fov {fov_id + 1} to disk")
    for t_idx, image_data_cur_t in enumerate(image_data):
        t_id = time_range_ids[t_idx]
        # make dictionary which will be the metdata for this TIFF
        metadata_json = json.dumps(
            {
                "fov": fov_id + 1,
                "t": t_id + 1,
                "planes": planes,
            }
        )
        tif_filename = f"{file_prefix}_t{t_id + 1:04d}xy{fov_id + 1:02d}.tif"
        tiff.imwrite(
            out_paths.tiff_folder / tif_filename,
            data=image_data_cur_t,
            description=metadata_json,
            compression="zlib",
            photometric="minisblack",
        )

    return fov_id


# define function for flipping the images on an FOV by FOV basis
def fix_orientation(
    image_data: np.ndarray, phase_idx: int, flip_image: FlipImage
) -> np.ndarray:
    """
    Fix the orientation. The standard direction for channels to open to is down.
    """
    if flip_image == FlipImage.yes:
        return image_data[..., ::-1, :]
    elif flip_image == FlipImage.no:
        return image_data
    elif flip_image == FlipImage.auto:
        # flip based on the index of the highest average row value
        brightest_row = np.argmax(image_data[..., phase_idx, :, :].mean(axis=-1))
        midline = image_data[phase_idx].shape[-2] / 2
        if brightest_row < midline:
            return image_data[..., ::-1, :]
        return image_data
    else:
        raise ValueError(f"Invalid flip_image value: {flip_image}")


def nd2ToTIFF(
    in_paths: InPaths, run_params: RunParams, out_paths: OutPaths, single_thread=False
):
    """
    This script converts a Nikon Elements .nd2 file to individual TIFF files per time point.
    Multiple color planes are stacked in each time point to make a multipage TIFF.
    """
    # set up image folders if they do not already exist
    if not out_paths.tiff_folder.exists():
        out_paths.tiff_folder.mkdir()

    if not Path("./analysis").exists():
        Path("./analysis").mkdir()

    nd2file = in_paths.nd2_file
    file_prefix = os.path.split(os.path.splitext(nd2file)[0])[1]

    with nd2.ND2File(str(nd2file)) as nd2f:
        write_timetable(nd2f, out_paths.timetable)
        try:
            planes = nd2f.sizes["C"]
        except KeyError:
            planes = 1

        time_range_ids = list(range(run_params.image_start - 1, run_params.image_end))
        fov_list_ids = [fov_id - 1 for fov_id in run_params.fov_list]

        if single_thread:
            for fov_id in fov_list_ids:
                worker(
                    nd2f,
                    time_range_ids,
                    planes,
                    file_prefix,
                    run_params,
                    out_paths,
                    fov_id,
                )
            return

        # calculate how many threads we can safely launch.
        total_memory = psutil.virtual_memory().available
        # note: nd2f is thread-safe, so this can be shared.
        temp_fov = nd2f.asarray(fov_list_ids[0])
        fov_memory = temp_fov.nbytes
        del temp_fov
        # .6 arbitrary for safety. / 2 due to needing a working copy of the data.
        possible_threads = min(0.6 * total_memory / fov_memory / 2, os.cpu_count())
        print(f"Launching {int(possible_threads)} processes.")

        temp_worker = partial(
            worker,
            nd2f,
            time_range_ids,
            planes,
            file_prefix,
            run_params,
            out_paths,
        )
        with cf.ThreadPoolExecutor(max_workers=int(possible_threads)) as executor:
            it = executor.map(temp_worker, fov_list_ids)
            for fov_id in progress(it, total=len(fov_list_ids), desc="Processing FOVs"):
                print(f"finished {fov_id + 1}")
            print("Finished analysis of all FOVs.")


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
        super().__init__(viewer)
        self.viewer = viewer

        self.in_paths = InPaths()
        try:
            self.run_params = gen_default_run_params(self.in_paths)
            self.out_paths = OutPaths()
            self.initialized = True
            self.regen_widgets()
        except FileNotFoundError | ValueError:
            self.initialized = False
            self.regen_widgets()

    def regen_widgets(self):
        super().regen_widgets()

        self["flip_image"].changed.connect(self.preview_fov)
        self["stabilize"].changed.connect(self.update_fov_idx)
        self["stabilize"].changed.connect(self.preview_fov)
        self["rotate"].changed.connect(self.preview_fov)
        self["horizontal_crop_upper"].changed.connect(self.preview_fov)
        self["horizontal_crop_lower"].changed.connect(self.preview_fov)
        self["vertical_crop_lower"].changed.connect(self.preview_fov)
        self["vertical_crop_upper"].changed.connect(self.preview_fov)

        self.w_gen_timetable = PushButton(text="gen_timetable")
        self.append(self.w_gen_timetable)
        self.w_gen_timetable.changed.connect(self.gen_timetable_run)

        # quick hacks so I don't have to work with dask arrays/multithreading.
        self.w_fov = SpinBox(
            label="preview_fov_idx",
            min=min(self.run_params.fov_list),
            max=max(self.run_params.fov_list),
        )
        self.fov_id = 1
        self.append(self.w_fov)
        self.w_fov.changed.connect(self.update_fov_idx)
        self.w_fov.changed.connect(self.preview_fov)

        self.w_t_idx = SpinBox(
            label="preview_t_idx",
            min=self.run_params.image_start,
            max=self.run_params.image_end,
        )
        self.t_idx = 1
        self.append(self.w_t_idx)
        self.w_t_idx.changed.connect(self.update_t_idx)
        self.w_t_idx.changed.connect(self.preview_fov)

        self.preview_fov()

    def update_fov_idx(self):
        self.fov_id = self.w_fov.value
        self.cur_fov = load_fov(self.in_paths, self.fov_id)
        if self.run_params.stabilize:
            self.cur_fov = stabilize_fov(self.cur_fov[:, 0])

    def update_t_idx(self):
        self.t_idx = self.w_t_idx.value

    def gen_timetable_run(self):
        # make analysis folder if it doesn't exist already
        if not self.out_paths.timetable.parent.exists():
            self.out_paths.timetable.parent.mkdir()

        with nd2.ND2File(str(self.in_paths.nd2_file)) as nd2f:
            write_timetable(nd2f, self.out_paths.timetable)

    def run(self):
        print(self.in_paths)
        print(self.run_params)
        print(self.out_paths)
        nd2ToTIFF(self.in_paths, self.run_params, self.out_paths)

    def preview_fov(self):
        viewer = self.viewer

        if not hasattr(self, "cur_fov"):
            self.cur_fov = load_fov(self.in_paths, self.fov_id)
            if self.run_params.stabilize:
                self.cur_fov = stabilize_fov(self.cur_fov[:, 0])

        # record current dim positions
        pos = tuple(viewer.dims.current_step)

        viewer.dims.current_step = pos
        fov_img = self.cur_fov[self.t_idx - 1]
        fov_img = fix_rotation(self.run_params.rotate, fov_img)
        shape = fov_img.shape
        row_min = int(shape[-2] * (1 - self.run_params.vertical_crop_upper))
        row_max = int(shape[-2] * (1 - self.run_params.vertical_crop_lower))
        col_min = int(shape[-1] * self.run_params.horizontal_crop_lower)
        col_max = int(shape[-1] * self.run_params.horizontal_crop_upper)
        fov_img = fov_img[
            ...,
            row_min:row_max,
            col_min:col_max,
        ]
        fov_img = fix_orientation(fov_img, 0, self.run_params.flip_image)
        if "fov_img" in viewer.layers:
            layer = viewer.layers["fov_img"]
            layer.data = fov_img
        else:
            viewer.add_image(
                fov_img,
            )
        viewer.dims.current_step = pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ND2 files to TIFF format with optional image processing"
    )

    # Input/Output parameters
    parser.add_argument(
        "--nd2-file",
        type=Path,
        default=None,
        help="Path to the .nd2 file to convert (default: first .nd2 in current directory)",
    )
    parser.add_argument(
        "--tiff-folder",
        type=Path,
        default=Path("./TIFF/"),
        help="Output folder for TIFF files (default: ./TIFF/)",
    )
    parser.add_argument(
        "--timetable",
        type=Path,
        default=Path("./analysis/timetable.json"),
        help="Output path for timetable.json (default: ./analysis/timetable.json)",
    )

    # Image processing parameters
    parser.add_argument(
        "--image-start",
        type=int,
        default=None,
        help="Starting time point (default: 1)",
    )
    parser.add_argument(
        "--image-end",
        type=int,
        default=None,
        help="Ending time point (default: last time point in file)",
    )
    parser.add_argument(
        "--fov-list",
        type=str,
        default=None,
        help="Field of view indices to process (e.g., '1,3,5' or '1-5,10', default: all FOVs)",
    )
    parser.add_argument(
        "--stabilize",
        action="store_true",
        help="Enable image stabilization (default: False)",
    )
    parser.add_argument(
        "--rotate",
        type=float,
        default=0.0,
        help="Rotation angle in degrees, -90 to 90 (default: 0)",
    )
    parser.add_argument(
        "--vertical-crop-lower",
        type=float,
        default=0.0,
        help="Vertical crop lower fraction (0.0 to 1.0, default: 0.0)",
    )
    parser.add_argument(
        "--vertical-crop-upper",
        type=float,
        default=1.0,
        help="Vertical crop upper fraction (0.0 to 1.0, default: 1.0)",
    )
    parser.add_argument(
        "--horizontal-crop-lower",
        type=float,
        default=0.0,
        help="Horizontal crop lower fraction (0.0 to 1.0, default: 0.0)",
    )
    parser.add_argument(
        "--horizontal-crop-upper",
        type=float,
        default=1.0,
        help="Horizontal crop upper fraction (0.0 to 1.0, default: 1.0)",
    )

    args = parser.parse_args()

    # Create InPaths
    if args.nd2_file is not None:
        in_files = InPaths()
        in_files.nd2_file = args.nd2_file
    else:
        in_files = InPaths()

    # Get defaults from gen_default_run_params
    default_params = gen_default_run_params(in_files)

    # Override defaults with command-line arguments
    image_start = (
        args.image_start if args.image_start is not None else default_params.image_start
    )
    image_end = (
        args.image_end if args.image_end is not None else default_params.image_end
    )

    if args.fov_list is not None:
        fov_list = FOVList(args.fov_list)
    else:
        fov_list = default_params.fov_list

    # Create RunParams with all parameters
    run_params = RunParams(
        image_start=image_start,
        image_end=image_end,
        fov_list=fov_list,
        stabilize=args.stabilize,
        rotate=args.rotate,
        vertical_crop_lower=args.vertical_crop_lower,
        vertical_crop_upper=args.vertical_crop_upper,
        horizontal_crop_lower=args.horizontal_crop_lower,
        horizontal_crop_upper=args.horizontal_crop_upper,
    )

    # Create OutPaths
    out_paths = OutPaths(
        tiff_folder=args.tiff_folder,
        timetable=args.timetable,
    )

    nd2ToTIFF(in_files, run_params, out_paths)
