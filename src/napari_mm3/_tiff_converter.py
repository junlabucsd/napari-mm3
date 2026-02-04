import copy
import datetime
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import nd2
import numpy as np
import tifffile as tiff
from magicgui.widgets import PushButton, SpinBox
from napari import Viewer
from napari.utils import progress
from scipy.ndimage import rotate, shift
from skimage.registration import phase_cross_correlation

from ._deriving_widgets import (
    FOVList,
    MM3Container2,
    information,
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
        nd2_time_range = set([0])

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
    return rotate(image_data, float(angle), axes=(-1, -2))


def stabilize_fov(
    raw_stack,
    upsample_factor=2,
):
    """
    Drift correction. Registers against first frame and last frame, weighting them in proportion to how close you are.
    Seems to work pretty well. Might be nice to have a third registration (middle) but it's prolly not worth it.
    Sometimes a registration agaisnt the frame ahlfway through works best.

    If people report issues, would be good to get this to use less memory.
    But, a typical raw stack is at most ~1GB, so this is a reasonable expectation.
    """
    print("Stabilizing")
    raw_stack = np.stack(raw_stack)  # [T, N, H, W]

    T, C, H, W = raw_stack.shape
    corrected_stack = np.empty_like(raw_stack)
    ref = raw_stack[T // 2, 0]  # Reference frame from channel 0

    shifts_f = []
    for i in range(0, T):
        shift_est, _, _ = phase_cross_correlation(
            ref, raw_stack[i, 0], upsample_factor=upsample_factor
        )
        shifts_f.append(shift_est)
    shifts_f = np.array(shifts_f)

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
    shifts = shifts_f

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


@dataclass
class RunParams:
    image_start: int
    image_end: int
    fov_list: FOVList
    stabilize: bool
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


def nd2ToTIFF(in_paths: InPaths, run_params: RunParams, out_paths: OutPaths):
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
    information(f"Extracting {file_prefix} ...")
    with nd2.ND2File(str(nd2file)) as nd2f:
        # load in the time table.
        write_timetable(nd2f, out_paths.timetable)
        starttime = nd2f.text_info["date"]
        starttime = parse_datetime_flexible(starttime)
        try:
            planes = nd2f.sizes["C"]
        except KeyError:
            planes = 1

        # if only 1 fov, then just analyze the single FOV.
        total_fovs = nd2f.sizes["P"] if ("P" in nd2f.sizes) else 1
        time_range_ids = list(range(run_params.image_start - 1, run_params.image_end))
        fov_list_ids = [fov_id - 1 for fov_id in run_params.fov_list]
        for fov_id in fov_list_ids:
            if "P" not in nd2f.sizes:
                image_data = nd2f.asarray()
                if "T" not in nd2f.sizes:
                    image_data = image_data[np.newaxis, np.newaxis, ...]
                else:
                    image_data = image_data[np.newaxis, ...]
            else:
                image_data = nd2f.asarray(fov_id)
                if "T" not in nd2f.sizes:
                    image_data = image_data[np.newaxis, ...]

            image_data = image_data[time_range_ids, 0, ...]

            if run_params.stabilize:
                image_data = stabilize_fov(image_data)

            if run_params.rotate != 0:
                image_data = fix_rotation(run_params.rotate, image_data)

            # add extra axis to make below slicing simpler. removed automatically if only one color.
            # sanity check: what do if it's 1 channel/1 time?
            if len(image_data.shape) < 3:
                image_data = np.expand_dims(image_data, axis=0)

            # for just a simple crop
            if (run_params.vertical_crop_lower > 0.0) or (
                run_params.vertical_crop_upper < 1.0
            ):
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

            for t_idx, image_data_cur_t in enumerate(image_data):
                t_id = time_range_ids[t_idx]
                try:
                    milliseconds = copy.deepcopy(
                        nd2f.events()[2 * (t_id * total_fovs + fov_id)]["Time [s]"]
                    )
                    acq_days = milliseconds / 60.0 / 60.0 / 24.0
                    acq_time = starttime.timestamp() + acq_days
                except IndexError:
                    acq_time = None
                    print("No acquistion time found")
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
                #     information("Saving %s." % tif_filename)
                tiff.imwrite(
                    out_paths.tiff_folder / tif_filename,
                    data=image_data_cur_t,
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
        if "fov_img" in viewer.layers:
            layer = viewer.layers["fov_img"]
            layer.data = fov_img
        else:
            viewer.add_image(
                fov_img,
            )
        viewer.dims.current_step = pos


if __name__ == "__main__":
    in_files = InPaths()
    run_params: RunParams = gen_default_run_params(in_files)
    out_paths = OutPaths()

    nd2ToTIFF(in_files, run_params, out_paths)
