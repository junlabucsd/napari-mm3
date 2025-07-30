import argparse
import copy
import datetime
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import napari
import nd2
import numpy as np
import tifffile as tiff
from magicgui.widgets import Container, FileEdit, FloatSpinBox, PushButton
from napari import Viewer

from ._deriving_widgets import (
    FOVChooser,
    FOVList,
    MM3Container2,
    TimeRangeSelector,
    information,
    range_string_to_indices,
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
        return nd2f.sizes["T"]


def nd2_iter(nd2f: nd2.ND2File, time_range, fov_list):
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
    nd2_time_range = set(range(0, nd2f.sizes["T"]))

    # if only 1 fov, then just yield the single FOV.
    if "P" not in nd2f.sizes:
        im = nd2f.asarray()
        for t in nd2_time_range:
            if t not in time_range:
                continue
            image_data = im[t]
            yield t, 0, image_data
        return

    nd2_fov_list = set(range(0, nd2f.sizes["P"]))
    # TODO: Move this into the UI code.
    fov_list = [fov - 1 for fov in fov_list]
    if fov_list != []:
        valid_fovs = set(fov_list).intersection(nd2_fov_list)
    else:
        valid_fovs = nd2_fov_list

    if valid_fovs != set(fov_list):
        information("The following FOVs were not in the nd2, and thus were omitted:")
        information(set(fov_list) - valid_fovs)

    for fov in valid_fovs:
        im = nd2f.asarray(fov)
        for t in nd2_time_range:
            if t not in time_range:
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
        fov_list=FOVList(range(1, total_fovs + 1)),
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
    if not os.path.exists(out_paths.tif_dir):
        os.makedirs(out_paths.tif_dir)

    if not os.path.exists(Path(".") / "analysis"):
        os.makedirs(Path(".") / "analysis")

    nd2file = in_paths.nd2_file
    file_prefix = os.path.split(os.path.splitext(nd2file)[0])[1]
    information("Extracting {file_prefix} ...")
    with nd2.ND2File(str(nd2file)) as nd2f:
        # load in the time table.
        # TODO: Add analysis
        write_timetable(nd2f, Path(".") / "analysis" / "timetable.json")
        starttime = nd2f.text_info["date"]
        starttime = datetime.datetime.strptime(starttime, "%m/%d/%Y %I:%M:%S %p")

        planes = nd2f.sizes["C"]

        # Extraction range is the time points that will be taken out.
        time_range = range(run_params.image_start, run_params.image_end)
        for t_id, fov_id, image_data in nd2_iter(
            nd2f, time_range=time_range, fov_list=run_params.fov_list
        ):
            # timepoint and fov output name (1 indexed rather than 0 indexed)
            t, fov = t_id + 1, fov_id + 1
            try:
                milliseconds = copy.deepcopy(nd2f.events()[t_id * 2]["Time [s]"])
                acq_days = milliseconds / 60.0 / 60.0 / 24.0
                acq_time = starttime.timestamp() + acq_days
            except IndexError:
                acq_time = None

            # make dictionary which will be the metdata for this TIFF
            metadata_json = json.dumps(
                {
                    "fov": fov,
                    "t": t,
                    "jd": acq_time,
                    "planes": planes,
                }
            )

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

            tif_filename = f"{file_prefix}_t{t:04d}xy{fov:02d}.tif"
            information("Saving %s." % tif_filename)
            tiff.imwrite(
                out_paths.tif_dir / tif_filename,
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


class TIFFExport2(Container):
    """No good way to make this derive MM3Widget; have to roll a custom version here."""

    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()

        nd2files = list(Path(".").glob("*.nd2"))
        self.nd2files_found = len(nd2files) != 0

        self.nd2file = nd2files[0]
        self.valid_times = [1, get_nd2_times(self.nd2file)]
        self.valid_fovs = [1, get_nd2_fovs(self.nd2file)]

        self.data_path_widget = FileEdit(
            label="data_path",
            value=self.nd2file,
            mode="r",
            tooltip="Path to raw data (.nd2 or other Bio-Formats supported type)",
        )
        self.exp_dir_widget = FileEdit(
            label="experiment_directory",
            value=Path("."),
            mode="d",
            tooltip="Directory within which to put your TIFFs",
        )
        self.FOVs_range_widget = FOVChooser(self.valid_fovs)
        self.time_range_widget = TimeRangeSelector(self.valid_times)
        self.upper_crop_widget = FloatSpinBox(
            label="Crop y max", value=1, min=0, max=1, step=0.01
        )
        self.lower_crop_widget = FloatSpinBox(
            label="Crop y min", value=0, min=0, max=1, step=0.01
        )

        self.left_crop_widget = FloatSpinBox(
            label="Crop x min", value=0, min=0, max=1, step=0.01
        )
        self.right_crop_widget = FloatSpinBox(
            label="Crop x max", value=1, min=0, max=1, step=0.01
        )

        self.display_nd2_widget = PushButton(text="visualize all FOVs (.nd2 only)")
        self.run_widget = PushButton(text="run")

        self.data_path_widget.changed.connect(self.set_data_path)
        self.data_path_widget.changed.connect(self.set_widget_bounds)
        self.exp_dir_widget.changed.connect(self.set_exp_dir)
        self.FOVs_range_widget.connect_callback(self.set_fovs)
        self.time_range_widget.changed.connect(self.set_time_range)
        self.upper_crop_widget.changed.connect(self.set_upper_crop)
        self.lower_crop_widget.changed.connect(self.set_lower_crop)
        self.left_crop_widget.changed.connect(self.set_left_crop)
        self.right_crop_widget.changed.connect(self.set_right_crop)
        self.run_widget.clicked.connect(self.run)
        self.display_nd2_widget.clicked.connect(self.render_nd2)

        self.append(self.data_path_widget)
        self.append(self.exp_dir_widget)
        self.append(self.FOVs_range_widget)
        self.append(self.time_range_widget)
        self.append(self.lower_crop_widget)
        self.append(self.upper_crop_widget)
        self.append(self.left_crop_widget)
        self.append(self.right_crop_widget)
        self.append(self.display_nd2_widget)
        self.append(self.run_widget)

        self.set_data_path()
        self.set_fovs(list(range(self.valid_fovs[0], self.valid_fovs[1])))
        self.set_time_range()
        self.set_exp_dir()
        self.set_upper_crop()
        self.set_lower_crop()
        self.set_left_crop()
        self.set_right_crop()

        napari.current_viewer().window._status_bar._toggle_activity_dock(True)

    def run(self):
        nd2ToTIFF(
            self.data_path,
            self.exp_dir / "TIFF",
            image_start=self.time_range[0] - 1,
            image_end=self.time_range[1],
            vertical_crop_lower=self.lower_crop,
            vertical_crop_upper=self.upper_crop,
            horizontal_crop_lower=self.left_crop,
            horizontal_crop_upper=self.right_crop,
            fov_list=self.fovs,
        )

        information("Finished TIFF export")

    # TODO: Fix this one up.
    def render_nd2(self):
        viewer = self.viewer
        viewer.layers.clear()
        viewer.grid.enabled = True

        nd2file = self.data_path

        with nd2.ND2Reader(str(nd2file)) as ndx:
            sizes = ndx.sizes

            if "T" not in sizes:
                sizes["T"] = 1
            if "P" not in sizes:
                sizes["P"] = 1
            if "C" not in sizes:
                sizes["C"] = 1
            ndx.bundle_axes = "zcyx"
            ndx.iter_axes = "t"
            n = len(ndx)

            shape = (
                sizes["t"],
                sizes["z"],
                sizes["v"],
                sizes["c"],
                sizes["y"],
                sizes["x"],
            )
            image = np.zeros(shape, dtype=np.float32)

            for i in range(n):
                image[i] = ndx.get_frame(i)

        image = np.squeeze(image)

        viewer.add_image(image, channel_axis=1, colormap="gray")
        viewer.grid.shape = (-1, 3)
        viewer.dims.current_step = (0, 0)
        viewer.layers.link_layers()  ## allows user to set contrast limits for all FOVs at once

    def set_widget_bounds(self):
        self.valid_fovs = [1, get_nd2_fovs(self.data_path)]
        self.valid_times = [1, get_nd2_times(self.data_path)]
        self.FOVs_range_widget.max_FOV = min(self.valid_fovs)
        self.FOVs_range_widget.max_FOV = max(self.valid_fovs)
        self.FOVs_range_widget.label = (
            f"FOVs ({min(self.valid_fovs)}-{max(self.valid_fovs)}"
        )
        self.FOVs_range_widget.value = f"{min(self.valid_fovs)}-{max(self.valid_fovs)}"
        self.time_range_widget.label = (
            f"time range (frames {self.valid_times[0]}-{self.valid_times[1]})"
        )
        self.time_range_widget.start.min = min(self.valid_times)
        self.time_range_widget.start.max = max(self.valid_times)
        self.time_range_widget.stop.min = min(self.valid_times)
        self.time_range_widget.stop.max = max(self.valid_times)
        self.time_range_widget.start.value = min(self.valid_times)
        self.time_range_widget.stop.value = max(self.valid_times)

    def set_fovs(self, fovs):
        self.fovs = fovs

    def set_exp_dir(self):
        self.exp_dir = self.exp_dir_widget.value

    def set_data_path(self):
        self.data_path = self.data_path_widget.value

    def set_time_range(self):
        self.time_range = (
            self.time_range_widget.value.start,
            self.time_range_widget.value.stop,
        )

    def set_upper_crop(self):
        self.upper_crop = self.upper_crop_widget.value

    def set_lower_crop(self):
        self.lower_crop = self.lower_crop_widget.value

    def set_left_crop(self):
        self.left_crop = self.left_crop_widget.value

    def set_right_crop(self):
        self.right_crop = self.right_crop_widget.value


if __name__ == "__main__":
    cur_dir = Path(".")
    nd2files = list(cur_dir.glob("*.nd2"))
    nd2files_found = len(nd2files) != 1
    if not nd2files_found:
        print(
            "No ND2 files found. Please run this in a folder containing a single nd2 file."
        )
    nd2file = nd2files[0]

    end_time = get_nd2_times(nd2file)

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

    total_fovs = get_nd2_fovs(nd2file)
    if p.fovs == "":
        fovs = list(range(total_fovs))
    else:
        fovs = range_string_to_indices(p.fovs)
        for fov in fovs:
            if fov >= total_fovs:
                raise ValueError("Some FOVs are out of range for your nd2 file.")

    if (p.start_time < 0) or (p.end_time > end_time) or (p.start_time > p.end_time):
        raise ValueError("Times out of range")

    nd2ToTIFF(
        nd2file,
        cur_dir / "TIFF",
        image_start=p.start_time - 1,
        image_end=p.end_time - 1,
        vertical_crop=[0, 1],
        horizontal_crop=[0, 1],
        fov_list=fovs,
    )
