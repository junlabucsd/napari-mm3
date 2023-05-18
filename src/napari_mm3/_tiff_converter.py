from cgitb import reset
import os
import napari
import copy
import dask.array as da
import json
import nd2reader
import tifffile as tiff
import re
import io
import numpy as np

from pathlib import Path
from skimage import io
from napari.utils import progress
from magicgui.widgets import Container, FileEdit, CheckBox, PushButton, FloatSpinBox
from ._deriving_widgets import FOVChooser, TimeRangeSelector, warning, information


def get_nd2_fovs(data_path):
    with nd2reader.reader.ND2Reader(str(data_path)) as nd2f:
        return nd2f.sizes["v"]


def get_nd2_times(data_path):
    with nd2reader.reader.ND2Reader(str(data_path)) as nd2f:
        return nd2f.sizes["t"]


def get_bioformats_times(data_path):
    posix_path = data_path.as_posix()
    print("data path: " + posix_path)
    md = bioformats.get_omexml_metadata(path=posix_path)
    xml_data = bioformats.OMEXML(md)

    times = xml_data.image(0).Pixels.SizeT
    # javabridge.kill_vm()
    return times


def get_bioformats_fovs(data_path):

    posix_path = data_path.as_posix()
    print("data path: " + posix_path)
    md = bioformats.get_omexml_metadata(path=posix_path)
    xml_data = bioformats.OMEXML(md)

    fovs = xml_data.image_count
    return fovs


def nd2_iter(nd2f: nd2reader.ND2Reader, time_range, fov_list):
    """
    Iterates over the contents of an ND2File.

    params:
        nd2f: The ND2Reader object to use.

    returns:
        t: the time-index of the returned frame.
        fov: the fov-index of the returned frame.
        image_data: the image at the given time/fov.
    """
    # TODO: Move this into the UI code.
    fov_list = [fov - 1 for fov in fov_list]
    nd2_fov_list = set(range(0, nd2f.sizes["v"]))
    if fov_list == []:
        fov_list = nd2_fov_list
    nd2_time_range = set(range(0, nd2f.sizes["t"]))

    valid_fovs = set(fov_list).intersection(nd2_fov_list)
    if valid_fovs != set(fov_list):
        information("The following FOVs were not in the nd2, and thus were omitted:")
        information(set(fov_list) - valid_fovs)
    nd2f.iter_axes = "t"
    for fov in valid_fovs:
        nd2f.default_coords["v"] = fov
        for t in nd2_time_range:
            if t not in time_range:
                continue
            image_data = nd2f[t]
            yield t, fov, image_data


def bioformats_iter(data_path: Path, time_range, fov_list):
    """
    Iterates over the contents of a bioformats-compatible file.

    params:
        filepath: The bioformats object to use.

    returns:
        t: the time-index of the returned frame.
        fov: the fov-index of the returned frame.
        image_data: the image at the given time/fov.
    """
    import javabridge
    import bioformats

    javabridge.start_vm(class_path=bioformats.JARS)

    # TODO: Move this into the UI code.
    fov_list = [fov - 1 for fov in fov_list]
    base_fov_list = set(range(0, get_bioformats_fovs(data_path)))
    if fov_list == []:
        fov_list = base_fov_list
    base_time_range = set(range(0, get_bioformats_times(data_path)))

    valid_fovs = set(fov_list).intersection(base_fov_list)
    if valid_fovs != set(fov_list):
        information("The following FOVs were not in the nd2, and thus were omitted:")
        information(set(fov_list) - valid_fovs)

    for fov in valid_fovs:
        for t in base_time_range:
            if t not in time_range:
                continue

            image_data = bioformats.load_image(str(data_path), t=t, series=fov)
            yield t, fov, image_data
    javabridge.kill_vm()


def bioformats_import(
    data_path: Path,
    tif_dir: str,
    tif_compress: int,
    image_start: int,
    image_end: int,
    vertical_crop=None,
    tworow_crop=None,
    fov_list=[],
):
    if not os.path.exists(tif_dir):
        os.makedirs(tif_dir)

    file_prefix = os.path.split(os.path.splitext(data_path)[0])[1]
    information("Extracting {file_prefix} ...")
    # Extraction range is the time points that will be taken out.
    time_range = range(image_start, image_end)
    for t_id, fov_id, image_data in bioformats_iter(
        data_path, time_range=time_range, fov_list=fov_list
    ):
        # timepoint and fov output name (1 indexed rather than 0 indexed)
        t, fov = t_id + 1, fov_id + 1

        # add extra axis to make below slicing simpler. removed automatically if only one color
        if len(image_data.shape) < 3:
            image_data = np.expand_dims(image_data, axis=0)

        # crop tiff if specified. Lots of flags for if there are double rows or  multiple colors
        if tworow_crop:
            crop1_y1, crop1_y2 = tworow_crop[0][0], tworow_crop[0][1]
            crop2_y1, crop2_y2 = tworow_crop[0][0], tworow_crop[0][1]
            image_upper_row = image_data[:, crop1_y1:crop2_y2, :]
            image_lower_row = image_data[:, crop2_y1:crop2_y2, :]

            upper_row_filename = f"{file_prefix}_t{t:04d}xy{fov:02d}_1.tif"
            information("Saving %s." % tif_filename)
            # TODO: Make the channel order correct, here and below.
            tiff.imsave(
                tif_dir / upper_row_filename,
                image_upper_row,
                compress=tif_compress,
                photometric="minisblack",
            )

            # cut and save bottom row
            lower_row_filename = f"{file_prefix}_t{t:04d}xy{fov:02d}_2.tif"
            information("Saving %s." % lower_row_filename)
            tiff.imsave(
                tif_dir / lower_row_filename,
                image_lower_row,
                description=metadata_json,
                photometric="minisblack",
            )
            continue
        # for just a simple crop
        elif vertical_crop:
            nc, H, W = image_data.shape
            ylo = int(vertical_crop[0] * H)
            yhi = int(vertical_crop[1] * H)
            image_data = image_data[:, ylo:yhi, :]

        tif_filename = f"{file_prefix}_t{t:04d}xy{fov:02d}.tif"
        information("Saving %s." % tif_filename)
        tiff.imsave(
            tif_dir / tif_filename,
            image_data,
            compress=tif_compress,
            photometric="minisblack",
        )


def nd2ToTIFF(
    data_path: Path,
    tif_dir: str,
    tif_compress: int,
    image_start: int,
    image_end: int,
    vertical_crop=None,
    tworow_crop=None,
    fov_list=[],
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
        tif_compress: image compression level, 1-9
    """
    # set up image folders if they do not already exist
    if not os.path.exists(tif_dir):
        os.makedirs(tif_dir)

    nd2file = data_path
    file_prefix = os.path.split(os.path.splitext(nd2file)[0])[1]
    information("Extracting {file_prefix} ...")
    with nd2reader.reader.ND2Reader(str(nd2file)) as nd2f:
        starttime = nd2f.metadata["date"]  # starttime is jd
        planes = list(nd2f.metadata["channels"])
        if len(planes) > 1:
            nd2f.bundle_axes = ["c", "y", "x"]

        # Extraction range is the time points that will be taken out.
        time_range = range(image_start, image_end)
        for t_id, fov_id, image_data in nd2_iter(
            nd2f, time_range=time_range, fov_list=fov_list
        ):
            # timepoint and fov output name (1 indexed rather than 0 indexed)
            t, fov = t_id + 1, fov_id + 1
            try:
                milliseconds = copy.deepcopy(
                    image_data.metadata["events"][t_id]["time"]
                )
                acq_days = milliseconds / 1000.0 / 60.0 / 60.0 / 24.0
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
            if len(image_data.shape) < 3:
                image_data = np.expand_dims(image_data, axis=0)

            # crop tiff if specified. Lots of flags for if there are double rows or  multiple colors
            if tworow_crop:
                crop1_y1, crop1_y2 = tworow_crop[0][0], tworow_crop[0][1]
                crop2_y1, crop2_y2 = tworow_crop[0][0], tworow_crop[0][1]
                image_upper_row = image_data[:, crop1_y1:crop2_y2, :]
                image_lower_row = image_data[:, crop2_y1:crop2_y2, :]

                upper_row_filename = f"{file_prefix}_t{t:04d}xy{fov:02d}_1.tif"
                information("Saving %s." % tif_filename)
                # TODO: Make the channel order correct, here and below.
                tiff.imsave(
                    tif_dir / upper_row_filename,
                    image_upper_row,
                    description=metadata_json,
                    compress=tif_compress,
                    photometric="minisblack",
                )

                # cut and save bottom row
                lower_row_filename = f"{file_prefix}_t{t:04d}xy{fov:02d}_2.tif"
                information("Saving %s." % lower_row_filename)
                tiff.imsave(
                    tif_dir / lower_row_filename,
                    image_lower_row,
                    description=metadata_json,
                    compress=tif_compress,
                    photometric="minisblack",
                )
                continue
            # for just a simple crop
            elif vertical_crop:
                nc, H, W = image_data.shape
                ylo = int(vertical_crop[0] * H)
                yhi = int(vertical_crop[1] * H)
                image_data = image_data[:, ylo:yhi, :]

            tif_filename = f"{file_prefix}_t{t:04d}xy{fov:02d}.tif"
            information("Saving %s." % tif_filename)
            tiff.imsave(
                tif_dir / tif_filename,
                image_data,
                description=metadata_json,
                compress=tif_compress,
                photometric="minisblack",
            )


class TIFFExport(Container):
    """No good way to make this derive MM3Widget; have to roll a custom version here."""

    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()

        nd2files = list(Path(".").glob("*.nd2"))
        self.nd2files_found = len(nd2files) != 0
        if not self.nd2files_found:
            warning(
                "No nd2 files found in current directory. \nIf your data is in .nd2 format, launch napari from the directory containing the .nd2 file.\nNow looking for bioformats..."
            )

            global bioformats, javabridge
            import javabridge
            import bioformats

            javabridge.start_vm(class_path=bioformats.JARS)

            self.nd2file = ""
            self.valid_times = [1, 1]
            self.valid_fovs = [1, 1]
        else:
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
            label="Crop y min", value=0, min=0, max=0.5, step=0.01
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
        self.run_widget.clicked.connect(self.run)
        self.display_nd2_widget.clicked.connect(self.render_nd2)

        self.append(self.data_path_widget)
        self.append(self.exp_dir_widget)
        self.append(self.FOVs_range_widget)
        self.append(self.time_range_widget)
        self.append(self.upper_crop_widget)
        self.append(self.lower_crop_widget)
        self.append(self.display_nd2_widget)
        self.append(self.run_widget)

        self.set_data_path()
        self.set_fovs(list(range(self.valid_fovs[0], self.valid_fovs[1])))
        self.set_time_range()
        self.set_exp_dir()
        self.set_upper_crop()
        self.set_lower_crop()

        napari.current_viewer().window._status_bar._toggle_activity_dock(True)

    def run(self):
        if self.nd2files_found:
            nd2ToTIFF(
                self.data_path,
                self.exp_dir / "TIFF",
                tif_compress=5,  # TODO: assign from UI
                image_start=self.time_range[0] - 1,
                image_end=self.time_range[1],
                vertical_crop=[self.lower_crop, self.upper_crop],
                fov_list=self.fovs,
                tworow_crop=None,
            )
        else:
            bioformats_import(
                self.data_path,
                self.exp_dir / "TIFF",
                tif_compress=5,  # TODO: assign from UI
                image_start=self.time_range[0] - 1,
                image_end=self.time_range[1],
                vertical_crop=[self.lower_crop, self.upper_crop],
                fov_list=self.fovs,
            )

        information("Finished TIFF export")

    def render_nd2(self):
        viewer = self.viewer
        viewer.layers.clear()
        viewer.grid.enabled = True

        try:
            # nd2file = list(self.data_path.glob('*.nd2'))[0]
            nd2file = self.data_path
        except:
            warning(
                f"Could not find .nd2 file to display in directory {self.data_path.resolve()}"
            )
            return

        with nd2reader.reader.ND2Reader(str(nd2file)) as ndx:
            sizes = ndx.sizes

            if "t" not in sizes:
                sizes["t"] = 1
            if "z" not in sizes:
                sizes["z"] = 1
            if "c" not in sizes:
                sizes["c"] = 1
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

    def set_widget_bounds(self):
        if self.nd2files_found:
            self.valid_fovs = [1, get_nd2_fovs(self.data_path)]
            self.valid_times = [1, get_nd2_times(self.data_path)]
        else:
            self.valid_fovs = [1, get_bioformats_fovs(self.data_path)]
            self.valid_times = [1, get_bioformats_times(self.data_path)]
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
