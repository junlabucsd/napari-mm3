from cgitb import reset
import os
import napari
import copy
import dask.array as da
import json
import pims_nd2
import tifffile as tiff
import re
import io
import numpy as np
import datetime

from dask import delayed
from pathlib import Path
from skimage import io
from napari.utils import progress
from magicgui.widgets import Container, FileEdit, CheckBox, PushButton
from ._deriving_widgets import FOVChooser, TimeRangeSelector, information


def julian_day_number():
    """
    Need this to solve a bug in pims_nd2.nd2reader.ND2_Reader instance initialization.
    The bug is in /usr/local/lib/python2.7/site-packages/pims_nd2/ND2SDK.py in function `jdn_to_datetime_local`, when the year number in the metadata (self._lim_metadata_desc) is not in the correct range. This causes a problem when calling self.metadata.
    https://en.wikipedia.org/wiki/Julian_day
    """
    dt = datetime.datetime.now()
    tt = dt.timetuple()
    jdn = (
        (1461.0 * (tt.tm_year + 4800.0 + (tt.tm_mon - 14.0) / 12)) / 4.0
        + (367.0 * (tt.tm_mon - 2.0 - 12.0 * ((tt.tm_mon - 14.0) / 12))) / 12.0
        - (3.0 * ((tt.tm_year + 4900.0 + (tt.tm_mon - 14.0) / 12.0) / 100.0)) / 4.0
        + tt.tm_mday
        - 32075
    )

    return jdn


def get_nd2_fovs(exp_dir):
    nd2files = list(exp_dir.glob("*.nd2"))

    for nd2_file in nd2files:
        with pims_nd2.ND2_Reader(nd2_file) as nd2f:
            return (1, nd2f.sizes["m"])


def get_nd2_times(exp_dir):
    nd2files = list(exp_dir.glob("*.nd2"))

    for nd2_file in nd2files:
        with pims_nd2.ND2_Reader(nd2_file) as nd2f:
            return (1, nd2f.sizes["t"])


def nd2ToTIFF(
    experiment_directory: Path,
    tif_dir: str,
    tif_compress: int,
    image_start: int,
    image_end: int,
    vertical_crop=None,
    tworow_crop=None,
    fov_list=[],
    reset_numbering=False,
):
    """
    This script converts a Nikon Elements .nd2 file to individual TIFF files per time point. Multiple color planes are stacked in each time point to make a multipage TIFF.

    params:
        experiment_directory: Path to the experimental data
        tif_dir: Where to put the TIFFs when we are done.
        tif_filename: A prefix for the output tifs
        vertical_crop: [ymin, ymax]. Percentage crop. Optional.
        tworow_crop: [[y1_min, y1_max], [y1_min, y2_max]]. Used for cropping if you have multiple rows; currently only two are supported.
        FOVs: String specifying a range (or a single number) of FOVs to convert to nd2
        image_start, image_end: Image range that we want to turn into TIFFs (inclusive)
        tif_compress: image compression level, 1-9
    """

    # Load the project parameters file
    information("Loading experiment parameters.")

    # set up image and analysis folders if they do not already exist
    if not os.path.exists(tif_dir):
        os.makedirs(tif_dir)

    # Load ND2 files into a list for processing
    information(f"Experiment directory: {experiment_directory.name}")
    nd2files = list(experiment_directory.glob("*.nd2"))
    information(f"Found {len(nd2files)} files to analyze in experiment directory.")

    for nd2_file in nd2files:
        file_prefix = os.path.split(os.path.splitext(nd2_file)[0])[1]
        information("Extracting {file_prefix} ...")

        # load the nd2. the nd2f file object has lots of information thanks to pims
        with pims_nd2.ND2_Reader(nd2_file) as nd2f:
            try:
                starttime = nd2f.metadata["time_start_jdn"]  # starttime is jd
                information("Starttime got from nd2 metadata.")
            except ValueError:
                # problem with the date
                jdn = julian_day_number()
                nd2f._lim_metadata_desc.dTimeStart = jdn
                starttime = nd2f.metadata["time_start_jdn"]  # starttime is jd
                information("Starttime found from lim.")

            # get the color names out. Kinda roundabout way.
            planes = [
                nd2f.metadata[md]["name"]
                for md in nd2f.metadata
                if md[0:6] == "plane_" and not md == "plane_count"
            ]

            # this insures all colors will be saved when saving tiff
            if len(planes) > 1:
                nd2f.bundle_axes = ["c", "y", "x"]

            # extraction range is the time points that will be taken out. Note the indexing,
            # it is zero indexed to grab from nd2, but TIFF naming starts at 1.
            # if there is more than one FOV (len(nd2f) != 1), make sure the user input
            # last time index is before the actual time index. Ignore it.
            image_start = max(1, image_start)
            if not image_end:
                image_end = len(nd2f)
            elif len(nd2f) > 1:
                image_end = min(len(nd2f), image_end)
            extraction_range = range(image_start, image_end + 1)

            # loop through time points
            for t in progress(extraction_range):
                # timepoint output name (1 indexed rather than 0 indexed)
                t_id = t - 1
                # set counter for FOV output name
                # fov = fov_naming_start
                out_fov_number = 0
                for fov_id in range(0, nd2f.sizes["m"]):  # for every FOV
                    # fov_id is the fov index according to elements, fov is the output fov ID
                    fov = fov_id + 1

                    # skip FOVs as specified above
                    if len(fov_list) > 0 and not (fov in fov_list):
                        continue

                    # Only want to increment this if we are saving the current image
                    if reset_numbering:
                        out_fov_number += 1
                    else:
                        out_fov_number = fov

                    # set the FOV we are working on in the nd2 file object
                    nd2f.default_coords["m"] = fov_id

                    # get time picture was taken
                    seconds = copy.deepcopy(nd2f[t_id].metadata["t_ms"]) / 1000.0
                    minutes = seconds / 60.0
                    hours = minutes / 60.0
                    days = hours / 24.0
                    acq_time = starttime + days

                    # get physical location FOV on stage
                    x_um = nd2f[t_id].metadata["x_um"]
                    y_um = nd2f[t_id].metadata["y_um"]

                    # make dictionary which will be the metdata for this TIFF
                    metadata_t = {
                        "fov": fov,
                        "t": t,
                        "jd": acq_time,
                        "x": x_um,
                        "y": y_um,
                        "planes": planes,
                    }
                    metadata_json = json.dumps(metadata_t)

                    # get the pixel information
                    image_data = nd2f[t_id]

                    # crop tiff if specified. Lots of flags for if there are double rows or  multiple colors
                    if vertical_crop or tworow_crop:
                        # add extra axis to make below slicing simpler.
                        if len(image_data.shape) < 3:
                            image_data = np.expand_dims(image_data, axis=0)

                        # for dealing with two rows of channel
                        if tworow_crop:
                            # cut and save top row
                            image_data_one = image_data[
                                :, tworow_crop[0][0] : tworow_crop[0][1], :
                            ]
                            tif_filename = file_prefix + "_t%04dxy%02d_1.tif" % (
                                t,
                                out_fov_number,
                            )
                            information("Saving %s." % tif_filename)
                            tiff.imsave(
                                tif_dir / tif_filename,
                                image_data_one,
                                description=metadata_json,
                                compress=tif_compress,
                                photometric="minisblack",
                            )

                            # cut and save bottom row
                            metadata_t["fov"] = fov  # update metdata
                            metadata_json = json.dumps(metadata_t)
                            image_data_two = image_data[
                                :, tworow_crop[1][0] : tworow_crop[1][1], :
                            ]
                            tif_filename = file_prefix + "_t%04dxy%02d_2.tif" % (
                                t,
                                out_fov_number,
                            )
                            information("Saving %s." % tif_filename)
                            tiff.imsave(
                                tif_dir / tif_filename,
                                image_data_two,
                                description=metadata_json,
                                compress=tif_compress,
                                photometric="minisblack",
                            )

                        # for just a simple crop
                        elif vertical_crop:
                            nc, H, W = image_data.shape
                            ylo = int(vertical_crop[0] * H)
                            yhi = int(vertical_crop[1] * H)
                            image_data = image_data[:, ylo:yhi, :]

                            # save the tiff
                            tif_filename = file_prefix + "_t%04dxy%02d.tif" % (
                                t,
                                out_fov_number,
                            )
                            information("Saving %s." % tif_filename)
                            tiff.imsave(
                                tif_dir / tif_filename,
                                image_data,
                                description=metadata_json,
                                compress=tif_compress,
                                photometric="minisblack",
                            )

                    else:  # just save the image if no cropping was done.
                        tif_filename = file_prefix + "_t%04dxy%02d.tif" % (
                            t,
                            out_fov_number,
                        )
                        information("Saving %s." % tif_filename)
                        tiff.imsave(
                            tif_dir / tif_filename,
                            image_data,
                            description=metadata_json,
                            compress=tif_compress,
                            photometric="minisblack",
                        )

                    # increase FOV counter
                    fov += 1


class Nd2ToTIFF(Container):
    """No good way to make this derive MM3Widget; have to roll a custom version here."""

    def __init__(self):
        super().__init__()

        self.valid_times = get_nd2_times(Path("."))
        self.valid_fovs = get_nd2_fovs(Path("."))

        self.experiment_directory_widget = FileEdit(
            label="experiment_directory",
            value=Path("."),
            tooltip="Directory within which all your nd2 files are located.",
        )
        self.image_directory_widget = FileEdit(
            label="image_directory",
            value=Path("./TIFF"),
            tooltip="Directory within which to put your TIFFs",
        )
        self.FOVs_range_widget = FOVChooser(self.valid_fovs)
        self.time_range_widget = TimeRangeSelector(self.valid_times)
        self.display_after_export_widget = CheckBox(label="display after export")
        self.reset_numbering_widget = CheckBox(
            label="reset numbering",
            tooltip="Whether or not to preserve FOV numbering in the TIFF filenames. Use this to remove unwanted FOVs (e.g, blank FOVs)",
            value=True,
        )
        self.run_widget = PushButton(text="run")

        self.experiment_directory_widget.changed.connect(self.set_experiment_directory)
        self.experiment_directory_widget.changed.connect(self.set_widget_bounds)
        self.image_directory_widget.changed.connect(self.set_image_directory)
        self.FOVs_range_widget.connect_callback(self.set_fovs)
        self.time_range_widget.changed.connect(self.set_time_range)
        self.run_widget.clicked.connect(self.run)

        self.append(self.experiment_directory_widget)
        self.append(self.image_directory_widget)
        self.append(self.FOVs_range_widget)
        self.append(self.time_range_widget)
        self.append(self.display_after_export_widget)
        self.append(self.reset_numbering_widget)
        self.append(self.run_widget)

        self.set_experiment_directory()
        self.set_fovs(list(range(self.valid_fovs[0], self.valid_fovs[1] + 1)))
        self.set_time_range()
        self.set_image_directory()

        napari.current_viewer().window._status_bar._toggle_activity_dock(True)

    def run(self):
        nd2ToTIFF(
            self.experiment_directory,
            self.image_directory,
            tif_compress=5,  # TODO: assign from UI
            image_start=self.time_range[0],
            image_end=self.time_range[1] + 1,
            vertical_crop=None,  # TODO: assign from UI
            fov_list=self.fovs,
            tworow_crop=None,
            reset_numbering=self.reset_numbering_widget.value,
        )

        if self.display_after_export_widget.value:
            self.render_images()
        information("Finished TIFF export")

    def render_images(self):
        viewer = napari.current_viewer()
        viewer.layers.clear()

        image_name_list = [
            filename.name for filename in self.image_directory.glob("*xy*")
        ]
        fov_regex = re.compile(r"xy\d*",re.IGNORECASE)
        fovs = list(
            sorted(
                set(
                    int(fov_regex.search(filename).group()[2:])
                    for filename in image_name_list
                )
            )
        )
        if self.fovs:
            fovs = self.fovs

        viewer.grid.enabled = True
        viewer.grid.shape = (-1, 4)

        # Print out results!
        for fov_id in fovs:
            # TODO: Can allow xy in any position via regex! But it currently does not
            found_files = self.image_directory.glob(f"*xy{fov_id:02d}.tif")

            found_files = sorted(found_files)  # should sort by timepoint

            sample = io.imread(found_files[0])

            lazy_imread = delayed(io.imread)  # lazy reader
            lazy_arrays = [lazy_imread(fn) for fn in found_files]
            dask_arrays = [
                da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
                for delayed_reader in lazy_arrays
            ]
            # Stack into one large dask.array
            stack = da.stack(dask_arrays, axis=0)

            viewer.add_image(stack, name="FOV %02d" % fov_id, contrast_limits=[90, 250])
            # viewer.add_image(stack,name='FOV %02d' % fov_id)

    def set_widget_bounds(self):
        self.valid_fovs = get_nd2_fovs(self.experiment_directory)
        self.FOVs_range_widget.max_FOV = min(self.valid_fovs)
        self.FOVs_range_widget.max_FOV = max(self.valid_fovs)
        self.FOVs_range_widget.label = (
            f"FOVs ({min(self.valid_fovs)}-{max(self.valid_fovs)}"
        )
        self.FOVs_range_widget.value = f"{min(self.valid_fovs)}-{max(self.valid_fovs)}"

        self.valid_times = get_nd2_times(self.experiment_directory)
        self.time_range_widget.start.min = min(self.valid_times)
        self.time_range_widget.start.max = max(self.valid_times)
        self.time_range_widget.stop.min = min(self.valid_times)
        self.time_range_widget.stop.max = max(self.valid_times)
        self.time_range_widget.start.value = min(self.valid_times)
        self.time_range_widget.stop.value = max(self.valid_times)

    def set_fovs(self, fovs):
        self.fovs = fovs

    def set_image_directory(self):
        self.image_directory = self.image_directory_widget.value

    def set_experiment_directory(self):
        self.experiment_directory = self.experiment_directory_widget.value

    def set_time_range(self):
        self.time_range = (
            self.time_range_widget.value.start,
            self.time_range_widget.value.stop,
        )
