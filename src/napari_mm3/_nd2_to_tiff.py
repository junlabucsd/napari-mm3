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

from dask import delayed
from magicgui import magic_factory
from pathlib import Path
from skimage import io
from napari.utils import progress
from ._function import range_string_to_indices, information, julian_day_number


def nd2ToTIFF(
    experiment_directory: Path,
    tif_dir: str,
    tif_compress: int,
    image_start: int,
    image_end: int,
    vertical_crop=None,
    tworow_crop=None,
    fov_list=[],
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

    for nd2_file in progress(nd2files):
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

                for fov_id in range(0, nd2f.sizes["m"]):  # for every FOV
                    # fov_id is the fov index according to elements, fov is the output fov ID
                    fov = fov_id + 1

                    # skip FOVs as specified above
                    if len(fov_list) > 0 and not (fov in fov_list):
                        continue

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
                            tif_filename = file_prefix + "_t%04dxy%02d_1.tif" % (t, fov)
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
                            tif_filename = file_prefix + "_t%04dxy%02d_2.tif" % (t, fov)
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
                            tif_filename = file_prefix + "_t%04dxy%02d.tif" % (t, fov)
                            information("Saving %s." % tif_filename)
                            tiff.imsave(
                                tif_dir / tif_filename,
                                image_data,
                                description=metadata_json,
                                compress=tif_compress,
                                photometric="minisblack",
                            )

                    else:  # just save the image if no cropping was done.
                        tif_filename = file_prefix + "_t%04dxy%02d.tif" % (t, fov)
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


@magic_factory(
    experiment_directory={
        "mode": "d",
        "tooltip": "Directory within which all your data and analyses will be located.",
    },
    image_directory={
        "tooltip": "Required. Location (within working directory) for the input images. 'working directory/TIFF/' by default."
    },
    image_start={
        "tooltip": "Required. First time stamp for which we would like to do analysis."
    },
    image_end={
        "tooltip": "Required. Last time stamp for which we would like to do analysis."
    },
    FOVs_range={
        "tooltip": "Optional. Range of FOVs to include. By default, all will be processed."
    },
)
def Nd2ToTIFF(
    experiment_directory=Path(),
    image_directory=Path(),
    image_start: int = 1,
    image_end: int = 50,
    FOVs_range: str = "",
):
    """Converts an Nd2 file to a series of TIFFs.
    TODO: Range inference, or similar."""
    tif_dir = Path()
    fov_list = range_string_to_indices(FOVs_range)
    nd2ToTIFF(
        experiment_directory,
        tif_dir,
        tif_compress=5,
        image_start=image_start,
        image_end=image_end,
        vertical_crop=None,
        fov_list=fov_list,
        tworow_crop=None,
    )

    viewer = napari.current_viewer()
    viewer.layers.clear()

    image_name_list = [filename.name for filename in tif_dir.glob("*xy*")]
    fov_regex = re.compile(r"xy\d*")
    fovs = list(
        sorted(
            set(
                int(fov_regex.search(filename).group()[2:])
                for filename in image_name_list
            )
        )
    )
    if fov_list:
        fovs = fov_list

    # Print out results!
    for fov_id in fovs:
        # TODO: Can allow xy in any position via regex! But it currently does not
        found_files = tif_dir.glob(f"*xy{fov_id:02d}.tif")

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

    viewer.grid.enabled = True
    grid_w = int(len(fovs) / 17) + 1
    grid_h = int(len(fovs) / grid_w) + 1
    viewer.grid.shape = (-1, 4)

    print("Done.")
