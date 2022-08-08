from magicgui import magic_factory, magicgui
from pathlib import Path
import multiprocessing
import napari
import os
import six
import tifffile as tiff
import h5py
import numpy as np

from ._function import information, warnings, load_specs, load_stack, segment_image

# Do segmentation for an channel time stack
def segment_chnl_stack(fov_id, peak_id):
    """
    For a given fov and peak (channel), do segmentation for all images in the
    subtracted .tif stack.

    Called by
    mm3_Segment.py

    Calls
    mm3.segment_image
    """

    information("Segmenting FOV %d, channel %d." % (fov_id, peak_id))

    # load subtracted images
    sub_stack = load_stack(
        params, fov_id, peak_id, color="sub_{}".format(params["phase_plane"])
    )

    # # set up multiprocessing pool to do segmentation. Will do everything before going on.
    # pool = Pool(processes=params['num_analyzers'])

    # # send the 3d array to multiprocessing
    # segmented_imgs = pool.map(segment_image, sub_stack, chunksize=8)

    # pool.close() # tells the process nothing more will be added.
    # pool.join() # blocks script until everything has been processed and workers exit

    # image by image for debug
    segmented_imgs = []
    for sub_image in sub_stack:
        segmented_imgs.append(segment_image(params, sub_image))

    # stack them up along a time axis
    segmented_imgs = np.stack(segmented_imgs, axis=0)
    segmented_imgs = segmented_imgs.astype("uint8")

    # save out the segmented stack
    if params["output"] == "TIFF":
        seg_filename = params["experiment_name"] + "_xy%03d_p%04d_%s.tif" % (
            fov_id,
            peak_id,
            params["seg_img"],
        )
        tiff.imsave(
            os.path.join(params["seg_dir"], seg_filename), segmented_imgs, compress=5
        )
        # if fov_id==1:
        viewer = napari.current_viewer()

        viewer.add_labels(
            segmented_imgs,
            name="Segmented"
            + "_xy%03d_p%04d" % (fov_id, peak_id)
            + "_"
            + str(params["seg_img"])
            + ".tif",
            visible=True,
        )

    if params["output"] == "HDF5":
        h5f = h5py.File(os.path.join(params["hdf5_dir"], "xy%03d.hdf5" % fov_id), "r+")

        # put segmented channel in correct group
        h5g = h5f["channel_%04d" % peak_id]

        # delete the dataset if it exists (important for debug)
        if "p%04d_%s" % (peak_id, params["seg_img"]) in h5g:
            del h5g["p%04d_%s" % (peak_id, params["seg_img"])]

        h5ds = h5g.create_dataset(
            "p%04d_%s" % (peak_id, params["seg_img"]),
            data=segmented_imgs,
            chunks=(1, segmented_imgs.shape[1], segmented_imgs.shape[2]),
            maxshape=(None, segmented_imgs.shape[1], segmented_imgs.shape[2]),
            compression="gzip",
            shuffle=True,
            fletcher32=True,
        )
        h5f.close()

    information("Saved segmented channel %d." % peak_id)

    return True


def segmentOTSU(params):

    information("Loading experiment parameters.")
    p = params

    if p["FOV"]:
        if "-" in p["FOV"]:
            user_spec_fovs = range(
                int(p["FOV"].split("-")[0]), int(p["FOV"].split("-")[1]) + 1
            )
        else:
            user_spec_fovs = [int(val) for val in p["FOV"].split(",")]
    else:
        user_spec_fovs = []

    # create segmenteation and cell data folder if they don't exist
    if not os.path.exists(p["seg_dir"]) and p["output"] == "TIFF":
        os.makedirs(p["seg_dir"])
    if not os.path.exists(p["cell_dir"]):
        os.makedirs(p["cell_dir"])

    # set segmentation image name for saving and loading segmented images
    p["seg_img"] = "seg_otsu"

    # load specs file
    specs = load_specs(params)

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    information("Segmenting %d FOVs." % len(fov_id_list))

    ### Do Segmentation by FOV and then peak #######################################################
    information("Segmenting channels using Otsu method.")

    for fov_id in fov_id_list:
        # determine which peaks are to be analyzed (those which have been subtracted)
        ana_peak_ids = []
        for peak_id, spec in six.iteritems(specs[fov_id]):
            if (
                spec == 1
            ):  # 0 means it should be used for empty, -1 is ignore, 1 is analyzed
                ana_peak_ids.append(peak_id)
        ana_peak_ids = sorted(ana_peak_ids)  # sort for repeatability

        for peak_id in ana_peak_ids:
            # send to segmentation
            segment_chnl_stack(fov_id, peak_id)

    information("Finished segmentation.")


@magicgui(
    auto_call=True,
    first_opening_size=dict(widget_type="SpinBox", step=1),
    OTSU_threshold=dict(widget_type="FloatSpinBox", min=0, max=2, step=0.01),
)
def DebugOtsu(
    OTSU_threshold=1.0,
    first_opening_size: int = 2,
    distance_threshold: int = 2,
    second_opening_size: int = 1,
    min_object_size: int = 25,
):

    warnings.filterwarnings("ignore", "The probability range is outside [0, 1]")

    params["segment"]["OTSU_threshold"] = OTSU_threshold
    params["segment"]["first_opening_size"] = first_opening_size
    params["segment"]["distance_threshold"] = distance_threshold
    params["segment"]["second_opening_size"] = second_opening_size
    params["segment"]["min_object_size"] = min_object_size

    p = params

    if p["FOV"]:
        if "-" in p["FOV"]:
            user_spec_fovs = range(
                int(p["FOV"].split("-")[0]), int(p["FOV"].split("-")[1]) + 1
            )
        else:
            user_spec_fovs = [int(val) for val in p["FOV"].split(",")]
    else:
        user_spec_fovs = []

    # set segmentation image name for saving and loading segmented images
    p["seg_img"] = "seg_otsu"

    # load specs file
    specs = load_specs(params)

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    ### Do Segmentation by FOV and then peak #######################################################

    for fov_id in fov_id_list:
        fov_id_d = fov_id
        # determine which peaks are to be analyzed (those which have been subtracted)
        for peak_id, spec in six.iteritems(specs[fov_id]):
            if (
                spec == 1
            ):  # 0 means it should be used for empty, -1 is ignore, 1 is analyzed
                peak_id_d = peak_id
                break
        break

    ## pull out first fov & peak id with cells
    sub_stack = load_stack(
        params, fov_id_d, peak_id_d, color="sub_{}".format(params["phase_plane"])
    )

    # image by image for debug
    segmented_imgs = []
    for sub_image in sub_stack:
        segmented_imgs.append(segment_image(params, sub_image))

    # stack them up along a time axis
    segmented_imgs = np.stack(segmented_imgs, axis=0)
    segmented_imgs = segmented_imgs.astype("uint8")

    viewer = napari.current_viewer()
    viewer.layers.clear()
    viewer.add_labels(segmented_imgs, name="Labels")


@magic_factory(
    experiment_directory={"mode": "d"}, phase_plane={"choices": ["c1", "c2", "c3"]}
)
def SegmentOtsu(
    experiment_name: str = "",
    experiment_directory=Path(),
    image_directory: str = "TIFF/",
    FOV: str = "1-5",
    interactive: bool = False,
    phase_plane="c1",
    OTSU_threshold=1.0,
    first_opening_size: int = 2,
    distance_threshold: int = 2,
    second_opening_size: int = 1,
    min_object_size: int = 25,
):

    global params
    params = dict()
    params["experiment_name"] = experiment_name
    params["experiment_directory"] = experiment_directory
    params["image_directory"] = image_directory
    params["analysis_directory"] = "analysis"
    params["output"] = "TIFF"
    params["FOV"] = FOV
    params["interactive"] = interactive
    params["phase_plane"] = phase_plane

    params["segment"] = dict()
    params["segment"]["OTSU_threshold"] = OTSU_threshold
    params["segment"]["first_opening_size"] = first_opening_size
    params["segment"]["distance_threshold"] = distance_threshold
    params["segment"]["second_opening_size"] = second_opening_size
    params["segment"]["min_object_size"] = min_object_size
    params["num_analyzers"] = multiprocessing.cpu_count()

    # useful folder shorthands for opening files
    params["TIFF_dir"] = os.path.join(
        params["experiment_directory"], params["image_directory"]
    )
    params["ana_dir"] = os.path.join(
        params["experiment_directory"], params["analysis_directory"]
    )
    params["hdf5_dir"] = os.path.join(params["ana_dir"], "hdf5")
    params["chnl_dir"] = os.path.join(params["ana_dir"], "channels")
    params["empty_dir"] = os.path.join(params["ana_dir"], "empties")
    params["sub_dir"] = os.path.join(params["ana_dir"], "subtracted")
    params["seg_dir"] = os.path.join(params["ana_dir"], "segmented")
    params["cell_dir"] = os.path.join(params["ana_dir"], "cell_data")
    params["track_dir"] = os.path.join(params["ana_dir"], "tracking")

    ## if debug is checked, clicking run will launch this new widget. need to pass fov & peak
    if params["interactive"]:
        viewer = napari.current_viewer()
        viewer.window.add_dock_widget(DebugOtsu, name="debugotsu")
    else:
        segmentOTSU(params)
