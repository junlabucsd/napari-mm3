import multiprocessing
from multiprocessing import Pool
import napari
import os
import six
import tifffile as tiff
import h5py
import numpy as np
import warnings

from magicgui.widgets import FloatSpinBox, SpinBox, PushButton, CheckBox
from scipy import ndimage as ndi
from skimage import segmentation, morphology
from skimage.filters import threshold_otsu
from napari.utils import progress

from ._deriving_widgets import (
    MM3Container,
    PlanePicker,
    FOVChooser,
    load_specs,
    information,
    load_stack_params,
    warning,
)

# Do segmentation for an channel time stack
def segment_chnl_stack(params, fov_id, peak_id, view_result: bool = False):
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
    sub_stack = load_stack_params(
        params, fov_id, peak_id, postfix="sub_{}".format(params["phase_plane"])
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
        if view_result:
            viewer = napari.current_viewer()

            viewer.grid.enabled = True

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


# segmentation algorithm
def segment_image(params, image):
    """Segments a subtracted image and returns a labeled image

    Parameters
    image : a ndarray which is an image. This should be the subtracted image

    Returns
    labeled_image : a ndarray which is also an image. Labeled values, which
        should correspond to cells, all have the same integer value starting with 1.
        Non labeled area should have value zero.
    """
    # load in segmentation parameters
    OTSU_threshold = params["segment"]["OTSU_threshold"]
    first_opening_size = params["segment"]["first_opening_size"]
    distance_threshold = params["segment"]["distance_threshold"]
    second_opening_size = params["segment"]["second_opening_size"]
    min_object_size = params["segment"]["min_object_size"]

    # threshold image
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            thresh = threshold_otsu(image)  # finds optimal OTSU threshhold value
    except:
        return np.zeros_like(image)

    threshholded = image > OTSU_threshold * thresh  # will create binary image

    # if there are no cells, good to clear the border
    # because otherwise the OTSU is just for random bullshit, most
    # likely on the side of the image
    threshholded = segmentation.clear_border(threshholded)

    # Opening = erosion then dialation.
    # opening smooths images, breaks isthmuses, and eliminates protrusions.
    # "opens" dark gaps between bright features.
    morph = morphology.binary_opening(threshholded, morphology.disk(first_opening_size))

    # if this image is empty at this point (likely if there were no cells), just return
    # zero array
    if np.amax(morph) == 0:
        return np.zeros_like(image)

    ### Calculate distance matrix, use as markers for random walker (diffusion watershed)
    # Generate the markers based on distance to the background
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        distance = ndi.distance_transform_edt(morph)

    # threshold distance image
    distance_thresh = np.zeros_like(distance)
    distance_thresh[distance < distance_threshold] = 0
    distance_thresh[distance >= distance_threshold] = 1

    # do an extra opening on the distance
    distance_opened = morphology.binary_opening(
        distance_thresh, morphology.disk(second_opening_size)
    )

    # remove artifacts connected to image border
    cleared = segmentation.clear_border(distance_opened)
    # remove small objects. Remove small objects wants a
    # labeled image and will fail if there is only one label. Return zero image in that case
    # could have used try/except but remove_small_objects loves to issue warnings.
    cleared, label_num = morphology.label(cleared, connectivity=1, return_num=True)
    if label_num > 1:
        cleared = morphology.remove_small_objects(cleared, min_size=min_object_size)
    else:
        # if there are no labels, then just return the cleared image as it is zero
        return np.zeros_like(image)

    # relabel now that small objects and labels on edges have been cleared
    markers = morphology.label(cleared, connectivity=1)

    # just break if there is no label
    if np.amax(markers) == 0:
        return np.zeros_like(image)

    # the binary image for the watershed, which uses the unmodified OTSU threshold
    threshholded_watershed = threshholded
    threshholded_watershed = segmentation.clear_border(threshholded_watershed)

    # label using the random walker (diffusion watershed) algorithm
    try:
        # set anything outside of OTSU threshold to -1 so it will not be labeled
        markers[threshholded_watershed == 0] = -1
        # here is the main algorithm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labeled_image = segmentation.random_walker(-1 * image, markers)
        # put negative values back to zero for proper image
        labeled_image[labeled_image == -1] = 0
    except:
        return np.zeros_like(image)

    return labeled_image


def segmentOTSU(params, view_result: bool = False):

    information("Loading experiment parameters.")
    p = params

    user_spec_fovs = p["FOV"]

    # create segmenteation and cell data folder if they don't exist
    if not os.path.exists(p["seg_dir"]) and p["output"] == "TIFF":
        os.makedirs(p["seg_dir"])
    if not os.path.exists(p["cell_dir"]):
        os.makedirs(p["cell_dir"])

    # set segmentation image name for saving and loading segmented images
    p["seg_img"] = "seg_otsu"

    # load specs file
    specs = load_specs(params["ana_dir"])

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    information("Segmenting %d FOVs." % len(fov_id_list))

    ### Do Segmentation by FOV and then peak #######################################################
    information("Segmenting channels using Otsu method.")

    for fov_id in progress(fov_id_list):
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
            segment_chnl_stack(params, fov_id, peak_id, view_result)

    information("Finished segmentation.")


class SegmentOtsu(MM3Container):
    def create_widgets(self):
        """Overriding method. Serves as the widget constructor. See MM3Container for more details."""
        self.viewer.grid.enabled = False
        self.viewer.grid.shape = (-1, 20)

        self.plane_picker_widget = PlanePicker(self.valid_planes, label="phase plane")
        self.otsu_threshold_widget = FloatSpinBox(
            label="OTSU threshold", min=0.0, max=2.0, step=0.01, value=1
        )
        self.first_opening_size_widget = SpinBox(
            label="first opening size", min=0, value=2
        )
        self.distance_threshold_widget = SpinBox(
            label="distance threshold", min=0, value=2
        )
        self.second_opening_size_widget = SpinBox(
            label="second opening size", min=0, value=1
        )
        self.min_object_size_widget = SpinBox(label="min object size", min=0, value=25)
        self.preview_widget = PushButton(label="generate preview")
        self.fov_widget = FOVChooser(self.valid_fovs)
        self.view_result_widget = CheckBox(label="display output", value=True)

        self.plane_picker_widget.changed.connect(self.set_phase_plane)
        self.fov_widget.connect_callback(self.set_fovs)
        self.otsu_threshold_widget.changed.connect(self.set_OTSU_threshold)
        self.first_opening_size_widget.changed.connect(self.set_first_opening_size)
        self.distance_threshold_widget.changed.connect(self.set_distance_threshold)
        self.second_opening_size_widget.changed.connect(self.set_second_opening_size)
        self.min_object_size_widget.changed.connect(self.set_min_object_size)
        self.preview_widget.clicked.connect(self.render_preview)
        self.view_result_widget.changed.connect(self.set_view_result)

        self.append(self.plane_picker_widget)
        self.append(self.otsu_threshold_widget)
        self.append(self.first_opening_size_widget)
        self.append(self.distance_threshold_widget)
        self.append(self.second_opening_size_widget)
        self.append(self.min_object_size_widget)
        self.append(self.preview_widget)
        self.append(self.fov_widget)
        self.append(self.view_result_widget)

        self.set_fovs(self.valid_fovs)
        self.set_phase_plane()
        self.set_OTSU_threshold()
        self.set_first_opening_size()
        self.set_distance_threshold()
        self.set_second_opening_size()
        self.set_min_object_size()

        try:
            self.render_preview()
        except FileNotFoundError:
            warning(f"Failed to render preview from plane {self.phase_plane}")

    def set_params(self):
        self.params = dict()
        self.params["experiment_name"] = self.experiment_name
        self.params["output"] = "TIFF"
        self.params["FOV"] = self.fovs
        self.params["phase_plane"] = self.phase_plane

        self.params["segment"] = dict()
        self.params["segment"]["OTSU_threshold"] = self.OTSU_threshold
        self.params["segment"]["first_opening_size"] = self.first_opening_size
        self.params["segment"]["distance_threshold"] = self.distance_threshold
        self.params["segment"]["second_opening_size"] = self.second_opening_size
        self.params["segment"]["min_object_size"] = self.min_object_size
        self.params["num_analyzers"] = multiprocessing.cpu_count()

        # useful folder shorthands for opening files
        self.params["TIFF_dir"] = self.TIFF_folder
        self.params["ana_dir"] = self.analysis_folder
        self.params["hdf5_dir"] = self.params["ana_dir"] / "hdf5"
        self.params["chnl_dir"] = self.params["ana_dir"] / "channels"
        self.params["empty_dir"] = self.params["ana_dir"] / "empties"
        self.params["sub_dir"] = self.params["ana_dir"] / "subtracted"
        self.params["seg_dir"] = self.params["ana_dir"] / "segmented"
        self.params["cell_dir"] = self.params["ana_dir"] / "cell_data"
        self.params["track_dir"] = self.params["ana_dir"] / "tracking"

    def set_phase_plane(self):
        self.phase_plane = self.plane_picker_widget.value

    def set_fovs(self, fovs):
        self.fovs = fovs

    def set_OTSU_threshold(self):
        self.OTSU_threshold = self.otsu_threshold_widget.value

    def set_first_opening_size(self):
        self.first_opening_size = self.first_opening_size_widget.value

    def set_distance_threshold(self):
        self.distance_threshold = self.distance_threshold_widget.value

    def set_second_opening_size(self):
        self.second_opening_size = self.second_opening_size_widget.value

    def set_min_object_size(self):
        self.min_object_size = self.min_object_size_widget.value

    def set_view_result(self):
        self.view_result = self.view_result_widget.value

    def render_preview(self):
        self.viewer.layers.clear()
        self.set_params()
        # TODO: Add ability to change these to other FOVs
        valid_fov = self.valid_fovs[0]
        specs = load_specs(self.params["ana_dir"])
        # Find first cell-containing peak
        valid_peak = [key for key in specs[valid_fov] if specs[valid_fov][key] == 1][0]
        ## pull out first fov & peak id with cells
        sub_stack = load_stack_params(
            self.params,
            valid_fov,
            valid_peak,
            postfix="sub_{}".format(self.params["phase_plane"]),
        )

        # image by image for debug
        segmented_imgs = []
        for sub_image in sub_stack:
            segmented_imgs.append(segment_image(self.params, sub_image))

        # stack them up along a time axis
        segmented_imgs = np.stack(segmented_imgs, axis=0)
        segmented_imgs = segmented_imgs.astype("uint8")

        images = self.viewer.add_image(sub_stack)
        images.gamma = 1
        labels = self.viewer.add_labels(segmented_imgs, name="Labels")
        labels.opacity = 0.5

    def run(self):
        self.set_params()
        self.viewer.window._status_bar._toggle_activity_dock(True)
        segmentOTSU(self.params, self.view_result_widget.value)
