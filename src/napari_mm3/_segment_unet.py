from __future__ import print_function, division
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, losses
import h5py
import multiprocessing
import numpy as np
import napari
from magicgui import magicgui
from magicgui.widgets import FileEdit, SpinBox, FloatSlider, CheckBox
from napari.types import ImageData, LabelsData
import os

from skimage import segmentation, morphology
from skimage.filters import median

import six
import tifffile as tiff

from ._deriving_widgets import (
    FOVChooser,
    MM3Container,
    PlanePicker,
    load_specs,
    information,
    load_stack_params,
)

# loss functions for model
def dice_coeff(y_true, y_pred):
    smooth = 1.0
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def get_pad_distances(unet_shape, img_height, img_width):
    """Finds padding and trimming sizes to make the input image the same as the size expected by the U-net model.

    Padding is done evenly to the top and bottom of the image. Trimming is only done from the right or bottom.
    """

    half_width_pad = (unet_shape[1] - img_width) / 2
    if half_width_pad > 0:
        left_pad = int(np.floor(half_width_pad))
        right_pad = int(np.ceil(half_width_pad))
        right_trim = 0
    else:
        left_pad = 0
        right_pad = 0
        right_trim = img_width - unet_shape[1]

    half_height_pad = (unet_shape[0] - img_height) / 2
    if half_height_pad > 0:
        top_pad = int(np.floor(half_height_pad))
        bottom_pad = int(np.ceil(half_height_pad))
        bottom_trim = 0
    else:
        top_pad = 0
        bottom_pad = 0
        bottom_trim = img_height - unet_shape[0]

    pad_dict = {
        "top_pad": top_pad,
        "bottom_pad": bottom_pad,
        "right_pad": right_pad,
        "left_pad": left_pad,
        "bottom_trim": bottom_trim,
        "right_trim": right_trim,
    }

    return pad_dict


def save_out(params, segmented_imgs, fov_id, peak_id):
    # save out the segmented stacks
    if params["output"] == "TIFF":
        seg_filename = params["experiment_name"] + "_xy%03d_p%04d_%s.tif" % (
            fov_id,
            peak_id,
            params["seg_img"],
        )
        tiff.imwrite(
            params["seg_dir"] / seg_filename,
            segmented_imgs,
            compression=("zlib", 4),
        )

    if params["output"] == "HDF5":
        h5f = h5py.File(params["hdf5_dir"] / ("xy%03d.hdf5" % fov_id), "r+")
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

    return


def save_predictions(predictions, params, fov_id, peak_id):
    pred_filename = params["experiment_name"] + "_xy%03d_p%04d_%s.tif" % (
        fov_id,
        peak_id,
        params["pred_img"],
    )
    if not os.path.isdir(params["pred_dir"]):
        os.makedirs(params["pred_dir"])
    int_preds = (predictions * 255).astype("uint8")
    tiff.imwrite(
        params["pred_dir"] / pred_filename,
        int_preds,
        compression=("zlib", 4),
    )


def normalize_to_one(img_stack):
    med_stack = np.zeros(img_stack.shape)
    selem = morphology.disk(1)

    for frame_idx in range(img_stack.shape[0]):
        tmpImg = img_stack[frame_idx, ...]
        med_stack[frame_idx, ...] = median(tmpImg, selem)

    # robust normalization of peak's image stack to 1
    # max_val = np.max(med_stack)
    img_avg = np.mean(img_stack, axis=(1, 2))
    img_std = np.std(img_stack, axis=(1, 2))
    # permute axes to make use of numpy slicing then permute back
    img_stack = np.transpose((np.transpose(img_stack) - img_avg) / img_std)
    return img_stack


## post-processing of u-net output
# binarize, remove small objects, clear border, and label
def binarize_and_label(predictions, cellClassThreshold, min_object_size):

    predictions[predictions >= cellClassThreshold] = 1
    predictions[predictions < cellClassThreshold] = 0
    predictions = predictions.astype("uint8")

    segmented_imgs = np.zeros(predictions.shape, dtype="uint8")
    # process and label each frame of the channel
    for frame in range(segmented_imgs.shape[0]):
        # get rid of small holes
        predictions[frame, :, :] = morphology.remove_small_holes(
            predictions[frame, :, :], min_object_size
        )
        # get rid of small objects.
        predictions[frame, :, :] = morphology.remove_small_objects(
            morphology.label(predictions[frame, :, :], connectivity=1),
            min_size=min_object_size,
        )
        # remove labels which touch the boarder
        predictions[frame, :, :] = segmentation.clear_border(predictions[frame, :, :])
        # relabel now
        segmented_imgs[frame, :, :] = morphology.label(
            predictions[frame, :, :], connectivity=1
        )
    return segmented_imgs


def trim_and_pad(img_stack, unet_shape, pad_dict):
    # trim and pad image to correct size
    img_stack = img_stack[:, : unet_shape[0], : unet_shape[1]]
    img_stack = np.pad(
        img_stack,
        (
            (0, 0),
            (pad_dict["top_pad"], pad_dict["bottom_pad"]),
            (pad_dict["left_pad"], pad_dict["right_pad"]),
        ),
        mode="constant",
    )
    return img_stack


def pad_back(predictions, unet_shape, pad_dict):
    # used in post processing
    # remove padding including the added last dimension
    predictions = predictions[
        :,
        pad_dict["top_pad"] : unet_shape[0] - pad_dict["bottom_pad"],
        pad_dict["left_pad"] : unet_shape[1] - pad_dict["right_pad"],
        0,
    ]

    # pad back in case the image had been trimmed
    predictions = np.pad(
        predictions,
        ((0, 0), (0, pad_dict["bottom_trim"]), (0, pad_dict["right_trim"])),
        mode="constant",
    )
    return predictions
    
def segment_fov_unet(fov_id: int, specs: dict, model, params: dict, color=None):
    """
    Segments the channels from one fov using the U-net CNN model.

    Parameters
    ----------
    fov_id : int
    specs : dict
    model : TensorFlow model
    """

    information("Segmenting FOV {} with U-net.".format(fov_id))

    if color is None:
        color = params["phase_plane"]

    # load segmentation parameters
    unet_shape = (
        params["segment"]["trained_model_image_height"],
        params["segment"]["trained_model_image_width"],
    )

    ### determine stitching of images.
    # need channel shape, specifically the width. load first for example
    # this assumes that all channels are the same size for this FOV, which they should
    for peak_id, spec in six.iteritems(specs[fov_id]):
        if spec == 1:
            break  # just break out with the current peak_id

    img_stack = load_stack_params(params, fov_id, peak_id, postfix=color)
    img_height = img_stack.shape[1]
    img_width = img_stack.shape[2]

    pad_dict = get_pad_distances(unet_shape, img_height, img_width)

    # dermine how many channels we have to analyze for this FOV
    ana_peak_ids = []
    for peak_id, spec in six.iteritems(specs[fov_id]):
        if spec == 1:
            ana_peak_ids.append(peak_id)
    ana_peak_ids.sort()  # sort for repeatability

    segment_cells_unet(ana_peak_ids, fov_id, pad_dict, unet_shape, model, params)

    information("Finished segmentation for FOV {}.".format(fov_id))

    return


def segment_cells_unet(ana_peak_ids, fov_id, pad_dict, unet_shape, model, params):

    @magicgui(auto_call=True, threshold={"widget_type": "FloatSlider", "max": 1})
    def DebugUnet(image_input: ImageData, threshold=0.6) -> LabelsData:
        image_out = np.copy(image_input)
        image_out[image_out >= threshold] = 1
        image_out[image_out < threshold] = 0
        image_out = image_out.astype(bool)

        return image_out
    # parameters
    batch_size = params["segment"]["batch_size"]
    cellClassThreshold = params["segment"]["cell_class_threshold"]
    min_object_size = params["segment"]["min_object_size"]

    # arguments to predict
    predict_args = dict(
        use_multiprocessing=True, workers=params["num_analyzers"], verbose=1
    )

    # linearized version for debugging
    # predict_args = dict(use_multiprocessing=False,
    #                     verbose=1)

    for peak_id in ana_peak_ids:
        information("Segmenting peak {}.".format(peak_id))
        img_stack = load_stack_params(
            params, fov_id, peak_id, postfix=params["phase_plane"]
        )
        if params["segment"]["normalize_to_one"]:
            img_stack = normalize_to_one(img_stack)
        img_stack = trim_and_pad(img_stack, unet_shape, pad_dict)
        img_stack = np.expand_dims(img_stack, -1)  # TF expects images to be 4D
        # set up image generator
        image_datagen = ImageDataGenerator()
        image_generator = image_datagen.flow(
            x=img_stack, batch_size=batch_size, shuffle=False
        )  # keep same order

        # predict cell locations. This has multiprocessing built in but I need to mess with the parameters to see how to best utilize it. ***
        predictions = model.predict_generator(image_generator, **predict_args)
        predictions = pad_back(predictions, unet_shape, pad_dict)
        img_stack_out = pad_back(img_stack, unet_shape, pad_dict)

        if params["segment"]["save_predictions"]:
            save_predictions(predictions, params, fov_id, peak_id)

        if params["interactive"]:
            viewer = napari.current_viewer()
            viewer.layers.clear()
            viewer.add_image(predictions, name="Predictions")
            viewer.window.add_dock_widget(DebugUnet)
            viewer.add_image(img_stack_out)
            try:
                viewer.layers.move(2, 1)
            except:
                pass
            return

        # binarized and label (if there is a threshold value, otherwise, save a grayscale for debug)
        if cellClassThreshold:
            segmented_imgs = binarize_and_label(
                predictions,
                cellClassThreshold=cellClassThreshold,
                min_object_size=min_object_size,
            )

        else:  # in this case you just want to scale the 0 to 1 float image to 0 to 255
            information("Converting predictions to grayscale.")
            segmented_imgs = np.around(predictions * 100)

        # both binary and grayscale should be 8bit. This may be ensured above and is unneccesary
        segmented_imgs = segmented_imgs.astype("uint8")

        # save the segmented images
        save_out(params, segmented_imgs, fov_id, peak_id)


def segmentUNet(params):

    information("Loading experiment parameters.")
    p = params

    user_spec_fovs = params["FOV"]

    information("Using {} threads for multiprocessing.".format(p["num_analyzers"]))

    # create segmenteation and cell data folder if they don't exist
    if not os.path.exists(p["seg_dir"]):
        os.makedirs(p["seg_dir"])
    if not os.path.exists(p["cell_dir"]):
        os.makedirs(p["cell_dir"])

    # set segmentation image name for saving and loading segmented images
    p["seg_img"] = "seg_unet"
    p["pred_img"] = "pred_unet"

    # load specs file
    specs = load_specs(params["ana_dir"])
    # print(specs) # for debugging

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    information("Processing %d FOVs." % len(fov_id_list))

    ### Do Segmentation by FOV and then peak #######################################################
    information("Segmenting channels using U-net.")

    # load model to pass to algorithm
    information("Loading model...")

    model_file_path = p["segment"]["model_file"]

    # *** Need parameter for weights
    seg_model = models.load_model(
        model_file_path,
        custom_objects={"bce_dice_loss": bce_dice_loss, "dice_loss": dice_loss},
    )
    information("Model loaded.")

    for fov_id in fov_id_list:
        segment_fov_unet(fov_id, specs, seg_model, params, color=p["phase_plane"])

    del seg_model
    information("Finished segmentation.")


class SegmentUnet(MM3Container):
    def create_widgets(self):
        """Overriding method. Serves as the widget constructor. See MM3Container for more details."""
        self.fov_widget = FOVChooser(self.valid_fovs)
        self.plane_widget = PlanePicker(
            self.valid_planes,
            label="phase plane",
            tooltip="pick the phase plane. first channel is c1, second is c2, etc.",
        )
        self.model_file_widget = FileEdit(
            mode="r",
            label="model file",
            tooltip="required. denotes where the model file is stored.",
        )
        self.min_object_size_widget = SpinBox(
            label="min object size",
            tooltip="the minimum size for an object to be recognized",
            value=25,
            min=0,
            max=100,
        )
        self.batch_size_widget = SpinBox(
            label="batch size",
            tooltip="how large to make the batches. different speeds are faster on different computers.",
            value=20,
            min=1,
            max=9999,
        )
        self.cell_class_threshold_widget = FloatSlider(
            label="cell class threshold", min=0, max=1.0, value=0.6
        )
        self.normalize_widget = CheckBox(label="normalize to one", value=True)
        self.height_widget = SpinBox(label="image height", min=1, max=5000, value=256)
        self.width_widget = SpinBox(label="image width", min=1, max=5000, value=32)
        self.interactive_widget = CheckBox(label="interactive", value=False)

        self.append(self.fov_widget)
        self.append(self.plane_widget)
        self.append(self.model_file_widget)
        self.append(self.min_object_size_widget)
        self.append(self.batch_size_widget)
        self.append(self.cell_class_threshold_widget)
        self.append(self.normalize_widget)
        self.append(self.height_widget)
        self.append(self.width_widget)
        self.append(self.interactive_widget)

        self.fov_widget.connect_callback(self.set_fovs)

    def run(self):
        """Overriding method. Perform mother machine analysis."""
        params = dict()
        params["experiment_name"] = self.experiment_name
        params["image_directory"] = self.TIFF_folder
        params["FOV"] = self.fovs
        params["output"] = "TIFF"
        params["interactive"] = self.interactive_widget.value
        params["phase_plane"] = self.plane_widget.value
        params["subtract"] = dict()
        params["segment"] = dict()
        params["segment"]["model_file"] = self.model_file_widget.value
        params["segment"]["trained_model_image_height"] = self.height_widget.value
        params["segment"]["trained_model_image_width"] = self.width_widget.value
        params["segment"]["batch_size"] = self.batch_size_widget.value
        params["segment"][
            "cell_class_threshold"
        ] = self.cell_class_threshold_widget.value
        params["segment"]["save_predictions"] = False
        params["segment"]["min_object_size"] = self.min_object_size_widget.value
        params["segment"]["normalize_to_one"] = self.normalize_widget.value
        params["num_analyzers"] = multiprocessing.cpu_count()

        # useful folder shorthands for opening files
        params["TIFF_dir"] = self.TIFF_folder
        params["ana_dir"] = self.analysis_folder

        params["hdf5_dir"] = params["ana_dir"] / "hdf5"
        params["chnl_dir"] = params["ana_dir"] / "channels"
        params["empty_dir"] = params["ana_dir"] / "empties"
        params["sub_dir"] = params["ana_dir"] / "subtracted"
        params["seg_dir"] = params["ana_dir"] / "segmented"
        params["pred_dir"] = params["ana_dir"] / "predictions"
        params["foci_seg_dir"] = params["ana_dir"] / "segmented_foci"
        params["foci_pred_dir"] = params["ana_dir"] / "predictions_foci"
        params["cell_dir"] = params["ana_dir"] / "cell_data"
        params["track_dir"] = params["ana_dir"] / "tracking"
        params["foci_track_dir"] = params["ana_dir"] / "tracking_foci"

        segmentUNet(params)

    def set_fovs(self, fovs):
        self.fovs = fovs
