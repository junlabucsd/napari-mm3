from __future__ import print_function, division
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import models, losses
from tensorflow.python.ops import array_ops, math_ops
from keras import backend as K

import h5py
import multiprocessing
import numpy as np
import napari
from magicgui import magicgui
from magicgui.widgets import (
    FileEdit,
    SpinBox,
    FloatSlider,
    CheckBox,
    ComboBox,
    PushButton,
)
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
    warning,
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


def pixelwise_weighted_bce(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Pixel-wise weighted binary cross-entropy loss.
    The code is adapted from the Keras TF backend.
    (see their github)

    Parameters
    ----------
    y_true : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    y_pred : Tensor
        Predicted segmentation masks.

    Returns
    -------
    Tensor
        Pixel-wise weight binary cross-entropy between inputs.

    """
    try:
        # The weights are passed as part of the y_true tensor:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass

    # Make background weights be equal to the model's prediction
    bool_bkgd = weight == 0 / 255
    weight = tf.where(bool_bkgd, y_pred, weight)

    epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    cond = y_pred >= zeros
    relu_logits = math_ops.select(cond, y_pred, zeros)
    neg_abs_logits = math_ops.select(cond, -y_pred, y_pred)
    entropy = math_ops.add(
        relu_logits - y_pred * seg,
        math_ops.log1p(math_ops.exp(neg_abs_logits)),
        name=None,
    )

    loss = K.mean(math_ops.multiply(weight, entropy), axis=-1)

    loss = tf.scalar_mul(
        10**6, tf.scalar_mul(1 / tf.math.sqrt(tf.math.reduce_sum(weight)), loss)
    )

    return loss


def binary_acc(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Unstacks the mask from the weights in the output tensor for
    segmentation and computes binary accuracy

    Parameters
    ----------
    y_true : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    y_pred : Tensor
        Predicted segmentation masks.

    Returns
    -------
    Tensor
        Binary prediction accuracy.

    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass

    return keras.metrics.binary_accuracy(seg, y_pred)


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


def normalize_to_one(img_stack):
    # # robust normalization of peak's image stack to 1

    # permute to take advantage of numpy slicing
    img_stack = np.transpose(
        (np.transpose(img_stack) - np.min(img_stack, axis=(1, 2)))
        / np.ptp(img_stack, axis=(1, 2))
    )

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


def segment_fov_unet(
    fov_id: int, specs: dict, model, params: dict, color=None, view_result: bool = False
):
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

    segment_cells_unet(
        ana_peak_ids, fov_id, pad_dict, unet_shape, model, params, view_result
    )

    information("Finished segmentation for FOV {}.".format(fov_id))

    return


def segment_cells_unet(
    ana_peak_ids, fov_id, pad_dict, unet_shape, model, params, view_result: bool = False
):
    """
    Segments cells using U-net model.

    Args:
        ana_peak_ids (list): List of peak IDs to segment.
        fov_id (str): FOV ID.
        pad_dict (dict): Dictionary containing padding information.
        unet_shape (tuple): Shape of the U-net model.
        model: U-net model.
        params (dict): Parameters for cell segmentation.
        view_result (bool, optional): Whether to display the segmentation results. Defaults to False.
    """
    # params
    cellClassThreshold = params["segment"]["cell_class_threshold"]
    min_object_size = params["segment"]["min_object_size"]

    for peak_id in ana_peak_ids:
        information("Segmenting peak {}.".format(peak_id))
        img_stack = load_stack_params(
            params, fov_id, peak_id, postfix=params["phase_plane"]
        )

        # do the segmentation
        predictions = segment_peak_unet(img_stack, unet_shape, pad_dict, model, params)

        # binarized and label
        segmented_imgs = binarize_and_label(
            predictions,
            cellClassThreshold=cellClassThreshold,
            min_object_size=min_object_size,
        )

        if view_result:
            viewer = napari.current_viewer()
            viewer.grid.enabled = True
            viewer.grid.shape = (-1, 20)

            viewer.add_labels(
                segmented_imgs.astype("int"),
                name=peak_id,
                visible=True,
            )

        # should be 8bit
        segmented_imgs = segmented_imgs.astype("uint8")

        # save the segmented images
        save_out(params, segmented_imgs, fov_id, peak_id)


def segment_peak_unet(img_stack, unet_shape, pad_dict, model, params):
    """
    Segments a peak using U-net model.

    Args:
        img_stack: Image stack for the peak.
        unet_shape (tuple): Shape of the U-net model.
        pad_dict (dict): Dictionary containing padding information.
        model: U-net model.
        params (dict): Parameters for cell segmentation.

    Returns:
        numpy.ndarray: Predictions of the U-net model for the peak.
    """
    # arguments to predict
    predict_args = dict(
        use_multiprocessing=True, workers=params["num_analyzers"], verbose=1
    )

    # linearized version for debugging
    # predict_args = dict(use_multiprocessing=False,
    #                     verbose=1)

    batch_size = params["segment"]["batch_size"]

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
    predictions = model.predict(image_generator, **predict_args)
    predictions = pad_back(predictions, unet_shape, pad_dict)

    return predictions


def segmentUNet(params, custom_objects, view_result):
    """
    Segments cells using U-net model.

    Args:
        params (dict): Experiment parameters.
        custom_objects (dict): Custom objects for model loading.
        view_result (bool): Whether to display the segmentation results.
    """
    information("Loading experiment parameters.")
    p = params

    user_spec_fovs = params["FOV"]

    information("Using {} threads for multiprocessing.".format(p["num_analyzers"]))

    # create segmentation and cell data folder if they don't exist
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
        custom_objects=custom_objects,
    )
    information("Model loaded.")

    for fov_id in fov_id_list:
        segment_fov_unet(
            fov_id,
            specs,
            seg_model,
            params,
            color=p["phase_plane"],
            view_result=view_result,
        )

    del seg_model
    information("Finished segmentation.")


class SegmentUnet(MM3Container):
    def create_widgets(self):
        """Overriding method. Serves as the widget constructor. See MM3Container for more details."""
        self.viewer.text_overlay.visible = False

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
            label="cell class threshold", min=0, max=1.0, value=0.5
        )
        self.normalize_widget = CheckBox(label="Rescale pixel intensity", value=True)
        self.height_widget = SpinBox(label="image height", min=1, max=5000, value=256)
        self.width_widget = SpinBox(label="image width", min=1, max=5000, value=32)
        self.model_source_widget = ComboBox(
            label="Model source", choices=["Pixelwise weighted", "Unweighted"]
        )
        self.preview_widget = PushButton(label="generate preview", value=False)
        self.display_widget = CheckBox(label="Display results", value=True)

        self.append(self.fov_widget)
        self.append(self.plane_widget)
        self.append(self.model_file_widget)
        self.append(self.min_object_size_widget)
        self.append(self.batch_size_widget)
        self.append(self.cell_class_threshold_widget)
        self.append(self.normalize_widget)
        self.append(self.height_widget)
        self.append(self.width_widget)
        self.append(self.model_source_widget)
        self.append(self.preview_widget)
        self.append(self.display_widget)

        self.fov_widget.connect_callback(self.set_fovs)
        self.preview_widget.clicked.connect(self.render_preview)
        self.display_widget.clicked.connect(self.set_view_result)
        self.model_source_widget.changed.connect(self.set_model_source)

        self.set_fovs(self.valid_fovs)
        self.set_view_result()
        self.set_model_source()

    def set_params(self):
        self.params = dict()
        self.params["experiment_name"] = self.experiment_name
        self.params["image_directory"] = self.TIFF_folder
        self.params["FOV"] = self.fovs
        self.params["output"] = "TIFF"
        self.params["phase_plane"] = self.plane_widget.value
        self.params["subtract"] = dict()
        self.params["segment"] = dict()
        self.params["segment"]["model_file"] = self.model_file_widget.value
        self.params["segment"]["trained_model_image_height"] = self.height_widget.value
        self.params["segment"]["trained_model_image_width"] = self.width_widget.value
        self.params["segment"]["batch_size"] = self.batch_size_widget.value
        self.params["segment"][
            "cell_class_threshold"
        ] = self.cell_class_threshold_widget.value
        self.params["segment"]["min_object_size"] = self.min_object_size_widget.value
        self.params["segment"]["normalize_to_one"] = self.normalize_widget.value
        self.params["num_analyzers"] = multiprocessing.cpu_count()

        # useful folder shorthands for opening files
        self.params["TIFF_dir"] = self.TIFF_folder
        self.params["ana_dir"] = self.analysis_folder

        self.params["hdf5_dir"] = self.params["ana_dir"] / "hdf5"
        self.params["chnl_dir"] = self.params["ana_dir"] / "channels"
        self.params["empty_dir"] = self.params["ana_dir"] / "empties"
        self.params["sub_dir"] = self.params["ana_dir"] / "subtracted"
        self.params["seg_dir"] = self.params["ana_dir"] / "segmented"
        self.params["pred_dir"] = self.params["ana_dir"] / "predictions"
        self.params["cell_dir"] = self.params["ana_dir"] / "cell_data"
        self.params["track_dir"] = self.params["ana_dir"] / "tracking"

    def run(self):
        """Overriding method. Perform mother machine analysis."""
        self.set_params()
        self.viewer.layers.clear()

        if self.model_source == "Pixelwise weighted":
            custom_objects = {
                "binary_acc": binary_acc,
                "pixelwise_weighted_bce": pixelwise_weighted_bce,
            }

        elif self.model_source == "Unweighted":
            custom_objects = {"bce_dice_loss": bce_dice_loss, "dice_loss": dice_loss}

        segmentUNet(self.params, custom_objects, view_result=self.view_result)

    def set_fovs(self, fovs):
        self.fovs = fovs

    def set_view_result(self):
        self.view_result = self.display_widget.value

    def set_model_source(self):
        self.model_source = self.model_source_widget.value

    def render_preview(self):
        self.viewer.layers.clear()
        self.set_params()

        self.model_source = self.model_source_widget.value

        if self.model_source == "Pixelwise weighted":
            custom_objects = {
                "binary_acc": binary_acc,
                "pixelwise_weighted_bce": pixelwise_weighted_bce,
            }

        elif self.model_source == "Unweighted":
            custom_objects = {"bce_dice_loss": bce_dice_loss, "dice_loss": dice_loss}

        # TODO: Add ability to change these to other FOVs
        valid_fov = self.valid_fovs[0]
        specs = load_specs(self.params["ana_dir"])
        # Find first cell-containing peak
        valid_peak = [key for key in specs[valid_fov] if specs[valid_fov][key] == 1][0]
        ## pull out first fov & peak id with cells
        # load segmentation parameters
        unet_shape = (
            self.params["segment"]["trained_model_image_height"],
            self.params["segment"]["trained_model_image_width"],
        )

        img_stack = load_stack_params(
            self.params, valid_fov, valid_peak, postfix=self.params["phase_plane"]
        )
        img_height = img_stack.shape[1]
        img_width = img_stack.shape[2]

        pad_dict = get_pad_distances(unet_shape, img_height, img_width)

        model_file_path = self.params["segment"]["model_file"]

        # *** Need parameter for weights
        seg_model = models.load_model(
            model_file_path,
            custom_objects=custom_objects,
        )

        predictions = segment_peak_unet(
            img_stack, unet_shape, pad_dict, seg_model, self.params
        )

        del seg_model

        min_object_size = self.params["segment"]["min_object_size"]
        cellClassThreshold = self.params["segment"]["cell_class_threshold"]

        # binarized and label (if there is a threshold value, otherwise, save a grayscale for debug)
        segmented_imgs = binarize_and_label(
            predictions,
            cellClassThreshold,
            min_object_size,
        )
        # segmented_imgs = predictions.astype("uint8")
        images = self.viewer.add_image(img_stack)
        images.gamma = 1
        labels = self.viewer.add_labels(segmented_imgs, name="Labels")
        labels.opacity = 0.5
