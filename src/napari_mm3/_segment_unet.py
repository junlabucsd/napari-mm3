from __future__ import print_function, division
import os
import multiprocessing
import six
import argparse
import numpy as np
from skimage import segmentation, morphology
from pathlib import Path

import tifffile as tiff
import h5py
import tensorflow as tf
from tensorflow import keras
from keras import models, losses
from tensorflow.python.ops import array_ops, math_ops
from keras import backend as K
import napari
from magicgui.widgets import (
    FileEdit,
    SpinBox,
    FloatSlider,
    CheckBox,
    ComboBox,
    PushButton,
)

from .utils import TIFF_FILE_FORMAT_PEAK

from ._deriving_widgets import (
    FOVChooser,
    MM3Container,
    PlanePicker,
    range_string_to_indices,
    get_valid_fovs_folder,
    get_valid_times,
    load_specs,
    information,
    load_tiff,
)


# loss functions for model
def dice_coeff(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Dice coefficient for segmentation accuracy.
    Parameters
    ----------
    y_true : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    y_pred : Tensor
        Predicted segmentation masks.
    Returns
    -------
    Tensor
        Dice coefficient between inputs.
    """
    smooth = 1.0
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
    return score


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Dice loss
    Parameters
    ----------
    y_true: Tensor
        ground truth labels
    y_pred: Tensor
        predicted labels

    Returns
    -------
    loss: Tensor
        dice loss between inputs
    """
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Combined cross entropy and dice loss
    Parameters
    ----------
    y_true: Tensor
        ground truth labels
    y_pred: Tensor
        predicted labels"""
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
    loss: Tensor
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


def save_out(
    experiment_name: str,
    seg_img: str,
    seg_dir: str,
    segmented_imgs: np.ndarray,
    fov_id: int,
    peak_id: int,
) -> None:
    """Saves out the segmented images."""
    # save out the segmented stacks
    seg_filename = experiment_name + "_xy%03d_p%04d_%s.tif" % (
        fov_id,
        peak_id,
        seg_img,
    )
    tiff.imwrite(
        seg_dir / seg_filename,
        segmented_imgs,
        compression=("zlib", 4),
    )
    return


def normalize(img_stack: np.ndarray) -> np.ndarray:
    """
    Normalizes the image stack to [0, 1] by subtracting the minimum and dividing by the range.
    """
    # permute to take advantage of numpy slicing
    img_stack = np.transpose(
        (np.transpose(img_stack) - np.min(img_stack, axis=(1, 2)))
        / np.ptp(img_stack, axis=(1, 2))
    )

    return img_stack


## post-processing of u-net output
# binarize, remove small objects, clear border, and label
def binarize_and_label(predictions, cellClassThreshold, min_object_size) -> np.ndarray:
    """Binarizes the predictions, removes small objects, clears the border, and labels the objects."""

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
    """trim and pad image to correct size
    Parameters
    ----------
    img_stack : np.ndarray
        Image stack.
    unet_shape : tuple
        Shape of the U-net.
    pad_dict : dict
        Dictionary of padding values.

    Returns
    -------
    np.ndarray
        Trimmed and padded image stack.
    """
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
    fov_id: int,
    specs: dict,
    model,
    experiment_name: str,
    phase_plane: str,
    ana_dir: str,
    seg_dir: str,
    seg_img,
    trained_model_image_height,
    trained_model_image_width,
    batch_size,
    cell_class_threshold,
    min_object_size,
    num_analyzers,
    normalize_to_one,
    view_result: bool = False,
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

    color = phase_plane

    # load segmentation parameters
    unet_shape = (trained_model_image_height, trained_model_image_width)

    ### determine stitching of images.
    # need channel shape, specifically the width. load first for example
    # this assumes that all channels are the same size for this FOV, which they should
    for peak_id, spec in six.iteritems(specs[fov_id]):
        if spec == 1:
            break  # just break out with the current peak_id

    # dermine how many channels we have to analyze for this FOV
    ana_peak_ids = []
    for peak_id, spec in six.iteritems(specs[fov_id]):
        if spec == 1:
            ana_peak_ids.append(peak_id)
    ana_peak_ids.sort()  # sort for repeatability

    segment_cells_unet(
        ana_peak_ids,
        fov_id,
        unet_shape,
        model,
        experiment_name,
        phase_plane,
        ana_dir,
        seg_dir,
        seg_img,
        batch_size,
        cell_class_threshold,
        min_object_size,
        num_analyzers,
        normalize_to_one,
        view_result,
    )

    information("Finished segmentation for FOV {}.".format(fov_id))

    return


def segment_cells_unet(
    ana_peak_ids: list,
    fov_id: int,
    unet_shape: tuple,
    model: keras.Model,
    experiment_name: str,
    phase_plane: str,
    ana_dir: str,
    seg_dir: str,
    seg_img: str,
    batch_size: int,
    cell_class_threshold: float,
    min_object_size: int,
    num_analyzers: int,
    normalize_to_one: bool,
    display_result: bool = False,
) -> None:
    """
    Segments cells using U-net model, filtering by a threshold a mininimum object size.

    """
    for peak_id in ana_peak_ids:
        information("Segmenting peak {}.".format(peak_id))

        img_stack_filename = TIFF_FILE_FORMAT_PEAK % (
            experiment_name,
            fov_id,
            peak_id,
            phase_plane,
        )
        img_stack = load_tiff(ana_dir / "channels" / img_stack_filename)

        img_height = img_stack.shape[1]
        img_width = img_stack.shape[2]

        pad_dict = get_pad_distances(unet_shape, img_height, img_width)

        # do the segmentation
        predictions = segment_peak_unet(
            img_stack,
            unet_shape,
            pad_dict,
            model,
            batch_size,
            num_analyzers,
            normalize_to_one,
        )

        # binarized and label
        segmented_imgs = binarize_and_label(
            predictions,
            cellClassThreshold=cell_class_threshold,
            min_object_size=min_object_size,
        )

        if display_result:
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
        save_out(experiment_name, seg_img, seg_dir, segmented_imgs, fov_id, peak_id)


def segment_peak_unet(
    img_stack: np.ndarray,
    unet_shape: tuple,
    pad_dict: dict,
    model: keras.Model,
    batch_size: int,
    num_analyzers: int,
    normalize_to_one: bool,
) -> np.ndarray:
    """
    Segments a peak using U-net model.
    """
    # arguments to predict
    predict_args = dict(use_multiprocessing=True, workers=num_analyzers, verbose=1)

    if normalize_to_one:
        img_stack = normalize(img_stack)

    img_stack = trim_and_pad(img_stack, unet_shape, pad_dict)
    img_stack = np.expand_dims(img_stack, -1)  # TF expects images to be 4D

    input_data = tf.data.Dataset.from_tensor_slices(img_stack)
    input_data = input_data.batch(batch_size)

    predictions = model.predict(input_data, **predict_args)
    predictions = pad_back(predictions, unet_shape, pad_dict)

    return predictions


def segmentUNet(
    experiment_name: str,
    image_directory: str,
    fovs: list,
    phase_plane: str,
    model_file: str,
    trained_model_image_height: int,
    trained_model_image_width: int,
    batch_size: int,
    cell_class_threshold: float,
    min_object_size: int,
    normalize_to_one: bool,
    num_analyzers: int,
    ana_dir: str,
    seg_dir: str,
    cell_dir: str,
    custom_objects: dict,
    view_result: bool,
) -> None:
    information("Loading experiment parameters.")

    information("Using {} threads for multiprocessing.".format(num_analyzers))

    # create segmentation and cell data folder if they don't exist
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
    if not os.path.exists(cell_dir):
        os.makedirs(cell_dir)

    # set segmentation image name for saving and loading segmented images
    seg_img = "seg_unet"
    pred_img = "pred_unet"

    # load specs file
    specs = load_specs(ana_dir)
    # print(specs) # for debugging

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in fovs]

    information("Processing %d FOVs." % len(fov_id_list))

    ### Do Segmentation by FOV and then peak ######
    information("Segmenting channels using U-net.")

    # load model to pass to algorithm
    information("Loading model...")

    # *** Need parameter for weights
    seg_model = models.load_model(model_file, custom_objects=custom_objects)
    information("Model loaded.")

    for fov_id in fov_id_list:
        segment_fov_unet(
            fov_id,
            specs,
            seg_model,
            experiment_name,
            phase_plane,
            ana_dir,
            seg_dir,
            seg_img,
            trained_model_image_height,
            trained_model_image_width,
            batch_size,
            cell_class_threshold,
            min_object_size,
            num_analyzers,
            normalize_to_one,
            view_result,
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
        self.model_file_widget.changed.connect(self.set_model_file)
        self.min_object_size_widget.changed.connect(self.set_min_object_size)
        self.batch_size_widget.changed.connect(self.set_batch_size)
        self.cell_class_threshold_widget.changed.connect(self.set_cell_class_threshold)
        self.normalize_widget.changed.connect(self.set_normalize_to_one)
        self.height_widget.changed.connect(self.set_trained_model_image_height)
        self.width_widget.changed.connect(self.set_trained_model_image_width)

        self.set_fovs(self.valid_fovs)
        self.set_view_result()
        self.set_model_source()
        self.set_phase_plane()
        self.set_model_file()
        self.set_trained_model_image_height()
        self.set_trained_model_image_width()
        self.set_batch_size()
        self.set_cell_class_threshold()
        self.set_min_object_size()
        self.set_normalize_to_one()
        self.set_num_analyzers()

    def run(self):
        """Overriding method. Perform mother machine analysis."""
        self.viewer.layers.clear()

        if self.model_source == "Pixelwise weighted":
            custom_objects = {
                "binary_acc": binary_acc,
                "pixelwise_weighted_bce": pixelwise_weighted_bce,
            }

        elif self.model_source == "Unweighted":
            custom_objects = {"bce_dice_loss": bce_dice_loss, "dice_loss": dice_loss}

        segmentUNet(
            experiment_name=self.experiment_name,
            image_directory=self.TIFF_folder,
            fovs=self.fovs,
            phase_plane=self.phase_plane,
            model_file=self.model_file,
            trained_model_image_height=self.trained_model_image_height,
            trained_model_image_width=self.trained_model_image_width,
            batch_size=self.batch_size,
            cell_class_threshold=self.cell_class_threshold,
            min_object_size=self.min_object_size,
            normalize_to_one=self.normalize_to_one,
            num_analyzers=self.num_analyzers,
            ana_dir=self.analysis_folder,
            seg_dir=self.analysis_folder / "segmented",
            cell_dir=self.analysis_folder / "cell_data",
            custom_objects=custom_objects,
            view_result=self.view_result,
        )

    def set_fovs(self, fovs):
        self.fovs = fovs

    def set_view_result(self):
        self.view_result = self.display_widget.value

    def set_model_source(self):
        self.model_source = self.model_source_widget.value

    def set_phase_plane(self):
        self.phase_plane = self.plane_widget.value

    def set_model_file(self):
        self.model_file = self.model_file_widget.value

    def set_trained_model_image_height(self):
        self.trained_model_image_height = self.height_widget.value

    def set_trained_model_image_width(self):
        self.trained_model_image_width = self.width_widget.value

    def set_batch_size(self):
        self.batch_size = self.batch_size_widget.value

    def set_cell_class_threshold(self):
        self.cell_class_threshold = self.cell_class_threshold_widget.value

    def set_min_object_size(self):
        self.min_object_size = self.min_object_size_widget.value

    def set_normalize_to_one(self):
        self.normalize_to_one = self.normalize_widget.value

    def set_num_analyzers(self):
        self.num_analyzers = multiprocessing.cpu_count()

    def render_preview(self):
        self.viewer.layers.clear()
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
        specs = load_specs(self.analysis_folder)
        # Find first cell-containing peak
        valid_peak = [key for key in specs[valid_fov] if specs[valid_fov][key] == 1][0]
        ## pull out first fov & peak id with cells
        # load segmentation parameters
        unet_shape = (
            self.trained_model_image_height,
            self.trained_model_image_width,
        )
        img_stack_filename = TIFF_FILE_FORMAT_PEAK % (
            self.experiment_name,
            valid_fov,
            valid_peak,
            self.phase_plane,
        )
        img_stack = load_tiff(self.analysis_folder / "channels" / img_stack_filename)

        img_height = img_stack.shape[1]
        img_width = img_stack.shape[2]

        pad_dict = get_pad_distances(unet_shape, img_height, img_width)

        model_file_path = self.model_file

        # *** Need parameter for weights
        seg_model = models.load_model(
            model_file_path,
            custom_objects=custom_objects,
        )

        predictions = segment_peak_unet(
            img_stack,
            unet_shape,
            pad_dict,
            seg_model,
            self.batch_size,
            multiprocessing.cpu_count(),
            self.normalize_to_one,
        )

        del seg_model

        # binarized and label (if there is a threshold value, otherwise, save a grayscale for debug)
        segmented_imgs = binarize_and_label(
            predictions,
            self.cellClassThreshold,
            self.min_object_size,
        )
        # segmented_imgs = predictions.astype("uint8")
        images = self.viewer.add_image(img_stack)
        images.gamma = 1
        labels = self.viewer.add_labels(segmented_imgs, name="Labels")
        labels.opacity = 0.5


if __name__ == "__main__":
    cur_dir = Path(".")
    end_time = get_valid_times(cur_dir / "TIFF")
    all_fovs = get_valid_fovs_folder(cur_dir / "TIFF")

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
    parser.add_argument(
        "--model_path",
        help="path to the model",
        type=int,
        required=True,
    )
    parser.add_argument("--fovs", help="Which FOVs to include?", default="", type=str)
    p = parser.parse_args()

    if p.fovs == "":
        fovs = all_fovs
    else:
        fovs = range_string_to_indices(p.fovs)
        for fov in fovs:
            if fov not in all_fovs:
                raise ValueError("Some FOVs are out of range for your nd2 file.")

    if (p.start_time < 0) or (p.end_time > end_time) or (p.start_time > p.end_time):
        raise ValueError("Times out of range")
    
    model_path = Path(p.model_path)

    model_source = "Pixelwise weighted"
    if model_source == "Pixelwise weighted":
        custom_objects = {
            "binary_acc": binary_acc,
            "pixelwise_weighted_bce": pixelwise_weighted_bce,
        }
    elif model_source == "Unweighted":
        custom_objects = {"bce_dice_loss": bce_dice_loss, "dice_loss": dice_loss}

    segmentUNet(
        experiment_name="",
        image_directory=cur_dir / "TIFF",
        fovs=fovs,
        phase_plane="c1",
        model_file=model_path,
        trained_model_image_height=256,
        trained_model_image_width=32,
        batch_size=20,
        cell_class_threshold=0.5,
        min_object_size=25,
        normalize_to_one=True,
        num_analyzers=multiprocessing.cpu_count(),
        ana_dir=cur_dir / "analysis",
        seg_dir=cur_dir / "analysis" / "segmented",
        cell_dir=cur_dir / "analysis" "cell_data",
        custom_objects=custom_objects,
    )
