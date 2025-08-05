from __future__ import division, print_function

import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import napari
import numpy as np
import six
import tensorflow as tf
import tifffile as tiff
from keras import backend as K
from keras import losses, models
from magicgui.widgets import (
    FloatSlider,
    PushButton,
)
from napari import Viewer
from skimage import morphology, segmentation
from tensorflow import keras
from tensorflow.python.ops import array_ops, math_ops

from ._deriving_widgets import (
    FOVList,
    MM3Container2,
    get_valid_fovs_folder,
    get_valid_planes,
    information,
    load_specs,
    load_tiff,
)
from .utils import TIFF_FILE_FORMAT_PEAK


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
    losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


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
    except:  # noqa: E722
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
    except:  # noqa: E722
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
    channels_dir: str,
    seg_dir: Path,
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
        channels_dir,
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
    channels_dir: str,
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
        img_stack = load_tiff(channels_dir / img_stack_filename)

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
    predict_args = {}  # dict(use_multiprocessing=True, workers=num_analyzers, verbose=1)

    if normalize_to_one:
        img_stack = normalize(img_stack)

    img_stack = trim_and_pad(img_stack, unet_shape, pad_dict)
    img_stack = np.expand_dims(img_stack, -1)  # TF expects images to be 4D

    input_data = tf.data.Dataset.from_tensor_slices(img_stack)
    input_data = input_data.batch(batch_size)

    predictions = model.predict(input_data, **predict_args)
    predictions = pad_back(predictions, unet_shape, pad_dict)

    return predictions


@dataclass
class InPaths:
    """
    1. check folders for existence, fetch FOVs & times & planes
        -> upon failure, simply show a list of inputs + update button.
    """

    model_file: Path = Path(__file__).parent / "default_unet_model.hdf5"
    channel_dir: Annotated[Path, {"mode": "d"}] = Path("./analysis/channels")
    specs_file: Path = Path("./analysis/specs.yaml")
    experiment_name: str = ""


@dataclass
class OutPaths:
    seg_dir: Annotated[Path, {"mode": "d"}] = Path("./analysis/segmented")
    cell_dir: Annotated[Path, {"mode": "d"}] = Path("./analysis/cell_data")


@dataclass
class RunParams:
    FOVs: FOVList
    phase_plane: str
    trained_model_image_height: Annotated[int, {"min": 1, "max": 8192}] = 256
    trained_model_image_width: Annotated[int, {"min": 1, "max": 8192}] = 32
    batch_size: Annotated[
        int,
        {"tooltip": "different speeds are faster on different computers.", "min": 1},
    ] = 20
    cell_class_threshold: Annotated[
        float, {"min": 0, "max": 1, "widget_type": FloatSlider}
    ] = 0.5
    min_object_size: Annotated[int, {"min": 0, "max": 100}] = 25
    normalize_to_one: Annotated[
        bool, {"tooltip": "Whether or not to normalize pixel intensities."}
    ] = True
    num_analyzers: int = multiprocessing.cpu_count()
    model_source: Annotated[str, {"choices": ["Pixelwise weighted", "Unweighted"]}] = (
        "Pixelwise weighted"
    )
    view_result: bool = True


def gen_default_run_params(in_files: InPaths):
    try:
        all_fovs = get_valid_fovs_folder(in_files.channel_dir)
        # TODO: get the brightest channel as the default phase plane!
        channels = get_valid_planes(in_files.channel_dir)
        # move this into runparams somehow!
        params = RunParams(FOVs=FOVList(all_fovs), phase_plane=channels[0])
        params.__annotations__["phase_plane"] = Annotated[str, {"choices": channels}]
        return params
    except FileNotFoundError:
        raise FileNotFoundError("TIFF folder not found")
    except ValueError:
        raise ValueError(
            "Invalid filenames. Make sure that timestamps are denoted as t[0-9]* and FOVs as xy[0-9]*"
        )


def segmentUNet(in_paths: InPaths, run_params: RunParams, out_paths: OutPaths):
    if run_params.model_source == "Pixelwise weighted":
        custom_objects = {
            "binary_acc": binary_acc,
            "pixelwise_weighted_bce": pixelwise_weighted_bce,
        }

    elif run_params.model_source == "Unweighted":
        custom_objects = {"bce_dice_loss": bce_dice_loss, "dice_loss": dice_loss}
    information("Loading experiment parameters.")

    # create segmentation and cell data folder if they don't exist
    if not out_paths.seg_dir.exists():
        out_paths.seg_dir.mkdir()
    if not out_paths.cell_dir.exists():
        out_paths.cell_dir.mkdir()

    # set segmentation image name for saving and loading segmented images
    seg_img = "seg_unet"

    # load specs file
    specs = load_specs(in_paths.specs_file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])
    if run_params.FOVs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in run_params.FOVs]

    information("Loading model...")
    seg_model = models.load_model(
        str(in_paths.model_file), custom_objects=custom_objects, compile=False
    )
    information("Model loaded.")

    for fov_id in fov_id_list:
        segment_fov_unet(
            fov_id,
            specs,
            seg_model,
            in_paths.experiment_name,
            run_params.phase_plane,
            in_paths.channel_dir,
            out_paths.seg_dir,
            seg_img,
            run_params.trained_model_image_height,
            run_params.trained_model_image_width,
            run_params.batch_size,
            run_params.cell_class_threshold,
            run_params.min_object_size,
            run_params.num_analyzers,
            run_params.normalize_to_one,
            run_params.view_result,
        )

    del seg_model
    information("Finished segmentation.")


class SegmentUnet(MM3Container2):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self.viewer = viewer
        self.in_paths = InPaths()
        try:
            self.run_params = gen_default_run_params(self.in_paths)
            self.out_paths = OutPaths()
            self.initialized = True
            self.regen_widgets()

            self.preview_widget = PushButton(label="generate preview")
            self.append(self.preview_widget)
            self.preview_widget.changed.connect(self.render_preview)
        except FileNotFoundError | ValueError:
            self.initialized = False
            self.regen_widgets()

    def run(self):
        segmentUNet(self.in_paths, self.run_params, self.out_paths)

    def render_preview(self):
        self.viewer.layers.clear()

        if self.run_params.model_source == "Pixelwise weighted":
            custom_objects = {
                "binary_acc": binary_acc,
                "pixelwise_weighted_bce": pixelwise_weighted_bce,
            }

        elif self.model_source == "Unweighted":
            custom_objects = {"bce_dice_loss": bce_dice_loss, "dice_loss": dice_loss}

        # TODO: Add ability to change these to other FOVs
        valid_fov = self.run_params.FOVs[0]
        specs = load_specs(self.in_paths.specs_file)
        # Find first cell-containing peak
        valid_peak = [key for key in specs[valid_fov] if specs[valid_fov][key] == 1][0]
        ## pull out first fov & peak id with cells
        # load segmentation parameters
        unet_shape = (
            self.run_params.trained_model_image_height,
            self.run_params.trained_model_image_width,
        )
        img_stack_filename = TIFF_FILE_FORMAT_PEAK % (
            self.in_paths.experiment_name,
            valid_fov,
            valid_peak,
            self.run_params.phase_plane,
        )
        img_stack = load_tiff(self.in_paths.channel_dir / img_stack_filename)

        img_height = img_stack.shape[1]
        img_width = img_stack.shape[2]

        pad_dict = get_pad_distances(unet_shape, img_height, img_width)

        model_file_path = self.in_paths.model_file

        # *** Need parameter for weights
        seg_model = models.load_model(
            model_file_path, custom_objects=custom_objects, compile=False
        )

        predictions = segment_peak_unet(
            img_stack,
            unet_shape,
            pad_dict,
            seg_model,
            self.run_params.batch_size,
            self.run_params.num_analyzers,
            self.run_params.normalize_to_one,
        )

        del seg_model

        # binarized and label (if there is a threshold value, otherwise, save a grayscale for debug)
        segmented_imgs = binarize_and_label(
            predictions,
            self.run_params.cell_class_threshold,
            self.run_params.min_object_size,
        )
        # segmented_imgs = predictions.astype("uint8")
        images = self.viewer.add_image(img_stack)
        images.gamma = 1
        labels = self.viewer.add_labels(segmented_imgs, name="Labels")
        labels.opacity = 0.5


if __name__ == "__main__":
    in_paths = InPaths()
    run_params = gen_default_run_params(in_paths)
    out_paths = OutPaths()
    segmentUNet(in_paths, run_params, out_paths)
