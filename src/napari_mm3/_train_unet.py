import os
import glob
import time
import numpy as np
from re import search, match
from pathlib import Path
import itertools

# image modules
# import png # can pip this package to save pngs at any bitsize
from scipy import ndimage as ndi  # use for binary_fill_holes
from scipy import interpolate
import skimage.transform as trans
from skimage import morphology as morph  # use for remove small objects
from skimage.filters import gaussian
from skimage import io
import tifffile as tiff
from pprint import pprint
import elasticdeform
import matplotlib.pyplot as plt

import multiprocessing

import napari
from napari import Viewer
from magicgui import magicgui
from magicgui.widgets import FileEdit, SpinBox, FloatSpinBox

# learning modules
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import models, losses
from tensorflow.python.ops import array_ops, math_ops
from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Dropout,
    UpSampling2D,
    Concatenate,
)
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from magicgui.widgets import (
    Container,
    FileEdit,
    CheckBox,
    PushButton,
    SpinBox,
    FloatSlider,
)

from typing import (
    cast,
    Tuple,
    List,
    Dict,
    Union,
    Callable,
    Iterator,
    Generator,
    Any,
    Optional,
)
import numpy.typing as npt
import random
import copy

from ._deriving_widgets import MM3Container, information

# Portions of this script are adapted from https://gitlab.com/dunloplab/delta under the MIT license:
#
# ---------------------------------------------------------------------------------------------------
# MIT License

# Copyright (c) 2019 jblugagne

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------

Image = npt.NDArray[np.float32]


def data_augmentation(
    images_input: List[npt.NDArray[Any]],
    aug_par: Dict[str, Any],
    order: Union[int, List[int], Tuple[int, ...]] = 0,
    time: Union[int, List[int]] = 0,
) -> List[npt.NDArray[Any]]:
    """
    ADAPTED FROM DELTA
    https://gitlab.com/dunloplab/delta

    Data augmentation function

    Parameters
    ----------
    images_input : list of 2D numpy arrays
        Images to apply augmentation operations to.
    aug_par : dict
        Augmentation operations parameters. Accepted key-value pairs:
            illumination_voodoo: bool.
                Whether to apply the illumination voodoo operation.
            histogram_voodoo: bool.
                Whether to apply the histogram voodoo operation.
            elastic_deformation: dict.
                If key exists, the elastic deformation operation is applied.
                The parameters are given as key-value pairs. sigma values are
                given under the sigma key, deformation points are given under
                the points key. See elasticdeform doc.
            gaussian_noise: float.
                Apply gaussian noise to the image. The sigma value of the
                gaussian noise is uniformly sampled between 0 and +gaussian_noise.
            gaussian_blur: float.
                Apply gaussian blur to the image. The sigma value is the
                standard deviation of the kernel in the x and y direction.
            horizontal_flip: bool.
                Whether to flip the images horizontally. Input images have a
                50% chance of being flipped
            vertical_flip: bool.
                Whether to flip the images vertically. Input images have a 50%
                chance of being flipped
            rotations90d: bool.
                Whether to randomly rotate the images in 90° increments.
                Each 90° rotation has a 25% chance of happening
            rotation: int or float.
                Range of random rotation to apply. The angle is uniformly
                sampled in the range [-rotation, +rotation]
            zoom: float.
                Range of random "zoom" to apply. The image is randomly zoomed
                by a factor that is sampled from an exponential distribution
                with a lamba of 3/zoom. The random factor is clipped at +zoom.
            shiftX: int/float.
                The range of random shifts to apply along X. A uniformly
                sampled shift between [-shiftX, +shiftX] is applied
            shiftY: int/float.
                The range of random shifts to apply along Y. A uniformly
                sampled shift between [-shiftY, +shiftY] is applied

            Note that the same operations are applied to all inputs except for
            the timeshift ones.
    order : int or list/tuple of ints, optional
        Interpolation order to use for each image in the input stack. If order
        is a scalar, the same order is applied to all images. If a list of
        orders is provided, each image in the stack will have its own operaiton
        order. See skimage.transform.wrap doc.
        Note that the histogram voodoo operation is only applied to images with
        a non-zero order.
        The default is 0.
    time : int or list of ints, optional
        Timepoint of each input. If a list is provided, inputs belonging to the
        same timepoint (e.g. 0 for previous timepoint images and 1 for current)
        will be treated the same in time-related transformations (e.g.
        timeshift, where current frames are shifted relative to previous frames)

    Returns
    -------
    output : list of 2D numpy arrays
        Augmented images array.

    """

    # processing inputs / initializing variables::
    output = list(images_input)
    orderlist = [order] * len(images_input) if isinstance(order, int) else list(order)

    # Apply augmentation operations:

    if "illumination_voodoo" in aug_par:
        if aug_par["illumination_voodoo"]:
            for index, item in enumerate(output):
                # Not super elegant, but tells me if binary or grayscale image
                if orderlist[index] > 0:
                    output[index] = illumination_voodoo(item)

    if "histogram_voodoo" in aug_par:
        if aug_par["histogram_voodoo"]:
            for index, item in enumerate(output):
                # Not super elegant, but tells me if binary or grayscale image
                if orderlist[index] > 0:
                    output[index] = histogram_voodoo(item)

    if "gaussian_noise" in aug_par:
        if aug_par["gaussian_noise"]:
            sigma = np.random.rand() * aug_par["gaussian_noise"]
            for index, item in enumerate(output):
                # Not super elegant, but tells me if binary or grayscale image
                if orderlist[index] > 0:
                    item = item + np.random.normal(
                        0, sigma, item.shape
                    )  # Add Gaussian noise
                    output[index] = (item - np.min(item)) / np.ptp(
                        item
                    )  # Rescale to 0-1

    if "gaussian_blur" in aug_par:
        if aug_par["gaussian_blur"]:
            sigma = np.random.rand() * aug_par["gaussian_blur"]
            for index, item in enumerate(output):
                # Not super elegant, but tells me if binary or grayscale image
                if orderlist[index] > 0:
                    item = gaussian(item, sigma, truncate=1 / 5)  # blur image
                    # item = cv2.GaussianBlur(item, (5, 5), sigma)  # blur image
                    output[index] = item

    if "elastic_deformation" in aug_par:
        output = elasticdeform.deform_random_grid(
            output,
            sigma=aug_par["elastic_deformation"]["sigma"],
            points=aug_par["elastic_deformation"]["points"],
            # Using bicubic interpolation instead of bilinear here
            order=[i * 3 for i in orderlist],
            mode="nearest",
            axis=(0, 1),
            prefilter=False,
        )

    if "horizontal_flip" in aug_par:
        if aug_par["horizontal_flip"]:
            if random.randint(0, 1):  # coin flip
                for index, item in enumerate(output):
                    output[index] = np.fliplr(item)

    if "vertical_flip" in aug_par:
        if aug_par["vertical_flip"]:
            if random.randint(0, 1):  # coin flip
                for index, item in enumerate(output):
                    output[index] = np.flipud(item)

    if "rotations_90d" in aug_par:  # Only works with square images right now!
        if aug_par["rotations_90d"]:
            rot = random.randint(0, 3) * 90.0
            if rot > 0:
                for index, item in enumerate(output):
                    output[index] = trans.rotate(
                        item, rot, mode="edge", order=orderlist[index]
                    )

    if "rotation" in aug_par:
        rot = random.uniform(-aug_par["rotation"], aug_par["rotation"])
        for index, item in enumerate(output):
            output[index] = trans.rotate(item, rot, mode="edge", order=orderlist[index])

    # Zoom and shift operations are processed together:
    if "zoom" in aug_par:
        # I want most of them to not be too zoomed
        zoom = random.expovariate(3 * 1 / aug_par["zoom"])
        zoom = aug_par["zoom"] if zoom > aug_par["zoom"] else zoom
    else:
        zoom = 0

    if "shiftX" in aug_par:
        shiftX = random.uniform(-aug_par["shiftX"], aug_par["shiftX"])
    else:
        shiftX = 0

    if "shiftY" in aug_par:
        shiftY = random.uniform(-aug_par["shiftY"], aug_par["shiftY"])
    else:
        shiftY = 0

    # Apply zoom & shifts:
    if any([abs(x) > 0 for x in [zoom, shiftX, shiftY]]):
        for index, item in enumerate(output):
            output[index] = zoomshift(
                item, zoom + 1, shiftX, shiftY, order=orderlist[index]
            )

    return output


def zoomshift(
    I: Image, zoomlevel: float, shiftX: float, shiftY: float, order: int = 0
) -> Image:
    """
    Adapted from DeLTA
    https://gitlab.com/dunloplab/delta

    This function zooms and shifts images.

    Parameters
    ----------
    I : 2D numpy array
        input image.
    zoomlevel : float
        Additional zoom to apply to the image.
    shiftX : float
        X-axis shift to apply to the image, in pixels.
    shiftY : float
        Y-axis shift to apply to the image, in pixels.
    order : int, optional
        Interpolation order. The default is 0.

    Returns
    -------
    I : 2D numpy array
        Zoomed and shifted image of same size as input.

    """

    oldshape = I.shape
    I = trans.rescale(I, zoomlevel, mode="edge", channel_axis=None, order=order)
    shiftX = shiftX * I.shape[0]
    shiftY = shiftY * I.shape[1]
    I = shift(I, (shiftY, shiftX), order=order)
    i0 = (
        round(I.shape[0] / 2 - oldshape[0] / 2),
        round(I.shape[1] / 2 - oldshape[1] / 2),
    )
    I = I[i0[0] : (i0[0] + oldshape[0]), i0[1] : (i0[1] + oldshape[1])]
    return I


def shift(image: Image, vector: Tuple[float, float], order: int = 0) -> Image:
    """
    Image shifting function

    Parameters
    ----------
    image : 2D numpy array
        Input image.
    vector : tuple of floats
        Translation/shit vector.
    order : int, optional
        Interpolation order. The default is 0.

    Returns
    -------
    shifted : 2D numpy image
        Shifted image.

    """
    transform = trans.AffineTransform(translation=vector)
    shifted = trans.warp(image, transform, mode="edge", order=order)

    return shifted


def histogram_voodoo(image: Image, num_control_points: int = 3) -> Image:
    """
    This function kindly provided by Daniel Eaton from the Paulsson lab.
    It performs an elastic deformation on the image histogram to simulate
    changes in illumination

    Parameters
    ----------
    image : 2D numpy array
        Input image.
    num_control_points : int, optional
        Number of inflection points to use on the histogram conversion curve.
        The default is 3.

    Returns
    -------
    2D numpy array
        Modified image.

    """
    control_points = np.linspace(0, 1, num=num_control_points + 2)
    sorted_points = copy.copy(control_points)
    random_points = np.random.uniform(low=0.1, high=0.9, size=num_control_points)
    sorted_points[1:-1] = np.sort(random_points)
    mapping = interpolate.PchipInterpolator(control_points, sorted_points)

    return mapping(image)


def illumination_voodoo(image: Image, num_control_points: int = 5) -> Image:
    """
    This function inspired by the one above.
    It simulates a variation in illumination along the length of the chamber

    Parameters
    ----------
    image : 2D numpy array
        Input image.
    num_control_points : int, optional
        Number of inflection points to use on the illumination multiplication
        curve.
        The default is 5.

    Returns
    -------
    newimage : 2D numpy array
        Modified image.

    """

    # Create a random curve along the length of the chamber:
    control_points = np.linspace(0, image.shape[0] - 1, num=num_control_points)
    random_points = np.random.uniform(low=0.1, high=0.9, size=num_control_points)
    mapping = interpolate.PchipInterpolator(control_points, random_points)
    curve = mapping(np.linspace(0, image.shape[0] - 1, image.shape[0]))
    # Apply this curve to the image intensity along the length of the chamebr:
    newimage = np.multiply(
        image,
        np.reshape(
            np.tile(np.reshape(curve, curve.shape + (1,)), (1, image.shape[1])),
            image.shape,
        ),
    )
    # Rescale values to original range:
    newimage = np.interp(
        newimage, (newimage.min(), newimage.max()), (image.min(), image.max())
    )

    return newimage


def readreshape(
    filename: str,
    target_size: Tuple[int, int] = (256, 32),
    binarize: bool = False,
    order: int = 1,
    rangescale: bool = True,
    preserve_range: bool = False,
) -> npt.NDArray[Any]:
    """
    Read image from disk and format it

    Parameters
    ----------
    filename : string
        Path to file. Only PNG, JPG or single-page TIFF files accepted
    target_size : tupe of int or None, optional
        Size to reshape the image.
        The default is (256,32).
    binarize : bool, optional
        Use the binarizerange() function on the image.
        The default is False.
    order : int, optional
        interpolation order (see skimage.transform.warp doc).
        0 is nearest neighbor
        1 is bilinear
        The default is 1.
    rangescale : bool, optional
        Scale array image values to 0-1 if True.
        The default is True.
    mode : str, optional
        Resize the
    Raises
    ------
    ValueError
        Raised if image file is not a PNG, JPEG, or TIFF file.

    Returns
    -------
    i : numpy 2d array of floats
        Loaded array.

    """
    fext = os.path.splitext(filename)[1].lower()
    if fext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        i = io.imread(filename)
    else:
        raise ValueError("Only PNG, JPG or single-page TIF files accepted")

    if i.ndim == 3:
        i = i[:, :, 0]
    # For the mother machine, all images are resized in 256x32
    img = trans.resize(
        i, target_size, anti_aliasing=True, order=order, preserve_range=preserve_range
    )

    if binarize:
        img = binarizerange(img)
    if rangescale:
        if np.ptp(img) != 0:
            img = (img - np.min(img)) / np.ptp(img)
    if np.max(img) == 255:
        img = img / 255
    return img


def binarizerange(array: Image) -> Image:
    """
    This function will binarize a numpy array by thresholding it in the middle
    of its range

    Parameters
    ----------
    array : 2D numpy array
            Input array/image.

    Returns
    -------
    newi : 2D numpy array
           Binarized image.

    """

    threshold = (np.amin(array) + np.amax(array)) / 2
    return np.array(array > threshold, dtype=array.dtype)


def seg_weights_2D(
    mask: npt.NDArray[np.uint8], classweights: Tuple[int, int] = (1, 1)
) -> npt.NDArray[np.float32]:
    """
    Adapted from DeLTA
    https://gitlab.com/dunloplab/delta

    Compute custom weightmaps designed for bacterial images
    where borders are difficult to distinguish

    Parameters
    ----------
    mask : 2D array
        Training output segmentation mask.
    classweights : tuple of 2 int/floats, optional
        Weights to apply to cells and border
        The default is (1,1)


    Returns
    -------
    weightmap : 2D array
        Weights map image.

    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if np.max(mask) == 255:
        mask[mask > 0] = 1

    # Extract all pixels that include the cells and its border
    border = morph.binary_closing(mask, footprint=morph.disk(20))

    # Set all pixels that include the cells to zero to leave behind the border only
    border[mask > 0] = 0

    # Erode the segmentation to avoid putting high emphasis on edges of cells
    mask_erode = morph.binary_erosion(mask, footprint=morph.disk(2))

    # Get the skeleton of the segmentation and border
    mask_skel = morph.skeletonize(mask_erode > 0)
    border_skel = morph.skeletonize(border > 0)

    # Find the distances from the skeleton of the segmention and border
    s, border_dist = morph.medial_axis(border_skel < 1, return_distance=True)
    s, mask_dist = morph.medial_axis(mask_skel < 1, return_distance=True)

    # Use the distance from the skeletons to create a gradient towards the skeleton
    border_gra = border * (classweights[1]) / (border_dist + 1) ** 2
    mask_gra = mask / (mask_dist + 1) ** 2

    # Set up weights array
    weightmap = np.zeros((mask.shape), dtype=np.float32)

    # Add the gradients for the segmentation and border into the weights array
    weightmap[mask_erode > 0] = mask_gra[mask_erode > 0]
    weightmap[border > 0] = border_gra[border > 0]

    # Set the skeletons of the segmentation and borders to the maximum values
    weightmap[mask_skel > 0] = classweights[0]
    weightmap[border_skel > 0] = classweights[1]

    # Keep the background zero and set the erode values in the seg/border to a minimum of 1
    bkgd = np.ones((mask.shape)) - mask - border
    weightmap[((weightmap == 0) * (bkgd < 1))] = 1 / 255

    return weightmap


def save_weights(mask_source_path, weights_path):

    mask_files = list(mask_source_path.glob("*.tif"))

    if not weights_path.exists():
        weights_path.mkdir()

    for mask_file in mask_files:
        information(f"Constructing weight map for {mask_file}")
        with tiff.TiffFile(mask_file) as tif:
            mask = tif.asarray()
        weightmap = seg_weights_2D(mask)
        tiff.imwrite(weights_path / mask_file.name, weightmap)


def predictGenerator(
    files_path: str,
    files_list: Union[List[str], Tuple[str, ...]] = None,
    target_size: Tuple[int, int] = (256, 32),
    crop: bool = False,
) -> Iterator[np.ndarray]:
    """
    Adapted from DeLTA
    https://gitlab.com/dunloplab/delta

    Get a generator for predicting segmentation on new image files
    once the segmentation U-Net has been trained.

    Parameters
    ----------
    files_path : string
        Path to image files folder.
    files_list : list/tuple of strings, optional
        List of file names to read in the folder. If empty, all
        files in the folder will be read.
        The default is [].
    target_size : tuple of 2 ints, optional
        Size for the images to be resized.
        The default is (256,32).

    Returns
    -------
    mygen : generator
        Generator that will yield single image files as 4D numpy arrays of
        size (1, target_size[0], target_size[1], 1).

    """

    files_list = files_list or sorted(os.listdir(files_path))

    def generator(
        files_path: str,
        files_list: Union[List[str], Tuple[str, ...]],
        target_size: Tuple[int, int],
    ) -> Iterator[np.ndarray]:
        for index, fname in enumerate(files_list):
            try:
                img = readreshape(
                    os.path.join(files_path, fname), target_size=target_size, order=1
                )
                # Tensorflow needs one extra single dimension (so that it is a 4D tensor)
                img = np.reshape(img, (1,) + img.shape)

                yield img
            except ValueError:
                pass

    mygen = generator(files_path, files_list, target_size)
    return mygen


def trainGenerator(
    img_names: np.ndarray,
    batch_size: int,
    mask_path: str,
    weight_path: str,
    target_size: Tuple[int, int] = (256, 32),
    augment_params: Dict[str, Any] = {},
    seed: int = 1,
) -> Iterator[Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]:

    """
    Generator for training the segmentation U-Net.

    Parameters
    ----------
    batch_size : int
        Batch size, number of training samples to concatenate together.
    img_path : string
        Path to folder containing training input images.
    mask_path : string
        Path to folder containing training segmentation groundtruth.
    weight_path : string or None.
        Path to folder containing weight map images.
    target_size : tuple of 2 ints, optional
        Input and output image size.
        The default is (256,32).
    augment_params : dict, optional
        Data augmentation parameters. See data_augmentation() doc for more info
        The default is {}.
    seed : int, optional
        Seed for numpy's random generator. see numpy.random.seed() doc
        The default is 1.

    Yields
    ------
    image_arr : 4D numpy array of floats
        Input images for the U-Net training routine. Dimensions of the tensor
        are (batch_size, target_size[0], target_size[1], 1)
    mask_wei_arr : 4D numpy array of floats
        Output masks and weight maps for the U-Net training routine. Dimensions
        of the tensor are (batch_size, target_size[0], target_size[1], 2)

    """

    # Reset the pseudo-random generator:
    random.seed(a=seed)

    image_arr = np.empty((batch_size,) + target_size + (1,), dtype=np.float32)
    mask_wei_arr = np.empty((batch_size,) + target_size + (2,), dtype=np.float32)

    while True:
        # Reset image arrays:

        for b in range(batch_size):
            # Pick random image index:
            index = random.randrange(0, len(img_names))

            # Read images:
            filename = img_names[index]
            img = readreshape(filename, target_size=target_size, order=1)
            mask = readreshape(
                os.path.join(mask_path, os.path.basename(filename)),
                target_size=target_size,
                binarize=True,
                order=1,
                rangescale=False,
            )

            weight = readreshape(
                os.path.join(weight_path, os.path.basename(filename)),
                target_size=target_size,
                order=0,
                rangescale=False,
            )

            # Data augmentation:
            [img, mask, weight] = data_augmentation(
                [img, mask, weight], augment_params, order=[1, 0, 0]
            )

            # Compile into output arrays:
            image_arr[b, :, :, 0] = img
            mask_wei_arr[b, :, :, 0] = mask
            mask_wei_arr[b, :, :, 1] = weight

        yield (image_arr, mask_wei_arr)


# define what happens at each layer
def conv_block(input_tensor, num_filters):
    """Creates a block of two convolutional layers with ReLU activation and batch normalization.

    Args:
        input_tensor: Input tensor to the block.
        num_filters: Number of filters in each convolutional layer.

    Returns:
        encoder: Output tensor of the block.
    """
    encoder = Conv2D(num_filters, (3, 3), padding="same", activation="relu")(
        input_tensor
    )
    encoder = Conv2D(num_filters, (3, 3), padding="same", activation="relu")(encoder)
    return encoder


def encoder_block(input_tensor, num_filters):

    """Creates an encoder block consisting of a convolutional block followed by max pooling.

    Args:
        input_tensor: Input tensor to the block.
        num_filters: Number of filters in the convolutional block.

    Returns:
        encoder_pool: Max pooled output tensor of the block.
        encoder: Output tensor of the convolutional block.
    """

    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):

    """Creates a decoder block consisting of an up-sampling layer, concatenation with skip connection,
    and a convolutional block.

    Args:
        input_tensor: Input tensor to the block.
        concat_tensor: Skip connection tensor from the encoder block.
        num_filters: Number of filters in the convolutional block.

    Returns:
        decoder: Output tensor of the decoder block.
    """

    decoder = Conv2DTranspose(
        num_filters, (2, 2), strides=(2, 2), padding="same", activation="relu"
    )(input_tensor)
    decoder = Concatenate(axis=-1)([concat_tensor, decoder])
    decoder = Conv2D(num_filters, (3, 3), padding="same", activation="relu")(decoder)
    decoder = Conv2D(num_filters, (3, 3), padding="same", activation="relu")(decoder)

    return decoder


def unet(target_size=(256, 32, 1), num_filters=64):
    """Creates a U-Net model consisting of an encoder, center, and decoder.

    Args:
        target_size: Size of the input image.

    Returns:
        model: U-Net model.
    """

    # make the layers
    inputs = Input(shape=target_size)
    # 256
    encoder0_pool, encoder0 = encoder_block(inputs, num_filters)
    # 128
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, num_filters * 2)
    # 64
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, num_filters * 4)
    # 32
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, num_filters * 8)
    # 16
    center = conv_block(encoder3_pool, num_filters * 16)  # we were using 128 before
    # center
    # 32
    decoder3 = decoder_block(center, encoder3, num_filters * 8)
    # 64
    decoder2 = decoder_block(decoder3, encoder2, num_filters * 4)
    # 64
    decoder1 = decoder_block(decoder2, encoder1, num_filters * 2)
    # 128
    decoder0 = decoder_block(decoder1, encoder0, num_filters)
    # 256
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(decoder0)

    # make the model
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


# Use the following model for segmentation:
def unet_seg(
    pretrained_weights: str = None,
    input_size: Tuple[int, int, int] = (256, 32, 1),
    levels: int = 5,
) -> Model:
    """
    Cell segmentation U-Net definition function.

    Parameters
    ----------
    pretrained_weights : hdf5 file, optional
        Model will load weights from hdf5 and start training.
        The default is None
    input_size : tuple of 3 ints, optional
        Dimensions of the input tensor, without batch size.
        The default is (256,32,1).
    levels : int, optional
        Number of levels of the U-Net, ie number of successive contraction then
        expansion blocks are combined together.
        The default is 5.

    Returns
    -------
    model : Model
        Segmentation U-Net (compiled).

    """

    model = unet(input_size)

    loss = pixelwise_weighted_bce
    metrics = [binary_acc]

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=loss,
        metrics=metrics,
    )

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


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


def train_model(
    image_dir,
    mask_dir,
    weights_dir,
    test_dir,
    target_size_seg,
    savefile,
    model_source,
    validation_split,
    batch_size=4,
    epochs=600,
    patience=50,
):

    # Data generator parameters:
    data_gen_args = dict(
        rotation=2,
        rotations_90d=True,
        zoom=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        histogram_voodoo=True,
        illumination_voodoo=True,
        gaussian_noise=0.03,
        gaussian_blur=1,
    )

    # Get training image files list:
    image_name_arr = glob.glob(str(image_dir / "*.png")) + glob.glob(
        str(image_dir / "*.tif")
    )

    # randomize and split between training and test
    np.random.shuffle(image_name_arr)

    training_n = int(len(image_name_arr) * validation_split)
    training_set = image_name_arr[:training_n]
    validation_set = image_name_arr[training_n:]

    information(f"Training set has {len(training_set)} images")
    information(f"Validation set has {len(validation_set)} images")

    # Training data generator:
    train_generator = trainGenerator(
        training_set,
        batch_size,
        mask_dir,
        weights_dir,
        augment_params=data_gen_args,
        target_size=target_size_seg,
    )

    # Validation data generator:
    val_generator = trainGenerator(
        validation_set,
        batch_size,
        mask_dir,
        weights_dir,
        augment_params=data_gen_args,
        target_size=target_size_seg,
    )

    # Define model:
    model = unet_seg(pretrained_weights=model_source, input_size=target_size_seg + (1,))
    model.summary()

    # Callbacks:
    model_checkpoint = ModelCheckpoint(
        savefile, monitor="val_loss", verbose=2, save_best_only=True
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", verbose=2, patience=patience
    )

    steps_per_epoch_train = int(np.ceil(len(training_set) / float(batch_size)))
    steps_per_epoch_val = int(np.ceil(len(validation_set) / float(batch_size)))

    # Train:
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch_train,
        epochs=epochs,
        callbacks=[model_checkpoint, early_stopping],
        validation_data=val_generator,
        validation_steps=steps_per_epoch_val,
    )

    try:
        predict_gen = predictGenerator(str(test_dir))
        predictions = model.predict(predict_gen)
        napari.current_viewer().add_image(predictions)

    except:
        pass


class TrainUnet(MM3Container):
    def __init__(self, napari_viewer: Viewer):
        super().__init__(napari_viewer=napari_viewer, validate_folders=False)

    def create_widgets(self):

        self.image_widget = FileEdit(
            mode="d",
            label="image directory",
            value=self.analysis_folder / "training" / "images",
        )

        self.mask_widget = FileEdit(
            mode="d",
            label="mask directory",
            value=self.analysis_folder / "training" / "masks",
        )

        self.weights_widget = FileEdit(
            mode="d",
            label="weights directory",
            value=self.analysis_folder / "training" / "weights",
        )

        self.test_widget = FileEdit(
            mode="d",
            label="test data directory",
            value=self.analysis_folder / "training" / "test",
        )

        self.load_existing_widget = CheckBox(label="Load pretrained weights")

        self.model_source_widget = FileEdit(mode="r", label="source model", value=None)

        self.model_file_widget = FileEdit(
            mode="r",
            label="model file",
            tooltip="Location to save model to.",
            value=Path(".") / "models" / "test.hdf5",
        )

        self.validation_split_widget = FloatSpinBox(
            label="Training / validation split", min=0, max=1, step=0.05, value=0.9
        )

        self.batch_size_widget = SpinBox(
            label="batch size",
            tooltip="how large to make the batches. different speeds are faster on different computers.",
            value=4,
            min=1,
            max=9999,
        )

        self.epochs_widget = SpinBox(
            label="epochs",
            tooltip="Number of epochs to train for",
            value=200,
            min=1,
            max=9999,
        )

        self.patience_widget = SpinBox(
            label="patience", tooltip="Patience", value=50, min=1, max=9999
        )

        self.height_widget = SpinBox(label="image height", min=1, max=5000, value=256)
        self.width_widget = SpinBox(label="image width", min=1, max=5000, value=32)

        self.run_widget = PushButton(text="Train model")

        self.predict_widget = PushButton(text="Check predictions")
        self.preview_widget = PushButton(text="Preview training data")

        self.append(self.image_widget)
        self.append(self.mask_widget)
        self.append(self.weights_widget)
        self.append(self.test_widget)
        self.append(self.model_file_widget)
        self.append(self.load_existing_widget)
        self.append(self.model_source_widget)
        self.append(self.validation_split_widget)
        self.append(self.batch_size_widget)
        self.append(self.epochs_widget)
        self.append(self.patience_widget)
        self.append(self.height_widget)
        self.append(self.width_widget)
        self.append(self.run_widget)
        self.append(self.preview_widget)
        self.append(self.predict_widget)

        self.image_dir = self.image_widget.value
        self.mask_dir = self.mask_widget.value
        self.test_dir = self.test_widget.value
        self.target_size = (self.height_widget.value, self.width_widget.value)
        self.model_path_out = self.model_file_widget.value
        self.weights_dir = self.weights_widget.value
        self.epochs = self.epochs_widget.value
        self.patience = self.patience_widget.value
        self.batch_size = self.batch_size_widget.value
        self.model_source = self.model_source_widget.value
        self.load_existing = self.load_existing_widget.value
        self.validation_split = self.validation_split_widget.value

        self.run_widget.clicked.connect(self.run)
        self.predict_widget.clicked.connect(self.predict)
        self.preview_widget.clicked.connect(self.render_preview)

        self.model_file_widget.changed.connect(self.set_model_file)
        self.image_widget.changed.connect(self.set_image_dir)
        self.mask_widget.changed.connect(self.set_mask_dir)
        self.weights_widget.changed.connect(self.set_weights_dir)
        self.test_widget.changed.connect(self.set_test_dir)
        self.epochs_widget.changed.connect(self.set_epochs)
        self.patience_widget.changed.connect(self.set_patience)
        self.height_widget.changed.connect(self.set_target_size)
        self.width_widget.changed.connect(self.set_target_size)
        self.model_source_widget.changed.connect(self.set_model_source)
        self.load_existing_widget.changed.connect(self.set_pretrained_weights)
        self.validation_split_widget.changed.connect(self.set_validation_split)

    def run(self):
        """Overriding method. Perform mother machine analysis."""

        information("Making pixelwise weight maps")
        save_weights(self.mask_dir, self.weights_dir)

        if self.load_existing:
            model_source = self.model_source
        else:
            model_source = None

        information("Loading training data")

        train_model(
            self.image_dir,
            self.mask_dir,
            self.weights_dir,
            self.test_dir,
            self.target_size,
            self.model_path_out,
            model_source,
            self.validation_split,
            batch_size=self.batch_size,
            epochs=self.epochs,
            patience=self.patience,
        )

    def predict(self):

        self.viewer.layers.clear()

        model = models.load_model(
            self.model_path_out,
            custom_objects={
                "binary_acc": binary_acc,
                "pixelwise_weighted_bce": pixelwise_weighted_bce,
            },
        )
        predict_gen, predict_gen_v = itertools.tee(predictGenerator(str(self.test_dir)))

        predictions = model.predict(predict_gen)

        self.viewer.add_image(np.squeeze(np.array(list(predict_gen_v))))
        self.viewer.add_image(np.squeeze(predictions), opacity=0.2, colormap="yellow")

    def render_preview(self):
        self.viewer.layers.clear()

        img_path = self.image_widget.value

        preload_mask = []
        preload_img = []
        preload_weight = []

        # Get training image files list:
        image_name_arr = glob.glob(str(img_path / "*.png")) + glob.glob(
            str(img_path / "*.tif")
        )

        for i, name in enumerate(image_name_arr):
            print(str(i) + " " + name)

        save_weights(self.mask_dir, self.weights_dir)

        # If preloading, load the images and compute weight maps:
        for filename in image_name_arr:
            preload_img.append(
                readreshape(
                    filename,
                    target_size=self.target_size,
                    order=1,
                )
            )
            preload_mask.append(
                readreshape(
                    os.path.join(self.mask_dir, os.path.basename(filename)),
                    target_size=self.target_size,
                    binarize=True,
                    order=1,
                    rangescale=False,
                )
            )
            preload_weight.append(
                readreshape(
                    os.path.join(self.weights_dir, os.path.basename(filename)),
                    target_size=self.target_size,
                    order=0,
                    rangescale=False,
                ),
            )
        self.viewer.add_image(np.stack(preload_img))
        self.viewer.add_labels(np.stack(preload_mask).astype(int))
        self.viewer.add_image(np.stack(preload_weight), name="Weights")
        self.viewer.layers[-1].opacity = 0.25
        self.viewer.layers[-2].opacity = 0.5
        return

    def set_model_source(self):
        self.model_source = self.model_source_widget.value

    def set_model_file(self):
        self.model_path_out = self.model_file_widget.value

    def set_image_dir(self):
        self.image_dir = self.image_widget.value

    def set_mask_dir(self):
        self.mask_dir = self.mask_widget.value

    def set_weights_dir(self):
        self.weights_dir = self.weights_widget.value

    def set_test_dir(self):
        self.test_dir = self.test_widget.value

    def set_epochs(self):
        self.epochs = self.epochs_widget.value

    def set_patience(self):
        self.patience = self.patience_widget.value

    def set_target_size(self):
        self.target_size = (self.height_widget.value, self.width_widget.value)

    def set_pretrained_weights(self):
        self.load_existing = self.load_existing_widget.value

    def set_validation_split(self):
        self.validation_split = self.validation_split_widget.value
