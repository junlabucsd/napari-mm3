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

import multiprocessing

import napari
from napari import Viewer
from magicgui import magicgui
from magicgui.widgets import FileEdit, SpinBox, FloatSlider, CheckBox

# learning modules
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import models
from tensorflow.python.ops import array_ops, math_ops
from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
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

from ._deriving_widgets import MM3Container, warning

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
    I = trans.rescale(I, zoomlevel, mode="edge", multichannel=False, order=order)
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
    crop: bool = False,
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
    if not crop:
        img = trans.resize(i, target_size, anti_aliasing=True, order=order)

    else:
        fill_shape = [
            target_size[j] if i.shape[j] < target_size[j] else i.shape[j]
            for j in range(2)
        ]
        img = np.zeros((fill_shape[0], fill_shape[1]))
        img[0 : i.shape[0], 0 : i.shape[1]] = i

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
    # border = cv2.morphEx(mask, cv2.MORPH_CLOSE, kernel(20))
    border = morph.binary_closing(mask, footprint=morph.disk(20))

    # Set all pixels that include the cells to zero to leave behind the border only
    border[mask > 0] = 0

    # Erode the segmentation to avoid putting high emphasiss on edges of cells
    # mask_erode = cv2.erode(mask, kernel(2))
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

    mask_files = glob.glob(os.path.join(mask_source_path, "*.tif"))
    mask_names = [name.split("/")[-1] for name in mask_files]

    if not weights_path.exists():
        weights_path.mkdir()

    for (mask_file, mask_name) in zip(mask_files, mask_names):

        with tiff.TiffFile(mask_file) as tif:
            mask = tif.asarray()

        weightmap = seg_weights_2D(mask)

        tiff.imwrite(weights_path / mask_name, weightmap)


def predictGenerator_seg(
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
                    os.path.join(files_path, fname),
                    target_size=target_size,
                    order=1,
                    crop=crop,
                )
            except ValueError:
                warning('Could not resize image')
                continue
            # Tensorflow needs one extra single dimension (so that it is a 4D tensor)
            img = np.reshape(img, (1,) + img.shape)

            yield img

    mygen = generator(files_path, files_list, target_size)
    return mygen


def trainGenerator_seg(
    batch_size: int,
    img_path: str,
    mask_path: str,
    weight_path: Optional[str],
    target_size: Tuple[int, int] = (256, 32),
    augment_params: Dict[str, Any] = {},
    preload: bool = False,
    seed: int = 1,
    crop_windows: bool = False,
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
    preload : bool, optional
        Flag to load all training inputs in memory during intialization of the
        generator.
        The default is False.
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

    preload_mask = []
    preload_img = []
    preload_weight = []

    # Get training image files list:
    image_name_arr = glob.glob(str(img_path / "*.png")) + glob.glob(
        str(img_path / "*.tif")
    )

    #load the images and compute weight maps:
    if preload:
        for filename in image_name_arr:
            preload_img.append(
                readreshape(
                    filename, target_size=target_size, order=1, crop=crop_windows
                )
            )
            preload_mask.append(
                readreshape(
                    os.path.join(mask_path, os.path.basename(filename)),
                    target_size=target_size,
                    binarize=True,
                    order=0,
                    rangescale=False,
                    crop=crop_windows,
                )
            )
            if weight_path is not None:
                preload_weight.append(
                    readreshape(
                        os.path.join(weight_path, os.path.basename(filename)),
                        target_size=target_size,
                        order=0,
                        rangescale=False,
                        crop=crop_windows,
                    ),
                )

    # Reset the pseudo-random generator:
    random.seed(a=seed)

    image_arr = np.empty((batch_size,) + target_size + (1,), dtype=np.float32)
    if weight_path is None:
        mask_wei_arr = np.empty((batch_size,) + target_size + (1,), dtype=np.float32)
    else:
        mask_wei_arr = np.empty((batch_size,) + target_size + (2,), dtype=np.float32)

    while True:
        # Reset image arrays:

        for b in range(batch_size):
            # Pick random image index:
            index = random.randrange(0, len(image_name_arr))

            if preload:
                # Get from preloaded arrays:
                img = preload_img[index]
                mask = preload_mask[index]
                weight = preload_weight[index]
            else:
                # Read images:
                filename = image_name_arr[index]
                img = readreshape(
                    filename, target_size=target_size, order=1, crop=crop_windows
                )
                mask = readreshape(
                    os.path.join(mask_path, os.path.basename(filename)),
                    target_size=target_size,
                    binarize=True,
                    order=0,
                    rangescale=False,
                    crop=crop_windows,
                )
                if weight_path is not None:
                    weight = readreshape(
                        os.path.join(weight_path, os.path.basename(filename)),
                        target_size=target_size,
                        order=0,
                        rangescale=False,
                        crop=crop_windows,
                    )

            if crop_windows:
                y0 = np.random.randint(0, img.shape[0] - (target_size[0] - 1))
                y1 = y0 + target_size[0]
                x0 = np.random.randint(0, img.shape[1] - (target_size[1] - 1))
                x1 = x0 + target_size[1]

                img = img[y0:y1, x0:x1]
                mask = mask[y0:y1, x0:x1]
                if weight_path is not None:
                    weight = weight[y0:y1, x0:x1]

            # Data augmentation:
            if weight_path is not None:
                [img, mask, weight] = data_augmentation(
                    [img, mask, weight], augment_params, order=[1, 0, 0]
                )
            else:
                [img, mask] = data_augmentation(
                    [img, mask], augment_params, order=[1, 0]
                )

            # Compile into output arrays:
            image_arr[b, :, :, 0] = img
            mask_wei_arr[b, :, :, 0] = mask
            if weight_path is not None:
                mask_wei_arr[b, :, :, 1] = weight

        yield (image_arr, mask_wei_arr)


def expanding_block(
    input_layer: tf.Tensor,
    skip_layer: tf.Tensor,
    filters: int,
    conv2d_parameters: Dict,
    dropout: float = 0,
    name: str = "Expanding",
) -> tf.Tensor:
    """
    A block of layers for 1 expanding level of the U-Net

    Parameters
    ----------
    input_layer : tf.Tensor
        The convolutional layer that is the output of the lower level's
        expanding block
    skip_layer : tf.Tensor
        The convolutional layer that is the output of this level's
        contracting block
    filters : int
        filters input for the Conv2D layers of the block.
    conv2d_parameters : dict()
        kwargs for the Conv2D layers of the block.
    dropout : float, optional
        Dropout layer rate in the block. Valid range is [0,1). If 0, no dropout
        layer is added.
        The default is 0
    name : str, optional
        Name prefix for the layers in this block. The default is "Expanding".

    Returns
    -------
    conv3 : tf.Tensor
        Output of this level's expanding block.

    """

    # Up-sampling:
    up = UpSampling2D(size=(2, 2), name=name + "_UpSampling2D")(input_layer)
    conv1 = Conv2D(filters, 2, **conv2d_parameters, name=name + "_Conv2D_1")(up)

    # Merge with skip connection layer:
    merge = Concatenate(axis=3, name=name + "_Concatenate")([skip_layer, conv1])

    # Convolution layers:
    conv2 = Conv2D(filters, 3, **conv2d_parameters, name=name + "_Conv2D_2")(merge)
    conv3 = Conv2D(filters, 3, **conv2d_parameters, name=name + "_Conv2D_3")(conv2)

    # Add dropout if necessary, otherwise return:
    if dropout == 0:
        return conv3
    else:
        drop = Dropout(dropout, name=name + "_Dropout")(conv3)
        return drop


def contracting_block(
    input_layer: tf.Tensor,
    filters: int,
    conv2d_parameters: Dict,
    dropout: float = 0,
    name: str = "Contracting",
) -> tf.Tensor:
    """
    A block of layers for 1 contracting level of the U-Net

    Parameters
    ----------
    input_layer : tf.Tensor
        The convolutional layer that is the output of the upper level's
        contracting block.
    filters : int
        filters input for the Conv2D layers of the block.
    conv2d_parameters : dict()
        kwargs for the Conv2D layers of the block.
    dropout : float, optional
        Dropout layer rate in the block. Valid range is [0,1). If 0, no dropout
        layer is added.
        The default is 0
    name : str, optional
        Name prefix for the layers in this block. The default is "Contracting".

    Returns
    -------
    conv2 : tf.Tensor
        Output of this level's contracting block.

    """

    # Pooling layer: (sample 'images' down by factor 2)
    pool = MaxPooling2D(pool_size=(2, 2), name=name + "_MaxPooling2D")(input_layer)

    # First convolution layer:
    conv1 = Conv2D(filters, 3, **conv2d_parameters, name=name + "_Conv2D_1")(pool)

    # Second convolution layer:
    conv2 = Conv2D(filters, 3, **conv2d_parameters, name=name + "_Conv2D_2")(conv1)

    # Add dropout if necessary, otherwise return:
    if dropout == 0:
        return conv2
    else:
        drop = Dropout(dropout, name=name + "_Dropout")(conv2)
        return drop


# Generic unet declaration:
def unet(
    input_size: Tuple[int, int, int] = (256, 32, 1),
    final_activation: str = "sigmoid",
    output_classes: int = 1,
    dropout: float = 0,
    levels: int = 5,
) -> Model:
    """
    Generic U-Net declaration.

    Parameters
    ----------
    input_size : tuple of 3 ints, optional
        Dimensions of the input tensor, excluding batch size.
        The default is (256,32,1).
    final_activation : string or function, optional
        Activation function for the final 2D convolutional layer. see
        keras.activations
        The default is 'sigmoid'.
    output_classes : int, optional
        Number of output classes, ie dimensionality of the output space of the
        last 2D convolutional layer.
        The default is 1.
    dropout : float, optional
        Dropout layer rate in the contracting & expanding blocks. Valid range
        is [0,1). If 0, no dropout layer is added.
        The default is 0.
    levels : int, optional
        Number of levels of the U-Net, ie number of successive contraction then
        expansion blocks are combined together.
        The default is 5.

    Returns
    -------
    model : Model
        Defined U-Net model (not compiled yet).

    """

    # Default conv2d parameters:
    conv2d_parameters = {
        "activation": "relu",
        "padding": "same",
        "kernel_initializer": "he_normal",
    }

    # Inputs layer:
    inputs = Input(input_size, name="true_input")

    # First level input convolutional layers:
    filters = 64
    conv = Conv2D(filters, 3, **conv2d_parameters, name="Level0_Conv2D_1")(inputs)
    conv = Conv2D(filters, 3, **conv2d_parameters, name="Level0_Conv2D_2")(conv)

    # Generate contracting path:
    level = 0
    contracting_outputs = [conv]
    for level in range(1, levels):
        filters *= 2
        contracting_outputs.append(
            contracting_block(
                contracting_outputs[-1],
                filters,
                conv2d_parameters,
                dropout=dropout,
                name=f"Level{level}_Contracting",
            )
        )

    # Generate expanding path:
    expanding_output = contracting_outputs.pop()
    while level > 0:
        level -= 1
        filters = int(filters / 2)
        expanding_output = expanding_block(
            expanding_output,
            contracting_outputs.pop(),
            filters,
            conv2d_parameters,
            dropout=dropout,
            name=f"Level{level}_Expanding",
        )

    # Final output layer:
    output = Conv2D(output_classes, 1, activation=final_activation, name="true_output")(
        expanding_output
    )

    model = Model(inputs=inputs, outputs=output)

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

    model = unet(
        input_size=input_size,
        final_activation="sigmoid",
        output_classes=1,
        levels=levels,
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=pixelwise_weighted_binary_crossentropy_seg,
        metrics=[unstack_acc],
    )

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def pixelwise_weighted_binary_crossentropy_seg(
    y_true: tf.Tensor, y_pred: tf.Tensor
) -> tf.Tensor:
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


def unstack_acc(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
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
    test_dir,
    target_size_seg,
    savefile,
    model_source,
    batch_size=4,
    epochs=600,
    steps_per_epoch=300,
    patience=50,
):
    # Adapted from DeLTA
    # https://gitlab.com/dunloplab/delta

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

    # Generator init:
    myGene = trainGenerator_seg(
        batch_size,
        image_dir,
        mask_dir,
        image_dir.parent / "weights",
        augment_params=data_gen_args,
        target_size=target_size_seg,
        crop_windows=False,
    )

    # Define model:
    model = unet_seg(pretrained_weights=model_source, input_size=target_size_seg + (1,))
    model.summary()

    # Callbacks:
    model_checkpoint = ModelCheckpoint(
        savefile, monitor="loss", verbose=2, save_best_only=True
    )
    early_stopping = EarlyStopping(
        monitor="loss", mode="min", verbose=2, patience=patience
    )

    # Train:
    history = model.fit(
        myGene,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[model_checkpoint, early_stopping],
    )

    predict_gen = predictGenerator_seg(str(test_dir))

    predictions = model.predict(predict_gen)

    napari.current_viewer().add_image(predictions)


class TrainUnet(MM3Container):
    def __init__(self, napari_viewer: Viewer):
        super().__init__(napari_viewer=napari_viewer, validate_folders=False)

    def create_widgets(self):

        self.image_widget = FileEdit(
            mode="d",
            label="image directory",
            value=Path(self.analysis_folder / "training/images/"),
        )

        self.mask_widget = FileEdit(
            mode="d",
            label="mask directory",
            value=Path(self.analysis_folder / "training/masks/"),
        )

        self.weights_widget = FileEdit(
            mode="d",
            label="weights directory",
            value=Path(self.analysis_folder / "training/weights"),
        )

        self.test_widget = FileEdit(
            mode="d",
            label="test data directory",
            value=Path(self.analysis_folder / "training/test/"),
        )

        self.model_source_widget = FileEdit(mode="r", label="source model", value=None)

        self.model_file_widget = FileEdit(
            mode="r",
            label="model file",
            tooltip="Location to save model to.",
            value=Path(self.analysis_folder / "models/test.hdf5"),
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

        self.steps_per_epoch_widget = SpinBox(
            label="steps per epoch",
            tooltip="Number of steps per epoch",
            value=300,
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
        self.append(self.test_widget)
        self.append(self.weights_widget)
        self.append(self.model_file_widget)
        self.append(self.model_source_widget)
        self.append(self.batch_size_widget)
        self.append(self.epochs_widget)
        self.append(self.steps_per_epoch_widget)
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
        self.steps_per_epoch = self.steps_per_epoch_widget.value
        self.patience = self.patience_widget.value
        self.batch_size = self.batch_size_widget.value
        self.model_source = self.model_source_widget.value

        self.run_widget.clicked.connect(self.run)
        self.predict_widget.clicked.connect(self.predict)
        self.preview_widget.clicked.connect(self.render_preview)

        self.model_file_widget.changed.connect(self.set_model_file)
        self.image_widget.changed.connect(self.set_image_dir)
        self.weights_widget.changed.connect(self.set_weights_dir)
        self.test_widget.changed.connect(self.set_test_dir)
        self.epochs_widget.changed.connect(self.set_epochs)
        self.steps_per_epoch_widget.changed.connect(self.set_steps_per_epoch)
        self.patience_widget.changed.connect(self.set_patience)
        self.height_widget.changed.connect(self.set_target_size)
        self.width_widget.changed.connect(self.set_target_size)
        self.model_source_widget.changed.connect(self.set_model_source)

    def run(self):
        """Overriding method. Perform mother machine analysis."""
        if self.model_source == Path('.'):
            self.model_source = None
       
        save_weights(self.mask_dir, self.weights_dir)

        train_model(
            self.image_dir,
            self.mask_dir,
            self.test_dir,
            self.target_size,
            self.model_path_out,
            self.model_source,
            batch_size=self.batch_size,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            patience=self.patience,
        )

    def predict(self):

        self.viewer.layers.clear()

        model = models.load_model(
            self.model_path_out,
            custom_objects={
                "unstack_acc": unstack_acc,
                "pixelwise_weighted_binary_crossentropy_seg": pixelwise_weighted_binary_crossentropy_seg,
            },
        )

        predict_gen, predict_gen_v = itertools.tee(
            predictGenerator_seg(str(self.test_dir),crop=False)
        )

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

        # load pre-computed weights if they exist
        try:
            for filename in image_name_arr:
                preload_weight.append(
                readreshape(
                    os.path.join(self.weights_dir, os.path.basename(filename)),
                    target_size=self.target_size,
                    order=0,
                    rangescale=False,
                    crop=False,
                ),
                )
        # if not, compute new weights
        except ValueError:
            preload_weight = []
            save_weights(self.mask_dir, self.weights_dir)
            for filename in image_name_arr:
                preload_weight.append(
                readreshape(
                    os.path.join(self.weights_dir, os.path.basename(filename)),
                    target_size=self.target_size,
                    order=0,
                    rangescale=False,
                    crop=False,
                ),
                )

        #load the images and weight maps:
        for filename in image_name_arr:
            preload_img.append(
                readreshape(
                    filename, target_size=self.target_size, order=1, crop=False
                )
            )
            preload_mask.append(
                readreshape(
                    os.path.join(self.mask_dir, os.path.basename(filename)),
                    target_size=self.target_size,
                    binarize=True,
                    order=0,
                    rangescale=False,
                    crop=False,
                )
            )
        self.viewer.add_image(np.stack(preload_img))
        self.viewer.add_labels(np.stack(preload_mask))
        self.viewer.add_image(np.stack(preload_weight), name="Weights")
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

    def set_steps_per_epoch(self):
        self.steps_per_epoch = self.steps_per_epoch_widget.value

    def set_patience(self):
        self.patience = self.patience_widget.value

    def set_target_size(self):
        self.target_size = (self.height_widget.value, self.width_widget.value)
