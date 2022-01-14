"""
This module is an example of a barebones function plugin for napari

It implements the ``napari_experimental_provide_function`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from enum import Enum
import numpy as np
from napari_plugin_engine import napari_hook_implementation
from skimage.filters import threshold_otsu # segmentation
from skimage import morphology # many functions is segmentation used from this
from skimage import segmentation # used in make_masks and segmentation
from scipy import ndimage as ndi # labeling and distance transform

if TYPE_CHECKING:
    import napari


# This is the actual plugin function, where we export our function
# (The functions themselves are defined below)
@napari_hook_implementation
def napari_experimental_provide_function():
    # we can return a single function
    # or a tuple of (function, magicgui_options)
    # or a list of multiple functions with or without options, as shown here:
    #return [Segment, threshold, image_arithmetic]
    return [Segment]


# 1.  First example, a simple function that thresholds an image and creates a labels layer
def threshold(data: "napari.types.ImageData", threshold: int) -> "napari.types.LabelsData":
    """Threshold an image and return a mask."""
    return (data > threshold).astype(int)

# 2.  Segmentation of Subtsracted images
def Segment(datas: "napari.types.ImageData", OTSU_threshold: float =1.0 , first_opening_size:int=2,
distance_threshold:int=2, second_opening_size:int=1, min_object_size:int=25) -> "napari.types.LabelsData":
    """Segments subtracted images and returns the labeled images"""

    labels=np.zeros_like(datas)
    for i in range(len(datas)):
        data=datas[i]
        # threshold image
        try:
            thresh = threshold_otsu(data) # finds optimal OTSU threshhold value
        except:
            continue

        threshholded = data > OTSU_threshold*thresh # will create binary image

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
            continue

        ### Calculate distance matrix, use as markers for random walker (diffusion watershed)
        # Generate the markers based on distance to the background
        distance = ndi.distance_transform_edt(morph)

        # threshold distance image
        distance_thresh = np.zeros_like(distance)
        distance_thresh[distance < distance_threshold] = 0
        distance_thresh[distance >= distance_threshold] = 1

        # do an extra opening on the distance
        distance_opened = morphology.binary_opening(distance_thresh,
                                                    morphology.disk(second_opening_size))

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
            continue

        # relabel now that small objects and labels on edges have been cleared
        markers = morphology.label(cleared, connectivity=1)

        # just break if there is no label
        if np.amax(markers) == 0:
            continue

        # the binary image for the watershed, which uses the unmodified OTSU threshold
        threshholded_watershed = threshholded
        threshholded_watershed = segmentation.clear_border(threshholded_watershed)

        # label using the random walker (diffusion watershed) algorithm
        try:
            # set anything outside of OTSU threshold to -1 so it will not be labeled
            markers[threshholded_watershed == 0] = -1
            # here is the main algorithm
            labeled_image = segmentation.random_walker(-1*data, markers)
            # put negative values back to zero for proper image
            labeled_image[labeled_image == -1] = 0
        except:
            continue
        
        labels[i]=labeled_image
    return labels

# 3. Second example, a function that adds, subtracts, multiplies, or divides two layers

# using Enums is a good way to get a dropdown menu.  Used here to select from np functions
class Operation(Enum):
    add = np.add
    subtract = np.subtract
    multiply = np.multiply
    divide = np.divide


def image_arithmetic(
    layerA: "napari.types.ImageData", operation: Operation, layerB: "napari.types.ImageData"
) -> "napari.types.LayerDataTuple":
    """Adds, subtracts, multiplies, or divides two same-shaped image layers."""
    return (operation.value(layerA, layerB), {"colormap": "turbo"})
