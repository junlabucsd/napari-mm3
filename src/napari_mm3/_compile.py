import multiprocessing
import re
import h5py
import os
import yaml
import six
import pickle
import glob
import sys
import traceback
import tifffile as tiff
import numpy as np
import json
import struct
import nd2reader

from scipy import ndimage as ndi
from skimage.feature import match_template
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint
from scipy.signal import find_peaks_cwt
from magicgui.widgets import FloatSpinBox, SpinBox, PushButton, ComboBox
from napari import Viewer
from napari.utils import progress
from ._deriving_widgets import (
    MM3Container,
    FOVChooser,
    TimeRangeSelector,
    PlanePicker,
    information,
    warning,
    load_stack_params,
)


#### Helpful utility functions.


def get_plane(filepath):
    pattern = r"(c\d+).tif"
    res = re.search(pattern, filepath, re.IGNORECASE)
    if res != None:
        return res.group(1)
    else:
        return None


def get_fov(filepath):
    pattern = r"xy(\d+)\w*.tif"
    res = re.search(pattern, filepath, re.IGNORECASE)
    if res != None:
        return int(res.group(1))
    else:
        return None


def get_time(filepath):
    pattern = r"t(\d+)xy\w+.tif"
    res = re.search(pattern, filepath, re.IGNORECASE)
    if res != None:
        return np.int_(res.group(1))
    else:
        return None


### Functions for working with TIFF metadata ###

# get params is the major function which processes raw TIFF images
def get_tif_params(params, image_filename, find_channels=True):
    """This is a damn important function for getting the information
    out of an image. It loads a tiff file, pulls out the image data, and the metadata,
    including the location of the channels if flagged.

    it returns a dictionary like this for each image:

    'filename': image_filename,
    'fov' : image_metadata['fov'], # fov id
    't' : image_metadata['t'], # time point
    'jdn' : image_metadata['jdn'], # absolute julian time
    'plane_names' : image_metadata['plane_names'] # list of plane names
    'channels': cp_dict, # dictionary of channel locations, in the case of Unet-based channel segmentation, it's a dictionary of channel labels

    Called by
    mm3_Compile.py __main__

    Calls
    mm3.extract_metadata
    mm3.find_channels
    """

    try:
        # open up file and get metadata
        with tiff.TiffFile(params["TIFF_dir"] / image_filename) as tif:
            image_data = tif.asarray()

            if params["TIFF_source"] == "TIFF_from_elements":
                image_metadata = get_tif_metadata_elements(tif)
            elif params["TIFF_source"] == "nd2":
                image_metadata = get_tif_metadata_nd2(tif)
            elif params["TIFF_source"] == "BioFormats":
                image_metadata = get_tif_metadata_filename(tif)

        # look for channels if flagged
        if find_channels:
            # fix the image orientation and get the number of planes
            image_data = fix_orientation(params, image_data)

            # if the image data has more than 1 plane restrict image_data to phase,
            # which should have highest mean pixel data
            if len(image_data.shape) > 2:
                # ph_index = np.argmax([np.mean(image_data[ci]) for ci in range(image_data.shape[0])])
                ph_index = int(params["phase_plane"][1:]) - 1
                image_data = image_data[ph_index]

            # get shape of single plane
            img_shape = [image_data.shape[0], image_data.shape[1]]

            # find channels on the processed image
            chnl_loc_dict = find_channel_locs(params, image_data)

        information("Analyzed %s" % image_filename)

        # return the file name, the data for the channels in that image, and the metadata
        return {
            "filepath": params["TIFF_dir"] / image_filename,
            "fov": image_metadata["fov"],  # fov id
            "t": image_metadata["t"],  # time point
            "jd": image_metadata["jd"],  # absolute julian time
            "planes": image_metadata["planes"],  # list of plane names
            "shape": img_shape,  # image shape x y in pixels
            # 'channels' : {1 : {'A' : 1, 'B' : 2}, 2 : {'C' : 3, 'D' : 4}}}
            "channels": chnl_loc_dict,
        }  # dictionary of channel locations

    except:
        warning(f"Failed get_params for {image_filename}")
        information(sys.exc_info()[0])
        information(sys.exc_info()[1])
        information(traceback.print_tb(sys.exc_info()[2]))
        return {
            "filepath": params["TIFF_dir"] / image_filename,
            "analyze_success": False,
        }


def get_tif_metadata_nd2(tif):
    """This function pulls out the metadata from a tif file and returns it as a dictionary.
    This if tiff files as exported by the mm3 function mm3_nd2.py. All the metdata
    is found in that script and saved in json format to the tiff, so it is simply extracted here

    Paramters:
        tif: TIFF file object from which data will be extracted
    Returns:
        dictionary of values:
            'fov': int,
            't' : int,
            'jdn' (float)
            'planes' (list of strings)

    Called by
    mm3_Compile.get_tif_params

    """
    # get the first page of the tiff and pull out image description
    # this dictionary should be in the above form

    for tag in tif.pages[0].tags:
        if tag.name == "ImageDescription":
            idata = tag.value
            break

    # print(idata)
    idata = json.loads(idata)
    return idata


def get_tif_metadata_filename(tif):
    """This function pulls out the metadata from a tif filename and returns it as a dictionary.
    This just gets the tiff metadata from the filename and is a backup option when the known format of the metadata is not known.

    Paramters:
        tif: TIFF file object from which data will be extracted
    Returns:
        dictionary of values:
            'fov': int,
            't' : int,
            'jdn' (float)

    Called by
    mm3_Compile.get_tif_params

    """
    idata = {
        "fov": get_fov(tif.filename),  # fov id
        "t": get_time(tif.filename),  # time point
        "jd": -1 * 0.0,  # absolute julian time
        "planes": get_plane(tif.filename),
    }  # y position on stage [um]

    return idata


def get_tif_metadata_elements(tif):
    """
    This function pulls out the metadata from a tif file which has been exported with Nikon Elements
    and returns it as a dictionary.
    This if tiff files as exported by Nikon Elements as a stacked tiff, each for one tpoint.
    tif is an opened tif file (using the package tifffile)


    arguments:
        fname (tifffile.TiffFile): TIFF file object from which data will be extracted
    returns:
        dictionary of values:
            'jdn' (float)
            'x' (float)
            'y' (float)
            'plane_names' (list of strings)

    Called by
    mm3.Compile

    """

    # image Metadata
    idata = {
        "fov": -1,
        "t": -1,
        "jd": -1 * 0.0,
        "planes": [],
    }

    # get the fov and t simply from the file name
    idata["fov"] = int(tif.fname.split("xy")[1].split(".tif")[0])
    idata["t"] = int(tif.fname.split("xy")[0].split("t")[-1])

    # a page is plane, or stack, in the tiff. The other metdata is hidden down in there.
    for page in tif:
        for tag in page.tags.values():
            # print("Checking tag",tag.name,tag.value)
            t = tag.name, tag.value
            t_string = ""
            time_string = ""
            # Interesting tag names: 65330, 65331 (binary data; good stuff), 65332
            # we wnat to work with the tag of the name 65331
            # if the tag name is not in the set of tegs we find interesting then skip this cycle of the loop
            if tag.name not in (
                "65331",
                "65332",
                "strip_byte_counts",
                "image_width",
                "orientation",
                "compression",
                "new_subfile_type",
                "fill_order",
                "max_sample_value",
                "bits_per_sample",
                "65328",
                "65333",
            ):
                # print("*** " + tag.name)
                # print(tag.value)
                pass
            # if tag.name == '65330':
            #    return tag.value
            if tag.name in ("65331"):
                # make info list a list of the tag values 0 to 65535 by zipoing up a paired list of two bytes, at two byte intervals i.e. ::2
                # note that 0X100 is hex for 256
                infolist = [
                    a + b * 0x100 for a, b in zip(tag.value[0::2], tag.value[1::2])
                ]
                # get char values for each element in infolist
                for c_entry in range(0, len(infolist)):
                    # the element corresponds to an ascii char for a letter or bracket (and a few other things)
                    if infolist[c_entry] < 127 and infolist[c_entry] > 64:
                        # add the letter to the unicode string t_string
                        t_string += chr(infolist[c_entry])
                    # elif infolist[c_entry] == 0:
                    #    continue
                    else:
                        t_string += " "

                # this block will find the dTimeAbsolute and print the subsequent integers
                # index 170 is counting seconds, and rollover of index 170 leads to increment of index 171
                # rollover of index 171 leads to increment of index 172
                # get the position of the array by finding the index of the t_string at which dTimeAbsolute is listed not that 2*len(dTimeAbsolute)=26
                # print(t_string)

                arraypos = t_string.index("dXPos") * 2 + 16
                xarr = tag.value[arraypos : arraypos + 4]
                b = "".join(chr(i) for i in xarr)
                idata["x"] = float(struct.unpack("<f", b)[0])

                arraypos = t_string.index("dYPos") * 2 + 16
                yarr = tag.value[arraypos : arraypos + 4]
                b = "".join(chr(i) for i in yarr)
                idata["y"] = float(struct.unpack("<f", b)[0])

                arraypos = t_string.index("dTimeAbsolute") * 2 + 26
                shortarray = tag.value[arraypos + 2 : arraypos + 10]
                b = "".join(chr(i) for i in shortarray)
                idata["jd"] = float(struct.unpack("<d", b)[0])

                # extract plane names
                il = [a + b * 0x100 for a, b in zip(tag.value[0::2], tag.value[1::2])]
                li = [a + b * 0x100 for a, b in zip(tag.value[1::2], tag.value[2::2])]

                strings = list(zip(il, li))

                allchars = ""
                for c_entry in range(0, len(strings)):
                    if 31 < strings[c_entry][0] < 127:
                        allchars += chr(strings[c_entry][0])
                    elif 31 < strings[c_entry][1] < 127:
                        allchars += chr(strings[c_entry][1])
                    else:
                        allchars += " "

                allchars = re.sub(" +", " ", allchars)

                words = allchars.split(" ")

                planes = []
                for idx in [
                    i for i, x in enumerate(words) if x == "sOpticalConfigName"
                ]:
                    planes.append(words[idx + 1])

                idata["planes"] = planes

    return idata


### Functions for dealing with cross-correlations, which are used to determine empty/full channels ###
# calculate cross correlation between pixels in channel stack
def channel_xcorr(params, fov_id, peak_id):
    """
    Function calculates the cross correlation of images in a
    stack to the first image in the stack. The output is an
    array that is the length of the stack with the best cross
    correlation between that image and the first image.

    The very first value should be 1.
    """

    pad_size = params["compile"]["alignment_pad"]

    # Use this number of images to calculate cross correlations
    number_of_images = 20

    # load the phase contrast images
    image_data = load_stack_params(
        params, fov_id, peak_id, postfix=params["phase_plane"]
    )

    # if there are more images than number_of_images, use number_of_images images evenly
    # spaced across the range
    if image_data.shape[0] > number_of_images:
        spacing = int(image_data.shape[0] / number_of_images)
        image_data = image_data[::spacing, :, :]
        if image_data.shape[0] > number_of_images:
            image_data = image_data[:number_of_images, :, :]

    # we will compare all images to this one, needs to be padded to account for image drift
    first_img = np.pad(image_data[0, :, :], pad_size, mode="reflect")

    xcorr_array = []  # array holds cross correlation vaues
    for img in image_data:
        # use match_template to find all cross correlations for the
        # current image against the first image.
        xcorr_array.append(np.max(match_template(first_img, img)))

    return xcorr_array


### functions about trimming, padding, and manipulating images
# cuts out channels from the image
def cut_slice(image_data, channel_loc):
    """Takes an image and cuts out the channel based on the slice location
    slice location is the list with the peak information, in the form
    [][y1, y2],[x1, x2]]. Returns the channel slice as a numpy array.
    The numpy array will be a stack if there are multiple planes.

    if you want to slice all the channels from a picture with the channel_masks
    dictionary use a loop like this:

    for channel_loc in channel_masks[fov_id]: # fov_id is the fov of the image
        channel_slice = cut_slice[image_pixel_data, channel_loc]
        # ... do something with the slice

    NOTE: this function will try to determine what the shape of your
    image is and slice accordingly. It expects the images are in the order
    [t, x, y, c]. It assumes images with three dimensions are [x, y, c] not
    [t, x, y].
    """

    # case where image is in form [x, y]
    if len(image_data.shape) == 2:
        # make slice object
        channel_slicer = np.s_[
            channel_loc[0][0] : channel_loc[0][1], channel_loc[1][0] : channel_loc[1][1]
        ]

    # case where image is in form [x, y, c]
    elif len(image_data.shape) == 3:
        channel_slicer = np.s_[
            channel_loc[0][0] : channel_loc[0][1],
            channel_loc[1][0] : channel_loc[1][1],
            :,
        ]

    # case where image in form [t, x , y, c]
    elif len(image_data.shape) == 4:
        channel_slicer = np.s_[
            :,
            channel_loc[0][0] : channel_loc[0][1],
            channel_loc[1][0] : channel_loc[1][1],
            :,
        ]

    # slice based on appropriate slicer object.
    channel_slice = image_data[channel_slicer]

    # pad y of channel if slice happened to be outside of image
    y_difference = (channel_loc[0][1] - channel_loc[0][0]) - channel_slice.shape[1]
    if y_difference > 0:
        paddings = [[0, 0], [0, y_difference], [0, 0], [0, 0]]  # t  # y  # x  # c
        channel_slice = np.pad(channel_slice, paddings, mode="edge")

    return channel_slice


# same thing as tiff_stack_slice_and_write but do it for hdf5
def hdf5_stack_slice_and_write(params, images_to_write, channel_masks, analyzed_imgs):
    """Writes out 4D stacks of TIFF images to an HDF5 file.

    Called by
    __main__
    """

    # make an array of images and then concatenate them into one big stack
    image_fov_stack = []

    # make arrays for filenames and times
    image_filenames = []
    image_times = []  # times is still an integer but may be indexed arbitrarily
    image_jds = []  # jds = julian dates (times)

    # go through list of images, load and fix them, and create arrays of metadata
    for n, image in enumerate(images_to_write):
        image_name = image[0]  # [0] is the key, [1] is jd

        # analyzed_imgs dictionary will be found in main scope.
        image_params = analyzed_imgs[image_name]
        information("Loading %s." % image_params["filepath"].name)

        # add information to metadata arrays
        image_filenames.append(image_name)
        image_times.append(image_params["t"])
        image_jds.append(image_params["jd"])

        # declare identification variables for saving using first image
        if n == 1:
            # same across fov
            fov_id = image_params["fov"]
            x_loc = image_params["x"]
            y_loc = image_params["y"]
            image_shape = image_params["shape"]
            image_planes = image_params["planes"]

        # load the tif and store it in array
        with tiff.TiffFile(image_params["filepath"]) as tif:
            image_data = tif.asarray()

        # channel finding was also done on images after orientation was fixed
        image_data = fix_orientation(params, image_data)

        # add additional axis if the image is flat
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, 0)

        # change axis so it goes X, Y, Plane
        image_data = np.rollaxis(image_data, 0, 3)

        # add it to list. The images should be in time order
        image_fov_stack.append(image_data)

    # concatenate the list into one big ass stack
    image_fov_stack = np.stack(image_fov_stack, axis=0)

    # create the HDF5 file for the FOV, first time this is being done.
    with h5py.File(
        params["hdf5_dir"] / ("xy%03d.hdf5" % fov_id), "w", libver="earliest"
    ) as h5f:

        # add in metadata for this FOV
        # these attributes should be common for all channel
        h5f.attrs.create("fov_id", fov_id)
        h5f.attrs.create("stage_x_loc", x_loc)
        h5f.attrs.create("stage_y_loc", y_loc)
        h5f.attrs.create("image_shape", image_shape)
        # encoding is because HDF5 has problems with numpy unicode
        h5f.attrs.create("planes", [plane.encode("utf8") for plane in image_planes])
        h5f.attrs.create("peaks", sorted(channel_masks[fov_id].keys()))

        # this is for things that change across time, for these create a dataset
        h5ds = h5f.create_dataset(
            "filenames",
            data=np.expand_dims(image_filenames, 1),
            chunks=True,
            maxshape=(None, 1),
            dtype="S100",
            compression="gzip",
            shuffle=True,
            fletcher32=True,
        )
        h5ds = h5f.create_dataset(
            "times",
            data=np.expand_dims(image_times, 1),
            chunks=True,
            maxshape=(None, 1),
            compression="gzip",
            shuffle=True,
            fletcher32=True,
        )
        h5ds = h5f.create_dataset(
            "times_jd",
            data=np.expand_dims(image_jds, 1),
            chunks=True,
            maxshape=(None, 1),
            compression="gzip",
            shuffle=True,
            fletcher32=True,
        )

        # cut out the channels as per channel masks for this fov
        for peak, channel_loc in six.iteritems(channel_masks[fov_id]):
            # information('Slicing and saving channel peak %s.' % channel_filename.split('/')[-1])
            information("Slicing and saving channel peak %d." % peak)

            # create group for this channel
            h5g = h5f.create_group("channel_%04d" % peak)

            # add attribute for peak_id, channel location
            h5g.attrs.create("peak_id", peak)
            h5g.attrs.create("channel_loc", channel_loc)

            # channel masks should only contain ints, but you can use this for a hard fix
            # for i in range(len(channel_loc)):
            #     for j in range(len(channel_loc[i])):
            #         channel_loc[i][j] = int(channel_loc[i][j])

            # slice out channel.
            # The function should recognize the shape length as 4 and cut all time points
            channel_stack = cut_slice(image_fov_stack, channel_loc)

            # save a different dataset for all colors
            for color_index in range(channel_stack.shape[3]):

                # create the dataset for the image. Review docs for these options.
                h5ds = h5g.create_dataset(
                    "p%04d_c%1d" % (peak, color_index + 1),
                    data=channel_stack[:, :, :, color_index],
                    chunks=(1, channel_stack.shape[1], channel_stack.shape[2]),
                    maxshape=(None, channel_stack.shape[1], channel_stack.shape[2]),
                    compression="gzip",
                    shuffle=True,
                    fletcher32=True,
                )

                # h5ds.attrs.create('plane', image_planes[color_index].encode('utf8'))

                # write the data even though we have more to write (free up memory)
                h5f.flush()

    return


# slice_and_write cuts up the image files one at a time and writes them out to tiff stacks
def tiff_stack_slice_and_write(params, images_to_write, channel_masks, analyzed_imgs):
    """Writes out 4D stacks of TIFF images per channel.
    Loads all tiffs from and FOV into memory and then slices all time points at once.

    Called by
    __main__
    """

    # make an array of images and then concatenate them into one big stack
    image_fov_stack = []

    # go through list of images and get the file path
    for n, image in enumerate(images_to_write):
        # analyzed_imgs dictionary will be found in main scope. [0] is the key, [1] is jd
        image_params = analyzed_imgs[image[0]]

        information("Loading %s." % image_params["filepath"].name)

        if n == 1:
            # declare identification variables for saving using first image
            fov_id = image_params["fov"]

        # load the tif and store it in array
        with tiff.TiffFile(image_params["filepath"]) as tif:
            image_data = tif.asarray()

        # channel finding was also done on images after orientation was fixed
        image_data = fix_orientation(params, image_data)

        # add additional axis if the image is flat
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, 0)

        # change axis so it goes Y, X, Plane
        image_data = np.rollaxis(image_data, 0, 3)

        # add it to list. The images should be in time order
        image_fov_stack.append(image_data)

    # concatenate the list into one big ass stack
    image_fov_stack = np.stack(image_fov_stack, axis=0)

    # cut out the channels as per channel masks for this fov
    for peak, channel_loc in six.iteritems(channel_masks[fov_id]):
        # information('Slicing and saving channel peak %s.' % channel_filename.split('/')[-1])
        information("Slicing and saving channel peak %d." % peak)

        # channel masks should only contain ints, but you can use this for hard fix
        # for i in range(len(channel_loc)):
        #     for j in range(len(channel_loc[i])):
        #         channel_loc[i][j] = int(channel_loc[i][j])

        # slice out channel.
        # The function should recognize the shape length as 4 and cut all time points
        channel_stack = cut_slice(image_fov_stack, channel_loc)

        # save a different time stack for all colors
        for color_index in range(channel_stack.shape[3]):
            # this is the filename for the channel
            # # chnl_dir and p will be looked for in the scope above (__main__)
            channel_filename = params["chnl_dir"] / (
                params["experiment_name"]
                + "_xy%03d_p%04d_c%1d.tif" % (fov_id, peak, color_index + 1)
            )
            # save stack
            tiff.imwrite(
                channel_filename,
                channel_stack[:, :, :, color_index],
                compression=("zlib", 4),
            )

    return


# make masks from initial set of images (same images as clusters)
def make_masks(params, analyzed_imgs):
    """
    Make masks goes through the channel locations in the image metadata and builds a consensus
    Mask for each image per fov, which it returns as dictionary named channel_masks.
    The keys in this dictionary are fov id, and the values is a another dictionary. This dict's keys are channel locations (peaks) and the values is a [2][2] array:
    [[minrow, maxrow],[mincol, maxcol]] of pixel locations designating the corner of each mask
    for each channel on the whole image

    One important consequence of these function is that the channel ids and the size of the
    channel slices are decided now. Updates to mask must coordinate with these values.

    Parameters
    analyzed_imgs : dict
        image information created by get_params

    Returns
    channel_masks : dict
        dictionary of consensus channel masks.

    Called By
    mm3_Compile.py

    Calls
    """
    information("Determining initial channel masks...")

    # declare temp variables from yaml parameter dict.
    crop_wp = int(
        params["compile"]["channel_width_pad"] + params["compile"]["channel_width"] / 2
    )
    chan_lp = int(params["compile"]["channel_length_pad"])

    # intiaize dictionary
    channel_masks = {}

    # get the size of the images (hope they are the same)
    for img_k in analyzed_imgs.keys():
        img_v = analyzed_imgs[img_k]
        image_rows = img_v["shape"][0]  # x pixels
        image_cols = img_v["shape"][1]  # y pixels
        break  # just need one. using iteritems mean the whole dict doesn't load

    # get the fov ids
    fovs = []
    for img_k in analyzed_imgs.keys():
        img_v = analyzed_imgs[img_k]
        if img_v["fov"] not in fovs:
            fovs.append(img_v["fov"])

    # max width and length across all fovs. channels will get expanded by these values
    # this important for later updates to the masks, which should be the same
    max_chnl_mask_len = 0
    max_chnl_mask_wid = 0

    # for each fov make a channel_mask dictionary from consensus mask
    for fov in fovs:
        # initialize a the dict and consensus mask
        channel_masks_1fov = (
            {}
        )  # dict which holds channel masks {peak : [[y1, y2],[x1,x2]],...}
        consensus_mask = np.zeros([image_rows, image_cols])  # mask for labeling

        # bring up information for each image
        for img_k in analyzed_imgs.keys():
            img_v = analyzed_imgs[img_k]
            # skip this one if it is not of the current fov
            if img_v["fov"] != fov:
                continue

            # for each channel in each image make a single mask
            img_chnl_mask = np.zeros([image_rows, image_cols])

            # and add the channel mask to it
            for chnl_peak, peak_ends in six.iteritems(img_v["channels"]):
                # pull out the peak location and top and bottom location
                # and expand by padding (more padding done later for width)
                x1 = max(chnl_peak - crop_wp, 0)
                x2 = min(chnl_peak + crop_wp, image_cols)
                y1 = max(peak_ends["closed_end_px"] - chan_lp, 0)
                y2 = min(peak_ends["open_end_px"] + chan_lp, image_rows)

                # add it to the mask for this image
                img_chnl_mask[y1:y2, x1:x2] = 1

            # add it to the consensus mask
            consensus_mask += img_chnl_mask

        # Normalize concensus mask between 0 and 1.
        consensus_mask = consensus_mask.astype("float32") / float(
            np.amax(consensus_mask)
        )

        # threshhold and homogenize each channel mask within the mask, label them
        # label when value is above 0.1 (so 90% occupancy), transpose.
        # the [0] is for the array ([1] is the number of regions)
        # It transposes and then transposes again so regions are labeled left to right
        # clear border it to make sure the channels are off the edge
        consensus_mask = ndi.label(consensus_mask)[0]

        # go through each label
        for label in np.unique(consensus_mask):
            if label == 0:  # label zero is the background
                continue
            binary_core = consensus_mask == label

            # clean up the rough edges
            poscols = np.any(binary_core, axis=0)  # column positions where true (any)
            posrows = np.any(binary_core, axis=1)  # row positions where true (any)

            # channel_id givin by horizontal position
            # this is important. later updates to the positions will have to check
            # if their channels contain this median value to match up
            channel_id = int(np.median(np.where(poscols)[0]))

            # store the edge locations of the channel mask in the dictionary. Will be ints
            min_row = np.min(np.where(posrows)[0])
            max_row = np.max(np.where(posrows)[0])
            min_col = np.min(np.where(poscols)[0])
            max_col = np.max(np.where(poscols)[0])

            # if the min/max cols are within the image bounds,
            # add the mask, as 4 points, to the dictionary
            if min_col > 0 and max_col < image_cols:
                channel_masks_1fov[channel_id] = [
                    [min_row, max_row],
                    [min_col, max_col],
                ]

                # find the largest channel width and height while you go round
                max_chnl_mask_len = int(max(max_chnl_mask_len, max_row - min_row))
                max_chnl_mask_wid = int(max(max_chnl_mask_wid, max_col - min_col))

        # add channel_mask dictionary to the fov dictionary, use copy to play it safe
        channel_masks[fov] = channel_masks_1fov.copy()

    # update all channel masks to be the max size
    cm_copy = channel_masks.copy()

    for fov, peaks in six.iteritems(channel_masks):
        # f_id = int(fov)
        for peak, chnl_mask in six.iteritems(peaks):
            # p_id = int(peak)
            # just add length to the open end (bottom of image, low column)
            if chnl_mask[0][1] - chnl_mask[0][0] != max_chnl_mask_len:
                cm_copy[fov][peak][0][1] = chnl_mask[0][0] + max_chnl_mask_len
            # enlarge widths around the middle, but make sure you don't get floats
            if chnl_mask[1][1] - chnl_mask[1][0] != max_chnl_mask_wid:
                wid_diff = max_chnl_mask_wid - (chnl_mask[1][1] - chnl_mask[1][0])
                if wid_diff % 2 == 0:
                    cm_copy[fov][peak][1][0] = max(chnl_mask[1][0] - wid_diff / 2, 0)
                    cm_copy[fov][peak][1][1] = min(
                        chnl_mask[1][1] + wid_diff / 2, image_cols - 1
                    )
                else:
                    cm_copy[fov][peak][1][0] = max(
                        chnl_mask[1][0] - (wid_diff - 1) / 2, 0
                    )
                    cm_copy[fov][peak][1][1] = min(
                        chnl_mask[1][1] + (wid_diff + 1) / 2, image_cols - 1
                    )

            # convert all values to ints
            chnl_mask[0][0] = int(chnl_mask[0][0])
            chnl_mask[0][1] = int(chnl_mask[0][1])
            chnl_mask[1][0] = int(chnl_mask[1][0])
            chnl_mask[1][1] = int(chnl_mask[1][1])

    # save the channel mask dictionary to a pickle and a text file
    # with open(os.path.join(params['ana_dir'], 'channel_masks.pkl'), 'wb') as cmask_file:
    #     pickle.dump(cm_copy, cmask_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(params["ana_dir"], "channel_masks.txt"), "w") as cmask_file:
        pprint(cm_copy, stream=cmask_file)
    with open(os.path.join(params["ana_dir"], "channel_masks.yaml"), "w") as cmask_file:
        yaml.dump(data=cm_copy, stream=cmask_file, default_flow_style=False, tags=None)

    information("Channel masks saved.")

    return cm_copy


# function for loading the channel masks
def load_channel_masks(params):
    """Load channel masks dictionary. Should be .yaml but try pickle too."""
    information("Loading channel masks dictionary.")

    # try loading from .yaml before .pkl
    try:
        information("Path:", os.path.join(params["ana_dir"], "channel_masks.yaml"))
        with open(
            os.path.join(params["ana_dir"], "channel_masks.yaml"), "r"
        ) as cmask_file:
            channel_masks = yaml.safe_load(cmask_file)
    except:
        warning("Could not load channel masks dictionary from .yaml.")

        try:
            information("Path:", os.path.join(params["ana_dir"], "channel_masks.pkl"))
            with open(
                os.path.join(params["ana_dir"], "channel_masks.pkl"), "rb"
            ) as cmask_file:
                channel_masks = pickle.load(cmask_file)
        except ValueError:
            warning("Could not load channel masks dictionary from .pkl.")

    return channel_masks


# make a lookup time table for converting nominal time to elapsed time in seconds
def make_time_table(params, analyzed_imgs):
    """
    Loops through the analyzed images and uses the jd time in the metadata to find the elapsed
    time in seconds that each picture was taken. This is later used for more accurate elongation
    rate calculation.

    Parametrs
    ---------
    analyzed_imgs : dict
        The output of get_tif_params.
    params['use_jd'] : boolean
        If set to True, 'jd' time will be used from the image metadata to use to create time table. Otherwise the 't' index will be used, and the parameter 'seconds_per_time_index' will be used from the parameters.yaml file to convert to seconds.

    Returns
    -------
    time_table : dict
        Look up dictionary with keys for the FOV and then the time point.
    """
    information("Making time table...")

    # initialize
    time_table = {}

    first_time = float("inf")

    # need to go through the data once to find the first time
    for iname, idata in six.iteritems(analyzed_imgs):
        if params["use_jd"]:
            try:
                if idata["jd"] < first_time:
                    first_time = idata["jd"]
            except:
                if idata["t"] < first_time:
                    first_time = idata["t"]
        else:
            if idata["t"] < first_time:
                first_time = idata["t"]

        # init dictionary for specific times per FOV
        if idata["fov"] not in time_table:
            time_table[idata["fov"]] = {}

    for iname, idata in six.iteritems(analyzed_imgs):
        if params["use_jd"]:
            # convert jd time to elapsed time in seconds
            try:
                t_in_seconds = np.around(
                    (idata["jd"] - first_time) * 24 * 60 * 60, decimals=0
                ).astype("uint32")
            except:
                t_in_seconds = np.around(
                    (idata["t"] - first_time) * params["seconds_per_time_index"],
                    decimals=0,
                ).astype("uint32")
        else:
            t_in_seconds = np.around(
                (idata["t"] - first_time) * params["seconds_per_time_index"], decimals=0
            ).astype("uint32")

        time_table[int(idata["fov"])][int(idata["t"])] = int(t_in_seconds)

    # save to .pkl. This pkl will be loaded into the params
    # with open(os.path.join(params['ana_dir'], 'time_table.pkl'), 'wb') as time_table_file:
    #     pickle.dump(time_table, time_table_file, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(params['ana_dir'], 'time_table.txt'), 'w') as time_table_file:
    #     pprint(time_table, stream=time_table_file)
    with open(
        os.path.join(params["ana_dir"], "time_table.yaml"), "w"
    ) as time_table_file:
        yaml.dump(
            data=time_table, stream=time_table_file, default_flow_style=False, tags=None
        )
    information("Time table saved.")

    return time_table


# finds the location of channels in a tif
def find_channel_locs(params, image_data):
    """Finds the location of channels from a phase contrast image. The channels are returned in
    a dictionary where the key is the x position of the channel in pixel and the value is a
    dicionary with the open and closed end in pixels in y.


    Called by
    mm3_Compile.get_tif_params

    """

    # declare temp variables from yaml parameter dict.
    chan_w = params["compile"]["channel_width"]
    chan_sep = params["compile"]["channel_separation"]
    crop_wp = int(params["compile"]["channel_width_pad"] + chan_w / 2)
    chan_snr = params["compile"]["channel_detection_snr"]

    # Detect peaks in the x projection (i.e. find the channels)
    projection_x = image_data.sum(axis=0).astype(np.int32)
    # find_peaks_cwt is a function which attempts to find the peaks in a 1-D array by
    # convolving it with a wave. here the wave is the default Mexican hat wave
    # but the minimum signal to noise ratio is specified
    # *** The range here should be a parameter or changed to a fraction.
    peaks = find_peaks_cwt(
        projection_x, np.arange(chan_w - 5, chan_w + 5), min_snr=chan_snr
    )

    # If the left-most peak position is within half of a channel separation,
    # discard the channel from the list.
    if peaks[0] < (chan_sep / 2):
        peaks = peaks[1:]
    # If the diference between the right-most peak position and the right edge
    # of the image is less than half of a channel separation, discard the channel.
    if image_data.shape[1] - peaks[-1] < (chan_sep / 2):
        peaks = peaks[:-1]

    # Find the average channel ends for the y-projected image
    projection_y = image_data.sum(axis=1)
    # find derivative, must use int32 because it was unsigned 16b before.
    proj_y_d = np.diff(projection_y.astype(np.int32))
    # use the top third to look for closed end, is pixel location of highest deriv
    onethirdpoint_y = int(projection_y.shape[0] / 3.0)
    default_closed_end_px = proj_y_d[:onethirdpoint_y].argmax()
    # use bottom third to look for open end, pixel location of lowest deriv
    twothirdpoint_y = int(projection_y.shape[0] * 2.0 / 3.0)
    default_open_end_px = twothirdpoint_y + proj_y_d[twothirdpoint_y:].argmin()
    default_length = default_open_end_px - default_closed_end_px  # used for checks

    # go through peaks and assign information
    # dict for channel dimensions
    chnl_loc_dict = {}
    # key is peak location, value is dict with {'closed_end_px': px, 'open_end_px': px}

    for peak in peaks:
        # set defaults
        chnl_loc_dict[peak] = {
            "closed_end_px": default_closed_end_px,
            "open_end_px": default_open_end_px,
        }
        # redo the previous y projection finding with just this channel
        channel_slice = image_data[:, peak - crop_wp : peak + crop_wp]
        slice_projection_y = channel_slice.sum(axis=1)
        slice_proj_y_d = np.diff(slice_projection_y.astype(np.int32))
        slice_closed_end_px = slice_proj_y_d[:onethirdpoint_y].argmax()
        slice_open_end_px = twothirdpoint_y + slice_proj_y_d[twothirdpoint_y:].argmin()
        slice_length = slice_open_end_px - slice_closed_end_px

        # check if these values make sense. If so, use them. If not, use default
        # make sure lenght is not 30 pixels bigger or smaller than default
        # *** This 15 should probably be a parameter or at least changed to a fraction.
        if slice_length + 15 < default_length or slice_length - 15 > default_length:
            continue
        # make sure ends are greater than 15 pixels from image edge
        if slice_closed_end_px < 15 or slice_open_end_px > image_data.shape[0] - 15:
            continue

        # if you made it to this point then update the entry
        chnl_loc_dict[peak] = {
            "closed_end_px": slice_closed_end_px,
            "open_end_px": slice_open_end_px,
        }

    return chnl_loc_dict


# define function for flipping the images on an FOV by FOV basis
def fix_orientation(params, image_data):
    """
    Fix the orientation. The standard direction for channels to open to is down.

    called by
    process_tif
    get_params
    """

    # user parameter indicates how things should be flipped
    image_orientation = params["compile"]["image_orientation"]

    # if this is just a phase image give in an extra layer so rest of code is fine
    flat = False  # flag for if the image is flat or multiple levels
    if len(image_data.shape) == 2:
        image_data = np.expand_dims(image_data, 0)
        flat = True

    # setting image_orientation to 'auto' will use autodetection
    if image_orientation == "auto":
        # use 'phase_plane' to find the phase plane in image_data, assuming c1, c2, c3... naming scheme here.
        try:
            ph_channel = int(re.search("[0-9]", params["phase_plane"]).group(0)) - 1
        except:
            # Pick the plane to analyze with the highest mean px value (should be phase)
            ph_channel = np.argmax(
                [np.mean(image_data[ci]) for ci in range(image_data.shape[0])]
            )

        # flip based on the index of the higest average row value
        # this should be closer to the opening
        if (
            np.argmax(image_data[ph_channel].mean(axis=1))
            < image_data[ph_channel].shape[0] / 2
        ):
            image_data = image_data[:, ::-1, :]
        else:
            pass  # no need to do anything

    # flip if up is chosen
    elif image_orientation == "up":
        return image_data[:, ::-1, :]

    # do not flip the images if "down is the specified image orientation"
    elif image_orientation == "down":
        pass

    if flat:
        image_data = image_data[0]  # just return that first layer

    return image_data


def compile(params):
    """mm3_Compile.py locates and slices out mother machine channels into image stacks."""

    # Load the project parameters file
    information("Loading experiment parameters.")
    p = params

    user_spec_fovs = params["FOV"]

    information("Using {} threads for multiprocessing.".format(p["num_analyzers"]))

    # only analyze images up until this t point. Put in None otherwise
    if "t_end" in p["compile"]:
        t_end = p["compile"]["t_end"]
        if t_end == "None":
            t_end = None
    else:
        t_end = None
    # only analyze images at and after this t point. Put in None otherwise
    if "t_start" in p["compile"]:
        t_start = p["compile"]["t_start"]
        if t_start == "None":
            t_start = None
    else:
        t_start = None

    # create the subfolders if they don't exist
    if not os.path.exists(p["ana_dir"]):
        os.makedirs(p["ana_dir"])
    if p["output"] == "TIFF":
        if not os.path.exists(p["chnl_dir"]):
            os.makedirs(p["chnl_dir"])
    elif p["output"] == "HDF5":
        if not os.path.exists(p["hdf5_dir"]):
            os.makedirs(p["hdf5_dir"])

    # declare information variables
    analyzed_imgs = {}  # for storing get_params pool results.

    ## need to stack phase and fl plane if not exported from .nd2
    if p["TIFF_source"] == "BioFormats":
        information("Checking if phase & fluorescence planes are separated")
        found_files = list(p["TIFF_dir"].glob("*.tif"))
        found_files = sorted(found_files)  # sort by timepoint

        string_c1 = re.compile("c1", re.IGNORECASE)
        string_c2 = re.compile("c2", re.IGNORECASE)

        ## if there is a second plane, stack and save them out
        if string_c2:
            information("Restacking TIFFs")
            found_files_c1 = [f for f in found_files if re.search(string_c1, f.name)]
            found_files_c2 = [f for f in found_files if re.search(string_c2, f.name)]

            found_files_c1 = sorted(found_files_c1)
            found_files_c2 = sorted(found_files_c2)

            for f1, f2 in zip(found_files_c1, found_files_c2):
                information("Merging images " + str(f1) + " and " + str(f2))
                im1 = tiff.imread(f1)
                im2 = tiff.imread(f2)
                im_out = np.stack((im1, im2), axis=0)
                name_out = str(f1).replace("C1", "")
                # 'minisblack' necessary to ensure that it interprets image as black/white.
                tiff.imwrite(name_out, im_out, photometric="minisblack")

                ## make a new directory rather than just deleting the old images
                old_tiff_path = p["TIFF_dir"].parent / "TIFF_unstacked"
                if not old_tiff_path.exists():
                    old_tiff_path.mkdir()
                    information("Creating directory for original TIFFs")
                f1.replace(str(f1).replace(str(p["TIFF_dir"]), "TIFF_unstacked"))
                f2.replace(str(f2).replace(str(p["TIFF_dir"]), "TIFF_unstacked"))
        else:
            pass

    ### process TIFFs for metadata #################################################################
    if not p["compile"]["do_metadata"]:
        information("Loading image parameters dictionary.")

        with open(
            os.path.join(p["ana_dir"], "TIFF_metadata.pkl"), "rb"
        ) as tiff_metadata:
            analyzed_imgs = pickle.load(tiff_metadata)

    else:
        information("Finding image parameters.")
        # get all the TIFFs in the folder
        found_files = p["TIFF_dir"].glob("*.tif")  # get all tiffs
        found_files = [filepath.name for filepath in found_files]  # remove pre-path
        found_files = sorted(found_files)  # should sort by timepoint

        # keep images starting at this timepoint
        if t_start is not None:
            information("Removing images before time {}".format(t_start))
            # go through list and find first place where timepoint is equivalent to t_start
            for n, ifile in enumerate(found_files):
                string = re.compile(
                    "t{:0=3}xy|t{:0=4}xy".format(t_start, t_start), re.IGNORECASE
                )  # account for 3 and 4 digit
                # if re.search == True then a match was found
                if re.search(string, ifile):
                    # cut off every file name prior to this one and quit the loop
                    found_files = found_files[n:]
                    break

        # remove images after this timepoint
        if t_end is not None:
            information("Removing images after time {}".format(t_end))
            # go through list and find first place where timepoint is equivalent to t_end
            for n, ifile in enumerate(found_files):
                string = re.compile(
                    "t%03dxy|t%04dxy" % (t_end, t_end), re.IGNORECASE
                )  # account for 3 and 4 digit
                if re.search(string, ifile):
                    found_files = found_files[:n]
                    break

        # if user has specified only certain FOVs, filter for those
        if len(user_spec_fovs) > 0:
            information("Filtering TIFFs by FOV.")
            fitered_files = []
            for fov_id in user_spec_fovs:
                fov_string = re.compile(
                    "xy%02d|xy%03d" % (fov_id, fov_id), re.IGNORECASE
                )
                fitered_files += [
                    ifile for ifile in found_files if re.search(fov_string, ifile)
                ]

            found_files = fitered_files[:]

        # get information for all these starting tiffs
        if len(found_files) > 0:
            information("Found %d image files." % len(found_files))
        else:
            warning("No TIFF files found")

        # initialize pool for analyzing image metadata
        pool = Pool(p["num_analyzers"])

        # loop over images and get information
        for fn in found_files:
            # get_params gets the image metadata and puts it in analyzed_imgs dictionary
            # for each file name. True means look for channels

            # This is the non-parallelized version (useful for debug)
            # analyzed_imgs[fn] = get_tif_params(params,fn, True)

            # # Parallelized
            analyzed_imgs[fn] = pool.apply_async(
                get_tif_params, args=(params, fn, True)
            )

        information("Waiting for image analysis pool to be finished.")

        pool.close()  # tells the process nothing more will be added.
        pool.join()  # blocks script until everything has been processed and workers exit

        information("Image analysis pool finished, getting results.")

        # get results from the pool and put them in a dictionary
        for fn in analyzed_imgs.keys():
            result = analyzed_imgs[fn]
            if result.successful():
                analyzed_imgs[
                    fn
                ] = result.get()  # put the metadata in the dict if it's good
            else:
                analyzed_imgs[fn] = False  # put a false there if it's bad

        # save metadata to a .pkl and a human readable txt file
        information("Saving metadata from analyzed images...")
        with open(
            os.path.join(p["ana_dir"], "TIFF_metadata.pkl"), "wb"
        ) as tiff_metadata:
            pickle.dump(analyzed_imgs, tiff_metadata, protocol=pickle.HIGHEST_PROTOCOL)
        with open(
            os.path.join(p["ana_dir"], "TIFF_metadata.txt"), "w"
        ) as tiff_metadata:
            pprint(analyzed_imgs, stream=tiff_metadata)
        information("Saved metadata from analyzed images.")

    ### Make table for jd time to FOV and time point
    if not p["compile"]["do_time_table"]:
        information("Skipping time table creation.")
    else:
        time_table = make_time_table(params, analyzed_imgs)

    ### Make consensus channel masks and get other shared metadata #################################
    if not p["compile"]["do_channel_masks"] and p["compile"]["do_slicing"]:
        channel_masks = load_channel_masks(params)

    elif p["compile"]["do_channel_masks"]:

        # only calculate channels masks from images before t_end in case it is specified
        if t_start:
            analyzed_imgs = {
                fn: i_metadata
                for fn, i_metadata in six.iteritems(analyzed_imgs)
                if i_metadata["t"] >= t_start
            }
        if t_end:
            analyzed_imgs = {
                fn: i_metadata
                for fn, i_metadata in six.iteritems(analyzed_imgs)
                if i_metadata["t"] <= t_end
            }

        # Uses channelinformation from the already processed image data
        channel_masks = make_masks(params, analyzed_imgs)

    ### Slice and write TIFF files into channels ###################################################
    if p["compile"]["do_slicing"]:

        information("Saving channel slices.")

        # do it by FOV. Not set up for multiprocessing
        for fov, peaks in six.iteritems(channel_masks):

            # skip fov if not in the group
            if user_spec_fovs and fov not in user_spec_fovs:
                continue

            information("Loading images for FOV %03d." % fov)

            # get filenames just for this fov along with the julian date of acquistion
            send_to_write = [
                [k, v["t"]] for k, v in six.iteritems(analyzed_imgs) if v["fov"] == fov
            ]

            # sort the filenames by jdn
            send_to_write = progress(sorted(send_to_write, key=lambda time: time[1]))

            if p["output"] == "TIFF":
                # This is for loading the whole raw tiff stack and then slicing through it
                tiff_stack_slice_and_write(
                    params, send_to_write, channel_masks, analyzed_imgs
                )

            elif p["output"] == "HDF5":
                # Or write it to hdf5
                hdf5_stack_slice_and_write(
                    params, send_to_write, channel_masks, analyzed_imgs
                )

        information("Channel slices saved.")

    ### Cross correlations ########################################################################
    if p["compile"]["do_crosscorrs"]:
        # a nested dict to hold cross corrs per channel per fov.
        crosscorrs = {}

        # for each fov find cross correlations (sending to pull)
        for fov_id in progress(user_spec_fovs):
            information("Calculating cross correlations for FOV %d." % fov_id)

            # nested dict keys are peak_ids and values are cross correlations
            crosscorrs[fov_id] = {}

            # initialize pool for analyzing image metadata
            pool = Pool(p["num_analyzers"])

            # find all peak ids in the current FOV
            for peak_id in sorted(channel_masks[fov_id].keys()):
                information("Calculating cross correlations for peak %d." % peak_id)

                # linear loop
                # crosscorrs[fov_id][peak_id] = channel_xcorr(params, fov_id, peak_id)

                # multiprocessing verion
                crosscorrs[fov_id][peak_id] = pool.apply_async(
                    channel_xcorr,
                    args=(
                        params,
                        fov_id,
                        peak_id,
                    ),
                )

            information(
                "Waiting for cross correlation pool to finish for FOV %d." % fov_id
            )

            pool.close()  # tells the process nothing more will be added.
            pool.join()  # blocks script until everything has been processed and workers exit

            information("Finished cross correlations for FOV %d." % fov_id)

        # get results from the pool and put the results in the dictionary if succesful
        for fov_id, peaks in six.iteritems(crosscorrs):
            for peak_id, result in six.iteritems(peaks):
                if result.successful():
                    # put the results, with the average, and a guess if the channel
                    # is full into the dictionary
                    crosscorrs[fov_id][peak_id] = {
                        "ccs": result.get(),
                        "cc_avg": np.average(result.get()),
                    }
                else:
                    crosscorrs[fov_id][peak_id] = False  # put a false there if it's bad

        # linear loop for debug
        # get results from the pool and put the results in the dictionary if succesful
        # for fov_id, peaks in six.iteritems(crosscorrs):
        #     for peak_id, result in six.iteritems(peaks):
        #         crosscorrs[fov_id][peak_id] = {'ccs' : result,
        #                                            'cc_avg' : np.average(result),
        #                                            'full' : np.average(result) < p['compile']['channel_picking_threshold']}

        # write cross-correlations to pickle and text
        information("Writing cross correlations file.")
        with open(os.path.join(p["ana_dir"], "crosscorrs.pkl"), "wb") as xcorrs_file:
            pickle.dump(crosscorrs, xcorrs_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(p["ana_dir"], "crosscorrs.txt"), "w") as xcorrs_file:
            pprint(crosscorrs, stream=xcorrs_file)
        information("Wrote cross correlations files.")

    # try to load previously calculated cross correlations
    else:
        information("Loading precalculated cross-correlations.")
        try:
            with open(
                os.path.join(p["ana_dir"], "crosscorrs.pkl"), "rb"
            ) as xcorrs_file:
                crosscorrs = pickle.load(xcorrs_file)
        except:
            crosscorrs = None
            information("Could not load cross-correlations.")


def load_fov(image_directory, fov_id):
    information("getting files")
    found_files = image_directory.glob("*.tif")
    file_string = re.compile(f"xy{fov_id:02d}.*.tif", re.IGNORECASE)
    found_files = [f.name for f in found_files if re.search(file_string, f.name)]

    information("sorting files")
    found_files = sorted(found_files)  # should sort by timepoint

    if len(found_files) == 0:
        information("No data found for FOV " + str(fov_id))
        return

    image_fov_stack = []

    information("Loading files")
    for img_filename in found_files:
        with tiff.TiffFile(image_directory / img_filename) as tif:
            image_fov_stack.append(tif.asarray())

    information("numpying files")
    return np.array(image_fov_stack)


class Compile(MM3Container):
    def __init__(self, napari_viewer: Viewer):
        super().__init__(napari_viewer=napari_viewer, validate_folders=False)

    def create_widgets(self):
        """Override method. Serves as the widget constructor. See MM3Container for more details."""
        self.viewer.text_overlay.visible = False

        self.fov_widget = FOVChooser(self.valid_fovs)
        # TODO: Auto-infer?
        self.image_source_widget = ComboBox(
            label="image source",
            choices=["nd2", "BioFormats", "TIFF_from_elements"],
        )
        self.phase_plane_widget = PlanePicker(
            self.valid_planes, label="phase plane channel"
        )
        self.time_range_widget = TimeRangeSelector(self.valid_times)
        self.seconds_per_frame_widget = SpinBox(
            value=150,
            label="seconds per frame",
            tooltip="Required if TIFF source is not .nd2. Time interval in seconds "
            + "between consecutive imaging rounds.",
            min=1,
            max=60 * 60 * 24,
        )
        self.channel_width_widget = SpinBox(
            value=10,
            label="channel width",
            tooltip="Required. Approx. width of traps in pixels.",
            min=1,
            max=10000,
        )
        self.channel_separation_widget = SpinBox(
            value=45,
            label="channel separation",
            tooltip="Required. Center-to-center distance between traps in pixels.",
            min=1,
            max=10000,
        )
        self.xcorr_threshold_widget = FloatSpinBox(
            label="cross correlation threshold",
            value=0.97,
            tooltip="Recommended. Threshold for designating channels as full /empty"
            + "based on time correlation of trap intensity profile. "
            + "Traps above threshold will be set as empty.",
            step=0.01,
            min=0,
            max=1,
        )

        self.inspect_widget = PushButton(text="visualize all FOVs (from .nd2)")

        self.fov_widget.connect_callback(self.set_fovs)
        self.image_source_widget.changed.connect(self.set_image_source)
        self.phase_plane_widget.changed.connect(self.set_phase_plane)
        self.time_range_widget.changed.connect(self.set_range)
        self.seconds_per_frame_widget.changed.connect(self.set_seconds_per_frame)
        self.channel_width_widget.changed.connect(self.set_channel_width)
        self.channel_separation_widget.changed.connect(self.set_channel_separation)
        self.inspect_widget.clicked.connect(self.display_all_fovs)

        self.append(self.fov_widget)
        self.append(self.image_source_widget)
        self.append(self.phase_plane_widget)
        self.append(self.time_range_widget)
        self.append(self.seconds_per_frame_widget)
        self.append(self.channel_width_widget)
        self.append(self.channel_separation_widget)
        self.append(self.inspect_widget)

        self.set_image_source()
        self.set_phase_plane()
        self.set_fovs(self.valid_fovs)
        self.set_range()
        self.set_seconds_per_frame()
        self.set_channel_width()
        self.set_channel_separation()

        self.display_single_fov()
        # self.render_images()

    def run(self):
        """Overriding method. Performs Mother Machine Analysis"""
        # global params. Ideally, this is rendered obsolete. However, old code uses this
        # Fixing it up would take a very long time, and as such is being deferred to later.
        params = {
            "experiment_name": self.experiment_name,
            "analysis_directory": self.analysis_folder,
            "FOV": self.fovs,
            "TIFF_source": self.image_source,
            "output": "TIFF",
            "phase_plane": self.phase_plane,
            "seconds_per_time_index": self.seconds_per_frame,
            "compile": {
                "do_metadata": True,
                "do_time_table": True,
                "do_channel_masks": True,
                "do_slicing": True,
                "t_start": self.time_range[0],
                "t_end": self.time_range[1] + 1,
                "image_orientation": "auto",
                "channel_width": self.channel_width,
                "channel_separation": self.channel_separation,
                "channel_detection_snr": 1,
                "channel_length_pad": 10,
                "channel_width_pad": 10,
                "do_crosscorrs": True,
                "alignment_pad": 10,
            },
            "num_analyzers": multiprocessing.cpu_count(),
            "TIFF_dir": self.TIFF_folder,
            "ana_dir": self.analysis_folder,
            "hdf5_dir": self.analysis_folder / "hdf5",
            "chnl_dir": self.analysis_folder / "channels",
            "empty_dir": self.analysis_folder / "empties",
            "sub_dir": self.analysis_folder / "subtracted",
            "seg_dir": self.analysis_folder / "segmented",
            "pred_dir": self.analysis_folder / "predictions",
            "cell_dir": self.analysis_folder / "cell_data",
            "track_dir": self.analysis_folder / "tracking",
            # use jd time in image metadata to make time table. Set to false if no jd time
            "use_jd": self.image_source in {"nd2", "TIFF_from_elements"},
        }
        self.viewer.window._status_bar._toggle_activity_dock(True)

        compile(params)
        information("Finished.")

    def display_single_fov(self):
        self.viewer.layers.clear()
        self.viewer.text_overlay.visible = False
        image_fov_stack = load_fov(self.TIFF_folder, min(self.valid_fovs))
        images = self.viewer.add_image(np.array(image_fov_stack))
        self.viewer.dims.current_step = (0, 0)
        images.reset_contrast_limits()
        # images.gamma = 0.5

    def display_all_fovs(self):
        viewer = self.viewer
        viewer.layers.clear()
        viewer.grid.enabled = True

        filepath = Path(".")
        nd2file = list(filepath.glob("*.nd2"))[0]

        if not nd2file:
            warning(
                f"Could not find .nd2 file to display in directory {filepath.resolve()}"
            )
            return

        with nd2reader.reader.ND2Reader(str(nd2file)) as ndx:
            sizes = ndx.sizes

            if "t" not in sizes:
                sizes["t"] = 1
            if "z" not in sizes:
                sizes["z"] = 1
            if "c" not in sizes:
                sizes["c"] = 1
            ndx.bundle_axes = "zcyx"
            ndx.iter_axes = "t"
            n = len(ndx)

            shape = (
                sizes["t"],
                sizes["z"],
                sizes["v"],
                sizes["c"],
                sizes["y"],
                sizes["x"],
            )
            image = np.zeros(shape, dtype=np.float32)

            for i in range(n):
                image[i] = ndx.get_frame(i)

        image = np.squeeze(image)

        viewer.add_image(image, channel_axis=1, colormap="gray")
        viewer.grid.shape = (-1, 3)

        viewer.dims.current_step = (0, 0)

    def set_image_source(self):
        self.image_source = self.image_source_widget.value

    def set_phase_plane(self):
        self.phase_plane = self.phase_plane_widget.value

    # NOTE! This is different from the other functions in that it requires a parameter.
    def set_fovs(self, new_fovs):
        self.fovs = list(
            set(new_fovs)
        )  # set(new_fovs).intersection(set(self.valid_fovs))

    def set_range(self):
        self.time_range = (
            self.time_range_widget.start.value,
            self.time_range_widget.stop.value,
        )

    def set_seconds_per_frame(self):
        self.seconds_per_frame = self.seconds_per_frame_widget.value

    def set_channel_width(self):
        self.channel_width = self.channel_width_widget.value

    def set_channel_separation(self):
        self.channel_separation = self.channel_separation_widget.value
