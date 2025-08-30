import multiprocessing
from dataclasses import dataclass
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Annotated

import napari
import numpy as np
import six
import tifffile as tiff
from napari import Viewer
from skimage.feature import match_template

from ._deriving_widgets import (
    FOVList,
    MM3Container2,
    get_valid_fovs_folder,
    get_valid_planes,
    information,
    load_specs,
    load_tiff,
    warning,
)
from .utils import TIFF_FILE_FORMAT_NO_PEAK, TIFF_FILE_FORMAT_PEAK


def subtract_phase(
    alignment_pad: int, cropped_channel: np.ndarray, empty_channel: np.ndarray
) -> np.ndarray:
    """subtract_phase aligns and subtracts an empty phase contrast channel (trap) from a channel containing cells.
    The subtracted image returned is the same size as the image given. It may however include
    data points around the edge that are meaningless but not marked.
    """

    # this is for aligning the empty channel to the cell channel.
    ### Pad cropped channel.
    pad_size = alignment_pad  # pixel size to use for padding (amount that alignment could be off)
    padded_chnl = np.pad(cropped_channel, pad_size, mode="reflect")

    # ### Align channel to empty using match template.
    # use match template to get a correlation array and find the position of maximum overlap
    try:
        match_result = match_template(padded_chnl, empty_channel)
    except:
        information(
            "match_template failed. This is likely due to cropping issues with the image of the channel containing bacteria."
        )
        information(
            "Consider marking this channel as disabled in specs.yaml, or increasing the pad_size."
        )
        raise
    # get row and column of max correlation value in correlation array
    y, x = np.unravel_index(np.argmax(match_result), match_result.shape)

    # pad the empty channel according to alignment to be overlaid on padded channel.
    empty_paddings = [
        [y, padded_chnl.shape[0] - (y + empty_channel.shape[0])],
        [x, padded_chnl.shape[1] - (x + empty_channel.shape[1])],
    ]
    aligned_empty = np.pad(empty_channel, empty_paddings, mode="reflect")
    # now trim it off so it is the same size as the original channel
    aligned_empty = aligned_empty[pad_size : -1 * pad_size, pad_size : -1 * pad_size]

    ### Compute the difference between the empty and channel phase contrast images
    # subtract cropped cell image from empty channel.
    channel_subtracted = aligned_empty.astype("int32") - cropped_channel.astype("int32")
    # channel_subtracted = cropped_channel.astype('int32') - aligned_empty.astype('int32')

    # just zero out anything less than 0. This is what Sattar does
    channel_subtracted[channel_subtracted < 0] = 0
    channel_subtracted = channel_subtracted.astype("uint16")  # change back to 16bit

    return channel_subtracted


def subtract_phase_helper(all_args):
    return subtract_phase(*all_args)


def subtract_fluor(
    cropped_channel: np.ndarray, empty_channel: np.ndarray
) -> np.ndarray:
    """subtract_fluor does a simple subtraction of one image to another. Unlike subtract_phase,
    there is no alignment. Also, the empty channel is subtracted from the full channel.

    Parameters
    image_pair : tuple of length two with; (image, empty_mean)

    Returns
    channel_subtracted : np.array
        The subtracted image.
    """

    # check frame size of cropped channel and background, always keep crop channel size the same
    crop_size = np.shape(cropped_channel)[:2]
    empty_size = np.shape(empty_channel)[:2]
    if crop_size != empty_size:
        if crop_size[0] > empty_size[0] or crop_size[1] > empty_size[1]:
            pad_row_length = max(crop_size[0] - empty_size[0], 0)  # prevent negatives
            pad_column_length = max(crop_size[1] - empty_size[1], 0)
            empty_channel = np.pad(
                empty_channel,
                [
                    [
                        np.int(0.5 * pad_row_length),
                        pad_row_length - np.int(0.5 * pad_row_length),
                    ],
                    [
                        np.int(0.5 * pad_column_length),
                        pad_column_length - np.int(0.5 * pad_column_length),
                    ],
                    [0, 0],
                ],
                "edge",
            )
        empty_size = np.shape(empty_channel)[:2]
        if crop_size[0] < empty_size[0] or crop_size[1] < empty_size[1]:
            empty_channel = empty_channel[
                : crop_size[0],
                : crop_size[1],
            ]

    ### Compute the difference between the empty and channel phase contrast images
    # subtract cropped cell image from empty channel.
    channel_subtracted = cropped_channel.astype("int32") - empty_channel.astype("int32")
    # channel_subtracted = cropped_channel.astype('int32') - aligned_empty.astype('int32')

    # just zero out anything less than 0.
    channel_subtracted[channel_subtracted < 0] = 0
    channel_subtracted = channel_subtracted.astype("uint16")  # change back to 16bit
    return channel_subtracted


def subtract_fluor_helper(all_args):
    return subtract_fluor(*all_args)


# this function is used when one FOV doesn't have an empty
def copy_empty_stack(
    empty_dir: Path,
    experiment_name: str,
    from_fov: int,
    to_fov: int,
    color="c1",
):
    """
    Copy an empty stack from one FOV to another.

    """

    # load empty stack from one FOV
    information(
        "Loading empty stack from FOV {} to save for FOV {}.".format(from_fov, to_fov)
    )
    empty_filename = TIFF_FILE_FORMAT_NO_PEAK % (
        experiment_name,
        from_fov,
        f"empty_{color}",
    )
    avg_empty_stack = load_tiff(empty_dir / empty_filename)

    # save out data
    # make new name and save it
    empty_filename = experiment_name + "_xy%03d_empty_%s.tif" % (
        to_fov,
        color,
    )
    tiff.imwrite(empty_dir / empty_filename, avg_empty_stack, compression=("zlib", 4))

    information("Saved empty channel for FOV %d." % to_fov)


# Do subtraction for an fov over many timepoints
def subtract_fov_stack(
    empty_dir: Path,
    channel_dir: Path,
    experiment_name: str,
    alignment_pad: int,
    num_analyzers: int,
    sub_dir: Path,
    fov_id: int,
    specs: dict,
    color: str = "c1",
    method: str = "phase",
    preview: bool = False,
) -> bool:
    """
    For a given FOV, loads the precomputed empty stack and does subtraction on
    all peaks in the FOV designated to be analyzed
    """

    information("Subtracting peaks for FOV %d." % fov_id)

    empty_filename = TIFF_FILE_FORMAT_NO_PEAK % (
        experiment_name,
        fov_id,
        f"empty_{color}",
    )
    avg_empty_stack = load_tiff(empty_dir / empty_filename)

    # determine which peaks are to be analyzed
    ana_peak_ids = []
    for peak_id, spec in six.iteritems(specs[fov_id]):
        if spec == 1:  # 0 means it should be used for empty, -1 is ignore
            ana_peak_ids.append(peak_id)
    ana_peak_ids = sorted(ana_peak_ids)  # sort for repeatability
    information("Subtracting %d channels for FOV %d." % (len(ana_peak_ids), fov_id))

    # just break if there are to peaks to analyze
    if not ana_peak_ids:
        return False

    # load images for the peak and get phase images'
    for peak_id in ana_peak_ids:
        information("Subtracting peak %d." % peak_id)
        image_filename = TIFF_FILE_FORMAT_PEAK % (
            experiment_name,
            fov_id,
            peak_id,
            color,
        )
        image_data = load_tiff(channel_dir / image_filename)
        # make a list for all time points to send to a multiprocessing pool
        # list will length of image_data with tuples (image, empty)
        # # set up multiprocessing pool to do subtraction. Should wait until finished
        pool = Pool(processes=num_analyzers)
        subtract_pairs = zip(image_data, avg_empty_stack)

        # set up multiprocessing pool to do subtraction. Should wait until finished
        pool = Pool(processes=num_analyzers)

        if method == "phase":
            subtract_phase_args = [
                (alignment_pad, pair[0], pair[1]) for pair in subtract_pairs
            ]
            subtracted_imgs = pool.map(
                subtract_phase_helper, subtract_phase_args, chunksize=10
            )
        elif method == "fluor":
            subtract_fl_args = [(pair[0], pair[1]) for pair in subtract_pairs]
            subtracted_imgs = pool.map(
                subtract_fluor_helper, subtract_fl_args, chunksize=10
            )

        pool.close()  # tells the process nothing more will be added.
        pool.join()  # blocks script until everything has been processed and workers exit

        # linear loop for debug
        # # stack them up along a time axis
        subtracted_stack = np.stack(subtracted_imgs, axis=0)

        # save out the subtracted stack
        sub_filename = experiment_name + "_xy%03d_p%04d_sub_%s.tif" % (
            fov_id,
            peak_id,
            color,
        )
        # TODO: Make this respect compression levels
        tiff.imwrite(
            sub_dir / sub_filename, subtracted_stack, compression=("zlib", 4)
        )  # save it

        if preview:
            napari.current_viewer().add_image(
                subtracted_stack,
                name="Subtracted" + "_xy%03d_p%04d" % (fov_id, peak_id),
                visible=True,
            )

        information("Saved subtracted channel %d." % peak_id)

    return True


# averages a list of empty channels
def average_empties(alignment_pad: int, imgs: list, align: bool = True) -> np.ndarray:
    """
    This function averages a set of images (empty channels) and returns a single image
    of the same size. It first aligns the images to the first image before averaging.

    Alignment is done by enlarging the first image using edge padding.
    Subsequent images are then aligned to this image and the offset recorded.
    These images are padded such that they are the same size as the first (padded) image but
    with the image in the correct (aligned) place. Edge padding is again used.
    The images are then placed in a stack and averaged. This image is trimmed so it is the size
    of the original images

    Called by
    average_empties_stack
    """

    aligned_imgs = []  # list contains the aligned, padded images

    if align:
        # pixel size to use for padding (amount that alignment could be off)
        pad_size = alignment_pad

        for n, img in enumerate(imgs):
            # if this is the first image, pad it and add it to the stack
            if n == 0:
                ref_img = np.pad(
                    img, pad_size, mode="reflect"
                )  # padded reference image
                aligned_imgs.append(ref_img)

            # otherwise align this image to the first padded image
            else:
                # find correlation between a convolution of img against the padded reference
                match_result = match_template(ref_img, img)

                # find index of highest correlation (relative to top left corner of img)
                y, x = np.unravel_index(np.argmax(match_result), match_result.shape)

                # pad img so it aligns and is the same size as reference image
                pad_img = np.pad(
                    img,
                    (
                        (y, ref_img.shape[0] - (y + img.shape[0])),
                        (x, ref_img.shape[1] - (x + img.shape[1])),
                    ),
                    mode="reflect",
                )
                aligned_imgs.append(pad_img)
    else:
        # don't align, just link the names to go forward easily
        aligned_imgs = imgs

    # stack the aligned data along 3rd axis
    aligned_imgs = np.dstack(aligned_imgs)
    # get a mean image along 3rd axis
    avg_empty = np.nanmean(aligned_imgs, axis=2)
    # trim off the padded edges (only if images were aligned, otherwise there was no padding)
    if align:
        avg_empty = avg_empty[pad_size : -1 * pad_size, pad_size : -1 * pad_size]
    # change type back to unsigned 16 bit not floats
    avg_empty = avg_empty.astype(dtype="uint16")

    return avg_empty


# average empty channels from stacks, making another TIFF stack
def average_empties_stack(
    channels_dir: Path,
    experiment_name: str,
    empty_dir: Path,
    fov_id: int,
    specs: dict,  # specifies if a channel is analyzed (1), ignored (-1), or reference empty (0)
    alignment_pad: int,
    color: str = "c1",
    align: bool = True,
) -> bool:
    """Takes the fov file name and the peak names of the designated empties,
    averages them and saves the image

    Saves empty stack to analysis folder, True upon success.
    """

    information("Creating average empty channel for FOV %d." % fov_id)

    # get peak ids of empty channels for this fov
    empty_peak_ids = []
    for peak_id, spec in six.iteritems(specs[fov_id]):
        if spec == 0:  # 0 means it should be used for empty
            empty_peak_ids.append(peak_id)
    empty_peak_ids = sorted(empty_peak_ids)  # sort for repeatability

    # depending on how many empties there are choose what to do
    # if there is no empty the user is going to have to copy another empty stack
    if len(empty_peak_ids) == 0:
        information("No empty channel designated for FOV %d." % fov_id)
        return False

    # if there is just one then you can just copy that channel
    elif len(empty_peak_ids) == 1:
        peak_id = empty_peak_ids[0]
        information("One empty channel (%d) designated for FOV %d." % (peak_id, fov_id))
        avg_empty_stack_filename = TIFF_FILE_FORMAT_PEAK % (
            experiment_name,
            fov_id,
            peak_id,
            color,
        )
        avg_empty_stack = load_tiff(channels_dir / avg_empty_stack_filename)

    # but if there is more than one empty you need to align and average them per timepoint
    elif len(empty_peak_ids) > 1:
        # load the image stacks into memory
        empty_stacks = []  # list which holds phase image stacks of designated empties
        for peak_id in empty_peak_ids:
            # load data and append to list
            empty_stack_filename = TIFF_FILE_FORMAT_PEAK % (
                experiment_name,
                fov_id,
                peak_id,
                color,
            )
            image_data = load_tiff(channels_dir / empty_stack_filename)
            empty_stacks.append(image_data)

        information(
            "%d empty channels designated for FOV %d." % (len(empty_stacks), fov_id)
        )

        # go through time points and create list of averaged empties
        avg_empty_stack = []  # list will be later concatenated into numpy array
        time_points = range(image_data.shape[0])  # index is time
        for t in time_points:
            # get images from one timepoint at a time and send to alignment and averaging
            imgs = [stack[t] for stack in empty_stacks]
            avg_empty = average_empties(alignment_pad, imgs, align=align)
            avg_empty_stack.append(avg_empty)

        # concatenate list and then save out to tiff stack
        avg_empty_stack = np.stack(avg_empty_stack, axis=0)

    # save out data
    # make new name and save it
    empty_filename = experiment_name + "_xy%03d_empty_%s.tif" % (
        fov_id,
        color,
    )
    tiff.imwrite(empty_dir / empty_filename, avg_empty_stack, compression=("zlib", 4))

    information("Saved empty channel for FOV %d." % fov_id)

    return True


@dataclass
class InPaths:
    """
    1. check folders for existence, fetch FOVs & times & planes
        -> upon failure, simply show a list of inputs + update button.
    """

    channels_folder: Annotated[Path, {"mode": "d"}] = (
        Path(".") / "analysis" / "channels"
    )
    specs_file: Path = Path("./analysis/specs.yaml")
    empty_folder: Annotated[Path, {"mode": "d"}] = Path("./analysis/empties")


@dataclass
class OutPaths:
    experiment_name: str = ""
    subtracted_dir: Annotated[Path, {"mode": "d"}] = Path("./analysis/subtracted")


@dataclass
class RunParams:
    FOVs: FOVList
    subtraction_plane: str
    alignment_pad: Annotated[
        int,
        {
            "tooltip": "Required. Padding for images. Larger => slower, but also larger => more tolerant of size differences between template and comparison image."
        },
    ] = 10
    num_analyzers: int = multiprocessing.cpu_count()
    fluor_mode: Annotated[str, {"choices": ["phase", "fluorescence"]}] = "phase"
    preview: bool = True


def gen_default_run_params(in_files: InPaths):
    """Initializes RunParams from a given in_files.
    Probably better to do this in __new__, but..."""
    # TODO: combine all planes into one click
    try:
        all_fovs = get_valid_fovs_folder(in_files.channels_folder)
        # TODO: get the brightest channel as the default phase plane!
        channels = get_valid_planes(in_files.channels_folder)
        # move this into runparams somehow!
        params = RunParams(
            subtraction_plane=channels[0],
            FOVs=FOVList(all_fovs),
        )
        params.__annotations__["subtraction_plane"] = Annotated[
            str, {"choices": channels}
        ]
        return params
    except FileNotFoundError:
        raise FileNotFoundError("TIFF folder not found")
    except ValueError:
        raise ValueError(
            "Invalid filenames. Make sure that timestamps are denoted as t[0-9]* and FOVs as xy[0-9]*"
        )


def subtract(in_paths: InPaths, run_params: RunParams, out_paths: OutPaths):
    """subtract averages empty channels and then subtracts them from channels with cells"""

    viewer = napari.current_viewer()
    viewer.layers.clear()
    viewer.grid.enabled = True
    # Set the shape better here.
    viewer.grid.shape = (-1, 20)

    user_spec_fovs = set(run_params.FOVs)

    if not in_paths.empty_folder.exists():
        in_paths.empty_folder.mkdir()
    if not out_paths.subtracted_dir.exists():
        out_paths.subtracted_dir.mkdir()

    specs = load_specs(in_paths.specs_file)

    # make list of FOVs to process (keys of specs file)
    fov_id_list = set(sorted(specs.keys()))
    if user_spec_fovs:
        fov_id_list = fov_id_list.intersection(user_spec_fovs)

    information("Found %d FOVs to process." % len(fov_id_list))

    # determine if we are doing fluorescence or phase subtraction, and set flags
    align = True
    sub_method = "phase" if run_params.fluor_mode == "phase" else "fluor"

    information(
        "Calculating averaged empties for channel {}.".format(
            run_params.subtraction_plane
        )
    )

    need_empty = []  # list holds fov_ids of fov's that did not have empties
    for fov_id in fov_id_list:
        # send to function which will create empty stack for each fov.
        averaging_result = average_empties_stack(
            in_paths.channels_folder,
            out_paths.experiment_name,
            in_paths.empty_folder,
            fov_id,
            specs,
            run_params.alignment_pad,
            color=run_params.subtraction_plane,
            align=align,
        )
        # add to list for FOVs that need to be given empties from other FOVs
        if not averaging_result:
            need_empty.append(fov_id)

    # deal with those problem FOVs without empties
    have_empty = list(fov_id_list.difference(set(need_empty)))  # fovs with empties
    if not have_empty:
        warning("No empty channels found. Return to channel selection")
        return

    for fov_id in need_empty:
        from_fov = min(
            have_empty, key=lambda x: abs(x - fov_id)
        )  # find closest FOV with an empty
        _ = copy_empty_stack(
            in_paths.empty_folder,
            out_paths.experiment_name,
            from_fov,
            fov_id,
            color=run_params.subtraction_plane,
        )

    ### Subtract ###########
    information(
        "Subtracting channels for channel {}.".format(run_params.subtraction_plane)
    )
    for fov_id in fov_id_list:
        # send to function which will create empty stack for each fov.
        subtract_fov_stack(
            in_paths.empty_folder,
            in_paths.channels_folder,
            out_paths.experiment_name,
            run_params.alignment_pad,
            run_params.num_analyzers,
            out_paths.subtracted_dir,
            fov_id,
            specs,
            color=run_params.subtraction_plane,
            method=sub_method,
            preview=run_params.preview,
        )
    information("Finished subtraction.")

    viewer = napari.current_viewer()


class Subtract(MM3Container2):
    def __init__(self, viewer: Viewer):
        super().__init__(viewer)
        self.viewer = viewer
        self.in_paths = InPaths()
        try:
            self.run_params = gen_default_run_params(self.in_paths)
            self.out_paths = OutPaths()
            self.initialized = True
            self.regen_widgets()
        except FileNotFoundError | ValueError:
            self.initialized = False
            self.regen_widgets()

    def run(self):
        print(self.run_params)
        subtract(self.in_paths, self.run_params, self.out_paths)


if __name__ == "__main__":
    in_paths = InPaths()
    run_params: RunParams = gen_default_run_params(in_paths)
    out_paths = OutPaths()
    subtract(in_paths, run_params, out_paths)
