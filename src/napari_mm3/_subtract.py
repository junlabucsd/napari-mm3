from pathlib import Path
from skimage.feature import match_template
from multiprocessing.pool import Pool
from magicgui.widgets import SpinBox, ComboBox, CheckBox

import tifffile as tiff
import argparse
import numpy as np
import multiprocessing
import napari
import six

from .utils import TIFF_FILE_FORMAT_NO_PEAK, TIFF_FILE_FORMAT_PEAK

from ._deriving_widgets import (
    MM3Container,
    FOVChooser,
    PlanePicker,
    range_string_to_indices,
    get_valid_times,
    get_valid_fovs_folder,
    load_specs,
    information,
    warning,
    load_tiff,
)


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
    ana_dir: Path,
    experiment_name: str,
    empty_dir: Path,
    from_fov: int,
    to_fov: int,
    color="c1",
) -> None:
    """
    Copy an empty stack from one FOV to another.

    """

    # load empty stack from one FOV
    information(
        "Loading empty stack from FOV {} to save for FOV {}.".format(from_fov, to_fov)
    )
    empty_dir = ana_dir / "empties"
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
    ana_dir: Path,
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

    empty_dir = ana_dir / "empties"
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
        image_data = load_tiff(ana_dir / "channels" / image_filename)
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
    ana_dir: Path,
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
        avg_empty_stack = load_tiff(ana_dir / "channels" / avg_empty_stack_filename)

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
            image_data = load_tiff(ana_dir / "channels" / empty_stack_filename)
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


def subtract(
    ana_dir: Path,
    experiment_name: str,
    fovs: list,
    alignment_pad: int,
    num_analyzers: int,
    subtraction_plane: str,
    fluor_mode: bool,
    preview=False,
):
    """subtract averages empty channels and then subtracts them from channels with cells"""

    # Load the project parameters file
    information("Loading experiment parameters.")

    viewer = napari.current_viewer()
    viewer.layers.clear()
    viewer.grid.enabled = True
    # Set the shape better here.
    viewer.grid.shape = (-1, 20)

    user_spec_fovs = set(fovs)

    sub_plane = subtraction_plane
    empty_dir = ana_dir / "empties"
    sub_dir = ana_dir / "subtracted"
    # Create folders for subtracted info if they don't exist
    if not empty_dir.exists():
        empty_dir.mkdir()
    if not sub_dir.exists():
        sub_dir.mkdir()

    # load specs file
    specs = load_specs(ana_dir)

    # make list of FOVs to process (keys of specs file)
    fov_id_list = set(sorted(specs.keys()))

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list = fov_id_list.intersection(user_spec_fovs)

    information("Found %d FOVs to process." % len(fov_id_list))

    # determine if we are doing fluorescence or phase subtraction, and set flags
    align = True
    sub_method = "phase"
    if fluor_mode:
        align = False
        sub_method = "fluor"

    ### Make average empty channels ###############################################################
    information("Calculating averaged empties for channel {}.".format(sub_plane))

    need_empty = []  # list holds fov_ids of fov's that did not have empties
    for fov_id in fov_id_list:
        # send to function which will create empty stack for each fov.
        averaging_result = average_empties_stack(
            ana_dir,
            experiment_name,
            empty_dir,
            fov_id,
            specs,
            alignment_pad,
            color=sub_plane,
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
        copy_result = copy_empty_stack(
            ana_dir, experiment_name, empty_dir, from_fov, fov_id, color=sub_plane
        )

    ### Subtract ###########
    information("Subtracting channels for channel {}.".format(sub_plane))
    for fov_id in fov_id_list:
        # send to function which will create empty stack for each fov.
        subtract_fov_stack(
            ana_dir,
            experiment_name,
            alignment_pad,
            num_analyzers,
            sub_dir,
            fov_id,
            specs,
            color=sub_plane,
            method=sub_method,
            preview=preview,
        )
    information("Finished subtraction.")

    viewer = napari.current_viewer()


class Subtract(MM3Container):
    def create_widgets(self):
        """Overriding method. Serves as the widget constructor. See MM3Container for more details."""
        self.viewer.text_overlay.visible = False
        self.fov_widget = FOVChooser(self.valid_fovs)
        self.alignment_pad_widget = SpinBox(
            label="alignment pad",
            value=10,
            min=0,
            tooltip="Required. Padding for images. Larger => slower, but also larger => more tolerant of size differences between template and comparison image.",
        )
        self.mode_widget = ComboBox(
            choices=[
                "phase",
                "fluorescence",
            ],
            label="subtraction mode",
        )
        self.subtraction_plane_widget = PlanePicker(
            self.valid_planes, label="subtraction plane"
        )
        self.output_display_widget = CheckBox(label="display output")

        self.fov_widget.connect_callback(self.set_fovs)
        self.alignment_pad_widget.changed.connect(self.set_alignment_pad)
        self.subtraction_plane_widget.changed.connect(self.set_subtraction_plane)
        self.mode_widget.changed.connect(self.set_mode)

        self.append(self.fov_widget)
        self.append(self.alignment_pad_widget)
        self.append(self.mode_widget)
        self.append(self.subtraction_plane_widget)
        self.append(self.output_display_widget)

        self.set_fovs(self.valid_fovs)
        self.set_alignment_pad()
        self.set_mode()
        self.set_subtraction_plane()
        self.set_view_result()

    def run(self):
        """Overriding method. Perform mother machine analysis."""
        subtract(
            ana_dir=self.analysis_folder,
            experiment_name=self.experiment_name,
            fovs=self.fovs,
            alignment_pad=self.alignment_pad,
            num_analyzers=multiprocessing.cpu_count(),
            subtraction_plane=self.subtraction_plane,
            fluor_mode=self.fluor_mode,
            preview=self.view_result,
        )

    def set_fovs(self, fovs):
        self.fovs = fovs

    def set_alignment_pad(self):
        self.alignment_pad = self.alignment_pad_widget.value

    def set_subtraction_plane(self):
        self.subtraction_plane = self.subtraction_plane_widget.value

    def set_mode(self):
        self.fluor_mode = self.mode_widget.value == "fluorescence"

    def set_view_result(self):
        self.view_result = self.output_display_widget.value


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

    subtract(
        ana_dir=cur_dir / "analysis",
        experiment_name="",
        fovs=fovs,
        alignment_pad=10,
        num_analyzers=1,  # the assumption is you're running headless to debug
        subtraction_plane="c1",
        fluor_mode=False,
    )
