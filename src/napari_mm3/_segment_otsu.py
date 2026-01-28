import multiprocessing
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import napari
import numpy as np
import six
import tifffile as tiff
from magicgui.widgets import Container, PushButton
from napari import Viewer
from napari.qt.threading import thread_worker
from napari.utils import progress
from scipy import ndimage as ndi
from skimage import morphology, segmentation
from skimage.filters import threshold_otsu

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


# Do segmentation for a channel time stack
def segment_chnl_stack(
    subtracted_dir,
    experiment_name,
    phase_plane,
    seg_dir,  # The directory where the segmented images will be saved.
    num_analyzers,
    fov_id,
    peak_id,
    OTSU_threshold,
    first_opening_size,
    distance_threshold,
    second_opening_size,
    min_object_size,
    display_result: bool = False,
) -> bool:
    """
    For a given fov and peak (channel), do segmentation for all images in the
    subtracted .tif stack.

    Returns
    -------
    bool
        True if the segmentation was successful.
    """

    information("Segmenting FOV %d, channel %d." % (fov_id, peak_id))

    # load subtracted images
    # sub_stack = load_stack_params(
    #    params, fov_id, peak_id, postfix="sub_{}".format(params["phase_plane"])
    # )
    sub_filename = TIFF_FILE_FORMAT_PEAK % (
        experiment_name,
        fov_id,
        peak_id,
        f"sub_{phase_plane}",
    )
    sub_stack = load_tiff(subtracted_dir / sub_filename)

    # # set up multiprocessing pool to do segmentation. Will do everything before going on.
    # pool = Pool(processes=params['num_analyzers'])

    # # send the 3d array to multiprocessing
    # segmented_imgs = pool.map(segment_image, sub_stack, chunksize=8)

    # pool.close() # tells the process nothing more will be added.
    # pool.join() # blocks script until everything has been processed and workers exit

    # image by image for debug
    segmented_imgs = []
    for sub_image in sub_stack:
        segmented_imgs.append(
            segment_image(
                OTSU_threshold,
                first_opening_size,
                distance_threshold,
                second_opening_size,
                min_object_size,
                sub_image,
            )
        )

    # stack them up along a time axis
    segmented_imgs = np.stack(segmented_imgs, axis=0)
    segmented_imgs = segmented_imgs.astype("uint8")

    # save out the segmented stack
    seg_filename = experiment_name + "_xy%03d_p%04d_seg_otsu.tif" % (
        fov_id,
        peak_id,
    )
    tiff.imwrite(
        os.path.join(seg_dir, seg_filename),
        segmented_imgs,
        compression="zlib",
    )
    if display_result:
        viewer = napari.current_viewer()

        viewer.grid.enabled = True

        viewer.add_labels(
            segmented_imgs,
            name="Segmented"
            + "_xy%03d_p%04d" % (fov_id, peak_id)
            + "_"
            + "seg_otsu"
            + ".tif",
            visible=True,
        )

    information("Saved segmented channel %d." % peak_id)

    return True


# segmentation algorithm
def segment_image(
    OTSU_threshold,
    first_opening_size,
    distance_threshold,
    second_opening_size,
    min_object_size,
    image,
) -> np.ndarray:
    """Segments a subtracted image and returns a labeled image
    Returns
    -------
    labeled_image : a ndarray which is also an image. Labeled values, which
        should correspond to cells, all have the same integer value starting with 1.
        Non labeled area should have value zero.
    """
    # threshold image
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            thresh = threshold_otsu(image)  # finds optimal OTSU threshhold value
    except Exception as e:
        warnings.warn(f"Otsu thresholding failed: {e}")
        return np.zeros_like(image)

    threshholded = image > OTSU_threshold * thresh  # will create binary image

    # Opening = erosion then dialation.
    # opening smooths images, breaks isthmuses, and eliminates protrusions.
    # "opens" dark gaps between bright features.
    morph = morphology.binary_opening(threshholded, morphology.disk(first_opening_size))

    # if this image is empty at this point (likely if there were no cells), just return
    # zero array
    if np.amax(morph) == 0:
        return np.zeros_like(image)

    ### Calculate distance matrix, use as markers for random walker (diffusion watershed)
    # Generate the markers based on distance to the background
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        distance = ndi.distance_transform_edt(morph)

    # threshold distance image
    distance_thresh = np.zeros_like(distance)
    distance_thresh[distance < distance_threshold] = 0
    distance_thresh[distance >= distance_threshold] = 1

    # do an extra opening on the distance
    distance_opened = morphology.binary_opening(
        distance_thresh, morphology.disk(second_opening_size)
    )

    # remove artifacts connected to image border
    cleared = segmentation.clear_border(distance_opened)
    # remove small objects. Remove small objects wants a
    # labeled image and will fail if there is only one label. Return zero image in that case
    # could have used try/except but remove_small_objects loves to issue warnings.
    labeled, label_num = morphology.label(cleared, connectivity=1, return_num=True)
    if label_num > 1:
        labeled = morphology.remove_small_objects(labeled, min_size=min_object_size)
    else:
        # if there are no labels, then just return the cleared image as it is zero
        return np.zeros_like(image)

    # relabel now that small objects and labels on edges have been cleared
    markers = morphology.label(labeled, connectivity=1)

    # just break if there is no label
    if np.amax(markers) == 0:
        return np.zeros_like(image)

    # the binary image for the watershed, which uses the unmodified OTSU threshold
    threshholded_watershed = threshholded
    # threshholded_watershed = segmentation.clear_border(threshholded_watershed)

    # label using the random walker (diffusion watershed) algorithm
    try:
        # set anything outside of OTSU threshold to -1 so it will not be labeled
        markers[threshholded_watershed == 0] = -1
        # here is the main algorithm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labeled_image = segmentation.random_walker(-1 * image, markers)
        # put negative values back to zero for proper image
        labeled_image[labeled_image == -1] = 0
    except Exception as e:
        warnings.warn(f"Random walker segmentation failed: {e}")
        return np.zeros_like(image)

    return labeled_image


@dataclass
class InPaths:
    specs_path: Path = Path("./analysis/specs.yaml")
    subtracted_folder: Annotated[Path, {"mode": "d"}] = Path("./analysis/subtracted")
    experiment_name: str = ""


@dataclass
class OutPaths:
    segment_folder: Annotated[Path, {"mode": "d"}] = Path("./analysis/segmented")
    cell_data_folder: Annotated[Path, {"mode": "d"}] = Path("./analysis/cell_data")


@dataclass
class RunParams:
    FOVs: FOVList
    phase_plane: str
    num_analyzers: int = multiprocessing.cpu_count()
    OTSU_threshold: float = 1.30
    first_opening_size: Annotated[int, {"min": 0}] = 2
    distance_threshold: Annotated[int, {"min": 0}] = 2
    second_opening_size: Annotated[int, {"min": 0}] = 1
    min_object_size: Annotated[int, {"min": 0}] = 25
    view_result: bool = True


def gen_default_run_params(in_paths: InPaths):
    """Initializes RunParams from a given in_files.
    Probably better to do this in __new__, but..."""
    # TODO: combine all planes into one click
    try:
        all_fovs = get_valid_fovs_folder(in_paths.subtracted_folder)
        # TODO: get the brightest channel as the default phase plane!
        channels = get_valid_planes(in_paths.subtracted_folder)
        # move this into runparams somehow!
        params = RunParams(
            phase_plane=channels[0],
            FOVs=FOVList(all_fovs),
        )
        params.__annotations__["phase_plane"] = Annotated[str, {"choices": channels}]
        return params
    except FileNotFoundError:
        raise FileNotFoundError("TIFF folder not found")
    except ValueError:
        raise ValueError(
            "Invalid filenames. Make sure that timestamps are denoted as t[0-9]* and FOVs as xy[0-9]*"
        )


@thread_worker
def preview_image(
    in_paths: InPaths,
    run_params: RunParams,
    fov_idx: int,
    peak_idx: int,
):
    valid_fov = run_params.FOVs[fov_idx]
    specs = load_specs(in_paths.specs_path)
    # Find first cell-containing peak
    valid_peak = [key for key in specs[valid_fov] if specs[valid_fov][key] == 1][
        peak_idx
    ]
    ## pull out first fov & peak id with cells

    sub_filename = TIFF_FILE_FORMAT_PEAK % (
        in_paths.experiment_name,
        valid_fov,
        valid_peak,
        f"sub_{run_params.phase_plane}",
    )
    sub_stack = load_tiff(in_paths.subtracted_folder / sub_filename)

    # image by image for debug
    segmented_imgs = []
    for sub_image in sub_stack:
        segmented_imgs.append(
            segment_image(
                run_params.OTSU_threshold,
                run_params.first_opening_size,
                run_params.distance_threshold,
                run_params.second_opening_size,
                run_params.min_object_size,
                sub_image,
            )
        )

    # stack them up along a time axis
    segmented_imgs = np.stack(segmented_imgs, axis=0)

    return segmented_imgs.astype("uint8")


def segmentOTSU(
    in_paths: InPaths,
    run_params: RunParams,
    out_paths: OutPaths,
):
    """
    Segments all channels in all FOVs using the OTSU method.
    """

    information("Loading experiment parameters.")

    # create segmentation and cell data folder if they don't exist
    if not out_paths.segment_folder.exists():
        out_paths.segment_folder.mkdir()
    if not out_paths.cell_data_folder.exists():
        out_paths.cell_data_folder.mkdir()

    # load specs file
    specs = load_specs(in_paths.specs_path)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])
    fov_id_list[:] = [fov for fov in fov_id_list if fov in run_params.FOVs]

    ### Do Segmentation by FOV and then peak #######################################################
    information("Segmenting channels using Otsu method.")

    for fov_id in progress(fov_id_list):
        # determine which peaks are to be analyzed (those which have been subtracted)
        ana_peak_ids = []
        for peak_id, spec in six.iteritems(specs[fov_id]):
            if (
                spec == 1
            ):  # 0 means it should be used for empty, -1 is ignore, 1 is analyzed
                ana_peak_ids.append(peak_id)
        ana_peak_ids = sorted(ana_peak_ids)  # sort for repeatability

        for peak_id in ana_peak_ids:
            # send to segmentation
            segment_chnl_stack(
                in_paths.subtracted_folder,
                in_paths.experiment_name,
                run_params.phase_plane,
                out_paths.segment_folder,
                run_params.num_analyzers,
                fov_id,
                peak_id,
                run_params.OTSU_threshold,
                run_params.first_opening_size,
                run_params.distance_threshold,
                run_params.second_opening_size,
                run_params.min_object_size,
                run_params.view_result,
            )

    information("Finished segmentation.")


class SegmentOtsu(MM3Container2):
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

    def regen_widgets(self):
        super().regen_widgets()
        if not self.initialized:
            return
        # would be cool to do this with dask but i am lazy for now.
        self.cur_fov_idx = 0
        self.increment_fov_widget = PushButton(text="next fov")
        self.decrement_fov_widget = PushButton(text="prev fov")
        self.increment_fov_widget.clicked.connect(self.increment_fov)
        self.increment_fov_widget.clicked.connect(self.render_preview)
        self.decrement_fov_widget.clicked.connect(self.decrement_fov)
        self.decrement_fov_widget.clicked.connect(self.render_preview)

        self.change_cur_fov_widget = Container(
            widgets=[self.decrement_fov_widget, self.increment_fov_widget],
            layout="horizontal",
        )
        self.append(self.change_cur_fov_widget)

        self.cur_peak_idx = 0
        self.increment_peak_widget = PushButton(text="next peak")
        self.decrement_peak_widget = PushButton(text="prev peak")
        self.increment_peak_widget.clicked.connect(self.increment_peak)
        self.increment_peak_widget.clicked.connect(self.render_preview)
        self.decrement_peak_widget.clicked.connect(self.decrement_peak)
        self.decrement_peak_widget.clicked.connect(self.render_preview)

        self.change_cur_peak_widget = Container(
            widgets=[self.decrement_peak_widget, self.increment_peak_widget],
            layout="horizontal",
        )
        self.append(self.change_cur_peak_widget)

        self.preview_widget = PushButton(label="generate preview")
        self.append(self.preview_widget)
        self.preview_widget.changed.connect(self.render_preview)
        self.render_preview()

    def run(self):
        segmentOTSU(self.in_paths, self.run_params, self.out_paths)

    def increment_fov(self):
        self.cur_fov_idx = min(self.cur_fov_idx + 1, len(self.run_params.FOVs) - 1)
        self.cur_peak_idx = 0

    def decrement_fov(self):
        self.cur_fov_idx = max(self.cur_fov_idx - 1, 0)
        self.cur_peak_idx = 0

    def increment_peak(self):
        valid_fov = self.run_params.FOVs[self.cur_fov_idx]
        specs = load_specs(self.in_paths.specs_path)
        total_peaks = len(
            [key for key in specs[valid_fov] if specs[valid_fov][key] == 1]
        )
        self.cur_peak_idx = min(self.cur_peak_idx + 1, total_peaks - 1)

    def decrement_peak(self):
        self.cur_peak_idx = max(self.cur_peak_idx - 1, 0)

    def render_preview(self):
        self.viewer.layers.clear()
        valid_fov = self.run_params.FOVs[self.cur_fov_idx]
        specs = load_specs(self.in_paths.specs_path)
        # Find first cell-containing peak
        valid_peak = [key for key in specs[valid_fov] if specs[valid_fov][key] == 1][
            self.cur_peak_idx
        ]
        print(f"rendering xy{valid_fov}p{valid_peak}")
        ## pull out first fov & peak id with cells

        sub_filename = TIFF_FILE_FORMAT_PEAK % (
            self.in_paths.experiment_name,
            valid_fov,
            valid_peak,
            f"sub_{self.run_params.phase_plane}",
        )
        sub_stack = load_tiff(self.in_paths.subtracted_folder / sub_filename)

        # image by image for debug

        images = self.viewer.add_image(sub_stack)
        images.gamma = 1
        worker = preview_image(
            self.in_paths, self.run_params, self.cur_fov_idx, self.cur_peak_idx
        )  # create "worker" object
        worker.returned.connect(self.viewer.add_labels)  # connect callback functions
        worker.start()  # start the thread!


if __name__ == "__main__":
    in_paths = InPaths()
    run_params = gen_default_run_params(in_paths)
    out_paths = OutPaths()
    segmentOTSU(in_paths, run_params, out_paths)
