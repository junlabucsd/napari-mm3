"""
Two separate pipelines here.
1. Preview
    a. Find the blobs in a peak.  -- DONE
    b. Assign each blob to a cell.  -- DONE
    c. Filter blobs off SNR -- DONE
    d. Return screenspace positions of each blob. -- DONE

2. Full foci analysis.
    z. for each fov, peak, time
    a. Find the blobs in the image => Reuse from (1)
    b. Assign each blob to a cell, if possible. => Reuse from (1)
    c. Filter blobs off SNR -- TODO: Check what blob_log does as threshold => Reuse from (1)
    d. Write the new Cells file.

---
Also need to regression test!
It is possible to keep the entire Cells file in memory, so in-place editing isn't necessary.
"""

from __future__ import print_function

import pickle
import numpy as np
import scipy.signal as signal

from pathlib import Path
from magicgui.widgets import SpinBox, ComboBox, FileEdit, FloatSpinBox, PushButton
from skimage.feature import blob_log  # used for foci finding
from dataclasses import dataclass

from .utils import (
    organize_cells_by_channel,
    write_cells_to_json,
    read_cells_from_json,
    TIFF_FILE_FORMAT_PEAK,
    Cells,
)

from ._deriving_widgets import (
    MM3Container,
    PlanePicker,
    SegmentationMode,
    load_tiff,
    load_specs,
    load_subtracted_stack,
    InteractivePeakChooser,
)


@dataclass
class FociParams:
    minsig: int
    maxsig: int
    threshold: float
    median_ratio: float


def find_img_blobs(img: np.ndarray, foci_params: FociParams):
    # find blobs using difference of gaussian
    # if two blobs overlap by more than this fraction, smaller blob is cut:
    over_lap = 0.95
    # number of division to consider between min ang max sig.
    numsig = foci_params.maxsig - foci_params.minsig + 1
    # This is supposed to help with detection.
    blobs = blob_log(
        img,
        min_sigma=foci_params.minsig,
        max_sigma=foci_params.maxsig,
        overlap=over_lap,
        num_sigma=numsig,
        threshold=foci_params.threshold,
    )
    return blobs


def compute_blob_signal(img: np.ndarray, blob):
    row = np.rint(blob[0]).astype(np.int32)
    col = np.rint(blob[1]).astype(np.int32)
    radius = np.rint(blob[2]).astype(np.int32)
    row_min = row - radius
    row_max = row + radius
    col_min = col - radius
    col_max = col + radius
    img_box = img[row_min:row_max, col_min:col_max]
    # in this case, SNR is arbitrary-enough units anyway,
    # so this doesn't need to actually mean anything.
    # that said... should find a more principled way to do it.
    sorted_brightness = np.sort(img_box.flatten())
    return np.average(sorted_brightness[-4:])


def compute_cell_noise(cell_region: np.ndarray, img_foci: np.ndarray, blob):
    img_foci_masked = np.copy(img_foci).astype(float)
    img_foci_masked[~cell_region] = np.nan
    row = np.rint(blob[0]).astype(np.int32)
    col = np.rint(blob[1]).astype(np.int32)
    radius = np.rint(blob[2]).astype(np.int32)
    row_min = row - radius
    row_max = row + radius
    col_min = col - radius
    col_max = col + radius
    # Turn the foci into a blank.
    img_foci_masked[row_min:row_max, col_min:col_max] = np.nan
    cell_fl_std = np.nanstd(img_foci_masked)

    return cell_fl_std


def find_peak_blobs(peak: np.ndarray, foci_params: FociParams):
    blob_times = []
    all_blobs = []
    for t, img in enumerate(peak):
        blobs = find_img_blobs(img, foci_params)
        all_blobs += list(blobs)
        blob_times += len(blobs) * [t + 1]
    all_blobs = np.array(all_blobs)
    return blob_times, all_blobs


def assign_blobs_to_cells(
    blobs,
    blob_times,
    cell_dict: Cells,
    seg_stack: np.ndarray,
    img_stack: np.ndarray,
    signal_to_noise: float,
    absolute_locs=False,
):
    # Generate a map that takes in a time and spits out a list of cells at that time.
    time_to_cells_map = {}
    for cell_id, cell in cell_dict.items():
        for time_real in cell.times:
            if time_real in time_to_cells_map:
                time_to_cells_map[time_real].append(cell_id)
            else:
                time_to_cells_map[time_real] = [cell_id]

    cell_foci_ys = []
    cell_foci_xs = []
    foci_ys = []
    foci_xs = []
    foci_hs = []
    times = []
    blob_assignments = []
    for blob, time_real in zip(blobs, blob_times):
        # The fucking time is 1-indexed...
        time_idx = time_real - 1
        if not (time_real in time_to_cells_map):
            continue
        same_time_cells = time_to_cells_map[time_real]
        for cell_id in same_time_cells:
            cell = cell_dict[cell_id]
            cell_y, cell_x, cell_time = cell.place_in_cell(blob[1], blob[0], time_real)
            # need a better signal-to-noise definition...
            # Ok -- to test things out, for now remove filtering completely.
            if cell_y is not None:
                cell_mask = seg_stack[time_idx] == cell.labels[cell_time]
                noise = compute_cell_noise(cell_mask, img_stack[time_idx], blob)
                blob_signal = compute_blob_signal(img_stack[time_idx], blob)

                if blob_signal / noise < signal_to_noise:
                    continue

                # this is where we do the median check, as well.
                cell_foci_ys.append(cell_y)
                cell_foci_xs.append(cell_x)
                foci_hs.append(blob_signal)
                blob_assignments.append(cell.id)
                foci_ys.append(blob[0])
                foci_xs.append(blob[1])
                times.append(time_real)
                break

    if absolute_locs:
        return foci_ys, foci_xs, foci_hs, blob_assignments, times
    return cell_foci_ys, cell_foci_xs, foci_hs, blob_assignments, times


def foci_preview(
    cells: Cells, seg_stack: np.ndarray, img_stack: np.ndarray, foci_params: FociParams
):
    blob_times, blobs = find_peak_blobs(img_stack, foci_params)
    cell_foci_ys, cell_foci_xs, foci_hs, assignments, times_real = (
        assign_blobs_to_cells(
            blobs,
            blob_times,
            cells,
            seg_stack,
            img_stack,
            foci_params.median_ratio,
            absolute_locs=True,
        )
    )
    time_idx = [time_real - 1 for time_real in times_real]
    return cell_foci_ys, cell_foci_xs, foci_hs, assignments, time_idx


# find foci using a difference of gaussians method.
# the idea of this one is to be run on a single preview
def gen_image_kymo(stack: np.ndarray, n_steps=50):
    image_kymo = []
    for i in range(stack.shape[0] - n_steps):
        sub_stack_fl2 = stack[i : i + n_steps, :, :]
        mini_kymo = np.hstack(sub_stack_fl2)
        image_kymo.append(mini_kymo)
    return np.array(image_kymo)


def cell_foci(
    cells_in_peak: Cells,
    seg_stack: np.ndarray,
    img_stack: np.ndarray,
    foci_params: FociParams,
):
    """
    Runs foci picker on a particular peak.
    Modifies 'cells_in_peak' in-place -- thus, does not return anything.
    """
    # get peak img_stack
    blob_times, blobs = find_peak_blobs(img_stack, foci_params)
    foci_ys, foci_xs, foci_hs, assignments, times = assign_blobs_to_cells(
        blobs,
        blob_times,
        cells_in_peak,
        seg_stack,
        img_stack,
        foci_params.median_ratio,
    )
    for i in range(len(times)):
        foci_y, foci_x, foci_h = foci_ys[i], foci_xs[i], foci_hs[i]
        cell_id = assignments[i]
        cur_cell = cells_in_peak[cell_id]
        cell_time_idx = cur_cell.times.index(times[i])
        if (not hasattr(cur_cell, "disp_l")) or (cur_cell.disp_l is None):
            # Doing it this way to avoid reference issues with lists.
            # (ie, lack of a default deep copy of list objects)
            cur_cell.disp_l = [[] for _ in cur_cell.times]
            cur_cell.disp_w = [[] for _ in cur_cell.times]
            cur_cell.foci_h = [[] for _ in cur_cell.times]
        cur_cell.disp_l[cell_time_idx].append(foci_y)
        cur_cell.disp_w[cell_time_idx].append(foci_x)
        cur_cell.foci_h[cell_time_idx].append(foci_h)


def foci(
    ana_dir: Path,
    experiment_name: str,
    foci_params: FociParams,
    fl_plane: str,
    seg_method: SegmentationMode,
    cell_file_path: Path,
):
    """
    Main function for foci analysis. Loads cells, finds foci, and saves out the results.
    Parameters
    ----------
    params : dict
        Dictionary of parameters
    fl_plane : str
        Name of fluorescence plane to use for foci analysis
    seg_method : str
        Name of segmentation method to use for foci analysis
    cell_file_path : str
        Path to cell file

    Returns
    -------
    Cells
    """
    try:
        cells = read_cells_from_json(cell_file_path)
    except Exception:
        print("cells json not found. checking for pickle file (legacy)")
        with open(cell_file_path, "rb") as cell_file:
            cells = pickle.load(cell_file)
    specs = load_specs(ana_dir)
    # time_table = load_time_table(ana_dir)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])
    print("Starting foci analysis.")

    # create dictionary which organizes cells by fov and peak_id
    cells_by_peak = organize_cells_by_channel(cells, specs)
    # for each set of cells in one fov/peak, find the foci
    for fov_id in fov_id_list:
        if not (fov_id in cells_by_peak):
            continue

        for peak_id, cells_of_peak in cells_by_peak[fov_id].items():
            if len(cells_of_peak) == 0:
                return

            img_stack = load_subtracted_stack(
                ana_dir=ana_dir,
                experiment_name=experiment_name,
                fov_id=fov_id,
                peak_id=peak_id,
                postfix=f"sub_{fl_plane}",
            )

            seg_str = "seg_otsu" if seg_method == SegmentationMode.OTSU else "seg_unet"
            img_filename = TIFF_FILE_FORMAT_PEAK % (
                experiment_name,
                fov_id,
                peak_id,
                seg_str,
            )
            seg_stack = load_tiff(ana_dir / "segmented" / img_filename)

            print(f"running foci analysis for peak {peak_id} and fov {fov_id}")
            cell_foci(cells_of_peak, seg_stack, img_stack, foci_params)

    return cells


class Foci(MM3Container):
    def create_widgets(self):
        self.cellfile_widget = FileEdit(
            label="cell_file",
            value=self.analysis_folder / "cell_data/all_cells.json",
            tooltip="Cell file to be analyzed",
        )

        self.plane_widget = PlanePicker(
            self.valid_planes,
            label="analysis plane",
            tooltip="Fluoresence plane that you would like to analyze",
        )
        self.plane_widget.value = self.valid_planes[-1]
        self.segmentation_method_widget = ComboBox(
            label="segmentation method", choices=["Otsu", "U-net"]
        )
        self.segmentation_method_widget.value = "U-net"

        # minimum and maximum sigma of laplacian to convolve in pixels.
        # Scales with minimum foci width to detect as 2*sqrt(2)*minsig
        self.log_minsig_widget = SpinBox(
            value=2,
            label="LoG min sigma",
            tooltip="min sigma of laplacian to convolve in pixels",
            min=1,
        )
        self.log_maxsig_widget = SpinBox(
            value=4,
            label="LoG max sigma",
            tooltip="max sigma of laplacian to convolve in pixels",
            min=1,
        )
        # absolute threshold laplacian must reach to record potential foci. Keep low to detect dimmer spots.
        self.log_thresh_widget = FloatSpinBox(
            value=0.00001,
            step=0.00001,
            label="LoG threshold",
            tooltip="absolute threshold laplacian must reach to record potential foci",
        )  # default: 0.002;
        # foci peaks must be this many times greater than median cell intensity.
        # Think signal to noise ratio
        self.log_peak_ratio_widget = FloatSpinBox(
            value=0.1,
            step=0.1,
            label="SNR",
            tooltip="Minimum foci peak to background ratio for detection",
        )

        self.preview_widget = PushButton(label="generate preview", value=False)

        self.specs = load_specs(self.analysis_folder)

        self.n_steps = 20

        self.set_plane()
        self.set_segmentation_method()
        self.set_cellfile()
        self.set_fovs(self.valid_fovs)
        self.set_log_maxsig()
        self.set_log_minsig()
        self.set_log_thresh()
        self.set_log_peak_ratio()
        self.generate_fovs_to_peaks()
        self.peak_switch_widget = InteractivePeakChooser(
            self.valid_fovs, self.fov_to_peaks
        )
        self.set_peak_and_fov()

        self.plane_widget.changed.connect(self.set_plane)
        self.segmentation_method_widget.changed.connect(self.set_segmentation_method)
        self.cellfile_widget.changed.connect(self.set_cellfile)
        self.log_minsig_widget.changed.connect(self.set_log_minsig)
        self.log_maxsig_widget.changed.connect(self.set_log_maxsig)
        self.log_thresh_widget.changed.connect(self.set_log_thresh)
        self.log_peak_ratio_widget.changed.connect(self.set_log_peak_ratio)
        self.viewer.window._status_bar._toggle_activity_dock(True)
        self.preview_widget.clicked.connect(self.render_preview)
        self.peak_switch_widget.connect(self.set_peak_and_fov)

        self.append(self.plane_widget)
        self.append(self.segmentation_method_widget)
        self.append(self.cellfile_widget)
        self.append(self.log_minsig_widget)
        self.append(self.log_maxsig_widget)
        self.append(self.log_thresh_widget)
        self.append(self.log_peak_ratio_widget)
        self.append(self.preview_widget)
        self.append(self.peak_switch_widget)

    def generate_fovs_to_peaks(self):
        cells = read_cells_from_json(str(self.cellfile))
        self.fov_to_peaks = {}
        for valid_fov in self.valid_fovs:
            valid_peaks = organize_cells_by_channel(cells, self.specs)[valid_fov].keys()
            self.fov_to_peaks[valid_fov] = valid_peaks

    def run(self):
        foci_params = FociParams(
            self.log_minsig, self.log_maxsig, self.log_thresh, self.log_peak_ratio
        )

        computed_cells = foci(
            self.analysis_folder,
            self.experiment_name,
            foci_params,
            self.fl_plane,
            self.segmentation_method,
            str(self.cellfile),
        )
        new_file_name = self.cellfile.name[:-5] + "_foci.json"
        write_cells_to_json(computed_cells, self.cellfile.parent / new_file_name)
        print("Foci detection complete.")

    def load_preview_cells(self):
        cells = read_cells_from_json(str(self.cellfile))
        self.cells = organize_cells_by_channel(cells, self.specs)[self.preview_fov][
            self.preview_peak
        ]

        foci_params = FociParams(
            self.log_minsig, self.log_maxsig, self.log_thresh, self.log_peak_ratio
        )
        all_times = set()
        for cellid, cell in self.cells.items():
            all_times.union(set(cell.times))

        foci_stack = load_subtracted_stack(
            self.analysis_folder,
            self.experiment_name,
            self.preview_fov,
            self.preview_peak,
            f"sub_{self.fl_plane}",
        )
        seg_str = (
            "seg_otsu"
            if self.segmentation_method == SegmentationMode.OTSU
            else "seg_unet"
        )
        img_filename = TIFF_FILE_FORMAT_PEAK % (
            self.experiment_name,
            self.preview_fov,
            self.preview_peak,
            seg_str,
        )
        seg_stack = load_tiff(self.analysis_folder / "segmented" / img_filename)

        self.y_pts, self.x_pts, self.radii, self.assignments, self.times = foci_preview(
            self.cells, seg_stack, foci_stack, foci_params
        )

        self.x_pts, self.y_pts, self.radii, self.times = (
            np.array(self.x_pts),
            np.array(self.y_pts),
            np.array(self.radii),
            np.array(self.times),
        )

    def render_preview(self):
        """
        Previews foci in something resembling a kymograph.
        """
        self.load_preview_cells()
        kymos = []
        # pull out first fov & peak id with cells
        # then, create a p x (t_total - n_steps + 1) x L x (W * n_steps)
        # with p as the number of planes,
        # this displays 'sets' of n_steps images.
        for plane in self.valid_planes:
            postfix = f"sub_{plane}"
            sub_stack_fl = load_subtracted_stack(
                self.analysis_folder,
                self.experiment_name,
                self.preview_fov,
                self.preview_peak,
                postfix,
            )
            image_kymo = gen_image_kymo(sub_stack_fl, n_steps=self.n_steps)
            kymos.append(image_kymo)

        kymos = np.array(kymos)
        self.img_width = kymos[0].shape[2] // self.n_steps

        self.viewer.layers.clear()
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()
        self.viewer.grid.enabled = False
        self.viewer.add_image(np.array(kymos))
        self.viewer.layers["Image"].reset_contrast_limits()
        self.viewer.reset_view()

        specs = load_specs(self.analysis_folder)
        cells_by_peak = organize_cells_by_channel(self.cells, specs)
        cells_visible = cells_by_peak[self.preview_fov][self.preview_peak]
        # only going to show cells that were lineage tracked.

        seg_str = (
            "seg_otsu"
            if self.segmentation_method == SegmentationMode.OTSU
            else "seg_unet"
        )
        img_filename = TIFF_FILE_FORMAT_PEAK % (
            self.experiment_name,
            self.preview_fov,
            self.preview_peak,
            seg_str,
        )

        seg_stack = load_tiff(self.analysis_folder / "segmented" / img_filename)

        cell_filter_stack = [[] for _ in range(len(seg_stack))]
        # check about frame 0.
        for cell_id, cell in cells_visible.items():
            for i, time in enumerate(cell.times):
                if time == 0:
                    print("time 0 found")
                cell_filter_stack[time - 1].append(cell.labels[i])
        # only keep the regions including cells we've saved.
        new_seg_stack = []
        for t, time_useful_regions in enumerate(cell_filter_stack):
            mask = np.full(seg_stack.shape[1:], False)
            for good_region in time_useful_regions:
                mask = mask | (seg_stack[t] == good_region)
            masked = np.where(mask, seg_stack[t], 0)
            new_seg_stack.append(masked)
        new_seg_stack = np.array(new_seg_stack)
        print(f"seg stack: {seg_stack.shape}")
        print(f"new stack: {new_seg_stack.shape}")
        self.viewer.add_labels(gen_image_kymo(new_seg_stack, n_steps=self.n_steps))

        # Add these here because napari doesn't like it if you try to access these events
        # without first creating the relevant layers.
        self.draw_points()
        self.viewer.dims.events.current_step.connect(self.draw_points)
        self.viewer.dims.events.current_step.connect(self.reset_contrast_limits)
        self.viewer.layers[-1].selected_data = []

    def reset_contrast_limits(self):
        try:
            self.viewer.layers["Image"].reset_contrast_limits()
        except:
            pass

    def draw_points(self):
        cur_time = self.viewer.dims.current_step[1]
        times_lower_bound = cur_time <= self.times
        times_upper_bound = cur_time + self.n_steps > self.times
        times_in_range_mask = times_lower_bound & times_upper_bound
        times_in_range = self.times[times_in_range_mask]
        visible_xs = self.x_pts[times_in_range_mask]
        visible_xs_kymo_offset = (
            visible_xs + (times_in_range - cur_time) * self.img_width
        )
        visible_ys = self.y_pts[times_in_range_mask]
        points = np.stack((visible_ys, visible_xs_kymo_offset)).T

        # if the points layer exists => update it.
        # if the points layer does not exist => create it.
        try:
            self.viewer.layers["points"].data = points
        except:
            self.points = self.viewer.add_points(
                data=points,
                # size=1.5 * np.array(self.radii[times_in_range_mask]),
                name="points",
                face_color="orange",
                edge_color="white",
            )
            # self.display_pts_from_cells()

    def set_peak_and_fov(self):
        self.preview_peak = self.peak_switch_widget.cur_peak
        self.preview_fov = self.peak_switch_widget.cur_fov

    def set_plane(self):
        self.fl_plane = self.plane_widget.value

    def set_segmentation_method(self):
        self.segmentation_method = (
            SegmentationMode.OTSU
            if self.segmentation_method_widget.value == "Otsu"
            else SegmentationMode.UNET
        )

    def set_cellfile(self):
        self.cellfile = self.cellfile_widget.value

    def set_fovs(self, fovs):
        self.fovs = list(set(fovs))

    def set_log_peak_ratio(self):
        self.log_peak_ratio = self.log_peak_ratio_widget.value

    def set_log_minsig(self):
        self.log_minsig = self.log_minsig_widget.value

    def set_log_maxsig(self):
        self.log_maxsig = self.log_maxsig_widget.value

    def set_log_thresh(self):
        self.log_thresh = self.log_thresh_widget.value
