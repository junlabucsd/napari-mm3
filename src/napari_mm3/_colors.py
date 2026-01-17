import multiprocessing
import os
import pickle
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import numpy as np
from napari import Viewer
from skimage import morphology

from ._deriving_widgets import (
    MM3Container2,
    SegmentationMode,
    get_valid_planes,
    information,
    load_specs,
    load_tiff,
    load_timetable,
    warning,
)
from .utils import TIFF_FILE_FORMAT_PEAK, organize_cells_by_channel


def find_cell_intensities(
    ana_dir: Path,
    experiment_name: str,
    time_table,
    fov_id,
    peak_id,
    cells,
    midline=False,
    channel_name="c2",
    seg_mode=SegmentationMode.OTSU,
):
    """
    Finds fluorescenct information for cells. All the cells in cells
    should be from one fov/peak. See the function
    organize_cells_by_channel()
    """

    # Load fluorescent images and segmented images for this channel
    try:
        sub_channel = "sub_" + channel_name
        # fl_stack = load_stack_params(params, fov_id, peak_id, postfix=sub_channel)

        img_dir = ana_dir / "subtracted"
        img_filename = TIFF_FILE_FORMAT_PEAK % (
            experiment_name,
            fov_id,
            peak_id,
            sub_channel,
        )
        fl_stack = load_tiff(img_dir / img_filename)
        information("Loading subtracted channel to analyze.")
    except FileNotFoundError:
        warning("Could not find subtracted channel! Skipping.")
        return

    # seg_stack = load_stack_params(params, fov_id, peak_id, postfix="seg_unet")
    seg_str = "seg_otsu" if seg_mode == SegmentationMode.OTSU else "seg_unet"
    img_filename = TIFF_FILE_FORMAT_PEAK % (
        experiment_name,
        fov_id,
        peak_id,
        seg_str,
    )
    seg_stack = load_tiff(ana_dir / "segmented" / img_filename)
    # determine absolute time index
    times_all = []
    for fov in time_table:
        times_all = np.append(times_all, list(time_table[fov].keys()))
    times_all = np.unique(times_all)
    times_all = np.sort(times_all)
    times_all = np.array(times_all, np.int_)
    t0 = times_all[0]  # first time index

    # Loop through cells
    for cell in cells.values():
        # give this cell two lists to hold new information
        cell.fl_tots = []  # total fluorescence per time point
        cell.fl_area_avgs = []  # avg fluorescence per unit area by timepoint
        cell.fl_vol_avgs = []  # avg fluorescence per unit volume by timepoint

        if midline:
            cell.mid_fl = []  # avg fluorescence of midline

        # and the time points that make up this cell's life
        for n, t in enumerate(cell.times):
            # create fluorescent image only for this cell and timepoint.
            fl_image_masked = np.copy(fl_stack[t - t0])
            fl_image_masked[seg_stack[t - t0] != cell.labels[n]] = 0

            # append total flourescent image
            cell.fl_tots.append(np.sum(fl_image_masked))
            # and the average fluorescence
            cell.fl_area_avgs.append(np.sum(fl_image_masked) / cell.areas[n])
            cell.fl_vol_avgs.append(np.sum(fl_image_masked) / cell.volumes[n])

            if midline:
                # add the midline average by first applying morphology transform
                bin_mask = np.copy(seg_stack[t - t0])
                bin_mask[bin_mask != cell.labels[n]] = 0
                med_mask, _ = morphology.medial_axis(bin_mask, return_distance=True)
                # med_mask[med_dist < np.floor(cap_radius/2)] = 0
                # print(img_fluo[med_mask])
                if np.shape(fl_image_masked[med_mask])[0] > 0:
                    cell.mid_fl.append(np.nanmean(fl_image_masked[med_mask]))
                else:
                    cell.mid_fl.append(0)

    # The cell objects in the original dictionary will be updated,
    # no need to return anything specifically.


def find_cell_intensities_worker(
    analysis_dir: Path,
    experiment_name: str,
    fov_id,
    peak_id,
    cells,
    midline=True,
    channel="c2",
    seg_mode=SegmentationMode.OTSU,
):
    """
    Finds fluorescenct information for cells. All the cells in cells
    should be from one fov/peak. See the function
    organize_cells_by_channel()
    This version is the same as find_cell_intensities but return the cells object for collection by the pool.
    The original find_cell_intensities is kept for compatibility.
    """
    information("Processing peak {} in FOV {}".format(peak_id, fov_id))
    # Load fluorescent images and segmented images for this channel
    # fl_stack = load_stack_params(params, fov_id, peak_id, postfix=channel)
    # switch postfix to c1/c2/c3 auto??
    fl_filename = TIFF_FILE_FORMAT_PEAK % (
        experiment_name,
        fov_id,
        peak_id,
        channel,
    )
    fl_stack = load_tiff(analysis_dir / "channels" / fl_filename)

    seg_str = "seg_otsu" if seg_mode == SegmentationMode.OTSU else "seg_unet"
    seg_filename = TIFF_FILE_FORMAT_PEAK % (
        experiment_name,
        fov_id,
        peak_id,
        seg_str,
    )
    seg_stack = load_tiff(analysis_dir / "segmented" / seg_filename)

    # determine absolute time index
    time_table = load_timetable(analysis_dir)
    times_all = []
    for fov in time_table:
        times_all = np.append(times_all, [int(x) for x in fov.keys()])
    times_all = np.unique(times_all)
    times_all = np.sort(times_all)
    times_all = np.array(times_all, np.int_)
    t0 = times_all[0]  # first time index

    # Loop through cells
    for cell in cells.values():
        # give this cell two lists to hold new information

        fl_tots = []  # total fluorescence per time point
        fl_area = []  # avg fluorescence per unit area by timepoint
        fl_vol = []  # avg fluorescence per unit volume by timepoint

        if midline:
            cell.mid_fl = []  # avg fluorescence of midline

        # and the time points that make up this cell's life
        for n, t in enumerate(cell.times):
            # create fluorescent image only for this cell and timepoint.
            fl_image_masked = np.copy(fl_stack[t - t0])
            fl_image_masked[seg_stack[t - t0] != cell.labels[n]] = 0

            # append total fluorescent image
            fl_tots.append(np.sum(fl_image_masked))
            # and the average fluorescence
            fl_area.append(np.sum(fl_image_masked) / cell.areas[n])
            fl_vol.append(np.sum(fl_image_masked) / cell.volumes[n])

            if midline:
                # add the midline average by first applying morphology transform
                bin_mask = np.copy(seg_stack[t - t0])
                bin_mask[bin_mask != cell.labels[n]] = 0
                med_mask, _ = morphology.medial_axis(bin_mask, return_distance=True)
                if np.shape(fl_image_masked[med_mask])[0] > 0:
                    cell.mid_fl.append(np.nanmean(fl_image_masked[med_mask]))
                else:
                    cell.mid_fl.append(0)

        cell.setattr("fl_tots_c{0}".format(channel), fl_tots)
        cell.setattr("fl_area_c{0}".format(channel), fl_area)
        cell.setattr("fl_vol_c{0}".format(channel), fl_vol)

    # return the cell object to the pool initiated by mm3_Colors.
    return cells


# load cell file
def colors(
    cell_folder: Path,
    experiment_name: str,
    analysis_dir: Path,
    num_analyzers: int,
    fl_channel: str,
    seg_method: str,
    cellfile_path: Path,
):
    """
    Finds fluorescenct information for cells.
    """
    information("Loading cell data.")

    with open(cellfile_path, "rb") as cell_file:
        complete_cells = pickle.load(cell_file)

    specs = load_specs(analysis_dir)
    time_table = load_timetable(analysis_dir)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])
    information("Processing %d FOVs." % len(fov_id_list))
    cells_by_peak = organize_cells_by_channel(complete_cells, specs)

    # multiprocessing
    color_multiproc = False
    if color_multiproc:
        cells_to_pool = []
        for fov_id in fov_id_list:
            peak_ids = cells_by_peak[fov_id].keys()
            peak_id_cells = cells_by_peak[fov_id].values()
            fov_ids = [fov_id] * len(peak_ids)

            cells_to_pool += zip(fov_ids, peak_ids, peak_id_cells)
        pool = Pool(processes=num_analyzers)

        mapfunc = partial(
            find_cell_intensities_worker,
            analysis_dir,
            experiment_name,
            fl_channel,
            seg_method,
            cellfile_path,
        )
        cells_updates = pool.starmap_async(mapfunc, cells_to_pool)
        # [pool.apply_async(find_cell_intensities(fov_id, peak_id, cells, midline=True, channel=namespace.channel)) for fov_id in fov_id_list for peak_id, cells in cells_by_peak[fov_id].items()]

        pool.close()  # tells the process nothing more will be added.
        pool.join()
        update_cells = (
            cells_updates.get()
        )  # the result is a list of cells dictionary, each dict contains several cells
        update_cells = {
            cell_id: cell for cells in update_cells for cell_id, cell in cells.items()
        }
        for cell_id, cell in update_cells.items():
            complete_cells[cell_id] = cell

    # for each set of cells in one fov/peak, compute the fluorescence
    else:
        for fov_id in fov_id_list:
            if fov_id in cells_by_peak:
                information("Processing FOV {}.".format(fov_id))
                for peak_id, cells in cells_by_peak[fov_id].items():
                    information("Processing peak {}.".format(peak_id))
                    find_cell_intensities(
                        analysis_dir,
                        experiment_name,
                        time_table,
                        fov_id,
                        peak_id,
                        cells,
                        midline=False,
                        channel_name=fl_channel,
                    )

    # Just the complete cells, those with mother and daugther
    cell_filename = os.path.basename(cellfile_path)
    with open(cell_folder / (cell_filename[:-4] + "_fl.pkl"), "wb") as cell_file:
        pickle.dump(complete_cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    information("Finished.")


@dataclass
class InPaths:
    cell_file: Path = Path("./analysis/cell_data/complete_cells.pkl")
    cell_data_folder: Path = Path("./analysis/cell_data")
    analysis_folder: Path = Path("./analysis")
    channels_dir: Path = Path("./analysis/channels")
    segmented_dir: Path = Path("./analysis/segmented")
    seg_method: Annotated[str, {"choices": ["Otsu", "U-net"]}] = "U-net"


@dataclass
class OutPaths:
    experiment_name: str = ""


@dataclass
class RunParams:
    analysis_plane: Annotated[str, {"tooltip": "fluorescence plane to analyze"}]
    num_analyzers: int = multiprocessing.cpu_count()


def gen_default_run_params(in_files: InPaths):
    try:
        channels = get_valid_planes(in_files.channels_dir)
        params = RunParams(
            analysis_plane=channels[0],
        )
        params.__annotations__["analysis_plane"] = Annotated[str, {"choices": channels}]
        return params
    except FileNotFoundError:
        raise FileNotFoundError("TIFF folder not found")
    except ValueError:
        raise ValueError(
            "Invalid filenames. Make sure that timestamps are denoted as t[0-9]* and FOVs as xy[0-9]*"
        )


class Colors(MM3Container2):
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
        self.viewer.window._status_bar._toggle_activity_dock(True)
        colors(
            self.in_paths.cell_data_folder,
            self.out_paths.experiment_name,
            self.in_paths.analysis_folder,
            self.run_params.num_analyzers,
            self.run_params.analysis_plane,
            self.in_paths.seg_method,
            self.in_paths.cell_file,
        )


if __name__ == "__main__":
    in_paths = InPaths()
    run_params = gen_default_run_params(in_paths)
    out_paths = OutPaths()
    colors(
        in_paths.cell_data_folder,
        out_paths.experiment_name,
        in_paths.analysis_folder,
        run_params.num_analyzers,
        run_params.analysis_plane,
        in_paths.seg_method,
        in_paths.cell_file,
    )
