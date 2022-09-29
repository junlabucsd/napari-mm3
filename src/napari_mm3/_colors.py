import multiprocessing
from multiprocessing import Pool
import numpy as np
import pickle
import os
from pathlib import Path
from functools import partial

from skimage import morphology


from ._deriving_widgets import MM3Container, PlanePicker, FOVChooser
from magicgui.widgets import ComboBox, FileEdit

from .utils import organize_cells_by_channel

from ._deriving_widgets import (
    load_specs,
    load_time_table,
    information,
    warning,
    load_stack_params,
)


def find_cell_intensities(
    params, time_table, fov_id, peak_id, Cells, midline=False, channel_name="c2"
):
    """
    Finds fluorescenct information for cells. All the cells in Cells
    should be from one fov/peak. See the function
    organize_cells_by_channel()
    """

    # Load fluorescent images and segmented images for this channel
    try:
        sub_channel = "sub_" + channel_name
        fl_stack = load_stack_params(params, fov_id, peak_id, postfix=sub_channel)
        information("Loading subtracted channel to analyze.")
    except FileNotFoundError:
        warning("Could not find subtracted channel! Skipping.")
        return

    seg_stack = load_stack_params(params, fov_id, peak_id, postfix="seg_unet")

    # determine absolute time index
    times_all = []
    for fov in time_table:
        times_all = np.append(times_all, list(time_table[fov].keys()))
    times_all = np.unique(times_all)
    times_all = np.sort(times_all)
    times_all = np.array(times_all, np.int_)
    t0 = times_all[0]  # first time index

    # Loop through cells
    for Cell in Cells.values():
        # give this cell two lists to hold new information
        Cell.fl_tots = []  # total fluorescence per time point
        Cell.fl_area_avgs = []  # avg fluorescence per unit area by timepoint
        Cell.fl_vol_avgs = []  # avg fluorescence per unit volume by timepoint

        if midline:
            Cell.mid_fl = []  # avg fluorescence of midline

        # and the time points that make up this cell's life
        for n, t in enumerate(Cell.times):
            # create fluorescent image only for this cell and timepoint.
            fl_image_masked = np.copy(fl_stack[t - t0])
            fl_image_masked[seg_stack[t - t0] != Cell.labels[n]] = 0

            # append total flourescent image
            Cell.fl_tots.append(np.sum(fl_image_masked))
            # and the average fluorescence
            Cell.fl_area_avgs.append(np.sum(fl_image_masked) / Cell.areas[n])
            Cell.fl_vol_avgs.append(np.sum(fl_image_masked) / Cell.volumes[n])

            if midline:
                # add the midline average by first applying morphology transform
                bin_mask = np.copy(seg_stack[t - t0])
                bin_mask[bin_mask != Cell.labels[n]] = 0
                med_mask, _ = morphology.medial_axis(bin_mask, return_distance=True)
                # med_mask[med_dist < np.floor(cap_radius/2)] = 0
                # print(img_fluo[med_mask])
                if np.shape(fl_image_masked[med_mask])[0] > 0:
                    Cell.mid_fl.append(np.nanmean(fl_image_masked[med_mask]))
                else:
                    Cell.mid_fl.append(0)

    # The cell objects in the original dictionary will be updated,
    # no need to return anything specifically.


def find_cell_intensities_worker(
    params, fov_id, peak_id, Cells, midline=True, channel="c2"
):
    """
    Finds fluorescenct information for cells. All the cells in Cells
    should be from one fov/peak. See the function
    organize_cells_by_channel()
    This version is the same as find_cell_intensities but return the Cells object for collection by the pool.
    The original find_cell_intensities is kept for compatibility.
    """
    information("Processing peak {} in FOV {}".format(peak_id, fov_id))
    # Load fluorescent images and segmented images for this channel
    fl_stack = load_stack_params(params, fov_id, peak_id, postfix=channel)
    seg_stack = load_stack_params(params, fov_id, peak_id, postfix="seg_otsu")

    # determine absolute time index
    time_table = load_time_table(params["ana_dir"])
    times_all = []
    for fov in time_table:
        times_all = np.append(times_all, [int(x) for x in fov.keys()])
    times_all = np.unique(times_all)
    times_all = np.sort(times_all)
    times_all = np.array(times_all, np.int_)
    t0 = times_all[0]  # first time index

    # Loop through cells
    for Cell in Cells.values():
        # give this cell two lists to hold new information

        fl_tots = []  # total fluorescence per time point
        fl_area = []  # avg fluorescence per unit area by timepoint
        fl_vol = []  # avg fluorescence per unit volume by timepoint

        if midline:
            Cell.mid_fl = []  # avg fluorescence of midline

        # and the time points that make up this cell's life
        for n, t in enumerate(Cell.times):
            # create fluorescent image only for this cell and timepoint.
            fl_image_masked = np.copy(fl_stack[t - t0])
            fl_image_masked[seg_stack[t - t0] != Cell.labels[n]] = 0

            # append total fluorescent image
            fl_tots.append(np.sum(fl_image_masked))
            # and the average fluorescence
            fl_area.append(np.sum(fl_image_masked) / Cell.areas[n])
            fl_vol.append(np.sum(fl_image_masked) / Cell.volumes[n])

            if midline:
                # add the midline average by first applying morphology transform
                bin_mask = np.copy(seg_stack[t - t0])
                bin_mask[bin_mask != Cell.labels[n]] = 0
                med_mask, _ = morphology.medial_axis(bin_mask, return_distance=True)
                if np.shape(fl_image_masked[med_mask])[0] > 0:
                    Cell.mid_fl.append(np.nanmean(fl_image_masked[med_mask]))
                else:
                    Cell.mid_fl.append(0)

        Cell.setattr("fl_tots_c{0}".format(channel), fl_tots)
        Cell.setattr("fl_area_c{0}".format(channel), fl_area)
        Cell.setattr("fl_vol_c{0}".format(channel), fl_vol)

    # return the cell object to the pool initiated by mm3_Colors.
    return Cells


# load cell file
def colors(params, fl_channel, seg_method, cellfile_path):
    information("Loading cell data.")
    # if namespace.cellfile:
    #     cell_file_path = namespace.cellfile.name
    # else:
    #     warning('No cell file specified. Using complete_cells.pkl.')
    #     cell_file_path = p['cell_dir'] / 'complete_cells.pkl'

    with open(cellfile_path, "rb") as cell_file:
        Complete_Cells = pickle.load(cell_file)

    # if namespace.seg_method:
    #     seg_method = 'seg_'+str(namespace.seg_method)
    # else:
    #     warning('Defaulting to otsu segmented cells')
    #     seg_method = 'seg_otsu'

    # load specs file
    specs = load_specs(params["ana_dir"])

    # load time table. Puts in params dictionary
    time_table = load_time_table(params["ana_dir"])
    params["time_table"] = time_table

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    # if user_spec_fovs:
    #     fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    information("Processing %d FOVs." % len(fov_id_list))

    # create dictionary which organizes cells by fov and peak_id
    Cells_by_peak = organize_cells_by_channel(Complete_Cells, specs)

    # multiprocessing
    color_multiproc = False
    if color_multiproc:
        Cells_to_pool = []
        for fov_id in fov_id_list:
            peak_ids = Cells_by_peak[fov_id].keys()
            peak_id_Cells = Cells_by_peak[fov_id].values()
            fov_ids = [fov_id] * len(peak_ids)

            Cells_to_pool += zip(fov_ids, peak_ids, peak_id_Cells)
        pool = Pool(processes=params["num_analyzers"])

        mapfunc = partial(
            find_cell_intensities_worker, params, fl_channel, seg_method, cellfile_path
        )
        Cells_updates = pool.starmap_async(mapfunc, Cells_to_pool)
        # [pool.apply_async(find_cell_intensities(fov_id, peak_id, Cells, midline=True, channel=namespace.channel)) for fov_id in fov_id_list for peak_id, Cells in Cells_by_peak[fov_id].items()]

        pool.close()  # tells the process nothing more will be added.
        pool.join()
        update_cells = (
            Cells_updates.get()
        )  # the result is a list of Cells dictionary, each dict contains several cells
        update_cells = {
            cell_id: cell for cells in update_cells for cell_id, cell in cells.items()
        }
        for cell_id, cell in update_cells.items():
            Complete_Cells[cell_id] = cell

    # for each set of cells in one fov/peak, compute the fluorescence
    else:
        for fov_id in fov_id_list:
            if fov_id in Cells_by_peak:
                information("Processing FOV {}.".format(fov_id))
                for peak_id, Cells in Cells_by_peak[fov_id].items():
                    information("Processing peak {}.".format(peak_id))
                    find_cell_intensities(
                        params,
                        time_table,
                        fov_id,
                        peak_id,
                        Cells,
                        midline=False,
                        channel_name=fl_channel,
                    )

    # Just the complete cells, those with mother and daugther
    cell_filename = os.path.basename(cellfile_path)
    with open(params["cell_dir"] / (cell_filename[:-4] + "_fl.pkl"), "wb") as cell_file:
        pickle.dump(Complete_Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    information("Finished.")


class Colors(MM3Container):
    def create_widgets(self):
        self.cellfile_widget = FileEdit(
            label="cell_file",
            value=Path("./analysis/cell_data/complete_cells.pkl"),
            tooltip="Cell file to be analyzed",
        )

        ## allow user to choose multiple planes to analyze
        self.plane_widget = PlanePicker(
            self.valid_planes,
            label="analysis plane",
            tooltip="Fluoresence plane that you would like to analyze",
        )
        self.segmentation_method_widget = ComboBox(
            label="segmentation method", choices=["Otsu", "U-net"]
        )
        self.fov_widget = FOVChooser(self.valid_fovs)

        self.set_plane()
        self.set_segmentation_method()
        self.set_cellfile()
        self.set_fovs(self.valid_fovs)

        self.plane_widget.changed.connect(self.set_plane)
        self.segmentation_method_widget.changed.connect(self.set_segmentation_method)
        self.cellfile_widget.changed.connect(self.set_cellfile)
        self.fov_widget.connect_callback(self.set_fovs)

        self.append(self.plane_widget)
        self.append(self.segmentation_method_widget)
        self.append(self.cellfile_widget)
        self.append(self.fov_widget)

    def set_params(self):
        # These have been wittled down to bare minimum.
        self.params = {
            "experiment_name": self.experiment_name,
            "ana_dir": self.analysis_folder,
            "FOV": self.fovs,
            "fl_plane": self.fl_plane,
            "cell_file": self.cellfile,
            "num_analyzers": multiprocessing.cpu_count(),
            "output": "TIFF",
            "chnl_dir": self.analysis_folder / "channels",
            "seg_dir": self.analysis_folder / "segmented",
            "cell_dir": self.analysis_folder / "cell_data",
            "sub_dir": self.analysis_folder / "subtracted",
        }

    def run(self):
        self.set_params()
        self.viewer.window._status_bar._toggle_activity_dock(True)

        colors(self.params, self.fl_plane, self.segmentation_method, str(self.cellfile))

    def set_plane(self):
        self.fl_plane = self.plane_widget.value

    def set_segmentation_method(self):
        self.segmentation_method = self.segmentation_method_widget.value

    def set_cellfile(self):
        self.cellfile = self.cellfile_widget.value

    def set_fovs(self, fovs):
        self.fovs = list(set(fovs))
