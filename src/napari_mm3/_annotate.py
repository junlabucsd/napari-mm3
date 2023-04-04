from magicgui.widgets import PushButton, ComboBox, FileEdit
from pathlib import Path
import numpy as np
import tifffile as tiff
import yaml
import re
import napari

from ._deriving_widgets import (
    MM3Container,
    FOVChooserSingle,
    PlanePicker,
    warning,
    load_tiff,
)


def load_specs(analysis_folder):
    """
    Loads the specifications YAML file for the analysis and returns its contents as a dictionary.

    Parameters
    ----------
    analysis_folder : Path
        The path to the analysis folder.

    Returns
    -------
    dict
        The contents of the specifications YAML file as a dictionary.
    """
    with (analysis_folder / "specs.yaml").open(mode="r") as specs_file:
        return yaml.safe_load(specs_file)


def get_peaks(specs, fov):
    """
    Returns a list of peaks for a given FOV.

    Parameters
    ----------
    specs : dict
        A dictionary containing the specifications for the analysis.
    fov : str
        The name of the field of view.

    Returns
    -------
    list
        A list of peak names for the given FOV.
    """
    return [peak for peak in specs[fov].keys() if specs[fov][peak] == 1]


class PeakCounter:
    """
    A class that keeps track of the current peak being annotated.

    Parameters
    ----------
    specs : dict
        A dictionary containing the specifications for the analysis.
    fov : str
        The name of the field of view.
    """

    def __init__(self, specs, fov):
        self.specs = specs
        self.fov = fov
        self.peak_index = 0
        self.peak = get_peaks(self.specs, self.fov)[self.peak_index]

    def increment(self):
        """
        Increments the peak index and updates the current peak.
        """
        self.peak_index += 1
        self.peak = get_peaks(self.specs, self.fov)[self.peak_index]

    def decrement(self):
        """
        Decrements the peak index and updates the current peak.
        """
        self.peak_index -= 1
        self.peak = get_peaks(self.specs, self.fov)[self.peak_index]

    def set_peak(self, peak):
        """
        Sets the current peak to the specified peak.

        Parameters
        ----------
        peak : str
            The name of the peak to set as the current peak.
        """
        peaks = get_peaks(self.specs, self.fov)
        self.peak_index = peaks.index(peak)
        self.peak = get_peaks(self.specs, self.fov)[self.peak_index]

    def set_fov(self, fov):
        """
        Sets the current FOV to the specified FOV.

        Parameters
        ----------
        fov : str
            The name of the FOV to set as the current FOV.
        """
        self.fov = fov
        self.peak_index = 0
        self.peak = get_peaks(self.specs, self.fov)[self.peak_index]


class Annotate(MM3Container):
    """
    A class that handles the user interface for manual annotation.
    """

    def create_widgets(self):
        """Overriding method. Serves as the widget constructor. See _deriving_widgets.MM3Container for details."""
        self.load_recent_widget.hide()
        self.run_widget.hide()

        self.data_source_widget = ComboBox(
            label="Data source", choices=["Stack", "Manual annotation"]
        )

        self.plane_picker_widget = PlanePicker(self.valid_planes, label="phase plane")

        self.mask_source_widget = ComboBox(
            label="Mask source", choices=["None", "Otsu", "unet"]
        )

        self.fov_widget = FOVChooserSingle(self.valid_fovs)
        self.next_peak_widget = PushButton(
            label="next peak",
            tooltip="Jump to the next peak (typically the next channel)",
        )
        self.prior_peak_widget = PushButton(
            label="prior_peak",
            tooltip="Jump to the previous peak (typically the previous channel)",
        )
        self.save_out_widget = PushButton(
            label="save", tooltip="save the current label"
        )

        self.fov = self.fov_widget.value
        self.peak_cntr = PeakCounter(load_specs(self.analysis_folder), self.fov)

        self.data_source_widget.changed.connect(self.set_data_source)
        self.plane_picker_widget.changed.connect(self.set_phase_plane)
        self.fov_widget.connect(self.change_fov)
        self.next_peak_widget.clicked.connect(self.next_peak)
        self.prior_peak_widget.clicked.connect(self.prior_peak)
        self.save_out_widget.changed.connect(self.save_out)
        self.mask_source_widget.changed.connect(self.set_mask_source)

        self.append(self.data_source_widget)
        self.append(self.plane_picker_widget)
        self.append(self.mask_source_widget)
        self.append(self.fov_widget)
        self.append(self.next_peak_widget)
        self.append(self.prior_peak_widget)
        self.append(self.save_out_widget)

        self.mask_src = self.mask_source_widget.value
        self.phase_plane = self.plane_picker_widget.value
        self.set_data_source()
        self.load_data()

    def set_data_source(self):
        self.data_source = self.data_source_widget.value

    def set_phase_plane(self):
        self.phase_plane = self.plane_picker_widget.value
        self.load_data()

    def next_peak(self):
        # Save current peak, update new one, display current peak.
        # self.save_out()
        self.peak_cntr.increment()
        self.load_data()

    def prior_peak(self):
        # Save current peak, update new one, display current peak.
        # self.save_out()
        self.peak_cntr.decrement()
        self.load_data()

    def change_fov(self):
        # Save the previous FOV. Update the current fov, reset the peak. Display the new FOV.
        # self.save_out()

        self.fov = self.fov_widget.value
        self.peak_cntr.set_fov(self.fov)

        self.load_data()

    # def set_image_source(self):
    #     self.image_src = self.image_source_widget.value
    #     self.load_data()

    def set_mask_source(self):
        self.mask_src = self.mask_source_widget.value
        self.load_data()

    def save_out(self):
        # save mask and raw image
        self.save_out_mask()
        self.save_out_image()

    def save_out_mask(self):
        # Save segmentation mask
        fov = self.fov
        peak = self.peak_cntr.peak
        training_dir: Path = self.analysis_folder / "training" / "masks"
        if not training_dir.exists():
            training_dir.mkdir(parents=True)

        labels = self.viewer.layers[1].data.astype(np.uint8)
        cur_label = labels[self.viewer.dims.current_step[0], :, :]
        cur_label = np.array(cur_label > 0, dtype=np.uint8)

        fileout_name = (
            training_dir
            / f"{self.experiment_name}_xy{fov:03d}_p{peak:04d}_t{self.viewer.dims.current_step[0]:04d}.tif"
        )
        tiff.imsave(fileout_name, cur_label)
        print("Training data saved")

    def save_out_image(self):
        # Save raw image in parallel with mask
        fov = self.fov
        peak = self.peak_cntr.peak
        training_dir: Path = self.analysis_folder / "training" / "images"
        if not training_dir.exists():
            training_dir.mkdir(parents=True)

        img = self.viewer.layers[0].data
        cur_img = img[self.viewer.dims.current_step[0], :, :]

        fileout_name = (
            training_dir
            / f"{self.experiment_name}_xy{fov:03d}_p{peak:04d}_t{self.viewer.dims.current_step[0]:04d}.tif"
        )
        tiff.imsave(fileout_name, cur_img)

    def load_data(self):
        self.viewer.layers.clear()

        if self.data_source == "Stack":
            self.load_from_stack()
        else:
            self.load_from_existing()

    def load_from_existing(self):
        # Load training data and masks from manually annotated (single page) tiffs
        fov = self.fov
        peak = self.peak_cntr.peak

        img_dir = self.analysis_folder / "training" / "images"
        # img_files = f"{self.experiment_name}_xy{fov:03d}_p{peak:04d}_t*.tif"
        img_files = f"{self.experiment_name}_xy{fov:03d}_p{peak:04d}_t*.tif"

        img_files = sorted(list(img_dir.glob(img_files)))

        mask_dir = self.analysis_folder / "training" / "masks"

        if not mask_dir.exists():
            mask_dir.mkdir(parents=True)

        # Load all masks from given fov/peak. Add them to viewer.
        mask_files = f"{self.experiment_name}_xy{fov:03d}_p{peak:04d}_t*.tif"
        # mask_files = f"{self.experiment_name}_*.tif"
        mask_files = sorted(list(mask_dir.glob(mask_files)))

        img_stack = np.stack([load_tiff(f) for f in img_files])
        mask_stack = np.stack([load_tiff(m) for m in mask_files])

        self.viewer.add_image(img_stack)
        self.viewer.add_labels(mask_stack)

    def load_from_stack(self):
        # Load training data and masks from tiff stacks output by Compile & Segment widgets
        fov = self.fov
        peak = self.peak_cntr.peak

        img_dir = self.analysis_folder / "channels"

        img_filename = (
            img_dir
            / f"{self.experiment_name}_xy{fov:03d}_p{peak:04d}_{self.phase_plane}.tif"
        )

        with tiff.TiffFile(img_filename) as tif:
            img_stack = tif.asarray()

        self.viewer.add_image(img_stack)

        mask_dir = self.analysis_folder / "training" / "masks"
        if not mask_dir.exists():
            mask_dir.mkdir(parents=True)

        # Load all masks from given fov/peak. Add them to viewer.
        mask_filenames = f"{self.experiment_name}_xy{fov:03d}_p{peak:04d}_t*.tif"
        filenames = list(mask_dir.glob(mask_filenames))

        if self.mask_src != "None":
            img_filename = (
                self.analysis_folder
                / "segmented"
                / f"{self.experiment_name}_xy{self.fov:03d}_p{self.peak_cntr.peak:04d}_seg_{self.mask_src}.tif"
            )
            try:
                with tiff.TiffFile(img_filename) as tif:
                    mask_stack = tif.asarray()
                self.viewer.add_labels(mask_stack, name="Labels")
            except:
                warning("Could not load segmentation masks")
                mask_stack = self.make_masks(filenames, img_stack)
                self.viewer.add_labels(mask_stack, name="Labels")
        else:
            mask_stack = self.make_masks(filenames, img_stack)
            self.viewer.add_labels(mask_stack, name="Labels")

    def make_masks(self, filenames, img_stack):
        get_numbers = re.compile(r"t(\d+).tif", re.IGNORECASE)
        timestamps = [
            int(get_numbers.findall(filename.name)[0]) for filename in filenames
        ]
        mask_stack = np.zeros(np.shape(img_stack), dtype=int)
        for timestamp, filename in zip(timestamps, filenames):
            with tiff.TiffFile(filename) as tif:
                mask_stack[timestamp, :, :] = tif.asarray()
        return mask_stack
