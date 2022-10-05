from magicgui.widgets import PushButton
from pathlib import Path
import numpy as np
import tifffile as tiff
import yaml
import re

from ._deriving_widgets import MM3Container, FOVChooserSingle


def load_specs(analysis_folder):
    with (analysis_folder / "specs.yaml").open(mode="r") as specs_file:
        return yaml.safe_load(specs_file)


def get_peaks(specs, fov):
    return [peak for peak in specs[fov].keys() if specs[fov][peak] == 1]


class PeakCounter:
    def __init__(self, specs, fov):
        self.specs = specs
        self.fov = fov
        self.peak_index = 0
        self.peak = get_peaks(self.specs, self.fov)[self.peak_index]

    def increment(self):
        self.peak_index += 1
        self.peak = get_peaks(self.specs, self.fov)[self.peak_index]

    def decrement(self):
        self.peak_index -= 1
        self.peak = get_peaks(self.specs, self.fov)[self.peak_index]

    def set_peak(self, peak):
        peaks = get_peaks(self.specs, self.fov)
        self.peak_index = peaks.index(peak)
        self.peak = get_peaks(self.specs, self.fov)[self.peak_index]

    def set_fov(self, fov):
        self.fov = fov
        self.peak_index = 0
        self.peak = get_peaks(self.specs, self.fov)[self.peak_index]


class Annotate(MM3Container):
    def create_widgets(self):
        """Overriding method. Serves as the widget constructor. See _deriving_widgets.MM3Container for details."""
        self.load_recent_widget.hide()
        self.run_widget.hide()

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

        self.fov_widget.connect(self.change_fov)
        self.next_peak_widget.clicked.connect(self.next_peak)
        self.prior_peak_widget.clicked.connect(self.prior_peak)
        self.save_out_widget.changed.connect(self.save_out)

        self.append(self.fov_widget)
        self.append(self.next_peak_widget)
        self.append(self.prior_peak_widget)
        self.append(self.save_out_widget)

        self.load_data()

    def next_peak(self):
        # Save current peak, update new one, display current peak.
        self.save_out()
        self.peak_cntr.increment()
        self.load_data()

    def prior_peak(self):
        # Save current peak, update new one, display current peak.
        self.save_out()
        self.peak_cntr.decrement()
        self.load_data()

    def change_fov(self):
        # Save the previous FOV. Update the current fov, reset the peak. Display the new FOV.
        self.save_out()

        self.fov = self.fov_widget.value
        self.peak_cntr.set_fov(self.fov)

        self.load_data()

    def save_out(self):
        fov = self.fov
        peak = self.peak_cntr.peak
        training_dir: Path = self.analysis_folder / "training_dir"
        if not training_dir.exists():
            training_dir.mkdir()

        labels = self.viewer.layers[1].data.astype(np.uint8)
        cur_label = labels[self.viewer.dims.current_step[0], :, :]

        fileout_name = (
            training_dir
            / f"{self.experiment_name}_xy{fov:03d}_p{peak:04d}_t{self.viewer.dims.current_step[0]:04d}_seg.tif"
        )
        tiff.imsave(fileout_name, cur_label)
        print("Training data saved")

    def load_data(self):
        fov = self.fov
        peak = self.peak_cntr.peak

        img_filename = (
            self.analysis_folder
            / "channels"
            / f"{self.experiment_name}_xy{fov:03d}_p{peak:04d}_c1.tif"
        )

        with tiff.TiffFile(img_filename) as tif:
            img_stack = tif.asarray()

        self.viewer.layers.clear()
        self.viewer.add_image(img_stack)

        training_dir = self.analysis_folder / "training_dir"
        if not training_dir.exists():
            training_dir.mkdir()

        # Load all masks from given fov/peak. Add them to viewer.
        mask_filenames = f"{self.experiment_name}_xy{fov:03d}_p{peak:04d}_t*_seg.tif"
        filenames = list(training_dir.glob(mask_filenames))
        get_numbers = re.compile(r"t(\d+)_seg.tif",re.IGNORECASE)
        timestamps = [
            int(get_numbers.findall(filename.name)[0]) for filename in filenames
        ]
        mask_stack = np.zeros(np.shape(img_stack), dtype=int)
        for timestamp, filename in zip(timestamps, filenames):
            with tiff.TiffFile(filename) as tif:
                mask_stack[timestamp, :, :] = tif.asarray()
        self.viewer.add_labels(mask_stack, name="Labels")
