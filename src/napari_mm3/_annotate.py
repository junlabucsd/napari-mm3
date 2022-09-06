from magicgui.widgets import PushButton
import numpy as np
import tifffile as tiff
import yaml
import os
from napari import Viewer

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
    def __init__(self, napari_viewer: Viewer):
        super().__init__(napari_viewer)
        self.create_widgets()
        self.load_data_widget.clicked.connect(self.delete_widgets)
        self.load_data_widget.clicked.connect(self.create_widgets)

    def create_widgets(self):
        """Serves as the widget constructor."""
        self.fov_widget = FOVChooserSingle(self.valid_fovs)
        self.next_peak_widget = PushButton(label="next peak", tooltip="Jump to the next peak (typically the next channel)")
        self.prior_peak_widget = PushButton(label="prior_peak", tooltip = "Jump to the previous peak (typically the previous channel)")
        self.save_out_widget = PushButton(label="save", tooltip = "save the current label")


        self.fov = self.fov_widget.fov
        self.peak_cntr = PeakCounter(load_specs(self.analysis_folder), self.fov)

        self.fov_widget.connect_callback(self.change_fov)
        self.next_peak_widget.clicked.connect(self.next_peak)
        self.prior_peak_widget.clicked.connect(self.prior_peak)
        self.save_out_widget.changed.connect(self.save_out)

        self.append(self.fov_widget)
        self.append(self.next_peak_widget)
        self.append(self.prior_peak_widget)
        self.append(self.save_out_widget)

        self.load_data()

    def delete_widgets(self):
        """Serves as the widget destructor. See MM3Container for more details."""
        self.pop() # Pop fov_widget
        self.pop() # Pop next_peak button
        self.pop() # Pop prior_peak button
        self.pop() # Pop sav_out button.


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

        self.fov = self.fov_widget.fov
        self.peak_cntr.set_fov(self.fov)

        self.load_data()

    def save_out(self):
        fov = self.fov
        peak = self.peak_cntr.peak
        training_dir = self.data_directory_widget.value / "training_data"
        if not os.path.isdir(training_dir):
            os.mkdir(training_dir)

        labels = self.viewer.layers[1].data.astype(np.uint8)
        fileout_name = (
            training_dir / f"{self.experiment_name}_xy{fov:03d}_p{peak:04d}_seg.tif"
        )
        tiff.imsave(fileout_name, labels)
        print("Training data saved")

    def load_data(self):
        fov = self.fov
        peak = self.peak_cntr.peak

        img_filename = (
            self.analysis_folder
            / "channels"
            / f"{self.experiment_name}_xy{fov:03d}_p{peak:04d}_c1.tif"
        )
        mask_filename = (
            self.analysis_folder
            / "segmented"
            / f"{self.experiment_name}_xy{fov:03d}_p{peak:04d}_seg_otsu.tif"
        )

        with tiff.TiffFile(mask_filename) as tif:
            mask_stack = tif.asarray()
        with tiff.TiffFile(img_filename) as tif:
            img_stack = tif.asarray()

        self.viewer.layers.clear()
        self.viewer.add_image(img_stack)

        try:
            self.viewer.add_labels(mask_stack, name="Labels")
        except:
            pass

        current_layers = [l.name for l in self.viewer.layers]

        if not "Labels" in current_layers:
            empty = np.zeros(np.shape(img_stack), dtype=int)
            self.viewer.add_labels(empty, name="Labels")