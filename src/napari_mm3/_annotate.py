from magicgui.widgets import FileEdit, Container, SpinBox, LineEdit, PushButton
import numpy as np
import tifffile as tiff
import yaml
import os
import re
from pathlib import Path
from napari import Viewer


class MM3Container(Container):
    def __init__(self, napari_viewer):
        super().__init__()
        # TODO: Remove 'reload data' button. Make it all a bit more dynamic.
        self.viewer = napari_viewer

        self.data_directory_widget = FileEdit(
            mode="d",
            label="data directory",
            value=Path("."),
            tooltip="Directory within which all your data and analyses will be located.",
        )
        self.data_directory_widget.changed.connect(self.set_data_directory)
        self.set_data_directory()
        self.append(self.data_directory_widget)

        self.analysis_folder_widget = FileEdit(
            mode="d",
            label="analysis folder",
            tooltip="Required. Location (within working directory) for outputting analysis. If in doubt, leave as default.",
            value=Path("./analysis"),
        )
        self.analysis_folder_widget.changed.connect(self.set_analysis_folder)
        self.set_analysis_folder()
        self.append(self.analysis_folder_widget)

        self.TIFF_folder_widget = FileEdit(
            mode="d",
            label="TIFF folder",
            tooltip="Required. Location (within working directory) for the input images. If in doubt, leave as default.",
            value=Path("./TIFF"),
        )
        self.TIFF_folder_widget.changed.connect(self.set_TIFF_folder)
        # Automatically try to set the TIFF folder from the default.
        self.set_TIFF_folder()
        self.append(self.TIFF_folder_widget)

        self.experiment_name_widget = LineEdit(
            label="output prefix",
            tooltip="Optional. A prefix that will be prepended to output files. If in doubt, leave blank.",
        )
        self.experiment_name_widget.changed.connect(self.set_experiment_name)
        self.set_experiment_name()
        self.append(self.experiment_name_widget)

        self.load_data_widget = PushButton(
            label="reload data",
            tooltip="Load data from specified directories.",
        )
        self.load_data_widget.clicked.connect(self.set_valid_fovs)
        self.set_valid_fovs()
        self.append(self.load_data_widget)

    def set_data_directory(self):
        self.data_directory = self.data_directory_widget.value

    def set_analysis_folder(self):
        self.analysis_folder = self.analysis_folder_widget.value

    def set_experiment_name(self):
        self.experiment_name = self.experiment_name_widget.value

    def set_TIFF_folder(self):
        self.TIFF_folder = self.TIFF_folder_widget.value

    def set_valid_fovs(self):
        self.fovs = self.get_valid_fovs(self.TIFF_folder)

    def get_valid_fovs(self, TIFF_folder):
        found_files = TIFF_folder.glob("*.tif")
        filenames = [f.name for f in found_files]
        get_fov_regex = re.compile("xy(\d*)")
        fov_strings = set(get_fov_regex.findall(filename)[0] for filename in filenames)
        fovs = map(int, sorted(fov_strings))
        return list(fovs)


class SingleFOVChooser(SpinBox):
    def __init__(self, permitted_FOVS):
        min_FOV = min(permitted_FOVS)
        max_FOV = max(permitted_FOVS)
        label_str = f"FOV ({min_FOV}-{max_FOV})"
        super().__init__(
            label=label_str,
            tooltip="The FOV you would like to annotate.",
            min=min_FOV,
            max=max_FOV,
        )


class FOVChooser:
    def __init__(self, permitted_FOVs):
        pass


class Annotate(MM3Container):
    def __init__(self, napari_viewer: Viewer):
        super().__init__(napari_viewer)

        self.next_peak_widget = PushButton(label="next peak")
        self.next_peak_widget.tooltip = (
            "Jump to the next peak (typically the next channel)"
        )
        self.prior_peak_widget = PushButton(label="prior_peak")
        self.prior_peak_widget.tooltip = (
            "Jump to the previous peak (typically the previous channel)"
        )
        self.FOV_id_widget = SingleFOVChooser(self.fovs)
        self.FOV_id_widget.value = 1

        self.peak_id = 0
        # Need this to detect *changes* in the FOV_id_widget.
        self.FOV_id = 1

        self.next_peak_widget.clicked.connect(self.next_peak)
        self.prior_peak_widget.clicked.connect(self.prior_peak)
        self.FOV_id_widget.changed.connect(self.FOV_id_changed)

        self.append(self.next_peak_widget)
        self.append(self.prior_peak_widget)
        self.append(self.FOV_id_widget)

        self.load_data()

    def load_specs(self):
        with (self.analysis_folder / "specs.yaml").open(mode="r") as specs_file:
            return yaml.safe_load(specs_file)

    def get_cur_fov(self, specs):
        return list(specs.keys())[self.FOV_id]

    def get_cur_peak(self, specs):
        fov = self.get_cur_fov(specs)
        peak_list_in_fov = [peak for peak in specs[fov].keys() if specs[fov][peak] == 1]
        return peak_list_in_fov[self.peak_id]

    def load_data(self):
        specs = self.load_specs()
        fov = self.get_cur_fov(specs)
        peak = self.get_cur_peak(specs)

        experiment_name = self.experiment_name

        img_filename = (
            self.analysis_folder
            / "channels"
            / f"{experiment_name}_xy{fov:03d}_p{peak:04d}_c1.tif"
        )
        mask_filename = (
            self.analysis_folder
            / "segmented"
            / f"{experiment_name}_xy{fov:03d}_p{peak:04d}_seg_unet.tif"
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

    def next_peak(self):
        self.save_out()
        self.peak_id += 1
        self.load_data()

    def prior_peak(self):
        self.save_out()
        self.peak_id -= 1
        self.load_data()

    def FOV_id_changed(self):
        self.save_out()
        self.FOV_id = self.FOV_id_widget.value
        self.load_data()

    def save_out(self):
        specs = self.load_specs()
        fov = self.get_cur_fov(specs)
        peak = self.get_cur_peak(specs)
        training_dir = self.data_directory_widget.value / "training_data"
        if not os.path.isdir(training_dir):
            os.mkdir(training_dir)

        labels = self.viewer.layers[1].data.astype(np.uint8)
        fileout_name = (
            training_dir / f"{self.experiment_name}_xy{fov:03d}_p{peak:04d}_seg.tif"
        )
        tiff.imsave(fileout_name, labels)
        print("Training data saved")
