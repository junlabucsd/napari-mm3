from magicgui.widgets import FileEdit, Container, SpinBox, LineEdit, PushButton
import numpy as np
import tifffile as tiff
import yaml
import os


class Annotate(Container):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.data_directory_widget = FileEdit(mode="d", label="data directory")
        self.data_directory_widget.tooltip = (
            "Directory within which all your data and analyses will be located."
        )
        # TODO: Make this auto-inferred from a function type signature.
        self.experiment_name_widget = LineEdit(label="output prefix")
        self.experiment_name_widget.tooltip = (
            "Optional. A prefix that will be prepended to output files."
        )
        self.analysis_folder_widget = LineEdit(label="analysis folder")
        self.analysis_folder_widget.tooltip = "Required. Location (within working directory) for outputting analysis. 'working directory/analysis/' by default."
        self.analysis_folder_widget.value = "analysis"
        self.next_peak_widget = PushButton(label="next peak")
        self.next_peak_widget.tooltip = (
            "Jump to the next peak (typically the next channel)"
        )
        self.prior_peak_widget = PushButton(label="prior_peak")
        self.prior_peak_widget.tooltip = (
            "Jump to the previous peak (typically the previous channel)"
        )
        self.FOV_id_widget = SpinBox(label="FOV")
        self.FOV_id_widget.tooltip = "The FOV you would like to annotate."
        self.FOV_id_widget.value = 1

        self.peak_id = 0
        # Need this to detect *changes* in the FOV_id_widget.
        self.FOV_id = 1

        self.data_directory_widget.changed.connect(self.load_data)
        self.analysis_folder_widget.changed.connect(self.load_data)
        self.next_peak_widget.clicked.connect(self.next_peak)
        self.prior_peak_widget.clicked.connect(self.prior_peak)
        self.FOV_id_widget.changed.connect(self.FOV_id_changed)

        self.insert(0, self.data_directory_widget)
        self.insert(1, self.analysis_folder_widget)
        self.insert(2, self.experiment_name_widget)
        self.insert(3, self.next_peak_widget)
        self.insert(4, self.prior_peak_widget)
        self.insert(5, self.FOV_id_widget)

    def load_specs(self):
        with (
            self.data_directory_widget.value
            / self.analysis_folder_widget.value
            / "specs.yaml"
        ).read("r") as specs_file:
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

        data_directory = self.data_directory_widget.value
        experiment_name = self.experiment_name_widget.value

        img_filename = (
            data_directory
            / self.analysis_folder_widget.value
            / "channels"
            / f"{experiment_name}_xy{fov:03d}_p{peak:04d}_c1.tif"
        )
        mask_filename = (
            data_directory
            / self.analysis_folder_widget.value
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
            training_dir
            / f"{self.experiment_name_widget.value}_xy{fov:03d}_p{peak:04d}_seg.tif"
        )
        tiff.imsave(fileout_name, labels)
        print("Training data saved")
