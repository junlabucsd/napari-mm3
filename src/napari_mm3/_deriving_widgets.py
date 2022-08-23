from ._function import range_string_to_indices
from magicgui.widgets import Container, FileEdit, LineEdit, SpinBox, TextEdit, PushButton
from pathlib import Path
import re

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
    """
    Widget for specifying a single FOV; extends magicgui.widgets.SpinBox.
    Instead of using the standard SpinBox.changed.connect(...), use the custom 
    SingleFOVChooser.fixed_connect(...). It provides a workaround for a known Qt bug.
    """
    def __init__(self, permitted_FOVS):
        min_FOV = min(permitted_FOVS)
        max_FOV = max(permitted_FOVS)
        label_str = f"FOV ({min_FOV}-{max_FOV})"
        super().__init__(
            label=label_str,
            tooltip="The FOV you would like to work with.",
            min=min_FOV,
            max=max_FOV,
        )
    
    def connect_callback(self, func):
        """
        Use this method when giving this SpinBox a function.
        This is a workaround for a Qt bug, where if a function connected 
        to a spinbox takes too long to execute, the spinbox skips a value.
        """
        self.changed.pause()
        self.changed.connect(func)
        self.changed.resume()

class FOVChooser(TextEdit):
    """Widget for choosing multiple FOVs."""
    def __init__(self, permitted_FOVs):
        self.min_FOV = min(permitted_FOVs)
        self.max_FOV = max(permitted_FOVs)
        label_str = f"FOVs ({self.min_FOV}-{self.max_FOV})"
        value_str = f"{self.min_FOV}-{self.max_FOV}"
        super().__init__(
            label=label_str,
            value=value_str,
            tooltip="A list of FOVs to analyze. Ranges and comma separated values allowed (e.g. '1-30', '2-4,15,18'.)",
        )


    def connect_callback(self, func):
        """Replaces self.changed.connect(...).
        Interprets any text in the box as a list of FOVs.
        Thus 'func' should operate on a list of FOVs, filtered by those that actually exist in the TIFs.
        """
        def func_with_range(line):
            user_fovs = range_string_to_indices(line)
            if user_fovs:
                func(user_fovs)
        self.changed.connect(func_with_range)

