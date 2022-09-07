from napari import Viewer
from napari.utils.notifications import show_warning
from ._function import range_string_to_indices
from magicgui.widgets import (
    Container,
    FileEdit,
    LineEdit,
    SpinBox,
    PushButton,
    RangeEdit,
    ComboBox,
)
from pathlib import Path
import tifffile as tiff
import re


def get_valid_planes(TIFF_folder):
    found_files = TIFF_folder.glob("*.tif")
    filepaths = [f for f in found_files][0]
    num_channels = tiff.imread(filepaths).shape[0]
    return [f"c{c+1}" for c in range(num_channels)]


def get_valid_fovs(TIFF_folder):
    found_files = TIFF_folder.glob("*.tif")
    filenames = [f.name for f in found_files]
    get_fov_regex = re.compile("xy(\d*)")
    fov_strings = set(get_fov_regex.findall(filename)[0] for filename in filenames)
    fovs = map(int, sorted(fov_strings))
    return list(fovs)


def get_valid_times(TIFF_folder):
    found_files = TIFF_folder.glob("*.tif")
    filenames = [f.name for f in found_files]
    get_time_regex = re.compile("t(\d*)")
    time_strings = set(get_time_regex.findall(filename)[0] for filename in filenames)
    times = list(map(int, sorted(time_strings)))
    return (min(times), max(times))


class MM3Container(Container):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        # TODO: Remove 'reload data' button. Make it all a bit more dynamic.
        self.viewer = napari_viewer

        self.data_directory_widget = FileEdit(
            mode="d",
            label="data directory",
            value=Path("."),
            tooltip="Directory within which all your data and analyses will be located.",
        )
        self.analysis_folder_widget = FileEdit(
            mode="d",
            label="analysis folder",
            tooltip="Required. Location (within working directory) for outputting analysis. If in doubt, leave as default.",
            value=Path("./analysis"),
        )
        self.TIFF_folder_widget = FileEdit(
            mode="d",
            label="TIFF folder",
            tooltip="Required. Location (within working directory) for the input images. If in doubt, leave as default.",
            value=Path("./TIFF"),
        )
        self.experiment_name_widget = LineEdit(
            label="output prefix",
            tooltip="Optional. A prefix that will be prepended to output files. If in doubt, leave blank.",
        )
        self.load_data_widget = PushButton(
            label="set new directories",
            tooltip="Load data from specified directories.",
        )

        self.data_directory_widget.changed.connect(self.set_data_directory)
        self.experiment_name_widget.changed.connect(self.set_experiment_name)
        self.TIFF_folder_widget.changed.connect(self.set_TIFF_folder)
        self.analysis_folder_widget.changed.connect(self.set_analysis_folder)
        self.load_data_widget.clicked.connect(self.set_valid_fovs)
        self.load_data_widget.clicked.connect(self.set_valid_planes)
        self.load_data_widget.clicked.connect(self.set_valid_times)

        self.set_data_directory()
        self.set_experiment_name()
        self.set_TIFF_folder()
        self.set_analysis_folder()
        self.set_valid_fovs()
        self.set_valid_planes()
        self.set_valid_times()

        self.append(self.data_directory_widget)
        self.append(self.experiment_name_widget)
        self.append(self.TIFF_folder_widget)
        self.append(self.analysis_folder_widget)
        self.append(self.load_data_widget)

    def set_data_directory(self):
        self.data_directory = self.data_directory_widget.value

    def set_analysis_folder(self):
        self.analysis_folder = self.analysis_folder_widget.value

    def set_experiment_name(self):
        self.experiment_name = self.experiment_name_widget.value

    def set_TIFF_folder(self):
        self.TIFF_folder = self.TIFF_folder_widget.value
        if not self.TIFF_folder.exists():
            show_warning(
                "No TIFF folder found.\n"
                + "Did you make sure to launch napari from your experiment's directory?"
            )

    def set_valid_fovs(self):
        try:
            self.valid_fovs = get_valid_fovs(self.TIFF_folder)
        except:
            self.valid_fovs = ["unknown"]

    def set_valid_times(self):
        try: 
            self.valid_times = get_valid_times(self.TIFF_folder)
        except:
            self.valid_times = ["unknown"]

    def set_valid_planes(self):
        try: 
            self.valid_planes = get_valid_planes(self.TIFF_folder)
        except: 
            self.valid_planes = []


class TimeRangeSelector(RangeEdit):
    def __init__(self, permitted_times):
        label_str = f"time range (frames {permitted_times[0]}-{permitted_times[1]})"
        super().__init__(
            label=label_str,
            tooltip="The time range to analyze",
            start=permitted_times[0],
            stop=permitted_times[1],
            min=permitted_times[0],
            max=permitted_times[1],
        )


class FOVChooserSingle(Container):
    def __init__(self, valid_fovs):
        super().__init__(layout="horizontal", labels=False)
        self.valid_fovs = valid_fovs
        if valid_fovs != []:
            min_FOV = min(self.valid_fovs)
            max_FOV = max(self.valid_fovs)
            self.name = f"FOV ({min_FOV}-{max_FOV})"
        else:
            min_FOV = "not found"
            max_FOV = "not found"
            self.name = f"FOV (unknown-unknown)"
        self.pick_fov_widget = LineEdit(
            value=min_FOV,
            tooltip="The FOV you would like to work with.",
        )
        self.next_fov_widget = PushButton(label="+")
        self.prev_fov_widget = PushButton(label="-")

        self.pick_fov_widget.changed.connect(self.set_fov)
        self.next_fov_widget.changed.connect(self.increment_fov)
        self.prev_fov_widget.changed.connect(self.decrement_fov)

        self.append(self.pick_fov_widget)
        self.append(self.next_fov_widget)
        self.append(self.prev_fov_widget)

        self.fov = min_FOV

    def set_fov(self):
        try:
            self.fov = int(self.pick_fov_widget.value)
        except:
            # Int casting failure is not a big deal. No point throwing an exception.
            print("Attempted to access invalid FOV")
            return
        # Enforce bounds on self.fov
        self.fov = max(min(self.valid_fovs), self.fov)
        self.fov = min(max(self.valid_fovs), self.fov)
        # After setting bounds, update UI accordingly
        self.pick_fov_widget.value = self.fov

    def connect_callback(self, func):
        self.pick_fov_widget.changed.connect(func)

    def increment_fov(self):
        # Update UI; then, act as if the user changed the number manually
        self.pick_fov_widget.value = self.fov + 1
        self.set_fov()

    def decrement_fov(self):
        # Update UI; then, act as if the user changed the number manually
        self.pick_fov_widget.value = self.fov - 1
        self.set_fov()


class PlanePicker(ComboBox):
    def __init__(
        self,
        permitted_planes,
        label="microscopy plane",
        tooltip="The plane you would like to use.",
    ):
        super().__init__(label=label, choices=permitted_planes, tooltip=tooltip)


class FOVChooser(LineEdit):
    """Widget for choosing multiple FOVs."""

    def __init__(self, permitted_FOVs):
        print(permitted_FOVs)
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

        def func_with_range():
            user_fovs = range_string_to_indices(self.value)
            if user_fovs:
                func(user_fovs)

        self.changed.connect(func_with_range)