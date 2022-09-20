from unicodedata import decimal
from napari import Viewer
from napari.utils.notifications import show_info
from ._function import range_string_to_indices
from datetime import datetime
from magicgui.widgets import (
    Container,
    FileEdit,
    LineEdit,
    PushButton,
    RangeEdit,
    ComboBox,
)
from collections import namedtuple
from pathlib import Path
import json
import tifffile as tiff
import re


def serialize_widget(widget):
    if isinstance(widget, RangeEdit) or isinstance(widget, TimeRangeSelector):
        print("Range edit spotted!")
        start_value = widget.start.value
        final_value = widget.stop.value
        return (start_value, final_value)

    if isinstance(widget, PushButton):
        return None

    return widget.value


def apply_seralized_widget(widget, value):
    if isinstance(widget, RangeEdit) or isinstance(widget, TimeRangeSelector):
        print("Range edit spotted!")
        widget.start.value = value[0]
        widget.stop.value = value[1]
        return

    if isinstance(widget, PushButton):
        return

    widget.value = value


def get_valid_planes(TIFF_folder):
    found_files = TIFF_folder.glob("*.tif")
    filepaths = [f for f in found_files][0]
    num_channels = tiff.imread(filepaths).shape[0]
    return [f"c{c+1}" for c in range(num_channels)]


def get_valid_fovs(TIFF_folder):
    found_files = TIFF_folder.glob("*.tif")
    filenames = [f.name for f in found_files]
    get_fov_regex = re.compile("xy(\d+)")
    fov_strings = set(get_fov_regex.findall(filename)[0] for filename in filenames)
    fovs = map(int, sorted(fov_strings))
    return list(fovs)


def get_valid_times(TIFF_folder):
    found_files = TIFF_folder.glob("*.tif")
    filenames = [f.name for f in found_files]
    get_time_regex = re.compile("t(\d+)")
    time_strings = set(get_time_regex.findall(filename)[0] for filename in filenames)
    times = list(map(int, sorted(time_strings)))
    return (min(times), max(times))


class MM3Container(Container):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.analysis_folder_widget = FileEdit(
            mode="d",
            label="analysis folder",
            tooltip="Required. Location for outputting analysis. If in doubt, leave as default.",
            value=Path("./analysis"),
        )
        self.TIFF_folder_widget = FileEdit(
            mode="d",
            label="TIFF folder",
            tooltip="Required. Location for the input images. If in doubt, leave as default.",
            value=Path("./TIFF"),
        )
        self.experiment_name_widget = LineEdit(
            label="output prefix",
            tooltip="Optional. A prefix that will be prepended to output files. If in doubt, leave blank.",
        )
        self.load_recent_widget = PushButton(
            label="load last run settings",
            tooltip="load settings from the most recent run. we look for past runs in ./.history",
        )
        self.load_data_widget = PushButton(
            label="set new directories",
            tooltip="Load data from specified directories.",
        )
        self.run_widget = PushButton(
            label="run",
        )

        self.experiment_name_widget.changed.connect(self.set_experiment_name)
        self.TIFF_folder_widget.changed.connect(self.set_TIFF_folder)
        self.analysis_folder_widget.changed.connect(self.set_analysis_folder)
        self.load_recent_widget.clicked.connect(self.load_most_recent_settings)
        self.load_data_widget.clicked.connect(self.set_valid_fovs)
        self.load_data_widget.clicked.connect(self.set_valid_planes)
        self.load_data_widget.clicked.connect(self.set_valid_times)
        self.load_data_widget.clicked.connect(self.delete_extra_widgets)
        self.load_data_widget.clicked.connect(self.load_from_data_conditional)
        self.run_widget.clicked.connect(self.save_settings)
        self.run_widget.clicked.connect(self.run_conditional)

        self.set_experiment_name()
        self.set_TIFF_folder()
        self.set_analysis_folder()
        self.set_valid_fovs()
        self.set_valid_planes()
        self.set_valid_times()

        self.append(self.experiment_name_widget)
        self.append(self.TIFF_folder_widget)
        self.append(self.analysis_folder_widget)
        self.append(self.load_recent_widget)
        self.append(self.load_data_widget)

        self.load_from_data_conditional()

    def create_widgets(self):
        """Method to override. Place all widget initialization here."""
        pass

    def run(self):
        """Method to override. Any execution methods go here."""
        pass

    def load_from_data_conditional(self):
        if self.found_planes and self.found_fovs and self.found_times:
            self.create_widgets()
            self.append(self.run_widget)

    def run_conditional(self):
        if self.found_planes and self.found_fovs and self.found_times:
            self.run()

    def is_preset_widget(self, widget):
        labels = {
            self.experiment_name_widget.label,
            self.TIFF_folder_widget.label,
            self.analysis_folder_widget.label,
            self.load_data_widget.label,
            self.load_recent_widget.label,
        }
        return widget.label in labels

    def delete_extra_widgets(self):
        """Delete any widgets that come after the 'reload directories' button.
        This allows for easy UI resets in deriving widgets (see, e.g. _track.py)"""
        while not self.is_preset_widget(self[-1]):
            self.pop()

    def set_analysis_folder(self):
        self.analysis_folder = self.analysis_folder_widget.value

    def set_experiment_name(self):
        self.experiment_name = self.experiment_name_widget.value

    def set_TIFF_folder(self):
        self.TIFF_folder = self.TIFF_folder_widget.value

    def set_valid_fovs(self):
        try:
            self.valid_fovs = get_valid_fovs(self.TIFF_folder)
            self.found_fovs = True
        except:
            self.found_fovs = False

    def set_valid_times(self):
        try:
            self.valid_times = get_valid_times(self.TIFF_folder)
            self.found_times = True
        except:
            self.found_times = False

    def set_valid_planes(self):
        try:
            self.valid_planes = get_valid_planes(self.TIFF_folder)
            self.found_planes = True
        except:
            self.found_planes = False

    def get_most_recent_run(self):
        """Gets the parameters from the most recent run of the current
        widget."""
        try:
            with open("./history.json", "r") as h:
                history = json.load(h)
        except:
            return {}
        # get the most recent run of the relevant widget.
        old_params = {}
        for historic_widget_name, _, params in reversed(history):
            if historic_widget_name == self.parent.name:
                old_params = params
                break

        return old_params

    def load_most_recent_settings(self):
        """
        Load most most recent entry in the history file that has
        name == 'widget_name'.
        Apply the saved parameters to the currently extant widgets.
        """
        old_params = self.get_most_recent_run()
        if old_params:
            # assign old_params to current widgets.
            for widget in self:
                if self.is_preset_widget(widget):
                    continue
                apply_seralized_widget(widget, old_params.get(widget.label, ""))

    def save_settings(self):
        """
        Save the current settings for all non-preset widgets.
        name == 'widget_name'.
        Apply the saved parameters to the currently extant widgets.
        """
        widget_name = self.parent.name
        history = []
        if Path("./history.json").exists():
            with open("./history.json", "r") as h:
                history = json.load(h)

        # Generate a dictionary of the current parameters.
        current_params = {}
        for widget in self:
            if self.is_preset_widget(widget):
                continue
            if isinstance(widget, PushButton):
                continue
            current_params[widget.label] = serialize_widget(widget)

        # If the most recent run has the same parameters as our current run, do nothing.
        old_params = self.get_most_recent_run()
        if old_params and old_params == current_params:
            return
        timestamp = datetime.now()
        history.append((widget_name, str(timestamp), current_params))

        with open("./history.json", "w") as h:
            json.dump(history, h, indent=2)


class TimeRangeSelector(RangeEdit):
    def __init__(self, permitted_times):
        label_str = f"time range (frames {permitted_times[0]}-{permitted_times[1]})"
        super().__init__(
            label=label_str,
            tooltip="The time range to analyze. Note that 'step' is currently not supported.",
            start=permitted_times[0],
            stop=permitted_times[1],
            min=permitted_times[0],
            max=permitted_times[1],
        )


class InteractiveSpinBox(Container):
    """Our custom version of magicgui's 'SpinBox' widget.
     * Supports floats (auto-rounds to 3 decimal points).
     * 'Atomic' updates: If an expensive (single-thread) method is called on value change, this will work
        as expected (unlike the default spinbox).
    Try to only use this in contexts where you would like to perform single-threaded operations
    upon changing a spinbox."""

    def __init__(
        self, min=0, max=99999, value=1, step=1, tooltip="", use_float=False, label=""
    ):
        super().__init__(
            layout="horizontal",
            labels=False,
            tooltip=tooltip,
        )
        self.margins = (0, 0, 0, 0)
        self.min = min
        self.max = max
        self.step = step
        self.value = value
        self.use_float = use_float
        self.name = label

        self.text_widget = LineEdit(
            value=self.value,
        )
        self.increment_widget = PushButton(label="+")
        self.decrement_widget = PushButton(label="-")

        self.text_widget.changed.connect(self.set_value)
        self.increment_widget.changed.connect(self.increment)
        self.decrement_widget.changed.connect(self.decrement)

        self.append(self.text_widget)
        self.append(self.increment_widget)
        self.append(self.decrement_widget)

    def set_value(self):
        try:
            if self.use_float:
                self.value = float(self.text_widget.value)
            else:
                self.value = int(self.text_widget.value)
        except:
            # Casting failure is not a big deal. No point throwing an exception.
            print("Failed to turn text into a number.")
            return
        # Enforce bounds on self.value
        self.value = max(self.min, self.value)
        self.value = min(self.max, self.value)

    def connect(self, func):
        self.text_widget.changed.connect(func)

    def increment(self):
        # Update internal value, then update displayed value.
        # Desyncing the 'display' and 'internal' values allows us to display
        # rounded floating points.
        self.value = self.value + self.step
        self.value = min(self.max, self.value)
        if self.use_float:
            self.text_widget.value = f"{self.value:.3f}"
        else:
            self.text_widget.value = self.value

    def decrement(self):
        # Update internal value, then update displayed value.
        self.value = self.value - self.step
        self.value = max(self.min, self.value)
        if self.use_float:
            self.text_widget.value = f"{self.value:.3f}"
        else:
            self.text_widget.value = self.value


class FOVChooserSingle(InteractiveSpinBox):
    def __init__(self, valid_fovs):
        self.min_FOV = min(valid_fovs)
        self.max_FOV = max(valid_fovs)
        super().__init__(
            label=f"FOV ({self.min_FOV}-{self.max_FOV})",
            min=self.min_FOV,
            max=self.max_FOV,
            value=self.min_FOV,
            step=1,
            tooltip="Pick an FOV",
            use_float=False,
        )


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
