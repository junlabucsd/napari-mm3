from datetime import datetime
from magicgui.widgets import (
    Container,
    FileEdit,
    LineEdit,
    PushButton,
    RangeEdit,
    ComboBox,
)
from pathlib import Path
from .utils import TIFF_FILE_FORMAT_PEAK
from magicgui.types import FileDialogMode
import h5py
import pickle
import yaml
import json
import tifffile as tiff
import re
import time
import sys
import traceback
from enum import Enum
from napari import Viewer
import numpy as np


class SegmentationMode(Enum):
    OTSU = 1
    UNET = 2


# print a warning
def warning(*objs):
    print(time.strftime("%H:%M:%S WARNING:", time.localtime()), *objs, file=sys.stderr)


def information(*objs):
    print(time.strftime("%H:%M:%S", time.localtime()), *objs, file=sys.stdout)


def load_tiff(tiff_location: Path):
    with tiff.TiffFile(tiff_location) as tif:
        return tif.asarray()


def load_hdf5(hdf5_location: Path, dataset_name: str):
    with h5py.File(hdf5_location, "r") as h5f:
        return h5f[dataset_name]


def load_specs(analysis_dir: Path) -> dict:
    """
    Load specs file which indicates which channels should be analyzed,
    used as empties, or ignored.
    """
    try:
        with (analysis_dir / "specs.yaml").open("r") as specs_file:
            specs = yaml.safe_load(specs_file)
    except:
        try:
            with (analysis_dir / "specs.pkl").open("rb") as specs_file:
                specs = pickle.load(specs_file)
        except ValueError:
            warning("Could not load specs file.")

    return specs


# load the time table
def load_time_table(ana_dir: Path) -> dict:
    """
    Load the time table.
    This is so it can be used during Cell creation.
    """

    # try first for yaml, then for pkl
    try:
        with open(ana_dir / "timetable.json", "r") as time_table_file:
            timetable = json.load(time_table_file)  # loads in keys as strings.
            new_timetable = {}
            for fov, time_idxs in timetable.items():
                new_timetable[int(fov)] = {}
                for time_idx, real_time in time_idxs.items():
                    new_timetable[int(fov)][int(time_idx)] = real_time
            return new_timetable
    except FileNotFoundError:
        with open(ana_dir / "time_table.yaml", "rb") as time_table_file:
            return yaml.safe_load(time_table_file)


def get_valid_planes(TIFF_folder):
    """Get the valid planes from a TIFF folder.
    This is done by looking for all files with the .tif extension.
    Then, the first file is loaded and the number of dimensions is checked.
    If there are 2 dimensions, then there is only one plane.
    If there are 3 dimensions, then there are multiple planes.
    If there are more than 3 dimensions, raise an error (currently no support for z stacks)

    Parameters
    ----------
    TIFF_folder : Path
        The path to the TIFF folder

    Returns
    -------
    valid_planes : list
        A list of strings indicating the valid imaging planes
    """
    found_files = TIFF_folder.glob("*.tif")
    filepaths = [f for f in found_files]
    if len(filepaths) == 0:
        raise ValueError(f"No TIFF files found in '{TIFF_folder}'.")
    # pull out first tiff to extract dims
    filepath = filepaths[0]
    test_file = tiff.imread(filepath)
    test_file = np.squeeze(test_file)
    dim = test_file.ndim
    if dim == 3:
        # there are multiple planes
        num_channels = test_file.shape[0]
    elif dim == 2:
        pattern = r"(c\d+)"
        num_channels = len(
            set(
                [re.search(pattern, str(f), re.IGNORECASE).group(1) for f in filepaths]
            )  # type:ignore
        )
    else:
        raise ValueError(f"Expected 2 or 3 dimensions but found {dim}.")

    return [f"c{c+1}" for c in range(num_channels)]


def get_valid_fovs_specs(analysis_folder):
    specs = load_specs(analysis_folder)
    all_fovs = [int(key) for key in specs.keys()]
    return list(sorted(all_fovs))


def get_valid_fovs_folder(TIFF_folder):
    found_files = TIFF_folder.glob("*.tif")
    filenames = [f.name for f in found_files]
    get_fov_regex = re.compile(r"xy(\d+)", re.IGNORECASE)
    fov_strings = set(get_fov_regex.findall(filename)[0] for filename in filenames)
    fovs = map(int, sorted(fov_strings))
    return list(fovs)


def get_valid_times(TIFF_folder):
    found_files = list(TIFF_folder.glob("*.tif"))
    filenames = [f.name for f in found_files]
    if len(filenames) == 0:
        raise ValueError(f"No files found in '{TIFF_folder}'")
    get_time_regex = re.compile(r"t(\d+)", re.IGNORECASE)

    try:
        time_strings = set(
            get_time_regex.findall(filename)[0] for filename in filenames
        )
        times = list(map(int, sorted(time_strings)))
    except IndexError:
        img = load_tiff(found_files[0])
        return (1, img.shape[0])
    return (min(times), max(times))


def _serialize_widget(widget):
    if isinstance(widget, RangeEdit) or isinstance(widget, TimeRangeSelector):
        print("Range edit spotted!")
        start_value = widget.start.value
        final_value = widget.stop.value
        return (start_value, final_value)

    if isinstance(widget, PushButton):
        return None

    if isinstance(widget, FileEdit):
        print(str(widget.value))
        return str(widget.value)

    return widget.value


def _apply_seralized_widget(widget, value):
    if isinstance(widget, RangeEdit) or isinstance(widget, TimeRangeSelector):
        print("Range edit spotted!")
        widget.start.value = value[0]
        widget.stop.value = value[1]
        return

    if isinstance(widget, PushButton):
        return

    widget.value = value


def range_string_to_indices(range_string):
    """Convert a range string to a list of indices."""
    try:
        range_string = range_string.replace(" ", "")
        split = range_string.split(",")
        indices = []
        for items in split:
            # If it's a range
            if "-" in items:
                limits = list(map(int, items.split("-")))
                if len(limits) == 2:
                    # Make it an inclusive range, as users would expect
                    limits[1] += 1
                    indices += list(range(limits[0], limits[1]))
            # If it's a single item.
            else:
                indices += [int(items)]
        print("Index range string valid!")
        return indices
    except:
        print(
            "Index range string invalid. Returning empty range until a new string is specified."
        )
        return []


class DataclassWidget(Container):
    def __init__(self, dataclass):
        for annotation in dataclass.annotations:
            pass


class MM3Container(Container):
    """
    Preset class for MM3 widgets.
    In order to use, extend the class and override the following methods:
        * create_widgets: This is a constructor. If the MM3Container finds valid TIFFs, these widgets will be added to the UI
        * run: This is the function that will be executed when the user clicks the 'run' button.
    This class supplies the followng fields from the user directly:
        * experiment_name (self-explanatory)
        * analysis_folder (the location to which outputs will be written)
        * TIFF_folder (the folder in which it will look for input TIFFs)
    It will also acquire the following metadata for you:
        * valid_fovs (a range of valid fovs)
        * valid_times (a range of valid times)
        * valid_planes (a set of valid microscopy (eg, phase, fluorescence, etc))
    Finally, it will also automatically write any 'runs' to history.json, and give you the ability to restore the most recent run's settings.
    """

    def __init__(self, napari_viewer: Viewer, validate_folders: bool = True):
        super().__init__()
        self.viewer = napari_viewer
        self.validate_folders = validate_folders

        self.analysis_folder_widget = FileEdit(
            mode=FileDialogMode.EXISTING_DIRECTORY,
            label="analysis folder",
            tooltip="Required. Location for outputting analysis. If in doubt, leave as default.",
            value=Path(".") / "analysis",
        )
        self.TIFF_folder_widget = FileEdit(
            mode=FileDialogMode.EXISTING_DIRECTORY,
            label="TIFF folder",
            tooltip="Required. Location for the input images. If in doubt, leave as default.",
            value=Path(".") / "TIFF",
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

        self.experiment_name_widget.changed.connect(self._set_experiment_name)
        self.TIFF_folder_widget.changed.connect(self._set_TIFF_folder)
        self.analysis_folder_widget.changed.connect(self._set_analysis_folder)
        self.load_recent_widget.clicked.connect(self._load_most_recent_settings)
        self.load_data_widget.clicked.connect(self._set_valid_fovs)
        self.load_data_widget.clicked.connect(self._set_valid_planes)
        self.load_data_widget.clicked.connect(self._set_valid_times)
        self.load_data_widget.clicked.connect(self._delete_extra_widgets)
        self.load_data_widget.clicked.connect(self._load_from_data_conditional)
        self.run_widget.clicked.connect(self._save_settings)
        self.run_widget.clicked.connect(self._run_conditional)

        self._set_experiment_name()
        self._set_TIFF_folder()
        self._set_analysis_folder()
        self._set_valid_fovs()
        self._set_valid_planes()
        self._set_valid_times()

        self.append(self.experiment_name_widget)
        self.append(self.TIFF_folder_widget)
        self.append(self.analysis_folder_widget)
        self.append(self.load_recent_widget)
        self.append(self.load_data_widget)

        self._load_from_data_conditional()

    def create_widgets(self):
        """Method to override. Place all widget initialization here."""
        pass

    def run(self):
        """Method to override. Any execution methods go here."""
        pass

    def _load_from_data_conditional(self):
        if self.validate_folders and not self._validate_folders():
            print("A folder validation was requested but not successful.\n")
            print("Limited traceback:")
            traceback.print_stack(limit=1)
            return
        if self.found_planes and self.found_fovs and self.found_times:
            self.create_widgets()
            self.append(self.run_widget)
            return
        print("Failed to find a key piece of info:")
        print(f"planes found: {self.found_planes}")
        print(f"fovs found: {self.found_fovs}")
        print(f"times found: {self.found_times}")
        print("Limited traceback:")
        traceback.print_stack(limit=1)

    def _run_conditional(self):
        if self.found_planes and self.found_fovs and self.found_times:
            self.run()

    def _is_preset_widget(self, widget):
        labels = {
            self.experiment_name_widget.label,
            self.TIFF_folder_widget.label,
            self.analysis_folder_widget.label,
            self.load_data_widget.label,
            self.load_recent_widget.label,
            "Interactive Peak Picker",
        }
        return widget.label in labels

    def _delete_extra_widgets(self):
        """Delete any widgets that come after the 'reload directories' button.
        This allows for easy UI resets in deriving widgets (see, e.g. _track.py)"""
        while not self._is_preset_widget(self[-1]):
            self.pop()

    def _set_analysis_folder(self):
        self.analysis_folder = Path(self.analysis_folder_widget.value)  # type:ignore

    def _set_experiment_name(self):
        self.experiment_name = str(self.experiment_name_widget.value)  # type:ignore

    def _set_TIFF_folder(self):
        self.TIFF_folder = Path(self.TIFF_folder_widget.value)  # type:ignore

    def _set_valid_fovs(self):
        try:
            try:
                self.valid_fovs = get_valid_fovs_specs(self.analysis_folder)
            except:
                self.valid_fovs = get_valid_fovs_folder(self.TIFF_folder)
            self.found_fovs = True
        except:
            self.found_fovs = False

    def _set_valid_times(self):
        try:
            self.valid_times = get_valid_times(self.TIFF_folder)
            self.found_times = True
        except ValueError as e:
            try:
                self.valid_times = get_valid_times(self.analysis_folder / "subtracted")
                self.found_times = True
            except ValueError as e:
                warning(f"Failed to get times: {e}")
                self.found_times = False

    def _set_valid_planes(self):
        try:
            self.valid_planes = get_valid_planes(self.TIFF_folder)
            self.found_planes = True
        except ValueError as e:
            try:
                self.valid_planes = get_valid_planes(
                    self.analysis_folder / "subtracted"
                )
                self.found_planes = True
            except ValueError as e:
                warning(f"Failed to get planes: {e}")
                self.found_planes = False

    def _validate_folders(self):
        return self.TIFF_folder.exists() or self.analysis_folder.exists()

    def _get_most_recent_run(self):
        """
        Gets the parameters from the most recent run of the current
        widget.
        """
        try:
            with open("./history.json", "r") as h:
                history = json.load(h)
        except:
            return {}
        # get the most recent run of the relevant widget.
        old_params = {}
        try:
            for historic_widget_name, _, params in reversed(history):
                if historic_widget_name == self.parent.name:
                    old_params = params
                    break
        except AttributeError:
            pass

        return old_params

    def _load_most_recent_settings(self):
        """
        Load most most recent entry in the history file that has
        name == 'widget_name'.
        Apply the saved parameters to the currently extant widgets.
        """
        old_params = self._get_most_recent_run()
        if old_params:
            # assign old_params to current widgets.
            for widget in self:
                if self._is_preset_widget(widget):
                    continue
                _apply_seralized_widget(widget, old_params.get(widget.label, ""))

    def _save_settings(self):
        """
        Save the current settings for all non-preset widgets.
        name == 'widget_name'.
        Apply the saved parameters to the currently extant widgets.
        """
        try:
            widget_name = self.parent.name
        except AttributeError:
            warning("Could not extract widget name to save settings.")
            widget_name = ""
        history = []
        if Path("./history.json").exists():
            with open("./history.json", "r") as h:
                history = json.load(h)

        # Generate a dictionary of the current parameters.
        current_params = {}
        for widget in self:
            if self._is_preset_widget(widget):
                continue
            if isinstance(widget, PushButton):
                continue
            current_params[widget.label] = _serialize_widget(widget)

        # If the most recent run has the same parameters as our current run, do nothing.
        old_params = self._get_most_recent_run()
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
    """
    Our custom version of magicgui's 'SpinBox' widget.
     * Supports floats (auto-rounds to 3 decimal points).
     * 'Atomic' updates: If an expensive (single-thread) method is called on value change, this will work
        as expected (unlike the default spinbox).
    Try to only use this in contexts where you would like to perform single-threaded operations
    upon changing a spinbox.
    """

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
            value=str(self.value),
        )
        self.increment_widget = PushButton(label="+")
        self.decrement_widget = PushButton(label="-")

        self.text_widget.changed.connect(self._set_value)
        self.increment_widget.changed.connect(self._increment)
        self.decrement_widget.changed.connect(self._decrement)

        self.append(self.text_widget)
        self.append(self.increment_widget)
        self.append(self.decrement_widget)

    def connect(self, func):
        self.text_widget.changed.connect(func)

    def _set_value(self):
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

    def _increment(self):
        # Update internal value, then update displayed value.
        # Desyncing the 'display' and 'internal' values allows us to display
        # rounded floating points.
        self.value = self.value + self.step
        self.value = min(self.max, self.value)
        if self.use_float:
            self.text_widget.value = f"{self.value:.3f}"
        else:
            self.text_widget.value = str(self.value)

    def _decrement(self):
        # Update internal value, then update displayed value.
        self.value = self.value - self.step
        self.value = max(self.min, self.value)
        if self.use_float:
            self.text_widget.value = f"{self.value:.3f}"
        else:
            self.text_widget.value = str(self.value)


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


class InteractivePeakChooser(Container):
    def __init__(self, valid_fovs, fov_choices):
        super().__init__(
            layout="horizontal",
            labels=True,
            tooltip="Interactive peak picker.",
        )

        self.label = "Interactive Peak Picker"

        self.valid_fovs = valid_fovs
        self.fov_choices = fov_choices
        self.cur_fov = min(self.valid_fovs)
        self.cur_peak = min(fov_choices[self.cur_fov])

        self.fov_chooser_widget = FOVChooserSingle(valid_fovs=valid_fovs)
        self.peak_chooser_widget = ComboBox(
            value=self.cur_peak, choices=fov_choices[self.cur_fov], label="peak"
        )

        self.fov_chooser_widget.changed.connect(self.set_fov)
        self.peak_chooser_widget.changed.connect(self.set_peak)

        self.append(self.fov_chooser_widget)
        self.append(self.peak_chooser_widget)

    def set_fov(self):
        self.cur_fov = self.fov_chooser_widget.value
        self.peak_chooser_widget._default_choices = self.fov_choices[self.cur_fov]
        self.peak_chooser_widget.choices = self.fov_choices[self.cur_fov]
        self.peak_chooser_widget.value = min(self.fov_choices[self.cur_fov])
        self.cur_peak = self.peak_chooser_widget.value

    def set_peak(self):
        self.cur_peak = self.peak_chooser_widget.value

    def connect(self, callback):
        self.peak_chooser_widget.changed.connect(callback)


class PlanePicker(ComboBox):
    def __init__(
        self,
        permitted_planes,
        label="microscopy plane",
        tooltip="The plane you would like to use.",
    ):
        super().__init__(label=label, choices=permitted_planes, tooltip=tooltip)


class FOVChooser(LineEdit):
    """
    Widget for choosing multiple FOVs.
    Use connect_callback(...) instead of super().changed.connect(...).
    Additionally, the input function for connect_callback accepts a single
    parameter (the range of FOVs)
    """

    def __init__(self, permitted_FOVs, custom_label=None):
        self.min_FOV = min(permitted_FOVs)
        self.max_FOV = max(permitted_FOVs)
        if custom_label:
            label_str = f"{custom_label} ({self.min_FOV}-{self.max_FOV})"
        else:
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
