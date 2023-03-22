import pickle
import napari
import numpy as np
import yaml
import tifffile as tiff
import re

from napari import Viewer
from ._deriving_widgets import (
    MM3Container,
    FOVChooserSingle,
    InteractiveSpinBox,
    PlanePicker,
    information,
    warning,
)

TRANSLUCENT_RED = np.array([1.0, 0.0, 0.0, 0.25])
TRANSLUCENT_GREEN = np.array([0.0, 1.0, 0.0, 0.25])
TRANSLUCENT_BLUE = np.array([0.0, 0.0, 1.0, 0.25])
TRANSPARENT = np.array([0, 0, 0, 0])

SPEC_TO_COLOR = {
    -1: TRANSLUCENT_RED,
    0: TRANSLUCENT_BLUE,
    1: TRANSLUCENT_GREEN,
}

OVERLAY_TEXT = (
    "Interactive channel picker. Click to change channel designation. "
    "Color code:\n"
    "    Green: A channel with bacteria, \n"
    "    Blue: An empty channel without bacteria, to be used as a template. \n"
    "    Red: A channel to ignore. \n"
    "The number above the channel is the cross-correlation. \n"
    "A higher value means that that the channel is more likely to be empty.\n"
    "'Shapes' layer must be selected to change channel assignment."
)


# function for loading the channel masks
def load_channel_masks(analysis_directory):
    """Load channel masks dictionary. Should be .yaml but try pickle too."""
    information("Loading channel masks dictionary.")

    # try loading from .yaml before .pkl
    try:
        information("Path:", analysis_directory / "channel_masks.yaml")
        with open(analysis_directory / "channel_masks.yaml", "r") as cmask_file:
            channel_masks = yaml.safe_load(cmask_file)
    except:
        warning("Could not load channel masks dictionary from .yaml.")

        try:
            information("Path:", analysis_directory / "channel_masks.pkl")
            with open(analysis_directory / "channel_masks.pkl", "rb") as cmask_file:
                channel_masks = pickle.load(cmask_file)
        except ValueError:
            warning("Could not load channel masks dictionary from .pkl.")

    return channel_masks


def load_specs(analysis_directory):
    with (analysis_directory / "specs.yaml").open("r") as specs_file:
        specs = yaml.safe_load(specs_file)
    if specs == None:
        file_location = analysis_directory / "specs.yaml"
        raise FileNotFoundError(
            f"Specs file not found. Looked for it in the following location:\n {file_location.absolute().as_posix()}"
        )
    return specs


def save_specs(analysis_folder, specs):
    with (analysis_folder / "specs.yaml").open("w") as specs_file:
        yaml.dump(data=specs, stream=specs_file, default_flow_style=False, tags=None)
    information("Saved channel classifications to specs file")


def load_fov(image_directory, fov_id):
    information("getting files")
    found_files = image_directory.glob("*.tif")
    file_string = re.compile(f"xy{fov_id:02d}.*.tif", re.IGNORECASE)
    found_files = [
        f.name for f in found_files if re.search(file_string, f.name)
    ]  # remove pre-path
    information("sorting files")
    found_files = sorted(found_files)  # should sort by timepoint

    if len(found_files) == 0:
        information("No data found for FOV " + str(fov_id))
        return

    image_fov_stack = []

    information("Loading files")
    for img_filename in found_files:
        with tiff.TiffFile(image_directory / img_filename) as tif:
            image_fov_stack.append(tif.asarray())

    information("numpying files")
    return np.array(image_fov_stack)


def load_crosscorrs(analysis_directory, fov_id=None):
    information("Getting crosscorrs")
    with (analysis_directory / "crosscorrs.pkl").open("rb") as data:
        cross_corrs = pickle.load(data)
        if fov_id == None:
            return cross_corrs
    fov_crosscorrs = cross_corrs[fov_id]
    average_crosscorrs = {
        peak: fov_crosscorrs[peak]["cc_avg"] for peak in fov_crosscorrs
    }
    return average_crosscorrs


def display_image_stack(viewer: Viewer, image_fov_stack, plane):
    images = viewer.add_image(np.array(image_fov_stack))
    viewer.dims.current_step = (0, plane, 0, 0)
    images.reset_contrast_limits()
    images.gamma = 0.5


def threshold_fov(fov, threshold, specs, crosscorrs, channel_masks=None):
    if crosscorrs:
        # update dictionary on initial guess from cross correlations
        peaks = crosscorrs[fov]
        specs[fov] = {}
        for peak_id, xcorrs in peaks.items():
            # default to don't analyze
            specs[fov][peak_id] = -1
            if xcorrs["cc_avg"] < threshold:
                specs[fov][peak_id] = 1
    else:
        # We don't have crosscorrelations for this FOV -- default to ignoring peaks
        specs[fov] = {}
        channel_masks = channel_masks
        for peaks in channel_masks[fov]:
            specs[fov] = {peak_id: -1 for peak_id in peaks.keys()}

    return specs


def display_rectangles(
    viewer: napari.Viewer, coords, sorted_peaks, sorted_specs, crosscorrs
):
    # Set up crosscorrelation text
    properties = {"peaks": sorted_peaks, "crosscorrs": crosscorrs.values()}
    text_parameters = {
        "text": "{crosscorrs:.03f}",
        "size": 8,
        "anchor": "upper_left",
        "visible": True,
        "color": "white",
    }

    curr_colors = [SPEC_TO_COLOR[n] for n in sorted_specs]

    # Add channel boxes.
    shapes_layer = viewer.add_shapes(
        coords,
        shape_type="rectangle",
        face_color=curr_colors,
        edge_color=TRANSPARENT,
        properties=properties,
        text=text_parameters,
        opacity=1,
    )

    return shapes_layer


def regenerate_fov_specs(analysis_folder, fov, threshold, overwrite=False):
    try:
        specs = load_specs(analysis_folder)
    except:
        specs = {}

    if not overwrite and (fov in specs):
        return specs

    try:
        crosscorrs = load_crosscorrs(analysis_folder)
    except:
        crosscorrs = {}

    try:
        channel_masks = load_channel_masks(analysis_directory=analysis_folder)
    except:
        channel_masks = None

    new_specs = threshold_fov(fov, threshold, specs, crosscorrs, channel_masks)
    save_specs(analysis_folder=analysis_folder, specs=new_specs)
    return new_specs


class ChannelPicker(MM3Container):
    def create_widgets(self):
        """Overriding method. Serves as the widget constructor. See MM3Container for more details."""

        self.experiment_name_widget.hide()
        self.load_recent_widget.hide()
        self.run_widget.hide()

        # Set up viewer
        self.viewer.grid.enabled = False
        self.viewer.text_overlay.text = OVERLAY_TEXT
        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.color = "white"

        self.fov_picker_widget = FOVChooserSingle(self.valid_fovs)
        self.fov_picker_widget.connect(self.update_fov)
        self.append(self.fov_picker_widget)

        self.threshold_widget = InteractiveSpinBox(
            label="crosscorrelation threshold",
            tooltip="the autocorrelation threshold for discerning whether or not a given channel is empty.",
            min=0,
            max=1,
            value=0.97,
            step=0.005,
            use_float=True,
        )
        self.threshold_widget.connect(self.update_threshold)
        self.append(self.threshold_widget)

        self.threshold = self.threshold_widget.value
        self.plane_widget = PlanePicker(
            self.valid_planes,
            label="Default plane",
            tooltip="Imaging plane to display by default",
        )
        self.plane_widget.changed.connect(self.set_plane)
        self.append(self.plane_widget)
        self.set_plane()

        try:
            self.update_fov()
        except:
            Warning("Failed to load FOV")

    def update_fov(self):
        self.cur_fov = self.fov_picker_widget.value
        self.specs = regenerate_fov_specs(
            self.analysis_folder, self.cur_fov, self.threshold, overwrite=False
        )
        image_fov_stack = load_fov(self.TIFF_folder, self.cur_fov)
        self.sorted_peaks = list(sorted(self.specs[self.cur_fov].keys()))
        self.sorted_specs = [self.specs[self.cur_fov][p] for p in self.sorted_peaks]

        self.crosscorrs = load_crosscorrs(self.analysis_folder, self.cur_fov)

        self.viewer.layers.clear()

        display_image_stack(self.viewer, image_fov_stack, self.default_plane)

        # Set up selection box dimensions
        height = image_fov_stack.shape[-2]
        width = image_fov_stack.shape[-1]
        

        channel_height = height
        channel_width = width / len(self.sorted_peaks)
        spread = channel_width // 2
        self.coords = [
            [[0, p - spread], [channel_height, p + spread]] for p in self.sorted_peaks
        ]
        shapes_layer = display_rectangles(
            self.viewer,
            self.coords,
            self.sorted_peaks,
            self.sorted_specs,
            self.crosscorrs,
        )

        shapes_layer.mouse_drag_callbacks.append(self.update_classification)

    def update_classification(self, shapes_layer, event):
        # Figure out what is under our cursors. If nothing, kick out.
        cursor_data_coordinates = shapes_layer.world_to_data(event.position)
        shapes_under_cursor = shapes_layer.get_value(cursor_data_coordinates)
        # Nothing under cursor
        if shapes_under_cursor is None:
            return
        shape_i = shapes_under_cursor[0]
        # Image under cursor, but no channel
        if shape_i == None:
            return

        # Would be nice to do this with modulo, but sadly we chose -1 0 1 as our convention instead of 0 1 2
        next_color = {-1: 0, 0: 1, 1: -1}
        # Switch to the next color!
        self.sorted_specs[shape_i] = next_color[self.sorted_specs[shape_i]]

        # Redraw extant rectangles
        curr_colors = [SPEC_TO_COLOR[n] for n in self.sorted_specs]
        ## update the shape color accordingly
        # clear existing shapes
        # information(self.viewer.layers)
        self.viewer.layers[1].data = []
        # redraw with updated colors
        shapes_layer.add(self.coords, shape_type="rectangle", face_color=curr_colors)

        # update specs
        self.specs[self.cur_fov][self.sorted_peaks[shape_i]] = self.sorted_specs[
            shape_i
        ]
        save_specs(self.analysis_folder, self.specs)

    def update_threshold(self, shapes_layer):
        self.threshold = self.threshold_widget.value
        self.specs = regenerate_fov_specs(
            self.analysis_folder, self.cur_fov, self.threshold, overwrite=True
        )
        self.sorted_peaks = list(sorted(self.specs[self.cur_fov].keys()))
        self.sorted_specs = [self.specs[self.cur_fov][p] for p in self.sorted_peaks]
        self.viewer.layers.pop()
        shapes_layer = display_rectangles(
            self.viewer,
            self.coords,
            self.sorted_peaks,
            self.sorted_specs,
            self.crosscorrs,
        )
        shapes_layer.mouse_drag_callbacks.append(self.update_classification)

    def set_plane(self):
        self.default_plane = int(self.plane_widget.value[-1]) - 1
        self.viewer.dims.current_step = (0, self.default_plane, 0, 0)
        try:
            self.viewer.layers[0].reset_contrast_limits()
            self.viewer.layers[0].gamma = 0.5
        except IndexError:
            pass
