from cgitb import enable
from gc import callbacks
import pickle
from pathlib import Path
import napari
import numpy as np
import yaml
import tifffile as tiff

from ._deriving_widgets import MM3Container, FOVChooserSingle

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


def load_fov(image_directory, fov_id):
    print("getting files")
    found_files = image_directory.glob(f"*xy{fov_id:02d}.tif")
    found_files = [filepath.name for filepath in found_files]  # remove pre-path
    print("sorting files")
    found_files = sorted(found_files)  # should sort by timepoint

    if len(found_files) == 0:
        print("No data found for FOV " + str(fov_id))
        return

    image_fov_stack = []

    print("Loading files")
    for img_filename in found_files:
        with tiff.TiffFile(image_directory / img_filename) as tif:
            image_fov_stack.append(tif.asarray())

    print("numpying files")
    return np.array(image_fov_stack)


def load_crosscorrs(analysis_directory, fov_id):
    print("Getting crosscorrs")
    with (analysis_directory / "crosscorrs.pkl").open("rb") as data:
        cross_corrs = pickle.load(data)
    fov_crosscorrs = cross_corrs[fov_id]
    average_crosscorrs = {
        peak: fov_crosscorrs[peak]["cc_avg"] for peak in fov_crosscorrs
    }
    return average_crosscorrs


def display_image_stack(viewer: napari.Viewer, image_fov_stack):
    images = viewer.add_image(np.array(image_fov_stack))
    viewer.dims.current_step = (0, 0)
    images.reset_contrast_limits()
    images.gamma = 0.5


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

class ChannelPicker(MM3Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(viewer)

        self.experiment_name_widget.hide()
        self.create_widgets()
        self.load_data_widget.clicked.connect(self.delete_widgets)
        self.load_data_widget.clicked.connect(self.create_widgets)

        # Set up viewer
        self.viewer.grid.enabled = False
        self.viewer.text_overlay.text = OVERLAY_TEXT
        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.color = "white"

    def create_widgets(self):
        """Serves as the widget constructor. See MM3Container for more details."""
        self.fov_picker_widget = FOVChooserSingle(self.valid_fovs)
        self.fov_picker_widget.connect_callback(self.update_fov)
        self.append(self.fov_picker_widget)

        self.specs = load_specs(self.analysis_folder)
        self.update_fov()

    def delete_widgets(self):
        """Serves as the widget destructor. See MM3Container for more details."""
        self.pop() # Remove FOV picker


    def update_fov(self):
        self.cur_fov = self.fov_picker_widget.fov
        image_fov_stack = load_fov(self.TIFF_folder, self.cur_fov)
        self.sorted_peaks = list(sorted(self.specs[self.cur_fov].keys()))
        self.sorted_specs = [self.specs[self.cur_fov][p] for p in self.sorted_peaks]

        self.crosscorrs = load_crosscorrs(self.analysis_folder, self.cur_fov)

        self.viewer.layers.clear()
        display_image_stack(self.viewer, image_fov_stack)

        # Set up selection box dimensions
        height = image_fov_stack.shape[2]
        width = image_fov_stack.shape[3]

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
        print(self.viewer.layers)
        self.viewer.layers[1].data = []
        # redraw with updated colors
        shapes_layer.add(self.coords, shape_type="rectangle", face_color=curr_colors)

        # update specs
        self.specs[self.cur_fov][self.sorted_peaks[shape_i]] = self.sorted_specs[
            shape_i
        ]
        save_specs(self.analysis_folder, self.specs)
        print("Saved channel classifications to specs file")
