import os
import pickle
from pathlib import Path
from turtle import shape
import magicgui
import napari
import numpy as np
import yaml
import tifffile as tiff


TRANSLUCENT_RED = np.array([1.0, 0.0, 0.0, 0.25])
TRANSLUCENT_GREEN = np.array([0.0, 1.0, 0.0, 0.25])
TRANSLUCENT_BLUE = np.array([0.0, 0.0, 1.0, 0.25])
TRANSPARENT = np.array([0, 0, 0, 0])


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


@magicgui.magic_factory(
    auto_call=True,
    data_directory={
        "mode": "d",
        "tooltip": "Directory within which all your data and analyses will be located.",
    },
    current_fov={
        "widget_type": "SpinBox",
        "min": 1,
        "step": 1,
        "tooltip": "The FOV for which you are performing labelling."
    },
    image_directory={
        "tooltip": "Required. Location (within working directory) for the input images. 'working directory/TIFF/' by default."
    },
    analysis_directory={
        "tooltip": "Required. Location (within working directory) for outputting analysis. 'working directory/analysis/' by default."
    },
)
def ChannelPicker(
    viewer: napari.Viewer,
    data_directory: Path = Path(),
    image_directory: str = "TIFF",
    analysis_directory: str = "analysis",
    current_fov: int = 1,
):
    specs = None
    with (data_directory / analysis_directory / "specs.yaml").open("r") as specs_file:
        specs = yaml.safe_load(specs_file)
    if specs == None:
        file_location = data_directory / analysis_directory / "specs.yaml"
        raise FileNotFoundError(
            f"Specs file not found. Looked for it in the following location:\n {file_location.absolute().as_posix()}"
        )

    fov_id_list = list(specs.keys())
    if current_fov not in fov_id_list:
        raise IndexError(
            f"FOV not found. Max FOV: {max(fov_id_list)}, Min FOV: {min(fov_id_list)}"
        )
    fov_id = current_fov

    # Set up viewer.
    viewer.layers.clear()
    viewer.grid.enabled = False
    viewer.text_overlay.text = (
        "Interactive channel picker. Click to change channel designation. "
        "Color code:\n"
        "    Green: A channel with bacteria, \n"
        "    Blue: An empty channel without bacteria, to be used as a template. \n"
        "    Red: A channel to ignore. \n"
        "The number above the channel is the cross-correlation. \n"
        "A higher value means that that the channel is more likely to be empty.\n"
        "'Shapes' layer must be selected to change channel assignment."
    )
    viewer.text_overlay.visible = True
    viewer.text_overlay.color = "white"

    print("Rendering images")
    image_fov_stack = load_fov(data_directory / image_directory, fov_id)
    images = viewer.add_image(np.array(image_fov_stack))
    viewer.dims.current_step = (0, 0)
    images.reset_contrast_limits()
    images.gamma = 0.5

    # Set up selection box dimensions
    sorted_peaks = list(sorted(specs[fov_id].keys()))
    sorted_specs = [specs[fov_id][p] for p in sorted_peaks]
    height = image_fov_stack.shape[2]
    width = image_fov_stack.shape[3] / len(sorted_peaks)
    spread = width // 2
    coords = [[[0, p - spread], [height, p + spread]] for p in sorted_peaks]

    # Set up crosscorrelation text
    crosscorrs = load_crosscorrs(data_directory / analysis_directory, current_fov)
    properties = {"peaks": sorted_peaks, "crosscorrs": crosscorrs.values()}
    text_parameters = {
        "text": "{crosscorrs:.03f}",
        "size": 8,
        "anchor": "upper_left",
        "visible": True,
        "color": "white",
    }

    # Set box colors
    spec_to_color = {
        -1: TRANSLUCENT_RED,
        0: TRANSLUCENT_BLUE,
        1: TRANSLUCENT_GREEN,
    }
    curr_colors = [spec_to_color[n] for n in sorted_specs]

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

    @shapes_layer.mouse_drag_callbacks.append
    def update_classification(shapes_layer, event):
        cursor_data_coordinates = shapes_layer.world_to_data(event.position)
        shapes_under_cursor = shapes_layer.get_value(cursor_data_coordinates)
        if shapes_under_cursor is None:
            # Nothing found under cursor
            return
        shape_i = shapes_under_cursor[0]
        if shape_i == None:
            # Image under cursor, but no channel
            return

        # Would be nice to do this with modulo, but sadly we chose -1 0 1 as our convention instead of 0 1 2
        next_color = {-1: 0, 0: 1, 1: -1}
        # Switch to the next color!
        sorted_specs[shape_i] = next_color[sorted_specs[shape_i]]

        ## update the shape color accordingly
        curr_colors[shape_i] = spec_to_color[sorted_specs[shape_i]]

        # clear existing shapes
        viewer.layers["Shapes"].data = []

        # redraw with updated colors
        shapes_layer.add(coords, shape_type="rectangle", face_color=curr_colors)

        # update specs
        specs[fov_id][sorted_peaks[shape_i]] = sorted_specs[shape_i]

        with (data_directory / analysis_directory / "specs.yaml").open(
            "w"
        ) as specs_file:
            yaml.dump(
                data=specs, stream=specs_file, default_flow_style=False, tags=None
            )
        print("Saved channel classifications to specs file")
