import glob
import os
from pathlib import Path
import magicgui
import napari
import numpy as np
import yaml
import tifffile as tiff

def load_fov(data_directory, fov_id):
    print("getting files")
    found_files = glob.glob(os.path.join(data_directory,'TIFF/', '*xy%02d.tif' % (fov_id)))# get all tiffs
    found_files = [filepath.split('/')[-1] for filepath in found_files] # remove pre-path
    print("sorting files")
    found_files = sorted(found_files) # should sort by timepoint

    if len(found_files) == 0:
        print('No data found for FOV '+ str(fov_id))
        return

    image_fov_stack = []

    print("Loading files")
    for img_filename in found_files:
        with tiff.TiffFile(os.path.join(data_directory,'TIFF/',img_filename)) as tif:
            image_fov_stack.append(tif.asarray())

    print("numpying files")
    return np.array(image_fov_stack)

@magicgui.magic_factory(auto_call=True, data_directory={"mode": "d"})
def ChannelPicker(viewer: napari.Viewer, data_directory = Path("~"), cur_fov = 0):
    specs = None
    with open(os.path.join(data_directory, 'analysis/specs.yaml'), 'r') as specs_file:
        specs = yaml.safe_load(specs_file)
    if specs == None:
        print("Error: No specs file")
        return

    fov_id_list = [fov_id for fov_id in specs.keys()]
    try:
        fov_id = fov_id_list[cur_fov]
    except IndexError:
        print('Error: FOV not found')
        return

    image_fov_stack = load_fov(data_directory, fov_id)

    print("Rendering image")
    viewer.layers.clear()
    viewer.grid.enabled = False
    images = viewer.add_image(np.array(image_fov_stack))
    sorted_peaks = sorted([peak_id for peak_id in specs[fov_id].keys()])
    sorted_specs = [specs[fov_id][p] for p in sorted_peaks]
    images.reset_contrast_limits()

    ## get height of fov stack
    height = len(image_fov_stack[0,0,:,0])

    ## get tiff height and width
    coords = [[[0,p-20],[height,p+20]]for p in sorted_peaks]

    ## add list of colors for each rectangle... this should really be an enum
    spec_to_color = {
        -1: 'red',
        0: 'blue',
        1: 'green',
    }

    curr_colors = [spec_to_color[n] for n in sorted_specs]
    shapes_layer = viewer.add_shapes(coords,shape_type='rectangle',face_color=curr_colors,properties = sorted_peaks,opacity=.25)

    @shapes_layer.mouse_drag_callbacks.append
    def update_classification(shapes_layer,event):
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
        next_color = {-1: 0, 0: 1, 1:-1}
        # Switch to the next color!
        sorted_specs[shape_i] = next_color[sorted_specs[shape_i]]

        ## update the shape color accordingly
        curr_colors[shape_i] = spec_to_color[sorted_specs[shape_i]]

        # clear existing shapes
        viewer.layers['Shapes'].data=[]

        # redraw with updated colors
        shapes_layer.add(coords,shape_type='rectangle',face_color=curr_colors)

        # update specs
        specs[fov_id][sorted_peaks[shape_i]] = sorted_specs[shape_i]

        with open(os.path.join(data_directory, 'analysis/specs.yaml'), 'w') as specs_file:
            yaml.dump(data=specs, stream=specs_file, default_flow_style=False, tags=None)
        print('Saved channel classifications to specs file')
