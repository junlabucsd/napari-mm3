from multiprocessing import Process, Queue
import magicgui
from magicgui.widgets import FileEdit, Slider, Container, SpinBox, LineEdit, PushButton
from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLineEdit, QLabel, QFileDialog, QHBoxLayout
from qtpy.QtCore import Qt
from magicgui import magic_factory
from pathlib import Path

import napari
import numpy as np
import tifffile as tiff
import yaml
import os
import glob


class Annotate(Container):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.data_directory_widget = FileEdit(mode='d', label='data directory')
        # TODO: Make this auto-inferred from a function type signature.
        self.experiment_name_widget = LineEdit(label='experiment name')
        self.next_peak_widget = PushButton(label = "next peak")
        self.prior_peak_widget = PushButton(label = "prior_peak")
        self.FOV_id_widget = SpinBox(label = "FOV")

        self.peak_id = 0
        # Need this to detect *changes* in the FOV_id_widget.
        self.FOV_id = 0

        self.data_directory_widget.changed.connect(self.load_data)
        self.next_peak_widget.clicked.connect(self.next_peak)
        self.prior_peak_widget.clicked.connect(self.prior_peak)
        self.FOV_id_widget.changed.connect(self.FOV_id_changed)

        self.insert(0, self.data_directory_widget)
        self.insert(1, self.experiment_name_widget)
        self.insert(2, self.next_peak_widget)
        self.insert(3, self.prior_peak_widget)
        self.insert(4, self.FOV_id_widget)

    def load_specs(self):
        with open(os.path.join(self.data_directory_widget.value, 'analysis/specs.yaml'), 'r') as specs_file:
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

        img_filename = data_directory / 'analysis' / 'channels' / f'{experiment_name}_xy{fov:03d}_p{peak:04d}_c1.tif'
        mask_filename = data_directory / 'analysis' / 'segmented' / f'{experiment_name}_xy{fov:03d}_p{peak:04d}_seg_unet.tif'

        with tiff.TiffFile(mask_filename) as tif:
            mask_stack = tif.asarray()
        with tiff.TiffFile(img_filename) as tif:
            img_stack = tif.asarray()

        self.viewer.layers.clear()
        self.viewer.add_image(img_stack)

        try:
            self.viewer.add_labels(mask_stack,name='Labels')
        except:
            pass

        current_layers = [l.name for l in self.viewer.layers]
        
        if not "Labels" in current_layers:
            empty = np.zeros(np.shape(img_stack),dtype=int)
            self.viewer.add_labels(empty,name="Labels")

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
        fileout_name = training_dir / f'{self.experiment_name_widget.value}_xy{fov:03d}_p{peak:04d}_seg.tif'
        tiff.imsave(fileout_name,labels)
        print('Training data saved')

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
