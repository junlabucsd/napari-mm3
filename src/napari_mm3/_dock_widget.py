from multiprocessing import Process, Queue
import magicgui
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

class Annotate(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        expDirLabel = QLabel('Data directory')
        expNameLabel = QLabel('Experiment name')

        self.expDir = QLineEdit('/Users/ryan/Data/test/20200911_sj1536/')
        self.expName = QLineEdit('20200911_sj1536')

        advancePeakButton = QPushButton("Next peak")
        advancePeakButton.setShortcut("Ctrl+P")
        advancePeakButton.clicked.connect(self.next_peak)

        priorPeakButton = QPushButton("Prior peak")
        priorPeakButton.clicked.connect(self.prior_peak)

        advanceFOVButton = QPushButton("Next FOV")
        advanceFOVButton.clicked.connect(self.next_fov)

        priorFOVButton = QPushButton("Prior FOV")
        priorFOVButton.clicked.connect(self.prior_fov)

        loadDataButton = QPushButton("Load data")
        loadDataButton.clicked.connect(self.load_data)


        saveButton = QPushButton("Save and next frame")
        saveButton.setShortcut("Ctrl+S")
        saveButton.clicked.connect(self.save_out)

        self.setLayout(QVBoxLayout())

        self.layout().addWidget(expDirLabel)
        self.layout().addWidget(self.expDir)
        self.layout().addWidget(expNameLabel)
        self.layout().addWidget(self.expName)

        self.layout().addWidget(advancePeakButton)
        self.layout().addWidget(priorPeakButton)

        
        self.layout().addWidget(advanceFOVButton)
        self.layout().addWidget(priorFOVButton)

        self.layout().addWidget(loadDataButton)

    def load_data(self):

        try:
            self.exp_dir
        except AttributeError:
            self.exp_dir = self.expDir.text()
        try:
            self.exp_name
        except AttributeError:
            self.exp_name = self.expName.text()
        try:
            self.specs
        except AttributeError:
            with open(os.path.join(self.exp_dir, 'analysis/specs.yaml'), 'r') as specs_file:
                self.specs = yaml.safe_load(specs_file)
        try:
            self.fovIndex
        except AttributeError:
            self.fovIndex = 0
        try:
            self.peakIndex
        except AttributeError:
            self.peakIndex = 0

        self.fov_id_list = [fov_id for fov_id in self.specs.keys()]
        self.fov_id = self.fov_id_list[self.fovIndex]
        self.peak_id_list_in_fov = [peak_id for peak_id in self.specs[self.fov_id].keys() if self.specs[self.fov_id][peak_id] == 1]

        self.peak_id = self.peak_id_list_in_fov[self.peakIndex]

        img_filename = os.path.join(self.exp_dir,'analysis/channels/',self.exp_name +'_xy%03d_p%04d_c1.tif' % (self.fov_id, self.peak_id))
        mask_filename = os.path.join(self.exp_dir,'analysis/segmented/',self.exp_name+'_xy%03d_p%04d_seg_unet.tif' % (self.fov_id, self.peak_id))

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

    def next_fov(self):
        self.save_out()
        self.fovIndex += 1
        self.fov_id = self.fov_id_list[self.fovIndex]
        self.peak_id_list_in_fov = [peak_id for peak_id in self.specs[self.fovIndex].keys() if self.specs[self.fov_id][peak_id] == 1]
        self.load_data()
    
    def prior_fov(self):
        self.save_out()
        self.fovIndex -= 1
        self.fov_id = self.fov_id_list[self.fovIndex]
        self.peak_id_list_in_fov = [peak_id for peak_id in self.specs[self.fovIndex].keys() if self.specs[self.fov_id][peak_id] == 1]
        self.load_data()

    def next_peak(self):
        self.peakIndex+=1
        try:
            self.peak_id = self.peak_id_list_in_fov[self.peakIndex]
        except IndexError:
            print('No more peaks in this FOV')
        self.load_data()

    def prior_peak(self):
        self.peakIndex-=1
        try:
            self.peak_id = self.peak_id_list_in_fov[self.peakIndex]
        except IndexError:
            print('No earlier peaks in this FOV')
        self.load_data()

    def save_out(self):
        labels = self.viewer.layers[1].data.astype(np.uint8)
        ## need to get current fov
        for peak_id in self.specs[self.fov_id].keys():
            fileout_name = os.path.join(self.trainingDir,self.exp_name+'_xy%03d_p%04d_seg.tif' % (self.fov_id, self.peak_id))
            tiff.imsave(fileout_name,labels)
        print('Training data saved')

    def open_directory(self):
        self.expDir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

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

    # go through list of images and get the file path
    print("Loading files")
    for img_filename in found_files:
        with tiff.TiffFile(os.path.join(data_directory,'TIFF/',img_filename)) as tif:
            image_fov_stack.append(tif.asarray())

    print("numpying files")
    return np.array(image_fov_stack)

def load_fov_write_to_queue(queue, data_directory, fov_id):
    queue.put(load_fov(data_directory, fov_id))

expected_fov_id = -1
p = None
q = Queue()
def load_fov_multiproc(data_directory, fov_id):
    """
    Experimental; use at your own risk. Preloads the FOV after the current one.
    """
    global expected_fov_id, p, q
    image_fov_stack = None
    if fov_id == expected_fov_id and p != None:
        # If the right image is in the queue, then we can go ahead and grab it
        image_fov_stack = q.get()
        p.join()
        print("Successfully multiprocessed!")
    else:
        # If the wrong image is in the queue, we should load from scratch.
        image_fov_stack = load_fov(data_directory, fov_id)

    # Start loading our predicted next image
    expected_fov_id = fov_id + 1
    p = Process(target = load_fov_write_to_queue, args=(q, data_directory, expected_fov_id))
    p.start()
    print("Launched our background process")

    return image_fov_stack


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
