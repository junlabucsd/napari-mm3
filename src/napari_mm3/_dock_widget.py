from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLineEdit, QLabel, QFileDialog, QHBoxLayout
from qtpy.QtCore import Qt
from magicgui import magic_factory

import numpy as np
import tifffile as tiff
import yaml
import os
import glob

class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")

def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    #return [ExampleQWidget, example_magic_widget]
    return []

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

class ChannelPicker(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        expDirLabel = QLabel('Data directory')
        expNameLabel = QLabel('Experiment name')

        advanceFOVButton = QPushButton("Next FOV")
        advanceFOVButton.clicked.connect(self.next_fov)

        priorFOVButton = QPushButton("Prior FOV")
        priorFOVButton.clicked.connect(self.prior_fov)

        saveOutButton = QPushButton("Save Out")
        saveOutButton.clicked.connect(self.save_out)

        self.expDir = QLineEdit('/Users/ryan/Data/test/20200911_sj1536/')
        self.expName = QLineEdit('20200911_sj1536')

        loadDataButton = QPushButton("Load data")
        loadDataButton.clicked.connect(self.load_data)

        self.setLayout(QVBoxLayout())

        self.layout().addWidget(expDirLabel)
        self.layout().addWidget(self.expDir)
        self.layout().addWidget(expNameLabel)
        self.layout().addWidget(self.expName)

        self.layout().addWidget(loadDataButton)
        self.layout().addWidget(advanceFOVButton)
        self.layout().addWidget(priorFOVButton)
        self.layout().addWidget(saveOutButton)


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

        self.fov_id_list = [fov_id for fov_id in self.specs.keys()]

        try:
            self.fovIndex
        except AttributeError:
            self.fovIndex = 0

        try:
            self.fov_id = self.fov_id_list[self.fovIndex]
        except IndexError:
            print('FOV not found')
            return

        self.peak_id_list_in_fov = [peak_id for peak_id in self.specs[self.fov_id].keys() if self.specs[self.fov_id][peak_id] == 1]

        found_files = glob.glob(os.path.join(self.exp_dir,'TIFF/', '*xy%02d.tif' % (self.fov_id)))# get all tiffs
        found_files = [filepath.split('/')[-1] for filepath in found_files] # remove pre-path
        found_files = sorted(found_files) # should sort by timepoint

        if len(found_files) == 0:
            print('No data found for FOV '+ str(self.fov_id))
            return

        image_fov_stack = []

        # go through list of images and get the file path
        for img_filename in found_files:
            with tiff.TiffFile(os.path.join(self.exp_dir,'TIFF/',img_filename)) as tif:
                image_fov_stack.append(tif.asarray())

        image_fov_stack = np.array(image_fov_stack)

        self.viewer.layers.clear()
        self.viewer.grid.enabled = False
        # self.viewer.add_image(np.array(image_fov_stack),contrast_limits=[90,250])
        self.viewer.add_image(np.array(image_fov_stack))
        self.sorted_peaks = sorted([peak_id for peak_id in self.specs[self.fov_id].keys()])
        self.sorted_specs = [self.specs[self.fov_id][p] for p in self.sorted_peaks]

        ## get height of fov stack
        height = len(image_fov_stack[0,0,:,0])

        ## get tiff height and width

        coords = [[[0,p-20],[height,p+20]]for p in self.sorted_peaks]
        ## add list of colors for each rectangle

        def spec_to_color(n):
            if n==-1:
                return 'red'
            if n==0:
                return 'blue'
            if n==1:
                return 'green'

        curr_colors = [spec_to_color(n) for n in self.sorted_specs]
        shapes_layer = self.viewer.add_shapes(coords,shape_type='rectangle',face_color=curr_colors,properties = self.sorted_peaks,opacity=.25)
        @shapes_layer.mouse_double_click_callbacks.append
        def update_classification(shapes_layer,event):
            #get the selected shape
            try:
                shape_i = next(iter(shapes_layer.selected_data))
            except:
                print('Select a channel from shapes layer using the selection tool')
                return
            
            #permute the channel value
            if self.sorted_specs[shape_i] == -1:
                self.sorted_specs[shape_i] = 0
            elif self.sorted_specs[shape_i] == 0:
                self.sorted_specs[shape_i] = 1
            elif self.sorted_specs[shape_i] == 1:
                self.sorted_specs[shape_i] = -1

            ## update the shape color accordingly
            curr_colors[shape_i] = spec_to_color(self.sorted_specs[shape_i])        

            # clear existing shapes
            self.viewer.layers['Shapes'].data=[]

            # redraw with updated colors
            shapes_layer.add(coords,shape_type='rectangle',face_color=curr_colors)

            # update specs
            self.specs[self.fov_id][self.sorted_peaks[shape_i]] = self.sorted_specs[shape_i]

    def next_fov(self):
        self.fovIndex+=1
        self.load_data()
    
    def prior_fov(self):
        self.fovIndex-=1
        self.load_data()

    def save_out(self):
        with open(os.path.join(self.exp_dir, 'analysis/specs.yaml'), 'w') as specs_file:
            yaml.dump(data=self.specs, stream=specs_file, default_flow_style=False, tags=None)
        print('Saved channel classifications to specs file')