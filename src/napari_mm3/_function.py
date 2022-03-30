"""
This module is an example of a barebones function plugin for napari

It implements the ``napari_experimental_provide_function`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING, DefaultDict

from enum import Enum
import numpy as np
import multiprocessing
from multiprocessing import Pool
import os
from napari_plugin_engine import napari_hook_implementation
from skimage.filters import threshold_otsu # segmentation
from skimage import morphology # many functions is segmentation used from this
from skimage import segmentation # used in make_masks and segmentation
from scipy import ndimage as ndi # labeling and distance transform
from mm3_Compile import compile

if TYPE_CHECKING:
    import napari

# This is the actual plugin function, where we export our function
# (The functions themselves are defined below)
@napari_hook_implementation
def napari_experimental_provide_function():
    # we can return a single function
    # or a tuple of (function, magicgui_options)
    # or a list of multiple functions with or without options, as shown here:
    #return [Segment, threshold, image_arithmetic]
    return [MM3]


# 1.  First example, a simple function that thresholds an image and creates a labels layer
def threshold(data: "napari.types.ImageData", threshold: int) -> "napari.types.LabelsData":
    """Threshold an image and return a mask."""
    return (data > threshold).astype(int)

# 2.  MM3 analysis
# def MM3(experiment_name: str='exp1', experiment_directory: str= '/Users/sharan/Desktop/Physics/mm3-latest/exp1/', 
# image_directory: str='TIFF/', analysis_directory:str= 'analysis/', TIFF_source:str= 'nd2ToTIFF', output: str= 'TIFF',
# debug:str= False, phase_plane: str='c1', pxl2um:float= 0.11, image_start : int=1, image_end: int=None,
# number_of_rows :int = 1, crop_ymin :float=None, crop_ymax :float= None, tworow_crop : tuple =None, tiff_compress :int=5,
# external_directory : str='None', do_metadata: bool= True, do_time_table : bool= True, do_channel_masks : bool= True, do_slicing : bool= True,
# t_end : int=None, find_channels_method: str= 'peaks', model_file_traps: str= '/Users/sharan/Desktop/Physics/mm3-latest/weights/feature_weights_512x512_normed.hdf5', 
# image_orientation : str= 'up', channel_width : int=10, channel_separation : int=45, channel_detection_snr : int=1, channel_length_pad : int=10, 
# channel_width_pad : int=10, trap_crop_height: int=256, trap_crop_width: int=27, trap_area_threshold: int=2, channel_prediction_batch_size: int=15, 
# merged_trap_region_area_threshold: int=400, do_crosscorrs: bool=True, do_CNN: bool=False, interactive: bool=True, do_seg: bool=False, 
# first_image: int=1, last_image: int=0, channel_picking_threshold: float =0.5, channel_picker_model_file: str= '/Users/sharan/Desktop/Physics/mm3-latest/weights/empties_weights.hdf5',
# do_empties: bool=True, do_subtraction: bool=True, alignment_pad: int=10, do_segmentation: bool=True, do_lineages: bool=True, OTSU_threshold: float= 1.0, first_opening_size: int=2,
# distance_threshold: int=2, second_opening_size: int=1, min_object_size:int= 25,  model_file:str='None', trained_model_image_height: int=256, trained_model_image_width: int=32,
# batch_size: int=210, cell_class_threshold: float= 0.60, save_predictions: bool=True):
#->"napari.types.LabelsData":

def MM3(experiment_name: str='exp1', experiment_directory: str= '/Users/sharan/Desktop/Physics/napari-mm3/exp1/', external_directory: str= '/Users/sharan/Desktop/Physics/napari-mm3/exp1/', analysis_directory:str= 'analysis/', debug:str= False, pxl2um:float= 0.11, image_start : int=1,
number_of_rows :int = 1,
image_orientation : str= 'up', channel_width : int=10, channel_separation : int=45, channel_detection_snr : int=1, channel_length_pad : int=10, 
channel_width_pad : int=10, trap_crop_height: int=256, trap_crop_width: int=27, trap_area_threshold: int=2, channel_prediction_batch_size: int=15, 
merged_trap_region_area_threshold: int=400, first_image: int=1, channel_picking_threshold: float =0.5, alignment_pad: int=10, OTSU_threshold: float= 1.0, first_opening_size: int=2,
distance_threshold: int=2, second_opening_size: int=1, min_object_size:int= 25, trained_model_image_height: int=256, trained_model_image_width: int=32,
batch_size: int=210, cell_class_threshold: float= 0.60):

    """Performs Mother Machine Analysis"""    
    global params
    params=dict()
    params['experiment_name']=experiment_name
    params['experiment_directory']=experiment_directory
    params['image_directory']='TIFF/'
    params['analysis_directory']=analysis_directory
    params['TIFF_source']='nd2ToTIFF'
    params['output']='TIFF'
    params['debug']=debug
    params['phase_plane']='c1'
    params['pxl2um']=pxl2um
    params['nd2ToTIFF']=dict()
    params['nd2ToTIFF']['image_start']=image_start
    params['nd2ToTIFF']['image_end']=None
    params['nd2ToTIFF']['number_of_rows']=number_of_rows
    params['nd2ToTIFF']['crop_ymin']=None
    params['nd2ToTIFF']['crop_ymax']=None
    params['nd2ToTIFF']['2row_crop']=None
    params['nd2ToTIFF']['tiff_compress']=5
    params['nd2ToTIFF']['external_directory']=external_directory
    params['compile']=dict()
    params['compile']['do_metadata' ]=True
    params['compile']['do_time_table']=True
    params['compile']['do_channel_masks']=True
    params['compile']['do_slicing']=True
    params['compile']['t_end']=None
    params['compile']['find_channels_method']='peaks'
    params['compile']['model_file_traps']='/Users/sharan/Desktop/Physics/mm3-latest/weights/feature_weights_512x512_normed.hdf5'
    params['compile']['image_orientation']=image_orientation
    params['compile']['channel_width']=channel_width
    params['compile']['channel_separation']=channel_separation
    params['compile']['channel_detection_snr']=channel_detection_snr
    params['compile']['channel_length_pad']=channel_length_pad
    params['compile']['channel_width_pad']=channel_width_pad
    params['compile']['trap_crop_height']=trap_crop_height
    params['compile']['trap_crop_width']=trap_crop_width
    params['compile']['trap_area_threshold']=trap_area_threshold*1000
    params['compile']['channel_prediction_batch_size']=channel_prediction_batch_size
    params['compile']['merged_trap_region_area_threshold']=merged_trap_region_area_threshold*1000
    params['channel_picker']=dict()
    params['channel_picker']['do_crosscorrs']=True
    params['channel_picker']['do_CNN']=False
    params['channel_picker']['interactive']=True
    params['channel_picker']['do_seg']=False
    params['channel_picker']['first_image']=first_image
    params['channel_picker']['last_image']=-1
    params['channel_picker']['channel_picking_threshold']=channel_picking_threshold
    params['channel_picker']['channel_picker_model_file']='/Users/sharan/Desktop/Physics/mm3-latest/weights/empties_weights.hdf5'
    params['subtract']=dict()
    params['subtract']['do_empties']=True
    params['subtract']['do_subtraction']=True
    params['subtract']['alignment_pad']=alignment_pad
    params['segment']=dict()
    params['segment']['do_segmentation']=True
    params['segment']['do_lineages']=True
    params['segment']['otsu']=dict()
    params['segment']['otsu']['OTSU_threshold']=OTSU_threshold
    params['segment']['otsu']['first_opening_size']=first_opening_size
    params['segment']['otsu']['distance_threshold']=distance_threshold
    params['segment']['otsu']['second_opening_size']=second_opening_size
    params['segment']['otsu']['min_object_size']=min_object_size
    params['segment']['model_file']='None'
    params['segment']['trained_model_image_height']=trained_model_image_height
    params['segment']['trained_model_image_width']=trained_model_image_width
    params['segment']['batch_size']=batch_size
    params['segment']['cell_class_threshold']=cell_class_threshold
    params['segment']['unet']=dict()
    params['segment']['unet']['save_predictions']=True

    params['num_analyzers'] = multiprocessing.cpu_count()

    # useful folder shorthands for opening files
    params['TIFF_dir'] = os.path.join(params['experiment_directory'], params['image_directory'])
    params['ana_dir'] = os.path.join(params['experiment_directory'], params['analysis_directory'])
    params['hdf5_dir'] = os.path.join(params['ana_dir'], 'hdf5')
    params['chnl_dir'] = os.path.join(params['ana_dir'], 'channels')
    params['empty_dir'] = os.path.join(params['ana_dir'], 'empties')
    params['sub_dir'] = os.path.join(params['ana_dir'], 'subtracted')
    params['seg_dir'] = os.path.join(params['ana_dir'], 'segmented')
    params['pred_dir'] = os.path.join(params['ana_dir'], 'predictions')
    params['foci_seg_dir'] = os.path.join(params['ana_dir'], 'segmented_foci')
    params['foci_pred_dir'] = os.path.join(params['ana_dir'], 'predictions_foci')
    params['cell_dir'] = os.path.join(params['ana_dir'], 'cell_data')
    params['track_dir'] = os.path.join(params['ana_dir'], 'tracking')
    params['foci_track_dir'] = os.path.join(params['ana_dir'], 'tracking_foci')

    # use jd time in image metadata to make time table. Set to false if no jd time
    if params['TIFF_source'] == 'elements' or params['TIFF_source'] == 'nd2ToTIFF':
        params['use_jd'] = True
    else:
        params['use_jd'] = False

    if not 'save_predictions' in params['segment'].keys():
        params['segment']['save_predictions'] = False

    compile(params)
    return 

# 3. Second example, a function that adds, subtracts, multiplies, or divides two layers

# using Enums is a good way to get a dropdown menu.  Used here to select from np functions
class Operation(Enum):
    add = np.add
    subtract = np.subtract
    multiply = np.multiply
    divide = np.divide


def image_arithmetic(
    layerA: "napari.types.ImageData", operation: Operation, layerB: "napari.types.ImageData"
) -> "napari.types.LayerDataTuple":
    """Adds, subtracts, multiplies, or divides two same-shaped image layers."""
    return (operation.value(layerA, layerB), {"colormap": "turbo"})
