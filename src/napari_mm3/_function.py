from __future__ import print_function, division
import copy
import glob
import json
import os
import multiprocessing
from multiprocessing import Pool
import matplotlib as mpl
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}
mpl.rc('font', **font)
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.patches as mpatches
from matplotlib import image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import napari
from napari_plugin_engine import napari_hook_implementation
import os
try:
    import cPickle as pickle
except:
    import pickle
from pprint import pprint
from pathlib import Path
import pims_nd2
import re
from skimage import io, segmentation, morphology, measure
from skimage.measure import regionprops
from skimage.exposure import rescale_intensity
import seaborn as sns
sns.set(style='ticks', color_codes=True)
sns.set_palette('deep')
import six
from tensorflow.keras import models
import tifffile as tiff
import warnings
import yaml

import napari_mm3._mm3_helpers as mm3_helpers

# This is the actual plugin function, where we export our function
# (The functions themselves are defined below)
@napari_hook_implementation
def napari_experimental_provide_function():
    # we can return a single function
    # or a tuple of (function, magicgui_options)
    # or a list of multiple functions with or without options, as shown here:
    #return [Segment, threshold, image_arithmetic]
    return [Compile, ChannelPicker, Segment, Track_Standard]

def nd2ToTIFF(params):
    '''
    This script converts a Nikon Elements .nd2 file to individual TIFF files per time point. Multiple color planes are stacked in each time point to make a multipage TIFF.
    '''

    # Load the project parameters file
    mm3_helpers.information('Loading experiment parameters.')
    print("ND2TOTIFF")
    p=params

    if p['FOV']:
        if '-' in p['FOV']:
            user_spec_fovs = range(int(p['FOV'].split("-")[0]),
                                   int(p['FOV'].split("-")[1])+1)
        else:
            user_spec_fovs = [int(val) for val in p['FOV'].split(",")]
    else:
        user_spec_fovs = []

    # number of rows of channels. Used for cropping.
    number_of_rows = p['nd2ToTIFF']['number_of_rows']

    # cropping
    vertical_crop = False
    if number_of_rows == 1 and p['nd2ToTIFF']['crop_ymin'] and p['nd2ToTIFF']['crop_ymax']:
        try:
            # This is for percentage crop
            vertical_crop = [p['nd2ToTIFF']['crop_ymin'], p['nd2ToTIFF']['crop_ymax']]
        except KeyError:
            pass
    elif number_of_rows == 2:
        # if there is more than one row, make a list of pairs
        # [[y1_min, y1_max], [y2_min, y2_max]]
        vertical_crop = p['nd2ToTIFF']['2row_crop']

    # number between 0 and 9, 0 is no compression, 9 is most compression.
    tif_compress = p['nd2ToTIFF']['tiff_compress']

    # set up image and analysis folders if they do not already exist
    if not os.path.exists(p['TIFF_dir']):
        os.makedirs(p['TIFF_dir'])

    # Load ND2 files into a list for processing
    if p['nd2ToTIFF']['external_directory']:
        nd2files = glob.glob(os.path.join(p['nd2ToTIFF']['external_directory'], "*.nd2"))
        mm3_helpers.information("Found %d files to analyze from external directory." % len(nd2files))
    else:
        mm3_helpers.information("Experiment directory: {:s}".format(p['experiment_directory']))
        nd2files = glob.glob(os.path.join(p['experiment_directory'], "*.nd2"))
        mm3_helpers.information("Found %d files to analyze in experiment directory." % len(nd2files))

    for nd2_file in nd2files:
        file_prefix = os.path.split(os.path.splitext(nd2_file)[0])[1]
        mm3_helpers.information('Extracting %s ...' % file_prefix)

        # load the nd2. the nd2f file object has lots of mm3_helpers.information thanks to pims
        with pims_nd2.ND2_Reader(nd2_file) as nd2f:
            try:
                starttime = nd2f.metadata['time_start_jdn'] # starttime is jd
                mm3_helpers.information('Starttime got from nd2 metadata.')
            except ValueError:
                # problem with the date
                jdn = mm3_helpers.julian_day_number()
                nd2f._lim_metadata_desc.dTimeStart = jdn
                starttime = nd2f.metadata['time_start_jdn'] # starttime is jd
                mm3_helpers.information('Starttime found from lim.')

            # get the color names out. Kinda roundabout way.
            planes = [nd2f.metadata[md]['name'] for md in nd2f.metadata if md[0:6] == u'plane_' and not md == u'plane_count']

            # this insures all colors will be saved when saving tiff
            if len(planes) > 1:
                nd2f.bundle_axes = [u'c', u'y', u'x']

            # extraction range is the time points that will be taken out. Note the indexing,
            # it is zero indexed to grab from nd2, but TIFF naming starts at 1.
            # if there is more than one FOV (len(nd2f) != 1), make sure the user input
            # last time index is before the actual time index. Ignore it.
            if (p['nd2ToTIFF']['image_start'] < 1):
                p['nd2ToTIFF']['image_start'] = 1
            if p['nd2ToTIFF']['image_end'] == 'None':
                p['nd2ToTIFF']['image_end'] = False
            if p['nd2ToTIFF']['image_end']:
                if len(nd2f) > 1 and len(nd2f) < p['nd2ToTIFF']['image_end']:
                    p['nd2ToTIFF']['image_end'] = len(nd2f)
            else:
                p['nd2ToTIFF']['image_end'] = len(nd2f)
            extraction_range = range(p['nd2ToTIFF']['image_start'],
                                     p['nd2ToTIFF']['image_end']+1)

            # loop through time points
            for t in extraction_range:
                # timepoint output name (1 indexed rather than 0 indexed)
                t_id = t - 1
                # set counter for FOV output name
                #fov = fov_naming_start

                for fov_id in range(0, nd2f.sizes[u'm']): # for every FOV
                    # fov_id is the fov index according to elements, fov is the output fov ID
                    fov = fov_id + 1

                    # skip FOVs as specified above
                    if len(user_spec_fovs) > 0 and not (fov in user_spec_fovs):
                        continue

                    # set the FOV we are working on in the nd2 file object
                    nd2f.default_coords[u'm'] = fov_id

                    # get time picture was taken
                    seconds = copy.deepcopy(nd2f[t_id].metadata['t_ms']) / 1000.
                    minutes = seconds / 60.
                    hours = minutes / 60.
                    days = hours / 24.
                    acq_time = starttime + days

                    # get physical location FOV on stage
                    x_um = nd2f[t_id].metadata['x_um']
                    y_um = nd2f[t_id].metadata['y_um']

                    # make dictionary which will be the metdata for this TIFF
                    metadata_t = { 'fov': fov,
                                   't' : t,
                                   'jd': acq_time,
                                   'x': x_um,
                                   'y': y_um,
                                   'planes': planes}
                    metadata_json = json.dumps(metadata_t)

                    # get the pixel mm3_helpers.information
                    image_data = nd2f[t_id]

                    # crop tiff if specified. Lots of flags for if there are double rows or  multiple colors
                    if vertical_crop:
                        # add extra axis to make below slicing simpler.
                        if len(image_data.shape) < 3:
                            image_data = np.expand_dims(image_data, axis=0)

                        # for just a simple crop
                        if number_of_rows == 1:

                            nc, H, W = image_data.shape
                            ylo = int(vertical_crop[0]*H)
                            yhi = int(vertical_crop[1]*H)
                            image_data = image_data[:, ylo:yhi, :]

                            # save the tiff
                            tif_filename = file_prefix + "_t%04dxy%02d.tif" % (t, fov)
                            mm3_helpers.information('Saving %s.' % tif_filename)
                            tiff.imsave(os.path.join(p['TIFF_dir'], tif_filename), image_data, description=metadata_json, compress=tif_compress, photometric='minisblack')

                        # for dealing with two rows of channel
                        elif number_of_rows == 2:
                            # cut and save top row
                            image_data_one = image_data[:,vertical_crop[0][0]:vertical_crop[0][1],:]
                            tif_filename = file_prefix + "_t%04dxy%02d_1.tif" % (t, fov)
                            mm3_helpers.information('Saving %s.' % tif_filename)
                            tiff.imsave(os.path.join(p['TIFF_dir'], tif_filename), image_data_one, description=metadata_json, compress=tif_compress, photometric='minisblack')

                            # cut and save bottom row
                            metadata_t['fov'] = fov # update metdata
                            metadata_json = json.dumps(metadata_t)
                            image_data_two = image_data[:,vertical_crop[1][0]:vertical_crop[1][1],:]
                            tif_filename = file_prefix + "_t%04dxy%02d_2.tif" % (t, fov)
                            mm3_helpers.information('Saving %s.' % tif_filename)
                            tiff.imsave(os.path.join(p['TIFF_dir'], tif_filename), image_data_two, description=metadata_json, compress=tif_compress, photometric='minisblack')

                    else: # just save the image if no cropping was done.
                        tif_filename = file_prefix + "_t%04dxy%02d.tif" % (t, fov)
                        mm3_helpers.information('Saving %s.' % tif_filename)
                        tiff.imsave(os.path.join(p['TIFF_dir'], tif_filename), image_data, description=metadata_json, compress=tif_compress, photometric='minisblack')

                    # increase FOV counter
                    fov += 1

def compile(params):
    '''mm3_Compile.py locates and slices out mother machine channels into image stacks.'''

    # Load the project parameters file
    mm3_helpers.information('Loading experiment parameters.')
    p=params

    if p['FOV']:
        if '-' in p['FOV']:
            user_spec_fovs = range(int(p['FOV'].split("-")[0]),
                                   int(p['FOV'].split("-")[1])+1)
        else:
            user_spec_fovs = [int(val) for val in p['FOV'].split(",")]
    else:
        user_spec_fovs = []

    mm3_helpers.information('Using {} threads for multiprocessing.'.format(p['num_analyzers']))

    # only analyze images up until this t point. Put in None otherwise
    if 't_end' in p['compile']:
        t_end = p['compile']['t_end']
        if t_end == 'None':
            t_end = None
    else:
        t_end = None
    # only analyze images at and after this t point. Put in None otherwise
    if 't_start' in p['compile']:
        t_start = p['compile']['t_start']
        if t_start == 'None':
            t_start = None
    else:
        t_start = None

    # create the subfolders if they don't
    if not os.path.exists(p['ana_dir']):
        os.makedirs(p['ana_dir'])
    if p['output'] == 'TIFF':
        if not os.path.exists(p['chnl_dir']):
            os.makedirs(p['chnl_dir'])
    elif p['output'] == 'HDF5':
        if not os.path.exists(p['hdf5_dir']):
            os.makedirs(p['hdf5_dir'])

    # declare mm3_helpers.information variables
    analyzed_imgs = {} # for storing get_params pool results.

    ### process TIFFs for metadata #################################################################
    if not p['compile']['do_metadata']:
        mm3_helpers.information("Loading image parameters dictionary.")

        with open(os.path.join(p['ana_dir'], 'TIFF_metadata.pkl'), 'rb') as tiff_metadata:
            analyzed_imgs = pickle.load(tiff_metadata)

    else:
        mm3_helpers.information("Finding image parameters.")

        # get all the TIFFs in the folder
        found_files = glob.glob(os.path.join(p['TIFF_dir'],'*.tif')) # get all tiffs
        found_files = [filepath.split('/')[-1] for filepath in found_files] # remove pre-path
        found_files = sorted(found_files) # should sort by timepoint

        # keep images starting at this timepoint
        if t_start is not None:
            mm3_helpers.information('Removing images before time {}'.format(t_start))
            # go through list and find first place where timepoint is equivalent to t_start
            for n, ifile in enumerate(found_files):
                string = re.compile('t{:0=3}xy|t{:0=4}xy'.format(t_start,t_start)) # account for 3 and 4 digit
                # if re.search == True then a match was found
                if re.search(string, ifile):
                    # cut off every file name prior to this one and quit the loop
                    found_files = found_files[n:]
                    break

        # remove images after this timepoint
        if t_end is not None:
            mm3_helpers.information('Removing images after time {}'.format(t_end))
            # go through list and find first place where timepoint is equivalent to t_end
            for n, ifile in enumerate(found_files):
                string = re.compile('t%03dxy|t%04dxy' % (t_end, t_end)) # account for 3 and 4 digit
                if re.search(string, ifile):
                    found_files = found_files[:n]
                    break


        # if user has specified only certain FOVs, filter for those
        if (len(user_spec_fovs) > 0):
            mm3_helpers.information('Filtering TIFFs by FOV.')
            fitered_files = []
            for fov_id in user_spec_fovs:
                fov_string = 'xy%02d' % fov_id # xy01
                fitered_files += [ifile for ifile in found_files if fov_string in ifile]

            found_files = fitered_files[:]

        # get mm3_helpers.information for all these starting tiffs
        if len(found_files) > 0:
            mm3_helpers.information("Found %d image files." % len(found_files))
        else:
            mm3_helpers.warning('No TIFF files found')

        if p['compile']['find_channels_method'] == 'peaks':

            # initialize pool for analyzing image metadata
            pool = Pool(p['num_analyzers'])

            # loop over images and get mm3_helpers.information
            for fn in found_files:
                # get_params gets the image metadata and puts it in analyzed_imgs dictionary
                # for each file name. True means look for channels

                # This is the non-parallelized version (useful for debug)
                # analyzed_imgs[fn] = mm3_helpers.get_tif_params(params, fn, True)

                # Parallelized
                analyzed_imgs[fn] = pool.apply_async(mm3_helpers.get_tif_params, args=(params, fn, True))

            mm3_helpers.information('Waiting for image analysis pool to be finished.')

            pool.close() # tells the process nothing more will be added.
            pool.join() # blocks script until everything has been processed and workers exit

            mm3_helpers.information('Image analysis pool finished, getting results.')

            # get results from the pool and put them in a dictionary
            for fn in analyzed_imgs.keys():
                result = analyzed_imgs[fn]
                if result.successful():
                    analyzed_imgs[fn] = result.get() # put the metadata in the dict if it's good
                else:
                    analyzed_imgs[fn] = False # put a false there if it's bad

        elif p['compile']['find_channels_method'] == 'Unet':
            # Use Unet trained on trap and central channel locations to locate, crop, and align traps
            mm3_helpers.information("Identifying channel locations and aligning images using U-net.")

            # load model to pass to algorithm
            mm3_helpers.information("Loading model...")

            # if namespace.modelfile:
            #     model_file_path = namespace.modelfile
            # else:
            model_file_path = p['compile']['model_file_traps']
            # *** Need parameter for weights
            model = models.load_model(model_file_path, custom_objects={'mm3_helpers.tversky_loss': mm3_helpers.tversky_loss,'mm3_helpers.cce_tversky_loss': mm3_helpers.cce_tversky_loss})
            mm3_helpers.information("Model loaded.")

            # initialize pool for getting image metadata
            pool = Pool(p['num_analyzers'])

            # loop over images and get mm3_helpers.information
            for fn in found_files:
                # get_params gets the image metadata and puts it in analyzed_imgs dictionary
                # for each file name. Won't look for channels, just gets the metadata for later use by Unet

                # This is the non-parallelized version (useful for debug)
                # analyzed_imgs[fn] = mm3_helpers.get_initial_tif_params(params, fn)

                # Parallelized
                analyzed_imgs[fn] = pool.apply_async(mm3_helpers.get_initial_tif_params, args=(params,fn,))

            mm3_helpers.information('Waiting for image metadata pool to be finished.')
            pool.close() # tells the process nothing more will be added.
            pool.join() # blocks script until everything has been processed and workers exit

            mm3_helpers.information('Image metadata pool finished, getting results.')

            # get results from the pool and put them in a dictionary
            for fn in analyzed_imgs.keys():
               result = analyzed_imgs[fn]
               if result.successful():
                   analyzed_imgs[fn] = result.get() # put the metadata in the dict if it's good
               else:
                   analyzed_imgs[fn] = False # put a false there if it's bad

            # print(analyzed_imgs)

            # set up some variables for Unet and image aligment/cropping
            file_names = [key for key in analyzed_imgs.keys()]
            file_names.sort() # sort the file names by time
            file_names = np.asarray(file_names)
            fov_ids = [analyzed_imgs[key]['fov'] for key in analyzed_imgs.keys()]

            unique_fov_ids = np.unique(fov_ids)

            if p['compile']['do_channel_masks']:
                channel_masks = {}

            for fov_id in unique_fov_ids:

                mm3_helpers.information('Performing trap segmentation for fov_id: {}'.format(fov_id))
                #print(analyzed_imgs)
                fov_indices = np.where(fov_ids == fov_id)[0]
                # print(fov_indices)
                fov_file_names = [file_names[idx] for idx in fov_indices]
                trap_align_metadata = {'first_frame_name': fov_file_names[0],
                                    'frame_count': len(fov_file_names),
                                    'plane_number': len(analyzed_imgs[fn]['planes']),
                                    'trap_height': p['compile']['trap_crop_height'],
                                    'trap_width': p['compile']['trap_crop_width'],
                                    'phase_plane': p['phase_plane'],
                                    'phase_plane_index': p['moviemaker']['phase_plane_index'],
                                    'shift_distance': 256,
                                    'full_frame_size': 2048}

                dilator = np.ones((1,300))

                # create weights for taking weighted mean of several runs of Unet over various crops of the first image in the series. This helps remove "blind spots" from the neural network at the edges of each crop of the original image.
                stack_weights = get_weights_array(np.zeros((trap_align_metadata['full_frame_size'],trap_align_metadata['full_frame_size'])), trap_align_metadata['shift_distance'], subImageNumber=16, padSubImageNumber=25)[0,...]
                # print(stack_weights.shape) #uncomment for debugging

                # get prediction of where traps are located in first image
                imgPath = os.path.join(p['experiment_directory'], p['image_directory'],
                                       trap_align_metadata['first_frame_name'])
                img = io.imread(imgPath)
                # detect if there are multiple imaging channels, and rearrange image if necessary, keeping only the phase image
                img = permute_image(img, trap_align_metadata)
                if p['debug']:
                    io.imshow(img/np.max(img))
                    plt.title("Initial phase image")
                    plt.show()

                # produces predition stack with 3 "pages", index 0 is for traps, index 1 is for central tough, index 2 is for background
                mm3_helpers.information("Predicting trap locations for first frame.")
                first_frame_trap_prediction = get_frame_predictions(img,
                                                                        model,
                                                                        stack_weights,
                                                                        trap_align_metadata['shift_distance'],
                                                                        subImageNumber=16,
                                                                        padSubImageNumber=25,
                                                                        debug=p['debug'])

                if p['debug']:
                    fig,ax = plt.subplots(nrows=1, ncols=4, figsize=(12,12))
                    ax[0].imshow(img)
                    for i in range(first_frame_trap_prediction.shape[-1]):
                        ax[i+1].imshow(first_frame_trap_prediction[:,:,i])
                    plt.show()

                # flatten prediction stack such that each pixel of the resulting 2D image is the index of the prediction image above with the highest predicted probability
                class_predictions = np.argmax(first_frame_trap_prediction, axis=2)

                traps = class_predictions == 0 # returns boolean array where our intial guesses at trap locations are True

                if p['debug']:
                    io.imshow(traps)
                    plt.title('Initial trap masks')
                    plt.show()

                trap_labels = measure.label(traps)
                trap_props = measure.regionprops(trap_labels)

                trap_area_threshold = p['compile']['trap_area_threshold']
                trap_bboxes = get_frame_trap_bounding_boxes(trap_labels,
                                                                   trap_props,
                                                                   trapAreaThreshold=trap_area_threshold,
                                                                   trapWidth=trap_align_metadata['trap_width'],
                                                                   trapHeight=trap_align_metadata['trap_height'])

                # create boolean array to contain filtered, correctly-shaped trap bounding boxes
                first_frame_trap_mask = np.zeros(traps.shape)
                for i,bbox in enumerate(trap_bboxes):
                    first_frame_trap_mask[bbox[0]:bbox[2],bbox[1]:bbox[3]] = True

                good_trap_labels = measure.label(first_frame_trap_mask)
                good_trap_props = measure.regionprops(good_trap_labels)

                # widen the traps to merge them into "trap regions" above and below the central trough
                dilated_traps = morphology.dilation(first_frame_trap_mask, dilator)

                if p['debug']:
                    io.imshow(dilated_traps)
                    plt.title('Dilated trap masks')
                    plt.show()

                dilated_trap_labels = measure.label(dilated_traps)
                dilated_trap_props = measure.regionprops(dilated_trap_labels)
                # filter merged trap regions by area
                areas = [reg.area for reg in dilated_trap_props]
                labels = [reg.label for reg in dilated_trap_props]

                for idx,area in enumerate(areas):
                    if area < p['compile']['merged_trap_region_area_threshold']:

                        label = labels[idx]
                        dilated_traps[dilated_trap_labels == label] = 0

                dilated_trap_labels = measure.label(dilated_traps)
                dilated_trap_props = measure.regionprops(dilated_trap_labels)

                if p['debug']:
                    io.imshow(dilated_traps)
                    plt.title("Area-filtered dilated traps")
                    plt.show()

                # get centroids for each "trap region" identified in first frame
                centroids = np.round(np.asarray([reg.centroid for reg in dilated_trap_props]))
                if p['debug']:
                    print(centroids)

                # test whether we could crop a (512,512) square from each "trap region", with the centroids as the centers of the crops, withoug going out-of-bounds
                top_test = centroids[:,0]-256 > 0
                bottom_test = centroids[:,0]+256 < dilated_trap_labels.shape[0]
                test_array = np.stack((top_test,bottom_test))

                # get the index of the first identified "trap region" that we can get our (512,512) crop from, use that centroid for nucleus of cropping a stack of phase images with shape (frame_number,512,512,1) from all images in series
                if p['debug']:
                    print(test_array)
                    print(np.all(test_array,axis=0))

                good_trap_region_index = np.where(np.all(test_array, axis=0))[0][0]
                centroid = centroids[good_trap_region_index,:].astype('uint16')
                if p['debug']:
                    print(centroid)

                # get the (frame_number,512,512,1)-sized stack for image aligment
                align_region_stack = np.zeros((trap_align_metadata['frame_count'],512,512,1), dtype='uint16')

                for frame,fn in enumerate(fov_file_names):
                    imgPath = os.path.join(p['experiment_directory'],p['image_directory'],fn)
                    frame_img = io.imread(imgPath)
                    # detect if there are multiple imaging channels, and rearrange image if necessary, keeping only the phase image
                    frame_img = permute_image(frame_img, trap_align_metadata)
                    align_region_stack[frame,:,:,0] = frame_img[centroid[0]-256:centroid[0]+256,
                                                             centroid[1]-256:centroid[1]+256]

                # if p['debug']:
                #     colNum = 10
                #     fig,ax = plt.subplots(ncols=colNum, figsize=(20,20))

                #     for pltIdx in range(colNum):
                #         ax[pltIdx].imshow(align_region_stack[pltIdx*10,:,:,0])

                #     plt.title('Alignment stack images');
                #     plt.show();

                # run model on all frames
                batch_size=p['compile']['channel_prediction_batch_size']
                mm3_helpers.information("Predicting trap regions for (512,512) slice through all frames.")

                data_gen_args = {'batch_size':batch_size,
                         'n_channels':1,
                         'normalize_to_one':True,
                         'shuffle':False}
                predict_gen_args = {'verbose':1,
                        'use_multiprocessing':True,
                        'workers':p['num_analyzers']}
                # predict_gen_args = {'verbose':1,
                #         'use_multiprocessing':False}

                img_generator = TrapSegmentationDataGenerator(align_region_stack, **data_gen_args)

                align_region_predictions = model.predict(img_generator, **predict_gen_args)
                #align_region_stack = mm3.apply_median_filter_and_normalize(align_region_stack)
                #align_region_predictions = model.predict(align_region_stack, batch_size=batch_size)
                # reduce dimensionality such that the class predictions are now (frame_number,512,512), and each voxel is labelled as the predicted region, i.e., 0=trap, 1=central trough, 2=background.
                align_region_class_predictions = np.argmax(align_region_predictions, axis=3)

                # if p['debug']:
                #     colNum = 10
                #     fig,ax = plt.subplots(ncols=colNum, figsize=(20,20))

                #     for pltIdx in range(colNum):
                #         ax[pltIdx].imshow(align_region_class_predictions[pltIdx*10,:,:])

                #     plt.title('Alignment stack predictions');
                #     plt.show();

                # get boolean array where trap predictions are True
                align_traps = align_region_class_predictions == 0

                # if p['debug']:
                #     colNum = 10
                #     fig,ax = plt.subplots(ncols=colNum, figsize=(20,20))

                #     for pltIdx in range(colNum):
                #         ax[pltIdx].imshow(align_traps[pltIdx*10,:,:])

                #     plt.title('Alignment trap masks');
                #     plt.show();

                # allocate array to store filtered traps over time
                align_trap_mask_stack = np.zeros(align_traps.shape)
                for frame in range(trap_align_metadata['frame_count']):

                    frame_trap_labels = measure.label(align_traps[frame,:,:])
                    frame_trap_props = measure.regionprops(frame_trap_labels)

                    trap_bboxes = get_frame_trap_bounding_boxes(frame_trap_labels,
                                                                    frame_trap_props,
                                                                    trapAreaThreshold=trap_area_threshold,
                                                                    trapWidth=trap_align_metadata['trap_width'],
                                                                    trapHeight=trap_align_metadata['trap_height'])

                    for i,bbox in enumerate(trap_bboxes):
                        align_trap_mask_stack[frame,bbox[0]:bbox[2],bbox[1]:bbox[3]] = True

                # if p['debug']:
                #     colNum = 10
                #     fig,ax = plt.subplots(ncols=colNum, figsize=(20,20))

                #     for pltIdx in range(colNum):
                #         ax[pltIdx].imshow(align_trap_mask_stack[pltIdx*10,:,:])

                #     plt.title('Filtered alignment trap masks');
                #     plt.show();

                labelled_align_trap_mask_stack = measure.label(align_trap_mask_stack)

                trapTriggered = False
                for frame in range(trap_align_metadata['frame_count']):
                    anyTraps = np.any(labelled_align_trap_mask_stack[frame,:,:] > 0)
                    # if anyTraps is False, that means no traps were detected for this frame. This usuall occurs due to a bug in our imaging system,
                    #    which can cause it to miss the occasional frame. Should be fine to snag labels from prior frame.
                    if not anyTraps:
                        trapTriggered = True
                        mm3_helpers.information("Frame at index {} has no detected traps. Borrowing labels from an adjacent frame.".format(frame))
                        if frame > 0:
                            labelled_align_trap_mask_stack[frame,:,:] = labelled_align_trap_mask_stack[frame-1,:,:]
                        else:
                            labelled_align_trap_mask_stack[frame,:,:] = labelled_align_trap_mask_stack[frame+1,:,:]

                if trapTriggered:
                    repaired_align_trap_mask_stack = labelled_align_trap_mask_stack > 0
                    labelled_align_trap_mask_stack = measure.label(repaired_align_trap_mask_stack)

                align_trap_props = measure.regionprops(labelled_align_trap_mask_stack)

                areas = np.array([trap.area for trap in align_trap_props])
                labels = [trap.label for trap in align_trap_props]
                good_align_trap_props = []
                bad_align_trap_props = []
                #mode_area = stats.mode(areas)[0]
                expected_area = trap_align_metadata['trap_width'] * trap_align_metadata['trap_height'] * trap_align_metadata['frame_count']

                if p['debug']:
                    pprint(areas)
                    print(expected_area)

                    if not expected_area in areas:
                        print("No trap has expected total area. Saving labelled masks for debugging as labelled_align_trap_mask_stack.tif")
                        io.imsave("labelled_align_trap_mask_stack.tif", labelled_align_trap_mask_stack.astype('uint8'))
                        io.imsave("masks.tif", align_traps.astype('uint8'))
                        # occasionally our microscope misses an image, resulting in no traps for a single frame. This obviously messes up image alignment here....

                for trap in align_trap_props:
                    if trap.area != expected_area:
                        bad_align_trap_props.append(trap.label)
                    else:
                        good_align_trap_props.append(trap)

                for label in bad_align_trap_props:
                    labelled_align_trap_mask_stack[labelled_align_trap_mask_stack == label] = 0

                align_centroids = []
                for frame in range(trap_align_metadata['frame_count']):
                    align_centroids.append([reg.centroid for reg in measure.regionprops(labelled_align_trap_mask_stack[frame,:,:])])

                align_centroids = np.asarray(align_centroids)
                shifts = np.mean(align_centroids - align_centroids[0,:,:], axis=1)
                integer_shifts = np.round(shifts).astype('int16')

                good_trap_bboxes_dict = {}
                for trap in good_trap_props:
                    good_trap_bboxes_dict[trap.label] = trap.bbox

                # pprint(good_trap_bboxes_dict) # uncomment for debugging
                bbox_shift_dict = shift_bounding_boxes(good_trap_bboxes_dict, integer_shifts, img.shape[0])
                # pprint(bbox_shift_dict) # uncomment for debugging

                trap_images_fov_dict, trap_closed_end_px_dict =crop_traps(fov_file_names, good_trap_props, good_trap_labels, bbox_shift_dict, trap_align_metadata)

                for fn in fov_file_names:
                    analyzed_imgs[fn]['channels'] = trap_closed_end_px_dict[fn]

                if p['compile']['do_channel_masks']:
                    fov_channel_masks = make_channel_masks_CNN(bbox_shift_dict)
                    channel_masks[fov_id] = fov_channel_masks
                    # pprint(channel_masks) # uncomment for debugging

                if p['compile']['do_slicing']:

                    if p['output'] == "TIFF":

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            mm3_helpers.save_tiffs(params, trap_images_fov_dict, analyzed_imgs, fov_id)

                    elif p['output'] == "HDF5":
                        # Or write it to hdf5
                        mm3_helpers.save_hdf5(params, trap_images_fov_dict, fov_file_names, analyzed_imgs, fov_id, channel_masks)

        # save metadata to a .pkl and a human readable txt file
        mm3_helpers.information('Saving metadata from analyzed images...')
        with open(os.path.join(p['ana_dir'], 'TIFF_metadata.pkl'), 'wb') as tiff_metadata:
            pickle.dump(analyzed_imgs, tiff_metadata, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(p['ana_dir'], 'TIFF_metadata.txt'), 'w') as tiff_metadata:
            pprint(analyzed_imgs, stream=tiff_metadata)
        mm3_helpers.information('Saved metadata from analyzed images.')

    ### Make table for jd time to FOV and time point
    if not p['compile']['do_time_table']:
       mm3_helpers.information('Skipping time table creation.')
    else:
        time_table = mm3_helpers.make_time_table(params, analyzed_imgs)

    ### Make consensus channel masks and get other shared metadata #################################
    if not p['compile']['do_channel_masks'] and p['compile']['do_slicing']:
        channel_masks = mm3_helpers.load_channel_masks(params)

    elif p['compile']['do_channel_masks']:

        if p['compile']['find_channels_method'] == 'peaks':
            # only calculate channels masks from images before t_end in case it is specified
            if t_start:
                analyzed_imgs = {fn : i_metadata for fn, i_metadata in six.iteritems(analyzed_imgs) if i_metadata['t'] >= t_start}
            if t_end:
                analyzed_imgs = {fn : i_metadata for fn, i_metadata in six.iteritems(analyzed_imgs) if i_metadata['t'] <= t_end}

            # Uses channelinformation from the already processed image data
            channel_masks = mm3_helpers.make_masks(params, analyzed_imgs)

        elif p['compile']['find_channels_method'] == 'Unet':

            # save the channel mask dictionary to a pickle and a text file
            with open(os.path.join(p['ana_dir'], 'channel_masks.pkl'), 'wb') as cmask_file:
                pickle.dump(channel_masks, cmask_file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(p['ana_dir'], 'channel_masks.txt'), 'w') as cmask_file:
                pprint(channel_masks, stream=cmask_file)

    ### Slice and write TIFF files into channels ###################################################
    if p['compile']['do_slicing']:

        mm3_helpers.information("Saving channel slices.")
        if p['compile']['find_channels_method'] == 'peaks':

            # do it by FOV. Not set up for multiprocessing
            for fov, peaks in six.iteritems(channel_masks):

                # skip fov if not in the group
                if user_spec_fovs and fov not in user_spec_fovs:
                    continue

                mm3_helpers.information("Loading images for FOV %03d." % fov)

                # get filenames just for this fov along with the julian date of acquistion
                send_to_write = [[k, v['t']] for k, v in six.iteritems(analyzed_imgs) if v['fov'] == fov]

                # sort the filenames by jdn
                send_to_write = sorted(send_to_write, key=lambda time: time[1])

                if p['output'] == 'TIFF':
                    #This is for loading the whole raw tiff stack and then slicing through it
                    mm3_helpers.tiff_stack_slice_and_write(params, send_to_write, channel_masks, analyzed_imgs)

                elif p['output'] == 'HDF5':
                    # Or write it to hdf5
                    mm3_helpers.hdf5_stack_slice_and_write(params, send_to_write, channel_masks, analyzed_imgs)

            mm3_helpers.information("Channel slices saved.")

def fov_plot_channels(fov_id, crosscorrs, specs, outputdir='.', phase_plane='c1'):
    '''
    Creates a plot with the channels with guesses for empties and full channels.
    The plot is saved in PDF format.

    Parameters
    fov_id : str
        file name of the hdf5 file name in originals
    crosscorrs : dictionary
        dictionary for cross correlation values for all fovs.
    specs: dictionary
        dictionary for channal assignment (Analyze/Don't Analyze/Background).

    '''

    mm3_helpers.information("Plotting channels for FOV %d." % fov_id)

    # set up figure for user assited choosing
    n_peaks = len(specs[fov_id].keys())
    axw=1
    axh=4*axw
    nrows=3
    ncols=int(n_peaks)
    fig = plt.figure(num='none', facecolor='w',figsize=(ncols*axw,nrows*axh))
    gs = gridspec.GridSpec(nrows,ncols,wspace=0.5,hspace=0.1,top=0.90)

    # plot the peaks peak by peak using sorted list
    sorted_peaks = sorted([peak_id for peak_id in specs[fov_id].keys()])
    npeaks = len(sorted_peaks)

    for n, peak_id in enumerate(sorted_peaks):
        if crosscorrs:
            peak_xc = crosscorrs[fov_id][peak_id] # get cross corr data from dict

        # load data for figure
        image_data = mm3_helpers.load_stack(params, fov_id, peak_id, color=phase_plane)

        first_img = rescale_intensity(image_data[0,:,:]) # phase image at t=0
        last_img = rescale_intensity(image_data[-1,:,:]) # phase image at end

        # append an axis handle to ax list while adding a subplot to the figure which has a
        axhi = fig.add_subplot(gs[0,n])
        axmid = fig.add_subplot(gs[1,n])
        axlo = fig.add_subplot(gs[2,n])

        # plot the first image in each channel in top row
        ax=axhi
        ax.imshow(first_img,cmap=plt.cm.gray, interpolation='nearest')
        ax.axis('off')
        ax.set_title(str(peak_id), fontsize = 12)
        if n == 0:
            ax.set_ylabel("first time point")

        # plot middle row using last time point with highlighting for empty/full
        ax=axmid
        ax.axis('off')
        #ax.imshow(last_img,cmap=plt.cm.gray, interpolation='nearest')
        #H,W = last_img.shape
        #img = np.zeros((H,W,3))
        if specs[fov_id][peak_id] == 1: # 1 means analyze, show green
            #img[:,:,1]=last_img
            cmap=plt.cm.Greens_r
        elif specs[fov_id][peak_id] == 0: # 0 means reference, show blue
            #img[:,:,2]=last_img
            cmap=plt.cm.Blues_r
        else: # otherwise show red, means don't analyze
            #img[:,:,0]=last_img
            cmap=plt.cm.Reds_r
        ax.imshow(last_img,cmap=cmap, interpolation='nearest')

        # format
        if n == 0:
            ax.set_ylabel("last time point")

        # finally plot the cross correlations a cross time
        ax=axlo
        if crosscorrs: # don't try to plot if it's not there.
            ccs = peak_xc['ccs'] # list of cc values
            ax.plot(ccs,range(len(ccs)))
            ax.set_title('avg=%1.2f' % peak_xc['cc_avg'], fontsize = 8)
        else:
            ax.plot(np.zeros(10), range(10))

        ax.get_xaxis().set_ticks([0.8,0.9,1.0])
        ax.set_xlim((0.8,1))
        ax.tick_params('x',labelsize=8)
        if not n == 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel("time index, CC on X")


    fig.suptitle("FOV {:d}".format(fov_id),fontsize=14)
    fileout=os.path.join(outputdir,'fov_xy{:03d}.pdf'.format(fov_id))
    fig.savefig(fileout,bbox_inches='tight',pad_inches=0)
    plt.close('all')
    mm3_helpers.information("Written FOV {}'s channels in {}".format(fov_id,fileout))

    return specs

def fov_cell_segger_plot_channels(fov_id, predictionDict, specs, outputdir='.', phase_plane='c1'):
    '''
    Creates a plot with the channels with guesses for empties and full channels.
    The plot is saved in PDF format.

    Parameters
    fov_id : str
        file name of the hdf5 file name in originals
    predictionDict : dictionary
        dictionary for cross correlation values for all fovs.
    specs: dictionary
        dictionary for channal assignment (Analyze/Don't Analyze/Background).

    '''

    mm3_helpers.information("Plotting channels for FOV %d." % fov_id)

    # set up figure for user assited choosing
    n_peaks = len(specs[fov_id].keys())
    axw=1
    axh=4*axw
    nrows=3
    ncols=int(n_peaks)
    fig = plt.figure(num='none', facecolor='w',figsize=(ncols*axw,nrows*axh))
    gs = gridspec.GridSpec(nrows,ncols,wspace=0.5,hspace=0.1,top=0.90)

    # plot the peaks peak by peak using sorted list
    sorted_peaks = sorted([peak_id for peak_id in specs[fov_id].keys()])
    npeaks = len(sorted_peaks)

    for n, peak_id in enumerate(sorted_peaks):
        if predictionDict:
            predictions = predictionDict[fov_id][peak_id] # get predictions array

        # load data for figure
        image_data = mm3_helpers.load_stack(params, fov_id, peak_id, color=phase_plane)

        first_img = rescale_intensity(image_data[0,:,:]) # phase image at t=0
        last_img = rescale_intensity(image_data[-1,:,:]) # phase image at end

        # append an axis handle to ax list while adding a subplot to the figure which has a
        axhi = fig.add_subplot(gs[0,n])
        axmid = fig.add_subplot(gs[1,n])
        axlo = fig.add_subplot(gs[2,n])

        # plot the first image in each channel in top row
        ax=axhi
        ax.imshow(first_img,cmap=plt.cm.gray, interpolation='nearest')
        ax.axis('off')
        ax.set_title(str(peak_id), fontsize = 12)
        if n == 0:
            ax.set_ylabel("first time point")

        # plot middle row using last time point with highlighting for empty/full
        ax=axmid
        ax.axis('off')
        #ax.imshow(last_img,cmap=plt.cm.gray, interpolation='nearest')
        #H,W = last_img.shape
        #img = np.zeros((H,W,3))
        if specs[fov_id][peak_id] == 1: # 1 means analyze, show green
            #img[:,:,1]=last_img
            cmap=plt.cm.Greens_r
        elif specs[fov_id][peak_id] == 0: # 0 means reference, show blue
            #img[:,:,2]=last_img
            cmap=plt.cm.Blues_r
        else: # otherwise show red, means don't analyze
            #img[:,:,0]=last_img
            cmap=plt.cm.Reds_r
        ax.imshow(last_img,cmap=cmap, interpolation='nearest')

        # format
        if n == 0:
            ax.set_ylabel("last time point")

        # finally plot the prediction values as horizontal bar chart
        ax=axlo
        if predictionDict:
            ax.barh(range(len(predictions)), predictions)
            #ax.vlines(x=p['channel_picker']['channel_picking_threshold'], ymin=-1, ymax=5, linestyles='dashed',colors='red')
            ax.set_title('cell count', fontsize = 8)
        else:
            ax.plot(np.zeros(10), range(10))

        # ax.set_xlim((0,1)) # set limits to (0,1)
        #ax.get_xaxis().set_ticks([])
        if not n == 0:
            ax.get_yaxis().set_ticks([])
        else:
            ax.set_yticklabels(labels=["","1","2","3","4","5"])
            ax.set_ylabel("")

    fig.suptitle("FOV {:d}".format(fov_id),fontsize=14)
    fileout=os.path.join(outputdir,'fov_xy{:03d}.pdf'.format(fov_id))
    fig.savefig(fileout,bbox_inches='tight',pad_inches=0)
    plt.close('all')
    mm3_helpers.information("Written FOV {}'s channels in {}".format(fov_id,fileout))

    return specs

# function which makes the UI plot
def fov_choose_channels_UI(fov_id, crosscorrs, specs, UI_images):
    '''Creates a plot with the channels with guesses for empties and full channels,
    and requires the user to choose which channels to use for analysis and which to
    average for empties and subtraction.

    Parameters
    fov_file : str
        file name of the hdf5 file name in originals
    fov_xcorrs : dictionary
        dictionary for cross correlation values for all fovs.

    Returns
    bgdr_peaks : list
        list of peak id's (int) of channels to be used for subtraction
    spec_file_pkl : pickle file
        saves the lists cell_peaks, bgrd_peaks, and drop_peaks to a pkl file

    '''

    mm3_helpers.information("Starting channel picking for FOV %d." % fov_id)

    # define functions here so they have access to variables
    # for UI. change specification of channel
    def onclick_cells(event):
        try:
            peak_id = int(event.inaxes.get_title())
        except AttributeError:
            return

        # reset image to be updated based on user clicks
        ax_id = sorted_peaks.index(peak_id) * 3 + 1
        new_img = last_imgs[sorted_peaks.index(peak_id)]
        ax[ax_id].imshow(new_img, cmap=plt.cm.gray, interpolation='nearest')

        # if it says analyze, change to empty
        if specs[fov_id][peak_id] == 1:
            specs[fov_id][peak_id] = 0
            ax[ax_id].imshow(np.dstack((ones_array*0.1, ones_array*0.1, ones_array)), alpha=0.25)
            #mm3.mm3_helpers.information("peak %d now set to empty." % peak_id)

        # if it says empty, change to don't analyze
        elif specs[fov_id][peak_id] == 0:
            specs[fov_id][peak_id] = -1
            ax[ax_id].imshow(np.dstack((ones_array, ones_array*0.1, ones_array*0.1)), alpha=0.25)
            #mm3.mm3_helpers.information("peak %d now set to ignore." % peak_id)

        # if it says don't analyze, change to analyze
        elif specs[fov_id][peak_id] == -1:
            specs[fov_id][peak_id] = 1
            ax[ax_id].imshow(np.dstack((ones_array*0.1, ones_array, ones_array*0.1)), alpha=0.25)
            #mm3.mm3_helpers.information("peak %d now set to analyze." % peak_id)

        plt.draw()
        return

    # set up figure for user assited choosing
    n_peaks = len(specs[fov_id].keys())
    fig = plt.figure(figsize=(int(n_peaks/2), 12))
    fig.set_size_inches(int(n_peaks/2),12)
    ax = [] # for axis handles

    # plot the peaks peak by peak using sorted list
    sorted_peaks = sorted([peak_id for peak_id in specs[fov_id].keys()])
    npeaks = len(sorted_peaks)
    last_imgs = [] # list that holds last images for updating figure

    for n, peak_id in enumerate(sorted_peaks, start=1):
        if crosscorrs:
            peak_xc = crosscorrs[fov_id][peak_id] # get cross corr data from dict

        # load data for figure
        # image_data = mm3_helpers.load_stack(params, fov_id, peak_id, color='c1')

        # first_img = rescale_intensity(image_data[0,:,:]) # phase image at t=0
        # last_img = rescale_intensity(image_data[-1,:,:]) # phase image at end
        last_imgs.append(UI_images[fov_id][peak_id]['last']) # append for updating later
        # del image_data # clear memory (maybe)

        # append an axis handle to ax list while adding a subplot to the figure which has a
        # column for each peak and 3 rows

        # plot the first image in each channel in top row
        ax.append(fig.add_subplot(3, npeaks, n))
        ax[-1].imshow(UI_images[fov_id][peak_id]['first'],
                      cmap=plt.cm.gray, interpolation='nearest')
        ax = format_channel_plot(ax, peak_id) # format axis and title
        if n == 1:
            ax[-1].set_ylabel("first time point")

        # plot middle row using last time point with highlighting for empty/full
        ax.append(fig.add_subplot(3, npeaks, n + npeaks))
        ax[-1].imshow(UI_images[fov_id][peak_id]['last'],
                      cmap=plt.cm.gray, interpolation='nearest')

        # color image based on if it is thought empty or full
        ones_array = np.ones_like(UI_images[fov_id][peak_id]['last'])
        if specs[fov_id][peak_id] == 1: # 1 means analyze, show green
            ax[-1].imshow(np.dstack((ones_array*0.1, ones_array, ones_array*0.1)), alpha=0.25)
        else: # otherwise show red, means don't analyze
            ax[-1].imshow(np.dstack((ones_array, ones_array*0.1, ones_array*0.1)), alpha=0.25)

        # format
        ax = format_channel_plot(ax, peak_id)
        if n == 1:
            ax[-1].set_ylabel("last time point")

        # finally plot the cross correlations a cross time
        ax.append(fig.add_subplot(3, npeaks, n + 2*npeaks))
        if crosscorrs: # don't try to plot if it's not there.
            ccs = peak_xc['ccs'] # list of cc values
            ax[-1].plot(ccs, range(len(ccs)))
            ax[-1].set_title('avg=%1.2f' % peak_xc['cc_avg'], fontsize = 8)
        else:
            pass
            # ax[-1].plot(np.zeros(10), range(10))

        ax[-1].set_xlim((0.8,1))
        ax[-1].get_xaxis().set_ticks([])
        if not n == 1:
            ax[-1].get_yaxis().set_ticks([])
        else:
            ax[-1].set_ylabel("time index, CC on X")

    # show the plot finally
    fig.suptitle("FOV %d" % fov_id)

    # enter user input
    # ask the user to correct cell/nocell calls
    cells_handler = fig.canvas.mpl_connect('button_press_event', onclick_cells)
    #plt.close() resolves the event loop running error but the window stops getting displayed
    #https://stackoverflow.com/questions/27280777/pylab-show-qcoreapplicationexec-the-event-loop-is-already-running

    # matplotlib has difefrent behavior for interactions in different versions.
    if float(mpl.__version__[:3]) < 1.5: # check for verions less than 1.5
        plt.show(block=False)
        raw_input("Click colored channels to toggle between analyze (green), use for empty (blue), and ignore (red).\nPrees enter to go to the next FOV.")
    else:
        print("Click colored channels to toggle between analyze (green), use for empty (blue), and ignore (red).\nClose figure to go to the next FOV.")
        plt.show(block=True)
    fig.canvas.mpl_disconnect(cells_handler)
    plt.close()

    return specs

def fov_choose_channels_UI_II(fov_id, crosscorrs, specs, UI_images, params):

    if not os.path.isdir(params['channel_select_dir']):
        os.mkdir(params['channel_select_dir'])

    n_peaks = len(specs[fov_id].keys())
    fig = plt.figure(figsize=(int(n_peaks/2), 9))
    fig.set_size_inches(int(n_peaks/2),9)
    ax=[]
    # plot the peaks peak by peak using sorted list
    sorted_peaks = sorted([peak_id for peak_id in specs[fov_id].keys()])
    npeaks = len(sorted_peaks)
    last_imgs = [] # list that holds last images for updating figure

    for n, peak_id in enumerate(sorted_peaks, start=1):
        if crosscorrs:
            peak_xc = crosscorrs[fov_id][peak_id] # get cross corr data from dict

        # load data for figure
        # image_data = mm3_helpers.load_stack(params, fov_id, peak_id, color='c1')

        # first_img = rescale_intensity(image_data[0,:,:]) # phase image at t=0
        # last_img = rescale_intensity(image_data[-1,:,:]) # phase image at end
        last_imgs.append(UI_images[fov_id][peak_id]['last']) # append for updating later
        # del image_data # clear memory (maybe)

        # append an axis handle to ax list while adding a subplot to the figure which has a
        # column for each peak and 3 rows

        # plot the first image in each channel in top row
        ax.append(fig.add_subplot(3, npeaks, n))
        ax[-1].imshow(UI_images[fov_id][peak_id]['first'],
                      cmap=plt.cm.gray, interpolation='nearest')
        ax = format_channel_plot(ax, peak_id) # format axis and title
        if n == 1:
            ax[-1].set_ylabel("first time point")

        # plot middle row using last time point with highlighting for empty/full
        ax.append(fig.add_subplot(3, npeaks, n + npeaks))
        ax[-1].imshow(UI_images[fov_id][peak_id]['last'],
                      cmap=plt.cm.gray, interpolation='nearest')

        # # color image based on if it is thought empty or full
        # ones_array = np.ones_like(UI_images[fov_id][peak_id]['last'])
        # if specs[fov_id][peak_id] == 1: # 1 means analyze, show green
        #     ax[-1].imshow(np.dstack((ones_array*0.1, ones_array, ones_array*0.1)), alpha=0.25)
        # else: # otherwise show red, means don't analyze
        #     ax[-1].imshow(np.dstack((ones_array, ones_array*0.1, ones_array*0.1)), alpha=0.25)

        # format
        ax = format_channel_plot(ax, peak_id)
        if n == 1:
            ax[-1].set_ylabel("last time point")

        # finally plot the cross correlations a cross time
        ax.append(fig.add_subplot(3, npeaks, n + 2*npeaks))
        if crosscorrs: # don't try to plot if it's not there.
            ccs = peak_xc['ccs'] # list of cc values
            ax[-1].plot(ccs, range(len(ccs)))
            ax[-1].set_title('avg=%1.2f' % peak_xc['cc_avg'], fontsize = 8)
        else:
            pass
            # ax[-1].plot(np.zeros(10), range(10))

        ax[-1].set_xlim((0.8,1))
        ax[-1].get_xaxis().set_ticks([])
        if not n == 1:
            ax[-1].get_yaxis().set_ticks([])
        else:
            ax[-1].set_ylabel("time index, CC on X")

    # show the plot finally
    fig.suptitle("FOV %d" % fov_id)
    plt.tight_layout(pad=0.2)
    plt.savefig(os.path.join(params['channel_select_dir'],f"{fov_id}.png"))
    plt.close()

    mm3_helpers.information("Starting channel picking for FOV %d." % fov_id)
    im=image.imread(os.path.join(params['channel_select_dir'],f"{fov_id}.png"))

    tot=np.array(im)
    napari.current_viewer().add_image(tot, name="Fov"+str(fov_id)+"_img", visible=False)
    napari.current_viewer().add_points([], name="Fov"+str(fov_id)+"_pts", visible=False, face_color='r', size=20)
    offset=43
    # Add points in the points layer according to the cross-correlation value
    namei="Fov"+str(fov_id)+"_img"
    namep="Fov"+str(fov_id)+"_pts"
    (max_height,max_width,_)=napari.current_viewer().layers[namei].data_raw.shape
    width_per_peak=(max_width-offset)//npeaks

    for i,peak_id in enumerate(sorted_peaks):

        # Estimate the coordinates from peak id's index
        x= offset+ width_per_peak*i + (4*width_per_peak)//5

        # Adding the points at 1/3 height from bottom
        # should be good enough for visualization
        y1= (max_height)//2
        y2= (max_height)//2 + 50

        if specs[fov_id][peak_id]==0:
            # empty(0) => 1 points in the peak partition
            napari.current_viewer().layers[namep].add((y1, x))
        elif specs[fov_id][peak_id]==-1:
            # ignore(-1) => 2 points in the peak partition
            napari.current_viewer().layers[namep].add((y1, x))
            napari.current_viewer().layers[namep].add((y2, x))

    return specs

def channelProcessor(params):
    ana_dir = params['ana_dir']
    specs = yaml.safe_load(Path(ana_dir+'specs.yaml').read_text())

    if params['FOV']:
        if '-' in params['FOV']:
            user_spec_fovs = range(int(params['FOV'].split("-")[0]),
                                   int(params['FOV'].split("-")[1])+1)
        else:
            user_spec_fovs = [int(val) for val in params['FOV'].split(",")]
    else:
        user_spec_fovs = []

    # load channel masks
    channel_masks = mm3_helpers.load_channel_masks(params)

    # make list of FOVs to process (keys of channel_mask file), but only if there are channels
    fov_id_list = sorted([fov_id for fov_id, peaks in six.iteritems(channel_masks) if peaks])

    # remove fovs if the user specified so
    if (len(user_spec_fovs) > 0):
        fov_id_list = [int(fov) for fov in fov_id_list if fov in user_spec_fovs]

    # Mark all as analyze intitially
    # The points layer analysis will take care later on
    for fov_id in fov_id_list:
        sorted_peaks = sorted([peak_id for peak_id in specs[fov_id].keys()])
        for peak_id in sorted_peaks:
            specs[fov_id][peak_id]=1

    for fov_id in fov_id_list:
        sorted_peaks = sorted([peak_id for peak_id in specs[fov_id].keys()])
        npeaks = len(sorted_peaks)

        namei="Fov"+str(fov_id)+"_img"
        namep="Fov"+str(fov_id)+"_pts"
        (_,max_width,_)=napari.current_viewer().layers[namei].data_raw.shape
        pts=napari.current_viewer().layers[namep]._view_data
        # print(f'{fov_id}', pts)
        offset=43
        width_per_peak=(max_width-offset)//npeaks

        # analyze(1) => 0 points in the peak partition
        # empty(0) => 1 points in the peak partition
        # ignore(-1) => 2 points in the peak partititon

        for pt in pts:
            peak_id=sorted_peaks[max(0,min(int((pt[1]-offset)//width_per_peak), len(sorted_peaks)-1))]
            specs[fov_id][peak_id]-=1

    # Save out specs file in yaml format

    with open(os.path.join(ana_dir, 'specs.yaml'), 'w') as specs_file:
        yaml.dump(data=specs, stream=specs_file, default_flow_style=False, tags=None)
    # print(specs)
    print("Channel Picking Completed")

# function for better formatting of channel plot
def format_channel_plot(ax, peak_id):
    '''Removes axis and puts peak as title from plot for channels'''
    ax[-1].get_xaxis().set_ticks([])
    ax[-1].get_yaxis().set_ticks([])
    ax[-1].set_title(str(peak_id), fontsize = 8)
    return ax

# function to preload all images for all FOVs, hopefully saving time
def preload_images(specs, fov_id_list):
    '''This dictionary holds the first and last image
    for all channels in all FOVS. It is passed to the UI so that the
    figures can be populated much faster
    '''

    # Intialized the dicionary
    UI_images = {}

    for fov_id in fov_id_list:
        mm3_helpers.information("Preloading images for FOV {}.".format(fov_id))
        UI_images[fov_id] = {}
        for peak_id in specs[fov_id].keys():
            image_data = mm3_helpers.load_stack(params, fov_id, peak_id, color=params['phase_plane'])
            UI_images[fov_id][peak_id] = {'first' : None, 'last' : None} # init dictionary
             # phase image at t=0. Rescale intenstiy and also cut the size in half
            first_image = params['channel_picker']['first_image']
            UI_images[fov_id][peak_id]['first'] = image_data[first_image,::2,::2]
            last_image = params['channel_picker']['last_image']
            # phase image at end
            UI_images[fov_id][peak_id]['last'] = image_data[last_image,::2,::2]

    return UI_images

### For when this script is run from the terminal ##################################
def channelPicker(params):
    '''mm3_ChannelPicker.py allows the user to identify full and empty channels.'''

    # Load the project parameters file
    mm3_helpers.information('Loading experiment parameters.')
    p = params

    if p['FOV']:
        if '-' in p['FOV']:
            user_spec_fovs = range(int(p['FOV'].split("-")[0]),
                                   int(p['FOV'].split("-")[1])+1)
        else:
            user_spec_fovs = [int(val) for val in p['FOV'].split(",")]
    else:
        user_spec_fovs = []

    specfile = None

    # set cross correlation calculation flag
    # if namespace.saved_cross_correlations:
    #     do_crosscorrs = False
    # else:
    do_crosscorrs = p['channel_picker']['do_crosscorrs']

    do_CNN = p['channel_picker']['do_CNN']
    do_seg = p['channel_picker']['do_seg']

    # set interactive flag
    # if namespace.noninteractive:
    #     interactive = False
    # else:
    interactive = p['channel_picker']['interactive']

    # assign shorthand directory names
    ana_dir = os.path.join(p['experiment_directory'], p['analysis_directory'])
    chnl_dir = os.path.join(p['experiment_directory'], p['analysis_directory'], 'channels')
    hdf5_dir = os.path.join(p['experiment_directory'], p['analysis_directory'], 'hdf5')

    # load channel masks
    channel_masks = mm3_helpers.load_channel_masks(params)

    # make list of FOVs to process (keys of channel_mask file), but only if there are channels
    fov_id_list = sorted([fov_id for fov_id, peaks in six.iteritems(channel_masks) if peaks])

    # remove fovs if the user specified so
    if (len(user_spec_fovs) > 0):
        fov_id_list = [int(fov) for fov in fov_id_list if fov in user_spec_fovs]

    mm3_helpers.information("Found %d FOVs to process." % len(fov_id_list))

    ### Cross correlations ########################################################################
    if do_CNN:
        # a nested dict to hold predictions per channel per fov.
        crosscorrs = None
        predictionDict = {}

        mm3_helpers.information('Loading model ....')

        # read in model for inference of empty vs good traps
        model_file_path = p['channel_picker']['channel_picker_model_file']
        model = models.load_model(model_file_path)

        mm3_helpers.information("Model loaded.")

        for fov_id in fov_id_list:

            predictionDict[fov_id] = {}

            mm3_helpers.information('Inferring good, empty, and defective traps on fov_id {} using CNN.'.format(fov_id))

            # get list of tiff file names
            tiff_file_names = glob.glob(os.path.join(chnl_dir, "*xy{:0=3}*_c1.tif".format(fov_id)))
            tiff_file_names.sort()
            #print(len(tiff_file_names)) # uncomment for debugging

            # parameters to pass to custom image generator class, TrapKymographPredictionDataGenerator
            cnn_params = {'dim': (210,256),
                      'batch_size': 40,
                      'n_classes': 4,
                      'n_channels': 1,
                      'shuffle': False}
            # set up the image data generator
            channel_image_generator = TrapKymographPredictionDataGenerator(tiff_file_names, **cnn_params)

            # run the model
            predictions = model.predict(channel_image_generator)
            #print(predictions.shape)
            predictions = predictions[:len(tiff_file_names),:]
            #print(predictions.shape)

            # assign each prediction to the proper fov_id, peak_id in predictions dict
            for i,peak_id in enumerate(sorted(channel_masks[fov_id].keys())):
                # put prediction array into dictionary
                #print(i, peak_id) # uncomment for debugging
                predictionDict[fov_id][peak_id] = predictions[i,:]

        # write predictions to pickle and text
        mm3_helpers.information("Writing channel picking predictions file.")
        with open(os.path.join(ana_dir,"channel_picker_CNN_results.pkl"), 'wb') as preds_file:
            pickle.dump(predictionDict, preds_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(ana_dir,"channel_picker_CNN_results.txt"), 'w') as preds_file:
            pprint(predictionDict, stream=preds_file)
        mm3_helpers.information("Wrote channel picking predictions files.")

    elif do_seg:
        # a nested dict to hold predictions per channel per fov.
        crosscorrs = None
        predictionDict = {}

        mm3_helpers.information('Loading model ....')

        # read in model for inference of empty vs good traps
        model_file_path = p['segment']['model_file']
        model = models.load_model(model_file_path,
                                  custom_objects={'mm3_helpers.bce_dice_loss': mm3_helpers.bce_dice_loss,
                                                  'mm3_helpers.dice_loss': mm3_helpers.dice_loss})
        unet_shape = (p['segment']['trained_model_image_height'],
                      p['segment']['trained_model_image_width'])

        cellClassThreshold = p['segment']['cell_class_threshold']
        if cellClassThreshold == 'None': # yaml imports None as a string
            cellClassThreshold = False
        min_object_size = p['segment']['min_object_size']

        mm3_helpers.information("Model loaded.")

        # arguments to data generator
        data_gen_args = {'batch_size':p['segment']['batch_size'],
                         'n_channels':1,
                         'normalize_to_one':True,
                         'shuffle':False}
        # arguments to predict_generator
        predict_args = dict(use_multiprocessing=True,
                            workers=p['num_analyzers'],
                            verbose=1)

        # predict_args = dict(use_multiprocessing=False,
        #                     verbose=1)

        for fov_id in fov_id_list:

            predictionDict[fov_id] = {}

            mm3_helpers.information('Inferring number of cells in five evenly spaced frames for each trap in fov {}.'.format(fov_id))

            # assign each prediction to the proper fov_id, peak_id in predictions dict
            counter = 0
            peak_number = len(channel_masks[fov_id])
            for i,peak_id in enumerate(sorted(channel_masks[fov_id].keys())):
                # get list of tiff file names
                tiff_file_name = glob.glob(os.path.join(chnl_dir, "*xy{:0=3}_p{:0=4}_c1.tif".format(fov_id, peak_id)))[0]

                img_array = io.imread(tiff_file_name)
                img_height = img_array.shape[1]
                img_width = img_array.shape[2]
                slice_increment = int(img_array.shape[0]/5)

                # set up stack for images from all peaks
                # this is a bit more complicated than just doing 5 images at a time, but it is much faster
                #   because you don't have nearly as many data transfer steps
                if i == 0:
                    img_stack = np.zeros((5*peak_number,img_height,img_width),dtype='uint16')

                # grab 5 images to load and run cell segmentation
                for j in range(5):
                    img_stack[counter,...] = img_array[slice_increment*j,...]
                    counter += 1

            pad_dict = mm3_helpers.get_pad_distances(unet_shape, img_height, img_width)

            # pad image to correct size
            if p['debug']:
                print("Padding dictionary:", pad_dict)

            img_stack = np.pad(img_stack,
                               ((0,0),
                               (pad_dict['top_pad'],pad_dict['bottom_pad']),
                               (pad_dict['left_pad'],pad_dict['right_pad'])),
                               mode='constant')
            img_stack = np.expand_dims(img_stack, -1)

            # set up image generator
            image_generator = CellSegmentationDataGenerator(img_stack, **data_gen_args)
            # run predictions
            predictions = model.predict(image_generator, **predict_args)[:,:,:,0]
            if p['debug']:
                fig,ax = plt.subplots(ncols=5)
                for i in range(5):
                    ax[i].imshow(predictions[i,:,:])
                plt.show()

            # binarized and label (if there is a threshold value, otherwise, save a grayscale for debug)
            if cellClassThreshold:
                predictions[predictions >= cellClassThreshold] = 1
                predictions[predictions < cellClassThreshold] = 0
                predictions = predictions.astype('uint8')

                segmented_imgs = np.zeros(predictions.shape, dtype='uint8')
                # process and label each frame of the channel
                for frame in range(segmented_imgs.shape[0]):
                    # get rid of small holes
                    predictions[frame,:,:] = morphology.remove_small_holes(predictions[frame,:,:], min_object_size)
                    # get rid of small objects.
                    predictions[frame,:,:] = morphology.remove_small_objects(morphology.label(predictions[frame,:,:], connectivity=1), min_size=min_object_size)
                    # remove labels which touch the boarder
                    predictions[frame,:,:] = segmentation.clear_border(predictions[frame,:,:])
                    # relabel now
                    segmented_imgs[frame,:,:] = morphology.label(predictions[frame,:,:], connectivity=1)

            else: # in this case you just want to scale the 0 to 1 float image to 0 to 255
                mm3_helpers.information('Converting predictions to grayscale.')
                segmented_imgs = np.around(predictions * 100)

            # put number of cells detected into array for predictionDict
            counter = 0
            for i,peak_id in enumerate(sorted(channel_masks[fov_id].keys())):

                cell_count_array = np.zeros(5, dtype='uint8')
                for j in range(5):
                    cell_count_array[j] = int(np.max(segmented_imgs[counter,:,:]))
                    counter += 1

                predictionDict[fov_id][peak_id] = cell_count_array

        # write predictions to pickle and text
        mm3_helpers.information("Writing channel picking predictions file.")
        with open(os.path.join(ana_dir,"channel_picker_seg_results.pkl"), 'wb') as preds_file:
            pickle.dump(predictionDict, preds_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(ana_dir,"channel_picker_seg_results.txt"), 'w') as preds_file:
            pprint(predictionDict, stream=preds_file)
        mm3_helpers.information("Wrote channel picking predictions files.")

    elif do_crosscorrs:
        # a nested dict to hold cross corrs per channel per fov.
        crosscorrs = {}

        # for each fov find cross correlations (sending to pull)
        for fov_id in fov_id_list:
            mm3_helpers.information("Calculating cross correlations for FOV %d." % fov_id)

            # nested dict keys are peak_ids and values are cross correlations
            crosscorrs[fov_id] = {}

            # initialize pool for analyzing image metadata
            # pool = Pool(p['num_analyzers'])

            # find all peak ids in the current FOV
            for peak_id in sorted(channel_masks[fov_id].keys()):
                mm3_helpers.information("Calculating cross correlations for peak %d." % peak_id)

                # linear loop
                crosscorrs[fov_id][peak_id] = mm3_helpers.channel_xcorr(params, fov_id, peak_id)

                # multiprocessing verion
                # crosscorrs[fov_id][peak_id] = pool.apply_async(mm3_helpers.channel_xcorr, args=(params, fov_id, peak_id,))

            mm3_helpers.information('Waiting for cross correlation pool to finish for FOV %d.' % fov_id)

            # pool.close() # tells the process nothing more will be added.
            # pool.join() # blocks script until everything has been processed and workers exit

            mm3_helpers.information("Finished cross correlations for FOV %d." % fov_id)

        # get results from the pool and put the results in the dictionary if succesful
        # for fov_id, peaks in six.iteritems(crosscorrs):
        #     for peak_id, result in six.iteritems(peaks):
        #         if result.successful():
        #             # put the results, with the average, and a guess if the channel
        #             # is full into the dictionary
        #             crosscorrs[fov_id][peak_id] = {'ccs' : result.get(),
        #                                            'cc_avg' : np.average(result.get()),
        #                                            'full' : np.average(result.get()) < p['channel_picker']['channel_picking_threshold']}
        #         else:
        #             crosscorrs[fov_id][peak_id] = False # put a false there if it's bad

        # linear loop for debug
        # get results from the pool and put the results in the dictionary if succesful
        for fov_id, peaks in six.iteritems(crosscorrs):
            for peak_id, result in six.iteritems(peaks):
                crosscorrs[fov_id][peak_id] = {'ccs' : result,
                                                   'cc_avg' : np.average(result),
                                                   'full' : np.average(result) < p['channel_picker']['channel_picking_threshold']}

        # write cross-correlations to pickle and text
        mm3_helpers.information("Writing cross correlations file.")
        with open(os.path.join(ana_dir,"crosscorrs.pkl"), 'wb') as xcorrs_file:
            pickle.dump(crosscorrs, xcorrs_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(ana_dir,"crosscorrs.txt"), 'w') as xcorrs_file:
            pprint(crosscorrs, stream=xcorrs_file)
        mm3_helpers.information("Wrote cross correlations files.")

    # try to load previously calculated cross correlations
    else:
        mm3_helpers.information('Loading precalculated cross-correlations.')
        try:
            with open(os.path.join(ana_dir,'crosscorrs.pkl'), 'rb') as xcorrs_file:
                crosscorrs = pickle.load(xcorrs_file)
        except:
            crosscorrs = None
            mm3_helpers.information('Could not load cross-correlations.')

    ### User selection (channel picking) #####################################################
    if specfile == None:
        mm3_helpers.information('Initializing specifications file.')
        # nested dictionary of {fov : {peak : spec ...}) for if channel should
        # be analyzed, used for empty, or ignored.
        specs = {}

        # if there is cross corrs, use it. Otherwise, just make everything -1
        if crosscorrs:
            # update dictionary on initial guess from cross correlations
            for fov_id, peaks in six.iteritems(crosscorrs):
                specs[fov_id] = {}
                for peak_id, xcorrs in six.iteritems(peaks):
                    # update the guess incase the parameters file was changed
                    xcorrs['full'] = xcorrs['cc_avg'] < p['channel_picker']['channel_picking_threshold']

                    if xcorrs['full'] == True:
                        specs[fov_id][peak_id] = 1
                    else: # default to don't analyze
                        specs[fov_id][peak_id] = -1
        elif do_CNN:

            # update dictionary with inference from CNN

            for fov_id, peakPredictionsDict in six.iteritems(predictionDict):
                fov_id = int(fov_id)
                specs[fov_id] = {}
                for peak_id, predictions in six.iteritems(peakPredictionsDict):

                    if predictions[0] > p['channel_picker']['channel_picking_threshold']:
                        specs[fov_id][peak_id] = 1
                    else:
                        specs[fov_id][peak_id] = -1

            #pprint(specs) # uncomment for debugging

        elif do_seg:

            # update dictionary with inference from cell segmentation based decision

            for fov_id, peakPredictionsDict in six.iteritems(predictionDict):
                fov_id = int(fov_id)
                specs[fov_id] = {}
                for peak_id, predictions in six.iteritems(peakPredictionsDict):

                    # if there was at least one cell in any of the frames checked, keep the trap
                    if np.max(predictions) > 0:
                        specs[fov_id][peak_id] = 1
                    else:
                        specs[fov_id][peak_id] = -1

        else: # just set everything to 1 and go forward.
            for fov_id, peaks in six.iteritems(channel_masks):
                specs[fov_id] = {peak_id: -1 for peak_id in peaks.keys()}

    else:
        mm3_helpers.information('Loading supplied specifiication file.')
        with open(specfile, 'r') as fin:
            specs = yaml.load(fin)

    if interactive:
        # preload the images
        mm3_helpers.information('Preloading images.')
        UI_images = preload_images(specs, fov_id_list)

        mm3_helpers.information('Starting channel picking.')
        # go through the fovs again, same as above
        for fov_id in fov_id_list:

            if do_CNN:
                specs = fov_CNN_choose_channels_UI(fov_id, predictionDict, specs, UI_images)
            elif do_seg:
                specs = fov_cell_segger_choose_channels_UI(fov_id, predictionDict, specs, UI_images)
            else: # crosscorrs == None will default to just picking with no help.
                #specs = fov_choose_channels_UI(fov_id, crosscorrs, specs, UI_images)
                specs=fov_choose_channels_UI_II(fov_id, crosscorrs, specs, UI_images, params)

    else:
        outputdir = os.path.join(ana_dir, "fovs")
        if not os.path.isdir(outputdir):
            os.makedirs(outputdir)
        for fov_id in fov_id_list:
            if crosscorrs:
                specs = fov_plot_channels(fov_id, crosscorrs, specs,
                                          outputdir=outputdir, phase_plane=p['phase_plane'])
            elif do_CNN:
                specs = fov_CNN_plot_channels(fov_id, predictionDict, specs,
                                              outputdir=outputdir, phase_plane=p['phase_plane'])
            elif do_seg:
                specs = fov_cell_segger_plot_channels(fov_id, predictionDict, specs,
                                              outputdir=outputdir, phase_plane=p['phase_plane'])

    # Save out specs file in yaml format
    with open(os.path.join(ana_dir, 'specs.yaml'), 'w') as specs_file:
        yaml.dump(data=specs, stream=specs_file, default_flow_style=False, tags=None)

    mm3_helpers.information('Finished.')

def subtract(params):
    '''mm3_Subtract.py averages empty channels and then subtractions them from channels with cells'''

    # Load the project parameters file
    mm3_helpers.information('Loading experiment parameters.')
    #p = mm3_.init_mm3_helpers() # initialized the helper library
    p=params

    if p['FOV']:
        if '-' in p['FOV']:
            user_spec_fovs = range(int(p['FOV'].split("-")[0]),
                                   int(p['FOV'].split("-")[1])+1)
        else:
            user_spec_fovs = [int(val) for val in p['FOV'].split(",")]
    else:
        user_spec_fovs = []

    mm3_helpers.information('Using {} threads for multiprocessing.'.format(p['num_analyzers']))

    sub_plane = 'c1'

    # Create folders for subtracted info if they don't exist
    if p['output'] == 'TIFF':
        if not os.path.exists(p['empty_dir']):
            os.makedirs(p['empty_dir'])
        if not os.path.exists(p['sub_dir']):
            os.makedirs(p['sub_dir'])

    # load specs file
    specs = mm3_helpers.load_specs(params)

    # make list of FOVs to process (keys of specs file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    mm3_helpers.information("Found %d FOVs to process." % len(fov_id_list))

    # determine if we are doing fluorescence or phase subtraction, and set flags
    if sub_plane == p['phase_plane']:
        align = True # used when averaging empties
        sub_method = 'phase' # used in mm3_helpers.subtract_fov_stack
    else:
        align = False
        sub_method = 'fluor'

    ### Make average empty channels ###############################################################
    if not p['subtract']['do_empties']:
        mm3_helpers.information("Loading precalculated empties.")
        pass # just skip this part and go to subtraction

    else:
        mm3_helpers.information("Calculating averaged empties for channel {}.".format(sub_plane))

        need_empty = [] # list holds fov_ids of fov's that did not have empties
        for fov_id in fov_id_list:
            # send to function which will create empty stack for each fov.
            averaging_result = mm3_helpers.average_empties_stack(params, fov_id, specs,
                                                         color=sub_plane, align=align)
            # add to list for FOVs that need to be given empties from other FOvs
            if not averaging_result:
                need_empty.append(fov_id)

        # deal with those problem FOVs without empties
        have_empty = list(set(fov_id_list).difference(set(need_empty))) # fovs with empties
        for fov_id in need_empty:
            from_fov = min(have_empty, key=lambda x: abs(x-fov_id)) # find closest FOV with an empty
            copy_result = mm3_helpers.copy_empty_stack(params, from_fov, fov_id, color=sub_plane)

    ### Subtract ##################################################################################
    if p['subtract']['do_subtraction']:
        mm3_helpers.information("Subtracting channels for channel {}.".format(sub_plane))
        for fov_id in fov_id_list:
            # send to function which will create empty stack for each fov.
            subtraction_result = mm3_helpers.subtract_fov_stack(params, fov_id, specs,
                                                        color=sub_plane, method=sub_method)
        mm3_helpers.information("Finished subtraction.")

    # Else just end, they only wanted to do empty averaging.
    else:
        mm3_helpers.information("Skipping subtraction.")
        pass

def segmentUNet(params):

    mm3_helpers.information('Loading experiment parameters.')
    p=params

    if p['FOV']:
        if '-' in p['FOV']:
            user_spec_fovs = range(int(p['FOV'].split("-")[0]),
                                   int(p['FOV'].split("-")[1])+1)
        else:
            user_spec_fovs = [int(val) for val in p['FOV'].split(",")]
    else:
        user_spec_fovs = []

    mm3_helpers.information('Using {} threads for multiprocessing.'.format(p['num_analyzers']))

    # create segmenteation and cell data folder if they don't exist
    if not os.path.exists(p['seg_dir']) and p['output'] == 'TIFF':
        os.makedirs(p['seg_dir'])
    if not os.path.exists(p['cell_dir']):
        os.makedirs(p['cell_dir'])

    # set segmentation image name for saving and loading segmented images
    p['seg_img'] = 'seg_unet'
    p['pred_img'] = 'pred_unet'

    # load specs file
    specs = mm3_helpers.load_specs(params)
    # print(specs) # for debugging

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    mm3_helpers.information("Processing %d FOVs." % len(fov_id_list))

    ### Do Segmentation by FOV and then peak #######################################################
    mm3_helpers.information("Segmenting channels using U-net.")

    # load model to pass to algorithm
    mm3_helpers.information("Loading model...")

    # if namespace.modelfile:
    #     model_file_path = namespace.modelfile
    # else:
    model_file_path = p['segment']['model_file']
    # *** Need parameter for weights
    seg_model = models.load_model(model_file_path,
                              custom_objects={'mm3_helpers.bce_dice_loss': mm3_helpers.bce_dice_loss,
                                              'mm3_helpers.dice_loss': mm3_helpers.dice_loss})
    mm3_helpers.information("Model loaded.")

    for fov_id in fov_id_list:
        mm3_helpers.segment_fov_unet(params, fov_id, specs, seg_model, color=p['phase_plane'])

    del seg_model
    mm3_helpers.information("Finished segmentation.")

def segmentOTSU(params):

    mm3_helpers.information('Loading experiment parameters.')
    p=params

    if p['FOV']:
        if '-' in p['FOV']:
            user_spec_fovs = range(int(p['FOV'].split("-")[0]),
                                   int(p['FOV'].split("-")[1])+1)
        else:
            user_spec_fovs = [int(val) for val in p['FOV'].split(",")]
    else:
        user_spec_fovs = []

    # create segmenteation and cell data folder if they don't exist
    if not os.path.exists(p['seg_dir']) and p['output'] == 'TIFF':
        os.makedirs(p['seg_dir'])
    if not os.path.exists(p['cell_dir']):
        os.makedirs(p['cell_dir'])

    # set segmentation image name for saving and loading segmented images
    p['seg_img'] = 'seg_otsu'

    # load specs file
    specs = mm3_helpers.load_specs(params)

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    mm3_helpers.information("Segmenting %d FOVs." % len(fov_id_list))

    ### Do Segmentation by FOV and then peak #######################################################
    mm3_helpers.information("Segmenting channels using Otsu method.")

    for fov_id in fov_id_list:
        # determine which peaks are to be analyzed (those which have been subtracted)
        ana_peak_ids = []
        for peak_id, spec in six.iteritems(specs[fov_id]):
            if spec == 1: # 0 means it should be used for empty, -1 is ignore, 1 is analyzed
                ana_peak_ids.append(peak_id)
        ana_peak_ids = sorted(ana_peak_ids) # sort for repeatability

        for peak_id in ana_peak_ids:
            # send to segmentation
            mm3_helpers.segment_chnl_stack(params, fov_id, peak_id)

    mm3_helpers.information("Finished segmentation.")

def Track(params):

    # Load the project parameters file
    mm3_helpers.information('Loading experiment parameters.')
    p=params

    if p['FOV']:
        if '-' in p['FOV']:
            user_spec_fovs = range(int(p['FOV'].split("-")[0]),
                                   int(p['FOV'].split("-")[1])+1)
        else:
            user_spec_fovs = [int(val) for val in p['FOV'].split(",")]
    else:
        user_spec_fovs = []

    mm3_helpers.information('Using {} threads for multiprocessing.'.format(p['num_analyzers']))

    mm3_helpers.information("Using {} images for tracking.".format(p['track']['seg_img']))

    # create segmenteation and cell data folder if they don't exist
    if not os.path.exists(p['seg_dir']) and p['output'] == 'TIFF':
        os.makedirs(p['seg_dir'])
    if not os.path.exists(p['cell_dir']):
        os.makedirs(p['cell_dir'])

    # load specs file
    specs = mm3_helpers.load_specs(params)

    # make list of FOVs to process (keys of channel_mask file)
    fov_id_list = sorted([fov_id for fov_id in specs.keys()])

    # remove fovs if the user specified so
    if user_spec_fovs:
        fov_id_list[:] = [fov for fov in fov_id_list if fov in user_spec_fovs]

    ### Create cell lineages from segmented images
    mm3_helpers.information("Creating cell lineages using standard algorithm.")

    # Load time table, which goes into params
    mm3_helpers.load_time_table(params)

    # This dictionary holds mm3_helpers.information for all cells
    Cells = {}

    # do lineage creation per fov, so pooling can be done by peak
    for fov_id in fov_id_list:
        # update will add the output from make_lineages_function, which is a
        # dict of Cell entries, into Cells
        Cells.update(mm3_helpers.make_lineages_fov(fov_id, specs))

    mm3_helpers.information("Finished lineage creation.")

    ### Now prune and save the data.
    mm3_helpers.information("Curating and saving cell data.")

    # this returns only cells with a parent and daughters
    Complete_Cells = mm3_helpers.find_complete_cells(Cells)

    ### save the cell data. Use the script mm3_OutputData for additional outputs.
    # All cell data (includes incomplete cells)
    with open(p['cell_dir'] + '/all_cells.pkl', 'wb') as cell_file:
        pickle.dump(Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    # Just the complete cells, those with mother and daugther
    # This is a dictionary of cell objects.
    with open(os.path.join(p['cell_dir'],'complete_cells.pkl'), 'wb') as cell_file:
        pickle.dump(Complete_Cells, cell_file, protocol=pickle.HIGHEST_PROTOCOL)

    mm3_helpers.information("Finished curating and saving cell data.")

def find_cells_of_fov_and_peak(Cells, fov_id, peak_id):
    '''Return only cells from a specific fov/peak
    Parameters
    ----------
    fov_id : int corresponding to FOV
    peak_id : int correstonging to peak
    '''

    fCells = {} # f is for filtered

    for cell_id in Cells:
        if Cells[cell_id].fov == fov_id and Cells[cell_id].peak == peak_id:
            fCells[cell_id] = Cells[cell_id]

    return fCells

def plot_lineage_images(Cells, fov_id, peak_id, Cells2=None, bgcolor='c1', fgcolor='seg', plot_tracks=True, trim_time=False, time_set=(0,100), t_adj=1):
    '''
    Plot linages over images across time points for one FOV/peak.
    Parameters
    ----------
    bgcolor : Designation of background to use. Subtracted images look best if you have them.
    fgcolor : Designation of foreground to use. This should be a segmented image.
    Cells2 : second set of linages to overlay. Useful for comparing lineage output.
    plot_tracks : bool
        If to plot cell traces or not.
    t_adj : int
        adjust time indexing for differences between t index of image and image number
    '''

    # filter cells
    Cells = find_cells_of_fov_and_peak(Cells, fov_id, peak_id)

    # load subtracted and segmented data
    image_data_bg = mm3_helpers.load_stack(params, fov_id, peak_id, color=bgcolor)

    if fgcolor:
        image_data_seg = mm3_helpers.load_stack(params, fov_id, peak_id, color=fgcolor)

    if trim_time:
        image_data_bg = image_data_bg[time_set[0]:time_set[1]]
        if fgcolor:
            image_data_seg = image_data_seg[time_set[0]:time_set[1]]

    n_imgs = image_data_bg.shape[0]
    image_indicies = range(n_imgs)

    if fgcolor:
        # calculate the regions across the segmented images
        regions_by_time = [regionprops(timepoint) for timepoint in image_data_seg]

        # Color map for good label colors
        vmin = 0.5 # values under this color go to black
        vmax = 100 # max y value
        cmap = mpl.colors.ListedColormap(sns.husl_palette(vmax, h=0.5, l=.8, s=1))
        cmap.set_under(color='black')

    # Trying to get the image size down
    figxsize = image_data_bg.shape[2] * n_imgs / 100.0
    figysize = image_data_bg.shape[1] / 100.0

    # plot the images in a series
    fig, axes = plt.subplots(ncols=n_imgs, nrows=1,
                             figsize=(figxsize, figysize),
                             facecolor='black', edgecolor='black')
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    transFigure = fig.transFigure.inverted()

    # change settings for each axis
    ax = axes.flat # same as axes.ravel()
    for a in ax:
        a.set_axis_off()
        a.set_aspect('equal')
        ttl = a.title
        ttl.set_position([0.5, 0.05])

    for i in image_indicies:
        ax[i].imshow(image_data_bg[i], cmap=plt.cm.gray, aspect='equal')

        if fgcolor:
            # make a new version of the segmented image where the
            # regions are relabeled by their y centroid position.
            # scale it so it falls within 100.
            seg_relabeled = image_data_seg[i].copy().astype(np.float)
            for region in regions_by_time[i]:
                rescaled_color_index = region.centroid[0]/image_data_seg.shape[1] * vmax
                seg_relabeled[seg_relabeled == region.label] = int(rescaled_color_index)-0.1 # subtract small value to make it so there is not overlabeling
            ax[i].imshow(seg_relabeled, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)

        ax[i].set_title(str(i + t_adj), color='white')

    # save just the segmented images
    # lin_dir = params['experiment_directory'] + params['analysis_directory'] + 'lineages/'
    # if not os.path.exists(lin_dir):
    #     os.makedirs(lin_dir)
    # lin_filename = params['experiment_name'] + '_xy%03d_p%04d_nolin.png' % (fov_id, peak_id)
    # lin_filepath = lin_dir + lin_filename
    # fig.savefig(lin_filepath, dpi=75)
    # plt.close()

    # Annotate each cell with mm3_helpers.information
    if plot_tracks:
        for cell_id in Cells:
            for n, t in enumerate(Cells[cell_id].times):
                t -= t_adj # adjust for special indexing

                # don't look at time points out of the interval
                if trim_time:
                    if t < time_set[0] or t >= time_set[1]-1:
                        break

                x = Cells[cell_id].centroids[n][1]
                y = Cells[cell_id].centroids[n][0]

                # add a circle at the centroid for every point in this cell's life
                circle = mpatches.Circle(xy=(x, y), radius=2, color='white', lw=0, alpha=0.5)
                ax[t].add_patch(circle)

                # draw connecting lines between the centroids of cells in same lineage
                try:
                    if n < len(Cells[cell_id].times) - 1:
                        # coordinates of the next centroid
                        x_next = Cells[cell_id].centroids[n+1][1]
                        y_next = Cells[cell_id].centroids[n+1][0]
                        t_next = Cells[cell_id].times[n+1] - t_adj # adjust for special indexing

                        # get coordinates for the whole figure
                        coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
                        coord2 = transFigure.transform(ax[t_next].transData.transform([x_next, y_next]))

                        # create line
                        line = mpl.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                                transform=fig.transFigure,
                                                color='white', lw=1, alpha=0.5)

                        # add it to plot
                        fig.lines.append(line)
                except:
                    pass

                # draw connecting between mother and daughters
                try:
                    if n == len(Cells[cell_id].times)-1 and Cells[cell_id].daughters:
                        # daughter ids
                        d1_id = Cells[cell_id].daughters[0]
                        d2_id = Cells[cell_id].daughters[1]

                        # both daughters should have been born at the same time.
                        t_next = Cells[d1_id].times[0] - t_adj

                        # coordinates of the two daughters
                        x_d1 = Cells[d1_id].centroids[0][1]
                        y_d1 = Cells[d1_id].centroids[0][0]
                        x_d2 = Cells[d2_id].centroids[0][1]
                        y_d2 = Cells[d2_id].centroids[0][0]

                        # get coordinates for the whole figure
                        coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
                        coordd1 = transFigure.transform(ax[t_next].transData.transform([x_d1, y_d1]))
                        coordd2 = transFigure.transform(ax[t_next].transData.transform([x_d2, y_d2]))

                        # create line and add it to plot for both
                        for coord in [coordd1, coordd2]:
                            line = mpl.lines.Line2D((coord1[0],coord[0]),(coord1[1],coord[1]),
                                                    transform=fig.transFigure,
                                                    color='white', lw=1, alpha=0.5, ls='dashed')
                            # add it to plot
                            fig.lines.append(line)
                except:
                    pass

        # this is for plotting the traces from a second set of cells
        if Cells2 and plot_tracks:
            Cells2 = find_cells_of_fov_and_peak(Cells2, fov_id, peak_id)
            for cell_id in Cells2:
                for n, t in enumerate(Cells2[cell_id].times):
                    t -= t_adj

                    # don't look at time points out of the interval
                    if trim_time:
                        if t < time_set[0] or t >= time_set[1]-1:
                            break

                    x = Cells2[cell_id].centroids[n][1]
                    y = Cells2[cell_id].centroids[n][0]

                    # add a circle at the centroid for every point in this cell's life
                    circle = mpatches.Circle(xy=(x, y), radius=2, color='yellow', lw=0, alpha=0.25)
                    ax[t].add_patch(circle)

                    # draw connecting lines between the centroids of cells in same lineage
                    try:
                        if n < len(Cells2[cell_id].times) - 1:
                            # coordinates of the next centroid
                            x_next = Cells2[cell_id].centroids[n+1][1]
                            y_next = Cells2[cell_id].centroids[n+1][0]
                            t_next = Cells2[cell_id].times[n+1] - t_adj

                            # get coordinates for the whole figure
                            coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
                            coord2 = transFigure.transform(ax[t_next].transData.transform([x_next, y_next]))

                            # create line
                            line = mpl.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                                                    transform=fig.transFigure,
                                                    color='yellow', lw=1, alpha=0.25)

                            # add it to plot
                            fig.lines.append(line)
                    except:
                        pass

                    # draw connecting between mother and daughters
                    try:
                        if n == len(Cells2[cell_id].times)-1 and Cells2[cell_id].daughters:
                            # daughter ids
                            d1_id = Cells2[cell_id].daughters[0]
                            d2_id = Cells2[cell_id].daughters[1]

                            # both daughters should have been born at the same time.
                            t_next = Cells2[d1_id].times[0] - t_adj

                            # coordinates of the two daughters
                            x_d1 = Cells2[d1_id].centroids[0][1]
                            y_d1 = Cells2[d1_id].centroids[0][0]
                            x_d2 = Cells2[d2_id].centroids[0][1]
                            y_d2 = Cells2[d2_id].centroids[0][0]

                            # get coordinates for the whole figure
                            coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
                            coordd1 = transFigure.transform(ax[t_next].transData.transform([x_d1, y_d1]))
                            coordd2 = transFigure.transform(ax[t_next].transData.transform([x_d2, y_d2]))

                            # create line and add it to plot for both
                            for coord in [coordd1, coordd2]:
                                line = mpl.lines.Line2D((coord1[0],coord[0]),(coord1[1],coord[1]),
                                                        transform=fig.transFigure,
                                                        color='yellow', lw=1, alpha=0.25, ls='dashed')
                                # add it to plot
                                fig.lines.append(line)
                    except:
                        pass

            # this is for putting cell id on first time cell appears and when it divides
            # this is broken, need to translate coordinates to correct location on figure.
            # if n == 0 or n == len(Cells[cell_id].times)-1:
            #     print(x/100.0, y/100.0, cell_id[9:])
            #     ax[t].text(x/100.0, y/100.0, cell_id[9:], color='red', size=10, ha='center', va='center')

    return fig, ax

def find_cells_of_birth_label(Cells, label_num=1):
    '''Return only cells whose starting region label is given.
    If no birth_label is given, returns the mother cells.
    label_num can also be a list to include cells of many birth labels
    '''

    fCells = {} # f is for filtered

    if type(label_num) is int:
        label_num = [label_num]

    for cell_id in Cells:
        if Cells[cell_id].birth_label in label_num:
            fCells[cell_id] = Cells[cell_id]

    return fCells

def Lineage(params,numSamples=10):
    # plotting lineage trees for complete cells
    # load specs file
    with open(os.path.join(params['ana_dir'], 'specs.yaml'), 'r') as specs_file:
        specs = yaml.safe_load(specs_file)
    with open(os.path.join(params['cell_dir'], 'all_cells.pkl'), 'rb') as cell_file:
        Cells = pickle.load(cell_file)
    with open(os.path.join(params['cell_dir'], 'complete_cells.pkl'), 'rb') as cell_file:
        Cells2 = pickle.load(cell_file)
        Cells2 = find_cells_of_birth_label(Cells2, label_num=[1,2])

    lin_dir = os.path.join(params['experiment_directory'], params['analysis_directory'],
                        'lineages')
    if not os.path.exists(lin_dir):
        os.makedirs(lin_dir)

    # determine number of lineages to make
    no_samples = numSamples
    # list contains tuples of (fov_id, peak_id, timepoint)
    sample_ids = []
    for fov_id in specs.keys():
        for peak_id, spec in specs[fov_id].items():
            if spec == 1:
                # now put the fov, peak, and a random timepoint in a tuple
                sample_ids.append((fov_id, peak_id))

    # choose a random subset of these and sort them
    sample_indicies = np.random.choice(len(sample_ids), no_samples)
    sample_ids = sorted([sample_ids[i] for i in sample_indicies])

    for sample in sample_ids:
        fov_id, peak_id = sample
        fig, ax = plot_lineage_images(Cells, fov_id, peak_id, Cells2,
                                                bgcolor=params['phase_plane'])
        lin_filename = params['experiment_name'] + '_xy%03d_p%04d_lin.png' % (fov_id, peak_id)
        lin_filepath = os.path.join(lin_dir, lin_filename)
        fig.savefig(lin_filepath, dpi=75)
        img = image.imread(lin_filepath)
        napari.current_viewer().add_image(img, name=lin_filename, visible=False)
        plt.close(fig)

    mm3_helpers.information("Completed Plotting")

def Compile(experiment_name: str='exp1', experiment_directory: str= '/Users/sharan/Desktop/exp1/', image_directory:str='TIFF/', external_directory: str= '/Users/sharan/Desktop/exp1/',  analysis_directory:str= 'analysis/', FOV:str='1-5', TIFF_source:str='nd2ToTIFF',
output:str='TIFF', debug:str= False, pxl2um:float= 0.11, phase_plane: str ='c1', image_start : int=1, number_of_rows :int = 1, tiff_compress:int=5,
do_metadata: bool=True, do_time_table: bool=True, do_channel_masks: bool=True, do_slicing:bool=True, find_channels_method:str='peaks',
image_orientation : str= 'up', channel_width : int=10, channel_separation : int=45, channel_detection_snr : int=1, channel_length_pad : int=10,
channel_width_pad : int=10, trap_crop_height: int=256, trap_crop_width: int=27, trap_area_threshold: int=2, channel_prediction_batch_size: int=15,
merged_trap_region_area_threshold: int=400):
    """Performs Mother Machine Analysis"""

    global params
    params=dict()
    params['experiment_name']=experiment_name
    params['experiment_directory']=experiment_directory
    params['image_directory']=image_directory
    params['analysis_directory']=analysis_directory
    params['FOV']=FOV
    params['TIFF_source']=TIFF_source
    params['output']=output
    params['debug']=debug
    params['phase_plane']=phase_plane
    params['pxl2um']=pxl2um
    params['nd2ToTIFF']=dict()
    params['nd2ToTIFF']['image_start']=image_start
    params['nd2ToTIFF']['image_end']=None
    params['nd2ToTIFF']['number_of_rows']=number_of_rows
    params['nd2ToTIFF']['crop_ymin']=None
    params['nd2ToTIFF']['crop_ymax']=None
    params['nd2ToTIFF']['2row_crop']=None
    params['nd2ToTIFF']['tiff_compress']=tiff_compress
    params['nd2ToTIFF']['external_directory']=external_directory
    params['compile']=dict()
    params['compile']['do_metadata' ]=do_metadata
    params['compile']['do_time_table']=do_time_table
    params['compile']['do_channel_masks']=do_channel_masks
    params['compile']['do_slicing']=do_slicing
    params['compile']['t_end']=None
    params['compile']['find_channels_method']=find_channels_method
    #model_file_traps: str='/Users/sharan/Desktop/Physics/mm3-latest/weights/feature_weights_512x512_normed.hdf5',
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

    nd2ToTIFF(params)
    compile(params)
    return

def ChannelPicker(experiment_name: str='exp1', experiment_directory: str= '/Users/sharan/Desktop/exp1/', image_directory:str='TIFF/', external_directory: str= '/Users/sharan/Desktop/exp1/',  analysis_directory:str= 'analysis/', FOV:str='1-5', TIFF_source:str='nd2ToTIFF',
output:str='TIFF', debug:str= False, pxl2um:float= 0.11, phase_plane: str ='c1', do_crosscorrs:bool=True, do_CNN:bool=False, interactive:bool=True, do_seg:bool=False,
first_image: int=1, channel_picking_threshold: float =0.99, channel_picker_model_file='/Users/sharan/Desktop/Physics/mm3-latest/weights/empties_weights.hdf5', do_empties:bool=True, do_subtraction: bool=True, alignment_pad: int=10, selection_done:bool=False):
    """Performs Mother Machine Analysis"""

    global params
    params=dict()
    params['experiment_name']=experiment_name
    params['experiment_directory']=experiment_directory
    params['image_directory']=image_directory
    params['analysis_directory']=analysis_directory
    params['FOV']=FOV
    params['TIFF_source']=TIFF_source
    params['output']=output
    params['debug']=debug
    params['phase_plane']=phase_plane
    params['pxl2um']=pxl2um
    params['subtract']=dict()
    params['subtract']['do_empties']=do_empties
    params['subtract']['do_subtraction']=do_subtraction
    params['subtract']['alignment_pad']=alignment_pad
    params['channel_picker']=dict()
    params['channel_picker']['do_crosscorrs']=do_crosscorrs
    params['channel_picker']['do_CNN']=do_CNN
    params['channel_picker']['interactive']=interactive
    params['channel_picker']['do_seg']=do_seg
    params['channel_picker']['first_image']=first_image
    params['channel_picker']['last_image']=-1
    params['channel_picker']['channel_picking_threshold']=channel_picking_threshold
    params['channel_picker']['channel_picker_model_file']=channel_picker_model_file

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
    params['channel_select_dir']= os.path.join(params['ana_dir'], 'channel_picker')

    # use jd time in image metadata to make time table. Set to false if no jd time
    if params['TIFF_source'] == 'elements' or params['TIFF_source'] == 'nd2ToTIFF':
        params['use_jd'] = True
    else:
        params['use_jd'] = False

    if selection_done:
        channelProcessor(params)
    else:
        channelPicker(params)
    return

def Segment(experiment_name: str='exp1', experiment_directory: str= '/Users/sharan/Desktop/exp1/', model_file: str='/Users/sharan/Desktop/exp1/20200921_MG1655_256x32.hdf5', image_directory:str='TIFF/', external_directory:str= '/Users/sharan/Desktop/exp1/',  analysis_directory:str= 'analysis/', FOV:str='1-5', TIFF_source:str='nd2ToTIFF',
output:str='TIFF', debug:str= False, pxl2um:float= 0.11, phase_plane: str ='c1', do_empties:bool=True, do_subtraction: bool=True, alignment_pad: int=10, do_segmentation=True, do_lineages=True,  OTSU_threshold: float= 1.0, first_opening_size: int=2,
distance_threshold: int=2, second_opening_size: int=1, min_object_size:int= 25, trained_model_image_height: int=256, trained_model_image_width: int=32,
batch_size: int=210, cell_class_threshold: float= 0.60, normalize_to_one:bool= False, save_predictions:bool=False, OTSU :bool=True, UNet: bool=False):
    """Performs Mother Machine Analysis"""

    global params
    params=dict()
    params['experiment_name']=experiment_name
    params['experiment_directory']=experiment_directory
    params['image_directory']=image_directory
    params['analysis_directory']=analysis_directory
    params['external_directory']=external_directory
    params['FOV']=FOV
    params['TIFF_source']=TIFF_source
    params['output']=output
    params['debug']=debug
    params['phase_plane']=phase_plane
    params['pxl2um']=pxl2um
    params['subtract']=dict()
    params['subtract']['do_empties']=do_empties
    params['subtract']['do_subtraction']=do_subtraction
    params['subtract']['alignment_pad']=alignment_pad
    params['segment']=dict()
    params['segment']['do_segmentation']=do_segmentation
    params['segment']['do_lineages']=do_lineages
    params['segment']['OTSU_threshold']=OTSU_threshold
    params['segment']['first_opening_size']=first_opening_size
    params['segment']['distance_threshold']=distance_threshold
    params['segment']['second_opening_size']=second_opening_size
    params['segment']['min_object_size']=min_object_size
    params['segment']['model_file']=model_file
    params['segment']['trained_model_image_height']=trained_model_image_height
    params['segment']['trained_model_image_width']=trained_model_image_width
    params['segment']['batch_size']=batch_size
    params['segment']['cell_class_threshold']=cell_class_threshold
    params['segment']['save_predictions']=save_predictions
    params['segment']['normalize_to_one']=normalize_to_one
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

    subtract(params)

    if OTSU:
        params['seg_img'] = 'seg_otsu'
        segmentOTSU(params)

    if UNet:
        params['seg_img'] = 'seg_unet'
        params['pred_img'] = 'pred_unet'
        segmentUNet(params)

    return

def Track_Standard(experiment_name: str='exp1', experiment_directory: str= '/Users/sharan/Desktop/exp1/', image_directory:str='TIFF/', external_directory:str= '/Users/sharan/Desktop/exp1/',  analysis_directory:str= 'analysis/', FOV:str='1-5', TIFF_source:str='nd2ToTIFF',
output:str='TIFF', debug:str= False, pxl2um:float= 0.11, phase_plane: str ='c1', lost_cell_time:int= 3, new_cell_y_cutoff:int= 150, new_cell_region_cutoff:float= 4, max_growth_length:float= 1.5, min_growth_length:float= 0.7, max_growth_area:float= 1.5, min_growth_area:float= 0.7 , numSamples:int=10, seg_img :str='seg_otsu'):
    """Performs Mother Machine Analysis"""

    global params
    params=dict()
    params['experiment_name']=experiment_name
    params['experiment_directory']=experiment_directory
    params['image_directory']=image_directory
    params['analysis_directory']=analysis_directory
    params['external_directory']=external_directory
    params['FOV']=FOV
    params['TIFF_source']=TIFF_source
    params['output']=output
    params['debug']=debug
    params['phase_plane']=phase_plane
    params['pxl2um']=pxl2um
    params['num_analyzers'] = multiprocessing.cpu_count()
    params['track']=dict()
    params['track']['lost_cell_time']= lost_cell_time
    params['track']['new_cell_y_cutoff']= new_cell_y_cutoff
    params['track']['new_cell_region_cutoff']= new_cell_region_cutoff
    params['track']['max_growth_length']= max_growth_length
    params['track']['min_growth_length']= min_growth_length
    params['track']['max_growth_area']= max_growth_area
    params['track']['min_growth_area']= min_growth_area
    params['track']['seg_img']=seg_img

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

    Track(params)
    Lineage(params, numSamples)
    return
