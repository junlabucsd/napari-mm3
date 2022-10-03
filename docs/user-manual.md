# User Manual

## Overview
Generally, each step of the pipeline has a single widget.
This assumes you are using Otsu segmentation -- the procedure can be modified if you are using U-Net.

0. [nd2ToTIFF](/docs/user-manual.md#nd2ToTIFF) -- Turn your microscopy data into TIFFs. 
1. [Compile](/docs/user-manual.md#compile) -- Locate traps, separate them into their own TIFFs, and return metadata.
2. [PickChannels](/docs/user-manual.md#pickchannels) -- User guided selection of empty and full traps.
3. [Subtract](/docs/user-manual.md#subtract) -- Remove (via subtraction) empty traps from the background of traps that contain cells. 
4. [SegmentOtsu](/docs/user-manual.md#segmentotsu) -- Use Otsu segmentation to segment cells.
5. [Track](/docs/user-manual.md#track) -- Acquire individual cell properties and track lineages.

Additionally, we have a few widgets to assist in other tasks that may come up:

6. [Annotate](/docs/user-manual.md#annotate)-- annotate images for ML (U-Net or similar) training purposes.
7. [SegmentUnet](/docs/user-manual.md#segmentunet) -- Run U-Net segmentation (you will need to supply your own model)
8. [Colors](/docs/user-manual.md#colors) -- Calculate fluorescence information.
9. [Foci](/docs/user-manual.md#foci) -- We use this to track `foci' (bright fluorescent spots) inside of cells.

For additional information, you may wish to refer to the following documents:

[Input image guidelines](/docs/input-images-guidelines.md)

[File structure](/docs/file-structure.md)

[Output file structure](/docs/Cell-class-docs.md)


<a name="nd2ToTIFF"></a>
## 0. Generating a TIFF stack 

**Input**
* .nd2 file as produced by Nikon Elements

**Output**
* Individual TIFF files. 

mm3 currently currently takes individual TIFF images as its input. 
If there are multiple color layers, then each TIFF image should be a stack of planes corresponding to a color. 
The quality of your images is important for mm3 to work properly.

The working directory now contains:
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
```

### Notes on metadata

mm3_nd2ToTIFF.py reads the metadata directly from the .nd2 and then writes it into the header of the TIFF file when saving. The format is a json representation of a Python dictionary, and is recognized later by Compile.  


 <a name="compile"></a>
## 1. Locate channels, create channel stacks, and return metadata (Compile widget).
<img width="1187" alt="fov_inspect1" src="https://user-images.githubusercontent.com/40699438/177629474-5fd7ee80-682e-4aaa-bf6e-dd547e40c458.png">

The Compile widget attempts to automatically identify and crop out individual growth channels. Images corresponding to a specific channel are then stacked in time, and these "channel stacks" are the basis of further analysis. If there are multiple colors, a channel stack is made for each color for each channel.

It is also at this time that metadata is drawn from the images and saved. 

**Parameters**

* `TIFF_source` needs to be specified to indicate how the script should look for TIFF metadata. Choices are `elements`, `nd2ToTIFF`, and `other`. `elements` indicates that the TIFFs came from nikon elements. `nd2ToTIFF` indicates that TIFF files were exported by our [nd2ToTIFF](#0-generating-a-tiff-stack) script. Finally, `other` indicates that the subsequent scripts should simply read in information from TIFF names.
* `channel_width`, `channel_separation`, and `channel_detection_snr`, which are used to help find the channels.
* `channel_length_pad` and `channel_width_pad` will increase the size of your channel slices.
* `phase_plane` is the postfix of the channel which contains the phase images
* `start time`, `end time` : Will only analyze images up to this time point. Useful for debugging.

**Outputs**

* Stacked TIFFs through time for each channel (colors saved in separate stacks). These are saved to the `channels/` subfolder in the analysis directory.
* Metadata for each TIFF in a Python dictionary. These are saved as `TIFF_metadata.pkl` and `.txt`. The pickle file is read by subsequent scripts, the text file is simply for the user (true of all metadata files).
* Channel masks for each FOV. These are saved as `channel_masks.pkl` and `.txt`. A Python dictionary that records the location of the channels in each FOV. Is a nested dictionaries of FOVs and then channel peaks. The final values are 4 pixel coordinates, ((y1, y2), (x1, x2)).
* Time table for all time points and FOVs. These are saved as `time_table.pkl` and `.txt`. A Python dictionary by FOV which maps the actual time (elapsed seconds since the start of the experiment) each nominal time point was taken.
* crosscorrs.pkl and .txt : Python dictionary that contains image correlation value for channels over time. Used to guess if a channel is full or empty. Same structure as channel_masks.


The working directory now contains:
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
├── analysis
│   ├── time_table.pkl
│   ├── time_table.txt
│   ├── TIFF_metadata.pkl
│   ├── TIFF_metadata.txt
│   ├── channel_masks.pkl
│   ├── channel_masks.txt
│   └── channels
```

<a name="pickchannels"></a> 
## 2. User guided selection of empty and full channels (PickChannels). 
<img width="1177" alt="channel_picker" src="https://user-images.githubusercontent.com/40699438/177629496-73b6c4cf-7427-41e6-ac20-720b6fbf2ba1.png">

The Compile widget identifies all growth channels, regardless of if they contain or do not contain cells. ChannelSorter first attempts to guess, and then presents the user with a GUI to decide which channels should be analyzed, which channels should be ignored, and which channels should be used as empty channels during subtraction. This information is contained within the specs.yaml file.

Clicking the "Load data" button displays the first FOV analyzed, along with the program's predicted channel classification.

Click on the colored channels until they are as you wish. To navigate between fields of view click the "next FOV" or "prior FOV" buttons.  The widget will output the specs file with channels indicated as:

| Color       | Description     | specs.yaml value |
| ----------- | --------------- | ---------------- |
| Green       | Contains Cells  | 1                |
| Red         | Ignore          | 0                |
| Blue        | Reference Empty | -1               |

Make sure to have **one reference channel** per FOV, and **at least one cell-containing channel** per FOV.

Click on a channel to change its classification.

**Parameters**

* `channel_picking_threshold` is a measure of correlation between a series of images, so a value of 1 would mean the same image over and over. Channels with values above this value (like empty channels) will be designated as empty before the user selection GUI.

The working directory is now:
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
├── analysis
│   ├── time_table.pkl
│   ├── time_table.txt
│   ├── TIFF_metadata.pkl
│   ├── TIFF_metadata.txt
│   ├── channel_masks.pkl
│   ├── channel_masks.txt
│   ├── channels
│   ├── crosscorrs.pkl
│   ├── crosscorrs.txt
│   ├── specs.yaml
```

**Output**
* specs.pkl and .txt : Python dictionary which is the specifications of channels as full (1), empty (0), or ignore (-1). Same structure as channel_masks.

<a name="subtract"></a> 
## 3. Subtract phase contrast images (Subtract widget). 
<img width="1183" alt="subtract" src="https://user-images.githubusercontent.com/40699438/177629512-c5ba4abd-0e03-4540-a4bb-7414ad0560d0.png">

This widget averages empty channel to be used for subtraction, and then subtracts the empty channel from the specified channel in the phase contrast plane.


Downstream analysis of phase contrast (brightfield) images requires background subtraction to remove artifacts of the PDMS device in the images. 

The working directory is now:
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
├── analysis
│   ├── time_table.pkl
│   ├── time_table.txt
│   ├── TIFF_metadata.pkl
│   ├── TIFF_metadata.txt
│   ├── channel_masks.pkl
│   ├── channel_masks.txt
│   ├── channels
│   ├── crosscorrs.pkl
│   ├── crosscorrs.txt
│   ├── empties
│   ├── specs.yaml
│   └── subtracted
```

**Input**
* TIFF channel stacks (phase contrast only).
* specs.txt file.

**Output**
* Averaged empty stack. Saved in the `empties/` subfolder in the analysis directory.
* Subtracted channel stacks. Saved in the `subtracted/` subfolder in the analysis directory.

### Notes on use

If for a specific FOV there are multiple empty channels designated, then those channels are averaged together by timepoint to create an averaged empty channel. If only one channel is designated in the specs file as empty, then it will simply be copied over. If no channels are designated as empty, than this FOV is skipped, and the user is required to copy one of the empty channels from `empties/` subfolder and rename with the absent FOV ID.

<a name="segmentotsu"></a> 
## 4. Segment images. 

<img width="1486" alt="otsu" src="https://user-images.githubusercontent.com/40699438/177629756-2bf87d2e-6ec8-4580-8675-648d68b29cb5.png">
<img width="1180" alt="unet" src="https://user-images.githubusercontent.com/40699438/177629546-81c2f826-73e8-41ef-adbd-7ceb191db461.png">
mm3 can use either deep learning or a traditional machine vision approach (Otsu thresholding, morphological operations and watershedding) to locate cells from the subtracted images. For info on the deep learning-based segmentation widget, see [SegmentUnet](/docs/user-manual.md#segmentunet).

The following four parameters are important for finding markers in order to do watershedding/diffusion for segmentation. They should be changed depending on cell size and magnification/imaging conditions.

**OTSU parameters**

* `first_opening_size` : Size in pixels of first morphological opening during segmentation.
* `distance_threshold` : Distance in pixels which thresholds distance transform of binary cell image.
* `second_opening_size` : Size in pixels of second morphological opening.
* `min_object_size` : Objects smaller than this area in pixels will be removed before labeling.


The working directory is now:
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
├── analysis
│   ├── time_table.pkl
│   ├── time_table.txt
│   ├── TIFF_metadata.pkl
│   ├── TIFF_metadata.txt
│   ├── channel_masks.pkl
│   ├── channel_masks.txt
│   ├── channels
│   ├── crosscorrs.pkl
│   ├── crosscorrs.txt
│   ├── empties
│   ├── segmented
│   ├── specs.yaml
│   └── subtracted
```


<a name="track"></a> 
## 5. Create cell lineages (Track widget). 

<img width="1188" alt="lineage" src="https://user-images.githubusercontent.com/40699438/177629704-b866d74e-cd80-4171-a6cf-92a887617160.png">

This widget reconstructs cell lineages from the segmented images.

After cells are found for each channel in each time point, these labeled cells are connected across time to create complete cells and lineages.

**Input**
* segmented channel TIFFs.
* specs.pkl file.

**Parameters**

* `lost_cell_time` : If this many time points pass and a region has not yet been linked to a future region, it is dropped.
* `max_growth_length` : If a region is to be connected to a previous region, it cannot be larger in length by more than this ratio.
* `min_growth_length` : If a region is to be connected to a previous region, it cannot be smaller in length by less than this ratio.
* `max_growth_area` : If a region is to be connected to a previous region, it cannot be larger in area by more than this ratio.
* `min_growth_area` : If a region is to be connected to a previous region, it cannot be smaller in area by less than this ratio.
* `new_cell_y_cutoff` : distance in pixels from closed end of image above which new regions are not considered for starting new cells.
* `segmentation_method`: whether to construct lineage from cells segmented by the Otsu or U-net method

The working directory is now:
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
├── analysis
│   ├── time_table.pkl
│   ├── time_table.txt
│   ├── TIFF_metadata.pkl
│   ├── TIFF_metadata.txt
│   ├── cell_data
│   │   └── complete_cells.pkl
│   ├── channel_masks.pkl
│   ├── channel_masks.txt
│   ├── channels
│   ├── crosscorrs.pkl
│   ├── crosscorrs.txt
│   ├── empties
│   ├── segmented
│   ├── specs.pkl
│   ├── specs.txt
│   └── subtracted
└── params.yaml
```

**Output**
* Data for all cells. Saved as a dictionary of Cell objects (see below) in the `cell_data` subfolder in the analysis directory. The file `all_cells.pkl` contains all identified cells, while `complete_cells.pkl` contains only cells with a mother and daughter identified.
* (Optional) Lineage images. Lineage and segmentation information overlayed on the subtracted images across time frames.

### Notes on use

Lineages are made by connecting the segmented regions into complete cells using basic rules. These rules include that a region in one time point must be a similar size and overlap with a region in a previous time point to which it will link. For a cell to divide, the two daughter cells' combined size must be similar and also overlap. For a cell to be considered complete, it must have both a mother and two daughters.

When cells are made during lineage creation, the information is stored per cell in an object called Cell. The Cell object is the fundamental unit of data produced by mm3. Every Cell object has a unique identifier (`id`) as well as all the other pertinent information. The default data format save by the track widget is a dictionary of these cell objecs, where the keys are each cell id and the values are the object itself. Below is a description of the information contained in a Cell object and how to export it to other formats. For an overview on classes in Python see [here](https://learnpythonthehardway.org/book/ex40.html).

For more information on the Cell object description in [Cell-class-docs.md](/docs/Cell-class-docs.md)

<a name="annotate"></a> 
## 6. Curate training data for U-net segmentation. 

<a name="segmentunet"></a> 
## 7. Run U-net segmentation

Segment cells using a U-net model.

**Parameters**

* `threshold` : threshold value (between 0 and 1) for cell classification
* `min_object_size` : Objects smaller than this area in pixels will be removed before labeling.


<a name="colors"></a> 
## 8. Fluorescence analysis.

The cell data output by the tracking widget contains information about all cells in the experiment, including which images and locations in the images they came from. We can use this to go back to additional image planes (colors) for secondary analysis. This widget computes integrated fluorescence signal per cell and similar attributes for a given input plane. See [Cell-class-docs.md] for more information.

**Input**
*  .pkl file of cell objects from Track widget.

**Output**
*  .pkl file with a dictionary of cell objects, with integrated fluorescence intensity and fluorescence per pixel and per cell volume stored as attributes of the corresponding cell objects.

<a name="foci"></a> 
## 9. Foci picking.

Finds foci using a Laplacian convolution. See https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html for details.
Foci are linked to the cell objects in which they appear.

**Input**
*  .pkl file of cell objects from Track widget.

**Output**
*  .pkl file with a dictionary of cell objects, with x, y and time positions of foci detections stored as attributes of the corresponding cell objects.

**Parameters**

* `LoG_min_sig` : Minimum sigma of laplacian to convolve in pixels. Scales with minimum foci width to detect as 2*sqrt(2)*minsig
* `LoG_max_sig`: Maximum sigma of laplacian to convolve in pixels. Scales with maximum foci width to detect as 2*sqrt(2)*maxsig
* `LoG_threshold` : Absolute threshold laplacian must reach to record potential foci. Keep low to detect dimmer spots.
* `LoG_peak_ratio` : Focus peaks must be this many times greater than median cell intensity. Think signal to noise ratio