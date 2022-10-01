<a name="nd2ToTIFF"></a>
### 0. Generating a TIFF stack 

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

 <a name="compile"></a>
### 1. Locate channels, create channel stacks, and return metadata (Compile widget).
<img width="1187" alt="fov_inspect1" src="https://user-images.githubusercontent.com/40699438/177629474-5fd7ee80-682e-4aaa-bf6e-dd547e40c458.png">

The Compile widget attempts to automatically identify and crop out individual growth channels. Images corresponding to a specific channel are then stacked in time, and these "channel stacks" are the basis of further analysis. If there are multiple colors, a channel stack is made for each color for each channel.

It is also at this time that metadata is drawn from the images and saved. 

**Parameters**

* `TIFF_source` needs to be specified to indicate how the script should look for TIFF metadata. Choices are `elements` and `nd2ToTIFF`.
* `channel_width`, `channel_separation`, and `channel_detection_snr`, which are used to help find the channels.
* `channel_length_pad` and `channel_width_pad` will increase the size of your channel slices.
* `t_end` : Will only analyze images up to this time point. Useful for debugging.

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

## Notes on metadata

mm3_nd2ToTIFF.py reads the metadata directly from the .nd2 and then writes it into the header of the TIFF file when saving. The format is a json representation of a Python dictionary, and is recognized later by Compile.  


<a name="pickchannels"></a> 
### 2. User guided selection of empty and full channels (PickChannels). 
<img width="1177" alt="channel_picker" src="https://user-images.githubusercontent.com/40699438/177629496-73b6c4cf-7427-41e6-ac20-720b6fbf2ba1.png">

The Compile widget identifies all growth channels, regardless of if they contain or do not contain cells. ChannelSorter first attempts to guess, and then presents the user with a GUI to decide which channels should be analyzed, which channels should be ignored, and which channels should be used as empty channels during subtraction. This information is contained within the specs.yaml file.

Clicking the "Load data" button displays the first FOV analyzed, along with the program's predicted channel classification. The user is asked to click on the channels to change their designation between analyze (green), empty (blue) and ignore (red).

Click on the colored channels until they are as you wish. To navigate between fields of view click the "next FOV" or "prior FOV" buttons.  The widget will output the specs file with channels indicated as analyzed (green, 1), empty for subtraction (blue, 0), or ignore (red, -1).

**Parameters**

* `phase_plane` is the postfix of the channel which contains the phase images
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

<a name="subtract"></a> 
### 3. Subtract phase contrast images (Subtract widget). 
<img width="1183" alt="subtract" src="https://user-images.githubusercontent.com/40699438/177629512-c5ba4abd-0e03-4540-a4bb-7414ad0560d0.png">

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

<a name="segmentotsu"></a> 
### 4. Segment images (SegmentOTSU or SegmentUnet). 

<img width="1486" alt="otsu" src="https://user-images.githubusercontent.com/40699438/177629756-2bf87d2e-6ec8-4580-8675-648d68b29cb5.png">
<img width="1180" alt="unet" src="https://user-images.githubusercontent.com/40699438/177629546-81c2f826-73e8-41ef-adbd-7ceb191db461.png">
mm3 can use either deep learning or a traditional machine vision approach (Otsu thresholding, morphological operations and watershedding) to locate cells from the subtracted images.

**OTSU parameters**

* `first_opening_size` : Size in pixels of first morphological opening during segmentation.
* `distance_threshold` : Distance in pixels which thresholds distance transform of binary cell image.
* `second_opening_size` : Size in pixels of second morphological opening.

**U-net parameters**

* `threshold` : threshold value (between 0 and 1) for cell classification
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
### 5. Create cell lineages (Track widget). 

<img width="1188" alt="lineage" src="https://user-images.githubusercontent.com/40699438/177629704-b866d74e-cd80-4171-a6cf-92a887617160.png">

After cells are found for each channel in each time point, these labeled cells are connected across time to create complete cells and lineages.

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
