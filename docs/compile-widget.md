## Locate channels, create channel stacks, and return metadata (Compile widget).
<img width="1187" alt="fov_inspect1" src="https://user-images.githubusercontent.com/40699438/177629474-5fd7ee80-682e-4aaa-bf6e-dd547e40c458.png">

The Compile widget attempts to automatically identify and crop out individual growth channels. Images corresponding to a specific channel are then stacked in time, and these "channel stacks" are the basis of further analysis. If there are multiple colors, a channel stack is made for each color for each channel.

It is also at this time that metadata is drawn from the images and saved. 

**Parameters**

* `TIFF_source` needs to be specified to indicate how the script should look for TIFF metadata. Choices are `nd2`, `BioFormats`. `elements`. `nd2` indicates that TIFF files were exported by nd2reader within the [TIFFconverter](https://github.com/junlabucsd/napari-mm3/blob/main/docs/tiffconvert-widget.md) script. `BioFormats` indicates that the TIFFs were extracted via Bio-Formats in the TIFFconverter widget and subsequent scripts should read in information from the TIFF names. Lastly, `elements` indicates that the TIFFs came from Nikon Elements.
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

