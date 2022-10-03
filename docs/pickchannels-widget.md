## User guided selection of empty and full channels (PickChannels). 
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

