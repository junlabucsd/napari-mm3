# ChannelPicker

This script takes the channel stacks as created by mm3_Compile.py and identifies which channels are full and empty. It does this first by calculating the cross correlation value across time of the images in the stack to the first image in the stack. Images which have high correlation are considered empty, lower values are considered full. It then presents a GUI to the user who can manually curate which channels should be analyzed, which ones should be ignored, and which ones should be used as empty channels for subtraction.

**Input**
* TIFF channel stacks (phase contrast only).

**Output**
* crosscorrs.pkl and .txt : Python dictionary that contains image correlation value for channels over time. Used to guess if a channel is full or empty. Same structure as channel_masks.
* specs.pkl and .txt : Python dictionary which is the specifications of channels as full (1), empty (0), or ignore (-1). Same structure as channel_masks.

## Notes on use

When the cross correlations are calculated or loaded, the GUI is then launched. The user is asked to click on the channels to change their designation between analyze (green), empty (blue) and ignore (red).

The GUI shows all the channel for one FOV in columns. The channels are labeled by the cross correlation value (X, between 0.8 and 1), across time (Y, 0-100 where 0 is the start of the experiment and 100 is the end).

Click on the colored channels until they are as you wish. The script will output the specs file with channels indicated as analyzed (green, 1), empty for subtraction (blue, 0), or ignore (red, -1).
