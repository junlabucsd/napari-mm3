# User Manual

## Overview
Generally, the workflow can be split into a few parts:
0. **Preprocessing**

    0a. [nd2ToTIFF](/docs/nd2totiff-widget.md) -- Turn your nd2 microscopy data into TIFFs. If your data is not in the nd2 format, follow the [input image guidelines](/docs/input-images-guidelines.md). Make sure to set `image source' in Compile to `Other'.

    0b. [Compile](/docs/compile-widget.md) -- Locate traps, separate their timelapses into their own TIFFs, and return metadata.

1. **Segmentation**

    **With Otsu:**

    1a. [PickChannels](/docs/pickchannels-widget.md) -- User guided selection of empty and full traps.

    1b. [Subtract](/docs/subtract-widget.md) -- Remove (via subtraction) empty traps from the background of traps that contain cells; run this on the phase contrast channel.

    1c. [SegmentOtsu](/docs/segmentotsu-widget.md) -- Use Otsu segmentation to segment cells.

    **With UNet:**

    1a. Annotate -- annotate images for ML (U-Net or similar) training purposes; you can generate a model via TODO.

    1b. [SegmentUnet](/docs/segmentunet-widget.md) -- Run U-Net segmentation (you will need to supply your own model)

2. **Tracking**

    2a. [Track](/docs/track-widget.md) -- Acquire individual cell properties and track lineages.

3. **Fluorescence data analysis**

    3a. [PickChannels](/docs/pickchannels-widget.md) -- If you've already done this (e.g. for otsu segmentation), no need to do it again. User guided selection of empty and full traps. 

    3b. [Subtract](/docs/subtract-widget.md) -- Remove (via subtraction) empty traps from the background of traps that contain cells. This time, run this on your fluorescence channels.

    3c. [Colors](/docs/colors-widget.md) -- Calculate fluorescence information.

4. (Uncommon) **Foci tracking**

    4a. Foci -- We use this to track `foci' (bright fluorescent spots) inside of cells.


## Outputs, inputs, and file structure
Finally, to better understand the data formats, you may wish to refer to the following documents:

[Input image guidelines](/docs/input-images-guidelines.md)

[File structure](/docs/file-structure.md)

[Output file structure](/docs/Cell-class-docs.md)
