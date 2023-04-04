## Segment images. 

<img width="1486" alt="otsu" src="https://user-images.githubusercontent.com/40699438/177629756-2bf87d2e-6ec8-4580-8675-648d68b29cb5.png">

MM3 can use either deep learning or a traditional machine vision approach (Otsu thresholding, morphological operations and watershedding) to locate cells from the subtracted images. For info on the deep learning-based segmentation widget, see [SegmentUNet](https://github.com/junlabucsd/napari-mm3/blob/main/docs/segmentunet-widget.md).

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
