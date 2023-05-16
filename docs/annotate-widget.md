## Annotate training data for U-Net segmentation

<img width="1184" alt="annotate" src="https://user-images.githubusercontent.com/40699438/230508777-522d89ad-7361-47a9-8946-0228e2681ef6.png">

This widget allows manual annotation and curation of segmentation masks for training the U-Net model. By default, it looks for image stacks in the format output by the Compile widget in the "channels" subfolder within the user's chosen analysis directory. The masks can be drawn as labels using napari's paintbrush tool.

NOTE: the current U-Net implementation takes in binary masks. Consequently, the input masks must be segmented to leave at least a 1-pixel gap between adjacent cells - otherwise the model will not learn to separate contiguous cells.

**Parameters**

* `Data source` : Whether to load raw image data from channel stacks output by Compile, or from previously edited individual (unstacked) images.
* `Phase plane` : Imaging channel with phase contrast images for segmentation.
* `Mask source` : Option to seed segmentation masks for training with output of Otsu or U-Net segmentation.

The resulting file structure is:

```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
├── analysis
│   ├── training
|   |   |── images
|   |   └── masks
```
