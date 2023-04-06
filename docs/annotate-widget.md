## Annotate training data for U-Net segmentation

This widget allows manual annotation and curation of segmentation masks for training the U-Net model. By default, it looks for image stacks in the format output by the Compile widget in the "channels" subfolder within the user's chosen analysis directory. The masks can be drawn as labels using napari's paintbrush tool.

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