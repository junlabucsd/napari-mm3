## Generating a TIFF stack 

**Input**
* .nd2 file as produced by Nikon Elements
OR

* Any bioformats supported file.

**Output**
* Individual TIFF files. 

By default, napari-mm3 will first look for raw images in the directory from which napari is launched.

napari- mm3 currently currently takes individual TIFF images as its input. If there are multiple color layers, then each TIFF image should be a stack of planes corresponding to a color. The quality of your images is important for mm3 to work properly.

The working directory now contains:
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
```

