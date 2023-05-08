## Generating a TIFF stack 

**Input**
* .nd2 file as produced by Nikon Elements

OR

* Any bioformats supported file.

**Output**
* Individual TIFF files. 

napari should be launched from the directory containing the raw images to be analyzed. This allows napari-mm3 to automatically locate the image files and saves the user from re-entering the data directory in subsequent analysis steps. By default, napari-mm3 will first look for raw images in the directory from which napari is launched.

napari-MM3 can take any Bio-Formats supported file as input, or can process individual TIFF images. If the data is in the .nd2 format, the widget will use the nd2reader package to extract metadata from the file. Processing other file formats requires the user to [install Bio-Formats](https://pypi.org/project/python-bioformats/).

napari-mm3 can also take individual TIFF images as its input. In this case, the skip TIFFConverter and move directly to the [Compile widget](https://github.com/junlabucsd/napari-mm3/blob/main/docs/compile-widget.md). If there are multiple color layers, then each TIFF image should be a stack of planes corresponding to a color. The quality of your images is important for mm3 to work properly. See [Input image guidelines](https://github.com/junlabucsd/napari-mm3/blob/main/docs/Input-images-guidelines.md) for more details.

The working directory now contains:
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
```


