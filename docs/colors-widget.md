## Fluorescence analysis.

The cell data output by the tracking widget contains information about all cells in the experiment, including which images and locations in the images they came from. We can use this to go back to additional image planes (colors) for secondary analysis. This widget computes integrated fluorescence signal per cell and similar attributes for a given input plane. See [Cell-class-docs.md](https://github.com/junlabucsd/napari-mm3/blob/main/docs/Cell-class-docs.md) for more information.

**Input**
*  .pkl file of cell objects from Track widget.

**Output**
*  .pkl file with a dictionary of cell objects, with integrated fluorescence intensity and fluorescence per pixel and per cell volume stored as attributes of the corresponding cell objects.

