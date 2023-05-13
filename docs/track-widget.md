## 5. Create cell lineages (Track widget). 

<img width="1188" alt="lineage" src="https://user-images.githubusercontent.com/40699438/177629704-b866d74e-cd80-4171-a6cf-92a887617160.png">

This widget reconstructs cell lineages from the segmented images.

After cells are found for each channel in each time point, these labeled cells are connected across time to create complete cells and lineages. The lineage information is saved to a .json file. The notebook [here](https://github.com/junlabucsd/napari-mm3/blob/main/notebooks/napari_mm3_analysis_template.ipynb) demonstrates how to extract and visualize the data.

**Input**
* segmented channel TIFFs.
* specs.pkl file.

**Parameters**

* `lost_cell_time` : If this many time points pass and a region has not yet been linked to a future region, it is dropped.
* `max_growth_length` : If a region is to be connected to a previous region, it cannot be larger in length by more than this ratio.
* `min_growth_length` : If a region is to be connected to a previous region, it cannot be smaller in length by less than this ratio.
* `max_growth_area` : If a region is to be connected to a previous region, it cannot be larger in area by more than this ratio.
* `min_growth_area` : If a region is to be connected to a previous region, it cannot be smaller in area by less than this ratio.
* `new_cell_y_cutoff` : distance in pixels from closed end of image above which new regions are not considered for starting new cells.
* `segmentation_method`: whether to construct lineage from cells segmented by the Otsu or U-net method

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
│   ├── cell_data
│   │   └── complete_cells.pkl
│   ├── channel_masks.pkl
│   ├── channel_masks.txt
│   ├── channels
│   ├── crosscorrs.pkl
│   ├── crosscorrs.txt
│   ├── empties
│   ├── segmented
│   ├── specs.pkl
│   ├── specs.txt
│   └── subtracted
└── params.yaml
```

**Output**
* Data for all cells. Saved as a dictionary of Cell objects (see below) in the `cell_data` subfolder in the analysis directory. The files `all_cells.pkl` and `all_cells.json` contain all identified cells, while `complete_cells.pkl` and `complete_cells.json` contain only cells with a mother and daughter identified.
* (Optional) Lineage images. Lineage and segmentation information overlayed on the subtracted images across time frames.

### Notes on use

Lineages are made by connecting the segmented regions into complete cells using basic rules. These rules include that a region in one time point must be a similar size and overlap with a region in a previous time point to which it will link. For a cell to divide, the two daughter cells' combined size must be similar and also overlap. For a cell to be considered complete, it must have both a mother and two daughters.

When cells are made during lineage creation, the information is stored per cell in an object called Cell. The Cell object is the fundamental unit of data produced by mm3. Every Cell object has a unique identifier (`id`) as well as all the other pertinent information. The default data format save by the track widget is a dictionary of these cell objecs, where the keys are each cell id and the values are the object itself. Below is a description of the information contained in a Cell object and how to export it to other formats. For an overview on classes in Python see [here](https://learnpythonthehardway.org/book/ex40.html).

For more information on the Cell object description in [Cell-class-docs.md](/docs/Cell-class-docs.md)

