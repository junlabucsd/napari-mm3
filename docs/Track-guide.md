# Track

This widget reconstructs cell lineages from the segmented images.

**Input**
* segmented channel stacks.
* specs.pkl file.

**Output**
* Data for all cells. Saved as a dictionary of Cell objects (see below) in the `cell_data` subfolder in the analysis directory. The file `all_cells.pkl` contains all identified cells, while `complete_cells.pkl` contains only cells with a mother and daughter identified.
* (Optional) Lineage images. Lineage and segmentation information overlayed on the subtracted images across time frames.

**Parameters**

The following parameters are concerned with rules for linking segmented regions to create the lineages. They are not necessarily changed from experiment to experiment.

* `lost_cell_time` : If this many time points pass and a region has not yet been linked to a future region, it is dropped.
* `max_growth_length` : If a region is to be connected to a previous region, it cannot be larger in length by more than this ratio.
* `min_growth_length` : If a region is to be connected to a previous region, it cannot be smaller in length by less than this ratio.
* `max_growth_area` : If a region is to be connected to a previous region, it cannot be larger in area by more than this ratio.
* `min_growth_area` : If a region is to be connected to a previous region, it cannot be smaller in area by less than this ratio.
* `new_cell_y_cutoff` : distance in pixels from closed end of image above which new regions are not considered for starting new cells.


## Notes on use

Lineages are made by connecting the segmented regions into complete cells using basic rules. These rules include that a region in one time point must be a similar size and overlap with a region in a previous time point to which it will link. For a cell to divide, the two daughter cells' combined size must be similar and also overlap. For a cell to be considered complete, it must have both a mother and two daughters.

When cells are made during lineage creation, the information is stored per cell in an object called Cell. The Cell object is the fundamental unit of data produced by mm3. Every Cell object has a unique identifier (`id`) as well as all the other pertinent information. The default data format save by the track widget is a dictionary of these cell objecs, where the keys are each cell id and the values are the object itself. Below is a description of the information contained in a Cell object and how to export it to other formats. For an overview on classes in Python see [here](https://learnpythonthehardway.org/book/ex40.html).

For more information on the Cell object see Cell_data_description.md.