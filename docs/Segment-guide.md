# Segment

This widget segments the subtracted images.

**Input**
* TIFF channel stacks (subtracted images).
* specs.pkl file.

**Output**
* Segmented channels stacks. Saved in the `segmented/` subfolder in the analysis directory.

**Parameters File**

There are some key parameters which influence segmentation as well as building the lineages. The first four parameters are important for finding markers in order to do watershedding/diffusion for segmentation. They should be changed depending on cell size and magnification/imaging conditions.

* `first_opening_size` : Size in pixels of first morphological opening during segmentation.
* `distance_threshold` : Distance in pixels which thresholds distance transform of binary cell image.
* `second_opening_size` : Size in pixels of second morphological opening.
* `min_object_size` : Objects smaller than this area in pixels will be removed before labeling.

**Hardcoded parameters**

There are few hardcoded parameters at the start of the executable Python script (right after __main__).

* `do_segmentation` : Segment the subtracted channel stacks. If False attempt to load them.
* `do_lineages` : Create lineages from segmented images and save cell data.

## Notes on use

mm3_Segment.py consists of two parts, segmenting individual images, and then looking across time at those segments and linking them together to create growing cells.

Use the IPython notebook `mm3_Segment.ipynb` in the folder `notebooks` to decide which parameters to use during segmentation. You can start an IPython notebook session by typing `ipython notebook` in Terminal and navigating to the notebook using the browser.

Lineages are made by connecting the segmented regions into complete cells using basic rules. These rules include that a region in one time point must be a similar size and overlap with a region in a previous time point to which it will link. For a cell to divide, the two daughter cells' combined size must be similar and also overlap. For a cell to be considered complete, it must have both a mother and two daughters.
