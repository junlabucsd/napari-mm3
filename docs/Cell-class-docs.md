# Description of cell data

When cells are made during lineage creation, the information is stored per cell in an object called Cell. The Cell object is the fundamental unit of data produced by mm3. Every Cell object has a unique identifier (`id`) as well as all the other pertinent information. The default data format saved by the track widget is a dictionary of these cell objecs, where the keys are each cell id and the values are the object itself. Below is a description of the information contained in a Cell object and how to export it to other formats. For an overview on classes in Python see [here](https://docs.python.org/3/tutorial/classes.html).

## Cell object attributes

The following is a list of the attributes of a Cell.

#### Standard attributes
* `Cell.id` : The cell id is a string in the form `f0p0t0r0` which represents the FOV, channel peak number, time point the cell came from as well as which segmented region it is in that image.
* `Cell.fov` : FOV the cell came from.
* `Cell.peak` : Channel peak number the cell came from.
* `Cell.birth_label` : The segmented region number the cell was born at. The regions are numbered from the closed end of the channel, so mother cells should have a birth label of 1.
* `Cell.parent` : This cell's mother cell's id.
* `Cell.daughters` : A list of the ids of this cell's two daughter cells.
* `Cell.birth_time` : Nominal time point at time of birth.
* `Cell.division_time` : Nominal division time of cell. Note that this is equal to the birth time of the daughters.
* `Cell.times` : A list of time points for which this cell grew. Includes the first time point but does not include the last time point. It is the same length as the following attributes, but it may not be sequential because of dropped segmentations.
* `Cell.labels` : The segmented region labels over time.
* `Cell.bboxes` : The bounding boxes of each region in the segmented channel image over time.
* `Cell.areas` : The areas of the segmented regions over time in pixels^2.
* `Cell.orientations`: The angle between the cell's major axis and the positive x axis (within [pi/2, -pi/2]) over time.
* `Cell.centroids`: The y and x positions (in that order) in pixels of the centroid of the cell over time.
* `Cell.lengths` : The long axis length in pixels of the regions over time.
* `Cell.widths` : The long axis width in pixels of the regions over time.
* `Cell.times_w_div` : Same as Cell.times but includes the division time.
* `Cell.lengths_w_div` : The long axis length in microns of the regions over time, including the division length.
* `Cell.sb` : Birth length of cell in microns.
* `Cell.sd` : Division length of cell in microns. The division length is the combined birth length of the daugthers.
* `Cell.delta` : Cell.sd - Cell.sb. Simply for convenience.
* `Cell.tau` : Nominal generation time of the cell.
* `Cell.elong_rate` : Elongation rate of the cell using a linear fit of the log lengths.
* `Cell.septum_position` : The birth length of the first daughter (closer to closed end) divided by the division length. 

#### Fluorescence attributes:
* `Cell.fl_tots`: Total integrated fluorescence per time point. The plane which was analyzed is appended to the attribute name, so that e.g. Cell.fl_tots_c1 represents the integrated fluorescence from the cell in plane c1.
* `Cell.fl_area_avgs`: Mean fluorescence per pixel by timepoint. The plane which was analyzed is appended to the attribute name, e.g. as Cell.fl_area_avgs_c1.
* `Cell.fl_vol_avgs`: Mean fluorescence per cell volume. The plane which was analyzed is appended to the attribute name, e.g. as Cell.fl_vol_avgs_c1.

#### Foci tracking attributes:
* `Cell.disp_l`: Displacement on long axis in pixels of each focus from the center of the cell, by time point. (1D np.array)
* `Cell.disp_w`: Displacement on short axis in pixels of each focus from the center of the cell, by time point. (1D np.array)
* `Cell.foci_h`: Focus "height." Sum of the intensity of the gaussian fitting area, by time point. (1D np.array) 
