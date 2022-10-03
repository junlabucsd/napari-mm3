## Foci picking.

Finds foci using a Laplacian convolution. See https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html for details.
Foci are linked to the cell objects in which they appear.

**Input**
*  .pkl file of cell objects from Track widget.

**Output**
*  .pkl file with a dictionary of cell objects, with x, y and time positions of foci detections stored as attributes of the corresponding cell objects.

**Parameters**

* `LoG_min_sig` : Minimum sigma of laplacian to convolve in pixels. Scales with minimum foci width to detect as 2*sqrt(2)*minsig
* `LoG_max_sig`: Maximum sigma of laplacian to convolve in pixels. Scales with maximum foci width to detect as 2*sqrt(2)*maxsig
* `LoG_threshold` : Absolute threshold laplacian must reach to record potential foci. Keep low to detect dimmer spots.
* `LoG_peak_ratio` : Focus peaks must be this many times greater than median cell intensity. Think signal to noise ratio
