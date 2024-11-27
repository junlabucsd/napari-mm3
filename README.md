# napari-mm3

[![License](https://img.shields.io/pypi/l/napari-mm3.svg?color=green)](https://github.com/junlabucsd/napari-mm3/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-mm3.svg?color=green)](https://pypi.org/project/napari-mm3)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-mm3.svg?color=green)](https://python.org)
[![tests](https://github.com/junlabucsd/napari-mm3/workflows/tests/badge.svg)](https://github.com/junlabucsd/napari-mm3/actions)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-mm3)](https://napari-hub.org/plugins/napari-mm3)

A plugin for mother machine image analysis by the [Jun Lab](https://jun.ucsd.edu/).

Reference:
[Tools and methods for high-throughput single-cell imaging with the mother machine. Ryan Thiermann, Michael Sandler, Gursharan Ahir, John T. Sauls, Jeremy W. Schroeder, Steven D. Brown, Guillaume Le Treut, Fangwei Si, Dongyang Li, Jue D. Wang, Suckjoon Jun. eLife12:RP88463
https://doi.org/10.7554/eLife.88463.4](https://elifesciences.org/articles/88463)

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.


https://github.com/junlabucsd/napari-mm3/assets/40699438/1b3e6121-f5e1-475f-aca3-c6ed1b5bab3a



## Installation

We describe installation with conda. First, clone with git and navigate inside the folder.
```
git clone git@github.com:junlabucsd/napari-mm3.git
cd napari-mm3
```
If you do not have ssh configured, you can replace the URL with `https://github.com/junlabucsd/napari-mm3.git`; we recommend setting up SSH.
Now, install dependencies (this step can take a while):
```
conda env create -f environment.yml
```
By default, 'napari-mm3' will be the environment name.
Finally, switch to the environment you've created, and install the plugin itself WITHOUT dependencies (if you miss the flag, you will likely run into trouble!!):
```
conda activate napari-mm3
pip install -e . --no-dependencies
```
This supplies you with the latest, most recent version of our code.

napari-MM3 can use the [python-bioformats](https://pypi.org/project/python-bioformats/) library to import various image file formats. It can be installed with pip:
```
pip install python-bioformats
```
If your raw images are in the .nd2 format, they will be read in with the nd2 package. In this case, Bio-Formats is not required.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## Usage guide


https://github.com/junlabucsd/napari-mm3/assets/8302475/68c726be-620e-4375-b1c9-3db56ac9a82a

Additional reference information is available below.
### a. Preprocessing

* [TIFFConverter](https://github.com/junlabucsd/napari-mm3/blob/main/docs/tiffconvert-widget.md) -- Turn your nd2 microscopy data, or other format via bioformats, into TIFFs. If your data is already exported as individual TIFF files, skip to the [Compile](https://github.com/junlabucsd/napari-mm3/blob/main/docs/compile-widget.md) widget. Take note of the [input image guidelines](https://github.com/junlabucsd/napari-mm3/blob/main/docs/Input-images-guidelines.md).

* [Compile](https://github.com/junlabucsd/napari-mm3/blob/main/docs/compile-widget.md) -- Locate traps, separate their timelapses into their own TIFFs, and return metadata.

* [PickChannels](https://github.com/junlabucsd/napari-mm3/blob/main/docs/pickchannels-widget.md) -- User guided selection of empty and full traps.

### b. Segmentation

___With Otsu's method:___

* [Subtract](https://github.com/junlabucsd/napari-mm3/blob/main/docs/subtract-widget.md) -- Remove (via subtraction) empty traps from the background of traps that contain cells; run this on the phase contrast channel.

* [SegmentOtsu](https://github.com/junlabucsd/napari-mm3/blob/main/docs/segmentotsu-widget.md) -- Use Otsu's method to segment cells.

___With U-Net:___

* [Annotate](https://github.com/junlabucsd/napari-mm3/blob/main/docs/annotate-widget.md) -- annotate images for ML (U-Net or similar) training purposes.

* [Train U-Net](https://github.com/junlabucsd/napari-mm3/blob/main/docs/trainunet-widget.md) -- Train a U-Net model for cell segmentation.

* [SegmentUnet](https://github.com/junlabucsd/napari-mm3/blob/main/docs/segmentunet-widget.md) -- Run U-Net segmentation.

### c. Tracking

* [Track](https://github.com/junlabucsd/napari-mm3/blob/main/docs/track-widget.md) -- Acquire individual cell properties and track lineages.

### d. Fluorescence data analysis

* [Subtract](https://github.com/junlabucsd/napari-mm3/blob/main/docs/subtract-widget.md) -- Remove (via subtraction) empty traps from the background of traps that contain cells. This time, run this on your fluorescence channels.

* [Colors](https://github.com/junlabucsd/napari-mm3/blob/main/docs/colors-widget.md) -- Calculate fluorescence information.

### e. Focus tracking

* [Foci](https://github.com/junlabucsd/napari-mm3/blob/main/docs/foci-widget.md) -- We use this to track `foci` (bright fluorescent spots) inside of cells.

### f. Extracting data and plotting

* The notebook [here](https://github.com/junlabucsd/napari-mm3/blob/main/notebooks/napari_mm3_analysis_template.ipynb) demonstrates how to extract, filter and visualize the lineage data output by the [Track](https://github.com/junlabucsd/napari-mm3/blob/main/docs/track-widget.md) widget.


### g. Outputs, inputs, and file structure
Finally, to better understand the data formats, you may wish to refer to the following documents:

* [Input image guidelines](https://github.com/junlabucsd/napari-mm3/blob/main/docs/Input-images-guidelines.md)

* [File structure](https://github.com/junlabucsd/napari-mm3/blob/main/docs/file-structure.md)

* [Output data structure](https://github.com/junlabucsd/napari-mm3/blob/main/docs/Cell-class-docs.md)

## License

Distributed under the terms of the [BSD-3] license,
"napari-mm3" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/junlabucsd/napari-mm3/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
