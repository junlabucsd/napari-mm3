# napari-mm3

[![License](https://img.shields.io/pypi/l/napari-mm3.svg?color=green)](https://github.com/ahirsharan/napari-mm3/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-mm3.svg?color=green)](https://pypi.org/project/napari-mm3)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-mm3.svg?color=green)](https://python.org)
[![tests](https://github.com/ahirsharan/napari-mm3/workflows/tests/badge.svg)](https://github.com/ahirsharan/napari-mm3/actions)
[![codecov](https://codecov.io/gh/ahirsharan/napari-mm3/branch/main/graph/badge.svg)](https://codecov.io/gh/ahirsharan/napari-mm3)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-mm3)](https://napari-hub.org/plugins/napari-mm3)

A plugin for Mother Machine Image Analysis by [Jun Lab](https://jun.ucsd.edu/).

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

## Installation

Load up a new environment. We run the following command, replacing `environment-name-here` with a name of your choosing:

`conda create -y -n environment-name-here python=3.9 napari tensorflow` 

Now, to install our code: if you would like to have the latest version, do the following.

1. You can clone the repository with `git clone git@github.com:junlabucsd/napari-mm3.git` (SSH) or `git clone https://github.com/junlabucsd/napari-mm3.git` (https)
2. With your environment active, run `pip install -e .` from inside your cloned repo.

If you would like to have a more stable verison, simply run `pip install napari-mm3`.

NOTE:
Not running the conda command and trying to install things in a different way may lead to difficult issues with PyQt5. 
We recommend following the above commands to simplify the situation.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## Usage guide

### a. Preprocessing

* [nd2ToTIFF](/docs/nd2totiff-widget.md) -- Turn your nd2 microscopy data into TIFFs. If your data is not in the nd2 format, follow the [input image guidelines](/docs/input-images-guidelines.md). Make sure to set `image source' in Compile to `Other'.

* [Compile](/docs/compile-widget.md) -- Locate traps, separate their timelapses into their own TIFFs, and return metadata.

### b. Segmentation

___With Otsu:___

* [PickChannels](/docs/pickchannels-widget.md) -- User guided selection of empty and full traps.

* [Subtract](/docs/subtract-widget.md) -- Remove (via subtraction) empty traps from the background of traps that contain cells; run this on the phase contrast channel.

* [SegmentOtsu](/docs/segmentotsu-widget.md) -- Use Otsu segmentation to segment cells.

___With UNet:___

* Annotate -- annotate images for ML (U-Net or similar) training purposes; you can generate a model via TODO.

* [SegmentUnet](/docs/segmentunet-widget.md) -- Run U-Net segmentation (you will need to supply your own model)

### c. Tracking

* [Track](/docs/track-widget.md) -- Acquire individual cell properties and track lineages.

### d. Fluorescence data analysis

* [PickChannels](/docs/pickchannels-widget.md) -- If you've already done this (e.g. for otsu segmentation), no need to do it again. User guided selection of empty and full traps. 

* [Subtract](/docs/subtract-widget.md) -- Remove (via subtraction) empty traps from the background of traps that contain cells. This time, run this on your fluorescence channels.

* [Colors](/docs/colors-widget.md) -- Calculate fluorescence information.

### e. (Uncommon) Foci tracking

* Foci -- We use this to track `foci' (bright fluorescent spots) inside of cells.


### f. Outputs, inputs, and file structure
Finally, to better understand the data formats, you may wish to refer to the following documents:

* [Input image guidelines](/docs/input-images-guidelines.md)

* [File structure](/docs/file-structure.md)

* [Output file structure](/docs/Cell-class-docs.md)

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

[file an issue]: https://github.com/ahirsharan/napari-mm3/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
