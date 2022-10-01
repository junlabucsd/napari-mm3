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

Load up a new environment. We run the following command, replacing `environment-name-here` with something of your choosing:

`conda create -y -n environment-name-here python=3.9 napari tensorflow` 

If you would like to have the latest version: 

1. You can clone the repository with `git clone git@github.com:junlabucsd/napari-mm3.git` (SSH) or `git clone https://github.com/junlabucsd/napari-mm3.git` (https)
2. With your environment active, run `pip install -e .` from inside your cloned repo.

If you would like to have a more stable verison, simply run `pip install napari-mm3`.

NOTE:
There are occasionally issues with PyQt5 -- you may need to resolve those yourself; those are an upstream bug that can be very sensitive to your choice of system.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## Workflow


**Detailed usage guide:** [User manual](/docs/user-manual.md)

Generally, each step of the pipeline has a single widget.
This assumes you are using Otsu segmentation -- the procedure can be modified if you are using U-Net.
0. [nd2ToTIFF](/docs/user-manual.md#nd2ToTIFF) -- Turn your microscopy data into TIFFs. 
1. [Compile](/docs/user-manual.md#compile) -- Locate traps, separate them into their own TIFFs, and return metadata.
2. [PickChannels](/docs/user-manual.md#pickchannels) -- User guided selection of empty and full traps.
3. [Subtract](/docs/user-manual.md#subtract) -- Remove (via subtraction) empty traps from the background of traps that contain cells. 
4. [SegmentOtsu](/docs/user-manual.md#segment) -- Use Otsu segmentation to segment cells.
5. [Track](/docs/user-manual.md#track) -- Acquire individual cell properties and track lineages.

Additionally, we have a few widgets to assist in other tasks that may come up:

6. Annotate -- annotate images for ML (U-Net or similar) training purposes.
7. SegmentUnet -- Run U-Net segmentation (you will need to supply your own model)
8. Colors -- Calculate fluorescence information.
9. Foci -- We use this to track `foci' (bright fluorescent spots) inside of cells.

For additional information, you may wish to refer to the following documents:

[Input image guidelines](/docs/input-images-guidelines.md)

[File structure](/docs/file-structure.md)

[Output file structure](/docs/Cell-class-docs.md)

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
