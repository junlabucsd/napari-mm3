"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui import magic_factory


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


#@magic_factory
#def example_magic_widget(img_layer: "napari.layers.Image"):
#    print(f"you have selected {img_layer}")


import datetime
from enum import Enum
from pathlib import Path

from magicgui import magicgui


class Medium(Enum):
    """Using Enums is a great way to make a dropdown menu."""
    Glass = 1.520
    Oil = 1.515
    Water = 1.333
    Air = 1.0003


@magicgui(
    call_button="Calculate",
    layout="vertical",
    # numbers default to spinbox widgets, but we can make
    # them sliders using the `widget_type` option

    #slider_int={"widget_type": "Slider", "readout": False},
    #radio_option={
    #    "widget_type": "RadioButtons",
    #    "orientation": "horizontal",
    #    "choices": [("first option", 1), ("second option", 2)],
    #},
    filename={"label": "Pick a file:"},  # custom label
)
def widget_demo(
    OTSU_Threshold=1.0,
    first_opening_size=2,
    distance_Threshold=2,
    second_opening_size=1,
    min_object_size=25,
    filename=Path.home(),  # path objects are provided a file picker
):
    """Run some computation."""
    return locals().values()

widget_demo.show(run = True) # if running locally, use `show(run=True)`

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    #return [ExampleQWidget, example_magic_widget]
    return []

