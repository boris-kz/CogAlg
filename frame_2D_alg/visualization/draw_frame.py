# utf-8

"""
Provide the function to display
frame of blobs interactively.
"""
import functools

import numpy as np
import matplotlib.pyplot as plt

from types import SimpleNamespace
from .visualizers import BlobVisualizer

ROOT_TYPES = ["frame", "rblob", "edge", "PP"]

def visualize(frame):
    """
    Visualize frame after clustering.
    Transition of visualization layers:
    frame ─→ blobs ─→ (r)blobs ─→ rblobs ...
               │          └─────→  edge  ...
               └────→   edge   ─→   rPP  ...
                          └─────→   dPP  ...
    """
    print("Preparing for visualization ...", end="")

    # get frame size
    _, _, height, width = frame.box

    # create visualizer
    state = SimpleNamespace(
        visualizer=BlobVisualizer(frame, title="Visualization")
    )
    fig, ax = visualizer.fig, visualizer.ax
    state.visualizer.reset()

    # TODO : cleaning up in transitions i.e:

    # declare callback sub-routines

    def on_mouse_movement(event):
        """Highlight the blob the mouse is hovering on."""
        hovered_element = state.visualizer.get_hovered_element(event.xdata, event.ydata)
        if hovered_element is not state.visualizer.hovered_element:
            state.visualizer.update_hovered_element(hovered_element)
            state.visualizer.update_img()
            state.visualizer.update_info()

    def on_click(event):
        """Transition between layers."""
        if event.key == "control": ret = state.visualizer.go_deeper(fd=False)
        elif event.key == "shift": ret = state.visualizer.go_deeper(fd=True)
        else: ret = state.visualizer.go_back()  # go back 1 layer
        if ret is None: return
        state.visualizer = ret
        state.visualizer.reset()

    def on_key_press(event):
        # TODO : check if visualizer has flag
        if event.key == 'd': state.visualizer.show_gradient = not state.visualizer.show_gradient
        elif event.key == 'z': state.visualizer.show_slices = not state.visualizer.show_slices
        elif event.key == 'x': state.visualizer.show_links = not state.visualizer.show_links
        else: return
        state.visualizer.update_img()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_movement)
    fig.canvas.mpl_connect('button_release_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show()