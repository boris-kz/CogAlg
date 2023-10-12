# utf-8

"""
Provide the function to display
frame of blobs interactively.
"""
import functools

import numpy as np
import matplotlib.pyplot as plt

from types import SimpleNamespace
from .visualizers import (
    BlobVisualizer, layerT,
    WHITE, BLACK, RED, GREEN, BLUE,
    POSE2COLOR, BACKGROUND_COLOR,
)

ROOT_TYPES = ["frame", "rblob", "edge", "PP"]

def visualize(frame):
    """
    Visualize frame after clustering.
    Parameters
    ----------
    frame : CBlob
        The frame of layer to visualize.
    """
    print("Preparing for visualization ...", end="")

    # TODO: re-write visualization code. requirements:
    # - Separate display functions for blobs, edges, PPs, Ps
    # - Simple transition : frame ─→ blobs ─→ (r)blobs ─→ rblobs ...
    #                                  │          └─────→  edge  ...
    #                                  └────→   edge   ─→   PP
    # - Hover high-lights:
    #   + Blobs : by masks
    #   + edge  : by blob's mask
    #   + PP    : by combined P's cells
    # - Toggles:
    #   + Blobs   : show gradients
    #   + edge/PP : show slices/links

    # get frame size
    _, _, height, width = frame.box

    # create display window
    img = blank_image((height, width))
    fig, ax = plt.subplots()
    fig.canvas.set_window_title("Blob Visualization")
    imshow_obj = ax.imshow(img)

    # Prepare state object
    state = SimpleNamespace(
        visualizer=None,
        # history of roots
        layers_stack=None,
        # variables
        img=img,
        img_slice = None,
        background=blank_image((height, width)),
        idmap=np.full((height, width), -1, 'int64'),
        gradient=np.zeros((2, height, width), float),
        gradient_mask=np.zeros((height, width), bool),
        element_id=None, element_cls=None,
        # plot object handles
        fig=fig,
        ax=ax,
        imshow_obj=imshow_obj,
        quiver=None,
        blob_slices=None,
        P_links=None,
        # flags
        show_gradient=False,
        show_slices=False,
        show_links=False,
    )
    state.layers_stack = [layerT(frame, 'frame', BlobVisualizer(state))]    # start with frame of blobs
    state.visualizer = state.layers_stack[0].visualizer
    state.visualizer.reset()    # first-time reset

    # declare callback sub-routines

    def on_mouse_movement(event):
        """Highlight the blob the mouse is hovering on."""
        if event.xdata is None or event.ydata is None:
            element_id = -1
        else:
            element_id = state.idmap[round(event.ydata), round(event.xdata)]
        if state.element_id != element_id:
            state.element_id = element_id
            state.visualizer.update_element_id()

    def on_click(event):
        """Transition between layers."""
        if event.key == "control":
            ret = state.visualizer.go_deeper(fd=False)
        elif event.key == "shift":
            ret = state.visualizer.go_deeper(fd=True)
        else:
            # go back 1 layer
            ret = state.visualizer.go_back()
        if ret is None:
            return
        state.visualizer = ret
        state.visualizer.reset()

    def on_key_press(event):
        if event.key == 'd':
            state.show_gradient = not state.show_gradient
        elif event.key == 'z':
            state.show_slices = not state.show_slices
        elif event.key == 'x':
            state.show_links = not state.show_links
        else:
            return
        state.visualizer.update_img()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_movement)
    fig.canvas.mpl_connect('button_release_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show()


def blank_image(shape, fill_val=None):
    '''Create an empty numpy array of desired shape.'''

    if len(shape) == 2:
        height, width = shape
    else:
        y0, x0, yn, xn = shape
        height = yn - y0
        width = xn - x0
    if fill_val is None:
        fill_val = BACKGROUND_COLOR
    return np.full((height, width, 3), fill_val, 'uint8')