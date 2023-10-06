# utf-8

"""
Provide the function to display
frame of blobs interactively.
"""
import functools

import numpy as np
import matplotlib.pyplot as plt

from types import SimpleNamespace
from collections import namedtuple
from .visualizers import (
    FrameVisualizer,
    WHITE, BLACK, RED, GREEN, BLUE,
    POSE2COLOR, BACKGROUND_COLOR,
)

layerT = namedtuple("layerT", "root,type")

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
        layers_stack=[layerT(frame, 'frame')],
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
    state.visualizer = FrameVisualizer(state)   # start with frame of blobs
    state.visualizer.reset()    # first-time reset

    # declare callback sub-routines

    # def update_blob_slices():
    #     if state.blob_slices is not None:
    #         for line, L_text in state.blob_slices:
    #             line.remove()
    #             L_text.remove()
    #         state.blob_slices = None
    #
    #     blob = state.blob_cls.get_instance(state.blob_id)
    #     if blob is None or not blob.dlayers or not blob.dlayers[0] or not state.show_slices:
    #         return
    #     edge = blob.dlayers[0][0]
    #     if not edge.node_t: return
    #     print(f"PP_fd = {state.PP_fd}", end="")
    #     y0, x0, *_ = blob.ibox
    #
    #     state.blob_slices = []
    #     for P in get_P_(edge, state.PP_fd):        # show last layer
    #         y, x = P.yx
    #         y_, x_, *_ = np.array([*zip(*P.dert_)])
    #         L = len(x_)
    #         if L > 1:
    #             blob_slices_plot = ax.plot(x_+x0, y_+y0, 'bo-', linewidth=1, markersize=2)[0]
    #         else:
    #             s, c = P.axis
    #             x_ = np.array([x-c, x, x+c])
    #             y_ = np.array([y-s, y, y+s])
    #             blob_slices_plot = ax.plot(x_ + x0, y_ + y0, 'b-', linewidth=1, markersize=2)[0]
    #         state.blob_slices += [(
    #             blob_slices_plot,
    #             ax.text(x+x0, y+y0, str(L), color = 'b', fontsize = 12),
    #         )]
    #
    # def update_P_links():
    #     if state.P_links is not None:
    #         for line in state.P_links:
    #             line.remove()
    #         state.P_links = None
    #
    #     blob = state.blob_cls.get_instance(state.blob_id)
    #     if blob is None or not blob.dlayers or not blob.dlayers[0] or not state.show_links:
    #         return
    #     edge = blob.dlayers[0][0]
    #     if not edge.node_t: return
    #     y0, x0, *_ = blob.ibox
    #     state.P_links = []
    #     for P in get_P_(edge, state.PP_fd):
    #         for derP in P.link_H[0]:
    #             (_y, _x), (y, x) = (derP._P.yx, derP.P.yx)
    #             state.P_links += ax.plot([_x+x0,x+x0], [_y+y0,y+y0], 'ko-', linewidth=2, markersize=4)

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
            state.visualizer.go_deeper(fd=False)
        elif event.key == "shift":
            state.visualizer.go_deeper(fd=True)
        else:
            # go back 1 layer
            # if (len(state.layers_stack) > 1):
            #     state.layers_stack.pop()
            #     reset_state()
            pass

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

def get_P_(edge, fd):
    P_ = []
    PP_ = [edge]
    while PP_:
        PP = PP_.pop()
        if not PP.node_t: continue
        if isinstance(PP.node_t[0], list):
            PP_ += PP.node_t[fd]
        else:  # is P_
            P_ += PP.node_t

    return P_