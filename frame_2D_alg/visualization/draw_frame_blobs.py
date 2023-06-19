# utf-8

"""
Provide the function visualize_blobs to display
formed blobs interactively.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from types import SimpleNamespace

MIN_WINDOW_WIDTH = 640
MIN_WINDOW_HEIGHT = 480
WHITE = 255, 255, 255
BLACK = 0, 0, 0
RED = 0, 0, 255
GREEN = 0, 255, 0
BLUE = 255, 0, 0
POSE2COLOR = {
    0:RED,
    1:GREEN,
    2:BLUE,
}

MASKING_VAL = 128  # Pixel at this value can be over-written


def visualize_blobs(frame, layer='r'):
    """
    Visualize blobs after clustering.
    Highlight the blob the mouse is hovering on and its
    adjacents.
    Parameters
    ----------
    frame : CBlob
        The frame of layer to visualize.
    layer : str, optional
        The layer to visualize. Must be 'r' or 'd'. Defaults to 'r'.
    """
    print("Preparing for visualization ...", end="")

    height, width = frame.der__t[0].shape

    # Prepare state object
    state = SimpleNamespace(
        img=blank_image((height, width)),
        background=blank_image((height, width)),
        idmap=np.full((height, width), -1, 'int64'),
        gradient=np.zeros((2, height, width), float),
        gradient_mask=np.zeros((height, width), bool),
        blob_id=None, img_slice=None, blob_cls=None,
        layers_stack=[(frame, layer)],
        # flags
        show_gradient=False, quiver=None,
    )

    fig, ax = plt.subplots()
    fig.canvas.set_window_title("Blob Visualization")
    imshow_obj = ax.imshow(state.img)

    def update_gradient():
        if state.quiver is not None:
            state.quiver.remove()
            state.quiver = None

        blob = state.blob_cls.get_instance(state.blob_id)
        if blob is None or not state.show_gradient:
            return
        # Reset gradient
        state.gradient[:] = 1e-3
        state.gradient_mask[:] = False

        # Use indexing to get the gradient of the blob
        dy__, dx__ = state.gradient
        y0, yn, x0, xn = blob.box
        box_slice = slice(y0, yn), slice(x0, xn)
        dy_slice = dy__[state.img_slice][box_slice][~blob.mask__]
        dx_slice = dx__[state.img_slice][box_slice][~blob.mask__]
        dy_index = 4 if len(blob.der__t) > 5 else 1
        dy_slice[:] = blob.der__t[dy_index][~blob.mask__]
        dx_slice[:] = blob.der__t[dy_index + 1][~blob.mask__]
        state.gradient_mask[state.img_slice][box_slice] = ~blob.mask__
        iy, ix = state.gradient_mask.nonzero()

        # Apply quiver
        state.quiver = ax.quiver(ix, iy, dx_slice, -dy_slice)

    def update_img():
        update_gradient()
        imshow_obj.set_data(state.img)
        fig.canvas.draw_idle()

    def reset_state():
        frame, layer = state.layers_stack[-1]

        if layer == 'r':
            blob_ = frame.rlayers[0]
        elif layer == 'd':
            blob_ = frame.dlayers[0]
        else:
            raise ValueError("layer must be 'r' or 'd'")

        state.blob_cls = blob_[0].__class__
        y0, yn, x0, xn = frame.box
        if frame.root_der__t:
            rY, rX = frame.root_der__t[0].shape
            y0e = max(0, y0 - 1)
            yne = min(rY, yn + 1)
            x0e = max(0, x0 - 1)
            xne = min(rX, xn + 1)  # e is for extended
            state.img_slice = slice(y0e, yne), slice(x0e, xne)
        else:
            state.img_slice = slice(None), slice(None)
        state.background[:] = MASKING_VAL
        state.idmap[:] = -1
        # Prepare blob ID map and background
        for blob in blob_:
            paint_over(state.idmap[state.img_slice], None, blob.box,
                       mask=blob.mask__,
                       fill_color=blob.id)
            paint_over(state.background[state.img_slice], None, blob.box,
                       mask=blob.mask__,
                       fill_color=[blob.sign * 32] * 3)
        state.img = state.background.copy()
        state.blob_id = -1
        update_img()

    reset_state()

    def on_mouse_movement(event):
        """Highlight the blob the mouse is hovering on."""
        if event.xdata is None or event.ydata is None:
            blob_id = -1
        else:
            blob_id = state.idmap[round(event.ydata), round(event.xdata)]
        if state.blob_id != blob_id:
            state.blob_id = blob_id
            # override color of the blob
            state.img[:] = state.background[:]
            blob = state.blob_cls.get_instance(state.blob_id)
            if blob is None:
                print("\r"f"│{' ' * 10}│{' ' * 6}│"
                      f"{' ' * 10}│{' ' * 10}│{' ' * 10}│{' ' * 10}│"
                      f"{' ' * 10}│{' ' * 10}│{' ' * 16}│{' ' * 10}│",
                      end="")
                sys.stdout.flush()
                update_img()
                return

            # ... print blobs properties.
            print("\r"f"│{blob.id:^10}│{'+' if blob.sign else '-':^6}│"
                  f"{blob.I:^10.2e}│{blob.Dy:^10.2e}│{blob.Dx:^10.2e}│{blob.G:^10.2e}│"
                  f"{blob.M:^10.2e}│{blob.A:^10.2e}│"
                  f"{'-'.join(map(str, blob.box)):^16}│"
                  f"{blob.prior_forks:^10}│",
                  end="")
            sys.stdout.flush()

            # paint over the blob ...
            paint_over(state.img[state.img_slice], None, blob.box,
                       mask=blob.mask__,
                       fill_color=WHITE)
            # ... and its adjacents
            for adj_blob, pose in zip(*blob.adj_blobs):
                paint_over(state.img[state.img_slice], None, adj_blob.box,
                           mask=adj_blob.mask__,
                           fill_color=POSE2COLOR[pose])
            # Finally, update the image
            update_img()

    def on_click(event):
        """Transition between layers."""
        blob = state.blob_cls.get_instance(state.blob_id)
        if event.key == "control":       # Look into rlayer
            if blob.rlayers and blob.rlayers[0] and blob is not None:
                state.layers_stack.append((blob, 'r'))
                reset_state()
        elif event.key == "shift":    # Look into dlayer
            if blob.dlayers and blob.dlayers[0] and blob is not None:
                state.layers_stack.append((blob, 'd'))
                reset_state()
        else:
            if (len(state.layers_stack) > 1):
                state.layers_stack.pop()
                reset_state()

    def on_key_press(event):
        if event.key == 'd':
            state.show_gradient = not state.show_gradient
            update_img()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_movement)
    fig.canvas.mpl_connect('button_release_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    print("hit 'q' to exit")
    print("hit 'd' to toggle show gradient")
    print("blob:")
    print(f"┌{'─'*10}┬{'─'*6}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*10}"
          f"┬{'─'*10}┬{'─'*10}┬{'─'*16}┬{'─'*10}┐")
    print(f"|{'id':^10}│{'sign':^6}│{'I':^10}│{'Dy':^10}│{'Dx':^10}│{'G':^10}│"
          f"{'M':^10}│{'A':^10}│{'box':^16}│{'fork':^10}│")
    print(f"├{'─' * 10}┼{'─' * 6}┼{'─' * 10}┼{'─' * 10}┼{'─' * 10}┼{'─' * 10}"
          f"┼{'─' * 10}┼{'─' * 10}┼{'─' * 16}┼{'─' * 10}┤")


    plt.show()

    print()


def blank_image(shape, fill_val=None):
    '''Create an empty numpy array of desired shape.'''

    if len(shape) == 2:
        height, width = shape
    else:
        y0, yn, x0, xn = shape
        height = yn - y0
        width = xn - x0
    if fill_val is None:
        fill_val = MASKING_VAL
    return np.full((height, width, 3), fill_val, 'uint8')

def paint_over(map, sub_map, sub_box,
               box=None, mask=None, mv=MASKING_VAL,
               fill_color=None):
    '''Paint the map of a sub-structure onto that of parent-structure.'''

    if box is None:
        y0, yn, x0, xn = sub_box
    else:
        y0, yn, x0, xn = localize_box(sub_box, box)
    if mask is None:
        if fill_color is None:
            map[y0:yn, x0:xn][sub_map != mv] = sub_map[sub_map != mv]
        else:
            map[y0:yn, x0:xn][sub_map != mv] = fill_color
    else:
        if fill_color is None:
            map[y0:yn, x0:xn][~mask] = sub_map[~mask]
        else:
            map[y0:yn, x0:xn][~mask] = fill_color
    return map

def localize_box(box, global_box):
    '''
    Compute local coordinates for given bounding box.
    Used for overwriting map of parent structure with
    maps of sub-structure, or other similar purposes.
    Parameters
    ----------
    box : tuple
        Bounding box need to be localized.
    global_box : tuple
        Reference to which box is localized.
    Return
    ------
    out : tuple
        Box localized with localized coordinates.
    '''
    y0s, yns, x0s, xns = box
    y0, yn, x0, xn = global_box

    return y0s - y0, yns - y0, x0s - x0, xns - x0