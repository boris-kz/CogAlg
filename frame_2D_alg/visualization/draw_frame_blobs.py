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

    _, _, height, width = frame.box

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
        box_slice = blob.box.slice()
        dy_slice = dy__[state.img_slice][box_slice][~blob.mask__]
        dx_slice = dx__[state.img_slice][box_slice][~blob.mask__]
        dy_index = 3 if len(blob.der__t) > 5 else 1
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
        state.img_slice = blob_[0].root_ibox.slice()
        state.background[:] = MASKING_VAL
        state.idmap[:] = -1
        # Prepare blob ID map and background
        local_idmap = state.idmap[state.img_slice]
        local_background = state.background[state.img_slice]
        for blob in blob_:
            local_idmap[blob.box.slice()][~blob.mask__] = blob.id  # fill idmap with blobs' ids
            local_background[blob.box.slice()][~blob.mask__] = blob.sign * 32  # fill image with blobs
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
            state.img[state.img_slice][blob.box.slice()][~blob.mask__] = WHITE
            # ... and its adjacents
            for adj_blob, pose in zip(*blob.adj_blobs):
                state.img[state.img_slice][adj_blob.box.slice()][~adj_blob.mask__] = POSE2COLOR[pose]
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
        y0, x0, yn, xn = shape
        height = yn - y0
        width = xn - x0
    if fill_val is None:
        fill_val = MASKING_VAL
    return np.full((height, width, 3), fill_val, 'uint8')