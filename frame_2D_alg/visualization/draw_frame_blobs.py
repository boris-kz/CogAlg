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
        # plots
        quiver=None,
        blob_slices=None,
        P_links=None,
        # flags
        show_gradient=False,
        show_slices=False,
        show_links=False,
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
        box_slice = blob.ibox.slice()
        state.gradient[1][box_slice] = -blob.der__t.dy
        state.gradient[0][box_slice] = blob.der__t.dx
        state.gradient_mask[box_slice] = blob.mask__

        # Apply quiver
        state.quiver = ax.quiver(
            *state.gradient_mask.nonzero()[::-1],
            *state.gradient[:, state.gradient_mask])

    def update_blob_slices():
        if state.blob_slices is not None:
            for line, L_text in state.blob_slices:
                line.remove()
                L_text.remove()
            state.blob_slices = None

        blob = state.blob_cls.get_instance(state.blob_id)
        if blob is None or not blob.dlayers or not blob.dlayers[0] or not state.show_slices:
            return
        edge = blob.dlayers[0][0]
        if not edge.node_: return
        y0, x0, *_ = blob.ibox
        state.blob_slices = []
        for P in edge.node_:
            y, x = P.yx
            y_, x_, *_ = np.array([*zip(*P.dert_)])
            L = len(x_)
            if L > 1:
                blob_slices_plot = ax.plot(x_+x0, y_+y0, 'bo-', linewidth=1, markersize=2)[0]
            else:
                s, c = P.axis
                x_ = np.array([x-c, x, x+c])
                y_ = np.array([y-s, y, y+s])
                blob_slices_plot = ax.plot(x_ + x0, y_ + y0, 'b-', linewidth=1, markersize=2)[0]
            state.blob_slices += [(
                blob_slices_plot,
                ax.text(x+x0, y+y0, str(L), color = 'b', fontsize = 12),
            )]

    def update_P_links():
        if state.P_links is not None:
            for line in state.P_links:
                line.remove()
            state.P_links = None

        blob = state.blob_cls.get_instance(state.blob_id)
        if blob is None or not blob.dlayers or not blob.dlayers[0] or not state.show_links:
            return
        edge = blob.dlayers[0][0]
        if not edge.node_: return
        link_ = set()
        for P in edge.node_:
            for _P, _ in P.link_H[0]:
                if _P.id < P.id:
                    link_ |= {(_P, P)}
        y0, x0, *_ = blob.ibox
        state.P_links = []
        for _P, P in link_:
            (_y, _x), (y, x) = _P.yx, P.yx
            state.P_links += ax.plot([_x+x0,x+x0], [_y+y0,y+y0], 'ko-', linewidth=2, markersize=4)

    def update_img():
        update_gradient()
        update_blob_slices()
        update_P_links()
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
            local_idmap[blob.box.slice()][blob.mask__] = blob.id  # fill idmap with blobs' ids
            local_background[blob.box.slice()][blob.mask__] = blob.sign * 32  # fill image with blobs
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
            state.img[state.img_slice][blob.box.slice()][blob.mask__] = WHITE
            # ... and its adjacents
            for adj_blob, pose in zip(*blob.adj_blobs):
                state.img[state.img_slice][adj_blob.box.slice()][adj_blob.mask__] = POSE2COLOR[pose]
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
                pass
                # state.layers_stack.append((blob, 'd'))
                # reset_state()
        else:
            if (len(state.layers_stack) > 1):
                state.layers_stack.pop()
                reset_state()

    def on_key_press(event):
        if event.key == 'd':
            state.show_gradient = not state.show_gradient
        elif event.key == 'z':
            state.show_slices = not state.show_slices
        elif event.key == 'x':
            state.show_links = not state.show_links
        else:
            return
        update_img()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_movement)
    fig.canvas.mpl_connect('button_release_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    print("hit 'q' to exit")
    print("hit 'd' to toggle show gradient")
    print("hit 'z' to toggle show slices")
    print("hit 'x' to toggle show links")
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