"""
Provide the function visualize_blobs to display
formed blobs interactively.
"""

import sys
import numpy as np
import cv2 as cv

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


def visualize_blobs(frame, layer='r', window_size=None, winname="Blobs"):
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
    window_size : tuple, optional
        The size of the window to display.
    winname : str, optional
        The name of the window. Defaults to "Blobs".
    """
    print("Preparing for visualization ...", end="")

    height, width = frame.dert__[0].shape
    if window_size is None:
        window_size = (
            max(width, MIN_WINDOW_WIDTH),
            max(height, MIN_WINDOW_HEIGHT),
        )

    # Prepare state object
    state = SimpleNamespace(
        img=None, background=None, idmap=None, blob_id=None, img_slice=None, blob_cls=None,
        layers_stack=[(frame, layer)],
    )

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
        rY, rX = frame.root_dert__[0].shape
        y0e = max(0, y0 - 1)
        yne = min(rY, yn + 1)
        x0e = max(0, x0 - 1)
        xne = min(rX, xn + 1)  # e is for extended
        state.img_slice = slice(y0e, yne), slice(x0e, xne)
        state.background = blank_image((height, width))
        idmap = np.full((height, width), -1).astype('int64')
        # Prepare blob ID map and background
        for blob in blob_:
            paint_over(idmap[state.img_slice], None, blob.box,
                       mask=blob.mask__,
                       fill_color=blob.id)
            paint_over(state.background[state.img_slice], None, blob.box,
                       mask=blob.mask__,
                       fill_color=[blob.sign * 32] * 3)
        state.img = state.background.copy()
        state.idmap = cv.resize(idmap.astype('uint64'), window_size,
                          interpolation=cv.INTER_NEAREST)
        state.blob_id = -1

    reset_state()
    def mouse_call(event, x, y, flags, param):
        wx, wy = window_size
        x = max(0, min(wx - 1, x))
        y = max(0, min(wy - 1, y))
        blob_id = state.idmap[y, x]
        if event == cv.EVENT_MOUSEMOVE:
            if state.blob_id != blob_id:
                state.blob_id = blob_id
                # override color of the blob
                state.img[:] = state.background[:]
                blob = state.blob_cls.get_instance(state.blob_id)
                if blob is None:
                    print("\r", end="\t\t\t\t\t\t\t")
                    sys.stdout.flush()
                    return
                paint_over(state.img[state.img_slice], None, blob.box,
                           mask=blob.mask__,
                           fill_color=WHITE)
                # ... and its adjacents
                for adj_blob, pose in zip(*blob.adj_blobs):
                    paint_over(state.img[state.img_slice], None, adj_blob.box,
                               mask=adj_blob.mask__,
                               fill_color=POSE2COLOR[pose])


                # ... print blobs properties.
                print("\rblob:",
                      "id =", blob.id,
                      "sign =", "'+'" if blob.sign else "'-'",
                      "I =", blob.I,
                      "Dy =", blob.Dy,
                      "Dx =", blob.Dx,
                      "G =", blob.G,
                      "M = ",blob.M,
                      "A =", blob.A,
                      "box =", blob.box,
                      "fork =", ''.join(blob.prior_forks),
                      end="\t\t\t")
                sys.stdout.flush()
        elif event == cv.EVENT_LBUTTONUP:
            blob = state.blob_cls.get_instance(state.blob_id)
            if flags & cv.EVENT_FLAG_CTRLKEY:       # Look into rlayer
                if blob.rlayers and blob.rlayers[0]:
                    state.layers_stack.append((blob, 'r'))
                    reset_state()
            elif flags & cv.EVENT_FLAG_SHIFTKEY:    # Look into dlayer
                if blob.dlayers and blob.dlayers[0]:
                    state.layers_stack.append((blob, 'd'))
                    reset_state()
            else:                                   # Go back to parent
                if (len(state.layers_stack) > 1):
                    state.layers_stack.pop()
                    reset_state()


    cv.namedWindow(winname)
    cv.setMouseCallback(winname, mouse_call)
    print("hit 'q' to exit")

    while True:
        cv.imshow(winname,
                  cv.resize(state.img,
                            window_size,
                            interpolation=cv.INTER_NEAREST))
        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()
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