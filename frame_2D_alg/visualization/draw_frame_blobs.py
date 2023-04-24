"""
Provide the function visualize_blobs to display
formed blobs interactively.
"""

import sys
import numpy as np
import cv2 as cv
from utils import blank_image, paint_over

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


def visualize_blobs(idmap, blob_, window_size=None, winname="Blobs"):
    """
    Visualize blobs after clustering.
    Highlight the blob the mouse is hovering on and its
    adjacents.
    """
    print("Preparing for visualization ...", end="")

    blob_cls = blob_[0].__class__
    height, width = idmap.shape

    # Prepare the image
    if window_size is None:
        window_size = (
            max(width, MIN_WINDOW_WIDTH),
            max(height, MIN_WINDOW_HEIGHT),
        )
    background = blank_image((height, width))

    # Prepare blob ID map
    for blob in blob_:
        paint_over(background, None, blob.box,
                   mask=blob.mask__,
                   fill_color=[blob.sign * 32] * 3)

    idmap = cv.resize(idmap.astype('uint64'), window_size,
                      interpolation=cv.INTER_NEAREST)
    img = background.copy()
    state = dict(
        blob_id=-1,
        layer=0,
    )

    def mouse_call(event, x, y, flags, param):
        wx, wy = window_size
        x = max(0, min(wx - 1, x))
        y = max(0, min(wy - 1, y))
        if event == cv.EVENT_MOUSEMOVE:
            if state['blob_id'] != idmap[y, x]:
                state['blob_id'] = idmap[y, x]
                # override color of the blob
                img[:] = background.copy()
                blob = blob_cls.get_instance(state['blob_id'])
                if blob is None:
                    print("\r", end="\t\t\t\t\t\t\t")
                    sys.stdout.flush()
                    return
                paint_over(img, None, blob.box,
                           mask=blob.mask__,
                           fill_color=WHITE)
                # ... and its adjacents
                for adj_blob, pose in zip(*blob.adj_blobs):
                    paint_over(img, None, adj_blob.box,
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
                      "fork =", blob.prior_forks,
                      end="\t\t\t")
                sys.stdout.flush()

    cv.namedWindow(winname)
    cv.setMouseCallback(winname, mouse_call)
    print("hit 'q' to exit")

    while True:
        cv.imshow(winname,
                  cv.resize(img,
                            window_size,
                            interpolation=cv.INTER_NEAREST))
        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()
    print()