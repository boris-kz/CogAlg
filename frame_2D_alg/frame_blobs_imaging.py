"""
Provide the function visualize_blobs to display
formed blobs interactively.
"""

import sys
import numpy as np
import cv2 as cv
from utils import blank_image, over_draw

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


def visualize_blobs(idmap, blob_cls, window_size=None, winname="blobs"):
    """
    Visualize blobs after clustering.
    Highlight the blob the mouse is hovering on and its
    adjacents.
    """
    print("Preparing for visualization...", end="")
    height, width = idmap.shape
    if window_size is None:
        window_size = (
            max(width, MIN_WINDOW_WIDTH),
            max(height, MIN_WINDOW_HEIGHT),
        )
    background = blank_image((height, width))
    masks = {}
    for blobid in range(idmap.max() + 1):
        blob = blob_cls.get_instance(blobid)
        y0, yn, x0, xn = blob.box
        # Form masks for fast transition
        mask = ~(idmap[y0:yn, x0:xn] == blobid)
        masks[blobid] = mask
        # Prepare background image
        over_draw(background, None, blob.box,
                  mask=mask,
                  fill_color=[blob.sign * 32] * 3)

    idmap = cv.resize(idmap.astype('uint64'), window_size, interpolation=cv.INTER_NEAREST)
    img = background.copy()
    blobid = [-1]

    def mouse_call(event, x, y, flags, param):
        x = min(window_size[0] - 1, max(x, 0))
        y = min(window_size[1] - 1, max(y, 0))
        if event == cv.EVENT_MOUSEMOVE:
            if blobid[0] != idmap[y, x]:
                blobid[0] = idmap[y, x]
                # override color of the blob
                img[:] = background.copy()
                blob = blob_cls.get_instance(blobid[0])
                if blob is None:
                    print("\r", end="\t\t\t\t\t\t\t")
                    sys.stdout.flush()
                    return
                over_draw(img, None, blob.box,
                          mask=masks[blobid[0]],
                          fill_color=WHITE)
                # ... and its adjacents
                for adj_blob, pose in blob.adj_blobs[0]:
                    over_draw(img, None, adj_blob.box,
                              mask=masks[adj_blob.id],
                              fill_color=POSE2COLOR[pose])
                # ... print blobs properties.
                print("\rblob:",
                      "id =", blob.id,
                      "sign =", "'+'" if blob.sign else "'-'",
                      "I =", blob.I,
                      "G =", blob.G,
                      "Dy =", blob.Dy,
                      "Dx =", blob.Dx,
                      "S =", blob.S,
                      end="\t\t\t")
                sys.stdout.flush()

    cv.namedWindow(winname)
    cv.setMouseCallback(winname, mouse_call)
    print()
    while True:
        cv.imshow(winname,
                  cv.resize(img,
                            window_size,
                            interpolation=cv.INTER_NEAREST))
        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()