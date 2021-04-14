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


def visualize_blobs(idmap, blob_, window_size=None, winname="Blobs"):
    """
    Visualize blobs after clustering.
    Highlight the blob the mouse is hovering on and its
    adjacents.
    """
    print("Preparing for visualization...", end="")
    blob_cls = blob_[0].__class__
    height, width = idmap.shape
    if window_size is None:
        window_size = (
            max(width, MIN_WINDOW_WIDTH),
            max(height, MIN_WINDOW_HEIGHT),
        )
    background = blank_image((height, width))
    for blob in blob_:
        over_draw(background, None, blob.box,
                  mask=blob.mask__,
                  fill_color=[blob.sign * 32] * 3)

    idmap = cv.resize(idmap.astype('uint64'), window_size,
                      interpolation=cv.INTER_NEAREST)
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
                          mask=blob.mask__,
                          fill_color=WHITE)
                # ... and its adjacents
                for adj_blob, pose in zip(*blob.adj_blobs):
                    over_draw(img, None, adj_blob.box,
                              mask=adj_blob.mask__,
                              fill_color=POSE2COLOR[pose])

                rD = blob.Dert.Dy / blob.Dert.Dx if blob.Dert.Dx else 2 * blob.Dert.Dy
                dir_val = abs(blob.Dert.G * rD)

                # ... print blobs properties.
                print("\rblob:",
                      "id =", blob.id,
                      "sign =", "'+'" if blob.sign else "'-'",
                      "I =", blob.Dert.I,
                      "G =", blob.Dert.G,
                      "Dy =", blob.Dert.Dy,
                      "Dx =", blob.Dert.Dx,
                      "S =", blob.A,
                      "M = ",blob.Dert.M,
                      "dir_val = ", dir_val,
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
    print()