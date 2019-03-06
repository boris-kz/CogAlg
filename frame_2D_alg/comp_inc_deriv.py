import numpy as np
from collections import deque
from frame_2D_alg import generic
from frame_2D_alg.misc import get_filters
get_filters(globals()) # imports all filters at once

# ***************************************************** INC_DERIV FUNCTIONS *********************************************
# Functions:
# -inc_deriv()
# -comp_g()
# ***********************************************************************************************************************

def inc_deriv(blob):    # same functionality as image_to_blobs() in frame_blobs.py

    global height, width
    height, width = blob.map.shape
    sub_blob = [0, 0, 0, 0, []]
    comp_g(sub_blob, blob.dert__[:, :, -1], blob.map)
    seg_ = deque()

    for y in range(1, height - 1):
        P_ = generic.form_P_(y, sub_blob)  # horizontal clustering
        P_ = generic.scan_P_(P_, seg_, sub_blob)
        seg_ = generic.form_seg_(P_, sub_blob)

    while seg_:  generic.form_blob(seg_.popleft(), sub_blob)
    blob.sub_blob_.append(sub_blob)
    return sub_blob

    # ---------- inc_deriv() end ----------------------------------------------------------------------------------------

def comp_g(sub_blob, g__, map):  # compare g within sub blob
    dert__ = ma.empty(shape=(width, height, 4), dtype=int)  # initialize dert__

    dy__ = g__[2:, 1:-1] - g__[:-2, 1:-1]   # vertical comp between rows -> dy, (1:-1): first and last column are discarded
    dx__ = g__[1:-1, 2:] - g__[1:-1, :-2]   # lateral comp between columns -> dx, (1:-1): first and last row are discarded
    gg__ = np.abs(dy__) + np.abs(dx__) - ave  # deviation of gradient, initially approximated as |dy| + |dx|

    dert__[:, :, 0] = g__
    dert__[1:-1, 1:-1, 1] = dy__  # first row, last row, first column and last-column are discarded
    dert__[1:-1, 1:-1, 2] = dx__
    dert__[1:-1, 1:-1, 3] = gg__

    sub_blob.append(dert__)
    # ---------- comp_g() end -------------------------------------------------------------------------------------------
