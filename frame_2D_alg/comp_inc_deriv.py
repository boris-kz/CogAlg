import numpy as np
import numpy.ma as ma
from collections import deque
import generic
# Filters ------------------------------------------------------------------------
from frame_2D_alg.filters import get_filters
get_filters(globals()) # imports all filters at once
# --------------------------------------------------------------------------------

# ***************************************************** INC_DERIV FUNCTIONS *********************************************
# Functions:
# -inc_deriv()
# -comp_g()
# ***********************************************************************************************************************

def inc_deriv(blob):    # same functionality as image_to_blobs() in frame_blobs.py

    global height, width
    height, width = blob.map.shape

    blob.params.append(0)       # Add inc_deriv's I
    blob.params.append(0)       # Add inc_deriv's Dy
    blob.params.append(0)       # Add inc_deriv's Dx
    blob.params.append(0)       # Add inc_deriv's G
    blob.sub_blob_.append([])   # add inc_deriv_blob_

    if height < 3 or width < 3:
        return False

    g__ = ma.array(blob.dert__[:, :, 3], mask=~blob.map)    # apply mask = ~map
    comp_g(blob, g__)

    if blob.new_dert__[0].mask.all():
        return False

    seg_ = deque()

    for y in range(1, height - 1):
        P_ = generic.form_P_(y, blob)           # horizontal clustering
        P_ = generic.scan_P_(P_, seg_, blob)    # vertical clustering
        seg_ = generic.form_seg_(P_, blob)      # vertical clustering

    while seg_:  generic.form_blob(seg_.popleft(), blob)    # terminate last running segments

    return True
    # ---------- inc_deriv() end ----------------------------------------------------------------------------------------
def comp_g(blob, g__):  # compare g within sub blob
    dert__ = ma.empty(shape=(height, width, 4), dtype=int)  # initialize dert__

    dy__ = g__[2:, 1:-1] - g__[:-2, 1:-1]   # vertical comp between rows -> dy, (1:-1): first and last column are discarded
    dx__ = g__[1:-1, 2:] - g__[1:-1, :-2]   # lateral comp between columns -> dx, (1:-1): first and last row are discarded
    gg__ = np.hypot(dy__, dx__) - ave       # deviation of gradient, initially approximated as |dy| + |dx|

    # pack all derts into dert__
    dert__[:, :, 0] = g__
    dert__[1:-1, 1:-1, 1] = dy__  # first row, last row, first column and last-column are discarded
    dert__[1:-1, 1:-1, 2] = dx__
    dert__[1:-1, 1:-1, 3] = gg__

    blob.new_dert__[0] = dert__ # pack dert__ into blob
    # ---------- comp_g() end -------------------------------------------------------------------------------------------