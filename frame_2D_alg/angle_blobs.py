import numpy as np
import numpy.ma as ma
from collections import deque
import generic
# Filters ------------------------------------------------------------------------
from frame_2D_alg.filters import get_filters
get_filters(globals())          # imports all filters at once
# --------------------------------------------------------------------------------
'''
    angle_blob is a component of intra_blob
'''
# ***************************************************** ANGLE BLOBS FUNCTIONS *******************************************
# Functions:
# -blob_to_ablobs()
# -get_angle()
# -comp_angle()
# -correct_da
# ***********************************************************************************************************************

def blob_to_ablobs(blob):  # compute and compare angle, define ablobs, accumulate a, dxa, dya, ga in all reps within gblob
    ''' same functionality as image_to_blobs() in frame_blobs.py'''

    global height, width
    height, width = blob.map.shape

    a__ = get_angle(blob.dert__, blob.map)
    comp_angle(blob, a__)
    seg_ = deque()

    for y in range(1, height - 1):
        P_ = generic.form_P_(y, blob)  # horizontal clustering
        P_ = generic.scan_P_(P_, seg_, blob)  # vertical clustering
        seg_ = generic.form_seg_(P_, blob)

    while seg_:  generic.form_blob(seg_.popleft(), blob)

    blob.e_.append(sub_blob)
    # ---------- blob_to_ablobs() end -----------------------------------------------------------------------------------

def get_angle(dert__, map):  # default = False: no higher-line for first line
    " compute angle of gradient in and adjacent to selected gblob "

    dy = dert__[:, :, 1]
    dx = dert__[:, :, 2]
    a__ = ma.empty(map.shape, dtype=int)

    a__[map] = np.arctan2(dy, dx, where=[map])[map] * angle_coef + 128
    a__.mask = ~map
    return a__
    # ---------- get_angle() end ----------------------------------------------------------------------------------------

def comp_angle(sub_blob, a__):
    " compare angle of adjacent gradients within frame per gblob "
    dert__ = ma.empty(shape=(width, height, 4), dtype=int)  # initialize dert__

    dy__ = correct_da(a__[2:, 1:-1] - a__[:-2, 1:-1])   # vertical comp between rows -> dy, (1:-1): first and last column are discarded
    dx__ = correct_da(a__[1:-1, 2:] - a__[1:-1, :-2])   # lateral comp between columns -> dx, (1:-1): first and last row are discarded
    g__ = np.abs(dy__) + np.abs(dx__) - ave             # deviation of gradient, initially approximated as |dy| + |dx|

    dert__[:, :, 0] = a__
    dert__[1:-1, 1:-1, 1] = dy__  # first row, last row, first column and last-column are discarded
    dert__[1:-1, 1:-1, 2] = dx__
    dert__[1:-1, 1:-1, 3] = g__

    # blob.new_dert__ =
    # ---------- comp_angle() end ---------------------------------------------------------------------------------------

def correct_da(da):
    " make da 0 - 128 instead of 0 - 255 "
    where = da > 128
    da[where] = da[where] - 256
    where = da < -128
    da[where] = da[where] + 256
    return da
    # ---------- correct_da() end ---------------------------------------------------------------------------------------
