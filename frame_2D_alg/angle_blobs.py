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

    blob.params.append(0)       # Add A
    blob.params.append(0)       # Add Day
    blob.params.append(0)       # Add Dax
    blob.params.append(0)       # Add Ga

    blob.sub_blob_.append([])

    if height < 3 or width < 3:
        return False

    a__ = get_angle(blob.dert__, blob.map)
    comp_angle(blob, a__)

    if blob.new_dert__[0].mask.all():
        return False

    seg_ = deque()

    for y in range(1, height - 1):
        P_ = generic.form_P_(y, blob)           # horizontal clustering
        P_ = generic.scan_P_(P_, seg_, blob)    # vertical clustering
        seg_ = generic.form_seg_(P_, blob)      # vertical clustering

    while seg_:  generic.form_blob(seg_.popleft(), blob)    # terminate last running segments

    return True
    # ---------- blob_to_ablobs() end -----------------------------------------------------------------------------------

def get_angle(dert__, map):  # default = False: no higher-line for first line
    " compute angle of gradient in and adjacent to selected gblob "

    dy = ma.array(dert__[:, :, 1], mask=~map)
    dx = ma.array(dert__[:, :, 2], mask=~map)

    a__ = np.arctan2(dy, dx) * angle_coef + 128
    return a__
    # ---------- get_angle() end ----------------------------------------------------------------------------------------

def comp_angle(blob, a__):
    " compare angle of adjacent gradients within frame per gblob "
    dert__ = ma.empty(shape=(height, width, 4), dtype=int)  # initialize dert__

    dy__ = correct_da(a__[2:, 1:-1] - a__[:-2, 1:-1])   # vertical comp between rows -> dy, (1:-1): first and last column are discarded
    dx__ = correct_da(a__[1:-1, 2:] - a__[1:-1, :-2])   # lateral comp between columns -> dx, (1:-1): first and last row are discarded
    g__ = np.hypot(dy__, dx__) - ave             # deviation of gradient, initially approximated as |dy| + |dx|

    # pack all derts into dert__
    dert__[:, :, 0] = a__
    dert__[1:-1, 1:-1, 1] = dy__  # first row, last row, first column and last-column are discarded
    dert__[1:-1, 1:-1, 2] = dx__
    dert__[1:-1, 1:-1, 3] = g__

    blob.new_dert__[0] = dert__ # pack dert__ into blob
    # ---------- comp_angle() end ---------------------------------------------------------------------------------------

def correct_da(da):
    " make da 0 - 128 instead of 0 - 255 "
    where = da > 128
    da[where] = da[where] - 256
    where = da < -128
    da[where] = da[where] + 256
    return da
    # ---------- correct_da() end ---------------------------------------------------------------------------------------