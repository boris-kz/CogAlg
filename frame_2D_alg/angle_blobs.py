import numpy as np
import numpy.ma as ma
# Filters ------------------------------------------------------------------------
from frame_2D_alg.filters import get_filters
get_filters(globals())          # imports all filters at once
# --------------------------------------------------------------------------------
'''
    angle_blob is a component of intra_blob
'''
# ***************************************************** ANGLE BLOBS FUNCTIONS *******************************************
# Functions:
# -angle_blobs()
# -correct_da()
# ***********************************************************************************************************************

def angle_blobs(blob):  # compute and compare angles

    # Compute angles:

    dy = ma.array(blob.dert__[:, :, 1], mask=~blob.map)
    dx = ma.array(blob.dert__[:, :, 2], mask=~blob.map)

    a__ = np.arctan2(dy, dx) * angle_coef + 128

    # Compare angles:

    dert__ = ma.empty(shape=blob.dert__.shape, dtype=int)  # initialize dert__

    day__ = correct_da(a__[2:, 1:-1] - a__[:-2, 1:-1])  # vertical comp between rows -> dy, (1:-1): first and last column are discarded
    dax__ = correct_da(a__[1:-1, 2:] - a__[1:-1, :-2])  # lateral comp between columns -> dx, (1:-1): first and last row are discarded
    ga__ = np.hypot(day__, dax__) - ave                 # deviation of gradient, initially approximated as |dy| + |dx|

    # pack all derts into dert__
    dert__[:, :, 0] = a__
    dert__[1:-1, 1:-1, 1] = day__  # first row, last row, first column and last-column are discarded
    dert__[1:-1, 1:-1, 2] = dax__
    dert__[1:-1, 1:-1, 3] = ga__

    blob.new_dert__[0] = dert__  # pack dert__ into blob
    # ---------- angle_blobs() end --------------------------------------------------------------------------------------

def correct_da(da):
    " make da -128->127 instead of -256->255 "
    where = da > 128
    da[where] = da[where] - 256
    where = da < -128
    da[where] = da[where] + 256
    return da
    # ---------- correct_da() end ---------------------------------------------------------------------------------------