import numpy as np
import numpy.ma as ma
from math import hypot
from collections import deque
import generic
from filters import get_filters
get_filters(globals())  # imports all filters at once
# --------------------------------------------------------------------------------
'''
    inc_range is a component of intra_blob
'''
# ***************************************************** INC_RANGE FUNCTIONS *********************************************
# Functions:
# -inc_range()
# -comp_p()
# ***********************************************************************************************************************

def inc_range(blob): # same functionality as image_to_blobs() in frame_blobs.py

    global height, width
    height, width = blob.map.shape

    for _ in range(4):
        blob.params.append(0)
    blob.sub_blob_.append([])

    if height < 3 or width < 3:
        return False

    comp_p(blob, blob.rng + 1)

    if blob.new_dert__[0].mask.all():
        return False

    seg_ = deque()

    for y in range(1, height - 1):
        P_ = generic.form_P_(y, blob)                   # horizontal clustering
        P_ = generic.scan_P_(P_, seg_, blob, inc_rng=True)    # vertical clustering
        seg_ = generic.form_seg_(P_, blob, inc_rng=True)

    while seg_:  generic.form_blob(seg_.popleft(), blob, inc_rng=True)

    return True

    # ---------- inc_range() end ----------------------------------------------------------------------------------------

def comp_p(blob, rng):   # compare rng-distant pixels within blob

    p__ = ma.array(blob.dert__[:, :, 0], mask=~blob.map)  # apply mask = ~map
    dy__ = ma.array(blob.dert__[:, :, 1], mask=~blob.map)
    dx__ = ma.array(blob.dert__[:, :, 2], mask=~blob.map)

    dert__ = ma.empty(shape=(height, width, 4), dtype=int)  # initialize new dert__
    comp_rng = rng * 2

    # vertical comp:
    d__ = p__[comp_rng:, rng:-rng] - p__[:-comp_rng, rng:-rng]  # bilateral comparison between p at coordinates (x, y + rng) and p at coordinates (x, y - rng)
    dy__[rng:-rng, rng:-rng] += d__                             # bilateral accumulation on dy (x, y)

    # horizontal comp:
    d__ = p__[rng:-rng, comp_rng:] - p__[rng:-rng, :-comp_rng]  # bilateral comparison between p at coordinates (x + rng, y) and p at coordinates (x - rng, y)
    dx__[rng:-rng, rng:-rng] += d__                             # bilateral accumulation on dy (x, y)

    # diagonal comparison:

    for xd in range(1, rng):
        yd = rng - xd
        bxd = xd * 2
        byd = comp_rng - bxd    # y and x distance between comparands
        hyp = hypot(byd, bxd)
        y_coef = byd / hyp      # to decompose d into dy
        x_coef = bxd / hyp      # to decompose d into dx

        # top-left and bottom-right quadrants:

        d__ = p__[byd:, bxd:] - p__[:-byd, :-bxd]   # comparison between p (x - xd, y - yd) and p (x + xd, y + yd)
        # decompose d to dy, dx:
        temp_dy__ = d__ * y_coef                    # buffer for dy accumulation
        temp_dx__ = d__ * x_coef                    # buffer for dx accumulation
        # accumulate dy, dx:
        dy__[yd:-yd, xd:-xd] += temp_dy__.astype(int)   # bilateral accumulation on dy (x, y)
        dx__[yd:-yd, xd:-xd] += temp_dx__.astype(int)   # bilateral accumulation on dx (x, y)

        # top-right and bottom-left quadrants:

        d__ = p__[byd:, :-bxd] - p__[:-byd, bxd:]   # comparison between p (x + xd, y - yd) and p (x - xd, y + yd)
        # decompose d to dy, dx:
        temp_dy__ = d__ * y_coef                    # buffer for dy accumulation
        temp_dx__ = -(d__ * x_coef)                 # buffer for dx accumulation, sign inverted with comp direction
        # accumulate dy, dx:
        dy__[yd:-yd, xd:-xd] += temp_dy__.astype(int)   # bilateral accumulation on dy (x, y)
        dx__[yd:-yd, xd:-xd] += temp_dx__.astype(int)   # bilateral accumulation on dx (x, y)


    g__ = np.abs(dy__[rng:-rng, rng:-rng]) + np.abs(dx__[rng:-rng, rng:-rng]) - ave * blob.ncomp  # compute g__

    dert__[:, :, 0] = p__
    dert__[:, :, 1] = dy__
    dert__[:, :, 2] = dx__
    dert__[rng:-rng, rng:-rng, 3] = g__

    blob.new_dert__[0] = dert__
    # ---------- comp_p() end -------------------------------------------------------------------------------------------
