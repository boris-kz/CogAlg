import numpy as np
import numpy.ma as ma
from math import hypot
from collections import deque
import generic_branch
from filters import get_filters
get_filters(globals())  # imports all filters at once
# --------------------------------------------------------------------------------
'''
    inc_range is a component of intra_blob
'''
# ***************************************************** INC_RANGE FUNCTIONS *********************************************
# Functions:
# -comp_range()
# ***********************************************************************************************************************

def comp_range(blob):   # compare rng-distant pixels within blob

    rng = blob.rng + 1
    p__ = ma.array(blob.dert__[:, :, 0], mask=~blob.map)  # apply mask = ~map
    dy__ = ma.array(blob.dert__[:, :, 1], mask=~blob.map)
    dx__ = ma.array(blob.dert__[:, :, 2], mask=~blob.map)

    dert__ = ma.empty(shape=blob.dert__.shape, dtype=int)  # initialize new dert__
    comp_rng = rng * 2

    # vertical comp:
    d__ = p__[comp_rng:, rng:-rng] - p__[:-comp_rng, rng:-rng]  # bilateral comparison between p at coordinates (x, y + rng) and p at coordinates (x, y - rng)
    dy__[rng:-rng, rng:-rng] += d__                             # bilateral accumulation on dy (x, y)

    # horizontal comp:
    d__ = p__[rng:-rng, comp_rng:] - p__[rng:-rng, :-comp_rng]  # bilateral comparison between p at coordinates (x + rng, y) and p at coordinates (x - rng, y)
    dx__[rng:-rng, rng:-rng] += d__                             # bilateral accumulation on dy (x, y)

    # diagonal comparison:

    for xd in range(1, rng):
        yd = rng - xd               # half y and x distance between comparands
        bi_xd = xd * 2
        bi_yd = comp_rng - bi_xd    # y and x distance between comparands
        hyp = hypot(bi_yd, bi_xd)
        y_coef = bi_yd / hyp      # to decompose d into dy
        x_coef = bi_xd / hyp      # to decompose d into dx

        # top-left and bottom-right quadrants:

        d__ = p__[bi_yd:, bi_xd:] - p__[:-bi_yd, :-bi_xd]   # comparison between p (x - xd, y - yd) and p (x + xd, y + yd)
        # decompose d to dy, dx:
        temp_dy__ = d__ * y_coef                    # buffer for dy accumulation
        temp_dx__ = d__ * x_coef                    # buffer for dx accumulation
        # accumulate dy, dx:
        dy__[yd:-yd, xd:-xd] += temp_dy__.astype(int)   # bilateral accumulation on dy (x, y)
        dx__[yd:-yd, xd:-xd] += temp_dx__.astype(int)   # bilateral accumulation on dx (x, y)

        # top-right and bottom-left quadrants:

        d__ = p__[bi_yd:, :-bi_xd] - p__[:-bi_yd, bi_xd:]   # comparison between p (x + xd, y - yd) and p (x - xd, y + yd)
        # decompose d to dy, dx:
        temp_dy__ = d__ * y_coef                    # buffer for dy accumulation
        temp_dx__ = -(d__ * x_coef)                 # buffer for dx accumulation, sign inverted with comp direction
        # accumulate dy, dx:
        dy__[yd:-yd, xd:-xd] += temp_dy__.astype(int)   # bilateral accumulation on dy (x, y)
        dx__[yd:-yd, xd:-xd] += temp_dx__.astype(int)   # bilateral accumulation on dx (x, y)

    g__ = np.hypot(dy__, dx__) - ave * blob.ncomp  # compute g__

    # pack all derts into dert__

    dert__[:, :, 0] = p__
    dert__[:, :, 1] = dy__
    dert__[:, :, 2] = dx__
    dert__[:, :, 3] = g__

    blob.new_dert__[0] = dert__ # pack dert__ into blob

    return rng
    # ---------- comp_range() end ---------------------------------------------------------------------------------------