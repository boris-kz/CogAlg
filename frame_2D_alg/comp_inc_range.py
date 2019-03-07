import numpy.ma as ma
from math import hypot
from collections import deque
from frame_2D_alg import generic
from frame_2D_alg.filters import get_filters
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
    rng = blob.rng + 1
    ncomp = blob.ncomp + rng
    sub_blob = [0, 0, 0, 0, [], rng, ncomp]

    comp_p(blob.dert__, blob.map, rng)  # comp_p over the whole sub-blob, rng measure is unilateral
    seg_ = deque()

    for y in range(rng, height - rng):
        P_ = generic.form_P_(y, sub_blob)  # horizontal clustering
        P_ = generic.scan_P_(P_, seg_, sub_blob)
        seg_ = generic.form_seg_(P_, sub_blob)

    while seg_: generic.form_blob(seg_.popleft(), sub_blob)

    blob.rng_sub_blob = sub_blob
    return sub_blob

    # ---------- inc_range() end ----------------------------------------------------------------------------------------

def comp_p(dert__, map, rng):   # compare rng-distant pixels within blob

    p__ = dert__[:, :, 0]
    mask = ~map     # complemented blob.map is a mask of array
    dy__ = ma.zeros(map.shape, dtype=int)   # initialize dy__ as array masked for selective computation
    dx__ = ma.zeros(map.shape, dtype=int)
    dy__.mask = dx__.mask = mask    # all operations on masked arrays ignore elements at mask == True.

    # vertical comp:
    d__ = p__[rng:] - p__[:-rng]    # comparison between p at coordinates (x, y) and p at coordinates (x, y+ rng)
    dy__[rng:] += d__               # bilateral accumulation on dy (x, y+ rng)
    dy__[:-rng] += d__              # bilateral accumulation on dy (x, y)

    # horizontal comp:
    d__ = p__[:, rng:] - p__[:, :-rng]  # comparison between p (x, y) and p (x + rng, y)
    dx__[:, rng:] += d__                # bilateral accumulation on dx (x + rng, y)
    dx__[:, :-rng] += d__               # bilateral accumulation on dx (x, y)

    # diagonal comparison:

    for xd in range(1, rng):
        yd = rng - xd           # y and x distance between comparands
        hyp = hypot(xd, yd)
        y_coef = yd / hyp       # to decompose d into dy
        x_coef = xd / hyp       # to decompose d into dx

        # top-left and bottom-right quadrants:

        d__ = p__[yd:, xd:] - p__[:-yd, :-xd]   # comparison between p (x, y) and p (x + xd, y + yd)
        # decompose d to dy, dx:
        temp_dy__ = d__ * y_coef                # buffer for dy accumulation
        temp_dx__ = d__ * x_coef                # buffer for dx accumulation
        # accumulate dy, dx:
        dy__[yd:, xd:] += temp_dy__.astype(int)     # bilateral accumulation on dy (x + xd, y + yd)
        dy__[:-yd, :-xd] += temp_dy__.astype(int)   # bilateral accumulation on dy (x, y)
        dx__[yd:, xd:] += temp_dx__.astype(int)     # bilateral accumulation on dx (x + xd, y + yd)
        dx__[:-yd, :-xd] += temp_dx__.astype(int)   # bilateral accumulation on dx (x, y)

        # top-right and bottom-left quadrants:

        d__ = p__[yd:, :-xd] - p__[:-yd, xd:]   # comparison between p (x + xd, y) and p (x, y + yd)
        # decompose d to dy, dx:
        temp_dy__ = d__ * y_coef                # buffer for dy accumulation
        temp_dx__ = -(d__ * x_coef)             # buffer for dx accumulation, sign inverted with comp direction
        # accumulate dy, dx:
        dy__[yd:, :-xd] += temp_dy__.astype(int)    # bilateral accumulation on dy (x, y + yd)
        dy__[:-yd, xd:] += temp_dy__.astype(int)    # bilateral accumulation on dy (x + xd, y)
        dx__[yd:, :-xd] += temp_dx__.astype(int)    # bilateral accumulation on dx (x, y + yd)
        dx__[:-yd, xd:] += temp_dx__.astype(int)    # bilateral accumulation on dx (x + xd, y)

    dert__[:, :, 2] += dx__  # add dx to shorter-rng-accumulated dx
    dert__[:, :, 3] += dy__  # add dy to shorter-rng-accumulated dy
    # ---------- comp_p() end -------------------------------------------------------------------------------------------