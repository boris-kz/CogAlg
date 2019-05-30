import numpy.ma as ma
from math import hypot
from collections import deque
from old.frame_2D_alg_classes import Classes
from old.frame_2D_alg_classes.misc import get_filters
get_filters(globals())  # imports all filters at once
# --------------------------------------------------------------------------------
'''
    inc_range is a component of intra_blob
'''
# ***************************************************** INC_RANGE FUNCTIONS *********************************************
# Functions:
# -bilateral()
# -inc_range()
# -comp_p()
# -calc_g()
# ***********************************************************************************************************************

def bilateral(blob):
    ''' reversed-direction image_to_blobs() in frame_blobs.py '''

    global Y, X
    Y, X = blob.map.shape
    sub_blob = Classes.cl_frame(blob.dert__, map=blob.map, copy_dert=True)
    seg_ = deque()

    dert__ = sub_blob.dert__
    dert__[:, 1:, 2] += dert__[:, 1:, 0] - dert__[:, :-1, 0]    # compare left pixel, accumulate to dx__
    dert__[1:, :, 3] += dert__[1:, :, 0] - dert__[:-1, :, 0]    # compare higher pixel, accumulate to dy__
    # or:
    # p__ = dert__[:, :, 0]
    # dy__ = dert__[:, :, 3]
    # dx__ = dert__[:, :, 2]
    # dy__[1:] += p__[1:] - p__[:-1]            # compare higher pixel
    # dx__[:, 1:] += p__[:, 1:] - p__[:, -1]    # compare left pixel

    for y in range(1, Y):                       # discard first incomplete row
        P_ = calc_g(y, sub_blob)
        P_ = Classes.scan_P_(y, P_, seg_, sub_blob)
        seg_ = Classes.form_segment(y, P_, sub_blob)

    y = Y
    while seg_: Classes.form_blob(y, seg_.popleft(), sub_blob)

    sub_blob.terminate()
    blob.rng_sub_blob = sub_blob
    return sub_blob  # ncomp = 2
    # ---------- bilateral() end ----------------------------------------------------------------------------------------

def inc_range(blob):
    ''' same functionality as image_to_blobs() in frame_blobs.py
        with rng > 1 '''

    global Y, X
    Y, X = blob.map.shape

    rng = blob.rng + 1
    ncomp = blob.ncomp + rng * 2
    sub_blob = Classes.cl_frame(blob.dert__, map=blob.map, rng=rng, ncomp=ncomp, copy_dert=True)
    seg_ = deque()

    comp_p(sub_blob.dert__, sub_blob.map, rng)  # comp_p over the whole sub-blob, rng measure is unilateral

    for y in range(rng, Y - rng):
        P_ = calc_g(y, sub_blob)
        P_ = Classes.scan_P_(y, P_, seg_, sub_blob)
        seg_ = Classes.form_segment(y, P_, sub_blob)

    y = Y - rng
    while seg_: Classes.form_blob(y, seg_.popleft(), sub_blob)

    sub_blob.terminate()
    blob.rng_sub_blob = sub_blob

    return sub_blob
    # ---------- inc_range() end ----------------------------------------------------------------------------------------

def comp_p(dert__, map, rng):
    " compare rng-distant pixels within blob "
    p__ = dert__[:, :, 0]
    mask = ~map     # complemented blob.map is a mask of array

    dy__ = ma.zeros((Y, X), dtype=int)   # initialize dy__ as array masked for selective computation
    dx__ = ma.zeros((Y, X), dtype=int)
    dy__.mask = dx__.mask = mask    # all operations on masked arrays ignore elements at mask == True.

    # vertical comp:
    d__ = p__[rng:] - p__[:-rng]    # comparison between p at coordinates (x, y) and p at coordinates (x, y+ rng)
    dy__[rng:] += d__               # bilateral accumulation on dy (x, y+ rng)
    dy__[:-rng] += d__              # bilateral accumulation on dy (x, y)

    # horizontal comp:
    d__ = p__[:, rng:] - p__[:, :-rng]  # comparison between p (x, y) and p (x + rng, y)
    dx__[:, rng:] += d__                # bilateral accumulation on dx (x + rng, y)
    dx__[:, :-rng] += d__               # bilateral accumulation on dx (x, y)

    # diagonal comps:

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

def calc_g(y, sub_blob):
    " compute g from dx, dy; form Ps "

    dert_ = sub_blob.dert__[y]
    P_map = sub_blob.map[y]
    rng = sub_blob.rng
    ncomp = sub_blob.ncomp

    P_ = deque()
    x = rng                 # discard first rng columns
    if rng > 1:
        x_stop = X - rng    # discard last rng columns
    else:
        x_stop = X          # if rng = 1, no right comp, no need to discard last column
    while x < x_stop:
        while x < x_stop and not P_map[x]:
            x += 1
        if x < x_stop and P_map[x]:
            P = Classes.cl_P(x0=x, num_params=dert_.shape[1])  # P initialization
            while x < x_stop and P_map[x]:
                dert = dert_[x]
                dx, dy = dert[2:4]
                g = abs(dx) + abs(dy) - ave * ncomp
                dert[1] = g
                s = g > 0
                P = Classes.form_P(x, y, s, dert, P, P_)
                x += 1
            P.terminate(x, y)   # P' x_last
            P_.append(P)

    return P_
    # ---------- calc_g() end -------------------------------------------------------------------------------------------