import numpy.ma as ma
from math import hypot
from collections import deque
from frame_2D_alg import Classes
from frame_2D_alg.misc import get_filters
get_filters(globals()) # imports all filters at once
# --------------------------------------------------------------------------------

# ***************************************************** ANGLE BLOBS FUNCTIONS *******************************************
# Functions:
# -inc_range()
# -comp_g()
# ***********************************************************************************************************************

def inc_range(blob, rng):
    ''' same functionality as image_to_blobs() in frame_blobs.py '''

    global Y, X
    Y, X = blob.map.shape

    dert__ = Classes.init_dert__(0, blob.dert__)
    sub_blob = Classes.cl_frame(dert__, map=blob.map, copy_dert=True)
    seg_ = deque()

    comp_p(dert__, blob.map, rng // 2)  # comp_p over the whole sub-blob. use half rng for computation, for convenience

    for y in range(rng, Y - rng):
        P_ = cal_g(y, dert__[y], sub_blob.map[y], rng // 2)
        P_ = Classes.scan_P_(y, P_, seg_, sub_blob)
        seg_ = Classes.form_segment(y, P_, sub_blob)

    y = Y - rng
    while seg_: Classes.form_blob(y, seg_.popleft(), sub_blob)

    sub_blob.terminate()

    blob.r_sub_blob = sub_blob
    # ---------- inc_range() end ----------------------------------------------------------------------------------------

def comp_p(dert__, map, rng):
    " compare rng-spaced pixels per blob "
    p__ = dert__[:, :, 0]
    mask = ~map     # complemented blob.map is the mask of array

    dy__ = ma.zeros(p__.shape, dtype=int)   # initialize dy__ as masked array - basically just numpy arrays with masks, for selective computation
    dx__ = ma.zeros(p__.shape, dtype=int)   # initialize dx__ as masked array
    dy__.mask = dx__.mask = mask            # all operations performed on masked arrays ignore where mask == True.

    # pure vertical comp:
    d__ = p__[rng:] - p__[:-rng]    # comparison performed on p at coordinate (x, y) and p at coordinate (x, y + rng)
    dy__[rng:] += d__               # bilateral accumulation on dy at coordinate (x, y + rng)
    dy__[:-rng] += d__              # bilateral accumulation on dy at coordinate (x, y)

    # pure horizontal comp:
    d__ = p__[:, rng:] - p__[:, :-rng]  # comparison performed on p at coordinate (x, y) and p at coordinate (x + rng, y)
    dx__[:, rng:] += d__                # bilateral accumulation on dx at coordinate (x + rng, y)
    dx__[:, :-rng] += d__               # bilateral accumulation on dx at coordinate (x, y)

    # diagonal comps:
    for rx in range(1, rng):
        ry = rng - rx           # ry, rx: difference in position of comparants

        hyp = hypot(rx, ry)     # hyp = sqrt(rx^2 + ry^2)
        y_coef = ry / hyp       # y_coef = ry / sqrt(rx^2 + ry^2) - for decomposition of d in to dy
        x_coef = rx / hyp       # x_coef = rx / sqrt(rx^2 + ry^2) - for decomposition of d in to dx

        # top-left and bottom-right quadrant:

        d__ = p__[ry:, rx:] - p__[:-ry, :-rx]   # comparison performed on p at coordinate (x, y) and p at coordinate (x + rx, y + ry)

        # decompose every d into dy, dx:
        temp_dy__ = d__ * y_coef                # buffer for dy accumulation
        temp_dx__ = d__ * x_coef                # buffer for dx accumulation

        # accumulate dy, dx:
        dy__[ry:, rx:] += temp_dy__             # bilateral accumulation on dy at coordinate (x + rx, y + ry)
        dy__[:-ry, :-rx] += temp_dy__           # bilateral accumulation on dy at coordinate (x, y)

        dx__[ry:, rx:] += temp_dx__             # bilateral accumulation on dx at coordinate (x + rx, y + ry)
        dx__[:-ry, :-rx] += temp_dx__           # bilateral accumulation on dx at coordinate (x, y)

        # top-right and bottom-left quadrant:

        d__ = p__[ry:, :-rx] - p__[:-ry, rx:]   # comparison performed on p at coordinate (x + rx, y) and p at coordinate (x, y + ry)

        # decompose every d into dy, dx:
        temp_dy__ = d__ * y_coef                # buffer for dy accumulation
        temp_dx__ = -(d__ * x_coef)             # buffer for dx accumulation. inverted sign because of inverted x comp direction

        # accumulate dy, dx:
        dy__[ry:, :-rx] += temp_dy__            # bilateral accumulation on dy at coordinate (x, y + ry)
        dy__[:-ry, rx:] += temp_dy__            # bilateral accumulation on dy at coordinate (x + rx, y)

        dx__[ry:, :-rx] += temp_dx__            # bilateral accumulation on dx at coordinate (x, y + ry)
        dx__[:-ry, rx:] += temp_dx__            # bilateral accumulation on dx at coordinate (x + rx, y)

    dert__[:, :, 2] += dx__  # add the result to the result of previous rng comps
    dert__[:, :, 3] += dy__  # add the result to the result of previous rng comps
    # ---------- comp_p() end -------------------------------------------------------------------------------------------

def cal_g(y, dert_, P_map, rng):
    " use dx, dy to compute g and form Ps "
    P_ = deque()
    x = rng             # discard first rng column
    while x < X - rng:  # discard last rng column
        while x < X - rng and not P_map[x]:
            x += 1
        if x < X - rng and P_map[x]:
            P = Classes.cl_P(x0=x, num_params=dert_.shape[1])  # P initialization
            while x < X - rng and P_map[x]:
                dert = dert_[x]
                dx, dy = dert[2:4]
                g = hypot(dx, dy) - ave         # ?
                dert[1] = g
                s = g > 0
                P = Classes.form_P(x, y, s, dert, P, P_)
                x += 1
            P.terminate(x, y)   # P' x_last
            P_.append(P)

    return P_
    # ---------- cal_g() end --------------------------------------------------------------------------------------------
