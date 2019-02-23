import numpy as np
from collections import deque
from frame_2D_alg import Classes
from frame_2D_alg.misc import get_filters
get_filters(globals()) # imports all filters at once
# --------------------------------------------------------------------------------
'''
    comp_inc_deriv is a component of intra_blob
'''
# ***************************************************** ANGLE BLOBS FUNCTIONS *******************************************
# Functions:
# -inc_deriv()
# -comp_g()
# ***********************************************************************************************************************

def inc_deriv(blob):  # compare gradient to define gg_blobs
    ''' same functionality as image_to_blobs() in frame_blobs.py '''

    global Y, X
    Y, X = blob.map.shape

    dert__ = Classes.init_dert__(3, blob.dert__)    # 3 params added: gg, dxg, dyg
    blob = Classes.cl_frame(dert__, map=blob.map)  # initialize blob object per ablob
    seg_ = deque()
    dert_ = dert__[0]
    P_map = blob.map[0]

    for y in range(Y - 1):
        lower_dert_ = dert__[y + 1]
        lower_P_map = blob.map[y + 1]

        P_ = comp_g(y, dert_, lower_dert_, P_map, lower_P_map) # vertical and lateral g comparison
        P_ = Classes.scan_P_(y, P_, seg_, blob)   # P_ scans _P_ from seg_
        seg_ = Classes.form_segment(y, P_, blob)  # form segments with P_ and their fork_s
        dert_, P_map = lower_dert_, lower_P_map    # buffers for next line

    y = Y - 1   # frame ends, merge segs of last line into their blobs:
    while seg_: Classes.form_blob(y, seg_.popleft(), blob)

    blob.terminate()  # delete frame.dert__ and frame.map
    blob.gblob = blob
    # ---------- inc_deriv() end ----------------------------------------------------------------------------------------

def comp_g(y, dert_, lower_dert_, P_map, lower_P_map):
    " compare gradient adjacent derts within frame per ablob "

    comp_map = np.logical_and(P_map, lower_P_map)
    comp_map = np.logical_and(comp_map[:-1], comp_map[1:])

    g_ = dert_[:-1, 1]  # assigned manually for now. Will change in the future when dert syntax is consistent
    right_g_ = dert_[1:, 1]
    lower_g_ = lower_dert_[:-1, 1]

    dxg_ = np.zeros(g_.shape, dtype=int)
    dxg_[comp_map] = right_g_[comp_map] - g_[comp_map]
    dyg_ = np.zeros(g_.shape, dtype=int)
    dyg_[comp_map] = lower_g_[comp_map] - g_[comp_map]

    gg_ = np.hypot(dyg_, dxg_) - ave

    dert_[:-1, -3] = gg_  # assign gg_ to a slice of dert_
    dert_[:-1, -2] = dxg_ # assign dxg_ to a slice of dert_
    dert_[:-1, -1] = dyg_ # assign dyg_ to a slice of dert_

    P_ = deque()
    x = 0
    while x < X - 1:  # exclude last column
        while x < X - 1 and not comp_map[x]:
            x += 1
        if x < X - 1 and comp_map[x]:
            P = Classes.cl_P(x0=x, num_params=dert_.shape[1]+1)  # P initialization
            while x < X - 1 and comp_map[x]:
                dert = dert_[x]
                gg = dert[-3]
                s = gg > 0
                P = Classes.form_P(x, y, s, dert, P, P_)
                x += 1
            P.terminate(x, y)  # P' x_last
            P_.append(P)

    return  P_
    # ---------- comp_g() end -------------------------------------------------------------------------------------------