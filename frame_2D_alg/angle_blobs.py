import numpy as np
from collections import deque
from frame_2D_alg import Classes
from frame_2D_alg.misc import get_filters
get_filters(globals()) # imports all filters at once
# --------------------------------------------------------------------------------
'''
    angle_blob is a component of intra_blob
'''
# ***************************************************** ANGLE BLOBS FUNCTIONS *******************************************
# Functions:
# -blob_to_ablobs()
# -get_angle()
# -comp_angle()
# ***********************************************************************************************************************

def blob_to_ablobs(blob):  # compute and compare angle, define ablobs, accumulate a, da, sda in all reps within gblob
    ''' same functionality as image_to_blobs() in frame_blobs.py'''

    global Y, X
    Y, X = blob.map.shape

    dert__ = Classes.init_dert__(2, blob.dert__)
    frame = Classes.cl_frame(dert__, map=blob.map)  # initialize frame object per gblob
    seg_ = deque()
    dert_ = dert__[0]
    P_map_ = frame.map[0]
    a_ = get_angle(dert_, P_map_)  # compute angle of max gradients within gblob (contiguous area of same-sign gradient)

    for y in range(Y - 1):
        lower_dert_ = dert__[y + 1]
        lower_P_map_ = frame.map[y + 1]
        lower_a_ = get_angle(lower_dert_, lower_P_map_, P_map_)

        P_ = comp_angle(y, a_, lower_a_, dert_, P_map_) # vertical and lateral angle comparison
        P_ = Classes.scan_P_(y, P_, seg_, frame)        # aP_ scans _aP_ from seg_
        seg_ = Classes.form_segment(y, P_, frame)       # form segments with P_ and their fork_s
        a_, dert_, P_map_ = lower_a_, lower_dert_, lower_P_map_  # buffers for next line

    y = Y - 1   # frame ends, merge segs of last line into their blobs:
    while seg_: Classes.form_blob(y, seg_.popleft(), frame)

    frame.terminate()  # delete frame.dert__ and frame.map
    blob.frame_ablobs = frame
    return frame
    # ---------- blob_to_ablobs() end -----------------------------------------------------------------------------------

def get_angle(dert_, P_map_, _P_map_ = False):  # default = False: no higher-line for first line
    " compute angle of gradient in and adjacent to selected gblob"

    a_ = np.full(P_map_.shape, -1)
    marg_angle_ = np.zeros(P_map_.shape, dtype=bool)            # to compute angle in blob-marginal derts
    marg_angle_[0] = P_map_[0]
    marg_angle_[1:] = np.logical_or(P_map_[:-1], P_map_[1:])    # derts right-adjacent to blob, for lower-line lateral comp
    marg_angle_ = np.logical_or(marg_angle_, _P_map_)           # derts down-adjacent to blob, for higher-line vertical comp
    dx_, dy_ = dert_ [:, 2:4].T                                 # dx, dy as slices of dert_

    a_[marg_angle_] = np.arctan2(dy_[marg_angle_], dx_[marg_angle_]) * angle_coef + 128  # computes angle if marg_angle_== True
    return a_
    # ---------- compute_angle() end ------------------------------------------------------------------------------------

def comp_angle(y, a_, lower_a_, dert_, P_map_):
    " compare angle of adjacent gradients within frame per gblob "

    sda_ = np.abs(a_[1:] - a_[:-1]) + np.abs(lower_a_[:-1] - a_[:-1]) - 2 * ave # calculate sda_
    dert_[:, 4] = a_        # assign a_ to a slice of dert_
    dert_[:-1, 5] = sda_    # assign sda_ to a slice of dert_
    P_ = deque()
    x = 0
    while x < X - 1:  # exclude last column
        while x < X - 1 and not P_map_[x]:
            x += 1
        if x < X - 1 and P_map_[x]:
            P = Classes.cl_P(x0=x, num_params=dert_.shape[1]+1)  # aP initialization
            while x < X - 1 and P_map_[x]:
                dert = dert_[x]
                sda = dert[5]
                s = sda > 0
                P = Classes.form_P(x, y, s, dert, P, P_)
                x += 1
            P.terminate(x, y)  # aP' x_last
            P_.append(P)

    return  P_
    # ---------- compare_angle() end ------------------------------------------------------------------------------------
