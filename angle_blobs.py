import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import Classes
from frame_blobs import form_P
from frame_blobs import scan_P_
from frame_blobs import form_segment
from frame_blobs import form_blob
# Filters ------------------------------------------------------------------------
from misc import get_filters
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
# ***********************************************************************************************************************

def blob_to_ablobs(blob):  # compute and compare angle, define ablobs, accumulate a, da, sda in all reps within gblob
    ''' same functionality as image_to_blobs() in frame_blobs.py'''

    frame = Classes.frame(blob.dert__, blob_map=blob.blob_map, num_params=9)
    # initialize frame object: initialize blob_ and params, assign dert__ and blob_map, assign frame shape
    _P_ = deque()
    global y, Y, X
    Y, X = frame.dert__.shape
    a_ = get_angle(frame.dert__[0], frame.blob_map[0])  # compute max gradient angles within gblob
    for y in range(Y - 1):
        a_, _P_ = comp_angle(a_, _P_, frame)  # vertical and lateral pixel comparison

    y = Y - 1   # frame ends, merge segs of last line into their blobs:
    while _P_:  form_blob(form_segment(_P_.popleft(), frame), frame)

    frame.terminate()  # delete frame.dert__ and frame.blob_map
    blob.frame_ablobs = frame
    return frame
    # ---------- blob_to_ablobs() end -----------------------------------------------------------------------------------

def get_angle(dert_, P_map_, _P_map_ = False):  # default = False: no higher-line for first line
    " compute angle of gradient in and adjacent to selected gblob"
    a_ = np.full(P_map_.shape, -1)

    marg_angle_ = np.zeros(P_map_.shape, dtype=bool)           # to compute angle in blob-marginal derts
    marg_angle_[:-1] = np.logical_or(P_map_[:-1], P_map_[1:])  # derts right-adjacent to blob, for lower-line lateral comp
    marg_angle_ = np.logical_or(marg_angle_, _P_map_)          # derts down-adjacent to blob, for higher-line vertical comp

    dx_, dy_ = np.array([[dx, dy] for p, g, dx, dy in dert_]).T  # construct dx, dy array

    a_[marg_angle_] = np.arctan2(dy_[marg_angle_], dx_[marg_angle_]) * angle_coef + 128  # computes angle if marg_angle_== True
    return a_
    # ---------- compute_angle() end ------------------------------------------------------------------------------------

def comp_angle(a_, _P_, frame):
    " compare angle of adjacent pixels within frame == gblob "

    dert_, lower_dert_ = frame.dert__[y:y+2]
    P_map_, lower_P_map_ = frame.blob_map[y:y+2]

    lower_a_ = get_angle(lower_dert_, P_map_, lower_P_map_)
    sda_ = np.abs(a_[1:] - a_[:-1]) + np.abs(lower_a_[:-1] - a_[:-1]) - 2 * ave

    P_ = deque()
    buff_ = deque()
    x = 0
    while x < X - 1:  # excludes last column
        while x < X - 1 and not P_map_[x]:
            x += 1
        if x < X - 1 and P_map_[x]:
            aP = Classes.P(y, x_start=x, num_params=7)    # aP initialization
            while x < X - 1 and P_map_[x]:
                a = a_[x]
                sda = sda_[x]
                dert = dert_[x] + [a, sda]
                s = sda > 0
                aP = form_P(s, dert, x, aP, P_, buff_, _P_, frame)
                x += 1
            aP.terminate(x)  # aP' x_end
            scan_P_(aP, P_, buff_, _P_, frame)  # P scans hP_, constructing asegs and ablobs
            
    while buff_:
        seg = buff_.popleft()
        if seg.roots != 1:
            form_blob(seg, frame)
    while _P_:
        form_blob(form_segment(_P_.popleft(), frame), frame)
    return lower_a_, P_
    # ---------- compare_angle() end ------------------------------------------------------------------------------------
