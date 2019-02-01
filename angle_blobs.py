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

    frame = Classes.frame(blob.dert__, blob_map=blob.blob_map, num_params=9)    # initialize frame object: initialize blob_ and params, assign dert__ and blob_map, assign frame shape
    _P_ = deque()
    global y, Y, X
    Y, X = frame.dert__.shape
    # compute init a_:
    a_ = get_angle(frame.dert__[0], frame.blob_map[0])
    for y in range(Y - 1):
        a_, _P_ = comp_angle(a_, _P_, frame)  # vertical and lateral pixel comparison

    # frame ends, merge segs of last line into their blobs:
    y = Y - 1
    while _P_:  form_blob(form_segment(_P_.popleft(), frame), frame)

    frame.terminate()   # delete frame.dert__ and frame.blob_map

    blob.frame_of_ablob = frame
    return frame
    # ---------- blob_to_ablobs() end -----------------------------------------------------------------------------------
def get_angle(dert_, P_map_, _P_map_ = False):  # _P_map_ default value = False for first line: no higher-line
    " selectively compute angle of maximal gradient "

    a_ = np.full(P_map_.shape, -1)

    angle_cal_ = np.zeros(P_map_.shape, dtype=bool)  # for computing angle selectively
    angle_cal_[:-1] = np.logical_or(P_map_[:-1], P_map_[1:])    # includes pixels on the right of pixels that are inside the blob for lower-line lateral comp
    angle_cal_ = np.logical_or(angle_cal_, _P_map_)             # includes pixels below higher-pixels that are inside the blob for higher-line vertical comp

    dx_, dy_ = np.array([[dx, dy] for p, g, dx, dy in dert_]).T # construct dx, dy array

    a_[angle_cal_] = np.arctan2(dy_[angle_cal_], dx_[angle_cal_]) * angle_coef + 128    # only calculate at position where angle_cal == True
    return a_
    # ---------- compute_angle() end ------------------------------------------------------------------------------------

def comp_angle(a_, _P_, frame):
    " compute and compare angle of adjacent pixels "

    dert_, lower_dert_ = frame.dert__[y:y+2]
    P_map_, lower_P_map_ = frame.blob_map[y:y+2]

    lower_a_ = get_angle(lower_dert_, P_map_, lower_P_map_)
    sda_ = np.abs(a_[1:] - a_[:-1]) + np.abs(lower_a_[:-1] - a_[:-1]) - 2 * ave

    P_ = deque()
    buff_ = deque()
    x = 0
    while x < X - 1:    # exludes last column
        while x < X - 1 and not P_map_[x]:
            x += 1
        if x < X - 1 and P_map_[x]:
            aP = Classes.P(y, x_start=x, num_params=7)    # init aP
            while x < X - 1 and P_map_[x]:
                a = a_[x]
                sda = sda_[x]
                dert = dert_[x] + [a, sda]
                s = sda > 0
                aP = form_P(s, dert, x, aP, P_, buff_, _P_, frame)
                x += 1
            aP.terminate(x)  # P's x_end
            scan_P_(aP, P_, buff_, _P_, frame)  # P scans hP_
    while buff_:
        seg = buff_.popleft()
        if seg.roots != 1:
            form_blob(seg, frame)
    while _P_:
        form_blob(form_segment(_P_.popleft(), frame), frame)
    return lower_a_, P_
    # ---------- compare_angle() end ------------------------------------------------------------------------------------