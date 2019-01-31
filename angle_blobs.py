import math
import numpy as np
from collections import deque
import Classes
from frame_blobs import form_P
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
# -compute_angle()
# -compare_angle()
# -form_aP()
# -scan_aP_()
# -form_asegment()
# -form_ablob()
# ***********************************************************************************************************************

def blob_to_ablobs(blob):  # compute and compare angle, define ablobs, accumulate a, da, sda in all reps within gblob
    ''' same functionality as image_to_blobs() in frame_blobs.py'''
    # No need to make new dert_map of a or sda because there's no further recursion in this branch:
    frame = Classes.frame(9)  # 2 more parameters: A and sDa
    p__, g__, d__ = frame.dert_map_ = blob.dert_map_
    frame.blob_map = blob.blob_map

    _P_ = deque()
    global y, Y, X
    Y, X = frame.dert_map_[0].shape
    # compute init a_:
    d_ = d__[0]
    b_ = frame.blob_map[0]
    a_ = compute_angle(d_, b_)
    for y in range(Y - 1):
        a_, d_, b_, _P_ = compare_angle(a_, d_, b_, _P_, frame)  # vertical and lateral pixel comparison

    # frame ends, merge segs of last line into their blobs:
    y = Y - 1
    while _P_:  form_blob(form_segment(_P_.popleft(), frame), frame)
    del frame.dert_map_
    del frame.blob_map

    blob.frame_of_ablob = frame
    return frame
    # ---------- blob_to_ablobs() end -----------------------------------------------------------------------------------
def compute_angle(d_, l_b_, b_=False):
    " selectively compute angle of maximal gradient "
    a_ = np.full(l_b_.shape, -1)
    s_ = np.zeros(l_b_.shape, dtype=bool)  # selective array
    s_[:1] = np.bitwise_or(np.bitwise_or(l_b_[:1], l_b_[1:]), b_)     # includes pixels right-down (outside) of the blob (for comparing)
    a_[s_] = np.arctan2(d_[s_, 0], d_[s_, 1]) * angle_coef + 128    # element-wise atan2, compute only where no_angle == True
    return a_
    # ---------- compute_angle() end ------------------------------------------------------------------------------------

def compare_angle(a_, d_, b_, _P_, frame):
    " compute and compare angle of adjacent pixels "

    p__, g__, d__ = frame.dert_map_
    b__ = frame.blob_map
    p_, g_, l_d_, l_b_ = p__[y], g__[y], d__[y+1], b__[y+1]
    l_a_ = compute_angle(l_d_, l_b, b)
    sda_ = np.abs(a_[1:] - a_[:-1]) + np.abs(l_a_[:-1] - a_[:-1])
    dert_ = zip(p_[:-1], g_[:-1], d_[:-1, 1], d_[:-1, 0], a_[:-1], sda_)    # p, g, dx, dy, s, sda

    P_ = deque()
    buff_ = deque()
    x = 0
    while x < X - 1:
        while not b_[x]:
            x += 1
        aP = Classes.P(y, x)    # init aP
        while b_[x]:
            dert = dert_[x]
            sda = dert[-1]
            s = sda > 0
            aP = P = form_P(s, dert, x, aP, P_, buff_, _P_, frame)
            x += 1
        P.terminate(x)  # P's x_end
        scan_P_(P, P_, buff_, _P_, frame)  # P scans hP_
    while buff_:
        seg = buff_.popleft()
        if seg.roots != 1:
            form_blob(seg, frame)
    while _P_:
        form_blob(form_segment(_P_.popleft(), frame), frame)
    return l_a, l_d, l_b, P_
    # ---------- compare_angle() end ------------------------------------------------------------------------------------