import numpy as np
from collections import deque
from frame_2D_alg import Classes
from frame_2D_alg.misc import get_filters
get_filters(globals()) # imports all filters at once
# --------------------------------------------------------------------------------

# ***************************************************** INC_DERIV FUNCTIONS *********************************************
# Functions:
# -inc_deriv()
# -comp_g()
# -cluster_dert()
# ***********************************************************************************************************************

def inc_deriv(blob):
    ''' same functionality as image_to_blobs() in frame_blobs.py'''

    global Y, X
    Y, X = blob.map.shape

    sub_blob = Classes.cl_frame(blob.dert__, map=blob.map, copy_dert=True)   # initialize sub_blob object per gblob

    comp_g(sub_blob.dert__, sub_blob.map)

    seg_ = deque()

    for y in range(Y - 1):

        P_ = cluster_dert(y, sub_blob)                    # cluster derts into g
        P_ = Classes.scan_P_(y, P_, seg_, sub_blob)     # P_ scans _P_ from seg_
        seg_ = Classes.form_segment(y, P_, sub_blob)    # form segments with P_ and their fork_s

    y = Y - 1   # sub_blob ends, merge segs of last line into their blobs:
    while seg_: Classes.form_blob(y, seg_.popleft(), sub_blob)

    sub_blob.terminate()  # delete sub_blob.dert__ and sub_blob.map
    blob.g_sub_blob = sub_blob

    return sub_blob
    # ---------- inc_deriv() end ----------------------------------------------------------------------------------------
def comp_g(dert__, map):
    " compare g within sub blob "

    g = dert__[:, :, 1]
    map[:-1] = np.logical_and(map[:-1], map[1:])
    map[:, :-1] = np.logical_and(map[:, :-1], map[:, 1:])

    dx = np.empty(g.shape)
    dy = np.empty(g.shape)
    gg = np.empty(g.shape)

    dx[:-1] = g[1:] - g[:-1]
    dy[:, :-1] = g[:, 1:] - g[:, :-1]

    g[map] = np.abs(dx[map]) + np.abs(dy[map]) - ave

    dert__[:, :, 0] = g
    dert__[:, :, 1] = gg
    dert__[:, :, 2] = dx
    dert__[:, :, 3] = dy
    # ---------- comp_g() end -------------------------------------------------------------------------------------------

def cluster_dert(y, sub_blob):
    " cluster derts into Ps "
    dert_ = sub_blob.dert__[y]
    P_map = sub_blob.map[y]

    P_ = deque()
    x = 0
    while x < X - 1:  # exclude last column
        while x < X - 1 and not P_map[x]:
            x += 1
        if x < X - 1 and P_map[x]:
            P = Classes.cl_P(x0=x, num_params=dert_.shape[1]+1)  # P initialization
            while x < X - 1 and P_map[x]:
                dert = dert_[x]
                gg = dert[1]
                s = gg > 0
                P = Classes.form_P(x, y, s, dert, P, P_)
                x += 1
            P.terminate(x, y)  # P' x_last
            P_.append(P)

    return  P_
    # ---------- cluster_dert() end -------------------------------------------------------------------------------------
