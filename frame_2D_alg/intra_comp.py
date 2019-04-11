import numpy as np
from math import hypot
from math import atan2
from collections import deque, namedtuple

nt_blob = namedtuple('blob', 'typ sign Y X Ly L Derts seg_ sub_blob_ layers_f sub_Derts map box add_dert rng ncomp')

# ************ FUNCTIONS ************************************************************************************************
# -form_sub_blob()
# -unfold_blob()
# -form_P_()
# -scan_P_()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************

''' this module is under revision '''

def form_sub_blob(root_blob, dert___):  # redefine blob as branch-specific master blob: local equivalent of frame

    seg_ = deque()

    while dert_:
        P_ = form_P_()                     # horizontal clustering
        P_ = scan_P_()       # vertical clustering
        seg_ = form_seg_()   # vertical clustering
    while seg_: form_blob()  # terminate last running segments

    # ---------- add_sub_blob() end -----------------------------------------------------------------------------------------

def unfold_blob(blob, branch_comp, rng=1):     # unfold and compare

    dert___ = []

    y0, yn, x0, xn = blob.box
    y = y0                      # iterating y (y0 -> yn - 1)
    i = 0                       # iterating segment index

    dert_buff___ = deque(maxlen=rng)        # buffer of incomplete derts

    while y < yn and i < len(blob.seg_):

        seg_ = []                           # buffer of segments containing line y

        while blob.seg_[i][0] == y:
            seg_.append(blob.seg_[i])

        P_ = []                             # buffer for Ps at line y
        for seg in seg_:
            if y < seg[0] + seg[1][0]:      # y < y0 + Ly (y within segment):

                P_.append(seg[2][y - seg[0]])   # append P at line y of seg

        for seg in seg_:
            if not y < seg[0] + seg[1][0]:  # y >= y0 + Ly (out of segment):

                seg_.remove(seg)

        # operations:

        branch_comp(P_, dert_buff___, dert___)

    while dert_buff__:   # add remaining dert_s in dert_buff__ into dert__
        dert__.append(dert_buff___.pop())

    form_sub_blob(blob, dert___)

    # ---------- unfold_blob() end ------------------------------------------------------------------------------------------

def form_P_():  # cluster and sum horizontally consecutive pixels and their derivatives into Ps
    return
    # ---------- form_P_() end ------------------------------------------------------------------------------------------


def scan_P_():  # this function detects connections (forks) between Ps and _Ps, to form blob segments
    return
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------

def form_seg_():  # Convert or merge every P into segment. Merge blobs
    return
    # ---------- form_seg_() end --------------------------------------------------------------------------------------------


def form_blob():  # terminated segment is merged into continued or initialized blob (all connected segments)
    return
    # ---------- form_blob() end -------------------------------------------------------------------------------------