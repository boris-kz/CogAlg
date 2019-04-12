from math import hypot
from collections import deque, namedtuple

nt_blob = namedtuple('blob', 'typ sign Ly L Derts seg_ root_blob sub_blob_ sub_Derts layer_f map box rng')

# ************ FUNCTIONS ************************************************************************************************
# -form_sub_blob()
# -unfold_blob()
# -hypot_g()
# -form_P_()
# -scan_P_()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************

''' this module is under revision '''

def form_sub_blobs(root_blob):  # redefine blob as branch-specific master blob: local equivalent of frame
    return
    # ---------- form_sub_blobs() end ---------------------------------------------------------------------------------------

def unfold_blob(blob, comp_branch, rdn, rng=1):     # unfold and compare

    dert___ = []

    y0, yn, x0, xn = blob.box
    y = y0                      # iterating y (y0 -> yn - 1)
    i = 0                       # iterating segment index

    blob.seg_.sort(key=lambda seg: seg[0])

    dert_buff___ = deque(maxlen=rng)            # buffer of incomplete derts

    seg_ = []  # buffer of segments containing line y

    while y < yn and i < len(blob.seg_):

        while i < len(blob.seg_) and blob.seg_[i][0] == y:
            seg_.append(blob.seg_[i])
            i += 1

        P_ = []                                 # buffer for Ps at line y
        for seg in seg_:
            if y < seg[0] + seg[1][0]:          # y < y0 + Ly (y within segment):

                P_.append(seg[2][y - seg[0]])   # append P at line y of seg

        for seg in seg_:
            if not y < seg[0] + seg[1][0]:      # y >= y0 + Ly (out of segment):

                seg_.remove(seg)

        P_.sort(key=lambda P: P[1]) # sort by x coordinate, left -> right

        # operation:

        comp_branch(P_, dert_buff___, dert___)

        y += 1

    while dert_buff___:   # add remaining dert_s in dert_buff__ into dert__
        dert___.append(dert_buff___.pop())

    return dert___

    # ---------- unfold_blob() end ------------------------------------------------------------------------------------------

def hypot_g(P_, dert_buff___, dert___):

    dert__ = []     # initial line dert__
    for P in P_:
        x0 = P[1]
        dert_ = P[-1]
        for index, (i, dy, dx, g) in enumerate(dert_):
            g = hypot(dx, dy)

            dert_[index] = [i, (4, dy, dx, g)]      # [i, (ncomp, dy, dx, g)]

        dert__.append((x0, dert_))

    dert___.append((dert__))
    # ---------- hypot_g() end ----------------------------------------------------------------------------------------------

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