from math import hypot
from collections import deque, namedtuple

nt_blob = namedtuple('blob', 'typ sign Ly L Derts seg_ root_blob sub_blob_ sub_Derts layer_f map box rng')

# ************ FUNCTIONS ************************************************************************************************
# -unfold_blob()
# -form_sub_blobs()
# -form_P_()
# -scan_P_()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************

''' 
unfold blob into derts, 
perform branch-specific comparison, 
convert blob into root_blob with new sub_blob_ 
'''

def unfold_blob(blob, comp_branch, rdn, rng=1):  # unfold and compare

    y0, yn, x0, xn = blob.box
    y = y0  # current y, from seg y0 -> yn - 1
    i = 0   # segment index
    blob.seg_.sort(key=lambda seg: seg[0])  # sort by y0 coordinate
    dert___ = []   # complete derts
    dert_buff___ = deque(maxlen=rng)  # incomplete derts
    seg_ = []    # buffer of segments containing line y

    while y < yn and i < len(blob.seg_):

        while i < len(blob.seg_) and blob.seg_[i][0] == y:
            seg_.append(blob.seg_[i])
            i += 1
        P_ = []  # line y Ps
        for seg in seg_:
            if y < seg[0] + seg[1][0]:          # y < y0 + Ly within segment, or len(Py)?
                P_.append(seg[2][y - seg[0]])   # append P at line y of seg

        for seg in seg_:
            if not y < seg[0] + seg[1][0]:      # y >= y0 + Ly (out of segment):
                seg_.remove(seg)

        P_.sort(key=lambda P: P[1])  # sort by x0 coordinate
        # operations:
        comp_branch(P_, dert___, dert_buff___)  # no dert_buff___ in hypot_g or future dx_g
        y += 1

    while dert_buff___:   # add remaining dert_s in dert_buff__ into dert___
        dert___.append(dert_buff___.pop())

    return dert___

    # ---------- unfold_blob() end ------------------------------------------------------------------------------------------

def hypot_g(P_, dert___):  # here for testing only
    dert__ = []  # dert_ per P, dert__ per line, dert___ per blob

    for P in P_:
        x0 = P[1]
        dert_ = P[-1]
        for index, (i, dy, dx, g) in enumerate(dert_):
            g = hypot(dx, dy)

            dert_[index] = [(i, dy, dx, g)]  # ncomp=1: multiple of min n, specified in deeper derts only
        dert__.append((x0, dert_))
    dert___.append(dert__)

    # ---------- hypot_g() end ----------------------------------------------------------------------------------------------


def form_sub_blobs(root_blob):  # redefine blob as branch-specific root blob: local equivalent of frame
    return
    # ---------- form_sub_blobs() end ---------------------------------------------------------------------------------------

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