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

def unfold_blob(blob, comp_branch, rdn):  # unfold and compare

    y0, yn, x0, xn = blob.box
    y = y0  # current y, from seg y0 -> yn - 1
    i = 0   # segment index
    blob.seg_.sort(key=lambda seg: seg[0])  # sort by y0 coordinate
    seg_ = []   # buffer of segments containing line y
    # sseg_ = []  # buffer of sub-segments

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
        comp_branch.operation(P_)     # no dert_buff___ in hypot_g or future dx_g
        # sP_ = form_P_(dert__)       # horizontal clustering
        # sP_ = scan_P_(sP_, sseg_, blob)
        # sseg_ = form_sseg_(y, sP_, blob)

        y += 1

    # while dert_buff___:   # from sub blobs with remaining dert_s in dert_buff__
        # sP_ = form_P_(dert__)
        # sP_ = scan_P_(sP_, sseg_, blob)
        # sseg_ = form_seg_(y, sP_, blob)

    return comp_branch.dert___
    # ---------- unfold_blob() end ------------------------------------------------------------------------------------------

class hypot_g:  # here for testing only

    def __init__(self):
        self.dert___ = []

    def operation(self, P_):
        dert__ = []  # dert_ per P, dert__ per line, dert___ per blob

        for P in P_:
            x0 = P[1]
            dert_ = P[-1]
            for index, (p, dy, dx, g) in enumerate(dert_):
                g = hypot(dx, dy)

                dert_[index] = [(p, g, dy, dx, 4)]  # ncomp=4: multiple of min n, specified in deeper derts only
            dert__.append((x0, dert_))

        self.dert___.append(dert__)

        return self
    # ---------- hypot_g() end ----------------------------------------------------------------------------------------------


def form_sub_blobs(root_blob):  # redefine blob as branch-specific root blob: local equivalent of frame
    return
    # ---------- form_sub_blobs() end ---------------------------------------------------------------------------------------

def form_P_(*arg):  # cluster and sum horizontally consecutive pixels and their derivatives into Ps
    return 0
    # ---------- form_P_() end ------------------------------------------------------------------------------------------


def scan_P_(*arg):  # this function detects connections (forks) between Ps and _Ps, to form blob segments
    return 0
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------

def form_seg_(*arg):  # Convert or merge every P into segment. Merge blobs
    return 0
    # ---------- form_seg_() end --------------------------------------------------------------------------------------------


def form_blob(*arg):  # terminated segment is merged into continued or initialized blob (all connected segments)
    return 0
    # ---------- form_blob() end -------------------------------------------------------------------------------------