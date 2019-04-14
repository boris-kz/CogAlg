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
    buff___ = deque(maxlen=rng)
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
        sP_ = comp_branch(P_, dert___, buff___)     # no dert_buff___ in hypot_g or future dx_g
        # sP_ = scan_P_(sP_, sseg_, blob)
        # sseg_ = form_sseg_(y, sP_, blob)

        y += 1

    # while dert_buff___:   # from sub blobs with remaining dert_s in dert_buff__
        # sP_ = form_P_(dert__)
        # sP_ = scan_P_(sP_, sseg_, blob)
        # sseg_ = form_seg_(y, sP_, blob)
    # ---------- unfold_blob() end ------------------------------------------------------------------------------------------

def hypot_g(P_, buff___):  # here for testing only

    sP_ = []

    for P in P_:

        x0 = P[1]
        derts_ = P[-1]

        for index, (p, dy, dx, g) in enumerate(derts_):
            g = hypot(dx, dy)

            derts_[index] = [(p, 4, dy, dx, g)]  # ncomp=4: multiple of min n, specified in deeper derts only

        sP_ += form_P_(derts_, x0, derts_index=0)

    return sP_
    # ---------- hypot_g() end ----------------------------------------------------------------------------------------------

def form_sub_blobs(root_blob):  # redefine blob as branch-specific root blob: local equivalent of frame
    return
    # ---------- form_sub_blobs() end ---------------------------------------------------------------------------------------

def form_P_(derts__, x_start, derts_index):  # horizontally cluster and sum consecutive pixels and their derivatives into Ps

    P_ = []  # row of Ps

    derts_ = [derts_[derts_index] for derts_ in derts__]

    i, ncomp, dy, dx, g = derts_[1][0]  # first derts
    x0, L, I, Dy, Dx, G = x_start, 1, p, g, dy // ncomp, dx // ncomp # P params
    _s = g > 0  # sign

    for x, (i, ncomp, dy, dx, g) in enumerate(dert_[2:-1], start=x_start + 1):
        s = g > 0
        if s != _s:  # P is terminated and new P is initialized
            P_.append([_s, x0, L, I, Dy, Dx, G, derts_[x0:x0 + L]])
            x0, L, I, Dy, Dx, G = x, 0, 0, 0, 0, 0

        # accumulate P params:
        L += 1
        I += i
        Dy += dy // ncomp
        Dx += dx // ncomp
        G += g

        _s = s  # prior sign

    P_.append([_s, x0, L, I, Dy, Dx, G, derts_[x0:x0 + L]])  # last P in row
    return P_
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