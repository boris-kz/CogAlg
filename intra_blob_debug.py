from collections import deque
import math as math
from time import  time
import frame_blobs
'''
    intra_blob() is an extension to frame_blobs, it performs evaluation for comp_P and recursive frame_blobs within each blob.
    Currently it's mostly a draft, combined with frame_blobs it will form a 2D version of first-level algorithm
    inter_blob() will be second-level 2D algorithm, and a prototype for meta-level algorithm

    colors will be defined as color / sum-of-colors, color Ps are defined within sum_Ps: reflection object?
    relative colors may match across reflecting objects, forming color | lighting objects?     
    comp between color patterns within an object: segmentation?

    inter_olp_blob: scan alt_typ_ ) alt_color, rolp * mL > ave * max_L?   
    intra_blob rdn is eliminated by merging blobs, reduced by full inclusion: mediated access?
'''
# ************ MAIN FUNCTIONS *******************************************************************************************
# -pixel_comp()
# -form_P()
# -scan_P_()
# -form_segment()
# -form_blob()
# -intra_blobs()
# ***********************************************************************************************************************
def blob_eval(blob):
    [min_x, max_x, min_y, max_y, xD, abs_xD, Ly], [L, I, G, Dx, Dy], root_, _ = blob[1:]    # sign is always 1

    if L > A_cost:  # fixed per blob: ini params + ini params * (added params / ini params), converted to min L?
        if G > (ave + a_cost) * L:  # comp_a delay - comp delay per dert,
            blob = comp_angle(blob)  # angle comp, ablob def; a, da, sda accum in higher-composition reps
    return blob

def comp_angle(blob):
    # compute and compare angle, define ablobs, accumulate a, da, sda in higher-composition reps within input blob
    s, [min_x, max_x, min_y, max_y, xD, abs_xD, Ly], [L, I, G, Dx, Dy, abs_Dx, abs_Dy], root_ = blob
    A, Da, sDa = 0, 0, 0
    for i, seg in enumerate(range(len(root_))):
        [min_xs, max_xs, min_ys, xDs, ave_x], [Ls, Is, Gs, Dxs, Dys], Py_, fork_, roots, blob_ref = seg[1:]  # ignore s
        # first P of seg: scan higher-line _Ps in fork_
        P, xd = Py_[0]
        lateral_comp_a(P)
        _P_ = []
        for fork in fork_:
            _P_.append(fork[3][-1][0])  # get a list of _P from fork_
        P = vertical_comp_a(P, _P_)  # reconstruct P
        Py_[0] = P, xd
        As, Das, sDas = P[2][-3:]  # P[2]: P's params
        for ii, P in enumerate(range(len(Py_[1:]))):
            _P = Py_[ii - 1][0]
            P, xd = P
            lateral_comp_a(P)
            P = vertical_comp_a(P, _P)
            Py_[ii] = P, xd
            As += P[2][-3]
            Das += P[2][-2]
            sDas += P[2][-1]
        root_[i] = s, (min_xs, max_xs, min_ys, xDs), (Ls, Is, Gs, Dxs, Dys, As, Das, sDas), tuple(Py_), fork_, roots
        A += As
        Da += Das
        sDa += sDas

    return s, (min_x, max_x, min_y, xD, Ly), (L, I, G, Dx, Dy, A, Da, sDa), tuple(root_)


def lateral_comp_a(P):
    dert_ = P[3]
    dx, dy = dert_[0][-2:]  # first dert
    _a = int((math.atan2(dy, dx)) * degree) + 128  # angle from 0 -> 255
    da = ave
    dert_[0] += _a, da
    for i, dert in enumerate(dert_[1:]):
        dx, dy = dert[-2:]
        a = int((math.atan2(dy, dx)) * degree) + 128
        da = abs(a - _a)
        dert_[i] += a, da
        _a = a
    P[3] = dert_


def vertical_comp_a(P, *_P_):
    s, [min_x, max_x], [L, I, G, Dx, Dy], dert_ = P
    x = min_x;
    i = 0
    for _P in _P_:
        [_min_x, _max_x], _dert_ = _P[1], _P[3]
        if x < _min_x:
            i += _min_x - x
            x = _min_x
            _i = 0
        else:
            _i = x - min_x
        while _dert_[_i] and dert_[i]:
            _a = dert_[i][-2]
            p, g, dx, dy, a, da = dert_[i]
            da += abs(a - _a)
            sda = 2 * ave - da
            dert_[i] = p, g, dx, dy, a, da, sda
            x += 1;
            i += 1;
            _i += 1

    A, Da, sDa = 0, 0, 0
    for i, dert in enumerate(dert_):
        p, g, dx, dy, a, da = dert
        if dert[6]:
            sda = dert[6]
        else:
            sda = ave - da  # da += ave; sda = 2 * ave - da <=> sda = ave - da ?
        dert_[i] = (p, g, dx, dy, a, da, sda)
        A += a
        Da += da
        sDa += sda
    # aP = form_P(dert, _dert)  # i/o must be extended
    return s, (min_x, max_x), (L, I, G, Dx, Dy, A, Da, sDa), tuple(dert_)

def intra_blob(frame):  # evaluate blobs for orthogonal flip, incr_rng_comp, incr_der_comp, comp_P
    I, G, Dx, Dy, xD, abs_xD, Ly, blob_ = frame
    new_blob_ = []
    for blob in blob_:
        if blob[0]:  # positive g sign
            new_blob_.append(blob_eval(blob))
    frame = I, G, Dx, Dy, xD, Ly, new_blob_
# ************ MAIN FUNCTIONS END ***************************************************************************************
# ************ PROGRAM BODY *********************************************************************************************
# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:
ave = frame_blobs.ave   # g value that coincides with average match: gP filter
div_ave = 1023          # filter for div_comp(L) -> rL, summed vars scaling
flip_ave = 10000        # cost of form_P and deeper?
ave_rate = 0.25         # match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)
dim = 2                 # number of dimensions
rng = 2                 # number of pixels compared to each pixel in four directions
min_coord = rng * 2 - 1 # min x and y for form_P input: ders2 from comp over rng*2 (bidirectional: before and after pixel p)
degree = 128 / math.pi  # coef to convert radian to 256 degrees
A_cost = 1000
a_cost = 15

# Main ---------------------------------------------------------------------------
start_time = time()
frame = intra_blob(frame_blobs.frame_of_blobs)
end_time = time() - start_time
print(end_time)