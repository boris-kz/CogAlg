from collections import deque
import math as math
from time import time
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

    dCx = max_x - min_x + 1;  dCy = max_y - min_y + 1
    rC = dCx / dCy  # width / height, vs shift / height: abs(xD) / Ly for oriented blobs only?
    rD = max(abs_Dx, abs_Dy) / min(abs_Dx, abs_Dy)  # lateral variation / vertical variation, for flip and comp_P eval
'''


def blob_eval(blob):
    blob = comp_angle(blob)  # angle comp, ablob def; a, da, sda accum in higher-composition reps
    return blob
def comp_angle(blob):  # compute and compare angle, define ablobs, accumulate a, da, sda in all reps within gblob
    ablob_ = []
    for segment in blob[3]:
        global y
        y = segment[1][2]   # y = segment's min_y
        # extract haP_ from fork
        haP_ = []
        P = segment[3][0][0]    # top-line P of segment
        for fork in segment[5]:
            fork_haP_, fork_remaining_roots = fork[-2:] # buffered haPs and remaining roots counter of segment's fork
            i = 0
            while i < len(fork_haP_):
                _aP = fork_haP_[i][0]
                while _aP[1][0] <= P[1][1] and P[1][0] <= _aP[1][1]:   # only takes overlapping haPs
                    haP_.append(fork_haP_.pop(i))
                    if i < len(fork_haP_):
                        _aP = fork_haP_[i][0]
                    else:
                        break
                i += 1
            while not fork_remaining_roots and fork_haP_:
                form_ablob(form_asegment(fork_haP_, ablob_), a_blob)    # terminate haPs with no connections
        for (P, xd) in segment[3]:  # iterate vertically
            # extract a higher-line aP_ from haP_
            _aP_ = []
            for haP in haP_:
                _aP_ += haP[0]
            # init:
            aP = [-1, [P[1][0], -1], [0, 0, 0], []] # P's init: [s, boundaries, params, dert_]
            aP_ = []
            buff_ = deque()
            i = 0                   # corresponding P's dert index
            _i = 0
            x = P[1][0]             # x = min_x
            _a = ave
            if not _aP_:
                no_higher_line = True
            else:
                _aP = _aP_.pop(0)
                while _aP[1][1] < P[1][0] and _aP_:  # repeat until _aP olp with or right-ward to P or no _aP left
                    _aP = _aP_.pop(0)
                if not _aP_:     # if no _aP left
                    no_higher_line = True
                else:
                    no_higher_line = False
                    _i = P[1][0] - _aP[1][0] # _aP's dert index
            # iteration:
            while i < P[2][0]       # while i < P's L
                dy, dx = P[3][i][:-2]  # first P's dert: i = 0
                a = int((math.atan2(dy, dx)) * degree) + 128
                # Lateral comp:
                mx = ave - abs(_a - a)
                _a = a
                # Vertical comp:
                my = ave
                if not no_higher_line and _i >= 0:  #
                    __a = _aP[3][i][0]  # vertically prior pixel's angle of gradient
                    my -= abs(__a - a)
                m = mx + my
                dert = a, m
                aP = form_aP(dert, aP, aP_, buff_, ablob_)
                x += 1
                i += 1
                _i += 1
                if not no_higher_line and _i > _aP[1][1]:   # end of _aP, pop next _aP
                    if _aP_:
                        _aP = _aP_.pop(0)
                    else:   # if no more _aP
                        no_higher_line = True
            y += 1
            haP_ = aP_  # haP_ buffered for next line
def intra_blob(frame):  # evaluate blobs for orthogonal flip, incr_rng_comp, incr_der_comp, comp_P
    I, G, Dx, Dy, xD, abs_xD, Ly, blob_ = frame
    new_blob_ = []
    for blob in blob_:
        if blob[0]:  # positive g sign
            new_blob_.append(blob_eval(blob))
    frame = I, G, Dx, Dy, xD, abs_xD, Ly, new_blob_
    return frame
# ************ MAIN FUNCTIONS END ***************************************************************************************

# ************ PROGRAM BODY *********************************************************************************************

# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:
ave = 15  # g value that coincides with average match: gP filter
div_ave = 1023  # filter for div_comp(L) -> rL, summed vars scaling
flip_ave = 10000  # cost of form_P and deeper?
ave_rate = 0.25  # match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)
dim = 2  # number of dimensions
rng = 2  # number of pixels compared to each pixel in four directions
min_coord = rng * 2 - 1  # min x and y for form_P input: ders2 from comp over rng*2 (bidirectional: before and after pixel p)
degree = 128 / math.pi  # coef to convert radian to 256 degrees
A_cost = 1000
a_cost = 15
# Main ---------------------------------------------------------------------------
start_time = time()
frame = intra_blob(frame_blobs.frame_of_blobs)
end_time = time() - start_time
print(end_time)
