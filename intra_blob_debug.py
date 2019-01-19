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
def blob_eval(blob, dert__):
    comp_angle(blob, dert__)  # angle comp, ablob def; a, da, sda accum in higher-composition reps
    return blob
def comp_angle(blob, dert__):  # compute and compare angle, define ablobs, accumulate a, da, sda in all reps within gblob
    ''' - Sort list of segments (root_) based on their top line P's coordinate (segment's min_y)    <---------------------------------------|
        - Iterate through each line in the blob (from blob's min_y to blob's max_y):                                                        |
            + Have every segment that contains current-line P in a list (seg_). This action is simplified by sorting step above  -----------|
            + Extract current-line slice of the blob - or the list of every P of this line (P_)
            + Have every out-of-bound segment removed from list (seg_)
            + Perform angle computing, comparison and clustering in every dert in P_ '''
    root_ = blob[3]
    blob[3] = sorted(root_, key=lambda segment: segment[1][2])  # sorted by min_y
    ablob_ = []
    global y
    y = blob[1][2]      # min_y of this blob
    seg_ = []
    haP_ = []
    i = 0    # iterator
    while i < len(root_):
        P_ = []
        while root_[i][1][2] == y:
            seg_.append([root_[i], 0])      # runningSegment consists of segments that contains y-line P and that P's index
        for ii, (seg, iP) in enumerate(seg_): # for every segment that contains y-line P
            P_.append(seg[3][ii][0])        # P = Py_[ii][0] = seg[3][ii][0]
            if y = seg[1][3]:               # if y has reached segment's bottom
                seg_.pop(ii)          # remove from list
            else:
                seg_[ii] += 1         # index point to next-line P
        # actual comp_angle:
        aP_ = []
        for P in P_:
            [min_x, max_x], L, dert_ = P[1], P[2][0], P[3]
            aP = [-1, [min_x, -1], [0, 0, 0], []]
            buff_ = []
            # init previous horizontal pixel's angle:
            if min_x == 1:  # no previous horizontal pixel's angle
                _a = 0      # this may not be the best value, needs further consideration
            else:
                _dert = dert__[y][min_x - 1]
                if len(_dert) < 5:             # angle hasn't been computed for this pixe
                    dx, dy = _dert
                    _a = math.atan2(dy, dx) * degree + 128    # angle label: 0 to 255 <--> -pi to pi in radian
                else:
                    _a = _dert[4]
            # init previous vertical pixel's angle
            if min_y == 1:  # no previous vertical pixel's angle
                _dert_ = (0, 0, 0, 0, 0) * L            # create a zero _dert_
            else:
                _dert_ = dert__[y - 1][min_x:max_x+1]   # get corresponding higher-line dert_
            x = min_x
            for dert, _dert in zip(dert_, _dert_):
                dx, dy = dert[2:]
                a = math.atan2(dy, dx) * degree + 128
                if len(_dert) < 5:
                    _dx, _dy = _dert[2:]
                    __a = math.atan2(_dy, _dx) * degree + 128
                else:
                    __a = _dert[4]
                sda = abs(a - _a) + abs(a - __a) - 2 * ave
                dert += a, sda
                aP = form_aP(dert, x, max_x, aP, aP_, haP_, buff_, ablob_)
                _a = a
                x += 1
        # buffers for next line
        haP_ = aP_
    blob[4] = ablob_
def form_aP(dert, x, x_stop, aP, aP_, buff_, haP_, ablob_):
    a, sda = dert[-2:]
    s = 1 if sda > 0 else 0
    pri_s = aP[0]
    if s != pri_s and pri_s != -1:  # aP is terminated:
        aP[1][1] = x - 1  # aP's max_x
        scan_aP_(aP, aP_, buff_, haP_, ablob_)  # aP scans haP_
        P = [s, [x, -1], [0, 0, 0], []]  # new aP initialization
    [min_x, max_x], [L, A, sDa], dert_ = P[1:]  # continued or initialized input and derivatives are accumulated:
    L += 1  # length of a pattern
    A += a  # summed angle
    sDa += sda  # summed sda
    dert_.append(dert)  # der2s are buffered for oriented rescan and incremental range | derivation comp
    aP = [s, [min_x, max_x], [L, A, sDa], dert_]
    if x == x_stop:  # aP is terminated:
        P[1][1] = x  # aP's max_x
        scan_aP_(aP, aP_, buff_, haP_, ablob_)
    return aP  # accumulated within line, P_ is a buffer for conversion to _P_
def scan_aP_(aP, aP_, _buff_, haP_, ablob_):
    fork_ = []  # refs to haPs connected to input aP
    _min_x = 0  # to start while loop, next ini_x = _x + 1
    min_x, max_x = aP[1]
    while _min_x <= max_x:  # while x values overlap between aP and _aP
        if _buff_:
            haP = _buff_.popleft()  # haP was extended to segment and buffered in prior scan_aP_
        elif haP_:
            haP = form_asegment(haP_.popleft(), ablob_)
        else:
            break  # higher line ends, all haPs are converted to segments
        roots = haP[4]
        _aP = haP[3][-1][0]
        _min_x, _max_x = _aP[1]  # first_x, last_x
        if aP[0] == _aP[0] and min_x <= _max_x and _min_x <= max_x:
            roots += 1
            haP[4] = roots
            fork_.append(haP)  # aP-connected haPs will be converted to segments at each _fork
        if _max_x > max_x:  # x overlap between haP and next aP: haP is buffered for next scan_aP_, else haP included in a blob segment
            _buff_.append(haP)
        elif roots != 1:
            form_ablob(haP, ablob_)  # segment is terminated and packed into its blob
        _min_x = _max_x + 1  # = first x of next _aP
    P_.append((aP, fork_))  # aP with no overlap to next _aP is extended to haP and buffered for next-line scan_aP_
def form_asegment(haP, ablob_):
    _aP, fork_ = haP
    s, [min_x, max_x], params = _aP[:-1]
    ave_x = (params[0] - 1) // 2  # extra-x L = L-1 (1x in L)

    if not fork_:  # seg is initialized with initialized blob (params, coordinates, incomplete_segments, root_, xD)
        ablob = [s, [min_x, max_x, y - 1, -1, 0, 0, 0], [0, 0, 0], [], 1]  # s, coords, params, root_, incomplete_segments
        haP = [s, [min_x, max_x, y - 1, -1, 0, 0, ave_x], params, [(_aP, 0)], 0, fork_, ablob]
        ablob[3].append(haP)
    else:
        if len(fork_) == 1 and fork_[0][4] == 1:  # haP has one fork: haP[2][0], and that fork has one root: haP
            # haP is merged into higher-line blob segment (Pars, roots, _fork_, ave_x, xD, aPy_, blob) at haP[2][0]:
            fork = fork_[0]
            fork[1][0] = min(fork[1][0], min_x)
            fork[1][1] = max(fork[1][1], max_x)
            xd = ave_x - fork[1][5]
            fork[1][4] += xd
            fork[1][5] += abs(xd)
            fork[1][6] = ave_x
            L, A, sDa = params
            Ls, As, sDas = fork[2]  # seg params
            fork[2] = [Ls + L, As + A, sDas + sDa]
            fork[3].append((_aP, xd))  # aPy_: vertical buffer of aPs merged into seg
            fork[4] = 0  # reset roots
            haP = fork  # replace segment with including fork's segment
            ablob = haP[6]

        else:  # if >1 forks, or 1 fork that has >1 roots:
            haP = [s, [min_x, max_x, y - 1, -1, 0, 0, ave_x], params, [(_aP, 0)], 0, fork_, fork_[0][6]]  # seg is initialized with fork's blob
            ablob = haP[6]
            ablob[3].append(haP)  # segment is buffered into root_
            if len(fork_) > 1:  # merge blobs of all forks
                if fork_[0][4] == 1:  # if roots == 1
                    form_ablob(fork_[0], ablob_, 1)  # merge seg of 1st fork into its blob

                for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                    if fork[4] == 1:
                        form_ablob(fork, ablob_, 1)
                    if not fork[6] is ablob:
                        [min_x, max_x, min_y, max_y, xD, abs_xD, Ly], [L, I, G, Dx, Dy], root_, incomplete_segments = fork[6][1:]  # ommit sign
                        ablob[1][0] = min(min_x, ablob[1][0])
                        ablob[1][1] = max(max_x, ablob[1][1])
                        ablob[1][2] = min(min_y, ablob[1][2])
                        ablob[1][4] += xD
                        ablob[1][5] += abs_xD
                        ablob[1][6] += Ly
                        ablob[2][0] += L
                        ablob[2][1] += A
                        ablob[2][2] += sDa
                        ablob[4] += incomplete_segments
                        for seg in root_:
                            if not seg is fork:
                                seg[6] = ablob  # blobs in other forks are references to blob in the first fork
                                ablob[3].append(seg)  # buffer of merged root segments
                        fork[6] = ablob
                        ablob[3].append(fork)
                    ablob[4] -= 1
        ablob[1][0] = min(min_x, ablob[1][0])
        ablob[1][1] = max(max_x, ablob[1][1])
    return haP
def form_ablob(term_seg, ablob_, y_carry=0):
    [min_x, max_x, min_y, max_y, xD, abs_xD, ave_x], [L, A, sDa], Py_, roots, fork_, ablob = term_seg[1:]  # ignore sign
    ablob[1][4] += xD  # ave_x angle, to evaluate blob for re-orientation
    ablob[1][5] += len(Py_)  # Ly = number of slices in segment
    ablob[2][0] += L
    ablob[2][1] += A
    ablob[2][2] += sDa
    ablob[4] += roots - 1  # reference to term_seg is already in blob[9]
    aterm_seg[1][3] = y - 1 - y_carry  # y_carry: min elevation of term_seg over current hP
    if not ablob[4]:
        ablob[1][3] = term_seg[1][3]
        ablob_.append(ablob)
def intra_blob(frame):  # evaluate blobs for orthogonal flip, incr_rng_comp, incr_der_comp, comp_P
    I, G, Dx, Dy, xD, abs_xD, Ly, blob_, dert__ = frame
    new_blob_ = []
    for blob in blob_:
        if blob[0]:  # positive g sign
            new_blob_.append(blob_eval(blob, dert__))
    frame = I, G, Dx, Dy, xD, abs_xD, Ly, new_blob_, dert__
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
rng = 1  # number of pixels compared to each pixel in four directions
min_coord = rng  # min x and y for form_P input: ders2 from comp over rng*2 (bidirectional: before and after pixel p)
degree = 128 / math.pi  # coef to convert radian to 256 degrees
A_cost = 1000
a_cost = 15
# Main ---------------------------------------------------------------------------
start_time = time()
frame = intra_blob(frame_blobs.frame_of_blobs)
end_time = time() - start_time
print(end_time)