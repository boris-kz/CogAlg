import cv2
import argparse
import numpy as np
from scipy import misc
from time import time
from collections import deque

'''   
    frame() is core algorithm of levels 1 + 2, modified for 2D: segmentation of image into blobs, then recursive search within blobs.
    frame_blobs() is frame() limited to definition of initial blobs per each of 4 derivatives, vs. per 2 gradients in frame_draft.
    frame_dblobs() is updated version of frame_blobs with only one blob verte: dblob, to ease debugging, currently in progress.

    Each performs several levels (Le) of encoding, incremental per scan line defined by vertical coordinate y, outlined below.
    value of y per Le line is shown relative to y of current input line, incremented by top-down scan of input image:

    1Le, line y:    x_comp(p_): lateral pixel comparison -> tuple of derivatives ders ) array ders_
    2Le, line y- 1: y_comp(ders_): vertical pixel comp -> 2D tuple ders2 ) array ders2_ 
    3Le, line y- 1+ rng*2: form_P(ders2) -> 1D pattern P
    4Le, line y- 2+ rng*2: scan_P_(P, hP) -> hP, roots: down-connections, fork_: up-connections between Ps 
    5Le, line y- 3+ rng*2: form_segment(hP, seg) -> seg: merge vertically-connected _Ps in non-forking blob segments
    6Le, line y- 4+ rng*2+ seg depth: form_blob(seg, blob): merge connected segments in fork_' incomplete blobs, recursively  

    for y = rng *2, the outputs are: y P ) y-1 hP ) y-2 seg ) y-4 blob ) y-5 frame

    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match.
    Each vertical and horizontal derivative forms separate blobs, suppressing overlapping orthogonal representations.

    They can also be summed to estimate diagonal or hypot derivatives, for blob orientation to maximize primary derivatives.
    Orientation increases primary dimension of blob to maximize match, and decreases secondary dimension to maximize difference.
    Subsequent union of lateral and vertical patterns is by strength only, orthogonal sign is not commeasurable?

    Initial pixel comparison is not novel, I design from the scratch to make it organic part of hierarchical algorithm.
    It would be much faster with matrix computation, but this is minor compared to higher-level processing.
    I implement it sequentially for consistency with accumulation into blobs: irregular and very difficult to map to matrices.

    All 2D functions (y_comp, scan_P_, form_blob) input two lines: higher and lower, convert elements of lower line 
    into elements of new higher line, and displace elements of old higher line into higher function.
    Higher-line elements include additional variables, derived while they were lower-line elements.

    prefix '_' denotes higher-line variable or pattern, vs. same-verte lower-line variable or pattern,
    postfix '_' denotes array name, vs. same-name elements of that array:
'''


# ************ MISCELLANEOUS FUNCTIONS **********************************************************************************
# Includes:
# -rebuild_blobs()
# -segment_sort_by_height()
# -fetch_frame()
# ***********************************************************************************************************************

def rebuild_blobs(frame, print_separate_blobs=0):
    " Rebuilt data of blobs into an image "
    blob_image = np.array([[[127] * 4] * X] * Y)

    for index, blob in enumerate(frame[2]):  # Iterate through blobs
        if print_separate_blobs: blob_img = np.array([[[127] * 4] * X] * Y)
        for seg in blob[2]:  # Iterate through segments
            y = seg[7]
            for (P, dx) in reversed(seg[5]):
                x = P[1]
                for i in range(P[2]):
                    blob_image[y, x, : 3] = [255, 255, 255] if P[0] else [0, 0, 0]
                    if print_separate_blobs: blob_img[y, x, : 3] = [255, 255, 255] if P[0] else [0, 0, 0]
                    x += 1
                y -= 1
        if print_separate_blobs:
            min_x, max_x, min_y, max_y = blob[1][:4]
            cv2.rectangle(blob_img, (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), (0, 255, 255), 1)
            cv2.imwrite('./images/raccoon_eye/blob%d.jpg' % (index), blob_img)

    return blob_image
    # ---------- rebuild_blobs() end ------------------------------------------------------------------------------------


def segment_sort_by_height(seg):
    " Used in sort() function at blob termination "
    return seg[7]
    # ---------- segment_sort_by_height() end ---------------------------------------------------------------------------


def fetch_frame(video):
    _, frame = video.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# ************ MISCELLANEOUS FUNCTIONS END ******************************************************************************

# ************ MAIN FUNCTIONS *******************************************************************************************
# Includes:
# -lateral_comp() x
# -vertical_comp() x
# -temporal_comp() x
# -form_P()
# -scan_P_()
# -form_segment()
# -form_blob()
# -video_to_tblobs() x
# ***********************************************************************************************************************
def lateral_comp(pixel_):
    " Comparison over x coordinate, within rng of consecutive pixels on each line "

    ders1_ = []  # tuples of complete 1D derivatives: summation range = rng
    rng_ders1_ = deque(maxlen=rng)  # incomplete ders1s, within rng from input pixel: summation range < rng
    rng_ders1_.append((0, 0, 0))
    max_index = rng - 1  # max index of rng_ders1_

    for x, p in enumerate(pixel_):  # pixel p is compared to rng of prior pixels within horizontal line, summing d and m per prior pixel
        back_fd, back_fm = 0, 0  # fuzzy derivatives from rng of backward comps per pri_p
        for index, (pri_p, fd, fm) in enumerate(rng_ders1_):
            d = p - pri_p
            m = ave - abs(d)
            fd += d  # bilateral fuzzy d: running sum of differences between pixel and all prior and subsequent pixels within rng
            fm += m  # bilateral fuzzy m: running sum of matches between pixel and all prior and subsequent pixels within rng
            back_fd += d
            back_fm += m  # running sum of d and m between pixel and all prior pixels within rng

            if index < max_index:
                rng_ders1_[index] = (pri_p, fd, fm)
            elif x > rng * 2 - 1:  # after pri_p comp over full bilateral rng
                ders1_.append((pri_p, fd, fm))  # completed bilateral tuple is transferred from rng_ders_ to ders_

        rng_ders1_.appendleft((p, back_fd, back_fm))  # new tuple with initialized d and m, maxlen displaces completed tuple
    # last incomplete rng_ders1_ in line are discarded, vs. ders1_ += reversed(rng_ders1_)
    return ders1_
    # ---------- lateral_comp() end -------------------------------------------------------------------------------------


def vertical_comp(ders1_, rng_ders2__):
    " Comparison to bilateral rng of vertically consecutive pixels, forming ders2: pixel + lateral and vertical derivatives"
    ders2_ = []  # tuples of complete 2D derivatives: summation range = rng
    new_rng_ders2__ = deque()  # 2D array: line of ders2_s buffered for next-line comp
    max_index = rng - 1  # max ders2_ index
    x = rng  # lateral coordinate of pixel in input ders1

    for (p, dx, mx), rng_ders2_ in zip(ders1_, rng_ders2__):  # pixel comp to rng _pixels in ders2_, summing dy and my
        index = 0
        back_dy, back_my = 0, 0
        for (_p, _dx, fdy, _mx, fmy) in rng_ders2_:  # vertical derivatives are incomplete; prefix '_' denotes higher-line variable
            dy = p - _p
            my = ave - abs(dy)
            fdy += dy  # running sum of differences between pixel _p and all higher and lower pixels within rng
            fmy += my  # running sum of matches between pixel _p and all higher and lower pixels within rng
            back_dy += dy;
            back_my += my  # running sum of d and m between pixel _p and all higher pixels within rng
            if index < max_index:
                rng_ders2_[index] = (_p, _dx, fdy, _mx, fmy)
            elif y > min_coord:
                ders2_.append((_p, _dx, fdy, _mx, fmy))  # completed bilateral tuple is transferred from rng_ders2_ to ders2_
            index += 1

        rng_ders2_.appendleft((p, dx, back_dy, mx, back_my))  # new ders2 displaces completed one in vertical ders2_ via maxlen
        new_rng_ders2__.append(rng_ders2_)  # 2D array of vertically-incomplete 2D tuples, converted to ders2__, for next-line vertical comp
        x += 1

    return ders2_, new_rng_ders2__
    # ---------- vertical_comp() end ------------------------------------------------------------------------------------


def temporal_comp(ders2_, rng_ders3___, _xP_, _yP_, _tP_, frame, videoo):
    ''' ders2_: an array of complete 2D ders
        rng_ders3___: an older frame of 3D tuple arrays, sliced into an array
        comparison between t_rng temporally consecutive pixels, forming ders3: 3D tuple of derivatives per pixel '''
    # lateral pattern = pri_s, x0, L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0:6 ders2_
    xP = [[0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []], \
          [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]]
    yP = [[0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []], \
          [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]]
    tP = [[0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []], \
          [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]]
    xP_ = [deque(), deque()]
    yP_ = [deque(), deque()]  # line y - 1 + rng*2
    tP_ = [deque(), deque()]
    xbuff_ = [deque(), deque()]
    ybuff_ = [deque(), deque()]  # line y - 2 + rng*2: _Ps buffered by previous run of scan_P_
    tbuff_ = [deque(), deque()]
    # Note: each one of P, P_, buff_ is a list of 4 corresponding to 4 types of patterns ( m, my, d, dy respectively )
    rng_ders3__ = rng_ders3___.pop(0)
    new_rng_ders3__ = deque()  # 2D array: line of rng_ders2_s buffered for next-line comp
    max_index = t_rng - 1  # max rng_ders3_ index
    x = rng  # lateral coordinate of pixel in input ders1

    for (p, dx, dy, mx, my), rng_ders3_ in zip(ders2_, rng_ders3__):  # pixel comp to rng _pixels in rng_ders2_, summing dy and my
        index = 0
        back_dt, back_mt = 0, 0
        for (_p, _dx, _dy, fdt, _mx, _my, fmt) in rng_ders3_:  # vertical derivatives are incomplete; prefix '_' denotes higher-line variable
            dt = p - _p
            mt = ave - abs(dt)
            fdt += dt  # running sum of differences between pixel _p and all higher and lower pixels within rng
            fmt += mt  # running sum of matches between pixel _p and all higher and lower pixels within rng
            back_dt += dt
            back_mt += mt  # running sum of d and m between pixel _p and all higher pixels within rng

            if index < max_index:
                rng_ders3_[index] = (_p, _dx, _dy, fdt, _mx, _my, fmt)
            elif t > t_min_coord:
                ders = _p, _dx, _dy, fdt, _mx, _my, fmt
                xP, xP_, xbuff_, _xP_, frame = form_P(ders, x, X - rng - 1, xP, xP_, xbuff_, _xP_, frame, 0)  # lateral mP, typ = 0
                yP, yP_, ybuff_, _yP_, frame = form_P(ders, x, X - rng - 1, yP, yP_, ybuff_, _yP_, frame, 1)  # vertical mP, typ = 1
                tP, tP_, tbuff_, _tP_, frame = form_P(ders, x, X - rng - 1, tP, tP_, tbuff_, _tP_, frame, 2)  # temporal mP, typ = 2
            index += 1

        rng_ders3_.appendleft((p, dx, dy, back_dt, mx, my, back_mt))  # new ders2 displaces completed one in vertical rng_ders2_ via maxlen
        new_rng_ders3__.append(rng_ders3_)  # 2D array of vertically-incomplete 2D tuples, converted to rng_ders2__, for next-line vertical comp
        x += 1

    # Unlike mPs, dPs are disjointed, and the algorithm only deal with continuous Ps
    # To deal with disjointed dPs, 2 cases must be solved: not overlapping P - hP and unfinished hPs
    # Unfinished lateral hPs of are handled here:
    typ = dim  # lateral dP, typ = 3
    while xbuff_[1]:
        hP = xbuff_[1].popleft()
        if hP[1] != 1:  # no roots
            frame = form_blob(hP, frame, typ)
    while _xP_[1]:
        hP, frame = form_segment(_xP_[1].popleft(), frame, typ)
        frame = form_blob(hP, frame, typ)

    # Same for vertical hPs:
    typ += 1  # vertical dP, typ = 4
    while ybuff_[1]:
        hP = ybuff_[1].popleft()
        if hP[1] != 1:  # no roots
            frame = form_blob(hP, frame, typ)
    while _yP_[1]:
        hP, frame = form_segment(_yP_[1].popleft(), frame, typ)
        frame = form_blob(hP, frame, typ)

    # ...and temporal hPs:
    typ += 1  # vertical dP, typ = 4
    while tbuff_[1]:
        hP = tbuff_[1].popleft()
        if hP[1] != 1:  # no roots
            frame = form_blob(hP, frame, typ)
    while _tP_[1]:
        hP, frame = form_segment(_tP_[1].popleft(), frame, typ)
        frame = form_blob(hP, frame, typ)

    rng_ders3___.append(new_rng_ders3__)  # rng_ders3__ for next frame

    return rng_ders3___, xP_, yP_, tP_, frame, videoo

    # ---------- temporal_comp() end ------------------------------------------------------------------------------------


def form_P(ders, x, max_x, P, P_, buff_, hP_, frame, typ, is_dP=0):
    " Initializes, and accumulates 1D pattern "
    # is_dP could be calculated from typ: is_dP = bool(typ // dim)
    # but use direct entry for speed?

    p, dx, dy, dt, mx, my, mt = ders  # 2D tuple of derivatives per pixel, "y" denotes vertical vs. lateral derivatives
    if typ == 0:
        core = mx; alt0 = dx; alt1 = my; alt2 = mt; alt3 = dy; alt4 = dt
    elif typ == 1:
        core = my; alt0 = dy; alt1 = mx; alt2 = mt; alt3 = dx; alt4 = dt
    elif typ == 2:
        core = mt; alt0 = dt; alt1 = mx; alt2 = my; alt3 = dx; alt4 = dy
    elif typ == 3:
        core = dx; alt0 = mx; alt1 = dy; alt2 = dt; alt3 = my; alt4 = mt
    elif typ == 4:
        core = dy; alt0 = my; alt1 = dx; alt2 = dt; alt3 = mx; alt4 = mt
    else:
        core = dt; alt0 = mt; alt1 = dx; alt2 = dy; alt3 = mx; alt4 = my

    s = 1 if core > 0 else 0
    pri_s, x0 = P[is_dP][:2]  # P[0] is mP, P[1] is dP of the same dimension

    if not (s == pri_s or x == x0):  # P is terminated
        P, P_, buff_, hP_, frame = term_P(s, x, P, P_, buff_, hP_, frame, typ, is_dP)

    pri_s, x0, L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4, ders_ = P[is_dP]

    L += 1  # length of a pattern, continued or initialized input and derivatives are accumulated:
    I += p  # summed input
    Dx += dx  # lateral D
    Dy += dy  # vertical D
    Dt += dt  # temporal D
    Mx += mx  # lateral M
    My += my  # vertical M
    Mt += mt  # temporal M
    Alt0 += abs(alt0)  # abs alt cores indicate value of alt-core Ps, to compute P redundancy rate
    Alt1 += abs(alt1)  # vs. specific overlaps: cost > gain in precision?
    Alt2 += abs(alt2)
    Alt3 += abs(alt3)
    Alt4 += abs(alt4)

    # in frame_x_blobs, alt_Der and alt_Both are computed for comp_P eval, but add to rdn only within neg mPs

    ders_.append(ders)  # ders2s are buffered for oriented rescan and incremental range | derivation comp
    P[is_dP] = s, x0, L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4, ders_

    if x == max_x:  # P is terminated
        P, P_, buff_, hP_, frame = term_P(s, x + 1, P, P_, buff_, hP_, frame, typ, is_dP)

    return P, P_, buff_, hP_, frame  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_P() end -------------------------------------------------------------------------------------------


def term_P(s, x, P, P_, buff_, hP_, frame, typ, is_dP):
    " Terminates 1D pattern when sign-change is detected or at the end of the line "

    pri_s, x0, L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4, ders_ = P[is_dP]
    if not is_dP and not pri_s:  # dPs formed inside of negative mP
        # P[0] == mP; P[1] == dP
        P[1] = [-1, x0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]

        for i in range(L):
            P, P_, buff_, _P_, frame = form_P(ders_[i], x0 + i, x0 + L - 1, P, P_, buff_, hP_, frame, typ + dim, True)  # is_dP = 1

        P[0] = pri_s, x0, L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4, ders_, P_[1]

    if y == rng * 2:  # 1st line: form_P converts P to initialized hP, forming initial P_ -> hP_
        P_[is_dP].append([P[is_dP], 0, [], x - 1])  # P, roots, _fork_, x
    else:
        P_[is_dP], buff_[is_dP], hP_[is_dP], frame \
            = scan_P_(x - 1, P[is_dP], P_[is_dP], buff_[is_dP], hP_[is_dP], frame, typ)  # scans higher-line Ps for contiguity
        # x-1 for prior p
    P[is_dP] = s, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []  # new P initialization

    return P, P_, buff_, hP_, frame
    # ---------- term_P() end -------------------------------------------------------------------------------------------


def scan_P_(x, P, P_, _buff_, hP_, frame, typ):
    " P scans shared-x-coordinate hPs in higher P_, combines overlapping Ps into blobs "

    buff_ = deque()  # new buffer for displaced hPs (higher-line P tuples), for scan_P_(next P)
    fork_ = []  # refs to hPs connected to input P
    ini_x = 0  # to start while loop, next ini_x = _x + 1
    x0 = P[1]

    while ini_x <= x:  # while x values overlap between P and _P
        if _buff_:
            hP = _buff_.popleft()  # hP was extended to segment and buffered in prior scan_P_
        elif hP_:
            hP, frame = form_segment(hP_.popleft(), frame, typ)
        else:
            break  # higher line ends, all hPs are converted to segments

        roots = hP[1]
        _x0 = hP[5][-1][0][1]  # firt_x
        _x = _x0 + hP[5][-1][0][2] - 1  # last_x = first_x + L - 1

        # 3 conditions for P and _P to overlap: s == _s, _last_x >= first_x and last_x >= _first_x
        if P[0] == hP[6][0][0] and not _x < x0 and not x < _x0:
            roots += 1;
            hP[1] = roots
            fork_.append(hP)  # P-connected hPs will be converted to segments at each _fork

        if _x > x:  # x overlap between hP and next P: hP is buffered for next scan_P_, else hP included in a blob segment
            buff_.append(hP)
        elif roots != 1:
            frame = form_blob(hP, frame, typ)  # segment is terminated and packed into its blob

        ini_x = _x + 1  # = first x of next _P

    buff_ += _buff_  # _buff_ is likely empty
    P_.append([P, 0, fork_, x])  # P with no overlap to next _P is extended to hP and buffered for next-line scan_P_

    return P_, buff_, hP_, frame  # hP_ and buff_ contain only remaining _Ps, with _x => next x
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_segment(hP, frame, typ):
    " Convert hP into new segment or add it to higher-line segment, merge blobs "
    _P, roots, fork_, last_x = hP
    [s, first_x], params = _P[:2], list(_P[2:15])
    ave_x = (_P[2] - 1) // 2  # extra-x L = L-1 (1x in L)

    if not fork_:  # seg is initialized with initialized blob (params, coordinates, remaining_roots, root_, xD)
        blob = [[s, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [_P[1], hP[3], y - rng - 1, 0, 0], 1, []]
        hP = [params, roots, fork_, ave_x, 0, [(_P, 0)], blob]
        blob[3].append(hP)
    else:
        if len(fork_) == 1 and fork_[0][1] == 1:  # hP has one fork: hP[2][0], and that fork has one root: hP
            # hP is merged into higher-line blob segment (Pars, roots, _fork_, ave_x, xD, Py_, blob) at hP[2][0]:
            fork = fork_[0]
            L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4 = params
            Ls, Is, Dxs, Dys, Dts, Mxs, Mys, Mts, Alt0s, Alt1s, Alt2s, Alt3s, Alt4s = fork[0]  # seg params
            fork[0] = [Ls + L, Is + I, Dxs + Dx, Dys + Dy, Dts + Dt, Mxs + Mx, Mys + My, Mts + Mt, \
                       Alt0s + Alt0, Alt1s + Alt1, Alt2s + Alt2, Alt3s + Alt3, Alt4s + Alt4]
            fork[1] = roots
            dx = ave_x - fork[3]
            fork[3] = ave_x
            fork[4] += dx  # xD for seg normalization and orientation, or += |dx| for curved yL?
            fork[5].append((_P, dx))  # Py_: vertical buffer of Ps merged into seg
            hP = fork  # replace segment with including fork's segment
            blob = hP[6]

        else:  # if >1 forks, or 1 fork that has >1 roots:
            hP = [params, roots, fork_, ave_x, 0, [(_P, 0)], fork_[0][6]]  # seg is initialized with fork's blob
            blob = hP[6]
            blob[3].append(hP)  # segment is buffered into root_

            if len(fork_) > 1:  # merge blobs of all forks
                if fork_[0][1] == 1:  # if roots == 1
                    frame = form_blob(fork_[0], frame, typ, 1)  # merge seg of 1st fork into its blob

                for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                    if fork[1] == 1:
                        frame = form_blob(fork, frame, typ, 1)

                    if not fork[6] is blob:
                        [s, L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4], [min_x, max_x, min_y, xD, Ly], remaining_roots, root_ = fork[6]
                        blob[0][1] += L
                        blob[0][2] += I
                        blob[0][3] += Dx
                        blob[0][4] += Dy
                        blob[0][5] += Dt
                        blob[0][6] += Mx
                        blob[0][7] += My
                        blob[0][8] += Mt
                        blob[0][9] += Alt0
                        blob[0][10] += Alt1
                        blob[0][11] += Alt2
                        blob[0][12] += Alt3
                        blob[0][13] += Alt4

                        blob[1][0] = min(min_x, blob[1][0])
                        blob[1][1] = max(max_x, blob[1][1])
                        blob[1][2] = min(min_y, blob[1][2])
                        blob[1][3] += xD
                        blob[1][4] += Ly
                        blob[2] += remaining_roots
                        for seg in root_:
                            if not seg is fork:
                                seg[6] = blob  # blobs in other forks are references to blob in the first fork
                                blob[3].append(seg)  # buffer of merged root segments
                        fork[6] = blob
                        blob[3].append(fork)
                    blob[2] -= 1

        blob[1][0] = min(first_x, blob[1][0])  # min_x
        blob[1][1] = max(last_x, blob[1][1])  # max_x
    return hP, frame
    # ---------- form_segment() end -----------------------------------------------------------------------------------------


def form_blob(term_seg, frame, typ, y_carry=0):
    " Terminated segment is merged into continued or initialized blob (all connected segments) "

    [L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4], roots, fork_, x, xD, Py_, blob = term_seg  # unique blob in fork_[0][6] is ref'd by other forks
    blob[0][1] += L
    blob[0][2] += I
    blob[0][3] += Dx
    blob[0][4] += Dy
    blob[0][5] += Dt
    blob[0][6] += Mx
    blob[0][7] += My
    blob[0][8] += Mt
    blob[0][9] += Alt0
    blob[0][10] += Alt1
    blob[0][11] += Alt2
    blob[0][12] += Alt3
    blob[0][13] += Alt4

    blob[1][3] += xD  # ave_x angle, to evaluate blob for re-orientation
    blob[1][4] += len(Py_)  # Ly = number of slices in segment

    blob[2] += roots - 1  # reference to term_seg is already in blob[9]
    term_seg.append(y - rng - 1 - y_carry)  # y_carry: min elevation of term_seg over current hP

    if not blob[2]:  # if remaining_roots == 0: blob is terminated and packed in frame
        [s, L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4], [min_x, max_x, min_y, xD, Ly], remaining_roots, root_ = blob
        if not typ:  # frame P are to compute averages, redundant for same-scope alt_frames
            frame[0][1] += I
            frame[0][2] += Dx
            frame[0][3] += Dy
            frame[0][4] += Dt
            frame[0][5] += Mx
            frame[0][6] += My
            frame[0][7] += Mt
        if not s and typ < dim:
            frame[0][0][typ] += L  # L of negative mblobs are summed

        frame[typ + 1][0] += xD  # ave_x angle, to evaluate frame for re-orientation
        frame[typ + 1][1] += Ly  # +L
        root_.sort(key=segment_sort_by_height)  # Sort segments by max_y
        frame[typ + 1].append(((s, L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4), (min_x, max_x, min_y, term_seg[7], xD, Ly), root_))

    return frame  # no term_seg return: no root segs refer to it
    # ---------- form_blob() end ----------------------------------------------------------------------------------------


def video_to_tblobs(video):  # currently only a draft
    ''' Main body of the operation,
        postfix '_' denotes array vs. element,
        prefix '_' denotes prior- pixel, line, or frame variable '''

    # higher-line same- d-, m-, dy-, my- sign 1D patterns
    _xP_ = [deque(), deque()]
    _yP_ = [deque(), deque()]
    _tP_ = [deque(), deque()]

    # prior frame: [[neg_mL, neg_myL, neg_mtL], I, Dx, Dy, Dt, Mx, My, Mt], 6 x [xD, Ly, blob_]
    _frame = [[[0, 0, 0], 0, 0, 0, 0, 0, 0, 0], [0, 0, []], [0, 0, []], [0, 0, []], [0, 0, []], [0, 0, []], [0, 0, []]]

    videoo = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [], [], [], [], [], []]  # [[Dxf, Lf, If, Dxf, Dyf, Dtf, Mxf, Myf, Mtf], net_]
    global t, y
    t = 0  # temporal coordinate of current frame

    # initialization:
    rng_ders2__ = []  # horizontal line of vertical buffers: array of 2D tuples
    rng_ders3___ = []  # temporal buffer per pixel of a frame: 3D tuples in 3D -> 2D array

    y = 0  # initial line
    line_ = fetch_frame(video)  # first frame of lines?
    pixel_ = line_[0, :]  # first line of pixels
    ders1_ = lateral_comp(pixel_)  # after partial comp, while x < rng?

    for (p, d, m) in ders1_:
        ders2 = p, d, 0, m, 0  # dy, my initialized at 0
        rng_ders2_ = deque(maxlen=rng)  # vertical buffer of incomplete derivatives tuples, for fuzzy ycomp
        rng_ders2_.append(ders2)  # only one tuple in first-line rng_ders2_
        rng_ders2__.append(rng_ders2_)

    for y in range(1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?
        pixel_ = line_[y, :]  # vertical coordinate y is index of new line p_
        ders1_ = lateral_comp(pixel_)  # lateral pixel comparison
        ders2_, rng_ders2__ = vertical_comp(ders1_, rng_ders2__)  # vertical pixel comparison, ders2_ is array of complete der2s on y line
        # Just like ders1_, incomplete ders2_ are discarded
        if y > min_coord:
            # Transfer complete list of tuples of ders2 into line y of ders3___
            rng_ders3__ = []
            for (p, dx, dy, mx, my) in ders2_:
                ders3 = p, dx, dy, 0, mx, my, 0  # dt, mt initialized at 0
                rng_ders3_ = deque(maxlen=t_rng)  # temporal buffer of incomplete derivatives tuples, for fuzzy ycomp
                rng_ders3_.append(ders3)  # only one tuple in first-frame rng_ders3_
                rng_ders3__.append(rng_ders3_)

            rng_ders3___.append(rng_ders3__)

    # frame ends, last vertical rng of incomplete ders2__ is discarded,

    for t in range(1, T):  # actual processing
        if not video.isOpened():  # Terminate at the end of video
            break
        # Main operations
        frame = [[[0, 0, 0], 0, 0, 0, 0, 0, 0, 0], [0, 0, []], [0, 0, []], [0, 0, []], [0, 0, []], [0, 0, []], [0, 0, []]]
        line_ = fetch_frame(video)
        for y in range(0, Y):
            pixel_ = line_[y, :]
            ders1_ = lateral_comp(pixel_)  # lateral pixel comparison
            ders2_, rng_ders2__ = vertical_comp(ders1_, rng_ders2__)  # vertical pixel comparison
            if y > min_coord:
                rng_ders3___, _xP_, _yP_, _tP_, frame, videoo = temporal_comp(ders2_, rng_ders3___, _xP_, _yP_, _tP_, frame, videoo)  # temporal pixel comparison

        # merge segs of last line into their blobs:
        y = Y
        for is_dP in range(2):
            typ = is_dP * dim
            hP_ = _xP_[is_dP]
            while hP_:
                hP, frame = form_segment(hP_.popleft(), frame, typ)
                frame = form_blob(hP, frame, typ)

            typ += 1
            hP_ = _yP_[is_dP]
            while hP_:
                hP, frame = form_segment(hP_.popleft(), frame, typ)
                frame = form_blob(hP, frame, typ)

            typ += 1
            hP_ = _tP_[is_dP]
            while hP_:
                hP, frame = form_segment(hP_.popleft(), frame, typ)
                frame = form_blob(hP, frame, typ)

        _frame = frame

    # sequence ends, incomplete ders3__ discarded, but vertically incomplete blobs are still inputted in scan_blob_?

    cv2.destroyAllWindows()  # Part of video read
    return videoo  # frame of 2D patterns is outputted to level 2
    # ---------- video_to_tblobs() end ----------------------------------------------------------------------------------


# ************ MAIN FUNCTIONS END ***************************************************************************************


# ************ PROGRAM BODY *********************************************************************************************

# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:

t_rng = 3  # Number of pixels compared to each pixel in time D
rng = 2  # Number of pixels compared to each pixel in four directions
min_coord = rng * 2 - 1  # min x and y for form_P input: ders2 from comp over rng*2 (bidirectional: before and after pixel p)
t_min_coord = t_rng * 2 - 1  # min t for form_P input: ders3 from comp over t_rng*2 (bidirectional: before and after pixel p)
ave = 15  # |d| value that coincides with average match: mP filter
ave_rate = 0.25  # not used; match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)
dim = 3  # Number of dimensions: x, y and t
record = bool(0)  # Set to True yield file outputs

# Load inputs --------------------------------------------------------------------
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-v', '--video', help='path to video file', default='./videos/Test01.avi')
arguments = vars(argument_parser.parse_args())
video = cv2.VideoCapture(arguments['video'], 0)

line_ = fetch_frame(video)
Y, X = line_.shape  # image height and width
T = 10  # number of frame read limit

# Main ---------------------------------------------------------------------------
start_time = time()
persistents = video_to_tblobs(video)
end_time = time() - start_time
print(end_time)

# ************ PROGRAM BODY END ******************************************************************************************