import cv2
import argparse
import numpy as np
from scipy import misc
from time import time
from collections import deque

''' This file is currently just a stab.
    Comparison over a sequence frames in a video, currently only initial ders-per-pixel tuple formation: 

    immediate pixel comparison to rng consecutive pixels over lateral x, vertical y, temporal t coordinates,
    then resulting 3D tuples (p, dx, mx, dy, my, dt, mt) per pixel are combined into 

    incremental-dimensionality patterns: 1D Ps ) 2D blobs ) TD persists, not oriented for inclusion? 
    evaluated for orientation, re-composition, incremental-dimensionality comparison, and recursion? 

    recursive input scope unroll: .multiple ( integer ( binary, accessed if hLe match * lLe total, 
    comp power = depth of content: normalized by hLe pwr miss if hLe diff * hLe match * lLe total
    3rd comp to 3rd-level ff -> filter pattern: longer-range nP forward, binary trans-level fb:
    complemented P: longer-range = higher-level ff & higher-res fb, recursion eval for positive Ps?

    colors will be defined as color / sum-of-colors, and single-color patterns are within grey-scale patterns: 
    primary white patterns ( sub-patterns per relative color, not cross-compared: already complementary?
'''


# ************ UTILITY FUNCTIONS ****************************************************************************************
# Includes:
# -rebuild_blobs()
# -segment_sort_by_height()
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


# ************ UTILITY FUNCTIONS END ************************************************************************************

# ************ MAIN FUNCTIONS *******************************************************************************************
# Includes:
# -lateral_comp()
# -vertical_comp()
# -form_P()
# -scan_P_()
# -form_segment()
# -form_blob()
# -spatial_eval()
# -video_to_persistents()
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
    ders1_.append((0, 0, 0))  # A trick for last P to get last ders on a line, in form_P()
    return ders1_
    # ---------- lateral_comp() end -------------------------------------------------------------------------------------

def vertical_comp(ders1_, tders_, ders2__, _P_, frame):
    " Comparison to bilateral rng of vertically consecutive pixels, forming ders2: pixel + lateral and vertical derivatives"

    # lateral pattern = pri_s, x0, L, I, D, Dy, V, Vy, ders2_
    P = [[0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []],
         [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []],
         [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []],
         [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []],
         [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []],
         [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]]
    P_ = [deque(), deque(), deque(), deque(), deque(), deque()]  # line y - 1 + rng*2
    buff_ = [deque(), deque(), deque(), deque(), deque(), deque()]  # line y - 2 + rng*2: _Ps buffered by previous run of scan_P_
    # Note: each one of P, P_, buff_ is a list of 4 corresponding to 4 types of patterns ( m, my, d, dy respectively )
    new_ders2__ = deque()  # 2D array: line of ders2_s buffered for next-line comp
    max_index = rng - 1  # max ders2_ index
    min_coord = rng * 2 - 1  # min x and y for form_P input: ders2 from comp over rng*2 (bidirectional: before and after pixel p)
    x = rng  # lateral coordinate of pixel in input ders1

    for (p, d, m),(dt, mt), ders2_ in zip(ders1_, tders_, ders2__):  # pixel comp to rng _pixels in ders2_, summing dy and my
        index = 0
        back_dy, back_my = 0, 0
        for (_p, _d, fdy, _dt, _m, fmy, _mt) in ders2_:  # vertical derivatives are incomplete; prefix '_' denotes higher-line variable
            dy = p - _p
            my = ave - abs(dy)
            fdy += dy  # running sum of differences between pixel _p and all higher and lower pixels within rng
            fmy += my  # running sum of matches between pixel _p and all higher and lower pixels within rng
            back_dy += dy
            back_my += my  # running sum of d and m between pixel _p and all higher pixels within rng

            if index < max_index:
                ders2_[index] = (_p, _d, fdy, _dt, _m, fmy, _mt)
            elif y > min_coord + ini_y:
                ders = _p, _d, fdy, _dt, _m, fmy, _mt
                for typ in range(0, 6, 2):  # vertes of P: dP | dyP | mP | myP
                    P, P_, buff_, _P_, frame = form_P(ders, x, X - rng, P, P_, buff_, _P_, frame, typ)
            index += 1

        ders2_.appendleft((p, d, back_dy, dt, m, back_my, mt))  # new ders2 displaces completed one in vertical ders2_ via maxlen
        new_ders2__.append(ders2_)  # 2D array of vertically-incomplete 2D tuples, converted to ders2__, for next-line vertical comp
        x += 1

    # Unlike mPs, dPs are disjointed, and the algorithm only deal with continuous Ps
    # To deal with disjointed dPs, 2 cases must be solved: not overlapping P - hP and unfinished hPs
    # Unfinished hPs are handled here:
    for typ in range(1, 6, 2):
        while buff_[typ]:
            hP = buff_[typ].popleft()
            if hP[1] != 1:
                frame = form_blob(hP, frame, typ)
        while _P_[typ]:
            hP, frame = form_segment(_P_[typ].popleft(), frame, typ)
            frame = form_blob(hP, frame, typ)

    return new_ders2__, P_, frame
    # ---------- vertical_comp() end ------------------------------------------------------------------------------------

def form_P(ders, x, max_x, P, P_, buff_, hP_, frame, typ):
    " Initializes, accumulates, and terminates 1D pattern "

    p, d, dy, dt, m, my, mt = ders  # 2D tuple of derivatives per pixel, "y" denotes vertical vs. lateral derivatives
    if typ == 0:    core = m; alt_der = d; alt_dir = my; alt_both = dy
    elif typ == 1:  core = d; alt_der = m; alt_dir = dy; alt_both = my
    elif typ == 2:  core = my; alt_der = dy; alt_dir = m; alt_both = d
    elif typ == 3:  core = dy; alt_der = my; alt_dir = d; alt_both = m
    elif typ == 4:  core = mt; alt_der = dt; alt_dir = 0; alt_both = 0  # Sorry I missed the part where you told me
    else:            core = dt; alt_der = mt; alt_dir = 0; alt_both = 0  # how to assign alt_dir and alt_both

    s = 1 if core > 0 else 0
    pri_s, x0, L, I, D, Dy, M, My, alt_Der, alt_Dir, alt_Both, ders_ = P[typ]

    if not (s == pri_s or x == x0) or x == max_x:  # P is terminated
        if typ < 2 and not pri_s:  # dPs formed inside of negative mP. typ < 2: mPs. not pri_s: negative pattern
            P[typ + 1] = [-1, x0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]
            ders_.append((0, 0, 0, 0, 0, 0, 0))  # A trick for last dP to get last ders inside an mP

            for i in range(L + 1):
                P, P_, buff_, _P_, frame = form_P(ders_[i], x0 + i, x0 + L, P, P_, buff_, hP_, frame, typ + 1)

            ders_.pop()  # pop the empty last ders added above

            P[typ] = pri_s, x0, L, I, D, Dy, M, My, alt_Der, alt_Dir, alt_Both, ders_, P_[typ + 1]

        if y == rng * 2 + ini_y:  # 1st line: form_P converts P to initialized hP, forming initial P_ -> hP_
            P_[typ].append([P[typ], 0, [], x - 1])  # P, roots, _fork_, x
        else:
            P_[typ], buff_[typ], hP_[typ], frame = scan_P_(x - 1, P[typ], P_[typ], buff_[typ], hP_[typ], frame, typ)  # scans higher-line Ps for contiguity
            # x-1 for prior p
        x0, L, I, D, Dy, M, My, alt_Der, alt_Dir, alt_Both, ders_ = x, 0, 0, 0, 0, 0, 0, 0, 0, 0, []  # new P initialization

    L += 1  # length of a pattern, continued or initialized input and derivatives are accumulated:
    I += p  # summed input
    D += d  # lateral D
    Dy += dy  # vertical D
    M += m  # lateral M
    My += my  # vertical M
    alt_Der += abs(alt_der)  # abs alt cores indicate value of alt-core Ps, to compute P redundancy rate
    alt_Dir += abs(alt_dir)  # vs. specific overlaps: cost > gain in precision?
    alt_Both += abs(alt_both)

    # in frame_x_blobs, alt_Der and alt_Both are computed for comp_P eval, but add to rdn only within neg mPs

    ders_.append(ders)  # ders2s are buffered for oriented rescan and incremental range | derivation comp
    P[typ] = s, x0, L, I, D, Dy, M, My, alt_Der, alt_Dir, alt_Both, ders_

    return P, P_, buff_, hP_, frame  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_P() end -------------------------------------------------------------------------------------------


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
    [s, first_x], params = _P[:2], list(_P[2:11])
    ave_x = (_P[2] - 1) // 2  # extra-x L = L-1 (1x in L)

    if not fork_:  # seg is initialized with initialized blob (params, coordinates, remaining_roots, root_, xD)
        blob = [[s, 0, 0, 0, 0, 0, 0, 0, 0, 0], [_P[1], hP[3], y - rng - 1, 0, 0], 1, []]
        hP = [params, roots, fork_, ave_x, 0, [(_P, 0)], blob]
        blob[3].append(hP)
    else:
        if len(fork_) == 1 and fork_[0][1] == 1:  # hP has one fork: hP[2][0], and that fork has one root: hP
            # hP is merged into higher-line blob segment (Pars, roots, _fork_, ave_x, xD, Py_, blob) at hP[2][0]:
            fork = fork_[0]
            L, I, D, Dy, M, My, alt0, alt1, alt2 = params
            Ls, Is, Ds, Dys, Ms, Mys, alt0s, alt1s, alt2s = fork[0]  # seg params
            fork[0] = [Ls + L, Is + I, Ds + D, Dys + Dy, Ms + M, Mys + My, alt0s + alt0, alt1s + alt1, alt2s + alt2]
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
                        [s, L, I, D, Dy, M, My, alt0, alt1, alt2], [min_x, max_x, min_y, xD, Ly], remaining_roots, root_ = fork[6]
                        blob[0][1] += L
                        blob[0][2] += I
                        blob[0][3] += D
                        blob[0][4] += Dy
                        blob[0][5] += M
                        blob[0][6] += My
                        blob[0][7] += alt0
                        blob[0][8] += alt1
                        blob[0][9] += alt2
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

    [L, I, D, Dy, M, My, alt0, alt1, alt2], roots, fork_, x, xD, Py_, blob = term_seg  # unique blob in fork_[0][6] is ref'd by other forks
    blob[0][1] += L
    blob[0][2] += I
    blob[0][3] += D
    blob[0][4] += Dy
    blob[0][5] += M
    blob[0][6] += My
    blob[0][7] += alt0
    blob[0][8] += alt1
    blob[0][9] += alt2

    blob[1][3] += xD  # ave_x angle, to evaluate blob for re-orientation
    blob[1][4] += len(Py_)  # Ly = number of slices in segment

    blob[2] += roots - 1  # reference to term_seg is already in blob[9]
    term_seg.append(y - rng - 1 - y_carry)  # y_carry: min elevation of term_seg over current hP

    if not blob[2]:  # if remaining_roots == 0: blob is terminated and packed in frame
        [s, L, I, D, Dy, M, My, alt0, alt1, alt2], [min_x, max_x, min_y, xD, Ly], remaining_roots, root_ = blob
        if not typ:  # frame P are to compute averages, redundant for same-scope alt_frames
            if not s: frame[0][0][0] += L  # L of negative horizontal mblobs are summed
            frame[0][1] += I
            frame[0][2] += D
            frame[0][3] += Dy
            frame[0][4] += M
            frame[0][5] += My
        elif not s:
            if typ == 2:
                frame[0][0][1] += L  # L of negative vertical mblobs are summed
            if typ == 4:
                frame[0][0][2] += L  # L of negative temporal mblobs are summed

        frame[typ + 1][0] += xD  # ave_x angle, to evaluate frame for re-orientation
        frame[typ + 1][1] += Ly  # +L
        root_.sort(key=segment_sort_by_height)  # Sort segments by max_y
        frame[typ + 1][2].append(((s, L, I, D, Dy, M, My, alt0, alt1, alt2), (min_x, max_x, min_y, term_seg[7], xD, Ly), root_))

    return frame  # no term_seg return: no root segs refer to it
    # ---------- form_blob() end ----------------------------------------------------------------------------------------

def spatial_eval(image, tders__, __frame, persistents):
    ''' Handle spatial comparisons and form 2D blobs,
    postfix '_' denotes array vs. element,
    prefix '_' denotes higher-line vs. lower-line variable,
    prefix '__' denotes prior-frame vs. post-frame variable '''

    _P_ = [deque(), deque(), deque(), deque(), deque(), deque()]  # higher-line same- d-, m-, dy-, my- sign 1D patterns
    frame = [[[0, 0, 0], 0, 0, 0, 0, 0], [0, 0, []], [0, 0, []], [0, 0, []], [0, 0, []], [0, 0, []], [0, 0, []]]
    # [[neg_mL, neg_myL, neg_mtL], I, D, Dy, M, My], 4 x [xD, Ly, blob_]

    global y
    y = ini_y  # initial line
    ders2__ = []  # horizontal line of vertical buffers: 2D array of 2D tuples, deque for speed?
    tders_ = tders__[ini_y][:]
    pixel_ = image[ini_y, :]  # first line of pixels at y == 0
    ders1_ = lateral_comp(pixel_)  # after partial comp, while x < rng?

    for (p, dx, mx),( dt, mt ) in zip(ders1_, tders_):
        ders2 = p, dx, 0, dt, mx, 0, mt # dy, my initialized at 0
        ders2_ = deque(maxlen=rng)  # vertical buffer of incomplete derivatives tuples, for fuzzy ycomp
        ders2_.append(ders2)  # only one tuple in first-line ders2_
        ders2__.append(ders2_)

    for y in range(ini_y + 1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?
        tders_ = tders__[ini_y][:]
        pixel_ = image[y, :]  # vertical coordinate y is index of new line p_
        ders1_ = lateral_comp(pixel_)  # lateral pixel comparison
        ders2__, _P_, frame = vertical_comp(ders1_, tders_, ders2__, _P_, frame)  # vertical pixel comparison

    # frame ends, last vertical rng of incomplete ders2__ is discarded,
    # merge segs of last line into their blobs:
    y = Y
    for typ in range(6):
        hP_ = _P_[typ]
        while hP_:
            hP, frame = form_segment(hP_.popleft(), frame, typ)
            frame = form_blob(hP, frame, typ)

    return frame, persistents
    # ---------- spatial_eval() end -------------------------------------------------------------------------------------

def video_to_persistents(vid):
    ''' Main body of the operation,
    postfix '_' denotes array vs. element,
    prefix '_' denotes higher-line vs. lower-line variable,
    prefix '__' denotes prior-frame vs. post-frame variable '''

    if record:
        mxblob_o = cv2.VideoWriter("./videos/mxblobs.avi", -1, 15, (X, Y))
        dxblob_o = cv2.VideoWriter("./videos/dxblobs.avi", -1, 15, (X, Y))
        myblob_o = cv2.VideoWriter("./videos/myblobs.avi", -1, 15, (X, Y))
        dyblob_o = cv2.VideoWriter("./videos/dyblobs.avi", -1, 15, (X, Y))
        mtblob_o = cv2.VideoWriter("./videos/mtblobs.avi", -1, 15, (X, Y))
        dtblob_o = cv2.VideoWriter("./videos/dtblobs.avi", -1, 15, (X, Y))

    # Initializations
    frame = [[[0, 0, 0], 0, 0, 0, 0, 0], [0, 0, []], [0, 0, []], [0, 0, []], [0, 0, []], [0, 0, []], [0, 0, []]]
    persistents = deque()   # Just a list of tblobs for now
    global t    # init t

    tders___ = deque(maxlen= t_rng)
    tders___.append((np.zeros((Y, X)), np.zeros((Y, X)), np.zeros((Y, X))))   # (p__, dt__, mt__)
    max_index = t_rng - 1

    t = 0
    while (vid.isOpened() and t < max_t):
        # Capture frame-by-frame ---------------------------------------------------------
        ret, image = vid.read()
        if not ret:
            break

        print 'Current frame: %d\n' % (t)

        # Main operations ----------------------------------------------------------------
        pixel__ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Array of gray-scale pixels
        back_dt__ = np.zeros((Y, X)); back_mt__ = np.zeros((Y, X))
        for index, (p__, dt__, mt__) in enumerate(tders___):
            idt__ = pixel__ - p__
            imt__ = ave - np.absolute(idt__)  # pixel-wise comparison

            dt__ += idt__
            mt__ += imt__

            back_dt__ += idt__
            back_mt__ += imt__

            if index < max_index:
                tders___[index] = (p__, dt__, mt__)
            elif t >= t_rng * 2:
                tders__ = [ zip(dt_, mt_) for dt_, mt_ in zip(dt__[:, rng:-rng + 1], mt__[:, rng:-rng + 1]) ] # For compatible with lateral comp ders
                frame, persistents = spatial_eval(p__, tders__, frame, persistents)  # spatial pixel comparison + partial blob comparison

        tders___.append((pixel__, back_dt__, back_mt__))

        # Rebuild blob -------------------------------------------------------------------
        if record:
            mxblob_o.write(np.uint8(rebuild_blobs(frame[1])))
            dxblob_o.write(np.uint8(rebuild_blobs(frame[2])))
            myblob_o.write(np.uint8(rebuild_blobs(frame[3])))
            dyblob_o.write(np.uint8(rebuild_blobs(frame[4])))
            mtblob_o.write(np.uint8(rebuild_blobs(frame[5])))
            dtblob_o.write(np.uint8(rebuild_blobs(frame[6])))

        t += 1

    cv2.destroyAllWindows()
    if record:
        mxblob_o.release()
        dxblob_o.release()
        myblob_o.release()
        dyblob_o.release()
        mtblob_o.release()
        dtblob_o.release()

    return persistents
    # ---------- video_to_persistents() end -----------------------------------------------------------------------------

# ************ MAIN FUNCTIONS END ***************************************************************************************


# ************ PROGRAM BODY *********************************************************************************************

# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:

t_rng = 3   # Number of pixels compared to each pixel in time D
rng = 2     # Number of pixels compared to each pixel in four directions
ave = 15    # |d| value that coincides with average match: mP filter
ave_rate = 0.25  # not used; match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)
ini_y = 0   # not used
max_t = 20  # For testing
record = bool(0)  # Set to True yield file outputs

# Load inputs --------------------------------------------------------------------
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-v', '--video', help='path to video file', default='./videos/Test01.avi')
arguments = vars(argument_parser.parse_args())
video = cv2.VideoCapture(arguments['video'], 0)

ret, image = video.read()
Y, X, _ = image.shape  # image height and width

# Main ---------------------------------------------------------------------------
start_time = time()
persistents = video_to_persistents( video )
end_time = time() - start_time
print(end_time)

# ************ PROGRAM BODY END ******************************************************************************************