import cv2
import argparse
import numpy as np
from scipy import misc
from time import time
from collections import deque


''' Temporal blob composition over a sequence of frames in a video: 
    pixel comparison to rng consecutive pixels over lateral x, vertical y, temporal t coordinates,
    resulting 3D tuples are combined into incremental-dimensionality 1D Ps ) 2D blobs ) 3D tblobs.     	    
    
    Selective temporal forking due to blob scale variation within a frame * temporal variation between frames.
    Selection is by top-level comp / top-level scan: inclusion if rel_olp * mL > ave * max(L, _L) 
    Only top-level because variation is far lower than between discontinuous Ps
    
    tblobs will be evaluated for orientation and incremental-dimensionality intra-tblob comparison
'''

# ************ MISCELLANEOUS FUNCTIONS **********************************************************************************
# Includes:
# -rebuild_segment()
# -rebuild_blob()
# -rebuild_frame()
# -fetch_frame()
# ***********************************************************************************************************************


def rebuild_segment(dir, index, seg, blob_img, frame_img, print_separate_blobs=0, print_separate_segs=0):
    if print_separate_segs: seg_img = np.array([[[127] * 4] * X] * Y)
    y = seg[7][2]  # min_y
    for (P, xd) in seg[5]:
        x = P[1]
        for i in range(P[2]):
            frame_img[y, x, : 3] = [255, 255, 255] if P[0] else [0, 0, 0]
            if print_separate_blobs: blob_img[y, x, : 3] = [255, 255, 255] if P[0] else [0, 0, 0]
            if print_separate_segs: seg_img[y, x, : 3] = [255, 255, 255] if P[0] else [0, 0, 0]
            x += 1
        y += 1

    if print_separate_segs:
        min_x, max_x, min_y, max_y = seg[7]
        cv2.rectangle(seg_img, (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), (0, 255, 255), 1)
        cv2.imwrite(dir + 'seg%d.jpg' % (index), seg_img)
    return blob_img
    # ---------- rebuild_segment() end ----------------------------------------------------------------------------------


def rebuild_blob(dir, index, blob, frame_img, print_separate_blobs=0, print_separate_segs=0):
    " Rebuilt data of a blob into an image "
    if print_separate_blobs: blob_img = np.array([[[127] * 4] * X] * Y)
    for ids, id in enumerate(blob[4][0]):  # Iterate through segments' sorted id
        blob_img = rebuild_segment(dir + '/blob%d' % (index), ids, blob[3][id], blob_img, frame_img, print_separate_blobs, print_separate_segs)

    if print_separate_blobs:
        min_x, max_x, min_y, max_y = blob[1][:4]
        cv2.rectangle(blob_img, (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), (0, 255, 255), 1)
        cv2.imwrite(dir + '/blob%d.jpg' % (index), blob_img)
    return frame_img
    # ---------- rebuild_blob() end -------------------------------------------------------------------------------------


def rebuild_frame(dir, frame, print_separate_blobs=0, print_separate_segs=0):
    " Rebuilt data of a frame into an image "
    frame_img = np.array([[[127] * 4] * X] * Y)
    if (print_separate_blobs or print_separate_segs) and not os.path.exists(dir):
        os.mkdir(dir)
    for indices, index in enumerate(frame[3][0]):  # Iterate through blobs' sorted id
        frame_img = rebuild_blob(dir, indices, frame[2][index], frame_img, print_separate_blobs, print_separate_segs)
    cv2.imwrite(dir + '.jpg', frame_img)
    # ---------- rebuild_frame() end ------------------------------------------------------------------------------------

def bin_search(blob_, atb, i, j0, j, target, take_right=0, rdepth=0):  # take_right -> right_olp, rdepth -> right_olp_L?
    ''' a binary search module:
        - search in: pri_blob, i
        - search for: j
        - search condition: pri_blob[i[j-1]][1][0] <= target < pri_blob[i[j]][1][0] '''
    if target + take_right <= blob_[i[j0]][1][atb]:
        return j0
    elif blob_[i[j - 1]][1][atb] < target + take_right:  # right_olp?
        return j
    else:
        jm = (j0 + j) // 2
        if blob_[i[jm]][1][atb] < target + take_right:
            return bin_search(blob_, atb, i, jm, j, target, take_right, rdepth + 1)
        else:
            return bin_search(blob_, atb, i, j0, jm, target, take_right, rdepth + 1)
    # ---------- bin_search() end ---------------------------------------------------------------------------------------

def fetch_frame(video):
    " Short call to read a gray-scale frame"
    _, frame = video.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('int32')

# ************ MISCELLANEOUS FUNCTIONS END ******************************************************************************

# ************ MAIN FUNCTIONS *******************************************************************************************
# Includes:
# -lateral_comp()
# -vertical_comp()
# -temporal_comp()
# -form_P()
# -scan_P_()
# -form_segment()
# -form_blob()
# -scan_blob_()
# -video_to_tblobs()
# ***********************************************************************************************************************
def lateral_comp(pixel_):
    # Comparison over x coordinate, within rng of consecutive pixels on each line

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
            back_fd += d  # running sum of d between pixel and all prior pixels within rng
            back_fm += m  # running sum of m between pixel and all prior pixels within rng

            if index < max_index:
                rng_ders1_[index] = (pri_p, fd, fm)
            elif x > min_coord:  # after pri_p comp over full bilateral rng
                ders1_.append((pri_p, fd, fm))  # completed bilateral tuple is transferred from rng_ders_ to ders_

        rng_ders1_.appendleft((p, back_fd, back_fm))  # new tuple with initialized d and m, maxlen displaces completed tuple
    # last incomplete rng_ders1_ in line are discarded, vs. ders1_ += reversed(rng_ders1_)
    return ders1_
    # ---------- lateral_comp() end -------------------------------------------------------------------------------------


def vertical_comp(ders1_, rng_ders2__):
    # Comparison to bilateral rng of vertically consecutive pixels forms ders2: pixel + lateral and vertical derivatives

    ders2_ = []  # line of tuples with complete 2D derivatives: summation range = rng
    new_rng_ders2__ = deque()  # 2D array: line of ders2_s buffered for next-line comp
    max_index = rng - 1  # max ders2_ index
    x = rng  # lateral coordinate of pixel in input ders1

    for (p, dx, mx), rng_ders2_ in zip(ders1_, rng_ders2__):  # pixel comp to rng _pixels in rng_ders2_, summing dy and my
        index = 0
        back_dy, back_my = 0, 0
        for (_p, _dx, fdy, _mx, fmy) in rng_ders2_:  # vertical derivatives are incomplete; prefix '_' denotes higher-line variable
            dy = p - _p
            my = ave - abs(dy)
            fdy += dy  # running sum of differences between pixel _p and all higher and lower pixels within rng
            fmy += my  # running sum of matches between pixel _p and all higher and lower pixels within rng
            back_dy += dy  # running sum of dy between pixel p and all higher pixels within rng
            back_my += my  # running sum of my between pixel p and all higher pixels within rng
            if index < max_index:
                rng_ders2_[index] = (_p, _dx, fdy, _mx, fmy)
            elif y > min_coord:
                ders2_.append((_p, _dx, fdy, _mx, fmy))  # completed bilateral tuple is transferred from rng_ders2_ to ders2_
            index += 1

        rng_ders2_.appendleft((p, dx, back_dy, mx, back_my))  # new ders2 displaces completed one in vertical ders2_ via maxlen
        new_rng_ders2__.append(rng_ders2_)  # 2D array of vertically-incomplete 2D tuples, converted to rng_ders2__, for next line
        x += 1

    return ders2_, new_rng_ders2__
    # ---------- vertical_comp() end ------------------------------------------------------------------------------------


def temporal_comp(ders2_, rng_ders3___, _xP_, _yP_, _tP_, frame, _frame, videoo):
    # ders2_: input line of complete 2D ders
    # rng_ders3___: prior frame of incomplete 3D ders buffers, sliced into lines
    # comparison between t_rng temporally consecutive pixels, forming ders3: 3D tuple of derivatives per pixel

    # each of the following contains 2 types, per core variables m and d:
    xP = [[0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []],  # lateral pattern = pri_s, x0, L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0:6 ders2_
          [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]]
    yP = [[0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []],
          [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]]
    tP = [[0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []],
          [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]]
    xP_ = [deque(), deque()]
    yP_ = [deque(), deque()]  # line y - rng
    tP_ = [deque(), deque()]
    xbuff_ = [deque(), deque()]
    ybuff_ = [deque(), deque()]  # line y - rng - 1: _Ps buffered by previous run of scan_P_
    tbuff_ = [deque(), deque()]

    rng_ders3__ = rng_ders3___.pop(0)
    new_rng_ders3__ = deque()  # 2D array: line of rng_ders3_s buffered for next-frame comp
    max_index = t_rng - 1  # max rng_ders3_ index
    x = rng  # lateral coordinate of pixel

    for (p, dx, dy, mx, my), rng_ders3_ in zip(ders2_, rng_ders3__):  # pixel comp to rng _pixels in rng_ders3_, summing dt and mt
        index = 0
        back_dt, back_mt = 0, 0
        for (_p, _dx, _dy, fdt, _mx, _my, fmt) in rng_ders3_:  # temporal derivatives are incomplete; prefix '_' denotes previous-frame variable
            dt = p - _p
            mt = ave - abs(dt)
            fdt += dt   # running sum of differences between pixel _p and all previous and subsequent pixels within t_rng
            fmt += mt   # running sum of matches between pixel _p and all previous and subsequent pixels within t_rng
            back_dt += dt   # running sum of dt between pixel p and all previous pixels within t_rng
            back_mt += mt   # running sum of mt between pixel p and all previous pixels within t_rng

            if index < max_index:
                rng_ders3_[index] = (_p, _dx, _dy, fdt, _mx, _my, fmt)
            elif t > t_min_coord:
                ders = _p, _dx, _dy, fdt, _mx, _my, fmt
                xP, xP_, xbuff_, _xP_, frame, _frame, videoo = form_P(ders, x, X - rng - 1, xP, xP_, xbuff_, _xP_, frame, _frame, videoo, 0)  # mxP: typ = 0
                yP, yP_, ybuff_, _yP_, frame, _frame, videoo = form_P(ders, x, X - rng - 1, yP, yP_, ybuff_, _yP_, frame, _frame, videoo, 1)  # myP: typ = 1
                tP, tP_, tbuff_, _tP_, frame, _frame, videoo = form_P(ders, x, X - rng - 1, tP, tP_, tbuff_, _tP_, frame, _frame, videoo, 2)  # mtP: typ = 2
            index += 1

        rng_ders3_.appendleft((p, dx, dy, back_dt, mx, my, back_mt))  # new ders3 displaces completed one in temporal rng_ders3_ via maxlen
        new_rng_ders3__.append(rng_ders3_)  # rng_ders3__: line of incomplete ders3 buffers, to be added to next-frame rng_ders3___
        x += 1

    typ = dim  # terminate last higher line dxP (typ = 3) within neg mxPs
    while xbuff_[1]:
        hP = xbuff_[1].popleft()
        if hP[1] != 1:  # no roots
            frame, _frame, videoo = form_blob(hP, frame, _frame, videoo, typ)
    while _xP_[1]:
        hP, frame, _frame, videoo = form_segment(_xP_[1].popleft(), frame, _frame, videoo, typ)
        frame, _frame, videoo = form_blob(hP, frame, _frame, videoo, typ)

    typ += 1  # terminate last higher line dyP (typ = 4) within neg myPs
    while ybuff_[1]:
        hP = ybuff_[1].popleft()
        if hP[1] != 1:  # no roots
            frame, _frame, videoo = form_blob(hP, frame, _frame, videoo, typ)
    while _yP_[1]:
        hP, frame, _frame, videoo = form_segment(_yP_[1].popleft(), frame, _frame, videoo, typ)
        frame, _frame, videoo = form_blob(hP, frame, _frame, videoo, typ)

    typ += 1  # terminate last higher line dtP (typ = 5) within neg mtPs
    while tbuff_[1]:
        hP = tbuff_[1].popleft()
        if hP[1] != 1:  # no roots
            frame, _frame, videoo = form_blob(hP, frame, _frame, videoo, typ)
    while _tP_[1]:
        hP, frame, _frame, videoo = form_segment(_tP_[1].popleft(), frame, _frame, videoo, typ)
        frame, _frame, videoo = form_blob(hP, frame, _frame, videoo, typ)

    rng_ders3___.append(new_rng_ders3__)  # rng_ders3___ for next frame

    return rng_ders3___, xP_, yP_, tP_, frame, _frame, videoo

    # ---------- temporal_comp() end ------------------------------------------------------------------------------------


def form_P(ders, x, max_x, P, P_, buff_, hP_, frame, _frame, videoo, typ, is_dP=0):
    # Initializes and accumulates 1D pattern
    # is_dP = bool(typ // dim), computed directly for speed and clarity:

    p, dx, dy, dt, mx, my, mt = ders  # 3D tuple of derivatives per pixel, "x" for lateral, "y" for vertical, "t" for temporal
    if     typ == 0:   core = mx; alt0 = dx; alt1 = my; alt2 = mt; alt3 = dy; alt4 = dt
    elif   typ == 1:   core = my; alt0 = dy; alt1 = mx; alt2 = mt; alt3 = dx; alt4 = dt
    elif   typ == 2:   core = mt; alt0 = dt; alt1 = mx; alt2 = my; alt3 = dx; alt4 = dy
    elif   typ == 3:   core = dx; alt0 = mx; alt1 = dy; alt2 = dt; alt3 = my; alt4 = mt
    elif   typ == 4:   core = dy; alt0 = my; alt1 = dx; alt2 = dt; alt3 = mx; alt4 = mt
    else:              core = dt; alt0 = mt; alt1 = dx; alt2 = dy; alt3 = mx; alt4 = my

    s = 1 if core > 0 else 0
    pri_s, x0 = P[is_dP][:2]  # P[0] is mP, P[1] is dP
    if not (s == pri_s or x == x0):  # P is terminated
        P, P_, buff_, hP_, frame, _frame, videoo = term_P(s, x, P, P_, buff_, hP_, frame, _frame, videoo, typ, is_dP)

    pri_s, x0, L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4, ders_ = P[is_dP]
    # Continued or initialized input and derivatives are accumulated:
    L += 1  # length of a pattern
    I += p  # summed input
    Dx += dx  # lateral D
    Dy += dy  # vertical D
    Dt += dt  # temporal D
    Mx += mx  # lateral M
    My += my  # vertical M
    Mt += mt  # temporal M
    Alt0 += abs(alt0)  # alternative derivative: m | d;   indicate value, thus redundancy rate, of overlapping alt-core blobs
    Alt1 += abs(alt1)  # alternative dimension:  x | y | t
    Alt2 += abs(alt2)  # second alternative dimension
    Alt3 += abs(alt3)  # alternative derivative and dimension
    Alt4 += abs(alt4)  # second alternative derivative and dimension

    ders_.append(ders)  # ders3s are buffered for oriented rescan and incremental range | derivation comp
    P[is_dP] = s, x0, L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4, ders_

    if x == max_x:  # P is terminated
        P, P_, buff_, hP_, frame, _frame, videoo = term_P(s, x + 1, P, P_, buff_, hP_, frame, _frame, videoo, typ, is_dP)

    return P, P_, buff_, hP_, frame, _frame, videoo  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_P() end -------------------------------------------------------------------------------------------


def term_P(s, x, P, P_, buff_, hP_, frame, _frame, videoo, typ, is_dP):
    # Terminates 1D pattern if sign change or P_ end

    pri_s, x0, L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4, ders_ = P[is_dP]
    if not is_dP and not pri_s:
        P[1] = [-1, x0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]  # dPs (P[1]) formed inside of negative mP (P[0])

        for i in range(L):
            P, P_, buff_, _P_, frame, _frame, videoo = form_P(ders_[i], x0 + i, x0 + L - 1, P, P_, buff_, hP_, frame, _frame, videoo, typ + dim, True)  # is_dP = 1
        P[0] = pri_s, x0, L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4, ders_, P_[1]

    if y == rng * 2:  # 1st line P_ is converted to init hP_;  scan_P_(), form_segment(), form_blob() use one type of Ps, hPs, buffs
        P_[is_dP].append([P[is_dP], 0, [], x - 1])  # P, roots, _fork_, x
    else:
        P_[is_dP], buff_[is_dP], hP_[is_dP], frame, _frame, videoo \
            = scan_P_(x - 1, P[is_dP], P_[is_dP], buff_[is_dP], hP_[is_dP], frame, _frame, videoo, typ)  # P scans hP_
    P[is_dP] = s, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []  # new P initialization

    return P, P_, buff_, hP_, frame, _frame, videoo
    # ---------- term_P() end -------------------------------------------------------------------------------------------


def scan_P_(x, P, P_, _buff_, hP_, frame, _frame, videoo, typ):
    # P scans shared-x-coordinate hPs in higher P_, combines overlapping Ps into blobs

    buff_ = deque()  # new buffer for displaced hPs (higher-line P tuples), for scan_P_(next P)
    fork_ = []  # refs to hPs connected to input P
    _x0 = 0  # to start while loop
    x0 = P[1]

    while _x0 <= x:  # while x values overlap between P and _P
        if _buff_:
            hP = _buff_.popleft()  # hP was extended to segment and buffered in prior scan_P_
        elif hP_:
            hP, frame, _frame, videoo = form_segment(hP_.popleft(), frame, _frame, videoo, typ)
        else:
            break  # higher line ends, all hPs are converted to segments
        roots = hP[1]
        _x0 = hP[5][-1][0][1]  # first_x
        _x = _x0 + hP[5][-1][0][2] - 1  # last_x = first_x + L - 1

        if P[0] == hP[6][0][0] and not _x < x0 and not x < _x0:  # P comb -> blob if s == _s, _last_x >= first_x and last_x >= _first_x
            roots += 1
            hP[1] = roots
            fork_.append(hP)  # P-connected hPs will be converted to segments at each _fork

        if _x > x:  # x overlap between hP and next P: hP is buffered for next scan_P_, else hP included in a blob segment
            buff_.append(hP)
        elif roots != 1:
            frame, _frame, videoo = form_blob(hP, frame, _frame, videoo, typ)  # segment is terminated and packed into its blob
        _x0 = _x + 1  # = first x of next _P

    buff_ += _buff_  # _buff_ is likely empty
    P_.append([P, 0, fork_, x])  # P with no overlap to next _P is extended to hP and buffered for next-line scan_P_

    return P_, buff_, hP_, frame, _frame, videoo  # hP_ and buff_ contain only remaining _Ps, with _x => next x
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_segment(hP, frame, _frame, videoo, typ):
    # Convert hP into new segment or add it to higher-line segment, merge blobs

    _P, roots, fork_, last_x = hP
    [s, first_x], params = _P[:2], list(_P[2:15])
    ave_x = (_P[2] - 1) // 2  # extra-x L = L-1 (1x in L)

    if not fork_:  # seg is initialized with initialized blob (params, coords, remaining_roots, root_, xD)
        blob = [[s, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [_P[1], hP[3], y - rng - 1, 0, 0], 1, []]
        hP = [params, roots, fork_, ave_x, 0, [(_P, 0)], blob, [first_x, last_x]]
        blob[3].append(hP)
    else:
        if len(fork_) == 1 and fork_[0][1] == 1:  # hP has one fork: hP[2][0], and that fork has one root: hP
            # hP is merged into higher-line blob segment (Pars, roots, _fork_, ave_x, xD, Py_, blob) at hP[2][0]:
            fork = fork_[0]
            L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4 = params
            Ls, Is, Dxs, Dys, Dts, Mxs, Mys, Mts, Alt0s, Alt1s, Alt2s, Alt3s, Alt4s = fork[0]  # seg params
            fork[0] = [Ls + L, Is + I, Dxs + Dx, Dys + Dy, Dts + Dt, Mxs + Mx, Mys + My, Mts + Mt,
                       Alt0s + Alt0, Alt1s + Alt1, Alt2s + Alt2, Alt3s + Alt3, Alt4s + Alt4]
            fork[1] = roots
            xd = ave_x - fork[3]
            fork[3] = ave_x
            fork[4] += xd  # xD for seg normalization and orientation, or += |dx| for curved yL?
            fork[5].append((_P, xd))  # Py_: vertical buffer of Ps merged into seg
            fork[7][0] = min(first_x, fork[7][0])
            fork[7][1] = max(last_x, fork[7][1])
            hP = fork  # replace segment with including fork's segment

        else:  # if >1 forks, or 1 fork that has >1 roots:
            hP = [params, roots, fork_, ave_x, 0, [(_P, 0)], fork_[0][6],[first_x, last_x]]  # seg is initialized with fork's blob
            blob = hP[6]
            blob[3].append(hP)  # segment is buffered into root_

            if len(fork_) > 1:  # merge blobs of all forks
                if fork_[0][1] == 1:  # if roots == 1
                    frame, _frame, videoo = form_blob(fork_[0], frame, _frame, videoo, typ, 1)  # merge seg of 1st fork into its blob

                for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                    if fork[1] == 1:
                        frame, _frame, videoo = form_blob(fork, frame, _frame, videoo, typ, 1)

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

    return hP, frame, _frame, videoo
    # ---------- form_segment() end -----------------------------------------------------------------------------------------


def form_blob(term_seg, frame, _frame, videoo, typ, y_carry=0):
    # Terminated segment is merged into continued or initialized blob (all connected segments)

    [L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4], roots, fork_, x, xD, Py_, blob, [min_x, max_x] = term_seg
    last_y = y - rng - 1 - y_carry  # yd: min elevation of term_seg over current hP
    first_y = last_y - len(Py_) + 1
    # unique blob in fork_[0][6] is referenced by other forks
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
    # blob[1][2] min_y changes only if blobs merge
    blob[1][3] += xD    # ave_x angle, to evaluate blob for re-orientation
    blob[1][4] += len(Py_)  # Ly = number of slices in segment
    blob[2] += roots - 1  # reference to term_seg is already in blob[9]
    term_seg[7] += [first_y, last_y]   # Add min_y and max_y to term_seg's coordinate

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
        sorted_root_id_ = [sorted(range(len(root_)), key=lambda i: root_[i][7][0]), # id of segments' sorted by min_x
                           sorted(range(len(root_)), key=lambda i: root_[i][7][1]), # id of segments' sorted by max_x
                           sorted(range(len(root_)), key=lambda i: root_[i][7][2]), # id of segments' sorted by min_y
                           sorted(range(len(root_)), key=lambda i: root_[i][7][3]),]# id of segments' sorted by max_y
        # lists of indices of sorted root_ are added to complete root,
        # blob[2] (== 0) will be used as troots
        blob[1] = [min_x, max_x, min_y, term_seg[7][3], xD, Ly]
        blob.append(sorted_root_id_)
        frame[typ + 1][2].append(blob)
        if t > t_rng * 2:
            frame, _frame, videoo = scan_blob_(blob, frame, _frame, videoo, typ)

    return frame, _frame, videoo  # no term_seg return: no root segs refer to it
    # ---------- form_blob() end ----------------------------------------------------------------------------------------

def scan_blob_(blob, frame, pri_frame, videoo, typ):
    # blob scans overlapping pri_blobs in pri_frame, forms forks if rolp * mL > ave * max(L, _L)

    [s, L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4], [min_x, max_x, min_y, max_y, xD, Ly], troots, root_, sorted_root_id_ = blob

    pri_blob_ = pri_frame[typ + 1][2] # list of same type pri_blobs
    _id_by_min_x_, _id_by_max_x_, _id_by_min_y_, _id_by_max_y_ = pri_frame[typ + 1][3]  # lists of indices sorted by pri_blobs min_x, max_x, min_y, max_y respectively

    # Search for boundaries of sorted pri_blobs that meet the prerequisites of overlapping with current frame blob
    _num_blobs = len(_id_by_min_x_)
    # Binary search:
    _min_x_id = bin_search(pri_blob_, 0, _id_by_min_x_, 0, _num_blobs, max_x, 1)    # bin_search(blob, atribute, sorted_indices_,
    _max_x_id = bin_search(pri_blob_, 1, _id_by_max_x_, 0, _num_blobs, min_x, 0)    # first_index, last_index, target, equal)
    _min_y_id = bin_search(pri_blob_, 2, _id_by_min_y_, 0, _num_blobs, max_y, 1)    #
    _max_y_id = bin_search(pri_blob_, 3, _id_by_max_y_, 0, _num_blobs, min_y, 0)    # (see iterative search below)

    _min_x_less_or_equal_max_x_indices = _id_by_min_x_[:_min_x_id]      # overlap prerequisite: _min_x <= max_x
    _min_y_less_or_equal_max_y_indices = _id_by_min_y_[:_min_y_id]      # overlap prerequisite: _min_y <= max_y
    _max_x_greater_or_equal_min_x_indices = _id_by_max_x_[_max_x_id:]   # overlap prerequisite: _max_x <= min_x
    _max_y_greater_or_equal_min_y_indices = _id_by_max_y_[_max_y_id:]   # overlap prerequisite: _max_y <= min_y

    # Set of overlapping pri_blobs is common subset of 4 sets that meet the 4 prerequisites
    olp_id_ = np.intersect1d(np.intersect1d(_min_x_less_or_equal_max_x_indices, _max_x_greater_or_equal_min_x_indices),
                             np.intersect1d(_min_y_less_or_equal_max_y_indices, _max_y_greater_or_equal_min_y_indices))


    # Iterative search:
    # _min_x_id = 0; _max_x_id = 0; _min_y_id = 0; _max_y_id = 0
    # while _min_x_id < _num_blobs and pri_blob_[ _id_by_min_x_[_min_x_id] ][1][0] <= max_x: _min_x_id += 1 # pri_blob_[i][1][0:4]: min_x, max_x, min_y, max_y index i pri_blob
    # while _max_x_id < _num_blobs and pri_blob_[ _id_by_max_x_[_max_x_id] ][1][1] < min_x: _max_x_id += 1  # _id_by_min[i]: index of blob that has i-th smallest min_x
    # while _min_y_id < _num_blobs and pri_blob_[ _id_by_min_y_[_min_y_id] ][1][2] <= max_y: _min_y_id += 1 #
    # while _max_y_id < _num_blobs and pri_blob_[ _id_by_max_y_[_max_y_id] ][1][3] < min_y: _max_y_id += 1  #
    # olp_id_ = np.intersect1d(np.intersect1d(_id_by_min_x_[:_min_x_id], _id_by_max_x_[_max_x_id:]),
    #                          np.intersect1d(_id_by_min_y_[:_min_y_id], _id_by_max_y_[_max_y_id:]))


    # For Debugging --------------------------------------------------------------
    # Print first blob formed in frame at t = t_rng * 2 +  and all it's overlapped blobs in previous frame
    global olp_debug
    if t == t_rng * 2 + 1 and len(olp_id_[:]) and olp_debug:
        filtered_hframe = np.array([[[127] * 4] * X] * Y)
        rebuild_blob('./images/', 0, blob, filtered_hframe, 1)
        for i in _id_by_min_x_[:_min_x_id]:
            rebuild_blob('./images/min_x', i, pri_blob_[i], filtered_hframe, 1)
        for i in _id_by_max_x_[_max_x_id:]:
            rebuild_blob('./images/max_x', i, pri_blob_[i], filtered_hframe, 1)
        for i in _id_by_min_y_[:_min_y_id]:
            rebuild_blob('./images/min_y', i, pri_blob_[i], filtered_hframe, 1)
        for i in _id_by_max_y_[_max_y_id:]:
            rebuild_blob('./images/max_y', i, pri_blob_[i], filtered_hframe, 1)
        for i in olp_id_:
            rebuild_blob('./images/olp', i, pri_blob_[i], filtered_hframe, 1)
        olp_debug = False
    # ----------------------------------------------------------------------------

    # for olp_id in olp_id_:
    #     pri_blob = pri_blob_[olp_id]
        # Partial comp between blob and pri_blob goes here

    return frame, pri_frame, videoo
    # ---------- scan_blob_() end ---------------------------------------------------------------------------------------

def video_to_tblobs(video):
    # Main body of the operation,
    # postfix '_' denotes array vs. element,
    # prefix '_' denotes prior- pixel, line, or frame variable

    # higher-line same- d-, m-, dy-, my- sign 1D patterns
    _xP_ = [deque(), deque()]
    _yP_ = [deque(), deque()]
    _tP_ = [deque(), deque()]

    # prior frame: [[neg_mL, neg_myL, neg_mtL], I, Dx, Dy, Dt, Mx, My, Mt], 6 x [xD, Ly, blob_]
    _frame = [[[0, 0, 0], 0, 0, 0, 0, 0, 0, 0],
              [0, 0, [], []],   # mxblobs
              [0, 0, [], []],   # myblobs
              [0, 0, [], []],   # mtblobs
              [0, 0, [], []],   # dxblobs
              [0, 0, [], []],   # dyblobs
              [0, 0, [], []]]   # dtblobs

    videoo = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [], [], [], [], [], []]  # output: [[Dxf, Lf, If, Dxf, Dyf, Dtf, Mxf, Myf, Mtf], 6 x tblob_]
    global t, y

    # Initialization:
    t = 0  # temporal coordinate of current frame
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
        frame = [[[0, 0, 0], 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, [], []],  # mxblobs
                  [0, 0, [], []],  # myblobs
                  [0, 0, [], []],  # mtblobs
                  [0, 0, [], []],  # dxblobs
                  [0, 0, [], []],  # dyblobs
                  [0, 0, [], []]]  # dtblobs
        line_ = fetch_frame(video)
        for y in range(0, Y):
            pixel_ = line_[y, :]
            ders1_ = lateral_comp(pixel_)  # lateral pixel comparison
            ders2_, rng_ders2__ = vertical_comp(ders1_, rng_ders2__)  # vertical pixel comparison
            if y > min_coord:  # temporal pixel comparison:
                rng_ders3___, _xP_, _yP_, _tP_, frame, _frame, videoo = temporal_comp(ders2_, rng_ders3___, _xP_, _yP_, _tP_, frame, _frame, videoo)

        y = Y  # merge segs of last line into their blobs:
        for is_dP in range(2):
            typ = is_dP * dim
            hP_ = _xP_[is_dP]
            while hP_:
                hP, frame, _frame, videoo = form_segment(hP_.popleft(), frame, _frame, videoo, typ)
                frame, _frame, videoo = form_blob(hP, frame, _frame, videoo, typ)
            # Sort blobs' indices based on min_x, max_x, min_y, max_y:
            blob_ = frame[typ + 1][2]
            blob_sorted_ = [sorted(range(len(blob_)), key=lambda i: blob_[i][1][0]), # id of segments' sorted by min_x
                            sorted(range(len(blob_)), key=lambda i: blob_[i][1][1]), # id of segments' sorted by max_x
                            sorted(range(len(blob_)), key=lambda i: blob_[i][1][2]), # id of segments' sorted by min_y
                            sorted(range(len(blob_)), key=lambda i: blob_[i][1][3]),]# id of segments' sorted by max_y
            frame[typ + 1][3] = blob_sorted_

            typ += 1
            hP_ = _yP_[is_dP]
            while hP_:
                hP, frame, _frame, videoo = form_segment(hP_.popleft(), frame, _frame, videoo, typ)
                frame, _frame, videoo = form_blob(hP, frame, _frame, videoo, typ)
            # Sort blobs' indices based on min_x, max_x, min_y, max_y:
            blob_ = frame[typ + 1][2]
            blob_sorted_ = [sorted(range(len(blob_)), key=lambda i: blob_[i][1][0]),  # id of segments' sorted by min_x
                            sorted(range(len(blob_)), key=lambda i: blob_[i][1][1]),  # id of segments' sorted by max_x
                            sorted(range(len(blob_)), key=lambda i: blob_[i][1][2]),  # id of segments' sorted by min_y
                            sorted(range(len(blob_)), key=lambda i: blob_[i][1][3]), ]  # id of segments' sorted by max_y
            frame[typ + 1][3] = blob_sorted_

            typ += 1
            hP_ = _tP_[is_dP]
            while hP_:
                hP, frame, _frame, videoo = form_segment(hP_.popleft(), frame, _frame, videoo, typ)
                frame, _frame, videoo = form_blob(hP, frame, _frame, videoo, typ)
            # Sort blobs' indices based on min_x, max_x, min_y, max_y:
            blob_ = frame[typ + 1][2]
            blob_sorted_ = [sorted(range(len(blob_)), key=lambda i: blob_[i][1][0]),  # id of segments' sorted by min_x
                            sorted(range(len(blob_)), key=lambda i: blob_[i][1][1]),  # id of segments' sorted by max_x
                            sorted(range(len(blob_)), key=lambda i: blob_[i][1][2]),  # id of segments' sorted by min_y
                            sorted(range(len(blob_)), key=lambda i: blob_[i][1][3]), ]  # id of segments' sorted by max_y
            frame[typ + 1][3] = blob_sorted_

        if record and t == frame_output_at: # change these in program body
            rebuild_frame('./images/mblobs_horizontal',frame[1], record_blobs, record_segs)
            rebuild_frame('./images/mblobs_vertical', frame[2], record_blobs, record_segs)
            rebuild_frame('./images/mblobs_temporal', frame[3], record_blobs, record_segs)
            rebuild_frame('./images/dblobs_horizontal', frame[4], record_blobs, record_segs)
            rebuild_frame('./images/dblobs_vertical', frame[5], record_blobs, record_segs)
            rebuild_frame('./images/dblobs_temporal', frame[6], record_blobs, record_segs)

        _frame = frame

    # sequence ends, incomplete ders3__ discarded, but vertically incomplete blobs are still inputted in scan_blob_?

    cv2.destroyAllWindows()  # Part of video read
    return videoo  # tblobs output to level 2
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
dim = 3  # Number of dimensions: x, y and t

# For outputs:
record = bool(0)  # Set to True yield file outputs
record_blobs = bool(1)
record_segs = bool(0)
frame_output_at = t_rng * 2  # first frame that computes 2D blobs

global olp_debug
olp_debug = bool(0)

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
videoo = video_to_tblobs(video)
end_time = time() - start_time
print(end_time)

# ************ PROGRAM BODY END ******************************************************************************************