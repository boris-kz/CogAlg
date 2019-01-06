import os
import cv2
import argparse
import numpy as np
from scipy import misc
from time import time
from collections import deque

''' Temporal blob composition over a sequence of frames in a video: 
    pixels are compared to rng adjacent pixels over lateral x, vertical y, temporal t coordinates,
    then resulting 3D tuples are combined into incremental-dimensionality patterns: 1D Ps ) 2D blobs ) 3D tblobs.     
    tblobs will then be evaluated for intra-tblob search.
'''

# ************ REUSABLE CLASSES *****************************************************************************************
class frame_of_patterns(object):
    def __init__(self, typ):
        self.core = typ[0]
        self.dimension = typ[1]
        self.level = typ[2:]
        self.xD = 0
        self.abs_xD = 0
        self.e_ = []
        if self.level == 'frame':
            self.Ly = 0
        else:
            self.Lt = 0
            self.yD = 0
            self.abs_yD = 0
        if self.core == 'm':
            self.L = 0
            self.I = 0
            self.Dx = 0; self.Dy = 0; self.Dt = 0
            self.Mx = 0; self.My = 0; self.Mt = 0

    def accum_params(self, params):
        " add lower-composition params to higher-composition params "
        self.L += params[0]
        self.I += params[1]
        self.Dx += params[2]
        self.Dy += params[3]
        self.Dt += params[4]
        self.Mx += params[5]
        self.My += params[6]
        self.Mt += params[7]


class pattern(object):
    def __init__(self, typ, x_coord=(9999999, -1), y_coord=(9999999, -1), t_coord=(9999999, -1), sign=-1):
        " initialize P, segment or blob "
        self.core = typ[0]
        self.dimension = typ[1]
        self.level = typ[2:]
        self.sign = sign
        self.L = 0  # length/area of a pattern
        self.I = 0  # summed input
        self.Dx = 0; self.Dy = 0; self.Dt = 0  # lateral, vertical, temporal D
        self.Mx = 0; self.My = 0; self.Mt = 0  # lateral, vertical, temporal M
        self.Alt0 = 0; self.Alt1 = 0; self.Alt2 = 0; self.Alt3 = 0; self.Alt4 = 0  # indicate value of overlapping alt-core blobs
        self.min_x, self.max_x = x_coord
        self.e_ = []
        self.terminated = False

        # Pattern level-specific init:
        if self.level != 'P':
            self.min_y, self.max_y = y_coord
            if not self.level in ['segment', 'blob']:
                self.min_t, self.max_t = t_coord

    def type(self):
        return self.core + self.dimension + self.level

    def params(self):
        " params = [L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4] "
        return [self.L, self.I, self.Dx, self.Dy, self.Dt, self.Mx, self.My, self.Mt,
                self.Alt0, self.Alt1, self.Alt2, self.Alt3, self.Alt4]

    def coords(self):
        " coords = [min_x, max_x, min_y, max_y] "
        if self.level == 'P':
            return [self.min_x, self.max_x]
        elif self.level in ['segment', 'blob']:
            return [self.min_x, self.max_x, self.min_y, self.max_y]
        else:
            return [self.min_x, self.max_x, self.min_y, self.max_y, self.min_t, self.max_t]

    def accum_params(self, params):
        " update internal params by summing with given params "
        self.L += params[0]
        self.I += params[1]
        self.Dx += params[2]
        self.Dy += params[3]
        self.Dt += params[4]
        self.Mx += params[5]
        self.My += params[6]
        self.Mt += params[7]
        self.Alt0 += params[8]
        self.Alt1 += params[9]
        self.Alt2 += params[10]
        self.Alt3 += params[11]
        self.Alt4 += params[12]

    def extend_coords(self, coords):
        " replace min/max coords with min/max that includes min/max of input coords "
        self.min_x = min(self.min_x, coords[0])
        self.max_x = max(self.max_x, coords[1])
        if len(coords) > 2:
            self.min_y = min(self.min_y, coords[2])
            self.max_y = max(self.max_y, coords[3])
            if len(coords) > 4:
                self.min_t = min(self.min_t, coords[4])
                self.max_t = max(self.max_t, coords[5])

    def rename(self, list):
        " rename multiple internal attributes at once "
        for old_name, new_name in list:
            self.__dict__[new_name] = self.__dict__.pop(old_name)

    def set(self, dict):
        " create new or set values of multiple attributes at once "
        for key, value in dict.iteritems():
            self.__dict__[key] = value


# ************ REUSABLE CLASSES ENDS ************************************************************************************

# ************ MISCELLANEOUS FUNCTIONS **********************************************************************************
# Includes:
# -rebuild_segment()
# -rebuild_blob()
# -rebuild_frame()
# -fetch_frame()
# ***********************************************************************************************************************
def rebuild_segment(dir, index, seg, blob_img, frame_img, print_separate_blobs=0, print_separate_segs=1):
    if print_separate_segs: seg_img = np.array([[[127] * 4] * X] * Y)
    y = seg.min_y  # min_y
    for P in seg.e_:
        x = P.min_x
        for i in range(P.L):
            frame_img[y, x, : 3] = [255, 255, 255] if P.sign else [0, 0, 0]
            if print_separate_blobs: blob_img[y, x, : 3] = [255, 255, 255] if P.sign else [0, 0, 0]
            if print_separate_segs: seg_img[y, x, : 3] = [255, 255, 255] if P.sign else [0, 0, 0]
            x += 1
        y += 1
    if print_separate_segs:
        min_x, max_x, min_y, max_y = seg.coords()
        cv2.rectangle(seg_img, (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), (0, 255, 255), 1)
        cv2.imwrite(dir + 'seg%d.jpg' % (index), seg_img)
    return blob_img
    # ---------- rebuild_segment() end ----------------------------------------------------------------------------------


def rebuild_blob(dir, index, blob, frame_img, print_separate_blobs=1, print_separate_segs=0):
    " Rebuilt data of a blob into an image "
    if print_separate_blobs:
        blob_img = np.array([[[127] * 4] * X] * Y)
    else:
        blob_img = None
    for idxs, idx in enumerate(blob.sorted_min_x_idx_):  # Iterate through segments' sorted id
        blob_img = rebuild_segment(dir + 'blob%d' % (index), idxs, blob.e_[idx], blob_img, frame_img, print_separate_blobs, print_separate_segs)
    if print_separate_blobs:
        min_x, max_x, min_y, max_y = blob.coords()
        cv2.rectangle(blob_img, (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), (0, 255, 255), 1)
        cv2.imwrite(dir + 'blob%d.jpg' % (index), blob_img)
    return frame_img
    # ---------- rebuild_blob() end -------------------------------------------------------------------------------------


def rebuild_frame(dir, frame, print_separate_blobs=0, print_separate_segs=0):
    " Rebuilt data of a frame into an image "
    frame_img = np.array([[[127] * 4] * X] * Y)
    if (print_separate_blobs or print_separate_segs) and not os.path.exists(dir):
        os.mkdir(dir)
    for iindex, index in enumerate(frame.sorted_min_x_idx_):  # Iterate through blobs' sorted indices
        frame_img = rebuild_blob(dir + '/', iindex, frame.e_[index], frame_img, print_separate_blobs, print_separate_segs)
    cv2.imwrite(dir + '.jpg', frame_img)
    # ---------- rebuild_frame() end ------------------------------------------------------------------------------------


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
# -term_segment_()
# -form_blob()
# -sort_coords()
# -scan_blob_()
# -scan_segment_()
# -find_overlaps()
# -find_olp_index()
# -form_tsegment()
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


def temporal_comp(ders2_, rng_ders3___, _xP_, _yP_, _tP_, frame, _frame):
    # ders2_: input line of complete 2D ders
    # rng_ders3___: prior frame of incomplete 3D tuple buffers, sliced into lines
    # comparison between t_rng temporally consecutive pixels, forming ders3: 3D tuple of derivatives per pixel

    # each of the following contains 2 types, per core variables m and d:
    xP = [pattern('mxP', (rng, -1)), pattern('dxP', (rng, -1))]  # initialize with min_x = rng, max_x = -1
    yP = [pattern('myP', (rng, -1)), pattern('dyP', (rng, -1))]
    tP = [pattern('mtP', (rng, -1)), pattern('dtP', (rng, -1))]
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
            fdt += dt  # running sum of differences between pixel _p and all previous and subsequent pixels within t_rng
            fmt += mt  # running sum of matches between pixel _p and all previous and subsequent pixels within t_rng
            back_dt += dt  # running sum of dt between pixel p and all previous pixels within t_rng
            back_mt += mt  # running sum of mt between pixel p and all previous pixels within t_rng

            if index < max_index:
                rng_ders3_[index] = (_p, _dx, _dy, fdt, _mx, _my, fmt)
            elif t > t_min_coord:
                ders = _p, _dx, _dy, fdt, _mx, _my, fmt
                xP, xP_, xbuff_, _xP_, frame, _frame = form_P(ders, x, X - rng - 1, xP, xP_, xbuff_, _xP_, frame, _frame, 0)  # mxP: typ = 0
                yP, yP_, ybuff_, _yP_, frame, _frame = form_P(ders, x, X - rng - 1, yP, yP_, ybuff_, _yP_, frame, _frame, 1)  # myP: typ = 1
                tP, tP_, tbuff_, _tP_, frame, _frame = form_P(ders, x, X - rng - 1, tP, tP_, tbuff_, _tP_, frame, _frame, 2)  # mtP: typ = 2
            index += 1

        rng_ders3_.appendleft((p, dx, dy, back_dt, mx, my, back_mt))  # new ders3 displaces completed one in temporal rng_ders3_ via maxlen
        new_rng_ders3__.append(rng_ders3_)  # rng_ders3__: line of incomplete ders3 buffers, to be added to next-frame rng_ders3___
        x += 1

    # terminate last higher line dP (typ = 3 -> 5) within neg mPs
    for typ in range(dim, dim * 2):
        if typ == 3: buff_ = xbuff_[1]; hP_ = _xP_[1]
        if typ == 4: buff_ = ybuff_[1]; hP_ = _yP_[1]
        if typ == 5: buff_ = tbuff_[1]; hP_ = _tP_[1]
        while buff_:
            hP = buff_.popleft()
            if hP.roots != 1:  # no roots
                frame, _frame = form_blob(hP, frame, _frame, typ)
        hP_, frame, _frame = term_segment_(hP_, frame, _frame, typ)

    rng_ders3___.append(new_rng_ders3__)  # rng_ders3___ for next frame

    return rng_ders3___, xP_, yP_, tP_, frame, _frame

    # ---------- temporal_comp() end ------------------------------------------------------------------------------------


def form_P(ders, x, term_x, P, P_, buff_, hP_, frame, _frame, typ, is_dP=0):
    # Initializes and accumulates 1D pattern
    # is_dP = bool(typ // dim), computed directly for speed and clarity:

    p, dx, dy, dt, mx, my, mt = ders  # 3D tuple of derivatives per pixel, "x" for lateral, "y" for vertical, "t" for temporal
    if   typ == 0: core = mx; alt0 = dx; alt1 = my; alt2 = mt; alt3 = dy; alt4 = dt
    elif typ == 1: core = my; alt0 = dy; alt1 = mx; alt2 = mt; alt3 = dx; alt4 = dt
    elif typ == 2: core = mt; alt0 = dt; alt1 = mx; alt2 = my; alt3 = dx; alt4 = dy
    elif typ == 3: core = dx; alt0 = mx; alt1 = dy; alt2 = dt; alt3 = my; alt4 = mt
    elif typ == 4: core = dy; alt0 = my; alt1 = dx; alt2 = dt; alt3 = mx; alt4 = mt
    else:          core = dt; alt0 = mt; alt1 = dx; alt2 = dy; alt3 = mx; alt4 = my

    s = 1 if core > 0 else 0
    if not (s == P[is_dP].sign or x == P[is_dP].min_x):  # P is terminated. P[0] is mP, P[1] is dP
        P, P_, buff_, hP_, frame, _frame = term_P(s, x, P, P_, buff_, hP_, frame, _frame, typ, is_dP)

    # Continued or initialized input and derivatives are accumulated:
    P[is_dP].accum_params([1, p, dx, dy, dt, mx, my, mt, abs(alt0), abs(alt1), abs(alt2), abs(alt3), abs(alt4)])
    # params = [L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4]
    P[is_dP].e_.append(ders)
    if P[is_dP].sign == -1: P[is_dP].sign = s

    if x == term_x:  # P is terminated
        P, P_, buff_, hP_, frame, _frame = term_P(s, x + 1, P, P_, buff_, hP_, frame, _frame, typ, is_dP)

    return P, P_, buff_, hP_, frame, _frame  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_P() end -------------------------------------------------------------------------------------------


def term_P(s, x, P, P_, buff_, hP_, frame, _frame, typ, is_dP):
    # Terminates 1D pattern if sign change or P_ end
    if not is_dP and P[is_dP].sign == 0:
        x0, L, ders_ = P[0].min_x, P[0].L, P[0].e_

        P[1] = pattern(typ_str[typ + dim] + 'P', (x0, -1))  # dPs (P[1]) formed inside of negative mP (P[0])

        for i in range(L):
            P, P_, buff_, _P_, frame, _frame = form_P(ders_[i], x0 + i, x - 1, P, P_, buff_, hP_, frame, _frame, typ + dim, True)  # is_dP = 1

    P[is_dP].max_x = x - 1
    P[is_dP].terminated = True
    if y == rng * 2:  # 1st line P_ is converted to init hP_;  scan_P_(), form_segment(), form_blob() use one type of Ps, hPs, buffs
        P_[is_dP].append((P[is_dP], []))  # P, _fork_, no root yet
    else:
        P_[is_dP], buff_[is_dP], hP_[is_dP], frame, _frame \
            = scan_P_(x - 1, P[is_dP], P_[is_dP], buff_[is_dP], hP_[is_dP], frame, _frame, typ)  # P scans hP_
    P[is_dP] = pattern(typ_str[typ] + 'P', (x, -1), sign=s)  # new P initialization at x0 = x

    return P, P_, buff_, hP_, frame, _frame
    # ---------- term_P() end -------------------------------------------------------------------------------------------


def scan_P_(x, P, P_, _buff_, hP_, frame, _frame, typ):
    # P scans shared-x-coordinate hPs in higher P_, combines overlapping Ps into blobs

    buff_ = deque()  # new buffer for displaced hPs (higher-line P tuples), for scan_P_(next P)
    fork_ = []  # refs to hPs connected to input P
    _x0 = 0  # to start while loop
    x0 = P.min_x

    while _x0 <= x:  # while x values overlap between P and _P
        if _buff_:
            hP = _buff_.popleft()  # hP was extended to segment and buffered in prior scan_P_
        elif hP_:
            hP, frame, _frame = form_segment(hP_.popleft(), frame, _frame, typ)
        else:
            break  # higher line ends, all hPs are converted to segments
        roots = hP.roots
        _x0 = hP.e_[-1].min_x  # hP.e_[-1] is _P
        _x = hP.e_[-1].max_x

        if P.sign == hP.sign and not _x < x0 and not x < _x0:  # P comb -> blob if s == _s, _last_x >= first_x and last_x >= _first_x
            roots += 1
            hP.roots = roots
            fork_.append(hP)  # P-connected hPs will be converted to segments at each _fork

        if _x > x:  # x overlap between hP and next P: hP is buffered for next scan_P_, else hP included in a blob segment
            buff_.append(hP)
        elif roots != 1:
            frame, _frame = form_blob(hP, frame, _frame, typ)  # segment is terminated and packed into its blob
        _x0 = _x + 1  # = first x of next _P

    buff_ += _buff_  # _buff_ is likely empty
    P_.append([P, fork_])  # P with no overlap to next _P is extended to hP and buffered for next-line scan_P_

    return P_, buff_, hP_, frame, _frame  # hP_ and buff_ contain only remaining _Ps, with _x => next x
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_segment(hP, frame, _frame, typ):
    # Add hP to higher-line segment or convert it into new segment; merge blobs

    _P, fork_ = hP
    ave_x = (_P.L - 1) // 2  # extra-x L = L-1 (1x in L)

    if len(fork_) == 1 and fork_[0].roots == 1:  # hP has one fork: hP.fork_[0], and that fork has one root: hP
        fork = fork_[0]
        # hP is merged into higher-line blob segment (Pars, roots, ave_x, xD, Py_, coords) at hP.fork_[0]:
        fork.accum_params(_P.params())  # params = [L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4]
        fork.roots = 0  # roots
        xd = ave_x - fork.ave_x
        fork.ave_x = ave_x  # ave_x
        fork.xD += xd  # xD for seg normalization and orientation, or += |dx| for curved yL?
        fork.abs_xD += abs(xd)
        fork.xd_.append(xd)
        fork.e_.append(_P)  # Py_: vertical buffer of Ps merged into seg
        fork.extend_coords(_P.coords())  # min_x, max_x
        hP = fork  # replace segment with including fork's segment
    else:  # new segment is initialized:
        hP = pattern(typ_str[typ] + 'segment', (_P.min_x, _P.max_x), (y - rng - 1, -1), sign=_P.sign)  # new instance of pattern class
        hP.accum_params(_P.params())  # initialize params with _P's params, etc.
        hP.roots = 0  # init roots
        hP.fork_ = fork_  # init fork_
        hP.ave_x = ave_x  # ave_x
        hP.xD = 0  # xD
        hP.abs_xD = 0
        hP.xd_ = [0]  # xd_ of corresponding Py_
        hP.e_.append(_P)  # Py_

        if not fork_:  # if no fork_: initialize blob
            blob = pattern(typ_str[typ] + 'blob', (_P.min_x, _P.max_x), (y - rng - 1, -1), sign=hP.sign)
            blob.xD = 0
            blob.abs_xD = 0
            blob.Ly = 0
            blob.remaining_roots = 1
        else:  # else merge into fork's blob
            blob = fork_[0].blob
        hP.blob = blob  # merge hP into blob
        blob.e_.append(hP)  # segment is buffered into blob's root_

        if len(fork_) > 1:  # merge blobs of all forks
            if fork_[0].roots == 1:  # if roots == 1
                frame, _frame = form_blob(fork_[0], frame, _frame, typ, 1)  # terminate seg of 1st fork

            for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                if fork.roots == 1:
                    frame, _frame = form_blob(fork, frame, _frame, typ, 1)

                if not fork.blob is blob:  # if not already merged/same
                    blobs = fork.blob
                    blob.accum_params(blobs.params())  # params = [L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4]
                    blob.extend_coords(blobs.coords())  # coord = [min_x, max_x, min_y, max_y]
                    blob.xD += blobs.xD
                    blob.abs_xD += blobs.abs_xD
                    blob.Ly += blobs.Ly
                    blob.remaining_roots += blobs.remaining_roots

                    for seg in blobs.e_:
                        if not seg is fork:
                            seg.blob = blob  # blobs in other forks are references to blob in the first fork
                            blob.e_.append(seg)  # buffer of merged root segments
                    fork.blob = blob
                    blob.e_.append(fork)
                blob.remaining_roots -= 1

    return hP, frame, _frame
    # ---------- form_segment() end -----------------------------------------------------------------------------------------


def term_segment_(hP_, frame, _frame, typ):
    # merge segments of last line into their blobs
    while hP_:
        hP, frame, _frame = form_segment(hP_.popleft(), frame, _frame, typ)
        frame, _frame = form_blob(hP, frame, _frame, typ)
    return hP_, frame, _frame
    # ---------- term_segment_() end ----------------------------------------------------------------------------------------


def form_blob(term_seg, frame, _frame, typ, y_carry=0):
    # Terminated segment is merged into continued or initialized blob (all connected segments)
    blob = term_seg.blob
    term_seg.max_y = y - rng - 1 - y_carry  # set max_y <- current y; y_carry: min elevation of term_seg over current hP
    blob.accum_params(term_seg.params())  # params = [L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4]
    blob.extend_coords(term_seg.coords())  # coords = [min_x, max_x, min_y, max_y]
    blob.xD += term_seg.xD  # ave_x angle, to evaluate blob for re-orientation
    blob.abs_xD += term_seg.abs_xD
    blob.Ly += len(term_seg.e_)  # Ly = number of slices in segment
    blob.remaining_roots += term_seg.roots - 1  # reference to term_seg is already in blob[9]
    term_seg.terminated = True

    if blob.remaining_roots == 0:  # if remaining_roots == 0: blob is terminated and packed in frame
        # sort indices of blob' segments by their min and max coordinates
        blob.sorted_min_x_idx_, blob.sorted_max_x_idx_, blob.sorted_min_y_idx_, blob.sorted_max_y_idx_, \
        blob.sorted_min_x_, blob.sorted_max_x_, blob.sorted_min_y_, blob.sorted_max_y_ = sort_segments(blob.e_)
        # terminated blob is packed into frame
        if term_seg.core == 'm' and term_seg.sign == 0:  # is negative mblob
            frame[typ].accum_params(term_seg.params())

        frame[typ].xD += blob.xD  # ave_x angle, to evaluate frame for re-orientation
        frame[typ].abs_xD += blob.abs_xD
        frame[typ].Ly += blob.Ly  # +Ly
        delattr(blob, 'remaining_roots')
        blob.terminated = True
        frame[typ].e_.append(blob)
        # initialize tsegment with terminated blob:
        blob.fork_ = []
        if t > t_rng * 2:
            frame[typ], _frame[typ] = scan_blob_(blob, frame[typ], _frame[typ])

    return frame, _frame
    # ---------- form_blob() end ----------------------------------------------------------------------------------------

def sort_segments(e_):
    " sort indices by min|max coords of segments"
    sorted_idx_min_x_ = sorted(range(len(e_)), key=lambda i: e_[i].min_x)  # segment indices sorted by min_x
    sorted_idx_max_x_ = sorted(range(len(e_)), key=lambda i: e_[i].max_x)  # segment indices sorted by max_x
    sorted_idx_min_y_ = sorted(range(len(e_)), key=lambda i: e_[i].min_y)  # segment indices sorted by min_y
    sorted_idx_max_y_ = sorted(range(len(e_)), key=lambda i: e_[i].max_y)  # segment indices sorted by max_y
    # the following lists are for zoning olp segs
    return sorted_idx_min_x_, sorted_idx_max_x_, sorted_idx_min_y_, sorted_idx_max_y_, \
           [e_[sorted_idx_min_x_[i]].min_x for i in range(len(e_))], \
           [e_[sorted_idx_max_x_[i]].max_x for i in range(len(e_))], \
           [e_[sorted_idx_min_y_[i]].min_y for i in range(len(e_))], \
           [e_[sorted_idx_max_y_[i]].max_y for i in range(len(e_))]
    # ---------- sort_coords() end --------------------------------------------------------------------------------------


def scan_blob_(blob, frame, _frame):
    # blob scans pri_blobs in higher frame, combines overlapping blobs into tblobs
    # Select only overlapping pri_blobs in _frame for speed?
    debug_idx_ = []
    olp_idx_ = find_overlaps(_frame, blob.coords())
    if len(olp_idx_) != 0:
        pri_tseg_ = _frame.e_  # list of same type pri_tsegs
        for olp_idx in olp_idx_:
            pri_tseg = pri_tseg_[olp_idx]
            pri_blob = pri_tseg.e_[-1]
            if pri_blob.sign == blob.sign:  # Check sign
                olp_min_x = max(pri_blob.min_x, blob.min_x)
                olp_max_x = min(pri_blob.max_x, blob.max_x)
                olp_min_y = max(pri_blob.min_y, blob.min_y)
                olp_max_y = min(pri_blob.max_y, blob.max_y)
                if scan_segment_(blob, pri_blob, [olp_min_x, olp_max_x, olp_min_y, olp_max_y]):
                    pri_tseg.roots += 1
                    blob.fork_.append(pri_tseg)
                    debug_idx_.append(olp_idx)
    # For Debugging --------------------------------------------------------------
    # Print selected blob formed in frame at t > t_rng * 2 and all it's overlapping blobs in previous frame
    global olp_debug, debug_case
    if olp_debug and t > t_rng * 2 and len(debug_idx_) != 0:
        if debug_case == output_at_case:
            filtered_pri_frame = np.array([[[127] * 4] * X] * Y)
            rebuild_blob('./images/', 0, blob, filtered_pri_frame, 1)
            for i in debug_idx_:
                rebuild_blob('./images/olp_', i, _frame.e_[i].e_[-1], filtered_pri_frame, 1)
            olp_debug = False
        else:
            debug_case += 1
    # ----------------------------------------------------------------------------
    return frame, _frame
    # ---------- scan_blob_() end ---------------------------------------------------------------------------------------


def scan_segment_(blob, pri_blob, bounding_box):
    # scans segments for overlap
    # choose only segments inside olp to check:
    idx = find_overlaps(blob, bounding_box)
    pri_idx = find_overlaps(pri_blob, bounding_box)

    for i in idx:
        seg = blob.e_[i]
        olp_idx_ = np.intersect1d(find_overlaps(pri_blob, seg.coords()), pri_idx)
        if len(olp_idx_) != 0:
            pri_seg_ = pri_blob.e_
            for olp_idx in olp_idx_:
                pri_seg = pri_seg_[olp_idx]
                olp_min_y = max(pri_seg.min_y, seg.min_y)  # olp_min/max_y indicates
                olp_max_y = min(pri_seg.max_y, seg.max_y)  # potentially overlapping Ps
                olp_P_idx_stop = olp_max_y - seg.min_y + 1
                olp_P_idx = olp_min_y - seg.min_y
                olp_pri_P_idx = olp_min_y - pri_seg.min_y
                while olp_P_idx < olp_P_idx_stop:
                    P = seg.e_[olp_P_idx]
                    pri_P = pri_seg.e_[olp_pri_P_idx]
                    if P.min_x <= pri_P.max_x and P.max_x >= pri_P.min_x:
                        return True
                    olp_P_idx += 1
                    olp_pri_P_idx += 1
    return False
    # ---------- scan_segment_() end ------------------------------------------------------------------------------------


def find_overlaps(obj, bounding_box):
    # Search for boundaries of sorted pri_blobs that overlap boundaries of input blob
    N = len(obj.e_)
    min_x, max_x, min_y, max_y = bounding_box
    # find_olp_index(a_, first_index, last_index, target, right_olp):
    _min_x_idx = find_olp_index(obj.sorted_min_x_, 0, N, max_x, 1)
    _max_x_idx = find_olp_index(obj.sorted_max_x_, 0, N, min_x, 0)
    _min_y_idx = find_olp_index(obj.sorted_min_y_, 0, N, max_y, 1)
    _max_y_idx = find_olp_index(obj.sorted_max_y_, 0, N, min_y, 0)

    _min_x_less_or_equal_max_x_indices = obj.sorted_min_x_idx_[:_min_x_idx]  # overlap prerequisite: _min_x <= max_x
    _min_y_less_or_equal_max_y_indices = obj.sorted_min_y_idx_[:_min_y_idx]  # overlap prerequisite: _min_y <= max_y
    _max_x_greater_or_equal_min_x_indices = obj.sorted_max_x_idx_[_max_x_idx:]  # overlap prerequisite: _max_x <= min_x
    _max_y_greater_or_equal_min_y_indices = obj.sorted_max_y_idx_[_max_y_idx:]  # overlap prerequisite: _max_y <= min_y
    # e_ overlap is a common subset of the above 4 sets
    return np.intersect1d(np.intersect1d(_min_x_less_or_equal_max_x_indices, _max_x_greater_or_equal_min_x_indices),
                          np.intersect1d(_min_y_less_or_equal_max_y_indices, _max_y_greater_or_equal_min_y_indices))
    # ---------- find_overlaps() end ------------------------------------------------------------------------------------


def find_olp_index(a_, i0, i, target, right_olp=0):
    # a binary search module
    if target + right_olp <= a_[i0]:
        return i0
    elif a_[i - 1] < target + right_olp:
        return i
    else:
        im = (i0 + i) // 2
        if a_[im] < target + right_olp:
            return find_olp_index(a_, im, i, target, right_olp)
        else:
            return find_olp_index(a_, i0, im, target, right_olp)
    # ---------- find_olp_index() end -----------------------------------------------------------------------------------


def form_tsegment(blob, videoo, typ):
    # Add blob to previous-frame tsegment or convert it into new tsegment; merge tblobs
    ave_x = (blob.max_x - blob.min_x) // 2
    ave_y = (blob.max_y - blob.min_y) // 2
    fork_ = blob.fork_
    delattr(blob, 'fork_')
    if len(fork_) == 1 and fork_[0].roots == 1:  # hP has one fork: hP.fork_[0], and that fork has one root: hP
        fork = fork_[0]
        # hP is merged into higher-line blob segment (Pars, roots, ave_x, xD, Py_, coords) at hP.fork_[0]:
        fork.accum_params(blob.params())  # params = [L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4]
        fork.roots = 0  # roots
        xd = ave_x - fork.ave_x
        yd = ave_y - fork.ave_y
        fork.ave_x = ave_x  # ave_x
        fork.ave_y = ave_y  # ave_y
        fork.xD += xd  # xD for seg normalization and orientation
        fork.yD += yd
        fork.abs_xD += abs(xd)
        fork.abs_yD += abs(yd)
        fork.xyd_.append((xd, yd))
        fork.e_.append(blob)  # blob_
        fork.extend_coords(blob.coords())  # min_x, max_x, min_y, max_y
        return fork, videoo  # replace blob with including fork's tsegment
    else:  # new segment is initialized:
        tsegment = pattern(typ_str[typ] + 'tsegment', (blob.min_x, blob.max_x), (blob.min_y, blob.max_y), (t - t_rng, -1), sign=blob.sign)
        # new instance of pattern class
        tsegment.accum_params(blob.params())  # init params with blob's params
        tsegment.roots = 0  # init roots
        tsegment.fork_ = fork_  # init fork_
        tsegment.ave_x = ave_x  # ave_x
        tsegment.ave_y = ave_y
        tsegment.xD = 0  # xD
        tsegment.yD = 0  # yD
        tsegment.abs_xD = 0
        tsegment.abs_yD = 0
        tsegment.xyd_ = [(0, 0)]  # xyd_ of blob_
        tsegment.e_.append(blob)  # blob_

        if not fork_:  # if no forks: initialize tblob
            tblob = pattern(typ_str[typ] + 'tblob', (blob.min_x, blob.max_x), (blob.min_y, blob.max_y), (t - t_rng, -1), sign=tsegment.sign)
            tblob.xD = 0
            tblob.yD = 0
            tblob.abs_xD = 0
            tblob.abs_yD = 0
            tblob.Lt = 0
            tblob.remaining_roots = 1
        else:  # else merge into fork's tblob
            tblob = fork_[0].tblob
        tsegment.tblob = tblob  # merge tsegment into tblob
        tblob.e_.append(tsegment)  # tsegment is buffered into tblob's root_

        if len(fork_) > 1:  # merge tblobs of all forks
            if fork_[0].roots == 1:  # if roots == 1
                videoo = form_tblob(fork_[0], videoo, typ, 1)  # terminate tsegment of 1st fork

            for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                if fork.roots == 1:
                    videoo = form_tblob(fork, videoo, typ, 1)

                if not fork.tblob is tblob:  # if not already merged/same
                    tblobs = fork.tblob
                    tblob.accum_params(tblobs.params())  # params = [L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4]
                    tblob.extend_coords(tblobs.coords())  # coord = [min_x, max_x, min_y, max_y, min_t, max_t]
                    tblob.xD += tblobs.xD
                    tblob.yD += tblobs.yD
                    tblob.abs_xD += tblobs.abs_xD
                    tblob.abs_yD += tblobs.abs_yD
                    tblob.Lt += tblobs.Lt
                    tblob.remaining_roots += tblobs.remaining_roots

                    for tseg in tblobs.e_:
                        if not tseg is fork:
                            tseg.tblob = tblob  # tblobs in other forks are references to tblob in the first fork
                            tblob.e_.append(tseg)  # buffer of merged root tsegments
                    fork.tblob = tblob
                    tblob.e_.append(fork)
                tblob.remaining_roots -= 1
    return tsegment, videoo
    # ---------- form_tsegment() end ------------------------------------------------------------------------------------

def form_tblob(term_tseg, videoo, typ, t_carry = 0):
    # Terminated tsegment is merged into continued or initialized tblob (all connected tsegments)
    tblob = term_tseg.tblob
    term_tseg.max_t = t - t_rng - 1 - t_carry  # set max_t <- current t; t_carry: min elevation of term_tseg over current pri_blob
    tblob.accum_params(term_tseg.params())  # params = [L, I, Dx, Dy, Dt, Mx, My, Mt, Alt0, Alt1, Alt2, Alt3, Alt4]
    tblob.extend_coords(term_tseg.coords())  # coords = [min_x, max_x, min_y, max_y, min_t, max_t]
    tblob.xD += term_tseg.xD  # ave_x angle, to evaluate tblob for re-orientation
    tblob.yD += term_tseg.yD
    tblob.abs_xD += term_tseg.abs_xD
    tblob.abs_yD += term_tseg.abs_yD
    tblob.Lt += len(term_tseg.e_)  # Lt = number of slices in tsegment
    tblob.remaining_roots += term_tseg.roots - 1  # reference to term_tseg is already in tblob
    term_tseg.terminated = True

    if tblob.remaining_roots == 0:  # if remaining_roots == 0: tblob is terminated and packed in videoo
        if term_tseg.core == 'm' and term_tseg.sign == 0:  # is negative mblob
            videoo[typ].accum_params(term_tseg.params())

        videoo[typ].xD += tblob.xD  # ave_x angle, to evaluate frame for re-orientation
        videoo[typ].yD += tblob.yD
        videoo[typ].abs_xD += tblob.abs_xD
        videoo[typ].abs_yD += tblob.abs_yD
        videoo[typ].Lt += tblob.Lt
        delattr(tblob, 'remaining_roots')
        tblob.terminated = True
        videoo[typ].e_.append(tblob)

    return videoo
    # ---------- form_tblob() end ---------------------------------------------------------------------------------------


def video_to_tblobs(video):
    # Main body of the operation,
    # postfix '_' denotes array vs. element,
    # prefix '_' denotes prior- pixel, line, or frame variable

    # higher-line same- d-, m-, dy-, my- sign 1D patterns
    _xP_ = [deque(), deque()]
    _yP_ = [deque(), deque()]
    _tP_ = [deque(), deque()]
    _frame = [frame_of_patterns(typ_str[i] + 'frame') for i in range(2 * dim)]

    # Main output: [[Dxf, Lf, If, Dxf, Dyf, Dtf, Mxf, Myf, Mtf], tblob_]
    videoo = [frame_of_patterns(typ_str[i] + 'videoo') for i in range(2 * dim)]
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
        # incomplete ders2_ s are discarded
        if y > min_coord:
            # Transfer complete list of tuples of ders2 into line y of ders3___
            rng_ders3__ = []
            for (p, dx, dy, mx, my) in ders2_:
                ders3 = p, dx, dy, 0, mx, my, 0  # dt, mt initialized at 0
                rng_ders3_ = deque(maxlen=t_rng)  # temporal buffer of incomplete derivatives tuples, for fuzzy ycomp
                rng_ders3_.append(ders3)  # only one tuple in first-frame rng_ders3_
                rng_ders3__.append(rng_ders3_)

            rng_ders3___.append(rng_ders3__)  # 1st frame, last vertical rng of incomplete ders2__ is discarded

    # Main operations
    for t in range(1, T):
        if not video.isOpened():  # Terminate at the end of video
            break
        frame = [frame_of_patterns(typ_str[i] + 'frame') for i in range(2 * dim)]
        line_ = fetch_frame(video)
        for y in range(0, Y):
            pixel_ = line_[y, :]
            ders1_ = lateral_comp(pixel_)  # lateral pixel comparison
            ders2_, rng_ders2__ = vertical_comp(ders1_, rng_ders2__)  # vertical pixel comparison
            if y > min_coord:
                # temporal pixel comparison:
                rng_ders3___, _xP_, _yP_, _tP_, frame, _frame = temporal_comp(ders2_, rng_ders3___, _xP_, _yP_, _tP_, frame, _frame)

        # merge segs of last line into their blobs:
        if t > t_min_coord:
            y = Y
            for typ in range(6):
                is_dP = typ // dim
                dimension = typ % dim
                if dimension == 0: hP_ = _xP_[is_dP]
                if dimension == 1: hP_ = _yP_[is_dP]
                if dimension == 2: hP_ = _tP_[is_dP]
                hP_, frame, _frame = term_segment_(hP_, frame, _frame, typ)
                # Sort blobs' indices based on min_x, max_x, min_y, max_y:
                frame[typ].sorted_min_x_idx_, frame[typ].sorted_max_x_idx_, frame[typ].sorted_min_y_idx_, frame[typ].sorted_max_y_idx_, \
                frame[typ].sorted_min_x_, frame[typ].sorted_max_x_, frame[typ].sorted_min_y_, frame[typ].sorted_max_y_ = sort_segments(frame[typ].e_)

                # tsegments, tblobs operations:
                for tsegment in _frame[typ].e_:
                    if tsegment.roots != 1:
                        videoo = form_tblob(tsegment, videoo, typ)
                for i in range(len(frame[typ].e_)):
                    frame[typ].e_[i], videoo = form_tsegment(frame[typ].e_[i], videoo, typ)

        if record and t == frame_output_at:  # change these in program body
            rebuild_frame('./images/mblobs_horizontal', frame[0], record_blobs, record_segs)
            rebuild_frame('./images/mblobs_vertical', frame[1], record_blobs, record_segs)
            rebuild_frame('./images/mblobs_temporal', frame[2], record_blobs, record_segs)
            rebuild_frame('./images/dblobs_horizontal', frame[3], record_blobs, record_segs)
            rebuild_frame('./images/dblobs_vertical', frame[4], record_blobs, record_segs)
            rebuild_frame('./images/dblobs_temporal', frame[5], record_blobs, record_segs)

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
dim = 3  # Number of dimensions: x, y and t
typ_str = ('mx', 'my', 'mt', 'dx', 'dy', 'dt')
der_str = ('m', 'd')  # derivatives string
dim_str = ('x', 'y', 't')  # dimensions string

# For outputs:
record = bool(0)  # Set to True yield file outputs
record_blobs = bool(0)
record_segs = bool(0)
frame_output_at = t_rng * 2  # first frame that computes 2D blobs

global olp_debug, debug_case
olp_debug = bool(0)
debug_case = 0
output_at_case = 0

# Load inputs --------------------------------------------------------------------
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-v', '--video', help='path to video file', default='./videos/Test01.avi')
arguments = vars(argument_parser.parse_args())
video = cv2.VideoCapture(arguments['video'], 0)

line_ = fetch_frame(video)
Y, X = line_.shape  # image height and width
T = 8  # number of frame read limit

# Main ---------------------------------------------------------------------------
start_time = time()
videoo = video_to_tblobs(video)
end_time = time() - start_time
print(end_time)

# ************ PROGRAM BODY END ******************************************************************************************