# import cv2
# import argparse
from scipy import misc
from time import time
from collections import deque

'''   
    frame() is my core algorithm of levels 1 + 2, modified for 2D: segmentation of image into blobs, then search within and between blobs.
    frame_blobs() is frame() limited to definition of initial blobs per each of 4 derivatives, vs. per 2 gradients in current frame().
    frame_dblobs() is updated version of frame_blobs with only one blob type: dblob, to ease debugging, currently in progress.
    
    Each performs several levels (Le) of encoding, incremental per scan line defined by vertical coordinate y, outlined below.
    value of y per Le line is shown relative to y of current input line, incremented by top-down scan of input image,

    1Le, line y:    x_comp(p_): lateral pixel comparison -> tuple of derivatives ders ) array ders_
    2Le, line y- 1: y_comp(ders_): vertical pixel comp -> 2D tuple ders2 ) array ders2_ 
    3Le, line y- 1+ rng*2: form_P(ders2) -> 1D pattern P) hP  
    4Le, line y- 2+ rng*2: scan_P_(P, hP) -> hP, roots: down-connections, fork_: up-connections between Ps 
    5Le, line y- 3+ rng*2: form_segment: merge vertically-connected _Ps in non-forking blob segments
    6Le, line y- 4+ rng*2+ segment depth: form_blob: merge connected segments into blobs
    
    These functions are tested through form_P, I am currently debugging scan_P_. 
    All 2D functions (y_comp, scan_P_, etc.) input two lines: higher and lower, convert elements of lower line 
    into elements of new higher line, and displace elements of old higher line into higher function.
    Higher-line elements include additional variables, derived while they were lower-line elements.
    
    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match.
    Each vertical and horizontal derivative forms separate blobs, suppressing overlapping orthogonal representations.
    They can also be summed to estimate diagonal or hypot derivatives, for blob orientation to maximize primary derivatives.
    Orientation increases primary dimension of blob to maximize match, and decreases secondary dimension to maximize difference.
    
    Subsequent union of lateral and vertical patterns is by strength only, orthogonal sign is not commeasurable?
    prefix '_' denotes higher-line variable or pattern, vs. same-type lower-line variable or pattern,
    postfix '_' denotes array name, vs. same-name elements of that array:
'''

def lateral_comp(pixel_):  # comparison over x coordinate: between min_rng of consecutive pixels within each line

    ders1_ = []  # tuples of complete 1D derivatives: summation range = rng
    rng_ders1_ = deque(maxlen=rng)  # array of ders1 within rng from input pixel: summation range < rng
    max_index = rng - 1  # max index of rng_ders1_
    pri_d, pri_m = 0, 0  # fuzzy derivatives in prior completed tuple

    for p in pixel_:  # pixel p is compared to rng of prior pixels within horizontal line, summing d and m per prior pixel
        for index, (pri_p, d, m) in enumerate(rng_ders1_):

            d += p - pri_p  # fuzzy d: running sum of differences between pixel and all subsequent pixels within rng
            m += min(p, pri_p)  # fuzzy m: running sum of matches between pixel and all subsequent pixels within rng

            if index < max_index:
                rng_ders1_[index] = (pri_p, d, m)
            else:
                ders1_.append((pri_p, d + pri_d, m + pri_m))  # completed bilateral tuple is transferred from rng_ders_ to ders_
                pri_d = d; pri_m = m  # to complement derivatives of next rng_t_: derived from next rng of pixels

        rng_ders1_.appendleft((p, 0, 0))  # new tuple with initialized d and m, maxlen displaces completed tuple from rng_t_

    ders1_ += reversed(rng_ders1_)  # or tuples of last rng (incomplete, in reverse order) are discarded?
    return ders1_


def vertical_comp(ders1_, ders2__, _dP_, dframe):
    # comparison between rng vertically consecutive pixels, forming ders2: tuple of 2D derivatives per pixel

    dP = 0, 0, 0, 0, 0, 0, 0, []  # lateral difference pattern = pri_s, L, I, D, Dy, V, Vy, ders2_
    dP_ = deque()  # line y - 1+ rng*2
    dbuff_ = deque()  # line y- 2+ rng*2: _Ps buffered by previous run of scan_P_
    new_ders2__ = deque()  # 2D: line of ders2_s buffered for next-line comp

    x = 0  # lateral coordinate of current pixel
    max_index = rng - 1  # max ders2_ index
    min_coord = rng * 2 - 1  # min x and y for form_P input: ders2 from comp over rng*2 (bidirectional: before and after pixel p)
    dy, my = 0, 0  # for initial rng of lines, to reload _dy, _vy = 0, 0 in higher tuple

    for (p, d, m), (ders2_, _dy, _my) in zip(ders1_, ders2__):  # pixel comp to rng _pixels in ders2_, summing dy and my per _pixel
        x += 1
        index = 0
        for (_p, _d, dy, _m, my) in ders2_:  # vertical derivatives are incomplete; prefix '_' denotes higher-line variable

            dy += p - _p  # fuzzy dy: running sum of differences between pixel and all lower pixels within rng
            my += min(p, _p)  # fuzzy my: running sum of matches between pixel and all lower pixels within rng

            if index < max_index:
                ders2_[index] = (_p, d, dy, m, my)

            elif x > min_coord and y > min_coord + ini_y:

                _v = _m - abs(d) - ave  # projected m is cancelled by negative d: d/2, + rdn value of overlapping dP: d/2?
                vy = my + _my - abs(dy) - ave
                ders2 = _p, _d, dy + _dy, _v, vy
                dP, dP_, dbuff_, _dP_, dframe = form_P(ders2, x, dP, dP_, dbuff_, _dP_, dframe)

            index += 1

        ders2_.appendleft((p, d, 0, m, 0))  # initial dy and my = 0, new ders2 replaces completed t2 in vertical ders2_ via maxlen
        new_ders2__.append((ders2_, dy, my))  # vertically-incomplete 2D array of tuples, converted to ders2__, for next-line ycomp

    if y > min_coord + ini_y:  # not-terminated P at the end of each line is buffered or scanned:

        if y == rng * 2 + ini_y:  # _P_ initialization by first line of Ps, empty until vertical_comp returns P_
            dP_.append([dP, 0, [], x])  # empty _fork_ in the first line of hPs, x-1: delayed P displacement
        else:
            dP_, dbuff_, _dP_, dframe = scan_P_(x, dP, dP_, dbuff_, _dP_, dframe)  # scans higher-line Ps for contiguity

    return new_ders2__, dP_, dframe  # extended in scan_P_; net_s are packed into frames


def form_P(ders, x, P, P_, buff_, hP_, frame):  # initializes, accumulates, and terminates 1D pattern: dP | vP | dyP | vyP

    p, d, dy, v, vy = ders  # 2D tuple of derivatives per pixel, "y" denotes vertical vs. lateral derivatives
    s = 1 if d > 0 else 0  # core = 0 is negative: no selection?

    if s == P[0] or x == rng * 2:  # s == pri_s or initialized: P is continued, else terminated:
        pri_s, L, I, D, Dy, V, Vy, ders_ = P
    else:
        if y == rng * 2 + ini_y:  #  1st line: form_P converts P to initialized hP, forming initial P_ -> hP_
            P_.append([P, 0, [], x-1])
        else:
            P_, buff_, hP_, frame = scan_P_(x-1, P, P_, buff_, hP_, frame)  # scans higher-line Ps for contiguity
            # x-1: ends with prior p
        L, I, D, Dy, V, Vy, ders_ = 0, 0, 0, 0, 0, 0, []  # new P initialization

    L += 1  # length of a pattern, continued or initialized input and derivatives are accumulated:
    I += p  # summed input
    D += d  # lateral D
    Dy += dy  # vertical D
    V += v  # lateral V
    Vy += vy  # vertical V
    ders_.append(ders)  # ders2s are buffered for oriented rescan and incremental range | derivation comp

    P = s, L, I, D, Dy, V, Vy, ders_
    return P, P_, buff_, hP_, frame  # accumulated within line, P_ is a buffer for conversion to _P_


def scan_P_(x, P, P_, _buff_, hP_, frame):  # P scans shared-x-coordinate hPs in hP_, forms overlaps

    buff_ = deque()  # new buffer for displaced hPs, for scan_P_(next P)
    fork_ = []  # hPs connected to input P
    ini_x = 0  # always starts while, next ini_x = _x + 1

    while ini_x <= x:  # while x values overlap between P and hP
        if _buff_:
            hP = _buff_.popleft()  # hP buffered in prior scan_P_, seg id == _fork_ id for ref by root Ps
            _P, roots, _fork_, _x = hP
        elif hP_:
            hP = hP_.popleft()  # roots = 0: number of Ps connected to _P: pri_s, L, I, D, Dy, V, Vy, ders_
            _P, roots, _fork_, _x = hP
        else:
            break  # higher line ends, all hPs are converted to seg

        if P[0] == _P[0]:  # if s == _s: core sign match, + selective inclusion if contiguity eval?
            roots += 1; hP[1] = roots
            fork_.append(hP)  # P-connected hPs will be converted to segments at each _fork

        if _x > x:  # x overlap between hP and next P: hP is buffered for next scan_P_, else hP included in a blob segment
            buff_.append(hP)
        else:
            ave_x = _x - (_P[1]-1) // 2  # average x of _P; _P[1]-1: extra-x L = L-1 (1x in L)

            if y == rng * 2 + 1 + ini_y:  # 1st line: scan_P_ converts hPs to initialized blob segments as hhP:
                hP[0] = list(_P[1:7]); hP += 0, [(_P, 0)], [_P[0],0,0,0,0,0,0,0,y,[]]  # seg: Vars, roots, _fork_, ave_x, Dx, Py_, blob
            else:
                if len(hP[2]) == 1 and hP[2][0][1] == 1:  # hP has one fork: hP[2][0], and that fork has one root
                    # blob segment at hP[2][0] is incremented with hP, then moved into hP:
                    s, L, I, D, Dy, V, Vy, ders_ = _P
                    Ls, Is, Ds, Dys, Vs, Vys = hP[2][0][0]
                    Ls += L; Is += I; Ds += D; Dys += Dy; Vs += V; Vys += Vy
                    hP[0] = [Ls, Is, Ds, Dys, Vs, Vys]
                    # hP[1] = roots, not modified
                    hP[3] = ave_x
                    if y == rng * 2 + 2 + ini_y:  # seg forks are still hPs
                        dx = 0  # Dx for seg norm and orient eval, | += |xd| for curved yL?
                    else: dx = ave_x - hP[2][0][3]
                    hP[4] = hP[2][0][4] + dx  # Dx for seg norm and orient eval, | += |xd| for curved yL?  if y == rng * 2 + 2 + ini_y:?
                    hP[5] = hP[2][0][5].append((_P, dx))  # buffer Py_
                    hP[6] = hP[2][0][6]  # blob
                    hP[2] = hP[2][0][2]  # seg fork_; last step?
                else:
                    hP[0] = list(_P[1:7]); hP += 0, [(_P, 0)], [_P[0],0,0,0,0,0,0,0,y,[]]
                    # hP is converted into new hhP (segment): Vars, roots, _fork_, ave_x, Dx, Py_, blob

                if roots == 0:  # immediate blob, no y > rng * 2 + 2 + ini_y: y P ) y-1 hP ) y-2 seg ) y-4 blob ) y-5 frame?
                    frame = form_blob(hP, frame)  # bottom segment is terminated and added to internal blob

        ini_x = _x + 1  # first x of next hP

    buff_ += _buff_  # _buff_ is likely empty
    P_.append([P, 0, fork_, x])  # P with no overlap to next _P is buffered for next-line scan_P_, converted to hP

    return P_, buff_, hP_, frame  # hP_ and buff_ contain only remaining _Ps, with _x => next x


def form_blob(term_seg, frame):  # continued or initialized blob (connected segments) is incremented by terminated segment
    [L, I, D, Dy, V, Vy], roots, fork_, x, xD, Py_, blob = term_seg
    if fork_:  # seg forks are segs

        fork_[0][6][1] += L  # unique blob -> fork_[0][4], ref by other forks, no return by index, seg in enumerate(fork_):
        fork_[0][6][2] += I
        fork_[0][6][3] += D
        fork_[0][6][4] += Dy
        fork_[0][6][5] += V
        fork_[0][6][6] += Vy
        fork_[0][6][7] += xD
        fork_[0][6][8] = max(len(Py_), fork_[0][6][8])  # blob yD += max root seg Py_:  if y - len(Py_) +1 < min_y?
        fork_[0][6][9].append(([[L, I, D, Dy, V, Vy], x, xD, Py_], blob))  # term_seg is appended to fork[0] _root_

        fork_[0][1] -= 1  # roots -= 1, because root segment was terminated
        if fork_[0][1] == 0:  # recursive higher-level segment-> blob inclusion and termination test
            frame = form_blob(fork_[0], frame)  # no return: del (seg[:]); seg += iseg; fork_[index] = seg: ref from blob only?

        for index, fork in enumerate(fork_[1:len(fork_)]):
            fork[6] = fork_[0][6]  # ref to unique blob, for each root? fork_-> root_ mapping vs. separate seg[6][9].append?
            fork[1] -= 1
            if fork[1] == 0:  # seg roots; recursive higher-level segment -> blob inclusion and termination test
               frame = form_blob(fork, frame)  # return for ref from lateral forks?
            fork_[index] = fork

    # co_roots += co_fork, term subb (sub_blob ! blob) while binary co_root | co_fork count?
    # right_count and left_1st (for transfer at roots+forks term)
    # right: cont roots and len(_fork_), each summed at current subb term, for blob term eval?

    else:  # fork_ == 0: blob is terminated and added to frame:
        s, L, I, D, Dy, V, Vy, xD, yD, root_ = blob
        frame[0] += L  # frame Vars to compute averages, redundant for same-scope alt_frames
        frame[1] += I
        frame[2] += D
        frame[3] += Dy
        frame[4] += V
        frame[5] += Vy
        frame[6] += xD  # for frame orient eval, += |xd| for curved max_L?
        frame[7] += yD
        frame[8].append(((s, L, I, D, Dy, V, Vy, x - xD//2, xD, y, yD), root_))  # blob_; xD for blob orient eval before comp_P

    return frame  # no term_seg return needed?  no return term_seg[5] = fork_: no roots to ref


def image_to_blobs(image):  # postfix '_' denotes array vs. element, prefix '_' denotes higher-line vs. lower-line variable

    _P_ = deque()  # higher-line same- d-, v-, dy-, vy- sign 1D patterns
    frame = [0, 0, 0, 0, 0, 0, 0, 0, []]  # L, I, D, Dy, V, Vy, xD, yD, blob_
    global y
    y = ini_y  # initial input line, set at 400 as that area in test image seems to be the most diverse

    ders2_ = deque(maxlen=rng)  # vertical buffer of incomplete derivatives tuples, for fuzzy ycomp
    ders2__ = []  # horizontal line of vertical buffers: 2D array of 2D tuples, deque for speed?
    pixel_ = image[ini_y, :]  # first line of pixels at y == 0
    ders1_ = lateral_comp(pixel_)  # after part_comp (pop, no t_.append) while x < rng?

    for (p, d, m) in ders1_:
        ders2 = p, d, 0, m, 0  # dy, my initialized at 0
        ders2_.append(ders2)  # only one tuple per first-line ders2_
        ders2__.append((ders2_, 0, 0))  # _dy, _my initialized at 0

    for y in range(ini_y + 1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?

        pixel_ = image[y, :]  # vertical coordinate y is index of new line p_
        ders1_ = lateral_comp(pixel_)  # lateral pixel comparison
        ders2__, _P_, frame = vertical_comp(ders1_, ders2__, _P_, frame)  # vertical pixel comparison

    # frame ends, last vertical rng of incomplete ders2__ is discarded,
    # vertically incomplete P_ patterns are still inputted in scan_P_?
    return frame  # frame of 2D patterns to be outputted to level 2


# pattern filters: eventually updated by higher-level feedback, initialized here as constants:

rng = 2  # number of leftward or upward pixels compared to each input pixel
ave = 63 * rng * 2  # average match: value pattern filter
ave_rate = 0.25  # average match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)
ini_y = 400

image = misc.face(gray=True)  # read image as 2d-array of pixels (gray scale):
image = image.astype(int)
Y, X = image.shape  # image height and width

# or:
# argument_parser = argparse.ArgumentParser()
# argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/raccoon.jpg')
# arguments = vars(argument_parser.parse_args())
# image = cv2.imread(arguments['image'], 0).astype(int)

start_time = time()
frame_of_blobs = image_to_blobs(image)
end_time = time() - start_time
print(end_time)

