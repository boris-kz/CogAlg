import cv2
import argparse
from time import time
from collections import deque

'''   
    frame() is my core algorithm of levels 1 + 2, modified for 2D: segmentation of image into blobs, then search within and between blobs.
    frame_blobs() is frame() limited to definition of initial blobs per each of 4 derivatives, vs. per 2 gradients in current frame().
    frame_dblobs() is updated version of frame_blobs with only one blob type: dblob, to ease debugging, currently in progress.
    
    Each performs several levels (Le) of encoding, incremental per scan line defined by vertical coordinate y, outlined below.
    value of y per Le line is shown relative to y of current input line, incremented by top-down scan of input image,
    prefix '_' denotes higher-line variable or pattern, vs. same-type lower-line variable or pattern,
    postfix '_' denotes array name, vs. same-name elements of that array:

    1Le, line y:    x_comp(p_): lateral pixel comparison -> tuple of derivatives ders ) array ders_
    2Le, line y- 1: y_comp(ders_): vertical pixel comp -> 2D tuple ders2 ) array ders2_ 
    3Le, line y- 1+ rng*2: form_P(ders2) -> 1D pattern P ) P_  
    4Le, line y- 2+ rng*2: scan_P_(P, _P) -> _P, fork_, root_: downward and upward connections between Ps of adjacent lines 
    5Le, line y- 3+ rng*2: form_blob(_P, blob) -> blob: merge vertically-connected Ps into non-forking blob segments
    6Le, line y- 4+ rng*2+ blob depth: term_blob, form_net -> net: merge connected segments into network of terminated forks
    
    These functions are tested through form_P, I am currently debugging scan_P_. 
    All 2D functions (y_comp, scan_P_, etc.) input two lines: higher and lower, convert elements of lower line 
    into elements of new higher line, and displace elements of old higher line into some higher function.
    Higher-line elements include additional variables, derived while they were lower-line elements.
    
    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match.
    Each vertical and horizontal derivative forms separate blobs, suppressing overlapping orthogonal representations.
    They can also be summed to estimate diagonal or hypot derivatives, for blob orientation to maximize primary derivatives.
    Orientation increases primary dimension of blob to maximize match, and decreases secondary dimension to maximize difference.
    
    Subsequent union of lateral and vertical patterns: by strength only, orthogonal sign is not commeasurable?
    Initial input line could be 400 for debugging, that area in test image seems to be the most diverse  
'''

def lateral_comp(pixel_):  # comparison over x coordinate: between min_rng of consecutive pixels within each line

    ders_ = []  # tuples of complete derivatives: summation range = rng
    rng_ders_ = deque(maxlen=rng)  # array of tuples within rng of current pixel: summation range < rng
    max_index = rng - 1  # max index of rng_ders_
    pri_d, pri_m = 0, 0  # fuzzy derivatives in prior completed tuple

    for p in pixel_:  # pixel p is compared to rng of prior pixels within horizontal line, summing d and m per prior pixel
        for index, (pri_p, d, m) in enumerate(rng_ders_):

            d += p - pri_p  # fuzzy d: running sum of differences between pixel and all subsequent pixels within rng
            m += min(p, pri_p)  # fuzzy m: running sum of matches between pixel and all subsequent pixels within rng

            if index < max_index:
                rng_ders_[index] = (pri_p, d, m)
            else:
                ders_.append((pri_p, d + pri_d, m + pri_m))  # completed bilateral tuple is transferred from rng_ders_ to ders_
                pri_d = d; pri_m = m  # to complement derivatives of next rng_t_: derived from next rng of pixels

        rng_ders_.appendleft((p, 0, 0))  # new tuple with initialized d and m, maxlen displaces completed tuple from rng_t_

    ders_ += reversed(rng_ders_)  # or tuples of last rng (incomplete, in reverse order) are discarded?
    return ders_


def vertical_comp(ders_, ders2__, _dP_, dframe):
    # comparison between rng vertically consecutive pixels, forming ders2: 2D tuple of derivatives per pixel

    dP = [(0, 0, 0, 0, 0, 0, [])]  # lateral difference pattern = pri_s, I, D, Dy, V, Vy, ders2_
    dP_ = deque()  # line y - 1+ rng2
    dbuff_ = deque()  # line y- 2+ rng2: _Ps buffered by previous run of scan_P_
    new_ders2__ = deque()  # 2D: line of ders2_s buffered for next-line comp

    x = 0  # lateral coordinate of current pixel
    max_index = rng - 1  # max ders2_ index
    min_coord = rng * 2 - 1  # min x and y for form_P input
    dy, my = 0, 0  # for initial rng of lines, to reload _dy, _vy = 0, 0 in higher tuple

    for (p, d, m), (ders2_, _dy, _my) in zip(ders_, ders2__):  # pixel is compared to rng higher pixels in ders2_, summing dy and my per higher pixel
        x += 1
        index = 0
        for (_p, _d, _m, dy, my) in ders2_:  # vertical derivatives are incomplete; prefix '_' denotes higher-line variable

            dy += p - _p  # fuzzy dy: running sum of differences between pixel and all lower pixels within rng
            my += min(p, _p)  # fuzzy my: running sum of matches between pixel and all lower pixels within rng

            if index < max_index:
                ders2_[index] = (_p, d, m, dy, my)

            elif x > min_coord and y > min_coord:  # or min y is increased by x_comp on line y=0?

                _v = _m - abs(d) - ave  # _m - abs(d): projected m cancelled by negative d: d/2, + projected rdn value of overlapping dP: d/2
                vy = my + _my - abs(dy) - ave
                ders2 = _p, _d, _v, dy + _dy, vy
                dP, dP_, dbuff_, _dP_, dframe = form_P(ders2, x, dP, dP_, dbuff_, _dP_, dframe)

            index += 1

        ders2_.appendleft((p, d, m, 0, 0))  # initial dy and my = 0, new ders2 replaces completed t2 in vertical ders2_ via maxlen
        new_ders2__.append((ders2_, dy, my))  # vertically-incomplete 2D array of tuples, converted to ders2__, for next-line ycomp

    return new_ders2__, dP_, dframe  # extended in scan_P_; net_s are packed into frames


def form_P(ders2, x, P, P_, buff_, _P_, frame):  # initializes, accumulates, and terminates 1D pattern: dP | vP | dyP | vyP

    p, d, v, dy, vy = ders2  # 2D tuple of derivatives per pixel, "y" denotes vertical derivatives:
    s = 1 if d > 0 else 0  # core = 0 is negative: no selection?

    if s == P[0] or x == rng * 2:  # s == pri_s or initialized pri_s: P is continued, else terminated:
        pri_s, I, D, Dy, V, Vy, ders2_ = P[0]  # tuple in a list container
    else:
        if y == rng * 2:  # first line of Ps -> P_, _P_ is empty until vertical_comp returns P_:
            P_.append([P, x-1, []])  # empty _fork_ in the first line of _Ps, x-1: delayed P displacement
        elif x < X - 99:  # right error margin: >len(fork_P[6])?
            P_, buff_, _P_, frame = scan_P_(x-1, P, P_, buff_, _P_, frame)  # scans higher-line Ps for contiguity

        I, D, Dy, V, Vy, ders2_ = 0, 0, 0, 0, 0, []  # new P initialization

    I += p  # summed input and derivatives are accumulated as P and alt_P parameters, continued or initialized:
    D += d  # lateral D
    Dy += dy  # vertical D
    V += v  # lateral V
    Vy += vy  # vertical V
    ders2_.append(ders2)  # t2s are buffered for oriented rescan and incremental range | derivation comp

    P = [(s, I, D, Dy, V, Vy, ders2_)]
    return P, P_, buff_, _P_, frame  # accumulated within line, P_ is a buffer for conversion to _P_


def scan_P_(x, P, P_, _buff_, _P_, frame):  # P scans shared-x-coordinate _Ps in _P_, forms overlaps

    buff_ = deque()  # new buffer for displaced _Ps, for scan_P_(next P)
    fork_ = []  # _Ps connected to input P, second [] is just a container with fixed id
    ix = x - len(P[0][6])  # initial x coordinate of P
    _ix = 0  # initial x coordinate of _P

    while _ix <= x:  # while horizontal overlap between P and _P, after that: P -> P_
        if _buff_:
            [_P, _x, _fork_, roots] = _buff_.popleft()  # load _P buffered in prior run of scan_P_, if any
        elif _P_:
            [_P, _x, _fork_] = _P_.popleft()  #
            roots = 0  # number of Ps connected to current _P[(pri_s, I, D, Dy, V, Vy, ders2_)]
        else:
            break
        _ix = _x - len(_P[0][6])

        if P[0][0] == _P[0][0]:  # if s == _s: core sign match, + selective inclusion by cont eval?
            fork_.append(_P)  # P-connected _Ps, appended with blob and converted to Py_ after P_ scan
            roots += 1

        if _x > ix:  # x overlap between _P and next P: _P is buffered for next scan_P_
            buff_.append([_P, _x, _fork_, roots])
        else:     # no x overlap between _P and next P: _P is included in unique blob segment:
            ini = 1
            if y > rng * 2 + 1:  # beyond 1st line of _fork_ _Ps, else: blob_seg ini only
                if len(_fork_[0]) == 1:
                    try:
                        if _fork_[0][0][4] == 1:  # _fork roots, see blob_seg init, second [] is a _P container with id
                            _P[0] = form_blob_seg(_fork_[0][0], _P[0], _x)  # blob_seg incr: _P packed in _fork_[0]
                            ini = 0  # no initialization
                            return ini, fork_
                    except:
                        break
            if ini == 1:  # blob_seg initialization by all not-included _Ps at y > rng * 2, same fork id for all root Ps:
                try:
                   _P[0] += [[_P[0]], _x - len(_P[0][6]) / 2, 0, roots, _fork_]
                   # flat _P vars + Py_, ave_x = _x - len(_P[0][6]) / 2, Dx = 0, seg-wide _fork_
                except:
                   break
            if roots == 0:
                _P[0] += _fork_[0][0]  # _P[0] is 1st-level blob, initialized with terminated blob_segment at _fork_[0][0]
                if len(_fork_) == 0:
                    frame = term_blob(_P[0], frame)  # all root-mediated forks terminated, blob is packed into frame
                else:
                    _P[0], frame = term_blob_seg(_P[0], _fork_, frame)  # recursive root blob termination test

    buff_ += _buff_  # _buff_ is likely empty
    P_.append([P, x, fork_])  # P with no overlap to next _P is buffered for next-line scan_P_, as _P

    return [P_, buff_, _P_, frame]  # _P_ and buff_ contain only _Ps with _x => next x


def term_blob_seg(blob, fork_, frame):  # blob initiated as a terminated blob segment, then added to terminated forks in its fork_

    for index, (_blob, _fork_, roots) in enumerate(fork_):
        _blob = form_blob(_blob, blob)  # terminated blob is included into its forks blobs

        if roots == 0:
            if len(_fork_) == 0:  # no fork-mediated roots left, terminated blob is packed in frame:
                frame = term_blob(blob, frame)
            else:
                _blob, frame = term_blob_seg(_blob, _fork_, frame)  # recursive root blob termination test

    return [blob, frame]  # fork_ contains incremented blobs


def form_blob_seg(blob_seg, P, last_x):  # continued or initialized blob segment is incremented by attached _P, replace by zip?

    (s, L2, I2, D2, Dy2, V2, Vy2, ders2_), Py_, _x, Dx, fork_, roots = blob_seg  # fork_ at init, roots at term?
    s, I, D, Dy, V, Vy, ders2_ = P  # s is identical, ders2_ is a replacement

    x = last_x - len(ders2_) / 2  # median x, becomes _x in blob, replaces lx
    dx = x - _x  # conditional full comp(x) and comp(S): internal vars are secondary?
    Dx += dx  # for blob normalization and orientation eval, | += |dx| for curved max_L norm, orient?
    L2 += len(ders2_)  # ders2_ in P buffered in Py_
    I2 += I
    D2 += D
    Dy2 += Dy
    V2 += V
    Vy2 += Vy
    Py_.append((s, x, dx, I, D, Dy, V, Vy, ders2_))  # dx to normalize P before comp_P?
    blob_seg = (s, L2, I2, D2, Dy2, V2, Vy2, ders2_), Py_, _x, Dx, fork_, roots  # redundant s and ders2_

    return blob_seg


def form_blob(blob, blob_seg):  # continued or initialized network is incremented by attached blob and _root_

    (s, xb, Dxb, Lb, Ib, Db, Dyb, Vb, Vyb), blob_seg_ = blob  # 2D blob_: fork_ per layer?
    ((s, L2, I2, D2, Dy2, V2, Vy2, ders2_), x, Dx, Py_), fork_ = blob_seg  # s is redundant, ders2_ ignored
    Dxb += Dx  # for net normalization, orient eval, += |Dx| for curved max_L?
    Lb += L2
    Ib += I2
    Db += D2
    Dyb += Dy2
    Vb += V2
    Vyb += Vy2
    blob_seg_.append((x, Dx, L2, I2, D2, Dy2, V2, Vy2, Py_, fork_))  # Dx is to normalize blob before comp_P
    blob = ((s, Lb, Ib, Db, Dyb, Vb, Vyb), xb, Dxb, Py_), blob_seg_  # separate S_par tuple?

    return blob


def term_blob(blob, frame):
    ((s, Lb, Ib, Db, Dyb, Vb, Vyb), xb, Dxb, Py_), blob_seg_ = blob
    Dxf, Lf, If, Df, Dyf, Vf, Vyf, blob_ = frame
    Dxf += Dxb  # for frame normalization, orient eval, += |Dxb| for curved max_L?
    Lf += Lb
    If += Ib  # to compute averages, for dframe only: redundant for same-scope alt_frames?
    Df += Db
    Dyf += Dyb
    Vf += Vb
    Vyf += Vyb
    blob_.append((xb, Dxb, Lb, Ib, Db, Dyb, Vb, Vyb, blob_seg_))  # Dxb to normalize blob before comp_P
    frame = Dxf, Lf, If, Df, Dyf, Vf, Vyf, blob_
    return frame


def image_to_blobs(image):  # postfix '_' denotes array vs. element, prefix '_' denotes higher-line vs. lower-line variable

    _P_ = deque()  # higher-line same- d-, v-, dy-, vy- sign 1D patterns
    frame = 0, 0, 0, 0, 0, 0, 0, []  # Dxf, Lf, If, Df, Dyf, Vf, Vyf, net_
    global y
    y = 0  # vertical coordinate of current input line
    # initial input line may be set at 400, that area in test image seems to be the most diverse

    ders2_ = deque(maxlen=rng)  # vertical buffer of incomplete derivatives tuples, for fuzzy ycomp
    ders2__ = []  # vertical buffer + horizontal line: 2D array of 2D tuples, deque for speed?
    pixel_ = image[0, :]  # first line of pixels
    ders_ = lateral_comp(pixel_)  # after part_comp (pop, no t_.append) while x < rng?

    for (p, d, m) in ders_:
        ders2 = p, d, m, 0, 0  # dy, my initialized at 0
        ders2_.append(ders2)  # only one tuple per first-line ders2_
        ders2__.append((ders2_, 0, 0))  # _dy, _my initialized at 0

    for y in range(1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?

        pixel_ = image[y, :]  # vertical coordinate y is index of new line p_
        ders_ = lateral_comp(pixel_)  # lateral pixel comparison
        ders2__, _P_, frame = vertical_comp(ders_, ders2__, _P_, frame)  # vertical pixel comparison

    # frame ends, last vertical rng of incomplete ders2__ is discarded,
    # vertically incomplete P_ patterns are still inputted in scan_P_?
    return frame  # frame of 2D patterns to be outputted to level 2


# pattern filters: eventually updated by higher-level feedback, initialized here as constants:

rng = 2  # number of leftward and upward pixels compared to each input pixel
ave = 127 * rng * 2  # average match: value pattern filter
ave_rate = 0.25  # average match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/raccoon.jpg')
arguments = vars(argument_parser.parse_args())

# read image as 2d-array of pixels (gray scale):
image = cv2.imread(arguments['image'], 0).astype(int)

# or read the same image online, without cv2:
# from scipy import misc
# image = misc.face(gray=True)  # read pix-mapped image
# image = image.astype(int)

Y, X = image.shape  # image height and width

start_time = time()
blobs = image_to_blobs(image)
end_time = time() - start_time
print(end_time)

