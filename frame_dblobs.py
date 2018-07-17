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
'''

def lateral_comp(pixel_):  # comparison over x coordinate: between min_rng of consecutive pixels within each line

    ders_ = []  # tuples of complete derivatives: summation range = rng
    rng_ders_ = deque(maxlen=rng)  # array of tuples within rng of current pixel: summation range < rng
    max_index = rng - 1  # max index of rng_ders_
    pri_d, pri_m = 0, 0  # fuzzy derivatives in prior completed tuple

    for p in pixel_:  # pixel p is compared to rng of prior pixels within horizontal line, summing d and m per prior pixel:
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

    dP = 0, 0, 0, 0, 0, 0, []  # lateral difference pattern = pri_s, I, D, Dy, V, Vy, ders2_
    dP_ = deque()  # line y - 1+ rng2
    dbuff_ = deque()  # line y- 2+ rng2: _Ps buffered by previous run of scan_P_
    new_ders2__ = deque()  # 2D: line of ders2_s buffered for next-line comp

    x = 0  # lateral coordinate of current pixel
    max_index = rng - 1  # max ders2_ index
    min_coord = rng * 2 - 1  # min x and y for form_P input
    dy, my = 0, 0  # for initial rng of lines, to reload _dy, _vy = 0, 0 in higher tuple

    for (p, d, m), (ders2_, _dy, _my) in zip(ders_, ders2__):  # pixel is compared to rng higher pixels in ders2_, summing dy and my per higher pixel:
        x += 1
        index = 0
        for (_p, _d, _m, dy, my) in ders2_:  # vertical derivatives are incomplete; prefix '_' denotes higher-line variable

            dy += p - _p  # fuzzy dy: running sum of differences between pixel and all lower pixels within rng
            my += min(p, _p)  # fuzzy my: running sum of matches between pixel and all lower pixels within rng

            if index < max_index:
                ders2_[index] = (_p, d, m, dy, my)

            elif x > min_coord and y > min_coord:  # or min y is increased by x_comp on line y=0?

                _v = _m - abs(d)/4 - ave  # _m - abs(d)/4: projected match is cancelled by negative d/2
                vy = my + _my - abs(dy)/4 - ave
                ders2 = _p, _d, _v, dy + _dy, vy
                dP, dP_, dbuff_, _dP_, dframe = form_P(ders2, x, dP, dP_, dbuff_, _dP_, dframe)

            index += 1

        ders2_.appendleft((p, d, m, 0, 0))  # initial dy and my = 0, new ders2 replaces completed t2 in vertical ders2_ via maxlen
        new_ders2__.append((ders2_, dy, my))  # vertically-incomplete 2D array of tuples, converted to ders2__, for next-line ycomp

    return new_ders2__, dP_, dframe  # extended in scan_P_; net_s are packed into frames


def form_P(ders2, x, P, P_, buff_, _P_, frame):  # initializes, accumulates, and terminates 1D pattern: dP | vP | dyP | vyP

    p, d, v, dy, vy = ders2  # 2D tuple of derivatives per pixel, "y" for vertical dimension:
    s = 1 if d > 0 else 0  # core = 0 is negative: no selection?

    if s == P[0] or x == rng * 2:  # s == pri_s or initialized pri_s: P is continued, else terminated:
        pri_s, I, D, Dy, V, Vy, ders2_ = P
    else:
        if y == rng * 2:  # first line of Ps -> P_, _P_ is empty till vertical_comp returns P_:
            P_.append((P, x-1, []))  # empty _fork_ in the first line of _Ps, x-1: delayed P displacement
        else:
            P_, buff_, _P_, frame = scan_P_(x-1, P, P_, buff_, _P_, frame)  # scans higher-line Ps for contiguity
        I, D, Dy, V, Vy, ders2_ = 0, 0, 0, 0, 0, []  # new P initialization

    I += p  # summed input and derivatives are accumulated as P and alt_P parameters, continued or initialized:
    D += d  # lateral D
    Dy += dy  # vertical D
    V += v  # lateral V
    Vy += vy  # vertical V
    ders2_.append(ders2)  # t2s are buffered for oriented rescan and incremental range | derivation comp

    P = s, I, D, Dy, V, Vy, ders2_
    return P, P_, buff_, _P_, frame  # accumulated within line


def scan_P_(x, P, P_, _buff_, _P_, frame):  # P scans shared-x-coordinate _Ps in _P_, forms overlaps

    buff_ = deque()  # new buffer for displaced _Ps, for scan_P_(next P)
    fork_ = []  # _Ps connected to input P
    ix = x - len(P[6])  # initial x coordinate of P( pri_s, I, D, Dy, V, Vy, ders2_)
    _ix = 0  # initial x coordinate of _P

    while _ix <= x:  # while horizontal overlap between P and _P, after that: P -> P_
        if _buff_:
            _P, _x, _fork_, root_ = _buff_.popleft()  # load _P buffered in prior run of scan_P_, if any
        elif _P_:
            _P, _x, _fork_ = _P_.popleft()  # load _P: y-2, _root_: y-3, starts empty, then contains blobs that replace _Ps
            root_ = []  # Ps connected to current _P
        else:
            break
        _ix = _x - len(_P[6])

        if P[0] == _P[0]:  # if s == _s: core sign match, also selective inclusion by cont eval?
            fork_.append([])  # mutable placeholder for blobs connected to P, filled after _P inclusion with complete root_
            root_.append(fork_[len(fork_)-1])  # binds forks to blob

        if _x > ix:  # x overlap between _P and next P: _P is buffered for next scan_P_, else included in blob:
            buff_.append((_P, _x, _fork_, root_))
        else:
            if len(_fork_) == 1 and _fork_[0][0][5] == 1 and y > rng * 2 + 1 and x < X - 99:  # no fork blob if x < X - len(fork_P[6])?
                # if blob _fork_ == 1 and _fork roots == 1, always > 0: a bug probably appends fork_ outside scan_P_?
                blob = form_blob(_fork_[0], _P, _x)  # y-2 _P is packed in y-3 _fork_[0] blob +__fork_
            else:
                ave_x = _x - len(_P[6]) / 2  # average x of P: always integer?
                blob = _P, [_P], ave_x, 0, _fork_, len(root_)  # blob init, Dx = 0, no new _fork_ for continued blob

            if len(root_) == 0:  # never happens, probably due to the same bug
                net = blob, [blob]  # first-level net is initialized with terminated blob, no root_ to rebind
                if len(_fork_) == 0:
                    frame = term_network(net, frame)  # all root-mediated forks terminated, net is packed into frame
                else:
                    net, frame = term_blob(net, _fork_, frame)  # recursive root network termination test
            else:
                while root_:  # no root_ in blob: no rebinding to net at roots == 0
                    root_fork = root_.pop()  # ref to referring fork, verify?
                    root_fork.append(blob)  # fork binding, no convert to tuple: forms a new object?

    buff_ += _buff_  # _buff_ is likely empty
    P_.append((P, x, fork_))  # P with no overlap to next _P is buffered for next-line scan_P_, via y_comp

    return P_, buff_, _P_, frame  # _P_ and buff_ contain only _Ps with _x => next x


def term_blob(net, fork_, frame):  # net starts as one terminated blob, then added to terminated forks in its fork_

    for index, (_net, _fork_, roots) in enumerate(fork_):
        _net = form_network(_net, net)  # terminated network is included into its forks networks

        if roots == 0:
            if len(_fork_) == 0:  # no fork-mediated roots left, terminated net is packed in frame:
                frame = term_network(net, frame)
            else:
                _net, frame = term_blob(_net, _fork_, frame)  # recursive root network termination test

    return net, frame  # fork_ contains incremented nets


def form_blob(blob, P, last_x):  # continued or initialized blob is incremented by attached _P, replace by zip?

    (s, L2, I2, D2, Dy2, V2, Vy2, ders2_), Py_, _x, Dx, fork_, roots = blob  # fork_ at init, roots at term?
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
    blob = (s, L2, I2, D2, Dy2, V2, Vy2, ders2_), Py_, _x, Dx, fork_, roots  # redundant s and ders2_

    return blob


def form_network(net, blob):  # continued or initialized network is incremented by attached blob and _root_

    (s, xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn), blob_ = net  # 2D blob_: fork_ per layer?
    ((s, L2, I2, D2, Dy2, V2, Vy2, ders2_), x, Dx, Py_), fork_ = blob  # s is redundant, ders2_ ignored
    Dxn += Dx  # for net normalization, orient eval, += |Dx| for curved max_L?
    Ln += L2
    In += I2
    Dn += D2
    Dyn += Dy2
    Vn += V2
    Vyn += Vy2
    blob_.append((x, Dx, L2, I2, D2, Dy2, V2, Vy2, Py_, fork_))  # Dx to normalize blob before comp_P
    net = ((s, Ln, In, Dn, Dyn, Vn, Vyn), xn, Dxn, Py_), blob_  # separate S_par tuple?

    return net


def term_network(net, frame):
    ((s, Ln, In, Dn, Dyn, Vn, Vyn), xn, Dxn, Py_), blob_ = net
    Dxf, Lf, If, Df, Dyf, Vf, Vyf, net_ = frame
    Dxf += Dxn  # for frame normalization, orient eval, += |Dxn| for curved max_L?
    Lf += Ln
    If += In  # to compute averages, for dframe only: redundant for same-scope alt_frames?
    Df += Dn
    Dyf += Dyn
    Vf += Vn
    Vyf += Vyn
    net_.append((xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn, blob_))  # Dxn to normalize net before comp_P
    frame = Dxf, Lf, If, Df, Dyf, Vf, Vyf, net_
    return frame


def image_to_blobs(f):  # postfix '_' distinguishes array vs. element, prefix '_' distinguishes higher-line vs. lower-line variable

    _P_ = deque()  # higher-line same- d-, v-, dy-, vy- sign 1D patterns
    frame = 0, 0, 0, 0, 0, 0, 0, []  # Dxf, Lf, If, Df, Dyf, Vf, Vyf, net_
    global y
    y = 0  # vertical coordinate of current input line

    ders2_ = deque(maxlen=rng)  # vertical buffer of incomplete derivatives tuples, for fuzzy ycomp
    ders2__ = []  # vertical buffer + horizontal line: 2D array of 2D tuples, deque for speed?
    pixel_ = f[0, :]  # first line of pixels
    ders_ = lateral_comp(pixel_)  # after part_comp (pop, no t_.append) while x < rng?

    for (p, d, m) in ders_:
        ders2 = p, d, m, 0, 0  # dy, my initialized at 0
        ders2_.append(ders2)  # only one tuple per first-line ders2_
        ders2__.append((ders2_, 0, 0))  # _dy, _my initialized at 0

    for y in range(1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?

        pixel_ = f[y, :]  # vertical coordinate y is index of new line p_
        ders_ = lateral_comp(pixel_)  # lateral pixel comparison
        ders2__, _P_, frame = vertical_comp(ders_, ders2__, _P_, frame)  # vertical pixel comparison

    # frame ends, last vertical rng of incomplete ders2__ is discarded,
    # but vertically incomplete P_ patterns are still inputted in scan_P_?
    return frame  # frame of 2D patterns is outputted to level 2


# pattern filters: eventually updated by higher-level feedback, initialized here as constants:

rng = 2  # number of leftward and upward pixels compared to each input pixel
ave = 127 * rng * 2  # average match: value pattern filter
ave_rate = 0.25  # average match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/racoon.jpg')
arguments = vars(argument_parser.parse_args())

# read image as 2d-array of pixels (gray scale):
image = cv2.imread(arguments['image'], 0).astype(int)
Y, X = image.shape  # image height and width

start_time = time()
blobs = image_to_blobs(image)
end_time = time() - start_time
print(end_time)

