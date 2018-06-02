import cv2
import argparse
from time import time
from collections import deque

''' A version of frame_blobs with only one blob type (dblob), to ease debugging '''

def lateral_comp(p_):  # comparison over x coordinate: between min_rng of consecutive pixels within each line

    t_ = []  # complete tuples: summation range = rng
    rng_t_ = deque(maxlen=rng)  # array of tuples within rng of current pixel: summation range < rng
    max_index = rng - 1  # max index of rng_t_
    pri_d, pri_m = 0, 0  # fuzzy derivatives in prior completed tuple

    for p in p_:  # pixel p is compared to rng of prior pixels within horizontal line, summing d and m per prior pixel:
        for index, (pri_p, d, m) in enumerate(rng_t_):

            d += p - pri_p  # fuzzy d: running sum of differences between pixel and all subsequent pixels within rng
            m += min(p, pri_p)  # fuzzy m: running sum of matches between pixel and all subsequent pixels within rng

            if index < max_index:
                rng_t_[index] = (pri_p, d, m)
            else:
                t_.append((pri_p, d + pri_d, m + pri_m))  # completed bilateral tuple is transferred from rng_t_ to t_
                pri_d = d; pri_m = m  # to complement derivatives of next rng_t_: derived from next rng of pixels

        rng_t_.appendleft((p, 0, 0))  # new tuple with initialized d and m, maxlen displaces completed tuple from rng_t_

    t_ += reversed(rng_t_)  # or tuples of last rng (incomplete, in reverse order) are discarded?
    return t_


def vertical_comp(t_, t2__, _dP_, dblob_, dframe):
    # comparison between rng vertically consecutive pixels, forming t2: 2D tuple of derivatives per pixel

    dP = 0, 0, 0, 0, 0, 0, []  # lateral difference pattern = pri_s, I, D, Dy, V, Vy, t2_
    dP_ = deque()  # line y - 1+ rng2
    dbuff_ = deque()  # line y- 2+ rng2: _Ps buffered by previous run of scan_P_
    new_t2__ = deque()  # 2D: line of t2_s buffered for next-line comp

    x = 0  # lateral coordinate of current pixel
    max_index = rng - 1  # max t2_ index
    min_coord = rng * 2 - 1  # min x and y for form_P input
    dy, my = 0, 0  # for initial rng of lines, to reload _dy, _vy = 0, 0 in higher tuple

    for (p, d, m), (t2_, _dy, _my) in zip(t_, t2__):  # pixel p is compared to rng of higher pixels in t2_, summing dy and my per higher pixel:
        x += 1
        index = 0
        for (_p, _d, _m, dy, my) in t2_:  # 2D tuples are vertically incomplete; prefix '_' denotes higher-line variable

            dy += p - _p  # fuzzy dy: running sum of differences between pixel and all lower pixels within rng
            my += min(p, _p)  # fuzzy my: running sum of matches between pixel and all lower pixels within rng

            if index < max_index:
                t2_[index] = (_p, d, m, dy, my)

            elif x > min_coord and y > min_coord:  # or min y is increased by x_comp on line y=0?
                _v = _m - ave
                vy = my + _my - ave
                t2 = _p, _d, _v, dy + _dy, vy
                dP, dP_, dbuff_, _dP_, dblob_, dframe = form_P(t2, x, dP, dP_, dbuff_, _dP_, dblob_, dframe)

            index += 1

        t2_.appendleft((p, d, m, 0, 0))  # initial dy and my = 0, new t2 replaces completed t2 in vertical t2_ via maxlen
        new_t2__.append((t2_, dy, my))  # vertically-incomplete 2D array of tuples, converted to t2__, for next-line ycomp

    return new_t2__, dP_, dblob_, dframe  # extended in scan_P_; also incomplete net_, before pack into frame?


def form_P(t2, x, P, P_, buff_, _P_, blob_, frame):  # terminates, initializes, accumulates 1D pattern: dP | vP | dyP | vyP

    p, d, v, dy, vy = t2  # 2D tuple of derivatives per pixel, "y" for vertical dimension:
    s = 1 if d > 0 else 0  # core = 0 is negative: no selection?

    if s == P[0] or x == rng * 2:  # s == pri_s or initialized pri_s: P is continued, else terminated:
        pri_s, I, D, Dy, V, Vy, t2_ = P
    else:
        if y == rng * 2:  # first line of Ps, _P_ is empty till vertical comp returns P_:
            P_.append((P, x - 1, []))  # empty _fork_ in the first line of _Ps, x-1 for delayed P displacement
        else:
            P_, buff_, _P_, blob_, frame = scan_P_(x - 1, P, P_, buff_, _P_, blob_, frame)  # scans higher-line Ps for contiguity
        I, D, Dy, V, Vy, t2_ = 0, 0, 0, 0, 0, []  # new P initialization

    I += p  # summed input and derivatives are accumulated as P and alt_P parameters, continued or initialized:
    D += d  # lateral D
    Dy += dy  # vertical D
    V += v  # lateral V
    Vy += vy  # vertical V
    t2_.append(t2)  # t2s are buffered for oriented rescan and incremental range | derivation comp

    P = s, I, D, Dy, V, Vy, t2_
    return P, P_, buff_, _P_, blob_, frame  # accumulated within line


def scan_P_(x, P, P_, _buff_, _P_, blob_, frame):  # P scans shared-x-coordinate _Ps in _P_, forms overlaps

    buff_ = deque()  # new buffer for displaced _Ps, for scan_P_(next P)
    fork_ = []  # _Ps connected to input P
    ix = x - len(P[6])  # initial x coordinate of P( pri_s, I, D, Dy, V, Vy, t2_)
    _ix = 0  # initial x coordinate of _P

    while _ix <= x:  # while horizontal overlap between P and _P, then P -> P_
        if _buff_:
            _P, _x, _fork_, roots = _buff_.popleft()  # _Ps buffered in prior run of scan_P_
        elif _P_:
            _P, _x, _fork_ = _P_.popleft()  # _P: y-2, _root_: y-3, contains blobs that replace _Ps
            roots = 0  # count of Ps connected to current _P
        else:
            break
        _ix = _x - len(_P[6])

        if P[0] == _P[0]:  # if s == _s (core sign match):
            fork_.append((_P, _fork_))  # _Ps connected to P, _fork_ for terminated blob transfer to network
            roots += 1  # count of Ps connected to _P, assigned to fork at _P displacement

        if _x > ix:  # next x <= _x:
            buff_.append((_P, _x, _fork_, roots))  # for next scan_P_
        else:  # no x overlap between _P and next P: never happens?

            if len(_fork_) == 1 and _fork_[0][1] == 1 and blob_:  # _P'_fork_ == 1 and _fork blob roots == 1 and y > rng *2
                blob = form_blob(_fork_[0], _P, _x)  # y-2 _P is packed in y-3 blob _fork_[0], binding is passed to y-1 P_?
            else:
                blob = (_P, (_x - len(P[6]) / 2), 0, [_P])  # blob init, ave_x = _x - len(t2_)/2, Dx = 0, also roots
            _P = blob  # binds blob to fork refs in its root Ps?

            if roots == 0:
                net = blob, [(blob, _fork_)]  # net is initialized with term'blob and _fork_
                del blob  # or blob[n]:  no blob = net: 0 roots to bind

                if len(_fork_) == 0:
                    frame = form_frame(net, frame)  # all root-mediated forks terminated, net is packed into frame
                else:
                    net, frame = term_blob(net, _fork_, frame)  # recursive root network termination test
            else:
                blob_.append((blob, roots, _fork_))  # new | continued blobs exposed to P_, with final roots, replace _P_,
                # no contiguous, del if roots == 0?  next pack -> net_: ref by terminated blobs forks?

    buff_ += _buff_  # _buff_ is likely empty
    P_.append((P, x, fork_))  # P with no overlap to next _P is buffered for next-line scan_P_, via y_comp

    return P_, buff_, _P_, blob_, frame  # _P_ and buff_ exclude _Ps displaced into blob_


def term_blob(net, fork_, frame):  # net starts as one terminated blob, then added to terminated forks in its fork_

    for index, (_net, _roots, _fork_) in enumerate(fork_):
        _net = form_network(_net, net)  # terminated network (blob) is included into its forks networks
        fork_[index][0] = _net  # return
        _roots -= 1

        if _roots == 0:
            if len(_fork_) == 0:  # no fork-mediated roots left, terminated net is packed in frame:
                frame = form_frame(net, frame)
            else:
                _net, frame = term_blob(_net, _fork_, frame)  # recursive root network termination test

    return net, frame  # fork_ contains incremented nets


def form_blob(blob, P, last_x):  # continued or initialized blob is incremented by attached _P, replace by zip?

    (s, L2, I2, D2, Dy2, V2, Vy2, t2_), _x, Dx, Py_ = blob
    s, I, D, Dy, V, Vy, t2_ = P  # s is identical, t2_ is a replacement

    x = last_x - len(t2_) / 2  # median x, becomes _x in blob, replaces lx
    dx = x - _x  # conditional full comp(x) and comp(S): internal vars are secondary?
    Dx += dx  # for blob normalization and orientation eval, | += |dx| for curved max_L norm, orient?
    L2 += len(t2_)  # t2_ in P buffered in Py_
    I2 += I
    D2 += D
    Dy2 += Dy
    V2 += V
    Vy2 += Vy
    Py_.append((s, x, dx, I, D, Dy, V, Vy, t2_))  # dx to normalize P before comp_P?
    blob = (s, L2, I2, D2, Dy2, V2, Vy2, t2_), _x, Dx, Py_  # redundant s and t2_

    return blob


def form_network(net, blob):  # continued or initialized network is incremented by attached blob and _root_

    (s, xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn), blob_ = net  # 2D blob_: fork_ per layer?
    ((s, L2, I2, D2, Dy2, V2, Vy2, t2_), x, Dx, Py_), fork_ = blob  # s is redundant
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


def form_frame(net, frame):
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
    blob_ = deque()  # line y- 3+ rng2: replaces _P_, exposed blobs include _Ps
    frame = 0, 0, 0, 0, 0, 0, 0, []  # Dxf, Lf, If, Df, Dyf, Vf, Vyf, net_
    global y
    y = 0  # vertical coordinate of current input line

    t2_ = deque(maxlen=rng)  # vertical buffer of incomplete quadrant tuples, for fuzzy ycomp
    t2__ = []  # vertical buffer + horizontal line: 2D array of 2D tuples, deque for speed?
    p_ = f[0, :]  # first line of pixels
    t_ = lateral_comp(p_)  # after part_comp (pop, no t_.append) while x < rng?

    for (p, d, m) in t_:
        t2 = p, d, m, 0, 0  # dy, my initialized at 0
        t2_.append(t2)  # only one tuple per first-line t2_
        t2__.append((t2_, 0, 0))  # _dy, _my initialized at 0

    for y in range(1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?

        p_ = f[y, :]  # vertical coordinate y is index of new line p_
        t_ = lateral_comp(p_)  # lateral pixel comparison
        t2__, _P_, blob_, frame = vertical_comp(t_, t2__, _P_, blob_, frame)  # vertical pixel comparison

    # frame ends, last vertical rng of incomplete t2__ is discarded,
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
