import cv2
import argparse
from time import time
from collections import deque

''' Comparison over a sequence frames in a video, currently only initial pixel tuple formation: 
    
    immediate pixel comparison to rng consecutive pixels over lateral x, vertical y, temporal t coordinates,
    then pattern composition out of resulting 3D t3 tuples (p, d, m, dy, my, dt, mt) per pixel,
    
    sequentially forming 1D patterns ) 2D blobs ) TD durables, which are then evaluated for:
    orientation, re-composition, incremental-dimensionality comparison, and its recursion? 
    
    2D blob synq: bottom-up and right-to-left, 
    potential overlap by max / min coord, from form_network,
    
    then comp of ave coord, as in form_blob, but results are evaluated before computing 
    specific overlap by cross-comparing constituent blobs, not oriented for inclusion? 
    
    2D * persistence: TP eval -> orientation, re-scan, recursion?
    orientation in 2D only, though summed in time?
    
    '''

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


def vertical_comp(t_, rng_t2__):  # comparison between rng vertically consecutive pixels forms t2: 2D tuple of derivatives per pixel

    t2_ = []  # complete t2 tuples
    new_t2__ = deque()  # 2D: line of t2_s buffered for next-line comp
    max_index = rng - 1  # max t2_ index
    dy, my = 0, 0  # for initial rng of lines, to reload _dy, _vy = 0, 0 in higher tuple

    for (p, d, m), (rng_t2_, _dy, _my) in zip(t_, rng_t2__):  # pixel p is compared to rng of higher pixels in t2_, summing dy and my per higher pixel:
        index = 0
        for (_p, _d, _m, dy, my) in rng_t2_:  # 2D tuples are vertically incomplete; prefix '_' denotes higher-line variable

            dy += p - _p  # fuzzy dy: running sum of differences between pixel and all lower pixels within rng
            my += min(p, _p)  # fuzzy my: running sum of matches between pixel and all lower pixels within rng

            if index < max_index:
                rng_t2_[index] = (_p, d, m, dy, my)  # update
            else:
                t2_.append((_p, _d, _m, dy, my))  # output of complete t2
            index += 1

        rng_t2_.appendleft((p, d, m, 0, 0))  # initial dy and my = 0, new t2 replaces completed t2 in vertical t2_ via maxlen
        new_t2__.append((rng_t2_, dy, my))  # vertically-incomplete 2D array of tuples, converted to t2__, for next-line ycomp?

    return new_t2__, t2_  # no laterally incomplete tuples?


def temporal_comp(time, t2_, rng_t3__, _dP_, sequence):

    # t2_: a frame of 2D tuples, all scan lines are spliced into one array
    # rng_t3__: an older frame of 3D tuple arrays, all scan lines are spliced into one array
    #  comparison between rng temporally consecutive pixels, forming t3: 3D tuple of derivatives per pixel

    dP = 0, 0, 0, 0, 0, 0, []  # lateral difference pattern = pri_s, I, D, Dy, V, Vy, t2_
    dP_ = deque()  # Ps with attached blobs, with attached durons? vs. blobs in a frame, line y - 1+ rng2?
    dbuff_ = deque()  # line y- 2+ rng2: blobs buffered by previous run of scan_P_
    new_t3__ = deque()  # 3D: frame of t3_s buffered for next-frame comp

    x, y = 0, 0  # coordinates of current pixel
    max_index = rng - 1  # max t3__ index
    min_coord = rng * 2 - 1  # min x, y, t for form_P input
    dt, mt = 0, 0  # for initial rng of lines, to reload _dt, _vt = 0, 0 in higher tuple

    for (p, d, m, dy, my), (t3_, _dt, _mt) in zip(t2_, rng_t3__):  # compares same-x, same-y pixels within temporal range = rng
        index = 0
        x += 1; y += 1
        for (_p, _d, _m, _dy, _my, dt, mt) in t3_:  # temporally incomplete tuples; prefix '_' denotes prior-frame variable

            dt += p - _p   # fuzzy dt: running sum of differences between pixel and all lower pixels within rng
            mt += min(p, _p)  # fuzzy mt: running sum of matches between pixel and all lower pixels within rng

            if index < max_index:
                t3_[index] = (_p, d, m, dy, my, dt, mt)

            elif x > min_coord and y > min_coord and time > min_coord:
                _v = _m - ave
                _vy = _my - ave
                vt = mt +_mt - ave
                t3 = _p, _d, _v, _dy, _vy, dt + _dt, vt

                dP, dP_, dbuff_, _dP_, sequence = form_P(t3, x, dP, dP_, dbuff_, _dP_, sequence)  # start form 1D, but blob_buff?
            index += 1

        rng_t3__.appendleft((p, d, m, dy, my, 0, 0))  # initial dt and mt = 0, new t3 replaces completed t3 in temporal t3_ via maxlen
        new_t3__.append((t3_, dt, mt))  # temporally-incomplete 2D array of tuples, converted to t3__ for next-frame comp

    return new_t3__, dP_, sequence  # extended in scan_P_; net_s are packed into frames


def form_P(typ, t2, x, P, P_, buff_, _P_, sequence):  # terminates, initializes, accumulates 1D pattern: dP | vP | dyP | vyP

    p, d, v, dy, vy = t2  # 2D tuple of derivatives per pixel, "y" for vertical dimension:

    if   typ == 0: core = d; alt_der = v; alt_dir = dy; alt_both = vy  # core: variable that defines current type of pattern,
    elif typ == 1: core = v; alt_der = d; alt_dir = vy; alt_both = dy  # alt cores define overlapping alternative-type patterns:
    elif typ == 2: core = dy; alt_der = vy; alt_dir = d; alt_both = v  # alt derivative, alt direction, alt derivative_and_direction
    else:          core = vy; alt_der = dy; alt_dir = v; alt_both = d

    s = 1 if core > 0 else 0  # core = 0 is negative: no selection?

    if s == P[0] or x == rng*2:  # s == pri_s or initialized pri_s: P is continued, else terminated:
        pri_s, I, D, Dy, V, Vy, alt_Der, alt_Dir, alt_Both, t2_ = P
    else:
        if y == rng*2:  # first line of Ps, _P_ is empty till vertical comp returns P_:
            P_.append((P, x-1, []))  # empty _fork_ in the first line of _Ps, x-1 for delayed P displacement
        else:
            P_, buff_, _P_, blob_, frame = scan_P_(typ, x-1, P, P_, buff_, _P_, blob_, frame)  # scans higher-line Ps for contiguity
        I, D, Dy, V, Vy, alt_Der, alt_Dir, alt_Both, t2_ = 0,0,0,0,0,0,0,0,[]  # new P initialization

    I += p  # summed input and derivatives are accumulated as P and alt_P parameters, continued or initialized:
    D += d    # lateral D
    Dy += dy  # vertical D
    V += v    # lateral V
    Vy += vy  # vertical V
    alt_Der += abs(alt_der)  # abs alt cores indicate value of alt-core Ps, to compute P redundancy rate
    alt_Dir += abs(alt_dir)  # vs. specific overlaps: cost > gain in precision?
    alt_Both+= abs(alt_both)
    t2_.append(t2)  # t2s are buffered for oriented rescan and incremental range | derivation comp

    P = s, I, D, Dy, V, Vy, alt_Der, alt_Dir, alt_Both, t2_
    return P, P_, buff_, _P_, sequence  # accumulated within line


''' Color: primarily white, internal sub-patterns per relative color, not cross-compared because already complementary? 

    recursive access of compositionally-lower levels of pattern: normalized for comp if min d(dim) -> r(dim)? 
'''


def sequence_to_durables(f):  # postfix '_' distinguishes array vs. element, prefix '_' distinguishes higher-line vs. lower-line variable

    _P_ = deque()  # higher-line same- d-, v-, dy-, vy- sign 1D patterns
    frame = 0, 0, 0, 0, 0, 0, 0, []  # Dxf, Lf, If, Df, Dyf, Vf, Vyf, net_
    global y
    y = 0  # vertical coordinate of current input line

    t2_ = deque(maxlen=rng)  # vertical buffer of incomplete quadrant tuples, for fuzzy ycomp
    rng_t2__= []  # vertical buffer + horizontal line: 2D array of 2D tuples, deque for speed?
    p_ = f[0, :]  # first line of pixels
    t_ = lateral_comp(p_)  # after part_comp (pop, no t_.append) while x < rng?

    for (p, d, m) in t_:
        t2 = p, d, m, 0, 0  # dy, my initialized at 0
        t2_.append(t2)  # only one tuple per first-line t2_
        rng_t2__.append((t2_, 0, 0))  # _dy, _my initialized at 0

    for y in range(1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?

        p_ = f[y, :]  # vertical coordinate y is index of new line p_
        t_ = lateral_comp(p_)  # lateral pixel comparison
        t2__, rng_t2__ = vertical_comp(t_, rng_t2__)  # vertical pixel comparison

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
durables = sequence_to_durables(image)
end_time = time() - start_time
print(end_time)
