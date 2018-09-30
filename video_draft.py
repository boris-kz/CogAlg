import cv2
import argparse
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
                pri_d = d; pri_m = m  # to complement derivatives of next rng_ders_: derived from next rng of pixels

        rng_ders_.appendleft((p, 0, 0))  # new tuple with initialized d and m, maxlen displaces completed tuple from rng_ders_

    ders_ += reversed(rng_ders_)  # or tuples of last rng (incomplete, in reverse order) are discarded?
    return ders_


def vertical_comp(ders_, rng_ders2__):  # comparison between rng vertically consecutive pixels forms t2: 2D tuple of derivatives per pixel

    ders2_ = []  # tuples of complete derivatives per pixel
    new_ders2__ = deque()  # 2D: line of ders2_s buffered for next-line comp
    max_index = rng - 1  # max ders2_ index
    dy, my = 0, 0  # for initial rng of lines, to reload _dy, _vy = 0, 0 in higher tuple

    for (p, d, m), (rng_ders2_, _dy, _my) in zip(ders_, rng_ders2__):  # pixel p is compared to rng higher pixels, summing dy and my per higher pixel:
        index = 0
        for (_p, _d, _m, dy, my) in rng_ders2_:  # 2D derivatives are vertically incomplete; prefix '_' denotes higher-line variable

            dy += p - _p  # fuzzy dy: running sum of differences between pixel and all lower pixels within rng
            my += min(p, _p)  # fuzzy my: running sum of matches between pixel and all lower pixels within rng

            if index < max_index:
                rng_ders2_[index] = (_p, d, m, dy, my)  # update
            else:
                ders2_.append((_p, _d, _m, dy, my))  # output of complete ders2
            index += 1

        rng_ders2_.appendleft((p, d, m, 0, 0))  # initial dy and my = 0, new ders2 replaces completed ders2 in vertical ders2_ via maxlen
        new_ders2__.append((rng_ders2_, dy, my))  # vertically-incomplete 2D array of tuples, converted to ders2__, for next-line ycomp?

    return new_ders2__, ders2_  # no laterally incomplete tuples?


def temporal_comp(ders2_, rng_ders3__, _dP_, _blob_, sequence):

    # ders2_: a frame of 2D tuples, all scan lines are spliced into one array
    # rng_ders3__: an older frame of 3D tuple arrays, all scan lines are spliced into one array
    # comparison between rng temporally consecutive pixels, forming ders3: 3D tuple of derivatives per pixel

    dP = 0, 0, 0, 0, 0, 0, []  # lateral difference pattern = pri_s, I, D, Dy, V, Vy, ders2_
    dP_ = deque()  # Ps with attached blobs, with attached durons? vs. blobs in a frame, line y - 1+ rng2?
    dbuff_ = deque()  # line y- 2+ rng2: blobs buffered by previous run of scan_P_
    new_ders3__ = deque()  # 3D: frame of ders3_s buffered for next-frame comp

    x, y = 0, 0  # coordinates of current pixel
    max_index = rng - 1  # max ders3__ index
    min_coord = rng * 2 - 1  # min x, y, t coordinates for form_P
    dt, mt = 0, 0  # for initial rng of lines, to reload _dt, _vt = 0, 0 in higher tuple

    for (p, d, m, dy, my), (ders3_, _dt, _mt) in zip(ders2_, rng_ders3__):  # compares same-x, same-y pixels within temporal range = rng
        index = 0
        x += 1; y += 1
        for (_p, _d, _m, _dy, _my, dt, mt) in ders3_:  # temporally incomplete tuples; prefix '_' denotes prior-frame variable

            dt += p - _p   # fuzzy dt: running sum of differences between pixel and all lower pixels within rng
            mt += min(p, _p)  # fuzzy mt: running sum of matches between pixel and all lower pixels within rng

            if index < max_index:
                ders3_[index] = (_p, d, m, dy, my, dt, mt)

            elif x > min_coord and y > min_coord and t > min_coord:

                _v = _m - abs(d)/4 - ave  # _m - abs(d)/4: projected match is cancelled by negative d/2
                _vy = _my - abs(dy)/4 - ave
                vt = mt +_mt - abs(dt)/4 - ave
                ders3 = _p, _d, _v, _dy, _vy, dt + _dt, vt
                dP, dP_, dbuff_, _dP_, sequence = form_P(0, ders3, x, y, dP, dP_, dbuff_, _dP_, sequence)

            index += 1  # start form 1D, but blob_buff? # generic form_P( dP)

        rng_ders3__.appendleft((p, d, m, dy, my, 0, 0))  # initial dt and mt = 0, new ders3 replaces completed ders3 in temporal ders3_ via maxlen
        new_ders3__.append((ders3_, dt, mt))  # temporally-incomplete 2D array of tuples, converted to ders3__ for next-frame comp

    return new_ders3__, dP_, _blob_, sequence  # extended in scan_P_; net_s are packed into frames


def form_P(typ, ders3, x, y, P, P_, buff_, _P_, sequence):  # terminates, initializes, accumulates 1D pattern: dP | vP | dyP | vyP

    p, dx, vx, dy, vy, dt, vt = ders3  # 3D tuple of derivatives per pixel, "x" for lateral D, "y" for vertical D, "t" for temporal D:

    if   typ == 0: core = dx;  alt0 = vx;  alt1 = dy;  alt2 = dt;  alt3 = vy;  alt4 = vt
    elif typ == 1: core = vx;  alt0 = dx;  alt1 = vy;  alt2 = vt;  alt3 = dy;  alt4 = dt
    elif typ == 2: core = dy;  alt0 = vy;  alt1 = dx;  alt2 = dt;  alt3 = vx;  alt4 = vt
    elif typ == 3: core = vy;  alt0 = dy;  alt1 = vx;  alt2 = vt;  alt3 = dx;  alt4 = dt
    elif typ == 4: core = dt;  alt0 = vt;  alt1 = dx;  alt2 = dy;  alt3 = vx;  alt4 = vy
    else:          core = vt;  alt0 = dt;  alt1 = vx;  alt2 = vy;  alt3 = dx;  alt4 = dy

    # core: variable that defines current type of pattern, 5 alt cores define overlapping alternative-type patterns:
    # alt derivative, alt direction, alt 2nd direction, alt derivative + direction, alt 2nd derivative + direction

    s = 1 if core > 0 else 0  # core = 0 is negative: no selection?

    if s == P[0] or x == rng*2 or y == rng*2:  # s == pri_s or initialized pri_s: P is continued, else terminated:
        pri_s, I, Dx, Dy, Dt, Vx, Vy, Vt, Alt0, Alt1, Alt2, Alt3, Alt4, ders3_ = P
    else:
        if t == rng*2:  # first frame of Ps, _P_ is empty till temporal comp returns P_:
            P_.append((P, x-1, y-1, []))  # empty _fork_ in the first frame of _Ps, x-1 and y-1: delayed P displacement
        else:
            P_, buff_, _P_, sequence = scan_P_(typ, x-1, y-1, P, P_, buff_, _P_, sequence)
            # scans prior-frame Ps for contiguity
        I, Dx, Dy, Dt, Vx, Vy, Vt, Alt0, Alt1, Alt2, Alt3, Alt4, ders3_ = 0,0,0,0,0,0,0,0,0,0,0,0,[]  # P initialization

    I += p  # summed input and derivatives are accumulated as P and alt_P parameters, continued or initialized:
    Dx += dx  # lateral D
    Dy += dy  # vertical D
    Dt += dt  # temporal D
    Vx += vx  # lateral V
    Vy += vy  # vertical V
    vt += vt  # temporal V

    Alt0 += abs(alt0)  # abs Alt cores indicate value of redundant alt-core Ps, to compute P redundancy rate
    Alt1 += abs(alt1)  # vs. specific overlaps: cost >> gain in precision?
    Alt2 += abs(alt2)
    Alt3 += abs(alt3)
    Alt4 += abs(alt4)
    ders3_.append(ders3)  # ders3 is buffered for oriented rescan and incremental range | derivation comp

    P = s, I, Dx, Dy, Dt, Vx, Vy, Vt, Alt0, Alt1, Alt2, Alt3, Alt4, ders3_
    return P, P_, buff_, _P_, sequence  # accumulated within a frame

''' 
    to be added:
    
    scan_P_, form_blob, term_blob, form_net, term_net, 
    then sequential dimension add, from root() at frame end: 
    
    scan_blob_: 2D synq: bottom-up and right-to-left, potential overlap by max / min coord (from form_network),
    ave coord comp -> match, evaluated before computing specific overlap by cross-comparing blob segments, 
    
    hier contig eval: possible cont -> ave_coord comp -> dim & ave comp -> exact cont & comp?
    
    orientation in 2D only, time is neutral unless mapped to depth?
    but orient eval after persistence term: for comp over sequence?    
'''

def sequence_to_persistents(f):  # currently only a draft
    # postfix '_' denotes array vs. element, prefix '_' denotes prior- pixel, line, or frame variable

    _P_ = deque()  # higher line of same- d- | v- | dy- | vy- sign 1D patterns
    _blob_ = deque()  # prior frame of same-sign 2D blobs
    frame = 0, 0, 0, 0, 0, 0, 0, []  # Dxf, Lf, If, Df, Dyf, Vf, Vyf, net_
    global t; t = 0  # temporal coordinate of current frame

    ders2_ = deque(maxlen=rng)  # vertical buffer of incomplete ders2s, for fuzzy y_comp init
    # ders3_ = deque(maxlen=rng)  # temporal buffer of incomplete ders3s, for fuzzy t_comp init?
    rng_ders2__= []  # vertical buffer + horizontal line: 2D array of 2D tuples, deque for speed?
    rng_ders3__= deque()  # temporal buffer per pixel of a frame: 3D tuples in 3D -> 2D array

    # initialization:

    line_ = f[0]  # first frame of lines?
    pixel_= line_[0, :]  # first line of pixels
    ders_ = lateral_comp(pixel_)  # after part_comp (pop, no ders_.append) while x < rng?

    for (p, d, m) in ders_:
        ders2 = p, d, m, 0, 0  # dy, my initialized at 0
        ders2_.append(ders2)  # only one tuple per first-line ders2_
        rng_ders2__.append((ders2_, 0, 0))  # _dy, _my initialized at 0

    for y in range(1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?  or no comp, 1st frame initialization only?

        pixel_ = f[y, :]  # vertical coordinate y is index of new line pixel_
        ders_ = lateral_comp(pixel_)  # lateral pixel comparison
        ders2__, rng_ders2__ = vertical_comp(ders_, rng_ders2__)  # vertical pixel comparison

    for t in range(1, T):  # actual processing

        line_ = f[t, :]  # temporal coordinate t is index of new frame line_
        pixel_ = f[t, :]
        ders_ = lateral_comp(pixel_)  # lateral pixel comparison
        ders2__, rng_ders2__ = vertical_comp(ders_, rng_ders2__)  # vertical pixel comparison
        new_ders3__, P_, blob_, sequence = temporal_comp(ders2_, rng_ders3__, _P_, _blob_, sequence)  # temporal pixel comparison

    # sequence ends, incomplete ders3__ discarded, but vertically incomplete blobs are still inputted in scan_blob_?
    return frame  # frame of 2D patterns is outputted to level 2


# pattern filters: eventually updated by higher-level feedback, initialized here as constants:

rng = 2  # number of leftward and upward pixels compared to each input pixel
ave = 127 * rng * 2  # average match: value pattern filter
ave_rate = 0.25  # average match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/raccoon.jpg')
arguments = vars(argument_parser.parse_args())

# read image as 2d-array of pixels (gray scale):
# this is wrong for video, just a placeholder

image = cv2.imread(arguments['image'], 0).astype(int)
Y, X = image.shape  # image height and width

start_time = time()
persistents = sequence_to_persistents(image)
end_time = time() - start_time
print(end_time)
