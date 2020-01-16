import cv2
import argparse
from time import time
from line_1D_alg.utils import *

# pattern filters or hyper-parameters: eventually from higher-level feedback, initialized here as constants:

ave = 10   # |difference| between pixels that coincides with average value of mP - redundancy to overlapping dPs
# ave_m = 15 # for segmentation, or by d sign only?
ave_min = 5  # for m defined as min |d|: smaller?
ave_M = 100  # min M for initial incremental-range comparison(t_), higher cost than der_comp?
ave_D = 20   # min |D| for initial incremental-derivation comparison(d_)
ave_L = 4    # min L for sub_cluster(d)
ave_nP = 5   # average number of sub_Ps in P, to estimate intra-costs?
ave_rdn_inc = 1 + 1 / ave_nP  # 1.2
ini_y = 500
# min_rng = 1  # >1 if fuzzy pixel comparison range, for sensor-specific noise only

''' 
  line_patterns is a principal version of 1st-level 1D algorithm, contains operations: 

- Cross-compare consecutive pixels within each row of image, forming dert_ queue of derts: tuples of derivatives per pixel. 
  dert_ is then segmented by match deviation into patterns Ps: contiguous sequences of pixels that form same-sign m: +P or -P. 
  Initial match is inverse deviation of variation: m = ave_|d| - |d|, not min: brightness doesn't correlate with predictive value.

- Positive Ps: spans of pixels forming positive match, are evaluated for cross-comp of dert input param over incremented range 
  (positive match means that pixels have high predictive value, thus likely to match more distant pixels).
- Median Ps: |d| is too weak for immediate comp_d, but d sign (direction) may persist, and d sign match is partial d match.
  Then dert_ is segmented by same d sign, accumulating segment sD to evaluate der_comp within a segment.  
- Negative Ps: high-variation spans, are evaluated for cross-comp of difference, which forms higher derivatives.
  (d match = min: rng+ comp value: predictive value of difference is proportional to its magnitude, although inversely so)
  
  Both extended cross-comp forks are recursive: resulting sub-patterns are evaluated for deeper cross-comp, same as top patterns.
  The signs for these forks may be defined independently, for overlaps or gaps in spectrum, but then rdn is far more complex. 

  Initial bi-lateral cross-comp here is 1D slice of 2D 3x3 kernel, while uni-lateral d is equivalent to 2x2 kernel.
  Odd kernels preserve resolution of pixels, while 2x2 kernels preserve resolution of derivatives, in resulting derts.
  The former should be used in comp_rng and the latter in comp_d, which may alternate with intra_comp.
  
  postfix '_' denotes array name, vs. same-name elements
  prefix '_' denotes prior of two same-name variables
  prefix 'f' denotes binary flag
  '''

def cross_comp(frame_of_pixels_):  # converts frame_of_pixels to frame_of_patterns, each pattern maybe nested

    Y, X = image.shape  # Y: frame height, X: frame width
    frame_of_patterns_ = []
    for y in range(ini_y + 1, Y):
        # initialization per row:
        pixel_ = frame_of_pixels_[y, :]  # y is index of new line pixel_
        P_ = []  # row of patterns
        __p, _p = pixel_[0:2]  # each prefix '_' denotes prior
        _d = _p - __p  # initial comp, use d as bi_d, no /2: effectively *2 for back-projection
        _m = ave - abs(_d)
        # initialize P with dert_[0]:
        P = _m > 0, [], 1, __p, _d, _m, [(__p, _d, _m, None)], []  # sign, depth, L, I, D, M, dert_, sub_

        for p in pixel_[2:]:  # pixel p is compared to prior pixel _p in a row
            d = p - _p
            m = ave - abs(d)  # initial match is inverse deviation of |difference|
            bi_d = (d + _d); bi_d >>= 1  # ave bilateral difference, init only?
            bi_m = (m + _m); bi_d >>= 1  # ave bilateral match
            dert = _p, bi_d, bi_m, _d
            # accumulate or terminate mP: span of pixels forming same-sign m:
            P, P_ = form_P(P, P_, dert)
            _p, _d, _m = p, d, m
        # terminate last P in row:
        dert = _p, _d, _m, _d  # no / 2: last d and m are forward-projected to bilateral values
        P, P_ = form_P(P, P_, dert)
        P_ += [P]  # adds incomplete P
        # evaluate sub-recursion per P:
        intra_P(P_, fid=False, rdn=1, rng=1, fseg=False)  # recursive
        frame_of_patterns_ += [P_]  # line of patterns is added to frame of patterns

    return frame_of_patterns_  # frame of patterns will be output to level 2

''' Recursion extends pattern structure to 1d hierarchy and then 2d hierarchy, to be adjusted by macro-feedback:
    P_:
    fseg,  # flag: 0 for rng+ | der+, 1 for segment
    fid,   # flag: input is derived: magnitude correlates with predictive value: m = min-ave, else m = ave-|d|
    rdn,   # redundancy to higher layers, possibly lateral overlap of rng+, seg_d, der+, rdn += 1 * typ coef?
    rng,   # rng+ count  
    P:
    sign,  # 1 -> rng+, 0 -> segment | der+? 
    Dert = L, I, D, M, 
    dert_, # input for sub_segment or extend_comp
           # conditional 1d array of next layer:
    sub_,  # multiple layers of (hyper, Dert, sub_P_) from segment or extended comp, nested to depth = sub_[n] 
           # for layer-parallel access and comp, similar to frequency domain representation
           # sub_P_: flat or nested for mapping to higher-layer sub_P_ element?
    root   # reference to higher P for layer-sequential feedback 

    orders of composition: 1st: dert_, 2nd: seg_|sub_(derts), 3rd: P layer_(sub_P_(derts))? 
    line-wide layer-sequential recursion and feedback, for clarity and slice-mapped SIMD processing? 
'''

def form_P(P, P_, dert):  # initialization, accumulation, termination, recursion

    _sign, LL, L, I, D, M, dert_, sub_ = P  # each sub in sub_ is nested to depth = sub_[n]
    p, d, m, uni_d = dert
    sign = m > 0
    if sign != _sign:
        # sign change: terminate P
        P_.append((_sign, LL, L, I, D, M, dert_, sub_))  # LL(depth), L for visibility only
        LL, L, I, D, M, dert_, sub_ = [], 0, 0, 0, 0, [], []  # reset accumulated params
    # accumulate params with bilateral values:
    L += 1; I += p; D += d; M += m
    dert_ += [(p, d, m, uni_d)]  # uni_d for der_comp and segment
    P = sign, LL, L, I, D, M, dert_, sub_  # sub_ is accumulated in intra_P

    return P, P_


def intra_P(P_, fid, rdn, rng, fseg):  # evaluate for sub-recursion in line P_, filling its sub_P_ with the results

    deep_sub_ = []  # sub_ extension feedback from intra_P

    for sign, LL, L, I, D, M, dert_, sub_ in P_:  # sub_: list of lower pattern layers, nested to depth = sub_[n]

        if sign == 0:  # low-variation P, rdn+1.2: 1 + (1 / ave_nP): rdn to higher derts + ave rdn to higher sub_
            if M > ave_M * rdn and L > 4:  # rng+ eval vs. fixed cost = ave_M
                rng += 1  # n of extensions, mapped comp rng = rng**2: 1, 2, 4.., kernel = rng * 2 + 1: 3, 5, 9
                lat_sub_ = rng_comp(dert_, fid)
                lL = len(lat_sub_)  # lateral L, for visibility
                rdn += 1 / lL - 0.2  # adjust distributed rdn, estimated in intra_P
                sub_ += [[( fseg, sign, lL, fid, rdn, rng, lat_sub_ )]]  # 1st layer
                sub_ += intra_P(lat_sub_, fid, rdn+1.2, rng, fseg=False)  # recursion eval and deep layers feedback

        elif fseg:  # P is seg_P: d sign match is partial d match and pre-condition for der_comp:
            if (D > ave_D * rdn) and L > 4:  # der+ fixed cost eval
                lat_sub_ = der_comp(dert_)   # cross-comp uni_ds in dert[3]
                lL = len(lat_sub_); rdn += 1 / lL - 0.2
                sub_ += [[( fseg, sign, lL, True, rdn, rng, lat_sub_)]]  # 1st layer
                sub_ += intra_P(lat_sub_, True, rdn + 1.2, rng, fseg=False)  # deep layers feedback

        elif L > ave_L * rdn:  # high variation, filter by L only because d sign may differ
            lat_sub_ = segment(dert_)  # segment dert_ by ds: sign match covers variable cost of der+?
            lL = len(lat_sub_); rdn += 1 / lL - 0.2
            sub_ += [[( fseg, sign, lL, True, rdn, rng, lat_sub_)]]  # 1st layer
            sub_ += intra_P(lat_sub_, True, rdn+1.2, rng, fseg=True)  # will trigger der+ eval

        LL[:] = [len(sub_)]
        # each: else merge non-selected sub_Ps within P, if in max recursion depth? Eval per P_: same op, !layer

        for i, sub in enumerate(sub_):
            if sub == []: break
            try: deep_sub_[i].extend(sub)
            except IndexError: deep_sub_.append(sub)

    return deep_sub_  # add return of Dert and hypers, same in sub_[0]? [] fill if min_nP: L, LL?


def segment(dert_):  # P segmentation by same d sign: initialization, accumulation, termination

    sub_ = []  # replaces P.sub_
    _p, _d, _m, _uni_d = dert_[1]  # skip dert_[0]: no uni_d; prefix '_' denotes prior
    _sign = _uni_d > 0
    LL, L, I, D, M, seg_dert_ = [], 1, _p, _d, _m, [(_p, _d, _m, _uni_d)]
    # initialize seg_P, same as P
    for p, d, m, uni_d in dert_[2:]:
        sign = uni_d > 0
        if _sign != sign:
            sub_.append((_sign, LL, L, I, D, M, seg_dert_, []))  # terminate seg_P, same as P
            LL, L, I, D, M, seg_dert_, sub_ = [], 0, 0, 0, 0, [], []  # reset accumulated seg_P params
        _sign = sign
        L +=1; I += p; D += d; M += m  # accumulate seg_P params, or D += uni_d?
        seg_dert_.append((p, d, m, uni_d))

    sub_.append((_sign, LL, L, I, D, M, seg_dert_, []))  # pack last segment, nothing to accumulate
    # also sum Dert in seg_ and sub_?
    return sub_  # replace P.sub_


def rng_comp(dert_, fid):  # skip odd derts for sparse rng+ comp: 1 skip / 1 add, to maintain 2x overlap

    sub_P_ = []  # replaces P.sub_; prefix '_' denotes the prior of same-name variables, initialization:
    (__i, __short_bi_d, __short_bi_m, _), _, (_i, _short_bi_d, _short_bi_m, _) = dert_[0:3]
    _d = _i - __i  # bi_d = d, no /2: effective *2 for back-projection
    if fid: _m = min(__i, _i) - ave_min + __short_bi_m
    else:   _m = ave - abs(_d) + __short_bi_m  # accumulate, then ave * rng for sign eval?
    _d += __short_bi_d  # accumulate only?
    # initialize P with dert_[0]:
    sub_P = _m > 0, [], 1, __i, _d, _m, [(__i, _d, _m, None)], []  # sign, LL, L, I, D, M, dert_, sub_

    for n in range(4, len(dert_), 2):  # backward comp, ave | cumulative ders and filters?
        i, short_bi_d, short_bi_m = dert_[n][:3]  # shorter-rng dert
        d = i - _i
        if fid:  # match = min: magnitude of derived vars correlates with stability
            m = min(i, _i) - ave_min + _short_bi_m / 2  # sum m in rng
        else:  # inverse match: intensity doesn't correlate with stability
            m = ave - abs(d) + _short_bi_m
        d = d + _short_bi_d  # sum d in rng
        # d and m combine bi_d | bi_m at rng-1
        bi_d = _d + d  # bilateral difference, accum in rng
        bi_m = _m + m  # bilateral match, accum in rng
        dert = _i, bi_d, bi_m, _d
        _i, _d, _m, _short_bi_d, _short_bi_m = i, d, m, short_bi_d, short_bi_d
        # P accumulation or termination:
        sub_P, sub_P_ = form_P(sub_P, sub_P_, dert)

    # terminate last sub_P in dert_:
    dert = _i, _d, _m, _d  # no / 2: forward-project unilateral to bilateral d and m values
    sub_P, sub_P_ = form_P(sub_P, sub_P_, dert)
    sub_P_ += [sub_P]
    return sub_P_  # replaces P.sub_


def der_comp(dert_):  # cross-comp consecutive uni_ds in same-sign dert_: sign match is partial d match
    # dd and md may match across d sign, only as spec in comp_P?

    sub_P_ = []  # return to replace P.sub_, initialization:
    (_, _, _, __i), (_, _, _, _i) = dert_[1:3]  # each prefix '_' denotes prior
    _d = _i - __i  # initial comp
    _m = min(__i, _i) - ave_min  # signed,
    # initialize P with dert_[1]:
    sub_P = _m > 0, [], 1, __i, _d, _m, [(__i, _d, _m, None)], []  # sign, LL, L, I, D, M, dert_, sub_

    for dert in dert_[3:]:  # only within seg: no sign comp?
        i = dert[3]  # unilateral d
        d = i - _i   # d is dd
        m = min(i, _i) - ave_min  # md = min: magnitude of derived vars corresponds to predictive value
        bi_d = (_d + d); bi_d >>= 1  # ave bilateral d-difference per _i
        bi_m = (_m + m); bi_d >>= 1  # ave bilateral d-match per _i
        dert = _i, bi_d, bi_m, _d
        _i, _d, _m = i, d, m
        # P accumulation or termination:
        sub_P, sub_P_ = form_P(sub_P, sub_P_, dert)
    # terminate last sub_P in dert_:
    dert = _i, _d, _m, _d  # no / 2: forward-project unilateral to bilateral d and m values
    sub_P, sub_P_ = form_P(sub_P, sub_P_, dert)
    sub_P_ += [sub_P]
    return sub_P_  # replaces P.sub_


if __name__ == "__main__":
    # Parse argument (image)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image',
                                 help='path to image file',
                                 default='.//raccoon.jpg')
    arguments = vars(argument_parser.parse_args())
    # Read image
    image = cv2.imread(arguments['image'], 0).astype(int)  # load pix-mapped image
    assert image is not None, "Couldn't find image in the path!"
    image = image.astype(int)
    # same image loaded online, without cv2:
    # from scipy import misc
    # image = misc.face(gray=True).astype(int)

    start_time = time()
    # Main
    frame_of_patterns_ = cross_comp(image)
    end_time = time() - start_time
    print(end_time)

'''
2nd level cross-compares resulting patterns Ps (s, L, I, D, M, r, nested e_) and evaluates them for deeper cross-comparison. 
Depth of cross-comparison (discontinuous if generic) is increased in lower-recursion e_, then between same-recursion e_s:

comp (s)?  # same-sign only
    comp (L, I, D, M)?  # in parallel or L first, equal-weight or I is redundant?  
        comp (r)?  # same-recursion (derivation) order e_
            cross_comp (e_)
            
Then extend this 2nd level alg to a recursive meta-level algorithm

match=min: local vs. global
'''
