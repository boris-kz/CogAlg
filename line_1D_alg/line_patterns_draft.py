import cv2
import argparse
from time import time
from line_1D_alg.utils import *
from itertools import zip_longest

''' 
  line_patterns is a principal version of 1st-level 1D algorithm, contains operations: 

- Cross-compare consecutive pixels within each row of image, forming dert_ queue of derts: tuples of derivatives per pixel. 
  dert_ is then segmented into patterns mPs and dPs: contiguous sequences of pixels forming same-sign match or difference. 
  Initial match is inverse deviation of variation: m = ave_|d| - |d|, rather than minimum for directly defined match: 
  albedo or intensity of reflected light doesn't correlate with predictive value of the object that reflects it.

- Match patterns mPs are spans of inputs forming same-sign match. Positive mPs contain high-match pixels, which are likely 
  to match more distant pixels. Thus, positive mPs are evaluated for cross-comp of pixels over incremented range.
- Difference patterns dPs are spans of inputs forming same-sign ds. d sign match is a precondition for d match, so only
  same-sign spans (dPs) are evaluated for cross-comp of constituent differences, which forms higher derivatives.
  (d match = min: rng+ comp value: predictive value of difference is proportional to its magnitude, although inversely so)
  
  Both extended cross-comp forks are recursive: resulting sub-patterns are evaluated for deeper cross-comp, same as top patterns.
  Both forks currently process all inputs (full overlap), but they can be exclusive or partly overlapping to reduce redundancy. 

  Initial bi-lateral cross-comp here is 1D slice of 2D 3x3 kernel, while uni-lateral d is equivalent to 2x2 kernel.
  Odd kernels preserve resolution of pixels, while 2x2 kernels preserve resolution of derivatives, in resulting derts.
  The former should be used in rng_comp and the latter in der_comp, which may alternate with intra_P.
  
  postfix '_' denotes array name, vs. same-name elements
  prefix '_' denotes prior of two same-name variables
  prefix 'f' denotes binary flag
  '''
# pattern filters or hyper-parameters: eventually from higher-level feedback, initialized here as constants:

ave = 15   # |difference| between pixels that coincides with average value of mP - redundancy to overlapping dPs
ave_min = 2  # for m defined as min |d|: smaller?
ave_M = 50   # min M for initial incremental-range comparison(t_), higher cost than der_comp?
ave_D = 5    # min |D| for initial incremental-derivation comparison(d_)
ave_nP = 5   # average number of sub_Ps in P, to estimate intra-costs? ave_rdn_inc = 1 + 1 / ave_nP # 1.2
ave_rdm =.5  # average dm / m, to project bi_m = m * 1.5
ini_y = 500

def cross_comp(frame_of_pixels_):  # converts frame_of_pixels to frame_of_patterns, each pattern maybe nested

    Y, X = image.shape  # Y: frame height, X: frame width
    frame_of_patterns_ = []

    for y in range(ini_y + 1, Y):  # y is index of new line pixel_
        # initialization:
        pixel_ = frame_of_pixels_[y, :]
        dert_ = []
        __p, _p = pixel_[0:2]  # each prefix '_' denotes prior
        _d = _p - __p  # initial comparison
        _m = ave - abs(_d)
        dert_.append((__p, None, _m * 1.5))  # project _m to bilateral m, first dert is for comp_P only?

        for p in pixel_[2:]:  # pixel p is compared to prior pixel _p in a row
            d = p - _p
            m = ave - abs(d)  # initial match is inverse deviation of |difference|
            dert_.append((_p, _d, m + _m))  # pack dert: prior p, prior d, bilateral match
            _p, _d, _m = p, d, m
        dert_.append((_p, _d, _m * 1.5))  # unilateral d, forward-project last m to bilateral m

        mP_ = form_mP_(dert_)  # forms m-sign patterns
        intra_mP_(mP_, fid=False, rdn=1, rng=3)  # evaluates sub-recursion per mP

        frame_of_patterns_ += [[mP_]]  # line of patterns is added to frame of patterns

    return frame_of_patterns_  # frame of patterns will be output to level 2


def form_mP_(P_dert_):  # initialization, accumulation, termination

    P_ = []  # initialization:
    p, d, m = P_dert_[0]
    _sign = m > 0
    if d is None: D = 0
    else: D = d
    L, I, M, dert_, sub_H = 1, p, m, [(p, d, m)], []  # sub_Le in sub_H is nested to depth = sub_H[n]

    for p, d, m in P_dert_[1]:  # cluster P_derts by m | d sign
        sign = m > 0
        if sign != _sign:  # sign change: terminate P
            P_.append((_sign, L, I, D, M, dert_, sub_H))
            L, I, D, M, dert_, sub_H = 0, 0, 0, 0, [], []
            # reset params
        L += 1; I += p; D += d; M += m  # accumulate params, bilateral m: for eval per pixel
        dert_ += [(p, d, m)]
        _sign = sign

    P_.append((_sign, L, I, D, M, dert_, sub_H))  # incomplete P

    return P_


def form_dP_(P_dert_):  # pattern initialization, accumulation, termination, parallel der+ and rng+?

    P_ = []  # initialization:
    p, d, m = P_dert_[1]  # skip dert_[0]: d is None
    _sign = d > 0
    L, I, D, M, dert_, sub_H = 1, p, 0, m, [(p, d, m)], []  # sub_Le in sub_H is nested to depth = sub_H[n]

    for p, d, m in P_dert_[2]:  # cluster P_derts by d sign
        sign = d > 0
        if sign != _sign:  # sign change: terminate P
            P_.append((_sign, L, I, D, M, dert_, sub_H))
            L, I, D, M, dert_, sub_H = 0, 0, 0, 0, [], []
            # reset accumulated params
        L += 1; I += p; D += d; M += m  # accumulate params, bilateral m: for eval per pixel
        dert_ += [(p, d, m)]
        _sign = sign

    P_.append((_sign, L, I, D, M, dert_, sub_H))  # incomplete P
    return P_


''' Recursion in intra_P extends pattern with sub_: hierarchy of sub-patterns, to be adjusted by macro-feedback:
    P_:
    fdP,  # flag: select dP vs. mP forks in form_P_ and intra_P
    fid,  # flag: input is derived: magnitude correlates with predictive value: m = min-ave, else m = ave-|d|
    rdn,  # redundancy to higher layers, possibly lateral overlap of rng+ & der+, rdn += 1 * typ coef?
    rng,  # comparison range
    P:
    sign,  # of core param: m | d 
    Dert = L, I, D, M, 
    dert_, # input for extended comp
    dsub_, # multiple layers of (hyper, Dert, sub_P_) from segment or extended comp, nested to depth = sub_[n], 
    rsub_, # for layer-parallel access and comp, similar to frequency domain representation
           # sub_P_: flat or nested for mapping to higher-layer sub_P_ element?
    root   # reference to higher P for layer-sequential feedback 

    orders of composition: 1st: dert_, 2nd: lateral_sub_( derts), 3rd: sub_( lateral_sub_( derts))? 
    line-wide layer-sequential recursion and feedback, for clarity and slice-mapped SIMD? 
'''

def intra_mP_(P_, fid, rdn, rng):  # evaluate for sub-recursion in line P_, fil sub_P_ with results

    deep_sub_ = []  # intra_P initializes sub_hierarchy with 1st sub_P_ layer, extending root sub_H_ by feedback
    adj_M_proj = 0  # project adjacent P M on current P span, contrast value

    for sign, L, I, D, M, dert_, sub_H_ in P_:  # each sub in sub_H_ is nested to depth = sub_H_[n]

        if M > ave_M * rdn and L > 4:  # low-variation span, eval comp at rng*3 (2+1): 1, 3, 9, kernel: 3, 7, 19

            r_dert_ = rng_comp(dert_, fid)  # rng+ comp, skip predictable next dert
            sub_mP_ = form_mP_(r_dert_); lL = len(sub_mP_)
            sub_H_ += [[(lL, False, fid, rdn, rng, sub_mP_)]]  # 1st layer, Dert=[], fill if lL > min?
            sub_H_ += intra_mP_(sub_mP_, fid, rdn + 1 + 1 / lL, rng*2 + 1)  # feedback, LL[:] = [len(sub_H_)]

        elif ~sign and min(adj_M_proj, abs(D)) > ave_D * rdn and L > 3:  # max value of abs_D is PM projected on neg_mP
            ''' 
            comb_m = comb_M / comb_S: 
            ave m/ complemented span, combined rdn projection: cross-sign M cancels-out?
            not co-derived but co-projected m?
            edge projection value|cost = comb_m * |D| -> der+, doesn't affect rng+: local and primary?
            
            same-sign comp: parallel edges?
            cross-sign comp: M - (~M/2 * rL) -> contrast as 1D difference?
            '''

            sub_dP_ = form_dP_(dert_); lL = len(sub_dP_)  # cluster by d sign match: partial d match, else no der+
            sub_H_ += [[(lL, True, 1, rdn, rng, sub_dP_)]]  # 1st layer, Dert=[], fill if lL > min?
            sub_H_ += intra_dP_(sub_dP_, adj_M_proj, rdn + 1 + 1 / lL, rng+1)  # der_comp eval per dP

        deep_sub_ = [deep_sub + sub_H_ for deep_sub, sub_H_ in zip_longest(deep_sub_, sub_H_, fillvalue=[])]
        # deep_sub_ and deep_dsub_ are spliced into deep_sub_ hierarchy

    return sub_H_  # or deep_sub_H_?

def intra_dP_(P_, adjacent_PM, rdn, rng):

    sub_H_ = []  # intra_P initializes sub_hierarchy with 1st sub_P_ layer, extending root sub_H_ by feedback
    for sign, L, I, D, M, dert_, sub_H_ in P_:  # each sub in sub_H_ is nested to depth = sub_H_[n]

        if min(adjacent_PM, abs(D)) > ave_D * rdn and L > 3:  # max value of abs_D is PM

            d_dert_ = der_comp(dert_)
            sub_mP_ = form_mP_(d_dert_); lL = len(sub_mP_)
            sub_H_ += [[(lL, False, rdn, rng, sub_mP_)]]  # 1st layer, Dert=[], fill if lL > min?
            sub_H_ += intra_mP_(sub_mP_, rdn + 1 + 1 / lL, rng*2 + 1)  # feedback, LL[:] = [len(sub_H_)]

    return sub_H_  # fill layer Dert if n_sub_P > min?


def intra_P_(P_, rdn, rng, fdP, fid):  # evaluate for sub-recursion in line P_, filling its sub_P_ with the results

    deep_sub_ = []  # intra_P recursion extends rsub_ and dsub_ hierarchies by sub_P_ layer
    for sign, dLL, rLL, L, I, D, M, dert_, dsub_, rsub_ in P_:  # each sub in sub_ is nested to depth = sub_[n]

        if fdP:  # P = dP: d sign match is partial d match, precondition for der+, or in -mPs to avoid overlap
            if abs(D) > ave_D * rdn and L > 3:  # cross-comp uni_ds at rng+1:
                ext_dert_ = der_comp(dert_)
            else:
                ext_dert_ = []
        elif sign:  # P = positive mP: low-variation span, eval comp at rng*3 (2+1): 1, 3, 9, kernel: 3, 7, 19
            if M > ave_M * rdn and L > 4:  # skip comp of predictable next dert:
                ext_dert_ = rng_comp(dert_, fid)
            else:
                ext_dert_ = []  # also merge not-selected P into non_P?
        else:
            ext_dert_ = []  # new dert_ from extended- range or derivation comp
        if ext_dert_:

            sub_dP_ = form_dP_(ext_dert_); lL = len(sub_dP_)
            dsub_ += [[(lL, True, True, rdn, rng, sub_dP_)]]  # 1st layer: lL, fdP, fid, rdn, rng, sub_P_
            dsub_ += intra_P_(sub_dP_, True, True, rdn + 1 + 1 / lL, rng+1)  # deep layers feedback
            dLL[:] = [len(dsub_)]   # deeper P rdn + 1: rdn to higher derts, + 1 / lL: rdn to higher sub_

            sub_mP_ = form_mP_(ext_dert_); lL = len(sub_mP_)
            rsub_ += [[(lL, False, fid, rdn, rng, sub_mP_)]]  # 1st layer, Dert=[], fill if lL > min?
            rsub_ += intra_P_(sub_mP_, False, fid, rdn + 1 + 1 / lL, rng*2 + 1)  # deep layers feedback
            rLL[:] = [len(rsub_)]

            deep_sub_ = [deep_sub + sub_H_ for deep_sub, sub_H_ in zip_longest(deep_sub_, sub_H_, fillvalue=[])]
            # deep_sub_ and deep_dsub_ are spliced into deep_sub_ hierarchy
            # fill layer Dert if n_sub_P > min
    return deep_sub_


def rng_comp(dert_, fid):  # skip odd derts for sparse rng+ comp: 1 skip / 1 add, to maintain 2x overlap

    rdert_ = []   # prefix '_' denotes the prior of same-name variables, initialization:
    (__i, _, __short_rng_m), _, (_i, _short_rng_d, _short_rng_m) = dert_[0:3]  # no __short_rng_d
    _d = _i - __i
    if fid: _m = min(__i, _i) - ave_min
    else:   _m = ave - abs(_d)  # no ave * rng: m and d value is cumulative
    _rng_m = _m * 1.5 + __short_rng_m  # back-project bilateral m
    rdert_.append((__i, None, _rng_m))   # no _rng_d = _d + __short_rng_d

    for n in range(4, len(dert_), 2):  # backward comp
        i, short_rng_d, short_rng_m = dert_[n]  # shorter-rng dert
        d = i - _i
        if fid: m = min(i, _i) - ave_min  # match = min: magnitude of derived vars correlates with stability
        else:   m = ave - abs(d)  # inverse match: intensity doesn't correlate with stability
        rng_d = _d + _short_rng_d      # difference accumulated in rng
        rng_m = _m + m + _short_rng_m  # bilateral match accumulated in rng
        rdert_.append((_i, rng_d, rng_m))
        _i, _d, _m, _short_rng_d, _short_rng_m =\
            i, d, m, short_rng_d, short_rng_m

    rdert_.append((_i, _d + _short_rng_d, _m * 1.5 + _short_rng_m))  # forward-project m to bilateral m
    return rdert_


def der_comp(dert_):  # cross-comp consecutive uni_ds in same-sign dert_: sign match is partial d match
    # dd and md may match across d sign, but likely in high-match area, spliced by spec in comp_P?

    ddert_ = []   # initialization:
    (_, __i, _), (_, _i, _) = dert_[1:3]  # each prefix '_' denotes prior
    __i = abs(__i); _i = abs(_i)
    _d = _i - __i  # initial comp
    _m = min(__i, _i) - ave_min
    ddert_.append((__i, None, _m * 1.5))  # no __d, back-project __m = _m * .5

    for dert in dert_[3:]:
        i = abs(dert[1])  # unilateral d, same sign in dP
        d = i - _i   # d is dd
        m = min(i, _i) - ave_min  # md = min: magnitude of derived vars corresponds to predictive value
        ddert_.append((_i, _d, _m + m))  # unilateral _d and bilateral m per _i
        _i, _d, _m = i, d, m

    ddert_.append((_i, _d, _m * 1.5))  # forward-project bilateral m
    return ddert_


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
        cross_comp (sub_)?  # same-recursion (derivation) order elements
            cross_comp (dert_)
            
Then extend this 2nd level alg to a recursive meta-level algorithm 
'''