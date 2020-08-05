import cv2
import argparse
from time import time
from utils import *
from itertools import zip_longest
''' 
  line_patterns is a principal version of 1st-level 1D algorithm
  Operations: 

- Cross-compare consecutive pixels within each row of image, forming dert_: queue of derts, each a tuple of derivatives per pixel. 
  dert_ is then segmented into patterns mPs and dPs: contiguous sequences of pixels forming same-sign match or difference. 
  Initial match is inverse deviation of variation: m = ave_|d| - |d|, rather than minimum for directly defined match: 
  albedo or intensity of reflected light doesn't correlate with predictive value of the object that reflects it.

- Match patterns mPs are spans of inputs forming same-sign match. Positive mPs contain high-match pixels, which are likely 
  to match more distant pixels. Thus, positive mPs are evaluated for cross-comp of pixels over incremented range.
- Difference patterns dPs are spans of inputs forming same-sign ds. d sign match is a precondition for d match, so only
  same-sign spans (dPs) are evaluated for cross-comp of constituent differences, which forms higher derivatives.
  (d match = min: rng+ comp value: predictive value of difference is proportional to its magnitude, although inversely so)
  
  Both extended cross-comp forks are recursive: resulting sub-patterns are evaluated for deeper cross-comp, same as top patterns.
  Both forks are currently exclusive per P to avoid redundancy, but they can be made partly or fully overlapping.  

  Initial bi-lateral cross-comp here is 1D slice of 2D 3x3 kernel, while uni-lateral d is equivalent to 2x2 kernel.
  Odd kernels preserve resolution of pixels, while 2x2 kernels preserve resolution of derivatives, in resulting derts.
  The former should be used in rng_comp and the latter in der_comp, which may alternate with intra_P.
  
  postfix '_' denotes array name, vs. same-name elements
  prefix '_' denotes prior of two same-name variables
  prefix 'f' denotes binary flag
  '''
# pattern filters or hyper-parameters: eventually from higher-level feedback, initialized here as constants:

ave = 15     # |difference| between pixels that coincides with average value of mP
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
        if len(mP_) > 4:
            adj_M_ = form_adjacent_M_(mP_)  # compute adjacent Ms for borrowing
            intra_mP_(mP_, adj_M_ , fid=False, rdn=1, rng=3)  # evaluates for sub-recursion per mP

        frame_of_patterns_.append( [mP_] )
        # line of patterns is added to frame of patterns
    return frame_of_patterns_  # frame of patterns will be output to level 2


def form_mP_(P_dert_):  # initialization, accumulation, termination

    P_ = []  # initialization:
    p, d, m = P_dert_[0]
    _sign = m > 0
    if d is None: D = 0
    else: D = d
    L, I, M, dert_, sub_H = 1, p, m, [(p, d, m)], []
    # cluster P_derts by m sign
    for p, d, m in P_dert_[1:]:
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

def form_dP_(P_dert_):  # cluster by d sign, within -mPs: min neg m spans

    P_ = []  # initialization:
    p, d, m = P_dert_[1]  # skip dert_[0]: d is None
    _sign = d > 0
    L, I, D, M, dert_, sub_H = 1, p, 0, m, [(p, d, m)], []
    # cluster P_derts by d sign
    for p, d, m in P_dert_[2:]:
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


def form_adjacent_M_(mP_):  # compute array of adjacent Ms, for contrastive borrow evaluation

    pri_M = mP_[0][4]  # comp_g value is borrowed from adjacent opposite-sign Ms
    M = mP_[1][4]
    adj_M_ = [abs(M)]  # initial next_M, no / 2: projection for first P, abs for bilateral adjustment

    for _, _, _, _, next_M, _, _ in mP_[2:]:
        adj_M_.append((abs(pri_M / 2) + abs(next_M / 2)))  # exclude M
        pri_M = M
        M = next_M
    adj_M_.append(abs(pri_M))  # no / 2: projection for last P

    return adj_M_

''' 
    Recursion in intra_P extends pattern with sub_: hierarchy of sub-patterns, to be adjusted by macro-feedback:
    P:
    sign,  # of m | d 
    Dert = L, I, D, M, 
    dert_, # input for extended cross-comp
    # next fork:
    fdP, # flag: select dP vs. mP forks in form_P_
    fid, # flag: input is derived: magnitude correlates with predictive value: m = min-ave, else m = ave-|d|
    rdn, # redundancy to higher layers, possibly lateral overlap of rng+ & der+, rdn += 1 * typ coef?
    rng, # comp range
    sub_layers: # multiple layers of sub_P_s from d segmentation or extended comp, nested to depth = sub_[n]
                # for layer-parallel access and comp, as in frequency domain representation
                # orders of composition: 1st: dert_, 2nd: sub_P_[ derts], 3rd: sub_layers[ sub_Ps[ derts]] 
'''

def intra_mP_(P_, adj_M_, fid, rdn, rng):  # evaluate for sub-recursion in line mP_, pack results into sub_mP_

    comb_layers = []  # combine into root P sub_layers[1:]
    for (sign, L, I, D, M, dert_, sub_layers), adj_M in zip(P_, adj_M_):  # each sub_layer is nested to depth = sub_layers[n]

        if sign:  # +mP: low-variation span, eval comp at rng=2^n: 2, 4., kernel: 5, 9., rng=1 cross-comp is kernels 2 and 3
            if M - adj_M > ave_M * rdn and L > 4:  # reduced by lending to contrast: all comps form params for hLe comp?

                r_dert_ = range_comp(dert_, fid)  # rng+ comp, skip predictable next dert
                sub_mP_ = form_mP_(r_dert_); Ls = len(sub_mP_)  # cluster by m sign
                sub_layers += [[(Ls, False, fid, rdn, rng, sub_mP_)]]  # 1st layer, Dert=[], fill if Ls > min?
                if len(sub_mP_) > 4:
                    sub_adj_M_ = form_adjacent_M_(sub_mP_)
                    sub_layers += intra_mP_(sub_mP_, sub_adj_M_, fid, rdn + 1 + 1 / Ls, rng*2 + 1)  # feedback
                    comb_layers = [comb_layers + sub_layers for comb_layers, sub_layers in
                                   zip_longest(comb_layers, sub_layers, fillvalue=[])]  # splice sub_layers across sub_Ps

        else:  # -mP: high-variation span, min neg M is contrast value, borrowed from adjacent +mPs:
            if min(-M, adj_M) > ave_D * rdn and L > 3:  # cancelled M+ val, M = min | ~v_SAD

                rel_adj_M = adj_M / -M  # for allocation of -mP' adj_M to each of its internal dPs
                sub_dP_ = form_dP_(dert_); Ls = len(sub_dP_)  # cluster by input d sign match: partial d match
                sub_layers += [[(Ls, True, 1, rdn, rng, sub_dP_)]]  # 1st layer, Dert=[], fill if Ls > min?

                sub_layers += intra_dP_(sub_dP_, rel_adj_M, rdn + 1 + 1 / Ls, rng + 1)  # der_comp eval per nmP
                comb_layers = [comb_layers + sub_layers for comb_layers, sub_layers in
                               zip_longest(comb_layers, sub_layers, fillvalue=[])]
                               # splice sub_layers across sub_Ps for return as root sub_layers[1:]
    return comb_layers


def intra_dP_(dP_, rel_adj_M, rdn, rng):  # evaluate for sub-recursion in line P_, packing results in sub_P_

    comb_layers = []
    for sign, L, I, D, M, dert_, sub_layers in dP_:  # each sub in sub_ is nested to depth = sub_[n]
        if min( abs(D), abs(D) * rel_adj_M) > ave_D * rdn and L > 3:  # abs(D) * rel_adj_M: allocated adj_M
            # if fid: abs(D), else: M + ave*L: complementary m is more precise than inverted diff?

            d_dert_ = deriv_comp(dert_)  # cross-comp of uni_ds
            sub_mP_ = form_mP_(d_dert_); Ls = len(sub_mP_)  # cluster dP derts by md, won't happen
            sub_layers += [[(Ls, 1, 1, rdn, rng, sub_mP_)]]  # 1st layer: Ls, fdP, fid, rdn, rng, sub_P_
            if len(sub_mP_) > 3:
                sub_adj_M_ = form_adjacent_M_(sub_mP_)
                sub_layers += intra_mP_(sub_mP_, sub_adj_M_, 1, rdn + 1 + 1 / Ls, rng+1)
                comb_layers = [comb_layers + sub_layers for comb_layers, sub_layers in
                               zip_longest(comb_layers, sub_layers, fillvalue=[])]
    ''' 
    adj_M is not affected by primary range_comp per mP?
    no comb_m = comb_M / comb_S, if fid: comb_m -= comb_|D| / comb_S: alt rep cost
    same-sign comp: parallel edges, cross-sign comp: M - (~M/2 * rL) -> contrast as 1D difference?
    '''
    return comb_layers


def range_comp(dert_, fid):  # skip odd derts for sparse rng+ comp: 1 skip / 1 add, to maintain 2x overlap

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
        rng_d = _d + _short_rng_d   # difference accumulated in rng
        rng_m = _m + m + _short_rng_m  # bilateral match accumulated in rng
        rdert_.append((_i, rng_d, rng_m))
        _i, _d, _m, _short_rng_d, _short_rng_m =\
            i, d, m, short_rng_d, short_rng_m

    rdert_.append((_i, _d + _short_rng_d, _m * 1.5 + _short_rng_m))  # forward-project m to bilateral m
    return rdert_


def deriv_comp(dert_):  # cross-comp consecutive uni_ds in same-sign dert_: sign match is partial d match
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


def intra_neg_mP_(mP_, adj_M_, rdn, rng):  # deprecated:  evaluate for sub-clustering by d sign

    comb_layers = []
    for (sign, L, I, D, M, dert_, sub_layers), adj_M in zip(mP_, adj_M_):

        if min(-M, adj_M) > ave_D * rdn and L > 3:  # |D| val = cancelled M+ val, not per L: decay is separate?

            sub_dP_ = form_dP_(dert_); Ls = len(sub_dP_)  # cluster by input d sign match: partial d match
            sub_layers += [[(Ls, True, 1, rdn, rng, sub_dP_)]]  # 1st layer, Dert=[], fill if Ls > min?
            sub_layers += intra_dP_(sub_dP_, rdn + 1 + 1 / Ls, rng+1)  # der_comp eval per nmP

            comb_layers = [comb_layers + sub_layers for comb_layers, sub_layers in
                           zip_longest(comb_layers, sub_layers, fillvalue=[])]

    return comb_layers


if __name__ == "__main__":
    # Parse argument (image)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image',
                                 help='path to image file',
                                 default='.//raccoon.jpg')
    arguments = vars(argument_parser.parse_args())
    # Read image
    image = cv2.imread(arguments['image'], 0).astype(int)  # load pix-mapped image
    assert image is not None, "No image in the path"
    image = image.astype(int)

    start_time = time()
    # Main
    frame_of_patterns_ = cross_comp(image)
    end_time = time() - start_time
    print(end_time)

