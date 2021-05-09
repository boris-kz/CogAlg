'''
  line_patterns is a principal version of 1st-level 1D algorithm
  Operations:

- Cross-compare consecutive pixels within each row of image, forming dert_: queue of derts, each a tuple of derivatives per pixel.
  dert_ is then segmented into patterns Pms and Pds: contiguous sequences of pixels forming same-sign match or difference.
  Initial match is inverse deviation of variation: m = ave_|d| - |d|, rather than minimum for directly defined match:
  albedo or intensity of reflected light doesn't correlate with predictive value of the object that reflects it.

- Match patterns Pms are spans of inputs forming same-sign match. Positive Pms contain high-match pixels, which are likely
  to match more distant pixels. Thus, positive Pms are evaluated for cross-comp of pixels over incremented range.
- Difference patterns Pds are spans of inputs forming same-sign ds. d sign match is a precondition for d match, so only
  same-sign spans (Pds) are evaluated for cross-comp of constituent differences, which forms higher derivatives.
  (d match = min: rng+ comp value: predictive value of difference is proportional to its magnitude, although inversely so)

  Both extended cross-comp forks are recursive: resulting sub-patterns are evaluated for deeper cross-comp, same as top patterns.
  Both forks are currently exclusive per P to avoid redundancy, but they can be made partly or fully overlapping.

  Initial bilateral cross-comp here is 1D slice of 2D 3x3 kernel, while unilateral d is equivalent to 2x2 kernel.
  Odd kernels preserve resolution of pixels, while 2x2 kernels preserve resolution of derivatives, in resulting derts.
  The former should be used in rng_comp and the latter in der_comp, which may alternate with intra_P.
'''

import cv2
import argparse
from time import time
from utils import *
from itertools import zip_longest
from class_cluster import ClusterStructure, NoneType

class Cdert(ClusterStructure):
    p = int
    d = int
    m = int

class CP(ClusterStructure):
    sign = NoneType
    L = int
    I = int
    D = int
    M = int
    dert_ = list
    sub_layers = list
    smP = NoneType
    fdert = NoneType

# pattern filters or hyper-parameters: eventually from higher-level feedback, initialized here as constants:

ave = 15  # |difference| between pixels that coincides with average value of Pm
ave_min = 2  # for m defined as min |d|: smaller?
ave_M = 50  # min M for initial incremental-range comparison(t_), higher cost than der_comp?
ave_D = 5  # min |D| for initial incremental-derivation comparison(d_)
ave_nP = 5  # average number of sub_Ps in P, to estimate intra-costs? ave_rdn_inc = 1 + 1 / ave_nP # 1.2
ave_rdm = .5  # average dm / m, to project bi_m = m * 1.5
init_y = 0  # starting row, the whole frame doesn't need to be processed

'''
    Conventions:
    postfix '_' denotes array name, vs. same-name elements
    prefix '_' denotes prior of two same-name variables
    prefix 'f' denotes binary flag
    capitalized variables are normally summed same-letter small-case variables
'''

def cross_comp_spliced(frame_of_pixels_):  # converts frame_of_pixels to frame_of_patterns, each pattern maybe nested
    '''
    process all image rows as a single line, vertically consecutive and preserving horizontal direction
    '''
    Y, X = frame_of_pixels_.shape  # Y: frame height, X: frame width
    pixel__ = []

    for y in range(init_y + 1, Y):  # y is index of new line
        pixel__.append([ frame_of_pixels_[y, :] ])  # splice all rows into pixel__

    # initialization:
    dert_ = []
    __p, _p = pixel__[0:2]  # each prefix '_' denotes prior
    _d = _p - __p  # initial comparison
    _m = ave - abs(_d)
    dert_.append( Cdert(p=__p, d=None, m=(_m + _m / 2)))  # project _m to bilateral m, first dert is for comp_P only?

    for p in pixel__[2:]:  # pixel p is compared to prior pixel _p in a row
        d = p - _p
        m = ave - abs(d)  # initial match is inverse deviation of |difference|
        dert_.append( Cdert(p=_p, d=_d, m=m + _m))  # pack dert: prior p, prior d, bilateral match
        _p, _d, _m = p, d, m
    dert_.append( Cdert(p=_p, d=_d, m=(_m + _m / 2)))  # unilateral d, forward-project last m to bilateral m

    Pm_ = form_Pm_(dert_)  # forms m-sign patterns
    if len(Pm_) > 4:
        adj_M_ = form_adjacent_M_(Pm_)  # compute adjacent Ms to evaluate contrastive borrow potential
        intra_Pm_(Pm_, adj_M_, fid=False, rdn=1, rng=3)  # evaluates for sub-recursion per Pm

    return Pm_  # frame of patterns, an output to line_PPs (level 2 processing)


def cross_comp(frame_of_pixels_):  # converts frame_of_pixels to frame_of_patterns, each pattern maybe nested

    Y, X = frame_of_pixels_.shape  # Y: frame height, X: frame width
    frame_of_patterns_ = []

    # put a brake point here, the code only needs one row to process
    for y in range(init_y + 1, Y):  # y is index of new line pixel_
        # initialization:
        pixel_ = frame_of_pixels_[y, :]
        dert_ = []
        __p, _p = pixel_[0:2]  # each prefix '_' denotes prior
        _d = _p - __p  # initial comparison
        _m = ave - abs(_d)
        dert_.append( Cdert(p=__p, d=None, m=(_m + _m / 2)))  # project _m to bilateral m, first dert is for comp_P only?

        for p in pixel_[2:]:  # pixel p is compared to prior pixel _p in a row
            d = p - _p
            m = ave - abs(d)  # initial match is inverse deviation of |difference|
            dert_.append( Cdert(p=_p, d=_d, m=m + _m))  # pack dert: prior p, prior d, bilateral match
            _p, _d, _m = p, d, m
        dert_.append( Cdert(p=_p, d=_d, m=(_m + _m / 2)))  # unilateral d, forward-project last m to bilateral m

        Pm_ = form_Pm_(dert_)  # forms m-sign patterns
        if len(Pm_) > 4:
            adj_M_ = form_adjacent_M_(Pm_)  # compute adjacent Ms to evaluate contrastive borrow potential
            intra_Pm_(Pm_, adj_M_, fid=False, rdn=1, rng=3)  # evaluates for sub-recursion per Pm

        frame_of_patterns_.append([Pm_])
        # line of patterns is added to frame of patterns

    return frame_of_patterns_  # frame of patterns will be output to level 2


def form_Pm_(P_dert_):  # initialization, accumulation, termination

    P_ = []  # initialization:
    dert = P_dert_[0]

    _sign = dert.m > 0
    D = dert.d or 0
    L, I, M, dert_, sub_H = 1, dert.p, dert.m, [dert], []
    # cluster P_derts by m sign
    for dert in P_dert_[1:]:
        sign = dert.m > 0
        if sign != _sign:  # sign change, terminate P
            P_.append(CP(sign=_sign, L=L, I=I, D=D, M=M, dert_=dert_, sub_layers=sub_H, smP=False, fdert=False))
            L, I, D, M, dert_, sub_H = 0, 0, 0, 0, [], []  # reset params

        L += 1; I += dert.p; D += dert.d; M += dert.m  # accumulate params, bilateral m: for eval per pixel
        dert_ += [dert]
        _sign = sign

    P_.append(CP(sign=_sign, L=L, I=I, D=D, M=M, dert_=dert_, sub_layers=sub_H, smP=False, fdert=False))  # incomplete P
    return P_


def form_Pd_(P_dert_):  # cluster by d sign, within -Pms: min neg m spans

    P_ = []  # initialization:
    dert = P_dert_[1]  # skip dert_[0]: d is None
    _sign = dert.d > 0
    L, I, D, M, dert_, sub_H = 1, dert.p, 0, dert.m, [dert], []
    # cluster P_derts by d sign
    for dert in P_dert_[2:]:
        sign = dert.d > 0
        if sign != _sign:  # sign change, terminate P
            P_.append(CP(sign=_sign, L=L, I=I, D=D, M=M, dert_=dert_, sub_layers=sub_H, smP=False, fdert=False))
            L, I, D, M, dert_, sub_H = 0, 0, 0, 0, [], []  # reset accumulated params

        L += 1; I += dert.p; D += dert.d; M += dert.m  # accumulate params, m for eval per pixel is bilateral
        dert_ += [dert]
        _sign = sign

    P_.append(CP(sign=_sign, L=L, I=I, D=D, M=M, dert_=dert_, sub_layers=sub_H, smP=False, fdert=False))  # incomplete P
    return P_


def form_adjacent_M_(Pm_):  # compute array of adjacent Ms, for contrastive borrow evaluation
    '''
    Value is projected match, while variation has contrast value only: it matters to the extent that it interrupts adjacent match: adj_M.
    In noise, there is a lot of variation. but no adjacent match to cancel, so noise has no predictive value.
    On the other hand, we may have a 2D outline or 1D contrast with low gradient / difference, but it defines a large adjacent uniform span.
    That contrast is salient because it borrows predictive value from adjacent uniform span of inputs.
    '''

    pri_M = Pm_[0].M  # comp_g value is borrowed from adjacent opposite-sign Ms
    M = Pm_[1].M
    adj_M_ = [abs(Pm_[1].M)]  # initial next_M, no / 2: projection for first P, abs for bilateral adjustment

    for Pm in Pm_[2:]:
        next_M = Pm.M
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
    fPd, # flag: select Pd vs. Pm forks in form_P_
    fid, # flag: input is derived: magnitude correlates with predictive value: m = min-ave, else m = ave-|d|
    rdn, # redundancy to higher layers, possibly lateral overlap of rng+ & der+, rdn += 1 * typ coef?
    rng, # comp range
    sub_layers: # multiple layers of sub_P_s from d segmentation or extended comp, nested to depth = sub_[n]
                # for layer-parallel access and comp, as in frequency domain representation
                # orders of composition: 1st: dert_, 2nd: sub_P_[ derts], 3rd: sub_layers[ sub_Ps[ derts]] 
'''

def intra_Pm_(P_, adj_M_, fid, rdn, rng):  # evaluate for sub-recursion in line Pm_, pack results into sub_Pm_

    comb_layers = []  # combine into root P sub_layers[1:]
    for P, adj_M in zip(P_, adj_M_):  # each sub_layer is nested to depth = sub_layers[n]

        if P.sign:  # +Pm: low-variation span, eval comp at rng=2^n: 2, 4., kernel: 5, 9., rng=1 cross-comp is kernels 2 and 3
            if P.M - adj_M > ave_M * rdn and P.L > 4:  # reduced by lending to contrast: all comps form params for hLe comp?

                r_dert_ = range_comp(P.dert_, fid)  # rng+ comp, skip predictable next dert
                sub_Pm_ = form_Pm_(r_dert_)  # cluster by m sign
                Ls = len(sub_Pm_)
                P.sub_layers += [[(Ls, False, fid, rdn, rng, sub_Pm_)]]  # 1st layer, Dert=[], fill if Ls > min?
                if len(sub_Pm_) > 4:
                    sub_adj_M_ = form_adjacent_M_(sub_Pm_)
                    P.sub_layers += intra_Pm_(sub_Pm_, sub_adj_M_, fid, rdn + 1 + 1 / Ls, rng * 2 + 1)  # feedback
                    # splice sub_layers across sub_Ps:
                    comb_layers = [comb_layers + sub_layers for comb_layers, sub_layers in
                                   zip_longest(comb_layers, P.sub_layers, fillvalue=[])]

        else:  # -Pm: high-variation span, min neg M is contrast value, borrowed from adjacent +Pms:
            if min(-P.M, adj_M) > ave_D * rdn and P.L > 3:  # cancelled M+ val, M = min | ~v_SAD

                rel_adj_M = adj_M / -P.M  # for allocation of -Pm' adj_M to each of its internal Pds
                sub_Pd_ = form_Pd_(P.dert_)  # cluster by input d sign match: partial d match
                Ls = len(sub_Pd_)
                P.sub_layers += [[(Ls, True, 1, rdn, rng, sub_Pd_)]]  # 1st layer, Dert=[], fill if Ls > min?

                P.sub_layers += intra_Pd_(sub_Pd_, rel_adj_M, rdn + 1 + 1 / Ls, rng + 1)  # der_comp eval per nPm
                # splice sub_layers across sub_Ps, for return as root sub_layers[1:]:
                comb_layers = [comb_layers + sub_layers for comb_layers, sub_layers in
                               zip_longest(comb_layers, P.sub_layers, fillvalue=[])]

    return comb_layers


def intra_Pd_(Pd_, rel_adj_M, rdn, rng):  # evaluate for sub-recursion in line P_, packing results in sub_P_

    comb_layers = []
    for P in Pd_:  # each sub in sub_ is nested to depth = sub_[n]
        if min(abs(P.D), abs(P.D) * rel_adj_M) > ave_D * rdn and P.L > 3:  # abs(D) * rel_adj_M: allocated adj_M
            # if fid: abs(D), else: M + ave*L: complementary m is more precise than inverted diff?

            d_dert_ = deriv_comp(P.dert_)  # cross-comp of uni_ds
            sub_Pm_ = form_Pm_(d_dert_)  # cluster Pd derts by md, won't happen
            Ls = len(sub_Pm_)
            P.sub_layers += [[(Ls, 1, 1, rdn, rng, sub_Pm_)]]  # 1st layer: Ls, fPd, fid, rdn, rng, sub_P_
            if len(sub_Pm_) > 3:
                sub_adj_M_ = form_adjacent_M_(sub_Pm_)
                P.sub_layers += intra_Pm_(sub_Pm_, sub_adj_M_, 1, rdn + 1 + 1 / Ls, rng + 1)
                # splice sub_layers across sub_Ps:
                comb_layers = [comb_layers + sub_layers for comb_layers, sub_layers in
                               zip_longest(comb_layers, P.sub_layers, fillvalue=[])]
    ''' 
    adj_M is not affected by primary range_comp per Pm?
    no comb_m = comb_M / comb_S, if fid: comb_m -= comb_|D| / comb_S: alt rep cost
    same-sign comp: parallel edges, cross-sign comp: M - (~M/2 * rL) -> contrast as 1D difference?
    '''
    return comb_layers


def range_comp(dert_, fid):  # skip odd derts for sparse rng+ comp: 1 skip / 1 add, to maintain 2x overlap

    rdert_ = []  # prefix '_' denotes the prior of same-name variables, initialization:
    __dert = dert_[0]  # prior-prior dert
    __i = __dert.p
    _dert = dert_[2]  # initialize _dert with sparse p_, skipping odd ps
    _i = _dert.p
    _short_rng_d = _dert.d
    _short_rng_m = _dert.m

    _d = _i - __i
    if fid:  # flag: input is d, from deriv_comp
        _m = min(__i, _i) - ave_min
    else:
        _m = ave - abs(_dert.d)  # no ave * rng: m and d value is cumulative
    _rng_m = (_m + _m / 2) + __dert.m  # back-project missing m as _m / 2: induction decays with distance
    rdert_.append(Cdert(p=__i, d=None, m=_rng_m))  # no _rng_d = _d + __short_rng_d

    for n in range(4, len(dert_), 2):  # backward comp

        dert = dert_[n]
        i = dert.p
        short_rng_d = dert.d
        short_rng_m = dert.m
        d = i - _i
        if fid:
            m = min(i, _i) - ave_min  # match = min: magnitude of derived vars correlates with stability
        else:
            m = ave - abs(d)  # inverse match: intensity doesn't correlate with stability
        rng_d = _d + _short_rng_d  # difference accumulated in rng
        rng_m = _m + m + _short_rng_m  # bilateral match accumulated in rng
        rdert_.append(Cdert(p=_i, d=rng_d, m=rng_m))
        _i, _d, _m, _short_rng_d, _short_rng_m = \
            i, d, m, short_rng_d, short_rng_m

    rdert_.append(Cdert(p=_i, d=_d + _short_rng_d, m=(_m + _m / 2) + _short_rng_m))  # forward-project _m to bilateral m
    return rdert_


def deriv_comp(dert_):  # cross-comp consecutive uni_ds in same-sign dert_: sign match is partial d match
    # dd and md may match across d sign, but likely in high-match area, spliced by spec in comp_P?

    ddert_ = []  # initialization:
    __i = dert_[1].d  # each prefix '_' denotes prior
    _i = dert_[2].d

    __i = abs(__i);  _i = abs(_i)
    _d = _i - __i  # initial comp
    _m = min(__i, _i) - ave_min
    ddert_.append(Cdert(p=_i, d=None, m=(_m + _m / 2)))  # no __d, back-project __m = _m * .5

    for dert in dert_[3:]:
        i = abs(dert.d)  # unilateral d, same sign in Pd
        d = i - _i  # d is dd
        m = min(i, _i) - ave_min  # md = min: magnitude of derived vars corresponds to predictive value
        ddert_.append(Cdert(p=_i, d=_d, m=_m + m))  # unilateral _d and bilateral m per _i
        _i, _d, _m = i, d, m

    ddert_.append(Cdert(p=_i, d=_d, m=(_m + _m / 2)))  # forward-project bilateral m
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
    assert image is not None, "No image in the path"
    image = image.astype(int)

    start_time = time()
    fline_PPs = 1
    # Main
    frame_of_patterns_ = cross_comp(image)

    from line_PPs_draft import *

    frame_dert_P_ = []
    frame_PPm_ = []
    if fline_PPs:  # debug line_PPs
        for y, P_ in enumerate(frame_of_patterns_):
            dert_P_ = comp_P_(P_[0])
            frame_dert_P_.append(dert_P_)
            if len(dert_P_) > 1:
                frame_PPm_.append(form_PPm(dert_P_))

            # check if there is false sign
            if dert_P_:
                for dert_P in dert_P_:
                    if not dert_P.smP:  # check false sign
                        print('False sign in line' + str(y))

    end_time = time() - start_time
    print(end_time)
