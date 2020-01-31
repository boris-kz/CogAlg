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
ave_D = 10   # min |D| for initial incremental-derivation comparison(d_)
ave_nP = 5   # average number of sub_Ps in P, to estimate intra-costs? ave_rdn_inc = 1 + 1 / ave_nP # 1.2
ave_rdd =.5  # average dd / d, for projection: uni_d *= 1.5?
ini_y = 500

def cross_comp(frame_of_pixels_):  # converts frame_of_pixels to frame_of_patterns, each pattern maybe nested

    Y, X = image.shape  # Y: frame height, X: frame width
    frame_of_patterns_ = []

    for y in range(ini_y + 1, Y):  # y is index of new line pixel_, initialization:
        pixel_ = frame_of_pixels_[y, :]
        dert_ = []
        __p, _p = pixel_[0:2]  # each prefix '_' denotes prior
        _d = _p - __p  # initial comp
        _m = ave - abs(_d)
        dert_.append((__p, _d * 1.5, _m * 1.5, None))  # back-project _d and _m to bilateral values

        for p in pixel_[2:]:  # pixel p is compared to prior pixel _p in a row
            d = p - _p
            m = ave - abs(d)  # initial match is inverse deviation of |difference|
            dert_.append((_p, d + _d, m + _m, _d))  # pack prior p, bilateral difference and match, prior d
            _p, _d, _m = p, d, m
        dert_.append((_p, _d * 1.5, _m * 1.5, _d))  # forward-project last d and m to bilateral values

        mP_ = form_P_(dert_, fdP=False)  # form m-sign patterns
        intra_P(mP_, fdP=False, fid=False, rdn=1, rng=2)  # evaluate sub-recursion per mP
        dP_ = form_P_(dert_, fdP=True)  # form d-sign patterns
        intra_P(dP_, fdP=True, fid=True, rdn=1, rng=1)  # evaluate sub-recursion per dP

        frame_of_patterns_ += [(mP_, dP_)]  # line of patterns is added to frame of patterns
    return frame_of_patterns_  # frame of patterns will be output to level 2


def form_P_(P_dert_, fdP):  # pattern initialization, accumulation, termination, parallel der+ and rng+?

    P_ = []  # initialization:
    p, d, m, uni_d = P_dert_[0]  # uni_d for der_comp
    ini_dert = 1
    if fdP:  # flag dP, selects between form_dP_ and form_mP_ forks
        if uni_d is None:
            p, d, m, uni_d = P_dert_[1]  # skip dert_[0]
            _sign = uni_d > 0
            ini_dert = 2
        else:
            _sign = uni_d > 0
        D = uni_d
    else:
        _sign = m > 0
        D = d
    dLL, rLL, L, I, M, dert_, dsub_, rsub_ = [], [], 1, p, m, [(p, d, m, uni_d)], [], []
    # LL: depth of sub-hierarchy, each sub-pattern in sub_ is nested to depth = sub_[n]

    for p, d, m, uni_d in P_dert_[ini_dert:]:
        if fdP: sign = uni_d > 0
        else:   sign = m > 0
        if sign != _sign:  # sign change: terminate P
            P_.append((_sign, dLL, rLL, L, I, D, M, dert_, dsub_, rsub_))
            dLL, rLL, L, I, D, M, dert_, dsub_, rsub_ = [], [], 0, 0, 0, 0, [], [], []
            # reset accumulated params
        L += 1; I += p; D += d; M += m  # accumulate params with bilateral values
        # uni_D = D - dert_[1][-1] - dert_[-1][-1]?
        dert_ += [(p, d, m, uni_d)]
        _sign = sign

    P_.append((_sign, dLL, rLL, L, I, D, M, dert_, dsub_, rsub_))  # incomplete P
    # also sum Dert per P_?
    return P_

''' Recursion in intra_P extends pattern with sub_: hierarchy of sub-patterns, to be adjusted by macro-feedback:
    P_:
    fdP,  # flag: select dP or mP forks in form_P_ and intra_P
    fid,  # flag: input is derived: magnitude correlates with predictive value: m = min-ave, else m = ave-|d|
    rdn,  # redundancy to higher layers, possibly lateral overlap of rng+ & der+, rdn += 1 * typ coef?
    rng,  # range extension count  
    P:
    sign,  # of core param: m | d 
    Dert = L, I, D, M, 
    dert_, # input for extended comp
    dsub_, # multiple layers of (hyper, Dert, sub_P_) from segment or extended comp, nested to depth = sub_[n], 
    rsub_, # for layer-parallel access and comp, similar to frequency domain representation
           # sub_P_: flat or nested for mapping to higher-layer sub_P_ element?
    root   # reference to higher P for layer-sequential feedback 

    orders of composition: 1st: dert_, 2nd: lateral_sub_( derts), 3rd: sub_( lateral_sub_( derts))? 
    line-wide layer-sequential recursion and feedback, for clarity and slice-mapped SIMD processing? 
'''

def intra_P(P_, fdP, fid, rdn, rng):  # evaluate for sub-recursion in line P_, filling its sub_P_ with the results

    deep_sub_ = []  # intra_P recursion extends rsub_ and dsub_ hierarchies by sub_P_ layer
    ext_dert_ = []  # new dert_ from extended- range or derivation comp

    for sign, dLL, rLL, L, I, D, M, dert_, dsub_, rsub_ in P_:  # each sub in sub_ is nested to depth = sub_[n]

        if fdP:  # P = dP: d sign match is partial d match, precondition for der+, or in -mPs to avoid overlap
            if (abs(D) > ave_D * rdn) and L > 3: # remove 1st and last uni_d projection?
                ext_dert_ = der_comp(dert_)  # cross-comp uni_ds
            else:
                ext_dert_ = []
        elif sign:  # P = positive mP: low-variation span, evaluate for pixel cross-comp over passed rng + 1:
            if M > ave_M * rdn and L > 4:  # comp range = rng**2: 1, 2, 4, kernel = range * 2 + 1: 3, 5, 9:
                ext_dert_ = rng_comp(dert_, fid)
            else:
                ext_dert_ = []  # also merge not-selected P into non_P?
        if ext_dert_:

            sub_dP_ = form_P_(ext_dert_, True); lL = len(sub_dP_)
            dsub_ += [[(lL, True, True, rdn, rng, sub_dP_)]]  # 1st layer: lL, fdP, fid, rdn, rng, sub_P_
            dsub_ += intra_P(sub_dP_, True, True, rdn + 1 + 1 / lL, rng)  # deep layers feedback
            dLL[:] = [len(dsub_)]   # deeper P rdn + 1: rdn to higher derts, + 1 / lL: rdn to higher sub_

            sub_mP_ = form_P_(ext_dert_, False); lL = len(sub_mP_)
            rsub_ += [[(lL, False, fid, rdn, rng, sub_mP_)]]  # 1st layer, Dert=[], fill if lL > min?
            rsub_ += intra_P(sub_mP_, False, fid, rdn + 1 + 1 / lL, rng + 1)  # deep layers feedback
            rLL[:] = [len(rsub_)]

            deep_sub_ = [deep_sub + dsub + rsub for deep_sub, dsub, rsub in zip_longest(deep_sub_, dsub_, rsub_, fillvalue=[])]
            # deep_rsub_ and deep_dsub_ are spliced into deep_sub_ hierarchy;   fill Dert per layer if n_sub_P > min?

    return deep_sub_


def rng_comp(dert_, fid):  # skip odd derts for sparse rng+ comp: 1 skip / 1 add, to maintain 2x overlap

    rdert_ = []   # prefix '_' denotes the prior of same-name variables, initialization:
    (__i, __short_bi_d, __short_bi_m, _), _, (_i, _short_bi_d, _short_bi_m, _) = dert_[0:3]
    _d = _i - __i
    if fid: _m = min(__i, _i) - ave_min;
    else:   _m = ave - abs(_d)  # no ave * rng: actual m and d value is cumulative?
    _bi_d = _d * 1.5 + __short_bi_d
    _bi_m = _m * 1.5 + __short_bi_m  # back-project _m and d
    rdert_.append((__i, _bi_d, _bi_m, None))

    for n in range(4, len(dert_), 2):  # backward comp, ave | cumulative ders and filters?
        i, short_bi_d, short_bi_m = dert_[n][:3]  # shorter-rng dert
        d = i - _i
        if fid: m = min(i, _i) - ave_min  # match = min: magnitude of derived vars correlates with stability
        else:   m = ave - abs(d)  # inverse match: intensity doesn't correlate with stability
        bi_d = _d + d + _short_bi_d  # bilateral difference, accum in rng
        bi_m = _m + m + _short_bi_m  # bilateral match, accum in rng
        rdert_.append((_i, bi_d, bi_m, _d))
        _i, _d, _m, _short_bi_d, _short_bi_m = i, d, m, short_bi_d, short_bi_m

    rdert_.append((_i, _d * 1.5 + _short_bi_d, _m * 1.5 + _short_bi_m, _d))
    # forward-project unilateral to bilateral d and m values
    return rdert_


def der_comp(dert_):  # cross-comp consecutive uni_ds in same-sign dert_: sign match is partial d match
    # dd and md may match across d sign, but likely in high-match area, spliced by spec in comp_P?

    ddert_ = []   # initialization:
    (_, _, _, __i), (_, _, _, _i) = dert_[1:3]  # each prefix '_' denotes prior
    __i = abs(__i); _i = abs(_i)
    _d = _i - __i  # initial comp
    _m = min(__i, _i) - ave_min
    ddert_.append((__i, _d * 1.5, _m * 1.5, None))  # __d and __m are back-projected as = _d or _m

    for dert in dert_[3:]:
        i = abs(dert[3])  # unilateral d in same-d-sign seg, no sign comp
        d = i - _i   # d is dd
        m = min(i, _i) - ave_min  # md = min: magnitude of derived vars corresponds to predictive value
        bi_d = _d + d  # bilateral d-difference per _i
        bi_m = _m + m  # bilateral d-match per _i
        ddert_.append((_i, bi_d, bi_m, _d))
        _i, _d, _m = i, d, m

    ddert_.append((_i, _d * 1.5, _m * 1.5, _d))  # forward-project unilateral to bilateral d and m values
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
        cross_comp (sub_)?  # same-recursion (derivation) order e_
            cross_comp (dert_)
            
Then extend this 2nd level alg to a recursive meta-level algorithm

for partial overlap:

def form_mP_(dert_):  # initialization, accumulation, termination

    _sign, LL, L, I, D, M, dert_, sub_ = P  # each sub in sub_ is nested to depth = sub_[n]
    p, d, m, uni_d = dert
    sign = m > 0
    if sign != _sign:
        # sign change: terminate P
        P_.append((_sign, LL, L, I, D, M, dert_, sub_))  # LL(sub_ depth), L (len dert_)  for visibility only
        LL, L, I, D, M, dert_, sub_ = [], 0, 0, 0, 0, [], []  # reset accumulated params
    # accumulate params with bilateral values:
    L += 1; I += p; D += d; M += m
    dert_ += [(p, d, m, uni_d)]  # uni_d for der_comp and segment
    P = sign, LL, L, I, D, M, dert_, sub_  # sub_ is accumulated in intra_P

    return P, P_

def form_dP_(dert_):  # P segmentation by same d sign: initialization, accumulation, termination

    sub_ = []  # becomes lateral_sub_
    _p, _d, _m, _uni_d = dert_[0]  # prefix '_' denotes prior
    try:
        _sign = _uni_d > 0; ini = 1
    except:
        _p, _d, _m, _uni_d = dert_[1]  # skip dert_[0] if uni_d is None: 1st dert in comp sequence
        _sign = _uni_d > 0; ini = 2
        
    if _uni_d > min_d: md_sign = 1  # > variable cost of der+
    else: md_sign = 0  # no der+ eval

    LL, L, I, D, M, seg_dert_ = [], 1, _p, _uni_d, _m, [(_p, _d, _m, _uni_d)]  # initialize dP

    for p, d, m, uni_d in dert_[ini:]:
        sign = uni_d > 0
        if _sign != sign:
            sub_.append((_sign, LL, L, I, D, M, seg_dert_, []))  # terminate seg_P, same as P
            LL, L, I, D, M, seg_dert_, sub_ = [], 0, 0, 0, 0, [], []  # reset accumulated seg_P params
        _sign = sign
        L += 1; I += p; D += uni_d; M += m  # D += uni_d to eval for comp uni_d
        seg_dert_.append((p, d, m, uni_d))

    sub_.append((_sign, LL, L, I, D, M, seg_dert_, []))  # pack last segment, nothing to accumulate
    # also Dert in sub_ [], fill if min lLL?
    return sub_  # becomes lateral_sub_
    
def splice(listOfLists, *otherLoLs, fillvalue=[]):
    "Splice nested lists laterally."
    return [[*flatten(li)] for li in zip_longest(listOfLists, *otherLoLs, fillvalue=fillvalue)]

    flatten = lambda l: [item for sublist in l for item in sublist]
    r_deep_sub_, d_deep_sub_ = intra_P(sub_dP_, True, True, sub_rdn + 1.2, rng)  # deep layers feedback
    d_sub_ += [flatten([flatten(r_deep_sub_), flatten(d_deep_sub_)])]

    r_deep_sub_, d_deep_sub_ = intra_P(sub_mP_, False, fid, sub_rdn + 1.2, rng + 1)
    r_sub_ += [flatten([flatten(r_deep_sub_), flatten(d_deep_sub_)])]
    
    if rdn > 1: rdn += 1 / lL - 0.2  # adjust distributed part of estimated rdn
    [len(rsub) for rsub in rsub_]
'''
