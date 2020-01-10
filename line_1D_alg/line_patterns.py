import cv2
import argparse
from time import time
from collections import deque

# pattern filters or hyper-parameters: eventually from higher-level feedback, initialized here as constants:

ave = 20  # |difference| between pixels that coincides with average value of mP - redundancy to overlapping dPs
ave_m = 10  # for m defined as min, same?
ave_M = 255  # min M for initial incremental-range comparison(t_), higher cost than der_comp?
ave_D = 127  # min |D| for initial incremental-derivation comparison(d_)
ave_L = 8  # min L for sub_cluster(d)
ave_nP = 4  # average number of sub_Ps in P, to estimate intra-costs?
ini_y = 664
# min_rng = 1  # >1 if fuzzy pixel comparison range, for sensor-specific noise only

''' 
  line_patterns is a principal version of 1st-level 1D algorithm, contains following operations: 
- Cross-compare consecutive pixels within each row of image, forming dert_ queue of derts: tuples of derivatives per pixel. 
  dert_ is then segmented by match deviation, forming mPs, each a contiguous sequence of pixels that form same-sign m: +mP or -mP. 
  Initial match is inverse deviation of variation: m = ave_|d| - |d|, not min: brightness doesn't correlate with predictive value.
- Positive mPs: spans of pixels forming positive match, are evaluated for cross-comp of dert input param over incremented range 
  (positive match means that pixels have high predictive value, thus likely to match more distant pixels).
- Negative mPs: high-variation spans, are evaluated for cross-comp of difference, which forms higher derivatives.
  Both types of extended cross-comp are recursive: resulting sub-patterns are evaluated for deeper cross-comp, 
  same as top patterns.
- Median mPs: comp d sign (direction) -> seg_d, D is too weak for immediate comp_d, but may accumulate within same-sign segs.
  these forks may overlap, but then rdn computation is far more expensive. 
  If value of incremental range or derivation cross-comp over full pattern is low, but the pattern is long enough, then its dert_ 
  is segmented by (m-ave) for +mPs and d-sign match for -mPs. Value density of resulting positive segments (seg_Ps) is higher than 
  in full pattern: +mm_seg_ave_M = ave_M / 2, and same-d-sign seg_D is not sign-cancelled (+mm: flat seg, ds: directed seg?)
  Thus, positive m-segments are evaluated for local rng_comp, and positive d-segments are evaluated for local der_comp.     
  comp_d eval by D because match of opposite-sign ds is -min: low comp value, but adjacent dd signs may still match?  
  (match = min: rng+ comp value, because predictive value of difference is proportional to its magnitude, although inversely so)
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

        pixel_ = frame_of_pixels_[y, :]  # y is index of new line pixel_
        P_ = []  # row of patterns, initialized at each line
        _p = pixel_[0]  # prefix '_' denotes prior
        _d, _m, _uni_d = 0, 0, 0  # no backward d, m at x = 0
        # initialize P( sign, I, D, M, dert_, seg_, sub_, layer_) with dert_[0]:
        P = 0, _p, _d, _m, [(_p, _d, _m, _uni_d)], [], []  # _uni_d for consistency, skipped in comp_p

        for p in pixel_[1:]:  # pixel p is compared to prior pixel _p in a row
            d = p - _p
            m = ave - abs(d)  # initial match is inverse deviation of |difference|
            bi_d = d + _d  # bilateral difference
            bi_m = m + _m  # bilateral match
            dert = _p, bi_d, bi_m, d
            # accumulate or terminate mP: span of pixels forming same-sign m:
            P, P_ = form_pattern(P, P_, dert)
            _p, _d, _m = p, d, m  # uni_d is not used in comp
        # terminate last P in row:
        dert = _p, _d * 2, _m * 2, _d  # last d and m are forward-projected to bilateral values
        P, P_ = form_pattern(P, P_, dert)
        P_ += [P]
        # evaluate sub-recursion per P:
        P_ = intra_P(P_, fid=False, rdn=1, rng=1, sD=0)  # recursive
        frame_of_patterns_ += [P_]  # line of patterns is added to frame of patterns

    return frame_of_patterns_  # frame of patterns will be output to level 2


def form_pattern(P, P_, dert):  # initialization, accumulation, termination, recursion

    _sign, I, D, M, dert_, sub_, layer_ = P  # each layer is nested to depth = len(layer_)
    p, d, m, uni_d = dert
    if m > 0:
        if m > ave_m:
            sign = 0  # low variation: comp rng+ eval P;  Ternary sign
        else:
            sign = 1  # medium variation: segment P.dert by d sign
    else:
        sign = 2  # high variation: comp der+ eval P

    if sign != _sign:  # sign change: terminate mP
        P_.append(P)
        I, D, M, dert_ = 0, 0, 0, []  # reset accumulated params
    # accumulate params with bilateral values:
    I += p;
    D += d;
    M += m
    dert_ += [(p, d, m, uni_d)]  # uni_d for intra_comp(d) and segment(d)
    P = sign, I, D, M, dert_, sub_, layer_  # sub_ and seg_ are accumulated in intra_comp and sub_segment

    return P, P_


''' Recursion extends pattern structure to 1d hierarchy and then 2d hierarchy, to be adjusted by macro-feedback:
    P_:
    fid,   # flag: input is derived: magnitude correlates with predictive value: m = min-ave, else m = ave-|d|
    rdn,   # redundancy to higher layers, possibly lateral overlap of rng+, seg_d, der+, rdn += 1 * typ coef?
    rng,   # comp range, + frng | fder?  
    P:
    sign,  # ternary: 0 -> rng+, 1 -> segment_d, 2 -> der+ 
    Dert = I, D, M,  # L = len(dert_) * rng
    dert_, # input for sub_segment or extend_comp
           # conditional 1d array of next layer:
    sub_,  # seg_P_ from sub_segment or sub_P_ from extend_comp
           # conditional 2d array of deeper layers, each layer maps to higher layer for feedback:
    layer_,  # each layer has Dert and array of seg_|sub_s, each with crit and fid, nested to different depth
             # for layer-parallel access and comp, similar to frequency domain representation
    root_P   # reference for layer-sequential feedback 
    orders of composition: 1st: dert_, 2nd: seg_|sub_(derts), 3rd: P layer_(sub_P_(derts))? 
    line-wide layer-sequential recursion and feedback, for clarity and slice-mapped SIMD processing? '''


def intra_P(P_, fid, rdn, rng,
            sD):  # evaluate for sub-recursion in line P_, filling sub_ | seg_, len(sub_)- adjusted rdn?
    
    for n, (sign, I, D, M, dert_, sub_, layer_) in enumerate(P_):  # pack fid, rdn, rng in P_?



        if sign == 0:  # low-variation P: eval for extend_comp(rng+):
            if M > ave_M * rdn and len(dert_) > 2:  # ave * rdn is fixed cost of comp_rng forming len(sub_P_) = ave_nP
                ssub_ = extend_comp(dert_, True, fid, rdn + 1, rng)  # frng=1
                P_[n][5][2][:] = intra_P(ssub_[3], fid, rdn + 1, rng, 0)  # sub_ = frng=0, fid=1, sub_

        elif sign == 1:  # mid-variation P: segment dert_ by d sign
            if len(dert_) > ave_L * rdn:  # fixed costs of new P_ are translated to ave L
                ssub_, sD = segment_d(dert_, rdn + 1, rng)  # P.sub_ = fid=1, rdn, rng, sub_
                P_[n][5][2][:] = intra_P(ssub_[3], True, rdn + 1, rng, sD)  # sign = 2: will trigger last fork:

        else:  # high-variation P: eval for extend_comp(der+):
            if sD:
                vD = sD  # intra_P is called after segment_d, sD was accumulated in same-d-sign segment
            else:
                vD = -M
            if vD > ave_D * rdn and len(dert_) > 2:  # >fixed costs of full-P comp_der+, obviates seg_dP_
                fid = True
                ssub_ = extend_comp(dert_, False, fid, rdn + 1, rng=1)
                P_[n][5][2][:] = intra_P(ssub_[3], fid, rdn + 1, rng, 0)  # P.sub_ = frng=0, fid=1, sub_

        # each: else merge short sub_Ps between Ps, if allowed by max recursion depth?
        # no recursion eval per new layer, P_ only
    return P_



def segment_d(P_dert_, rdn, rng):  # mP segmentation by d sign: initialization, accumulation, termination

    P_sub_ = []  # replaces P.sub_
    sD = 0
    _p, _d, _m, _uni_d = P_dert_[0]  # prefix '_' denotes prior
    _sign = _uni_d > 0
    I = _p;
    D = _d;
    M = _m;
    dert_ = [(_p, _d, _m, _uni_d)];
    sub_ = [];
    layer_ = []  # initialize seg_P, same as P

    for p, d, m, uni_d in P_dert_[1:]:
        sign = uni_d > 0
        if _sign != sign:
            sD += D
            P_sub_.append((_sign, I, D, M, dert_, sub_, layer_))  # terminate seg_P, same as P
            I, D, M, dert_, sub_ = 0, 0, 0, [], []  # reset accumulated seg_P params
        _sign = sign
        I += p;
        D += d;
        M += m  # accumulate seg_P params, not uni_d
        dert_.append((p, d, m, uni_d))
    sD += D
    P_sub_.append((_sign, I, D, M, dert_, sub_, layer_))  # pack last segment, nothing to accumulate
    rdn *= ave_nP / len(P_sub_)  # cost of higher layers is distributed over all sub_Ps

    return (True, rdn, rng, P_sub_), sD  # replace P.sub_


def extend_comp(dert_, frng, fid, rdn, rng):  # extended cross_comp within P.dert_, comp(rng) if frng, else comp(d)

    
    sub_P_ = []  # return to replace P.sub_; prefix '_' denotes prior of two same-name variables:

    if frng:  # flag comp(rng), sparse bilateral comp, 1 pruned dert / 1 extended dert to maintain 2x overlap:
        m = dert_[0][2]
        if m > 0:
            if m > ave_m:
                sign = 0  # Ternary sign
            else:
                sign = 1
        else:
            sign = 2
        sub_P = sign, 0, 0, 0, [], [], []  # sign, I, D, M, sub_, dert_, layer_: same as master P
        rng *= 2  # 1, 2, 4, 8, L+=rng, kernel = rng * 2 + 1: 3, 5, 9, 17
        _i, _d, _m = dert_[0][:3]  # initial _d = 0 + rng-1 bi_d, initial _m = 0 + rng-1 bi_m

        for n in range(2, len(dert_), 2):  # backward comp, skip 1 dert to maintain overlap rate, that defines ave
            i, short_bi_d, short_bi_m = dert_[n][:3]  # shorter-rng dert
            d = i - _i
            if fid:  # match = min: magnitude of derived vars correlates with stability
                m = min(i, _i) - ave_m + short_bi_m  # m accum / i number of comps
            else:  # inverse match: intensity doesn't correlate with stability
                m = ave - abs(d) + short_bi_m
                # no ave * rdn, bi_m ave_m * rng-2: comp cost is separate from mP definition for comp_P
            d += short_bi_d  # _d and _m combine bi_d | bi_m at rng-1
            if n == 2:
                d *= 2;
                m *= 2  # back-projection for unilateral ders
            bi_d = _d + d  # bilateral difference, accum in rng
            bi_m = _m + m  # bilateral match, accum in rng
            dert = _i, bi_d, bi_m, d
            _i, _d, _m = i, d, m
            # P accumulation or termination:
            
            sub_P, sub_P_ = form_pattern(sub_P, sub_P_, dert)
            

    else:  # frng=0: bilateral comp between consecutive uni_ds in dert_, dd and md may match across d sign
        fid = True  # flag i is derived
        sub_P = 0, 0, 0, 0, [], [], []  # sign, I, D, M, sub_, dert_, layer_: same as master P
        _i, _d, _m = dert_[0][3], 0, 0  # initial _d and _m = 0
        ini = 1
        for dert in dert_[1:]:
            i = dert[3]  # unilateral d
            d = i - _i  # d is dd
            m = min(i, _i) - ave_m  # md = min: magnitude of derived vars corresponds to predictive value
            if ini == 1:
                d *= 2;
                m *= 2;
                ini = 0  # back-project unilateral ders
            bi_d = _d + d  # bilateral d-difference per _i
            bi_m = _m + m  # bilateral d-match per _i
            dert = _i, bi_d, bi_m, d
            _i, _d, _m = i, d, m
            # P accumulation or termination:
            sub_P, sub_P_ = form_pattern(sub_P, sub_P_, dert)

    # terminate last sub_P in dert_:
    dert = _i, _d * 2, _m * 2, _d  # forward-project unilateral to bilateral d and m values
    sub_P, sub_P_ = form_pattern(sub_P, sub_P_, dert)
    sub_P_ += [sub_P]

    return fid, rdn * ave_nP / len(sub_P_), rng, sub_P_  # replaces P.sub_


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
Deprecated:
def intra_P(P_, fid, rdn, rng):  # evaluate for sub-recursion in line P_, filling sub_ | seg_, len(sub_)- adjusted rdn?
    for n, (_, _, _, _, _, _, layer_) in enumerate(P_):  # pack fid, rdn, rng in P_?
        _, dert__ = layer_[-1]  # Dert and derts of newly formed layer
        for _sign, I, D, M, dert_ in dert__:  # dert_s of all sub_Ps in a layer
            if _sign == 0:  # low-variation P: segment by mm, segment(ds) eval/ -mm seg, extend_comp(rng) eval/ +mm seg:

                if M > ave_M * rdn and len(dert_) > 2:  # ave * rdn is fixed cost of comp_rng forming len(sub_P_) = ave_nP
                    ssub_, rdn = extend_comp(dert_, True, fid, rdn + 1, rng)  # frng=1
                    seg_[n][6][2][:], rdn = intra_P(ssub_, fid, rdn + 1, rng)  # eval per seg_, sub_ = frng=0, fid=1, sub_
                if len(dert_) > ave_Lm * rdn:  # fixed costs of new P_ are translated to ave L
                    sub_, rdn = sub_segment(dert_, True, fid, rdn+1, rng)
                    P_[n][5][2][:], rdn = intra_seg(sub_, True, fid, rdn+1, rng)  # eval per seg_P, P.seg_ = fmm=1, fid, seg_
            else:
                if -M > ave_D * rdn and len(dert_) > rng * 2:  # -M > fixed costs of full-P comp_d -> d_mP_, obviates seg_dP_
                    fid = True
                    sub_, rdn = extend_comp(dert_, False, fid, rdn+1, rng=1)
                    P_[n][5][2][:], rdn = intra_P(sub_, fid, rdn+1, rng)  # eval per sub_P, P.sub_ = frng=0, fid=1, sub_
                # else merge short sub_Ps between Ps, if allowed by max recursion depth?
    intra_P(P_, fid, rdn, rng)  # recursion eval per new layer, in P_.layer_[-1]: array of mixed seg_s and sub_s
    return P_
def intra_seg(seg_, fmm, fid, rdn, rng):  # evaluate for sub-recursion in P_, filling sub_ | seg_, use adjusted rdn
    for n, (sign, I, D, M, dert_, sseg_, ssub_) in enumerate(seg_):
        if fmm:  # comp i over rng incremented as 2**n: 1, 2, 4, 8:
            if M > ave_M * rdn and len(dert_) > 2:  # ave * rdn is fixed cost of comp_rng forming len(sub_P_) = ave_nP
                ssub_, rdn = extend_comp(dert_, True, fid, rdn+1, rng)  # frng=1
                seg_[n][6][2][:], rdn = intra_P(ssub_, fid, rdn+1, rng)  # eval per seg_, sub_ = frng=0, fid=1, sub_
        elif len(dert_) > ave_Ld * rdn:  # sub-segment by d sign, der+ eval per seg, merge neg segs into nSeg for rng+?
            fid = True
            sseg_, rdn = sub_segment(dert_, False, fid, rdn+1, rng)  # d-sign seg_ = fmm=0, fid, seg_
            for sign_d, Id, Dd, Md, dert_d_, seg_d_, sub_d_ in sseg_[2]:  # seg_P in sub_seg_
                if Dd > ave_D * rdn and len(dert_) > 2:  # D accumulated in same-d-sign segment may be higher that P.D
                    sub_d_, rdn = extend_comp(dert_, False, fid, rdn+1, rng)  # frng = 0, fid = 1: cross-comp d
    layer cycle or mixed-fork: breadth-first beyond same-fork sub_?
    rng+, der+, seg_d( der+ | merge-> rng+): fixed cycle? 
    also merge initial non-selected rng+ | der+?
    sseg__ += [intra_seg(sseg_, False, fid, rdn, rng)]  # line-wide for caching only?
    return seg_
def sub_segment(P_dert_, fmm, fid, rdn, rng):  # mP segmentation by mm or d sign: initialization, accumulation, termination
    P_seg_ = []  # replaces P.seg_
    _p, _d, _m, _uni_d = P_dert_[0]  # prefix '_' denotes prior
    if fmm: _sign = _m - ave > 0  # flag: segmentation criterion is sign of mm, else sign of uni_d
    else:   _sign = _uni_d > 0
    I =_p; D =_d; M =_m; dert_= [(_p, _d, _m, _uni_d)]; seg_= []; sub_= []; layer_=[]  # initialize seg_P, same as P
    for p, d, m, uni_d in P_dert_[1:]:
        if fmm:
            sign = m - ave > 0  # segmentation crit = mm sign
        else:
            sign = uni_d > 0  # segmentation crit = uni_d sign
        if _sign != sign:
            seg_.append((_sign, I, D, M, dert_, seg_, sub_, layer_))  # terminate seg_P, same as P
            I, D, M, dert_, seg_, sub_ = 0, 0, 0, [], [], []  # reset accumulated seg_P params
        _sign = sign
        I += p; D += d; M += m; dert_.append((p, d, m, uni_d))  # accumulate seg_P params, not uni_d
    P_seg_.append((_sign, I, D, M, dert_, seg_, sub_, layer_))  # pack last segment, nothing to accumulate
    rdn *= len(P_seg_) / ave_nP  # cost per seg?
    intra_seg(P_seg_, fmm, fid, rdn+1, rng)  # evaluate for sub-recursion, different fork of intra_P, pass fmm?
    return (fmm, fid, P_seg_), rdn  # replace P.seg_, rdn
'''