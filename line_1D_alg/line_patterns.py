import cv2
import argparse
from time import time
from collections import deque

# pattern filters or hyper-parameters: eventually from higher-level feedback, initialized here as constants:
ave = 20  # |difference| between pixels that coincides with average value of mP - redundancy to overlapping dPs
ave_m = 10  # for m defined as min
ave_M = 255  # min M for initial incremental-range comparison(t_), higher cost than der_comp?
ave_D = 127  # min |D| for initial incremental-derivation comparison(d_)
ave_Lm = 20  # min L for sub_cluster(m), higher cost than der_comp?
ave_Ld = 10  # min L for sub_cluster(d)
ini_y = 400
# min_rng = 1  # >1 if fuzzy pixel comparison range, for sensor-specific noise only

''' 
  line_patterns is a principal version of 1st-level 1D algorithm, contains following operations: 

- Cross-compare consecutive pixels within each row of image, forming dert_ queue of derts: tuples of derivatives per pixel. 
  dert_ is then segmented by match deviation, forming mPs, each a contiguous sequence of pixels that form same-sign m: +mP or -mP. 
  Initial match is inverse deviation of variation: m = ave_|d| - |d|, not min: brightness doesn't correlate with predictive value.

- Positive mPs: spans of pixels forming positive match, are evaluated for cross-comp of dert input param over incremented range 
  (positive match means that pixels have high predictive value, thus likely to match more distant pixels).
- Negative mPs: high-variation spans, are evaluated for cross-comp of difference, which forms higher derivatives.
  Both types of extended cross-comp are recursive: resulting sub-patterns are evaluated for deeper cross-comp, same as top patterns. 

  If value of incremental range or derivation cross-comp over full pattern is low, but the pattern is long enough, then its dert_ 
  is segmented by (m-ave) for +mPs and d-sign match for -mPs. Value density of resulting positive segments (seg_Ps) is higher than 
  in full pattern: +mm_seg_ave_M = ave_M / 2, and same-d-sign seg_D is not sign-cancelled (+mm: flat seg, ds: directed seg?)
  Thus, positive m-segments are evaluated for local rng_comp, and positive d-segments are evaluated for local der_comp.     

  comp_d eval by D because match of opposite-sign ds is -min: low comp value, but adjacent dd signs may still match?  
  (match = min: rng+ comp value, because predictive value of difference is proportional to its magnitude, although inversely so)

  Initial cross-comp here is 1D slice of 2D 3x3 kernel, while single unidirectional d is equivalent to 2x2 kernel.
  Odd kernels preserve resolution of pixels, while 2x2 kernels preserve resolution of derivatives, in resulting derts.
  The former should be used in rng_comp and the latter in der_comp, which may alternate with intra_comp.
  '''

def cross_comp(frame_of_pixels_):  # postfix '_' denotes array name, vs identical name of its elements; non-fuzzy version

    Y, X = image.shape  # Y: frame height, X: frame width
    frame_of_patterns_ = []  # output frame of mPs: match patterns, including sub_patterns from recursive form_pattern
    for y in range(ini_y + 1, Y):

        pixel_ = frame_of_pixels_[y, :]  # y is index of new line pixel_
        P_ = []  # row of patterns, initialized at each line
        P = 0, 0, 0, 0, 0, [], [], []  # s, L, I, D, M, sub_, seg_, dert_
        pri_p = pixel_[0]
        pri_d, pri_m = 0, 0  # no d, m at x = 0

        for x, p in enumerate(pixel_[1:], start=1):  # pixel p is compared to prior pixel pri_p in a row
            d = p - pri_p
            m = ave - abs(d)  # initial match is inverse deviation of |difference|
            bi_d = d + pri_d  # bilateral difference
            bi_m = m + pri_m  # bilateral match
            dert = pri_p, bi_d, bi_m, d
            # accumulate or terminate mP: span of pixels forming same-sign m:
            P, P_ = form_pattern(P, P_, dert, x, X, fid=0, rdn=1, rng=1)
            pri_p = p
            pri_d = d
            pri_m = m
        # terminate last P in row:
        dert = p, d * 2, m * 2, d  # project unilateral to bilateral values
        P_ = form_pattern(P, P_, dert, x, X, fid=0, rdn=1, rng=1)

        frame_of_patterns_ += [P_]  # line of patterns is added to frame of patterns

    return frame_of_patterns_  # frame of patterns is output to level 2


def form_pattern(P, P_, dert, x, X, fid, rdn, rng):  # initialization, accumulation, termination, recursion
    '''
    rdn, rng are incremental per layer, seq access, rdn += 1 * typ coef?
    fid: flag input is derived, magnitude correlates with predictive value
    exclusive forks or eval by ave * rdn?  M = summed (ave |d| - |d|)
    '''
    pri_s, L, I, D, M, sub_, seg_, dert_ = P  # sub_ and seg_ are stacks of layers, appended by feedback
    pri_p, d, m, uni_d = dert
    s = m > 0  # m sign, defines positive | negative mPs

    if (x > rng * 2 and s != pri_s) or x == X:  # sign change: terminate mP, evaluate for sub_segment & intra_comp
        if s:  # positive mP
            if M > ave_M * rdn and L > rng * 2:  # M > fixed costs of full-P comp_rng, /=2 in sub_segment
                sub_ = intra_comp(dert_, 1, fid, rdn+1, rng+1)  # comp_rng: sub_ = frng=1, fid, sub_

            elif L > ave_Lm * rdn:  # long but weak +mP may contain high-mm segments: comp_rng eval, no M eval?
                seg_ = sub_segment(dert_, 1, fid, rdn+1, rng)  # segment by mm sign: seg_ = fmm=1, fid, seg_
        else:
            if -M > ave_D * rdn and L > rng * 2:  # -M > fixed costs of full-P comp_d: sub_ = frng=0, fid=1, sub_:
                sub_ = intra_comp(dert_, 0, 1, rdn+1, rng=1)  # comp_d: unidirectional to preserve resolution

            elif L > ave_Ld * rdn:  # long but weak -mP may contain high-D segments: comp_d eval, no D eval?
                seg_ = sub_segment(dert_, 0, 1, rdn+1, rng=1)  # segment by d sign: seg_ = fmm=0, fid=1, seg_

        P[5][:] = sub_; P[6][:] = seg_  # may have been filled by intra_comp and sub_segment
        P_.append(P)
        L, I, D, M, sub_, seg_, dert_ = 0, 0, 0, 0, [], [], []  # reset accumulated params

    pri_s = s  # current sign is stored as prior sign
    L += 1     # length of mP | dP
    I += pri_p  # accumulate params with bilateral values:
    D += d
    M += m
    dert_ += [(pri_p, d, m, uni_d)]  # uni_d is for comp_d and seg_d
    P = pri_s, L, I, D, M, sub_, seg_, dert_  # sub_ and seg_ are accumulated in intra_comp and sub_segment

    return P, P_


def intra_comp(dert_, frng, fid, rdn, rng):  # extended cross_comp within P_dert_, range comp if frng, else deriv comp

    sub_P = int(dert_[0][2] > 0), 0, 0, 0, 0, [], [], []  # s, L, I, D, M, sub_, seg_, dert_: same as master P or segment
    sub_P_ = [] # return to replace sub_
    buff_ = deque([])
    if frng:  # initialize prior-rng derts with _i, _d=0, _m=0
        for x in range(rng):
            buff_.append((dert_[x][0], 0, 0))
    else:   # initialize prior-rng derts with uni_d, _d=0, _m=0
        for x in range(rng):
            buff_.append((dert_[x][3], 0, 0))

    for x in range(rng, len(dert_) + 1):  # backward rng-distant comp, prefix '_' denotes prior of two same-name variables

        if frng:  # bilateral comp between rng-distant pixels in dert_, forming rng_d and rng_m
            i, _bi_d, _bi_m = dert_[x][:2]  # rng-1 dert
            _i, _d, _m = buff_.popleft   # _d, _m include rng-1 bi_d | bi_m values;  vs. [x-rng]
            d = i - _i
            if fid: m = min(i, _i) - ave_m  # match=min: magnitude of derived vars correlate with stability
            else:   m = ave - abs(d)  # inverse match: intensity doesn't correlate with stability
            d += _bi_d
            m += _bi_m
            buff_.append((i, d, m))  # future _i, _d, _m
            bi_d = _d + d  # bilateral difference, accum in rng
            bi_m = _m + m  # bilateral match, accum in rng
            dert = _i, bi_d, bi_m, d
            # P accumulation or termination:
            sub_P, sub_P_ = form_pattern(sub_P, sub_P_, dert, x, len(dert_), fid, rdn + 1, rng)

        else:  # bilateral comp between rng-distant unilateral ds in dert_, forming dd and md (may match across d sign)
            fid = 1  # flag i is derived
            i = dert_[x][3]  # i is unilateral d
            _i, _d, _m = buff_.popleft
            d = i - _i  # d is dd
            m = min(i, _i) - ave_m  # md, min because magnitude of derived vars corresponds to predictive value
            bi_d = _d + d  # bilateral d-difference per _i
            bi_m = _m + m  # bilateral d-match per _i
            buff_.append((i, d, m))  # future _i, _d, _m
            dert = _i, bi_d, bi_m, d
            # P accumulation or termination:
            sub_P, sub_P_ = form_pattern(sub_P, sub_P_, dert, x, len(dert_), fid, rdn + 1, rng)

    # terminate last sub_P in dert_:
    dert = i, d * 2, m * 2, d  # project unilateral to bilateral values
    _, sub_P_ = form_pattern(sub_P, sub_P_, dert, x+1, len(dert_), fid, rdn + 1, rng)

    return [frng, fid, sub_P_]  # replaces P.sub_


def sub_segment(P_dert_, fmm, fid, rdn, rng):  # mP segmentation by mm or d sign: initialization, accumulation, termination

    seg_ = []  # replaces P.seg_
    _p, _d, _m, _uni_d = P_dert_[1]
    if fmm: pri_s = _m - ave > 0
    else:   pri_s = _uni_d > 0
    L, I, D, M, sub_, dert_ = 1, _p, _d, _m, [], [(_p, _d, _m, _uni_d)]  # initialize seg_P

    for p, d, m, uni_d in P_dert_[1:]:
        if fmm: s = m - ave > 0  # segmentation crit = mm sign
        else:   s = uni_d > 0  # segmentation crit = uni_d sign
        if pri_s != s:
            # terminate segment:
            seg_P = pri_s, L, I, D, M, sub_, dert_  # same as P except no seg_
            if fmm:
                if M > ave_M / 2 * rdn and L > rng * 2:  # ave_M / 2: reduced because mm filter = m filter * 2
                    seg_P = intra_comp(dert_, 1, fid, rdn + 1, rng + 1)  # frng = 1: incremental-range cross-comp
                seg_.append(seg_P)
            else:
                fid = 1
                if D > ave_D * rdn and L > rng * 2:  # D of same-d-sign segment may be higher that P.D
                    seg_P = intra_comp(dert_, 0, fid, rdn + 1, rng)  # frng = 0, fid = 1: cross-comp d
                seg_.append(seg_P)
            L, I, D, M, sub_, dert_ = 0, 0, 0, 0, [], []  # reset accumulated seg_P params

        L += 1; I += p; D += d; M += m; dert_.append((p, d, m))  # accumulate seg_P params, not uni_d?
        pri_s = s

    seg_.append((pri_s, L, I, D, M, sub_, dert_))  # pack last segment, nothing to accumulate

    return [fmm, fid, seg_]  # replace P.seg_


if __name__ == "__main__":
    # Parse argument (image)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image',
                                 help='path to image file',
                                 default='.//raccoon.jpg')
    arguments = vars(argument_parser.parse_args())
    # Read image
    image = cv2.imread(arguments['image'], 0).astype(int)  # load pix-mapped image
    # same image loaded online, without cv2:
    # from scipy import misc
    # image = misc.face(gray=True).astype(int)

    # Initialize timer
    start_time = time()
    # Main
    frame_of_patterns_ = cross_comp(image)
    # Stop timer and print the result
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

Deprecated functions:

def der_comp(P, rdn, rng):  # cross_comp of ds in dert_, forming md and dd (may match across d sign)

    s, L, I, D, M, r, dert_ = P  # full P or sub_P from sub-clustering
    P[5] = 1  # r: recursion flag | count = number of sub_P_s at the end of terminated dert_
    dif_mP_ = []  # new sub-patterns:
    dif_mP = int(dert_[0][2] > 0), 0, 0, 0, 0, 0, []  # pri_sd, Ld, Id, Dd, Md, rd, ddert_

    pri_d = dert_[0][1]  # input d;  new dert, no p?
    pri_dd, pri_md = 0, 0  # for bilateral summation, no d, m at x = 0

    for x, d in enumerate(dert_[1:]):  # pixel p is compared to prior pixel in a row
        dd = d - pri_d
        md = min(d, pri_d) - ave_m  # evaluation of md (min d: magnitudes derived from d correspond to predictive value)
        # form dif_mPs: spans of derts with same-sign md:
        dif_mP, dif_mP_ = form_pattern(dif_mP, dif_mP_, pri_d, dd + pri_dd, md + pri_md, x+1, X, 1, rng, rdn)
        pri_d = d
        pri_dd = dd
        pri_md = md
    # terminate last dif_mP in P.dert_:
    dif_mP, dif_mP_ = form_pattern(dif_mP, dif_mP_, pri_d, dd, md, x + 1, X, 1, rng, rdn)
    dert_.append((1, dif_mP_))  # append deeper layer' typ (rng_mP=0 | dif_mP=1 | dP=2) and P_ to terminated dert_

    return P

def rng_comp(P, fd, rdn, rng):  # cross-comp of rng-distant ps in dert_; fd: flag dderived

    s, L, I, D, M, r, dert_ = P
    P[5] = 1  # r: recursion flag | count = number of sub_P_s at the end of terminated dert_
    rng_mP_ = []  # new sub_patterns:
    rng_mP = int(dert_[0][2] > 0), 0, 0, 0, 0, 0, []  # pri_sr, Lr, Ir, Dr, Mr, rr, rdert_

    for i in range(rng, L + 1):  # bilateral comp between rng-distant pixels
        p, acc_d, acc_m = dert_[i]
        _p, _acc_d, _acc_m = dert_[i - rng]
        d = p - _p
        if fd:
            m = min(p, _p) - ave_m  # magnitude of vars derived from d corresponds to predictive value, thus direct match
        else:
            m = ave - abs(d)  # magnitude of brightness doesn't correlate with stability, thus inverse match
        acc_d += d
        acc_m += m  # accumulates difference and match between p and all prior ps in extended rng
        dert_[i] = (p, acc_d, acc_m)
        _acc_d += d  # accumulates difference and match between p and all prior and subsequent ps in extended rng
        _acc_m += m
        if i >= rng * 2:    # form rng_mPs: spans of pixels with same-sign acc_m:
            rng_mP, rng_mP_ = form_pattern(rng_mP, rng_mP_, _p, _acc_d, _acc_m, i, L, fd, rdn + 1, rng)
    # terminate last rng_mP in P.dert_:
    rng_mP, rng_mP_ = form_pattern(rng_mP, rng_mP_, _p, _acc_d, _acc_m, i, L, fd, rdn + 1, rng)
    dert_.append((0, rng_mP_))  # append deeper layer' typ (rng_mP=0 | dif_mP=1 | dP=2) and P_ to terminated dert_

    return P

def sub_cluster_d(P, rdn, rng):  # d-specific version of sub-clustering, similar m version

    P[5] += 1  # r: recursive dert_ sub-clustering cnt
    dert_ = P[6]  # s, L, I, D, M, r, dert_ = P
    dP_ = []
    pri_p, pri_d, pri_m = dert_[0]
    pri_sd = pri_d > 0
    dP = pri_sd, 1, pri_p, pri_d, pri_m, [(pri_p, pri_d, pri_m)]  # initialize dP: sd, Ld, Id, Dd, Md, ddert_

    for p, d, m in dert_[1:]:
        sd = d > 0
        if pri_sd != sd:  # terminate dP

            if dP[3] > ave_D:
                dP = der_comp(dP, rdn + 1, rng * 2 - 1)  # rng between central pixels avoids overlap in d scope

            dP_.append(dP)
            dP = sd, 0, 0, 0, 0, []  # reset accumulated params

        dP = sd, dP[1] + 1, dP[2] + p, dP[3] + d, dP[4] + m, dP[5].append((p, d, m))  # accumulate Ld, Id, Dd, Md, ddert_
        pri_sd = sd

    dP_.append(dP)  # pack last dP in P.dert_, nothing to accumulate
    P[6].append((2, dP_))  # append deeper layer' typ (rng_mP=0 | dif_mP=1 | dP=2) and P_ to terminated dert_

    return P
'''