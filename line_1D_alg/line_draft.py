import cv2
import argparse
from time import time
from collections import deque

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='.//raccoon.jpg')
arguments = vars(argument_parser.parse_args())
image = cv2.imread(arguments['image'], 0).astype(int)  # load pix-mapped image

# same image loaded online, without cv2:
# from scipy import misc
# image = misc.face(gray=True).astype(int)
''' 
  line_POC is a principal version of 1st-level 1D algorithm, with the following operations: 

- Cross-compare consecutive pixels within each row of image, forming dert_ queue of derts: tuples of derivatives per pixel. 
  dert_ is then segmented by match deviation, forming mPs, each a contiguous sequence of pixels that form same-sign m: +mP or -mP. 
  Initial match is inverse deviation of variation: m = ave_|d| - |d|, not min: brightness doesn't correlate with predictive value.
   
- Positive mPs: spans of pixels forming positive match, are evaluated for cross-comp of dert input param over incremented range 
  (positive match means that pixels have high predictive value, thus likely to match more distant pixels).
- Negative mPs: high-variation spans, are evaluated for cross-comp of difference, which forms higher derivatives.
  Both types of extended cross-comp are recursive: resulting sub-patterns are evaluated for deeper cross-comp, same as top patterns. 

  If value of incremental range or derivation cross-comp over full pattern is low, but the pattern is long enough, then its dert_ 
  is segmented by (m-ave) for +mPs and d-sign match for -mPs. Value of resulting positive segments (seg_Ps) may be higher than that 
  of full pattern: seg_ave_M < ave_M, and D in same-d-sign segments (direction Ps) is not sign-cancelled?
  Thus, positive m-segments are evaluated for local rng_comp, positive d-segments are evaluated for local der_comp.     
  
  Signed D because match of opposite-sign ds is -min, low comp value, but adjacent signs may still match? 
  (match = min: rng+ comp value, because predictive value of difference is proportional to its magnitude, although inversely so)
  '''

def cross_comp(frame_of_pixels_):  # postfix '_' denotes array name, vs identical name of its elements; non-fuzzy version

    frame_of_patterns_ = []  # output frame of mPs: match patterns, including sub_patterns from recursive form_pattern
    for y in range(ini_y + 1, Y):

        pixel_ = frame_of_pixels_[y, :]    # y is index of new line pixel_
        P_ = []; P = 0, 0, 0, 0, 0, 0, []  # s, L, I, D, M, r, dert_  # initialized at each line
        pri_p = pixel_[0]
        pri_d, pri_m = 0, 0  # no d, m at x = 0

        for x, p in enumerate(pixel_[1:]):  # pixel p is compared to prior pixel pri_p in a row

            d = p - pri_p
            m = ave - abs(d)  # initial match is inverse deviation of |difference|
            bi_d = d + pri_d  # bilateral difference
            bi_m = m + pri_m  # bilateral match
            P, P_ = form_pattern(P, P_, pri_p, bi_d, bi_m, x+1, X, fd=0, rdn=1, rng=1)
            # forms mPs: spans of pixels that form same-sign m
            pri_p = p
            pri_d = d
            pri_m = m

        # terminate last P in a row (or last incomplete dert is discarded?):
        P_ = form_pattern(P, P_, p, d, m, x+1, X, fd=0, rdn=1, rng=1)

        frame_of_patterns_ += [P_]  # line of patterns is added to frame of patterns

    return frame_of_patterns_  # frame of patterns is output to level 2


def form_pattern(P, P_, pri_p, d, m, x, X, fd, rdn, rng):  # initialization, accumulation, termination, recursion
    '''
    rdn, rng are incremental if seq access, rdn += 1 * typ coef, = r * typ coefs?  no typ in form_pattern?
    fd: flag input is derived, thus correlates with predictive value, = 1 if sub_P_ fd or arg fd
    all forks >> P.dert_ += [sub_P_], exclusive or eval by ave * rdn?
    M = summed (ave |d| - |d|)
    '''
    pri_s, L, I, D, M, r, dert_ = P  # r: recursion flag, or count for par comp: n of sub_P_ appends to terminated dert_
    s = m > 0  # m sign, defines positive | negative mPs

    if (x > 2 and s != pri_s) or x == X:  # sign change: terminate and evaluate mP for sub-clustering and recursive comp

        if s:  # positive mP
            if M > ave_M * rdn and L > rng * 2:  # M > fixed costs of rng_comp: full-P, no sub-segmentation:
                P = rng_comp(P, fd, rdn+1, rng+1)  # P.dert_ += [rng_mP_]

            elif L > ave_Lm * rdn:  # long but weak +mP: may contain mmPs fit for rng_comp, no M eval?
                P = sub_segment(P, 1, fd, rdn+1, rng)  # segment by mm sign: |-m| - ave, eval rng_comp per seg_mP
        else:
            if D > ave_D * rdn and L > rng * 2:  # D > fixed costs of der_comp: full-P, no sub-segmentation:
                P = der_comp(P, rdn+1, rng*2-1)  # rng between central pixels = d accumulation range * 2

            elif L > ave_Ld * rdn:  # long but weak -mP: may contain dPs fit for der_comp, no D eval?
                P = sub_segment(P, 0, fd, rdn+1, rng)  # segment by d sign (non-random), eval der_comp per seg_dP

        P_.append(P)  # sub_P_s are appended to dert_ with comp fork typ (sub_P type)
        L, I, D, M, dert_ = 0, 0, 0, 0, []  # reset accumulated params

    if x == X: d *= 2; m *= 2  # project bilateral values from incomplete unilateral values

    pri_s = s  # current sign is stored as prior sign;
    L += 1     # length of mP | dP
    I += pri_p  # accumulate params
    D += d
    M += m
    dert_ += [(pri_p, d, m)]
    P = pri_s, L, I, D, M, r, dert_

    return P, P_


def sub_segment(P, fm, fd, rdn, rng):  # mP segmentation by mm or d sign: initialization, accumulation, termination

    P[5] = 1  # r: recursion flag, may replace with count
    P_dert_ = P[6]  # s, L, I, D, M, r, dert_ = P
    seg_P_ = []   # +|- mm|d- sign segments
    pri_p, pri_d, pri_m = P_dert_[0]
    if fm: pri_s = pri_d > 0
    else:  pri_s = pri_m - ave > 0
    L, I, D, M, r, dert_ = 1, pri_p, pri_d, pri_m, 0, [(pri_p, pri_d, pri_m)]  # initialize seg_P

    for p, d, m in P_dert_[1:]:
        if fm: s = d > 0
        else:  s = m - ave > 0
        if pri_s != s:
            # terminate seg_P:
            seg_P = pri_s, L, I, D, M, r, dert_
            if fm:
                if M > ave_M * rdn and L > rng * 2:
                    seg_P = rng_comp(seg_P, fd, rdn + 1, rng + 1)  # incremental-range cross-comp
                seg_P_.append(seg_P)
            else:
                if D > ave_D * rdn and L > rng * 2:
                    seg_P = der_comp(seg_P, rdn + 1, rng * 2 - 1)  # incremental-derivation cross-comp
                seg_P_.append(seg_P)
            L, I, D, M, dert_ = 0, 0, 0, 0, []  # reset accumulated params

        L += 1; I += p; D += d; M += m; dert_.append((p, d, m))  # accumulate params
        pri_s = s

    seg_P_.append((pri_s, L, I, D, M, r, dert_))  # pack last sub_P, nothing to accumulate
    P[6].append((fm, seg_P_))  # append typ and sub_P_ to pre-terminated dert_

    return P


def intra_comp(P, frng, fd, rdn, rng):  # extended cross_comp within dert_, range comp if frng, else deriv comp

    s, L, I, D, M, r, dert_ = P  # full P or sub_P from sub-clustering
    P[5] = 1  # r: recursion flag | count = number of sub_P_s at the end of terminated dert_
    sub_mP_ = []  # new sub-patterns:
    sub_mP = int(dert_[0][2] > 0), 0, 0, 0, 0, 0, []  # pri_ss, sL, sI, sD, sM, sr, sdert_
    sub_dert_ = []  # sub_mP[-1]

    if frng:  # rng_comp: bilateral cross-comp between rng-distant pixels in dert_

        for x in range(rng, L + 1):
            i, acc_d, acc_m = dert_[x]
            _i, _acc_d, _acc_m = dert_[x - rng]
            d = i - _i
            if fd:  # flag derived
                m = min(i, _i) - ave_m  # magnitude of vars derived from d corresponds to predictive value, thus direct match
            else:
                m = ave - abs(d)  # magnitude of brightness doesn't correlate with stability, thus inverse match
            acc_d += d  # back_d: accumulated difference between p and all prior ps in rng
            acc_m += m  # back_m: accumulated match between p and all prior and subsequent ps in rng
            dert_[x] = (i, acc_d, acc_m)
            _acc_d += d  # bi_d: accumulated difference between p and all prior ps in rng
            _acc_m += m  # bi_d: accumulated match between p and all prior and subsequent ps in rng

            if x >= rng * 2:  # form rng_mPs: spans of pixels with same-sign acc_m, else ignore sign, continue accumulation
                sub_mP, sub_mP_ = form_pattern(sub_mP, sub_mP_, _i, _acc_d, _acc_m, x, L, fd, rdn + 1, rng)

    else:   # der_comp: cross_comp between consecutive ds in dert_, forming md and dd (may match across d sign)

        sub_dert_[:rng][0] = dert_[rng:rng * 2][1]  # i_ = d_
        sub_dert_[:rng][1] = dert_[rng:rng * 2][1] - dert_[:rng][1]  # dd_ = d_ - _d_
        sub_dert_[:rng][2] = min(dert_[rng:rng * 2][1], dert_[:rng][1])  # dm_ = min(d_,_d_)

        for x in range(rng, L + 1):
            i = dert_[x][1]  # i = d
            _i, back_d, back_m = sub_dert_[x][1]
            d = i - _i  # dd
            m = min(i, _i) - ave_m  # md, magnitude of vars derived from d corresponds to predictive value, thus direct match
            bi_d = back_d + d  # bilateral d-difference in rng
            bi_m = back_m + m  # bilateral d-match in rng
            sub_dert_[x] = i, d, m  # _i, back_d, back_m

            if x >= rng * 2:  # else id, im = 0, 0: initialized
                sub_mP, sub_mP_ = form_pattern(sub_mP, sub_mP_, _i, bi_d, bi_m, x, L, fd, rdn + 1, rng)

    # terminate last sub_mP in P.dert_:
    sub_mP, sub_mP_ = form_pattern(sub_mP, sub_mP_, _i, bi_d, bi_m, x + 1, X, 1, rng, rdn)
    dert_.append((frng, sub_mP_))  # append typ and sub_P_ to per-terminated dert_

    return P


def der_comp(P, rdn, rng):  # cross_comp of ds in dert_, forming md and dd (may match across d sign)

    s, L, I, D, M, r, dert_ = P  # full P or sub_P from sub-clustering
    P[5] = 1  # r: recursion flag | count = number of sub_P_s at the end of terminated dert_
    dif_mP_ = []  # new sub-patterns:
    dif_mP = int(dert_[0][2] > 0), 0, 0, 0, 0, 0, []  # pri_sd, Ld, Id, Dd, Md, rd, ddert_

    pri_d = dert_[0][1]  # input d
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

# pattern filters / hyper-parameters: eventually from higher-level feedback, initialized here as constants:


ave = 20  # |difference| between pixels that coincides with average value of mP - redundancy to overlapping dPs
ave_m = 10  # for m defined as min
ave_M = 255  # min M for initial incremental-range comparison(t_)
ave_D = 127  # min |D| for initial incremental-derivation comparison(d_)
ave_Lm = 20  # min L for sub_cluster(m), higher cost than der_comp?
ave_Ld = 10  # min L for sub_cluster(d)
ini_y = 0
# min_rng = 1  # >1 if fuzzy pixel comparison range, initialized here but eventually adjusted by higher-level feedback

Y, X = image.shape  # Y: frame height, X: frame width

start_time = time()
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