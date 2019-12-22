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
line_POC is a principal version of 1st-level 1D algorithm: 

- Cross-compare consecutive pixels within each row of image, forming tuples of derivatives per pixel, then cluster them by match.
  It forms match patterns mPs: representations of contiguous sequence of pixels that form same-sign m: +mP or -mP. 
  Initial match is inverse deviation of variation: m = ave_|d| - |d|, not min: brightness doesn't correlate with predictive value.
  
- Negative mPs contain elements e_ with high-variation derts (tuples of input + derivatives), evaluated for two intra-forks:

  - direction: sub-clustering into direction patterns dPs: spans of pixels forming same-sign differences, for comp_P only
    if -M > ave_D: evaluate direction match if variation of any sign
  - derivation: cross-comp of ds per pixel forms dif_mPs: spans of ds forming same-sign md, same recursion eval as in initial mPs 
    if D > ave_D: signed because match of opposite-sign ds is -min, low comp value (but adjacent signs may still match) 
    (match = min: rng+ comp value, because predictive value of difference is proportional to its magnitude, although inversely so)
   
- Positive mPs: spans of pixels forming positive match, are evaluated for cross-comp of dert input param over incremented range 
  (positive match means that pixels have high predictive value, thus likely to match more distant pixels).
  Subsequent sub-clustering in selected +mPs forms rng_mPs, which are also processed the same way as initial mPs, recursively. 
  '''

def cross_comp(frame_of_pixels_):  # postfix '_' denotes array name, vs identical name of its elements; non-fuzzy version

    frame_of_patterns_ = []  # output frame of mPs: match patterns, including sub_patterns from recursive form_pattern
    for y in range(ini_y + 1, Y):

        pixel_ = frame_of_pixels_[y, :]  # y is index of new line pixel_
        P_ = []; P = 0, 0, 0, 0, 0, 0, []  # s, L, I, D, M, r, dert_ # initialized at each line
        pri_p = pixel_[0]
        pri_d, pri_m = 0, 0  # no d, m at x = 0

        for x, p in enumerate(pixel_[1:]):  # pixel p is compared to prior pixel pri_p in a row

            d = p - pri_p
            m = ave_d - abs(d)  # initial match is inverse deviation of |difference|
            bi_d = d + pri_d  # bilateral difference
            bi_m = m + pri_m  # bilateral match
            P, P_ = form_pattern(P, P_, pri_p, bi_d, bi_m, x+1, X)  # forms mPs: spans of pixels with same-sign m
            pri_p = p
            pri_d = d
            pri_m = m

        P_ = form_pattern(P, P_, p, d, m, x+1, X)  # terminate last P in a row, or last incomplete dert is discarded?
        frame_of_patterns_ += [P_]  # line of patterns is added to frame of patterns

    return frame_of_patterns_  # frame of patterns is output to level 2


def form_pattern(P, P_, pri_p, d, m, x, X):  # termination, evaluation for recursion, re-initialization, accumulation

    # also pass fd, rdn, rng: attach to P_ for decoding;    fd: input is derived, correlates with predictive value

    pri_s, L, I, D, M, r, dert_ = P  # r: dert_ recursion cnt = length of sub_P__ at the end of terminated dert_
    s = m > 0  # sign

    if (x > 2 and s != pri_s) or x == X:  # sign change: terminate and evaluate mP for sub-clustering and recursive comp

        if D > ave_D:  # if high accumulated-variation mP: der+ comp fork
            P = der_comp(P, rng=1, rdn=1)  # P.dert_ += [dif_mP_]

        elif -M > ave_D:  # if high unsigned-variation mP (M = summed inverse deviation of |d|)
            P = form_dP(P)  # sub-cluster by d sign (binary match, no recursive comp value), P.dert_ += [dP_]

        elif M > ave_M:  # if high-match mP: rng+ comp fork
            P = rng_comp(P, rng=1, rdn=1, fd=0)  # P.dert_ += [rng_mP_]

        P_.append(P)  # fork indicates P type
        L, I, D, M, dert_ = 0, 0, 0, 0, []  # reset accumulated params

    pri_s = s  # current sign is stored as prior sign;
    L += 1     # length of mP | dP
    I += pri_p  # accumulate params, even if incomplete at X?
    D += d
    M += m
    dert_ += [(pri_p, d, m)]
    P = pri_s, L, I, D, M, r, dert_

    return P, P_


def form_dP(P):  # simple P sub-clustering by d sign: accumulation, termination, re-initialization

    P[5] += 1  # r: recursive dert_ sub-clustering cnt;  s, L, I, D, M, r, dert_ = P
    dert_ = P[6]
    dP_ = []
    pri_p, pri_d, pri_m = dert_[0]
    pri_sd = pri_d > 0
    dP = pri_sd, 1, pri_p, pri_d, pri_m, [(pri_p, pri_d, pri_m)]  # sd, Ld, Id, Dd, Md, ddert_; no rd: no comp in dert_

    for i, (p, d, m) in enumerate(dert_[1:]):
        sd = d > 0
        if pri_sd != sd:  # terminate dP
            dP_.append(dP)
            dP[1:5] = 0, 0, 0, 0, []  # reset accumulated params

        # accumulate dP params: Ld, Id, Dd, Md, ddert_:
        dP = pri_sd, dP[1] + 1, dP[2] + pri_p, dP[3] + pri_d, dP[4] + pri_m, dP[5].append((pri_p, d, m))
        pri_p = p
        pri_d = d
        pri_m = m

    # terminate last dP in P.dert_:
    dP = sd, dP[1] + 1, dP[2] + p, dP[3] + d, dP[4] + m, dP[5].append((pri_p, d, m))
    dP_.append(dP)
    P[6].append((2, dP_))  # append deeper layer' typ (rng_mP=0 | dif_mP=1 | dP=2) and P_ to terminated dert_

    return P


def der_comp(P, rng, rdn):  # same as cross_comp for ds in dert_, forming md and dd (may match across d sign)

    s, L, I, D, M, r, dert_ = P
    dif_mP_ = []  # new sub-patterns:
    dif_mP = int(dert_[0][2] > 0), 0, 0, 0, 0, 0, []  # pri_sd, Ld, Id, Dd, Md, rd, ddert_
    r += 1  # dert_ recursion count = length of sub_P__ at the end of terminated dert_

    pri_d = dert_[0][1]  # input d
    pri_dd, pri_md = 0, 0  # for bilateral summation, no d, m at x = 0

    for x, d in enumerate(dert_[1:]):  # pixel p is compared to prior pixel in a row
        dd = d - pri_d
        md = min(d, pri_d) - ave_m  # evaluation of md (min d: magnitudes derived from d correspond to predictive value)
        # form dif_mPs: spans of derts with same-sign md:
        dif_mP, dif_mP_ = form_pattern(dif_mP, dif_mP_, pri_d, dd + pri_dd, md + pri_md, x+1, X)  # fd = 1 if typ?
        pri_d = d
        pri_dd = dd
        pri_md = md

    # terminate last dif_mP in P.dert_:
    dif_mP, dif_mP_ = form_pattern(dif_mP, dif_mP_, pri_d, dd + pri_dd, md + pri_md, x + 1, X)
    dert_.append((1, dif_mP_))  # append deeper layer' typ (rng_mP=0 | dif_mP=1 | dP=2) and P_ to terminated dert_

    return P


def rng_comp(P, rng, rdn, fd):  # rng+ fork: cross-comp rng-distant ps in dert_; fd: flag dderived

    s, L, I, D, M, r, dert_ = P
    rng_mP_ = []  # new sub_patterns:
    rng_mP = int(dert_[0][2] > 0), 0, 0, 0, 0, 0, []  # pri_sr, Lr, Ir, Dr, Mr, rr, rdert_
    r += 1  # dert_ recursion count = length of sub_P__ at the end of terminated dert_
    rng += 1

    for i in range(rng, L + 1):  # comp between rng-distant pixels, also bilateral, if L > rng * 2?
        p, acc_d, acc_m = dert_[i]
        _p, _acc_d, _acc_m = dert_[i - rng]
        d = p - _p  # rng difference
        if fd:
            m = min(p, _p) - ave_m  # magnitude of vars derived from d corresponds to predictive value, thus direct match
        else:
            m = ave_d - abs(d)  # magnitude of brightness doesn't correlate with stability, thus inverse match
        acc_d += d
        acc_m += m  # accumulates difference and match between p and all prior ps in extended rng
        dert_[i] = (p, acc_d, acc_m)
        _acc_d += d  # accumulates difference and match between p and all prior and subsequent ps in extended rng
        _acc_m += m
        if i >= rng * 2:  # form rng_mPs: spans of pixels with same-sign _acc_m:
            rng_mP, rng_mP_ = form_pattern(rng_mP, rng_mP_, _p, _acc_d, _acc_m, rdn + 1, rng, i, L)

    # terminate last rng_mP in P.dert_
    rng_mP, rng_mP_ = form_pattern(rng_mP, rng_mP_, _p, _acc_d, _acc_m, rdn + 1, rng, i, L)
    dert_.append((0, rng_mP_))  # append deeper layer' typ (rng_mP=0 | dif_mP=1 | dP=2) and P_ to terminated dert_

    return P

# pattern filters / hyper-parameters: eventually from higher-level feedback, initialized here as constants:

ave_m = 10  # min dm for positive dmP
ave_d = 20  # |difference| between pixels that coincides with average value of mP - redundancy to overlapping dPs
ave_M = 127  # min M for initial incremental-range comparison(t_)
ave_D = 127  # min |D| for initial incremental-derivation comparison(d_), also for form dP?
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
'''