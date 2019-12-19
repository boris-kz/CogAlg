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

- Cross-comparison of consecutive pixels within each row of image, then clustering them by match, forming match patterns mPs.
  Initial match is indirect: m = ave_|d| - |d|: inverse variation, because brightness doesn't correlate with predictive value.
  Each mP represents contiguous spans of pixels that form same-sign m. 

- Negative mPs contain elements e_ with high-variation derts (tuples of input + derivatives), evaluated for two intra-forks:

  - direction: sub-clustering into direction patterns dPs: spans of pixels forming same-sign differences, 
    if -M > ave: value of direction is variation of any sign? 
  - derivation: cross-comp of element ds, forming mdPs: spans of ds forming same-sign md, mdP recursion eval, same as initial mPs 
    if D > ave: signed because match of opposite-sign ds is -min, low comp value (but adjacent signs may still match) 
    (match=min: rng+ comp value, because predictive value of difference is proportional to its magnitude, although inversely so).
   
- Positive mPs: spans of pixels forming positive match, are evaluated for cross-comp of dert input param over incremented range 
  (positive match means that pixels have high predictive value, thus likely to match more distant pixels).
  Subsequent sub-clustering in selected +mPs forms rng_mPs, which are also processed the same way as initial mPs, recursively. 
  '''

def cross_comp(frame_of_pixels_):  # postfix '_' denotes array name, vs identical name of its elements; non-fuzzy version

    frame_of_patterns_ = []  # output frame of mPs: match patterns, including sub_patterns from recursive form_pattern
    for y in range(ini_y + 1, Y):

        pixel_ = frame_of_pixels_[y, :]  # y is index of new line pixel_
        P_ = []; P = 0, 0, 0, 0, 0, 0, []  # pri_s, L, I, D, M, r, dert_ # initialized at each line
        pri_p = pixel_[0]
        pri_d, pri_m = 0, 0  # no d, m at x = 0

        for x, p in enumerate(pixel_[1:]):  # pixel p is compared to prior pixel in a row
            d = p - pri_p
            m = ave_d - abs(d)  # initial match is defined as inverse deviation of |difference|
            P, P_ = form_pattern(P, P_, pri_p, d + pri_d, m + pri_m, 1, x+1, X)  # forms mPs: spans of pixels with same-sign m
            pri_p = p
            pri_d = d
            pri_m = m

        P_ = form_pattern(P, P_, pri_p, d + pri_d, m + pri_m, 1, x + 1, X)  # terminate last incomplete P
        frame_of_patterns_ += [P_]  # line of patterns is added to frame of patterns, last incomplete dert are discarded

    return frame_of_patterns_  # frame of patterns is output to level 2


def form_pattern(P, P_, pri_p, d, m, ini, x, X):  # accumulation, termination, recursion eval, flag to stop recursion in dP?

    s = m > 0  # sign
    if (x > 2 and s != P[0]) or x == X:  # sign change: terminate and evaluate mP for sub-clustering and recursive comp

        if ini: # initial call, vs =0 for cluster by d call

            if P[3] > ave_D:  # if D (high cross-sign-cancelled variation mP): recursive cross-comp(d), der+ fork
                P = der_comp(P, rng=1, rdn=1, Pr=1)  # includes P.e_ += [mdP_]

            elif -P[4] > ave_D: # if -M (high-variation mP): cluster by d sign: binary match, no value for cross-comp recursion
                dP_ = []
                dert_ = P[-1]  # list of element derts
                dP = int(dert_[0][1] > 0), 0, 0, 0, 0, 0, []  # pri_sd, Ld, Id, Dd, Md, rng, ed_
                L = P[1]
                for i in range(L + 1):
                    id, idd, imd = dert_[i]
                    sd = 1 if id > 0 else 0
                    if (dP[0] != sd and i > 0) or i == L:
                        dP, dP_ = form_pattern(dP, dP_, id, idd, imd, 0, i, L)  # ini=0: simple clustering by d sign
                dert_.append(dP_)  # P.e_ += [dP_]

            elif P[4] > ave_M:  # if M (high-match mP): recursive cross-comp(rng_p), rng+ fork
                P = rng_comp(P, rng=1, rdn=1, Pr=1, dderived=0)  # includes P.e_ += [rng_mP_]

        P_.append(P)  # fork indicates P type
        s, L, I, D, M, dert_ = 0, 0, 0, 0, 0, []
    else:
        pri_s, L, I, D, M, dert_ = P  # depth of derts in dert_ = r: depth of prior comp recursion within P

    L += 1; I += pri_p; D += d; M += m; dert_.append((pri_p, d, m))
    P = s, L, I, D, M, dert_

    return P, P_


def der_comp(P, rng, rdn, Pr):  # same as cross_comp for ds in dert_, forming md and dd (may match across d sign)

    s, L, I, D, M, r, dert_ = P
    mdP_ = []  # new sub_patterns:
    mdP = int(dert_[0][2] > 0), 0, 0, 0, 0, 0, []  # pri_sd, Ld, Id, Dd, Md, rd, ddert_
    P[5] += 1  # r: dert_ recursion count = depth of P_ in dert_[-1]

    pri_d = dert_[0][1]  # input d
    pri_dd, pri_md = 0, 0  # for bilateral summation, no d, m at x = 0

    for x, d in enumerate(dert_[1:]):  # pixel p is compared to prior pixel in a row
        dd = d - pri_d
        md = min(d, pri_d) - ave_m  # evaluation of md (min d: magnitudes derived from d correspond to predictive value)
        mdP, mdP_ = form_pattern(mdP, mdP_, pri_d, dd + pri_dd, md + pri_md, 1, x+1, X)  # dderived = 1
        # forms mPs: spans of pixels with same-sign md, eval for der+ and rng+ comp
        pri_d = d
        pri_dd = dd
        pri_md = md

    mdP, mdP_ = form_pattern(mdP, mdP_, pri_d, dd + pri_dd, md + pri_md, 1, x + 1, X)  # terminate last incomplete P
    P[-1].append(mdP_)  # dert_ += [mdP_], or replace dert_?

    return P

def rng_comp(P, rng, rdn, Pr, dderived):  # rng+ fork: cross-comp rng-distant ps in dert_

    s, L, I, D, M, r, dert_ = P
    rmP_ = []  # new sub_patterns:
    rmP = int(dert_[0][2] > 0), 0, 0, 0, 0, 0, []  # pri_sr, Lr, Ir, Dr, Mr, rr, rdert_
    P[5] += 1  # r: dert_ recursion count = depth of P_ in dert_[-1]
    rng += 1

    for i in range(rng, L + 1):  # comp between rng-distant pixels, also bilateral, if L > rng * 2?
        ip, fd, fm = dert_[i]
        _ip, _fd, _fm = dert_[i - rng]
        ed = ip - _ip  # ed: element d, em: element m:
        if dderived:
            em = min(ip, _ip) - ave_m  # magnitude of vars derived from d corresponds to predictive value, thus direct match
        else:
            em = ave_d - abs(ed)  # magnitude of brightness has low correlation with stability, thus match is defined via d
        fd += ed
        fm += em  # accumulates difference and match between ip and all prior ips in extended rng
        dert_[i] = (ip, fd, fm)
        _fd += ed  # accumulates difference and match between ip and all prior and subsequent ips in extended rng
        _fm += em
        if i >= rng * 2:
            rmP, rmP_ = form_pattern_r(rmP, rmP_, _ip, _fd, _fm, dderived, rdn + 1, rng, i, L)
            #  forms rmPs: spans of pixels with same-sign _fm, eval for der+ and rng+ comp

    rmP, rmP_ = form_pattern(rmP, rmP_, _ip, _fd, _fm, dderived, rdn + 1, rng, i, L)  # terminate last incomplete P
    P[-1].append([rmP_])  # dert_ += [rmP_], or replace dert_?

    return P


def form_pattern_r(P, P_, pri_p, d, m, dderived, rdn, rng, x, X):  # P accumulation, termination, and initialization
    # or one generic form_pattern?

    s = m > 0  # sign, 0 is positive?  form_P -> pri mP ( sub_dP_:
    pri_s = P[0]
    dert_ = P[-1]  # depth of elements in e_ = r: depth of prior comp recursion within P

    if (x > rng * 2 and s != pri_s) or x == X:  # sign change: terminate and evaluate mP for sub-clustering and recursive comp

        if not pri_s:  # negative mP:
            P_.append((P))
            L, I, D, M, r, e_ = 0, 0, 0, 0, 0, []  # new P initialization

            if P[3] > ave_D:  # if D (high cross-sign-cancelled variation mP): recursive cross-comp(d), der+ fork
                P = der_comp(P, rng=1, rdn=1, Pr=1)  # includes P.e_ += [mdP_]

            elif -P[4] > ave_D: # if -M (high-variation mP): cluster by d sign: binary match, no value for cross-comp recursion
                dP_ = []
                dert_ = P[-1]  # list of element derts
                dP = int(dert_[0][1] > 0), 0, 0, 0, 0, 0, []  # pri_sd, Ld, Id, Dd, Md, rng, ed_
                L = P[1]
                for i in range(L + 1):
                    id, idd, imd = dert_[i]
                    sd = 1 if id > 0 else 0
                    if (dP[0] != sd and i > 0) or i == L:
                        dP, dP_ = form_pattern(dP, dP_, id, idd, imd, 0, i, L)  # ini=0: simple clustering by d sign
                dert_.append(dP_)  # P.e_ += [dP_]

            elif P[4] > ave_M:  # if M (high-match mP): recursive cross-comp(rng_p), rng+ fork
                P = rng_comp(P, rng=1, rdn=1, Pr=1, dderived=0)  # includes P.e_ += [rng_mP_]

        P_.append(P)   # terminated P output to second level; x == X-1 doesn't always terminate?
        s, L, I, D, M, r, dert_ = 0, 0, 0, 0, 0, 0, []
    else:
        pri_s, L, I, D, M, r, dert_ = P  # depth of derts in dert_ = r: depth of prior comp recursion within P

    pri_s = s  # current sign is stored as prior sign; P (span of pixels forming same-sign m | d) is incremented:
    L += 1  # length of mP | dP
    I += pri_p
    D += d
    M += m
    dert_ += [(pri_p, d, m)]
    P = pri_s, L, I, D, M, r, dert_

    return P, P_

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