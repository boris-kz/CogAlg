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

- Negative mPs are evaluated for two forks: 
  - sub-clustering into direction patterns dPs: representations of spans of pixels forming same-sign differences, and
  - internal cross-comp of element ds, which forms mdPs: spans of ds forming same-sign md = min d (match is defined directly 
    because predictive value of difference is proportional to its magnitude, although inversely so).
    This fork is recursive: each mdP is processed the same way as initial mPs.

- Positive mPs: spans of pixels forming positive match, are evaluated for cross-comp of pixel | input in dert over incremented range 
  (positive match means that pixels have high predictive value, thus likely match to more distant pixels). 
  Subsequent sub-clustering in selected +mPs forms rng_mPs, which are also processed the same way as initial mPs, recursively. 
  '''

def cross_comp(frame_of_pixels_):  # postfix '_' denotes array name, vs. identical name of its elements

    frame_of_patterns_ = []  # output frame of mPs: nested match patterns
    # top-layer mPs may contain dPs: direction patterns | rng_mPs: mPs from rng+ comp, etc., formed by recursive form_pattern

    for y in range(ini_y + 1, Y):
        pixel_ = frame_of_pixels_[y, :]  # y is index of new line pixel_
        P_ = []; P = 0, 0, 0, 0, 0, 0, []  # pri_s, L, I, D, M, r, dert_ # initialized at each line
        max_index = min_rng - 1  # max index of rng_dert_
        dert_ = deque(maxlen=min_rng)  # array of incomplete dert, within rng from input pixel: summation range < rng
        dert_.append((0, 0, 0))  # prior tuple, no d, m at x = 0

        for x, p in enumerate(pixel_):  # pixel p is compared to rng prior pixels in horizontal row, summing d and m per prior pixel
            back_fd, back_fm = 0, 0
            for index, (pri_p, fd, fm) in enumerate(dert_):

                d = p - pri_p
                m = ave_d - abs(d)  # initial match is defined as inverse deviation of |difference|
                fd += d   # bilateral fuzzy d: running sum of differences between pixel and all prior and subsequent pixels in rng
                fm += m   # bilateral fuzzy m: running sum of matches between pixel and all prior and subsequent pixels in rng
                back_fd += d
                back_fm += m  # running sum of d and m between pixel and all prior pixels in rng

                if index < max_index:
                    dert_[index] = (pri_p, fd, fm)

                elif x > min_rng * 2 - 1:
                    P, P_ = form_pattern(P, P_, pri_p, fd, fm, min_rng, x, X)  # forms mPs: spans of pixels with same-sign m

            dert_.appendleft((p, back_fd, back_fm))  # p + initialized d and m, maxlen displaces completed tuple from rng_t_
        frame_of_patterns_ += [P_]  # line of patterns is added to frame of patterns, last incomplete dert are discarded
    return frame_of_patterns_  # frame of patterns is output to level 2

def form_pattern(P, P_, pri_p, d, m, rng, x, X):  # accumulation, termination, recursion eval

    s = m > 0  # sign
    if (x > rng * 2 and s != P[0]) or x == X:  # sign change, mP is terminated, new mP is initialized
        # put cluster_d and recursion eval here
        P_.append(P)
        s, L, I, D, M, e_ = 0, 0, 0, 0, 0, []
    else:
        pri_s, L, I, D, M, e_ = P  # depth of elements in e_ = r: depth of prior comp recursion within P

    L += 1; I += pri_p; D += d; M += m; e_.append((pri_p, d, m))
    P = s, L, I, D, M, e_

    return P, P_

def form_pattern_r(dderived, P, P_, pri_p, d, m, rdn, rng, x, X):  # accumulation and termination
    # cross-compare dert input param within e_ of above-min length and M | D

    s = m > 0  # sign, 0 is positive?  form_P -> pri mP ( sub_dP_:
    pri_s, L, I, D, M, r, e_ = P  # depth of elements in e_ = r: depth of prior comp recursion within P

    if (x > rng * 2 and s != pri_s) or x == X:  # sign change, mP is terminated and evaluated for sub-clustering and recursive comp

        if not pri_s:  # negative mP:
            # cluster_d: eval by summed d: e_ -> dP_ by d sign match, for comp_P only,
            # or comp_d: eval by summed |d|? 
            dP_ = []
            dP = int(e_[0][1] > 0), 0, 0, 0, 0, 0, []  # pri_s, L, I, D, M, r, e_
            e_.append((0, 0, 0))

            for i in range(L + 1):
                ip, id, im = e_[i]
                sd = 1 if id > 0 else 0
                pri_sd, Ld, Id, Dd, Md, rd, ed_ = dP

                if (pri_sd != sd and i > 0) or i == L:
                    if Ld > rng * 2 and Dd > ave_M * rdn:  # comp range increase within e_, rdn (redundancy) is incremented per comp recursion
                        rd = 1
                        mdP_ = []
                        mdP = 0, 0, 0, 0, 0, 0, []  # pri_s, L, I, D, M, r, e_;  no Alt: M is defined through abs(d)
                        ed_.append((0, 0, 0))
                        fdd, fmd = 0, 0

                        for j in range(1, Ld + 1):  # bilateral comp between consecutive dj s
                            dj = ed_[j][1]
                            _dj = ed_[j - 1][1]
                            dd = dj - _dj  # comparison between consecutive differences
                            md = min(dj, _dj) - ave_m  # evaluation of md (min d: magnitudes derived from d correspond to predictive value)
                            fdd += dd  # bilateral difference and match between ds
                            fmd += md
                            if j > 1:
                                mdP, mdP_ = form_pattern_r(1, mdP, mdP_, _dj, fdd, fmd, rdn + 1, rng, j, Ld)  # dderived = 1, m=min
                                # evaluation for deeper recursion within mdPs: patterns defined by match between ds, but overlapping form_P?
                            fdd = dd
                            fmd = md

                        ed_ = mdP_  # dert are replaced with mdPs: spans of pixels that form same-sign md
                    dP_.append((pri_sd, Ld, Id, Dd, Md, rd, ed_))
                    sd, Ld, Id, Dd, Md, dr, de_ = 0, 0, 0, 0, 0, 0, []

                Ld += 1; Id += ip; Dd += id; Md += im; ed_.append((ip, id, im))
                dP = sd, Ld, Id, Dd, Md, rd, ed_

            e_ = e_, dP_
            rng += 1

            # if L > rng * 2 and -M > ave_M * rdn:  # if strong negative mP: comp_deriv(): derivation increase within e_,
            # dd may match across d sign change, eval by mag only, d ma regardless of sign:

        if L > rng * 2 and M > ave_M * rdn:  # if strong positive mP: comp_range(): rng increase within e_, no parallel der+
            # rdn (redundancy) is incremented per comp recursion
            r = 1  # recursion count
            rng += 1
            rng_mP_ = []
            rng_mP = 0, 0, 0, 0, 0, 0, []  # pri_s, L, I, D, M, r, e_;  no Alt: M is defined through abs(d)
            er_ = [(0, 0, 0)]

            for i in range(rng, L + 1):  # comp between rng-distant pixels, also bilateral, if L > rng * 2?
                ip, fd, fm = er_[i]  # new er_ =
                _ip, _fd, _fm = er_[i - rng]
                ed = ip - _ip  # ed: element d, em: element m:
                if dderived:
                    em = min(ip, _ip) - ave_m  # magnitude of vars derived from d corresponds to predictive value, thus direct match
                else:
                    em = ave_d - abs(ed)  # magnitude of brightness has low correlation with stability, thus match is defined via d
                fd += ed
                fm += em  # accumulates difference and match between ip and all prior ips in extended rng
                er_[i] = (ip, fd, fm)
                _fd += ed  # accumulates difference and match between ip and all prior and subsequent ips in extended rng
                _fm += em
                if i >= rng * 2:
                    rng_mP, rng__mP_ = form_pattern_r(dderived, rng_mP, rng_mP_, _ip, _fd, _fm, rdn + 1, rng, i, L)

            er_ = rng_mP_  # dert replaced with sub_mPs: spans of pixels that form same-sign m


        P_.append((pri_s, L, I, D, M, r, e_))  # terminated P output to second level; x == X-1 doesn't always terminate?
        L, I, D, M, r, e_ = 0, 0, 0, 0, 0, []  # new P initialization

    pri_s = s  # current sign is stored as prior sign; P (span of pixels forming same-sign m | d) is incremented:
    L += 1  # length of mP | dP
    I += pri_p  # input ps summed within mP | dP
    D += d  # fuzzy ds summed within mP | dP
    M += m  # fuzzy ms summed within mP | dP
    e_ += [(pri_p, d, m)]  # recursive comp over tuples vs. pixels, dP' p, m ignore: no accum but buffer: no copy in mP?

    P = pri_s, L, I, D, M, r, e_
    return P, P_

# pattern filters: eventually from higher-level feedback, initialized here as constants:

ave_m = 10  # min dm for positive dmP
ave_d = 20  # |difference| between pixels that coincides with average value of mP - redundancy to overlapping dPs
ave_M = 127  # min M for initial incremental-range comparison(t_)
ave_D = 127  # min |D| for initial incremental-derivation comparison(d_)
ini_y = 0
min_rng = 2  # fuzzy pixel comparison range, initialized here but eventually adjusted by higher-level feedback
Y, X = image.shape  # Y: frame height, X: frame width

start_time = time()
frame_of_patterns_ = cross_comp(image)
end_time = time() - start_time
print(end_time)

'''
Next level cross-compares resulting patterns Ps (s, L, I, D, M, r, nested e_) and evaluates them for deeper cross-comparison. 
The depth of cross-comparison (discontinuous if generic) is increased in lower-recursion e_, then between same-recursion e_s:
comp (s)?  # same-sign only
    comp (L, I, D, M)?  # in parallel or L first, equal-weight or I is redundant?  
        comp (r)?  # same-recursion (derivation) order e_
            cross_comp (e_)
'''