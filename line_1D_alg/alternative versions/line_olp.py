import cv2
import argparse
from time import time
from collections import deque

''' 
This is a version of 1st level 1D alg with full overlap between difference patterns and match patterns, vs. exclusive dP | positive mP in line_POC. 
Also possible is additional filter for form_dP -> partial overlap | gap between positive mPs and dPs, but post-comp selection is better? 

Initial match is defined as average_abs(d) - abs(d): secondary to difference because a stable visual property of objects is albedo 
(vs. brightness), and spatio-temporal stability of albedo itself has low correlation with its magnitude. 
Although an indirect measure of match, low abs(d) should be predictive: uniformity across space correlates with stability over time.

Illumination is locally stable, so variation of albedo can be approximated as difference (vs. ratio) of brightness. 
Spatial difference in albedo indicates change in some property of observed objects, thus should serve as an edge | fill-in detector. 
Magnitude of such change or lack thereof is more predictive of some ultimate impact on observer than magnitude of original brightness. 
Hence, match of differences and matches is defined here as min magnitude, vs. inverse deviation of abs(d) as match of brightness.

Cross-comparison between consecutive pixels within horizontal scan line (row).
Resulting difference patterns dPs (spans of pixels forming same-sign differences)
and relative match patterns mPs (spans of pixels forming same-sign match)
are redundant representations of each line of pixels.

form_pattern() is conditionally recursive, cross-comparing derived variables within a queue e_ of a above-minimal summed value.
This recursion forms hierarchical mPs and dPs of variable depth, which will be cross-compared on the next level of search.
In the code below, postfix '_' denotes array name, vs. identical name of array elements, 
and capitalized variable names indicate sums of corresponding small-case vars'''


def form_pattern(typ, dderived, P, P_, pri_p, d, m, rdn, rng, x, X):  # accumulation, termination, and recursive comp within pattern mP | dP

    if typ: s = 1 if m >= 0 else 0  # sign of core var m, 0 is positive?
    else:   s = 1 if d >= 0 else 0  # sign of core var d, 0 is positive?

    pri_s, L, I, D, M, r, e_ = P  # depth of elements in e_ = r: flag of comp recursion within P
    if (x > rng * 2 and s != pri_s) or x == X-1:  # core var sign change, P is terminated and evaluated for recursive comp

        if typ:  # if typ==1: P is mP, if typ==0: P is dP
            if L > rng * 2 + 3 and pri_s == 1 and M > ave_M * rdn:  # comp range incr within e_= ders_, rdn: redundancy incr per recursion
                dP_= []; dP = 0,0,0,0,0,0,[]; mP_= []; mP = 0,0,0,0,0,0,[]  # sub- m_pattern initialization: pri_s, L, I, D, M, r, ders_
                r = 1
                rng += 1
                for i in range(rng, L):  # comp between extended-rng- distant pixels (ip) within e_
                    ip, fd, fm = e_[i]
                    _ip, _fd, _fm = e_[i - rng]
                    ed = ip - _ip  # ed: element d, em: element m:
                    if dderived:
                        em = min(ip, _ip) - ave_m  # magnitude of vars derived from d corresponds to predictive value, thus direct match
                    else:
                        em = ave_d - abs(ed)  # magnitude of brightness has low correlation with stability, thus match is defined through d
                    fd += ed
                    fm += em  # accumulates difference and match between ip and all prior ips in extended rng
                    e_[i] = (ip, fd, fm)
                    _fd += ed  # accumulates difference and match between ip and all prior and subsequent ips in extended rng
                    _fm += em
                    if i > rng * 2 - 1:
                        mP, mP_ = form_pattern(1, dderived, mP, mP_, _ip, _fd, _fm, rdn + 1, rng, i, L)  # mP: span of pixels with same-sign m
                        dP, dP_ = form_pattern(0, dderived, dP, dP_, _ip, _fd, _fm, rdn + 1, rng, i, L)  # dP: span of pixels with same-sign d
                e_ = (dP_, mP_)
        else:
            if L > 3 and abs(D) > ave_D * rdn:  # comp derivation increase within e_ = d_, no p, m comp or accumulation
                dP_= []; dP = 0,0,0,0,0,0,[]; mP_= []; mP = 0,0,0,0,0,0,[]  # sub- d_pattern initialization: pri_s, L, I, D, M, r, d_
                r = 1
                pri_d = e_[0]
                for i in range(1, L):  # comp between ds at rng=1, unilateral to preserve resolution, but missing back comp before rng incr?
                    ed = e_[i]
                    dd = ed - pri_d
                    md = min(d, pri_d) - ave_m  # d magnitude (change) has negative predictive value, thus direct match calculation
                    mP, mP_ = form_pattern(1, 1, mP, mP_, pri_d, dd, md, rdn+1, 1, i, L)  # forms mdP: span of ds with same-sign md
                    dP, dP_ = form_pattern(0, 1, dP, dP_, pri_d, dd, md, rdn+1, 1, i, L)  # forms ddP: span of ds with same-sign dd
                    pri_d = d
                e_= (dP_, mP_)

        P_.append((typ, pri_s, L, I, D, M, r, e_))  # terminated P is output to the next level of search
        L, I, D, M, r, e_ = 0, 0, 0, 0, 0, []  # new P initialization

    pri_s = s   # current sign is stored as prior sign; P (span of pixels forming same-sign m | d) is incremented:
    L += 1      # length of mP | dP
    I += pri_p  # input ps summed within mP | dP
    D += d      # fuzzy ds summed within mP | dP
    M += m      # fuzzy ms summed within mP | dP
    if typ:
        e_.append((pri_p, d, m))  # inputs for extended-range comp are tuples, vs. pixels for initial comp
    else:
        e_.append(abs(d))  # inputs for higher-derivation comp are ds (vs. pixels), p and m are not used?

    P = pri_s, L, I, D, M, r, e_
    return P, P_


def cross_comp(frame_of_pixels_):  # postfix '_' denotes array name, vs. identical name of its elements
    frame_of_patterns_ = []  # output frame of mPs: match patterns, and dPs: difference patterns

    for y in range(ini_y, Y):
        pixel_ = frame_of_pixels_[y, :]  # y is index of new line pixel_

        dP_= []; dP = 0,0,0,0,0,0,[]  # initialized at each line,
        mP_= []; mP = 0,0,0,0,0,0,[]  # pri_s, L, I, D, M, r, ders_
        max_index = min_rng - 1  # max index of rng_ders_
        ders_ = deque(maxlen=min_rng)  # array of incomplete ders, within rng from input pixel: summation range < rng
        ders_.append((0, 0, 0))  # prior tuple, no d, m at x = 0

        for x, p in enumerate(pixel_):  # pixel p is compared to rng of prior pixels in horizontal line, summing d and m per prior pixel
            back_fd, back_fm = 0, 0
            for index, (pri_p, fd, fm) in enumerate(ders_):

                d = p - pri_p
                m = ave_d - abs(d)
                fd += d  # bilateral fuzzy d: running sum of differences between pixel and all prior and subsequent pixels within rng
                fm += m  # bilateral fuzzy m: running sum of matches between pixel and all prior and subsequent pixels within rng
                back_fd += d
                back_fm += m  # running sum of d and m between pixel and all prior pixels within rng

                if index < max_index:
                    ders_[index] = (pri_p, fd, fm)

                elif x > min_rng * 2 - 1:   # after pri_p comp over full bilateral rng
                    mP, mP_ = form_pattern(1, 0, mP, mP_, pri_p, fd, fm, 1, min_rng, x, X)  # forms mP: span of pixels with same-sign m
                    dP, dP_ = form_pattern(0, 0, dP, dP_, pri_p, fd, fm, 1, min_rng, x, X)  # forms dP: span of pixels with same-sign d

            ders_.appendleft((p, back_fd, back_fm)) # new tuple with initialized d and m, maxlen displaces completed tuple from rng_t_
        frame_of_patterns_ += [(dP_, mP_)]  # line of patterns is added to frame of patterns, last incomplete ders are discarded
    return frame_of_patterns_  # frame of patterns is output to level 2


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/raccoon.jpg')
arguments = vars(argument_parser.parse_args())
image = cv2.imread(arguments['image'], 0).astype(int)

# the same image can be loaded online, without cv2:
# from scipy import misc
# f = misc.face(gray=True)  # load pix-mapped image
# f = f.astype(int)

# pattern filters are initialized here as constants, eventually adjusted by higher-level feedback:

ave_m = 10  # min d-match for inclusion in positive d_mP
ave_d = 20  # |difference| between pixels that coincides with average value of mP - redundancy to overlapping dPs
ave_M = 127  # min M for initial incremental-range comparison(t_)
ave_D = 127  # min |D| for initial incremental-derivation comparison(d_)
ini_y = 400
min_rng = 2  # fuzzy pixel comparison range, adjusted by higher-level feedback
Y, X = image.shape  # Y: frame height, X: frame width

start_time = time()
frame_of_patterns_ = cross_comp(image)
end_time = time() - start_time
print(end_time)