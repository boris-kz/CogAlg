import cv2
import argparse
from time import time
from collections import deque

''' 
Updated 1D version of core algorithm, with match = ave abs(d) - abs(d). Match is secondary to difference because a stable 
visual property of objects is albedo (vs. brightness), and stability of albedo has low correlation with its value. 
Although indirect measure of match, low abs(d) is still predictive: uniformity across space correlates with stability over time.
Illumination is locally stable, so variation of albedo can be approximated as difference (vs. ratio) of brightness.

Cross-comparison between consecutive pixels within horizontal scan line (row).
Resulting difference patterns dPs (spans of pixels forming same-sign differences)
and relative match patterns mPs (spans of pixels forming same-sign match)
are redundant representations of each line of pixels.

Recursive_comp() cross-compares derived variables, within a queue of a above- minimal length and summed value.
For example, if differences between pixels turn out to be more predictive than value of these pixels, 
then all differences will be cross-compared by secondary comp(d), forming d_mPs and d_dPs.
These secondary patterns will be evaluated for further internal recursion after cross-comparison on the next level.

In the code below, postfix '_' denotes array name, vs. identical name of array elements '''


def form_pattern(typ, P, P_, pri_p, d, m, rdn, rng, x, X):  # accumulation, termination, and recursive comp within pattern mP | dP

    if typ: s = 1 if m >= 0 else 0  # sign of core var m, 0 is positive?
    else:   s = 1 if d >= 0 else 0  # sign of core var d, 0 is positive?

    pri_s, L, I, D, M, r, e_ = P  # depth of elements in e_ = r: depth of comp recursion within P
    if (x > rng * 2 and s != pri_s) or x == X-1:  # core var sign change, P is terminated and evaluated for recursive comp

        dP_, mP_ = [], []
        dP = 0, 0, 0, 0, 0, 0, []  # pri_s, L, I, D, M, r, d_;  no Alt: M in both is already defined through abs(d)
        mP = 0, 0, 0, 0, 0, 0, []  # pri_s, L, I, D, M, r, ders_

        if typ:
            if L > rng + 3 and pri_s == 1 and M > ave_M * rdn:  # comp range increase within e_ = ders_:
                r = 1                                   # rdn: redundancy, incremented per comp recursion
                rng += 1
                for i in range(rng, L):  # comp between rng-distant pixels, also bilateral, if L > rng * 2?
                    ip = e_[i][0];  pri_ip, i_d, i_m = e_[i-rng]

                    i_d += ip - pri_ip  # accumulates difference between p and all prior and subsequent ps in extended rng
                    i_m += ave - abs(i_d)  # accumulates match within extended rng, no discrete buffer?

                    if i_m >= 0:
                        mP, mP_ = form_pattern(1, mP, mP_, pri_ip, i_d, i_m, rdn+1, rng, i, L)  # forms mP: span of pixels with same-sign m
                    else:
                        dP, dP_ = form_pattern(0, dP, dP_, pri_ip, i_d, i_m, rdn+1, rng, i, L)  # forms dP: span of pixels with same-sign d
                e_= (dP_, mP_)
        else:
            if L > 3 and abs(D) > ave_D * rdn:  # comp derivation increase within e_ = d_:
                r = 1
                pri_ip = e_[0]
                for i in range(1, L):  # comp between consecutive ip = d, bilateral?
                    ip = e_[i]
                    i_d = ip - pri_ip  # one-to-one comp, no accumulation till recursion?
                    i_m = min(ip, pri_ip) - ave  # d is a proxy of change, thus direct match, immediate eval, separate ave?

                    mP, mP_ = form_pattern(1, mP, mP_, pri_ip, i_d, i_m, rdn+1, 1, i, L)  # forms mP: span of pixels with same-sign m
                    dP, dP_ = form_pattern(0, dP, dP_, pri_ip, i_d, i_m, rdn+1, 1, i, L)  # forms dP: span of pixels with same-sign d
                    pri_ip = ip
                e_= (dP_, mP_)

        P_.append((typ, pri_s, L, I, D, M, r, e_))  # terminated P output to second level; doesn't always terminate, ?
        L, I, D, M, r, e_ = 0, 0, 0, 0, 0, []  # new P initialization

    pri_s = s   # current sign is stored as prior sign; P (span of pixels forming same-sign m | d) is incremented:
    L += 1      # length of mP | dP
    I += pri_p  # input ps summed within mP | dP
    D += d      # fuzzy ds summed within mP | dP
    M += m      # fuzzy ms summed within mP | dP

    if typ:
        e_.append((pri_p, d, m))  # inputs for greater rng comp are tuples, vs. pixels for initial comp
    else: e_.append(d)  # prior ds of the same sign are buffered within dP

    P = pri_s, L, I, D, M, r, e_
    return P, P_


def cross_comp(frame_of_pixels_):  # postfix '_' denotes array name, vs. identical name of its elements

    frame_of_patterns_ = []  # output frame of mPs: match patterns, and dPs: difference patterns
    for y in range(ini_y +1, Y):

        pixel_ = frame_of_pixels_[y, :]  # y is index of new line pixel_
        dP_, mP_ = [], []  # initialized at each line
        dP = 0, 0, 0, 0, 0, 0, []  # pri_s, L, I, D, M, r, d_
        mP = 0, 0, 0, 0, 0, 0, []  # pri_s, L, I, D, M, r, ders_

        max_index = min_rng - 1  # max index of rng_ders_
        ders_ = deque(maxlen=min_rng)  # array of incomplete ders, within rng from input pixel: summation range < rng
        ders_.append((0, 0, 0))  # prior tuple, no d, m at x = 0
        back_d, back_m = 0, 0  # fuzzy derivatives from rng of backward comps per prior pixel

        for x, p in enumerate(pixel_):  # pixel p is compared to rng of prior pixels in horizontal line, summing d and m per prior pixel
            for index, (pri_p, d, m) in enumerate(ders_):

                d += p - pri_p  # fuzzy d: running sum of differences between pixel and all subsequent pixels within rng
                m += ave - abs(d)  # fuzzy m: running sum of matches between pixel and all subsequent pixels within rng

                if index < max_index:
                    ders_[index] = (pri_p, d, m)

                elif x > min_rng * 2 - 1:
                    bi_d = d + back_d; back_d = d  # d and m are accumulated over full bilateral (before and after pri_p) min_rng
                    bi_m = m + back_m; back_m = m

                    mP, mP_ = form_pattern(1, mP, mP_, pri_p, bi_d, bi_m, 1, min_rng, x, X)  # forms mP: span of pixels with same-sign m
                    dP, dP_ = form_pattern(0, dP, dP_, pri_p, bi_d, bi_m, 1, min_rng, x, X)  # forms dP: span of pixels with same-sign d

            ders_.appendleft((p, 0, 0))  # new tuple with initialized d and m, maxlen displaces completed tuple from rng_t_
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

# pattern filters: eventually from higher-level feedback, initialized here as constants:

min_rng = 2  # fuzzy pixel comparison range, initialized here but eventually a higher-level feedback
ave = 31  # |difference| between pixels that coincides with average value of mP - redundancy to overlapping dPs
ave_M = 127  # min M for initial incremental-range comparison(t_)
ave_D = 127  # min |D| for initial incremental-derivation comparison(d_)
ini_y = 400
Y, X = image.shape  # Y: frame height, X: frame width

start_time = time()
frame_of_patterns_ = cross_comp(image)
end_time = time() - start_time
print(end_time)

