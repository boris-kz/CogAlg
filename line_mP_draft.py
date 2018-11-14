import cv2
import argparse
from time import time
from collections import deque

''' 
1D version of core algorithm with match value = abs(d) - ave abs(d). 
This match is secondary to difference because a stable visual property of objects is albedo (vs. brightness), 
and stability of albedo has low correlation with its intensity. 
Illumination is locally stable, so variation of albedo can be approximated as difference (vs. ratio) of brightness.

Cross-comparison between consecutive pixels within horizontal scan line (row).
Resulting difference patterns dPs (spans of pixels forming same-sign differences)
and relative match patterns mPs (spans of pixels forming same-sign match)
are redundant representations of each line of pixels.

The same comp() will cross-compare derived variables, to the extent that they are predictive.
For example, if differences between pixels turn out to be more predictive than value of these pixels, 
then all differences will be cross-compared by secondary comp(d), forming d_mPs and d_dPs.
These secondary patterns will be evaluated for further internal recursion and cross-compared on the next level.

In my code below, postfix '_' denotes array name, vs. identical name of array elements '''


def recursive_comp(x, p, pri_p, d, m, dP, mP, dP_, mP_, redun, rng):

    # incremental-range comp within mPs or incremental-derivation comp within dPs,
    # called from pre_recursive_comp(), which is called from form_pattern
    # comp(d) is different: md = min?

    d += p - pri_p  # fuzzy d accumulates differences between p and all prior and subsequent ps in extended rng
    m += abs(d) - ave  # fuzzy m accumulates deviation of match within bilateral extended rng

    dP, dP_ = form_pattern(0, dP, dP_, pri_p, d, m, x, redun, rng)
    mP, mP_ = form_pattern(1, mP, mP_, pri_p, d, m, x, redun, rng)

    # form difference pattern dP: span of pixels with same-sign d, or match pattern mP: span of pixels with same-sign m
    return d, m, dP, mP, dP_, mP_  # for next-p comp, dP and mP increment, no pri_m, pri_d for next recursive_comp: both are cumulative?


def pre_recursive_comp(typ, element_, redun, rng):  # pre-processing for comp recursion over elements of selected pattern

    re_X = len(element_) - 1
    dP_, mP_ = [], []
    dP = 0, 0, 0, 0, 0, 0, []  # pri_s, I, D, M, Alt, recomp, d_
    mP = 0, 0, 0, 0, 0, 0, []  # pri_s, I, D, M, Alt, recomp, ders_

    if typ: # comparison range increment within element_ = ders_ of mP

        for x in range(rng, re_X):
            p = element_[x][0]  # accumulation of pri_p, d, m with d and m from comp of rng-distant pixels:
            pri_p, d, m = element_[x-rng]
            d, m, dP, mP, dP_, mP_ = recursive_comp(x, p, pri_p, d, m, dP, mP, dP_, mP_, redun, rng)

    else: # comparison derivation increment within element_ = d_ of dP:
        pri_p = element_[0]
        d, m = 0, 0
        for x in range(1, re_X):
            p = element_[x]
            d, m, dP, vP, dP_, mP_ = recursive_comp(x, p, pri_p, d, m, dP, mP, dP_, mP_, redun, rng)
            pri_p = p

    return dP_, mP_  # tuple of local (dP_, mP_), indicated by recomp == 1, replaces ders_ or d_


def form_pattern(typ, P, P_, pri_p, d, m, x, redun, rng):  # accumulation, termination, recursive comp in P: mP | dP

    if typ: s = 1 if m >= 0 else 0  # sign of d, 0 is positive?
    else:   s = 1 if d >= 0 else 0  # sign of m, 0 is positive?

    pri_s, I, D, M, Alt, recomp, element_ = P  # e_: type of elements in P depends on the level of comp recursion
    if x > rng + 2 and (s != pri_s or x == X - 1):  # P is terminated and evaluated for recursion

        if typ:
            if len(element_) > rng + 3 and pri_s == 1 and M > ave_M * redun:
                recomp = 1  # flag for recursive comp range increase within e_ = ders_:
                element_.append(pre_recursive_comp(1, element_, redun+1, rng+1))
        else:
            if len(element_) > 3 and abs(D) > ave_D * redun:
                recomp = 1  # flag for recursive derivation increase within e_ = d_:
                element_.append(pre_recursive_comp(0, element_, redun+1, 1))

        P = typ, pri_s, I, D, M, Alt, recomp, element_
        P_.append(P)  # output to second level

        I, D, M, Alt, recomp, element_ = 0, 0, 0, 0, 0, []  # initialized P

    pri_s = s   # current sign is stored as prior sign; P (span of pixels forming same-sign m | d) is incremented:
    I += pri_p  # input ps summed within mP | dP
    D += d      # fuzzy ds summed within mP | dP
    M += m      # fuzzy ms summed within mP | dP

    if typ:
        Alt += abs(d)  # estimated value of alternative-type Ps, to compute redundancy for next_level eval(P)
        element_.append((pri_p, d, m))  # inputs for greater rng comp are tuples, vs. pixels for initial comp
    else:
        Alt += abs(m)
        element_.append(d)  # prior ds of the same sign are buffered within dP

    P = pri_s, I, D, M, Alt, recomp, element_
    return P, P_


def frame(frame_of_pixels_):  # postfix '_' denotes array name, vs. identical name of its elements
    frame_of_patterns_ = []  # output frame of vPs: relative-match patterns, and dPs: difference patterns

    for y in range(Y):
        pixel_ = frame_of_pixels_[y, :]  # y is index of new line p_

        dP_, mP_ = [], []  # initialized at each line
        dP = 0, 0, 0, 0, 0, 0, []  # pri_s, I, D, M, Alt, recomp, d_
        mP = 0, 0, 0, 0, 0, 0, []  # pri_s, I, D, M, Alt, recomp, ders_
        x = 0
        ders_ = deque(maxlen=min_rng)  # array of incomplete ders, within rng from input pixel: summation range < rng
        max_index = min_rng - 1  # max index of rng_ders_
        pri_d, pri_m = 0, 0  # fuzzy derivatives in prior completed tuple

        for p in pixel_:  # pixel p is compared to rng of prior pixels within horizontal line, summing d and m per prior pixel
            x += 1
            for index, (pri_p, d, m) in enumerate(ders_):

                d += p - pri_p  # fuzzy d: running sum of differences between pixel and all subsequent pixels within rng
                m += abs(d) - ave  # fuzzy m: running sum of matches between pixel and all subsequent pixels within rng

                if index < max_index:
                    ders_[index] = (pri_p, d, m)

                elif x > min_rng * 2 - 1:
                    d += pri_d; m += pri_m  # d and m are accumulated over full bilateral min_rng: before and after pri_p

                    dP, dP_ = form_pattern(0, dP, dP_, pri_p, d, m, x, 1, min_rng)  # forms diff. pattern dP: span of pixels with same-sign d
                    mP, mP_ = form_pattern(1, mP, mP_, pri_p, d, m, x, 1, min_rng)  # forms match pattern mP: span of pixels with same-sign m

                    pri_d = d; pri_m = m
                ders_.appendleft((p, 0, 0))  # new tuple with initialized d and m, maxlen displaces completed tuple from rng_t_

        frame_of_patterns_.append((dP_, mP_))  # line of patterns is added to frame of patterns, last incomplete ders are discarded

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

min_rng = 3  # fuzzy pixel comparison range, initialized here but eventually a higher-level feedback
ave = 63  # |difference| between pixels that coincides with average value of mP - redundancy to overlapping dPs
ave_M = 127  # min M for initial incremental-range comparison(t_)
ave_D = 127  # min |D| for initial incremental-derivation comparison(d_)

Y, X = image.shape  # Y: frame height, X: frame width

start_time = time()
frame_of_patterns_ = frame(image)
end_time = time() - start_time
print(end_time)

