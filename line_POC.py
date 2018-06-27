import cv2
import argparse
from time import time
from collections import deque

''' core algorithm level 1: 1D-only proof of concept, 
applied here to process lines of grey-scale pixels but not effective for recognition of 2D images. 

Cross-comparison between consecutive pixels within horizontal scan line (row).
Resulting difference patterns dPs (spans of pixels forming same-sign differences)
and relative match patterns vPs (spans of pixels forming same-sign predictive value)
are redundant representations of each line of pixels.

postfix '_' denotes array name, vs. identical name of array elements '''


def recursive_comparison(x, p, pri_p, d, v, pri_d, pri_m, dP, vP, dP_, vP_, X, redun, rng):

    # incremental-range comp within vPs or incremental-derivation comp within dPs,
    # called from pre_recursive_comp(), which is called from form_pattern

    d += p - pri_p  # fuzzy d accumulates differences between p and all prior and subsequent ps in extended rng
    m = min(p, pri_p)
    v += m + pri_m - abs(d + pri_d) /4 - ave *2  # fuzzy v accumulates deviation of match within bilateral extended rng

    dP, dP_ = form_pattern(0, dP, dP_, pri_p, d, v, x, X, redun, rng)
    vP, vP_ = form_pattern(1, vP, vP_, pri_p, d, v, x, X, redun, rng)

    # forms difference pattern dP: span of pixels with same-sign d, or value pattern vP: span of pixels with same-sign v
    pri_d = d; pri_m = m  # for next recursive_comp

    return pri_d, pri_m, d, v, dP, vP, dP_, vP_  # for next-p comp, dP and vP increment, output


def pre_recursive_comp(typ, element_, redun, rng):  # pre-processing for comp recursion over elements of selected pattern

    X = len(element_) - 1
    dP_, vP_ = [], []
    dP = 0, 0, 0, 0, 0, 0, []  # pri_s, I, D, V, Alt, recomp, d_
    vP = 0, 0, 0, 0, 0, 0, []  # pri_s, I, D, V, Alt, recomp, ders_
    pri_d, pri_m = 0, 0

    if typ: # comparison range increment within element_ = ders_ of vP

        for x in range(rng, X):
            p = element_[x][0]  # accumulation of pri_p, d, v with d and v from comp of rng-distant pixels:
            pri_p, d, v = element_[x-rng]
            pri_d, pri_m, d, v, dP, vP, dP_, vP_ = recursive_comparison(x, p, pri_p, d, v, pri_d, pri_m, dP, vP, dP_, vP_, X, redun, rng)

    else: # comparison derivation increment within element_ = d_ of dP:
        pri_p = element_[0]
        d, v = 0, 0
        for x in range(1, X):
            p = element_[x]
            pri_d, pri_m, d, v, dP, vP, dP_, vP_ = recursive_comparison(x, p, pri_p, d, v, pri_d, pri_m, dP, vP, dP_, vP_, X, redun, rng)
            pri_p = p

    return dP_, vP_  # tuple of local (dP_, vP_), indicated by recomp == 1, replaces ders_ or d_


def form_pattern(typ, P, P_, pri_p, d, v, x, X, redun, rng):  # accumulation, termination, recursive comp in P: vP | dP

    if typ: s = 1 if v >= 0 else 0  # sign of d, 0 is positive?
    else:   s = 1 if d >= 0 else 0  # sign of v, 0 is positive?

    pri_s, I, D, V, Alt, recomp, element_ = P  # e_: type of elements in P depends on the level of comp recursion
    if x > rng + 2 and (s != pri_s or x == X - 1):  # P is terminated and evaluated for recursion

        if typ:
            if len(element_) > rng + 3 and pri_s == 1 and V > ave_V * redun:
                recomp = 1  # flag for recursive comp range increase within e_ = ders_:
                element_.append(pre_recursive_comp(1, element_, redun+1, rng+1))
        else:
            if len(element_) > 3 and abs(D) > ave_D * redun:
                recomp = 1  # flag for recursive derivation increase within e_ = d_:
                element_.append(pre_recursive_comp(0, element_, redun+1, 1))

        P = typ, pri_s, I, D, V, Alt, recomp, element_
        P_.append(P)  # output to second level

        I, D, V, Alt, recomp, element_ = 0, 0, 0, 0, 0, []  # initialized P

    pri_s = s   # current sign is stored as prior sign; P (span of pixels forming same-sign v | d) is incremented:
    I += pri_p  # input ps summed within vP | dP
    D += d      # fuzzy ds summed within vP | dP
    V += v      # fuzzy vs summed within vP | dP

    if typ:
        Alt += abs(d)  # estimated value of alternative-type Ps, to compute redundancy on the next level
        element_.append((pri_p, d, v))  # inputs for greater rng comp are tuples, vs. pixels for initial comp
    else:
        Alt += abs(v)
        element_.append(d)  # prior ds of the same sign are buffered within dP

    P = pri_s, I, D, V, Alt, recomp, element_
    return P, P_


def comparison(x, p, pri_d, pri_m, rng_ders_, dP, vP, dP_, vP_, X):  # pixel is compared to rng prior pixels

    max_index = min_rng - 1
    for index, (pri_p, d, m) in enumerate(rng_ders_):

        d += p - pri_p  # fuzzy d: running sum of differences between pixel and all subsequent pixels within min_rng
        m += min(p, pri_p)  # fuzzy m: running sum of matches between pixel and all subsequent pixels within min_rng

        if index < max_index:
            rng_ders_[index] = (pri_p, d, m)

        elif x > min_rng * 2 - 1:  # ders are accumulated over full bilateral rng: before and rng after displaced pixel

            v = (m + pri_m) - abs(d + pri_d) /4 - ave * min_rng *2  # m - abs(d)/4: bilateral projected match is reduced by neg d/2
            # predictive value of match, sign determines inclusion into positive | negative vP

            # completed tuple (pri_p, d, v) of summation range = rng (maxlen in rng_t_) transferred to form_pattern,
            # to form difference pattern dP: span of pixels with same-sign d, or value pattern vP: span of pixels with same-sign v:

            dP, dP_ = form_pattern(0, dP, dP_, pri_p, d, v, x, X, 1, min_rng)
            vP, vP_ = form_pattern(1, vP, vP_, pri_p, d, v, x, X, 1, min_rng)

    next_d = d; next_m = m  # full-rng fuzzy d and m, to become pri_d, pri_m for pixel displaced by next comparison()
    rng_ders_.appendleft((p, 0, 0))  # new tuple is added, maxlen parameter removes completed tuple in deque

    return next_d, next_m, rng_ders_, dP, vP, dP_, vP_  # for next-p comparison, vP and dP increment, output


def frame(frame_of_pixels_):  # postfix '_' denotes array name, vs. identical name of its elements

    frame_of_patterns_ = []  # output frame of vPs: relative-match patterns, and dPs: difference patterns
    Y, X = frame_of_pixels_.shape  # Y: frame height, X: frame width

    for y in range(Y):
        pixel_ = frame_of_pixels_[y, :]   # y is index of new line p_

        dP_, vP_ = [], []  # initialized at each level
        dP = 0, 0, 0, 0, 0, 0, []  # pri_s, I, D, V, Alt, recomp, d_
        vP = 0, 0, 0, 0, 0, 0, []  # pri_s, I, D, V, Alt, recomp, ders_

        rng_ders_ = deque(maxlen=min_rng)  # tuple of incomplete fuzzy derivatives: summation range < rng
        rng_ders_.append((pixel_[0], 0, 0))  # prior tuple, no d, m at x = 0
        pri_d, pri_m = 0, 0  # fuzzy d and m over rng before prior p

        for x in range(1, X):  # cross-compares consecutive pixels within each line

            p = pixel_[x]  # new pixel, for fuzzy comparison to rng_ders_:
            pri_d, pri_m, rng_ders_, dP, vP, dP_, vP_ = comparison(x, p, pri_d, pri_m, rng_ders_, dP, vP, dP_, vP_, X)

        # line ends, last rng of incomplete ders is discarded
        frame_of_patterns_.append((dP_, vP_))  # line of patterns is added to frame of patterns

    return frame_of_patterns_  # frame of patterns is output to level 2

# from scipy import misc
# f = misc.face(gray=True)  # input frame of pixels
# f = f.astype(int)

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/racoon.jpg')
arguments = vars(argument_parser.parse_args())
image = cv2.imread(arguments['image'], 0).astype(int)

# pattern filters: eventually from higher-level feedback, initialized here as constants:

min_rng = 3  # fuzzy pixel comparison range, initialized here but eventually a higher-level feedback
ave = 95  # average match between pixels, minimal for inclusion into positive vP
ave_V = 127  # min V for initial incremental-range comparison(t_)
ave_D = 127  # min |D| for initial incremental-derivation comparison(d_)

start_time = time()
frame_of_patterns_ = frame(image)
end_time = time() - start_time
print(end_time)

