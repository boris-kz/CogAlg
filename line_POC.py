import cv2
import argparse
from time import time
from collections import deque

''' core algorithm level 1: 1D-only proof of concept, 
applied here to process lines of grey-scale pixels but not effective in recognition of 2D images. 

Cross-comparison between consecutive pixels within horizontal scan line (row).
Resulting difference patterns dPs (spans of pixels forming same-sign differences)
and relative match patterns vPs (spans of pixels forming same-sign predictive value)
are redundant representations of each line of pixels.

postfix '_' denotes array name, vs. identical name of its elements '''


def recursive_comparison(x, p, pri_p, d, v, vP, dP, vP_, dP_, olp, X, redun, rng):

    # incremental-range comp within vPs or incremental-derivation comp within dPs,
    # called from pre_recursive_comp(), which is called from form_P

    d += p - pri_p  # fuzzy d accumulates differences between p and all prior ps in extended rng
    v += min(p, pri_p) - ave  # fuzzy v accumulates deviation of match between p and all prior ps in extended rng

    vP, dP, vP_, dP_, olp = form_pattern(1, vP, dP, vP_, dP_, olp, pri_p, d, v, x, X, redun, rng)
    # forms value pattern vP: span of pixels with same-sign vs
    dP, vP, dP_, vP_, olp = form_pattern(0, dP, vP, dP_, vP_, olp, pri_p, d, v, x, X, redun, rng)
    # forms difference pattern dP: span of pixels with same-sign ds

    olp += 1  # overlap between concurrent vP and dP, to be buffered in olp_s at termination
    return d, v, vP, dP, vP_, dP_, olp  # for next-p comp, vP and dP increment, output


def pre_recursive_comp(typ, e_, redun, rng):  # pre-processing for comp recursion over e_: elements of selected pattern

    redun += 1  # count of redundant sets of patterns formed by recursive_comparison
    X = len(e_)

    olp, vP_, dP_ = 0, [], []  # olp: overlap between vP and dP:
    vP = 0, 0, 0, 0, 0, [], []  # pri_s, I, D, V, recomp, t_, olp_
    dP = 0, 0, 0, 0, 0, [], []  # pri_s, I, D, V, recomp, d_, olp_

    if typ: # comparison range increment within e_ = t_ of vP

        for x in range(rng, X):
            p = e_[x][0]  # input d and v are not used, accumulation with pri_p' d and v only
            pri_p, d, v = e_[x-rng]  # comp of rng-distant pixels:
            d, v, vP, dP, vP_, dP_, olp = \
            recursive_comparison(x, p, pri_p, d, v, vP, dP, vP_, dP_, olp, X, redun, rng)
    else:
        pri_d = e_[0]  # comparison derivation incr within e_ = d_ of dP
        d, v = 0, 0
        for x in range(1, X):
            d = e_[x]
            d, v, vP, dP, vP_, dP_, olp = \
            recursive_comparison(x, d, pri_d, d, v, vP, dP, vP_, dP_, olp, X, redun, rng)
            pri_d = d

    return vP_, dP_  # local vP_ + dP_ replaces t_ or d_


def form_pattern(typ, P, alt_P, P_, alt_P_, olp, pri_p, d, v, x, X, redun, rng):  # accumulation, termination, recursive comp in P: vP | dP

    if typ: s = 1 if v >= 0 else 0  # sign of d, 0 is positive?
    else:   s = 1 if d >= 0 else 0  # sign of v, 0 is positive?

    pri_s, I, D, V, recomp, e_, olp_ = P  # e_: type of elements in P depends on the level of comp recursion
    if x > rng + 2 and (s != pri_s or x == X - 1):  # P is terminated and evaluated for recursion

        if typ:
            if len(e_) > rng + 3 and pri_s == 1 and V > ave_V * redun:  # for minimum of 3 tuples
                recomp = 1  # recursive comp range increase flag
                e_.append(pre_recursive_comp(1, e_, redun+1, rng+1))  # comparison range increase within e_ = t_
        else:
            if len(e_) > 3 and abs(D) > ave_D * redun:  # for a minimum of 3 ds
                recomp = 1  # recursive derivation increase flag
                e_.append(pre_recursive_comp(0, e_, redun+1, 1))  # comparison derivation increase within e_ = d_

        P = typ, pri_s, I, D, V, recomp, e_, olp_
        P_.append(P)  # output to second level

        o = len(P_), olp  # index of current P and terminated olp are buffered in alt_olp_
        alt_P[6].append(o)
        o = len(alt_P_), olp  # index of current alt_P and terminated olp buffered in olp_
        olp_.append(o)

        olp, I, D, V, recomp, e_, olp_ = 0, 0, 0, 0, 0, [], []  # initialized P and olp

    pri_s = s   # current sign is stored as prior sign; P (span of pixels forming same-sign v | d) is incremented:
    I += pri_p  # ps summed within vP | dP
    D += d      # fuzzy ds summed within vP | dP
    V += v      # fuzzy vs summed within vP | dP

    if typ:
        e_.append((pri_p, d, v))  # inputs for greater rng comp are tuples, vs. pixels for initial comp
    else:
        e_.append(d)  # prior ds of the same sign are buffered within dP

    P = pri_s, I, D, V, recomp, e_, olp_

    return P, alt_P, P_, alt_P_, olp  # alt_ and _alt_ are accumulated per line


def comparison(x, p, rng_t_, vP, dP, vP_, dP_, olp, X):  # pixel is compared to rng prior pixels

    max_index = min_rng-1
    for index, (pri_p, d, m) in enumerate(rng_t_):

        d += p - pri_p  # fuzzy d: running sum of differences between pixel and all subsequent pixels within min_rng
        m += min(p, pri_p)  # fuzzy m: running sum of matches between pixel and all subsequent pixels within min_rng

        if index < max_index:
            rng_t_[index] = (pri_p, d, m)
        else:
            v = m - ave * min_rng # value or deviation of match, sign determines inclusion into positive or negative vP
            # completed tuple (pri_p, d, v) of summation range = rng (maxlen in rng_t_) is transferred to form_pattern:

            vP, dP, vP_, dP_, olp = form_pattern(1, vP, dP, vP_, dP_, olp, pri_p, d, v, x, X, 1, min_rng)
            # forms value pattern vP: span of pixels with same-sign vs
            dP, vP, dP_, vP_, olp = form_pattern(0, dP, vP, dP_, vP_, olp, pri_p, d, v, x, X, 1, min_rng)
            # forms difference pattern dP: span of pixels with same-sign ds

            olp += 1  # overlap between vP and dP, stored in both and terminated with either

    rng_t_.appendleft((p, 0, 0))  # new tuple is added, maxlen parameter removes completed tuple in deque

    return rng_t_, vP, dP, vP_, dP_, olp  # for next-p comparison, vP and dP increment, output


def frame(frame_of_pixels_):  # postfix '_' denotes array name, vs. identical name of its elements

    frame_of_patterns_ = []  # output frame of vPs: relative-match patterns, and dPs: difference patterns
    Y, X = frame_of_pixels_.shape  # Y: frame height, X: frame width

    for y in range(Y):
        p_ = frame_of_pixels_[y, :]   # y is index of new line p_

        olp, vP_, dP_ = 0, [], []  # initialized at each level
        vP = 0, 0, 0, 0, 0, [], []  # pri_s, I, D, V, recomp, t_, olp_
        dP = 0, 0, 0, 0, 0, [], []  # pri_s, I, D, V, recomp, d_, olp_

        rng_t_ = deque(maxlen=min_rng)  # incomplete fuzzy tuples: summation range < rng
        rng_t_.append((p_[0], 0, 0))  # prior tuple, no d, m at x = 0

        for x in range(1, X):  # cross-compares consecutive pixels within each line

            p = p_[x]  # new pixel, for fuzzy comparison to it_:
            it_, vP, dP, vP_, dP_, olp = comparison(x, p, rng_t_, vP, dP, vP_, dP_, olp, X)

        # line ends, last rng of incomplete tuples is discarded
        frame_of_patterns_.append((vP_, dP_))  # line of patterns is added to frame of patterns

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
ave = 63  # average match between pixels, minimal for inclusion into positive vP
ave_V = 63  # min V for initial incremental-range comparison(t_)
ave_D = 63  # min |D| for initial incremental-derivation comparison(d_)

start_time = time()
frame_of_patterns_ = frame(image)
end_time = time() - start_time
print(end_time)

