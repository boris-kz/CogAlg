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


def recursive_comparison(x, p, pri_p, fd, fv, vP, dP, vP_, dP_, olp, X, Ave, rng):

    # incremental-range comp within vPs or incremental-derivation comp within dPs,
    # called from pre_recursive_comp(), which is called from form_P

    d = p - pri_p      # difference between consecutive pixels
    m = min(p, pri_p)  # match between consecutive pixels
    v = m - Ave        # value: deviation of match between consecutive pixels

    fd += d  # fuzzy d accumulates ds between p and all prior ps in rng, same accum for fv:
    fv += v  # fuzzy v; shorter-rng fv and lower-der fd are in lower Ps, different for p and pri_p

    vP, dP, vP_, dP_, olp = form_pattern(1, vP, dP, vP_, dP_, olp, pri_p, fd, fv, x, X, Ave, rng)
    # forms value pattern vP: span of pixels forming same-sign fv s

    dP, vP, dP_, vP_, olp = form_pattern(0, dP, vP, dP_, vP_, olp, pri_p, fd, fv, x, X, Ave, rng)
    # forms difference pattern dP: span of pixels forming same-sign fd s

    olp += 1  # overlap between concurrent vP and dP, to be buffered in olp_s at termination

    return fd, fv, vP, dP, vP_, dP_, olp  # for next-p comp, vP and dP increment, output


def pre_recursive_comp(typ, e_, Ave, rng):  # pre-processing for comp recursion within selected pattern

    Ave += ave  # filter accumulation compensates for redundancy of derivatives formed by recursive_comparison
    X = len(e_)

    olp, vP_, dP_ = 0, [], []  # olp: overlap between vP and dP:
    vP = 0, 0, 0, 0, 0, [], []  # pri_s, I, D, V, recomp, t_, olp_
    dP = 0, 0, 0, 0, 0, [], []  # pri_s, I, D, V, recomp, d_, olp_

    if typ:  # comparison range increment within e_ = t_ of vP

        rng += 1  # comp range counter, recorded within Ps formed by recursive_comp
        for x in range(rng+1, X):

            p = e_[x][0]  # input fd and fv are not used, directional pri_p accum only
            pri_p, fd, fv = e_[x-rng]  # for comparison of rng-pixel-distant pixels:

            fd, fv, vP, dP, vP_, dP_, olp = \
            recursive_comparison(x, p, pri_p, fd, fv, vP, dP, vP_, dP_, olp, X, Ave, rng)

    else:  # comparison derivation incr within e_ = d_ of dP (not tuples per range incr?)

        pri_d = e_[0]  # no deriv_incr while rng < rng, only more fuzzy
        fd, fv = 0, 0

        for x in range(1, X):
            d = e_[x]
            fd, fv, vP, dP, vP_, dP_, olp = \
            recursive_comparison(x, d, pri_d, fd, fv, vP, dP, vP_, dP_, olp, X, Ave, rng)
            pri_d = d

    return vP_, dP_  # local vP_ + dP_ replaces t_ or d_


def form_pattern(typ, P, alt_P, P_, alt_P_, olp, pri_p, fd, fv, x, X, ave, rng):

    # accumulation, termination, recursion within patterns (vPs and dPs)

    if typ: s = 1 if fv >= 0 else 0  # sign of fd, 0 is positive?
    else:   s = 1 if fd >= 0 else 0  # sign of fv, 0 is positive?

    pri_s, I, D, V, recomp, e_, olp_ = P  # debug: 0 values in P?

    if x > rng + 2 and (s != pri_s or x == X - 1):  # P is terminated and evaluated

        if typ:
            if len(e_) > rng + 3 and pri_s == 1 and V > ave + ave_V:  # minimum of 3 tuples
                recomp = 1  # recursive comp range increase flag
                e_.append(pre_recursive_comp(1, e_, ave, rng))
                # comparison range increase within e_ = t_
        else:
            if len(e_) > 3 and abs(D) > ave + ave_D:  # minimum of 3 ds
                recomp = 1  # recursive derivation increase flag
                rng = 1  # comp between consecutive ds:
                e_.append(pre_recursive_comp(0, e_, ave, rng))
                # comparison derivation increase within e_ = d_

        P = typ, pri_s, I, D, V, recomp, e_, olp_
        P_.append(P)  # output to level_2
        # print ("typ:", typ, "pri_s:", pri_s, "I:", I, "D:", D, "V:", V, "recomp:", recomp, "e_:", e_, "olp_:", olp_)

        o = len(P_), olp  # index of current P and terminated olp are buffered in alt_olp_
        alt_P[6].append(o)
        o = len(alt_P_), olp  # index of current alt_P and terminated olp buffered in olp_
        olp_.append(o)

        olp, I, D, V, recomp, e_, olp_ = 0, 0, 0, 0, 0, [], []  # initialized P and olp

    pri_s = s   # current sign is stored as prior sign; P (span of pixels forming same-sign v|d) is incremented:
    I += pri_p  # ps summed within vP
    D += fd     # fuzzy ds summed within vP
    V += fv     # fuzzy vs summed within vP

    if typ:
        e_.append((pri_p, fd, fv))  # inputs for greater rng comp are tuples, vs. pixels for initial comp
    else:
        e_.append(fd)  # prior fds of the same sign are buffered within dP

    P = pri_s, I, D, V, recomp, e_, olp_

    return P, alt_P, P_, alt_P_, olp  # alt_ and _alt_ are accumulated per line


def comparison(x, p, it_, vP, dP, vP_, dP_, olp, X, rng):  # pixel is compared to rng prior pixels

    index = 0  # alternative: for index in range(0, len(it_)-1): doesn't work quite right?

    for it in it_:  # incomplete tuples with fd, fm summation range from 0 to rng
        pri_p, fd, fm = it

        d = p - pri_p  # difference between consecutive pixels
        m = min(p, pri_p)  # match between consecutive pixels

        fd += d  # fuzzy d: sum of ds between p and all prior ps within it_
        fm += m  # fuzzy m: sum of ms between p and all prior ps within it_

        it_[index] = (pri_p, fd, fm)
        index += 1

    if len(it_) == rng:  # current tuple fd and fm are accumulated over range = rng

        fv = fm - ave  # fuzzy value: deviation of fuzzy match between consecutive pixels
        # fv sign determines completed tuple's inclusion into positive or negative vP

        vP, dP, vP_, dP_, olp = form_pattern(1, vP, dP, vP_, dP_, olp, pri_p, fd, fv, x, X, ave, rng)
        #  forms value pattern vP: span of pixels forming same-sign fv s

        dP, vP, dP_, vP_, olp = form_pattern(0, dP, vP, dP_, vP_, olp, pri_p, fd, fv, x, X, ave, rng)
        # forms difference pattern dP: span of pixels forming same-sign fd s

        olp += 1  # overlap between vP and dP, stored in both and terminated with either

    it_.appendleft((p, 0, 0))  # new tuple is added, displacing completed tuple
    # or left_fd and left_fm, for bilateral accumulation?

    return it_, vP, dP, vP_, dP_, olp  # for next-p comparison, vP and dP increment, output


def frame(Fp_):  # postfix '_' denotes array name, vs. identical name of its elements

    FP_ = []  # output frame of vPs: relative-match patterns, and dPs: difference patterns
    Y, X = Fp_.shape  # Y: frame height, X: frame width

    for y in range(Y):
        p_ = Fp_[y, :]   # y is index of new line p_

        olp, vP_, dP_ = 0, [], []  # initialized at each level
        vP = 0, 0, 0, 0, 0, [], []  # pri_s, I, D, V, recomp, t_, olp_
        dP = 0, 0, 0, 0, 0, [], []  # pri_s, I, D, V, recomp, d_, olp_

        it_ = deque(maxlen=rng)  # incomplete fuzzy tuples: summation range < rng
        it_.append((p_[0], 0, 0))  # prior tuple, no d, m at x = 0

        for x in range(1, X):  # cross-compares consecutive pixels
            p = p_[x]  # new pixel, for fuzzy comparison to it_:

            it_, vP, dP, vP_, dP_, olp = \
            comparison(x, p, it_, vP, dP, vP_, dP_, olp, X, rng)

        # line ends, last rng of incomplete tuples is discarded

        LP_ = vP_, dP_   # line of patterns is formed from a line of pixels
        FP_.append(LP_)  # line of patterns is added to frame of patterns at y = len(FP_)

    return FP_  # frame of patterns is output to level 2

# from scipy import misc
# f = misc.face(gray=True)  # input frame of pixels
# f = f.astype(int)

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/racoon.jpg')
arguments = vars(argument_parser.parse_args())
image = cv2.imread(arguments['image'], 0).astype(int)

# pattern filters: eventually from higher-level feedback, initialized here as constants:

rng = 3  # fuzzy pixel comparison range, initialized here but eventually a higher-level feedback
ave = 63 * rng  # average match between pixels, minimal for inclusion into positive vP
ave_V = 63  # min V for initial incremental-range comparison(t_)
ave_D = 63  # min |D| for initial incremental-derivation comparison(d_)

start_time = time()
FP_ = frame(image)
end_time = time() - start_time
print(end_time)

