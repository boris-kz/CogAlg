import cv2
import argparse
from time import time
from collections import deque

''' 
line_POC without recursion 
'''

def form_pattern(typ, P, P_, pri_p, d, m, rng, x, X):  # accumulation, termination, and recursive comp within pattern mP | dP

    if typ: s = 1 if m >= 0 else 0  # sign of core var m, 0 is positive?
    else:   s = 1 if d >= 0 else 0  # sign of core var d, 0 is positive?

    pri_s, L, I, D, M, e_ = P  # depth of elements in e_ = r: flag of comp recursion within P
    if (x > rng * 2 and s != pri_s) or x == X-1:  # core var sign change, P is terminated and evaluated for recursive comp

        P_.append((typ, pri_s, L, I, D, M, e_))  # terminated P is output to the next level of search
        L, I, D, M, e_ = 0, 0, 0, 0, []  # new P initialization

    pri_s = s   # current sign is stored as prior sign; P (span of pixels forming same-sign m | d) is incremented:
    L += 1      # length of mP | dP
    I += pri_p  # input ps summed within mP | dP
    D += d      # fuzzy ds summed within mP | dP
    M += m      # fuzzy ms summed within mP | dP
    e_.append((pri_p, d, m))
    P = pri_s, L, I, D, M, e_
    return P, P_


def cross_comp(frame_of_pixels_):  # postfix '_' denotes array name, vs. identical name of its elements
    frame_of_patterns_ = []  # output frame of mPs: match patterns, and dPs: difference patterns

    for y in range(ini_y, Y):
        pixel_ = frame_of_pixels_[y, :]  # y is index of new line pixel_

        dP_= []; dP = 0,0,0,0,0,[]  # initialized at each line,
        mP_= []; mP = 0,0,0,0,0,[]  # pri_s, L, I, D, M, ders_
        max_index = min_rng - 1  # max index of rng_ders_
        ders_ = deque(maxlen=min_rng)  # array of incomplete ders, within rng from input pixel: summation range < rng
        ders_.append((0, 0, 0))  # prior tuple, no d, m at x = 0
        back_ = []  # fuzzy derivatives d and m from rng of backward comps per prior pixel

        for x, p in enumerate(pixel_):  # pixel p is compared to rng of prior pixels in horizontal line, summing d and m per prior pixel
            for index, (pri_p, d, m) in enumerate(ders_):

                d += p - pri_p  # fuzzy d: running sum of differences between pixel and all subsequent pixels within rng
                m += ave_d - abs(p - pri_p)  # fuzzy m: running sum of matches between pixel and all subsequent pixels within rng
                if index < max_index:
                    ders_[index] = (pri_p, d, m)

                elif x > min_rng * 2 - 1:
                    back_d, back_m = back_.pop(0)  # back_d|m is for bilateral sum, rng-distant from i_d|m, buffered in back_
                    bi_d = d + back_d  # d and m are accumulated over full bilateral (before and after pri_p) min_rng
                    bi_m = m + back_m
                    mP, mP_ = form_pattern(1, mP, mP_, pri_p, bi_d, bi_m, min_rng, x, X)  # forms mP: span of pixels with same-sign m
                    dP, dP_ = form_pattern(0, dP, dP_, pri_p, bi_d, bi_m, min_rng, x, X)  # forms dP: span of pixels with same-sign d

            back_.append((d, m))  # accumulated through ders_ comp
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

