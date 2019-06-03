import cv2
import argparse
from time import time
from collections import deque

''' 
Principal version of 1st level 1D algorithm: dP definition and recursion eval within negative mPs only, vs. independently in line_olp.
Secondary dmPs, etc. are defined through direct match (min d, independent from dd), so it should use overlapping form_P?

Initial match is defined as ave abs(d) - abs(d): secondary to difference because stable visual property of objects is albedo, but 
such stability has low correlation with the value of albedo. 

Although indirect measure of match, low abs(d) is still predictive: uniformity across space correlates with stability over time.
Illumination is locally stable, so variation of albedo can be approximated as difference (vs. ratio) of brightness.

Cross-comparison of consecutive pixels of horizontal line forms match patterns mPs (spans of pixels with same-sign match deviation),
and difference patterns dPs (spans of pixels with same-sign differences) within each negative mP.

form_pattern() is conditionally recursive, cross-comparing p | d within a queue of above- minimal length and summed M | D.
In the code, postfix '_' denotes array name, vs. identical name of array elements '''


def form_pattern(dderived, P, P_, pri_p, d, m, rdn, rng, x, X):  # accumulation, termination, and recursive comp within exclusive mP ( dP

    s = m > 0  # sign, 0 is positive?   form_P-> pri mP ( sub_dP_, no type:
    pri_s, L, I, D, M, r, e_ = P  # depth of elements in e_ = r: depth of prior comp recursion within P

    if ( x > rng * 2 and s != pri_s ) or x == X:  # m sign change, mP is terminated and evaluated for recursive comp
        if not pri_s:  # negative mP forms dP_, is evaluated for recursion
            dP_ = []
            dP = int(e_[0][1] > 0), 0, 0, 0, 0, 0, []  # pri_s, L, I, D, M, r, e_
            e_.append( ( 0, 0, 0 ) )

            for i in range(L + 1):
                ip, id, im = e_[i]
                sd = 1 if id > 0 else 0
                pri_sd, Ld, Id, Dd, Md, rd, ed_ = dP

                if ( pri_sd != sd and i > 0 ) or i == L:
                    if Ld > rng * 2 and Dd > ave_M * rdn:  # comp range increase within e_, rdn (redundancy) is incremented per comp recursion
                        rd = 1
                        mdP_= []
                        mdP = 0, 0, 0, 0, 0, 0, []  # pri_s, L, I, D, M, r, e_;  no Alt: M is defined through abs(d)
                        ed_.append((0, 0, 0))
                        fdd, fmd = 0, 0

                        for j in range(1, Ld + 1):  # bilateral comp between consecutive dj s
                            dj = ed_[j][1]
                            _dj = ed_[j-1][1]
                            dd = dj - _dj
                            md = min(dj, _dj) - ave_m  # magnitude of vars derived from d corresponds to predictive value, thus direct match
                            fdd += dd  # bilateral difference and match between ds
                            fmd += md
                            if j > 1: mdP, mdP_ = form_pattern(1, mdP, mdP_, _dj, fdd, fmd, rdn+1, rng, j, Ld)
                            fdd = dd
                            fmd = md

                        ed_ = mdP_  # ders are replaced with mdPs: spans of pixels that form same-sign md
                    dP_.append( ( pri_sd, Ld, Id, Dd, Md, rd, ed_ ) )
                    sd, Ld, Id, Dd, Md, dr, de_ = 0, 0, 0, 0, 0, 0, []

                Ld += 1; Id += ip; Dd += id; Md += im; ed_.append( ( ip, id, im ) )
                dP = sd, Ld, Id, Dd, Md, rd, ed_

            rng += 1

            if L > rng * 2 and -M > ave_M * rdn:  # comp range increase within e_, rdn (redundancy) is incremented per comp recursion
                r = 1
                sub_mP_= []
                sub_mP = 0, 0, 0, 0, 0, 0, []  # pri_s, L, I, D, M, r, e_;  no Alt: M is defined through abs(d)
                e_.append( ( 0, 0, 0 ) )

                for i in range(rng, L+1):  # comp between rng-distant pixels, also bilateral, if L > rng * 2?
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
                    if i >= rng * 2: sub_mP, sub_mP_ = form_pattern(dderived, sub_mP, sub_mP_, _ip, _fd, _fm, rdn+1, rng, i, L)

                e_= sub_mP_  # ders replaced with sub_mPs: spans of pixels that form same-sign m
            e_ = e_, dP_

        P_.append((pri_s, L, I, D, M, r, e_))  # terminated P output to second level; x == X-1 doesn't always terminate?
        L, I, D, M, r, e_ = 0, 0, 0, 0, 0, []  # new P initialization

    pri_s = s  # current sign is stored as prior sign; P (span of pixels forming same-sign m | d) is incremented:
    L += 1  # length of mP | dP
    I += pri_p  # input ps summed within mP | dP
    D += d  # fuzzy ds summed within mP | dP
    M += m  # fuzzy ms summed within mP | dP
    e_+= [(pri_p, d, m)]  # recursive comp over tuples vs. pixels, dP' p, m ignore: no accum but buffer: no copy in mP?

    P = pri_s, L, I, D, M, r, e_
    return P, P_


def cross_comp(frame_of_pixels_):  # postfix '_' denotes array name, vs. identical name of its elements
    frame_of_patterns_ = []  # output frame of mPs: match patterns, and dPs: difference patterns

    for y in range(ini_y +1, Y):
        pixel_ = frame_of_pixels_[y, :]  # y is index of new line pixel_
        P_ = []; P = 0, 0, 0, 0, 0, 0, []  # pri_s, L, I, D, M, r, ders_ # initialized at each line
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

                elif x > min_rng * 2 - 1:
                    P, P_ = form_pattern(0, P, P_, pri_p, fd, fm, 1, min_rng, x, X)  # forms mPs: spans of pixels with same-sign m

            ders_.appendleft((p, back_fd, back_fm))  # new tuple with initialized d and m, maxlen displaces completed tuple from rng_t_
        frame_of_patterns_ += [P_]  # line of patterns is added to frame of patterns, last incomplete ders are discarded
    return frame_of_patterns_  # frame of patterns is output to level 2


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/raccoon_eye.jpg')
arguments = vars(argument_parser.parse_args())
image = cv2.imread(arguments['image'], 0).astype(int)

# the same image can be loaded online, without cv2:
# from scipy import misc
# f = misc.face(gray=True)  # load pix-mapped image
# f = f.astype(int)

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








