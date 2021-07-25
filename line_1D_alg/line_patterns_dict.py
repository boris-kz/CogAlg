'''
  line_patterns using dicts vs. classes, Kelvin's port
'''

# add ColAlg folder to system path
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname("CogAlg"), '..')))

import cv2
import csv
import argparse
from time import time
from utils import *
from itertools import zip_longest
from frame_2D_alg.class_cluster import setdict_attr, NoneType, comp_param


ave = 15  # |difference| between pixels that coincides with average value of Pm
ave_min = 2  # for m defined as min |d|: smaller?
ave_M = 50  # min M for initial incremental-range comparison(t_), higher cost than der_comp?
ave_D = 5  # min |D| for initial incremental-derivation comparison(d_)
ave_nP = 5  # average number of sub_Ps in P, to estimate intra-costs? ave_rdn_inc = 1 + 1 / ave_nP # 1.2
ave_rdm = .5  # obsolete: average dm / m, to project bi_m = m * 1.5
ave_merge = 50  # to merge a kernel of 3 adjacent Ps
init_y = 0  # starting row, the whole frame doesn't need to be processed

'''
    Conventions:
    postfix '_' denotes array name, vs. same-name elements
    prefix '_' denotes prior of two same-name variables
    prefix 'f' denotes flag
    capitalized variables are normally summed small-case variables
'''

def cross_comp(frame_of_pixels_):  # converts frame_of_pixels to frame_of_patterns, each pattern maybe nested

    Y, X = frame_of_pixels_.shape  # Y: frame height, X: frame width
    frame_of_patterns_ = []
    '''
    if cross_comp_spliced: process all image rows as a single line, vertically consecutive and preserving horizontal direction:
    pixel_=[]; dert_=[]  
    for y in range(init_y + 1, Y):  
        pixel_.append([ frame_of_pixels_[y, :]])  # splice all rows into pixel_
    _i = pixel_[0]
    else:
    '''
    for y in range(init_y + 1, Y):  # y is index of new line pixel_, a brake point here, we only need one row to process
        # initialization:
        dert_ = []  # line-wide i_, p_, d_, m__
        pixel_ = frame_of_pixels_[y, :]
        _i = pixel_[0]
        # pixel i is compared to prior pixel _i in a row:
        for i in pixel_[1:]:
            d = i -_i
            p = i +_i
            m = ave - abs(d)  # for consistency with deriv_comp output, otherwise redundant
            dert_.append({'i':i,'p':p,'d':d,'m':m})
            _i = i
        # form m Patterns, evaluate intra_Pm_ per Pm:
        Pm_ = form_P_(dert_, rdn=1, rng=1, fPd=False)
        # add line of patterns to frame of patterns:
        frame_of_patterns_.append(Pm_)  # skip if cross_comp_spliced

    return frame_of_patterns_  # frame of patterns is an intput to level 2


def form_P_(dert_, rdn, rng, fPd):  # accumulation and termination
    # initialization:
    P_ = []
    x = 0
    _sign = None  # to initialize 1st P, (None != True) and (None != False) are both True

    for dert in dert_:  # segment by sign
        if fPd: sign = dert['d'] > 0
        else:   sign = dert['m'] > 0
        if sign != _sign:
            # sign change, initialize and append P
            P = {'sign':_sign, 'L':1, 'I':dert['p'], 'D':dert['d'], 'M':dert['m'], 'x0':x, 'dert_':[dert], 'sublayers':[], 'fPd':fPd}
            P_.append(P)  # still updated with accumulation below
        else:
            # accumulate params:
            P['L'] += 1; P['I'] += dert['p']; P['D'] += dert['d']; P['M'] += dert['m']
            P['dert_'] += [dert]
        x += 1
        _sign = sign
    if len(P_) > 4:
        #P_ = splice_P_(P_, fPd=0)  # merge meanI- or meanD- similar and weakly separated Ps
        #if len(P_) > 4:
        intra_Pm_(P_, rdn, rng, not fPd)  # evaluates range_comp | deriv_comp sub-recursion per Pm
    '''
    with open("frame_of_patterns_2.csv", "a") as csvFile:  # current layer visualization
        write = csv.writer(csvFile, delimiter=",")
        for item in range(len(P_)):
            # print(P_[item].L, P_[item].I, P_[item].D, P_[item].M, P_[item].x0)
            write.writerow([P_[item].L, P_[item].I, P_[item].D, P_[item].M, P_[item].x0])
    '''
    return P_

''' 
Sub-recursion in intra_P extends pattern with sub_: hierarchy of sub-patterns, to be adjusted by macro-feedback:
'''
def intra_Pm_(P_, rdn, rng, fPd):  # evaluate for sub-recursion in line Pm_, pack results into sub_Pm_

    adj_M_ = form_adjacent_M_(P_)  # compute adjacent Ms to evaluate contrastive borrow potential
    comb_layers = []  # combine into root P sublayers[1:]

    for P, adj_M in zip(P_, adj_M_):  # each sub_layer is nested to depth = sublayers[n]
        if P['L'] > 2 ** (rng+1):  # rng+1 because rng is initialized at 0, as all params

            if P['M'] > 0:  # low-variation span, eval comp at rng=2^n: 1, 2, 3; kernel size 2, 4, 8...
                if P['M'] - adj_M > ave_M * rdn:  # reduced by lending to contrast: all comps form params for hLe comp?
                    '''
                    if localized filters:
                    P_ave = (P.M - adj_M) / P.L  
                    loc_ave = (ave - P_ave) / 2  # ave is reduced because it's for inverse deviation, possibly negative?
                    loc_ave_min = (ave_min + P_ave) / 2
                    rdert_ = range_comp(P.dert_, loc_ave, loc_ave_min, fid)
                    '''
                    rdert_ = range_comp(P['dert_'])  # rng+ comp with localized ave, skip predictable next dert
                    rdn += 1; rng += 1
                    sub_Pm_ = form_P_(rdert_, rdn, rng, fPd=False)  # cluster by m sign, eval intra_Pm_
                    Ls = len(sub_Pm_)
                    P['sublayers'] += [[(Ls, False, fPd, rdn, rng, sub_Pm_)]]  # add Dert=[] if Ls > min?
                    # 1st sublayer is single-element, packed in double brackets only to allow nesting for deeper sublayers
                    if len(sub_Pm_) > 4:
                        P['sublayers'] += intra_Pm_(sub_Pm_, rdn+1 + 1/Ls, rng+1, fPd)  # feedback
                        # add param summation within sublayer, for comp_sublayers?
                        # splice sublayers across sub_Ps:
                        comb_layers = [comb_layers + sublayers for comb_layers, sublayers in
                                       zip_longest(comb_layers, P['sublayers'], fillvalue=[])]

            else:  # neg Pm: high-variation span, min neg M is contrast value, borrowed from adjacent +Pms:
                if min(-P['M'], adj_M) > ave_D * rdn:  # cancelled M+ val, M = min | ~v_SAD

                    rel_adj_M = adj_M / -P['M']  # for allocation of -Pm' adj_M to each of its internal Pds
                    sub_Pd_ = form_P_(P['dert_'], rdn+1, rng, fPd=True)  # cluster by d sign: partial d match, eval intra_Pm_(Pdm_)
                    Ls = len(sub_Pd_)
                    P['sublayers'] += [[(Ls, True, True, rdn, rng, sub_Pd_)]]  # 1st layer, Dert=[], fill if Ls > min?

                    P['sublayers'] += intra_Pd_(sub_Pd_, rel_adj_M, rdn+1 + 1/Ls, rng)  # der_comp eval per nPm
                    # splice sublayers across sub_Ps, for return as root sublayers[1:]:
                    comb_layers = [comb_layers + sublayers for comb_layers, sublayers in
                                   zip_longest(comb_layers, P['sublayers'], fillvalue=[])]

    return comb_layers

def intra_Pd_(Pd_, rel_adj_M, rdn, rng):  # evaluate for sub-recursion in line P_, packing results in sub_P_

    comb_layers = []
    for P in Pd_:  # each sub in sub_ is nested to depth = sub_[n]

        if min(abs(P['D']), abs(P['D']) * rel_adj_M) > ave_D * rdn and P['L'] > 3:  # abs(D) * rel_adj_M: allocated adj_M
            # cross-comp of ds:
            ddert_ = deriv_comp(P['dert_'])  # i is d
            sub_Pm_ = form_P_(ddert_, rdn+1, rng+1, fPd=True)  # cluster Pd derts by md sign, eval intra_Pm_(Pdm_), won't happen
            Ls = len(sub_Pm_)
            # 1st layer: Ls, fPd, fid, rdn, rng, sub_P_:
            P['sublayers'] += [[(Ls, True, True, rdn, rng, sub_Pm_ )]]

            if len(sub_Pm_) > 3:
                P['sublayers'] += intra_Pm_(sub_Pm_, rdn+1 + 1/Ls, rng + 1, fPd=True)
                # splice sublayers across sub_Ps:
                comb_layers = [comb_layers + sublayers for comb_layers, sublayers in
                               zip_longest(comb_layers, P['sublayers'], fillvalue=[])]
    ''' 
    adj_M is not affected by primary range_comp per Pm?
    no comb_m = comb_M / comb_S, if fid: comb_m -= comb_|D| / comb_S: alt rep cost
    same-sign comp: parallel edges, cross-sign comp: M - (~M/2 * rL) -> contrast as 1D difference?
    '''
    return comb_layers

def form_adjacent_M_(Pm_):  # compute array of adjacent Ms, for contrastive borrow evaluation
    '''
    Value is projected match, while variation has contrast value only: it matters to the extent that it interrupts adjacent match: adj_M.
    In noise, there is a lot of variation. but no adjacent match to cancel, so that variation has no predictive value.
    On the other hand, we may have a 2D outline or 1D contrast with low gradient / difference, but it terminates some high-match area.
    Contrast is salient to the extent that it can borrow sufficient predictive value from adjacent high-match area.
    '''
    M_ = [Pm['M'] for Pm in Pm_]  # list of Ms in the order of Pm_

    adj_M_ = [(abs(prev_M) + abs(next_M)) / 2
              for prev_M, next_M in zip(M_[:-2], M_[2:])]  # adjacent Ms, first and last Ms
    adj_M_ = [M_[1]] + adj_M_ + [M_[-2]]  # sum previous and next adjacent Ms

    ''' expanded:
    pri_M = Pm_[0].M  # deriv_comp value is borrowed from adjacent opposite-sign Ms
    M = Pm_[1].M
    adj_M_ = [abs(Pm_[1].M)]  # initial next_M, also projected as prior for first P
    for Pm in Pm_[2:]:
        next_M = Pm.M
        adj_M_.append((abs(pri_M / 2) + abs(next_M / 2)))  # exclude M
        pri_M = M
        M = next_M
    adj_M_.append(abs(pri_M))  # no / 2: projection for last P
    '''
    return adj_M_

def range_comp(dert_):  # cross-comp of 2**rng- distant pixels: 4,8,16.., skipping intermediate pixels
    rdert_ = []
    _i = dert_[0]['i']

    for dert in dert_[2::2]:  # all inputs are sparse, skip odd pixels compared in prior rng: 1 skip / 1 add, to maintain 2x overlap
        d = dert['i'] -_i
        rp = dert['p'] + _i  # intensity accumulated in rng
        rd = dert['d'] + d   # difference accumulated in rng
        rm = dert['m'] + ave - abs(d)  # m accumulated in rng
        # for consistency with deriv_comp, else m is redundant
        rdert_.append({'i':dert['i'],'p':rp,'d':rd,'m':rm})
        _i = dert['i']

    return rdert_

def deriv_comp(dert_):  # cross-comp consecutive ds in same-sign dert_: sign match is partial d match
    # dd and md may match across d sign, but likely in high-match area, spliced by spec in comp_P?
    # initialization:
    ddert_ = []
    _d = abs( dert_[0]['d'])  # same-sign in Pd

    for dert in dert_[1:]:
        # same-sign in Pd
        d = abs( dert['d'] )
        rd = d + _d
        dd = d - _d
        md = min(d, _d) - abs( dd/2) - ave_min  # min_match because magnitude of derived vars corresponds to predictive value
        ddert_.append({'i':dert['d'],'p':rd,'d':dd,'m':md})
        _d = d

    return ddert_

def splice_P_(P_, fPd):
    '''
    Initial P separation is determined by pixel-level sign change, but resulting opposite-sign pattern may be relatively weak,
    and same-sign patterns it separates relatively strong.
    Another criterion to re-evaluate separation is similarity of defining param: M/L for Pm, D/L for Pd, among the three Ps
    If relative proximity * relative similarity > merge_ave: all three Ps should be merged into one.
    '''
    new_P_ = []
    while len(P_) > 2:  # at least 3 Ps
        __P = P_.pop(0)
        _P = P_.pop(0)
        P = P_.pop(0)

        if splice_eval(__P, _P, P, fPd) > ave_merge:  # no * ave_rM * (1 + _P.L / (__P.L+P.L) / 2): _P.L is not significant
            # for debugging
            #print('P_'+str(_P.id)+' and P_'+str(P.id)+' are merged into P_'+str(__P.id))
            # merge _P and P into __P
            for merge_P in [_P, P]:
                __P.x0 = min(__P.x0, merge_P.x0)
                __P.accum_from(merge_P)
                __P.dert_+= merge_P.dert_
            # back splicing
            __P = splice_P_back(new_P_, __P, fPd)
            P_.insert(0, __P)  # insert merged __P back into P_ to continue merging
        else:
            new_P_.append(__P) # append __P to P_ when there is no further merging process for __P
            P_.insert(0, P)    # insert P back into P_ for the consecutive merging process
            P_.insert(0, _P)  # insert _P back into P_ for the consecutive merging process

    # pack remaining Ps:
    if P_: new_P_ += P_
    return new_P_

def splice_P_back(new_P_, P, fPd):  # P is __P in calling splice_P_

    while len(new_P_) > 2:  # at least 3 Ps
        _P = new_P_.pop()
        __P = new_P_.pop()

        if splice_eval(__P, _P, P, fPd) > ave_merge:  # no * ave_rM * (1 + _P.L / (__P.L+P.L) / 2):
            # match projected at distance between P,__P: rM is insignificant
            # for debug purpose
            #print('P_'+str(_P.id)+' and P_'+str(P.id)+' are backward merged into P_'+str(__P.id))
            # merge _P and P into __P
            for merge_P in [_P, P]:
                __P.x0 = min(__P.x0, merge_P.x0)
                __P.accum_from(merge_P)
                __P.dert_+= merge_P.dert_
            P = __P  # also returned
        else:
            new_P_+= [__P, _P]
            break

    return P

def splice_eval(__P, _P, P, fPd):  # should work for splicing Pps too
    '''
    For 3 Pms, same-sign P1 and P3, and opposite-sign P2:
    relative proximity = abs((M1+M3) / M2)
    relative similarity = match (M1/L1, M3/L3) / miss (match (M1/L1, M2/L2) + match (M3/L3, M2/L2)) # both should be negative
    '''
    if fPd:
        proximity = abs((__P.D + P.D) / _P.D) if _P.D != 0 else 0  # prevents /0
        __mean=__P.D/__P.L; _mean=_P.D/_P.L; mean=P.D/P.L
    else:
        proximity = abs((__P.M + P.M) / _P.M) if _P.M != 0 else 0  # prevents /0
        __mean=__P.M/__P.L; _mean=_P.M/_P.L; mean=P.M/P.L
    m13 = min(mean, __mean) - abs(mean-__mean)/2   # P1 & P3
    m12 = min(_mean, __mean) - abs(_mean-__mean)/2 # P1 & P2
    m23 = min(_mean, mean) - abs(_mean- mean)/2    # P2 & P3

    similarity = m13 / abs( m12 + m23)  # both should be negative
    merge_value = proximity * similarity

    return merge_value

if __name__ == "__main__":
    ''' 
    Parse argument (image)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='.//raccoon.jpg')
    arguments = vars(argument_parser.parse_args())
    # Read image
    image = cv2.imread(arguments['image'], 0).astype(int)  # load pix-mapped image
    '''
    # show image in the same window as a code
    image = cv2.imread('.//raccoon.jpg', 0).astype(int)  # manual load pix-mapped image
    assert image is not None, "No image in the path"
    render = 0
    verbose = 0

    if render:
        plt.figure();plt.imshow(image, cmap='gray')  # show the image below in gray

    # for visualization:
    with open("frame_of_patterns_2.csv", "w") as csvFile:
        write = csv.writer(csvFile, delimiter=",")
        fieldnames = ("L=", "I=", "D=", "M=", "x0=")
        write.writerow(fieldnames)

    start_time = time()
    # Main
    frame_of_patterns_ = cross_comp(image)  # returns Pm__
    # from pprint import pprint
    # pprint(frame_of_patterns_[0])  # shows 1st layer Pm_ only

    fline_PPs = 0
    if fline_PPs:  # debug line_PPs_draft
        from line_PPs_draft import *
        frame_PP_ = []

        for y, P_ in enumerate(frame_of_patterns_):
            PP_ = search(P_)
            frame_PP_.append(PP_)

    end_time = time() - start_time
    print(end_time)