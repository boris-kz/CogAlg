'''
  line_patterns is a principal version of 1st-level 1D algorithm
  Operations:
  -
- Cross-compare consecutive pixels within each row of image, forming dert_: queue of derts, each a tuple of derivatives per pixel.
  dert_ is then segmented into patterns Pms and Pds: contiguous sequences of pixels forming same-sign match or difference.
  Initial match is inverse deviation of variation: m = ave_|d| - |d|,
  rather than a minimum for directly defined match: albedo of an object doesn't correlate with its predictive value.
  -
- Match patterns Pms are spans of inputs forming same-sign match. Positive Pms contain high-match pixels, which are likely
  to match more distant pixels. Thus, positive Pms are evaluated for cross-comp of pixels over incremented range.
  -
- Difference patterns Pds are spans of inputs forming same-sign ds. d sign match is a precondition for d match, so only
  same-sign spans (Pds) are evaluated for cross-comp of constituent differences, which forms higher derivatives.
  (d match = min: rng+ comp value: predictive value of difference is proportional to its magnitude, although inversely so)
  -
  Both extended cross-comp forks are recursive: resulting sub-patterns are evaluated for deeper cross-comp, same as top patterns.
  These forks here are exclusive per P to avoid redundancy, but they do overlap in line_patterns_olp.
'''

# add ColAlg folder to system path
import sys
from os.path import dirname, join, abspath

from numpy import int16, int32
sys.path.insert(0, abspath(join(dirname("CogAlg"), '../..')))
import cv2
# import argparse
import pickle
from time import time
from matplotlib import pyplot as plt
from itertools import zip_longest
from frame_2D_alg.class_cluster import ClusterStructure, NoneType, comp_param

class Cdert(ClusterStructure):
    i = int  # input for range_comp only
    p = int  # accumulated in rng
    d = int  # accumulated in rng
    m = int  # distinct in deriv_comp only

class CP(ClusterStructure):
    L = int
    I = int
    D = int
    M = int
    x0 = int
    dert_ = list  # contains (i, p, d, m)
    sublayers = list  # multiple layers of sub_P_s from d segmentation or extended comp, nested to depth = sub_[n]
    # for layer-parallel access and comp, ~ frequency domain, composition: 1st: dert_, 2nd: sub_P_[ derts], 3rd: sublayers[ sub_Ps[ derts]]
    fPd = bool  # P is Pd if true, else Pm; also defined per layer

verbose = False
# pattern filters or hyper-parameters: eventually from higher-level feedback, initialized here as constants:
ave = 15  # |difference| between pixels that coincides with average value of Pm
ave_min = 2  # for m defined as min |d|: smaller?
ave_M = 20  # min M for initial incremental-range comparison(t_), higher cost than der_comp?
ave_D = 5  # min |D| for initial incremental-derivation comparison(d_)
ave_nP = 5  # average number of sub_Ps in P, to estimate intra-costs? ave_rdn_inc = 1 + 1 / ave_nP # 1.2
ave_rdm = .5  # obsolete: average dm / m, to project bi_m = m * 1.5
ave_splice = 50  # to merge a kernel of 3 adjacent Ps
init_y = 0  # starting row, the whole frame doesn't need to be processed

'''
    Conventions:
    postfix '_' denotes array name, vs. same-name elements
    prefix '_' denotes prior of two same-name variables
    prefix 'f' denotes flag
    capitalized variables are normally summed small-case variables
'''

def cross_comp(frame_of_pixels_):  # converts frame_of_pixels to frame_of_patterns, each pattern may be nested

    Y, X = frame_of_pixels_.shape  # Y: frame height, X: frame width
    frame_of_patterns_ = []
    '''
    if cross_comp_spliced: process all image rows as a single line, vertically consecutive and preserving horizontal direction:
    pixel_=[]; dert_=[]  
    for y in range(init_y, Y):  
        pixel_.append([ frame_of_pixels_[y, :]])  # splice all rows into pixel_
    _i = pixel_[0]
    else:
    '''
    for y in range(init_y, Y):  # y is index of new line pixel_, a brake point here, we only need one row to process
        if logging:
            global logs_2D, logs_3D  # to share between functions
            logs_2D = np.empty((0, 6), dtype=int32) # empty 2D array for filling by layer0 output variables

        # initialization:
        dert_ = []  # line-wide i_, p_, d_, m__
        pixel_ = frame_of_pixels_[y, :]
        _i = pixel_[0]
        # pixel i is compared to prior pixel _i in a row:
        for i in pixel_[1:]:
            d = i -_i  # accum in rng
            p = i +_i  # accum in rng
            m = ave - abs(d)  # accum in rng, for consistency with deriv_comp output, else redundant
            dert_.append( Cdert( i=i, p=p, d=d, m=m) )
            _i = i
        # form m Patterns, evaluate intra_Pm_ per Pm:
        comb_layers, Pm_ = form_P_(dert_, rdn=1, rng=1, fPd=False)

        # add line of patterns to frame of patterns:
        frame_of_patterns_.append(Pm_)  # skip if cross_comp_spliced

        if logging:
            if len(logs_2D) < max_P_len:  # number of Ps per row, align arrays before stacking:
                no_data = np.zeros((max_P_len - len(logs_2D), 6), dtype=int32)
                logs_2D = np.append(logs_2D, no_data, axis=0)
            if logs_3D is not None:
                logs_3D = np.dstack((logs_3D, logs_2D))  # form 3D array of logs by stacking 2D arrays
            else:
                logs_3D = logs_2D # form 3D array of logs by stacking 2D arrays

    if logging:
        logs_3D = logs_3D.T  # rotate for readability
        data_dim1, data_dim2, data_dim3 = logs_3D.shape # define the dimensions of the array
        # add headers to log file
        with open("layer0_log.csv", "w") as csvFile:
            write = csv.writer(csvFile, delimiter=",")
            csv_header = ("row", "parameter", "values...")
            write.writerow(csv_header)
        # define the formatting
        parameter_names = ["L=", "I=", "D=", "M=", "x0=", "depth="]  # depth: nesting in sublayers
        row_numbers = list(range(data_dim1))  # P_len for every image row
        logs_flat = logs_3D.reshape(data_dim1 * data_dim2, data_dim3) # transform 3D array to 2D table
        df = pd.DataFrame(  # write log to dataframe and save as csv file
            data=logs_flat,
            index=pd.MultiIndex.from_product([row_numbers, parameter_names]))
        df.to_csv('layer0_log.csv', mode='a', header=True, index=True)

    return frame_of_patterns_  # frame of patterns is an input to level 2


def form_P_(dert_, rdn, rng, fPd):  # accumulation and termination
    # initialization:
    P_ = []
    x = 0
    _sign = None  # to initialize 1st P, (None != True) and (None != False) are both True

    for dert in dert_:  # segment by sign
        if fPd: sign = dert.d > 0
        else:   sign = dert.m > 0
        if sign != _sign:
            # sign change, initialize and append P
            P = CP(L=1, I=dert.p, D=dert.d, M=dert.m, x0=x, dert_=[dert], sublayers=[], fPd=fPd)
            P_.append(P)  # still updated with accumulation below
        else:
            # accumulate params:
            P.L += 1; P.I += dert.p; P.D += dert.d; P.M += dert.m
            P.dert_ += [dert]
        x += 1
        _sign = sign

    # draft:
    comb_layers = intra_P_(P_, rdn, rng, fPd)

    if logging:  # fill the array with layer0 params
        global logs_2D  # reset for each row
        for i in range(len(P_)):  # for each P
            logs_1D = np.array(([P_[i].L], [P_[i].I], [P_[i].D], [P_[i].M], [P_[i].x0], [len(P_[i].sublayers)] )).T  # 2D structure
            logs_2D = np.append(logs_2D, logs_1D, axis=0)  # log for every image row

    # print(logs_2D.shape)
    return comb_layers, P_  # sub_Ps,
    # or no return, intra_Pm_ forms P.sublayers instead, including sub_Ps in 1st sublayer?


def intra_P_(P_, rdn, rng, fPd):

    comb_layers = []
    adj_M_ = form_adjacent_M_(P_)  # compute adjacent Ms to evaluate contrastive borrow potential

    for P, adj_M in zip(P_, adj_M_):
        if P.L > 2 * (rng+1):  # vs. **? rng+1 because rng is initialized at 0, as all params
            rel_adj_M = adj_M / -P.M  # for allocation of -Pm' adj_M to each of its internal Pds
            # Pd
            if fPd and min(abs(P.D), abs(P.D) * rel_adj_M) > ave_D * rdn and P.L > 0:
                ddert_ = deriv_comp(P.dert_)  # i is d
                sub_comb_layers, sub_Pm_ = form_P_(ddert_, rdn+1, rng+1, fPd=True)  # cluster Pd derts by md sign, eval intra_Pm_(Pdm_), won't happen
                P.sublayers += [[[True, True, rdn+1, rng+1, sub_Pm_, []]]] # 1st sublayer: fPd, fid, rdn, rng, sub_P_, sub_Ppm__=[], + Dert=[]?
                if sub_comb_layers: P.sublayers += [sub_comb_layers] # + subsequent deeper layers
                comb_layers = [comb_layer + sublayers for comb_layer, sublayers in
                                   zip_longest(comb_layers, P.sublayers,  fillvalue=[])]
            # Pm
            else:
                if P.M > 0:  # low-variation span, eval comp at rng=2^n: 1, 2, 3; kernel size 2, 4, 8...
                    if P.M > ave_M * rdn:  # no -adj_M: reduced by lending to contrast, should be reflected in ave?
                        '''
                        if localized filters:
                        loc_ave = (ave + (P.M - adj_M) / P.L) / 2  # mean ave + P_ave, possibly negative?
                        loc_ave_min = (ave_min + (P.M - adj_M) / P.L) / 2  # if P.M is min?
                        rdert_ = range_comp(P.dert_, loc_ave, loc_ave_min, fid)
                        '''
                        rdert_ = range_comp(P.dert_)  # rng+ comp, skip predictable next dert, localized ave?
                        # redundancy to higher levels, or +=1 for the weaker layer?
                        sub_comb_layers, sub_Pm_ = form_P_(rdert_, rdn+1, rng+1, fPd=False)  # cluster by m sign, eval intra_Pm_
                        # 1st sublayer is one element, double brackets are for concatenation, sub_Ppm__=[], + Dert=[]:
                        # current sub layer
                        P.sublayers += [[[False, fPd, rdn+1, rng+1, sub_Pm_, []]]]
                        if sub_comb_layers: P.sublayers += [sub_comb_layers] # + subsequent deeper layers

                        # splice sublayers across sub_Ps1:]:
                        comb_layers = [comb_layer + sublayers for comb_layer, sublayers in
                                   zip_longest(comb_layers, P.sublayers,  fillvalue=[])]

                else:  # neg Pm: high-variation span, min neg M is contrast value, borrowed from adjacent +Pms:
                    if min(-P.M, adj_M) > ave_D * rdn:  # cancelled M+ val, M = min | ~v_SAD

                        rel_adj_M = adj_M / -P.M  # for allocation of -Pm' adj_M to each of its internal Pds
                        sub_comb_layers, sub_Pd_ = form_P_(P.dert_, rdn+1, rng+1, fPd=True)  # cluster by d sign: partial d match, eval intra_Pm_(Pdm_)
                        # move the below to form_P_?
                        P.sublayers += [[[True, True, rdn+1, rng+1, sub_Pd_, []]]] # 1st sublayer, sub_Ppm__=[], + Dert=[]?
                        if sub_comb_layers: P.sublayers += [sub_comb_layers] # + subsequent deeper layers
                        # splice sublayers across sub_Ps
                        comb_layers = [comb_layer + sublayers for comb_layer, sublayers in
                                   zip_longest(comb_layers, P.sublayers,  fillvalue=[])]

    return comb_layers

''' 
Sub-recursion in intra_P extends pattern with a hierarchy of sub-patterns (sub_), to be adjusted by feedback:
'''
def intra_Pm_(P_, irdn, irng, fPd):  # evaluate for sub-recursion in line Pm_, pack results into sub_Pm_

    adj_M_ = form_adjacent_M_(P_)  # compute adjacent Ms to evaluate contrastive borrow potential
    comb_layers = []  # combine into root P sublayers[1:]  # move to form_P_?

    for P, adj_M in zip(P_, adj_M_):  # each sub_layer is nested to depth = sublayers[n]
        if P.L > 0: #2 * (rng+1):  # vs. **? rng+1 because rng is initialized at 0, as all params
            rng, rdn = irng, irdn  # prevent rng&rdn increases as we loop through multiple Ps, each layer should have same rng&rdn

            if P.M > 0:  # low-variation span, eval comp at rng=2^n: 1, 2, 3; kernel size 2, 4, 8...
                if P.M > ave_M * rdn:  # no -adj_M: reduced by lending to contrast, should be reflected in ave?
                    '''
                    if localized filters:
                    loc_ave = (ave + (P.M - adj_M) / P.L) / 2  # mean ave + P_ave, possibly negative?
                    loc_ave_min = (ave_min + (P.M - adj_M) / P.L) / 2  # if P.M is min?
                    rdert_ = range_comp(P.dert_, loc_ave, loc_ave_min, fid)
                    '''
                    rdert_ = range_comp(P.dert_)  # rng+ comp, skip predictable next dert, localized ave?
                    rng += 1; rdn += 1  # redundancy to higher levels, or +=1 for the weaker layer?
                    sub_Pm_ = form_P_(rdert_, rdn, rng, fPd=False)  # cluster by m sign, eval intra_Pm_
                    # move the below to form_P_?
                    # 1st sublayer is one element, double brackets are for concatenation, sub_Ppm__=[], + Dert=[]:
                    P.sublayers += [[False, fPd, rdn, rng, sub_Pm_, []]]
                    if len(sub_Pm_) > 4:
                        P.sublayers += [intra_Pm_(sub_Pm_, rdn+1, rng+1, fPd)]  # feedback, add sublayer param summing for comp_sublayers?
                        # splice sublayers across sub_Ps, for return as root P sublayers[1:]:
                        comb_layers = [comb_layers + sublayers for comb_layers, sublayers in
                                       zip_longest(comb_layers, P.sublayers, fillvalue=[])]

            else:  # neg Pm: high-variation span, min neg M is contrast value, borrowed from adjacent +Pms:
                if min(-P.M, adj_M) > ave_D * rdn:  # cancelled M+ val, M = min | ~v_SAD

                    rel_adj_M = adj_M / -P.M  # for allocation of -Pm' adj_M to each of its internal Pds
                    sub_Pd_ = form_P_(P.dert_, rdn+1, rng, fPd=True)  # cluster by d sign: partial d match, eval intra_Pm_(Pdm_)
                    # move the below to form_P_?
                    P.sublayers += [[True, True, rdn, rng, sub_Pd_, []]]  # 1st sublayer, sub_Ppm__=[], + Dert=[]?

                    P.sublayers += [intra_Pd_(sub_Pd_, rel_adj_M, rdn+1, rng)]  # der_comp eval per neg Pm
                    # splice sublayers across sub_Ps, for return as root P sublayers[1:]:
                    comb_layers = [comb_layers + sublayers for comb_layers, sublayers in
                                   zip_longest(comb_layers, P.sublayers, fillvalue=[])]

    return comb_layers

def intra_Pd_(Pd_, rel_adj_M, irdn, irng):  # evaluate for sub-recursion in line P_, packing results in sub_P_

    comb_layers = []
    for P in Pd_:  # each sub in sub_ is nested to depth = sub_[n]

        if min(abs(P.D), abs(P.D) * rel_adj_M) > ave_D * irdn and P.L > 3:  # abs(D) * rel_adj_M: allocated adj_M
            rng, rdn = irng, irdn
            # cross-comp of ds:
            ddert_ = deriv_comp(P.dert_)  # i is d
            sub_Pm_ = form_P_(ddert_, rdn+1, rng+1, fPd=True)  # cluster Pd derts by md sign, eval intra_Pm_(Pdm_), won't happen
            # move the below to form_P_?
            P.sublayers += [[[True, True, rdn, rng, sub_Pm_, []]]]  # 1st sublayer: fPd, fid, rdn, rng, sub_P_, sub_Ppm__=[], + Dert=[]?
            if len(sub_Pm_) > 3:
                P.sublayers += intra_Pm_(sub_Pm_, rdn+1, rng + 1, fPd=True)
                # splice sublayers across sub_Ps, for return as root P sublayers[1:]:
                comb_layers = [comb_layers + sublayers for comb_layers, sublayers in
                               zip_longest(comb_layers, P.sublayers, fillvalue=[])]
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
    M_ = [0] + [Pm.M for Pm in Pm_] + [0]  # list of adj M components in the order of Pm_, + first and last M=0,

    adj_M_ = [ (abs(prev_M) + abs(next_M)) / 2  # mean adjacent Ms
               for prev_M, next_M in zip(M_[:-2], M_[2:])  # exclude first and last Ms
             ]
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
    _i = dert_[0].i

    for dert in dert_[2::2]:  # all inputs are sparse, skip odd pixels compared in prior rng: 1 skip / 1 add to maintain 2x overlap
        d = dert.i -_i
        rp = dert.p + _i  # intensity accumulated in rng
        rd = dert.d + d   # difference accumulated in rng
        rm = dert.m + ave - abs(d)  # m accumulated in rng
        # for consistency with deriv_comp, else redundant
        rdert_.append( Cdert( i=dert.i,p=rp,d=rd,m=rm ))
        _i = dert.i

    return rdert_

def deriv_comp(dert_):  # cross-comp consecutive ds in same-sign dert_: sign match is partial d match
    # dd and md may match across d sign, but likely in high-match area, spliced by spec in comp_P?
    # initialization:
    ddert_ = []
    _d = abs( dert_[0].d)  # same-sign in Pd

    for dert in dert_[1:]:
        # same-sign in Pd
        d = abs( dert.d )
        rd = d + _d
        dd = d - _d
        md = min(d, _d) - abs( dd/2) - ave_min  # min_match because magnitude of derived vars corresponds to predictive value
        ddert_.append( Cdert( i=dert.d,p=rd,d=dd,m=md ))
        _d = d

    return ddert_

def splice_P_(P_, fPd):  # currently not used, replaced by compact() in line_PPs
    '''
    Initial P termination is by pixel-level sign change, but resulting separation may be insignificant on a pattern level.
    That is, separating opposite-sign pattern is weak relative to separated same-sign patterns.
    The criterion to re-evaluate separation is similarity of P-defining param: M/L for Pm, D/L for Pd, among the three Ps
    If relative similarity > merge_ave: all three Ps are merged into one.
    '''
    splice_val_ = [splice_eval(__P, _P, P, fPd)  # compute splice values
                   for __P, _P, P in zip(P_, P_[1:], P_[2:])]
    sorted_splice_val_ = sorted(enumerate(splice_val_),
                                key=lambda k: k[1],
                                reverse=True)   # sort index by splice_val_
    if sorted_splice_val_[0][1] <= ave_splice:  # exit recursion
        return P_

    folp_ = np.zeros(len(P_), bool)  # if True: P is included in another spliced triplet
    spliced_P_ = []
    for i, splice_val in sorted_splice_val_:  # loop through splice vals
        if splice_val <= ave_splice:  # stop, following splice_vals will be even smaller
            break
        if folp_[i : i+3].any():  # skip if overlap
            continue
        folp_[i : i+3] = True     # splice_val > ave_splice: overlapping Ps folp=True
        __P, _P, P = P_[i : i+3]  # triplet to splice
        # merge _P and P into __P:
        __P.accum_from(_P, excluded=['x0', 'ix0'])
        __P.accum_from(P, excluded=['x0', 'ix0'])

        if hasattr(__P, 'pdert_'):  # for splice_Pp_ in line_PPs
            __P.pdert_ += _P.pdert_ + P.pdert_
        else:
            __P.dert_ += _P.dert_ + P.dert_
        spliced_P_.append(__P)

    # add remaining Ps into spliced_P
    spliced_P_ += [P_[i] for i, folp in enumerate(folp_) if not folp]
    spliced_P_.sort(key=lambda P: P.x0)  # back to original sequence

    if len(spliced_P_) > 4:
        splice_P_(spliced_P_, fPd)

    return spliced_P_


def splice_eval(__P, _P, P, fPd):  # should work for splicing Pps too
    '''
    For 3 Pms, same-sign P1, P3, opposite-sign P2:
    relative continuity vs separation = abs(( M2/ ( M1+M3 )))
    relative similarity = match (M1/L1, M3/L3) / miss (match (M1/L1, M2/L2) + match (M3/L3, M2/L2)) # both should be negative
    or P2 is reinforced as contrast - weakened as distant -> same value, not merged?
    splice P1, P3: by proj mean comp, ~ comp_param, ave / contrast P2
    re-run intra_P in line_PPs Pps?
    also distance / meanL, if 0: fractional distance = meanL / olp? reduces ave, not m?
    '''
    if fPd:
        if _P.D==0: _P.D =.1  # prevents /0
        rel_continuity = abs((__P.D + P.D) / _P.D)
        __mean= __P.D/__P.L; _mean= _P.D/_P.L; mean= P.D/P.L
    else:
        if _P.M == 0: _P.M =.1  # prevents /0
        rel_continuity = abs((__P.M + P.M) / _P.M)
        __mean= __P.M/__P.L; _mean= _P.M/_P.L; mean= P.M/P.L

    m13 = min(mean, __mean) - abs(mean-__mean)/2    # inverse match of P1, P3
    m12 = min(_mean, __mean) - abs(_mean-__mean)/2  # inverse match of P1, P2, should be negative
    m23 = min(_mean, mean) - abs(_mean- mean)/2     # inverse match of P2, P3, should be negative

    miss = abs(m12 + m23) if not 0 else .1
    rel_similarity = (m13 * rel_continuity) / miss  # * rel_continuity: relative value of m13 vs m12 and m23
    # splice_value = rel_continuity * rel_similarity

    return rel_similarity

if __name__ == "__main__":
    ''' 
    Parse argument (image)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='.//raccoon.jpg')
    arguments = vars(argument_parser.parse_args())
    # Read image
    image = cv2.imread(arguments['image'], 0).astype(int)  # load pix-mapped image
    '''
    logging = 0  # log dataframes
    fpickle = 2  # 0: read from the dump; 1: pickle dump; 2: no pickling
    render = 0
    fline_PPs = 0
    start_time = time()

    if logging:
        import csv
        import numpy as np
        import pandas as pd
        # initialize the 3D stack to store the nested structure of layer0 parameters
        max_P_len = 380 # defined empirically
        logs_3D = None  # empty array to store all log data
    if fpickle == 0:
        # Read frame_of_patterns from saved file instead
        with open("frame_of_patterns_.pkl", 'rb') as file:
            frame_of_patterns_ = pickle.load(file)
    else:
        # Run functions
        image = cv2.imread('../raccoon.jpg', 0).astype(int)  # manual load pix-mapped image
        assert image is not None, "No image in the path"
        # Main
        frame_of_patterns_ = cross_comp(image)  # returns Pm__
        if fpickle == 1: # save the dump of the whole data_1D to file
            with open("frame_of_patterns_.pkl", 'wb') as file:
                pickle.dump(frame_of_patterns_, file)

    if render:
        image = cv2.imread('../raccoon.jpg', 0).astype(int)  # manual load pix-mapped image
        plt.figure(); plt.imshow(image, cmap='gray'); plt.show() # show the image below in gray

    if fline_PPs:  # debug line_PPs_draft
        from line_PPs_draft import *
        frame_PP__ = []

        for y, P_ in enumerate(frame_of_patterns_):
            rdn_Pp__ = search(P_, fPd=0)
            frame_PP__.append(rdn_Pp__)
        # draw_PP_(image, frame_PP_)  # debugging

    end_time = time() - start_time
    print(end_time)


    # below is old parts of line_PPs:

    '''     
    param_ = [[ getattr( P, param_name[0]), P.L, P.x0] for P in P_]  # param values
    D_ = [ getattr(P, "D") for P in P_]
    _par_= []
    for (I, L, x0),(D,_,_) in zip(param_[:-1], layer0["D_"][:-1]):  # _I in (I,L,x0) is forward projected by _D in (D,L,x0)
        _par_.append((I-(D/2), L, x0))
    par_= []
    for (I, L, x0),(D,_,_) in zip(param_[1:], layer0["D_"][1:]): # I in (I,L,x0) is backward projected by D in (D,L,x0)
        par_.append((I+(D/2), L, x0))

    rL_ = [_L[0]/L[0] for _L, L in zip(layer0["L_"][:-1],layer0["L_"][1:])] # _L = L_[:-1], L = L_[1:], div_comp L, no search?
    mL_ = [int(rL) * min(_L[0],L[0]) for _L, L, rL in zip(layer0["L_"][:-1],layer0["L_"][1:], rL_)] # definition of div_match
    '''
    def search_old(P_, fPd):  # cross-compare patterns within horizontal line

        sub_search_recursive(P_, fPd)  # search with incremental distance: first inside sublayers
        layer0 = {'L_': [], 'I_': [], 'D_': [], 'M_': []}  # param_name: [params]

        if len(P_) > 1:  # at least 2 comparands
            Ldert_ = []; rL_ = []
            # unpack Ps:
            for P in P_:
                if "_P" in locals():  # not the 1st P
                    L = P.L; _L = _P.L
                    rL = L / _L  # div_comp L: higher-scale, not accumulated: no search
                    mL = int(max(rL, 1 / rL)) * min(L, _L)  # match in comp by division as additive compression, not directional
                    Ldert_.append(Cdert(i=L, p=L + _L, d=rL, m=mL))
                    rL_.append(rL)
                _P = P
                layer0['I_'].append([P.I, P.L, P.x0])  # I tuple
                layer0['D_'].append([P.D, P.L, P.x0])  # D tuple
                layer0['M_'].append([P.M, P.L, P.x0])  # M tuple

            dert1__ = [Ldert_]  # no search for L, step=1 only, contains derts vs. pderts
            Pdert__ = [Ldert_]  # Pp elements: pderts if param is core m, else derts

            for param_name in ["I_", "D_", "M_"]:
                param_ = layer0[param_name]  # param values
                par_ = param_[1:]  # compared vectors:
                _par_ = [[_par * rL, L, x0] for [_par, L, x0], rL in zip(param_[:-1], rL_)]  # normalize by rL

                if ((param_name == "I_") and not fPd) or ((param_name == "D_") and fPd):  # dert-level P-defining params
                    if not fPd:
                        # project I by D, or D by Dd in deriv_comp sub_Ps:
                        _par_ = [[_par - (D / 2), L, x0] for [_par, L, x0], [D, _, _] in zip(_par_, layer0["D_"][:-1])]
                        # _I in (I,L,x0) is  - (D / 2): forward projected by _D in (D,L,x0)
                        par_ = [[par + (D / 2), L, x0] for [par, L, x0], [D, _, _] in zip(par_, layer0["D_"][1:])]
                        # I in (I,L,x0) is backward projected by D in (D,L,x0)
                        Pdert__ += [search_param_(_par_, par_, P_[:-1], ave, rave=1)]  # pdert_ if "I_"
                    del _P
                    _rL_ = []
                    for P in P_:  # form rLs to normalize cross-comp of same-M-sign Ps in pdert2_
                        if "_P" in locals():  # not the 1st P
                            if "__P" in locals():  # not the 2nd P
                                _rL_.append(P.L / __P.L)
                            __P = _P
                        _P = P
                    __par_ = [[__par * _rL, L, x0] for [__par, L, x0], _rL in zip(param_[:-2], _rL_)]  # normalize by _rL
                    # step=2 comp for P splice, one param: (I and not fPd) or (D and fPd):
                    dert2_ = [comp_param(__par, par, param_name[0], ave) for __par, par in zip(__par_, par_[1:])]
                # else step=1 only:

                dert1_ = [comp_param(_par, par, param_name[0], ave) for _par, par in zip(_par_, par_)]  # append pdert1_ per param_
                dert1__ += [dert1_]
                if not param_name == "I_": Pdert__ += [dert1_]  # dert_ = comp_param_

            rdn__ = sum_rdn_(layer0, Pdert__, fPd=1)  # assign redundancy to lesser-magnitude m|d in param pair for same-_P Pderts
            rdn_Ppm__ = []

            for param_name, Pdert_, rdn_ in zip(layer0, Pdert__, rdn__):  # segment Pdert__ into Pps
                if param_name == "I_" and not fPd:  # = isinstance(Pdert_[0], Cpdert)
                    Ppm_ = form_Pp_rng(Pdert_, rdn_, P_)
                else:
                    Ppm_ = form_Pp_(Pdert_, param_name, rdn_, P_, fPd=0)  # Ppd_ is formed in -Ppms only, in intra_Ppm_
                # list of param rdn_Ppm_s:
                rdn_Ppm__ += [form_rdn_Pp_(Ppm_, param_name, dert1__, dert2_, fPd=0)]

        return rdn_Ppm__


    def form_PP_(params_derPp____, fPd):  # Draft:
        '''
        unpack 4-layer derPp____: _names ( _Pp_ ( names ( Pp_ ))),
        pack derPps with overlapping match: sum of concurrent mPps > ave_M * rolp, into PPs of PP_
        '''
        rdn = [.25, .5, .25, .5]  # {'L_': .25, 'I_': .5, 'D_': .25, 'M_': .5}
        names = ['L_', 'I_', 'D_', 'M_']
        Rolp = 0
        PP_ = []
        _sign = None
        # init new empty derPp____ with the same list structure as params_derPp____, for [i][j][k] indexing later
        derPp____ = [[[[] for param_derPp_ in param_derPp__] \
                      for param_derPp__ in param_derPp___] \
                     for param_derPp___ in params_derPp____]
        param_name_ = derPp____.copy()

        for i, _param_derPp___ in enumerate(params_derPp____):  # derPp___ from comp_Pp (across params)
            for j, _Pp_derPp__ in enumerate(_param_derPp___):  # from comp_Pp (param_Pp_, other params)
                for k, param_derPp_ in enumerate(_Pp_derPp__):  # from comp_Pp (_Pp, other params)
                    for (derPp, rolp, _name, name) in param_derPp_:  # from comp_Pp (_Pp, other param' Pp_)
                        # debugging
                        if names[i] != _name: raise ValueError("Wrong _name")
                        if names[k] != name: raise ValueError("Wrong name")

                        if "pre_PP" not in locals(): pre_PP = CPP(derPp____=derPp____.copy())
                        # if fPd: derPp_val = derPp.dPp; ave = ave_D
                        # else:   derPp_val = derPp.mPp; ave = ave_M
                        # mean_rdn = (rdn[i] + rdn[k]) / 2  # of compared params
                        # if derPp_val * mean_rdn > ave:
                        # else: pre_PP = CPP(derPp____=derPp____.copy())
                        # accum either sign, no eval or sub_PP_ per layer:
                        Rolp += rolp
                        pre_PP.accum_from(derPp)
                        pre_PP.derPp____[i][j][k].append(derPp)
                        pre_PP.param_name_.append((names[i], names[k]))
            '''    
            We can't evaluate until the top loop because any overlap may form sufficient match. 
            Then we only define pre_PPs by overlap of any element of any layer to any other element of any other layer.
            But there are so many possible overlaps that pre_PP may never terminate.
            Another way to define them is by minimal combined-layers' match per x (iP). 
            But then we are back to immediate multi-param comp_P_, which is pointless because different derivatives anti-correlate.
                    # inclusion into higher layer of pre_PP by the sum of concurrent mPps > ave_M * Rolp, over all lower layers:
                    if "pre_PP" in locals() and pre_PP.derPp____[i][j][k] and not pre_PP.mPp > ave_M * Rolp:
                        pre_PP = CPP(derPp____=derPp____.copy())
                # pre_PP.derPp____[i][j] is a nested list, we need to check recursively to determine whether there is any appended derPp
                if "pre_PP" in locals() and not emptylist(pre_PP.derPp____[i][j]) and not pre_PP.mPp > ave_M * Rolp:
                    pre_PP = CPP(derPp____=derPp____.copy())
            '''
            if "pre_PP" in locals() and not emptylist(pre_PP.derPp____[i]):
                if pre_PP.mPp > ave_M * Rolp:
                    PP_.append(pre_PP)  # no negative PPs?
                    _sign = True
                else:
                    _sign = False
                    pre_PP = CPP(derPp____=derPp____.copy())
        return PP_

    # https://stackoverflow.com/questions/1593564/python-how-to-check-if-a-nested-list-is-essentially-empty
    def emptylist(in_list):
        '''
        check if nested list is totally empty
        '''
        if isinstance(in_list, list):  # Is a list
            return all(map(emptylist, in_list))
        return False  # Not a list

    def comp_Pp(_Pp, Pp, layer0):
        '''
        next level line_PPPs:
        PPm_ = search_Pp_(layer0, fPd=0)  # calls comp_Pp_ and form_PP_ per param
        PPd_ = search_Pp_(layer0, fPd=1)
        '''
        mPp = dPp = 0
        layer1 = dict({'L': .0, 'I': .0, 'D': .0, 'M': .0})
        dist_coef = ave_rM * (1 + _Pp.negL / _Pp.L)
        # average match projected at current distance, needs a review
        for param_name in layer1:
            if param_name == "I":
                ave = ave_inv  # * dist_coef
            else:
                ave = ave_min  # * dist_coef
            param = getattr(_Pp, param_name)
            _param = getattr(Pp, param_name)
            dert = comp_param(_param, param, [], ave)
            rdn = layer0[param_name + '_'][1]  # index 1 =rdn
            mPp += dert.m * rdn
            dPp += dert.d * rdn
            layer1[param_name] = dert

        negM = _Pp.negM - Pp.negM
        negL = _Pp.L - Pp.negL
        negiL = _Pp.iL - Pp.negiL

        '''
        options for div_comp, etc.    
        if P.sign == _P.sign: mP *= 2  # sign is MSB, value of sign match = full magnitude match?
        if mP > 0
            # positive forward match, compare sublayers between P.sub_H and _P.sub_H:
           comp_sublayers(_P, P, mP)
        if isinstance(_P.derP, CderP):  # derP is created in comp_sublayers
            _P.derP.sign = sign
            _P.derP.layer1 = layer1
            _P.derP.accumulate(mP=mP, neg_M=neg_M, neg_L=neg_L, P=_P)
            derP = _P.derP
        else:
            derP = CderP(sign=sign, mP=mP, neg_M=neg_M, neg_L=neg_L, P=_P, layer1=layer1)
            _P.derP = derP
        '''
        derPp = CderPp(mPp=mPp, dPp=dPp, negM=negM, negL=negL, negiL=negiL, _Pp=_Pp, Pp=Pp, layer1=layer1)

        return derPp


    def div_comp_P(PP_):  # draft, check all PPs for x-param comp by division between element Ps
        '''
        div x param if projected div match: compression per PP, no internal range for ind eval.
        ~ (L*D + L*M) * rm: L=min, positive if same-sign L & S, proportional to both but includes fractional miss
        + PPm' DL * DS: xP difference compression, additive to x param (intra) compression: S / L -> comp rS
        also + ML * MS: redundant unless min or converted?
        vs. norm param: Var*rL-> comp norm param, simpler but diffs are not L-proportional?
        '''
        for PP in PP_:
            vdP = (PP.adj_mP + PP.P.M) * abs(PP.dP) - ave_div
            if vdP > 0:
                # if irM * D_vars: match rate projects der and div match,
                # div if scale invariance: comp x dVars, signed
                ''' 
                | abs(dL + dI + dD + dM): div value ~= L, Vars correlation: stability of density, opposite signs cancel-out?
                | div_comp value is match: min(dL, dI, dD, dM) * 4, | sum of pairwise mins?
                '''
                _derP = PP.derP_[0]
                # smP, vmP, neg_M, neg_L, iP, mL, dL, mI, dI, mD, dD, mM, dM = P,
                # _sign, _L, _I, _D, _M, _dert_, _sub_H, __smP = _derP.P
                _P = _derP.P
                for i, derP in enumerate(PP.derP_[1:]):
                    P = derP.P
                    # DIV comp L, SUB comp (summed param * rL) -> scale-independent d, neg if cross-sign:
                    rL = P.L / _P.L
                    # mL = whole_rL * min_L?
                    '''
                    dI = I * rL - _I  # rL-normalized dI, vs. nI = dI * rL or aI = I / L
                    mI = ave - abs(dI)  # I is not derived, match is inverse deviation of miss
                    dD = D * rL - _D  # sum if opposite-sign
                    mD = min(D, _D)   # same-sign D in dP?
                    dM = M * rL - _M  # sum if opposite-sign
                    mM = min(M, _M)   # - ave_rM * M?  negative if x-sign, M += adj_M + deep_M: P value before layer value?
                    mP = mI + mM + mD  # match(P, _P) for derived vars, defines norm_PPm, no ndx: single, but nmx is summed
                    '''
                    for (param, _param) in zip([P.I, P.D, P.M], [_P.I, _P.D, _P.M]):
                        dm = comp_param(param, _param, [], ave, rL)
                        layer1.append([dm.d, dm.m])
                        mP += dm.m; dP += dm.d

                    if dP > P.derP.dP:
                        ndP_rdn = 1; dP_rdn = 0  # Not sure what to do with these
                    else:
                        dP_rdn = 1; ndP_rdn = 0

                    if mP > derP.mP:
                        rrdn = 1  # added to rdn, or diff alt, olp, div rdn?
                    else:
                        rrdn = 2
                    if mP > ave * 3 * rrdn:
                        # rvars = mP, mI, mD, mM, dI, dD, dM  # redundant vars: dPP_rdn, ndPP_rdn, assigned in each fork?
                        rvars = layer1
                    else:
                        rvars = []
                    # append rrdn and ratio variables to current derP:
                    # PP.derP_[i] += [rrdn, rvars]
                    PP.derP_[i].rrdn = rrdn
                    PP.derP_[i].layer1 = rvars
                    # P vars -> _P vars:
                    _P = P
                    '''
                    m and d from comp_rate is more accurate than comp_norm?
                    rm, rd: rate value is relative? 
                    also define Pd, if strongly directional? 
                    if dP > ndP: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
                    else:        dPP_rdn = 1; ndPP_rdn = 0
                    '''
        return PP_

    def form_adjacent_mP(derPp_):  # not used in discontinuous search?
        pri_mP = derPp_[0].mP
        mP = derPp_[1].mP
        derPp_[0].adj_mP = derPp_[1].mP

        for i, derP in enumerate(derPp_[2:]):
            next_mP = derP.mP
            derPp_[i + 1].adj_mP = (pri_mP + next_mP) / 2
            pri_mP = mP
            mP = next_mP

        return derPp_

