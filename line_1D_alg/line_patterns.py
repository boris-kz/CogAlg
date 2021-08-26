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
sys.path.insert(0, abspath(join(dirname("CogAlg"), '..')))
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
        Pm_ = form_P_(dert_, rdn=1, rng=1, fPd=False)
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
    P.sublayers += [[False, fPd, rdn, rng, P_, []]]
    '''
    if len(P_) > 4:
        comb_layers = []  # combine into root P sublayers[1:]
        P.sublayers += [ intra_P_(P_, rdn, rng, fPd)]  # feedback, add sublayer param summing for comp_sublayers?
        # combine intra_Pm_ and intra_Pd into intra_P_
        # or rdn+1, ?rng+1?  # evaluates range_comp | deriv_comp per Pm.dert_
        # splice sublayers across sub_Ps, for return as root P sublayers[1:]:
        P.sublayers += [comb_layers + sublayers for comb_layers, sublayers in
                       zip_longest(comb_layers, P.sublayers, fillvalue=[])]
    '''
    if logging:  # fill the array with layer0 params
        global logs_2D  # reset for each row
        for i in range(len(P_)):  # for each P
            logs_1D = np.array(([P_[i].L], [P_[i].I], [P_[i].D], [P_[i].M], [P_[i].x0], [len(P_[i].sublayers)] )).T  # 2D structure
            logs_2D = np.append(logs_2D, logs_1D, axis=0)  # log for every image row

    # print(logs_2D.shape)
    return P_  # sub_Ps,
    # or no return, intra_Pm_ forms P.sublayers instead, including sub_Ps in 1st sublayer?

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
        image = cv2.imread('.//raccoon.jpg', 0).astype(int)  # manual load pix-mapped image
        assert image is not None, "No image in the path"
        # Main
        frame_of_patterns_ = cross_comp(image)  # returns Pm__
        if fpickle == 1: # save the dump of the whole data_1D to file
            with open("frame_of_patterns_.pkl", 'wb') as file:
                pickle.dump(frame_of_patterns_, file)

    if render:
        image = cv2.imread('.//raccoon.jpg', 0).astype(int)  # manual load pix-mapped image
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