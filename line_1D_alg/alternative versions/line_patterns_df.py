'''
  line_patterns using dataframes vs. classes, Kelvin's port
'''
# add ColAlg folder to system path
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname("CogAlg"), '../../../../AppData/Roaming/JetBrains/PyCharmCE2021.1')))

import cv2
import argparse
from time import time
from utils import *
from itertools import zip_longest
import pandas as pd
import numpy as np
ave = 15  # |difference| between pixels that coincides with average value of Pm
ave_min = 2  # for m defined as min |d|: smaller?
ave_M = 50  # min M for initial incremental-range comparison(t_), higher cost than der_comp?
ave_D = 5  # min |D| for initial incremental-derivation comparison(d_)
ave_nP = 5  # average number of sub_Ps in P, to estimate intra-costs? ave_rdn_inc = 1 + 1 / ave_nP # 1.2
ave_rdm = .5  # average dm / m, to project bi_m = m * 1.5
init_y = 0  # starting row, the whole frame doesn't need to be processed
'''
    Conventions:
    postfix '_' denotes array name, vs. same-name elements
    prefix '_' denotes prior of two same-name variables
    prefix 'f' denotes flag
    capitalized variables are normally summed small-case variables
    df = dataframe
    prefix 'F' Filtered dataframe  
'''
def cross_comp(frame_of_pixels_):  # converts frame_of_pixels to frame_of_patterns, each pattern maybe nested
    frame_of_patterns_ = []
    image_df = pd.DataFrame(frame_of_pixels_)
    d_df = image_df.diff(axis=1).dropna(axis=1) #calculate difference (next value - prev value)
    p_df = ((image_df.transpose().rolling(2).sum()).transpose()).dropna(axis=1) # calculae sum of consecutive pixels
    m_df = ave - abs(d_df)
    for i_,p_,d_,m_ in zip(image_df.iterrows(),p_df.iterrows(),d_df.iterrows(),m_df.iterrows()):
        df_dert = pd.DataFrame(data={'p':p_[1],'d':d_[1],'m':m_[1]}).dropna(axis=1)
        df_dert['i'] = i_[1]
        df_dert = df_dert.apply(pd.to_numeric)
        Pm_ = form_P_(df_dert, rdn=1, rng=1, fPd=False)  # forms m-sign patterns
        frame_of_patterns_.append(Pm_) #list of P dfs

    return frame_of_patterns_  # frame of patterns is an output to level 2

def form_P_(df_dert,rdn,rng, fPd=False):  # initialization, accumulation, termination

    # initialization:
    P_ = [] #list of dfs
    #vectorized methods are used to apply transformations on whole data at once
    if fPd: df_dert['sign'] = df_dert.d > 0 #create a new column'sign' based on d,m signs - Vectorized
    else:   df_dert['sign'] = df_dert.m > 0
    for i,dert_  in df_dert.groupby((df_dert['sign'].shift() != df_dert['sign']).cumsum()): #Group same sign dert
        #groupby df_dert.sign - it returns groups of similar signs in incremental fashion
        #shift() creates a mask of provided column by hsifting down the values by 1, so 1st value shifted to 2nd index and so on, useful for forward comparison
        #cumsum() is used to differentiate the each group from the previous group
        dert = dert_.sum(axis=0) #accumulate each group/ sum rows of each group its a vectorized operation
        #append dict into P_ for later assignment into dataframe
        P_.append({'sign':dert.sign>0, 'L':len(dert_), 'I':dert.p, 'D':dert.d, 'M':dert.m, 'x0':i, 'dert_':dert_, 'sublayers':[], '_smP':False, 'fPd':fPd})
    df_P = pd.DataFrame(P_) #form dataframe of P_
    if len(df_P) > 4:
        # P_ = splice_P_(df_P, fPd=0)  # merge meanI- or meanD- similar and weakly separated Ps - NOT REVISED
        if len(P_) > 4:
            intra_Pm_(df_P, rdn, rng, not fPd)  # evaluates range_comp | deriv_comp sub-recursion per Pm


    return df_P

def form_adjacent_M_(Pm_):  # compute array of adjacent Ms, for contrastive borrow evaluation
    '''
    Value is projected match, while variation has contrast value only: it matters to the extent that it interrupts adjacent match: adj_M.
    In noise, there is a lot of variation. but no adjacent match to cancel, so variation in noise has no predictive value.
    On the other hand, we may have a 2D outline or 1D contrast with low gradient / difference, but it terminates adjacent uniform span.
    That contrast may be salient if it can borrow sufficient predictive value from that adjacent high-match span.
    '''
    Pm_['adj_M'] = Pm_['M']/2 + Pm_['M'].shift()/2 #shift Pm_['M'] by one and compute adj_M - vectorized operaiton
    #Pm_['adj_M'].iloc[0] = Pm_['M'].iloc[1] #assign 1st adj_M as after shifting 1st index of M becomes nan
    Pm_.at[0, 'adj_M'] = Pm_['M'].iloc[1] #assign 1st adj_M as after shifting 1st index of M becomes nan
    return Pm_

def intra_Pm_(P_, fid=False, rdn=1, rng=1):  # evaluate for sub-recursion in line Pm_, pack results into sub_Pm_
    #need to be vectorized
    P_ = form_adjacent_M_(P_)
    comb_layers = []  # combine into root P sublayers[1:]
    #P_ = df_P.loc[df_P.L>2 **(rng+1) & df_P.sign== True & df_P.M - df_P.adj_M > ave_M * rdn]
    #df_rdert = P_['dert_'].apply(range_comp)
    #df_sub_Pm = form_P_(df_rdert, fPd=False)
    #df_P.at[P_.Index, 'sublayers'] += [[(Ls, False, fid, rdn, rng, sub_Pm_, [], [])]]

    for P in P_.itertuples():  # each sub_layer is nested to depth = sublayers[n]
        if P.L > 2 ** (rng+1):  # rng+1 because rng is initialized at 0, as all params

            if P.sign:  # +Pm: low-variation span, eval comp at rng=2^n: 1, 2, 3; kernel size 2, 4, 8...
                if P.M - P.adj_M > ave_M * rdn:  # reduced by lending to contrast: all comps form params for hLe comp?
                    '''
                    if localized filters:
                    P_ave = (P.M - adj_M) / P.L  
                    loc_ave = (ave - P_ave) / 2  # ave is reduced because it's for inverse deviation, possibly negative?
                    loc_ave_min = (ave_min + P_ave) / 2
                    rdert_ = range_comp(P.dert_, loc_ave, loc_ave_min, fid)
                    '''
                    rdert_ = range_comp(P.dert_)  # rng+ comp with localized ave, skip predictable next dert
                    #rdert = P.dert_.apply(range_comp)
                    sub_Pm_ = form_P_(rdert_, rdn=1,rng=1,fPd=False)  # cluster by m sign
                    Ls = len(sub_Pm_)
                    #P.sublayers += [[(Ls, False, fid, rdn, rng, sub_Pm_, [], [])]]  # sub_PPm_, sub_PPd_, add Dert=[] if Ls > min?
                    P_.at[P.Index, 'sublayers'] += [[(Ls, False, fid, rdn, rng, sub_Pm_, [], [])]]
                    # 1st sublayer is single-element, packed in double brackets only to allow nesting for deeper sublayers
                    if len(sub_Pm_) > 4:
                        sub_Pm_ = form_adjacent_M_(sub_Pm_)
                        P_.at[P.Index, 'sublayers'] += intra_Pm_(sub_Pm_, fid, rdn=rdn+1 + 1/Ls, rng=rng+1)  # feedback
                        # add param summation within sublayer, for comp_sublayers?
                        # splice sublayers across sub_Ps:
                        comb_layers = [comb_layers + sublayers for comb_layers, sublayers in
                                       zip_longest(comb_layers, P.sublayers, fillvalue=[])]

            else:  # -Pm: high-variation span, min neg M is contrast value, borrowed from adjacent +Pms:
                if min(-P.M, P.adj_M) > ave_D * rdn:  # cancelled M+ val, M = min | ~v_SAD

                    rel_adj_M = P.adj_M / -P.M  # for allocation of -Pm' adj_M to each of its internal Pds
                    sub_Pd_ = form_P_(P.dert_, rdn=1, rng=1,fPd=True)  # cluster by input d sign match: partial d match
                    Ls = len(sub_Pd_)
                    P_.at[P.Index, 'sublayers'] += [[(Ls, True, True, rdn, rng, sub_Pd_)]]  # 1st layer, Dert=[], fill if Ls > min?

                    P_.at[P.Index, 'sublayers'] += intra_Pd_(sub_Pd_, rel_adj_M, rdn+1 + 1/Ls, rng)  # der_comp eval per nPm
                    # splice sublayers across sub_Ps, for return as root sublayers[1:]:
                    comb_layers = [comb_layers + sublayers for comb_layers, sublayers in
                                   zip_longest(comb_layers, P.sublayers, fillvalue=[])]

    return comb_layers


def intra_Pd_(Pd_, rel_adj_M, rdn=1, rng=1):  # evaluate for sub-recursion in line P_, packing results in sub_P_

    comb_layers = []
    for P in Pd_.itertuples():  # each sub in sub_ is nested to depth = sub_[n]

        if min(abs(P.D), abs(P.D) * rel_adj_M) > ave_D * rdn and P.L > 3:  # abs(D) * rel_adj_M: allocated adj_M
            # cross-comp of ds:
            ddert_ = deriv_comp(P.dert_)  # i_ is d
            sub_Pm_ = form_P_(ddert_, rdn=1,rng=1,fPd=True)  # cluster Pd derts by md, won't happen
            Ls = len(sub_Pm_)
            # 1st layer: Ls, fPd, fid, rdn, rng, sub_P_, sub_PPm_, sub_PPd_:
            Pd_.at[P.Index, 'sublayers'] += [[(Ls, True, True, rdn, rng, sub_Pm_, [], [] )]]

            if len(sub_Pm_) > 3:
                sub_Pm_ = form_adjacent_M_(sub_Pm_)
                Pd_.at[P.Index, 'sublayers'] += intra_Pm_(sub_Pm_, fid=True, rdn=rdn+1 + 1/Ls, rng=rng + 1)
                # splice sublayers across sub_Ps:
                comb_layers = [comb_layers + sublayers for comb_layers, sublayers in
                               zip_longest(comb_layers, P.sublayers, fillvalue=[])]
    ''' 
    adj_M is not affected by primary range_comp per Pm?
    no comb_m = comb_M / comb_S, if fid: comb_m -= comb_|D| / comb_S: alt rep cost
    same-sign comp: parallel edges, cross-sign comp: M - (~M/2 * rL) -> contrast as 1D difference?
    '''
    return comb_layers

def range_comp(dert_):  # cross-comp of 2**rng- distant pixels: 4,8,16.., skipping intermediate pixels

    d = dert_['i'].loc[2::2] - dert_['i'].loc[2::2].shift()
    rp = dert_['p'].loc[2::2] + dert_['i'].loc[2::2].shift()
    rd = dert_['d'].loc[2::2] + d
    rm = dert_['m'].loc[2::2] + ave - abs(d)

    df_rdert = pd.DataFrame(data={'i':dert_.i,'p':rp,'d':rd,'m':rm}).apply(pd.to_numeric)



    return df_rdert


def deriv_comp(dert_):  # cross-comp consecutive ds in same-sign dert_: sign match is partial d match
    # dd and md may match across d sign, but likely in high-match area, spliced by spec in comp_P?
    # initialization:

    i = dert_.d
    dp = abs(dert_['d']) + dert_['d'].shift()
    dd = abs(dert_['d']) - dert_['d'].shift()
    dm = np.fmin(dert_.d, dert_.d.shift()) - abs(dd/2) - ave_min
    df_ddert = pd.DataFrame(data={'i':i,'p':dp,'d':dd,'m':dm}).apply(pd.to_numeric)

    return df_ddert


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
    # Parse argument (image)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image',
                                 help='path to image file',
                                 default='.//raccoon.jpg')
    arguments = vars(argument_parser.parse_args())
    # Read image
    image = cv2.imread(arguments['image'], 0).astype(int)  # load pix-mapped image
    assert image is not None, "No image in the path"

    start_time = time()
    # Main
    frame_of_patterns_ = cross_comp(image)  # returns Pm__

    fline_PPs = 0
    if fline_PPs:  # debug line_PPs_draft
        from line_PPs_draft import *
        frame_PP_ = []

        for y, P_ in enumerate(frame_of_patterns_):
            PPm_, PPd_ = search(P_)
            frame_PP_.append([PPm_, PPd_])

    end_time = time() - start_time
    print(end_time)