import sys
import numpy as np
from itertools import zip_longest
from copy import deepcopy, copy
from class_cluster import ClusterStructure, NoneType, comp_param, Cdert
import math as math
from comp_slice import *
'''
Blob edges may be represented by higher-composition PPPs, etc., if top param-layer match,
in combination with spliced lower-composition PPs, etc, if only lower param-layers match.
This may form closed edge patterns around flat blobs, which defines stable objects.   
'''

# agg-recursive versions should be more complex?
class CderPP(ClusterStructure):  # tuple of derivatives in PP uplink_ or downlink_, PP can also be PPP, etc.

    # draft
    players = list  # PP derivation level, flat, decoded by mapping each m,d to lower-level param
    mplayer = lambda: [None]  # list of ptuples in current derivation layer per fork
    dplayer = lambda: [None]
    mval = float  # summed player vals, both are signed, PP sign by fPds[-1]
    dval = float
    box = list
    _PP = object  # higher comparand
    PP = object  # lower comparand
    root = lambda:None  # segment in sub_recursion
    # higher derivatives
    rdn = int  # mrdn, + uprdn if branch overlap?
    uplink_layers = lambda: [[],[]]  # init a layer of dderPs and a layer of match_dderPs
    downlink_layers = lambda: [[],[]]
   # from comp_dx
    fdx = NoneType

class CPPP(CPP, CderPP):

    players = list  # max n ptuples in layer = n ptuples in all lower layers: 1, 1, 2, 4, 8...
    mplayer = lambda: [None]  # list of ptuples in current derivation layer per fork
    dplayer = lambda: [None]
    rng = lambda: 1  # rng starts with 1
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs
    Rdn = int  # for accumulation only
    nP = int  # len 2D derP__ in levels[0][fPd]?  ly = len(derP__), also x, y?
    uplink_layers = lambda: [[],[]]
    downlink_layers = lambda: [[],[]]
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    box = list  # for visualization only, original box before flipping
    mask__ = bool
    P__ = list  # input  # input, includes derPs, same as agg_levels[-1]?
    rlayers = list  # | mlayers
    dlayers = list  # | alayers
    seg_levels = lambda: [[]]  # 1st agg_recursion: segs ) segPs(r,d)) segPPs(r,d)..
    agg_levels = lambda: [[]]  # 2nd agg_recursion: PPs ) PPPs PPPPs..


    root = lambda:None  # higher-order segP or PPP

# draft
def agg_recursion(dir_blob, PP_, fPd):  # compositional recursion per blob.Plevel.
    # P, PP, PPP are relative terms, each may be of any composition order

    comb_levels = [[], []]
    ave_PP = ave_dPP if fPd else ave_mPP
    V = sum([PP.dval for PP in PP_]) if fPd else sum([PP.mval for PP in PP_])
    if V > ave_PP:

        derPP_t = comp_PP_(PP_, fPd=fPd)  # compare all PPs to the average (centroid) of all other PPs, is generic for lower level
        PPPm_, PPPd_ = form_PPP_t(derPP_t)  # calls individual comp_PP if mPPP > ave_mPPP, converting derPP to CPPP,
        # may splice PPs instead of forming PPPs
        dir_blob.rlayers = [PPPm_]; dir_blob.dlayers = [PPPd_]
        # use sub_recursion_eval, agg_recursion_eval instead:
        if PPPm_:
            sub_recursion(PPPm_, fPd=0)
        if PPPd_:
            sub_recursion(PPPd_, fPd=1)

        comb_levels[0].append(PPPm_); comb_levels[1].append(PPPd_)  # pack current level PPP
        m_comb_levels, d_comb_levels = [[],[]], [[],[]]

        if len(PPPm_)>1:  # add eval
            m_comb_levels = agg_recursion(PPPm_, fPd=0)
        if len(PPPd_)>1:
            d_comb_levels = agg_recursion(PPPd_, fPd=1)

        # combine sub_PPm_s and sub_PPd_s from each layer:
        for m_sub_PPPm_, d_sub_PPPm_ in zip_longest(m_comb_levels[0], d_comb_levels[0], fillvalue=[]):
            comb_levels[0] += [m_sub_PPPm_ + d_sub_PPPm_]
        for m_sub_PPPd_, d_sub_PPPd_ in zip_longest(m_comb_levels[1], d_comb_levels[1], fillvalue=[]):
            comb_levels[1] += [m_sub_PPPd_ + d_sub_PPPd_]

    return comb_levels

'''
- Compare each PP to the average (centroid) of all other PPs in PP_, or maximal cartesian distance, forming derPPs.  
- Select above-average derPPs as PPPs, representing summed derivatives over comp range, overlapping between PPPs.
Full overlap, no selection for derPPs per PPP. 
Selection and variable rdn per derPP requires iterative centroid clustering per PPP.  
This will probably be done in blob-level agg_recursion, it seems too complex for edge tracing, mostly contiguous?
'''

def comp_PP_(PP_, fsubder=0, fPd=0):  # PP can also be PPP, etc.

    pre_PPPm_, pre_PPPd_ = [],[]

    for PP in PP_:
        compared_PP_ = copy(PP_)  # shallow copy
        compared_PP_.remove(PP)

        Players = []  # initialize params

        for compared_PP in compared_PP_:  # accum summed_params over compared_PP_:
            sum_players(Players, compared_PP.players)

        mplayer, dplayer = comp_players(PP.players, Players)  # sum_params is now ave_params
        # comp to ave params of compared PPs, pre_PPP inherits PP.params, forms new player: derivatives of all lower layers,
        # initial 3 layer nesting diagram: https://github.com/assets/52521979/ea6d436a-6c5e-429f-a152-ec89e715ebd6

        pre_PPP = CPP(players=deepcopy(PP.players) + [dplayer if fPd else mplayer],
                      fPds=deepcopy(PP.fPds)+[fPd], x0=PP.x0, xn=PP.xn, y0=PP.y0, yn=PP.yn,
                      P__ = compared_PP_)  # temporary, will be replaced with derPP later

        pre_PPPm_.append(copy_P(pre_PPP))
        pre_PPPd_.append(copy_P(pre_PPP))

    return pre_PPPm_, pre_PPPd_
'''
1st and 2nd layers are single sublayers, the 2nd adds tuple pair nesting. Both are unpacked by func_pairs, not func_layers.  
Multiple sublayers start on the 3rd layer, because it's derived from comparison between two (not one) lower layers. 
4th layer is derived from comparison between 3 lower layers, where the 3rd layer is already nested, etc:
'''

# looks like this may not needed now
def comp_levels(_levels, levels, der_levels, fsubder=0):  # only for agg_recursion, each param layer may consist of sub_layers

    # recursive unpack of nested param layers, each layer is ptuple pair_layers if from der+
    der_levels += [comp_players(_levels[0], levels[0])]

    # recursive unpack of deeper layers, nested in 3rd and higher layers, if any from agg+, down to nested tuple pairs
    for _level, level in zip(_levels[1:], levels[1:]):  # level = deeper sub_levels, stop if none
        der_levels += [comp_levels(_level, level, der_levels=[], fsubder=fsubder)]

    return der_levels # possibly nested param layers

# old:
def sum_levels(Params, params):  # Capitalized names for sums, as comp_levels but no separate der_layers to return

    if Params:
        sum_players(Params[0], params[0])  # recursive unpack of nested ptuple layers, if any from der+
    else:
        Params.append(deepcopy(params[0]))  # no need to sum

    for Level, level in zip_longest(Params[1:], params[1:], fillvalue=[]):
        if Level and level:
            sum_levels(Level, level)  # recursive unpack of higher levels, if any from agg+ and nested with sub_levels
        elif level:
            Params.append(deepcopy(level))  # no need to sum


def form_PPP_t(pre_PPP_t):  # form PPs from match-connected segs
    PPP_t = []

    for fPd, pre_PPP_ in enumerate(pre_PPP_t):
        # sort by value of last layer: derivatives of all lower layers:
        if fPd: pre_PPP_ = sorted(pre_PPP_, key=lambda pre_PPP: pre_PPP.dval, reverse=True)  # descending order
        else:   pre_PPP_ = sorted(pre_PPP_, key=lambda pre_PPP: pre_PPP.mval, reverse=True)  # descending order

        PPP_ = []
        for i, pre_PPP in enumerate(pre_PPP_):
            if fPd: pre_PPP_val = pre_PPP.dval
            else:   pre_PPP_val = pre_PPP.mval

            for mptuple, dptuple in zip(pre_PPP.mplayer, pre_PPP.dplayer):
                if mptuple and dptuple:  # could be None
                    if fPd: pre_PPP.rdn += dptuple.val > mptuple.val
                    else:   pre_PPP.rdn += mptuple.val > dptuple.val
            '''
            for param_layer in pre_PPP.params:  # may need recursive unpack here
                pre_PPP.rdn += sum_named_param(param_layer, 'val', fPd=fPd)> sum_named_param(param_layer, 'val', fPd=1-fPd)
            '''

            ave = vaves[fPd] * pre_PPP.rdn * (i+1)  # derPP is redundant to higher-value previous derPPs in derPP_
            if pre_PPP_val > ave:
                PPP_ += [pre_PPP]  # base derPP and PPP is CPP
                if pre_PPP_val > ave*10:
                    indiv_comp_PP_(pre_PPP, fPd)  # derPP is converted from CPP to CPPP
            else:
                break  # ignore below-ave PPs
        PPP_t.append(PPP_)
    return PPP_t

def indiv_comp_PP_(pre_PPP, fPd):  # 1-to-1 comp, _PP is converted from CPP to higher-composition CPPP

    derPP_ = []
    rng = 1  # default value of rng, pre_PPP.dplayer[-1] could be None
    if fPd:  # use int to get minimum value of 1
        if pre_PPP.dplayer[-1]: rng = int(pre_PPP.dplayer[-1].val/ 3)  # 3: ave per rel_rng+=1, actual rng is Euclidean distance:
    else:
        if pre_PPP.mplayer[-1]: rng = int(pre_PPP.mplayer[-1].val/ 3)

    for PP in pre_PPP.P__:  # 1/1 comparison between _PP and other PPs within rng
        derPP = CderPP(PP=PP)
        _area = pre_PPP.players[0][0].L
        area = PP.players[0][0].L
        dx = ((pre_PPP.xn-pre_PPP.x0)/2)/_area -((PP.xn-PP.x0)/2)/area
        dy = pre_PPP.y/_area - PP.y/area
        distance = np.hypot(dy, dx)  # Euclidean distance between PP centroids

        _val = 1; val = 1
        if fPd:
            if pre_PPP.dplayer[-1]: _val = pre_PPP.dplayer[-1].val  # dplayer[-1] is not None
            if PP.dplayer[-1]: val = PP.dplayer[-1].val
        else:
            if pre_PPP.mplayer[-1]: _val = pre_PPP.mplayer[-1].val  # mplayer[-1] is not None
            if PP.mplayer[-1]: val = PP.mplayer[-1].val
        if distance / ((_val+val)/2) < rng:  # distance relative to value, vs. area?
            mplayer, dplayer = comp_players(pre_PPP.players, PP.players)
            if fPd: player = dplayer
            else:   player = mplayer
            # not sure below:
            pre_PPP.downlink_layers += [player]
            PP.uplink_layers += [player]
            derPP.players = player

            derPP_ += [derPP]

    pre_PPP.P__ = derPP_
    for i, _derPP in enumerate(derPP_):  # cluster derPPs into PPPs by connectivity, overwrite derPP[i]
        val = 0
        if fPd:
            if _derPP.dplayer[-1]: val = _derPP.dplayer[-1].val
        else:
            if _derPP.mplayer[-1]: val = _derPP.mplayer[-1].val
        if val:
            PPP = CPPP(players=deepcopy(_derPP.players))  # not sure if we need still need layers?
            PPP.accum_from(_derPP)  # initialization
            _derPP.root = PPP
            for derPP in derPP_[i+1:]:
                if not derPP.PP.root:  # not sure this is needed
                    Val = 0
                    if fPd:
                        if derPP.dplayer[-1]: Val = derPP.dplayer[-1].val
                    else:
                        if derPP.mplayer[-1]: Val = derPP.mplayer[-1].val
                    if Val:  # positive and not in PPP yet
                        PPP.layers.append(derPP)  # multiple composition orders
                        PPP.accum_from(_derPP)
                        derPP.root = PPP
                    elif Val > ave*len(derPP.players)-1:
                         # splice PP and their segs
                         pass
    '''
    if derPP.match params[-1]: form PPP
    elif derPP.match params[:-1]: splice PPs and their segs? 
    '''
'''
if derPP.match params[-1]: form PPP
elif derPP.match params[:-1]: splice PPs and their segs? 
'''

def form_segPPP_root(PP_, root_rdn, fPd):  # not sure about form_seg_root

    for PP in PP_:
        link_eval(PP.uplink_layers, fPd)
        link_eval(PP.downlink_layers, fPd)

    for PP in PP_:
        form_segPPP_(PP)

def form_segPPP_(PP):
    pass

# pending update
def splice_segs(seg_):  # in 1st run of agg_recursion
    pass

# draft, splice 2 PPs for now
def splice_PPs(PP_, frng):  # splice select PP pairs if der+ or triplets if rng+

    spliced_PP_ = []
    while PP_:
        _PP = PP_.pop(0)  # pop PP, so that we can differentiate between tested and untested PPs
        tested_segs = []  # new segs may be added during splicing, their links also need to be checked for splicing
        _segs = _PP.seg_levels[0]

        while _segs:
            _seg = _segs.pop(0)
            _avg_y = sum([P.y for P in _seg.P__])/len(_seg.P__)  # y centroid for _seg

            for link in _seg.uplink_layers[1] + _seg.downlink_layers[1]:
                seg = link.P.root  # missing link of current seg

                if seg.root is not _PP:  # after merging multiple links may have the same PP
                    avg_y = sum([P.y for P in seg.P__])/len(seg.P__)  # y centroid for seg

                    # test for y distance (temporary)
                    if (_avg_y - avg_y) < ave_splice:
                        if seg.root in PP_: PP_.remove(seg.root)  # remove merged PP
                        elif seg.root in spliced_PP_: spliced_PP_.remove(seg.root)
                        # splice _seg's PP with seg's PP
                        merge_PP(_PP, seg.root)

            tested_segs += [_seg]  # pack tested _seg
        _PP.seg_levels[0] = tested_segs
        spliced_PP_ += [_PP]

    return spliced_PP_


def merge_PP(_PP, PP, fPd):  # only for PP splicing

    for seg in PP.seg_levels[fPd][-1]:  # merge PP_segs into _PP:
        accum_PP(_PP, seg, fPd)
        _PP.seg_levels[fPd][-1] += [seg]

    # merge uplinks and downlinks
    for uplink in PP.uplink_layers:
        if uplink not in _PP.uplink_layers:
            _PP.uplink_layers += [uplink]
    for downlink in PP.downlink_layers:
        if downlink not in _PP.downlink_layers:
            _PP.downlink_layers += [downlink]