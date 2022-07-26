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
    params = list  # PP derivation level, flat, decoded by mapping each m,d to lower-level param
    x0 = int  # redundant to params:
    x = float  # median x
    L = int  # pack in params?
    sign = NoneType  # g-ave + ave-ga sign
    y = int  # for vertical gaps in PP.P__, replace with derP.P.y?
    PP = object  # lower comparand
    _PP = object  # higher comparand
    root = lambda:None  # segment in sub_recursion
    # higher derivatives
    rdn = int  # mrdn, + uprdn if branch overlap?
    uplink_layers = lambda: [[],[]]  # init a layer of dderPs and a layer of match_dderPs
    downlink_layers = lambda: [[],[]]
   # from comp_dx
    fdx = NoneType

class CPPP(CPP, CderPP):

    # draft
    params = list  # derivation layers += derP params per der+, param L is actually Area
    sign = bool
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
    P__ = list  # input  # derP__ = list  # redundant to P__
    seg_levels = lambda: [[[]],[[]]]  # from 1st agg_recursion, seg_levels[0] is seg_t, higher seg_levels are segP_t s
    agg_levels = list  # from 2nd agg_recursion, PP_t = levels[0], from form_PP, before recursion
    rlayers = list  # | mlayers
    dlayers = list  # | alayers
    root = lambda:None  # higher-order segP or PPP

# draft
def agg_recursion(PP_, fiPd):  # compositional recursion per blob.Plevel.
    # P, PP, PPP are relative terms, each may be of any composition order

    comb_levels = [[], []]

    if fiPd: ave_PP = ave_dPP
    else: ave_PP = ave_mPP

    V = sum([sum_named_param(PP.params, "val", fPd=fiPd) for PP in PP_])  # combined across plevels, as is comp_PP_ below
    if V > ave_PP:

        derPP_t = comp_PP_(PP_, fPd=fiPd)  # compare all PPs to the average (centroid) of all other PPs, is generic for lower level
        PPPm_, PPPd_ = form_PPP_t(derPP_t)  # calls individual comp_PP if mPPP > ave_mPPP, converting derPP to CPPP,

        # sections below need further update, sub_recursion_eval is applicable on single PPP only
        # may splice PPs instead of forming PPPs
        sub_recursion_eval(PPPm_)  # fPd=0, rng+  # for derPP_ in PPPs formed by indiv_comp_PP_
        sub_recursion_eval(PPPd_)  # fPd=1, der+

        comb_levels[0].append(PPPm_); comb_levels[1].append(PPPd_)  # pack current level PPP
        m_comb_levels, d_comb_levels = [[],[]], [[],[]]
        if len(PPPm_)>1:
            m_comb_levels = agg_recursion(PPPm_, fiPd=0)
        if len(PPPd_)>1:
            d_comb_levels = agg_recursion(PPPd_, fiPd=1)

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

        if fPd: layers = PP.dlayers
        else:   layers = PP.rlayers

        pre_PPP = CPP(params=deepcopy(PP.params), layers= layers+[PP_], x0=PP.x0, xn=PP.xn, y0=PP.y0, yn=PP.yn )
        summed_params = []  # initialize params
        for compared_PP in compared_PP_:
            sum_levels(summed_params, compared_PP.params, pre_PPP.mptuple, pre_PPP.dptuple)  # accum summed_params over compared_PP_
        # comp_ave- defined pre_PPP inherits PP.params
        pre_PPP.params += [comp_levels(PP.params, summed_params, der_levels=[], fsubder=fsubder)]  # sum_params is now ave_params
        '''
        comp to ave params of compared PPs, form new layer: derivatives of all lower layers, 
        initial 3 layer nesting diagram: https://github.com/assets/52521979/ea6d436a-6c5e-429f-a152-ec89e715ebd6
        '''
        pre_PPPm_.append(copy_P(pre_PPP, Ptype=2))  # Ptype 2 is now PPP, we don't need Ptype 3?
        pre_PPPd_.append(copy_P(pre_PPP, Ptype=2))

    return pre_PPPm_, pre_PPPd_
'''
1st and 2nd layers are single sublayers, the 2nd adds tuple pair nesting. Both are unpacked by func_pairs, not func_layers.  
Multiple sublayers start on the 3rd layer, because it's derived from comparison between two (not one) lower layers. 
4th layer is derived from comparison between 3 lower layers, where the 3rd layer is already nested, etc:
'''

def comp_levels(_levels, levels, der_levels, fsubder=0):  # only for agg_recursion, each param layer may consist of sub_layers

    # recursive unpack of nested param layers, each layer is ptuple pair_layers if from der+
    der_levels += [comp_layers(_levels[0], levels[0], der_layers=[], fsubder=fsubder)]

    # recursive unpack of deeper layers, nested in 3rd and higher layers, if any from agg+, down to nested tuple pairs
    for _level, level in zip(_levels[1:], levels[1:]):  # level = deeper sub_levels, stop if none
        der_levels += [comp_levels(_level, level, der_levels=[], fsubder=fsubder)]

    return der_levels # possibly nested param layers


def sum_named_param(plevel, param_name, fPd):
    psum = 0  # sum of named param across param layers

    if isinstance(plevel, Cptuple):  # 1st level is ptuple, not for angle and aangle elements, if any
        psum += getattr(plevel, param_name)

    elif len(plevel)>1:  # for multi-level params
        if isinstance(plevel[0], Cptuple) and isinstance(plevel[1], Cptuple) :  # 1st level is 2 vertuples, from der+
            psum += getattr(plevel[fPd], param_name)
        else:  # keep unpacking:
            for sub_plevel in plevel:
                psum += sum_named_param(sub_plevel, param_name, fPd)
    else:
        for sub_plevel in plevel:
            psum += sum_named_param(sub_plevel, param_name, fPd)
    return psum


def form_PPP_t(pre_PPP_t):  # form PPs from match-connected segs
    PPP_t = []

    for fPd, pre_PPP_ in enumerate(pre_PPP_t):
        # sort by value of last layer: derivatives of all lower layers:
        pre_PPP_ = sorted(pre_PPP_, key=lambda pre_PPP: sum_named_param(pre_PPP.params[-1], 'val', fPd), reverse=True)  # descending order
        PPP_ = []
        for i, pre_PPP in enumerate(pre_PPP_):
            pre_PPP_val = sum_named_param(pre_PPP.params, 'val', fPd=fPd)
            for param_layer in pre_PPP.params:  # may need recursive unpack here
                pre_PPP.rdn += sum_named_param(param_layer, 'val', fPd=fPd)> sum_named_param(param_layer, 'val', fPd=1-fPd)
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
    rng = sum_named_param(pre_PPP.params[-1], 'val', fPd)/ 3  # 3: ave per rel_rng+=1, actual rng is Euclidean distance:

    for PP in pre_PPP.layers[-1]:  # 1/1 comparison between _PP and other PPs within rng
        derPP = CderPP(PP=PP)
        _area = sum_named_param(pre_PPP.params[0], 'L', fPd)
        area = sum_named_param(PP.params[0], 'L', fPd)
        dx = ((pre_PPP.xn-pre_PPP.x0)/2)/_area -((PP.xn-PP.x0)/2)/area
        dy = pre_PPP.y/_area - PP.y/area
        distance = np.hypot(dy, dx)  # Euclidean distance between PP centroids
        _val = sum_named_param(pre_PPP.params[-1], 'val', fPd)
        val = sum_named_param(PP.params[-1], 'val', fPd)

        if distance / ((_val+val)/2) < rng:  # distance relative to value, vs. area?
            params = comp_levels(pre_PPP.params, PP.params, der_levels=[])
            pre_PPP.downlink_layers += [params]
            PP.uplink_layers += [params]
            derPP.params = params
            derPP_ += [derPP]

    pre_PPP.P__ = derPP_

    for i, _derPP in enumerate(derPP_):  # cluster derPPs into PPPs by connectivity, overwrite derPP[i]
        if sum_named_param(_derPP.params[-1], 'val', fPd):
            PPP = CPPP(params=deepcopy(_derPP.params), layers=[_derPP.PP])
            PPP.accum_from(_derPP)  # initialization
            _derPP.root = PPP
            for derPP in derPP_[i+1:]:
                if not derPP.PP.root:  # not sure this is needed
                    Val = sum_named_param(derPP.params[-1], 'val', fPd)
                    if Val:  # positive and not in PPP yet
                        PPP.layers.append(derPP)  # multiple composition orders
                        PPP.accum_from(_derPP)
                        derPP.root = PPP
                    elif Val > ave*len(derPP.params)-1:
                         # splice PP and their segs
                         pass
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