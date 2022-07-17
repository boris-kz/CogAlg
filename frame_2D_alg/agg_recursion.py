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
    params = list  # PP derivation layer, flat, decoded by mapping each m,d to lower-layer param
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
    PPP_levels = list  # from 2nd agg_recursion, PP_t = levels[0], from form_PP, before recursion
    layers = list  # from sub_recursion, each is derP_t
    root = lambda:None  # higher-order segP or PPP


def agg_recursion(blob, fseg):  # compositional recursion per blob.Plevel. P, PP, PPP are relative terms, each may be of any composition order

    if fseg: PP_t = [blob.seg_levels[0][-1], blob.seg_levels[1][-1]]   # blob is actually PP, recursion forms segP_t, seg_PP_t, etc.
    else:    PP_t = [blob.levels[0][-1], blob.levels[1][-1]]  # input-level composition Ps, initially PPs

    PPP_t = []  # next-level composition Ps, initially PPPs  # for fiPd, PP_ in enumerate(PP_t): fiPd = fiPd % 2  # dir_blob.M += PP.M += derP.m
    n_extended = 0

    for fiPd, PP_ in enumerate(PP_t):   # fiPd = fiPd % 2
        if fiPd: ave_PP = ave_dPP
        else:    ave_PP = ave_mPP
        if fseg: M = sum_named_param(blob.params, "val", fPd=0) - ave  # actually ave * nptuples in params
        else:    M = ave-abs(blob.G)  # if M > ave_PP * blob.rdn and len(PP_)>1:  # >=2 comparands

        if len(PP_)>1:
            n_extended += 1
            derPP_t = comp_PP_(PP_)  # compare all PPs to the average (centroid) of all other PPs, is generic for lower level
            PPP_t = form_PPP_t(derPP_t)
            # call individual comp_PP if mPPP > ave_mPPP, converting derPP to CPPP
            splice_PPs(PPP_t)  # for initial PPs only: if PP is CPP?
            sub_recursion_eval(PPP_t[0])  # fPd=0, rng+
            sub_recursion_eval(PPP_t[1])  # fPd=1, der+
        else:
            PPP_t += [[], []]  # replace with neg PPPs?

    if fseg: blob.seg_levels += [PPP_t]  # new level of segPs
    else:    blob.levels += [PPP_t]  # levels of dir_blob are Plevels

    if n_extended/len(PP_t) > 0.5:  # mean ratio of extended PPs
        agg_recursion(blob, fseg)
'''
- Compare each PP to the average (centroid) of all other PPs in PP_, or maximal cartesian distance, forming derPPs.  
- Select above-average derPPs as PPPs, representing summed derivatives over comp range, overlapping between PPPs.
'''

def comp_PP_(PP_, fsubder=0):  # PP can also be PPP, etc.

    pre_PPPm_, pre_PPPd_ = [],[]

    for PP in PP_:
        compared_PP_ = copy(PP_)  # shallow copy
        compared_PP_.remove(PP)
        summed_params = [[]]  # initialize params[0]
        for compared_PP in compared_PP_:
            sum_layers(summed_params, compared_PP.params)  # accum summed_params over compared_PP_

        pre_PPP = CPP(params=deepcopy(PP.params), layers= PP.layers+[PP_])  # comp_ave- defined pre_PPP inherits PP.params
        pre_PPP.params += [comp_layers(PP.params, summed_params, der_layers=[], fsubder=fsubder)]  # sum_params is now ave_params
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

def comp_layers(_layers, layers, der_layers, fsubder=0):  # only for agg_recursion, each param layer may consist of sub_layers

    # recursive unpack of nested param layers, each layer is ptuple pair_layers if from der+
    der_layers += [comp_pair_layers(_layers[0], layers[0], der_pair_layers=[], fsubder=fsubder)]

    # recursive unpack of deeper layers, nested in 3rd and higher layers, if any from agg+, down to nested tuple pairs
    for _layer, layer in zip(_layers[1:], layers[1:]):  # layer = deeper sub_layers, stop if none
        der_layers += [comp_layers(_layer, layer, der_layers=[], fsubder=fsubder)]

    return der_layers # possibly nested param layers


def sum_layers(Params, params):  # Capitalized names for sums, as comp_layers but no separate der_layers to return

    if Params[0] and params[0]:
        sum_pair_layers(Params[0], params[0])  # recursive unpack of nested ptuple pair_layers, if any from der+
    elif params[0]:
        Params[0] = deepcopy(params[0])  # no need to sum

    for Layer, layer in zip_longest(Params[1:], params[1:], fillvalue=[]):
        if Layer and layer:
            sum_layers(Layer, layer)  # recursive unpack of higher layers, if any from agg+ and nested with sub_layers
        elif layer:
            Params.append( deepcopy(layer))


def sum_named_param(p_layer, param_name, fPd):
    psum = 0  # sum of named param across param layers

    if isinstance(p_layer, Cptuple):  # 1st layer is ptuple, not for angle and aangle elements, if any
        psum += getattr(p_layer, param_name)
    elif len(p_layer)>1:  # for multiple layer params
        if isinstance(p_layer[0], Cptuple) and isinstance(p_layer[1], Cptuple) :  # 1st layer is 2 vertuples, from der+
            psum += getattr(p_layer[fPd], param_name)
        else:  # keep unpacking:
            for sub_p_layer in p_layer:
                psum += sum_named_param(sub_p_layer, param_name, fPd)
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
        dx = pre_PPP.x/_area - PP.x/area
        dy = pre_PPP.y/_area - PP.y/area
        distance = np.hypot(dy, dx)  # Euclidean distance between PP centroids
        _val = sum_named_param(pre_PPP.params[-1], 'val', fPd)
        val = sum_named_param(PP.params[-1], 'val', fPd)

        if distance / ((_val+val)/2) < rng:  # distance relative to value, vs. area?
            params = comp_layers(pre_PPP.params, PP.params, der_layers=[])
            pre_PPP.downlink_layers += [params]
            PP.uplink_layers += [params]
            derPP.params = params
            derPP_ += [derPP]

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

                if seg.root is not _PP:  # this may occur after the merging where multiple links are having same PP
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