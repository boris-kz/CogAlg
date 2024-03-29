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
    mplayer = list  # list of ptuples in current derivation layer per fork
    dplayer = list
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
    mplayer = list  # list of ptuples in current derivation layer per fork
    dplayer = list
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
def agg_recursion(dir_blob, PP_, fPd, fseg=0):  # compositional recursion per blob.Plevel; P, PP, PPP are relative to each other

    comb_levels = []
    ave_PP = ave_dPP if fPd else ave_mPP
    V = sum([PP.dval for PP in PP_]) if fPd else sum([PP.mval for PP in PP_])
    if V > ave_PP:
        # cross-comp -> bilateral match assign list per PP, re-clustering by rdn match to centroids: ave PPP params
        PPPm_, PPPd_ = comp_PP_(PP_)  # cross-comp all PPs within rng,

        comp_centroid(PPPm_)  # may splice PPs instead of forming PPPs
        comp_centroid(PPPd_)

        sub_recursion_eval(PPPm_, fPd=0)  # test within PP_ for each PPP (PP_ is PPP.P__)
        sub_recursion_eval(PPPd_, fPd=1)
        agg_recursion_eval(PPPm_, dir_blob, fPd=0, fseg=fseg)  # test within PPP_
        agg_recursion_eval(PPPd_, dir_blob, fPd=1, fseg=fseg)

        for PPP_ in PPPm_, PPPd_:
            for PPP in PPP_:
                for i, (comb_level, level) in enumerate(zip_longest(comb_levels, PPP.agg_levels, fillvalue=[])):
                    if level:
                        if i > len(comb_levels)-1: comb_levels += [[level]]  # add new level
                        else: comb_levels[i] += [level]  # append existing layer

        comb_levels = [[PPPm_, PPPd_]] + comb_levels

    return comb_levels
'''
- Compare each PP to the average (centroid) of all other PPs in PP_, or maximal cartesian distance, forming derPPs.  
- Select above-average derPPs as PPPs, representing summed derivatives over comp range, overlapping between PPPs.
Full overlap, no selection for derPPs per PPP. 
Selection and variable rdn per derPP requires iterative centroid clustering per PPP.  
This will probably be done in blob-level agg_recursion, it seems too complex for edge tracing, mostly contiguous?
'''

def comp_PP_(iPP_):  # rng cross-comp, draft

    PPP_t = []
    for fPd in 0,1:
        PP_ = [copy_P(PP) for PP in iPP_]  # we need 2 different set of PP_ for each fPd
        PPP_ = []
        iPPP_ = [CPPP( PP=PP, players=deepcopy(PP.players), fPds=deepcopy(PP.fPds)+[fPd], x0=PP.x0, xn=PP.xn, y0=PP.y0, yn=PP.yn) for PP in PP_]

        while PP_:  # compare _PP to all other PPs within rng
            _PP, _PPP = PP_.pop(), iPPP_.pop()
            _PP.root = _PPP  # actual root which is having same index with PP?
            _PP.roots.append(_PPP)  # not sure
            for PPP, PP in zip(iPPP_, PP_):
                # all possible comparands in dy<rng, with incremental y, accum derPPs in PPPs
                area = PP.players[0][0].L; _area = _PP.players[0][0].L  # not sure
                dx = ((_PP.xn-_PP.x0)/2)/_area -((PP.xn-PP.x0)/2)/area
                dy = _PP.y/_area - PP.y/area
                distance = np.hypot(dy, dx)  # Euclidean distance between PP centroids

                if distance * ((_PP.mval+PP.mval)/2 / ave_mPP) <= 3:  # ave_rng
                    # comp PPs:
                    mplayer, dplayer = comp_players(_PP.players, PP.players)
                    mval = sum([ptuple.val for ptuple in mplayer])
                    derPP = CderPP(players = deepcopy(_PP.players), mplayer=mplayer, dplayer=dplayer, _PP=_PP, PP=PP, mval=mval)
                    if mval > ave_mPP:
                        _PP.roots.append(PPP)  # not sure
                        derPP.fin = derPP._fin = 1  # PPs match, sum derPP in both PPP and _PPP, m fork only:
                        sum_players(_PPP.players, derPP.players + [derPP.mplayer])
                        sum_players(PPP.players, derPP.players + [derPP.mplayer])
                    _PPP.derPP_ += [derPP]
                    PPP.derPP_ += [derPP]  # bilateral derPP assign regardless of sign, to re-eval in centroid clustering
                    '''
                    if derPP.match params[-1]: form PPP
                    elif derPP.match params[:-1]: splice PPs and their segs? 
                    '''
            PPP_.append(_PPP)
        PPP_t += [PPP_]
    return PPP_t

def comp_PP_centroid(PP_, fsubder=0, fPd=0):  # PP can also be PPP, etc.

    pre_PPPm_, pre_PPPd_ = [],[]

    for PP in PP_:
        compared_PP_ = copy(PP_)
        compared_PP_.remove(PP)
        Players = []  # initialize params

        for compared_PP in compared_PP_:  # accum summed_params over compared_PP_:
            sum_players(Players, compared_PP.players)

        mplayer, dplayer = comp_players(PP.players, Players)  # sum_params is now ave_params

        pre_PPP = CPPP(players=deepcopy(PP.players) + [dplayer if fPd else mplayer],
                      fPds=deepcopy(PP.fPds)+[fPd], x0=PP.x0, xn=PP.xn, y0=PP.y0, yn=PP.yn,
                      P__ = compared_PP_)  # temporary P__, will be replaced with derPP later

        pre_PPPm_.append(copy_P(pre_PPP))
        pre_PPPd_.append(copy_P(pre_PPP))

    return pre_PPPm_, pre_PPPd_
'''
1st and 2nd layers are single sublayers, the 2nd adds tuple pair nesting. Both are unpacked by func_pairs, not func_layers.  
Multiple sublayers start on the 3rd layer, because it's derived from comparison between two (not one) lower layers. 
4th layer is derived from comparison between 3 lower layers, where the 3rd layer is already nested, etc:
'''

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

def sub_recursion_eval(PP_, fPd):  # for PP or dir_blob

    from agg_recursion import agg_recursion
    for PP in PP_:

        if fPd: ave_PP = ave_dPP; val = PP.dval; alt_val = PP.mval
        else:   ave_PP = ave_mPP; val = PP.mval; alt_val = PP.dval
        ave =   ave_PP * (3 + PP.rdn + 1 + (alt_val > val))  # fork rdn, 3: agg_coef

        if val > ave and len(PP.P__) > ave_nsub:
            sub_recursion(PP, fPd)  # comp_P_der | comp_P_rng in PPs -> param_layer, sub_PPs
            ave*=2  # 1+PP.rdn incr

        if val > ave and len(PP.seg_levels[-1]) > ave_nsub:
            PP.seg_levels += agg_recursion(PP, PP.seg_levels[-1], fPd, fseg=1)
            # rng comp, centroid clustering

def sub_recursion(PP, fPd):  # evaluate each PP for rng+ and der+

    P__  = [P_ for P_ in reversed(PP.P__)]  # revert to top down
    if fPd: Pm__, Pd__ = comp_P_der(P__)  # returns top-down
    else:   Pm__, Pd__ = comp_P_rng(P__, PP.rng + 1)

    PP.rdn += 2  # two-fork rdn, priority is not known?
    sub_segm_ = form_seg_root(Pm__, fPd=0, fPds=PP.fPds)
    sub_segd_ = form_seg_root(Pd__, fPd=1, fPds=PP.fPds)  # returns bottom-up
    sub_PPm_, sub_PPd_ = form_PP_root((sub_segm_, sub_segd_), PP.rdn + 1)  # PP is parameterized graph of linked segs

    sub_recursion_eval(sub_PPm_, fPd=0)  # add rlayers, dlayers, seg_levels to select sub_PPs
    sub_recursion_eval(sub_PPd_, fPd=1)

    comb_rlayers, comb_dlayers = [],[]
    for fd, (comb_layers, sub_PP_) in enumerate( zip( (comb_rlayers, comb_dlayers), (sub_PPm_, sub_PPd_))):

        for sub_PP in sub_PP_:  # splice deeper layers between sub_PPs into comb_layers:
            for i, (comb_layer, PP_layer) in enumerate(zip_longest(comb_layers, sub_PP.dlayers if fd else sub_PP.dlayers, fillvalue=[])):

                if i > len(comb_layers) - 1: comb_layers += [PP_layer]  # add new r|d layer
                else: comb_layers[i] += PP_layer  # splice r|d PP layer into existing layer

        if fPd: PP.dlayers = [sub_PP_] + comb_layers
        else:   PP.rlayers = [sub_PP_] + comb_layers


def accum_ptuple(Ptuple, ptuple, fneg=0):  # lataple or vertuple

    if fneg:  # subtraction
        for param_name in Ptuple.numeric_params:
            new_value = getattr(Ptuple, param_name) - getattr(ptuple, param_name)
            setattr(Ptuple, param_name, new_value)
    else:  # accumulation
        Ptuple.accum_from(ptuple, excluded=["angle", "aangle"])

    fAngle = isinstance(Ptuple.angle, list)
    fangle = isinstance(ptuple.angle, list)

    if fAngle and fangle:  # both are latuples:  # not be needed if ptuples are always same-type
        for i, param in enumerate(ptuple.angle):
            if fneg:
                Ptuple.angle[i] -= param  # always in vector representation
            else:
                Ptuple.angle[i] += param
        for i, param in enumerate(ptuple.aangle):
            if fneg:
                Ptuple.aangle[i] -= param
            else:
                Ptuple.aangle[i] += param

    elif not fAngle and not fangle:  # both are vertuples:
        if fneg:
            Ptuple.angle -= ptuple.angle
            Ptuple.aangle -= ptuple.aangle
        else:
            Ptuple.angle += ptuple.angle
            Ptuple.aangle += ptuple.aangle

def accum_ptuple(Ptuple, ptuple, fneg=0):  # lataple or vertuple

    for i, (Param, param) in enumerate( zip(Ptuple[:-2], ptuple[:-2])):
        # (x, L, I, M, Ma, angle, aangle, n, val, G, Ga)
        # n, val from all levels?
        if isinstance(Param, list):  # angle or aangle, same-type ptuples
            for j, (Par, par) in enumerate( zip(Param, param)):
                if fneg: Param[j] = Par - par
                else:    Param[j] = Par + par
        else:
            if fneg: Ptuple[i] = Param - param
            else:    Ptuple[i] = Param + param

mplayer = lambda: [None]  # list of ptuples in current derivation layer per fork, [None] for single-P seg/PPs
dplayer = lambda: [None]  # not needed: player mvals, dvals, map to implicit sub_layers in m|dplayer


def agg_recursion(dir_blob, PP_, fPd, fseg=0):  # compositional recursion per blob.Plevel; P, PP, PPP are relative to each other

    comb_levels = []
    ave_PP = ave_dPP if fPd else ave_mPP
    V = sum([PP.dval for PP in PP_]) if fPd else sum([PP.mval for PP in PP_])
    if V > ave_PP:
        rng = V / ave_PP  # cross-comp, bilateral match assign list per PP, re-clustering by match to PPP centroids

        PPP_ = comp_PP_(PP_, rng, fPd)  # cross-comp all PPs within rng, same PPP_ for both forks
        PPPm_, PPPd_ = comp_centroid(PPP_, fPd)  # if top level miss, lower levels match: splice PPs vs form PPPs
        # add separate centroid clustering forks
        sub_recursion_agg(PPPm_, fPd=0)  # reform PP_ per PPP for rng+
        sub_recursion_agg(PPPd_, fPd=1)  # reform PP_ per PPP for der+
        agg_recursion_eval(PPPm_, dir_blob, fPd=0)
        agg_recursion_eval(PPPd_, dir_blob, fPd=1)

        for PPP_ in PPPm_, PPPd_:
            for PPP in PPPm_:
                for i, (comb_level, level) in enumerate(zip_longest(comb_levels, PPP.agg_levels, fillvalue=[])):
                    if level:
                        if i > len(comb_levels) - 1:
                            comb_levels += [[level]]  # add new level
                        else:
                            comb_levels[i] += [level]  # append existing layer
            sub_comb_levels = [PPP_] + comb_levels
        comb_levels += [sub_comb_levels]  # not sure

    return comb_levels


# draft
def sub_recursion_agg(PPP_, fPd):  # rng+: extend PP_ per PPP, der+: replace PP with derPP in PPt

    comb_layers = []
    for PPP in PPP_:
        val = PPP.dval if fPd else PPP.mval
        ave = ave_dPP if fPd else ave_mPP

        if val > ave and len(PPP.PP_) > ave_nsub:
            PP_ = [PPt[0] for PPt in PPP.PP_]

            sub_PPP_ = comp_PP_(PP_, rng=int(val / ave), fPd=fPd)  # cross-comp all PPs within rng
            comp_centroid(sub_PPP_, fPd)  # may splice PPs instead of forming PPPs

            sublayers = PPP.dlayers if fPd else PPP.rlayers
            sublayers += sub_recursion_agg_(sub_PPP_, fPd=fPd)

            for i, (comb_layer, PPP_layer) in enumerate(zip_longest(comb_layers, PPP.dlayers if fPd else PPP.rlayers, fillvalue=[])):
                if PPP_layer:
                    if i > len(comb_layers) - 1:
                        comb_layers += [PPP_layer]  # add new r|d layer
                    else:
                        comb_layers[i] += PPP_layer  # splice r|d PP layer into existing layer

    return comb_layers


def agg_recursion_eval(PP_, root, fPd, fseg=0):  # from agg_recursion per fork, adds agg_level to agg_PP or dir_blob

    if isinstance(root, CPP):
        dval = root.dval; mval = root.mval
    else:
        dval = root.G; mval = root.M
    if fPd:
        ave_PP = ave_dPP; val = dval; alt_val = mval
    else:
        ave_PP = ave_mPP; val = mval; alt_val = dval
    ave = ave_PP * (3 + root.rdn + 1 + (alt_val > val))  # fork rdn per PP, 3: agg_coef

    if val > ave and len(PP_) > ave_nsub:
        root.levels += [agg_recursion(root, root.levels[-1], fPd, fseg)]


def comp_PP_(PP_, rng, fPd):  # rng cross-comp, draft

    PPP_ = []
    iPPP_ = [CPPP(PP=PP, players=deepcopy(PP.players), fPds=deepcopy(PP.fPds), x0=PP.x0, xn=PP.xn, y0=PP.y0, yn=PP.yn) for PP in PP_]

    while PP_:  # compare _PP to all other PPs within rng
        _PP, _PPP = PP_.pop(), iPPP_.pop()
        _PP.root = _PPP
        for PPP, PP in zip(iPPP_[:rng], PP_[:rng]):  # need to revise for partial rng

            # all possible comparands in dy<rng, with incremental y, accum derPPs in PPPs
            area = PP.players[0][0].L;
            _area = _PP.players[0][0].L  # not sure
            dx = ((_PP.xn - _PP.x0) / 2) / _area - ((PP.xn - PP.x0) / 2) / area
            dy = _PP.y / _area - PP.y / area
            distance = np.hypot(dy, dx)  # Euclidean distance between PP centroids

            if fPd:
                val = ((_PP.dval + PP.dval) / 2 / ave_dPP)
            else:
                val = ((_PP.mval + PP.mval) / 2 / ave_mPP)
            if distance * val <= 3:  # ave_rng
                # comp PPs:
                mplayer, dplayer = comp_players(_PP.players, PP.players, _PP.fPds, PP.fPds)
                mval = sum([mtuple.val for mtuple in mplayer])
                dval = sum([dtuple.val for dtuple in dplayer])
                derPP = CderPP(player=[mplayer, dplayer], mval=mval, dval=dval)  # derPP is single-layer, PP stays as is
                if (not fPd and mval > ave_mPP) or (fPd and dval > ave_dPP):
                    fin = 1  # PPs match, sum derPP in both PPP and _PPP, m fork:
                    sum_players(_PPP.players, PP.players)
                    sum_players(PPP.players, PP.players)  # same fin for both in comp_PP_
                else:
                    fin = 0
                _PPP.PP_ += [[PP, derPP, fin]]
                _PP.cPP_ += [[PP, derPP, 1]]  # rdn refs, initial fin=1, derPP is reversed
                PPP.PP_ += [[_PP, derPP, fin]]
                PP.cPP_ += [[_PP, derPP, 1]]  # bilateral assign to eval in centroid clustering, derPP is reversed
                '''
                if derPP.match params[-1]: form PPP
                elif derPP.match params[:-1]: splice PPs and their segs? 
                '''
        PPP_.append(_PPP)
    return PPP_


def comp_centroid(PPP_, fPd):  # comp PP to average PP in PPP, sum >ave PPs into new centroid, recursion while update>ave

    update_val = 0  # update val, terminate recursion if low

    for PPP in PPP_:
        PPP_val = 0  # new total, may delete PPP
        PPP_rdn = 0  # rdn of PPs to cPPs in other PPPs
        PPP_players = []
        for i, (PP, _, fin) in enumerate(PPP.PP_):  # comp PP to PPP centroid, derPP is replaced, use comp_plevels?

            mplayer, dplayer = comp_players(PPP.players, PP.players, PPP.fPds, PP.fPds)  # norm params in comp_ptuple
            mval = sum([mtuple.val for mtuple in mplayer])
            dval = sum([dtuple.val for dtuple in dplayer])
            derPP = CderPP(player=[mplayer, dplayer], mval=mval, dval=dval)  # derPP is recomputed at each call
            # compute rdn:
            cPP_ = PP.cPP_
            if fPd:
                cPP_ = sorted(cPP_, key=lambda cPP: cPP[1].dval, reverse=True)  # sort by derPP.val per recursion call
            else:
                cPP_ = sorted(cPP_, key=lambda cPP: cPP[1].mval, reverse=True)
            rdn = 1
            for (cPP, cderPP, cfin) in cPP_:
                if cderPP.mval > derPP.mval:  # cPP is instance of PP, eval derPP.mval only
                    if cfin: PPP_rdn += 1  # n of cPPs redundant to PP, if included and >val
                else:
                    break
            fneg = dval < ave_dPP * rdn if fPd else mval < ave_mPP * rdn  # rdn per PP
            if (fneg and fin) or (not fneg and not fin):  # re-clustering: exclude included or include excluded PP
                PPP.PP_[i][2] = not fin
                update_val += abs(mval)  # or sum abs mparams?
            if not fneg:  # include PP in PPP:
                PPP_val += mval
                PPP_rdn += rdn
                if PPP_players:
                    sum_players(PPP_players, PP.players, fneg)  # no fneg now?
                else:
                    PPP_players = PP.players  # initialization is simpler
                PPP.PP_[i][1] = derPP  # replace not sum

                for i, cPPt in enumerate(PP.cPP_):
                    cPPP = cPPt[0].root
                    for j, PPt in enumerate(cPPP.cPP_):  # get PPP and replace their derPP
                        if PPt[0] is PP:
                            cPPP.cPP_[j][1] = derPP
                    if cPPt[0] is PP:  # replace cPP's derPP
                        PPP.cPP_[i][1] = derPP
            PPP.mval = PPP_val  # m fork for now

        if PPP_players: PPP.players = PPP_players
        if PPP_val < PP_aves[fPd] * PPP_rdn:  # ave rdn-adjusted value per cost of PPP

            update_val += abs(PPP_val)  # or sum abs mparams?
            PPP_.remove(PPP)  # PPPs are hugely redundant, need to be pruned

            for (PP, derPP, fin) in PPP.PP_:  # remove refs to local copy of PP in other PPPs
                for (cPP, _, _) in PP.cPP_:
                    for i, (ccPP, _, _) in enumerate(cPP.cPP_):  # ref of ref
                        if ccPP is PP:
                            cPP.cPP_.pop(i)  # remove ccPP tuple
                            break

    if update_val > PP_aves[fPd]:
        comp_centroid(PPP_)  # recursion while min update value


def accum_PP(PP, inp, fd):  # comp_slice inp is seg, or segPP from agg+

    sum_players(PP.players, inp.players)  # not empty inp's players
    for i in 0,1: PP.valt[i] += inp.valt[i]
    PP.x0 = min(PP.x0, inp.x0)  # external params: 2nd player?
    PP.xn = max(PP.xn, inp.xn)
    PP.y0 = min(inp.y0, PP.y0)
    PP.yn = max(inp.yn, PP.yn)
    PP.Rdn += inp.rdn  # base_rdn + PP.Rdn / PP: recursion + forks + links: nderP / len(P__)?
    PP.nderP += len(inp.P__[-1].uplink_layers[-1][fd])  # redundant derivatives of the same P

    if PP.P__ and not isinstance(PP.P__[0], list):  # PP is seg if fseg in agg_recursion
        PP.uplink_layers[-1] += [inp.uplink_.copy()]  # += seg.link_s, they are all misses now
        PP.downlink_layers[-1] += [inp.downlink_.copy()]

        for P in inp.P__:  # add Ps in P__[y]:
            # assign
            P.root = object  # reset root, to be assigned next sub_recursion
            PP.P__.append(P)
    else:
        for P in inp.P__:  # add Ps in P__[y]:
            if not PP.P__:
                PP.P__.append([P])
            else:
                append_P(PP.P__, P)  # add P into nested list of P__
            # add terminated seg links for rng+:
            for derP in inp.P__[0].downlink_layers[-1][fd]:  # if downlink not in current PP's downlink and not part of the seg in current PP:
                if derP not in PP.downlink_layers[-2] and derP.P.roott[fd] not in PP.seg_levels[-1]:
                    PP.downlink_layers[-2] += [derP]
            for derP in inp.P__[-1].uplink_layers[-1][fd]:  # if downlink not in current PP's downlink and not part of the seg in current PP:
                if derP not in PP.downlink_layers[-2] and derP.P.roott[fd] not in PP.seg_levels[-1]:
                    PP.uplink_layers[-2] += [derP]

    for P_ in PP.P__[:-1]:  # add derP root, except for top row and bottom row derP
        for P in P_:
            for derP in P.uplink_layers[-1][fd]:
                derP.roott[fd] = PP

