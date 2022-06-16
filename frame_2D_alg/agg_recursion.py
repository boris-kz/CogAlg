import sys
import numpy as np
from itertools import zip_longest
from copy import deepcopy, copy
from class_cluster import ClusterStructure, NoneType, comp_param, Cder
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

    for i, PP_ in enumerate(PP_t):   # fiPd = fiPd % 2
        fiPd = i % 2
        if fiPd: ave_PP = ave_dPP
        else:    ave_PP = ave_mPP

        if fseg: M = ave- np.hypot(blob.params[0][5], blob.params[0][6])  # hypot(dy, dx)
        else: M = ave-abs(blob.G)
#        if M > ave_PP * blob.rdn and len(PP_)>1:  # >=2 comparands
        if len(PP_)>1:
            n_extended += 1 
            
            PPm_, PPd_ = [comp_PP_(PP_t)]  # compare all PPs to the average (centroid) of all other PPs
            # PP is generic for lower-level composition
            ''' to be updated:
            segm_ = form_segPPP_root(PPm_, root_rdn=2, fPd=0)  # forms segments: parameterized stacks of (P,derP)s
            segd_ = form_segPPP_root(PPd_, root_rdn=2, fPd=1)  # seg is a stack of (P,derP)s
            '''

            PPPm_, PPPd_ = form_PPP_([PPm_, PPd_], base_rdn=2)  # PPP is generic next-level composition
            
            splice_PPs(PPPm_, frng=1)
            splice_PPs(PPPd_, frng=0)
            
            PPP_t += [PPPm_, PPPd_]  # flat version

            if PPPm_: sub_recursion([], PPPm_, frng=1)  # rng+
            if PPPd_: sub_recursion([], PPPd_, frng=0)  # der+
        else:
            PPP_t += [[], []]  # replace with neg PPPs?

    if fseg:
        blob.seg_levels += [PPP_t]  # new level of segPs
    else:
        blob.levels.append(PPP_t)  # levels of dir_blob are Plevels

    if n_extended/len(PP_t) > 0.5:  # mean ratio of extended PPs
        agg_recursion(blob, fseg)

# draft,
'''
old comment:
1st compare each node to the average (centroid) of all surrounding nodes within a maximal cartesian distance. 
This determines if the node is part of proximity cluster, else it starts its own cluster. 
2nd, that proximity cluster is mapped out through all connected nodes, still via some form of connectivity clustering or flood-fill.

But then that proximity cluster forms its own centroid: mean constituent node params, and may be refined through centroid-based sub-clustering, 
a global-range version of current sub_recursion().
That's different from conventional centroid clustering in that initial cluster is defined through connectivity, 
there is no randomization of any sort.
'''
def comp_PP_(PP_):  # PP can also be PPP, etc.

    for PP in PP_:
        ave_params = []
        other_PPs = PP_.remove(PP)
        n = len(other_PPs)
        avePP = CPP
        for PP in other_PPs:
            sum_param = 0
            for param in PP.params:
                sum_param += param
        ave_params += [sum_param / n]
        derPP = CderPP
        # comp to centroid:
        for _param_layer, param_layer in zip(PP.params, ave_params):
            derPP.params += [comp_derP(_param_layer, param_layer)]

    PPm_ = [copy_P(PP, ftype=2) for PP in PP_]
    PPd_ = [copy_P(PP, ftype=2) for PP in PP_] 

    return PPm_, PPd_
    

def comp_PP(PP, avePP):  # draft
    '''
    probably not relevant:
    PP.uplink_layers[-1] += [derPP]  # each layer has Mlayer, Dlayer
    _PP.downlink_layers[-1] += [derPP]
    '''

# draft
def form_PPP_t(derPP_t, base_rdn):  # form PPs from match-connected segs
    for fPd, derPP_ in enumerate(derPP_t):
        # sort derPP_ by value param:
        for i, derPP in enumerate( sorted(derPP_, key=lambda derPP: derPP.params[fPd], reverse=True)):

            derPP.rdn += derPP.params[fPd] > derPP.params[fPd+1]  # ?
            ave = vaves[fPd] * derPP.rdn
            # the weaker links are redundant to the stronger, added to derP.P.link_layers[-1]) in prior loops:

            if derPP.params[fPd] > ave * len(derPP_[i:]):  # ave * cross-PPP rdn, derPPs are proto-PPPs
                PPP_ = derPP_[i:]  # PPP here is syntactically identical to derPP?
                break
    '''
    if derPP.match params[-1]: form PPP
    elif derPP.match params[:-1]: splice PPs and their segs?
    '''

# old:

def form_segPPP_root(PP_, root_rdn, fPd):  # not sure about form_seg_root
    
    for PP in PP_:
        link_eval(P.uplink_layers, fPd)
        link_eval(P.downlink_layers, fPd)
    
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
        tested_segs = []  # we need this because we may add new seg during splicing process, and those new seg need to check their link for splicing too
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

# to be updated
def merge_PP(_PP, PP, fPd):  # only for PP splicing

    for seg in PP.seg_levels[fPd][-1]:  # merge PP_segs into _PP:
        accum_CPP(_PP, seg, fPd)
        _PP.seg_levels[fPd][-1] += [seg]

    # merge uplinks and downlinks
    for uplink in PP.uplink_layers:
        if uplink not in _PP.uplink_layers:
            _PP.uplink_layers += [uplink]
    for downlink in PP.downlink_layers:
        if downlink not in _PP.downlink_layers:
            _PP.downlink_layers += [downlink]


def comp_dx(P):  # cross-comp of dx s in P.dert_

    Ddx = 0
    Mdx = 0
    dxdert_ = []
    _dx = P.dert_[0][2]  # first dx
    for dert in P.dert_[1:]:
        dx = dert[2]
        ddx = dx - _dx
        if dx > 0 == _dx > 0: mdx = min(dx, _dx)
        else: mdx = -min(abs(dx), abs(_dx))
        dxdert_.append((ddx, mdx))  # no dx: already in dert_
        Ddx += ddx  # P-wide cross-sign, P.L is too short to form sub_Ps
        Mdx += mdx
        _dx = dx
    P.dxdert_ = dxdert_
    P.Ddx = Ddx
    P.Mdx = Mdx

# june 22
def sub_recursion(root_layers, PP_, frng):  # compares param_layers of derPs in generic PP, form or accum top derivatives

    comb_layers = []
    for PP in PP_:  # PP is generic higher-composition pattern, P is generic lower-composition pattern
                    # both P and PP may be recursively formed higher-derivation derP and derPP, etc.
        if frng: PP_V = PP.params[-1][0] - ave_mPP * PP.rdn; rng = PP.rng+1; min_L = rng * 2  # V: value of sub_recursion per PP
        else:    PP_V = PP.params[-1][1] - ave_dPP * PP.rdn; rng = PP.rng; min_L = 3  # need 3 Ps to compute layer2, etc.
        if PP_V > 0 and PP.nderP > min_L:

            PP.rdn += 1  # rdn to prior derivation layers
            PP.rng = rng
            Pm__ = comp_P_rng(PP.P__, rng)
            Pd__ = comp_P_der(PP.P__)

            sub_segm_ = form_seg_root([Pm_ for Pm_ in reversed(Pm__)], root_rdn=PP.rdn, fPd=0)
            sub_segd_ = form_seg_root([Pd_ for Pd_ in reversed(Pd__)], root_rdn=PP.rdn, fPd=1)
            sub_PPm_, sub_PPd_ = form_PP_root(( sub_segm_, sub_segd_), base_rdn=PP.rdn)  # forms PPs: parameterized graphs of linked segs

            PP.layers = [(sub_PPm_, sub_PPd_)]
            if sub_PPm_:
                # rng+=1, |+=n to reduce clustering costs?
                sub_recursion(PP.layers, sub_PPm_, frng=1)  # rng+ comp_P in PPms, form param_layer, sub_PPs
            if sub_PPd_:
                sub_recursion(PP.layers, sub_PPd_, frng=0)  # der+ comp_P in PPds, form param_layer, sub_PPs

            if PP.layers:  # pack added sublayers:
                new_comb_layers = []
                for (comb_sub_PPm_, comb_sub_PPd_), (sub_PPm_, sub_PPd_) in zip_longest(comb_layers, PP.layers, fillvalue=([], [])):
                    comb_sub_PPm_ += sub_PPm_
                    comb_sub_PPd_ += sub_PPd_
                    new_comb_layers.append((comb_sub_PPm_, comb_sub_PPd_))  # add sublayer
                comb_layers = new_comb_layers

    if comb_layers: root_layers += comb_layers


def comp_P_rng(iP__, rng):  # rng+ sub_recursion in PP.P__, adding two link_layers per P

    P__ = [P_ for P_ in reversed(iP__)]  # revert to top-down
    uplinks__ = [[ [] for P in P_] for P_ in P__[rng:]]  # rng derP_s per P, exclude 1st rng rows
    downlinks__ = [[ [] for P in P_] for P_ in P__[:-rng]]  # exclude last rng rows

    for y, _P_ in enumerate(P__[:-rng]):  # higher compared row, skip last rng: no lower comparand rows
        for x, _P in enumerate(_P_):

            for pri_rng_derP in _P.downlink_layers[-1]:  # get linked Ps at dy = rng-1
                pri_P = pri_rng_derP.P
                for ini_derP in pri_P.downlink_layers[0]:  # lower comparands are linked Ps at dy = rng
                    P = ini_derP.P
                    if isinstance(P, CPP) or isinstance(P, CderP):  # rng+ fork for derPs, very unlikely
                        derP = comp_derP(_P, P)  # form higher vertical derivatives of derP or PP params
                    else:
                        derP = comp_P(_P, P)  # form vertical derivatives of horizontal P params
                    # += links:
                    downlinks__[y][x] += [derP]
                    up_x = P__[y+rng].index(P)  # index of P in P_ at y+rng
                    uplinks__[y][up_x] += [derP]  # uplinks__[y] = P__[y+rng]: uplinks__= P__[rng:]

    for P_, uplinks_ in zip( P__[rng:], uplinks__):  # skip 1st rmg rows, no uplinks
        for P, uplinks in zip(P_, uplinks_):
            P.uplink_layers += [uplinks, []]  # add rng_derP_ to P.link_layers

    for P_, downlinks_ in zip(P__[:-rng], downlinks__):  # skip last rng rows, no downlinks
        for P, downlinks in zip(P_, downlinks_):
            P.downlink_layers += [downlinks, []]

    return iP__  # return bottom-up P__