import sys
import numpy as np
from itertools import zip_longest
from copy import deepcopy, copy
from class_cluster import ClusterStructure, NoneType, comp_param, Cdert
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

    for i, PP_ in enumerate(PP_t):   # fiPd = fiPd % 2
        fiPd = i % 2
        if fiPd: ave_PP = ave_dPP
        else:    ave_PP = ave_mPP

        if fseg: M = ave- np.hypot(blob.params[0][5], blob.params[0][6])  # hypot(dy, dx)
        else: M = ave-abs(blob.G)  # if M > ave_PP * blob.rdn and len(PP_)>1:  # >=2 comparands

        if len(PP_)>1:
            n_extended += 1

            derPP_t = comp_PP_(PP_)  # compare all PPs to the average (centroid) of all other PPs, is generic for lower level
            PPP_t = form_PPP_t(derPP_t)  # call to individual comp_PP if mPPP > ave_mPPP, converting derPP to CPPP

            splice_PPs(PPP_t)  # for initial PPs only: if PP is CPP?
            sub_recursion_eval(PPP_t)  # rng+ or der+, if PP is CPPP?
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
def comp_PP_(PP_):  # PP can also be PPP, etc.

    derPPm_, derPPd_ = [],[]

    for PP in PP_:
        compared_PP_ = copy(PP_)  # shallow copy
        compared_PP_.remove(PP)
        n = len(compared_PP_)
        # sum same-type params across other_PP_:
        sum_params_layers = [[0 for param in params_layer] for params_layer in PP.params]
        for compared_PP in compared_PP_:
            for i, params_layer in enumerate(compared_PP.params):
                for j, param in enumerate(params_layer):
                    sum_params_layers[i][j] += param
        # ave params of other_PP_:
        ave_params = [[param/n for param in sum_params] for sum_params in sum_params_layers]

        derPP = CPP(params=deepcopy(PP.params), layers=[PP_])  # derPP inherits PP.params
        # comp to the average (centroid) of other PPs in PP_:
        for _param_layer, param_layer in zip(PP.params, ave_params):
            derPP.params[-1] += [comp_params(_param_layer, param_layer)]  # last layer: derivatives of all lower layers

        derPPm_.append(copy_P(derPP, Ptype=2))
        derPPd_.append(copy_P(derPP, Ptype=2))

    return derPPm_, derPPd_


def form_PPP_t(derPP_t):  # form PPs from match-connected segs
    PPP_t = []

    for fPd, derPP_ in enumerate(derPP_t):
        # sort by value of last layer: derivatives of all lower layers:
        derPP_ = sorted(derPP_, key=lambda derPP: derPP.params[-1][fPd], reverse=True)  # descending order
        PPP_ = []
        for i, derPP in enumerate(derPP_):
            param_value = 0
            for param_layer in derPP.params:
                derPP.rdn += param_layer[fPd] > param_layer[1-fPd]
                param_value += param_layer[fPd]

            ave = vaves[fPd] * derPP.rdn * len(derPP_[i+1])  # derPP is redundant to higher-value previous derPPs in derPP_
            if param_value > ave:
                PPP_ += [derPP]  # base derPP and PPP is CPP
                if param_value > ave*10:
                    ind_comp_PP_(derPP, fPd)  # derPP is converted from CPP to CPPP
            else:
                break
        PPP_t.append(PPP_)
    return PPP_t

# draft
def ind_comp_PP_(_PP, fPd):  # 1-to-1 comp, _PP is converted from CPP to higher-composition CPPP

    derPP_ = []
    for PP in _PP.layers[0]:  # 1-to-1 comparison between _PP and all other PPs
        derPP=CderPP  # add some _PP variables?

        for _param_layer, param_layer in zip(_PP.params, PP.params):  # or top-down, continue if match?
            derPP.params += [comp_params(_param_layer, param_layer)]
        derPP_ += [derPP]

    # cluster derPPs into PPPs by connectivity:
    for i, _derPP in enumerate(derPP_):

        if _derPP.params[-1][fPd]:
            PPP = CPPP(params=deepcopy(_derPP.params), layers=[_derPP.PP])
            PPP.accum_from(_derPP)  # initialization
            _derPP.root = PPP
            for derPP in derPP_[i+1:]:
                if not derPP.PP.root:
                    if derPP.params[-1][fPd]:  # positive and not in PPP yet
                        PPP.layers.append(derPP)  # multiple composition orders
                        PPP.accum_from(_derPP)
                        derPP.root = PPP

                    elif derPP.params[:-1][fPd]  # pseudo
                        # splice PP and their segs
                        pass
        '''
    if derPP.match params[-1]: form PPP
    elif derPP.match params[:-1]: splice PPs and their segs? why params[:-1]?
    '''

def comp_params(_params, params):

    nparams = len(_params)
    derivatives = []
    hyps = []

    for i, (_param, param) in enumerate(zip(params, params)):
        # get param type:
        param_type = int(i/ (2 ** (nparams-1)))  # for 9 compared params, but there are more in higher layers?

        if param_type == 0:  # x
            _x = param; x = param
            dx = _x - x; mx = ave_dx - abs(dx)
            derivatives.append(dx); derivatives.append(mx)
            hyps.append(np.hypot(dx, 1))

        elif param_type == 1:  # I
            _I = _param; I = param
            dI = _I - I; mI = ave_I - abs(dI)
            derivatives.append(dI); derivatives.append(mI)

        elif param_type == 2:  # G
            hyp = hyps[i%param_type]
            _G = _param; G = param
            dG = _G - G/hyp;  mG = min(_G, G)  # if comp_norm: reduce by hypot
            derivatives.append(dG); derivatives.append(mG)

        elif param_type == 3:  # Ga
            _Ga = _param; Ga = param
            dGa = _Ga - Ga;  mGa = min(_Ga, Ga)
            derivatives.append(dGa); derivatives.append(mGa)

        elif param_type == 4:  # M
            hyp = hyps[i%param_type]
            _M = _param; M = param
            dM = _M - M/hyp;  mM = min(_M, M)
            derivatives.append(dM); derivatives.append(mM)

        elif param_type == 5:  # Ma
            _Ma = _param; Ma = param
            dMa = _Ma - Ma;  mMa = min(_Ma, Ma)
            derivatives.append(dMa); derivatives.append(mMa)

        elif param_type == 6:  # L
            hyp = hyps[i%param_type]
            _L = _param; L = param
            dL = _L - L/hyp;  mL = min(_L, L)
            derivatives.append(dL); derivatives.append(mL)

        elif param_type == 7:  # angle, (sin_da, cos_da)
            if isinstance(_param, tuple):  # (sin_da, cos_da)
                 _sin_da, _cos_da = _param; sin_da, cos_da = param
                 sin_dda = (cos_da * _sin_da) - (sin_da * _cos_da)  # sin(α - β) = sin α cos β - cos α sin β
                 cos_dda = (cos_da * _cos_da) + (sin_da * _sin_da)  # cos(α - β) = cos α cos β + sin α sin β
                 dangle = (sin_dda, cos_dda)  # da
                 mangle = ave_dangle - abs(np.arctan2(sin_dda, cos_dda))  # ma is indirect match
                 derivatives.append(dangle); derivatives.append(mangle)
            else: # m or scalar
                _mangle = _param; mangle = param
                dmangle = _mangle - mangle;  mmangle = min(_mangle, mangle)
                derivatives.append(dmangle); derivatives.append(mmangle)

        elif param_type == 8:  # dangle   (sin_da0, cos_da0, sin_da1, cos_da1)
            if isinstance(_param, tuple):  # (sin_da, cos_da)
                _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _param
                sin_da0, cos_da0, sin_da1, cos_da1 = param

                sin_dda0 = (cos_da0 * _sin_da0) - (sin_da0 * _cos_da0)
                cos_dda0 = (cos_da0 * _cos_da0) + (sin_da0 * _sin_da0)
                sin_dda1 = (cos_da1 * _sin_da1) - (sin_da1 * _cos_da1)
                cos_dda1 = (cos_da1 * _cos_da1) + (sin_da1 * _sin_da1)
                daangle = (sin_dda0, cos_dda0, sin_dda1, cos_dda1)
                # day = [-sin_dda0 - sin_dda1, cos_dda0 + cos_dda1]
                # dax = [-sin_dda0 + sin_dda1, cos_dda0 + cos_dda1]
                gay = np.arctan2( (-sin_dda0 - sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in y?
                gax = np.arctan2( (-sin_dda0 + sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in x?
                maangle = ave_dangle - abs(np.arctan2(gay, gax))  # match between aangles, probably wrong
                derivatives.append(daangle); derivatives.append(maangle)

            else:  # m or scalar
                _maangle = _param; maangle = param
                dmaangle = _maangle - maangle;  mmaangle = min(_maangle, maangle)
                derivatives.append(dmaangle); derivatives.append(mmaangle)

    return derivatives

# old:

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