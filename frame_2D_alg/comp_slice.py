'''
Comp_slice is a terminal fork of intra_blob.
-
It traces blob axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)
-
Vectorization is clustering of parameterized Ps + their derivatives (derPs) into PPs: patterns of Ps that describe edge blob.
This process is a reduced-dimensionality (2D->1D) version of cross-comp and clustering cycle, common across this project.
As we add higher dimensions (2D alg, 3D alg), this dimensionality reduction is done in salient high-aspect blobs
(likely edges / contours in 2D or surfaces in 3D) to form more compressed "skeletal" representations of full-D patterns.
'''

from collections import deque
import sys
import numpy as np
from itertools import zip_longest
from copy import deepcopy, copy
from class_cluster import ClusterStructure, NoneType, comp_param, Cdert
from segment_by_direction import segment_by_direction

# import warnings  # to detect overflow issue, in case of infinity loop
# warnings.filterwarnings('error')

ave_inv = 20  # ave inverse m, change to Ave from the root intra_blob?
ave = 5  # ave direct m, change to Ave_min from the root intra_blob?
ave_g = 30  # change to Ave from the root intra_blob?
ave_ga = 0.78  # ga at 22.5 degree
flip_ave = .1
flip_ave_FPP = 0  # flip large FPPs only (change to 0 for debug purpose)
div_ave = 200
ave_rmP = .7  # the rate of mP decay per relative dX (x shift) = 1: initial form of distance
ave_ortho = 20
aveB = 50
# comp_param coefs:
ave_I = ave_inv
ave_M = ave  # replace the rest with coefs:
ave_Ma = 10
ave_G = 10
ave_Ga = 2  # related to dx?
ave_L = 10
ave_dx = 5  # difference between median x coords of consecutive Ps
ave_dangle = 2  # vertical difference between angles
ave_daangle = 2
ave_mP = 10
ave_dP = 10
ave_mPP = 10
ave_dPP = 10
ave_splice = 10

param_names = ["x", "I", "M", "Ma", "L", "angle", "aangle"]  # angle = Dy, Dx; aangle = sin_da0, cos_da0, sin_da1, cos_da1; recompute Gs for comparison?
aves = [ave_dx, ave_I, ave_M, ave_Ma, ave_L, ave_G, ave_Ga, ave_mP, ave_dP]
vaves = [ave_mP, ave_dP]

class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP

    params = list  # 9 compared horizontal params: x, L, I, M, Ma, G, Ga, Ds( Dy, Dx, Sin_da0), Das( Cos_da0, Sin_da1, Cos_da1)
    # I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1 are summed from dert[3:], M, Ma from ave- g, ga
    # G, Ga are recomputed from Ds, Das; M, Ma are not restorable from G, Ga
    x0 = int
    x = float  # median x
    y = int  # for vertical gap in PP.P__
    L = int
    sign = NoneType  # g-ave + ave-ga sign
    # all the above are redundant to params
    rdn = int  # blob-level redundancy, ignore for now
    # composite params:
    dert_ = list  # array of pixel-level derts, redundant to uplink_, only per blob?
    uplink_layers = lambda: [[],[]]  # init a layer of derPs and a layer of match_derPs
    downlink_layers = lambda: [[],[]]
    root = lambda:None  # segment that contains this P, PP is root.root
    # only in Pd:
    Pm = object  # reference to root P
    dxdert_ = list
    # only in Pm:
    Pd_ = list
    # if comp_dx:
    Mdx = int
    Ddx = int

class CderP(ClusterStructure):  # tuple of derivatives in P uplink_ or downlink_

    # dP, mP are packed in params[0,1]
    params = list  # P derivation layer, n_params = 9 * 2**der_cnt, flat, decoded by mapping each m,d to lower-layer param
    x0 = int  # redundant to params:
    x = float  # median x
    L = int  # pack in params?
    sign = NoneType  # g-ave + ave-ga sign
    y = int  # for vertical gaps in PP.P__, replace with derP.P.y?
    P = object  # lower comparand
    _P = object  # higher comparand
    root = lambda:None  # segment in sub_recursion
    # higher derivatives
    rdn = int  # mrdn, + uprdn if branch overlap?
    uplink_layers = lambda: [[],[]]  # init a layer of dderPs and a layer of match_dderPs
    downlink_layers = lambda: [[],[]]
   # from comp_dx
    fdx = NoneType

class CPP(CP, CderP):  # P and derP params are combined into param_layers?

    params = list  # derivation layers += derP params per der+, param L is actually Area
    sign = bool
    xn = int
    yn = int
    rng = lambda: 1  # rng starts with 1
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs
    Rdn = int  # for accumulation only
    nP = int  # len 2D derP__ in levels[0][fPd]?  ly = len(derP__), also x, y?
    nderP = int
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
    root = lambda:None  # higher-order PP, segP, or PPP

# Functions:

def comp_slice_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    from agg_recursion import agg_recursion

    segment_by_direction(blob, verbose=False)  # forms blob.dir_blobs
    for dir_blob in blob.dir_blobs:  # dir_blob should be CBlob

        P__ = slice_blob(dir_blob, verbose=False)  # cluster dir_blob.dert__ into 2D array of blob slices
        # comp_dx_blob(P__), comp_dx?
        Pm__ = comp_P_root(deepcopy(P__))  # scan_P_, comp_P | link_layer, adds mixed uplink_, downlink_ per P,
        Pd__ = comp_P_root(deepcopy(P__))  # deepcopy before assigning link derPs to Ps

        segm_ = form_seg_root(Pm__, root_rdn=2, fPd=0)  # forms segments: parameterized stacks of (P,derP)s
        segd_ = form_seg_root(Pd__, root_rdn=2, fPd=1)  # seg is a stack of (P,derP)s

        PPm_, PPd_ = form_PP_root((segm_, segd_), base_rdn=2)  # forms PPs: parameterized graphs of linked segs
        # rng+, der+ fork eval per PP, forms param_layer and sub_PPs:
        sub_recursion_eval(PPm_)
        sub_recursion_eval(PPd_)

        for PP_ in (PPm_, PPd_):  # 1st agglomerative recursion is per PP, appending PP.seg_levels, not blob.levels:
            for PP in PP_:
                agg_recursion(PP, fseg=1)  # higher-composition comp_seg -> segPs.. per seg__[n], in PP.seg_levels
        dir_blob.levels = [[PPm_], [PPd_]]
        agg_recursion(dir_blob, fseg=0)  # 2nd call per dir_blob.PP_s formed in 1st call, forms PPP..s and dir_blob.levels

    splice_dir_blob_(blob.dir_blobs)


def slice_blob(blob, verbose=False):  # forms horizontal blob slices: Ps, ~1D Ps, in select smooth edge (high G, low Ga) blobs

    mask__ = blob.mask__  # same as positive sign here
    dert__ = zip(*blob.dert__)  # convert 10-tuple of 2D arrays into 1D array of 10-tuple blob rows
    dert__ = [zip(*dert_) for dert_ in dert__]  # convert 1D array of 10-tuple rows into 2D array of 10-tuples per blob

    height, width = mask__.shape
    if verbose: print("Converting to image...")
    P__ = []  # blob of Ps

    for y, (dert_, mask_) in enumerate(zip(dert__, mask__)):  # unpack lines
        P_ = []  # line of Ps
        _mask = True
        for x, (dert, mask) in enumerate(zip(dert_, mask_)):  # unpack derts: tuples of 10 params
            if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()

            if not mask:  # masks: if 0,_1: P initialization, if 0,_0: P accumulation, if 1,_0: P termination
                if _mask:  # initialize P params with first unmasked dert:
                    Pdert_ = [dert]
                    params = [ave_g-dert[1], ave_ga-dert[2], *dert[3:]]  # m, ma, dert[3:]: i, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1
                else:
                    # dert and _dert are not masked, accumulate P params from dert params (comp G vs. Dx, Dy->der+?):
                    params[1] += ave_g-dert[1]; params[2] += ave_ga-dert[2]  # M, Ma
                    for i, (Param, param) in enumerate(zip(params[2:], dert[3:]), start=2):  # I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1
                        params[i] = Param + param
                    Pdert_.append(dert)
            elif not _mask:
                # _dert is not masked, dert is masked, terminate P:
                L = len(Pdert_)
                P_.append( CP(params= [x-(L-1)/2, L] + list(params), x0=x-(L-1), L=L, y=y, dert_=Pdert_))

            _mask = mask
        if not _mask:  # pack last P:
            L = len(Pdert_)
            P_.append( CP(params = [x-(L-1)/2, L] + list(params), x0=x-(L-1), L=L, y=y, dert_=Pdert_))
        P__ += [P_]

    blob.P__ = P__
    return P__


def comp_P_root(P__):  # vertically compares y-adjacent and x-overlapping Ps: blob slices, forming 2D derP__

    _P_ = P__[0]  # upper row, top-down
    for P_ in P__[1:]:  # lower row
        for P in P_:
            for _P in _P_:  # test for x overlap(_P,P) in 8 directions, derts are positive in all Ps:
                if (P.x0 - 1 < _P.x0 + _P.L) and (P.x0 + P.L + 1 > _P.x0):
                    derP = comp_P(_P, P)
                    P.uplink_layers[-2] += [derP]  # append derPs, uplink_layers[-1] is match_derPs
                    _P.downlink_layers[-2] += [derP]
                elif (P.x0 + P.L) < _P.x0:
                    break  # no P xn overlap, stop scanning lower P_
        _P_ = P_

    return P__

def comp_P_rng(P__, rng):  # rng+ sub_recursion in PP.P__, switch to rng+n to skip clustering?

    for P_ in P__:  # add 2 link layers: rng_derP_ and match_rng_derP_
        for P in P_:
            P.uplink_layers += [[],[]]; P.downlink_layers += [[],[]]

    for y, _P_ in enumerate(P__[:-rng]):  # higher compared row, skip last rng: no lower comparand rows
        for x, _P in enumerate(_P_):
            # get linked Ps at dy = rng-1:
            for pri_derP in _P.downlink_layers[-3]:
                pri_P = pri_derP.P
                # compare linked Ps at dy = rng:
                for ini_derP in pri_P.downlink_layers[0]:
                    P = ini_derP.P
                    # add new Ps, their link layers and reset their roots:
                    if P not in [P for P_ in P__ for P in P_]:
                        append_P(P__, P)
                        P.uplink_layers += [[],[]]; P.downlink_layers += [[],[]]; P.root = object

                    if isinstance(P, CPP) or isinstance(P, CderP):  # rng+ fork for derPs, very unlikely
                        derP = comp_derP(_P, P)  # form higher vertical derivatives of derP or PP params
                    else:
                        derP = comp_P(_P, P)  # form vertical derivatives of horizontal P params
                    P.uplink_layers[-2] += [derP]
                    _P.downlink_layers[-2] += [derP]

    Pm__= [[copy_P(P, Ptype=0) for P in P_] for P_ in P__ ]
    Pd__= [[copy_P(P, Ptype=0) for P in P_] for P_ in P__ ]

    return Pm__, Pd__  # new_mP__, new_dP__


def comp_P_der(P__):  # der+ sub_recursion in PP.P__, compare P.uplinks to P.downlinks

    dderPs__ = []  # derP__ = [[] for P_ in P__[:-1]]  # init derP rows, exclude bottom P row

    for P_ in P__[1:-1]:  # higher compared row, exclude 1st: no +ve uplinks, and last: no +ve downlinks
        dderPs_ = []  # row of dderPs
        for P in P_:
            dderPs = []  # dderP for each _derP, derP pair in P links
            for _derP in P.uplink_layers[-1]:
                for derP in P.downlink_layers[-1]:
                    # there maybe no x overlap between recomputed Ls of _derP and derP, compare anyway,
                    # mderP * (ave_olp_L / olp_L)? or olp(_derP._P.L, derP.P.L)?
                    # gap: neg_olp, ave = olp-neg_olp?
                    dderP = comp_derP(_derP, derP)  # form higher vertical derivatives of derP or PP params
                    derP.uplink_layers[0] += [dderP]  # pre-init layer per derP
                    _derP.downlink_layers[0] += [dderP]
                    dderPs += [dderP]
                # compute x overlap between dderP'__P and P, in form_seg_ or comp_layer?
            dderPs_ += dderPs  # row of dderPs
        dderPs__ += [dderPs_]

    dderPm__ = [[copy_P(dderP, Ptype=1) for dderP in dderP_] for dderP_ in dderPs__ ]
    dderPd__ = [[copy_P(dderP, Ptype=1) for dderP in dderP_] for dderP_ in dderPs__ ]

    return dderPm__, dderPd__


def form_seg_root(P__, root_rdn, fPd):  # form segs from Ps

    for P_ in P__[1:]:  # scan bottom-up, append link_layers[-1] with branch-rdn adjusted matches in link_layers[-2]:
        for P in P_: link_eval(P.uplink_layers, fPd)  # uplinks_layers[-2] matches -> uplinks_layers[-1]

    for P_ in P__[:-1]:  # form downlink_layers[-1], different branch rdn, for termination eval in form_seg_?
        for P in P_: link_eval(P.downlink_layers, fPd)  # downinks_layers[-2] matches -> downlinks_layers[-1]

    seg_ = []
    for P_ in reversed(P__):  # get a row of Ps bottom-up, different copies per fPd
        while P_:
            P = P_.pop(0)
            if P.uplink_layers[-1]:  # last matching derPs layer is not empty
                form_seg_(seg_, P__, [P], fPd)  # test P.matching_uplink_, not known in form_seg_root
            else:
                seg_.append( sum2seg([P], fPd))  # no link_s, terminate seg_Ps = [P]

    return seg_

def form_seg_(seg_, P__, seg_Ps, fPd):  # form contiguous segments of vertically matching Ps

    if len(seg_Ps[-1].uplink_layers[-1]) > 1:  # terminate seg
        seg_.append( sum2seg( seg_Ps, fPd))  # convert seg_Ps to CPP seg
    else:
        uplink_ = seg_Ps[-1].uplink_layers[-1]
        if uplink_ and len(uplink_[0]._P.downlink_layers[-1])==1:
            # one P.uplink AND one _P.downlink: add _P to seg, uplink_[0] is sole upderP:
            P = uplink_[0]._P
            [P_.remove(P) for P_ in P__ if P in P_]  # remove P from P__ so it's not inputted in form_seg_root
            seg_Ps += [P]  # if P.downlinks in seg_down_misses += [P]

            if seg_Ps[-1].uplink_layers[-1]:
                form_seg_(seg_, P__, seg_Ps, fPd)  # recursive compare sign of next-layer uplinks
            else:
                seg_.append( sum2seg(seg_Ps, fPd))
        else:
            seg_.append( sum2seg(seg_Ps, fPd))  # terminate seg at 0 matching uplink


def link_eval(link_layers, fPd):

    # sort derPs in link_layers[-2] by their value param:
    for i, derP in enumerate( sorted( link_layers[-2], key=lambda derP: derP.params[fPd], reverse=True)):

        if fPd: derP.rdn += derP.params[0] > derP.params[1]  # mP > dP
        else: rng_eval(derP, fPd)  # reset derP.val, derP.rdn

        if derP.params[fPd][0] > vaves[fPd] * derP.rdn * (i+1):  # ave * rdn to stronger derPs in link_layers[-2]
            link_layers[-1].append(derP)  # misses = link_layers[-2] not in link_layers[-1]


def rng_eval(derP, fPd):  # compute value of combined mutual derPs: overlap between P uplinks and _P downlinks

    _P, P = derP._P, derP.P
    common_derP_ = []

    for _downlink_layer, uplink_layer in zip(_P.downlink_layers, P.uplink_layers):  # overlap in P uplinks and _P downlinks
        common_derP_ += list( set(_downlink_layer).intersection(uplink_layer))  # get common derP in mixed uplinks
    rdn = 1
    olp_val = 0
    for derP in common_derP_:
        rdn += derP.params[fPd] > derP.params[1-fPd]  # dP > mP if fPd, else mP > dP
        olp_val += derP.params[fPd][0]

    nolp = len(common_derP_)
    derP.params[fPd][0] = olp_val / nolp
    derP.rdn += (rdn / nolp) > .5  # no fractional rdn?


def form_PP_root(seg_t, base_rdn):  # form PPs from match-connected segs

    PP_t = []
    for fPd in 0, 1:
        PP_ = []
        seg_ = seg_t[fPd]
        for seg in seg_:  # bottom-up
            if not isinstance(seg.root, CPP):  # seg is not already in PP initiated by some prior seg
                PP_segs = [seg]
                # add links in PP_segs:
                if seg.P__[-1].uplink_layers[-1]:
                    form_PP_(PP_segs, seg.P__[-1].uplink_layers[-1].copy(), fPd, fup=1)
                if seg.P__[0].downlink_layers[-1]:
                    form_PP_(PP_segs, seg.P__[0].downlink_layers[-1].copy(), fPd, fup=0)
                # convert PP_segs to PP:
                PP_ += [sum2PP(PP_segs, base_rdn, fPd)]

        PP_t.append(PP_)  # PPm_, PPd_
    return PP_t

def form_PP_(PP_segs, link_, fPd, fup): # flood-fill PP_segs with vertically linked segments:
    '''
    PP is a graph with segs as 1D "vertices", each has two sets of edges / branching points: seg.uplink_ and seg.downlink_.
    '''
    for derP in link_:  # uplink_ or downlink_
        if fup: seg = derP._P.root
        else:   seg = derP.P.root

        if seg and seg not in PP_segs:  # top and bottom row Ps are not in segs
            PP_segs += [seg]
            uplink_ = seg.P__[-1].uplink_layers[-1]  # top-P uplink_
            if uplink_:
                form_PP_(PP_segs, uplink_, fPd, fup=1)
            downlink_ = seg.P__[0].downlink_layers[-1]  # bottom-P downlink_
            if downlink_:
                form_PP_(PP_segs, downlink_, fPd, fup=0)


def sum2seg(seg_Ps, fPd):  # sum params of vertically connected Ps into segment

    uplinks, uuplinks  = seg_Ps[-1].uplink_layers[-2:]  # uplinks of top P
    miss_uplink_ = [uuplink for uuplink in uuplinks if uuplink not in uplinks]  # in layer-1 but not in layer-2

    downlinks, ddownlinks = seg_Ps[0].downlink_layers[-2:]  # downlinks of bottom P, downlink.P.seg.uplinks= lower seg.uplinks
    miss_downlink_ = [ddownlink for ddownlink in ddownlinks if ddownlink not in downlinks]
    # seg rdn: up cost to init, up+down cost for comp_seg eval, in 1st agg_recursion?
    # P rdn is up+down M/n, but P is already formed and compared?

    seg = CPP(x0=seg_Ps[0].x0, P__=seg_Ps, uplink_layers=[miss_uplink_], downlink_layers = [miss_downlink_],
              L = len(seg_Ps), y0 = seg_Ps[0].y, params=[[],[]])  # seg.L is Ly
    iP = seg_Ps[0]
    if isinstance(iP, CPP): accum_P = accum_CPP
    elif isinstance(iP, CderP): accum_P = accum_CderP
    else: accum_P = accum_CP

    for P in seg_Ps[:-1]:
        accum_P(seg, P, fPd)
        accum_CderP(seg, P.uplink_layers[-1][0], fPd)
    accum_P(seg, seg_Ps[-1], fPd)  # accumulate last P

    return seg


def sum2PP(PP_segs, base_rdn, fPd):  # sum params: derPs into segment or segs into PP

    PP = CPP(x0=PP_segs[0].x0, rdn=base_rdn, sign=PP_segs[0].sign, L= len(PP_segs), params=[[],[]])
    PP.seg_levels[fPd][0] = PP_segs  # PP_segs is seg_levels[0]

    for seg in PP_segs:
        accum_CPP(PP, seg, fPd)

    return PP


def accum_CP(seg, P, fPd):

    accum_nested(seg.params[0], P.params)
    P.root = seg
    seg.x0 = min(seg.x0, P.x0)


# i think we need accum_CderP too, some params in PP doesn't exist in der
def accum_CderP(PP, inp, fPd):  # inp is seg or PP in recursion

    accum_nested(PP.params[1], inp.params)
    inp.root = PP
    # may add more assignments here

def accum_CPP(PP, inp, fPd):  # inp is seg or PP in recursion

    accum_nested(PP.params[0], inp.params[0])
    accum_nested(PP.params[1], inp.params[1])

    inp.root = PP
    PP.x += inp.x*inp.L  # or in inp.params?
    PP.y += inp.y*inp.L
    PP.xn = max(PP.x0, inp.x0)
    PP.yn = max(inp.y, PP.y)  # or arg y instead of derP.y?
    PP.Rdn += inp.rdn  # base_rdn + PP.Rdn / PP: recursion + forks + links: nderP / len(P__)?
    PP.nderP += len(inp.P__[-1].uplink_layers[-1])  # redundant derivatives of the same P

    if PP.P__ and not isinstance(PP.P__[0], list):  # PP is seg if fseg in agg_recursion
        PP.uplink_layers[-1] += [inp.uplink_.copy()]  # += seg.link_s, they are all misses now
        PP.downlink_layers[-1] += [inp.downlink_.copy()]

        for P in inp.P__:  # add Ps in P__[y]:
            P.root = object  # reset root, to be assigned next sub_recursion
            PP.P__.append(P)
    else:
        for P in inp.P__:  # add Ps in P__[y]:
            if not PP.P__:
                PP.P__.append([P])
            else:  # not reviewed
                append_P(PP.P__, P)  # add P into nested list of P__

            # add seg links: we may need links of all terminated segs, for rng+
            for derP in inp.P__[0].downlink_layers[-1]:  # if downlink not in current PP's downlink and not part of the seg in current PP:
                if derP not in PP.downlink_layers[-1] and derP.P.root not in PP.seg_levels[fPd][-1]:
                    PP.downlink_layers[-1] += [derP]
            for derP in inp.P__[-1].uplink_layers[-1]:  # if downlink not in current PP's downlink and not part of the seg in current PP:
                if derP not in PP.downlink_layers[-1] and derP.P.root not in PP.seg_levels[fPd][-1]:
                    PP.uplink_layers[-1] += [derP]

# change to ops per param, as in comp_ptuple
# tuple of 2 params:  [mx, mL, mM, mMa, mI, mG, mGa, , mangle, mP, maangle], [dx, dL, dM, dMa, dI, dG, dGa, , dangle, dP, daangle]
def accum_ptuple(Ptuple, ptuple):

    for i, (_param, param) in enumerate(zip(Ptuple, ptuple)):  # include all summable derP variables into params?
        if isinstance(_param, tuple):
            if len(_param) == 2:  # (sin_da, cos_da)
                _sin_da, _cos_da = _param
                sin_da, cos_da = param
                sum_sin_da = (cos_da * _sin_da) + (sin_da * _cos_da)  # sin(α + β) = sin α cos β + cos α sin β
                sum_cos_da = (cos_da * _cos_da) - (sin_da * _sin_da)  # cos(α + β) = cos α cos β - sin α sin β
                Ptuple[i] = (sum_sin_da, sum_cos_da)
            else:  # (sin_da0, cos_da0, sin_da1, cos_da1)
                _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _param
                sin_da0, cos_da0, sin_da1, cos_da1 = param
                sum_sin_da0 = (cos_da0 * _sin_da0) + (sin_da0 * _cos_da0)  # sin(α + β) = sin α cos β + cos α sin β
                sum_cos_da0 = (cos_da0 * _cos_da0) - (sin_da0 * _sin_da0)  # cos(α + β) = cos α cos β - sin α sin β
                sum_sin_da1 = (cos_da1 * _sin_da1) + (sin_da1 * _cos_da1)
                sum_cos_da1 = (cos_da1 * _cos_da1) - (sin_da1 * _sin_da1)
                Ptuple[i] = (sum_sin_da0, sum_cos_da0, sum_sin_da1, sum_cos_da1)
        else:  # scalar
            Ptuple[i] += param


# P params : x, L, m, ma, I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1
def accum_p(P_params, p_params):  # can be accumulated directly
    for i, p_param in enumerate(p_params):
        P_params[i] += p_param

def append_P(P__, P):  # pack P into P__ in top down sequence

    current_ys = [P_[0].y for P_ in P__]  # list of current-layer seg rows
    if P.y in current_ys:
        P__[current_ys.index(P.y)].append(P)  # append P row
    elif P.y > current_ys[0]:  # P.y > largest y in ys
        P__.insert(0, [P])
    elif P.y < current_ys[-1]:  # P.y < smallest y in ys
        P__.append([P])
    elif P.y < current_ys[0] and P.y > current_ys[-1]:  # P.y in between largest and smallest value
        for i, y in enumerate(current_ys):  # insert y if > next y
            if P.y > y: P__.insert(i, [P])  # PP.P__.insert(P.y - current_ys[-1], [P])


def sub_recursion_eval(PP_):  # evaluate each PP for rng+ and der+

    comb_layers = [[], []]  # no separate rng_comb_layers and der_comb_layers?

    for PP in PP_:  # PP is generic higher-composition pattern, P is generic lower-composition pattern
        mPP = dPP = 0
        ''' unpack tuple pairs first?
        for PP_params in PP.params[1:]:  # sum from all layers except the 1st layer：
            mPP += PP_params[0][0]
            dPP += PP_params[1][0]
        '''
        mrdn = dPP > mPP  # fork rdn, only applies if both forks are taken

        if mPP > ave_mPP * (PP.rdn + mrdn) and len(PP.P__) > (PP.rng+1) * 2:  # value of rng+ sub_recursion per PP
            m_comb_layers = sub_recursion(PP, base_rdn=PP.rdn+mrdn+1, fPd=0)
        else: m_comb_layers = [[], []]

        if dPP > ave_dPP * (PP.rdn +(not mrdn)) and len(PP.P__) > 3:  # value of der+, need 3 Ps to compute layer2, etc.
            d_comb_layers = sub_recursion(PP, base_rdn=PP.rdn+(not mrdn)+1, fPd=1)
        else: d_comb_layers = [[], []]

        PP.layers = [[], []]
        for i, (m_comb_layer, mm_comb_layer, dm_comb_layer) in \
                enumerate(zip_longest(comb_layers[0], m_comb_layers[0], d_comb_layers[0], fillvalue=[])):
            PP.layers[0] += [mm_comb_layer +  dm_comb_layer]
            m_comb_layers += [mm_comb_layer +  dm_comb_layer]
            if i > len(comb_layers[0][i])-1:  # new depth for comb_layers, pack new m_comb_layer
                comb_layers[0][i].append(m_comb_layers)

        for i, (d_comb_layer, dm_comb_layer, dd_comb_layer) in \
                enumerate(zip_longest(comb_layers[1], m_comb_layers[1], d_comb_layers[1], fillvalue=[])):
            PP.layers[1] += [dm_comb_layer +  dd_comb_layer]
            d_comb_layers += [dm_comb_layer + dd_comb_layer]
            if i > len(comb_layers[1][i])-1:  # new depth for comb_layers, pack new m_comb_layer
                comb_layers[1][i].append(d_comb_layers)

    return comb_layers


def sub_recursion(PP, base_rdn, fPd):  # compares param_layers of derPs in generic PP, form or accum top derivatives

    P__ = [P_ for P_ in reversed(PP.P__)]  # revert to top down

    if fPd: Pm__, Pd__ = comp_P_rng(P__, PP.rng+1)
    else:   Pm__, Pd__ = comp_P_der(P__)  # returns top-down

    sub_segm_ = form_seg_root(Pm__, base_rdn, fPd=0)
    sub_segd_ = form_seg_root(Pd__, base_rdn, fPd=1)  # returns bottom-up

    sub_PPm_, sub_PPd_ = form_PP_root((sub_segm_, sub_segd_), base_rdn)  # forms PPs: parameterized graphs of linked segs
    PPm_comb_layers, PPd_comb_layers = [[],[]], [[],[]]
    if sub_PPm_:
        PPm_comb_layers = sub_recursion_eval(sub_PPm_)  # rng+ comp_P in PPms -> param_layer, sub_PPs, rng+=n to skip clustering?
    if sub_PPd_:
        PPd_comb_layers = sub_recursion_eval(sub_PPd_)  # der+ comp_P in PPds -> param_layer, sub_PPs

    comb_layers = [[], []]
    # combine sub_PPm_s and sub_PPd_s from each layer:
    for m_sub_PPm_, d_sub_PPm_ in zip_longest(PPm_comb_layers[0], PPd_comb_layers[0], fillvalue=[]):
        comb_layers[0] += [m_sub_PPm_ + d_sub_PPm_]
    for m_sub_PPd_, d_sub_PPd_ in zip_longest(PPm_comb_layers[1], PPd_comb_layers[1], fillvalue=[]):
        comb_layers[1] += [m_sub_PPd_ + d_sub_PPd_]

    return comb_layers


def comp_P(_P, P, instance=CderP, finP=1, foutderP=1):  # forms vertical derivatives of params per P in _P.uplink, conditional ders from norm and DIV comp

    if finP:  # input is CderP
        _P_params = _P.params; P_params = P.params
    else:  # input is param layer
        _P_params = _P; P_params = P

    # compared P params:
    _x, _L, _M, _Ma, _I, _Dx, _Dy, _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _P_params
    x, L, M, Ma, I, Dx, Dy, sin_da0, cos_da0, sin_da1, cos_da1 = P_params

    dx = _x - x;  mx = ave_dx - abs(dx)  # mean x shift, if dx: rx = dx / ((L+_L)/2)? no overlap, offset = abs(x0 -_x0) + abs(xn -_xn)?
    dI = _I - I;  mI = ave_I - abs(dI)
    dM = _M - M;  mM = min(_M, M)
    dMa = _Ma - Ma;  mMa = min(_Ma, Ma)  # dG, dM are directional, re-direct by dx?
    dL = _L - L * np.hypot(dx, 1); mL = min(_L, L)  # if abs(dx) > ave: adjust L as local long axis, no change in G,M
    # G, Ga:
    G = np.hypot(Dy, Dx); _G = np.hypot(_Dy, _Dx)  # compared as scalars
    dG = _G - G;  mG = min(_G, G)
    Ga = (cos_da0 + 1) + (cos_da1 + 1); _Ga = (_cos_da0 + 1) + (_cos_da1 + 1)  # gradient of angle, +1 for all positives?
    # or Ga = np.hypot( np.arctan2(*Day), np.arctan2(*Dax)?
    dGa = _Ga - Ga;  mGa = min(_Ga, Ga)

    # comp angle:
    _sin = _Dy / (1 if _G==0 else _G); _cos = _Dx / (1 if _G==0 else _G)
    sin  = Dy / (1 if G==0 else G); cos = Dx / (1 if G==0 else G)
    sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
    dangle = np.arctan2(sin_da, cos_da)  # vertical difference between angles
    mangle = ave_dangle - abs(dangle)  # indirect match of angles, not redundant as summed

    # comp angle of angle: forms daa, not gaa?
    sin_dda0 = (cos_da0 * _sin_da0) - (sin_da0 * _cos_da0)
    cos_dda0 = (cos_da0 * _cos_da0) + (sin_da0 * _sin_da0)
    sin_dda1 = (cos_da1 * _sin_da1) - (sin_da1 * _cos_da1)
    cos_dda1 = (cos_da1 * _cos_da1) + (sin_da1 * _sin_da1)

    daangle = (sin_dda0, cos_dda0, sin_dda1, cos_dda1)
    # day = [-sin_dda0 - sin_dda1, cos_dda0 + cos_dda1]
    # dax = [-sin_dda0 + sin_dda1, cos_dda0 + cos_dda1]
    gay = np.arctan2( (-sin_dda0 - sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in y?
    gax = np.arctan2( (-sin_dda0 + sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in x?
    daangle = np.arctan2( gay, gax)  # probably wrong
    maangle = ave_daangle - abs(daangle)  # match between aangles, not redundant as summed

    dP = abs(dx)-ave_dx + abs(dI)-ave_I + abs(G)-ave_G + abs(Ga)-ave_Ga + abs(dM)-ave_M + abs(dMa)-ave_Ma + abs(dL)-ave_L
    # sum to evaluate for der+, abs diffs are distinct from directly defined matches:
    mP = mx + mI + mG + mGa + mM + mMa + mL + mangle + maangle

    params = [[mP, mx, mL, mI, mG, mGa, mM, mMa, mangle, maangle],
              [dP, dx, dL, dI, dG, dGa, dM, dMa, dangle, daangle]]

    if foutderP:
        # or summable params only, compute Gs at termination?
        x0 = min(_P.x0, P.x0)
        xn = max(_P.x0+_P.L, P.x0+P.L)
        L = xn-x0
        return instance(x0=x0, L=L, y=_P.y, params=params, P=P, _P=_P)

    else:
        return params


def accum_nested(_params, params):

    if len(params)>0 and isinstance(params[0], list):
        for i, (_sub_params, sub_params) in enumerate(zip_longest(_params, params, fillvalue=[])):
            accum_nested(_sub_params, sub_params)
            if i>len(_params)-1:  # add new layer of params
                _params.append(_sub_params)
    else:
        if not _params: _params[:] = [ 0 for param in params ]  # initialize _params if it is empty
        if len(params) == 11:  # P params : x, L, m, ma, I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1
            accum_p(_params, params)
        elif len(params) == 10:  # [mx, mL, mM, mMa, mI, mG, mGa, mangle, mP, maangle], [dx, dL, dM, dMa, dI, dG, dGa, dangle, dP, daangle]
            accum_ptuple(_params, params)

# below is temporary
# replace with inline derP initialization?
def comp_derP(_derP, derP, instance=CderP, finP=1, foutderP=1):

    # instance, finP, foutderP should not be needed, comp_params part should be done by comp_ptuple

    derivatives_t = []
    mP = 0  # for rng+ eval
    dP = 0  # for der+ eval

    if finP:
        if isinstance(_derP, CderP):  # params is in tuple of 2, each with 10 elements
            for _params, params in zip(_derP.params, derP.params):
                derivatives_t += [comp_ptuple(_params, params)]

        else:  # params is layered
            derivatives_t = [comp_P(_derP.params[0], derP.params[0], finP=0, foutderP=0)]
            mP += derivatives_t[0][0][0]  # 1st index = 1st layer, 2nd index select m | d, 3rd index selecting mP | dP
            dP += derivatives_t[0][1][0]
            for _params_layer, params_layer in zip(_derP.params[1:], derP.params[1:]):
                derivatives = []
                for _params, params in zip(_params_layer, params_layer):
                    derivatives += [comp_ptuple(_params, params)]
                derivatives_t += [derivatives]
                mP += derivatives[0][0]
                dP += derivatives[1][0]

    else:  # _derP and derP is params layer
        derivatives_t = []
        for _params, params in zip(_derP, derP):
            derivatives_t += [comp_ptuple(_params, params)]


    if foutderP:  # return derP instance
        x0 = min(_derP.x0, derP.x0)
        xn = max(_derP.x0 + _derP.L, derP.x0+derP.L)
        L = xn-x0

        dderP = instance(x0=x0, L=L, y=_derP.y, params=derivatives_t, P=derP, _P=_derP)
        return dderP
    else:  # return only the derivatives
        return derivatives_t


def comp_ptuple(_params, params):  # compare 2 10-tuples of params, as in comp_P, similar operations for m and d params

    # modify to unpack and separately compare common and differential subsets of lataple and vertuple
    derivatives = [[], []]

    _P, _x, _L, _I, _G, _Ga, _M, _Ma, _angle, _aangle = _params
    P, x, L, I, G, Ga, M, Ma, angle, aangle = params
    # P
    dP = _P - P; mP = ave_mP - abs(dP)
    derivatives[0].append(dP); derivatives[1].append(mP)
    # x
    dx = _x - x; mx = ave_dx - abs(dx)
    derivatives[0].append(dx); derivatives[1].append(mx)
    hyp = np.hypot(dx, 1)
    # L
    dL = _L - L/hyp;  mL = min(_L, L)
    derivatives[0].append(dL); derivatives[1].append(mL)
    # I
    dI = _I - I; mI = ave_I - abs(dI)
    derivatives[0].append(dI); derivatives[1].append(mI)
    # G
    dG = _G - G/hyp;  mG = min(_G, G)  # if comp_norm: reduce by hypot
    derivatives[0].append(dG); derivatives[1].append(mG)
    # Ga
    dGa = _Ga - Ga;  mGa = min(_Ga, Ga)
    derivatives[0].append(dGa); derivatives[1].append(mGa)
    # M
    dM = _M - M/hyp;  mM = min(_M, M)
    derivatives[0].append(dM); derivatives[1].append(mM)
    # Ma
    dMa = _Ma - Ma;  mMa = min(_Ma, Ma)
    derivatives[0].append(dMa); derivatives[1].append(mMa)
    # angle
    if isinstance(_angle, tuple):
        # (sin_da, cos_da)
         _sin_da, _cos_da = _angle; sin_da, cos_da = angle
         sin_dda = (cos_da * _sin_da) - (sin_da * _cos_da)  # sin(α - β) = sin α cos β - cos α sin β
         cos_dda = (cos_da * _cos_da) + (sin_da * _sin_da)  # cos(α - β) = cos α cos β + sin α sin β
         dangle = (sin_dda, cos_dda)  # da
         mangle = ave_dangle - abs(np.arctan2(sin_dda, cos_dda))  # ma is indirect match
         derivatives[0].append(dangle); derivatives[1].append(mangle)
    else:
        # scalar mangle
        _mangle = _angle; mangle = angle
        dmangle = _mangle - mangle;  mmangle = min(_mangle, mangle)
        derivatives[0].append(dmangle); derivatives[1].append(mmangle)
    # aangle
    if isinstance(_aangle, tuple):
        _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _aangle
        sin_da0, cos_da0, sin_da1, cos_da1 = aangle

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
        derivatives[0].append(daangle); derivatives[1].append(maangle)

    else:  # scalar maangle
        _maangle = _aangle; maangle = aangle
        dmaangle = _maangle - maangle;  mmaangle = min(_maangle, maangle)
        derivatives[0].append(dmaangle); derivatives[1].append(mmaangle)

    return derivatives  # tuple of 2, each with 2 tuple 10 params


def copy_P(P, Ptype):   # Ptype =0: P is CP | =1: P is CderP | =2: P is CPP | =3: P is CderPP

    uplink_layers, downlink_layers = P.uplink_layers, P.downlink_layers  # local copy of link layers
    P.uplink_layers, P.downlink_layers = [], []  # reset link layers
    seg = P.root  # local copy
    P.root = None
    if Ptype == 1:
        P_derP, _P_derP = P.P, P._P  # local copy of derP.P and derP._P
        P.P, P._P = None, None  # reset
    elif Ptype == 2:
        seg_levels = P.seg_levels
        PPP_levels = P.PPP_levels
    elif Ptype == 3:
        PP_derP, _PP_derP = P.PP, P._PP  # local copy of derP.P and derP._P
        P.PP, P._PP = None, None  # reset

    new_P = P.copy()  # copy P with empty root and link layers, reassign link layers:
    new_P.uplink_layers += uplink_layers + [[], []]
    new_P.downlink_layers += downlink_layers + [[], []]

    P.uplink_layers, P.downlink_layers = uplink_layers, downlink_layers  # reassign link layers
    P.root = seg  # reassign root
    # reassign other list params
    if Ptype == 1:
        new_P.P, new_P._P = P_derP, _P_derP
        P.P, P._P = P_derP, _P_derP
    elif Ptype == 2:
        P.seg_levels = seg_levels
        P.PPP_levels = PPP_levels
    elif Ptype == 3:
        new_P.PP, new_P._PP = PP_derP, _PP_derP
        P.PP, P._PP = PP_derP, _PP_derP

    return new_P