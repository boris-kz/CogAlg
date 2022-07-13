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
ave_dI = ave_inv
ave_M = ave  # replace the rest with coefs:
ave_Ma = 10
ave_G = 10
ave_Ga = 2  # related to dx?
ave_L = 10
ave_dx = 5  # inv, difference between median x coords of consecutive Ps
ave_dangle = 2  # vertical difference between angles
ave_daangle = 2
ave_mval = ave_dval = 10  # should be different
ave_mPP = 10
ave_dPP = 10
ave_splice = 10

param_names = ["x", "I", "M", "Ma", "L", "angle", "aangle"]
aves = [ave_dx, ave_dI, ave_M, ave_Ma, ave_L, ave_G, ave_Ga, ave_mval, ave_dval]
vaves = [ave_mval, ave_dval]


class Cptuple(ClusterStructure):  # bottom-layer tuple of lateral or vertical params: lataple in P or vertuple in derP

    # add prefix v in vertuples: 9 vs 10 params:
    x = int
    L = int  # area in PP
    I = int
    M = int
    Ma = float
    angle = lambda: [0, 0]  # in lataple only, replaced by float in vertuple
    aangle = lambda: [0, 0, 0, 0]
    n = int  # accumulation count
    # only in lataple, for comparison but not summation:
    G = float
    Ga = float
    # only in vertuple, combined tuple m|d value:
    val = float


class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP

    params = object  # x, L, I, M, Ma, G, Ga, angle( Dy, Dx), aangle( Sin_da0, Cos_da0, Sin_da1, Cos_da1)
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

    params = list  # P derivation pair_layers, n ptuples = 2**der_cnt
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

    params = list  # P.params (ptuple), += derP params if >1 P|seg, L is Area
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
    layers = list  # elements for sub_recursion, each is derP_t
    root = lambda:None  # higher-order PP, segP, or PPP

# Functions:

def comp_slice_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    from agg_recursion import agg_recursion

    segment_by_direction(blob, verbose=False)  # forms blob.dir_blobs
    for dir_blob in blob.dir_blobs:  # dir_blob should be CBlob

        P__ = slice_blob(dir_blob, verbose=False)  # cluster dir_blob.dert__ into 2D array of blob slices
        # comp_dx_blob(P__), comp_dx?
        # form PPm_ is not revised yet, probably should be from generic P_
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

    splice_dir_blob_(blob.dir_blobs)  # draft


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
        for x, (dert, mask) in enumerate(zip(dert_, mask_)):  # dert = i, g, ga, ri, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1

            if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()
            g, ga, ri, angle, aangle = dert[1], dert[2], dert[3], list(dert[4:6]), list(dert[6:])
            if not mask:  # masks: if 0,_1: P initialization, if 0,_0: P accumulation, if 1,_0: P termination
                if _mask:  # initialize P params with first unmasked dert (m, ma, i, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1):
                    Pdert_ = [dert]
                    params = Cptuple(M=ave_g-g,Ma=ave_ga-ga,I=ri, angle=angle, aangle=aangle)
                else:
                    # dert and _dert are not masked, accumulate P params:
                    params.M+=ave_g-g; params.Ma+=ave_ga-ga; params.I+=ri; params.angle[0]+=angle[0]; params.angle[1]+=angle[1]
                    params.aangle = [sum(aangle_tuple) for aangle_tuple in zip(params.aangle, aangle)]
                    Pdert_.append(dert)
            elif not _mask:
                # _dert is not masked, dert is masked, terminate P:
                params.L = len(Pdert_)  # G, Ga are recomputed; M, Ma are not restorable from G, Ga:
                params.G = np.hypot(*params.angle)  # Dy, Dx
                params.Ga = (params.aangle[1] + 1) + (params.aangle[3] + 1)  # Cos_da0, Cos_da1
                P_.append( CP(params=params, x0=x-(params.L-1), y=y, dert_=Pdert_))
            _mask = mask

        if not _mask:  # pack last P, same as above:
            params.L = len(Pdert_)
            params.G = np.hypot(*params.angle)
            params.Ga = (params.aangle[1] + 1) + (params.aangle[3] + 1)
            P_.append(CP(params=params, x0=x - (params.L - 1), y=y, dert_=Pdert_))
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

    for P_ in P__:
        for P in P_:  # add 2 link layers: rng_derP_ and match_rng_derP_:
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
                    derP = comp_P(_P, P)
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
                    dderP = comp_P(_derP, derP, fsubder=1)  # form higher vertical derivatives of derP or PP params
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
    for i, derP in enumerate( sorted( link_layers[-2], key=lambda derP: derP.params[fPd].val, reverse=True)):

        if fPd: derP.rdn += derP.params[fPd].val > derP.params[1-fPd].val  # mP > dP
        else: rng_eval(derP, fPd)  # reset derP.val, derP.rdn

        if derP.params[fPd].val > vaves[fPd] * derP.rdn * (i+1):  # ave * rdn to stronger derPs in link_layers[-2]
            link_layers[-1].append(derP)  # misses = link_layers[-2] not in link_layers[-1]


def rng_eval(derP, fPd):  # compute value of combined mutual derPs: overlap between P uplinks and _P downlinks

    _P, P = derP._P, derP.P
    common_derP_ = []

    for _downlink_layer, uplink_layer in zip(_P.downlink_layers, P.uplink_layers):  # overlap in P uplinks and _P downlinks
        common_derP_ += list( set(_downlink_layer).intersection(uplink_layer))  # get common derP in mixed uplinks
    rdn = 1
    olp_val = 0
    for derP in common_derP_:
        rdn += derP.params[fPd].val > derP.params[1-fPd].val  # dP > mP if fPd, else mP > dP
        olp_val += derP.params[fPd].val

    nolp = len(common_derP_)
    derP.params[fPd].val = olp_val / nolp
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
              L = len(seg_Ps), y0 = seg_Ps[0].y, params=[[]])  # seg.L is Ly
    iP = seg_Ps[0]
    if isinstance(iP, CderP): accum = accum_derP
    elif isinstance(iP, CPP): accum = accum_PP  # 2 layers only
    else: accum = accum_P  # iP is CP

    seg.params[0] = deepcopy(seg_Ps[0].params)  # init seg params with 1st P and derP
    seg_Ps[0].root = seg
    seg.x0 = min(seg.x0, seg_Ps[0].x0)

    if len(seg_Ps)>1:
        seg.params += [[deepcopy(seg_Ps[0].uplink_layers[-1][0].params)]]   # add nested layer in params if P was compared
        accum(seg, seg_Ps[-1], fPd)  # accum last P
    # else seg.params = seg_Ps[0].params

    for P in seg_Ps[1:-1]:  # skip 1st and last P
        accum(seg, P, fPd)
        derP = P.uplink_layers[-1][0]
        seg.params[1][0] = sum_pair_layers(seg.params[1][0], derP.params, n=0)  # derP.params maybe nested
        derP.root = seg

    return seg


def sum2PP(PP_segs, base_rdn, fPd):  # sum params: derPs into segment or segs into PP

    PP = CPP(x0=PP_segs[0].x0, rdn=base_rdn, sign=PP_segs[0].sign, L= len(PP_segs))
    PP.seg_levels[fPd][0] = PP_segs  # PP_segs is seg_levels[0]

    for seg in PP_segs:
        accum_PP(PP, seg, fPd)

    return PP

def accum_P(seg, P, fPd):

    accum_ptuple(seg.params[0], P.params, n=0)
    P.root = seg
    seg.x0 = min(seg.x0, P.x0)

def accum_derP(PP, inp, fPd):  # inp is seg or PP in recursion

    PP.params = sum_pair_layers(PP.params, inp.params, n=0)
    inp.root = PP
    # may add more assignments here

def accum_PP(PP, inp, fPd):  # comp_slice inp is seg, PP in agg+ only

    accum_ptuple(PP.params[0], inp.params[0], n=0)  # PP has two param layers
    # conditional, 2nd layer is empty if single seg|P?:
    PP.params[1] = sum_pair_layers(PP.params[1], inp.params[1], n=0)
    '''
    for PPP:
    for i, (PP_params, inp_params) in enumerate(zip_longest(PP.params, inp.params, fillvalue=[])):
        if not PP_params: PP_params = deepcopy(inp_params)    # if PP's current layer params is empty, copy from input
        else: sum_layers([PP_params], [inp_params])           # accumulate params
        if i > len(PP.params)-1: PP.params.append(PP_params)  # pack new layer
    '''
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


def comp_pair_layers(_pair_layers, pair_layers, der_pair_layers, fsubder):  # recursively unpack nested m,d tuple pairs, if any from der+

    if isinstance(_pair_layers, Cptuple):
        der_pair_layers += comp_ptuple(_pair_layers, pair_layers)  # 1st-layer pair_layers is latuple

    elif isinstance(_pair_layers[0], Cptuple):  # pairs is two vertuples, 1st layer in der+
        dtuple = comp_ptuple(_pair_layers[1], _pair_layers[1])
        if fsubder:  # sub_recursion mtuples are not compared
            der_pair_layers += [dtuple]
        else:
            mtuple = comp_ptuple(_pair_layers[0], _pair_layers[0])
            der_pair_layers += [mtuple, dtuple]

    else:  # keep unpacking pair_layers:
        for _pair, pair in zip(_pair_layers, pair_layers):
            der_pair_layers += [comp_pair_layers(_pair, pair, der_pair_layers, fsubder=fsubder)]

    return der_pair_layers  # possibly nested m,d ptuple pairs


def sum_pair_layers(Pairs, pairs, n):  # recursively unpack pairs (short for pair_layers): m,d tuple pairs from der+

    if isinstance(Pairs, Cptuple):
        Pairs = accum_ptuple(Pairs, pairs, n)  # pairs is a latuple, in 1st layer only

    elif isinstance(Pairs[0], Cptuple):  # pairs is two vertuples, 1st layer in der+

        Pairs[0] = accum_ptuple(Pairs[0], pairs[0], n)
        Pairs[1] = accum_ptuple(Pairs[1], pairs[1], n)

    else:  # pair is pair_layers, keep unpacking:
        for Pair, pair, n in zip(Pairs, pairs):
            sum_pair_layers(Pair, pair, n)

    return Pairs

def accum_ptuple(Ptuple, ptuple, n):  # lataple or vertuple

    Ptuple.accum_from(ptuple, excluded=["angle", "aangle"])

    if isinstance(Ptuple.angle, list):  # latuple:
        for i, param in enumerate(ptuple.angle): Ptuple.angle[i] += param  # always in vector representation
        for i, param in enumerate(ptuple.aangle): Ptuple.aangle[i] += param
    else:
        Ptuple.angle += ptuple.angle
        Ptuple.aangle += ptuple.aangle

    if n: Ptuple = [Ptuple, n + 1]  # for ave_layers
    else: Ptuple.n += 1

    return Ptuple


def comp_P(_P, P, fsubder=0):  # forms vertical derivatives of params per P in _P.uplink, conditional ders from norm and DIV comp

    if isinstance(_P.params, Cptuple):  # just to save the call, or this testing can be done in comp_pair_layers
        derivatives = comp_ptuple(_P.params, P.params)  # comp lataple (P)
    else:
        derivatives = comp_pair_layers(_P.params, P.params, [], fsubder=fsubder)  # comp vertuple pairs (derP)

    x0 = min(_P.x0, P.x0)
    xn = max(_P.x0+_P.L, P.x0+P.L)
    L = xn-x0

    return CderP(x0=x0, L=L, y=_P.y, params=derivatives, P=P, _P=_P)


def comp_ptuple(_params, params):  # compare lateral or vertical tuples, similar operations for m and d params

    dtuple, mtuple = Cptuple(), Cptuple()
    dval, mval = 0, 0

    flatuple = isinstance(_params.angle, list)  # else vertuple
    # same set:
    comp("I", _params.I, params.I, dval, mval, dtuple, mtuple, ave_dI, finv=flatuple)  # inverse match if latuple
    comp("x", _params.x, params.x, dval, mval, dtuple, mtuple, ave_dx, finv=flatuple)
    hyp = np.hypot(dtuple.x, 1)  # dx, project param orthogonal to blob axis:
    comp("L", _params.L, params.L / hyp, dval, mval, dtuple, mtuple, ave_L, finv=0)
    comp("M", _params.M, params.M / hyp, dval, mval, dtuple, mtuple, ave_M, finv=0)
    comp("Ma",_params.Ma, params.Ma / hyp, dval, mval, dtuple, mtuple, ave_Ma, finv=0)
    # diff set
    if flatuple:
        comp("G", _params.G, params.G / hyp, dval, mval, dtuple, mtuple, ave_G, finv=0)
        comp("Ga", _params.Ga, params.Ga / hyp, dval, mval, dtuple, mtuple, ave_Ga, finv=0)
        # angle:
        _Dy,_Dx = _params.angle[:]; Dy,Dx = params.angle[:]
        _G = np.hypot(_Dy,_Dx); G = np.hypot(Dy,Dx)
        sin = Dy / (.1 if G == 0 else G); cos = Dx / (.1 if G == 0 else G)
        _sin = _Dy / (.1 if _G == 0 else _G); _cos = _Dx / (.1 if _G == 0 else _G)
        sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
        cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
        dangle = np.arctan2(sin_da, cos_da)  # vertical difference between angles
        mangle = ave_dangle - abs(dangle)  # inverse match, not redundant as summed
        dtuple.angle = dangle; mtuple.angle= mangle
        dval += dangle; mval += mangle

        # angle of angle:
        _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _params.aangle
        sin_da0, cos_da0, sin_da1, cos_da1 = params.aangle
        sin_dda0 = (cos_da0 * _sin_da0) - (sin_da0 * _cos_da0)
        cos_dda0 = (cos_da0 * _cos_da0) + (sin_da0 * _sin_da0)
        sin_dda1 = (cos_da1 * _sin_da1) - (sin_da1 * _cos_da1)
        cos_dda1 = (cos_da1 * _cos_da1) + (sin_da1 * _sin_da1)
        # for 2D, not reduction to 1D:
        # aaangle = (sin_dda0, cos_dda0, sin_dda1, cos_dda1)
        # day = [-sin_dda0 - sin_dda1, cos_dda0 + cos_dda1]
        # dax = [-sin_dda0 + sin_dda1, cos_dda0 + cos_dda1]
        gay = np.arctan2( (-sin_dda0 - sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in y?
        gax = np.arctan2( (-sin_dda0 + sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in x?
        daangle = np.arctan2(gay, gax)  # diff between aangles, probably wrong
        maangle = ave_daangle - abs(daangle)  # inverse match, not redundant as summed
        dtuple.aangle = daangle; mtuple.aangle = maangle
        dval += abs(daangle); mval += maangle

    else:  # vertuple, all ders are scalars:
        comp("val", _params.val, params.val / hyp, dval, mval, dtuple, mtuple, ave_mval, finv=0)
        comp("angle", _params.angle, params.angle / hyp, dval, mval, dtuple, mtuple, ave_dangle, finv=0)
        comp("aangle", _params.aangle, params.aangle / hyp, dval, mval, dtuple, mtuple, ave_daangle, finv=0)

    mtuple.val = mval; dtuple.val = dval

    return [mtuple, dtuple]

def comp(param_name, _param, param, dval, mval, dtuple, mtuple, ave, finv):

    d = _param-param
    if finv: m = ave - abs(d)  # inverse match for primary params, no mag/value correlation
    else:    m = min(_param,param) - ave
    dval += abs(d)
    mval += m
    setattr(dtuple, param_name, d)  # dtuple.param_name = d
    setattr(mtuple, param_name, m)  # mtuple.param_name = m


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
        layers = P.layers
        P.seg_levels, P.PPP_levels, P.layers = [], [], []  # reset
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
        P.layers = layers
        new_P.layers = layers
    elif Ptype == 3:
        new_P.PP, new_P._PP = PP_derP, _PP_derP
        P.PP, P._PP = PP_derP, _PP_derP

    return new_P

# old draft
def splice_dir_blob_(dir_blobs):

    for i, _dir_blob in enumerate(dir_blobs):
        for fPd in 0, 1:
            PP_ = _dir_blob.levels[0][fPd]

            if fPd: PP_val = sum([PP.mP for PP in PP_])
            else:   PP_val = sum([PP.dP for PP in PP_])

            if PP_val - ave_splice > 0:  # high mPP pr dPP

                _top_P_ = _dir_blob.P__[0]
                _bottom_P_ = _dir_blob.P__[-1]

                for j, dir_blob in enumerate(dir_blobs):
                    if _dir_blob is not dir_blob:

                        top_P_ = dir_blob.P__[0]
                        bottom_P_ = dir_blob.P__[-1]
                        # test y adjacency
                        if (_top_P_[0].y-1 == bottom_P_[0].y) or (top_P_[0].y-1 == _bottom_P_[0].y):
                            # tet x overlap
                             _x0 = min([_P.x0 for _P_ in _dir_blob.P__ for _P in _P_])
                             _xn = min([_P.x0+_P.L for _P_ in _dir_blob.P__ for _P in _P_])
                             x0 = min([P.x0 for P_ in dir_blob.P__ for P in P_])
                             xn = min([P.x0+_P.L for P_ in dir_blob.P__ for P in P_])
                             if (x0 - 1 < _xn and xn + 1 > _x0) or  (_x0 - 1 < xn and _xn + 1 > x0) :
                                 splice_2dir_blobs(_dir_blob, dir_blob)  # splice dir_blob into _dir_blob
                                 dir_blobs[j] = _dir_blob

def splice_2dir_blobs(_blob, blob):
    # merge blob into _blob here
    pass