'''
Comp_slice is a terminal fork of intra_blob.
-
In natural images, objects look very fuzzy and frequently interrupted, only vaguely suggested by initial blobs and contours.
Potential object is proximate low-gradient (flat) blobs, with rough / thick boundary of adjacent high-gradient (edge) blobs.
These edge blobs can be dimensionality-reduced to their long axis / median line: an effective outline of adjacent flat blob.
-
Median line can be connected points that are most equidistant from other blob points, but we don't need to define it separately.
An edge is meaningful if blob slices orthogonal to median line form some sort of a pattern: match between slices along the line.
In simplified edge tracing we cross-compare among blob slices in x along y, where y is the longer dimension of a blob / segment.
Resulting patterns effectively vectorize representation: they represent match and change between slice parameters along the blob.
-
This process is very complex, so it must be selective. Selection should be by combined value of gradient deviation of edge blobs
and inverse gradient deviation of flat blobs. But the latter is implicit here: high-gradient areas are usually quite sparse.
A stable combination of a core flat blob with adjacent edge blobs is a potential object.
-
So, comp_slice traces blob axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)
-
Vectorization is clustering of parameterized Ps + their derivatives (derPs) into PPs: patterns of Ps that describe edge blob.
This process is a reduced-dimensionality (2D->1D) version of cross-comp and clustering cycle, common across this project.
As we add higher dimensions (3D and time), this dimensionality reduction is done in salient high-aspect blobs
(likely edges in 2D or surfaces in 3D) to form more compressed "skeletal" representations of full-dimensional patterns.
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
ave_nsub = 1

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
    n = lambda: 1  # accumulation count
    # only in lataple, for comparison but not summation:
    G = float
    Ga = float
    # only in vertuple, combined tuple m|d value:
    val = float

class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP

    ptuple = object  # x, L, I, M, Ma, G, Ga, angle( Dy, Dx), aangle( Sin_da0, Cos_da0, Sin_da1, Cos_da1)
    x0 = int
    y = int  # for vertical gap in PP.P__
    sign = NoneType  # g-ave + ave-ga sign, redundant to params?
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

    params = list  # P derivation layers, n ptuples = 2**der_cnt
    ptuple = lambda: Cptuple()
    x0 = int
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

    params = list  # P.ptuple + derP.ptuple if >1 P|seg, L is Area
    ptuple = lambda: Cptuple()
    sign = bool
    x0 = int  # box, update by max, min
    xn = int
    y0 = int
    yn = int
    rng = lambda: 1  # rng starts with 1
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs
    Rdn = int  # for accumulation only
    nP = int  # len 2D derP__ in levels[0][fPd]?  ly = len(derP__), also x, y?
    nderP = int
    rlayers = list  # or mlayers
    dlayers = list  # or alayers
    uplink_layers = lambda: [[],[]]
    downlink_layers = lambda: [[],[]]
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    mask__ = bool
    P__ = list  # input, includes derPs
    seg_levels = lambda: [[[]],[[]]]  # from 1st agg_recursion, seg_levels[0] is seg_, higher seg_levels are segP_..s
    # I don't think we need nesting above, top seg_level should be one type, depending on fPd?
    root = lambda:None  # higher-order PP, segP, or PPP

# Functions:

def comp_slice_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    segment_by_direction(blob, verbose=False)  # forms blob.dir_blobs
    for dir_blob in blob.dir_blobs:  # same-direction blob should be CBlob

        P__ = slice_blob(dir_blob, verbose=False)  # cluster dir_blob.dert__ into 2D array of blob slices
        # comp_dx_blob(P__), comp_dx?
        # form PPm_ is not revised yet, probably should be from generic P_
        Pm__ = comp_P_root(deepcopy(P__))  # scan_P_, comp_P | link_layer, adds mixed uplink_, downlink_ per P,
        Pd__ = comp_P_root(deepcopy(P__))  # deepcopy before assigning link derPs to Ps

        segm_ = form_seg_root(Pm__, root_rdn=2, fPd=0)  # forms segments: parameterized stacks of (P,derP)s
        segd_ = form_seg_root(Pd__, root_rdn=2, fPd=1)  # seg is a stack of (P,derP)s

        PPm_, PPd_ = form_PP_root((segm_, segd_), base_rdn=2)  # forms PPs: parameterized graphs of linked segs
        dir_blob.rlayers = [PPm_]; dir_blob.dlayers = [PPd_]

        if dir_blob.M > ave_mPP*dir_blob.rdn+1 and len(PPm_) > ave_nsub:
            dir_blob.rlayers += sub_recursion(PPm_, ave_mPP, fPd=0)  # rng+ comp_P in PPms -> param_layer, sub_PPs
        if dir_blob.G > ave_dPP*dir_blob.rdn+1 and len(PPd_) > ave_nsub:
            dir_blob.dlayers += sub_recursion(PPd_, ave_dPP, fPd=1)  # der+ comp_P in PPds -> param_layer, sub_PPs

        for PP_ in (PPm_, PPd_):  # 1st-call agglomerative recursion is per PP, appending PP.levels, not blob.levels:
            for PP in PP_:
                agg_recursion_eval(PP, fseg=1)  # 1st call, comp_seg -> segPs.. per seg__[n], in PP.levels
                # if fseg: use seg_levels, else agg_levels
        dir_blob.agg_levels = [[PPm_], [PPd_]]
        agg_recursion_eval(dir_blob, fseg=0)  # 2nd call, dir_blob.PP_s formed in 1st call, forms PPPs and dir_blob.levels

    splice_dir_blob_(blob.dir_blobs)  # draft


def agg_recursion_eval(blob, fseg):

    from agg_recursion import agg_recursion
    m_comb_levels, d_comb_levels = [[], []], [[], []]
    if fseg:
        PPm_, PPd_ = blob.seg_levels[0][-1], blob.seg_levels[1][-1]  # top level
    else:
        PPm_, PPd_ = blob.agg_levels[0][-1], blob.agg_levels[1][-1]  # top level

    if len(PPm_) > 1: m_comb_levels = agg_recursion(PPm_, fiPd=0)  # no n_extended, unconditional?
    if len(PPd_) > 1: d_comb_levels = agg_recursion(PPd_, fiPd=1)

    # combine PPP sub-levels:
    for m_sub_PPPm_, d_sub_PPPm_ in zip_longest(m_comb_levels[0], d_comb_levels[0], fillvalue=[]):
        if m_sub_PPPm_ or d_sub_PPPm_: blob.levels[-1][0] += [m_sub_PPPm_ + d_sub_PPPm_]

    for m_sub_PPPd_, d_sub_PPPd_ in zip_longest(m_comb_levels[1], d_comb_levels[1], fillvalue=[]):
        if m_sub_PPPd_ or d_sub_PPPd_: blob.levels[-1][1] += [m_sub_PPPd_ + d_sub_PPPd_]


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
                P_.append( CP(ptuple=params, x0=x-(params.L-1), y=y, dert_=Pdert_))
            _mask = mask

        if not _mask:  # pack last P, same as above:
            params.L = len(Pdert_)
            params.G = np.hypot(*params.angle)
            params.Ga = (params.aangle[1] + 1) + (params.aangle[3] + 1)
            P_.append(CP(ptuple=params, x0=x - (params.L - 1), y=y, dert_=Pdert_))
        P__ += [P_]

    blob.P__ = P__
    return P__


def comp_P_root(P__):  # vertically compares y-adjacent and x-overlapping Ps: blob slices, forming 2D derP__

    _P_ = P__[0]  # upper row, top-down
    for P_ in P__[1:]:  # lower row
        for P in P_:
            for _P in _P_:  # test for x overlap(_P,P) in 8 directions, derts are positive in all Ps:
                if (P.x0 - 1 < _P.x0 + _P.ptuple.L) and (P.x0 + P.ptuple.L + 1 > _P.x0):
                    derP = comp_P(_P, P)
                    P.uplink_layers[-2] += [derP]  # append derPs, uplink_layers[-1] is match_derPs
                    _P.downlink_layers[-2] += [derP]
                elif (P.x0 + P.ptuple.L) < _P.x0:
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

def form_PP_(PP_segs, link_, fPd, fup):  # flood-fill PP_segs with vertically linked segments:
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

    seg = CPP(x0=seg_Ps[0].x0, P__= seg_Ps, uplink_layers=[miss_uplink_], downlink_layers = [miss_downlink_],
              L = len(seg_Ps), y0 = seg_Ps[0].y)  # seg.L is Ly
    iP = seg_Ps[0]
    if isinstance(iP, CPP): accum = accum_PP  # in agg+
    else: accum = accum_P   # iP is CP or CderP in der+

    for P in seg_Ps[:-1]:
        accum(seg, P, fPd)
        derP = P.uplink_layers[-1][0]
        if len(seg.params)>1: sum_layers(seg.params[1][0], derP.params, seg.ptuple)  # derP.params maybe nested
        else:
            seg.params.append([deepcopy(derP.params)])  # init 2nd layer
            accum_ptuple(seg.ptuple, derP.ptuple)
        derP.root = seg
    accum(seg, seg_Ps[-1], fPd)   # top P uplink_layers are not part of seg

    return seg


def sum2PP(PP_segs, base_rdn, fPd):  # sum PP_segs into PP

    PP = CPP(x0=PP_segs[0].x0, rdn=base_rdn, sign=PP_segs[0].sign, L= len(PP_segs))
    PP.seg_levels[fPd][0] = PP_segs  # PP_segs is levels[0]

    for seg in PP_segs:
        accum_PP(PP, seg, fPd)

    return PP

def accum_P(seg, P, _):  # P is derP if from der+

    if seg.params:
        if isinstance(P, CderP):
            sum_layers(seg.params[0], P.ptuple)  # possibly multiple layers in derP.params
        else:
            accum_ptuple(seg.params[0], P.ptuple)
    else:
        seg.params.append(deepcopy(P.ptuple))  # init 1st level of seg.params with P.ptuple
        seg.x0 = P.x0

    accum_ptuple(seg.ptuple, P.ptuple)
    P.root = seg

    if isinstance(P, CderP): L = P.P.ptuple.L
    else:                    L = P.ptuple.L
    seg.x0 = min(seg.x0, P.x0)
    seg.y0 = min(seg.y0, P.y)
    seg.xn = max(seg.xn, P.x0 + L)
    seg.yn = max(seg.yn, P.y)


def accum_PP(PP, inp, fPd):  # comp_slice inp is seg, or segPP in agg+

    sum_levels(PP.params, inp.params, PP.ptuple)  # PP has only 2 param levels, PP.params.x += inp.params.x*inp.L

    inp.root = PP
    PP.x0 = min(PP.x0, inp.x0)
    PP.xn = max(PP.xn, inp.xn)
    PP.y0 = min(inp.y0, PP.y0)
    PP.yn = max(inp.yn, PP.yn)
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
            else:
                append_P(PP.P__, P)  # add P into nested list of P__

            # add terminated seg links for rng+:
            for derP in inp.P__[0].downlink_layers[-1]:  # if downlink not in current PP's downlink and not part of the seg in current PP:
                if derP not in PP.downlink_layers[-1] and derP.P.root not in PP.seg_levels[fPd][-1]:
                    PP.downlink_layers[-1] += [derP]
            for derP in inp.P__[-1].uplink_layers[-1]:  # if downlink not in current PP's downlink and not part of the seg in current PP:
                if derP not in PP.downlink_layers[-1] and derP.P.root not in PP.seg_levels[fPd][-1]:
                    PP.uplink_layers[-1] += [derP]


def accum_ptuples(Ptuple, ptuples):  # sum all ptuples into Ptuple

    if isinstance(ptuples, Cptuple):
        accum_ptuple(Ptuple, ptuples)
    else:
        for ptuple in ptuples:
            accum_ptuples(Ptuple, ptuple)

def accum_ptuple(Ptuple, ptuple):  # lataple or vertuple

    Ptuple.accum_from(ptuple, excluded=["angle", "aangle"])
    fAngle = isinstance(Ptuple.angle, list)
    fangle = isinstance(ptuple.angle, list)

    if fAngle and fangle:  # both are latuples:
        for i, param in enumerate(ptuple.angle): Ptuple.angle[i] += param  # always in vector representation
        for i, param in enumerate(ptuple.aangle): Ptuple.aangle[i] += param

    elif not fAngle and not fangle:  # both are vertuples:
        Ptuple.angle += ptuple.angle
        Ptuple.aangle += ptuple.aangle


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


def comp_layers(_layers, layers, der_layers, fsubder):  # recursively unpack layers: m,d ptuple pairs, if any from der+

    if isinstance(_layers, Cptuple):  # 1st-level layers is latuple
        der_layers += comp_ptuple(_layers, layers)

    elif isinstance(_layers[0], Cptuple):  # layers is two vertuples, 1st level in der+
        dtuple = comp_ptuple(_layers[1], _layers[1])

        if fsubder:  # sub_recursion mtuples are not compared
            der_layers += [dtuple]
        else:
            mtuple = comp_ptuple(_layers[0], _layers[0])
            der_layers += [mtuple, dtuple]

    else:  # keep unpacking layers:
        for _layer, layer in zip(_layers, layers):
            der_layers += [comp_layers(_layer, layer, der_layers, fsubder=fsubder)]

    return der_layers  # m,d ptuple pair in each layer, possibly nested


def sum_levels(Params, params, Ptuple):  # Capitalized names for sums, as comp_levels but no separate der_layers to return

    if Params:
        sum_layers(Params[0], params[0], Ptuple)  # recursive unpack of nested ptuple layers, if any from der+
    else:
        Params.append(deepcopy(params[0]))  # no need to sum
        accum_ptuples(Ptuple, params[0])

    for Level, level in zip_longest(Params[1:], params[1:], fillvalue=[]):
        if Level and level:
            sum_levels(Level, level, Ptuple)  # recursive unpack of higher levels, if any from agg+ and nested with sub_levels
        elif level:
            Params.append(deepcopy(level))  # no need to sum
            accum_ptuples(Ptuple, level)


def sum_layers(Layers, layers, Ptuple):  # recursively unpack layers: m,d tuple pairs from der+

    if isinstance(layers, Cptuple):   # layers is a latuple, in 1st layer only
        accum_ptuple(Ptuple, layers)
        accum_ptuple(Layers, layers)
    else:
        # layer is layers or two vertuples, keep unpacking
        for Layer, layer in zip_longest(Layers, layers, fillvalue=[]):
            if Layer and layer:
                sum_layers(Layer, layer, Ptuple)
            elif layer:
                Layers.append(deepcopy(layer))
                sum_layers([], layer, Ptuple)  #?

    return Ptuple


def comp_P(_P, P, fsubder=0):  # forms vertical derivatives of params per P in _P.uplink, conditional ders from norm and DIV comp

    if isinstance(_P.ptuple, Cptuple):  # just to save the call, or this testing can be done in comp_layers
        derivatives = comp_ptuple(_P.ptuple, P.ptuple)  # comp lataple (P)
    else:
        derivatives = comp_layers(_P.ptuple, P.ptuple, [], fsubder=fsubder)  # comp vertuple pairs (derP)

    if isinstance(_P, CderP):
        _L = _P.P.ptuple.L; L = P.P.ptuple.L
    else:
        _L = _P.ptuple.L; L = P.ptuple.L
    #?
    x0 = min(_P.x0, P.x0)
    xn = max(_P.x0+_L, P.x0+L)  # i guess this is not needed?

    ptuple = Cptuple()
    sum_layers([], derivatives, ptuple)

    return CderP(x0=x0, L=L, y=_P.y, params=derivatives, ptuple=ptuple, P=P, _P=_P)


def comp_ptuple(_params, params):  # compare lateral or vertical tuples, similar operations for m and d params

    dtuple, mtuple = Cptuple(), Cptuple()
    dval, mval = 0, 0

    flatuple = isinstance(_params.angle, list)  # else vertuple
    rn = _params.n / params.n  # normalize param as param*rn, for n-invariant ratio of compared params:
    # _param / param*rn = (_param/_n) / (param/n)?
    # same set:
    comp("I", _params.I, params.I*rn, dval, mval, dtuple, mtuple, ave_dI, finv=flatuple)  # inverse match if latuple
    comp("x", _params.x, params.x*rn, dval, mval, dtuple, mtuple, ave_dx, finv=flatuple)
    hyp = np.hypot(dtuple.x, 1)  # dx, project param orthogonal to blob axis:
    comp("L", _params.L, params.L*rn / hyp, dval, mval, dtuple, mtuple, ave_L, finv=0)
    comp("M", _params.M, params.M*rn / hyp, dval, mval, dtuple, mtuple, ave_M, finv=0)
    comp("Ma",_params.Ma, params.Ma*rn / hyp, dval, mval, dtuple, mtuple, ave_Ma, finv=0)
    # diff set
    if flatuple:
        comp("G", _params.G, params.G*rn / hyp, dval, mval, dtuple, mtuple, ave_G, finv=0)
        comp("Ga", _params.Ga, params.Ga*rn / hyp, dval, mval, dtuple, mtuple, ave_Ga, finv=0)
        # angle:
        _Dy,_Dx = _params.angle[:]; Dy,Dx = params.angle[:]
        _G = np.hypot(_Dy,_Dx); G = np.hypot(Dy*rn,Dx*rn)
        sin = Dy*rn / (.1 if G == 0 else G); cos = Dx*rn / (.1 if G == 0 else G)
        _sin = _Dy / (.1 if _G == 0 else _G); _cos = _Dx / (.1 if _G == 0 else _G)
        sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
        cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
        # dangle is scalar now?
        dangle = np.arctan2(sin_da, cos_da)  # vertical difference between angles
        mangle = ave_dangle - abs(dangle)  # inverse match, not redundant as summed
        dtuple.angle = dangle; mtuple.angle= mangle
        dval += dangle; mval += mangle

        # angle of angle:
        _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _params.aangle
        sin_da0, cos_da0, sin_da1, cos_da1 = params.aangle
        sin_dda0 = (cos_da0*rn * _sin_da0) - (sin_da0*rn * _cos_da0)
        cos_dda0 = (cos_da0*rn * _cos_da0) + (sin_da0*rn * _sin_da0)
        sin_dda1 = (cos_da1*rn * _sin_da1) - (sin_da1*rn * _cos_da1)
        cos_dda1 = (cos_da1*rn * _cos_da1) + (sin_da1*rn * _sin_da1)
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
        comp("val", _params.val, params.val*rn / hyp, dval, mval, dtuple, mtuple, ave_mval, finv=0)
        comp("angle", _params.angle, params.angle*rn / hyp, dval, mval, dtuple, mtuple, ave_dangle, finv=0)
        comp("aangle", _params.aangle, params.aangle*rn / hyp, dval, mval, dtuple, mtuple, ave_daangle, finv=0)

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
        layers = P.layers
        P.seg_levels, P.layers = [], []  # reset
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
                            # test x overlap
                             if (dir_blob.x0 - 1 < _dir_blob.xn and dir_blob.xn + 1 > _dir_blob.x0) \
                                or (_dir_blob.x0 - 1 < dir_blob.xn and _dir_blob.xn + 1 > dir_blob.x0):
                                 splice_2dir_blobs(_dir_blob, dir_blob)  # splice dir_blob into _dir_blob
                                 dir_blobs[j] = _dir_blob

def splice_2dir_blobs(_blob, blob):
    # merge blob into _blob here
    pass


def sub_recursion_eval(PP):  # evaluate each PP for rng+ and der+

    sub_PPm_, sub_PPd_ = PP.rlayers[-1], PP.dlayers[-1]

    if sub_PPm_>ave_nsub:
        PP.rlayers += sub_recursion(sub_PPm_, ave_mPP, fPd=0)  # rng+ comp_P in PPms -> param_layer, sub_PPs, rng+=n to skip clustering?
    if sub_PPd_>ave_nsub:
        PP.dlayers += sub_recursion(sub_PPd_, ave_dPP, fPd=1)  # der+ comp_P in PPds -> param_layer, sub_PPs


def sub_recursion(PP_, ave, fPd):  # evaluate each PP for rng+ and der+

    comb_layers = []  # combined rng_comb_layers, der_comb_layers

    for PP in PP_:  # PP is generic higher-composition pattern, P is generic lower-composition pattern

        P__ =  [P_ for P_ in reversed(PP.P__)]  # revert to top down
        if fPd: Pm__, Pd__ = comp_P_der(P__)  # returns top-down
        else:   Pm__, Pd__ = comp_P_rng(P__, PP.rng+1)

        PP.rdn += 2  # 2 sub-clustering forks, unless select stronger?
        sub_segm_ = form_seg_root(Pm__, root_rdn=PP.rdn, fPd=0)
        sub_segd_ = form_seg_root(Pd__, root_rdn=PP.rdn, fPd=1)  # returns bottom-up

        sub_PPm_, sub_PPd_ = form_PP_root((sub_segm_, sub_segd_), PP.rdn+1)  # forms PPs: parameterized graphs of linked segs
        PP.rlayers = [sub_PPm_]; PP.dlayers = [sub_PPd_]

        if PP.ptuple.val > ave*PP.rdn:
            # we need separate PP.mtuple and PP.dtuple for these evaluations:
            if len(sub_PPm_) > ave_nsub:
                PP.rlayers += sub_recursion(sub_PPm_, ave_mPP, fPd=0)  # rng+ comp_P in PPms -> param_layer, sub_PPs, rng+=n to skip clustering?
            if len(sub_PPd_) > ave_nsub:
                PP.dlayers += sub_recursion(sub_PPd_, ave_dPP, fPd=1)  # der+ comp_P in PPds -> param_layer, sub_PPs

        for i, (comb_layer, rlayer, dlayer) in enumerate(zip_longest(comb_layers, PP.rlayers, PP.dlayers, fillvalue=[])):
            if i > len(comb_layers)-1:  # pack new comb_layer, if any
                comb_layers.append(rlayer+dlayer)
            else:
                comb_layers[i] += rlayer+dlayer  # layers element is m|d pair

    return comb_layers
