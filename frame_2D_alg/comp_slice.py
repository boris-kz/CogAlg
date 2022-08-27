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
ave_agg = 3

param_names = ["x", "I", "M", "Ma", "L", "angle", "aangle"]
aves = [ave_dx, ave_dI, ave_M, ave_Ma, ave_L, ave_G, ave_Ga, ave_mval, ave_dval]
vaves = [ave_mval, ave_dval]
PP_aves = [ave_mPP, ave_dPP]


class Cptuple(ClusterStructure):  # bottom-layer tuple of lateral or vertical params: lataple in P or vertuple in derP

    # add prefix v in vertuples: 9 vs. 10 params:
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

class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

    ptuple = object  # x, L, I, M, Ma, G, Ga, angle( Dy, Dx), aangle( Sin_da0, Cos_da0, Sin_da1, Cos_da1), n, val
    x0 = int
    y = int  # for vertical gap in PP.P__
    rdn = int  # blob-level redundancy, ignore for now
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
    '''
    Derivation forms a binary tree where the root is latuple and all forks are vertuples.
    Players in derP are zipped with fds: taken forks. Players, mplayer, dplayer are replaced by those in PP
    '''
    players = list  # max n ptuples in layer = n ptuples in all lower layers: 1, 1, 2, 4, 8...
    valt = lambda: [0,0]  # tuple of mval, dval, summed from last player, both are signed
    x0 = int
    y = int  # or while isinstance(P, CderP): P = CderP._P; else y = _P.y
    _P = object  # higher comparand
    P = object  # lower comparand
    root = lambda:None  # segment in sub_recursion
    # higher derivatives
    rdn = int  # mrdn, + uprdn if branch overlap?
    uplink_layers = lambda: [[],[]]  # init a layer of dderPs and a layer of match_dderPs
    downlink_layers = lambda: [[],[]]
   # from comp_dx
    fdx = NoneType

class CPP(CderP):  # derP params include P.ptuple

    oPP_ = list  # adjacent opposite-sign PPs from above, below, left and right
    players = list  # 1st plevel, same as in derP but L is area
    valt = lambda: [0,0]  # mval, dval summed across players
    fds = list  # fPd per player except 1st, to comp in agg_recursion
    x0 = int  # box, update by max, min; this is a 2nd plevel?
    xn = int
    y0 = int
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
    mask__ = bool
    P__ = list  # input + derPs, common root for downward layers and upward levels:
    rlayers = list  # or mlayers: sub_PPs from sub_recursion within PP
    dlayers = list  # or alayers
    seg_levels = list  # from 1st agg_recursion[fPd], seg_levels[0] is seg_, higher seg_levels are segP_..s
    root = object  # higher-order segP | PPP
    cPP_ = list  # rdn reps in other PPPs, to eval and remove

# Functions:

def comp_slice_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    segment_by_direction(blob, verbose=False)  # forms blob.dir_blobs
    for dir_blob in blob.dir_blobs:  # same-direction blob should be CBlob

        P__ = slice_blob(dir_blob, verbose=False)  # cluster dir_blob.dert__ into 2D array of blob slices
        comp_P_root(P__)  # scan_P_, comp_P | link_layer, adds mixed uplink_, downlink_ per P; comp_dx_blob(P__), comp_dx?

        # form segments: parameterized stacks of (P,derP)s:
        segm_ = form_seg_root([[copy_P(P) for P in P_] for P_ in P__], fPd=0, fds=[0])
        segd_ = form_seg_root([[copy_P(P) for P in P_] for P_ in P__], fPd=1, fds=[0])

        # form PPs: parameterized graphs of connected segs:
        PP_ = form_PP_root((segm_, segd_), base_rdn=2)  # update to mixed-fork PP_, select by PP.fPd:
        dir_blob.rlayers, dir_blob.dlayers = sub_recursion_eval(PP_)  # add rlayers, dlayers, seg_levels to select PPs

        comb_levels, levels = [],[]  # dir_blob agg_levels
        M = dir_blob.M; G = dir_blob.G  # weak-fork rdn, or combined value?
        # intra-PP:
        if ((M - ave_mPP * (1+(G>M)) + (G - ave_dPP * (1+M>=G)) - ave_agg * (dir_blob.rdn+1) > 0) and len(PP_) > ave_nsub):
            dir_blob.valt = [M,G]
            from agg_recursion import agg_recursion, CgPP
            # convert PPs to CgPPs:
            for i, PP in enumerate(PP_):
                players_t = [[], []]
                fd = PP.fds[-1]
                players_t[fd] = PP.players
                players_t[1-fd] = PP.oPP_[0].players
                for oPP in PP.oPP_[1:]: sum_players(players_t[1-fd], oPP.players)  # sum all oPPs
                PP_[i] = CgPP(PP=PP, players_t=players_t, fds=deepcopy(PP.fds), x0=PP.x0, xn=PP.xn, y0=PP.y0, yn=PP.yn)
            # cluster PPs into graphs:
            levels = agg_recursion(dir_blob, PP_, rng=2, fseg=0)

        for i, (comb_level, level) in enumerate(zip_longest(comb_levels, levels, fillvalue=[])):
            if level:
                if i > len(comb_levels)-1: comb_levels += [level]  # add new level
                else: comb_levels[i] += [level]  # append existing level
        dir_blob.agg_levels = [[PP_]] + comb_levels  # 1st + higher agg_levels

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

def comp_P(_P, P):  # forms vertical derivatives of params per P in _P.uplink, conditional ders from norm and DIV comp

    if isinstance(_P, CP):
        mtuple, dtuple = comp_ptuple(_P.ptuple, P.ptuple)
        valt = [mtuple.val, dtuple.val]
        mplayer = [mtuple]; dplayer = [dtuple]
        players = [[_P.ptuple]]
    else:  # P is derP
        mplayer, dplayer = comp_players(_P.players, P.players)  # passed from seg.fds
        valt = [sum([mtuple.val for mtuple in mplayer]), sum([dtuple.val for dtuple in dplayer])]
        players = deepcopy(_P.players)

    return CderP(x0=min(_P.x0, P.x0), y=_P.y, players=deepcopy(players)+[[mplayer,dplayer]], valt=valt, P=P, _P=_P)

# pending link copy update, copying in the end won't help
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

    Pm__= [[copy_P(P) for P in P_] for P_ in P__ ]
    Pd__= [[copy_P(P) for P in P_] for P_ in P__ ]

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
                    dderP = comp_P(_derP, derP)  # form higher vertical derivatives of derP or PP params
                    derP.uplink_layers[0] += [dderP]  # pre-init layer per derP
                    _derP.downlink_layers[0] += [dderP]
                    dderPs += [dderP]
                # compute x overlap between dderP'__P and P, in form_seg_ or comp_layer?
            dderPs_ += dderPs  # row of dderPs
        dderPs__ += [dderPs_]

    dderPm__ = [[copy_P(dderP) for dderP in dderP_] for dderP_ in dderPs__ ]
    dderPd__ = [[copy_P(dderP) for dderP in dderP_] for dderP_ in dderPs__ ]

    return dderPm__, dderPd__


def form_seg_root(P__, fPd, fds):  # form segs from Ps

    for P_ in P__[1:]:  # scan bottom-up, append link_layers[-1] with branch-rdn adjusted matches in link_layers[-2]:
        for P in P_: link_eval(P.uplink_layers, fPd)  # uplinks_layers[-2] matches -> uplinks_layers[-1]

    for P_ in P__[:-1]:  # form downlink_layers[-1], different branch rdn, for termination eval in form_seg_?
        for P in P_: link_eval(P.downlink_layers, fPd)  # downinks_layers[-2] matches -> downlinks_layers[-1]

    seg_ = []
    for P_ in reversed(P__):  # get a row of Ps bottom-up, different copies per fPd
        while P_:
            P = P_.pop(0)
            if P.uplink_layers[-1]:  # last matching derPs layer is not empty
                form_seg_(seg_, P__, [P], fPd, fds)  # test P.matching_uplink_, not known in form_seg_root
            else:
                seg_.append( sum2seg([P], fPd, fds))  # no link_s, terminate seg_Ps = [P]

    return seg_

def form_seg_(seg_, P__, seg_Ps, fPd, fds):  # form contiguous segments of vertically matching Ps

    if len(seg_Ps[-1].uplink_layers[-1]) > 1:  # terminate seg
        # we can't do this here, PP is not formed yet?
        if _PP and _PP not in PP.oPP_:  # vertically adjacent opposite-sign PPs, may be multiple above and below?
            PP.oPP_ += [_PP]; _PP.oPP_ += [PP]

        seg_.append( sum2seg( seg_Ps, fPd, fds))  # convert seg_Ps to CPP seg

    else:
        uplink_ = seg_Ps[-1].uplink_layers[-1]
        if uplink_ and len(uplink_[0]._P.downlink_layers[-1])==1:
            # one P.uplink AND one _P.downlink: add _P to seg, uplink_[0] is sole upderP:
            P = uplink_[0]._P
            [P_.remove(P) for P_ in P__ if P in P_]  # remove P from P__ so it's not inputted in form_seg_root
            seg_Ps += [P]  # if P.downlinks in seg_down_misses += [P]

            if seg_Ps[-1].uplink_layers[-1]:
                form_seg_(seg_, P__, seg_Ps, fPd, fds)  # recursive compare sign of next-layer uplinks
            else:
                seg_.append( sum2seg(seg_Ps, fPd, fds))
        else:
            seg_.append( sum2seg(seg_Ps, fPd, fds))  # terminate seg at 0 matching uplink

# not sure:
def link_eval(link_layers, fPd):

    # sort derPs in link_layers[-2] by their value param:
    derP_ = sorted( link_layers[-2], key=lambda derP: derP.valt[fPd], reverse=True)

    for i, derP in enumerate(derP_):
        if not fPd:
            rng_eval(derP, fPd)  # reset derP.valt, derP.rdn
        mrdn = derP.valt[1] > derP.valt[0]
        derP.rdn += not mrdn if fPd else mrdn

        if derP.valt[fPd] > vaves[fPd] * derP.rdn * (i+1):  # ave * rdn to stronger derPs in link_layers[-2]
            link_layers[-1].append(derP)  # misses = link_layers[-2] not in link_layers[-1]

# not sure:
def rng_eval(derP, fPd):  # compute value of combined mutual derPs: overlap between P uplinks and _P downlinks

    _P, P = derP._P, derP.P
    common_derP_ = []

    for _downlink_layer, uplink_layer in zip(_P.downlink_layers, P.uplink_layers):  # overlap in P uplinks and _P downlinks
        common_derP_ += list( set(_downlink_layer).intersection(uplink_layer))  # get common derP in mixed uplinks
    rdn = 1
    olp_val = 0
    nolp = len(common_derP_)
    for derP in common_derP_:
        rdn += derP.valt[fPd] > derP.valt[1-fPd]
        olp_val += derP.valt[fPd]  # olp_val not reset for every derP?
        derP.valt[fPd] = olp_val / nolp
    '''
    for i, derP in enumerate( sorted( link_layers[-2], key=lambda derP: derP.params[fPd].val, reverse=True)):
    if fPd: derP.rdn += derP.params[fPd].val > derP.params[1-fPd].val  # mP > dP
    else: rng_eval(derP, fPd)  # reset derP.val, derP.rdn
    if derP.params[fPd].val > vaves[fPd] * derP.rdn * (i+1):  # ave * rdn to stronger derPs in link_layers[-2]
    '''
    derP.rdn += (rdn / nolp) > .5  # no fractional rdn?


def form_PP_root(seg_t, base_rdn):  # form PPs from match-connected segs

    PP_ = []  # PP fork = PP.fds[-1]
    for fPd in 0, 1:
        _PP = None
        seg_ = seg_t[fPd]
        for seg in seg_:  # bottom-up
            if not isinstance(seg.root, CPP):  # seg is not already in PP initiated by some prior seg
                PP_segs = [seg]
                # add links in PP_segs:
                if seg.P__[-1].uplink_layers[-1]:
                    form_PP_(PP_segs, seg.P__[-1].uplink_layers[-1].copy(), fup=1)
                if seg.P__[0].downlink_layers[-1]:
                    form_PP_(PP_segs, seg.P__[0].downlink_layers[-1].copy(), fup=0)
                # convert PP_segs to PP:
                PP = sum2PP(PP_segs, base_rdn)
                # still need this?:
                if _PP and _PP not in PP.oPP_:  # vertically adjacent opposite-sign PPs, may be multiple above and below?
                    PP.oPP_ += [_PP]; _PP.oPP_ += [PP]
                PP_ += [PP]
                _PP = PP
    return PP_


def form_PP_(PP_segs, link_, fup):  # flood-fill PP_segs with vertically linked segments:
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
                form_PP_(PP_segs, uplink_, fup=1)
            downlink_ = seg.P__[0].downlink_layers[-1]  # bottom-P downlink_
            if downlink_:
                form_PP_(PP_segs, downlink_, fup=0)


def sum2seg(seg_Ps, fPd, fds):  # sum params of vertically connected Ps into segment

    uplinks, uuplinks  = seg_Ps[-1].uplink_layers[-2:]  # uplinks of top P
    miss_uplink_ = [uuplink for uuplink in uuplinks if uuplink not in uplinks]  # in layer-1 but not in layer-2

    downlinks, ddownlinks = seg_Ps[0].downlink_layers[-2:]  # downlinks of bottom P, downlink.P.seg.uplinks= lower seg.uplinks
    miss_downlink_ = [ddownlink for ddownlink in ddownlinks if ddownlink not in downlinks]
    # seg rdn: up cost to init, up+down cost for comp_seg eval, in 1st agg_recursion?
    # P rdn is up+down M/n, but P is already formed and compared?
    seg = CPP(x0=seg_Ps[0].x0, P__= seg_Ps, uplink_layers=[miss_uplink_], downlink_layers = [miss_downlink_], y0 = seg_Ps[0].y)

    for P in seg_Ps[:-1]:
        accum_derP(seg, P.uplink_layers[-1][0])  # derP = P.uplink_layers[-1][0]

    accum_derP(seg, seg_Ps[-1])  # accum last P only, top P uplink_layers are not part of seg
    seg.y0 = seg_Ps[0].y
    seg.yn = seg.y0 + len(seg_Ps)
    seg.fds = fds + [fPd]  # fds of root PP

    return seg

def accum_derP(seg, derP):  # derP might be CP, though unlikely

    seg.x0 = min(seg.x0, derP.x0)

    if isinstance(derP, CP):
        derP.root = seg  # no need for derP.root
        if seg.players: sum_players(seg.players, [[derP.ptuple]])
        else:           seg.players.append([deepcopy(derP.ptuple)])
        seg.xn = max(seg.xn, derP.x0 + derP.ptuple.L)
    else:
        sum_players(seg.players, derP.players)  # last derP player is current mplayer, dplayer
        for i, val in enumerate(derP.valt): seg.valt[i]+=val
        seg.xn = max(seg.xn, derP.x0 + derP.players[0][0].L)


def sum2PP(PP_segs, base_rdn):  # sum PP_segs into PP

    PP = CPP(x0=PP_segs[0].x0, rdn=base_rdn)  # L = yn-y0, redundant
    PP.seg_levels = [PP_segs]  # PP_segs is levels[0]

    for seg in PP_segs:
        accum_PP(PP, seg)
    PP.fds = copy(seg.fds)

    return PP

def accum_PP(PP, inp):  # comp_slice inp is seg, or segPP in agg+

    for i in range(2): PP.valt[i] += inp.valt[i]
    sum_players(PP.players, inp.players)  # not empty inp's players
    inp.root = PP
    PP.x0 = min(PP.x0, inp.x0)  # external params: 2nd player?
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
                if derP not in PP.downlink_layers[-1] and derP.P.root not in PP.seg_levels[-1]:
                    PP.downlink_layers[-1] += [derP]
            for derP in inp.P__[-1].uplink_layers[-1]:  # if downlink not in current PP's downlink and not part of the seg in current PP:
                if derP not in PP.downlink_layers[-1] and derP.P.root not in PP.seg_levels[-1]:
                    PP.uplink_layers[-1] += [derP]

    for P_ in PP.P__[:-1]:  # add derP root, except for top row and bottom row derP
        for P in P_:
            for derP in P.uplink_layers[-1]:
                derP.root = PP


def sum_players(Layers, layers, fneg=0):  # no accum across fPd, that's checked in comp_players?

    if not Layers:
        if not fneg: Layers.append(deepcopy(layers[0]))
    else: accum_ptuple(Layers[0][0], layers[0][0], fneg)  # latuples, in purely formal nesting

    for Layer, layer in zip_longest(Layers[1:], layers[1:], fillvalue=[]):
        if layer:
            if Layer:
                for fork_Layer, fork_layer in zip(Layer, layer):
                    sum_player(fork_Layer, fork_layer, fneg=fneg)
            elif not fneg: Layers.append(deepcopy(layer))

def sum_player(Player, player, fneg=0):  # accum players in Players

    for i, (Ptuple, ptuple) in enumerate(zip_longest(Player, player, fillvalue=[])):
        if ptuple:
            if Ptuple: accum_ptuple(Ptuple, ptuple, fneg)
            elif Ptuple == None: Player[i] = ptuple  # not sure
            elif not fneg: Player.append(deepcopy(ptuple))

def accum_ptuple(Ptuple, ptuple, fneg=0):  # lataple or vertuple

    for param_name in Ptuple.numeric_params:
        if param_name != "G" and param_name != "Ga":
            Param = getattr(Ptuple, param_name)
            param = getattr(ptuple, param_name)
            if fneg: out = Param-param
            else:    out = Param+param
            setattr(Ptuple, param_name, out)  # update value

    if isinstance(Ptuple.angle, list):
        for i, angle in enumerate(ptuple.angle):
            if fneg: Ptuple.angle[i] -= angle
            else:    Ptuple.angle[i] += angle
        for i, aangle in enumerate(ptuple.aangle):
            if fneg: Ptuple.aangle[i] -= aangle
            else:    Ptuple.aangle[i] += aangle
    else:
        if fneg: Ptuple.angle -= ptuple.angle; Ptuple.aangle -= ptuple.aangle
        else:    Ptuple.angle += ptuple.angle; Ptuple.aangle += ptuple.aangle

# need update on fds later
def comp_players(_layers, layers, _fds=[0], fds=[0]):  # unpack and compare der layers, if any from der+

    mtuple, dtuple = comp_ptuple(_layers[0][0], layers[0][0])  # initial latuples, always present and nested
    mplayer=[mtuple]; dplayer=[dtuple]

    for i, (_layer, layer) in enumerate(zip(_layers[1:], layers[1:])):
        # compare fPd ptuples on all layers:
        for _ptuple, ptuple in zip(_layer[_fds[i]], layer[fds[i]]):
            mtuple, dtuple = comp_ptuple(_ptuple, ptuple)
            mplayer+=[mtuple]; dplayer+=[dtuple]

    return mplayer, dplayer

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

    return mtuple, dtuple


def comp(param_name, _param, param, dval, mval, dtuple, mtuple, ave, finv):

    d = _param-param
    if finv: m = ave - abs(d)  # inverse match for primary params, no mag/value correlation
    else:    m = min(_param,param) - ave
    dval += abs(d)
    mval += m
    setattr(dtuple, param_name, d)  # dtuple.param_name = d
    setattr(mtuple, param_name, m)  # mtuple.param_name = m


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


def copy_P(P, iPtype=None):   # Ptype =0: P is CP | =1: P is CderP | =2: P is CPP | =3: P is CderPP | =4: P is CaggPP

    if not iPtype:  # assign Ptype based on instance type if no input type is provided
        if isinstance(P, CPP):     Ptype = 2
        elif isinstance(P, CderP): Ptype = 1
        elif isinstance(P, CP):    Ptype = 0
    else: Ptype = iPtype

    uplink_layers, downlink_layers = P.uplink_layers, P.downlink_layers  # local copy of link layers
    P.uplink_layers, P.downlink_layers = [], []  # reset link layers
    seg = P.root  # local copy
    P.root = None
    if Ptype == 1:
        P_derP, _P_derP = P.P, P._P  # local copy of derP.P and derP._P
        P.P, P._P = None, None  # reset
    elif Ptype == 2:
        seg_levels = P.seg_levels
        rlayers = P.rlayers
        dlayers = P.dlayers
        P__ = P.P__
        oPP_ = P.oPP_
        P.seg_levels, P.rlayers, P.dlayers, P.P__, P.oPP_ = [], [], [], [], []  # reset
    elif Ptype == 3:
        PP_derP, _PP_derP = P.PP, P._PP  # local copy of derP.P and derP._P
        P.PP, P._PP = None, None  # reset
    elif Ptype == 4:
        PP_, cPP_ = P.PP_, P.cPP_
        rlayers, dlayers = P.rlayers, P.dlayers
        levels, root = P.levels, P.root
        P.PP_, P.cPP_, P.rlayers, P.dlayers, P.levels, P.root = [], [], [], [], [], None  # reset

    new_P = P.copy()  # copy P with empty root and link layers, reassign link layers:
    new_P.uplink_layers += copy(uplink_layers)
    new_P.downlink_layers += copy(downlink_layers)

    # shallow copy to create new list
    P.uplink_layers, P.downlink_layers = copy(uplink_layers), copy(downlink_layers)  # reassign link layers
    P.root = seg  # reassign root
    # reassign other list params
    if Ptype == 1:
        new_P.P, new_P._P = P_derP, _P_derP
        P.P, P._P = P_derP, _P_derP
    elif Ptype == 2:
        P.seg_levels = seg_levels
        P.rlayers = rlayers
        P.dlayers = dlayers
        P.oPP_ = oPP_
        new_P.rlayers = copy(rlayers)
        new_P.dlayers = copy(dlayers)
        new_P.P__ = copy(P__)
        new_P.seg_levels = copy(seg_levels)
        new_P.oPP_ = copy(oPP_)
    elif Ptype == 3:
        new_P.PP, new_P._PP = PP_derP, _PP_derP
        P.PP, P._PP = PP_derP, _PP_derP
    elif Ptype == 4:
        P.PP_ = PP_
        P.cPP_ = cPP_
        P.rlayers = rlayers
        P.dlayers = dlayers
        P.levels = levels
        P.root = root
        new_P.PP_ = copy(PP_)
        new_P.cPP_ = copy(cPP_)
        new_P.rlayers = copy(rlayers)
        new_P.dlayers = copy(dlayers)
        new_P.levels = copy(levels)
        new_P.root = root

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
                        # test y adjacency:
                        if (_top_P_[0].y-1 == bottom_P_[0].y) or (top_P_[0].y-1 == _bottom_P_[0].y):
                            # test x overlap:
                             if (dir_blob.x0 - 1 < _dir_blob.xn and dir_blob.xn + 1 > _dir_blob.x0) \
                                or (_dir_blob.x0 - 1 < dir_blob.xn and _dir_blob.xn + 1 > dir_blob.x0):
                                 splice_2dir_blobs(_dir_blob, dir_blob)  # splice dir_blob into _dir_blob
                                 dir_blobs[j] = _dir_blob

def splice_2dir_blobs(_blob, blob):
    # merge blob into _blob here
    pass


def sub_recursion_eval(PP_):  # for PP or dir_blob

    from agg_recursion import agg_recursion
    mcomb_layers, dcomb_layers, PPm_, PPd_ = [], [], [], []
    for PP in PP_:

        fPd = PP.fds[-1]
        if fPd: comb_layers = dcomb_layers; PP_layers = PP.dlayers; PPd_ += [PP]
        else:   comb_layers = mcomb_layers; PP_layers = PP.rlayers; PPm_ += [PP]
        val = PP.valt[fPd]; alt_val = PP.valt[not fPd]  # for fork rdn:
        ave = PP_aves[fPd] * (PP.rdn + 1 + (alt_val > val))
        if val > ave and len(PP.P__) > ave_nsub:

            sub_recursion(PP)  # comp_P_der | comp_P_rng in PPs -> param_layer, sub_PPs
            ave*=2  # 1+PP.rdn incr
            # splice deeper layers between PPs into comb_layers:
            for i, (comb_layer, PP_layer) in enumerate(zip_longest(comb_layers, PP_layers, fillvalue=[])):
                if PP_layer:
                    if i > len(comb_layers) - 1: comb_layers += [PP_layer]  # add new r|d layer
                    else: comb_layers[i] += PP_layer  # splice r|d PP layer into existing layer

        # segs rng_comp agg_recursion, centroid reclustering:
        if val > ave*3 and len(PP.seg_levels[-1]) > ave_nsub:   # 3: agg_coef
            PP.seg_levels += agg_recursion(PP, PP.seg_levels[-1], rng=2, fseg=1)

    return [[PPm_] + mcomb_layers], [[PPd_] + dcomb_layers]  # including empty comb_layers


def sub_recursion(PP):  # evaluate each PP for rng+ and der+

    P__  = [P_ for P_ in reversed(PP.P__)]  # revert to top down
    Pm__, Pd__ = comp_P_der(P__) if PP.fds[-1] else comp_P_rng(P__, PP.rng + 1)   # returns top-down

    PP.rdn += 2  # two-fork rdn, priority is not known?
    sub_segm_ = form_seg_root(Pm__, fPd=0, fds=PP.fds)
    sub_segd_ = form_seg_root(Pd__, fPd=1, fds=PP.fds)  # returns bottom-up
    sub_PP_ = form_PP_root((sub_segm_, sub_segd_), PP.rdn + 1)  # PP is parameterized graph of linked segs

    sub_rlayers, sub_dlayers = sub_recursion_eval(sub_PP_)  # add rlayers, dlayers, seg_levels to select sub_PPs
    PP.rlayers += [sub_rlayers]; PP.dlayers += [sub_dlayers]