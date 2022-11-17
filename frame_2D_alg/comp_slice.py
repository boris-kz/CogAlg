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

import sys
import numpy as np
from itertools import zip_longest
from copy import deepcopy, copy
from class_cluster import ClusterStructure, NoneType, comp_param, Cdert

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
ave_dy = 5
ave_dangle = 2  # vertical difference between angles
ave_daangle = 2
ave_mval = ave_dval = 10  # should be different
ave_mPP = 10
ave_dPP = 10
ave_splice = 10
ave_nsub = 1
ave_sub = 2  # cost of calling sub_recursion and looping
ave_agg = 3  # cost of agg_recursion
ave_overlap = 10
ave_rotate = 10

param_names = ["x", "I", "M", "Ma", "L", "angle", "aangle"]
aves = [ave_dx, ave_dI, ave_M, ave_Ma, ave_L, ave_G, ave_Ga, ave_mval, ave_dval]
vaves = [ave_mval, ave_dval]
PP_aves = [ave_mPP, ave_dPP]


class Cptuple(ClusterStructure):  # bottom-layer tuple of lateral or vertical params: lataple in P or vertuple in derP

    # compared params, add prefix v in vertuples: 9 vs. 10 params:
    I = int
    M = int
    Ma = float
    angle = lambda: [0, 0]  # in lataple only, replaced by float in vertuple
    aangle = lambda: [0, 0, 0, 0]
    # only in lataple, for comparison but not summation:
    G = float
    Ga = float
    # only in vertuple, combined tuple m|d value:
    val = float
    # in P, m|d if vertuple, layered?:
    n = lambda: 1  # accum count
    x = int  # median vs. x0?
    daxis = lambda: None  # final dangle in rotate_P
    L = int

class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

    ptuple = object  # I, M, Ma, G, Ga, angle(Dy, Dx), aangle( Sin_da0, Cos_da0, Sin_da1, Cos_da1), n, val
    daxis = lambda: None  # final dangle in rotate_P, move to ptuple?
    x0 = int
    y0 = int  # for vertical gap in PP.P__
    rdn = int  # blob-level redundancy, ignore for now
    dert_ = list  # array of pixel-level derts, redundant to uplink_, only per blob?
    uplink_layers = lambda: [ [], [[],[]] ]  # init a layer of derPs and a layer of match_derPs
    downlink_layers = lambda: [ [], [[],[]] ]
    roott = lambda: [None,None]  # m,d seg|PP that contain this P
    # only in Pd:
    dxdert_ = list
    # only in Pm:
    Pd_ = list
    # if comp_dx:
    Mdx = int
    Ddx = int

class CderP(ClusterStructure):  # tuple of derivatives in P uplink_ or downlink_: binary tree with latuple root and vertuple forks
    # players in derP are zipped with fds: taken forks. Players, mplayer, dplayer are replaced by those in PP

    players = list  # max ntuples in layer = ntuples in lower layers: 1, 1, 2, 4, 8..: one selected fork per compared ptuple
    # nesting = len players ( player ( sub_player.., explicit to align comparands?
    lplayer = lambda: [[], []]  # last mplayer, dplayer; der+ comp: players+dplayer or dplayer: link only?
    valt = lambda: [0,0]  # tuple of mval, dval, summed from last player, both are signed
    x0 = int
    y0 = int  # or while isinstance(P, CderP): P = CderP._P; else y = _P.y0
    _P = object  # higher comparand
    P = object  # lower comparand
    roott = lambda: [None,None]  # for der++
    # higher derivatives
    rdn = int  # mrdn, + uprdn if branch overlap?
    uplink_layers = lambda: [[],[[], []]]  # init a layer of dderPs and a layer of match_dderPs
    downlink_layers = lambda: [[],[[], []]]
   # from comp_dx
    fdx = NoneType

class CPP(CderP):  # derP params include P.ptuple

    players = list  # 1st plevel, same as in derP but L is area
    valt = lambda: [0,0]  # mval, dval summed across players
    nvalt = lambda: [0,0]  # from neg derPs:
    nderP_ =list  # miss links, add with nvalt for complemented PP
    fds = list  # fPd per player except 1st, to comp in agg_recursion
    x0 = int  # box, update by max, min; this is a 2nd plevel?
    xn = int
    y0 = int
    yn = int
    rng = lambda: 1  # rng starts with 1
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs
    Rdn = int  # for accumulation only
    alt_rdn = int  # overlapping redundancy between core and edge
    P_cnt = int  # len 2D derP__ in levels[0][fPd]?  ly = len(derP__), also x, y?
    derP_cnt = int  # redundant per P
    uplink_layers = lambda: [[]]  # the links here will be derPPs from discontinuous comp, not in layers?
    downlink_layers = lambda: [[]]
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    mask__ = bool
    P__ = list  # input + derPs, common root for downward layers and upward levels:
    rlayers = list  # or mlayers: sub_PPs from sub_recursion within PP
    dlayers = list  # or alayers
    mseg_levels = list  # from 1st agg_recursion[fPd], seg_levels[0] is seg_, higher seg_levels are segP_..s
    dseg_levels = list
    roott = lambda: [None,None]  # PPPm, PPPd that contain this PP
    altPP_ = list  # adjacent alt-fork PPs per PP, from P.roott[1] in sum2PP
    cPP_ = list  # rdn reps in other PPPs, to eval and remove

# Functions:

def comp_slice_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    from sub_recursion import sub_recursion_eval, rotate_P_
    P__ = slice_blob(blob, verbose=False)  # form 2D array of blob slices in blob.dert__
    rotate_P_(P__, blob.dert__, blob.mask__)  # rotate each P to align it with direction of P gradient

    comp_P_root(P__)  # rotated Ps are sparse or overlapping: derPs are partly redundant, but results are not biased?
    # segment is stack of (P,derP)s:
    segm_ = form_seg_root([copy(P_) for P_ in P__], fd=0, fds=[0])  # shallow copy: same Ps in different lists
    segd_ = form_seg_root([copy(P_) for P_ in P__], fd=1, fds=[0])  # initial latuple fd=0
    # PP is graph of segs:
    blob.PPm_, blob.PPd_ = form_PP_root((segm_, segd_), base_rdn=2)
    # micro and macro re-clustering:
    sub_recursion_eval(blob)  # intra PP, add rlayers, dlayers, seg_levels to select PPs, sum M,G
    agg_recursion_eval(blob, [copy(blob.PPm_), copy(blob.PPd_)])  # cross PP, Cgraph conversion doesn't replace PPs?


def slice_blob(blob, verbose=False):  # form blob slices nearest to slice Ga: Ps, ~1D Ps, in select smooth edge (high G, low Ga) blobs

    mask__ = blob.mask__  # same as positive sign here
    dert__ = zip(*blob.dert__)  # convert 10-tuple of 2D arrays into 1D array of 10-tuple blob rows
    dert__ = [zip(*dert_) for dert_ in dert__]  # convert 1D array of 10-tuple rows into 2D array of 10-tuples per blob
    P__ = []
    height, width = mask__.shape
    if verbose: print("Converting to image...")

    for y, (dert_, mask_) in enumerate(zip(dert__, mask__)):  # unpack lines, each may have multiple slices -> Ps:
        P_ = []
        _mask = True
        for x, (dert, mask) in enumerate(zip(dert_, mask_)):  # dert = i, g, ga, ri, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1

            if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()
            g, ga, ri, angle, aangle = dert[1], dert[2], dert[3], list(dert[4:6]), list(dert[6:])
            if not mask:  # masks: if 0,_1: P initialization, if 0,_0: P accumulation, if 1,_0: P termination
                if _mask:  # ini P params with first unmasked dert (m, ma, i, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1):
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
                P_.append( CP(ptuple=params, x0=x-(params.L-1), y0=y, dert_=Pdert_))
            _mask = mask

        if not _mask:  # pack last P, same as above:
            params.L = len(Pdert_)
            params.G = np.hypot(*params.angle)
            params.Ga = (params.aangle[1] + 1) + (params.aangle[3] + 1)
            P_.append(CP(ptuple=params, x0=x - (params.L - 1), y0=y, dert_=Pdert_))
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
                    break  # no xn overlap, stop scanning lower P_
        _P_ = P_

    return P__

def comp_P(_P, P):  # forms vertical derivatives of params per P in _P.uplink, conditional ders from norm and DIV comp

    Daxis = _P.daxis - P.daxis
    dx = _P.x0-len(_P.dert_)/2 - P.x0-len(P.dert_)/2
    Daxis *= np.hypot(dx, 1)  # project param orthogonal to blob axis, dy=1
    '''
    form and compare extuple: m|d x, axis, L, -> explayers || players, vs macro as in graph?
    or ptuple extension += [med_x. axis, L]?
        
    comp("x", _params.x, params.x*rn, dval, mval, dtuple, mtuple, ave_dx, finv=flatuple)
    comp("L", _params.L, params.L*rn / daxis, dval, mval, dtuple, mtuple, ave_L, finv=0)
    '''
    if isinstance(_P, CP):
        mtuple, dtuple = comp_ptuple(_P.ptuple, P.ptuple, Daxis)
        valt = [mtuple.val, dtuple.val]
        mplayer = [mtuple]; dplayer = [dtuple]
        players = [[_P.ptuple]]

    else:  # P is derP
        mplayer, dplayer, mval, dval = comp_players(_P.players, P.players)  # passed from seg.fds
        valt = [mval, dval]
        players = deepcopy(_P.players)

    return CderP(x0=min(_P.x0, P.x0), y0=_P.y0, players=players, lplayer=[mplayer,dplayer], valt=valt, P=P, _P=_P)


def form_seg_root(P__, fd, fds):  # form segs from Ps

    for P_ in P__[1:]:  # scan bottom-up, append link_layers[-1] with branch-rdn adjusted matches in link_layers[-2]:
        for P in P_: link_eval(P.uplink_layers, fd)  # uplinks_layers[-2] matches -> uplinks_layers[-1]
                   # forms both uplink and downlink layers[-1]
    seg_ = []
    for P_ in reversed(P__):  # get a row of Ps bottom-up, different copies per fPd
        while P_:
            P = P_.pop(0)
            if P.uplink_layers[-1][fd]:  # last matching derPs layer is not empty
                form_seg_(seg_, P__, [P], fd, fds)  # test P.matching_uplink_, not known in form_seg_root
            else:
                seg_.append( sum2seg([P], fd, fds))  # no link_s, terminate seg_Ps = [P]
    return seg_


def link_eval(link_layers, fd):

    # sort derPs in link_layers[-2] by their value param:
    derP_ = sorted( link_layers[-2], key=lambda derP: derP.valt[fd], reverse=True)

    for i, derP in enumerate(derP_):
        if not fd:
            rng_eval(derP, fd)  # reset derP.valt, derP.rdn
        mrdn = derP.valt[1] > derP.valt[0]
        derP.rdn += not mrdn if fd else mrdn

        if derP.valt[fd] > vaves[fd] * derP.rdn * (i+1):  # ave * rdn to stronger derPs in link_layers[-2]
            link_layers[-1][fd].append(derP)
            derP._P.downlink_layers[-1][fd] += [derP]
            # misses = link_layers[-2] not in link_layers[-1], sum as PP.nvalt[fd] in sum2seg and sum2PP

# not sure:
def rng_eval(derP, fd):  # compute value of combined mutual derPs: overlap between P uplinks and _P downlinks

    _P, P = derP._P, derP.P
    common_derP_ = []

    for _downlink_layer, uplink_layer in zip(_P.downlink_layers[1::2], P.uplink_layers[1::2]):
        # overlap between +ve P uplinks and +ve _P downlinks:
        common_derP_ += list( set(_downlink_layer[fd]).intersection(uplink_layer[fd]))
    rdn = 1
    olp_val = 0
    nolp = len(common_derP_)
    for derP in common_derP_:
        rdn += derP.valt[fd] > derP.valt[1-fd]
        olp_val += derP.valt[fd]  # olp_val not reset for every derP?
        derP.valt[fd] = olp_val / nolp
    '''
    for i, derP in enumerate( sorted( link_layers[-2], key=lambda derP: derP.params[fPd].val, reverse=True)):
    if fPd: derP.rdn += derP.params[fPd].val > derP.params[1-fPd].val  # mP > dP
    else: rng_eval(derP, fPd)  # reset derP.val, derP.rdn
    if derP.params[fPd].val > vaves[fPd] * derP.rdn * (i+1):  # ave * rdn to stronger derPs in link_layers[-2]
    '''
#    derP.rdn += (rdn / nolp) > .5  # no fractional rdn?

def form_seg_(seg_, P__, seg_Ps, fd, fds):  # form contiguous segments of vertically matching Ps

    if len(seg_Ps[-1].uplink_layers[-1][fd]) > 1:  # terminate seg
        seg_.append( sum2seg( seg_Ps, fd, fds))
        # convert seg_Ps to CPP seg
    else:
        uplink_ = seg_Ps[-1].uplink_layers[-1][fd]
        if uplink_ and len(uplink_[0]._P.downlink_layers[-1][fd])==1:
            # one P.uplink AND one _P.downlink: add _P to seg, uplink_[0] is sole upderP:
            P = uplink_[0]._P
            [P_.remove(P) for P_ in P__ if P in P_]  # remove P from P__ so it's not inputted in form_seg_root
            seg_Ps += [P]  # if P.downlinks in seg_down_misses += [P]

            if seg_Ps[-1].uplink_layers[-1][fd]:
                form_seg_(seg_, P__, seg_Ps, fd, fds)  # recursive compare sign of next-layer uplinks
            else:
                seg_.append( sum2seg(seg_Ps, fd, fds))
        else:
            seg_.append( sum2seg(seg_Ps, fd, fds))  # terminate seg at 0 matching uplink


def form_PP_root(seg_t, base_rdn):  # form PPs from match-connected segs

    PP_t = []  # PP fork = PP.fds[-1]
    for fd in 0, 1:
        PP_ = []
        seg_ = seg_t[fd]
        for seg in seg_:  # bottom-up
            if not isinstance(seg.roott[fd], CPP):  # seg is not already in PP initiated by some prior seg
                PP_segs = [seg]
                # add links in PP_segs:
                if seg.P__[-1].uplink_layers[-1][fd]:
                    form_PP_(PP_segs, seg.P__[-1].uplink_layers[-1][fd].copy(), fup=1, fd=fd)
                if seg.P__[0].downlink_layers[-1][fd]:
                    form_PP_(PP_segs, seg.P__[0].downlink_layers[-1][fd].copy(), fup=0, fd=fd)
                # convert PP_segs to PP:
                PP_ += [sum2PP(PP_segs, base_rdn, fd)]
        # P.roott = seg.root after term of all PPs, for form_PP_
        for PP in PP_:
            for P_ in PP.P__:
                for P in P_:
                    P.roott[fd] = PP  # update root from seg to PP
                    if fd:
                        PPm = P.roott[0]
                        if PPm not in PP.altPP_:
                            PP.altPP_ += [PPm]  # bilateral assignment of altPPs
                        if PP not in PPm.altPP_:
                            PPm.altPP_ += [PP]  # PPd here
        PP_t += [PP_]
    return PP_t


def form_PP_(PP_segs, link_, fup, fd):  # flood-fill PP_segs with vertically linked segments:
    '''
    PP is a graph with segs as 1D "vertices", each has two sets of edges / branching points: seg.uplink_ and seg.downlink_.
    '''
    for derP in link_:  # uplink_ or downlink_
        if fup: seg = derP._P.roott[fd]
        else:   seg = derP.P.roott[fd]

        if seg and seg not in PP_segs:  # top and bottom row Ps are not in segs
            PP_segs += [seg]
            uplink_ = seg.P__[-1].uplink_layers[-1][fd]  # top-P uplink_
            if uplink_:
                form_PP_(PP_segs, uplink_, fup=1, fd=fd)
            downlink_ = seg.P__[0].downlink_layers[-1][fd]  # bottom-P downlink_
            if downlink_:
                form_PP_(PP_segs, downlink_, fup=0, fd=fd)


def sum2seg(seg_Ps, fd, fds):  # sum params of vertically connected Ps into segment

    uplink_, uuplink_t  = seg_Ps[-1].uplink_layers[-2:]  # uplinks of top P
    miss_uplink_ = [uuplink for uuplink in uuplink_t[fd] if uuplink not in uplink_]  # in layer-1 but not in layer-2

    downlink_, ddownlink_t = seg_Ps[0].downlink_layers[-2:]  # downlinks of bottom P, downlink.P.seg.uplinks= lower seg.uplinks
    miss_downlink_ = [ddownlink for ddownlink in ddownlink_t[fd] if ddownlink not in downlink_]
    # seg rdn: up cost to init, up+down cost for comp_seg eval, in 1st agg_recursion?
    # P rdn is up+down M/n, but P is already formed and compared?
    seg = CPP(x0=seg_Ps[0].x0, P__= seg_Ps, uplink_layers=[miss_uplink_], downlink_layers = [miss_downlink_], y0 = seg_Ps[0].y0)
    lplayer = []
    for P in seg_Ps[:-1]:
        derP = P.uplink_layers[-1][fd][0]
        sum_ptuples(lplayer, derP.lplayer[fd])  # not comp derP.players, fd only
        accum_derP(seg, derP, fd)  # derP = P.uplink_layers[-1][0]
        P.roott[fd] = seg
        for derP in P.uplink_layers[-2]:
            if derP not in P.uplink_layers[-1][fd]:
                seg.nvalt[fd] += derP.valt[fd]  # -ve links in full links, not in +ve links
                seg.nderP_ += [derP]

    accum_derP(seg, seg_Ps[-1], fd)  # accum last P only, top P uplink_layers are not part of seg
    if lplayer:
        if fd: seg.players += [lplayer]  # der+
        else:  seg.players = [lplayer]  # rng+
        seg.players += [lplayer]  # add new player
    seg.y0 = seg_Ps[0].y0
    seg.yn = seg.y0 + len(seg_Ps)
    seg.fds = fds + [fd]  # fds of root PP

    return seg

def accum_derP(seg, derP, fd):  # derP might be CP, though unlikely

    seg.x0 = min(seg.x0, derP.x0)

    if isinstance(derP, CP):
        derP.roott[fd] = seg
        if seg.players: sum_players(seg.players, [[derP.ptuple]])
        # players:
        else:           seg.players.append([deepcopy(derP.ptuple)])
        seg.xn = max(seg.xn, derP.x0 + derP.ptuple.L)
    else:
        sum_players(seg.players, derP.players)  # last derP player is current mplayer, dplayer
        seg.valt[0] += derP.valt[0]; seg.valt[1] += derP.valt[1]
        seg.xn = max(seg.xn, derP.x0 + derP.players[0][0].L)


def sum2PP(PP_segs, base_rdn, fd):  # sum PP_segs into PP

    from sub_recursion import append_P

    PP = CPP(x0=PP_segs[0].x0, rdn=base_rdn, rlayers=[[]], dlayers=[[]])
    if fd: PP.dseg_levels, PP.mseg_levels = [PP_segs], [[]]  # empty alt_seg_levels
    else:  PP.mseg_levels, PP.dseg_levels = [PP_segs], [[]]

    for seg in PP_segs:
        seg.roott[fd] = PP
        # selection should be alt, not fd, only in convert?
        sum_players(PP.players, seg.players)  # not empty inp's players
        PP.fds = copy(seg.fds)
        PP.x0 = min(PP.x0, seg.x0)  # external params: 2nd player?
        PP.xn = max(PP.xn, seg.xn)
        PP.y0 = min(seg.y0, PP.y0)
        PP.yn = max(seg.yn, PP.yn)
        PP.Rdn += seg.rdn  # base_rdn + PP.Rdn / PP: recursion + forks + links: nderP / len(P__)?
        PP.derP_cnt += len(seg.P__[-1].uplink_layers[-1][fd])  # redundant derivatives of the same P
        # only PPs are sign-complemented, seg..[not fd]s are empty:
        PP.valt[fd] += seg.valt[fd]
        PP.nvalt[fd] += seg.nvalt[fd]
        PP.nderP_ += seg.nderP_
        # += miss seg.link_s:
        for i, (PP_layer, seg_layer) in enumerate(zip_longest(PP.uplink_layers, seg.uplink_layers, fillvalue=[])):
            if seg_layer:
               if i > len(PP.uplink_layers)-1: PP.uplink_layers.append(copy(seg_layer))
               else: PP_layer += seg_layer
        for i, (PP_layer, seg_layer) in enumerate(zip_longest(PP.downlink_layers, seg.downlink_layers, fillvalue=[])):
            if seg_layer:
               if i > len(PP.downlink_layers)-1: PP.downlink_layers.append(copy(seg_layer))
               else: PP_layer += seg_layer
        for P in seg.P__:
            if not PP.P__: PP.P__.append([P])  # pack 1st P
            else: append_P(PP.P__, P)  # pack P into PP.P__

        # add derPs from seg branches:
        for derP in seg.P__[-1].uplink_layers[-2]:
            for i, val in enumerate(derP.valt):
                if derP in seg.P__[-1].uplink_layers[-1][fd]:  # +ve links
                    PP.valt[i] += val
                else:  # -ve links
                    PP.nvalt[i] += val
                    PP.nderP_ += [derP]
    return PP


def sum_players(Layers, layers, fneg=0):  # no accum across fPd, that's checked in comp_players?

    for Layer, layer in zip_longest(Layers, layers, fillvalue=[]):
        if layer:
            if Layer: sum_ptuples(Layer, layer, fneg=fneg)
            elif not fneg: Layers.append(deepcopy(layer))

def sum_ptuples(Player, player, fneg=0):  # accum players in Players

    for i, (Ptuple, ptuple) in enumerate(zip_longest(Player, player, fillvalue=[])):
        if ptuple:
            if Ptuple: sum_ptuple(Ptuple, ptuple, fneg)  # accum ptuple
            elif Ptuple == None: Player[i] = deepcopy(ptuple)
            elif not fneg: Player.append(deepcopy(ptuple))

def sum_ptuple(Ptuple, ptuple, fneg=0):

    for param_name in Ptuple.numeric_params:
        if param_name != "G" and param_name != "Ga":
            Param = getattr(Ptuple, param_name)
            param = getattr(ptuple, param_name)
            if fneg: out = Param - param
            else:    out = Param + param
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


def comp_players(_layers, layers):  # unpack and compare der layers, if any from der+

    mptuples, dptuples = [],[]
    mval, dval = 0,0

    for _layer, layer in zip(_layers, layers):
        for _ptuple, ptuple in zip(_layer, layer):
            mtuple, dtuple = comp_ptuple(_ptuple, ptuple)
            mptuples +=[mtuple]; mval+=mtuple.val
            dptuples +=[dtuple]; dval+=dtuple.val

    return mptuples, dptuples, mval, dval


def comp_ptuple(_params, params, daxis):  # compare lateral or vertical tuples, similar operations for m and d params

    dtuple, mtuple = Cptuple(), Cptuple()
    dval, mval = 0, 0
    rn = _params.n / params.n  # normalize param as param*rn for n-invariant ratio: _param / param*rn = (_param/_n) / (param/n)

    flatuple = isinstance(_params.angle, list)  # else vertuple
    if flatuple:
        comp("G", _params.G, params.G*rn / daxis, dval, mval, dtuple, mtuple, ave_G, finv=0)
        comp("Ga", _params.Ga, params.Ga*rn / daxis, dval, mval, dtuple, mtuple, ave_Ga, finv=0)
        # angle:
        mangle, dangle = comp_angle(_params.angle, params.angle, rn)
        dtuple.angle = dangle; dval += abs(dangle)
        mtuple.angle = mangle; mval += mangle
        # angle of angle:
        maangle, daangle = comp_aangle(_params.aangle, params.aangle, rn)
        dtuple.aangle = daangle; dval += abs(daangle)
        mtuple.aangle = maangle; mval += maangle
    else:  # vertuple, all scalars:
        comp("val", _params.val, params.val*rn / daxis, dval, mval, dtuple, mtuple, ave_mval, finv=0)
        comp("angle", _params.angle, params.angle*rn / daxis, dval, mval, dtuple, mtuple, ave_dangle, finv=0)
        comp("aangle", _params.aangle, params.aangle*rn / daxis, dval, mval, dtuple, mtuple, ave_daangle, finv=0)
    # both:
    comp("I", _params.I, params.I*rn, dval, mval, dtuple, mtuple, ave_dI, finv=flatuple)  # inverse match if latuple
    comp("M", _params.M, params.M*rn / daxis, dval, mval, dtuple, mtuple, ave_M, finv=0)
    comp("Ma",_params.Ma, params.Ma*rn / daxis, dval, mval, dtuple, mtuple, ave_Ma, finv=0)

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

def comp_angle(_angle, angle, rn=1):

    _Dy, _Dx = _angle
    Dy, Dx = angle
    _G = np.hypot(_Dy,_Dx); G = np.hypot(Dy*rn,Dx*rn)
    sin = Dy*rn / (.1 if G == 0 else G); cos = Dx*rn / (.1 if G == 0 else G)
    _sin = _Dy / (.1 if _G == 0 else _G); _cos = _Dx / (.1 if _G == 0 else _G)
    sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β

    dangle = np.arctan2(sin_da, cos_da)  # scalar, vertical difference between angles
    mangle = ave_dangle - abs(dangle)  # inverse match, not redundant as summed across sign

    return mangle, dangle

def comp_aangle(_aangle, aangle, rn):

    _sin_da0, _cos_da0, _sin_da1, _cos_da1 = aangle
    sin_da0, cos_da0, sin_da1, cos_da1 = aangle

    sin_dda0 = (cos_da0 * rn * _sin_da0) - (sin_da0 * rn * _cos_da0)
    cos_dda0 = (cos_da0 * rn * _cos_da0) + (sin_da0 * rn * _sin_da0)
    sin_dda1 = (cos_da1 * rn * _sin_da1) - (sin_da1 * rn * _cos_da1)
    cos_dda1 = (cos_da1 * rn * _cos_da1) + (sin_da1 * rn * _sin_da1)
    # for 2D, not reduction to 1D:
    # aaangle = (sin_dda0, cos_dda0, sin_dda1, cos_dda1)
    # day = [-sin_dda0 - sin_dda1, cos_dda0 + cos_dda1]
    # dax = [-sin_dda0 + sin_dda1, cos_dda0 + cos_dda1]
    gay = np.arctan2((-sin_dda0 - sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in y?
    gax = np.arctan2((-sin_dda0 + sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in x?
    daangle = np.arctan2(gay, gax)  # diff between aangles, probably wrong
    maangle = ave_daangle - abs(daangle)  # inverse match, not redundant as summed

    return maangle, daangle


def agg_recursion_eval(dir_blob, PP_t):
    from agg_recursion import agg_recursion, Cgraph
    from sub_recursion import CPP2graph, CBlob2graph

    if not isinstance(dir_blob, Cgraph):
        fseg = isinstance(dir_blob, CPP)
        convert = CPP2graph if fseg else CBlob2graph

        dir_blob = convert(dir_blob, fseg=fseg, Cgraph=Cgraph)  # convert root to graph
        for PP_ in PP_t:
            for i, PP in enumerate(PP_):
                PP_[i] = CPP2graph(PP, fseg=fseg, Cgraph=Cgraph)  # convert PP to graph

    M, G = dir_blob.valt
    fork_rdnt = [1+(G>M), 1+(M>=G)]
    for fd, PP_ in enumerate(PP_t):  # PPm_, PPd_

        if (dir_blob.valt[fd] > PP_aves[fd] * ave_agg * (dir_blob.rdn+1) * fork_rdnt[fd]) \
            and len(PP_) > ave_nsub and dir_blob.alt_rdn < ave_overlap:
            dir_blob.rdn += 1  # estimate
            agg_recursion(dir_blob, PP_, fseg=fseg)

