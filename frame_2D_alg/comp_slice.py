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
from frame_blobs import CBlob

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
ave_sub = 2  # cost of calling sub_recursion and looping
ave_agg = 3  # cost of agg_recursion
ave_overlap = 10

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

class CderP(ClusterStructure):  # tuple of derivatives in P uplink_ or downlink_
    '''
    Derivation forms a binary tree where the root is latuple and all forks are vertuples.
    Players in derP are zipped with fds: taken forks. Players, mplayer, dplayer are replaced by those in PP
    '''
    players = list  # max n ptuples in layer = n ptuples in all lower layers: 1, 1, 2, 4, 8...
    link_player = lambda: [[],[]]  # last mplayer, dplayer
    valt = lambda: [0,0]  # tuple of mval, dval, summed from last player, both are signed
    x0 = int
    y = int  # or while isinstance(P, CderP): P = CderP._P; else y = _P.y
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
    nderP_t = lambda: [[],[]]  # miss links, add with nvalt for complemented PP
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

    segment_by_direction(blob, verbose=False)  # forms blob.dir_blobs
    for dir_blob in blob.dir_blobs:

        P__ = slice_blob(dir_blob, verbose=False)  # cluster dir_blob.dert__ into 2D array of blob slices
        comp_P_root(P__)  # scan_P_, comp_P | link_layer, adds mixed uplink_, downlink_ per P; comp_dx_blob(P__), comp_dx?

        # segments are stacks of (P,derP)s:
        segm_ = form_seg_root([copy(P_) for P_ in P__], fd=0, fds=[0])  # shallow copy: same Ps in different lists
        segd_ = form_seg_root([copy(P_) for P_ in P__], fd=1, fds=[0])  # initial latuple fd=0
        # PPs are graphs of segs:
        PPm_, PPd_ = form_PP_root((segm_, segd_), base_rdn=2)  # not mixed-fork PP_, select by PP.fPd?
        # not sure if this is a right place for it:
        dir_blob = CBlob2Cgraph(dir_blob, fseg, Cgraph)
        # intra PP:
        dir_blob.rlayers = sub_recursion_eval(PPm_, fd=0)
        dir_blob.dlayers = sub_recursion_eval(PPd_, fd=1)  # add rlayers, dlayers, seg_levels to select PPs
        # cross PP:
        dir_blob.mlevels = [PPm_]; dir_blob.dlevels = [PPd_]  # agg levels
        agg_recursion_eval(dir_blob, [PPm_, PPd_], fseg=0)

        # splice_dir_blob_(blob.dir_blobs)  # draft


def agg_recursion_eval(dir_blob, PP_t, fseg):
    from agg_recursion import agg_recursion, Cgraph

    if fseg: M = dir_blob.players[0][0].M; G = dir_blob.players[0][0].G  # dir_blob is CPP
    else:    M = dir_blob.M; G = dir_blob.G; dir_blob.valt = [M, G]      # update valt only if dir_blob is Cblob
    fork_rdnt = [1+(G>M), 1+(M>=G)]

    # cross PP:
    for fd, PP_ in enumerate(PP_t):
        alt_Rdn = 0
        for PP in PP_:
            if fseg:
                PP = PP.roott[PP.fds[-1]]  # seg root
            PP_P_ = [P for P_ in PP.P__ for P in P_]  # PPs' Ps
            for altPP in PP.altPP_:  # overlapping Ps from each alt PP
                altPP_P_ = [P for P_ in altPP.P__ for P in P_]  # altPP's Ps
                alt_rdn = len(set(PP_P_).intersection(altPP_P_))
                if not fseg:
                    PP.alt_rdn += alt_rdn  # count overlapping PPs, not bilateral, each PP computes its own alt_rdn
                alt_Rdn += alt_rdn  # sum across PP_

        if (dir_blob.valt[fd] > PP_aves[fd] * ave_agg * (dir_blob.rdn+1) * fork_rdnt[fd]) and len(PP_) > ave_nsub and alt_Rdn < ave_overlap:
            dir_blob.rdn += 1  # estimate
            agg_recursion(dir_blob, PP_, rng=2, fseg=fseg)


# rough draft:

def CBlob2Cgraph(dir_blob, fseg, Cgraph):

    root = Cgraph(node_=deepcopy(dir_blob.PPm_), alt_node_=deepcopy(dir_blob.PPd_))

    for fd, PP_ in enumerate([root.node_, root.alt_node_]):
        gPP_ = []
        for PP in PP_:
            # need to sum PP.alt_players from PP.altPP_
            sum_players(root.alt_plevels[0] if fd else root.plevels[0], PP.alt_players if fd else PP.players)
            gPP_ += [CPP2Cgraph(PP, fseg, Cgraph)]

        valt = [dir_blob.M, dir_blob.G]

    # graph = sum2graph_([[gPP_, valt]], fd)[0]  # pack PP into graph
    return root

def CPP2Cgraph(PP, fseg, Cgraph):
    alt_players, alt_fds = [], []
    alt_valt = [0, 0]

    if not fseg and PP.altPP_:  # seg doesn't have altPP_
        alt_fds = PP.altPP_[0].fds
        for altPP in PP.altPP_[1:]:  # get fd sequence common for all altPPs:
            for i, (_fd, fd) in enumerate(zip(alt_fds, altPP.fds)):
                if _fd != fd:
                    alt_fds = alt_fds[:i]
                    break
        for altPP in PP.altPP_:
            sum_players(alt_players, altPP.players[:len(alt_fds)])  # sum same-fd players only
            alt_valt[0] += altPP.valt[0];  alt_valt[1] += altPP.valt[1]

    alt_plevels = [[alt_players, alt_fds, alt_valt]]
    plevels = [[deepcopy(PP.players), deepcopy(PP.fds), deepcopy(PP.valt)]]

    return Cgraph(PP=PP, node_=[PP], plevels=plevels, alt_plevels=alt_plevels, fds=deepcopy(PP.fds), x0=PP.x0, xn=PP.xn, y0=PP.y0, yn=PP.yn)


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
        mplayer, dplayer, mval, dval = comp_players(_P.players, P.players)  # passed from seg.fds
        valt = [mval, dval]
        players = deepcopy(_P.players)

    return CderP(x0=min(_P.x0, P.x0), y=_P.y, players=players, link_player=[mplayer,dplayer], valt=valt, P=P, _P=_P)


def comp_P_rng(P__, rng):  # rng+ sub_recursion in PP.P__, switch to rng+n to skip clustering?

    for P_ in P__:
        for P in P_:  # add 2 link layers: rng_derP_ and match_rng_derP_:
            P.uplink_layers += [[],[[],[]]]; P.downlink_layers += [[],[[],[]]]

    for y, _P_ in enumerate(P__[:-rng]):  # higher compared row, skip last rng: no lower comparand rows
        for x, _P in enumerate(_P_):
            # get linked Ps at dy = rng-1:
            for pri_derP in _P.downlink_layers[-3][0]:  # fd always = 0 here
                pri_P = pri_derP.P
                # compare linked Ps at dy = rng:
                for ini_derP in pri_P.downlink_layers[0]:
                    P = ini_derP.P
                    # add new Ps, their link layers and reset their roots:
                    if P not in [P for P_ in P__ for P in P_]:
                        append_P(P__, P)
                        P.uplink_layers += [[],[[],[]]]; P.downlink_layers += [[],[[],[]]]
                    derP = comp_P(_P, P)
                    P.uplink_layers[-2] += [derP]
                    _P.downlink_layers[-2] += [derP]

    return P__

def comp_P_der(P__):  # der+ sub_recursion in PP.P__, compare P.uplinks to P.downlinks

    dderPs__ = []  # derP__ = [[] for P_ in P__[:-1]]  # init derP rows, exclude bottom P row

    for P_ in P__[1:-1]:  # higher compared row, exclude 1st: no +ve uplinks, and last: no +ve downlinks
        # not revised:
        dderPs_ = []  # row of dderPs
        for P in P_:
            dderPs = []  # dderP for each _derP, derP pair in P links
            for _derP in P.uplink_layers[-1][1]:  # fd=1
                for derP in P.downlink_layers[-1][1]:
                    # there maybe no x overlap between recomputed Ls of _derP and derP, compare anyway,
                    # mderP * (ave_olp_L / olp_L)? or olp(_derP._P.L, derP.P.L)?
                    # gap: neg_olp, ave = olp-neg_olp?
                    dderP = comp_P(_derP, derP)  # form higher vertical derivatives of derP or PP params
                    derP.uplink_layers[0] += [dderP]  # pre-init layer per derP
                    _derP.downlink_layers[0] += [dderP]
                    dderPs_ += [dderP]  # actually it could be dderPs_ ++ [derPP]
                # compute x overlap between dderP'__P and P, in form_seg_ or comp_layer?
            dderPs_ += dderPs  # row of dderPs
        dderPs__ += [dderPs_]

    return dderPs__


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
                        if PPm not in PP.altPP_: PP.altPP_ += [PPm]  # bilateral assignment of altPPs
                        if PP not in PPm.altPP_: PPm.altPP_ += [PP]  # PPd here
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
    seg = CPP(x0=seg_Ps[0].x0, P__= seg_Ps, uplink_layers=[miss_uplink_], downlink_layers = [miss_downlink_], y0 = seg_Ps[0].y)
    new_player = []
    for P in seg_Ps[:-1]:
        derP = P.uplink_layers[-1][fd][0]
        sum_player(new_player, derP.link_player[fd])
        accum_derP(seg, derP, fd)  # derP = P.uplink_layers[-1][0]
        P.roott[fd] = seg
        for derP in P.uplink_layers[-2]:
            if derP not in P.uplink_layers[-1][fd]:
                seg.nvalt[fd] += derP.valt[fd]  # -ve links in full links, not in +ve links
                seg.nderP_t += [derP]

    accum_derP(seg, seg_Ps[-1], fd)  # accum last P only, top P uplink_layers are not part of seg
    if new_player: seg.players += [new_player]  # add new player
    seg.y0 = seg_Ps[0].y
    seg.yn = seg.y0 + len(seg_Ps)
    seg.fds = fds + [fd]  # fds of root PP

    return seg

def accum_derP(seg, derP, fd):  # derP might be CP, though unlikely

    seg.x0 = min(seg.x0, derP.x0)

    if isinstance(derP, CP):
        derP.roott[fd] = seg
        if seg.players: sum_players(seg.players, [[derP.ptuple]])
        else:           seg.players.append([deepcopy(derP.ptuple)])
        seg.xn = max(seg.xn, derP.x0 + derP.ptuple.L)
    else:
        sum_players(seg.players, derP.players)  # last derP player is current mplayer, dplayer
        seg.valt[0] += derP.valt[0]; seg.valt[1] += derP.valt[1]
        seg.xn = max(seg.xn, derP.x0 + derP.players[0][0].L)


def sum2PP(PP_segs, base_rdn, fd):  # sum PP_segs into PP

    PP = CPP(x0=PP_segs[0].x0, rdn=base_rdn)
    if fd: PP.dseg_levels, PP.mseg_levels = [PP_segs], [[]]  # empty alt_seg_levels
    else:  PP.mseg_levels, PP.dseg_levels = [PP_segs], [[]]

    for seg in PP_segs:
        seg.roott[fd] = PP
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
        PP.nderP_t[fd] += seg.nderP_t[fd]
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
                    PP.nderP_t[i] += [derP]
    return PP


def sum_players(Layers, layers, fneg=0):  # no accum across fPd, that's checked in comp_players?

    if not Layers:
        if not fneg: Layers.append(deepcopy(layers[0]))
    else: accum_ptuple(Layers[0][0], layers[0][0], fneg)  # latuples, in purely formal nesting

    for Layer, layer in zip_longest(Layers[1:], layers[1:], fillvalue=[]):
        if layer:
            if Layer: sum_player(Layer, layer, fneg=fneg)
            elif not fneg: Layers.append(deepcopy(layer))


def sum_player(Player, player, fneg=0):  # accum players in Players

    for i, (Ptuple, ptuple) in enumerate(zip_longest(Player, player, fillvalue=[])):
        if ptuple:
            if Ptuple: accum_ptuple(Ptuple, ptuple, fneg)
            elif Ptuple == None: Player[i] = deepcopy(ptuple)
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


def comp_players(_layers, layers):  # unpack and compare der layers, if any from der+

    mplayer, dplayer = [],[]
    mval, dval = 0,0

    for _layer, layer in zip(_layers, layers):
        for _ptuple, ptuple in zip(_layer, layer):

            mtuple, dtuple = comp_ptuple(_ptuple, ptuple)
            mplayer +=[mtuple]; mval+=mtuple.val
            dplayer +=[dtuple]; dval+=dtuple.val

    return mplayer, dplayer, mval, dval


def comp_ptuple(_params, params):  # compare lateral or vertical tuples, similar operations for m and d params

    dtuple, mtuple = Cptuple(), Cptuple()
    dval, mval = 0, 0

    flatuple = isinstance(_params.angle, list)  # else vertuple
    rn = _params.n / params.n  # normalize param as param*rn for n-invariant ratio: _param / param*rn = (_param/_n) / (param/n)
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


def copy_P(P, iPtype=None):  # Ptype =0: P is CP | =1: P is CderP | =2: P is CPP | =3: P is CderPP | =4: P is CaggPP

    if not iPtype:  # assign Ptype based on instance type if no input type is provided
        if isinstance(P, CPP):
            Ptype = 2
        elif isinstance(P, CderP):
            Ptype = 1
        elif isinstance(P, CP):
            Ptype = 0
    else:
        Ptype = iPtype

    uplink_layers, downlink_layers = P.uplink_layers, P.downlink_layers  # local copy of link layers
    P.uplink_layers, P.downlink_layers = [], []  # reset link layers
    roott = P.roott  # local copy
    P.roott = [None, None]
    if Ptype == 1:
        P_derP, _P_derP = P.P, P._P  # local copy of derP.P and derP._P
        P.P, P._P = None, None  # reset
    elif Ptype == 2:
        mseg_levels, dseg_levels = P.mseg_levels, P.dseg_levels
        rlayers, dlayers = P.rlayers, P.dlayers
        P__ = P.P__
        P.mseg_levels, P.dseg_levels, P.rlayers, P.dlayers, P.P__ = [], [], [], [], []  # reset
    elif Ptype == 3:
        PP_derP, _PP_derP = P.PP, P._PP  # local copy of derP.P and derP._P
        P.PP, P._PP = None, None  # reset
    elif Ptype == 4:
        gPP_, cPP_ = P.gPP_, P.cPP_
        rlayers, dlayers = P.rlayers, P.dlayers
        mlevels, dlevels = P.mlevels, P.dlevels
        P.gPP_, P.cPP_, P.rlayers, P.dlayers, P.mlevels, P.dlevels = [], [], [], [], [], []  # reset

    new_P = deepcopy(P)  # copy P with empty root and link layers, reassign link layers:
    new_P.uplink_layers += copy(uplink_layers)
    new_P.downlink_layers += copy(downlink_layers)

    # shallow copy to create new list
    P.uplink_layers, P.downlink_layers = copy(uplink_layers), copy(downlink_layers)  # reassign link layers
    P.roott = roott  # reassign root
    # reassign other list params
    if Ptype == 1:
        new_P.P, new_P._P = P_derP, _P_derP
        P.P, P._P = P_derP, _P_derP
    elif Ptype == 2:
        P.mseg_levels, P.dseg_levels = mseg_levels, dseg_levels
        P.rlayers, P.dlayers = rlayers, dlayers
        new_P.rlayers, new_P.dlayers = copy(rlayers), copy(dlayers)
        new_P.P__ = copy(P__)
        new_P.mseg_levels, new_P.dseg_levels = copy(mseg_levels), copy(dseg_levels)
    elif Ptype == 3:
        new_P.PP, new_P._PP = PP_derP, _PP_derP
        P.PP, P._PP = PP_derP, _PP_derP
    elif Ptype == 4:
        P.gPP_, P.cPP_ = gPP_, cPP_
        P.rlayers, P.dlayers = rlayers, dlayers
        P.roott = roott
        new_P.gPP_, new_P.cPP_ = [], []
        new_P.rlayers, new_P.dlayers = copy(rlayers), copy(dlayers)
        new_P.mlevels, new_P.dlevels = copy(mlevels), copy(dlevels)

    return new_P


# old draft
def splice_dir_blob_(dir_blobs):
    for i, _dir_blob in enumerate(dir_blobs):
        for fPd in 0, 1:
            PP_ = _dir_blob.levels[0][fPd]
            if fPd:
                PP_val = sum([PP.mP for PP in PP_])
            else:
                PP_val = sum([PP.dP for PP in PP_])

            if PP_val - ave_splice > 0:  # high mPP pr dPP
                _top_P_ = _dir_blob.P__[0]
                _bottom_P_ = _dir_blob.P__[-1]
                for j, dir_blob in enumerate(dir_blobs):
                    if _dir_blob is not dir_blob:

                        top_P_ = dir_blob.P__[0]
                        bottom_P_ = dir_blob.P__[-1]
                        # test y adjacency:
                        if (_top_P_[0].y - 1 == bottom_P_[0].y) or (top_P_[0].y - 1 == _bottom_P_[0].y):
                            # test x overlap:
                            if (dir_blob.x0 - 1 < _dir_blob.xn and dir_blob.xn + 1 > _dir_blob.x0) \
                                    or (_dir_blob.x0 - 1 < dir_blob.xn and _dir_blob.xn + 1 > dir_blob.x0):
                                splice_2dir_blobs(_dir_blob, dir_blob)  # splice dir_blob into _dir_blob
                                dir_blobs[j] = _dir_blob

def splice_2dir_blobs(_blob, blob):
    # merge blob into _blob here
    pass


def sub_recursion_eval(PP_, fd):  # for PP or dir_blob

    from agg_recursion import agg_recursion, Cgraph
    mcomb_layers, dcomb_layers, PPm_, PPd_ = [], [], [], []
    for PP in PP_:
        if fd:  # add root to derP for der+:
            for P_ in PP.P__[1:-1]:  # skip 1st and last row
                for P in P_:
                    for derP in P.uplink_layers[-1][fd]:
                        derP.roott[fd] = PP
            comb_layers = dcomb_layers; PP_layers = PP.dlayers; PPd_ += [PP]
        else:
            comb_layers = mcomb_layers; PP_layers = PP.rlayers; PPm_ += [PP]
        val = PP.valt[fd]; alt_val = PP.valt[1-fd]  # for fork rdn:
        ave = PP_aves[fd] * (PP.rdn + 1 + (alt_val > val))
        if val > ave and len(PP.P__) > ave_nsub:
            sub_recursion(PP)  # comp_P_der | comp_P_rng in PPs -> param_layer, sub_PPs
            ave*=2  # 1+PP.rdn incr
            # splice deeper layers between PPs into comb_layers:
            for i, (comb_layer, PP_layer) in enumerate(zip_longest(comb_layers, PP_layers, fillvalue=[])):
                if PP_layer:
                    if i > len(comb_layers) - 1: comb_layers += [PP_layer]  # add new r|d layer
                    else: comb_layers[i] += PP_layer  # splice r|d PP layer into existing layer

        # segs agg_recursion:
        agg_recursion_eval(PP, [PP.mseg_levels[-1], PP.dseg_levels[-1]], fseg=1)

    return [[PPm_] + mcomb_layers], [[PPd_] + dcomb_layers]  # including empty comb_layers


def sub_recursion(PP):  # evaluate each PP for rng+ and der+

    P__  = [P_ for P_ in reversed(PP.P__)]  # revert to top down
    P__ = comp_P_der(P__) if PP.fds[-1] else comp_P_rng(P__, PP.rng + 1)   # returns top-down

    PP.rdn += 2  # two-fork rdn, priority is not known?
    sub_segm_ = form_seg_root([copy(P_) for P_ in P__], fd=0, fds=PP.fds)
    sub_segd_ = form_seg_root([copy(P_) for P_ in P__], fd=1, fds=PP.fds)  # returns bottom-up
    sub_PPm_, sub_PPd_ = form_PP_root((sub_segm_, sub_segd_), PP.rdn + 1)  # PP is parameterized graph of linked segs

    PP.rlayers = sub_recursion_eval(sub_PPm_, fd=0)  # add rlayers, dlayers, seg_levels to select sub_PPs
    PP.dlayers = sub_recursion_eval(sub_PPd_, fd=1)