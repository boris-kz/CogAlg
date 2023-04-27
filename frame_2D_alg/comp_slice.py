# warnings.filterwarnings('error')
# import warnings  # to detect overflow issue, in case of infinity loop
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
from class_cluster import ClusterStructure, NoneType

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
ave_x = 1
ave_dx = 5  # inv, difference between median x coords of consecutive Ps
ave_dy = 5
ave_daxis = 2
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
med_decay = .5
pnames = ["I", "M", "Ma", "axis", "angle", "aangle","G", "Ga", "x", "L"]
aves = [ave_dI, ave_M, ave_Ma, ave_daxis, ave_dangle, ave_daangle, ave_G, ave_Ga, ave_dx, ave_L, ave_mval, ave_dval]
vaves = [ave_mval, ave_dval]
PP_aves = [ave_mPP, ave_dPP]

class Cptuple(ClusterStructure):  # bottom-layer tuple of compared params in P, derH per par in derP, or PP

    I = int  # [m,d] in higher layers:
    M = int
    Ma = float
    axis = lambda: [1, 0]  # ini dy=1,dx=0, old angle after rotation
    angle = lambda: [0, 0]  # in latuple only, replaced by float in vertuple
    aangle = lambda: [0, 0, 0, 0]
    G = float  # for comparison, not summation:
    Ga = float
    x = int  # median: x0+L/2
    L = int  # len dert_ in P, area in PP
    n = lambda: 1  # accum count, combine from CpH?
    valt = lambda: [0,0]
    rdnt = lambda: [1,1]

class CQ(ClusterStructure):  # vertuple, hierarchy, or generic sequence

    Q = list  # generic sequence or index increments in ptuple, derH, etc
    Qm = list  # in-graph only
    Qd = list
    ext = lambda: [[],[]]  # [ms,ds], per subH only
    valt = lambda: [0,0]  # in-graph vals
    rdnt = lambda: [1,1]  # none if represented m and d?
    out_valt = lambda: [0,0]  # of non-graph links, as alt?
    fds = list
    rng = lambda: 1  # is it used anywhere?
    n = int  # accum count in ptuple


class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

    ptuple = object  # latuple: I, M, Ma, G, Ga, angle(Dy, Dx), aangle( Sin_da0, Cos_da0, Sin_da1, Cos_da1), ?[n, val, x, L, A]?
    x0 = int
    y0 = int  # for vertical gap in PP.P__
    dert_ = list  # array of pixel-level derts, redundant to uplink_, only per blob?
    link_ = list  # all links
    link_t = lambda: [[],[]]  # +ve rlink_, dlink_
    roott = lambda: [None,None]  # m,d PP that contain this P
    rdn = int  # blob-level redundancy, ignore for now
    dxdert_ = list  # only in Pd
    Pd_ = list  # only in Pm
    # if comp_dx:
    Mdx = int
    Ddx = int

class CderP(ClusterStructure):  # tuple of derivatives in P link: binary tree with latuple root and vertuple forks

    rngQ = lambda: [[],0]  # [derQ, nlink]: sum links / rng+, max ntuples / der layer = ntuples in lower layers: 1, 1, 2, 4, 8...
    valt = lambda: [0,0]
    rdnt = lambda: [1,1]  # mrdn + uprdn if branch overlap?
    _P = object  # higher comparand
    P = object  # lower comparand
    roott = lambda: [None,None]  # for der++
    x0 = int
    y0 = int
    L = int
    fdx = NoneType  # if comp_dx
'''
lay1: par     # derH per param in vertuple, each layer is derivatives of all lower layers:
lay2: [m,d]   # implicit nesting, brackets for clarity:
lay3: [[m,d], [md,dd]]: 2 sLays,
lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays:
'''

class CPP(CderP):  # derP params include P.ptuple

    rngH = lambda: [[],0]  # [derQ, nlink] s, 1st plevel, zipped with alt_derH in comp_derH
    valt = lambda: [0,0]
    rdnt = lambda: [1,1]  # recursion count + Rdn / nderPs + mrdn + uprdn if branch overlap?
    Rdn = int  # for accumulation only?
    rng = lambda: 1
    alt_rdn = int  # overlapping redundancy between core and edge
    alt_PP_ = list  # adjacent alt-fork PPs per PP, from P.roott[1] in sum2PP
    altuple = list  # summed from alt_PP_, sub comp support, agg comp suppression?
    nval = int
    box = lambda: [0,0,0,0]  # y0,yn, x0,xn
    P_cnt = int  # len 2D derP__ in levels[0][fPd]?  ly = len(derP__), also x, y?
    derP_cnt = int  # redundant per P
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    nderP_ = list  # miss links, add with nvalt for complemented PP?
    mask__ = bool
    link__ = list  # combined P link_t[fd] s?
    P__ = list  # input + derPs
    rlayers = list  # or mlayers: sub_PPs from sub_recursion within PP?
    dlayers = list  # or alayers
    roott = lambda: [None,None]  # PPPm, PPPd that contain this PP
    cPP_ = list  # rdn reps in other PPPs, to eval and remove


def comp_slice_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    from sub_recursion import sub_recursion_eval, rotate_P_, agg_recursion_eval

    P__ = slice_blob(blob, verbose=False)  # form 2D array of Ps: blob slices in dert__
    # rotate each P to align it with the direction of P gradient:
    rotate_P_(P__, blob.dert__, blob.mask__)  # rotated Ps are sparse or overlap via redundant derPs, results are not biased?
    # scan rows top-down, comp y-adjacent, x-overlapping Ps, form derP__:
    _P_ = P__[0]  # higher row
    for P_ in P__[1:]:  # lower row
        for P in P_:
            for _P in _P_:  # test for x overlap(_P,P) in 8 directions, derts are positive in all Ps:
                _L = len(_P.dert_); L = len(P.dert_)
                if (P.x0 - 1 < _P.x0 + _L) and (P.x0 + L > _P.x0):
                    vertuple = comp_ptuple(_P.ptuple, P.ptuple)
                    valt = copy(vertuple.valt); rdnt = copy(vertuple.rdnt)
                    derP = CderP(rngQ=[[vertuple], 1], valt=valt, rdnt=rdnt, P=P, _P=_P, x0=_P.x0, y0=_P.y0, L=len(_P.dert_))
                    P.link_+=[derP]  # all links
                    if valt[0] > aveB*rdnt[0]: P.link_t[0] += [derP]  # +ve links, fork overlap?
                    if valt[1] > aveB*rdnt[1]: P.link_t[1] += [derP]
                elif (P.x0 + L) < _P.x0:
                    break  # no xn overlap, stop scanning lower P_
        _P_ = P_
    PPm_,PPd_ = form_PP_t(P__, base_rdn=2)
    blob.PPm_, blob.PPd_ = PPm_, PPd_; blob.rlayers += [PPm_]; blob.dlayers += [PPd_]
    # re comp, cluster:
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
                params.G = np.hypot(*params.angle)  # Dy, Dx  # recompute G, Ga, which can't reconstruct M, Ma
                params.Ga = (params.aangle[1] + 1) + (params.aangle[3] + 1)  # Cos_da0, Cos_da1
                L = len(Pdert_)
                params.L = L; params.x = x-L/2
                P_.append( CP(ptuple=params, x0=x-(L-1), y0=y, dert_=Pdert_))
            _mask = mask
        # pack last P, same as above:
        if not _mask:
            params.G = np.hypot(*params.angle); params.Ga = (params.aangle[1] + 1) + (params.aangle[3] + 1)
            L = len(Pdert_); params.L = L; params.x = x-L/2
            P_.append(CP(ptuple=params, x0=x - (L - 1), y0=y, dert_=Pdert_))
        P__ += [P_]

    blob.P__ = P__
    return P__


def form_PP_t_Chee(P__, base_rdn):  # form PPs from derP.valt[fd] -connected Ps

    PP_t = []
    for fd in 0, 1:
        fork_P__ = reversed([P for P in P_ for P_ in P__][1:])  # scan bottom-up, copy per fd (flat fork_P__ now)
        PP_ = []
        while fork_P__:
            P = fork_P__.pop(0)
            if P.link_t[fd]:
                qPP = [[P], 0]
                for derP in P.link_t[fd]:
                    if derP._P in fork_P__:
                        qPP[0] += [derP._P]
                        qPP[1] += derP.valt[fd]
                        fork_P__.remove(derP._P)
                PP_ += [qPP]
        # prune PPs by P val:
        rePP_ = [reval_PP(qPP, aveB, fd, base_rdn) for qPP in PP_]  # qPP = [qPP,val]
        if rePP_:
            PP_[:] = [sum2PP(rePP, fd) for rePP in rePP_]
        PP_t += [rePP_]

    return PP_t  # add_alt_PPs_(graph_t)?


def form_PP_t(P__, base_rdn):  # form PPs from derP.valt[fd] -connected Ps

    PP_t = []
    for fd in 0, 1:
        fork_P__ = reversed([copy(P_) for P_ in P__][1:])  # scan bottom-up, copy per fd
        iPP_ = []
        for P_ in fork_P__:
            for P in P_:
                if P.link_t[fd]: iPP_ += [P]  # all Ps with +ve links, not in PPs yet
        PP_,rePP_ = [],[]
        while iPP_:  # all PPs not removed in init_PP_
            P = iPP_.pop(); qPP = [P]  # iPPs are single-P, not sure
            PP_ += [init_PP(qPP, iPP_, P, fd, val=0)]  # flood-fill qPPs with vertically matched | diffed Ps
        # prune PPs by P val:
        rePP_ = [reval_PP(qPP, aveB, fd, base_rdn) for qPP in PP_]  # qPP = [qPP,val]
        if rePP_:
            PP_[:] = [sum2PP(rePP, fd) for rePP in rePP_]
        PP_t += [rePP_]

    return PP_t  # add_alt_PPs_(graph_t)?

def init_PP(qPP, iPP_, P, fd, val):  # recursive qPP_+=[_P]

    for derP in P.link_t[fd]:  # all positive links init PP
        _P = derP._G
        if _P in iPP_:  # _P is not removed in prior loop
            qPP += [_P]
            iPP_.remove(_P)
            val += derP.valt[fd]
            val += init_PP(qPP, iPP_, _P, fd, val)
    return [qPP, val]

def reval_PP(qPP, ave, fd, base_rdn):  # prune qPP by link_ val adjusted by _P.val

    for P_ in qPP[0]:  # P__
        for P in P_:
            link_eval(P, fd, qPP)  # prune P.link_ as in graph_reval, accum pre_PP_ and reval

    if max([reval for (P,reval) in qPP]) > ave:  # if any node should be revalued
        reval_PP(qPP, ave, fd, base_rdn)
    else:
        return sum2PP(qPP, fd=fd)

# partial draft
def link_eval(P, fd, qPP):  # prune P.link_ as in graph_reval, accum pre_PP_ and reval
    '''
    if not fd: rng_eval(derP, fd)  # reset derP.valt, derP.rdn
    mrdn = derP.valt[1 - fd] > derP.valt[fd]
    derP.rdnt[fd] += not mrdn if fd else mrdn
    '''
    reval = 0
    for i, derP in enumerate(P.link_t[fd]):
        med_val = 0
        val = derP.valt[fd]
        for link in derP._P.link_t[fd]:  # _P.link_t was updated in previous round
            if link!=derP: med_val += link.valt[fd]  # not circular link
        if val + med_val*med_decay < vaves[fd] * derP.rdnt[fd] * (len(derP.rngQ)-2):  # ave * rdn to stronger|prior derPs?
            del(P.link_t[fd][i])
            reval += val  # no P.valt to adjust?
    # add section to prune Ps, as graph_reval
    # append pre_PP in P.roott[fd] with P or merge in _P.roott[fd] if any,
    # term pre_PP if empty pre_PP link__

# not fully updated
def sum2PP(P_, base_rdn, fds, fd):  # sum PP_segs into PP

    from sub_recursion import append_P

    PP = CPP(rngH = [Cptuple()], x0=P_[0].x0, rdn=base_rdn, fds=copy(fds)+[fd], rlayers=[[]], dlayers=[[]])

    rngQ = [[], 0]
    PP.box = [P_[0].y0, 0, P_[0].x0, 0]
    for P in P_:
        P.roott[fd] = PP

        # get missed links
        for link in P.link:
            if link not in P.link_t[fd]:
                PP.link__ += [link]

        # update box
        PP.box[0] = min(PP.box[0], P.y0)  # y0
        PP.box[1] = max(PP.box[1], P.y0)  # yn
        PP.box[2] = min(PP.box[2], P.x0)  # x0
        PP.box[3] = max(PP.box[3], P.x0 + len(P.dert_))  # xn

        # sum P
        sum_ptuple(PP.rngH[0], P.ptuple)
        # sum links
        if P.link_[fd]:
            # pack derQ
            for derP in P.link_[fd]:
                sum_rngQ(rngQ, derP.rngQ)
            for derP in P.link_:
                # positive links: update valt and rdnt
                if derP in P.link_t[fd]:
                    for i in 0, 1:
                        PP.valt[i] += derP.valt[i]
                        PP.rdnt[i] += derP.rdnt[i]
                # negative links: update nval and nderP
                else:
                    PP.nval += derP.valt[fd]  # negative link
                    PP.nderP_ += [derP]

        # pack P into PP.P__
        if not PP.P__: PP.P__.append([P])  # pack 1st P
        else: append_P(PP.P__, P)  # pa

    if rngQ: # pack new derQ
        PP.rngH += [rngQ]  # so PP.rngQ's first element is ptuple?

    return PP


# change to sum2PP:
def sum2seg(seg_Ps, fd, fds):  # sum params of vertically connected Ps into segment

    uplink_, uuplink_t  = seg_Ps[-1].uplink_layers[-2:]  # uplinks of top P? or this is a bottom?
    miss_uplink_ = [uuplink for uuplink in uuplink_t[fd] if uuplink not in uplink_]  # in layer-1 but not in layer-2
    downlink_, ddownlink_t = seg_Ps[0].downlink_layers[-2:]  # downlinks of bottom P, downlink.P.seg.uplinks= lower seg.uplinks
    miss_downlink_ = [ddownlink for ddownlink in ddownlink_t[fd] if ddownlink not in downlink_]
    # seg rdn: up ini cost, up+down comp_seg cost in 1st agg+? P rdn = up+down M/n?
    P = seg_Ps[0]  # top P in stack
    L = len(P.dert_) if isinstance(P,CP) else P.L
    seg = CPP(P__= seg_Ps, uplink_layers=[miss_uplink_], downlink_layers = [miss_downlink_], fds=copy(fds)+[fd],
              box=[P.y0, P.y0+len(seg_Ps), P.x0, P.x0+L-1])
    P.roott[fd] = seg  # first and last P doesn't belong to any seg or PP? Else we need to assign it too
    derH = []
    derQ = deepcopy(P.uplink_layers[-1][fd][0].derQ) if len(seg_Ps)>1 else []  # 1 up derP per P in stack
    for P in seg_Ps[1:-1]:
        sum_derH(derH, [P.ptuple] if isinstance(P,CP) else P.derQ)  # if P is derP
        derP = P.uplink_layers[-1][fd][0]  # must exist
        sum_derH(derQ, derP.derQ)
        L = len(P.dert_) if isinstance(P,CP) else P.L
        seg.box[2]= min(seg.box[2],P.x0); seg.box[3]= max(seg.box[3],P.x0+L-1)
        # AND seg.fds?
        P.roott[fd] = seg
        for derP in P.uplink_layers[-2]:
            if derP not in P.uplink_layers[-1][fd]:
                seg.nval += derP.valt[fd]  # negative link
                seg.nderP_ += [derP]
    P = seg_Ps[-1]  # sum last P only, last P uplink_layers are not part of seg:
    P.roott[fd] = seg
    L = len(P.dert_) if isinstance(P,CP) else P.L
    sum_derH(derH, [P.ptuple] if isinstance(P,CP) else P.derQ)
    seg.box[2] = min(seg.box[2],P.x0); seg.box[3] = max(seg.box[3],P.x0+L-1)
    seg.derH = derH
    if derQ: # if fd:
        seg.derH += derQ  # der+
        # else: seg.derH[int(len(derH)/2):] = derQ  # rng+, replace last layer
    return seg
'''
        for PP in PP_:
            for P_ in PP.P__:
                for P in P_:
                    if fd:
                        PPm = P.roott[0].roott[0]  # get PP from P.seg, prevent seg is replaced by PP, because we need to access P.seg in sub+ later
                        if PPm not in PP.alt_PP_:
                            PP.alt_PP_ += [PPm]  # bilateral assignment of alt_PPs
                        if PP not in PPm.alt_PP_:
                            PPm.alt_PP_ += [PP]  # PPd here
'''

def sum2PP(PP_segs, base_rdn, fd):  # sum PP_segs into PP
    from sub_recursion import append_P
    PP = CPP(x0=PP_segs[0].x0, rdn=base_rdn, fds=copy(PP_segs[0].fds)+[fd], rlayers=[[]], dlayers=[[]])
    if fd: PP.dseg_levels, PP.mseg_levels = [PP_segs], [[]]  # empty alt_seg_levels
    else:  PP.mseg_levels, PP.dseg_levels = [PP_segs], [[]]
    for seg in PP_segs:
        seg.roott[fd] = PP
        # selection should be alt, not fd, only in convert?
        if PP.derH:
            sum_derH(PP.derH, seg.derH)  # not empty
        else: PP.derH = deepcopy(seg.derH)
        Y0,Yn,X0,Xn = PP.box; y0,yn,x0,xn = PP.box
        PP.box[:] = min(Y0,y0),max(Yn,yn),min(X0,x0),max(Xn,xn)
        for i in range(2):
            PP.valt[i] += seg.valt[i]
            PP.rdnt[i] += seg.rdnt[i]  # base_rdn + PP.Rdn / PP: recursion + forks + links: nderP / len(P__)?
        PP.derP_cnt += len(seg.P__[-1].uplink_layers[-1][fd])  # redundant derivatives of the same P
        # only PPs are sign-complemented, seg..[not fd]s are empty:
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
        for derP in seg.P__[-1].uplink_layers[-2]:  # loop terminal branches
            if derP in seg.P__[-1].uplink_layers[-1][fd]:  # +ve links
                PP.valt[0] += derP.valt[0]  # sum val
                PP.valt[1] += derP.valt[1]
            else:  # -ve links
                PP.nval += derP.valt[fd]  # different from altuple val
                PP.nderP_ += [derP]
    return PP


def sum_rngQ(RngQ, rngQ, fneg=0):

    for DerQ, derQ  in zip_longest(RngQ[0], rngQ[0], fillvalue=[]):
        if derQ:
            if DerQ:
                for Vertuple, vertuple  in zip_longest(DerQ, derQ, fillvalue=[]):
                    if vertuple:
                        if Vertuple:
                            if isinstance(vertuple, CQ):
                                sum_vertuple(Vertuple, vertuple, fneg)
                            else:
                                sum_ptuple(Vertuple, vertuple, fneg)
                        else:
                            DerQ += [deepcopy(vertuple)]
            elif not fneg:
                DerQ[:] = deepcopy(derQ)
            RngQ[1] += rngQ[1]

def sum_derH(DerH, derH, fneg=0):  # same fds from comp_derH

    for Vertuple, vertuple in zip_longest(DerH, derH, fillvalue=[]):
        if vertuple:
            if Vertuple:
                if isinstance(vertuple, CQ):
                    sum_vertuple(Vertuple, vertuple, fneg)
                else:
                    sum_ptuple(Vertuple, vertuple, fneg)
            elif not fneg:
                DerH += [deepcopy(vertuple)]


def sum_vertuple(Vertuple, vertuple, fneg=0):

    for i, (m, d) in enumerate(zip(vertuple.Qm, vertuple.Qd)):
        Vertuple.Qm[i] += -m if fneg else m
        Vertuple.Qd[i] += -d if fneg else d
    for i in 0,1:
        Vertuple.valt[i] += vertuple.valt[i]
        Vertuple.rdnt[i] += vertuple.rdnt[i]
    Vertuple.n += 1

def sum_ptuple(Ptuple, ptuple, fneg=0):

    for pname, ave in zip(pnames, aves):
        Par = getattr(Ptuple, pname); par = getattr(ptuple, pname)
        # angle or aangle:
        if isinstance(Par, list):
            for j, (P,p) in enumerate(zip(Par,par)): Par[j] = P-p if fneg else P+p
        else:
            Par += (-par if fneg else par)
        setattr(Ptuple, pname, Par)

    Ptuple.valt[0] += ptuple.valt[0]; Ptuple.valt[1] += ptuple.valt[1]
    Ptuple.n += 1

def comp_vertuple(_vertuple, vertuple):

    dtuple=CQ(n=_vertuple.n, Q=copy(_vertuple.Q))  # no selection here
    rn = _vertuple.n/vertuple.n  # normalize param as param*rn for n-invariant ratio: _param/ param*rn = (_param/_n)/(param/n)

    for _par, par, ave in zip(_vertuple.Qd, vertuple.Qd, aves):

        m,d = comp_par(_par[1], par[1]*rn, ave)
        dtuple.Qm+=[m]; dtuple.Qd+=[d]; dtuple.valt[0]+=m; dtuple.valt[1]+=d

    return dtuple

def comp_ptuple(_ptuple, ptuple):

    dtuple = CQ(n=_ptuple.n, Q=[0 for par in pnames])
    rn = _ptuple.n / ptuple.n  # normalize param as param*rn for n-invariant ratio: _param / param*rn = (_param/_n) / (param/n)

    for pname, ave in zip(pnames, aves):
        _par = getattr(_ptuple, pname)
        par = getattr(ptuple, pname)
        if pname=="aangle": m,d = comp_aangle(_par, par)
        elif pname in ("axis","angle"): m,d = comp_angle(_par, par)
        else:
            if pname!="x": par*=rn  # normalize by relative accum count
            if pname=="x" or pname=="I": finv = 1
            else: finv=0
            m,d = comp_par(_par, par, ave, finv)

        dtuple.Qm += [m]; dtuple.Qd += [d]; dtuple.valt[0] += m; dtuple.valt[1] += d

    return dtuple

def comp_par(_param, param, ave, finv=0):  # comparand is always par or d in [m,d]

    d = _param - param
    if finv: m = ave - abs(d)  # inverse match for primary params, no mag/value correlation
    else:    m = min(_param, param) - ave
    return [m,d]

def comp_angle(_angle, angle):  # rn doesn't matter for angles

    _Dy, _Dx = _angle
    Dy, Dx = angle
    _G = np.hypot(_Dy,_Dx); G = np.hypot(Dy,Dx)
    sin = Dy / (.1 if G == 0 else G);     cos = Dx / (.1 if G == 0 else G)
    _sin = _Dy / (.1 if _G == 0 else _G); _cos = _Dx / (.1 if _G == 0 else _G)
    sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β

    dangle = np.arctan2(sin_da, cos_da)  # scalar, vertical difference between angles
    mangle = ave_dangle - abs(dangle)  # inverse match, not redundant as summed across sign

    return [mangle, dangle]

def comp_aangle(_aangle, aangle):

    _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _aangle
    sin_da0, cos_da0, sin_da1, cos_da1 = aangle

    sin_dda0 = (cos_da0 * _sin_da0) - (sin_da0 * _cos_da0)
    cos_dda0 = (cos_da0 * _cos_da0) + (sin_da0 * _sin_da0)
    sin_dda1 = (cos_da1 * _sin_da1) - (sin_da1 * _cos_da1)
    cos_dda1 = (cos_da1 * _cos_da1) + (sin_da1 * _sin_da1)
    # for 2D, not reduction to 1D:
    # aaangle = (sin_dda0, cos_dda0, sin_dda1, cos_dda1)
    # day = [-sin_dda0 - sin_dda1, cos_dda0 + cos_dda1]
    # dax = [-sin_dda0 + sin_dda1, cos_dda0 + cos_dda1]
    gay = np.arctan2((-sin_dda0 - sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in y?
    gax = np.arctan2((-sin_dda0 + sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in x?

    daangle = np.arctan2(gay, gax)  # diff between aangles, probably wrong
    maangle = ave_daangle - abs(daangle)  # inverse match, not redundant as summed

    return [maangle,daangle]

# not revised:
def rng_eval(iderP, fd):  # compute value of combined mutual derPs: overlap between P uplinks and _P downlinks

    _P, P = iderP._P, iderP.P

    _derP_, derP_ = [], []
    _P_, P_ = [_P], [P]

    # get all uplinks of _P and P
    while _P_:
        _P = _P_.pop(0)
        for _derP in _P.link_t[fd]:
            _derP_ += _P.link_t[fd]
            _P_ += [_derP._P]
    while P_:
        P = P_.pop(0)
        for derP in P.link_t[fd]:
            derP_ += P.link_t[fd]
            P_ += [derP._P]

    # overlap between +ve P uplinks and +ve _P downlinks:
    common_derP_ = list( set(_derP_).intersection(derP_))
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