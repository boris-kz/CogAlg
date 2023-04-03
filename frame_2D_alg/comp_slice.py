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
from class_cluster import ClusterStructure, NoneType, comp_param, Cdert

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

class CpQ(ClusterStructure):  # vertuple or generic sequence

    Q = list  # param set | H | sequence
    N = list  # names'indices if selective representation, in agg+
    valt = lambda: [0,0]
    rdnt = lambda: [1,1]  # for all Qs? rdn if par: both m and d are represented?
    n = lambda: 1  # accum count, combine from CpH?
    nval = int  # of open links: base alt rep
    fds = list  # der+|rng+
    rng = lambda: 1

class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

    ptuple = object  # latuple: I, M, Ma, G, Ga, angle(Dy, Dx), aangle( Sin_da0, Cos_da0, Sin_da1, Cos_da1), ?[n, val, x, L, A]?
    x0 = int
    y0 = int  # for vertical gap in PP.P__
    dert_ = list  # array of pixel-level derts, redundant to uplink_, only per blob?
    uplink_layers = lambda: [[], [[],[]]]  # init a layer of derPs and a layer of match_derPs
    downlink_layers = lambda: [[], [[],[]]]
    roott = lambda: [None,None]  # m,d seg|PP that contain this P
    rdn = int  # blob-level redundancy, ignore for now
    # only in Pd:
    dxdert_ = list
    # only in Pm:
    Pd_ = list
    # if comp_dx:
    Mdx = int
    Ddx = int

class CderP(ClusterStructure):  # tuple of derivatives in P uplink_ or downlink_: binary tree with latuple root and vertuple forks

    derQ = list  # last player only, max ntuples in layer = ntuples in lower layers: 1, 1, 2, 4, 8...
    fds = list  # fd: der+|rng+, forming m,d per par of derH, same for clustering by m->rng+ or d->der+
    valt = lambda: [0,0]
    rdnt = lambda: [1,1]  # mrdn + uprdn if branch overlap?
    _P = object  # higher comparand
    P = object  # lower comparand
    roott = lambda: [None,None]  # for der++
    x0 = int
    y0 = int
    L = int
    uplink_layers = lambda: [[], [[],[]]]  # init a layer of dderPs and a layer of match_dderPs, not updated
    downlink_layers = lambda: [[], [[],[]]]
    fdx = NoneType  # if comp_dx

class CPP(CderP):  # derP params include P.ptuple
    '''
    lay1: par     # derH per param in vertuple, each layer is derivatives of all lower layers:
    lay2: [m,d]   # implicit nesting, brackets for clarity:
    lay3: [[m,d], [md,dd]]: 2 sLays,
    lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays:
    '''
    derH = list  # 1st plevel, zipped with alt_derH in comp_derH
    fds = list
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
    uplink_layers = lambda: [[]]  # the links here will be derPPs from discontinuous comp, not in layers?
    downlink_layers = lambda: [[]]
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    nderP_ = list  # miss links, add with nvalt for complemented PP?
    mask__ = bool
    P__ = list  # input + derPs, common root for downward layers and upward levels:
    rlayers = list  # or mlayers: sub_PPs from sub_recursion within PP
    dlayers = list  # or alayers
    mseg_levels = list  # from 1st agg_recursion[fPd], seg_levels[0] is seg_, higher seg_levels are segP_..s
    dseg_levels = list
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
                    derP = CderP(derQ=[vertuple], valt=copy(vertuple.valt), rdnt=(vertuple.rdnt), P=P, _P=_P, x0=_P.x0, y0=_P.y0, L=len(_P.dert_))
                    P.uplink_layers[-2] += [derP]  # uplink_layers[-1] is match_derPs
                    _P.downlink_layers[-2] += [derP]
                elif (P.x0 + L) < _P.x0:
                    break  # no xn overlap, stop scanning lower P_
        _P_ = P_
    # form segments: stacks of (P,derP)s:
    segm_ = form_seg_root([copy(P_) for P_ in P__], fd=0, fds=[0])  # shallow copy: same Ps in different lists
    segd_ = form_seg_root([copy(P_) for P_ in P__], fd=1, fds=[0])  # initial latuple fd=0
    # form PPs: graphs of segs:
    PPm_, PPd_ = form_PP_root((segm_, segd_), base_rdn=2)
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
        mrdn = derP.valt[1-fd] > derP.valt[fd]  # sum because they are valt
        derP.rdnt[fd] += not mrdn if fd else mrdn

        if derP.valt[fd] > vaves[fd] * derP.rdnt[fd] * (i+1):  # ave * rdn to stronger derPs in link_layers[-2]
            link_layers[-1][fd].append(derP)
            derP._P.downlink_layers[-1][fd] += [derP]
            # misses = link_layers[-2] not in link_layers[-1], sum as PP.nvalt[fd] in sum2seg and sum2PP
# ?
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
        seg_.append( sum2seg( seg_Ps, fd, fds))  # convert seg_Ps to CPP seg
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
                        if PPm not in PP.alt_PP_:
                            PP.alt_PP_ += [PPm]  # bilateral assignment of alt_PPs
                        if PP not in PPm.alt_PP_:
                            PPm.alt_PP_ += [PP]  # PPd here
        PP_t += [PP_]
    return PP_t

def form_PP_(PP_segs, link_, fup, fd):  # flood-fill PP_segs with vertically linked segments:

    # PP is a graph of 1D segs, with two sets of edges/branches: seg.uplink_, seg.downlink_.
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

    uplink_, uuplink_t  = seg_Ps[-1].uplink_layers[-2:]  # uplinks of top P? or this is a bottom?
    miss_uplink_ = [uuplink for uuplink in uuplink_t[fd] if uuplink not in uplink_]  # in layer-1 but not in layer-2

    downlink_, ddownlink_t = seg_Ps[0].downlink_layers[-2:]  # downlinks of bottom P, downlink.P.seg.uplinks= lower seg.uplinks
    miss_downlink_ = [ddownlink for ddownlink in ddownlink_t[fd] if ddownlink not in downlink_]
    # seg rdn: up ini cost, up+down comp_seg cost in 1st agg+? P rdn = up+down M/n?
    P = seg_Ps[0]  # top P in stack
    L = len(P.dert_) if isinstance(P,CP) else P.L
    seg = CPP(P__= seg_Ps, uplink_layers=[miss_uplink_], downlink_layers = [miss_downlink_], fds=copy(fds)+[fd],
              box=[P.y0, P.y0+len(seg_Ps), P.x0, P.x0+L-1])
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
    L = len(P.dert_) if isinstance(P,CP) else P.L
    sum_derH(derH, [P.ptuple] if isinstance(P,CP) else P.derQ)
    seg.box[2] = min(seg.box[2],P.x0); seg.box[3] = max(seg.box[3],P.x0+L-1)
    seg.derH = derH
    if derQ: # if fd:
        seg.derH += derQ  # der+
        # else: seg.derH[int(len(derH)/2):] = derQ  # rng+, replace last layer

    return seg

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


def sum_derH(DerH, derH, fneg=0):  # same fds from comp_derH

    for Vertuple, vertuple in zip_longest(DerH, derH, fillvalue=[]):
        if vertuple:
            if Vertuple:
                if isinstance(vertuple, CpQ):
                    sum_vertuple(Vertuple, vertuple, fneg)
                else:
                    sum_ptuple(Vertuple, vertuple, fneg)
            elif not fneg:
                DerH += [deepcopy(vertuple)]


def sum_vertuple(Vertuple, vertuple, fneg=0):

    for Part, part in zip(Vertuple.Q, vertuple.Q):
        # [mpar,dpar] each
        Part[0] += (-part[0] if fneg else part[0])
        Part[1] += (-part[1] if fneg else part[1])
    for i in 0,1:
        Vertuple.valt[i] += vertuple.valt[i]
        Vertuple.rdnt[i] += vertuple.rdnt[i]
    Vertuple.n += 1


def sum_ptuple(Ptuple, ptuple, fneg=0):

    for pname, ave in zip(pnames, aves):
        Par = getattr(Ptuple, pname); par = getattr(ptuple, pname)

        if pname in ("angle","axis") and isinstance(Par, list):
            sin_da0 = (Par[0] * par[1]) + (Par[1] * par[0])  # sin(A+B)= (sinA*cosB)+(cosA*sinB)
            cos_da0 = (Par[1] * par[1]) - (Par[0] * par[0])  # cos(A+B)=(cosA*cosB)-(sinA*sinB)
            Par = [sin_da0, cos_da0]
        elif pname == "aangle" and isinstance(Par, list):
            _sin_da0, _cos_da0, _sin_da1, _cos_da1 = Par
            sin_da0, cos_da0, sin_da1, cos_da1 = par
            sin_dda0 = (_sin_da0 * cos_da0) + (_cos_da0 * sin_da0)
            cos_dda0 = (_cos_da0 * cos_da0) - (_sin_da0 * sin_da0)
            sin_dda1 = (_sin_da1 * cos_da1) + (_cos_da1 * sin_da1)
            cos_dda1 = (_cos_da1 * cos_da1) - (_sin_da1 * sin_da1)
            Par = [sin_dda0, cos_dda0, sin_dda1, cos_dda1]
        else:
            Par += (-par if fneg else par)
        setattr(Ptuple, pname, Par)

    Ptuple.valt[0] += ptuple.valt[0]; Ptuple.valt[1] += ptuple.valt[1]

    Ptuple.n += 1

def comp_derH(_derH, derH):  # no need to check fds in comp_slice

    dderH = []; valt = [0,0]; rdnt = [1,1]
    for i, (_ptuple,ptuple) in enumerate(zip(_derH, derH)):

        dtuple = comp_vertuple(_ptuple,ptuple) if isinstance(_ptuple, CpQ) else comp_ptuple(_ptuple,ptuple)
        # dtuple = comp_vertuple(_ptuple,ptuple) if i else comp_ptuple(_ptuple,ptuple)
        dderH += [dtuple]
        for j in 0,1:
            valt[j] += dtuple.valt[j]; rdnt[j] += dtuple.rdnt[j]

    return dderH, valt, rdnt

def comp_vertuple(_vertuple, vertuple):

    dtuple=CpQ(n=_vertuple.n)
    rn = _vertuple.n/vertuple.n  # normalize param as param*rn for n-invariant ratio: _param/ param*rn = (_param/_n)/(param/n)

    for _par, par, ave in zip(_vertuple.Q, vertuple.Q, aves):
        m,d = comp_par(_par[1], par[1]*rn, ave)
        dtuple.Q += [[m,d]]
        dtuple.valt[0]+=m; dtuple.valt[1]+=d

    return dtuple

def comp_ptuple(_ptuple, ptuple):

    dtuple = CpQ(n=_ptuple.n)
    rn = _ptuple.n / ptuple.n  # normalize param as param*rn for n-invariant ratio: _param / param*rn = (_param/_n) / (param/n)

    for pname, ave in zip(pnames, aves):  # comp full derH of each param between ptuples:
        _par = getattr(_ptuple, pname); par = getattr(ptuple, pname)
        if pname=="aangle": m,d = comp_aangle(_par, par)
        elif pname in ("axis","angle"): m,d = comp_angle(_par, par)
        else:
            if pname!="x": par*=rn  # normalize by relative accum count
            if pname=="x" or pname=="I": finv = 1
            else: finv=0
            m,d = comp_par(_par, par, ave, finv)
        dtuple.Q += [[m,d]]
        dtuple.valt[0] += m; dtuple.valt[1] += d

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