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
aveB = 50  # same for m and d?
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


class CQ(ClusterStructure):  # generic sequence or hierarchy

    Q = list  # generic sequence or index increments in ptuple, derH, etc
    Qm = list  # in-graph only
    Qd = list
    ext = lambda: [[],[]]  # [ms,ds], per subH only
    valt = lambda: [0,0]  # in-graph vals
    rdnt = lambda: [1,1]  # none if represented m and d?
    out_valt = lambda: [0,0]  # of non-graph links, as alt?
    fds = list
    rng = lambda: 1  # is it used anywhere?
    n = lambda: 1  # accum count, not needed?

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
    n = lambda: 1  # not needed?


class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

    ptuple = lambda: Cptuple()  # latuple: I, M, Ma, G, Ga, angle(Dy, Dx), aangle( Sin_da0, Cos_da0, Sin_da1, Cos_da1), ?[n, val, x, L, A]?
    derH = lambda: [[[],[]]]  # 1vertuple / 1layer in comp_slice, extend in der+
    fds = list  # per derLay
    valt = lambda: [0,0]  # of fork links, represented in derH
    rdnt = lambda: [1,1]
    # n = lambda: 1
    x0 = int
    y0 = int  # for vertical gap in PP.P__
    dert_ = list  # array of pixel-level derts, redundant to uplink_, only per blob?
    link_ = list  # all links
    link_t = lambda: [[],[]]  # +ve rlink_, dlink_
    roott = lambda: [None,None]  # m,d PP that contain this P
    dxdert_ = list  # only in Pd
    Pd_ = list  # only in Pm
    # if comp_dx:
    Mdx = int
    Ddx = int

class CderP(ClusterStructure):  # tuple of derivatives in P link: binary tree with latuple root and vertuple forks

    derH = list  # vertuple_ per layer, unless implicit? sum links / rng+, layers / der+?
    fds = list
    valt = lambda: [0,0]  # also of derH
    rdnt = lambda: [1,1]  # mrdn + uprdn if branch overlap?
    _P = object  # higher comparand
    P = object  # lower comparand
    roott = lambda: [None,None]  # for der++
    x0 = int
    y0 = int
    L = int
    fdx = NoneType  # if comp_dx
'''
max ntuples / der layer = ntuples in lower layers: 1, 1, 2, 4, 8...
lay1: par     # derH per param in vertuple, each layer is derivatives of all lower layers:
lay2: [m,d]   # implicit nesting, brackets for clarity:
lay3: [[m,d], [md,dd]]: 2 sLays,
lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays:
'''

class CPP(CderP):

    ptuple = lambda: Cptuple()  # summed P__ ptuples, = 0th derLay
    P__ = list  # 2D array of nodes, may be sub-PPs
    derH = lambda: [[[],[]]]  # 1vertuple / 1layer in comp_slice, extend in der+
    fds = list  # fd per derLay
    valt = lambda: [0,0]  # summed fork links vals
    rdnt = lambda: [1,1]  # recursion count + Rdn / nderPs + mrdn + uprdn if branch overlap?
    Rdn = int  # for accumulation only?
    rng = lambda: 1
    nval = int  # of links to alt PPs?
    alt_rdn = int  # overlapping redundancy between core and edge
    alt_PP_ = list  # adjacent alt-fork PPs per PP, from P.roott[1] in sum2PP
    altuple = list  # summed from alt_PP_, sub comp support, agg comp suppression?
    box = lambda: [0,0,0,0]  # y0,yn, x0,xn
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    mask__ = bool
    link_ = list  # all links summed from Ps
    link_t = lambda: [[],[]]  # +ve rlink_, dlink_
    roott = lambda: [None,None]  # PPPm, PPPd that contain this PP
    cPP_ = list  # rdn reps in other PPPs, to eval and remove


def comp_slice_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    from sub_recursion import sub_recursion_eval, rotate_P_, agg_recursion_eval

    P__ = slice_blob(blob, verbose=verbose)  # form 2D array of Ps: horizontal blob slices in dert__
    rotate_P_(P__, blob.dert__, blob.mask__)  # reform Ps around centers along G, sides may overlap
    # scan rows top-down, comp y-adjacent, x-overlapping Ps to form derPs:
    _P_ = P__[0]  # higher row
    for P_ in P__[1:]:  # lower row
        for P in P_:
            Mtuple, Dtuple = [], []
            links = [P.link_, P.link_t[0], P.link_t[1]]  # for new P
            Valt = [0, 0]; Rdnt = [0, 0]
            for _P in _P_:  # test for x overlap(_P,P) in 8 directions, derts are positive in all Ps:
                _L = len(_P.dert_); L = len(P.dert_)
                if (P.x0 - 1 < _P.x0 + _L) and (P.x0 + L > _P.x0):
                    comp_P(_P,P, Mtuple,Dtuple, links, Valt, Rdnt, fd=0)
                elif (P.x0 + L) < _P.x0:
                    P.derH=[[Mtuple,Dtuple]]  # single vertuple
                    break  # no xn overlap, stop scanning lower P_
        _P_ = P_
    PPm_,PPd_ = form_PP_t(P__, base_rdn=2)
    blob.PPm_, blob.PPd_  = PPm_, PPd_
    # re comp, cluster:
    for i, PP_ in enumerate([PPm_, PPd_]):  # derH, fds per PP
        if sum([PP.valt[i] for PP in PP_]) > ave * sum([PP.rdnt[i] for PP in PP_]):
            sub_recursion_eval(blob, PP_, fd=i)  # intra PP
            # agg_recursion_eval(blob, copy(PP_))  # cross PP, Cgraph conversion doesn't replace PPs?


def slice_blob(blob, verbose=False):  # form blob slices nearest to slice Ga: Ps, ~1D Ps, in select smooth edge (high G, low Ga) blobs

    mask__ = blob.mask__  # same as positive sign here
    dert__ = zip(*blob.dert__)  # convert 10-tuple of 2D arrays into 1D array of 10-tuple blob rows
    dert__ = [zip(*dert_) for dert_ in dert__]  # convert 1D array of 10-tuple rows into 2D array of 10-tuples per blob
    P__ = []
    height, width = mask__.shape
    if verbose: print("Converting to image...")

    for y, (dert_, mask_) in enumerate(zip(dert__, mask__)):  # unpack lines, each may have multiple slices -> Ps:
        P_ = []
        _mask = True  # mask the cell before 1st dert
        for x, (dert, mask) in enumerate(zip(dert_, mask_)):
            if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()

            g, ga, ri, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = dert[1:]  # skip i
            if not mask:  # masks: if 0,_1: P initialization, if 0,_0: P accumulation, if 1,_0: P termination
                if _mask:  # ini P params with first unmasked dert
                    Pdert_ = [dert]
                    params = Cptuple(M=ave_g-g,Ma=ave_ga-ga,I=ri, angle=[dy,dx], aangle=[sin_da0, cos_da0, sin_da1, cos_da1])
                else:
                    # dert and _dert are not masked, accumulate P params:
                    params.M+=ave_g-g; params.Ma+=ave_ga-ga; params.I+=ri; params.angle[0]+=dy; params.angle[1]+=dx
                    params.aangle = [_par+par for _par,par in zip(params.aangle,[sin_da0,cos_da0,sin_da1,cos_da1])]
                    Pdert_ += [dert]
            elif not _mask:
                # _dert is not masked, dert is masked, terminate P:
                params.G = np.hypot(*params.angle)  # Dy,Dx  # recompute G,Ga, it can't reconstruct M,Ma
                params.Ga = (params.aangle[1]+1) + (params.aangle[3]+1)  # Cos_da0, Cos_da1
                L = len(Pdert_)
                params.L = L; params.x = x-L/2  # params.valt = [params.M+params.Ma, params.G+params.Ga]
                P_+=[CP(ptuple=params, x0=x-(L-1), y0=y, dert_=Pdert_)]
            _mask = mask
        # pack last P, same as above:
        if not _mask:
            params.G = np.hypot(*params.angle); params.Ga = (params.aangle[1]+1) + (params.aangle[3]+1)
            L = len(Pdert_); params.L = L; params.x = x-L/2  # params.valt=[params.M+params.Ma,params.G+params.Ga]
            P_ += [CP(ptuple=params, x0=x-(L-1), y0=y, dert_=Pdert_)]
        P__ += [P_]

    blob.P__ = P__
    return P__

def form_PP_t(P__, base_rdn):  # form PPs of derP.valt[fd] + connected Ps'val

    PP_t = []
    for fd in 0, 1:
        fork_P__ = ([copy(P_) for P_ in reversed(P__)])  # scan bottom-up
        PP_ = []; packed_P_ = []  # form initial sequence-PPs:
        for P_ in fork_P__:
            for P in P_:
                if P not in packed_P_:
                    qPP = [[[P]], P.valt[fd]]  # init PP is 2D queue of node Ps and sum([P.val+P.link_val])
                    uplink_ = P.link_t[fd]; uuplink_ = []  # next-line links for recursive search
                    while uplink_:
                        for derP in uplink_:
                            if derP._P not in packed_P_:
                                qPP[0].insert(0, [derP._P])  # pack top down
                                qPP[1] += derP.valt[fd]
                                packed_P_ += [derP._P]
                                uuplink_ += derP._P.link_t[fd]
                        uplink_ = uuplink_
                        uuplink_ = []
                    PP_ += [qPP + [ave+1]]  # + [ini reval]
        # prune qPPs by med links val:
        rePP_= reval_PP_(PP_, fd)  # PP = [qPP,val,reval]
        CPP_ = [sum2PP(PP, base_rdn, fd) for PP in rePP_]
        PP_t += [CPP_]  # may be empty

    return PP_t  # add_alt_PPs_(graph_t)?

def reval_PP_(PP_, fd):  # recursive eval / prune Ps for rePP

    rePP_ = []
    while PP_:  # init P__
        P__, val, reval = PP_.pop(0)
        if val > ave:
            if reval < ave:  # same graph, skip re-evaluation:
                rePP_ += [[P__,val,0]]  # reval=0
            else:
                rePP = reval_P_(P__, fd)  # recursive node and link revaluation by med val
                if rePP[1] > ave:  # min adjusted val
                    rePP_ += [rePP]
    if rePP_ and max([rePP[2] for rePP in rePP_]) > ave:  # recursion if any min reval:
        rePP_ = reval_PP_(rePP_, fd)

    return rePP_

def reval_P_(P__, fd):  # prune qPP by (link_ + mediated link__) val

    prune_ = []; Val, reval = 0,0  # comb PP value and recursion value

    for P_ in P__:
        for P in P_:
            P_val = 0; remove_ = []
            for link in P.link_t[fd]:
                # recursive mediated link layers eval-> med_valH:
                _,_,med_valH = med_eval(link._P.link_t[fd], old_link_=[], med_valH=[], fd=fd)
                # link val + mlinks val: single med order, no med_valH in comp_slice?:
                link_val = link.valt[fd] + sum([mlink.valt[fd] for mlink in link._P.link_t[fd]]) * med_decay  # + med_valH
                if link_val < vaves[fd]:
                    remove_+= [link]; reval += link_val
                else: P_val += link_val
            for link in remove_:
                P.link_t[fd].remove(link)  # prune weak links
            if P_val < vaves[fd]:
                prune_ += [P]
            else:
                Val += P_val
    for P in prune_:
        for link in P.link_t[fd]:  # prune direct links only?
            _P = link._P
            _link_ = _P.link_t[fd]
            if link in _link_:
                _link_.remove(link); reval += link.valt[fd]

    if reval > aveB:
        P__, Val, reval = reval_P_(P__, fd)  # recursion
    return [P__, Val, reval]

def med_eval(last_link_, old_link_, med_valH, fd):  # compute med_valH

    curr_link_ = []; med_val = 0

    for llink in last_link_:
        for _link in llink._P.link_t[fd]:
            if _link not in old_link_:  # not-circular link
                old_link_ += [_link]  # evaluated mediated links
                curr_link_ += [_link]  # current link layer,-> last_link_ in recursion
                med_val += _link.valt[fd]
    med_val *= med_decay ** (len(med_valH) + 1)
    med_valH += [med_val]
    if med_val > aveB:
        # last med layer val-> likely next med layer val
        curr_link_, old_link_, med_valH = med_eval(curr_link_, old_link_, med_valH, fd)  # eval next med layer

    return curr_link_, old_link_, med_valH


def sum2PP(qPP, base_rdn, fd):  # sum Ps and links into PP

    P__, val, _ = qPP  # proto-PP is a list
    # init:
    P0 = P__[0][0]
    PP = CPP(box=[P0.y0,P__[-1][0].y0,P0.x0,P0.x0+len(P0.dert_)], fds=[fd], P__ = P__)
    PP.valt[fd] = val; PP.rdnt[fd] += base_rdn
    # accum:
    for P_ in P__:  # top-down
        for P in P_:  # left-to-right
            P.roott[fd] = PP
            sum_ptuple(PP.ptuple, P.ptuple)
            # already done in comp_P:
            # for derP in P.link_t[fd]:  # uplinks only: no sum in _P, 1-vertuple layers, 1 in comp_slice, 2 in der+:
            #    [sum_vertuple(L[0],l[0]) for L,l in zip((P.derH, derP.derH))]  # loop 1|2 layers
            [sum_vertuple(L[0],l[0]) for L,l in zip((PP.derH, P.derH))]
            PP.link_ += P.link_
            for Link_,link_ in zip(PP.link_t, P.link_t):
                Link_ += link_  # all unique links in PP, to replace n
            PP.box[0] = min(PP.box[0], P.y0)  # y0
            PP.box[2] = min(PP.box[2], P.x0)  # x0
            PP.box[3] = max(PP.box[3], P.x0 + len(P.dert_))  # xn

    return PP


def sum_derH(DerH, derH):  # derH is layers, not selective here. Sum both forks in each node to comp alt fork

    for Layer, layer in zip_longest(DerH, derH, fillvalue=None):
        if Layer:
            if layer:
                for Vertuple, vertuple in zip_longest(Layer, layer, fillvalue=None):
                    if Vertuple:
                        if vertuple:
                            sum_vertuple(Vertuple, vertuple)
                    else:
                        Layer += [deepcopy(vertuple)]
        else:
            DerH += [deepcopy(layer)]


def sum_vertuple(Vertuple, vertuple, fneg=0):  # [mtuple,dtuple]

    # one of the tuples may be empty?
    [sum_tuple(Tuple,tuple,fneg) for Tuple,tuple in zip(Vertuple, vertuple)]

def sum_tuple(Tuple,tuple, fneg=0):  # mtuple or dtuple

    for i, (Par, par) in enumerate(zip_longest(Tuple, tuple, fillvalue=None)):
        if par:
            if Par:
                Tuple[i] = Par + -par if fneg else par
            elif not fneg:
                Tuple += [par]

def sum_ptuple(Ptuple, ptuple, fneg=0):

    for pname, ave in zip(pnames, aves):
        Par = getattr(Ptuple, pname); par = getattr(ptuple, pname)

        if isinstance(Par, list):  # angle or aangle
            for j, (P,p) in enumerate(zip(Par,par)): Par[j] = P-p if fneg else P+p
        else:
            Par += (-par if fneg else par)
        setattr(Ptuple, pname, Par)

    Ptuple.n += 1

# draft:
def comp_P(_P,P, Mtuple, Dtuple, links, Valt, Rdnt, fd, valt=None):

    if fd:
        _tuple, tuple = P.derH[-1][0][1], _P.derH[-1][0][1]  # 1-vertuple derH before feedback
        comp = comp_vertuple  # compare ds only
    else:
        _tuple, tuple = _P.ptuple, P.ptuple
        comp = comp_ptuple
    mtuple,dtuple = comp(_tuple, tuple)
    mval = sum(mtuple); if fd: mval += valt[0]
    dval = sum(dtuple); if fd: dval += valt[1]
    mrdn = 1+dval>mval; drdn = 1+(not mrdn)

    derP = CderP(derH=[[[mtuple,dtuple]]],fds=P.fds+[fd], valt=[mval,dval],rdnt=[mrdn,drdn], P=P,_P=_P, x0=_P.x0,y0=_P.y0,L=len(_P.dert_))
    links[0] += [derP]  # all links
    if mval > aveB*mrdn:
        links[1]+=[derP]; Valt[0]+=mval; Rdnt[0]+=mrdn  # +ve links, fork selection in form_PP_t
        sum_tuple(Mtuple, mtuple)
    if dval > aveB*drdn:
        links[2]+=[derP]; Valt[1]+=dval; Rdnt[1]+=drdn
        sum_tuple(Dtuple, dtuple)


def comp_vertuple(_vertuple, vertuple, rn):

    mtuple, dtuple = [],[]
    for _par, par, ave in zip(_vertuple[1], vertuple[1], aves):  # compare ds only?

        m,d = comp_par(_par, par*rn, ave)
        dtuple+=[m]; dtuple+=[d]

    return [mtuple, dtuple]

def comp_ptuple(_ptuple, ptuple):

    mtuple, dtuple = [],[]  # in the order of ("I", "M", "Ma", "axis", "angle", "aangle","G", "Ga", "x", "L")
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
        mtuple += [m]
        dtuple += [d]

    return [mtuple, dtuple]

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