'''
Vectorize is a terminal fork of intra_blob.

comp_slice traces edge axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These are low-M high-Ma blobs, vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)
-
Vectorization is clustering of parameterized Ps + their derivatives (derPs) into PPs: patterns of Ps that describe edge blob.
This process is a reduced-dimensionality (2D->1D) version of cross-comp and clustering cycle, common across this project.

PP clustering in vertical (along axis) dimension is contiguous and exclusive because 
Ps are relatively lateral (cross-axis), their internal match doesn't project vertically. 

Primary clustering by match between Ps over incremental distance (rng++), followed by forming overlapping
Secondary clusters of match of incremental-derivation (der++) difference between Ps. 

As we add higher dimensions (3D and time), this dimensionality reduction is done in salient high-aspect blobs
(likely edges in 2D or surfaces in 3D) to form more compressed 'skeletal' representations of full-dimensional patterns.

comp_slice traces edge blob axis by cross-comparing vertically adjacent Ps: slices across edge blob, along P.G angle.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)

Connectivity in P_ is traced through root_s of derts adjacent to P.dert_, possibly forking. 
len prior root_ sorted by G is root.rdn, to eval for inclusion in PP or start new P by ave*rdn
'''

  # root function:
import numpy as np
from collections import deque, defaultdict
from copy import deepcopy, copy
from itertools import zip_longest, combinations
from math import inf
from class_cluster import CBase, init_param as z
from .filters import ave, ave_dI, aves, P_aves, PP_aves
from .slice_edge import comp_angle
# from typing import List, Tuple
# from .classes import add_, comp_, negate, get_match, Clink, CG, CH
# from utils import box2slice, accum_box, sub_box2box
'''
Vectorize is a terminal fork of intra_blob.

comp_slice traces edge axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These are low-M high-Ma blobs, vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)
-
Vectorization is clustering of parameterized Ps + their derivatives (derPs) into PPs: patterns of Ps that describe edge blob.
This process is a reduced-dimensionality (2D->1D) version of cross-comp and clustering cycle, common across this project.

PP clustering in vertical (along axis) dimension is contiguous and exclusive because 
Ps are relatively lateral (cross-axis), their internal match doesn't project vertically. 

Primary clustering by match between Ps over incremental distance (rng++), followed by forming overlapping
Secondary clusters of match of incremental-derivation (der++) difference between Ps. 

As we add higher dimensions (3D and time), this dimensionality reduction is done in salient high-aspect blobs
(likely edges in 2D or surfaces in 3D) to form more compressed 'skeletal' representations of full-dimensional patterns.

comp_slice traces edge blob axis by cross-comparing vertically adjacent Ps: slices across edge blob, along P.G angle.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)

Connectivity in P_ is traced through root_s of derts adjacent to P.dert_, possibly forking. 
len prior root_ sorted by G is root.rdn, to eval for inclusion in PP or start new P by ave*rdn
'''

  # root function:
def ider_recursion(root, PP, fd=0):  # node-mediated correlation clustering: keep same Ps and links, increment link derH, then P derH in sum2PP

    if fd:  # add prelinks per P if not initial call:
        for P in PP.P_: P.link_ += [copy(unpack_last_link_(P.link_))]
    rng_recursion(PP, rng=1, fd=fd)  # extend PP.link_, derHs by same-der rng+ comp

    form_PP_t(PP, PP.P_, iRt = PP.iderH.Et[2:4] if PP.iderH else [0,0])  # der+ is mediated by form_PP_t
    if root is not None: root.fback_ += [PP.iderH]  # feedback from PPds


def rng_recursion(PP, rng=1, fd=0):  # similar to agg+ rng_recursion, but contiguously link mediated, because

    iP_ = PP.P_
    while True:
        P_ = []; V = 0
        for P in iP_:
            if not P.link_: continue
            prelink_ = []  # new prelinks per P
            _prelink_ = P.link_.pop()  # old prelinks per P
            for _link in _prelink_:
                _P = _link._node if fd else _link
                dy,dx = np.subtract(_P.yx,P.yx)
                distance = np.hypot(dy,dx)  # distance between P midpoints, /= L for eval?
                if distance < rng:  # | rng * ((P.val+_P.val)/ ave_rval)?
                    if fd and not (_P.derH and P.derH): continue  # nothing to compare
                    mlink = comp_P(_link if fd else [_P,P, distance,[dy,dx]])
                    if mlink:  # return if match
                        V += mlink.dderH.Et[0]
                        if rng > 1:  # test to add nesting to P.link_:
                            if rng == 2 and not isinstance(P.link_[0], list): P.link_[:] = [P.link_[:]]  # link_ -> link_H
                            if len(P.link_) < rng: P.link_ += [[]]  # add new link_
                        link_ = unpack_last_link_(P.link_)
                        if not fd: link_ += [mlink]
                        _link_ = unpack_last_link_(_P.link_[:-1])  # skip prelink_
                        prelink_ += [link._node if link.node is _P else link.node for link in _link_]  # connected __Ps
            P.link_ += [prelink_]  # temporary pre-links, maybe empty
            if prelink_: P_ += [P]
        rng += 1
        if V > ave * len(P_) * 6:  #  implied val of all __P_s, 6: len mtuple
            iP_ = P_
        else:
            for P in PP.P_: P.link_.pop()
            break
    PP.rng=rng
    '''
    der++ is tested in PPds formed by rng++, no der++ inside rng++: high diff @ rng++ termination only?
    '''

def comp_P(link):

    if isinstance(link, Clink):  # der+ only
        _P,P = link._node, link.node
        rn = (_P.derH.n if P.derH else len(_P.dert_)) / P.derH.n  # lower P must have derH
        aveP = P_aves[1]
    else:  # rng+
        _P,P, S,A = link
        rn = len(_P.dert_) / len(P.dert_)
        H = comp_latuple(_P.latuple, P.latuple, rn)
        vm = sum(H[::2]); vd = sum(abs(d) for d in H[1::2])
        rm = 1 + vd > vm; rd = 1 + vm >= vd
        n = (len(_P.dert_)+len(P.dert_)) / 2  # der value = ave compared n?
        aveP = P_aves[0]
        link = Clink(node=P,_node=_P, dderH = CH(nest=0,Et=[vm,vd,rm,rd],H=H,n=n), S=S, A=A, roott=[[],[]])
    # both:
    if _P.derH and P.derH:  # append link dderH, init in form_PP_t rng++, comp_latuple was already done
        # der+:
        dderH = comp_(_P.derH, P.derH, rn)
        vm,vd,rm,rd = dderH.Et[:4]  # also works if called from comp_G
        rm += vd > vm; rd += vm >= vd
        aveP = P_aves[1]
        He = link.dderH  # append link dderH:
        if not He.nest: He = link.He = CH(nest=1, Et=[*He.Et], H=[He])  # nest md_ as derH
        He.Et = np.add(He.Et, [vm,vd,rm,rd])
        He.H += [dderH]

    if vm > aveP * rm:  # always rng+
        return link


def form_PP_t(root, P_, iRt):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    for fd in 0,1:
        P_Ps = []; Link_ = []
        for P in P_:  # not PP.link_: P uplinks are unique, only G links overlap
            Ps, link_ = [], []
            for derP in unpack_last_link_(P.link_):
                Ps += [derP._node]; link_ += [derP]  # not needed for PPs?
            P_Ps += [Ps]; Link_ += [link_]  # aligned with P_
        inP_ = []  # clustered Ps and their val,rdn s for all Ps
        for P in root.P_:
            if P in inP_: continue  # already packed in some PP
            cP_, clink_ = [P], []  # clustered Ps and their val,rdn s
            if P in P_:
                P_index = P_.index(P)
                clink_ += Link_[P_index]
                perimeter = deque(P_Ps[P_index])  # recycle with breadth-first search, up and down:
                while perimeter:
                    _P = perimeter.popleft()
                    if _P in cP_ or _P not in P_: continue  # _P in P_ should be checked here? If _P not in P_, there's no need to add it into CP too
                    cP_ += [_P]
                    clink_ += Link_[P_.index(_P)]
                    perimeter += P_Ps[P_.index(_P)] # append linked __Ps to extended perimeter of P
            PP = sum2PP(root, cP_, clink_, iRt, fd)
            PP_t[fd] += [PP]  # no if Val > PP_aves[fd] * Rdn:
            inP_ += cP_  # update clustered Ps

    for PP in PP_t[1]:  # eval der+ / PPd only, after form_PP_t -> P.root
        if PP.iderH and PP.iderH.Et[0] * len(PP.link_) > PP_aves[1] * PP.iderH.Et[2]:
            # node-mediated correlation clustering:
            ider_recursion(root, PP, fd=1)
        if root.fback_:
            feedback(root)  # after der+ in all nodes, no single node feedback

    root.node_ = PP_t  # nested in der+, add_alt_PPs_?


def form_PP_t_old(root, P_, iRt):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    for fd in 0,1:
        P_Ps = []; Link_ = []
        for P in P_:  # not PP.link_: P uplinks are unique, only G links overlap
            Ps = []
            for derP in unpack_last_link_(P.link_):
                Ps += [derP._node]; Link_ += [derP]  # not needed for PPs?
            P_Ps += [Ps]  # aligned with P_
        inP_ = []  # clustered Ps and their val,rdn s for all Ps
        for P in root.P_:
            if P in inP_: continue  # already packed in some PP
            cP_ = [P]  # clustered Ps and their val,rdn s
            if P in P_:
                perimeter = deque(P_Ps[P_.index(P)])  # recycle with breadth-first search, up and down:
                while perimeter:
                    _P = perimeter.popleft()
                    if _P in cP_: continue
                    cP_ += [_P]
                    if _P in P_:
                        perimeter += P_Ps[P_.index(_P)] # append linked __Ps to extended perimeter of P
            PP = sum2PP(root, cP_, Link_, iRt, fd)
            PP_t[fd] += [PP]  # no if Val > PP_aves[fd] * Rdn:
            inP_ += cP_  # update clustered Ps

    for PP in PP_t[1]:  # eval der+ / PPd only, after form_PP_t -> P.root
        if PP.iderH and PP.iderH.Et[0] * len(PP.link_) > PP_aves[1] * PP.iderH.Et[2]:
            # node-mediated correlation clustering:
            ider_recursion(root, PP, fd=1)
        if root.fback_:
            feedback(root)  # after der+ in all nodes, no single node feedback

    root.node_ = PP_t  # nested in der+, add_alt_PPs_?


def sum2PP(root, P_, derP_, iRt, fd):  # sum links in Ps and Ps in PP

    PP = CG(fd=fd,root=root,P_=P_,rng=root.rng+1, link_=[], box=[inf,inf,-inf,-inf], latuple=[0,0,0,0,0,[0,0]])
    # += uplinks:
    for derP in derP_:
        if derP.node not in P_ or derP._node not in P_: continue
        if derP.dderH:
            add_([], derP.node.derH, derP.dderH, iRt)
            add_([], derP._node.derH, negate(deepcopy(derP.dderH)), iRt)  # to reverse uplink direction
        PP.link_ += [derP]; derP.roott[fd] = PP
        PP.A = np.add(PP.A,derP.A); PP.S += derP.S
        PP.n += derP.dderH.n  # *= ave compared P.L?
    # += Ps:
    celly_,cellx_ = [],[]
    for P in P_:
        L = P.latuple[-2]
        PP.area += L; PP.n += L  # no + P.derH.n: current links only?
        PP.latuple = [P+p for P,p in zip(PP.latuple[:-1],P.latuple[:-1])] + [[A+a for A,a in zip(PP.latuple[-1],P.latuple[-1])]]
        if P.derH:
            add_([], PP.iderH, P.derH)
        for y,x in P.yx_:
            y = int(round(y)); x = int(round(x))  # summed with float dy,dx in slice_edge?
            PP.box = accum_box(PP.box, y, x); celly_+=[y]; cellx_+=[x]
    if PP.iderH:
        PP.iderH.Et[2:4] = [R+r for R,r in zip(PP.iderH.Et[2:4], iRt)]
    # pixmap:
    y0,x0,yn,xn = PP.box
    PP.mask__ = np.zeros((yn-y0, xn-x0), bool)
    celly_ = np.array(celly_); cellx_ = np.array(cellx_)
    PP.mask__[(celly_-y0, cellx_-x0)] = True

    return PP

def feedback(root):  # in form_PP_, append new der layers to root PP, single vs. root_ per fork in agg+

    HE = deepcopy(root.fback_.pop(0))
    while root.fback_:
        He  = root.fback_.pop(0)
        add_([], HE, He)
    add_([], root.iderH, HE.H[-1] if HE.nest else HE)  # last md_ in H or sum md_

    if root.root and isinstance(root.root, CG):  # skip if root is Edge
        rroot = root.root  # single PP.root, can't be P
        fback_ = rroot.fback_
        node_ = rroot.node_[1] if rroot.node_ and isinstance(rroot.node_[0],list) else rroot.P_  # node_ is updated to node_t in sub+
        fback_ += [HE]
        if fback_ and (len(fback_)==len(node_)):  # all nodes terminated and fed back
            feedback(rroot)  # sum2PP adds derH per rng, feedback adds deeper sub+ layers


def comp_latuple(_latuple, latuple, rn, fagg=0):  # 0der params

    _I, _G, _M, _Ma, _L, (_Dy, _Dx) = _latuple
    I, G, M, Ma, L, (Dy, Dx) = latuple

    dI = _I - I*rn;  mI = ave_dI - dI
    dG = _G - G*rn;  mG = min(_G, G*rn) - aves[1]
    dL = _L - L*rn;  mL = min(_L, L*rn) - aves[2]
    dM = _M - M*rn;  mM = get_match(_M, M*rn) - aves[3]  # M, Ma may be negative
    dMa= _Ma- Ma*rn; mMa = get_match(_Ma, Ma*rn) - aves[4]
    mAngle, dAngle = comp_angle((_Dy,_Dx), (Dy,Dx))

    ret = [mL,dL,mI,dI,mG,dG,mM,dM,mMa,dMa,mAngle-aves[5],dAngle]

    if fagg:  # add norm m,d: ret = [ret, Ret]
        # max possible m,d per compared param
        Ret = [max(_L,L),abs(_L)+abs(L), max(_I,I),abs(_I)+abs(I), max(_G,G),abs(_G)+abs(G), max(_M,M),abs(_M)+abs(M), max(_Ma,Ma),abs(_Ma)+abs(Ma), 1,.5]
        mval, dval = sum(ret[::2]),sum(ret[1::2])
        mrdn, drdn = dval>mval, mval>dval
        mdec, ddec = 0, 0
        for fd, (ptuple,Ptuple) in enumerate(zip((ret[::2],ret[1::2]),(Ret[::2],Ret[1::2]))):
            for i, (par, maxv, ave) in enumerate(zip(ptuple, Ptuple, aves)):  # compute link decay coef: par/ max(self/same)
                if fd: ddec += abs(par)/ abs(maxv) if maxv else 1
                else:  mdec += (par+ave)/ (maxv+ave) if maxv else 1

        ret = [mval, dval, mrdn, drdn, mdec, ddec], ret
    return ret

#draft
def append_(HE,He, fmerge=0):

    if fmerge:
        HE.H += He.H; HE.nest = He.nest
    else:
        HE.H += [He]; HE.nest = max(1, He.nest)

    HE.Et[:] = [V+v for V,v in zip(HE.Et, He.Et)]
    HE.n += He.n


def add_(HE_root, HE, He, irdnt=[]):  # unpack tuples (formally lists) down to numericals and sum them

    if He:  # to be summed
        if HE:  # to sum in
            ddepth = abs(HE.nest - He.nest)  # compare nesting depth, nest lesser He: md_-> derH-> subH-> aggH:
            if ddepth:
                nHe = [HE,He][HE.nest>He.nest]  # He to be nested
                while ddepth > 0:
                   nHe.nest += 1; nHe.H = [nHe.H]; ddepth -= 1

            if isinstance(HE.H[0],CH):  # no and isinstance(lay.H[0],list): same nesting unless cpr?
                for Lay,lay in zip_longest(HE.H, He.H, fillvalue=[]):
                    add_(HE, Lay,lay, irdnt)  # recursive unpack
            else:
                HE.H = np.add(HE.H, He.H)  # both have numericals in H
                Et, et = HE.Et, He.Et  # always numerical
                Et[:] = [E+e for E,e in zip(Et, et)]
                if irdnt: Et[2:4] = [E+e for E,e in zip(Et[2:4], irdnt)]
                HE.n += He.n  # combined param accumulation span
        else:
            if isinstance(HE, CH): HE.copy(He)
            elif isinstance(HE_root, CH): append_(HE_root, He)

    return HE  # to sum

def comp_(_He,He, rn=1, fagg=0):  # unpack tuples (formally lists) down to numericals and compare them

    ddepth = abs(_He.nest - He.nest)
    n = 0
    if ddepth:  # unpack the deeper He: md_<-derH <-subH <-aggH:
        uHe = [He,_He][_He.nest>He.nest]
        while ddepth > 0:
            uHe = uHe.H[0]; ddepth -= 1  # comp 1st layer of deeper He:
        _cHe,cHe = [uHe,He] if _He.nest>He.nest else [_He,uHe]
    else: _cHe,cHe = _He,He

    if isinstance(_cHe.H[0], CH):  # _lay is He_, same for lay: they are aligned above
        Et = [0,0,0,0,0,0]  # Vm,Vd, Rm,Rd, Dm,Dd
        dH = []
        for _lay,lay in zip(_cHe.H,cHe.H):  # md_| ext| derH| subH| aggH, eval nesting, unpack,comp ds in shared lower layers:
            if _lay and lay:  # ext is empty in single-node Gs
                dlay = comp_(_lay,lay, rn, fagg)
                Et[:] = [E+e for E,e in zip(Et,dlay.Et)]
                n += dlay.n
                dH += [dlay]  # CH
            else:
                dH += [[]]
    else:  # H is md_, numerical comp:
        vm,vd,rm,rd, decm,decd = 0,0,0,0, 0,0
        dH = []
        for i, (_d,d) in enumerate(zip(_cHe.H[1::2], cHe.H[1::2])):  # compare ds in md_ or ext
            d *= rn  # normalize by comparand accum span
            diff = _d-d
            match = min(abs(_d),abs(d))
            if (_d<0) != (d<0): match = -match  # if only one comparand is negative
            if fagg:
                maxm = max(abs(_d), abs(d))
                decm += abs(match) / maxm if maxm else 1  # match / max possible match
                maxd = abs(_d) + abs(d)
                decd += abs(diff) / maxd if maxd else 1  # diff / max possible diff
            vm += match - aves[i]  # fixed param set?
            vd += diff
            dH += [match,diff]  # flat
        Et = [vm,vd,rm,rd]
        if fagg: Et += [decm, decd]

        n = (_He.n+He.n) /2 * (len(_cHe.H)/12)  # ave compared n, /2 if ext: 6 params vs 12 in md_

    return CH(nest=min(_He.nest,He.nest), Et=Et, H=dH, n=n)


class CH(CBase):  # generic derivation hierarchy of variable nesting

    nest: int = 0  # nesting depth: -1/ ext, 0/ md_, 1/ derH, 2/ subH, 3/ aggH
    Et: list = z([])  # evaluation tuple: valt, rdnt, normt
    H: list = z([])  # hierarchy of der layers or md_
    n: int = 0  # total number of params compared to form derH, summed in comp_G and then from nodes in sum2graph

    def __bool__(self):  # to test empty
        if self.n: return True
        else: return False
    '''
    len layer +extt: 2, 3, 6, 12, 24,
    or without extt: 1, 1, 2, 4, 8..: max n of tuples per der layer = summed n of tuples in all lower layers:
    lay1: par     # derH per param in vertuple, layer is derivatives of all lower layers:
    lay2: [m,d]   # implicit nesting, brackets for clarity:
    lay3: [[m,d], [md,dd]]: 2 sLays,
    lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays
    '''

# for reference, redundant to slice_edge
class CP(CBase):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

    latuple: list = z([])  # lateral params to compare vertically: I,G,M,Ma,L, (Dy,Dx)
    derH: object = z(CH())  # nested vertical derivatives summed from links
    dert_: list = z([])  # array of pixel-level derts, ~ node_
    link_: list = z([[]])  # uplinks per comp layer, nest in rng+)der+
    cells: dict = z({})  # pixel-level kernels adjacent to P axis, combined into corresponding derts projected on P axis.
    roott: list = z([])  # PPrm,PPrd that contain this P, single-layer
    yx: list = z([])
    yx_: list = z([])
    # n = len dert_
    # axis: list = z([0,0])  # for P rotation, not used
    # dxdert_: list = z([])  # only in Pd
    # Pd_: list = z([])  # only in Pm
    # Mdx: int = 0  # if comp_dx
    # Ddx: int = 0
    def __bool__(self):  # to test empty
        if self.dert_: return True
        else: return False


class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster

    Et: list = z([])  # external eval tuple, summed from rng++ before forming new graph and appending G.extH

    latuple: list = z([])   # summed from Ps: lateral I,G,M,Ma,L,[Dy,Dx]
    iderH: object = z(CH())  # summed from PPs
    derH: object = z(CH())  # nested derH in Gs: [[subH,valt,rdnt,dect]], subH: [[derH,valt,rdnt,dect]]: 2-fork composition layers
    node_: list = z([])  # node_t after sub_recursion
    link_: list = z([])  # links per comp layer, nest in rng+)der+
    roott: list = z([])  # Gm,Gd that contain this G, single-layer
    area: int = 0
    box: list = z([inf, inf, -inf, -inf])  # y,x,y0,x0,yn,xn
    rng: int = 1
    fd: int = 0  # fork if flat layers?
    n: int = 0
    # graph-external, +level per root sub+:
    rim_H: list = z([])  # direct links, depth, init rim_t, link_tH in base sub+ | cpr rd+, link_tHH in cpr sub+
    # iextH: object = z(CH())  # not needed?
    extH: object = z(CH())  # G-external daggH( dsubH( dderH, summed from rim links
    S: float = 0.0  # sparsity: distance between node centers
    A: list = z([0,0])  # angle: summed dy,dx in links
    # tentative:
    alt_graph_: list = z([])  # adjacent gap+overlap graphs, vs. contour in frame_graphs
    # dynamic attrs:
    P_: list = z([])  # in PPs
    mask__: object = None
    root: list = z([]) # for feedback
    fback_: list = z([])  # feedback [[aggH,valt,rdnt,dect]] per node layer, maps to node_H
    compared_: list = z([])
    Rim_H: list = z([])  # links to the most mediated nodes

    # Rdn: int = 0  # for accumulation or separate recursion count?
    # it: list = z([None,None])  # graph indices in root node_s, implicitly nested
    # depth: int = 0  # n sub_G levels over base node_, max across forks
    # nval: int = 0  # of open links: base alt rep
    # id_H: list = z([[]])  # indices in the list of all possible layers | forks, not used with fback merging
    # top aggLay: derH from links, lower aggH from nodes, only top Lay in derG:
    # top Lay from links, lower Lays from nodes, hence nested tuple?

    def __bool__(self):  # to test empty
        if self.n: return True
        else: return False

class Clink(CBase):  # the product of comparison between two nodes

    _node: object = None  # prior comparand
    node:  object = None
    dderH: object = z(CH())  # derivatives produced by comp, nesting dertv -> aggH
    roott: list = z([None, None])  # clusters that contain this link
    S: float = 0.0  # sparsity: distance between node centers
    A: list = z([0,0])  # angle: dy,dx between centers
    # dir: bool  # direction of comparison if not G0,G1, only needed for comp link?

def get_match(_par, par):
    match = min(abs(_par),abs(par))
    return -match if (_par<0) != (par<0) else match    # match = neg min if opposite-sign comparands

def negate(He):
    if isinstance(He.H[0], CH):
        for lay in He.H:
            negate(lay)
    else:  # md_
        He.H[1::2] = [-d for d in He.H[1::2]]

def unpack_last_link_(link_):  # unpack last link layer

    while link_ and isinstance(link_[-1], list): link_ = link_[-1]
    return link_

def accum_box(box, y, x):
    """Box coordinate accumulation."""
    y0, x0, yn, xn = box
    return min(y0, y), min(x0, x), max(yn, y+1), max(xn, x+1)