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
from typing import List, Tuple
from .classes import add_, comp_, negate, get_match, Clink, CG, CH
from .filters import ave, ave_dI, aves, P_aves, PP_aves
from .slice_edge import comp_angle
from utils import box2slice, accum_box, sub_box2box

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
def der_recursion(root, PP, fd=0):  # node-mediated correlation clustering: keep same Ps and links, increment link derH, then P derH in sum2PP

    if fd:  # add prelinks per P if not initial call:
        for P in PP.P_: P.link_ += [copy(unpack_last_link_(P.link_))]

    rng_recursion(PP, rng=1, fd=fd)  # extend PP.link_, derHs by same-der rng+ comp

    form_PP_t(PP, PP.P_, iRt = PP.derH.Et[2:4] if PP.derH else [0,0])  # der+ is mediated by form_PP_t
    if root: root.fback_ += [PP.derH]  # feedback from PPds


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
                dy,dx = np.subtract(_P.yx, P.yx)
                distance = np.hypot(dy,dx)  # distance between P midpoints, /= L for eval?
                if distance < rng:  # | rng * ((P.val+_P.val) / ave_rval)?
                    mlink = comp_P(_link if fd else [_P,P, distance,[dy,dx]], fd)  # return link if match
                    if mlink:
                        V += mlink.dderH.Et[0]  # unpack last link layer:
                        link_ = P.link_[-1] if P.link_ and isinstance(P.link_[-1], list) else P.link_  # der++ if PP.He[0] depth==1
                        if rng > 1:
                            if rng == 2: link_[:] = [link_[:]]  # link_ -> link_H
                            if len(link_) < rng: link_ += [[]]  # new link_
                            link_ = link_[-1]  # last rng layer
                        link_ += [mlink]
                        prelink_ += unpack_last_link_(_P.link_[:-1])  # get last link layer, skip old prelinks
            if prelink_:
                if not fd: prelink_ = [link._P for link in prelink_]  # prelinks are __Ps, else __links
                P.link_ += [prelink_]  # temporary prelinks
                P_ += [P]  # for next loop
        rng += 1
        if V > ave * len(P_) * 6:  #  implied val of all __P_s, 6: len mtuple
            iP_ = P_
        else:
            for P in P_: P.link_.pop()
            break
    PP.rng=rng
    '''
    der++ is tested in PPds formed by rng++, no der++ inside rng++: high diff @ rng++ termination only?
    '''

def comp_P(link, fd):

    if isinstance(link,Clink): _P, P = link._node, link.node  # in der+
    else:                      _P, P, S, A = link  # list in rng+
    rn = len(_P.dert_) / len(P.dert_)

    if _P.derH and P.derH:
        # der+: append link derH, init in rng++ from form_PP_t
        dHe = comp_(_P.derH, P.derH, rn=rn)
        vm,vd,rm,rd = dHe.Et[:4]  # for call from comp_G
        rm += vd > vm; rd += vm >= vd
        aveP = P_aves[1]
    else:
        # rng+: add link derH
        H = comp_latuple(_P.latuple, P.latuple, rn)
        vm = sum(H[::2]); vd = sum(abs(d) for d in H[1::2])
        rm = 1 + vd > vm; rd = 1 + vm >= vd
        n = (len(_P.dert_)+len(P.dert_)) /2  # ave compared n
        aveP = P_aves[0]

    if vm > aveP*rm:  # always rng+
        if fd:
            He = link.dderH
            if not He.nest: He = link.He = CH(nest=1, Et=[*He.Et],H=[He])  # nest md_ as derH
            He.Et = np.add(He.Et,[vm,vd,rm,rd])
            He.H += [dHe]
        else:
            link = Clink(node=P,_node=_P, dderH = CH(nest=0,Et=[vm,vd,rm,rd],H=H,n=n), S=S, A=A, roott=[[],[]])

        return link


def form_PP_t(root, P_, iRt):  # form PPs of derP.valt[fd] + connected Ps val

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
        if PP.derH and PP.derH.Et[0] * len(PP.link_) > PP_aves[1] * PP.derH.Et[2]:
            # node-mediated correlation clustering:
            der_recursion(root, PP, fd=1)
        if root.fback_:
            feedback(root)  # after der+ in all nodes, no single node feedback

    root.node_ = PP_t  # nested in der+, add_alt_PPs_?


def sum2PP(root, P_, derP_, iRt, fd):  # sum links in Ps and Ps in PP

    PP = CG(fd=fd,root=root,P_=P_,rng=root.rng+1, link_=[], box=[inf,inf,-inf,-inf], latuple=[0,0,0,0,0,[0,0]])
    # += uplinks:
    for derP in derP_:
        if derP.node not in P_ or derP._node not in P_: continue
        if derP.dderH:
            add_(derP.node.derH, derP.dderH, iRt)
            add_(derP._node.derH, negate(deepcopy(derP.dderH)), iRt)  # to reverse uplink direction
        PP.link_ += [derP]; derP.roott[fd] = PP
        PP.A = np.add(PP.A,derP.A); PP.S += derP.S
        PP.n += derP.n
    # += Ps:
    celly_,cellx_ = [],[]
    for P in P_:
        L = P.latuple[-2]
        PP.area += L; PP.n += L  # no + P.derH.n: current links only?
        PP.latuple = [P+p for P,p in zip(PP.latuple[:-1],P.latuple[:-1])] + [[A+a for A,a in zip(PP.latuple[-1],P.latuple[-1])]]
        if P.derH:
            add_(PP.derH, P.derH)
        for y,x in P.cells:
            PP.box = accum_box(PP.box, y, x); celly_+=[y]; cellx_+=[x]
    if PP.derH:
        PP.derH.Et[2:4] = [R+r for R,r in zip(PP.derH.Et[2:4], iRt)]
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
        add_(HE, He)
    add_(root.derH, HE.H[-1] if HE.nest else HE)  # last md_ in H or sum md_

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

    ret = [mI,dI,mG,dG,mM,dM,mMa,dMa,mAngle-aves[5],dAngle,mL,dL]

    if fagg:  # add norm m,d: ret = [ret, Ret]
        # max possible m,d per compared param
        Ret = [max(_I,I),abs(_I)+abs(I), max(_G,G),abs(_G)+abs(G), max(_M,M),abs(_M)+abs(M), max(_Ma,Ma),abs(_Ma)+abs(Ma), max(_L,L),abs(_L)+abs(L), 1,.5]
        mval, dval = sum(ret[::2]),sum(ret[1::2])
        mrdn, drdn = dval>mval, mval>dval
        mdec, ddec = 0, 0
        for fd, (ptuple,Ptuple) in enumerate(zip((ret[::2],ret[1::2]),(Ret[::2],Ret[1::2]))):
            for i, (par, maxv, ave) in enumerate(zip(ptuple, Ptuple, aves)):  # compute link decay coef: par/ max(self/same)
                if fd: ddec += abs(par)/ abs(maxv) if maxv else 1
                else:  mdec += (par+ave)/ (maxv+ave) if maxv else 1

        ret = [mval, dval, mrdn, drdn, mdec, ddec], ret
    return ret

def unpack_last_link_(link_):  # unpack last link layer

    while link_ and isinstance(link_[-1], list): link_ = link_[-1]
    return link_