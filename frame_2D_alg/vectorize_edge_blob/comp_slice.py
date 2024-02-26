import numpy as np
from collections import deque, defaultdict
from copy import deepcopy, copy
from itertools import zip_longest, combinations
from typing import List, Tuple
from .classes import add_, comp_, negate, get_match, CPP, Cptuple, CderP, z
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
    form_PP_t(PP, PP.P_, iRt = PP.Rt)  # der+ is mediated by form_PP_t
    if root: root.fback_ += [[PP.He, PP.valt, PP.rdnt]]  # feedback from PPds


def rng_recursion(PP, rng=1, fd=0):  # similar to agg+ rng_recursion, but contiguously link mediated, because

    iP_ = PP.P_
    while True:
        P_ = []; V = 0
        for P in iP_:
            if not P.link_: continue
            prelink_ = []  # new prelinks per P
            _prelink_ = P.link_.pop()  # old prelinks per P
            for _link in _prelink_:
                _P = _link._P if fd else _link
                dy,dx = np.subtract(_P.yx, P.yx)
                distance = np.hypot(dy,dx)  # distance between P midpoints, /= L for eval?
                if distance < rng:  # | rng * ((P.val+_P.val) / ave_rval)?
                    mlink = comp_P(_link if fd else [_P,P, distance,[dy,dx]], fd)  # return link if match
                    if mlink:
                        V += mlink.vt[0]  # unpack last link layer:
                        link_ = P.link_[-1] if PP.He and PP.He[0] else P.link_  # der++ if PP.He[0] depth==1
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

    if isinstance(link,z): _P, P = link._P, link.P  # in der+
    else:                  _P, P, S, A = link  # list in rng+
    rn = len(_P.dert_) / len(P.dert_)

    if _P.He and P.He:
        # der+: append link derH, init in rng++ from form_PP_t
        (vm,vd,rm,rd),H = comp_(_P.He, P.He, rn=rn)
        rm += vd > vm; rd += vm >= vd
        aveP = P_aves[1]
    else:
        # rng+: add link derH
        H = comp_ptuple(_P.ptuple, P.ptuple, rn)
        vm = sum(H[::2]); vd = sum(abs(d) for d in H[1::2])
        rm = 1 + vd > vm; rd = 1 + vm >= vd
        aveP = P_aves[0]

    if vm > aveP*rm:  # always rng+
        if fd:
            He = link.He
            if not He[0]: He = link.He = [1,[*He[1]],[He]]  # nest md_ as derH
            He[1] = np.add(He[1],[vm,vd,rm,rd])
            He[2] += [[0, [vm,vd,rm,rd], H]]  # nesting, Et, H
            link.vt = np.add(link.vt,[vm,vd]); link.rt = np.add(link.rt,[rm,rd])
        else:
            link = CderP(typ='derP', P=P,_P=_P, He=[0,[vm,vd,rm,rd],H], vt=[vm,vd], rt=[rm,rd], S=S, A=A, roott=[[],[]])

        return link


def form_PP_t(root, P_, iRt):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    for fd in 0,1:
        P_Ps = []; Link_ = []
        for P in P_:  # not PP.link_: P uplinks are unique, only G links overlap
            Ps = []
            for derP in unpack_last_link_(P.link_):
                Ps += [derP._P]; Link_ += [derP]  # not needed for PPs?
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
        if PP.Vt[1] * len(PP.link_) > PP_aves[1] * PP.Rt[1]:
            # node-mediated correlation clustering:
            der_recursion(root, PP, fd=1)
        if root.fback_:
            feedback(root)  # after der+ in all nodes, no single node feedback

    root.node_ = PP_t  # nested in der+, add_alt_PPs_?


def sum2PP(root, P_, derP_, iRt, fd):  # sum links in Ps and Ps in PP

    PP = CPP(typ='PP',fd=fd,root=root,P_=P_,rng=root.rng+1, Vt=[0,0],Rt=[1,1],Dt=[0,0], link_=[], box=[0,0,0,0],  # not inf,inf,-inf,-inf?
           ptuple = z(typ='ptuple',I=0, G=0, M=0, Ma=0, angle=[0,0], L=0), He=[])
    # += uplinks:
    S,A = 0, [0,0]
    for derP in derP_:
        if derP.P not in P_ or derP._P not in P_: continue
        if derP.He:
            add_(derP.P.He, derP.He, iRt)
            add_(derP._P.He, negate(deepcopy(derP.He)), iRt)
        PP.link_ += [derP]; derP.roott[fd] = PP
        PP.Vt = [V+v for V,v in zip(PP.Vt, derP.vt)]
        PP.Rt = [R+r+ir for R,r,ir in zip(PP.Rt, derP.rt, iRt)]
        derP.A = np.add(A,derP.A); S += derP.S
    PP.ext = [len(P_), S, A]  # all from links
    # += Ps:
    celly_,cellx_ = [],[]
    for P in P_:
        PP.ptuple += P.ptuple
        add_(PP.He, P.He)
        for y,x in P.cells:
            PP.box = accum_box(PP.box, y, x); celly_+=[y]; cellx_+=[x]
    # pixmap:
    y0,x0,yn,xn = PP.box
    PP.mask__ = np.zeros((yn-y0, xn-x0), bool)
    celly_ = np.array(celly_); cellx_ = np.array(cellx_)
    PP.mask__[(celly_-y0, cellx_-x0)] = True

    return PP


def feedback(root):  # in form_PP_, append new der layers to root PP, single vs. root_ per fork in agg+

    HE, Valt, Rdnt = deepcopy(root.fback_.pop(0))  # root update
    while root.fback_:
        He, valt, rdnt = root.fback_.pop(0)
        Valt = [V+v for V,v in zip(Valt, valt)]; Rdnt = [R+r for R,r in zip(Rdnt, rdnt)]
        add_(HE, He)
    add_(root.He, HE if HE[0] else HE[2][-1])  # sum md_ or last md_ in H
    root.valt = [V+v for V,v in zip(root.valt, Valt)]; root.rdnt = [R+r for R,r in zip(root.rdnt, Rdnt)]

    if root.typ != "edge":  # skip if root is Edge
        rroot = root.root  # single PP.root, can't be P
        fback_ = rroot.fback_
        node_ = rroot.node_[1] if rroot.node_ and isinstance(rroot.node_[0],list) else rroot.P_  # node_ is updated to node_t in sub+
        fback_ += [(HE, Valt, Rdnt)]
        if fback_ and (len(fback_)==len(node_)):  # all nodes terminated and fed back
            feedback(rroot)  # sum2PP adds derH per rng, feedback adds deeper sub+ layers

# this is replaced by comp_?
def comp_dtuple(_ptuple, ptuple, rn, fagg=0):

    mtuple, dtuple = [],[]
    if fagg: Mtuple, Dtuple = [],[]

    for _par, par, ave in zip(_ptuple, ptuple, aves):  # compare ds only
        par *= rn
        mtuple += [get_match(_par, par) - ave]
        dtuple += [_par - par]
        if fagg:
            Mtuple += [max(abs(par),abs(par))]
            Dtuple += [abs(_par)+abs(par)]
    ret = [mtuple, dtuple]
    if fagg: ret += [Mtuple, Dtuple]
    return ret


def comp_ptuple(_ptuple, ptuple, rn, fagg=0):  # 0der params

    _I, _G, _M, _Ma, (_Dy, _Dx), _L = _ptuple.I, _ptuple.G, _ptuple.M,_ptuple.Ma,_ptuple.angle,_ptuple.L
    I, G, M, Ma, (Dy, Dx), L = ptuple.I, ptuple.G, ptuple.M,ptuple.Ma,ptuple.angle,ptuple.L

    dI = _I - I*rn;  mI = ave_dI - dI
    dG = _G - G*rn;  mG = min(_G, G*rn) - aves[1]
    dL = _L - L*rn;  mL = min(_L, L*rn) - aves[2]
    dM = _M - M*rn;  mM = get_match(_M, M*rn) - aves[3]  # M, Ma may be negative
    dMa= _Ma- Ma*rn; mMa = get_match(_Ma, Ma*rn) - aves[4]
    mAngle, dAngle = comp_angle((_Dy,_Dx), (Dy,Dx))

    ret = [mI,dI,mG,dG,mM,dM,mMa,dMa,mAngle-aves[5],dAngle,mL,dL]

    if fagg:
        Ret = [max(_I,I), abs(_I)+abs(I),max(_G,G),abs(_G)+abs(G), max(_M,M), abs(_M)+abs(M), max(_Ma,Ma), abs(_Ma)+abs(Ma), 2, 2, max(_L,L),abs(_L)+abs(L)]
        # ret = [ret, Ret]
        # ?
        mval, dval = sum(ret[::2]),sum(ret[1::2])
        mrdn, drdn = dval>mval, mval>dval
        mdec, ddec = 0, 0
        for fd, (ptuple,Ptuple) in enumerate(zip((ret[::2],ret[1::2]),(Ret[::2],Ret[1::2]))):
            for i, (par, maxv, ave) in enumerate(zip(ptuple, Ptuple, aves)):
                # compute link decay coef: par/ max(self/same)
                if fd: ddec += abs(par)/ abs(maxv) if maxv else 1
                else:  mdec += (par+ave)/ (maxv+ave) if maxv else 1
        mdec /= 6; ddec /= 6  # ave of 6 params
        ret = [mval, dval, mrdn, drdn, mdec, ddec], ret
    return ret


def comp_ptuple_generic(_ptuple, ptuple, rn):  # 0der

    mtuple, dtuple, Mtuple = [],[],[]
    # _n, n = _ptuple, ptuple: add to rn?
    for i, (_par, par, ave) in enumerate(zip(_ptuple, ptuple, aves)):
        if isinstance(_par, list) or isinstance(_par, tuple):
             m,d = comp_angle(_par, par)
             maxv = 2
        else:  # I | M | G L
            npar= par*rn  # accum-normalized param
            d = _par - npar
            if i: m = min(_par,npar)-ave
            else: m = ave-abs(d)  # inverse match for I, no mag/value correlation
            maxv = max(_par, par)
        mtuple+=[m]
        dtuple+=[d]
        Mtuple+=[maxv]
    return [mtuple, dtuple, Mtuple]

def unpack_last_link_(link_):  # unpack last link layer

    while link_ and isinstance(link_[-1], list): link_ = link_[-1]
    return link_