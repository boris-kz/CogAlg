import numpy as np
from collections import deque, defaultdict
from copy import deepcopy, copy
from itertools import zip_longest, combinations
from typing import List, Tuple
from .classes import add_, sub_, acc_, get_match, CderH, CderP, Cgraph
from .filters import ave, ave_dI, aves, P_aves, PP_aves
from .slice_edge import comp_angle

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
(likely edges in 2D or surfaces in 3D) to form more compressed "skeletal" representations of full-dimensional patterns.

comp_slice traces edge blob axis by cross-comparing vertically adjacent Ps: slices across edge blob, along P.G angle.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)

Connectivity in P_ is traced through root_s of derts adjacent to P.dert_, possibly forking. 
len prior root_ sorted by G is root.rdn, to eval for inclusion in PP or start new P by ave*rdn
'''

  # root function:
def der_recursion(root, PP):  # node-mediated correlation clustering: keep same Ps and links, increment link derH, then P derH in sum2PP

    # n_uplinks = defaultdict(int)  # number of uplinks per P, not used?
    # for derP in PP.link_: n_uplinks[derP.P] += 1

    if PP.derH.depth == 1:  # not initial call, form prelinks per P
        for P in PP.P_:  # add prelink_ for rng++:
            P.link_ += [[link._P for link in unpack_last_link_(P.link_)]]

    rng_recursion(PP, rng=1)  # extend PP.link_, derHs by same-der rng+ comp
    form_PP_t(PP, PP.P_, irdn=PP.rdnt[1])  # der+ is mediated by form_PP_t
    if root: root.fback_ += [[PP.derH, PP.valt, PP.rdnt]]  # feedback from PPds


def rng_recursion(PP, rng=1):  # similar to agg+ rng_recursion, but contiguously link mediated, because

    iP_ = PP.P_
    while True:
        P_ = []; V = 0
        for P in iP_:
            if not P.link_: continue
            __P_ = []  # temporary prelinks per P
            _P_ = P.link_.pop()  # P.link_ nesting doesn't matter
            for _P in _P_:
                if len(_P.derH.H)!= len(P.derH.H): continue  # compare same der layers only
                dy,dx = np.subtract(_P.yx, P.yx)
                distance = np.hypot(dy,dx)  # distance between P midpoints, /= L for eval?
                if distance < rng:  # | rng * ((P.val+_P.val) / ave_rval)?
                    __P_ += [_P]  # for next rng+
                    mlink = comp_P([_P,P, distance,[dy,dx]])  # return link if match
                    if mlink:
                        V += mlink.vt[0]  # unpack last link layer:
                        link_ = P.link_[-1] if PP.derH.depth else P.link_  # der++ if derH.depth==1
                        if rng > 1:
                            if rng == 2: link_[:] = [link_[:]]  # link_ -> link_H
                            if len(link_) < rng: link_ += [[]]  # new link_
                            link_ = link_[-1]  # last rng layer
                        link_ += [mlink]
                        __P_ += [link._P for link in unpack_last_link_(_P.link_[:-1])]  # get last link layer, skip prelinks
            if __P_:
                P.link_ += [__P_]  # temporary, appended and popped regardless of nesting
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

def comp_P(link):

    if isinstance(link,CderP): _P, P = link._P, link.P  # in der+
    else:                      _P, P, S, A = link  # list in rng+
    rn = len(_P.dert_) / len(P.dert_)

    if _P.derH.H and P.derH.H:
        # der+: append link derH, init in rng++ from form_PP_t
        derLay, vt,rt,_ = comp_derH(_P.derH, P.derH, rn=rn)  # += fork rdn
        aveP = P_aves[1]
        fd=1
    else:
        # rng+: add link derH
        mtuple, dtuple = comp_ptuple(_P.ptuple, P.ptuple, rn)
        vt = [sum(mtuple), sum(abs(d) for d in dtuple)]
        rt = [1+(vt[1]>vt[0]), 1+(1-(vt[1]>vt[0]))]  # or rdn = Dval/Mval?
        aveP = P_aves[0]
        fd=0
    if vt[0] > aveP*rt[0]:  # always rng+
        if fd:
            if link.derH.depth==0:  # add nesting dertv-> derH:
                link.derH.H = [link.derH.H], link.derH.depth==1
            link.derH += derLay; link.vt=vt; link.rt=rt
        else:
            derH = CderH(H=[mtuple, dtuple], valt=vt, rdnt=rt, depth=0)  # dertv
            link = CderP(P=P,_P=_P, derH=derH, vt=copy(vt), rt=copy(rt), S=S, A=A)

        return link


def form_PP_t(root, P_, irdn):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    for fd in 0,1:
        P_Ps = defaultdict(list)  # key: P, val: _Ps
        Link_ = []
        for P in P_:  # not PP.link_: P uplinks are unique, only G links overlap
            for derP in unpack_last_link_(P.link_):
                P_Ps[P] += [derP._P]; Link_ += [derP]  # not needed for PPs?
        inP_ = []  # clustered Ps and their val,rdn s for all Ps
        for P in root.P_:
            if P in inP_: continue  # already packed in some PP
            cP_ = [P]  # clustered Ps and their val,rdn s
            perimeter = deque(P_Ps[P])  # recycle with breadth-first search, up and down:
            while perimeter:
                _P = perimeter.popleft()
                if _P in cP_: continue
                cP_ += [_P]
                perimeter += P_Ps[_P]  # append linked __Ps to extended perimeter of P
            PP = sum2PP(root, cP_, Link_, irdn, fd)
            PP_t[fd] += [PP]  # no if Val > PP_aves[fd] * Rdn:
            inP_ += cP_  # update clustered Ps

    for PP in PP_t[1]:  # eval der+ / PPd only, after form_PP_t -> P.root
        if PP.Vt[1] * len(PP.link_) > PP_aves[1] * PP.Rt[1]:
            # node-mediated correlation clustering:
            der_recursion(root, PP)
        if root.fback_:
            feedback(root)  # after der+ in all nodes, no single node feedback

    root.node_ = PP_t  # nested in der+, add_alt_PPs_?


def sum2PP(root, P_, derP_, irdn, fd):  # sum links in Ps and Ps in PP

    PP = Cgraph(fd=fd, root=root, P_=P_, rng=root.rng+1)  # initial PP.box = (inf,inf,-inf,-inf)
    # += uplinks:
    S,A = 0, [0,0]
    for derP in derP_:
        if derP.P not in P_ or derP._P not in P_: continue
        derP.P.derH += derP.derH  # +base_rdn?
        derP._P.derH -= derP.derH  # reverse d signs downlink
        PP.link_ += [derP]; derP.roott[fd] = PP
        PP.Vt = np.add(PP.Vt,derP.vt)
        PP.Rt = np.add(np.add(PP.Rt,derP.rt), [irdn,irdn])
        derP.A = np.add(A,derP.A); S += derP.S
    PP.ext = [len(P_), S, A]  # all from links
    depth = root.derH.depth or fd  # =1 at 1st der+
    PP.derH.depth = depth
    # += Ps:
    celly_,cellx_ = [],[]
    for P in P_:
        P.derH.depth = depth  # or copy from links
        PP.ptuple += P.ptuple
        PP.derH += P.derH
        for y,x in P.cells:
            PP.box = PP.box.accumulate(y,x); celly_+=[y]; cellx_+=[x]
    # pixmap:
    y0,x0,yn,xn = PP.box
    PP.mask__ = np.zeros((yn-y0, xn-x0), bool)
    celly_ = np.array(celly_); cellx_ = np.array(cellx_)
    PP.mask__[(celly_-y0, cellx_-x0)] = True

    return PP


def feedback(root):  # in form_PP_, append new der layers to root PP, single vs. root_ per fork in agg+

    derH, valt, rdnt = CderH(),[0,0],[0,0]
    while root.fback_:
        _derH, _valt, _rdnt = root.fback_.pop(0)
        derH += _derH; acc_(valt,_valt); acc_(rdnt,_rdnt)

    root.derH += derH; add_(root.valt,_valt); add_(root.rdnt,_rdnt)

    if isinstance(root.root, Cgraph):  # skip if root is Edge
        rroot = root.root  # single PP.root, can't be P
        fback_ = rroot.fback_
        node_ = rroot.node_[1] if isinstance(rroot.node_[0],list) else rroot.node_  # node_ is updated to node_t in sub+
        fback_ += [(derH, valt, rdnt)]
        if fback_ and (len(fback_)==len(node_)):  # all nodes terminated and fed back
            feedback(rroot)  # sum2PP adds derH per rng, feedback adds deeper sub+ layers


def comp_dtuple(_ptuple, ptuple, rn, fagg=0):

    mtuple, dtuple = [],[]
    if fagg: Mtuple, Dtuple = [],[]

    for _par, par, ave in zip(_ptuple, ptuple, aves):  # compare ds only
        npar = par * rn
        mtuple += [get_match(_par, npar) - ave]
        dtuple += [_par - npar]
        if fagg:
            Mtuple += [max(abs(par),abs(npar))]
            Dtuple += [abs(_par)+abs(npar)]
    ret = [mtuple, dtuple]
    if fagg: ret += [Mtuple, Dtuple]
    return ret

def comp_ptuple(_ptuple, ptuple, rn, fagg=0):  # 0der params

    _I, _G, _M, _Ma, (_Dy, _Dx), _L = _ptuple
    I, G, M, Ma, (Dy, Dx), L = ptuple

    dI = _I - I*rn;  mI = ave_dI - dI
    dG = _G - G*rn;  mG = min(_G, G*rn) - aves[1]
    dL = _L - L*rn;  mL = min(_L, L*rn) - aves[2]
    dM = _M - M*rn;  mM = get_match(_M, M*rn) - aves[3]  # M, Ma may be negative
    dMa= _Ma- Ma*rn; mMa = get_match(_Ma, Ma*rn) - aves[4]
    mAngle, dAngle = comp_angle((_Dy,_Dx), (Dy,Dx))

    mtuple = [mI, mG, mM, mMa, mAngle-aves[5], mL]
    dtuple = [dI, dG, dM, dMa, dAngle, dL]
    ret = [mtuple, dtuple]
    if fagg:
        Mtuple = [max(_I,I), max(_G,G), max(_M,M), max(_Ma,Ma), 2, max(_L,L)]
        Dtuple = [abs(_I)+abs(I), abs(_G)+abs(G), abs(_M)+abs(M), abs(_Ma)+abs(Ma), 2, abs(_L)+abs(L)]
        ret += [Mtuple, Dtuple]
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


def comp_derH(_derH, derH, rn=1, fagg=0):  # derH is a list of der layers or sub-layers, each = ptuple_tv

    Ht = []
    for derH in [_derH, derH]:  # init H is dertuplet, local convert to dertv_ (permanent conversion in sum2PP):
        Ht += [derH.H] if isinstance(derH.H[0],CderH) else [CderH(H=derH.H, valt=copy(derH.valt), rdnt=copy(derH.rdnt), dect=copy(derH.dect), depth=0)]
    derLay = []; Vt,Rt,Dt = [0,0],[0,0],[0,0]

    for _lay, lay in zip(Ht):
        mtuple,dtuple, Mtuple,Dtuple = comp_dtuple(_lay.H[1], lay.H[1], rn=rn, fagg=1)
        valt = [sum(mtuple),sum(abs(d) for d in dtuple)]
        rdnt = [valt[1] > valt[0], valt[1] <= valt[0]]
        dect = [0,0]
        for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
            for (par, max, ave) in zip(ptuple, Ptuple, aves):  # different ave for comp_dtuple
                if fagg:
                    if fd: dect[1] += abs(par)/ abs(max) if max else 1
                    else:  dect[0] += (par+ave)/ (max+ave) if max else 1
        if fagg:
            dect[0] = dect[0]/6; dect[1] = dect[1]/6  # ave of 6 params

        Vt = np.add(Vt,valt); Rt = np.add(Rt,rdnt)
        if fagg: Dt = np.divide(np.add(Dt,dect),2)
        derLay += [CderH(H=[mtuple,dtuple], valt=valt,rdnt=rdnt,dect=dect, depth=0)]  # dertvs

    return derLay, Vt,Rt,Dt  # to sum in each G Et


# replaced by += overload for CderH in classes:
def sum_derH(T, t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    DerH, Valt, Rdnt = T; derH, valt, rdnt = t
    for i in 0,1:
        Valt[i] += valt[i]
        Rdnt[i] += rdnt[i] + base_rdn
    DerH[:] = [
        # sum der layers, dertuple is mtuple | dtuple, fneg*i: for dtuple only:
        [ sum_dertuple(Mtuple, mtuple, fneg=0), sum_dertuple(Dtuple, dtuple, fneg=fneg) ]
        for [Mtuple, Dtuple], [mtuple, dtuple]
        in zip_longest(DerH, derH, fillvalue=[[0,0,0,0,0,0],[0,0,0,0,0,0]])  # mtuple,dtuple
    ]

def sum_dertuple(Ptuple, ptuple, fneg=0):
    _I, _G, _M, _Ma, _A, _L = Ptuple
    I, G, M, Ma, A, L = ptuple
    if fneg: Ptuple[:] = [_I-I, _G-G, _M-M, _Ma-Ma, _A-A, _L-L]
    else:    Ptuple[:] = [_I+I, _G+G, _M+M, _Ma+Ma, _A+A, _L+L]
    return   Ptuple

def sum_derH_generic(T, t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    DerH, Valt, Rdnt = T; derH, valt, rdnt = t
    Rdnt += rdnt + base_rdn
    Valt += valt
    if DerH:
        for Layer, layer in zip_longest(DerH,derH, fillvalue=[]):
            if layer:
                if Layer:
                    for i in 0,1:
                        sum_dertuple(Layer[0][i], layer[0][i], fneg and i)  # ptuplet, only dtuple is directional: needs fneg
                        Layer[1][i] += layer[1][i]  # valt
                        Layer[2][i] += layer[2][i] + base_rdn  # rdnt
                else:
                    DerH += [deepcopy(layer)]
    else:
        DerH[:] = deepcopy(derH)


def unpack_last_link_(link_):  # unpack last link layer
    link_ = []
    while link_ and isinstance(link_[-1], list): link_ = link_[-1]
    return link_

