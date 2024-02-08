import numpy as np
from collections import deque, defaultdict
from copy import deepcopy
from itertools import zip_longest, combinations
from typing import List, Tuple
from .classes import get_match, CderH, CderP, Cgraph, Ct, CP, Cangle
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
As we add higher dimensions (3D and time), this dimensionality reduction is done in salient high-aspect blobs
(likely edges in 2D or surfaces in 3D) to form more compressed "skeletal" representations of full-dimensional patterns.

comp_slice traces edge blob axis by cross-comparing vertically adjacent Ps: slices across edge blob, along P.G angle.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)

Connectivity in P_ is traced through root_s of derts adjacent to P.dert_, possibly forking. 
len prior root_ sorted by G is rdn of each root, to evaluate it for inclusion in PP, or starting new P by ave*rdn.
'''

  # root function:
def der_recursion(root, PP):  # node-mediated correlation clustering: keep same Ps and links, increment link derH, then P derH in sum2PP

    # n_uplinks = defaultdict(int)  # number of uplinks per P
    # for derP in PP.link_: n_uplinks[derP.P] += 1

    rng_recursion(PP, PP.rng)  # extend PP.link_, derHs by same-der rng+ comp
    form_PP_t(PP, PP.link_, base_rdn=PP.rdnt[1])  # der+ is mediated by form_PP_t
    if root: root.fback_ += [[PP.derH, PP.valt, PP.rdnt]]  # feedback from PPds only


def rng_recursion(PP, rng=1, der=1):  # similar to agg+ rng_recursion, but contiguously link mediated
    _P_ = PP.P_

    while True:
        P_ = []; V = 0; rng += 1
        for P in _P_:
            link_ = P.link_  # recursive unpack:
            if der>1: link_ = link_.pop
            if rng>1: link_ = link_.pop
            for link in link_.pop:  # prelink, append back if confirmed:
                mlink = comp_P(link)
                if mlink:  # return confirmed links only
                    V += mlink.Vt[0]
                    link_ += [link]
                    _P = link._P  # uplinks, can't be cyclic
                    if len(_P.derH) < len(P.derH):  # if der+: compare same der layers only
                        continue
                    dy,dx = _P.yx-P.yx; distance = np.hypot(dy,dx)  # distance between P midpoints, /= L for eval?
                    if rng-1 > distance >= rng:
                        continue
                    if P.valt[0]+_P.valt[0] < ave * (P.rdnt[0]+_P.rdnt[0]):
                        continue
                    link_ += [_P,P, distance, [dy,dx]]  # add prelink for next cycle
                    # recursive nesting, len link_H should increase with rng, add this for der+ too:
                    if len(P.link_) < rng: link_ = [link_]
                    P.link_ += [link_]
                    # recycle P to get higher __Ps:
                    if P not in P_: P_ += [P]
        # extended Ps val:
        if V > ave * len(P_) * 6:  # len mtuple
            _P_ = P_
        else:
            break
    PP.rng=rng


def comp_P(link):

    if isinstance(link,CderP): _P, P = link._P, link.P  # in der+
    else:                      _P, P, S, A = link  # list in rng+
    rn = len(_P.dert_) / len(P.dert_)

    if _P.derH and P.derH:
        # der+: append link derH, init in rng++ from form_PP_t
        dderH, valt, rdnt = comp_derH(_P.derH, P.derH, rn=rn)  # += fork rdn
        aveP = P_aves[1]
        fd=1
    else:
        # rng+: add link derH
        mtuple, dtuple = comp_ptuple(_P.ptuple, P.ptuple, rn, fagg=0)
        valt = Ct(sum(mtuple), sum(abs(d) for d in dtuple))
        rdnt = Ct(1+(valt.d>valt[0]), 1+(1-(valt[1]>valt[1])))  # or rdn = Dval/Mval?
        aveP = P_aves[0]
        fd=0
    if valt[0] > aveP*rdnt[0]:  # always rng+
        if fd:
            link.derH += dderH; link.valt+=valt; link.Vt+=valt; link.rdnt+=rdnt; link.Rt=rdnt
        else:
            link = CderH(derH=[[mtuple, dtuple]], valt=valt, Vt=valt, rdnt=rdnt, Rt=rdnt, S=S, A=A)

        return link


def form_PP_t(root, root_link_, base_rdn):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    for fd in 0,1:
        P_Ps = defaultdict(list)
        derP_ = []
        for derP in root_link_:
            if derP.valt[fd] > P_aves[fd] * derP.rdnt[fd] or root.root==None:  # init 0 valt
                P_Ps[derP.P] += [derP._P]  # key:P, vals:linked _Ps, up and down
                P_Ps[derP._P] += [derP.P]
                derP_ += [derP]  # filtered derP
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
            PP = sum2PP(root, cP_, derP_, base_rdn, fd)
            PP_t[fd] += [PP]  # no if Val > PP_aves[fd] * Rdn:
            inP_ += cP_  # update clustered Ps

    for PP in PP_t[1]:  # eval der+/ PPd only, after form_PP_t -> P.root
        if PP.valt[1] * len(PP.link_) > PP_aves[1] * PP.rdnt[1]:
            der_recursion(root, PP)  # node-mediated correlation clustering

        if root.fback_:
            feedback(root)  # after der+ in all nodes, no single node feedback up multiple layers

    root.node_ = PP_t  # nested in der+, add_alt_PPs_?


def sum2PP(root, P_, derP_, base_rdn, fd):  # sum links in Ps and Ps in PP

    PP = Cgraph(fd=fd, root=root, P_=P_, rng=root.rng +(1-fd))  # initial PP.box = (inf,inf,-inf,-inf)

    for derP in derP_:
        # accum links:
        if derP.P not in P_ or derP._P not in P_: continue
        derP.roott[fd] = PP
        derH,valt,rdnt = derP.derH,derP.valt,derP.rdnt
        P = derP.P  # uplink
        P.derH += derH; P.valt += valt; P.rdnt += rdnt + (base_rdn, base_rdn)  # P.derH sums link derH s
        _P = derP._P  # bilateral accum downlink, reverse d signs:
        _P.derH -= derH; _P.valt += valt; _P.rdnt += rdnt + (base_rdn, base_rdn)
        PP.ext[2] += derP.A
        PP.ext[1] += derP.S
        PP.link_ += [derP]

    celly_,cellx_ = [],[]
    for P in P_:
        # accum Ps:
        PP.ptuple += P.ptuple  # accum ptuple
        for y,x in P.cells:
            PP.box = PP.box.accumulate(y,x)
            celly_ += [y]; cellx_ += [x]
        # unilateral sum:
        PP.derH += P.derH
        # below should be not needed if it's in PP.derH now?
        PP.valt += P.valt
        PP.rdnt += P.rdnt + (base_rdn, base_rdn)
        PP.ext[0] += 1  # or PP.ext[0] = len(P_)

    y0, x0, yn, xn = PP.box
    PP.mask__ = np.zeros((yn-y0, xn-x0), bool)
    celly_ = np.array(celly_); cellx_ = np.array(cellx_)
    PP.mask__[(celly_-y0, cellx_-x0)] = True  # assign PP mask
    PP.L = PP.ptuple[-1]  # or unpack it when needed?

    return PP


def feedback(root):  # in form_PP_, append new der layers to root PP, single vs. root_ per fork in agg+

    derH, valt, rdnt = CderH(), Ct(0,0), Ct(0,0)
    while root.fback_:
        _derH, _valt, _rdnt = root.fback_.pop(0)
        derH += _derH; valt += _valt; rdnt += _rdnt

    root.derH += derH; root.valt += valt; root.rdnt += rdnt

    if isinstance(root.root, Cgraph):  # skip if root is Edge
        rroot = root.root  # single PP.root, can't be P
        fback_ = rroot.fback_
        node_ = rroot.node_[1] if isinstance(rroot.node_[0],list) else rroot.node_  # node_ is updated to node_t in sub+
        fback_ += [(derH, valt, rdnt)]
        if fback_ and (len(fback_)==len(node_)):  # all nodes terminated and fed back
            feedback(rroot)  # sum2PP adds derH per rng, feedback adds deeper sub+ layers


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

# old:
def comp_ptuple_gen(_ptuple, ptuple, rn):  # 0der

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


def comp_derH(_derH, derH, rn):  # derH is a list of der layers or sub-layers, each = ptuple_tv

    dderH = []  # or not-missing comparand: xor?
    Mval, Dval, Mrdn, Drdn = 0,0,1,1

    for _lay, lay in zip(_derH.H, derH.H):  # compare common lower der layers | sublayers in derHs
        # if lower-layers match: Mval > ave * Mrdn?
        mtuple, dtuple = comp_dtuple(_lay[1], lay[1], rn, fagg=0)  # compare dtuples only
        mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
        mrdn = dval > mval; drdn = dval < mval
        Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn
        dderH += [[mtuple, dtuple]]

    return CderH(H=dderH, valt=Ct(Mval,Dval), rdnt=Ct(Mrdn,Drdn),depth=0)  # new derLayer,= 1/2 combined derH


def sum_derH_gen(T, t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

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