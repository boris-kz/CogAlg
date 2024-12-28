import numpy as np
from copy import deepcopy
from itertools import zip_longest
from collections import deque, defaultdict
from frame_2D_alg.slice_edge import comp_angle
from frame_2D_alg.vectorize_edge_blob.classes import CderP, CPP
from frame_2D_alg.vectorize_edge_blob.filters import ave, aves, P_aves, PP_aves

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

def comp_P_(edge):  # cross-comp P_ in edge: high-gradient blob, sliced in Ps in the direction of G

    P_ = edge.node_t  # init as P_
    edge.node_t = [[],[]]  # fill with sub_PPm_, sub_PPd_ in form_PP_t, ~ sub+ but rng+ only:
    for P in P_:
        link_ = []
        for _P in P.link_H[-1]:  # scan, comp contiguously uplinked Ps, rn: relative weight of comparand
            comp_P(link_,_P,P, rn=len(_P.dert_)/len(P.dert_), fd=0)
        P.link_H[-1] = link_

    form_PP_t(edge, P_, base_rdn=2)  # replace edge.node_t with PP_t, may be nested by sub+


def comp_P(link_, _P,P, rn, fd=1, derP=None):  #  derP if der+, reused as S if rng+
    aveP = P_aves[fd]

    if fd:  # der+: extend in-link derH, in sub+ only
        dderH, valt, rdnt = comp_derH(_P.derH, P.derH, rn)  # += fork rdn
        derP = CderP(derH = derP.derH+dderH, valt=valt, rdnt=rdnt, P=P,_P=_P, S=derP.S)  # dderH valt,rdnt for new link
        mval,dval = valt[:2]; mrdn,drdn = rdnt  # exclude maxv

    else:  # rng+: add derH
        mtuple,dtuple = comp_ptuple(_P.ptuple, P.ptuple, rn)
        mval = sum(mtuple); dval = sum(dtuple)
        mrdn = 1+(dval>mval); drdn = 1+(1-(dval>mval))  # or rdn = Dval/Mval?
        derP = CderP(derH=[[[mtuple,dtuple], [mval,dval],[mrdn,drdn]]], valt=[mval,dval], rdnt=[mrdn,drdn], P=P,_P=_P, S=derP)

    if mval > aveP*mrdn or dval > aveP*drdn:
        link_ += [derP]

# rng+ and der+ are called from sub_recursion:
def comp_rng(iP_, rng):  # form new Ps and links, switch to rng+n to skip clustering?

    P_ = []
    for P in iP_:
        link_ = []
        for derP in P.link_H[-1]:  # scan last-layer links
            if derP.valt[0] > P_aves[0] * derP.rdnt[0]:
                _P = derP._P
                for _derP in _P.link_H[-1]:  # next layer of all links, also lower-layer?
                   if _derP.valt[0] >  P_aves[0]* _derP.rdnt[0]:
                        __P = _derP._P  # next layer of Ps
                        distance = np.hypot(__P.yx[1]-P.yx[1], __P.yx[0]-P.yx[0])   # distance between midpoints
                        if distance > rng:  # distance=S, mostly lateral, /= L for eval?
                            comp_P(link_,__P,P, rn=len(__P.dert_)/len(P.dert_), fd=0, derP=distance)

        P.link_H += [link_]  # add new link layer, in rng+ only
        P_ += [P]

    return P_

def comp_der(P_):  # keep same Ps and links, increment link derH, then P derH in sum2PP

    for P in P_:
        if not P.derH: continue
        link_ = []
        for derP in P.link_H[-1]:  # scan root PP links, no concurrent rng+
            if derP._P in P_ and derP._P.derH and derP.valt[1] > P_aves[1] * derP.rdnt[1]:
                _P = derP._P
                # comp extended derH of previously compared Ps, sum in lower-composition sub_PPs,
                # weight of compared derH is relative compound scope of (sum linked Ps( sum P derts)):
                rn = (len(_P.dert_) / len(P.dert_)) * (len(_P.link_H[-1]) / len(P.link_H[-1]))
                comp_P(link_,_P,P, rn, fd=1, derP=derP)

        P.link_H[-1] = link_  # replace with extended-derH derPs
    return P_


def form_PP_t(root, P_, base_rdn):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    for fd in 0,1:
        link_map = defaultdict(list)
        for P in P_:
            for derP in P.link_H[-1]:
                if derP.valt[fd] > P_aves[fd] * derP.rdnt[fd]:
                    link_map[P] += [derP._P]  # keys:Ps, vals: linked _P_s, up and down
                    link_map[derP._P] += [P]
        for P in P_:
            if P.root_t[fd]: continue  # skip if already packed in some PP
            cP_ = [P]  # clustered Ps and their val,rdn s
            perimeter = deque(link_map[P])  # recycle with breadth-first search, up and down:
            Val,Rdn = 0,0
            while perimeter:
                _P = perimeter.popleft()
                if _P in cP_: continue
                for derP in _P.link_H[-1]:
                    if derP._P in cP_: continue  # circular link? or derP._P in cP_?
                    _val, _rdn = derP.valt[fd], derP.rdnt[fd]
                    if _val > P_aves[fd]/2 * _rdn:  # no interference by -ve links? lower filter for link vs. P
                        Val += _val; Rdn += _rdn
                        cP_ += [_P]
                        perimeter += link_map[_P]  # append linked __Ps to extended perimeter of P
            PP_t[fd] += [sum2PP(root, cP_, base_rdn, fd)]  # no if Val > PP_aves[fd] * Rdn:

    for fd, PP_ in enumerate(PP_t):   # after form_PP_t -> P.root_t
        sub_recursion(root, PP_, fd)  # eval rng+/ PPm or der+/ PPd
        if root.fback_t and root.fback_t[fd]:
            feedback(root, fd)  # after sub+ in all nodes, no single node feedback up multiple layers

    root.node_t = PP_t  # PPs maybe nested in sub+, add_alt_PPs_?


def sum2PP(root, P_, base_rdn, fd):  # sum links in Ps and Ps in PP

    PP = CPP(fd=fd, root=root, node_t=P_, rng=root.rng +(1-fd))   # initial PP.box = (inf,inf,-inf,-inf)
    # accum:
    celly_,cellx_ = [],[]
    for P in P_:
        P.root_t[fd] = PP
        sum_ptuple(PP.ptuple, P.ptuple)   # accum ptuple
        for y, x in P.cells:
            PP.box = PP.box.accumulate(y,x)
            celly_ += [y]; cellx_ += [x]

        for derP in P.link_H[-1]:
            if derP.valt[fd] > P_aves[fd] * derP.rdnt[fd]:
                derH, valt, rdnt = derP.derH, derP.valt, derP.rdnt
                sum_derH([P.derH,P.valt,P.rdnt], [derH,valt,rdnt], base_rdn, fneg=0)  # uplink
                _P = derP._P  # bilateral accum downlink, reverse d signs:
                sum_derH([_P.derH,_P.valt,_P.rdnt], [derH,valt,rdnt], base_rdn, fneg=1)
        # unilateral sum:
        sum_derH([PP.derH,PP.valt,PP.rdnt], [P.derH,P.valt,P.rdnt], base_rdn)
    y0, x0, yn, xn = PP.box
    PP.mask__ = np.zeros((yn-y0, xn-x0), bool)
    celly_ = np.array(celly_); cellx_ = np.array(cellx_)
    PP.mask__[(celly_-y0, cellx_-x0)] = True
    return PP

'''
Each call to comp_rng | comp_der forms dderH: new layer of derH. Both forks are merged in feedback to contain complexity
(deeper layers are appended by feedback, if nested we need fback_tree: last_layer_nforks = 2^n_higher_layers)
'''
def sub_recursion(root, PP_, fd):  # called in form_PP_, evaluate PP for rng+ and der+, add layers to select sub_PPs

    for PP in PP_:
        P_ = PP.node_t  # flat before sub+
        rng = PP.rng+(1-fd)
        if PP.valt[fd] * (len(P_)-1)*rng > PP_aves[fd] * PP.rdnt[fd]:  # val*len*rng: sum ave matches, - fixed PP costs?
            # der+|rng+:
            comp_der(P_) if fd else comp_rng(P_, rng)  # same else new links
            PP.rdnt[fd] += PP.valt[fd] - PP_aves[fd] * PP.rdnt[fd] > PP.valt[1-fd] - PP_aves[1-fd] * PP.rdnt[1-fd]
            for P in P_: P.root_t = [[],[]]  # fill with sub_PPm_,sub_PPd_ between nodes and PP:
            form_PP_t(PP, P_, base_rdn=PP.rdnt[fd])
            root.fback_t[fd] += [[PP.derH, PP.valt, PP.rdnt]]  # merge in root.fback_t fork, else need fback_tree


def feedback(root, fd):  # in form_PP_, append new der layers to root PP, single vs. root_ per fork in agg+

    Fback = root.fback_t[fd].pop(0)  # init with 1st [derH,valt,rdnt]
    while root.fback_t[fd]:
        sum_derH(Fback, root.fback_t[fd].pop(0), base_rdn=0)
    sum_derH([root.derH, root.valt, root.rdnt], Fback, base_rdn=0)  # both fder forks sum into a same root

    if isinstance(root, CPP):  # root is not CEdge, which has no roots
        rroot = root.root  # single PP.root, can't be P
        fd = root.fd  # current node_ fd
        fback_ = rroot.fback_t[fd]
        fback_ += [Fback]
        if fback_ and (len(fback_) == len(rroot.node_t)):  # still flat, all nodes terminated and fed back
            feedback(rroot, fd)  # sum2PP adds derH per rng, feedback adds deeper sub+ layers


def sum_derH(T, t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    DerH, Valt, Rdnt = T; derH, valt, rdnt = t
    for i in 0,1:
        Valt[i] += valt[i]
        Rdnt[i] += rdnt[i] + base_rdn
    DerH[:] = [
        # sum der layers, dertuple is mtuple | dtuple, fneg*i: for dtuple only:
        [ [sum_dertuple(Dertuple,dertuple, fneg*i) for i,(Dertuple,dertuple) in enumerate(zip(Tuplet,tuplet))],
          [Val + val for Val, val in zip(Valt, valt)], [Rdn + rdn + base_rdn for Rdn, rdn in zip(Rdnt,rdnt)]
        ]
        for [Tuplet,Valt,Rdnt], [tuplet,valt,rdnt]
        in zip_longest(DerH, derH, fillvalue=[([0,0,0,0,0,0],[0,0,0,0,0,0]), (0,0),(0,0)])  # ptuplet, valt, rdnt
    ]

def sum_ptuple(Ptuple, ptuple, fneg=0):
    I, G, M, Ma, (Dy, Dx), L = Ptuple
    _I, _G, _M, _Ma, (_Dy, _Dx), _L = ptuple
    if fneg: Ptuple[:] = (_I-I, _G-G, _M-M, _Ma-Ma, [_Dy-Dy,_Dx-Dx], _L-L)
    else:    Ptuple[:] = (_I+I, _G+G, _M+M, _Ma+Ma, [_Dy+Dy,_Dx+Dx], _L+L)

def sum_dertuple(Ptuple, ptuple, fneg=0):
    I, G, M, Ma, A, L = Ptuple
    _I, _G, _M, _Ma, _A, _L = ptuple
    if fneg: Ptuple[:] = [_I-I, _G-G, _M-M, _Ma-Ma, _A-A, _L-L]
    else:    Ptuple[:] = [_I+I, _G+G, _M+M, _Ma+Ma, _A+A, _L+L]
    return   Ptuple

def comp_derH(_derH, derH, rn, fagg=0):  # derH is a list of der layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    dderH = []  # or not-missing comparand: xor?
    Mval, Dval, Mrdn, Drdn = 0,0,1,1
    if fagg: maxM,maxD = 0,0

    for _lay, lay in zip_longest(_derH, derH, fillvalue=[]):  # compare common lower der layers | sublayers in derHs
        if _lay and lay:  # also if lower-layers match: Mval > ave * Mrdn?

            ret = comp_dtuple(_lay[0][1], lay[0][1], rn, fagg)  # compare dtuples only, mtuples are for evaluation
            mtuple, dtuple = ret[:2]
            mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
            mrdn = dval > mval; drdn = dval < mval
            derLay = [[mtuple,dtuple],[mval,dval],[mrdn,drdn]]
            Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn
            if fagg:
                Mtuple, Dtuple = ret[2:]
                maxm = sum(Mtuple); maxd = sum(Dtuple)
                maxM+= maxm; maxD+= maxd
                derLay += [[maxm,maxd]]  # or += [Mtuple,Dtuple] for future comp?
            dderH += [derLay]

    ret = [dderH, [Mval,Dval], [Mrdn,Drdn]]  # new derLayer,= 1/2 combined derH
    if fagg: ret += [[maxM,maxD]]
    return ret


def comp_dtuple(_ptuple, ptuple, rn, fagg=0):

    mtuple, dtuple = [],[]
    if fagg: Mtuple, Dtuple = [],[]

    for _par, par, ave in zip(_ptuple, ptuple, aves):  # compare ds only
        npar = par * rn
        mtuple += [matchF(_par, npar) - ave]
        dtuple += [_par - npar]
        if fagg:
            Mtuple += [max(abs(_par),abs(npar))]
            Dtuple += [abs(_par)+abs(npar)]
    ret = [mtuple, dtuple]
    if fagg: ret += [Mtuple, Dtuple]
    return ret

def matchF(_par, par):
    match = min(abs(_par),abs(par))
    return -match if (_par<0) != (par<0) else match    # match = neg min if opposite-sign comparands

def comp_ptuple(_ptuple, ptuple, rn, fagg=0):  # 0der params

    I, G, M, Ma, (Dy, Dx), L = _ptuple
    _I, _G, _M, _Ma, (_Dy, _Dx), _L = ptuple

    dI = _I - I*rn;  mI = ave-dI
    dG = _G - G*rn;  mG = min(_G, G*rn) - ave
    dL = _L - L*rn;  mL = min(_L, L*rn) - ave
    dM = _M - M*rn;  mM = matchF(_M, M*rn) - ave  # M, Ma may be negative
    dMa= _Ma- Ma*rn; mMa = matchF(_Ma, Ma*rn) - ave
    mAngle, dAngle = comp_angle((_Dy,_Dx), (Dy,Dx))

    mtuple = [mI, mG, mM, mMa, mAngle, mL]
    dtuple = [dI, dG, dM, dMa, dAngle, dL]
    if fagg:
        Mtuple = [max(_I,I), max(_G,G), max(_M,M), max(_Ma,Ma), 2, max(_L,L)]
        Dtuple = [abs(_I)+abs(I), abs(_G)+abs(G), abs(_M)+abs(M), abs(_Ma)+abs(Ma), 2, abs(_L)+abs(L)]

    ret = [mtuple, dtuple]
    if fagg: ret += [Mtuple, Dtuple]
    return ret


def comp_ptuple_gen(_ptuple, ptuple, rn):  # 0der

    mtuple, dtuple, Mtuple = [],[],[]
    # _n, n = _ptuple, ptuple: add to rn?
    for i, (_par, par, ave) in enumerate(zip(_ptuple, ptuple, aves)):
        if isinstance(_par, list) or isinstance(_par, tuple):
             m,d = comp_angle(_par, par)
             maxv = 2
        else:  # I | M | G L
            npar= par*rn  # accum-normalized par
            d = _par - npar
            if i: m = min(_par,npar)-ave
            else: m = ave-abs(d)  # inverse match for I, no mag/value correlation
            maxv = max(_par, par)
        mtuple+=[m]
        dtuple+=[d]
        Mtuple+=[maxv]
    return [mtuple, dtuple, Mtuple]

def sum_derH_gen(T, t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    DerH, Valt, Rdnt = T; derH, valt, rdnt = t
    for i in 0, 1:
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i] + base_rdn
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