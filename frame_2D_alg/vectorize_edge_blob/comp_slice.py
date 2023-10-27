import numpy as np
from copy import copy, deepcopy
from itertools import zip_longest, combinations
from collections import deque, defaultdict
from .slice_edge import comp_angle
from .classes import CderP, CPP
from .filters import ave, aves, P_aves, PP_aves

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

def comp_P_(edge, adj_Pt_):  # cross-comp P_ in edge: high-gradient blob, sliced in Ps in the direction of G

    for _P, P in adj_Pt_:  # scan, comp contiguously uplinked Ps, rn: relative weight of comparand
        comp_P(edge.link_, _P,P, rn=len(_P.dert_)/len(P.dert_), fd=0)

    form_PP_t(edge, P_=edge.node_, base_rdn=2)  # replace edge.node_ with PP_t, may be nested by sub+


def comp_P(link_,_P, P, rn, fd=1, derP=None):  #  derP if der+, reused as S if rng+
    aveP = P_aves[fd]

    if fd:  # der+: extend in-link derH, in sub+ only
        dderH, valt, rdnt = comp_derH(_P.derH, P.derH, rn)  # += fork rdn
        derP = CderP(derH = derP.derH+dderH, valt=valt, rdnt=rdnt, P=P,_P=_P, S=derP.S)  # dderH valt,rdnt for new link
        mval,dval = valt[:2]; mrdn,drdn = rdnt  # exclude maxv

    else:  # rng+: add derH
        mtuple,dtuple = comp_ptuple(_P.ptuple, P.ptuple, rn)
        mval = sum(mtuple); dval = sum(dtuple)
        mrdn = 1+(dval>mval); drdn = 1+(1-(dval>mval))  # or rdn = Dval/Mval?
        derP = CderP(derH=[[mtuple,dtuple]], valt=[mval,dval], rdnt=[mrdn,drdn], P=P,_P=_P, S=derP)

    if mval > aveP*mrdn or dval > aveP*drdn:
        link_ += [derP]

# rng+ and der+ are called from sub_recursion:
def comp_rng(ilink_, rng):  # form new Ps and links, switch to rng+n to skip clustering?

    link_ = []  # rng+ links
    for _derP, derP in combinations(ilink_, 2):  # scan last-layer link pairs
        _P = _derP.P; P = derP.P
        if _derP.P is not derP._P: continue  # same as derP._P is _derP._P or derP.P is _derP.P
        __P = _derP._P  # next layer of Ps
        distance = np.hypot(__P.yx[1]-P.yx[1],__P.yx[0]-P.yx[0])   # distance between midpoints
        if distance < rng:  # distance=S, mostly lateral, /= L for eval?
            comp_P(link_, __P, P, rn=len(__P.dert_)/len(P.dert_), fd=0, derP=distance)

    return link_

def comp_der(ilink_):  # keep same Ps and links, increment link derH, then P derH in sum2PP

    # compute number of uplinks per P
    n_uplinks = defaultdict(int)
    for derP in ilink_: n_uplinks[derP.P] += 1

    link_ = []     # extended-derH derPs
    for derP in ilink_:  # scan root PP links, no concurrent rng+
        P = derP.P; _P = derP._P
        if not P.derH or not _P.derH: continue
        # comp extended derH of previously compared Ps, sum in lower-composition sub_PPs,
        # weight of compared derH is relative compound scope of (sum linked Ps( sum P derts)):
        rn = (len(_P.dert_) / len(P.dert_)) * (n_uplinks[_P] / n_uplinks[P])
        comp_P(link_, _P, P, rn, fd=1, derP=derP)

    return link_


def form_PP_t(root, P_, base_rdn):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    for fd in 0,1:
        link_map = defaultdict(list)
        derP_ = []
        for derP in root.link_:
            if derP.valt[fd] > P_aves[fd] * derP.rdnt[fd]:
                link_map[derP.P] += [derP._P]  # keys:Ps, vals: linked _P_s, up and down
                link_map[derP._P] += [derP.P]
                derP_ += [derP]     # filtered derP
        for P in P_:
            if P.root_t[fd]: continue  # skip if already packed in some PP
            cP_ = [P]  # clustered Ps and their val,rdn s
            perimeter = deque(link_map[P])  # recycle with breadth-first search, up and down:
            while perimeter:
                _P = perimeter.popleft()
                if _P in cP_: continue
                for derP in derP_:
                    if derP.P is not _P: continue
                    if derP._P in cP_: continue  # circular link? or derP._P in cP_?
                    if derP.valt[fd] > P_aves[fd]/2 * derP.rdnt[fd]:  # no interference by -ve links? lower filter for link vs. P
                        cP_ += [_P]
                        perimeter += link_map[_P]  # append linked __Ps to extended perimeter of P
                        break   # break to avoid duplicate P
            PP = sum2PP(root, cP_, derP_, base_rdn, fd)
            PP_t[fd] += [PP]  # no if Val > PP_aves[fd] * Rdn:

            # after form_PP_t -> P.root_t
            sub_recursion(root, PP, fd)  # eval rng+/ PPm or der+/ PPd

    if root.fback_t:
        for fd in 0,1:
            if root.fback_t[fd]:
                feedback(root, fd)  # after sub+ in all nodes, no single node feedback up multiple layers

    root.node_ = PP_t  # PPs maybe nested in sub+, add_alt_PPs_?


def sum2PP(root, P_, derP_, base_rdn, fd):  # sum links in Ps and Ps in PP

    PP = CPP(fd=fd, root=root, node_=P_, rng=root.rng +(1-fd))  # initial PP.box = (inf,inf,-inf,-inf)

    # accum derP:
    for derP in derP_:
        if derP.P not in P_ or derP._P not in P_: continue
        derH,valt,rdnt = derP.derH,derP.valt,derP.rdnt
        P = derP.P
        sum_derH([P.derH,P.valt,P.rdnt], [derH,valt,rdnt], base_rdn, fneg=0)  # uplink
        _P = derP._P  # bilateral accum downlink, reverse d signs:
        sum_derH([_P.derH,_P.valt,_P.rdnt], [derH,valt,rdnt], base_rdn, fneg=1)  # downlink
        PP.link_ += [derP]
    # accum P:
    celly_,cellx_ = [],[]
    for P in P_:
        P.root_t[fd] = PP
        sum_ptuple(PP.ptuple, P.ptuple)  # accum ptuple
        for y, x in P.cells:
            PP.box = PP.box.accumulate(y, x)
            celly_ += [y]; cellx_ += [x]
        # unilateral sum:
        sum_derH([PP.derH, PP.valt, PP.rdnt], [P.derH, P.valt, P.rdnt], base_rdn)

    y0, x0, yn, xn = PP.box
    PP.mask__ = np.zeros((yn-y0, xn-x0), bool)
    celly_ = np.array(celly_); cellx_ = np.array(cellx_)
    PP.mask__[(celly_-y0, cellx_-x0)] = True  # assign PP mask

    return PP

'''
Each call to comp_rng | comp_der forms dderH: new layer of derH. Both forks are merged in feedback to contain complexity
(deeper layers are appended by feedback, if nested we need fback_tree: last_layer_nforks = 2^n_higher_layers)
'''
def sub_recursion(root, PP, fd):  # called in form_PP_, evaluate PP for rng+ and der+, add layers to select sub_PPs

    P_ = PP.node_  # flat before sub+
    rng = PP.rng+(1-fd)
    if PP.valt[fd] * (len(P_)-1)*rng <= PP_aves[fd] * PP.rdnt[fd]: return    # val*len*rng: sum ave matches, - fixed PP costs?
    # der+|rng+:
    PP.link_ = comp_der(PP.link_) if fd else comp_rng(PP.link_, rng)  # same else new links
    PP.rdnt[fd] += (PP.valt[fd] - PP_aves[fd] * PP.rdnt[fd]) > (PP.valt[1-fd] - PP_aves[1-fd] * PP.rdnt[1-fd])
    for P in P_: P.root_t = [None,None]  # fill with sub_PPm_,sub_PPd_ between nodes and PP:
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
        if fback_ and (len(fback_) == len(rroot.node_)):  # still flat, all nodes terminated and fed back
            feedback(rroot, fd)  # sum2PP adds derH per rng, feedback adds deeper sub+ layers


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

def sum_ptuple(Ptuple, ptuple, fneg=0):
    I, G, M, Ma, (Dy, Dx), L = ptuple
    _I, _G, _M, _Ma, (_Dy, _Dx), _L = Ptuple
    if fneg: Ptuple[:] = (_I-I, _G-G, _M-M, _Ma-Ma, [_Dy-Dy,_Dx-Dx], _L-L)
    else:    Ptuple[:] = (_I+I, _G+G, _M+M, _Ma+Ma, [_Dy+Dy,_Dx+Dx], _L+L)

def sum_dertuple(Ptuple, ptuple, fneg=0):
    I, G, M, Ma, A, L = ptuple
    _I, _G, _M, _Ma, _A, _L = Ptuple
    if fneg: Ptuple[:] = [_I-I, _G-G, _M-M, _Ma-Ma, _A-A, _L-L]
    else:    Ptuple[:] = [_I+I, _G+G, _M+M, _Ma+Ma, _A+A, _L+L]
    return   Ptuple

def comp_derH(_derH, derH, rn):  # derH is a list of der layers or sub-layers, each = ptuple_tv

    dderH = []  # or not-missing comparand: xor?
    Mval, Dval, Mrdn, Drdn = 0,0,1,1

    for _lay, lay in zip_longest(_derH, derH, fillvalue=[]):  # compare common lower der layers | sublayers in derHs
        if _lay and lay:  # also if lower-layers match: Mval > ave * Mrdn?
            # compare dtuples only:
            mtuple, dtuple = comp_dtuple(_lay[1], lay[1], rn, fagg=0)
            mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
            mrdn = dval > mval; drdn = dval < mval
            Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn
            dderH += [[mtuple, dtuple]]

    return dderH, [Mval,Dval], [Mrdn,Drdn]  # new derLayer,= 1/2 combined derH


def comp_dtuple(_ptuple, ptuple, rn, fagg=0):

    mtuple, dtuple = [],[]
    if fagg: Mtuple, Dtuple = [],[]

    for _par, par, ave in zip(_ptuple, ptuple, aves):  # compare ds only
        npar = par * rn
        mtuple += [match_func(_par, npar) - ave]
        dtuple += [_par - npar]
        if fagg:
            Mtuple += [max(abs(_par),abs(npar))]
            Dtuple += [abs(_par)+abs(npar)]
    ret = [mtuple, dtuple]
    if fagg: ret += [Mtuple, Dtuple]
    return ret

def match_func(_par, par):
    match = min(abs(_par),abs(par))
    return -match if (_par<0) != (par<0) else match    # match = neg min if opposite-sign comparands

def comp_ptuple(_ptuple, ptuple, rn, fagg=0):  # 0der params

    I, G, M, Ma, (Dy, Dx), L = _ptuple
    _I, _G, _M, _Ma, (_Dy, _Dx), _L = ptuple

    dI = _I - I*rn;  mI = ave-dI
    dG = _G - G*rn;  mG = min(_G, G*rn) - ave
    dL = _L - L*rn;  mL = min(_L, L*rn) - ave
    dM = _M - M*rn;  mM = match_func(_M, M*rn) - ave  # M, Ma may be negative
    dMa= _Ma- Ma*rn; mMa = match_func(_Ma, Ma*rn) - ave
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