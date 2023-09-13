import numpy as np
from copy import copy, deepcopy
from itertools import zip_longest
from collections import deque
from .slice_edge import comp_angle
from .classes import CderP, CPP
from .filters import ave, aves, vaves, med_decay, aveB, P_aves, PP_aves, ave_nsubt
from dataclasses import replace

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


def comp_P_(edge):  # renamed for consistency, cross-comp P_ in edge: high-gradient blob, sliced in Ps in the direction of G

    P_ = edge.node_tt  # init as P_
    edge.node_tt = ([[],[]], [])  # (rng+, der+(always empty))
    # ~ sub+:
    for P in P_:  # scan and compare contiguously uplinked Ps, rn = relative weight of comparand:

        derP_ = [comp_P(_P, P, rn=len(_P.dert_)/len(P.dert_), fder=0) for _P in P.link_H[-1]]
        P.link_H[-1] = [derP for derP in derP_ if derP is not None]  # replace link _Ps with derPs

    for fd in 0,1:  # replace P_ with PP_t, root fork is rng+ only:
        form_PP_(edge, P_, base_rdn=2, fder=0, fd=fd)  # may be nested by sub+ in form_PP_


def comp_P(_P,P, rn, fder=1, derP=None):  #  derP if der+, reused as S if rng+
    aveP = P_aves[fder]

    if fder:  # der+: extend in-link derH, in sub+ only
        dderH, valt, rdnt = comp_derH(_P.derH, P.derH, rn)  # += fork rdn
        derP = CderP(derH = derP.derH+dderH, valt=valt, rdnt=rdnt, P=P,_P=_P, S=derP.S)  # dderH valt,rdnt for new link
        mval,dval = valt[:2]; mrdn,drdn = rdnt  # exclude maxv

    else:  # rng+: add derH
        mtuple,dtuple,Mtuple = comp_ptuple(_P.ptuple, P.ptuple, rn)
        mval = sum(mtuple); dval = sum(dtuple); maxv = sum(Mtuple)
        mrdn = 1+(dval>mval); drdn = 1+(1-(dval>mval))  # or rdn = Dval/Mval?
        derP = CderP(derH=[[[mtuple,dtuple], [mval,dval],[mrdn,drdn]]], valt=[mval,dval,maxv], rdnt=[mrdn,drdn], P=P,_P=_P, S=derP)

    if mval > aveP*mrdn or dval > aveP*drdn:
        return derP

# rng+ and der+ are called from sub_recursion
def comp_rng(iP_, rng):  # form new Ps and links, switch to rng+n to skip clustering?

    P_, derP_ = [],[]

    for P in iP_:
        for derP in P.link_H[-1]:  # scan last-layer links
            if derP.valt[0] >  P_aves[0] * derP.rdnt[0]:
                _P = derP._P
                for _derP in _P.link_H[-1]:  # next layer of all links, also lower-layer?
                   if _derP.valt[0] >  P_aves[0]* _derP.rdnt[0]:
                        __P = _derP._P  # next layer of Ps
                        distance = np.hypot(__P.yx[1]-P.yx[1], __P.yx[0]-P.yx[0])   # distance between midpoints
                        if distance > rng:  # distance=S, mostly lateral, /= L for eval?
                            derP = comp_P(__P,P, rn=len(__P.dert_)/len(P.dert_), fder=0, derP=distance)
                            if derP: derP_ += [derP]  # not None

        P.link_H += [derP_]  # add new link layer, in rng+ only
        P_ += [P]

    return P_

def comp_der(P_, frng):  # keep same Ps and links, increment link derH, then P derH in sum2PP

    derP_ = []
    for P in P_:
        link_ = P.link_H[-(1+frng)]
        for derP in link_:  # scan root-PP links, exclude top layer if formed by concurrent rng+
            if derP.valt[1] >  P_aves[1]* derP.rdnt[1]:
                _P = derP._P  # comp extended derH of previously compared Ps, sum in lower-composition sub_PPs
                # weight of compared derH is relative compound scope of (sum linked Ps( sum P derts)):
                rn = (len(_P.dert_) / len(P.dert_)) * (len(_P.link_H[-(1+frng)]) / len(link_))
                derP = comp_P(_P,P, rn, fder=1, derP=derP)
                if derP: derP_ += [derP]  # not None
        link_[:] = derP_  # replace with extended-derH derPs
    return P_


# tentative:
def form_PP_(root, P_, base_rdn, fder, fd):  # form PPs of derP.valt[fd] + connected Ps val

    qPP_ = []  # initial pre_PPs are in list format
    for P in P_:
        if P.root_tt[fder][fd]:  continue  # else already packed in some qPP
        qPP = [[P]]  # init PP is 2D queue of (P,val)s of all layers, assigned in the end
        P.root_tt[fder][fd] = qPP; val = 0
        uplink_ = P.link_H[-1]
        uuplink_ = []  # next layer of uplinks
        while uplink_:  # test for next-line uuplink_, set at loop's end
            for derP in uplink_:
                if derP.valt[fder] <= P_aves[fder]*derP.rdnt[fder]: continue  # link _P should not be in qPP
                _P = derP._P
                if _P not in P_:  # _P is outside qPP, add it
                    _PP = _P.root_tt[fder][fd]
                    if _PP:  # _P was clustered as P in prior loops
                        for __P in _PP[0]:  # merge _PP into qPP
                            qPP[0] += [__P]; __P.root_tt[fder][fd] = qPP
                        qPP_.remove(_PP)
                else:  # _P is in qPP
                    _qPP = _P.root_tt[fder][fd]
                    if _qPP:
                        if _qPP is not qPP:  # _P may be added to qPP via other down-linked P
                            val += _qPP[1]  # merge _qPP in qPP:
                            for qP in _qPP[0]:
                                qP.root_tt[fder][fd] = qPP
                                qPP[0] += [qP] # qP_+=[qP]
                            qPP_.remove(_qPP)
                    else:
                        qPP[0] += [_P]  # pack bottom up
                        _P.root_tt[fder][fd] = qPP
                        val += derP.valt[fd]
                        uuplink_ += derP._P.link_H[-1]
            uplink_ = uuplink_
            uuplink_ = []
        qPP += [val, ave+1]  # ini reval=ave+1, keep qPP same object for ref in P.root_tt
        qPP_ += [qPP]

    rePP_ = reval_PP_(qPP_, fder, fd)  # prune qPPs by mediated links vals, PP = [qPP,valt,reval]
    PP_ = [sum2PP(root, qPP, base_rdn, fder, fd) for qPP in rePP_]

    sub_recursion(root.fback_tt[fder][fd], PP_)  # eval rng+,der+ per PP.P_
    if root.fback_tt and root.fback_tt[fder][fd]:
        feedback(root, fder, fd)  # feedback after sub+ is terminated in all root fork nodes, to avoid individual traffic

    root.node_tt[fder][fd] = PP_  # PPs maybe nested in sub+, revert node_tt if empty, add_alt_PPs_(graph_t)?


def reval_PP_(PP_, fder, fd):  # recursive eval / prune Ps for rePP

    rePP_ = []
    while PP_:  # init P__
        P_, val, reval = PP_.pop(0)
        # Ave = ave * 1+(valt[fd] < valt[1-fd])  # * PP.rdn for more selective clustering?
        if val > ave:
            if reval < ave:  # same graph, skip re-evaluation:
                rePP_ += [[P_,val,0]]  # reval=0
            else:
                rePP = reval_P_(P_,fd)  # recursive node and link revaluation by med val
                if val > ave:  # min adjusted val
                    rePP_ += [rePP]
                else:
                    for P in rePP: P.root_tt[fder][fd] = []
        else:  # low-val qPPs are removed
            for P in P_: P.root_tt[fder][fd] = []

    if rePP_ and max([rePP[2] for rePP in rePP_]) > ave:  # recursion if any min reval:
        rePP_ = reval_PP_(rePP_, fder, fd)

    return rePP_

# draft:
def reval_P_(P_, fd):  # prune qPP by link_val + mediated link__val

    Val, reval = 0,0  # comb PP value and recursion value
    for P in P_:
        P_val = 0
        for link in P.link_H[-1]:  # link val + med links val: single mediation layer in comp_slice:
            link_val = link.valt[fd] + sum([link.valt[fd] for link in link._P.link_H[-1]]) * med_decay
            if link_val >= vaves[fd]:
                P_val += link_val
        if P_val * P.rdnt[fd] > vaves[fd]:
            Val += P_val * P.rdnt[fd]

    if reval > aveB:
        P_, Val, reval = reval_P_(P_, fd)  # recursion
    return [P_, Val, reval]


def sum2PP(root, pre_PP, base_rdn, fder, fd):  # sum links in Ps and Ps in PP

    P_,_,_ = pre_PP  # proto-PP is a list
    PP = CPP(fd=fd, node_tt=P_)
    PP.root_tt[fder][fd] = root
    # accum:
    for i, P in enumerate(P_):
        P.root_tt[fder][fd] = PP
        sum_ptuple(PP.ptuple, P.ptuple)
        L = P.ptuple[-1]
        Dy = P.axis[0]*L/2; Dx = P.axis[1]*L/2; y,x =P.yx
        if i: Y0=min(Y0,(y-Dy)); Yn=max(Yn,(y+Dy)); X0=min(X0,(x-Dx)); Xn=max(Xn,(x+Dx))
        else: Y0=y-Dy; Yn=y+Dy; X0=x-Dx; Xn=x+Dx  # init

        for derP in P.link_H[-1]:
            if derP.valt[fder] > P_aves[fder]* derP.rdnt[fder]:
                derH, valt, rdnt = derP.derH, derP.valt, derP.rdnt
                sum_derH([P.derH,P.valt,P.rdnt], [derH,valt,rdnt], base_rdn)
                _P = derP._P  # bilateral summation:
                sum_derH([_P.derH,_P.valt,_P.rdnt], [derH,valt,rdnt], base_rdn)
        # excluding bilateral sums:
        sum_derH([PP.derH,PP.valt,PP.rdnt], [P.derH,P.valt,P.rdnt], base_rdn)

    PP.box =(Y0,Yn,X0,Xn)
    return PP


def sum_derH(T, t, base_rdn):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    DerH, Valt, Rdnt = T
    derH, valt, rdnt = t
    for i in 0, 1:
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i] + base_rdn
    if DerH:
        for Layer, layer in zip_longest(DerH,derH, fillvalue=[]):
            if layer:
                if Layer:
                    for i in 0,1:
                        sum_ptuple(Layer[0][i], layer[0][i])  # ptuplet
                        Layer[1][i] += layer[1][i]  # valt
                        Layer[2][i] += layer[2][i] + base_rdn  # rdnt
                else:
                    DerH += [deepcopy(layer)]
    else:
        DerH[:] = deepcopy(derH)

def sum_ptuple(Ptuple, ptuple, fneg=0):

    for i, (Par, par) in enumerate(zip_longest(Ptuple, ptuple, fillvalue=None)):
        if par != None:
            if Par != None:
                if isinstance(Par, list) or isinstance(Par, tuple):  # angle or aangle
                    for i,(P,p) in enumerate(zip(Par,par)):
                        Par[i] = P-p if fneg else P+p
                else:
                    Ptuple[i] += (-par if fneg else par)  # now includes n in ptuple[-1]?
            elif not fneg:
                Ptuple += [copy(par)]

def comp_derH(_derH, derH, rn):  # derH is a list of der layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    dderH = []  # or = not-missing comparand if xor?
    Mval, Dval, Mrdn, Drdn, Maxv = 0,0,1,1,0

    for _lay, lay in zip_longest(_derH, derH, fillvalue=[]):  # compare common lower der layers | sublayers in derHs
        if _lay and lay:  # also if lower-layers match: Mval > ave * Mrdn?

            mtuple, dtuple, Mtuple = comp_dtuple(_lay[0][1], lay[0][1], rn)  # compare dtuples only, mtuples are for evaluation
            mval = sum(mtuple); dval = sum(dtuple); maxv = sum(Mtuple)
            mrdn = dval > mval; drdn = dval < mval
            dderH += [[[mtuple,dtuple],[mval,dval,maxv],[mrdn,drdn]]]
            Mval+=mval; Dval+=dval; Maxv+=maxv; Mrdn+=mrdn; Drdn+=drdn

    return dderH, [Mval,Dval,Maxv], [Mrdn,Drdn]  # new layer, 1/2 combined derH

def comp_dtuple(_ptuple, ptuple, rn):

    mtuple, dtuple, Mtuple = [],[], []
    for _par, par, ave in zip(_ptuple, ptuple, aves):  # compare ds only?
        npar= par*rn
        mtuple += [min(_par, npar) - ave]
        dtuple += [_par - npar]
        Mtuple += [max(_par, npar)]

    return [mtuple, dtuple, Mtuple]

def comp_ptuple(_ptuple, ptuple, rn):  # 0der

    mtuple, dtuple, Mtuple = [],[], []
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

'''
Each call to comp_rng | comp_der forms dderH: a layer of derH
Comp fork fder and clustering fork fd are not represented in derH because they are merged in feedback, to control complexity
(deeper layers are appended by feedback, if nested we need fback_tree: last_layer_nforks = 2^n_higher_layers)
'''
def sub_recursion(fback_, PP_):  # called in form_PP_, evaluate PP for rng+ and der+, add layers to select sub_PPs

    for PP in PP_:
        P_ = PP.node_tt  # flat before sub+
        PP.node_tt = (([],[]),([],[]))  # sub_PPm_t, sub_PPd_t, revert if not fr
        fr = 0  # sub+ in any fork

        for fder in 0,1:  # eval all| last layer?
            if PP.valt[fder] * np.sqrt(len(P_)-1) if P_ else 0 > P_aves[fder] * PP.rdnt[fder]:  # comp_der|rng in PP->parLayer

                comp_der(P_, fr) if fder else comp_rng(P_, PP.rng+1)  # same else new P_ and links
                fr = 1  # eval per comp fork, not clustering sub-forks
                PP.rdnt[fder] += PP.valt[fder] - PP_aves[fder] * PP.rdnt[fder] > PP.valt[1-fder] - PP_aves[1-fder] * PP.rdnt[1-fder]
                for fd in 0,1:
                    form_PP_(root=PP, P_=P_, base_rdn=PP.rdnt[fder], fder=fder, fd=fd)
                    fback_ += [[PP.derH, PP.valt, PP.rdnt]]  # merge in root.fback_tt fork: || root.node_tt vs. fback_tree
        if not fr:
            PP.node_tt = P_  # revert if empty node_tt


def feedback(root, fder, fd):  # from form_PP_, append new der layers to root PP, single vs. root_ per fork in agg+

    Fback = root.fback_tt[fder][fd].pop(0)  # init with 1st [derH,valt,rdnt]
    while root.fback_tt[fder][fd]:
        sum_derH(Fback, root.fback_tt[fder][fd].pop(0), base_rdn=0)
    sum_derH([root.derH, root.valt, root.rdnt], Fback, base_rdn=0)  # both fder forks sum into a same root

    if isinstance(root, CPP):  # root is not CEdge, which has no roots
        for fder, root_tt in enumerate(root.root_tt):
            for fd, rroot in enumerate(root.root_t):
                if rroot:  # may be empty if the fork was not taken
                    rroot.fback_tt[fder][fd] += [Fback]
                    fback_ = rroot.fback_tt[fder][fd]
                    if fback_ and (len(fback_) == len(rroot.node_tt[fder][fd])):
                        # all rroot nodes terminated and fed back
                        feedback(rroot, fder, fd) # sum2PP adds derH per rng, feedback adds deeper sub+ layers