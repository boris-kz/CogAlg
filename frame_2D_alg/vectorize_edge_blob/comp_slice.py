import numpy as np
from itertools import zip_longest
from copy import copy, deepcopy
from .classes import CderP, CPP
from .filters import ave, aves, vaves, ave_dangle, ave_daangle,med_decay, aveB, P_aves
from dataclasses import replace

'''
comp_slice traces edge blob axis by cross-comparing vertically adjacent Ps: slices across edge blob, along P.G angle.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)

P: any angle, connectivity in P_ is traced through root_s of derts adjacent to P.dert_, possibly forking. 
len prior root_ sorted by G is rdn of each root, to evaluate it for inclusion in PP, or starting new P by ave*rdn.
'''

def comp_slice(blob, verbose=False):  # high-G, smooth-angle blob, composite dert core param is v_g + iv_ga

    P_ = blob.P_  # must be contiguous, gaps filled after slice_blob

    for P in P_:
        link_,link_m,link_d = [],[],[]  # empty in initial Ps
        derT=[[],[]]; valT=[0,0]; rdnT=[1,1]  # to sum links in comp_P
        for _P in P.link_:
            comp_P(_P,P, link_,link_m,link_d, derT,valT,rdnT, fd=0)
            P.link_ = link_  # convert links from Ps to derPs
        if P.link_:
            P.link_t=[link_m,link_d]
            P.derT=derT; P.valT=valT; P.rdnT=rdnT  # single Mtuple, Dtuple

    PPm_,PPd_ = form_PP_t(P_, base_rdn=2)
    blob.PPm_, blob.PPd_  = PPm_, PPd_


def form_PP_t(P_, base_rdn):  # form PPs of derP.valt[fd] + connected Ps'val

    PP_t = []
    for fd in 0,1:
        qPP_ = []  # initial sequence-PPs
        for P in copy(P_):
            if not P.roott[fd]:  # else already packed in qPP
                qPP = [[[P]]]  # init PP is 2D queue of Ps, + valt of all layers?
                P.roott[fd]=qPP; valt = [0,0]
                uplink_ = P.link_t[fd]; uuplink_ = []
                # next-line links for recursive search
                while uplink_:
                    for derP in uplink_:
                        _P = derP._P; _qPP = _P.roott[fd]
                        if _qPP:  # merge _qPP in qPP:
                            for i in 0, 1: valt[i] += _qPP[1][i]
                            for qP in _qPP.P_:
                                qP.roott[fd] = qPP; qPP[0] += [qP]  # append qP_
                            qPP_.remove(_qPP)
                        else:
                            qPP[0].insert(0,_P)  # pack top down
                            _P.root[fd] = qPP
                            for i in 0,1: valt[i] += np.sum(derP.valT[i])
                            uuplink_ += derP._P.link_t[fd]
                    uplink_ = uuplink_
                    uuplink_ = []
                qPP_ += [qPP + [valt,ave+1]]  # ini reval=ave+1
        # prune qPPs by med links val:
        rePP_= reval_PP_(qPP_, fd)  # PP = [qPP,valt,reval]
        CPP_ = [sum2PP(qPP, base_rdn, fd) for qPP in rePP_]
        PP_t += [CPP_]  # may be empty

    return PP_t  # add_alt_PPs_(graph_t)?


def reval_PP_(PP_, fd):  # recursive eval / prune Ps for rePP

    rePP_ = []
    while PP_:  # init P__
        P_, valt, reval = PP_.pop(0)
        Ave = ave * 1+(valt[fd] < valt[1-fd])  # * PP.rdnT[fd]?
        if valt[fd] > Ave:
            if reval < Ave:  # same graph, skip re-evaluation:
                rePP_ += [[P_,valt,0]]  # reval=0
            else:
                rePP = reval_P_(P_, fd)  # recursive node and link revaluation by med val
                if valt[fd] > Ave:  # min adjusted val
                    rePP_ += [rePP]
    if rePP_ and max([rePP[2] for rePP in rePP_]) > ave:  # recursion if any min reval:
        rePP_ = reval_PP_(rePP_, fd)

    return rePP_


def reval_P_(P_, fd):  # prune qPP by link_val + mediated link__val

    prune_=[]; Valt=[0,0]; reval=0  # comb PP value and recursion value

    for P in P_:
        P_val = 0; remove_ = []
        for link in P.link_t[fd]:  # link val + med links val: single med layer in comp_slice:
            link_val = np.sum(link.valT[fd]) + sum([np.sum(mlink.valT[fd]) for mlink in link._P.link_t[fd]]) * med_decay
            if link_val < vaves[fd]:
                remove_ += [link]; reval += link_val
            else: P_val += link_val
        for link in remove_:
            P.link_t[fd].remove(link)  # prune weak links
        if P_val * np.sum(P.rdnT[fd]) < vaves[fd]:
            prune_ += [P]
        else:
            Valt[fd] += P_val * np.sum(P.rdnT[fd])
    for P in prune_:
        for link in P.link_t[fd]:  # prune direct links only?
            _P = link._P
            _link_ = _P.link_t[fd]
            if link in _link_:
                _link_.remove(link); reval += link.valt[fd]

    if reval > aveB:
        P_, Valt, reval = reval_P_(P_, fd)  # recursion
    return [P_, Valt, reval]


def sum2PP(qPP, base_rdn, fd):  # sum Ps and links into PP

    P_,_,_ = qPP  # proto-PP is a list
    # init:
    P = (P_[0])
    Ptuple, DerT,ValT,RdnT, Link_,Link_m,Link_d, y,x = \
    deepcopy(P.ptuple),deepcopy(P.derT),deepcopy(P.valT),deepcopy(P.rdnT), copy(P.link_),copy(P.link_t[0]),copy(P.link_t[1]), P.y,P.x
    L = Ptuple[7]; Dy = P.axis[0]*L/2; Dx = P.axis[1]*L/2  # side-accumulated sin,cos
    Y0 = y-Dy; Yn = y+Dy; X0 = x-Dx; Xn = x+Dx
    PP = CPP(fd=fd, P_=P_)
    # accum:
    for i, P in enumerate(P_):
        P.roott[fd] = PP
        if i:  # exclude init P
            sum_ptuple(Ptuple, P.ptuple)
            Link_+=P.link_; Link_m+=P.link_t[0]; Link_d+=P.link_t[1]
            L=P.ptuple[7]; Dy=P.axis[0]*L/2; Dx=P.axis[1]*L/2; y=P.y; x=P.x
            Y0=min(Y0,(y-Dy)); Yn=max(Yn,(y+Dy)); X0=min(X0,(x-Dx)); Xn=max(Xn,(x-Dx))
            if P.derT[0]:
                for j in 0,1:
                    if isinstance(P.valT[0], list):  # der+: H = 1fork) 1layer before feedback
                        sum_unpack([DerT[j],ValT[j],RdnT[j]], [P.derT[j],P.valT[j],P.rdnT[j]])
                    else:  # rng+: 1 vertuple
                        sum_ptuple(DerT[j], P.derT[j]); ValT[j]+=P.valT[j]; RdnT[j]+=P.rdnT[j]

    PP.ptuple, PP.derT, PP.valT, PP.rdnT, PP.box, PP.link_, PP.link_t \
    = Ptuple, DerT, ValT, RdnT, (Y0,Yn,X0,Xn), Link_, (Link_m,Link_d)
    return PP


def sum_unpack(Q,q):  # recursive unpack two pairs of nested sequences to sum final ptuples

    Que,Val_,Rdn_ = Q; que,val_,rdn_ = q  # max nesting: H( layer( fork( ptuple|scalar)))
    for i, (Ele,Val,Rdn, ele,val,rdn) in enumerate(zip_longest(Que,Val_,Rdn_, que,val_,rdn_, fillvalue=[])):
        if ele:
            if Ele:
                if isinstance(val,list):  # element is layer or fork
                    sum_unpack([Ele,Val,Rdn], [ele,val,rdn])
                else:  # ptuple
                    Val_[i] += val; Rdn_[i] += rdn
                    sum_ptuple(Ele, ele)
            else:
                Que += [deepcopy(ele)]; Val_+= [deepcopy(val)]; Rdn_+= [deepcopy(rdn)]

def sum_ptuple(Ptuple, ptuple, fneg=0):

    for i, (Par, par) in enumerate(zip_longest(Ptuple, ptuple, fillvalue=None)):
        if par != None:
            if Par != None:
                if isinstance(Par, list):  # angle or aangle
                    for i,(P,p) in enumerate(zip(Par,par)):
                        Par[i] = P-p if fneg else P+p
                else:
                    Ptuple[i] += (-par if fneg else par)
            elif not fneg:
                Ptuple += [copy(par)]


def comp_P(_P,P, link_,link_m,link_d, LayT,ValT,RdnT, fd=0, derP=None):  #  derP if der+, S if rng+

    aveP = P_aves[fd]
    rn = len(_P.dert_)/ len(P.dert_)

    if fd:  # der+: extend old link
        rn *= len(_P.link_t[1]) / len(P.link_t[1])  # derT is summed from links
        # comp last layer:
        layT, valT, rdnT = comp_unpack(_P.derT[1][-1], P.derT[1][-1], rn)  # comp lower lays formed derP.derT
        mval = valT[0][-1][-1]; dval = valT[1][-1][-1]  # should be scalars here
        mrdn = 1+(dval>mval); drdn = 1+(1-(dval>mval))
        for i in 0,1:  # append new layer
            derP.derT[i]+=[layT[i]]; derP.valT[i] += [valT[i]]; derP.rdnT[i] += [rdnT[i]]
    else:
        # rng+: add new link
        mtuple,dtuple = comp_ptuple(_P.ptuple, P.ptuple, rn)
        mval = sum(mtuple); dval = sum(dtuple)
        mrdn = 1+(dval>mval); drdn = 1+(1-(dval>mval))  # rdn = Dval/Mval?
        derP = CderP(derT=[mtuple,dtuple],valT=[mval,dval],rdnT=[mrdn,drdn], P=P,_P=_P, S=derP)
    link_ += [derP]  # all links
    if mval > aveP*mrdn:
        link_m+=[derP]  # +ve links, fork selection in form_PP_t
        if fd: sum_unpack([LayT[0],ValT[0],RdnT[0]], [derP.derT[0][-1],derP.valT[0][-1],derP.rdnT[0][-1]])
        else:
            sum_ptuple(LayT[0],mtuple); ValT[0]+=mval; RdnT[0]+=mrdn
    if dval > aveP*drdn:
        link_d+=[derP]
        if fd: sum_unpack([LayT[1],ValT[1],RdnT[1]], [derP.derT[1][-1],derP.valT[1][-1],derP.rdnT[1][-1]])
        else:
            sum_ptuple(LayT[1],dtuple); ValT[1]+=dval; RdnT[1]+=drdn


def comp_unpack(Que,que, rn):  # recursive unpack nested sequence to compare final ptuples

    DerT,ValT,RdnT = [[],[]],[[],[]],[[],[]]  # max nesting: T(H( layer( fork( ptuple|scalar))

    for Ele,ele in zip_longest(Que,que, fillvalue=[]):
        if Ele and ele:
            if isinstance(Ele[0],list):
                derT,valT,rdnT = comp_unpack(Ele, ele, rn)
            else:
                # elements are ptuples
                mtuple, dtuple = comp_dtuple(Ele, ele, rn)  # accum rn across higher composition orders
                mval=sum(mtuple); dval=sum(dtuple)
                derT = [mtuple, dtuple]
                valT = [mval, dval]
                rdnT = [int(mval<dval),int(mval>=dval)]  # to use np.sum
            for i in 0,1:
                DerT[i]+=[derT[i]]; ValT[i]+=[valT[i]]; RdnT[i]+=[rdnT[i]]

    return DerT,ValT,RdnT

def comp_ptuple(_ptuple, ptuple, rn):  # 0der

    mtuple, dtuple = [],[]
    for i, (_par, par, ave) in enumerate(zip(_ptuple, ptuple, aves)):
        if isinstance(_par, list):
            if len(_par)==2: m,d = comp_angle(_par, par)
            else:            m,d = comp_aangle(_par, par)
        else:  # I | M | Ma | G | Ga | L
            npar= par*rn  # accum-normalized par
            d = _par - npar
            if i: m = min(_par,npar)-ave
            else: m = ave-abs(d)  # inverse match for I, no mag/value correlation

        mtuple+=[m]; dtuple+=[d]
    return [mtuple, dtuple]

def comp_dtuple(_ptuple, ptuple, rn):

    mtuple, dtuple = [],[]
    for _par, par, ave in zip(_ptuple, ptuple, aves):  # compare ds only?
        npar= par*rn
        mtuple += [min(_par, npar) - ave]
        dtuple += [_par - npar]

    return [mtuple, dtuple]

def comp_angle(_angle, angle):  # rn doesn't matter for angles

    _Dy, _Dx = _angle
    Dy, Dx = angle
    _G = np.hypot(_Dy,_Dx); G = np.hypot(Dy,Dx)
    sin = Dy / (.1 if G == 0 else G);     cos = Dx / (.1 if G == 0 else G)
    _sin = _Dy / (.1 if _G == 0 else _G); _cos = _Dx / (.1 if _G == 0 else _G)
    sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    # cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
    dangle = sin_da
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

    # daangle = sin_dda0 + sin_dda1?
    daangle = np.arctan2(gay, gax)  # diff between aangles, probably wrong
    maangle = ave_daangle - abs(daangle)  # inverse match, not redundant as summed

    return [maangle,daangle]