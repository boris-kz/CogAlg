import numpy as np
from itertools import zip_longest
from copy import copy, deepcopy
from .classes import CderP, CPP
from .filters import ave, aves, vaves, ave_dangle, ave_daangle,med_decay, aveB, P_aves

'''
comp_slice traces edge blob axis by cross-comparing vertically adjacent Ps: slices across edge blob, along P.G angle.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)
'''

def comp_slice(blob, verbose=False):  # high-G, smooth-angle blob, composite dert core param is v_g + iv_ga

    P__ = blob.P__
    _P_ = P__[0]  # higher row
    for P_ in P__[1:]:  # lower row
        for P in P_:
            link_,link_m,link_d = [],[],[]  # empty in initial Ps
            derT=[[],[]]; valT=[0,0]; rdnT=[1,1]  # to sum links in comp_P
            for _P in _P_:
                _L = len(_P.dert_); L = len(P.dert_); _x0=_P.box[2]; x0=P.box[2]
                # test for x overlap(_P,P) in 8 directions, all derts positive:
                if (x0 - 1 < _x0 + _L) and (x0 + L > _x0):
                    comp_P(_P,P, link_,link_m,link_d, derT,valT,rdnT, fd=0)
                elif (x0 + L) < _x0:
                    break  # no xn overlap, stop scanning lower P_
            if link_:
                P.link_=link_; P.link_t=[link_m,link_d]
                P.derT=derT; P.valT=valT; P.rdnT=rdnT  # single Mtuple, Dtuple
        _P_ = P_
    PPm_,PPd_ = form_PP_t(P__, base_rdn=2)
    blob.PPm_, blob.PPd_  = PPm_, PPd_


def form_PP_t(P__, base_rdn):  # form PPs of derP.valt[fd] + connected Ps'val

    PP_t = []
    for fd in 0,1:
        fork_P__ = ([copy(P_) for P_ in reversed(P__)])  # scan bottom-up
        qPP_ = []  # form initial sequence-PPs:
        for P_ in fork_P__:
            for P in P_:
                if not P.roott[fd]:
                    qPP = [[P]]  # init PP is 2D queue of Ps, + valt of all layers?
                    P.roott[fd]=qPP; valt = [0,0]
                    uplink_ = P.link_t[fd]; uuplink_ = []
                    # next-line links for recursive search
                    while uplink_:
                        for derP in uplink_:
                            _P = derP._P; _qPP = _P.roott[fd]
                            if _qPP:
                                for i in 0, 1: valt[i] += _qPP[1][i]
                                merge(qPP[0],_qPP[0])  # merge P__s, not sure how to align them
                            else:
                                qPP[0].insert(0,_P)  # pack top down
                                _P.root[fd] = qPP
                                for i in 0,1: valt[i] += np.sum(derP.valT[i])
                                uuplink_ += derP._P.link_t[fd]
                        uplink_ = uuplink_
                        uuplink_ = []
                    qPP_ += [[qPP, valt, ave+1]]  # ini reval=ave+1
        # prune qPPs by med links val:
        rePP_= reval_PP_(qPP_, fd)  # PP = [qPP,valt,reval]
        CPP_ = [sum2PP(qPP, base_rdn, fd) for qPP in rePP_]
        PP_t += [CPP_]  # may be empty

    return PP_t  # add_alt_PPs_(graph_t)?

# draft
def merge(P__, _P__):  # the P__s should not have shared Ps

    '''
    not sure how align P__s at the moment
    _P__ = [_P for _P_ in _qPP[0] for _P in _P_]
    if derP._P in _P__:
        for _P_ in _qPP[0]:  # pack Ps into P__ based on y
            for _P in _P_:
                if _P is not derP._P and _P not in _P__:
                    # pack P into P__ in top down sequence
                    current_ys = [P_[0].box[0] for P_ in P__]  # list of current-layer seg rows
                    if P.box[0] in current_ys:
                        if P not in P__[current_ys.index(P.box[0])]:
                            P__[current_ys.index(P.box[0])].append(P)  # append P row
                    elif P.box[0] > current_ys[0]:  # P.y0 > largest y in ys
                        P__.insert(0, [P])
                    elif P.box[0] < current_ys[-1]:  # P.y0 < smallest y in ys
                        P__.append([P])
                    elif P.box[0] < current_ys[0] and P.box[0] > current_ys[-1]:  # P.y0 in between largest and smallest value
                        for i, y in enumerate(current_ys):  # insert y if > next y
                            if P.box[0] > y: P__.insert(i, [P])  # PP.P__.insert(P.y0 - current_ys[-1], [P])
    '''

def reval_PP_(PP_, fd):  # recursive eval / prune Ps for rePP

    rePP_ = []
    while PP_:  # init P__
        P__, valt, reval = PP_.pop(0)
        Ave = ave * 1+(valt[fd] < valt[1-fd])  # * PP.rdnT[fd]?
        if valt[fd] > Ave:
            if reval < Ave:  # same graph, skip re-evaluation:
                rePP_ += [[P__,valt,0]]  # reval=0
            else:
                rePP = reval_P_(P__, fd)  # recursive node and link revaluation by med val
                if valt[fd] > Ave:  # min adjusted val
                    rePP_ += [rePP]
    if rePP_ and max([rePP[2] for rePP in rePP_]) > ave:  # recursion if any min reval:
        rePP_ = reval_PP_(rePP_, fd)

    return rePP_


def reval_P_(P__, fd):  # prune qPP by link_val + mediated link__val

    prune_=[]; Valt=[0,0]; reval=0  # comb PP value and recursion value

    for P_ in P__:
        for P in P_:
            P_val = 0; remove_ = []
            for link in P.link_t[fd]:  # link val + med links val: single med layer in comp_slice:
                link_val = np.sum(link.valT[fd]) + \
                           sum([np.sum(mlink.valT[fd]) for mlink in link._P.link_t[fd]]) * med_decay
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
        P__, Valt, reval = reval_P_(P__, fd)  # recursion
    return [P__, Valt, reval]


def sum2PP(qPP, base_rdn, fd):  # sum Ps and links into PP

    P__,_,_ = qPP  # proto-PP is a list
    # init:
    P = (P__[0][0])
    Ptuple, DerT,ValT,RdnT, (Y0,Yn,X0,Xn), Link_, (Link_m,Link_d) \
    = deepcopy(P.ptuple),deepcopy(P.derT),deepcopy(P.valT),deepcopy(P.rdnT),copy(P.box),copy(P.link_),copy(P.link_t)
    PP = CPP(fd=fd, P__=P__)
    # accum:
    for i, P_ in enumerate(P__):  # top-down
        for j, P in enumerate(P_):  # left-to-right
            P.roott[fd] = PP
            if i or j:  # exclude init P
                sum_ptuple(Ptuple, P.ptuple)
                Link_+=P.link_; Link_m+=P.link_t[0]; Link_d+=P.link_t[1]
                y0,yn,x0,xn = P.box
                Y0=min(Y0,y0); Yn=max(Yn,yn); X0=min(X0,x0); Xn=max(Xn,xn)
                if P.derT[0]:
                    for k in 0,1:
                        if isinstance(P.valT[0], list):  # der+: H = 1fork) 1layer before feedback
                            sum_unpack([DerT[k],ValT[k],RdnT[k]], [P.derT[k],P.valT[k],P.rdnT[k]])
                        else:  # rng+: 1 vertuple
                            sum_ptuple(DerT[k], P.derT[k]); ValT[k]+=P.valT[k]; RdnT[k]+=P.rdnT[k]

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


def comp_P(_P,P, link_,link_m,link_d, LayT,ValT,RdnT, fd=0, derP=None):  #  derP if fd==1

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
        mrdn = 1+(dval>mval); drdn = 1+(1-(dval>mval))  # or greyscale rdn = Dval/Mval?
        derP = CderP(derT=[mtuple,dtuple],valT=[mval,dval],rdnT=[mrdn,drdn], P=P,_P=_P, box=copy(_P.box), # or box of means?
                     L=len(_P.dert_))
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