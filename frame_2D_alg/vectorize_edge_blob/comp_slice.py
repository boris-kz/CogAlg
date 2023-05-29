import numpy as np
from itertools import zip_longest
from copy import copy, deepcopy
from .classes import CderP, CPP
from .filters import ave, aves, vaves, PP_vars, ave_dangle, ave_daangle,med_decay, aveB, P_aves

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
            LayT = [[[[]]],[[[]]]]; valT = [[[0]],[[0]]]; rdnT = [[[1]],[[1]]]
            for _P in _P_:
                _L = len(_P.dert_); L = len(P.dert_); _x0=_P.box[2]; x0=P.box[2]
                # test for x overlap(_P,P) in 8 directions, all derts positive:
                if (x0 - 1 < _x0 + _L) and (x0 + L > _x0):
                    comp_P(_P,P, link_,link_m,link_d, LayT,valT,rdnT, fd=0)
                elif (x0 + L) < _x0:
                    break  # no xn overlap, stop scanning lower P_
            if link_:  # | link_t?
                P.link_=link_; P.link_t=[link_m,link_d]
                P.derT =[[LayT[0]],[LayT[1]]]; P.valT=valT; P.rdnT=rdnT  # single Mtuple, Dtuple derT
        _P_ = P_
    PPm_,PPd_ = form_PP_t(P__, base_rdn=2)
    blob.PPm_, blob.PPd_  = PPm_, PPd_


def form_PP_t(P__, base_rdn):  # form PPs of derP.valt[fd] + connected Ps'val

    PP_t = []
    for fd in 0, 1:
        fork_P__ = ([copy(P_) for P_ in reversed(P__)])  # scan bottom-up
        PP_ = []; packed_P_ = []  # form initial sequence-PPs:
        for P_ in fork_P__:
            for P in P_:
                if P not in packed_P_:
                    qPP = [[[P]], P.valT[fd][-1][-1]]  # init PP is 2D queue of node Ps and sum([P.val+P.link_val])
                    uplink_ = P.link_t[fd]; uuplink_ = []  # next-line links for recursive search
                    while uplink_:
                        for derP in uplink_:
                            if derP._P not in packed_P_:
                                qPP[0].insert(0, [derP._P])  # pack top down
                                qPP[1] += derP.valT[fd]
                                packed_P_ += [derP._P]
                                uuplink_ += derP._P.link_t[fd]
                        uplink_ = uuplink_
                        uuplink_ = []
                    PP_ += [qPP + [ave+1]]  # + [ini reval]
        # prune qPPs by med links val:
        rePP_= reval_PP_(PP_, fd)  # PP = [qPP,val,reval]
        CPP_ = [sum2PP(PP, base_rdn, fd) for PP in rePP_]
        PP_t += [CPP_]  # may be empty

    return PP_t  # add_alt_PPs_(graph_t)?

def reval_PP_(PP_, fd):  # recursive eval / prune Ps for rePP

    rePP_ = []
    while PP_:  # init P__
        P__, val, reval = PP_.pop(0)
        if val > ave:
            if reval < ave:  # same graph, skip re-evaluation:
                rePP_ += [[P__,val,0]]  # reval=0
            else:
                rePP = reval_P_(P__, fd)  # recursive node and link revaluation by med val
                if rePP[1] > ave:  # min adjusted val
                    rePP_ += [rePP]
    if rePP_ and max([rePP[2] for rePP in rePP_]) > ave:  # recursion if any min reval:
        rePP_ = reval_PP_(rePP_, fd)

    return rePP_

def reval_P_(P__, fd):  # prune qPP by (link_ + mediated link__) val

    prune_ = []; Val, reval = 0,0  # comb PP value and recursion value

    for P_ in P__:
        for P in P_:
            P_val = 0; remove_ = []
            for link in P.link_t[fd]:
                # recursive mediated link layers eval-> med_valT:
                _,_,med_valT = med_eval(link._P.link_t[fd], old_link_=[], med_valH=[], fd=fd)
                # link val + mlinks val: single med order, no med_valH in comp_slice?:
                link_val = link.valt[fd] + sum([mlink.valt[fd] for mlink in link._P.link_t[fd]]) * med_decay  # + med_valH
                if link_val < vaves[fd]:
                    remove_+= [link]; reval += link_val
                else: P_val += link_val
            for link in remove_:
                P.link_t[fd].remove(link)  # prune weak links
            if P_val < vaves[fd]:
                prune_ += [P]
            else:
                Val += P_val
    for P in prune_:
        for link in P.link_t[fd]:  # prune direct links only?
            _P = link._P
            _link_ = _P.link_t[fd]
            if link in _link_:
                _link_.remove(link); reval += link.valt[fd]

    if reval > aveB:
        P__, Val, reval = reval_P_(P__, fd)  # recursion
    return [P__, Val, reval]


def med_eval(last_link_, old_link_, med_valH, fd):  # compute med_valH: values of links mediated by incremental number of nodes

    curr_link_ = []; med_val = 0

    for llink in last_link_:
        for _link in llink._P.link_t[fd]:
            if _link not in old_link_:  # not-circular link
                old_link_ += [_link]  # evaluated mediated links
                curr_link_ += [_link]  # current link layer,-> last_link_ in recursion
                med_val += _link.valt[fd]
    med_val *= med_decay ** (len(med_valH) + 1)
    med_valH += [med_val]
    if med_val > aveB:
        # last med layer val-> likely next med layer val
        curr_link_, old_link_, med_valH = med_eval(curr_link_, old_link_, med_valH, fd)  # eval next med layer

    return curr_link_, old_link_, med_valH


def sum2PP(qPP, base_rdn, fd):  # sum Ps and links into PP

    P__,_,_ = qPP  # proto-PP is a list
    PP = CPP(box=copy(P__[0][0].box), fd=fd, P__ = P__)
    DerT = [[[[[]]]],[[[[]]]]]  # ptuple ) fork ) layer ) H ) T
    ValT = [[[0]],[[0]]]  # fork_val ) layer ) H ) T
    RdnT = [[[base_rdn]],[[base_rdn]]]
    # accum:
    for P_ in P__:  # top-down
        for P in P_:  # left-to-right
            P.roott[fd] = PP
            sum_ptuple(PP.ptuple, P.ptuple)
            if P.derT[0]:
                # always 1ptuple) 1fork) 1layer here, nesing is added by feedback, not sure if we can extend it there instead?
                sum_unpack([DerT[0],ValT[0],RdnT[0]], [P.derT[0],P.valT[0],P.rdnT[0]])
                sum_unpack([DerT[1],ValT[1],RdnT[1]], [P.derT[1],P.valT[1],P.rdnT[1]])
                PP.link_ += P.link_
                for Link_,link_ in zip(PP.link_t, P.link_t):
                    Link_ += link_  # all unique links in PP, to replace n
            Y0,Yn,X0,Xn = PP.box; y0,yn,x0,xn = P.box
            PP.box = [min(Y0,y0), max(Yn,yn), min(X0,x0), max(Xn,xn)]
    for i in 0,1:
        PP.derT[i] += [DerT[i]]
        PP.valT[i] += [ValT[i]]
        PP.rdnT[i] += [RdnT[i]]
    return PP

# draft
def sum_unpack(Q,q):  # recursive unpack nested sequence to sum final ptuples

    Que,Val_,Rdn_ = Q; que,val_,rdn_ = q  # max Que nesting: H ( layer ( fork (ptuple)))

    for Ele,Val,Rdn, ele,val,rdn in zip_longest(Que,Val_,Rdn_, que,val_,rdn_, fillvalue=[]):
        if ele:
            if Ele:
                if isinstance(val,list):  # H or layer
                    sum_unpack([Ele,Val,Rdn], [ele,val,rdn])
                else:  # fork
                    Val += val; Rdn += rdn
                    for Ptuple, ptuple in zip(Ele,ele):
                        sum_ptuple(Ptuple, ptuple)
                        Val += sum(ptuple)  # sum in fork_val, also adjust rdn?
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


def comp_P(_P,P, link_,link_m,link_d, Lay, valT, rdnT, fd=0, derP=None):  #  if der+

    aveP = P_aves[fd]
    rn = len(_P.dert_)/ len(P.dert_)

    if fd:  # der+: comp last lay in old link, comp lower lays formed derP.derT:
        rn *= len(_P.link_t[1]) / len(P.link_t[1])
        # derT is summed from links
        derT, valT, rdnT = comp_unpack(_P.derT[1][-1], P.derT[1][-1], rn)
        mval = valT[0][-1][-1]; dval = valT[1][-1][-1]  # in fb: np.sum(valT[fd])
        mrdn = 1+(dval>mval); drdn = 1+(1-mrdn)
        for i in 0,1:
            derT[i]+=derT[i]; derP.valt[i]+= dval if i else mval; derP.rdnt[i]+= drdn if i else mrdn
    else:
        # rng+: add new link
        mtuple,dtuple = comp_ptuple(_P.ptuple, P.ptuple, rn)
        mval = sum(mtuple); dval = sum(dtuple)
        mrdn = 1+(dval>mval); drdn = 1+(1-mrdn)  # or greyscale rdn = Dval/Mval?
        derP = CderP(derT=[[mtuple],[dtuple]], valT=[mval,dval],rdnT=[mrdn,drdn], P=P,_P=_P, box=copy(_P.box), # or box of means?
                     L=len(_P.dert_))
    link_ += [derP]  # all links
    if mval > aveP*mrdn:
        link_m+=[derP]; valT[0][-1][-1]+=mval; rdnT[0][-1][-1]+=mrdn  # +ve links, fork selection in form_PP_t
        if fd: sum_unpack(Lay, derP.derT[-1])  # sum fork of old layers
        else: sum_ptuple(Lay[0][-1][-1], mtuple)
    if dval > aveP*drdn:
        link_d+=[derP]; valT[1][-1][-1]+=dval; rdnT[1][-1][-1]+=drdn
        if fd: sum_unpack(Lay, derP.derT)
        else: sum_ptuple(Lay[1][-1][-1], dtuple)

    if fd: derP.derT += [Lay]  # derT must be summed above with old derP.derT

# draft
def comp_unpack(Que,que, rn):  # recursive unpack nested sequence to compare final ptuples

    DerT,ValT,RdnT = [[],[]], [0,0], [1,1]
    # max T = mH,dH: derH( layer( fork (ptuple( par_scalar)))) or val|rdnH( layer( fork_scalar))

    for Ele,ele in zip_longest(Que,que, fillvalue=[]):
        if Ele and ele:
            if isinstance(Ele[0],list):
                derT,valT,rdnT = comp_unpack(Ele, ele, rn)
            else:  # elements are ptuples
                mtuple, dtuple = comp_dtuple(Ele, ele, rn)  # accum rn across higher composition orders
                mval=sum(mtuple); dval=sum(dtuple)
                derT = [mtuple, dtuple]
                valT = [mval, dval]
                rdnT = [mval<dval, mval>=dval]
            for i in 0,1:
                DerT[i]+=[derT[i]]; ValT[i]+=valT[i]; RdnT[i]+=rdnT[i]

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
            else: m = [ave-abs(d)]  # inverse match for I, no mag/value correlation

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