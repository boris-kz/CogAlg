import numpy as np
from itertools import zip_longest
from copy import copy, deepcopy
from .classes import CderP, CPP
from .filters import ave, aves, vaves, ave_dangle, ave_daangle,med_decay, aveB, P_aves

'''
comp_slice traces edge blob axis by cross-comparing vertically adjacent Ps: slices across edge blob, along P.G angle.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)

P: any angle, connectivity in P_ is traced through root_s of derts adjacent to P.dert_, possibly forking. 
len prior root_ sorted by G is rdn of each root, to evaluate it for inclusion in PP, or starting new P by ave*rdn.
'''

def comp_slice(blob, verbose=False):  # high-G, smooth-angle blob, composite dert core param is v_g + iv_ga

    P_ = []
    for P in blob.P_:  # must be contiguous, gaps filled in scan_P_rim
        link_ = copy(P.link_); P.link_=[]
        P_ += [[P,link_]]

    for P, link_ in P_:
        ''' 
        lateral splicing, draft:
        spliced_link_ = []
        __P = link_.pop()
        for _P in link_.pop():
            spliced_link_ += lat_comp_P(__P, _P)  # comp uplinks, merge if close and similar, return merged __P
            __P = _P
        '''
        for _P in link_:  # or spliced_link_ if active
            comp_P(_P,P)  # replaces P.link_ Ps with derPs

    PPm_,PPd_ = form_PP_t([Pt[0] for Pt in P_], base_rdn=2)
    blob.PPm_, blob.PPd_  = PPm_, PPd_

# draft, ignore for now:
def lat_comp_P(_P,P):  # to splice, no der+

    ave = P_aves[0]
    rn = len(_P.dert_)/ len(P.dert_)

    mtuple,dtuple = comp_ptuple(_P.ptuple[:-1], P.ptuple[:-1], rn)

    _L, L = _P.ptuple[-1], P.ptuple[-1]
    gap = np.hypot((_P.y - P.y), (_P.x, P.x))
    rL = _L - L
    mM = min(_L, L) - ave
    mval = sum(mtuple); dval = sum(dtuple)
    mrdn = 1+(dval>mval); drdn = 1+(1-(dval>mval))  # rdn = Dval/Mval?


def comp_P(_P,P, fd=0, derP=None):  #  derP if der+, S if rng+

    aveP = P_aves[fd]
    rn = len(_P.dert_)/ len(P.dert_)

    if fd:  # der+: extend old link derP
        rn *= len(_P.link_t[1]) / len(P.link_t[1])  # derH is summed from links
        dderH, valt, rdnt = comp_derH(_P.derH, P.derH, rn)
        derP.derH += [dderH]  # flat, appended with each der+
        for i in 0,1:
            derP.valt[i]+=valt[i]; derP.rdnt[i]+=rdnt[i]
    else:
        # rng+: add new link derP
        mtuple,dtuple = comp_ptuple(_P.ptuple, P.ptuple, rn)
        mval = sum(mtuple); dval = sum(dtuple)
        mrdn = 1+(dval>mval); drdn = 1+(1-(dval>mval))  # rdn = Dval/Mval?
        derP = CderP(derH=[[mtuple,dtuple, mval,dval,mrdn,drdn]], valt=[mval,dval], rdnt=[mrdn,drdn], P=P,_P=_P, S=derP)
        P.link_ += [derP]  # all links
        if mval > aveP*mrdn: P.link_t[0] += [derP]  # +ve links, fork selection in form_PP_t
        if dval > aveP*drdn: P.link_t[1] += [derP]

def comp_derH(_derH, derH, rn):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    dderH = []  # or = not-missing comparand if xor?
    Mval, Dval, Mrdn, Drdn = 0,0,1,1

    for _lay, lay in zip_longest(_derH, derH, fillvalue=[]):
        if _lay and lay:
            mtuple, dtuple = comp_dtuple(_lay[1], lay[1], rn)
            mval = sum(mtuple); dval = sum(dtuple)
            mrdn = dval > mval; drdn = dval < mval
            dderH += [[mtuple,dtuple,mval,dval,mrdn,drdn]]
            Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn

    return dderH, [Mval,Dval], [Mrdn,Drdn]  # new layer, 1/2 combined derH

def comp_dtuple(_ptuple, ptuple, rn):

    mtuple, dtuple = [],[]
    for _par, par, ave in zip(_ptuple, ptuple, aves):  # compare ds only?
        npar= par*rn
        mtuple += [min(_par, npar) - ave]
        dtuple += [_par - npar]

    return [mtuple, dtuple]

def form_PP_t(P_, base_rdn):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = []
    for fd in 0,1:
        qPP_ = []  # initial sequence_PPs
        for P in P_:
            if not P.roott[fd]:  # else already packed in qPP
                qPP = [[P]]  # init PP is 2D queue of (P,val)s of all layers?
                P.roott[fd]=qPP; val = 0
                uplink_ = P.link_t[fd]
                uuplink_ = []  # next layer of links
                while uplink_:
                    for derP in uplink_:
                        _P = derP._P; _qPP = _P.roott[fd]
                        if _qPP:
                            if _qPP is not qPP:  # _P may be added to qPP via other downlinked P
                                val += _qPP[1]  # merge _qPP in qPP:
                                for qP in _qPP[0]:
                                    qP.roott[fd] = qPP; qPP[0] += [qP]  # append qP_
                                qPP_.remove(_qPP)
                        else:
                            qPP[0] += [_P]  # pack bottom up
                            _P.roott[fd] = qPP
                            val += derP.valt[fd]
                            uuplink_ += derP._P.link_t[fd]
                    uplink_ = uuplink_
                    uuplink_ = []
                qPP += [val,ave+1]  # ini reval=ave+1, keep qPP same object for ref in P.roott
                qPP_ += [qPP]
        # prune qPPs by mediated links vals:
        rePP_= reval_PP_(qPP_, fd)  # PP = [qPP,valt,reval]
        CPP_ = [sum2PP(qPP, base_rdn, fd) for qPP in rePP_]
        PP_t += [CPP_]  # may be empty

    return PP_t  # add_alt_PPs_(graph_t)?


def reval_PP_(PP_, fd):  # recursive eval / prune Ps for rePP

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
    if rePP_ and max([rePP[2] for rePP in rePP_]) > ave:  # recursion if any min reval:
        rePP_ = reval_PP_(rePP_,fd)

    return rePP_

def reval_P_(P_, fd):  # prune qPP by link_val + mediated link__val

    prune_=[]; Val=0; reval=0  # comb PP value and recursion value

    for P in P_:
        P_val = 0; remove_ = []
        for link in P.link_t[fd]:    # link val + med links val: single mediation layer in comp_slice:
            link_val = link.valt[fd] + sum([mlink.valt[fd] for mlink in link._P.link_t[fd]]) * med_decay
            if link_val < vaves[fd]:
                remove_ += [link]; reval += link_val
            else: P_val += link_val
        for link in remove_:
            P.link_t[fd].remove(link)  # prune weak links
        if P_val * P.rdnt[fd] < vaves[fd]:
            prune_ += [P]
        else:
            Val += P_val * P.rdnt[fd]
    for P in prune_:
        for link in P.link_t[fd]:  # prune direct links only?
            _P = link._P
            _link_ = _P.link_t[fd]
            if link in _link_:
                _link_.remove(link); reval += link.valt[fd]

    if reval > aveB:
        P_, Val, reval = reval_P_(P_, fd)  # recursion
    return [P_, Val, reval]


def sum2PP(qPP, base_rdn, fd):  # sum links in Ps and Ps in PP

    P_,_,_ = qPP  # proto-PP is a list
    PP = CPP(fd=fd, node_=P_)
    # accum:
    for i, P in enumerate(P_):
        P.roott[fd] = PP
        sum_ptuple(PP.ptuple, P.ptuple)
        L = P.ptuple[-1]
        Dy = P.axis[0]*L/2; Dx = P.axis[1]*L/2; y=P.y; x=P.x
        if i: Y0=min(Y0,(y-Dy)); Yn=max(Yn,(y+Dy)); X0=min(X0,(x-Dx)); Xn=max(Xn,(x+Dx))
        else: Y0=y-Dy; Yn=y+Dy; X0=x-Dx; Xn=x+Dx  # init

        for derP in P.link_t[fd]:
            derH, valt, rdnt = derP.derH, derP.valt, derP.rdnt
            if valt[0] < valt[1]: valt[0]+=1  # fork rdn
            else: valt[1]+=1
            sum_derH([P.derH,P.valt,P.rdnt], [derH,valt,rdnt], base_rdn)
            _P = derP._P  # bilateral summation:
            sum_derH([_P.derH,_P.valt,_P.rdnt], [derH,valt,rdnt], base_rdn)
        # excluding bilateral sums:
        sum_derH([PP.derH,PP.valt,PP.rdnt], [P.derH,P.valt,P.rdnt], base_rdn)
        # not needed: PP.link_+=P.link_; PP.link_t[0]+=P.link_t[0]; PP.link_t[1]+=P.link_t[1]

    PP.box =(Y0,Yn,X0,Xn)
    return PP


def sum_derH(T, t, base_rdn):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    DerH, Valt, Rdnt = T
    derH, valt, rdnt = t
    for i in 0, 1:
        Valt[i] += valt[i]
        Rdnt[i] += rdnt[i] + base_rdn
    if DerH:
        for Layer, layer in zip_longest(DerH,derH, fillvalue=[]):
            if layer:
                if Layer:
                    for i, param in enumerate(layer):
                        if i<2: sum_ptuple(Layer[i], param)  # mtuple | dtuple
                        elif i<4: Layer[i] += param  # mval | dval
                        else:     Layer[i] += param + base_rdn # | mrdn | drdn
                else:
                    DerH += [deepcopy(layer)]
    else:
        DerH[:] = deepcopy(derH)

def sum_ptuple(Ptuple, ptuple, fneg=0):

    for i, (Par, par) in enumerate(zip_longest(Ptuple, ptuple, fillvalue=None)):
        if par != None:
            if Par != None:
                if isinstance(Par, list):  # angle or aangle
                    for i,(P,p) in enumerate(zip(Par,par)):
                        Par[i] = P-p if fneg else P+p
                else:
                    Ptuple[i] += (-par if fneg else par)  # now includes n in ptuple[-1]?
            elif not fneg:
                Ptuple += [copy(par)]

def comp_ptuple(_ptuple, ptuple, rn):  # 0der

    mtuple, dtuple = [],[]
    # _n, n = _ptuple, ptuple: add to rn?
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