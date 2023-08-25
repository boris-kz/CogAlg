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

def comp_slice(edge, verbose=False):  # high-G, smooth-angle blob, composite dert core param is v_g + iv_ga

    P_ = []
    for P in edge.node_:  # init P_, must be contiguous, gaps filled in scan_P_rim
        link_ = copy(P.link_H[-1])  # init rng+
        P.link_H[-1] = []  # fill with derPs in comp_P
        P_ +=[[P,link_]]
    for P, link_ in P_:
        for _P in link_:  # or spliced_link_ if active
            comp_P(_P,P, fder=0)  # replaces P.link_ Ps with derPs
    # convert node_ to node_tt:
    edge.node_T = [form_PP_t([Pt[0] for Pt in P_], PP_=None, base_rdn=2, fder=0), [[], []]]  # root fork is rng+ only


# not revised:
def comp_P(_P,P, fder=1, derP=None):  #  derP if der+, S if rng+

    aveP = P_aves[fder]
    rn = len(_P.dert_)/ len(P.dert_)

    if fder:  # der+: extend in-link derH
        rn *= len(_P.link_H[-2]) / len(P.link_H[-2])  # derH is summed from links
        dderH, valt, rdnt = comp_derH(_P.derH, P.derH, rn)  # += fork rdn
        derP = CderP(derH = derP.derH+dderH, valt=valt, rdnt=rdnt, P=P,_P=_P, S=derP.S)  # dderH valt,rdnt for new link
        mval,dval = valt[:2]; mrdn,drdn = rdnt
    else:  # rng+: add derH
        mtuple,dtuple,Mtuple = comp_ptuple(_P.ptuple, P.ptuple, rn)
        mval = sum(mtuple); dval = sum(dtuple); maxv = sum(Mtuple)
        mrdn = 1+(dval>mval); drdn = 1+(1-(dval>mval))  # rdn = Dval/Mval?
        derP = CderP(derH=[[[mtuple,dtuple], [mval,dval],[mrdn,drdn]]], valt=[mval,dval,maxv], rdnt=[mrdn,drdn], P=P,_P=_P, S=derP)

    if mval > aveP*mrdn or dval > aveP*drdn:
        P.link_H[-1] += [derP]


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

# not reviewed:
def form_PP_t(P_, PP_, base_rdn, fder):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = []
    for fd in 0,1:
        qPP_ = []  # initial sequence_PP s
        for P in P_:
            if not P.root_tt[fder][fd]:  # else already packed in qPP
                qPP = [[P]]  # init PP is 2D queue of (P,val)s of all layers?
                P.root_tt[fder][fd] = qPP; val = 0
                uplink_ = P.link_H[-1]
                uuplink_ = []  # next layer of uplinks
                while uplink_:  # later uuplink_
                    for derP in uplink_:
                        if derP.valt[fder] > P_aves[fder]* derP.rdnt[fder]:
                            _P = derP._P
                            if _P not in P_:  # _P is outside of current PP, merge its root PP:
                                _PP = _P.root_tt[fder][fd]
                                if _PP:  # _P is already clustered
                                    for _node in _PP.node_:
                                        if _node not in qPP[0]:
                                            qPP[0] += [_node]; _node.root_tt[fder][fd] = qPP  # reassign root
                                    PP_.remove(_PP)
                            else:
                                _qPP = _P.root_tt[fder][fd]
                                if _qPP:
                                    if _qPP is not qPP:  # _P may be added to qPP via other down-linked P
                                        val += _qPP[1]  # merge _qPP in qPP:
                                        for qP in _qPP[0]:
                                            qP.root_tt[fder][fd] = qPP
                                            qPP[0] += [qP]  # qP_+=[qP]
                                        qPP_.remove(_qPP)
                                else:
                                    qPP[0] += [_P]  # pack bottom up
                                    _P.root_tt[fder][fd] = qPP
                                    val += derP.valt[fd]
                                    uuplink_ += derP._P.link_H[-1]
                    uplink_ = uuplink_
                    uuplink_ = []
                qPP += [val, ave + 1]  # ini reval=ave+1, keep qPP same object for ref in P.roott
                qPP_ += [qPP]

        # prune qPPs by mediated links vals:
        rePP_ = reval_PP_(qPP_, fder, fd)  # PP = [qPP,valt,reval]
        CPP_ = [sum2PP(qPP, base_rdn, fder, fd) for qPP in rePP_]

        PP_t += [CPP_]  # least one PP in rePP_, which would have node_ = P_

    return PP_t  # add_alt_PPs_(graph_t)?


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

    Val=0; reval=0  # comb PP value and recursion value

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


def sum2PP(qPP, base_rdn, fder, fd):  # sum links in Ps and Ps in PP

    P_,_,_ = qPP  # proto-PP is a list
    PP = CPP(fd=fd, node_T=P_)
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
                    for i in range(0,1):
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
                if isinstance(Par, list):  # angle or aangle
                    for i,(P,p) in enumerate(zip(Par,par)):
                        Par[i] = P-p if fneg else P+p
                else:
                    Ptuple[i] += (-par if fneg else par)  # now includes n in ptuple[-1]?
            elif not fneg:
                Ptuple += [copy(par)]

def comp_ptuple(_ptuple, ptuple, rn):  # 0der

    mtuple, dtuple, Mtuple = [],[], []
    # _n, n = _ptuple, ptuple: add to rn?
    for i, (_par, par, ave) in enumerate(zip(_ptuple, ptuple, aves)):
        if isinstance(_par, list):
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


def comp_angle(_angle, angle):  # rn doesn't matter for angles

    # angle = [dy,dx]
    (_sin, sin), (_cos, cos) = [*zip(_angle, angle)] / np.hypot(*zip(_angle, angle))

    dangle = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    # cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
    mangle = ave_dangle - abs(dangle)  # inverse match, not redundant if sum cross sign

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