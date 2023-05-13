from .classes import CQ, Cptuple, CP, CderP, CPP
from .filters import (
    aves, vaves, PP_vars, PP_aves,
    ave_inv, ave, ave_g, ave_ga,
    flip_ave, flip_ave_FPP,
    div_ave,
    ave_rmP, ave_ortho, aveB, ave_dI, ave_M, ave_Ma, ave_G, ave_Ga, ave_L,
    ave_x, ave_dx, ave_dy, ave_daxis, ave_dangle, ave_daangle,
    ave_mval, ave_mPP, ave_dPP, ave_splice,
    ave_nsub, ave_sub, ave_agg, ave_overlap, ave_rotate,
    med_decay,
)
from .sub_recursion import sub_recursion_eval, copy
from .agg_convert import agg_recursion_eval


def comp_slice(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga
    # scan rows top-down, comp y-adjacent, x-overlapping Ps to form derPs

    P__ = blob.P__
    _P_ = P__[0]  # higher row
    for P_ in P__[1:]:  # lower row
        for P in P_:
            link_, link_m,link_d = [],[],[]  # empty in initial Ps:
            Valt = [0,0]; Rdnt = [0,0]
            Mtuple,Dtuple = [],[]
            for _P in _P_:  # test for x overlap(_P,P) in 8 directions, derts are positive in all Ps:
                _L = len(_P.dert_); L = len(P.dert_)
                if (P.x0 - 1 < _P.x0 + _L) and (P.x0 + L > _P.x0):
                    comp_P(_P,P, Mtuple,Dtuple, link_,link_m,link_d, Valt, Rdnt, fd=0)
                elif (P.x0 + L) < _P.x0:
                    P.derH=[[[Mtuple,Dtuple]]]  # single vertuple
                    P.link_=link_; P.link_t=[link_m,link_d]; P.valt=Valt; P.rdnt = Rdnt
                    break  # no xn overlap, stop scanning lower P_
        _P_ = P_
    PPm_,PPd_ = form_PP_t(P__, base_rdn=2)
    blob.PPm_, blob.PPd_  = PPm_, PPd_
    # re comp, cluster:
    for i, PP_ in enumerate([PPm_, PPd_]):  # derH, fds per PP
        if sum([PP.valt[i] for PP in PP_]) > ave * sum([PP.rdnt[i] for PP in PP_]):
            sub_recursion_eval(blob, PP_, fd=i)  # intra PP
            # agg_recursion_eval(blob, copy(PP_))  # cross PP, Cgraph conversion doesn't replace PPs?


def form_PP_t(P__, base_rdn):  # form PPs of derP.valt[fd] + connected Ps'val

    PP_t = []
    for fd in 0, 1:
        fork_P__ = ([copy(P_) for P_ in reversed(P__)])  # scan bottom-up
        PP_ = []; packed_P_ = []  # form initial sequence-PPs:
        for P_ in fork_P__:
            for P in P_:
                if P not in packed_P_:
                    qPP = [[[P]], P.valt[fd]]  # init PP is 2D queue of node Ps and sum([P.val+P.link_val])
                    uplink_ = P.link_t[fd]; uuplink_ = []  # next-line links for recursive search
                    while uplink_:
                        for derP in uplink_:
                            if derP._P not in packed_P_:
                                qPP[0].insert(0, [derP._P])  # pack top down
                                qPP[1] += derP.valt[fd]
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
                # recursive mediated link layers eval-> med_valH:
                _,_,med_valH = med_eval(link._P.link_t[fd], old_link_=[], med_valH=[], fd=fd)
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

def med_eval(last_link_, old_link_, med_valH, fd):  # compute med_valH

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

    P__, val, _ = qPP  # proto-PP is a list
    # init:
    P0 = P__[0][0]
    PP = CPP(box=[P0.y0,P__[-1][0].y0,P0.x0,P0.x0+len(P0.dert_)], fds=[fd], P__ = P__)
    PP.valt[fd] = val; PP.rdnt[fd] += base_rdn
    # accum:
    for P_ in P__:  # top-down
        for P in P_:  # left-to-right
            P.roott[fd] = PP
            sum_ptuple(PP.ptuple, P.ptuple)
            if PP.derH: [sum_vertuple(L[0],l[0]) for L,l in zip_longest(PP.derH, P.derH, fillvalue=[])]
            else: PP.derH = deepcopy(P.derH)  # sum_derH(PP.derH, P.derH)
            PP.link_ += P.link_
            for Link_,link_ in zip(PP.link_t, P.link_t):
                Link_ += link_  # all unique links in PP, to replace n
            PP.box[0] = min(PP.box[0], P.y0)  # y0
            PP.box[2] = min(PP.box[2], P.x0)  # x0
            PP.box[3] = max(PP.box[3], P.x0 + len(P.dert_))  # xn

    return PP


def sum_derH(DerH, derH):  # derH is layers, not selective here. Sum both forks in each node to comp alt fork

    for Layer, layer in zip_longest(DerH, derH, fillvalue=None):
        if layer != None:
            if Layer != None:
                for Vertuple, vertuple in zip_longest(Layer, layer, fillvalue=None):  # vertuple is [mtuple, dtuple]
                    if vertuple != None:
                        if Vertuple != None:
                            sum_vertuple(Vertuple, vertuple)
                        else:
                            Layer += [deepcopy(vertuple)]
            else:
                DerH += [deepcopy(layer)]



def sum_vertuple(Vertuple, vertuple):  # [mtuple,dtuple]

    for Ptuple, ptuple in zip_longest(Vertuple, vertuple, fillvalue=None):
        if ptuple != None:
            if Ptuple != None:
                sum_tuple(Ptuple, ptuple)
            else:
                Vertuple += deepcopy(vertuple)


def sum_tuple(Ptuple,ptuple, fneg=0):  # mtuple or dtuple

    for i, (Par, par) in enumerate(zip_longest(Ptuple, ptuple, fillvalue=None)):
        if par != None:
            if Par:
                Ptuple[i] = Par + -par if fneg else par
            elif not fneg:
                Ptuple += [par]

def sum_ptuple(Ptuple, ptuple, fneg=0):

    for pname, ave in zip(pnames, aves):
        Par = getattr(Ptuple, pname); par = getattr(ptuple, pname)

        if isinstance(Par, list):  # angle or aangle
            for j, (P,p) in enumerate(zip(Par,par)): Par[j] = P-p if fneg else P+p
        else:
            Par += (-par if fneg else par)
        setattr(Ptuple, pname, Par)

    Ptuple.n += 1


def comp_P(_P,P, Mtuple, Dtuple, link_,link_m,link_d, Valt, Rdnt, fd, derP=None, Mt0=None, Dt0=None):  #  last three if der+

    if fd:
        _dtuple, dtuple = _P.derH[-1][0][1], P.derH[-1][0][1]  # 1-vertuple derH before feedback, compare dtuple
        mtuple,dtuple = comp_vertuple(_dtuple, dtuple, rn=len(_P.link_t[1])/len(P.link_t[1]))  # comp_tuple
    else:
        mtuple,dtuple = comp_ptuple(_P.ptuple, P.ptuple)
    mval = sum(mtuple)
    if fd: mval += derP.valt[0]
    dval = sum(dtuple)
    if fd: dval += derP.valt[1]
    mrdn = 1+ dval>mval; drdn = 1+(not mrdn)

    derP = CderP(derH=[[[mtuple,dtuple]]],fds=P.fds+[fd], valt=[mval,dval],rdnt=[mrdn,drdn], P=P,_P=_P, x0=_P.x0,y0=_P.y0,L=len(_P.dert_))
    link_ += [derP]  # all links
    if mval > aveB*mrdn:
        link_m+=[derP]; Valt[0]+=mval; Rdnt[0]+=mrdn  # +ve links, fork selection in form_PP_t
        sum_tuple(Mtuple, mtuple)
        if fd: sum_tuple(Mt0, derP.derH[0][0][0])
    if dval > aveB*drdn:
        link_d+=[derP]; Valt[1]+=dval; Rdnt[1]+=drdn
        sum_tuple(Dtuple, dtuple)
        if fd: sum_tuple(Dt0, derP.derH[0][0][1])


def comp_vertuple(_vertuple, vertuple, rn):

    mtuple, dtuple = [],[]
    for _par, par, ave in zip(_vertuple[1], vertuple[1], aves):  # compare ds only?

        m,d = comp_par(_par, par*rn, ave)
        mtuple+=[m]; dtuple+=[d]

    return [mtuple, dtuple]

def comp_ptuple(_ptuple, ptuple):

    mtuple, dtuple = [],[]  # in the order of ("I", "M", "Ma", "axis", "angle", "aangle","G", "Ga", "x", "L")
    rn = _ptuple.n / ptuple.n  # normalize param as param*rn for n-invariant ratio: _param / param*rn = (_param/_n) / (param/n)

    for pname, ave in zip(pnames, aves):
        _par = getattr(_ptuple, pname)
        par = getattr(ptuple, pname)
        if pname=="aangle": m,d = comp_aangle(_par, par)
        elif pname in ("axis","angle"): m,d = comp_angle(_par, par)
        else:
            if pname!="x": par*=rn  # normalize by relative accum count
            if pname=="x" or pname=="I": finv = 1
            else: finv=0
            m,d = comp_par(_par, par, ave, finv)
        mtuple += [m]
        dtuple += [d]

    return [mtuple, dtuple]

def comp_par(_param, param, ave, finv=0):  # comparand is always par or d in [m,d]

    d = _param - param
    if finv: m = ave - abs(d)  # inverse match for primary params, no mag/value correlation
    else:    m = min(_param, param) - ave
    return [m,d]

def comp_angle(_angle, angle):  # rn doesn't matter for angles

    _Dy, _Dx = _angle
    Dy, Dx = angle
    _G = np.hypot(_Dy,_Dx); G = np.hypot(Dy,Dx)
    sin = Dy / (.1 if G == 0 else G);     cos = Dx / (.1 if G == 0 else G)
    _sin = _Dy / (.1 if _G == 0 else _G); _cos = _Dx / (.1 if _G == 0 else _G)
    sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β

    dangle = np.arctan2(sin_da, cos_da)  # scalar, vertical difference between angles
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

    daangle = np.arctan2(gay, gax)  # diff between aangles, probably wrong
    maangle = ave_daangle - abs(daangle)  # inverse match, not redundant as summed

    return [maangle,daangle]