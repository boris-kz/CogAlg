from itertools import zip_longest
import numpy as np
from copy import copy, deepcopy
from .filters import PP_aves, ave_nsub
from .classes import CP, CPP
from .comp_slice import comp_P, form_PP_t, sum_derH
from dataclasses import replace


def sub_recursion_eval(root, PP_):  # fork PP_ in PP or blob, no derH in blob

    termt = [1,1]
    for PP in PP_:
        P_ = copy(PP.node_); sub_PP_t = []
        fr = 0
        for fd in 0,1:
            if PP.valt[fd] > PP_aves[fd] * PP.rdnt[fd] and len(PP.node_) > ave_nsub:
                termt[fd] = 0; fr = 1
                sub_PP_t += [sub_recursion(PP, P_, fd=fd)]  # comp_der|rng in PP -> parLayer
            else:
                sub_PP_t += [P_]
                if isinstance(root, CPP):  # separate feedback per fork?:
                    root.fback_t[fd] += [[PP.derH, PP.valt, PP.rdnt]]
        if fr: PP.node_ = sub_PP_t
    for fd in 0,1:
        if termt[fd] and isinstance(root, CPP) and root.fback_t[fd]:
            feedback(root, fd)  # upward recursive extend root.derT, forward eval only


def sub_recursion(PP, P_, fd):  # evaluate PP for rng+ and der+, add layers to select sub_PPs

    # same else new P_:
    P_ = comp_der(P_) if fd else comp_rng(P_, PP.rng+1)
    PP.rdnt[fd] += PP.valt[fd] - PP_aves[fd]*PP.rdnt[fd] > PP.valt[1-fd] - PP_aves[1-fd]*PP.rdnt[1-fd]  # not last layer val?

    cP_ = [replace(P, roott=[None,None], link_t=[[],[]]) for P in P_]  # reassign roots to sub_PPs
    sub_PP_t = form_PP_t(cP_, base_rdn=PP.rdnt[fd])  # replace P_ with sub_PPm_, sub_PPd_

    for i, sub_PP_ in enumerate(sub_PP_t):
        if sub_PP_:  # add eval
            for sPP in sub_PP_: sPP.roott[i] = PP
            sub_recursion_eval(PP, sub_PP_)

    return sub_PP_t  # for 4 nested forks in replaced P_?


def comp_rng(iP_, rng):  # form new Ps and links, switch to rng+n to skip clustering?

    P_ = []
    for P in iP_:
        cP = CP(ptuple=deepcopy(P.ptuple), dert_=copy(P.dert_))  # replace links, then derT in sum2PP
        # trace mlinks:
        for derP in P.link_t[0]:
            _P = derP._P
            for _derP in _P.link_:  # next layer, of all links?
                __P = _derP._P  # next layer of Ps
                distance = np.hypot(__P.x-P.x, __P.y-P.y)  # distance between mid points
                if distance > rng:
                    comp_P(cP,__P, fd=0, derP=distance)  # distance=S, mostly lateral, relative to L for eval?
        P_ += [cP]
    return P_

def comp_der(P_):  # keep same Ps and links, increment link derTs, then P derTs in sum2PP

    for P in P_:
        for derP in P.link_t[1]:  # trace dlinks
            if derP._P.link_t[1]:  # else no _P.derT to compare
                _P = derP._P
                comp_P(_P,P, fd=1, derP=derP)
    return P_

def feedback(root, fd):  # append new der layers to root

    Fback = deepcopy(root.fback_t[fd].pop())  # init with 1st fback: [derH,valt,rdnt], derH: [[mtuple,dtuple, mval,dval, mrdn, drdn]]
    while root.fback_t[fd]:
        sum_derH(Fback,root.fback_t[fd].pop(), base_rdn=0)
    sum_derH([root.derH, root.valt,root.rdnt], Fback, base_rdn=0)

    if isinstance(root.roott[fd], CPP):  # not blob
        root = root.roott[fd]
        root.fback_t[fd] += [Fback]
        if len(root.fback_t[fd]) == len(root.node_[fd]):  # all nodes term, fed back to root.fback_t
            feedback(root, fd)  # derH/ rng layer in sum2PP, deeper rng layers are appended by feedback


