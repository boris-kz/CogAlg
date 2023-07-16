from itertools import zip_longest
import numpy as np
from copy import copy, deepcopy
from .filters import PP_aves, ave_nsubt
from .classes import CP, CPP
from .comp_slice import comp_P, form_PP_t, sum_derH  # add_ext_layer
from dataclasses import replace

def sub_recursion_eval(root, PP_):  # fork PP_ in PP or blob, no derH in blob

    termt = [1,1]
    # PP_ in PP_t:
    for PP in PP_:
        sub_tt = []  # from rng+, der+
        fr = 0  # recursion in any fork
        for fd in 0,1:  # rng+ and der+:
            if len(PP.node_) > ave_nsubt[fd] and PP.valt[fd] > PP_aves[fd] * PP.rdnt[fd]:
                termt[fd] = 0
                for P in PP.node_: add_ext_layer(PP, P, fd, fr)  # add link_ and root, for both forks if not fr
                sub_tt += [sub_recursion(PP, fd=fd)]  # comp_der|rng in PP->parLayer
                fr = 1
            else:
                sub_tt += [PP.node_]
                if isinstance(root, CPP):  # separate feedback per terminated comp fork:
                    root.fback_t[fd] += [[PP.derH, PP.valt, PP.rdnt]]
        if fr:
            PP.node_ = sub_tt
            # nested PP_ tuple from 2 comp forks, each returns sub_PP_t: 2 clustering forks, if taken
    return termt

def sub_recursion(PP, fd):  # evaluate PP for rng+ and der+, add layers to select sub_PPs

    comp_der(PP.node_) if fd else comp_rng(PP.node_, PP.rng+1)  # same else new P_ and links
    # eval all or last layer?:
    PP.rdnt[fd] += PP.valt[fd] - PP_aves[fd]*PP.rdnt[fd] > PP.valt[1-fd] - PP_aves[1-fd]*PP.rdnt[1-fd]

    sub_PP_t = form_PP_t(PP.node_, base_rdn=PP.rdnt[fd])  # replace P_ with sub_PPm_, sub_PPd_

    for i, sub_PP_ in enumerate(sub_PP_t):  # sub_PP_ has at least one sub_PP: len node_ > ave_nsubt[fd]
        for sPP in sub_PP_: sPP.roott[i] = PP
        termt = sub_recursion_eval(PP, sub_PP_)
        if any(termt):
            for fd in 0, 1:
                if termt[fd] and PP.fback_t[fd]:
                    feedback(PP, fd)
                    # upward recursive extend root.derH, forward eval only
    return sub_PP_t

def feedback(root, fd):  # append new der layers to root

    Fback = deepcopy(root.fback_t[fd].pop())
    # init with 1st fback: [derH,valt,rdnt], derH: [[mtuple,dtuple, mval,dval, mrdn, drdn]]
    while root.fback_t[fd]:
        sum_derH(Fback,root.fback_t[fd].pop(), base_rdn=0)
    sum_derH([root.derH, root.valt,root.rdnt], Fback, base_rdn=0)

    if isinstance(root.roott[fd], CPP):  # not blob
        root = root.roott[fd]
        root.fback_t[fd] += [Fback]
        if len(root.fback_t[fd]) == len(root.node_):  # all original nodes term, fed back to root.fback_t
            feedback(root, fd)  # derH/ rng layer in sum2PP, deeper rng layers are appended by feedback


def comp_rng(iP_, rng):  # form new Ps and links, switch to rng+n to skip clustering?

    P_ = []
    for P in iP_:
        # trace mlinks:
        for derP in P.link_tH[-1][0]:
            _P = derP._P
            for _derP in _P.link_tH[-1][0]:  # next layer, of all links
                __P = _derP._P  # next layer of Ps
                distance = np.hypot(__P.yx[1]-P.yx[1], __P.yx[0]-P.yx[0])   # distance between mid points
                if distance > rng:
                    comp_P(P,__P, fd=0, derP=distance)  # distance=S, mostly lateral, relative to L for eval?
        P_ += [P]
    return P_

def comp_der(P_):  # keep same Ps and links, increment link derH, then P derH in sum2PP

    for P in P_:
        link_ = []  # exclude uplinks to nested _sub_PPs

        while P.link_tH[-1][1]:
            derP = P.link_tH[-1][1].pop()  # trace dlinks
            _P = derP._P
            if len(_P.derH) == len(P.derH):  # if _P is not _sub_PP:
                comp_P(_P,P, fd=1, derP=derP)
                link_+= [derP]

        P.link_tH[-1][1] = link_  # exclude uplinks to nested _sub_PPs
    return P_


def add_ext_layer(PP, P, fd, fr):

    if not fr:
        P.root_tH += [[None, None]]
    if fd:
        P.link_H = copy(P.link_H[-1])  # inherit from last layer
        mlink_, dlink_ = P.link_tH[-1]
        P.link_tH += [[copy(mlink_), copy(dlink_)]]
    else:
        P.link_H += []
        P.link_tH += [[], []]  # rng+ forms new links

    P.root_tH[-1][fd] = PP

''' 
lateral splicing, draft:
spliced_link_ = []
__P = link_.pop()
for _P in link_.pop():
    spliced_link_ += lat_comp_P(__P, _P)  # comp uplinks, merge if close and similar, return merged __P
    __P = _P
'''

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

