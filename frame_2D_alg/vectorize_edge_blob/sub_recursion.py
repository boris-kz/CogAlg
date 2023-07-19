from itertools import zip_longest
import numpy as np
from copy import copy, deepcopy
from .filters import PP_aves, ave_nsubt
from .classes import CP, CPP
from .comp_slice import comp_P, form_PP_t, sum_derH
from dataclasses import replace

def sub_recursion_eval(root, PP_):  # fork PP_ in PP or blob, no derH in blob

    termt = [1,1]
    # PP_ in PP_t:
    for PP in PP_:
        sub_tt = []  # from rng+, der+
        fr = 0  # recursion in any fork
        for fder in 0,1:  # rng+ and der+:
            if len(PP.node_) > ave_nsubt[fder] and PP.valt[fder] > PP_aves[fder] * PP.rdnt[fder]:
                termt[fder] = 0
                if not fr:  # add link_tt and root_tt for both comp forks:
                    for P in PP.node_:
                        P.root_tt = [[None,None],[None,None]]  # replace root PPs with sub_PPs, not sure it will work here?
                        P.link_H += [[]]; P.link_tH += [[[],[]]]
                        # root_t and link_t of form forks are added in sub+:
                sub_tt += [sub_recursion(PP, PP_, fder)]  # comp_der|rng in PP->parLayer
                fr = 1
            else:
                sub_tt += [PP.node_]
                if isinstance(root, CPP):  # separate feedback per terminated comp fork:
                    root.fback_t[fder] += [[PP.derH, PP.valt, PP.rdnt]]
        if fr:
            PP.node_ = sub_tt
            # nested PP_ tuple from 2 comp forks, each returns sub_PP_t: 2 clustering forks, if taken
    return termt


def sub_recursion(PP, PP_, fder):  # evaluate PP for rng+ and der+, add layers to select sub_PPs

    comp_der(PP.node_) if fder else comp_rng(PP.node_, PP.rng+1)  # same else new P_ and links
    # eval all| last layer?:
    PP.rdnt[fder] += PP.valt[fder] - PP_aves[fder]*PP.rdnt[fder] > PP.valt[1-fder] - PP_aves[1-fder]*PP.rdnt[1-fder]
    sub_PP_t = form_PP_t(PP.node_, PP_, base_rdn=PP.rdnt[fder], fder=fder)  # replace node_ with sub_PPm_, sub_PPd_

    for fd, sub_PP_ in enumerate(sub_PP_t):  # not empty: len node_ > ave_nsubt[fd]
        for sPP in sub_PP_: sPP.root_tt[fder][fd] = PP
        termt = sub_recursion_eval(PP, sub_PP_)

        if any(termt):  # fder: comp fork, fd: form fork:
            for fder in 0,1:
                if termt[fder] and PP.fback_t[fder]:
                    feedback(PP, fder=fder, fd=fd)
                    # upward recursive extend root.derH, forward eval only
    return sub_PP_t

def feedback(root, fder, fd):  # append new der layers to root

    Fback = deepcopy(root.fback_t[fd].pop())
    # init with 1st fback: [derH,valt,rdnt], derH: [[mtuple,dtuple, mval,dval, mrdn, drdn]]
    while root.fback_t[fd]:
        sum_derH(Fback,root.fback_t[fd].pop(), base_rdn=0)
    sum_derH([root.derH, root.valt,root.rdnt], Fback, base_rdn=0)

    if isinstance(root.root_tt[fder][fd], CPP):  # not blob
        root = root.root_tt[fder][fd]
        root.fback_t[fd] += [Fback]
        if len(root.fback_t[fd]) == len(root.node_):  # all original nodes term, fed back to root.fback_t
            feedback(root, fder, fd)  # derH/ rng layer in sum2PP, deeper rng layers are appended by feedback


def comp_rng(iP_, rng):  # form new Ps and links, switch to rng+n to skip clustering?

    P_ = []
    for P in iP_:
        for derP in P.link_tH[-2][0]:  # scan lower-layer mlinks
            _P = derP._P
            for _derP in _P.link_tH[-2][0]:  # next layer of all links, also lower-layer?
                __P = _derP._P  # next layer of Ps
                distance = np.hypot(__P.yx[1]-P.yx[1], __P.yx[0]-P.yx[0])   # distance between mid points
                if distance > rng:
                    comp_P(P,__P, fder=0, derP=distance)  # distance=S, mostly lateral, relative to L for eval?
        P_ += [P]
    return P_

def comp_der(P_):  # keep same Ps and links, increment link derH, then P derH in sum2PP

    for P in P_:
        for derP in P.link_tH[-2][1]:  # scan lower-layer dlinks
            # comp extended derH of the same Ps, to sum in lower-composition sub_PPs:
            comp_P(derP._P,P, fder=1, derP=derP)
    return P_

''' 
lateral splicing of initial Ps, not needed: will be spliced in PPs through common forks? 
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