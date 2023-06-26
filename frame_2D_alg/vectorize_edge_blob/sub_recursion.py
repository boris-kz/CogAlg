from itertools import zip_longest
import numpy as np
from copy import copy, deepcopy
from .filters import PP_aves, ave_nsub, P_aves, G_aves
from .classes import CP, CPP
from .comp_slice import comp_P, form_PP_t, sum_unpack
from dataclasses import replace

def sub_recursion_eval(root, PP_, fd):  # fork PP_ in PP or blob, no derT,valT,rdnT in blob

    term = 1
    for PP in PP_:
        if np.sum(PP.valT[fd]) > PP_aves[fd] * np.sum(PP.rdnT[fd]) and len(PP.P_) > ave_nsub:
            term = 0
            sub_recursion(PP, fd)  # comp_der|rng in PP -> parLayer, sub_PPs
        elif isinstance(root, CPP):
            root.fback_ += [[[PP.derT[fd][-1]], PP.valT[fd][-1], PP.rdnT[fd][-1]]]  # [derT,valT,rdnT]
            # feedback last layer, added in sum2PP
    if term and isinstance(root, CPP):
        feedback(root, fd)  # upward recursive extend root.derT, forward eval only


def feedback(root, fd):  # append new der layers to root

    Fback = root.fback_.pop()  # init with 1st fback derT,valT,rdnT
    while root.fback_:
        sum_unpack(Fback, root.fback_.pop())  # sum | append fback in Fback
    derT,valT,rdnT = Fback
    for i in 0,1:
        root.derT[i]+=derT[i]; root.valT[i]+=valT[i]; root.rdnT[i]+=rdnT[i]  # concat Fback layers to root layers

    if isinstance(root.roott[fd], CPP):
        root = root.roott[fd]
        root.fback_ += [Fback]
        if len(root.fback_) == len(root.P__[fd]):  # all nodes term, fed back to root.fback_
            feedback(root, fd)  # derT=[1st layer] in sum2PP, deeper layers(forks appended by recursive feedback

# not revised
def sub_recursion(PP, fd):  # evaluate PP for rng+ and der+, add layers to select sub_PPs

    if fd:
        if not isinstance(PP.valT[0], list): nest(PP)  # PP created from 1st rng+ is not nested too
        [nest(P) for P in PP.P_]  # add layers and forks?
        P_ = comp_der(PP.P_)  # returns top-down
        PP.rdnT[fd][-1][-1][-1] += np.sum(PP.valT[fd][-1]) > np.sum(PP.valT[1 - fd][-1])
        base_rdn = PP.rdnT[fd][-1][-1][-1]  # link Rdn += PP rdn?
    else:
        P_ = comp_rng(PP.P_, PP.rng + 1)
        PP.rdnT[fd] += PP.valT[fd] > PP.valT[1 - fd]
        base_rdn = PP.rdnT[fd]

    cP_ = [replace(P, roott=[None, None], link_t=[[], []]) for P in P_]  # reassign roots to sub_PPs
    PP.P_ = form_PP_t(cP_, base_rdn=base_rdn)  # replace P_ with sub_PPm_, sub_PPd_

    for fd, sub_PP_ in enumerate(PP.P_):
        if sub_PP_:
            for sub_PP in sub_PP_: sub_PP.roott[fd] = PP
            sub_recursion_eval(PP, sub_PP_, fd=fd)


def comp_rng(iP_, rng):  # form new Ps and links, switch to rng+n to skip clustering?

    P_ = []
    for P in iP_:
        cP = CP(ptuple=deepcopy(P.ptuple), dert_=copy(P.dert_))  # replace links, then derT in sum2PP
        # trace mlinks:
        for derP in P.link_t[0]:
            _P = derP._P
            for _derP in _P.link_t[0]:  # next layer of mlinks
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


def nest(P, ddepth=3):  # default ddepth is nest 3 times: tuple->fork->layer->H
    # agg+ adds depth: number brackets before the tested bracket: P.valT[0], P.valT[0][0], etc?

    if not isinstance(P.valT[0],list):
        curr_depth = 0
        while curr_depth < ddepth:
            P.derT[0]=[P.derT[0]]; P.valT[0]=[P.valT[0]]; P.rdnT[0]=[P.rdnT[0]]
            P.derT[1]=[P.derT[1]]; P.valT[1]=[P.valT[1]]; P.rdnT[1]=[P.rdnT[1]]
            curr_depth += 1

        if isinstance(P, CP):
            for derP in P.link_t[1]:
                curr_depth = 0
                while curr_depth < ddepth:
                    derP.derT[0]=[derP.derT[0]]; derP.valT[0]=[derP.valT[0]]; derP.rdnT[0]=[derP.rdnT[0]]
                    derP.derT[1]=[derP.derT[1]]; derP.valT[1]=[derP.valT[1]]; derP.rdnT[1]=[derP.rdnT[1]]
                    curr_depth += 1