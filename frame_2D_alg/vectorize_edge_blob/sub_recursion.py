from itertools import zip_longest
from copy import copy, deepcopy
from .filters import PP_aves, ave_nsub, P_aves, G_aves
from .classes import CP, CPP
from .comp_slice import comp_P, form_PP_t, sum_vertuple, sum_layer, sum_derH


def sub_recursion_eval(root, PP_, fd):  # fork PP_ in PP or blob, no derH,valH,rdnH in blob

    term = 1
    for PP in PP_:
        if PP.valt[fd] > PP_aves[fd] * PP.rdnt[fd] and len(PP.P__) > ave_nsub:  # no select per ptuple
            term = 0
            sub_recursion(PP, fd)  # comp_der|rng in PP -> parLayer, sub_PPs
        elif isinstance(root, CPP):
            root.fd_ += [[PP.derH[-1], [fd], PP.valH[-1][fd], PP.rdnH[-1][fd]]]
            # feed back last layer, added in sum2PP
    if term and isinstance(root, CPP):
        feedback(root, fd)  # upward recursive extend root.derH, forward eval only


def feedback(root, fd):  # append new der layers to root

    Fback = root.fb_.pop()  # init with 1st fback ders: derH, fds, valH, rdnH
    while root.fb_:
        sum_ders(Fback, root.fb_.pop())  # sum fback in Fback, add Fback nesting if missing
    # append Fback params to root params:
    root.derH += Fback[0]
    root.fds  += Fback[1]
    root.valH += Fback[2]
    root.rdnH += Fback[3]
    if isinstance(root.root, CPP):
        root = root.root
        root.fb_ += [Fback]
        if len(root.fb_) == len(root.P__[fd]):  # all nodes terminated, fed back to root.fb_
            feedback(root, fd)
    '''
     PP.derH: layer) derH before feedback, indefinitely nested layer) derH) rngH... after feedback
     fb Ders: append to root.derH, nesting is incremented in recursive feedback.
    '''
# draft
def sum_ders(Fback, fback):  # sum or append fb in Fb, for deeper feedback:

    DerH, Fds, ValH, RdnH = Fback
    derH, fds, valH, rdnH = fback
    while True:
        for Lay, Fd, Val, Rdn, lay, fd, val, rdn in zip_longest(
            DerH, Fds, ValH, RdnH, derH, fds, valH, rdnH, fillvalue=None):  # loop bottom-up,
            # terminate summation at 1st None or Fd!=fd?
            # not sure:
            if lay != None and Lay != None and Fd==fd:
                sum_layer(Lay,lay); Val+=val; Rdn+=rdn
            # else append copy?

def sub_recursion(PP, fd):  # evaluate PP for rng+ and der+, add layers to select sub_PPs

    P__ = comp_der(PP.P__) if fd else comp_rng(PP.P__, PP.rng+1)   # returns top-down
    PP.rdnt[fd] += PP.valt[fd] > PP.valt[1-fd]
    # link Rdn += PP rdn?
    cP__ = [copy(P_) for P_ in P__]
    PP.P__ = form_PP_t(cP__,base_rdn=PP.rdnt[fd])  # P__ = sub_PPm_, sub_PPd_

    for fd, sub_PP_ in enumerate(PP.P__):
        if sub_PP_:  # der+ | rng+
            for sub_PP in sub_PP_: sub_PP.roott[fd] = PP
            sub_recursion_eval(PP, sub_PP_, fd=fd)
        '''
        if PP.valt[fd] > ave * PP.rdnt[fd]:  # adjusted by sub+, ave*agg_coef?
            agg_recursion_eval(PP, copy(sub_PP_), fd=fd)  # comp sub_PPs, form intermediate PPs
        else:
            feedback(PP, fd)  # add aggH, if any: 
        implicit nesting: rngH(derH / sub+fb, aggH(subH / agg+fb: subH is new order of rngH(derH?
        '''
# mderP * (ave_olp_L / olp_L)? or olp(_derP._P.L, derP.P.L)? gap: neg_olp, ave = olp-neg_olp?
# __Ps in rng+ are mediated by PP.rng layers of _Ps:

def comp_rng(iP__, rng):  # form new Ps and links in rng+ PP.P__, switch to rng+n to skip clustering?

    P__ = []
    for iP_ in reversed(iP__[:-rng]):  # lower compared row, follow uplinks, no uplinks in last rng rows
        P_ = []
        for P in iP_:
            link_, link_m, link_d = [],[],[]  # for new P
            Valt = [0,0]; Rdnt = [0,0]
            DerH = [[[],[]]]  # Mt,Dt
            for iderP in P.link_t[0]:  # mlinks
                _P = iderP._P
                for _derP in _P.link_t[0]:  # next layer of mlinks
                    __P = _derP._P  # next layer of Ps
                    comp_P(P,__P, link_,link_m,link_d, Valt, Rdnt, DerH, fd=0)
            if Valt[0] > P_aves[0] * Rdnt[0]:
                # add new P in rng+ PP:
                P_ += [CP(ptuple=deepcopy(P.ptuple), derH=[DerH], dert_=copy(P.dert_), fds=copy(P.fds)+[0], box=copy(P.box),
                      valt=Valt, rdnt=Rdnt, link_=link_, link_t=[link_m,link_d])]
        P__+= [P_]
    return P__

def comp_der(iP__):  # form new Ps and links in rng+ PP.P__, extend their link.derH, P.derH, _P.derH

    P__ = []
    for iP_ in reversed(iP__[:-1]):  # lower compared row, follow uplinks, no uplinks in last row
        P_ = []
        for P in iP_:
            DerH, DerLay = [],[]  # new lower and last layer
            link_, link_m, link_d = [],[],[]  # for new P
            Valt = [0,0]; Rdnt = [0,0]
            for iderP in P.link_t[1]:  # dlinks
                if iderP._P.link_t[1]:  # else no _P links and derH to compare
                    _P = iderP._P
                    comp_P(_P, P, link_,link_m,link_d, Valt, Rdnt, DerH, fd=1, derP=iderP, DerLay=DerLay)
            if Valt[1] > P_aves[1] * Rdnt[1]:
                # add new P in der+ PP:
                P_ += [CP(ptuple=deepcopy(P.ptuple), derH=DerH+[DerLay], dert_=copy(P.dert_), fds=copy(P.fds)+[1],
                          box=copy(P.box), valt=Valt, rdnt=Rdnt, rdnlink_=link_, link_t=[link_m,link_d])]
        P__+= [P_]
    return P__