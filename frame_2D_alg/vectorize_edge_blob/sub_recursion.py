from itertools import zip_longest
from copy import copy, deepcopy
from .filters import PP_aves, ave_nsub, P_aves, G_aves
from .classes import CP, CPP
from .comp_slice import comp_P, form_PP_t, sum_vertuple, sum_layer, sum_derH


def sub_recursion_eval(root, PP_, fd):  # fork PP_ in PP or blob, no rngH, valt,rdnt in blob?

    term = 1
    for PP in PP_:
        # fork val, rdn, no select per ptuple:
        if PP.valt[fd] > PP_aves[fd] * PP.rdnt[fd] and len(PP.P__) > ave_nsub:
            term = 0
            sub_recursion(PP, fd)  # comp_der|rng in PP -> parLayer, sub_PPs
        else:
            Ders=[0,[]]  # init feedback from PP.P__ Ps, nesting depth=layer
            for P_ in PP.P__[1:]:  # no derH in top row
                for P in P_:
                    sum_layer(Ders[1], P.derH[-1])
            PP.derH = [PP.derH]  # derH->rngH
            if isinstance(root, CPP):
                # exclude blob
                if fd: root.derH[-1] += [Ders[1]]  # append derLay in last rngLay
                else:  root.derH += [[Ders[1]]]  # append rngLay derH in root rngH
                root.fb_ += [[Ders, PP.valt[fd], PP.rdnt[fd]]]  # from sum2PP
                # to extend root.derH:
    if term and isinstance(root, CPP):
        feedback(root, fd)  # upward recursive, forward eval only

def feedback(root, fd):  # append new der layers to root

    Val=0; Rdn=0; Ders = [0,[]]  # Ders = [nesting,feedback]

    while root.fb_:
        ders, val, rdn = root.fb_.pop()
        sum_ders(Ders[1],ders[1], Ders[0],ders[0], fd)  # Ders,ders are layer|derH|rngH, append Ders nesting if missing
        root.valt[fd] += val; root.rdnt[fd] += rdn
        Val += val; Rdn += rdn

    root.derH = [root.derH]  # derH->rngH
    sum_ders(root.derH,Ders[1], 2,Ders[0])  # root.derH is layer) derH) rngH, Ders is layer|derH|rngH

    if isinstance(root.root, CPP):
        root = root.root
        root.fb_ += [[Ders, Val, Rdn]]
        if len(root.fb_) == len(root.P__[fd]):  # all nodes terminated, fed back to root.fb_
            feedback(root, fd)
    '''
     PP.derH: layer) derH before feedback, or layer) derH) rngH.. after feedback
     fb Ders: Initial nesting is layer, incremented in recursive feedback.
    '''
# draft
def sum_ders(iDers, iders, Nest, nest, fd):  # nest: nesting depth of ders: 0 if layer, 1 if derH, 2 if rngH..

    # equalize nesting:
    while Nest > nest:  # sum in last element of Ders
        Ders = Ders[-1]; Nest -= 1
    while Nest < nest:  # add nesting to Ders
        Ders[:] = [Ders]; Nest += 1
    # draft, combine with equalize nesting?:
    if fd:
        Ders = iDers[-1]; ders=iders; Nest -= 1
    else:
        Ders = iDers; ders = [iders]; nest += 1

    # sum or append ders in Ders, for deeper feedback:
    for Der,der in zip_longest(Ders,ders, fillvalue=None):
        if der != None:
            if Der != None:
                if nest==0: sum_vertuple(Der,der)
                else: sum_ders(Der,der,Nest-1,nest-1, fd)
            else:
                Ders += [deepcopy(der)]


def sub_recursion(PP, fd):  # evaluate PP for rng+ and der+, add layers to select sub_PPs

    P__ = comp_der(PP.P__) if fd else comp_rng(PP.P__, PP.rng+1)   # returns top-down
    PP.rdnt[fd] += PP.valt[fd] > PP.valt[1-fd]
    # link Rdn += PP rdn?
    cP__ = [copy(P_) for P_ in P__]
    PP.P__ = form_PP_t(cP__,base_rdn=PP.rdnt[fd])  # P__ = sub_PPm_, sub_PPd_

    for fd, sub_PP_ in enumerate(PP.P__):
        if sub_PP_:  # der+ | rng+
            for sub_PP in sub_PP_: sub_PP.root = PP
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