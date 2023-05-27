from itertools import zip_longest
from copy import copy, deepcopy
from .filters import PP_aves, ave_nsub, P_aves, G_aves
from .classes import CP, CPP
from .comp_slice import comp_P, form_PP_t, sum_unpack


def sub_recursion_eval(root, PP_, fd):  # fork PP_ in PP or blob, no derH,valH,rdnH in blob

    term = 1
    for PP in PP_:
        if PP.valH[-1][fd] > PP_aves[fd] * PP.rdnH[-1][fd] and len(PP.P__) > ave_nsub:  # no select per ptuple
            term = 0
            sub_recursion(PP, fd)  # comp_der|rng in PP -> parLayer, sub_PPs
        elif isinstance(root, CPP):
            root.fb_ += [[[PP.derH[-1]], PP.valH[-1][fd],PP.rdnH[-1][fd]]]  # [derH, valH, rdnH]
            # feedback last layer, added in sum2PP
    if term and isinstance(root, CPP):
        feedback(root, fd)  # upward recursive extend root.derH, forward eval only


def feedback(root, fd):  # append new der layers to root

    Fback = root.fb_.pop()  # init with 1st fback ders: derH, fds, valH, rdnH
    while root.fb_:
        sum_unpack(Fback, root.fb_.pop())  # sum | append fback in Fback
    derH,valH,rdnH = Fback
    root.derH+=derH; root.valH+=valH; root.rdnH+=rdnH  # concat Fback layers to root layers

    if isinstance(root.roott[fd], CPP):
        root = root.roott[fd]
        root.fb_ += [Fback]
        if len(root.fb_) == len(root.P__[fd]):  # all nodes term, fed back to root.fb_
            feedback(root, fd)  # derH=[1st layer] in sum2PP, deeper layers(forks appended by recursive feedback


def sub_recursion(PP, fd):  # evaluate PP for rng+ and der+, add layers to select sub_PPs

    P__ = comp_der(PP.P__) if fd else comp_rng(PP.P__, PP.rng+1)   # returns top-down
    PP.rdnH[-1][fd] += PP.valH[-1][fd] > PP.valH[-1][1-fd]
    # link Rdn += PP rdn?
    cP__ = [copy(P_) for P_ in P__]
    PP.P__ = form_PP_t(cP__,base_rdn=PP.rdnH[-1][fd])  # P__ = sub_PPm_, sub_PPd_

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
# __Ps: above PP.rng layers of _Ps:
def comp_rng(iP__, rng):  # form new Ps and links in rng+ PP.P__, switch to rng+n to skip clustering?

    P__ = []
    for iP_ in reversed(iP__[:-rng]):  # lower compared row, follow uplinks, no uplinks in last rng rows
        P_ = []
        for P in iP_:
            link_, link_m, link_d = [],[],[]  # for new P
            Lay, ValH, RdnH = [[[],[]]],[[0,0]],[[1,1]]
            # not sure
            for iderP in P.link_t[0]:  # mlinks
                _P = iderP._P
                for _derP in _P.link_t[0]:  # next layer of mlinks
                    __P = _derP._P  # next layer of Ps
                    comp_P(P,__P, link_,link_m,link_d, Lay, ValH, RdnH, fd=0)
            if ValH[0][0] > P_aves[0] * RdnH[0][0]:  # not sure
                # add new P in rng+ PP:
                P_ += [CP(ptuple=deepcopy(P.ptuple), derH=[Lay], dert_=copy(P.dert_), fd=0, box=copy(P.box),
                          valH=ValH, rdnH=RdnH, link_=link_, link_t=[link_m,link_d])]
        P__+= [P_]
    return P__

def comp_der(iP__):  # form new Ps and links in rng+ PP.P__, extend their link.derH, P.derH, _P.derH

    P__ = []
    for iP_ in reversed(iP__[:-1]):  # lower compared row, follow uplinks, no uplinks in last row
        P_ = []
        for P in iP_:
            link_, link_m, link_d = [],[],[]  # for new P
            Lay, ValH, RdnH = [[[],[]]],[[0,0]],[[1,1]]
            # not sure
            for iderP in P.link_t[1]:  # dlinks
                if iderP._P.link_t[1]:  # else no _P links and derH to compare
                    _P = iderP._P
                    comp_P(_P, P, link_,link_m,link_d, Lay, ValH, RdnH, fd=1, derP=iderP)
            if ValH[0][1] > P_aves[1] * RdnH[0][1]:
                # add new P in der+ PP:
                P_ += [CP(ptuple=deepcopy(P.ptuple), derH=[P.derH+[Lay]], dert_=copy(P.dert_), fd=1, box=copy(P.box),
                          valH=ValH, rdnH=RdnH, rdnlink_=link_, link_t=[link_m,link_d])]
        P__+= [P_]
    return P__