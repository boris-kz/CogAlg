from copy import copy, deepcopy
from .filters import PP_aves, ave_nsub, P_aves, G_aves
from .classes import CP, CPP
from .comp_slice import comp_P, form_PP_t, sum_derH


def sub_recursion_eval(root, PP_, fd):  # fork PP_ in PP or blob, no rngH, valt,rdnt in blob?

    for PP in PP_:
        # fork val, rdn, no select per ptuple:
        if PP.valt[fd] > PP_aves[fd] * PP.rdnt[fd] and len(PP.P__) > ave_nsub:
            sub_recursion(PP, fd)  # comp_der|rng in PP -> parLayer, sub_PPs
        else:
            PP.fterm = 1
            if all([[node.fterm for node in root.P__[fd]]]) and isinstance(root,CPP):  # not blob
                feedback(PP, fd)  # always starts with PP.P__= CPs, updates root rngH, valt, rdnt


def sub_recursion(PP, fd):  # evaluate PP for rng+ and der+, add layers to select sub_PPs

    P__ = comp_der(PP.P__) if fd else comp_rng(PP.P__, PP.rng+1)   # returns top-down
    PP.rdnt[fd] += PP.valt[fd] > PP.valt[1-fd]
    # link Rdn += PP rdn?
    cP__ = [copy(P_) for P_ in P__]
    PP.P__ = form_PP_t(cP__,base_rdn=PP.rdnt[fd])  # P__ = sub_PPm_, sub_PPd_

    for fd, sub_PP_ in enumerate(PP.P__):
        for sub_PP in sub_PP_:
            sub_PP.root = PP
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
                P_ += [CP(ptuple=deepcopy(P.ptuple), derH=[DerH], dert_=copy(P.dert_), fds=copy(P.fds)+[0], x0=P.x0, y0=P.y0,
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
                          x0=P.x0, y0=P.y0, valt=Valt, rdnt=Rdnt, rdnlink_=link_, link_t=[link_m,link_d])]
        P__+= [P_]
    return P__


# draft:
def feedback(root, fd):  # bottom-up update root.rngH, breadth-first, separate for each fork?

    fCP = 1  # bottom layer
    VAL = 0; RDN = 1  # sum across layers
    RngH, DerH = [],[]  # new rng|der lays, not in root.derH
    root.derH = [root.derH]

    while True:
        root.fterm = 1
        Val = 0; Rdn = 1
        DerLay = []  # not in root.derH
        for PP in root.P__[fd]:
            if fCP:
                for P_ in PP.P__:
                    for P in P_:  # sum in root, PP was updated in sum2PP:
                        sum_derH(DerLay, P.derH[-1]); Val += P.valt[fd]; Rdn += P.rdnt[fd]
            else:  # sum in PP:
                sum_derH(DerLay, PP.derH[-1]); Val += PP.valt[fd]; Rdn += PP.rdnt[fd]
        DerH += [DerLay]
        if fd:  # der+
            root.derH[-1] += [DerLay]   # new der lay in last derH
        else:  # rng+, derH->rngH:
            root.derH += RngH if RngH else [DerH]  # append new rng lays or rng lay = terminated DerH?
            RngH += [DerH]  # not sure; comp in agg+ only

        root.valt[fd] += Val; root.rdnt[fd] += Rdn
        VAL += Val; RDN += Rdn
        root = root.root
        fCP = 0  # higher layers
        # continue while sub+ terminated in all nodes and root is not blob:
        if VAL/RDN < G_aves[root.fds[-1]] or not isinstance(root,CPP) or not all([[node.fterm for node in root.P__[fd]]]):
            break
        # locals are lost if not all nodes are terminated, they should be kept to accumulate for future terminated nodes?