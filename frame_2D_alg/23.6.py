def reval_P_(P__, fd):  # prune qPP by (link_ + mediated link__) val

    prune_ = []; Val=0; reval = 0  # comb PP value and recursion value

    for P_ in P__:
        for P in P_:
            P_val = 0; remove_ = []
            for link in P.link_t[fd]:
                # recursive mediated link layers eval-> med_valH:
                _,_,med_valH = med_eval(link._P.link_t[fd], old_link_=[], med_valH=[], fd=fd)
                # link val + mlinks val: single med order, no med_valH in comp_slice?:
                link_val = link.valT[fd] + sum([mlink.valT[fd] for mlink in link._P.link_t[fd]]) * med_decay  # + med_valH
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

def sum2PP(qPP, base_rdn, fd):  # sum Ps and links into PP

    P__,_,_ = qPP  # proto-PP is a list
    PP = CPP(box=copy(P__[0][0].box), fd=fd, P__ = P__)
    DerT,ValT,RdnT = [[],[]],[[],[]],[[],[]]
    # accum:
    for P_ in P__:  # top-down
        for P in P_:  # left-to-right
            P.roott[fd] = PP
            sum_ptuple(PP.ptuple, P.ptuple)
            if P.derT[0]:  # P links and both forks are not empty
                for i in 0,1:
                    if isinstance(P.valT[0], list):  # der+: H = 1fork) 1layer before feedback
                        sum_unpack([DerT[i],ValT[i],RdnT[i]], [P.derT[i],P.valT[i],P.rdnT[i]])
                    else:  # rng+: 1 vertuple
                        if isinstance(ValT[0], list):  # we init as list, so we need to change it into ints here
                            ValT = [0,0]; RdnT = [0,0]
                        sum_ptuple(DerT[i], P.derT[i]); ValT[i]+=P.valT[i]; RdnT[i]+=P.rdnT[i]
                PP.link_ += P.link_
                for Link_,link_ in zip(PP.link_t, P.link_t):
                    Link_ += link_  # all unique links in PP, to replace n
            Y0,Yn,X0,Xn = PP.box; y0,yn,x0,xn = P.box
            PP.box = [min(Y0,y0), max(Yn,yn), min(X0,x0), max(Xn,xn)]
    if DerT[0]:
        PP.derT = DerT; PP.valT = ValT; PP.rdnT = RdnT  # we init PP above, so their params should be always empty and hence we can just assign them here?
        """
        for i in 0,1:
            PP.derT[i]+=DerT[i]; PP.valT[i]+=ValT[i]; PP.rdnT[i]+=RdnT[i]  # they should be in same depth, so bracket is not needed here
        """
    return PP