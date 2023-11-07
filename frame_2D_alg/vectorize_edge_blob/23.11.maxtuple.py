def vectorize_root(blob, verbose):  # vectorization pipeline is 3 composition levels of cross-comp,clustering:

    edge, adj_Pt_ = slice_edge(blob, verbose)  # lateral kernel cross-comp -> P clustering
    comp_P_(edge, adj_Pt_)  # vertical, lateral-overlap P cross-comp -> PP clustering
    # PP cross-comp -> discontinuous graph clustering:
    for fd, node_ in enumerate(edge.node_):  # node_t
        if edge.valt[fd] * (len(node_)-1)*(edge.rng+1) <= G_aves[fd] * edge.rdnt[fd]: continue
        G_= []
        for PP in node_:  # convert PP_t CPPs to Cgraphs:
            derH,valt,rdnt = PP.derH,PP.valt,PP.rdnt
            derH[:] = [  # convert to ptuple_tv_: [[ptuplet, maxtuplet, valt, rdnt]]:
                [[mtuple,dtuple], maxtuplet, [sum(mtuple),sum(dtuple)],
                 [sum([0 if m>d else 1 for m,d in zip(mtuple,dtuple)]), sum([0 if d > m else 1 for m, d in zip(mtuple,dtuple)])]
                ] for (mtuple,dtuple), maxtuplet in zip(derH, reform_maxtuplet_([PP.node_]))
            ]
            G_ += [Cgraph( ptuple=PP.ptuple, derH=[derH,valt,rdnt,[0,0]], valHt=[[valt[0]],[valt[1]]], rdnHt=[[rdnt[0]],[rdnt[1]]],
                           L=PP.ptuple[-1], box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box))]
        node_ = G_
        edge.valHt[0][0] = edge.valt[0]; edge.rdnHt[0][0] = edge.rdnt[0]  # copy
        agg_recursion(None, edge, node_, fd=0)  # edge.node_ = graph_t, micro and macro recursive

def sum2graph(root, cG_, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    graph = Cgraph(root=root, fd=fd, L=len(cG_))  # n nodes, transplant both node roots
    SubH = [[],[0,0],[1,1]]; Mval,Dval, Mrdn,Drdn = 0,0,0,0
    Link_= []
    for i, (G,link_) in enumerate(cG_[3]):
        # sum nodes in graph:
        sum_box(graph.box, G.box)
        sum_ptuple(graph.ptuple, G.ptuple)
        sum_derHv(graph.derH, G.derH, base_rdn=1)
        sum_aggHv(graph.aggH, G.aggH, base_rdn=1)
        sum_Hts(graph.valHt,graph.rdnHt, G.valHt,G.rdnHt)
        # sum external links:
        subH=[[],[0,0],[1,1]]; mval,dval, mrdn,drdn = 0,0, 0,0
        for derG in link_:
            if derG.valt[fd] > G_aves[fd] * derG.rdnt[fd]:  # sum positive links only:
                _subH = derG.subH
                (_mval,_dval),(_mrdn,_drdn) = valt,rdnt = derG.valt,derG.rdnt
                if derG not in Link_:
                    derG.roott[fd] = graph
                    sum_subHv(SubH, [_subH,valt,rdnt] , base_rdn=1)  # new aggLev, not from nodes: links overlap
                    Mval+=_mval; Dval+=_dval; Mrdn+=_mrdn; Drdn+=_drdn
                    graph.A[0] += derG.A[0]; graph.A[1] += derG.A[1]; graph.S += derG.S
                    Link_ += [derG]
                mval+=_mval; dval+=_dval; mrdn+=_mrdn; drdn+=_drdn
                sum_subHv(subH, [_subH,valt,rdnt], base_rdn=1, fneg = G is derG.G)  # fneg: reverse link sign
                sum_box(G.box, derG.G[0].box if derG._G[0] is G else derG._G[0].box)  # derG.G is proto-graph
        # from G links:
        if subH: G.aggH += [subH]
        G.i = i
        G.valHt[0]+=[mval]; G.valHt[1]+=[dval]; G.rdnHt[0]+=[mrdn]; G.rdnHt[1]+=[drdn]
        # G.root[fd] = graph  # replace cG_
        graph.node_t += [G]  # converted to node_t by feedback
    # + link layer:
    graph.link_ = Link_  # use in sub_recursion, as with PPs, no link_?
    graph.valHt[0]+=[Mval]; graph.valHt[1]+=[Dval]; graph.rdnHt[0]+=[Mrdn]; graph.rdnHt[1]+=[Drdn]
    Y,X,Y0,X0,Yn,Xn = graph.box
    graph.box[:2] = [(Y0+Yn)/2, (X0+Xn)/2]

    return graph

def sum_Hts(ValHt, RdnHt, valHt, rdnHt):
    # loop m,d Hs, add combined decayed lower H/layer?
    for ValH,valH, RdnH,rdnH in zip(ValHt,valHt, RdnHt,rdnHt):
        ValH[:] = [V+v for V,v in zip_longest(ValH, valH, fillvalue=0)]
        RdnH[:] = [R+r for R,r in zip_longest(RdnH, rdnH, fillvalue=0)]

def comp_G(link_, _G, G, fd):

    Mval,Dval, Mrdn,Drdn = 0,0, 1,1
    link = CderG( _G=_G, G=G)
    # keep separate P ptuple and PP derH, empty derH in single-P G, + empty aggH in single-PP G
    # / P:
    mtuple, dtuple, Mtuple, Dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1, fagg=1)
    for ptuple, maxtuple, Dec in zip((mtuple,Mtuple),(dtuple,Dtuple)):
        for par,max in zip(ptuple,maxtuple):
            try: Dec += par/max  # link decay coef: m|d / max, base self/same
            except ZeroDivisionError: pass
    mval, dval = sum(mtuple), sum(abs(d) for d in dtuple)  # mval is signed, m=-min in comp x sign
    mrdn = dval>mval; drdn = dval<=mval
    derLay0 = [[mtuple,dtuple],[mval,dval],[mrdn,drdn]]
    Mval+=mval; Dval+=dval; Mrdn += mrdn; Drdn += drdn
    # / PP:
    _derH,derH = _G.derH,G.derH
    if _derH[0] and derH[0]:  # empty in single-node Gs
        dderH, valt, rdnt = comp_derHv(_derH[0], derH[0], rn=1)
        # or pack this in comp_derHv?:
        for ((mtuple,dtuple),(Mtuple,Dtuple)),_,_ in dderH:
            for ptuple, maxtuple, Dec in zip((mtuple,Mtuple),(dtuple,Dtuple)):
                for par, max in zip(ptuple, maxtuple):
                    try: Dec += par/max  # link decay coef: m|d / max, base self/same
                    except ZeroDivisionError: pass
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
    else:
        dderH = []
    derH = [[derLay0]+dderH, [Mval,Dval], [Mrdn,Drdn]]  # appendleft derLay0 from comp_ptuple
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval],[Mrdn,Drdn])
    SubH = [der_ext, derH]  # two init layers of SubH, higher layers added by comp_aggH:
    # / G:
    if fd:  # else no aggH yet?
        subH, valt, rdnt = comp_aggHv(_G.aggH, G.aggH, rn=1)
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
        link.subH = SubH+subH  # append higher subLayers: list of der_ext | derH s
        link.valt = [Mval,Dval]; link.rdnt = [Mrdn,Drdn]  # complete proto-link
        link_[G] += [link]
        link_[_G]+= [link]
    elif Mval > ave_Gm or Dval > ave_Gd:  # or sum?
        link.subH = SubH
        link.valt = [Mval,Dval]; link.rdnt = [Mrdn,Drdn] # complete proto-link
        link_[G] += [link]
        link_[_G]+= [link]
        # dict: key=G, values=derGs
        # link maxt is computed from maxtuplets in Gs
    return Mval,Dval, Mrdn,Drdn

# draft:
def comp_aggHv(_aggH, aggH, rn):  # no separate ext
    SubH = []
    Mval,Dval, Mrdn,Drdn = 0,0,1,1

    for _lev, lev in zip_longest(_aggH, aggH, fillvalue=[]):  # compare common lower layer|sublayer derHs
        if _lev and lev:  # also if lower-layers match: Mval > ave * Mrdn?
            # compare dsubH only:
            dsubH, valt,rdnt = comp_subHv(_lev, lev, rn)  # no more valt and rdnt in subH now
            SubH += dsubH  # flatten to keep subH
            mval,dval = valt; Mval += mval; Dval += dval
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + mval <= dval

    return SubH, [Mval,Dval],[Mrdn,Drdn]

def comp_subHv(_subH, subH, rn):
    DerH = []
    Mval,Dval, Mrdn,Drdn = 0,0,1,1

    for _lay, lay in zip(_subH, subH):  # compare common lower layer|sublayer derHs
        # if lower-layers match: Mval > ave * Mrdn?
        if _lay[0] and isinstance(_lay[0][0],list):
            # _lay[0] is derHv
            dderH, valt, rdnt, maxt = comp_derHv(_lay[0], lay[0], rn)
            DerH += [[dderH, valt, rdnt, maxt]]  # flat derH
            mval,dval = valt; Mval += mval; Dval += dval
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + dval <= mval
        else:  # _lay[0][0] is L, comp dext:
            DerH += [comp_ext(_lay[1],lay[1],[Mval,Dval],[Mrdn,Drdn])]
            # pack extt as ptuple
    return DerH, [Mval,Dval],[Mrdn,Drdn]  # new layer,= 1/2 combined derH

def comp_derHv(_derH, derH, rn):  # derH is a list of der layers or sub-layers, each is ptuple_tv

    dderH = []  # or not-missing comparand: xor?
    Mval,Dval, Mrdn,Drdn, maxM,maxD = 0,0,1,1,0,0

    for _lay, lay in zip(_derH, derH):  # compare common lower der layers | sublayers in derHs, if lower-layers match?
        # compare dtuples only, mtuples are for evaluation:
        mtuple, dtuple, Mtuple, Dtuple = comp_dtuple(_lay[0][1], lay[0][1], rn, fagg=1)
        # sum params:
        mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
        mrdn = dval > mval; drdn = dval < mval
        maxm = sum(Mtuple); maxd = sum(Dtuple)
        Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn; maxM+= maxm; maxD+= maxd
        ptuple_tv = [[mtuple,dtuple],[mval,dval],[mrdn,drdn],[maxm,maxd]]  # or += [Mtuple,Dtuple] for future comp?
        dderH += [ptuple_tv]  # derLay

    return dderH, [Mval,Dval],[Mrdn,Drdn],[maxM,maxD]  # new derLayer,= 1/2 combined derH

def reform_dect_(node_t_):

    Dect_ = [[0,0]]  # default 1st layer dect from ptuple
    S_ = [0]  # accum count per derLay

    while True:  # unpack lower node_ layer
        sub_node_t_ = []  # subPP_t or P_ per higher PP
        for node_t in node_t_:
            if node_t[0] and isinstance(node_t[0],list) and (isinstance(node_t[0][0],CPP) or (node_t[1] and isinstance(node_t[1][0],CPP))):
                for fd, PP_ in enumerate(node_t):  # node_t is [sub_PPm_,sub_PPd_]
                    for PP in PP_:
                        for link in PP.link_:  # get compared Ps and find param maxes
                            _P, P = link._P, link.P
                            S_[0] += 6  # 6 pars
                            for _par, par in zip(_P.ptuple, P.ptuple):
                                if hasattr(par,"__len__"):
                                    Dect_[0][0] +=2; Dect_[0][1] +=2  # angle
                                else:  # scalar
                                    if _par and par:  # link decay coef: m|d / max, base self/same:
                                        Dect_[0][0] += par/ (abs(_par)+abs(par)); Dect_[0][1] += par/ max(_par,par)
                                    else:
                                        Dect_[0][0] += 1; Dect_[0][1] += 1  # prevent /0
                            for i, (_tuplet,tuplet, Dect) in enumerate(zip_longest(_P.derH,P.derH, Dect_[1:], fillvalue=None)):
                                if _tuplet and tuplet:                              # loop derLays bottom-up
                                    mdec, ddec = 0,0
                                    for fd,(_ptuple,ptuple) in enumerate(zip(_tuplet,tuplet)):
                                        for _par, par in zip(_ptuple,ptuple):
                                            if fd: mdec += par/ max(_par,par) if _par and par else 1  # prevent /0
                                            else:  ddec += par/ (abs(_par)+abs(par)) if _par and par else 1
                                    if Dect:
                                        Dect_[i][0] += mdec; Dect_[i][1] += mdec; S_[i] += 6  # accum 6 pars
                                    else:
                                        Dect_ += [[mdec,ddec]]; S_ += [6]  # extend both
                        if PP.node_[0] and isinstance(PP.node_[0], list) and (isinstance(PP.node_[0][0], CPP)
                            or (PP.node_[1] and isinstance(PP.node_[1][0], CPP))):
                            sub_node_t_ += [PP.node_]
        if sub_node_t_:
            node_t_ = sub_node_t_  # deeper nesting layer
        else:
            break

    # skip when S = 0
    return [[Dect[0]/S, Dect[1]/S] if S>0 else Dect for Dect, S in zip(Dect_, S_)]  # normalize by n sum

def sum_ptuple(Ptuple, ptuple, fneg=0):
    _I, _G, _M, _Ma, (_Dy, _Dx), _L = Ptuple
    I, G, M, Ma, (Dy, Dx), L = ptuple
    if fneg: Ptuple[:] = (_I-I, _G-G, _M-M, _Ma-Ma, [_Dy-Dy,_Dx-Dx], _L-L)
    else:    Ptuple[:] = (_I+I, _G+G, _M+M, _Ma+Ma, [_Dy+Dy,_Dx+Dx], _L+L)

def sum_dertuple(Ptuple, ptuple, fneg=0):
    _I, _G, _M, _Ma, _A, _L = Ptuple
    I, G, M, Ma, A, L = ptuple
    if fneg: Ptuple[:] = [_I-I, _G-G, _M-M, _Ma-Ma, _A-A, _L-L]
    else:    Ptuple[:] = [_I+I, _G+G, _M+M, _Ma+Ma, _A+A, _L+L]
    return   Ptuple
