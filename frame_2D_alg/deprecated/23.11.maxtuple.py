def vectorize_root(blob, verbose):  # vectorization pipeline is 3 composition levels of cross-comp,clustering:

    edge, adj_Pt_ = slice_edge(blob, verbose)  # lateral kernel cross-comp -> P clustering
    comp_P_(edge, adj_Pt_)  # vertical, lateral-overlap P cross-comp -> PP clustering
    # PP cross-comp -> discontinuous graph clustering:
    for fd, node_ in enumerate(edge.node_t):
        if edge.valt[fd] * (len(node_)-1) * (edge.rng+1) <= G_aves[fd] * edge.rdnt[fd]: continue
        G_ = []
        for PP in node_:  # convert select CPPs to Cgraphs:
            if PP.valt[fd] * (len(node_)-1) * (PP.rng+1) <= G_aves[fd] * PP.rdnt[fd]: continue
            derH,valt,rdnt = PP.derH,PP.valt,PP.rdnt
            decHt = reform_dect_(PP.node_t[0]+PP.node_t[1], PP.link_) if PP.link_ else [[1],[1]]
            derH[:] = [  # convert to ptuple_tv_: [[ptuplet,valt,rdnt,dect]]:
                [[mtuple,dtuple],
                 [sum(mtuple),sum(dtuple)],
                 [sum([m<d for m,d in zip(mtuple,dtuple)]), sum([d<=m for m,d in zip(mtuple,dtuple)])],
                 [mdec,ddec]]
                for (mtuple,dtuple), mdec,ddec in zip(derH, decHt[0],decHt[1])]
            G_ += [Cgraph( ptuple=PP.ptuple, derH=[derH,valt,rdnt,[0,0]], valHt=[[valt[0]],[valt[1]]], rdnHt=[[rdnt[0]],[rdnt[1]]],
                           decHt=decHt, L=PP.ptuple[-1], box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box),
                           link_=PP.link_, node_t=PP.node_t)]
        if G_:
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


def reform_dect_(node_, link_):

    Dec_t = [[],[]]  # default 1st layer dect from ptuple
    S_ = []  # accum count per derLay

    while True:  # unpack lower node_ layer
        for link in link_:  # get compared Ps and find param maxes
            _P, P = link._P, link.P
            _mmaxt,_dmaxt = [],[]
            for _par, par in zip(_P.ptuple, P.ptuple):
                if hasattr(par,"__len__"):
                    _mmaxt += [2]; _dmaxt += [2]  # angle
                else:  # scalar
                    _mmaxt += [max(abs(_par),abs(par))]; _dmaxt += [(abs(_par)+abs(par))]
            mmaxt_,dmaxt_ = [_mmaxt],[_dmaxt]  # append with maxes from all lower dertuples, empty if no comp
            L = 0
            rn  = len(_P.dert_)/ len(P.dert_)
            while len(_P.derH) > L and len(P.derH) > L:  # len derLay = len low Lays: 1,1,2,4. tuplets, map to _vmaxt_
                hL = max(2*L,1)  # init L=0
                _Lay,Lay, mDec_,dDec_ = _P.derH[L:hL],P.derH[L:hL],Dec_t[0][L:hL],Dec_t[1][L:hL]
                for _tuplet,tuplet,_mmaxt,_dmaxt, mDec,dDec in zip_longest(_Lay,Lay,mmaxt_,dmaxt_,mDec_,dDec_, fillvalue=None):
                    if not _tuplet or not tuplet: continue
                    mmaxt,dmaxt = [],[]; dect = [0,0]
                    for fd,(_ptuple, ptuple, vmax_) in enumerate(zip(_tuplet,tuplet,(_mmaxt,_dmaxt))):
                        for _par, par, vmax in zip(_ptuple,ptuple, vmax_):
                            dect[fd] += par*rn/vmax if vmax else 1  # link decay = val/max, 0 if no prior comp
                            if fd: dmaxt += [abs(_par)+abs(par*rn) if _par and par else 0]  # was not compared
                            else:  mmaxt += [max(abs(_par),abs(par*rn)) if _par and par else 0]
                    if mDec:
                        Dec_t[0][L]+=dect[0]; Dec_t[1][L]+=dect[1]; S_[L] += 6  # accum 6 pars
                    else:
                        Dec_t[0] +=[dect[0]]; Dec_t[1] +=[dect[1]]; S_ += [6]  # extend both
                    mmaxt_+=[mmaxt]; dmaxt_+=[dmaxt]  # from all lower layers
                    L += 1  # combined len of all lower layers = len next layer
        sub_node_, sub_link_ = [],[]
        for sub_PP in node_:
            sub_link_ += list(set(sub_link_ + sub_PP.link_))
            sub_node_ += sub_PP.node_t[0] + sub_PP.node_t[1]
        if not sub_link_: break
        link_ = sub_link_  # deeper nesting layer
        node_ = sub_node_
    # decHt:
    return [[mDec / S for mDec, S in zip(Dec_t[0], S_)], [dDec / S for dDec, S in zip(Dec_t[1], S_)]]  # normalize by n sum

def get_maxtt(_ptuple, ptuple):  # 0der params

    _I, _G, _M, _Ma, _, _L = _ptuple
    I, G, M, Ma, _, L = ptuple

    mmaxI, dmaxI = max(_I, I), _I + I
    mmaxG, dmaxG = max(_G, G), _G + G
    mmaxM, dmaxM = max(abs(_M), abs(M)), abs(_M) + abs(M)
    mmaxMa, dmaxMa = max(abs(_Ma), abs(Ma)), abs(_Ma) + abs(Ma)
    mmaxDa, dmaxDa = 2, 2
    mmaxL, dmaxL = max(_L, L), _L + L

    mmaxt = mmaxI, mmaxG, mmaxM, mmaxMa, mmaxDa, mmaxL
    dmaxt = dmaxI, dmaxG, dmaxM, dmaxMa, dmaxDa, dmaxL
    return mmaxt,dmaxt

def get_dect(_tuplet, tuplet, _mmaxt, _dmaxt, rn):
    # unpack:
    (_mI, _mG, _mM, _mMa, _mDa, _mL), (_dI, _dG, _dM, _dMa, _dDa, _dL) = _tuplet
    (mI, mG, mM, mMa, mDa, mL), (dI, dG, dM, dMa, dDa, dL) = tuplet
    _mmaxI, _mmaxG, _mmaxM, _mmaxMa, _mmaxDa, _mmaxL = _mmaxt
    _dmaxI, _dmaxG, _dmaxM, _dmaxMa, _dmaxDa, _dmaxL = _dmaxt
    # compute link decay dec = val/max, 0 if no prior comp
    mdect = [
        (mI*rn/_mmaxI) if _mmaxI else 1,
        (mG*rn/_mmaxG)if _mmaxG else 1,
        (mM*rn/_mmaxM) if _mmaxM else 1,
        (mMa*rn/_mmaxMa) if _mmaxMa else 1,
        (mDa*rn/_mmaxDa) if _mmaxDa else 1,
        (mL*rn/_mmaxL) if _mmaxL else 1 ]
    ddect = [
        (dI*rn/_dmaxI) if _dmaxI else 1,
        (dG*rn/_dmaxG) if _dmaxG else 1,
        (dM*rn/_dmaxM) if _dmaxM else 1,
        (dMa*rn/_dmaxMa) if _dmaxMa else 1,
        (dDa*rn/_dmaxDa) if _dmaxDa else 1,
        (dL*rn/_dmaxL) if _dmaxL else 1 ]
    # get maxtt
    mmaxt = [
        max(abs(_mI), abs(mI*rn)), max(abs(_mG), abs(mG*rn)), max(abs(_mM), abs(mM*rn)), max(abs(_mMa), abs(mMa*rn)), max(abs(_mDa), abs(mDa*rn)), max(abs(_mL), abs(mL*rn)),
        max(abs(_dI), abs(dI*rn)), max(abs(_dG), abs(dG*rn)), max(abs(_dM), abs(dM*rn)), max(abs(_dMa), abs(dMa*rn)), max(abs(_dDa), abs(dDa*rn)), max(abs(_dL), abs(dL*rn)) ]
    dmaxt = [
        abs(_mI)+abs(mI), abs(_mG)+abs(mG), abs(_mM)+abs(mM), abs(_mMa)+abs(mMa), abs(_mDa)+abs(mDa), abs(_mL)+abs(mL),
        abs(_dI)+abs(dI), abs(_dG)+abs(dG), abs(_dM)+abs(dM), abs(_dMa)+abs(dMa), abs(_dDa)+abs(dDa), abs(_dL)+abs(dL) ]

    return (mmaxt, dmaxt), (sum(mdect), sum(ddect))


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

def node_connect(iG_,link_,fd):  # sum surround values to define node connectivity, over incr.mediated links
    '''
    aggregate indirect links by associated nodes (vs. individually), iteratively recompute connectivity in multiple cycles,
    effectively blending direct and indirect connectivity measures for each node over time.
    In each cycle, connectivity per node includes aggregated contributions from the previous cycles, propagated through the network.
    Math: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/node_connect.png
    '''
    _Gt_ = []; ave = G_aves[fd]
    for G in iG_:
        valt,rdnt,dect = [0,0],[0,0], [0,0]; rim = copy(link_[G])  # all links that contain G
        for link in rim:
            if link.valt[fd] > ave * link.rdnt[fd]:  # skip negative links
                for i in 0,1:
                    valt[i] += link.valt[i]; rdnt[i] += link.rdnt[i]; dect[i] += link.dect[i]  # sum direct link vals
        _Gt_ += [[G, rim,valt,rdnt,dect, len(rim)]]  # no norm here
    _tVal,_tRdn = 0,0

    while True:  # eval same Gs,links, but with cross-accumulated node connectivity values
        tVal, tRdn = 0,0  # loop totals
        Gt_ = []
        for G, rim, ivalt, irdnt, idect, N in _Gt_:
            valt, rdnt, dect = [0,0],[0,0],[0,0]
            rim_val, rim_rdn = 0,0
            for n, link in enumerate(rim):
                if link.valt[fd] < ave * link.rdnt[fd]: continue  # skip negative links
                _G = link.G if link._G is G else link._G
                if _G not in iG_: continue
                _Gt = _Gt_[G.i]
                _G,_rim_,_valt,_rdnt,_dect = _Gt
                decay = link.dect[fd]  # node vals * relative link val:
                for i in 0,1:
                    linkV = _valt[i] * decay; valt[i]+=linkV
                    if fd==i: rim_val+=linkV
                    linkR = _rdnt[i] * decay; rdnt[i]+=linkR
                    if fd==i: rim_rdn+=linkR
                    dect[i] += link.dect[i]
            if rim:
                n += 1  # normalize for rim accum to prevent overflow, there's got to be a better way:
                for i in 0,1:
                    ivalt[i] = (ivalt[i] + valt[i]/n) / 2
                    irdnt[i] = (irdnt[i] + rdnt[i]/n) / 2
                    idect[i] = (idect[i] + dect[i]/n) / 2
            else: n = 0
            Gt_ += [[G, rim, ivalt,irdnt,idect, N+n]]
            tVal += rim_val
            tRdn += rim_rdn
        if tVal-_tVal <= ave * (tRdn-_tRdn):
            break
        _tVal,_tRdn = tVal,tRdn

    return Gt_

def segment_node_(root, Gt_, root_link_, fd, root_fd):  # eval rim links with summed surround vals for density-based clustering

    ave = G_aves[fd]
    graph_ = []  # graph += [node] if >ave (surround connectivity * relative value of link to any internal node)
    link_map = defaultdict(list)
    derG_ = []
    for derG in root_link_:
        if derG.valt[fd] > ave * derG.rdnt[fd]:
            link_map[derG.G] += [derG._G]  # key:Gt, vals: linked _Gts
            link_map[derG._G] += [derG.G]
            derG_ += [derG]  # filtered derG

def segment_node_par(root, Gt_, fd, root_fd):  # eval rim links with summed surround vals for density-based clustering

    # graph += [node] if >ave (surround connectivity * relative value of link to any internal node)
    igraph_ = []; ave = G_aves[fd]

    for Gt in Gt_:
        G,rimt,valt,rdnt,dect, uprimt, *_ = Gt
        subH = [[],[0,0],[1,1],[0,0]]
        Link_= defaultdict(list)
        A, S = [0,0],0
        for link in rimt[fd][G]:
            if link.valt[fd] > G_aves[fd] * link.rdnt[fd]:
                sum_subHv(subH, [link.subH,link.valt,link.rdnt,link.dect], base_rdn=1)
                Link_[G] += [link]; A[0] += link.A[0]; A[1] += link.A[1]; S += link.S
        grapht = [[Gt],{node:list(link_) for node,link_ in rimt[fd].items()}, copy(valt),copy(rdnt),copy(dect),A,S,subH,Link_]
        G.root[fd] = grapht; igraph_ += [grapht]
    _tVal,_tRdn = 0,0
    _graph_ = igraph_
    while True:
        tVal,tRdn = 0,0  # loop totals
        graph_ = []
        while _graph_:  # extend graph Rim
            grapht = _graph_.pop()
            nodet_,Rim, Valt,Rdnt,Dect, A,S, subH,link_ = grapht
            inVal,inRdn = 0,0  # in-graph: positive
            new_Rim = defaultdict(list)
            for link in get_link_(Rim):  # unique links
                if link.G is nodet_[0][0]:
                    Gt = Gt_[link.G.it[root_fd]]; _Gt = Gt_[link._G.it[root_fd]] if Gt[0] in root.node_t[root_fd] else None
                else:
                    Gt = Gt_[link._G.it[root_fd]]; _Gt = Gt_[link.G.it[root_fd]] if Gt[0] in root.node_t[root_fd] else None
                if _Gt is None: continue  # not in root.node_
                if _Gt in nodet_: continue
                # node match * surround M|D match: of potential in-graph position?
                comb_val = link.valt[fd] + get_match(Gt[2][fd],_Gt[2][fd])
                comb_rdn = link.rdnt[fd] + (Gt[3][fd] + _Gt[3][fd]) / 2
                # merge nodes
                if comb_val > ave*comb_rdn:
                    # sum links
                    _nodet_,_Rim,_Valt,_Rdnt,_Dect,_A,_S,_subH,_link_ = _Gt[0].root[fd]
                    if _Gt[0].root[fd] in grapht: grapht.remove(_Gt[0].root[fd])   # remove overlapping root
                    for _nodet in _nodet_: _nodet[0].root[fd] = grapht  # assign new merged root
                    sum_subHv(subH, _subH, base_rdn=1)
                    A[0] += _A[0]; A[1] += _A[1]; S += _S; link_.update(_link_)
                    for i in 0,1:
                        Valt[i] += _Valt[i]; Rdnt[i] += _Rdnt[i]; Dect[i] += _Dect[i]
                    inVal += _Valt[fd]; inRdn += _Rdnt[fd]
                    nodet_ += [__Gt for __Gt in _Gt[0].root[fd][0] if __Gt not in nodet_]
                    Rim.update(_Rim)
                    new_Rim.update(_Rim)
            tVal += inVal; tRdn += inRdn  # signed?
            if len(get_link_(new_Rim)) * inVal > ave * inRdn:
                # eval new_Rim for extension:
                graph_ += [[nodet_,new_Rim,Valt,Rdnt,Dect,A,S, subH, link_]]

        if len(graph_) * (tVal-_tVal) <= ave * (tRdn-_tRdn):  # even low-Val extension may be valuable if Rdn decreases?
            break
        _graph_ = graph_
        _tVal,_tRdn = tVal,_tRdn
    # -> Cgraphs if Val > ave * Rdn:
    return [sum2graph(root, graph, fd) for graph in graph_ if graph[2] > ave * graph[3]]


def form_graph_t(root, Valt,Rdnt, G_, link_, fd):  # form mgraphs and dgraphs of same-root nodes

    for G in G_: G.root = [None,None]  # replace with mcG_|dcG_ in segment_node_, replace with Cgraphs in sum2graph

    Gt_ = node_connect(G_, link_, fd)  # AKA Graph Convolution of Correlations
    graph_t = []
    for i in 0,1:
        if Valt[i] > ave * Rdnt[i]:  # else no clustering
            graph_t += [segment_node_(root, Gt_, i, fd)]  # if fd: node-mediated Correlation Clustering; add alt_graphs?
        else: graph_t += [[]]  # or G_?
    for i, graph_ in enumerate(graph_t):
        if graph_:
            for graph in graph_:
                if graph: sum2graph(root, graph, i)
        else:
            for G in G_: G.i_ += [None]  # no sub-root