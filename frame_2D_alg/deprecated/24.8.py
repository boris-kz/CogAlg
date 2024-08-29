def comp_N(Link, iEt, rng, rev=None):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = rev is not None  # compared links have binary relative direction?
    _N, N = Link.nodet; rn = _N.n / N.n

    if fd:  # CLs, form single layer:
        DLay = _N.derH.comp_H(N.derH, rn, fagg=1)  # new link derH = local dH
        N.mdext = comp_ext(2,2, _N.S,N.S/rn, _N.angle, N.angle if rev else [-d for d in N.angle])  # reverse for left link
        N.Et = np.array(N.Et) + DLay.Et + N.mdext.Et
        DLay.root = Link
    else:   # CGs
        mdlat = comp_latuple(_N.latuple, N.latuple, rn,fagg=1)
        mdLay = _N.mdLay.comp_md_(N.mdLay, rn, fagg=1)  # not in CG of links?
        mdext = comp_ext(len(_N.node_), len(N.node_), _N.S, N.S / rn, _N.A, N.A)
        DLay = CH(  # 1st derLay
            md_t=[mdlat,mdLay,mdext], Et=np.array(mdlat.Et)+mdLay.Et+mdext.Et, Rt=np.array(mdlat.Rt)+mdLay.Rt+mdext.Rt, n=2.5)
        mdlat.root = DLay; mdLay.root = DLay; mdext.root = DLay
        if _N.derH and N.derH:
            dLay = _N.derH.comp_H(N.derH,rn,fagg=1,frev=rev)
            DLay.add_H(dLay)  # also append discrete higher subLays in dLay.H_[0], if any?
            # no comp extH: current ders
    if fd: Link.derH.append_(DLay)
    else:  Link.derH = DLay
    iEt[:] = np.add(iEt,DLay.Et)  # init eval rng+ and form_graph_t by total m|d?
    for i in 0,1:
        Val, Rdn = DLay.Et[i::2]
        if Val > G_aves[i] * Rdn * (rng+1): Link.ft[i] = 1  # fork inclusion tuple
        _N.Et[i] += Val; N.Et[i] += Val  # not selective
        _N.Et[2+i] += Rdn; N.Et[2+i] += Rdn  # per fork link in both Gs
        # if select fork links: iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dH.Et[i::2])]
    if any(Link.ft):
        Link.n = min(_N.n,N.n)  # comp shared layers
        Link.yx = np.add(_N.yx, N.yx) / 2
        Link.S += (_N.S + N.S) / 2
        for rev, node in zip((0,1),(_N,N)):  # ?reversed Link direction
            if fd:
                if len(node.rimt_) == rng:
                    node.rimt_[-1][1-rev] += [[Link,rev]]  # add in last rng layer, opposite to _N,N dir
                else: node.rimt_ += [[[[Link,rev]],[]]] if dir else [[[],[[Link,rev]]]]  # add rng layer
            else:
                if len(node.extH.H) == rng:
                    node.rim_[-1] += [[Link, rev]]  # accum last rng layer
                else: node.rim_ += [[[Link, rev]]]  # init rng layer

        return True

def comp_core(_node_, node_, fmerge=1):  # compare partial graphs in merge or kLay in rng_kern_

    dderH = CH()
    fd = isinstance(_node_[0], CL)
    _pars = sum_N_(_node_,fd) if fmerge else _node_
    pars = sum_N_(node_,fd) if fmerge else node_
    n, L, S, A, derH = pars[:5]; _n,_L,_S,_A,_derH = _pars[:5]
    rn = _n/n
    mdext = comp_ext(_L,L, _S,S/rn, _A,A)
    dderH.n = mdext.n;  dderH.Et = np.array(mdext.Et); dderH.Rt = np.array(mdext.Rt)
    if fd:
        dderH.H = [[mdext]]
    else:
        _latuple, _mdLay = _pars[5:]; latuple, mdLay = pars[5:]
        if any(_latuple[:5]) and any(latuple[:5]):  # latuple is empty in CL
            mdlat = comp_latuple(_latuple, latuple, rn, fagg=1)
            dderH.n+=mdlat.n; dderH.Et+=mdlat.Et; dderH.Rt+=mdlat.Rt
        if _mdLay and mdLay:
            mdlay = _mdLay.comp_md_(mdLay, rn, fagg=1)
            dderH.n+=mdlay.n; dderH.Et+=mdlay.Et; dderH.Rt+=mdlay.Rt
    if _derH and derH:
        ddderH = _derH.comp_H(derH, rn, fagg=1)  # append and sum new dderH to base dderH
        dderH.H += ddderH.H  # merge derlay
        dderH.n+=ddderH.n; dderH.Et+=ddderH.Et; dderH.Rt+=ddderH.Rt

    return dderH

def comp_L(_pars, pars):  # compare partial graphs in merge

    _n,_L,_S,_A,_derH = _pars; n, L, S, A, derH = pars
    rn = _n / n
    dderH = derH.comp_H(derH, rn, fagg=1)  # new link derH = local dH
    mdext = comp_ext(_L,L, _S,S/rn, _A,A)
    return dderH, mdext


def segment_N_(root, iN_, fd, rng):  # form proto sub-graphs in iN_: Link_ if fd else G_

    # cluster by weight of shared links, initially single linkage, + similarity of partial clusters in merge
    N_ = []
    max_ = []
    for N in iN_:   # init Gt per N:
        # mediator: iN if fd else iL
        med_ = N.nodet if fd else [Lt[0] for Lt in (N.rim_[-1] if isinstance(N,CG) else N.rimt_[-1][0] + N.rimt_[-1][1])]
        if fd:  # get links of nodet
            eN_ = [Lt[0] for _N in med_ for Lt in (_N.rim_[-1] if isinstance(_N,CG) else _N.rimt_[-1][0]+_N.rimt_[-1][1])]
        else:   # get nodes of rim links
            eN_ = [_N for _L in med_ for _N in _L.nodet]
        ext_N_ = [e for e in eN_ if e is not N and e in iN_]
        _N_t = [ext_N_, [N]]
        Gt = [[N], [], copy(med_), _N_t, [0,0,0,0]]  # [node_, link_, Lrim, Nrim_t, Et]
        N.root = Gt; N_ += [Gt]
        emax_ = []  # if exemplar G|Ls:
        for eN in _N_t[0]:
            eEt,Et = (eN.derH.Et, N.derH.Et) if fd else (eN.extH.Et, N.extH.Et)
            if eEt[fd] >= Et[fd] or eN in max_: emax_ += [eN]  # _N if _N == N
        if not emax_: max_ += [N]  # no higher-val neighbors
        # add rrim mediation: V * k * max_rng?
    max_ = [N.root for N in max_]
    for Gt in N_: Gt[3][0] = [_N.root for _N in Gt[3][0]]  # replace ext Ns with their Gts
    for Gt in max_ if max_ else N_:
        # merge connected _Gts if >ave shared external links (Lrim) + internal similarity (summed node_)
        node_, link_, Lrim, Nrim_t, Et = Gt
        Nrim = Nrim_t[0]  # ext N Gts, connected by Gt-external links, not added to node_ yet
        for _Gt, _L in zip(Nrim, Lrim):
            if _Gt not in N_:
                continue  # was merged
            oL_ = set(Lrim).intersection(set(_Gt[2])).union([_L])  # shared external links + potential _L, or oL_ = [Lr[0] for Lr in _Gt[2] if Lr in Lrim]
            oV = sum([L.derH.Et[fd] - ave * L.derH.Et[2+fd] for L in oL_])
            # eval by Nrim similarity = oV + olp between G,_G,
            # ?pre-eval by max olp: _node_ = _Gt[0]; _Nrim = _Gt[3][0],
            # if len(Nrim)/len(node_) > ave_L or len(_Nrim)/len(_node_) > ave_L:
            sN_ = set(node_); _sN_ = set(_Gt[0])
            # or int_N_, ext_N_ is not in node_:
            oN_ = sN_.intersection(_sN_)  # ext N_ overlap
            xN_ = list(sN_- oN_)  # exclusive node_: remove overlap
            _xN_ = list(_sN_- oN_)
            if _xN_ and xN_:
                dderH = comp_G( sum_N_(_xN_), sum_N_(xN_));  dderH.node_ = xN_+_xN_  # ?
                oV += (dderH.Et[fd] - ave * dderH.Et[2+fd])  # norm by R, * dist_coef * agg_coef?
            if oV > ave:
                link_ += [_L]
                merge(Gt,_Gt); N_.remove(_Gt)

    return [sum2graph(root, Gt, fd, rng) for Gt in N_]

def sum_N_(N_):  # sum params of partial grapht for comp_G in merge

    N = N_[0]
    fd = isinstance(N, CL)
    n = N.n; S = N.S
    L, A = (N.span, N.angle) if fd else (len(N.node_), N.A)
    if not fd:
        latuple = deepcopy(N.latuple)
        mdLay = CH().copy(N.mdLay)
    derH = CH().copy(N.derH) if N.derH else None
    for N in N_[1:]:
        if not fd:
            add_lat(latuple, N.latuple)
            mdLay.add_md_(N.mdLay)
        n += N.n; S += N.S
        L += N.span if fd else len(N.node_)
        A = [Angle+angle for Angle,angle in zip(A, N.angle if fd else N.A)]
        if N.derH: derH.add_H(N.derH)

    return n, L, S, A, None if fd else latuple, None if fd else mdLay, derH

def merge(Gt, gt):

    N_,L_, Lrim, Nrim_t, Et = Gt
    n_,l_, lrim, nrim_t, et = gt
    N_ += n_
    L_ += l_  # internal, no overlap
    Lrim[:] = list(set(Lrim + lrim))  # exclude shared external links, direction doesn't matter?
    Nrim_t[:] = [[G for G in nrim_t[0] if G not in Nrim_t[0]], list(set(Nrim_t[1] + nrim_t[1]))]  # exclude shared external nodes
    Et[:] = np.add(Et,et)

def sum2graph(root, grapht, fd, rng):  # sum node and link params into graph, aggH in agg+ or player in sub+

    N_,L_,_,_,Et = grapht  # [node_, link_, Lrim, Nrim_t, Et]
    graph = CG(fd=fd, node_=node_, link_= [] if fd else L_, rng=rng, Et=Et)  # reset link_ if fd?
    graph.root = root
    extH = CH(node_=node_)  # convert to graph derH
    yx = [0,0]
    for N in N_:
        graph.area += N.area
        graph.box = extend_box(graph.box, N.box)
        if isinstance(N,CG):
            graph.add_md_t(N.md_t)
            add_lat(graph.latuple, N.latuple)
        graph.n += N.n  # non-derH accumulation?
        if N.derH: graph.derH.add_H(N.derH)
        if not fd:
            N.root = graph  # root is assigned to links if fd else to nodes
            extH.add_H(N.extH) if extH else extH.append_(N.extH, flat=1)  # G.extH.i is not relevant right?
        yx = np.add(yx, N.yx)
    L = len(N_)
    yx = np.divide(yx,L); graph.yx = yx
    graph.aRad = sum([np.hypot(*np.subtract(yx,N.yx)) for N in N_]) / L  # average distance between graph center and node center
    for link in L_:  # sum last layer of unique current-layer links
        graph.S += link.span
        graph.A = np.add(graph.A,link.angle)  # np.add(graph.A, [-link.angle[0],-link.angle[1]] if rev else link.angle)
        if fd:
            link.root = graph
            extH.add_H(link.derH) if extH else extH.append_(link.derH, flat=1)
    graph.derH.append_(extH, flat=0)  # graph derH = node derHs + [summed Link_ derHs], may be nested by rng
    if fd:
        # assign alt graphs from d graph, after both linked m and d graphs are formed
        for link in graph.link_:
            mgraph = link.root
            if mgraph:
                for fd, (G, alt_G) in enumerate(((mgraph,graph), (graph,mgraph))):  # bilateral assign:
                    if G not in alt_G.alt_graph_:
                        G.alt_graph_ += [alt_G]
    return graph

def comp_N(Link, iEt, rng, rev=None):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = rev is not None  # compared links have binary relative direction?
    _N, N = Link.nodet; rn = _N.n/N.n
    # no comp extH, it's current ders
    if fd:  # CLs
        DLay = _N.derH.comp_H(N.derH, rn, fagg=1)  # new link derH = local dH
        _A,A = _N.angle, N.angle if rev else [-d for d in N.angle] # reverse if left link
    else:   # CGs
        DLay = comp_G([_N.n,len(_N.node_),_N.S,_N.A,_N.latuple,_N.mdLay,_N.derH],
                      [N.n, len(N.node_), N.S, N.A, N.latuple, N.mdLay, N.derH])
        DLay.root = Link
        _A,A = _N.A,N.A
    DLay.node_ = [_N,N]
    Link.mdext = comp_ext(2,2, _N.S,N.S/rn, _A,A)
    if fd:
        Link.derH.append_(DLay)
    else:  Link.derH = DLay
    iEt[:] = np.add(iEt,DLay.Et)  # init eval rng+ and form_graph_t by total m|d?
    for i in 0,1:
        Val, Rdn = DLay.Et[i::2]
        if Val > G_aves[i] * Rdn * (rng+1): Link.ft[i] = 1  # fork inclusion tuple
        _N.Et[i] += Val; N.Et[i] += Val  # not selective
        _N.Et[2+i] += Rdn; N.Et[2+i] += Rdn  # per fork link in both Gs
        # if select fork links: iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dH.Et[i::2])]
    if any(Link.ft):
        Link.n = min(_N.n,N.n)  # comp shared layers
        Link.yx = np.add(_N.yx, N.yx) / 2
        Link.S += (_N.S + N.S) / 2
        for rev, node in zip((0,1),(_N,N)):  # ?reversed Link direction
            if fd:
                if len(node.rimt_) == rng:
                    node.rimt_[-1][1-rev] += [[Link,rev]]  # add in last rng layer, opposite to _N,N dir
                else: node.rimt_ += [[[[Link,rev]],[]]] if dir else [[[],[[Link,rev]]]]  # add rng layer
            else:
                if len(node.extH.H) == rng:
                    node.rim_[-1] += [[Link, rev]]  # accum last rng layer
                else: node.rim_ += [[[Link, rev]]]  # init rng layer

        return True
'''
rL_ = list(set([Lt[0] for N in rN_ for Lt in N.rim]))
Lt_ = [(L, mN_t) for L, mN_t in zip(L_, mN_t_) if any(mN_t)]; if Lt_: L_,_mN_t_ = map(list, zip(*Lt_))  # map list to convert tuple from zip(*)?

# form exemplar seg_ to segment (parallelize) clustering:
    if max_:
        seg_ = [[] for _ in max_]  # non-maxes assigned to the nearest max
        # not revised
        for Gt in N_:
            if Gt in max_: continue
            node_,link_, Lrim, Nrim, Et = Gt
            yx = np.mean([N.yx for N in (link_ if fd else node_)], axis=0)  # mean yx of pre-graph
            min_dist = np.inf  # init min distance
            for max in max_:
                _node_,_link_,_Lrim,_Nrim,_Et = max
                _yx = np.mean([N.yx for N in (link_ if fd else node_)], axis=0)  # mean yx of max
                dy, dx = np.subtract(_yx, yx)
                dist = np.hypot(dy, dx)  # distance between max and Gt
                if dist < min_dist: new_max = max  # nearest max for current Gt in N_
            max_index = max_.index(new_max)
            seg_[max_index] += [Gt]
'''
def rng_kern_(N_, rng):  # comp Gs summed in kernels, ~ graph CNN without backprop, not for CLs

    _G_ = []
    Et = [0,0,0,0]
    for N in N_:
        if hasattr(N,'crim'): N.rim += N.crim
        N.crim = []  # current rng links, add in comp_N
    # comp_N:
    for (_G, G) in list(combinations(N_,r=2)):
        if _G in [g for visited_ in G.visited__ for g in visited_]:  # compared in any rng++
            continue
        dy,dx = np.subtract(_G.yx,G.yx)
        dist = np.hypot(dy,dx)
        aRad = (G.aRad+_G.aRad) / 2  # ave G radius
        # eval relative distance between G centers:
        if dist / max(aRad,1) <= max_dist * rng:
            for _g,g in (_G,G),(G,_G):
                if len(g.elay.H) == rng:
                    g.visited__[-1] += [_g]
                else: g.visited__ += [[_g]]  # init layer
            Link = CL(nodet=[_G,G], S=2,A=[dy,dx], box=extend_box(G.box,_G.box))
            if comp_N(Link, Et, rng):
                for g in _G,G:
                    if g not in _G_: _G_ += [g]
    G_ = []  # init conv kernels:
    for G in (_G_):
        krim = []
        for link,rev in G.crim:  # form new krim from current-rng links
            if link.ft[0]:  # must be mlink
                _G = link.nodet[0] if link.nodet[1] is G else link.nodet[1]
                krim += [_G]
                if len(G.elay.H)==rng: G.elay.H[-1].add_H(link.derH)
                else:                  G.elay.append_(link.derH)
                G._kLay = sum_kLay(G,_G); _G._kLay = sum_kLay(_G,G)  # next krim comparands
        if krim:
            if rng>1: G.kHH[-1] += [krim]  # kH = lays(nodes
            else:     G.kHH = [[krim]]
            G_ += [G]
    Gd_ = copy(G_)  # Gs with 1st layer kH, dLay, _kLay
    _G_ = G_; n=0  # n higher krims
    # convolution: Def,Sum,Comp kernel rim, in separate loops for bilateral G,_G assign:
    while True:
        G_ = []
        for G in _G_:  # += krim
            G.kHH[-1] += [[]]; G.visited__ += [[]]
        for G in _G_:
            #  append G.kHH[-1][-1]:
            for _G in G.kHH[-1][-2]:
                for link, rev in _G.crim:
                    __G = link.nodet[0] if link.nodet[1] is G else link.nodet[1]
                    if __G in _G_:
                        if __G not in G.kHH[-1][-1] + [g for visited_ in G.visited__ for g in visited_]:
                            # bilateral add layer of unique mediated nodes
                            G.kHH[-1][-1] += [__G]; __G.kHH[-1][-1] += [G]
                            for g,_g in zip((G,__G),(__G,G)):
                                g.visited__[-1] += [_g]
                                if g not in G_:  # in G_ only if in visited__[-1]
                                    G_ += [g]
        for G in G_: G.visited__ += [[]]
        for G in G_: # sum kLay:
            for _G in G.kHH[-1][-1]:  # add last krim
                if _G in G.visited__[-1] or _G not in _G_:
                    continue  # Gs krim appended when _G was G
                G.visited__[-1] += [_G]; _G.visited__[-1] += [G]
                # sum alt G lower kLay:
                G.kLay = sum_kLay(G,_G); _G.kLay = sum_kLay(_G,G)
        for G in G_: G.visited__[-1] = []
        for G in G_:
            for _G in G.kHH[-1][0]:  # convo in direct kernel only
                if _G in G.visited__[-1] or _G not in G_: continue
                G.visited__[-1] += [_G]; _G.visited__[-1] += [G]
                # comp G kLay -> rng derLay:
                rlay = comp_pars(_G._kLay, G._kLay)
                if rlay.Et[0] > ave * rlay.Et[2] * (rng+n):  # layers add cost
                    _G.elay.add_H(rlay); G.elay.add_H(rlay)  # bilateral
        _G_ = G_; G_ = []
        for G in _G_:  # eval dLay
            G.visited__.pop()  # loop-specific layer
            if G.elay.Et[0] > ave * G.elay.Et[2] * (rng+n+1):
                G_ += [G]
        if G_:
            for G in G_: G._kLay = G.kLay  # comp in next krng
            _G_ = G_; n += 1
        else:
            for G in Gd_:
                G.rim += G.crim; G.visited__.pop()  # kH - specific layer
                delattr(G,'_kLay'); delattr(G,'crim')
                if hasattr(G,"kLay)"): delattr(G,'kLay')
            break
    return Gd_, Et  # all Gs with dLay added in 1st krim

def agg_recursion(root, N_, fL, rng=1):  # fL: compare node-mediated links, else < rng-distant CGs
    '''
    der+ forms Link_, link.node_ -> dual trees of matching links in der+rng+, complex higher links
    rng+ keeps N_ + distant M / G.M, less likely: term by >d/<m, less complex
    '''
    N_,Et,rng = rng_link_(N_) if fL else rng_node_(N_, rng)  # N_ = L_ if fL else G_, both rng++
    if len(N_) > ave_L:
        node_t = form_graph_t(root, N_, Et,rng)  # fork eval, depth-first sub+, sub_Gs += fback_
        if node_t:
            for fd, node_ in zip((0,1), node_t):  # after sub++ in form_graph_t
                N_ = []
                for N in node_:  # prune, select max node_ derH.i
                    derH = N.derH
                    if derH.Et[fd] > G_aves[fd] * derH.Et[2+fd]:
                        for i,lay in enumerate(sorted(derH.H, key=lambda lay:lay.Et[fd], reverse=True)):
                            di = lay.i-i # lay.i: index in H, di: der rdn - value rdn
                            lay.Et[2+fd] += di  # der rdn -> val rdn
                            if not i:  # max val lay
                                N.node_ = lay.node_; derH.ii = lay.i  # exemplar lay index
                        N.elay = CH()  # reset for agg+ below
                        N_ += [N]
                if root.derH.Et[0] * (max(0,(len(N_)-1)*root.rng)) > G_aves[1]*root.derH.Et[2]:
                    # agg+rng+, val *= n comparands, forms CGs:
                    agg_recursion(root, N_, fL=0)
            root.node_[:] = node_t
        # else keep root.node_

def form_graph_t(root, N_, Et, rng):  # segment N_ to Nm_, Nd_

    node_t = []
    for fd in 0,1:
        # N_ is Link_ if fd else G_
        if Et[fd] > ave * Et[2+fd] * rng:
            if isinstance(N_[0],CG):  # link.root is empty?
                 for G in N_: G.root = []  # new root will be intermediate graph
            # G: Ls/D: directional if fd, else Ns/M: symmetric, sum in node:
            graph_ = segment_N_(root, N_, fd, rng)
            for graph in graph_:
                Q = graph.link_ if fd else graph.node_  # xcomp -> max_dist * rng+1, comp links if fd
                if len(Q) > ave_L and graph.derH.Et[fd] > G_aves[fd] * graph.derH.Et[fd+2] * rng:
                    set_attrs(Q)  # add/reset sub+ attrs
                    agg_recursion(graph, Q, fL=isinstance(Q[0],CL), rng=rng)  # rng+, recursive in node_, no sub-clustering in link_?
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]
    for fd, graph_ in enumerate(node_t):  # combined-fork fb
        for graph in graph_:
            if graph.derH:  # single G's graph doesn't have any derH.H to feedback
                root.fback_t[fd] += [graph.derH] if fd else [graph.derH.H[-1]]  # der+ forms new links, rng+ adds new layer
            # sub+-> sub root-> init root
    if any(root.fback_t): feedback(root)

    return node_t

def set_attrs(Q):
    for e in Q:
        if isinstance(e,CL):
            e.rimt_ = [[[],[]]]  # der+'rng+ is not recursive
            e.med = 1  # med rng = len rimt_?
        else:
            e.rim = [e.rim, []]  # add nesting, rng layer / rng+'rng+
            e.visited_ = []
        e.derH.append_(e.elay)
        e.elay = CH()  # set in sum2graph
        e.root = None
        e.Et = [0,0,0,0]
        e.aRad = 0

def rng_link_(_L_):  # comp CLs: der+'rng+ in root.link_ rim_t node rims: directional and node-mediated link tracing

    _mN_t_ = [[[L.nodet[0]],[L.nodet[1]]] for L in _L_]  # rim-mediating nodes in both directions
    HEt = [0,0,0,0]
    rL_ = []
    rng = 1
    while True:
        Et = [0,0,0,0]
        mN_t_ = [[[],[]] for _ in _L_]  # new rng lay of mediating nodes, traced from all prior layers?
        for L, _mN_t, mN_t in zip(_L_, _mN_t_, mN_t_):
            for rev, _mN_, mN_ in zip((0,1), _mN_t, mN_t):
                # comp L, _Ls: nodet mN 1st rim, -> rng+ _Ls/ rng+ mm..Ns, flatten rim_s:
                rim_ = [n.rim_ if isinstance(n,CG) else n.rimt_[0][0] + n.rimt_[0][1] for n in _mN_]
                for rim in rim_:
                    for _L,_rev in rim:  # _L is reversed relative to its 2nd node
                        if _L is L or _L in L.visited_: continue
                        if not hasattr(_L,"rimt_"): set_attrs([_L])  # _L not in root.link_, same derivation
                        L.visited_ += [_L]; _L.visited_ += [L]
                        dy,dx = np.subtract(_L.yx, L.yx)
                        Link = CL(nodet=[_L,L], S=2, A=[dy,dx], box=extend_box(_L.box, L.box))
                        # L.rim_t += new Link
                        if comp_N(Link, Et, rng, rev ^ _rev):  # negate ds if only one L is reversed
                            # L += rng+'mediating nodes, link orders: nodet < L < rimt_, mN.rim || L
                            mN_ += _L.nodet  # get _Ls in mN.rim
                            if _L not in _L_:
                                _L_ += [_L]; mN_t_ += [[[],[]]]  # not in root
                            elif _L not in rL_: rL_ += [_L]
                            if L not in rL_:    rL_ += [L]
                            mN_t_[_L_.index(_L)][1 - rev] += L.nodet
                            for node in (L, _L):
                                node.elay.add_H(Link.derH)  # if lay/rng++, elif fagg: derH.H[der][rng]?
        L_, _mN_t_ = [],[]
        for L, mN_t in zip(_L_, mN_t_):
            if any(mN_t):
                L_ += [L]; _mN_t_ += [mN_t]
        if L_:
            _L_ = L_; rng += 1
        else:
            break
    return list(set(rL_)), Et, rng

def agg_recursion(root, N_):  # calls both cross-comp forks, first rng++

    mH, Et = rng_node_(root, N_)  # recursive rng+ and layer clustering
    Htt = [[mH,Et]]
    if Et[1] > G_aves[1] * Et[3]:
        L_ = set_attrs([Lt[0] for N in N_ for Lt in N.rim_[-1]])
        # rng++ comp, cluster links from rng_node_:
        dH,dEt = rng_link_(root, L_)
        Et = np.add(Et, dEt)
        Htt += [[dH, dEt]]
    else:
        Htt += [[]]  # empty dH and dEt
        L_ = []
    root.link_ = L_  # root.node_ is replaced in-place?
    # 2 rng++ comp forks -> 4 cluster fork hierarchies:
    for fd, Q, Ht in zip((0,1), (N_,L_), Htt):
        for N in Q: # update lay derR to valR:
            for i, lay in enumerate(sorted(N.derH.H, key=lambda lay: lay.Et[fd], reverse=True)):
                di = lay.i - i  # lay.i: index in H
                lay.Et[2+fd] += di  # derR-valR
                if not i:  # max val lay
                    N.node_ = lay.node_; N.derH.ii = lay.i  # exemplar lay index
            N.elay = CH()   # for agg+:
        for rng, H in enumerate(Ht[0]):  # two cluster forks per comp fork
            (mG_, dG_), et = H  # each layer of [graph_t, Et]
            for i, G_ in zip((0,1), (mG_,dG_)):
                while len(G_) > ave_L and Et[i] > G_aves[i] * Et[2+i] * rng:
                    # agg+/ rng layer, recursive?
                    hHtt, hEt = agg_recursion(root, G_)
                    if hHtt[0][0]:  # mH is not empty?
                        Et = np.add(Et,hEt)
                        G_[:] = hHtt  # else keep G_

def agg_recursion(root, aggH):  # fork rng++, two clustering forks per layer: two separate recursion forks?

    inG_, ilG_, iL_, iEt = aggH[0]  # top input aggLay, fixed syntax
    mLay, dLay, LL_, LEt = [],[],[],[0,0,0,0]  # new aggLay
    # or aggLay is rngH: combine lays?

    for fd, N_ in zip((0,1), (inG_,ilG_)):
        # rng++ cross-comp in both forks of top agg lay:
        rngH, L_, Et = rng_node_(root, N_) if isinstance(N_[0],CG) else rng_link_(root, N_)
        root.link_ = L_
        for N in N_:
            for f in 0,1:  # in rng++ elay: fd derR -> valR:
                for i, lay in enumerate(sorted(N.elay.H, key=lambda lay: lay.Et[fd], reverse=True)):
                    di = lay.i - i  # lay.i: index in H
                    lay.Et[2+f] += di  # derR-valR
                    if not i:  # max value lay
                        N.node_ = lay.node_; N.derH.it[f] = lay.i  # assigns exemplar lay index per fd
        # aggLay is rngH: combine lays?
        hrH = []
        hEt = [0,0,0,0]
        for rng, rngLay in enumerate(rngH):
            rnG_,rlG_,rL_,rEt = rngLay
            for rfd,_G_ in zip((0,1),(rnG_,rlG_)):
                dEt = rEt  # val / agg++/ rlay fd
                while dEt[rfd] > G_aves[rfd] * dEt[2+rfd] * rng:  # and (np.add(len(_G_)) > ave_L: sum len elements in _G_, nested below?
                    layH = agg_recursion(root, rngLay)
                    if layH:
                        dnG_,dlG_,dL_,dEt = layH[0]  # aggLay added to rngLay
                        if dEt[0] > G_aves[0] * dEt[2]:
                            rEt = np.add(rEt,dEt)
                            hEt = np.add(hEt,rEt)
                            _G_ = [dnG_,dlG_]  # for next agg+, additional nesting?
            if hrH:
                iH = [hH] + iH   # appendleft newly aggregated H
                hHt += [hH]  # returns 2 higher agg forks?
                iEt[:] = np.add(iEt, hEt)
        LEt = np.add(LEt, hEt)

    if sum(LEt[:1]) > sum(G_aves) * sum(LEt[2:]):
        # else aggH is not extended
        return [[mLay,dLay,LL_,LEt]] + aggH

def rng_node_(root,_N_):  # each rng+ forms rim_ layer per N, then nG_,lG_,Et:

    rngH, L__ = [],[]; HEt = [0,0,0,0]
    rng = 1
    while True:
        N_,Et = rng_kern_(_N_,rng)  # adds a layer of links to _Ns with pars summed in G.kHH[-1]
        nG_ = segment_N_(root, N_,0,rng)  # cluster N_ by link M
        L_ = [link for G in nG_ for link in G.link_]; L__ += L_
        if Et[1] > ave * Et[3] * rng:
            # eval in agg++: incremental nesting?
            # der+ conditional on D (no sum/N: direction is lost and D is redundant to M), recursive/ lrLay-> higher link orders?
            lrH, lL_,lEt = rng_link_(root,L_)  # cluster L_ by llink Md
            L_ = lrH  # or lG_: no nesting, else nested rng_link_, form lllinks?
            Et = np.add(Et,lEt)
        rngH += [[nG_,L_, Et]]
        HEt = np.add(HEt, Et)
        if Et[0] > ave * Et[2] * rng:
            _N_ = N_; rng += 1
        else:
            break
    return rngH, L_, HEt

def segment_N_(root, iN_, fd, rng):  # cluster iN_(G_|L_) by weight of shared links, initially single linkage

    max_, N_ = [],[]
    for N in iN_:  # init Gt per G|L node:
        if fd:
            if isinstance(N.nodet[0],CG):
                  Nrim = [Lt[0] for n in N.nodet for Lt in n.rim_[-1] if (Lt[0] is not N and Lt[0] in iN_)]  # external nodes
            else: Nrim = [Lt[0] for Lt in N.rimt_[-1][0]+ N.rimt_[-1][1] if (Lt[0] is not N and Lt[0] in iN_)]  # nodet-mediated links, same der order as N
            Lrim = Nrim  # mediated Ls to cluster, no L links yet (maybe simplified)
        else:
            Lrim = [Lt[0] for Lt in N.rim_[-1]] if isinstance(N,CG) else [Lt[0] for Lt in N.rimt_[-1][0] + N.rimt_[-1][1]]  # external links
            Nrim = [_N for L in Lrim for _N in L.nodet if (_N is not N and _N in iN_)]  # external nodes
        Gt = [[N],[], Lrim, Nrim, [0,0,0,0]]  # node_,link_,Lrim,Nrim, Et
        N.root_+= [Gt]
        N_ += [Gt]  # select exemplar maxes to segment clustering:
        emax_ = [eN for eN in Nrim if eN.Et[fd] >= N.Et[fd] or eN in max_]  # _N if _N == N
        if not emax_: max_ += [Gt]  # N.root, if no higher-val neighbors
        # extended rrim max: V * k * max_rng?
    for Gt in N_: Gt[3] = [_N.root_[-1] for _N in Gt[3]]  # replace eNs with Gts
    # merge Gts with shared +ve links:
    for Gt in max_ if max_ else N_:
        node_,link_, Lrim, Nrim, Et = Gt
        for _Gt, _L in zip(Nrim,Lrim):  # merge connected _Gts if >ave shared external links (Lrim)
            if _Gt not in N_: continue  # was merged
            sL_ = set(Lrim).intersection(_Gt[2]).union([_L])  # shared external nodes if fd else links, + potential _N|_L
            Et = np.sum([L.Et for L in sL_], axis=0)
            if Et[fd] > ave * Et[2+fd] * rng:  # value of shared links or nodes
                if not fd: link_ += [_L]
                merge(Gt,_Gt)
                N_.remove(_Gt)
    return [sum2graph(root, Gt, fd, rng) for Gt in N_]

def feedback(root):  # called from agg_recursion

    mDerLay = CH()  # added per rng+
    while root.fback_t[0]:
        mDerLay.add_H(root.fback_t[0].pop())
    dDerH = CH()  # from higher-order links
    while root.fback_t[1]:
        dDerH.add_H(root.fback_t[1].pop())
    DderH = mDerLay.append_(dDerH, flat=1)
    m,d, mr,dr = DderH.Et
    if m+d > sum(G_aves) * (mr+dr):
        root.derH.append_(DderH, flat=1)  # multiple lays appended upward
    # recursion after agg++ in all nodes of both forks:
    if root.roott:  # not Edge
        for fd, rroot in zip((0,1), root.roott):  # root may belong to graphs of both forks
            if rroot:  # empty if root is not in this fork
                rroot.fback_t[fd] += [DderH]
                if all(len(f_) == len(rroot.node_) for f_ in rroot.fback_t):  # both forks of sub+ end for all nodes
                    feedback(rroot)  # sum2graph adds higher aggH, feedback adds deeper aggH layers
