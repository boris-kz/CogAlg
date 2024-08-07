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

def sum_N_(N_, fd=0):  # sum partial grapht in merge

    N = N_[0]
    n = N.n; S = N.S
    L, A = (N.span, N.angle) if fd else (len(N.node_), N.A)
    if not fd:
        latuple = deepcopy(N.latuple)
        mdLay = CH().copy(N.mdLay)
        extH = CH().copy(N.extH)
    derH = CH().copy(N.derH)
    # Et = copy(N.Et)
    for N in N_[1:]:
        if not fd:
            add_lat(latuple, N.latuple)
            if N.mdLay: mdLay.add_md_(N.mdLay)
        n += N.n; S += N.S
        L += N.span if fd else len(N.node_)
        A = [Angle+angle for Angle,angle in zip(A, N.angle if fd else N.A)]
        if N.derH: derH.add_H(N.derH)
        if N.extH: extH.add_H(N.extH)

    if fd: return n, L, S, A, derH, extH
    else:  return n, L, S, A, derH, extH, latuple, mdLay  # no comp Et

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

'''
    # draft recursive feedback, propagated when sub+ ends in all nodes of both forks:
    if root.roott:  # not Edge
        for fd, rroot in zip((0,1), root.roott):  # root may belong to graphs of both forks
            if rroot:  # empty if root is not in this fork
                rroot.fback_t[fd] += [DderH]
                if all(len(f_) == len(rroot.node_) for f_ in rroot.fback_t):  # both forks of sub+ end for all nodes
                    feedback(rroot)  # sum2graph adds higher aggH, feedback adds deeper aggH layers

'''

def segment_N_(root, iN_, fd, rng):  # iN_: Link_ if fd else G_, ~ parallelized https://en.wikipedia.org/wiki/Watershed_(image_processing)
    '''
    cluster by weight of shared links, initially single linkage, + similarity of partial clusters in merge
    '''
    N_ = []  # init Gts per node in iN_, merge if Lrim overlap + similarity of exclusive node_s
    max_ = []
    for N in iN_:  # init graphts:
        if isinstance(N,CG):            rim = [Lt[0] for rim in N.rim_ for Lt in rim]
        elif isinstance(N.nodet[0],CG): rim = [Lt[0] for _N in N.nodet for rim in _N.rim_ for Lt in rim]
        else:                           rim = [Lt[0] for rimt in N.rimt_ for rim in rimt for Lt in rim]
        # get [ext_N_,int_N_], no extH if not in iN_:
        _N_t = [[_N for L in rim for _N in L.nodet if _N is not N and _N in iN_], [N]]
        Gt = [[N],[],copy(rim),_N_t,[0,0,0,0]]  # [node_, link_, Lrim, Nrim_t, Et]
        N.root = Gt; N_ += [Gt]
        emax_ = [] # if exemplar G|Ls:
        for eN in _N_t[0]:
            eEt,Et = (eN.derH.Et, N.derH.Et) if fd else (eN.extH.Et, N.extH.Et)
            if eEt[fd] >= Et[fd] or eN in max_: emax_ += [eN]  # _N if _N == N
        if not emax_: max_ += [N]  # no higher-val neighbors
        # add rrim mediation: V * k * max_rng?
    max_ = [N.root for N in max_]
    # replace extNs with their Gts:
    for Gt in N_: Gt[3][0] = [_N.root for _N in Gt[3][0]]
    # merge with connected _Gts:
    for Gt in max_ if max_ else N_:
        node_, link_, Lrim, Nrim_t, Et = Gt
        Nrim = Nrim_t[0]
        for _Gt, _L in zip(Nrim, Lrim):
            if _Gt not in N_:
                continue  # was merged
            oL_ = set(Lrim).intersection(set(_Gt[2])).union([_L])  # shared external links + potential _L # oL_ = [Lr[0] for Lr in _Gt[2] if Lr in Lrim]
            oV = sum([L.derH.Et[fd] - ave * L.derH.Et[2+fd] for L in oL_])
            # eval by Nrim similarity = oV + olp between G,_G,
            # ?pre-eval by max olp: _node_ = _Gt[0]; _Nrim = _Gt[3][0],
            # if len(Nrim)/len(node_) > ave_L or len(_Nrim)/len(_node_) > ave_L:
            sN_ = set(node_); _sN_ = set(_Gt[0])
            oN_ = sN_.intersection(_sN_)  # Nrim overlap
            xN_ = list(sN_- oN_)  # exclusive node_
            _xN_ = list(_sN_- oN_)
            if _xN_ and xN_:
                dderH = comp_G( sum_N_(_xN_), sum_N_(xN_))
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

# not fully updated
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
