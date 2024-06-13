def rng_trace_rim(N_, Et):  # comp Clinks: der+'rng+ in root.link_ rim_t node rims: directional and node-mediated link tracing

    _L_ = N_
    rng = 1
    while _L_:
        L_ = []
        for L in _L_:
            for dir, rim_ in zip((0,1), L.rim_t):  # Link rng layers
                for mL in rim_[-1]:
                    for N in mL.nodet:  # N in L.nodet, _N in _L.nodet
                        if N.root is not L.nodet:  # N.root = nodet
                            _L = N.root[-1]     # nodet[-1] = L
                            if _L is L or _L in L.compared_:
                                continue
                            if not hasattr(_L,"rimt"):
                                add_der_attrs( link_=[_L])  # _L is outside root.link_, still same derivation
                            L.compared_ += [_L]
                            _L.compared_ += [L]
                            Link = Clink(nodet =[_L,L], box=extend_box(_L.box, L.box))
                            comp_N(Link, L_, Et, rng, dir) # L_+=nodet, L.rim_t+=Link
                            break  # only one N mediates _L
    # or
            for dir, rim_ in zip((0,1), L.rim_t):  # Link rng layers
                for mL in rim_[-1]: # prior _L
                    # mL has Links in rim_t 1st layer:
                    for link in mL.rim_t[0][0] + mL.rim_t[0][1]:
                        for _L in link.nodet:
                            if _L is L or _L in L.compared_:  # if _L is mL
                                continue
                            if not hasattr(_L,"rim_t"):
                                add_der_attrs( link_=[_L])  # _L is outside root.link_, still same derivation
                            L.compared_ += [_L]
                            _L.compared_ += [L]
                            Link = Clink(nodet =[_L,L], box=extend_box(_L.box, L.box))
                            comp_N(Link, L_, Et, rng, dir) # L_+=nodet, L.rim_t+=Link
                            break  # only one N mediates _L
        _L_=L_; rng+=1
    return N_, rng, Et

def rng_trace_rim(N_, Et):  # comp Clinks: der+'rng+ in root.link_ rim_t node rims: directional and node-mediated link tracing

    _L_ = N_
    rng = 1
    while _L_:
        L_ = []
        for L in _L_:
            for nodet in L.nodet[-1]:  # variable nesting _L-mediating nodes, init Gs
                for dir, N_ in zip((0,1),nodet):  # Link direction
                    if not isinstance(N_,list): N_ = [N_]
                    for N in N_:
                        rim = N.rim if isinstance(N,CG) else N.rimt_[-1][0]+N.rimt_[-1][1]  # concat dirs
                        for _L in rim:
                            if _L is L or _L in L.compared_: continue
                            if not hasattr(_L,"rimt_"):
                                add_der_attrs( link_=[_L])  # _L is outside root.link_, still same derivation
                            L.compared_ += [_L]; _L.compared_ += [L]
                            # not needed?:
                            if rng > 1:  # draft
                                coSpan = L.span * np.cos( np.arctan2(*np.subtract(L.angle, (dy,dx))))
                                _coSpan = _L.span * np.cos( np.arctan2(*np.subtract(_L.angle, (dy,dx))))
                                if dist / ((coSpan +_coSpan) / 2) > max_dist * rng:  # Ls are too distant
                                    continue
                            Link = Clink(nodet =[_L,L], box=extend_box(_L.box, L.box))
                            comp_N(Link, L_, Et, rng, dir)  # L_+=nodet, L.rim_t+=Link
        _L_=L_; rng+=1
    return N_, rng, Et

def rng_node_(N_, Et, rng=1):  # comp Gs|kernels in agg+, links | link rim_t node rims in sub+
                               # similar to graph convolutional network but without backprop
    _G_ = N_
    G_ = []  # eval comp_N -> G_:
    for link in list(combinations(_G_,r=2)):
        _G, G = link
        if _G in G.compared_: continue
        dy,dx = np.subtract(_G.yx,G.yx)
        dist = np.hypot(dy,dx)
        aRad = (G.aRad+_G.aRad) /2  # ave radius to eval relative distance between G centers:
        if dist / max(aRad,1) <= max_dist * rng:
            G.compared_ += [_G]; _G.compared_ += [G]
            Link = Clink(nodet=[_G,G], span=dist, angle=[dy,dx], box=extend_box(G.box,_G.box))
            if comp_N(Link, Et):
                G_ += [_G,G]
    _G_ = list(set(G_))
    for G in _G_:  # init kernel as [krim]
        krim =[]; DerH =CH() # layer/krim
        for link in G.rim:
            DerH.add_(link.derH)
            krim += [link.nodet[0] if link.nodet[1] is G else link.nodet[1]]
        if DerH: G.derH.append_(DerH, flat=0)
        G.kH = [krim]
    rng = 1  # aggregate rng+: recursive center node DerH += linked node derHs for next-loop cross-comp
    while len(_G_) > 2:
        G_ = []
        for G in _G_:
            if len(G.rim) < 2: continue  # one link is always overlapped
            for link in G.rim:
                if link.derH.Et[0] > ave:  # link.Et+ per rng
                    comp_krim(link, G_, rng)  # + kernel rim / loop, sum in G.extH, derivatives in link.extH?
        _G_ = list(set(G_))
        rng += 1
    for G in _G_:
        for i, link in enumerate(G.rim):
            G.extH.add_(link.DerH) if i else G.extH.append_(link.DerH, flat=1)  # for segmentation

    return N_, rng, Et
'''
    _Gt_ = []
    for G in G_:  # def kernel rim per G:
        _Gi_,M = [],0
        for link in G.rim:
            _G = [link.nodet[0] if link.nodet[1] is G else link.nodet[1]]
            _Gi_ += [G_.index(_G)]
            M += link.derH.Et[0]
        _Gt_ += [[G,_Gi_,M,[]]]  # _Gi_: krim indices, []: local compared_
    rng = 1
    while _Gt_:  # aggregate rng+ cross-comp: recursive center node DerH += linked node derHs for next loop
        Gt_ = []
        for i, [G,_Gi_,_M,_compared_] in enumerate(_Gt_):
            Gi_, M, compared_ = [],0,[]
            for _i in _Gi_:  # krim indices
                if _i in _compared_: continue
                _G,__Gi_,__M,__compared_ = _Gt_[_i]
                compared_ += [i]; __compared_ += [_i]  # bilateral assign
                dderH = _G.derH.comp_(G.derH)
                m = dderH.Et[0]
                if m > ave * dderH.Et[2]:
                    M += m; __M += m
                    Gi_ += [_i]; __Gi_+=[i]
                    for g in _G,G:
                        g.DerH.H[-1].add_(dderH, irdnt=dderH.H[-1].Et[2:]) if len(g.DerH.H)==rng else g.DerH.append_(dderH,flat=0)
            if M-_M > ave:
                Gt_ += [[G, Gi_, M, compared_]]
'''
def comp_G(_G, G):  # comp without forming links

    dderH = CH()
    _n,_L,_S,_A,_derH,_extH,_iderH,_latuple = _G.n,len(_G.node_),_G.S,_G.A,_G.derH,_G.extH,_G.iderH,_G.latuple
    n, L, S, A, derH, extH, iderH, latuple = G.n,len(G.node_), G.S, G.A, G.derH, G.extH, G.iderH, G.latuple
    rn = _n/n
    et, rt, md_ = comp_ext(_L,L, _S,S/rn, _A,A)
    Et, Rt, Md_ = comp_latuple(_latuple, latuple, rn, fagg=1)
    dderH.n = 1; dderH.Et = np.add(Et,et); dderH.relt = np.add(Rt,rt)
    dderH.H = [CH(Et=et,relt=rt,H=md_,n=.5),CH(Et=Et,relt=Rt,H=Md_,n=1)]
    _iderH.comp_(iderH, dderH, rn, fagg=1, flat=0)

    if _derH and derH: _derH.comp_(derH, dderH, rn, fagg=1, flat=0)  # append and sum new dderH to base dderH
    if _extH and extH: _extH.comp_(extH, dderH, rn, fagg=1, flat=1)

    return dderH

'''
G.DerH sums krim _G.derHs, not from links, so it's empty in the first loop.
_G.derHs can't be empty in comp_krim: init in loop link.derHs
link.DerH is ders from comp G.DerH in comp_krim
G.extH sums link.DerHs: '''

def comp_krim(link, G_, nrng, fd=0):  # sum rim _G.derHs, compare to form link.DerH layer

    _G,G = link.nodet  # same direction
    ave = G_aves[fd]
    for node in _G, G:
        if node in G_: continue  # new krim is already added
        krim = []  # kernel rim
        for _node in node.kH[-1]:
            for _link in _node.rim:
                __node = _link.nodet[0] if _link.nodet[1] is _node else _link.nodet[1]
                krim += [_G for _G in __node.kH[-1] if _G not in krim]
                if node.DerH: node.DerH.add_(__node.derH, irdnt=_node.Et[2:])
                else:         node.DerH = deepcopy(__node.derH)  # init
        node.kH += [krim]
    _skrim = set(_G.kH[-1]); skrim = set(G.kH[-1])
    _xrim = list(_skrim - skrim)
    xrim = list(skrim - _skrim)  # exclusive kernel rims
    if _xrim and xrim:
        dderH = comp_N_(_xrim, xrim)
        if dderH.Et[0] > ave * dderH.Et[2]:
            G_ += [_G,G]  # update node_, use nested link.derH vs DerH?
            link.DerH.H[-1].add_(dderH, irdnt=dderH.H[-1].Et[2:]) if len(link.DerH.H)==nrng else link.DerH.append_(dderH,flat=1)

    # connectivity eval in segment_graph via decay = (link.relt[fd] / (link.derH.n * 6)) ** nrng  # normalized decay at current mediation
