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
