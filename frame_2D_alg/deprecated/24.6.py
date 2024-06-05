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

