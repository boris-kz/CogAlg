def rng_link_(N_):  # comp Clinks: der+'rng+ in root.link_ rim_t node rims: directional and node-mediated link tracing

    _mN_t_ = [[[N.nodet[0]],[N.nodet[1]]] for N in N_]  # rim-mediating nodes
    rng = 1; L_ = N_[:]
    Et = [0,0,0,0]
    while True:
        mN_t_ = [[[],[]] for _ in L_]
        for L, _mN_t, mN_t in zip(L_, _mN_t_, mN_t_):
            for rev, _mN_, mN_ in zip((0,1), _mN_t, mN_t):
                # comp L, _Ls: nodet mN 1st rim, -> rng+ _Ls/ rng+ mm..Ns:
                rim_ = [n.rim if isinstance(n,CG) else n.rimt_[0][0] + n.rimt_[0][1] for n in _mN_]
                for rim in rim_:
                    for _L,_rev in rim:  # _L is reversed relative to its 2nd node
                        if _L is L or _L in L.compared_: continue
                        if not hasattr(_L,"rimt_"): add_der_attrs(link_=[_L])  # _L not in root.link_, same derivation
                        L.compared_ += [_L]; _L.compared_ += [L]
                        dy,dx = np.subtract(_L.yx, L.yx)
                        Link = Clink(nodet=[_L,L], span=2, angle=[dy,dx], box=extend_box(_L.box, L.box))
                        # L.rim_t += new Link
                        if comp_N(Link, Et, rng, rev^_rev):  # negate ds if only one L is reversed
                            # add rng+ mediating nodes to L, link order: nodet < L < rim_t, mN.rim || L
                            mN_ += _L.nodet  # get _Ls in mN.rim
                            if _L not in L_:  # not in root
                                L_ += [_L]; mN_t_ += [[[],[]]]
                            mN_t_[L_.index(_L)][1-rev] += L.nodet
        _L_, _mN_t = [],[]
        for L, mN_t in zip(L_, mN_t_):
            if any(mN_t):
                _L_ += [L]; _mN_t_ += [mN_t]
        if _L_:
            L_ = _L_; rng += 1
        else:
            break
        # Lt_ = [(L, mN_t) for L, mN_t in zip(L_, mN_t_) if any(mN_t)]
        # if Lt_: L_,_mN_t_ = map(list, zip(*Lt_))  # map list to convert tuple from zip(*)

    return N_, Et, rng
