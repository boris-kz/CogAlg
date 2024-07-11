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

def comp_N(Link, iEt, rng, rev=None):  # dir if fd, Link+=dderH, comparand rim+=Link

    fd = rev is not None  # compared links have binary relative direction?
    dH = CH(); _N, N = Link.nodet; rn = _N.n / N.n

    if fd:  # Clink Ns
        _N.derH.comp_(N.derH, dH, rn, fagg=1, flat=1, frev=rev)  # dH += [*dderH]
        # reverse angle direction for left link:
        _A, A = _N.angle, N.angle if rev else [-d for d in N.angle]
        Et, rt, md_ = comp_ext(2,2, _N.S,N.S/rn, _A,A)  # 2 nodes in nodet
        dH.append_(CH(Et=Et,relt=rt,H=md_,n=0.5,root=dH),flat=0)  # dH += [dext]
    else:  # CG Ns
        et, rt, md_ = comp_latuple(_N.latuple, N.latuple, rn,fagg=1)
        dH.append_(CH(Et=et,relt=rt,H=md_,n=1,root=dH),flat=0)  # dH = [dlatuple], or also pack in derH? then same sequence as in fd?
        _N.derH.comp_(N.derH, dH,rn,fagg=1,flat=1,frev=rev)     # dH += [*dderH]
        Et, Rt, Md_ = comp_ext(len(_N.node_),len(N.node_), _N.S,N.S/rn,_N.A,N.A)
        dH.append_(CH(Et=Et,relt=Rt,H=Md_,n=0.5,root=dH),flat=0)  # dH += [dext]
    # / N, if >1 PPs | Gs:
    if _N.extH and N.extH:
        _N.extH.comp_(N.extH, dH, rn, fagg=1, flat=1, frev=rev)
    # link.derH += dH:
    if fd: Link.derH.append_(dH, flat=1)
    else:  Link.derH = dH
    iEt[:] = np.add(iEt,dH.Et)  # init eval rng+ and form_graph_t by total m|d?
    fin = 0
    for i in 0,1:
        Val, Rdn = dH.Et[i::2]
        if Val > G_aves[i] * Rdn: fin = 1
        _N.Et[i] += Val; N.Et[i] += Val  # not selective
        _N.Et[2+i] += Rdn; N.Et[2+i] += Rdn  # per fork link in both Gs
        # if select fork links: iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dH.Et[i::2])]
    if fin:
        Link.n = min(_N.n,N.n)  # comp shared layers
        Link.yx = np.add(_N.yx, N.yx) / 2
        Link.S += (_N.S + N.S) / 2
        for rev, node in zip((0,1),(_N,N)):  # ?reversed Link direction
            if fd:
                if len(node.rimt_) == rng:
                    node.rimt_[-1][1-rev] += [[Link,rev]]  # add in last rng layer, opposite to _N,N dir
                else:
                    node.rimt_ += [[[[Link,rev]],[]]] if dir else [[[],[[Link,rev]]]]  # add rng layer
            else:
                node.rim += [[Link,rev]]
                if len(node.extH.H)==rng:
                    node.extH.H[-1].H[-1].add_(Link.derH)  # accum last layer
                else:
                    rngLay = CH()
                    rngLay.append_(Link.derH, flat=0)
                    node.extH.append_(rngLay, flat=0)  # init last layer
        return True
