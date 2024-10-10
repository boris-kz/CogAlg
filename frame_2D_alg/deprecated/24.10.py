def cluster_N__(root, N__, L__, fd):  # cluster G__|L__ by value density of +ve links per node

    Gt__ = []  # init Gt for each N, may be unpacked and present multiple layers:
    for rng, N_,L_ in enumerate( zip(N__,L__)):  # trace external Ls
        Gt_ = []
        for N in N_:
            if N.root_: continue  # Gt was initialized in lower N__[i]
            Gt = [{N}, set(), np.array([.0,.0,.0,.0]), 0]
            N.root_ = [rng, Gt]  # 1st element is rng of the lowest root?
            Gt_ += [Gt]
        Gt__ += [Gt_]
    # cluster rngLay:
    for rng, Gt_ in enumerate(Gt__, start=1):
        if len(Gt_) < ave_L:
            continue
        for G in set.union(*N__[:rng+1]):  # in all lower Gs
            G.merged = 0
        for _node_,_link_,_Et, mrg in Gt_:
            if mrg: continue
            Node_, Link_, Et = set(),set(), np.array([.0,.0,.0,.0])  # m,r only?
            for G in _node_:
                if not G.merged and len(G.nrim_) > rng:
                    node_ = G.nrim_[rng]- Node_
                    if not node_: continue  # no new rim nodes
                    node_,link_,et = cluster_from_G(G, node_, G.lrim_[rng]-Link_, rng)
                    Node_.update(node_)
                    Link_.update(link_)
                    Et += et
            if Et[0] > Et[2] * ave:  # additive current-layer V: form higher Gt
                Node_.update(_node_); Link_.update(_link_)
                Gt = [Node_, Link_, Et+_Et, 0]
                for n in Node_:
                    n.root_.append(Gt)
    n__ = []
    for rng, Gt_ in enumerate(Gt__, start=1):  # selective convert Gts to CGs
        n_ = []
        for Gt in Gt_:
            if Gt[2][0] > Gt[2][2] * ave:  # eval additive Et
                n_ += [sum2graph(root, [list(Gt[0]), list(Gt[1]), Gt[2]], fd, rng)]
            else:
                for n in Gt[0]:  # eval weak Gt node_
                    if n.ET[0] > n.Et[2] * ave * rng:  # eval with added rng
                        n.lrim_, n.nrim_ = [],[]
                        n_ += [n]
        n__ += [n_]
    N__[:] = n__

def cluster_from_G(G, _nrim, _lrim, rng):

    node_, link_, Et = {G}, set(), np.array([.0,.0,.0,.0])  # m,r only?
    while _lrim:
        nrim, lrim = set(), set()
        for _G,_L in zip(_nrim, _lrim):
            if _G.merged or not _G.root_ or len(_G.lrim_) <= rng:
                continue  # root_ is empty if _G not in N__
            for g in node_:  # compare external _G to all internal nodes, add if any match
                L = next(iter(g.lrim_[rng] & _G.lrim_[rng]), None)  # intersect = [+link] | None
                if L:
                    if ((g.extH.Et[0]-ave*g.extH.Et[2]) + (_G.extH.Et[0]-ave*_G.extH.Et[2])) * (L.derH.Et[0]/ave) > ave * ccoef:
                        # merge roots,
                        # else: node_.add(_G); link_.add(_L); Et += _L.derH.Et
                        _node_,_link_,_Et,_merged = _G.root_[-1]
                        if _merged: continue
                        node_.update(_node_)
                        link_.update(_link_| {_L})  # add external L
                        Et += _L.derH.Et + _Et
                        for n in _node_: n.merged = 1
                        _G.root_[-1][3] = 1
                        nrim.update(set(_G.nrim_[rng]) - node_)
                        lrim.update(set(_G.lrim_[rng]) - link_)
                        _G.merged = 1
                        break
        _nrim,_lrim = nrim, lrim
    return node_, link_, Et

def cluster_N__1(root, N__,L__, fd):  # cluster G__|L__ by value density of +ve links per node

    Gt__ = []
    for rng, (N_,L_) in enumerate(zip(N__,L__), start=1):  # all Ls and current-rng Gts are unique
        Gt_ = []
        if len(L_) < ave_L: continue
        for N in N_:
            N.merged = 0
            if not N.root_:  # always init root graph for generic merging process
                Gt = [{N}, set(), np.array([.0,.0,.0,.0])]; N.root_ = [Gt]
        # cluster from L_:
        for L in L_:
            for G in L.nodet:
                if G.merged: continue
                node_, link_, et = G.root_[-1]  # lower-rng graph, mrg = 0
                Node_, Link_, Et = node_.copy(), link_.copy(), et.copy()  # init current-rng Gt
                # extend Node_:
                for g in node_:
                    _lrim = get_rim(g, Link_, fd, rng)
                    while _lrim:
                        lrim = set()
                        for _L in _lrim:
                            _G = _L.nodet[1] if _L.nodet[0] is g else _L.nodet[0]
                            if _G.merged or _G not in N_ or _G is G: continue
                            _node_,_link_,_Et = _G.root_[-1]  # lower-rng _graph
                            cV = 0  # intersect V
                            xlrim = set()  # add to lrim
                            for _g in _node_:  # no node_ overlap
                                __lrim = get_rim(_g, [], fd, rng)
                                clrim = _lrim & __lrim  # rim intersect
                                xlrim.update(__lrim - clrim)  # new rim
                                for __L in clrim:  # eval common rng Ls
                                    v = ((g.extH.Et[0]-ave*g.extH.Et[2]) + (_g.extH.Et[0]-ave*_G.extH.Et[2])) * (__L.derH.Et[0]/ave)
                                    if v > 0: cV += v
                            if cV > ave * ccoef:  # additional eval to merge roots:
                                lrim.update(xlrim)  # add new rim links
                                Node_.update(_node_)
                                Link_.update(_link_|{_L})  # add external L
                                Et += _L.derH.Et + _Et
                                for n in _node_: n.merged = 1
                        _lrim = lrim
                if Et[0] > Et[2] * ave:  # additive current-layer V: form higher Gt
                    Gt = [Node_, Link_, Et + _Et]
                    for n in Node_: n.root_+= [Gt]
                    L.root_ = Gt  # rng-specific
                    Gt_ += [Gt]
        for G in set.union( *N__[:rng]): G.merged = 0  # in all lower Gs
        Gt__ += [Gt_]
    n__ = []
    for rng, Gt_ in enumerate(Gt__, start=1):  # selective convert Gts to CGs
        n_ = []
        for Gt in Gt_:
            if Gt[2][0] > Gt[2][2] * ave:  # eval additive Et
                n_ += [sum2graph(root, [list(Gt[0]), list(Gt[1]), Gt[2]], fd, rng)]
            else:
                for n in Gt[0]:  # unpack weak Gt
                    if n.ET[0] > n.Et[2] * ave * rng: n_ += [n]  # eval / added rng
        n__ += [n_]
    N__[:] = n__  # replace some Ns with Gts

def rng_link_(iL_):  # comp CLs via directional node-mediated link tracing: der+'rng+ in root.link_ rim_t node rims:

    L__, LL__, pLL__, ET = [],[],[], np.array([.0,.0,.0,.0])  # all links between Ls in potentially extended L__
    fd = isinstance(iL_[0].nodet[0], CL)
    _mL_t_ = [] # init _mL_t_: [[n.rimt_[0][0]+n.rimt_[0][1] if fd else n.rim_[0] for n in iL.nodet] for iL in iL_]
    for L in iL_:
        mL_t = []
        for n in L.nodet:
            L_ = []
            for (_L,rev) in (n.rimt_[0][0]+n.rimt_[0][1] if fd else n.rim_[0]):  # all rims are inside root node_
                if _L is not L and _L.Et[0] > ave * _L.Et[2]:
                    if fd:
                        _L.rimt_,_L.root_,_L.visited_,_L.aRad,_L.merged,_L.extH = [],[],[], 0,0, CH()
                    L_ += [[_L,rev]]; L.visited_ += [L,_L]; _L.visited_ += [_L,L]
            mL_t += [L_]
        _mL_t_ += [mL_t]
    _L_ = iL_; med = 1  # rng = n intermediate nodes
    # comp _L_:
    while True:
        L_,LL_,pLL_,Et = set(),[],[], np.array([.0,.0,.0,.0])
        for L, mL_t in zip(_L_,_mL_t_):  # packed comparands
            for rev, rim in zip((0,1), mL_t):
                for _L,_rev in rim:  # reverse _L med by nodet[1]
                    rn = _L.n / L.n
                    if rn > ave_rn: continue  # scope disparity
                    Link = CL(nodet=[_L,L], S=2, A=np.subtract(_L.yx,L.yx), box=extend_box(_L.box, L.box))
                    # comp L,_L:
                    et = comp_N(Link, rn, rng=med, dir = 1 if (rev^_rev) else -1)  # d = -d if one L is reversed
                    LL_ += [Link]  # include -ves, L.rim_t += Link, order: nodet < L < rimt_, mN.rim || L
                    if et is not None:
                        L_.update({_L,L}); pLL_+=[Link]; Et += et
        L__+=[L_]; LL__+=[LL_]; pLL__+=[pLL_]; ET += Et
        # rng+ eval:
        Med = med + 1
        if Et[0] > ave * Et[2] * Med:  # project prior-loop value - new cost
            nxt_L_, mL_t_, nxt_Et = set(),[], np.array([.0,.0,.0,.0])
            for L, _mL_t in zip(_L_,_mL_t_):  # mediators
                mL_t, lEt = [set(),set()], np.array([.0,.0,.0,.0])  # __Ls per L
                for rev, rim in zip((0,1),_mL_t):
                    for _L,_rev in rim:
                        for i, n in enumerate(_L.nodet):
                            rim_ = n.rimt_ if fd else n.rim_
                            if len(rim_) == med:  # append in comp loop
                                rim = rim_[-1][0]+rim_[-1][1] if fd else rim_[-1]
                                for __L, rev in rim:
                                    if __L in L.visited_ or __L not in iL_: continue
                                    L.visited_ += [__L]; __L.visited_ += [L]
                                    et = __L.derH.Et
                                    if et[0] > ave * et[2] * Med:  # /__L
                                        mL_t[i].add((__L, 1-i))  # incrementally mediated direction L_
                                        lEt += et
                if lEt[0] > ave * lEt[2] * Med:
                    nxt_L_.add(L); mL_t_ += [mL_t]; nxt_Et += lEt  # rng+/ L is different from comp/ L above
            # refine eval:
            if nxt_Et[0] > ave * nxt_Et[2] * Med:
                _L_=nxt_L_; _mL_t_=mL_t_; med=Med
            else:
                break
        else:
            break
    return L__, LL__, pLL__, ET, med # =rng



