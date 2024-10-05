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

def cluster_N__(root, N__,L__, fd):  # cluster G__|L__ by value density of +ve links per node

    for rng, N_ in enumerate(N__):  # init Gt for each N, may be unpacked and present multiple layers
        for N in N_:
            if N.root_: continue  # Gt was initialized in lower N__[i]
            Gt = [{N}, set(), np.array([.0,.0,.0,.0]), 0]
            N.root_ = [Gt]  # 1st element is rng of the lowest root?
    # cluster from L_:
    Gt__ = []
    for rng, L_ in enumerate(L__, start=1):  # all Ls and current-rng Gts are unique
        Gt_ = []
        if len(L_) < ave_L:
            continue
        for L in L_:
            for N in L.nodet:
                _node_, _link_, _Et, mrg = N.root_[-1]  # <= rng in all L-connected Gs
                if mrg: continue  # 1 in current-rng overlap, 0 in only one G of nodet
                Node_, Link_, Et = _node_[:], _link_[:], _Et[:]  # current-rng Gt init
                for G in _node_:
                    if not G.merged and len(G.nrim_) > rng:
                        node_ = G.nrim_[rng] - Node_
                        if not node_: continue  # no new rim nodes
                        node_, link_, et = cluster_from_G(G, node_, G.lrim_[rng] - Link_, rng)
                        Node_.update(node_)
                        Link_.update(link_)
                        Et += et
                if Et[0] > Et[2] * ave:  # additive current-layer V: form higher Gt
                    Node_.update(_node_)
                    Link_.update(_link_)
                    Gt = [Node_, Link_, Et + _Et, 0]
                    for n in Node_:
                        n.root_.append(Gt)
                    N.root_ += [Gt]
                    L.root = [rng, Gt]  # rng-specific
                    Gt_ += [Gt]
        # not sure now:
        for G in set.union(*N__[:rng+1]):  # in all lower Gs
            G.merged = 0
        Gt__ += [Gt_]
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



