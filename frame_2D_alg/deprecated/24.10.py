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

