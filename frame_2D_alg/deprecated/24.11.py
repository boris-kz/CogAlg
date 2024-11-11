def agg_recursion(root, iLay, iQ, fd):  # parse the deepest Lay of root derH, breadth-first cross-comp, clustering, recursion

    Q = []
    for e in iQ:
        if isinstance(e, list): continue  # skip Gts: weak
        e.root_, e.extH, e.merged = [], CH(), 0
        Q += [e]
    # cross-comp link_ or node_:
    N_, L_, Et, rng = comp_link_(Q) if fd else comp_node_(Q)
    m, d, mr, dr = Et
    fvd = d > ave_d * dr*(rng+1)
    fvm = m > ave * mr * (rng+1)
    nest = 0
    if fvd or fvm:
        # extend root derH:
        if not fd:  # init derH with iLay, formed in prior comp_node_
            Lay0 = CH(root=iLay).copy(iLay); iLay.nest += 1; Lay0.deep = iLay.nest
            for he in Lay0.H: he.deep += 1
            iLay.H = [Lay0]
            nest = 1
        Lay = CH().add_H([L.derH for L in L_])  # top Lay, nest-> rngH in cluster_N_, derH in der+
        Lay.deep=iLay.nest; for he in Lay.H: he.deep += 1
        iLay.append_(Lay)  # extend derH, formed above or in prior comp_link_
        # recursion:
        if fvd and len(L_) > ave_L:
            nestd = agg_recursion(root, iLay, L_,fd=1)  # der+ comp_link_
        if fvm:
            pL_ = {l for n in N_ for l,_ in get_rim(n,fd)}
            G_ = cluster_N_(root, pL_, fd)  # optionally divisive clustering
            if len(G_) > ave_L:
                nestr = agg_recursion(root, Lay, G_,fd=0)  # rng+ comp clustered node_, no passing iLay?
                Lay.nest += nestr
        nest += max((nestd if nestd in locals() else 0), (nestr if nestr in locals() else 0))
        iLay.nest += nest

    return nest

def add_H(HE, He_, irdnt=[]):  # unpack derHs down to numericals and sum them

        if not isinstance(He_,list): He_ = [He_]
        for He in He_:
            if HE:
                for i, (Lay,lay) in enumerate(zip_longest(HE.H, He.H, fillvalue=None)):  # cross comp layer
                    if lay:
                        if Lay: Lay.add_H(lay, irdnt)
                        else:
                            if Lay is None:
                                HE.append_(CH().copy(lay))  # pack a copy of new lay in HE.H
                            else:
                                HE.H[i] = CH(root=HE).copy(lay)  # Lay was []
                HE.accum_lay(He, irdnt)
                HE.node_ += [node for node in He.node_ if node not in HE.node_]  # node_ is empty in CL derH?
            else:
                HE.copy(He)  # init
            HE.update_root(He)  # feedback, ideally buffered from all elements before summing in root, ultimately G|L

            depth = 1  # feedback depth relative to the root
            while True:
                root = HE.root
                if not isinstance(root, CH): root = root.derH  # root is G|L, add Et?
                if not root: break
                while depth > len(root.H): root.H += [CH(root=HE).copy()]
                root.H[-depth].add_H(He)  # merge in lay
                He = HE;
                depth += 1
                HE = root
            # | feedback, batch all same-layer nodes per root? not relevant in agg++?
            root = HE.root
            if root is not None:  # not frame
                if not isinstance(root, CH): root = root.derH  # root is G|L
                while depth > len(root.H): root.H += [CH(root=HE)]
                root.H[-depth].add_H(He, depth+1)  # merge both forks in same root lay

        return HE

def append_(HE,He, irdnt=None, flat=0):

        if irdnt is None: irdnt = []
        if flat:
            for i, lay in enumerate(He.H):  # different refs for L.derH and root.derH.H:
                lay = CH().copy(lay)
                lay.i = len(HE.H)+i; lay.root = HE; HE.H += [lay]
        else:
            He.i = len(HE.H); He.root = HE; HE.H += [He]
        HE.accum_lay(He, irdnt)
        return HE.update_root(He)

def update_root(HE, He):

        root = HE.root
        while root is not None:
            if not isinstance(root, CH):
                root = root.derH  # root is G|L, add Et?
            if not root: break
            while He.depth > len(root.H): root.H += [CH()]
            root.H[He.depth].add_H(He)
            root.Et += He.Et
            root.node_ += [node for node in He.node_ if node not in HE.node_]
            root.n += He.n
            root = root.root
        return HE

def agg_recursion(root, iQ, fd):  # parse the deepest Lay of root derH, breadth-first cross-comp, clustering, recursion

    Q = []
    for e in iQ:
        if isinstance(e, list): continue  # skip Gts: weak
        e.root_, e.extH, e.merged = [], CH(), 0
        Q += [e]
    # cross-comp link_ or node_:
    N_, L_, Et, rng = comp_link_(Q) if fd else comp_node_(Q)
    m, d, mr, dr = Et
    fvd = d > ave_d * dr*(rng+1)
    fvm = m > ave * mr * (rng+1)
    if fvd or fvm:
        Lay = CH().add_H([L.derH for L in L_])  # top Lay, nest-> rngH/cluster_N_, derH/der+
        if fd:
            derH = root.derH  # comp_link_, nest single-lay derH formed in prior comp_node_:
            derH.H = [CH(root=derH).copy(derH)]
            derH.append_(Lay)
        else: root.derH = Lay
        # comp_node_ and comp_link_ per node_, deeper comps are mediated by clustering, which forms new node_
        if not fd and fvd and len(L_) > ave_L:
            agg_recursion(root, L_,fd=1)  # der+ comp_link_
        if fvm:
            pL_ = {l for n in N_ for l,_ in get_rim(n,fd)}
            G_ = cluster_N_(root, pL_, fd)  # optionally divisive clustering
            if len(G_) > ave_L:
                agg_recursion(root, G_,fd=0)  # rng+ comp clustered node_
    # or:
    N_,L_,lay, fvm,fvd = comp_Q(root.node_, fd=0)
    He = root.derH; l = len(He.H); layt = []  # nesting incr/ derivation: composition of compared Gs,longer derH from comp_link_?
    if fvm:
        layt += [lay]; He.accum_lay(lay); lay.i=l; lay.root=He
        cluster_eval(root, N_, fd=0)
    if fvd:
        dN_,dL_,dlay,_,_ = comp_Q(L_,fd=1)  # comp new L_, root.link_ was compared in root-forming for alt clustering
        layt += [dlay]; He.accum_lay(dlay); dlay.i=l; dlay.root=He
        cluster_eval(root, dN_, fd=1)
    if layt:
        root.derH.H += [layt]

    def copy(_He, He):
        for attr, value in He.__dict__.items():
            if attr != '_id' and attr != 'root' and attr in _He.__dict__.keys():  # copy attributes, skip id, root
                if attr == 'H':
                    if He.H:
                        _He.H = []
                        if isinstance(He.H[0], CH):
                            for lay in He.H:
                                if isinstance(lay, list): _He.H += [[]]              # empty list layer
                                else:                     _He.H += [CH().copy(lay)]  # can't deepcopy CH.root
                        else: _He.H = deepcopy(He.H)  # md_
                elif attr == "node_":
                    _He.node_ = copy(He.node_)
                else:
                    setattr(_He, attr, deepcopy(value))
        return _He

def cluster_N_(root, _L_, fd, nest=1):  # top-down segment L_ by >ave ratio of L.dists
    ave_rL = 1.2  # define segment and sub-cluster

    _L_ = sorted(_L_, key=lambda x: x.dist, reverse=True)
    _L = _L_[0]; L_ = [_L], et = _L.derH.Et
    L__ = []
    # segment _L_ by contiguous dist, long links first:
    for i, L in enumerate(_L_[1:]):
        rel_dist = _L.dist / L.dist  # >1
        if rel_dist < ave_rL or et[0] < ave or len(_L_[i:]) < ave_L:  # ~=dist Ns or either side of L is weak
            L_ += [L]; et += L.derH.Et
        else:  # terminate dist segment
            L__ += [L_]; L_=[L]; et = L.derH.Et
        _L = L
    L__ += [L_]  # last segment
    G__ = []
    for L_ in L__:  # cluster Ns via Ls in dist segment
        Gt_ = []
        for L in L_:
            node_, link_, et = [n for n in L.nodet if not n.merged], {L}, copy(L.derH.Et)  # init Gt
            _eL_ = {eL for N in L.nodet for eL,_ in get_rim(N,fd) if eL in L_}  # init current-dist ext Ls
            while _eL_:
                eL_ = set()  # extended rim Ls
                for _eL in _eL_:  # cluster node-connected ext Ls
                    et += _eL.derH.Et
                    node_ += [n for n in _eL.nodet if not n.merged and n not in node_]
                    link_.add(_eL)
                    for eL in {eL for N in _eL.nodet for eL,_ in get_rim(N, fd)}:
                        if eL not in link_ and eL in L_: eL_.add(eL)
                _eL_ = eL_
            Gt = [node_, link_, et]  # maxL is per segment, not G
            for n in node_: n.merged = 1
            Gt_ += [Gt]
        G_ = []
        for Gt in Gt_:
            for n in Gt[0]: n.merged = 0  # reset per segment
            M, R = Gt[2][0::2]  # Gt: node_,link_,et,max_dist
            if M > R * ave * nest:  # rdn+1 in lower segments
                G_ += [sum2graph(root, Gt, fd, nest)]
                # add root_G node_ += G, G.root += root_G?
        nest += 1
        G__ += G_

    return G__[0]  # top Gs only, lower Gs should be nested in their node_
