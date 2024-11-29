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

def cluster_(N_):  # initial flood fill via N.crim
    clust_ = []
    while N_:
        node_, peri_, M = set(), set(), 0
        N = N_.pop();
        _ref_ = [(N, 0)]
        while _ref_:
            ref_ = set()
            for _ref in _ref_:
                _eN = _ref[0]
                node_.add(_eN)  # to compute medoid in refine_
                peri_.update(_eN.perim)  # for comp to medoid in refine_
                for ref in _eN.crim:
                    eN, eM = ref
                    if eN in N_:  # not merged
                        N_.remove(eN)
                        ref_.add(ref)  # to merge
                        M += eM
            _ref_ = ref_
        clust_ += [[node_, peri_, M]]

    return clust_

def refine_(clust):
    _node_, _peri_, _M = clust

    dM = ave + 1
    while dM > ave:
        node_, peri_, M = set(), set(), 0
        _N = medoid(_node_)
        for N in _peri_:
            M = comp_cN(_N, N)
            if M > ave:
                peri_.add(N)
                if M > ave*2: node_.add(N)
        dM = M - _M
        _node_,_peri_,_M = node_,peri_,M

    clust[:] = [list(node_),list(peri_),M]  # final cluster


    N_ = frame.subG_  # should be complemented graphs: m-type core + d-type contour
    for N in N_:
        N.perim = set(); N.crim = set(); N.root_ += [frame]
    xcomp_(N_)
    clust_ = cluster_(N_)  # global connectivity clusters
    for clust in clust_:
        refine_(clust)  # re-cluster by node_ medoid
    frame.clust_ = clust_  # new param for now
    if len(frame.clust_) > ave_L:
        agg_recursion(frame)  # alternating connectivity clustering

    iN_ = []
    for N in frame.subG_:
        iN_ += [N] if N.derH.Et[0] < ave * N.derH.Et[2] else N.subG_  # eval to unpack top N

    iN_ = [N if N.derH.Et[0] < ave * N.derH.Et[2] else N.subG_ for N in frame.subG_]  # eval to unpack top N


def comp_P(_P,P, angle=None, distance=None, fder=0):  # comp dPs if fd else Ps

    fd = isinstance(P,CdP)
    _y,_x = _P.yx; y,x = P.yx
    if fd:
        # der+: comp dPs
        rn = _P.mdLay[2] / P.mdLay[2]  # mdLay.n
        derLay = comp_md_(_P.mdLay[0], P.mdLay[0], rn=rn)  # comp md_latuple: H
        angle = np.subtract([y,x],[_y,_x])  # dy,dx of node centers
        distance = np.hypot(*angle)  # between node centers
    else:
        # rng+: comp Ps
        rn = len(_P.dert_) / len(P.dert_)
        md_ = comp_latuple(_P.latuple, P.latuple, rn)
        vm = sum(md_[::2]); vd = sum(np.abs(md_[1::2]))
        rm = 1 + vd > vm; rd = 1 + vm >= vd
        n = (len(_P.dert_)+len(P.dert_)) / 2  # der value = ave compared n?
        derLay = np.array([md_, np.array([vm,vd,rm,rd]), n], dtype=object)
    # get aves:
    latuple = (_P.latuple + P.latuple) /2
    link = CdP(nodet=[_P,P], mdLay=derLay, angle=angle, span=distance, yx=[(_y+y)/2,(_x+x)/2], latuple=latuple)
    # if v > ave * r:
    Et = link.mdLay[1]
    if Et[fder] > aves[fder] * Et[fder+2]:
        P.lrim += [link]; _P.lrim += [link]
        P.prim +=[_P]; _P.prim +=[P]
        return link

def agg_cluster_(frame):  # breadth-first (node_,L_) cross-comp, clustering, recursion

    def cluster_eval(frame, N_, fd):

        pL_ = {l for n in N_ for l,_ in get_rim(n, fd)}
        if len(pL_) > ave_L:
            G_ = cluster_N_(frame, pL_, fd)  # optionally divisive clustering
            frame.subG_ = G_
            if len(G_) > ave_L:
                get_exemplar_(frame)  # may call centroid clustering
    '''
    cross-comp converted edges, then GGs, GGGs, etc, interlaced with exemplar selection: 
    D is borrowed from co-projected M, not independent.
    So Et should be [proj_m, proj_d, rdn], where
    proj_d = abs d * (m/ave): must be <m,
    proj_m = m - proj_d,
    rdn is external, common for both m and d
    '''
    N_,L_,(m,d,mr,dr) = comp_node_(frame.subG_)  # exemplars, extrapolate to their Rims?
    if m > ave * mr:
        mlay = CH().add_H([L.derH for L in L_])  # mfork, else no new layer
        frame.derH = CH(H=[mlay], md_t=deepcopy(mlay.md_t), Et=copy(mlay.Et), n=mlay.n, root=frame); mlay.root=frame.derH  # init
        vd = d * (m/ave) - ave_d * dr  # vd = proportional borrow from co-projected m
        # proj_m = m - d * (m/ave): must be <m, no generic rdn?
        if vd > 0:
            for L in L_:
                L.root_ = [frame]; L.extH = CH(); L.rimt = [[],[]]
            lN_,lL_,md = comp_link_(L_)  # comp new L_, root.link_ was compared in root-forming for alt clustering
            vd *= md / ave
            frame.derH.append_(CH().add_H([L.derH for L in lL_]))  # dfork
            # recursive der+ eval_: cost > ave_match, add by feedback if < _match?
        else:
            frame.derH.H += [[]]  # empty to decode rng+|der+, n forks per layer = 2^depth
        # + aggLays and derLays, each has exemplar selection:
        cluster_eval(frame, N_, fd=0)
        if vd > 0: cluster_eval(frame, lN_, fd=1)

    def xcomp_(N_):  # initial global cross-comp
        for g in N_: g.M, g.Mr = 0,0  # setattr

        for _G, G in combinations(N_, r=2):
            rn = _G.n/G.n
            if rn > ave_rn: continue  # scope disparity
            # use regular comp_N and compute Links, we don't need to compare again in eval_overlap?
            M,R = comp_cN(_G, G)
            vM = M - ave * R
            for _g,g in (_G,G),(G,_G):
                if vM > 0:
                    g.perim.add((_g,M))  # loose match ref (unilateral link)
                    if vM > ave * R:
                        g.Rim.add((_g,M))  # strict match ref
                        g.M += M

    def eval_overlap(N):  # check for reciprocal refs in _Ns, compare to diffs, remove the weaker ref if <ave diff

        fadd = 1
        for ref in copy(N.Rim):
            _N, _m = ref
            if _N in N.compared_: continue  # also skip in next agg+
            N.compared_ += [_N]; _N.compared_ += [N]
            for _ref in copy(_N.Rim):
                if _ref[0] is N:  # reciprocal to ref
                    dy,dx = np.subtract(_N.yx,N.yx)  # no dist eval
                    # replace Rim refs with links in xcomp_, instead of comp_N here?
                    Link = comp_N(_N,N, _N.n/N.n, angle=[dy,dx], dist=np.hypot(dy,dx))
                    minN, r = (_N,_ref) if N.M > _N.M else (N,ref)
                    if Link.derH.Et[1] < ave_d:
                        # exemplars are similar, remove min
                        minN.Rim.remove(r); minN.M -= Link.derH.pm
                        if N is minN: fadd = 0
                    else:  # exemplars are different, keep both
                        if N is minN: N.Mr += 1
                        _N.extH.add_H(Link.derH), N.extH.add_H(Link.derH)
                    break
        return fadd
