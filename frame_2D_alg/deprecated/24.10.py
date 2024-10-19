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
                                    '''
                                    Et = np.sum([lay.Et for lay in G.extH.H[:rng]]); _Et = np.sum([lay.Et for lay in _G.extH.H[:rng]])  # lower-rng surround density / node
                                    v = (((Et[0]-ave*Et[2]) + (_Et[0]-ave*_Et[2])) * (__L.derH.Et[0]/ave))  # multiply by link strength? or just eval links:
                                    '''
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

def cluster_N__2(root, N__, fd):  # cluster G__|L__ by value density of +ve links per node

    Gt__ = []
    for rng, N_ in enumerate(N__, start=1):  # all Ls and current-rng Gts are unique
        Gt_ = []   # init Gts for merging
        for N in N_: N.merged = 0
        for N in N_:
            if N.merged: continue
            if not N.root_:  # always true in 1st N_
                Gt = [[N],set(),np.array([.0,.0,.0,.0]), get_rim(N,fd,rng), 0]
                N.root_ = [Gt]
            else:
                node_, link_, et, rim, mrg = N.root_[-1]
                Link_ = list(link_); rng_rim = []
                for n in node_:
                    n.merged = 1
                    for L in get_rim(n,fd,rng):
                        if all([n in node_ for n in L.nodet]): Link_ += [L]
                        else: rng_rim += [L]  # one external node
                Gt = [[node_[:], set(Link_), np.array([.0,.0,.0,.0]), set(rng_rim), 0]]
                for n in node_: n.root_ += [Gt]  # includes N
            Gt_ += [Gt]
            if len(Gt_) < ave_L:
                Gt__ += [N_]; break  # skip clustering
        GT_ = []  # merged Gts
        for Gt in Gt_:
            node_, link_, et, rim, mrg = Gt
            if mrg: continue
            while any(rim):  # extend node_,link_, replace rim
                ext_rim = set()
                for _L in rim:
                    G,_G = _L.nodet if _L.nodet[0] in node_ else list(reversed(_L.nodet)) # one is outside node_
                    if _G.root_[-1] is Gt: continue  # merged in prior loop
                    _node_, _link_, _et, _rim, _ = _G.root_[-1]
                    crim = (rim | ext_rim) & _rim  # intersect with extended rim
                    xrim = _rim - crim   # exclusive _rim
                    cV = 0  # common val
                    for __L in crim:  # common Ls
                        v = __L.derH.Et[0] - ave * __L.derH.Et[2]
                        if v > 0: cV += __L.derH.Et[0]  # cluster by +ve links only
                    if cV / _et[0] > ave * ccoef:  # normalized by _M: behavioural independence of _G?
                        _G.root_[-1][-1] = 1  # set mrg
                        ext_rim.update(xrim)  # add new links
                        for _node in _node_:
                            if _node not in node_:
                                _node.root_[-1] = Gt; node_ += [_node]
                        link_.update(_link_|{_L}) # external L
                        et += _L.derH.Et + _et
                rim = ext_rim
            GT_ += [Gt]
        Gt__ += [GT_]
    n__ = []
    for rng, Gt_ in enumerate(Gt__, start=1):  # selective convert Gts to CGs
        if isinstance(Gt_, set): continue  # recycled N_
        n_ = []
        for node_,link_,et,_,_ in Gt_:
            if et[0] > et[2] * ave * rng:  # additive rng Et
                n_ += [sum2graph(root, [list(node_),list(link_),et], fd, rng)]
            else:  # weak Gt
                n_ += [[node_,link_,et]]  # skip in current-agg xcomp, unpack if extended lower-agg xcomp
        n__ += [n_]
    N__[:] = n__  # replace Ns with Gts, if any

def get_rim(N, fd, rng):

    rim_ = N.rimt_ if fd else N.rim_
    if len(rim_) < rng: return set()  # empty lrim
    else:
        return set([Lt[0] for Lt in (rim_[rng-1][0]+rim_[rng-1][1] if fd else rim_[rng-1])
                    if Lt[0].derH.Et[0] > ave * Lt[0].derH.Et[2] * rng])

def comp_node_(_N_):  # rng+ forms layer of rim_ and extH per N, appends N__,L__,Et, ~ graph CNN without backprop

    N__,L__,ET = [],[], np.array([.0,.0,.0,.0])  # rng H
    _Gp_ = []  # [G pair + co-positionals]
    for _G, G in combinations(_N_, r=2):
        rn = _G.n / G.n
        if rn > ave_rn: continue  # scope disparity
        radii = G.aRad + _G.aRad
        dy,dx = np.subtract(_G.yx,G.yx)
        dist = np.hypot(dy,dx)
        _Gp_ += [(_G,G, rn, dy,dx, radii, dist)]
    icoef = .5  # internal M proj_val / external M proj_val
    rng = 1  # len N__
    while True:  # prior rng vM
        Gp_,N_,L_, Et = [],set(),[], np.array([.0,.0,.0,.0])
        for Gp in _Gp_:
            _G,G, rn, dy,dx, radii, dist = Gp
            _nrim = set([(Lt[0].nodet[1] if Lt[0].nodet[0] is _G else Lt[0].nodet[0]) for rim in _G.rim_ for Lt in rim])
            nrim = set([(Lt[0].nodet[1] if Lt[0].nodet[0] is G else Lt[0].nodet[0]) for rim in G.rim_ for Lt in rim])
            if _nrim & nrim:  # not sure: skip indirectly connected Gs, no direct match priority?
                continue
            M = (_G.mdLay.Et[0]+G.mdLay.Et[0]) *icoef**2 + (_G.derH.Et[0]+G.derH.Et[0])*icoef + (_G.extH.Et[0]+G.extH.Et[0])
            # comp if < max distance of likely matches *= prior G match * radius:
            if dist < max_dist * (radii*icoef**3) * M:
                # rim/rng, but dists should be clustered in cluster_N_, before Ns?
                while len(_G.rim_) < rng-1: _G.rim_ += [[]]  # add empty rim for missing-rng comps
                while len(G.rim_) < rng-1: G.rim_ += [[]]  # rng-1: new rim may be added in comp_N
                Link = CL(nodet=[_G,G], S=2, A=[dy,dx], box=extend_box(G.box,_G.box))
                et = comp_N(Link, rn, rng)
                L_ += [Link]  # include -ve links
                if et is not None:
                    N_.update({_G, G}); Et += et  # for clustering
            else:
                Gp_ += [Gp]  # re-evaluate not-compared pairs with one incremented N.M:
        if Et[0] > ave * Et[2]:  # current-rng vM
            rng += 1
            N__+= [N_]; L__+= [L_]; ET += Et  # sub-cluster pL_,N_
            _Gp_ = [Gp for Gp in Gp_ if (Gp[0] in N_ or Gp[1] in N_)]  # one incremented N.M
        else:  # low projected rng+ vM
            break
    return N__,L__,ET,rng
'''
# pre-cluster pLs by difference in their distances:
L__ = sorted(pL_, key=lambda x: x.S)  # short links first
N__ = []
_L = L__[0]; pL_ = [_L]
for L in L__[1:]:
    ddist = L.S - _L.S  # always positive
    if ddist < ave_L:
        pL_ += [_L]  # cluster Ls with slowly increasing distances, or directly segment by distance? 
    else:
        N__ += list(set( [L.nodet[:] for L in pL_] ))  # ~= dist span for N clustering
        pL_ = [L]  # init ~= dist span
    _L = L
'''
def sort_by_dist(_L_, fd):  # sort Ls -> rngH, if cluster within rng, same for N.rim?

    _L_ = sorted(_L_, key=lambda x: x.dist)  # short links first
    Max_dist = max_dist
    N__,L_ = [],[]
    for L in _L_:
        if L.dist < Max_dist: L_ += [L]  # Ls within Max_dist
        else:
            if fd: N__ += [L_]  # L__ here, L_ may be empty
            else:  N__ += [list(set([node for L in L_ for node in L.nodet]))]  # dist span for N clustering, may be empty
            L_ = []  # init rng Lay
            Max_dist += max_dist
    if L_:  # last
        if fd: N__ += [L_]
        else:  N__ += [list(set([node for L in L_ for node in L.nodet]))]

    return  N__  # L__ if fd

def comp_ext(_L,L,_S,S,_A,A):  # compare non-derivatives:

    dL = _L - L; mL = min(_L,L) - ave_L  # direct match
    dS = _S/_L - S/L; mS = min(_S,S) - ave_L  # sparsity is accumulated over L
    mA, dA = comp_angle(_A,A)  # angle is not normalized
    M = mL + mS + mA
    D = abs(dL) + abs(dS) + abs(dA)  # normalize relative to M, signed dA?

    return CH(H=[mL,dL, mS,dS, mA,dA], Et=np.array([M,D,M>D,D<=M]), n=0.5)
'''
in sum2graph:
        graph.S += link.S
        graph.A = np.add(graph.A,link.angle)  # np.add(graph.A, [-link.angle[0],-link.angle[1]] if rev else link.angle)
'''

def add_md_C(HE, He, irdnt=[]):  # sum dextt | dlatt | dlayt

    HE.H[:] = [V + v for V, v in zip_longest(HE.H, He.H, fillvalue=0)]
    HE.n += He.n  # combined param accumulation span
    HE.Et += He.Et
    if any(irdnt): HE.Et[2:] = [E + e for E, e in zip(HE.Et[2:], irdnt)]
    return HE

def accum_lay(HE, He, irdnt):
    if HE.md_t:
        for MD_C, md_C in zip(HE.md_t, He.md_t):  # dext_C, dlat_C, dlay_C
            MD_C.add_md_C(md_C)

def comp_md_C(_He, He, rn=1, dir=1):

    vm, vd, rm, rd = 0,0,0,0
    derLay = []
    for i, (_d, d) in enumerate(zip(_He.H[1::2], He.H[1::2])):  # compare ds in md_ or ext
        d *= rn  # normalize by compared accum span
        diff = (_d - d) * dir  # in comp link: -1 if reversed nodet else 1
        match = min(abs(_d), abs(d))
        if (_d < 0) != (d < 0): match = -match  # negate if only one compared is negative
        vm += match - aves[i]  # fixed param set?
        vd += diff
        rm += vd > vm; rd += vm >= vd
        derLay += [match, diff]  # flat

    return CH(H=derLay, Et=np.array([vm,vd,rm,rd],dtype='float'), n=1)


