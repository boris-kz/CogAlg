def form_B__(nG, lG):  # form and trace edge / boundary / background per node:

    for Lg in lG.N_:  # in frame or blob?
        rB_ = {n.root for L in Lg.N_ for n in L.N_ if n.root and n.root.root is not None}  # rdn core Gs, exclude frame
        if rB_:
            rB_ = {(core,rdn) for rdn,core in enumerate(sorted(rB_, key=lambda x:(x.Et[0]/x.Et[2]), reverse=True), start=1)}
            Lg.rB_ = [rB_, np.sum([i.Et for i,_ in rB_], axis=0)]  # reciprocal boundary
    def R(L):
        return L.root if L.root is None or L.root.root is lG else R(L.root)
    for Ng in nG.N_:
        Et, Rdn, B_ = np.zeros(3), 0, []  # core boundary clustering
        LR_ = {R(L) for n in Ng.N_ for L in n.rim}  # lGs for nGs, individual nodes and rims are too weak to bound
        for LR in LR_:
            if LR and LR.rB_:  # not None, eval Lg.B_[1]?
                for core, rdn in LR.rB_[0]:  # map contour rdns to core N:
                    if core is Ng:
                        B_ += [LR]; Et += core.Et; Rdn += rdn
        Ng.B_ = [B_,Et, Rdn]

def clust_B__(G, lG, rc):  # trace edge / boundary / background per node:

    for Lg in lG.N_:  # form rB_, in Fg?
        rB_ = {n.root for L in Lg.N_ for n in L.N_ if n.root and n.root.root is not None}  # core Gs, exclude frame
        Lg.rB_ = sorted(rB_, key=lambda x:(x.Et[0]/x.Et[2]), reverse=True)  # N rdn = index+1

    def R(L): return L.root if L.root is None or L.root.root is lG else R(L.root)

    for N in G.N_:  # replace boundary Ls with RLs if any, rdn = stronger cores of RL
        B_, Et, rdn = [], np.zeros(3), 0
        for L in N.B_:  # neg Ls, may be clustered
            RL = R(L)
            if RL: B_ += [RL]; Et += RL.Et; rdn += RL.rB_.index(N) + 1
            else:  L.sub += 1; B_ += [L]; Et += L.Et; rdn += 1
        N.B_ = [B_, Et, rdn]
        if val_(Et,0,(len(N.B_)-1)*Lw, rc+rdn+compw) > 0:  # norm by core_ rdn
            trace_edge(N, rc)

def comp_Q(iN_, rc, fC):  # comp pairs of nodes or links within max_dist

    N__,L_,ET,O = [],[],np.zeros(3),1; rng=1; _N_ = copy(iN_)

    while True:  # _vM, rng in rim only? dir_cluster rng-layered pre-links for base Ns?
        N_, Et,o = [],np.zeros(3),1
        for N in _N_: N.pL_ = []
        for _N, N in combinations(_N_, r=2):
            if _N in N.compared or _N.sub != N.sub: continue
            if fC==2:  # dCs
                m_, d_ = comp_derT(_N.derTT[1], N.derTT[1])
                ad_ = np.abs(d_); t_ = m_ + ad_ + eps  # ~ max comparand
                et = np.array([m_ / t_ @ wTTf[0], ad_ / t_ @ wTTf[1], min(_N.Et[2], N.Et[2])])  # signed
                dC = CN(N_=[_N,N], Et=et); L_ += [dC]; Et += et
                for n in _N, N:
                    N_ += [n]; n.rim += [dC]; n.et += et
            else:  # spatial
                dy_dx = _N.yx- N.yx; dist = np.hypot(*dy_dx)  # * angl in comp_A, for canonical links in L_
                olp = (N.rc + _N.rc) / 2
                if fC or ({l for l in _N.rim if l.Et[0] > ave} & {l for l in N.rim if l.Et[0] > ave}):  # common match, if bilateral proj eval
                    Link = comp_N(_N,N, olp, rc, A=dy_dx, span=dist, rng=rng, lH=L_)
                    if val_(Link.Et, aw=contw+olp+rc) > 0:
                        N_+= [_N,N]; Et+=Link.Et; o+=olp
                else:
                    V = proj_V(_N, N, dy_dx, dist)  # or prelink dir_cluster, proj to nearest _N in dir only?
                    if adist * V/olp > dist:
                        N.pL_ += [[dist,dy_dx,olp,_N]]; _N.pL_ += [[dist,-dy_dx,olp,N]]
        if not fC:  # else compared directly above
            cT_ = set()  # compared pairs per loop
            for N in _N_:
                for dist, dy_dx, olp, _N in N.pL_:
                    cT = tuple(sorted((N.id,_N.id)))
                    if cT in cT_: continue
                    cT_.add(cT)
                    Link = comp_N(_N, N, olp, rc, A=dy_dx, span=dist, rng=rng, lH=L_)
                    if val_(Link.Et, aw=contw+rc+olp) > 0:
                        N_+= [_N,N]; Et+= Link.Et; o+=olp
        for N in _N_:
            delattr(N,'pL_')
        N_ = list(set(N_))
        if fC: N__= N_; ET = Et; break  # no rng-banding
        elif N_:
            N__+= [N_]; ET+=Et; O+=o  # rng+ eval / loop
            if not fC and val_(Et, mw=(len(N_)-1)*Lw, aw=compw+rc+o) > 0:  # current-rng vM
                _N_ = N_; rng +=1
            else: break
        else: break
    return N__, L_, ET, O

