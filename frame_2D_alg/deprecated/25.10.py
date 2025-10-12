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

def dir_sort(N):  # get rng_ pre-links per direction

    dir_ = []
    for dist, angl, _N in N.pL_:
        max_mA =-1; mA_dir = None
        for dir in dir_:
            mA,_ = comp_A(angl, dir[0])
            if mA > .8 and mA > max_mA:
                max_mA = mA; max_dir = dir
        if mA_dir: mA_dir[0] += angl; mA_dir[1] += [[dist,angl,_N]]
        else:      dir_+= [[copy(angl), [[dist,angl,_N]]]]  # new dir
    for _,pL_ in dir_:
        pL_[:] = sorted(pL_, key=lambda x: x[0])  # by distance
    N.pL_ = dir_

def comp_q(iN_, rc, fC):  # comp pairs of nodes or links if proj_V per dir

    N_,L_, Et,o = [],[],np.zeros(3),1
    if fC:
        for _N, N in combinations(iN_, r=2):  # no olp for dCs?
            m_, d_ = comp_derT(_N.derTT[1], N.derTT[1])
            ad_ = np.abs(d_); t_ = m_ + ad_ + eps  # ~ max comparand
            et = np.array([m_ / t_ @ wTTf[0], ad_ / t_ @ wTTf[1], min(_N.Et[2], N.Et[2])])  # signed
            dC = CN(N_=[_N,N], Et=et); L_ += [dC]; Et += et  # add o?
            for n in _N, N:
                N_ += [n]; n.rim += [dC]; n.et += et
    else:   # spatial
        for i, N in enumerate(iN_):
            N.pL_ = []
            for _N in iN_[i:]:  # get unique pairs
                if _N.sub == N.sub:  # check max dist?
                    dy_dx = _N.yx- N.yx; dist = np.hypot(*dy_dx)  # * angl in comp_A, for canonical links in L_
                    N.pL_ += [[dist,dy_dx,_N]]  # no reciprocal
        for N in iN_:
            if not N.pL_: continue
            dir_sort(N)  # get rng_ pre-links per direction
            for dir in N.pL_:
                for dist,dy_dx,_N in dir:  # enumerate rng?
                    V = proj_V(_N, N, dy_dx, dist)
                    olp = (N.rc + _N.rc) / 2
                    if adist * V/olp > dist:
                        Link = comp_N(_N,N, olp, rc, A=dy_dx, span=dist, lH=L_)
                        if val_(Link.Et, aw=olp+rc) > 0:
                            N_+= [_N,N]; Et+=Link.Et; o+=olp
                    else:
                        break  # distant pre_V = proj_V * mA, ?negative?
        sL_ = sorted(L_,key=lambda l:l.span)
        L_, Lseg = [], [L_[0]]
        for _L,L in zip(L_,L_[1:]):  # segment by ddist:
            if _L.span-L.span < adist: Lseg += [L]
            else: L_ += [Lseg]; Lseg =[L]

    return list(set(N_)), L_, Et,o

def dir_cluster(N):  # get neg links to block projection?

    dir_ = []
    for pL in N.pL_:
        angl = pL[1]; max_pL_t = []; max_mA = -1
        for pL_t in dir_:
            Angl,_ = pL_t
            mA,_ = comp_A(angl, Angl)
            if mA > .8 and mA > max_mA: max_mA = mA; max_pL_t = pL_t
        if max_pL_t: max_pL_t[0] += angl; max_pL_t[1] += [pL]
        else:        dir_ += [[copy(angl), [pL]]]  # init dir
    sel_pL_ = []
    for _,pL_ in dir_:
        sel_pL_ += [pL_[ np.argmin([pL[0] for pL in pL_])]]  # nearest pL per direction
    N.pL_ = sel_pL_

    rdist = _dist / dist  # dist > _dist
    V += _V * mA * rdist  # mA in 0:1, symmetrical, but ?

def cross_proj(_N, N, angle, dist):  # estimate cross-induction between N and _N before comp

    def proj_L_(L_, int_w=1):
        V = 0
        for L in L_:
            cos = ((L.angl[0]*L.angl[1]) @ angle) / np.hypot(*(L.angl[0]*L.angl[1]) * np.hypot(*angle))  # angl: [[dy,dx],dir]
            mang = (cos+ abs(cos)) / 2  # = max(0, cos): 0 at 90°/180°, 1 at 0°
            V += L.Et[1-fi] * mang * int_w * mW * (L.span/dist * av)  # decay = link.span / l.span * cosine ave?
            # += proj rim-mediated nodes?
        return V
    V = 0; fi = N.fi; av = ave if fi else avd
    for node in _N,N:
        if node.et[2]:  # mainly if fi?
            v = node.et[1-fi]  # external, already weighted?
            V+= proj_L_(node.rim) if v*mW > av*node.rc else v  # too low for indiv L proj
        v = node.Et[1-fi]  # internal, lower weight
        V+= proj_L_(node.L_,intw) if (fi and v*mW > av*node.rc) else v  # empty link L_

    return V * dec * (dist / ((_N.span+N.span)/2))

def comp_Q1(iN_, rc, fC):

    N_,L_, Et,olp = [],[],np.zeros(3),1
    if fC:
        for _N, N in combinations(iN_, r=2):  # no olp for dCs?
            m_, d_ = comp_derT(_N.derTT[1], N.derTT[1])
            ad_ = np.abs(d_); t_ = m_ + ad_ + eps  # ~ max comparand
            et = np.array([m_ / t_ @ wTTf[0], ad_ / t_ @ wTTf[1], min(_N.Et[2], N.Et[2])])  # signed
            dC = CN(N_=[_N,N], Et=et); L_ += [dC]; Et += et  # add o?
            for n in _N, N:
                N_ += [n]; n.rim += [dC]; n.et += et
            L_ = [L_]  # convert to nested, to  enable a same unpacking sequence in Cluster
    else: # spatial
        for i, N in enumerate(iN_):  # get unique pre-links per N, not _N
            N.pL_ = []
            for _N in iN_[i+1:]:
                if _N.sub != N.sub: continue  # not sure
                dy_dx = _N.yx - N.yx; dist = np.hypot(*dy_dx)
                N.pL_ += [[dist, dy_dx, _N]]
            N.pL_.sort(key=lambda x: x[0])  # global distance sort
        for N in iN_:
            pVt_ = []  # [dist, dy_dx, _N, V]
            for dist, dy_dx, _N in N.pL_:  # angl is not canonic in rim?
                O = (N.rc+_N.rc) / 2       # G|PP node induction, if dec is defined per adist:
                V = proj_V(_N,N, dy_dx, dist) if _N.L_ and N.L_ else val_((_N.Et+N.Et) * (dec** (dist/adist)), aw=rc+O)
                # + induction of pri L Vs projected on curr pL:
                for _dist,_dy_dx,__N,_V in pVt_:
                    mA, _ = comp_A(dy_dx, _dy_dx)   # mA and rel dist in 0:1:
                    ldist = np.hypot(*(_N.yx-__N.yx)) /2  # between link midpoints
                    rdist = ldist / ((_dist+dist)/2)  # relative to ave link, exp decay:
                    V += _V * (dec** (rdist/adist)) * ((mA*wA + dist/_dist*distw) / (wA+distw))
                    # _V decay / dist, only partial cancel by link ext attr miss?
                if V > ave*O:
                    Link = comp_N(_N,N, O,rc, A=dy_dx, span=dist, lH=L_)
                    if val_(Link.Et, aw=contw+O+rc) > 0:
                        N_ += [_N,N]; Et+= Link.Et; olp+=O
                    V = val_(Link.Et, aw=rc+O)  # else keep V
                else: break
                pVt_ += [[dist, dy_dx, _N, V]]

    return list(set(N_)), L_, Et, olp

