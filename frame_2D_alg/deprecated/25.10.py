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
'''
    for dist, dy_dx, _N in N.pL_:  # angl is not canonic in rim?
            O = (N.rc+_N.rc) / 2; Ave=ave*rc*O
            iV = (_N.Et[0]+N.Et[0]) * dec** (dist/(_N.span+N.span)/2) - Ave
            eV = (_N.et[0]+N.et[0]) * dec** (dist/adist) - Ave  # we need ave rim span instead of adist
            V = iV+eV; fcomp = 1 if V>Ave else 0 if V<specw else 2  # uncertainty
            if fcomp==2:
                if eV > specw:
                    eV = 0  # recompute from individual ext Ls
                    for _dist,_dy_dx,__N,_V in pVt_:  # * link ext miss value?
                        mA, _ = comp_A(dy_dx,_dy_dx)  # mA and rel dist in 0:1:
                        ldist = np.hypot(*(_N.yx-__N.yx)) /2  # between link midpoints
                        rdist = ldist / ((_dist+dist)/2)
                        eV += _V * (dec** (rdist/adist)) * ((mA*wA + dist/_dist*distw) / (wA+distw))
                    V = iV+eV; fcomp = 1 if V>Ave else 0 if V<specw else 2
                if fcomp==2 and N.fi and _N.L_ and N.L_ and iV > specw:  # different specw for L_?
                    iV = proj_L_(_N,N, dy_dx, dist)  # recompute from individual Ls
                    V = iV+eV; fcomp = V > 0  # no further spec
            if fcomp:
                Link = comp_N(_N,N, O,rc, A=dy_dx, span=dist, lH=L_)
                if val_(Link.Et, aw=contw+O+rc) > 0:
                    N_ += [_N,N]; Et+= Link.Et; olp+=O
                V = val_(Link.Et, aw=rc+O)  # else keep V
            else: break
            pVt_ += [[dist, dy_dx, _N, V]]
'''
def proj_Fg(Fg, yx):

    def proj_N(N):
        dy, dx = N.yx - yx
        Ndist = np.hypot(dy, dx)  # external dist
        rdist = Ndist / N.span
        Angle = np.array([dy, dx]); angle = N.angl[0] * N.angl[1]  # external and internal angles
        cos_d = N.angl[0].dot(Angle) / (np.hypot(*angle) * Ndist)  # N-specific alignment, * N.angl[1]?
        M,D,n = N.Et  # add N.et? Because Fg.Et is summed from graph.Et, which summed from link.Et, and it's summed in N.et too
        dec = rdist * (M/(M+D))  # match decay rate, * ddecay for ds?
        prj_H = proj_dH(N.derH.H, cos_d * rdist, dec)
        prjTT, pEt = sum_H( prj_H)
        pD = pEt[1]*dec; dM = M*dec  # use val_ for any proj?
        pM = dM - pD * (dM/(ave*n))  # -= borrow, regardless of surprise?
        return np.array([pM,pD,n]), prjTT, prj_H

    specw = 9
    ET, DerTT, DerH = proj_N(Fg)
    pV = val_(ET, mw=len(Fg.N_)*Lw, aw=contw)
    if pV > ave: return Fg
    elif pV > specw:
        ET = np.zeros(3); DerTT = np.zeros((2,9)); N_ = []
        for N in Fg.N_:  # sum _N-specific projections for cross_comp
            if N.derH:
                pEt, prjTT, prj_H = proj_N(N)
                if val_(pEt, aw=contw):
                    ET+=pEt; DerTT+=prjTT; N_+= [CN(N_=N.N_,Et=pEt,derTT=prjTT,derH=prj_H,root=CN())]  # same target position?
        if val_(ET, mw=len(N_)*Lw, aw=contw):
            return CN(N_=N_,L_=Fg.L_,Et=ET, derTT=DerTT)  # proj Fg, add Prj_H?

def comp_N_(iN_, rc):

    def proj_L_(_N, N, angle, dist):  # estimate cross-induction between N and _N before comp

        V = 0; fi = N.fi; av = ave if fi else avd
        for edir, node in zip((1,-1),(_N,N)):
            for L in node.L_:
                '''
                cos = ((L.angl[0] *edir *L.angl[1]) @ angle) / (np.hypot(*L.angl[0]) * np.hypot(*angle))  # angl: [[dy,dx],dir]
                mang = (cos + abs(cos)) / 2  # = max(0, cos): 0 at 90°/180°, 1 at 0°
                V += L.Et[1-fi] * mang * intw * mW * (L.span/dist * av)  # decay = link.span / l.span * cosine ave?
                '''
                dy, dx = L.yx - node.yx
                Ndist = np.hypot(dy, dx)  # external dist
                rdist = Ndist / L.span
                Angle = node.angl[0] * node.angl[1]; angle = L.angl[0] * L.angl[1]  # external and internal angles
                cos_d = angle.dot(Angle) / (np.hypot(*angle) * Ndist)  # N-to-yx alignment
                M, D, cnt = L.Et
                dec = rdist * (M / (M+D))  # match decay rate, * ddecay for ds?
                NEt, NTT, NH = proj_dH(L.derH.H, cos_d, dec, M, cnt)
                V += val_(NEt, mw=(len(L.N_)-1)*Lw, aw=contw)
        return V

    def proj_V(_N, N, dist, dy_dx, Ave, specw, pVt_):
            # _N x N induction
        iV = (_N.Et[0]+N.Et[0]) * dec**(dist/((_N.span+N.span)/2)) - Ave
        eV = (_N.et[0]+N.et[0]) * dec**(dist/np.mean([l.span for l in _N.rim+N.rim])) - Ave  # ave rim span
        V = iV + eV
        if V > Ave or V < specw: return V
        if eV > specw:
            eV = 0
            for _dist, _dy_dx, __N, _V in pVt_:
                pN = proj_N(N,_dist,_dy_dx); _pN = proj_N(N,_dist,-_dy_dx)
                eV += val_(_pN.Et+pN.Et)
                mA, _ = comp_A(dy_dx, _dy_dx)
                ldist = np.hypot(*(_N.yx-__N.yx)) /2
                rdist = ldist / ((_dist + dist) / 2)
                eV += _V * (dec**(rdist/adist)) * ((mA*wA + dist/_dist*distw) / (wA+distw))
            V = iV + eV  # specific eV from prox links
        if V > Ave: return V
        if N.fi and _N.L_ and N.L_ and iV > specw:
            V = eV + proj_L_(_N,N, dy_dx, dist)  # specific iV from L_s
        return V

    N_,L_, Et,olp = [],[], np.zeros(3),1
    for i, N in enumerate(iN_):  # get unique pre-links per N, not _N, prox prior
        N.pL_ = []
        for _N in iN_[i+1:]:
            if _N.sub != N.sub: continue  # not sure
            dy_dx = _N.yx - N.yx; dist = np.hypot(*dy_dx)
            N.pL_ += [[dist, dy_dx, _N]]
        N.pL_.sort(key=lambda x: x[0])  # global distance sort
    for N in iN_:
        pVt_ = []  # [dist, dy_dx, _N, V]
        for dist, dy_dx, _N in N.pL_:  # rim angl not canonic
            O = (N.rc +_N.rc) / 2; Ave = ave * rc * O
            V = proj_V(_N,N, dist, dy_dx, Ave,2, pVt_)
            if V > Ave:
                Link = comp_N(_N,N, O,rc, A=dy_dx, span=dist, lH=L_)
                if val_(Link.Et, aw=contw+O+rc) > 0:
                    N_ += [_N, N]; Et += Link.Et; olp += O
                pVt_ += [[dist, dy_dx, _N, val_(Link.Et, aw=rc+O)]]
            else:
                break  # no induction
    return list(set(N_)), L_, Et, olp

def proj_H(cH, cos_d, dec):

    pH = CdH()
    for _lay in cH.H:
        pTT = np.array([_lay.derTT[0], _lay.derTT[1] * cos_d * dec])
        pEt = np.array([_lay.Et[0], np.sum(pTT[1]), _lay.Et[2]])  # or _lay Et is recomputed from proj_H:?
        lay = CdH(H=[proj_H(l, cos_d, dec) for l in _lay.H], Et=pEt, derTT=pTT, root=pH)
        pH.H += [lay]; pH.Et += pEt; pH.derTT += pTT
    pD = pH.Et[1]  # already *= dec
    dM = cH.Et[0] * dec
    n  = cH.Et[2]
    pM = dM - pD * (dM / (ave*n))  # -= borrow, scaled by rV of normalized decayed M
    pH.Et = np.array([pM, pD, n])
    return pH

def cross_comp_(Fg_, rc):  # top-composition xcomp, add margin search extend and splice?

        n_,l_,c_ = [],[],[]
        for g in Fg_:
            if g: n_ += g.N_; l_ += g.L_; c_ += g.C_[0] if g.C_ else []  # + dC_?
        nG, lG, cG = cross_comp(CN(N_=n_),rc), cross_comp(CN(N_=l_),rc+1), cross_comp(CN(N_=c_),rc+2,fC=1)

        Et = np.sum([g.Et for g in (nG,lG,cG) if g], axis=0)
        rc = np.mean([g.rc for g in (nG,lG,cG) if g])

        return CN(Et=Et, rc=rc, N_= nG.N_ if nG else [], L_= lG.N_ if lG else [], C_= cG.N_ if cG else [],
                  nH=nG.nH if nG else [], lH=lG.nH if lG else [])
                  # we need to assign nH and lH for feedback too?

def Cluster_F(root, iL_, rc):  # called from cross_comp(Fg_)

    dN_, dL_, dC_ = [], [], []
    for Link in iL_:  # links between Fgs
        dN_ += Link.N_; dL_ += Link.L_; dC_ += Link.C_
    Et = np.zeros(3)
    N_,L_,C_ = [],[],[]
    if len(dN_) > 1:
        nG = cluster_N(root, dN_, rc)
        if nG:
            rc+=1; N_ = nG.N_; Et += nG.Et
            if val_(nG.Et, mw=(len(N_)-1)+Lw, aw=rc) > 0:
                nG = cross_comp(nG, rc)
                if nG: rc+=1; N_ = nG.N_; Et += nG.Et
    if len(dL_) > 1:
        lG = cluster_N(root, dL_, rc)
        if lG:
            rc+=1; L_+= lG.N_; Et += lG.Et
            if val_(nG.Et, mw=(len(L_)-1)+Lw, aw=rc) > 0:
                lG = cross_comp(nG, rc)
                if lG: rc+=1; N_ = nG.N_; Et += nG.Et
    if len(dC_) > 1:
        cG = cluster_n(root, dC_, rc)
        if cG:
            rc+=1; C_+= cG.N_; Et += cG.Et
            if val_(cG.Et, mw=(len(C_)-1)+Lw, aw=rc) > 0:
                cG = cross_comp(cG, rc)
                if cG: rc+=1; N_ = nG.N_; Et += nG.Et

    return CN(Et=Et,rc=rc, N_=N_,L_=L_,C_=C_, root=root)

def spec(_N_,N_, Et, olp, L_=None):  # for N_|B_

    for _N in _N_:
        for N in N_:
            if _N is not N:
                L = comp_N(_N,N, olp, rc=1, fspec=1); Et += L.Et
                if L_ is not None: L_+=[L]  # splice if Fcluster
                for _l,l in [(_l,l) for _l in _N.rim for l in N.rim]:  # l nested in _l
                    if _l is l: Et += l.Et  # overlap val?

def comp_N(_N,N, olp,rc, A=np.zeros(2), span=None, rng=1, lH=None, fspec=0):  # compare links, optional angl,span,dang?

    derTT, Et, rn = base_comp(_N, N); fN = N.root  # not Fg
    baseT = (rn*_N.baseT+N.baseT) /2  # not new
    yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])  # ext
    fi = N.fi
    angl = [A, np.sign(derTT[1] @ wTTf[1])]  # preserve canonic direction
    Link = CN(fi=0, Et=Et,rc=olp, nt=[_N,N], N_=_N.N_+N.N_, L_=_N.L_+N.L_, baseT=baseT,derTT=derTT, yx=yx, box=box, span=span, angl=angl, rng=rng)
    V = val_(Et, aw=olp+rc)
    if fN and V * (1- 1/(min(len(N.derH.H),len(_N.derH.H)) or eps)) > ave:  # rdn to derTT, else derH is empty
        H = [CdH(Et=Et, derTT=copy(derTT), root=Link)]  # + 2nd | higher layers:
        if _N.derH and N.derH:
            dH = comp_dH(_N.derH, N.derH, rn, Link)
            H += dH.H; dTT = dH.derTT; dEt = dH.Et
        else:
            m_,d_ = comp_derT(rn*_N.derTT[1], N.derTT[1]); dEt = np.array([np.sum(m_),np.sum(d_),min(_N.Et[2],N.Et[2])]); dTT = np.array([m_,d_])
            H += [CdH(Et=dEt, derTT=dTT, root=Link)]
        Et += dEt; derTT += dTT
        Link.derH = CdH(H=H, Et=Et, derTT=derTT, root=Link)  # same as Link Et,derTT
    if fi and V > ave * (rc+1+compw):
        rc += olp
        if N.L_: rc+=1; spec(_N.N_, N.N_, Et, rc, Link.N_)  # if N.L_ to skip PP
        if fN:  # not Fg
            if _N.B_ and N.B_:
                _B_,_bEt,_bO = _N.B_; B_,bEt,bO = N.B_; bO+=_bO+rc
                if val_(_bEt+bEt,1,(min(len(_B_),len(B_))-1)*Lw, bO+compw) > 0:
                    rc+=1; spec(_B_,B_, Et, bO, Link.lH)  # lH = dspec; spec C_: overlap,offset?
        else:
            # Fg, global C_, -+L_, lower arg rc, splice Link N_,L_,C_ globally, no clustering?
            if V * (min(len(_N.L_),len(N.L_))-1)*Lw > ave*rc:  # add N.et,olp?
                rc+=1; spec(_N.L_,N.L_, Et, rc, Link.L_)
            if _N.C_ and N.C_:
                _C_,_cEt = _N.C_; C_,cEt = N.C_  # add val_ cEt, no separate olp?
                if V * (min(len(C_),len(C_))-1)*Lw > ave*rc:
                    rc+=1; spec(_C_,C_, Et, rc, Link.C_)
    if lH is not None:
        lH += [Link]
    # span,angl in spec?
    for n, _n in (_N,N), (N,_N):  # if rim-mediated comp: reverse dir in _N.rim: rev^_rev?
        n.rim += [Link]; n.et += Et; n.compared.add(_n)
    return Link

def val_(Et, fi=1, mw=1, aw=1, _Et=np.zeros(3)):  # m,d eval per cluster, for projection only?

    if mw <= 0: return (0,0) if fi == 2 else 0
    am = ave * aw  # includes olp, M /= max I | M+D? div comp / mag disparity vs. span norm
    ad = avd * aw  # dval is borrowed from co-projected or higher-scope mval
    m, d, n = Et
    # m,d may be negative, but deviation is +ve?
    if fi==2: val = np.array([m-am, d-ad]) * mw  # abs ! rel?
    else:     val = (m-am if fi else d-ad) * mw  # m: m/ (m+d), d: d/ (m+d)?
    if _Et[2]:
        _m,_d,_n = _Et
        rn =_n/n  # borrow rational deviation of contour if fi else root Et:
        if fi==2: val *= np.array([_d/ad, _m/am]) * mw * rn
        else:     val *= (_d/ad if fi else _m/am) * mw * rn
    return val

def val__(Et, fi=1, mw=1, aw=1, _Et=np.zeros(4)):  # m,d eval per cluster

    if mw <= 0: return (0,0) if fi == 2 else 0
    m, d, n, t = Et
    rm = m / t; rd = d / t  # t = np.sum(np.abs(derTT))
    if _Et[2]:
        _m,_d,_n,_t = _Et
        rn = _n / n  # weight of parent's contribution
        rm = rm * (1-rn) + (_d/_t) * rn  # match borrows from parent diff
        rd = rd * (1-rn) + (_m/_t) * rn  # diff borrows from parent match
    vm = rm * mw - ave* aw
    vd = rd * mw - avd* aw  # raves
    if fi==1: return vm
    if fi==0: return vd
    else:     return vm,vd

def base_comp(_N,N, fC=0):  # comp Et, baseT, extT, derTT

    fi = N.fi
    _M,_D,_n =_N.Et; _I,_G,_Dy,_Dx =_N.baseT; _L = len(_N.N_)  # len nodet.N_s, no baseT in links?
    M, D, n  = N.Et; I, G, Dy, Dx = N.baseT; L = len(N.N_)
    rn = _n/n
    _pars = np.array([_M*rn,_D*rn,_n*rn,_I*rn,_G*rn, [_Dy,_Dx],_L*rn,_N.span], dtype=object)  # Et, baseT, extT
    pars  = np.array([M,D,n, (I,aI),G, [Dy,Dx], L,(N.span,aS)], dtype=object)
    mA,dA = comp_A(_N.angl[0]*_N.angl[1], N.angl[0]*N.angl[1])  # ext angle
    m_,d_ = comp(_pars,pars, mA,dA)  # M,D,n, I,G,A, L,S,eA
    if fC:
        dm_,dd_ = comp_derT(rn*_N.derTT[1], N.derTT[1])
        m_+=dm_; d_+=dd_
    DerTT = np.array([m_,d_])
    ad_= np.abs(d_)
    t_ = m_ + ad_ + eps  # max comparand
    Et = np.array([m_ * (1, mA)[fi] /t_ @ wTTf[0],  # norm, signed?
                   ad_* (1, 2-mA)[fi] /t_ @ wTTf[1], min(_n,n)])  # shared?

class CN(CBase):
    name = "node"
    def __init__(n, **kwargs):
        super().__init__()
        n.fi  = kwargs.get('fi', 1)  # if G else 0, fd_: list of forks forming G?
        n.nt  = kwargs.get('nt',[])  # nodet, empty if fi
        n.N_  = kwargs.get('N_',[])  # nodes, concat in links
        n.L_  = kwargs.get('L_',[])  # internal links, +|- if failed?
        n.rim = kwargs.get('rim',[])  # external links, rng-nest?
        n.Et  = kwargs.get('Et',np.zeros(4))  # sum from L_
        n.et  = kwargs.get('et',np.zeros(4))  # sum from rim
        n.rc  = kwargs.get('rc',1)  # redundancy to ext Gs, ave in links? separate rc for rim, or internally overlapping?
        n.baseT = kwargs.get('baseT', np.zeros(4))  # I,G,A: not ders
        n.derTT = kwargs.get('derTT',np.zeros((2,9)))  # sum derH -> m_,d_ [M,D,n, I,G,A, L,S,eA], dertt: comp rims + overlap test?
        n.derH  = kwargs.get('derH', CdH())  # sum from clustered L_s
        n.dLay  = kwargs.get('dLay', CdH())  # sum from terminal L_, Fg only?
        n.yx  = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.rng = kwargs.get('rng',1)  # or med: loop count in comp_node_|link_
        n.box = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.span = kwargs.get('span',0) # distance in nodet or aRad, comp with baseT and len(N_) but not additive?
        n.angl = kwargs.get('angl',[np.zeros(2),0])  # (dy,dx),dir, sum from L_
        n.mang = kwargs.get('mang',1)  # ave match of angles in L_, =1 in links
        n.B_ = kwargs.get('B_', [])  # ext boundary | background neg Ls/lG: [B_,Et,R], add dB_?
        n.rB_= kwargs.get('rB_',[])  # reciprocal cores for lG, vs higher-der lG.B_
        n.C_ = kwargs.get('C_', [])  # int centroid Gs, add dC_?
        n.rC_= kwargs.get('rC_',[])  # reciprocal root centroids
        n.nH = kwargs.get('nH', [])  # top-down hierarchy of sub-node_s: CN(sum_N_(Nt_))/ lev, with single added-layer derH, empty nH
        n.lH = kwargs.get('lH', [])  # bottom-up hierarchy of L_ graphs: CN(sum_N_(Lt_))/ lev, within each nH lev
        n.root = kwargs.get('root',None)  # immediate
        n.sub  = 0  # full-composition depth relative to top-composition peers
        n.fin  = kwargs.get('fin',0)  # clustered, temporary
        n.exe  = kwargs.get('exe',0)  # exemplar, temporary
        n.compared = set()
        # n.fork_tree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def __bool__(n): return bool(n.N_)

def add_dH(DH, dH):  # rn = n/mean, no rev, merge/append lays

    DH.Et += dH.Et
    DH.dTT += dH.dTT
    off_H = []
    for D, d in zip_longest(DH.H, dH.H):
        if D and d: add_dH(D, d)
        elif d:     off_H += [copy_(d, DH)]
    DH.H += off_H
    return DH

def comp_dH(_dH, dH, rn, root):  # unpack derH trees down to numericals and compare them

    H = []
    if _dH.H and dH.H:  # 2 or more layers each, eval rdn to dTT as in comp_N?
        Et = np.zeros(3); dTT = np.zeros((2,9))
        for D, d in zip(_dH.H, dH.H):
            ddH = comp_dH(D,d, rn, root); H += [ddH]; Et += ddH.Et; dTT += ddH.dTT
    else:
        dTT = comp_derT(_dH.dTT[1], dH.dTT[1] * rn)  # ext A align replaced dir/rev
        Et = np.array([np.sum(dTT[0]), np.sum(np.abs(dTT[1])), min([_dH.Et[2],dH.Et[2]])])

    return CdH(H=H, Et=Et, dTT=dTT, root=root)

def sum_H(H):  # use add_dH?
    dTT = np.zeros((2,9)); Et = np.zeros(3)
    for lay in H:
        dTT += lay.dTT; Et += lay.Et
    return dTT, Et

def sum2graph(root, N_,L_,C_,B_,olp,rng):  # sum node,link attrs in graph, aggH in agg+ or player in sub+

    n0 = Copy_(N_[0]); yx_=[n0.yx]; box=n0.box; baseT=n0.baseT; derH=n0.derH; dTT=np.zeros((2,9)); ang=np.zeros(2)
    fi = n0.fi; fg = fi and n0.L_  # not PPs
    for L in L_:
        add_dH(derH,L.derH); dTT+=L.dTT; ang += L.angl[0]
    A = np.array([ang, np.sign(dTT[1] @ wTTf[1])],dtype=object)  # canonical dir = summed diff sign
    if fg: Nt = Copy_(n0)  # add_N(Nt,Nt.Lt)?
    dTT += n0.dTT
    for N in N_[1:]:
        add_dH(derH,N.derH); baseT+=N.baseT; dTT+=N.dTT; box=extend_box(box,N.box); yx_+=[N.yx]
        if fg: add_N(Nt,N)  # froot = 0
    yx = np.mean(yx_,axis=0); dy_,dx_ = (yx_-yx).T; dist_ = np.hypot(dy_,dx_); span = dist_.mean() # node centers distance to graph center
    graph = CN(root=root, fi=1,rng=rng, N_=N_,L_=L_,C_=C_,B_=B_,rc=olp, baseT=baseT, dTT=dTT, derH=derH, span=span, angl=A, yx=yx)
    for n in N_: n.root = graph
    if fg: graph.nH = Nt.nH + [Nt]  # pack prior top level
    if fi and len(L_) > 1:  # else default mang = 1
        graph.mang = np.sum([ comp_A(ang, l.angl[0]) for l in L_]) / len(L_)
        # ang is centroid, directionless?
    graph.M = sum(graph.dTT[0]); graph.D = sum(graph.dTT[1]); graph.n = graph.dTT[2]
    return graph

def cross_comp(root, rc, fC=0):  # rng+ and der+ cross-comp and clustering

    if fC: N_,L_,dTT = comp_C_(root.N_,rc); O = 0  # no alt rep is formed, cont along maxw attr?
    else:  N_,L_,dTT,O = comp_N_(root.N_,rc)  # rc: redundancy+olp, lG.N_ is Ls
    if len(L_) > 1:
        dV = val_(dTT, O+rc+compw, fi=0, mw=(len(L_)-1)*Lw); lG = []
        if dV > 0:
            if root.fi and root.L_: root.lH += [sum_N_(root.L_,rc,root)]  # or agglomeration root is always Fg?
            root.L_=L_; root.dTT += dTT; root.rc += O
            if fC < 2 and dV > avd:  # may be dC_, no comp ddC_
                lG = cross_comp(CN(N_=L_,root=root), O+rc+compw+1, fC*2)  # trace_edge via rB_|B_
                if lG: rc+=lG.rc; root.lH += [lG]+lG.nH; root.dTT+=lG.dTT; add_dH(root.dLay, lG.derH)  # lH extension
        if val_(dTT, O+rc+compw, mw=(len(L_)-1)*Lw) > 0:
            # tentative clust eval, no n.ed:
            for n in N_: n.em = sum([l.m for l in n.rim]) / len(n.rim)
            if root.root: nG = Cluster(root, L_, rc+O, fC)  # fC=0: get_exemplars, cluster_C, rng connect cluster
            else:         nG = Fcluster(root,L_, rc+O)  # root=frame, splice,cluster L'L_,lH_,C_
            if nG:        # batched nH extension
                rc+=nG.rc # redundant clustering layers
                if lG:
                    form_B__(nG, lG, rc+O+2)  # assign boundary per N, O+=B_[-1]|1?
                    if val_(dTT, rc+O+3+contw, mw=(len(nG.N_)-1)*Lw) > 0:  # mval
                        trace_edge(nG, rc+O+3)  # comp adjacent Ns via B_
                if val_(nG.dTT, rc+O+compw+3, mw=(len(nG.N_)-1)*Lw, _dTT=dTT) > 0:
                    nG = cross_comp(nG, rc+O+3) or nG  # connec agg+, fC = 0
                root.dTT += nG.dTT; root.N_ = nG.N_
                _H = root.nH; root.nH = []  # nG has own L_,lH
                nG.nH = _H+ [root] + nG.nH  # pack root.nH in higher-composition nG.nH
                return nG  # update root

def Fcluster(root, iL_, rc):  # called from cross_comp(Fg_)

    dN_, dL_, dC_ = [], [], []  # splice specs from links between Fgs within a larger frame
    for Link in iL_: dN_ += Link.L_; dL_ += Link.lH; dC_ += Link.C_

    dTT = np.zeros((2,9)); N_L_C_ = [[],[],[]]; c=0
    for i, (link_,clust, fC) in enumerate([(dN_,cluster_N,0),(dL_,cluster_N,0),(dC_,cluster_n,1)]):
        if link_:
            G = clust(root, link_, rc)
            if G:
                rc+=1; N_L_C_[i] = G.N_; dTT += G.dTT
                if val_(G.dTT, rc, mw=(len(G.N_)-1)*Lw) > 0:
                    G = cross_comp(G, rc, fC=fC)
                    if G: rc+=1; c += G.c; N_L_C_[i] = G.N_; dTT += G.dTT
    N_,L_,C_ = N_L_C_
    return CN(dTT=dTT, m=sum(dTT[0]), d=sum(dTT[1]), c=c, rc=rc, N_=N_,L_=L_,C_=C_, root=root)

class CN(CBase):
    name = "node"
    def __init__(n, **kwargs):
        super().__init__()
        n.fi  = kwargs.get('fi',1)  # if G else 0, fd_: list of forks forming G?
        n.nt  = kwargs.get('nt',[])  # nodet, empty if fi
        n.N_  = kwargs.get('N_',[])  # nodes, concat in links
        n.L_  = kwargs.get('L_',[])  # internal links, +|- if failed?
        n.rim = kwargs.get('rim',[])  # external links, rng-nest?
        n.m   = kwargs.get('m',0); n.d = kwargs.get('d',0); n.c = kwargs.get('c',0)   # sum L_ dTT -> rm, rd, content count
        n.em  = kwargs.get('em',0); n.ed = kwargs.get('ed',0); n.ec  = kwargs.get('ec',0)  # sum rim eTT
        n.rc  = kwargs.get('rc',1)  # redundancy to ext Gs, ave in links? separate rc for rim, or internally overlapping?
        n.dTT = kwargs.get('dTT',np.zeros((2,9)))  # sum derH-> m_,d_ [M,D,n, I,G,a, L,S,A], L: dLen, S: dSpan
        n.eTT = kwargs.get('eTT',np.zeros((2,9)))  # sum rim derH
        n.derH  = kwargs.get('derH', CdH())  # sum from clustered L_s
        n.dLay  = kwargs.get('dLay', CdH())  # sum from terminal L_, Fg only?
        n.baseT = kwargs.get('baseT',np.zeros(4))  # I,G,A: not ders, not in links?
        n.yx  = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.rng = kwargs.get('rng',1)  # or med: loop count in comp_node_|link_
        n.box = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.span = kwargs.get('span',0) # distance in nodet or aRad, comp with baseT and len(N_), not additive?
        n.angl = kwargs.get('angl',[np.zeros(2),0])  # (dy,dx),dir, sum from L_
        n.mang = kwargs.get('mang',1)  # ave match of angles in L_, =1 in links
        n.B_ = kwargs.get('B_', [])  # ext boundary | background neg Ls/lG: [B_,Et,R], add dB_?
        n.rB_= kwargs.get('rB_',[])  # reciprocal cores for lG, vs higher-der lG.B_
        n.C_ = kwargs.get('C_', [])  # int centroid Gs, add dC_?
        n.rC_= kwargs.get('rC_',[])  # reciprocal root centroids
        n.nH = kwargs.get('nH', [])  # top-down hierarchy of sub-node_s: CN(sum_N_(Nt_))/ lev, with single added-layer derH, empty nH
        n.lH = kwargs.get('lH', [])  # bottom-up hierarchy of L_ graphs: CN(sum_N_(Lt_))/ lev, within each nH lev
        n.root = kwargs.get('root',None)  # immediate
        n.sub  = 0  # full-composition depth relative to top-composition peers
        n.fin  = kwargs.get('fin',0)  # clustered, temporary
        n.exe  = kwargs.get('exe',0)  # exemplar, temporary
        n.compared = set()
        # n.fork_tree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def __bool__(n): return bool(n.N_)

