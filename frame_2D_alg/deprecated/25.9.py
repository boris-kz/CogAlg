def xcomp_C_(C_, root, rc):  # draft

    def merged(C): # get final C merge targets
        while C.fin: C = C.root
        return C
    def dC_spectrum(L_):  # PCA on dC.derTT[1]), sort L_ along the principal axis of variation (PC1).

        dert_ = np.array([dC.derTT[1] for dC in L_])  # arr(n_pairs, n_feat); pair-wise diffs per attr
        meanT = np.mean(dert_, axis=0)
        mad_T = np.mean(np.abs(dert_ - meanT), axis=0)  # per-attr mean abs deviation
        norm_ = (dert_- meanT) / (mad_T +eps)  # scale by per-attr dispersion
        cov_T = np.cov(norm_, rowvar=False)  # 9x9 covariance matrix (if 9 features)
        eig_, eig__ = np.linalg.eigh(cov_T)  # eigenvalues, eigenvectors (ascending)
        pc1_i = np.argmax(eig_)  # index of largest eigenvalue
        pc1_T = eig__[:, pc1_i]  # eigenvector corresponding to the largest eigenvalue
        proj_ = norm_ @ pc1_T  # sort by projection onto PC1
        # reorder L_/ PC1 to minimize ddC.derTT[1] between consecutive dCs:
        return [L_[i] for i in np.argsort(proj_)]

    dC_ = []
    for _C, C in combinations(C_, r=2):
        dy_dx = _C.yx-C.yx; dist = np.hypot(*dy_dx); olp = (C.olp+_C.olp) / 2
        dC_ += [comp_N(_C,C, olp, rc, lH=[], angl=dy_dx, span=dist)]
        # add comp wTT?
    L_, lH = [], []
    dC_ = sorted(dC_, key=lambda dC: dC.Et[1])  # from min D
    for i, dC in enumerate(dC_):
        if val_(dC.Et, fi=0, aw=rc+loopw) < 0:  # merge centroids, no re-comp: merged is similar
            _C,C = dC.N_; _C,C = merged(_C), merged(C)  # final merges
            if _C is C: continue  # was merged
            add_N(_C,C, fmerge=1, froot=1)  # +fin,root
            C_.remove(C)
        else:
            L_ = dC_[i:]  # distinct dCs
            if L_ and val_(np.sum([l.Et for l in L_], axis=0), fi=0, mw=(len(L_)-1)*Lw, aw=rc+loopw) > 0:
                for C in C_:
                    C.dertt = np.sum([l.derTT for l in C.rim]) / len(C.rim)  # ave
                L_ = dC_spectrum(L_)  # reorder L_/ PC1 to minimize ddC.derTT[1] between consecutive dCs
                dCt_ = []
                for _dC, dC in zip(L_,L_[1:]):  # comp consecutive dCs in D spectrum, not spatial
                    ddC = comp_N(_dC,dC, (dC.olp+_dC.olp)/2, rc)  # += ddC in dC.rim
                    dCt_ += [(_dC,ddC)]
                dCt_ += [(dC,None)]  # no ddC in last dCt
                seg_, seg = [], []
                for (_dC,_ddC), (dC,ddC) in zip(dCt_, dCt_[1:]):
                    if _ddC.Et[0] > ave: seg += [dC]  # pre-cluster, ddCs in rims
                    else:                seg_ += [seg]; seg = [dC]  # term old, init new G
                dCG_ = [sum_N_(dC_) for dC_ in seg_+[seg]]  # last
                lH = [sum_N_(dCG_)]  # single-level lH
            break
    root.cent_ = CN(N_=list({merged(C) for C in C_}), L_=L_, lH=lH)  # add Et, + mat?

def xcomp_C(C_, root, rc, first=1):  # draft

    def merged(C):  # get final C merge targets
        while C.fin: C = C.root
        return C
    dC_ = []
    for _C, C in combinations(C_, r=2):
        if first:
            dy_dx = _C.yx-C.yx; dist = np.hypot(*dy_dx); olp = (C.olp+_C.olp) / 2
            dC_ += [comp_N(_C,C, olp, rc, lH=[], angl=dy_dx, span=dist)]
            # add comp wTT?
        else:  # recursion: C_ = L_
            m_,d_ = comp_derT(_C.derTT[1], C.derTT[1])
            ad_ = np.abs(d_); t_ = m_+ ad_+ eps  # = max comparand
            Et = np.array([m_/t_ @ wTTf[0], ad_/t_ @ wTTf[1], min(_C.Et[2],C.Et[2])])
            dC_ += [CN(N_= [_C,C], Et=Et)]
    L_ = []
    dC_ = sorted(dC_, key=lambda dC: dC.Et[1])  # from min D
    for i, dC in enumerate(dC_):
        if val_(dC.Et, fi=0, aw=rc+loopw) < 0:  # merge centroids, no re-comp: merged is similar
            _C,C = dC.N_; _C,C = merged(_C), merged(C)  # final merges
            if _C is C: continue  # was merged
            add_N(_C,C, fmerge=1, froot=1)  # +fin,root
            C_.remove(C)
        elif first:  # for dCs, no recursion for ddCs
            L_ = dC_[i:]  # distinct dCs
            if L_ and val_(np.sum([l.Et for l in L_], axis=0), fi=0, mw=(len(L_)-1)*Lw, aw=rc+loopw) > 0:
                xcomp_C_(L_, root, rc+1, first=0)  # merge dCs in L_, no ddC_
                L_ = list({merged(L) for L in L_})
            break
    if first:
        root.cent_ = CN(N_=list({merged(C) for C in C_}), L_= L_)  # add Et, + mat?

def comp_derT(_i_,i_):
    m_ = np.minimum(np.abs(_i_), np.abs(i_))  # native vals, probably need to be signed as before?
    d_ = _i_-i_  # for next comp, from signed _i_,i_
    return np.array([m_,d_])

def comp(_pars, pars, meA=0, deA=0):  # raw inputs or derivatives, norm to 0:1 in eval only

    m_,d_ = [],[]
    for _p, p in zip(_pars, pars):
        if isinstance(_p, np.ndarray):
            mA, dA = comp_A(_p, p)
            m_ += [mA]; d_ += [dA]
        elif isinstance(p, tuple):  # massless I|S avd in p only
            p, avd = p
            m_ += [avd]  # placeholder for avd / (avd+ad), no separate match, or keep signed?
            d_ += [_p - p]
        else:  # massive
            m_ += [min(abs(_p),abs(p))]
            d_ += [_p - p]
    # add ext A:
    return np.array(m_+[meA]), np.array(d_+[deA])

def sum_H_(Q):  # sum derH in link_|node_, not used
    H = [lay.copy_ for lay in Q[0]]
    for h in Q[1:]: add_H(H,h)
    return H

def cluster_N(root, iN_, rN_, rc, rng=1):  # flood-fill node | link clusters

    def rroot(n):
        if n.root and n.root.rng > n.rng: return rroot(n.root) if n.root.root else n.root
        else:                             return None

    G_ = []  # exclusive per fork,ave, only centroids can be fuzzy
    for n in rN_: n.fin = 0
    for N in rN_:  # init
        if not N.exe or N.fin: continue  # exemplars or all N_
        N.fin = 1; in_,N_,link_,long_ = [],[],[],[]
        if rng==1 or (not N.root or N.root.rng==1):  # not rng-banded, L.root is empty
            cent_ = N.C_  # cross_comp C_?
            for l in N.rim:
                in_ += [l]
                if l.rng==rng and val_(l.Et+ett(l), aw=rc+1) > 0:
                    link_+=[l]; N_ += [N]  # l.Et potentiated by density term
                elif l.rng>rng: long_+=[l]
        else: # N is rng-banded, cluster top-rng roots
            n = N; R = rroot(n)
            if R and not R.fin:
                N_,link_,long_ = [R], R.L_, R.hL_; R.fin = 1; in_+=R.link_; cent_ = R.C_
        in_ = set(link_)  # all visited
        L_ = []
        while link_:  # extend clustered N_ and L_
            L_+=link_; _link_=link_; link_=[]; in_=[]  # global if N-parallel
            for L in _link_:  # buffer
                for _N in L.N_:
                    if _N not in iN_ or _N.fin: continue
                    if rng==1 or (not _N.root or _N.root.rng==1):  # not rng-banded
                        N_ += [_N]; cent_ += N.C_; _N.fin = 1
                        for l in N.rim:
                            if l in in_: continue
                            in_+=[l]
                            if l.rng==rng and val_(l.Et+ett(l), aw=rc) > 0: link_+=[l]
                            elif l.rng>rng: long_ += [l]
                    else:  # cluster top-rng roots
                        _n = _N; _R = rroot(_n)
                        if _R and not _R.fin:
                            in_+=_R.link_
                            if rolp(N, link_, R=1) > ave*rc:
                                N_+=[_R]; _R.fin=1; _N.fin=1; link_+=_R.link_; long_+=_R.hL_; cent_ += _R.C_
            link_ = list(set(link_)-in_)
            in_.update(set(in_))
        if N_:
            N_, long_ = list(set(N_)), list(set(long_))
            Et, olp = np.zeros(3), 0
            for n in N_: olp += n.olp  # from Ns, vs. Et from Ls?
            for l in L_: Et += l.Et
            if val_(Et,1, (len(N_)-1)*Lw, rc+olp, root.Et) > 0:
                G_ += [sum2graph(root, N_,L_,long_, set(cent_), Et,olp, rng)]
            else:
                G_ += N_
    if G_: return sum_N_(G_,root)  # nG

def xcomp_C(C_, root, rc, first=1):  # draft

    def merged(C):  # get final C merge targets
        while C.fin: C = C.root
        return C
    dC_ = []
    for _C, C in combinations(C_, r=2):
        if first:  # mfork
            dy_dx = _C.yx-C.yx; dist = np.hypot(*dy_dx); olp = (C.olp+_C.olp) / 2
            dC_ += [comp_N(_C,C, olp, rc, lH=[], angl=dy_dx, span=dist)]
            # add comp wTT?
        else:   # dfork recursion: C_ = L_
            m_,d_ = comp_derT(_C.derTT[1], C.derTT[1])
            ad_ = np.abs(d_); t_ = m_+ ad_+ eps  # = max comparand
            Et = np.array([m_/t_ @ wTTf[0], ad_/t_ @ wTTf[1], min(_C.Et[2],C.Et[2])])  # signed M?
            dC = CN(N_= [_C,C], Et=Et); _C.rim += [dC]; C.rim += [dC]; dC_ += [dC]
    # merge or cluster Cs:
    L_, cG, lH = [],[],[]
    dC_ = sorted(dC_, key=lambda dC: dC.Et[1])  # from min D
    for i, dC in enumerate(dC_):
        if val_(dC.Et, fi=0, aw=rc+loopw) < 0:  # merge centroids, no re-comp: merged is similar
            _C,C = dC.N_; _C,C = merged(_C), merged(C)  # final merges
            if _C is C: continue  # was merged
            add_N(_C,C, fmerge=1, froot=1); C_.remove(C)  # +fin,root
        else:
            L_ = dC_[i:]; C_ = list({merged(c) for c in C_})  # remaining Cs and dCs between them
            if L_:
                # ~ cross_comp, reconcile? no recursive cluster_C or agg+?
                mV,dV = val_(np.sum([l.Et for l in L_], axis=0), fi=2, mw=(len(L_)-1)*Lw, aw=rc+loopw)
                if first and dV > 0:
                    lH = [xcomp_C(L_, root, rc+1, first=0)]  # merge dCs in L_, no ddC_, lH = [lG]?
                if mV > ave * contw:
                    cG = cluster_n(root, C_,rc)  # by connectivity between Cs in feature space
            break
    if first:
        return cG or CN(N_= C_, L_= L_, lH = lH)  # add Et, + mat?

    def merged(C):  # get final C merge targets
        while C.fin: C = C.root
        return C

    def base_lev(y,x):

        PV__ = np.zeros(Ly,Lx)  # maps to level frame
        Fg_ = []  # processed lower frames
        while True:
            Fg = base_win(y,x)
            Fg_ += [Fg]
            if val_(Fg.Et, 1, (len(Fg.N_)-1)*Lw, Fg.rc+loopw+20):
                # proj, extend lev frame laterally:
                pFg = project_N_(Fg, np.array([y,x]))
                if pFg:
                    pFg = cross_comp(pFg, rc=Fg.rc) or pFg
                    if val_(pFg.Et, 1, (len(pFg.N_)-1)*Lw, pFg.rc+contw+20):
                        project_focus(PV__, y, x, Fg)  # += proj val in PV__
                        y, x = np.unravel_index(PV__.argmax(), PV__.shape)
                        if PV__[y,x] > ave: y = y*Ly; x=x*Lx
                        else: break
                    else: break
                else: break
            else: break
            if Fg_:
                N_,C_,L_ = [],[],[]
                for Fg in Fg_:
                    N_ += Fg.N_; C_ += Fg.C_; L_ += Fg.L_
                return N_,C_,L_


