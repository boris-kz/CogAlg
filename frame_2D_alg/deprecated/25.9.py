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

