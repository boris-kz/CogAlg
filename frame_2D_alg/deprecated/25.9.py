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

def cluster_C(E_, root, rc):  # form centroids by clustering exemplar surround via rims of new member nodes, within root

    _C_, _N_, cG = [],[],[]
    for E in E_:
        C = cent_attr( Copy_(E,root, init=2), rc); C.N_ = [E]   # all rims are within root
        C._N_ = [n for l in E.rim for n in l.N_ if n is not E]  # core members + surround -> comp_C
        _N_ += C._N_; _C_ += [C]
    # reset:
    for n in set(root.N_+_N_ ): n.rC_,n.mo_, n._C_,n._mo_ = [],[], [],[]  # aligned pairs, in cross_comp root
    # reform C_, refine C.N_s
    while True:
        C_,cnt,olp, mat,dif, Dm,Do = [],0,0,0,0,0,0; Ave = ave * (rc+loopw)
        _Ct_ = [[c, c.Et[0]/c.Et[2] if c.Et[0] !=0 else eps, c.rc] for c in _C_]
        for _C,_m,_o in sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True):
            if _m > Ave * _o:
                C = cent_attr( sum_N_(_C.N_, root=root, fC=1), rc); C.rC_ = []  # C update lags behind N_; non-local C.rc += N.mo_ os?
                _N_,_N__, mo_, M,D,O,comp, dm,do = [],[],[],0,0,0,0,0,0  # per C
                for n in _C._N_:  # core+ surround
                    if C in n.rC_: continue
                    _,(m,d,_),_ = min_comp(C,n)  # val,olp / C:
                    o = np.sum([mo[0] /m for mo in n._mo_ if mo[0]>m])  # overlap = higher-C inclusion vals / current C val
                    comp += 1  # comps per C
                    if m > Ave * o:
                        _N_+=[n]; M+=m; O+=o; mo_ += [np.array([m,o])]  # n.o for convergence eval
                        _N__ += [_n for l in n.rim for _n in l.N_ if _n is not n]  # +|-Ls
                        if _C not in n._C_: dm+=m; do+=o  # not in extended _N__
                    else:
                        if _C in n._C_: __m,__o = n._mo_[n._C_.index(_C)]; dm+=__m; do+=__o
                        D += abs(d)  # distinctive from excluded nodes (silhouette)
                mat+=M; dif+=D; olp+=O; cnt+=comp  # from all comps?
                if M > Ave * len(_N_) * O:
                    for n, mo in zip(_N_,mo_): n.mo_+=[mo]; n.rC_+=[C]; C.rC_+=[n]  # bilateral assign
                    C.ET += np.array([M,D,comp])  # C.Et is a comparand
                    C.N_ += _N_; C._N_ = list(set(_N__))  # core, surround elements
                    C_ += [C]; Dm+=dm; Do+=do  # new incl or excl
                else:
                    for n in _C._N_:
                        n.exe = n.Et[0]/n.Et[2] > 2 * ave  # refine exe, Et vals are already normalized, Et[2] no longer needed for eval?
                        for i, c in enumerate(n.rC_):
                            if c is _C: # remove mo mapping to culled _C
                                n.mo_.pop(i); n.rC_.pop(i); break
            else: break  # the rest is weaker
        if Dm > Ave * cnt * Do:  # dval vs. dolp, overlap increases as Cs may expand in each loop?
            _C_ = C_
            for n in root.N_: n._C_=n.rC_; n._mo_=n.mo_; n.rC_,n.mo_ = [],[]  # new n.rC_s, combine with vo_ in Ct_?
        else:  # converged
            Et = np.array([mat,dif,cnt])  # incl M, excl D, all comped | separate ns?
            break
    if C_:
        for n in [N for C in C_ for N in C.N_]:
            # exemplar V increased by summed n match to root C * rel C Match:
            n.exe = n.et[1-n.fi]+ np.sum([n.mo_[i][0] * (C.M / (ave*n.mo_[i][0])) for i, C in enumerate(n.rC_)]) > ave

        if val_(Et,1,(len(C_)-1)*Lw, rc+olp, root.Et) > 0:
            cG = cross_comp(sum_N_(C_), rc, fC=1)  # distant Cs or with different attrs
        root.C_ = [cG.N_ if cG else C_, Et]

def ffeedback(root):  # adjust filters: all aves *= rV, ultimately differential backprop per ave?

    wTTf = np.ones((2,9))  # sum derTT coefs: m_,d_ [M,D,n, I,G,A, L,S,eA]: Et, baseT, extT
    rM, rD, rVd = 1, 1, 0
    hLt = sum_N_(root.L_)  # links between top nodes
    _derTT = np.sum([l.derTT for l in hLt.N_])  # _derTT[np.where(derTT==0)] = eps
    for lev in reversed(root.nH):  # top-down
        if not lev.lH: continue
        Lt = lev.lH[-1]  # dfork
        _m, _d, _n = hLt.Et; m, d, n = Lt.Et
        rM += (_m / _n) / (m / n)  # relative higher val, | simple relative val?
        rD += (_d / _n) / (d / n)
        derTT = np.sum([l.derTT for l in Lt.N_])  # top link_ is all comp results
        wTTf += np.abs((_derTT / _n) / (derTT / n))
        if Lt.lH:  # ddfork only, not recursive?
            # intra-level recursion in dfork
            rVd, wTTfd = ffeedback(Lt)
            wTTf = wTTf + wTTfd

    return rM+rD+rVd, wTTf

class CLay(CBase):  # layer of derivation hierarchy, subset of CG
    name = "lay"
    def __init__(l, **kwargs):
        super().__init__()
        l.Et = kwargs.get('Et', np.zeros(2))
        l.rc = kwargs.get('rc', 1)  # ave nodet overlap
        l.node_ = kwargs.get('node_', [])  # concat across fork tree
        l.link_ = kwargs.get('link_', [])
        l.derTT = kwargs.get('derTT', np.zeros((2,9)))  # sum m_,d_ [M,D,n, I,G,A, L,S,eA] across fork tree
        # i: lay index in root node_,link_, to revise olp; i_: m,d priority indices in comp node|link H
        # ni: exemplar in node_, trace in both directions?
    def __bool__(l): return bool(l.node_)

    def copy_(lay, rev=0, i=None):  # comp direction may be reversed to -1, currently not used
        if i:  # reuse self
            C = lay; lay = i; C.node_=copy(i.node_); C.link_ = copy(i.link_); C.derTT=np.zeros((2,9))
        else:  # init new C
            C = CLay(node_=copy(lay.node_), link_=copy(lay.link_))
        C.Et = copy(lay.Et)
        for fd, tt in enumerate(lay.derTT):  # nested array tuples
            C.derTT[fd] += tt * -1 if (rev and fd) else deepcopy(tt)
        if not i: return C

    def add_lay(Lay, lay, rev=0, rn=1):  # no rev, merge lays, including mlay + dlay

        # rev = dir==-1, to sum/subtract numericals in m_,d_
        for fd, Fork_, fork_ in zip((0,1), Lay.derTT, lay.derTT):
            Fork_ += (fork_ * -1 if (rev and fd) else fork_) * rn  # m_| d_| t_
        # concat node_,link_:
        Lay.node_ += [n for n in lay.node_ if n not in Lay.node_]
        Lay.link_ += lay.link_
        Lay.Et += lay.Et * rn
        Lay.rc = (Lay.rc + lay.rc * rn) /2
        return Lay

    def comp_lay(_lay, lay, rn, root):  # unpack derH trees down to numericals and compare them

        derTT = comp_derT(_lay.derTT[1], lay.derTT[1] * rn)  # ext A align replaced dir/rev
        Et = np.array([derTT[0] @ wTTf[0], np.abs(derTT[1]) @ wTTf[1]])
        if root: root.Et[:2] += Et  # no separate n
        node_ = list(set(_lay.node_+ lay.node_))  # concat, or redundant to nodet?
        link_ = _lay.link_ + lay.link_
        return CLay(Et=Et, rc=(_lay.rc+lay.rc*rn)/2, node_=node_, link_=link_, derTT=derTT)

def add_H(H, h, root=0, root_derH_=[], rn=1):  # layer-wise add|append derH

    for Lay, lay in zip_longest(H,h):  # different len if lay-selective comp
        if lay:
            if Lay: add_lay(Lay, lay, rn)
            else:   H += copy_(lay)  # * rn?
            if root: root.derTT += lay.derTT*rn; root.Et[:2] += lay.Et*rn
            if root_derH_:
                DerH, derH = root_derH_
                DerH.Et += derH.Et * rn
                DerH.derTT += derH.derTT * rn
    return H

def comp_H(H,h, rn, ET=None, DerTT=None, root=None):  # one-fork derH

    derH, derTT, Et = [], np.zeros((2,9)), np.zeros(3)
    for _lay, lay in zip_longest(H,h):  # selective
        if _lay and lay:
            dlay = comp_lay(_lay, lay, rn, root=root)
            derH += [dlay]; derTT = dlay.derTT; Et[:2] += dlay.Et
    if Et[2]: DerTT += derTT; ET += Et
    return derH

def cross_comp(root, rc, fC=0):  # rng+ and der+ cross-comp and clustering

    if fC: N__, L_, Et = comp_sorted(root.N_,rc)
    else:  N__, L_, Et = comp_Q(root.N_, rc, fC)  # rc: redundancy+olp, lG.N_ is Ls

    # N__ is flat list if fC=1, list of bands if fC=0
    if L_ and len(L_) > 1:
        mV, dV = val_(Et, 2, (len(L_)-1) *Lw, rc+compw); lG = []
        if dV > 0:
            if root.fi and root.L_: root.lH += [sum_N_(root.L_)]
            root.L_ = L_; root.Et += Et
            if fC < 2 and dV > avd:  # dfork, no comp ddC_
                lG = cross_comp(CN(N_=L_), rc+compw+1, fC*2)  # batched lH extension
                if lG: rc+=lG.rc; root.lH+=[lG]+lG.nH; root.Et+=lG.Et; add_dH(root.derH, lG.derH)  # new lays
        if mV > 0:
            nG = Cluster(root, N__, rc, fC)  # get_exemplars, cluster_C, rng connectivity cluster
            if nG:  # batched nH extension
                rc += nG.rc  # redundant clustering layers
                if lG:
                    comb_B_(nG, lG, rc+2); Et += lG.Et  # assign boundary, in feature spectrum if fC
                if val_(nG.Et, 1, (len(nG.N_)-1)*Lw, rc+compw+2, Et) > 0:
                    nG = cross_comp(nG, rc+2, fC) or nG
                root.N_ = nG.N_
                _H = root.nH; root.nH = []  # nG has own L_,lH
                nG.nH = _H + [root] + nG.nH  # pack root.nH in higher-composition nG.nH
                return nG  # update root

def add_sett(Sett,sett):
    if Sett: N_,Et = Sett; n_ = sett[0]; N_.update(n_); Et += np.sum([t.Et for t in n_-N_])
    else:    Sett += [copy(par) for par in sett]  # B_, Et

def comp_seq(N_, rc):  # comp consecutive nodes along each direction in the edge

    def cluster_dir(N):
        rim = []
        for l in N._rim:
            if l.span < adist: l.L_ += [l]; l.L_ += [l]  # use L_ as lGt
        for _l,l in combinations(rim,r=2):
            if _l in l.compared: continue
            l.compared.add(l[-2]); _l.compared.add(l)
            mA,_ = comp_A(_l.angl,l.angl)
            if mA > 0.5:  # placeholer ama
                for pre_link in l.L_.pop():  # lGt
                    l.fin=1; _l.L_ += [l]
        N._rim = [L.L_ for L in rim if L.L_]
        # directions
    Et = np.zeros(3)
    for N in N_: N._rim = []
    for _N, N in combinations(N_,r=2):  # get proximity only
        if _N in N.compared: continue
        N.compared.add(_N); _N.compared.add(N)
        dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx)
        L = CN(fi=0, span=dist, angl=dy_dx, N_=[_N,N]); N._rim += [L]; _N._rim += [L]
    G_ = []
    for N in N_: N.compared=set(); cluster_dir(N)
    for N in N_:
        for dir in N._rim:  # _pre-link directions clustered by mA
            Li = np.argmin([l.span for l in dir])
            pL = dir[Li]; _n,n = pL.N_
            if n in _n.compared: continue
            o = n.rc+_n.rc  # V = proj_V(_n,n, dy_dx, dist)  # eval _N,N cross-induction for comp
            if adist * (val_(n.Et+_n.Et, aw=o) / ave*rc) > pL.span:  # min induction
                Link = comp_N(_n,n, o,rc, angl=pL.angl, span=pL.span)
                if val_(Link.Et, aw=compw+o+rc) > 0:
                    Et += Link.Et
        N.Gt = [N]
    for N in N_:
        if N.fin: continue
        for L in N.rim:  # one min dist L per direction
            if val_(L.Et, aw = contw+rc):
                nt=L.N_; _N = nt[0] if nt[0] is N else nt[1]
                for n in N.Gt.pop():
                    n.fin=1; _N.Gt += [n]
                break
    for N in N_:
        if N.Gt: G = sum_N_(N.Gt); G.root=N.root; N.root=G; G_+=[G]
        elif not N.fin: N.sub+=1; G_ += [N]  # singleton
        delattr(N, 'Gt')
    return G_
'''
if B_:
    bG = CN(N_=B_, Et=Et)
    if val_(Et, 0, (len(B_) - 1) * Lw, rc + Rdn + compw) > 0:  # norm by core_ rdn
        bG = cross_comp(bG, rc) or bG
    Ng.B_ = [list(set(bG.N_)), bG.Et]
'''
