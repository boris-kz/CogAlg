def cross_comp(root, rc, fcon=1, fL=0):  # core function, mediates rng+ and der+ cross-comp and clustering, rc = rdn + olp

    iN_,L_,TT,c,TTd,cd = comp_N_((root.N_,root.B_)[fL], rc) if fcon else comp_C_(root.N_,rc); nG_=[]  # nodes| centroids
    if L_:
        cr = cd / (c+cd) *.5  # dfork borrow ratio, .5 for one direction
        if val_(TT, rc+compw, TTw(root), (len(L_)-1)*Lw,1,TTd,cr) > 0 or fL:  # default for links
            root.L_=L_; sum2T(L_,rc,root,'Lt')  # new ders, no Lt.N_, root.B_,Bt if G
            for n in iN_: n.em = sum([l.m for l in n.rim]) / len(n.rim)  # pre-val_
            nG_,rc = Cluster(root, L_,rc,fcon,fL)  # Bt in cluster_N, sub+ in sum2G
        # agg+:
        if nG_ and val_(TT, rc+connw, TTw(root), (len(nG_)-1)*Lw) > 0:  # mval only
            nG_,rc = trace_edge(root.N_,rc,root)  # comp Ns x N.Bt|B_.nt, with/out mfork?
        if nG_ and val_(root.dTT, rc+compw, TTw(root), (len(root.N_)-1)*Lw,1, TTd,cr) > 0:
            nG_,rc = cross_comp(root, rc)

    return nG_,rc  # nG_: recursion flag


def Cluster(root, iL_, rc, fcon=1, fL=0):  # generic clustering root

    def trans_cluster(root, iL_, rc):  # called from cross_comp(Fg_), others?

        dN_,dB_,dC_ = [],[],[]  # splice specs from links between Fgs in Fg cluster
        for Link in iL_:
            dN_+= Link.N_; dB_+= Link.B_; dC_+= Link.C_
        for tL_,nf_,nft,fC in (dN_,'tN_','tNt',0), (dB_,'tB_','tBt',0), (dC_,'tC_','tCt',1):
            if tL_:
                Ft = sum2T(tL_,rc, root,nft); N_ = list({n for L in tL_ for n in L.nt})
                for N in N_: N.exe=1
                cluster_N(Ft, N_,rc)  # default fork redundancy
                if val_(Ft.dTT, rc, TTw(root), (len(Ft.N_)-1)*Lw) > 0:
                    cross_comp(Ft, rc)  # unlikely, doesn't add rc?
                setattr(root,nf_, tL_)
                rc += 1
    def get_exemplars(N_, rc):  # multi-layer non-maximum suppression -> sparse seeds for diffusive centroid clustering
        E_ = set()
        for rdn, N in enumerate(sorted(N_, key=lambda n:n.em, reverse=True), start=1):  # strong-first
            oL_ = set(N.rim) & {l for e in E_ for l in e.rim}
            roV = vt_(sum([l.dTT for l in oL_]), rc)[0] if oL_ else 0 / (N.em or eps)  # relative rim olp V
            if N.em * N.c > ave * (rc+rdn+compw+ roV):  # ave *= rV of overlap by stronger-E inhibition zones
                E_.update({n for l in N.rim for n in l.nt if n is not N and N.em > ave*rc})  # selective nrim
                N.exe = 1  # in point cloud of focal nodes
            else: break  # the rest of N_ is weaker, trace via rims
        return E_
    G_ = []
    if fcon:  # connect-cluster, no exemplars or recursive centroid clustering
        N_,L_,i = [],[],0  # add pL_, pN_ for pre-links?
        if fcon > 1:  # merge similar Cs, not dCs, no recomp
            link_ = sorted(list({L for L in iL_}), key=lambda link: link.d)  # from min D
            for i, L in enumerate(link_):
                if val_(L.dTT, rc+compw, wTTf,fi=0) < 0:
                    _N, N = L.nt; root.L_.remove(L)  # merge nt, remove link
                    if _N is not N:  # not yet merged
                        for n in N.N_: add_N(_N, n); _N.N_ += [n]
                        for l in N.rim: l.nt = [_N if n is N else n for n in l.nt]
                        if N in N_: N_.remove(N)
                        N_ += [_N]
                else: L_ = link_[i:]; break
        else: L_ = iL_
        N_ = list(set([n for L in L_ for n in L.nt] + N_))  # include merged Cs
        if N_ and (val_(root.dTT, rc+connw, TTw(root), mw=(len(N_)-1)*Lw) > 0 or fcon>1):
            G_,rc = cluster_N(root, N_,rc, fL)  # contig in feature space if centroids, no B_,C_?
            if G_:  # top root.N_
                tL_ = [tl for n in root.N_ for l in n.L_ for tl in l.N_]  # trans-links
                if sum(tL.m for tL in tL_) * ((len(tL_)-1)*Lw) > ave*(rc+connw):  # use tL.dTT?
                    trans_cluster(root, tL_, rc+1)  # sets tTs
                    mmax_ = []
                    for F,tF in zip((root.Nt,root.Bt,root.Ct), (root.tNt,root.tBt,root.tCt)):
                        if F and tF:
                            maxF,minF = (F,tF) if F.m>tF.m else (tF,F)
                            mmax_+= [max(F.m,tF.m)]; minF.rc+=1  # rc+=rdn
                    sm_ = sorted(mmax_, reverse=True)
                    for m, (Ft, tFt) in zip(mmax_,((root.Nt, root.tNt),(root.Bt, root.tBt),(root.Ct, root.tCt))): # +rdn in 3 fork pairs
                        r = sm_.index(m); Ft.rc+=r; tFt.rc+=r  # rc+=rdn
    else:
        # primary centroid clustering:
        N_ = list({N for L in iL_ for N in L.nt if N.em})  # newly connected only
        E_ = get_exemplars(N_,rc)
        if E_ and val_(np.sum([g.dTT for g in E_],axis=0), rc+centw, TTw(root), (len(E_)-1)*Lw) > 0:
            G_,rc = cluster_C(E_,root,rc)  # forms root.Ct, may call cross_comp-> cluster_N, incr rc
    return  G_,rc

