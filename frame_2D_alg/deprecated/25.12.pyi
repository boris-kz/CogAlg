def vect_edge(tile, rV=1, wTTf=[]):  # PP_ cross_comp and floodfill to init focal frame graph, no recursion:

    if np.any(wTTf):
        global ave, avd, arn, aveB, aveR, Lw, adist, amed, intw, compw, centw, contw, wM,wD,wc, wI,wG,wa, wL,wS,wA
        ave, avd, arn, aveB, aveR, Lw, adist, amed, intw, compw, centw, contw = (
            np.array([ave,avd, arn,aveB,aveR, Lw, adist, amed, intw, compw, centw, contw]) / rV)  # projected value change
        wTTf = np.multiply([[wM,wD,wc, wI,wG,wa, wL,wS,wA]], wTTf)  # or dw_ ~= w_/ 2?
    Edge_ = []
    for blob in tile.N_:
        if not blob.sign and blob.G > aveB:
            edge = slice_edge(blob, rV)
            if edge.G * ((len(edge.P_)-1)*Lw) > ave * sum([P.latT[4] for P in edge.P_]):
                PPm_ = comp_slice(edge, rV, wTTf)
                Edge = sum2G([PP2N(PPm) for PPm in PPm_], rc=1, root=None)
                if edge.link_:
                    L_,B_ = [],[]; Gd_ = [PP2N(PPd)for PPd in edge.link_]
                    [L_.append(Gd) if Gd.m > ave else B_.append(Gd) if Gd.d > avd else None for Gd in Gd_]
                    Lt = add_T_(B_,rc=2,root=Edge, nF='Lt'); Edge.L_=L_;
                    Bt = add_T_(B_,rc=2,root=Edge, nF='Bt'); Edge.B_=B_
                    form_B__(Edge)  # add Edge.Bt
                    if val_(Edge.dTT,3, mw=(len(PPm_)-1) *Lw) > 0:
                        trace_edge(Edge,3)  # cluster complemented G x G.B_, ?Edge.N_=G_, skip up
                        if B_ and val_(Bt.dTT,4, fi=0, mw=(len(B_)-1) *Lw) > 0:
                            Bt.N_ = B_; trace_edge(Bt,4) # ?Bt.N_= bG_
                            # for cross_comp in frame_H?
                Edge_ += [Edge]  # default?
    if Edge_:
        return sum2G(Edge_,2,None)

N_, mL_,mTT,mc, dL_,dTT,dc = comp_C_(root.N_,rc) if fC else comp_N_(root.N_,rc); nG_,up = [],0
# m fork:
if len(mL_)>1 and val_(mTT, rc+compw, mw=(len(mL_)-1)*Lw) > 0:
    root.L_= mL_; add_T_(mL_,rc,root,'Lt')  # new ders, no Lt.N_
    for n in N_: n.em = sum([l.m for l in n.rim]) / len(n.rim)  # pre-val_
    nG_ = Cluster(root, mL_, rc,fC)  # fC=0: get_exemplars, cluster_C, rng cluster
    if nG_:  # not N_
        rc+=1; root.N_=nG_; add_T_(nG_,rc,root,'Nt')
        if isinstance(root,CF):  # Bt | Ct
            for n in N_: (n.rC_ if fC else n.rB_).append(*nG_)

def add_T(Ft,ft, root,fH):
    if fH:  # Nt
        for Lev,lev in zip_longest(Ft.N_, ft.N_):
            if lev:  # norm /C?
                if Lev: Lev.N_+= lev.N_; Lev.dTT+=lev.dTT; Lev.c+=lev.c  # flat
                else:   root.Nt.N_ += [lev]
    else:  # alt fork roots:
        for n in ft.N_:
            if n not in Ft.N_: n.root=Ft; Ft.N_ += [n]
    Ft.dTT += ft.dTT

def add_N_(N_, rc, root, TT=None, c=1, flat=0):  # forms G of N_|L_

    N = N_[0]; fTT= TT is not None
    G = Copy_(N, root, init=1, typ=2)
    if fTT: G.dTT= TT; G.c= c
    n_ = list(N.N_)  # flatten core fork, alt forks stay nested
    for N in N_[1:]:
        add_N(G, N, fTT); n_ += N.N_
    if flat: G.N_ = n_  # flatten N.N_, or in F.N_ only?
    else:
        G.N_ = N_
        if n_ and N.typ: G.Nt.N_.insert(0, CF(N_=n_,root=G.Nt))  # not PP.P_, + Lt.dTT
    G.m, G.d = vt_(G.dTT)
    G.rc = rc
    return G

def cluster_n(root, N_, rc):  # simplified flood-fill, for C_ or trans_N_

    def extend_G(_link_, node_,cent_,link_,b_,in_):
        for L in _link_:  # spliced rim
            if L in in_: continue  # already clustered
            in_.add(L)
            for _N in L.nt:
                if not _N.fin and _N in N_:
                    node_ += [_N]; cent_ += _N.C_; _N.fin = 1
                    for l in _N.rim:
                        if l in in_: continue  # cluster by link+density:
                        if Lnt(l) > ave*rc: link_ += [l]
                        else: b_ += [l]
    # root attrs:
    G_,N__,L__,Lt_,TT,lTT,C,lC, in_ = [],[],[],[],np.zeros((2,9)),np.zeros((2,9)),0,0, set()
    for N in N_: N.fin = 0
    for N in N_:  # form G / remaining N
        node_,link_,L_,B_ = [N],[],[],[]
        cent_ = N.C_
        _L_= [l for l in N.rim if Lnt(l) > ave*rc]  # link+nt eval
        while _L_:
            extend_G(_L_, node_, cent_, L_, B_, in_)  # _link_: select rims to extend G:
            if L_: link_ += L_; _L_ = list(set(L_)); L_ = []
            else:  break
        if node_:
            N_= list(set(node_)); L_= list(set(link_)); C_ = list(set(cent_))
            tt,ltt,c,lc,o = np.zeros((2,9)),np.zeros((2,9)),0,0,0
            for n in N_: tt+=n.dTT; c+=n.c; o+=n.rc  # from Ns, vs. Et from Ls?
            for l in L_: ltt+=l.dTT; lc+=l.c; o+=l.rc  # combine?
            if val_(ltt, rc+o,_TT=root.dTT) > 0:  # include singletons?
                G_ += [sum2G(N_,o,root,L_,C_)]
                N__+= N_;L__+=L_; Lt_+=[n.Lt for n in N_]; TT+=tt; lTT+ltt; C+=c; lC+=lc
    if G_ and val_(TT, rc+1, mw=(len(G_)-1)*Lw) > 0:
        rc += 1
        root_replace(root,rc, G_,N__,L__,Lt_,TT,lTT,C,lC)
    return G_, rc

def cluster_N(root, rL_, rc, rng=1):  # flood-fill node | link clusters

    def rroot(n): return rroot(n.root) if n.root and n.root!=root else n

    def extend_Gt(_link_, node_, cent_, link_, b_, in_):
        for L in _link_:  # spliced rim
            if L in in_: continue  # already clustered
            in_.add(L)
            for _N in L.nt:
                if _N.fin: continue
                if not _N.root or _N.root==root or not _N.L_:  # not rng-banded
                    node_+=[_N]; cent_+=_N.Ct.N_; _N.fin = 1
                    for l in _N.rim:
                        if l in in_: continue  # cluster by link+density:
                        if l in rL_:
                            if Lnt(l) > ave*rc: link_ += [l]
                            else: b_ += [l]
                else:  # cluster top-rng roots
                    _n = _N; _R = rroot(_n)
                    if _R and not _R.fin:
                        if rolp(N, link_, R=1) > ave * rc:
                            node_ += [_R]; _R.fin = 1; _N.fin = 1
                            link_ += _R.L_; cent_ += _R.Ct.N_
    # root attrs:
    G_,N__,L__,Lt_,TT,lTT,C,lC, in_ = [],[],[],[],np.zeros((2,9)),np.zeros((2,9)),0,0, set()
    rN_ = {N for L in rL_ for N in L.nt}
    for n in rN_: n.fin = 0
    for N in rN_:  # form G per remaining rng N
        if N.fin or (root.root and not N.exe): continue  # no exemplars in Fg
        node_,cent_,Link_,_link_,B_ = [N],[],[],[],[]
        if rng==1 or not N.root or N.root==root:  # not rng-banded
            cent_ = [C.root for C in N.C_]
            for l in N.rim:
                if l in rL_:  # curr rng
                    if Lnt(l) > ave*rc: _link_ += [l]
                    else: B_ += [l]  # or dval?
        else:  # N is rng-banded, cluster top-rng roots
            n = N; R = rroot(n)
            if R and not R.fin: node_,_link_,cent_ = [R], R.L_[:], [C.root for C in R.C_]; R.fin = 1
        N.fin = 1; link_ = []
        while _link_:
            Link_ += _link_
            extend_Gt(_link_, node_, cent_, link_, B_, in_)
            if link_: _link_ = list(set(link_)); link_ = []  # extended rim
            else: break
        if node_:
            N_,L_,C_,B_ = list(set(node_)),list(set(Link_)),list(set(cent_)),list(set(B_))
            tt,ltt,c,lc,o = np.zeros((2,9)),np.zeros((2,9)),0,0,0
            for n in N_: tt+=n.dTT; c+=n.c; o+=n.rc  # from Ns, vs. Et from Ls?
            for l in L_: ltt+=l.dTT; lc+=l.c; o+=l.rc  # combine?
            if val_(lTT, rc+o,_TT=root.dTT) > 0:  # include singletons?
                G_ += [sum2G(N_,o,root, L_,C_)]
                N__+=N_;L__+=L_; Lt_+=[n.Lt for n in N_]; TT+=tt; lTT+ltt; C+=c; lC+=lc
    if G_ and val_(TT, rc+1, mw=(len(G_)-1)*Lw) > 0:
        rc += 1
        root_replace(root,rc, G_,N__,L__,Lt_,TT,lTT,C,lC)
    return G_, rc

def Cluster(root, iL_, rc, fC):  # generic clustering root

    def trans_cluster(root, iL_,rc):  # called from cross_comp(Fg_), others?

        dN_,dB_,dC_ = [],[],[]  # splice specs from links between Fgs in Fg cluster
        for Link in iL_: dN_+= Link.N_; dB_+= Link.B_; dC_+= Link.C_
        frc = rc  # trans-G links
        for ft,f_,link_,clust, fC in [('tNt','tN_',dN_,cluster_N,0), ('tBt','tB_',dB_,cluster_N,0), ('tCt','tC_',dC_,cluster_n,1)]:
            if link_:
                frc += 1  # trans-fork redundancy count, then reassign?
                Ft = add_N_(link_,frc,root); clust(Ft, link_, frc)
                if val_(Ft.dTT, frc, mw=(len(Ft.N_)-1)*Lw) > 0:
                    cross_comp(Ft, frc, fC=fC)  # unlikely, doesn't add rc?
                setattr(root,f_,link_); setattr(root, ft, Ft)  # trans-fork_ via trans-G links
    G_ = []
    if fC or not root.root:  # base connect via cluster_n, no exemplars or centroid clustering
        N_, L_, i = [],[],0  # add pL_, pN_ for pre-links?
        if fC < 2:  # merge similar Cs, not dCs, no recomp
            link_ = sorted(list({L for L in iL_}), key=lambda link: link.d)  # from min D
            for i, L in enumerate(link_):
                if val_(L.dTT,rc+compw, fi=0) < 0:  # merge
                    _N, N = L.nt
                    if _N is not N:  # not merged
                        sum2G(_N.N_,rc, root=N, init=0)  # add_N(_N,N,froot=1, merge=1): merge Cs
                        for l in N.rim: l.nt = [_N if n is N else n for n in l.nt]
                        if N in N_: N_.remove(N)  # if multiple merging
                        N_ += [_N]
                else: L_ = link_[i:]; break
            root.L_ = [l for l in root.L_ if l not in L_]  # cleanup regardless of break
        else: L_ = iL_
        N_ += list({n for L in L_ for n in L.nt})  # include merged Cs
        if val_(root.dTT, rc+contw, mw=(len(N_)-1)*Lw) > 0:
            G_,rc = cluster_N(root, N_, rc, rng=0)  # in feature space if centroids, no B_,C_?
            if G_:  # no higher root.N_
                tL_ = [tl for n in root.N_ for l in n.L_ for tl in l.N_]  # trans-links
                if sum(tL.m for tL in tL_) * ((len(tL_)-1)*Lw) > ave*(rc+contw):  # use tL.dTT?
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
        # primary centroid clustering
        N_ = list({N for L in iL_ for N in L.nt if N.em})  # newly connected only
        E_ = get_exemplars(N_,rc)
        if E_ and val_(np.sum([g.dTT for g in E_],axis=0), rc+centw, mw=(len(E_)-1)*Lw, _TT=root.dTT) > 0:
            cluster_C(E_,root,rc)  # doesn't add rc?
        L_ = sorted(iL_, key=lambda l: l.span)
        L__, Lseg = [], [iL_[0]]
        for _L,L in zip(L_, L_[1:]):  # segment by ddist:
            if L.span -_L.span < adist: Lseg += [L]  # or short seg?
            else: L__ += [Lseg]; Lseg = [L]
        L__ += [Lseg]
        for rng, rL_ in enumerate(L__,start=1):  # bottom-up rng-banded clustering
            rc+= rng + contw
            if rL_ and sum([l.m for l in rL_]) * ((len(rL_)-1)*Lw) > ave*rc:
                G_,rc = cluster_N(root, rL_, rc, rng)  # add root_replace
    return G_,rc

def rolp(N, _N_, R=0):  # rel V of L_|N.rim overlap with _N_: inhibition|shared zone, oN_ = list(set(N.N_) & set(_N.N_)), no comp?

    n_ = {n for l in (N.L_ if R else N.rim) for n in l.nt if n is not N}  # nrim
    olp_ = n_ and _N_
    if olp_:
        oT = np.sum([o.dTT[0] for o in olp_], axis=0)
        return sum(oT[0]) / (sum(N.Lt.dTT[0] if R else N.eTT[0]))  # relative M (not rm) of overlap?
    else:
        return 0

def sum2T(T_, rc, root, nF, TT=None, c=1):  # N_ -> fork T

    T = T_[0]; fV = TT is None
    F = CF(root=root); T.root=F  # no L_,B_,C_,Nt,Bt,Ct yet
    if fV: F.dTT=T.dTT; F.c=T.c
    else:  F.dTT=TT; F.c=c
    for T in T_[1:]: add_T(F,T, nF,fV)
    F.m, F.d = vt_(F.dTT,rc)
    F.rc=rc; setattr(root, nF,F)
    root_update(root, F)
    if nF in ('Bt','Ct'): F.N_ = list(T_);  # else external N_, Nt.N_ insert, no Lt,N_
    return F

def add_T(F,T, nF, fV=1):
    if nF=='Nt' and F.N_:
        F.N_[0].N_ += T.N_  # top level = flattened T.N_s
        for Lev,lev in zip_longest(F.N_[1:], T.Nt.N_):  # deeper levels
            if lev:
                if Lev: Lev.N_+=lev.N_; Lev.dTT+=lev.dTT; Lev.c+=lev.c
                else:   F.N_ += [CopyF_(lev, root=F)]
    T.root=F
    if fV: F.dTT+=T.dTT; F.c+=T.c
