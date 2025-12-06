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
