def cross_comp(Ft, ir, nF='Nt'):  # core function mediating recursive rng+ and der+ cross-comp and clustering

    N_, G_,R = Ft.N_,[], Ft.root  # rc=rdn+olp, comp N_|B_|C_:
    iN_,L_,TT,c,r, TTd,cd = comp_N_(N_,combinations(N_,2),ir) if N_[0].typ else comp_C_(N_,ir,fC=1); r+=ir
    if L_:
        if val_(TT, r+connw,TTw(Ft),(len(L_)-1)*Lw,1,TTd,r) > 0:  # G.L_ = Nt.Lt.N_, flat, additional root +=:
            m,d,tt,lc,lr = sum_vt(L_,root=getattr(Ft,'Lt'))
            rc = c/(R.c+c); R.dTT+=(tt-R.dTT)*rc; R.r+=(r-R.r)*rc; R.c+=c  # Ft: transitive root
            E_ = get_exemplars({N for L in L_ for N in L.nt if N.rim.m>ave}, r)  # N|C?
            G_,r = cluster_N(Ft, E_,r)  # form Bt, trans_cluster, sub+ in sum2G
            if G_:
                Ft = sum2F(G_, nF, Ft.root)
                if val_(Ft.dTT, r+nw, TTw(Ft), (len(N_)-1)*Lw,1, TTd,r) > 0:
                    G_,r = cross_comp(Ft,r,nF)  # agg+,trans-comp
    return G_,r  # G_ is recursion flag?

def comb_Ft(Nt, Lt, Bt, Ct, root, fN=0):  # default Nt

    T = CopyF(Nt)
    if fN: T = CN(Nt=T,dTT=Nt.dTT,c=Nt.c,r=Nt.r,root=root); T.Nt.root=T; add_Nt(T,Nt)  # +H and ext attrs
    else:  T.N_ = [Nt,Lt,Bt,Ct]; T.root=root  # nested tFt
    dF_ = []  # comp_N tFs | sum2G dFs
    if Lt: sum_vt([T,Lt],root=T.Nt); add_Lt(T.Nt,Lt); dF_ += [comp_F(T,Lt, T.r,T.Nt)]  # Lt in sum2G, tLt in comp_N
    if Bt: sum_vt([T,Bt],root=T); T.Bt=Bt; dF_ += [comp_F(T,Bt, T.r,root)]  # no ext pars?
    if Ct: sum_vt([T,Ct],root=T); T.Ct=Ct; dF_ += [comp_F(T,Ct, T.r,root)]
    if dF_:
        m,d,tt,c,r = sum_vt(dF_,root=getattr(T,'Lt'), merge=1); sum_vt([T,T.Lt], root=T)  # cross-fork covariance
        R=T.root; rc=c/(R.c+c); R.dTT+=(tt-R.dTT)*rc; R.r+=(r-R.r)*rc; R.c+=c  # G

    @property
    def L_(N):
        if isinstance(N.Nt.Lt, list): N.Nt.Lt =CF(root=N.Nt)
        return N.Nt.Lt.N_
    @L_.setter
    def L_(N,v):
        if isinstance(N.Nt.Lt, list): N.Nt.Lt =CF(root=N.Nt)
        N.Nt.Lt.N_ = v

    return T

