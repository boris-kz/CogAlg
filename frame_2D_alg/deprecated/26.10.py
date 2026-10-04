def comp_N(_N,N, r,c, full=1, A=None,span=None, rL=None):

    def sum_dF_(dFt,L):
        for dF in dFt:
            add2F(getattr(L.H[0], dF.nF), dF)  # each level in L.H is dFt
            if dF.nF=='Nt': add_H(L.H,dF.H, L)  # or in add2F?
    def sum_dN_(dN_,L):
        for dN in dN_: add_H(L.H, dN.H, L)
        TT,C,R = sum_vt(dn_)
        w = C/L.c * (L.r/R)
        L.dTT = (L.dTT + TT*w) / (1+w); L.m,L.d = val_(L.dTT,ttN,1)
        L.r = (L.r+ R*w) / (1+w)
        L.c += C
    def comp_H(_N, N, L):
        dH,TT,C,R = [],np.zeros((2,9)),0,0
        for _lev, lev in zip(_N.H+[_N], N.H+[N]):  # should be top-down
            if not (_lev and lev): continue  # skip empty level
            tt = comp_derT(_lev.dTT[1],lev.dTT[1])
            lc = min(_lev.c,lev.c); lr = (_lev.r+lev.r)/2
            TT += tt*lc; C += lc; R += lr*lc; m,d = val_(tt,ttN,1)
            dH += [CF(dTT=tt,m=m,d=d,c=lc,r=lr,root=L)]
        TT/=C; R/=C
        w = C/L.c * (L.r/R)  # dH weight
        L.dTT = (L.dTT + TT*w) / (1+w)
        L.r = (L.r+ R*w) / (1+w); L.c += C; L.m, L.d = val_(L.dTT,ttN,1)
        L.H += dH
    L = CL(N_=[_N,N], c=c,r=r,root=rL)
    if full:
        dTT = base_comp(_N,N)[0]
        if span is None: span = np.hypot(*_N.yx - N.yx)
        yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx
        box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
        angl = [np.zeros(2) if A is None else A, np.sign(dTT[1] @ ttN[1])]
        L.yx=yx; L.box=box; L.span=span; L.angl=angl; L.kern=(_N.kern+N.kern)/2
    else: dTT = comp_derT(_N.dTT[1],N.dTT[1])
    m,d = val_(dTT, ttN,1); L.dTT,L.m,L.d = dTT,m,d
    if N.typ >1:  # skip PPs, Nts?
        dn_ = []  # spec, cross_comp N_|Ft_-> top tLev
        if N.typ==3 and gv_(m* (c*wN /(r*cN)) - ave):  # CN, add L2N?
            for i,(_Ft,Ft, tnF) in enumerate(zip((_N.Nt,_N.Lt,_N.Bt,_N.Ct),(N.Nt,N.Lt,N.Bt,N.Ct),('Nt','Lt','Bt','Ct'))):  # no comp Ct?
                if _Ft and Ft: dn_ += [comp_F(_Ft, Ft, r,L)]; r+=(i or 1)-1  # unique Nt,Lt
        elif gv_(m* (c*wF /(r*cF)) - ave):   # Lt| Ct| Nt, merge?
            for _n,n in product(_N.N_,N.N_):
                dn_ += [comp_N(_n, n, r,min(_n.c,n.c), rL=L,full=0)]  # CN L.nt, rL spec, full=CN?
        if dn_:
            sum_dF_(dn_,L) if N.typ==3 else sum_dN_(dn_,L)  # merge if no or weak Bt? comp x fork, H levs?
        else:
            L.H = [[]]  # empty top dFt
            if _N.H and N.H: comp_H(_N, N, L)  # light spec
    if full:
        for n, _n in (_N,N),(N,_N): n.rim += [L]
    FV_(CoF.get(), L.dTT, L.c, L.r)
    # or merge N -> _N?
    return L

def sum2G(F_, wTT, root=None, _r=0):  # finalize cluster

    G = CN(root=root,wTT=wTT)
    G_, L_,pL_= [],[],[]  # sub_Gs for CC, L_ can't be empty?
    N_,_L_,B_ = F_
    for L in _L_:
        if L.typ==1: L_+=[L]
        else: pL_ += [L]  # projected
    if pL_ and sum_vt(pL_,fm=1,wTT=wTT)[0]*wN > ave*(cN*np.mean([L.r for L in pL_])):
        L_ += [comp_N(*L.N_,L.r,L.c,1,L.angl[0],L.span) for L in pL_]
    [sum2F(F_,G,nF=nF) for F_,nF in zip((N_,B_,L_),('Nt','Bt','Lt')) if F_]
    G.m,G.d = val_(G.dTT,wTT,1)
    if Bt := G.Bt:  # der+'sub+
        bd,bc,br = Bt.d,Bt.c,Bt.r+_r+1
        if N_[0].typ!=1 and gv_(bd*bc*wX - ave*(br+cX)):  # no ddfork, eval len B_?
            cross_comp(F2N(G.Bt), proj_L_(combinations([F2N(L) for L in Bt.N_],2),G,br),br)
        if RR:= root.root: Bt.brrw = Bt.m* (RR.m* (decay* (RR.span/G.span)))  # root - external lend?
    if Lt := G.Lt:  # rng+,sub+
        m,c,r = Lt.m,Lt.c, Lt.r+_r
        if gv_(m* (c*wX / (r*cX)) - ave):  # rng+
            if g_ := cross_comp(G, proj_L_(combinations(N_,2), G,r,nexp=L_[0].nexp+1), r,fagg=0):
                G.H+=[sum2F(G.N_)]; N_= G.N_= g_
        if gv_(m* (c*wcN / (r*ccN)) * ((len(N_)-1)*wL) - ave):
            if G_ := cluster_N(G, get_exemplars(N_,r), r+1,c):  # higher filter: r+1,-> sub_Gs for CC
                sum2F(G_,G, nF='Nt' if G_[0].typ==3 else 'Ct')  # unpack tentative G.N_?
    if G.Lt or G.Bt: G.dTT,G.c,G.r = sum_vt([G.Nt,G.Lt,G.Bt]); G.m,G.d = val_(G.dTT,G.wTT,fd=1)  # recompute after deeper sub
    FV_(CoF.get(), G.dTT, G.c, G.r)
    return G, G_
