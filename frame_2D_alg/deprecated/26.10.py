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

# med_= list({C.N_[int(np.argmax(C.m_))] for C in C_})  # medoids, shouldn't be C-specific

def get_medoids(N_, _r):  # fable
        # strong-first NMS in C-space, net of Cs already represented
        N_ = sorted(N_, key=lambda N: max(r[1] for r in N.root_), reverse=True)
        med_, bM = [], {}  # bM: best medoid m per C
        for N in N_:
            M = sum(max(0, m - bM.get(C, 0)) for C, m, _ in N.root_)  # membership not yet represented
            if M > ave * _r:
                med_ += [N]
                for C, m, _ in N.root_: bM[C] = max(bM.get(C, 0), m)

def get_exemplars(N_,_r):  # multi-layer non-maximum suppression -> sparse clustering seeds, for medoids if N.Ct?

    for n in N_:
        rc = sum(r[0].c for r in n.root_); C = n.c + rc
        n.w = ((n.Rt.m * n.c) + sum([r[1]*r[0].c for r in n.root_]))/C
        # combined lateral and vertical match
    N_= sorted(N_, key=lambda n: n.w, reverse=True); E_,Inh_ = [],set()
    for rdn, N in enumerate(N_, start=1):  # strong-first
        inh_ = list(Inh_ & set(N.rim))  # stronger Es in N.rim
        oM = sum_vt(inh_,fm=1, wTT=ttE)[0] if inh_ else 0
        oV = oM / (N.Rt.m or eps)  # relative olp V
        if N.Rt.m * N.c * wE > ave* (_r+rdn+cE+oV):
            E_+=[N]; N.exe = 1  # point cloud of focal nodes
            Inh_.update(set(N.rim))  # extend inhibition zone
        else:
            break  # the rest of N_ is weaker, trace via rims
    if E_: FV_(CoF.get(), *sum_vt(E_))
    else:  E_ = [N_[0]]; N_[0].exe=1  # no gain, no inhibition, any N can be seed
    return E_

def cross_comp(root, pL_,r, dF=None, fall=1):  # recursion root

    def medoid_(N_, C_, _r, nexp):  # not reviewed
        # eval rng+ by combined membership value in n.root_, suppressed by stronger Ns
        C_ = set(C_)  # current batch only: root_ also holds nested batches' memberships
        for n in N_: n.w = sum(m * C.m for C, m, _ in n.root_ if C in C_) * n.c  # typicality * class coherence, summed over classes
        N_ = sorted(N_, key=lambda n: n.w, reverse=True); M_, Inh_ = [], set()
        for rdn, N in enumerate(N_, start=1):  # strong-first
            oM = sum(m for C,m,_ in N.root_ if C in Inh_) * N.c  # value in classes already probed
            oV = oM / (N.w or eps)  # relative overlap, 0:1
            if N.w * wX > ave * (_r + nexp + rdn + cX + oV):
                M_ += [N]; Inh_.update(C for C,_,_ in N.root_ if C in C_)  # its classes are covered
            elif N.w * wX <= ave * (_r+nexp+rdn+cX):
                break  # the rest is weaker w/o olp
        return M_
    L_, N_,G_ = [],[],[]
    if isinstance(pL_,tuple): pL_,iN_ = pL_
    for dist, dy_dx, _N,N, lc,lr, pTT,m,d,nexp in pL_:
        if _N != N and (fall or m>0):
            if gv_(m*(lc*wN/(lr*cN)) - ave* (r+cN)):
                Link = comp_N(_N,N, lr,lc, full=not dF, A=dy_dx, span=dist, rL=root)
                Link.rTT = np.abs(pTT-Link.dTT) / eps_(Link.dTT)  # prediction error
                L_+=[Link]; N_+=[_N,N]; Link.nexp=nexp
            elif not dF:  # pack as prelink
                _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
                pL = CL(typ=-1, N_=[_N,N],dTT=pTT,m=m,d=d,c=lc,r=lr,span=dist,box=box,nexp=nexp,angl=[dy_dx,1], yx=np.add(_N.yx,N.yx)/2)
                L_+= [pL]; N.rim+=[pL]; _N.rim += [pL]; N_+=pL.N_
    if L_:
        Lt = sum2F(L_:= L_+root.L_,None if dF else root,nF='Lt')  # rng_L_+= lower-rng_L_
        if dF: add2F(dF,Lt,merge=1); return  # comp_F: no agg+, dF out
        for N in (N_:= list(set(N_))): sum2F(N.rim, N, nF='Rt')  # -> N.Rt
        tt,m,c,r = Lt.dTT,Lt.m,Lt.c,Lt.r
        if gv_(m * (c*wcN/(r*ccN)) * ((len(L_)-1)*wL) - ave):
            root.Ct = CF(root=root); nexp=1
            G_ = cluster_N(root, exemplar_(N_,r:=r+1), r,c)  # provisional G_-> CC-> root.Ct
            while G_ and root.Ct and (med_:= medoid_(N_, root.Ct.N_, r:=r+1, nexp:=nexp+1)):  # rng+/CC, []: rng exhaustion
                cross_comp(root, pL_=(proj_L_(combinations(med_,2),root,r,nexp=nexp), iN_), r=r)  # recluster iN_ @ rng+
            if G_:
                sum2F(G_, root, nF='Nt')  # final G_->root.N_,_Nt->H
                cross_comp(root, proj_L_(combinations(G_,2), root, r:=r+1), r)  # agg+, same block one level up
        FV_(CoF.get(),tt,c,r)
    return G_

def expand_(N_, root, _r):  # rng+ eval

    for n in N_:  # combine medoid vals
        n.w = sum(m * C.c for C,m,_ in n.root_)
    N_ = sorted(N_, key=lambda n: n.w, reverse=True); rng_,Inh_ = [], set()
    for rdn, N in enumerate(N_, start=1):  # strong-first
        oM = sum(m for C,m,_ in N.root_ if C in Inh_) * N.c  # prior medoids
        oV = oM / (N.w or eps)  # relative overlap, 0:1
        if N.w * wX > ave * (_r+rdn+cX+oV):
            rng_ += [N]; Inh_.update(C for C,_,_ in N.root_ if C in root.C_)
        elif N.w * wX <= ave * (_r+rdn+cX):
            break  # weak w/o overlap
    return rng_  # Ns for rng+ comp

def sum2F(N_, root=None, m_=[],d_=[], merge=0, froot=0, nF=None):  # -> CF/CL/CN

    c_ = np.array([n.c for n in N_], dtype=float); N = N_[0]
    C = c_.sum(); TT,R = np.zeros((2,9)),0; fC = any(m_)
    w_ = c_ * np.array(m_) if fC else c_.copy()
    w_ /= w_.sum()
    typ = 0 if nF=='Nt' else 2 if fC else N.typ  # Nt: summary only
    cls_ = [CF,CL,CL,CN]  # typ=2 CCs
    for i, (n,w) in enumerate(zip(N_,w_)):
        if i:
            TT += n.dTT*w; R+=n.r*w; n_ += (n.N_ if merge else [n])
            if hasattr(n,'yx'):  kern+=n.kern*w; span+=n.span*w; yx+=n.yx*w  # links and higher
            if hasattr(n,'box'): box=extend_box(box,n.box)
        else:  # init
            TT = n.dTT*w; R=n.r*w; n_ = copy(n.N_ if merge else [n])
            if hasattr(n,'yx'):  kern=n.kern*w; span=n.span*w; yx=n.yx*w
            if hasattr(n,'box'): box=copy(n.box)
    F = (cls_[typ])(dTT=TT, c=C, r=R, nF=nF,typ=typ); F.N_ = n_
    if fC: F.m_,F.d_ = m_,d_; F.m,F.d = sum(m*c for m,c in zip(m_,c_))/C, sum(d*c for d,c in zip(d_,c_))/C; F.w=F.m
    else:  F.m, F.d = val_(TT,fd=1)
    if nF=='Lt' and root:
        angl = np.sum([L.angl[0] for L in N_],axis=0)
        root.mang = np.mean([comp_A(angl,L.angl[0])[0] for L in N_])
        root.angl = [angl,np.sign(root.dTT[1] @ ttcN[1])]
    if nF=='Nt':
        for N in N_: add_H(F.H, N.H, F)  # concat lower levs
    if 'yx' in locals():  F.kern=kern; F.span=span; F.yx=yx
    if 'box' in locals(): F.box = box
    if root is not None:
        F.wTT = root.wTT
        if nF in ('Nt','Lt'):
            w = F.c/root.c
            if 'yx' in locals(): root.kern+=kern*w; root.span+=span*w; root.yx+=yx*w
            if 'box' in locals(): root.box = copy(box)
            if nF=='Nt':
                if root.Nt: F.H = copy(root.H) + [Copy_(root.Nt, root=root)]  # previous top level
                root.Nt = F; F.root = root; root.dTT=copy(F.dTT); root.m,root.d,root.c,root.r = F.m,F.d,F.c,F.r
            else:
                if nF and not fC: setattr(root,nF,F); F.root = root  # root.Lt|Rt|Ct
                add2F(root,F,2)  # tF's params will be summed separately
    if froot == 1:
        for n in N_: n.root = root or F
    elif froot == 2: F.root = root
    if fC:
        for N, m, d in zip(N_, m_, d_): N.root_ += [[F,m,d]]
    return F

def cross_comp1(root, pL_,_r, dF=None, fall=1, iN_=None):  # recursion root, iN_: recluster/rng+

    L_,N_,G_, Nexp = [],[],[],0
    for dist, dy_dx, _N,N, lc,lr, pTT,m,d,nexp in pL_:
        if _N != N and (fall or m>0):
            if gv_(m*(lc*wN/(lr*cN)) - ave* (_r+cN)):
                Link = comp_N(_N,N, lr,lc, full=not dF, A=dy_dx, span=dist, rL=root)
                Link.rTT = np.abs(pTT-Link.dTT) / eps_(Link.dTT)  # prediction error
                L_+=[Link]; N_+=[_N,N]; Link.nexp=nexp
            elif not dF:  # pack as prelink
                _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
                pL = CL(typ=-1, N_=[_N,N], dTT=pTT,m=m,d=d,c=lc,r=lr,span=dist,box=box,nexp=nexp,angl=[dy_dx,1], yx=np.add(_N.yx,N.yx)/2)
                L_+= [pL]; N.rim+=[pL]; _N.rim += [pL]; N_+=pL.N_

    def Rng_(n_):  # w: combined membership weight / rdn: 1 + stronger members per C:
        for n in n_: n.w = sum(m * C.c / (1 + sum(_m>m for _m in C.m_)) for C,m,_ in n.root_)
        return [n for n in n_ if (n.m+n.w) * (n.c*wX / (n.r+cX)) > ave]  # or w - ave: w overlaps m?
    if L_:
        Lt = sum2F(L_,root=None if (dF or iN_) else root, nF='Lt'); tt,m,c,r = Lt.dTT,Lt.m,Lt.c,Lt.r
        if dF: add2F(dF,Lt, merge=1); return  # comp_F: no agg+, dF out
        for N in (N_:=list(set(N_))): sum2F(N.rim, N, nF='Rt')  # -> N.Rt
        if iN_: add2F(root.Lt, Lt, merge=1)
        else:   iN_ = N_
        if gv_(m * (c*wcN/(r*ccN)) * ((len(L_)-1)*wL) - ave):
            G_,C_ = cluster_N(root, exemplar_(iN_ or N_,r+1), r+1,c) or ([],[])   # extended N.rim -> tentative G_,C_,root_ for rng+:
        FV_(CoF.get(), tt,c,r)  # current rng xcomp
        if G_:
            if (rng_:= Rng_(iN_)) and (pL_:= proj_L_(combinations(rng_,2), root,_r,nexp=nexp+1)):  # rng+, w/o C_
                if gv_(sum(n.w for n in rng_) * (wX/cX) - ave) and (_G_:= cross_comp(root, pL_,_r, iN_=iN_)):
                    return _G_  # -> G_:
            if C_: sum2F(C_,root,nF='Ct')  # full rng
            elif root.Ct: root.Ct.H += [Copy_(root.Ct)]; root.Ct.N_ = []
            Nt = sum2F(G_, root,nF='Nt'); nm,nc,nr = Nt.m,Nt.c,Nt.r
            if gv_(nm * (nc*wX/(nr*cX)) - ave):  # agg+
                G_ = cross_comp(root, proj_L_(combinations(G_,2),root,_r+1),_r+1) or G_
    return G_

def cluster_C(root, G_,_r):  # sub-G seeds expand through their members' rims

    N_,_C_,tC_ = set(),G_,[]
    for C in _C_: N_.update(C.N_)
    for n in N_: n.root_,n._root_ = [],[]
    while True:  # reform C_, update,test C rims
        C_,cnt,rdn,Up,lc = [],0,0,0,0
        for _C in _C_:
            in_ = set(_C.N_); _N_ = _C.N_
            n_,m_,d_,M,T,R,up,c = [],[],[],0,0,0,0,0
            for exp in (0,1):  # prune members, then expand across their rims
                for n in _N_:
                    if n not in N_: N_.add(n); n.root_,n._root_ = [],[]
                    dtt,_ = base_comp(_C,n)
                    m,d = val_(dtt,ttcC,1); c+=n.c; _C.m += n.c * m  # mass-weighted?
                    _m,_d = next(((rt[1],rt[2]) for rt in n._root_ if rt[0] is _C), (0,0))  # prior membership, 0 if new
                    if m * n.c > ave* n.r:  # inclusion
                        up += abs(_m-m)+abs(_d-d); n_+=[n]; m_+=[m]; d_+=[d]; T+=n.c; M+=m*n.c; R+=n.r*n.c # separate c?
                    else: up += _m + abs(_d)  # exclusion
                if not exp:
                    _N_ = list({_n for n in n_ for L in n.rim for _n in L.N_} - in_)  # frontier: rims of kept members, excl. old members
            if not n_: continue
            r = _r + R/T
            if M * _C.m * (wcC*c / (r*_C.r*ccC)) * (((L:=len(n_))-1)*wL) > ave:
                C = sum2F(n_,root,m_,d_,nF='Ct'); C.typ=2  # -> n.root_
                if up * (c*wcC / (r*ccC)) * ((L-1)*wL)  > avd:  # len?
                    C_+=[C]; Up+=up; lc+=c  # active Cs' change and estimated next-loop work
                else: tC_+=[C]  # terminated if low membership change
                cnt+=T; rdn+=R  # process costs
        r = _r + rdn/(cnt or eps)
        oC_ = tC_ + C_  # output includes terminated C_
        if C_ and Up * (lc*wcC / (r*ccC)) > avd:
            N__ = [n for C in oC_ for n in C.N_]
            posL = sum(rt[1]>ave for n in N__ for rt in n.root_)
            totL = len(N__) * len(oC_)
            if gv_(posL/totL * (lc*wcP / (r*ccP)) * ((len(oC_)-1)*wL) - ave):
                oC_ = cluster_P(oC_,root); break
            else:
                for n in N_: n._root_ = n.root_; n.root_ = [rt for rt in n.root_ if rt[0] in tC_]
                _C_ = C_  # for next loop
        else: break
    if oC_:
        # sum2F(oC_,root=root,nF='Ct')  # agg+ root.N_ after higher CC selects them as medoids? This will be done in cross_comp now?
        FV_(CoF.get(), *sum_vt(oC_))
        return oC_

