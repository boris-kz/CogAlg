def comp_N(_N,N, r,c, full=1, A=None,span=None, rL=None, _j=0):  # _j: index in dH / spec

    def comp_H(_N, N, L):
        dH,TT,C,R = [],np.zeros((2,9)),0,0
        for j, (_lev,lev) in enumerate( zip(_N.H,N.H)):  # current dTT already compared, levs 2+ from comp_H
            if not (_lev and lev): dH += [[]]; continue  # preserve levels
            lc = min(_lev.c,lev.c); lr = (_lev.r+lev.r)/2
            tt = comp_dT(_lev.dTT[1], lev.dTT[1], lc, j)  # up to _j?
            C += lc; R += lr*lc; TT += tt*lc; m = val_(tt,ttN)
            dH += [CF(dTT=tt,m=m,c=lc,r=lr,root=L)]
        if not C: return
        TT/=C; R/=C
        w = C/L.c * (L.r/R)
        L.dTT = (L.dTT+ TT*w) / (1+w); L.m= val_(L.dTT,ttN)
        L.r = (L.r+R*w) / (1+w); L.c += C
        L.H += dH  # lev1 from comp N.N_
    def sum_dn_(dn_, L, _lev):
        lev = sum2F(dn_,nF='tF',root=L); TT,C,R = lev.dTT,lev.c,lev.r  # dN_ maybe dFt, works the same
        for dn in dn_: add_H(L.H, dn.H, L)  # lower der level, if hasattr H?
        L.H += [add2F(_lev,lev,merge=2) if _lev else lev]  # merge top and new der level?
        w = C/L.c * (L.r/R)
        L.dTT = (L.dTT+ TT*w) / (1+w); L.m = val_(L.dTT,ttN); L.r = (L.r+R*w)/(1+w); L.c+=C
    L = CL(N_=[_N,N], c=c,r=r,root=rL)
    if full:
        dTT,ddTT = base_comp(_N,N,_j)  # !rn?
        if span is None: span = np.hypot(*_N.yx-N.yx)
        yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx
        box = np.array([min(_y,y),min(_x,x), max(_y,y),max(_x,x)])
        angl = [np.zeros(2) if A is None else A, np.sign(dTT[1] @ ttN[1])]
        L.yx=yx; L.box=box; L.span=span; L.angl=angl; L.kern=(_N.kern+N.kern)/2
    else:
        dTT = comp_dT(_N.dTT[1],N.dTT[1], c, _j+1)  # no M,D,ext,kern?
    m = val_(dTT,ttN); L.dTT,L.m = dTT,m  # exclude ddTT from m?
    lev = None
    if "ddTT" in locals():
        m+=(dm:= val_(ddTT,ttN)); lev = CF(N_=[_N,N],dTT=ddTT,m=dm,c=c)  # H=[lev], same c?
    if N.typ:  # converted Ls, skip PPs, Fts?
        dn_=[]
        if N.typ==3:
            if gv_(m* (c*wN /(r*cN)) - ave):
                for i,(_Ft,Ft) in enumerate(zip((_N.Nt,_N.Lt,_N.Bt,_N.Ct), (N.Nt,N.Lt,N.Bt,N.Ct))):
                    if _Ft and Ft:
                        dn_ += dF:= [comp_F(_Ft,Ft, r,L,_j)]; r+= (i or 1)-1
                        if not i: L.H = dF.H  # or dNt.H + [dLt.H]?
        elif gv_(m* (c*wF /(r*cF)) - ave):  # L|B, forms trans-Ls?
            for _n,n in product(_N.N_,N.N_):
                if _n is not n: dn_+= [comp_N(_n,n, r,min(_n.c,n.c,_j=_j),rL=L,full=0,_j=_j)]  # CC: medoid Ns may overlap?
        if dn_: sum_dn_(dn_,L, lev)  # tFt if CN else dN_
        else:
            L.H += [[]]  # placeholder for skipped diff(N.N_)
            if _N.H and N.H: comp_H(_N,N,L)
            if lev: L.H +=[lev]
    if full:
        for n,_n in (_N,N),(N,_N): n.rim += [L]
    FV_(CoF.get(),L.dTT,L.c,L.r)
    return L

def comp_F(_F, F, ir=0, rL=None,_j=0):

    c = min(_F.c,F.c); ddTT = comp_dT(_F.dTT[1],F.dTT[1],c,_j)  # no M,D,ext,kern?
    m = val_(ddTT,ttF); r = (_F.r+F.r)/2 + ir
    dF = CF(dTT=ddTT, m=m,r=r,c=c, nF=F.nF)
    if (_N_:=_F.N_) and (N_:=F.N_):
        Np_ = list(product(_N_,N_))  # pairs; Lt,Bt: links, Ct: centroids
        tt = (rL.dTT*rL.c + ddTT*dF.c) / (rL.c+dF.c)
        if gv_(val_(tt,ttX) * (c*wX / (r*cX)) - ave):  # any sub-comp?
            if F.nF!='Nt': [F2N(n) for n in set(_N_+N_) if not isinstance(n,CN)]  # Rt,Nt,Bt
            cross_comp(rL, proj_L_(Np_,2,r), r,dF,_j=_j)
    # add comp_H?
    FV_(CoF.get(), dF.dTT, dF.c, dF.r)
    return dF  # no cross-fork N_, no L ext updates?

def sum_dn_(dn_,L):
        lev = sum2F(dn_,nF='tF',root=L); TT,C,R = lev.dTT,lev.c,lev.r  # dN_ maybe dFt, works the same
        for dn in dn_: add_H(L.H, dn.H, L)  # lower der level, if hasattr H?
        L.H += [CF(H=[lev],dTT=TT,m=lev.m,c=C,r=R,root=L)]  # new top level?
        w = C/L.c * (L.r/R)
        L.dTT = (L.dTT+ TT*w) / (1+w); L.m = val_(L.dTT,ttN); L.r = (L.r+R*w)/(1+w); L.c+=C

def comp(_pars, pars, meA,deA, c=1, j=0):  # compute m_,d_ from inputs or derivatives

    m_,d_ = [],[]; _i_, i_ = [],[]
    for i, (_p,p, mass,avad,avi,wd) in enumerate(zip(_pars,pars, mass__[j],avad__[j],avi__[j],dcov__[j])):  # prior-frame mean |d| = massless m
        if isinstance(_p, list):  # vector angle
            mA, dA = comp_A(_p, p); m_ += [mA]; d_ += [dA]; _i_+=[0]; i_ += [0]
        else:
            m = min(abs(_p), abs(p))
            if (_p<0) != (p<0): m =- m  # gap between comparands
            ard = avad * (((abs(_p)+abs(p))/2) / avi) ** wd  # massless m
            m_ += [m* mass + ard* (1-mass)]
            # nd = avad * ((_ap+ap)/2 / avi) ** wd  # expected |d| at this magnitude: normalizes |d| in val_, outside the mix
            # m_ += [m* mass + nd* (1-mass)]  # massless m = expected |d|
            d_ += [_p - p]
            _i_+=[_p]; i_ += [p]
    global i___,d___,c__  # init dH=[lev0], mA mass=0:
    i___[j] += [(np.abs(_i_+[0]) + np.abs(i_+[0])) / 2]; d___[j]+= [np.abs(d_+[deA])]; c__[j]+=[c]
    return np.array(m_+[meA]), np.array(d_+[deA])
'''
    if Bt:= G.Bt:  # der+
        bm,bc,br = Bt.m,Bt.c,Bt.r+_r+1
        if gv_(bm * (bc*wX / (br*cX)) - ave):
            cross_comp(F2N(G.Bt), proj_L_(combinations(Bt.N_,2),G,br),br)
        if RR:= root.root if root else None: Bt.brrw = Bt.m* (RR.m* (decay* (RR.span/G.span)))
'''
def Mass_():
    # not revised
    # define mass/par: anti-correlation between |i| and positional change of object, must be temporal?
    for i__, d__, c_, mass_, avad_, avi_, dcov_ in zip(i___, d___, c__, mass__, avad__, avi__, dcov__):
        if not c_: continue  # no new comps in dLev
        c_ = np.array(c_)[:, None];
        C = c_.sum()
        avi_[:] = (i__ * c_).sum(0) / C  # weighted mean inp / par
        avd_ = (d__ * c_).sum(0) / C  # weighted mean |d| / par
        devI__ = i__ - avi_  # input dev / comp
        devD__ = avd_ - d__  # inv_m: inverse |d| dev / comp
        icov__ = devI__ * devD__ * c_  # weighted co-dev / comp, or by div_comp?
        mcov_ = icov__.sum(0) / eps_(np.abs(icov__).sum(0))  # -1:1
        dcov_[:] = -mcov_
        '''
        dcov_[:] = -rcov_  # 1: |d| grows with |i|, 0: independent, -1: shrinks
        if e_:  # expected d, base lev only
            e_ = np.array(e_)[:,None]; devE_ = e_ - (e_*c_).sum() / C
            mass_[:] = (rcov_+1) / 2  # 0:1: share of |i| co-dev that agrees with ext match, .5 if unrelated
        elif j: mass_[:] = mass__[j-1]  # d of massive is massive '''
        mass_[:] = (mcov_ + 1) / 2  # 0:1: share of co-dev that agrees in sign, .5 if unrelated
        avad_[:] = avd_  # ave abs d_

def add_fH(nlev):  # add empty Levs before cross_comp
    while len(i___)<nlev: i___.append([]); d___.append([]); c__.append([])
    while len(mass__)<nlev:
        mass__.append(copy(mass__[-1]) if mass__ else np.zeros(8))  # init from j-1: d of massive is massive
        avad__.append(copy(avad__[-1]) if avad__ else copy(wT))
        avi__.append(copy(avi__[-1]) if avi__ else np.ones(8))
        dcov__.append(copy(dcov__[-1]) if dcov__ else np.zeros(8))

def comp_(_pars, pars, reA,deA, c=1, j=0):  # compute m_,d_ from inputs or derivatives

    rd_,d_ = [],[]; _i_, i_ = [],[]
    for i, (_p,p, avad) in enumerate(zip(_pars,pars, avad__[j])):
        if isinstance(_p, list):  # vector angle
            rA,dA = comp_A(_p,p); rd_+= [rA]; d_+= [dA]; _i_+=[0]; i_+= [0]
        else:
            d_ += [d := (_p - p)]  # ad borrows from avm: 0 @ ad=0, 0.5 @ avad, 1 @ inf:
            rd_+= [abs(d) / avad]  # core|cntr -1:1 vd: (avad-ad)/(avad+ad) = (rd-1)/(rd+1), v = abs(vd* (c/r)) -avad
    global dt___,c__; dt___[j]+= [np.abs(d_+[deA])]; c__[j]+=[c]
    return np.array(rd_+[reA]), np.array(d_+[deA])

def comp_dT(_i_,i_, c, j):  # scalars, j: dLev index, >1 in comp_H only
    '''
    _a_, a_ = np.abs(_i_), np.abs(i_)
    mm_ = np.minimum(_a_,a_); mm_[(_i_<0) != (i_<0)] *= -1
    sd_ = scaled_avad_ = avad_ * (eps_((_a_+a_)/2) / eps_(avi_)) ** dcov_
    m_ = mm_* mass_ + sd_* (1-mass_)  # avad_: prior regime mean |d| = massless m
    global i___,d___,c__
    i___[j] += [(_a_+a_)/2]; d___[j] += [np.abs(d_)]; c__[j]+=[c]  # dLev += [comp]
    return np.array([m_,d_])
    '''
    avad_ = avad__[j]  # mass_,avad_,avi_,dcov_ = mass__[j], avad__[j], avi__[j], dcov__[j]
    d_ = _i_ - i_
    rd_ = np.abs(d_) / eps_(avad_)
    global d___, cc__; d___[j] += [np.abs(d_)]; cc__[j] += [c]
    return np.array([rd_,d_])

def val_(TT, wT=wT):  # multi-variate rel match for membership, rel diff for boundary
    '''
    restore this
    m_, d_ = TT; ad_ = np.abs(d_)  # dval = brrw from different-base m: directional Lt|Bt
    rm_ = m_ / eps_(m_+ ad_)  # if may be massive, revised for symmetric inv_m in -1:1:
    rm_ = np.abs(ad_- m_) / eps_(ad_+ m_)  # massless m = avad
    # L.v = ((rm_:=TT[0]) / (2-rm_)) @ wTT[0] * (c/r) - avm_
    '''
    return ((1-(rd_:=TT[0])) / (1+rd_)) @ wT / wT.sum()  # -1:1 for +ve weights

def sum_vt(N_, fr=0, fm=0, wTT=wTT, fdiv=1):  # basic weighted sum of CN|CF list

    C = sum(n.c for n in N_); rD=R=0; TT = np.zeros((2,8))
    for n in N_:
        w = n.c/C; tt = (n.rTT if fr else n.dTT) * w; TT+=tt; r= n.r * w; R += r
        if fm:
            rd = tt[0] @ wTT[0] / 8  # in -1:1?
            rD += (rd / (ave*r)) if fdiv else (rd - ave*r)
    return (rD,TT,C,R) if fm else (TT,C,R)  # sum dTT for comp, not eval?

def Mass_1():  # per dLev: avad from comps, mass from clusters
        for p__,m__,gc_,d__,cc_, mass_,avad_ in zip(p___,m___,gc__,d___,cc__, mass__,avad__):
            if cc_:
                cc_ = np.array(cc_)[:,None]; avad_[:] = (d__*cc_).sum(0) / cc_.sum()  # mean |d| / param
            if gc_:
                gc_ = np.array(gc_)[:,None]; C = gc_.sum()
                devI__ = p__ - (p__*gc_).sum(0) / C  # |param| dev / G
                devM__ = m__ - (m__*gc_).sum(0) / C  # dev of G.m w/o own column / G
                covI__ = devI__ * devM__ * gc_
                covIV_ = (acovI_ := np.abs(covI__).sum(0)) > 0
                mass_[covIV_] = (covI__.sum(0) / eps_(acovI_))[covIV_]  # -1:1: share of co-deviation that agrees in sign, else keep init?

def comp_F1(_F, F, nF, ir=0, rL=None, j=None):

    c = min(_F.c,F.c)
    ddTT = comp_dT(_F.dTT[1],F.dTT[1],c, j+1)  # 1st der, same dLev as base_comp ddTT
    m = val_(ddTT,Fw_); r = (_F.r+F.r)/2 + ir
    dF = CF(dTT=ddTT, m=m,r=r,c=c, nF=nF, root=rL)
    if nF in ('Nt','Lt','Dt'):  # spec draft
        if (_N_:=_F.N_) and (N_:=F.N_):  # always true?
            tt = (rL.dTT*rL.c + ddTT*c) / (rL.c+c)  # vertical comb?
            if gv_(val_(tt,Xw_) * (c*wX /(r*cX)) - ave):
                cross_comp(rL, proj_L_(list(product(_N_,N_)),2,r), r, dF,_j=j)  # tL-> dF.N_
            elif _F.H and F.H:  # dF.H[-1]+[dLt]?
                comp_H(_F,F, dF,j)  # rL: lev in L.H, no cross-fork N_, no L ext updates?
    setattr(rL,nF,dF)
    FV_(CoF.get(), dF.dTT, dF.c, dF.r)

def cross_comp(root, pL_, _r, dF=None, fall=1, iN_=None, frng=1, _j=0):  # recursion root, iN_: recluster/rng+

    def Rng_(n_):  # w: combined membership weight / rdn: 1 + stronger members per C:
        for n in n_: n.w = sum(m * C.c / (1 + sum(_m>m for _m in C.m_)) for C,m in n.root_)
        return [n for n in n_ if (n.m+n.w) * (n.c*wX / (n.r+cX)) > ave]

    L_, N_,G_,C_,Dt = [],[],[],[],[]
    for dist, dy_dx, _N,N, lc,lr, pTT,m,nrng in pL_:
        if _N != N and (fall or m>0):
            if gv_(m*(lc*wN/(lr*cN)) - ave* (_r+cN)):
                Link = comp_N(_N,N, lr,lc, full=not dF, A=dy_dx, span=dist, rL=root, _j=_j)  # if H not updated, +rroot.Hs?
                Link.rTT = np.abs(pTT-Link.dTT) / eps_(Link.dTT)  # prediction error
                L_+=[Link]; N_+=[_N,N]; Link.nrng=nrng
            elif not dF:  # += prelink
                _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
                pL = CN(typ=-1, nt=[_N,N], dTT=pTT,m=m,c=lc,r=lr,span=dist,box=box,angl=[dy_dx,1], yx=np.add(_N.yx,N.yx)/2)
                pL.nrng=nrng; L_+= [pL]; N.rim+=[pL]; _N.rim += [pL]; N_+=pL.nt
    if L_:
        Lt = sum2F(L_,root=None if (dF or iN_) else root, nF='Lt'); tt,m,c,r = Lt.dTT,Lt.m,Lt.c,Lt.r
        if dF:  # comp_F: adds to dF vals, no agg+
            add2F(dF,Lt, merge=1); return
        for N in (N_:= list(set(N_))): sum2F(N.rim, N, nF='Rt')  #->N.Rt
        if iN_: add2F(root.Lt, Lt, merge=1)
        else: iN_ = N_; [mT___(N,_j) for N in N_]  # add N.mt_
        # cluster:
        if gv_(m * (c*wcN/(r*ccN)) * ((len(L_)-1)*wL) - ave):  # len coef is tentative
            G_, C_ = cluster_N(root, exemplar_(iN_ or N_,r+1), r+1,c,_j)  # tentative G_|C_|P_/ rng+
        if len(dL_:= [l for l in L_ if l.typ == 1 and l.nrng == 1 and -l.m > ave]) > 1:  # pooled contrast links
            Dt = sum2F(dL_, root=root, nF='Dt'); Dt.r += _r + 1
            if gv_(-Dt.m * (Dt.c*wX /(Dt.r*cX)) - ave):  # der+
                cross_comp(F2N(Dt), proj_L_(combinations(dL_,2),Dt.r,_j=_j), root,Dt.r,_j=_j)
            if Dt.Lt: add2F(root.Lt if root.nF=='Dt' else root, Dt.Lt, merge=2,fH=0)  # der+ sum in Dt.Lt
        FV_(CoF.get(), tt,c,r)  # current-rng xcomp
        if G_:
            if not frng: return G_,C_  # tentative, in rng+ loop only
            while (rng_:= Rng_(iN_)) and gv_(sum(n.w for n in rng_)* (wX/cX) -ave) and (pL_:= proj_L_(combinations(rng_,2),root,_r,nrng=(nrng:=nrng+1),_j=_j)):
                _G_,_C_ = cross_comp(root, pL_,_r,iN_=iN_,frng=0,_j=_j) or ([],[])
                if not _G_: break  # max rng
                G_,C_= _G_,_C_
            if C_: sum2F(C_,root,nF='Ct')  # agg+ / max-rng Gs
            G_ = [sum2G(g,ttcN,root,_r,_j) for g in G_]
            Nt = sum2F(G_,root,nF='Nt'); nm,nc,nr = Nt.m,Nt.c,Nt.r
            if gv_(nm * (nc*wX/(nr*cX)) - ave):  # agg+
                G_ = cross_comp(root, proj_L_(combinations(G_,2),root,_r+1,_j=_j),_r+1,_j=_j) or G_
    return G_
def cross_comp1(root, pL_, _r, dF=None, fall=1, fN=1, _j=0):  # recursion root, fN CN_ else CC_

    L_, N_,G_,C_,Dt = [],[],[],[],[]
    for dist, dy_dx, _N,N, lc,lr, pTT,m,nrng in pL_:
        if _N != N and (fall or m>0):
            if gv_(m*(lc*wN/(lr*cN)) - ave* (_r+cN)):
                Link = comp_N(_N,N, lr,lc, full=not dF, A=dy_dx, span=dist, rL=root, _j=_j)  # if H not updated, +rroot.Hs?
                Link.rTT = np.abs(pTT-Link.dTT) / eps_(Link.dTT)  # prediction error
                L_+=[Link]; N_+=[_N,N]; Link.nrng=nrng
            elif not dF:  # += prelink
                _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
                pL = CN(typ=-1, nt=[_N,N], dTT=pTT,m=m,c=lc,r=lr,span=dist,box=box,angl=[dy_dx,1], yx=np.add(_N.yx,N.yx)/2)
                pL.nrng=nrng; L_+= [pL]; N.rim+=[pL]; _N.rim += [pL]; N_+=pL.nt
    if L_:
        Lt = sum2F(L_,root=None if dF else root, nF='Lt'); tt,m,c,r = Lt.dTT,Lt.m,Lt.c,Lt.r
        if dF:  # comp_F: adds to dF vals, no agg+
            add2F(dF,Lt, merge=1); return
        for N in (N_:= list(set(N_))): mT___(N,_j); sum2F(N.rim, N, nF='Rt')  #->N.Rt
        if fN:  # cluster_N:
            if gv_(m * (c*wcN/(r*ccN)) * ((len(L_)-1)*wL) - ave):  # len coef is tentative
                G_, C_ = cluster_N(root, exemplar_(N_,r+1), r+1,c,_j)  # tentative G_|C_|P_
        elif gv_(m * (c*wcC/(r*ccC)) * ((len(L_)-1)*wL) - ave):
            C_ = cluster_C(root, exemplar_(N_,r+1), r+1,_j)  # still need exemplars?
        if len(dL_:= [l for l in L_ if l.typ == 1 and l.nrng == 1 and -l.m > ave]) > 1:  # pooled contrast links
            Dt = sum2F(dL_, root=root, nF='Dt'); Dt.r += _r + 1
            if gv_(-Dt.m * (Dt.c*wX /(Dt.r*cX)) - ave):  # der+
                cross_comp(F2N(Dt), proj_L_(combinations(dL_,2),Dt.r,_j=_j),Dt.r,_j=_j)
            if Dt.Lt: add2F(root.Lt if root.nF=='Dt' else root, Dt.Lt, merge=2,fH=0)  # der+ sum in Dt.Lt
        FV_(CoF.get(),tt,c,r)  # current composition
        if C_:  # run current  cfork first? Because the subsequent cross_comp may run higher level's C fork?
            sum2F(C_,root,nF='Ct')
            if gv_(sum(C.m for C in C_)* (wX/cX) -ave) and (pL_:= proj_L_(combinations(C_,2),root,_r,_j=_j)):
                C_ = cross_comp(F2N(root.Ct), pL_,_r,fN=0,_j=_j)  # C_ agg+: skip cluster_N
        if G_:
            G_ = [sum2G(g, ttcN, root, _r, _j) for g in G_]; sum2F(G_,root,nF='Nt')
            if gv_(sum(G.m for G in G_)* (wX/cX) -ave) and (pL_:= proj_L_(combinations(G_,2),root,_r,_j=_j)):  # no nrng=(nrng:=nrng+1)?
                G_ = cross_comp(root, pL_,_r,_j=_j)  # G_ agg+
    return G_

def cluster_N(root, _N_, _r,_c,_j):  # flood-fill node | link clusters, flat, replace iL_ with E_?

    Gt_,in__,fin_,G_,C_ = [],set(),[],[],[]  # root attrs, add prelink pL_,pN_? fin_ should be tile-wide?
    for N in _N_:  # form G per remaining N
        if N in fin_ or (root.root and not N.exe): continue  # no exemplars in Fg
        N_ = [N]; L_,B_ = [],[]; fin_ += [N]  # init G
        in_ = set(); __L_= N.rim  # spliced rim
        while __L_:
            _L_ = []
            for L in set(__L_) - (in__|in_):  # flood-fill via frontier links
                _N = L.nt[0] if L.nt[1] in fin_ else L.nt[1]; in_.add(L)
                if _N not in fin_:
                    pm,nm = nt_vt(*L.nt)  # +ve,-ve m
                    if pm > ave *_r:  # additional filter, cluster nt, L,C_ by combined rim density, add gv_?
                        span = np.sqrt(len(N_))  # approx span
                        if span > 3:  # refine by rim connectivity / norm span
                            iM = sum([L.m for L in _N.rim if (L.nt[0] if L.nt[1] is _N else L.nt[1]) in N_])
                            if iM / (span*decay) < ave * (_r-1): continue  # normalized N-to-N_ match, sum for sum2G?
                        L_ += [L]; N_ += [_N]; fin_ += [_N]
                        _L_+= [l for l in _N.rim if l not in (in_|in__) and ((l.nt[0] in fin_) ^ (l.nt[1] in fin_))]  # new frontier links, +|-?
                    elif L.nrng==1: B_ += [L]  # boundary
            __L_ = list(set(_L_))
        if N_:
            ft_ = []
            for i,(F_,nF) in enumerate(zip((N_,L_,B_),('Nt','Lt','Bt'))):  # no Ct yet, formed below
                F_ = list(set(F_)) or []; tt,fc,fr = sum_vt(F_,wTT=ttcN) if F_ else (np.zeros((2,8)),0,0)
                ft_+= [[tt,fc,fr]]
            (nt,nc,nr),(lt,lc,lr),(bt,bc,br) = ft_
            c = nc+lc+bc; r = (nr*nc +lr*lc +br*bc) /c; tt = (nt*nc +lt*lc) /nc+lc   # br includes overlap?
            if L_:
                nm,lm,bm = [val_(tt, root.wTT[0]*cNw_) for tt in (nt,lt,bt)]
                spanR = np.mean([n.span for n in N_]) / np.mean([l.span for l in L_])
                coreM = (nm * decay**spanR * nc + lm * lc) / (nc+lc)
                m = coreM * (1-bm)  # core mass * contour specificity
            if gv_(m * (c*wcN/(r*ccN)) * ((len(N_)-1)*wL) - ave):
                C = sum2F(N_,root,nF='C',typ=3,m=m); C.L_,C.B_ = L_,B_  # initial connectivity G
                Gt_ += [(C,m,c,r)]; in__|=in_
            else:
                for n in N_: fin_.remove(n)
    if Gt_:
        C,M,R = 0,0,0; _r+=1
        for G,m,c,r in Gt_: G_+=[G]; C+=c; M+=m*c; R+=r*c
        M/=C; R/=C
        FV_(CoF.get(), *sum_vt(G_)[:-1],R)
        if gv_(M * (C*wcC / ((_r+R)*ccC)) * ((len(G_)-1)*wL) - ave):
            C_ = cluster_C(root, G_,_r+R,_j)
    return G_,C_

def sum2G(F_, wTT, root=None, _r=0, _j=0):  # finalize CN from C or (N_,L_,B_), eval trans_cluster, der+, sub+
    def trans_cluster(G):
        for L in G.L_:  # draft:
            for tL in (L.Nt.N_ + L.Bt.N_):  # top level
                if tL.m * (tL.c*wcN / (tL.r*ccN)) > ave:
                    pass  # sum m across levs for merge eval?
                for lev in L.H:  # tFt/Nt | tN_/Bt, not Lt|Ct  # not revised:
                    if lev:
                        for tF in lev.N_ if not lev.N_[0].nt else [lev]:  # CF|CL: Nt|Bt, from spec?
                            for tL in tF.N_:
                                if tL.m * (tL.c*wcN / (tL.r*ccN)) > ave:  # merge trans_link.N_.roots:
                                    rt0 = getattr(tL.N_[0].root,'root',None); rt1 = getattr(tL.N_[1].root,'root',None)
                                    if rt0 and rt1 and rt0 != rt1:
                                        if rt1.H: add_H(rt0.H, rt1.H, rt0, fN=1)
                                        add2F(rt0,rt1, merge=1,fH=0)  # recompute Nt attrs / G
                    else: break  # !top lev
    G = CN(root=root,wTT=wTT)
    sG_, L_, pL_ = [],[],[]
    if isinstance(F_,tuple): N_,_L_,B_ = F_  # from trace_edge, vect_edge, frame_H
    else:  # C from CN,CC: extract internal _L_ and perimeter B_ from N_ rims
        N_,_L_,B_ = F_.N_,F_.L_,F_.B_
    for L in _L_:
        if L.typ==-1: pL_+= [L]
        else:         L_ += [L]
    if pL_:
        pm, pc, pr = sum_vt(pL_, fm=1, wTT=wTT)[0], sum([L.c for L in pL_]), np.mean([L.r for L in pL_])
        if pm * (wN*pc /(cN*pr)) > ave:
            for pL in pL_:
                L_ += [comp_N(*pL.nt,pL.r,pL.c,1,pL.angl[0],pL.span,root,_j)]
                for N in pL.nt: N.rim.remove(pL)
    for N in N_:
        if N.rim: N.Rt = sum2F(N.rim,root=N,nF='Rt')
    [sum2F(f_,G,nF=nF) for f_,nF in zip((N_,B_,L_),('Nt','Bt','Lt')) if f_]  # no Dt?
    if G.L_: trans_cluster(G)  # splice trans_links, merge L.nt.roots
    if Lt := G.Lt:  # sub+ CN, no rng+, agg+?
        m,c,r = Lt.m,Lt.c, Lt.r+_r
        if gv_(m* (c*wcN / (r*ccN)) * ((len(N_)-1)*wL) - ave):  #-> sub_Gs @ higher filter: r+1
            sG_ = cluster_N(G, exemplar_(N_,r), r+1,c,_j,fsub=1)  # spliced sub-Cs
    Nt,Lt,Bt = G.Nt,G.Lt,G.Bt
    G.dTT,G.c,G.r = sum_vt([f for f in (Nt,Lt) if f])  # combined dTT? use m_func?
    spanR = np.mean([n.span for n in Nt.N_]) / np.mean([l.span for l in Lt.N_]) if Lt else 0
    coreM = (Nt.m * decay**spanR * Nt.c + Lt.m * Lt.c) / (Nt.c+Lt.c)
    G.m = coreM * (1-Bt.m)  # core mass * contour specificity
    nm_,lm_,bm_ = [m_func(tt[0]) for tt in (Nt.dTT,Lt.dTT,Bt.dTT)]
    G.mt_ = ((nm_* decay**spanR *Nt.c) + lm_*Lt.c) / (Nt.c+Lt.c) * (1-bm_)
    FV_(CoF.get(), G.dTT, G.c, G.r)
    return sG_

def cluster_P(_C_, root,_j): # multi-seed mean shift: parallel centroid refine, _C_ varies via split/merge

    iC_ = _C_; cnt = 0; Ln = len(N_:= list(set([N for C in _C_ for N in C.N_])))  # all Ns are in all Cs
    _m__ = np.zeros((Ln, len(_C_)))  # NxC, cols aligned to _C_
    for i, N in enumerate(N_):
        for c,m in N.root_:
            if c in _C_: _m__[i,_C_.index(c)] = m
    while True:
        for N in N_: N.root_ = []  # reset, append in sum2F
        Lc = len(_C_); totL = Lc*Ln  # loop Lc,L,m__
        m__ = np.zeros((Ln,Lc))
        for j,N in enumerate(N_):
            for i,C in enumerate(_C_): m__[j,i] = val_(base_comp(C,N,_j,fder=0)[0],cPw_)
        C_ = [sum2F(N_, root, m__[:,i],nF='C',typ=3) for i in range(Lc)]  # mean shift, aligned to m__, add typ?
        removed = []
        posL = (m__ > ave).sum()
        if gv_(posL*wcP - ave*(root.r+ccP*totL)):  # merge redundant Cs
            for i,_C in enumerate(C_):
                if _C in removed: continue
                for C in C_[i+1:]:
                    if C in removed: continue
                    m = val_(base_comp(C,_C,_j,fder=0)[0],cPw_); r = (C.r+_C.r)
                    if m*wF > ave*(r+cF):
                        add2F(_C,C,2); removed += [C]; _C.m = val_(_C.dTT,cPw_)  # for next-loop base comp
                        for N in C.N_:
                            for rt in N.root_:
                                if rt[0] is C: rt[0] = _C  # update root from C to _C, for olp computation below
        cnt += 1
        dM = (m__-_m__).mean() if m__.shape==_m__.shape else 0  # convergence, may be negative
        if C_ and dM * (wcP / (cnt+root.r+ccP)) > ave:  # next loop, tentative
            _C_ = []; i_ = []
            for i, _C in enumerate(C_):
                if _C in removed: continue
                _C.olp = sum(rt[1] for n in _C.N_ for rt in n.root_ if rt[0] is not _C and rt[1] > _C.m)
                if _C.m * wcP > ave * (_C.r + _C.olp + ccP):  # prune
                    _C_ += [_C]; i_ += [i]  # also after merge and converge
            _m__ = m__[:,i_]
        else:
            if dM<0: C_ = _C_  # revert to better prior
            else: i_ = [i for i,C in enumerate(C_) if C not in removed]; _m__ = m__[:,i_]; C_ = [C_[i] for i in i_]
            break
    out_ = []
    for N in N_: N.root_ = []  # replace with out_ Cs:
    for i, C in enumerate(C_):
        if C.m > ave * C.r:  # prune, add olp as stronger ms?
            N_,m_ = [],[]
            for N, m in zip(C.N_, _m__[:,i]):
                if m*N.c > ave*N.r: N_+=[N]; m_+=[m]
            if N_:
                oC = sum2F(N_,root, m_,nF='C', typ=4)
                out_ += [oC]
    if out_:
        iTT, iC, iR = sum_vt(iC_); oTT,oC,oR = sum_vt(out_)
        FV_(CoF.get(), iTT-oTT, iC-oC, iR-oR)
    return out_

