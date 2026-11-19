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
