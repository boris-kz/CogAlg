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
