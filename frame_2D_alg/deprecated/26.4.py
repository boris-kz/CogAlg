def ffeedback(frame):  # adjust filters: all aves *= rV, ultimately differential backprop per ave?

    def init_C_(N_):
        typ_ = []  # group calls by nF type
        for oF in N_:
            tF = next((fork for fork in typ_ if fork.nF==oF.nF), None)
            if tF: tF.N_ += [oF]  # calls / function type
            else:  typ_ += [CF(nF=oF.nF, N_=[oF])]
        return typ_
    # obsolete, there is no Z.Nt.H, just flat Z.call_( indiv oF.call_s)?
    sum2F(Z.call_,'Nt', Z)  # fset Z.Nt, sum nested calls in H
    Z.Ct.H = [init_C_(lev.N_) for lev in Z.Nt.H]  # maps to Z.H
    H, _prom_ = [],[]; r = frame.r+1
    for lev in Z.Ct.H:  # bottom-up merge nested typ_F calls? it's flat at his point
        for tF in _prom_:
            _tF = next((fork for fork in tF.N_ if fork.nF==tF.nF), None)
            if _tF: Q2R(list(set(_tF.N_+tF.N_)), _tF, merge=0,froot=0)
            else:   lev += [tF]
        H+= [Q2R([tF for tF in lev if tF.m > ave*r], froot=0) if lev else CF()]
        _prom_ = [tF for tF in lev if tF.m > ave*(r+1)]  # promote to next level
        r += 1
    Z.Ct.H = H  # lower levs
    Z.Ct.N_ = init_C_(Z.N_ + [oF for tF in _prom_ for oF in tF.N_])  # top lev

    Q2R(Z.Ct.N_, Z.Ct, froot=0)
    # sum ratios between consecutive-level TTs:
    _N,_C,_n,_c = frame.dTT, frame.Ct.dTT, frame.TTn, frame.TTc; rTT_ = [np.ones((2,9)) for _ in range(4)]
    for lev in frame.H:
        # top-down frame expansion levels, not lev-selective or sub-lev recursive
        N,C,n,c = lev.dTT, lev.Ct.dTT, lev.TTn, lev.TTc
        rN, rC, rn, rc = [np.abs(_tt / eps_(tt)) for _tt,tt in zip((_N,_C,_n,_c), (N,C,n,c))]
        for rTT,rtt in zip(rTT_,(rN,rC,rn,rc)): rTT += rtt
        _N,_C,_n,_c = N,C,n,c
    rM = rD = 0
    for i, rTT in enumerate(rTT_):
        rm, rd = vt_(rTT, frame.r, wTT_[i]); rM+=rm; rD+=rd
    return rM+rD, rTT_

class CoF(CF):  # oF/ code fork, Ct: types, dTT=results, w=m or separate sum wTT?
    name = "func"
    _cur = contextvars.ContextVar('oF')
    def __init__(f, **kw):
        super().__init__(**kw)
        f.call_ = kw.get('call_',[])  # function call trace, nest by functions in ffeedback
    @staticmethod
    def get(): return CoF._cur.get(Z)
    @staticmethod
    def sum_w(self):
        F = CoF
        if F.call_: F.w = sum([call.w for call in F.call_])  # all calls must terminate
        else:
            TT = getattr(F,'rTT', CoF.dTT)
            (m, d) = vt_(TT, F.r) if np.any(TT) else (ave, avd)
            F.w = abs(m) / (abs(d) or eps)  # 0:inf, mean~1
    @staticmethod
    def traced(func):  # _oF.call_ += call oF
        if getattr(func, 'wrapped', False): return func
        @wraps(func)
        def inner(*a, **kw):
            _CoF = CoF._cur.get()  # not for Z
            oF = CoF(nF=func.__name__, root=_CoF)
            _CoF.call_ += [oF]
            tF = next((f for f in Z.N_ if f.nF==oF.nF), None)  # Z.N_ is not used for data: redundant to frame, same for cross_comp oF?
            if tF: tF.N_ += [oF]  # typ_, top-down in data depth?
            else:  Z.N_ += [CF(nF=oF.nF, N_=[oF])]  # group calls by function name
            CoF._cur.set(oF); result = func(*a, **kw)
            CoF._cur.set(_CoF)
            return result
        inner.wrapped = True
        return inner

def cross_comp(Ft, rr, nF='Nt'):  # core function mediating recursive rng+ and der+ cross-comp and clustering

    N_,G_ = Ft.N_,[]; fC = N_[0].typ==3  # rc=rdn+olp, comp N_|B_|C_:
    L_, TT,c,r,TTd,cd,rd = comp_C_(N_,rr,fC=1) if fC else comp_N_(combinations(N_,2),rr)
    if L_:  # Lm_, no +|- Ft.Lt?
        M, D = vt_(TT, r, TTw(Ft.root,fC+2))
        if M * ((len(L_)-1)*Lw) * wn > ave:  # global cw,nw,Nw,Cw / ffeedback
            # vs TTn = root.Lt.dTT: +ve links between initial nodes, does not represent -ves?
            setattr( Ft.root,('TTn','TTc')[fC], TT+TTd)  # comp->TT, clust->dTT, compression = TT-dTT, in sum2G
            E_ = get_exemplars({N for L in L_ for N in L.nt}, r)
            G_,r = cluster_N(Ft, E_, r)  # -> cluster_C, _P
            if G_:
                Ft = sum2F(G_, nF, Ft.root); fC = G_[0].N_[0].typ==3  # not sure, G.N_= spliced C_ if more valuable?
                if val_(TT*wn, r, TTw(Ft.root,fC+2), (len(G_)-1)*Lw,1,TTd, rd/(r+rd)) > 0:
                    G_,r = cross_comp(Ft,r,nF)  # agg+, trans-comp
    return G_, r  # G_ is recursion flag

conditional_ = ['add_H','add_Lt','cluster_C','cluster_N','cluster_P','comp_F','ffeedback','frame_H','get_exemplars','mdecay','proj_N','proj_focus','proj_TT',
                'sum2G','sum_H','trace_edge','Q2R','add_Nt','cent_TT','comp_N','comp_N_','cross_comp','sum2C','sum2F']

def sum_crw(F, call_=None):  # no sum wTT?
    N_ = call_ if call_ is not None else F.typ_
    F.c = C = sum(n.c for n in N_); F.r = sum(f.r* (f.c/C) for f in N_); F.w = sum(f.w* (f.c/C) for f in N_)

def Q2R(N_, R=None, merge=1, froot=1, fN=0, fr=0):  # update root with N_

    if R is None: R = CN() if fN else CF()  # root
    R.m, R.d, R.dTT, R.c, R.r = sum_vt(N_, fr,fm=1)  # fr: val = local surprise
    if merge:
        for N in [n for N in N_ for n in N.N_] if merge==3 else (N_[1].N_ if merge == 2 else N_):  # 2: pair merge
            R.N_ += [N]
            if froot: N.root=R
            if fN: R.C_ += N.C_; R.TTn += N.TTn; R.TTc += N.TTc  # frame_H only?
    if fN and R.C_: R.Ct = Q2R(R.C_)
    return R

def add2F(F, n, fr=0, merge=0):

    a = 'rTT' if fr else 'dTT'  # or wTT?
    if F.c:
        C=F.c+n.c; _rc,rc = F.c/C, n.c/C; F.r = F.r*_rc + n.r*rc; F.c = C
        if hasattr(n,'wc'): F.wc = (F.wc*_rc + n.wc*rc) if F.wc else n.wc
        setattr(F, a, getattr(F,a)*_rc + getattr(n,a)*rc)
    else: setattr(F,a,getattr(n,a)); F.c=n.c; F.r=n.r
    if merge: F.N_ += n.N_ if merge>1 else [n]
    if getattr(n,'H',None): add_H(F.H, n.H, F)
    if hasattr(n,'C_'): F.C_ = getattr(F, 'C_', []) + n.C_  # same for L_?

def sum2F(N_, root=None, m_=[],d_=[], merge=0, froot=0):  # -> CF/CN

    c_ = np.array([n.c for n in N_], dtype=float); C = c_.sum(); w_ = c_/C
    for i, (n,w) in enumerate(zip(N_,w_)):
        if i: TT+=n.dTT*w; R+=n.r*w; n_ += n.N_ if merge>1 else [n] if merge else []
        else: TT = n.dTT*w; R=n.r*w; n_=[]
    if not merge: n_ = N_
    F = CF(N_=n_, dTT=TT, c=C, r=R)
    if np.any(m_): F.typ=3; F.m_,F.d_=m_,d_; F.m,F.d=sum(m_),sum(d_); F.kern=n.kern*w
    else:  F.m, F.d = vt_(TT)
    root = add2F(root,F) if root!=None else F
    if froot:
        for n in N_: n.root=root
    sum_H(N_, root)
    if getattr(root,'C_',None):
        root.Ct = sum2F(root.C_, CF(root=root))
    return root

def sum_H(N_, Ft, rc_):
    H = []
    for N in N_:
        if H: H[0]+= N.N_  # top lev
        elif  N.N_: H = [list(N.N_)]
        if hasattr(N, 'Nt'): add_H(Ft.H, N.Nt.H, Ft)
    Ft.H = [sum2F(lev, CF(), rc_) for lev in H] if H else []

def cluster_P(__C_, N_, _c, root):  # Parallel centroid refining, _C_ from cluster_C, N_= root.N_, if global val*overlap > min

    for N in N_: N._m_,N._o_,N._d_,N._r_ = [],[],[],[]
    for C in __C_: C.rN_= N_  # soft assign all Ns per C
    _C_=__C_; cnt = 0
    while True:
        M = O = dM = dO = 0
        for N in N_:
            N.m_,N.d_,N.r_,N.rN_ = map(list,zip(*[vt_(base_comp(C,N)[0],C.r,wTTC) + (C.r,C) for C in _C_]))  # distance-weight match?
            N.o_ = list(np.argsort(np.argsort(N.m_)[::-1])+1)  # rank of each C_[i] = rdn of C in C_
            dM+= sum(abs(m-_m) for _m,m in zip(N._m_,N.m_)) if cnt else sum(N.m_)
            dO+= sum(abs(o-_o) for _o,o in zip(N._o_,N.o_)) if cnt else sum(N.o_)
            M += sum(N.m_); O += sum(N.o_)
        C_ = [sum2C(N_,_C, i, root=root) for i, _C in enumerate(_C_)]  # update with new coefs, _C.m, N->C refs
        Ave = ave * (root.r+wC)
        cnt += 1
        if M > Ave*O and dM > Ave*dO:  # strong update
            for N in N_: N._m_,N._o_,N._d_,N._r_ = N.m_,N.o_,N.d_,N.r_
            _C_ = C_
        else: break
    out_ = []
    for i, C in enumerate(C_):
        keep = C.m > ave*(wC+C.r)
        if keep:
            C.N_ = [n for n in C.N_ if n.m_[i] * C.m > ave * n.r_[i] * n.o_[i]]
            keep = bool(C.N_)
        out_ += [C if keep else []]
    for N in N_:
        for _v_,v_ in zip((N._m_,N._d_,N._r_,N._o_), (N.m_,N.d_,N.r_,N.o_)):
            v_[:] = [v for v,c in zip(v_,out_) if c]; _v_[:] = []
        N.rN_ = [c for c,keep in zip(N.rN_,out_) if keep]

    C_ = [c for c in out_ if c]
    dCt = sum2F(set(__C_)-set(C_),CF())  # compression
    oF = CoF.get(); oF.N_=C_; oF.rTT=dCt.dTT; oF.r = dCt.r; oF.c = _c - dCt.c  # data
    return C_  # or out_?

def cluster_C(Ft, E_, r,_c):  # form centroids by clustering exemplar surround via rims of new member nodes, within root

    oF = CoF.get(); oF.c += _c; oF.r += r  # revert if no clustering
    C_,_C_ = [],[]  # form root.Ct, may call cross_comp-> cluster_N, incr rc
    for n in Ft.N_: n._C_,n.m_,n._m_,n.o_,n._o_,n.rN_ = [],[],[],[],[],[]
    for E in E_:
        C = Copy_(E,Ft,init=1,typ=3)  # all rims in root, sequence along eigenvector?
        for TT, wTT, ww in zip([C.dTT,C.Ct.dTT,C.TTn,C.TTc], C.wTT_, [wN,wC,wn,wc]):
            wTT[:] = cent_TT(TT,r) * ww
        C._N_ = list({n for l in E.rim for n in l.nt if (n is not E and n in Ft.N_)})
        C._L_ = set(E.rim)  # init peer links
        for n in C._N_+C.N_: n._m_+=[C.m*(n.c/C.c)]; n._o_+=[1]; n._C_+=[C]
        _C_ += [C]
    oC_ = []  # output stable Cs
    while True:  # reform C_, add direct in-C_ cross-links for membership?
        C_,cnt,olp, mat,dif, DTT,Dm,Do = [],0,0,0,0,np.zeros((2,9)),0,0; Ave = ave*(r+wn); Avd = avd*(r+wn)
        _Ct_ = [[c, c.m/c.c, c.r] for c in _C_]
        for cr, (_C,_m,_o) in enumerate(sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True),start=1):
            if _m > Ave *_o:
                L_, N_,N__,m_,o_,M,D,O,cc, dTT,dm,do = [],[],[],[],[],0,0,0,0, np.zeros((2,9)),0,0  # /C
                for n in set(_C.N_+_C._N_):  # current + frontier
                    dtt,_ = base_comp(_C, n); cc+=1  # or comp_N, decay?
                    m,d = vt_(dtt, cr, wTTC); dTT += dtt; m *= n.c  # rm,olp / C
                    odm = np.sum([_m-m for _m in n._m_ if _m>m])  # higher-m overlap
                    oL_ = set(n.rim) & _C._L_  # replace peer rim overlap with more precise m
                    if oL_: _m,_d = sum_vt(oL_,fm=1)[:2]; m+=_m; d+=_d  # abs m?
                    m_+=[m]; o_+= [odm]  # from all comps
                    M += m; D += abs(d)
                    if m > 0 and m > Ave * odm:
                        N_+=[n]; L_+=n.L_; O+=odm  # convergence val
                        for _n in [_n for l in n.rim for _n in l.nt if _n is not n]:
                            if not hasattr(_n,'_m_'): _n._C_,_n.m_,_n._m_,_n.o_,_n._o_,_n.rN_ = [],[],[],[],[],[]
                            N__ += [_n]  # +|-Ls
                        if _C not in n._C_: dm+=m; do+=odm  # not in extended _N__
                    else:
                        if _C in n._C_: i = n._C_.index(_C); dm+=n._m_[i]; do+=n._o_[i]
                DTT+=dTT; mat+=M; dif+=D; olp+=O; cnt+=cc
                if M > Ave*O and val_(dTT, cr+O, TTw(_C,1),(len(N_)-1)*Lw) > 0:  # dTT * wTTC is more precise?
                    C = sum2C(N_,root=Ft)
                    for n,m,o in zip(N_,m_,o_):
                        n.rN_ += [C]; n.m_+=[m]; n.o_+=[o]
                    C._N_ = list(set(N__)-set(N_))  # new frontier
                    C._L_ = set(L_)  # peer links
                    if D< Avd*O: oC_+= [C]  # output if stable, actually if val_(DTT, fi=0) + D?
                    else:        C_ += [C]  # reform
                else:
                    for n in _C._N_+_C.N_:
                        n.exe = n.m/n.c > 2 * ave
                        for i, c in enumerate(n.rN_):
                            if c is _C:  # remove _C-mapping m,o:
                                n.rN_.pop(i); n.m_.pop(i);n.o_.pop(i); break
                Dm+=dm; Do+=do
            else: break  # the rest is weaker
        for n in Ft.N_:
            n._C_ = n.rN_; n._m_= n.m_; n._o_= n.o_; n.rN_,n.m_,n.o_ = [],[],[]  # new n.Ct.N_s, combine with v_ in Ct_?
        if oC_+C_ and mat*dif*olp > ave*wC*2:  # if val_(DTT,len(oC_+C_)?
            oC_+= C_; cc = sum([f.c for f in oC_])
            oC_ = cluster_P(oC_, Ft.N_, cc, Ft)  # refine all memberships in parallel by global backprop|EM
            break
        if Do and Dm/Do > Ave: _C_=C_  # dval vs. dolp: overlap increases with Cs expansion
        else: oC_ += C_; break  # converged
    if oC_:
        for n in [N for C in oC_ for N in C.N_]:  # exemplar V + sum n match_dev to Cs, m* ||C rvals:
            n.exe = (n.d if n.typ==1 else n.m) + np.sum([m-ave*o for m, o in zip(n.m_, n.o_)]) - ave
        if val_(DTT, r+olp, TTw(Ft.root,1), (len(oC_)-1)*Lw) > 0:
            Ct = sum2F(oC_, Ft.root.Ct)  # ?
            _, r = cross_comp(Ct,r)  # all distant Cs, seq C_ in eigenvector = argmax(root.wTT)?
            sel_TT = (Ct.dTT*Ct.c - Ft.dTT*Ft.c) / eps_(Ct.dTT * Ct.c)
            oF = CoF.get(); oF.N_=oC_; oF.rTT=sel_TT; oF.r=Ct.r; oF.c = Ft.c-Ct.c  # data, select in Nt.N_?
    return oC_,r

def sum2C(N_, _C, u_=None, root=None):  # fuzzy sum + base attrs for centroids

    uc_ = []  #  N/C contribution = c-scaled u_
    for i, N in enumerate(N_):
        if u_ is None:   # cluster_C, fdiv m_, proj C survival?
            i = N._C_.index(_C); m,o = N._m_[i],N._o_[i]; uc_ += [N.c * (m/(ave*o))]  # rational overlap?
        else: uc_ += [N.c * u_[i]]  # N/C contribution
    U = sum(uc_)
    R = 0; TT = np.zeros((2,9)); kern = np.zeros(4); span = 0; yx = np.zeros(2)
    for N, uc in zip(N_,uc_):
        rc = uc/U; TT+= N.dTT*rc; kern+= N.kern*rc; span+= N.span*rc; yx+= N.yx*rc; R+= N.r*rc
    wTT = cent_TT(TT, R) * Fw_[cltC if _C else cltP]  # set param correlation weights
    m,d = vt_(TT,R, wTT)
    C = CN(typ=3, Nt= CF(N_=N_),dTT=TT,m=m,d=d,c=U,r=R, yx=yx, kern=kern,span=span, root=root, wTT=wTT)
    return C

# cmpN_, cmpC_, cmpN, cmpF, exem, cltN, cltC, cltP, xcmp, frmH, vctE, trcE, fbac, prjN = range(len(onF_))  # pre-call nF indices

def TTw(G, wTT): return wTT if G.wTT is None else G.wTT


def vt_(TT, r, wTT=wTT, fdiv=0):  # brief val_ to get m,d, rc=0 to return raw vals, Wn for comp_N

    m_,d_ = TT; ad_ = np.abs(d_); t_ = eps_(m_+ad_)  # ~ max comparand
    m = m_/t_ @ wTT[0]; d = ad_/t_ @ wTT[1]  # norm by co-derived
    if fdiv: m/= ave*r; d/= avd*r  # in 0-inf for summation
    else:    m-= ave*r; d-= avd*r  # in -1:1 without r
    return m,d

def val_(TT, r, wTT=wTT, mw=1.0,fi=1, _TT=None, cr=.5):  # m,d eval per cluster, cr = cd / cm+dc, default half-weight?

    t_ = eps_(TT[0] + np.abs(TT[1]))  # comb val/attr, match can be negative?
    rv = TT[0] / t_ @ wTT[0] if fi else TT[1] / t_ @ wTT[1]  # fork / total per scalar
    if _TT is not None:
        _t_ = eps_(np.abs(_TT[0]) + np.abs(_TT[1]))
        _rv = _TT[0] / _t_ @ wTT[0] if fi else _TT[1] / _t_@ wTT[1]
        rv  = rv * (1-cr) + _rv * cr  # + borrowed alt fork val, cr: d count ratio, must be passed with _TT?
    return rv*mw - (ave if fi else avd) * r

def frame_H(image, iY,iX, Ly,Lx, Y,X, rV, max_elev=4):  # all initial args set manually

    def base_tile(y,x):  # 1st level, higher levels get Tg s
        T = frame_blobs_root( comp_pixel( image[y:y+Ly, x:x+Lx]), rV)
        T = vect_edge(T, rV)  # form, trace PP_
        if T: cross_comp(T.Nt,T.r)
        return T

    def expand_lev(_iy, _ix, elev, T):  # seed tile is pixels in 1st lev, or Fg in higher levs

        frame = np.full((Ly, Lx), None, dtype=object)  # level scope
        iy,ix = _iy,_ix; cy,cx = (Ly-1)//2, (Lx-1)//2; y,x = cy,cx  # start at mean
        T_, PV__,C,R = [],np.zeros([Ly,Lx]),0,0  # maps to level frame
        while True:
            # each loop adds one tile to lev_frame
            if not elev: T = base_tile(iy, ix)
            if T and val_(T.dTT,T.r+cFrm+elev, T.wTT*ttFrm, mw=(len(T.N_)-1)*Lw) > 0:
                frame[y,x] = T; T_ += [T]
                dy_dx = np.array([T.yx[0]-y, T.yx[1]-x]); pTT = proj_N(T, np.hypot(*dy_dx), dy_dx, elev,T.c)
                if 0 < val_(pTT, elev, T.wTT*ttFrm) < ave:  # extend lev by combined proj T_
                    proj_focus(PV__,y,x, T)  # PV__+= pV__
                    pv__ = PV__.copy(); pv__[frame != None] = 0  # exclude processed
                    y, x = np.unravel_index(pv__.argmax(), PV__.shape)
                    if PV__[y,x] > ave:
                        iy = _iy + (y-cy) * Ly**elev; ix = _ix + (x-cx) * Lx**elev  # feedback to shifted coords
                        if elev:
                            T = frame_H(image, iy, ix, Ly, Lx, Y,X, rV, elev)  # up to current level
                    else: break
                else: break
            else: break
        if T_:
            TT,C,R = sum_vt(T_); R += elev
            if val_(TT, R/len(T_) + elev, ttFrm, mw=(len(T_)-1)*Lw) <= 0: T_=[]; C=0; R=0
        return T_,C,R
    global ave,avd, Lw,intw,distw, Fw_,FTT_
    (ave,avd, Lw,intw,distw), Fw_,FTT_ = np.array([ave,avd, Lw,intw,distw])/rV, np.array(Fw_)/rV, np.array(FTT_)/rV

    elev = 0; F, tile = [],[]  # frame, seed lower tile, if any
    while elev < max_elev:
        tile_,C,R = expand_lev(iY,iX, elev, tile); oF=CoF.get();oF.c += C; oF.r += R
        if tile_:  # sparse higher-scope tile, if expanded
            F = CN(box=np.array([0,0,Y,X]), yx=np.array([Y//2, X//2]))  # same center on all levels
            F.H = []; F.N_ = tile_  # or Nt = Q2R(tile_)?
            if elev: [add_H(F.H,T.H, F,fN=1) for T in tile_]
            F.H += [lev := sum2F([n for N in tile_ for n in N.N_], CF())]; add2F(F,lev)  # concat prior top lev
            if cross_comp(F.Nt, rr=elev)[0]:  # spec->tN_,tC_,tL_, proj comb N_'L_?
                elev += 1
                if rV > ave:  # not revised:
                    add_typ_(Z); dTT_ = [None for _ in range(len(onF_))]
                    for typ in Z.typ_: dTT_[typ.nF] = getattr(typ, 'rTT', typ.dTT)
                    for i, (dTT, Fw, Fc) in enumerate(zip(dTT_, Fw_, Fc_)):
                        if dTT is not None: FTT_[i] = cent_TT(F.dTT,2) * Fw  # max-scope correlation weights
                    if elev == max_elev:  # fb / top lev
                        rV, wTT_ = ffeedback(F,C,R)  # update globals, rV is not used?
                tile = F  # lev tile_ is next extension seed
            else: break
        else: break
    return F  # for intra-lev feedback

def add_typ_(oF):  # record oF vals for weighting
    call_, typ_ = flat_(oF), []
    for F in call_:
        tF = next((f for f in typ_ if f.nF == F.nF), None)
        if tF: tF.typ_ += [F]
        else: typ_ += [CoF(nF=F.nF, typ_=[F])]
    for tF in typ_: sum2F(tF.typ_,tF)
    oF.typ_ = [tF for tF in typ_ if tF.w * tF.c > ave]  # add coef
    if oF.typ_: sum2F(oF.typ_,oF)  # tVs replace all Vs?

def prune_C_(C_, root):
    out_ = []
    for i, C in enumerate(C_):
        if C.m > ave * C.r:  # final pruning, C vals are competitive
            N_,m_,d_ = [],[],[]
            for N, m, d in zip(C.N_,C._m_,C._d_):
                if m * N.c > ave * N.r: N_+= [N]; m_+= [m]; d_+= [d]
            if N_: out_+= [sum2C(N_,m_,d_,i, root=root, final=1)]
    return out_

def cluster_P1(_C_, _c, root):  # FCM-style parallel centroid refine, may add proj_C

    cnt = 0  # r = root.r+1?:
    N_ = list(set([N for C in _C_ for N in C.N_]))  # all Ns are in all Cs
    n_ = [0 for _ in N_]; Cm_,Cd_ = copy(n_),copy(n_)  # N_/C
    c_ = [0 for _ in _C_]  # C_/N, align with _C_
    for j, N in enumerate(N_):
        m_,d_ = copy(c_),copy(c_)  # C_/N vals
        for c,m,d in zip(N.c_,N.m_,N.d_):
            i =_C_.index(c); m_[i] = m; d_[i] = d
            Cm_[j] = [m]; Cd_[j] = [d]  # per C, aligned with N_
        N.m_, N.d_ = m_, d_  # reciprocal per N, aligned with C_
    while True:
        u__,d__, dU,dD = [],[],0,0  # fixed-length C_ and N_
        C_ = [sum2C(N_,u_,d_,i, root) for i,(u_,d_) in enumerate(zip(_u__,_d__))]  # same N_ * updated membership
        for N,_u_,_d_ in zip(N_,_u__,_d__):
            u_,d_ = [],[]
            for C in C_:
                TT,_ = base_comp(C,N); u,d = vt_(TT, ttcP); u_+=[u]; d_+=[d]
            s = sum(u_); u_ = [f/s for f in u_]  # per-N membership, Σ_u_=1: norm for cross-C redundancy, or for totals only?
            dU += sum(abs(u-_u) for u,_u in zip(u_,_u_))  # normalize?
            dD += sum(abs(d-_d) for d,_d in zip(d_,_d_))
            u__+=[u_]; d__+=[d_]
        cnt += 1
        if (dU+dD) * wcP > ave * (root.r+ccP) * len(N_):  # memberships change
            _u__= u__; _d__= d__
        else: break  # converged
    out_ = []
    for i, C in enumerate(C_):
        if C.m > ave * C.r:  # final pruning, C vals are competitive
            N_,m_,d_ = [],[],[]
            for N,m,d in zip(C.N_,C._m_,C._d_):
                if m * N.c > ave * N.r: N_+=[N]; m_+=[m]; d_+=[d]
            if N_: out_ += [sum2C(N_, m_, d_, i, root=root, final=1)]
    if out_:
        dCt = sum2F(list(set(_C_)-set(out_)),CF())  # compress
        oF = CoF.get(); oF.N_=C_; oF.rTT=dCt.dTT; oF.r=dCt.r; oF.c=_c-dCt.c
        return out_

def sum2C1(N_, m_,d_, i, root=None, final=0, wO=wcC):  # fuzzy sum + base attrs for centroids

    L_, mc_, dc_ = [],[],[]  # m_ * c_
    for j, (N,m,d) in enumerate(zip(N_,m_,d_)):
        mc_ += [N.c * m_[j if final else i]]  # N/C else C/N contribution
    M = sum(mc_)
    R = 0; TT = np.zeros((2,9)); kern = np.zeros(4); span = 0; yx = np.zeros(2)
    for N, mc in zip(N_,mc_):
        rc = mc/M; TT+= N.dTT*rc; kern+= N.kern*rc; span+= N.span*rc; yx+= N.yx*rc; R+= N.r*rc
    wTT = cent_TT(TT,R) * wO  # set param correlation weights
    m,d = vt_(TT,wTT)
    C = CN(typ=3, Nt= CF(N_=N_),dTT=TT,m=m,d=d,c=M,r=R, yx=yx, kern=kern,span=span, root=root, wTT=wTT, L_=L_)
    C.m_=m_; C.d_=d_
    for n,m,d in zip(C.N_,C.m_,C.d_): n.c_[i] = C; n.m_[i] = m; n.d_[i] = d  # mapping-C vals

    if final==1:  # this L could be from comp instead?
        for j, N in enumerate(N_):
            dTT, _ = base_comp(C, N)  # but C isn't constructed yet — order issue
            dy_dx = C.yx - N.yx; span = np.hypot(*dy_dx); m, d = vt_(dTT, ttcC)
            L = CN(typ=1,w=[N.c * m_[j]],nt=[C,N],dTT=dTT,m=m,d=d,c=N.c*m_[j],r=(C.r+N.r)/2,span=span,angl=np.array([dy_dx,1],dtype=object),yx=(C.yx+N.yx)/2,kern=(C.kern+N.kern)/2, root=C)
            L_ += [L]
    return C

def cluster_C1(Ft, E_,_r,_c):  # form centroids by clustering exemplar surround via rims of new member nodes, within root

    N_= Ft.N_; oF = CoF.get(); oF.c += _c; oF.r+=_r  # revert if 0 clusters?
    q = [None for _ in E_]
    for n in N_: n.c_=copy(q); n.m_=copy(q); n.d_=copy(q); n._c_=copy(q); n._m_=copy(q); n._d_=copy(q)
    _C_ = []
    for i,E in enumerate(E_):  # along eigenvector?
        C = Copy_(E, Ft,init=1,typ=3)
        C.N_,C.L_,C.m_,C.d_ = [E],[],[1],[0]
        E._c_[i], E._m_[i], E._d_[i] = C,1,0  # aligned, self m,d
        C._N_= list({n for l in E.rim for n in l.nt if (n is not E and n in N_)})  # init frontier
        _C_ += [C]
    out_ = []
    while True:  # reform C_
        C_, cnt,mat,dif,rdn,DTT,Up = [],0,0,0,0,np.zeros((2,9)),0; Ave = ave*(_r+ccC); Avd = avd*(_r+ccC)
        for i,_C in enumerate(_C_):  # C.m,d /rTT? sort/ sum(_C.m_)?
            if _C==None: continue  # to map i
            N__,n_,m_,d_,M,D,T,R,dTT,up = [],[],[],[],0,0,0,0, np.zeros((2,9)),0  # /C
            for n in _C.N_+_C._N_:  # current + frontier
                dtt,_ = base_comp(_C,n)  # or comp_N, decay?
                m,d = vt_(dtt,ttcC); dTT+=dtt
                n_+=[n]; m_+=[m]; d_+=[d]; c=n.c; T+=c; M+=m*c; D+=abs(d)*c; R+=n.r*c  # scale totals only?
                if _C in n._c_: up += abs(n._m_[i]-m) + abs(n._d_[i]-d)  # update
                else:           up += m+abs(d)  # not in extended _N__
            r = _r+ R/T  # loop-local, not ave?
            if M*(_C.w+ wC_*len(n_)+wC_*len(n_)) > Ave*(r+_C.r+ cC_*len(n_)):  # else: Up+= sum(_C._m_)+ sum([abs(d) for d in _C._d_])?
                for n in [_n for n in n_ for l in n.rim for _n in l.nt if _n is not n]:
                    N__ += [n]  # +|-Ls
                C = sum2C(n_,m_,d_, i, final=2, root=Ft)
                C._N_ = list(set(N__)- set(n_))  # new frontier
                if D<Avd: out_+=[C]; C_+=[None]  # output if stable, keep slot for alignment
                else:     C_ += [C]  # reform
                DTT+=dTT; mat+=M; dif+=D; cnt+=T; rdn+=R; Up+=up
        r = _r+ rdn/(cnt or eps)
        V_ = vQ(C_); L = len(out_+V_); olp = sum([len(N.c_) for N in N_])  # prioritize stronger?
        if (mat+dif)* (wcP*L) > Ave* (r+olp+ ccP*L):
            out_+=V_; T=sum([f.c for f in out_])
            out_ = cluster_P(out_,T, Ft)  # refine all memberships in parallel by global backprop
            break
        if Up*(wcC*len(V_)) > Avd*(r+(ccC*len(V_))):  # do next loop
            for C in V_: C._m_ = C.m_; C._d_ = C.d_  # not aligned, merge frontier C_ += C._N_'c__?
            for n in N_: n._c_ = n.c_; n._m_ = n.m_; n._d_ = n.d_
            q = [None for _ in _C_]  # last loop
            for n in set([_n for _C in _C_ for _n in _C.N_+_C._N_]): n.c_=copy(q); n.m_=copy(q); n.d_=copy(q)  # fill /C index
            _C_ = C_
        else: out_+=V_; break  # converged
    if out_:
        for n in [N for C in out_ for N in C.N_]:  # exemplar V + sum n match_dev to Cs, m* ||C rvals:
            n.exe = (n.d if n.typ==1 else n.m) + np.sum(vQ(n.m_)) - ave
        L= len(out_)
        if val_(DTT,r, Ft.root.wTT*ttcC, (len(out_)-1)*Lw) > 0:
            Ct = sum2F(out_,Ft.root.Ct)
            if not Ft.root.Ct: Ft.root.Ct = Ct; Ct.root = Ft.root
            _,r = cross_comp(Ct,r)  # all distant Cs, seq C_ in eigenvector = argmax(root.wTT)?
            sel_TT = (Ct.dTT*Ct.c - Ft.dTT*Ft.c) / eps_(Ct.dTT * Ct.c)
            oF = CoF.get(); oF.N_=out_; oF.rTT=sel_TT; oF.r=Ct.r; oF.c = Ft.c-Ct.c  # data, select in Nt.N_?
    return out_,r

def vQ(Q): return [i for i in Q if i is not None]  # valid Cs or vals

def val_1(TT, r, wTT=wTT, mw=1.0,fi=1, _TT=None, cr=.5):  # m,d eval per cluster, cr = cd / cm+dc, default half-weight?

    t_ = eps_(TT[0] + np.abs(TT[1]))  # comb val/attr, match can be negative?
    rv = TT[0] / t_ @ wTT[0] if fi else TT[1] / t_ @ wTT[1]  # fork / total per scalar
    if _TT is not None:
        _t_ = eps_(np.abs(_TT[0]) + np.abs(_TT[1]))
        _rv = _TT[0] / _t_ @ wTT[0] if fi else _TT[1] / _t_@ wTT[1]
        rv  = rv * (1-cr) + _rv * cr  # + borrowed alt fork val, cr: d count ratio, must be passed with _TT?
    return rv*mw - (ave if fi else avd) * r

def sum2C2(N_, m_,d_, root=None, final=0, wO=wcC):  # fuzzy sum + base attrs for centroids

    a = 'rTT' if fr else 'dTT'
    c_ = np.array([n.c for n in N_], dtype=float); C = c_.sum(); rc_ = c_/C
    TT = np.einsum('i,ijk->jk', rc_, np.stack([getattr(n,a) for n in N_]))  # weighted sum
    R  = rc_ @ np.array([n.r  for n in N_])  # wC = rc_ @ np.array([getattr(n,'wc',1) for n in N_])

    TT= np.zeros((2,9)); kern= np.zeros(4); yx= np.zeros(2); T= R= span= 0
    M = sum(m_) or eps  # C-normalizer
    A = np.zeros(2)  # for base_comp
    for N, m in zip(N_,m_):
        w = m/M; TT+= N.dTT*w; kern+= N.kern*w; span+= N.span*w; yx+= N.yx*w; R+= N.r*w; A += N.angl[0]; T+=N.c  # not weighted
    wTT = cent_TT(TT,R)* wO; m,d = vt_(TT,wTT)  # set param correlation weights
    C = F2N(CF())(typ=3, N_=N_,dTT=TT,m=m,d=d,c=T,r=R, root=root, wTT=wTT, yx=yx,kern=kern,span=span)
    C.angl = np.array([A, np.sign(C.dTT[1] @ ttcN[1])], dtype=object); C.m_=m_; C.d_=d_
    for n,m,d in zip(N_,m_,d_):
        n.c_ += [C]; n.m_ = [m]; n.d_ = [d]  # mapping-C vals
    if final:
        C.L_ = []
        for N in N_:
            dy_dx = C.yx-N.yx; span = np.hypot(*dy_dx); angl = np.array([dy_dx, np.sign(N.dTT[1] @ ttcN[1])], dtype=object)
            C.L_ += [CN(typ=1,nt=[C,N],dTT=N.dTT,c=N.c,r=N.r,m=N.m,d=N.d,span=span,angl=angl)]  # for proj_C?
    return C

class CoF1(CF):  # oF/ code fork, N_,dTT: data scope, w = vt_(wTT)[0]?
    name = "func"
    _cur = contextvars.ContextVar('oF')
    _W_,_C_ = np.zeros(len(onF_)),np.zeros(len(onF_))  # sum called oF weights and costs, global Fw_ is not summed
    def __init__(f, **kw):
        super().__init__(**kw)
        f.call_= kw.get('call_',[])  # tree
        f.typ_ = kw.get('typ_', [])  # flattened and nested with tFs
    @staticmethod
    def get(): return CoF._cur.get(Z)
    @staticmethod
    def traced(func):
        if getattr(func,'wrapped',False): return func
        @wraps(func)
        def inner(*a, **kw):
            _CoF = CoF._cur.get()
            oF = CoF(nF=onF_.index(func.__name__), root=_CoF); oF.wTT = FTT_[oF.nF]; oF.wc = Fc_[oF.nF]; _CoF.call_+=[oF]
            CoF._cur.set(oF); out = func(*a, **kw)
            if oF.call_:  # complete at this point
                if (len(flat_(oF))-1)*Lw > ave*(_CoF.wc+oF.r):
                    sum2F(oF.call_,oF)  # represents all nested call_s
            oF.wTT= cent_TT(getattr(oF,'rTT',oF.dTT), oF.r)  # rTT covers cluster compression
            oF.w += np.mean(wTT)  # default Fw_[oF.nF], * c, / r?
            _CoF._W_[oF.nF] += Fw_[oF.nF]  # nF is index in onF_
            _CoF._C_[oF.nF] += Fc_[oF.nF]
            CoF._cur.set(_CoF)
            return out
        inner.wrapped = True
        return inner
    def __bool__(f): return bool(f.call_)

def F2N(F):  # extend for cross_comp F.N_
    F.N_, F.L_, F.B_, F.C_, F.X_, F.rim = prop_F_('Nt'),prop_F_('Lt'),prop_F_('Bt'),prop_F_('Ct'),prop_F_('Xt'),prop_F_('Rt')
    F.Nt, F.Lt, F.Bt, F.Ct, F.Xt, F.Rt = sum2F(F.N_), CF(typ='Lt'), CF(typ='Bt'), CF(typ='Ct'), CF(typ='Xt'), CF(typ='Rt')
    F.compared = set(); F.fin = 0
    F.__class__ = CN

def add2F1(F, n, merge=0, fr=0):  # unpack for batching in sum2F

    a = 'rTT' if fr else 'dTT'  # or wTT?
    if F.c:
        C=F.c+n.c; _w,w = F.c/C, n.c/C; F.r = F.r*_w + n.r*w; F.c = C
        if isinstance(F,CoF) and isinstance(n,CoF): F.fw += n.m  # sum subtree gain, fixed cost, extensive: no weighting?
        setattr(F, a, getattr(F,a)*_w + getattr(n,a)*w)
    else:
        setattr(F,a,getattr(n,a)); F.c=n.c; F.r=n.r
        if isinstance(F,CoF) and isinstance(n,CoF): F.fw=n.m
    F.N_ += (n.N_ if merge else [n])
    if hasattr(F,'H') and getattr(n,'H',None): add_H(F.H, n.H, F)
    if hasattr(n,'C_'): F.C_ = getattr(F,'C_',[]) + n.C_  # same for L_?
    return F

def add_H(H,h, root, fN=0):
    # bottom-up:
    for Lev,lev in zip_longest(H, h):
        if lev:
            if Lev: add2F(Lev,lev,1)
            else: H.append((CopyF,Copy_)[fN](lev, root))

def sum_H1(N_, Ft):  # merge L.H to lev, if any
    H = []
    for N in N_:
        if H: H[0] += N.N_
        elif N.N_: H = [list(N.N_)]
    Ft.H = [sum2F(H[0], CF())] if H else []

def prop_F_(F):  # factory function to get and update top-composition fork.N_
    def get(N): return getattr(N,F).N_
    def set(N, new_N): setattr(getattr(N,F),'N_', new_N)
    return property(get,set)

