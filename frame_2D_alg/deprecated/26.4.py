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

def sum2F(N_, root, fr=0, merge=0, froot=0):  # -> CF/CN

    _C=root.c; n=N_[0]; C=n.c; TT,R,wC = np.zeros((2,9)),0,0
    vt_ = [[n.rTT if fr else n.dTT], n.c, n.r, n.wc, n.H]
    for n in N_:
        vt_ += [[n.rTT if fr else n.dTT], n.c, n.r, n.wc, n.H]
        if merge: root.N_ += n.N_ if merge>1 else [n]
        if froot: n.root = root
    for tt,c,r,wc,H in N_:
        rc = c/C; TT += tt*rc; R+=r*rc; wC+=wc*rc; add_H(root.H,H, root, rc)
    new = CF(N_=N_, dTT=TT, c=C, r=R/C, wc=wC, H=H)
    if _C: add2F(root,new)
    else: root=new
    if getattr(root,'C_',None):  # same for L_,H, etc?
        root.Ct = sum2F(root.C_,CF(root=root))  # external init R, setattr(root, nF,R)
    return root

def sum_H(N_, Ft, rc_):
    H = []
    for N in N_:
        if H: H[0]+= N.N_  # top lev
        elif  N.N_: H = [list(N.N_)]
        if hasattr(N, 'Nt'): add_H(Ft.H, N.Nt.H, Ft)
    Ft.H = [sum2F(lev, CF(), rc_) for lev in H] if H else []




