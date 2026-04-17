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



