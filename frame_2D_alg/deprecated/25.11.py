class CH(CBase):  # nesting hierarchy or a level thereof

    name = "H"    # from top-composition = bottom derivation
    def __init__(n, **kwargs):
        super().__init__()
        n.dTT = kwargs.get('dTT',np.zeros((2,9)))  # m_,d_ [M,D,n, I,G,a, L,S,A]: single or sum H
        n.H   = kwargs.get('H',[])  # for nesting, empty if single layer represented by F_,dTT
        n.F_  = kwargs.get('F_',[])  # N_,B_,C_|Nt,Bt,Ct, each [n_,m,d,c, rc], empty if H
        n.rc  = kwargs.get('rc',0)  # complement to root.rc, for lev ranking
        n.m   = kwargs.get('m',0); n.d = kwargs.get('d',0); n.c = kwargs.get('c',0)  # to set rc
        n.root = kwargs.get('root',[])  # to pass vals?
        # n.depth = 0  # max nesting depth in H
    def __bool__(n): return bool(n.rc)

def comp_H(_H, H, rc, root, TT=None):  # unpack derH trees down to numericals and compare them

    spec = 1  # default
    if TT is None:  # recursive call, else dTT is passed from comp_N, _H.dTT and H.dTT are redundant
        TT = comp_derT(_H.dTT[1], H.dTT[1]*rc)
        if not _H.H and H.H and val_(TT, rc, mw=(1-min(len(H),len(_H)) *Lw)) < 0:
            spec = 0  # spec eval for recursive call only, 2 or more levs/H, different Lw for H?
    dH = []
    if spec:
        for Lev,lev in zip(_H.H, H.H):
            tt = np.zeros((2,9)); fork_ = [[],[],[]]  # or 6?
            for i, (F,f) in enumerate(zip_longest(Lev, lev)):
                if F and f:
                    # add comp_derT(fork dTTs), spec eval?
                    N_,L_,mTT,B_,dTT = comp_N_(F[0],rc,f[0]) if i<2 else comp_C_(F[0],rc,f[0])
                    fTT = mTT + dTT; tt += fTT
                    fork_[i] = [[N_,fTT]]  # do we need B_,L_ per fork?
            TT += tt; dH += [[fork_,tt]]
            # add fork_,dH sort and rc assign
    return CN(H=dH, dTT=TT, root=root, rc=rc, m=sum(TT[0]), d=sum(TT[1]), c=min(_H.c, H.c))

def agg_frame(foc, image, iY, iX, rV=1, wTTf=[], fproj=0):  # search foci within image, additionally nested if floc

    if foc:
        dert__ = image  # focal img was converted to dert__
    else:
        dert__ = comp_pixel(image)  # global
        global ave, Lw, intw, compw, centw, contw, adist, amed, medw, mW, dW
        ave, Lw, intw, compw, centw, contw, adist, amed, medw = np.array([ave, Lw, intw, compw, centw, contw, adist, amed, medw]) / rV
        # fb rws: ~rvs
    nY, nX = dert__.shape[-2] // iY, dert__.shape[-1] // iX  # n complete blocks
    Y, X = nY * iY, nX * iX  # sub-frame dims
    frame = CN(box=np.array([0, 0, Y, X]), yx=np.array([Y // 2, X // 2]))
    dert__ = dert__[:, :Y, :X]  # drop partial rows/cols
    win__ = dert__.reshape(dert__.shape[0], iY, iX, nY, nX).swapaxes(1, 2)  # dert=5, wY=64, wX=64, nY=13, nX=20
    PV__ = win__[3].sum(axis=(0, 1)) * intw  # init prj_V = sum G, shape: nY=13, nX=20
    aw = contw*10
    while np.max(PV__) > ave * aw:  # max G * int_w + pV, add prj_V foci, only if not foc?
        # max win index:
        y, x = np.unravel_index(PV__.argmax(), PV__.shape)
        PV__[y, x] = -np.inf  # to skip?
        if foc:
            Fg = frame_blobs_root(win__[:, :, :, y, x], rV)  # [dert, iY, iX, nY, nX]
            vect_edge(Fg, rV, wTTf); Fg.L_ = []  # focal dert__ clustering
            cross_comp(Fg, rc=frame.rc)
        else:
            Fg = agg_frame(1, win__[:, :, :, y, x], wY, wX, rV=1, wTTf=[])  # use global wY,wX in nested call
            if Fg and Fg.L_:  # only after cross_comp(PP_)
                rV, wTTf = ffeedback(Fg)  # adjust filters
                Fg = cent_attr(Fg,2)  # compute Fg.wTT: correlation weights in frame dTT
                wTTf *= Fg.wTT; mW = np.sum(wTTf[0]); dW = np.sum(wTTf[1])
                wTTf[0] *= 9 / mW; wTTf[1] *= 9 / dW
                # re-norm weights
        if Fg and Fg.L_:
            if fproj and val_(Fg.dTT,Fg.rc+compw, mw=(len(Fg.N_)-1)*Lw, ):
                pFg = proj_N(Fg, np.array([y, x]))
                if pFg:
                    cross_comp(pFg, rc=Fg.rc)
                    if val_(pFg.dTT,pFg.rc+contw,mw=(len(pFg.N_)-1)*Lw):
                        proj_focus(PV__, y,x, Fg)  # += proj val in PV__
            # no target proj
            frame = add_N(frame, Fg, fmerge=1, froot=1) if frame else Copy_(Fg)
            aw *= frame.rc

    if frame.N_ and val_(frame.dTT, frame.rc+compw, mw=(len(frame.N_)-1)*Lw) > 0:
        # recursive xcomp
        Fn = cross_comp(frame, rc=frame.rc + compw)
        if Fn: frame.N_ = Fn.N_; frame.dTT += Fn.dTT  # other frame attrs remain
        # spliced foci cent_:
        if frame.C_ and val_(frame.C_[1], frame.rc+centw, mw=(len(frame.N_)-1)*Lw, ) > 0:
            Fc = cross_comp(frame, rc=frame.rc + compw, fC=1)
            if Fc: frame.C_ = [Fc.N_, Fc.dTT]; frame.dTT += Fc.dTT
        if not foc:
            return frame  # foci are unpacked

def proj_H(cH, cos_d, dec):
    pH = CN()
    if cH.H:  # recursion
        for lay in cH.H:
            play = proj_H(lay, cos_d, dec); pH.H += [play]; pH.dTT += play.dTT
    else:   # proj terminal dTT
        pH.dTT = np.array([cH.dTT[0] * dec, cH.dTT[1] * cos_d * dec])
        # same dec for M and D?
    return pH  # m = val_(pH.dTT), scale by rV of norm decay M, no effect on D?

def proj_N(N, dist, A):  # recursively specified N projection, rim proj is currently macro in comp_N_?

    rdist = dist / N.span   # internal x external angle:
    cos_d = (N.angl[0].dot(A) / (np.hypot(*N.angl[0]) * dist)) * N.angl[1]  # N-to-yx alignment
    m,d = N.m,N.d  # tentative
    dec = rdist * (m / (m+d))  # match decay rate, * ddecay for ds?
    H = proj_H(N.derH, cos_d, dec)
    iV = val_(N.dTT, contw, mw=(len(N.N_)-1)*Lw)  # rc = contw?
    pH = copy_(H, N)
    if N.L_:  # from terminal comp
        LH = proj_H(N.dLay, cos_d, dec); add_H(pH,LH)
        eV = val_(LH.dTT, contw, mw=(len(N.L_)-1)*Lw)
    else: eV = 0
    if iV + eV > ave:
        return CN(N_=N.N_,L_=N.L_, dTT=pH.dTT, derH=pH)
    LH,L_,H,N_ = CN(),[],CN(),[]
    if not N.root and eV * ((len(N.L_)-1)*Lw) > specw:  # only for Fg?
        for l in N.L_:  # sum L-specific projections
            lA = A - (N.yx-l.yx); ldist = np.hypot(*lA)
            pl = proj_N(l,ldist,lA)
            if pl and pl.m > ave: add_H(LH,pl.derH); L_+= [pl]  # or diff proj, if D,avd?
        eV = val_(LH.dTT, contw, mw=(len(L_)-1)*Lw)
    if iV * ((len(N.N_)-1)*Lw) > specw:
        for n in (N.N_ if N.fi else N.nt):  # sum _N-specific projections
            if n.derH:
                nA = A - (N.yx-n.yx); ndist = np.hypot(*nA)
                pn = proj_N(n,ndist,nA)
                if pn and pn.m > ave: add_H(H,pn.derH); N_+= [pn]
        iV = val_(H.dTT, contw, mw=(len(N_)-1)*Lw)
    if iV + eV > 0:
        if L_ or N_: pH = add_H(H,LH)  # recomputed from individual Ls and Ns
        return CN(N_=N_,L_=L_, dTT=pH.dTT, derH=pH)

def comp_sub(_N,N, rc, root, TT):  # unpack node trees down to numericals and compare them

    _H, H = _N.H, N.H; dH,Nt,Bt,Ct = [],[],[],[]; C = 0
    if _H and H:
        for _lev,lev in zip(_H, H):
            C += min(_lev.c, lev.c)
            TT += comp_derT(_lev.dTT[1], lev.dTT[1]*rc)  # default
            if val_(TT,rc) > 0:  # sub-recursion
                dH = comp_sub(_lev,lev, rc,root, TT)
        c = C / min(len(_H),len(H))
    else:
        # comp fork_s, | H[0],fork_?
        tt = np.zeros((2,9)); dfork_ = [[],[],[]]  # 6 in H only?
        for i, (F,f) in enumerate(zip((_N.Nt,_N.Bt,_N.Ct), (N.Nt,N.Bt,N.Ct))):
            if F and f:
                N_,L_,mTT,B_,dTT = comp_N_(F[0],rc,f[0]) if i<2 else comp_C_(F[0],rc,f[0])
                fTT = mTT + dTT; tt += fTT
                dfork_[i] = [N_,fTT,sum(fTT[0]),sum(fTT[1]), min(F[4],f[4]), min(F[5],f[5])]  # + B_,L_ per fork?
        TT += tt; Nt,Bt,Ct = dfork_; c = min(_N.c,N.c)

    root.c += c  # additive? fork_,dH sort and rc assign in cluster, not link?
    return CN(H=dH, dTT=TT, Nt=Nt, Bt=Bt, Ct=Ct, root=root, rc=rc, m=sum(TT[0]), d=sum(TT[1]), c=c)

def proj_TT(N, cos_d, dec, rc, pTT = np.zeros((2,9))):

    pTT += np.array([N.dTT[0]*dec, N.dTT[1]*cos_d*dec])  # coarse approximation
    V = val_(pTT,rc)
    if abs(V) > ave: return pTT,V  # ave: sym certainty margin proxy
    # refine projection if not certain:
    if N.H:
        for lev in N.H:
            pTT,V = proj_TT(lev, cos_d, dec, rc+1, pTT)  # acc pTT
            if abs(V) > ave: return pTT,V
    else:   # project forks
        for i, Ft in enumerate([N.dTT, N.Bt, N.Ct]):  # N.dTT is summed from N_, add trans-dTT?
            if Ft:
                if len(Ft) > 2: Ft = [Ft]
                for fork in Ft:
                    fTT = fork.dTT if i else fork
                    pTT += np.array([fTT[0]*dec, fTT[1]*cos_d*dec]); V = val_(pTT, rc)
                    if abs(V) > ave: return pTT,V
    return pTT,V

def comp_sub1(_N,N, rc, root):  # unpack node trees down to numericals and compare them

    _H, H = _N.H,N.H; TT = np.zeros((2,9))  # root comp_derT is in base_comp
    if _H and H:
        dH = []; C=0
        for _lev,lev in zip(_H, H):
            tt = comp_derT(_lev.dTT[1], lev.dTT[1])  # default
            TT+= tt; m,d = vt_(tt); c = min(_lev.c,lev.c); C += c
            dlev = CN(typ=1, dTT=tt, m=m,d=d,c=c, rc=min(_lev.rc,lev.rc), root=root)
            if (_lev.N_ or _lev.H) and (lev.N_ or lev.H) and val_(tt,rc) > 0:
                comp_sub(_lev,lev, rc,dlev)  # sub-recursion adds to dlev
            dH += [dlev]
    else:
        if N.H: N = N.H[0]  # comp 1st lev only: same elevation as N_
        if _N.H: _N = _N.H[0]
        N_,L_,mTT,B_,dTT, C = comp_N_(_N.zN_(),rc,N.zN_()); tt = mTT+dTT; m,d = vt_(tt); TT+=tt; root.N_ = L_+B_  # L.N_: trans-links
        dH = [CN(typ=1, N_=L_+B_,dTT=tt, m=m,d=d,c=C, rc=rc, root=root)]  # ders
        # no sub-recursion?
    if _N.B_ and N.B_:  # add in 1st lev only? + Bt,Ct, + tNt,tBt,tCt from trans_cluster?
        N_,bL_,mTT,bB_,dTT,c = comp_N_(_N.B_,rc, N.B_); tt = mTT+dTT; TT+=tt; C+=c; root.B_ = bL_+bB_
    if _N.C_ and N.C_:
        N_,cL_,mTT,cB_,dTT,c = comp_C_(_N.C_,rc, N.C_); tt = mTT+dTT; TT+=tt; C+=c; root.C_ = cL_+cB_

    root.H = dH; root.dTT+=TT; root.c+=C  # root.m = val_(TT,rc); root.d = val_(TT,rc,fi=0)?

def proj_TT1(L, cos_d, dec, rc, pTT = np.zeros((2,9))):  # always links

    pTT += np.array([L.dTT[0]*dec, L.dTT[1]*cos_d*dec])  # coarse approximation
    V = val_(pTT,rc)
    if abs(V) > ave: return pTT,V  # ave: sym certainty margin proxy
    # refine projection if not certain:
    if L.H:
        for lev in L.H:  # or always H in links?
            pTT,V = proj_TT(lev, cos_d, dec, rc+1, pTT)  # acc pTT
            if abs(V) > ave: return pTT,V
    else:   # project forks
        for i, Ft in enumerate([L.dTT, L.Bt, L.Ct]):  # N.dTT is summed from N_, add trans-dTT?
            if Ft:
                if len(Ft) > 2: Ft = [Ft]
                for fork in Ft:
                    fTT = fork.dTT if i else fork
                    pTT += np.array([fTT[0]*dec, fTT[1]*cos_d*dec]); V = val_(pTT, rc)
                    if abs(V) > ave: return pTT,V
    return pTT,V

def proj_N1(N, dist, A, rc):  # arg rc += N.rc+contw, recursively specify N projection val, add pN if comp_pN?

    rdist = dist / N.span   # internal x external angle:
    cos_d = (N.angl[0].dot(A) / (np.hypot(*N.angl[0]) * dist)) * N.angl[1]  # N-to-yx alignment
    m,d = N.m,N.d  # tentative
    dec = rdist * (m / (m+d))  # match decay rate, * ddecay for ds?
    # dec = link.m ** (1 + dist / link.span))
    # uncertainty = 1 - abs(dec_m - ave)

    iTT, eTT = np.zeros((2,9)), np.zeros((2,9))  # if separate eval?
    for L in N.L_+ N.B_: pTT,_ = proj_TT(L, cos_d, dec, L.rc+rc, iTT)
    iV = val_(iTT, rc)  # no _TT?
    for L in N.rim: pTT,_= proj_TT(L, cos_d, dec, L.rc+rc, eTT)
    eV = val_(eTT, rc)

    # info gain = (N.dTT[0] @ wTTf[0]) * ave_uncertainty:
    # val should be cumulative, -= borrow by N.dTT[1] @ wTTf[1]?
    return iTT+eTT, iV+eV  # * uncertainty * N.m?

def comp_sub(_N,N, rc, root):  # unpack node trees down to numericals and compare them

    _H, H = _N.H,N.H; TT = np.zeros((2,9))
    if _H and H:
        dH = []; C = 0
        for _lev,lev in zip(_H, H):
            C += min(_lev.c, lev.c)
            tt = comp_derT(_lev.dTT[1], lev.dTT[1]*rc); TT += tt  # default
            dlev = Cn(dTT=tt, m=sum(tt[0]), d=sum(tt[1]), root=root, c=min(_lev.c,lev.c), rc=min(_lev.rc,lev.rc))
            if val_(tt,rc) > 0:  # sub-recursion:
                comp_sub(_lev,lev, rc, dlev)
            dH += [dlev]
    else:
        dN_,dB_,dC_ = [],[],[]
        if N.H: N = N.H[0]  # comp 1st lev only
        if _N.H: _N = _N.H[0]
        for i, (F,f,dF) in enumerate(zip((_N.zN_(),_N.B_,_N.C_), (N.zN_(),N.B_,N.C_), (dN_,dB_,dC_))):
            if F and f:  # N_ is never empty
                N_,L_,mTT,B_,dTT = comp_N_(F,rc,f) if i<2 else comp_C_(F,rc,f)
                fTT = mTT + dTT; TT += fTT
                dF[:] = L_  # diffs
        dH = [Cn(N_=dN_,B_=dB_,C_=dC_, dTT=TT, m=sum(TT[0]), d=sum(TT[1]), root=root, rc=rc, c=min(_N.c, N.c))]

    root.H = dH; root.dTT += TT  # root.m = val_(TT,rc); root.d = val_(TT,rc,fi=0)?

def copy_(H, root):  # simplified
    cH = CN(dTT=deepcopy(H.dTT), root=root, rc=H.rc, m=H.m,d=H.d,c=H.c)  # summed across H
    cH.H = [copy_(lay,cH) for lay in H.H]
    cH.F_ = [copy(f) for f in cH.fork_]
    return cH

class Cn(CBase):  # light CN for non-locals: H levels, Bt, Ct, nG, Bt?
    name = "Nt"
    def __init__(n, **kwargs):
        super().__init__()
        n.H = kwargs.get('H', [])  # empty if N_, for nested root H levels
        n.N_, n.L_ = kwargs.get('N_',[]),kwargs.get('L_',[])  # principals
        n.dTT = kwargs.get('dTT',np.zeros((2,9)))  # separate N_,L_ TTs?
        n.m,  n.d, n.c = kwargs.get('m',0), kwargs.get('d',0), kwargs.get('c',0)  # sum dTT
        n.B_, n.C_, n.R_ = kwargs.get('B_', []), kwargs.get('C_', []), kwargs.get('R_', [])  # sub-forks?
        n.rc = kwargs.get('rc',1)  # redundancy
        n.typ = kwargs.get('fi',1)  # for compatibility?
        n.root = kwargs.get('root', None)  # immediate

    def zN_(n):  # get 1st level N_
        return n.N_ if n.N_ else n.H[0].zN_()
    def __bool__(n): return bool(n.c)
    # is_H(n: Cn) := bool(n.H); is_leaf(n: Cn) := bool(n.N_)

def add_N(N, n, init=0, fC=0, froot=0):  # rn = n.n / mean.n

    if froot: n.fin = 1; n.root = N
    N.dTT += n.dTT
    _cnt,cnt = N.c,n.c; Cnt = _cnt+cnt+1  # to weigh contribution of intensive params:
    if fC: n.rc = np.sum([mo[1] for mo in n._mo_]); N.rC_ += n.rC_; N.mo_ += n.mo_
    else:  N.rc = (N.rc*_cnt)+(n.rc*cnt) / Cnt
    N.c = (N.c*_cnt)+(n.c*cnt) / Cnt  # cnt / mass, same for centroids?
    n.C_ += [C for C in n.C_ if C not in N.C_]  # centroids, concat regardless
    if init:
        N.H = [Copy_(n, n.rc,root=N)]  # init? top layer
        if n.H: N.H += copy(n.H)
    else:  # concat
        add_N(N.H[0],n, init=not N.H[0].H)
        if n.H:  # N.H init above,
            for Lev,lev in zip(N.H[1:],n.H): add_N(Lev,lev)
    if N.typ:
        N.eTT += n.eTT
        N.baseT += n.baseT
        N.mang = (N.mang*_cnt + n.mang*cnt) / Cnt
        N.span = (N.span*_cnt + n.span*cnt) / Cnt
        N.angl = (N.angl*_cnt + n.angl[0]*cnt) / Cnt  # vect only?
        A,a = N.angl[0],n.angl[0]; A[:] = (A*_cnt+a*cnt) / Cnt  # not sure
        N.yx += [n.yx]  # weigh by Cnt?
        N.box = extend_box(N.box, n.box)
        for Ft, ft in zip((N.Bt, N.Ct, N.Nt), (n.Bt, n.Ct, N.Nt)):
            if ft: add_N(Ft, ft)  # Ft maybe empty CN, weigh by Cnt?
        for L in n.rim:
            if L.m > ave: N.L_ += [L]
            else: N.B_ += [L]
    # if N is Fg: margin = Ns of proj max comp dist > min _Fg point dist: cross_comp Fg_?
    return N

def add_sub(N,n, root):  # add n.H|n.N_, n.Bt,n.Ct to N, analogous to comp_sub

    N.dTT += n.dTT; N.rc += n.rc; N.c += n.c
    if n.H:
        for Lev, lev in zip_longest(N.H,n.H):
            if Lev and lev: add_sub(Lev,lev, N)
            elif lev:
                N.H += [Copy_(lev, N)]
    else:
        N.N_ += [n for n in n.N_ if n not in N.N_]  # single lev
    # add in root?
    for L_,l_ in zip((N.L_, N.B_, N.R_), (n.L_, n.B_, n.R_)):
        L_ += [l for l in l_ if l not in L_]
    ''' 
    not sure:
    for Ft, ft in zip((N.Bt, N.Ct), (n.Bt, n.Ct)):
        if ft:
            if Ft:
                for L_,l_ in zip((Ft.N_,Ft.L_,Ft.B_,Ft.R_), (ft.N_,ft.L_,ft.B_,ft.R_)):
                L_ += [l for l in l_ if l not in L_]
                Ft.rc += ft.rc; Ft.c += ft.c
                Ft.dTT += ft.dTT
                Ft.m = sum(Ft.dTT[0])  # recompute m
                Ft.d = sum(Ft.dTT[1])  # recompute d
            else copy
    '''
def cross_comp(root, rc, fC=0):  # rng+ and der+ cross-comp and clustering, fT: convert return to tuple

    N_, mL_,mTT, dL_,dTT,_ = comp_C_(root.N_,rc) if fC else comp_N_(root.N_,rc)  # rc: redundancy+olp, fi=1|0
    Bt = CN(typ=0, root=root, N_=dL_)
    if fC<2 and dL_ and val_(dTT, rc+compw, fi=0, mw=(len(dL_)-1)*Lw) > avd:  # comp dL_| dC_, not ddC_
        cross_comp(Bt, rc+compw+1, fC*2)  # d fork, trace_edge via nt s
    # m fork:
    if len(mL_) > 1 and val_(mTT, rc+compw, mw=(len(mL_)-1)*Lw) > 0:
        for n in N_: n.em = sum([l.m for l in n.rim]) / len(n.rim)  # tentative before val_
        if Cluster(root, mL_, rc, fC):  # fC=0: get_exemplars, cluster_C, rng connect cluster, update root in-place
            rc = root.rc  # include new clustering layers
            if Bt.Nt:  # add eval?
                form_B__(root,Bt)  # add boundary to N and N to Bg R_s, no root update
                if val_(mTT, rc+3+contw, mw=(len(root.N_)-1)*Lw) > 0:  # mval
                    trace_edge(root,rc+3)  # comp Ns with shared N.Bt
            if val_(root.dTT, rc+compw+3, mw=(len(root.N_)-1)*Lw, _TT=mTT) > 0:
                cross_comp(root, rc+3)  # connec agg+, fC = 0
    if Bt.Nt:  # was clustered
        root.B_=dL_; root.Bt=Bt; root.dTT+=Bt.dTT  # new boundary
