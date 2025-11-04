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

