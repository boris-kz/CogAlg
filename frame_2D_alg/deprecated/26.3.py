def cross_comp(Ft, ir, nF='Nt'):  # core function mediating recursive rng+ and der+ cross-comp and clustering

    N_, G_,R = Ft.N_,[], Ft.root  # rc=rdn+olp, comp N_|B_|C_:
    iN_,L_,TT,c,r, TTd,cd = comp_N_(N_,combinations(N_,2),ir) if N_[0].typ else comp_C_(N_,ir,fC=1); r+=ir
    if L_:
        if val_(TT, r+connw,TTw(Ft),(len(L_)-1)*Lw,1,TTd,r) > 0:  # G.L_ = Nt.Lt.N_, flat, additional root +=:
            m,d,tt,lc,lr = sum_vt(L_,root=getattr(Ft,'Lt'))
            rc = c/(R.c+c); R.dTT+=(tt-R.dTT)*rc; R.r+=(r-R.r)*rc; R.c+=c  # Ft: transitive root
            E_ = get_exemplars({N for L in L_ for N in L.nt if N.rim.m>ave}, r)  # N|C?
            G_,r = cluster_N(Ft, E_,r)  # form Bt, trans_cluster, sub+ in sum2G
            if G_:
                Ft = sum2F(G_, nF, Ft.root)
                if val_(Ft.dTT, r+nw, TTw(Ft), (len(N_)-1)*Lw,1, TTd,r) > 0:
                    G_,r = cross_comp(Ft,r,nF)  # agg+,trans-comp
    return G_,r  # G_ is recursion flag?

def comb_Ft(Nt, Lt, Bt, Ct, root, fN=0):  # default Nt

    T = CopyF(Nt)
    if fN: T = CN(Nt=T,dTT=Nt.dTT,c=Nt.c,r=Nt.r,root=root); T.Nt.root=T; add_Nt(T,Nt)  # +H and ext attrs
    else:  T.N_ = [Nt,Lt,Bt,Ct]; T.root=root  # nested tFt
    dF_ = []  # comp_N tFs | sum2G dFs
    if Lt: sum_vt([T,Lt],root=T.Nt); add_Lt(T.Nt,Lt); dF_ += [comp_F(T,Lt, T.r,T.Nt)]  # Lt in sum2G, tLt in comp_N
    if Bt: sum_vt([T,Bt],root=T); T.Bt=Bt; dF_ += [comp_F(T,Bt, T.r,root)]  # no ext pars?
    if Ct: sum_vt([T,Ct],root=T); T.Ct=Ct; dF_ += [comp_F(T,Ct, T.r,root)]
    if dF_:
        m,d,tt,c,r = sum_vt(dF_,root=getattr(T,'Lt'), merge=1); sum_vt([T,T.Lt], root=T)  # cross-fork covariance
        R=T.root; rc=c/(R.c+c); R.dTT+=(tt-R.dTT)*rc; R.r+=(r-R.r)*rc; R.c+=c  # G

    @property
    def L_(N):
        if isinstance(N.Nt.Lt, list): N.Nt.Lt =CF(root=N.Nt)
        return N.Nt.Lt.N_
    @L_.setter
    def L_(N,v):
        if isinstance(N.Nt.Lt, list): N.Nt.Lt =CF(root=N.Nt)
        N.Nt.Lt.N_ = v

    return T

def get_exemplars(N_,r):  # multi-layer non-maximum suppression -> sparse seeds for diffusive clustering, cluster_N too?
    E_ = set()
    for rdn,N in enumerate(sorted(N_, key=lambda n:n.Rt.m, reverse=True), start=1):  # strong-first
        oL_ = set(N.rim) & {l for e in E_ for l in e.rim}
        roV = (sum_vt(list(oL_), rr=r)[0] if oL_ else 0) / (N.Rt.m or eps)  # relative rim olp V
        # ave *= rV of overlap by stronger-E inhibition zones:
        if N.Rt.m * N.c > ave * (r+rdn+nw+ roV):
            E_.update({n for l in N.rim for n in l.nt if n is not N and N.Rt.m > ave*r})
            N.exe = 1  # in point cloud of focal nodes  (N.exe is 1, but never added to E? We didn't update E.exe here?)
        else:
            break  # the rest of N_ is weaker, trace via rims
    return E_

def comp_N_(iN_, pairs, r, tnF=None, rL=None):  # incremental-distance cross_comp, max dist depends on prior match

    def proj_V(_N,N, dist, pVt_, dec):  # _N x N induction
        Dec = dec or decay ** ((dist/((_N.span+N.span)/2)))
        iTT = (_N.dTT + N.dTT) * Dec
        eTT = (_N.Rt.dTT + N.Rt.dTT) * Dec
        if abs( vt_(eTT,r)[0]) * ((len(pVt_)-1)*Lw) > ave*specw:  # spec N links
            eTT = np.zeros((2,9)) # recompute
            for _dist,_dy_dx,__N,_V in pVt_:
                eTT += proj_N(N,_dist, _dy_dx, r, dec)  # proj N L_,B_,rim, if pV>0: eTT += pTT?
                eTT += proj_N(_N,_dist, -_dy_dx, r, dec)  # reverse direction
        return iTT+eTT
    N_, TT, C,R = [], np.zeros((2,9)), 0,0
    for N in iN_: N.pL_ = []  # init
    for _N, N in pairs:  # get all-to-all pre-links
        if _N.sub != N.sub: continue  # or comp x composition?
        if N is _N:  # overlap = unit match, no miss, or skip?
            tt = np.array([N.dTT[1],np.zeros(9)]); TT+=tt; C+=min(N.c,_N.c); R+= (N.r+_N.r)/2
        else:
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx)
            N.pL_+= [[dist,dy_dx,_N]]; _N.pL_+= [[dist,-dy_dx,N]]; N_ += [N,_N]
    if not N_: return 0,0,0,0,r,0,0
    for N in set(N_): N.pL_.sort(key=lambda x: x[0])  # proximity prior, test compared?
    N_,L_,dpTT, TTd,cd = [],[],np.zeros((2,9)),np.zeros((2,9)),0  # any global use of dLs, rd?
    for N in iN_:
        pVt_ = []  # [[dist, dy_dx, _N, V]]
        for dist, dy_dx, _N in N.pL_:  # rim angl is not canonic
            pTT = proj_V(_N,N, dist, pVt_, rL.m if rL else decay** (dist/((_N.span+N.span)/2)))
            lr = r+ (N.r+_N.r) / 2; m,d = vt_(pTT,lr)  # +|-match certainty
            if m > 0:
                if abs(m) < ave * nw:  # different ave for projected surprise value, comp in marginal predictability
                    Link = comp_N(_N,N, lr, full=not tnF, A=dy_dx, span=dist, rL=rL)
                    dTT, m,d,c,lr = Link.dTT,Link.m,Link.d,Link.c,Link.r  # prevent replacing of the input r
                    if   m > ave: TT+=dTT*c; C+=c; R+=lr*c; L_+=[Link]; N_+=[_N,N]
                    elif d > avd: TTd+=dTT*c; cd+=c  # no overlap to simplify
                    dpTT += pTT-dTT  # prediction error to fit code, not implemented
                else:
                    pL = CN(typ=-1, nt=[_N,N], dTT=pTT,m=m,d=d,c=min(N.c,_N.c), r=lr, angl=np.array([dy_dx,1],dtype=object),span=dist)
                    L_+= [pL]; N.rim+=[pL]; N_+=pL.nt; _N.rim+=[pL]; TT+=pTT; C+=pL.c  # same as links in clustering
                pVt_ += [[dist,dy_dx,_N,m]]  # for next rim eval
            else: break  # beyond induction range
    for N in set(N_):
        if N.rim: sum_vt(N.rim,getattr(N,'Rt'))
    return list(set(N_)), L_,TT,C,R, TTd,cd  # + dpTT for code-fitting backprop

def cluster_C(Ft, E_, r):  # form centroids by clustering exemplar surround via rims of new member nodes, within root

    C_,_C_ = [],[]  # form root.Ct, may call cross_comp-> cluster_N, incr rc
    for n in Ft.N_: n._C_,n.m_,n._m_,n.o_,n._o_,n.rN_ = [],[],[],[],[],[]
    for E in E_:
        C = cent_TT(Copy_(E, Ft, init=2,typ=0), r)  # all rims are in root, sequence along eigenvector?
        C._N_ = list({n for l in E.rim for n in l.nt if (n is not E and n in Ft.N_)})  # init C.N_=[]
        C._L_ = set(E.rim)  # init peer links
        for n in C._N_: nm = C.m*(n.c/C.c); n.m_+=[nm]; n._m_+=[nm]; n.o_+=[1]; n._o_+=[1]; n.rN_+=[C]; n._C_+=[C]
        _C_ += [C]
    __C_ = []  # buffer stable Cs
    while True:  # reform C_, add direct in-C_ cross-links for membership?
        C_,cnt,olp, mat,dif, DTT,Dm,Do = [],0,0,0,0,np.zeros((2,9)),0,eps; Ave = ave*(r+nw); Avd = avd*(r+nw)
        _Ct_ = [[c, c.m/c.c, c.r] for c in _C_]
        for cr, (_C,_m,_o) in enumerate(sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True),start=1):
            if _m > Ave *_o:
                L_, N_,N__,m_,o_,M,D,O,cc, dTT,dm,do = [],[],[],[],[],0,0,0,0, np.zeros((2,9)),0,0  # /C
                for n in set(_C.N_+_C._N_):  # current + frontier, refine all
                    dtt,_ = base_comp(_C, n); cc+=1  # add decay / dist?
                    m,d = vt_(dtt,cr); dTT += dtt; nm = m * n.c  # rm,olp / C
                    odm = np.sum([_m-nm for _m in n._m_ if _m>m])  # higher-m overlap
                    oL_ = set(n.rim) & _C._L_  # replace peer rim overlap with more precise m
                    if oL_: m += sum_vt(oL_,rr=cr,rm=m)[0]  # add deviation from redundant mC, add eval?
                    if m > 0 and nm > Ave * odm:
                        N_+=[n]; L_+=n.L_; M+=m; O+=odm; m_+=[nm]; o_+=[odm]  # n.o for convergence eval
                        N__ += [_n for l in n.rim for _n in l.nt if _n is not n]  # +|-Ls
                        if _C not in n._C_: dm+=m; do+=odm  # not in extended _N__
                    else:
                        if _C in n._C_: i = n._C_.index(_C); dm+=n._m_[i]; do+=n._o_[i]
                        D += abs(d)  # distinctive from excluded nodes (background)
                mat+=M; dif+=D; olp+=O; cnt+=cc  # all comps are selective
                DTT += dTT
                if M > Ave * len(N_) * O and val_(dTT, cr+O, TTw(_C),(len(N_)-1)*Lw):
                    C = sum2C(N_,_C,_Ci=None)
                    for n,m,o in zip(N_,m_,o_):
                        if _C in n.rN_: i = n.rN_.index(_C); n.rN_.pop(i); n.m_.pop(i); n.o_.pop(i)  # clear stale _C mapping?
                        n.m_+=[m]; n.o_+=[o]; n.rN_+= [C]; C.rN_+= [n]  # reciprocal root assign
                    C._N_ = list(set(N__)-set(N_))  # frontier
                    C._L_ = set(L_)  # peer links
                    if D< Avd*len(N_)*O: __C_ += [C]  # stable
                    else: C_ += [C]; Dm+=dm; Do+=do  # reform in next loop
                else:
                    for n in _C._N_:  # not revised
                        n.exe = n.m/n.c > 2 * ave  # refine exe
                        for i, c in enumerate(n.rN_):
                            if c is _C:  # remove _C-mapping m,o:
                                n.rN_.pop(i); n.m_.pop(i);n.o_.pop(i); break
            else: break  # the rest is weaker
        for n in Ft.N_:
            n._C_ = n.rN_; n._m_= n.m_; n._o_= n.o_; n.rN_,n.m_,n.o_ = [],[],[]  # new n.Ct.N_s, combine with v_ in Ct_?
        if mat * dif * olp > ave * centw*2:
            __C_ = cluster_P(__C_, Ft.N_, r)  # refine all memberships in parallel by global backprop|EM
            break
        if Dm / Do > Ave:  # dval vs. dolp: overlap increases with Cs expansion
            _C_ = C_
        else: break  # converged
    C_ = [C for C in __C_ if val_(C.dTT, r, TTw(C))]  # prune C_
    if C_:
        for n in [N for C in C_ for N in C.N_]:  # exemplar V + sum n match_dev to Cs, m* ||C rvals:
            n.exe = (n.d if n.typ==1 else n.m) + np.sum([m-ave*o for m, o in zip(n.m_, n.o_)]) - ave
        if val_(DTT, r+olp, TTw(Ft), (len(C_)-1)*Lw) > 0:
            Ct = sum2F(C_,'Ct', Ft.root, fCF=0)
            _, r = cross_comp(Ct,r)  # all distant Cs, seq C_ in eigenvector = argmax(root.wTT)? Nt|Ct priority eval?
    return C_, r

def sum2C(N_, _C, _i=None, root=None):  # fuzzy sum + base attrs for centroids

    cc_ = []
    for N in N_:
        if _i is None: i = N._C_.index(_C); m,o = N._m_[i], N._o_[i]  # cluster_C
        else:          m,o = N.m_[_i],N.o_[_i]  # current m_,o_ in cluster_P
        cc_ += [N.c * (m/(ave*o) * _C.m)]  # *_C.m: proj survival?
    Cc = sum(cc_)
    R = 0; TT = np.zeros((2,9)); kern = np.zeros(4); span = 0; yx = np.zeros(2)
    for N, c in zip(N_,cc_):
        rc = c/ Cc; TT += N.dTT*rc; kern += N.kern*rc; span += N.span*rc; yx += N.yx*rc; R += N.r*rc
    m,d = vt_(TT,R)
    C = CN(typ=3, Nt=CF(N_=N_), dTT=TT, m=m, d=d, c=Cc, r=R, kern=kern, span=span, yx=yx, root=root)

    return cent_TT(C, C.r)

def comb_Ft_(Nt, Lt, Bt, Ct, root, fN=0):  # root = G|L, default Nt

    r = root.r  # new G.Fs / sum2G | old L.tFs ->L.N_ / comp_N:
    T = CopyF(Nt)  # temporary accumulator
    if fN: R = CN(Nt=Nt,Lt=Lt,Bt=Bt,Ct=Ct, root=root); Nt.root=R; Lt.root=R; Bt.root=R; Ct.root=R  # new G / sum2G
    else:  R = root.Xt  # keep root Link
    dF_ = [CF(),CF(),CF()]
    if Lt: dF_[0] = comp_F(T,Lt, r,R); sum_vt([T,Lt],T)  # Lt in sum2G, L.tLt in comp_N
    if Bt: dF_[1] = comp_F(T,Bt, r,R); sum_vt([T,Bt],T)  # * brrw /G update?
    if Ct: dF_[2] = comp_F(T,Ct, r,R); sum_vt([T,Ct],T)  # * rdn /G update?
    sum_vt([R,T], R)
    if any(dF_):
        if fN: Xt = R.Xt; RR = R; dFt = Xt
        else:  Xt = R; RR = root; Xt.Lt = Xt.Lt or CF(root=Xt); dFt = Xt.Lt
        sum_vt(dF_,dFt, merge=1)  # cross-fork covariance, dFt.N_=dF_
        sum_vt([RR, Xt], RR)  # update Link or root
    if fN:  # no L ext update / tLs
        add_Nt(R, Nt)  # add H,kern,ext /G, doesn't affect comp_F
        if Lt: add_Lt(R, Lt)
        return R  # in sum2G

def sum_vt(N_, root=None, rc=0,rr=0,rm=0,rd=0, merge=0, f2=0):  # weighted sum of CN|CF list

    C = sum(n.c for n in N_); R = 0; TT = np.zeros((2,9))
    for n in N_:
        rc = n.c/C; TT += n.dTT*rc; R += n.r*rc  # * weight
    R = (R + rr) / 2  # rr*rc?
    m,d = vt_(TT, R); m-=rm; d-=rd  # deviations from tentative m,d
    if root is not None:
        root.dTT=TT; root.r=R; root.c=C; root.m=m; root.d=d  # * brrw/Bt, rdn/Ct?
        if merge:
            n_ = N_[1].N_ if f2 else N_  # f2: two Ns, merge 2nd into 1st
            for n in n_: n.root=root; root.N_ += [n]
    return m,d, TT, C,R

def ffeedback(frame):  # adjust filters: all aves *= rV, ultimately differential backprop per ave?

    def L_ders(Fg, TT,wTT):  # get current-level ders: from L_ only
        m, d = vt_(TT, wTT=getattr(Fg, wTT))
        return m,d,dTT

    # draft:
    wTT_ = []
    for nTT,nwTT in zip(('TT','dTT','TTn','TTc'),('wTTn','wTTc','wTTN','wTTC')):  # combine for total cent_TT(wTTx)?
        dTT = getattr(frame, nTT)
        wTTf = np.ones((2,9))  # sum dTT weights: m_,d_ [M,D,n, I,G,A, L,S,ext_A]: Et, kern, extT
        rM, rD = 1, 1
        _m,_d,_dTT = 0,0,np.zeros((2,9))
        for lev in frame.H:  # top-down, not lev-selective, not recursive
            m,d,dTT =  L_ders(lev, nTT, nwTT)
            rM += _m / (m or eps)  # mat,dif change per level
            rD += _d / (d or eps)
            wTTf += np.abs(_dTT / (dTT+eps))
            _m,_d,_dTT = m,d,dTT
        wTT_ += [wTTf]
    return rM+rD, *wTT_

def frame_H(image, iY,iX, Ly,Lx, Y,X, rV, max_elev=4, wTTf=np.ones((2,9))):  # all initial args set manually

    def base_tile(y,x):  # 1st level, higher levels get Fg s
        Fg = frame_blobs_root( comp_pixel( image[y:y+Ly, x:x+Lx]), rV)
        Fg = vect_edge(Fg, rV)  # form, trace PP_
        if Fg: cross_comp(Fg.Nt, Fg.r)
        return Fg

    def expand_lev(_iy,_ix, elev, Fg):  # seed tile is pixels in 1st lev, or Fg in higher levs

        tile = np.full((Ly,Lx),None, dtype=object)  # exclude from PV__
        PV__ = np.zeros([Ly,Lx])  # maps to current-level tile
        Fg_ = []; iy,ix =_iy,_ix; y,x = 31,31  # start at tile mean
        while True:
            if not elev: Fg = base_tile(iy,ix)  # 1st level or previously cross_comped arg tile
            if Fg and val_(Fg.dTT, Fg.r+nw+elev, Fg.wTT, mw=(len(Fg.N_)-1)*Lw) > 0:
                tile[y,x] = Fg; Fg_+=[Fg]
                dy_dx = np.array([Fg.yx[0]-y,Fg.yx[1]-x])
                pTT = proj_N(Fg, np.hypot(*dy_dx), dy_dx, elev)
                if 0< val_(pTT, elev, TTw(Fg)) < ave:  # search in marginal +ve predictability?
                    # extend lev by feedback within current tile:
                    proj_focus(PV__,y,x,Fg)  # PV__+= pV__
                    pv__ = PV__.copy(); pv__[tile!=None] = 0  # exclude processed
                    y, x = np.unravel_index(pv__.argmax(), PV__.shape)
                    if PV__[y,x] > ave:
                        iy = _iy + (y-31)* Ly**elev  # feedback to shifted coords in full-res image | space:
                        ix = _ix + (x-31)* Lx**elev  # y0,x0 in projected bottom tile:
                        if elev:
                            subF = frame_H(image, iy,ix, Ly,Lx, Y,X, rV, elev, wTTf)  # up to current level
                            Fg = subF.N_[-1] if subF else []
                    else: break
                else: break
            else: break
        if Fg_:
            gTT,gC,gRc = sum_vt(Fg_); gRc+=elev
            if val_(gTT, gRc/len(Fg_)+elev, wTTf, mw=(len(Fg_)-1)*Lw) > 0:  # not updated
                return Fg_

    global ave,avd, Lw, intw, cw,nw, centw,connw, distw, mW, dW
    ave,avd, Lw, intw, cw,nw, centw,connw, distw = np.array([ave,avd, Lw, intw, cw, nw, centw, connw, distw]) / rV

    frame = CN(box=np.array([0,0,Y,X]), yx=np.array([Y//2, X//2]))
    Fg=[]; elev=0
    while elev < max_elev:  # same center in all levels
        Fg_ = expand_lev(iY,iX, elev, Fg)
        if Fg_:  # higher-scope sparse tile
            N_2R(Fg_,root=frame)
            if Fg and cross_comp(Fg.Nt, rr=elev)[0]:  # spec->tN_,tC_,tL_, proj non-selective Fg.L_?
                frame.N_ = frame.N_+[Fg]; elev+=1  # forward comped tile
                if max_elev == 4:  # seed, not from expand_lev
                    rV,wTTf = ffeedback(Fg)  # set filters
                    Fg = cent_TT(Fg,2)  # set Fg.dTT correlation weights
                    wTTf *= Fg.wTT; mW = np.sum(wTTf[0]); dW = np.sum(wTTf[1])
                    wTTf[0] *= 9/(mW+eps); wTTf[1] *= 9/(dW+eps)
            else: break
        else: break
    return frame  # for intra-lev feedback

def cent_TT(C, r):  # weight attr matches | diffs by their match to the sum, recompute to convergence

    wTT = []  # Cs can be fuzzy only to the extent that their correlation weights are different?
    tot = C.dTT[0] + np.abs(C.dTT[1])  # m_* align, d_* 2-align for comp only?

    for fd, derT, wT in zip((0,1), C.dTT, wTTf):
        if fd: derT = np.abs(derT)  # ds
        _w_ = np.ones(9)  # or 4 if cross fork, weigh by feedback:
        val_ = derT / tot * wT  # signed ms, abs ds
        V = np.sum(val_)
        while True:
            mean = max(V / max(np.sum(_w_),eps), eps)
            inverse_dev_ = np.minimum(val_/mean, mean/val_)  # rational deviation from mean rm in range 0:1, if m=mean
            w_ = inverse_dev_/.5  # 2/ m=mean, 0/ inf max/min, 1 / mid_rng | ave_dev?
            w_ *= 9 / np.sum(w_)  # mean w = 1, M shouldn't change?
            if np.sum(np.abs(w_-_w_)) > ave*r:
                V = np.sum(val_ * w_)
                _w_ = w_
            else: break  # weight convergence
        wTT += [_w_]
    C.wTT = np.array(wTT)  # replace wTTf
    # single-mode dTT, extend to 2D-3D lev cycles in H, cross-level param max / centroid?
    return C

def cent_TT1(_wTT, r, rTT=None):  # weight attr matches | diffs by their match to the sum, recompute to convergence

    wTT = []  # Cs can be fuzzy only to the extent that their correlation weights are different?
    tot = rTT[0] + np.abs(rTT[1])  # m_* align, d_* 2-align for comp only?

    for fd, rT, wT in zip((0,1), rTT,_wTT):  # G.wTT_[wi] *= ffeedback ratios, default ones?
        if fd: rT = np.abs(rT)  # ds
        _w_ = np.ones(9)  # or 4 if cross fork, weigh by feedback:
        rv_ = rT / tot * wT  # signed ms, abs ds
        V = np.sum(rv_)
        while True:
            mean = max(V / max(np.sum(_w_),eps), eps)
            inverse_dev_ = np.minimum(rv_/mean, mean/rv_)  # rational deviation from mean rm in range 0:1, if m=mean
            w_ = inverse_dev_/.5  # 2/ m=mean, 0/ inf max/min, 1 / mid_rng | ave_dev?
            w_ *= 9 / np.sum(w_)  # mean w = 1, M shouldn't change?
            if np.sum(np.abs(w_-_w_)) > ave*r:
                V = np.sum(rv_ * w_)
                _w_ = w_
            else: break  # weight convergence
        wTT += [_w_]
    _wTT[:] = np.array(wTT)  # replace wTTf
    # single-mode dTT, extend to 2D-3D lev cycles in H, cross-level param max / centroid?

def sum2F(N_, nF, root, TT=np.zeros((2,9)), C=0, R=0, fset=1, fCF=1):  # -> CF/CN

    def sum_H(N_, Ft):
        H = []
        for N in N_:
            if H: H[0] += N.N_  # new top level
            elif  N.N_: H = [list(N.N_)]
            for Lev,lev in zip_longest(H[1:], N.Nt.H):  # aligned top-down
                if lev:
                    if Lev is not None: Lev += lev.N_
                    else: H += [list(lev.N_)]
        Ft.H = [sum2F(lev,'lev',Ft,fset=0) for lev in H] if H else []
    if C: m,d = vt_(TT,R)
    else: m,d,TT,C,R = sum_vt(N_,fm=1)
    Ft = (CN,CF)[fCF](nF=nF, dTT=TT,m=m,d=d,c=C,r=R, root=root); setattr(Ft,'N_',N_)   # root Bt|Ct ->CN
    if any([n.N_ for n in N_]):
        sum_H(N_,Ft)  # sum lower levels, if any
    if fset:
        setattr(root, Ft.nF,Ft)
        for N in N_: N.root = Ft
    return Ft

