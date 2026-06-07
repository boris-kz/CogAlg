F_call_T_, F_call_i_ = [],[]
for F in oF_:
    for i,p in enumerate(flat_body(F.body)):
        if isinstance(p,CoF): F_call_i_ += [i]; F_call_T_ += [np.zeros((2,9))]  # to sum dTTs per oF.call
# or F_call_T_ = [[[i, np.zeros((2,9))] for i,p in enumerate(flat_body(F.body)) if isinstance(p,CoF)] for F in oF_]

def flat_body(body, out=None):
    if out is None: out = []
    for p in body:
        out += [p]
        if isinstance(p, tuple): flat_body(p[1], out)
    return out

def frame_H(image, iY,iX, Ly,Lx, Y,X, rV, max_elev=4):  # all initial args set manually

    def base_tile(y,x):  # 1st level, higher levels get Tg s
        T = frame_blobs_root( comp_pixel( image[y:y+Ly, x:x+Lx]), rV)
        T = vect_edge(T, rV)  # form, trace PP_
        if T: cross_comp(T.Nt,T.r)
        return T

    def expand_lev(_iy, _ix, elev, T):  # seed tile is pixels in 1st lev, or Fg in higher levs

        frame = np.full((Ly, Lx), None, dtype=object)  # level scope
        iy,ix = _iy,_ix; cy,cx = (Ly-1)//2, (Lx-1)//2; y,x = cy,cx  # start=mean
        T_, PV__,C,R = [],np.zeros([Ly,Lx]),0,0  # tiles, maps to level frame
        while True:
            if not elev: T = base_tile(iy, ix)
            if T and sum(vt_(T.dTT, T.wTT*ttFrm))*(wFrm*(len(T.N_)-1)) > (ave+avd)*(T.r+elev+cFrm*(len(T.N_)-1)):
                frame[y,x] = T; T_ += [T]  # loop adds one tile to level
                dy_dx = np.array([T.yx[0]-y, T.yx[1]-x])
                pTT = proj_N(T, np.hypot(*dy_dx), dy_dx, elev,T.c)
                if 0 < sum(vt_(pTT,T.wTT*ttFrm))*(elev+cFrm) < ave:  # extend lev by combined proj T_
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
            TT,C,R = sum_vt(T_, wTT=ttFrm); R += elev; L = len(T_)-1
            if sum(vt_(TT,ttFrm))*(wFrm*L) <= (ave+avd)*(R+cFrm*L): T_=[]; C=0; R=0
        return T_,C,R
    elev = 0
    F,tile = [],[]  # frame, seed lower tile, if any
    global ave,avd, Fw_,FTT_  # update from ffeedback:
    while elev < max_elev:
        tile_,C,R = expand_lev(iY,iX, elev, tile); oF=CoF.get();oF.c += C; oF.r += R
        if tile_:  # sparse higher-scope tile, if expanded
            b__ = np.array([T.box for T in tile_])
            box = np.array([*b__[:, :2].min(0), *b__[:, 2:].max(0)])  # y,x,Y,X
            Fr = CN(box=box, span=np.hypot(box[2] - box[0], box[3] - box[1]) / 2, yx=tile_[0].yx)  # keep seed T center
            Fr.span = np.hypot(box[2] - box[0], box[3] - box[1]) / 2  # mean extent
            F.H =[]; F.N_= tile_
            if elev: [add_H(Fr.H, T.H, Fr, fN=1) for T in tile_]  # lower level tiles, add_H unpacks nested oH(aH
            F.H += [lev := sum2F(tile_)]  # include top lev, same vals as F?
            if cross_comp(lev, rr=elev)[0]:  # spec->tN_,tC_,tL_, proj comb N_'L_?
                elev += 1
                if rV > ave:
                    if elev== max_elev: rV,FTT_ = ffeedback(F)  # from top lev
                    for i, tF in enumerate(oF_):
                        if tF: Fw_[i] = tF.fw/tF.c; FTT_[i] = lev.wTT_[i] = tF.wTT
                    ave/=rV; avd/=rV; Fw_,FTT_ = np.array(Fw_) / rV, np.array(FTT_) / rV  # Fc_ is fixed
                tile = F  # lev tile_ is next extension seed
            else: break
        else: break
    return F  # for intra-lev feedback

def ffeedback(frame, rV,elev):  # adjust filters via cross-level wTT ratios; fork: reform oF_ when converged

    global ave,avd, Fw_,FTT_
    H = frame.H
    rTT = np.divide(H[0].wTT, H[1].wTT); _wTT = H[1].wTT
    for lev in H[2:]:  # exclude packed levs?
        rTT += np.divide(_wTT,lev.wTT); _wTT = lev.wTT
    rm,rd = vt_(rTT,wTT); rV = rm+rd
    if rV * wBac > ave * cBac:  # filter update terminates old aH
        if len(H) > (i := next((j+1 for j in reversed(range(len(H))) if H[j].nF in ('aH','oH')),0)):  # list tail
            aH = sum2F(frame.H[i:], nF='aH'); aH.wTT=frame.wTT  # default CN, conditional sum?
            frame.H += [aH]  # same_filter_levs
        for i, tF in enumerate(oF_):
            if tF: Fw_[i] = tF.fw/tF.c; FTT_[i] = frame.wTT_[i] = tF.wTT
        ave /= rV; avd /= rV
        Fw_ = np.array(Fw_)/rV; FTT_ = np.array(FTT_)/rV
        if rV * wBac**2 > ave* cBac**2:  # oF_ reform terminates old oH
            split_oF_(); oF_n = cluster_oF_()
            for i, F in enumerate(oF_n): F.nF = i
            if len(H) > (i := next((j+1 for j in reversed(range(len(H))) if H[j].nF=='oH'),0)):
                if a_ := [l for l in H[i:] if l.nF=='aH']:
                    oH = sum2F(a_, nF='oH'); oH.wTT=frame.wTT; frame.H += [oH]  # same_oF_levs
    else: rV = 1  # no-op
    return rV, rTT

def wrap(H, nF, in_=('aH', 'oH')):  # terminate: group trailing levs formed under old regime
        i = next((j + 1 for j in reversed(range(len(H))) if H[j].nF in in_), 0)  # end of last closed group
        if len(H) > i:
            G = CN(root=frame, nF=nF);
            G.H = H[i:];
            G.dTT, G.c, G.r = sum_vt(G.H);
            G.m, G.d = vt_(G.dTT)
            G.wTT = copy(frame.wTT)  # old-regime stamp: cross-regime rTT, add_H alignment
            return H[:i] + [G]
        return H
