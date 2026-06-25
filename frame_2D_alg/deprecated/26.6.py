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

def ffeedback1(frame):  # adjust filters via cross-level wTT ratios; fork: reform oF_ when aH chain converged

    global ave,avd; H = frame.H
    i = next((j+1 for j in reversed(range(len(H))) if H[j].nF in ('aH','oH')), 0)  # open tail start, prior levs are packed
    if len(tail:= H[i:]) > 1:
        _dTT = tail[0].dTT  # drift anchor
        D = np.sum(np.abs(sum(_dTT - lev.dTT for lev in tail[1:])) * wTT)  # drift
        if (vD := D * wBac > ave * cBac) > 0:  # filter update
            aH = CN(nF='aH', H=tail, root=frame)
            if vD > ave: aH.dTT,aH.c,aH.r = sum_vt(tail); aH.m,aH.d = vt_(aH.dTT)  # deep summary
            frame.H = H[:i] + [aH]  # same_filter_levs
            DTT = aH.dTT if aH.c else tail[-1].dTT  # end or summary
            ave, avd = vt_(DTT)  # if cross drift M? other filters are ave coefs
    # reform tail aH_'oF_:
    i = next((j+1 for j in reversed(range(len(frame.H))) if frame.H[j].nF=='oH'), 0)  # aH chain
    if len(aH_ := [l for l in frame.H[i:] if l.nF=='aH']) > 1:
        _dTT = aH_[0].dTT
        D = np.sum(np.abs(sum(_dTT - a.dTT for a in aH_[1:])) * wTT)  # cross-regime drift
        if (vD := D*wBac**2 - ave*cBac**2) > 0:  # code stale -> reform oF_, oH:
            oH = CN(nF='oH', H=aH_, root=frame); oH.dTT = aH_[-1].dTT
            if vD > ave: oH.dTT,oH.c,oH.r = sum_vt(aH_); oH.m,oH.d = vt_(oH.dTT)
            frame.H = frame.H[:i] + [oH] + frame.H[i+len(aH_):]  # same_oF_levs, open tail follows
            split_oF_(); cluster_oF_()  # recompute code structure = the oF_ coefs
    oF = CoF.get(); oF.N_=H; oF.dTT=frame.dTT; oF.c=frame.c; oF.r=frame.r

    def base_tile(y,x,T=0):  # pixels at elev=0, lower frame_H above that
        if fH:
            T = frame_H(image, y,x, Ly,Lx, Y,X, rV, max_elev, fH-1)
        else:  # flat H
            if not T:  # pixels at elev=0
                T = frame_blobs_root(comp_pixel(image[y:y + Ly, x:x + Lx]), rV)
                T = vect_edge(T, rV)  # form, trace PP_:
        if T and not fH:  # pixel-tile, sub-frames keep their own real box
            T.yx = np.array([y+Ly//2, x+Lx//2]); T.box = np.array([y,x, min(y+Ly,Y),min(x+Lx,X)]); T.span = np.hypot(Ly,Lx) / 2
        return T

def add_Nt(G, Nt):  # in sum2G and trans_cluster

    yx_ = []; C = G.c + Nt.c  # but G is not empty in trans_comp?
    for N in Nt.N_:
        N.fin = 1; N.root = G; c = N.c
        if hasattr(N,'m_'):
            if not hasattr(G,'m_'): G.root_,G.m_,G.d_ = [],[],[]  # or G.rm_,G.rd_?
            G.C_ += N.root_; G.m_+= N.m_; G.d_+= N.d_  # Ct || Nt
        G.kern = (G.kern*(C-c) + N.kern*c) / C  # massive?
        G.box = extend_box(G.box, N.box)
        yx_ = (G.yx*(C-c) + N.yx*c) / C  # replace with:
    G.yx = yx = np.mean(yx_, axis=0); dy_,dx_ = (np.array(yx_)-yx).T  # weigh by c?

def fill_frame(_iy,_ix, elev, T):  # seed tile is lower frame if elev else pixels

        frame = np.full((Ly,Lx),None, dtype=object)  # level scope
        cy,cx = (Ly-1)//2, (Lx-1)//2; y,x = cy,cx  # start=mean
        T_, PV__,C,R = [],np.zeros([Ly,Lx]),0,0  # tiles, maps to level frame
        while True:
            if not elev:  # init pixel-level seed tile
                T = base_tile(_iy,_ix)
            if T and sum(vt_(T.dTT,T.wTT*ttFrm)) * wFrm > (ave+avd)*(T.r+cFrm):  # complex gate, make incremental?
                frame[y,x]=T; T+=[T]  # loop adds one tile to level
                dy_dx = np.array([T.yx[0]-y, T.yx[1]-x])
                pTT = proj_N(T, np.hypot(*dy_dx), dy_dx, elev,T.c)
                if 0 < sum(vt_(pTT, T.wTT*ttFrm)) * wFrm < ave * cFrm:  # extend lev by combined proj T_
                    proj_focus(PV__,y,x, T)  # PV__+= pV__
                    pv__ = PV__.copy(); pv__[frame != None] = 0  # exclude processed
                    y, x = np.unravel_index(pv__.argmax(), PV__.shape)
                    if PV__[y,x] > ave:
                        iy = _iy+ (y-cy)*(T.box[2]-T.box[0]); ix = _ix+ (x-cx)*(T.box[3]-T.box[1])  # step by sub-frame footprint
                        nxt_T = base_tile(iy,ix); _elev = 0  # expand to current level
                        while nxt_T and _elev < elev:
                            t_,c,r = fill_frame(iy,ix, _elev, nxt_T)
                            if not t_: break  # eval / c,r?
                            nxt_T = sum2F(t_)
                            if cross_comp(nxt_T.Nt, rr=_elev)[0]: _elev += 1
                            else: break  # discard nxt_T?
                        T = nxt_T
                    else: break
                else: break
            else: break
        if T_:
            TT,C,R = sum_vt(T_, wTT=ttFrm); R+= elev
            if sum(vt_(TT,ttFrm))*(C*wFrm) > (ave+avd)*(R+cFrm):   # cancel weak frame, null T_,C,R
                return T_,C,R
        return [], 0, 0

def comp_N():
    if N.typ and m* wN_*c > ave*(r+cN_):  # skip PPs
        sub_ = []  # cross_comp N_| Fts -> top Lev
        if N.typ==1:  # L
            for _n,n in product(_N.N_,N.N_): sub_ += [comp_N(_n,n,r,c, rL=L)]  # CN L.nt, rL spec in comp.N
        else:
            for i,(_Ft,Ft, tnF) in enumerate(zip((_N.Nt,_N.Lt,_N.Bt,_N.Ct),(N.Nt,N.Lt,N.Bt,N.Ct),('Nt','Lt','Bt','Ct'))):
                if _Ft and Ft: sub_ += [comp_F(_Ft,Ft,r,L)]; r+=(i or 1)-1  # unique Nt,Lt, rL spec in comp_F
        # if sub_:
        for tLev in L.H:  # trans-levs added above
            ffb = 0
            for tFt in tLev.N_:  # trans-forks
                if tFt.N_:  # curr fb_, python-batched bottom-up
                    ffb = 1; tFt.dTT,tFt.c,tFt.r = sum_vt(tFt.N_)  # update tFt
            if ffb: tLev.dTT,tLev.c,tLev.r = sum_vt(tLev.N_)  # update tLev, no L+=Ft
'''
oF.m = val dTT, or
if proj comp: F.root.wTT *= rdpTT * (F.c/ F.root.c)  # c-weighted feedback
if select clust: (F.m - F.root.Lt.m) * (F.root.c-Ft.c)  # clustering value = loss reduction: root.Lt.m < selective F.m, add dval? 
'''
def cluster_P(_C_, _c, root):  # multi-seed mean shift: parallel centroid refine, may add proj_C

    cnt = 0  # r = root.r+1?:
    N_ = list(set([N for C in _C_ for N in C.N_]))  # all Ns are in all Cs
    Ln,Lc = len(N_),len(_C_); L=Lc*Ln  # localy constant
    _md__ = np.zeros((Ln, Lc, 2))  # NxC
    for j, N in enumerate(N_):
        for c,m,d in zip(N.root_,N.m_,N.d_):
            if c: i= _C_.index(c); _md__[j,i] = m,d
    while True:
        md__ = np.zeros_like(_md__); O = 0
        for j,N in enumerate(N_):
            for i,C in enumerate(_C_):
                TT,_ = base_comp(C,N); md__[j,i] = vt_(TT, ttcP)  # use rc?
                # sum overlap to eval for split ) merge?
        if O:   # merge eval by comp to reduce redundancy, split eval by higher filter
            mrg_ = []
            for _C, C in combinations(_C_,r=2):
                if _C in mrg_ or C in mrg_: continue  # not needed?
                c = min(C.c,_C.c); r = (C.r+_C.r)/2; dy_dx = _C.yx-C.yx; dist = np.hypot(*dy_dx)
                L = comp_N(_C,C,r,c, A=dy_dx,span=dist)
                if L.m*wF < avd*(r+cF): add2F(_C,C,1); mrg_ += [C]
            _C_= list(set(_C_)-set(mrg_))
        C_ = [sum2F(N_, root, md__[:,i,0], md__[:,i,1]) for i in range(Lc)]
        Mt = md__[:,:,0].sum()  # total V
        dM = np.abs(md__[:,:,0] -_md__[:,:,0]).sum()
        dD = np.abs(md__[:,:,1] -_md__[:,:,1]).sum()  # updates
        cnt += 1
        if Mt * (dM+dD) * (wcP*L) > ave * (root.r+ccP*L):
            _C_ = C_; _md__ = md__
        else: break  # weak * converged
    out_ = []
    for N in N_: N.root_ = []  # replace with out_ Cs:
    for i, _C in enumerate(C_):
        if _C.m > ave * _C.r:  # prune, add olp as stronger ms?
            N_,m_,d_ = [],[],[]
            for N, m,d in zip(_C.N_, md__[:,i,0], md__[:,i,1]):
                if m*N.c > ave*N.r: N_+=[N]; m_+=[m]; d_+=[d]
            if N_:
                C = sum2F(N_,root, m_,d_)
                for N in N_:
                    L = CN(typ=1, dTT=N.dTT,c=N.c,r=N.r,m=N.m,d=N.d, span=np.hypot(*(dy_dx:=C.yx-N.yx)), angl=[dy_dx,np.sign(N.dTT[1]@ttcN[1])])
                    L.N_ = [N,C]; C.L_ += [L]
                out_ += [C]
    if out_:
        dCt = sum2F(list(set(_C_)-set(out_)))  # compress, out_ for CoF?
        return out_, dCt

def form_PP_(iP_, fd):  # form PPs of dP.valt[fd] + connected Ps val

    PPt_ = []; M, D = eps, eps; VerT = np.full((2,6),eps)

    for P in iP_: P.fin = 0
    for P in iP_:  # dP from link_ if fd
        if P.fin: continue
        _prim_ = P.prim; _lrim_ = P.lrim; B_ = []
        if fd: m, d = P.m, P.d  # summed verT, min L in dP
        else:  I,G,Dy,Dx,M,D,L = P.latT; m, d = M, G+abs(D)
        _P_ = {P}; link_ = set()
        verT = np.full((2,6),eps)
        while _prim_:
            prim_,lrim_ = set(),set()
            for _P,_link in zip(_prim_,_lrim_):
                if _P.fin: continue
                if [_link.m, _link.d][fd] > [ave,avd][fd]:
                    _P_.add(_P); link_.add(_link)
                    verT += _link.verT
                    if fd: _m, _d = _P.m, _P.d
                    else: _I,_G,_Dy,_Dx,_M,_D,_L = _P.latT; _m, _d = _M,_G+abs(_D)
                    m += _m; d += _d  # intra-P similarity and variance
                    prim_.update(set(_P.prim) - _P_)
                    lrim_.update(set(_P.lrim) - link_)
                    _P.fin = 1
                else: B_ += [_link]  # PP boundary-> comb_B
            _prim_, _lrim_ = prim_, lrim_
        M += m; D += d; VerT += verT
        PPt_ += [sum2PP(list(_P_), list(link_), list(set(B_)), m, d)]

    return PPt_, VerT, M, D

def comp_P_(edge):  # form links from prelinks

    edge.rng = 1
    for _P, pre_ in edge.pre__.items():
        for P in pre_:  # prelinks
            dy,dx = np.subtract(P.yx,_P.yx)  # between node centers
            if abs(dy)+abs(dx) <= edge.rng * 2: # <max Manhattan distance
                angle=[dy,dx]; distance=np.hypot(dy,dx)
                verT, m, d = comp_latT(_P.latT, P.latT, len(_P.dert_), len(P.dert_))
                dP = convert_to_dP(_P,P, verT, angle,distance, m, d)
                _P.rim += [dP]  # up only
                edge.dP_ += [dP]  # to form PPd_ by dval, separate from PPm_
    del edge.pre__

def comp_dP_(edge, _M):  # node_- mediated: comp node.rim dPs, call from form_PP_

    rM = _M / ave  # dP D borrows from normalized PP M
    for _dP in edge.dP_: _dP.prim = []; _dP.lrim = []
    for _dP in edge.dP_:
        D, M = _dP.d, _dP.m
        if D/M * rM > avd:
            _P, P = _dP.nt  # _P is lower
            rc = M/_M; minn = min(_M,M)
            for dP in P.rim:  # higher links
                if dP not in edge.dP_: continue  # skip removed node links
                verT, m, d = comp_verT(_dP.verT[1], dP.verT[1]*rc, minn)
                angle = np.subtract(dP.yx,_dP.yx)  # dy,dx of node centers
                distance = np.hypot(*angle)  # between node centers
                # up only:
                _dP.lrim += [convert_to_dP(_dP,dP, verT, angle, distance, m, d)]
                _dP.prim += [dP]

def convert_to_dP(_P,P, verT, angle, distance, m, d):

    link = CdP(nt=[_P,P], m=m, d=d, verT=verT, angle=angle, span=distance, yx=np.add(_P.yx, P.yx)/2)
    # Ps are dPs
    _P.verT+= link.verT; P.verT += link.verT
    _P.lrim += [link];   P.lrim += [link]
    _P.prim += [P];      P.prim +=[_P]
    link.L = min(_P.latT[-1],P.latT[-1]) if isinstance(_P,CP) else min(_P.L,P.L)  # P is CdP
    return link

def proj_TT(L, cos_d, dist, r, pTT, wTT, dec=1, fdec=0, frec=0):  # accumulate link pTT with iTT | eTT internally, L may be N

    dec = dist if fdec else ave ** (1+ dist*dec / L.span)  # not fully revised, ave = match decay rate / unit distance
    TT = np.array([L.dTT[0] * dec, L.dTT[1] * cos_d * dec])  # IxE angle alignment * decay?
    cert = abs(sum(vt_(TT,wTT)) * wPrj)  # approximation
    if cert > (ave+avd)*(r+cPrj): # certainty margin
        pTT+=TT; return
    if not frec:  # non-recursive
        for lev in L.Nt.N_:  # refine pTT
            proj_TT(lev, cos_d, dec, r+1, pTT, wTT, fdec=1, frec=1)
    pTT+=TT  # L.dTT is redundant to H, neither is redundant to Bt,Ct
    if L.Bt: # + trans-link tNt, tBt, tCt?
        TT = L.Bt.dTT
        if TT is not None:
            pTT += np.array([TT[0] * dec, TT[1] * cos_d * dec])

def get_gmap(gmap,n,gi=0):

    if isinstance(n,tuple):
        for c in n[1]: gi = get_gmap(gmap, c, gi)
        return gi
    if isinstance(n,ast.Call) and n.function.id == 'gv_':
        gmap[n.lineno] = gi
        return gi + 1
    return gi

class CoF(CF):
    name = "func"
    _cur = contextvars.ContextVar('oF')
    def __init__(f, **kw):
        super().__init__(**kw)
        f.call_ = kw.get('call_',[])  # called oFs only
        f.body = kw.get('body',[])  # static AST ops + CoF refs in source order
        f.fw,f.fc,f.fr = [kw.get(x,0) for x in ('fw','fc','fr')]  # fr if nested oF?
        f.caller_ = kw.get('caller_', [])
        f.gv_ = kw.get('gv_',[])  # gating vals per callee
        f.vt_ = kw.get('vt_',[])  # vals per call
    @staticmethod
    def get(): return CoF._cur.get()  # Frm?
    @staticmethod
    def traced(func):
        if getattr(func, 'wrapped', False): return func
        @wraps(func)
        def inner(*a, **kw):
            _CoF = CoF._cur.get(None)
            oF = CoF(nF=iF_[func.__name__], root=_CoF)
            i = iF_[func.__name__]; oF_[i].call_ += [oF]  # F_call_T_[i][oF.nF] += oF.dTT
            oF.gv_ = np.zeros(len(oF_[i].gv_map))
            if _CoF is not None:
                _CoF.call_ += [oF]
                oF_[iF_[func.__name__]].caller_.add(_CoF)  # for comp_caller_
            _oF = CoF._cur.set(oF)
            if out := func(*a, **kw):
                C = oF.c; TT=np.zeros((2,9)); R=0
                for tt,c,r in oF.vt_: w= c/(C or eps); TT+=tt*w; R+=r*w
                oF.dTT,oF.r = TT,R
                oF.w = vt_(oF.dTT)[0] + sum(oF.gv_)
            if oF.call_:
                tree = flat_(oF)  # if len(tree)-1?
                tt,_,r = sum_vt(tree); oF.wTT = cent_TT(tt, r)  # subtree dTT/r, not own
                if _CoF is not None:
                    if (j := F_call_i_[_CoF.nF].get(inspect.currentframe().f_back.f_lineno)) is not None:  # get callee site
                        F_call_T_[_CoF.nF][j] += oF.dTT  # add,c,r: results per callee to refine the code
            CoF._cur.reset(_oF)
            return out
        inner.wrapped = True
        return inner
    def __bool__(f): return bool(f.call_)

def gv_sites(fdef, fi=1):  # g_[i] <-> oF.gv_[i], same order as build
    g_ = []                # eval gate / oF.gv_[i] after accumulation
    def rec(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == 'gv_':
                if fi: node.args[-1].value = len(g_)   # overwrite gv_ index
                g_.append(node)
            if node.func.id in iF_: return   # build substitutes callee, no descent
        for child in ast.iter_child_nodes(node): rec(child)
    rec(fdef); return g_

def ffeedback2(frame, aTT,oTT, aL,oL):  # recompute filters from regime drift; fork: reform oF_ on cross-regime drift

    global ave, avd
    dTT = dc = dr = 0
    _ac,_ar = (aL.c,aL.r) if aL else (0,0); _oc,_or = (oL.c,oL.r) if oL else (0,0)
    # H init @ 1st term:
    if aL := pack_seg(frame,'aH',wBac, cBac, aTT):  # L: new level
        dTT = aL.dTT-aTT; aTT=aL.dTT; dc= aL.c-_ac; dr= aL.r-_ar
        ave, avd = vt_(aTT)
        # filters *= ave
        if oL := pack_seg(frame,'oH', wBac, cBac**2, oTT):
            dTT += oL.dTT-oTT; oTT=oL.dTT; dc+=oL.c-_oc; dr+=oL.r-_or
            split_oF_(); cluster_oF_()  # add eval?
            # reform oF_
    Fvt_([frame],dTT,dc,dr)
    return frame, aTT, oTT, aL, oL
