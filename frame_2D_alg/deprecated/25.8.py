def comp_slice(edge, rV=1, ww_t=None):  # root function

    global ave, avd, wM, wD, wI, wG, wA, wL, ave_L, ave_PPm, ave_PPd, w_t
    ave, avd, ave_L, ave_PPm, ave_PPd = np.array([ave, avd, ave_L, ave_PPm, ave_PPd]) / rV  # projected value change
    if np.any(ww_t):
        w_t = [[wM, wD, wI, wG, wA, wL]] * ww_t
        # der weights
    for P in edge.P_:  # add higher links
        P.vertuple = np.zeros((2,6))
        P.rim = []; P.lrim = []; P.prim = []
    edge.dP_ = []
    comp_P_(edge)  # vertical P cross-comp -> PP clustering, if lateral overlap
    PPt, mvert, mEt = form_PP_(edge.P_, fd=0)  # all Ps are converted to PPs
    if PPt:
        edge.node_ = PPt
        comp_dP_(edge, mEt)
        edge.link_, dvert, dEt = form_PP_(edge.dP_, fd=1)
        edge.vert = mvert + dvert
        edge.Et = Et = mEt + dEt
    return PPt

def rroot(N):  # get top root
    R = N.root
    while R.root and R.root.rng > N.rng: N = R; R = R.root
    return R

def comb_cent_(nG, rc):

    cent_, node_, cM = [], [], 0
    for g in nG.N_:
        if g.cent_:
            node_.extend({n for c in g.cent_[0] for n in c.N_})
            cent_.extend(g.cent_[0]); cM += g.cent_[1]
    if cM > ave * rc * loopw:
        comp_(node_, rc, fC=1)  # extend rims of mo_-reinforced nodes, if not in seen_
        cluster_C_(nG, cent_,rc, fext=1)  # use combined cent_ as iN_

def cluster_C_(root, N_, rc, fdeep=0, fext=0):  # form centroids by clustering exemplar surround, drifting via rims of new member nodes

    def comp_C(C, n):
        _,et,_ = base_comp(C,n, rc, C.wTT)
        if fdeep:
            if val_(et,1,len(n.derH)-2,rc):
                comp_H(C.derH, n.derH, n.Et[2]/C.Et[2], et)
            for L in C.rim:
                if L in n.rim: et += L.Et  # overlap
        return et[0]/et[2] if et[0]!=0 else 1e-7

    _C_, _N_ = [], []
    for N in N_:
        if not N.exe: continue  # exemplars or all
        if fext:
            C = N  # extend centroid
            C._N_ = [n for _N in C.N_ for l in _N.rim for n in l.N_ if n is not _N and l not in C.L_]  # not tested links
        else:  # init centroid
            C = cent_attr(Copy_(N,root, init=2), rc); C.N_ = [N]; # C.wTT = np.ones((2,8)) why we are init wTT to 1 again?
            C._N_ = [n for l in N.rim for n in l.N_ if n is not N]  # core members + surround for comp to N_ mean
        _N_ += C._N_; _C_ += [C]
    N__ = copy(set(_N_))  # pack all possible ns, including those out of graph's ns
    # reset:
    for n in set(root.N_+_N_): n.C_,n.mo_, n._C_,n._mo_ = [],[], [],[]  # aligned pairs
    # reform C_, refine C.N_s, which may extend beyond root:
    while True:
        C_, cnt,mat,olp, Dm,Do = [],0,0,0,0,0; Ave = ave * rc * loopw
        _Ct_ = [[c, c.Et[0]/c.Et[2] if c.Et[0] !=0 else 1e-7, c.olp] for c in _C_]
        for _C,_m,_o in sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True):
            if _m > Ave*_o:
                C = sum_N_(_C.N_, root=root, fC=1, rc=rc)  # add olp in N.mo_ to C.olp?
                _N_,_N__,L_, mo_, M,O, dm,do = [],[],[],[],0,0,0,0  # per C
                for n in _C._N_:  # core+ surround
                    if C in n.C_: continue
                    m = comp_C(C,n)  # val,olp / C:
                    o = np.sum([mo[0]/ m for mo in n._mo_ if mo[0]>m/3])  # overlap = higher-C inclusion vals / current C val
                    cnt += 1  # count comps per loop
                    if m > Ave * o:
                        _N_+=[n]; L_+=n.rim; M+=m; O+=o; mo_ += [np.array([m,o])]  # n.o for convergence eval
                        _N__ += [_n for l in n.rim for _n in l.N_ if _n is not n]  # +|-ls for comp C
                        if _C not in n._C_: dm += m; do += o  # not in extended _N__
                    elif _C in n._C_:
                        __m,__o = n._mo_[n._C_.index(_C)]; dm +=__m; do +=__o
                if M > Ave * len(_N_) * O:
                    for n, mo in zip(_N_,mo_): n.mo_ += [mo]; n.C_ += [C]
                    # we need to init C._L_ in sum_N?  Since we pack need to pack it into C.L_
                    C.M = M; C.L_+= C._L_; C._L_= list(set(L_)); C.N_+= _N_; C._N_= list(set(_N__))  # core, surround elements
                    C_+=[C]; mat+=M; olp+=O; Dm+=dm; Do+=do   # new incl or excl
                    for n in _N_: add_N(C, n, fC=1)  # we need this to update C params? Else their derTT won't be updated too
                else:
                    for n in _C._N_:
                        for i, c in enumerate(n.C_):
                            if c is _C: n.mo_.pop(i); n.C_.pop(i); break  # remove mo mapping to culled _C
            else: break  # the rest is weaker
        if Dm > Ave * cnt * Do:  # dval vs. dolp, overlap increases as Cs may expand in each loop?
            _C_ = [cent_attr(C,rc) for C  in C_]
            for n in N__: n._C_ = n.C_; n._mo_= n.mo_; n.C_,n.mo_ = [],[]  # new n.C_s, combine with vo_ in Ct_?
        else:  # converged
            break
    if  mat > Ave * cnt * olp:
        root.cent_ = (set(C_),mat)
        # cross_comp, low overlap eval in comp_node_?

def base_comp(_N,N, rc, wTT=None):  # from comp Et, Box, baseT, derTT
    """
    pairwise similarity kernel:
    m_ = element‑wise min(shared quantity) / max(total quantity) of eight attributes (sign‑aware)
    d_ = element‑wise signed difference after size‑normalisation
    DerTT = np.vstack([m_,d_])
    Et[0] = Σ(m_ * wTTf[0])    # total match (≥0) = relative shared quantity
    Et[1] = Σ(|d_| * wTTf[1])  # total absolute difference (≥0)
    Et[2] = min(_n, n)        # min accumulation span
    """
    _M,_D,_n = _N.Et; M,D,n = N.Et
    dn = _n - n; mn = min(_n,n) / max(_n,n)  # or multiplicative for ratios: min * rn?
    rn = _n / n  # size ratio, add _o/o?
    o, _o = N.olp, _N.olp
    o*=rn; do = _o - o; mo = min(_o,o) / max(_o,o)
    M*=rn; dM = _M - M; mM = min(_M,M) / max(_M,M)
    D*=rn; dD = _D - D; mD = min(_D,D) / max(_D,D)
    # skip baseT?
    _I,_G,_Dy,_Dx = _N.baseT; I,G,Dy,Dx = N.baseT  # I, G|D, angle
    I*=rn; dI = _I - I; mI = abs(dI) / aI
    G*=rn; dG = _G - G; mG = min(_G,G) / max(_G,G)
    mA, dA = comp_angle((_Dy,_Dx),(Dy*rn,Dx*rn))  # current angle if CL
    # comp dimension:
    if N.fi: # dimension is n nodes
        _L,L = len(_N.N_), len(N.N_)
        mL,dL = min(_L,L)/ max(_L,L), _L - L
    else:  # dimension is distance
        _L,L = _N.span, N.span   # dist, not cumulative, still positive?
        mL,dL = min(_L,L)/ max(_L,L), _L - L
    # comp depth, density:
    # lenH and ave span, combined from N_ in links?
    m_,d_ = np.array([[mM,mD,mn,mo,mI,mG,mA,mL], [dM,dD,dn,do,dI,dG,dA,dL]])
    # comp derTT:
    id_ = N.derTT[1] * rn; _id_ = _N.derTT[1]  # norm by span
    if N.fi:
        dd_ = _id_-id_  # dangle insignificant for nodes
    else:  # comp angle in link Ns
        _dy,_dx = _N.angl; dy, dx = N.angl
        dot = dy * _dy + dx * _dx  # d_orientation
        leP = np.hypot(dy, dx) * np.hypot(_dy,_dx)
        cos_da = dot / leP if leP else 1  # keep in [–1,1]
        rot = 1 if dy * _dx - dx * _dy >= 0 else -1  # +1 CW, −1 CCW
        # projected abs diffs * combined input and dangle sign:
        dd_ = np.abs(_id_-id_) * ((1-cos_da)/2) * np.sign(_id_) * np.sign(id_) * rot
    _a_,a_ = np.abs(_id_),np.abs(id_)
    md_ = np.divide( np.minimum(_a_,a_), np.maximum.reduce([_a_,a_, np.zeros(8) + 1e-7]))  # rms
    md_ *= np.where((_id_<0)!=(id_<0), -1,1)  # match is negative if comparands have opposite sign
    m_ += md_; d_ += dd_
    DerTT = np.array([m_,d_])  # [M,D,n,o, I,G,A,L]
    if wTT is None:  wm_,wd_ = wTTf  # global only
    else:  # centroid _N: combine, re-norm weights
        mP = wTTf[0]*wTT[0]; mS = np.sum(mP); wm_ = mP * (8/mS) if mS>0 else np.ones(8)
        dP = wTTf[1]*wTT[1]; dS = np.sum(dP); wd_ = dP * (8/dS) if dS>0 else np.ones(8)
    Et = np.array([np.sum(m_* wm_ * rc),
                   np.sum(np.abs(d_ * wd_ * rc)), # * ffeedback * correlation if comp_C
                   min(_n, n)])  # shared scope?
    return DerTT, Et, rn

    def comp_cos(C, n):
        _wd_, wd_ = C.derTT[1] * wTTf[1], n.derTT[1] * wTTf[1]
        cos_M = (_wd_ @ wd_) / (np.linalg.norm(_wd_) * np.linalg.norm(wd_) + 1e-12)
        # [-1,1] -> [0,1]:
        return 0.5 * (cos_M + 1.0)  # * wTT from cent_attr?

# replace with proj_L_:
def proj_span(N, dir_vec):  # project N.span in link direction, in proportion to internal collinearity N.mang

    cos_proj = np.dot(N.angl, dir_vec) / (np.linalg.norm(N.angl) * np.linalg.norm(dir_vec))  # cos angle = (a⋅b) / (||a||⋅||b||)
    oriented = (1 - N.mang) + (N.mang * 2 * abs(cos_proj))  # from 1/span to 2?
    #          (1 - N.mang) * N.span + N.mang * (1 + (2 * N.span - 1) * abs(cos_angle)), * elongation?
    return N.span * oriented  # from 1 to N.span * len L_?

def comp_(iN_, rc, fi=1, fC=0, fdeep=0):  # comp pairs of nodes or links within max_dist

    N__,L_, ET = [],[], np.zeros(3); rng,olp_,_N_ = 1,[],copy(iN_)
    # frng: range-band?
    while True:  # _vM
        Np_ = []  # [G pair + co-positionals], for top-nested Ns, unless cross-nesting comp:
        for _N, N in combinations(_N_, r=2):
            if _N in N.seen_ or len(_N.nH) != len(N.nH):  # | root.nH: comp top nodes only
                continue
            dy,dx = np.subtract(_N.yx,N.yx); dist = np.hypot(dy,dx); dy_dx = np.array([dy,dx])
            radii = proj_span(_N, dy_dx) + proj_span(N, dy_dx)  # directionally projected
            # replace radii, vett with pL_tV, computed in main sequence:
            Np_ += [(_N,N, dy,dx, radii, dist)]
        N_,Et = [],np.zeros(3)
        for Np in Np_:
            _N,N, dy,dx, radii, dist = Np
            (_m,_d,_n), (m,d,n) = _N.Et,N.Et; olp = (_N.olp+N.olp)/2  # add directional projection?
            if fC: _m += np.sum([i[0] for i in _N.mo_]); m += np.sum([i[0] for i in N.mo_])  # matches to centroids
            # vett = _N.et[1-fi]/_N.et[2] + N.et[1-fi]/N.et[2]  # density term
            pL_tV = proj_L_(_N,N) if fdeep else 0
            # directional N.L_, N.rim val combination of variable depth, += pL_t of mediated nodes?
            mA,dA = comp_angle(_N.angl, N.angl)
            conf = (_N.mang + N.mang) / 2; mA *= conf; dA *= conf  # N collinearity | angle confidence
            if fi: V = ((_m + m + mA) * intw + pL_tV) / (ave*(_n+n))
            else:  V = ((_d + d + mA) * intw + pL_tV) / (avd*(_n+n))
            max_dist = adist * (radii / aveR) * V
            # min induction distance:
            if max_dist > dist or set(_N.rim) & set(N.rim):  # close or share matching mediators
                Link = comp_N(_N,N, rc, L_=L_, angl=dy_dx, span=dist, dang=[mA,dA], et=np.array([mA,dA,1]), fdeep = dist < max_dist/2, rng=rng)
                if val_(Link.Et, aw=loopw*olp) > 0:
                    Et += Link.Et; olp_ += [olp]  # link.olp is the same with o
                    for n in _N,N:
                        if n not in N_ and val_(n.et, aw=rc+rng-1+loopw+olp) > 0:  # cost+ / rng?
                            N_ += [n]  #-> rng+ eval
        if N_:
            N__ += [N_]; ET += Et
            if val_(Et, mw=(len(N_)-1)*Lw, aw=loopw * (sum(olp_) if olp_ else 1)) > 0:  # current-rng vM
                _N_ = N_; rng += 1; olp_ = []  # reset
            else: break  # low projected rng+ vM
        else: break
    return N__, L_, ET

def cos_comp(C, N, fdeep=0):  # eval cosine similarity for centroid clustering

    mw_ = wTTf[0] * np.sqrt(C.wTT[0]); mW = mw_.sum()  # wTT is derTT correlation term from cent_attr, same for base attrs?
    _M,_D,_n = C.Et; M,D,n = N.Et  # for comp only
    rn = _n/n  # size norm, +_o/o?
    o,_o = N.olp, C.olp
    _I,_G,_Dy,_Dx = C.baseT; I,G,Dy,Dx = N.baseT  # I, G|D, angle
    _L,L = (len(C.N_), len(N.N_)) if N.fi else (C.span, N.span)  # dimension, no angl? positive?
    bw_ = mw_[[0,1,2,3,4,5,7]]; bW = bw_.sum()  # exclude ang w
    BaseT = np.array([_M,_D,_n,_o,_I,_G,_L]); wD_ = BaseT * bw_  # all
    baseT = np.array([ M, D, n, o, I, G, L]); wd_ = baseT * bw_ * rn
    denom = np.linalg.norm(wD_) * np.linalg.norm(wd_) + 1e-12
    Mb = ((wD_ @ wd_) / denom + 1) / 2  # element-wise normalized dot-product, in 0:1
    # separate:
    mA,_ = comp_angle((_Dy,_Dx),(Dy*rn, Dx*rn))
    Mb = (Mb*bW + mA*mw_[6]) / (bW+mw_[6])  # combine for weighted ave cos
    # comp derT:
    dw_ = wTTf[1] * np.sqrt(C.wTT[1]); dW = dw_.sum()
    wD_ = C.derTT[1] * dw_; wd_ = N.derTT[1] * dw_ * rn
    denom = np.linalg.norm(wD_) * np.linalg.norm(wd_) + 1e-12
    Md = ((wD_ @ wd_) / denom + 1) / 2
    tW = mW + dW
    MT = (Mb*mW + Md*dW) / (tW + 1e-12)  # weighted baseT + derT
    if fdeep:
        if MT * tW * (len(N.derH)-2) > ave * ((o+_o)/2):
            MH = comp_H_cos(C.derH, N.derH, rn, C.wTT)
            ddW = tW * (dW / mW)  # ders W < inputs W
            MT = (MT*tW + MH*ddW) / (tW+ddW + 1e-12)
        # add N.rim overlap?
    return MT

def min_comp(_N,N, rc):  # from comp Et, Box, baseT, derTT
    """
    pairwise similarity kernel:
    m_ = element‑wise min(shared quantity) / max(total quantity) of eight attributes (sign‑aware)
    d_ = element‑wise signed difference after size‑normalisation
    DerTT = np.vstack([m_,d_])
    Et[0] = Σ(m_ * wTTf[0])    # total match (≥0) = relative shared quantity
    Et[1] = Σ(|d_| * wTTf[1])  # total absolute difference (≥0)
    Et[2] = min(_n, n)        # min accumulation span
    """
    _M,_D,_n = _N.Et; M,D,n = N.Et
    dn = _n - n; mn = min(_n,n) / max(_n,n)  # or multiplicative for ratios: min * rn?
    rn = _n / n  # size ratio, add _o/o?
    o, _o = N.olp, _N.olp
    o*=rn; do = _o - o; mo = min(_o,o) / max(_o,o)
    M*=rn; dM = _M - M; mM = min(_M,M) / max(_M,M)
    D*=rn; dD = _D - D; mD = min(_D,D) / max(_D,D)
    # comp baseT:
    _I,_G,_Dy,_Dx = _N.baseT; I,G,Dy,Dx = N.baseT  # I, G|D, angle
    I*=rn; dI = _I - I; mI = abs(dI) / aI
    G*=rn; dG = _G - G; mG = min(_G,G) / max(_G,G)
    mA, dA = comp_angle((_Dy,_Dx),(Dy*rn,Dx*rn))  # current angle if CL
    # comp dimension:
    if N.fi: # dimension is n nodes
        _L,L = len(_N.N_), len(N.N_)
        mL,dL = min(_L,L)/ max(_L,L), _L - L
    else:  # dimension is distance
        _L,L = _N.span, N.span  # projectable in links?
        mL,dL = min(_L,L)/ max(_L,L), _L - L
    # comp depth, density:
    # lenH and ave span, combined from N_ in links?
    m_,d_ = np.array([[mM,mD,mn,mo,mI,mG,mA,mL], [dM,dD,dn,do,dI,dG,dA,dL]])
    # comp derTT:
    id_ = N.derTT[1] * rn; _id_ = _N.derTT[1]  # norm by span
    if N.fi:
        dd_ = _id_-id_  # dangle insignificant for nodes
    else:  # comp angle in link Ns
        _dy,_dx = _N.angl; dy, dx = N.angl
        dot = dy * _dy + dx * _dx  # orientation
        leP = np.hypot(dy,dx) * np.hypot(_dy,_dx)
        cos_da = dot / leP if leP else 1  # keep in [–1,1]
        rot = 1 if dy * _dx - dx * _dy >= 0 else -1  # +1 CW, −1 CCW
        # projected abs diffs * combined input and dangle sign:
        dd_ = np.abs(_id_-id_) * ((1-cos_da)/2) * np.sign(_id_) * np.sign(id_) * rot
    _a_,a_ = np.abs(_id_),np.abs(id_)
    md_ = np.divide( np.minimum(_a_,a_), np.maximum.reduce([_a_,a_, np.zeros(8) + 1e-7]))  # rms
    md_ *= np.where((_id_<0)!=(id_<0), -1,1)  # match is negative if comparands have opposite sign
    m_ += md_; d_ += dd_
    DerTT = np.array([m_,d_])  # [M,D,n,o, I,G,A,L]
    Et = np.array([np.sum(m_* wTTf[0] * rc),
                   np.sum(np.abs(d_ * wTTf[1] * rc)), # frame ffeedback * correlation
                   min(_n, n)])  # shared scope?
    return DerTT, Et, rn

def cosv(Et, fi=1):  # convert Et to cosine-like similarity, add aw?

    m, d,_ = Et
    return m / (m+d) if fi else d / (m+d)  # m is redundant to d?

def comp_H_cos(H, h, rn, wTT):
    M = 0
    dw_ = wTTf[1] * np.sqrt(wTT[1])
    for Lay, lay in zip_longest(H, h, fillvalue=None):
        if Lay and lay:
            wD_ = Lay.derTT[1] * dw_
            wd_ = lay.derTT[1] * dw_ * rn
            denom = np.linalg.norm(wD_) * np.linalg.norm(wd_) + 1e-12
            M += 0.5 * ((wD_ @ wd_) / denom + 1.0)
    return M

