import numpy as np, inspect, contextvars
import ast
from copy import copy, deepcopy
from math import atan2, cos, pi  # from functools import reduce
from itertools import zip_longest, combinations, product  # from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, imread, comp_pixel, CBase
from slice_edge import slice_edge
from comp_slice import comp_slice, w_t
from meta_code import oF_,iF_,nF_,CF,CL,CN,CoF,wT,wTT, eps,eps_,ave,avd,decay, trace_func,parse_funcs, call_sites, split_oF,clust_oF_,inject_oF_,gv_, val_,sum_vt
'''
This is a main module of open-ended clustering algorithm, designed to discover empirical patterns of indefinite complexity. 
Lower modules cross-comp and cluster image pixels and blob slices(Ps), the input here is resulting PPs: segments of matching Ps.
Cycles of (generative cross-comp, compressive clustering, filter-adjusting feedback) should form hierarchical model of input stream: 

Cross-comp forms Miss and Match (min: shared_quantity for directly predictive params, else inverse deviation of miss or variation), in 2 forks:
rng+: incremental-range cross-comp nodes: edge segments at < max distance, cluster if they match. 
der+: incremental-derivation cross-comp links, from node cross-comp, if abs_diff * relative_adjacent_match

Clustering compressively groups the elements into compositional hierarchy, initially by pair-wise similarity or density thereof.
High-contrast links are correlation clustered to form a boundary per connectivity cluster of the nodes.
Each level may extend clustering through 4 increasingly fuzzy stages, each seeded by prior-stage output:

- select sparse exemplars to seed the clusters, top k for parallelization (get_exemplars), no clustering?
- connectivity-based agglomerative ( divisive clustering in cluster_N, with boundary link clustering in Bt
- centroid-based marginally fuzzy and extensible clustering, in divisive phase (cluster_C), or ave_linkage?
- centroid-parallel fully fuzzy FCM-like refine, if significant global overlap (cluster_P), no higher stage?

Clustering forms hierarchical graphs, each a dual tree of down-forking elements: node_H, and up-forking clusters: root_H:
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
Similar to neural dendritic input tree and axonal output tree, but with lateral cross-comp and nested param sets per layer.

Overall fitness function is predictive value of the model, estimated through multiple orders of projection:
initially summed match, refined by external projection in space and time (where match is combined with directional diffs),
then by comparing projected to actual match, testing the accuracy of prior cross_comp and clustering process, etc. 
Higher projection orders should be generated recursively from lower-order cross-comp, per sufficient number of newly formed clusters.

Feedback: projected match adjusts filters to maximize next marginal match, with coord filters selecting new input (in ffeedback()),
to be refined by cross_comp of co-projected patterns: "imagination, planning, action" in part 3)   
This is similar to backprop but sparse, and the filters only control operations, they are not weights on input parameters.

Higher-order feedback should modify the code by adding weights on code elements according to their contribution to projected match.
Code weights, currently nw,cw,Nw,Cw,specw, control skipping / recursing over corresponding functions ( blocks ( operations.
Also cross-comp and cluster (compress) code elements and function calls, real and projected, though much more coarsely than data?

notation:
prefix  f denotes flag
postfix t denotes tuple, multiple ts is a nested tuple
prefix  _ denotes prior of two same-name vars, multiple _s for relative precedence
postfix _ denotes array of same-name elements, multiple _s is nested array
capitalized vars are summed small-case vars
'''
wM,wD, wi, wG,wI,wa, wL,wS,wA = wT; Ly = Lx = 64  # fractal tile dims
cFrm,cX,cTrc,cN,cF, cE,ccN,ccC,ccP,csG, cBac,cPrj, cVct = (  # function complexity
wFrm,wX,wTrc,wN,wF, wE,wcN,wcC,wcP,wsG, wBac,wPrj, wVct ) = [F.fc for F in oF_]  # ave gain/call, init=cost
ttFrm,ttX,ttTrc,ttN,ttF, ttE,ttcN,ttcC,ttcP, ttsG,ttBac,ttPrj,ttVct = [F.wTT for F in oF_]  # attr weight tuples added per function

def FV_(F, tt,c,r):  # combine m,d per oF
    tF = oF_[F.nF]; tF.c += c
    V = val_(tt*tF.wTT) * (c/r); tF.V_ += [V]; return V

def cent_TT(dTT, r):  # EM-like weight attr matches | diffs by their match to the sum, recompute to convergence

    wTT,_wTT = [],np.ones((2,9)); coT = np.abs(dTT[0])+np.abs(dTT[1])  # complemented vals
    while True:
        for fd, _wT, dT in zip((0,1), _wTT, dTT):
            vT = np.abs(dT)  # if -m: wrong, or surprise value?
            rvT = vT / eps_(coT) * _wT  # weighted normalized vals
            mean = rvT.mean() or eps  # scalar
            invdev_ = np.minimum(rvT / mean, mean / eps_(rvT))
            wT = invdev_ / (invdev_.mean() or eps)   # mean(wT)=1
            wTT += [wT]
        if np.sum(np.abs(wTT-_wTT)) < ave * r:  # if np.linalg.norm(wT - _wT, 1) < r?
            break
        _wTT = np.array(wTT); wTT = []
    return _wTT  # single-mode dTT, extend to 2D-3D lev cycles in H, cross-level param max / centroid?
''' 
  cycle:
- cross-comp nodes, evaluate incremental-derivation cross-comp of new >ave difference links, recursively. 
- connectivity-cluster from exemplars by >ave match links, correlation-cluster links by >ave diff
- form core+contour objects, divisive sub-clustering, higher-composition cross_comp
- centroid-cluster, from exemplars via their rim within a frame, increasing overlap
- parallel fully-fuzzy centroid refining if >min overlap
- forward: selective extend cross-comp, clustering across tiles, re-order centroids by eigenvalues
- feedback filter updates 
'''
def cross_comp(root,pL_,r,nF='Nt',dF=None,rL=None, fall=1):

    L_,N_ = [],[]
    for dist, dy_dx, _N,N, lc,lr, pTT,m,_ in pL_:
        if _N != N and (fall or (m>0 and gv_(m*(lc*wN/(lr*cN)) - ave* (r+cN)))):
            Link = comp_N(_N,N, lr,lc, full=not dF, A=dy_dx, span=dist, rL=root)
            Link.rTT = np.abs(pTT-Link.dTT) / eps_(Link.dTT); L_+=[Link]; N_+=[_N,N]  # relative prediction error/oF
    if L_:
        Lt = root.Lt = sum2F(L_,root); m,d,tt,c,r = Lt.m,Lt.d,Lt.dTT,Lt.c,Lt.r; FV_(CoF.get(),tt,c,r)
        if dF:  # comp_F: no cluster_,recursion
            add2F(dF,CF(N_=L_,m=m,d=d,dTT=tt,c=c,r=r),merge=1); add2F(rL,dF,merge=2); return
        for N in (N_:= list(set(N_))):
            N.Rt = sum2F(N.rim,root=N,nF='Rt') if N.rim else CF(root=N,nF='Rt')
        if gv_(val_(tt,ttcN) * (c*wcN/(r*ccN)) * ((len(L_)-1)*wL) - ave):
            G_,med_ = cluster_(root, get_exemplars(N_,r,c),r,c)
            for fsub,n_,R in ((1,med_,root.H[0]), (0,G_,root)):  # sub+ xcomp new G_, G_'N_| root H[-1] only?
                if n_ and gv_(R.m*R.c*wX - ave*(R.r+1+cX)):  # mdecay(L_)-decay? eval len n_?
                    g_ = cross_comp(R, proj_L_(combinations(n_,2),R,r), r,R.nF)
                    if fsub and g_: G_+=g_  # same level
            return G_  # for the above

def comp_N(_N,N, r,c, full=1, A=None,span=None, rL=None):

    def comp_H(_Nt,Nt, Link):  # tentative pre-comp, before comp N_?
        dH, tt, C,R = [],np.zeros((2,9)),0,0
        for _lev, lev in zip([_Nt]+_Nt.H, [Nt]+Nt.H):  # should be top-down
            if not (_lev and lev): continue  # skip empty level
            ltt = comp_derT(_lev.dTT[1],lev.dTT[1])
            lc = min(_lev.c,lev.c); lr = (_lev.r+lev.r)/2; m,d = val_(ltt,ttN,1)
            dH += [CF(dTT=ltt,m=m,d=d,c=lc,r=lr,root=Link)]
            tt += ltt*lc; C+=lc; R+=lr*lc
        return dH,tt,C, (r* Link.c+R)/ (Link.c+C)  # same norm for tt?

    L = CL(N_=[_N,N], c=c,r=r,root=rL)
    if full:
        dTT = base_comp(_N,N)[0]
        if span is None: span = np.hypot(*_N.yx - N.yx)
        yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx
        box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
        angl = [np.zeros(2) if A is None else A, np.sign(dTT[1] @ ttN[1])]
        L.yx=yx; L.box=box; L.span=span; L.angl=angl; L.kern=(_N.kern+N.kern)/2
    else: dTT = comp_derT(_N.dTT[1],N.dTT[1])
    m,d = val_(dTT, ttN,1); L.dTT,L.m,L.d = dTT,m,d
    if N.typ > 1 and gv_(m *(c/r) *wN - ave*(r+cN)):  # skip PPs, Nts?
        L.H = [Copy_(L)]  # lev0 to preserve resolution before adding deeper tLevs, min len H = 2
        htt,hc,hr = np.zeros((2,9)),0,0; dn_ = []  # cross_comp N_| Ft_-> top tLev
        for _n,n in product(_N.N_,N.N_):  # breadth first per N_ batch
            dH,dtt,dc,dr = comp_H(_n.Nt, n.Nt, L)
            add_H(L.H,dH,L); htt+=dtt; hc+=dc; hr+=dr
        if hc: L.dTT = (L.dTT*L.c+htt)/ (L.c+hc); L.m,L.d = val_(L.dTT,ttN,1); L.r += hr
        if N.typ < 3:  # L | C | Nt, merge?
            for _n,n in product(_N.N_,N.N_): dn_ += [comp_N(_n,n,r,c, rL=L)]  # CN L.nt, rL spec in comp.N
        else:  # CN
            for i,(_Ft,Ft, tnF) in enumerate(zip((_N.Nt,_N.Lt,_N.Bt,_N.Ct),(N.Nt,N.Lt,N.Bt,N.Ct),('Nt','Lt','Bt','Ct'))):
                if _Ft and Ft: dn_ += [comp_F(_Ft,Ft,r,L)]; r+=(i or 1)-1  # unique Nt,Lt, rL spec in comp_F
        if dn_:
            [add_H(L.H, d.H, L) for d in dn_ if d.H]  # lower levs
            L.H += [sum2F(dn_,L)]  # top lev
        # merge if no or weak Bt? comp x fork, H levs?
    if full:
        for n, _n in (_N,N),(N,_N): n.rim += [L]
    FV_(CoF.get(), L.dTT, L.c, L.r)
    # or merge N -> _N?
    return L

def comp_F(_F, F, ir=0, rL=None):

    ddTT = comp_derT(_F.dTT[1],F.dTT[1]); c= min(_F.c,F.c); r=(_F.r+F.r)/2
    m,d = val_(ddTT,ttF,1); r+=ir
    dF = CF(dTT=ddTT, m=m,d=d,r=r,c=c, nF=F.nF)
    if _F.nF == F.nF:  # sub-comp, no comp F.Lt: included in F.dTT?
        _N_,N_=_F.N_,F.N_; nF=F.nF; l=nF=='Lt'
        if  _N_ and N_:
            if l: Np_ = [[_n,n] for _n,n in zip(_N_,N_) if _n and n]  # same forks
            else: Np_ = list(product(_N_,N_))  # pairs
            L = len(Np_)-1
            tt = (rL.dTT*rL.c + ddTT*dF.c) / (rL.c+dF.c)
            if gv_(val_(tt,ttF) * (c/r) * (wF*L) - ave* (r+cF*L)):
                if l: L_= [L for Np in Np_ for L in comp_F(*Np, r,rL=dF).N_]; TT,C,R = sum_vt(L_, wTT=ttF)
            else:
                if nF != 'Nt': [(F2N(_N),F2N(N)) for _N,N in Np_]  # convert for comp_N_ below
                cross_comp(rL, proj_L_(Np_,2,r), r,nF,dF,rL)  # root=2: N pair span computes decay?
    FV_(CoF.get(), dF.dTT, dF.c, dF.r)
    return dF  # no cross-fork N_, no L ext updates?

def base_comp(_N,N):  # comp Et, kern, extT, dTT

    _M,_D,_C = _N.m,_N.d,_N.c; _I,_G,_Dy,_Dx =_N.kern; _L = len(_N.N_)  # N_: density within span
    M, D, C = N.m, N.d, N.c; I, G, Dy, Dx = N.kern; L = len(N.N_)
    rn = C/_C; mA=dA=0
    _pars = np.array([_M*rn,_D*rn,_C,_I*rn,_G*rn, [_Dy,_Dx],_L*rn,_N.span], dtype=object)  # Et, kern, extT
    pars  = np.array([M,D,C, (I,wI),G, [Dy,Dx], L,(N.span,wS)], dtype=object)
    if hasattr(_N,'angl') and _N.angl and N.angl:  # mostly Ls, some Gs
        mA,dA = comp_A(_N.angl[0]*_N.angl[1], N.angl[0]*N.angl[1])
    m_,d_ = comp(_pars,pars, mA,dA)  # M,D,n, I,G,a, L,S,A
    dm_,dd_ = comp_derT(_N.dTT[1], N.dTT[1])

    return np.array([m_+dm_,d_+dd_]), rn  # or rm, rv?
'''
    if np.hypot(*_N.angl[0])*_N.mang + np.hypot(*N.angl[0])*N.mang > ave*wA:  # aligned L_'As, mang *= (len_H)+fi+1
    mang = (rn*_N.mang + N.mang) / (1+rn)  # ave, weight each side by rn
    align = 1 - mang* (1-mA)  # in 0:1, weigh mA 
'''
def comp_derT(_i_,i_):

    d_ = _i_ - i_  # both arrays, no angles or massless params
    m_ = np.minimum( np.abs(_i_), np.abs(i_))
    m_[(_i_<0) != (i_<0)] *= -1  # negate opposite signs in-place
    return np.array([m_,d_])

def comp(_pars, pars, meA,deA):  # compute m_,d_ from inputs or derivatives

    m_,d_ = [],[]
    for _p, p in zip(_pars, pars):
        if isinstance(_p, list):  # vector angle
            mA, dA = comp_A(_p, p)
            m_ += [mA]; d_ += [dA]
        elif isinstance(p, tuple):  # massless: I|S avd in p only
            p, avd = p
            d = _p - p
            m_ += [avd - abs(d)]  # + complement max(avd,ad)?
            d_ += [d]
        else:  # massive
            _a,a = abs(_p), abs(p)
            m_ += [min(_a,a) if (_p<0)==(p<0) else -min(_a,a)]  # + complement max(_a,a) for +ves?
            d_ += [_p - p]
    # for general mass in 0:1, m = dir_m * mass + inv_m * (1-mass)?
    return np.array(m_+[meA]), np.array(d_+[deA])

def comp_A(_A,A):

    dA = atan2(*_A)- atan2(*A)  # * direction
    if   dA > pi: dA -= 2 * pi  # rotate CW
    elif dA <-pi: dA += 2 * pi  # rotate CCW
    '''  or 
    den = np.hypot(*_A) * np.hypot(*A) + eps
    mA = (_A @ A / den +1) / 2  # cos_da in 0:1, no rot = 1 if dy * _dx - dx * _dy >= 0 else -1  # +1 CW, −1 CCW, ((1-cos_da)/2) * rot?
    '''
    return (cos(dA)+1) /2, dA/pi  # mA in 0:1, dA in -1:1, or invert dA, may be negative?

def get_exemplars(N_,_r,_c):  # no need for _c? multi-layer non-maximum suppression -> sparse clustering seeds, medoids if N.Ct?

    for n in N_:
        rc = sum(r[0].c for r in n.root_); C = n.c + rc
        n.w = ((n.Rt.m * n.c) + sum([r[1]*r[0].c for r in n.root_]))/C
        # combined lateral and vertical match
    N_= sorted(N_, key=lambda n: n.w, reverse=True); E_,Inh_ = set(),set()
    for rdn, N in enumerate(N_, start=1):  # strong-first
        inh_ = list(Inh_ & set(N.rim))  # stronger Es in N.rim
        oM = sum_vt(inh_,fm=1, wTT=ttE)[0] if inh_ else 0
        oV = oM / (N.Rt.m or eps)  # relative olp V
        if N.Rt.m * N.c * wE > ave* (_r+rdn+cE+oV):
            E_.add(N); N.exe = 1  # point cloud of focal nodes
            Inh_.update(set(N.rim))  # extend inhibition zone
        else:
            break  # the rest of N_ is weaker, trace via rims
    if E_: FV_(CoF.get(), *sum_vt(list(E_)))
    else:  E_ = set([N_[0]]); N_[0].exe=1  # no gain, no inhibition, any N can be seed
    return E_

def nt_vt(n,_n):
    M, D = 0,0  # exclusive match, contrast
    for l in set(n.rim+_n.rim):
        if l.m > 0:   M += l.m
        elif l.d > 0: D += l.d
    return M, D

# astra draft:
def cluster_loop(G_,C_,root__,_r, seed_,pack,prioritize):  # one round; helpers retain cluster_ scope

    F = CoF.get()
    old_ = {T.i:(T.typ,{n:(m,d,T.r_.get(n,0)) for n,m,d in zip(T.N_,T.m_,T.d_)}) for T in G_+C_}
    for T in G_+C_: T.fin=0  # continuation is global
    G_ = cluster_N(G_,_r,root__)
    for G in G_:
        if G.d * (G.c*wcC/(G.r*ccC)) * ((len(G.N_)-1)*wL) > avd:
            for N in G.N_:
                if N not in seed_ and N.m * (G.c*wcC/(G.r*ccC)) > ave:
                    C_ += [pack([N],root=G.root,fseed=1,fC=1)]; seed_.add(N)
    C_ = cluster_C(C_,_r)
    T_ = G_+C_; root__ = prioritize(T_); front_ = []
    for T in T_:
        w,c = (wcC,ccC) if T.typ==2 else (wcN,ccN)
        if T.w*T.c*w*((len(T.N_)-1)*wL) > ave*T.r*c: front_ += [T]
    if len(front_) != len(T_): root__ = prioritize(front_)

    Up,Dr = 0,0; new_ = {T.i:T for T in front_}
    for i in old_.keys() | new_.keys():
        T = new_.get(i); typ = T.typ if T is not None else old_[i][0]
        md_ = old_.get(i,(typ,{}))[1]
        md = {n:(m,d,T.r_[n]) for n,m,d in zip(T.N_,T.m_,T.d_)} if T is not None else {}
        w,c = (wcC,ccC) if typ==2 else (wcN,ccN)
        for n in md_.keys() | md.keys():  # includes removed memberships and proposals
            _m,_d,_rn = md_.get(n,(0,0,0)); m,d,rn = md.get(n,(0,0,0))
            Up += n.c*w*(abs(m-_m)+abs(d-_d))
            Dr += ave*n.c*c*(rn-_rn)  # signed redundancy cost, in update-value units

    F.fc = oF_[F.nF].fc; call_ = list(F.call_)
    while call_:  # static AST estimates, once per executed traced call
        f = call_.pop(); F.fc += oF_[f.nF].fc; call_ += f.call_
    F.w = Up-Dr; oF_[F.nF].V_ += [F.w]
    G_ = [T for T in front_ if T.typ==3]; C_ = [T for T in front_ if T.typ==2]
    return G_,C_,root__,gv_(Up - ave*F.fc - Dr)

def cluster_(Ft, E_,_r,_c):  # fC(E): CC, else CN; returns G_,Ct,med_

    def pack(N_, nF='Nt', root=None, fseed=0, fC=0):  # summary only, no member or root-value updates
        m,d,dTT,c,r = sum_vt(N_, fm=1)
        T = (CF,(CN,CL)[fC])[fseed](N_=N_,nF=nF,root=root,wTT=Ft.wTT,dTT=dTT,m=m,d=d,c=c,r=r)
        if fseed:  # summary only if not fT, no member or root-value updates
            E = N_[0]; T.m,T.d = E.m,E.d  # raw vals for base_comp in cluster_C
            T.kern,T.yx,T.span,T.angl = copy(E.kern),copy(E.yx),E.span,copy(E.angl)
            T.L_,T.B_,T.m_,T.d_ = [],[],[E.m],[E.d]; T._N_ = list(set(n for L in E.rim for n in L.N_ if n is not E))
            T._r=_r+E.r; T.r=T._r; T.r_={}; T.olp=0; T.fin=0; T.i=T.id
            T.typ = 3-fC
        return T
    def prioritize(T_):  # stronger membership charges weaker; ties split the charge
        root__ = {}
        for T in T_:
            T.r_ = {}; w,c = (wcC,ccC) if T.typ==2 else (wcN,ccN)
            for n,m in zip(T.N_,T.m_):
                v = max(m,0)*w / (c*(_r+n.r)); root__.setdefault(n,[]).append((T,v))
        for n,rt_ in root__.items():
            for T,v in rt_: T.r_[n] = sum(_v/v if _v>v else .5 if _v==v else 0 for _T,_v in rt_ if _T is not T) if v>0 else 0
        for T in T_:
            T.olp = sum(n.c*T.r_[n] for n in T.N_) / sum(n.c for n in T.N_)
            T.r = T._r+T.olp
        return {n:[(T.i,v) for T,v in rt_] for n,rt_ in root__.items()}

    _C_ = {C for n in Ft.N_ for C,m,d in n.root_ if getattr(C,'source',None) is Ft}
    G_,C_,root__ = [],[],{}
    for i,E in enumerate(sorted(E_,key=lambda E:E.id)):
        G_ += [pack([E],root=Ft,fseed=1)]
    while G_ or C_:
        _front_ = G_+C_; _T_ = {T.i:(set(T.N_),dict(zip(T.N_,zip(T.m_,T.d_))),dict(T.r_)) for T in G_+C_}
        for G in (G_ := cluster_N(G_,_r,root__)):  # preselected for G.m
            if G.d * (G.c*wcC/(G.r*ccC)) * ((len(G.N_)-1)*wL) > avd:  # high-variance G
                C_ += [pack([N],root=Ft,fseed=1,fC=1) for N in G.N_ if N.m * (G.c*wcC/(G.r*ccC)) > ave]  # any high-m N?
        C_ = cluster_C(list(set(C_)),_r)
        T_ = G_+C_; root__ = prioritize(T_); front_ = []  # or keep separate?
        for T in T_:
            w,c = (wcC,ccC) if T.typ==2 else (wcN,ccN)  # C.typ is 2
            if T.w*T.c*w*((len(T.N_)-1)*wL) > ave*T.r*c: front_ += [T]
        if len(front_) != len(T_): root__ = prioritize(front_)  # removed proposals no longer charge survivors
        Up,Dr = 0,0  # eval convergence per front_, rather than each T?
        for T in front_:
            N_,md_,r_ = _T_.get(T.i,(set(),{},{})); md = dict(zip(T.N_,zip(T.m_,T.d_)))  # new Ts won't be in _T_
            dr = max((abs(T.r_.get(n,0)-r_.get(n,0)) for n in set(T.r_) | set(r_)),default=0)
            up = sum(abs(md[n][0]-md_.get(n,(0,0))[0]) + abs(md[n][1]-md_.get(n,(0,0))[1]) for n in T.N_)
            Dr = max(Dr,dr); Up += up
            if T.typ==2: T.fin = up*wcC <= avd*(T.r+ccC*len(T.N_)) and dr<=ave
            else:        T.fin = set(T.N_)==N_ and dr<=ave
        G_ = [T for T in front_ if T.typ==3]; C_ = [T for T in front_ if T.typ==2]
        if Up:
            G_ = [T for T in _front_ if T.typ==3]; C_ = [T for T in _front_ if T.typ==2]
        else: break
    if C_:
        C_ = cluster_P(C_,Ft)  # add eval here
        Ct = pack(C_,'Ct', Ft.root); Ft.root.Ct = Ct  # Bs may form Cs in der+, as Ns
    for n in {n for C in list(_C_)+C_ for n in C.N_}:
        n.root_ = [rt for rt in n.root_ if rt[0] not in _C_]  # replace this scope only
    for C in C_:  # not reviewed
        C.root=Ct
        for n,m,d in zip(C.N_,C.m_,C.d_): n.root_ += [[C,m,d]]
    med_= list(set(C.N_[np.argmax(C.m_)] for C in C_))
    oG_ = []
    for T in G_:
        Ft_ = [pack(N_,nF) for N_,nF in zip((T.N_,T.B_,T.C_),('Nt','Bt','Ct'))]
        for F in Ft_:
            if Ft: F.r += _r+T.olp  # persistent fork costs, not member n.r (why if Ft?)
        c_ = list(set(C for n in T.N_ for C in n.C_))  # lower-level Ct only
        G = comb_Ft(*Ft_,pack(c_,'Ct'),Ft,wTT=ttcN)
        G.r_=T.r_; G.olp=T.olp; oG_ += [G]
    if oG_:
        TT,C,R = sum_vt(oG_)
        lev = Copy_(Ft); lt = Copy_(Ft.Lt,root=lev); lev.Lt = lt # packed N_|L_, Lt for med_ xcomp
        Ft.H += [lev]; Ft.N_ = oG_; Ft.dTT,Ft.c,Ft.r = TT,C,R; Ft.m,Ft.d = val_(TT,ttcN,1)
    return oG_,med_

def cluster_N(_G_,_r,root__):  # one CN expansion round, previous-round memberships

    G_ = []
    for _G in _G_:
        G = copy(_G); G.N_=list(_G.N_); G.L_=list(_G.L_)
        if not G.fin:
            in_ = set(_G.N_); span = np.sqrt(len(in_))
            rim_ = list(set(L for n in _G.N_ for L in n.rim))
            for L in rim_:
                if len(set(L.N_) & in_) != 1: continue
                n = next(n for n in L.N_ if n not in in_)
                l_ = [l for l in n.rim if any(_n in in_ for _n in l.N_)]
                v = max(val_(sum_vt(l_)[0],ttcN),0)*wcN / (ccN*(_r+n.r))
                olp = sum(_v/v if _v>v else .5 if _v==v else 0 for i,_v in root__.get(n,[]) if i!=G.i) if v>0 else 0
                r = _G._r-1+olp; m,d = nt_vt(*L.N_)
                if m > ave*r:
                    if span > 3:
                        iM = sum(l.m for l in l_)
                        if iM/(span*decay) < ave*r: continue
                    G.N_ += [n]; G.L_ += [L]
            G.N_ = list(set(G.N_)); G.L_ = list(set(G.L_))
        G_ += [G]
    merged_ = []
    for G in G_:  # merge same-fork intersections, including stopped CNs
        while True:
            if join_:= [H for H in merged_ if set(G.N_) & set(H.N_)]:
                for H in join_:
                    G.N_ = list(set(G.N_+H.N_)); G.L_ = list(set(G.L_+H.L_)); merged_.remove(H); G.fin=0
            else: break
        merged_ += [G]
    for G in merged_:
        in_ = set(G.N_); G.B_ = []
        for L in set(L for n in G.N_ for L in n.rim):
            m,d = nt_vt(*L.N_)
            if all(n in in_ for n in L.N_):
                if m > ave*(G._r-1) and L not in G.L_: G.L_ += [L]
            elif m <= ave*(G._r-1) and d > avd*(G._r-1): G.B_ += [L]
        if not G.fin:  # refresh summaries after expansion, merging, and boundary collection
            G.dTT,G.c,r = sum_vt(G.N_+G.L_+G.B_); G._r = _r+r
            G.m,G.d = val_(G.dTT,ttcN,1); G.w = G.m
            G.m_,G.d_ = [],[]
            for n in G.N_:  # per-member internal-link support for overlap priority
                L_ = [L for L in G.L_ if n in L.N_]
                m,d = val_(sum_vt(L_)[0],ttcN,1) if L_ else (0,0)
                G.m_ += [m]; G.d_ += [d]
            G.r = G._r+G.olp  # priority refreshes overlap after both forks return
    return merged_

def cluster_C(_C_,_r):  # one CC refinement round; reuse matches to the previous centroid

    C_ = []
    for _C in _C_:
        C = Copy_(_C)
        if not C.fin:  # keep stopped Cs for overlap evaluation, without extending them
            md_ = dict(zip(_C.N_,zip(_C.m_,_C.d_))) if hasattr(_C,'w') else {}  # unrefined seed has no cached scores
            C.N_ = list(set(_C.N_+_C._N_))
            C.m_,C.d_ = map(list,zip(*(md_[n] if n in md_ else val_(base_comp(_C,n)[0],ttcC,1) for n in C.N_)))
            # compare only new members, or the initial seed; existing scores already describe _C
            in_ = set(C.N_)
            C._N_ = list(set(n for N in C.N_ for L in N.rim for n in L.N_ if n not in in_))
            c_ = np.array([n.c for n in C.N_]); C.c = c = c_.sum(); w_ = c_/c
            C._r = _r + sum(n.r*w for n,w in zip(C.N_,w_))
            w_ = c_*C.m_; w_ = w_/w_.sum() if w_.sum() else c_/c  # reform using matches to _C
            for a in ('dTT','kern','yx','span','m','d'):
                setattr(C,a,sum(getattr(n,a)*w for n,w in zip(C.N_,w_)))
            A_ = [n.angl[0] for n in C.N_ if n.angl is not None]
            C.angl = [sum(A_),np.sign(C.dTT[1] @ ttcC[1])] if A_ else None
            C.m_,C.d_ = map(list,zip(*(val_(base_comp(C,n)[0],ttcC,1) for n in C.N_)))  # score reformed C; cache for next round
            C.w = c_ @ C.m_ / c  # membership value for priority and selection, separate from centroid data
            C.r = C._r+C.olp
        C_ += [C]  # cluster_ evaluates both forks together; no further centroid reformation
    return C_

def cluster_P(_C_, root):  # multi-seed mean shift: parallel centroid refine, _C_ varies via split/merge

    iC_ = _C_; cnt = 0; Ln = len(N_:= list(set([N for C in _C_ for N in C.N_])))  # all Ns are in all Cs
    _md__ = np.zeros((Ln, len(_C_), 2))  # NxC, cols aligned to _C_
    for i, N in enumerate(N_):
        for c,m,d in N.root_:
            if c in _C_: _md__[i,_C_.index(c)] = m,d
    _dM = 0; ddM_ = []
    while True:
        for N in N_: N.root_ = []  # reset, append in sum2F
        Lc = len(_C_); L = Lc*Ln  # loop Lc,L,md__
        md__ = np.zeros((Ln,Lc,2)); O = 0
        for j,N in enumerate(N_):
            for i,C in enumerate(_C_): md__[j,i] = val_(base_comp(C,N)[0], ttcP,fd=1)
            m_ = md__[j,:,0]; O += m_.sum() - m_.max()  # cross-C ambiguity, gates split/merge
        C_ = [sum2F(N_, root, md__[:,i,0], md__[:,i,1],nF='Ct') for i in range(Lc)]  # mean shift, aligned to md__
        conv = 0
        dM = np.abs(md__[:,:,0]-_md__[:,:,0]).sum() if md__.shape==_md__.shape else 0
        if dM:
            if _dM:  # first loop doesn't have prior _dM
                ddM_ += [_dM - dM]; ddM_ = ddM_[-5:]  # summed over last 5 loops
                if sum(ddM_)<0: break  # break and preserve the last Cs before the divergence
                conv = wcP*sum(ddM_) <= ave*(cnt+root.r+ccP)
            _dM = dM
        else: conv = 1
        removed = []
        if gv_(O*wcP - ave*(root.r+ccP*L)):  # merge redundant Cs
            for i,_C in enumerate(C_):
                if _C in removed: continue
                for C in C_[i+1:]:
                    if C in removed: continue
                    l = comp_N(C,_C,(C.r+_C.r)/2, min(C.c,_C.c), A=(a:=_C.yx-C.yx), span=np.hypot(*a))
                    if l.m*wF > ave*(l.r+cF):
                        add2F(_C,C,2); removed += [C]; _C.m,_C.d = val_(_C.dTT,ttcP,fd=1)  # for next-loop base comp
                        for N in C.N_:
                            for rt in N.root_:
                                if rt[0] is C: rt[0] = _C  # update root from C to _C, for olp computation below
        _C_ = []; i_ = []  # for next loop
        for i, _C in enumerate(C_):
            if _C in removed: continue
            _C.olp = sum(rt[1] for n in _C.N_ for rt in getattr(n, 'root_', []) if rt[0] is not _C and rt[1] > _C.m)
            if _C.m * wcP > ave * (_C.r + _C.olp + ccP):  # prune
                _C_ += [_C]; i_ += [i]  # also after merge and converge
        md__ = md__[:,i_]; _md__ = md__
        if not _C_ or (conv and (not removed) and len(_C_)==Lc):  # skip if all _C_ failed the  c.m*wcP > ave*c.r*ccP eval above
            break
        cnt += 1
    out_ = []
    for N in N_: N.root_ = []  # replace with out_ Cs:
    for i, _C in enumerate(_C_):
        if _C.m > ave * _C.r:  # prune, add olp as stronger ms?
            N_,m_,d_ = [],[],[]
            for N, m,d in zip(_C.N_, _md__[:,i,0], _md__[:,i,1]):
                if m*N.c > ave*N.r: N_+=[N]; m_+=[m]; d_+=[d]
            if N_:
                C = sum2F(N_,root, m_,d_,nF='Ct')
                for N in N_:
                    L = CN(typ=1, dTT=N.dTT,c=N.c,r=N.r,m=N.m,d=N.d, span=np.hypot(*(dy_dx:=C.yx-N.yx)), angl=[dy_dx,np.sign(N.dTT[1]@ttcN[1])])
                    L.N_ = [N,C]; C.L_ += [L]
                out_ += [C]
    if out_:
        iTT, iC, iR = sum_vt(iC_); oTT,oC,oR = sum_vt(out_)
        FV_(CoF.get(), iTT-oTT, iC-oC, iR-oR)
    return out_

def sum2F(N_, root=None, m_=[],d_=[], merge=0, froot=0, nF=None):  # -> CF/CL/CN

    c_ = np.array([n.c for n in N_], dtype=float); C = c_.sum(); w_ = c_/C; N = N_[0]
    fC = any(m_)
    typ = 2 if fC else N.typ
    if fC: w_ *= np.array(m_)/sum(m_)  # differential initialization to break symmetry
    cls_ = [CF,CL,CL,CN]  # typ=2 CCs
    for i, (n,w) in enumerate(zip(N_,w_)):
        if i:
            TT += n.dTT*w; R+=n.r*w; n_ += (n.N_ if merge else [n])
            if typ:  # links and higher
                kern+=n.kern*w; span+=n.span*w; yx+=n.yx*w
                if n.angl is not None: angl = copy(n.angl[0]) if angl is None else angl+n.angl[0]
                if typ==3: box=extend_box(box,n.box)
        else:  # init
            TT = n.dTT*w; R=n.r*w; n_ = copy(n.N_ if merge else [n])
            if typ:
                kern=n.kern*w; span=n.span*w; yx=n.yx*w; angl = copy(n.angl[0]) if n.angl is not None else None
                if typ==3: box=copy(n.box)
    F = (cls_[typ])(dTT=TT, c=C, r=R, nF=nF); F.N_ = n_
    if np.any(m_):
        F.m_,F.d_ = m_,d_; C = sum(c_); F.c=C  # for CCs only
        F.m, F.d = sum(m*c for m,c in zip(m_,c_))/C, sum(d*c for d,c in zip(d_,c_))/C
    else:
        F.m, F.d = val_(TT, fd=1)   # consolidate all val_(TT) with a flag like FV_?
    F.w = sum(m * c for m, c in zip(m_, c_)) / C
    if typ==3: F.Nt.dTT = copy(TT); F.Nt.c = C; F.Nt.r = R
    if typ:
        F.kern=kern; F.span=span; F.yx=yx
        if angl is not None: F.angl = [angl, np.sign(F.dTT[1] @ ttcP[1])]
        if typ==3:  # frame?
            for N in N_: add_H(F.H, N.H, F)  # concat lower levs
            F.H += [sum2F([n for N in N_ for n in N.N_], F)]  # top lev
            F.box = box
    if typ==3: F.Nt.dTT = copy(TT); F.Nt.c = C; F.Nt.r = R; F.Nt.m,F.Nt.d = F.m,F.d
    if root is not None:
        F.wTT = root.wTT
        if nF not in ('Ct','Rt'): add2F(root,F,2)
    if froot == 1:
        for n in N_: n.root = root or F
    elif froot == 2: F.root = root
    if fC:  # add root_, m_ and d_ per N in C.N_:
        for N, m, d in zip(N_, m_, d_): N.root_ += [[F,m,d]]
    return F

def add2F(F, n, merge=0):  # unpack for batching in sum2F

    if F.c:
        C=F.c+n.c; _w,w = F.c/C, n.c/C; F.r = F.r*_w + n.r*w; F.c = C
        if isinstance(F,CoF) and isinstance(n,CoF): F.fw += n.m  # sum subtree gain, fixed cost, extensive: no weighting?
        setattr(F, 'dTT', getattr(F,'dTT')*_w + getattr(n,'dTT')*w)
    else:
        setattr(F,'dTT',getattr(n,'dTT')); F.c=n.c; F.r=n.r
        if isinstance(F,CoF) and isinstance(n,CoF): F.fw=n.m
    F.m, F.d = val_(F.dTT,fd=1)
    if merge <2:
        F.N_ += (n.N_ if merge else [n])
    if hasattr(F,'H') and getattr(n,'H',None): add_H(F.H, n.H, F)  # redundant to add_Nt?
    if hasattr(n,'C_'): F.C_ = getattr(F,'C_',[]) + n.C_  # same for L_?
    return F

def add_H(H,h, root, fN=0):
    for i, (Lev,lev) in enumerate(zip_longest(H, h)):  # bottom-up
        if lev is not None:
            if Lev:
                if lev: add2F(Lev,lev,1)  # unpack nested Hs in add2H (both are CF levs)
            else:  # Lev is list or None, lev is CF or list
                new_lev = Copy_(lev, root, cls=(CF,CN)[fN]) if lev else []
                if Lev is None: H.append(new_lev)  # pack empty list to preserve level
                else:           H[i] = new_lev  # replaces empty list with lev

def sum2G(ft_, fTT, root=None, init=1):  # finalize cluster

    if not init:
        N_,_,ntt,nc,nr = ft_[0]; N_+=root.N_; ntt+=root.Nt.dTT; nc+=root.Nt.c; nr+=root.Nt.r; ft_[0] = N_,_,ntt,nc,nr
        if len(ft_)>1: L_,_,ltt,lc,lr=ft_[1]; L_+=root.L_; ltt+=root.Lt.dTT; lc+=root.Lt.c; lr+=root.Lt.r; ft_[1]=L_,_,ltt,lc,lr
    Ft_ = []
    for ft, nF in zip_longest(ft_,('Nt','Lt','Bt')):
        if ft: n_,_,tt,c,r = ft; Ft_+= [CF(N_=n_,nF=nF,dTT=tt,m=(vt:=val_(tt,wTT,1))[0],d=vt[1],c=c,r=r)]
        else:  Ft_ += [CF()]
    C_= [c for N in Ft_[0].N_ for c in N.C_]  # splice centroids
    Ft_ += [sum2F(list(set(C_)), root.Ct if root else None ,nF='Ct') if C_ else CF(nF='Ct')]  # add multiple root_ in Cs?
    G = comb_Ft(*Ft_, root, wTT=fTT)
    N_ = G.N_; N=N_[0]; r=G.r; Av=ave+avd
    if Lt:= G.Lt:  # rng+'sub+
        lm,lc,lr = Lt.m,Lt.c,Lt.r  # no levR = 1/len(L_): represented by c
        if gv_(lm*lc*wX - Av* (lr+1+cX)):  # mdecay(L_)-decay?
            cross_comp(G, proj_L_(combinations(N_,2),G,lr), lr+1,'Nt')  # replace G Nt,Lt?
    if Bt:= G.Bt:  # der+'sub+
        bd,bc,br = Bt.d,Bt.c,Bt.r; rroot = root.root if root.root else 0
        if N.typ!=1 and gv_(bd*bc*wX - Av*(br+1+cX)):  # no ddfork, eval len?
            cross_comp(F2N(G.Bt), proj_L_(combinations([F2N(L) for L in Bt.N_],2),G,br),br, nF='Bt')
        if rroot: Bt.brrw = Bt.m * (rroot.m * (decay * (rroot.span/G.span)))  # external lend, subtract from root?
    if G.Lt or G.Bt: G.dTT,G.c,G.r = sum_vt([G.Nt,G.Lt,G.Bt]); G.m,G.d = val_(G.dTT,G.wTT,fd=1)  # recompute after deeper sub
    FV_(CoF.get(), G.dTT, G.c, G.r)
    return G

def comb_Ft(Nt, Lt, Bt, Ct, root,wTT):  # assemble core and boundary with separate link forks, no xcomp F_?

    G = CN(Nt=Nt,Lt=Lt,Bt=Bt,Ct=Ct,root=root,wTT=wTT)
    for F in Nt,Lt,Bt,Ct: F.root=G
    r = sum_vt([Nt,Nt.Lt,Bt,Bt.Lt])[2]
    if Lt:
        L_,pL_ = [],[]; [L_.append(L) if L.typ==1 else pL_.append(L) for L in Lt.N_]; angl = np.zeros(2)
        if pL_ and sum_vt(pL_,fm=1,wTT=wTT)[0]*wN > ave*(cN*np.mean([L.r for L in pL_])):
            L_ += [comp_N(*L.N_,r,L.c,1,L.angl[0],L.span) for L in pL_]
            root.Lt = sum2F(L_,nF='Lt')
        for L in L_: angl += L.angl[0]
        G.mang = np.mean([comp_A(angl,L.angl[0])[0] for L in L_])
        G.angl = [angl,np.sign(G.dTT[1] @ ttcN[1])]
    add_Nt(G)  # add2F(G, G.Nt, merge=2)? member geometry and hierarchy
    G.m,G.d = val_(G.dTT,wTT,1)
    return G

def add_Nt(G):  # in sum2G and trans_cluster

    N_ = G.N_; c_ = np.array([N.c for N in N_]); C = c_.sum()
    G.kern, G.yx = np.zeros(4),np.zeros(2); yx_ = []
    for N in N_:
        N.fin = 1; N.root = G; w = N.c/C
        G.root_ += [rt for rt in N.root_]  # Ct || Nt  (use rt for consistency, to be used in get_exemplars:  rc = sum(r[0].c for r in n.root_))
        G.kern += N.kern*w; yx = N.yx; yx_+=[yx]  # * w?
        G.box = extend_box(G.box, N.box)
        add_H(G.H, N.H, G); add_H(G.Ct.H, N.Ct.H, G.Ct)
    if (n_:=[n for N in N_ for n in N.N_]): G.H += [sum2F(n_, G)]  # new top lev  (n_ shouldn't be summed back to G here?)
    G.yx = np.mean(yx_,axis=0); G.span = (c_ @ np.hypot(*(np.array(yx_)-G.yx).T)) / C if len(N_)>1 else N_[0].span

# utilities:
def F2N(F):  # convert for cross_comp

    Nt = Copy_(F, root=F, cls=CF,typ=0); Nt.N_=F.N_; L_=F.L_  # replace in cross_comp
    F.__class__ = CN; F.Nt = Nt
    box = F.box if hasattr(F, 'box') else np.array([np.inf,np.inf,-np.inf,-np.inf])  # keep existing box
    Na_ = dict(H=copy(F.H), mang=1, box=box, exe=0, fin=0, root_=copy(F.root_), compared = set())
    if F.typ==0 and not hasattr(F, 'kern'):  # CF | PP, no overlap for Cs (only CF)
        Na_.update(kern=np.zeros(4), span=1, angl=None, yx=np.zeros(2))
    for k,v in Na_.items(): setattr(F, k, copy(v))
    [setattr(F, ft, CF(root=F)) for ft in ('Lt','Ct','Bt','Xt','Rt') if not getattr(F, ft, None)]
    if L_: F.H += [sum2F(L_, F)]
    return F

def Copy_(N, root=None, r=1, cls=None, init=0, typ=None, froot=0):
    cls = cls or type(N)
    a = dict(dTT=N.dTT*r,m=N.m,d=N.d,c=N.c,r=N.r, root=root or N.root, nF=N.nF,wTT=copy(N.wTT),typ=N.typ if typ is None else typ,
             N_=copy(N.N_), L_=copy(N.L_))
    if hasattr(N, 'w'): a['w'] = N.w
    if isinstance(N, CL): a.update(yx=copy(N.yx), kern=copy(N.kern), span=N.span, angl=[copy(N.angl[0]), N.angl[1]] if N.angl else None)
    if hasattr(N, 'root_'): a.update(root_=copy(N.root_))  # CC
    if isinstance(N,CoF): a.update(call_=copy(N.call_), typ_=copy(N.typ_) if hasattr(N, 'typ_') else [], fw=N.fw if hasattr(N, 'fw') else 0, fc=N.fc, fr=N.fr)
    C = cls(**a)
    if hasattr(N, 'i'): C.fin=N.fin; C.olp=N.olp; C.i=N.i  # for T in cluster_
    if isinstance(N, CN):
        if init:
            C.yx=[N.yx]; C.angl=[copy(N.angl[0]), N.angl[1]] if N.angl is not None else None
            C.L_=[l for l in N.rim if l.m>ave]; N.root=C; C.fin=0; C.N_=[N]
        else:
            for f in ('Nt','Bt','Ct','Xt','Rt'): setattr(C, f, Copy_(getattr(N,f), root=C))
            C.H = [Copy_(lev, root=C) for lev in N.H]
            C.angl=deepcopy(N.angl); C.yx=copy(N.yx); C.box=copy(N.box); C.mang=N.mang; C.exe=N.exe; C.root_=list(N.root_)
    if froot:
        for n in C.N_+C.L_: n.root = C  # reassign for feedback, or root_+= if multiple roots?
    return C

def extend_box(_box, box):
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return np.array((min(y0,_y0), min(x0,_x0), max(yn,_yn), max(xn,_xn)))

def sort_H(H, fi):  # lev.rc = complementary to root.rc and priority index in H, if selective and aligned

    i_ = []  # priority indices
    for i, lay in enumerate(sorted(H.node_, key=lambda lay: [lay.m,lay.d][fi], reverse=True)):
        di = lay.i - i  # lay index in H
        lay.r += di  # derR - valR
        i_ += [lay.i]
    H.i_ = i_  # H priority indices: node/m | link/d
    if fi > 1:
        H.root.node_ = H.node_
    # more advanced ordering: dH | H as medoid cluster of layers, nested cent_TT across layers?

def eval(V, weights):  # conditional progressive eval, with default ave in weights[0]
    W = 1
    for w in weights:
        W *= w
        if V < W: return 0
    return 1

def mdecay(L_):  # slope function
    L_ = sorted(L_, key=lambda l: l.span)
    dm_ = np.diff([l.m/l.c for l in L_])
    ddist_ = np.diff([l.span for l in L_])
    return - (dm_/ eps_(ddist_)).mean()  # -dm/ddist

'''
frame expansion: cross_comp lower-tile N_,C_, forward results to next lev, project feedback to scan new lower windows

add comp_prj_nt?
def comp_prj_dH(_N, N, ddH, rn, link, angl, span, dec):

    # comp proj dH to actual dH-> surprise, not used
    _cos_da = angl.dot(_N.angl) / (span * _N.span)  # .dot for scalar cos_da
    cos_da = angl.dot(N.angl) / (span * N.span)
    _rdist = span / _N.span
    rdist  = span / N.span
    prj_DH = add_H(proj_H(_N.derH, _cos_da, _rdist * dec),
                   proj_H(N.derH, cos_da, rdist * dec))  # comb proj dHs
    # add imagination: cross_comp proj derHs?
    # Et+= confirm:
    dddH = comp_H(prj_DH, ddH, rn, link)
    link.m += dddH.m; link.d += dddH.d; link.c += dddH.c; link.dTT += dddH.dTT
    add_H(ddH, dddH)    
'''
def proj_L_(pairs, root, r, max=20, fall=1):

    def proj_V(_N, N, dist, dy_dx, dec, r):  # _N x N induction
        Dec = dec or decay ** ((dist / ((_N.span + N.span) / 2)))
        iTT = (_N.dTT + N.dTT) * Dec
        eTT = (_N.Rt.dTT + N.Rt.dTT) * Dec
        C = min(_N.c, N.c); R = (_N.r + N.r) / 2
        if val_((eTT + iTT) * ttPrj) * (C / (cPrj + r + R)) * wPrj > ave:  # not oF, spec / link:
            eTT += proj_N(N, dist, dy_dx, r, N.c, dec)[0]  # pTT/ L_,B_,rim, if pV >0
            eTT += proj_N(_N, dist, -dy_dx, r, _N.c, dec)[0]  # reverse direction
        return iTT + eTT

    pL_, olp_ = [], []  # no olp_?
    for _N, N in pairs:  # -> all-to-all pre-links
        if len(_N.H) != len(N.H): continue  # or comp x agg Lev?
        if N is _N: olp_ += [N]  # overlap = unit match, no miss
        else:
            dy_dx = _N.yx - N.yx; dist = np.hypot(*dy_dx)  # rim angl is not canonic
            if dist < max:
                pTT = proj_V(_N, N, dist, dy_dx, root.m if root != 2 else decay ** (dist / ((_N.span + N.span) / 2)), r)  # based on current rim
                m, d = val_(pTT, ttN, 1)
                if fall or m > ave:
                    lc = min(_N.c, N.c); lr = r + (N.r + _N.r) / 2  # +|-match certainty
                    pL_ += [[dist, dy_dx, _N, N, lc, lr, pTT, m, d]]
    return pL_

def proj_focus(PV__, y,x, tile, elev):  # radial accum of projected focus value in PV__

    m,d = val_(tile.dTT, tile.wTT*ttFrm, 1); c = tile.c  # m,d,n = tile.m, tile.d, tile.c  # add r?
    Vm,Vd = (m-ave)*c, (d-avd)*c
    H,W = PV__.shape  # = win__
    Dec = decay ** (np.hypot(H**elev,W**elev) / (np.hypot(*(tile.box[2:]-tile.box[:2])) +eps))  # per-step decay, in units of tile span
    rim_A_ = np.array([
    (-1,-1), (-1,0), (-1,1),
    ( 0,-1),         ( 0,1),
    ( 1,-1), ( 1,0), ( 1,1)
    ], dtype=float)  # rim dirs, same order as rim_coords, n-invariant
    A = tile.angl[0]
    if np.hypot(*A):
        mA_ = np.abs(rim_A_ @ A) * tile.mang  # axial alignment, scale-free: |A| cancels in w_, signed
    else:  mA_ = np.ones(8)  # if A==(0,0), no links at all, all singleton
    rim_dist_ = np.hypot(rim_A_[:,0], rim_A_[:,1])
    n = 1  # n rim layers
    while y-n>=0 and x-n>=0 and y+n<H and x+n<W:  # rim is within frame
        pV__ = (Vm + Vd*mA_) * Dec**(n*rim_dist_)  # diag dist is 1.4 * axial
        if np.max(pV__) < ave: break  # < min adjustment
        rim_coords = np.array([
        (y-n,x-n), (y-n,x), (y-n,x+n),
        (y, x-n),           (y, x+n),
        (y+n,x-n), (y+n,x), (y+n,x+n)
        ], dtype=int)
        row,col = rim_coords[:,0], rim_coords[:,1]
        PV__[row,col] += pV__  # in-place accum pV to rim
        n += 1

def proj_N(N, dist, A,_r,_c, dec=1):  # arg rc += N.rc+Nw, recursively specify N projection val, add pN if comp_pN?

    def proj_TT(L, cos_d, dist, r, pTT, wTT, dec=1, fdec=0):  # accumulate L|N' pTT with iTT|eTT internally
        m_,d_ = L.dTT; ad_ = np.abs(d_); rdist = dist / L.span
        Dec = dist if fdec else decay ** (1+ rdist*dec)
        ddec_ = (ad_/ eps_(m_+ad_) - (1-decay)) * cos_d * rdist  # projected local loss deviation
        dm_ = m_ * (1-Dec+ddec_)  # average loss + local deviation, in match units
        TT = np.array([m_ - dm_, d_ + np.where(d_ < 0, -dm_, dm_)])  # transfer lost match to signed difference
        cert = abs(val_(TT* wTT*ttPrj) * ((L.c+wPrj)/(r+cPrj)) - ave)  # approximation
        if cert > (ave + avd) * (r + cPrj):  # certainty margin
            pTT += TT

    cos_d = (N.angl[0].dot(A) / ((np.hypot(*N.angl[0]) * dist) or eps)) * N.angl[1] if N.angl else 0  # int x ext angle alignment, mean=0
    iTT, eTT = np.zeros((2,9)),np.zeros((2,9)); c = 0
    wTT = CoF.get().wTT*ttPrj
    for L in N.Nt.L_+ N.Bt.L_:  # or Bt.L_ proj cancels Nt.L_ proj?
        proj_TT(L, cos_d, dist, L.r+_r, iTT, wTT, dec); c+=L.c  # accum iTT internally
    for L in N.rim:
        proj_TT(L, cos_d,dist,L.r+_r,eTT,wTT,dec); c+=L.c
    pTT = iTT + eTT  # proj int,ext links, work the same?
    pc = c* decay ** (1+ dist/N.span)
    FV_(CoF.get(), pTT,pc,_r)
    return pTT,pc  # info_gain = N.m * average link uncertainty, should be separate

def trace_edge(N_,_G_,_TT,_C, r,root):  # cluster contiguous shapes via PPs in edge blobs or lGs in boundary/skeleton?

    L_, cT_ = [], set()  # comp co-mediated Ns:
    for N in N_: N.fin = 0  # curently PPs only
    for N in N_:
        _N_ = [rN for B in N.B_ for rN in B.root_ if rN is not N]   # + node-mediated
        for _N in list(set(_N_)):  # share boundary or cores if lG with N, same val?
            cT = tuple(sorted((N.id,_N.id)))
            if cT in cT_: continue
            cT_.add(cT)
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx)  # Rc = r+ (N.r+_N.r)/2
            L = comp_N(_N,N, r,_C,A=dy_dx, span=dist)  # current L is dPP
            if val_(L.dTT,ttTrc,1)[1] * ((L.c+wTrc)/(r+cTrc)) > ave: L_+=[L]
    Gt_ = []
    for N in N_:  # flood-fill G per seed N
        if N.rim: N.Rt = sum2F(N.rim,root=N,nF='Rt')
        if N.fin: continue
        N.fin=1; _N_=[N]; Gt=[]; N.root=Gt
        n_,ntt,nc = [N],N.dTT.copy(),(N.c or 1); l_,ltt,lc = [],np.zeros((2,9)),0  # Gt
        while _N_:
            _N = _N_.pop(0)
            for L in _N.rim:
                if L in L_:
                    n = L.N_[0] if L.N_[1] is _N else L.N_[1]
                    if n in N_:
                        if n.root is Gt: continue
                        l_+=[L]  # default link
                        if n.fin:  # merge n root
                            _root = n.root; n_+=_root[0];l_+=_root[3]; _root[6]=1
                            for _n in _root[0]: _n.root = Gt
                        else: n.fin=1; _N_+=[n]; n_+=[n];  # add single n
                        n.root = Gt
        ntt,nc,_ = sum_vt(n_, wTT=ttTrc)
        if l_: ltt,lc,_= sum_vt(l_, wTT=ttTrc); ltt*=lc/nc;
        Gt += [n_,ntt,nc, l_,ltt,lc, 0]; Gt_+=[Gt]
    G_, TT,C,R = [],np.zeros((2,9)),0,0
    for n_,ntt,nc,l_,ltt,lc,merged in Gt_:
        if not merged:
            if gv_(val_(ntt+ltt, ttTrc) * ((nc+lc+wTrc)/(r+cTrc)) - ave):  # wrap singletons too
                TT += ntt+ltt; C += nc+lc; R += r*(nc+lc)  # add Bt?
                G_ += [sum2G([(n_,'Nt',ntt,nc,r)]+([(l_,'Lt',ltt,lc,r)] if l_ else []), ttTrc, root)]
            else:
                for N in n_: N.fin=0; N.root=root
    if val_(TT*root.wTT*ttTrc) * ((C+wTrc)/(r+1+cTrc)) * ((len(G_)-1)*wL) > ave:
        _G_+=G_; _TT+=TT; _C+=C  # concat in tile
        FV_(CoF.get(), *sum_vt(_G_))
    return _G_, _TT, _C, r+R/_C

# comp_slice hand-off:
def vect_edge(T, iY,iX,Ly,Lx, rV=1):  # T=tile, PP_ cross_comp and floodfill to init focal frame graph, no recursion:

    global ave,avd,Fw_,Fc_  # /= projected V change:
    def PP2N(PP):
        P_,L_,B_,verT,latT,A,S,box,yx, m,d,c = PP
        kern = np.array(latT[:4])
        [mM, mD, mI, mG, mA, mL], [dM, dD, dI, dG, dA, dL] = verT  # re-pack in dTT:
        dTT = np.array([ np.array([mM, mD, mL, mI, mG, mA, mL, mL / 2, 0]),  # extA=0
                         np.array([dM, dD, dL, dI, dG, dA, dL, dL / 2, 0])])
        y,x,Y,X = box; dy,dx = Y+1-y, X+1-x
        A = [np.array(A), np.sign(dTT[1] @ ttVct[1])]  # append sign
        PP = CL(typ=0, dTT=dTT,m=m,d=d,c=c,r=1, kern=kern,yx=yx,angl=A,span=np.hypot(dy/2,dx/2))  # set root in trace_edge
        m_, d_ = np.zeros(6), np.zeros(6); PP.B_ = B_; PP.box = box
        for B in B_: m_ += B.verT[0]; d_ += B.verT[1];
        ad_ = np.abs(d_); t_ = m_ + ad_  # ~ max comparand
        m = m_/eps_(t_) @ w_t[0] - ave*2; d = ad_/eps_(t_) @ w_t[1] - avd*2
        PP.Bt = CF(N_=B_, m=m, d=d, r=2, root=PP,nF='Bt')
        for P in P_: P.root = PP
        if hasattr(P,'nt'):  # typ=1?
            PP.root_ = []  # Gd.root_: cores, no centroids? multiple PPms may share same PPd?
            for dP in P_:
                for P in dP.nt: PP.root_ += [P.root]  # PPm
        return PP
    blob_ = T.N_; G_,TT,C,R = [],np.zeros((2,9)),0,0
    for blob in blob_:
        if not blob.sign:
            if gv_(blob.G * wVct - ave * cVct):  # proxy for comp_slice and slice_edge
                edge = slice_edge(blob, rV); L = len(edge.P_)-1
                if gv_(edge.G * (wVct*L) - sum([P.latT[4] for P in edge.P_]) * (cVct*L)):
                    PPm_ = comp_slice(edge, rV, ttVct)  # add comp_slice's weights?
                    N_ = [PP2N(PPm) for PPm in PPm_]
                    for PPd in edge.link_: PP2N(PPd)  # we don't form Gds?
                    for N in N_:
                        if N.B_:
                            PPd_ = [B.root for B in N.B_]; sum2F(PPd_,N.Bt)
                            N.Bt.N_ = PPd_; [setattr(B,'root',N.Bt) for B in PPd_]
                    tt,c,r = sum_vt(N_); C += c
                    if gv_(val_(tt*ttVct) * ((c+wVct)/ (3+cVct)) * ((len(PPm_)-1)*wL) - ave):
                        G_,TT,c,R = trace_edge([F2N(N) for N in N_], G_,TT,c,3,T); C += c  # flatten B_-mediated Gs
    if G_:
        FV_(CoF.get(), TT, C,1)
        return sum2G([[G_, 'Nt', TT, C, 1]], ttVct, T)  # c,R?

def frame_H(image, iY,iX, Y,X, rV, elev=1, max_elev=4, ffb=0):

    def fill_frame(_iy,_ix, elev, T):  # expand level_frame from pixel-level seed tile, similar to frame_blobs

        frame = np.full((Ly,Lx),None, dtype=object)  # higer tile Ly,Lx = lower tile Ly,Lx **2
        cy,cx = int(_iy/Ly), int(_ix/Lx)  # seed is the center of frame
        PV__ = np.zeros([Ly,Lx])  # projected value map
        frame[cy,cx] = T; T_=[]; __T_=[(T,cy,cx)]  # output, eval wave
        while __T_:
            _T_ = []  # next wave
            for _T, y,x in __T_:  # last wave
                if gv_(val_(_T.dTT*_T.wTT*ttFrm) * ((_T.c+wFrm)/(_T.r+cFrm)) - ave):
                    T_ += [_T]; dy,dx = _T.box[2:] -_T.box[:2]
                    pTT, pc = proj_N(_T, np.hypot(dy,dx), np.array([dy,dx]), elev,_T.c)  # no proj r?
                    if gv_(val_(pTT*T.wTT*ttFrm) * ((pc+wFrm)/(_T.r+elev+cFrm)) - ave):  # +ve, no uncertainty projection yet
                        proj_focus(PV__,y,x,_T,elev)  # -> projected value map
                        for _y,_x in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):  # fill 4 adjacent cells
                            if not (0<=_y<Ly and 0<=_x<Lx) or frame[_y,_x] is not None: continue  # outside frame or checked
                            if gv_(PV__[_y,_x] - ave):  # accumulated from all adjacent tiles
                                iy = _y**elev; ix = _x**elev
                                if T := frame_H(image, iy,ix, Y,X, rV, max_elev=elev+1):  # agg+: incr to the current level
                                    _T_ += [(T,_y,_x)]; frame[_y,_x] = T
                                else: frame[_y,_x] = 0
            __T_ = _T_
        if T_:
            TT,C,R = sum_vt(T_, wTT=ttFrm); R += elev
            if val_(TT*ttFrm) * ((C+wFrm)/(R+cFrm)) > ave:
                return T_,C,R
        return [],0,0

    Fr, aH, oH = [],[],[]; aTT=oTT=np.zeros((2,9)); global ave,avd  # regime refs across levs / ffeedback
    T = vect_edge( frame_blobs_root( comp_pixel( image[iY:iY+Ly**elev, iX:iX+Lx**elev]), rV), iY,iX,Ly,Lx, rV)  # base process
    while T and elev < max_elev:
        tile_,C,R = fill_frame(iY, iX, elev, T)  # seed tile -> sparse higher scope tile( oH( aH
        if tile_:
            N_ = [g for t in tile_ for g in t.N_]; m,_,tt,c,r = sum_vt(N_,fm=1)  # concat edge Gs
            Fr = sum2G([(N_,'Nt',tt,c,r)],ttFrm)  # use sum2G to get angl and l_, for the next loop's T (T = Fr)
            Fr.H += [sum2F(tile_)]  # minimally processed level
            if gv_(m * c * wX - ave * (r+1+cX)):
                cross_comp(Fr, proj_L_(combinations(N_,2),Fr, r),r,'Nt')  # agg+
                if elev and ffb:  # ffb=1 in main, no ffeedback in side tiles
                    Fr,aTT,oTT,aH,oH = ffeedback(Fr, aTT,oTT,aH,oH)  # term,form oH ( aH
                    elev += 1; T=Fr  # next-extension seed
                else: break
            else: break
        else: break
    if Fr: FV_(CoF.get(), Fr.dTT, Fr.c, Fr.r)
    return Fr  # intra-lev feedback

def ffeedback(frame, aTT,oTT, aL,oL):  # recompute filters from regime drift; fork: reform oF_ on cross-regime drift

    global ave, avd, oF_, nF_, iF_
    for oF in oF_:
        if oF.V_: oF.w += sum(oF.V_) + sum(oF.gV_)  # all extensive
    dTT = dc = dr = 0
    _ac,_ar = (aL.c,aL.r) if aL else (0,0); _oc,_or = (oL.c,oL.r) if oL else (0,0)
    # H init @ 1st term:
    if aL := pack_seg(frame,'aH',wBac, cBac, aTT):  # L: new level
        dTT = aL.dTT-aTT; aTT=aL.dTT; dc= aL.c-_ac; dr= aL.r-_ar
        ave, avd = val_(aTT, fd=1)  # filters *= ave
        if oL := pack_seg(frame,'oH', wBac, cBac**2, oTT):
            dTT += oL.dTT- oTT; oTT=oL.dTT; dc+=oL.c-_oc; dr+=oL.r-_or
            for _oF in copy(oF_): _oF.body = [split_oF(t,_oF) for t in _oF.body]
            oF_ = clust_oF_()
            map_ = {}
            for T in oF_:
                if T.N_:  # new clustered T
                    T.caller_ = set().union(*[F.caller_ for F in T.N_]); T.c = sum(F.c for F in T.N_)
                    for F in T.N_:
                        if nF_[F.nF] is not None: map_[nF_[F.nF].name] = T.fdef.name  # map existing oFs to clustered oF
            # update nF_ and iF_:
            nF_ = [oF.fdef for oF in oF_]
            iF_.clear(); iF_.update({fd.name: i for i,fd in enumerate(nF_)})
            for oF in oF_:
                for n in call_sites(oF.fdef):  # callees may be replaced
                    if n.func.id in map_: n.func.id = map_[n.func.id]
            inject_oF_(oF_, globals())

    FV_(CoF.get(),dTT,dc,dr)
    return frame, aTT, oTT, aL, oL

def pack_seg(frame, nF, w, c, _dTT):  # drift-gated regime termination for aH and oH

    H = frame.H
    n = ('aH','oH') if nF=='aH' else ('oH',)
    i = next((j+1 for j in reversed(range(len(H))) if H[j].nF in n), 0)  # packed levs
    if tail := H[i:] if nF=='aH' else [l for l in H[i:] if l.nF=='aH']:  # lev_| aH_
        D = sum(np.sum(np.abs(t.dTT-_dTT) * wTT) for t in tail)  # drift
        if (vD := D*w - ave*c) > 0:  # update value
            seg = CN(nF=nF, root=frame); seg.H=tail; seg.dTT = tail[-1].dTT; seg.c = sum(l.c for l in tail)  # default regime summary
            if vD > ave: seg.dTT,seg.c,seg.r = sum_vt(tail); seg.m,seg.d = val_(seg.dTT,fd=1)  # deep summary
            frame.H = H[:i]+[seg]  # append
            return seg

if __name__ == "__main__":  # './images/toucan_small.jpg' './images/raccoon_eye.jpeg', add larger global image
    trace_func(vars()); g = vars()
    parse_funcs(["agg_recursion.py"])   # populate nF_
    for oF in oF_: oF.fdef = nF_[oF.nF]  # update fdef with ast node
    inject_oF_(oF_,g)
    Y, X = imread('./images/toucan.jpg').shape
    frame = frame_H(image=imread('./images/toucan.jpg'), iY=Y//2 -31, iX=X//2 -31, Y=Y, X=X, rV=1, ffb=1)
    # search frames ( tiles inside image, at this size it should be 4K, or 256K panorama, won't actually work on toucan