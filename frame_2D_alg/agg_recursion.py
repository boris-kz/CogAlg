import numpy as np, inspect, contextvars
from copy import copy, deepcopy
from math import atan2, cos, pi
from functools import reduce
from itertools import zip_longest, combinations, product  # from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, imread, comp_pixel, CBase
from slice_edge import slice_edge
from comp_slice import comp_slice, w_t
from meta_code import oF_,iF_,nF_,CF,CL,CN,CoF,wT,wTT, eps,eps_,ave,avd,decay, trace_func,parse_funcs, call_sites, split_oF,clust_oF_,inject_oF_,gv_
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

- select sparse exemplars to seed the clusters, top k for parallelization (exemplar_), no clustering?
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
ttFrm,ttX,ttTrc,ttN,ttF, ttE,ttcN,ttcC,ttcP,ttsG, ttBac,ttPrj,ttVct = [F.wTT for F in oF_]  # attr weight tuples added per function

def val_(TT, wTT=wTT):  # multi-variate rel match for membership, rel diff for boundary

    m_,d_ = TT; ad_ = np.abs(d_)
    return (m_/ eps_(m_+ad_)) @ wTT[0]  # brrw from different-base m: Lt vs Bt, for directional projection

def sum_vt(N_, fr=0, fm=0, wTT=wTT, fdiv=1):  # basic weighted sum of CN|CF list

    C = sum(n.c for n in N_); M=R=0; TT = np.zeros((2,9))
    for n in N_:
        w = n.c/C; TT += (n.rTT if fr else n.dTT) * w; R += n.r * w
        if fm:
            m = val_(TT, wTT)
            M += (m/ ave*R) if fdiv else (m- ave*R)  # in 0:inf or -1:1?
    return (M,TT,C,R) if fm else (TT,C,R)  # sum dTT as comparands, not value terms?

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
def cross_comp(root, pL_, _r, dF=None, fall=1, iN_=None):  # recursion root, iN_: recluster/rng+

    L_,N_,G_ = [],[],[]
    for dist, dy_dx, _N,N, lc,lr, pTT,m,nexp in pL_:
        if _N != N and (fall or m>0):
            if gv_(m*(lc*wN/(lr*cN)) - ave* (_r+cN)):
                Link = comp_N(_N,N, lr,lc, full=not dF, A=dy_dx, span=dist, rL=root)
                Link.rTT = np.abs(pTT-Link.dTT) / eps_(Link.dTT)  # prediction error
                L_+=[Link]; N_+=[_N,N]; Link.nexp=nexp
            elif not dF:  # pack as prelink
                _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
                pL = CL(typ=-1, N_=[_N,N], dTT=pTT,m=m,c=lc,r=lr,span=dist,box=box,nexp=nexp,angl=[dy_dx,1], yx=np.add(_N.yx,N.yx)/2)
                L_+= [pL]; N.rim+=[pL]; _N.rim += [pL]; N_+=pL.N_

    def Rng_(n_):  # w: combined membership weight / rdn: 1 + stronger members per C:
        for n in n_: n.w = sum(m * C.c / (1 + sum(_m>m for _m in C.m_)) for C,m in n.root_)
        return [n for n in n_ if (n.m+n.w) * (n.c*wX / (n.r+cX)) > ave]
    if L_:
        Lt = sum2F(L_,root=None if (dF or iN_) else root, nF='Lt'); tt,m,c,r = Lt.dTT,Lt.m,Lt.c,Lt.r
        if dF: add2F(dF,Lt, merge=1); return  # comp_F: no agg+, dF out
        for N in (N_:=list(set(N_))): sum2F(N.rim, N, nF='Rt')  #-> N.Rt
        if iN_: add2F(root.Lt, Lt, merge=1)
        else:   iN_ = N_
        if gv_(m * (c*wcN/(r*ccN)) * ((len(L_)-1)*wL) - ave):
            G_ = cluster_N(root, exemplar_(iN_ or N_,r+1), r+1,c) or []  # tentative G_|C_|P_ for rng+
        FV_(CoF.get(), tt,c,r)  # current rng xcomp
        if G_:
            if (rng_:= Rng_(iN_)) and (pL_:= proj_L_(combinations(rng_,2), root,_r,nexp=nexp+1)):  # rng+ xcomp:
                if gv_(sum(n.w for n in rng_) * (wX/cX) - ave) and (_G_:= cross_comp(root, pL_,_r, iN_=iN_)):
                    return _G_  # -> G_
            G_ = [sum2G(g, ttcN, root, _r) for g in G_]  # finalize full-rng Gs
            Nt = sum2F(G_, root, nF='Nt'); nm,nc,nr = Nt.m,Nt.c,Nt.r
            if gv_(nm * (nc*wX/(nr*cX)) - ave):  # agg+
                G_ = cross_comp(root, proj_L_(combinations(G_,2),root,_r+1),_r+1) or G_
    return G_

def sum2G(F_, wTT, root=None, _r=0):  # finalize CN from C or (N_,L_,B_), eval trans_cluster, der+, sub+

    def trans_cluster(G):
        for L in G.L_:
            for lev in L.H:  # tFt/Nt | tN_/Bt, not Lt|Ct
                if lev:
                    for tF in lev.N_ if not isinstance(lev.N_[0],CL) else [lev]:  # CF or CL: Nt|Bt
                        for tL in tF.N_:
                            if tL.m * (tL.c*wcN / (tL.r*ccN)) > ave:  # merge trans_link.N_.roots:
                                rt0 = getattr(tL.N_[0].root,'root',None); rt1 = getattr(tL.N_[1].root,'root',None)
                                if rt0 and rt1 and rt0 != rt1:
                                    if rt1.H: add_H(rt0.H, rt1.H, rt0, fN=1)
                                    add2F(rt0,rt1, merge=1,fH=0)  # recompute Nt attrs / G
                else:  # 0 top lev: no deeper N_ xcomp
                    break
    G = CN(root=root,wTT=wTT)
    L_, pL_ = [],[]
    if isinstance(F_,tuple):
        N_,_L_,B_ = F_  # from trace_edge, vect_edge, frame_H
    else:  # C from CN,CC: extract internal _L_ and perimeter B_ from N_ rims
        N_ = F_.N_; n_set = set(N_); _L_,B_ = [],[]
        for L in {l for n in N_ for l in n.rim}:
            if (L.N_[0] in n_set) and (L.N_[1] in n_set):
                if L.m > 0: _L_ += [L]
            elif L.nexp == 1 and -L.m > ave * (_r-1):  # -L.m, as in nt_vt?
                B_ += [L]
    for L in _L_:
        if L.typ==1: L_ += [L]
        else:       pL_ += [L]
    pm = sum_vt(pL_,fm=1,wTT=wTT)[0]; pc = sum([L.c for L in pL_]); pr = np.mean([L.r for L in pL_])
    if pL_ and (pm * (wN*pc / (cN*pr)) > ave):
        for pL in pL_:
            L_ += [L:= comp_N(*pL.N_,pL.r,pL.c,1,pL.angl[0],pL.span)]; L.nexp=pL.nexp
            for N in pL.N_: N.rim.remove(pL)
    for N in N_:
        if N.rim: N.Rt = sum2F(N.rim,root=N,nF='Rt')
    [sum2F(f_,G,nF=nF) for f_,nF in zip((N_,B_,L_),('Nt','Bt','Lt')) if f_]
    G.m = val_(G.dTT,wTT); G.r = G.Nt.r
    if G.L_: trans_cluster(G)  # splice trans_links, merge L.N_.roots
    if Bt:= G.Bt:  # der+
        bm,bc,br = Bt.m,Bt.c,Bt.r+_r+1
        if N_[0].typ!=1 and gv_(bm * (bc*wX / (br*cX)) - ave):
            cross_comp(F2N(G.Bt), proj_L_(combinations([F2N(L) for L in Bt.N_],2),G,br),br)
        if RR:= root.root if root else None: Bt.brrw = Bt.m* (RR.m* (decay* (RR.span/G.span)))
    if Lt:= G.Lt:  # sub+
        m,c,r = Lt.m,Lt.c, Lt.r+_r
        if gv_(m* (c*wcN / (r*ccN)) * ((len(N_)-1)*wL) - ave):  # higher filter: r+1 -> sub_Gs, CC
            if (g_:= cluster_N(G, exemplar_(N_,r), r+1,c)):
                sum2F([sum2G(g, ttcN, G, r+1) for g in g_], G,nF='Nt')  # G._Nt-> G.Nt.H[-1]
    if G.Lt or G.Bt:
        G.m,G.dTT,G.c,G.r = sum_vt([f for f in (G.Nt,G.Lt,G.Bt) if f], fm=1,wTT=wTT)
    FV_(CoF.get(), G.dTT, G.c, G.r)
    return G

def comp_N(_N,N, r,c, full=1, A=None,span=None, rL=None):

    def sum_dn_(dn_, L):
        L.H += [lev:= sum2F(dn_,nF='tF',root=L)]  # dN_ maybe dFt, works the same
        TT,C,R = lev.dTT,lev.c,lev.r
        w = C/L.c * (L.r/R)
        L.dTT = (L.dTT+ TT*w) / (1+w); L.m = val_(L.dTT,ttN); L.r = (L.r+R*w)/(1+w); L.c+=C
    def comp_H(_N, N, L):
        dH,TT,C,R = [],np.zeros((2,9)),0,0
        for _lev,lev in zip(_N.H,N.H):  # current dTT already compared
            if not (_lev and lev): continue
            tt = comp_derT(_lev.dTT[1],lev.dTT[1])
            lc = min(_lev.c,lev.c); lr = (_lev.r+lev.r)/2
            C += lc; R += lr*lc; TT += tt*lc; m = val_(tt,ttN)
            dH += [CF(dTT=tt,m=m,c=lc,r=lr,root=L)]
        if not C: return
        TT/=C; R/=C
        w = C/L.c * (L.r/R)
        L.dTT = (L.dTT+ TT*w) / (1+w); L.m= val_(L.dTT,ttN)
        L.r = (L.r+R*w) / (1+w); L.c += C
        L.H += dH  # add to 1st lev from comp N.N_

    L = CL(N_=[_N,N], c=c,r=r,root=rL)
    if full:
        dTT = base_comp(_N,N)[0]  # use rn?
        if span is None: span = np.hypot(*_N.yx - N.yx)
        yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx
        box = np.array([min(_y,y),min(_x,x), max(_y,y),max(_x,x)])
        angl = [np.zeros(2) if A is None else A, np.sign(dTT[1] @ ttN[1])]
        L.yx=yx; L.box=box; L.span=span; L.angl=angl; L.kern=(_N.kern+N.kern)/2
    else: dTT = comp_derT(_N.dTT[1],N.dTT[1])
    m = val_(dTT,ttN); L.dTT,L.m = dTT,m
    if N.typ:  # converted Ls, skip PPs, Fts?
        dn_=[]
        if N.typ==3 and gv_(m* (c*wN /(r*cN)) - ave):  # CN
            for i,(_Ft,Ft) in enumerate(zip((_N.Nt,_N.Lt,_N.Bt),(N.Nt,N.Lt,N.Bt))):
                if _Ft and Ft: dn_ += [comp_F(_Ft,Ft,r,L)]; r+=(i or 1)-1
        elif gv_(m* (c*wF /(r*cF)) - ave):  # L or B
            for _n,n in product(_N.N_,N.N_):
                if _n is not n: dn_ += [comp_N(_n,n, r,min(_n.c,n.c),rL=L,full=0)]  # CC: medoid Ns may overlap?
        if dn_: sum_dn_(dn_,L)  # tFt if CN else dN_
        else:
            L.H += [[]]  # placeholder for skipped diff(N.N_)
            if _N.H and N.H: comp_H(_N,N,L)
    if full:
        for n,_n in (_N,N),(N,_N): n.rim += [L]
    FV_(CoF.get(),L.dTT,L.c,L.r)
    return L

def comp_F(_F, F, ir=0, rL=None):

    ddTT = comp_derT(_F.dTT[1],F.dTT[1])
    m = val_(ddTT,ttF);  c= min(_F.c,F.c); r = (_F.r+F.r)/2 + ir
    dF = CF(dTT=ddTT, m=m,r=r,c=c, nF=F.nF)
    if (_N_:=_F.N_) and (N_:=F.N_):
        Np_ = list(product(_N_,N_))  # pairs; Lt,Bt: links, Ct: centroids
        tt = (rL.dTT*rL.c + ddTT*dF.c) / (rL.c+dF.c)
        if gv_(val_(tt,ttX) * (c*wX / (r*cX)) - ave):  # any sub-comp?
            if F.nF!='Nt': [F2N(n) for n in set(_N_+N_) if not isinstance(n,CN)]  # Rt,Nt,Bt
            cross_comp(rL, proj_L_(Np_,2,r), r,dF)
    FV_(CoF.get(), dF.dTT, dF.c, dF.r)
    return dF  # no cross-fork N_, no L ext updates?

def base_comp(_N,N):  # comp Et, kern, extT, dTT

    M, D = N.dTT.sum(axis=1);  C= N.c;  I, G, Dy, Dx = N.kern;  L = len(N.N_)  # 0-der params? or N.m, xsign sum d_?
    _M,_D=_N.dTT.sum(axis=1); _C=_N.c; _I,_G,_Dy,_Dx =_N.kern; _L = len(_N.N_)  # N_: density in span
    rn = C/_C; mA=dA=0
    _pars = np.array([_M*rn,_D*rn,_C,_I*rn,_G*rn, [_Dy,_Dx],_L*rn,_N.span], dtype=object)  # Et, kern, extT
    pars  = np.array([M,D,C, (I,wI),G, [Dy,Dx], L,(N.span,wS)], dtype=object)
    if hasattr(_N,'angl') and _N.angl and N.angl:  # mostly Ls, some Gs
        mA,dA = comp_A(_N.angl[0]*_N.angl[1], N.angl[0]*N.angl[1])
    m_,d_ = comp(_pars,pars, mA,dA)  # M,D,n, I,G,a, L,S,A
    dm_,dd_ = comp_derT(_N.dTT[1], N.dTT[1])  # higher-derivation than M,D

    return np.array([m_+dm_,d_+dd_]), rn  # rn is not relevant, rm, rv?
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
        else:  # massive, but this is a wrong default for most ders
            _a,a = abs(_p), abs(p)
            m_ += [min(_a,a) if (_p<0)==(p<0) else -min(_a,a)]  # + complement max(_a,a) for +ves?
            d_ += [_p - p]
    ''' for mass in 0:1:
        m = dir_m * mass + inv_m * (1-mass), | mass* max(abs(_p),abs(p)) + (1-mass)* avd - ad
    '''
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

def exemplar_(N_,_r):  # multi-layer non-maximum suppression -> sparse clustering seeds = rim-medoids

    for n in N_: n.w = n.Rt.m * n.c  # rep value
    N_= sorted(N_, key=lambda n: n.w, reverse=True); E_,Inh_ = [],set()
    for rdn, N in enumerate(N_, start=1):  # strong-first
        inh_ = list(Inh_ & set(N.rim))  # stronger Es in N.rim
        oM = sum_vt(inh_,fm=1, wTT=ttE)[0] if inh_ else 0
        oV = oM / (N.Rt.m or eps)  # relative olp V, lateral only
        if N.w * wE > ave* (_r+rdn+cE+oV):
            E_+=[N]; N.exe = 1  # point cloud of focal nodes
            Inh_.update(set(N.rim))  # rim-based: Ct olp would be other instances of the same class
        elif N.w * wE <= ave* (_r+rdn+cE):
            break  # rest is weaker w/o olp, trace via rims
    if E_: FV_(CoF.get(), *sum_vt(E_))
    else:  E_ = [N_[0]]; N_[0].exe=1  # no gain, no inhibition, any N can be seed
    return E_

def nt_vt(n,_n):
    M, D = 0,0  # exclusive match, contrast
    for l in set(n.rim+_n.rim):
        if (m:=l.m) > 0: M += m
        elif -m > ave:   D += m  # contrast: the opposite side of marginal predictability?
    return M, D

def cluster_N(root, _N_, _r,_c):  # flood-fill node | link clusters, flat, replace iL_ with E_?

    Gt_,in_,fin_ = [],set(),[]  # root attrs, add prelink pL_,pN_? fin_ should be tile-wide?
    for N in _N_:  # form G per remaining N
        if N in fin_ or (root.root and not N.exe): continue  # no exemplars in Fg
        N_ = [N]; L_,B_ = [],[]; fin_ += [N]  # init G
        __L_= N.rim  # spliced rim
        while __L_:
            _L_ = []
            for L in set(__L_) - in_:  # flood-fill via frontier links
                _N = L.N_[0] if L.N_[1] in fin_ else L.N_[1]; in_.add(L)
                if _N not in fin_:
                    m,d = nt_vt(*L.N_)
                    if m > ave *_r:  # additional filter, cluster nt, L,C_ by combined rim density, add gv_?
                        span = np.sqrt(len(N_))  # approx span
                        if span > 3:  # refine by rim connectivity / norm span
                            iM = sum([L.m for L in _N.rim if (L.N_[0] if L.N_[1] is _N else L.N_[1]) in N_])
                            if iM / (span*decay) < ave * (_r-1): continue  # normalized N-to-N_ match, sum for sum2G?
                        N_ += [_N]; L_ += [L]; fin_ += [_N]
                        _L_+= [l for l in _N.rim if l not in in_ and ((l.N_[0] in fin_) ^ (l.N_[1] in fin_))]   # new frontier links, +|-?
                    elif L.nexp==1 and d > ave *_r:
                        B_ += [L]  # contrast value
            __L_ = list(set(_L_))
        if N_:
            ft_ = []
            for i,(F_,nF) in enumerate(zip((N_,L_,B_),('Nt','Lt','Bt'))):  # no Ct yet, formed below
                F_ = list(set(F_)) or []; tt,fc,fr = sum_vt(F_,wTT=ttcN) if F_ else (np.zeros((2,9)),0,0)
                ft_+= [[tt,fc,fr]]
            (nt,nc,nr),(lt,lc,lr),(bt,bc,br) = ft_
            c = nc+lc+bc; r = (nr*nc +lr*lc +br*bc) /c;  tt = (nt*nc +lt*lc +bt*bc) /c  # br includes overlap?
            if gv_(val_(tt*root.wTT*ttcN) * (c*wcN/(r*ccN)) * ((len(N_)-1)*wL) - ave):
                C = sum2F(N_,root,nF='C',typ=1)  # initial connectivity G
                Gt_ += [(C, tt,c,r)]
            else:
                for n in N_: fin_.remove(n)
                for l in L_: in_.remove(l)
    if Gt_:
        C_,C,tt_,c_,r_ = [],0,[],[],[]; _r+=1
        for CC,tt,c,r in Gt_: C_+=[CC]; C+=c; tt_+=[tt]; c_+=[c]; r_+=[r]
        TT = np.zeros((2,9)); R = 0
        for tt,c,r in zip(tt_,c_,r_): w=c/C; TT+=tt*w; R+=r*w
        M = val_(TT*root.wTT*ttcN)
        FV_(CoF.get(), *sum_vt(C_)[:-1], R)
        if gv_(M * (C*wcC /((_r+R)*ccC)) * ((len(C_)-1)*wL) - ave):
            return cluster_C(root, C_, _r+R) or C_

def cluster_C(root, _C_, _r):  # refine C seeds and expand through their members' rims

    N_, tC_ = {n for C in _C_ for n in C.N_}, []
    for n in N_: n._root_ = n.root_; n.root_ = []  # preserve initial cluster_N memberships in _root_
    while True:  # reform C_, update and test C rims
        C_, cnt,rdn,Up,lc = [],0,0,0,0
        for _C in _C_:
            _N_ = set(_C.N_) | {_n for n in _C.N_ for L in n.rim for _n in L.N_}  # current members + rim frontier
            n_, m_, M, T, R, up, c = [], [], 0, 0, 0, 0, 0
            for n in _N_:
                if n not in N_: N_.add(n); n.root_, n._root_ = [], []
                dtt, _ = base_comp(_C, n)
                m = val_(dtt, ttcC); c += n.c
                _m = next(((rt[1]) for rt in n._root_ if rt[0] is _C), 0)  # prior membership, 0 if new
                if m * n.c > ave * n.r:  # += new nodes and candidate nodes
                    up += abs(_m - m); n_ += [n]; m_ += [m]; T += n.c; M += m * n.c; R += n.r * n.c
                else: up += _m  # exclusion
            if not n_: continue
            r = _r + R / T
            if M * _C.m * (wcC * c / (r * _C.r * ccC)) * (((L := len(n_)) - 1) * wL) > ave:
                C = sum2F(n_, root, m_, nF='C', typ=2)  # -> n.root_
                if up * (c*wcC / (r*ccC)) * ((L-1)*wL) > avd:
                    C_ += [C]; Up += up; lc += c  # active Cs' change and estimated next-loop work
                else: tC_ += [C]  # converged if low membership change
                cnt += T; rdn += R  # process costs
        r = _r + rdn / (cnt or eps)
        oC_ = tC_ + C_
        if C_ and Up * (lc * wcC / (r * ccC)) > avd:
            uN_ = {n for C in oC_ for n in C.N_}
            posL = sum(rt[1] > ave for n in uN_ for rt in n.root_)
            totL = len(uN_) * len(oC_)
            if gv_(posL / (totL or eps) * (lc * wcP / (r * ccP)) * ((len(oC_) - 1) * wL) - ave):
                oC_ = cluster_P(oC_, root); break
            else:
                for n in N_: n._root_ = n.root_; n.root_ = [rt for rt in n.root_ if rt[0] in tC_]
                _C_ = C_  # for next loop
        else: break
    if oC_:
        FV_(CoF.get(), *sum_vt(oC_))
        return oC_

def cluster_P(_C_, root):  # multi-seed mean shift: parallel centroid refine, _C_ varies via split/merge

    iC_ = _C_; cnt = 0; Ln = len(N_:= list(set([N for C in _C_ for N in C.N_])))  # all Ns are in all Cs
    _m__ = np.zeros((Ln, len(_C_)))  # NxC, cols aligned to _C_
    for i, N in enumerate(N_):
        for c,m in N.root_:
            if c in _C_: _m__[i,_C_.index(c)] = m
    while True:
        for N in N_: N.root_ = []  # reset, append in sum2F
        Lc = len(_C_); totL = Lc*Ln  # loop Lc,L,m__
        m__ = np.zeros((Ln,Lc))
        for j,N in enumerate(N_):
            for i,C in enumerate(_C_): m__[j,i] = val_(base_comp(C,N)[0], ttcP)
        C_ = [sum2F(N_, root, m__[:,i],nF='C',typ=2) for i in range(Lc)]  # mean shift, aligned to m__, add typ?
        removed = []
        posL = (m__ > ave).sum()
        if gv_(posL*wcP - ave*(root.r+ccP*totL)):  # merge redundant Cs
            for i,_C in enumerate(C_):
                if _C in removed: continue
                for C in C_[i+1:]:
                    if C in removed: continue
                    m = val_(base_comp(C,_C)[0],ttcP); r = (C.r+_C.r)
                    if m*wF > ave*(r+cF):
                        add2F(_C,C,2); removed += [C]; _C.m = val_(_C.dTT,ttcP)  # for next-loop base comp
                        for N in C.N_:
                            for rt in N.root_:
                                if rt[0] is C: rt[0] = _C  # update root from C to _C, for olp computation below
        cnt += 1
        dM = (m__-_m__).mean() if m__.shape==_m__.shape else 0  # convergence, may be negative
        if C_ and dM * (wcP / (cnt+root.r+ccP)) > ave:  # next loop, tentative
            _C_ = []; i_ = []
            for i, _C in enumerate(C_):
                if _C in removed: continue
                _C.olp = sum(rt[1] for n in _C.N_ for rt in n.root_ if rt[0] is not _C and rt[1] > _C.m)
                if _C.m * wcP > ave * (_C.r + _C.olp + ccP):  # prune
                    _C_ += [_C]; i_ += [i]  # also after merge and converge
            _m__ = m__[:,i_]
        else:
            if dM<0: C_ = _C_  # revert to better prior
            else: i_ = [i for i,C in enumerate(C_) if C not in removed]; _m__ = m__[:,i_]; C_ = [C_[i] for i in i_]
            break
    out_ = []
    for N in N_: N.root_ = []  # replace with out_ Cs:
    for i, C in enumerate(C_):
        if C.m > ave * C.r:  # prune, add olp as stronger ms?
            N_,m_ = [],[]
            for N, m in zip(C.N_, _m__[:,i]):
                if m*N.c > ave*N.r: N_+=[N]; m_+=[m]
            if N_:
                oC = sum2F(N_,root, m_,nF='C', typ=2)
                out_ += [oC]
    if out_:
        iTT, iC, iR = sum_vt(iC_); oTT,oC,oR = sum_vt(out_)
        FV_(CoF.get(), iTT-oTT, iC-oC, iR-oR)
    return out_

def sum2F(N_, root=None, m_=None, nF=None, typ=0):  # c|c*m -weighted sum -> CF|CL, bound to root.nF unless fuzzy C

    N = N_[0]; c_ = np.array([n.c for n in N_], dtype=float); C = c_.sum(); fC = nF=='C'
    w_ = c_ * np.array(m_) if fC else c_.copy()
    w_ /= w_.sum()
    TT = np.tensordot(w_, [n.dTT for n in N_], 1); R = w_ @ [n.r for n in N_]
    F = (CL if (geo:= hasattr(N,'yx')) else CF)(N_=list(N_), dTT=TT,c=C,r=R, nF=nF, root=root,typ=typ)
    if geo:  # links and higher
        F.yx,F.kern = [w_ @ np.array([getattr(n,a) for n in N_]) for a in ('yx','kern')]; F.span = w_ @ [n.span for n in N_]
        F.box = reduce(extend_box, [n.box for n in N_ if hasattr(n,'box')], np.array([np.inf,np.inf,-np.inf,-np.inf]))
    if fC:  # not revised
        F.m = c_@m_ /C; F.w = F.m; F.m_ = m_; F.n_ = N_  # pass separate F.N_ as arg, F.m += sum(m_)?
        for n,m in zip(N_,m_): n.root_ += [[F,m]]
    else: F.m = val_(TT)
    if root is None: return F
    F.wTT = root.wTT
    if nF in ('Nt','Lt','Bt'): root.dTT+=TT; root.c+=C  # root r is not affected?
    if nF=='Lt':  # oriented, as in base_comp
        root.angl = F.angl = (A:= [np.sum(A_:= [L.angl[0] * L.angl[1] for L in F.N_], axis=0), 1])  # Lt.angl for proj_V
        root.mang = F.mang = np.mean([comp_A(A[0], a)[0] for a in A_])
    elif nF=='Nt':
        root.yx,root.kern,root.box = F.yx,F.kern,F.box; root.span = np.hypot(*(F.box[2:]-F.box[:2]))/2  # extent, as in PP2N
        if root.Nt: F.H = copy(root.H) + [Copy_(root.Nt, root=root)]
        else:       [add_H(F.H, n.H, F) for n in N_]
    if not fC and nF and hasattr(root,nF): setattr(root,nF,F)  # Nt|Lt|Bt|Rt|C, not L.tF
    return F

def add2F(F,f, merge=0, fH=1):  # unpack for batching in sum2F

    if F.c:
        C=F.c+f.c; _w,w = F.c/C, f.c/C; F.r = F.r*_w + f.r*w; F.c = C
        if isinstance(F,CoF) and isinstance(f,CoF): F.fw += f.m  # sum subtree gain, fixed cost, extensive: no weighting?
        setattr(F, 'dTT', getattr(F,'dTT')*_w + getattr(f,'dTT')*w)
    else:
        setattr(F,'dTT',getattr(f,'dTT')); F.c=f.c; F.r=f.r
        if isinstance(F,CoF) and isinstance(f,CoF): F.fw=f.m
    F.m = val_(F.dTT)
    if merge <2:
        F.N_ += (f.N_ if merge else [f])
    if F.nF=='Lt' and (R:= F.root) is not None:  # rng+ merge into root.Lt, skip dF: no root, full=0 links have no angl
        R.dTT += f.dTT; R.c += f.c  # complete root binding, as in sum2F
        R.angl = F.angl = [A:= np.sum(A_:= [L.angl[0] * L.angl[1] for L in F.N_], axis=0), 1]
        R.mang = F.mang = np.mean([comp_A(A, a)[0] for a in A_])
    if fH and hasattr(F,'H') and getattr(f,'H',None): add_H(F.H, f.H, F)  # redundant to add_Nt?
    if hasattr(f,'C_'): F.C_ = getattr(F,'C_',[]) + f.C_  # same for L_?
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

# utilities:
def F2N(F):  # convert for cross_comp

    Nt = Copy_(F, root=F, cls=CF,typ=0); Nt.N_=F.N_; L_=F.L_; Nt.H = F.H  # replace in cross_comp
    F.__class__ = CN; F.Nt = Nt
    box = F.box if hasattr(F, 'box') else np.array([np.inf,np.inf,-np.inf,-np.inf])  # keep existing box
    Na_ = dict(H=copy(F.H), mang=1, box=box, exe=0, root_=copy(F.root_), compared = set())
    if F.typ==0 and not hasattr(F, 'kern'):  # CF | PP, no overlap for Cs (only CF)
        Na_.update(kern=np.zeros(4), span=1, angl=None, yx=np.zeros(2))
    for k,v in Na_.items(): setattr(F, k, copy(v))
    [setattr(F, ft, CF(root=F)) for ft in ('Lt','Bt','Xt','Rt') if not getattr(F, ft, None)]
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
    if isinstance(N, CN):
        if init:
            C.yx=[N.yx]; C.angl=[copy(N.angl[0]), N.angl[1]] if N.angl is not None else None
            C.L_=[l for l in N.rim if l.m>ave]; N.root=C; C.N_=[N]
        else:
            for f in ('Nt','Lt','Bt','Xt','Rt'):
                setattr(C, f, Copy_(getattr(N,f), root=C))
                if f=='Nt': getattr(C,f).H = [Copy_(lev, root=C) for lev in N.H]
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
def proj_L_(pairs, root, r, max=20, fall=1, nexp=1):

    def proj_V(_N, N, dist, dy_dx, dec, r):  # _N x N induction
        Dec = dec or decay ** ((dist / ((_N.span + N.span) / 2)))
        iTT = (_N.dTT + N.dTT) * Dec
        eTT = (_N.Rt.dTT + N.Rt.dTT) * Dec
        C = min(_N.c,N.c); R = (_N.r+N.r)/2
        if val_((eTT + iTT) * ttPrj) * (C / (cPrj + r + R)) * wPrj > ave:  # not oF, spec / link:
            eTT += proj_N(N, dist, dy_dx, r, N.c, dec)[0]  # pTT/ L_,B_,rim, if pV >0
            eTT += proj_N(_N,dist,-dy_dx, r,_N.c, dec)[0]  # reverse direction
        return iTT + eTT
    pL_, olp_ = [], []  # no olp_?
    for _N, N in pairs:  #-> all-to-all pre-links
        if len(_N.H) != len(N.H): continue  # or comp x agg Lev?
        if N is _N: olp_ += [N]  # overlap = unit match, no miss
        elif not (set(N.rim) & set(_N.rim)):  # not nexp-1 link
            dy_dx = _N.yx - N.yx; dist = np.hypot(*dy_dx)  # rim angl is not canonic
            if dist < max * nexp:
                pTT = proj_V(_N, N, dist, dy_dx, decay** (dist/ ((_N.span+N.span)/2)) if root==2 else root.m, r)  # based on current rim
                m = val_(pTT, ttN)
                if fall or m > ave:
                    lc = min(_N.c, N.c); lr = r + (N.r + _N.r) / 2  # +|-match certainty
                    pL_ += [[dist, dy_dx, _N, N, lc, lr, pTT, m, nexp]]
    return pL_

def proj_focus(PV__, y,x, tile, elev):  # radial accum of projected focus value in PV__

    m = val_(tile.dTT, tile.wTT*ttFrm); c = tile.c  # m,d,n = tile.m, tile.d, tile.c  # add r?
    Vm = (m-ave)*c
    H,W = PV__.shape  # = win__
    Dec = decay ** (np.hypot(H**elev,W**elev) / (np.hypot(*(tile.box[2:]-tile.box[:2])) +eps))  # per-step decay, in units of tile span
    rim_A_ = np.array([
    (-1,-1), (-1,0), (-1,1),
    ( 0,-1),         ( 0,1),
    ( 1,-1), ( 1,0), ( 1,1)
    ], dtype=float)  # rim dirs, same order as rim_coords, n-invariant
    A = tile.angl[0]
    if np.hypot(*A): mA_ = np.abs(rim_A_ @ A) * tile.mang  # axial alignment, scale-free: |A| cancels in w_, signed
    else:            mA_ = np.ones(8)  # if A==(0,0), no links at all, all singleton
    rim_dist_ = np.hypot(rim_A_[:,0], rim_A_[:,1])
    n = 1  # n rim layers
    while y-n>=0 and x-n>=0 and y+n<H and x+n<W:  # rim is within frame
        dec = Dec ** (n * rim_dist_)  # diag dist is 1.4 * axial
        pD = sum(tile.dTT[1] * mA_)  # D contribution = signed mag?
        pV__ = Vm*dec + pD*dec
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
    for L in N.L_+ N.B_:  # neg B_ proj cancels Nt.L_ proj?
        proj_TT(L, cos_d, dist, L.r+_r, iTT, wTT, dec); c+=L.c  # accum iTT internally
    for L in N.rim:
        proj_TT(L, cos_d,dist,L.r+_r,eTT,wTT,dec); c+=L.c
    pTT = iTT + eTT  # proj int,ext links, work the same?
    pc = c* decay ** (1+ dist/N.span)
    FV_(CoF.get(), pTT,pc,_r)
    return pTT,pc  # info_gain = N.m * average link uncertainty, should be separate

def trace_edge(N_,_G_,_TT,_C, r,root):  # cluster contiguous shapes via PPs in edge blobs or lGs in boundary/skeleton?

    L_,cT_ = [], set()  # comp co-mediated Ns:
    for N in N_:  # PPs
        B_= []; _N_= [rN for B in N.B_ for rN in B.root_ if rN is not N]  # + node-mediated
        for _N in list(set(_N_)):  # share boundary or cores if lG with N, same val?
            cT = tuple(sorted((N.id,_N.id)))
            if cT in cT_: continue
            cT_.add(cT)
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx)  # Rc = r+ (N.r+_N.r)/2
            L = comp_N(_N,N, r,min(N.c,_N.c),A=dy_dx, span=dist)  # dPP, in N.rim
            m = np.array(val_(L.dTT,ttTrc)) * ((L.c+wTrc)/(r+cTrc))
            if m > 0:      L_ += [L]
            elif -m > ave: B_ += [L]  # contrast: the opposite side of marginal predictability?
        N.B_ = B_
    Gt_ = []; fin_ = []
    for N in N_:  # flood-fill G per seed N
        if N.rim: N.Rt = sum2F(N.rim,root=N,nF='Rt')
        for B in N.B_: B.root_ = []
        if N in fin_: continue
        fin_ += [N]; _N_=[N]; Gt=[]; N.root=Gt
        n_,ntt,nc = [N],N.dTT.copy(),(N.c or 1); l_,ltt,lc = [],np.zeros((2,9)),0; b_,btt,bc = copy(N.B_),np.zeros((2,9)),0  # Gt
        while _N_:
            _N = _N_.pop(0)
            for L in _N.rim:
                if L in L_:
                    n = L.N_[0] if L.N_[1] is _N else L.N_[1]
                    if n in N_:
                        if n.root is Gt: continue
                        l_+=[L]  # default link
                        if n in fin_:  # merge n root
                            _root = n.root; n_+=_root[0];l_+=_root[3]; b_+=_root[6]; _root[-1]=1
                            for _n in _root[0]: _n.root = Gt
                        else: _N_+=[n]; n_+=[n]; b_+=n.B_; fin_ += [n]  # add single n
                        n.root = Gt
        ntt,nc,_ = sum_vt(n_, wTT=ttTrc)
        if l_: ltt,lc,_= sum_vt(l_, wTT=ttTrc); ltt*=lc/nc
        if b_: btt,bc,_= sum_vt(b_, wTT=ttTrc); btt*=bc/nc  # needs a review
        Gt += [n_,ntt,nc, l_,ltt,lc, b_,btt,bc, 0]; Gt_+=[Gt]
    G_, TT,C,R = [],np.zeros((2,9)),0,0
    for n_,ntt,nc,l_,ltt,lc,b_,btt,bc,merged in Gt_:
        if not merged:
            if gv_(val_(ntt+ltt+btt, ttTrc) * ((nc+lc+bc+wTrc)/(r+cTrc)) - ave):  # wrap singletons
                TT += ntt+ltt+btt; C += nc+lc+bc; R += r*(nc+lc+bc)  # add Bt from neg Ls per PP?
                G_ += [sum2G((n_,l_,b_), ttTrc, root, r)]  # add sub-PPs from sub+?
            else:
                for N in n_: N.root=root
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
        PP = CL(typ=0, dTT=dTT,m=m,c=c,r=1, kern=kern,yx=yx,angl=A,span=np.hypot(dy/2,dx/2))  # set root in trace_edge
        m_, d_ = np.zeros(6), np.zeros(6); PP.B_ = B_; PP.box = box
        for B in B_: m_ += B.verT[0]; d_ += B.verT[1];
        ad_ = np.abs(d_); t_ = m_ + ad_  # ~ max comparand
        m = m_/eps_(t_) @ w_t[0] - ave*2  # d = ad_/eps_(t_) @ w_t[1] - avd*2
        PP.Bt = CF(N_=B_, m=m, r=2, root=PP,nF='Bt')
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
                            PPd_ = list({B.root for B in N.B_}); N.Bt = sum2F(PPd_, nF='Bt')
                            N.Bt.N_ = PPd_; [setattr(B,'root',N.Bt) for B in PPd_]
                    tt,c,r = sum_vt(N_); C += c
                    if gv_(val_(tt*ttVct) * ((c+wVct)/ (3+cVct)) * ((len(PPm_)-1)*wL) - ave):
                        G_,TT,c,R = trace_edge([F2N(N) for N in N_], G_,TT,c,3,T); C += c  # flatten B_-mediated Gs
    if G_:
        FV_(CoF.get(), TT, C,1)
        return sum2G((G_,[],[]), ttVct, T,R)  # return sub-PPs?

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
            N_ = [g for t in tile_ for g in t.N_]; m,_,c,r = sum_vt(N_,fm=1)  # concat edge Gs
            Fr = sum2G((N_,[],[]),ttFrm)  # use sum2G to get angl and l_, for the next loop's T (T = Fr)
            Fr.H += [sum2F(tile_)]  # minimally processed level
            if gv_(m * c * wX - ave * (r+1+cX)):
                cross_comp(Fr, proj_L_(combinations(N_,2),Fr, r),r)  # agg+
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
        ave = val_(aTT)  # filters *= ave
        if oL := pack_seg(frame,'oH', wBac, cBac**2, oTT):
            dTT += oL.dTT- oTT; oTT=oL.dTT; dc+=oL.c-_oc; dr+=oL.r-_or
            for _oF in copy(oF_): _oF.body = [split_oF(t,_oF) for t in _oF.body]
            oF_ = clust_oF_()
            map_ = {}
            for T in oF_:
                if T.N_:  # new clustered T
                    T.caller_ = set().union(*[F.caller_ for F in T.N_]); T.c = sum(F.c for F in T.N_)
                    for F in T.N_:
                        if nF_[F.nF] is not None: map_[nF_[F.nF].name] = T.fdef.name  # map oFs to oF clusters
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
            if vD > ave: seg.dTT,seg.c,seg.r = sum_vt(tail); seg.m = val_(seg.dTT)  # deep summary
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