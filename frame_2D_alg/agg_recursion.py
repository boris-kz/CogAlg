import numpy as np, inspect, contextvars
from copy import copy, deepcopy
from math import atan2, cos, pi
from functools import reduce
from itertools import zip_longest, combinations, product  # from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, imread, comp_pixel, CBase
from slice_edge import slice_edge
from comp_slice import comp_slice, w_t
from meta_code import oF_,iF_,nF_,CF,CN,CoF,wT,wTT, eps,eps_,ave,avd,decay, trace_func,parse_funcs, call_sites, split_oF,clust_oF_,inject_oF_,gv_
'''
This is a main module of open-ended clustering algorithm, designed to discover empirical patterns/clusters of indefinite complexity. 
Lower modules cross-comp and cluster image pixels and blob slices(Ps), forming PPs: segments of matching Ps.
These PPs are atomic inputs here.
Cycles of (generative cross-comp, compressive clustering, filter-adjusting feedback) should form hierarchical model of input stream: 

Cross-comp forms Miss and Match (min: shared_quantity for directly predictive params, else inverse deviation of miss or variation), in 2 forks:
rng+: incremental-range cross-comp nodes: edge segments at < max distance, cluster if they match. 
der+: incremental-derivation cross-comp links, from node cross-comp, if abs_diff * relative_adjacent_match

Clustering compressively groups the elements into compositional hierarchy, initially by pair-wise similarity or density thereof.
High-contrast links are correlation clustered to form a boundary per connectivity cluster of the nodes.
Each level may extend clustering through 4 increasingly fuzzy stages, each seeded by prior-stage output:

- select sparse exemplars to seed the clusters, top k for parallelization (exemplar_), no clustering?
- CN: connectivity-based agglomerative ( divisive clustering (cluster_N), with boundary link clustering inside Bt
- CC: CN-seeded centroid-based marginally fuzzy extended clustering (cluster_C), selects medoids for incremental-distance xcomp
- CP: CC-seeded centroid-parallel FCM-like refine, if >ave global overlap (cluster_P)

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
capital vars are summed small-case vars
'''
wM, wi, wI,wG,wa, wL,wS,wA = wT; Ly=Lx= 64  # fractal tile dims
cFrm,cX,cTrc,cN,cF, cE,ccN,ccC,ccP,csG, cBac,cPrj, cVct = (  # function complexity
wFrm,wX,wTrc,wN,wF, wE,wcN,wcC,wcP,wsG, wBac,wPrj, wVct ) = [F.fc for F in oF_]  # ave gain/call, init=cost
ttFrm,ttX,ttTrc,ttN,ttF, ttE,ttcN,ttcC,ttcP,ttsG, ttBac,ttPrj,ttVct = [F.wTT for F in oF_]  # attr weight tuples added per function, [0] only?
Frmw_,Xw_,Trcw_,Nw_,Fw_,Ew_,cNw_,cCw_,cPw_,sGw_,Bacw_,Prjw_,Vctw_ = ttFrm[0],ttX[0],ttTrc[0],ttN[0],ttF[0],ttE[0],ttcN[0],ttcC[0],ttcP[0],ttsG[0],ttBac[0],ttPrj[0],ttVct[0]
# compute mass__:
p___,m___,gc__ = [],[],[]  # |par|, G.m-par: [der][clus][par], gc__: [der][clus]
d___,cc__ = [],[]  # |dpar|: [der][comp][par], cc__: [der][comp]
mass__,avad__,acov__,_span = [],[],[],0  # [der][par], acov__: evidence for mass__ over _span

def mT___(N, _j):  # per comparand: |params| per dLev vs its G.m without that param's column
    if not hasattr(N,'mt_'): return  # no Lt | Bt: no estimate
    I, G = N.kern[:2]
    j_it_c_ = [(_j, np.abs([0,N.c,I,G,0,len(N.N_),N.span,0]), N.c), (_j+1, np.abs(N.dTT[1]), N.c)]  # as in base_comp
    j_it_c_+= [(_j+1+d, np.abs(l.dTT[1]), l.c) for lev in N.H if lev for d,l in enumerate(lev.H) if l]  # as in comp_H
    for j, it, c in j_it_c_:
        add_fH(j+1); p___[j] += [it]; m___[j] += [N.mt_]; gc__[j] += [c]

def val_(TT, wT=wT): return TT[0] @ wT / wT.sum()  # multi-variate rel match for membership, diff for boundary, in -1:1

def sum_vt(N_, fr=0, fm=0, wTT=wTT):  # c-weighted mean dTT,r of CN|CF list, + m if fm

    C = sum(n.c for n in N_); R = 0; TT = np.zeros((2,8))
    for n in N_:
        w = n.c/C; TT += (n.rTT if fr else n.dTT) * w; R += n.r * w
    if fm: return val_(TT,wTT[0]), TT,C,R  # m of the mean: TT[0] is linear in m now
    return TT,C,R

def FV_(F, tt,c,r):  # combine m,d per oF
    tF = oF_[F.nF]; tF.c += c
    V = val_(tt,tF.wTT[0]) * (c/r); tF.V_ += [V]; return V

def cent_TT(dTT, r):  # EM-like weight attr matches | diffs by their match to the sum, recompute to convergence

    wTT,_wTT = [],np.ones((2,8)); coT = np.abs(dTT[0])+np.abs(dTT[1])  # complemented vals
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
def cross_comp(root, pL_, _r, dF=None, fall=1, _j=0):  # recursion root, forms C_: Cs ( Gs ( N_

    L_,N_,G_,Dt = [],[],[],[]
    for pL in pL_:
        dist, dy_dx, _N,N, lc,lr, pTT,m = pL if isinstance(pL,list) else (pL.span,pL.angl[0],*pL.nt,pL.c,pL.r,pL.dTT,pL.m)
        if _N != N and (fall or m>0):  # CN in frame_H,spec, else CC?
            if gv_(m*(lc*wN/(lr*cN)) - ave):
                Link = comp_N(_N,N, lr,lc, full=not dF, A=dy_dx, span=dist, rL=root, _j=_j)
                Link.rTT = np.abs(pTT-Link.dTT) / eps_(Link.dTT)  # prediction error
                L_+=[Link]; N_+=[_N,N]
            elif not dF:  # += prelink
                _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
                pL = CN(typ=-1, nt=[_N,N], dTT=pTT,m=m,c=lc,r=lr,span=dist,box=box,angl=[dy_dx,1],yx=np.add(_N.yx,N.yx)/2)
                L_+= [pL]; N.rim+= [pL]; _N.rim+= [pL]; N_+= pL.nt
    if L_:
        Lt = sum2F(L_,root=None if dF else root, nF='Lt'); tt,m,c,r = Lt.dTT,Lt.m,Lt.c,Lt.r
        if dF:  # comp_F: adds to dF vals, no agg+
            add2F(dF,Lt, merge=1); return
        if dL_:= [l for l in L_ if l.typ==1 and -l.m>ave]:  # combined int contrast
            Dt = sum2F(dL_, root=root, nF='Dt'); Dt.r += _r + 1
            if gv_(-Dt.m * (Dt.c*wX /(Dt.r*cX)) * ((len(dL_)-1)*wL) - ave):  # tentative wL
                cross_comp(F2N(Dt), combinations(dL_,2), Dt.r,_j=_j)  # der+
            if Dt.Lt: add2F(root.Lt if root.nF=='Dt' else root, Dt.Lt, merge=2,fH=0)  # Dt.Lt += dLs
        for N in (N_:= list(set(N_))): mT___(N,_j); sum2F(N.rim, N, nF='Rt')  # N.Rt
        FV_(CoF.get(),tt,c,r)  # level
        if gv_(m * (c*wcN /(r*ccN)) * ((len(L_)-1)*wL) - ave):  # add cm,cr?
            G_ = cluster_N(root, exemplar_(N_,r+1), r+1,c,_j)  #-> sum2G,cluster_C -> combined-m Gs
            if G_ and gv_(root.m * (root.c*wX /(root.r*cX)) * ((len(G_)-1)*wL) - ave):  # root Nt = Ct?
                G_ = cross_comp(root, proj_L_(combinations(G_,2),root,_r,_j=_j),_r,_j=_j) or G_  # agg+
    return G_

def comp_N(_N,N, r,c, full=1, A=None,span=None, rL=None, _j=0, nF='Nt'):  # _j: index in dH if spec

    L = CN(typ=1, nt=[_N,N], c=c,r=r,root=rL)
    if full:
        dTT,ddTT = base_comp(_N,N,_j)  # N.dTT is combined from top-level forks
        if span is None: span = np.hypot(*_N.yx-N.yx)
        yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx
        box = np.array([min(_y,y),min(_x,x), max(_y,y),max(_x,x)])
        angl = [np.zeros(2) if A is None else A, np.sign(dTT[1] @ ttN[1])]
        L.yx=yx; L.box=box; L.span=span; L.angl=angl; L.kern=(_N.kern+N.kern)/2; L.dTT = dTT+ddTT  # for comp_F
    else:
        L.dTT = ddTT = comp_dT(_N.dTT[1], N.dTT[1], c, _j+1)  # 1st der only, no M,D,ext,kern
    m = val_(ddTT,Nw_)  # (avad-ad)/(avad+ad) = (rd-1)/(rd+1), in -1:1
    if full: m += val_(dTT,Nw_)
    # 2 dO/ composition, same c
    if N.typ in (1,2) and gv_(m* (c*wF /(r*cF)) - ave):  # N|L
        for _Ft,Ft,nF in zip((_N.Nt,_N.Lt,_N.Bt,_N.Dt), (N.Nt,N.Lt,N.Bt,N.Dt), ('Nt','Lt','Bt','Dt')):
            if _Ft and Ft:  # assign dFt as L.Ft
                comp_F(_Ft,Ft, nF, r,L,_j)
    elif N.typ in (1,2,3) and gv_(m* (c*wN /(r*cN)) - ave):  # C_ or Ft.N_ from comp_F, forms trans-Ls
        dN_ = []
        for _n,n in product(_N.N_,N.N_):  # spec
            if _n is not n:  # if CC: medoid Ns may overlap?
                dN_+= [comp_N(_n,n, r,min(_n.c,n.c), rL=L,full=0,_j=_j)]
        if dN_: sum2F(dN_,L, nF=nF)  # per fork
    if dF_ := [F for F in (L.Nt,L.Lt,L.Bt,L.Dt) if F]:  # fork-wise | trans 1st der replaces whole-N ddTT
        ddTT,C,R = sum_vt(dF_); L.r = (L.r*L.c + R*C) / (L.c+C); L.c+=C
    if full: ddTT += dTT
    L.dTT = ddTT; L.m = val_(ddTT,Nw_)
    if full:
        for n,_n in (_N,N),(N,_N): n.rim += [L]
    FV_(CoF.get(),L.dTT,L.c,L.r)
    return L

def comp_F(_F, F, nF, ir=0, rL=None, j=None):

    c = min(_F.c,F.c)
    ddTT = comp_dT(_F.dTT[1],F.dTT[1],c, j+1)  # 1st der, same dLev as base_comp ddTT
    m = val_(ddTT,Fw_); r = (_F.r+F.r)/2 + ir
    dF = CF(dTT=ddTT, m=m,r=r,c=c, nF=nF, root=rL)
    if nF in ('Nt','Dt'):  # spec draft
        if (_N_:=_F.N_) and (N_:=F.N_):  # always true?
            tt = (rL.dTT*rL.c + ddTT*c) / (rL.c+c)  # vertical comb?
            if gv_(val_(tt,Xw_) * ((rL.c+c)*wX /(r*cX)) - ave):
                cross_comp(rL, proj_L_(list(product(_N_,N_)),2,r,_j=j), r, dF,_j=j)  # tL-> dF.N_
            elif _F.H and F.H:  # dF.H[-1]+[dLt]?
                comp_H(_F,F, dF,j)  # rL: lev in L.H, no cross-fork N_, no L ext updates?
    if nF=='Dt' and gv_(m * (c*wF /(r*cF)) - ave):
        ddF_ = []  # der+
        for nf in ('Lt','Dt'):  # current ders
            if (_f:=getattr(_F,nf,None)) and (f:=getattr(F,nf,None)):
                comp_F(_f,f, nf, r,dF, j+1); ddF_ += [getattr(dF,nf)]  # -> dF.Lt | dF.Dt
        if ddF_: dF.dTT,dF.c,dF.r = sum_vt([dF]+ddF_); dF.m = val_(dF.dTT,Fw_)  # higher ders
    setattr(rL,nF,dF)
    FV_(CoF.get(), dF.dTT, dF.c, dF.r)

def comp_H(_N, N, L,_j):
    dH = []
    for _lev, lev in zip(reversed(_N.H), reversed(N.H)):  # top-down in composition
        if _lev and lev:  # no break
            ddH = []
            for j, (_l, l) in enumerate(zip(_lev.H, lev.H)):  # bottom-up in derivation
                if _l and l:
                    lc = min(_l.c,l.c); lr = (_l.r+l.r)/2
                    ltt = comp_dT(_l.dTT[1], l.dTT[1], lc, _j+j+1)  # as in base_comp
                    ddH += [CF(dTT=ltt, m=val_(ltt,Nw_), c=lc, r=lr, root=L)]  # bottom-up
            if ddH:
                tt, c, r = sum_vt(ddH)
                dH += [CF(H=ddH, dTT=tt, m=val_(tt,Nw_), c=c, r=r, root=L)]  # lev.H: ders, lev.N_: dN_?
    # revert to bottom-up, on completion: L.dTT,L.c,L.r = sum_vt(L.H); L.m = val_(L.dTT,ttN)?
    L.H += dH[::-1]

def base_comp(_N,N, j=0, fder=1):  # j: root depth, comp Et, kern, extT, dTT; rational M?

    M = N.dTT[0].sum();  C= N.c;  I, G, Dy, Dx = N.kern;  L = len(N.N_)  # M,D, 0-der params? or N.m, xsign sum d_: directional or abs?
    _M=_N.dTT[0].sum(); _C=_N.c; _I,_G,_Dy,_Dx =_N.kern; _L = len(_N.N_)  # N_: density in span
    rn = C/_C; mA=dA=0
    _pars = np.array([_M*rn,_C, _I*rn,_G*rn, [_Dy,_Dx],_L*rn,_N.span], dtype=object)  # Et, kern, extT
    pars  = np.array([M,C, I,G,[Dy,Dx], L,N.span], dtype=object)
    if hasattr(_N,'angl') and _N.angl and N.angl:  # mostly Ls, some Gs
        mA,dA = comp_A(_N.angl[0]*_N.angl[1], N.angl[0]*N.angl[1])
    # dT: M,n, I,G,a, L,S,A:
    m_,d_ = comp_(_pars,pars, mA,dA, c:= min(_N.c,N.c), j, fder)
    dm_,dd_ = comp_dT(_N.dTT[1],N.dTT[1], c,j+1, fder)  # ddTT, same c / comb-F dTT?
    return np.array([m_,d_]), np.array([dm_,dd_])
'''
    if np.hypot(*_N.angl[0])*_N.mang + np.hypot(*N.angl[0])*N.mang > ave*wA:  # aligned L_'As, mang *= (len_H)+fi+1
    mang = (rn*_N.mang + N.mang) / (1+rn)  # ave, weight each side by rn
    align = 1 - mang* (1-mA)  # in 0:1, weigh mA 
'''
def comp_(_pars, pars, meA,deA, c=1, j=0, fder=1):  # m is min for massive core attrs, or combined with inv_m?

    m_,d_ = [],[]
    for i, (_p,p, mass,avad) in enumerate(zip(_pars,pars, mass__[j],avad__[j])):  # prior-frame mean |d| = massless m
        if isinstance(_p, list):  # vector angle
            mA, dA = comp_A(_p, p); m_+= [2*mA-1]; d_+= [dA]  # cos: 1 aligned, 0 orthogonal, -1 opposite
        else:
            d_ += [d := (_p - p)]
            rm = (avad-abs(d)) / (avad+abs(d))  # massless m
            if mass > 0:  # cov(mag,stability)
                m = min(abs(_p),abs(p)); M = max(abs(_p),abs(p))
                if (_p<0) != (p<0): m =- m  # gap in p vals
                rm = (m/M) * mass + rm * (1-mass)
            m_ += [rm]
    if fder: global d___, cc__; d___[j] += [np.abs(d_+[deA])]; cc__[j] += [c]  # ini dH=[lev0], mA mass=0
    return np.array(m_+[meA]), np.array(d_+[deA])

def comp_dT(_i_,i_, c, j, fder=1):  # scalars, j: dLev index, >1 in comp_H only

    add_fH(j+1); mass_,avad_ = mass__[j], avad__[j]  # der+ / comp_F, comp_H
    d_ = _i_ - i_; ad_ = np.abs(d_); _a_,a_ = np.abs(_i_), np.abs(i_)
    rm_ = (avad_-ad_) / eps_(avad_+ad_)  # massless m
    m_ = np.minimum(_a_,a_); m_[(_i_<0) != (i_<0)] *= -1  # gap in p vals
    M_ = np.maximum(_a_,a_)
    w_ = mass_ * (mass_ >0)  # massive share: cov(mag,stability) if +ve
    rm_ = (m_/eps_(M_)) * w_ + rm_ * (1-w_)
    if fder: global d___, cc__; d___[j] += [ad_]; cc__[j] += [c]
    return np.array([rm_,d_])

def comp_A(_A,A):
    dA = atan2(*_A)- atan2(*A)  # * direction
    if   dA > pi: dA -= 2 * pi  # rotate CW
    elif dA <-pi: dA += 2 * pi  # rotate CCW
    '''  or 
    den = np.hypot(*_A) * np.hypot(*A) + eps
    mA = (_A @ A / den +1) / 2  # cos_da in 0:1, no rot = 1 if dy * _dx - dx * _dy >= 0 else -1  # +1 CW, −1 CCW, ((1-cos_da)/2) * rot? '''
    return (cos(dA)+1) / 2, dA/pi  # mA in 0:1, dA in -1:1, or invert dA, may be negative?

def exemplar_(N_,_r):  # multi-layer non-maximum suppression -> sparse clustering seeds = rim-medoids

    for n in N_: n.w = n.Rt.m * n.c; n.exe=0  # rep value
    N_= sorted(N_, key=lambda n: n.w, reverse=True); E_,Inh_ = [],set()
    for rdn, N in enumerate(N_, start=1):  # strong-first
        inh_ = list(Inh_ & set(N.rim))  # stronger Es in N.rim
        oM,_,oC,_ = sum_vt(inh_,fm=1, wTT=ttE) if inh_ else (0,0,0,0)
        oV = oM*oC / (N.Rt.m*N.Rt.c or eps)  # relative olp V, lateral only
        if N.w * wE > ave* (_r+rdn+cE+oV):
            E_+=[N]; N.exe = 1  # point cloud of focal nodes
            Inh_.update(set(N.rim))  # rim-based
        elif N.w * wE <= ave* (_r+rdn+cE):
            break  # rest is weaker w/o olp, trace via rims
    if E_: FV_(CoF.get(), *sum_vt(E_))
    else:  E_ = [N_[0]]; N_[0].exe=1  # no gain, no inhibition, any N can be seed
    return E_

def nt_vt(n,_n):
    pM = nM = 0  # exclusive match | contrast
    for l in set(n.rim+_n.rim):
        if (m:= l.m) > ave: pM += m
        elif ave - m > ave: nM += m  # contrast: the opposite side of marginal predictability?
    return pM,nM
def rim2LB(N_):  # internal match links, perimeter, use in cluster_N, Ft2G?
    n_set = set(N_); L_,B_ = [],[]
    for L in {l for n in N_ for l in n.rim}:
        if (L.nt[0] in n_set) and (L.nt[1] in n_set):
            if L.m > ave: L_ += [L]
        else: B_ += [L]
    return L_,B_

def cluster_N(root, _N_, _r,_c,_j, fsub=0):  # flood-fill -> seeds with L_,B_ -> cluster_C -> sum2G

    Gt_,in__,fin_,G_ = [],set(),[],[]  # root attrs, add prelink pL_,pN_? fin_ should be tile-wide?
    for N in _N_:  # form seed per remaining N
        if N in fin_ or (root.root and not N.exe): continue  # no exemplars in Fg
        N_ = [N]; fin_ += [N]  # init G
        in_ = set(); __L_= N.rim  # spliced rim
        while __L_:
            _L_ = []
            for L in set(__L_) - (in__|in_):  # flood-fill via frontier links
                _N = L.nt[0] if L.nt[1] in fin_ else L.nt[1]; in_.add(L)
                if _N not in fin_:
                    pm,nm = nt_vt(*L.nt)  # +,- m
                    if pm > ave *_r:  # additional filter, cluster nt, L,C_ by combined rim density, add gv_?
                        span = np.sqrt(len(N_))  # approx span
                        if span > 3:  # refine by rim connectivity / norm span
                            iM = sum([L.m for L in _N.rim if (L.nt[0] if L.nt[1] is _N else L.nt[1]) in N_])
                            if iM / (span*decay) < ave * (_r-1): continue  # normalized N-to-N_ match, sum for sum2G?
                        N_ += [_N]; fin_ += [_N]
                        _L_+= [l for l in _N.rim if l not in (in_|in__) and ((l.nt[0] in fin_) ^ (l.nt[1] in fin_))]  # new frontier links, +|-?
            __L_ = list(set(_L_))
        if N_:
            n_set = set(N_); L_,B_ = [],[]  # internal match links, perimeter = CC frontier
            for L in {l for n in N_ for l in n.rim}:
                if (L.nt[0] in n_set) and (L.nt[1] in n_set):
                    if L.m > ave: L_+= [L]
                elif fsub: B_ += [L]  # in sub+ only
            ft_ = []
            for i,(F_,nF) in enumerate(zip((N_,L_,B_),('Nt','Lt','Bt'))):
                F_ = list(set(F_)) or []; tt,fc,fr = sum_vt(F_,wTT=ttcN) if F_ else (np.zeros((2,8)),0,0)
                ft_+= [[tt,fc,fr]]
            (nt,nc,nr),(lt,lc,lr),(bt,bc,br) = ft_
            c = nc+lc+bc; r = (nr*nc +lr*lc +br*bc) /c; m = 0  # br includes overlap?
            if L_:  # as was in sum2G:
                nm,lm,bm = [val_(tt, root.wTT[0]*cNw_) for tt in (nt,lt,bt)]
                spanR = np.mean([n.span for n in N_]) / np.mean([l.span for l in L_])
                tM,tC = 0,0
                if hasattr(root,'m_'):  # root is refined G: type mass of this sub-G's nodes in it
                    for n,tm,tc in zip(root.N_,root.m_,root.c_):
                        if n in n_set: tM += tm*tc; tC += tc
                coreM = (nm* decay**spanR *nc + lm*lc + tM) / (nc+lc+tC)  # Nt + Lt vals
                m = coreM * (1-bm)  # core mass * contour specificity
            if gv_(m * (c*wcN/(r*ccN)) * ((len(N_)-1)*wL) - ave):
                C = sum2F(N_,root,nF='C',typ=3,m=m); C.L_,C.B_ = L_,B_  # seed: raw G with links and perimeter
                Gt_ += [(C,m,c,r)]; in__|=in_
    if Gt_:
        seed_, R,C = [],0,0
        for G, m,c,r in Gt_:
            if fsub:
                if gv_(m* (c*wcC /((_r+r+1)*ccC)) * ((len(G.N_)-1)*wL) - ave):
                    seed_+= [G]; R+= r*c; C+= c
            elif gv_(m * (c*wsG /((_r+r+1)*csG)) * ((len(G.N_)-1)*wL) - ave):
                G_ += sum2G(G,wTT)  # splice sub_Gs
        if seed_:
            G_ += cluster_C(root, seed_,_r+R/C,_j) or []  # splice refined Cs
        sum2F(G_, root, nF='Nt')
        FV_(CoF.get(), *sum_vt(G_)[:-1],_r)
    return G_  # final

def cluster_C(root, _C_, _r,_j):  # refine/expand a fixed set of C seeds through their members' rims, add re-seeding by rejected Ns?

    N_= {n for C in _C_ for n in C.N_}; tC_= []
    for n in N_: n._root_ = n.root_; n.root_ = []  # preserve initial cluster_N memberships in _root_
    while True:  # reform C_, update and test C rims
        C_, cnt,rdn,Up,lc = [],0,0,0,0
        for _C in _C_:
            _N_ = set(_C.N_) | {_n for L in _C.B_ for _n in L.nt}  # current members + perimeter frontier
            n_, m_, M, T, R, up, c = [], [], 0, 0, 0, 0, 0
            for n in _N_:
                if n not in N_: N_.add(n); n.root_, n._root_ = [], []
                dtt = base_comp(_C, n,_j, fder=0)[0]
                m = val_(dtt,cCw_); c += n.c
                _m = next(((rt[1]) for rt in n._root_ if rt[0] is _C), 0)  # prior membership, 0 if new
                if m * n.c > ave * n.r:  # += new nodes and candidate nodes
                    up += abs(_m - m); n_ += [n]; m_ += [m]; T += n.c; M += m * n.c; R += n.r * n.c
                else: up += _m  # exclusion
            if not n_: continue
            r = _r + R/T
            if M * _C.m * (wcC*c / (_C.r* ccC*r)) * (((L := len(n_)) - 1) * wL) > ave:
                C = sum2F(n_, root, m_, nF='C', typ=3)  # -> n.root_
                # C.L_,C.B_ = rim2LB(n_)  # for next-pass frontier and Ft2G
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
                oC_ = cluster_P(oC_, root, _j)
                # for C in oC_: C.L_,C.B_ = rim2LB(C.N_)  # re-formed in cluster_P
                break
            else:
                for n in N_: n._root_ = n.root_; n.root_ = [rt for rt in n.root_ if rt[0] in tC_]
                _C_ = C_  # for next loop
        else: break
    if oC_:
        for C in oC_:
            for N, m in zip(C.N_, C.m_):
                if N.m > m: C.r += 1/len(C.N_)  # xr: cross-fork redundancy
                else: N.r += 1/len(oC_)
        if oC_ := [C for C in oC_ if C.m > ave * C.r]:  # -> xcomp med_N_ or G_
            for n in N_: n.root_ = [rt for rt in n.root_ if rt[0] in oC_]
            FV_(CoF.get(), *sum_vt(oC_))
            return oC_

def cluster_P(_C_, root,_j): # multi-seed mean shift: parallel centroid refine, _C_ varies via split/merge

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
            for i,C in enumerate(_C_): m__[j,i] = val_(base_comp(C,N,_j,fder=0)[0],cPw_)
        C_ = [sum2F(N_, root, m__[:,i],nF='C',typ=3) for i in range(Lc)]  # mean shift, aligned to m__, add typ?
        removed = []
        posL = (m__ > ave).sum()
        if gv_(posL*wcP - ave*(root.r+ccP*totL)):  # merge redundant Cs
            for i,_C in enumerate(C_):
                if _C in removed: continue
                for C in C_[i+1:]:
                    if C in removed: continue
                    m = val_(base_comp(C,_C,_j,fder=0)[0],cPw_); r = (C.r+_C.r)
                    if m*wF > ave*(r+cF):
                        add2F(_C,C,2); removed += [C]; _C.m = val_(_C.dTT,cPw_)  # for next-loop base comp
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
                oC = sum2F(N_,root, m_,nF='C', typ=3)
                out_ += [oC]
    if out_:
        iTT, iC, iR = sum_vt(iC_); oTT,oC,oR = sum_vt(out_)
        FV_(CoF.get(), iTT-oTT, iC-oC, iR-oR)
    return out_

def m_func(m_, wT=wT):  # G.m / param w/o its own column
    wm_ = m_ * wT
    return (wm_.sum() - wm_) / (wT.sum() - wT)

def sum2G(Ft, wTT, root=None, _r=0, _j=0):  # F2G, sub+ -> spliced sub_Gs
    G = Ft2G(Ft, wTT, root, _r,_j)
    sub_G_ = []
    if Lt := G.Lt:  # sub+ CN, no rng+, agg+?
        m,c,r = Lt.m,Lt.c, Lt.r+_r
        if gv_(m* (c*wcN / (r*ccN)) * ((len(G.N_)-1)*wL) - ave):  #-> sub_Gs @ higher filter: r+1
            sub_F_ = cluster_N(G, exemplar_(G.N_,r), r+1,c,_j, fsub=1)  # refined Ft sub_Cs
            sub_G_ = [Ft2G(Ft, wTT, root, r+1,_j) for Ft in sub_F_]  # CN_ sub_Cs
    return sub_G_ or [G]  # return [G} if no sub Gs?

def Ft2G(Ft, wTT, root=None, _r=0, _j=0):  # F_ | (N_,L_,B_) -> CN: forks, vals, no sub+
    def trans_cluster(G):
        for L in G.L_:  # draft:
            for tL in (L.Nt.N_ + L.Bt.N_):  # top level
                if tL.m * (tL.c*wcN / (tL.r*ccN)) > ave:
                    pass  # sum m across levs for merge eval?
                for lev in L.H:  # tFt/Nt | tN_/Bt, not Lt  # not revised:
                    if lev:
                        for tF in lev.N_ if not lev.N_[0].nt else [lev]:  # CF|CL: Nt|Bt, from spec?
                            for tL in tF.N_:
                                if tL.m * (tL.c*wcN / (tL.r*ccN)) > ave:  # merge trans_link.N_.roots:
                                    rt0 = getattr(tL.N_[0].root,'root',None); rt1 = getattr(tL.N_[1].root,'root',None)
                                    if rt0 and rt1 and rt0 != rt1:
                                        if rt1.H: add_H(rt0.H, rt1.H, rt0, fN=1)
                                        add2F(rt0,rt1, merge=1,fH=0)  # recompute Nt attrs / G
                    else: break  # !top lev
    G = CN(root=root,wTT=wTT)
    L_, pL_ = [],[]
    if isinstance(Ft,tuple): N_,_L_,B_ = Ft  # from trace_edge, vect_edge, frame_H
    else:  # C from CN,CC
        N_,_L_,B_ = Ft.N_,Ft.L_,Ft.B_
        if not B_:  # root seed: perimeter per G
            n_set = set(N_)
            B_ = [L for L in {l for n in N_ for l in n.rim} if not (L.nt[0] in n_set and L.nt[1] in n_set)]
        if hasattr(Ft,'m_'): G.m_,G.c_ = Ft.m_,Ft.c_  # refined C: memberships
    for L in _L_:
        if L.typ==-1: pL_+= [L]
        else:         L_ += [L]
    if pL_:
        pm, pc, pr = sum_vt(pL_, fm=1, wTT=wTT)[0], sum([L.c for L in pL_]), np.mean([L.r for L in pL_])
        if pm * (wN*pc /(cN*pr)) > ave:
            for pL in pL_:
                L_ += [comp_N(*pL.nt,pL.r,pL.c,1,pL.angl[0],pL.span,root,_j)]
                for N in pL.nt: N.rim.remove(pL)
    for N in N_:
        if N.rim: N.Rt = sum2F(N.rim,root=N,nF='Rt')
    [sum2F(f_,G,nF=nF) for f_,nF in zip((N_,B_,L_),('Nt','Bt','Lt')) if f_]  # no Dt?
    if G.L_:
        trans_cluster(G)  # splice trans_links, merge L.nt.roots
    Nt,Lt,Bt = G.Nt,G.Lt,G.Bt
    G.dTT, G.c, G.r = sum_vt([f for f in (Nt,Lt) if f])  # combined dTT? use m_func?
    spanR = np.mean([n.span for n in Nt.N_]) / np.mean([l.span for l in Lt.N_]) if Lt else 0
    cM,cC = (sum(np.multiply(G.m_,G.c_)), sum(G.c_)) if hasattr(G,'m_') else (0,0)
    coreM = (Nt.m* decay**spanR *Nt.c + Lt.m*Lt.c + cM) / (Nt.c+Lt.c +cC)  # Nt+Lt vals
    G.m = coreM * (1-Bt.m)  # core mass * contour specificity
    nm_,lm_,bm_ = [m_func(tt[0]) for tt in (Nt.dTT,Lt.dTT,Bt.dTT)]
    G.mt_ = ((nm_* decay**spanR *Nt.c) + lm_*Lt.c) / (Nt.c+Lt.c) * (1-bm_)
    FV_(CoF.get(), G.dTT, G.c, G.r)
    return G

def sum2F(N_, root=None, m_=None, nF=None, typ=3, m=None):  # c|c*m -weighted sum -> CF|CL, bound to root.nF unless fuzzy C

    N = N_[0]; c_ = np.array([n.c for n in N_], dtype=float); C = c_.sum(); fC = m_ is not None
    w_ = c_ * np.array(m_) if fC else c_.copy()
    w_ /= w_.sum()
    TT = np.tensordot(w_, [n.dTT for n in N_], 1); R = w_ @ [n.r for n in N_]
    F = (CN if (geo:= hasattr(N,'yx')) else CF)(dTT=TT,c=C,r=R, nF=nF, root=root,typ=typ); F.N_ = list(N_)
    if geo:  # links and higher
        F.yx,F.kern = [w_ @ np.array([getattr(n,a) for n in N_]) for a in ('yx','kern')]; F.span = w_ @ [n.span for n in N_]
        F.box = reduce(extend_box, [n.box for n in N_ if hasattr(n,'box')], np.array([np.inf,np.inf,-np.inf,-np.inf]))
    if fC:  # not revised
        F.w = F.m; F.m_ = m_; F.c_ = c_; F.m = c_@m_ /C  # ? This is still valid to get scaled m based on m_?
        for n,m in zip(N_,m_): n.root_ += [[F,m]]
        F.L_,F.B_ = rim2LB(N_) 
    else: F.m = val_(TT) if m is None else m
    if nF=='Lt': [add_H(F.H, n.H, F) for n in N_ if n.H]
    if root is None: return F
    F.wTT = root.wTT
    if nF in ('Nt','Lt'): root.dTT = (root.dTT*root.c + TT*C) / (root.c+C); root.c+=C  # root r is not affected?
    if nF=='Lt':  # oriented, as in base_comp
        root.angl = F.angl = (A:= [np.sum(A_:= [L.angl[0] * L.angl[1] for L in F.N_], axis=0), 1])  # Lt.angl for proj_V
        root.mang = F.mang = np.mean([comp_A(A[0], a)[0] for a in A_])
    elif nF == 'Nt':
        root.yx,root.kern,root.box = F.yx,F.kern,F.box; root.span=np.hypot(*(F.box[2:]+1 -F.box[:2])) / 2  # extent, as in PP2N
        for n in N_: n.root = root
        if Ft:=getattr(root,nF,[]): F.H = copy(Ft.H) + [Copy_(Ft, root=root)]
        else: [add_H(F.H, n.H, F) for n in N_]
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
        R.dTT = (R.dTT*R.c + f.dTT*f.c) / (R.c+f.c); R.c += f.c  # complete root binding, as in sum2F
        R.angl = F.angl = [A:= np.sum(A_:= [L.angl[0] * L.angl[1] for L in F.N_], axis=0), 1]
        R.mang = F.mang = np.mean([comp_A(A, a)[0] for a in A_])
        for N in f.N_: N.root = F
    if fH==2: F.H += [f]  # f is F's next level
    elif fH and hasattr(F,'H') and getattr(f,'H',None): add_H(F.H, f.H, F)  # redundant to add_Nt?
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

    Nt = Copy_(F, root=F, cls=CF,typ=3); Nt.N_=F.N_; L_=F.L_; H = F.H  # replace in cross_comp
    F.__class__ = CN; F.Nt = Nt
    [setattr(F, ft, CF(root=F)) for ft in ('Lt','Bt','Dt','Xt','Rt') if not getattr(F, ft, None)]
    box = F.box if hasattr(F, 'box') else np.array([np.inf,np.inf,-np.inf,-np.inf])  # keep existing box
    Na_ = dict(H=copy(H), mang=1, box=box, exe=0, root_=copy(F.root_), compared = set())
    if not hasattr(F, 'kern'):  # CF | PP, no overlap for Cs (only CF)
        Na_.update(kern=np.zeros(4), span=1, angl=None, yx=np.zeros(2))
    for k,v in Na_.items(): setattr(F, k, copy(v))
    if L_: F.H += [sum2F(L_, F)]
    return F

def Copy_(N, root=None, r=1, cls=None, init=0, typ=None, froot=0):
    cls = cls or type(N)
    a = dict(dTT=N.dTT*r,m=N.m,d=N.d,c=N.c,r=N.r, root=root or N.root, nF=N.nF,wTT=copy(N.wTT),typ=N.typ if typ is None else typ,
             N_=copy(N.N_), L_=copy(N.L_))
    if hasattr(N, 'w'): a['w'] = N.w
    if getattr(N,'nt',None): a['nt'] = copy(N.nt)
    if isinstance(N, CN): a.update(yx=copy(N.yx), kern=copy(N.kern), span=N.span, angl=[copy(N.angl[0]), N.angl[1]] if N.angl else None)
    if hasattr(N, 'root_'): a.update(root_=copy(N.root_))  # CC
    if isinstance(N,CoF): a.update(call_=copy(N.call_), typ_=copy(N.typ_) if hasattr(N,'typ_') else [], fw=N.fw if hasattr(N,'fw') else 0, fc=N.fc,fr=N.fr)
    C = cls(**a)
    if isinstance(N, CN):
        if init:
            C.yx=[N.yx]; C.angl=[copy(N.angl[0]), N.angl[1]] if N.angl is not None else None
            C.L_=[l for l in N.rim if l.m>ave]; N.root=C; C.N_=[N]
        else:
            for f in ('Nt','Lt','Bt','Dt','Xt','Rt'):
                setattr(C, f, Copy_(getattr(N,f), root=C))
                if f in ('Nt','Lt','Bt','Dt'): getattr(C,f).H = [Copy_(lev, root=C) if lev else [] for lev in getattr(N,f).H]
            C.angl=deepcopy(N.angl); C.yx=copy(N.yx); C.box=copy(N.box); C.mang=N.mang; C.exe=N.exe; C.root_=list(N.root_)
    if hasattr(N,'mt_'): C.mt_ = np.array(N.mt_, copy=True)
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
    if fi > 1: H.root.node_ = H.node_
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

''' frame expansion: cross_comp lower-tile N_,C_, forward results to next lev, project feedback to scan new lower windows

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
def proj_L_(pairs, root, r, max=20, fall=1, _j=0):

    def proj_V(_N, N, dist, dy_dx, dec, r):  # _N x N induction
        Dec = dec or decay ** ((dist / ((_N.span + N.span) / 2)))
        iTT = (_N.dTT + N.dTT) * Dec
        eTT = (_N.Rt.dTT + N.Rt.dTT) * Dec
        C = min(_N.c,N.c); R = (_N.r+N.r)/2
        if val_((eTT + iTT), Prjw_) * (C / (cPrj + r + R)) * wPrj > ave:  # not oF, spec / link:
            eTT += proj_N(N, dist, dy_dx, r, N.c, dec, _j)[0]  # pTT/ L_,B_,rim, if pV >0
            eTT += proj_N(_N,dist,-dy_dx, r,_N.c, dec, _j)[0]  # reverse direction
        return iTT + eTT
    pL_, olp_ = [], []  # no olp_?
    for _N, N in pairs:  #-> all-to-all pre-links
        if len(_N.H) != len(N.H): continue  # or comp x agg Lev?
        if N is _N: olp_ += [N]  # overlap = unit match, no miss
        elif not (set(N.rim) & set(_N.rim)):  # not nrng-1 link
            dy_dx = _N.yx - N.yx; dist = np.hypot(*dy_dx)  # rim angl is not canonic
            if dist < max :
                pTT = proj_V(_N, N, dist, dy_dx, decay** (dist/ ((_N.span+N.span)/2)) if root==2 else root.m, r)  # based on current rim
                m = val_(pTT, Nw_)
                if fall or m > ave:
                    lc = min(_N.c, N.c); lr = r + (N.r + _N.r) / 2  # +|-match certainty
                    pL_ += [[dist, dy_dx, _N, N, lc, lr, pTT, m]]
    return pL_

def proj_focus(PV__, new__, y,x, tile, elev):  # radial accum of projected focus value in PV__, new__: unprocessed cells

    H,W = PV__.shape  # = win__
    m = val_(tile.dTT, tile.wTT[0]*Frmw_); c = tile.c  # rel match
    d = (np.abs(tile.dTT[1]) / avad__[0]) @ wT / wT.sum()  # rel |d| in m weights, sign in angl
    A = tile.angl[0]*tile.angl[1] if tile.angl else np.zeros(2)  # direction of increase, as in base_comp, angl=None if no Lt
    Vm = (m-ave) * c  # dTT is averaged: *c for both
    Vd = d * tile.mang * c
    nA = np.hypot(*A) or eps
    Dec = decay ** (np.hypot(H**elev,W**elev) / (np.hypot(*(tile.box[2:]-tile.box[:2])) +eps))  # decay in units of tile span
    for n in range(1, max(y,x,H-1-y,W-1-x)+1):  # rim layer n: (2n+1)^2 kernel perimeter, 8n cells
        rim = np.arange(-n,n+1)
        dy_ = np.r_[np.full(2*n+1,-n), np.full(2*n+1,n), rim[1:-1], rim[1:-1]]  # top, bottom rows, left, right cols w/o corners
        dx_ = np.r_[rim, rim, np.full(2*n-1,-n), np.full(2*n-1,n)]
        y_,x_ = dy_+y, dx_+x; i_ = (y_>=0) & (y_<H) & (x_>=0) & (x_<W)  # clip to frame
        dy_,dx_,y_,x_ = dy_[i_], dx_[i_], y_[i_], x_[i_]
        dist_ = np.hypot(dy_,dx_)  # diag = 1.41 axial
        pV_ = (Vm + Vd* (dy_*A[0]+dx_*A[1]) / (dist_*nA)) * Dec**dist_  # layer projection: signed alignment, decay by cell dist
        PV__[y_,x_] += new__[y_,x_] * pV_  # skip processed cells
        if pV_.max() * Dec < ave:
            break  # projected next-layer max < min adjustment

def proj_N(N, dist, A,_r,_c, dec=1, j=0):  # arg rc += N.rc+Nw, recursively specify N projection val, add pN if comp_pN?

    def proj_TT(L, cos_d, dist, r, pTT, wTT, dec=1, fdec=0):  # accumulate L|N' pTT with iTT|eTT internally
        m_,d_ = L.dTT; ad_ = np.abs(d_); rdist = dist / L.span
        avad_ = avad__[j] + avad__[j+1]  # L.dTT[1] = d + dd: dLevs j and j+1
        Dec = dist if fdec else decay ** (1+ rdist*dec)
        rd_ = ad_ / eps_(avad_+ad_)  # net |d| vs regime mean |d|, in 0:1: avad_ in place of m_
        ddec_ = rd_ * cos_d * rdist  # directional loss: 0 if no net d
        dm_ = m_ * (1-Dec+ddec_)  # average loss + directional, in match units
        TT = np.array([m_-dm_, d_+ np.where(d_<0,-dm_,dm_) * avad_])  # transfer lost match to signed difference, in param units
        # or rTT: cert should be a ratio?
        cert = abs(val_(TT) * (L.c+wPrj)/(r+cPrj) - ave)  # approximation
        if cert > ave * (r+cPrj):  # certainty margin
            pTT += TT
    cos_d = (N.angl[0].dot(A) / ((np.hypot(*N.angl[0]) * dist) or eps)) * N.angl[1] if N.angl else 0  # int x ext angle alignment, mean=0
    iTT, eTT = np.zeros((2,8)),np.zeros((2,8)); c = 0
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
            m = np.array(val_(L.dTT,Trcw_)) * ((L.c+wTrc)/(r+cTrc))
            if m > 0:      L_ += [L]
            elif -m > ave: B_ += [L]  # contrast: the opposite side of marginal predictability?
        N.B_ = B_
    Gt_ = []; fin_ = []
    for N in N_:  # flood-fill G per seed N
        if N.rim: N.Rt = sum2F(N.rim,root=N,nF='Rt')
        for B in N.B_: B.root_ = []
        if N in fin_: continue
        fin_ += [N]; _N_=[N]; Gt=[]; N.root=Gt
        n_,ntt,nc = [N],N.dTT.copy(),(N.c or 1); l_,ltt,lc = [],np.zeros((2,8)),0; b_,btt,bc = copy(N.B_),np.zeros((2,8)),0  # Gt
        while _N_:
            _N = _N_.pop(0)
            for L in _N.rim:
                if L in L_:
                    n = L.nt[0] if L.nt[1] is _N else L.nt[1]
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
    G_, TT,C,R = [],np.zeros((2,8)),0,0
    for n_,ntt,nc,l_,ltt,lc,b_,btt,bc,merged in Gt_:
        if not merged:
            if gv_(val_(ntt+ltt+btt, Trcw_) * ((nc+lc+bc+wTrc)/(r+cTrc)) - ave):  # wrap singletons
                TT += ntt+ltt+btt; C += nc+lc+bc; R += r*(nc+lc+bc)  # add Bt from neg Ls per PP?
                G_ += sum2G((n_,l_,b_), ttTrc, root, r)  # add sub-PPs from sub+?
            else:
                for N in n_: N.root=root
    if val_(TT,root.wTT[0]*Trcw_) * ((C+wTrc)/(r+1+cTrc)) * ((len(G_)-1)*wL) > ave:
        _G_+=G_; _TT+=TT; _C+=C  # concat in tile
        FV_(CoF.get(), *sum_vt(_G_))
    return _G_, _TT, _C, r+R/_C

# comp_slice hand-off:
def vect_edge(T, iY,iX,Ly,Lx, rV=1):  # T=tile, PP_ cross_comp and floodfill to init focal frame graph, no recursion:

    global ave,avd,Fw_,Fc_  # /= projected V change:
    def PP2N(PP):
        P_,L_,B_,verT,latT,A,S,box,yx, m,d,c = PP
        kern = np.array(latT[:4])
        [mM, mI, mG, mA, mL], [dM, dI, dG, dA, dL] = verT  # re-pack in dTT:
        dTT = np.array([ np.array([mM, mL, mI, mG, mA, mL, mL / 2, 0]),  # extA=0
                         np.array([dM, dL, dI, dG, dA, dL, dL / 2, 0])])
        y,x,Y,X = box; dy,dx = Y+1-y, X+1-x
        A = [np.array(A), np.sign(dTT[1] @ ttVct[1])]  # append sign
        PP = CN(typ=0, dTT=dTT,m=m,c=c,r=1, kern=kern,yx=yx,angl=A,span=np.hypot(dy/2,dx/2))  # set root in trace_edge
        m_, d_ = np.zeros(5), np.zeros(5); PP.B_ = B_; PP.box = box
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
    blob_ = T.N_; G_,TT,C,R = [],np.zeros((2,8)),0,0
    for blob in blob_:
        if not blob.sign:
            if gv_(blob.G * wVct - ave * cVct):  # proxy for comp_slice and slice_edge
                edge = slice_edge(blob, rV); L = len(edge.P_)-1
                if gv_(edge.G * (wVct*L) - sum([P.latT[4] for P in edge.P_]) * (cVct*L)):
                    PPm_ = comp_slice(edge, rV, ttVct, avad__[0][[0,2,3,4,5]],mass__[0][[0,2,3,4,5]])  # add comp_slice's weights?
                    N_ = [PP2N(PPm) for PPm in PPm_]
                    for PPd in edge.link_: PP2N(PPd)  # we don't form Gds?
                    for N in N_:
                        if N.B_:
                            PPd_ = list({B.root for B in N.B_}); N.Bt = sum2F(PPd_, nF='Bt')
                            N.Bt.N_ = PPd_; [setattr(B,'root',N.Bt) for B in PPd_]
                    tt,c,r = sum_vt(N_)
                    if gv_(val_(tt,Vctw_) * ((c+wVct)/ (3+cVct)) * ((len(PPm_)-1)*wL) - ave):
                        G_,TT,c,R = trace_edge([F2N(N) for N in N_], G_,TT,c,3,T)  # flatten B_-mediated Gs
                    C += c  # after c adds new value from trace_edge
    if G_:
        FV_(CoF.get(), TT, C,1)
        return Ft2G((G_,[],[]), ttVct, T,R)  # return sub-PPs?

def frame_H(image, iY,iX, Y,X, rV, elev=1, max_elev=4, ffb=0):

    def fill_frame(_iy,_ix, elev, T):  # expand level_frame from pixel-level seed tile, similar to frame_blobs

        frame = np.full((Ly,Lx),None, dtype=object)  # higer tile Ly,Lx = lower tile Ly,Lx **2
        cy,cx = int(_iy/Ly), int(_ix/Lx)  # seed is the center of frame
        PV__ = np.zeros([Ly,Lx])  # projected value map
        frame[cy,cx] = T; T_=[]; __T_=[(T,cy,cx)]  # output, eval wave
        while __T_:
            _T_ = []  # next wave
            for _T, y,x in __T_:  # last wave
                if gv_(val_(_T.dTT,_T.wTT[0]*Frmw_) * ((_T.c+wFrm)/(_T.r+cFrm)) - ave):
                    T_ += [_T]; dy,dx = _T.box[2:] -_T.box[:2]
                    pTT, pc = proj_N(_T, np.hypot(dy,dx), np.array([dy,dx]), elev,_T.c)  # no proj r?
                    if gv_(val_(pTT,_T.wTT[0]*Frmw_) * ((pc+wFrm)/(_T.r+elev+cFrm)) - ave):  # +ve, no uncertainty projection yet
                        proj_focus(PV__,frame==None,y,x,_T,elev)  # -> projected value map
                        for _y,_x in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):  # fill 4 adjacent cells
                            if not (0<=_y<Ly and 0<=_x<Lx) or frame[_y,_x] is not None: continue  # outside frame or checked
                            if gv_(PV__[_y,_x] - ave):  # accumulated from all adjacent tiles
                                iy = _y**elev; ix = _x**elev
                                if T := frame_H(image, iy,ix, Y,X, rV, max_elev=elev+1):  # agg+: incr to the current level
                                    _T_ += [(T,_y,_x)]; frame[_y,_x] = T
                                else: frame[_y,_x] = 0
            __T_ = _T_
        if T_:
            TT,C,R = sum_vt(T_, wTT=Frmw_); R += elev
            if val_(TT,Frmw_) * ((C+wFrm)/(R+cFrm)) > ave:
                return T_,C,R
        return [],0,0
    Fr, aH, oH = [],[],[]; aTT=oTT=np.zeros((2,8)); global ave,avd  # regime refs across levs / ffeedback
    T = vect_edge( frame_blobs_root( comp_pixel( image[iY:iY+Ly**elev, iX:iX+Lx**elev]), rV), iY,iX,Ly,Lx, rV)  # base process
    while T and elev < max_elev:
        tile_,C,R = fill_frame(iY, iX, elev, T)  # seed tile -> sparse higher scope tile( oH( aH
        if tile_:
            N_ = [g for t in tile_ for g in t.N_]; m,_,c,r = sum_vt(N_,fm=1)  # concat edge Gs
            Fr = Ft2G((N_,[],[]),ttFrm)  # use sum2G to get angl and l_, for the next loop's T (T = Fr)
            Fr.Nt.H += [sum2F(tile_)]  # minimally processed level
            if gv_(m * c * wX - ave * (r+1+cX)):
                cross_comp(Fr, proj_L_(combinations(N_,2),Fr, r),r)  # agg+
                if elev and ffb:  # ffb=1 in main, no ffeedback in side tiles
                    Fr,aTT,oTT,aH,oH = ffeedback(Fr,aTT,oTT,aH,oH)  # term,form oH ( aH
                    elev += 1; T=Fr  # next-extension seed
                else: break
            else: break
        else: break
    if Fr: FV_(CoF.get(), Fr.dTT, Fr.c, Fr.r)
    return Fr  # intra-lev feedback

def ffeedback(frame, _TT,oTT, aL,oL):  # recompute filters from regime drift; fork: reform oF_ on cross-regime drift

    global ave,avd, oF_,nF_,iF_, p___,m___,gc__, d___,cc__, acov__,_span
    def Mass_(Dec):  # per dLev: avad from comps, mass from clusters
        for p__,m__,gc_,d__,cc_, mass_,avad_,acov_ in zip(p___,m___,gc__,d___,cc__, mass__,avad__,acov__):
            acov_ *= Dec  # past evidence, decayed by relative span
            if cc_:
                cc_ = np.array(cc_)[:,None]; avad_[:] = (d__*cc_).sum(0) / cc_.sum()  # mean |d| / param
            if gc_:
                gc_ = np.array(gc_)[:,None]; C = gc_.sum()
                devI__ = p__ - (p__*gc_).sum(0) / C  # |param| dev / G
                devM__ = m__ - (m__*gc_).sum(0) / C  # dev of G.m w/o own column / G
                covI__ = devI__ * devM__ * gc_
                acovI_ = np.abs(covI__).sum(0)  # epoch evidence
                mass_[:] = (mass_*acov_ + covI__.sum(0)) / eps_(acov_+acovI_)  # -1:1: prior * past evidence + epoch share * epoch evidence
                acov_ += acovI_
    for oF in oF_:
        if oF.V_: oF.w += sum(oF.V_) + sum(oF.gV_)  # all extensive
    dTT = dc = dr = 0
    _ac,_ar = (aL.c,aL.r) if aL else (0,0); _oc,_or = (oL.c,oL.r) if oL else (0,0)
    # H init @ 1st term:
    if aLev := dFilt(frame,'aH',wBac, cBac,_TT):  # update filters per Lev: lev_
        aL= aLev; dTT = aL.dTT-_TT; _TT=aL.dTT; dc= aL.c-_ac; dr= aL.r-_ar  # draft
        ave = val_(_TT)  # filters *= ave
        Dec = (decay ** (frame.span /_span)) if _span else 1  # or actual distance?
        Mass_(Dec); _span = _span+frame.span
        p___,m___,gc__,d___,cc__ = ([[[] for _ in mass__] for _ in range(5)])
        if oLev := dFilt(frame,'oH', wBac, cBac**2, oTT):
            oL= oLev; dTT += oL.dTT- oTT; oTT=oL.dTT; dc+=oL.c-_oc; dr+=oL.r-_or
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
    if dc: FV_(CoF.get(),dTT,dc,dr)
    return frame, _TT, oTT, aL, oL

def dFilt(frame, nF, w, c, _dTT):  # drift-gated regime termination for aH and oH

    H = frame.Nt.H
    n = ('aH','oH') if nF=='aH' else ('oH',)
    i = next((j+1 for j in reversed(range(len(H))) if H[j].nF in n), 0)  # packed levs
    if tail := H[i:] if nF=='aH' else [l for l in H[i:] if l.nF=='aH']:  # lev_| aH_
        D = sum(np.sum(np.abs(t.dTT-_dTT) * wTT) for t in tail)  # drift
        if (vD := D*w - ave*c) > 0:  # update value
            seg = CN(nF=nF, root=frame); seg.H=tail; seg.dTT = tail[-1].dTT; seg.c = sum(l.c for l in tail)  # default regime summary
            if vD > ave:
                seg.dTT,seg.c,seg.r = sum_vt(tail); seg.m = val_(seg.dTT)  # deep summary
            H = frame.Nt.H; frame.Nt.H = H[:i]+[seg]; seg.Nt.H = tail
            return seg

def add_fH(nlev):  # add dLevs: filters + their empty sample lists
    while len(mass__)<nlev:
        mass__.append(copy(mass__[-1]) if mass__ else np.zeros(8))  # init from j-1: d of massive is massive
        avad__.append(copy(avad__[-1]) if avad__ else copy(wT))
        acov__.append(copy(acov__[-1]) if acov__ else np.zeros(8))  # inherit evidence with the prior
        p___.append([]); m___.append([]); gc__.append([]); d___.append([]); cc__.append([])

if __name__ == "__main__":  # './images/toucan_small.jpg' './images/raccoon_eye.jpeg', add larger global image
    trace_func(vars()); g = vars()
    parse_funcs(["agg_recursion.py"])   # populate nF_
    for oF in oF_: oF.fdef = nF_[oF.nF]  # update fdef with ast node
    # inject_oF_(oF_,g)
    Y, X = imread('./images/toucan.jpg').shape
    add_fH(2)  # filters for 2 dLevs for base_comp
    frame = frame_H(image=imread('./images/toucan.jpg'), iY=Y//2 -31, iX=X//2 -31, Y=Y, X=X, rV=1, ffb=1)
    # search frames ( tiles inside image, at this size it should be 4K, or 256K panorama, won't actually work on toucan