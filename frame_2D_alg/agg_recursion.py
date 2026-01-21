import numpy as np
from copy import copy, deepcopy
from math import atan2, cos, floor, pi  # from functools import reduce
from itertools import zip_longest, combinations, chain, product  # from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, imread, comp_pixel, CBase
from slice_edge import slice_edge
from comp_slice import comp_slice, w_t
'''
This is a main module of open-ended clustering algorithm, designed to discover empirical patterns of indefinite complexity. 
Lower modules cross-comp and cluster image pixels and blob slices(Ps), the input here is resulting PPs: segments of matching Ps.
Cycles of (generative cross-comp, compressive clustering, filter-adjusting feedback) should form hierarchical model of input stream: 

Cross-comp forms Miss and Match (min: shared_quantity for directly predictive params, else inverse deviation of miss or variation), in 2 forks:
rng+: incremental-range cross-comp nodes: edge segments at < max distance, cluster if they match. 
der+: incremental-derivation cross-comp links, from node cross-comp, if abs_diff * relative_adjacent_match

Clustering compressively groups the elements into compositional hierarchy, initially by pair-wise similarity / density:
nodes are connectivity clustered, progressively reducing overlap by exemplar selection, centroid clustering, floodfill.
links are correlation clustered, forming contours that complement adjacent connectivity clusters.
Each composition cycle goes through <=4 stages, shifting from connectivity to centroid-based:

- select sparse exemplars to seed the clusters, top k for parallelization? (get_exemplars),
- connectivity/ density-based agglomerative clustering, followed by divisive clustering (cluster_N), 
- sequential centroid-based fuzzy clustering with iterative refinement, start in divisive phase (cluster_C),
- centroid-parallel frame refinement by two-layer EM if min global overlap, prune, next cros_comp cycle (cluster_P).

That forms hierarchical graph representation: dual tree of down-forking elements: node_H, and up-forking clusters: root_H:
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
Similar to neurons: dendritic input tree and axonal output tree, but with lateral cross-comp and nested param sets per layer.

The fitness function is predictive value of the model, estimated through multiple orders of projection:
initially summed match, refined by external projection in space and time: match combined with directional diffs,
then by comparing projected to actual match, testing the accuracy of prior cross_comp and clustering process. 
And so on, higher orders of projection should be generated recursively from lower-order comparisons.

Feedback: projected match adjusts filters (with coord filters selecting new input) to maximize next marginal match.
(currently in ffeedback(), to be refined by cross_comp of co-projected patterns: "imagination, planning, action" in part 3)   
Similar to backprop but sparse, and the filters only control operations, it's not weights that directly scale individual parameters.

It should also modify the code by adding weights on code elements according to their contribution to projected match.
These weights will trigger skipping or recursing over corresponding functions ( blocks ( operations.
Also cross-comp and cluster (compress) code elements and function calls, real and projected, though much more coarsely than data?

notation:
prefix  f denotes flag
postfix t denotes tuple, multiple ts is a nested tuple
prefix  _ denotes prior of two same-name vars, multiple _s for relative precedence
postfix _ denotes array of same-name elements, multiple _s is nested array
capitalized vars are summed small-case vars
'''
eps = 1e-7

def prop_F_(F):  # factory function to set property+setter to get,update top-composition fork.N_
    def Nf_(N):  # CN instance
        Ft = getattr(N, F)  # Nt| Lt| Bt| Ct
        return Ft.N_[0] if (Ft.N_ and isinstance(Ft.N_[0], CF)) else Ft
    def get(N): return getattr(Nf_(N),'N_')
    def set(N, new_N): setattr(Nf_(N),'N_',new_N)
    return property(get,set)

class CN(CBase):
    name = "node"
    # n.Ft.N_[-1] if n.Ft.N_ and isinstance(n.Ft.N_[-1],CF) else n.Ft.N_:
    N_, L_, B_, C_ = prop_F_('Nt'), prop_F_('Lt'), prop_F_('Bt'), prop_F_('Ct')
    def __init__(n, **kwargs):
        super().__init__()
        n.typ = kwargs.get('typ', 0)
        # 0=PP: block trans_comp, etc?
        # 1= L: typ,nt,dTT, m,d,c,rc, root,rng,yx,box,span,angl,fin,compared, Nt,Bt,Ct from comp_sub?
        # 2= G: + rim, eTT, em,ed,ec, baseT,mang,sub,exe, Lt, tNt,tBt,tCt packed in fork levels?
        # 3= C: base_comp subset, +m_,d_,r_,o_ in nodes?
        n.m,  n.d, n.c = kwargs.get('m',0), kwargs.get('d',0), kwargs.get('c',0)  # sum forks to borrow
        n.dTT = kwargs.get('dTT',np.zeros((2,9)))  # Nt+Lt dTT: m_,d_ [M,D,n, I,G,a, L,S,A]
        n.rim = kwargs.get('rim',[])  # external links, rng-nest?
        n.em, n.ed, n.ec = kwargs.get('em',0),kwargs.get('ed',0),kwargs.get('ec',0)  # sum dTT
        n.eTT = kwargs.get('eTT',np.zeros((2,9)))  # sum rim dTT
        n.rc  = kwargs.get('rc', 1)  # redundancy to ext Gs, ave in links?
        n.Nt, n.Bt, n.Ct, n.Lt = (kwargs.get(fork,CF()) for fork in ('Nt','Bt','Ct','Lt'))
        # Fork tuple: [N_,dTT,m,d,c,rc,nF,root], N_ may be H: [N_,dTT] per level, nest=len(N_)
        n.baseT = kwargs.get('baseT',np.zeros(4))  # I,G,A: not ders, in links for simplicity, mostly redundant
        n.nt    = kwargs.get('nt', [])  # nodet, links only
        n.yx    = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.box   = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.span  = kwargs.get('span',1) # distance in nodet or aRad, comp with baseT and len(N_), not additive?
        n.angl  = kwargs.get('angl',[np.zeros(2),0])  # (dy,dx),dir, sum from L_
        n.mang  = kwargs.get('mang',1)  # ave match of angles in L_, =1 in links
        n.root  = kwargs.get('root',None)  # immediate
        n.sub = 0 # composition depth relative to top-composition peers?
        n.fin = kwargs.get('fin',0)  # clustered, temporary
        n.exe = kwargs.get('exe',0)  # exemplar, temporary
        n.rN_ = kwargs.get('rN_',[]) # reciprocal root nG_ for bG | cG, nG has Bt.N_,Ct.N_ instead
        n.compared = set()
        # ftree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def __bool__(n): return bool(n.c)

class CF(CBase):
    name = "fork"
    def __init__(f, **kwargs):
        super().__init__()
        f.N_ = kwargs.get('N_',[])  # may be nested as H
        f.dTT= kwargs.get('dTT',np.zeros((2,9)))
        f.m  = kwargs.get('m', 0)
        f.d  = kwargs.get('d', 0)
        f.c  = kwargs.get('c', 0)
        f.rc = kwargs.get('rc', 0)
        f.nF = kwargs.get('nF','')  # 'Nt','Lt','Bt','Ct'?
        f.root = kwargs.get('root',None)
    def __bool__(f): return bool(f.c)

ave = .3; avd = ave*.5  # ave m,d / unit dist, top of filter specification hierarchy
wM,wD,wc, wG,wI,wa, wL,wS,wA = 10, 10, 20, 20, 5, 20, 2, 1, 1  # dTT param root weights = reversed relative estimated ave
wT = np.array([wM,wD,wc, wG,wI,wa, wL,wS,wA]); wTTf = np.array([wT*ave, wT*avd])
aveB, distw, Lw, intw = 100, .5,.5,.5  # not-compared attr filters, *= ave|avd?
nw,cw, connw,centw, specw = 10,5, 15,20, 10  # process filters, *= (ave,avd)[fi], cost only now
mW = dW = 9  # fb weights per dTT, adjust in agg+
wY = wX = 64; wYX = np.hypot(wY,wX)  # focus dimensions
decay = ave / (ave+avd)  # match decay / unit dist?

''' Core process per agg level, as described in top docstring:

- Cross-comp nodes, evaluate incremental-derivation cross-comp of new >ave difference links, recursively. 
- Select exemplar nodes, links, centroid-cluster their rim nodes, selectively spreading out within a frame. 

- Connectivity-cluster select exemplars/centroids by >ave match links, correlation-cluster links by >ave diff.
- Form complemented (core+contour) clusters for recursive higher-composition cross_comp. 

- Forward: extend cross-comp and clustering of top clusters across frames, re-order centroids by eigenvalues.
- Feedback coords to bottom level or prior-level in parallel pipelines, filter updates in more coarse cycles '''

def vt_(TT, rc=0, wTT=None):  # brief val_ to get m,d, rc=0 to return raw vals

    m_,d_= TT; ad_ = np.abs(d_); t_ = m_ + ad_ + eps  # ~ max comparand
    if wTT is None: wTT = wTTf
    return m_/t_ @ wTT[0] - ave*rc, ad_/t_ @ wTT[1] - avd*rc  # norm by co-derived variance

def val_(TT, rc, wTT, mw=1.0,fi=1, _TT=None,cr=.5):  # m,d eval per cluster, cr = cd / cm+dc, default half-weight?

    t_ = np.abs(TT[0]) + np.abs(TT[1])  #  combine vals, not sure about abs m_?
    rv = TT[0] / (t_+eps) @ wTT[0] if fi else TT[1] / (t_+eps) @ wTT[1]  # fork / total per scalar
    if _TT is not None:
        _t_ = np.abs(_TT[0]) + np.abs(_TT[1])
        _rv = _TT[0] / (_t_+eps) @ wTT[0] if fi else _TT[1] / (_t_+eps) @ wTT[1]
        rv  = rv * (1-cr) + _rv * cr  # + borrowed alt fork val, cr: d count ratio, must be passed with _TT?

    return rv * mw - (ave if fi else avd) * rc

def TTw(G): return getattr(G,'wTT',wTTf)

def cross_comp(root, rc, fL=0):  # core function mediating recursive rng+ and der+ cross-comp and clustering, rc=rdn+olp

    N_ = (root.N_,root.B_)[fL]; nG_ = []
    iN_, L_,TT,c,TTd,cd = comp_N_(N_,rc) if N_[0].typ<3 else comp_C_(N_,rc, fC=1)  # nodes | centroids
    if L_:
        for n in iN_: n.em, n.ed = vt_(np.sum([l.dTT for l in n.rim],axis=0), rc)
        cr = cd/(c+cd) *.5  # dfork borrow ratio, .5 for one direction
        if val_(TT, rc+connw, TTw(root), (len(L_)-1)*Lw,1, TTd,cr) > 0 or fL:
            sum2F(L_,'Lt',root)  # Bt in dfork
            E_ = get_exemplars({N for L in L_ for N in L.nt if N.em}, rc)  # exemplar N_|C_
            nG_,rc = cluster_N(root, E_,rc,fL)  # form Bt,Ct, 3-fork sub+ in sum2G
        # agg+:
        if nG_ and val_(root.dTT, rc+nw, TTw(root), (len(root.N_)-1)*Lw,1, TTd,cr) > 0:
            nG_,rc = cross_comp(root,rc)
            for nG in nG_:
                if isinstance(nG.Lt.N_[0],CF):  # Lt.N_=H,[1:] = globally spliced trans_links
                    trans_cluster(nG, rc)  # merge trans_link- connected Gs, from sub+ + agg+
    return nG_,rc   # nG_ is recursion flag

def trans_cluster(root, rc):  # trans_links mediate re-order in sort_H?

    def rroot(n): return rroot(n.root) if n.root and n.root != root else n  # root is nG
    FH_ = [[],[],[]]
    for L in root.Lt.N_:  # base links
        for FH,Ft in zip(FH_, (L.Lt,L.Bt,L.Ct)):  # L.Lt maps to N.Nt, else conflict with Lt.N_
            for Lev,lev in zip_longest(FH, Ft.N_):  # default H?
                if lev:
                    if Lev: Lev += lev.N_  # concat, sum2F later
                    else:   FH += [list(lev.N_)]
    # flat Lt.N_
    for FH, nF in zip(FH_, ('Lt','Bt','Ct')):  # spliced trans-forks, Lt maps to Nt
        if FH:  # merge Lt.fork.nt.roots
            for lev in reversed(FH):  # bottom-up to get incrementally higher roots
                for tL in lev:  # trans_link
                    rt0 = tL.nt[0].root; rt1 = tL.nt[1].root; fmerge=1  # same lev?
                    if rt0 is rt1: continue
                    if rt0 is root or rt1 is root: fmerge=0  # append vs. merge?
                    add_N(rt0, rt1, fmerge)  # if fmerge: rt0 should be higher?
            # set tFt:
            FH = [sum2F(n_,nF,getattr(root.Lt,nF)) for n_ in FH]; sum2F(FH,nF, root.Lt)
       # reval rc, not revised
        '''
        tL_ = [tL for n in root.N_ for l in n.L_ for tL in l.N_]  # trans-links
        if sum(tL.m for tL in tL_) * ((len(tL_)-1)*Lw) > ave*(rc+connw):  # use tL.dTT?
            mmax_ = []
            for F,tF in zip((root.Nt,root.Bt,root.Ct), (root.Lt.N_[-1])):
                if F and tF:
                    maxF,minF = (F,tF) if F.m>tF.m else (tF,F)
                    mmax_+= [max(F.m,tF.m)]; minF.rc+=1  # rc+=rdn
            sm_ = sorted(mmax_, reverse=True)
            tNt, tBt, tCt = root.Lt.N_[-1]
            for m, (Ft, tFt) in zip(mmax_,((root.Nt,tNt),(root.Bt,tBt),(root.Ct,tCt))): # +rdn in 3 fork pairs
                r = sm_.index(m); Ft.rc+=r; tFt.rc+=r  # rc+=rdn
        '''

def comp_N_(iN_, rc, _iN_=[]):  # incremental-distance cross_comp, max dist depends on prior match

    for i, N in enumerate(iN_):  # get all-to-all pre-links
        N.pL_ = []
        for _N in _iN_ if _iN_ else iN_[i+1:]:  # optional _iN_ as spec
            if _N.sub != N.sub: continue  # or comp x composition?
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx)
            N.pL_ += [[dist, dy_dx, _N]]
        N.pL_.sort(key=lambda x: x[0])  # proximity prior, test compared?

    def proj_V(_N,N, dist, pVt_):  # _N x N induction
        Dec = decay**(dist/((_N.span+N.span)/2))
        iTT = (_N.dTT + N.dTT) * Dec
        eTT = (_N.eTT + N.eTT) * Dec # rim
        if abs( vt_(eTT,rc)[0]) * ((len(pVt_)-1)*Lw) > ave*specw:  # spec N links
            eTT = np.zeros((2,9)) # recompute
            for _dist,_dy_dx,__N,_V in pVt_:
                eTT += proj_N(N,_dist,_dy_dx, rc)  # proj N L_,B_,rim, if pV>0: eTT += pTT?
                eTT += proj_N(_N,_dist,-_dy_dx, rc)  # reverse direction
        return iTT+eTT

    N_,L_,TTm,cm,TTd,cd = [],[],np.zeros((2,9)),0,np.zeros((2,9)),0; dpTT=np.zeros((2,9))  # no c?
    for N in iN_:
        pVt_ = []
        for dist, dy_dx, _N in N.pL_:  # rim angl is not canonic
            pTT = proj_V(_N,N, dist, pVt_); lrc = rc + (N.rc+_N.rc) / 2  # pVt_: [[dist, dy_dx, _N, V]]
            m, d = vt_(pTT,lrc)  # +|-match certainty
            if m > 0:
                if abs(m) < ave:  # different ave for projected surprise value, comp in marginal predictability
                    Link = comp_N(_N,N, lrc, A=dy_dx, span=dist)
                    dTT, m,d,c = Link.dTT,Link.m,Link.d,Link.c
                    if   m > ave: TTm+=dTT; cm+=c; L_+=[Link]; N_ += [_N,N]  # combined CN dTT and L_
                    elif d > avd: TTd+=dTT; cd+=c  # no overlap to simplify
                    dpTT += pTT-dTT  # prediction error to fit code, not implemented
                else:
                    pL = CN(typ=-1, nt=[_N,N], dTT=pTT,m=m,d=d,c=min(N.c,_N.c), rc=lrc, angl=np.array([dy_dx,1],dtype=object),span=dist)
                    L_+= [pL]; N.rim+=[pL]; N_+=pL.nt; _N.rim+=[pL]; TTm+=pTT; cm+=pL.c  # same as links in clustering, pack L.nt as N_?
                pVt_ += [[dist,dy_dx,_N,m]]  # for next rim eval
            else:
                break  # beyond induction range
    return list(set(N_)), L_,TTm,cm,TTd,cd  # + dpTT for code-fitting backprop?

def comp_N(_N,N, rc, A=np.zeros(2), span=None):  # compare links, optional angl,span,dang?

    TT,_ = base_comp(_N, N)
    yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])  # ext
    angl = [A, np.sign(TT[1] @ wTTf[1])]  # canonic direction
    m, d = vt_(TT,rc)
    Link = CN(typ=1,exe=1, nt=[_N,N], dTT=TT,m=m,d=d,c=min(N.c,_N.c),rc=rc, yx=yx,box=box,span=span,angl=angl, baseT=(_N.baseT+N.baseT)/2)
    for _Ft,Ft,nF in zip((_N.Nt,_N.Bt,_N.Ct),(N.Nt,N.Bt,N.Ct),('Nt','Bt','Ct')):
        if _Ft and Ft:  # add eval?
            comp_F_(_Ft.N_,Ft.N_,nF,rc, Link)  # comp_F_, deeper trans_comp in comp_C_'comp_N, unpack|reref levs?
            rc += 1  # default fork redundancy
        for n, _n in (_N,N), (N,_N):  # if rim-mediated comp: reverse dir in _N.rim: rev^_rev?
            n.rim += [Link]; n.eTT += TT; n.ec += Link.c; n.compared.add(_n)
    return Link

def comp_n(_N,N, TTm,TTd,cm,cd,rc, L_,N_=None):

    dtt = comp_derT(_N.dTT[1],N.dTT[1]) if N_ is None else base_comp(_N,N)[0]  # from comp_C_, no use for rn?
    m,d = vt_(dtt,rc); c = min(_N.c,N.c)
    link = CN(nt=[_N,N], dTT=dtt, m=m,d=d,c=c, span=np.hypot(*_N.yx-N.yx))
    _N.rim += [link]; N.rim += [link]
    if   link.m > ave*(connw+rc): TTm+=dtt; cm+=link.c; L_+= [link]; N_.extend([N,_N]) if N_ is not None else None
    elif link.d > avd*(connw+rc): TTd+=dtt; cd+=link.c  # not in N_
    return cm,cd

def comp_F_(_F_,F_,nF, rc, root):  # root is nG, unpack node trees down to numericals and compare them

    L_,TTm,C,TTd,Cd = [],np.zeros((2,9)),0,np.zeros((2,9)),0; Rc=cc=0  # comp count
    if isinstance(F_[0],CN):
        for _N, N in product(F_,_F_):
            if _N is N: dtt = np.array([N.dTT[1], np.zeros(9)]); TTm += dtt; C=1; Cd=0  # overlap is pure match
            else:       cm,cd = comp_n(_N,N, TTm,TTd,C,Cd,rc,L_); C+=cm; Cd+=cd
            Rc+=_N.rc+N.rc; cc += 1
    else:
        for _lev,lev in zip(_F_,F_):  # L_ = H
            rc += 1; lRc=lcc=0  # deeper levels are redundant
            TT = comp_derT(_lev.dTT[1],lev.dTT[1]); lRc+=1; lC=lcc=1  # min per dTT?
            _sN_,sN_ = set(_lev.N_), set(lev.N_)
            iN_ = list(_sN_ & sN_)  # intersect = match
            for n in iN_: TT+=n.dTT; lC+=n.c; lRc+=n.rc; lcc+=1
            _oN_= _sN_-sN_; oN_= sN_-_sN_; dN_= []
            for _n,n in product(_oN_,oN_):
                cm,_ = comp_n(_n,n, TTm,TTd,C,Cd,rc, dN_); lRc+=_n.rc+n.rc; lC+=cm; lcc+=1  # comp offsets
            TT += TTm; m,d = vt_(TT,rc); C+=lC; Rc+=lRc; cc+=lcc
            L_ += [CF(nF='tF',N_=dN_, dTT=TT,m=m,d=d,c=lC, rc=lRc/lcc, root=root)]
    # temp root is Link, then splice in G in sum2G
    if L_: setattr(root,nF, sum2F(L_,nF,root,TTm,C, fCF=0))  # pass Rc/cc?

def base_comp(_N,N):  # comp Et, baseT, extT, dTT

    _M,_D,_c = _N.m,_N.d,_N.c; _I,_G,_Dy,_Dx =_N.baseT; _L = len(_N.N_)  # N_: density within span
    M, D, c = N.m, N.d, N.c; I, G, Dy, Dx = N.baseT; L = len(N.N_)
    rn = _c/c
    _pars = np.array([_M*rn,_D*rn,_c*rn,_I*rn,_G*rn, [_Dy,_Dx],_L*rn,_N.span], dtype=object)  # Et, baseT, extT
    pars  = np.array([M,D,c, (I,wI),G, [Dy,Dx], L,(N.span,wS)], dtype=object)
    mA,dA = comp_A(_N.angl[0]*_N.angl[1], N.angl[0]*N.angl[1])
    m_,d_ = comp(_pars,pars, mA,dA)  # M,D,n, I,G,a, L,S,A
    dm_,dd_ = comp_derT(rn*_N.dTT[1], N.dTT[1])

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

def comp_C_(C_, rc,_C_=[], fall=1, fC=0):  # simplified for centroids, trans-N_s, levels
    # max attr sort to constrain C_ search in 1D, add K attrs and overlap?
    # proj C.L_: local?
    N_,L_,TTm,cm,TTd,cd = [],[],np.zeros((2,9)),0,np.zeros((2,9)),0
    if fall:
        pairs = product(C_,_C_) if _C_ else combinations(C_,r=2)  # comp between | within list
        for _C, C in pairs:
            if _C is C: dtt = np.array([C.dTT[1],np.zeros(9)]); TTm+=dtt; cm=1;cd=0  # overlap=match
            else:       dtt = base_comp(_C,C)[0]; TTm+=dtt; cm,cd = vt_(dtt,rc)
        if fC:
            _C_= C_; C_,exc_ = [],[]
            L_ = sorted(L_, key=lambda dC: dC.d)  # from min D
            for i, L in enumerate(L_):
                if val_(L.dTT, rc+nw, wTTf,fi=0) < 0:  # merge similar but distant centroids, they are non-local
                    _c,c = L.nt
                    if _c is c or c in exc_: continue  # not yet merged
                    for n in c.N_: add_N(_c,n); _N_ = _c.N_; _N_ += [n]
                    for l in c.rim: l.nt = [_c if n is c else n for n in l.nt]
                    C_ += [_c]; exc_+=[c]
                    if c in C_: C_.remove(c)
                else: L_ = L_[i:]; break
    else:
        # consecutive or distance-constrained cross_comp along eigenvector, or original yx in sub+?
        for C in C_: C.compared=set()
        C_ = sorted(C_, key=lambda C: C.dTT[0][np.argmax(wTTf[0])])  # max weight defines eigenvector
        for j in range( len(C_)-1):
            _C = C_[j]; C = C_[j+1]
            if _C in C.compared: continue
            dy_dx = _C.yx-C.yx; dist = np.hypot(*dy_dx)
            Link = comp_N(_C,C, rc, A=dy_dx, span=dist)  # or comp_derT?
            if   Link.m > ave*(connw+rc): TTm+=Link.dTT; cm+=Link.c; L_+=[Link]; N_ += [_C,C]
            elif Link.d > avd*(connw+rc): TTd+=Link.dTT; cd+=Link.c

    return list(set(N_)), L_,TTm,cm,TTd,cd

def get_exemplars(N_, rc):  # multi-layer non-maximum suppression -> sparse seeds for diffusive clustering, cluster_N too?
    E_ = set()
    for rdn,N in enumerate(sorted(N_, key=lambda n:n.em, reverse=True), start=1):  # strong-first
        oL_ = set(N.rim) & {l for e in E_ for l in e.rim}
        roV = ((vt_(np.sum([l.dTT for l in oL_], axis=0), rc)[0] if oL_ else 0) if oL_ else 0) / (N.em or eps)  # relative rim olp V
        if N.em * N.c > ave * (rc+rdn+nw+ roV):  # ave *= rV of overlap by stronger-E inhibition zones
            E_.update({n for l in N.rim for n in l.nt if n is not N and N.em > ave*rc})  # selective nrim
            N.exe = 1  # in point cloud of focal nodes
        else:
            break  # the rest of N_ is weaker, trace via rims
    return E_

def cluster_N(root, _N_, rc, fL=0):  # flood-fill node | link clusters, flat, replace iL_ with E_?

    def nt_vt(n,_n):
        M, D = 0,0  # exclusive match, contrast
        for l in set(n.rim+_n.rim):
            if l.m > 0:   M += l.m
            elif l.d > 0: D += l.d
        return M, D

    G_, N_,L_ = [],[],[]  # add prelink pL_,pN_? include merged Cs, in feature space for Cs
    if _N_ and (val_(root.dTT, rc+connw, TTw(root), mw=(len(_N_)-1)*Lw) > 0 or fL):
        for N in _N_: N.fin=0; N.exe=1  # not sure
        G_, L__= [],[]; TT,nTT,lTT = np.zeros((2,9)),np.zeros((2,9)),np.zeros((2,9)); C,nC,lC = 0,0,0; in_= set()  # root attrs
        for N in _N_: # form G per remaining N
            if N.fin or (root.root and not N.exe): continue  # no exemplars in Fg
            N_=[N]; L_,B_,C_=[],[],[]; N.fin=1  # init G
            __L_ = N.rim  # spliced rim
            while __L_:
                _L_ = []
                for L in __L_:  # flood-fill via frontier links
                    _N = L.nt[0] if L.nt[1].fin else L.nt[1]; in_.add(L)
                    if not _N.fin and _N in _N_:
                        m,d = nt_vt(*L.nt)
                        if m > ave * (rc-1):  # cluster nt, L,C_ by combined rim density:
                            N_ += [_N]; _N.fin = 1
                            L_ += [L]; C_ += _N.C_
                            _L_+= [l for l in _N.rim if l not in in_ and (l.nt[0].fin ^ l.nt[1].fin)]   # new frontier links, +|-?
                        elif d > avd * (rc-1):  # contrast value, exclusive?
                            B_ += [L]
                __L_ = list(set(_L_))
            if N_:
                Ft_ = ([list(set(N_)),np.zeros((2,9)),0,0], [list(set(L_)),np.zeros((2,9)),0,0], [list(set(B_)),np.zeros((2,9)),0,0], [list(set(C_)),np.zeros((2,9)),0,0])
                for i, (F_,tt,c,r) in enumerate(Ft_):
                    for F in F_:
                        tt += F.dTT; Ft_[i][2] += F.c
                        if i>1: Ft_[i][3] += F.rc  # define Bt,Ct rc /= ave node rc?
                (N_,nt,nc,_),(L_,lt,lc,_),(B_,bt,bc,br),(C_,ct,cc,cr) = Ft_
                c = nc + lc + bc*br + cc*cr
                tt = (nt*nc + lt*lc + bt*bc*br + ct*cc*cr) / c
                if val_(tt,rc, TTw(root), (len(N_)-1)*Lw) > 0 or fL:
                    G_ += [sum2G(((N_,nt,nc),(L_,lt,lc),(B_,bt,bc),(C_,ct,cc)), tt,c, rc,root)]  # calls sub+/ 3 forks, br,cr?
                    L__+=L_; TT+=tt; nTT+=nt; lTT+=lt; C+=c; nC+=nc; lC+=lc
                    # G.TT * cr * rcr?
        if G_ and (fL or val_(TT, rc+1, TTw(root), (len(G_)-1)*Lw)):  # include singleton lGs
            rc += 1
            root_replace(root,rc, TT,C, G_,nTT,nC,L__,lTT,lC)
    return G_, rc

def cluster_C(root, E_, rc):  # form centroids by clustering exemplar surround via rims of new member nodes, within root

    C_, _C_ = [],[]  # form root.Ct, may call cross_comp-> cluster_N, incr rc
    for n in root.N_: n._C_,n.m_,n._m_,n.o_,n._o_ = [],[],[],[],[]
    for E in E_:
        C = cent_TT( Copy_(E, root, init=2,typ=0), rc, init=1)  # all rims are in root, seq-> eigenvector?
        C._N_ = list({n for l in E.rim for n in l.nt if (n is not E and n in root.N_)})  # init C.N_=[]
        for n in C._N_: n.m_+=[C.m]; n._m_+=[C.m]; n.o_+=[1]; n._o_+=[1]; n.Ct.N_+=[C]; n._C_+=[C]
        _C_ += [C]
    while True:  # reform C_
        C_,cnt,olp, mat,dif, DTT,Dm,Do = [],0,0,0,0,np.zeros((2,9)),0,eps; Ave = ave * (rc+nw)
        _Ct_ = [[c, c.m/c.c if c.m !=0 else eps, c.rc] for c in _C_]
        for r, (_C,_m,_o) in enumerate(sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True)):
            if _m > Ave *_o:
                N_,N__,m_,o_,M,D,O,cc, dTT,dm,do = [],[],[],[],0,0,0,0,np.zeros((2,9)),0,0  # /C
                for n in _C._N_:  # frontier
                    dtt,_ = base_comp(_C, n); cc+=1  # add rim overlap to combined inclusion criterion?
                    m,d = vt_(dtt,rc); dTT += dtt; nm = m * n.c  # rm,olp / C
                    odm = np.sum([_m-nm for _m in n._m_ if _m>m])  # higher-m overlap
                    if m > 0 and nm > Ave * odm:
                        N_+=[n]; M+=m; O+=odm; m_+=[nm]; o_+=[odm]  # n.o for convergence eval
                        N__ += [_n for l in n.rim for _n in l.nt if _n is not n]  # +|-Ls
                        if _C not in n._C_: dm+=m; do+=odm  # not in extended _N__
                    else:
                        if _C in n._C_: i = n._C_.index(_C); dm+=n._m_[i]; do+=n._o_[i]
                        D += abs(d)  # distinctive from excluded nodes (background)
                mat+=M; dif+=D; olp+=O; cnt+=cc  # all comps are selective
                DTT += dTT
                if M > Ave * len(N_) * O and val_(dTT, rc+O, TTw(C),(len(N_)-1)*Lw):
                    C = cent_TT(sum2C(N_,_C,_Ci=None), rc=r)
                    for n,m,o in zip(N_,m_,o_): n.m_+=[m]; n.o_+=[o]; n.Ct.N_+= [C]; C.Nt.N_+= [n]  # reciprocal root assign
                    C._N_ = list(set(N__))  # frontier
                    C_ += [C]; Dm+=dm; Do+=do  # new incl or excl
                else:
                    for n in _C._N_:
                        n.exe = n.m/n.c > 2 * ave  # refine exe
                        for i, c in enumerate(n.Ct.N_):
                            if c is _C:  # remove _C-mapping m,o:
                                n.Ct.N_.pop(i); n.m_.pop(i);n.o_.pop(i); break
            else:  # the rest is weaker
                break
        for n in root.N_:
            n._C_ = n.Ct.N_; n._m_= n.m_; n._o_= n.o_; n.Ct.N_,n.m_,n.o_ = [],[],[]  # new n.Ct.N_s, combine with v_ in Ct_?
        if mat * dif * olp > ave * centw*2:
            C_ = cluster_P(C_, root.N_, rc)  # refine all memberships in parallel by global backprop|EM
            break
        if Dm / Do > Ave:  # dval vs. dolp: overlap increases with Cs expansion
            _C_ = C_
        else: break  # converged
    C_ = [C for C in C_ if val_(C.dTT, rc, TTw(C))]  # prune C_
    if C_:
        for n in [N for C in C_ for N in C.N_]:  # exemplar V + sum n match_dev to Cs, m* ||C rvals:
            n.exe = (n.d if n.typ==1 else n.m) + np.sum([m-ave*o for m, o in zip(n.m_, n.o_)]) - ave
        if val_(DTT, rc+olp, TTw(root), (len(C_)-1)*Lw) > 0:
            Ct = sum2F(C_,'Ct',root, fCF=0)
            _,rc = cross_comp(Ct, rc)  # all distant Cs, seq C_ in eigenvector = argmax(root.wTT)?
            root.C_=C_; root.Ct=Ct; root_update(root,Ct)
            Nt = sum2F(C_,'Nt',root); root.N_ = C_; root_update(root,Nt)
    return C_, rc

def cluster_P(_C_,N_,rc):  # Parallel centroid refining, _C_ from cluster_C, N_= root.N_, if global val*overlap > min
                           # cent cluster L_ ~ GNN?
    for C in _C_:  # fixed while training, then prune if weak C.v
        C.N_ = N_  # soft assign all Ns per C
        C.m,C.d,C.rc = C.em,C.ed, np.mean([l.rc for l in C.rim])  # not sure
    while True:
        M = O = dM = dO = 0
        for N in N_:
            N.m_,N.d_,N.r_ = map(list,zip(*[vt_(base_comp(C,N)[0]) + (C.rc,) for C in _C_]))  # distance-weight match?
            N.o_= np.argsort(np.argsort(N.m_)[::-1]) + 1  # rank of each C_[i] = rdn of C in C_
            dM += sum(abs(_m-m) for _m,m in zip(N._m_,N.m_))
            dO += sum(abs(_o-o) for _o,o in zip(N._o_,N.m_))
            M += sum(N.m_)
            O += sum(N.m_)
        C_ = [sum2C(N_,_C, i) for i, _C in enumerate(_C_)]  # update with new coefs, _C.m, N->C refs
        Ave = ave * (rc + centw)
        if M > Ave*O and dM > Ave*dO:  # strong update
            _C_ = C_
        else: break
    _C_,in_ = [], []
    for i, C in enumerate(C_):
        if C.m > ave * centw * C.rc:
            C.N_ = [n for n in C.N_ if n._m_[i] * C.m > ave * n._r_[i] * n._o_[i]]; _C_ += [C]; in_ += [1]
        else: in_ += [0]
    for N in N_:
        for _v,v_ in zip((N._m_,N._d_,N._r_), (N.m_,N.d_,N.r_)):
            v_[:] =[v for v,i in zip(v_,in_) if i]; _v[:] = []
    return _C_
'''
next order is level-parallel cluster_H, over multiple agg+? compress as autoencoder? '''

def sum2C(N_,_C, _Ci=None):  # fuzzy sum params used in base_comp

    c_,rc_,dTT_,baseT_,span_,yx_ = zip(*[(n.c, n.rc, n.dTT, n.baseT, n.span, n.yx) for n in N_])
    ccoef_ = []
    for N in N_:
        Ci = N.Ct.N_.index(_C) if _Ci is None else _Ci
        ccoef_ += [N._m_[Ci] / (ave* N._o_[Ci]) * _C.m]  # *_C.m: proj survival, vs post-prune eval?
        c_ = [c * max(cc, 0) for c,cc in zip(c_,ccoef_)]
        if _Ci is not None: N._m_,N._d_,N._r_ = N.m_,N.d_,N.r_    # should be valid only from cluster_P, when _Ci is not None
    tot = sum(c_)+eps; Par_ = []
    for par_ in rc_, dTT_, baseT_, span_, yx_:
        Par_.append(sum([p * c for p,c in zip(par_,c_)]))
    C = CN(c=tot, typ=0)
    C.rc, C.dTT, C.baseT, C.span, C.yx = [P / tot for P in Par_]
    C.m, C.d = vt_(C.dTT, C.rc)
    C.Nt = CF(N_=N_,dTT=deepcopy(C.dTT), m=C.m,d=C.d,c=C.c)
    return cent_TT(C, C.rc)

def sum2G(Ft_,tt,c,rc, root=None, init=1, typ=None, fsub=1):  # updates root if not init

    N_,ntt,nc = Ft_[0]; L_,ltt,lc = [],np.zeros((2,9)),0; B_,C_ = [],[]  # init tt,c s if B_
    if len(Ft_)>1: L_,ltt,lc = Ft_[1]  # in trace_edge
    if len(Ft_)>2: (B_,btt,bc),(C_,ctt,cc) = Ft_[2:]  # in cluster_N, Ct: alternative, Bt: complementary?
    if not init:
        N_+=root.N_; L_+=root.L_; B_+=root.B_; C_+=root.C_
        nc+=root.Nt.c; lc+=root.Lt.c; bc+=root.Bt.c; cc+=root.Ct.c
        ntt+=root.Nt.dTT; ltt+=root.Lt.dTT; btt+=root.Bt.dTT; ctt+=root.Ct.dTT
        # batch root_updates: n.dTT += update_dTT_?
    N = N_[0]
    if typ is None: typ = N.typ
    m, d = vt_(tt, rc)
    G = Copy_(N,root,init=1,typ=typ); G.dTT=tt; G.m=m; G.d=d; G.c=c; G.rc=rc
    G.Nt = sum2F(N_,'Nt',G, ntt, nc)
    for N in N_[1:]: add_N(G,N, coef=N.c/c)  # sum not-CF vars only?
    if L_:
        G.Lt = sum2F(L_,'Lt',G,ltt,lc)
        A = np.sum([l.angl[0] for l in L_], axis=0)
        G.angl = np.array([A, np.sign(G.dTT[1] @ wTTf[1])], dtype=object)  # angle dir = d sign
    if init:  # else same ext
        yx_ = np.array([n.yx for n in N_]); yx = yx_.mean(axis=0); dy_,dx_ = (yx_-yx).T
        G.span = np.hypot(dy_,dx_).mean()  # N centers dist to G center
        G.yx = yx
    if N_[0].typ==2 and G.L_:  # else mang = 1
        G.mang = np.mean([comp_A(G.angl[0], l.angl[0])[0] for l in G.L_])
    G.m,G.d = vt_(G.dTT,rc)
    if G.m > ave*specw:  # comp typ -1 pre-links
        L_,pL_= [],[]; [L_.append(L) if L.typ==1 else pL_.append(L) for L in G.L_]
        if pL_:
            if vt_(sum([L.dTT for L in pL_]),rc)[0] > ave * specw:
                for L in pL_:
                    link = comp_N(*L.nt, rc, L.angl[0], L.span)
                    G.Lt.dTT+= link.dTT-L.dTT; L_+=[link]
                G.Lt.m, G.Lt.d = vt_(G.Lt.dTT, rc)
            G.L_ = L_
    if B_ and typ > 1 and G.Bt and G.Bt.d > avd * rc * nw:  # no ddfork
        B_,_rc = cross_comp(G, rc, fL=1)  # comp Bt.N_
    if B_: sum2F(B_,'Bt',G, rc)  # maybe updated above
    if C_: sum2F(C_,'Ct',G, rc)
    if fsub:  # sub+
        if G.Lt.m * G.Lt.d * ((len(N_)-1)*Lw) > ave * avd * (rc+1) * cw:  # divisive clustering if Match * Variance
            V = G.m - ave * (rc+1)
            if mdecay(L_) > decay:
                if V > ave*centw: cluster_C(G,N_,rc+1)  # cent cluster: N_->Ct.N_, higher than G.C_
            elif V > ave*connw: cluster_N(G,N_,rc+1)  # conn cluster/ higher filter: N_->Nt.N_ | N_
    G.rN_= sorted(G.rN_, key=lambda x: (x.m/x.c), reverse=True)  # only if lG?
    return G

def sum2F(F_, nF, root, TT=None, C=0, fset=1, fCF=1):

    def add_F(F,f, cr, fmerge=1):
        F.N_.extend(f.N_) if fmerge else F.N_.append(N); F.dTT+=f.dTT*cr; F.c+=f.c; F.rc+=cr  # -> ave cr?
    if not C:
        C = sum([n.c for n in F_]); TT = np.sum([n.dTT for n in F_], axis=0)  # *= cr?
    NH =[]; m,d= vt_(TT)
    if fCF: Ft = CF(nF=nF,dTT=TT,m=m,d=d,c=C,root=root)
    else:   Ft = CN(dTT=TT,m=m,d=d,c=C,root=root); Ft.nF = nF
    for F in F_:
        if F.N_:
            cr = F.c / C
            if isinstance(F.N_[0],CF):  # G.Nt.N_=H, top-down, eval sort_H?
                if NH:
                    for Lev, lev in zip_longest(NH, F.N_):
                        if lev:
                            if Lev is None: NH+= [CopyT(lev,Ft,cr)]
                            else: add_F(Lev,lev, cr*(lev.c/F.c))
                else: NH = [CopyT(lev,Ft,cr*(lev.c/F.c)) for lev in F.N_]
            else:
                L1 = sum2F([n for n in F.N_], nF, F)  # L1.rc /= len(L1.N_)?
                if NH: add_F(NH[0], L1, cr)
                else:  NH = [[L1]]
    Ft.N_ = ([CF(N_=F_,dTT=TT, m=m,d=d,c=C,root=root)] + NH) if NH else F_  # add top lev
    if fset:
        setattr(root, nF,Ft); root_update(root, Ft)
    return Ft

def add_N(G, N, coef=1):  # sum not-CF vars only, add Fts if merge?

    N.fin = 1; N.root = G
    _c,c = G.c,N.c*coef; C=_c+c  # weigh contribution of intensive params
    if hasattr(G, 'm_'): G.rc = np.sum([o*coef for o in N.o_]); G.rN_+=N.rN_; G.m_+=N.m_; G.o_+=N.o_  # not sure
    if N.typ:  # not PP| Cent
        if N.C_: G.C_+= N.C_; G.Ct.dTT += N.Ct.dTT*coef; G.Ct.c += N.Ct.c*coef  # flat? L_,B_ stay nested
        G.span = (G.span*_c+N.span*c*coef) / C * coef
        A,a = G.angl[0],N.angl[0]; A[:] = (A*_c+a*c*coef) /C * coef  # vect only
        if isinstance(G.yx, list): G.yx += [N.yx]  # weigh by C?
    if N.typ>1: # nodes
        G.baseT = (G.baseT*_c+ N.baseT*c*coef) /C
        G.mang = (G.mang*_c + N.mang*c*coef) /C
        G.box = extend_box(G.box, N.box)
    # if N is Fg: margin = Ns of proj max comp dist > min _Fg point dist: cross_comp Fg_?
    return N

def cent_TT(C, rc, init=0):  # weight attr matches | diffs by their match to the sum, recompute to convergence

    wTT = []  # Cs can be fuzzy only to the extent that their correlation weights are different?
    tot = C.dTT[0] + np.abs(C.dTT[1])  # m_* align, d_* 2-align for comp only?

    for fd, derT, wT in zip((0,1), C.dTT, wTTf):
        if fd: derT = np.abs(derT)  # ds
        _w_ = np.ones(9)  # weigh by feedback:
        val_ = derT / tot * wT  # signed ms, abs ds
        V = np.sum(val_)
        while True:
            mean = max(V / max(np.sum(_w_),eps), eps)
            inverse_dev_ = np.minimum(val_/mean, mean/val_)  # rational deviation from mean rm in range 0:1, if m=mean
            w_ = inverse_dev_/.5  # 2/ m=mean, 0/ inf max/min, 1 / mid_rng | ave_dev?
            w_ *= 9 / np.sum(w_)  # mean w = 1, M shouldn't change?
            if np.sum(np.abs(w_-_w_)) > ave*rc:
                V = np.sum(val_ * w_)
                _w_ = w_
            else: break  # weight convergence
        wTT += [_w_]
    C.wTT = np.array(wTT)  # replace wTTf
    # single-mode dTT, extend to 2D-3D lev cycles in H, cross-level param max / centroid?
    return C

def mdecay(L_):  # slope function
    L_ = sorted(L_, key=lambda l: l.span)
    dm_ = np.diff([l.m/l.c for l in L_])
    ddist_ = np.diff([l.span for l in L_])
    return -(dm_ / (ddist_ + eps)).mean()  # -dm/ddist

def root_update(root, T):  # value attrs only?

    _c,c = root.c,T.c; C = _c+c; root.c = C  # c is not weighted, min(_lev.c,lev.c) if root is link?
    root.rc = (root.rc*_c + T.rc*c) / C
    if isinstance(root,CF) or T.nF=='Nt' or T.nF=='Lt':  # core forks
        root.dTT = (root.dTT*_c + T.dTT*c) /C
    else:  # borrow alt-fork deviations:
        root.m = (root.m*_c+T.m*c) /C; root.d = (root.d*_c+T.d*c) /C
    if root.root: root_update(root.root, T)   # upward recursion, batch in root?

def root_replace(root, rc, TT,C, N_,nTT,nc,L_,lTT,lc):

    root.dTT=TT; root.c=C; root.rc = rc  # not sure
    if hasattr(root,'wTT'): cent_TT(root, root.rc)
    sum2F(N_,'Nt',root, nTT,nc)
    sum2F(L_,'Lt',root, lTT,lc)

def CopyT(F, root=None, cr=1):  # F = CF|CN

    C = CF(dTT=F.dTT * cr, root=root or F.root)
    for nF in ['Nt','Bt','Ct']:
        if sub_F:=getattr(F,nF):
            if not sub_F.N_: continue
            if isinstance(sub_F.N_[0],CN): C.N_ = copy(F.N_)
            else: C.N_ = [CopyT(lev,cr) for lev in F.N_]
    return C

def Copy_(N, root=None, init=0, typ=None):

    if typ is None: typ = 2 if init<2 else N.typ  # G.typ = 2, C.typ=0
    C = CN(dTT=deepcopy(N.dTT),typ=typ); C.root = N.root if root is None else root
    for attr in ['m','d','c','rc']: setattr(C,attr, getattr(N,attr))
    if not init and C.typ==N.typ: C.Nt = CopyT(N.Nt,root=C)
    if typ:
        for attr in ['fin','span','mang','sub','exe']: setattr(C,attr, getattr(N,attr))
        for attr in ['nt','baseT','box','rim','compared']: setattr(C,attr, copy(getattr(N,attr)))
        if init:  # new G
            C.rim = []; C.em = C.ed = 0
            C.yx = [N.yx]; C.angl = np.array([copy(N.angl[0]), N.angl[1]],dtype=object)  # to get mean
            if init==1:  # else centroid
                C.L_=[l for l in N.rim if l.m>ave]; N.root=C; C.fin = 0
        else:
            C.Lt=CopyT(N.Lt); C.Bt=CopyT(N.Bt); C.Ct=CopyT(N.Ct)  # empty in init G
            C.angl = copy(N.angl); C.yx = copy(N.yx)
        if typ > 1:
            C.eTT = deepcopy(N.eTT); C.em,C.ed,C.ec = N.em,N.ed,N.ec
    if init==2:
        for n in C.N_: n.m_=[]; n._m_=[]; n.o_=[]; n._o_=[]
    return C

def extend_box(_box, box):
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return min(y0,_y0), min(x0,_x0), max(yn,_yn), max(xn,_xn)

def sort_H(H, fi):  # lev.rc = complementary to root.rc and priority index in H, if selective and aligned

    i_ = []  # priority indices
    for i, lay in enumerate(sorted(H.node_, key=lambda lay: [lay.m,lay.d][fi], reverse=True)):
        di = lay.i - i  # lay index in H
        lay.rc += di  # derR - valR
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

def ffeedback(root):  # adjust filters: all aves *= rV, ultimately differential backprop per ave?

    def L_ders(Fg):  # get current-level ders: from L_ only
        dTT = np.zeros((2,9)); m, d, c = 0, 0, 0
        for n in Fg.N_:
            for l in n.L_: m += l.m; d += l.d; c += l.c; dTT += l.dTT
        return m,d,c,dTT

    wTTf = np.ones((2,9))  # sum dTT weights: m_,d_ [M,D,n, I,G,A, L,S,ext_A]: Et, baseT, extT
    rM, rD, rVd = 1,1,0
    _m, _d, _n, _dTT = L_ders(root)
    for lev in root.Nt.N_:  # top-down, not lev-selective, not recursive
        m ,d ,n, dTT = L_ders(lev)
        rM += (_m / _n) / (m / n)  # mat,dif change per level
        rD += (_d / _n) / (d / n)
        wTTf += np.abs((_dTT /_n) / (dTT / n))
        _m, _d, _n, _dTT = m, d, n, dTT
    return rM+rD+rVd, wTTf

def proj_focus(PV__, y,x, Fg):  # radial accum of projected focus value in PV__

    m,d,n = Fg.m, Fg.d, Fg.c
    V = (m-ave*n) + (d-avd*n)
    dy,dx = Fg.angl[0]; a = dy / max(dx,eps)  # average link_ orientation, projection
    Dec = decay * (wYX / np.hypot(y-Fg.yx[0], x-Fg.yx[1]))  # unit_decay * rel_dist?
    H, W = PV__.shape  # = win__
    n = 1  # radial distance
    while y-n>=0 and x-n>=0 and y+n<H and x+n<W:  # rim is within frame
        dec = Dec * n
        pV__= np.array([
        V * dec * 1.4, V * dec * a, V * dec * 1.4,  # a = aspect = dy/dx, affects axial directions only
        V * dec / a,                V * dec / a,
        V * dec * 1.4, V * dec * a, V * dec * 1.4
        ], dtype=float)
        if np.sum(pV__) < ave * 8:
            break  # < min adjustment
        rim_coords = np.array([
        (y-n,x-n), (y-n,x), (y-n,x+n),
        (y, x-n),           (y, x+n),
        (y+n,x-n), (y+n,x), (y+n,x+n)
        ], dtype=int)
        row,col = rim_coords[:,0], rim_coords[:,1]
        PV__[row,col] += pV__  # in-place accum pV to rim
        n += 1

def proj_TT(L, cos_d, dist, rc, pTT, wTT, fdec=0, frec=0):  # accumulate link pTT with iTT or eTT internally

    dec = dist if fdec else ave ** (1 + dist / L.span)  # ave: match decay rate / unit distance
    TT = np.array([L.dTT[0] * dec, L.dTT[1] * cos_d * dec])
    cert = abs(val_(TT,rc,wTT) - ave)  # approximation
    if cert > ave:
        pTT+=TT; return  # certainty margin = ave
    if not frec:  # non-recursive
        for lev in L.Nt.N_:  # refine pTT
            proj_TT(lev, cos_d, dec, rc+1, pTT, wTT, fdec=1, frec=1)
    pTT += TT  # L.dTT is redundant to H, neither is redundant to Bt,Ct
    for TT in [L.Bt.dTT if L.Bt else None, L.Ct.dTT if L.Ct else None]:  # + trans-link tNt, tBt, tCt?
        if TT is not None:
            pTT += np.array([TT[0] * dec, TT[1] * cos_d * dec])

def proj_N(N, dist, A, rc):  # arg rc += N.rc+connw, recursively specify N projection val, add pN if comp_pN?

    cos_d = (N.angl[0].dot(A) / (np.hypot(*N.angl[0]) * dist + eps)) * N.angl[1]  # internal x external angle alignment
    iTT, eTT = np.zeros((2,9)), np.zeros((2,9))
    wTT = TTw(N)
    for L in N.L_+N.B_: proj_TT(L, cos_d, dist, L.rc+rc, iTT, wTT)  # accum TT internally
    for L in N.rim:     proj_TT(L, cos_d, dist, L.rc+rc, eTT, wTT)
    pTT = iTT + eTT  # projected int,ext links, work the same?

    return pTT  # val_(N.dTT,rc) * (1- val_(iTT+eTT, rc))  # info_gain = N.m * average link uncertainty, should be separate
'''
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

def vect_edge(tile, rV=1, wTT=None):  # PP_ cross_comp and floodfill to init focal frame graph, no recursion:

    global ave,avd, aveB, Lw,distw,intw, nw,centw,connw,specw, wTTf  # /= projected V change:
    if wTT is None: wTTf /= rV
    else:           wTTf = wTT  # detailed feedback
    ave,avd,aveB,Lw,distw,intw,nw,centw,connw,specw = (np.array([ave,avd,aveB,Lw,distw,intw,nw,centw,connw,specw]) / rV)
    def PP2N(PP):
        P_,L_,B_,verT,latT,A,S,box,yx, m,d,c = PP
        baseT = np.array(latT[:4])
        [mM, mD, mI, mG, mA, mL], [dM, dD, dI, dG, dA, dL] = verT  # re-pack in dTT:
        dTT = np.array([ np.array([mM, mD, mL, mI, mG, mA, mL, mL / 2, eps]),  # extA=eps
                         np.array([dM, dD, dL, dI, dG, dA, dL, dL / 2, eps])])
        y,x,Y,X = box; dy,dx = Y+1-y, X+1-x
        A = np.array([np.array(A), np.sign(dTT[1] @ wTTf[1])], dtype=object)  # append sign
        PP = CN(typ=0, N_=P_,L_=L_,B_=B_,dTT=dTT,m=m,d=d,c=c, baseT=baseT,box=box,yx=yx,angl=A,span=np.hypot(dy/2,dx/2))  # set root in trace_edge
        PP.Lt = CF(N_=L_, m=m,d=d,c=c,rc=2, root=PP,nF='Lt')  # PP.Nt = CF(dTT=deepcopy(dTT), m=m,d=d,c=c,rc=2, root=PP,nF='Nt')
        m_, d_ = np.zeros(6), np.zeros(6)
        for B in B_: m_ += B.verT[0]; d_ += B.verT[1];
        ad_ = np.abs(d_); t_ = m_ + ad_ + eps  # ~ max comparand
        m = m_/t_ @ w_t[0] - ave*2; d = ad_/t_ @ w_t[1] - avd*2
        PP.Bt = CF(N_=B_, m=m, d=d, rc=2, root=PP,nF='Bt')
        for P in P_: P.root = PP
        if hasattr(P,'nt'):  # PPd, assign rN_:
            for dP in P_:
                for P in dP.nt: PP.rN_ += [P.root]  # PPm
        return PP
    # TT,C, G_,nTT,nC,L_,lTT,lC in trace_edge:
    tT = [np.zeros((2,9)),0, [],np.zeros((2,9)),0, [],np.zeros((2,9)),0]
    blob_, G_ = tile.N_, []
    for blob in blob_:
        if not blob.sign and blob.G > aveB:
            edge = slice_edge(blob, rV)
            if edge.G * ((len(edge.P_)-1)*Lw) > ave * sum([P.latT[4] for P in edge.P_]):
                PPm_ = comp_slice(edge, rV, wTTf)
                nG_ = [PP2N(PPm) for PPm in PPm_]
                for PPd in edge.link_: PP2N(PPd)
                for nG in nG_:
                    if nG.B_: sum2F([B.root for B in nG.B_],'Bt',nG,1)
                if val_(np.sum([n.dTT for n in nG_],0),3, TTw(tile), (len(PPm_)-1)*Lw) > 0:
                    G_, rc = trace_edge(nG_,3, tile, tT)  # flatten, cluster B_-mediated Gs, init Nt (we need to return G_)?
    if G_:
        root_replace(tile,1, *tT)  # updates tile.wTT
        if vt_(tile.dTT)[0] > ave:
            return tile

def trace_edge(N_, rc, root, tT=[]):  # cluster contiguous shapes via PPs in edge blobs or lGs in boundary/skeleton?

    L_, cT_, lTT, lc = [],set(),np.zeros((2,9)),0  # comp co-mediated Ns:
    for N in N_: N.fin = 0
    for N in N_:
        _N_ = [rN for B in N.B_ for rN in B.rN_ if rN is not N]   # + node-mediated
        for _N in list(set(_N_)):  # share boundary or cores if lG with N, same val?
            cT = tuple(sorted((N.id,_N.id)))
            if cT in cT_: continue
            cT_.add(cT)
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx); Rc = rc+ (N.rc+_N.rc)/2
            Link = comp_N(_N,N, Rc, A=dy_dx, span=dist)
            if vt_(Link.dTT)[0] >ave*Rc:
                L_+=[Link]; lTT+=Link.dTT; lc+=Link.c

    Gt_,TT,C = [],np.zeros((2,9)),0
    for N in N_:  # flood-fill G per seed N
        if N.fin: continue
        N.fin=1; _N_=[N]; Gt=[]; N.root=Gt; n_,ntt,nc, l_,ltt,lc = [N],np.zeros((2,9)),0,[],np.zeros((2,9)),0  # Gt
        while _N_:
            _N = _N_.pop(0)
            for L in _N.rim:
                if L in L_:
                    n = L.nt[0] if L.nt[1] is _N else L.nt[1]
                    if n in N_:
                        if n.root is Gt: continue
                        if n.fin:  # merge n root
                            _root = n.root; n_+=_root[0]; ntt+=_root[1]; nc+=_root[2]; l_+=_root[3]; ltt+=_root[4]; lc+=_root[5]; _root[6]=1
                            for _n in _root[0]: _n.root = Gt
                        else:
                            n.fin=1; _N_+=[n]; n_+=[n];ntt+=n.dTT;nc+=n.c; l_+=[L];ltt+=L.dTT;lc+=L.c  # add single n
                        n.root = Gt
        Gt += [n_,ntt,nc, l_,ltt,lc, 0]; Gt_+=[Gt]; TT+=ntt+ltt; C+=nc+lc
    # totals:
    G_,nTT,nC, L_,lTT,lC = [],np.zeros((2,9)),0, [],np.zeros((2,9)),0
    for n_,ntt,nc,l_,ltt,lc,merged in Gt_:
        if not merged:
            if vt_(TT,rc)[0] > ave*rc:  # include singletons
                G_ += [sum2G(((n_,ntt,nc),(l_,ltt,lc)), TT,C,rc,root,fsub=0)]; nTT+=ntt; nC+=nc
                L_ += l_; lTT+=ltt; lC+=lc
            else:
                for N in n_: N.fin=0; N.root=root
    if val_(TT, rc+1, TTw(root), mw=1 if tT else (len(G_)-1)*Lw) > 0:
        if tT:  # concat in vect_edge
            for i, par in enumerate((TT,C,G_,nTT,nC,L_,lTT,lC)): tT[i] += par
        else: root_replace(root, rc, TT,C,G_,nTT,nC,L_,lTT,lC)
        rc += 1
    return G_, rc

# frame expansion per level: cross_comp lower-window N_,C_, forward results to next lev, project feedback to scan new lower windows

def frame_H(image, iY,iX, Ly,Lx, Y,X, rV, max_elev=4, wTTf=np.ones((2,9))):  # all initial args set manually

    def base_tile(y,x):  # 1st level, higher levels get Fg s
        Fg = frame_blobs_root( comp_pixel( image[y:y+Ly, x:x+Lx]), rV)
        Fg = vect_edge(Fg, rV)  # form, trace PP_
        if Fg: cross_comp(Fg, rc=Fg.rc)
        return Fg

    def expand_lev(_iy,_ix, elev, Fg):  # seed tile is pixels in 1st lev, or Fg in higher levs

        tile = np.full((Ly,Lx),None, dtype=object)  # exclude from PV__
        PV__ = np.zeros([Ly,Lx])  # maps to current-level tile
        Fg_ = []; iy,ix =_iy,_ix; y,x = 31,31  # start at tile mean
        while True:
            if not elev: Fg = base_tile(iy,ix)  # 1st level or previously cross_comped arg tile
            if Fg and val_(Fg.dTT, Fg.rc+nw+elev, Fg.wTT, mw=(len(Fg.N_)-1)*Lw) > 0:
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
        if Fg_ and val_(np.sum([g.dTT for g in Fg_],axis=0), np.mean([g.rc for g in Fg_])+elev, wTTf, mw=(len(Fg_)-1)*Lw) > 0:
            return Fg_

    global ave,avd, Lw, intw, cw,nw, centw,connw, distw, mW, dW
    ave,avd, Lw, intw, cw,nw, centw,connw, distw = np.array([ave,avd, Lw, intw, cw, nw, centw, connw, distw]) / rV

    frame = CN(box=np.array([0,0,Y,X]), yx=np.array([Y//2, X//2]))
    Fg=[]; elev=0
    while elev < max_elev:  # same center in all levels
        Fg_ = expand_lev(iY,iX, elev, Fg)
        if Fg_:  # higher-scope sparse tile
            frame = sum2G(Fg_, np.sum([fg.dTT for fg in Fg_],axis=0), sum([fg.c for fg in Fg_]), rc=1,root=frame,init=0)
            if Fg and cross_comp(Fg, rc=elev)[0]:  # or val_? spec->tN_,tC_,tL_
                frame.N_ = frame.N_+[Fg]; elev+=1  # forward comped tile
                if max_elev == 4:  # seed, not from expand_lev
                    rV,wTTf = ffeedback(Fg)  # set filters
                    Fg = cent_TT(Fg,2)  # set Fg.dTT correlation weights
                    wTTf *= Fg.wTT; mW = np.sum(wTTf[0]); dW = np.sum(wTTf[1])
                    wTTf[0] *= 9/(mW or eps); wTTf[1] *= 9/(dW or eps)
            else: break
        else: break
    return frame  # for intra-lev feedback

if __name__ == "__main__":  # './images/toucan_small.jpg' './images/raccoon_eye.jpeg', add larger global image

    Y,X = imread('./images/toucan.jpg').shape
    # frame = agg_frame(0, image=imread('./images/toucan.jpg'), iY=Y, iX=X)
    frame = frame_H(image=imread('./images/toucan.jpg'), iY=Y//2 -31, iX=X//2 -31, Ly=64,Lx=64, Y=Y, X=X, rV=1)
    # search frames ( tiles inside image, at this size it should be 4K, or 256K panorama, won't actually work on toucan