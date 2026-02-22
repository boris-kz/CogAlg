import numpy as np
from copy import copy, deepcopy
from math import atan2, cos, pi  # from functools import reduce
from itertools import zip_longest, combinations, product  # from multiprocessing import Pool, Manager
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
- centroid-parallel frame refine by two-layer EM if min global overlap, prune for next cros_comp cycle (cluster_P).

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

def prop_F_(F):  # factory function to get and update top-composition fork.N_
    def get(N): return getattr(N,F).N_
    def set(N, new_N): setattr(getattr(N,F),'N_', new_N)
    return property(get,set)

class CN(CBase):
    name = "node"
    N_,C_,B_ = prop_F_('Nt'),prop_F_('Ct'),prop_F_('Bt')  # ext|int- defined nodes,links, Lt/Ft, Ct/lev, Bt/G
    @property
    def L_(N):
        if isinstance(N.Nt.Lt, list): N.Nt.Lt =CF(root=N.Nt)
        return N.Nt.Lt.N_
    @L_.setter
    def L_(N,v):
        if isinstance(N.Nt.Lt, list): N.Nt.Lt =CF(root=N.Nt)
        N.Nt.Lt.N_ = v
    def __init__(n, **kwargs):
        super().__init__()
        n.typ = kwargs.get('typ', 0)
        # 0= PP: block trans_comp, etc?
        # 1= L:  typ,nt,dTT, m,d,c,r, root,rng,yx,box,span,angl,fin,compared, Nt,Bt,Ct from comp_sub
        # 2= G:  + rim,baseT,mang,sub,exe
        # 3= Cn: + m_,d_,r_,o_ in C.rN_
        n.m, n.d, n.c, n.r = [kwargs.get(x,0) for x in ('m','d','c','r')]  # combined forks
        n.dTT = kwargs.get('dTT',np.zeros((2,9)))  # Nt+Lt dTT: m_,d_ [M,D,n, I,G,a, L,S,A]
        n.rim = kwargs.get('rim',CF(root=n))  # external links, rng-nest?
        n.Nt, n.Bt, n.Ct, n.Lt = ((kwargs.get(f) if f in kwargs else CF(root=n) for f in ('Nt','Bt','Ct','Lt'))) # ->CN if nest, Ct||Nt
        n.nt  = kwargs.get('nt',[])  # nodet, links only
        n.baseT = kwargs.get('baseT',np.zeros(4))  # I,G,A: not ders, in links for simplicity, mostly redundant
        n.span  = kwargs.get('span',1) # distance in nodet or aRad, comp with baseT and len(N_), not additive?
        n.angl  = kwargs.get('angl',[np.zeros(2),0])  # (dy,dx),dir, sum from L_
        n.mang  = kwargs.get('mang',1)  # ave match of angles in L_, =1 in links
        n.box   = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.yx    = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.root= kwargs.get('root',None)  # immediate
        n.sub = 0  # composition depth relative to top-composition peers?
        n.fin = kwargs.get('fin',0)  # clustered, temporary
        n.exe = kwargs.get('exe',0)  # exemplar, temporary
        n.rN_ = kwargs.get('rN_',[]) # reciprocal root nG_ for bG|cG, nG has Bt.N_,Ct.N_ instead?
        n.compared = set()
        n.nF = kwargs.get('nF', 'Nt')  # to set attr in root_update
        n.fb_= kwargs.get('fb_', [])
        # ftree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def __bool__(n): return bool(n.c)

class CF(CBase):  # rim, Nt,Ct, Bt,Lt: ext|int- defined nodes, ext|int- defining links, Lt/Ft, Ct/lev, Bt/G
    name = "fork"
    def __init__(f, **kwargs):
        super().__init__()
        f.N_ = kwargs.get('N_',[])  # flat
        f.H  = kwargs.get('H', [])  # CF levs
        f.nF = kwargs.get('nF','Nt')
        f.Lt = kwargs.get('Lt',[])  # from Ft cross_comp, Ct is parallel to Nt
        f.dTT = kwargs.get('dTT',np.zeros((2,9)))
        f.m, f.d, f.c, f.r = [kwargs.get(x,0) for x in ('m','d','c','r')]
        f.root = kwargs.get('root',None)
        f.fb_ = kwargs.get('fb_',[])
    def __bool__(f): return bool(f.c)

ave = .3; avd = ave*.5  # ave m,d / unit dist, top of filter specification hierarchy
wM,wD,wc, wG,wI,wa, wL,wS,wA = 10, 10, 20, 20, 5, 20, 2, 1, 1  # dTT param root weights = reversed relative estimated ave
wT = np.array([wM,wD,wc, wG,wI,wa, wL,wS,wA]); wTTf = np.array([wT*ave, wT*avd])
aveB, distw, Lw, intw = 100, .5,.5,.5  # not-compared attr filters, *= ave|avd?
nw,cw, connw,centw, specw = 10,5, 15,20, 10  # process filters, *= (ave,avd)[fi], cost only now
mW = dW = 9  # fb weights per dTT, adjust in agg+
wY = wX = 64; wYX = np.hypot(wY,wX)  # focus dimensions
decay = ave / (ave+avd)  # match decay / unit dist?

def vt_(TT, r=0, wTT=None):  # brief val_ to get m,d, rc=0 to return raw vals

    m_,d_= TT; ad_ = np.abs(d_); t_ = m_ + ad_ + eps  # ~ max comparand
    if wTT is None: wTT = wTTf
    return m_/t_ @ wTT[0] - ave*r, ad_/t_ @ wTT[1] - avd*r  # norm by co-derived variance

def val_(TT, r, wTT, mw=1.0,fi=1, _TT=None,cr=.5):  # m,d eval per cluster, cr = cd / cm+dc, default half-weight?

    t_ = np.abs(TT[0]) + np.abs(TT[1])  #  combine vals, not sure about abs m_?
    rv = TT[0] / (t_+eps) @ wTT[0] if fi else TT[1] / (t_+eps) @ wTT[1]  # fork / total per scalar
    if _TT is not None:
        _t_ = np.abs(_TT[0]) + np.abs(_TT[1])
        _rv = _TT[0] / (_t_+eps) @ wTT[0] if fi else _TT[1] / (_t_+eps) @ wTT[1]
        rv  = rv * (1-cr) + _rv * cr  # + borrowed alt fork val, cr: d count ratio, must be passed with _TT?
    return rv*mw - (ave if fi else avd) * r

def TTw(G): return getattr(G,'wTT',wTTf)
''' 
  agg cycle:
- Cross-comp nodes, evaluate incremental-derivation cross-comp of new >ave difference links, recursively. 
- Select exemplars

- Connectivity-cluster select exemplars/centroids by >ave match links, correlation-cluster links by >ave diff,
- Form complemented (core+contour) clusters -> divisive sub-clustering, higher-composition cross_comp. 

- Centroid-cluster from exemplars, spreading via their rim within a frame, increasing overlap between centroids.
- Switch to globally parallel EM-like fuzzy centroid refining if overlap makes lateral expansion less efficient.

- Forward: extend cross-comp and clustering of top clusters across frames, re-order centroids by eigenvalues.
- Feedback coords to bottom level or prior-level in parallel pipelines, filter updates in more coarse cycles 
'''
def cross_comp(Ft, r, nF='Nt'):  # core function mediating recursive rng+ and der+ cross-comp and clustering

    N_,G_,root = Ft.N_,[],Ft.root  # rc=rdn+olp, comp N_|B_|C_: nodes| links| centroids:
    iN_,L_,TT,c, TTd,cd = comp_N_(N_,combinations(N_,2),r) if N_[0].typ else comp_C_(N_,r, fC=1)
    if L_:
        cr = cd / (c+cd) * .5  # dfork borrow ratio, .5 for one direction
        if val_(TT, r+connw,TTw(Ft),(len(L_)-1)*Lw,1,TTd,cr) > 0:
            # G.L_= Nt.Lt.N_, flat
            Lt = sum2f(L_,'Lt',Ft); root.c+=Lt.c; cr=Lt.c/root.c; root.dTT+=Lt.dTT*cr; root.r+=Lt.r*cr; root.Lt=Lt
            E_ = get_exemplars({N for L in L_ for N in L.nt if N.rim.m>ave}, r)  # N|C?
            G_,rc = cluster_N(Ft, E_,r)  # form Bt, trans_cluster, sub+ in sum2G
            if G_:
                Ft = sum2F(G_, nF, Ft.root)
                if val_(Ft.dTT, r+nw, TTw(Ft), (len(N_)-1)*Lw,1, TTd,cr) > 0:
                    G_,r = cross_comp(Ft,r,nF)  # agg+,trans-comp
    return G_,rc  # G_ is recursion flag?

def comp_N_(iN_, pairs, r, tnF=None, rL=None):  # incremental-distance cross_comp, max dist depends on prior match

    N_,L_,TTm,cm,TTd,cd,R = [],[],np.zeros((2,9)),0,np.zeros((2,9)),0,0; dpTT=np.zeros((2,9))
    for N in iN_: N.pL_ = []  # init
    for _N, N in pairs:  # get all-to-all pre-links
        if _N.sub != N.sub: continue  # or comp x composition?
        if N is _N: dtt= np.array([N.dTT[1],np.zeros(9)]); TTm+=dtt; cm+=1  # overlap = unit match, no miss
        else:
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx)
            N.pL_+= [[dist,dy_dx,_N]]; _N.pL_+= [[dist,-dy_dx,N]]
        N.pL_.sort(key=lambda x: x[0])  # proximity prior, test compared?

    def proj_V(_N,N, dist, pVt_, dec):  # _N x N induction
        Dec = dec or decay ** ((dist/((_N.span+N.span)/2)))
        iTT = (_N.dTT + N.dTT) * Dec
        eTT = (_N.rim.dTT + N.rim.dTT) * Dec
        if abs( vt_(eTT,r)[0]) * ((len(pVt_)-1)*Lw) > ave*specw:  # spec N links
            eTT = np.zeros((2,9)) # recompute
            for _dist,_dy_dx,__N,_V in pVt_:
                eTT += proj_N(N,_dist, _dy_dx, r, dec)  # proj N L_,B_,rim, if pV>0: eTT += pTT?
                eTT += proj_N(_N,_dist, -_dy_dx, r, dec)  # reverse direction
        return iTT+eTT

    for N in iN_:
        pVt_ = []  # [[dist, dy_dx, _N, V]]
        for dist, dy_dx, _N in N.pL_:  # rim angl is not canonic
            pTT = proj_V(_N,N, dist, pVt_, rL.m if rL else decay** (dist/((_N.span+N.span)/2)))
            lr = r+ (N.r+_N.r) / 2; m,d = vt_(pTT,lr)  # +|-match certainty
            if m > 0:
                if abs(m) < ave * nw:  # different ave for projected surprise value, comp in marginal predictability
                    Link = comp_N(_N,N, lr, full=not tnF, A=dy_dx, span=dist, rL=rL)
                    dTT, m,d,c,r = Link.dTT,Link.m,Link.d,Link.c,Link.r
                    if   m > ave: TTm+=dTT*c; cm+=c; R+=r*c; L_+=[Link]; N_+=[_N,N]
                    elif d > avd: TTd+=dTT*c; cd+=c  # no overlap to simplify
                    dpTT += pTT-dTT  # prediction error to fit code, not implemented
                else:
                    pL = CN(typ=-1, nt=[_N,N], dTT=pTT,m=m,d=d,c=min(N.c,_N.c), r=lr, angl=np.array([dy_dx,1],dtype=object),span=dist)
                    L_+= [pL]; N.rim+=[pL]; N_+=pL.nt; _N.rim+=[pL]; TTm+=pTT; cm+=pL.c  # same as links in clustering, L.N_=nt?
                pVt_ += [[dist,dy_dx,_N,m]]  # for next rim eval
            else: break  # beyond induction range
        if N.rim: sum2f(N.rim,'rim',N)
    return list(set(N_)), L_,TTm,cm,TTd,cd  # + dpTT for code-fitting backprop?

def comp_N(_N,N, r, full=1, A=np.zeros(2),span=None, rL=None):

    def comp_H(_Nt,Nt, Link):  # tentative pre-comp

        dH, tt,C,R = [],np.zeros((2,9)),0,0
        for _lev, lev in zip([_Nt]+_Nt.H, [Nt]+Nt.H):  # should be top-down
            ltt = comp_derT(_lev.dTT[1],lev.dTT[1])
            lc = min(_lev.c,lev.c); lr = (_lev.r+lev.r)/2; m,d = vt_(ltt,lr)
            dH += [CF(dTT=ltt,m=m,d=d,c=lc,rc=lr,root=Link)]
            tt += ltt; C+=lc; R+=lr
        return dH,tt,C, r+ R/len(dH) * (C/Link.c)  # same norm for tt?

    TT= base_comp(_N,N)[0] if full else comp_derT(_N.dTT[1],N.dTT[1]); m,d = vt_(TT,r)
    L = CN(typ=1, nt=[_N,N], dTT=TT, m=m, d=d, c=min(N.c,_N.c), r=r, root=rL, exe=1)
    if N.typ and m > ave*nw:
        dH,htt,C,R = comp_H(_N.Nt, N.Nt, L); r=R  # tentative subcomp
        if m + vt_(htt,R/C)[0] > ave*nw:  # subcomp -> tFs in L:
            if abs(N.typ)==1:
                for n,_n in product(_N.nt,N.nt): L.Nt.fb_ += [comp_N(n,_n,r)]  # link subcomp
            else:
                for _Ft, Ft, tnF in zip((_N.Nt,_N.Ct,_N.Bt), (N.Nt,N.Ct,N.Bt), ('Nt','Ct','Bt')):
                    if _Ft and Ft:  # fork sub-comp, add eval?
                        r+=1; comp_N_(_Ft.N_+Ft.N_, product(_Ft.N_,Ft.N_), r,tnF,L)
            for ft_, nF in zip((L.Nt.fb_, L.Bt.fb_, L.Ct.fb_), ('Nt','Bt','Ct')):
                if ft_: sum2f(ft_,nF,r)  # python-batched bottom-up
            Nt, Bt, Ct = L.Nt,L.Bt,L.Ct  # trans-links formed above
            Ft = CopyF(Nt)
            Ft.N_ = [Nt,Bt,Ct]; Ft.nF='tFt'; Ft.root=L  # sum as in sum2G, set fork r:
            if Bt: add_F(Ft,Bt, r=Nt.m>Bt.m)
            if Ct: add_F(Ft,Ct, r=Nt.m>Ct.m)
            L.Nt = Ft  # tFt, no regular L.Nt yet, use comb_Ft instead?
    if full:
        if span is None: span = np.hypot(*_N.yx - N.yx)
        yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx
        box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
        angl = [A, np.sign(TT[1] @ wTTf[1])]
        L.yx=yx; L.box=box; L.span=span; L.angl=angl; L.baseT=(_N.baseT+N.baseT)/2
    for n, _n in (_N,N),(N,_N):  # no links between tFts
        n.rim+=[L]; n.compared.add(_n)  # or unique comps
    return L

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
    N_,L_,TTm,cm = [],[],np.zeros((2,9)),0
    if fall:
        pairs = product(C_,_C_) if _C_ else combinations(C_,r=2)  # comp between | within list
        for _C, C in pairs:
            if _C is C: dtt = np.array([C.dTT[1],np.zeros(9)]); TTm+=dtt; cm=1  # overlap=match
            else:
                dtt = base_comp(_C,C)[0]; m,d = vt_(dtt,rc)  # or comp_n, packing L_?
                dC = CN(nt=[_C,C],dTT=dtt,m=m,d=d,c=_C.c+C.c, span=np.hypot(*_C.yx-C.yx))
                L_+= [dC]; _C.rim += [dC]; C.rim += [dC]
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
            L_ += [comp_N(_C,C, rc, A=dy_dx,span=dist)]  # or comp_derT?
    mL_=[]
    TTd,cd = np.zeros((2,9)),0
    for L in L_:
        if L.m > ave* (connw+rc): mL_+= [L]; N_+= L.nt
    if mL_: _,_,tt,c,_ = sum_vt(mL_); TTm+= tt*c; cm+=c  # sum r?
    return list(set(N_)), mL_,TTm,cm,TTd,cd

def get_exemplars(N_,r):  # multi-layer non-maximum suppression -> sparse seeds for diffusive clustering, cluster_N too?
    E_ = set()
    for rdn,N in enumerate(sorted(N_, key=lambda n:n.em, reverse=True), start=1):  # strong-first
        oL_ = set(N.rim.N_) & {l for e in E_ for l in e.rim.N_}
        roV = (sum_vt(list(oL_), r)[0] if oL_ else 0) / (N.rim.m or eps)  # relative rim olp V
        if N.rim.m * N.c > ave * (r+rdn+nw+ roV):  # ave *= rV of overlap by stronger-E inhibition zones
            E_.update({n for l in N.rim for n in l.nt if n is not N and N.em > ave*r})  # selective nrim
            N.exe = 1  # in point cloud of focal nodes
        else:
            break  # the rest of N_ is weaker, trace via rims
    return E_

def cluster_N(Ft, _N_, r):  # flood-fill node | link clusters, flat, replace iL_ with E_?

    def nt_vt(n,_n):
        M, D = 0,0  # exclusive match, contrast
        for l in set(n.rim+_n.rim):
            if l.m > 0:   M += l.m
            elif l.d > 0: D += l.d
        return M, D

    def trans_cluster(G):
        for L in G.L_:
            for tFt in L.Nt.N_:  # tNt, tBt, tCt
                for tL in tFt.N_:
                    if tL.m > ave*connw:  # merge trans_link.nt.roots
                        rt0 = tL.nt[0].root.root; rt1 = tL.nt[1].root.root  # CNs
                        if rt0 != rt1: add_N(rt0, rt1, merge=1)  # concat in higher G
            L.Nt=CF()   # reset
            ''' 
            reset rc:
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
    G_ = []  # add prelink pL_,pN_? include merged Cs, in feature space for Cs
    if _N_ and val_(Ft.dTT, r+connw, TTw(Ft), mw=(len(_N_)-1)*Lw) > 0:  #| fL?
        for N in _N_: N.fin=0; N.exe=1  # not sure
        G_= []; TT=np.zeros((2,9)); C=0; in_= set()  # root attrs
        for N in _N_:  # form G per remaining N
            if N.fin or (Ft.root.root and not N.exe): continue  # no exemplars in Fg
            N_ = [N]; L_,B_,C_ = [],[],[]; N.fin=1  # init G
            __L_= N.rim  # spliced rim
            while __L_:
                _L_ = []
                for L in __L_:  # flood-fill via frontier links
                    _N = L.nt[0] if L.nt[1].fin else L.nt[1]; in_.add(L)
                    if not _N.fin and _N in _N_:
                        m,d = nt_vt(*L.nt)
                        if m > ave * (r-1):  # cluster nt, L,C_ by combined rim density:
                            N_ += [_N]; _N.fin = 1
                            L_ += [L]; C_ += _N.rN_
                            _L_+= [l for l in _N.rim if l not in in_ and (l.nt[0].fin ^ l.nt[1].fin)]   # new frontier links, +|-?
                        elif d > avd * (r-1): B_ += [L]  # contrast value, exclusive?
                __L_ = list(set(_L_))
            if N_:
                ft_ = []
                for i, (F_, nF) in enumerate(zip((N_, L_, B_), ('Nt', 'Lt', 'Bt'))):
                    F_ = list(set(F_)) or []
                    _,_,tt,fc,fr = sum_vt(F_) if F_ else (0,0,np.zeros((2,9)),0,0)
                    ft_ += [[F_,nF, tt,fc,fr]]
                (_,_,nt,nc,_),(_,_,lt,lc,_),(_,_,bt,bc,br) = ft_
                c = nc + lc + bc*br  # B_ and C_ are redundant
                tt = (nt*nc + lt*lc + bt*bc*br) / c
                if val_(tt,r, TTw(Ft), (len(N_)-1)*Lw) > 0:
                    G_ += [sum2G(ft_,tt,c,r,Ft)]; TT+=tt; C+=c  # tt*cr? sub+/sum2G
        if G_:
            for G in G_: trans_cluster(G)  # splice trans_links, merge L.nt.roots
            if val_(TT, r+1, TTw(Ft), (len(G_)-1)*Lw):
                sum2F(G_,Ft.nF, Ft.root,TT,C); r+=1  # sub+, sum Rc? Ft.Lt is empty till cross_comp
    return G_, r

def cluster_C(Ft, E_, r):  # form centroids by clustering exemplar surround via rims of new member nodes, within root

    C_,_C_ = [],[]  # form root.Ct, may call cross_comp-> cluster_N, incr rc
    for n in Ft.N_: n._C_,n.m_,n._m_,n.o_,n._o_,n.rN_ = [],[],[],[],[],[]
    for E in E_:
        C = cent_TT(Copy_(E, Ft, init=2,typ=0), r)  # all rims are in root, sequence along eigenvector?
        C._N_ = list({n for l in E.rim for n in l.nt if (n is not E and n in Ft.N_)})  # init C.N_=[]
        C._L_ = set(E.rim)  # init peer links
        for n in C._N_: n.m_+=[C.m]; n._m_+=[C.m]; n.o_+=[1]; n._o_+=[1]; n.rN_+=[C]; n._C_+=[C]
        _C_ += [C]
    while True:  # reform C_, add direct in-C_ cross-links for membership?
        C_,cnt,olp, mat,dif, DTT,Dm,Do = [],0,0,0,0,np.zeros((2,9)),0,eps; Ave = ave * (r+nw)
        _Ct_ = [[c, c.m/c.c if c.m !=0 else eps, c.r] for c in _C_]
        for r, (_C,_m,_o) in enumerate(sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True),start=1):
            if _m > Ave *_o:
                L_, N_,N__,m_,o_,M,D,O,cc, dTT,dm,do = [],[],[],[],[],0,0,0,0,np.zeros((2,9)),0,0  # /C
                for n in _C._N_:  # frontier
                    dtt,_ = base_comp(_C, n); cc+=1  # add decay / dist?
                    m,d = vt_(dtt,r); dTT += dtt; nm = m * n.c  # rm,olp / C
                    odm = np.sum([_m-nm for _m in n._m_ if _m>m])  # higher-m overlap
                    oL_ = set(n.rim) & _C._L_  # replace peer rim overlap with more precise m
                    if oL_: m += sum_vt(oL_, r, m)[0]  # add deviation from redundant mC, add eval?
                    if m > 0 and nm > Ave * odm:
                        N_+=[n]; L_+=n.L_; M+=m; O+=odm; m_+=[nm]; o_+=[odm]  # n.o for convergence eval
                        N__ += [_n for l in n.rim.N_ for _n in l.nt if _n is not n]  # +|-Ls
                        if _C not in n._C_: dm+=m; do+=odm  # not in extended _N__
                    else:
                        if _C in n._C_: i = n._C_.index(_C); dm+=n._m_[i]; do+=n._o_[i]
                        D += abs(d)  # distinctive from excluded nodes (background)
                mat+=M; dif+=D; olp+=O; cnt+=cc  # all comps are selective
                DTT += dTT
                if M > Ave * len(N_) * O and val_(dTT, r+O, TTw(C),(len(N_)-1)*Lw):
                    C = cent_TT(sum2C(N_,_C,_Ci=None), r=r)
                    for n,m,o in zip(N_,m_,o_): n.m_+=[m]; n.o_+=[o]; n.rN_+= [C]; C.rN_+= [n] # reciprocal root assign
                    C._N_ = list(set(N__))  # frontier
                    C._L_ = set(L_)  # peer links
                    C_ += [C]; Dm+=dm; Do+=do  # new incl or excl
                else:
                    for n in _C._N_:
                        n.exe = n.m/n.c > 2 * ave  # refine exe
                        for i, c in enumerate(n.rN_):
                            if c is _C:  # remove _C-mapping m,o:
                                n.rN_.pop(i); n.m_.pop(i);n.o_.pop(i); break
            else:  # the rest is weaker
                break
        for n in Ft.N_:
            n._C_ = n.rN_; n._m_= n.m_; n._o_= n.o_; n.rN_,n.m_,n.o_ = [],[],[]  # new n.Ct.N_s, combine with v_ in Ct_?
        if mat * dif * olp > ave * centw*2:
            C_ = cluster_P(C_, Ft.N_, r)  # refine all memberships in parallel by global backprop|EM
            break
        if Dm / Do > Ave:  # dval vs. dolp: overlap increases with Cs expansion
            _C_ = C_
        else: break  # converged
    C_ = [C for C in C_ if val_(C.dTT, r, TTw(C))]  # prune C_
    if C_:
        for n in [N for C in C_ for N in C.N_]:  # exemplar V + sum n match_dev to Cs, m* ||C rvals:
            n.exe = (n.d if n.typ==1 else n.m) + np.sum([m-ave*o for m, o in zip(n.m_, n.o_)]) - ave
        if val_(DTT, r+olp, TTw(Ft), (len(C_)-1)*Lw) > 0:
            root = Ft.root
            Ct = sum2F(C_,'Ct',root, fCF=0)
            cross_comp(Ct,r)  # all distant Cs, seq C_ in eigenvector = argmax(root.wTT)? Nt|Ct priority eval?
    return C_, r

def cluster_P(_C_,N_,r):  # Parallel centroid refining, _C_ from cluster_C, N_= root.N_, if global val*overlap > min
                           # cent cluster L_ ~ GNN?
    for C in _C_:  # fixed while training, then prune if weak C.v
        C.rN_= N_  # soft assign all Ns per C
        C.m,C.d,C.rc = C.em,C.ed, np.mean([l.rc for l in C.rim])  # not sure
    while True:
        M = O = dM = dO = 0
        for N in N_:
            N.m_,N.d_,N.r_ = map(list,zip(*[vt_(base_comp(C,N)[0]) + (C.rc,) for C in _C_]))  # distance-weight match?
            N.o_= np.argsort(np.argsort(N.m_)[::-1]) + 1  # rank of each C_[i] = rdn of C in C_
            dM += sum(abs(_m-m) for _m,m in zip(N._m_,N.m_))
            dO += sum(abs(_o-o) for _o,o in zip(N._o_,N.o_))
            M += sum(N.m_)
            O += sum(N.m_)
        C_ = [sum2C(N_,_C, i) for i, _C in enumerate(_C_)]  # update with new coefs, _C.m, N->C refs
        Ave = ave * (r+centw)
        if M > Ave*O and dM > Ave*dO:  # strong update
            _C_ = C_
        else: break
    _C_,in_ = [], []
    for i, C in enumerate(C_):
        if C.m > ave * centw * C.r:
            C.N_ = [n for n in C.N_ if n._m_[i] * C.m > ave * n._r_[i] * n._o_[i]]; _C_ += [C]; in_ += [1]
        else: in_ += [0]
    for N in N_:
        for _v,v_ in zip((N._m_,N._d_,N._r_), (N.m_,N.d_,N.r_)):
            v_[:] =[v for v,i in zip(v_,in_) if i]; _v[:] = []
    return _C_
''' next order: level-parallel cluster_H / multiple agg+? compress as autoencoder? '''

def sum2C(N_,_C, _Ci=None):  # fuzzy sum params used in base_comp

    c_,r_,dTT_,baseT_,span_,yx_ = zip(*[(n.c, n.r, n.dTT, n.baseT, n.span, n.yx) for n in N_])
    ccoef_ = []
    for N in N_:
        Ci = N.rN_.index(_C) if _Ci is None else _Ci
        ccoef_ += [N._m_[Ci] / (ave* N._o_[Ci]) * _C.m]  # *_C.m: proj survival, vs post-prune eval?
        if _Ci is not None: N._m_,N._d_,N._r_ = N.m_,N.d_,N.r_  # from cluster_P
    c_ = [c * max(cc,0) for c,cc in zip(c_,ccoef_)]
    tot = sum(c_)+eps; Par_ = []
    for par_ in r_, dTT_, baseT_, span_, yx_:
        Par_.append(sum([p * c for p,c in zip(par_,c_)]))
    C = CN(c=tot, typ=0)
    C.rc, C.dTT, C.baseT, C.span, C.yx = [P / tot for P in Par_]
    C.m, C.d = vt_(C.dTT, C.r)
    C.Nt = CF(N_=N_,nF='Ct',dTT=deepcopy(C.dTT), m=C.m,d=C.d,c=C.c)
    return cent_TT(C, C.r)

def sum2G(Ft_,tt,c,r, root=None, init=1, typ=None, fsub=1):  # updates root if not init

    N_,_,ntt,nc,nr = Ft_[0]
    if not init: N_+=root.N_; ntt+=root.Nt.dTT; nc+=root.Nt.c  # add,norm nr? batch root updates?
    N = N_[0]
    m,d = vt_(tt,r)
    if typ is None: typ = N.typ
    G = Copy_(N,root,init=1,typ=typ); G.dTT=tt; G.m=m; G.d=d; G.c=c; G.r=r
    for N in N_[1:]:
        add_N(G,N, coef=N.c/c)  # skip forks
    Nt = sum2F(N_,'Nt',G, ntt, nc, nr)
    if len(Ft_) > 1 and Ft_[1]:  # Nt.Lt, empty if singleton, starting from trace_edge
        L_,_,ltt,lc,lr= Ft_[1]
        if init:  # else same ext
            yx_ = np.array([n.yx for n in N_]); G.yx=yx= yx_.mean(axis=0); dy_,dx_= (yx_-yx).T
            G.span = np.hypot(dy_,dx_).mean()  # N centers dist to G center
        else: L_+=root.L_; ltt+=root.Lt.dTT; lc+=root.Lt.c
        A = np.sum([l.angl[0] for l in L_], axis=0) if L_ else np.zeros(2)
        G.angl = np.array([A, np.sign(G.dTT[1] @ wTTf[1])], dtype=object)  # angle dir = d sign
        m, d = vt_(ltt,lr)
        Lt = CF(N_=L_,nF='Lt',root=G.Nt,dTT=ltt,m=m,d=d,c=lc,r=lr)  # Nt only? (should be dTT)
        if m > ave*specw:  # comp typ -1 pre-links
            L_,pL_= [],[]; [L_.append(L) if L.typ==1 else pL_.append(L) for L in G.L_]
            if pL_:
                if sum_vt(pL_,r)[0] > ave * specw:
                    for L in pL_:
                        link = comp_N(*L.nt, r, 1, L.angl[0], L.span)  # full = 1
                        Lt.dTT+= link.dTT-L.dTT; L_+=[link]
                    Lt.m, Lt.d = vt_(Lt.dTT, r)
        if fsub and Lt.m * Lt.d * ((len(N_)-1)*Lw) > ave*avd * (r+1) * cw:  # Variance * borrowed Match?
            V = G.m - ave*(r+1) * connw  # centw-=connw?
            if V > 0:  # cent|conn divisive clustering:
                if (mdecay(Lt.N_)- decay) * V > ave*centw:  # comb eval, Lt.N_ is L_, G.L_ is not assigned yet
                    cluster_C(G.Nt, N_,r+1)  # cent cluster: N_->Ct.N_, higher than G.C_
                else: cluster_N(G.Nt, N_,r+1)  # conn cluster/ rc+1: N_-> Nt.N_| N_
        Nt.Lt = Lt
    else: G.yx = G.yx[0]  # remove list
    if N_[0].typ==2 and G.L_:  # else mang = 1
        G.mang = np.mean([comp_A(G.angl[0], l.angl[0])[0] for l in G.L_])
    G.Nt = Nt  # Nt vals are pre-summed in Gt
    G.m, G.d = vt_(G.dTT,r)
    G.sub = N.sub+1 if G.L_ else N.sub  # remain the same sub value for singleton
    if len(Ft_) > 2:  # from cluster_N
        B_,_,btt,bc,br = Ft_[2]; m, d = vt_(btt,br)
        Bt = CN(dTT=btt,m=m,d=d,c=bc,r=br,root=G); Bt.N_=B_; Bt.nF='Bt' # use as root:
        if typ!=1 and d > avd*br*nw:  # no ddfork
            cross_comp(Bt, bc,'Bt')
        Bt.brrw = Bt.m * (root.root.m * (decay * (root.root.span/G.span)))  # trans-Bt contrast is local; subtract from root.m?
        G.Bt = Bt  # for sub-comp only, doesn't affect G vals?
    G.rN_ = sorted(G.rN_, key=lambda x: (x.m/x.c), reverse=True)  # if lG?
    # no Ct yet?
    return G

def add_N(G, N, coef=1, merge=0):  # sum Fts if merge

    N.fin = 1; N.root = G
    _c,c = G.c,N.c*coef; C=_c+c  # weigh contribution of intensive params
    if hasattr(G,'m_'): G.r = np.sum([o*coef for o in N.o_]); G.rN_+=N.rN_; G.m_+=N.m_; G.o_+=N.o_  # not sure
    if N.typ:  # not PP|Cent
        if N.rN_: G.C_+= N.rN_; G.Ct.dTT += N.Ct.dTT*coef; G.Ct.c += N.Ct.c*coef  # flat? L_,B_ stay nested
        G.span = (G.span*_c+N.span*c*coef) / C
        A,a = G.angl[0],N.angl[0]; A[:] = (A*_c+a*c*coef) /C  # vect only
        if isinstance(G.yx, list): G.yx += [N.yx]  # weigh by C?
    if N.typ>1: # nodes
        G.baseT = (G.baseT*_c+ N.baseT*c*coef) /C
        G.mang = (G.mang*_c + N.mang*c*coef) /C
        G.box = extend_box(G.box, N.box)
    if merge:
        merge_f(G, N, cc=G.c/N.c)  # if not batched
    # if N is Fg: margin = Ns of proj max comp dist > min _Fg point dist: cross_comp Fg_?
    return N

def sum_vt(n_, rr=0, rm=0, rd=0):  # weighted sum of CN|CF list

    C = sum(n.c for n in n_); R = 0; TT = np.zeros((2,9))
    for n in n_:
        cr = n.c/C; TT += n.dTT*cr; R += n.r*cr  # * weight
    m,d = vt_(TT, R+rr)  # use deviations from tentative m,d:
    return m-rm, d-rd, TT, C, R

def sum2f(n_, nF, root):
    m,d, tt,c,r = sum_vt(n_)
    Ft = CF(N_=n_,nF=nF, dTT=tt,m=m,d=d,c=c,r=r, root=root)
    for n in n_: n.root = Ft
    return Ft

def sum2F(N_, nF, root, TT=np.zeros((2,9)), C=0, R=0, fset=1, fCF=1):  # -> CF/CN

    def sum_H(N_, Ft):
        H = []
        for N in N_:
            if H: H[0] += N.N_  # new top level
            elif  N.N_: H = [list(N.N_)]
            for Lev,lev in zip_longest(H[1:], reversed(N.Nt.H)):  # top-down?
                if lev:
                    if Lev: Lev += lev.N_
                    else: H += [list(lev.N_)]
        Ft.H = [sum2F(lev,nF,Ft,fset=0) for lev in reversed(H)] if H else []  # bottom-up?
    if not C: m,d,TT,C,R = sum_vt(N_)
    else:     m,d = vt_(TT,R)
    Ft = (CN,CF)[fCF](nF=nF, dTT=TT,m=m,d=d,c=C,r=R, root=root)  # root Bt|Ct ->CN
    for N in N_: Ft.N_ += [N]; N.root = Ft
    sum_H(N_,Ft)  # sum lower levels
    if fset:
        setattr(root, Ft.nF, Ft)
        # root_update(root, Ft, ini=2 if nF[0]=='t' else 1)
    return Ft

def add_F(Ft,ft, r=1, merge=1):

    def sum_H(H, h, r, root):  # reverse to align bottom-up:
        for Lev,lev in zip_longest(reversed(H),reversed(h)):
            if lev:
                if Lev: add_F(Lev,lev, r, merge=1)
                else:   H.append(CopyF(lev, root))
        return list(reversed(H))
    Ft.c += ft.c
    cr = Ft.c/ft.c; Ft.dTT += ft.dTT*cr*r; Ft.rc += ft.rc*r; Ft.m,Ft.d = vt_(Ft.dTT,Ft.rc)
    if merge:
        for n in ft.N_: Ft.N_ += [n]; n.root = Ft
        if Ft.H and ft.H: Ft.Nt.H = sum_H(Ft.H, ft.H, r, Ft)
    else: Ft.N_.append(ft)
    return Ft

def merge_f(N,n, cc=1):
    for Ft, ft in zip((N.Nt, N.Bt, N.Lt), (n.Nt, n.Bt, n.Lt)):
        if ft:
            add_F(Ft, ft, (n.r + n.r*cc) / 2)  # ft*cc?
            setattr(N, Ft.nF, Ft)  # not sure about N.root update

def cent_TT(C, r):  # weight attr matches | diffs by their match to the sum, recompute to convergence

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
            if np.sum(np.abs(w_-_w_)) > ave*r:
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

def CopyF(F, root=None, r=1):  # F = CF
    C = CF(dTT=F.dTT*r, m=F.m, d=F.d, c=F.c, r=F.r, root=root or F.root)
    C.N_ = [Copy_(N,root=C) for N in F.N_]  # flat
    return C

def Copy_(N, root=None, init=0, typ=None):

    if typ is None: typ = 2 if init<2 else N.typ  # G.typ = 2, C.typ=0
    C = CN(dTT=deepcopy(N.dTT),typ=typ); C.root = N.root if root is None else root
    for attr in ['m','d','c','r']: setattr(C,attr, getattr(N,attr))
    if not init and C.typ==N.typ: C.Nt = CopyF(N.Nt,root=C)
    if typ:
        for attr in ['fin','span','mang','sub','exe']: setattr(C,attr, getattr(N,attr))
        for attr in ['nt','baseT','box','rim','compared']: setattr(C,attr, copy(getattr(N,attr)))
        for attr in ['Nt','Lt','Bt']:  setattr(C, attr, CopyF(getattr(N,attr),root=C))
        if init:  # new G
            C.yx = [N.yx]; C.angl = np.array([copy(N.angl[0]), N.angl[1]],dtype=object)  # to get mean
            if init==1:  # else centroid
                C.L_=[l for l in N.rim if l.m>ave]; N.root=C; C.fin = 0
        else:
            C.Lt=CopyF(N.Lt); C.Bt=CopyF(N.Bt)  # empty in init G
            C.angl = copy(N.angl); C.yx = copy(N.yx)
        if typ > 1:
            C.rim = CopyF(N.rim)
    if init==2:
        for n in C.N_: n.m_=[]; n._m_=[]; n.o_=[]; n._o_=[]
    return C

def extend_box(_box, box):
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return np.array((min(y0,_y0), min(x0,_x0), max(yn,_yn), max(xn,_xn)))

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
        L__ = [l for n in Fg.N_ for l in n.L_]
        if L__: m,d, dTT,c,_ = sum_vt(L__)
        return m,d,c,dTT

    wTTf = np.ones((2,9))  # sum dTT weights: m_,d_ [M,D,n, I,G,A, L,S,ext_A]: Et, baseT, extT
    rM, rD, rVd = 1,1,0
    _m, _d, _n, _dTT = L_ders(root)
    for lev in root.Nt.N_:  # top-down, not lev-selective, not recursive
        m ,d ,n, dTT = L_ders(lev)
        rM += _m / m or eps  # mat,dif change per level
        rD += _d / d or eps
        wTTf += np.abs(_dTT / (dTT+eps))
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

def proj_TT(L, cos_d, dist, r, pTT, wTT, fdec=0, frec=0, dec=1):  # accumulate link pTT with iTT or eTT internally, L may be N?

    dec = dist if fdec else ave ** (1 + (dist * dec) / L.span)  # not fully revised, ave = match decay rate / unit distance
    TT = np.array([L.dTT[0] * dec, L.dTT[1] * cos_d * dec])
    cert = abs(val_(TT,r,wTT) - ave)  # approximation
    if cert > ave:
        pTT+=TT; return  # certainty margin = ave
    if not frec:  # non-recursive
        for lev in L.Nt.N_:  # refine pTT
            proj_TT(lev, cos_d, dec, r+1, pTT, wTT, fdec=1, frec=1)
    pTT+=TT  # L.dTT is redundant to H, neither is redundant to Bt,Ct
    if L.Bt:  # + trans-link tNt, tBt, tCt?
        TT = L.Bt.dTT
        if TT is not None:
            pTT += np.array([TT[0] * dec, TT[1] * cos_d * dec])

def proj_N(N, dist, A, r, dec=1):  # arg rc += N.rc+connw, recursively specify N projection val, add pN if comp_pN?

    cos_d = (N.angl[0].dot(A) / (np.hypot(*N.angl[0]) * dist + eps)) * N.angl[1]  # internal x external angle alignment
    iTT, eTT = np.zeros((2,9)), np.zeros((2,9))
    wTT = TTw(N)
    for L in N.L_+N.B_: proj_TT(L, cos_d, dist, L.r+r, iTT, wTT, dec)  # accum TT internally
    for L in N.rim.N_:  proj_TT(L, cos_d, dist, L.r+r, eTT, wTT, dec)
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
        PP = CN(typ=0, dTT=dTT,m=m,d=d,c=c, baseT=baseT,box=box,yx=yx,angl=A,span=np.hypot(dy/2,dx/2))  # set root in trace_edge
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
    blob_ = tile.N_; G_,TT,C = [],np.zeros((2,9)),0
    for blob in blob_:
        if not blob.sign and blob.G > aveB:
            edge = slice_edge(blob, rV)
            if edge.G * ((len(edge.P_)-1)*Lw) > ave * sum([P.latT[4] for P in edge.P_]):
                PPm_ = comp_slice(edge, rV, wTTf)
                N_ = [PP2N(PPm) for PPm in PPm_]
                for PPd in edge.link_: PP2N(PPd)
                for N in N_:
                    if N.B_: N.Bt = sum2f([B.root for B in N.B_],'Bt',N)
                if val_(sum_vt(N_)[2], 3, TTw(tile), (len(PPm_)-1)*Lw) > 0:
                    G_,TT,C = trace_edge(N_,G_,TT,C, 3,tile)  # flatten, cluster B_-mediated Gs, init Nt
    if G_:
        setattr(tile,'Nt', sum2F(G_,'Nt',tile,TT,C,R=1))  # update tile.wTT?
        tile.dTT = TT; tile.c = C
        if vt_(TT)[0] > ave:
            return tile

def trace_edge(N_,_G_,_TT,_C, r,root):  # cluster contiguous shapes via PPs in edge blobs or lGs in boundary/skeleton?

    L_, cT_, lTT, lc = [],set(),np.zeros((2,9)),0  # comp co-mediated Ns:
    for N in N_: N.fin = 0
    for N in N_:
        _N_ = [rN for B in N.B_ for rN in B.rN_ if rN is not N]   # + node-mediated
        for _N in list(set(_N_)):  # share boundary or cores if lG with N, same val?
            cT = tuple(sorted((N.id,_N.id)))
            if cT in cT_: continue
            cT_.add(cT)
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx)  # Rc = r+ (N.r+_N.r)/2
            Link = comp_N(_N,N, r, A=dy_dx, span=dist)
            if vt_(Link.dTT)[0] > ave*r:  L_+=[Link]  # r = 1|2, add Bt?
            if L_: _,_, lTT, lc, _ = sum_vt(L_)
    Gt_ = []
    for N in N_:  # flood-fill G per seed N
        if N.fin: continue
        N.fin=1; _N_=[N]; Gt=[]; N.root=Gt; n_,ntt,nc, l_,ltt,lc = [N],N.dTT.copy(),N.c,[],np.zeros((2,9)),0  # Gt
        while _N_:
            _N = _N_.pop(0)
            for L in _N.rim:
                if L in L_:
                    n = L.nt[0] if L.nt[1] is _N else L.nt[1]
                    if n in N_:
                        if n.root is Gt: continue
                        if n.fin:  # merge n root
                            _root = n.root; n_+=_root[0];l_+=_root[3]; _root[6]=1
                            for _n in _root[0]: _n.root = Gt
                        else:
                            n.fin=1; _N_+=[n]; n_+=[n]; l_+=[L]  # add single n
                        n.root = Gt
        _,_,ntt,nc,_ = sum_vt(n_)
        if l_: _,_,ltt,lc,_ = sum_vt(l_); ltt*=lc/nc;
        Gt += [n_,ntt,nc, l_,ltt,lc, 0]; Gt_+=[Gt]
    G_, TT,C = [],np.zeros((2,9)),0
    for n_,ntt,nc,l_,ltt,lc,merged in Gt_:
        if not merged:
            if vt_(ntt+ltt,r)[0] > ave*r:  # wrap singletons, use Gt.rc?
                TT += ntt+ltt; C += nc+lc  # add Bt?
                G_ += [sum2G(((n_,'Nt',ntt,nc,r),(l_,'Lt',ltt,lc,r) if l_ else []), ntt+ltt,nc+lc,r,root,typ=2,fsub=0)]
            else:
                for N in n_: N.fin=0; N.root=root
    if val_(TT,r+1,TTw(root)) > 0: _G_+=G_;_TT+=TT;_C+=C  # eval per edge, concat in tile?
    return _G_,_TT,_C

# frame expansion per level: cross_comp lower-window N_,C_, forward results to next lev, project feedback to scan new lower windows

def frame_H(image, iY,iX, Ly,Lx, Y,X, rV, max_elev=4, wTTf=np.ones((2,9))):  # all initial args set manually

    def base_tile(y,x):  # 1st level, higher levels get Fg s
        Fg = frame_blobs_root( comp_pixel( image[y:y+Ly, x:x+Lx]), rV)
        Fg = vect_edge(Fg, rV)  # form, trace PP_
        if Fg: cross_comp(Fg.Nt, r=Fg.r)
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
        if Fg_:
            _,_,gTT,gC,gRc = sum_vt(Fg_, elev)
            if val_(gTT, gRc/len(Fg_)+elev, wTTf, mw=(len(Fg_)-1)*Lw) > 0:
                return Fg_

    global ave,avd, Lw, intw, cw,nw, centw,connw, distw, mW, dW
    ave,avd, Lw, intw, cw,nw, centw,connw, distw = np.array([ave,avd, Lw, intw, cw, nw, centw, connw, distw]) / rV

    frame = CN(box=np.array([0,0,Y,X]), yx=np.array([Y//2, X//2]))
    Fg=[]; elev=0
    while elev < max_elev:  # same center in all levels
        Fg_ = expand_lev(iY,iX, elev, Fg)
        if Fg_:  # higher-scope sparse tile
            _,_,fTT,fC,_ = sum_vt(Fg_)
            frame = sum2G(Fg_, fTT, fC, r=1,root=frame,init=0)
            if Fg and cross_comp(Fg.Nt, r=elev)[0]:  # or val_? spec->tN_,tC_,tL_
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