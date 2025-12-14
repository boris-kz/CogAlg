import numpy as np, weakref
from copy import copy, deepcopy
from math import atan2, cos, floor, pi  # from functools import reduce
from itertools import zip_longest, combinations, chain, product  # from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, imread, comp_pixel, CBase
from slice_edge import slice_edge
from comp_slice import comp_slice
'''
Lower modules start with cross-comp and clustering image pixels, here the initial input is PPs: segments of matching blob slices or Ps.

This is a main module of open-ended dual clustering algorithm, designed to discover empirical patterns of indefinite complexity. 
It's a recursive 4-stage cycle, forming agglomeration levels: 
generative cross-comp, compressive clustering, filter-adjusting feedback, code-extending forward (not yet implemented): 

Cross-comp forms Miss, Match: min= shared_quantity for directly predictive params, else inverse deviation of miss=variation, 2 forks:
rng+: incremental-range cross-comp nodes: edge segments at < max distance, cluster if they match. 
der+: incremental-derivation cross-comp links, from node cross-comp, if abs_diff * rel_adjacent_match

Clustering is compressively grouping the elements, by direct similarity to centroids or transitive similarity in graphs, 2 forks:
nodes: connectivity clustering / >ave M, progressively reducing overlap by exemplar selection, centroid clustering, floodfill.
links: correlation clustering if >ave D, forming contours that complement adjacent connectivity clusters.

That forms hierarchical graph representation: dual tree of down-forking elements: node_H, and up-forking clusters: root_H:
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
Similar to neurons: dendritic input tree and axonal output tree, but with lateral cross-comp and nested param sets per layer.

The fitness function is predictive value of the model, estimated through multiple orders of projection:
initially summed match, refined by projection in external space and time (combining match with directional diffs),
then by comparing projected match to actual match, testing the accuracy of cross_comp and clustering process. 
And so on, higher orders of projection should be generated recursively from lower-order comparisons.

Feedback of projected match adjusts filters to maximize next match, including coordinate filters that select new inputs.
(currently in ffeedback(), to be refined by cross_comp of co-projected patterns: "imagination, planning, action" in part 3)   
This is similar to backprop, but sparse and weights are only for evals, they don't change the data propagating up the hierarchy.

It should also modify the code by adding weights on code elements according to their contribution to projected match.
The weights should trigger skipping or recursing over corresponding functions ( blocks ( operations.
Also cross-comp and cluster (compress) code elements and function calls, real and projected, though much more coarsely that data?

notation:
prefix  f denotes flag
postfix t denotes tuple, multiple ts is a nested tuple
prefix  _ denotes prior of two same-name vars, multiple _s for relative precedence
postfix _ denotes array of same-name elements, multiple _s is nested array
capitalized vars are summed small-case vars
'''
eps = 1e-7
class CN(CBase):
    name = "node"
    def __init__(n, **kwargs):
        super().__init__()
        n.typ = kwargs.get('typ', 0)
        # 1=L: typ,nt,dTT, m,d,c,rc, root,rng,yx,box,span,angl,fin,compared, N_,B_,C_,L_,Nt,Bt,Ct from comp_sub?
        # 2=G: + rim, eTT, em,ed,ec, baseT,mang,sub,exe, Lt, tNt, tBt, tCt?
        # 0=PP: if typ: comp_sub?
        n.dTT = kwargs.get('dTT',np.zeros((2,9)))  # total forks dTT: m_,d_ [M,D,n, I,G,a, L,S,A]
        n.m,  n.d, n.c = kwargs.get('m',0), kwargs.get('d',0), kwargs.get('c',0)  # sum dTT
        n.rim = kwargs.get('rim',[])  # external links, rng-nest?
        n.eTT = kwargs.get('eTT',np.zeros((2,9)))  # sum rim dTT
        n.em, n.ed, n.ec = kwargs.get('em',0),kwargs.get('ed',0),kwargs.get('ec',0)  # sum eTT
        n.rc  = kwargs.get('rc', 1)  # redundancy to ext Gs, ave in links?
        n.N_, n.B_, n.C_, n.L_ = kwargs.get('N_',[]), kwargs.get('B_',[]), kwargs.get('C_',[]), kwargs.get('L_',[])  # base elements
        n.Nt, n.Bt, n.Ct, n.Lt= kwargs.get('Nt',CF()), kwargs.get('Bt',CF()), kwargs.get('Ct',CF()), kwargs.get('Lt',CF())  # nested elements
        # Ft is [dTT,m,d,c,rc,N_], N_=H: [[N_,dTT]] in Nt, nest=len(N_)
        n.baseT = kwargs.get('baseT',np.zeros(4))  # I,G,A: not ders, in links for simplicity, mostly redundant
        n.nt    = kwargs.get('nt', [])  # nodet, links only
        n.yx    = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.box   = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.span  = kwargs.get('span',0) # distance in nodet or aRad, comp with baseT and len(N_), not additive?
        n.angl  = kwargs.get('angl',[np.zeros(2),0])  # (dy,dx),dir, sum from L_
        n.mang  = kwargs.get('mang',1)  # ave match of angles in L_, =1 in links
        n.root  = kwargs.get('root',None)  # immediate
        n.rng = kwargs.get('rng',1)  # or med: loop count in comp_node_|link_
        n.sub = 0  # composition depth relative to top-composition peers?
        n.fin = kwargs.get('fin',0)  # clustered, temporary
        n.exe = kwargs.get('exe',0)  # exemplar, temporary
        n.rN_ = kwargs.get('rN_',[]) # reciprocal root nG_ for bG | cG, nG has Bt.N_,Ct.N_ instead
        n.tNt,n.tBt,n.tCt = kwargs.get('tNt',CF()), kwargs.get('tBt',CF()), kwargs.get('tCt',CF())
        n.compared = set()
        # ftree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def __bool__(n): return bool(n.c)

class CF(CBase):
    name = "fork"
    def __init__(f, **kwargs):
        super().__init__()
        f.m = kwargs.get('m', 0)
        f.d = kwargs.get('d', 0)
        f.c = kwargs.get('c', 0)
        f.rc = kwargs.get('rc', 0)
        f.N_ = kwargs.get('N_',[])  # H in Nt, empty in Lt, N_ in forks and lev
        f.dTT  = kwargs.get('dTT',np.zeros((2,9)))
        f.root = kwargs.get('root',None)
        # to use as root in cross_comp:
        f.L_, f.B_, f.C_ = kwargs.get('L_',[]),kwargs.get('B_',[]),kwargs.get('C_',[])
        # assigned by sum2T in cross_comp:
        f.Nt, f.Bt, f.Ct = kwargs.get('Nt',[]),kwargs.get('Bt',[]),kwargs.get('Ct',[])
    def __bool__(f): return bool(f.c)

ave = .3; avd = ave*.5  # ave m,d / unit dist, top of filter specification hierarchy
wM,wD,wc, wG,wI,wa, wL,wS,wA = 10, 10, 20, 20, 5, 20, 2, 1, 1  # dTT param root weights = reversed relative estimated ave
wT = np.array([wM,wD,wc, wG,wI,wa, wL,wS,wA]); wTTf = np.array([wT*ave, wT*avd])
aveB, distw, Lw, intw = 100, .5,.5,.5  # not-compared attr filters, *= ave|avd?
compw, centw, contw, specw = 5, 10, 15, 10  # process filters, *= (ave,avd)[fi], cost only now
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

def vt_(TT, wTT=None):  # brief val_ to get m, d

    m_,d_ = TT; ad_ = np.abs(d_); t_ = m_ + ad_ + eps  # ~ max comparand
    if wTT is None: wTT = wTTf
    return m_/t_ @ wTT[0], ad_/t_ @ wTT[1]

def val_(TT, rc, wTT, fi=1, mw=1.0, rn=.5, _TT=None):  # m,d eval per cluster, rn = n / (n+_n), .5 for equal weight _dTT?

    t_ = np.abs(TT[0]) + np.abs(TT[1])  # not sure about abs m_
    if wTT is None: wTT = wTTf  # or default wTT =  / cent_TT?
    rv = TT[0] / (t_+eps) @ wTT[0] if fi else TT[1] / (t_+eps) @ wTT[1]  # fork / total per scalar
    if _TT is not None:
        _t_ = np.abs(_TT[0]) + np.abs(_TT[1])
        _rv = _TT[0] / (_t_+eps) @ wTT[0] if fi else _TT[1] / (_t_+eps) @ wTT[1]
        rv = rv * (1-rn) + _rv * rn  # add borrowed root | boundary alt fork val?

    return rv * mw - (ave if fi else avd) * rc

def TTw(G): return getattr(G,'wTT', None) or getattr(G.root,'wTT', None) or wTTf

def cross_comp(root, rc, fC=0):  # core function, mediates rng+ and der+ cross-comp and clustering, rc=rdn+olp

    N_, mL_,mTT,mc, dL_,dTT,dc = comp_C_(root.N_,rc) if fC else comp_N_(root.N_,rc); nG_=[]
    # m fork:
    if len(mL_)>1 and val_(mTT, rc+compw, TTw(root), mw=(len(mL_)-1)*Lw) > 0:
        root.L_= mL_; sum2T(mL_,rc,root,'Lt')  # new ders, no Lt.N_
        for n in N_: n.em = sum([l.m for l in n.rim]) / len(n.rim)  # pre-val_
        nG_,rc = Cluster(root, mL_, rc,fC)  # fC=0: exemplars, cluster_,root_replace
    # d fork:
    if fC<2 and dL_ and val_(dTT, rc+compw, TTw(root), fi=0, mw=(len(dL_)-1)*Lw) > avd:  # comp dL_|dC_, not ddC_
        root.B_= dL_; sum2T(dL_,rc,root,'Bt')  # new ders
        bG_,rc = cross_comp(root.Bt, rc, fC*2)
        if bG_: sum2T(bG_,rc,root,'Bt')  # replace Bt
        form_B__(root.N_, root.B_)  # add boundary to Ns, N to Bg.rN_
    # recursion:
    if val_(mTT, rc+contw, mw=(len(root.N_)-1) *Lw, wTT=root.wTT) > 0:  # mval only
        nG_,rc = trace_edge(root.N_,rc,root)  # comp Ns x N.Bt|B_.nt, with/out mfork?
    if nG_ and val_(root.dTT, rc+compw, TTw(root), mw=(len(root.N_)-1)*Lw, _TT=mTT) > 0:
        nG_,rc = cross_comp(root, rc)
        # connect agg+, fC=0
    return nG_,rc  # nG_: recursion flag

def comp_C_(C_, rc,_C_=[], fall=1):  # simplified for centroids, trans-N_s, levels
    # max attr sort to constrain C_ search in 1D, add K attrs and overlap?
    # proj C.L_: local?
    N_,L_,mTT,mc, B_,dTT,dc = [],[],np.zeros((2,9)),0, [],np.zeros((2,9)),0
    if fall:
        pairs = product(C_,_C_) if _C_ else combinations(C_,r=2)  # comp between | within list
        for _C, C in pairs:
            dtt = comp_derT(_C.dTT[1], C.dTT[1]); m,d = vt_(dtt); c = min(_C.c,C.c)
            dC = CN(nt=[_C,C], m=m,d=d, c=c, dTT=dtt, span=np.hypot(*_C.yx - C.yx))
            _C.rim += [dC]; C.rim += [dC]
            if   dC.m > ave*(contw+rc) > 0: L_+=[dC]; mTT+=dC.dTT; mc+=c; N_ += [_C,C]
            elif dC.d > avd*(contw+rc) > 0: B_+=[dC]; dTT+=dC.dTT; dc+=c  # not in N_?
    else:
        # sort, select along eigenvector, may be muli-level, not yet implemented
        for C in C_: C.compared =set()
        C_ = sorted(C_, key=lambda C: C.dTT[0][np.argmax(wTTf[0]+wTTf[1])])  # max weight m
        for j in range( len(C_)-1):
            _C = C_[j]; C = C_[j+1]
            if _C in C.compared: continue
            dy_dx = _C.yx-C.yx; dist = np.hypot(*dy_dx)
            Link = comp_N(_C,C, rc, A=dy_dx, span=dist, rng=1)  # or comp_derT?
            if   Link.m > ave*(contw+rc) > 0: L_+=[Link]; mTT+=Link.dTT; mc+=Link.c; N_ += [_C,C]
            elif Link.d > avd*(contw+rc) > 0: B_+=[Link]; dTT+=Link.dTT; dc+=Link.c  # not in out_?

    return list(set(N_)), L_,mTT,mc, B_,dTT,dc

def comp_N_(iN_, rc, _iN_=[]):

    def proj_V(_N,N, dist, Ave, pVt_):  # _N x N induction

        Dec = decay**(dist/((_N.span+N.span)/2))
        iTT = (_N.dTT+N.dTT) * Dec  # includes Lt.dTT, etc
        eTT = (_N.eTT+N.eTT) * Dec  # summed from rim
        if abs( vt_(eTT)[0]) * ((len(pVt_)-1)*Lw) > Ave*specw:  # spec N links
            eTT = np.zeros((2,9)) # recompute
            for _dist,_dy_dx,__N,_V in pVt_:
                eTT += proj_N(N,_dist,_dy_dx, rc)  # proj N L_,B_,rim, if pV>0: eTT += pTT?
                eTT += proj_N(_N,_dist,-_dy_dx, rc)  # reverse direction
        return iTT+eTT

    N_, L_,mTT,mc, B_,dTT,dc, dpTT = [],[],np.zeros((2,9)),0, [],np.zeros((2,9)),0, np.zeros((2,9))
    for i, N in enumerate(iN_):  # get unique all-to-all pre-links
        N.pL_ = []
        for _N in _iN_ if _iN_ else iN_[i+1:]:  # optional _iN_ as spec
            if _N.sub != N.sub: continue  # or comp x composition?
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx)
            N.pL_ += [[dist, dy_dx, _N]]
        N.pL_.sort(key=lambda x: x[0])  # proximity prior
    for N in iN_:
        pVt_ = []  # [dist, dy_dx, _N, V]
        for dist, dy_dx, _N in N.pL_:  # rim angl not canonic
            olp = (N.rc +_N.rc) / 2; rc+=olp
            pTT = proj_V(_N,N, dist, ave*rc, pVt_)
            m, d = vt_(pTT); V = m - ave*rc  # +|-match certainty
            if V > 0:
                if abs(V) < ave:  # different ave for projected surprise value, comp in marginal predictability
                    Link = comp_N(_N,N, rc, A=dy_dx, span=dist)
                    if   Link.m > ave: L_+=[Link]; mTT+=Link.dTT; mc+=Link.c; N_ += [_N,N]  # combined CN dTT and L_
                    elif Link.d > avd: B_+=[Link]; dTT+=Link.dTT; dc+=Link.c  # no overlap to simplify
                    V = Link.m-ave*rc; dpTT+= pTT-Link.dTT  # prediction error to fit code
                else:
                    pL = CN(typ=-1, nt=[_N,N], dTT=pTT, m=m, d=d, c=min(N.c,_N.c), angl=np.array([dy_dx,1],dtype=object), span=dist)
                    L_+= [pL]; N.rim+=[pL]; _N.rim+=[pL]  # same as links in clustering
                pVt_ += [[dist,dy_dx,_N,V]]  # for next rim eval
            else:
                break  # beyond induction range
    return list(set(N_)), L_,mTT,mc, B_,dTT,dc  # + dpTT for code-fitting backprop?

def comp_N(_N,N, rc, A=np.zeros(2), span=None, rng=1):  # compare links, optional angl,span,dang?

    TT,_ = base_comp(_N, N)
    yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])  # ext
    angl = [A, np.sign(TT[1] @ wTTf[1])]  # canonic direction
    m, d = vt_(TT)
    Link = CN(typ=1,nt=[_N,N], dTT=TT,m=m,d=d,c=min(N.c,_N.c), yx=yx,box=box,span=span,angl=angl,rng=rng, rc=rc, baseT=(_N.baseT+N.baseT)/2)
    if m > ave*rc and _N.typ and N.typ:  # skip PP if called from sub_comp' comp_C_
        comp_sub(_N,N, rc,Link)  # root_update
    for n, _n in (_N,N),(N,_N):  # if rim-mediated comp: reverse dir in _N.rim: rev^_rev?
        n.rim += [Link]; n.eTT += TT; n.ec += Link.c; n.compared.add(_n)  # or conditional n.eTT / rim later?
    return Link

def comp_sub(_N,N, rc, root):  # unpack node trees down to numericals and compare them

    Rc = rc
    for _F_,F_,nF_,nFt in zip((_N.N_,_N.B_,_N.C_),(N.N_,N.B_,N.C_), ('N_','B_','C_'),('Nt','Bt','Ct')):  # + tN_,tB_,tC_ from trans_cluster?
        if _F_ and F_:
            N_,L_,mTT,mc, B_,dTT,dc = comp_C_(_F_,Rc,F_); dF_= L_+B_  # trans-links, callee comp_N may call deeper comp_sub
            if dF_:
                sum2T(dF_,Rc,root, nFt, mTT+dTT, mc+dc); setattr(root,nF_,dF_); Rc += 1
    for _lev,lev in zip(_N.Nt.N_, N.Nt.N_):  # no comp Bt,Ct: external to N,_N
        Rc += 1  # deeper levels are redundant
        tt = comp_derT(_lev.dTT[1],lev.dTT[1]); m,d = vt_(tt)
        oN_= set(_lev.N_) & set(lev.N_)  # intersect, or offset, or comp_len?
        dlev = CF(N_=oN_, dTT=tt, m=m,d=d, c=min(_lev.c,lev.c), rc=min(_lev.rc,lev.rc), root=root)  # min: only shared elements are compared
        root.Nt.N_ += [dlev]  # root:link, nt.N_:derH
        root_update(root.Nt, dlev)  # recursive

def base_comp(_N,N):  # comp Et,extT,dTT, baseT if N is G?

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
            m_ += [min(_a,a) if _p<0==p<0 else -min(_a,a)]  # + complement max(_a,a) for +ves?
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

def rolp(N, _N_, R=0):  # rel V of L_|N.rim overlap with _N_: inhibition|shared zone, oN_ = list(set(N.N_) & set(_N.N_)), no comp?

    n_ = set(N.N_) if R else {n for l in N.rim for n in l.nt if n is not N}  # nrim
    olp_ = n_ & _N_
    if olp_:
        oTT = np.sum([i.dTT for i in olp_], axis=0)
        return sum(oTT[0]) / sum(N.dTT[0] if R else N.eTT[0])  # relative M (not rm) of overlap?
    else:
        return 0

def get_exemplars(N_, rc):  # multi-layer non-maximum suppression -> sparse seeds for diffusive centroid clustering

    E_ = set()
    for rdn, N in enumerate(sorted(N_, key=lambda n:n.em, reverse=True), start=1):  # strong-first
        roV = rolp(N, E_)  #| olp in proportion to pairwise similarity?
        if N.em > ave * (rc+ rdn+ compw +roV):  # ave *= relV of overlap by stronger-E inhibition zones
            E_.update({n for l in N.rim for n in l.nt if n is not N and N.em > ave*rc})  # selective nrim
            N.exe = 1  # in point cloud of focal nodes
        else:
            break  # the rest of N_ is weaker, trace via rims
    return E_

def Cluster(root, iL_, rc, fC):  # generic clustering root

    def trans_cluster(root, iL_,rc):  # called from cross_comp(Fg_), others?

        dN_,dB_,dC_ = [],[],[]  # splice specs from links between Fgs in Fg cluster
        for Link in iL_: dN_+= Link.N_; dB_+= Link.B_; dC_+= Link.C_
        frc = rc  # trans-G links
        for ft,f_,link_,fC in [('tNt','tN_',dN_,0), ('tBt','tB_',dB_,0), ('tCt','tC_',dC_,1)]:
            if link_:
                frc += 1  # trans-fork redundancy count, re-assign later?
                Ft = sum_N_(link_,frc,root); cluster_N(Ft, {n for L in link_ for n in L.nt}, frc)
                if val_(Ft.dTT, frc, TTw(root), mw=(len(Ft.N_)-1)*Lw) > 0:
                    cross_comp(Ft, frc, fC=fC)  # unlikely, doesn't add rc?
                setattr(root,f_,link_); setattr(root, ft, Ft)  # trans-fork_ via trans-G links
    G_ = []
    if fC or not root.root:  # flat connect_cluster centroids, no exemplars or higher centroid clustering
        N_, L_, i = [],[],0  # add pL_, pN_ for pre-links?
        if fC < 2:  # merge similar Cs, not dCs, no recomp
            link_ = sorted(list({L for L in iL_}), key=lambda link: link.d)  # from min D
            for i, L in enumerate(link_):
                if val_(L.dTT, rc+compw, TTw(L), fi=0) < 0:
                    _N, N = L.nt; root.L_.remove(L)  # merge nt, remove link
                    if _N is not N:  # not yet merged
                        for n in N.N_: add_N(_N, n); _N.N_ += [n]
                        for l in N.rim: l.nt = [_N if n is N else n for n in l.nt]
                        if N in N_: N_.remove(N)
                        N_ += [_N]
                else: L_ = link_[i:]; break
        else: L_ = iL_
        N_ += list({n for L in L_ for n in L.nt})  # include merged Cs
        if val_(root.dTT, rc+contw, TTw(root), mw=(len(N_)-1)*Lw) > 0:
            G_,rc = cluster_N(root, N_, rc)  # in feature space if centroids, no B_,C_?
            if G_:  # no higher root.N_
                tL_ = [tl for n in root.N_ for l in n.L_ for tl in l.N_]  # trans-links
                if sum(tL.m for tL in tL_) * ((len(tL_)-1)*Lw) > ave*(rc+contw):  # use tL.dTT?
                    trans_cluster(root, tL_, rc+1)  # sets tTs
                    mmax_ = []
                    for F,tF in zip((root.Nt,root.Bt,root.Ct), (root.tNt,root.tBt,root.tCt)):
                        if F and tF:
                            maxF,minF = (F,tF) if F.m>tF.m else (tF,F)
                            mmax_+= [max(F.m,tF.m)]; minF.rc+=1  # rc+=rdn
                    sm_ = sorted(mmax_, reverse=True)
                    for m, (Ft, tFt) in zip(mmax_,((root.Nt, root.tNt),(root.Bt, root.tBt),(root.Ct, root.tCt))): # +rdn in 3 fork pairs
                        r = sm_.index(m); Ft.rc+=r; tFt.rc+=r  # rc+=rdn
    else:
        # primary centroid clustering:
        N_ = list({N for L in iL_ for N in L.nt if N.em})  # newly connected only
        E_ = get_exemplars(N_,rc)
        if E_ and val_(np.sum([g.dTT for g in E_],axis=0), rc+centw, TTw(root), mw=(len(E_)-1)*Lw, _TT=root.dTT) > 0:
            _,rc = cluster_C(E_,root,rc)  # forms root.Ct, may call cross_comp, incr rc
        # secondary connectivity clustering:
        L_ = sorted(iL_, key=lambda l: l.span)  # short first
        L__ =[]; seg = [L_[0]]
        for _L,L in zip(L_,L_[1:]):  # optional segment by rel ddist:
            if (L.span-_L.span) > distw * L.span: L__+=[seg]; seg=[L]  # terminate, init new seg
            else: seg+=[L]  # continue
        L__+=[seg]  # last seg
        if root.Lt.m * ((len(L__)-1)*Lw) < ave*rc: L__=[L_]  # skip nesting
        for rng, rL_ in enumerate(L__):  # rng-banded clustering
            rc+= rng + contw
            if rL_ and sum([l.m for l in rL_]) * ((len(rL_)-1)*Lw) > ave*rc:
                G_,rc = cluster_N(root, {N for L in rL_ for N in L.nt}, rc, rng)
    return G_,rc

def cluster_N(root, rN_, rc, rng=0):  # flood-fill node | link clusters, flat if rng=1

    def rroot(n): return rroot(n.root) if n.root and n.root!=root else n
    def extend_Gt(_link_, node_,cent_,link_,b_,in_):
        for L in _link_:  # spliced rim
            if L in in_: continue  # already clustered
            in_.add(L)
            for _N in L.nt:
                if _N.fin: continue
                if rng:  # cluster rN_ via top-rng roots
                    _R = rroot(_N)
                    if _R and not _R.fin:
                        if rolp(_N.rim, R.L_, R=1) > ave * rc:  # Gt is not complete?
                            node_ += [_R]; _R.fin = 1; _N.fin = 1
                            link_ += _R.L_; cent_ += _R.C_  # C_ is not rng-banded?
                else:   # flat
                    if (_N.root and _N in rN_) or _N.root==root or not _N.L_:  # link has no L_
                        node_+=[_N]; cent_+=_N.C_; _N.fin = 1
                        for l in _N.rim:
                            if l in in_: continue  # cluster by link+density:
                            (_link_ if Lnt(l) > ave*rc else b_).append(l)  # or dval?
    # root attrs:
    G_,N__,L__,Lt_,TT,lTT,C,lC, in_ = [],[],[],[],np.zeros((2,9)),np.zeros((2,9)),0,0, set()
    for N in rN_: N.fin = 0
    for N in rN_:  # form G per remaining rng N
        if N.fin or (root.root and not N.exe): continue  # no exemplars in Fg
        node_,cent_,Link_,_link_,B_ = [N],[],[],[],[]
        if rng:
            n = N; R = rroot(n)  # cluster top-rng roots
            if R and not R.fin: node_,_link_,cent_ = [R], R.L_[:], [C.root for C in R.C_]; R.fin = 1
        else:  # flat
            cent_ = N.C_
            for l in N.rim: (_link_ if Lnt(l) > ave*rc else B_).append(l)  # or dval?
        N.fin = 1; link_ = []
        while _link_:
            Link_ += _link_
            extend_Gt(_link_, node_, cent_, link_, B_, in_)
            if link_: _link_ = list(set(link_)); link_ = []  # extended rim
            else: break
        if node_:
            N_,L_,C_,B_ = list(set(node_)),list(set(Link_)),list(set(cent_)),list(set(B_))
            tt,ltt,c,lc,o = np.zeros((2,9)),np.zeros((2,9)),0,0,0
            for n in N_: tt+=n.dTT; c+=n.c; o+=n.rc  # from Ns, vs. Et from Ls?
            for l in L_: ltt+=l.dTT; lc+=l.c; o+=l.rc  # combine?
            if val_(ltt, rc+o, TTw(root),_TT=root.dTT) > 0:  # include singletons?
                G_ += [sum2G(N_,o,root, L_,C_)]
                N__+=N_; L__+=L_; Lt_+=[n.Lt for n in N_]; TT+=tt; lTT+=ltt; C+=c; lC+=lc
    if G_ and val_(TT, rc+1, TTw(root), mw=(len(G_)-1)*Lw) > 0:
        rc += 1
        root_replace(root,rc, G_,N__,L__,Lt_,TT,lTT,C,lC)
    return G_, rc

def cluster_C(E_, root, rc):  # form centroids by clustering exemplar surround via rims of new member nodes, within root

    _C_, _N_ = [],[]
    for E in E_:
        C = cent_TT( Copy_(E, root, init=2), rc, init=1); C.N_ = [E]  # all rims are within root, sequence along max w attr?
        C._N_ = [n for l in E.rim for n in l.nt if n is not E]  # core members + surround -> comp_C
        _N_ += C._N_; _C_ += [C]
        for N in C._N_: N.M, N.D, N.C, N.DTT = 0,0,0,np.zeros((2,9))  # init, they might be added to Ct.N_ later
    # reset:
    for n in set(root.N_+_N_ ): n.Ct.N_,n.mo_, n._C_,n._mo_ = [],[], [],[]  # aligned pairs, in cross_comp root
    # reform C_, refine C.N_s
    while True:
        C_,cnt,olp, mat, dif, DTT,Dm,Do = [],0,0,0,0,np.zeros((2,9)),0,eps; Ave = ave * (rc + compw)
        _Ct_ = [[c, c.m/c.c if c.m !=0 else eps, c.rc] for c in _C_]
        for _C,_m,_o in sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True):
            if _m > Ave * _o:
                C = cent_TT( sum2G(_C.N_, rc, root), rc, init=1)  # C update lags behind N_; non-local C.rc += N.mo_ os?
                _N_,_N__, mo_, M,D,O,comp,dTT,dm,do = [],[],[],0,0,0,0,np.zeros((2,9)),0,0  # per C
                for n in _C._N_:  # core+ surround
                    if C in n.Ct.N_: continue
                    dtt,_ = base_comp(C,n)  # val,olp / C:
                    m = sum(dTT[0]); d = sum(dTT[1]); dTT += dtt
                    o = np.sum([mo[0]/m for mo in n._mo_ if mo[0]>m])  # overlap = higher-C inclusion vals / current C val
                    comp += 1  # comps per C
                    if m > Ave * o:
                        _N_+=[n]; M+=m; O+=o; mo_ += [np.array([m,o])]  # n.o for convergence eval
                        _N__ += [_n for l in n.rim for _n in l.nt if _n is not n]  # +|-Ls
                        if _C not in n._C_: dm+=m; do+=o  # not in extended _N__
                    else:
                        if _C in n._C_: __m,__o = n._mo_[n._C_.index(_C)]; dm+=__m; do+=__o
                        D += abs(d)  # distinctive from excluded nodes (background)
                mat+=M; dif+=D; olp+=O; cnt+=comp  # from all comps?
                DTT += dTT
                if M > Ave * len(_N_) * O and val_(dTT, rc+O, TTw(C),1, (len(C.N_)-1)*Lw, _TT=C.dTT):
                    for n, mo in zip(_N_,mo_): n.mo_+=[mo]; n.Ct.N_+=[C]; C.Ct.N_+=[n]  # reciprocal root assign
                    C.M += M; C.D += D; C.C += comp; C.DTT += dTT
                    C.N_ += _N_; C._N_ = list(set(_N__))  # core, surround elements
                    C_ += [C]; Dm+=dm; Do+=do  # new incl or excl
                else:
                    for n in _C._N_:
                        n.exe = n.m/n.c > 2 * ave  # refine exe
                        for i, c in enumerate(n.Ct.N_):
                            if c is _C: # remove mo mapping to culled _C
                                n.mo_.pop(i); n.Ct.N_.pop(i); break
            else: break  # the rest is weaker
        if Dm/Do > Ave:  # dval vs. dolp, overlap increases as Cs may expand in each loop
            _C_ = C_
            for n in root.N_: n._C_=n.Nt.N_; n._mo_=n.mo_; n.Nt.N_,n.mo_ = [],[]  # new n.Ct.N_s, combine with vo_ in Ct_?
        else:  # converged
            break
    C_ = [C for C in C_ if val_(C.DTT, rc, TTw(C))]  # prune C_
    fC = 0
    if C_:
        for n in [N for C in C_ for N in C.N_]:
            # exemplar V + sum n match_dev to Cs, m * ||C rvals?
            n.exe = (n.D if n.typ==1 else n.M) + np.sum([mo[0]-ave*mo[1] for mo in n.mo_]) - ave
        # not revised:
        if val_(DTT, rc+olp, TTw(root),1, (len(C_)-1)*Lw, _TT=root.dTT) > 0:
            Ct = sum2G(C_,rc, root)
            _,rc = cross_comp(Ct,rc, fC=1)  # distant Cs, different attr weights?
            root.C_= C_; root.Ct=Ct; root_update(root, Ct)
            fC = 1
    return fC, rc

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
    if init: C.M, C.D, C.C, C.DTT = 0, 0, 0, np.zeros((2,9))  # init C.DTT
    # single-mode dTT, extend to 2D-3D lev cycles in H, cross-level param max / centroid?
    return C

def slope(link_):  # get ave 2nd rate of change with distance in cluster or frame?

    Link_ = sorted(link_, key=lambda x: x.span)
    dists = np.array([l.span for l in Link_])
    diffs = np.array([l.d/l.c for l in Link_])
    rates = diffs / dists
    # ave d(d_rate) / d(unit_distance):
    return (np.diff(rates) / np.diff(dists)).mean()

def Lnt(l): return ((l.nt[0].em + l.nt[1].em - l.m*2) * intw / 2 + l.m) / 2  # L.m is twice included in nt.em

def CopyF_(F, root=None):

    C = CF(dTT=deepcopy(F.dTT)); C.root = root or F.root
    for a in ['m','d','c','rc']: setattr(C,a, getattr(F,a))
    for a in ['L_','B_','C_']: setattr(C,a, copy(getattr(F,a)))
    [setattr(C,a,CopyF_(p,root=C)) for a in ['Nt','Bt','Ct'] if (p:=getattr(F,a))]
    if F.N_:
        if F.name=='Nt': C.N_ = [CopyF_(lev) for lev in F.N_]  # H levs are concat
        else:            C.N_ = copy(F.N_)  # alt roots
    return C

def Copy_(N, root=None, init=0, typ=None):

    if typ is None: typ = 2 if init else N.typ  # G.typ = 2
    C = CN(dTT=deepcopy(N.dTT), typ=typ); C.root = N.root if root is None else root
    for attr in ['m','d','c','rc']: setattr(C,attr, getattr(N,attr))
    if init: C.N_ = [N]
    else:
        if C.typ==N.typ: C.N_=list(N.N_); C.Nt = CopyF_(N.Nt,root=C) if N.Nt else N.Nt
        C.L_=list(N.L_); C.B_=list(N.B_); C.C_=list(N.C_)  # empty in init G
    if typ:  # then if typ>1?
        C.eTT=deepcopy(N.eTT)
        for attr in ['em','ed','ec','rng','fin','span','mang','sub','exe']: setattr(C,attr, getattr(N,attr))
        for attr in ['nt','baseT','box','rim','compared']: setattr(C,attr, copy(getattr(N,attr)))
        if init:  # new G
            C.yx = [N.yx]; C.angl = np.array([copy(N.angl[0]), N.angl[1]],dtype=object)  # to get mean
            if init == 1:  # else centroid
                C.L_= [l for l in N.rim if l.m>ave]; N.root = C; N.em, N.ed = vt_(N.eTT)
        else:
            C.Lt=CopyF_(N.Lt); C.Bt=CopyF_(N.Bt); C.Ct=CopyF_(N.Ct)  # empty in init G
            C.angl = copy(N.angl); C.yx = copy(N.yx)
        if hasattr(N,'mo_'): C.mo_ = deepcopy(N.mo_)
    return C

def sum2G(N_, rc, root=None, L_=[],C_=[],B_=[], dTT=None,c=1, rng=1, init=1):  # updates root if not init

    if not init: N_+=root.N_; L_+=root.L_; B_+=root.B_; C_+=root.C_
    G = sum_N_(N_,rc,root,dTT,c); G.rng=rng  # default, forms G.Nt
    if L_:
        G.L_=L_; Lt = sum2T(L_,rc,G,'Lt')  # no Lt.N_
        if N_[0].typ and G.Nt.N_:
            l0 = G.Nt.N_[0]; l0.dTT=Lt.dTT; l0.m=Lt.m; l0.d=Lt.d; l0.c=Lt.c  # vals must be level-specific
        A = np.sum([l.angl[0] for l in L_], axis=0)
        G.angl = np.array([A, np.sign(G.dTT[1] @ wTTf[1])], dtype=object)  # angle dir = d sign
    # alt forks:
    if B_: sum2T(B_,rc,G,'Bt'); G.B_=B_
    if C_: sum2T(C_,rc,G,'Ct'); G.C_=C_
    if init:  # else same ext
        yx_ = np.array([n.yx for n in N_]); yx = yx_.mean(axis=0); dy_,dx_ = (yx_-yx).T
        G.span = np.hypot(dy_,dx_).mean()  # N centers dist to G center
        G.yx = yx
    if N_[0].typ==2 and G.L_:  # else mang = 1
        G.mang = np.mean([comp_A(G.angl[0], l.angl[0])[0] for l in G.L_])
    G.m,G.d = vt_(G.dTT)
    if G.m > ave*specw:
        L_,pL_= [],[]; [L_.append(L) if L.typ==1 else pL_.append(L) for L in G.L_]
        if pL_:
            if G.m * vt_(sum([L.dTT for L in pL_]))[0] > ave*specw:
                for L in pL_:
                    link = comp_N(*L.nt, rc, L.angl[0], L.span, L.rng)
                    G.Lt.dTT+= link.dTT-L.dTT; L_+=[link]  # recompute m,d,c?
            G.L_ = L_
    return G

def sum_N_(N_, rc, root, TT=None, c=1, flat=0):  # forms G of N_|L_

    N = N_[0]; fTT= TT is not None
    lTT,lc = (N.Lt.dTT, N.Lt.c) if N.Lt else (np.zeros((2,9)),0)
    G = Copy_(N, root, init=1, typ=2)
    if fTT: G.dTT= TT; G.c= c
    n_ = list(N.N_)  # flatten core fork, alt forks stay nested
    for N in N_[1:]:
        add_N(G, N, fTT); n_+= N.N_
        if N.Lt: lTT+=N.Lt.dTT; lc+=N.Lt.c
    if flat: G.N_ = n_  # flatten N.N_, or in F.N_ only?
    else:
        G.N_ = N_
        if n_ and N.typ:  # not PP.P_, + Lt.dTT:
            m,d = vt_(lTT); G.Nt.N_.insert(0, CF(N_=n_,dTT=lTT,m=m,d=d,c=lc,root=G.Nt))
    G.m, G.d = vt_(G.dTT)
    G.rc = rc
    return G

def add_N(N, n, fTT=0, flat=0):  # flat currently not used

    n.fin = 1; n.root = N; fC = hasattr(n,'mo_')  # centroid
    if fC and not hasattr(N,'mo_'): N.mo_=[]
    _cnt,cnt = N.c,n.c; C=_cnt+cnt; N.c += n.c  # weigh contribution of intensive params
    if fC: n.rc = np.sum([mo[1] for mo in n.mo_]); N.rN_+=n.rN_; N.mo_+=n.mo_
    else:  N.rc = (N.rc*_cnt+n.rc*cnt) / C
    if not fTT: N.dTT = (N.dTT*_cnt + n.dTT*cnt) / C
    if n.typ:  # not PP
        for i, (T,t,F_,f_) in enumerate(zip((N.Nt,N.Lt,N.Bt,N.Ct), (n.Nt,n.Lt,n.Bt,n.Ct), (N.N_,N.L_,N.B_,N.C_), (n.N_,n.L_,n.B_,n.C_))):
            if f_:  # add 'tBt','tCt','tNt'?
                if flat: F_ += f_  # else stays nested, current default
                if i: T.N_ += [t]
                else:
                    for Lev,lev in zip_longest(T.N_, t.N_, fillvalue=None):
                        if lev:  # norm /C?
                            if Lev is None: N.Nt.N_ += [lev]
                            else: Lev.N_ += lev.N_; Lev.dTT+=lev.dTT; Lev.c+=lev.c  # flat
        N.span = (N.span*_cnt + n.span*cnt) / C
        A,a = N.angl[0],n.angl[0]; A[:] = (A*_cnt+a*cnt) / C  # vect only
        if isinstance(N.yx, list): N.yx += [n.yx]  # weigh by C?
    if N.typ>1: # nodes
        N.baseT = (N.baseT*_cnt+n.baseT*cnt) / C
        N.mang = (N.mang*_cnt + n.mang*cnt) / C
        N.box = extend_box(N.box, n.box)
    # if N is Fg: margin = Ns of proj max comp dist > min _Fg point dist: cross_comp Fg_?
    return N

def sum2T(T_, rc, root, nF, TT=None, c=1):  # N_ -> fork T

    T = T_[0]; fV = TT is None
    F = CF(root=root); T.root=F  # no L_,B_,C_,Nt,Bt,Ct yet
    if fV: F.dTT=T.dTT; F.c=T.c
    else:  F.dTT=TT; F.c=c
    for T in T_[1:]: add_T(F,T, nF,fV)
    F.m, F.d = vt_(F.dTT)
    F.rc=rc; setattr(root, nF,F)
    if nF in ('Bt','Ct'): F.N_ = list(T_); root_update(root,F)  # else external root replace, N_, Nt.N_ insert
    return F

def add_T(F,T, nF, fV=1):
    if nF=='Nt' and F.N_:
        F.N_[0].N_ += T.N_  # top level = flattened T.N_s
        for Lev,lev in zip_longest(F.N_[1:], T.Nt.N_):  # deeper levels
            if lev:
                if Lev: Lev.N_+=lev.N_; Lev.dTT+=lev.dTT; Lev.c+=lev.c
                else:   F.N_ += [CopyF_(lev, root=F)]
    T.root=F
    if fV: F.dTT+=T.dTT; F.c+=T.c

def root_update(root, T):  # value attrs only?

    _c,c = root.c,T.c; C = _c+c; root.c = C  # c is not weighted, min(_lev.c,lev.c) if root is link?
    root.dTT = (root.dTT*_c + T.dTT*c) /C
    root.rc = (root.rc*_c + T.rc*c) /C
    cent_TT(root, root.rc)
    if root.root: root_update(root.root, T)   # upward recursion, batch in root?

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
    if fi>1:
        H.root.node_ = H.node_
    # more advanced ordering: dH | H as medoid cluster of layers, nested cent_TT across layers?

def eval(V, weights):  # conditional progressive eval, with default ave in weights[0]
    W = 1
    for w in weights:
        W *= w
        if V < W: return 0
    return 1

def PP2N(PP):

    P_, L_, B_, verT, latT, A, S, box, yx, m, d, c = PP
    baseT = np.array(latT[:4])
    [mM, mD, mI, mG, mA, mL], [dM, dD, dI, dG, dA, dL] = verT  # re-pack in dTT:
    dTT = np.array([ np.array([mM, mD, mL, mI, mG, mA, mL, mL / 2, eps]),  # extA=eps
                     np.array([dM, dD, dL, dI, dG, dA, dL, dL / 2, eps])])
    y,x,Y,X = box; dy,dx = Y+1-y, X+1-x
    A = np.array([np.array(A), np.sign(dTT[1] @ wTTf[1])], dtype=object)  # append sign
    PP = CN(typ=0, N_=P_,L_=L_,B_=B_,dTT=dTT,m=m,d=d,c=c, baseT=baseT,box=box,yx=yx,angl=A,span=np.hypot(dy/2,dx/2), root=1)  # set root in trace_edge
    for P in P_: P.root = PP  # empty Nt, Bt, Ct?
    return PP

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

def proj_TT(L, cos_d, dist, rc, pTT, fdec=0, frec=0):  # accumulate link pTT with iTT or eTT internally

    dec = dist if fdec else ave ** (1 + dist / L.span)  # ave: match decay rate / unit distance
    TT = np.array([L.dTT[0] * dec, L.dTT[1] * cos_d * dec])
    cert = abs(val_(TT,rc) - ave)  # approximation
    if cert > ave:
        pTT+=TT; return  # certainty margin = ave
    if not frec:  # non-recursive
        for lev in L.Nt.N_:  # refine pTT
            proj_TT(lev, cos_d, dec, rc+1, pTT, fdec=1, frec=1)
    pTT += TT  # L.dTT is redundant to H, neither is redundant to Bt,Ct
    for TT in [L.Bt.dTT if L.Bt else None, L.Ct.dTT if L.Ct else None]:  # + trans-link tNt, tBt, tCt?
        if TT is not None:
            pTT += np.array([TT[0] * dec, TT[1] * cos_d * dec])

def proj_N(N, dist, A, rc):  # arg rc += N.rc+contw, recursively specify N projection val, add pN if comp_pN?

    cos_d = (N.angl[0].dot(A) / (np.hypot(*N.angl[0]) * dist)) * N.angl[1]  # internal x external angle alignment
    iTT, eTT = np.zeros((2,9)), np.zeros((2,9))

    for L in N.L_+N.B_: proj_TT(L, cos_d, dist, L.rc+rc, iTT)  # accum TT internally
    for L in N.rim:     proj_TT(L, cos_d, dist, L.rc+rc, eTT)
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
def form_B__(N_,B_):  # assign boundary / background per node from Bt, no root update?
    def R(L):
        root=L.root
        if root:
            if isinstance(root,CF): root=None  # Bt, not bG
            elif root.typ!=0: root= R(L.root)  # PPds are not clustered
        else: _N = L.nt[0] if L.nt[1] is N else L.nt[1]; root = _N.root  # direct L mediation
        return root

    for B in B_:  # bG.rN_ += reciprocal nGs:
        bG = R(B)
        if bG:
            rN_ = list({n.root for L in bG.N_ for n in L.nt if n.root and n.root.root is not None})  # core Gs, exclude frame
            bG.rN_ = sorted(rN_, key=lambda x:(x.m/x.c), reverse=True)
    for N in N_:
        if N.sub or not N.B_: continue
        bG_,dTT, rdn = [],np.zeros((2,9)),0
        for L in N.B_:
            bG = R(L)  # replace boundary L with its root in bG.rN_ if any, else L'_N
            if bG and bG not in bG_:
                if N not in bG.rN_: bG.rN_+=[N]
                bG.rN_ = sorted(bG.rN_, key=lambda x:(x.m/x.c), reverse=True)
                rdn += bG.rN_.index(N)+1  # n stronger cores of rB
                bG_ += [bG]; dTT+=bG.dTT
        N.Bt = CF(N_=bG_, dTT=dTT,m=sum(dTT[0]),d=sum(dTT[1]), c=sum(b.c for b in N.B_),rc=rdn, root=N)

def vect_edge(tile, rV=1, wTT=None):  # PP_ cross_comp and floodfill to init focal frame graph, no recursion:

    # if wTT is not None: # more detailed feedback than rV, not implemented
    global ave,avd, aveB, Lw,distw,intw, compw,centw,contw,specw, wTTf  # /= projected V change:
    ave,avd,aveB,Lw,distw,intw,compw,centw,contw,specw,wTTf = (np.array([ave,avd,aveB,Lw,distw,intw,compw,centw,contw,specw,wTTf],dtype=object) / rV)
    blob_,G_ = tile.N_,[]
    tT = [G_,[],[],[],np.zeros((2,9)),np.zeros((2,9)),0,0]   # [G_,N_,L_,Lt_,TT,lTT,C,lc], fill in trace_edge
    for blob in blob_:
        if not blob.sign and blob.G > aveB:
            edge = slice_edge(blob, rV)
            if edge.G * ((len(edge.P_)-1)*Lw) > ave * sum([P.latT[4] for P in edge.P_]):
                PPm_ = comp_slice(edge, rV, wTTf)
                N_ = [PP2N(PPm) for PPm in PPm_]; [PP2N(PPd) for PPd in edge.link_]
                form_B__(N_, B_=[L for PPm in N_ for L in PPm.B_])  # forms PPm.B_,Bt
                if val_(np.sum([n.dTT for n in N_],0),3, mw=(len(PPm_)-1)*Lw) > 0:
                    trace_edge(N_,3, tile, tT)  # flatten, cluster B_-mediated Gs, init Nt
    if G_:
        root_replace(tile,1, *tT)  # updates tile.wTT
        if vt_(tile.dTT)[0] > ave:
            return tile

def trace_edge(N_, rc, root, tT=[]):  # cluster contiguous shapes via PPs in edge blobs or lGs in boundary/skeleton?

    L_, cT_, lTT, lc = [],set(),np.zeros((2,9)),0  # comp co-mediated Ns:
    for N in N_: N.fin = 0
    for N in N_:
        _N_ = [B for rB in N.rN_ if rB.Bt for B in rB.Bt.N_ if B is not N]  # temporary
        if N.Bt: _N_+= [rN for B in N.Bt.N_ for rN in B.rN_ if rN is not N] + [rB for rB in N.rN_]  # + node-mediated
        for _N in list(set(_N_)):  # share boundary or cores if lG with N, same val?
            cT = tuple(sorted((N.id,_N.id)))
            if cT in cT_: continue
            cT_.add(cT)
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx); Rc = rc+ (N.rc+_N.rc)/2
            Link = comp_N(_N,N, Rc, A=dy_dx, span=dist)
            if vt_(Link.dTT)[0] >ave*Rc: L_+=[Link]; lTT+=Link.dTT; lc+=Link.c
    # totals:
    Gt_,Lt_,TT,C = [],[],np.zeros((2,9)),0  # Lt_: top lev / G?
    for N in N_:  # flood-fill G per seed N
        if N.fin: continue
        N.fin=1; _N_=[N]; Gt=[]; N.root=Gt; n_,l_,tt,c = [N],[],np.zeros((2,9)),0  # Gt
        while _N_:
            _N = _N_.pop(0)
            for L in _N.rim:
                if L in L_:
                    n = L.nt[0] if L.nt[1] is _N else L.nt[1]
                    if n in N_:
                        if n.root is Gt: continue
                        if n.fin:
                            _root = n.root; n_+=_root[0]; l_+=_root[1]; tt+=_root[2]; c+=_root[3]; _root[4] = 1; Lt_+=[root.Lt]  # merge n root
                            for _n in _root[0]: _n.root = Gt
                        else:
                            n.fin=1; _N_+=[n]; n_+=[n]; l_+=[L]; TT+=n.dTT; C+=n.c  # add single n
                            if n.Lt: Lt_+=[n.Lt]  # skip PPs
                        n.root = Gt
        Gt += [n_,l_,tt,c,0]; Gt_+=[Gt]; TT+=tt; C+=c
    G_= []
    for n_,l_, tt,c, merged in Gt_:
        if not merged:
            if vt_(tt)[0] > ave*rc:
                G_ += [sum2G(n_,rc,root,l_,dTT=tt,c=c)]  # include singletons
            else:
                for N in n_: N.fin=0; N.root=root
    if val_(TT, rc+1, mw=1 if tT else (len(G_)-1)*Lw) > 0:
        rc += 1
        if tT:  # concat in tile tuple in vect_edge, ext root_replace
            for i, par in enumerate((G_,N_,L_,Lt_,TT,lTT,C,lc)): tT[i] += par
        else: root_replace(root, rc, G_,N_,L_,Lt_,TT,lTT,C,lc)
    return G_, rc

def root_replace(root, rc, G_, N_,L_,Lt_, TT,lTT,c,lc):

    root.N_= G_; root.dTT=TT+lTT; root.c=c+lc; cent_TT(root, root.rc)
    sum2T(G_,rc, root,'Nt', TT, c)  # no root_update
    sum2T(L_,rc, root,'Lt',lTT,lc); root.m,root.d = vt_(root.dTT)
    lTT, lc = np.zeros((2,9)),0  # reset for top nested lev
    for Lt in Lt_: lTT+=Lt.dTT; lc+=Lt.c
    m,d = vt_(lTT)
    root.Nt.N_.insert(0, CF(N_=N_,dTT=lTT,m=m,d=d,c=lc, root=root))  # top nested level

# frame expansion per level: cross_comp lower-window N_,C_, forward results to next lev, project feedback to scan new lower windows

def frame_H(image, iY,iX, Ly,Lx, Y,X, rV, max_elev=4, wTTf=np.ones((2,9),dtype="float")):  # all initial args set manually

    def base_tile(y,x):  # 1st level, higher levels get Fg s
        Fg = frame_blobs_root( comp_pixel( image[y:y+Ly, x:x+Lx]), rV)
        Fg = vect_edge(Fg, rV)  # form, trace PP_
        if Fg: Fg.L_=[]; cross_comp(Fg, rc=Fg.rc)
        return Fg

    def expand_lev(_iy,_ix, elev, Fg):  # seed tile is pixels in 1st lev, or Fg in higher levs

        tile = np.full((Ly,Lx),None, dtype=object)  # exclude from PV__
        PV__ = np.zeros([Ly,Lx])  # maps to current-level tile
        Fg_ = []; iy,ix =_iy,_ix; y,x = 31,31  # start at tile mean
        while True:
            if not elev:
                Fg = base_tile(iy,ix)  # 1st level or cross_comped arg tile
            if Fg and val_(Fg.dTT, Fg.rc+compw+elev, mw=(len(Fg.N_)-1)*Lw) > 0:
                tile[y,x] = Fg; Fg_ += [Fg]; dy_dx = np.array([Fg.yx[0]-y,Fg.yx[1]-x])
                pTT = proj_N(Fg, np.hypot(*dy_dx), dy_dx, elev)
                if 1- abs(val_(pTT, rc=elev) > ave):  # search in marginal predictability
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
        if Fg_ and val_(np.sum([g.dTT for g in Fg_],axis=0), np.mean([g.rc for g in Fg_])+elev, mw=(len(Fg_)-1)*Lw) > 0:
            return Fg_

    global ave,avd, Lw, intw, compw, centw, contw, distw, mW, dW
    ave,avd, Lw, intw, compw, centw, contw, distw = np.array([ave,avd, Lw, intw, compw, centw, contw, distw]) / rV

    frame = CN(box=np.array([0,0,Y,X]), yx=np.array([Y//2, X//2]))
    Fg=[]; elev=0
    while elev < max_elev:  # same center in all levels
        Fg_ = expand_lev(iY,iX, elev, Fg)
        if Fg_:  # higher-scope sparse tile
            frame = sum2G(Fg_,1,frame,init=0)
            if Fg and cross_comp(Fg, rc=elev)[0]:  #| val_? spec->tN_,tC_,tL_
                frame.N_ += [Fg]; elev += 1  # forward comped tile
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
    # search frames ( tiles inside image, initially should be 4K, or 256K panorama, won't actually work on toucan