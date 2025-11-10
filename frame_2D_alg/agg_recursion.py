import numpy as np, weakref
from copy import copy, deepcopy
from math import atan2, cos, floor, pi  # from functools import reduce
from itertools import zip_longest, combinations, chain, product;  from multiprocessing import Pool, Manager
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

Induction / pre- comp_N projection:
feedback of projected match adjusts filters to maximize next match, including coordinate filters that select new inputs.
(initially ffeedback, can be refined by cross_comp of co-projected patterns: "imagination, planning, action" section of part 3)

Deduction:
in feedforward, extend code via some version of Abstract Interpretation? 
or cross-comp function calls and cluster code blocks of past and simulated processes? (not yet implemented)
compression: xcomp+ cluster, and recombination: search in combinatorial space, ~ ffeedback in real space?

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
        n.fi = kwargs.get('fi', 1)  # if G else 0, fd_: list of forks forming G?
        n.nt = kwargs.get('nt',[])  # nodet, empty if fi
        n.H  = kwargs.get('H', [])  # empty if N_:
        n.N_, n.L_ = kwargs.get('N_',[]), kwargs.get('L_')  # core attrs
        n.dTT = kwargs.get('dTT',np.zeros((2,9)))  # sum N_ dTT: m_,d_ [M,D,n, I,G,a, L,S,A], separate total,L_ TTs?
        n.m,  n.d, n.c = kwargs.get('m',0), kwargs.get('d',0), kwargs.get('c',0)  # sum dTT
        n.rim = kwargs.get('rim',[])  # external links, rng-nest?
        n.eTT = kwargs.get('eTT',np.zeros((2,9)))  # sum rim dTT
        n.em, n.ed, n.ec = kwargs.get('em',0),kwargs.get('ed',0),kwargs.get('ec',0)  # sum eTT
        n.rc  = kwargs.get('rc', 1)  # redundancy to ext Gs, ave in links?
        n.B_, n.C_, n.R_ = kwargs.get('B_',[]),kwargs.get('C_',[]),kwargs.get('R_',[])  # secondary dlink_,cent_,reciprocal root_
        n.Bt, n.Ct = kwargs.get('Bt',[]), kwargs.get('Ct',[])  # optional?
        n.baseT = kwargs.get('baseT',np.zeros(4))  # I,G,A: not ders, not in links?
        n.yx    = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.box   = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.span  = kwargs.get('span',0) # distance in nodet or aRad, comp with baseT and len(N_), not additive?
        n.angl  = kwargs.get('angl',[np.zeros(2),0])  # (dy,dx),dir, sum from L_
        n.mang  = kwargs.get('mang',1)  # ave match of angles in L_, =1 in links
        n.root  = kwargs.get('root',None)  # immediate
        n.rng = kwargs.get('rng',1)  # or med: loop count in comp_node_|link_
        n.sub = 0  # full-composition depth relative to top-composition peers
        n.fin = kwargs.get('fin',0)  # clustered, temporary
        n.exe = kwargs.get('exe',0)  # exemplar, temporary
        n.compared = set()
        # n.fork_tree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def zN_(n):  # get 1st level N_
        return n.N_ if n.N_ else n.H[0].zN_()
    def __bool__(n): return bool(n.N_)

class Cn(CBase):  # light CN for non-locals: H levels, Bt, Ct, nG, Bt?
    name = "Nt"
    def __init__(n, **kwargs):
        super().__init__()
        n.H = kwargs.get('H', [])  # empty if N_, only for nested root H levels
        n.N_, n.L_ = kwargs.get('N_',[]),kwargs.get('L_',[])  # principals
        n.dTT = kwargs.get('dTT',np.zeros((2,9)))  # separate N_,L_ TTs?
        n.m,  n.d, n.c = kwargs.get('m',0), kwargs.get('d',0), kwargs.get('c',0)  # sum dTT
        n.R_ =  kwargs.get('R_', [])
        n.B_, n.C_, n.R_ = kwargs.get('B_', []), kwargs.get('C_', []), kwargs.get('R_', [])  # sub-forks?
        n.rc = kwargs.get('rc',1)  # redundancy
        n.fi = kwargs.get('fi',1)  # for compatibility?
        n.root = kwargs.get('root', None)  # immediate

    def zN_(n):  # get 1st level N_
        return n.N_ if n.N_ else n.H[0].zN_()
    def __bool__(n): return bool(n.N_) ^ bool(n.H)
    # is_H(n: Cn) := bool(n.H); is_leaf(n: Cn) := bool(n.N_)

ave, avd, arn, aI, aS, aveB, aveR, Lw, intw, compw, centw, contw = .3, .2, 1.2, 100, 5, 100, 3, 5, 2, 5, 10, 15  # value filters + weights
dec = ave / (ave+avd)  # match decay per unit dist
adist, amed, distw, medw, specw = 10, 3, 2, 2, 2  # cost filters + weights, add alen?
wM, wD, wc, wG, wL, wI, wS, wa, wA = 10, 10, 20, 20, 5, 20, 2, 1, 1  # der params higher-scope weights = reversed relative estimated ave?
mW = dW = 9; wTTf = np.ones((2,9))  # fb weights per dTT, adjust in agg+
wY = wX = 64; wYX = np.hypot(wY,wX)  # focus dimensions

''' Core process per agg level, as described in top docstring:

- Cross-comp nodes, evaluate incremental-derivation cross-comp of new >ave difference links, recursively. 
- Select exemplar nodes, links, centroid-cluster their rim nodes, selectively spreading out within a frame. 

- Connectivity-cluster select exemplars/centroids by >ave match links, correlation-cluster links by >ave diff.
- Form complemented (core+contour) clusters for recursive higher-composition cross_comp. 

- Forward: extend cross-comp and clustering of top clusters across frames, re-order centroids by eigenvalues.
- Feedback coords to bottom level or prior-level in parallel pipelines, filter updates in more coarse cycles '''

def val_(TT, rc, fi=1, mw=1, rn=.5, _TT=None):  # m,d eval per cluster, rn = n / (n+_n), .5 for equal weight _dTT?

    t_ = np.abs(TT[0]) + np.abs(TT[1])  # not sure about abs m_
    rv = TT[0] / (t_+eps) @ wTTf[0] if fi else TT[1] / (t_+eps) @ wTTf[1]
    if _TT is not None:
        _t_ = np.abs(_TT[0]) + np.abs(_TT[1])
        _rv = _TT[0] / (_t_+eps) @ wTTf[0] if fi else _TT[1] / (_t_+eps) @ wTTf[1]
        rv = rv * (1-rn) + _rv * rn  # add borrowed root|boundary alt fork val?

    return rv * mw - (ave if fi else avd) * rc

def cross_comp(root, rc, fC=0, fT=0):  # rng+ and der+ cross-comp and clustering, fT: convert return to tuple

    N_, mL_,mTT, dL_,dTT = comp_C_(root.zN_(),rc) if fC else comp_N_(root.zN_(),rc)  # rc: redundancy+olp, fi=1|0
    nG, Bt = [], []
    if fC< 2 and dL_ and val_(dTT, rc+compw, fi=0, mw=(len(dL_)-1)*Lw) > avd:  # comp dL_| dC_, not ddC_
        Bt = cross_comp(CN(N_=dL_,root=root), rc+compw+1, fC*2, fT=1)  # trace_edge via nt s?
    # m fork:
    if len(mL_) > 1 and val_(mTT, rc+compw, mw=(len(mL_)-1)*Lw) > 0:
        for n in N_: n.em = sum([l.m for l in n.rim]) / len(n.rim)  # tentative before val_
        nG = Cluster(root, mL_, rc, fC)  # fC=0: get_exemplars, cluster_C, rng connect cluster
        if nG:  # batched H extension
            rc += nG.rc  # redundant clustering layers
            if Bt:
                form_B__(nG, Bt)  # add boundary to N, N to Bg R_s
                if val_(mTT, rc+3+contw, mw=(len(nG.N_)-1)*Lw) > 0:  # mval
                    trace_edge(nG,rc+3)  # comp Ns with shared N.Bt
            if val_(nG.dTT, rc+compw+3, mw=(len(nG.N_)-1)*Lw, _TT=mTT) > 0:
                nG = cross_comp(nG, rc+3) or nG  # connec agg+, fC = 0
            nG.H = root.H + [root] + nG.H  # nG.H is higher composition
        elif Bt:
            nG=root; nG.B_=dL_; nG.Bt=Bt  # new boundary of old core
    if nG:
        if fT: return Cn(N_=nG.R_,dTT=nG.dTT,m=sum(nG.dTT[0]),d=sum(nG.dTT[1]),c=sum(n.c for n in nG.N_),rc=nG.rc,root=nG)  # Bt
        else:  return nG  # replace root

def form_B__(G, bG):  # assign boundary / background per node from Bt tuple

    for bg in bG.N_:  # add R_ per boundary graph, in Fg?
        R_ = list({n.root for L in bg.N_ for n in L.nt if n.root and n.root.root is not None}) # core Gs, exclude frame
        bg.R_ = sorted(R_+[G], key=lambda x:(x.m/x.c), reverse=True)

    def R(L): return L.root if L.root is None or L.root in bG.R_ else R(L.root)

    for N in G.N_:
        if N.sub or not N.B_: continue
        bg_, dTT, rdn = [], np.zeros((2,9)), 0
        for L in N.B_:
            bg = R(L)  # replace boundary L with its root of the level that contains N in root.R_?
            if bg:
                bg_ +=[bg]; dTT+=bg.dTT; rdn += bg.R_.index(N)+1  # n stronger cores of rB
                if N not in bg.R_: bg.R_ += [N]  # reciprocal core
        N.Bt = Cn(N_=bg_,dTT=dTT,m=sum(dTT[0]),d=sum(dTT[1]), c=sum(b.c for b in N.B_),rc=rdn, root=N)  # N_=R_
    G.Bt = bG

def comp_C_(C_, rc, _C_=[], fall=1):  # max attr sort to constrain C_ search in 1D, add K attrs and overlap?

    mL_,mTT, dL_,dTT = [],np.zeros((2,9)), [],np.zeros((2,9)); out_ = []
    if fall:
        if _C_: pairs = product(C_,_C_); C_ += [C for C in _C_ if C not in C_]
        else:   pairs = combinations(C_, r=2)
        for _C, C in pairs:
            dtt = comp_derT(_C.dTT[1], C.dTT[1])
            m_,d_ = dtt; ad_ = np.abs(d_); t_ = m_ + ad_ + eps  # ~ max comparand
            m,d,c = m_/t_ @ wTTf[0], ad_/t_ @ wTTf[1], min(_C.c,C.c)
            dC = CN(nt=[_C,C], m=m, d=d, c=c, dTT=dtt, span=np.hypot(*_C.yx-C.yx))
            _C.rim += [dC]; C.rim += [dC]
            if   dC.m > ave*(contw+rc) > 0: mL_+=[dC]; mTT+=dC.dTT; out_ += [_C,C]
            elif dC.d > avd*(contw+rc) > 0: dL_+=[dC]; dTT+=dC.dTT  # not in out_?
    else:
        # sort,select along eigenvector, not implemented yet
        for C in C_: C.compared =set()
        C_ = sorted(C_, key=lambda C: C.dTT[0][np.argmax(wTTf[0]+wTTf[1])])  # max weight m
        for j in range( len(C_)-1):
            _C = C_[j]; C = C_[j+1]
            if _C in C.compared: continue
            dy_dx = _C.yx-C.yx; dist = np.hypot(*dy_dx)
            Link = comp_N(_C,C, rc, A=dy_dx, span=dist, rng=1)  # or use comp_derT?
            if   Link.m > ave*(contw+rc) > 0: mL_+=[Link]; mTT+=Link.dTT; out_ += [_C,C]
            elif Link.d > avd*(contw+rc) > 0: dL_+=[Link]; dTT+=Link.dTT  # not in out_?

    return list(set(out_)), mL_,mTT, dL_,dTT

def comp_N_(iN_,rc,_iN_=[]):

    def proj_V(_N,N, dist, Ave, pVt_):  # _N x N induction

        iV = (_N.m+N.m)/2 * dec**(dist/((_N.span+N.span)/2)) - Ave
        eV = sum([l.m * dec**(dist/l.span) - Ave for l in _N.rim+N.rim])
        V = iV + eV
        if abs(V) > Ave: return V  # +|-
        elif eV * ((len(pVt_)-1)*Lw) > specw:  # spec over rim, nested spec N_, not L_
            eTT = np.zeros((2,9))  # comb forks?
            for _dist,_dy_dx,__N,_V in pVt_:
                pTT,pV = proj_N(N,_dist,_dy_dx, rc)
                if pV>0: eTT += pTT  # +ve only?
                pTT.pV = proj_N(N,_dist,-_dy_dx, rc)
                if pV>0: eTT += pTT
            return iV + val_(eTT,rc)
        else: return V

    N_, L_,mTT,B_,dTT = [],[],np.zeros((2,9)),[],np.zeros((2,9))
    for i, N in enumerate(iN_):  # form unique all-to-all pre-links
        N.pL_ = []
        for _N in _iN_ if _iN_ else iN_[i+1:]:  # optional _iN_ as spec
            if _N.sub != N.sub: continue  # or comp x composition?
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx)
            N.pL_ += [[dist, dy_dx, _N]]
        N.pL_.sort(key=lambda x: x[0])  # proximity prior
    for N in iN_:
        pVt_ = []  # [dist, dy_dx, _N, V]
        for dist, dy_dx, _N in N.pL_:  # rim angl not canonic
            O = (N.rc +_N.rc) / 2; rc+=O; Ave = ave * rc
            V = proj_V(_N,N, dist, Ave, pVt_)
            if V > Ave:
                Link = comp_N(_N,N, O+rc, A=dy_dx, span=dist)
                if   Link.m > ave*(contw+rc): L_+=[Link]; mTT+=Link.dTT; N_ += [_N,N]  # combined CN dTT and L_
                elif Link.d > avd*(contw+rc): B_+=[Link]; dTT+=Link.dTT  # no overlap to simplify
                pVt_ += [[dist, dy_dx, _N, Link.m-ave*rc]]  # for distant rim eval
            else:
                break  # no induction
    return list(set(N_)), L_,mTT, B_,dTT

def comp_N(_N,N, rc, A=np.zeros(2), span=None, rng=1):  # compare links, optional angl,span,dang?

    TT,rn = base_comp(_N, N)
    baseT = (rn*_N.baseT+N.baseT) /2  # not new, for fi=0 base_comp
    yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])  # ext
    angl = [A, np.sign(TT[1] @ wTTf[1])]  # canonic direction
    Link = CN(fi=0, dTT=TT, nt=[_N,N], N_=_N.N_+N.N_, c=min(N.c,_N.c), baseT=baseT, yx=yx, box=box, span=span, angl=angl, rng=rng, rc=rc)
    if N.fi==1 and val_(TT,rc) > 0:  # not PP | Fg | link
        comp_sub(_N,N, rc, Link)
    else:  # not done in comp_sub
        Link.m = val_(Link.dTT,rc); Link.d = val_(Link.dTT,rc,fi=0)  # added in comp_sub
    for n, _n in (_N,N), (N,_N):  # if rim-mediated comp: reverse dir in _N.rim: rev^_rev?
        n.rim += [Link]; n.eTT += TT; n.ec += Link.c; n.compared.add(_n)  # or conditional n.eTT / rim later?
    return Link

# draft
def comp_sub(_N,N, rc, root):  # unpack node trees down to numericals and compare them

    _H, H = _N.H,N.H; TT = np.zeros((2,9))
    if _H and H:
        dH = []; C = 0
        for _lev,lev in zip(_H, H):
            C += min(_lev.c, lev.c)
            tt = comp_derT(_lev.dTT[1], lev.dTT[1]*rc); TT += tt  # default
            dlev = Cn(dTT=tt, m=sum(tt[0]), d=sum(tt[1]), root=root, c=min(_lev.c,lev.c), rc=min(_lev.rc,lev.rc))
            if val_(tt,rc) > 0:  # sub-recursion:
                comp_sub(_lev,lev, rc, dlev)
            dH += [dlev]
    else:
        if N.H: N = N.H[0]  # comp 1st lev only
        if _N.H: _N = _N.H[0]
        N_,nL_,mTT,B_,dTT = comp_N_(_N.zN_(),rc,N.zN_()); TT += mTT+dTT
        dlev = Cn(N_=nL_,c=min(_N.c,N.c), root=root, rc=rc)  # not sure
        dH = [dlev]
    # not Bt,Ct? always add in last lev only?
    if _N.B_ and N.B_:
        N_,bL_,mTT,B_,dTT = comp_N_(_N.B_,rc, N.B_); tt = mTT+dTT
        dlev.B_ = bL_; dlev.dTT+=tt; TT+=tt
    if _N.C_ and N.C_:
        N_,cL_,mTT,B_,dTT = comp_C_(_N.C_,rc, N.C_); tt = mTT+dTT
        dlev.B_ = cL_; dlev.dTT+=tt; TT+=tt

    dlev.dTT=TT; dlev.m = sum(TT[0]); dlev.d = sum(TT[1])
    root.H = dH; root.dTT += TT  # root.m = val_(TT,rc); root.d = val_(TT,rc,fi=0)?

def base_comp(_N,N):  # comp Et, baseT, extT, dTT

    _M,_D,_c =_N.m,_N.d,_N.c; _I,_G,_Dy,_Dx =_N.baseT; _L = len(_N.N_)  # N_: density within span
    M, D, c  = N.m, N.d, N.c; I, G, Dy, Dx = N.baseT; L = len(N.N_)
    rn = _c/c
    _pars = np.array([_M*rn,_D*rn,_c*rn,_I*rn,_G*rn, [_Dy,_Dx],_L*rn,_N.span], dtype=object)  # Et, baseT, extT
    pars  = np.array([M,D,c, (I,aI),G, [Dy,Dx], L,(N.span,aS)], dtype=object)
    mA,dA = comp_A(_N.angl[0]*_N.angl[1], N.angl[0]*N.angl[1])
    m_,d_ = comp(_pars, pars, mA, dA)  # M,D,n, I,G,a, L,S,A
    dm_,dd_ = comp_derT(rn*_N.dTT[1], N.dTT[1])

    return np.array([m_+dm_,d_+dd_]), rn
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
        elif isinstance(p, tuple):  # massless I|S avd in p only
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

def rolp(N, _N_, R=0): # rel V of L_|N.rim overlap with _N_: inhibition|shared zone, oN_ = list(set(N.N_) & set(_N.N_)), no comp?

    n_ = set(N.zN_()) if R else {n for l in N.rim for n in l.nt if n is not N}  # nrim
    olp_ = n_ & set(_N_)
    if olp_:
        odTT = np.sum([i.dTT for i in olp_], axis=0)
        _dTT = N.dTT if R else N.dtt  # not sure
        rV = (sum(odTT[0])/odTT[0][2]) / (sum(_dTT[0])/_dTT[0][2])
        return rV * val_(N.eTT, centw)  # contw for cluster?
    else:
        return 0

def get_exemplars(N_, rc):  # get sparse nodes by multi-layer non-maximum suppression

    E_ = set()
    for rdn, N in enumerate(sorted(N_, key=lambda n:n.em, reverse=True), start=1):  # strong-first
        roV = rolp(N, E_)
        if N.em > ave * (rc+ rdn+ compw +roV):  # ave *= relV of overlap by stronger-E inhibition zones
            E_.update({n for l in N.rim for n in l.nt if n is not N and N.em > ave*rc})  # selective nrim
            N.exe = 1  # in point cloud of focal nodes
        else:
            break  # the rest of N_ is weaker, trace via rims
    return E_

# below is not revised:
def Cluster(root, iL_, rc, iC):  # generic clustering root

    def trans_cluster(root, iL_,rc):  # called from cross_comp(Fg_), others?

        dN_,dL_,dC_ = [],[],[]  # splice specs from links between Fgs in Fg cluster
        for Link in iL_: dN_+= Link.L_; dL_+= Link.B_; dC_+= Link.C_  # trans-G links
        Nt, Lt, Ct = [],[],[]
        for f_,link_,clust, fC in [(Nt,dN_,cluster_N,0),(Lt,dL_,cluster_N,0),(Ct,dC_,cluster_n,1)]:
            if link_:
                G = clust(root, link_, rc)
                if G:
                    if val_(G.dTT, rc, mw=(len(G.N_)-1)*Lw) > 0: G = cross_comp(G,rc, fC=fC) or G
                    f_[:] = [G.N_, G.dTT, G.m, G.d, G.c, G.rc]
        # trans-fork_ via trans-G links:
        return Nt, Lt, Ct
    nG = []
    if iC or not root.root:  # input Fg, no exemplars or centroid clustering, not rng-banded, may call deep_cluster?
        # base connectivity clustering
        N_, L_, i = [],[],0
        if iC < 2:  # merge similar Cs, not dCs, no recomp
            link_ = sorted(list({L for L in iL_}), key=lambda link: link.d)  # from min D
            for i, L in enumerate(link_):
                if val_(L.dTT,rc+compw, fi=0) < 0:  # merge
                    _N, N = L.nt
                    if _N is not N:  # not merged
                        add_N(_N,N, rc, froot=1)  # fin,root.rim
                        for l in N.rim: l.nt = [_N if n is N else n for n in l.nt]
                        if N in N_: N_.remove(N)  # if multiple merging
                        N_ += [_N]
                else: L_ = link_[i:]; break
            root.L_ = [l for l in root.L_ if l not in L_]  # cleanup regardless of break
        else: L_ = iL_
        N_ += list({n for L in L_ for n in L.nt})  # include merged Cs
        if val_(root.dTT, rc+contw, mw=(len(N_)-1)*Lw) > 0:
            nG = cluster_n(root, N_,rc)  # in feature space if centroids, no B_,C_?
            tL_ = [l for n in nG.N_ for l in n.L_]  # trans-links
            if sum(tL.m for tL in tL_) *((len(tL_)-1)*Lw) > ave*(rc+contw):
                tF_ = trans_cluster(nG, tL_, rc+1)
                Ft_ = []  # [[F,tF]]
                for n_, tF in zip((nG.N_, nG.B_, nG.C_), tF_):
                    dtt = np.zeros((2,9)); c,rc = 0,0  # if no Fts yet?
                    for n in n_: dtt += n.dTT; c += n.c; rc += n.rc
                    Ft_ += [[[N_, dtt, np.sum(dtt[0]), np.sum(dtt[1]), c,rc], tF]]  # cis,trans fork pairs
                mmax_ = []
                for F,tF in Ft_:  # or Bt, Ct from cross_comp?
                    if F and tF:
                        m,tm = F[2],tF[2]; maxF,minF = (F,tF) if m>tm else (tF,F)
                        mmax_+= [max(m,tm)]; minF[-1]+=1  # rc+=rdn
                sm_ = sorted(mmax_, reverse=True)
                for m, Ft in zip(mmax_,Ft_): # +rdn in 3 fork pairs
                    r = sm_.index(m); Ft[0][-1]+=r; Ft[1][-1]+=r  # rc+=rdn
        if not nG: nG = sum_N_(N_, rc, root, L_)
    else:
        # primary centroid clustering
        N_ = list({N for L in iL_ for N in L.nt if N.em})  # newly connected only
        E_ = get_exemplars(N_,rc)
        if E_ and val_(np.sum([g.dTT for g in E_],axis=0), rc+centw, N_[0].fi, (len(E_)-1)*Lw, root.dTT) > 0:
            cluster_C(E_, root, rc)
        L_ = sorted(iL_, key=lambda l: l.span)
        L__, Lseg = [], [iL_[0]]
        for _L,L in zip(L_, L_[1:]):  # segment by ddist:
            if L.span -_L.span < adist: Lseg += [L]  # or short seg?
            else: L__ += [Lseg]; Lseg = [L]
        L__ += [Lseg]
        for rng, rL_ in enumerate(L__,start=1):  # bottom-up rng-banded clustering
            rc+= rng + contw
            if rL_ and sum([l.m for l in rL_]) * ((len(rL_)-1)*Lw) > ave*rc:
                nG = cluster_N(root, rL_, rc, rng) or nG
                # top valid-rng nG
    return nG

def cluster_n(root, iC_, rc):  # simplified flood-fill, for C_ or trans_N_

    def extend_G(_link_, node_,cent_,link_,b_,in_):
        for L in _link_:  # spliced rim
            if L in in_: continue  # already clustered
            in_.add(L)
            for _N in L.nt:
                if not _N.fin and _N in iC_:
                    node_ += [_N]; cent_ += _N.C_; _N.fin = 1
                    for l in _N.rim:
                        if l in in_: continue  # cluster by link+density:
                        if Lnt(l) > ave*rc: link_ += [l]
                        else: b_ += [l]
    G_, in_ = [], set()
    for C in iC_: C.fin = 0
    for C in iC_:  # form G per remaining C
        node_,link_,L_,B_ = [C],[],[],[]
        cent_ = C.C_[0][:] if C.C_ else []
        _L_= [l for l in C.rim if Lnt(l) > ave*rc]  # link+nt eval
        while _L_:
            extend_G(_L_, node_, cent_, L_, B_, in_)  # _link_: select rims to extend G:
            if L_: link_ += L_; _L_ = list(set(L_)); L_ = []
            else:  break
        if node_:
            N_= list(set(node_)); L_= list(set(link_)); C_ = list(set(cent_))
            dTT,olp = np.zeros((2,9)), 0
            for n in N_: olp += n.rc  # from Ns, vs. Et from Ls?
            for l in L_: dTT += l.dTT
            if val_(dTT,rc+olp, mw=(len(N_)-1)*Lw,_TT=root.dTT) > 0:
                G_ += [sum_N_(N_,olp,root, L_,C_)]
            elif n.fi:
                G_ += N_
    if G_: return sum_N_(G_, rc, root)  # nG

def cluster_N(root, rL_, rc, rng=1):  # flood-fill node | link clusters

    def rroot(n): return rroot(n.root) if n.root and n.root!=root else n

    def extend_Gt(_link_, node_, cent_, link_, b_, in_):
        for L in _link_:  # spliced rim
            if L in in_: continue  # already clustered
            in_.add(L)
            for _N in L.nt:
                if _N.fin: continue
                if not N.root or N.root == root or not N.L_:  # not rng-banded
                    node_ += [_N]; cent_ += _N.R_; _N.fin = 1
                    for l in _N.rim:
                        if l in in_: continue  # cluster by link+density:
                        if l in rL_:
                            if Lnt(l) > ave*rc: link_ += [l]
                            else: b_ += [l]
                else:  # cluster top-rng roots
                    _n = _N; _R = rroot(_n)
                    if _R and not _R.fin:
                        if rolp(N, link_, R=1) > ave * rc:
                            node_ += [_R]; _R.fin = 1; _N.fin = 1
                            link_ += _R.L_; cent_ += _R.rC_
    G_, in_ = [], set()
    rN_ = {N for L in rL_ for N in L.nt}
    for n in rN_: n.fin = 0
    for N in rN_:  # form G per remaining rng N
        if N.fin or (root.root and not N.exe): continue  # no exemplars in Fg
        node_,cent_,Link_,_link_,B_ = [N],[],[],[],[]
        if rng==1 or not N.root or N.root==root:  # not rng-banded
            cent_ = [C.root for C in N.C_]
            for l in N.rim:
                if l in rL_:  # curr rng
                    if Lnt(l) > ave*rc: _link_ += [l]
                    else: B_ += [l]  # or dval?
        else: # N is rng-banded, cluster top-rng roots
            n = N; R = rroot(n)
            if R and not R.fin: node_,_link_,cent_ = [R], R.L_[:], [C.root for C in R.C_]; R.fin = 1
        N.fin = 1; link_ = []
        while _link_:
            Link_ += _link_
            extend_Gt(_link_, node_, cent_, link_, B_, in_)
            if link_: _link_ = list(set(link_)); link_ = []  # extended rim
            else:     break
        if node_:
            N_, L_, C_, B_ = list(set(node_)), list(set(Link_)), list(set(cent_)), list(set(B_))
            dTT,olp = np.zeros((2,9)), 0
            for n in N_: olp += n.rc  # from Ns, vs. Et from Ls?
            for l in L_: dTT += l.dTT
            if val_(dTT, rc+olp, 1, (len(N_)-1)*Lw, root.dTT) > 0:
                G_ += [sum_N_(N_, olp, root, L_, C_, B_, rng)]
            elif n.fi:  # L_ is preserved anyway
                for n in N_: n.sub += 1
                G_ += N_
    if G_: return sum_N_(G_, rc, root)  # nG, skip attrs in sum_N_?

def cluster_C(E_, root, rc):  # form centroids by clustering exemplar surround via rims of new member nodes, within root

    _C_, _N_, cG = [],[],[]
    for E in E_:
        C = cent_attr( Copy_(E, rc, root, init=2), rc); C.N_ = [E]  # all rims are within root, sequence along max w attr?
        C._N_ = [n for l in E.rim for n in l.nt if n is not E]  # core members + surround -> comp_C
        _N_ += C._N_; _C_ += [C]
        for N in C._N_: N.M, N.D, N.C, N.DTT = 0, 0, 0, np.zeros((2,9))  # init, they might be added to rC_ later
    # reset:
    for n in set(root.N_+_N_ ): n.rC_,n.mo_, n._C_,n._mo_ = [],[], [],[]  # aligned pairs, in cross_comp root
    # reform C_, refine C.N_s
    while True:
        C_,cnt,olp, mat, dif, DTT,Dm,Do = [],0,0,0,0,np.zeros((2,9)),0,eps; Ave = ave * (rc + compw)
        _Ct_ = [[c, c.m/c.c if c.m !=0 else eps, c.rc] for c in _C_]
        for _C,_m,_o in sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True):
            if _m > Ave * _o:
                C = cent_attr( sum_N_(_C.N_, rc, root, fC=1), rc); C.R_ = []  # C update lags behind N_; non-local C.rc += N.mo_ os?
                _N_,_N__, mo_, M,D,O,comp,dTT,dm,do = [],[],[],0,0,0,0,np.zeros((2,9)),0,0  # per C
                for n in _C._N_:  # core+ surround
                    if C in n.rC_: continue
                    dtt, _ = base_comp(C,n, fC=1)  # val,olp / C:
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
                if M > Ave * len(_N_) * O:
                    for n, mo in zip(_N_,mo_): n.mo_+=[mo]; n.R_+=[C]; C.R_+=[n]  # bilateral assign
                    C.M += M; C.D += D; C.C += comp; C.DTT += dTT
                    C.N_ += _N_; C._N_ = list(set(_N__))  # core, surround elements
                    C_ += [C]; Dm+=dm; Do+=do  # new incl or excl
                else:
                    for n in _C._N_:
                        n.exe = n.m/n.c > 2 * ave  # refine exe, Et vals are already normalized, Et[2] no longer needed for eval?
                        for i, c in enumerate(n.R_):
                            if c is _C: # remove mo mapping to culled _C
                                n.mo_.pop(i); n.R_.pop(i); break
            else: break  # the rest is weaker
        if Dm/Do > Ave:  # dval vs. dolp, overlap increases as Cs may expand in each loop
            _C_ = C_
            for n in root.N_: n._C_=n.R_; n._mo_=n.mo_; n.R__,n.mo_ = [],[]  # new n.rC_s, combine with vo_ in Ct_?
        else:  # converged
            break
    C_ = [C for C in C_ if val_(C.DTT, rc)]  # prune C_
    if C_:
        for n in [N for C in C_ for N in C.N_]:
            n.exe = [n.M, n.D][n.fi] + np.sum([mo[0] - ave*mo[1] for mo in n.mo_]) - ave  # exemplar V + summed n match_dev to Cs
            # m * ||C rvals?
        if val_(DTT,1,(len(C_)-1)*Lw, rc+olp, _TT=root.dTT) > 0:
            cG = cross_comp(sum_N_(C_,rc,root), rc, fC=1)  # distant Cs, different attr weights?
        root.C_ = [cG.N_ if cG else C_, DTT]

def cent_attr(C, rc):  # weight attr matches | diffs by their match to the sum, recompute to convergence

    wTT = []  # Cs can be fuzzy only to the extent that their correlation weights are different?
    tot = C.dTT[0] + np.abs(C.dTT[1])  # m_* align, d_* 2-align for comp only?

    for fd, derT, wT in zip((0,1), C.dTT, wTTf):
        if fd: derT = np.abs(derT)  # ds
        _w_ = np.ones(9)  # weigh by feedback:
        val_ = derT / tot * wT  # signed ms, abs ds
        V = np.sum(val_)
        while True:
            mean = max(V / max(np.sum(_w_), eps), eps)
            inverse_dev_ = np.minimum(val_/mean, mean/val_)  # rational deviation from mean rm in range 0:1, if m=mean
            w_ = inverse_dev_ / .5  # 2/ m=mean, 0/ inf max/min, 1 / mid_rng | ave_dev?
            w_ *= 9 / np.sum(w_)  # mean w = 1, M shouldn't change?
            if np.sum(np.abs(w_-_w_)) > ave*rc:
                V = np.sum(val_ * w_)
                _w_ = w_
            else: break  # weight convergence
        wTT += [_w_]
    C.wTT = np.array(wTT)  # replace wTTf
    C.M, C.D, C.C, C.DTT = 0, 0, 0, np.zeros((2,9))  # init C.DTT
    # extend to lay cycles in derH
    return C

def slope(link_):  # get ave 2nd rate of change with distance in cluster or frame?

    Link_ = sorted(link_, key=lambda x: x.span)
    dists = np.array([l.span for l in Link_])
    diffs = np.array([l.d/l.c for l in Link_])
    rates = diffs / dists
    # ave d(d_rate) / d(unit_distance):
    return (np.diff(rates) / np.diff(dists)).mean()

def Lnt(l): return ((l.nt[0].em + l.nt[1].em - l.m*2) * intw / 2 + l.m) / 2  # L.m is twice included in nt.em

def copy_(H, root):  # simplified
    cH = CN(dTT=deepcopy(H.dTT), root=root, rc=H.rc, m=H.m,d=H.d,c=H.c)  # summed across H
    cH.H = [copy_(lay,cH) for lay in H.H]
    cH.F_ = [copy(f) for f in cH.fork_]
    return cH

def Copy_(N, rc=1, root=None, init=0):

    C = CN(root=root, dTT=deepcopy(N.dTT))
    for attr in ['nt', 'baseT', 'box', 'rim', 'B_', 'C_', 'R_']: setattr(C, attr, copy(getattr(N, attr)))
    for attr in ['m','d','c','em','ed','ec','rc','rng','fin','span','mang']: setattr(C, attr, getattr(N, attr))
    for attr in ['Bt','Ct']: setattr(C, attr, Copy_(getattr(N, attr)))
    if init: # new G
        C.N_ = [N]; C.H = copy(N.H); C.yx = [N.yx]; C.angl = N.angl[0]  # to get mean (H should copy N.H now even with init?)
        if init==1:  # else centroid
            C.L_= [l for l in N.rim if l.m>ave]; N.root = C
            N.em, N.ed = val_(N.eTT,rc), val_(N.eTT,rc,fi=0)
    else:
        C.N_,C.L_,C.H = list(N.N_),list(N.L_),copy_(N.H)
        C.angl = N.angl; N.root = root or N.root; C.yx = copy(N.yx); C.fi = N.fi  # else 1
    return C

def sum_N_(N_,rc, root=None, L_=[],C_=[],B_=[], rng=1,fC=0): # sum node,link attrs in graph, aggH in agg+ or player in sub+

    G = Copy_(N_[0],rc, root, init=fC+1)
    G.rc=rc; G.rng=rng; G.N_,G.L_,G.C_,G.B_ = N_,L_,C_,B_; ang=np.zeros(2)
    for N in N_[1:]:
        add_N(G,N, rc, init=1, fC=fC, froot=not fC)  # no need for froot?
    Lev = Cn(root=G)  # sum L.H into Lev
    for L in L_:
        for lev in L.H: add_sub(Lev,lev)  # combine L.H into Lev
        G.dTT+=L.dTT; ang+=L.angl[0]; add_sub(G,L)  # L.H added separately
    G.H += [Lev]  # nested level
    yx_ = G.yx; yx = np.mean(yx_,axis=0); dy_,dx_ = (yx_-yx).T
    G.yx = yx; G.span = np.hypot(dy_,dx_).mean()  # N centers dist to G center
    G.angl = np.array([ang, np.sign(G.dTT[1] @ wTTf[1])], dtype=object)
    # canonical angle dir = mean diff sign
    if N_[0].fi and len(L_) > 1:  # else default mang = 1
        G.mang = np.sum([ comp_A(G.angl[0], l.angl[0]) for l in L_]) / len(L_)
    G.m = val_(G.dTT,rc)
    G.d = val_(G.dTT,rc,fi=0)
    return G

def add_N(N, n, rc, init=0, fC=0, froot=0):  # rn = n.n / mean.n

    for Par,par in zip((N.baseT,N.dTT), (n.baseT,n.dTT)):
        Par += par  # extensive params, scale with c
    _cnt,cnt = N.c,n.c; Cnt = _cnt+cnt
    # weigh contribution of intensive params:
    add_sub(N, n)  # H or Nt,Bt,Ct
    N.mang = (N.mang*_cnt + n.mang*cnt) / Cnt
    N.span = (N.span*_cnt + n.span*cnt) / Cnt
    N.rc = (N.rc*_cnt + n.rc*cnt) / Cnt
    N.box = extend_box(N.box, n.box)
    N.c += n.c  # cnt / mass
    if init:  # N is G
        n.em, n.ed = val_(n.eTT,rc), val_(n.eTT,rc,fi=0); N.yx += [n.yx]
        N.angl = (N.angl*_cnt + n.angl[0]*cnt) / Cnt  # vect only
    else:     # merge n
        for node in n.N_: node.root = N; N.N_ += [node]
        A,a = N.angl[0],n.angl[0]; A[:] = (A*_cnt+a*cnt) / Cnt
        N.L_ += n.L_; N.rim += n.rim  # no L.root, rims can't overlap?
    n.C_ += [C for C in n.C_ if C not in N.C_]  # centroids, not in root Ct
    n.B_ += [B for B in n.B_ if B not in N.B_]  # high-diff links, not in root Bt?
    # if N is Fg: margin = Ns of proj max comp dist > min _Fg point dist: cross_comp Fg_?
    if fC:
        n.rc = np.sum([mo[1] for mo in n._mo_]); N.rC_ += n.rC_; N.mo_ += n.mo_
    if froot: n.fin = 1; n.root = N
    return N

def add_sub(N, n):  # add n.H|n.N_, n.Bt,n.Ct to N, analogous to comp_sub

    N.dTT += n.dTT
    if n.H:
        if n.fi:  # link H is summed into new level separately
            for Lev, lev in zip_longest(N.H,n.H):
                if Lev and lev: add_sub(Lev,lev)
                elif lev:
                    N.H += [copy_(lev, N)]
    else: # single lev not in H
        N.N_ += [n for n in n.N_ if n not in N.N_]
    for Ft, ft in zip((N.Bt, N.Ct), (n.Bt, n.Ct)):  # not redundant to H
        if ft:
            Ft.N_ += [n for n in ft.N_ if n not in Ft.N_]
            Ft.R_ += [n for n in ft.R_ if n not in Ft.R_]
            Ft.dTT += ft.dTT
            Ft.m = sum(Ft.dTT[0])  # recompute m
            Ft.d = sum(Ft.dTT[1])  # recompute d
            C = Ft.c + ft.c; Ft.c = C
            Ft.rc = (Ft.rc*Ft.c + ft.rc*ft.c)/C  # weigh rc

def extend_box(_box, box):
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return min(y0,_y0), min(x0,_x0), max(yn,_yn), max(xn,_xn)

# not used directly
def sort_H(H, fi):  # lev.rc = complementary to root.rc and priority index in H, if selective and aligned

    i_ = []  # priority indices
    for i, lay in enumerate(sorted(H.node_, key=lambda lay: [lay.m, lay.d][fi], reverse=True)):
        di = lay.i - i  # lay index in H
        lay.rc += di  # derR - valR
        i_ += [lay.i]
    H.i_ = i_  # H priority indices: node/m | link/d
    if fi:
        H.root.node_ = H.node_
    # more advanced ordering: dH | H as medoid cluster of layers?
    # nested cent_attr across layers?

def eval(V, weights):  # conditional progressive eval, with default ave in weights[0]
    W = 1
    for w in weights:
        W *= w
        if V < W: return 0
    return 1

def PP2N(PP):

    P_, link_, B_, verT, latT, A, S, box, yx, m, d, c = PP
    baseT = np.array(latT[:4])
    [mM, mD, mI, mG, mA, mL], [dM, dD, dI, dG, dA, dL] = verT  # re-pack in dTT:
    dTT = np.array([ np.array([mM, mD, mL, mI, mG, mA, mL, mL / 2, eps]),  # extA=eps
                     np.array([dM, dD, dL, dI, dG, dA, dL, dL / 2, eps])])
    y,x,Y,X = box; dy,dx = Y+1-y, X+1-x
    A = np.array([np.array(A), np.sign(dTT[1] @ wTTf[1])], dtype=object)  # append sign
    PP = CN(fi=2, N_=P_,B_=B_, m=m,d=d,c=c, baseT=baseT, dTT=dTT, box=box, yx=yx, angl=A, span=np.hypot(dy/2, dx/2))
    for P in PP.N_: P.root = PP
    return PP

def ffeedback(root):  # adjust filters: all aves *= rV, ultimately differential backprop per ave?

    def L_ders(Fg):  # get current-level ders: from L_ only
        dTT = np.zeros((2,9)); m, d, c = 0, 0, 0
        for n in Fg.N_:
            for l in n.L_:
                m += l.m; d += l.d; c += l.c; dTT += l.dTT
        return m, d, c, dTT

    wTTf = np.ones((2,9))  # sum dTT weights: m_,d_ [M,D,n, I,G,A, L,S,eA]: Et, baseT, extT
    rM, rD, rVd = 1,1,0
    _m, _d, _n, _dTT = L_ders(root)
    for lev in reversed(root.H):  # top-down, not lev-selective
        m ,d ,n, dTT = L_ders(lev)
        rM += (_m / _n) / (m / n)  # mat,dif change per level
        rD += (_d / _n) / (d / n)
        wTTf += np.abs((_dTT /_n) / (dTT / n))
        if lev.lH:
            # intra-level recursion in dfork
            for lH in lev.lH:
                rvd, wttf = ffeedback(lH)
                rVd += rvd; wTTf += wttf
        _m, _d, _n, _dTT = m, d, n, dTT
    return rM+rD+rVd, wTTf

def proj_focus(PV__, y,x, Fg):  # radial accum of projected focus value in PV__

    m,d,n = Fg.m, Fg.d, Fg.c
    V = (m-ave*n) + (d-avd*n)
    dy,dx = Fg.angl[0]; a = dy/ max(dx,eps)  # average link_ orientation, projection
    decay = (ave / (Fg.baseT[0]/n)) * (wYX / adist)  # base decay = ave_match / ave_template * rel dist (ave_dist is a placeholder)
    H, W = PV__.shape  # = win__
    n = 1  # radial distance
    while y-n>=0 and x-n>=0 and y+n<H and x+n<W:  # rim is within frame
        dec = decay * n
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

def proj_TT(L, cos_d, dist, rc, pTT):  # accumulate link pTT with iTT or eTT internally

    dec = ave ** (1 + dist / L.span)  # ave: match decay rate / unit distance
    TT = np.array([L.dTT[0] * dec, L.dTT[1] * cos_d * dec])
    cert = abs(val_(TT, rc) - ave)  # approximation
    if cert > ave:  # +|- certainty in ave margin
        pTT += TT; return
    if L.H:
        for lev in L.H: proj_TT(lev, cos_d, dist, rc + 1, pTT)  # accum refined pTT
    else: pTT += TT   # L.dTT is redundant to H, neither is redundant to Bt,Ct?

    for TT in [L.Bt.dTT if L.Bt else None, L.Ct.dTT if L.Ct else None]:  # + trans-link tNt, tBt, tCt?
        if TT is not None:
            pTT += np.array([TT[0] * dec, TT[1] * cos_d * dec])

def proj_N(N, dist, A, rc):  # arg rc += N.rc+contw, recursively specify N projection val, add pN if comp_pN?

    cos_d = (N.angl[0].dot(A) / (np.hypot(*N.angl[0]) * dist)) * N.angl[1]  # internal x external angle alignment
    iTT, eTT = np.zeros((2,9)), np.zeros((2,9))  # projections
    # from int,ext links, work the same?
    for L in N.L_+N.B_: proj_TT(L, cos_d, dist, L.rc+rc, iTT)  # accum iTT internally
    for L in N.rim:     proj_TT(L, cos_d, dist, L.rc+rc, eTT)

    pTT = iTT + eTT  # info_gain = N.m * average link uncertainty, all in 0:1:
    return pTT, val_(N.dTT,rc) * (1- val_(iTT+eTT, rc))

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
def vect_edge(tile, rV=1, wTTf=[]):  # PP_ cross_comp and floodfill to init focal frame graph, no recursion:

    if np.any(wTTf):
        global ave, avd, arn, aveB, aveR, Lw, adist, amed, intw, compw, centw, contw, wM,wD,wc, wI,wG,wa, wL,wS,wA
        ave, avd, arn, aveB, aveR, Lw, adist, amed, intw, compw, centw, contw = (
            np.array([ave,avd, arn,aveB,aveR, Lw, adist, amed, intw, compw, centw, contw]) / rV)  # projected value change
        wTTf = np.multiply([[wM,wD,wc, wI,wG,wa, wL,wS,wA]], wTTf)  # or dw_ ~= w_/ 2?
    Edge_ = []
    for blob in tile.N_:
        if not blob.sign and blob.G > aveB:
            edge = slice_edge(blob, rV)
            if edge.G * ((len(edge.P_)-1)*Lw) > ave * sum([P.latT[4] for P in edge.P_]):
                PPm_ = comp_slice(edge, rV, wTTf)
                Edge = sum_N_([PP2N(PPm) for PPm in PPm_],1,None); Edge.fi = 3
                if edge.link_:
                    bG = sum_N_([PP2N(PPd) for PPd in edge.link_],2, Edge)
                    form_B__(Edge,bG)  # add Edge.Bt
                    if val_(Edge.dTT,3, mw=(len(PPm_)-1)*Lw) > 0:
                        trace_edge(Edge,3)  # cluster complemented Gs via G.B_
                        if val_(bG.dTT,4, fi=0, mw=(len(bG.N_)-1)*Lw) > 0:
                            trace_edge(bG,4)  # for cross_comp in frame_H?
                Edge_ += [Edge]
    return sum_N_(Edge_,2,None)  # Fg, no root

def trace_edge(root, rc):  # cluster contiguous shapes via PPs in edge blobs or lGs in boundary/skeleton?

    N_ = root.zN_()  # clustering  is rB_|B_-mediated
    L_ = []; cT_ = set()  # comp pairs
    for N in N_: N.fin = 0
    for N in N_:
        _N_ = [B for rB in N.R_ if rB.Bt for B in rB.B_ if B is not N]  # temporary
        if N.Bt: _N_ += [rB for B in N.B_ for rB in B.R_ if rB is not N]
        for _N in list(set(_N_)):  # share boundary or cores if lG with N, same val?
            cT = tuple(sorted((N.id,_N.id)))
            if cT in cT_: continue
            cT_.add(cT)
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx); o = (N.rc+_N.rc) / 2
            Link = comp_N(_N,N, o+rc, A=dy_dx, span=dist)
            if val_(Link.dTT, contw+o+rc) > 0:
                L_+=[Link]; root.dTT+=Link.dTT
    Gt_ = []
    for N in N_:  # flood-fill G per seed N
        if N.fin: continue
        N.fin=1; _N_=[N]; Gt=[]; N.root=Gt; n_,l_,dTT,olp = [N],[],np.zeros((2,9)),0
        while _N_:
            _N = _N_.pop(0)
            for L in _N.rim:
                if L in L_:
                    n = L.nt[0] if L.nt[1] is _N else L.nt[1]
                    if n in N_:
                        if n.root is Gt: continue
                        if n.fin:
                            _root = n.root; n_+=_root[0]; l_+=_root[1]; dTT+=_root[2]; olp+=_root[3]; _root[4] = 1
                            for _n in _root[0]: _n.root = Gt
                        else: n.fin=1; _N_+=[n]; n_+=[n]; l_+=[L]; dTT+=L.dTT; olp+=L.rc
                        n.root = Gt
        Gt += [n_,l_,dTT,olp,0]; Gt_ += [Gt]
    G_= []
    for n_,l_, dtt,olp, merged in Gt_:
        if not merged and np.sum(dtt[0])/dtt[0][2] > ave*rc:  # or m/(m+d)?
            G_ += [sum_N_(n_,olp,root,l_)]  # include singletons
    for N in N_: N.fin = 0
    root.N_ = G_

# frame expansion per level: cross_comp lower-window N_,C_, forward results to next lev, project feedback to scan new lower windows

def frame_H(image, iY,iX, Ly,Lx, Y,X, rV, max_elev=4, wTTf=np.ones((2,9),dtype="float")):  # all initial args set manually

    def base_tile(y,x):  # 1st level, higher levels get Fg s

        Fg = frame_blobs_root( comp_pixel( image[y:y+Ly, x:x+Lx]), rV)
        Fg = vect_edge(Fg, rV, wTTf); Fg.L_=[]  # form, trace PP_
        return cross_comp(Fg, rc=Fg.rc)  #-> Fcluster?

    def expand_lev(_iy,_ix, elev, Fg):  # seed tile is pixels in 1st lev, or Fg in higher levs

        tile = np.full((Ly,Lx),None, dtype=object)  # exclude from PV__
        PV__ = np.zeros([Ly,Lx])  # maps to current-level tile
        Fg_ = []; iy,ix =_iy,_ix; y,x = 31,31  # start at tile mean
        while True:
            if not elev: Fg = base_tile(iy,ix)  # 1st level or cross_comped arg tile
            if Fg and val_(Fg.dTT, Fg.rc+compw+elev, mw=(len(Fg.N_)-1)*Lw) > 0:
                tile[y,x] = Fg; Fg_ += [Fg]; dy_dx = np.array([Fg.yx[0]-y,Fg.yx[1]-x])
                if proj_N(Fg, np.hypot(*dy_dx), dy_dx, elev)[1] > 0:  # pV
                    # extend lev by feedback within current tile:
                    proj_focus(PV__,y,x,Fg)  # PV__+= pV__
                    pv__ = PV__.copy(); pv__[tile!=None] = 0  # exclude processed
                    y, x = np.unravel_index(pv__.argmax(), PV__.shape)
                    if PV__[y,x] > ave:
                        iy = _iy + (y-31)* Ly**elev  # feedback to shifted coords in full-res image | space:
                        ix = _ix + (x-31)* Lx**elev  # y0,x0 in projected bottom tile:
                        if elev:
                            subF = frame_H(image, iy,ix, Ly,Lx, Y,X, rV, elev, wTTf)  # up to current level
                            Fg = subF.H[-1] if subF else []
                    else: break
                else: break
            else: break
        if Fg_ and val_(np.sum([g.dTT for g in Fg_],axis=0), np.mean([g.rc for g in Fg_])+elev, mw=(len(Fg_)-1)*Lw) > 0:
            return Fg_

    global ave, Lw, intw, compw, centw, contw, adist, amed, medw, mW, dW
    ave, Lw, intw, compw, centw, contw, adist, amed, medw = np.array([ave, Lw, intw, compw, centw, contw, adist, amed, medw]) / rV

    frame = CN(box=np.array([0,0,Y,X]), yx=np.array([Y//2, X//2]))
    Fg=[]; elev=0
    while elev < max_elev:  # same center in all levels
        Fg_ = expand_lev(iY,iX, elev, Fg)
        if Fg_:  # higher-scope tile
            Fg = cross_comp(CN(N_=Fg_), rc=elev)  # cross_comp(Fg_), root=None, spec-> N_,C_,L_ for Fcluster
            if Fg:
                frame.H += [Fg]; elev += 1  # forward comped tile
                if max_elev == 4:  # seed, not from expand_lev
                    rV,wTTf = ffeedback(Fg)  # set filters
                    Fg = cent_attr(Fg,2)  # set Fg.dTT correlation weights
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