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

class CdH(CBase):  # derivation hierarchy or a layer thereof, subset of CG
    name = "der"
    def __init__(d, **kwargs):
        super().__init__()
        d.H = kwargs.get('H',[])  # empty if single layer: redundant to Et,dTT
        d.dTT = kwargs.get('dTT', np.zeros((2,9)))  # m_,d_ [M,D,n, I,G,A, L,S,eA]: single layer or sum derH
        d.root = kwargs.get('root', [])  # to pass dTT
    def __bool__(d): return bool(np.any(d.dTT))  # n>0

def copy_(dH, root):
    cH = CdH(dTT=deepcopy(dH.dTT), root=root)  # summed across H
    cH.H = [copy_(lay, cH) for lay in dH.H]
    return cH

def add_dH(DH, dH):  # rn = n/mean, no rev, merge/append lays

    DH.dTT += dH.dTT
    off_H = []
    for D, d in zip_longest(DH.H, dH.H):
        if D and d: add_dH(D, d)
        elif d:     off_H += [copy_(d, DH)]
    DH.H += off_H
    return DH

def comp_dH(_dH, dH, rn, root):  # unpack derH trees down to numericals and compare them

    H = []
    if _dH.H and dH.H:  # 2 or more layers each, eval rdn to dTT as in comp_N?
        dTT = np.zeros((2,9))
        for D, d in zip(_dH.H, dH.H):
            ddH = comp_dH(D,d, rn, root)
            H += [ddH]; dTT += ddH.dTT
    else: dTT = comp_derT(_dH.dTT[1], dH.dTT[1]*rn)

    return CdH(H=H, dTT=dTT, root=root)

class CN(CBase):
    name = "node"
    def __init__(n, **kwargs):
        super().__init__()
        n.rc  = kwargs.get('rc',1)  # redundancy to ext Gs, ave in links? separate rc for rim, or internally overlapping?
        n.fi  = kwargs.get('fi',1)  # if G else 0, fd_: list of forks forming G?
        n.nt  = kwargs.get('nt',[])  # nodet, empty if fi
        n.N_  = kwargs.get('N_',[])  # nodes, concat in links
        n.L_  = kwargs.get('L_',[])  # internal links, +|- if failed?
        n.rim = kwargs.get('rim',[])  # external links, rng-nest?
        n.m   = kwargs.get('m', 0); n.d = kwargs.get('d', 0);  n.c = kwargs.get('c' ,0)  # sum L_ dTT -> rm,rd
        n.em  = kwargs.get('em',0); n.ed = kwargs.get('ed',0); n.ec = kwargs.get('ec',0)  # sum rim eTT
        n.dTT = kwargs.get('dTT',np.zeros((2,9)))  # sum derH-> m_,d_ [M,D,n, I,G,a, L,S,A], L: dLen, S: dSpan
        n.eTT = kwargs.get('eTT',np.zeros((2,9)))  # sum rim derH
        n.derH  = kwargs.get('derH', CdH())  # sum from clustered L_s
        n.dLay  = kwargs.get('dLay', CdH())  # sum from terminal L_, Fg only?
        n.baseT = kwargs.get('baseT',np.zeros(4))  # I,G,A: not ders, not in links?
        n.yx  = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.rng = kwargs.get('rng',1)  # or med: loop count in comp_node_|link_
        n.box = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.span = kwargs.get('span',0) # distance in nodet or aRad, comp with baseT and len(N_), not additive?
        n.angl = kwargs.get('angl',[np.zeros(2),0])  # (dy,dx),dir, sum from L_
        n.mang = kwargs.get('mang',1)  # ave match of angles in L_, =1 in links
        n.B_ = kwargs.get('B_', [])  # ext boundary | background neg Ls/lG: [B_,Et,R], add dB_?
        n.rB_= kwargs.get('rB_',[])  # reciprocal cores for lG, vs higher-der lG.B_
        n.C_ = kwargs.get('C_', [])  # int centroid Gs, add dC_?
        n.rC_= kwargs.get('rC_',[])  # reciprocal root centroids
        n.nH = kwargs.get('nH', [])  # top-down hierarchy of sub-node_s: CN(sum_N_(Nt_))/ lev, with single added-layer derH, empty nH
        n.lH = kwargs.get('lH', [])  # bottom-up hierarchy of L_ graphs: CN(sum_N_(Lt_))/ lev, within each nH lev
        n.root = kwargs.get('root',None)  # immediate
        n.sub  = 0  # full-composition depth relative to top-composition peers
        n.fin  = kwargs.get('fin',0)  # clustered, temporary
        n.exe  = kwargs.get('exe',0)  # exemplar, temporary
        n.compared = set()
        # n.fork_tree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def __bool__(n): return bool(n.N_)

ave, avd, arn, aI, aS, aveB, aveR, Lw, intw, compw, centw, contw = .3, .2, 1.2, 100, 5, 100, 3, 5, 2, 5, 10, 15  # value filters + weights
dec = ave / (ave+avd)  # match decay per unit dist
adist, amed, distw, medw, specw = 10, 3, 2, 2, 2  # cost filters + weights, add alen?
wM, wD, wN, wG, wL, wI, wS, wa, wA = 10, 10, 20, 20, 5, 20, 2, 1, 1  # der params higher-scope weights = reversed relative estimated ave?
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

def cross_comp(root, rc, fC=0):  # rng+ and der+ cross-comp and clustering

    N_, mL_,mTT, dL_,dTT = comp_C_(root.N_,rc) if fC else comp_N_(root.N_,rc)  # rc: redundancy+olp, lG.N_ is Ls
    nG, lG = [], []
    if len(dL_) > 1: # comp dL_
        dV = val_(dTT, rc+compw, fi=0, mw=(len(dL_)-1)*Lw)  # add N.ed?
        if dV > 0:
            if root.fi and root.L_: lH = [sum_N_(root.L_,rc,root)]  # agg+ root is always Fg?
            else: lH = []
            if fC < 2 and dV > avd:  # may be dC_, no comp ddC_
                lG = cross_comp(CN(N_=dL_,root=root), rc+compw+1, fC*2)  # trace_edge via rB_|B_
                if lG:
                    lH += [lG]+lG.nH; dLay = add_dH(root.dLay, lG.derH)  # lH extension
    # cluster mL_:
    if len(mL_) > 1 and val_(mTT, rc+compw, mw=(len(mL_)-1)*Lw) > 0:
        for n in N_: n.em = sum([l.m for l in n.rim]) / len(n.rim)  # tentative before val_
        if root.root: nG = Cluster(root, mL_, rc, fC)  # fC=0: get_exemplars, cluster_C, rng connect cluster
        else:         nG = Fcluster(root,mL_, rc)  # root=frame, splice,cluster L'L_,lH_,C_
        if nG:        # batched nH extension
            rc+=nG.rc # redundant clustering layers
            if lG:
                form_B__(nG, lG, rc+2)  # assign boundary per N, O+=B_[-1]|1?
                if val_(mTT, rc+3+contw, mw=(len(nG.N_)-1)*Lw) > 0:  # mval
                    trace_edge(nG, rc+3)  # comp adjacent Ns via B_
            if val_(nG.dTT, rc+compw+3, mw=(len(nG.N_)-1)*Lw, _TT=mTT) > 0:
                nG = cross_comp(nG, rc+3) or nG  # connec agg+, fC = 0
            nG.nH = root.nH + [root] + nG.nH  # nG.nH is higher composition
    if lG:
        if not nG: nG = CN(root=root)
        nG.dLay = dLay; nG.lH = lH + nG.lH
        nG.dTT += lG.dTT
        nG.rc += lG.rc  # not sure
    return nG  # replace root

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
            if   dC.m > ave*(contw+rc) > 0: mL_+=[dC]; mTT+=dC.dTT; out_ += [_C,C]
            elif dC.d > avd*(contw+rc) > 0: dL_+=[dC]; dTT+=dC.dTT  # not in out_?
    else:
        # sort,select along eigenvector, not implemented yet
        for C in C_: C.compared =set()
        i = np.argmax(wTTf[0]+wTTf[1])
        C_ = sorted(C_, key=lambda C: C.dTT[0][i])
        for j in range( len(C_)-1):
            _C = C_[j]; C = C_[j+1]
            if _C in C.compared: continue
            dy_dx = _C.yx-C.yx; dist = np.hypot(*dy_dx)
            Link = comp_N(_C,C, rc, A=dy_dx, span=dist, rng=1)  # or use comp_derT?
            if   Link.m > ave*(contw+rc) > 0: mL_+=[Link]; mTT+=Link.dTT; out_ += [_C,C]
            elif Link.d > avd*(contw+rc) > 0: dL_+=[Link]; dTT+=Link.dTT  # not in out_?
        C_ = list(set(out_))
    return C_, mL_,mTT, dL_,dTT

def comp_N_(iN_, rc, _iN_=[]):

    def proj_V(_N, N, dist, Ave, pVt_):  # _N x N induction

        iV = (_N.m+N.m)/2 * dec**(dist/((_N.span+N.span)/2)) - Ave
        eV = sum([l.m * dec**(dist/l.span) - Ave for l in _N.rim+N.rim])
        V = iV + eV
        if V > Ave: return V
        elif eV * ((len(pVt_)-1)*Lw) > specw:  # spec over rim, nested spec N_, not L_
            eTT = np.zeros((2,9))  # comb forks?
            for _dist,_dy_dx,__N,_V in pVt_:
                pN = proj_N(N,_dist,_dy_dx)
                if pN: eTT += pN.dTT
                _pN = proj_N(N,_dist,-_dy_dx)
                if _pN: eTT += _pN.dTT
            return iV + val_(eTT,rc)
        else: return V

    N_, mL_,mTT,dL_,dTT = [],[],np.zeros((2,9)),[],np.zeros((2,9))
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
            O = (N.rc +_N.rc) / 2; Ave = ave * rc * O
            V = proj_V(_N,N, dist, Ave, pVt_)
            if V > Ave:
                Link = comp_N(_N,N, O+rc, A=dy_dx, span=dist)
                if   Link.m > ave*(contw+O+rc) > 0: mL_+=[Link]; mTT+=Link.dTT; N_ += [_N,N]  # combined CN dTT and L_
                elif Link.d > avd*(contw+O+rc) > 0: dL_+=[Link]; dTT+=Link.dTT  # fork overlap is possible but much more complex
                pVt_ += [[dist, dy_dx, _N, Link.m-ave*(O+rc)]]
            else:
                break  # no induction
    return list(set(N_)), mL_,mTT, dL_,dTT

def comp_N(_N,N, rc, A=np.zeros(2), span=None, rng=1):  # compare links, optional angl,span,dang?

    dTT,rn = base_comp(_N, N); fN = N.root  # not Fg
    baseT = (rn*_N.baseT+N.baseT) /2  # not new, for fi=0 base_comp
    yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])  # ext
    fi = N.fi
    angl = [A, np.sign(dTT[1] @ wTTf[1])]  # canonic direction
    Link = CN(fi=0, nt=[_N,N], N_=_N.N_+N.N_, c=min(N.c,_N.c), baseT=baseT, yx=yx, box=box, span=span, angl=angl, rng=rng)
    V = val_(dTT,rc)
    if fN and V * (1 - 1/ max(min(len(N.derH.H),len(_N.derH.H)),eps)) > ave:  # rdn to dTT, else derH is empty
        H = [CdH(dTT=copy(dTT), root=Link)]; rc+=1  # + 2nd | higher layers:
        if _N.derH and N.derH:
            dH = comp_dH(_N.derH, N.derH, rn, Link); dtt = dH.dTT; H += dH.H
        else:
            m_,d_ = comp_derT(rn*_N.dTT[1], N.dTT[1]); dtt = np.array([m_,d_]); H += [CdH(dTT=dtt,root=Link)]
        dTT += dtt; Link.derH = CdH(H=H,dTT=dTT,root=Link)  # same as Link dTT?
        V = val_(dTT,rc)  # recompute
    if fi and N.L_: # exclude lGs,PPs
        # spec / N:
        if V * (min(len(_N.N_),len(N.N_))-1)*Lw > ave*rc:
            rc+=1; _, ml_,mtt,dl_,dtt = comp_N_(_N.N_,rc, N.N_)  # cross-nt links only
            Link.L_ = ml_+dl_; dTT+=mtt+dtt; V=val_(dTT,rc)
        if fN:  # not Fg
            if _N.B_ and N.B_:
                _B_,_btt,_bO = _N.B_; B_,btt,bO = N.B_; bO += _bO+rc; bTT =_btt+btt
                if val_(bTT, bO+compw, fi=0, mw=(min(len(_B_),len(B_))-1) *Lw) > 0:
                    rc+=1; _, ml_,mtt,dl_,dtt = comp_N_(_B_,rc, B_)
                    Link.B_ = ml_+dl_; dTT+=mtt+dtt
                    # +spec C_: overlap+offset?
        else:  # Fg: -rc, splice Link L_,lH,C_ in Fcluster:
            _L_,L_ = _N.L_,_N.L_
            if V * (min(len(_L_),len(L_))-1)*Lw > ave*rc:  # add N.et,olp?
                rc+=1; _, ml_,mtt,dl_,dtt = comp_N_(_L_,rc, L_)
                Link.lH = ml_+dl_; dTT+=mtt+dtt; V=val_(dTT,rc)  # higher links
            if _N.C_ and N.C_:
                _C_,_M = _N.C_; C_,M = N.C_  # add val_ cEt
                if (V+_M+M) * (min(len(C_),len(C_))-1)*Lw  > ave*rc:
                    rc+=1; _, ml_,mtt,dl_,dtt = comp_C_(C_,rc,_C_)
                    Link.C_ = ml_+dl_; dTT += mtt+dtt
    Link.rc = rc
    Link.dTT = dTT; Link.m = val_(dTT,rc); Link.d = val_(dTT,rc,fi=0)
    for n, _n in (_N,N), (N,_N):  # if rim-mediated comp: reverse dir in _N.rim: rev^_rev?
        n.rim += [Link]; n.eTT += dTT; n.ec += Link.c; n.compared.add(_n)  # or conditional n.eTT / rim later?
    return Link

def base_comp(_N,N, fC=0):  # comp Et, baseT, extT, dTT

    _M,_D,_c =_N.m,_N.d,_N.c; _I,_G,_Dy,_Dx =_N.baseT; _L = len(_N.N_)  # N_: density of span
    M, D, c  = N.m, N.d, N.c; I, G, Dy, Dx = N.baseT; L = len(N.N_)
    rn = _c/c
    _pars = np.array([_M*rn,_D*rn,_c*rn,_I*rn,_G*rn, [_Dy,_Dx],_L*rn,_N.span], dtype=object)  # Et, baseT, extT
    pars  = np.array([M,D,c, (I,aI),G, [Dy,Dx], L,(N.span,aS)], dtype=object)
    mA,dA = comp_A(_N.angl[0]*_N.angl[1], N.angl[0]*N.angl[1])  # ext angle
    m_,d_ = comp(_pars,pars, mA,dA)  # M,D,n, I,G,a, L,S,A: top dim
    if fC:
        dm_,dd_ = comp_derT(rn*_N.dTT[1], N.dTT[1])  # because comp_N is skipped?
        m_+= dm_; d_+= dd_
    '''
    if np.hypot(*_N.angl[0])*_N.mang + np.hypot(*N.angl[0])*N.mang > ave*wA:  # aligned L_'As, mang *= (len_nH)+fi+1
    mang = (rn*_N.mang + N.mang) / (1+rn)  # ave, weight each side by rn
    align = 1 - mang* (1-mA)  # in 0:1, weigh mA 
    '''
    return np.array([m_,d_]), rn

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

    n_ = set(N.N_) if R else {n for l in N.rim for n in l.nt if n is not N}  # nrim
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

def Cluster(root, iL_, rc, iC):  # generic clustering root

    nG = []
    if iC:
        # input centroid, base connectivity clustering
        C_, L_, i = [],[],0
        if iC < 2:  # merge similar Cs, not dCs, no recomp
            dC_ = sorted(list({L for L in iL_}), key=lambda dC: dC.d)  # from min D
            for i, dC in enumerate(dC_):
                if val_(dC.dTT,rc+compw, fi=0) < 0:  # merge
                    _C, C = dC.nt
                    if _C is not C:  # not merged
                        add_N(_C,C, rc, froot=1)  # fin,root.rim
                        for l in C.rim: l.nt = [_C if n is C else n for n in l.nt]
                        if C in C_: C_.remove(C)  # if multiple merging
                        C_ += [_C]
                else: L_ = dC_[i:]; break
            root.L_ = [l for l in root.L_ if l not in L_]  # cleanup regardless of break
        else: L_ = iL_
        C_ += list({n for L in L_ for n in L.nt})  # include merged Cs
        if val_(root.dTT, rc+contw, mw=(len(C_)-1)*Lw) > 0:
            nG = cluster_n(root, C_,rc)  # in feature space
        if not nG: nG = CN(N_=C_,L_=L_)
    else:
        # input graph, primary centroid clustering
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

def cluster_N(root, rL_, rc, rng=1):  # flood-fill node | link clusters

    def rroot(n): return rroot(n.root) if n.root and n.root!=root else n

    def extend_Gt(_link_, node_, cent_, link_, b_, in_):
        for L in _link_:  # spliced rim
            if L in in_: continue  # already clustered
            in_.add(L)
            for _N in L.nt:
                if _N.fin: continue
                if not N.root or N.root == root or not N.L_:  # not rng-banded
                    node_ += [_N]; cent_ += _N.rC_; _N.fin = 1
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
        if N.fin or (root.root and not N.exe): continue  # no exemplars in Fcluster
        node_,cent_,Link_,_link_,B_ = [N],[],[],[],[]
        if rng==1 or not N.root or N.root==root:  # not rng-banded
            cent_ = N.rC_[:]  # c_roots
            for l in N.rim:
                if l in rL_:
                    if Lnt(l) > ave*rc: _link_ += [l]
                    else: B_+= [l]  # rng-specific
        else: # N is rng-banded, cluster top-rng roots
            n = N; R = rroot(n)
            if R and not R.fin: node_,_link_,cent_ = [R], R.L_[:], R.rC_[:]; R.fin = 1
        N.fin = 1; link_ = []
        while _link_:
            Link_ += _link_
            extend_Gt(_link_, node_, cent_, link_, B_, in_)
            if link_: _link_ = list(set(link_)); link_ = []  # extended rim
            else:     break
        if node_:
            N_, L_, C_ = list(set(node_)), list(set(Link_)), list(set(cent_))
            dTT,olp = np.zeros((2,9)), 0
            for n in N_: olp += n.rc  # from Ns, vs. Et from Ls?
            for l in L_: dTT += l.dTT
            if val_(dTT, rc+olp, 1, (len(N_)-1)*Lw, root.dTT) > 0:
                G_ += [sum_N_(N_, olp, root, L_, [C_,np.sum([c.DTT for c in C_],axis=0)] if C_ else [[],np.zeros(3)], list(set(B_)), rng)]
            elif n.fi:  # L_ is preserved anyway
                for n in N_: n.sub += 1
                G_ += N_
    if G_: return sum_N_(G_, rc, root)  # nG, skip attrs in sum_N_?

def cluster_n(root, iC_, rc):  # simplified flood-fill, currently for for C_ only

    def extend_G(_link_, node_,cent_,link_,b_,in_):
        for L in _link_:  # spliced rim
            if L in in_: continue  # already clustered
            in_.add(L)
            for _N in L.nt:
                if not _N.fin and _N in iC_:
                    node_ += [_N]; cent_ += _N.rC_; _N.fin = 1
                    for l in _N.rim:
                        if l in in_: continue  # cluster by link+density:
                        if Lnt(l) > ave*rc: link_ += [l]
                        else: b_ += [l]
    G_, in_ = [], set()
    for C in iC_: C.fin = 0
    for C in iC_:  # form G per remaining C
        node_,Link_,link_,B_= [C],[],[],[]
        cent_ = C.C_[0][:] if C.C_ else []
        _link_= [l for l in C.rim if Lnt(l) > ave*rc]  # link+nt eval
        while _link_:
            extend_G(_link_, node_, cent_, link_, B_, in_)  # _link_: select rims to extend G:
            if link_: Link_ += link_; _link_ = list(set(link_)); link_ = []
            else:     break
        if node_:
            N_= list(set(node_)); L_= list(set(Link_)); C_ = list(set(cent_))
            dTT,olp = np.zeros((2,9)), 0
            for n in N_: olp += n.rc  # from Ns, vs. Et from Ls?
            for l in L_: dTT += l.dTT
            if val_(dTT,rc+olp, mw=(len(N_)-1)*Lw,_dTT=root.dTT) > 0:
                G_ += [sum_N_(N_, olp, root, L_,[C_,np.sum([c.DTT for c in C_],axis=0)] if C_ else [[],np.zeros(3)],list(set(B_)))]
            elif n.fi:
                G_ += N_
    if G_: return sum_N_(G_, rc, root)  # nG

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
                C = cent_attr( sum_N_(_C.N_, rc, root, fC=1), rc); C.rC_ = []  # C update lags behind N_; non-local C.rc += N.mo_ os?
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
                    for n, mo in zip(_N_,mo_): n.mo_+=[mo]; n.rC_+=[C]; C.rC_+=[n]  # bilateral assign
                    C.M += M; C.D += D; C.C += comp; C.DTT += dTT  # C.Et is a comparand
                    C.N_ += _N_; C._N_ = list(set(_N__))  # core, surround elements
                    C_ += [C]; Dm+=dm; Do+=do  # new incl or excl
                else:
                    for n in _C._N_:
                        n.exe = n.m/n.c > 2 * ave  # refine exe, Et vals are already normalized, Et[2] no longer needed for eval?
                        for i, c in enumerate(n.rC_):
                            if c is _C: # remove mo mapping to culled _C
                                n.mo_.pop(i); n.rC_.pop(i); break
            else: break  # the rest is weaker
        if Dm/Do > Ave:  # dval vs. dolp, overlap increases as Cs may expand in each loop
            _C_ = C_
            for n in root.N_: n._C_=n.rC_; n._mo_=n.mo_; n.rC_,n.mo_ = [],[]  # new n.rC_s, combine with vo_ in Ct_?
        else:  # converged
            break
    C_ = [C for C in C_ if val_(C.DTT, rc)]  # prune C_
    if C_:
        for n in [N for C in C_ for N in C.N_]:
            n.exe = n.et[1-n.fi] + np.sum([mo[0] - ave*mo[1] for mo in n.mo_]) - ave  # exemplar V + summed n match_dev to Cs
            # m * ||C rvals?
        if val_(DTT,1,(len(C_)-1)*Lw, rc+olp, _dTT=root.dTT) > 0:
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
    C.DTT = np.zeros((2,9))  # init C.DTT
    return C

def slope(link_):  # get ave 2nd rate of change with distance in cluster or frame?

    Link_ = sorted(link_, key=lambda x: x.span)
    dists = np.array([l.span for l in Link_])
    diffs = np.array([l.d/l.c for l in Link_])
    rates = diffs / dists
    # ave d(d_rate) / d(unit_distance):
    return (np.diff(rates) / np.diff(dists)).mean()

def Lnt(l): return ((l.nt[0].em + l.nt[1].em - l.m*2) * intw / 2 + l.m) / 2  # L.m is twice included in nt.em

def Copy_(N, rc=1, root=None, init=0):

    C = CN(root=root)
    if init:  # new G
        C.N_ = [N]; C.nH,C.lH = [],[]; C.yx = [N.yx]; C.angl = N.angl[0]  # to get mean
        if init==1:  # else centroid
            C.L_= [l for l in N.rim if l.m>ave]; N.root = C
            N.em, N.ed = val_(N.eTT,rc), val_(N.eTT,rc,fi=0)
    else:
        C.N_,C.L_,C.nH,C.lH = list(N.N_),list(N.L_),list(N.nH),list(N.lH)
        C.angl = N.angl; N.root = root or N.root; C.fi = N.fi  # else 1
    if N.derH:
        C.derH = copy_(N.derH, C)
    C.dTT = deepcopy(N.dTT)
    for attr in ['nt','baseT','yx','box','rim','B_','rB_','C_','rC_']: setattr(C, attr, copy(getattr(N, attr)))
    for attr in ['m','d','c','em','ed','ec','rc','rng','fin', 'span','mang']: setattr(C, attr, getattr(N, attr))
    return C

def sum_N_(N_,rc, root=None, L_=[],C_=[],B_=[], rng=1,fC=0): # sum node,link attrs in graph, aggH in agg+ or player in sub+

    G = Copy_(N_[0],root, init=fC+1)
    G.rc=rc; G.rng=rng; G.N_,G.L_,G.C_,G.B_ = N_,L_,C_,B_; ang=np.zeros(2)
    for N in N_[1:]:
        add_N(G,N, rc, init=1, fC=fC, froot=not fC)  # no need for froot?
    for L in L_:
        add_dH(G.derH,L.derH); ang+=L.angl[0]; G.dTT+=L.dTT  # weight by L.c?
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
        Par += par  # extensive params scale with c?
    N.c += n.c  # cnt / mass
    if n.derH: add_dH(N.derH,n.derH)
    N.box = extend_box(N.box, n.box)
    _cnt,cnt = N.c,n.c; Cnt = _cnt+cnt+eps  # weigh contribution of intensive params, not len N_: near density term?
    N.mang = (N.mang*_cnt + n.mang*cnt) / Cnt
    N.span = (N.span*_cnt + n.span*cnt) / Cnt
    N.rc = (N.rc*_cnt + n.rc*cnt) / Cnt
    if n.nH: add_nH(N.nH,n.nH, N)  # weight by Cnt?
    if n.lH: add_nH(N.lH,n.lH, N)
    if init:  # N is G
        n.em, n.ed = val_(n.eTT,rc), val_(n.eTT,rc,fi=0); N.yx += [n.yx]
        N.angl = (N.angl*_cnt + n.angl*cnt) / Cnt  # vect only
    else: # merge n
        for node in n.N_: node.root = N; N.N_ += [node]
        A,a = N.angl[0],n.angl[0]; A[:] = (A*_cnt+a*cnt) / Cnt
        N.L_ += n.L_; N.rim += n.rim  # no L.root, rims can't overlap?
    if froot: n.fin = 1; n.root = N
    if n.C_:
        if N.C_: N.C_ = [list(set(N.C_[0]+n.C_[0])), N.C_[1]+n.C_[1]]  # centroids: int cluster
        else:    N.C_ = [copy(n.C_[0]), copy(n.C_[1])]  # init
    if fC:
        n.rc = np.sum([mo[1] for mo in n._mo_]); N.rC_ += n.rC_; N.mo_ += n.mo_
    # no B_?
    # if N is Fg: margin = Ns of proj max comp dist > distance to nearest frame point, for cross_comp between frames?
    return N

def add_nH(H, h, root):

    for Lev, lev in zip_longest(H, h, fillvalue=None):  # always aligned?
        if lev:
            if Lev: add_N(Lev,lev)  # froot = 0
            else:   H += [Copy_(lev, root)]

def extend_box(_box, box):
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return min(y0,_y0), min(x0,_x0), max(yn,_yn), max(xn,_xn)

# not used, make H a centroid of layers, same for nH?
def sort_H(H, fi):  # re-assign olp and form priority indices for comp_tree, if selective and aligned

    i_ = []  # priority indices
    for i, lay in enumerate(sorted(H.node_, key=lambda lay: [lay.m, lay.d][fi], reverse=True)):
        di = lay.i - i  # lay index in H
        lay.rc += di  # derR - valR
        i_ += [lay.i]
    H.i_ = i_  # H priority indices: node/m | link/d
    if fi:
        H.root.node_ = H.node_

def eval(V, weights):  # conditional progressive eval, with default ave in weights[0]
    W = 1
    for w in weights:
        W *= w
        if V < W: return 0
    return 1

def agg_frame(foc, image, iY, iX, rV=1, wTTf=[], fproj=0):  # search foci within image, additionally nested if floc

    if foc:
        dert__ = image  # focal img was converted to dert__
    else:
        dert__ = comp_pixel(image)  # global
        global ave, Lw, intw, compw, centw, contw, adist, amed, medw, mW, dW
        ave, Lw, intw, compw, centw, contw, adist, amed, medw = np.array([ave, Lw, intw, compw, centw, contw, adist, amed, medw]) / rV
        # fb rws: ~rvs
    nY, nX = dert__.shape[-2] // iY, dert__.shape[-1] // iX  # n complete blocks
    Y, X = nY * iY, nX * iX  # sub-frame dims
    frame = CN(box=np.array([0, 0, Y, X]), yx=np.array([Y // 2, X // 2]))
    dert__ = dert__[:, :Y, :X]  # drop partial rows/cols
    win__ = dert__.reshape(dert__.shape[0], iY, iX, nY, nX).swapaxes(1, 2)  # dert=5, wY=64, wX=64, nY=13, nX=20
    PV__ = win__[3].sum(axis=(0, 1)) * intw  # init prj_V = sum G, shape: nY=13, nX=20
    aw = contw*10
    while np.max(PV__) > ave * aw:  # max G * int_w + pV, add prj_V foci, only if not foc?
        # max win index:
        y, x = np.unravel_index(PV__.argmax(), PV__.shape)
        PV__[y, x] = -np.inf  # to skip?
        if foc:
            Fg = frame_blobs_root(win__[:, :, :, y, x], rV)  # [dert, iY, iX, nY, nX]
            vect_edge(Fg, rV, wTTf); Fg.L_ = []  # focal dert__ clustering
            cross_comp(Fg, rc=frame.rc)
        else:
            Fg = agg_frame(1, win__[:, :, :, y, x], wY, wX, rV=1, wTTf=[])  # use global wY,wX in nested call
            if Fg and Fg.L_:  # only after cross_comp(PP_)
                rV, wTTf = ffeedback(Fg)  # adjust filters
                Fg = cent_attr(Fg,2)  # compute Fg.wTT: correlation weights in frame dTT
                wTTf *= Fg.wTT; mW = np.sum(wTTf[0]); dW = np.sum(wTTf[1])
                wTTf[0] *= 9 / mW; wTTf[1] *= 9 / dW
                # re-norm weights
        if Fg and Fg.L_:
            if fproj and val_(Fg.dTT,Fg.rc+compw, mw=(len(Fg.N_)-1)*Lw, ):
                pFg = proj_N(Fg, np.array([y, x]))
                if pFg:
                    cross_comp(pFg, rc=Fg.rc)
                    if val_(pFg.dTT,pFg.rc+contw,mw=(len(pFg.N_)-1)*Lw):
                        proj_focus(PV__, y,x, Fg)  # += proj val in PV__
            # no target proj
            frame = add_N(frame, Fg, fmerge=1, froot=1) if frame else Copy_(Fg)
            aw *= frame.rc

    if frame.N_ and val_(frame.dTT, frame.rc+compw, mw=(len(frame.N_)-1)*Lw) > 0:
        # recursive xcomp
        Fn = cross_comp(frame, rc=frame.rc + compw)
        if Fn: frame.N_ = Fn.N_; frame.dTT += Fn.dTT  # other frame attrs remain
        # spliced foci cent_:
        if frame.C_ and val_(frame.C_[1], frame.rc+centw, mw=(len(frame.N_)-1)*Lw, ) > 0:
            Fc = cross_comp(frame, rc=frame.rc + compw, fC=1)
            if Fc: frame.C_ = [Fc.N_, Fc.dTT]; frame.dTT += Fc.dTT
        if not foc:
            return frame  # foci are unpacked

def PP2N(PP):

    P_, link_, B_, verT, latT, A, S, box, yx, Et = PP
    baseT = np.array(latT[:4])
    [mM, mD, mI, mG, mA, mL], [dM, dD, dI, dG, dA, dL] = verT  # re-pack in dTT:
    dTT = np.array([ np.array([mM, mD, mL, mI, mG, mA, mL, mL / 2, eps]),  # extA=eps
                     np.array([dM, dD, dL, dI, dG, dA, dL, dL / 2, eps])])
    y,x,Y,X = box; dy,dx = Y+1-y, X+1-x
    A = np.array([np.array(A), np.sign(dTT[1] @ wTTf[1])], dtype=object)  # append sign
    PP = CN(fi=1, N_=P_,B_=B_, m=Et[0],d=Et[1],c=Et[2], baseT=baseT, dTT=dTT, box=box, yx=yx, angl=A, span=np.hypot(dy/2, dx/2))
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
    for lev in reversed(root.nH):  # top-down, not lev-selective
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
        pV__ = np.array([
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

def proj_H(cH, cos_d, dec):
    pH = CdH()
    if cH.H:  # recursion
        for lay in cH.H:
            play = proj_H(lay, cos_d, dec); pH.H += [play]; pH.dTT += play.dTT
    else:   # proj terminal dTT
        pH.dTT = np.array([cH.dTT[0] * dec, cH.dTT[1] * cos_d * dec])  # same dec for M and D?
        pH.Et = np.array([np.sum(pH.dTT[0]), np.sum( np.abs(pH.dTT[1])), cH.Et[2]])

    return pH  # m = val_(pH.dTT), scale by rV of norm decay M, no effect on D?

def proj_N(N, dist, A):  # recursively specified N projection, rim proj is currently macro in comp_N_?

    rdist = dist / N.span   # internal x external angle:
    cos_d = (N.angl[0].dot(A) / (np.hypot(*N.angl[0]) * dist)) * N.angl[1]  # N-to-yx alignment
    m,d = N.m,N.d  # tentative
    dec = rdist * (m / (m+d))  # match decay rate, * ddecay for ds?
    NH = proj_H(N.derH, cos_d, dec)
    iV = val_(N.dTT, contw, mw=(len(N.N_)-1)*Lw)  # rc = contw?
    pH = copy_(NH, N)
    if N.L_:  # from terminal comp
        LH = proj_H(N.dLay, cos_d, dec); add_dH(pH,LH)
        eV = val_(LH.dTT, contw, mw=(len(N.L_)-1)*Lw)
    else: eV = 0
    if iV + eV > ave:
        return CN(N_=N.N_,L_=N.L_, dTT=pH.dTT, derH=pH)
    LH,L_,NH,N_ = CdH(),[],CdH(),[]
    if not N.root and eV * ((len(N.L_)-1)*Lw) > specw:  # only for Fg?
        for l in N.L_:  # sum L-specific projections
            lA = A - (N.yx-l.yx); ldist = np.hypot(*lA)
            pl = proj_N(l,ldist,lA)
            if pl and pl.m > ave: add_dH(LH,pl.derH); L_+= [pl]  # or diff proj, if D,avd?
        eV = val_(LH.dTT, contw, mw=(len(L_)-1)*Lw)
    if iV * ((len(N.N_)-1)*Lw) > specw:
        for n in (N.N_ if N.fi else N.nt):  # sum _N-specific projections
            if n.derH:
                nA = A - (N.yx-n.yx); ndist = np.hypot(*nA)
                pn = proj_N(n,ndist,nA)
                if pn and pn.m > ave: add_dH(NH,pn.derH); N_+= [pn]
        iV = val_(NH.dTT, contw, mw=(len(N_)-1)*Lw)
    if iV + eV > 0:
        if L_ or N_: pH = add_dH(NH,LH)  # recomputed from individual Ls and Ns
        return CN(N_=N_,L_=L_, dTT=pH.dTT, derH=pH)

    def comp_prj_dH(_N, N, ddH, rn, link, angl, span, dec):
        # comp proj dH to actual dH-> surprise, not used
        _cos_da = angl.dot(_N.angl) / (span * _N.span)  # .dot for scalar cos_da
        cos_da = angl.dot(N.angl) / (span * N.span)
        _rdist = span / _N.span
        rdist = span / N.span
        prj_DH = add_dH(proj_H(_N.derH, _cos_da, _rdist * dec),
                        proj_H(N.derH, cos_da, rdist * dec))  # comb proj dHs
        # add imagination: cross_comp proj derHs?
        # Et+= confirm:
        dddH = comp_dH(prj_DH, ddH, rn, link)
        link.m += dddH.m; link.d += dddH.d; link.c += dddH.c; link.dTT += dddH.dTT
        add_dH(ddH, dddH)

def Fcluster(root, iL_, rc):  # called from cross_comp(Fg_)

    dN_, dL_, dC_ = [], [], []  # spliced from links between Fgs
    for Link in iL_: dN_ += Link.L_; dL_ += Link.lH; dC_ += Link.C_

    DTT = np.zeros((2,9)); N_L_C_ = [[],[],[]]
    for i, (link_,clust, fC) in enumerate([(dN_,cluster_N,0),(dL_,cluster_N,0),(dC_,cluster_n,1)]):
        if link_:
            G = clust(root, link_, rc)
            if G:
                rc+=1; N_L_C_[i] = G.N_; DTT += G.dTT
                if val_(G.dTT, rc, mw=(len(G.N_)-1)*Lw) > 0:
                    G = cross_comp(G, rc, fC=fC)
                    if G: rc+=1; N_L_C_[i] = G.N_; DTT += G.dTT
    N_,L_,C_ = N_L_C_
    return CN(dTT=DTT, rc=rc, N_=N_,L_=L_,C_=C_, root=root)

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
                Edge = sum_N_([PP2N(PPm) for PPm in PPm_],1,None)  # increment rc?
                if edge.link_:
                    lG = sum_N_([PP2N(PPd) for PPd in edge.link_],2, Edge)
                    Edge.rc+=lG.rc; Edge.lH+=[lG]+lG.nH; Edge.dTT+=lG.dTT; add_dH(Edge.derH,lG.derH)  # lH extension
                    form_B__(Edge,lG,3)
                    if val_(Edge.dTT,3, mw=(len(PPm_)-1)*Lw) > 0:
                        trace_edge(Edge,3)  # cluster complemented Gs via G.B_
                        if val_(lG.dTT,4, mw=(len(lG.N_)-1)*Lw):
                            trace_edge(lG,4)  # for cross_comp in frame_H?
                    Edge.L_ = lG.N_
                Edge_ += [Edge]
    return sum_N_(Edge_,2,None)  # Fg, no root

def form_B__(G, lG, rc):  # trace edge / boundary / background per node:

    for Lg in lG.N_:  # form rB_, in Fg?
        rB_ = {n.root for L in Lg.N_ for n in L.nt if n.root and n.root.root is not None}  # core Gs, exclude frame
        Lg.rB_ = sorted(rB_, key=lambda x:(x.m/x.c), reverse=True)  # N rdn = index+1

    def R(L): return L.root if L.root is None or L.root.root is lG else R(L.root)

    for N in G.N_:
        if N.sub or not N.B_: continue
        B_, dTT, rdn = [], np.zeros((2,9)), 0
        for L in N.B_:
            RL = R(L)  # replace boundary L with its root of the level that contains N in root.rB_?
            if RL: B_+=[RL]; dTT+=RL.dTT; rdn += RL.rB_.index(N)+1  # rdn = n stronger cores of RL
        N.B_ = [B_,dTT,rdn]

def trace_edge(root, rc):  # cluster contiguous shapes via PPs in edge blobs or lGs in boundary / skeleton?

    N_ = root.N_  # clustering  is rB_|B_-mediated
    L_ = []; cT_ = set()  # comp pairs
    for N in N_: N.fin = 0
    for N in N_:
        _N_ = [B for rB in N.rB_ if rB.B_ for B in rB.B_[0] if B is not N]
        if N.B_: _N_ += [rB for B in N.B_[0] for rB in B.rB_ if rB is not N]
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
        return cross_comp(Fg, rc=Fg.rc)

    def expand_lev(_iy,_ix, elev, Fg):  # seed tile is pixels in 1st lev, or Fg in higher levs

        tile = np.full((Ly,Lx),None, dtype=object)  # exclude from PV__
        PV__ = np.zeros([Ly,Lx])  # maps to current-level tile
        Fg_ = []; iy,ix =_iy,_ix; y,x = 31,31  # start at tile mean
        while True:
            if not elev: Fg = base_tile(iy,ix)  # 1st level or cross_comped arg tile
            if Fg and val_(Fg.dTT, Fg.rc+compw+elev, mw=(len(Fg.N_)-1)*Lw) > 0:
                tile[y,x] = Fg; Fg_ += [Fg]; dy_dx = np.array([Fg.yx[0]-y,Fg.yx[1]-x])
                pFg = proj_N(Fg, np.hypot(*dy_dx), dy_dx)  # extend lev by feedback within current tile
                if pFg and val_(pFg.dTT, pFg.rc+elev, mw=(len(pFg.N_)-1)*Lw) > 0:
                    proj_focus(PV__,y,x,Fg)  # PV__+= pV__
                    pv__ = PV__.copy(); pv__[tile!=None] = 0  # exclude processed
                    y, x = np.unravel_index(pv__.argmax(), PV__.shape)
                    if PV__[y,x] > ave:
                        iy = _iy + (y-31)* Ly**elev  # feedback to shifted coords in full-res image | space:
                        ix = _ix + (x-31)* Lx**elev  # y0,x0 in projected bottom tile:
                        if elev:
                            subF = frame_H(image, iy,ix, Ly,Lx, Y,X, rV, elev, wTTf)  # up to current level
                            Fg = subF.nH[-1] if subF else []
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
            Fg = cross_comp(CN(N_=Fg_), rc=elev)  # cross_comp(Fg_), spec/ N_,C_,L_, Cluster_F
            if Fg:
                frame.nH += [Fg]; elev += 1  # forward comped tile
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
    # search frames ( tiles inside image, initially should be 4K, or 256K panorama