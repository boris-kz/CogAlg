import numpy as np, weakref
from copy import copy, deepcopy
from functools import reduce
from itertools import zip_longest, combinations, chain, product;  from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, imread, unpack_blob_, comp_pixel, CN, CBase
from slice_edge import slice_edge, comp_angle
from comp_slice import comp_slice
'''
4-stage agglomeration cycle: generative cross-comp, compressive clustering, filter-adjusting feedback, and code-extending forward. 

Cross-comp forms Miss, Match: min= shared_quantity for directly predictive params, else inverse deviation of miss=variation, 2 forks:
rng+: incremental-range cross-comp nodes: edge segments at < max distance, cluster if they match. 
der+: incremental-derivation cross-comp links, from node cross-comp, if abs_diff * rel_adjacent_match

Clustering is compressive by grouping the elements, by direct similarity to centroids or transitive similarity in graphs, 2 forks:
nodes: connectivity clustering / >ave M, progressively reducing overlap by exemplar selection, centroid clustering, floodfill.
links: correlation clustering if >ave D, forming contours that complement adjacent connectivity clusters.

That forms hierarchical graph representation: dual tree of down-forking elements: node_H, and up-forking clusters: root_H:
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
Similar to neurons: dendritic input tree and axonal output tree, but with lateral cross-comp and nested param sets per layer.

Feedback of projected match adjusts filters to maximize next match, including coordinate filters to select new inputs.
(can be refined by cross_comp of co-projected patterns, see "imagination, planning, action" section of part 3 in Readme)
Forward pass may generate code by cross-comp of function calls and clustering code blocks of past and simulated processes.

notation:
prefix  f denotes flag
postfix t denotes tuple, multiple ts is a nested tuple
prefix  _ denotes prior of two same-name vars, multiple _s for relative precedence
postfix _ denotes array of same-name elements, multiple _s is nested array
capitalized vars are summed small-case vars
'''
class CLay(CBase):  # layer of derivation hierarchy, subset of CG
    name = "lay"
    def __init__(l, **kwargs):
        super().__init__()
        l.Et = kwargs.get('Et', np.zeros(3))
        l.olp = kwargs.get('olp', 1)  # ave nodet overlap
        l.node_ = kwargs.get('node_', [])  # concat across fork tree
        l.link_ = kwargs.get('link_', [])
        l.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m_,d_ [M,D,n,o, I,G,A,L], sum across fork tree,
        # alt_L / comp alt_?
        # add weights for cross-similarity, along with vertical aves, for both m_ and d_?
        # i: lay index in root node_,link_, to revise olp; i_: m,d priority indices in comp node|link H
        # ni: exemplar in node_, trace in both directions?
    def __bool__(l): return bool(l.node_)

    def copy_(lay, rev=0, i=None):  # comp direction may be reversed to -1

        if i:  # reuse self
            C = lay; lay = i; C.node_=copy(i.node_); C.link_ = copy(i.link_); C.derTT=np.zeros((2,8))
        else:  # init new C
            C = CLay(node_=copy(lay.node_), link_=copy(lay.link_))
        C.Et = copy(lay.Et)
        for fd, tt in enumerate(lay.derTT):  # nested array tuples
            C.derTT[fd] += tt * -1 if (rev and fd) else deepcopy(tt)

        if not i: return C

    def add_lay(Lay, lay, rev=0, rn=1):  # merge lays, including mlay + dlay

        # rev = dir==-1, to sum/subtract numericals in m_,d_
        for fd, Fork_, fork_ in zip((0,1), Lay.derTT, lay.derTT):
            Fork_ += (fork_ * -1 if (rev and fd) else fork_) * rn  # m_| d_
        # concat node_,link_:
        Lay.node_ += [n for n in lay.node_ if n not in Lay.node_]
        Lay.link_ += lay.link_
        Lay.Et += lay.Et * rn
        Lay.olp = (Lay.olp + lay.olp * rn) /2
        return Lay

    def comp_lay(_lay, lay, rn, root, dir=1):  # unpack derH trees down to numericals and compare them

        i_ = lay.derTT[1] * rn * dir; _i_ = _lay.derTT[1]  # i_ is ds, scale and direction- normalized
        d_ = _i_ - i_
        a_ = np.abs(i_); _a_ = np.abs(_i_)
        m_ = np.minimum(_a_,a_) / reduce(np.maximum,[_a_,a_,1e-7])  # match = min/max comparands
        m_ *= np.where((_i_<0) != (i_<0), -1,1)  # match is negative if comparands have opposite sign
        derTT = np.array([m_,d_])
        node_ = list(set(_lay.node_+ lay.node_))  # concat, or redundant to nodet?
        link_ = _lay.link_ + lay.link_
        M = sum(m_ * w_t[0]); D = sum(d_ * w_t[1])
        Et = np.array([M, D, 8])  # n compared params = 8
        if root: root.Et += Et
        return CLay(Et=Et, olp=(_lay.olp+lay.olp*rn)/2, node_=node_, link_=link_, derTT=derTT)

def copy_(N, root=None, init=0):

    C = CN(root=root)
    if init:  # init G with N
        C.N_,C.H,C.lH, N.root = ([N],[],[],C)
        if N.rim: N.L_ = N.rim.L_ if N.fi else N.rim.N_  # rim else nodet
        if N.alt: C.alt = [N.alt]  # not internalized as rim
    else:
        C.N_,C.L_,C.H,C.lH, N.root = (list(N.N_),list(N.L_),list(N.H),list(N.lH), root if root else N.root)
        if N.rim: C.rim = copy_(N.rim)
        if N.alt: C.alt = copy_(N.alt)
    C.derH  = [[fork.copy_() if fork else [] for fork in lay] for lay in N.derH]
    C.derTT = deepcopy(N.derTT)
    for attr in ['Et', 'baseT','yx','box','angle']: setattr(C, attr, copy(getattr(N, attr)))
    for attr in ['olp','rng', 'fi', 'fin', 'span']: setattr(C, attr, getattr(N, attr))
    return C

def norm_(n, L):

    if n.rim: norm_(n.rim, L)
    if n.alt: norm_(n.alt, L)
    for val in n.Et, n.baseT, n.derTT, n.span, n.yx: val /= L
    if n.derH:
        for lay in n.derH:
            for fork in lay:
                if fork: fork.derTT /= L; fork.Et /= L

def get_rim(N): return N.rim.L_ if N.fi else N.rim.L_[0] + N.rim.L_[1]

ave, avd, arn, aI, aveB, aveR, Lw, int_w, loop_w, clust_w = 10, 10, 1.2, 100, 100, 3, 5, 2, 5, 10  # value filters + weights
ave_dist, ave_med, dist_w, med_w = 10, 3, 2, 2  # cost filters + weights
wM, wD, wN, wO, wI, wG, wA, wL = 10, 10, 20, 20, 1, 1, 20, 20  # der params higher-scope weights = reversed relative estimated ave?
w_t = np.ones((2,8))  # fb weights per derTT, adjust in agg+
wY = wX = 64; wYX = np.hypot(wY,wX)  # focus dimensions
'''
initial PP_ cross_comp and connectivity clustering to initialize focal frame graph, no recursion:
'''
def vect_root(Fg, rV=1, ww_t=[]):  # init for agg+:
    if np.any(ww_t):
        global ave, avd, arn, aveB, aveR, Lw, ave_dist, int_w, loop_w, clust_w, wM, wD, wN, wO, wI, wG, wA, wL, w_t
        ave, avd, arn, aveB, aveR, Lw, ave_dist, int_w, loop_w, clust_w = (
            np.array([ave,avd,arn,aveB,aveR, Lw, ave_dist, int_w, loop_w, clust_w]) / rV)  # projected value change
        w_t = np.multiply([[wM,wD,wN,wO,wI,wG,wA,wL]], ww_t)  # or dw_ ~= w_/ 2?
        ww_t = np.delete(ww_t,(2,3), axis=1)  #-> comp_slice, = np.array([(*ww_t[0][:2],*ww_t[0][4:]),(*ww_t[0][:2],*ww_t[1][4:])])
    blob_ = unpack_blob_(Fg)
    Fg.N_,Fg.L_ = [],[]; lev = CN(); derlay = [CLay(root=Fg),CLay(root=Fg)]
    for blob in blob_:
        if not blob.sign and blob.G > aveB:
            edge = slice_edge(blob, rV)
            if edge.G*((len(edge.P_)-1)*Lw) > ave * sum([P.latuple[4] for P in edge.P_]):
                Et = comp_slice(edge, rV, ww_t)  # to scale vert
                if np.any(Et) and Et[0] > ave * Et[2] * clust_w:
                    cluster_edge(edge, Fg, lev, derlay)  # may be skipped
    Fg.derH = [derlay]
    if lev: Fg.H += [lev]
    return Fg

def cluster_edge(edge, frame, lev, derlay):  # non-recursive comp_PPm, comp_PPd, edge is not a PP cluster, unpack by default

    def cluster_PP_(PP_):
        G_,Et = [],np.zeros(3)
        while PP_:  # flood fill
            node_,link_, et = [],[], np.zeros(3)
            PP = PP_.pop(); _eN_ = [PP]
            while _eN_:
                eN_ = []
                for eN in _eN_:  # rim-connected ext Ns
                    node_ += [eN]
                    for L,_ in eN.rim.L_:  # all +ve, *= density?
                        if L not in link_:
                            for eN in L.N_:
                                if eN in PP_: eN_ += [eN]; PP_.remove(eN)  # merged
                            link_ += [L]; et += L.Et
                _eN_ = {*eN_}
            if val_(et, mw=(len(node_)-1)*Lw, aw=2+clust_w) > 0:  # rc=2
                Et += et
                G_ += [sum2graph(frame, node_,link_,[], et, olp=1, rng=1, fi=1)]  # single-lay link_derH
        return G_, Et

    def comp_PP_(PP_):
        N_,L_,mEt,dEt = [],[],np.zeros(3),np.zeros(3)
        for PP in PP_: PP.rim = CN()
        for _G, G in combinations(PP_, r=2):
            dy,dx = np.subtract(_G.yx,G.yx); dist = np.hypot(dy,dx)
            if dist - (G.span+_G.span) < ave_dist / 10:  # very short here
                L = comp_N(_G, G, ave, fi=_G.fi, angle=np.array([dy,dx]), span=dist, fdeep=1)
                m, d, n = L.Et
                if m > ave * n * _G.olp * loop_w: mEt += L.Et; N_ += [_G,G]  # mL_ += [L]
                if d > avd * n * _G.olp * loop_w: dEt += L.Et  # dL_ += [L]
                L_ += [L]
        dEt[2] = dEt[2] or 1e-7
        return set(N_),L_,mEt,dEt

    PP_ = edge.node_
    if val_(edge.Et, (len(PP_)-edge.rng)*Lw, loop_w) > 0:
        PP_,L_,mEt,dEt = comp_PP_([PP2N(PP,frame) for PP in PP_])
        if PP_:
            if val_(mEt, (len(PP_)-1)*Lw, clust_w, fi=1) > 0:
                G_, Et = cluster_PP_(copy(PP_))
            else: G_ = []
            if G_: frame.N_ += G_; lev.N_ += PP_; lev.Et += Et
            else:  frame.N_ += PP_ # PPm_
        lev.L_+= L_; lev.Et = mEt+dEt  # links between PPms
        for l in L_:
            derlay[0].add_lay(l.derH[0][0]); frame.baseT+=l.baseT; frame.derTT+=l.derTT; frame.Et += l.Et
        if val_(dEt, (len(L_)-1)*Lw, 2+clust_w, fi=0) > 0:
            Lt = cluster_N_(L2N(L_), rc=2, fi=0)
            if Lt:  # L_ graphs
                if lev.lH: lev.lH[0].N_ += Lt.N_; lev.lH[0].Et += Lt.Et
                else:      lev.lH += [Lt]
                lev.Et += Lt.Et

def val_(Et, mw=1, aw=1, _Et=np.zeros(3), fi=1):  # m,d eval per cluster or cross_comp

    if mw <= 0: return 0
    am = ave * aw  # includes olp, M /= max I | M+D? div comp / mag disparity vs. span norm
    ad = avd * aw  # diff value is borrowed from co-projected or higher-scope match value
    m, d, n = Et

    val = (m-am if fi else d-ad) * mw
    if _Et[2]:  # empty if n=0, borrow rational deviation of alt contour if fi else root Et, not circular
        _m, _d, _n = _Et
        val *= (_d/ad if fi else _m/am) * (mw*(_n/n))

    return val
''' 
Core process per agg level, as described in top docstring:
Cross-compare nodes, evaluate incremental-derivation cross-comp of new >ave difference links, recursively.
 
Select sparse exemplars of strong node types, may covert to centroids, refined & extended by mutual match.
Connectivity-cluster exemplars or centroids by >ave match links, correlation-cluster links by >ave difference.

Form complemented clusters (core+contour) for recursive higher-composition cross_comp, reorder by eigenvalues. 
Feedback coords to bottom level or prior-level in parallel pipelines, filter updates in more coarse cycles.

surprise: comp projected node ders x external link ders, conflict: cross-node comp proj -> feedback 
sub-centroids instead of divisive clustering? cluster params in derH? '''

def cross_comp(root, rc, fi=1):  # rc: redundancy+olp; (cross-comp, exemplar selection, clustering), recursion

    N__,L_,Et = comp_node_(root.N_,rc) if fi else comp_link_(root.L_,rc)
    if N__:
        dval = val_(Et, (len(L_)-1)*Lw, rc+loop_w, fi=0)  # L_ Et
        if dval > 0:
            root.angle = np.sum([L.angle for L in L_], axis=0)
            append_dH(root.derH, sum_H_([L.derH for L in L_], root))
            L2N(L_) # dfork x L_ of root mfork, m-cluster -> mm,md xcomp.:
            if dval > ave:
                Lt = cross_comp(sum_N_(L_), rc, fi=0)  # recursive dfork-> Lt.H
            else:  # lower-res
                Lt = cluster_N_(L_, rc+clust_w, fi=0, rng=1)  # cluster L_/ nodet
            if Lt:
                root.Et += Lt.Et; root.lH += [Lt] + Lt.H  # L_ graph levels / root
        # m cluster, recursion:
        n__ = []
        for n in {N for N_ in N__ for N in N_}: n__ += [n]; n.sel = 0  # for cluster_N_
        if val_(Et, (len(n__)-1)*Lw, rc+loop_w) > 0:
            E_, eEt = get_exemplars(root, n__, rc+loop_w, fi=1)  # focal nodes: point cloud
            if isinstance(E_,CN):
                root = E_  # Ct / cluster_C_)_NC_
            elif E_ and val_(eEt, (len(E_)-1)*Lw, rc+clust_w) > 0:
                Nt = []
                for rng, N_ in enumerate(N__, start=1):  # bottom-up rng incr
                    rng_E_ = [n for n in N_ if n.sel]    # cluster via exemplars
                    if rng_E_ and val_(np.sum([n.Et for n in rng_E_], axis=0), (len(rng_E_)-1)*Lw, rc) > 0:
                        Nt = cluster_N_(rng_E_, rc, 1, rng)  # top-rng G_
                if Nt:
                    if val_(Nt.Et, (len(Nt.N_)-1)*Lw, rc+loop_w, _Et=Et) > 0:
                        root = cross_comp(Nt, rc+clust_w)  # agg+
                    else: root = Nt
    return root

def comp_node_(_N_, rc):  # rng+ forms layer of rim and extH per N?

    N__,L_,ET = [],[],np.zeros(3); rng,olp_ = 1,[] # range banded if frng only?
    for n in _N_:
        n.compared_ = []; n.rim = CN()
    while True:  # _vM
        Gp_ = []  # [G pair + co-positionals], for top-nested Ns, unless cross-nesting comp:
        for _G, G in combinations(_N_, r=2):
            if _G in G.compared_ or len(_G.H) != len(G.H):  # | root.H: comp top nodes only
                continue
            radii = _G.span+G.span; dy,dx = np.subtract(_G.yx,G.yx); dist = np.hypot(dy,dx)
            Gp_ += [(_G,G, dy,dx, radii, dist)]
        N_,Et = [],np.zeros(3)
        for Gp in Gp_:
            _G, G, dy,dx, radii, dist = Gp
            (_m,_,_n),(m,_,n) = _G.Et,G.Et; olp = (_G.olp+G.olp)/2; _m = _m * int_w +_G.rim.Et[0]; m = m * int_w + G.rim.Et[0]
            # density / comp_N: et|M, 0/rng=1?
            max_dist = ave_dist * (radii/aveR) * ((_m+m)/(ave*(_n+n+0))/ int_w)  # ave_dist * radii * induction
            if max_dist > dist or set(_G.rim.N_) & set(G.rim.N_):
                # comp if close or share matching mediators: add to inhibited?
                Link = comp_N(_G,G, ave, fi=1, angle=np.array([dy,dx]), span=dist, fdeep = dist < max_dist/2, rng=rng)
                L_ += [Link]  # include -ve links
                if val_(Link.Et, aw=loop_w*olp) > 0:
                    Et += Link.Et; olp_ += [olp]  # link.olp is the same with o
                    for n,_n in zip((_G,G),(G,_G)):
                        n.compared_ += [_n]
                        if n not in N_ and val_(n.rim.Et, aw= rc+rng-1+loop_w+olp) > 0:  # cost+/ rng
                            N_ += [n]  # for rng+ and exemplar eval
        if N_:
            N__ += [N_]; ET += Et
            if val_(Et, (len(N_)-1)*Lw, loop_w* sum(olp_)/max(1,len(olp_))) > 0:  # current-rng vM
                _N_ = N_; rng += 1; olp_ = []  # reset
            else: break  # low projected rng+ vM
        else: break
    return N__,L_,ET

def comp_link_(iL_, rc):  # comp CLs via directional node-mediated link tracing: der+'rng+ in root.link_ rim_t node rims

    for L in iL_:  # init mL_t: nodet-mediated Ls:
        for rev, N, mL_ in zip((0, 1), L.N_, L.mL_t):  # L.mL_t is empty
            for _L, _rev in get_rim(N):
                if _L is not L and _L in iL_:
                    if L.Et[0] > ave * L.Et[2] * loop_w:
                        mL_ += [(_L, rev ^ _rev)]  # direction of L relative to _L
    med = 1; _L_ = iL_
    L__,LL_,ET = [],[],np.zeros(3)
    while True:  # xcomp _L_
        L_, Et = [], np.zeros(3)
        for L in _L_:
            for mL_ in L.mL_t:
                for _L, rev in mL_:  # rev is relative to L
                    if _L in L.compared_: continue
                    dy,dx = np.subtract(_L.yx,L.yx)
                    Link = comp_N(_L,L, ave*rc, 0, np.array([dy,dx]), np.hypot(dy,dx), -1 if rev else 1)  # d = -d if L is reversed relative to _L
                    Link.med = med
                    LL_ += [Link]; Et += Link.Et  # include -ves, link order: nodet < L < rim, mN.rim || L
                    for l,_l in zip((_L,L),(L,_L)):
                        l.compared_ += [_l]
                        if l not in L_ and val_(l.rim.Et, aw= rc+med+loop_w) > 0:  # cost+/ rng
                            L_ += [l]  # for med+ and exemplar eval
        if L_: L__ += L_; ET += Et
        # extend mL_t per last medL:
        if Et[0] > ave * Et[2] * (rc + loop_w + med*med_w):  # project prior-loop value, med adds fixed cost
            _L_, ext_Et = set(), np.zeros(3)
            for L in L_:
                mL_t, lEt = [set(),set()], np.zeros(3)  # __Ls per L
                for mL_,_mL_ in zip(mL_t, L.mL_t):
                    for _L, rev in _mL_:
                        for _rev, N in zip((0,1), _L.N_):
                            rim = get_rim(N)
                            if len(rim) == med:  # append in comp loop
                                for __L,__rev in rim:
                                    if __L in L.visited_ or __L not in iL_: continue
                                    L.visited_ += [__L]; __L.visited_ += [L]
                                    if __L.Et[0] > ave * __L.Et[2] * loop_w:
                                        mL_.add((__L, rev ^_rev ^__rev))  # combine reversals: 2 * 2 mLs, but 1st 2 are pre-combined
                                        lEt += __L.Et
                if lEt[0] > ave * lEt[2]:  # L rng+, vs. L comp above, add coef
                    L.mL_t = mL_t; _L_.add(L); ext_Et += lEt
            # refine eval by extension D:
            if ext_Et[0] > ave * ext_Et[2] * (loop_w + med*med_w): med += 1
            else: break
        else: break
    return list(set(L__)), LL_, ET

def base_comp(_N, N, dir=1):  # comp Et, Box, baseT, derTT
    """
    pairwise similarity kernel:
    m_ = element‑wise min/max ratio of eight attributes (sign‑aware)
    d_ = element‑wise signed difference after size‑normalisation
    DerTT = np.vstack([m_,d_])
    Et[0] = Σ(m_ * w_t[0])    # total match (≥0) = relative shared quantity
    Et[1] = Σ(|d_| * w_t[1])  # total absolute difference (≥0)
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
    if isinstance(N,CN): # dimension is n nodes
        _L,L = len(_N.N_), len(N.N_)
        mL,dL = min(_L,L)/ max(_L,L), _L - L
    else:  # dimension is distance
        _L,L = _N.span, N.span   # dist, not cumulative, still positive?
        mL,dL = min(_L,L)/ max(_L,L), _L - L
    # comp depth, density:
    # lenH and ave span, combined from N_ in links?
    _m_,_d_ = np.array([[mM,mD,mn,mo,mI,mG,mA,mL], [dM,dD,dn,do,dI,dG,dA,dL]])
    # comp derTT:
    _i_ = _N.derTT[1]; i_ = N.derTT[1] * rn  # normalize by compared accum span
    d_ = (_i_ - i_ * dir)  # np.arrays
    _a_,a_ = np.abs(_i_),np.abs(i_)
    m_ = np.divide( np.minimum(_a_,a_), reduce(np.maximum, [_a_,a_,1e-7]))  # rms
    m_ *= np.where((_i_<0) != (i_<0), -1,1)  # match is negative if comparands have opposite sign
    m_ += _m_; d_ += _d_
    DerTT = np.array([m_,d_])  # [M,D,n,o, I,G,A,L], weigh by centroid_M?
    Et = np.array([np.sum(m_* w_t[0]), np.sum(np.abs(d_* w_t[1])), min(_n,n)])  # feedback-weighted sum shared quantity between comparands

    return DerTT, Et, rn

def comp_N(_N,N, ave, fi=1, angle=None, span=None, dir=1, fdeep=0, fproj=0, rng=1):  # compare links, relative N direction = 1|-1, no angle,span?
    # use N.fi instead of arg?

    derTT, Et, rn = base_comp(_N, N, dir)
    # link M,D,A:
    baseT = np.array([*(_N.baseT[:2] + N.baseT[:2]*rn) / 2, *angle])  # redundant angle for generic base_comp, also span-> density?
    _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
    o = (_N.olp+N.olp) / 2
    Link = CN(rng=rng, olp=o, N_=[_N,N], baseT=baseT, derTT=derTT, yx=np.add(_N.yx,N.yx)/2, span=span, angle=angle, box=box, fi=0)
    Link.derH = [[CLay(root=Link, Et=Et, node_=[_N,N],link_=[Link], derTT=copy(derTT)),[]]]  # empty dlay
    # contour:
    if fi and (_N.alt and N.alt) and val_(_N.alt.Et+N.alt.Et, aw=2+o, fi=0) > 0:  # eval Ds
        Link.altL = comp_N(L2N(_N.alt), L2N(N.alt), ave*2,1, angle,span); Et += Link.altL.Et
    # derH, proj derH:
    if fdeep and (val_(Et, len(N.derH)-2, o) > 0 or fi==0):  # else derH is dext,vert
        ddH = comp_H(_N.derH, N.derH, rn, Link, derTT, Et)  # comp same lays, add dlay = []
        if fproj and val_(Et, len(ddH)-2, o) > 0:
            M,D,_ = Et; dec = M / (M+D)  # final Et
            comp_prj_dH(_N,N, ddH, rn, Link, angle, span, dec)  # surprise = comp combined-N prj_dH to actual ddH
        Link.derH += ddH  # flat
    Link.Et = Et
    if Et[0] > ave * Et[2]:
        for rev, node, _node in zip((0,1),(N,_N),(_N,N)):  # reverse Link dir in _N.rimt
            # simplified sum_N_:
            if fi: node.rim.L_ += [(Link,rev)]
            else:  node.rim.L_[1-rev] += [(Link,rev)]  # rimt opposite to _N,N dir
            node.rim.Et += Et; node.rim.N_ += [_node]; node.rim.baseT += baseT
            node.rim.derTT += derTT
            # simplified add_H(node.rim.derH, Link.derH, root=node, rev=rev)?
    return Link

def get_exemplars(root, iN_, rc, fi):  # get sparse nodes by multi-layer non-maximum suppression

    def nolp(N, inhib_): # inhibition zone
        olp_ = [L for L, _ in N.rim.L_ if [n for n in L.N_ if n in inhib_]]  # += in max dist?
        if olp_:
            oEt = np.sum([i.Et for i in olp_], axis=0)
            rM = (oEt[0]/oEt[2]) / (N.Et[0]/N.Et[2])
            return val_(N.Et * rM, clust_w)
        else: return 0

    N_,Et = [],np.zeros(3)
    _N_ = set()  # stronger-N inhibition zones
    for rdn, N in enumerate(sorted(iN_, key=lambda n: n.rim.Et[0]/n.rim.Et[2], reverse=True), start=1):
        # ave *= overlap by stronger-N inhibition zones:
        if val_(N.rim.Et, aw = rc + rdn + loop_w + nolp(N, list(_N_))) > 0:
            Et += N.rim.Et; _N_.update(N.rim.N_)
            N.sel = 1  # select for cluster_N_, add val to next N olp?
            N_ += [N]  # exemplars
        else:
            break  # the rest of N_ is weaker
    V = val_(Et, (len(N_)-1)*Lw, rc+clust_w)
    if V > 0:
        if fi and V > ave*clust_w:  # replace exemplars with centroids if high match: low decay, else no divisive clustering?
            for n in root.N_: n.C_,n.vo_, n._C_,n._vo_ = [],[],[],[]
            Ct = cluster_C_(root, N_, rc+clust_w)  # sub-cluster by rim + group similarity, no groups before
            if Ct: N_ = Ct, Et = Ct.Et
    return N_, Et

def cluster_C_(root, E_, rc):  # form centroids from exemplar _N_, drifting / competing via rims of new member nodes

    def comp_C(C, n):
        _,et,_ = base_comp(C, n)
        if C.rim and n.rim: _,_et,_= base_comp(C.rim,n.rim); et +=_et
        if C.alt and n.alt: _,_et,_= base_comp(C.alt,n.alt); et +=_et
        return et
    _C_ = []
    for E in E_:
        C = copy_(E,root); C.N_=[E]  # init centroid
        C._N_ = list({_n for n in E.rim.N_ for _n in n.rim.N_})  # core members + surround for comp to N_ mean
        _C_ += [C]
    while True:
        C_, ET, Olp, Dvo = [],np.zeros(3),0,np.zeros(2)  # per _C_
        for _C in sorted(_C_, key=lambda n: n.rim.Et[0]/n.rim.Et[2], reverse=True):
            if val_(_C.Et, (len(_C._N_)-1)*Lw, clust_w+rc) > 0:
                # get mean cluster:
                C = sum_N_(_C.N_, root=root, fC=1)  # merge rim,alt before cluster_NC_
                _N_,_N__, Et,O,dvo = [],[], np.zeros(3),0,np.zeros(2)  # per C
                for n in _C._N_:  # core + surround
                    if C in n.C_: continue
                    et = comp_C(C,n); v = val_(et,aw=rc)
                    if v > 0:
                        olp = np.sum([vo[0] / v for vo in n.vo_ if vo[0] > v])  # olp: rel n val in stronger Cs, not summed with C
                        vo = np.array([v,olp]); O += olp  # Cs overlap: norm by n+olp
                        if v > ave * olp:
                            n.C_ += [C]; n.vo_ += [vo]; Et += et; _N_ += [n]; _N__ += n.rim.N_
                        if C not in n._C_: dvo += vo
                    elif C in n._C_:
                        dvo += n._vo_[n._C_.index(C)]  # old vo_
                Dvo += dvo  # new incl or excl
                C.Et = Et; C.olp = O; C.N_ = list(set(_N_)); C._N_ = list(set(_N__))  # select core, surround
                if val_(C.Et, aw=clust_w+rc+ C.olp) > 0:
                    C_ += [C]; ET += Et; Olp += O
            else:
                break  # rest of N_ is weaker
        if np.sum(Dvo) > ave * loop_w+clust_w:  # both V and O represent change, comeasurable?
            _C_ = C_
            for n in root.N_:
                n._C_ = n.C_; n._vo_= n.vo_; n.C_,n.vo_ = [],[]  # new n.C_s
        else: break
        # converged, also break if low ET: no expansion?
    V = val_(ET, aw=rc+loop_w + Olp)  # C_ value
    if V > 0:
        Nt = sum_N_(E_)
        Nt.Et = ET  # redundant?
        if V * ((len(C_)-1)*Lw) > ave*clust_w:
            Nt = cluster_NC_(C_, rc+clust_w, Nt)
        return Nt

def cluster_NC_(_C_, rc, _Nt):  # cluster centroids if pairwise Et + overlap Et

    C_,L_ = [],[]; Et = np.zeros(3)
    # form links:
    for _C,C in combinations(_C_,2):
        oN_ = list(set(C.N_) & set(_C.N_))
        if not oN_:  # compare overlaps only
            continue
        dy,dx = np.subtract(_C.yx,C.yx)
        link = comp_N(_C, C, ave, angle=np.array([dy,dx]), span=np.hypot(dy,dx))
        oV = val_(np.sum([n.Et for n in oN_],axis=0), rc)
        _V = min(val_(C.Et, rc), val_(_C.Et, rc), 1e-7)
        link.Et[:2] *= int_w; link.Et[0] += oV/_V - arn  # oV priority
        if val_(link.Et, rc) > 0:
            C_ += [_C,C]; L_+= [link]; Et += link.Et
    C_ = list(set(C_))
    if val_(Et, (len(C_)-1)*Lw, rc) > 0:
        G_ = []
        for C in C_:
            if C.fin: continue
            G = copy_(C,init=1); C.fin = 1
            for _C, Lt in zip(C.rim.N_,C.rim.L_):
                G.L_+= [Lt[0]]
                if _C.fin and _C.root is not G:
                    add_N(G,_C.root, fmerge=1)
                else: add_N(G,_C); _C.fin=1
            G_ += [G]
        Gt = G_,L_,Et
        if val_(Et, (len(G_)-1)*Lw, rc+1) > 0:  # new len G_, aw
            Gt = cluster_NC_(G_, rc+1, Gt)  # agg recursion, also return top L_?
        return Gt
    else:
        return _Nt

def cluster_N_(N_, rc, fi, rng=1, fL=0, root=None):  # connectivity cluster exemplar nodes via rim or links via nodet or rimt

    def lolp(N, L_, fR=0):  # relative N.rim or R.link_ olp eval for clustering
        oL_ = [L for L in (N.L_ if fR else N.rim.L_) if L in L_]
        if oL_:
            oEt = np.sum([l.Et for l in oL_], axis=0)
            _Et = N.Et if fR else N.rim.Et
            rM  = (oEt[0]/oEt[2]) / (_Et[0]/_Et[2])
            return val_(_Et * rM, clust_w)
        else: return 0
    # flood-fill Gs, exclusive per fork,ave, only centroid-based can be fuzzy
    G_ = []
    for n in N_: n.fin = 0
    for N in N_:
        if N.fin: continue
        # init N cluster:
        if rng==1 or N.root.rng==1:  # N is not rng-nested, including centroids
            node_, link_, llink_, Et, olp = [N],[],[], N.Et+N.rim.Et, N.olp
            for l,_ in get_rim(N):  # +ve
                if l.rng==rng: link_ += [l]
                elif l.rng>rng: llink_ += [l]  # longer-rng rim
        else:
            n = N; R = n.root  # N.rng=1, R.rng > 1, cluster top-rng roots instead
            while R.root and R.root.rng > n.rng: n = R; R = R.root
            if R.fin: continue
            node_,link_,llink_,Et,olp = [R],R.L_,R.hL_,copy(R.Et),R.olp
            R.fin = 1
        nrc = rc+olp; N.fin = 1  # extend N cluster:
        if fL:  # cluster links via nodet
            for _N in N.N_:
                if _N.fin: continue
                _N.fin = 1; link_ += [_N]  # nodet is mediator
                for L,_ in get_rim(_N):
                    if L not in node_ and _N.Et[1] > avd * _N.Et[2] * nrc:  # direct eval diff
                        node_ += [L]; Et += L.Et; olp += L.olp  # /= len node_
        else:  # cluster nodes via links
            for L in link_[:]:  # snapshot
                for _N in L.N_:
                    if _N not in N_ or _N.fin: continue  # connectivity clusters don't overlap
                    if rng==1 or _N.root.rng==1:  # not rng-nested
                        if lolp(N, get_rim(_N)):
                            node_ +=[_N]; Et += _N.Et+_N.rim.Et; olp+=_N.olp; _N.fin=1
                            for l,_ in get_rim(_N):
                                if l not in link_ and l.rng == rng: link_ += [l]
                                elif l not in llink_ and l.rng>rng: llink_+= [l]  # longer-rng rim
                    else:
                        _n =_N; _R=_n.root  # _N.rng=1, _R.rng > 1, cluster top-rng roots if rim intersect:
                        while _R.root and _R.root.rng > _n.rng: _n=_R; _R=_R.root
                        if not _R.fin and lolp(N,link_,fR=1):
                            link_ = list(set(link_+_R.link_)); llink_ = list(set(llink_+_R.hL_))
                            node_+= [_R]; Et +=_R.Et; olp += _R.olp; _R.fin = 1; _N.fin = 1
        node_ = list(set(node_))
        nrc = rc + olp  # updated
        _Et = root.Et if (not fi and root) else np.zeros(3)  # or contour comb_alt_(node_,rc).Et: form alt_ here?
        if val_(Et, (len(node_)-1)*Lw, nrc, _Et, fi=fi) > 0:
            G_ += [sum2graph(root, node_, link_, llink_, Et, olp, rng)]

    return sum_N_(G_ or N_, root)   # root N_|L_ replacement

def sum2graph(root, node_,link_,llink_, Et, olp, rng, fC=0):  # sum node and link params into graph, aggH in agg+ or player in sub+

    n0 = copy_(node_[0]); derH = n0.derH
    l0 = copy_(link_[0]); DerH = l0.derH
    graph = CN(root=root, fi=1,rng=rng, N_=node_,L_=link_,olp=olp, Et=Et, box=n0.box, baseT=n0.baseT+l0.baseT, derTT=n0.derTT+l0.derTT)
    graph.hL_ = llink_
    n0.root = graph; yx_ = [n0.yx]; fg = isinstance(n0.N_[0],CN)  # not PPs
    Nt = copy_(n0)  #->CN, comb forks: add_N(Nt,Nt.Lt)?
    for N in node_[1:]:
        add_H(derH,N.derH,graph); graph.baseT+=N.baseT; graph.derTT+=N.derTT; graph.box=extend_box(graph.box,N.box); yx_+=[N.yx]
        if fg: add_N(Nt,N)
    for L in link_[1:]:
        add_H(DerH,L.derH,graph); graph.baseT+=L.baseT; graph.derTT+=L.derTT
    graph.derH = append_dH(derH,DerH)  # aligned lay0s? merged mfork += [merged dfork]
    if fg: graph.H = Nt.H + [Nt]  # pack prior top level
    yx = np.mean(yx_, axis=0); dy_,dx_ = (yx_-yx).T; dist_ = np.hypot(dy_,dx_)
    graph.span = dist_.mean()  # node centers distance to graph center
    graph.angle = np.sum([l.angle for l in link_],axis=0)
    graph.yx = yx
    if fC:
        m_,M = centroid_M(graph.derTT[0],ave*olp)  # weigh by match to mean m|d
        d_,D = centroid_M(graph.derTT[1],ave*olp)
        graph.derTT = np.array([m_,d_])
        graph.Et = np.array([M,D,Et[2]])
    if fi:
        alt_ = list(set([L.root for L in link_]))  # individual rims are too weak for contour
        if alt_: graph.alt = sum_N_(alt_)
    return graph

def append_dH(H, dH):  # merged mfork += [merged dfork], keep nesting

    for lay, dlay in zip_longest(H,dH):  # different len if lay-selective comp
        if dlay:
            if dlay[1]: dlay[0].add_lay(dlay[1]); dlay[1] = []
        if lay:
            if lay[1]: lay[0].add_lay(lay[1])
            if dlay:   lay[1] = dlay[0].copy_()
        elif dlay:     H += [dlay]
    return H

def centroid_M(m_, ave):  # adjust weights on attr matches | diffs, recompute with sum

    _w_ = np.ones(len(m_))
    am_ = np.abs(m_)  # m|d are signed, but their contribution to mean and w_ is absolute
    M = np.sum(am_)
    while True:
        mean = max(M / np.sum(_w_), 1e-7)
        inverse_dev_ = np.minimum(am_/mean, mean/am_)  # rational deviation from mean rm in range 0:1, 1 if m=mean, 0 if one is 0?
        w_ = inverse_dev_/.5  # 2/ m=mean, 0/ inf max/min, 1/ mid_rng | ave_dev?
        w_ *= 8 / np.sum(w_)  # mean w = 1, M shouldn't change?
        if np.sum(np.abs(w_-_w_)) > ave:
            M = np.sum(am_* w_)
            _w_ = w_
        else:
            break
        # recursion if weights change
    return m_* w_, M  # no return w_?

def comp_H(H,h, rn, root, DerTT=None, ET=None):  # one-fork derH if not fi, else two-fork derH

    derH, derTT, Et = [], np.zeros((2,8)), np.zeros(3)
    for _lay,lay in zip_longest(H,h):  # different len if lay-selective comp?
        if _lay and lay:
            dlay = []
            for _fork, fork in zip_longest(_lay, lay):  # dfork: xproj in link | xlink in node
                if _fork and fork:
                    dfork = _fork.comp_lay(fork, rn, root=root)
                    if dfork: dlay += [dfork]; derTT = dfork.derTT; Et += dfork.Et
                else: dlay += [[]]
            derH += [dlay]
    if Et[2]: DerTT += derTT; ET += Et
    return derH

def sum_H(H, Lay=CLay()):
    for lay in H:
        for fork in lay:
            if fork: Lay.add_lay(fork)  # merge in dfork
    return Lay

def sum_H_(Q, root, rev=0, DerH=[]):  # sum derH in link_|node_
    for derH in Q: add_H(DerH, derH, root, rev)
    return DerH

def add_H(H, h, root, rev=0, rn=1):  # add fork derHs

    for Lay, lay in zip_longest(H,h):  # different len if lay-selective comp
        if lay:
            if Lay:
                for F, f in zip_longest(Lay, lay):
                    if f:
                        if F: F.add_lay(f)
                        else: Lay[1] = f.copy_(rev)  # default mfork
            else:
                H += [[f.copy_(rev) if f else [] for f in lay]]
            for fork in lay:
                if fork: root.derTT += fork.derTT*rn; root.Et += fork.Et*rn
    return H

def sum_N_(node_, root=None, fC=0):  # form cluster G

    G = copy_(node_[0], root, init = 0 if fC else 1)
    if fC: G.L_=[]; G.N_= [node_[0]]
    for n in node_[1:]:
        add_N(G,n,fC)
    if G.alt: # list
        G.alt = sum_N_(G.alt)  # internal
        if not fC and val_(G.alt.Et, aw=G.olp, _Et=G.Et, fi=0):
            cross_comp(G.alt, G.olp+clust_w)
    G.olp /= len(node_)
    for L in G.L_:
        if L not in G.L_: G.Et += L.Et; G.L_+=[L]
    if fC: norm_(G, len(node_))
    return G   # no rim

def add_N(N,n, fmerge=0, fC=0):

    if fmerge:
        for node in n.N_: node.root=N; N.N_ += [node]
        N.L_ += n.L_  # no L.root assign
    elif fC and n.rim: add_N(N.rim, n.rim)  # skip other params?
    else:
        n.root=N; N.N_ += [n]
        if n.rim: N.L_ += n.rim.L_ if N.fi else n.rim.N_  # rim else nodet
    if n.alt: N.alt += [n.alt]
    if n.H: add_NH(N.H, n.H, root=N)
    if n.lH: add_NH(N.lH, n.lH, root=N)
    rn = n.Et[2]/N.Et[2]
    if n.derH: add_H(N.derH,n.derH, N, rn)
    for Par,par in zip((N.angle, N.baseT, N.derTT), (n.angle, n.baseT, n.derTT)):
        Par += par * rn
    N.Et += n.Et * rn
    N.olp = (N.olp + n.olp * rn) / 2  # ave?
    N.yx = (N.yx + n.yx * rn) / 2
    N.span = max(N.span,n.span)
    N.box = extend_box(N.box, n.box)
    if hasattr(n,'C_') and hasattr(N,'C_'):
        N.C_ += n.C_; N.vo_ += n.vo_
    return N

def add_NH(H, h, root, rn=1):

    for Lev, lev in zip_longest(H, h, fillvalue=None):  # always aligned?
        if lev:
            if Lev: add_N(Lev,lev)
            else:   H += [copy_(lev, root)]

def extend_box(_box, box):
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return min(y0,_y0), min(x0,_x0), max(yn,_yn), max(xn,_xn)

def L2N(link_):
    for L in link_: L.mL_t = [[],[]]; L.compared_,L.visited_ = [],[]; L.rim = CN(L_=[[],[]])
    return link_

def PP2N(PP, frame):

    P_, link_, vert, latuple, A, S, box, yx, Et = PP
    baseT = np.array(latuple[:4])
    [mM,mD,mI,mG,mA,mL], [dM,dD,dI,dG,dA,dL] = vert  # re-pack in derTT:
    derTT = np.array([[mM,mD,mL,1,mI,mG,mA,mL], [dM,dD,dL,1,dI,dG,dA,dL]])
    derH = [[CLay(node_=P_, link_=link_, derTT=deepcopy(derTT)), CLay()]]  # empty dfork
    y,x,Y,X = box; dy,dx = Y-y,X-x  # A = (dy,dx); L = np.hypot(dy,dx), rolp = 1
    et = np.array([*np.sum([L.Et for L in link_],axis=0), 1]) if link_ else np.array([.0,.0,1.])  # n=1

    return CN(root=frame, fi=1, Et=Et+et, N_=P_,L_=link_, baseT=baseT, derTT=derTT, derH=derH, box=box, yx=yx, span=np.hypot(dy/2,dx/2))

# not used:
def sort_H(H, fi):  # re-assign olp and form priority indices for comp_tree, if selective and aligned

    i_ = []  # priority indices
    for i, lay in enumerate(sorted(H.node_, key=lambda lay: lay.Et[fi], reverse=True)):
        di = lay.i - i  # lay index in H
        lay.olp += di  # derR - valR
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

def val_H(H):
    derTT = np.zeros((2,8)); Et = np.zeros(3)
    for lay in H:
        for fork in lay:
            if fork: derTT += fork.derTT; Et += fork.Et
    return derTT, Et

def prj_TT(_Lay, proj, dec):
    Lay = _Lay.copy_(); Lay.derTT[1] *= proj * dec  # only ds are projected?
    return Lay

def prj_dH(_H, proj, dec):
    H = []
    for lay in _H:
        H += [[lay[0].copy_(), prj_TT(lay[1],proj,dec) if lay[1] else []]]  # two-fork
    return H

def comp_prj_dH(_N,N, ddH, rn, link, angle, span, dec):  # comp combined int proj to actual ddH, as in project_N_

    _cos_da= angle.dot(_N.angle) / (span *_N.span)  # .dot for scalar cos_da
    cos_da = angle.dot(N.angle) / (span * N.span)
    _rdist = span/_N.span
    rdist  = span/ N.span
    prj_DH = add_H( prj_dH(_N.derH[1:], _cos_da *_rdist, _rdist*dec),  # derH[0] is positionals
                    prj_dH( N.derH[1:], cos_da * rdist, rdist*dec),
                    link)  # comb proj dHs | comp dH ) comb ddHs?
    # Et+= confirm:
    dddH = comp_H(prj_DH, ddH[1:], rn, link, link.derTT, link.Et)  # ddH[1:] maps to N.derH[1:]
    append_dH(ddH, dddH)
    # merged mfork += [merged dfork], keep nesting, Hs are aligned?

def project_N_(Fg, yx):

    dy,dx = Fg.yx - yx
    Fdist = np.hypot(dy,dx)   # external dist
    rdist = Fdist / Fg.span
    Angle = np.array([dy,dx]) # external angle
    angle = np.sum([L.angle for L in Fg.L_])
    cos_d = angle.dot(Angle) / (np.hypot(*angle) * Fdist)
    # difference between external and internal angles, *= rdist
    ET = np.zeros(3); DerTT = np.zeros((2,8))
    N_ = []
    for _N in Fg.N_:  # sum _N-specific projections for cross_comp
        if len(_N.derH) < 2: continue
        M,D,n = _N.Et
        dec = rdist * (M/(M+D))  # match decay rate, * ddecay for ds?
        prj_H = prj_dH(_N.derH[1:], cos_d * rdist, dec)  # derH[0] is positionals
        prjTT, pEt = val_H(prj_H)  # sum only ds here?
        pD = pEt[1]*dec; dM = M*dec
        pM = dM - pD * (dM/(ave*n))  # -= borrow, regardless of surprise?
        pEt = np.array([pM, pD, n])
        if val_(pEt, aw=clust_w):
            ET+=pEt; DerTT+=prjTT
            N_ += [CN(N_=_N.N_, Et=pEt, derTT=prjTT, derH=prj_H, root=CN())]  # same target position?
    # proj Fg:
    if val_(ET, mw=len(N_)*Lw, aw=clust_w):
        return CN(N_=N_,L_=Fg.L_,Et=ET, derTT=DerTT)  # proj Fg, add Prj_H?

def feedback(root):  # adjust weights: all aves *= rV, ultimately differential backprop per ave?

    rv_t = np.ones((2,8))  # sum derTT coefs: m_,d_ [M,D,n,o, I,G,A,L] / Et, baseT, dimension
    rM, rD, rVd = 1, 1, 0
    hLt = sum_N_(root.L_)  # links between top nodes
    _derTT = np.sum([l.derTT for l in hLt.N_])  # _derTT[np.where(derTT==0)] = 1e-7
    for lev in reversed(root.H):  # top-down
        if not lev.lH: continue
        Lt = lev.lH[-1]  # dfork
        _m, _d, _n = hLt.Et; m, d, n = Lt.Et
        rM += (_m / _n) / (m / n)  # relative higher val, | simple relative val?
        rD += (_d / _n) / (d / n)
        derTT = np.sum([l.derTT for l in Lt.N_])  # top link_ is all comp results
        rv_t += np.abs((_derTT / _n) / (derTT / n))
        if Lt.lH:  # ddfork only, not recursive?
            # intra-level recursion in dfork
            rVd, rv_td = feedback(Lt)
            rv_t = rv_t + rv_td

    return rM+rD+rVd, rv_t

def project_focus(PV__, y,x, Fg):  # radial accum of projected focus value in PV__

    m,d,n = Fg.Et
    V = (m-ave*n) + (d-avd*n)
    dy,dx = Fg.angle; a = dy/ max(dx,1e-7)  # average link_ orientation, projection
    decay = (ave / (Fg.baseT[0]/n)) * (wYX / ave_dist)  # base decay = ave_match / ave_template * rel dist (ave_dist is a placeholder)
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

def agg_frame(floc, image, iY, iX, rV=1, rv_t=[], fproj=0):  # search foci within image, additionally nested if floc

    if floc:   # focal img, converted to dert__
        dert__ = image
    else:      # global img
        dert__ = comp_pixel(image)
        global ave, Lw, int_w, loop_w, clust_w, ave_dist, ave_med, med_w
        ave, Lw, int_w, loop_w, clust_w, ave_dist, ave_med, med_w = np.array([ave, Lw, int_w, loop_w, clust_w, ave_dist, ave_med, med_w]) / rV
        # fb rws: ~rvs
    nY,nX = dert__.shape[-2] // iY, dert__.shape[-1] // iX  # n complete blocks
    Y, X  = nY * iY, nX * iX  # sub-frame dims
    frame = CN(box=np.array([0,0,Y,X]), yx=np.array([Y//2,X//2]))
    dert__= dert__[:,:Y,:X]  # drop partial rows/cols
    win__ = dert__.reshape(dert__.shape[0], iY,iX, nY,nX).swapaxes(1,2)  # dert=5, wY=64, wX=64, nY=13, nX=20
    PV__  = win__[3].sum( axis=(0,1)) * int_w  # init proj vals = sum G in dert[3],       shape: nY=13, nX=20
    aw = clust_w * 20
    while np.max(PV__) > ave * aw:  # max G * int_w + pV
        # max win index:
        y,x = np.unravel_index(PV__.argmax(), PV__.shape)
        PV__[y,x] = -np.inf  # to skip, | separate in__?
        if floc:
            Fg = frame_blobs_root( win__[:,:,:,y,x], rV)  # [dert, iY, iX, nY, nX]
            Fg = vect_root(Fg, rV, rv_t)  # focal dert__ clustering
            cross_comp(Fg, rc=frame.olp)
        else:
            Fg = agg_frame(1, win__[:,:,:,y,x], wY,wX, rV=1, rv_t=[])  # use global wY,wX in nested call
            if Fg and Fg.L_:  # only after cross_comp(PP_)
                rV, rv_t = feedback(Fg)  # adjust filters
        if Fg and Fg.L_:
            if fproj and val_(Fg.Et, mw=(len(Fg.N_)-1)*Lw, aw=Fg.olp+loop_w*20):
                pFg = project_N_(Fg, np.array([y,x]))
                if pFg:
                    cross_comp(pFg, rc=Fg.olp)  # skip compared_ in FG cross_comp
                    if val_(pFg.Et, mw=(len(pFg.N_)-1)*Lw, aw=pFg.olp+clust_w*20):
                        project_focus(PV__, y,x, Fg)  # += proj val in PV__
            # no target proj
            add_N(frame, Fg, fmerge=1)
            aw = clust_w * 20 * frame.Et[2] * frame.olp

    if frame.N_ and val_(frame.Et, mw=(len(frame.N_)-1)*Lw, aw=frame.olp+loop_w*20) > 0:
        # xcomp Fg.N_s:
        cross_comp(frame, rc=frame.olp+loop_w)  # recursive
        if not floc: return frame  # foci are not preserved

if __name__ == "__main__":  # './images/toucan_small.jpg' './images/raccoon_eye.jpeg', add larger global image

    iY,iX = imread('./images/toucan_small.jpg').shape
    frame = agg_frame(0, image=imread('./images/toucan.jpg'), iY=iY, iX=iX)
    # search frames ( foci inside image