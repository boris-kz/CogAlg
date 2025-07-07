import numpy as np, weakref
from copy import copy, deepcopy
from functools import reduce
from itertools import zip_longest, combinations, chain, product;  from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, imread, unpack_blob_, comp_pixel, CBase
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

class CN(CBase):
    name = "node"
    def __init__(n, **kwargs):
        super().__init__()
        n.N_ = kwargs.get('N_',[])  # N_| nrim
        n.L_ = kwargs.get('L_',[])  # L_| rim
        n.nH = kwargs.get('nH',[])  # top-down hierarchy of sub-node_s: CN(sum_N_(Nt_))/ lev, with single added-layer derH, empty nH
        n.lH = kwargs.get('lH',[])  # bottom-up hierarchy of L_ graphs: CN(sum_N_(Lt_))/ lev, within each nH lev
        n.Et = kwargs.get('Et',np.zeros(3))  # sum from L_ or rims
        n.olp= kwargs.get('olp',1)  # overlap to other Ns, same for links?
        n.fi = kwargs.get('fi', 1)  # if G else 0, fd_: list of forks forming G?
        n.derH  = kwargs.get('derH', [])  # sum from L_ or rims
        n.derTT = kwargs.get('derTT',np.zeros((2,8)))  # sum derH
        n.baseT = kwargs.get('baseT',np.zeros(4))
        n.yx    = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.rng   = kwargs.get('rng',1)  # or med: loop count in comp_node_|link_
        n.box   = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.span  = kwargs.get('span',0) # distance in nodet or aRad, comp with baseT and len(N_) but not additive?
        n.angle = kwargs.get('angle',np.zeros(2))  # dy,dx
        n.fin   = kwargs.get('fin',0)  # in cluster, temporary?
        n.root  = kwargs.get('root',[])  # immediate only
        # nested CN | lists, same for Ct,Lt but with heavy overlap?
        n.rim = kwargs.get('rim',[])  # rim.L_= links, rim.N_= nodes, the rest is summed from links
        n.alt = kwargs.get('alt',[])  # adjacent (contour) gap+overlap alt-fork graphs, converted to CG, empty alt.alt_: select+?
        # n.fork_tree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def __bool__(n): return bool(n.N_)

def copy_(N, root=None, init=0):

    C = CN(root=root)
    if init:  # init G with N
        C.N_,C.nH,C.lH, N.root = ([N],[],[],C)
        if N.rim: N.L_ = N.rim.L_ if N.fi else N.rim.N_  # rim else nodet
        if N.alt: C.alt = [N.alt]  # not internalized as rim
    else:
        C.N_,C.L_,C.nH,C.lH, N.root = (list(N.N_),list(N.L_),list(N.nH),list(N.lH), root if root else N.root)
        if N.rim: C.rim = copy_(N.rim)
        if N.alt: C.alt = copy_(N.alt)
    C.derH  = [lay.copy_() for lay in N.derH]
    C.derTT = deepcopy(N.derTT)
    for attr in ['Et', 'baseT','yx','box','angle']: setattr(C, attr, copy(getattr(N, attr)))
    for attr in ['olp','rng', 'fi', 'fin', 'span']: setattr(C, attr, getattr(N, attr))
    return C

def flat(rim):
    L_ = []
    for L in rim:  # leaves of deep binary L_| N_ tree, in shape of nodet
        if isinstance(L, list): L_.extend(flat(L))
        else: L_ += [L]
    return L_

ave, avd, arn, aI, aveB, aveR, Lw, intw, loopw, centw, contw = 10, 10, 1.2, 100, 100, 3, 5, 2, 5, 10, 15  # value filters + weights
adist, amed, distw, medw = 10, 3, 2, 2  # cost filters + weights
wM, wD, wN, wO, wI, wG, wA, wL = 10, 10, 20, 20, 1, 1, 20, 20  # der params higher-scope weights = reversed relative estimated ave?
w_t = np.ones((2,8))  # fb weights per derTT, adjust in agg+
wY = wX = 64; wYX = np.hypot(wY,wX)  # focus dimensions
'''
initial PP_ cross_comp and connectivity clustering to initialize focal frame graph, no recursion:
'''
def vect_root(Fg, rV=1, ww_t=[]):  # init for agg+:
    if np.any(ww_t):
        global ave, avd, arn, aveB, aveR, Lw, adist, amed, intw, loopw, centw, contw, wM, wD, wN, wO, wI, wG, wA, wL, w_t
        ave, avd, arn, aveB, aveR, Lw, adist, amed, intw, loopw, centw, contw = (
            np.array([ave,avd,arn,aveB,aveR, Lw, adist, amed, intw, loopw, centw, contw]) / rV)  # projected value change
        w_t = np.multiply([[wM,wD,wN,wO,wI,wG,wA,wL]], ww_t)  # or dw_ ~= w_/ 2?
        ww_t = np.delete(ww_t,(2,3), axis=1)  #-> comp_slice, = np.array([(*ww_t[0][:2],*ww_t[0][4:]),(*ww_t[0][:2],*ww_t[1][4:])])
    blob_ = unpack_blob_(Fg)
    Fg.N_,Fg.L_ = [],[]; lev = CN(); derlay = CLay(root=Fg)
    for blob in blob_:
        if not blob.sign and blob.G > aveB:
            edge = slice_edge(blob, rV)
            if edge.G*((len(edge.P_)-1)*Lw) > ave * sum([P.latuple[4] for P in edge.P_]):
                Et = comp_slice(edge, rV, ww_t)  # to scale vert
                if np.any(Et) and Et[0] > ave * Et[2] * contw:
                    cluster_edge(edge, Fg, lev, derlay)  # may be skipped
    Fg.derH = [derlay]
    if lev: Fg.nH += [lev]
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
            if val_(et, mw=(len(node_)-1)*Lw, aw=2+contw) > 0:  # rc=2
                Et += et
                G_ += [sum2graph(frame, node_,link_,[], et, olp=1, rng=1)]  # single-lay link_derH
        return G_, Et

    def comp_PP_(PP_):
        N_,L_,mEt,dEt = [],[],np.zeros(3),np.zeros(3)
        for PP in PP_: PP.rim = CN()
        for _G, G in combinations(PP_, r=2):
            dy,dx = np.subtract(_G.yx,G.yx); dist = np.hypot(dy,dx)
            if dist - (G.span+_G.span) < adist / 10:  # very short here
                L = comp_N(_G,G, ave, angle=np.array([dy,dx]), span=dist, fdeep=1)
                m, d, n = L.Et
                if m > ave * n * _G.olp * loopw: mEt += L.Et; N_ += [_G,G]  # mL_ += [L]
                if d > avd * n * _G.olp * loopw: dEt += L.Et  # dL_ += [L]
                L_ += [L]
        dEt[2] = dEt[2] or 1e-7
        return set(N_),L_,mEt,dEt  # can't be empty

    PP_ = edge.node_
    if val_(edge.Et, (len(PP_)-edge.rng)*Lw, loopw) > 0:
        PP_,L_,mEt,dEt = comp_PP_([PP2N(PP,frame) for PP in PP_])
        if val_(mEt, (len(PP_)-1)*Lw, contw, fi=1) > 0:
            G_, Et = cluster_PP_(copy(PP_))  # can't be empty
        else: G_ = []
        if G_: frame.N_ += G_; lev.N_ += PP_; lev.Et += Et
        else:  frame.N_ += PP_  # PPm_
        lev.L_+= L_; lev.Et = mEt+dEt  # links between PPms
        for l in L_: derlay.add_lay(l.derH[0]); frame.baseT+=l.baseT; frame.derTT+=l.derTT; frame.Et += l.Et
        if val_(dEt, (len(L_)-1)*Lw, 2+contw, fi=0) > 0:
            Lt = cluster(frame, PP_, rc=2,fi=0)
            if Lt:  # L_ graphs
                if lev.lH: lev.lH[0].N_ += Lt.N_; lev.lH[0].Et += Lt.Et
                else:      lev.lH += [Lt]
                lev.Et += Lt.Et

def val_(Et, mw=1, aw=1, _Et=np.zeros(3), fi=1):  # m,d eval per cluster or cross_comp

    am = ave * aw  # includes olp, M /= max I | M+D? div comp / mag disparity vs. span norm
    ad = avd * aw  # diff value is borrowed from co-projected or higher-scope match value
    m,d,n = Et
    if fi==2: val = np.array([m-am, d-ad]) * mw
    else:     val = (m-am if fi else d-ad) * mw  # m: m/ (m+d), d: d/ (m+d)?
    if _Et[2]:
        # empty if n=0, borrow rational deviation of alt contour if fi else root Et, not circular
        _m,_d,_n = _Et; _mw = mw*(_n/n)
        if fi==2: val *= np.array([_d/ad, _m/am]) * _mw
        else:     val *= (_d/ad if fi else _m/am) * _mw

    return val
''' 
Core process per agg level, as described in top docstring:
Cross-compare nodes, evaluate incremental-derivation cross-comp of new >ave difference links, recursively.
 
Select sparse exemplars of strong node types, may covert to sub-centroids, refined & extended by mutual match.
Connectivity-cluster exemplars or centroids by >ave match links, correlation-cluster links by >ave difference.

Form complemented clusters (core+contour) for recursive higher-composition cross_comp, reorder by eigenvalues. 
Feedback coords to bottom level or prior-level in parallel pipelines, filter updates in more coarse cycles 
'''

def cross_comp(root, rc, fi=1):  # rng+ and der+ cross-comp and clustering, + recursion

    N_,L_,Et = comp_node_(root.N_,rc) if fi else comp_link_(root.N_,rc)  # rc: redundancy+olp
    if len(L_) > 1:
        mV,dV = val_(Et, (len(L_)-1)*Lw, rc+loopw, fi=2)
        if dV > 0:
            if root.L_: root.lH += [sum_N_(root.L_)]  # replace L_ with agg+ L_:
            root.L_=L_; root.Et += Et
            if dV >avd: Lt = cross_comp(CN(N_=L2N(L_)), rc+contw, fi=0)  # -> Lt.nH, +2 der layers
            else:       Lt = cluster(root, N_, rc+contw, fi=0)  # cluster N.rim.L_s, +1 layer
            if Lt: root.lH += [Lt] + Lt.nH; root.Et += Lt.Et; root.derH += Lt.derH  # append new der lays
        if mV > 0:
            Nt = cluster(root, N_, rc+loopw, fi)  # with get_exemplars, cluster_C_, rng-banded if fi
            if Nt and val_(Nt.Et, (len(Nt.N_)-1)*Lw, rc+loopw+Nt.rng, _Et=Et) > 0:
                Nt = cross_comp(Nt, rc+loopw) or Nt  # agg+
            if Nt:
                _H = root.nH; root.nH = []
                Nt.nH = _H + [root] + Nt.nH  # pack root in Nt.nH, has own L_,lH
                # recursive feedback:
                return Nt

def comp_node_(_N_, rc):  # rng+ forms layer of rim and extH per N?

    N__,L_,ET = [],[],np.zeros(3); rng,olp_ = 1,[]  # range banded if frng only?
    for n in _N_:
        n.compared_ = []; n.rim = CN()
    while True:  # _vM
        Gp_ = []  # [G pair + co-positionals], for top-nested Ns, unless cross-nesting comp:
        for _G, G in combinations(_N_, r=2):
            if _G in G.compared_ or len(_G.nH) != len(G.nH):  # | root.nH: comp top nodes only
                continue
            radii = _G.span+G.span; dy,dx = np.subtract(_G.yx,G.yx); dist = np.hypot(dy,dx)
            Gp_ += [(_G,G, dy,dx, radii, dist)]
        N_,Et = [],np.zeros(3)
        for Gp in Gp_:
            _G, G, dy,dx, radii, dist = Gp
            (_m,_,_n),(m,_,n) = _G.Et,G.Et; olp = (_G.olp+G.olp)/2; _m = _m * intw +_G.rim.Et[0]; m = m * intw + G.rim.Et[0]
            # density / comp_N: et|M, 0/rng=1?
            max_dist = adist * (radii/aveR) * ((_m+m)/(ave*(_n+n+0))/ intw)  # ave_dist * radii * induction
            if max_dist > dist or set(_G.rim.N_) & set(G.rim.N_):
                # comp if close or share matching mediators: add to inhibited?
                Link = comp_N(_G,G, ave, angle=np.array([dy,dx]), span=dist, fdeep = dist < max_dist/2, rng=rng)
                L_ += [Link]  # include -ve links
                if val_(Link.Et, aw=loopw*olp) > 0:
                    Et += Link.Et; olp_ += [olp]  # link.olp is the same with o
                    for n,_n in zip((_G,G),(G,_G)):
                        n.compared_ += [_n]
                        if n not in N_ and val_(n.rim.Et, aw= rc+rng-1+loopw+olp) > 0:  # cost+/ rng
                            N_ += [n]  # for rng+ and exemplar eval
        if N_:
            N__ += [N_]; ET += Et
            if val_(Et, (len(N_)-1)*Lw, loopw * sum(olp_)/max(1,len(olp_))) > 0:  # current-rng vM
                _N_ = N_; rng += 1; olp_ = []  # reset
            else: break  # low projected rng+ vM
        else: break
    return N__,L_,ET

def comp_link_(iL_, rc):  # comp CLs via directional node-mediated link tracing: der+'rng+ in root.link_ rim_t node rims

    for L in iL_:  # init mL_t: nodet-mediated Ls:
        for rev, N, mL_ in zip((0, 1), L.N_, L.mL_t):  # L.mL_t is empty
            for _L,_rev in flat(N.rim.L_):
                if _L is not L and _L in iL_:
                    if L.Et[0] > ave * L.Et[2] * loopw:
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
                    Link = comp_N(_L,L, ave*rc, np.array([dy,dx]), np.hypot(dy,dx), -1 if rev else 1)  # d = -d if L is reversed relative to _L
                    Link.med = med
                    LL_ += [Link]; Et += Link.Et  # include -ves, link order: nodet < L < rim, mN.rim || L
                    for l,_l in zip((_L,L),(L,_L)):
                        l.compared_ += [_l]
                        if l not in L_ and val_(l.rim.Et, aw= rc+med+loopw) > 0:  # cost+/ rng
                            L_ += [l]  # for med+ and exemplar eval
        if L_: L__ += L_; ET += Et
        # extend mL_t per last medL:
        if Et[0] > ave * Et[2] * (rc + loopw + med*medw):  # project prior-loop value, med adds fixed cost
            _L_, ext_Et = set(), np.zeros(3)
            for L in L_:
                mL_t, lEt = [set(),set()], np.zeros(3)  # __Ls per L
                for mL_,_mL_ in zip(mL_t, L.mL_t):
                    for _L, rev in _mL_:
                        for _rev, N in zip((0,1), _L.N_):
                            rim = flat(N.rim.L_)
                            if len(rim) == med:  # append in comp loop
                                for __L,__rev in rim:
                                    if __L in L.visited_ or __L not in iL_: continue
                                    L.visited_ += [__L]; __L.visited_ += [L]
                                    if __L.Et[0] > ave * __L.Et[2] * loopw:
                                        mL_.add((__L, rev ^_rev ^__rev))  # combine reversals: 2 * 2 mLs, but 1st 2 are pre-combined
                                        lEt += __L.Et
                if lEt[0] > ave * lEt[2]:  # L rng+, vs. L comp above, add coef
                    L.mL_t = mL_t; _L_.add(L); ext_Et += lEt
            # refine eval by extension D:
            if ext_Et[0] > ave * ext_Et[2] * (loopw+ med*medw): med += 1
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

def comp_N(_N,N, ave, angle=None, span=None, dir=1, fdeep=0, rng=1):  # compare links, relative N direction = 1|-1, no angle,span?

    derTT, Et, rn = base_comp(_N, N, dir); fi = N.fi
    # link M,D,A:
    baseT = np.array([*(_N.baseT[:2] + N.baseT[:2]*rn) / 2, *angle])  # redundant angle for generic base_comp, also span-> density?
    _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
    o = (_N.olp+N.olp) / 2
    Link = CN(rng=rng, olp=o, N_=[_N,N], baseT=baseT, derTT=derTT, yx=np.add(_N.yx,N.yx)/2, span=span, angle=angle, box=box, fi=0)
    Link.derH = [CLay(root=Link, Et=Et, node_=[_N,N],link_=[Link], derTT=copy(derTT))]
    # derH:
    if fdeep and (val_(Et, len(N.derH)-2, o) > 0 or fi==0):  # else derH is dext,vert
        Link.derH += comp_H(_N.derH, N.derH, rn, Link, derTT,Et)  # append
    if fi:
        # each: separate overlap, may be CN or list: default individual cross_comp, same for rim and alt?
        if (_N.alt and N.alt) and val_(_N.alt.Et+N.alt.Et, aw=2+o, fi=0) > 0:  # contour
            Link.alt = comp_N(_N.alt, N.alt, ave*2, angle,span); Et += Link.alt.Et
        if (_N.Ct and N.Ct) and val_(_N.Ct.Et+N.Ct.Et, aw=2+o) > 0:  # sub-centroids
            Link.Ct = comp_N(_N.Ct, N.Ct, ave); Et += Link.Ct.Et  # merge,refine iCt_ in new G, in addition to new N_ Ct?
        # + cross_comp N.N_, comp rim, as in cluster_C_?
    Link.Et = Et
    for rev, node, _node in zip((0,1),(N,_N),(_N,N)):  # reverse Link dir in _N.rimt
        # simplified add_N_(rim), +|-:
        if fi: node.rim.L_ += [(Link,rev)]; node.rim.N_ += [_node]
        else:  node.rim.L_[1-rev] += [(Link,rev)]; node.rim.N_[1-rev] += [_node]  # rimt opposite to _N,N dir
        node.rim.Et += Et; node.rim.baseT += baseT; node.rim.derTT += derTT  # simplified add_H(node.rim.derH, Link.derH, root=node, rev=rev)?

    return Link

def rolp(N, _N_, fi, E=0, R=0): # rel V of N.rim | L_ overlap with _N_: inhibition or shared zone
    if R:
        olp_ = [n for n in (N.N_ if fi else N.L_) if n in _N_]  # fi=0 currently not used
    else:
        olp_ = [L for L, _ in flat(N.rim.N_ if fi else N.rim.L_)
                if any(n in _N_ for n in L.N_)]
    if olp_:
        oEt = np.sum([i.Et for i in olp_], axis=0)
        _Et = N.Et if (R or E) else N.rim.Et  # not sure
        rV = (oEt[1-fi]/oEt[2]) / (_Et[1-fi]/_Et[2])
        return val_(N.Et,centw) * rV  # contw for cluster?
    else:
        return 0

def get_exemplars(N_, rc, fi):  # get sparse nodes by multi-layer non-maximum suppression

    E_, Et = [], np.zeros(3)  # ~ point cloud of focal nodes
    _E_ = set()  # prior = stronger:
    for rdn, N in enumerate(sorted(N_, key=lambda n: n.rim.Et[1-fi]/ n.rim.Et[2], reverse=True), start=1):
        # ave *= overlap by stronger-E inhibition zones
        if val_(N.rim.Et, aw = rc + rdn + loopw + rolp(N, list(_E_), fi, E=1), fi=fi) > 0: # rolp is cost
            Et += N.rim.Et; _E_.update(N.rim.N_ if fi else flat(N.rim.L_))
            N.sel = 1  # select for cluster_N_
            E_ += [N]  # exemplars
        else:
            break  # the rest of N_ is weaker, trace via rims
    return N_, Et

def cluster(root, N_, rc, fi):  # clustering root

    Nf_ = list(set([N for n_ in N_ for N in n_])) if fi else N_
    E_,Et = get_exemplars(Nf_, rc, fi)  # with C_form eval

    if val_(Et, (len(E_)-1)*Lw, rc+contw, root.Et, fi) > 0:
        for n in Nf_: n.sel=0
        if fi:
            Nt = []  # bottom-up rng-banded clustering:
            for rng, rN_ in enumerate(N_, start=1):
                rE_ = [n for n in rN_ if n.sel]
                aw = rc * rng + contw  # cluster Nf_ via rng exemplars:
                if rE_ and val_(np.sum([n.Et for n in rE_], axis=0), (len(rE_)-1)*Lw, aw) > 0:
                    Nt = cluster_N_(root, rE_, aw, rng) or Nt  # keep top-rng Gt
        else:
            Nt = cluster_L_(root, N_, E_, rc)
        return Nt

def cluster_N_(root, E_, rc, rng=1):  # connectivity cluster exemplar nodes via rim or links via nodet or rimt

    G_ = []  # flood-fill Gs, exclusive per fork,ave, only centroid-based can be fuzzy
    for n in root.N_: n.fin = 0
    for N in E_:
        if N.fin: continue
        # init N cluster with rim | root:
        if rng==1 or N.root.rng==1: # N is not rng-nested
            node_,link_,llink_,Et, olp = [N],[],[], N.Et+N.rim.Et, N.olp
            for l,_ in flat(N.rim.L_):  # +ve
                if l.rng==rng and val_(l.Et,aw=rc+contw)>0: link_ += [l]
                elif l.rng>rng: llink_ += [l]  # longer-rng rim
        else:
            n = N; R = n.root  # N.rng=1, R.rng > 1, cluster top-rng roots instead
            while R.root and R.root.rng > n.rng: n = R; R = R.root
            if R.fin: continue
            node_,link_,llink_,Et, olp = [R],R.L_,R.hL_,copy(R.Et),R.olp
            R.fin = 1
        nrc = rc+olp; N.fin = 1  # extend N cluster:
        for L in link_[:]:  # cluster nodes via links
            for _N in L.N_:
                if _N not in root.N_ or _N.fin: continue  # connectivity clusters don't overlap
                if rng==1 or _N.root.rng==1:  # not rng-nested
                    if rolp(N, [l for l,_ in flat(N.rim.L_)], fi=1) > ave*nrc:
                        node_ +=[_N]; Et += _N.Et+_N.rim.Et; olp+=_N.olp; _N.fin=1
                        for l,_ in flat(N.rim.L_):
                            if l not in link_ and l.rng == rng: link_ += [l]
                            elif l not in llink_ and l.rng>rng: llink_+= [l]  # longer-rng rim
                else:
                    _n =_N; _R=_n.root  # _N.rng=1, _R.rng > 1, cluster top-rng roots if rim intersect:
                    while _R.root and _R.root.rng > _n.rng: _n=_R; _R=_R.root
                    if not _R.fin and rolp(N,link_,fi=1, R=1) > ave*nrc:
                        # cluster by N_|L_ connectivity: oN_ = list(set(N.N_) & set(_N.N_))  # exclude from comp and merge?
                        link_ = list(set(link_+_R.link_)); llink_ = list(set(llink_+_R.hL_))
                        node_+= [_R]; Et +=_R.Et; olp += _R.olp; _R.fin = 1; _N.fin = 1
                nrc = rc+olp
        node_ = list(set(node_))
        alt = [L.root for L in link_ if L.root]  # get link clusters, individual rims are too weak for contour
        if alt:
            alt = sum_N_(list(set(alt)))
            if val_(alt.Et, (len(alt.N_)-1)*Lw, olp, _Et=Et, fi=0):
                alt = cross_comp(alt, olp) or alt
            _Et = alt.Et
        else: _Et = np.zeros(3)
        V = val_(Et, (len(node_)-1)*Lw, nrc, _Et)
        if V > 0:
            if V > ave*centw: Ct = cluster_C_(root, node_, rc+centw)  # form N.c_, doesn't affect clustering?
            else:             Ct = []   # option to keep as list, same for alt and rim?
            G_ += [sum2graph(root, node_, link_, llink_, Et, olp, rng, Ct, alt)]
        if G_:
            return sum_N_(G_, root)   # root N_|L_ replacement

def cluster_L_(root, N_, E_, rc):  # connectivity cluster links from exemplar node.rims

    G_ = []  # cluster by exemplar rim overlap
    fi = not E_[0].rim  # empty in base L_
    for n in N_: n.fin = 0
    for E in E_:
        if E.fin: continue
        node_,Et, olp = [E], E.rim.Et, E.olp
        link_ = [l for l,_ in flat(E.rim.L_) if l.Et[1] > avd*l.Et[2]*rc]  # L val is always diff?
        nrc = rc+olp; E.fin = 1  # extend N cluster:
        if fi:  # cluster Ls by L diff
            for _N in E.rim.N_:  # links here
                if _N.fin: continue
                _N.fin = 1
                for n in _N.N_:  # L.nodet
                    for L,_ in flat(n.rim.L_):
                        if L not in node_ and _N.Et[1] > avd * _N.Et[2] * nrc:  # diff
                            link_+= [_N]; node_+= [L]; Et += L.Et; olp+= L.olp  # /= len node_
        else:  # cluster Ls by rimt match
            for L in link_[:]: # snapshot
                for _N in L.N_:
                    if _N in N_ and not _N.fin and rolp(E, [l for l,_ in flat(_N.rim.L_)], fi=0) > ave * nrc:
                        node_ +=[_N]; Et += _N.Et+_N.rim.Et; olp+=_N.olp; _N.fin=1
                        for l,_ in flat(_N.rim.L_):
                            if l not in link_: link_ += [l]
        node_ = list(set(node_))
        V = val_(Et, (len(node_)-1)*Lw, rc+olp, _Et=root.Et, fi=fi)
        if V > 0:
            if V > ave*centw: Ct = cluster_C_(root, node_, rc+centw)  # form N.c_, doesn't affect clustering?
            else:             Ct = []   # option to keep as list, same for alt and rim?
            G_ += [sum2graph(root, node_, link_,[], Et, olp, 1, Ct)]
        if G_:
            return sum_N_(G_,root)

def cluster_C_(root, E_, rc, fi=1):  # form centroids from exemplar _N_, drifting / competing via rims of new member nodes
                                     # divisive group-wise clustering: if surround n match in C.N_: C.N_+=[n] if match to C
    def comp_C(C, n):
        _,et,_ = base_comp(C, n)
        if C.rim and n.rim: _,_et,_= base_comp(C.rim,n.rim); et +=_et
        if C.alt and n.alt: _,_et,_= base_comp(C.alt,n.alt); et +=_et
        return et
    for n in root.N_: n.C_, n.vo_, n._C_, n._vo_ = [], [], [], []
    _C_ = []; av = ave if fi else avd
    for E in E_:  # N_
        Et = E.rim.Et; C = copy_(E,root); C.N_=[E]; C.L_=[]; C.Et=Et  # init centroid
        rim = flat(E.rim.N_ if fi else E.rim.L_)  # same nesting?
        C._N_ = list({_n for n in rim for _n in flat(n.rim.N_ if fi else n[0].rim.L_)})  # core members + surround for comp to N_ mean
        _C_ += [C]
    while True:  # recompute C, refine C.N_
        C_, ET, O, Dvo, Ave = [], np.zeros(3), 0, np.zeros(2), av*rc*loopw
        _Ct_ = [[c, c.Et[1-fi]/c.Et[2], c.olp] for c in _C_]
        for _C,_v,_o in sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True):
            if _v > Ave*_o:
                C = sum_N_(_C.N_, root=root, fC=1)  # merge rim,alt before cluster_NC_
                _N_,_N__,o,Et,dvo = [],[],0, np.zeros(3),np.zeros(2)  # per C
                for n in _C._N_:  # core + surround
                    if C in n.C_: continue  # clear/ loop?
                    et = comp_C(C,n); v = val_(et,aw=rc,fi=fi)
                    olp = np.sum([vo[0] / v for vo in n.vo_ if vo[0] > v])  # olp: rel n val in stronger Cs, not summed with C
                    vo = np.array([v,olp])  # val, overlap per current root C
                    if et[1-fi]/et[2] > Ave*olp:
                        n.C_ += [C]; n.vo_ += [vo]; Et += et; o+=olp; _N_ += [n]
                        _N__ += flat(n.rim.N_ if fi else n.rim.L_)
                        if C not in n._C_: dvo += vo
                    elif C in n._C_:
                        dvo += n._vo_[n._C_.index(C)]  # old vo_, or pack in _C_?
                if Et[1-fi]/Et[2] > Ave*o:
                    C.Et = Et; C.N_ = list(set(_N_)); C._N_ = list(set(_N__))  # core, surround elements
                    C_+=[C]; ET+=Et; O+=o; Dvo+=dvo  # new incl or excl
            else: break  # the rest is weaker
        if np.sum(Dvo) > Ave*O:  # both V and O represent change, comeasurable?
            _C_ = C_
            for n in root.N_: n._C_ = n.C_; n._vo_= n.vo_; n.C_,n.vo_ = [],[]  # new n.C_s, combine with vo_ in Ct_?
        else:
            break  # converged
    if val_(ET, aw=O+rc+loopw, fi=fi) > 0:  # C_ value, no _Et?
        return sum_N_(E_)

def sum2graph(root, node_,link_,llink_, Et, olp, rng, Ct, alt=[], fC=0):  # sum node,link attrs in graph, aggH in agg+ or player in sub+

    n0 = copy_(node_[0]); derH = n0.derH
    l0 = copy_(link_[0]); DerH = l0.derH
    graph = CN(root=root, fi=1,rng=rng, N_=node_,L_=link_,olp=olp, Et=Et, alt=alt, box=n0.box, baseT=n0.baseT+l0.baseT, derTT=n0.derTT+l0.derTT)
    if Ct: graph.Ct = Ct; graph.Et += Ct.Et  # or keep separate?
    graph.hL_ = llink_
    n0.root = graph; yx_ = [n0.yx]; fg = isinstance(n0.N_[0],CN)  # not PPs
    Nt = copy_(n0)  #->CN, comb forks: add_N(Nt,Nt.Lt)?
    for N in node_[1:]:
        add_H(derH,N.derH,graph); graph.baseT+=N.baseT; graph.derTT+=N.derTT; graph.box=extend_box(graph.box,N.box); yx_+=[N.yx]; N.root = graph
        if fg: add_N(Nt,N)
    for L in link_[1:]:
        add_H(DerH,L.derH,graph); graph.baseT+=L.baseT; graph.derTT+=L.derTT
    graph.derH = add_H(derH,DerH, graph)  # * rn?
    if fg: graph.nH = Nt.nH + [Nt]  # pack prior top level
    yx = np.mean(yx_, axis=0); dy_,dx_ = (yx_-yx).T; dist_ = np.hypot(dy_,dx_)
    graph.span = dist_.mean()  # node centers distance to graph center
    graph.angle = np.sum([l.angle for l in link_],axis=0)
    graph.yx = yx
    if fC:
        m_,M = centroid_M(graph.derTT[0],ave*olp)  # weigh by match to mean m|d
        d_,D = centroid_M(graph.derTT[1],ave*olp)
        graph.derTT = np.array([m_,d_])
        graph.Et = np.array([M,D,Et[2]])

    return graph

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

def comp_H(H,h, rn, root, DerTT=None, ET=None):  # one-fork derH

    derH, derTT, Et = [], np.zeros((2,8)), np.zeros(3)
    for _lay, lay in zip_longest(H,h):  # selective
        if _lay and lay:
            dlay = _lay.comp_lay(lay, rn, root=root)
            derH += [dlay]; derTT = dlay.derTT; Et += dlay.Et
    if Et[2]: DerTT += derTT; ET += Et
    return derH

def sum_H_(Q):  # sum derH in link_|node_, not used
    H = Q[0]; [add_H(H,h) for h in Q[1:]]
    return H

def add_H(H, h, root=0, rev=0, rn=1):  #  layer-wise add|append derH

    for Lay, lay in zip_longest(H,h):  # different len if lay-selective comp
        if lay:
            if Lay: Lay.add_lay(lay, rn)
            else:   H += [lay.copy_(rev)]  # * rn?
            if root: root.derTT += lay.derTT*rn; root.Et += lay.Et*rn
    return H

def sum_N_(node_, root=None, fC=0):  # form cluster G

    G = copy_(node_[0], root, init = 0 if fC else 1)
    if fC: G.L_=[]; G.N_= [node_[0]]
    for n in node_[1:]:
        add_N(G,n,fC)
    if G.alt: # list
        G.alt = sum_N_(G.alt)  # internal
        if not fC and val_(G.alt.Et, aw=G.olp, _Et=G.Et, fi=0):
            cross_comp(G.alt, G.olp+contw)
    G.olp /= len(node_)
    for L in G.L_:
        if L not in G.L_: G.Et += L.Et; G.L_+=[L]
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
    if n.nH: add_NH(N.nH, n.nH, root=N)
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
    for L in link_: L.mL_t = [[],[]]; L.compared_,L.visited_ = [],[]; L.rim = CN(N_=[[],[]],L_=[[],[]])
    return link_

def PP2N(PP, frame):

    P_, link_, vert, latuple, A, S, box, yx, Et = PP
    baseT = np.array(latuple[:4])
    [mM,mD,mI,mG,mA,mL], [dM,dD,dI,dG,dA,dL] = vert  # re-pack in derTT:
    derTT = np.array([[mM,mD,mL,1,mI,mG,mA,mL], [dM,dD,dL,1,dI,dG,dA,dL]])
    derH = [CLay(node_=P_, link_=link_, derTT=deepcopy(derTT))]
    y,x,Y,X = box; dy,dx = Y-y,X-x  # A = (dy,dx); L = np.hypot(dy,dx), rolp = 1
    et = np.array([*np.sum([L.Et for L in link_],axis=0), 1]) if link_ else np.array([.0,.0,1.])  # n=1

    return CN(root=frame, fi=1, Et=Et+et, N_=P_,L_=link_, baseT=baseT, derTT=derTT, derH=derH, box=box, yx=yx, span=np.hypot(dy/2,dx/2))

# not used, make H a centroid of layers, same for nH?
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
    add_H(ddH, dddH, link, 0, rn)

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
        if val_(pEt, aw=contw):
            ET+=pEt; DerTT+=prjTT
            N_ += [CN(N_=_N.N_, Et=pEt, derTT=prjTT, derH=prj_H, root=CN())]  # same target position?
    # proj Fg:
    if val_(ET, mw=len(N_)*Lw, aw=contw):
        return CN(N_=N_,L_=Fg.L_,Et=ET, derTT=DerTT)  # proj Fg, add Prj_H?

def ffeedback(root):  # adjust filters: all aves *= rV, ultimately differential backprop per ave?

    rv_t = np.ones((2,8))  # sum derTT coefs: m_,d_ [M,D,n,o, I,G,A,L] / Et, baseT, dimension
    rM, rD, rVd = 1, 1, 0
    hLt = sum_N_(root.L_)  # links between top nodes
    _derTT = np.sum([l.derTT for l in hLt.N_])  # _derTT[np.where(derTT==0)] = 1e-7
    for lev in reversed(root.nH):  # top-down
        if not lev.lH: continue
        Lt = lev.lH[-1]  # dfork
        _m, _d, _n = hLt.Et; m, d, n = Lt.Et
        rM += (_m / _n) / (m / n)  # relative higher val, | simple relative val?
        rD += (_d / _n) / (d / n)
        derTT = np.sum([l.derTT for l in Lt.N_])  # top link_ is all comp results
        rv_t += np.abs((_derTT / _n) / (derTT / n))
        if Lt.lH:  # ddfork only, not recursive?
            # intra-level recursion in dfork
            rVd, rv_td = ffeedback(Lt)
            rv_t = rv_t + rv_td

    return rM+rD+rVd, rv_t

def project_focus(PV__, y,x, Fg):  # radial accum of projected focus value in PV__

    m,d,n = Fg.Et
    V = (m-ave*n) + (d-avd*n)
    dy,dx = Fg.angle; a = dy/ max(dx,1e-7)  # average link_ orientation, projection
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

def agg_frame(floc, image, iY, iX, rV=1, rv_t=[], fproj=0):  # search foci within image, additionally nested if floc

    if floc:   # focal img, converted to dert__
        dert__ = image
    else:      # global img
        dert__ = comp_pixel(image)
        global ave, Lw, intw, loopw, centw, contw, adist, amed, medw
        ave, Lw, intw, loopw, centw, contw, adist, amed, medw = np.array([ave, Lw, intw, loopw, centw, contw, adist, amed, medw]) / rV
        # fb rws: ~rvs
    nY,nX = dert__.shape[-2] // iY, dert__.shape[-1] // iX  # n complete blocks
    Y, X  = nY * iY, nX * iX  # sub-frame dims
    frame = CN(box=np.array([0,0,Y,X]), yx=np.array([Y//2,X//2]))
    dert__= dert__[:,:Y,:X]  # drop partial rows/cols
    win__ = dert__.reshape(dert__.shape[0], iY,iX, nY,nX).swapaxes(1,2)  # dert=5, wY=64, wX=64, nY=13, nX=20
    PV__  = win__[3].sum( axis=(0,1)) * intw  # init proj vals = sum G in dert[3],       shape: nY=13, nX=20
    aw = contw * 20
    while np.max(PV__) > ave * aw:  # max G * int_w + pV
        # max win index:
        y,x = np.unravel_index(PV__.argmax(), PV__.shape)
        PV__[y,x] = -np.inf  # to skip, | separate in__?
        if floc:
            Fg = frame_blobs_root( win__[:,:,:,y,x], rV)  # [dert, iY, iX, nY, nX]
            Fg = vect_root(Fg, rV, rv_t)  # focal dert__ clustering
            Fg = cross_comp(Fg, rc=frame.olp) or Fg
        else:
            Fg = agg_frame(1, win__[:,:,:,y,x], wY,wX, rV=1, rv_t=[])  # use global wY,wX in nested call
            if Fg and Fg.L_:  # only after cross_comp(PP_)
                rV, rv_t = ffeedback(Fg)  # adjust filters
        if Fg and Fg.L_:
            if fproj and val_(Fg.Et, (len(Fg.N_)-1)*Lw, Fg.olp+loopw*20):
                pFg = project_N_(Fg, np.array([y,x]))
                if pFg:
                    pFg = cross_comp(pFg, rc=Fg.olp)  # skip compared_ in FG cross_comp
                    if pFg and val_(pFg.Et, (len(pFg.N_)-1)*Lw, pFg.olp+contw*20):
                        project_focus(PV__, y,x, Fg)  # += proj val in PV__
            # no target proj
            add_N(frame, Fg, fmerge=1)
            aw = contw *20 * frame.Et[2] * frame.olp

    if frame.N_ and val_(frame.Et, (len(frame.N_)-1)*Lw, frame.olp+loopw*20) > 0:

        F = cross_comp(frame, rc=frame.olp+loopw)  # recursive xcomp Fg.N_s
        if F and not floc:
            return F  # foci are not preserved

if __name__ == "__main__":  # './images/toucan_small.jpg' './images/raccoon_eye.jpeg', add larger global image

    iY,iX = imread('./images/toucan_small.jpg').shape
    frame = agg_frame(0, image=imread('./images/toucan.jpg'), iY=iY, iX=iX)
    # search frames ( foci inside image