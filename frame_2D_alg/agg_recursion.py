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
        m_ = np.minimum(_a_,a_) / np.maximum.reduce([_a_,a_,np.zeros(8)+ 1e-7])  # match = min/max comparands
        m_ *= np.where((_i_<0) != (i_<0), -1,1)  # match is negative if comparands have opposite sign
        derTT = np.array([m_,d_])
        node_ = list(set(_lay.node_+ lay.node_))  # concat, or redundant to nodet?
        link_ = _lay.link_ + lay.link_
        M = sum(m_ * w_t[0]); D = sum(abs(d_) * w_t[1])
        Et = np.array([M, D, 8])  # n compared params = 8
        if root: root.Et += Et
        return CLay(Et=Et, olp=(_lay.olp+lay.olp*rn)/2, node_=node_, link_=link_, derTT=derTT)

class CN(CBase):
    name = "node"
    def __init__(n, **kwargs):
        super().__init__()
        n.fi = kwargs.get('fi', 1)  # if G else 0, fd_: list of forks forming G?
        n.N_ = kwargs.get('N_',[])  # nodes, or ders in links
        n.L_ = kwargs.get('L_',[])  # links
        n.nH = kwargs.get('nH',[])  # top-down hierarchy of sub-node_s: CN(sum_N_(Nt_))/ lev, with single added-layer derH, empty nH
        n.lH = kwargs.get('lH',[])  # bottom-up hierarchy of L_ graphs: CN(sum_N_(Lt_))/ lev, within each nH lev
        n.Et = kwargs.get('Et',np.zeros(3))  # sum from L_, cent_?
        n.et = kwargs.get('et',np.zeros(3))  # sum from rim, outn_?
        n.olp = kwargs.get('olp',1)  # overlap to ext Gs, ave in links? separate olp for rim, or internally overlapping?
        n.rim = kwargs.get('rim',[])  # [(L,rev,N)] in nodes, mediated by nodet tree in links?
        n.derH  = kwargs.get('derH',[])  # sum from L_ or rims
        n.derTT = kwargs.get('derTT',np.zeros((2,8)))  # sum derH
        n.baseT = kwargs.get('baseT',np.zeros(4))
        n.yx    = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.rng   = kwargs.get('rng',1)  # or med: loop count in comp_node_|link_
        n.box   = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.span  = kwargs.get('span',0) # distance in nodet or aRad, comp with baseT and len(N_) but not additive?
        n.angle = kwargs.get('angle',np.zeros(2))  # dy,dx
        n.fin   = kwargs.get('fin',0)  # in cluster, temporary?
        n.root  = kwargs.get('root',[])  # immediate only
        n.cent_ = kwargs.get('cent_', [])  # int centroid Gs, replace/combine N_?
        n.outn_ = kwargs.get('outn_', [])  # ext contour Gs, replace/combine rim?
        # n.fork_tree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def __bool__(n): return bool(n.N_)

def Copy_(N, root=None, init=0):

    C = CN(root=root)
    if init:  # init G|C with N
        if init != 2: C.N_ = [N]  # init G or fi centroid
        C.nH, C.lH, N.root = [],[],C
        if N.rim:
            if init>1: N.N_ += rim_(N.rim,fi=1)  # init centroid
            else:      N.L_ += rim_(N.rim,fi=0)  # init G
    else:
        C.N_,C.L_,C.nH,C.lH, N.root = (list(N.N_),list(N.L_),list(N.nH),list(N.lH), root if root else N.root)
    C.derH  = [lay.copy_() for lay in N.derH]
    C.derTT = deepcopy(N.derTT)
    for attr in ['Et', 'baseT','yx','box','angle','rim','outn_','cent_']: setattr(C, attr, copy(getattr(N, attr)))
    for attr in ['olp','rng', 'fi', 'fin', 'span']: setattr(C, attr, getattr(N, attr))
    return C

def rim_(rim, fi=None):  # unpack terminal rt_s in L.rim
    Rim = []
    for r in rim:  # n or rt: (l,rev,_n), root-nested?
        if r not in Rim:
            if isinstance(r,CN): Rim.extend(rim_(r.rim[-1], fi))  # r is nodet[i], get last rim:
            else:                Rim += [r if fi is None else r[2] if fi else r[0]]  # (LL,rev,_L)
    return Rim

ave, avd, arn, aI, aveB, aveR, Lw, intw, loopw, centw, contw = 10, 10, 1.2, 100, 100, 3, 5, 2, 5, 10, 15  # value filters + weights
adist, amed, distw, medw = 10, 3, 2, 2  # cost filters + weights, add alen for ?
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
                    for L,_,_ in eN.rim:  # all +ve, *= density?
                        if L not in link_:
                            for eN in L.N_:
                                if eN in PP_: eN_ += [eN]; PP_.remove(eN)  # merged
                            link_ += [L]; et += L.Et
                _eN_ = {*eN_}
            if val_(et, mw=(len(node_)-1)*Lw, aw=2+contw) > 0:  # rc=2
                Et += et
                G_ += [sum2graph(frame, node_,link_,[],et, olp=1,rng=1, outn_=[],cent_=[])]  # single-lay link_derH
        return G_, Et

    def comp_PP_(PP_):
        N_,L_,mEt,dEt = [],[],np.zeros(3),np.zeros(3)
        for _G, G in combinations(PP_, r=2):
            dy,dx = np.subtract(_G.yx,G.yx); dist = np.hypot(dy,dx)
            if dist - (G.span+_G.span) < adist / 10:  # very short here
                L = comp_N(_G,G, 2, angle=np.array([dy,dx]), span=dist, fdeep=1)
                m, d, n = L.Et
                if m > ave * n * _G.olp * loopw: mEt += L.Et; N_ += [_G,G]  # mL_ += [L]
                if d > avd * n * _G.olp * loopw: dEt += L.Et  # dL_ += [L]
                L_ += [L]
        dEt[2] = dEt[2] or 1e-7
        return list(set(N_)),L_,mEt,dEt  # can't be empty

    PP_ = edge.node_
    if val_(edge.Et, (len(PP_)-edge.rng)*Lw, loopw) > 0:
        PP_,L_,mEt,dEt = comp_PP_([PP2N(PP,frame) for PP in PP_])
        if not L_: return
        if val_(mEt,1, (len(PP_)-1)*Lw, contw) > 0:
            G_, Et = cluster_PP_(copy(PP_))  # can't be empty
        else: G_ = []
        if G_: frame.N_ += G_; lev.N_ += PP_; lev.Et += Et
        else:  frame.N_ += PP_  # PPm_
        lev.L_+= L_; lev.Et = mEt+dEt  # links between PPms
        for l in L_: derlay.add_lay(l.derH[0]); frame.baseT+=l.baseT; frame.derTT+=l.derTT; frame.Et += l.Et
        if val_(dEt,0, (len(L_)-1)*Lw, 2+contw) > 0:
            Lt = Cluster(frame, PP_, rc=2,fi=0)
            if Lt:  # L_ graphs
                if lev.lH: lev.lH[0].N_ += Lt.N_; lev.lH[0].Et += Lt.Et
                else:      lev.lH += [Lt]
                lev.Et += Lt.Et

def val_(Et, fi=1, mw=1, aw=1, _Et=np.zeros(3)):  # m,d eval per cluster or cross_comp

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
Feedback coords to bottom level or prior-level in parallel pipelines, filter updates in more coarse cycles '''

def cross_comp(root, rc, fi=1):  # rng+ and der+ cross-comp and clustering

    N_,L_,Et = comp_node_(root.N_,rc) if fi else comp_link_(root.N_,rc)  # rc: redundancy+olp
    if len(L_) > 1:
        for n in [n for N in N_ for n in N] if fi else N_:
            for l in rim_(n.rim,fi=0): n.et+=l.Et  # !olp
        mV,dV = val_(Et,2, (len(L_)-1)*Lw, rc+loopw)
        if dV > 0:
            if root.L_: root.lH += [sum_N_(root.L_)]  # replace L_ with agg+ L_:
            root.L_=L_; root.Et += Et
            if dV >avd: Lt = cross_comp(CN(N_=L_), rc+contw, fi=0)  # -> Lt.nH, +2 der layers
            else:       Lt = Cluster(root, N_, rc+contw, fi=0)  # cluster N.rim Ls, +1 layer
            if Lt: root.lH += [Lt] + Lt.nH; root.Et += Lt.Et; root.derH += Lt.derH  # new der lays
        if mV > 0:
            Nt = Cluster(root, N_, rc+loopw, fi)   # get_exemplars, cluster_C_, rng-banded if fi
            if Nt and val_(Nt.Et,1, (len(Nt.N_)-1)*Lw, rc+loopw+Nt.rng, Et) > 0:
                Nt = cross_comp(Nt, rc+loopw) or Nt  # agg+
            if Nt:
                _H = root.nH; root.nH = []
                Nt.nH = _H + [root] + Nt.nH  # pack root in Nt.nH, has own L_,lH
                # recursive feedback:
                return Nt

def comp_node_(iN_, rc):  # rng+ forms layer of rim and extH per N?

    N__,L_,ET = [],[],np.zeros(3); rng,olp_,_N_ = 1,[],copy(iN_)  # range banded if frng only?
    for n in iN_: n.compared_ = set()
    while True:   # _vM
        Gp_ = []  # [G pair + co-positionals], for top-nested Ns, unless cross-nesting comp:
        for _G, G in combinations(_N_, r=2):
            if _G in G.compared_ or len(_G.nH) != len(G.nH):  # | root.nH: comp top nodes only
                continue
            radii = _G.span+G.span; dy,dx = np.subtract(_G.yx,G.yx); dist = np.hypot(dy,dx)
            Gp_ += [(_G,G, dy,dx, radii, dist)]
        N_,Et = [],np.zeros(3)
        for Gp in Gp_:
            _G, G, dy,dx, radii, dist = Gp
            (_m,_,_n),(m,_,n) = _G.Et,G.Et; olp = (_G.olp+G.olp)/2; _m = _m * intw +_G.et[0]; m = m * intw + G.et[0]
            # density / comp_N: et|M, 0/rng=1?
            max_dist = adist * (radii/aveR) * ((_m+m)/(ave*(_n+n+0))/ intw)  # ave_dist * radii * induction
            if max_dist > dist or set(_G.rim) & set(G.rim):
                # comp if close or share matching mediators: add to inhibited?
                Link = comp_N(_G,G, rc, L_=L_, angle=np.array([dy,dx]), span=dist, fdeep = dist < max_dist/2, rng=rng)
                if val_(Link.Et, aw=loopw*olp) > 0:
                    Et += Link.Et; olp_ += [olp]  # link.olp is the same with o
                    for n in _G,G:
                        if n not in N_ and val_(n.et, aw=rc+rng-1+loopw+olp) > 0:  # cost+ / rng?
                            N_ += [n]  # for rng+ and exemplar eval
        if N_:
            N__ += [N_]; ET += Et
            if val_(Et, mw=(len(N_)-1)*Lw, aw=loopw * sum(olp_)/max(1,len(olp_))) > 0:  # current-rng vM
                _N_ = N_; rng += 1; olp_ = []  # reset
            else: break  # low projected rng+ vM
        else: break
    return N__, L_, ET

def comp_link_(iL_, rc):  # comp links via directional node-mediated _link tracing with incr mediation

    for L in iL_:  # init conversion, rim = [(_n,n)] in links, [[(l,rev,_n)]] in nodes, nested by mediation:
        L.rim, L._rim, L.med, L.compared_ = [L.rim], [], 1, set()  # _rim append in comp
    med = 1
    L__,LL_,ET,_L_ = [],[], np.zeros(3), iL_
    while _L_ and med < 4:
        L_, Et = [], np.zeros(3)
        for L in _L_:
            for _L,_rev,_ in rim_(L.rim):  # top-med n.rim Ls, add _rev ^= lower-med revs?
                if _L not in L.compared_ and _L in iL_ and val_(_L.Et,0, aw=rc+med) > 0:  # high-d, new links can't be outside iL_
                    dy, dx = np.subtract(_L.yx, L.yx)
                    Link = comp_N(_L,L, rc, med, LL_, np.array([dy,dx]), np.hypot(dy,dx), -1 if _rev else 1)  # d = -d if L is reversed relative to _L
                    if val_(Link.Et,1, aw=rc+med) > 0:  # recycle Ls if high m
                        L_ += [L,_L]; Et += Link.Et
        for L in _L_:
            if L._rim: L.rim += [L._rim]; L._rim = []  # merge new rim per med loop
        if L_:
            L_ = list(set(L_)); L__ += L_; _L_ = []
            for l in L_:
                if l not in _L_ and len(l.rim) > med and val_(l.Et,0, aw=rc+med) > 0:
                    _L_ += [l]  # high-d, rim was extended in current loop
        med += 1
    return list(set(L__)), LL_, ET

def base_comp(_N,N, rc, dir=1):  # comp Et, Box, baseT, derTT
    """
    pairwise similarity kernel:
    m_ = element‑wise min(shared quantity) / max(total quantity) of eight attributes (sign‑aware)
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
    if N.fi: # dimension is n nodes
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
    m_ = np.divide( np.minimum(_a_,a_), np.maximum.reduce([_a_,a_, np.zeros(8) + 1e-7]))  # rms
    m_ *= np.where((_i_<0) != (i_<0), -1,1)  # match is negative if comparands have opposite sign
    m_ += _m_; d_ += _d_
    DerTT = np.array([m_,d_])  # [M,D,n,o, I,G,A,L], weigh by centroid_M?
    Et = np.array([np.sum(m_* w_t[0] * rc), np.sum(np.abs(d_* w_t[1] * rc)), min(_n,n)])
    # feedback*rc -weighted sum of m,d between comparands
    return DerTT, Et, rn

def comp_N(_N,N, rc, med=1, L_=None, angle=None, span=None, dir=1, fdeep=0, rng=1):  # compare links, relative N direction = 1|-1, no angle,span?

    derTT, Et, rn = base_comp(_N,N, rc, dir); fi = N.fi
    # link M,D,A:
    baseT = np.array([*(_N.baseT[:2] + N.baseT[:2]*rn) / 2, *angle])  # redundant angle for generic base_comp, also span-> density?
    _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
    o = (_N.olp+N.olp) / 2
    Link = CN(rng=rng, Et=Et, olp=o, rim=[_N,N], baseT=baseT, derTT=derTT, yx=np.add(_N.yx,N.yx)/2, span=span, angle=angle, box=box, med=med, fi=0)
    Link.derH = [CLay( root=Link, Et=Et, node_=[_N,N],link_=[Link], derTT=copy(derTT))]
    if fdeep:
        if val_(Et,1, len(N.derH)-2, o) > 0 or fi==0:  # else derH is dext,vert
            Link.derH += comp_H(_N.derH,N.derH, rn, Et, derTT, Link)  # append
        if fi:
            _N_,N_ = (_N.cent_,N.cent_) if (_N.cent_ and N.cent_) else (_N.N_,N.N_)  # or cent_ replaces N_? no rims, outl in top nodes
            spec(_N_,N_, rc,Et, Link.N_)
    if L_ is not None:
        L_ += [Link]
    for n, _n, rev in zip((N,_N),(_N,N),(0,1)):  # reverse Link dir in _N.rim:
        rim = n.rim if fi else n._rim
        rim += [(Link, rev,_n)]  # rev-nested?
        n.compared_.update(_n)
    return Link

def spec(_spe,spe, rc, Et, dspe=None, fdeep=0):  # for N_|cent_ | outn_
    for _N in _spe:
        for N in spe:
            if _N is not N:
                dN = comp_N(_N, N, rc); Et += dN.Et
                if dspe is not None: dspe += [dN]
                if fdeep:
                    for L,l in [(L,l) for L in rim_(_N.rim,0) for l in rim_(N.rim,0)]:
                        if L is l: Et += l.Et  # overlap val
                    if _N.outn_ and N.outn_: spec(_N.outn_,N.outn_, rc, Et)

def rolp(N, _N_, fi, E=0, R=0): # rV of N.rim |L_ overlap with _N_: inhibition|shared zone, oN_ = list(set(N.N_) & set(_N.N_)), no comp?

    if R: olp_ = [n for n in (N.N_ if fi else N.L_) if n in _N_]  # currently fi=1
    else: olp_ = [r for r in rim_(N.rim,fi) if any(n in _N_ for n in rim_(r.rim,fi))]
    if olp_:
        oEt = np.sum([i.Et for i in olp_], axis=0)
        _Et = N.Et if (R or E) else N.et  # not sure
        rV = (oEt[1-fi]/oEt[2]) / (_Et[1-fi]/_Et[2])
        return rV * val_(N.et, fi, aw=centw)  # contw for cluster?
    else:
        return 0

def get_exemplars(N_, rc, fi):  # get sparse nodes by multi-layer non-maximum suppression

    E_, Et = [], np.zeros(3)  # ~ point cloud of focal nodes
    _E_ = set()  # prior = stronger:
    for rdn, N in enumerate(sorted(N_, key=lambda n: n.et[1-fi]/ n.et[2], reverse=True), start=1):
        # ave *= relV of overlap by stronger-E inhibition zones
        roV = rolp(N, list(_E_), fi, E=1)
        if val_(N.et, fi, aw = rc + rdn + loopw + roV) > 0:  # cost
            _E_.update([r for r in rim_(N.rim,fi) if val_(r.Et,fi,aw=rc) > 0])  # selective nrim|lrim
            Et += N.et; N.sel = 1  # in cluster
            E_ += [N]  # exemplars
        else:
            break  # the rest of N_ is weaker, trace via rims
    return E_, Et

def Cluster(root, N_, rc, fi):  # clustering root

    Nflat_ = list(set([N for n_ in N_ for N in n_])) if fi else N_
    E_, Et = get_exemplars(Nflat_, rc, fi)

    if val_(Et,fi, (len(E_)-1)*Lw, rc+contw, root.Et) > 0:
        for n in Nflat_: n.sel=0
        if fi:
            Nt = []  # bottom-up rng-banded clustering:
            for rng, rN_ in enumerate(N_, start=1):
                rE_ = [n for n in rN_ if n.sel]
                aw = rc * rng + contw  # cluster Nf_ via rng exemplars:
                if rE_ and val_(np.sum([n.Et for n in rE_], axis=0),1,(len(rE_)-1)*Lw, aw) > 0:
                    Nt = cluster(root, Nflat_, rE_, aw, 1, rng) or Nt  # keep top-rng Gt
        else:
            Nt = cluster(root, N_, E_, rc, fi=0)
        return Nt

def cluster(root, N_, E_, rc, fi, rng=1):  # flood-fill node | link clusters

    G_ = []  # exclusive per fork,ave, only centroids can be fuzzy
    fln = not E_[0].rim
    for n in (root.N_ if fi else N_): n.fin = 0
    for E in E_:
        if E.fin: continue  # init:
        if fi: # node clustering
            if rng==1 or E.root.rng==1:  # E is not rng-nested
                node_,link_,llink_ = [E],[],[]
                for l in rim_(E.rim,0):  # lrim
                    if l.rng==rng and val_(l.Et,aw=rc) > 0: link_ += [l]  # +ve links only?
                    elif l.rng>rng: llink_ += [l]
            else: # E is rng-nested, cluster top-rng roots
                n = E; R = n.root
                while R.root and R.root.rng > n.rng: n = R; R = R.root
                if R.fin: continue
                node_,link_,llink_ = [R], R.L_, R.hL_
                R.fin = 1
        else: # link clustering
            node_,link_,llink_ = [],[],[]  # node_ is rim only
        E.fin = 1
        if fi:  # node cluster
            for L in link_[:]:  # snapshot
                for _N in L.N_:
                    if _N not in N_ or _N.fin: continue
                    if rng==1 or _N.root.rng==1:  # not rng-nested
                        if rolp(E, rim_(E.rim,0), fi=1) > ave*rc:
                            node_ += [_N]; _N.fin = 1
                            for l in rim_(_N.rim,0):
                                if l not in link_ and l.rng == rng and val_(l.Et,aw=rc) > 0: link_ += [l]
                                elif l not in llink_ and l.rng > rng: llink_ += [l]
                    else:  # cluster top-rng roots
                        _n = _N; _R = _n.root
                        while _R.root and _R.root.rng > _n.rng: _n = _R; _R = _R.root
                        if not _R.fin and rolp(E, link_, fi=1, R=1) > ave*rc:
                            node_ += [_R]; _R.fin = 1; _N.fin = 1
                            link_ = list(set(link_ + _R.link_))
                            llink_= list(set(llink_+ _R.hL_))
        else:  # link cluster
            if fln:  # cluster by L diff, rim is empty
                for L in link_[:]:
                    for n in L.rim:  # nodet, no eval
                        if n in N_ and not n.fin and rolp(E, rim_(n.rim,0), fi) > ave*rc:
                            link_ += [n]; n.fin = 1  # no link eval?
                            node_ = [l for l in rim_(n.rim,0) if l not in node_ and  val_(l.Et,0,aw=rc) > 0]
            else:  # cluster by rim match
                for _L in rim_(E.rim,0):
                    if _L.fin: continue
                    _L.fin = 1
                    for n in _L.rim:  # nodet
                        for LL in rim_(n.rim,0):
                            if LL not in link_ and val_(_L.Et,0,aw=rc) > 0:
                                node_ += [_L]; link_ += [LL]
        node_ = list(set(node_))
        Et, olp = np.zeros(3),0  # sum node_:
        for n in node_:
            Et += n.et; olp += n.olp  # not fork-specific
        if fi:
            outn_ = [L.root for L in link_ if L.root]  # outline is link clusters, individual rims are too weak for contour
            if outn_:
                O = CN(N_=list(set(outn_)), Et = np.sum([n.Et for n in outn_]))  # replace et?
                if val_(O.Et,0, (len(O.N_)-1)*Lw, olp,Et) > 0:
                    outn_ = cross_comp(O,olp) or outn_
                _Et = outn_.Et
            else: _Et = np.zeros(3)
        else:
            _Et = root.Et  # root is a lender
            outn_ = []
        mV,dV = val_(Et,2, (len(node_)-1)*Lw, rc+olp, _Et)
        if mV > 0:
            # replace with val_(np.sum([l.Et for l in link_ if val_(l.Et,0) >0])  # nEt?
            if dV > ave * centw:  # divisive clustering eval by variance, seed CC_ = E_, may extend beyond node_?
                Ct = cluster_C_(root, E_, rc+centw, fi=fi)
                if Ct:
                    cent_,et = Ct; Et += et  # add cross_comp if low search olp, replace node_ with cent_?
                else: cent_ = []
            else: cent_ = []
            G_ += [sum2graph(root, node_, link_, llink_, Et, olp, rng, cent_,outn_)]
    if G_:
        return sum_N_(G_, root)

def cluster_C_(root, E_, rc, fi=1, fdeep=0):  # form centroids by clustering exemplar surround, drifting via rims of new member nodes

    def comp_C(C, n):
        _,et,_ = base_comp(C,n, rc)
        if fdeep:
            if val_(Et,1,len(n.derH)-2,rc):
                comp_H(C.derH, n.derH, n.Et[2]/C.Et[2], Et)
            for L,l in [(L,l) for L in rim_(C.rim,0) for l in rim_(n.rim,0)]:
                if L is l: et += l.Et  # overlap
        return et

    _C_ = []; av = ave if fi else avd; _N_ = []
    for E in E_:
        C = Copy_(E,root, init=fi+2)  # init centroid
        C._N_ = list({n for N in rim_(E.rim,fi) for n in rim_(N.rim,fi)})  # core members + surround for comp to N_ mean
        _N_ += C._N_; _C_ += [C]
    # reset per root:
    for n in set(root.N_+_N_): n.C_, n.vo_, n._C_, n._vo_ = [], [], [], []
    # recompute C, refine / extend C.N_:
    while True:
        C_, ET, O, Dvo, Ave = [], np.zeros(3), 0, np.zeros(2), av*rc*loopw
        _Ct_ = [[c, c.Et[1-fi]/c.Et[2], c.olp] for c in _C_]
        for _C,_v,_o in sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True):
            if _v > Ave*_o:
                C = sum_N_(_C.N_, root=root, fC=1)  # merge rim,alt before cluster_NC_
                _N_,_N__,o,Et,dvo = [],[],0, np.zeros(3),np.zeros(2)  # per C
                for n in _C._N_:  # core + surround
                    if C in n.C_: continue  # clear/ loop?
                    et = comp_C(C,n); v = val_(et,fi,aw=rc)
                    olp = np.sum([vo[0] / v for vo in n.vo_ if vo[0] > v])  # olp: rel n val in stronger Cs, not summed with C
                    vo = np.array([v,olp])  # val, overlap per current root C
                    if et[1-fi]/et[2] > Ave*olp:
                        n.C_ += [C]; n.vo_ += [vo]; Et += et; o+=olp; _N_ += [n]
                        _N__ += rim_(n.rim,fi)
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
    if val_(ET,fi, aw=O+rc+loopw) > 0:  # C_ value, no _Et?
        return C_, ET

def sum2graph(root, node_,link_,llink_, Et, olp, rng, cent_, outn_, fC=0):  # sum node,link attrs in graph, aggH in agg+ or player in sub+

    n0 = Copy_(node_[0]); derH = n0.derH
    l0 = Copy_(link_[0]); DerH = l0.derH
    graph = CN(root=root, fi=1,rng=rng, N_=node_,L_=link_,olp=olp, Et=Et, outn_=outn_,cent_=cent_,box=n0.box, baseT=n0.baseT+l0.baseT, derTT=n0.derTT+l0.derTT)
    graph.hL_ = llink_
    n0.root = graph; yx_ = [n0.yx]; fg = isinstance(n0.N_[0],CN)  # not PPs
    Nt = Copy_(n0)  #->CN, comb forks: add_N(Nt,Nt.Lt)?
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

def comp_H(H,h, rn, ET=None, DerTT=None, root=None):  # one-fork derH

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

    G = Copy_(node_[0], root, init = 0 if fC else 1)
    if fC: G.L_=[]; G.N_= [node_[0]]
    for n in node_[1:]:
        add_N(G,n,fC)
    G.et += np.sum([n.Et for n in G.outn_])  # cross_comp in cluster
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
        if n.rim: N.L_ += rim_(n.rim,0)  # not nodet?
    if n.outn_: N.outn_ += n.outn_  # extra
    if n.cent_: N.cent_ += n.cent_  # intra
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
            else:   H += [Copy_(lev, root)]

def extend_box(_box, box):
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return min(y0,_y0), min(x0,_x0), max(yn,_yn), max(xn,_xn)

def PP2N(PP, frame):

    P_, link_, vert, latuple, A, S, box, yx, Et = PP
    baseT = np.array(latuple[:4])
    [mM,mD,mI,mG,mA,mL], [dM,dD,dI,dG,dA,dL] = vert  # re-pack in derTT:
    derTT = np.array([[mM,mD,mL,1,mI,mG,mA,mL], [dM,dD,dL,1,dI,dG,dA,dL]])
    derH = [CLay(node_=P_, link_=link_, derTT=deepcopy(derTT))]
    y,x,Y,X = box; dy,dx = Y-y,X-x  # A = (dy,dx); L = np.hypot(dy,dx), rolp = 1
    et = np.array([*np.sum([L.Et for L in link_],axis=0), 1]) if link_ else np.array([.0,.0,1.])  # n=1

    return CN(root=frame, fi=1, Et=Et+et, N_=P_, L_=link_, baseT=baseT, derTT=derTT, derH=derH, box=box, yx=yx, span=np.hypot(dy/2,dx/2))

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
    dddH = comp_H(prj_DH, ddH[1:], rn, link.Et, link.derTT, link)  # ddH[1:] maps to N.derH[1:]
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

def agg_frame(foc, image, iY, iX, rV=1, rv_t=[], fproj=0):  # search foci within image, additionally nested if floc

    if foc: dert__ = image  # focal img was converted to dert__
    else:
        dert__ = comp_pixel(image) # global
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
        if foc:
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
        if F and not foc:
            return F  # foci are not preserved

if __name__ == "__main__":  # './images/toucan_small.jpg' './images/raccoon_eye.jpeg', add larger global image

    iY,iX = imread('./images/toucan_small.jpg').shape
    frame = agg_frame(0, image=imread('./images/toucan.jpg'), iY=iY, iX=iX)
    # search frames ( foci inside image