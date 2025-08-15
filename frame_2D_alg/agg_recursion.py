import numpy as np, weakref
from copy import copy, deepcopy
from math import cos   # from functools import reduce
from itertools import zip_longest, combinations, chain, product;  from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, imread, unpack_blob_, comp_pixel, CBase
from slice_edge import slice_edge, comp_angle
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

Feedback of projected match adjusts filters to maximize next match, including coordinate filters that select new inputs.
(initially ffeedback, can be refined by cross_comp of co-projected patterns: "imagination, planning, action" section of part 3)

Forward should generate new code by cross-comp of function calls and clustering code blocks of past and simulated processes
or by comparing out-syntax to in-syntax, the difference should change out-process (not yet implemented)?

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
        l.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m_,d_ [M,D,n,o, I,G,A,L], sum across fork tree, centroid_M?
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
        M = sum(m_ * wTTf[0]); D = sum(abs(d_) * wTTf[1])
        Et = np.array([M, D, 8])  # n compared params = 8
        if root: root.Et += Et
        return CLay(Et=Et, olp=(_lay.olp+lay.olp*rn)/2, node_=node_, link_=link_, derTT=derTT)

class CN(CBase):
    name = "node"
    def __init__(n, **kwargs):
        super().__init__()
        n.fi = kwargs.get('fi', 1)  # if G else 0, fd_: list of forks forming G?
        n.N_ = kwargs.get('N_',[])  # nodes, or ders in links
        n.L_ = kwargs.get('L_',[])  # links, fi only?
        n.nH = kwargs.get('nH',[])  # top-down hierarchy of sub-node_s: CN(sum_N_(Nt_))/ lev, with single added-layer derH, empty nH
        n.lH = kwargs.get('lH',[])  # bottom-up hierarchy of L_ graphs: CN(sum_N_(Lt_))/ lev, within each nH lev
        n.Et = kwargs.get('Et',np.zeros(3))  # sum from L_, cent_?
        n.et = kwargs.get('et',np.zeros(3))  # sum from rim, altg_?
        n.olp = kwargs.get('olp',1)  # overlap to ext Gs, ave in links? separate olp for rim, or internally overlapping?
        n.rim = kwargs.get('rim',[])  # node-external links, rng-nested? set?
        n.derH  = kwargs.get('derH',[])  # sum from L_ or rims
        n.derTT = kwargs.get('derTT',np.zeros((2,8)))  # sum derH -> m_,d_ [M,D,n,o,I,G,A,L]
        n.baseT = kwargs.get('baseT',np.zeros(4))  # I,G,A not ders
        n.yx    = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.rng   = kwargs.get('rng',1)  # or med: loop count in comp_node_|link_
        n.box   = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.span  = kwargs.get('span',0) # distance in nodet or aRad, comp with baseT and len(N_) but not additive?
        n.angl  = kwargs.get('angl',np.zeros(2))  # dy,dx, sum from L_
        n.mang  = kwargs.get('mang',1) # ave match of angles in L_, = identity in links
        n.root  = kwargs.get('root', [])  # immediate only
        n.cent_ = kwargs.get('cent_',[])  # int centroid Gs, replace/combine N_?
        n.altg_ = kwargs.get('altg_',[])  # ext contour Gs, replace/combine rim?
        n.fin = kwargs.get('fin',0)  # clustered, temporary
        n.exe = kwargs.get('exe',1)  # exemplar, temporary
        n.seen_ = set()
        # n.fork_tree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def __bool__(n): return bool(n.N_)

def Copy_(N, root=None, init=0):

    C = CN(root=root)
    if init:  # init G|C with N
        C.N_ = [N]; C.nH, C.lH, N.root = [],[],C
        if init==1: N.L_ += N.rim  # empty in centroid
    else:
        C.N_,C.L_,C.nH,C.lH, N.root = (list(N.N_),list(N.L_),list(N.nH),list(N.lH), root if root else N.root)
    C.derH  = [lay.copy_() for lay in N.derH]
    C.derTT = deepcopy(N.derTT)
    for attr in ['Et', 'baseT','yx','box','angl','rim','altg_','cent_']: setattr(C, attr, copy(getattr(N, attr)))
    for attr in ['olp','rng', 'fi', 'fin', 'span', 'mang']: setattr(C, attr, getattr(N, attr))
    return C

ave, avd, arn, aI, aveB, aveR, Lw, intw, loopw, centw, contw = 10, 10, 1.2, 100, 100, 3, 5, 2, 5, 10, 15  # value filters + weights
adist, amed, distw, medw = 10, 3, 2, 2  # cost filters + weights, add alen?
wM, wD, wN, wO, wI, wG, wA, wL = 10, 10, 20, 20, 1, 1, 20, 20  # der params higher-scope weights = reversed relative estimated ave?
wTTf = np.ones((2,8))  # fb weights per derTT, adjust in agg+
wY = wX = 64; wYX = np.hypot(wY,wX)  # focus dimensions
'''
initial PP_ cross_comp and connectivity clustering to initialize focal frame graph, no recursion:
'''
def vect_root(Fg, rV=1, wTTf=[]):  # init for agg+:
    if np.any(wTTf):
        global ave, avd, arn, aveB, aveR, Lw, adist, amed, intw, loopw, centw, contw, wM, wD, wN, wO, wI, wG, wA, wL
        ave, avd, arn, aveB, aveR, Lw, adist, amed, intw, loopw, centw, contw = (
            np.array([ave,avd,arn,aveB,aveR, Lw, adist, amed, intw, loopw, centw, contw]) / rV)  # projected value change
        wTTf = np.multiply([[wM,wD,wN,wO,wI,wG,wA,wL]], wTTf)  # or dw_ ~= w_/ 2?
        wTTf = np.delete(wTTf,(2,3), axis=1)  #-> comp_slice, = np.array([(*wTTf[0][:2],*wTTf[0][4:]),(*wTTf[0][:2],*wTTf[1][4:])])
    blob_ = unpack_blob_(Fg); N_ = []
    for blob in blob_:
        if not blob.sign and blob.G > aveB:
            edge = slice_edge(blob, rV)
            if edge.G * ((len(edge.P_) - 1) * Lw) > ave * sum([P.latuple[4] for P in edge.P_]):
                N_ += comp_slice(edge, rV, wTTf)
    Fg.N_ = [PP2N(PP, Fg) for PP in N_]

def val_(Et, fi=1, mw=1, aw=1, _Et=np.zeros(3)):  # m,d eval per cluster or cross_comp

    if mw <= 0: return 0
    am = ave * aw  # includes olp, M /= max I | M+D? div comp / mag disparity vs. span norm
    ad = avd * aw  # dval is borrowed from co-projected or higher-scope mval
    m,d,n = Et
    if fi==2: val = np.array([m-am, d-ad]) * mw
    else:     val = (m-am if fi else d-ad) * mw  # m: m/ (m+d), d: d/ (m+d)?
    if _Et[2]:
        # borrow rational deviation of contour if fi else root Et: multiple? not circular
        _m,_d,_n = _Et; _mw = mw*(_n/n)
        if fi==2: val *= np.array([_d/ad, _m/am]) * _mw
        else:     val *= (_d/ad if fi else _m/am) * _mw
    return val
''' 
Core process per agg level, as described in top docstring:
Cross-compare nodes, evaluate incremental-derivation cross-comp of new >ave difference links, recursively.
 
Select sparse node exemplars, connectivity-cluster by >ave match links, correlation-cluster links by >ave difference.
Form complemented clusters (core+contour) for recursive higher-composition cross_comp, reorder by eigenvalues. 

Global centroid clustering from exemplar seeds, sellectively spreading via their rim across connectivity clusters.
Feedback coords to bottom level or prior-level in parallel pipelines, filter updates in more coarse cycles '''

def cross_comp(root, rc, fi=1):  # rng+ and der+ cross-comp and clustering

    N_,L_,Et = comp_(root.N_, rc, fi)  # rc: redundancy+olp, lG.N_ is Ls
    if len(L_) > 1:
        mV,dV = val_(Et,2, (len(L_)-1)*Lw, rc+loopw); lG = []
        if dV > 0:
            if root.L_: root.lH += [sum_N_(root.L_)]  # replace L_ with agg+ L_:
            root.L_ =L_; root.Et += Et
            if dV > avd:
                lG = cross_comp(CN(N_=L_), rc+contw, fi=0)  # link clustering, +2 der layers
                if lG: root.lH += [lG]+lG.nH; root.Et+=lG.Et; root.derH+=lG.derH  # new lays
        if mV > 0:
            nG = Cluster(root, N_, rc+loopw, fi)  # get_exemplars, rng-band, local cluster_C_
            if nG:
                if val_(np.sum([g.Et for g in nG.cent_]),1,(len(nG.N_)-1)*Lw, rc+centw+nG.rng, Et) > 0:
                    cluster_C_(nG, rc)  # cluster nG.cent_ exemplars, of any rng
                if lG: comb_altg_(nG,lG, rc+2)  # both fork alt Ns are clustered
                if val_(nG.Et,1, (len(nG.N_)-1)*Lw, rc+loopw+nG.rng, Et) > 0:
                    nG = cross_comp(nG, rc+loopw) or nG  # agg+
                _H = root.nH; root.nH = []
                nG.nH = _H + [root] + nG.nH  # pack root in Nt.nH, has own L_,lH
                return nG  # recursive feedback

def comb_altg_(nG, lG, rc):  # cross_comp contour/background per node:

    for Lg in lG.N_:
        altg_ = {n.root for L in Lg.N_ for n in L.N_ if n.root and n.root.root}  # rdn core Gs, exclude frame
        if altg_:
            altg_ = {(core,rdn) for rdn,core in enumerate(sorted(altg_, key=lambda x:(x.Et[0]/x.Et[2]), reverse=True), start=1)}
            Lg.altg_ = [altg_, np.sum([i.Et for i,_ in altg_], axis=0)]
    def R(L):
        if L.root: return L.root if L.root.root is lG else R(L.root)
        else:      return None
    for Ng in nG.N_:
        Et, Rdn, altl_ = np.zeros(3), 0, []  # contour/core clustering
        LR_ = {R(L) for n in Ng.N_ for L in n.rim}  # lGs, individual rims are too weak
        for LR in LR_:
            if LR and LR.altg_:  # not None, eval Lg.altg_[1]?
                for core, rdn in LR.altg_[0]:  # map contour rdns to core N:
                    if core is Ng:
                        altl_ += [LR]; Et += core.Et; Rdn += rdn  # add to Et[2]?
        if altl_:
            aG = CN(N_=altl_, Et=Et)
            if val_(Et,0,(len(altl_)-1)*Lw, rc+Rdn+loopw) > 0:  # norm by core_ rdn
                aG = cross_comp(aG, rc) or aG
            Ng.altg_ = (aG.N_,aG.Et)

# replace with proj_L_:
def proj_span(N, dir_vec):  # project N.span in link direction, in proportion to internal collinearity N.mang

    cos_proj = np.dot(N.angl, dir_vec) / (np.linalg.norm(N.angl) * np.linalg.norm(dir_vec))  # cos angle = (a⋅b) / (||a||⋅||b||)
    oriented = (1 - N.mang) + (N.mang * 2 * abs(cos_proj))  # from 1/span to 2?
    #          (1 - N.mang) * N.span + N.mang * (1 + (2 * N.span - 1) * abs(cos_angle)), * elongation?
    return N.span * oriented  # from 1 to N.span * len L_?

def comp_(iN_, rc, fi=1, fC=0, fdeep=0):  # comp pairs of nodes or links within max_dist

    N__,L_, ET = [],[], np.zeros(3); rng,olp_,_N_ = 1,[],copy(iN_)
    # frng: range-band?
    while True:  # _vM
        Np_ = []  # [G pair + co-positionals], for top-nested Ns, unless cross-nesting comp:
        for _N, N in combinations(_N_, r=2):
            if _N in N.seen_ or len(_N.nH) != len(N.nH):  # | root.nH: comp top nodes only
                continue
            dy,dx = np.subtract(_N.yx,N.yx); dist = np.hypot(dy,dx); dy_dx = np.array([dy,dx])
            radii = proj_span(_N, dy_dx) + proj_span(N, dy_dx)  # directionally projected
            # replace radii, vett with pL_tV, computed in main sequence:
            Np_ += [(_N,N, dy,dx, radii, dist)]
        N_,Et = [],np.zeros(3)
        for Np in Np_:
            _N,N, dy,dx, radii, dist = Np
            (_m,_d,_n), (m,d,n) = _N.Et,N.Et; olp = (_N.olp+N.olp)/2  # add directional projection?
            if fC: _m += np.sum([i[0] for i in _N.mo_]); m += np.sum([i[0] for i in N.mo_])  # matches to centroids
            # vett = _N.et[1-fi]/_N.et[2] + N.et[1-fi]/N.et[2]  # density term
            pL_tV = proj_L_(_N,N) if fdeep else 0
            # directional N.L_, N.rim val combination of variable depth, += pL_t of mediated nodes?
            mA,dA = comp_angle(_N.angl, N.angl)
            conf = (_N.mang + N.mang) / 2; mA *= conf; dA *= conf  # N collinearity | angle confidence
            if fi: V = ((_m + m + mA) * intw + pL_tV) / (ave*(_n+n))
            else:  V = ((_d + d + mA) * intw + pL_tV) / (avd*(_n+n))
            max_dist = adist * (radii / aveR) * V
            # min induction distance:
            if max_dist > dist or set(_N.rim) & set(N.rim):  # close or share matching mediators
                Link = comp_N(_N,N, rc, L_=L_, angl=dy_dx, span=dist, dang=[mA,dA], et=np.array([mA,dA,1]), fdeep = dist < max_dist/2, rng=rng)
                if val_(Link.Et, aw=loopw*olp) > 0:
                    Et += Link.Et; olp_ += [olp]  # link.olp is the same with o
                    for n in _N,N:
                        if n not in N_ and val_(n.et, aw=rc+rng-1+loopw+olp) > 0:  # cost+ / rng?
                            N_ += [n]  #-> rng+ eval
        if N_:
            N__ += [N_]; ET += Et
            if val_(Et, mw=(len(N_)-1)*Lw, aw=loopw * (sum(olp_) if olp_ else 1)) > 0:  # current-rng vM
                _N_ = N_; rng += 1; olp_ = []  # reset
            else: break  # low projected rng+ vM
        else: break
    return N__, L_, ET

def comp_N(_N,N, rc, med=1, L_=None, angl=np.zeros(2), span=None, dang=np.zeros(2), et=np.zeros(3), fdeep=0, rng=1):  # compare links, no angle,span?

    derTT, Et, rn = min_comp(_N,N, rc); fi = N.fi  # -1 if _rev else 1, d = -d if L is reversed relative to _L, obsoleted by sign in angle?
    Et += et  # from comp_angle
    baseT = np.array([*(_N.baseT[:2]+ N.baseT[:2]*rn /2), *dang])
    _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)]); o = (_N.olp+N.olp) / 2

    Link = CN(Et=Et,olp=o, et=_N.et+N.et, N_=[_N,N], baseT=baseT,derTT=derTT, yx=np.add(_N.yx,N.yx)/2,box=box, span=span,angl=angl, rng=rng,med=med, fi=0)
    Link.derH = [CLay(Et=copy(Et),node_=[_N,N],link_=[Link],derTT=copy(derTT), root=Link)]
    if fdeep:
        if val_(Et,1, len(N.derH)-2, o) > 0 or fi==0:  # else derH is dext,vert
            Link.derH += comp_H(_N.derH,N.derH, rn, Et, derTT, Link)  # append
        if fi:
            _N_,N_ = (_N.cent_,N.cent_) if (_N.cent_ and N.cent_) else (_N.N_,N.N_)  # or both? no rim,altg_ in top nodes
            if isinstance(N_[0],CN) and isinstance(_N_[0],CN):  # not PP
                spec(_N_,N_,rc,Et, Link.L_)  # use L_ for dspe?
    if fdeep==2: return Link  # or Et?
    if L_ is not None:
        L_ += [Link]
    for n, _n, rev in zip((N,_N),(_N,N),(0,1)):  # if rim-mediated comp: reverse dir in _N.rim: rev^_rev
        n.rim += [Link]; n.et += Et; n.seen_.add(_n)  # or extT += Link?
    return Link

def min_comp(_N,N, rc):  # from comp Et, Box, baseT, derTT
    """
    pairwise similarity kernel:
    m_ = element‑wise min(shared quantity) / max(total quantity) of eight attributes (sign‑aware)
    d_ = element‑wise signed difference after size‑normalisation
    DerTT = np.vstack([m_,d_])
    Et[0] = Σ(m_ * wTTf[0])    # total match (≥0) = relative shared quantity
    Et[1] = Σ(|d_| * wTTf[1])  # total absolute difference (≥0)
    Et[2] = min(_n, n)        # min accumulation span
    """
    _M,_D,_n = _N.Et; M,D,n = N.Et
    dn = _n - n; mn = min(_n,n) / max(_n,n)  # or multiplicative for ratios: min * rn?
    rn = _n / n  # size ratio, add _o/o?
    o, _o = N.olp, _N.olp
    o*=rn; do = _o - o; mo = min(_o,o) / max(_o,o)
    M*=rn; dM = _M - M; mM = min(_M,M) / max(_M,M)
    D*=rn; dD = _D - D; mD = min(_D,D) / max(_D,D)
    # comp baseT:
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
    m_,d_ = np.array([[mM,mD,mn,mo,mI,mG,mA,mL], [dM,dD,dn,do,dI,dG,dA,dL]])
    # comp derTT:
    id_ = N.derTT[1] * rn; _id_ = _N.derTT[1]  # norm by span
    if N.fi:
        dd_ = _id_-id_  # dangle insignificant for nodes
    else:  # comp angle in link Ns
        _dy,_dx = _N.angl; dy, dx = N.angl
        dot = dy * _dy + dx * _dx  # orientation
        leP = np.hypot(dy,dx) * np.hypot(_dy,_dx)
        cos_da = dot / leP if leP else 1  # keep in [–1,1]
        rot = 1 if dy * _dx - dx * _dy >= 0 else -1  # +1 CW, −1 CCW
        # projected abs diffs * combined input and dangle sign:
        dd_ = np.abs(_id_-id_) * ((1-cos_da)/2) * np.sign(_id_) * np.sign(id_) * rot
    _a_,a_ = np.abs(_id_),np.abs(id_)
    md_ = np.divide( np.minimum(_a_,a_), np.maximum.reduce([_a_,a_, np.zeros(8) + 1e-7]))  # rms
    md_ *= np.where((_id_<0)!=(id_<0), -1,1)  # match is negative if comparands have opposite sign
    m_ += md_; d_ += dd_
    DerTT = np.array([m_,d_])  # [M,D,n,o, I,G,A,L]
    Et = np.array([np.sum(m_* wTTf[0] * rc),
                   np.sum(np.abs(d_ * wTTf[1] * rc)), # frame ffeedback * correlation
                   min(_n, n)])  # shared scope?
    return DerTT, Et, rn
    
def cos_comp(C, N, fdeep=0):  # eval cosine similarity for centroid clustering

    mw_ = wTTf[0] * np.sqrt(C.wTT[0]); mW = mw_.sum()  # wTT is derTT correlation term from cent_attr, same for base attrs?
    _M,_D,_n = C.Et; M,D,n = N.Et  # for comp only
    rn = _n/n  # size norm, +_o/o?
    o,_o = N.olp, C.olp
    _I,_G,_Dy,_Dx = C.baseT; I,G,Dy,Dx = N.baseT  # I, G|D, angle
    _L,L = (len(C.N_), len(N.N_)) if N.fi else (C.span, N.span)  # dimension, no angl? positive?
    bw_ = mw_[[0,1,2,3,4,5,7]]; bW = bw_.sum()  # exclude ang w
    BaseT = np.array([_M,_D,_n,_o,_I,_G,_L]); wD_ = BaseT * bw_
    baseT = np.array([ M, D, n, o, I, G, L]); wd_ = baseT * bw_ * rn
    denom = np.linalg.norm(wD_) * np.linalg.norm(wd_) + 1e-12
    Mb = 0.5 * ((wD_ @ wd_) / denom + 1.0)  # [-1,1]→[0,1]
    # separate:
    mA,_ = comp_angle((_Dy,_Dx),(Dy*rn, Dx*rn))
    Mb = (Mb*bW + mA*mw_[6]) / (bW+mw_[6])  # combine for weighted ave cos
    # comp derT:
    dw_ = wTTf[1] * np.sqrt(C.wTT[1]); dW = dw_.sum()
    wD_ = C.derTT[1] * dw_; wd_ = N.derTT[1] * dw_ * rn
    denom = np.linalg.norm(wD_) * np.linalg.norm(wd_) + 1e-12
    Md = 0.5 * ((wD_ @ wd_) / denom + 1.0)  # [-1,1]→[0,1]
    tW = mW + dW
    MT = (Mb*mW + Md*dW) / (tW + 1e-12)  # weighted baseT + derT
    if fdeep:
        if MT * tW * (len(N.derH)-2) > ave * ((o+_o)/2):
            MH = comp_H_cos(C.derH, N.derH, rn, C.wTT)
            ddW = tW * (dW / mW)  # ders W < inputs W
            MT = (MT*tW + MH*ddW) / (tW+ddW +1e-12)
        # add N.rim overlap?
    return MT

def spec(_spe,spe, rc, Et, dspe=None, fdeep=0):  # for N_|cent_ | altg_
    for _N in _spe:
        for N in spe:
            if _N is not N:
                dN = comp_N(_N, N, rc, fdeep=2); Et += dN.Et  # may be d=cent_?
                if dspe is not None: dspe += [dN]
                if fdeep:
                    for L,l in [(L,l) for L in _N.rim for l in N.rim]:  # l nested in L
                        if L is l: Et += l.Et  # overlap val
                    if _N.altg_ and N.altg_: spec(_N.altg_,N.altg_, rc, Et)

def rolp(N, _N_, R=0): # rel V of L_|N.rim overlap with _N_: inhibition|shared zone, oN_ = list(set(N.N_) & set(_N.N_)), no comp?

    n_ = N.N_ if R else {n for l in N.rim for n in l.N_ if n is not N}  # nrim
    olp_ = n_ & _N_
    if olp_:
        oEt = np.sum([i.Et for i in olp_], axis=0)
        _Et = N.Et if R else N.et  # not sure
        rV = (oEt[0]/oEt[2]) / (_Et[0]/_Et[2])
        return rV * val_(N.et,1, aw=centw)  # contw for cluster?
    else:
        return 0

def get_exemplars(N_, rc):  # get sparse nodes by multi-layer non-maximum suppression

    _E_ = set()
    # stronger-first
    for rdn, N in enumerate(sorted(N_, key=lambda n: n.et[0]/n.et[2], reverse=True), start=1):
        # ave *= relV of overlap by stronger-E inhibition zones
        roV = rolp(N, _E_)
        if val_(N.et,1, aw = rc + rdn + loopw + roV) > 0:  # cost
            _E_.update({n for l in N.rim for n in l.N_ if n is not N and val_(n.Et,1,aw=rc) > 0})  # selective nrim
            N.exe = 1  # in point cloud of focal nodes
        else:
            break  # the rest of N_ is weaker, trace via rims

def Cluster(root, N_, rc, fi):  # clustering root

    if isinstance(N_[0],CN): Nf_ = N_; N_ = [N_]; Et = root.Et  # nest N_ as rng-banded
    else:                    Nf_ = list(set([N for n_ in N_ for N in n_])); Et = None
    get_exemplars(Nf_,rc)  # set n.exe, cluster rN_ via rng exemplars
    nG = []
    for rng, rN_ in enumerate(N_, start=1):  # bottom-up rng-banded clustering
        aw = rc * rng +contw
        Et = Et if Et is not None else np.sum([n.Et for n in rN_], axis=0)
        if rN_ and val_(Et,1, (len(rN_)-1)*Lw, aw) > 0:
            nG = cluster(root, Nf_, rN_, aw, rng) or nG
    # top valid nG:
    return nG

def cluster(root, iN_, rN_, rc, rng=1):  # flood-fill node | link clusters

    def rroot(n):
        if n.root and n.root.rng > n.rng: return rroot(n.root) if n.root.root else n.root
        else:                             return None
    G_, cent_ = [], []  # exclusive per fork,ave, only centroids can be fuzzy
    for n in iN_: n.fin = 0
    for N in rN_:  # init
        if not N.exe or N.fin: continue  # exemplars or all N_
        N.fin = 1; seen_,N_,link_,long_ = [],[],[],[]
        if rng==1 or (not N.root or N.root.rng==1):  # not rng-banded, L.root is empty
            for l in N.rim:
                seen_ += [l]
                if l.rng==rng and val_(l.Et+ett(l), aw=rc+1) > 0: link_+=[l]; N_ += [N]  # l.Et potentiated by density term
                elif l.rng>rng: long_+=[l]
        else: # N is rng-banded, cluster top-rng roots
            n = N; R = rroot(n)
            if R and not R.fin: N_,link_,long_ = [R], R.L_, R.hL_; R.fin = 1; seen_+=R.link_
        Seen_ = set(link_)  # all visited
        L_ = []
        while link_:  # extend clustered N_ and L_
            L_+=link_; _link_=link_; link_=[]; seen_=[]
            for L in _link_:  # buffer
                for _N in L.N_:
                    if _N not in iN_ or _N.fin: continue  # or always in iN_ now?
                    if rng==1 or (not _N.root or _N.root.rng==1):  # not rng-banded
                        N_ += [_N]; _N.fin = 1  # conditional
                        for l in N.rim:
                            if l in seen_: continue
                            seen_+=[l]
                            if l.rng==rng and val_(l.Et+ett(l), aw=rc) > 0: link_+=[l]
                            elif l.rng>rng: long_ += [l]
                    else:  # cluster top-rng roots
                        _n = _N; _R = rroot(_n)
                        if _R and not _R.fin:
                            seen_+=_R.link_
                            if rolp(N, link_, R=1) > ave*rc:
                                N_+=[_R]; _R.fin=1; _N.fin=1; link_+=_R.link_; long_+=_R.hL_
            link_ = list(set(link_)-Seen_)
            Seen_.update(set(seen_))
        if N_:
            N_, long_ = list(set(N_)), list(set(long_))
            Et, olp = np.zeros(3),0  # sum node_:
            for n in N_:
                Et += n.et; olp += n.olp  # any fork
            if val_(Et,1, (len(N_)-1)*Lw, rc+olp, root.Et) > 0:
                G = sum2graph(root, N_,L_, long_,Et, olp, rng)
                G_ += [G]   # dval:
                if val_(Et, fi=0, mw=G.span*2* slope(L_), aw=olp+centw) > 0:
                    cent_ += [n for n in N_ if n.exe]  # exemplars for cluster_C_ per frame?
    if G_:
        nG = sum_N_(G_, root)
        nG.cent_ = cent_
        return nG

def cluster_C_(root, rc, fdeep=0):  # form centroids by clustering exemplar surround, drifting via rims of new member nodes

    _C_, _N_ = [], []
    for E in root.cent_:
        C = cent_attr( Copy_(E,root, init=2), rc); C.N_ = [E]   # all rims are within root
        C._N_ = [n for l in E.rim for n in l.N_ if n is not E]  # core members + surround -> comp_C
        _N_ += C._N_; _C_ += [C]
    # reset:
    for n in set(root.root.N_+_N_ ): n.C_,n.mo_, n._C_,n._mo_ = [],[], [],[]  # aligned pairs, in cross_comp root
    # reform C_, refine C.N_s
    while True:
        C_, cnt,mat,olp, Dm,Do = [],0,0,0,0,0; Ave = ave * rc * loopw
        _Ct_ = [[c, c.Et[0]/c.Et[2] if c.Et[0] !=0 else 1e-7, c.olp] for c in _C_]
        for _C,_m,_o in sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True):
            if _m > Ave*_o:
                C = cent_attr( sum_N_(_C.N_, root=root, fC=1), rc)  # C update lags behind N_; non-local C.olp += N.mo_ os?
                _N_,_N__, mo_, M,O, dm,do = [],[],[],0,0,0,0  # per C
                for n in _C._N_:  # core+ surround
                    if C in n.C_: continue
                    m = cos_comp(C,n)  # val,olp / C:
                    o = np.sum([mo[0] /m for mo in n._mo_ if mo[0]>m])  # overlap = higher-C inclusion vals / current C val
                    cnt += 1  # count comps per loop
                    if m > Ave * o:
                        _N_+=[n]; M+=m; O+=o; mo_ += [np.array([m,o])]  # n.o for convergence eval
                        _N__ += [_n for l in n.rim for _n in l.N_ if _n is not n]  # +|-ls for comp C
                        if _C not in n._C_: dm += m; do += o  # not in extended _N__
                    elif _C in n._C_:
                        __m,__o = n._mo_[n._C_.index(_C)]; dm +=__m; do +=__o
                if M > Ave * len(_N_) * O:
                    for n, mo in zip(_N_,mo_): n.mo_ += [mo]; n.C_ += [C]
                    C.M = M; C.N_+= _N_; C._N_= list(set(_N__))  # core, surround elements
                    C_+=[C]; mat+=M; olp+=O; Dm+=dm; Do+=do   # new incl or excl
                else:
                    for n in _C._N_:
                        for i, c in enumerate(n.C_):
                            if c is _C: n.mo_.pop(i); n.C_.pop(i); break  # remove mo mapping to culled _C
            else: break  # the rest is weaker
        if Dm > Ave * cnt * Do:  # dval vs. dolp, overlap increases as Cs may expand in each loop?
            _C_ = C_
            for n in root.root.N_: n._C_ = n.C_; n._mo_= n.mo_; n.C_,n.mo_ = [],[]  # new n.C_s, combine with vo_ in Ct_?
        else:  # converged
            break
    if  mat > Ave * cnt * olp:
        root.cent_ = (set(C_),mat)
        # cross_comp, low overlap eval in comp_node_?

def cent_attr(C, rc):  # weight attr matches | diffs by their match to the sum, recompute to convergence

    wTT = []  # Cs can be fuzzy only to the extent that their correlation weights are different?

    for derT, wT in zip(C.derTT, wTTf):
        # weigh values by feedback before cross-correlation
        _w_ = np.ones(8)
        val_ = np.abs(derT) * wT  # m|d are signed, but their contribution is absolute
        V = np.sum(val_)
        while True:
            mean = max(V / max(np.sum(_w_), 1e-7), 1e-7)
            inverse_dev_ = np.minimum(val_/mean, mean/val_)  # rational deviation from mean rm in range 0:1, if m=mean
            w_ = inverse_dev_ / .5  # 2/ m=mean, 0/ inf max/min, 1 / mid_rng | ave_dev?
            w_ *= 8 / np.sum(w_)  # mean w = 1, M shouldn't change?
            if np.sum(np.abs(w_-_w_)) > ave*rc:
                V = np.sum(val_ * w_)
                _w_ = w_
            else:  break  # weight convergence
        wTT += [_w_]
    C.wTT = np.array(wTT)  # replace wTTf
    return C

def ett(L): return (L.N_[0].et + L.N_[1].et) * intw

def sum2graph(root, node_,link_,long_, Et, olp, rng):  # sum node,link attrs in graph, aggH in agg+ or player in sub+

    n0 = Copy_(node_[0]); derH = n0.derH; fi = n0.fi
    graph = CN(root=root, fi=1,rng=rng, N_=node_,L_=link_,olp=olp, Et=Et, box=n0.box, baseT=n0.baseT, derTT=n0.derTT)
    graph.hL_ = long_
    n0.root = graph; yx_ = [n0.yx]; fg = fi and isinstance(n0.N_[0],CN)   # not PPs
    Nt = Copy_(n0); DerH = []  #->CN, comb forks: add_N(Nt,Nt.Lt)?
    for N in node_[1:]:
        add_H(derH,N.derH,graph); graph.baseT+=N.baseT; graph.derTT+=N.derTT; graph.box=extend_box(graph.box,N.box); yx_+=[N.yx]; N.root = graph
        if fg: add_N(Nt,N)
    for L in link_:
        add_H(DerH,L.derH,graph); graph.baseT+=L.baseT; graph.derTT+=L.derTT
    if DerH: add_H(derH,DerH, graph)  # * rn?
    graph.derH = derH
    if fg: graph.nH = Nt.nH + [Nt]  # pack prior top level
    yx = np.mean(yx_, axis=0); dy_,dx_ = (yx_-yx).T; dist_ = np.hypot(dy_,dx_)
    graph.span = dist_.mean()  # node centers distance to graph center
    graph.angl = np.sum([l.angl for l in link_], axis=0)
    if fi and len(link_) > 1:  # else default mang = 1
        graph.mang = np.sum([comp_angle(graph.angl, l.angl)[0] for l in link_]) / len(link_)
    graph.yx=yx
    return graph

def slope(link_):  # get ave 2nd rate of change with distance in cluster or frame?

    Link_ = sorted(link_, key=lambda x: x.span)
    dists = np.array([l.span for l in Link_])
    diffs = np.array([l.Et[1]/l.Et[2] for l in Link_])
    rates = diffs / dists
    # ave d(d_rate) / d(unit_distance):
    return (np.diff(rates) / np.diff(dists)).mean()

def comp_H(H,h, rn, ET=None, DerTT=None, root=None):  # one-fork derH

    derH, derTT, Et = [], np.zeros((2,8)), np.zeros(3)
    for _lay, lay in zip_longest(H,h):  # selective
        if _lay and lay:
            dlay = _lay.comp_lay(lay, rn, root=root)
            derH += [dlay]; derTT = dlay.derTT; Et += dlay.Et
    if Et[2]: DerTT += derTT; ET += Et
    return derH

def comp_H_cos(H, h, rn, wTT):
    M = 0
    dw_ = wTTf[1] * np.sqrt(wTT[1])
    for Lay, lay in zip_longest(H, h, fillvalue=None):
        if Lay and lay:
            wD_ = Lay.derTT[1] * dw_
            wd_ = lay.derTT[1] * dw_ * rn
            denom = np.linalg.norm(wD_) * np.linalg.norm(wd_) + 1e-12
            M += 0.5 * ((wD_ @ wd_) / denom + 1.0)
    return M

def sum_H_(Q):  # sum derH in link_|node_, not used
    H = Q[0]; [add_H(H,h) for h in Q[1:]]
    return H

def add_H(H, h, root=0, rn=1, rev=0):  #  layer-wise add|append derH

    for Lay, lay in zip_longest(H,h):  # different len if lay-selective comp
        if lay:
            if Lay: Lay.add_lay(lay, rn)
            else:   H += [lay.copy_(rev)]  # * rn?
            if root: root.derTT += lay.derTT*rn; root.Et += lay.Et*rn
    return H

def add_sett(Sett,sett):
    if Sett: N_,Et = Sett; n_ = sett[0]; N_.update(n_); Et += np.sum([t.Et for t in n_-N_])
    else:    Sett += [copy(par) for par in sett]  # altg_, Et

def sum_N_(node_, root=None, fC=0):  # form cluster G

    G = Copy_(node_[0], root, init = 0 if fC else 1)
    if fC:
        G.L_=[]; G.N_= [node_[0]]
    for n in node_[1:]:
        add_N(G,n,0, fC)
    G.olp /= len(node_)
    if not fC:
        for L in G.L_: G.Et += L.Et
    return G   # no rim

def add_N(N,n, fmerge=0, fC=0):

    if fmerge:  # different in altg_?
        for node in n.N_: node.root=N; N.N_ += [node]
        N.L_ += n.L_  # no L.root assign
    else:
        n.root=N; N.N_ += [n]
        if hasattr(N,"wTT"): N.L_ += n.L_  # for extend C, wTT is recomputed
        else:                N.L_ += [l for l in n.rim if val_(l.Et)>0]
    if n.altg_: add_sett(N.altg_,n.altg_)  # ext clusters
    if n.cent_: add_sett(N.cent_,n.cent_)  # int clusters
    if n.nH: add_NH(N.nH,n.nH, root=N)
    if n.lH: add_NH(N.lH,n.lH, root=N)
    en,_en = n.Et[2],N.Et[2]; rn = en/_en
    for Par,par in zip((N.angl,N.mang,N.baseT,N.derTT,N.Et), (n.angl,n.mang,n.baseT,n.derTT,n.Et)):
        Par += par  # *rn for comp
    if n.derH: add_H(N.derH,n.derH, N, rn)
    if fC: N.olp += np.sum([mo[1] for mo in n._mo_])
    else:  N.olp = (N.olp*_en + n.olp*en) / (en+_en)  # ave of aves
    N.yx = (N.yx*_en + n.yx*en) /(en+_en)
    N.span = max(N.span,n.span)
    N.box = extend_box(N.box, n.box)
    if hasattr(n,'C_') and hasattr(N,'C_'):
        N.C_ += n.C_; N.mo_ += n.mo_  # not sure
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

    return CN(root=frame, fi=1, Et=Et+et, N_=P_, L_=link_, baseT=baseT, derTT=derTT, derH=derH, box=box, yx=yx, angl=A, span=np.hypot(dy/2,dx/2))

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

def comp_prj_dH(_N,N, ddH, rn, link, angl, span, dec):  # comp combined int proj to actual ddH, as in project_N_

    _cos_da= angl.dot(_N.angl) / (span *_N.span)  # .dot for scalar cos_da
    cos_da = angl.dot(N.angl) / (span * N.span)
    _rdist = span/_N.span
    rdist  = span/ N.span
    prj_DH = add_H( prj_dH(_N.derH[1:], _cos_da *_rdist, _rdist*dec),  # derH[0] is positionals
                    prj_dH( N.derH[1:], cos_da * rdist, rdist*dec),
                    link)  # comb proj dHs | comp dH ) comb ddHs?
    # Et+= confirm:
    dddH = comp_H(prj_DH, ddH[1:], rn, link.Et, link.derTT, link)  # ddH[1:] maps to N.derH[1:]
    add_H(ddH, dddH, link, rn)

def project_N_(Fg, yx):

    dy,dx = Fg.yx - yx
    Fdist = np.hypot(dy,dx)   # external dist
    rdist = Fdist / Fg.span
    Angle = np.array([dy,dx]) # external angle
    angle = np.sum([L.angl for L in Fg.L_])
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

    wTTf = np.ones((2,8))  # sum derTT coefs: m_,d_ [M,D,n,o, I,G,A,L] / Et, baseT, dimension
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
        wTTf += np.abs((_derTT / _n) / (derTT / n))
        if Lt.lH:  # ddfork only, not recursive?
            # intra-level recursion in dfork
            rVd, wTTfd = ffeedback(Lt)
            wTTf = wTTf + wTTfd

    return rM+rD+rVd, wTTf

def project_focus(PV__, y,x, Fg):  # radial accum of projected focus value in PV__

    m,d,n = Fg.Et
    V = (m-ave*n) + (d-avd*n)
    dy,dx = Fg.angl; a = dy/ max(dx,1e-7)  # average link_ orientation, projection
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

def agg_frame(foc, image, iY, iX, rV=1, wTTf=[], fproj=0):  # search foci within image, additionally nested if floc

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
            vect_root(Fg, rV,wTTf); Fg.L_=[] # focal dert__ clustering
            cross_comp(Fg, rc=frame.olp)
        else:
            Fg = agg_frame(1, win__[:,:,:,y,x], wY,wX, rV=1, wTTf=[])  # use global wY,wX in nested call
            if Fg and Fg.L_:  # only after cross_comp(PP_)
                rV, wTTf = ffeedback(Fg)  # adjust filters
                Fg = cent_attr(Fg,2)  # compute Fg.wTT: correlation weights in frame derTT?
                wTTf *= Fg.wTT; wTTf[0] *= 8/np.sum(wTTf[0]); wTTf[1] *= 8/np.sum(wTTf[1])
                # re-norm weights
        if Fg and Fg.L_:
            if fproj and val_(Fg.Et,1, (len(Fg.N_)-1)*Lw, Fg.olp+loopw*20):
                pFg = project_N_(Fg, np.array([y,x]))
                if pFg:
                    pFg = cross_comp(pFg, rc=Fg.olp)  # skip compared_ in FG cross_comp
                    if pFg and val_(pFg.Et,1, (len(pFg.N_)-1)*Lw, pFg.olp+contw*20):
                        project_focus(PV__, y,x, Fg)  # += proj val in PV__
            # no target proj
            frame = add_N(frame, Fg, fmerge=1) if frame else Copy_(Fg)
            aw = contw *20 * frame.Et[2] * frame.olp

    if frame.N_ and val_(frame.Et, (len(frame.N_)-1)*Lw, frame.olp+loopw*20) > 0:

        F = cross_comp(frame, rc=frame.olp+loopw)  # recursive xcomp Fg.N_s
        if F and not foc:
            return F  # foci are not preserved

if __name__ == "__main__":  # './images/toucan_small.jpg' './images/raccoon_eye.jpeg', add larger global image

    iY,iX = imread('./images/toucan_small.jpg').shape
    frame = agg_frame(0, image=imread('./images/toucan.jpg'), iY=iY, iX=iX)
    # search frames ( foci inside image