import numpy as np
from copy import copy, deepcopy
from functools import reduce
from itertools import zip_longest, combinations, product
from multiprocessing import Pool, Manager
from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_
from slice_edge import CP, slice_edge, comp_angle
from comp_slice import CdP, comp_slice
'''
Current code is starting with primary sensory data, just images here
Each agg+ cycle refines input nodes in cluster_C_ and connects then in complemented graphs in cluster_N_ 
That connectivity clustering phase has two forks:

rng+ fork: incremental-range cross-comp nodes: edge segments at < max distance, cluster if they match. 
der+ fork: incremental-derivation cross-comp links, from node cross-comp, if abs_diff * rel_adjacent_match 
(variance patterns borrow value from co-projected match patterns because their projections cancel-out)
 
So graphs should be assigned adjacent alt-fork (der+ to rng+) graphs, to which they lend predictive value.
But alt match patterns borrow already borrowed value, which is too tenuous to track, we use average borrowed value.
Clustering criterion within each fork is summed match of >ave vars (<ave vars are not compared and don't add comp costs).

Connectivity clustering is exclusive per fork,ave, with fork selected per variable | derLay | aggLay 
Fuzzy clustering can only be centroid-based: overlapping connectivity-based clusters will merge.
Param clustering if MM, compared along derivation sequence, or combinatorial?

Graph representation is nested in a dual tree of down-forking elements: node_, and up-forking clusters: root_.
That resembles neurons: dendritic input tree and axonal output tree. 
But graphs have recursively nested param sets per branching level, and very different comparison process generating these params.

Ultimate criterion is lateral match, with projecting sub-criteria to add distant | aggregate lateral match
If a property is found to be independently predictive, its match is defined as min comparands: their shared quantity.
Else match is an inverted deviation of miss: instability of that property. 

After computing projected match in forward pass, the feedback adjusts filters to maximize next match. 
That includes coordinate filters, which select new input within current frame of reference

The process may start from arithmetic: inverse ops in cross-comp and direct ops in clustering, for pairwise and group compression. 
But there is a huge number of possible variations, so it seems a lot easier to design meaningful initial code manually.

Meta-code will generate/compress base code by process cross-comp (tracing function calls), and clustering by evaluated code blocks.
Meta-feedback must combine code compression and data compression values: higher-level match is still the ultimate criterion.

Code-coordinate filters may extend base code by cross-projecting and combining patterns found in the original base code
(which may include extending eval function with new match-projecting derivatives) 
Similar to cross-projection by data-coordinate filters, described in "imagination, planning, action" section of part 3 in Readme.
-
diagrams: 
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/agg_recursion_unfolded.drawio.png
-
notation:
prefix  f denotes flag
postfix t denotes tuple, multiple ts is a nested tuple
prefix  _ denotes prior of two same-name variables, multiple _s for relative precedence
postfix _ denotes array of same-name elements, multiple _s is nested array
capitalized variables are usually summed small-case variables
'''
class CLay(CBase):  # layer of derivation hierarchy, subset of CG
    name = "lay"
    def __init__(l, **kwargs):
        super().__init__()
        l.Et = kwargs.get('Et', np.zeros(4))
        l.root = kwargs.get('root', None)  # higher node or link
        l.node_ = kwargs.get('node_', [])  # concat across fork tree
        l.link_ = kwargs.get('link_', [])
        l.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m_,d_ [M,D,n,o, I,G,A,L], sum across fork tree,
        # add weights for cross-similarity, along with vertical aves, for both m_ and d_?
        # altL = CLay from comp altG
        # i = kwargs.get('i', 0)  # lay index in root.node_, link_, to revise olp
        # i_ = kwargs.get('i_',[])  # priority indices to compare node H by m | link H by d
        # ni = 0  # exemplar in node_, trace in both directions?
    def __bool__(l): return bool(l.node_)

    def copy_(lay, root=None, rev=0, i=None):  # comp direction may be reversed to -1

        if i:  # reuse self
            C = lay; lay = i; C.node_=copy(i.node_); C.link_ = copy(i.link_); C.derTT=np.zeros((2,8)); C.root=root
        else:  # init new C
            C = CLay(root=root, node_=copy(lay.node_), link_=copy(lay.link_))
        C.Et = copy(lay.Et)
        for fd, tt in enumerate(lay.derTT):  # nested array tuples
            C.derTT[fd] += tt * -1 if (rev and fd) else deepcopy(tt)

        if not i: return C

    def add_lay(Lay, lay_, rev=0):  # merge lays, including mlay + dlay

        if not isinstance(lay_,list): lay_ = [lay_]
        for lay in lay_:
            # rev = dir==-1, to sum/subtract numericals in m_,d_:
            for fd, Fork_, fork_ in zip((0,1), Lay.derTT, lay.derTT):
                Fork_ += fork_ * -1 if (rev and fd) else fork_  # m_| d_
            # concat node_,link_:
            Lay.node_ += [n for n in lay.node_ if n not in Lay.node_]
            Lay.link_ += lay.link_
            Lay.Et[:3] += lay.Et[:3]  # fixed o
        return Lay

    def comp_lay(_lay, lay, rn, root, dir=1):  # unpack derH trees down to numericals and compare them

        i_ = lay.derTT[1] * rn * dir; _i_ = _lay.derTT[1]  # i_ is ds, scale and direction- normalized
        d_ = _i_ - i_
        a_ = np.abs(i_); _a_ = np.abs(_i_)
        m_ = np.minimum(_a_,a_) / reduce(np.maximum,[_a_,a_,1e-7])  # match = min/max comparands
        m_ *= np.where((_i_<0) != (i_<0), -1,1)  # match is negative if comparands have opposite sign
        derTT = np.array([m_,d_])
        node_ = list(set(_lay.node_+ lay.node_))  # concat
        link_ = _lay.link_ + lay.link_
        M = sum(m_ * w_t[0]); D = sum(d_ * w_t[1])
        Et = np.array([M, D, 8, (_lay.Et[3]+lay.Et[3])/2])  # n compared params = 8
        if root: root.Et += Et

        return CLay(Et=Et, root=root, node_=node_, link_=link_, derTT=derTT)

class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster
    # graph / node
    def __init__(G,  **kwargs):
        super().__init__()
        G.node_ = kwargs.get('node_',[])  # maybe rng-banded
        G.link_ = kwargs.get('link_',[])  # spliced node link_s
        G.H = kwargs.get('H',[])  # list of lower levels: [nG,lG]: pack node_,link_ in sum2graph; lG.H: packed-level lH:
        G.lH = kwargs.get('lH',[])  # link_ agg+ levels in top node_H level: node_,link_,lH
        G.Et = kwargs.get('Et',np.zeros(4))  # sum all M,D,n,o from link_
        G.et = kwargs.get('et',np.zeros(4))  # sum from rim
        G.baseT = kwargs.get('baseT',np.zeros(4))  # I,G,Dy,Dx  # from slice_edge,
        # baset from rim?
        G.derTT = kwargs.get('derTT',np.zeros((2,8)))  # m,d / Et,baseT: [M,D,n,o, I,G,A,L], summed across derH lay forks
        G.extTT = kwargs.get('extTT',np.zeros((2,8)))  # sum across extH
        G.derH = kwargs.get('derH',[])  # each lay is [m,d]: Clay(Et,node_,link_,derTT), sum|concat links across fork tree
        G.extH = kwargs.get('extH',[])  # sum from rims, single-fork
        G.yx = kwargs.get('yx', np.zeros(2))  # init PP.yx = [(y+Y)/2,(x,X)/2], then ave node yx
        G.box = kwargs.get('box', np.array([np.inf,np.inf,-np.inf,-np.inf]))  # y,x,Y,X area: (Y-y)*(X-x)
        G.maxL = kwargs.get('maxL', 0)  # if dist-nested in cluster_N_
        G.aRad = kwargs.get('aRad', 0)  # average distance between graph center and node center
        # empty alt.alt_: select+?
        G.alt_ = []  # adjacent (contour) gap+overlap alt-fork graphs, converted to CG
        # G.fork_tree: list = z([[]])  # indices in all layers(forks, if no fback merge
        # G.fback_ = []  # node fb buffer, n in fb[-1]
        G.fi = kwargs.get('fi',0)  # or fd_: list of forks forming G, 1 if cluster of Ls | lGs, for feedback only?
        G.rim = kwargs.get('rim',[])  # external links
        G._N_ = kwargs.get('_N_', set())  # linked nodes
        G.rng = kwargs.get('rng',1)  # nested node if >1
        G.root = kwargs.get('root')  # maybe a list of rng layers, add in cluster_N_
    def __bool__(G): return bool(G.node_)  # never empty

def copy_(N, root=None, init=0):
    C = CG(root=root)

    for name, value in N.__dict__.items():
        val = getattr(N, name)
        if name == '_id' or name == "Ct_": continue  # skip id and Ct_
        elif name == "node_" and init: C.node_ = [N]
        elif name == "H" and init: C.H = []
        elif name == 'derH':
            for lay in N.derH:
                C.derH += [[fork.copy_(root=C) for fork in lay]] if isinstance(N, CG) else [lay.copy_(root=C)]  # CL
        elif name == 'extH':
            C.extH = [lay.copy_(root=C) for lay in N.extH]
        elif isinstance(value,list) or isinstance(value,np.ndarray):
            setattr(C, name, copy(val))  # Et,yx,box, node_,link_,rim_, altG, baseT, derTT
        else:
            setattr(C, name, val)  # numeric or object: fi, root, maxL, aRad, nnest, lnest
    return C

class CL(CBase):  # link or edge, a product of comparison between two nodes or links
    name = "link"
    def __init__(l,  **kwargs):
        super().__init__()
        l.nodet = kwargs.get('nodet',[])  # e_ in kernels, else replaces _node,node: not used in kernels
        l.L = kwargs.get('L',0)  # distance between nodes
        l.Et = kwargs.get('Et', np.zeros(4))
        l.fi = kwargs.get('fi',0)  # nodet fi
        l.yx = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet
        l.box = kwargs.get('box', np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        l.baseT = kwargs.get('baseT', np.zeros(4))
        l.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m_,d_ [M,D,n,o, I,G,A,L], sum across derH
        l.derH  = kwargs.get('derH', [])  # list of single-fork CLays
        # add med, rimt, extH in der+
    def __bool__(l): return bool(l.nodet)

ave, avd, arn, aI, aveB, aveR, Lw, int_w, loop_w, clust_w = 10, 10, 1.2, 100, 100, 3, 5, 2, 5, 10  # value filters + weights
ave_dist, ave_med, dist_w, med_w = 10, 3, 2, 2  # cost filters + weights
wM, wD, wN, wO, wI, wG, wA, wL = 10, 10, 20, 20, 1, 1, 20, 20  # der params higher-scope weights = reversed relative estimated ave?
w_t = np.ones((2,8))  # fb weights per derTT, adjust in agg+
'''
initial PP_ cross_comp and LC clustering, no recursion:
'''
def vect_root(frame, rV=1, ww_t=[]):  # init for agg+:
    if np.any(ww_t):
        global ave, avd, arn, aveB, aveR, Lw, ave_dist, int_w, loop_w, clust_w, wM, wD, wN, wO, wI, wG, wA, wL, w_t
        ave, avd, arn, aveB, aveR, Lw, ave_dist, int_w, loop_w, clust_w = (
            np.array([ave,avd,arn,aveB,aveR, Lw, ave_dist, int_w, loop_w, clust_w]) / rV)  # projected value change
        w_t = [[wM,wD,wN,wO,wI,wG,wA,wL]] * ww_t  # or dw_ ~= w_/ 2?
        ww_t = np.delete(ww_t,(2,3), axis=1)  #-> comp_slice, = np.array([(*ww_t[0][:2],*ww_t[0][4:]),(*ww_t[0][:2],*ww_t[1][4:])])
    blob_ = unpack_blob_(frame)
    frame = CG(root = None)
    lev0, lev1, lH = [[],[]], [[],[]], [[],[],[]]  # two forks per level and derlay, two levs in lH
    derlay = [CLay(root=frame), CLay(root=frame)]
    for blob in blob_:
        if not blob.sign and blob.G > aveB * blob.root.olp:
            edge = slice_edge(blob, rV)
            if edge.G*((len(edge.P_)-1)*Lw) > ave * sum([P.latuple[4] for P in edge.P_]):
                Et = comp_slice(edge, rV, ww_t)  # to scale vert
                if np.any(Et) and Et[0] > ave * Et[2] * clust_w:
                    cluster_edge(edge, frame, lev0, lev1, lH, derlay)
                    # may be skipped
    G_t = [sum_N_(lev) if lev else [] for lev in lev0]
    for n_ in [G_t] + lev1 + lH:
        for n in n_:
            if n: frame.baseT+=n.baseT; frame.derTT+=n.derTT; frame.Et += n.Et
    if G_t[0] or G_t[1]: frame.H=[G_t]  # lev0
    frame.node_ = lev1[0]; frame.link_= lev1[1]
    frame.lH = lH  # two levs
    frame.derH = [derlay]
    return frame

def cluster_edge(edge, frame, lev0, lev1, lH, derlay):  # non-recursive comp_PPm, comp_PPd, edge is not a PP cluster, unpack by default

    def cluster_PP_(PP_):
        G_ = []
        while PP_:  # flood fill
            node_,link_, et = [],[], np.zeros(4)
            PP = PP_.pop(); _eN_ = [PP]
            while _eN_:
                eN_ = []
                for eN in _eN_:  # rim-connected ext Ns
                    node_ += [eN]
                    for L,_ in eN.rim:  # all +ve, * density: if L.Et[0]/ave_d * sum([n.extH.m * clust_w / ave for n in L.nodet])?
                        if L not in link_:
                            for eN in L.nodet:
                                if eN in PP_:
                                    eN_ += [eN]; PP_.remove(eN)  # merged
                            link_ += [L]; et += L.Et
                _eN_ = {*eN_}
            if val_(et, mw=(len(node_)-1)*Lw, aw=2*clust_w) > 0:  # rc=2
                Lay = CLay(); [Lay.add_lay(link.derH[0]) for link in link_]  # single-lay derH
                G_ += [sum2graph(frame, [node_,link_,et, Lay], fi=1)]
        return G_

    def comp_PP_(PP_):
        N_,L_,mEt,dEt = [],[],np.zeros(4),np.zeros(4)

        for _G, G in combinations(PP_, r=2):
            dy,dx = np.subtract(_G.yx,G.yx); dist = np.hypot(dy,dx)
            if dist - (G.aRad+_G.aRad) < ave_dist / 10:  # very short here
                L = comp_N(_G, G, ave, fi=1, angle=[dy, dx], dist=dist, fshort=1)
                m, d, n, o = L.Et
                if m > ave * n * o * loop_w: mEt += L.Et; N_ += [_G,G]  # mL_ += [L]
                if d > avd * n * o * loop_w: dEt += L.Et  # dL_ += [L]
                L_ += [L]
        dEt[2] = dEt[3] or 1e-7
        dEt[3] = dEt[2] or 1e-7
        return set(N_),L_,mEt,dEt

    for fi in 1,0:
        PP_ = edge.node_ if fi else edge.link_
        if val_(edge.Et, mw=(len(PP_)- edge.rng) *Lw, aw=loop_w) > 0:
            # PPm_|PPd_:
            PP_,L_,mEt,dEt = comp_PP_([PP2G(PP,frame) for PP in PP_])
            if PP_:
                if val_(mEt, mw=(len(PP_)-1)*Lw, aw=clust_w, fi=1) > 0:
                    G_ = cluster_PP_(copy(PP_))
                else: G_ = []
                if G_:
                    if fi: lev1[0] += G_  # node_
                    else:  lH[0] += G_
                    lev0[1-fi] += PP_  # H[0] = [PPm_,PPd_]
                elif fi: lev1[0] += PP_  # PPm_
                else:    lH[0] += PP_  # PPd_
            for l in L_: derlay[fi].add_lay(l.derH[0])
            if fi:  # mfork der+
                if val_(dEt, mw= (len(L_)-1)*Lw, aw=2*clust_w, fi=0) > 0:
                    Gd = cluster_N_(frame, L2N(L_), ave, rc=2, fi=0, fnodet=1)
                    Gd_ = Gd.node_ if Gd else []
                else: Gd_ = []
                lev1[1] += L_  # default
                if Gd_: lH[2] += Gd_  # new layer
            else:
                lH[1] += L_ # lL_

def val_(Et, _Et=None, mw=1, aw=1, fi=1):  # m+d val per cluster|cross_comp

    m, d, n, o = Et  # m->d lend cancels in Et scope, not higher-scope _Et?
    am, ad = ave * aw, avd * aw
    k = mw / n / o
    m, d = m*k, d*k
    if np.any(_Et):  # higher scope values
        _m,_d,_n,_o = _Et; _k = mw /_n /_o
        m += _m *_k  # local + global
        d += _d *_k
    dval = (d-ad) * (m/am)  # ddev borrow per rational mdev
    mval = (m-am) - dval  # additive if neg dval

    return mval if fi else dval

''' core process: 
 
 Cross-comp nodes, then eval cross-comp of resulting > ave difference links, with recursively higher derivation,
 Cluster nodes by >ave match links, correlation-cluster links by >ave diff:
  
 Select high-match exemplars (common node types) for connectivity clustering (CC)
 Selection is essential because CC is mainly generative: complexity of new links is greater than compression by clustering.
 Evaluate resulting clustered node_ or link_ for recursively higher-composition cross_comp 

 Min distance in CC is more restrictive than in cross-comp due to higher costs and density eval.
 CCs terminate at contour altGs, and next-level cross-comp is between core+contour clusters.'''

def comp_node_(_N_, rc, L=0):  # rng+ forms layer of rim and extH per N?

    for n in _N_: n.compared_ = []
    N__,L_,ET = [],[],np.zeros(4)  # range banded, default rng=1
    rng = 1
    while True:  # _vM
        Gp_ = []  # [G pair + co-positionals], for top-nested Ns, unless cross-nesting comp:
        if L: _N_ = [N for N in _N_ if len(N.derH)==L]
        for _G, G in combinations(_N_, r=2):
            if _G in G.compared_ or len(_G.H) != len(G.H):  # | root.H: comp top nodes only?
                continue
            radii = _G.aRad + G.aRad; dy,dx = np.subtract(_G.yx,G.yx); dist = np.hypot(dy,dx)
            Gp_ += [(_G,G, dy,dx, radii, dist)]
        N_,Et = [],np.zeros(4)
        for Gp in Gp_:
            _G, G, dy,dx, radii, dist = Gp
            (_m,_,_n,_o),(m,_,n,o) = _G.Et, G.Et; _m = _m * int_w +_G.et[0]; m = m * int_w + G.et[0]  # same n, o?
            # density/ comp_N: et or M?
            max_dist = ave_dist * (radii/aveR) * ((_m+m)/(ave*(_n+n+_o+o)) / int_w)  # ave_dist * radii * induction
            if max_dist > dist or _G._N_ & G._N_:
                # comp if close or share matching mediators
                Link = comp_N(_G,G, ave, fi=1, angle=[dy,dx], dist=dist, fshort = dist < max_dist/2)
                L_ += [Link]  # include -ve links
                if Link.Et[0] > ave * Link.Et[2] * loop_w:
                    Et += Link.Et
                    for n,_n in zip((_G,G),(G,_G)):
                        n.compared_ += [_n]
                        if n not in N_ and val_(n.et, aw= rc* rng* loop_w) > 0:  # cost+/ rng
                            N_ += [n]  # for rng+ and exemplar eval
        N__ += [N_]; ET += Et
        if Et[0] > ave * Et[2] * loop_w:  # current-rng vM
            _N_ = N_; rng += 1
        else:  # low projected rng+ vM
            break
    return N__,L_,ET  # range banded N__ and L__

def comp_link_(iL_, rc):  # comp CLs via directional node-mediated link tracing: der+'rng+ in root.link_ rim_t node rims

    fi = isinstance(iL_[0].nodet[0], CG)
    for L in iL_:
        # init mL_t: bilateral mediated Ls per L:
        for rev, N, mL_ in zip((0,1), L.nodet, L.mL_t):
            for _L,_rev in N.rim if fi else N.rimt[0]+N.rimt[1]:
                if _L is not L and _L in iL_:  # nodet-mediated
                    if L.Et[0] > ave * L.Et[2] * loop_w:
                        mL_ += [(_L, rev ^_rev)]  # direction of L relative to _L
    med = 1; _L_ = iL_
    L__,LL_,ET = [],[],np.zeros(4)
    while True:  # xcomp _L_
        L_, Et = [], np.zeros(4)
        for L in _L_:
            for mL_ in L.mL_t:
                for _L, rev in mL_:  # rev is relative to L
                    if _L in L.compared_: continue
                    dy,dx = np.subtract(_L.yx,L.yx)
                    Link = comp_N(_L,L, ave*rc, fi=0, angle=[dy,dx], dist=np.hypot(dy,dx), dir = -1 if rev else 1)  # d = -d if L is reversed relative to _L
                    Link.med = med
                    LL_ += [Link]; Et += Link.Et  # include -ves, link order: nodet < L < rimt, mN.rim || L
                    for l,_l in zip((_L,L),(L,_L)):
                        l.compared_ += [_l]
                        if l not in L_ and val_(l.et, aw= rc* med* loop_w) > 0:  # cost+/ rng
                            L_ += [l]  # for med+ and exemplar eval
        L__ += [L_]; ET += Et
        # extend mL_t per last medL:
        if Et[0] > ave * Et[2] * (rc + loop_w + med*med_w):  # project prior-loop value, med adds fixed cost
            _L_, ext_Et = set(), np.zeros(4)
            for L in L_:
                mL_t, lEt = [set(),set()], np.zeros(4)  # __Ls per L
                for mL_,_mL_ in zip(mL_t, L.mL_t):
                    for _L, rev in _mL_:
                        for _rev, N in zip((0,1), _L.nodet):
                            rim = N.rim if fi else N.rimt
                            if len(rim) == med:  # append in comp loop
                                for __L,__rev in rim if fi else rim[0]+rim[1]:
                                    if __L in L.visited_ or __L not in iL_: continue
                                    L.visited_ += [__L]; __L.visited_ += [L]
                                    if __L.Et[0] > ave * __L.Et[2] * loop_w:
                                        mL_.add((__L, rev ^_rev ^__rev))  # combine reversals: 2 * 2 mLs, but 1st 2 are pre-combined
                                        lEt += __L.Et
                if lEt[0] > ave * lEt[2]:  # L'rng+, vs L'comp above, add coef
                    L.mL_t = mL_t; _L_.add(L); ext_Et += lEt
            # refine eval by extension D:
            if ext_Et[0] > ave * ext_Et[2] * (loop_w + med*med_w): med += 1
            else: break
        else: break
    return L__, LL_, ET

def base_comp(_N, N, dir=1):  # comp Et, Box, baseT, derTT
    # comp Et:
    _M,_D,_n,_o = _N.Et; M,D,n,o = N.Et
    dn = _n - n; mn = min(_n,n) / max(_n,n)  # or multiplicative for ratios: min * rn?
    rn = _n / n
    o*=rn; do = _o - o; mo = min(_o,o) / max(_o,o)
    M*=rn; dM = _M - M; mM = min(_M,M) / max(_M,M)
    D*=rn; dD = _D - D; mD = min(_D,D) / max(_D,D)
    # comp baseT:
    _I,_G,_Dy,_Dx = _N.baseT; I,G,Dy,Dx = N.baseT  # I, G|D, angle
    I*=rn; dI = _I - I; mI = abs(dI) / aI
    G*=rn; dG = _G - G; mG = min(_G,G) / max(_G,G)
    mA, dA = comp_angle((_Dy,_Dx),(Dy*rn,Dx*rn))
    if isinstance(N, CL):  # dimension is distance
        _L,L = _N.L, N.L   # not cumulative
        mL,dL = min(_L,L)/ max(_L,L), _L - L
    else:  # dimension is box area
        _y0,_x0,_yn,_xn =_N.box; _A = (_yn+1-_y0) * (_xn+1-_x0)
        y0, x0, yn, xn = N.box;   A = (yn+1 - y0) * (xn+1 - x0)
        mL, dL = min(_A,A)/ max(_A,A), _A - A
        # mA, dA
    _m_,_d_ = np.array([[mM,mD,mn,mo,mI,mG,mA,mL], [dM,dD,dn,do,dI,dG,dA,dL]])
    # comp derTT:
    _i_ = _N.derTT[1]; i_ = N.derTT[1] * rn  # normalize by compared accum span
    d_ = (_i_ - i_ * dir)  # np.arrays
    _a_,a_ = np.abs(_i_),np.abs(i_)
    m_ = np.divide( np.minimum(_a_,a_), reduce(np.maximum, [_a_,a_,1e-7]))  # rms
    m_ *= np.where((_i_<0) != (i_<0), -1,1)  # match is negative if comparands have opposite sign

    # each [M,D,n,o, I,G,A,L]:
    return [m_+_m_, d_+_d_], rn

def comp_N(_N,N, ave, fi, angle=None, dist=None, dir=1, fshort=0):  # compare links, relative N direction = 1|-1, no need for angle, dist?
    dderH = []

    [m_,d_], rn = base_comp(_N, N, dir)
    baseT = np.array([(_N.baseT[0]+N.baseT[0])/2, (_N.baseT[1]+N.baseT[1])/2, *angle])  # link M,D,A
    derTT = np.array([m_, d_])
    M = np.sum(m_* w_t[0]); D = np.sum(np.abs(d_* w_t[1]))  # feedback-weighted sum
    Et = np.array([M,D, 8, (_N.Et[3]+N.Et[3]) /2])  # n comp vars, inherited olp
    _y,_x = _N.yx
    y, x = N.yx
    Link = CL(nodet=[_N,N], baseT=baseT, derTT=derTT, yx=np.add(_N.yx,N.yx)/2, L=dist, box=np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)]))
    # spec / lay:
    if fshort and M > ave and (len(N.derH) > 2 or isinstance(N,CL)):  # else base_comp only, derH is redundant to dext,vert
        dderH = comp_H(_N.derH, N.derH, rn, Link, Et, fi)  # comp shared layers, if any
        # spec/ comp_node_(node_|link_)
    Link.derH = [CLay(root=Link, Et=Et, node_=[_N,N],link_=[Link], derTT=copy(derTT)), *dderH]
    for lay in dderH: derTT += lay.derTT
    # spec / alt:
    if fi and _N.alt_ and N.alt_:
        et = _N.alt_.Et + N.alt_.Et  # comb val
        if val_(et, aw=2, fi=0) > 0:  # eval Ds
            Link.altL = comp_N(_N.alt_, N.alt_, ave*2, fi=1, angle=angle)
            Et += Link.altL.Et
    Link.Et = Et
    if Et[0] > ave * Et[2]:  # | both forks: np.add(Et[:2]) > ave * np.multiply(Et[2:])
        for rev, node, _node in zip((0,1),(N,_N),(_N,N)):  # reverse Link dir in _N.rimt
            node.et += Et
            node._N_.add(_node)
            if fi: node.rim += [(Link,dir)]
            else:  node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            add_H(node.extH, Link.derH, root=node, rev=rev, fi=0)
    return Link

def cross_comp(root, rc, iN_, fi=1):  # rc: recursion count, fc: centroid phase, cross-comp, clustering, recursion

    N__,L_,Et = comp_node_(iN_,rc) if fi else comp_link_(iN_,rc)  # flat node_ or link_
    if N__:  # CLs if not fi
        mL_,dL_ = [],[]
        for l in L_:  # not rng-banded
            if l.Et[0] > ave * l.Et[2]: mL_+= [l]
            if l.Et[1] > avd * l.Et[2]: dL_+= [l]
        nG, nG_ = [],[]
        # mfork
        if val_(Et, mw=(len(mL_)-1)*Lw, aw=(rc+2)*clust_w) > 0:  # np.add(Et[:2]) > (ave+avd) * np.multiply(Et[2:])?
            nG_ = [sum_N_(iN_)]  # default
            for N_ in N__:  # >ave rng|med bands, may be empty
                eN_ = get_exemplars(root, N_, ave*(rc+3), fi)  # typical and sparse nodes or links
                if eN_ and val_(np.sum([n.et for n in eN_],axis=0), Et, mw=((len(eN_)-1)*Lw), aw=(rc+3)*clust_w, fi=fi) > 0:
                    eN_ = cross_comp(root, rc+3, eN_, fi).node_  # no use for other params?
                    if eN_ and val_(np.sum([n.et for n in eN_],axis=0), Et, mw=((len(eN_)-1)*Lw), aw=(rc+4)*clust_w, fi=fi) > 0:
                        nG = cluster_N_(root,eN_, ave*(rc+4),rc+4, fi)  # cluster exemplars
                        if nG:
                            rnG = []  # eval recursive agglomeration forming root node_H:
                            if val_(nG.Et, Et, mw=(len(nG.node_)-1)*Lw, aw=(rc+5)*loop_w) > 0:
                                rnG = cross_comp(nG, rc+5, nG.node_)
                            nG_ += [rnG if rnG else nG]
        lG = []  # dfork
        dval = val_(Et, mw=(len(dL_)-1)*Lw, aw=(rc+3)*clust_w, fi=0)
        if dval > 0:
            if dval > ave:  # recursive derivation -> lH within node_H level, per Len band?
                lG = cross_comp(sum_N_(L_), rc+3, L2N(L_), fi=0)  # comp_link_, no CC
            else:  # lower res, dL_ eval?
                lG = cluster_N_(sum_N_(dL_), L2N(L_), ave*(rc+3), rc=rc+3, fi=0, fnodet=1)
        if nG_ or lG:
            nG = nG_[-1]  # top nG mediates lower-rng nGs
            if len(nG_) > 1:
                node_H = [list(nG.node_)]  # add nesting
                for ng in reversed(nG_[1:]):  # flatten and splice ng.node_H
                    node_H += [list(itertools.chain.from_iterable(n_)) for n_ in ng.node_]  # ng.node_: node_H of flat n_s
                nG.node_ = list(reversed(node_H))
            root.H += [[nG,lG]]  # current lev, each fork may be empty
            if nG: add_N(root,nG); add_node_H(root.H, nG.H, root)  # appends derH, H from recursion, if any
            if lG: add_N(root,lG); sum_N_(lG.node_, root=lG); root.lH += lG.H + [sum_N_(lG.node_, root=lG)]  # lH: H within node_ level
        if nG:
            return nG

def cluster_N_(root, N_, ave, rc, fi, fnodet=0):  # CC exemplar nodes via rim or links via nodet or rimt

    nG = CG(root= root); G_ = []  # flood-filled clusters
    for N in N_:
        if N.fin: continue
        N.fin = 1; node_, link_, Et, Lay = [N], [], copy(N.Et), CLay()
        if fnodet:  # not fi
            for _N in N.nodet[0].rim+N.nodet[1].rim if isinstance(N.nodet[0],CG) else [l for n in N.nodet for l,_ in n.rimt[0]+n.rimt[1]]:
                if _N in N_ and not _N.fin and _N.Et[1] > avd * _N.Et[2]:  # direct clustering by dval
                    link_ += N.nodet; Et += _N.Et; _N.fin = 1; node_ += [_N]
        else:
            for lL,_ in N.rim if fi else (N.rimt[0] + N.rimt[1]):
                _N = lL.nodet[0] if lL.nodet[1] is N else lL.nodet[1]
                if _N in N_ and not _N.fin:  # +ve only, eval by directional density?
                    link_ += [lL]
                    Et += lL.Et; _N.fin = 1; node_ += [_N]
        node_ = list(set(node_))
        if val_(Et, mw=(len(node_)-1)*Lw, aw=rc*clust_w, fi=fi) > 0:
            Lay = CLay(); n_ = node_ if fnodet else link_
            [Lay.add_lay(l) for l in sum_H(n_, nG, fi = isinstance(n_[0],CG))]
            m_,M = centroid_M(Lay.derTT[0],ave*rc)  # weigh by match to mean m|d
            d_,D = centroid_M(Lay.derTT[1],ave*rc); Lay.derTT = np.array([m_,d_])
            G_ += [sum2graph(nG, [node_, link_, np.array([M,D,*Et[2:]]), Lay], fi)]
    if G_:
        if fi: [comb_alt_(G.alt_, ave, rc) for G in G_ if isinstance(G.alt_,list)]  # no alt_ in dgraph?
        sum_N_(G_, root_G = nG)
        return nG

def get_exemplars(root, N_, ave, fi):
    exemplars = []  # next cross_comp
    _N_ = set()  # stronger-N inhibition zones

    for rdn, N in enumerate(sorted(N_, key=lambda n: n.et[0]/n.et[2], reverse=True), start=1):
        M,_,n,_ = N.et  # summed from rim
        if eval(M, weights=[ave, n, clust_w, rolp_M(M, N,_N_, fi) *rdn]):  # roM * rdn: lower rank?
            # 1 + rolp_M to intersection with stronger N inhibition zones
            NG = sum2graph(root, [[[N]+N._N_], N.rim, N.et, comb_H_(N, root)], fi)
            exemplars += [NG]; _N_.update(N._N_)
        else:
            break  # the rest of N_ is weaker
    return exemplars

def rolp_M(M, N, _N_, fi):
    oL_ = [L for L,_ in (N.rim if fi else N.rimt[0]+N.rimt[1]) if [n for n in L.nodet if n in _N_]]  # += in max dist?
    roM = 1 + sum([L.Et[0] for L in oL_]) / M  # w in range 1:2
    # rM weighting: L.Et[0] / ave*L.Et[2], proximity weighting?
    return roM

def eval(V, weights):  # conditionally progressive eval, with default ave in weights[0]
    W = 1
    for w in weights:
        W *= w
        if V < W: return 0
    return 1

def sum2graph(root, grapht, fi, minL=0, maxL=None):  # sum node and link params into graph, aggH in agg+ or player in sub+

    node_, link_, Et, mfork = grapht  # Et and mfork are summed from link_
    n0=node_[0]
    graph = CG(
        fi=fi, H = n0.H, Et=Et+n0.Et*int_w, link_=link_, box=n0.box, baseT=copy(n0.baseT), derTT=mfork.derTT, root=root, maxL=maxL,
        derH = [[mfork]])  # higher layers are added by feedback, dfork added from comp_link_:
    for L in link_:
        L.root = graph  # reassign when L is node
        if not fi:  # add mfork as link.nodet(CL).root dfork
            LR_ = set([n.root for n in L.nodet if isinstance(n.root,CG)]) # skip frame, empty roots
            if LR_:
                dfork = reduce(lambda F,f: F.add_lay(f), L.derH, CLay())  # combine lL.derH
                for LR in LR_:  # lay0 += dfork
                    if len(LR.derH[0])==2: LR.derH[0][1].add_lay(dfork)  # direct root only
                    else:                  LR.derH[0] += [dfork.copy_(root=LR)]  # init by another node
                    LR.derTT += dfork.derTT
    N_, n_,l_,lH, yx_ = [],[],[],[],[]
    fg = fi and isinstance(n0.node_[0],CG) # no PPs
    for i, N in enumerate(node_):
        if minL:  # max,min L.dist in graph.link_, inclusive, = lower-layer exclusive maxL, if G was dist-nested in cluster_N_
            fc = 0
            while N.root.maxL and N.root is not graph and (minL != N.root.maxL):  # maxL=0 in edge|frame, not fd
                if N.root is graph:
                    fc = 1; break  # graph was assigned as root via prior N
                else: N = N.root  # cluster prior-dist graphs vs nodes
            if fc:
                continue  # N.root, was clustered in prior loop
        N_ += [N]; yx_ += [N.yx]; N.root = graph  # roots if minL
        if i:
            graph.Et += N.Et*int_w; graph.baseT+=N.baseT; graph.box=extend_box(graph.box,N.box)
            if fg and N.H: add_node_H(graph.H, N.H, root=graph)
        if fg and isinstance(N.node_[0],CG): # skip Ps
            n_ += N.node_; l_ += N.link_
            if N.lH: add_node_H(lH, N.lH, graph)  # top level only
    if fg:  # pack prior top level
        graph.H += [[sum_N_(n_), sum_N_(l_)]]
        graph.lH = lH
    graph.node_ = N_  # current-dist link_ only?
    yx = np.mean(yx_, axis=0)
    dy_,dx_ = (graph.yx - yx_).T; dist_ = np.hypot(dy_,dx_)
    graph.aRad = dist_.mean()  # ave distance from graph center to node centers
    graph.yx = yx
    if not fi:  # dgraph, no mGs / dG for now  # and val_(Et, _Et=root.Et) > 0:
        alt_ = []  # mGs overlapping dG
        for L in node_:
            for n in L.nodet:  # map root mG
                mG = n.root
                if isinstance(mG, CG) and mG not in alt_:  # root is not frame
                    mG.alt_ += [graph]  # cross-comp|sum complete altG before next agg+ cross-comp, multi-layered?
                    alt_ += [mG]
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

def comb_alt_(G_, ave, rc=1):  # combine contour G.altG_ into altG (node_ defined by root=G),

    # internal vs. external alts: different decay / distance, background + contour?
    for G in G_:
        if G.alt_:
            if isinstance(G.alt_, list):
                G.alt_ = sum_N_(G.alt_)
                G.alt_.root=G; G.alt_.m=0
                if val_(G.alt_.Et, G.Et, ave, fi=0):  # alt D * G rM
                    cross_comp(G.alt_, rc, G.alt_.node_, fi=1)  # adds nesting
        elif G.H:  # not PP
            # alt_G = sum dlinks:
            dL_ = list(set([L for g in G.node_ for L,_ in (g.rim if isinstance(g, CG) else g.rimt[0]+g.rimt[1]) if val_(L.Et,G.Et, ave, fi=0) > 0]))
            if dL_ and val_(np.sum([l.Et for l in dL_], axis=0), G.Et, ave, aw=10, fi=0) > 0:
                alt_ = sum_N_(dL_)
                G.alt_ = copy_(alt_); G.altG.H = [alt_]; G.altG.root=G

def comb_H_(L_, root, fi=0):  # fi is always 0?
    if isinstance(L_,CG):
        derH = L_.extH  # L_ is an exemplar
    else:
        derH = sum_H(L_,root,fi=fi)
    Lay = CLay(root=root)
    for lay in derH:
        Lay.add_lay(lay); root.extTT += lay.derTT
    return Lay

def comp_H(H,h, rn, root, Et, fi):  # one-fork derH if not fi, else two-fork derH

    derH = []
    for _lay,lay in zip_longest(H,h):  # different len if lay-selective comp
        if _lay and lay:
            if fi:  # two-fork lays
                dLay = []
                for _fork,fork in zip_longest(_lay,lay):
                    if _fork and fork:
                        dlay = _fork.comp_lay(fork, rn,root=root)
                        if dLay: dLay.add_lay(dlay)  # sum ds between input forks
                        else:    dLay = dlay
            else:  # one-fork lays
                 dLay = _lay.comp_lay(lay, rn, root=root)
            Et += dLay.Et
            derH += [dLay]
    return derH

def sum_H(Q, root, rev=0, fi=1):  # sum derH in link_|node_
    DerH = []
    for e in Q: add_H(DerH, e.derH, root, rev, fi)
    return DerH

def add_H(H, h, root, rev=0, fi=1):  # add fork L.derHs

    for Lay,lay in zip_longest(H,h):  # different len if lay-selective comp
        if lay:
            if fi:  # two-fork lays
                if Lay:
                    for Fork,fork in zip_longest(Lay,lay):
                        if fork:
                            if Fork: Fork.add_lay(fork,rev=rev)
                            else:    Lay += [fork.copy_(root=root)]
                else:
                    Lay = []
                    for fork in lay:
                        Lay += [fork.copy_(root=root,rev=rev)]
                        root.derTT += fork.derTT; root.Et[:3] += fork.Et[:3]  # olp is not accumulated
                    H += [Lay]
            else:  # one-fork lays
                if Lay: Lay.add_lay(lay,rev=rev)
                else:   H += [lay.copy_(root=root,rev=rev)]
                root.extTT += lay.derTT; root.Et[:3] += lay.Et[:3]

def sum_N_(node_, root_G=None, root=None):  # form cluster G

    if isinstance(node_[0],list): node_ = node_[-1]  # rng-banded
    fi = isinstance(node_[0],CG)
    if root_G is not None: G = root_G
    else:
        G = copy_(node_.pop(0), init=1, root=root); G.fi=fi
    for n in node_:
        add_N(G,n, fi, fappend=1)
        if root: n.root=root
    if not fi:
        G.derH = [[lay] for lay in G.derH]  # nest
    return G

def add_N(N,n, fi=1, fappend=0):

    N.baseT+=n.baseT; N.derTT+=n.derTT; N.Et+=n.Et
    N.yx+=n.yx; N.box=extend_box(N.box, n.box)
    if isinstance(n,CG) and n.alt_:  # not CL
        N.alt_ = add_N(N.alt_ if N.alt_ else CG(), n.alt_)
    if fappend:
        N.node_ += [n]
        if fi: N.link_ += [n.link_]  # splice if CG
    elif fi:  # empty in append and altG
        if n.H: add_node_H(N.H, n.H, root=N)
        if n.lH: add_node_H(N.lH, n.lH, root=N)
    if n.derH:
        add_H(N.derH, n.derH, root=N, fi=fi)
    if hasattr(n,'extTT'):  # node, = fi?
        N.extTT += n.extTT; N.aRad += n.aRad
        if n.extH:
            add_H(N.extH, n.extH, root=N, fi=0)
    return N

def add_node_H(H, h, root):

    for Lev, lev in zip_longest(H, h, fillvalue=None):  # always aligned?
        if lev:
            if Lev:
                for i, (F,f) in enumerate(zip_longest(Lev, lev, fillvalue=None)):
                    if f:
                        if F: add_N(F,f)  # nG|lG
                        elif F is None: Lev += [f]
                        else:  Lev[i] = copy_(f, root=root)  # empty fork
            else:
                H += [copy_(f, root=root) for f in lev]

def extend_box(_box, box):  # extend box with another box
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box

    return min(y0,_y0), min(x0,_x0), max(yn,_yn), max(xn,_xn)

def L2N(link_):
    for L in link_:
        L.fi=0; L.et=np.zeros(4); L._N_=set(); L.mL_t, L.rimt = [[],[]], [[],[]]; L.aRad=0; L.extTT=np.zeros((2,8))
        L.compared_=[]; L.visited_, L.extH, L.node_, L.link_, L.H, L.lH = [],[],[],[],[],[]
        if not hasattr(L,'root'): L.root=[]
    return link_

def PP2G(PP, frame):
    P_, link_, vert, latuple, A, S, box, yx, Et = PP
    baseT = np.array((*latuple[:2], *latuple[-1]))  # I,G,Dy,Dx
    [mM,mD,mI,mG,mA,mL], [dM,dD,dI,dG,dA,dL] = vert
    derTT = np.array([[mM,mD,mL,1,mI,mG,mA,mL], [dM,dD,dL,1,dI,dG,dA,dL]])  # rolp = 1
    y,x,Y,X = box; dy,dx = Y-y,X-x
    # A = (dy,dx); L = np.hypot(dy,dx)
    G = CG(root=frame, fi=1, Et=Et, node_=P_,link_=link_, baseT=baseT, derTT=derTT, box=box, yx=yx, aRad=np.hypot(dy/2, dx/2),
           derH=[[CLay(node_=P_,link_=link_, derTT=deepcopy(derTT)), CLay()]])  # empty dfork
    return G

def norm_H(H, n):
    for lay in H:
        if lay:
            if isinstance(lay, CLay):
                for v_ in lay.derTT: v_ *= n  # array
                lay.Et *= n
            else:
                for fork in lay:
                    for v_ in fork.derTT: v_ *= n  # array
                    fork.Et *= n  # same node_, link_
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

def agg_level(inputs):  # draft parallel

    frame, H, elevation = inputs
    lev_G = H[elevation]
    cross_comp(lev_G, rc=0, iN_=lev_G.node_, fi=1)  # return combined top composition level, append frame.derH
    if lev_G:
        # feedforward
        if len(H) < elevation+1: H += [lev_G]  # append graph hierarchy
        else: H[elevation+1] = lev_G
        # feedback
        if elevation > 0:
            if np.sum( np.abs(lev_G.aves - lev_G._aves)) > ave:  # filter update value, very rough
                m, d, n, o = lev_G.Et
                k = n * o
                m, d = m/k, d/k
                H[elevation-1].aves = [m, d]
            # else break?

def agg_H_par(focus):  # draft parallel level-updating pipeline

    frame = frame_blobs_root(focus)
    intra_blob_root(frame)
    vect_root(frame)
    if frame.node_:  # converted edges
        G_ = []
        for edge in frame.node_:
            comb_alt_(edge.node_, ave)
            # cluster_C_(edge, ave)  # no cluster_C_ in vect_edge
            G_ += edge.node_  # unpack edges
        frame.node_ = G_
        manager = Manager()
        H = manager.list([CG(node_=G_)])
        maxH = 20
        with Pool(processes=4) as pool:
            pool.map(agg_level, [(frame, H, e) for e in range(maxH)])

        frame.aggH = list(H)  # convert back to list

def agg_H_seq(focus, image, rV=1, _rv_t=[]):  # recursive level-forming pipeline, called from cluster_C_

    global ave, Lw, int_w, loop_w, clust_w, ave_dist, ave_med, med_w
    ave, Lw, int_w, loop_w, clust_w, ave_dist, ave_med, med_w = np.array([ave, Lw, int_w, loop_w, clust_w, ave_dist, ave_med, med_w]) / rV
    # sum filtered params, feedback rws:~rvs?
    frame = frame_blobs_root(focus, rV)
    intra_blob_root(frame, rV)
    frame = vect_root(frame, rV, _rv_t)
    if frame.H:  # updated
        comb_alt_(frame.node_, ave*2)
        cross_comp(frame, rc=1, iN_=frame.node_)  # top level
        # adjust weights:
        rM, rD, rv_t = feedback(frame)
        if val_(frame.Et, mw=rM+rD, aw=clust_w*20):
            nG = frame.H[0][0] if frame.H[0][0] else sum_N_(frame.node_)  # base PP_ focus shift by dval + temp Dm_+Ddm_?
            dy,dx = nG.baseT[-2:]  # gA from summed Gs
            y,x,Y,X = nG.box  # current focus
            y = y+dy; x = x+dx; Y = Y+dy; X = X+dx  # alter focus shape, also focus size: +/m-, res decay?
            if y > 0 and x > 0 and Y < image.shape[0] and X < image.shape[1]:  # focus is inside the image
                # rerun agg+ with new focus and aves:
                agg_H_seq(image[y:Y,x:X], image, rV, rv_t)
                # all aves *= rV, but ultimately differential backprop per ave?
    return frame

def feedback(root):  # root is frame or lG

    rv_t = np.ones((2,8))  # sum derTT coefs: m_,d_ [M,D,n,o, I,G,A,L] / Et, baseT, dimension
    rM,rD = 1,1
    hlG = sum_N_(root.link_)
    for lev in reversed(root.H):
        lG = lev[1]  # level link_: all comp results?
        if not lG: continue
        lG.derTT[np.where(lG.derTT==0)] = 1e-7
        _m,_d,_n,_ = hlG.Et; m,d,n,_ = lG.Et
        rM += (_m/_n) / (m/n)  # o eval?
        rD += (_d/_n) / (d/n)
        rv_t += np.abs((hlG.derTT/_n) / (lG.derTT/n))
        if lG.H:  # ddfork, not recursive?
            rMd, rDd, rv_td = feedback(lG)  # intra-level recursion in lG
            rv_t = rv_t + rv_td; rM += rMd; rD += rDd

    return rM,rD,rv_t

def max_g_window(i__, wsize=64):  # set min,max coordinate filters, updated by feedback to shift the focus
    dy__ = (
            (i__[2:, :-2] - i__[:-2, 2:]) * 0.25 +
            (i__[2:, 1:-1] - i__[:-2, 1:-1]) * 0.50 +
            (i__[2:, 2:] - i__[:-2, 2:]) * 0.25 )
    dx__ = (
            (i__[:-2, 2:] - i__[2:, :-2]) * 0.25 +
            (i__[1:-1, 2:] - i__[1:-1, :-2]) * 0.50 +
            (i__[2:, 2:] - i__[:-2, 2:]) * 0.25
    )
    g__ = np.hypot(dy__, dx__)
    nY = (image.shape[0] + wsize-1) // wsize
    nX = (image.shape[1] + wsize-1) // wsize  # n windows

    max_window = g__[0:wsize, 0:wsize]; max_g = 0
    for iy in range(nY):
        for ix in range(nX):
            y0 = iy * wsize; yn = y0 + wsize
            x0 = ix * wsize; xn = x0 + wsize
            g = np.sum(g__[y0:yn, x0:xn])
            if g > max_g:
                max_window = i__[y0:yn, x0:xn]
                max_g = g
    return max_window

if __name__ == "__main__":
    image_file = './images/toucan_small.jpg'  # './images/toucan.jpg' './images/raccoon_eye.jpeg'
    image = imread(image_file)
    focus = max_g_window(image)
    frame = agg_H_seq(focus, image)  # focus will be shifted by internal feedback
''' without agg+:
    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vect_root(frame)
'''