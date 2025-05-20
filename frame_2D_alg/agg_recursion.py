import numpy as np, weakref
from copy import copy, deepcopy
from functools import reduce
from itertools import zip_longest, combinations, chain, product;  from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, intra_blob_root, imread, unpack_blob_, comp_pixel
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
class CBase:
    refs = []
    def __init__(obj):
        obj._id = len(obj.refs)
        obj.refs.append(weakref.ref(obj))
    def __hash__(obj): return obj.id
    @property
    def id(obj): return obj._id
    @classmethod
    def get_instance(cls, _id):
        inst = cls.refs[_id]()
        if inst is not None and inst.id == _id:
            return inst
    def __repr__(obj): return f"{obj.__class__.__name__}(id={obj.id})"

class CN(CBase):  # light version of CG
    name = "node"
    def __init__(N, **kwargs):
        super().__init__()
        N.N_ = kwargs.get('N_',[])  # top nodes, may include singletons of lower nodes, then links in corresponding H lev only?
        N.L_ = kwargs.get('L_',[])  # links between Ns
        N.Et = kwargs.get('Et',np.zeros(3))  # sum Ets from N_ and H
        N.et = kwargs.get('et',np.zeros(3))  # sum Ets from L_ and lH
        N.H  = kwargs.get('H', [])  # top-down: nested-node levels, each CN with corresponding L_,et,lH, no H
        N.lH = kwargs.get('lH',[])  # bottom-up: higher link graphs hierarchy, also CN levs
        N.C_ = kwargs.get('C_',[])  # make it CN?
        N.olp= kwargs.get('olp',1)  # overlap to other Ns, same for links?
        N.derH = kwargs.get('derH', [])  # [CLay], [m,d] in CG, merged in CL, sum|concat links across fork tree
        # no root?
    def __bool__(N): return bool(N.N_ or N.L_)

class CL(CN):  # link or edge, a product of comparison between two nodes or links
    name = "link"
    def __init__(L, **kwargs):
        super().__init__(**kwargs)
        # nodet: N_ or rim, may have different depth?
        L.fi =   kwargs.get('fi', 0)  # nodet fi, 1 if cluster of Ls | lGs, for feedback only? or fd_: list of forks forming G
        L.rng =  kwargs.get('rng', 1)  # or med: loop count in comp_node_|link_
        L.baseT= kwargs.get('baseT',np.zeros(4))  # I,G,Dy,Dx  # from slice_edge
        L.derTT= kwargs.get('derTT',np.zeros((2,8)))  # m,d / Et,baseT: [M,D,n,o, I,G,A,L], summed across derH lay forks
        L.yx =   kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        L.box =  kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        L.span = kwargs.get('span',0) # distance in nodet or aRad, comp with baseT and len(N_) but not additive?
        # splice L_ across N_s, cluster by diff
    def __bool__(L): return bool(L.N_)

class CG(CL):  # PP | graph | blob: params of single-fork node_ cluster
    # graph / node
    name = "graph"
    def __init__(G, **kwargs):
        super().__init__(**kwargs)
        G.extH = kwargs.get('extH',[])  # sum derH from rims, single-fork
        G.extTT= kwargs.get('extTT',np.zeros((2,8)))  # sum from extH, add baset from rim?
        G.rim =  kwargs.get('rim',[])  # external links
        G.nrim = kwargs.get('nrim',[])  # rim-linked nodes
        G.alt_ = []  # adjacent (contour) gap+overlap alt-fork graphs, converted to CG, empty alt.alt_: select+?
        G.root = kwargs.get('root')
        G.fin =  kwargs.get('fin',0)  # in cluster, temporary?
        # G.fork_tree: list = z([[]])  # indices in all layers(forks, if no fback merge
        # G.fback_ = []  # node fb buffer, n in fb[-1]
    def __bool__(G): return bool(G.N_)  # never empty

N_pars = ['N_', 'L_', 'Et', 'et', 'H', 'lH', 'C_', 'olp', 'derH']
L_pars = N_pars+ ['fi', 'rng', 'baseT', 'derTT', 'yx', 'box', 'span']
G_pars = L_pars+ ['extH', 'extTT', 'rim', 'nrim', 'alt_', 'root', 'fin', 'hL_']

class CLay(CBase):  # layer of derivation hierarchy, subset of CG
    name = "lay"
    def __init__(l, **kwargs):
        super().__init__()
        l.Et = kwargs.get('Et', np.zeros(3))
        l.olp = kwargs.get('olp', 1)  # ave nodet overlap
        l.node_ = kwargs.get('node_', [])  # concat across fork tree
        l.link_ = kwargs.get('link_', [])
        l.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m_,d_ [M,D,n,o, I,G,A,L], sum across fork tree,
        # add weights for cross-similarity, along with vertical aves, for both m_ and d_?
        # altL = CLay from comp altG
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

    def add_lay(Lay, lay_, rev=0):  # merge lays, including mlay + dlay

        if not isinstance(lay_,list): lay_ = [lay_]
        for lay in lay_:
            # rev = dir==-1, to sum/subtract numericals in m_,d_:
            for fd, Fork_, fork_ in zip((0,1), Lay.derTT, lay.derTT):
                Fork_ += fork_ * -1 if (rev and fd) else fork_  # m_| d_
            # concat node_,link_:
            Lay.node_ += [n for n in lay.node_ if n not in Lay.node_]
            Lay.link_ += lay.link_
            Lay.Et += lay.Et
            rn = lay.Et[2] / Lay.Et[2]
            Lay.olp = (Lay.olp + lay.olp*rn) /2
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
        Et = np.array([M, D, 8])  # n compared params = 8
        if root: root.Et += Et
        return CLay(Et=Et, olp=(_lay.olp+lay.olp*rn)/2, node_=node_, link_=link_, derTT=derTT)

def copy_(N, root=None, fCL=0, fCG=0, init=0):

    C, pars = (CG(),G_pars) if fCG else (CL(),L_pars) if fCL else (CN(),N_pars)
    C.root = root
    for name, val in N.__dict__.items():
        if name == "_id" or name not in pars: continue
        elif name == "N_" and init: C.N_ = [N]
        elif name == "H" and init: C.H = []
        elif name == 'derH':
            for lay in N.derH: C.derH += [[fork.copy_() for fork in lay]] if isinstance(N, CG) else [lay.copy_()]  # CL
        elif name == 'extH':   C.extH = [lay.copy_() for lay in N.extH]
        elif isinstance(val,list) or isinstance(val,np.ndarray):
            setattr(C, name, copy(val))  # Et,yx,box, node_,link_,rim_, altG, baseT, derTT
        else:
            setattr(C, name, val)  # numeric or object: fi, root, maxL, aRad, nnest, lnest
    return C

ave, avd, arn, aI, aveB, aveR, Lw, int_w, loop_w, clust_w = 10, 10, 1.2, 100, 100, 3, 5, 2, 5, 10  # value filters + weights
ave_dist, ave_med, dist_w, med_w = 10, 3, 2, 2  # cost filters + weights
wM, wD, wN, wO, wI, wG, wA, wL = 10, 10, 20, 20, 1, 1, 20, 20  # der params higher-scope weights = reversed relative estimated ave?
w_t = np.ones((2,8))  # fb weights per derTT, adjust in agg+
wY = wX = 64; wYX = np.hypot(wY,wX)  # focus dimensions
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
    frame = CG(root=None, box=frame.box, fin=1)
    lev = CN(); derlay = [CLay(root=frame),CLay(root=frame)]
    for blob in blob_:
        if not blob.sign and blob.G > aveB * blob.root.olp:
            edge = slice_edge(blob, rV)
            if edge.G*((len(edge.P_)-1)*Lw) > ave * sum([P.latuple[4] for P in edge.P_]):
                Et = comp_slice(edge, rV, ww_t)  # to scale vert
                if np.any(Et) and Et[0] > ave * Et[2] * clust_w:
                    cluster_edge(edge, frame, lev, derlay)  # may be skipped
    frame.derH = [derlay]
    if lev: frame.H += [lev]
    return frame

def cluster_edge(edge, frame, lev, derlay):  # non-recursive comp_PPm, comp_PPd, edge is not a PP cluster, unpack by default

    def cluster_PP_(PP_):
        G_ = []
        while PP_:  # flood fill
            node_,link_, et = [],[], np.zeros(3)
            PP = PP_.pop(); _eN_ = [PP]
            while _eN_:
                eN_ = []
                for eN in _eN_:  # rim-connected ext Ns
                    node_ += [eN]
                    for L,_ in eN.rim:  # all +ve, *= density?
                        if L not in link_:
                            for eN in L.N_:
                                if eN in PP_: eN_ += [eN]; PP_.remove(eN)  # merged
                            link_ += [L]; et += L.Et
                _eN_ = {*eN_}
            if val_(et, mw=(len(node_)-1)*Lw, aw=2+clust_w) > 0:  # rc=2
                Lay = CLay(); [Lay.add_lay(link.derH[0]) for link in link_]  # single-lay derH
                G_ += [sum2graph(frame, node_,link_,[], et, 1, Lay, rng=1, fi=1)]
        return G_

    def comp_PP_(PP_):
        N_,L_,mEt,dEt = [],[],np.zeros(3),np.zeros(3)

        for _G, G in combinations(PP_, r=2):
            dy,dx = np.subtract(_G.yx,G.yx); dist = np.hypot(dy,dx)
            if dist - (G.span+_G.span) < ave_dist / 10:  # very short here
                L = comp_N(_G, G, ave, fi=isinstance(_G, CG), angle=[dy, dx], span=dist, fdeep=1)
                m, d, n = L.Et
                if m > ave * n * _G.olp * loop_w: mEt += L.Et; N_ += [_G,G]  # mL_ += [L]
                if d > avd * n * _G.olp * loop_w: dEt += L.Et  # dL_ += [L]
                L_ += [L]
        dEt[2] = dEt[2] or 1e-7
        return set(N_),L_,mEt,dEt

    PP_ = edge.node_
    if val_(edge.Et, mw=(len(PP_)- edge.rng) *Lw, aw=loop_w) > 0:
        PP_,L_,mEt,dEt = comp_PP_([PP2N(PP,frame) for PP in PP_])
        if PP_:
            if val_(mEt, mw=(len(PP_)-1)*Lw, aw=clust_w, fi=1) > 0:
                G_ = cluster_PP_(copy(PP_))
            else: G_ = []
            if G_: frame.N_ += G_; lev.N_ += PP_
            else:  frame.N_ += PP_ # PPm_
        lev.L_+= L_  # links between PPms
        for l in L_:
            derlay[0].add_lay(l.derH[0]); frame.baseT+=l.baseT; frame.derTT+=l.derTT; frame.Et += l.Et
        if val_(dEt, mw= (len(L_)-1)*Lw, aw= 2+clust_w, fi=0) > 0:
            Lt = cluster_N_(L2N(L_), rc=2, fi=0)
            if Lt:  # L_ graphs
                if lev.lH: lev.lH[0].N_ += Lt.N_; lev.lH[0].Et += Lt.Et
                else:      lev.lH += [Lt]

def val_(Et, _Et=None, mw=1, aw=1, fi=1):  # m+d val per cluster|cross_comp

    m, d, n = Et  # m->d lend cancels in Et scope, not higher-scope _Et?
    am, ad = ave * aw, avd * aw  # includes olp
    k = mw / n
    m, d = m*k, d*k
    if np.any(_Et):  # higher scope values
        _m,_d,_n = _Et; _k = mw /_n
        m += _m *_k  # local + global
        d += _d *_k
    dval = (d-ad) * (m/am)  # ddev borrow per rational mdev
    mval = (m-am) - dval  # additive if neg dval

    return mval if fi else dval

def eval(V, weights):  # conditionally progressive eval, with default ave in weights[0]
    W = 1
    for w in weights:
        W *= w
        if V < W: return 0
    return 1
''' 
 Core process per agg level, add feedback to lower levels:
 Cross-compare nodes, evaluate incremental-derivation cross-comp of new >ave difference links, recursively.
 
 Select sparse exemplars of strong node types, convert to centroids of their rim, reselect by mutual match.
 This selection is essential due to complexity of subsequent connectivity clustering (CC):
  
 Connectivity cluster exemplar nodes by >ave match links, correlation-cluster links by >ave difference.
 Evaluate resulting clustered node_ or link_ for recursively higher-composition cross_comparison. 

 Min distance in CC is more restrictive than in cross-comp due to higher costs and density eval.
 CCs terminate at contour altGs, and next-level cross-comp is between core+contour clusters. '''

def comp_node_(_N_, rc):  # rng+ forms layer of rim and extH per N?

    for n in _N_: n.compared_ = []
    N__,L_,ET = [],[],np.zeros(3)  # range banded if frng, default rng=1
    rng, olp_ = 1,[]
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
            (_m,_,_n),(m,_,n) = _G.Et,G.Et; olp = (_G.olp+G.olp)/2; _m = _m * int_w +_G.et[0]; m = m * int_w + G.et[0]
            # density / comp_N: et|M, 0/rng=1?
            max_dist = ave_dist * (radii/aveR) * ((_m+m)/(ave*(_n+n+0))/ int_w)  # ave_dist * radii * induction
            if max_dist > dist or set(_G.nrim) & set(G.nrim):
                # comp if close or share matching mediators: add to inhibited?
                Link = comp_N(_G,G, ave, fi=1, angle=[dy,dx], span=dist, fdeep = dist < max_dist/2, rng=rng)
                L_ += [Link]  # include -ve links
                if Link.Et[0] > ave * Link.Et[2] * loop_w * olp:
                    Et += Link.Et; olp_ += [olp]  # link.olp is the same with o
                    for n,_n in zip((_G,G),(G,_G)):
                        n.compared_ += [_n]
                        if n not in N_ and val_(n.et, aw= rc+rng-1+loop_w+olp) > 0:  # cost+/ rng
                            n.med = rng; N_ += [n]  # for rng+ and exemplar eval
        N__ += [N_]; ET += Et
        if val_(Et, mw = (len(N_)-1)*Lw, aw = loop_w* sum(olp_)/len(olp_)) > 0:  # current-rng vM
            _N_ = N_; rng += 1; olp_ = []  # reset
        else:  # low projected rng+ vM
            break
    return N__,L_,ET

def comp_link_(iL_, rc):  # comp CLs via directional node-mediated link tracing: der+'rng+ in root.link_ rim_t node rims

    fi = isinstance(iL_[0].N_[0], CG)
    for L in iL_:
        # init mL_t: bilateral mediated Ls per L:
        for rev, N, mL_ in zip((0,1), L.N_, L.mL_t):
            for _L,_rev in N.rim if fi else N.rimt[0]+N.rimt[1]:
                if _L is not L and _L in iL_:  # nodet-mediated
                    if L.Et[0] > ave * L.Et[2] * loop_w:
                        mL_ += [(_L, rev ^_rev)]  # direction of L relative to _L
    med = 1; _L_ = iL_
    L__,LL_,ET = [],[],np.zeros(3)
    while True:  # xcomp _L_
        L_, Et = [], np.zeros(3)
        for L in _L_:
            for mL_ in L.mL_t:
                for _L, rev in mL_:  # rev is relative to L
                    if _L in L.compared_: continue
                    dy,dx = np.subtract(_L.yx,L.yx)
                    Link = comp_N(_L,L, ave*rc, fi=0, angle=[dy,dx], span=np.hypot(dy,dx), dir = -1 if rev else 1)  # d = -d if L is reversed relative to _L
                    Link.med = med
                    LL_ += [Link]; Et += Link.Et  # include -ves, link order: nodet < L < rimt, mN.rim || L
                    for l,_l in zip((_L,L),(L,_L)):
                        l.compared_ += [_l]
                        if l not in L_ and val_(l.et, aw= rc+med+loop_w) > 0:  # cost+/ rng
                            L_ += [l]  # for med+ and exemplar eval
        L__ += [L_]; ET += Et
        # extend mL_t per last medL:
        if Et[0] > ave * Et[2] * (rc + loop_w + med*med_w):  # project prior-loop value, med adds fixed cost
            _L_, ext_Et = set(), np.zeros(3)
            for L in L_:
                mL_t, lEt = [set(),set()], np.zeros(3)  # __Ls per L
                for mL_,_mL_ in zip(mL_t, L.mL_t):
                    for _L, rev in _mL_:
                        for _rev, N in zip((0,1), _L.N_):
                            rim = N.rim if fi else N.rimt
                            if len(rim) == med:  # append in comp loop
                                for __L,__rev in rim if fi else rim[0]+rim[1]:
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
    return L__, LL_, ET

def base_comp(_N, N, dir=1):  # comp Et, Box, baseT, derTT
    # comp Et:
    _M,_D,_n = _N.Et; M,D,n = N.Et
    dn = _n - n; mn = min(_n,n) / max(_n,n)  # or multiplicative for ratios: min * rn?
    rn = _n / n  # add _o/o?
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
    if isinstance(N,CG): # dimension is n nodes
        _L,L = len(_N.N_), len(N.N_)
        mL,dL = min(_L,L)/ max(_L,L), _L - L
    else:  # dimension is distance
        _L,L = _N.span, N.span   # dist, not cumulative, still positive?
        mL,dL = min(_L,L)/ max(_L,L), _L - L
    # comp len H, density: combined from N_ in links?
    _m_,_d_ = np.array([[mM,mD,mn,mo,mI,mG,mA,mL], [dM,dD,dn,do,dI,dG,dA,dL]])
    # comp derTT:
    _i_ = _N.derTT[1]; i_ = N.derTT[1] * rn  # normalize by compared accum span
    d_ = (_i_ - i_ * dir)  # np.arrays
    _a_,a_ = np.abs(_i_),np.abs(i_)
    m_ = np.divide( np.minimum(_a_,a_), reduce(np.maximum, [_a_,a_,1e-7]))  # rms
    m_ *= np.where((_i_<0) != (i_<0), -1,1)  # match is negative if comparands have opposite sign
    m_ += _m_; d_ += _d_
    DerTT = np.array([m_,d_])  # [M,D,n,o, I,G,A,L]:
    Et = np.array([np.sum(m_* w_t[0]), np.sum(np.abs(d_* w_t[1])), min(_n,n)])  # feedback-weighted sum shared quantity between comparands
    # centroid_M?
    return DerTT, Et, rn

def comp_N(_N,N, ave, fi, angle=None, span=None, dir=1, fdeep=0, rng=1):  # compare links, relative N direction = 1|-1, no need for angle, dist?

    dderH = []
    derTT, Et, rn = base_comp(_N, N, dir)
    # link M,D,A:
    baseT = np.array([*(_N.baseT[:2] + N.baseT[:2]*rn) / 2, *angle])  # redundant angle for generic base_comp, also span-> density?
    _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
    o = (_N.olp+N.olp) / 2
    Link = CL(rng=rng, olp=o, N_=[_N,N], baseT=baseT, derTT=derTT, yx=np.add(_N.yx,N.yx)/2, span=span, angle=angle, box=box)
    # spec / lay:
    if fdeep and (val_(Et, mw=len(N.derH)-2, aw=o) > 0 or N.name=='link'):  # else derH is dext,vert
        dderH = comp_H(_N.derH, N.derH, rn, Link, Et, fi)  # comp shared layers, if any
        # spec/ comp_node_(node_|link_)
    Link.derH = [CLay(root=Link, Et=Et, node_=[_N,N],link_=[Link], derTT=copy(derTT)), *dderH]
    for lay in dderH: derTT += lay.derTT
    # spec / alt:
    if fi and _N.alt_ and N.alt_:
        et = _N.alt_.Et + N.alt_.Et  # comb val
        if val_(et, aw=2+o, fi=0) > 0:  # eval Ds
            Link.alt_L = comp_N(_N.alt_, N.alt_, ave*2, fi=1, angle=angle); Et += Link.alt_L.Et
    Link.Et = Et
    if Et[0] > ave * Et[2]:
        for rev, node, _node in zip((0,1),(N,_N),(_N,N)):  # reverse Link dir in _N.rimt
            node.et += Et
            node.nrim += [node]
            if fi: node.rim += [(Link,rev)]
            else: node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            add_H(node.extH, Link.derH, root=node, rev=rev, fi=0)
    return Link

# core function
def cross_comp(iN_, rc, root, fi=1):  # rc: redundancy; (cross-comp, exemplar selection, clustering), recursion

    N__,L_,Et = comp_node_(iN_,rc) if fi else comp_link_(iN_,rc)  # rc has root.olp
    # if root[-1]: C_= comp_node_(C_), rng*= C.k, C_ olp N_?
    if N__:  # CLs if not fi
        Nt, n__ = [],[]
        for n in {N for N_ in N__ for N in N_}: n__ += [n]; n.sel = 0  # for cluster_N_
        # mfork:
        if val_(Et, mw=(len(n__)-1)*Lw, aw=rc+loop_w) > 0:  # local rc+
            E_,eEt = select_exemplars(root, n__, rc+loop_w, fi)  # typical sparse nodes, refine by cluster_C_
            if val_(eEt, mw=(len(E_)-1)*Lw, aw=rc+clust_w) > 0:
                for rng, N_ in enumerate(N__,start=1):  # bottom-up
                    rng_E_ = [n for n in N_ if n.sel]  # cluster via rng exemplars
                    if rng_E_ and val_(np.sum([n.Et for n in rng_E_],axis=0), mw=(len(rng_E_)-1)*Lw, aw=rc+clust_w*rng) > 0:
                        hNt = cluster_N_(rng_E_, rc+clust_w*rng, fi,rng)
                        if hNt: Nt = hNt  # else keep lower rng
                if Nt and val_(Nt.Et, Et, mw=(len(Nt.N_)-1)*Lw, aw=rc+clust_w*rng+loop_w) > 0:
                    cross_comp(Nt.N_, rc+clust_w*rng+loop_w, root=Nt)  # top rng, select lower-rng spec comp_N: scope+?
        Lt = []  # dfork:
        dval = val_(Et, mw=(len(L_)-1)*Lw, aw=rc+3+clust_w, fi=0)
        if dval > 0:
            L_ = L2N(L_)
            if dval > ave:  # recursive derivation -> lH / nLev, rng-banded?
                cross_comp(L_, rc+loop_w*2, root, fi=0)  # comp_link_, no centroids?
            else:  # lower res
                Lt = cluster_N_(L_, rc+clust_w*2, fi=0, fnode_=1)  # overlaps the mfork above
        if Nt:
            # higher-level feedback:
            new_lev = CN(N_=Nt.N_, Et=Nt.Et)  # pack N_ in H
            add_NH(root.H, Nt.H+[new_lev], root)  # assuming same H elevation?
            if Nt.H:  # lower levs: derH,H if recursion
                root.N_ = Nt.H.pop().N_  # top lev nodes
            comb_alt_(Nt.N_, rc + clust_w * 3)  # from dLs
        root.L_ = L_  # N_ links
        if Lt:
            root.et += Lt.Et; root.lH += [Lt] + Lt.H  # link graphs, flatten H if recursive?

def select_exemplars(root, N_, rc, fi, fC=0):  # get sparse representative nodes|links: non-maximum suppression via stronger-N inhibition zones

    exemplars = []; Et = np.zeros(3); _N_ = set()  # stronger-N inhibition zones

    for rdn, N in enumerate(sorted(N_, key=lambda n: n.et[0]/n.et[2], reverse=True), start=1):
        M,_,n = N.et  # summed from rim
        if eval(M, weights=[ave*rc, n, clust_w, rolp_M(M, N,_N_, fi, fC) *rdn]):  # roM * rdn: lower rank?
            # 1 + relative match to the intersection with stronger N inhibition zones
            Et += N.et; _N_.update(N.nrim)  # add as olp in exemplar?
            N.sel = 1  # select for cluster_N_
            exemplars += [N]
        else:
            break  # the rest of N_ is weaker
    if fi and not fC and val_(Et, mw=(len(exemplars)-1)*Lw, aw=rc+clust_w) > 0:
        cluster_C_(root, exemplars, Et, rc+clust_w, fi)  # refine _N_+_N__ by mutual similarity, not recursive in current scope

    return exemplars, Et

def rolp_M(M, N, _N_, fi, fC=0):  # rel sum of overlapping links Et, or _N_ Et in cluster_C_?

    if fC:  # from cluster_C_
        olp_N_ = [n for n in N.nrim if n in _N_]  # previously accumulated inhibition zone
    else:   # from sel_exemplars
        olp_N_ = [L for L, _ in (N.rim if fi else N.rimt[0] + N.rimt[1]) if
               [n for n in L.N_ if n in _N_]]  # += in max dist?
    roM = 1 + sum([n.Et[0] for n in olp_N_]) / M  # w in range 1:2
    # weigh by L.Et[0] / ave*L.Et[2]?
    return roM

def cluster_C_(root, E_,eEt, rc, fi):  # form centroids from exemplar _N_, drifting / competing via rims of new member nodes

    for N in E_:
        N.C_, N.et_ = [],[];  N.nrim += [N]  # N is a member
        N.nrim_ = list({_n for n in N.nrim for _n in n.nrim})  # surround + members for comp to _N_ mean
    while True:
        DEt, ET = np.zeros(3), np.zeros(3)  # olp?
        C_ = []
        for N in sorted(E_,key=lambda n: n.et[0]/n.et[2], reverse=True):  # et summed from rim
            if val_(N.et, mw=(len(N.nrim)-1)*Lw, aw=clust_w+rc) > 0:
                # get the mean of cluster members in nrim:
                C = sum_N_(copy(N.nrim), root=root)
                C.nrim_= list(set([_n for n in N.nrim_ for _n in n.nrim_]))
                if C.alt_ and isinstance(C.alt_, list): C.alt_ = sum_N_(C.alt_)
                k = len(N.nrim)
                for n in (C, C.alt_):
                    if n:
                        n.Et /= k; n.baseT /= k; n.derTT /= k; n.aRad /= k; n.yx /= k
                        if n.derH: norm_H(n.derH, k)
                _N_,_N__= [],[]
                Et, dEt = np.zeros(3), np.zeros(3)
                for n in N.nrim_:
                    _,et,_ = base_comp(C,n)  # comp to mean
                    if C.alt_ and n.alt_: _,aet,_ = base_comp(C.alt_,n.alt_); et += aet
                    if N in n.C_: dEt += et - n.et_[n.C_.index(N)]  # assigned in prior loop
                    else:         dEt += et
                    if val_(et, aw=loop_w+rc) > 0:
                         n.C_ += [C]; n.et_ += [et]; _N_ += [n]; _N__ += n.nrim
                    Et += et
                C.et = Et; C.nrim = list(set(_N_)); C.nrim_ = list(set(_N__))
                C.olp *= rolp_M(Et[0], C, _N_, 1, fC=1)  # roM?
                if val_(C.et, aw=clust_w+rc+C.olp) > 0:
                    C_ += [C]; DEt += dEt; ET += Et
                else: ET += N.et
            else:
                ET += N.et
                break  # rest of N_ is weaker
        if val_(DEt, mw=(len(C_)-1)*Lw, aw=(loop_w+clust_w+rc)) <= 0:
            break  # converged

    if val_(ET, mw=(len(C_)-1)*Lw, aw=rc+loop_w) > 0:
        root.C_ = C_  # higher-scope cross_comp in agg_search?
        Ec_, Et = select_exemplars(root, [n for C in C_ for n in C.N_], rc+loop_w, fi=fi, fC=1)
        if Ec_:   # refine exemplars by cEt
            remove_ = {n for C in C_ for n in C.N_}
            E_[:] = [n for n in E_ if n not in remove_] + Ec_
            ET += eEt

def cluster_N_(N_, rc, fi, rng=1, fnode_=0, root=None):  # connectivity cluster exemplar nodes via rim or links via nodet or rimt

    G_ = []  # flood-filled clusters
    for n in N_: n.fin = 0
    for N in N_:
        if N.fin: continue
        if rng == 1:
            node_, link_, llink_, Et, olp = [N],[],[], copy(N.Et), N.olp
            for l,_ in N.rim if fi else (N.rimt[0]+N.rimt[1]):  # +ve
                if l.rng ==rng: link_ += [l]
                elif l.rng>rng: llink_+= [l]  # longer-rng rim
        else:  # rng > 1, cluster top-rng roots instead
            n = N; R = n.root
            while R and R.rng > n.rng: n = R; R = R.root
            if R.fin: continue
            node_,link_,llink_,Et,olp = [R],R.L_,R.hL_,copy(R.Et),R.olp
            R.fin = 1
        nrc = rc+olp; N.fin = 1
        if fnode_:
            # cluster via nodet, which may also be CLs
            for _N,_ in N.N_[0].rim+N.N_[1].rim if isinstance(N.N_[0],CG) else list(set([lt for n in N.N_ for lt in n.rimt[0]+n.rimt[1]])):
                if not _N.fin and _N.Et[1] > avd * _N.Et[2] * nrc:
                    Et += _N.Et; olp += _N.olp; _N.fin = 1; node_ += [_N]
                    link_ += [n for n in _N.N_ if n not in link_]
        else:  # cluster via links
            for L in link_[:]:  # snapshot
                for _N in L.N_:
                    if _N not in N_ or _N.fin: continue  # connectivity clusters don't overlap
                    if rng == 1:
                        node_ += [_N]; Et += _N.Et; olp += _N.olp; _N.fin = 1
                        for l,_ in _N.rim if fi else (_N.rimt[0]+_N.rimt[1]):  # +ve
                            if l not in link_ and l.rng == rng: link_ += [l]
                            elif l not in llink_ and l.rng>rng: llink_+= [l]  # longer-rng rim
                    else:  # rng > 1, cluster top-rng roots if rim intersect:
                        _n =_N; _R=_n.root
                        while _R and isinstance(R, CG) and _R.rng > _n.rng: _n=_R; _R=_R.root
                        if isinstance(_R, list) or _R.fin: continue
                        lenI = len(list(set(llink_) & set(_R.hL_)))
                        if lenI and (lenI / len(llink_) >.2 or lenI / len(_R.hL_) >.2):  # min rim intersect | intersect oEt?
                            link_ = list(set(link_+_R.link_)); llink_ = list(set(llink_+_R.hL_))
                            node_+= [_R]; Et +=_R.Et; olp += _R.olp; _R.fin = 1; _N.fin = 1
        node_ = list(set(node_))
        nrc = rc + olp  # updated
        if val_(Et, mw=(len(node_)-1)*Lw, aw=nrc, fi=fi) > 0:
            Lay = CLay()  # sum combined n.derH:
            [Lay.add_lay(lay) for n in (node_ if fnode_ else link_) for lay in n.derH]  # always CL?
            m_,M = centroid_M(Lay.derTT[0],ave*nrc)  # weigh by match to mean m|d
            d_,D = centroid_M(Lay.derTT[1],ave*nrc); Lay.derTT = np.array([m_,d_])
            Et = Lay.Et + np.array([M, D, Et[2]]) * int_w
            olp = (Lay.olp + olp*int_w) / len(node_)
            G_ += [sum2graph(root, node_, link_, llink_, Et, olp, Lay, rng, fi)]
    if G_:
        return CN(N_=G_, Et=Et)   # root replace if fi else append?

def sum2graph(root, node_,link_,llink_,Et,olp, Lay, rng, fi):  # sum node and link params into graph, aggH in agg+ or player in sub+

    n0 = node_[0]; C_=root.C_ if root else []
    graph = CG(fi=fi,rng=rng,olp=olp, Et=Et,et=Lay.Et, N_=node_,L_=link_, box=n0.box, baseT=copy(n0.baseT), derTT=Lay.derTT, derH=[[Lay]], C_=C_,root=root)
    graph.hL_ = llink_
    n0.root = graph; yx_ = [n0.yx]; fg = fi and isinstance(n0.N_[0],CG)  # not PPs
    Nt = copy_(n0)  #->CN, comb forks: add_N(Nt,Nt.Lt)?
    for N in node_[1:]:
        graph.baseT+=N.baseT; graph.box=extend_box(graph.box,N.box); yx_ += [N.yx]
        if fg: add_N(Nt, N)
    if fg: graph.H = Nt.H + [Nt]  # pack prior top level
    yx = np.mean(yx_,axis=0); dy_,dx_ = (graph.yx-yx_).T; dist_ = np.hypot(dy_,dx_)
    graph.span = dist_.mean()  # ave distance from graph center to node centers
    graph.yx = yx
    if not fi:  # add mfork as link.node_(CL).root dfork
        for L in link_:  # higher derH layers are added by feedback, dfork added from comp_link_:
            LR_ = set([n.root for n in L.N_ if isinstance(n.root,CG)])  # nodet, skip frame, empty roots
            if LR_:
                dfork = reduce(lambda F,f: F.add_lay(f), L.derH, CLay())  # combine lL.derH
                for LR in LR_:  # lay0 += dfork
                    LR.derTT += dfork.derTT
                    if len(LR.derH[0])==2: LR.derH[0][1].add_lay(dfork)  # direct root only: 1-layer derH
                    else:                  LR.derH[0] += [dfork.copy_()]  # init by another node
                    if LR.lH: LR.lH[-1].N_ += [graph]  # last lev
                    else:     LR.lH += [CN(N_=[graph])]  # init
        alt_=[]  # add mGs overlapping dG
        for L in node_:
            for n in L.N_:  # map root mG
                mG = n.root
                if isinstance(mG, CG) and mG not in alt_:  # root is not frame
                    mG.alt_ += [graph]  # cross-comp|sum complete alt before next agg+ cross-comp, multi-layered?
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

def comb_alt_(G_, rc=1):  # combine contour G.altG_ into altG (node_ defined by root=G),
    # internal vs. external alts: different decay / distance, background + contour?
    for G in G_:
        o = G.olp
        if G.alt_:
            if isinstance(G.alt_, list):
                G.alt_ = sum_N_(G.alt_)  # G.alt_.root=G; G.alt_.m=0 or remove sum_N_ and sum Et separately?
                if val_(G.alt_.Et, G.Et, aw=o, fi=0):  # alt D * G rM
                    cross_comp(G.alt_.N_, rc, fi=1, root=G.alt_)  # adds nesting
        elif G.H:  # not PP
            # alt_G = sum dlinks:
            dL_ = list(set([L for g in G.N_ for L,_ in (g.rim if isinstance(g, CG) else g.rimt[0]+g.rimt[1]) if val_(L.Et,G.Et, aw=o, fi=0) > 0]))
            if dL_ and val_(np.sum([l.Et for l in dL_],axis=0), G.Et, aw=10+o, fi=0) > 0:
                alt_ = sum_N_(dL_)
                G.alt_ = copy_(alt_); G.alt_.H = [alt_]; G.alt_.root=G

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
                        root.derTT += fork.derTT; root.Et += fork.Et
                    H += [Lay]
            else:  # one-fork lays
                if Lay: Lay.add_lay(lay,rev=rev)
                else:   H += [lay.copy_(rev=rev)]
                root.extTT += lay.derTT; root.Et += lay.Et

def sum_N_(node_, root_G=None, root=None, fCL=0, fCG=0):  # form cluster G

    fi = isinstance(node_[0],CG); lenn = len(node_)
    if root_G is not None: G = root_G
    else:
        G = copy_(node_.pop(0), init=1, root=root); G.fi=fi
    for n in node_:
        add_N(G, n, fi, fCL, fCG, fappend=1)
        if root: n.root=root
    if not fi:
        G.derH = [[lay] for lay in G.derH]  # nest
    if hasattr(G, 'nrim'): G.nrim = list(set(G.nrim))  # nrim available in CG only
    G.olp /= lenn
    return G

def add_N(N,n, fi=1, fCL=0, fCG=0, fappend=0):

    rn = n.Et[2]/N.Et[2]
    N.olp += n.olp * rn  # olp is normalized later
    N.Et += n.Et * rn; N.et += n.et * rn
    if n.H: add_NH(N.H, n.H, root=N)
    if n.lH: add_NH(N.lH, n.lH, root=N)
    if fappend: N.N_ += [n]
    if hasattr(n,'C_') and n.C_: N.C_ += n.C_  # centroids
    if fCL:
        N.box=extend_box(N.box,n.box); N.span = max(N.span,n.span); N.angle+=n.angle*rn  # span angle in CG?
        N.baseT+=n.baseT*rn; N.derTT+=n.derTT*rn; N.yx+=n.yx*rn
        if n.derH: add_H(N.derH, n.derH, root=N, fi=fi)
        if fCG:
            N.nrim += n.nrim
            if n.extTT: N.extTT += n.extTT
            if n.extH: add_H(N.extH, n.extH, root=N, fi=0)
            if n.alt_: N.alt_ = add_N(N.alt_ if N.alt_ else CG(), n.alt_)
    # N = CG if fCG, else CL if fCL, else CN?
    return N

def add_NH(H, h, root):

    for Lev, lev in zip_longest(H, h, fillvalue=None):  # always aligned?
        if lev:
            if Lev: add_N(Lev,lev)
            else: H += [copy_(lev, root=root)]

def extend_box(_box, box):  # extend box with another box
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box

    return min(y0,_y0), min(x0,_x0), max(yn,_yn), max(xn,_xn)

def L2N(link_):
    for L in link_:
        L.fi=0; L.et=np.zeros(3); L.nrim=[]; L.mL_t, L.rimt = [[],[]], [[],[]]; L.extTT=np.zeros((2,8)); L.compared_, L.visited_, L.extH = [],[],[]; L.fin = 0
        if not hasattr(L,'root'): L.root=[]
    return link_

def PP2N(PP, frame):

    P_, link_, vert, latuple, A, S, box, yx, Et = PP
    baseT = np.array((*latuple[:2], *latuple[-1]))  # I,G,Dy,Dx
    [mM,mD,mI,mG,mA,mL], [dM,dD,dI,dG,dA,dL] = vert  # re-pack in derTT:
    derTT = np.array([[mM,mD,mL,1,mI,mG,mA,mL], [dM,dD,dL,1,dI,dG,dA,dL]])
    derH = [[CLay(node_=P_, link_=link_, derTT=deepcopy(derTT)), CLay()]]  # empty dfork
    y,x,Y,X = box; dy,dx = Y-y,X-x  # A = (dy,dx); L = np.hypot(dy,dx), rolp = 1
    et = np.array([*np.sum([L.Et for L in link_],axis=0), 1]) if link_ else np.array([.0,.0,1.])  # n=1

    return CG(root=frame, fi=1, Et=Et,et=et, N_=P_,L_=link_, baseT=baseT, derTT=derTT, derH=derH, box=box, yx=yx, span=np.hypot(dy/2,dx/2))

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

def init_frame(i__):  # set frame and focus, updated by feedback to shift the focus

    dert__ = comp_pixel(i__)
    Y, X = image.shape[0], image.shape[1]
    nY = (Y+ wY-1) // wY; nX = (X+ wY-1) // wX  # image / dimension -> n windows
    win_, max_g = [], 0
    for iy in range(nY):
        for ix in range(nX):
            y0 = iy * wY; yn = y0 + wY; x0 = ix * wX; xn = x0 + wX
            g = np.sum(dert__[3][y0:yn, x0:xn])  # g__
            if g > max_g:
               focus = np.array([y0,x0,yn,xn])
               max_g = g
    frame = CG(box = np.array([0,0,Y,X]), yx = np.array([Y/2,X/2]))  # define frame with whole image
    return frame, focus, dert__

def feedback(root):  # root is frame or lG

    rv_t = np.ones((2,8))  # sum derTT coefs: m_,d_ [M,D,n,o, I,G,A,L] / Et, baseT, dimension
    rM, rD = 1,1
    hLt = sum_N_(root.L_)  # links between top nodes
    _derTT = np.sum([l.derTT for l in hLt.N_])  # _derTT[np.where(derTT==0)] = 1e-7
    for lev in reversed(root.H):  # top-down
        if not lev.lH: continue
        Lt = lev.lH[-1]  # dfork
        _m,_d,_n = hLt.Et; m,d,n = Lt.Et
        rM += (_m/_n) / (m/n)  # o eval?
        rD += (_d/_n) / (d/n)
        derTT = np.sum([l.derTT for l in Lt.N_])  # top link_ is all comp results
        rv_t += np.abs((_derTT/_n) / (derTT/n))
        if Lt.lH:  # ddfork, not recursive?
            rMd, rDd, rv_td = feedback(Lt)  # intra-level recursion, always dfork
            rv_t = rv_t + rv_td
            rM += rMd; rD += rDd
    return rM, rD, rv_t

def agg_search(frame, focus,foci, image, rV=1, _rv_t=[], dert__=None):  # recursive level-forming pipeline

    global ave, Lw, int_w, loop_w, clust_w, ave_dist, ave_med, med_w
    ave, Lw, int_w, loop_w, clust_w, ave_dist, ave_med, med_w = np.array([ave, Lw, int_w, loop_w, clust_w, ave_dist, ave_med, med_w]) / rV
    # fb rws: ~rvs
    node_,C_ = [],[]  # extend frame.node_ with new foci from search
    y,x, Y,X = focus
    Fg = frame_blobs_root(image[y:Y,x:X],rV, dert__[y:Y,x:X]); Fg.box=focus; intra_blob_root(Fg,rV)
    Fg = vect_root(Fg, rV,_rv_t)
    lenn_, depth = len(Fg.N_), 0
    while len(Fg.H) > depth:  # added in cross_comp
        comb_alt_(Fg.N_, ave*2)
        cross_comp(Fg.N_, root=Fg, rc=frame.olp+loop_w)  # top level, Ft += G.C_ in sum_N_?
        # adjust weights: all aves *= rV, ultimately differential backprop per ave? this adjustment should be after search cycle
        rM, rD, rv_t = feedback(Fg)
        dy, dx = map(int, Fg.baseT[-2:])  # gA from summed Gs, eval focus shift:
        if val_(Fg.Et, mw=(rM+rD) * (np.hypot(dy,dx)/wYX), aw=frame.olp+clust_w*20):  # focus shift by dval + temp Dm_+Ddm_?
            ny = (abs(dy) + (wY-1)) // wY * np.sign(dy)  # ≥1 if _dy>0, n new windows along axis
            nx = (abs(dx) + (wX-1)) // wX * np.sign(dx)
            _y,_x,_Y,_X = focus
            y = _y+ ny*wY; x = _x+ nx*wX; Y = _Y+ ny*wY; X = _X+ nx*wX  # next box
            if y >= 0 and x >= 0 and Y < frame.box[2] and X < frame.box[3]:  # focus inside the image
                np.int_([y,x,Y,X]); fin = 0
                for _y,_x,_,_ in foci:
                    if y==_y and x==_x: fin=1; break
                if not fin:  # new focus
                    foci += [focus]  # rerun agg+ with new focus window and aves:
                    Fg = agg_search(frame, focus,foci, image, rV, rv_t, dert__[y:Y,x:X])
                    if Fg: node_ += Fg.N_; depth = len(Fg.H)
                    else: break
                else: break
            else: break
        else: break  # tentative
    # scope+ node_-> agg+ H, splice and cross_comp centroids in G.C_ within focus ) frame?
    if node_:
        Fg = sum_N_(node_, root=frame, fCG=1)
        if val_(frame.Et, mw=len(node_)/lenn_*Lw, aw=frame.olp+clust_w*20) > 0:  # node_ is combined across foci
            cross_comp(node_, rc=frame.olp+loop_w, root=Fg)
            # project higher-scope Gs, eval for new foci, splice foci into frame
        return Fg

if __name__ == "__main__":
    image = imread('./images/toucan_small.jpg')  # './images/toucan.jpg' './images/raccoon_eye.jpeg'
    frame, focus, dert__ = init_frame(image)
    frame = agg_search(frame,focus,[], image, dert__)  # focus is shifted within a frame by internal feedback
''' without agg+:
    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vect_root(frame)
'''