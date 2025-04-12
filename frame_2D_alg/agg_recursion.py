import numpy as np
from copy import copy, deepcopy
from functools import reduce
from itertools import zip_longest, combinations
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
            # rev = dir==-1, to sum/subtract numericals in m_ and d_:
            for fd, (F_, f_) in enumerate(zip(Lay.derTT, lay.derTT)):
                F_ += f_ * -1 if (rev and fd ) else f_  # m_|d_
            # concat node_,link_:
            Lay.node_ += [n for n in lay.node_ if n not in Lay.node_]
            Lay.link_ += lay.link_
            Lay.Et += lay.Et
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
        G.node_ = kwargs.get('node_',[])  # flat
        G.link_ = kwargs.get('link_',[])  # spliced node link_s
        G.H = kwargs.get('H',[])  # list of levels, each [nG,lG]: node_,link_ packed in recursion
        G.dH = kwargs.get('dH',[])  # single-fork levels from intra-link_ recursion
        G.Et = kwargs.get('Et', np.zeros(4))  # sum all params M,D,n,o
        G.baseT = kwargs.get('baseT', np.zeros(4))  # I,G,Dy,Dx  # from slice_edge
        G.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m,d / Et,baseT: [M,D,n,o, I,G,A,L], summed across derH lay forks
        G.extTT = kwargs.get('extTT', np.zeros((2,8)))  # sum across extH
        G.derH = kwargs.get('derH',[])  # each lay is [m,d]: Clay(Et,node_,link_,derTT), sum|concat links across fork tree
        G.extH = kwargs.get('extH',[])  # sum from rims, single-fork
        G.yx = kwargs.get('yx', np.zeros(2))  # init PP.yx = [(y+Y)/2,(x,X)/2], then ave node yx
        G.box = kwargs.get('box', np.array([np.inf,np.inf,-np.inf,-np.inf]))  # y,x,Y,X area: (Y-y)*(X-x)
        G.maxL = kwargs.get('maxL', 0)  # if dist-nested in cluster_N_
        G.aRad = kwargs.get('aRad', 0)  # average distance between graph center and node center
        # alt.altG is empty, more selective?
        G.altG = CG(altG=[], fi=0) if kwargs.get('altG') is None else kwargs.get('altG')  # adjacent (contour) gap+overlap alt-fork graphs, converted to CG
        # G.fork_tree: list = z([[]])  # indices in all layers(forks, if no fback merge
        # G.fback_ = []  # node fb buffer, n in fb[-1]
        G.fi = kwargs.get('fi',0)  # or fd_: list of forks forming G, 1 if cluster of Ls | lGs, for feedback only?
        G.rim = kwargs.get('rim',[])  # external links
        G.nrim = kwargs.get('nrim',[])  # needed?
        G.root = kwargs.get('root')  # may be a list in cluster_N_: same nodes in multiple dist layers
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
        l.fi = kwargs.get('fi',0)
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
    lev0, lev1, dH = [[],[]], [[],[]], [[],[]]  # two forks per level and derlay, two levs in dH
    derlay = [CLay(root=frame), CLay(root=frame)]
    for blob in blob_:
        if not blob.sign and blob.G > aveB * blob.root.olp:
            edge = slice_edge(blob, rV)
            if edge.G*((len(edge.P_)-1)*Lw) > ave * sum([P.latuple[4] for P in edge.P_]):
                Et = comp_slice(edge, rV, ww_t)  # to scale vert
                if Et[0] > ave * Et[2] * clust_w:
                    cluster_edge(edge, frame, lev0, lev1, dH, derlay)
                    # may be skipped
    G_t = [sum_N_(lev) if lev else [] for lev in lev0]
    for n_ in [G_t] + lev1 + dH:
        for n in n_:
            if n: frame.baseT+=n.baseT; frame.derTT+=n.derTT
    if G_t[0] or G_t[1]: frame.H=[G_t]  # lev0
    frame.node_ = lev1[0]; frame.link_= lev1[1]
    frame.dH = dH  # two levs
    frame.derH = [derlay]
    return frame

def cluster_edge(edge, frame, lev0, lev1, dH, derlay):  # non-recursive comp_PPm, comp_PPd, edge is not a PP cluster, unpack by default

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
            if et[0] * (len(node_)*Lw) > ave*2 * et[2] * clust_w:
                Lay = CLay(); [Lay.add_lay(link.derH[0]) for link in link_]  # single-lay derH
                G_ += [sum2graph(frame, [node_,link_,et, Lay], fi=1)]
        return G_

    def comp_PP_(PP_):
        L_,mEt,dEt, N_ = [], np.zeros(4),np.zeros(4), set()

        for _G, G in combinations(PP_, r=2):
            dy,dx = np.subtract(_G.yx,G.yx); dist = np.hypot(dy,dx)
            if dist - (G.aRad+_G.aRad) < ave_dist /10 :  # very short here
                L = comp_N(_G, G, ave, fi=1, angle=[dy, dx], dist=dist, fshort=1)
                m, d, n, o = L.Et
                if m > ave * n * loop_w: mEt += L.Et; N_.update({_G, G})  # mL_ += [L]
                if d > avd * n * loop_w: dEt += L.Et  # dL_ += [L]
                L_ += [L]
        return list(N_),L_, mEt, dEt

    for fi in 1,0:
        if (edge.Et[1-fi] *((len(edge.node_ if fi else edge.link_)-1)*(edge.rng+1) *Lw)
            > ave * edge.Et[2] * clust_w):
            N_ = [PP2G(PP, frame) for PP in (edge.node_ if fi else edge.link_)]
            PP_,L_,mEt,dEt = comp_PP_(N_)
            for l in L_:
                derlay[fi].add_lay(l.derH[0])
            G_ = []
            if val_(mEt, mEt, ave, mw=len(PP_)*Lw, aw=clust_w, fi=1) > 0:
                G_ = cluster_PP_(copy(PP_)) if mEt[0] * (len(PP_)-1)*Lw > ave * mEt[2] * clust_w else []
            if G_:
                lev1[1-fi] += G_; lev0[1-fi] += PP_  # H[0]
            else: lev1[1-fi] += PP_  # node_|link_
            if fi:  # der+, not recursive
                dH[0] += L_ # flat?
                Gd_ = []
                if val_(dEt, dEt, ave, fi=0) > 0:  # Gd_ = lG.H:
                    Gd_ = cluster_L_(frame, L2N(L_), ave, rc=2, fnodet=1).node_ if dEt[0] * (len(L_)-1) * Lw > ave * dEt[2] * clust_w else []
                if Gd_:
                    dH[1] += Gd_  # frame.dH

def val_(Et, _Et, ave, mw=1, aw=1, fi=1):  # m+d cluster | cross_comp eval, + cross|root _Et projection

    m, d, n, o = Et; _m,_d,_n,_o = _Et  # cross-fork induction of root Et alt, same overlap?
    m *= mw  # such as len*Lw

    d_loc = d * (_m - ave * aw * (_n/n))  # diff * co-projected m deviation, no bilateral deviation?
    d_ave = d - avd * ave  # d deviation, ave_d is always relative to ave m

    if fi: val = m + d_ave - d_loc  # match + proj surround val - blocking val, * decay?
    else:  val = d_ave + d_loc  # diff borrow val, generic + specific

    return val - ave * aw * n * o  # simplified: np.add(Et[:2]) > ave * np.multiply(Et[2:])

''' core process: 
 
 Cross-comp nodes, then eval cross-comp resulting > ave difference links, with recursively higher derivation,
 Cluster nodes by >ave match links, correlation-cluster links by >ave diff:
  
 Centroid cluster (CC) nodes regardless of local structure, select common node types for connectivity clustering
 Connectivity cluster (LC) exemplars of CCs by short links: less interference, with comp_N forming new derLay

 Evaluate resulting LC_: clustered node_ or link_, for recursively higher composition cross_comp and centroid clustering 

 LC is mainly generative: complexity of new links and their derivatives is greater than compression by clustering 
 CC is mainly compressive, by node similarity with same syntax, it only adds new node_ and M. 
 
 Min distance in LC is more restrictive than in cross-comp due to higher costs and density eval.
 LCs terminate at contour altGs, and next-level cross-comp is between core+contour clusters.'''

def cross_comp(root, rc, iN_, fi=1):  # rc: recursion count, fc: centroid phase, cross-comp, clustering, recursion

    N_,L_,Et = comp_node_(iN_, ave*rc) if fi else comp_link_(iN_, ave*rc)  # flat node_ or link_
    if N_:
        mL_,dL_ = [],[]
        mEt,dEt = np.zeros(4),np.zeros(4)
        for l in L_:
            if l.Et[0] > ave * l.Et[2]: mL_+= [l]; mEt += l.Et
            if l.Et[1] > avd * l.Et[2]: dL_+= [l]; dEt += l.Et
        nG, lG = [],[]
        # mfork:
        if val_(mEt,dEt, ave*(rc+2), mw=(len(mL_)-1)*Lw, aw=clust_w) > 0:  # np.add(Et[:2]) > (ave+ave_d) * np.multiply(Et[2:])?
            if fi:                                                         # cc_w if fc else lc_w? rc+=1 for all subseq ops?
                exemplars = cluster_C_(mL_,rc+2)  # form exemplar medoids, same derH, not in link_?
                if exemplars:
                    short_L_ = {l for n in exemplars for l in n.rim if l.L < ave_dist}; m,n = 0,1e-7
                    for l in short_L_: m+=l.Et[0]; n+=l.Et[2]
                    if m * ((len(short_L_)-1) *Lw) > ave * n * (rc+3) * clust_w:
                        nG = cluster_N_(root, short_L_, ave*(rc+3), rc+3)  # cluster exemplars via short rims
            else:  # N_ = dfork L_
                nG = cluster_L_(root, N_, ave*(rc+2), rc=rc+2, fnodet=0)  # no CC, via llinks, no dist-nesting
            if nG:
                if val_(nG.Et,dEt, ave*(rc+3), mw=(len(nG.node_)-1)*Lw, aw=loop_w) > 0:
                    cross_comp(nG, rc=rc+3, iN_=nG.node_)  # agglomerative recursion
        # dfork:
        dval = val_(dEt, mEt, avd*(rc+3), mw=(len(dL_)-1)*Lw, aw=clust_w, fi=0)  # mEt is alt fork
        if dval > 0:
            lG = sum_N_(L_) # not dL_: separate eval in cross_comp and cluster_L_?
            if dval > ave:  # derivational recursion
                cross_comp(lG, rc+3, L2N(dL_), fi=0)  # comp_link_, no CC, lG.H in the same lev
            else:  # lower res
                lG = cluster_L_(lG, L2N(dL_), ave*(rc+3), rc=rc+3, fnodet=1)
        if nG or lG:
            lev = []
            for g in nG,lG:
                if g: lev += [g]; add_N(root, g)  # appends root.derH
                else: lev += [[]]
            root.H += [lev] + nG.H if nG else []  # nG.H from recursion, if any

def comp_node_(_N_, ave, L=0):  # rng+ forms layer of rim and extH per N, appends N_,L_,Et, ~ graph CNN without backprop

    _Gp_ = []  # [G pair + co-positionals], for top-nested Ns, unless cross-nesting comp:
    if L: _N_ = filter(lambda N: len(N.derH)==L, _N_)
    # max len derH only
    for _G, G in combinations(_N_, r=2):
        if len(_G.H) != len(G.H):  # | root.H: comp top nodes only?
            continue
        _n, n = _G.Et[2], G.Et[2]; rn = _n/n if _n>n else n/_n
        radii = G.aRad + _G.aRad
        dy,dx = np.subtract(_G.yx,G.yx); dist = np.hypot(dy,dx)
        _G.add, G.add = 0, 0
        _Gp_ += [(_G,G, rn, dy,dx, radii, dist)]
    rng = 1
    N_,L_,ET = set(),[],np.zeros(4)
    _Gp_ = sorted(_Gp_, key=lambda x: x[-1])  # sort by dist, closest pairs first
    while True:  # prior vM
        Gp_,Et = [],np.zeros(4)
        for Gp in _Gp_:
            _G,G, rn, dy,dx, radii, dist = Gp
            _nrim = {L.nodet[1] if L.nodet[0] is _G else L.nodet[0] for L,_ in _G.rim}
            nrim = {L.nodet[1] if L.nodet[0] is G else L.nodet[0] for L,_ in G.rim}
            if _nrim & nrim:  # indirectly connected Gs,
                continue     # no direct match priority?
            # dist vs. radii * induction, mainly / extH?
            (_m,_,_n,_),(m,_,n,_) = _G.Et,G.Et
            weighted_max = ave_dist * ((radii/aveR * int_w**3) * (_m/_n + m/n)/2 / (ave*(_n+n)))  # all ratios
            if dist < weighted_max:   # no density, ext V is not complete
                Link = comp_N(_G,G, ave, fi=1, angle=[dy,dx], dist=dist, fshort = dist < weighted_max/2)
                L_ += [Link]  # include -ve links
                if Link.Et[0] > ave * Link.Et[2] * loop_w:
                    N_.update({_G,G}); Et += Link.Et; _G.add,G.add = 1,1
            else:
                Gp_ += [Gp]  # re-evaluate not-compared pairs with one incremented N.M
        ET += Et
        if Et[0] > ave * Et[0] * loop_w:  # current-rng vM
            _Gp_ = [Gp for Gp in Gp_ if Gp[0].add or Gp[1].add]  # one incremented N.M
            rng += 1
        else:  # low projected rng+ vM
            break

    return  list(N_), L_, ET  # flat N__ and L__

def comp_link_(iL_, ave):  # comp CLs via directional node-mediated link tracing: der+'rng+ in root.link_ rim_t node rims

    fi = isinstance(iL_[0].nodet[0], CG)
    for L in iL_:
        # init mL_t: bilateral mediated Ls per L:
        for rev, N, mL_ in zip((0,1), L.nodet, L.mL_t):
            for _L,_rev in N.rim if fi else N.rimt[0]+N.rimt[1]:
                if _L is not L and _L in iL_:  # nodet-mediated
                    if L.Et[0] > ave * L.Et[2] * loop_w:
                        mL_ += [(_L, rev ^_rev)]  # direction of L relative to _L
    _L_, out_L_, LL_, ET = iL_,set(),[],np.zeros(4)  # out_L_: positive subset of iL_, Et = np.zeros(4)?
    med = 1
    while True:  # xcomp _L_
        L_, Et = set(), np.zeros(4)
        for L in _L_:
            for mL_ in L.mL_t:
                for _L, rev in mL_:  # rev is relative to L
                    dy,dx = np.subtract(_L.yx,L.yx)
                    Link = comp_N(_L,L, ave, fi=0, angle=[dy,dx], dist=np.hypot(dy,dx), dir = -1 if rev else 1)  # d = -d if L is reversed relative to _L
                    Link.med = med
                    LL_ += [Link]  # include -ves, link order: nodet < L < rimt, mN.rim || L
                    if Link.Et[0] > ave * Link.Et[2] * loop_w:  # link induction
                        out_L_.update({_L,L}); L_.update({_L,L}); Et += Link.Et
        ET += Et
        # extend mL_t per last medL:
        if Et[0] > ave * Et[2] * (loop_w + med*med_w):  # project prior-loop value, med adds fixed cost
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
            if ext_Et[0] > ave * ext_Et[2] * (loop_w + med*med_w):
                med +=1
            else: break
        else: break
    return list(out_L_), LL_, ET

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
        _y0,_x0,_yn,_xn =_N.box; _A = (_yn-_y0) * (_xn-_x0)
        y0, x0, yn, xn = N.box;   A = (yn - y0) * (xn - x0)
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
    if fi and _N.altG and N.altG:
        et = _N.altG.Et + N.altG.Et  # comb val
        if val_(et, et, ave*2, fi=0) > 0:  # eval Ds
            Link.altL = comp_N(_N.altG, N.altG, ave*2, fi=1, angle=angle)
            Et += Link.altL.Et
    Link.Et = Et
    if Et[0] > ave * Et[2]:  # | both forks: np.add(Et[:2]) > ave * np.multiply(Et[2:])
        for rev, node in zip((0,1), (N,_N)):  # reverse Link direction for
            if fi: node.rim += [(Link,dir)]
            else:  node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            add_H(node.extH, Link.derH, root=node, rev=rev, fi=0)
            node.Et += Et
    return Link

def cluster_N_(root, L_, ave, rc):  # top-down segment L_ by >ave ratio of L.dists

    nG = CG(root=root)
    L_ = sorted(L_, key=lambda x: x.L)  # short links first
    min_dist = 0; Et = root.Et
    while True:
        # each loop forms G_ of L_ segment with contiguous distance values: L.L
        _L = L_[0]; N_, et = copy(_L.nodet), _L.Et
        for n in [n for l in L_ for n in l.nodet]:
            n.fin = 0
        for i, L in enumerate(L_[1:], start=1):
            rel_dist = L.L/_L.L  # >= 1
            if rel_dist < 1.2:  # no Et[0]*((len(L_[i:])-1)*Lw) < ave*Et[2]*loop_w: termination anyway, stop dist+ if weak?
                # cluster eval with surround density: (_Ete[0]+Ete[0])/2 / ave:
                _G,G = L.nodet; surr_V = (sum(_G.extTT[0]) + sum(G.extTT[0])/2) / (ave * G.Et[2])
                if surr_V * (L.Et[0]/L.Et[2]-ave) > ave:
                    _L = L; N_ += L.nodet; et += L.Et  # else skip weak link inside segment
            else:
                i -= 1; break  # terminate contiguous-distance segment
        max_dist = _L.L
        G_ = []  # cluster current distance segment N_:
        if val_(et, Et, ave*clust_w, aw=(len(N_)-1)*Lw) > 0:
            for N in {*N_}:
                if N.fin: continue  # clustered from prior _N_
                _eN_,node_,link_,et, = [N],[],[], np.zeros(4)
                while _eN_:
                    eN_ = []
                    for eN in _eN_:  # cluster rim-connected ext Ns, all in root Gt
                        node_+=[eN]; eN.fin = 1  # all rim
                        for L,_ in eN.rim:  # all +ve
                            if L not in link_:
                                eN_ += [n for n in L.nodet if not n.fin]
                                if L.L < max_dist:
                                    link_+=[L]; et+=L.Et
                    _eN_ = {*eN_}
                # form Gt:
                link_ = list({*link_})
                Lay = CLay(); [Lay.add_lay(lay) for lay in sum_H(link_, nG, fi=0)]
                derTT = Lay.derTT
                # weigh m_|d_ by similarity to mean m|d, weigh derTT:
                m_,M = centroid_M(derTT[0], ave=ave); d_,D = centroid_M(derTT[1], ave=ave)
                et[:2] = M,D; Lay.derTT = np.array([m_,d_])
                # cluster roots:
                if val_(et, Et, ave*clust_w) > 0:
                    G_ += [sum2graph(nG, [list({*node_}),link_, et, Lay], 1, min_dist, max_dist)]
            else:
                G_ += N_  # unclustered nodes
        # longer links:
        L_ = L_[i + 1:]
        if G_:
            [comb_altG_(G.altG.H, ave, rc) for G in G_ if G.altG.H]  # not nested in higher-dist Gs, but nodes have all-dist roots
        if L_:
            min_dist = max_dist  # next loop connects current-distance clusters via longer links
        else:
            break
    if G_:
        return sum_N_(G_, root_G=nG)  # highest dist segment, includes all nodes

def cluster_L_(root, L_, ave, rc, fnodet=1):  # CC links via nodet or rimt, no dist-nesting

    lG = CG(root=root)
    G_ = []  # flood-fill link clusters
    for L in L_: L.fin = 0
    for L in L_:
        if L.fin: continue
        L.fin = 1
        node_, link_, Et, Lay = [L], [], copy(L.Et), CLay()
        if fnodet:
            for _L in L.nodet[0].rim+L.nodet[1].rim if L.nodet[0].fi else [l for n in L.nodet for l,_ in n.rimt[0]+n.rimt[1]]:
                if _L in L_ and not _L.fin and _L.Et[1] > avd * _L.Et[2]:  # direct cluster by dval
                    link_ += L.nodet; Et += _L.Et; _L.fin = 1; node_ += [_L]
        else:
            for lL,_ in L.rimt[0] + L.rimt[1]:
                _L = lL.nodet[0] if lL.nodet[1] is L else lL.nodet[1]
                if _L in L_ and not _L.fin:  # +ve only, eval by directional density?
                    link_ += [lL]
                    Et += lL.Et; _L.fin = 1; node_ += [_L]
        if val_(Et, root.Et, ave*rc, mw=(len(node_)-1)*Lw, aw=clust_w) > 0:
            Lay = CLay()
            [Lay.add_lay(l) for l in sum_H(node_ if fnodet else link_, root=lG, fi=0)]
            G_ += [sum2graph(lG, [list({*node_}), link_, Et, Lay], fi=0)]
    if G_:
        [comb_altG_(G.altG.H, ave, rc) for G in G_]
        sum_N_(G_, root_G = lG)
        return lG

def cluster_C_(L_, rc):  # select medoids for LC, next cross_comp

    def cluster_C(N, M, medoid_):  # compute C per new medoid, compare C-node pairs, recompute C if max match is not medoid

        node_,_n_, med = [], [N], 1
        while med <= ave_med and _n_:  # node_ is _Ns connected to N by <=3 mediation degrees
            n_ = [n for _n in _n_ for link,_ in _n.rim for n in link.nodet  # +ve Ls
                  if med >1 or not n.C]  # skip directly connected medoids
            node_ += list(set(n_)); _n_ = n_; med += 1
        C = sum_N_(list(set(node_)))  # default
        k = len(node_)
        for n in (C, C.altG): n.Et /= k; n.baseT /= k; n.derTT /= k; n.aRad /= k; n.yx /= k; norm_H(n.derH, k)
        maxM, m_ = 0,[]
        for n in node_:
            m = sum( base_comp(C,n)[0][0])  # derTT[0][0]
            if C.altG and N.altG: m += sum( base_comp(C.altG,N.altG)[0][0])
            m_ += [m]
            if m > maxM: maxN = n; maxM = m
        if maxN.C:
            return  # or compare m to mroott_'C'm and replace with local C if stronger?
        if N.mroott_:
            for _medoid,_m in N.mroott_:
                rel_olp = len(set(C.node_) & set(_medoid.C.node_)) / (len(C.node_) + len(_medoid.C.node_))
                if _m > m:
                    if m > ave * rel_olp:  # add medoid
                        for _N, __m in zip(C.node_, m_): _N.mroott_ += [[maxN, __m]]  # _N match to C
                        maxN.C = C; medoid_ += [maxN]; M += m
                    elif _m < ave * rel_olp:
                        medoid_.C = None; medoid_.remove(_medoid); M += _m  # abs change? remove mroots?
        elif m > ave:
            N.mroott_ = [[maxN, m]]; maxN.C = C; medoid_ += [maxN]; M += m

    N_ = list(set([node for link in L_ for node in link.nodet]))
    ave = globals()['ave'] * rc
    medoid__ = []
    for N in N_:
        N.C = None; N.mroott_ = []  # init
    while N_:
        M, medoid_ = 0, []
        N_ = sorted(N_, key=lambda n: n.Et[0]/n.Et[2], reverse=True)  # strong nodes initialize centroids
        for N in N_:
            if N.Et[0] > ave * N.Et[2]:
                cluster_C(N, M, medoid_)
            else:
                break  # rest of N_ is weaker
        medoid__ += medoid_
        if M < ave * len(medoid_):
            break  # converged
        else:
            N_ = medoid_

    return {n for N in medoid__ for n in N.C.node_ if sum([Ct[1] for Ct in n.mroott_]) > ave * clust_w}  # exemplars

def layer_C_(root, L_, rc):  # node-parallel cluster_C_ in mediation layers, prune Cs in top layer?
    # same nodes on all layers, hidden layers mediate links up and down, don't sum or comp anything?
    pass

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
    N_, n_,l_,yx_ = [],[],[],[]
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
            if fg and N.H: add_node_H(graph.H, N.H)
        if fg: n_ += N.node_; l_ += N.link_
    if fg: graph.H += [[sum_N_(n_), sum_N_(l_)]]  # pack prior top level, current-dist link_ only?
    graph.node_ = N_
    yx = np.mean(yx_, axis=0)
    dy_,dx_ = (graph.yx - yx_).T; dist_ = np.hypot(dy_,dx_)
    graph.aRad = dist_.mean()  # ave distance from graph center to node centers
    graph.yx = yx
    if not fi:  # dgraph, no mGs / dG for now  # and val_(Et, _Et=root.Et) > 0:
        altG = []  # mGs overlapping dG
        for L in node_:
            for n in L.nodet:  # map root mG
                mG = n.root
                if isinstance(mG, CG) and mG not in altG:  # root is not frame
                    mG.altG.node_ += [graph]  # cross-comp|sum complete altG before next agg+ cross-comp, multi-layered?
                    altG += [mG]
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

def comb_altG_(G_, ave, rc=1):  # combine contour G.altG_ into altG (node_ defined by root=G),

    # internal vs. external alts: different decay / distance, background + contour?
    for G in G_:
        if G.altG:
            if G.altG.H:
                G.altG = sum_N_(G.altG.H)
                G.altG.root=G; G.altG.m=0
                if val_(G.altG.Et, G.Et, ave, fi=0):  # alt D * G rM
                    cross_comp(G.altG, rc, G.altG.H, fi=1)  # adds nesting
        elif G.H:
            # G is not PP, altG = sum dlinks
            dL_ = list(set([L for g in G.node_ for L,_ in (g.rim if isinstance(g, CG) else g.rimt[0]+g.rimt[1]) if val_(L.Et,G.Et, ave, fi=0) > 0]))
            if dL_ and val_(np.sum([l.Et for l in dL_], axis=0), G.Et, ave, aw=10, fi=0) > 0:
                altG = sum_N_(dL_)
                G.altG = copy_(altG); G.altG.H = [altG]; G.altG.root=G

def comb_H_(L_, root, fi):
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
                        root.derTT += fork.derTT; root.Et += fork.Et
                    H += [Lay]
            else:  # one-fork lays
                if Lay: Lay.add_lay(lay,rev=rev)
                else:   H += [lay.copy_(root=root,rev=rev)]
                root.extTT += lay.derTT; root.Et += lay.Et

def sum_N_(node_, root_G=None, root=None):  # form cluster G

    fi = isinstance(node_[0],CG)
    if root_G is not None: G = root_G
    else:
        G = copy_(node_[0], init=1, root=root); G.fi=fi
    for n in node_:
        add_N(G,n, fi, fappend=1)
        if root: n.root=root
    if not fi:
        G.derH = [[lay] for lay in G.derH]  # nest
    return G

def add_N(N,n, fi=1, fappend=0):

    N.baseT+=n.baseT; N.derTT+=n.derTT; N.Et+=n.Et
    N.yx+=n.yx; N.box=extend_box(N.box, n.box)
    if isinstance(n,CG) and n.altG:  # not CL
        add_N(N.altG, n.altG)
    if fappend:
        N.node_ += [n]
        if fi: N.link_ += [n.link_]  # splice if CG
    elif fi:  # empty in append and altG
        if n.H: add_node_H(N.H, n.H)
        if n.dH: add_node_H(N.dH, n.dH)
    if n.derH:
        add_H(N.derH, n.derH, root=N, fi=fi)
    if hasattr(n,'extTT'):  # node, = fi?
        N.extTT += n.extTT; N.aRad += n.aRad
        if n.extH:
            add_H(N.extH, n.extH, root=N, fi=0)
    return N

def add_node_H(H,h):

    for Lev, lev in zip(H, h):  # always aligned?
        for F, f in zip_longest(Lev, lev, fillvalue=None):
            if f:
                if F: add_N(F,f)  # nG|lG
                else: Lev += [f]  # if lG

def extend_box(_box, box):  # extend box with another box
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box

    return min(y0, _y0), min(x0, _x0), max(yn, _yn), max(xn, _xn)

def L2N(link_):
    for L in link_:
        L.fi=0; L.mL_t,L.rimt = [[],[]],[[],[]]; L.aRad=0; L.extTT=np.zeros((2,8)); L.visited_,L.extH,L.node_,L.link_,L.H,L.dH = [],[],[],[],[],[]
        if not hasattr(L,'root'): L.root=[]
    return link_

def PP2G(PP, frame):
    P_, link_, vert, latuple, A, S, box, yx, Et = PP
    baseT = np.array((*latuple[:2], *latuple[-1]))  # I,G,Dy,Dx
    [mM,mD,mI,mG,mA,mL], [dM,dD,dI,dG,dA,dL] = vert
    derTT = np.array([[mM,mD,mL,0,mI,mG,mA,mL], [dM,dD,dL,0,dI,dG,dA,dL]])
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
            comb_altG_(edge.node_, ave)
            cluster_C_(edge, ave)  # no cluster_C_ in vect_edge
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
        comb_altG_(frame.node_, ave*2)
        cross_comp(frame, rc=1, iN_=frame.node_)  # top level
        # adjust weights:
        rM, rD, rv_t = feedback(frame)
        if (rM+rD) * val_(frame.Et,frame.Et, ave) > ave * clust_w * 20:  # normalized?
            nG = frame.H[0][0]  # base PP_ focus shift by dval + temp Dm_+Ddm_?
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
    hlG = sum_N_(root.link_+ root.dH[1])  # not sure about root.dH (so dH has dL_ and Gd_, i think we need to select the higher Gd_ only?)
    for lev in reversed(root.H):
        lG = lev[1]  # eval link_: comp results per level?
        _m,_d,_n,_ = hlG.Et; m,d,n,_ = lG.Et
        rM += (_m/_n) / (m/n)  # no o eval?
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