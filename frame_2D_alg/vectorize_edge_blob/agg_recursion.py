import sys
sys.path.append("..")
from frame_blobs import CBase, imread
from slice_edge import comp_angle, CsliceEdge
from comp_slice import CcompSlice, comp_slice, comp_latuple, add_lat, aves, comp_md_, add_md_
from itertools import combinations, zip_longest
from copy import deepcopy, copy
import numpy as np
'''
This code is ostensibly for clustering segments within edge: high-gradient blob, but it's far too complex for the case.
That's because this is a prototype for open-ended compositional recursion: clustering blobs, graphs of blobs, etc.
We will later prune it down to lighter edge-specific version.
-
Primary incremental-range (rng+) fork cross-comp leads to clustering edge segments, initially PPs, that match over < max distance. 
Secondary incr-derivation (der+) fork cross-compares links from primary cross-comp, if >ave ~(abs_diff * primary_xcomp_match): 
variance patterns borrow value from co-projected match patterns, because their projections cancel-out.
- 
Thus graphs should be assigned adjacent alt-fork (der+ to rng+) graphs, to which they lend predictive value.
But alt match patterns borrow already borrowed value, which are too tenuous to track, we use average borrowed value.
Clustering criterion within each fork is summed match of >ave vars (<ave vars are not compared and don't add comp costs).
-
Clustering is exclusive per fork,ave, with fork selected per variable | derLay | aggLay 
Fuzzy clustering can only be centroid-based, because overlapping connectivity-based clusters will merge.
Param clustering if MM, compared along derivation sequence, or combinatorial?
-
Summed-graph representation is nested in a dual tree of down-forking elements: node_, and up-forking clusters: root_.
That resembles a neuron, which has dendritic tree as input and axonal tree as output. 
But we have recursively nested param sets packed in each level of the trees, which don't exist in neurons.
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
ave = 3
ave_d = 4
ave_L = 4
max_dist = 2
ave_rn = 1000  # max scope disparity
ccoef  = 10  # scaling match ave to clustering ave

class CH(CBase):  # generic derivation hierarchy of variable nesting: extH | derH, their layers and sub-layers

    name = "H"
    def __init__(He, node_=None, md_t=None, n=0, Et=None, H=None, root=None, i=None, i_=None):
        super().__init__()
        He.node_ = [] if node_ is None else node_  # concat bottom nesting order if CG, may be redundant to G.node_
        He.md_t = [] if md_t is None else md_t  # derivation layer in H: [mdlat,mdLay,mdext]
        He.H = [] if H is None else H  # nested derLays | md_ in md_C, empty in bottom layer
        He.n = n  # total number of params compared to form derH, to normalize comparands
        He.Et = np.array([.0,.0,.0,.0]) if Et is None else Et  # evaluation tuple: valt, rdnt
        He.root = None if root is None else root  # N or higher-composition He
        He.i = 0 if i is None else i  # lay index in root.H, to revise rdn
        He.i_ = [] if i_ is None else i_  # priority indices to compare node H by m | link H by d
        # He.ni = 0  # exemplar in node_, trace in both directions?
        # He.depth = 0  # nesting in H[0], -=i in H[Hi], in agg++? same as:
        # He.nest = nest  # nesting depth: -1/ ext, 0/ md_, 1/ derH, 2/ subH, 3/ aggH?

    def __bool__(H): return H.n != 0

    def accum_lay(HE, He, irdnt):

        if HE.md_t:
            for MD_, md_ in zip(HE.md_t, He.md_t):  # dext_, dlat_, dlay_
                add_md_(MD_, md_)
            HE.n += He.n
            HE.Et += He.Et
            if any(irdnt):
                HE.Et[2:] = [E+e for E,e in zip(HE.Et[2:], irdnt)]
        else:
            HE.md_t = deepcopy(He.md_t)
        HE.n += He.n  # combined param accumulation span
        HE.Et += He.Et
        if any(irdnt):
            HE.Et[2:] = [E+e for E,e in zip(HE.Et[2:], irdnt)]

    def add_H(HE, He, irdnt=[]):  # unpack derHs down to numericals and sum them

        if HE:
            for i, (Lay,lay) in enumerate(zip_longest(HE.H, He.H, fillvalue=None)):  # cross comp layer
                if lay:
                    if Lay: Lay.add_H(lay, irdnt)
                    else:
                        if Lay is None:
                            HE.append_(CH().copy(lay))  # pack a copy of new lay in HE.H
                        else:
                            HE.H[i] = CH(root=HE).copy(lay)  # Lay was []
            HE.accum_lay(He, irdnt)
            HE.node_ += [node for node in He.node_ if node not in HE.node_]  # node_ is empty in CL derH?
        else:
            HE.copy(He)  # init

        return HE.update_root(He)  # feedback, ideally buffered from all elements before summing in root, ultimately G|L

    def append_(HE,He, irdnt=None, flat=0):

        if irdnt is None: irdnt = []
        if flat:
            for i, lay in enumerate(He.H):  # different refs for L.derH and root.derH.H:
                lay = CH().copy(lay)
                lay.i = len(HE.H)+i; lay.root = HE; HE.H += [lay]
        else:
            He = CH().copy(He); He.i = len(HE.H); He.root = HE; HE.H += [He]

        HE.accum_lay(He, irdnt)
        return HE.update_root(He)

    def update_root(HE, He):  # should be batched or done in the calling function, this recursive version is a placeholder

        root = HE.root
        while root is not None:
            if isinstance(root, CH):
                root.Et += He.Et
                root.node_ += [node for node in He.node_ if node not in HE.node_]
                root.n += He.n
                root = root.root
            else:
               break  # root is G|L
        return HE

    def comp_H(_He, He, rn=1, dir=1):  # unpack each layer of CH down to numericals and compare each pair

        der_md_t = []; Et = np.array([.0,.0,.0,.0])
        for _md_, md_ in zip(_He.md_t, He.md_t):  # [mdlat, mdLay, mdext], default per layer

            der_md_ = comp_md_(_md_[0], md_[0], rn=1, dir=dir)
            der_md_t += [der_md_]
            Et += der_md_[1]

        DLay = CH(md_t = der_md_t, Et=Et, n=2.3)  # .3 added from comp ext, empty node_: no comp node_
        # comp,unpack H, empty in bottom or deprecated layer:

        for _lay, lay in zip(_He.H, He.H):  # sublay CH per rng | der, flat
            if _lay and lay:
                dLay = _lay.comp_H(lay, rn, dir)  # comp He.md_t, comp,unpack lay.H
                DLay.append_(dLay, flat=0)  # DLay.H += subLay
                # nested subHH ( subH?
        return DLay

    # not implemented:
    def sort_H(He, fd):  # re-assign rdn and form priority indices for comp_H, if selective and aligned

        i_ = []  # priority indices
        for i, lay in enumerate(sorted(He.H, key=lambda lay: lay.Et[fd], reverse=True)):
            di = lay.i - i  # lay index in H
            lay.Et[2+fd] += di  # derR- valR
            i_ += [lay.i]
        He.i_ = i_  # comp_H priority indices: node/m | link/d
        if not fd:
            He.root.node_ = He.H[i_[0]].node_  # no He.node_ in CL?

    def copy(_He, He):
        for attr, value in He.__dict__.items():
            if attr != '_id' and attr != 'root' and attr in _He.__dict__.keys():  # copy attributes, skip id, root
                if attr == 'H':
                    if He.H:
                        _He.H = []
                        if isinstance(He.H[0], CH):
                            for lay in He.H: _He.H += [CH().copy(lay)]  # can't deepcopy CH.root
                        else: _He.H = deepcopy(He.H)  # md_
                elif attr == "node_":
                    _He.node_ = copy(He.node_)
                else:
                    setattr(_He, attr, deepcopy(value))
        return _He

class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster

    def __init__(G, root_= None, node_=None, link_=None, latuple=None, mdLay=None, derH=None, extH=None, rng=1, fd=0, n=0, box=None, yx=None):
        super().__init__()
        G.n = n  # last layer?
        G.fd = 0 if fd else fd  # 1 if cluster of Ls | lGs?
        G.rng = rng
        G.root_ = [] if root_ is None else root_ # same nodes in higher rng layers
        G.node_ = [] if node_ is None else node_ # convert to GG_ in agg++
        G.link_ = [] if link_ is None else link_ # internal links per comp layer in rng+, convert to LG_ in agg++
        G.latuple = [0,0,0,0,0,[0,0]] if latuple is None else latuple  # lateral I,G,M,Ma,L,[Dy,Dx]
        G.mdLay = [[.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0], [.0,.0,.0,.0], 0] if mdLay is None else mdLay  # H here is md_latuple
        # maps to node_H / agg+|sub+:
        G.derH = CH(root=G) if derH is None else derH  # sum from nodes, then append from feedback
        G.extH = CH(root=G) if extH is None else extH  # sum from rim_ elays, H maybe deleted
        G.rim = []  # flat links of any rng, only nested in clustering
        G.aRad = 0  # average distance between graph center and node center
        G.box = [np.inf, np.inf, -np.inf, -np.inf] if box is None else box  # y0,x0,yn,xn
        G.yx = [0,0] if yx is None else yx  # init PP.yx = [(y0+yn)/2,(x0,xn)/2], then ave node yx
        G.alt_graph_ = []  # adjacent gap+overlap graphs, vs. contour in frame_graphs
        G.visited_ = []
        # G.Et = [0,0,0,0] if Et is None else Et   # derH.Et + extH.Et?
        # G.fback_ = []  # always from CGs with fork merging, no dderHm_, dderHd_
        # G.Rim = []  # links to the most mediated nodes
        # depth: int = 0  # n sub_G levels over base node_, max across forks
        # nval: int = 0  # of open links: base alt rep
        # id_H: list = z([[]])  # indices in the list of all possible layers | forks, not used with fback merging
        # top aggLay: derH from links, lower aggH from nodes, only top Lay in derG:
        # top Lay from links, lower Lays from nodes, hence nested tuple?
    def __bool__(G): return G.n != 0  # to test empty

class CL(CBase):  # link or edge, a product of comparison between two nodes or links
    name = "link"

    def __init__(l, nodet=None, dist=None, derH=None, angle=None, box=None, H_=None):
        super().__init__()
        # CL = binary tree of Gs, depth+/der+: CL nodet is 2 Gs, CL + CLs in nodet is 4 Gs, etc., unpack sequentially
        l.n = 1  # min(node_.n)
        l.derH = CH(root=l) if derH is None else derH
        l.nodet = [] if nodet is None else nodet  # e_ in kernels, else replaces _node,node: not used in kernels
        l.angle = [0,0] if angle is None else angle  # dy,dx between nodet centers
        l.dist = 0 if dist is None else dist  # distance between nodet centers
        l.box = [] if box is None else box  # sum nodet, not needed?
        l.Vt = [0,0]  # for rim-overlap modulated segmentation, init derH.Et[:2]
        l.H_ = [] if H_ is None else H_  # if agg++| sub++?
        # add rimt, elay | extH in der+
    def __bool__(l): return bool(l.derH.H)

def vectorize_root(image):  # vectorization in 3 composition levels of xcomp, cluster:

    frame = CsliceEdge(image).segment()
    for edge in frame.blob_:
        if (hasattr(edge, 'P_') and
            edge.latuple[-1] * (len(edge.P_)-1) > ave):
            comp_slice(edge)
            # init for agg+:
            edge.mdLay = CH(H=edge.mdLay[0], Et=edge.mdLay[1], n=edge.mdLay[2])
            edge.derH = CH(H=[CH()]); edge.derH.H[0].root = edge.derH; edge.fback_ = []
            if edge.mdLay.Et[0] * (len(edge.node_)-1)*(edge.rng+1) > ave * edge.mdLay.Et[2]:
                G_ = []
                for N in edge.node_:  # no comp node_, link_ | PPd_ for now
                    H,Et,n = N[3] if isinstance(N,list) else N.mdLay  # N is CP
                    if H and Et[0] > ave * Et[2]:  # convert PP|P to G:
                        if isinstance(N,list):
                            root, P_,link_,(H,Et,n),lat,A,S,area,box,[y,x],n = N  # PPt
                        else:  # single CP
                            root=edge; P_=[N]; link_=[]; (H,Et,n)=N.mdLay; lat=N.latuple; [y,x]=N.yx; n=N.n
                            box = [y,x-len(N.dert_), y,x]
                        PP = CG(fd=0, root_=root, node_=P_,link_=link_,mdLay=[H, Et, n],latuple=lat, box=box,yx=[y,x],n=n)
                        y0,x0,yn,xn = box
                        PP.aRad = np.hypot(*np.subtract(PP.yx,(yn,xn)))
                        G_ += [PP]
                if len(G_) > 10:
                    agg_recursion(edge, G_, fd=0)  # discontinuous PP_ xcomp, cluster

def agg_recursion(root, iQ, fd):  # breadth-first rng+ cross-comp -> eval clustering, recursion per fd fork: rng+ | der+

    Q = []
    for e in iQ:
        if isinstance(e, list): continue  # skip Gts: weak
        e.root_, e.extH, e.merged = [], CH(), 0
        Q += [e]
    # cross-comp link_ or node_:
    N_, L_, Et, rng = comp_link_(Q) if fd else comp_node_(Q)
    m, d, mr, dr = Et
    fvd = d > ave_d * dr*(rng+1)
    fvm = m > ave * mr*(rng+1)
    if fvd or fvm:
        L = L_[0]   # root += L.derH, before clustering:
        if fd: root.derH.append_(CH().append_(CH().copy(L.derH)))  # always new rngLay, aggLay
        else: root.derH.H[-1].append_(L.derH)  # append last aggLay
        for L in L_[1:]:
            root.derH.H[-1].H[-1].add_H(L.derH)  # accum Lay
        # comp_link_
        if fvd and len(L_) > ave_L:  # comp L if DL, sub-cluster LLs by mL:
            agg_recursion(root, L_, fd=1)  # appends last aggLay, L_ = lG_
        if fvm:
            G_ = cluster_N_(root, N_, fd)  # connectivity divisive clustering
            if len(G_) > ave_L:  # root.node_[:] = nested G_, if any:
                agg_recursion(root, G_, fd=0)  # comp_G_-> GG_
'''
     if flat derH:
        root.derH.append_(CH().copy(L_[0].derH))  # init
        for L in L_[1:]: root.derH.H[-1].add_H(L.derH)  # accum
'''

def comp_node_(_N_):  # rng+ forms layer of rim and extH per N, appends N_,L_,Et, ~ graph CNN without backprop

    _Gp_ = []  # [G pair + co-positionals]
    for _G, G in combinations(_N_, r=2):
        rn = _G.n / G.n
        if rn > ave_rn: continue  # scope disparity
        radii = G.aRad + _G.aRad
        dy,dx = np.subtract(_G.yx,G.yx)
        dist = np.hypot(dy,dx)
        _G.add, G.add = 0, 0
        _Gp_ += [(_G,G, rn, dy,dx, radii, dist)]
    icoef = .5  # internal M proj_val / external M proj_val
    rng = 1  # len N__
    N_, L_, ET = [],[], np.array([.0,.0,.0,.0])
    while True:  # prior vM
        Gp_,Et = [], np.array([.0,.0,.0,.0])
        for Gp in _Gp_:
            _G,G, rn, dy,dx, radii, dist = Gp
            _nrim = set([(Lt[0].nodet[1] if Lt[0].nodet[0] is _G else Lt[0].nodet[0]) for rim in _G.rim_ for Lt in rim])
            nrim = set([(Lt[0].nodet[1] if Lt[0].nodet[0] is G else Lt[0].nodet[0]) for rim in G.rim_ for Lt in rim])
            if _nrim & nrim:  # indirectly connected Gs,
                continue     # no direct match priority?
            M = (_G.mdLay[1][0]+G.mdLay[1][0]) *icoef**2 + (_G.derH.Et[0]+G.derH.Et[0])*icoef + (_G.extH.Et[0]+G.extH.Et[0])
            # comp if < max distance of likely matches *= prior G match * radius:
            if dist < max_dist * (radii * icoef**3) * M:
                Link = CL(nodet=[_G,G], angle=[dy,dx], dist=dist, box=extend_box(G.box,_G.box))
                et = comp_N(Link, rn, rng)
                L_ += [Link]  # include -ve links
                if et is not None:
                    N_ += [_G, G]
                    Et += et; _G.add,G.add = 1,1
            else:
                Gp_ += [Gp]  # re-evaluate not-compared pairs with one incremented N.M
        ET += Et
        if Et[0] > ave * Et[2]:  # current-rng vM
            rng += 1
            _Gp_ = [Gp for Gp in Gp_ if Gp[0].add or Gp[1].add]  # one incremented N.M
        else:  # low projected rng+ vM
            break

    return  N_,L_, ET, rng  # flat N__ and L__


def comp_link_(iL_):  # comp CLs via directional node-mediated link tracing: der+'rng+ in root.link_ rim_t node rims

    fd = isinstance(iL_[0].nodet[0], CL)
    for L in iL_:
        L.mL_t, L.rimt, L.aRad, L.visited_ = [[],[]], [], 0, [L]
        # init mL_t (mediated Ls):
        for rev, n, mL_ in zip((0,1), L.nodet, L.mL_t):
            rim = n.rimt if fd else n.rim
            for _L,_rev in rim[0] + rim[1] if fd else rim:
                if _L is not L and _L.derH.Et[0] > ave * _L.derH.Et[2]:
                    mL_ += [(_L, rev ^_rev)]  # direction of L relative to _L
    _L_, L_, LL_, ET = iL_,[],[], np.array([.0,.0,.0,.0])
    med = 1
    while True:  # xcomp _L_
        L_, LL_, Et = [],[], np.array([.0,.0,.0,.0])
        for L in _L_:
            for mL_ in L.mL_t:
                for _L, rev in mL_:  # rev is relative to L
                    rn = _L.n / L.n
                    if rn > ave_rn: continue  # scope disparity
                    dy,dx = np.subtract(_L.yx, L.yx)
                    Link = CL(nodet=[_L,L], angle=[dy,dx], dist=np.hypot(dy,dx), box=extend_box(_L.box, L.box))
                    # comp L,_L:
                    et = comp_N(Link, rn, rng=med, dir = -1 if rev else 1)  # d = -d if L is reversed relative to _L
                    LL_ += [Link]  # include -ves, link order: nodet < L < rimt, mN.rim || L
                    if et is not None:
                        L_ += [_L,L]; Et += et
        if not any(L_): break
        ET += Et  # rng+ eval:
        Med = med + 1
        if Et[0] > ave * Et[2] * Med:  # project prior-loop value - new cost
            nxt_L_, nxt_Et = set(), np.array([.0,.0,.0,.0])
            for L in L_:
                mL_t, lEt = [set(),set()], np.array([.0,.0,.0,.0])  # __Ls per L
                for mL_,_mL_ in zip(mL_t, L.mL_t):
                    for _L, rev in _mL_:
                        for _rev, n in zip((0,1), _L.nodet):
                            rim_ = n.rimt_ if fd else n.rim_
                            if len(rim_) == med:  # append in comp loop
                                for __L,__rev in rim_[-1][0]+rim_[-1][1] if fd else rim_[-1]:
                                    if __L in L.visited_ or __L not in iL_: continue
                                    L.visited_ += [__L]; __L.visited_ += [L]
                                    et = __L.derH.Et
                                    if et[0] > ave * et[2] * Med:  # /__L
                                        mL_.add((__L, rev ^_rev ^__rev))  # combine reversals: 2 * 2 mNs, 1st 2 are pre-combined
                                        lEt += et
                if lEt[0] > ave * lEt[2] * Med:  # rng+/ L is different from comp/ L above
                    L.mL_t = mL_t; nxt_L_.add(L); nxt_Et += lEt
            # refine eval:
            if nxt_Et[0] > ave * nxt_Et[2] * Med:
                _L_ = nxt_L_; med = Med
            else:
                break
        else:
            break
    return L_, LL_, ET, med  # =rng

def extend_box(_box, box):  # add 2 boxes
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return min(y0, _y0), min(x0, _x0), max(yn, _yn), max(xn, _xn)

def comp_area(_box, box):
    _y0,_x0,_yn,_xn =_box; _A = (_yn - _y0) * (_xn - _x0)
    y0, x0, yn, xn = box;   A = (yn - y0) * (xn - x0)
    return _A-A, min(_A,A)- ave_L**2  # mA, dA


def comp_N(Link, rn, rng, dir=None):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = dir is not None  # compared links have binary relative direction
    dir = 1 if dir is None else dir  # convert to numeric
    _N,N = Link.nodet  # comp externals:
    if fd:
        _L, L = _N.dist, N.dist;  dL = _L-L; mL = min(_L,L) - ave_L  # direct match
        mA,dA = comp_angle(_N.angle, [d*dir for d in N.angle])  # rev 2nd link in llink
    else:
        _L, L = len(_N.node_),len(N.node_); dL = _L-L; mL = min(_L,L) - ave_L
        mA,dA = comp_area(_N.box, N.box)  # compare area in CG vs angle in CL
    n = .3
    M = mL+mA; D = abs(dL)+abs(dA); Et = np.array([M,D, M>D,D<=M], dtype='float')
    md_t = [[[mL,dL, mA,dA], Et,n]]  # [mdext]
    if not fd:  # CG
        mdlat = comp_latuple(_N.latuple,N.latuple,rn,fagg=1)
        mdLay = comp_md_(_N.mdLay[0], N.mdLay[0], rn, dir)
        md_t += [mdlat,mdLay]; Et += mdlat[1] + mdLay[1]; n += mdlat[2] + mdLay[2]
    # | n = (_n+n)/2?
    # Et[0] += ave_rn - rn?
    elay = CH( H=[CH(n=n, md_t=md_t, Et=Et)], n=n, md_t=deepcopy(md_t), Et=copy(Et))
    if _N.derH and N.derH:
        dderH = _N.derH.comp_H(N.derH, rn, dir=dir)  # comp shared layers
        elay.append_(dderH, flat=1)
    # spec: comp_node_(node_|link_), mult agg Lays, combinatorial?
    Link.derH = elay; elay.root = Link; Link.n = min(_N.n,N.n); Link.nodet = [_N,N]; Link.yx = np.add(_N.yx,N.yx) /2
    # preset angle, dist
    Et = elay.Et
    if Et[0] > ave * Et[2] * (rng+1):
        for rev, node in zip((0,1), (N,_N)):
            # reverse Link direction for _N
            if fd: node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            else:  node.rim += [(Link, rev)]

        return Et

def cluster_N_(root, N_, fd, nest=1, max_dist=np.inf):

    def nest_rim(L):
        # not quite right, this append is redundant to 1st layer in rim and extH.H:
        for rev, N in zip((0,1), L.nodet):  # reverse Link direction for 2nd N
            if fd: N.rimt[-1][1-rev] += [(L,rev)]  # append rngLay
            else:  N.rim[-1] += [(L,rev)]
            N.extH.H[-1].add_H(L.derH)

    # form top clusters:
    Gt_ = []
    for N in N_:
        N.merged = 0
        N.extH = CH().append_(N.extH)
        rim = N.rimt if fd else N.rim; rim[:] = [rim]  # convert to rim_
    for N in N_:
        if N.merged: continue
        node_, link_, et = {N}, set(), np.array([.0,.0,.0,.0])  # init Gt
        L_ = [L for L,rev in (N.rimt[0]+N.rimt[1] if fd else N.rim) if L.dist < max_dist]
        _eN_ = {n for L in L_ for n in L.nodet if n not in node_ and not n.merged}  # init ext_N_
        while _eN_:
            eN_ = set()
            for eN in _eN_:  # known +ve link
                node_.add(eN)
                for L,rev in (eN.rimt[0]+eN.rimt[1] if fd else eN.rim):
                    if L not in link_:
                        link_.add(L); et += L.derH.Et
                        for G in L.nodet:
                            if G not in node_ and not G.merged: eN_.add(G)
            _eN_ = eN_
        Gt = [node_, link_, et]
        for n in node_:
            n.root_ = [Gt]; n.merged = 1
        Gt_ += [Gt]
    # top-down recursive form, cluster rng segments:
    for i, Gt in enumerate(Gt_):  # top Gt
        node_, link_, Et = Gt
        if len(link_) > ave_L:
            link_ = sorted(link_, key=lambda x: x.dist, reverse=True)
            _L = link_[0]
            et = _L.derH.Et  # rng segment
            for ii, L in enumerate(link_[1:], start=1):  # long links first
                ddist = _L.dist - L.dist  # positive
                if ddist < ave_L or et[0] < ave:  # ~= dist Ns or weak
                    nest_rim(L.nodet)
                    et += L.derH.Et
                elif et[0] > ave  * et[2] * nest+1:  # inverted rng
                    sub_N_ = {N for L in link_[ii:] for N in L.nodet}  # shorter links are not yet clustered
                    sub_Gt_ = cluster_N_(Gt, sub_N_, fd, nest+1, L.dist)
                    Gt_[i] += [sub_Gt_]
                    break
    G_ = []
    for Gt in Gt_:
        M, R = Gt[2][0::2]  # Gt: node_, link_, et, sub_Gt_
        if M > R * ave * nest:  # rdn incr / lower rng
            G_ += [sum2graph(root, Gt, fd, nest)]
    return G_

def sum2graph(root, grapht, fd, nest):  # sum node and link params into graph, aggH in agg+ or player in sub+

    node_, link_, Et, Gt_ = grapht
    node_ = list(node_)  # convert from set
    graph = CG(fd=fd, root_=[root], node_=node_, link_=link_, rng=nest)
    graph.sub_G_ = [ sum2graph(graph, Gt, fd, nest+1) for Gt in Gt_]  # depth-first
    yx = [0,0]
    lay0 = CH(node_=node_)  # comparands, vs. L_: summands?
    for link in link_:  # unique current-layer mediators: Ns if fd else Ls
        lay0.add_H(link.derH) if lay0 else lay0.append_(link.derH)
    graph.derH.append_(lay0)  # empty for single-node graph
    derH = CH()
    for N in node_:
        graph.n += N.n  # +derH.n, total nested comparable vars
        graph.box = extend_box(graph.box, N.box)  # pre-compute graph.area += N.area?
        yx = np.add(yx, N.yx)
        if N.derH: derH.add_H(N.derH)  # derH.Et=Et?
        if isinstance(N,CG):
            add_md_(graph.mdLay, N.mdLay)
            add_lat(graph.latuple, N.latuple)
        N.root_[-1] = graph
    graph.derH.append_(derH, flat=1)  # comp(derH) forms new layer, higher layers are added by feedback
    L = len(node_)
    yx = np.divide(yx,L); graph.yx = yx
    # ave distance from graph center to node centers:
    graph.aRad = sum([np.hypot(*np.subtract(yx,N.yx)) for N in node_]) / L
    if fd:
        # assign alt graphs from d graph, after both linked m and d graphs are formed
        for node in node_:  # CG or CL
            mgraph = node.root_[-1]
            if mgraph:
                for fd, (G, alt_G) in enumerate(((mgraph,graph), (graph,mgraph))):  # bilateral assign:
                    if G not in alt_G.alt_graph_:
                        G.alt_graph_ += [alt_G]
    return graph