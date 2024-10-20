import numpy as np
from itertools import combinations, zip_longest
from copy import deepcopy, copy
from frame_blobs import CBase
from .slice_edge import comp_angle, CsliceEdge
from .comp_slice import comp_slice, comp_latuple, add_lat, aves, comp_md_, add_md_

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
        G.rim_ = []  # direct external links, nested per rng
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
        # add rimt_, elay | extH if der+
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
        e.root_, e.extH = [], CH()
        Q += [e]
    # cross-comp link_ or node_:
    N__,L__, Et, rng = comp_link_(Q) if fd else comp_node_(Q)

    m,d,mr,dr = Et; fvd = d > ave_d * dr*(rng+1); fvm = m > ave * mr*(rng+1)
    if fvd or fvm:
        L_ = [L for L_ in L__ for L in L_]  # root += L.derH:
        if fd: root.derH.append_(CH().append_(CH().copy(L_[0].derH)))  # new rngLay, aggLay
        else:  root.derH.H[-1].append_(L_[0].derH)  # append last aggLay
        for L in L_[1:]:
            root.derH.H[-1].H[-1].add_H(L.derH)  # accum Lay
        # comp_link_
        if fvd and len(L_) > ave_L:  # comp L if DL, sub-cluster LLs by mL:
            agg_recursion(root, L_, fd=1)  # appends last aggLay, L_ = lG_
        if fvm:
            cluster_N__(root, N__, fd)  # merge and cluster rngLays in N__,
            for N_ in N__:  # replace root.node_ with nested graph, if any
                if len(N_) > ave_L:  # comp_node_
                    agg_recursion(root, N_, fd=0)  # forms higher-composition graphs
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
    N__,L__, ET = [],[], np.array([.0,.0,.0,.0])  # rng H
    while True:  # prior vM
        Gp_,N_,L_,Et = [],set(),[], np.array([.0,.0,.0,.0])
        for Gp in _Gp_:
            _G,G, rn, dy,dx, radii, dist = Gp
            _nrim = set([(Lt[0].nodet[1] if Lt[0].nodet[0] is _G else Lt[0].nodet[0]) for rim in _G.rim_ for Lt in rim])
            nrim = set([(Lt[0].nodet[1] if Lt[0].nodet[0] is G else Lt[0].nodet[0]) for rim in G.rim_ for Lt in rim])
            if _nrim & nrim:  # indirectly connected Gs,
                continue     # no direct match priority?
            M = (_G.mdLay[1][0]+G.mdLay[1][0]) *icoef**2 + (_G.derH.Et[0]+G.derH.Et[0])*icoef + (_G.extH.Et[0]+G.extH.Et[0])
            # comp if < max distance of likely matches *= prior G match * radius:
            if dist < max_dist * (radii * icoef**3) * M:
                while len(_G.rim_) < rng-1: _G.rim_ += [[]]  # add empty rng rim to align in cluster_N__
                while len(G.rim_) < rng-1: G.rim_ += [[]]
                # rng-1: new rim added in comp_N:
                Link = CL(nodet=[_G,G], angle=[dy,dx], dist=dist, box=extend_box(G.box,_G.box))
                et = comp_N(Link, rn, rng)
                L_ += [Link]  # include -ve links
                if et is not None:
                    N_.update({_G,G}); Et += et; _G.add,G.add = 1,1  # for clustering
            else:
                Gp_ += [Gp]  # re-evaluate not-compared pairs with one incremented N.M:
        # cluster or merge N_,L_ in cluster_N__:
        N__ += [[N_,Et]]; L__ += [L_]; ET += Et
        if Et[0] > ave * Et[2]:  # current-rng vM
            rng += 1
            _Gp_ = [Gp for Gp in Gp_ if Gp[0].add or Gp[1].add]  # one incremented N.M
        else:  # low projected rng+ vM
            break

    return N__, L__, ET,rng


def comp_link_(iL_):  # comp CLs via directional node-mediated link tracing: der+'rng+ in root.link_ rim_t node rims

    fd = isinstance(iL_[0].nodet[0], CL)
    for L in iL_:
        L.mL_t, L.rimt_, L.aRad, L.visited_ = [[],[]],[],0,[L]
        # init mL_t (mediated Ls):
        for rev, n, mL_ in zip((0,1), L.nodet, L.mL_t):
            rim_ = n.rimt_ if fd else n.rim_
            for _L,_rev in rim_[0][0] + rim_[0][1] if fd else rim_[0]:
                if _L is not L and _L.derH.Et[0] > ave * _L.derH.Et[2]:
                    mL_ += [(_L, rev ^_rev)]  # direction of L relative to _L
    _L_, L__, LL__, ET = iL_,[],[], np.array([.0,.0,.0,.0])
    med = 1
    while True:
        # xcomp _L_:
        L_,LL_, Et = set(),[], np.array([.0,.0,.0,.0])
        for L in _L_:
            for mL_ in L.mL_t:
                for _L, rev in mL_:  # rev is relative to L
                    rn = _L.n / L.n
                    if rn > ave_rn: continue  # scope disparity
                    dy,dx = np.subtract(_L.yx, L.yx)
                    Link = CL(nodet=[_L,L], angle=[dy,dx], dist=np.hypot(dy,dx), box=extend_box(_L.box, L.box))
                    # comp L,_L:
                    et = comp_N(Link, rn, rng=med, dir = -1 if rev else 1)  # d = -d if L is reversed relative to _L
                    LL_ += [Link]  # include -ves, L.rim_t += Link, order: nodet < L < rimt_, mN.rim || L
                    if et is not None:
                        L_.update({_L,L}); Et += et
        if not L_: break
        L__+=[[list(L_), Et]]; LL__+=[LL_]; ET += Et
        # rng+ eval:
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
                                        mL_.add((__L, rev^_rev^__rev))  # combine reversals: 2 * 2 mNs, 1st 2 are pre-combined
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
    return L__, LL__, ET, med  # =rng

def extend_box(_box, box):  # add 2 boxes
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return min(y0, _y0), min(x0, _x0), max(yn, _yn), max(xn, _xn)

def comp_area(_box, box):
    _y0,_x0,_yn,_xn =_box; _A = (_yn - _y0) * (_xn - _x0)
    y0, x0, yn, xn = box;   A = (yn - y0) * (xn - x0)
    return _A - A, min(_A, A) - ave_L  # mA, dA


def comp_N(Link, rn, rng, dir=None):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = dir is not None  # compared links have binary relative direction
    dir = 1 if dir is None else dir  # convert to numeric
    _N,N = Link.nodet  # comp externals:
    if fd:
        _L, L = _N.dist, N.dist;  dL = _L-L; mL = min(_L,L) - ave_L  # direct match
        mA,dA = comp_angle(_N.angle, [d*dir for d in N.angle])  # may invert 2nd comparand in link
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
    # spec: comp_node_(node_|link_), of different agg orders, combinatorial?
    Link.derH = elay; elay.root = Link; Link.n = min(_N.n,N.n); Link.nodet = [_N,N]; Link.yx = np.add(_N.yx,N.yx) /2  # prior angle, dist
    Et = elay.Et
    fv = Et[0] > ave * Et[2] * (rng+1)
    for rev, node in zip((0,1),(N,_N)):  # reverse Link direction for N
        # L_ includes negative Ls
        if (len(node.rimt_) if fd else len(node.rim_)) < rng:
            if fd: node.rimt_ += [[[(Link,rev)],[]]] if dir else [[[],[(Link,rev)]]]  # add rng layer
            else:  node.rim_ += [[(Link, rev)]]
        else:
            if fd: node.rimt_[-1][1-rev] += [(Link,rev)]  # add in last rng layer, opposite to _N,N dir
            else:  node.rim_[-1] += [(Link, rev)]
        if fv:  # select for next rng:
            if len(node.extH.H) < rng:  # init rng layer
                node.extH.append_(elay) # node.lrim_ += [{Link}]; node.nrim_ += [{(_N,N)[rev]}]  # _node
            else:  # append last layer
                node.extH.H[-1].add_H(elay)  # node.lrim_[-1].add(Link); node.nrim_[-1].add((_N,N)[rev])
    if fv:
        return Et

def cluster_N__(root, iN__, fd):  # form rng graphs by merging lower-rng graphs if >ave rim: cross-graph rng links

    N__ = []
    N_,et = iN__.pop()
    rng = len(iN__)
    while iN__:
        _N_,_et = iN__.pop()  # top-down
        if _et[0] < ave and et[0] < ave:  # merge weak rngs, higher into lower
            for n in N_:
                if n not in _N_: _N_ += [n]  # lower rng
                if isinstance(n, CL):
                    n.rimt_[rng-1][0] += n.rimt_[rng][0]; n.rimt_[rng-1][1] += n.rimt_[rng][1]; n.rimt_.pop(rng)  # merged rimt
                else:
                    n.rim_[rng-1] += n.rim_[rng]; n.rim_.pop(rng)  # merged rim
                n.extH.H[rng-1].add_H(n.extH.H[rng]); n.extH.H.pop(rng)  # merged extH
            _et += et
        else:
            N__ += [[_N_,_et]]; N_ = _N_; _et = et
        rng -= 1
    N__ += [[_N_,_et] if _N_ in locals else [N_,et]]  # 1st N_, [N_,et] if no while: single N_ in iN__
    Gt__ = []
    for rng, (N_,et) in enumerate(reversed(N__)):  # bottom-up
        if et[0] > ave:  # init Gt_ from N.root_[-1]
            Gt_ = init_Gt_(N_, fd, rng)
            if len(Gt_) < ave_L:
                Gt__ += [N_]; break
        else:
            Gt__ += [N_]; break
        Gt__ += [merge_Gt_(Gt_)]  # eval to merge connected Gts
    n__ = []
    # Gts -> CGs:
    for rng, Gt_ in enumerate(Gt__, start=1):
        if isinstance(Gt_, set): continue  # recycled N_
        n_ = []
        for node_,link_,et, _,_ in Gt_:
            M, R = et[0::2]
            if M > R * ave * rng:  # all rngs val
                n_ += [sum2graph(root, [node_,link_,et], fd, rng)]
            else:  # weak Gt
                n_ += [[node_,link_,et]]  # skip in current-agg xcomp, unpack if extended lower-agg xcomp
        n__ += [n_]
    iN__[:] = n__  # replace Ns with Gts


def init_Gt_(N_, fd, rng):  # init higher root Gt_ from N.root_[-1]

    for N in N_:
        if not N.root_:  # true in N__[0]
            rim_ = N.rimt_ if fd else N.rim_
            rim = set([Lt[0] for Lt in (rim_[rng][0]+rim_[rng][1] if fd else rim_[rng]) if Lt[0].derH.Et[0] > ave * Lt[0].derH.Et[2] * (rng+1)])
            _Gt = [[N], set(), np.array([.0,.0,.0,.0]), rim, 1]  # mrg = 1 to skip below
            N.root_ = [_Gt]
    _Gt_ = []
    for N in N_:
        if N.root_[-1] not in _Gt_: _Gt_ += [N.root_[-1]]  # unique roots, lower or initialized above
    Gt_ = []
    for _Gt in _Gt_:
        node_, link_, et, rim, mrg = _Gt
        if mrg:   # initialized above
            _Gt[-1] = 0; Gt_ += [_Gt]
        else:  # init with lower root
            Rim = set()
            for n in node_:
                rim_ = n.rimt_ if fd else n.rim_
                if len(rim_) > rng:
                    rim = set([Lt[0] for Lt in (rim_[rng][0]+rim_[rng][1] if fd else rim_[rng]) if Lt[0].derH.Et[0] > ave * Lt[0].derH.Et[2] * (rng+1)])
                    Rim.update(rim)
            Gt = [node_.copy(), link_.copy(), et.copy(), Rim, 0]  # Rim can't be empty
            for n in node_: n.root_ += [Gt]
            Gt_ += [Gt]
    return Gt_

def merge_Gt_(Gt_):  # eval connected Gts for merging in higher-rng Gts
    GT_ = []
    for Gt in Gt_:
        node_,link_,et, rim,mrg = Gt
        if mrg: continue
        while any(rim):  # extend node_,link_, replace rim
            ext_rim = set()
            for _L in rim:
                G,_G = _L.nodet if _L.nodet[0] in node_ else list(reversed(_L.nodet)) # one is outside node_
                if _G.root_[-1] is Gt: continue  # merged in prior loop
                _node_, _link_, _et, _rim, _ = _G.root_[-1]
                crim = (rim | ext_rim) & _rim  # intersect with extended rim
                xrim = _rim - crim   # exclusive _rim
                cV = 0  # common val
                for __L in crim:  # common Ls
                    M, R = __L.derH.Et[0::2]
                    v = M - ave * R
                    if v > 0: cV += M  # cluster by +ve links only
                if cV / (_et[0]+1) > ave * ccoef:  # /_M: lower-rng cohesion ~ behavioral independence, may be break up combined G?
                    _G.root_[-1][-1] = 1  # set mrg
                    ext_rim.update(xrim)  # add new links
                    for _node in _node_:
                        if _node not in node_:
                            _node.root_[-1] = Gt; node_ += [_node]
                    link_.update(_link_|{_L})  # external L
                    et += _L.derH.Et + _et
            rim = ext_rim
        GT_ += [Gt]
    return GT_

def sum2graph(root, grapht, fd, rng):  # sum node and link params into graph, aggH in agg+ or player in sub+

    N_, L_, Et = grapht  # node_,link_: flatten N__,L__ if cluster rng++?

    graph = CG(fd=fd, root_ = root, node_=N_, link_=L_, rng=rng)  # root_ will be updated to list roots in rng_node later?
    yx = [0,0]
    lay0 = CH(node_= N_)  # comparands, vs. L_: summands?
    for link in L_:  # unique current-layer mediators: Ns if fd else Ls
        lay0.add_H(link.derH) if lay0 else lay0.append_(link.derH)
    graph.derH.append_(lay0)  # empty for single-node graph
    derH = CH()
    for N in N_:
        graph.n += N.n  # +derH.n, total nested comparable vars
        graph.box = extend_box(graph.box, N.box)  # pre-compute graph.area += N.area?
        yx = np.add(yx, N.yx)
        if N.derH: derH.add_H(N.derH)  # derH.Et=Et?
        if isinstance(N,CG):
            add_md_(graph.mdLay, N.mdLay)
            add_lat(graph.latuple, N.latuple)
        N.root_[-1] = graph
    graph.derH.append_(derH, flat=1)  # comp(derH) forms new layer, higher layers are added by feedback
    L = len(N_)
    yx = np.divide(yx,L); graph.yx = yx
    # ave distance from graph center to node centers:
    graph.aRad = sum([np.hypot(*np.subtract(yx,N.yx)) for N in N_]) / L
    if fd:
        # assign alt graphs from d graph, after both linked m and d graphs are formed
        for node in graph.node_:  # CG or CL
            mgraph = node.root_[-1]
            if mgraph:
                for fd, (G, alt_G) in enumerate(((mgraph,graph), (graph,mgraph))):  # bilateral assign:
                    if G not in alt_G.alt_graph_:
                        G.alt_graph_ += [alt_G]
    return graph