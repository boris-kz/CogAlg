import numpy as np
from itertools import combinations, zip_longest
from copy import deepcopy, copy
from frame_blobs import CBase
from .slice_edge import comp_angle, CsliceEdge
from .comp_slice import comp_slice, comp_latuple, add_lat, aves

'''
This code is ostensibly for clustering segments within edge: high-gradient blob, but it's far too complex for the case.
That's because this is a prototype for open-ended compositional recursion: clustering blobs, graphs of blobs, etc.
We will later prune complete code for intra-edge version.
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
Combined representation of a graph is nested in a dual tree of down-forking elements: node_, and up-forking clusters: root_.
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
prefix  _ denotes prior of two same-name variables
postfix _ denotes array name, vs. same-name elements
capitalized variables are normally summed small-case variables
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
        He.node_ = [] if node_ is None else node_  # concat bottom nesting order, may be redundant to G.node_
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

    def add_md_C(HE, He, irdnt=[]):  # sum dextt | dlatt | dlayt

        HE.H[:] = [V + v for V, v in zip_longest(HE.H, He.H, fillvalue=0)]
        HE.n += He.n  # combined param accumulation span
        HE.Et += He.Et
        if any(irdnt): HE.Et[2:] = [E + e for E, e in zip(HE.Et[2:], irdnt)]
        return HE

    def accum_lay(HE, He, irdnt):

        if HE.md_t:
            for MD_C, md_C in zip(HE.md_t, He.md_t):  # dext_C, dlat_C, dlay_C
                MD_C.add_md_C(md_C)
            HE.n += He.n
            HE.Et += He.Et
            if any(irdnt):
                HE.Et[2:] = [E+e for E,e in zip(HE.Et[2:], irdnt)]
        else:
            HE.md_t = [CH().copy(md_) for md_ in He.md_t]
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

    def update_root(HE, He):  # should be batched or done in calling function, not recursive

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

    def comp_md_C(_He, He, rn=1, dir=1):

        vm, vd, rm, rd = 0,0,0,0
        derLay = []
        for i, (_d, d) in enumerate(zip(_He.H[1::2], He.H[1::2])):  # compare ds in md_ or ext
            d *= rn  # normalize by compared accum span
            diff = (_d - d) * dir  # in comp link: -1 if reversed nodet else 1
            match = min(abs(_d), abs(d))
            if (_d < 0) != (d < 0): match = -match  # negate if only one compared is negative
            vm += match - aves[i]  # fixed param set?
            vd += diff
            rm += vd > vm; rd += vm >= vd
            derLay += [match, diff]  # flat

        return CH(H=derLay, Et=np.array([vm,vd,rm,rd],dtype='float'), n=1)

    def comp_H(_He, He, rn=1, dir=1):  # unpack each layer of CH down to numericals and compare each pair

        der_md_t = []; Et = np.array([.0,.0,.0,.0])

        for _md_C, md_C in zip(_He.md_t, He.md_t):  # default per layer
            # lay: [mdlat, mdLay, mdext]
            der_md_C = _md_C.comp_md_C(md_C, rn=1, dir=dir)
            der_md_t += [der_md_C]
            Et += der_md_C.Et

        DLay = CH( node_=_He.node_+He.node_, md_t = der_md_t, Et=Et, n=2.5)
        # comp,unpack H, empty in bottom or deprecated layer, no comp node_:

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
                elif attr == "md_t":
                    _He.md_t += [CH().copy(md_) for md_ in He.md_t]
                elif attr == "node_":
                    _He.node_ = copy(He.node_)
                else:
                    setattr(_He, attr, deepcopy(value))
        return _He

class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster

    def __init__(G, root_= None, node_=None, link_=None, latuple=None, mdLay=None, derH=None, extH=None,
                 rng=1, fd=0, n=0, box=None, yx=None, S=0, A=(0,0), area=0):
        super().__init__()
        G.fd = 0 if fd else fd  # 1 if cluster of Ls | lGs?
        G.root_ = [] if root_ is None else root_ # same nodes in higher rng layers
        G.node_ = [] if node_ is None else node_ # convert to GG_ in agg++
        G.link_ = [] if link_ is None else link_ # internal links per comp layer in rng+, convert to LG_ in agg++
        G.latuple = [0,0,0,0,0,[0,0]] if latuple is None else latuple  # lateral I,G,M,Ma,L,[Dy,Dx]
        G.mdLay = CH(root=G) if mdLay is None else mdLay
        # maps to node_H / agg+|sub+:
        G.derH = CH(root=G) if derH is None else derH  # sum from nodes, then append from feedback
        G.extH = CH(root=G) if extH is None else extH  # sum from rim_ elays, H maybe deleted
        G.rim_ = []  # direct external links, nested per rng
        G.rng = rng
        G.n = n   # external n (last layer n)
        G.S = S  # sparsity: distance between node centers
        G.A = A  # angle: summed dy,dx in links
        G.area = area
        G.aRad = 0  # average distance between graph center and node center
        G.box = [np.inf, np.inf, -np.inf, -np.inf] if box is None else box  # y0,x0,yn,xn
        G.yx = [0,0] if yx is None else yx  # init PP.yx = [(y0+yn)/2,(x0,xn)/2], then ave node yx
        G.alt_graph_ = []  # adjacent gap+overlap graphs, vs. contour in frame_graphs
        G.visited_ = []
        # G.Et = [0,0,0,0] if Et is None else Et   # redundant to extH.Et? rim_ Et, val to cluster, -rdn to eval xcomp
        # G.fback_ = []  # always from CGs with fork merging, no dderHm_, dderHd_
        # Rdn: int = 0  # for accumulation or separate recursion count?
        # G.Rim = []  # links to the most mediated nodes
        # depth: int = 0  # n sub_G levels over base node_, max across forks
        # nval: int = 0  # of open links: base alt rep
        # id_H: list = z([[]])  # indices in the list of all possible layers | forks, not used with fback merging
        # top aggLay: derH from links, lower aggH from nodes, only top Lay in derG:
        # top Lay from links, lower Lays from nodes, hence nested tuple?
    def __bool__(G): return G.n != 0  # to test empty

class CL(CBase):  # link or edge, a product of comparison between two nodes or links
    name = "link"

    def __init__(l, nodet=None,derH=None, S=0, A=None, box=None, md_t=None, H_=None, root_=None):
        super().__init__()
        # CL = binary tree of Gs, depth+/der+: CL nodet is 2 Gs, CL + CLs in nodet is 4 Gs, etc.,
        # unpack sequentially
        l.nodet = [] if nodet is None else nodet  # e_ in kernels, else replaces _node,node: not used in kernels
        l.A = [0,0] if A is None else A  # dy,dx between nodet centers
        l.S = 0 if S is None else S  # span: distance between nodet centers, summed into sparsity in CGs
        l.area = 0  # sum nodet
        l.box = [] if box is None else box  # sum nodet
        l.derH = CH(root=l) if derH is None else derH
        l.H_ = [] if H_ is None else H_  # if agg++| sub++?
        l.Vt = [0,0]  # for rim-overlap modulated segmentation, init derH.Et[:2]
        l.n = 1  # min(node_.n)
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
                            box = [y,x-len(N.dert_), y,x]; area = 1; A,S = None,None
                        PP = CG(fd=0, root_=root, node_=P_,link_=link_,mdLay=CH(H=H,Et=Et,n=n),latuple=lat, A=A,S=S,area=area,box=box,yx=[y,x],n=n)
                        y0,x0,yn,xn = box
                        PP.aRad = np.hypot(*np.subtract(PP.yx,(yn,xn)))
                        G_ += [PP]
                if len(G_) > 10:
                    agg_recursion(edge, G_, fd=0)  # discontinuous PP_ xcomp, cluster

def agg_recursion(root, iQ, fd):  # breadth-first rng+ cross-comp -> eval clustering, recursion per fd fork: rng+ | der+

    Q = []
    for e in iQ:
        if isinstance(e, list): continue  # skip weak Gts
        e.root_, e.extH = [], CH()
        Q += [e]
    # cross-comp link_ or node_:
    N__,L__,Et,rng = rng_link_(Q) if fd else rng_node_(Q)

    m,d,mr,dr =Et; fvd = d > ave_d * dr*(rng+1); fvm = m > ave * mr*(rng+1)
    if fvd or fvm:
        L_ = [L for L_ in L__ for L in L_]  # root += L.derH:
        if fd: root.derH.append_(CH().append_(CH().copy(L_[0].derH)))  # new rngLay, aggLay
        else:  root.derH.H[-1].append_(L_[0].derH)  # append last aggLay
        for L in L_[1:]:
            root.derH.H[-1].H[-1].add_H(L.derH)  # accum Lay
        # rng_link_
        if fvd and len(L_) > ave_L: # comp L, sub-cluster /dL: mL is redundant to mN?
            agg_recursion(root, L_, fd=1)  # appends last aggLay, L_=lG_ if segment
        if fvm:
            cluster_N__(root, N__, fd)  # cluster rngLays in root.node_,
            for N_ in N__:  # replace root.node_ with nested H of graphs
                if len(N_) > ave_L:  # rng_node_
                    agg_recursion(root, N_, fd=0)  # forms higher-composition graphs
'''
     if flat derH:
        root.derH.append_(CH().copy(L_[0].derH))  # init
        for L in L_[1:]: root.derH.H[-1].add_H(L.derH)  # accum
'''

def rng_node_(_N_):  # rng+ forms layer of rim_ and extH per N, appends N__,L__,Et, ~ graph CNN without backprop

    N__,L__,ET = [],[], np.array([.0,.0,.0,.0])  # rng H
    _Gp_ = []  # [G pair+ co-positionals]
    for _G, G in combinations(_N_, r=2):
        rn = _G.n / G.n
        if rn > ave_rn: continue  # scope disparity
        radii = G.aRad + _G.aRad
        dy,dx = np.subtract(_G.yx,G.yx)
        dist = np.hypot(dy,dx)
        _Gp_ += [(_G,G, rn, dy,dx, radii, dist)]
    icoef = .5  # internal M proj_val / external M proj_val
    rng = 1  # len N__
    while True:  # prior rng vM
        Gp_,N_,L_, Et = [],set(),[], np.array([.0,.0,.0,.0])
        for Gp in _Gp_:
            _G,G, rn, dy,dx, radii, dist = Gp
            _nrim = set([(Lt[0].nodet[1] if Lt[0].nodet[0] is _G else Lt[0].nodet[0]) for rim in _G.rim_ for Lt in rim])
            nrim = set([(Lt[0].nodet[1] if Lt[0].nodet[0] is G else Lt[0].nodet[0]) for rim in G.rim_ for Lt in rim])
            if _nrim & nrim:  # not sure: skip indirectly connected Gs, no direct match priority?
                continue
            M = (_G.mdLay.Et[0]+G.mdLay.Et[0]) *icoef**2 + (_G.derH.Et[0]+G.derH.Et[0])*icoef + (_G.extH.Et[0]+G.extH.Et[0])
            # comp if < max distance of likely matches *= prior G match * radius:
            if dist < max_dist * (radii*icoef**3) * M:
                Link = CL(nodet=[_G,G], S=2, A=[dy,dx], box=extend_box(G.box,_G.box))
                et = comp_N(Link, rn, rng)
                L_ += [Link]  # include -ve links
                if et is not None:
                    N_.update({_G, G}); Et += et  # for clustering
            else:
                Gp_ += [Gp]  # re-evaluate not-compared pairs with one incremented N.M:
        if Et[0] > ave * Et[2]:  # current-rng vM
            rng += 1
            N__+= [N_]; L__+= [L_]; ET += Et  # sub-cluster pL_,N_
            _Gp_ = [Gp for Gp in Gp_ if (Gp[0] in N_ or Gp[1] in N_)]  # one incremented N.M
        else:  # low projected rng+ vM
            break
    return N__,L__,ET,rng


def rng_link_(iL_):  # comp CLs via directional node-mediated link tracing: der+'rng+ in root.link_ rim_t node rims

    fd = isinstance(iL_[0].nodet[0], CL)
    for L in iL_:
        L.mL_t, L.rimt_, L.aRad, L.visited_ = [[],[]],[],0,[L]
        # init mL_t (mediated Ls):
        for rev, n, mL_ in zip((0,1), L.nodet, L.mL_t):
            rim_ = n.rimt_ if fd else n.rim_
            for _L,_rev in rim_[0][0] + rim_[0][1] if fd else rim_[0]:
                if _L.derH.Et[0] > ave * _L.derH.Et[2]:
                    mL_ += [(_L, rev^_rev)]  # direction of L relative to _L
    _L_, L__, LL__,ET = iL_,[],[], np.array([.0,.0,.0,.0])
    med = 1
    while True:
        # xcomp _L_:
        L_,LL_, Et = set(),[], np.array([.0,.0,.0,.0])
        for L in _L_:
            for mL_ in L.mL_t:
                for _L, rev in mL_:  # rev is relative to L
                    rn = _L.n / L.n
                    if rn > ave_rn: continue  # scope disparity
                    Link = CL(nodet=[_L,L], S=2, A=np.subtract(_L.yx,L.yx), box=extend_box(_L.box, L.box))
                    # comp L,_L:
                    et = comp_N(Link, rn, rng=med, dir = -1 if rev else 1)  # d = -d if L is reversed relative to _L
                    LL_ += [Link]  # include -ves, L.rim_t += Link, order: nodet < L < rimt_, mN.rim || L
                    if et is not None:
                        L_.update({_L,L}); Et += et
        L__+=[L_]; LL__+=[LL_]; ET += Et
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
                                        mL_.add((__L, rev^_rev^__rev))  # combine revs: 2 * 2 mNs, 1st 2 are pre-combined
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

def extend_box(box, _box):  # add 2 boxes
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return min(y0, _y0), min(x0, _x0), max(yn, _yn), max(xn, _xn)


def comp_N(Link, rn, rng, dir=None):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = dir is not None  # compared links have binary relative direction
    dir = 1 if dir is None else dir  # convert from None into numeric
    _N, N = Link.nodet
    _L,L = (2,2) if fd else (len(_N.node_),len(N.node_)); _S,S = _N.S,N.S; _A,A = _N.A,N.A
    A = [d * dir for d in A]  # reverse angle direction if N is left link
    mdext = comp_ext(_L,L, _S,S/rn, _A,A)
    md_t = [mdext]; Et = mdext.Et.copy(); n = mdext.n
    if not fd:  # CG
        H, lEt, ln = comp_latuple(_N.latuple,N.latuple,rn,fagg=1)
        mdlat = CH(H=H, Et=lEt, n=ln)
        mdLay = _N.mdLay.comp_md_C(N.mdLay, rn, dir)
        md_t += [mdlat,mdLay]; Et += lEt + mdLay.Et; n += ln + mdLay.n
    # | n = (_n+n)/2?
    # Et[0] += ave_rn - rn?
    elay = CH( H=[CH(n=n, md_t=md_t, Et=Et)], n=n, md_t=[CH().copy(md_) for md_ in md_t], Et=copy(Et))
    if _N.derH and N.derH:
        dderH = _N.derH.comp_H(N.derH, rn, dir=dir)  # comp shared layers
        elay.append_(dderH, flat=1)
    # spec: rng_node_(node_|link_), of different agg orders, combinatorial?
    Link.derH = elay; elay.root = Link; Link.n = min(_N.n,N.n); Link.nodet = [_N,N]; Link.yx = np.add(_N.yx,N.yx) /2  # prior S,A
    Et = elay.Et
    fv = Et[0] > ave * Et[2] * (rng+1)
    for rev, node in zip((0,1),(N,_N)):  # reverse Link direction for N
        # L_ includes negative Ls
        if (len(node.rimt_) if fd else len(node.rim_)) < rng:
            if fd: node.rimt_ = [[[(Link,rev)],[]]] if dir else [[[],[(Link,rev)]]]  # add rng layer
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

def comp_ext(_L,L,_S,S,_A,A):  # compare non-derivatives:

    dL = _L - L; mL = min(_L,L) - ave_L  # direct match
    dS = _S/_L - S/L; mS = min(_S,S) - ave_L  # sparsity is accumulated over L
    mA, dA = comp_angle(_A,A)  # angle is not normalized
    M = mL + mS + mA
    D = abs(dL) + abs(dS) + abs(dA)  # normalize relative to M, signed dA?

    return CH(H=[mL,dL, mS,dS, mA,dA], Et=np.array([M,D,M>D,D<=M]), n=0.5)

'''
    Higher graphs are based on connectivity over current + shorter links: merge lower-rng graphs, 
    if >ave cross-graph rng links:
    Gt merge / shared-links V: relative similarity sim(N1, N2) * (N1_surround_V + N2_surround_V):
    LV = (sim(N1,N2) / ave) * ((N1_sur_Sim - ave * N2_sur_Rng) + (N2_sur_Sim - ave * N2_sur_Rng))
    surround was computed over variable distance, incr. if >ave similarity over shorter distance
'''
def cluster_N__(root, N__, fd):  # cluster G__|L__ by value density of +ve links per node

    Gt__ = []
    for rng, N_ in enumerate(N__, start=1):  # all Ls and current-rng Gts are unique
        if len(N_) < ave_L:
            Gt__ += [N_]; continue
        Gt_ = []
        for N in N_:  N.merged = 0
        for N in N_:  # add new root for generic merging:
            if N.merged: continue
            if not N.root_:
                Gt = [[[N], set(), np.array([.0,.0,.0,.0]), get_rim(N,fd,rng), 0]]
                N.root_ = [Gt]
            else:
                node_, link_, et, rim, mrg = N.root_[-1]
                Link_ = list(link_); rng_rim = []
                for n in node_:
                    n.merged = 1
                    for L in get_rim(n, fd, rng):
                        if all([n in node_ for n in L.nodet]): Link_ += [L]
                        else: rng_rim += [L]  # one external node
                Gt = [[node_[:], set(Link_), np.array([.0,.0,.0,.0]), set(rng_rim), 0]]
                for n in node_: n.root_ += [Gt]  # includes N
            Gt_ += [Gt]
        GT_ = []
        for Gt in Gt_:  # merge Gts
            node_, link_, et, rim, mrg = Gt
            if mrg: continue
            while any(rim):  # extend node_,link_, replace rim
                ext_rim = set()
                for _L in rim:
                    G,_G = _L.nodet if _L.nodet[0] in node_ else list(reversed(_L.nodet)) # one is outside node_
                    _node_,_link_,_et,_rim,_mrg = _G.root_[-1]
                    if _mrg or _G.root_[-1] is Gt: # _G may be merged in prior loop
                        continue
                    crim = (rim | ext_rim) & _rim  # intersect with old+new rim
                    xrim = _rim - crim   # exclusive _rim
                    cV = 0  # common val
                    for __L in crim:  # common Ls
                        '''
                        Et = np.sum([lay.Et for lay in G.extH.H[:rng]]); _Et = np.sum([lay.Et for lay in _G.extH.H[:rng]])  # lower-rng surround density / node
                        v = (((Et[0]-ave*Et[2]) + (_Et[0]-ave*_Et[2])) * (__L.derH.Et[0]/ave))  # multiply by link strength? or just eval links:
                        '''
                        v = __L.derH.Et[0] - ave * __L.derH.Et[2]
                        if v > 0: cV += v
                    if cV > ave * ccoef:  # cost of merging Gts
                        _G.root_[-1][-1] = 1  # set mrg
                        ext_rim.update(xrim)  # add new links
                        for _node in _node_:
                            if _node not in node_:
                                _node.root_[-1] = [Gt]; node_.add(_node)
                        link_.update(_link_|{_L}) # external L
                        et += _L.derH.Et + _et
                rim = ext_rim
            GT_ += [Gt]
        Gt__ += [GT_]
    n__ = []
    for rng, Gt_ in enumerate(Gt__, start=1):  # selective convert Gts to CGs
        if not isinstance(Gt_[0], list): continue  # not clustered
        n_ = []
        for node_,link_,et,_,_ in Gt_:
            if et[0] > et[2] * ave * rng:  # additive rng Et
                n_ += [sum2graph(root, [list(node_),list(link_),et], fd, rng)]
            else:  # weak Gt
                n_ += [[node_,link_,et]]  # skip in current-agg xcomp, unpack if extended lower-agg xcomp
        n__ += [n_]
    N__[:] = n__  # replace Ns with Gts, if any

def get_rim(N, fd, rng):

    rim_ = N.rimt_ if fd else N.rim_
    if len(rim_) < rng: return set()  # empty lrim
    else:
        lrim = set([Lt[0] for Lt in (rim_[rng-1][0]+rim_[rng-1][1] if fd else rim_[rng-1])
                    if Lt[0].derH.Et[0] > ave * Lt[0].derH.Et[2] * rng])
        return lrim


def sum2graph(root, grapht, fd, rng):  # sum node and link params into graph, aggH in agg+ or player in sub+

    N_, L_, Et = grapht  # [node_, link_, Et]
    # flattened N__, L__ if segment / rng++
    graph = CG(fd=fd, root_ = root, node_=N_, link_=L_, rng=rng)  # root_ will be updated to list roots in rng_node later?
    yx = [0,0]
    lay0 = CH(node_= N_)  # comparands, vs. L_: summands?
    for link in L_:  # unique current-layer mediators: Ns if fd else Ls
        graph.S += link.S
        graph.A = np.add(graph.A,link.A)  # np.add(graph.A, [-link.angle[0],-link.angle[1]] if rev else link.angle)
        lay0.add_H(link.derH) if lay0 else lay0.append_(link.derH)
    graph.derH.append_(lay0)  # empty for single-node graph
    derH = CH()
    for N in N_:
        graph.n += N.n  # +derH.n
        graph.area += N.area
        graph.box = extend_box(graph.box, N.box)
        yx = np.add(yx, N.yx)
        if N.derH: derH.add_H(N.derH)  # derH.Et=Et?
        if isinstance(N,CG):
            graph.mdLay.add_md_C(N.mdLay)
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