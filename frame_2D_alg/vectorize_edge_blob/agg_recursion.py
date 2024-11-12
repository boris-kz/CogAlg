import sys
sys.path.append("..")
from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_
from slice_edge import slice_edge, comp_angle, aveG
from comp_slice import comp_slice, comp_latuple, add_lat, aves, comp_md_, add_md_
from itertools import combinations, zip_longest
from copy import deepcopy, copy
import numpy as np

'''
This code is initially for clustering segments within edge: high-gradient blob, but it's far too complex for that.
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
        # He.fd = 0 if fd is None else fd  # 0: sum CGs, 1: sum CLs
        # He.ni = 0  # exemplar in node_, trace in both directions?
        # He.deep = 0 if deep is None else deep  # nesting in root H
        # He.nest = 0 if nest is None else nest  # nesting in H
    def __bool__(H): return H.n != 0

    def accum_lay(HE, He, irdnt=[]):

        if HE.md_t:
            for MD_, md_ in zip(HE.md_t, He.md_t):  # mdLat, mdLay, mdExt
                add_md_(MD_,md_)
            HE.Et += He.Et; HE.n += He.n  # combined param accum span
            if any(irdnt):
                HE.Et[2:] = [E+e for E,e in zip(HE.Et[2:], irdnt)]
        else:
            HE.md_t = deepcopy(He.md_t)
        HE.Et += He.Et; HE.n += He.n
        if any(irdnt):
            HE.Et[2:] = [E+e for E,e in zip(HE.Et[2:], irdnt)]

    def add_H(HE, He_, irdnt=[], root=None, ri=None):  # unpack derHs down to numericals and sum them

        if not isinstance(He_,list): He_ = [He_]
        for He in He_:
            if HE:
                for i, (Lay,lay) in enumerate(zip_longest(HE.H, He.H, fillvalue=None)):  # cross comp layer
                    if lay:
                        if Lay: Lay.add_H(lay, irdnt)
                        else:
                            if Lay is None: HE.append_(lay.copy_(root=HE))  # pack a copy of new lay in HE.H
                            else:           HE.H[i] = lay.copy_(root=HE)  # Lay was []
                HE.accum_lay(He, irdnt)
                HE.node_ += [node for node in He.node_ if node not in HE.node_]  # node_ is empty in CL derH?
            # not sure:
            elif root:
                if ri is None: root.derH = He.copy_(root=root)
                else:          root.H[ri]= He.copy_(root=root)
            else:
                HE = He.copy_(root=root)
        return HE

    def append_(HE,He, irdnt=None, flat=0):

        if irdnt is None: irdnt = []
        if flat:
            for i, lay in enumerate(He.H):  # different refs for L.derH and root.derH.H:
                if lay:
                    lay = lay.copy_(root=HE); lay.i = len(HE.H)+i
                HE.H += [lay]  # may be empty to trace forks
        else:
            He.i = len(HE.H); He.root = HE; HE.H += [He]  # He can't be empty
        HE.accum_lay(He, irdnt)

        return HE

    def copy_(_He, root, rev=0):  # comp direction may be reversed
        He = CH(root=root, node_=copy(_He.node_), Et=copy(_He.Et), n=_He.n, i=_He.i, i_=copy(_He.i_))

        for H,Et,n in _He.md_t:  # mdLat, mdLay, mdExt
            H = copy(H)
            if rev: H[1::2] *= -1  # negate ds
            He.md_t += [[H, copy(Et), n]]
        for he in _He.H:
            He.H += [he.copy_(root=He, rev=rev)] if he else [[]]
        return He

    def comp_H(_He, He, rn=1, dir=1):  # unpack each layer of CH down to numericals and compare each pair

        der_md_t = []; Et = np.array([.0,.0,.0,.0])
        for _md_, md_ in zip(_He.md_t, He.md_t):  # [mdlat, mdLay, mdext], default per layer

            der_md_ = comp_md_(_md_[0], md_[0], rn=1, dir=dir)
            der_md_t += [der_md_]
            Et += der_md_[1]

        DLay = CH(md_t = der_md_t, Et=Et, n=2.3)  # .3 added from comp ext, empty node_: no comp node_
        # empty H in bottom | deprecated layer:
        for rev, _lay, lay in zip((0,1), _He.H, He.H):  #  fork & layer CH / rng+|der+, flat
            if _lay and lay:
                dLay = _lay.comp_H(lay, rn, dir)  # comp He.md_t, comp,unpack lay.H
                DLay.append_(dLay)  # DLay.H += subLay
            else:
                l = _lay if _lay else lay  # only one not empty lay = difference between lays:
                if l: DLay.append_(l.copy_(root=DLay, rev=rev))
                else: DLay.H += [[]]  # to trace fork types
            # nested subHH ( subH?
        return DLay

    # not implemented yet:
    def sort_H(He, fd):  # re-assign rdn and form priority indices for comp_H, if selective and aligned

        i_ = []  # priority indices
        for i, lay in enumerate(sorted(He.H, key=lambda lay: lay.Et[fd], reverse=True)):
            di = lay.i - i  # lay index in H
            lay.Et[2+fd] += di  # derR- valR
            i_ += [lay.i]
        He.i_ = i_  # comp_H priority indices: node/m | link/d
        if not fd:
            He.root.node_ = He.H[i_[0]].node_  # no He.node_ in CL?

class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster

    def __init__(G, root_= None, node_=None, link_=None, latuple=None, mdLay=None, derH=None, extH=None, rng=1, fd=0, n=0, box=None, yx=None, minL=None):
        super().__init__()
        G.n = n  # last layer?
        G.fd = 0 if fd else fd  # 1 if cluster of Ls | lGs?
        G.rng = rng
        G.root_ = None if root_ is None else root_  # convert to root_ in cluster_N_: same nodes in higher rng layers
        G.node_ = [] if node_ is None else node_ # convert to GG_ or node_H in agg++
        G.link_ = [] if link_ is None else link_ # internal links per comp layer in rng+, convert to LG_ in agg++
        G.minL = 0 if minL is None else minL  # min link.dist in subG
        G.latuple = [0,0,0,0,0,[0,0]] if latuple is None else latuple  # lateral I,G,M,Ma,L,[Dy,Dx]
        G.mdLay = [[.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0], [.0,.0,.0,.0], 0] if mdLay is None else mdLay  # H here is md_latuple
        # maps to node_H / agg+|sub+:
        G.derH = CH(root=G) if derH is None else derH  # sum from nodes, then append from feedback
        G.extH = CH(root=G) if extH is None else extH  # sum from rim_ elays, H maybe deleted
        G.rim = []  # flat links of any rng, may be nested in clustering
        G.aRad = 0  # average distance between graph center and node center
        G.box = [np.inf, np.inf, -np.inf, -np.inf] if box is None else box  # y0,x0,yn,xn
        G.yx = [0,0] if yx is None else yx  # init PP.yx = [(y0+yn)/2,(x0,xn)/2], then ave node yx
        G.alt_graph_ = []  # adjacent gap+overlap graphs, vs. contour in frame_graphs
        G.visited_ = []
        # G.subG_ = [] if subG_ is None else subG_ # lower-rng Gs, nest in node_?
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

def vectorize_root(frame):

    blob_ = unpack_blob_(frame)
    for blob in blob_:
        if not blob.sign and blob.G > aveG * blob.root.rdn:
            edge = slice_edge(blob)
            if edge.G * (len(edge.P_) - 1) > ave:  # eval PP, rdn=1
                comp_slice(edge)
                if edge.mdLay[1][0] * (len(edge.node_)-1)*(edge.rng+1) > ave * edge.mdLay[1][2]:
                    edge.derH = CH()
                    if not hasattr(frame, 'derH'):
                        frame.derH = CH(); frame.root = None
                    G_ = []  # init for agg+:
                    for N in edge.node_:  # no comp node_, link_ | PPd_ for now
                        H,Et,n = N[3] if isinstance(N,list) else N.mdLay  # N is CP
                        if H and Et[0] > ave * Et[2]:  # convert PP|P to G:
                            if isinstance(N,list):
                                root_, P_,link_,(H,Et,n),lat,A,S,area,box,[y,x],n = N  # PPt
                            else:  # single CP
                                root_=edge; P_=[N]; link_=[]; (H,Et,n)=N.mdLay; lat=N.latuple; [y,x]=N.yx; n=N.n
                                box = [y, x-len(N.dert_), y,x]
                            PP = CG(fd=0, root_=[root_], node_=P_,link_=link_,mdLay=[H, Et, n],latuple=lat, box=box,yx=[y,x],n=n)
                            y0,x0,yn,xn = box
                            PP.aRad = np.hypot(*np.subtract(PP.yx,(yn,xn)))
                            G_ += [PP]
                    if len(G_) > ave_L:
                        edge.node_ = G_
                        agg_recursion(edge)  # discontinuous PP_ cross-comp, cluster

def agg_recursion(root):  # breadth-first node_-> L_ cross-comp, clustering, recursion

    def comp_Q(Q, fd):  # cross-comp node_ or L_
        for e in Q: e.extH = CH()

        N_,L_,Et,rng = comp_link_(Q) if fd else comp_node_(Q)
        m, d, mr, dr = Et
        fvm = m > ave * mr*(rng+1)
        fvd = d * (m/ (ave+m)) > ave_d * dr*(rng+1)
        # vd is borrowed from vm: *= rel_m, but it's also a complementary, looks redundant?
        return N_,L_, fvm,fvd

    def cluster_eval(root, N_, fd):

        pL_ = {l for n in N_ for l,_ in get_rim(n, fd)}
        if len(pL_) > ave_L:
            G_ = cluster_N_(root, pL_, fd)  # optionally divisive clustering
            if len(G_) > ave_L:
                if isinstance(root.node_[0],CG): root.node_ = [root.node_, G_]  # init node_ hierarchy
                else: root.node_ += [G_]
                agg_recursion(root)  # cross-comp clustered nodes

    N_,L_,fvm,fvd = comp_Q(root.node_ if isinstance(root.node_[0],CG) else root.node_[-1], fd=0)
    # predict_val = proj m - proj d: vd is borrowed from falsely projected vm, if any:
    if fvm:
        root.derH.append_(CH().add_H([L.derH for L in L_]))  # mfork, else no new layer
        if fvd:
            lN_,lL_,_,_ = comp_Q(L_,fd=1)  # comp new L_, root.link_ was compared in root-forming for alt clustering
            root.derH.append_(CH().add_H([L.derH for L in lL_]))  # dfork
            # recursive der+ eval_cost > ave_match, add by feedback if < _match?
        else:
            root.derH.H += [[]]  # empty to decode rng+|der+, n forks per layer = 2^depth
        # after feedback because clustering adds deeper derH lays, unless fork-specific?:
        if fvm: cluster_eval(root, N_, fd=0)
        if fvd: cluster_eval(root, lN_, fd=1)


def comp_node_(_N_):  # rng+ forms layer of rim and extH per N, appends N_,L_,Et, ~ graph CNN without backprop

    _Gp_ = []  # [G pair + co-positionals]
    for _G, G in combinations(_N_, r=2):
        rn = _G.n / G.n
        if rn > ave_rn: continue  # scope disparity
        radii = G.aRad + _G.aRad
        dy,dx = np.subtract(_G.yx,G.yx)  # i'm getting a same yx from different Gs, but each of the G has different node_
        dist = np.hypot(dy,dx)
        _G.add, G.add = 0, 0
        _Gp_ += [(_G,G, rn, dy,dx, radii, dist)]
    icoef = .1  # internal M proj_val / external M proj_val
    rng = 1  # len N__
    N_,L_,ET = set(),[], np.array([.0,.0,.0,.0])
    while True:  # prior vM
        Gp_,Et = [], np.array([.0,.0,.0,.0])
        for Gp in _Gp_:
            _G,G, rn, dy,dx, radii, dist = Gp
            _nrim = {L.nodet[1] if L.nodet[0] is _G else L.nodet[0] for L,_ in _G.rim}
            nrim = {L.nodet[1] if L.nodet[0] is G else L.nodet[0] for L,_ in G.rim}
            if _nrim & nrim:  # indirectly connected Gs,
                continue     # no direct match priority?
            M = (_G.mdLay[1][0]+G.mdLay[1][0]) *icoef**2 + (_G.derH.Et[0]+G.derH.Et[0])*icoef + (_G.extH.Et[0]+G.extH.Et[0])
            # comp if < max distance of likely matches *= prior G match * radius:
            if dist < max_dist * (radii * icoef**3) * M:
                Link = CL(nodet=[_G,G], angle=[dy,dx], dist=dist, box=extend_box(G.box,_G.box))
                et = comp_N(Link, rn, rng)
                L_ += [Link]  # include -ve links
                if et is not None:
                    N_.update({_G, G})
                    Et += et; _G.add,G.add = 1,1
            else:
                Gp_ += [Gp]  # re-evaluate not-compared pairs with one incremented N.M
        ET += Et
        if Et[0] > ave * Et[2]:  # current-rng vM
            rng += 1
            _Gp_ = [Gp for Gp in Gp_ if Gp[0].add or Gp[1].add]  # one incremented N.M
        else:  # low projected rng+ vM
            break

    return  list(N_),L_, ET, rng  # flat N__ and L__

def comp_link_(iL_):  # comp CLs via directional node-mediated link tracing: der+'rng+ in root.link_ rim_t node rims

    fd = isinstance(iL_[0].nodet[0], CL)
    for L in iL_:
        L.mL_t, L.rimt, L.aRad, L.visited_ = [[],[]], [[],[]], 0, [L]
        # init mL_t (mediated Ls) per L:
        for rev, n, mL_ in zip((0,1), L.nodet, L.mL_t):
            for _L,_rev in n.rimt[0]+n.rimt[1] if fd else n.rim:
                if _L is not L and _L.derH.Et[0] > ave * _L.derH.Et[2]:
                    mL_ += [(_L, rev ^_rev)]  # the direction of L relative to _L
    _L_, out_L_, LL_, ET = iL_,set(),[], np.array([.0,.0,.0,.0])
    med = 1
    while True:  # xcomp _L_
        L_, Et = set(), np.array([.0,.0,.0,.0])
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
                        out_L_.update({_L,L}); L_.update({_L,L}); Et += et
        if not any(L_): break
        # extend mL_t per last med_L
        ET += Et; Med = med + 1
        if Et[0] > ave * Et[2] * Med:  # project prior-loop value - new cost
            _L_, _Et = set(), np.array([.0,.0,.0,.0])
            for L in L_:
                mL_t, lEt = [set(),set()], np.array([.0,.0,.0,.0])  # __Ls per L
                for mL_,_mL_ in zip(mL_t, L.mL_t):
                    for _L, rev in _mL_:
                        for _rev, n in zip((0,1), _L.nodet):
                            rim = n.rimt if fd else n.rim
                            if len(rim) == med:  # append in comp loop
                                for __L,__rev in rim[0]+rim[1] if fd else rim:
                                    if __L in L.visited_ or __L not in iL_: continue
                                    L.visited_ += [__L]; __L.visited_ += [L]
                                    et = __L.derH.Et
                                    if et[0] > ave * et[2] * Med:  # /__L
                                        mL_.add((__L, rev ^_rev ^__rev))  # combine reversals: 2 * 2 mLs, but 1st 2 are pre-combined
                                        lEt += et
                if lEt[0] > ave * lEt[2] * Med:  # rng+/ L is different from comp/ L above
                    L.mL_t = mL_t; _L_.add(L); _Et += lEt
            # refine eval:
            if _Et[0] > ave * _Et[2] * Med:
                med = Med
            else:
                break
        else:
            break
    return out_L_, LL_, ET, med  # =rng

def extend_box(_box, box):  # add 2 boxes
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return min(y0, _y0), min(x0, _x0), max(yn, _yn), max(xn, _xn)

def comp_area(_box, box):
    _y0,_x0,_yn,_xn =_box; _A = (_yn - _y0) * (_xn - _x0)
    y0, x0, yn, xn = box;   A = (yn - y0) * (xn - x0)
    return _A-A, min(_A,A) - ave_L**2  # mA, dA


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
    elay = CH(H=[CH(n=n, md_t=md_t, Et=Et)], n=n, md_t=deepcopy(md_t), Et=copy(Et))
    if _N.derH and N.derH:
        dderH = _N.derH.comp_H(N.derH, rn, dir=dir)  # comp shared layers
        elay.append_(dderH, flat=1)
    elif _N.derH: elay.H += [_N.derH.copy_(root=elay)]  # one empty derH
    elif N.derH: elay.H += [N.derH.copy_(root=elay, rev=1)]
    # spec: comp_node_(node_|link_), combinatorial, node_ may be nested with rng-)agg+, use graph similarity search?
    # new lay in root G.derH:
    Link.derH = elay; elay.root = Link; Link.n = min(_N.n,N.n); Link.nodet = [_N,N]; Link.yx = np.add(_N.yx,N.yx) /2
    # preset angle, dist
    Et = elay.Et
    if Et[0] > ave * Et[2] * (rng+1):
        for rev, node in zip((0,1), (N,_N)):
            # reverse Link direction for _N
            if fd: node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            else:  node.rim += [(Link, rev)]
            node.extH.add_H(Link.derH, root=node.extH)
            # flat
        return Et

def get_rim(N,fd): return N.rimt[0] + N.rimt[1] if fd else N.rim  # add nesting in cluster_N_?

def cluster_N_(root, L_, fd, nest=1):  # top-down segment L_ by >ave ratio of L.dists
    ave_rL = 1.2 # define segment and sub-cluster

    L_ = sorted(L_, key=lambda x: x.dist, reverse=True)  # lower-dist links
    _L = L_[0]
    N_, et = {*_L.nodet}, _L.derH.Et
    # current dist segment:
    for i, L in enumerate(L_[1:], start=1):  # long links first
        rel_dist = _L.dist / L.dist  # >1
        if rel_dist < ave_rL or et[0] < ave or len(L_[i:]) < ave_L:  # ~=dist Ns or either side of L is weak
            _L = L; et += L.derH.Et
            for n in L.nodet: N_.add(n)
        else:
            min_dist = _L.dist; break  # terminate contiguous-dist segment
    Gt_ = []
    for N in N_:  # cluster current dist segment
        if len(N.root_) >= nest: continue  # N is already merged
        node_, link_, et = {N}, set(), np.array([.0,.0,.0,.0])
        Gt = [node_, link_, et, min_dist]
        N.root_ += [Gt]
        _eN_ = {l.nodet[1] if l.nodet[0] is N else l.nodet[0] for l,_ in get_rim(N, fd) if l.dist >= min_dist}  # init ext Ns
        while _eN_:
            eN_ = set()
            for eN in _eN_:  # cluster rim-connected ext Ns
                if len(eN.root_) >= nest: continue  # N is already merged
                eN.root_ += [Gt]
                node_.add(eN)
                for L,_ in get_rim(eN, fd):
                    if L.dist >= min_dist and L not in link_:
                        link_.add(L)
                        eN_.update(L.nodet); et += L.derH.Et  # L.nodet root_s will be checked above?
            _eN_ = eN_
        # form subG_ from shorter-L seg per Gt, depth-first:
        sub_L_ = set()
        if isinstance(root,list): root[0].update(node_)  # add lower nodes if root is not edge
        for N in node_:
            sub_L_.update({l for l,_ in get_rim(N,fd) if l.dist < min_dist})
        if len(sub_L_) > ave_L * nest:  # this eval looks wrong?
            subG_ = cluster_N_(Gt, sub_L_, fd, nest+1)
        else: subG_ = []
        Gt += [subG_]; Gt_ += [Gt]
    G_ = []
    for Gt in Gt_:
        M, R = Gt[2][0::2]  # Gt: node_, link_, et, subG_
        if M > R * ave * nest:  # rdn incr/ lower rng
            G_ += [sum2graph(root, Gt, fd, nest)]
        else:
            for n in Gt[0]: n.root_.pop()
    return G_

def sum2graph(root, grapht, fd, nest):  # sum node and link params into graph, aggH in agg+ or player in sub+

    node_, link_, Et, minL, subG_ = grapht
    graph = CG(fd=fd, root_=[root], node_= [list(node_),subG_], link_=link_, minL=minL, rng=nest)
    # incr node_ nesting, not just 2 layers as above? also root_?
    yx = [0,0]
    graph.derH.append_(CH(node_=node_).add_H([link.derH for link in link_]))  # new der layer
    derH = CH()
    for N in node_:
        graph.n += N.n  # +derH.n, total nested comparable vars
        graph.box = extend_box(graph.box, N.box)  # pre-compute graph.area += N.area?
        yx = np.add(yx, N.yx)
        if N.derH: derH.add_H(N.derH)  # derH.Et=Et?
        if isinstance(N,CG):
            add_md_(graph.mdLay, N.mdLay)
            add_lat(graph.latuple, N.latuple)
        N.root_[-1] = graph  # replace Gt
    graph.derH.append_(derH, flat=1)  # comp(derH) forms new layer, higher layers are added by feedback
    L = len(node_)
    yx = np.divide(yx,L); graph.yx = yx
    # ave distance from graph center to node centers:
    graph.aRad = sum([np.hypot( *np.subtract(yx, N.yx)) for N in node_]) / L
    if fd:
        # assign alt graphs from d graph, after both linked m and d graphs are formed
        for node in node_:  # CG or CL
            mgraph = node.root_[-1]
            if mgraph:
                for fd, (G, alt_G) in enumerate(((mgraph,graph), (graph,mgraph))):  # bilateral assign:
                    if G not in alt_G.alt_graph_:
                        G.alt_graph_ += [alt_G]
    return graph

if __name__ == "__main__":
    image_file = '../images/raccoon_eye.jpeg'
    image = imread(image_file)

    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vectorize_root(frame)