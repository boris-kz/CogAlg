import numpy as np
from copy import deepcopy, copy
from itertools import combinations, zip_longest
from .slice_edge import comp_angle, CsliceEdge
from .comp_slice import comp_slice, comp_latuple, add_lat, aves
from utils import extend_box
from frame_blobs import CBase

'''
Blob edges may be represented by higher-composition patterns, etc., if top param-layer match,
in combination with spliced lower-composition patterns, etc, if only lower param-layers match.
This may form closed edge patterns around flat core blobs, which defines stable objects. 

Graphs (predictive patterns) are formed from edges that match over < extendable max distance, 
then internal cross-comp rng/der is incremented per relative M/D: induction from prior cross-comp
(no lateral prediction skipping: it requires overhead that can only be justified in vertical feedback) 
- 
Primary value is match, diff.patterns borrow value from proximate match patterns, canceling their projected match. 
Thus graphs are assigned adjacent alt-fork graphs, to which they lend predictive value.
But alt match patterns borrow already borrowed value, which may be too tenuous to track, we use average borrowed value.
-
Clustering criterion is also M|D, summed across >ave vars if selective comp  (<ave vars are not compared, don't add costs).
Clustering is exclusive per fork,ave, with fork selected per var| derLay| aggLay 
Fuzzy clustering is centroid-based only, connectivity-based clusters will merge.
Param clustering if MM, along derivation sequence, or centroid-based if global?

There are concepts that include same matching vars: size, density, color, stability, etc, but in different combinations.
Weak value vars are combined into higher var, so derivation fork can be selected on different levels of param composition.
Clustering by variance: lend|borrow, contribution or counteraction to similarity | stability, such as metabolism? 
-
graph G:
Agg+ cross-comps top Gs and forms higher-order Gs, adding up-forking levels to their node graphs.
Sub+ re-compares nodes within Gs, adding intermediate Gs, down-forking levels to root Gs, and up-forking levels to node Gs.
-
Generic graph is a dual tree with common root: down-forking elements and up-forking clusters of this graph. 
This resembles a neuron, which has dendritic tree as input and axonal tree as output. 
But we have recursively structured param sets packed in each level of these trees, which don't exist in neurons.

Diagrams: 
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/agg_recursion_unfolded.drawio.png
'''
ave = 3
ave_d = 4
ave_L = 4
max_dist = 2
ccoef  = 10  # scaling match ave to clustering ave

class CH(CBase):  # generic derivation hierarchy of variable nesting: extH | derH, their layers and sublayers

    name = "H"
    def __init__(He, node_=None, md_t=None, n=0, Et=None, Rt=None, H=None, root=None, i=None, i_t=None):
        super().__init__()
        He.node_ = [] if node_ is None else node_  # concat bottom nesting order, may be redundant to G.node_
        He.md_t = [] if md_t is None else md_t  # derivation layer in H: [mdlat,mdLay,mdext]
        He.H = [] if H is None else H  # nested derLays | md_ in md_C, empty in bottom layer
        He.n = n  # total number of params compared to form derH, to normalize comparands
        He.Et = np.array([.0,.0,.0,.0]) if Et is None else Et  # evaluation tuple: valt, rdnt
        He.root = None if root is None else root  # N or higher-composition He
        He.i = 0 if i is None else i  # lay index in root.H, to revise rdn
        He.i_t = [[],[]] if i_t is None else i_t  # priority indices to compare node H by m | link H by d
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

    def update_root(HE, He):

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

        return CH(H=derLay, Et=np.array([vm,vd,rm,rd]), n=1)

    def comp_H(_He, He, rn=1, dir=1):  # unpack each layer of CH down to numericals and compare each pair

        der_md_t = []; Et = np.array([.0,.0,.0,.0])

        for _md_C, md_C in zip(_He.md_t, He.md_t):  # default per layer
            # lay: [mdlat, mdLay, mdext]
            der_md_C = _md_C.comp_md_C(md_C, rn=1, dir=dir)
            der_md_t += [der_md_C]
            Et += der_md_C.Et
        DLay = CH( node_=_He.node_+He.node_, md_t = der_md_t, Et=Et, n=2.5)

        # H=[] if bottom or deprecated layer, comp node_?:
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

        He.i_t[fd] = i_  # comp_H priority indices: node/m | link/d

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
                    _He.md_t += [CH().copy(md_) for md_ in He.md_t]  # can't deepcopy CH.root
                elif attr == "node_":
                    _He.node_ = copy(He.node_)
                else:
                    setattr(_He, attr, deepcopy(value))
        return _He

class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster

    def __init__(G, root_= None, node_=None, link_=None, Et=None, latuple=None, mdLay=None, derH=None, extH=None,
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
        G.lrim_ = []  # +ve only
        G.nrim_ = []
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
        G.it = ([None,None])  # graph indices in root node_s, implicitly nested
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
        l.root_ = [] if root_ is None else root_
        l.nodet = [] if nodet is None else nodet  # e_ in kernels, else replaces _node,node: not used in kernels
        l.A = [0,0] if A is None else A  # dy,dx between nodet centers
        l.S = 0 if S is None else S  # span: distance between nodet centers, summed into sparsity in CGs
        l.area = 0  # sum nodet
        l.box = [] if box is None else box  # sum nodet
        l.derH = CH(root=l) if derH is None else derH
        l.H_ = [] if H_ is None else H_  # if agg++| sub++?
        l.Vt = [0,0]  # for rim-overlap modulated segmentation, init derH.Et[:2]
        l.n = 1  # min(node_.n)
        l.lrim_ = []
        l.nrim_ = []
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


def agg_recursion(root, Q, fd):  # breadth-first rng++ cross-comp -> eval cluster, fd recursion

    N__,L__,Et,rng = rng_link_(Q) if fd else rng_node_(Q)  # cross-comp edge|frame PP_) node_
    m,d,mr,dr = Et
    fvd = d > ave_d * dr*(rng+1); fvm = m > ave * mr*(rng+1)  # op eval/ V-rdn, result eval/ V
    if fvd or fvm:
        L_ = [L for L_ in L__ for L in L_]  # root += L.derH:
        if fd: root.derH.append_(CH().append_(CH().copy(L_[0].derH)))  # new rngLay, aggLay
        else:  root.derH.H[-1].append_(L_[0].derH)  # append last aggLay
        for L in L_[1:]:
            root.derH.H[-1].H[-1].add_H(L.derH)  # accum Lay
        # rng_link_:
        if fvd and len(L_) > ave_L:  # comp L, sub-cluster by dL: mL is redundant to mN?
            set_attrs(L_,root)
            agg_recursion(root, L_, fd=1)  # appends last aggLay, L_=lG_ if segment
        # rng_node_:
        if fvm and len(N__[0]) > ave_L:  # cluster ave_L != xcomp ave_L?
            cluster_N__(root, N__, fd)  # cluster rngLays in root.node_?
            for N_ in N__:  # replace root.node_ with nested H of graphs
                if len(N_) > ave_L:
                    agg_recursion(root, N_, fd=0)  # adds higher aggLay / recursive call
'''
     if flat derH:
        root.derH.append_(CH().copy(L_[0].derH))  # init
        for L in L_[1:]: root.derH.H[-1].add_H(L.derH)  # accum
'''

def rng_node_(_N_):  # rng+ forms layer of rim_ and extH per N, appends N__,L__,Et, ~ graph CNN without backprop

    _Nt_,N__,L__, ET = [],[],[], np.array([.0,.0,.0,.0])

    # form _Nt_: prior N pair + co-positionals:
    for _G, G in combinations(_N_, r=2):
        radii = G.aRad + _G.aRad
        dy,dx = np.subtract(_G.yx,G.yx)
        dist = np.hypot(dy,dx)
        _Nt_ += [[_G,G, dy,dx, radii,dist]]

    icoef = .5  # internal M proj_val / external M proj_val
    rng = 1  # for clarity, redundant to len N__
    while True:  # if prior loop vM
        Nt_,N_,L_, Et = [],set(),[], np.array([.0,.0,.0,.0])
        for Nt in _Nt_:
            _G,G, dy,dx, radii, dist = Nt
            if _G.nrim_ and G.nrim_ and any([_nrim & nrim for _nrim,nrim in zip(_G.nrim_,G.nrim_)]):  # skip indirectly connected Gs, no direct match priority?
                continue
            M = (_G.mdLay.Et[0]+G.mdLay.Et[0]) *icoef**2 + (_G.derH.Et[0]+G.derH.Et[0])*icoef + (_G.extH.Et[0]+G.extH.Et[0])
            # comp if < max distance of likely matches *= prior G match * radius:
            if dist < max_dist * (radii*icoef**3) * M:
                Link = CL(nodet=[_G,G], S=2, A=[dy,dx], box=extend_box(G.box,_G.box))
                comp_N(Link, rng)
                et = Link.derH.Et
                Et += et; L_ += [Link]  # include -ve links
                if et[0] > ave * et[2] * (rng+1):  # eval to extend search
                    N_.update({_G,G})  # to extend search
            else: Nt_ += [Nt]
        if Et[0] > ave * Et[2]:  # current loop vM
            N__ += [N_]; L__ += [L_]; ET += Et
            rng += 1  # sub-cluster / rng N_
            _Nt_ = [Nt for Nt in Nt_ if (Nt[0] in N_ or Nt[1] in N_)]
            # re-evaluate not-compared pairs with one incremented N.M
        else:
            break
    return N__,L__,ET,rng

def rng_link_(iL_):  # comp CLs: der+'rng+ in root.link_ rim_t node rims: directional and node-mediated link tracing

    _N_t_ = [[[L.nodet[0]],[L.nodet[1]]] for L in iL_]  # Ns are rim-mediating nodes, starting with L.nodet
    L__,LL__, ET = [],[], np.array([.0,.0,.0,.0])  # all links between Ls in potentially extended L__
    rng = 1; _L_ = iL_[:]
    while True:
        L_,LL_,Et = [],[],np.array([.0,.0,.0,.0])
        N_t_ = [[[],[]] for _ in _L_]  # new rng lay of mediating nodes, traced from all prior layers?
        for L, _N_t, N_t in zip(_L_, _N_t_, N_t_):
            for rev, _N_, N_ in zip((0,1), _N_t, N_t):
                # comp L,_L mediated by nodets, flatten rim_, not only 1st layer in rimt_?
                rim_ = [rim for n in _N_ for rim in (n.rim_ if isinstance(n, CG) else [n.rimt_[0][0] + n.rimt_[0][1]])]
                for rim in rim_:
                    for _L,_rev in rim:  # _L is reversed relative to its 2nd node
                        if _L is L or _L in L.visited_: continue
                        if _L not in iL_: set_attrs([_L],_L_[0].root_[-1])
                        L.visited_ += [_L]; _L.visited_ += [L]
                        Link = CL(nodet=[_L,L], S=2, A=np.subtract(_L.yx,L.yx), box=extend_box(_L.box, L.box))
                        comp_N(Link, rng, dir = 1 if (rev^_rev) else -1)  # d = -d if one L is reversed
                        # L.rim_t += Link, order: nodet < L < rimt_, mN.rim || L
                        et = Link.derH.Et
                        Et += et; LL_ += [Link]  # include -ve links
                        if et[0] > ave * et[2] * (rng+1):  # eval to extend search
                            N_ += _L.nodet  # get _Ls in N_ rims
                            if _L not in _L_:
                                _L_ += [_L]; N_t_ += [[[],[]]]  # not in root
                            L_ += [_L]
                            N_t_[_L_.index(_L)][1-rev] += L.nodet  # rng+ -mediating nodes
        if L_:
            L__ += [L_]; LL__ += [LL_]; ET += Et
            V = 0; L_,_N_t_ = [],[]
            for L, N_t in zip(_L_,N_t_):
                if any(N_t):
                    L_ += [L]; _N_t_ += [N_t]
                    V += L.derH.Et[0] - ave * L.derH.Et[2] * rng
            if V > 0:  # rng+ if vM of extended N_t_
                _L_ = L_; rng += 1
            else: break
        else:
            break
    return L__, LL__, ET, rng

def comp_N(Link, rng, dir=None):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = dir is not None  # compared links have binary relative direction
    dir = 1 if dir is None else dir  # convert from None into numeric
    _N, N = Link.nodet
    _L,L = (2,2) if fd else (len(_N.node_),len(N.node_)); _S,S = _N.S,N.S; _A,A = _N.A,N.A
    A = [d * dir for d in A]  # reverse angle direction if N is left link
    rn = _N.n / N.n
    mdext = comp_ext(_L,L, _S,S/rn, _A,A)
    md_t = [mdext]; Et = mdext.Et.copy(); n = mdext.n
    if not fd:  # CG
        H, lEt, ln = comp_latuple(_N.latuple,N.latuple,rn,fagg=1)
        mdlat = CH(H=H, Et=lEt, n=ln)
        mdLay = _N.mdLay.comp_md_C(N.mdLay, rn, dir)
        md_t += [mdlat,mdLay]; Et += lEt + mdLay.Et; n += ln + mdLay.n
    # | n = (_n+n)/2?
    elay = CH( H=[CH(n=n, md_t=md_t, Et=Et)], n=n, md_t=[CH().copy(md_) for md_ in md_t], Et=copy(Et))
    if _N.derH and N.derH:
        dderH = _N.derH.comp_H(N.derH, rn, dir=dir)  # comp shared layers
        elay.append_(dderH, flat=1)
    # spec: comp node_,link_ by rng_node_?
    Link.derH = elay; elay.root = Link; Link.n = min(_N.n,N.n); Link.nodet = [_N,N]; Link.yx = np.add(_N.yx,N.yx) /2
    # prior S,A
    for rev, node in zip((0,1),(N,_N)):  # reverse Link direction for N
        if elay.Et[0] > ave:  # for bottom-up segment:
            if len(node.lrim_) < rng:  # add +ve layer
                node.extH.append_(elay); node.lrim_ += [{Link}]; node.nrim_ += [{(_N,N)[rev]}]  # _node
            else:  # append last layer
                node.extH.H[-1].add_H(elay); node.lrim_[-1].add(Link); node.nrim_[-1].add((_N,N)[rev])
        # include negative links to form L_:
        if (len(node.rimt_) if fd else len(node.rim_)) < rng:
            if fd: node.rimt_ = [[[[Link,rev]],[]]] if dir else [[[],[[Link,rev]]]]  # add rng layer
            else:  node.rim_ += [[[Link, rev]]]
        else:
            if fd: node.rimt_[-1][1-rev] += [[Link,rev]]  # add in last rng layer, opposite to _N,N dir
            else:  node.rim_[-1] += [[Link, rev]]

def comp_ext(_L,L,_S,S,_A,A):  # compare non-derivatives:

    dL = _L - L; mL = min(_L,L) - ave_L  # direct match
    dS = _S/_L - S/L; mS = min(_S,S) - ave_L  # sparsity is accumulated over L
    mA, dA = comp_angle(_A,A)  # angle is not normalized
    M = mL + mS + mA
    D = abs(dL) + abs(dS) + abs(dA)  # normalize relative to M, signed dA?

    return CH(H=[mL,dL, mS,dS, mA,dA], Et=np.array([M,D,M>D,D<=M]), n=0.5)

def cluster_from_G(G, _nrim, _lrim, rng=0):

    node_, link_, Et = {G}, set(), np.array([.0,.0,.0,.0])  # m,r only?
    while _lrim:
        nrim, lrim = set(), set()
        for _G,_L in zip(_nrim, _lrim):
            if _G.merged or len(_G.lrim_) < rng+1:
                continue
            for g in node_:  # compare external _G to all internal nodes, include if any of them match
                if len(g.lrim_) < rng + 1: continue
                L = next(iter(g.lrim_[rng] & _G.lrim_[rng]), None)  # intersect = [+link] | None
                if L:
                    if ((g.extH.Et[0]-ave*g.extH.Et[2]) + (_G.extH.Et[0]-ave*_G.extH.Et[2])) * (L.derH.Et[0]/ave) > ave * ccoef:
                        # rng+: merge roots:
                        if isinstance(_G.root_,list):
                            _node_,_link_,_Et,_merged = _G.root_[-1]
                            if _merged: continue
                            node_.update(_node_)
                            link_.update(_link_| {_L})  # L was external
                            Et += _L.derH.Et + _Et
                            for n in _node_: n.merged = 1
                            _G.root_[-1][3] = 1
                        else:  # rng=1: add Ns
                            node_.add(_G); link_.add(_L); Et += _L.derH.Et
                        nrim.update(set(_G.nrim_[rng]) - node_)
                        lrim.update(set(_G.lrim_[rng]) - link_)
                        _G.merged = 1
                        break
        _nrim,_lrim = nrim, lrim
    return node_, link_, Et

def cluster_N__(root, iN__, fd):  # cluster G__|L__ by value density of +ve links per node

    # rng=1: cluster connected Gs into Gts
    for G in iN__[0]: G.merged = 0
    N_, _re_N_ = [],[]
    for G in iN__[0]:
        if G.merged: continue  # is in prior Gt node_
        if not G.nrim_:
            N_.append(G); continue
        node_, link_, Et = cluster_from_G(G, G.nrim_[0], G.lrim_[0])
        if Et[0] > Et[2] * ave:
            Gt = [node_, link_, Et, 0]
            _re_N_.append(Gt)
            N_.append(Gt)
            for n in node_: n.root_ = [Gt]
    N__ = [N_]
    rng = 1
    # rng+: merge Gts connected via G.lrim_[rng] in their node_s into higher Gts
    while True:
        re_N_ = []
        for G in set.union(*iN__[:rng+1]): G.merged = 0  # reset all lower Gs
        for _node_,_link_,_Et,_merged in _re_N_:  # Gt
            if _merged: continue
            Node_, Link_, ET = set(),set(), np.array([.0,.0,.0,.0])  # m,r only?
            for G in _node_:
                if not G.merged and len(G.nrim_) > rng:
                    node_ = G.nrim_[rng]- Node_
                    if not node_: continue  # no new rim nodes
                    node_,link_,Et = cluster_from_G(G, node_, G.lrim_[rng]-Link_, rng)
                    Node_.update(node_)
                    Link_.update(link_)
                    ET += Et
            if ET[0] > ET[2] * ave:  # additive current-layer V: form higher Gt
                Node_.update(_node_); Link_.update(_link_)
                Gt = [Node_, Link_, ET+_Et, 0]
                for n in Node_:
                    if isinstance(n.root_,list): n.root_.append(Gt)
                    else: n.root_ = [Gt]
                re_N_.append(Gt)
        if re_N_:
            N__.append(re_N_)
            _re_N_ = re_N_
            rng += 1
        else:
            break
    for i, N_ in enumerate(N__):  # convert Gts to CGs
        for ii, N in enumerate(N_):
            if isinstance(N, list):
                N_[ii] = sum2graph(root, [list(N[0]), list(N[1]), N[2]], fd, rng=i)
    iN__[:] = N__

def set_attrs(Q, root):

    for e in Q:
        e.visited_ = []
        if isinstance(e, CL):
            e.rimt_ = []  # nodet-mediated links, same der order as e
            e.root_ = [root]
        if hasattr(e,'extH'): e.derH.append_(e.extH)  # no default CL.extH
        else: e.extH = CH()  # set in sum2graph
        e.aRad = 0
    return Q

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