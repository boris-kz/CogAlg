import numpy as np
from copy import deepcopy, copy
from itertools import combinations, zip_longest
from .slice_edge import comp_angle, CsliceEdge
from .comp_slice import comp_slice, comp_latuple, CH, CG
from utils import extend_box
from frame_blobs import CBase

'''
Blob edges may be represented by higher-composition patterns, etc., if top param-layer match,
in combination with spliced lower-composition patterns, etc, if only lower param-layers match.
This may form closed edge patterns around flat core blobs, which defines stable objects. 
Graphs are formed from blobs that match over < max distance. 
-
They are also assigned adjacent alt-fork graphs, which borrow predictive value from the graph. 
Primary value is match, diff.patterns borrow value from proximate match patterns, canceling their projected match. 
But alt match patterns borrow already borrowed value, which may be too tenuous to track, we use average borrowed value.
-
Clustering criterion is M|D, summed across >ave vars if selective comp (<ave vars are not compared and don't add costs).
Fork selection should be per var| der layer| agg level. Clustering is exclusive per fork,ave, overlapping between fork,aves.  
(same fork,ave fuzzy clustering is possible if centroid-based, connectivity-based clusters merge)

There are concepts that include same matching vars: size, density, color, stability, etc, but in different combinations.
Weak value vars are combined into higher var, so derivation fork can be selected on different levels of param composition.
Clustering by variance: lend|borrow, contribution or counteraction to similarity | stability, such as metabolism? 
-
graph G:
Agg+ cross-comps top Gs and forms higher-order Gs, adding up-forking levels to their node graphs.
Sub+ re-compares nodes within Gs, adding intermediate Gs, down-forking levels to root Gs, and up-forking levels to node Gs.
-
Generic graph is a dual tree with common root: down-forking input node Gs and up-forking output graph Gs. 
This resembles a neuron, which has dendritic tree as input and axonal tree as output. 
But we have recursively structured param sets packed in each level of these trees, which don't exist in neurons.

Diagrams: 
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/agg_recursion_unfolded.drawio.png
'''
ave = 3
ave_L = 4
G_aves = [4,6]  # ave_Gm, ave_Gd
max_dist = 2

class CL(CBase):  # link or edge, a product of comparison between two nodes or links
    name = "link"

    def __init__(l, nodet=None,derH=None, span=0, angle=None, box=None, md_t=None, H_=None):
        super().__init__()
        # CL = binary tree of Gs, depth+/der+: CL nodet is 2 Gs, CL + CLs in nodet is 4 Gs, etc.,
        # unpack sequentially
        l.nodet = [] if nodet is None else nodet  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.angle = [0,0] if angle is None else angle  # dy,dx between nodet centers
        l.span = span  # distance between nodet centers
        l.area = 0  # sum nodet
        l.box = [] if box is None else box  # sum nodet
        l.latuple = []  # sum nodet
        l.md_t = [] if md_t is None else md_t  # [mdlat,mdLay,mdext] per layer
        l.derH = CH(root=l) if derH is None else derH
        l.H_ = [] if H_ is None else H_  # if agg++| sub++
        l.mdext = []  # Et, Rt, Md_
        l.ft = [0,0]  # fork inclusion tuple, may replace Vt:
        l.Vt = [0,0]  # for rim-overlap modulated segmentation, init derH.Et[:2]
        l.n = 1  # min(node_.n)
        l.S = 0  # sum nodet
    def __bool__(l): return bool(l.derH.H)


def vectorize_root(image):  # vectorization in 3 composition levels of xcomp, cluster:

    frame = CsliceEdge(image).segment()
    for edge in frame.blob_:
        if hasattr(edge, 'P_') and edge.latuple[-1] * (len(edge.P_)-1) > G_aves[0]:  # eval PP, rdn=1  (ave_PPm or G_aves[0]?)
            comp_slice(edge)
            # init for agg+:
            edge.derH = CH(H=[CH()]); edge.derH.H[0].root = edge.derH
            edge.link_ = []; edge.fback_t = [[],[]]
            node_t, link_t = [[],[]], [[],[]]
            for fd, node_ in enumerate(copy(edge.node_)):  # always node_t
                if edge.mdLay.Et[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.mdLay.Et[2+fd]:
                    pruned_node_ = []
                    for PP in node_:  # PP -> G
                        if PP.mdLay and PP.mdLay.Et[fd] > G_aves[fd] * PP.mdLay.Et[2+fd]:  # v>ave*r
                            PP.root_ = []  # no feedback to edge?
                            PP.node_ = PP.P_  # revert node_
                            y0,x0,yn,xn = PP.box
                            PP.yx = [(y0+yn)/2, (x0+xn)/2]
                            PP.aRad = np.hypot(*np.subtract(PP.yx,(yn,xn)))
                            PP.Et = [0,0,0,0]  # [] in comp_slice
                            pruned_node_ += [PP]
                    if len(pruned_node_) > 10:  # discontinuous PP rng+ cross-comp, cluster -> G_t:
                        agg_recursion(edge, N_=pruned_node_, fL=0)
                        node_t[fd] = edge.node_  # edge.node_ is current fork node_t
                        link_t[fd] = edge.link_
            if any(node_t):
                edge.node_ = node_t; edge.link_ = link_t  # edge.node_ may be node_tt: node_t per fork?


def agg_recursion(root, N_, fL, rng=1):  # fL: compare node-mediated links, else < rng-distant CGs

    N_,Et,rng = rng_link_(N_) if fL else rng_node_(N_, rng)  # both rng-recursive
    if len(N_) > ave_L:
        node_t = form_graph_t(root, N_,Et,rng)  # fork Et eval, depth-first sub+, sub_Gs += fback_
        if node_t:
            for fd, node_ in zip((0,1), node_t):
                N_ = [n for n in node_ if n.derH.Et[fd] > G_aves[fd] * n.derH.Et[2+fd]]  # prune node_
                if root.derH.Et[0] * (max(0,(len(N_)-1)*root.rng)) > G_aves[1]*root.derH.Et[2]:
                    # agg+ rng+, val *= n comparands, forms CGs:
                    agg_recursion(root, N_, fL=0)
            root.node_[:] = node_t
        # else keep root.node_

def rng_node_(_N_, rng):  # forms discrete rng+ links, vs indirect rng+ in rng_kern_, still no sub_Gs / rng+

    rN_ = []; rEt = [0,0,0,0]; n = 0

    while True:
        N_, Et = rng_kern_(_N_, rng)  # += rng layer
        if Et[0] > ave * Et[2]:
            _N_ = N_
            rEt = [V+v for V,v in zip(rEt, Et)]
            if not n: rN_ = N_  # 1st pruned N_
            rng += 1
            n += 1
        else:
            break
    return rN_, rEt, rng

def rng_kern_(N_, rng):  # comp Gs summed in kernels, ~ graph CNN without backprop, not for CLs

    _G_ = []
    Et = [0,0,0,0]
    # comp_N:
    for (_G, G) in list(combinations(N_,r=2)):
        if _G in [G for visited_ in G.visited__ for G in visited_]:  # compared in any rng++
            continue
        dy,dx = np.subtract(_G.yx,G.yx)
        dist = np.hypot(dy,dx)
        aRad = (G.aRad+_G.aRad) / 2  # ave G radius
        # eval relative distance between G centers:
        if dist / max(aRad,1) <= max_dist * rng:
            for _g,g in (_G,G),(G,_G):
                if len(g.extH.H)==rng: g.visited__[-1] += [_g]
                else: g.visited__ += [[_g]]  # init layer
            Link = CL(nodet=[_G,G], span=2, angle=[dy,dx], box=extend_box(G.box,_G.box))
            if comp_N(Link, Et, rng):
                for g in _G,G:
                    if g not in _G_: _G_ += [g]
    # init conv kernels:
    for G in reversed(_G_):
        krim = []
        for link,rev in G.rim_[-1]:
            if link.ft[0]:  # must be mlink
                _G = link.nodet[0] if link.nodet[1] is G else link.nodet[1]
                krim += [_G]
                if hasattr(G,'dLay'):
                    G.dLay.add_H(link.derH); G._kLay.add_H(_G.derH)
                else:
                    G.dLay = CH().copy(link.derH); G._kLay = CH().copy(_G.derH)  # init ders, next krim comparand
        if krim:
            if rng>1: G.kHH[-1] += [krim]  # kH = lays ( nodes
            else:     G.kHH = [[krim]]
        else:
            _G_.remove(G)
    Gd_ = copy(_G_)  # Gs with 1st layer kH, dLay, _kLay
    n = 0   # n of higher kernel rims
    # convolution: kernel rim Def,Sum,Comp, in separate loops for bilateral G,_G assign:
    while True:
        G_ = []
        for G in _G_:  # += krim
            G.kHH[-1] += [[]]; G.visited__ += [[]]
        for G in _G_:
            #  append G.kHH[-1][-1]:
            for _G in G.kHH[-1][-2]:
                for link, rev in _G.rim_[-1]:
                    __G = link.nodet[0] if link.nodet[1] is G else link.nodet[1]
                    if __G in _G_:
                        if __G not in G.kHH[-1][-1] + [g for visited_ in G.visited__ for g in visited_]:
                            # bilateral add layer of unique mediated nodes
                            G.kHH[-1][-1] += [__G]; __G.kHH[-1][-1] += [G]
                            for g,_g in zip((G,__G),(__G,G)):
                                g.visited__[-1] += [_g]
                                if g not in G_:  # in G_ only if in visited__[-1]
                                    g.kLay = CH()  # init
                                    G_ += [g]
        for G in G_: G.visited__ += [[]]
        for G in G_: # sum kLay:
            for _G in G.kHH[-1][-1]:  # add last krim
                if _G in G.visited__[-1] or _G not in _G_:
                    continue  # Gs krim appended when _G was G
                G.visited__[-1] += [_G]; _G.visited__[-1] += [G]
                # sum alt G lower kLay:
                G.kLay.add_H(_G._kLay); _G.kLay.add_H(G._kLay)
        for G in G_: G.visited__[-1] = []
        for G in G_:
            for _G in G.kHH[-1][0]:  # convo in direct kernel only
                if _G in G.visited__[-1] or _G not in G_: continue
                G.visited__[-1] += [_G]; _G.visited__[-1] += [G]
                # comp G kLay:
                dlay = G.kLay.comp_H(_G.kLay,rn=1,fagg=1)
                if dlay.Et[0] > ave * dlay.Et[2] * (rng+n):  # layers add cost
                    _G.dLay = _G.dLay.add_H(dlay) if hasattr(_G,'dLay') else CH().copy(dlay)
                    G.dLay = G.dLay.add_H(dlay) if hasattr(G,'dLay') else CH().copy(dlay)  # bilateral
        # eval dLay:
        for G in reversed(G_):
            G.visited__.pop()  # loop-specific layer
            if not hasattr(G,'dLay') or G.dLay.Et[0] <= ave * G.dLay.Et[2] * (rng+n+1):
                G_.remove(G)
        for G in _G_:
            if G in G_: G._kLay = G.kLay  # comp in next krng
            else: G.visited__.pop()  # init or weak
        if G_:
            _G_ = G_; n += 1
        else:
            for G in Gd_:
                if rng: G.extH[-1].append(G.dLay)
                else:   G.extH.append(CH().append(G.dLay))
                delattr(G,'dLay'); delattr(G,'kLay'); delattr(G,'_kLay')
            break
    return Gd_, Et  # all Gs with dLay added in 1st krim


def rng_link_(_L_):  # comp CLs: der+'rng+ in root.link_ rim_t node rims: directional and node-mediated link tracing

    _mN_t_ = [[[L.nodet[0]],[L.nodet[1]]] for L in _L_]  # rim-mediating nodes
    rng = 1; rL_ = []
    Et = [0,0,0,0]
    while True:
        mN_t_ = [[[],[]] for _ in _L_]  # for next loop
        for L, _mN_t, mN_t in zip(_L_, _mN_t_, mN_t_):
            for rev, _mN_, mN_ in zip((0,1), _mN_t, mN_t):
                # comp L, _Ls: nodet mN 1st rim, -> rng+ _Ls/ rng+ mm..Ns, flatten rim_s:
                rim_ = [[l for rim in n.rim_ for l in rim] if isinstance(n,CG) else n.rimt_[0][0] + n.rimt_[0][1] for n in _mN_]
                for rim in rim_:
                    for _L,_rev in rim:  # _L is reversed relative to its 2nd node
                        if _L is L or _L in L.visited_: continue
                        if not hasattr(_L,"rimt_"): add_der_attrs(link_=[_L])  # _L not in root.link_, same derivation
                        L.visited_ += [_L]; _L.visited_ += [L]
                        dy,dx = np.subtract(_L.yx, L.yx)
                        Link = CL(nodet=[_L,L], span=2, angle=[dy,dx], box=extend_box(_L.box, L.box))
                        # L.rim_t += new Link
                        if comp_N(Link, Et, rng, rev ^ _rev):  # negate ds if only one L is reversed
                            # L += rng+'mediating nodes, link orders: nodet < L < rimt_, mN.rim || L
                            mN_ += _L.nodet  # get _Ls in mN.rim
                            if _L not in _L_:
                                _L_ += [_L]; mN_t_ += [[[],[]]]  # not in root
                            elif _L not in rL_: rL_ += [_L]
                            if L not in rL_:    rL_ += [L]
                            mN_t_[_L_.index(_L)][1 - rev] += L.nodet
        L_, mN_t_ = [],[]
        for L, mN_t in zip(_L_, mN_t_):
            if any(mN_t):
                L_ += [L]; _mN_t_ += [mN_t]
        if L_:
            _L_ = L_; rng += 1
        else:
            break
        # Lt_ = [(L, mN_t) for L, mN_t in zip(L_, mN_t_) if any(mN_t)]
        # if Lt_: L_,_mN_t_ = map(list, zip(*Lt_))  # map list to convert tuple from zip(*)

    return rL_, Et, rng


def comp_N(Link, iEt, rng, rev=None):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = rev is not None  # compared links have binary relative direction?
    _N, N = Link.nodet; rn = _N.n / N.n

    if fd:  # CLs, form single layer:
        DLay = _N.derH.comp_H(N.derH, rn, fagg=1)  # new link derH = local dH
        N.mdext = comp_ext(2,2, _N.S,N.S/rn, _N.angle, N.angle if rev else [-d for d in N.angle])  # reverse for left link
        N.Et = np.add(DLay.Et, N.mdext.Et)
    else:   # CGs
        mdlat = comp_latuple(_N.latuple, N.latuple, rn,fagg=1)
        mdLay = _N.mdLay.comp_md_(N.mdLay, rn, fagg=1)  # not in CG of links?
        mdext = comp_ext(len(_N.node_), len(N.node_), _N.S, N.S / rn, _N.A, N.A)
        DLay = CH(  # 1st derLay
            md_t=[mdlat,mdLay,mdext], Et=np.array(mdlat.Et)+mdLay.Et+mdext.Et, Rt=np.array(mdlat.Rt)+mdLay.Rt+mdext.Rt, n=2.5)
        mdlat.root = DLay; mdLay.root = DLay; mdext.root = DLay
        if _N.derH and N.derH:
            dLay = _N.derH.comp_H(N.derH,rn,fagg=1,frev=rev)
            DLay.add_H(dLay)  # also append discrete higher subLays in dLay.H_[0], if any?
            # no comp extH: current ders
    if fd: Link.derH.append_(DLay)
    else:  Link.derH = DLay
    iEt[:] = np.add(iEt,DLay.Et)  # init eval rng+ and form_graph_t by total m|d?
    for i in 0,1:
        Val, Rdn = DLay.Et[i::2]
        if Val > G_aves[i] * Rdn * (rng+1): Link.ft[i] = 1  # fork inclusion tuple
        _N.Et[i] += Val; N.Et[i] += Val  # not selective
        _N.Et[2+i] += Rdn; N.Et[2+i] += Rdn  # per fork link in both Gs
        # if select fork links: iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dH.Et[i::2])]
    if any(Link.ft):
        Link.n = min(_N.n,N.n)  # comp shared layers
        Link.yx = np.add(_N.yx, N.yx) / 2
        Link.S += (_N.S + N.S) / 2
        for rev, node in zip((0,1),(_N,N)):  # ?reversed Link direction
            if fd:
                if len(node.rimt_) == rng:
                    node.rimt_[-1][1-rev] += [[Link,rev]]  # add in last rng layer, opposite to _N,N dir
                else: node.rimt_ += [[[[Link,rev]],[]]] if dir else [[[],[[Link,rev]]]]  # add rng layer
            else:
                if len(node.extH.H) == rng:
                    node.rim_[-1] += [[Link, rev]]  # accum last rng layer
                else: node.rim_ += [[[Link, rev]]]  # init rng layer

        return True

def comp_ext(_L,L,_S,S,_A,A):  # compare non-derivatives:

    dL = _L - L; mL = min(_L,L) - ave_L  # direct match
    dS = _S/_L - S/L; mS = min(_S,S) - ave_L  # sparsity is accumulated over L
    mA, dA = comp_angle(_A,A)  # angle is not normalized
    M = mL + mS + mA
    D = abs(dL) + abs(dS) + abs(dA)  # signed dA?
    mrdn = M > D; drdn = D<= M
    mdec = mL/ max(L,_L) + mS/ max(S,_S) if S or _S else 1 + mA  # Amax = 1
    ddec = max_dist + mL/ (L+_L) + dS/ (S+_S) if S or _S else 1 + dA

    return CH(H=[mL,dL, mS,dS, mA,dA], Et=[M,D,mrdn,drdn], Rt=[mdec,ddec], n=0.5)


def form_graph_t(root, N_, Et, rng):  # segment N_ to Nm_, Nd_

    node_t = []
    for fd in 0,1:
        # der+: comp link via link.node_ -> dual trees of matching links in der+rng+, more likely but complex: higher-order links
        # rng+: distant M / G.M, less likely: term by >d/<m, but less complex
        if Et[fd] > ave * Et[2+fd]:
            if not fd:
                for G in N_: G.root = []  # only nodes have roots?
            graph_ = segment_N_(root, N_, fd, rng)
            for graph in graph_:
                Q = graph.link_ if fd else graph.node_  # xcomp -> max_dist * rng+1
                if len(Q) > ave_L and graph.derH.Et[fd] > G_aves[fd] * graph.derH.Et[fd+2]:
                    if fd: add_der_attrs(Q)
                    agg_recursion(graph, Q, fL=isinstance(Q[0],CL), rng=rng)  # fd rng+
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]
    for fd, graph_ in enumerate(node_t):  # mix forks fb
        for graph in graph_:
            root.fback_t[fd] += [graph.derH] if fd else [graph.derH.H[-1]]  # der+ forms new links, rng+ adds new layer
            # sub+-> sub root-> init root
    if any(root.fback_t): feedback(root)

    return node_t

def add_der_attrs(link_):
    for link in link_:
        link.extH = CH()  # for der+
        link.root = None  # dgraphs containing link
        link.rimt_ = [[[],[]]]  # dirt per rng layer
        link.visited_ = []
        link.med = 1  # comp med rng, replaces len rim_
        link.Et = [0,0,0,0]
        link.aRad = 0

# not revised below:
'''
cluster by weights of shared links + similarity of partial clusters, initially single linkage,
similar to parallelized https://en.wikipedia.org/wiki/Watershed_(image_processing).
sub-cluster: node assign by all in-graph weights, added in merges. 
'''
def segment_N_(root, iN_, fd, rng):

    N_ = []  # init Gts per node in iN_, merge if Lrim overlap + similarity of exclusive node_s
    max_ = []
    for N in iN_:
        # init graphts:
        rim = [Lt[0] for rim in N.rim_ for Lt in rim] if isinstance(N,CG) else [Lt[0] for rimt in N.rimt_ for rim in rimt for Lt in rim]
        _N_t = [[_N for L in rim for _N in L.nodet if _N is not N], [N]]  # [ext_N_, int_N_]
        Gt = [[N],[],copy(rim),_N_t,[0,0,0,0]]  # [node_, link_, Lrim, Nrim_t, Et]
        N.root = Gt; N_ += [Gt]
        if not fd:
            if not any([(eN.extH.H[-1].Et[0] > N.extH.H[-1].Et[0]) or (eN in max_)  # _N if _N.V == N.V
                        for eN in _N_t[0]]):  # use local maxes as Gt seeds or exemplars:
                max_ += [N]  # V * k * max_rng: + mediation if no max in rrim?
    max_ = [N.root for N in max_]
    for Gt in N_: Gt[3][0] = [_N.root for _N in Gt[3][0]]  # replace extNs with their Gts
    # merge with connected _Gts:
    for Gt in max_ if max_ else N_:
        node_, link_, Lrim, Nrim_t, Et = Gt
        Nrim = Nrim_t[0]
        for _Gt, _L in zip(Nrim, Lrim):
            if _Gt not in N_:
                continue  # was merged
            oL_ = set(Lrim).intersection(set(_Gt[2])).union([_L])  # shared external links + potential _L # oL_ = [Lr[0] for Lr in _Gt[2] if Lr in Lrim]
            oV = sum([L.derH.Et[fd] - ave * L.derH.Et[2+fd] for L in oL_])
            # eval by Nrim similarity = oV + olp between G,_G,
            # ?pre-eval by max olp: _node_ = _Gt[0]; _Nrim = _Gt[3][0],
            # if len(Nrim)/len(node_) > ave_L or len(_Nrim)/len(_node_) > ave_L:
            sN_ = set(node_); _sN_ = set(_Gt[0])
            oN_ = sN_.intersection(_sN_)  # Nrim overlap
            xN_ = list(sN_- oN_)  # exclusive node_
            _xN_ = list(_sN_- oN_)
            if _xN_ and xN_:
                dderH = comp_N_(_xN_, xN_)
                oV += (dderH.Et[fd] - ave * dderH.Et[2+fd])  # norm by R, * dist_coef * agg_coef?
            if oV > ave:
                link_ += [_L]
                merge(Gt,_Gt); N_.remove(_Gt)

    return [sum2graph(root, Gt, fd, rng) for Gt in N_]

def merge(Gt, gt):

    N_,L_, Lrim, Nrim_t, Et = Gt
    n_,l_, lrim, nrim_t, et = gt
    N_ += n_
    L_ += l_  # internal, no overlap
    Lrim[:] = list(set(Lrim + lrim))  # exclude shared external links, direction doesn't matter?
    Nrim_t[:] = [[G for G in nrim_t[0] if G not in Nrim_t[0]], list(set(Nrim_t[1] + nrim_t[1]))]  # exclude shared external nodes
    Et[:] = np.add(Et,et)


def sum_N_(N_, fd=0):  # sum partial grapht in merge

    N = N_[0]
    n = N.n; S = N.S
    L, A = (N.span, N.angle) if fd else (len(N.node_), N.A)
    if not fd:
        latuple = deepcopy(N.latuple)  # ignore if CL?
        mdLay = deepcopy(N.mdLay)
    derH = CH(); derH.copy(N.derH)
    extH = CH(); extH.copy(N.extH)
    # Et = copy(N.Et)
    for N in N_[1:]:
        if not fd:
            latuple = [P+p for P,p in zip(latuple[:-1],N.latuple[:-1])] + [[A+a for A,a in zip(latuple[-1],N.latuple[-1])]]
            if N.mdLay: mdLay.add_md_(N.mdLay)
        n += N.n; S += N.S
        L += N.span if fd else len(N.node_)
        A = [Angle+angle for Angle,angle in zip(A, N.angle if fd else N.A)]
        if N.derH: derH.add_H(N.derH)
        if N.extH: extH.add_H(N.extH)

    if fd: return n, L, S, A, derH, extH
    else:  return n, L, S, A, derH, extH, latuple, mdLay  # no comp Et

def comp_N_(_node_, node_):  # compare partial graphs in merge

    dderH = CH()
    fd = isinstance(_node_[0], CL)
    _pars = sum_N_(_node_,fd); _n,_L,_S,_A,_derH,_extH = _pars[:6]
    pars = sum_N_(node_,fd);    n, L, S, A, derH, extH = pars[:6]
    rn = _n/n
    mdext = comp_ext(_L,L, _S,S/rn, _A,A)
    dderH.n = mdext.n;  dderH.Et = np.array(mdext.Et); dderH.Rt = np.array(mdext.Rt)
    if fd:
        dderH.H = [[mdext]]
    else:
        _latuple, _mdLay = _pars[6:]; latuple, mdLay = pars[6:]
        if any(_latuple[:5]) and any(latuple[:5]):  # latuple is empty in CL
            mdlat = comp_latuple(_latuple, latuple, rn, fagg=1)
            dderH.n+=mdlat.n; dderH.Et+=mdlat.Et; dderH.Rt+=mdlat.Rt
        if _mdLay and mdLay:
            mdlay = _mdLay.comp_md_(mdLay, rn, fagg=1)
            dderH.n+=mdlay.n; dderH.Et+=mdlay.Et; dderH.Rt+=mdlay.Rt
    if _derH and derH:
        ddderH = _derH.comp_H(derH, rn, fagg=1)  # append and sum new dderH to base dderH
        dderH.H += ddderH.H  # merge derlay
        dderH.n+=ddderH.n; dderH.Et+=ddderH.Et; dderH.Rt+=ddderH.Rt

    return dderH

# not fully updated
def sum2graph(root, grapht, fd, rng):  # sum node and link params into graph, aggH in agg+ or player in sub+

    G_, Link_, _, _, Et = grapht
    graph = CG(fd=fd, node_=G_,link_=Link_, rng=rng, Et=Et)  # it's Link_ only now
    if fd: graph.root = root
    extH = CH()  # convert to graph derH
    yx = [0,0]
    for G in G_:
        extH.add_H(G.extH) if extH else extH.append_(G.extH,flat=1)
        graph.area += G.area
        graph.box = extend_box(graph.box, G.box)
        if isinstance(G, CG):  # add latuple to CL too?
            graph.latuple = [P+p for P,p in zip(graph.latuple[:-1],G.latuple[:-1])] + [[A+a for A,a in zip(graph.latuple[-1],G.latuple[-1])]]
        graph.n += G.n  # non-derH accumulation?
        if G.derH:
            graph.derH.add_H(G.derH)
        if fd: G.Et = [0,0,0,0]  # reset in last form_graph_t fork, Gs are shared in both forks
        else:  G.root = graph  # assigned to links if fd else to nodes?
        yx = np.add(yx, G.yx)
    L = len(G_)
    yx = np.divide(yx,L); graph.yx = yx
    graph.aRad = sum([np.hypot(*np.subtract(yx,G.yx)) for G in G_]) / L  # average distance between graph center and node center
    for link in Link_:
        # sum last layer of unique current-layer links
        graph.S += link.span
        graph.A = np.add(graph.A,link.angle)  # np.add(graph.A, [-link.angle[0],-link.angle[1]] if rev else link.angle)
        if fd: link.root = graph
    if extH: graph.derH.append_(extH, flat=0)  # graph derH = node derHs + [summed Link_ derHs], may be nested by rng
    if fd:
        # assign alt graphs from d graph, after both linked m and d graphs are formed
        for link in graph.link_:
            mgraph = link.root
            if mgraph:
                for fd, (G, alt_G) in enumerate(((mgraph,graph), (graph,mgraph))):  # bilateral assign:
                    if G not in alt_G.alt_graph_:
                        G.alt_graph_ += [alt_G]
    return graph

def feedback(root):  # called from form_graph_, always sub+, append new der layers to root

    mDerLay = CH()  # added per rng+, | kern+, | single kernel?
    while root.fback_t[0]: mDerLay.add_H(root.fback_t[0].pop())
    dDerH = CH()  # from higher-order links
    while root.fback_t[1]: dDerH.add_(root.fback_t[1].pop())
    DderH = mDerLay.append_(dDerH, flat=1)
    m,d, mr,dr = DderH.Et
    if m+d > sum(G_aves) * (mr+dr):
        root.derH.H[-1].append_(DderH, flat=0)  # append new derLay, maybe nested
    # eval max derH.H[i].node_ as node_,
    # max subH = derH.H[i]