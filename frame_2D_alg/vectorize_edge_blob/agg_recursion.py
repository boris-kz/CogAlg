import numpy as np
from copy import deepcopy, copy
from itertools import combinations, product, zip_longest
from .slice_edge import comp_angle, CsliceEdge
from .comp_slice import rng_recursion, form_PP_t, comp_latuple, CH, CG
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

class Clink(CBase):  # product of comparison between two nodes or links
    name = "link"

    def __init__(l, nodet=None,derH=None, span=0, angle=None, box=None):
        super().__init__()
        # Clink = binary tree of Gs, depth+/der+: Clink nodet is 2 Gs, Clink + Clinks in nodet is 4 Gs, etc., unpack sequentially.
        l.nodet = [] if nodet is None else nodet  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.n = 1  # min(node_.n)
        l.S = 0  # sum nodet
        l.angle = [0,0] if angle is None else angle  # dy,dx between node centers
        l.span = span  # distance between nodet centers
        l.box = [] if box is None else box  # sum nodet
        l.area = 0  # sum nodet
        l.latuple = []  # sum nodet
        l.Vt = [0,0]  # for rim-overlap modulated segmentation, init derH.Et[:2]
        l.derH = CH() if derH is None else derH
        l.DerH = CH()  # ders from kernels: G.DerH
    def __bool__(l): return bool(l.derH.H)


def vectorize_root(image):  # vectorization in 3 composition levels of xcomp, cluster:

    frame = CsliceEdge(image).segment()
    for edge in frame.blob_:
        if hasattr(edge, 'P_') and edge.latuple[-1] * (len(edge.P_)-1) > G_aves[0]:  # eval G, rdn=1
            edge.derH=CH(); edge.iderH = CH(); edge.Et = [0,0,0,0]; edge.link_= []
            edge.fback_=[]; edge.fback_t=[[],[]] # fback_/comp_slice, fback_t/agg+
            for P in edge.P_:
                P.derH = CH(); P.rim_ = []
            # vertical, lateral-overlap P cross-comp -> PP clustering:
            rng_recursion(edge)
            form_PP_t(edge, edge.P_)  # calls der+: PP P_,link_'replace, derH+ or rng++: PP.link_+
            node_t, link_t = [[],[]], [[],[]]
            for fd, node_ in enumerate(copy(edge.node_)):  # always node_t
                if edge.iderH and any(edge.iderH.Et):   # any for np array
                    if edge.iderH.Et[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.iderH.Et[2+fd]:
                        pruned_node_ = []
                        for PP in node_:  # PP -> G
                            if PP.iderH and PP.iderH.Et[fd] > G_aves[fd] * PP.iderH.Et[2+fd]:  # v> ave*r
                                PP.root_ = []  # no feedback to edge?
                                PP.node_ = PP.P_  # revert node_
                                y0,x0,yn,xn = PP.box
                                PP.yx = [(y0+yn)/2, (x0+xn)/2]
                                PP.aRad = np.hypot(*np.subtract(PP.yx,(yn,xn)))
                                PP.Et = [0,0,0,0]  # [] in comp_slice
                                pruned_node_ += [PP]
                        if len(pruned_node_) > 10:  # discontinuous PP rng+ cross-comp, cluster -> G_t:
                            node_t[fd] = pruned_node_
                            agg_recursion(edge, N_=pruned_node_, fagg=1)
                            link_t[fd] = edge.link_
            if any(node_t):
                edge.node_ = node_t; edge.link_ = link_t

def agg_recursion(root, N_, rng=1, fagg=0):  # rng for sub+'rng+ only

    Et = [0,0,0,0]
    N_,Et,_ = rng_node_(N_,Et,rng) if fagg else rng_link_(N_,Et)  # 1st call
    rng += fagg  # was incremented above
    fcompr,fcompd = 1,1
    while fcompr or fcompd:  # eval comp recursion per fork, while any last-loop comp
        fcompr,fcompd = 0,0
        if Et[0] > ave*Et[2]:  # init rng+ via convolution
            N_,Et, fcompr = rng_node_(N_,Et,rng) if isinstance(N_[0],CG) else rng_link_(N_,Et)
            rng += 1
        if Et[1] > ave*Et[3]:  # concat links from initial or recursive xcomp:
            if isinstance(N_[0],CG): N_ = list(set([linkt[0] for N in N_ for linkt in N.rim]))
            else:                    N_ = list(set([linkt[0] for N in N_ for linkt in N.rimt_[-1][0]+N.rimt_[-1][1]]))
            add_der_attrs(link_= N_)
            N_,Et,fcompd = rng_link_(N_,Et)  # N_ is higher-derivation links
    # sub+, fback_ sum in sub_roots, passed to root fback_:
    node_t = form_graph_t(root, N_, Et, rng)
    if node_t:
        fback_t = [[],[]]
        for fd, node_ in zip((0,1), node_t):
            N_ = [n for n in node_ if n.derH.Et[0] > G_aves[fd] * n.derH.Et[2]]  # pruned node_
            # comp val is proportional to n comparands:
            if root.derH.Et[0] * ((len(N_)-1)*root.rng) > G_aves[1]*root.derH.Et[2]:
                # agg+ / node_t, vs. sub+ / node_, always rng+:
                agg_recursion(root, N_, fagg=1)
                # each fork in agg+ fback_t sums both forks of sub+ fback_t:
                fback_t[fd] += [root.derH] if fd else [root.derH[-1]]  # from last rng+ only
        if any(fback_t):
            root.fback_t = fback_t
            feedback(root, fsub=0)
        root.node_[:] = node_t  # else keep root.node_


def rng_node_(N_, Et, rng):  # comp Gs|kernels in agg+, links | link rim_t node rims in sub+
                             # ~ graph convolutional network without backprop
    G_ = []
    fcomp=0
    for (_G, G) in list(combinations(N_,r=2)):  # eval comp_N-> G_
        if _G in G.compared_: continue
        dy,dx = np.subtract(_G.yx,G.yx)
        dist = np.hypot(dy,dx)
        aRad = (G.aRad+_G.aRad) / 2  # ave radius to eval relative distance between G centers:
        if dist / max(aRad,1) <= max_dist * rng:
            fcomp = 1; G.compared_ += [_G]; _G.compared_ += [G]
            Link = Clink(nodet=[_G,G], span=dist, angle=[dy,dx], box=extend_box(G.box,_G.box))
            if comp_N(Link, Et):  # in highder der+ rng++, G maybe a Clink
                for g in _G,G:
                    if g not in G_: G_ += [g]
                    if g.DerH: g.DerH.H[-1].add_(Link.derH)  # accum last DerH layer
                    else:      g.DerH.append_(Link.derH, flat=0)  # init DerH layer with Link.derH
    # def kernel rim per G:
    for G in G_:
        G.compared_ = []
        G.krim = [link.nodet[0] if link.nodet[1] is G else link.nodet[1] for link, rev in G.rim]
    n = 1  # n convolutions = n DerH layers
    iG_ = copy(G_)  # copy a list of G_ with added DerH
    while True:
        _G_ = []  # rng+ convolution, cross-comp: recursive center node DerH += linked node derHs for next loop:
        for G in G_:
            for _G in G.krim:
                if _G in G.compared_: continue
                G.compared_ += [_G]; _G.compared_ += [G]
                dderH = _G.DerH.H[-1].comp_(G.DerH.H[-1], dderH=CH(), rn=1, fagg=1, flat=1)  # comp last krims
                if dderH.Et[0] > ave * dderH.Et[2]:
                    for g in _G,G:  # bilateral assign
                        g.DerH.H[-1].add_(dderH) if len(g.DerH.H)==n+1 else g.DerH.append_(dderH,flat=0)
            # eval update to continue rng+/G:
            if len(G.DerH.H)>n and G.DerH.H[-1].Et[0] - G.DerH.H[-2].Et[0] > ave:  # G.DerH may not be appended
                _G_ += [G]
        if _G_:
            G_ = _G_
            for _G in _G_: _G.compared_ = []  # reset in intermediate rng+ only, nest in sub+'rng+
            n += 1
        else:
            break
    for G in iG_:
        delattr(G, "krim")
        G.extH.append_(G.DerH, flat=0)  # for segmentation

    return N_, Et, fcomp

def rng_link_(N_, Et):  # comp Clinks: der+'rng+ in root.link_ rim_t node rims: directional and node-mediated link tracing

    _mN_t_ = [[[N.nodet[0]],[N.nodet[1]]] for N in N_]  # rim-mediating nodes
    L_ = N_[:]
    fcomp = 0
    rng = 1
    while True:
        mN_t_ = [[[],[]] for _ in L_]
        for L, _mN_t, mN_t in zip(L_, _mN_t_, mN_t_):
            for rev, _mN_, mN_ in zip((0,1), _mN_t, mN_t):
                # comp L, _Ls: nodet mN 1st rim, -> rng+ _Ls/ rng+ mm..Ns:
                rim_ = [n.rim if isinstance(n,CG) else n.rimt_[0][0] + n.rimt_[0][1] for n in _mN_]
                for rim in rim_:
                    for _L,_rev in rim:  # _L is reversed relative to its 2nd node
                        if _L is L or _L in L.compared_: continue
                        if not hasattr(_L,"rimt_"): add_der_attrs(link_=[_L])  # _L not in root.link_, same derivation
                        fcomp = 1; L.compared_ += [_L]; _L.compared_ += [L]
                        dy,dx = np.subtract(_L.yx,L.yx)
                        Link = Clink(nodet=[_L,L], span=np.hypot(dy,dx), angle=[dy,dx], box=extend_box(_L.box, L.box))
                        # L.rim_t += new Link
                        if comp_N(Link, Et, rng, rev^_rev):  # negate ds if only one L is reversed
                            # add rng+ mediating nodes to L, link order: nodet < L < rim_t, mN.rim || L
                            mN_ += _L.nodet  # get _Ls in mN.rim
                            if _L not in L_:  # not in root
                                L_ += [_L]; mN_t_ += [[[],[]]]
                            mN_t_[L_.index(_L)][1-rev] += L.nodet
        _L_, _mN_t = [],[]
        for L, mN_t in zip(L_, mN_t_):
            if any(mN_t):
                _L_ += [L]; _mN_t_ += [mN_t]
        if _L_:
            L_ = _L_; rng += 1
        else:
            break
        # Lt_ = [(L, mN_t) for L, mN_t in zip(L_, mN_t_) if any(mN_t)]
        # if Lt_: L_,_mN_t_ = map(list, zip(*Lt_))  # map list to convert tuple from zip(*)

    return N_, Et, fcomp

def comp_N(Link, iEt, rng=None, rev=None):  # rng,dir if fd, Link+=dderH, comparand rim+=Link

    fd = rev is not None  # compared links have binary relative direction?
    dderH = CH(); _N, N = Link.nodet; rn = _N.n / N.n

    if fd:  # Clink Ns
        _A, A = _N.angle, N.angle if rev else [-d for d in N.angle] # reverse angle direction for left link
        Et, rt, md_ = comp_ext(_N.span,N.span, _N.S,N.S/rn, _A,A)
        # Et, Rt, Md_ = comp_latuple(_G.latuple, G.latuple,rn,fagg=1)  # low-value comp in der+
        dderH.n = 1; dderH.Et = Et; dderH.relt = rt
        dderH.H = [CH(Et=copy(Et),relt=copy(rt),H=md_,n=1)]
    else:  # CG Ns
        et, rt, md_ = comp_ext(len(_N.node_),len(N.node_), _N.S,N.S/rn, _N.A,N.A)
        Et, Rt, Md_ = comp_latuple(_N.latuple, N.latuple, rn,fagg=1)
        dderH.n = 1; dderH.Et = np.add(Et,et); dderH.relt = np.add(Rt,rt)
        dderH.H = [CH(Et=et,relt=rt,H=md_,n=.5),CH(Et=Et,relt=Rt,H=Md_,n=1)]
        if N.iderH:  # not in Clink N
            _N.iderH.comp_(N.iderH, dderH, rn, fagg=1, flat=0)
    # / N, if >1 PPs | Gs:
    if _N.derH and N.derH: _N.derH.comp_(N.derH, dderH, rn, fagg=1, flat=0, frev=rev)  # append and sum new dderH to base dderH
    if _N.extH and N.extH: _N.extH.comp_(N.extH, dderH, rn, fagg=1, flat=1, frev=rev)

    if fd: Link.derH.append_(dderH, flat=1)  # append dderH.H to link.derH.H
    else:  Link.derH = dderH
    iEt[:] = np.add(iEt,dderH.Et)  # init eval rng+ and form_graph_t by total m|d?
    fin = 0
    for i in 0, 1:
        Val, Rdn = dderH.Et[i::2]
        if Val > G_aves[i] * Rdn: fin = 1
        _N.Et[i] += Val; N.Et[i] += Val  # not selective
        _N.Et[2+i] += Rdn; N.Et[2+i] += Rdn  # per fork link in both Gs
        # if select fork links: iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dderH.Et[i::2])]
    if fin:
        Link.n = min(_N.n,N.n)  # comp shared layers
        Link.yx = np.add(_N.yx, N.yx) / 2
        Link.S += (_N.S + N.S) / 2
        for rev, node in zip((0,1),(_N,N)):  # ?reversed Link direction
            if fd:
                if len(node.rimt_) == rng:
                    node.rimt_[-1][1-rev] += [[Link,rev]]  # add in last rng layer, opposite to _N,N dir
                else:
                    node.rimt_ += [[[[Link,rev]],[]]] if dir else [[[],[[Link,rev]]]]  # add rng layer
            else:
                node.rim += [[Link,rev]]
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

    return [M,D,mrdn,drdn], [mdec,ddec], [mL,dL, mS,dS, mA,dA]


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
                Q = graph.link_ if fd else graph.node_
                if len(Q) > ave_L and graph.Et[fd] > G_aves[fd] * graph.Et[fd]:
                    if fd: add_der_attrs(Q)
                    # else sub+rng+: comp Gs at distance < max_dist * rng+1:
                    agg_recursion(graph, Q, rng+1, fagg=1-fd)  # graph.node_ is not node_t yet, rng for rng+ only
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]
    # all sub+ feedback, after fd fork because it may pop root.fback_t[0]:
    for fd, graph_ in enumerate(node_t):
        for graph in graph_:
            root.fback_t[fd] += [graph.derH] if fd else [graph.derH.H[-1]]  # rng+ adds single new layer
            # sub+ -> sub root -> init root
    if any(root.fback_t): feedback(root)

    return node_t

def add_der_attrs(link_):
    for link in link_:
        link.extH = CH()  # for der+
        link.root = None  # dgraphs containing link
        link.rimt_ = [[[],[]]]  # dirt per rng layer
        link.compared_ = []
        link.med = 1  # comp med rng, replaces len rim_
        link.Et = [0,0,0,0]
        link.aRad = 0
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
        rim = [Lt[0] for Lt in N.rim] if isinstance(N,CG) else [Lt[0] for rimt in N.rimt_ for rim in rimt for Lt in rim]  # flatten rim_t
        _N_t = [[_N for L in rim for _N in L.nodet if _N is not N], [N]]  # [ext_N_, int_N_]
        # node_, link_, Lrim, Nrim_t, Et:
        Gt = [[N],[],copy(rim),_N_t,[0,0,0,0]]
        N.root = Gt
        N_ += [Gt]
        if not fd:  # def Gt seeds as local maxes, | affinity clusters | samples?:
            if not any([eN.DerH.Et[0] > N.DerH.Et[0] or (eN in max_) for eN in _N_t[0]]):  # _N if _N.V == N.V
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
            # Nrim similarity = relative V deviation,
            # + partial G similarity:
            if len(node_)/len(Nrim) > ave_L and len(_Gt[0])/len(_Gt[3][1]) > ave_L:
                sN_ = set(node_); _sN_ = set(_Gt[0])
                oN_ = sN_.intersection(_sN_)  # Nrim overlap
                xN_ = list(sN_- oN_) # exclusive node_
                _xN_ = list(_sN_- oN_)
                if _xN_ and xN_:
                    dderH = comp_N_(_xN_, xN_)
                    oV += (dderH.Et[fd] - ave * dderH.Et[2+fd])  # norm by R, * dist_coef * agg_coef?
            if oV > ave:
                Gt[1] += [_L]
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
        latuple = deepcopy(N.latuple)  # ignore if Clink?
        iderH = deepcopy(N.iderH)
    derH = deepcopy(N.derH)
    extH = deepcopy(N.extH)
    # Et = copy(N.Et)
    for N in N_[1:]:
        if not fd:
            latuple = [P+p for P,p in zip(latuple[:-1],N.latuple[:-1])] + [[A+a for A,a in zip(latuple[-1],N.latuple[-1])]]
            if N.iderH: iderH.add_(N.iderH)
        n += N.n; S += N.S
        L += N.span if fd else len(N.node_)
        A = [Angle+angle for Angle,angle in zip(A, N.angle if fd else N.A)]
        if N.derH: derH.add_(N.derH)
        if N.extH: extH.add_(N.extH)

    if fd: return n, L, S, A, derH, extH
    else:  return n, L, S, A, derH, extH, latuple, iderH  # no comp Et

def comp_N_(_node_, node_):  # compare partial graphs in merge

    dderH = CH()
    fd = isinstance(_node_[0], Clink)
    _pars = sum_N_(_node_,fd); _n,_L,_S,_A,_derH,_extH = _pars[:6]
    pars = sum_N_(node_,fd);    n, L, S, A, derH, extH = pars[:6]
    rn = _n/n
    et, rt, md_ = comp_ext(_L,L, _S,S/rn, _A,A)
    if fd:
        dderH.n = 1; dderH.Et = et; dderH.relt = rt
        dderH.H = [CH(Et=copy(et),relt=copy(rt),H=md_,n=1)]
    else:
        _latuple, _iderH = _pars[6:]; latuple, iderH = pars[6:]
        Et, Rt, Md_ = comp_latuple(_latuple, latuple, rn, fagg=1)
        dderH.n = 1; dderH.Et = np.add(Et,et); dderH.relt = np.add(Rt,rt)
        dderH.H = [CH(Et=et,relt=rt,H=md_,n=.5),CH(Et=Et,relt=Rt,H=Md_,n=1)]
        _iderH.comp_(iderH, dderH, rn, fagg=1, flat=0)

    if _derH and derH: _derH.comp_(derH, dderH, rn, fagg=1, flat=0)  # append and sum new dderH to base dderH
    if _extH and extH: _extH.comp_(extH, dderH, rn, fagg=1, flat=1)

    return dderH


def sum2graph(root, grapht, fd, rng):  # sum node and link params into graph, aggH in agg+ or player in sub+

    G_, Link_, _, _, Et = grapht
    graph = CG(fd=fd, node_=G_,link_=Link_, rng=rng, Et=Et)  # it's Link_ only now
    if fd: graph.root = root
    extH = CH()
    yx = [0,0]
    for G in G_:
        extH.append_(G.extH,flat=1)if graph.extH else extH.add_(G.extH)
        graph.area += G.area
        graph.box = extend_box(graph.box, G.box)
        if isinstance(G, CG):  # add latuple to Clink too?
            graph.latuple = [P+p for P,p in zip(graph.latuple[:-1],G.latuple[:-1])] + [[A+a for A,a in zip(graph.latuple[-1],G.latuple[-1])]]
            if G.iderH:  # empty in single-P PP|Gs
                graph.iderH.add_(G.iderH)
        graph.n += G.n  # non-derH accumulation?
        graph.derH.add_(G.derH)
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
    graph.derH.append_(extH, flat=0)  # graph derH = node derHs + [summed Link_ derHs]
    if fd:
        # assign alt graphs from d graph, after both linked m and d graphs are formed
        for link in graph.link_:
            mgraph = link.root
            if mgraph:
                for fd, (G, alt_G) in enumerate(((mgraph,graph), (graph,mgraph))):  # bilateral assign:
                    if G not in alt_G.alt_graph_:
                        G.alt_graph_ += [alt_G]
    return graph

def feedback(root, fsub=1):  # called from form_graph_, append new der layers to root

    # each agg+ cycle may form one empty fork:
    DerH = deepcopy(root.fback_t[0].pop(0) if root.fback_t[0] else root.fback_t[1].pop(0))  # init DerH merged from both forks
    for fd in 0,1:
        while root.fback_t[fd]:
            derH = root.fback_t[fd].pop(0)
            DerH.add_(derH)
        if DerH.Et[fd] > G_aves[fd] * DerH.Et[fd+2]:  # merge combined DerH into root.derH
            if fsub: root.derH.append_(DerH, flat=1)  # append higher layers
            else:    root.derH.add_(DerH)  # sum shared layers, append the rest