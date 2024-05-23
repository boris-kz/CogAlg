import numpy as np
from copy import deepcopy, copy
from itertools import combinations, product, zip_longest
from .slice_edge import comp_angle, CsliceEdge
from .comp_slice import ider_recursion, comp_latuple, get_match, CH, CG, CdP
from utils import box2center, extend_box
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
ave_dist = 2
max_dist = 10

class Clink(CBase):  # product of comparison between two nodes or links
    name = "link"

    def __init__(l, node_=None,rim=None, derH=None, extH=None, root=None, distance=0, angle=None, box=None ):
        super().__init__()

        l.node_ = [] if node_ is None else node_  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.angle = [0,0] if angle is None else angle  # dy,dx between node centers
        l.distance = distance  # distance between node centers
        l.n = 1  # or min(node_.n)?
        l.S = 0  # sum node_
        l.box = []  # sum node_
        l.area = 0  # sum node_
        l.latuple = []  # sum node_
        l.Vt = [0,0]  # for rim-overlap modulated segmentation, init derH.Et[:2]
        l.derH = CH() if derH is None else derH
        l.DerH = CH()  # ders from G.DerH
        ''' add in der+:
        l.extH = CH() if extH is None else extH  # for der+
        l.ExtH = CH()  # sum from kernels in der+
        l.Et = [0,0,0,0]  # sum from rim_t or node_
        l.root = None if root is None else root  # dgraphs containing link
        l.rim_t = []  # dual tree of _links, each may have its own node-mediated links, instead of rim
        l.compared_ = []
        l.dir = bool  # direction of comparison if not G0,G1, for comp link?
        # add in sum2graph if graph is Clink, or always CG?
        # l.link_ = []
        '''
    def __bool__(l): return bool(l.derH.H)


def vectorize_root(image):  # vectorization in 3 composition levels of xcomp, cluster:

    frame = CsliceEdge(image).segment()
    for edge in frame.blob_:
        if hasattr(edge, 'P_') and edge.latuple[-1] * (len(edge.P_)-1) > G_aves[0]:  # eval G, rdn=1
            add_attrs(edge)

            ider_recursion(None, edge)  # vertical, lateral-overlap P cross-comp -> PP clustering
            node_t, link_t = [[],[]], [[],[]]
            for fd, node_ in enumerate(edge.node_):  # always node_t
                if edge.iderH and any(edge.iderH.Et):  # any for np array
                    if edge.iderH.Et[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.iderH.Et[2+fd]:
                        pruned_node_ = []
                        for PP in node_:  # PP -> G
                            if PP.iderH and PP.iderH.Et[fd] > G_aves[fd] * PP.iderH.Et[2+fd]:
                                PP.root_ = []  # no feedback to edge?
                                PP.node_ = PP.P_  # revert base node_
                                PP.Et = [0,0,0,0]  # [] in comp_slice
                                pruned_node_ += [PP]
                        if len(pruned_node_) > 10:  # discontinuous PP rng+ cross-comp, cluster -> G_t:
                            node_t[fd] = pruned_node_
                            edge.node_ = pruned_node_  # agg+ of PP nodes: (temporary)
                            agg_recursion(None, edge, fagg=1)
                            link_t[fd] = edge.link_
            if any(node_t):
                edge.node_ = node_t; edge.link_ = link_t

def add_attrs(edge):
    edge.iderH = CH()  # for iRt and feedback
    edge.fback_ = []
    edge.Et = [0,0,0,0]
    edge.link_ = []
    for P in edge.P_:  # add comp_slice attrs
        P.rim_ = [[CdP(node_=[_P,P]) for _P in P.rim_]]; P.derH = CH()


def agg_recursion(rroot, root, fagg=0):

    nrng, Et = rng_convolve(root, [0,0,0,0], fagg)  # += connected nodes in med rng

    node_t = form_graph_t(root, root.node_ if fagg else root.link_, Et, nrng)  # der++ and feedback per Gd?
    if node_t:
        for fd, node_ in enumerate(node_t):
            if root.Et[0] * ((len(node_)-1)*root.rng) > G_aves[1] * root.Et[2]:
                # agg+ / node_t, vs. sub+ / node_, always rng+:
                pruned_node_ = [node for node in node_ if node.Et[0] > G_aves[fd] * node.Et[2]]  # not be needed?
                if len(pruned_node_) > 10:
                    agg_recursion(rroot, root, fagg=1)
                    if rroot and fd and root.derH:  # der+ only (check not empty root.derH)
                        rroot.fback_ += [root.derH]
                        feedback(rroot)  # update root.root..
'''
~ graph convolutional network without backprop
'''
def rng_convolve(root, Et, fagg):  # comp Gs|kernels in agg+, links | link rim_t node rims in sub+

    nrng = 1
    if fagg:
        # comp CGs, summed in krims for rng>1
        L_ = []; G_ = root.node_
        # init kernels:
        for link in list(combinations(G_,r=2)):
            _G, G = link
            if _G in G.compared_: continue
            cy, cx = box2center(G.box); _cy, _cx = box2center(_G.box)
            dy = cy-_cy; dx = cx-_cx;  dist = np.hypot(dy,dx)
            if nrng==1: fcomp = dist <= ave_dist  # eval distance between node centers
            else:
                M = (G.Et[0]+_G.Et[0])/2; R = (G.Et[2]+_G.Et[2])/2  # local
                fcomp = M / (dist/ave_dist) > ave * R
            if fcomp:
                G.compared_ += [_G]; _G.compared_ += [G]
                Link = Clink(node_=[_G, G], distance=dist, angle=[dy, dx], box=extend_box(G.box, _G.box))
                comp_G(Link, Et, L_)
        for G in G_:  # init kernel with 1st krim
            krim = []
            for link in G.rim:
                if G.derH: G.derH.add_(link.derH)
                else: G.derH = deepcopy(link.derH)
                krim += [link.node_[0] if link.node_[1] is G else link.node_[1]]
            G.kH = [krim]
        # aggregate rng+: recursive center node DerH += linked node derHs for next-loop cross-comp
        iG_ = G_
        while len(G_) > 2:
            nrng += 1; _G_ = []
            for G in G_:
                if len(G.rim) < 2: continue  # one link is always overlapped
                for link in G.rim:
                    if link.derH.Et[0] > ave:  # link.Et+ per rng
                        comp_krim(link, _G_, nrng)  # + kernel rim / loop, sum in G.extH, derivatives in link.extH?
            G_ = _G_
        # G.extH+ for segmentation
        for G in iG_:
            for i, link in enumerate(G.rim):
                G.extH.add_(link.DerH) if i else G.extH.append_(link.DerH, flat=1)
        root.link_ = L_
    else:
        # comp Clinks: der+'rng+ in root.link_ rim_t node rims: directional and link node -mediated
        _L_ = root.link_
        link_ = []
        while _L_:
            link_ += _L_; L_ = []
            for link in _L_:
                if link.rim_t:
                    rimt = [copy(link.rim_t[0][-1]) if link.rim_t[0] else [], copy(link.rim_t[1][-1]) if link.rim_t[1] else []]
                elif isinstance(link.node_[0],CG):
                    rimt = [link.node_[0].rim, link.node_[1].rim]  # der+ nrng=1
                else:  # link.node_ Clinks in der+ nrng>1
                    rimt = [copy(link.node_[0].rim_t[0][-1]) if link.node_[0].rim_t[0] else [], copy(link.node_[1].rim_t[1][-1]) if link.node_[1].rim_t[1] else []]
                # two directions per layer:
                for dir,rim in zip((0,1),rimt):
                    for _link in rim:
                        _G = _link.node_[0] if _link.node_[1] in link.node_ else _link.node_[1]  # mediating node
                        _rim = _G.rim if isinstance(_G,CG) else (copy(_G.rim_t[dir][-1]) if _G.rim_t and _G.rim_t[dir] else [])
                        for _link in _rim:
                            if _link is link: continue
                            if not hasattr(_link, "rim_t"): add_der_attrs(link_=[_link])
                            Link = Clink(node_=[_link,link])
                            comp_G(Link, Et, L_, dir=dir)  # update Link.rim_t, L_
            _L_= L_
            nrng += 1
        root.link_ = link_

    return nrng, Et

def add_der_attrs(link_):
    for link in link_:
        link.extH = CH()  # for der+
        link.ExtH = CH()  # sum from kernels in der+
        link.root = None  # dgraphs containing link
        link.rim_t = []   # dual tree of _links, each may have its own node-mediated links, instead of rim
        link.compared_ = []
        link.dir = 0      # direction of comparison if not G0,G1, for comp link?
'''
G.DerH sums krim _G.derHs, not from links, so it's empty in the first loop.
_G.derHs can't be empty in comp_krim: init in loop link.derHs
link.DerH is ders from comp G.DerH in comp_krim
G.extH sums link.DerHs: '''

def comp_krim(link, G_, nrng, fd=0):  # sum rim _G.derHs, compare to form link.DerH layer

    _G,G = link.node_  # same direction
    ave = G_aves[fd]
    for node in _G, G:
        if node in G_: continue  # new krim is already added
        krim = []  # kernel rim
        for _node in node.kH[-1]:
            for _link in _node.rim:
                __node = _link.node_[0] if _link.node_[1] is _node else _link.node_[1]
                krim += [_G for _G in __node.kH[-1] if _G not in krim]
                if node.DerH: node.DerH.add_(__node.derH, irdnt=_node.Et[2:])
                else:         node.DerH = deepcopy(__node.derH)  # init
        node.kH += [krim]
        G_ += [node]
    # sum G-specific kernel rim:
    _n,_L,_S,_A,_latuple,_iderH,_derH,_Et = sum_krim(list(set(_G.kH[-1])-set(G.kH[-1])))
    n, L, S, A, latuple, iderH, derH, Et  = sum_krim(list(set(G.kH[-1])-set(_G.kH[-1])))
    rn = _n / n
    dderH = CH()
    et, rt, md_ = comp_ext(_L,L,_S,S/rn,_A,A)
    Et, Rt, Md_ = comp_latuple(_latuple, latuple, rn, fagg=1)
    dderH.n = 1; dderH.Et = np.add(Et,et); dderH.relt = np.add(Rt,rt)
    dderH.H = [CH(Et=et,relt=rt,H=md_,n=.5),CH(Et=Et,relt=Rt,H=Md_,n=1)]
    # / PP:
    _iderH.comp_(iderH, dderH, rn, fagg=1, flat=0)
    # / G, if >1 PPs | Gs:
    if _derH and derH: _derH.comp_(derH, dderH, rn, fagg=1, flat=0)  # append and sum new dderH to base dderH
    # empty extH
    if dderH.Et[0] > ave * dderH.Et[2]:  # use nested link.derH vs DerH?
        link.DerH.H[-1].add_(dderH, irdnt=dderH.H[-1].Et[2:]) if len(link.DerH.H)==nrng else link.DerH.append_(dderH,flat=1)

    # connectivity eval in segment_graph via decay = (link.relt[fd] / (link.derH.n * 6)) ** nrng  # normalized decay at current mediation

def sum_krim(krim):  # sum last kernel layer

    _G = krim[0]
    n,L,S,A = _G.n,len(_G.node_),_G.S,_G.A
    latuple = deepcopy(_G.latuple)
    iderH = deepcopy(_G.iderH)
    derH = deepcopy(_G.derH)
    Et = copy(_G.Et)
    for G in krim[1:]:
        latuple = [P+p for P,p in zip(latuple[:-1],G.latuple[:-1])] + [[A+a for A,a in zip(latuple[-1],G.latuple[-1])]]
        n+=G.n; L+=len(G.node_); S+=G.S; A=[Angle+angle for Angle,angle in zip(A,G.A)]
        if G.iderH: iderH.add_(G.iderH)
        if G.derH: derH.add_(G.derH)
        np.add(Et,G.Et)
    return n, L, S, A, latuple, iderH, derH, Et  # not sure about Et


def comp_G(link, iEt, link_, dir=None):  # add dderH to link and link to the rims of comparands: Gs or links

    fd = dir is not None  # compared links have binary relative direction?
    dderH = CH()  # new layer of link.dderH
    _G, G = link.node_

    if fd:  # Clink Gs
        rn = min(_G.node_[0].n, _G.node_[1].n) / min(G.node_[0].n, G.node_[1].n)
        _S, S = _G.node_[0].S + _G.node_[1].S, (G.node_[0].S + G.node_[1].S)  # then sum in link.S
        _A, A = _G.angle, G.angle if dir else [-d for d in G.angle]  # reverse angle direction for left link
        Et, rt, md_ = comp_ext(_G.distance,G.distance, _S,S/rn, _A,A)
        # Et, Rt, Md_ = comp_latuple(_G.latuple, G.latuple,rn,fagg=1)  # seems like a low-value comp in der+
        dderH.n = 1; dderH.Et = Et; dderH.relt = rt
        dderH.H = [CH(Et=copy(Et), relt=copy(rt), H=md_, n=1)]
    else:  # CG Gs
        rn= _G.n/G.n  # comp ext params prior: _L,L,_S,S,_A,A, dist, no comp_G unless match:
        et, rt, md_ = comp_ext(len(_G.node_),len(G.node_), _G.S,G.S/rn, _G.A,G.A)
        Et, Rt, Md_ = comp_latuple(_G.latuple, G.latuple, rn,fagg=1)
        dderH.n = 1; dderH.Et = np.add(Et,et); dderH.relt = np.add(Rt,rt)
        dderH.H = [CH(Et=et,relt=rt,H=md_,n=.5),CH(Et=Et,relt=Rt,H=Md_,n=1)]
        # / PP:
        _G.iderH.comp_(G.iderH, dderH, rn, fagg=1, flat=0)  # always >1P in compared PPs?
    # / G, if >1 PPs | Gs:
    if _G.derH and G.derH: _G.derH.comp_(G.derH, dderH, rn, fagg=1, flat=0)  # append and sum new dderH to base dderH
    if _G.extH and G.extH: _G.extH.comp_(G.extH, dderH, rn, fagg=1, flat=1)

    if fd: link.derH.append_(dderH, flat=0)  # append dderH.H to link.derH.H
    else:  link.derH = dderH
    iEt[:] = np.add(iEt,dderH.Et)  # init eval rng+ and form_graph_t by total m|d?
    fin = 0
    for i in 0, 1:
        Val, Rdn = dderH.Et[i::2]
        if Val > G_aves[i] * Rdn: fin = 1
        _G.Et[i] += Val; G.Et[i] += Val  # not selective
        _G.Et[2+i] += Rdn; G.Et[2+i] += Rdn  # per fork link in both Gs
        # if select fork links: iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dderH.Et[i::2])]
    if fin:
        if fd:  # reciprocal link assign
            for dir,(L,_L) in zip([0,1],((G,_G),(_G,G))):  # nodes are links
                if L.rim_t:  # not sure about dir and nesting
                    if L.rim_t[dir]: L.rim_t[dir][-1] += [link]  # add new link
                    else:            L.rim_t[dir] = [[link]]  # if len(L.rim_t[dir])==nrng?
                else: L.rim_t = [[],[[link]]] if dir else [[[link]],[]]  # 1st der+ link.rim_t = []
        else:
            link.S += _G.S + G.S
            for node in _G,G: node.rim += [link]
        link_ += [link]


def comp_ext(_L,L,_S,S,_A,A):  # compare non-derivatives:

    dL = _L - L;      mL = min(_L,L) - ave_L  # direct match
    dS = _S/_L - S/L; mS = min(_S,S) - ave_L  # sparsity is accumulated over L
    mA, dA = comp_angle(_A,A)  # angle is not normalized
    M = mL + mS + mA
    D = abs(dL) + abs(dS) + abs(dA)  # signed dA?
    mrdn = M > D; drdn = D<= M
    mdec = mL/ max(L,_L) + mS/ max(S,_S) if S or _S else 1 + mA  # Amax = 1
    ddec = max_dist + mL/ (L+_L) + dS/ (S+_S) if S or _S else 1 + dA

    return [M,D,mrdn,drdn], [mdec,ddec], [mL,dL, mS,dS, mA,dA]


def form_graph_t(root,Q, Et, nrng):  # form Gm_,Gd_ from same-root nodes
    '''
    der+: comp link via link.node_ -> dual trees of matching links in der+rng+, more likely but complex: higher-order links
    rng+: less likely: term by >d/<m, but rim overlap Vt[0] may indicate interrupted match, and less complex?
    '''
    node_t = []
    for fd in 0, 1:
        if Et[fd] > ave * Et[2+fd]:  # v>ave*r
            if not fd:  # nodes have roots
                for G in Q: G.root = []
            graph_ = segment_Q(root, Q, fd, nrng)
            for graph in graph_:
                q = graph.link_ if fd else graph.node_
                if len(q) > ave_L and graph.Et[fd] > G_aves[fd] * graph.Et[fd]:  # olp-modulated Et
                    if fd:
                        if not hasattr(Q[0],"rim_t"): add_der_attrs(Q)  # 1st der+
                    else:
                        for G in q: G.compared_ = []
                    agg_recursion(root, graph, fagg=(1-fd))  # graph.node_ is not node_t yet
                elif graph.derH:
                    root.fback_ += [graph.derH]
                    # feedback(root)  # update root.root.. per sub+
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]
    if any(node_t):
        root.node_[:] = node_t  # else keep root.node_
        return node_t


def segment_Q(root, Q, fd, nrng):  # recursive eval node_|link_ rims for cluster assignment

    link_V_xolp(Q, fd)  # recursive N Et += link.V + link.relt * node'_N_ overlap v
    # rim link Vt is potentiated by overlap between _N_s of their nodes, AKA Shared Nearest Neighbours
    max_ = []
    for G in Q:
        _G_ = [link.node_[0] if link.node_[1] is G else link.node_[1] for link in get_rim(G)]  # all connected G' rim's _Gs
        if not any([_G.DerH.Et[0] > G.DerH.Et[0] or (_G in max_) for _G in _G_]):  # _G if _G.V==G.V
            max_ += [G]  # local maxes are seeds for floodfill, V* k * max range: add mediation step if no maxes in rrim?
    # init graphts:
    iGt_ = []
    for N in max_:
        rim = get_rim(N)
        _N_ = [link.node_[0] if link.node_[1] is N else link.node_[1] for link in rim]
        Gt = [[[N,rim]], copy(rim),copy(rim),_N_,[0,0,0,0]]  # nodet_,link_,Rim,_N_,Et; nodet: N,rim
        N.root = Gt
        iGt_ += [Gt]
    _Gt_ = copy(iGt_)
    r = 1
    while _Gt_:  # single-linked floodfill per Gt, breadth-first for parallelization
        Gt_ = []
        for Gt in copy(_Gt_):
            nodet_,link_,_rim, _N_, Et = Gt
            rim = []  # per node
            for link in _rim:  # floodfill Gt by rim tracing and _N.root merge
                link_ += [link]
                Et[:2] = np.add(Et[:2],link.Vt); Et[2:] = np.add(Et[2:],link.derH.Et[2:])
                _N,N = link.node_ if link.node_[1] in _N_ else [link.node_[1],link.node_[0]]  # N is in, _N is outside _N_
                if _N.root:
                    if _N.root is not Gt:  # root was not merged, + eval|comp to keep connected Gt separate?
                        merge_Gt(Gt,_N.root, rim, fd)
                else:
                    _N_ += [_N]  # for next loop:
                    rim += [L for L in get_rim(_N) if L.Vt[fd] > ave*L.derH.Et[2+fd]]  # link Vt x= link node_'__N_'olp
            if rim:
                _rim[:] = rim; Gt_ += [Gt]
        _Gt_ = Gt_
        r += 1  # recursion depth

    return [sum2graph(root, Gt, fd, nrng) for Gt in iGt_ if Gt]  # not-empty clusters

def merge_Gt(Gt, gt, rim, fd):

    Nodet_,N_,Link_,Rim, Et = Gt; nodet_,n_,link_,rim, et = gt

    for link in link_:
        if link.derH.Et[fd] > ave * link.derH.Et[2+fd] and link not in Link_:
            N = link.node_[1] if link.node_[1] in n_ else link.node_[0]  # the node of current link in gt
            if N not in N_:
                N_ += [N]  # add other node of link into N_
                for nodet in nodet_:
                    if nodet[0] is N and nodet not in Nodet_:  # find nodet based on N
                        Nodet_ += [nodet]  # merge nodet
                Link_ += [link]  # merge link

    Rim += [L for L in rim if L not in Rim]


def link_V_xolp(iQ, fd):  # recursive N.V += link.relv * node'_N_ overlap Et

    _oV_ = [0 for n in iQ]
    _Q = copy(iQ)
    r = 1
    while True:
        OV, DOV = 0,0
        Q, oV_ = [],[]
        for N, _oV in zip(_Q, _oV_):  # update node rim link Ets
            oV, DoV = 0,0
            rim = get_rim(N)
            _N_ = [L.node_[0] if L.node_[1] is N else L.node_[1] for L in rim]
            for _N, link in zip(_N_,rim):
                _rim = get_rim(_N)
                __N_ = [L.node_[0] if L.node_[1] is _N else L.node_[1] for L in _rim]
                oN_ = list(set(_N_).intersection(set(__N_)))
                oV += sum([n.Et[fd]- ave*n.Et[2+fd] for n in oN_])  # sum V deviation of overlapping Ns
                doV = oV - _oV  # oV update to adjust Vt
                link.Vt[fd] += link.derH.relt[fd] / link.derH.n * doV  # update oN_-modulated V only
                DoV += doV
            DOV += DoV; OV += oV
            if DoV > ave:
                Q += [N]; oV_ += [oV]
        # test update value:
        if DOV < ave:
            break
        for N in Q:  # sum node Ets from updated node rim link Ets:
            N.Et[fd] = sum([link.derH.Et[fd] for link in get_rim(N)])
        r += 1
        _Q = Q; _oV_ = oV_

def get_rim(N):

    if isinstance(N, Clink):  # N is Clink
        if isinstance(N.node_[0],CG):
            rim = [link for G in N.node_ for link in G.rim]
        else:
            # get link node rim_t dirs opposite from each other, else covered by the other link rim_t[1-dir]?
            rim = [ L for dir, link in zip((0,1), N.node_) for L in (link.rim_t[1-dir][-1] if link.rim_t[dir] else [])]  # flat
    else:   rim = N.rim
    return  rim

# not revised
def sum2graph(root, grapht, fd, nrng):  # sum node and link params into graph, aggH in agg+ or player in sub+

    _, Link_, _, G_, Et = grapht
    graph = CG(fd=fd, node_=G_,link_=Link_, rng=nrng, Et=Et)
    if fd:
        graph.root = root
    extH = CH()
    for G in G_:
        extH.append_(G.extH,flat=1)if graph.extH else extH.add_(G.extH)
        graph.area += G.area
        graph.box = extend_box(graph.box, G.box)
        if isinstance(G, CG):
            graph.latuple = [P+p for P,p in zip(graph.latuple[:-1],G.latuple[:-1])] + [[A+a for A,a in zip(graph.latuple[-1],G.latuple[-1])]]
            if G.iderH:  # empty in single-P PP|Gs
                graph.iderH.add_(G.iderH)
        graph.n += G.n  # non-derH accumulation?
        graph.derH.add_(G.derH)
        if fd: G.Et = [0,0,0,0]  # reset in last form_graph_t fork, Gs are shared in both forks
        else:  G.root = graph  # assigned to links if fd else to nodes?

    for link in Link_:  # sum last layer of unique current-layer links
        graph.S += link.distance
        np.add(graph.A,link.angle)
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

def feedback(root):  # called from form_graph_, append new der layers to root

    DerH = deepcopy(root.fback_.pop(0))  # init
    while root.fback_:
        derH = root.fback_.pop(0)
        DerH.add_(derH)
    if DerH.Et[1] > G_aves[1] * DerH.Et[3]:
        root.derH.add_(DerH)
    if root.root and isinstance(root.root, CG):  # not Edge
        rroot = root.root
        if rroot:
            fback_ = rroot.fback_  # always node_t if feedback
            if fback_ and len(fback_) == len(rroot.node_[1]):  # after all nodes' sub+
                feedback(rroot)  # sum2graph adds higher aggH, feedback adds deeper aggH layers