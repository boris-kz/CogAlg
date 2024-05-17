from typing_extensions import Unpack

import numpy as np
from copy import deepcopy, copy
from itertools import combinations, product, zip_longest
from .slice_edge import comp_angle, CsliceEdge, Clink
from .comp_slice import ider_recursion, comp_latuple, get_match
from .filters import aves, ave_mL, ave_dangle, ave, G_aves, ave_Gm, ave_Gd, ave_dist, ave_mA, max_dist
from utils import box2center, extend_box
import sys
sys.path.append("..")
from frame_blobs import CH, CG

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

def vectorize_root(image):  # vectorization in 3 composition levels of xcomp, cluster:

    frame = CsliceEdge(image).segment()

    for edge in frame.blob_:
        if hasattr(edge, 'P_') and edge.latuple[-1] * (len(edge.P_)-1) > G_aves[0]:  # eval G, rdn=1
            ider_recursion(None, edge)  # vertical, lateral-overlap P cross-comp -> PP clustering

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
                            edge.node_ = pruned_node_  # agg+ of PP nodes:
                            agg_recursion(None, edge, fagg=1)

def agg_recursion(rroot, root, fagg=0):

    for link in root.link_:
        link.Et = copy(link.derH.Et); link.relt = copy(link.derH.relt)

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
~ graph convolutional network but no backprop in rng+, only olp feedback in clustering
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
                    if link.Et[0] > ave:  # link.Et+ per rng
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
                if link.rim_t: rimt = [link.rim_t[0][-1] if link.rim_t[0] else [], link.rim_t[1][-1] if link.rim_t[1] else []]
                else:          rimt = [link.node_[0].rim,link.node_[1].rim]
                for dir,rim in zip((0,1),rimt):  # two directions per layer
                    for _link in rim:
                        _G = _link.node_[0] if _link.node_[1] in link.node_ else _link.node_[1]  # mediating node
                        _rim = _G.rim if isinstance(_G,CG) else (_G.rim_t[dir][-1] if _G.rim_t and _G.rim_t[dir] else [])
                        for _link in _rim:
                            if _link is link: continue
                            Link = Clink(node_=[_link,link])
                            comp_G(Link, Et, L_, dir=dir)  # update Link.rim_t, L_
            _L_= L_
            nrng += 1
        root.link_ = link_

    return nrng, Et
'''
G.DerH sums krim _G.derHs, not from links, so it's empty in the first loop.
_G.derHs can't be empty in comp_krim: init in loop link.derHs
link.DerH is ders from comp G.DerH in comp_krim
G.extH sums link.DerHs
'''
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

    if fd: link.derH.append_(dderH, flat=0)  # append to derH.H
    else:  link.derH = dderH
    iEt[:] = np.add(iEt,dderH.Et)  # init eval rng+ and form_graph_t by total m|d?
    fin = 0
    link.Et = np.add(link.Et, dderH.Et)  # per rng
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

    dL = _L - L;      mL = min(_L,L) - ave_mL  # direct match
    dS = _S/_L - S/L; mS = min(_S,S) - ave_mL  # sparsity is accumulated over L
    mA, dA = comp_angle(_A,A)  # angle is not normalized
    M = mL + mS + mA
    D = abs(dL) + abs(dS) + abs(dA)  # signed dA?
    mrdn = M > D; drdn = D<= M
    mdec = mL/ max(L,_L) + mS/ max(S,_S) if S or _S else 1 + mA  # Amax = 1
    ddec = max_dist + mL/ (L+_L) + dS/ (S+_S) if S or _S else 1 + dA

    return [M,D,mrdn,drdn], [mdec,ddec], [mL,dL, mS,dS, mA,dA]


def form_graph_t(root,Q, Et, nrng):  # form Gm_,Gd_ from same-root nodes

    node_t = []
    for fd in 0, 1:
        if Et[fd] > ave * Et[2+fd]:  # eVal > ave * eRdn
            for G in Q: G.root = []  # reset per fork
            # graph_ = segment_graph(root, copy(Q), fd, nrng)  # copy to use upQ in both forks
            graph_ = segment_parallel(root, Q, fd, nrng)
            if fd:  # der+ after rng++ term by high ds
                for graph in graph_:
                    if graph.link_ and graph.Et[1] > G_aves[1] * graph.Et[3]:  # Et is summed from all links
                        agg_recursion(root, graph, fagg=0)  # graph.node_ is not node_t yet
                    elif graph.derH:
                        root.fback_ += [graph.derH]
                        feedback(root)  # update root.root.. per sub+
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]
    if any(node_t):
        root.node_[:] = node_t  # else keep root.node_
        return node_t

'''
forward/feedback message-passing between two layers: roots with node_ and nodes with root_: 
while > ave dOV: compute link_ oV for each (node,root); then assign each node to max root
root feedback refines node x root link overlap, vs fitting to higher node in GNN backprop 

der+: correlation clustering of links through their nodes, forming dual trees of matching links in rng+?

add rng+/seg+: direct comp nodes (vs summed in krim) in proportion to graph size * M: likely distant match,
comp individual nodes in node-mediated krims, replacing krims and ExtH layers, 
adding links overlap for segmentation.
'''
def segment_parallel(root, Q, fd, nrng):  # recursive eval node_|link_ rims for cluster assignment
    '''
    kernels = get_max_kernels(Q)  # for selective link tracing?
    grapht_ = select_merge(kernels)  # refine by match to max, or only use it to selectively merge?
    '''
    node_,root_ = [],[]
    for i, N in enumerate(Q):
        N.root = i  # index in root_,node_
        rim = get_rim(N, fd)
        _N_ = [link.node_[0] if link.node_[1] is N else link.node_[1] for link in rim]
        root_ += [[[],[N], [[0,0,0,0]]]]  # init link_=N.rim, node_=[N], oEt_
        node_ += [[N, rim,_N_, [i], []]]  # root_ indices, empty nEt_
    r = 0  # recursion count
    _OEt = [0,0,0,0]
    while True:
        OEt = [0,0,0,0]
        for N,rim,_N_,nroot_,_oEt_ in node_:  # update node roots
            oEt_ = []
            for i,(link_,N_,nEt_) in enumerate(root_):
                olink_,oEt = [],[0,0,0,0]
                for L,_N in zip(rim,_N_):  # get overlap between N rim and root link_
                    if _N in N_:  # in both N rim and root link_
                        oEt = np.add(oEt, L.Et)
                        olink_ += [L]
                if olink_:
                    OEt = np.add(OEt,oEt)
                    oEt_ += [oEt]
                # feedback to roots: add/remove N from rN_, adjusting next-loop overlap:
                if oEt[fd] > ave * oEt[2+fd]:  # N in root
                    if N not in N_:
                        nroot_ += [i]; N_ += [N]; nEt_ += [oEt]; link_[:] = list(set(link_).union(olink_))  # not directional
                elif N in N_:
                    Ni = N_.index(N); nroot_.pop(Ni); nEt_.pop(Ni); N_.remove(N); link_[:] = list(set(link_).difference(olink_))
            _oEt_[:] = oEt_
        r += 1
        if OEt[fd] - _OEt[fd] < ave:  # low total overlap update
            break
        _OEt = OEt

    # exclusive N assignment to clusters based on final oV:
    for N,rim, _N_, nroot_,oEt_ in node_:
         if len(nroot_) == 1:
             continue  # keep isolated Ns
         nroot_ = [root_[i] for i in nroot_]
         nroot_ = sorted(nroot_, key=lambda root: root[2][root[1].index(N)][fd])  # sort by NoV in roots
         N.root = nroot_.pop()  # max root
         for root in nroot_:  # remove N from other roots
             root[0][:] = list(set(root[0]).difference(rim))  # remove rim
             i = root[1].index(N); root[1].pop(i); root[2].pop(i)  # remove N and NoV
    '''
    This pruning should actually be recursive too, it reduces overlap. May as well merge graphs via floodfill.
    Which is sequential, the only parallelization is reusing merged-graph processor for some other root graph. 
    Or subdividing processing of large graphs, but that's a lot of overhead.
    '''
    return [sum2graph(root, grapht, fd, nrng) for grapht in root_ if grapht[1]]  # not-empty clusters


def get_rim(N, fd):
    if fd:  # N is Clink
        if isinstance(N.node_[0],CG):
            rim = [G.rim for G in N.node_]
        else:  # get link node rim_t dirs opposite from each other, else covered by the other link rim_t[1-dir]?
            rim = [(link.rim_t[1-dir][-1] if link.rim_t[dir] else []) for dir, link in zip((0,1), N.node_)]
    else:   rim = N.rim

    return rim

def segment_graph(root, Q, fd, nrng):  # recursive eval node_|link_ rims for cluster assignment
    '''
    kernels = get_max_kernels(Q)  # parallelization of link tracing, not urgent
    grapht_ = select_merge(kernels)  # refine by match to max, or only use it to selectively merge?
    '''
    _node_ = G_ = copy(Q)
    r = 0
    grapht_ = []
    while True:
        node_ = []
        for node in _node_:  # depth-first eval merge node_|link_ connected via their rims:
            if node not in G_: continue  # merged?
            if not node.root:  # init in 1st loop or empty root after root removal
                grapht = [[node],[],[0,0,0,0]]  # G_, Link_, Et
                grapht_ += [grapht]
                node.root = grapht
            upV, remaining_node_ = merge_node(grapht_, G_, node, fd, upV=0)  # G_ to remove merged node
            if upV > ave:  # graph update value, accumulate?
                node_ += [node]
            G_ += remaining_node_; node_ += remaining_node_    # re-add for next while loop
        if node_:
            _node_ = G_ = copy(node_)
        else:
            break  # no updates
        r += 1  # recursion count
    graph_ = [sum2graph(root, grapht[:3], fd, nrng) for grapht in grapht_ if  grapht[2][fd] > ave * grapht[2][2+fd]]

    return graph_

# not revised, sequential version
def merge_node(grapht_, iG_, G, fd, upV):

    if G in iG_: iG_.remove(G)
    G_, Link_, Et = G.root
    ave = G_aves[fd]
    remaining_node_ = []  # from grapht removed from grapht_
    fagg = isinstance(G,CG)

    for link in (G.rim if fagg else G.rim_t[0][-1] if G.rim_t[0] else [] + G.rim_t[1][-1] if G.rim_t[1] else []):  # be empty
        if link.Et[fd] > ave:  # fork eval
            if link not in Link_: Link_ += [link]
            _G = link.node_[0] if link.node_[1] is G else link.node_[1]
            if _G in G_:
                continue  # _G is already in graph
            # get link overlap between graph and node.rim:
            olink_ = list(set(Link_).intersection(_G.rim))
            oV = sum([olink.Et[fd] for olink in olink_])  # link overlap V
            if oV > ave and oV > _G.Et[fd]:  # higher inclusion value in new vs. current root
                upV += oV
                _G.Et[fd] = oV  # graph-specific, rdn?
                G_ += [_G]; Link_ += [link]; Et = np.add(Et,_G.Et)
                if _G.root:  # remove _G from root
                    _G_, _Link_, _Et = _G.root
                    _G_.remove(_G)
                    if link in _Link_: _Link_.remove(link)  # empty at grapht init in r==0
                    _Et = np.subtract(_Et, Et)
                    if _Et[fd] < ave*_Et[2+fd]:
                        grapht_.remove(_G.root)
                        for __G in _G_: __G.root = []   # reset root
                        remaining_node_ += _G_

                _G.root = G.root  # temporary
                upV, up_remaining_node_ = merge_node(grapht_, iG_, _G, fd, upV)
                remaining_node_ += up_remaining_node_

    return upV, [rnode for rnode in remaining_node_ if not rnode.root]  # skip node if they added to grapht during the subsequent merge_node process

# compatible with segment_parallel
def sum2graph(root, grapht, fd, nrng):  # sum node and link params into graph, aggH in agg+ or player in sub+

    Link_, G_, oEt_ = grapht
    graph = CG(fd=fd, node_=G_,link_=Link_, rng=nrng)
    if fd:
        graph.root = root
    extH = CH()
    for G, oEt in zip(G_, oEt_):
        graph.Et = np.add(graph.Et, oEt)
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

# not updated, for parallelization only
def get_max_kernels(G_):  # use local-max kernels to init sub-graphs for segmentation

    kernel_ = []
    for G in copy(G_):
        _G_ = []; fmax = 1
        for link in G.rim:
            _G = link.node_[0] if link.node_[1] is G else link.node_[1]
            if _G.DerH.Et[0] > G.DerH.Et[0]:  # kernel-specific
                fmax = 0
                break
            _G_ += [_G]
        if fmax:
            kernel = [G]+_G_  # immediate kernel
            for k in kernel:
                k.root += [kernel]  # node in overlapped area may get more than 1 kernel
                if k in G_: G_.remove(k)
            kernel_ += [kernel]
    for G in G_:  # remaining Gs are not in any kernels, append to the nearest kernel
        _G_ = [link.node_[0] if link.node_[1] is G else link.node_[1] for link in G.rim]  # directly connected Gs already checked
        __G_ = []
        while _G_:
            while True:
                for _G in _G_:
                    for link in _G.rim:
                        __G = link.node_[0] if link.node_[1] is _G else link.node_[1]  # indirectly connected Gs
                        if __G not in G_:  # in some kernel, append G to it:
                            G.root = __G.root
                            __G.root[-1] += [G]
                            break
                        __G_ += [__G]
                _G_ = __G_
    # each kernel may still have overlapped nodes
    return kernel_

# very initial draft, to merge overlapped kernels and form grapht
def select_merge(kernel_):

    for kernel in copy(kernel_):
        for node in copy(kernel):
            for _kernel in copy(node.root):  # get overlapped _kernel of kernel
                if _kernel is not kernel and _kernel in kernel_:  # not a same kernel and not a merged kernel
                    for link in node.rim:  # get link between 2 centers
                        if kernel[0] in link.node_ and _kernel[0] in link.node_:
                            break
                    if link.ExtH.Et[0] > ave:  # eval by center's link's ExtH?
                        for _node in _kernel:  # merge _kernel into kernel
                            if _node not in kernel:
                                kernel += [_node]
                                _node.root = kernel
                        kernel_.remove(_kernel)  # remove merged _kernel
            node.root = kernel  # remove list kernel
    grapht_ = []
    for kernel in kernel_:
        Et =  [sum(node.Et[0] for node in kernel), sum(node.Et[1] for node in kernel)]
        rim = list(set([link for node in kernel for link in node.rim]))
        grapht = [kernel, [], Et, rim]  # not sure
        grapht_ += [grapht]

    return grapht_