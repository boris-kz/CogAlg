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
                                PP.root = None  # no feedback to edge?
                                PP.node_ = PP.P_  # revert base node_
                                PP.Et = [0,0,0,0]  # [] in comp_slice
                                pruned_node_ += [PP]
                        if len(pruned_node_) > 10:  # discontinuous PP rng+ cross-comp, cluster -> G_t:
                            edge.node_ = pruned_node_  # agg+ of PP nodes:
                            agg_recursion(None, edge, fagg=1)

def agg_recursion(rroot, root, fagg=0):

    for link in root.link_:
        link.Et = copy(link.derH.Et); link.relt = copy(link.derH.relt)
        # += connected nodes:
    nrng, Et = rng_convolve(root, [0,0,0,0], fagg)
    # agg+'upG_|der+'upL_:
    upQ = []
    for G in root.node_ if fagg else root.link_:
        if sum(G.Et[:2]):  # G.rim was extended, sum in G.extH:
            for link in G.rim if fagg else G.rimt__[-1][-1][0] + G.rimt__[-1][-1][1]:
                if G.extH: G.extH.H[-1].add_(link.derH.H[-1],irdnt=link.derH.H[-1].Et[2:])  # sum last layer
                else:      G.extH.append_(link.derH.H[-1],flat=0)  # pack last layer
            upQ += [G]
    node_t = form_graph_t(root, upQ, Et, nrng)  # root_fd, eval der++ and feedback per Gd only
    if node_t:
        for fd, node_ in enumerate(node_t):
            if root.Et[0] * (len(node_)-1)*root.rng > G_aves[1] * root.Et[2]:
                # agg+ / node_t, vs. sub+ / node_, always rng+:
                pruned_node_ = [node for node in node_ if node.Et[0] > G_aves[fd] * node.Et[2]]  # not be needed?
                if len(pruned_node_) > 10:
                    agg_recursion(rroot, root, fagg=1)
                    if rroot and fd and root.derH:  # der+ only (check not empty root.derH)
                        rroot.fback_ += [root.derH]
                        feedback(rroot)  # update root.root..


def rng_convolve(root, Et, fagg):  # comp Gs|kernels in agg+, links | link rim__t node rims in sub+

    nrng = 1
    if fagg:  # comp CG
        G_ = []  # initialize kernels:
        for link in list(combinations(root.node_,r=2)):
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
                if comp_G(Link, Et, fd=0):
                    for node in _G,G:
                        node.rim += [Link]
                        if node not in G_: G_ += [node]
        for G in G_:
            G.kH = [[link.node_[0] if link.node_[1] is G else link.node_[1] for link in G.rim]]  # add 1st kernel layer
        while len(G_) > 2:  # rng+ with kernel rims formed per loop
            _G_ = []
            for G in G_:
                for link in G.rim:
                    if link.Et[0] > ave:  # link.Et+ per rng
                        comp_kernel(link, _G_, nrng) # sum full kernel: link val = rel node similarity * connectivity?
            nrng += 1
            G_ = _G_
    else:  # comp Clinks: der+'rng+ in root.link_ rim__t node rims
        link_ = root.link_
        _link_ = []
        while link_:
            for link in link_:  # der+'rng+: directional and node-mediated comp link
                for dir, rim__ in zip((0,1), link.rim__t):  # two directions of layers, each nested by mediating links
                    for _L in [L for rim in rim__[-1] for L in rim]:  # last nested layer
                        _L_ = []
                        _G = _L.node_[0] if _L.node_[1] in link.node_ else _L.node_[1]
                        for _link in _G.rim:
                            Link = Clink(node_=[link,_link])
                            if comp_G(Link, Et, fd=1):
                                _link_ += [link]  # old link
                                _L_ += [Link]  # new link
                        if _L_: link.rim__t[dir][-1] += [_L_]  # += list( matching _L-mediated links)
            nrng += 1
            link_ = _link_
    return nrng, Et

def comp_kernel(link, G_, nrng, fd=0):

    _G, G = link.node_  # same direction
    ave = G_aves[fd]
    _n,_L,_S,_A,_latuple,_iderH,_derH,_Et = sum_kernel(_G.kH, G.kH)
    n, L, S, A, latuple, iderH, derH, Et  = sum_kernel(G.kH, _G.kH)
    rn = _n/n
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
    if dderH.Et[0] > ave * dderH.Et[2]:
        link.ExtH.H[-1].add_(dderH, irdnt=dderH.H[-1].Et[2:]) if len(link.ExtH.H)==nrng else link.ExtH.append_(dderH,flat=1)
        for node in _G, G:
            if node in G_: continue  # new kernel is already added
            kLayer = []
            for _link in node.rim:
                _node = _link.node_[0] if _link.node_[1] is node else _link.node_[1]
                kLayer += [_G for _G in _node.kH[-1] if _G not in kLayer]
            node.kH += [kLayer]  # last kernel rim
            G_ += [node]
    # node connectivity eval in segment_graph only via decay = (link.relt[fd] / (link.derH.n * 6)) ** nrng  # normalized decay at current mediation


def sum_kernel(_kH, kH=[]):  # sum last kernel layer
    if not kH: kH = [[] for _ in _kH]

    for i, (_kernel, kernel) in enumerate(zip(_kH[-1], kH[-1])):
        _kernel = list(set(_kernel)-set(kernel))  # skip overlap
        if i == 0:
            _G = _kernel[0]
            n, L, S, A = _G.n, len(_G.node_), _G.S, _G.A
            latuple = deepcopy(_G.latuple)
            iderH = deepcopy(_G.iderH)
            derH = deepcopy(_G.derH)
            Et = copy(_G.Et)
        for G in _kernel[1:]:
            latuple = [P+p for P,p in zip(latuple[:-1],G.latuple[:-1])] + [[A+a for A,a in zip(latuple[-1],G.latuple[-1])]]
            n += G.n;  L += len(G.node_); S += G.S;  A =  [ Angle+angle for Angle, angle in zip(A, G.A)]
            if G.iderH: iderH.add_(G.iderH)
            if G.derH:  derH.add_(G.derH)
            np.add(Et,G.Et)

    return n, L, S, A, latuple, iderH, derH, Et  # not sure about Et


def comp_G(link, iEt, fd):  # add dderH to link and link to the rims of comparands, which may be Gs or links

    dderH = CH()  # new layer of link.dderH
    _G, G = link.node_
    if fd:  # Clink Gs
        rn= min(_G.node_[0].n,_G.node_[1].n)/ min(G.node_[0].n,G.node_[1].n)
        Et, rt, md_ = comp_ext(_G.distance,G.distance, len(_G.rim__t[0][-1])+len(_G.rim__t[1][-1]),len(G.rim__t[0][-1])+len(G.rim__t[1][-1]), _G.angle,G.angle)
        dderH.n = 1; dderH.Et = Et; dderH.relt = rt
        dderH.H = [CH(Et=copy(Et), relt=copy(rt), H=md_, n=1)]
    else:  # CG Gs
        rn= _G.n/G.n  # comp ext params prior: _L,L,_S,S,_A,A, dist, no comp_G unless match:
        et, rt, md_ = comp_ext(len(_G.node_),len(G.node_),_G.S,G.S/rn,_G.A,G.A)
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
    for i in 0, 1:
        Val, Rdn = dderH.Et[i::2]
        if Val > G_aves[i] * Rdn: fin = 1
        _G.Et[i] += Val; G.Et[i] += Val  # not selective
        _G.Et[2+i] += Rdn; G.Et[2+i] += Rdn  # per fork link in both Gs
        # if select fork links: iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dderH.Et[i::2])]
    if fin: return link


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


def form_graph_t(root, upQ, Et, nrng):  # form Gm_,Gd_ from same-root nodes

    node_t = []
    for fd in 0, 1:
        if Et[fd] > ave * Et[2+fd]:  # eVal > ave * eRdn
            graph_ = segment_graph(root, upQ, fd, nrng)
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


def get_max_kernels(graph):  # use local-max kernels to init sub-graphs for segmentation

    G_ = copy(graph.node_)
    kernel_ = []
    for G in G_:
        _G_ = []; fmax = 1
        for link in G.rim:
            _G = link.node_[0] if link.node_[1] is G else link.node_[1]
            if _G.Et[0] > G.Et[0]:
                fmax = 0
                break
            _G_ += [_G]
        if fmax:
            kernel = [G]+_G_  # immediate kernel
            for k in kernel:
                k.root = kernel
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
                            __G.root += [G]
                            break
                        __G_ += [__G]
                _G_ = __G_
    return kernel_

# not updated:
def segment_graph(root, Q, fd, nrng):  # eval rim links with summed surround vals for density-based clustering

    kernels = get_max_kernels(Q)  # may replace current segmentation not sure
    grapht_ = select_merge(kernels)  # refine by match to max, or only use it to selectively merge?

    # graph += [node] if >ave (surround connectivity * relative value of link to any internal node)
    igraph_ = []; ave = G_aves[fd]

    for e in Q:  # init per updated node or link
        uprim = e.rim if isinstance(e, CG) else e.rimt__[-1][-1][0] + e.rimt__[-1][-1][1]
        if uprim:  # skip nodes without add new added rim
            grapht = [[e],[],[*e.Et], uprim]  # link_ = updated rim
            e.root = grapht  # for merging
            igraph_ += [grapht]
        else: e.root = None
    _graph_ = copy(igraph_)

    while True:
        graph_ = []
        for grapht in _graph_:  # extend grapht Rim with +ve in-root links
            # not revised:
            if grapht not in igraph_: continue  # skip merged graphs
            G_, Link_, Et, Rim = grapht
            inVal, inRdn = 0,0  # new in-graph +ve
            new_Rim = []
            for link in Rim:  # unique links
                if link.node_[0] in G_: G,_G = link.node_  # one of the nodes is already clustered
                else:                   _G,G = link.node_
                if _G in G_: continue
                '''
                eval links by a combination of direct and node-mediated connectivity: sum rng+ kernels
                not sure about refining by in-graph kernels, recursive because each redefines the graph.
                
                _val,_rdn = _G.Et[fd::2]; val,rdn = G.Et[fd::2]  # or ExtH.H[-1].Et?
                lval,lrdn = link.Et[fd::2]; cval,crdn = dderH.Et[fd::2]
                lval+=cval; lrdn+=crdn # += current comp vals
                decay = (link.relt[fd] / (link.derH.n * 6)) ** nrng  # normalized decay at current mediation
                V = lval + ((val+_val) * decay) * .1  # med connectivity coef?
                R = lrdn + (rdn+_rdn) * .1  # no decay
                # dv = V - lval?
                if V > ave * R:  # for positive links only
                    link.ExtH.add_(dderH)
                    link.Et[fd] = V, link.Et[2+fd] = R
                    # separate eval to update node.Et?
                '''
                # connect by rel match of nodes * match of node Vs: surround M|Ds,
                # cval suggests how deeply inside the graph is G:
                cval = link.Et[fd] + get_match(_G.Et[fd], G.Et[fd])  # same coef for int and ext match?
                crdn = link.Et[2+fd] + (_G.Et[2+fd] + G.Et[2+fd]) / 2
                if cval > ave * crdn:  # _G and its root are effectively connected
                    # merge _G.root in grapht:
                    if _G.root:
                        _grapht = _G.root  # local and for feedback?
                        _G_,_Link_,_Et,_Rim = _grapht
                        if link not in Link_: Link_ += [link]
                        Link_[:] += [_link for _link in _Link_ if _link not in Link_]
                        for g in _G_:
                            g.root = grapht
                            if g not in G_: G_+=[g]
                        Et[:] = np.add(Et,_Et)
                        inVal += _Et[fd]; inRdn += _Et[2+fd]
                        igraph_.remove(_grapht)
                        new_Rim += [link for link in _Rim if link not in new_Rim+Rim+Link_]
                    else:  # _G doesn't have uprim and doesn't form any grapht
                        _G.root = grapht
                        G_ += [_G]
            # for next loop:
            if len(new_Rim) * inVal > ave * inRdn:
                grapht.pop(-1); grapht += [new_Rim]
                graph_ += [grapht]  # replace Rim
        # select graph expansion:
        if graph_: _graph_ = graph_
        else: break
    graph_ = []
    for grapht in igraph_:
        if grapht[2][fd] > ave * grapht[2][2+fd]:  # form Cgraphs if Val > ave* Rdn
            graph_ += [sum2graph(root, grapht[:3], fd, nrng)]

    return graph_


def sum2graph(root, grapht, fd, nrng):  # sum node and link params into graph, aggH in agg+ or player in sub+

    G_, Link_, Et = grapht
    graph = CG(fd=fd, Et=Et, node_=G_,link_=Link_, rng=nrng)
    if fd:
        graph.root = root
    for G in G_:
        graph.area += G.area
        graph.box = extend_box(graph.box, G.box)
        if isinstance(G, CG):
            graph.latuple = [P+p for P,p in zip(graph.latuple[:-1],G.latuple[:-1])] + [[A+a for A,a in zip(graph.latuple[-1],G.latuple[-1])]]
            if G.iderH:  # empty in single-P PP|Gs
                graph.iderH.add_(G.iderH)
        graph.n += G.n  # non-derH accumulation?
        graph.derH.add_(G.derH)
        if fd: G.Et = [0,0,0,0]  # reset in last form_graph_t fork, Gs are shared in both forks
        else:  G.root = graph    # assigned to links if fd else to nodes?
    extH = CH()
    for link in Link_:  # sum last layer of unique current-layer links
        if len(extH.H)==len(link.derH.H): extH.H[-1].add_(link.derH.H[-1], irdnt=link.derH.H[-1].Et[2:4])  # sum last layer
        else:                             extH.append_(link.derH.H[-1],flat=0)  # pack last layer
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