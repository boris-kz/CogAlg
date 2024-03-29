import numpy as np
from copy import deepcopy, copy
from itertools import combinations, zip_longest
from .slice_edge import comp_angle, CsliceEdge
from .comp_slice import ider_recursion, comp_latuple, get_match
from .filters import aves, ave_mL, ave_dangle, ave, G_aves, ave_Gm, ave_Gd, ave_dist, max_dist
from utils import box2center, extend_box
import sys
sys.path.append("..")
from frame_blobs import CH, CG, Clink


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

    frame = CsliceEdge(image).slice()

    for edge in frame.edge_:
        if edge.latuple[-1] * (len(edge.P_)-1) > G_aves[0]:
            # if G in latuple, rdn=1
            ider_recursion(None, edge)  # vertical, lateral-overlap P cross-comp -> PP clustering

            for fd, node_ in enumerate(edge.node_):  # always node_t
                if edge.iderH and edge.iderH.Et:
                    if edge.iderH.Et[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.iderH.Et[2+fd]:
                        pruned_node_ = []
                        for PP in node_:  # PP -> G
                            if PP.iderH and PP.iderH.Et[fd] > G_aves[fd] * PP.iderH.Et[2+fd]:
                                PP.root = None  # no feedback to edge?
                                PP.node_ = PP.P_  # revert base node_
                                PP.Et = [0,0,0,0]  # [] in comp_slice
                                pruned_node_ += [PP]
                        if len(pruned_node_) > 10:  # discontinuous PP rng+ cross-comp, cluster -> G_t:
                            # agg+ of PP nodes:
                            agg_recursion(None, edge, Q=list(combinations(pruned_node_,r=2)), nrng=1, fagg=1)


def agg_recursion(rroot, root, Q, nrng=1, fagg=0):  # lenH = len(root.aggH[-1][0]), lenHH: same in agg_compress

    Et = [0,0,0,0]  # eval tuple, sum from Link_
    # agg+ der=1 xcomp of new Gs if fagg, else sub+: der+ xcomp of old Gs,
    nrng, node_, Et = rng_recursion(rroot, root, Q, Et, nrng=nrng)  # rng+ appends prelink_ -> rim, link.dderH

    node_t = form_graph_t(root, node_, Et, nrng, fagg)  # root_fd, eval der++ and feedback per Gd, not sub-recursion in Gms
    if node_t:
        for fd, node_ in enumerate(node_t):
            if root.Et[0] * (len(node_)-1)*root.rng > G_aves[1] * root.Et[2]:
                # agg+ / node_t, vs. sub+ / node_, always rng+:
                pruned_node_ = [node for node in node_ if node.Et[0] > G_aves[fd] * node.Et[2]]
                if len(pruned_node_)>10:
                    agg_recursion(rroot, root, Q=list(combinations(pruned_node_,r=2)), nrng=1, fagg=1)
                    if rroot and fd and root.derH:  # der+ only (check not empty root.derH)
                        rroot.fback_ += [root.derH]
                        feedback(rroot)  # update root.root..


def rng_recursion(rroot, root, Q, iEt, nrng=1):  # rng++/G_, der+/link_ in sub+, -> rim_H

    fd = isinstance(Q[0],Clink)  # else [G,_G]
    Et = [0,0,0,0]  # for rng+
    node_ = []  # for rng+, append inside comp_G

    if fd:  # only in 1st rng+ from der+, extend root links
        for link in Q:
            if link.dderH.Et[1] > G_aves[1] * link.dderH.Et[3]:  # eval der+
                comp_G(link, node_, Et)
    else:
        for _G,G in Q:  # prelinks in rng+
            if _G in G.compared_: continue
            cy, cx = box2center(G.box); _cy, _cx = box2center(_G.box); dy = cy - _cy; dx = cx - _cx
            dist = np.hypot(dy, dx)  # distance between node centers
            if nrng > 1:  # pair eval:
                _M,_R, M,R = _G.Et[0],_G.Et[2], G.Et[0],G.Et[2]
            if nrng==1 or ((M+_M)/ (dist/ave_dist) > ave*(R+_R)):  # or directional?
                G.compared_+=[_G]; _G.compared_+=[G]
                comp_G([_G,G, dist, [dy,dx]], node_, Et)

    if Et[0] > ave_Gm * Et[2]:
        # rng+ eval per arg cluster because comp is bilateral, 2nd test per new pair
        iEt[:] = [V+v for V,v in zip(iEt, Et)]  # Vt[i]+=v; Rt[i]+=rt[i]; Dt[i]+=d
        if node_:  # eval rng+
            node_ = list(set(node_))
            nrng,_,_ = rng_recursion(rroot, root, list(combinations(node_,r=2)), iEt, nrng+1)

    return nrng, node_, Et


def comp_G(link, node_, iEt, nrng=None):  # add flat dderH to link and link to the rims of comparands

    dderH = CH()  # new layer of link.dderH
    if isinstance(link, Clink):
        # der+
        _G,G = link._node,link.node; rn = _G.n/G.n  # fd=1
    else:  # rng+
        _G,G, dist, [dy,dx] = link; rn = _G.n/G.n  # fd=0
        link = Clink(_node=_G, node=G, distance=dist, angle=[dy,dx])
        # / P
        Et, md_ = comp_latuple(_G.latuple, G.latuple, rn, fagg=1)
        dderH.n = 1; dderH.Et = Et
        dderH.H = [CH(nest=0, Et=[*Et], H=md_)]
        comp_ext(_G,G, dist, rn, dderH)
        # / PP, if >1 Ps:
        if _G.iderH and G.iderH: _G.iderH.comp_(G.iderH, dderH, rn, fagg=1, flat=0)
    # / G, if >1 PPs | Gs:
    if _G.extH and G.extH: _G.extH.comp_(G.extH, dderH, rn, fagg=1, flat=0)  # always true in der+
    if _G.derH and G.derH: _G.derH.comp_(G.derH, dderH, rn, fagg=1, flat=0)

    link.dderH.append_(dderH, flat=len(link.dderH.H)>0)  # append for higher-res lower-der summation in sub-G extH
    for i in 0,1:
        Val, Rdn = dderH.Et[i:4:2]  # exclude dect
        if Val > G_aves[i] * Rdn:
            if not i: node_ += [_G,G]  # for rng+;  if fd: comp_rim(node_,link, nrng)  # rng+/ matching-direction rim _Gs only?
            _G.Et[i]  += Val; G.Et[i]  += Val  # in both Gs
            _G.Et[2+i]+= Rdn; G.Et[2+i]+= Rdn
            # to eval rng+ and grapht in form_graph_t:
            iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dderH.Et[i::2])]  # or sum all for form_graph_t?
            if not i:
                for G in link.node, link._node:
                    rim_H = G.rim_H
                    if rim_H and isinstance(rim_H[0],list):  # rim is converted to rim_H in 1st sub+
                        if len(rim_H) == len(G.Rim_H): rim_H += [[]]  # no new rim layer yet
                        rim_H[-1] += [link]  # rim_H
                    else:
                        rim_H += [link]  # rim

def comp_ext(_G,G, dist, rn, dderH):  # compare non-derivatives: dist, node_' L,S,A:

    prox = ave_dist - dist  # proximity = inverted distance (position difference), no prior accum to n
    _L = len(_G.node_); L = len(G.node_); L/=rn
    _S, S = _G.S, G.S; S/=rn

    dL = _L - L;      mL = min(_L,L) - ave_mL  # direct match
    dS = _S/_L - S/L; mS = min(_S,S) - ave_mL  # sparsity is accumulated over L
    mA, dA = comp_angle(_G.A, G.A)  # angle is not normalized

    M = prox + mL + mS + mA
    D = dist + abs(dL) + abs(dS) + abs(dA)  # signed dA?
    mrdn = M > D; drdn = D<= M

    mdec = prox / max_dist + mL/ max(L,_L) + mS/ max(S,_S) + mA  # Amax = 1
    ddec = dist / max_dist + mL/ (L+_L) + dS/ (S+_S) + dA

    dderH.append_(CH(Et=[M,D,mrdn,drdn,mdec,ddec], H=[prox,dist, mL,dL, mS,dS, mA,dA], n=2/3), flat=0)  # 2/3 of 6-param unit


def form_graph_t(root, G_, Et, nrng, fagg=0):  # form Gm_,Gd_ from same-root nodes

    node_connect(G_)  # Graph Convolution of Correlations over init _G_
    node_t = []
    for fd in 0,1:
        if Et[fd] > ave * Et[2+fd]:  # eVal > ave * eRdn
            graph_ = segment_node_(root, G_, fd, nrng, fagg)  # fd: node-mediated Correlation Clustering
            if fd:  # der+ only, rng++ term by high diffs, can't be extended much in sub Gs
                for graph in graph_:
                    if graph.Et[1] > G_aves[1] * graph.Et[3]:
                        node_ = graph.node_  # not needed?
                        if isinstance(node_[0].rim_H[0], CG):  # 1st sub+, same rim nesting?
                            for node in node_: node.rim_H = [node.rim_H]  # rim -> rim_H
                            agg_recursion(root, graph, graph.link_, nrng, fagg=0)
                    elif graph.derH:
                        root.fback_ += [graph.derH]
                        feedback(root)  # update root.root.. per sub+
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]
    if any(node_t):
        G_[:] = node_t  # else keep root.node_
        return node_t


def node_connect(iG_):  # node connectivity = sum surround link vals, incr.mediated: Graph Convolution of Correlations
    '''
    Aggregate direct * indirect connectivity per node from indirect links via associated nodes, in multiple cycles.
    Each cycle adds contributions of previous cycles to linked-nodes connectivity, propagated through the network.
    Math: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/node_connect.png
    '''
    _G_ = iG_
    while True:
        # eval accumulated G connectivity with node-mediated range extension
        G_ = []  # next connectivity expansion, more selective by DV,Lent
        mediation = 1  # n intermediated nodes, increasing decay
        for G in _G_:
            uprim = []  # >ave updates of direct links
            rim = G.rim_H[-1] if G.rim_H and isinstance(G.rim_H[0], list) else G.rim_H
            for i in 0,1:
                val,rdn = G.Et[i::2]  # rng+ for both segment forks
                ave = G_aves[i]
                for link in rim:
                    # > ave derGs in fd rim
                    lval,lrdn,ldec = link.dderH.Et[i::2]  # step=2
                    decay =  (ldec/ (link.dderH.n * 6)) ** mediation  # normalized decay at current mediation
                    _G = link._node if link.node is G else link.node
                    _val,_rdn = _G.Et[i::2]
                    # current-loop vals and their difference from last-loop vals, before updating:
                    V = (val+_val) * decay; dv = V-lval
                    R = (rdn+_rdn)  # rdn doesn't decay
                    link.dderH.Et[i:4:2] = [V,R]  # last-loop vals for next loop | segment_node_, dect is not updated
                    if dv > ave * R:  # extend mediation if last-update val, may be negative
                        G.Et[i::2] = [V+v for V,v in zip(G.Et[i::2],[V,R])]  # last layer link vals
                        if link not in uprim: uprim += [link]
                    if V > ave * R:  # updated even if terminated
                        G.Et[i::2] = [V+v for V,v in zip(G.Et[i::2], [dv,R])]  # use absolute R?
            if uprim:  # prune rim for next loop
                rim[:] = uprim
                G_ += [G]
        if G_:
            mediation += 1  # n intermediated nodes in next loop
            _G_ = G_  # exclude weakly incremented Gs from next connectivity expansion loop
        else:
            break

def segment_node_(root, root_G_, fd, nrng, fagg):  # eval rim links with summed surround vals for density-based clustering

    # graph += [node] if >ave (surround connectivity * relative value of link to any internal node)
    igraph_ = []; ave = G_aves[fd]

    for G in root_G_:  # init per node
        link_ = copy(G.rim_H[-1] if G.rim_H and isinstance(G.rim_H[0],list) else G.rim_H)
        grapht = [[G],[], [*G.Et], link_]  # link_ = last rim
        G.root = grapht  # for G merge
        igraph_ += [grapht]
    _graph_ = copy(igraph_)

    while True:
        graph_ = []
        for grapht in _graph_:  # extend grapht Rim with +ve in-root links
            if grapht not in igraph_: continue  # skip merged graphs
            G_, Link_, Et, Rim = grapht
            inVal, inRdn = 0,0  # new in-graph +ve
            new_Rim = []
            for link in Rim:  # unique links
                if link.node in G_:  # one of the nodes is already clustered
                    G = link.node; _G = link._node
                else:
                    G = link._node; _G = link.node
                if _G in G_: continue
                # connect by rel match of nodes * match of node Vs: surround M|Ds,
                # cval suggests how deeply inside the graph is G:
                cval = link.dderH.Et[fd] + get_match(_G.Et[fd], G.Et[fd])  # same coef for int and ext match?
                crdn = link.dderH.Et[2+fd] + (_G.Et[2+fd] + G.Et[2+fd]) / 2
                if cval > ave * crdn:  # _G and its root are effectively connected
                    # merge _G.root in grapht:
                    _grapht = _G.root  # local and for feedback?
                    _G_,_Link_,_Et,_Rim = _grapht
                    if link not in Link_: Link_ += [link]
                    Link_[:] += [_link for _link in _Link_ if _link not in Link_]
                    for g in _G_:
                        g.root = grapht
                        if g not in G_: G_+=[g]
                    Et[:] = [V+v for V,v in zip(Et,_Et)]
                    inVal += _Et[fd]; inRdn += _Et[2+fd]
                    igraph_.remove(_grapht)
                    new_Rim += [link for link in _Rim if link not in new_Rim+Rim+Link_]
            # for next loop:
            if len(new_Rim) * inVal > ave * inRdn:
                grapht.pop(-1); grapht += [new_Rim]
                graph_ += [grapht]  # replace Rim

        if graph_: _graph_ = graph_  # selected graph expansion
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
        sum_last_lay(G) # G.extH += links dderH last layer
        graph.box = extend_box(graph.box, G.box)
        graph.latuple = [P+p for P,p in zip(graph.latuple[:-1],graph.latuple[:-1])] + [[A+a for A,a in zip(graph.latuple[-1],graph.latuple[-1])]]
        if G.iderH:  # empty in single-P PP|Gs
            graph.iderH.add_(G.iderH)
        if G.derH:  # empty in single-PP Gs
            graph.derH.add_(G.derH)
        if fd: G.Et = [0,0,0,0]  # reset in last form_graph_t fork, Gs are shared in both forks
        graph.n += G.n  # non-derH accumulation?
    extH = CH()
    for link in Link_:  # unique current-layer links
        last_lay = link.dderH.H[int(len(link.dderH.H)/2):]  # add last layer only, packed flat
        last_lay = CH(Et=np.sum([He.Et for He in last_lay],axis=0), H=last_lay, nest=last_lay[0].nest, n=sum([He.n for He in last_lay]))
        extH.add_( last_lay, irdnt=link.dderH.Et[2:4])
        graph.S += link.distance
        np.add(graph.A,link.angle)
        link.root = graph
    graph.derH.append_(extH, flat=0)  # graph derH = node derHs + [summed Link_ dderHs]

    if fd:  # assign alt graphs from d graph, after both linked m and d graphs are formed
        for link in graph.link_:
            mgraph = link.roott[0]
            if mgraph:
                for fd, (G, alt_G) in enumerate(((mgraph,graph), (graph,mgraph))):  # bilateral assign:
                    if G not in alt_G.alt_graph_:
                        G.alt_graph_ += [alt_G]
    return graph


def sum_last_lay(G):  # G.extH += last layer of link.daggH (dsubH|ddaggH)

    dderH = CH()
    for link in G.rim_H[-1] if G.rim_H and isinstance(G.rim_H[0],list) else G.rim_H:  # last link layer
        if link.dderH:
            last_lay = link.dderH.H[int(len(link.dderH.H)/2):]  # dderH layers are packed flat
            last_lay = CH(Et=np.sum([He.Et for He in last_lay],axis=0), H=last_lay, nest=last_lay[0].nest, n=sum([He.n for He in last_lay]))
            dderH.add_(last_lay, irdnt=link.dderH.Et[2:4])
    if dderH:
        G.extH.add_(dderH)


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