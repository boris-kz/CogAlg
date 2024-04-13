import numpy as np
from copy import deepcopy, copy
from itertools import combinations, zip_longest
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
        if edge.latuple[-1] * (len(edge.P_)-1) > G_aves[0]:  # eval G, rdn=1
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
                            # agg+ of PP nodes:
                            agg_recursion(None, edge, node_ = pruned_node_, Q=list(combinations(pruned_node_,r=2)), nrng=1, fagg=1)


def agg_recursion(rroot, root, node_, Q, nrng=1, fagg=0):  # lenH = len(root.aggH[-1][0]), lenHH: same in agg_compress

    Et = [0,0,0,0]  # eval tuple, sum from Link_

    if fagg: node_ = root.node_  # agg++
    else: node_ = root.link_  # sub+
    nrng, Et = rng_recursion(rroot, root, node_, Q, Et, nrng=nrng)  # rng+ appends prelink_ -> rim, link.derH

    for link in root.link_:
        link.Et = copy(link.derH.Et); link.relt = copy(link.derH.relt)  # for accumulation from surrounding nodes in convolve_graph

    convolve_graph(node_)  # convolution over graph node_|link_
    upnode_ = []  # uplink_ in der+
    for G in node_:
        if sum(G.Et[:2]):  # G.rim was extended, sum in G.extH:
            for link in G.rim:
                if len(G.extH.H)==len(link.derH.H): G.extH.H[-1].add_(link.derH.H[-1],irdnt=link.derH.H[-1].Et[2:4])  # sum last layer
                else:                               G.extH.append_(link.derH.H[-1],flat=0)  # pack last layer
            upnode_ += [G]
    node_t = form_graph_t(root, upnode_, Et, nrng, fagg)  # root_fd, eval der++ and feedback per Gd only
    if node_t:
        for fd, node_ in enumerate(node_t):
            if root.Et[0] * (len(node_)-1)*root.rng > G_aves[1] * root.Et[2]:
                # agg+ / node_t, vs. sub+ / node_, always rng+:
                pruned_node_ = [node for node in node_ if node.Et[0] > G_aves[fd] * node.Et[2]]  # not be needed?
                if len(pruned_node_) > 10:
                    agg_recursion(rroot, root, node_=pruned_node_, Q=list(combinations(pruned_node_,r=2)), nrng=1, fagg=1)
                    if rroot and fd and root.derH:  # der+ only (check not empty root.derH)
                        rroot.fback_ += [root.derH]
                        feedback(rroot)  # update root.root..


def rng_recursion(rroot, root, node_, prelinks, Et, nrng=1):  # rng++/G_, der+/link_ in sub+, -> rim_H

    for _G, G in prelinks:
        if _G in G.compared_: continue
        if isinstance(G,CG):
            cy, cx = box2center(G.box); _cy, _cx = box2center(_G.box)
            dy = cy-_cy; dx = cx-_cx
        else:
            (_y1,_x1),(_y2,_x2) = box2center(_G.node_[0].box), box2center(_G.node_[1].box)
            (y1,x1),(y2,x2)     = box2center(G.node_[0].box), box2center(G.node_[1].box)
            dy = (y1+y2)/2 -(_y1+_y2)/2; dx = (x1+x2)/2 -(_x1+_x2)/2
        dist = np.hypot(dy, dx)  # distance between node centers or link centers
        # der+'rng+ is directional
        if nrng > 1:  # pair eval:
            M = (G.Et[0]+_G.Et[0])/2; R = (G.Et[2]+_G.Et[2])/2  # local
            for link in _G.rim:
                if comp_angle((dy,dx), link.angle)[0] > ave_mA:
                    for med_link in link.link_:  # if link is hyperlink
                        M += med_link.derH.H[-1].Et[0]
                        R += med_link.derH.H[-1].Et[2]
        if (nrng==1 and dist<=ave_dist) or (nrng>1 and M / (dist/ave_dist) > ave * R):
            G.compared_ += [_G]; _G.compared_ += [G]
            comp_G([_G,G, dist, [dy,dx]], Et)

    if Et[0] > ave_Gm * Et[2]:  # rng+ eval per arg cluster because comp is bilateral, 2nd test per new pair
        nrng,_ = rng_recursion(rroot, root, node_, list(combinations(node_,r=2)), Et, nrng+1)

    return nrng, Et


def comp_G(link, iEt, nrng=None):  # add dderH to link and link to the rims of comparands, which may be Gs or links

    dderH = CH()  # new layer of link.dderH
    if isinstance(link, Clink):
        # der+
        _G,G = link._node,link.node; fd=1
    else:  # rng+
        _G,G, distance, [dy,dx] = link; fd=0
        link = Clink(node_=[_G, G], distance=distance, angle=[dy, dx])
    if isinstance(G,CG):
        fG = 1; rn = _G.n/G.n
    else:
        fG = 0; rn = min(_G.node_[0].n,_G.node_[1].n) / min(G.node_[0].n,G.node_[1].n)
    if not fd:  # form new Clink
        if fG:  # / P
            Et, relt, md_ = comp_latuple(_G.latuple, G.latuple, rn, fagg=1)
            dderH.n = 1; dderH.Et = Et; dderH.relt=relt
            dderH.H = [CH(nest=0, Et=copy(Et), relt=copy(relt), H=md_, n=1)]
            # / PP, if >1 Ps:
            if _G.iderH and G.iderH: _G.iderH.comp_(G.iderH, dderH, rn, fagg=1, flat=0)
            _L,L = len(_G.node_),len(G.node_); _S,S = _G.S, G.S/rn; _A,A = _G.A,G.A
        else: _L,L = _G.distance,G.distance; _S,S = len(_G.rim),len(G.rim); _A,A = _G.angle,G.angle
        comp_ext(_L,L,_S,S,_A,A, distance, dderH)
    # / G, if >1 PPs | Gs:
    if _G.extH and G.extH: _G.extH.comp_(G.extH, dderH, rn, fagg=1, flat=1)  # always true in der+
    if _G.derH and G.derH: _G.derH.comp_(G.derH, dderH, rn, fagg=1, flat=0)  # append and sum new dderH to base dderH

    link.derH.append_(dderH, flat=0)  # append nested, higher-res lower-der summation in sub-G extH
    iEt[:] = np.add(iEt,dderH.Et)  # init eval rng+ and form_graph_t by total m|d?
    for i in 0,1:
        Val, Rdn = dderH.Et[i::2]
        if Val > G_aves[i] * Rdn:
            if not fd:  # else old links
                for node in _G,G:
                    if fG:
                        for _link in node.rim:  # +med_links for der+
                            if comp_angle(link.angle, _link.angle)[0] > ave_mA:
                                _link.rim += [link]  # med_links angle should also match
                        node.rim += [link]
                    else:  # node is Clink, all mediating links in link.rim should have matching angle:
                        if comp_angle(node.rim[-1].angle, _link.angle)[0] > ave_mA:
                            node.rim += [link]
                fd = 1  # to not add the same link twice
            _G.Et[i] += Val; G.Et[i] += Val
            _G.Et[2+i] += Rdn; G.Et[2+i] += Rdn  # per fork link in both Gs
            # if select fork links: iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dderH.Et[i::2])]

def comp_ext(_L,L,_S,S,_A,A, dist, dderH):  # compare non-derivatives: dist, node_' L,S,A:

    prox = ave_dist - dist  # proximity = inverted distance (position difference), no prior accum to n

    dL = _L - L;      mL = min(_L,L) - ave_mL  # direct match
    dS = _S/_L - S/L; mS = min(_S,S) - ave_mL  # sparsity is accumulated over L
    mA, dA = comp_angle(_A,A)  # angle is not normalized

    M = prox + mL + mS + mA
    D = dist + abs(dL) + abs(dS) + abs(dA)  # signed dA?
    mrdn = M > D; drdn = D<= M

    mdec = prox / max_dist + mL/ max(L,_L) + mS/ max(S,_S) if S or _S else 1 + mA  # Amax = 1
    ddec = dist / max_dist + mL/ (L+_L) + dS/ (S+_S) if S or _S else 1 + dA

    dderH.append_(CH(Et=[M,D,mrdn,drdn], relt=[mdec,ddec], H=[prox,dist, mL,dL, mS,dS, mA,dA], n=2/3), flat=0)  # 2/3 of 6-param unit

# draft:
def convolve_graph(iG_):  # node connectivity = sum surround link vals, incr.mediated: Graph Convolution of Correlations
    '''
    Aggregate direct * indirect connectivity per node from indirect links via associated nodes, in multiple cycles.
    Each cycle adds contributions of previous cycles to linked-nodes connectivity, propagated through the network.
    Math: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/node_connect.png
    Link-mediated in all iterations, no layers of kernels, update node Et.
    Add der+: cross-comp root.link_, update link Et.
    Not sure: sum and compare kernel params: reduced-resolution rng+, lateral link-mediated vs. vertical in agg_kernels?
    '''
    _G_ = iG_; fd = isinstance(iG_[0],Clink)  # ave = G_aves[fd]  # ilink_ if fd, very rare?
    while True:
        # eval accumulated G connectivity with node-mediated range extension
        G_ = []  # next connectivity expansion, use link_ instead? more selective by DV,Lent
        mediation = 1  # n intermediated nodes, increasing decay
        for G in _G_:
            uprim = []  # >ave updates of direct links
            for i in 0,1:
                val,rdn = G.Et[i::2]  # rng+ for both segment forks
                if not val: continue  # G has no new links
                ave = G_aves[i]
                for link in G.rim:  # if fd: use hyper-Links between links: link.link_, ~ G.rim, then add in dgraph.link_?
                    if len(link.derH.H)<=len(G.extH.H): continue  # old links, else derH+= in comp_G, same for link Gs?
                    # > ave derGs in new fd rim:
                    lval,lrdn = link.Et[i::2]  # step=2, graph-specific vals accumulated from surrounding nodes, or use link.node_.Et instead?
                    decay =  (link.relt[i] / (link.derH.n * 6)) ** mediation  # normalized decay at current mediation
                    for _G in link.node_:
                        if _G is not G: break
                    # add G.derH.comp_(_G.extH.H[-1]): temporary derH summed from surrounding G.derHs,
                    # for lower-res rng+, if no direct rng+ and high _G.V? or replace mediated rng+?
                    _val,_rdn = _G.Et[i::2] # current-loop vals and their difference from last-loop vals, before updating:
                    # += adjacent Vs * dec: lateral modulation of node value, vs. current vertical modulation by Kernel comp:
                    V = (val+_val) * decay; dv = V-lval
                    R = (rdn+_rdn)  # rdn doesn't decay
                    link.Et[i::2] = [V,R]  # last-loop vals for next loop | segment_node_, dect is not updated
                    if dv > ave * R:  # extend mediation if last-update val, may be negative
                        G.Et[i::2] = [V+v for V,v in zip(G.Et[i::2],[V,R])]  # last layer link vals
                        if link not in uprim: uprim += [link]
                    if V > ave * R:  # updated even if terminated
                        G.Et[i::2] = [V+v for V,v in zip(G.Et[i::2], [dv,R])]  # use absolute R?
            if uprim:
                G_ += [G]  # list of nodes to check in next loop
        if G_:
            mediation += 1  # n intermediated nodes in next loop
            _G_ = G_  # exclude weakly incremented Gs from next connectivity expansion loop
        else:
            break


def form_graph_t(root, node_, Et, nrng, fagg=0):  # form Gm_,Gd_ from same-root nodes

    node_t = []
    for fd in 0,1:
        if Et[fd] > ave * Et[2+fd]:  # eVal > ave * eRdn
            graph_ = segment_graph(root, root.link_ if fd else node_, fd, nrng, fagg)
            if fd:  # der+ after rng++ term by high ds
                for graph in graph_:
                    if graph.link_ and graph.Et[1] > G_aves[1] * graph.Et[3]:  # Et is summed from all links
                        agg_recursion(root, graph, graph.node_, graph.link_, nrng, fagg=0)  # graph.node_ is not node_t yet
                    elif graph.derH:
                        root.fback_ += [graph.derH]
                        feedback(root)  # update root.root.. per sub+
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]
    if any(node_t):
        root.node_[:] = node_t  # else keep root.node_
        return node_t


# not updated:
def segment_graph(root, Q, fd, nrng, fagg):  # eval rim links with summed surround vals for density-based clustering

    # graph+= [node] if >ave (surround connectivity * relative value of link to any internal node)
    igraph_ = []; ave = G_aves[fd]

    for e in Q:  # init per node or link
        uprim = [link for link in e.rim if len(link.derH.H)==len(e.extH.H)]
        if uprim:  # skip nodes without add new added rim
            grapht = [[e],[],[*e.Et], uprim]  # link_ = updated rim
            e.root = grapht  # for merging
            igraph_ += [grapht]
        else: e.root = None
    _graph_ = copy(igraph_)

    while True:
        graph_ = []
        for grapht in _graph_:  # extend grapht Rim with +ve in-root links
            if grapht not in igraph_: continue  # skip merged graphs
            G_, Link_, Et, Rim = grapht
            inVal, inRdn = 0,0  # new in-graph +ve
            new_Rim = []
            for link in Rim:  # unique links
                if link.node_[0] in G_:  # one of the nodes is already clustered
                    _G, G = link.node_
                else:
                    G, _G = link.node_
                if _G in G_: continue
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
        graph.latuple = [P+p for P,p in zip(graph.latuple[:-1],graph.latuple[:-1])] + [[A+a for A,a in zip(graph.latuple[-1],graph.latuple[-1])]]
        if G.iderH:  # empty in single-P PP|Gs
            graph.iderH.add_(G.iderH)
        if G.derH:  # empty in single-PP Gs
            graph.derH.add_(G.derH)
        if fd: G.Et = [0,0,0,0]  # reset in last form_graph_t fork, Gs are shared in both forks
        graph.n += G.n  # non-derH accumulation?
    extH = CH()
    for link in Link_:  # sum last layer of unique current-layer links
        if len(extH.H)==len(link.derH.H): extH.H[-1].add_(link.derH.H[-1], irdnt=link.derH.H[-1].Et[2:4])  # sum last layer
        else:                              extH.append_(link.derH.H[-1],flat=0)  # pack last layer
        graph.S += link.distance
        np.add(graph.A,link.angle)
        link.root = graph
    graph.derH.append_(extH, flat=0)  # graph derH = node derHs + [summed Link_ derHs]

    if fd:  # assign alt graphs from d graph, after both linked m and d graphs are formed
        for link in graph.link_:
            mgraph = link.roott[0]
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