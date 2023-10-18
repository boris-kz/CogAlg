import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from collections import deque, defaultdict
from .classes import Cgraph, CderG, CPP
from .filters import ave_L, ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd, ave_dI, ave_G, ave_M, ave_Ma
from .slice_edge import slice_edge, comp_angle
from .comp_slice import comp_P_, comp_ptuple, sum_ptuple, sum_dertuple, comp_derH, matchF
from .agg_recursion import comp_G_, sum_link_tree_, sum2graph, feedback

'''
Implement sparse param tree in aggH: new graphs represent only high m|d params + their root params.
Compare layers in parallel by flipping H, comp forks independently. Flip HH,HHH.. in deeper processing? 

1st: cluster root params within param set, by match per param between previously cross-compared nodes
2nd: cluster root nodes per above-average param cluster, formed in 1st step. 

specifically, x-param comp-> p-clustering of root AggH( SubH( DerH( Vars, after all nodes cross comp, @xp rng 
param xcomp in derivation order, nested for lower level of hierarchical xcomp?  

compress aggH: param cluster nesting reflects root param set nesting (which is a superset of param clusters). 
exclusively deeper param cluster has empty (unpacked) higher param nesting levels.

Mixed-forks: connectivity cluster must be contiguous, not uniform, as distant nodes don't need to be similar?
Nodes are connected by m|d of different param sets in links, potentially clustered in pPs for compression.

Then combine graph with alt_graphs?
'''

def vectorize_root(blob, verbose):  # vectorization pipeline is 3 composition levels of cross-comp,clustering:

    edge = slice_edge(blob, verbose)  # lateral kernel cross-comp -> P clustering
    comp_P_(edge)  # vertical, lateral-overlap P cross-comp -> PP clustering
    # PP cross-comp -> discontinuous graph clustering:
    for fd in 0,1:
        node_ = edge.node_t[fd]  # always PP_t
        if edge.valt[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.rdnt[fd]:
            G_= []
            for PP in node_:  # convert CPPs to Cgraphs:
                derH,valt,rdnt = PP.derH,PP.valt,PP.rdnt  # init aggH is empty:
                for dderH in derH: dderH += [[0,0]]  # add maxt
                G_ += [Cgraph( ptuple=PP.ptuple, derH=[derH,valt,rdnt,[0,0]], valHt=[[valt[0]],[valt[1]]], rdnHt=[[rdnt[0]],[rdnt[1]]],
                               L=PP.ptuple[-1], box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box))]
            node_ = G_
            edge.valHt[0][0] = edge.valt[0]; edge.rdnHt[0][0] = edge.rdnt[0]  # copy
            agg_recursion(None, edge, node_, fd=0)  # edge.node_t = graph_t, micro and macro recursive


def agg_recursion(rroot, root, G_, fd):  # compositional agg+|sub+ recursion in root graph, clustering G_

    cluster_params(parHv = [root.aggH,sum(root.valHt[fd]),sum(root.rdnHt[fd]),sum(root.maxHt[fd])], fd=fd)
    # compress aggH-> pP_,V,R,M: select G V,R,M?
    Val,Rdn = comp_G_(G_, fd)  # rng|der cross-comp all Gs, form link_H[-1] per G, sum in Val,Rdn
    if Val > ave * Rdn:  # else no clustering
        root.valHt[fd]+=[0]; root.rdnHt[fd] += [1]  # sum in form_graph_t feedback, +root.maxHt[fd]?
        # pass Val,Rdn?
        GG_t = form_graph_t(root, G_)  # eval sub+ and feedback per graph
        # agg+ xcomp-> form_graph_t loop sub)agg+, vs. comp_slice:
        # sub+ loop-> eval-> xcomp
        for GG_ in GG_t:  # comp_G_ eval: ave_m * len*rng - fixed cost, root update in form_t:
            if root.valHt[0][-1] * (len(GG_)-1)*root.rng > G_aves[fd] * root.rdnHt[0][-1]:
                agg_recursion(rroot, root, GG_, fd=0)  # 1st xcomp in GG_

        G_[:] = GG_t

def cluster_params(parHv, fd):  # last v: value tuple valt,rdnt,maxt

    parH, rVal, rRdn, rMax = parHv  # compressed valt,rdnt,maxt per aggH replace initial summed G vals
    part_P_ = []  # pPs: nested clusters of >ave param tuples, as below:
    part_ = []  # [[subH, sub_part_P_t], Val,Rdn,Max]
    Val, Rdn, Max = 0, 0, 0
    parH = copy(parH)
    while parH:  # aggHv | subHv | derHv (ptupletv_), top-down
        subt = parH.pop()   # Hv | extt | ptupletv

        if isinstance(subt[0][0],list):  # not extt
            if isinstance(subt[0][0][0],list):
                subH, valt, rdnt, maxt = subt  # subt is Hv
                val, rdn, max = valt[fd],rdnt[fd],maxt[fd]
                if val > ave:  # recursive eval,unpack
                    Val+=val; Rdn+=rdn; Max+=max  # summed with sub-values:
                    sub_part_P_t = cluster_params(subt, fd)
                    part_ += [[subH, sub_part_P_t]]
                else:
                    if Val:  # empty sub_pP_ terminates root pP
                        part_P_ += [[part_,Val,Rdn,Max]]; rVal+=Val; rRdn+=Rdn; rMax+=Max  # root params
                        part_= [], Val,Rdn,Max = 0,0,0  # pP params
                        # reset
            else: cluster_ptuplet(subt, [part_P_,rVal,rRdn,rMax], [part_,Val,Rdn,Max], v=1)  # subt is ptupletv
        else:     cluster_ptuplet(subt, [part_P_,rVal,rRdn,rMax], [part_,Val,Rdn,Max], v=0)  # subt is extt
    if part_:
        part_P_ += [[part_,Val,Rdn,Max]]; rVal+=Val; rRdn+=Rdn; rMax+=Max

    return [part_P_,rVal,rRdn,rMax]  # root values


def cluster_ptuplet(ptuplet, part_P_v, part_v, v):  # ext or ptuple, params=vals

    part_P_,rVal,rRdn,rMax = part_P_v  # root params
    part_,Val,Rdn,Max = part_v  # pP params
    if v: ptuplet, valt,rdnt,maxt = ptuplet  # valt,rdnt,maxt

    valP_t = [[cluster_vals(ptuple) for ptuple in ptuplet if sum(ptuple) > ave]]
    if valP_t:
        part_ += [valP_t]  # params=vals, no sum-> Val,Rdn,Max?
    else:
        if Val:  # empty valP_ terminates root pP
            part_P_ += [[part_, Val, Rdn, Max]]
            part_P_v[1:] = rVal+Val, rRdn+Rdn, rMax+Max  # root values
        part_v[:] = [],0,0,0  # reset

def cluster_vals(ptuple):
    parP_ = []
    parP = [ptuple[0]] if ptuple[0] > ave else []  # init, need to use param type ave instead

    for par in ptuple[1:]:
        if par > ave: parP += [par]
        else:
            if parP: parP_ += [parP]  # terminate parP
            parP = []

    if parP: parP_ += [parP]  # terminate last parP
    return parP_  # may be empty

# very tentative, need to add pruning nodes from non-max roots and pruning weak graphs:
def segment_node_(Gt_, fd):  # form proto-graph_

    extend_val = 0; graph_ = []

    for G,Val,Rdn in Gt_:  # Gt from sum_link_tree_ + direct _links, _nodes, Nodes:
        graph_ += [[G,Val,Rdn, G.link_H[-1], [link._G if link.G is G else link.G for link in G.link_H[-1]], []]]

    while True:  # add incrementally mediated _links and _nodes to each node
        extend_val = 0
        for (node, Val, Rdn, _links, _nodes, Nodes) in graph_:
            links, nodes = [],[]  # per current-layer node
            for _node in _nodes:
                for link in _node.link_H[-1]:  # mediated links
                    __node = link._G if link.G is _node else link.G
                    if __node not in Nodes:  # not in lower-layer links
                        nodes += [__node]
                        links += [link]  # to adjust link.val in suppress_overlap
                        extend_val += link.valt[fd]  # * (__node Val - ave *__node Rdn)?

            Nodes+=nodes  # current link mediation order
        if extend_val < ave: break
'''
- add temporary +ve combined-value links to all connected nodes per top-layer node, to represent overlapping clusters 
- via these links, top nodes send roots to bottom nodes, which then sort them to select single root/exemplar, 
  and delete themselves from the links of other top-layer roots. 
- prune weak top nodes, the rest represent non-overlapping clusters of their links.
'''

def form_graph_t(root, G_):  # form mgraphs and dgraphs of same-root nodes

    graph_t = []
    for G in G_: G.root = [None,None]  # replace with mcG_|dcG_ in segment_node_, replace with Cgraphs in sum2graph
    for fd in 0,1:
        Gt_ = sum_link_tree_(G_, fd)  # sum surround link values @ incr rng,decay
        graph_t += [segment_node_(root, Gt_, fd)]  # add alt_graphs?

    # eval sub+, not in segment_node_: full roott must be replaced per node within recursion
    for fd, graph_ in enumerate(graph_t): # breadth-first for in-layer-only roots
        root.valHt[fd]+=[0]; root.rdnHt[fd]+=[1]  # remove if stays 0?

        for graph in graph_:  # external to agg+ vs internal in comp_slice sub+
            node_ = graph.node_t  # init flat
            if sum(graph.valHt[fd]) * (len(node_)-1)*root.rng > G_aves[fd] * sum(graph.rdnHt[fd]):  # eval fd comp_G_ in sub+
                agg_recursion(root, graph, node_, fd)  # replace node_ with node_t, recursive
            else:  # feedback after graph sub+:
                root.fback_t[fd] += [[graph.aggH, graph.valHt, graph.rdnHt, graph.maxHt]]
                root.valHt[fd][-1] += graph.valHt[fd][-1]  # last layer, or all new layers via feedback?
                root.rdnHt[fd][-1] += graph.rdnHt[fd][-1]  # merge forks in root fork
                root.maxHt[fd][-1] += graph.maxHt[fd][-1]
            i = sum(graph.valHt[0]) > sum(graph.valHt[1])
            root.rdnHt[i][-1] += 1  # add fork rdn to last layer, representing all layers after feedback

        if root.fback_t and root.fback_t[fd]:  # recursive feedback after all G_ sub+
            feedback(root, fd)  # update root.root.. aggH, valHt,rdnHt

    return graph_t  # root.node_t'node_ -> node_t: incr nested with each agg+?