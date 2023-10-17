import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from collections import deque, defaultdict
from .classes import Cgraph, CderG, CPP
from .filters import ave_L, ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd, ave_dI, ave_G, ave_M, ave_Ma
from .slice_edge import slice_edge, comp_angle
from .comp_slice import comp_P_, comp_ptuple, sum_ptuple, sum_dertuple, comp_derH, matchF
from .agg_recursion import comp_G_, sum_link_tree_, segment_node_, sum2graph, feedback

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

    cluster_params(parH=root.aggH, rVal=0,rRdn=0,rMax=0, fd=fd, G=root)
    # aggH-> pP_v: [part_P_,V,R,M] for compressed comp

    Val,Rdn = comp_G_(G_, fd)  # rng|der cross-comp all Gs, form link_H[-1] per G, sum in Val,Rdn
    if Val > ave*Rdn > ave:  # else no clustering?
        root.valHt[fd]+=[0]; root.rdnHt[fd] += [1]  # sum in form_graph_t feedback, +root.maxHt[fd]?

        GG_t = form_graph_t(root, G_)  # eval sub+ and feedback per graph
        # agg+ xcomp-> form_graph_t loop sub)agg+, vs. comp_slice:
        # sub+ loop-> eval-> xcomp
        for GG_ in GG_t:  # comp_G_ eval: ave_m * len*rng - fixed cost, root update in form_t:
            if root.valHt[0][-1] * (len(GG_)-1)*root.rng > G_aves[fd] * root.rdnHt[0][-1]:
                agg_recursion(rroot, root, GG_, fd=0)  # 1st xcomp in GG_

        G_[:] = GG_t


def cluster_params(parH, rVal,rRdn,rMax, fd, G=None):  # G for parH=aggH

    part_P_ = []  # pPs: nested clusters of >ave param tuples, as below:
    part_ = []  # [[subH, sub_part_P_t], Val,Rdn,Max]
    Val, Rdn, Max = 0, 0, 0
    parH = copy(parH)
    i=1
    while parH:  # aggH | subH | derH (then subH is ptuplet), top-down
        subH = parH.pop()
        if G:  # parH is aggH
            sub_cluster(subH, G.valHt[fd][-i],G.rdnHt[fd][-i],G.maxHt[fd][-i], part_P_,[part_,Val,Rdn,Max], rVal,rRdn,rMax, fd)
            i+=1
        elif isinstance(subH[0][0],list) and isinstance(subH[0][0][0],list):  # subH is derH, not extt
            subH, valt, rdnt, maxt = subH  # initial derH is a tuple
            sub_cluster(subH, valt[fd],rdnt[fd],maxt[fd], part_P_,[part_,Val,Rdn,Max], rVal,rRdn,rMax, fd)
        else:
            valP_t = [[cluster_vals(ptuple) for ptuple in subH if sum(ptuple)>ave]]  # extt in subH or ptuplet in derH
            if valP_t:
                part_ += [valP_t]  # params=vals, no sum-> Val,Rdn,Max?
            else:
                if Val:  # empty valP_ terminates root pP
                    part_P_ += [[part_,Val,Rdn,Max]]; rVal+=Val; rRdn+=Rdn; rMax+=Max  # root values
                part_=[]; Val,Rdn,Max = 0,0,0  # reset
    if part_:
        part_P_ += [[part_,Val,Rdn,Max]]; rVal+=Val; rRdn+=Rdn; rMax+=Max

    return [part_P_,rVal,rRdn,rMax]  # root values


def sub_cluster(subH,val,rdn,max, part_P_, part_t, rVal,rRdn,rMax, fd):
    part_, Val, Rdn, Max = part_t

    if val > ave:  # recursive eval,unpack
        Val+=val; Rdn+=rdn; Max+=max  # summed with sub-values:
        sub_part_P_t = cluster_params(subH, Val,Rdn,Max, fd)
        part_ += [[subH, sub_part_P_t]]
    else:
        if Val:  # empty sub_pP_ terminates root pP
            part_P_ += [[part_,Val,Rdn,Max]]; rVal+=Val; rRdn+=Rdn; rMax+=Max  # root values
            part_t[:] = [[],0,0,0]  # reset

def cluster_vals(ptuple):  # ext or ptuple, params=vals

    parP_ = []
    parP = [ptuple[0]] if ptuple[0] > ave else []  # init, need to use param type ave instead

    for par in ptuple[1:]:
        if par > ave: parP += [par]
        else:
            if parP: parP_ += [parP]  # terminate parP
            parP = []

    if parP: parP_ += [parP]  # terminate last parP
    return parP_  # may be empty


def form_mediation_layers(layer, layers, fder):  # layers are initialized with same nodes and incrementally mediated links

    # form link layers to back-propagate overlap of root graphs to segment node_, pruning non-max roots?
    out_layer = []; out_val = 0   # new layer, val

    for (node, _links, _nodes, Nodes) in layer:  # higher layers have incrementally mediated _links and _nodes
        links, nodes = [], []  # per current-layer node
        Val = 0
        for _node in _nodes:
            for link in _node.link_H[-(1+fder)]:  # mediated links
                __node = link.G1 if link.G0 is _node else link.G0
                if __node not in Nodes:  # not in lower-layer links
                    nodes += [__node]
                    links += [link]  # to adjust link.val in suppress_overlap
                    Val += link.valt[fder]
        # add fork val of link layer:
        node.val_Ht[fder] += [Val]
        out_layer += [[node, links, nodes, Nodes+nodes]]  # current link mediation order
        out_val += Val  # no permanent val per layer?

    layers += [out_layer]
    if out_val > ave:
        form_mediation_layers(out_layer, layers, fder)

'''
- sum_link_tree_: defines combined value of each node, to compute combined value of their links and to select max root
- form_mediation_layers: add temporary +ve combined-value links to all connected nodes per top-layer node 
  (these links represent overlapping clusters) 
- segment_node_: via these links, top nodes send roots to bottom nodes, which then sort them to select single root/exemplar, 
  and delete themselves from temporary links of other top-layer roots. 
- prune weak top nodes, the rest represent non-overlapping clusters of their links.
  this should be parallelizable?
'''
# replace with root backprop, as described above:
def segment_node_(root, Gt_, fd):
    link_map = defaultdict(list)
    ave = G_aves[fd]
    for G,_,_ in Gt_:
        for derG in G.link_H[-1]:
            if derG.valt[fd] > ave * derG.rdnt[fd]:  # or link val += node Val: prune +ve links to low Vals?
                link_map[G] += [derG._G]  # keys:Gs, vals: linked _G_s
                link_map[derG._G] += [G]
    graph_ = []
    # initialize proto-graphs with each node, eval links to add other nodes, skip added nodes next:
    for iG, iVal, iRdn in Gt_:
        if iVal > ave * iRdn and not iG.root[fd]:
            try: dec = iG.valHt[fd][-1] / iG.maxHt[fd][-1]
            except ZeroDivisionError: dec = 1  # add internal layers Val *= current-layer decay to init graph totals:
            tVal = iVal + sum(iG.valHt[fd]) * dec
            tRdn = iRdn + sum(iG.rdnHt[fd]) * dec
            cG_ = [iG]; iG.root[fd] = cG_  # clustered Gs
            perimeter = link_map[iG]       # recycle perimeter in breadth-first search, outward from iG:
            while perimeter:
                _G = perimeter.pop(0)
                for link in _G.link_H[-1]:
                    G = link.G if link._G is _G else link._G
                    if G in cG_ or G not in [Gt[0] for Gt in Gt_]: continue   # circular link
                    Gt = Gt_[G.it[fd]]; Val = Gt[1]; Rdn = Gt[2]
                    if Val > ave * Rdn:
                        try: decay = G.valHt[fd][-1] / G.maxHt[fd][-1]  # current link layer surround decay
                        except ZeroDivisionError: decay = 1
                        tVal += Val + sum(G.valHt[fd])*decay  # ext+ int*decay: proj match to distant nodes in higher graphs?
                        tRdn += Rdn + sum(G.rdnHt[fd])*decay
                        cG_ += [G]; G.root[fd] = cG_
                        perimeter += [G]
            if tVal > ave * tRdn:
                graph_ += [sum2graph(root, cG_, fd)]  # convert to Cgraphs

    return graph_

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