import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from collections import deque, defaultdict
from .classes import Cgraph, CderG, CPP
from .filters import ave_L, ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd, ave_dI, ave_G, ave_M, ave_Ma
from .slice_edge import slice_edge, comp_angle
from .comp_slice import comp_P_, comp_ptuple, sum_ptuple, sum_dertuple, comp_derH, matchF

'''
Implement sparse param tree in aggH: new graphs represent only high m|d params + their root params.
Compare layers in parallel: flip H on the side, comp even if other forks miss? 
Incrementally parallel: flip HH,HHH.. in deeper processing? 

Eval forming graph type for >ave matching forks, of same types? 
Or combine single-fork graphs into single variable-nfork graph?
'''

# potentially relevant, not revised:
# form link layers to back-propagate overlap of root graphs
def form_mediation_layers(layer, layers, fder):  # layers are initialized with same nodes and incrementally mediated links

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

def merge_root_tree(Root_t, root_t):  # not-empty fork layer is root_t, each fork may be empty list:

    for Root_, root_ in zip(Root_t, root_t):
        for Root, root in zip(Root_, root_):
            if root.root_t:  # not-empty root fork layer
                if Root.root_t: merge_root_tree(Root.root_t, root.root_t)
                else: Root.root_t[:] = root.root_t
        Root_[:] = list( set(Root_+root_))  # merge root_, may be empty


def select_init_(Gt_, fd):  # local max selection for sparse graph init, if positive link

    init_, non_max_ = [],[]  # pick max in direct links, no recursively mediated links max: discontinuous?

    for node, val in Gt_:
        if node in non_max_: continue  # can't init graph
        if val<=0:  # no +ve links
            if sum(node.val_Ht[fd]) > ave * sum(node.rdn_Ht[fd]):
                init_+= [[node, 0]]  # single-node proto-graph
            continue
        fmax = 1
        for link in node.link_H[-1]:
            _node = link.G if link._G is node else link._G
            if val > Gt_[_node.it[fd]][1]:
                non_max_ += [_node]  # skip as next node
            else:
                fmax = 0; break  # break is not necessary?
        if fmax:
            init_ += [[node,val]]
    return init_

def segment_node_(init_, Gt_, fd, root_t_):  # root_t_ for fuzzy graphs if partial param sets: sub-forks?

    graph_ = []  # initialize graphs with local maxes, eval their links to add other nodes:

    for inode, ival in init_:
        iroot_t = root_t_[inode.it[fd]]  # same order as Gt_, assign node roots to new graphs, init with max
        graph = [[inode], ival, [iroot_t]]
        inode.root_t[fd] += [graph]
        _nodet_ = [[inode, ival, [iroot_t]]]  # current perimeter of the graph, init = graph

        while _nodet_:  # search links outwards recursively to form overlapping graphs:
            nodet_ = []
            for _node, _val, _root_t in _nodet_:
                for link in _node.link_H[-1]:
                    node = link.G if link._G is _node else link._G
                    if node in graph[0]: continue
                    val = Gt_[node.it[fd]][1]
                    root_t = root_t_[node.it[fd]]
                    if val * (link.valt[fd] / link.maxt[fd]) > ave:  # node val * link decay
                        if node.root_t[fd]:
                            merge(graph, node.root_t[fd])  # or use dict link_map?
                        else:
                            graph[0] += [node]
                            graph[1] += val
                            graph[2] += [root_t]
                            nodet_ += [node, val, root_t]  # new perimeter
            _nodet_ = nodet_
        graph_ += [graph]
    return graph_

def prune_graph_(graph_, fd):  # compute graph overlap to prune weak graphs, not nodes: rdn doesn't change the structure
                               # prune rootless nodes?
    for graph in graph_:
        for node in graph[0]:  # root rank = graph/node rdn:
            roots = sorted(node.root_t[fd], key=lambda root: root[1], reverse=True)  # sort by net val, if partial param sub-forks
            # or grey-scale rdn = root_val / sum_higher_root_vals?
            for rdn, graph in enumerate(roots):
                graph[1] -= ave * rdn  # rdn to >val overlapping graphs per node, also >val forks, alt sparse param sets?
                # nodes are shared by multiple max-initialized graphs, pruning here still allows for some overlap
        pruned_graph_ = []
        for graph in graph_:
            if graph[1] > G_aves[fd]:  # rdn-adjusted Val for local sparsity, doesn't affect G val?
                pruned_graph_ += [graph]
            else:
                for node in graph[0]:
                    node.root_t[fd].remove(graph)
    return pruned_graph_


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

    comp_G_(G_, fd)  # rng|der cross-comp all Gs, nD array? form link_H per G
    root.valHt[fd] += [0]; root.rdnHt[fd] += [1]  # sum in form_graph_t feedback

    GG_t = form_graph_t(root, G_)  # eval sub+ and feedback per graph
    # agg+ xcomp-> form_graph_t loop sub)agg+, vs. comp_slice:
    # sub+ loop-> eval-> xcomp
    for GG_ in GG_t:  # comp_G_ eval: ave_m * len*rng - fixed cost, root update in form_t:
        if root.valHt[0][-1] * (len(GG_)-1)*root.rng > G_aves[fd] * root.rdnHt[0][-1]:
            agg_recursion(rroot, root, GG_, fd=0)  # 1st xcomp in GG_

    G_[:] = GG_t

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
