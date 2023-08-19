import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from .classes import Cgraph, CderG
from .filters import aves, ave, ave_nsubt, ave_sub, ave_agg, G_aves, med_decay, ave_distance, ave_Gm, ave_Gd
from .comp_slice import comp_angle, comp_ptuple, sum_ptuple, sum_derH, comp_derH, comp_aangle
# from .sub_recursion import feedback
'''
Blob edges may be represented by higher-composition patterns, etc., if top param-layer match,
in combination with spliced lower-composition patterns, etc, if only lower param-layers match.
This may form closed edge patterns around flat core blobs, which defines stable objects. 
Graphs are formed from blobs that match over <max distance. 
-
They are also assigned adjacent alt-fork graphs, which borrow predictive value from the graph. 
Primary value is match, so difference patterns don't have independent value. 
They borrow value from proximate or average match patterns, to the extent that they cancel their projected match. 
But alt match patterns borrow already borrowed value, which may be too tenuous to track, we can use the average instead.
-
Graph is abbreviated to G below:
Agg+ cross-comps top Gs and forms higher-order Gs, adding up-forking levels to their node graphs.
Sub+ re-compares nodes within Gs, adding intermediate Gs, down-forking levels to root Gs, and up-forking levels to node Gs.
-
Generic graph is a dual tree with common root: down-forking input node Gs and up-forking output graph Gs. 
This resembles a neuron, which has dendritic tree as input and axonal tree as output. 
But we have recursively structured param sets packed in each level of these trees, which don't exist in neurons.
Diagram: 
https://github.com/boris-kz/CogAlg/blob/76327f74240305545ce213a6c26d30e89e226b47/frame_2D_alg/Illustrations/generic%20graph.drawio.png
-
Clustering criterion is G.M|D, summed across >ave vars if selective comp (<ave vars are not compared, so they don't add costs).
Fork selection should be per var or co-derived der layer or agg level. 
There are concepts that include same matching vars: size, density, color, stability, etc, but in different combinations.
Weak value vars are combined into higher var, so derivation fork can be selected on different levels of param composition.
'''

def agg_recursion(root, node_):  # compositional recursion in root.PP_

    for i in 0,1: root.rdn_Ht[i][0] += 1  # estimate, no node.rdnt[fder] += 1?

    node_tt = [[[],[]],[[],[]]]  # fill with 4 clustering forks
    pri_root_T_ = []
    for node in node_:
        pri_root_T_ += [node.root_T]  # save root_T for new graphs, different per node
        node.root_T = [[[],[]],[[],[]]]  # replace node.root_T, then append [root,val] in each fork
        for i in 0,1:
            node.val_Ht[i]+=[0]; node.rdn_Ht[i]+=[1]  # new val,rdn layer, accum in comp_G_

    for fder in 0,1:  # comp forks, each adds a layer of links
        if fder and len(node_[0].link_H) < 2:  # 1st call, no der+ yet
            continue
        comp_G_(node_, pri_G_=None, f1Q=1, fder=fder)  # cross-comp all Gs in (rng,der), nD array? form link_H per G

        for fd in 0,1:  # clustering forks, each adds graph_: new node_ in node_tt:
            if sum(root.val_Ht[fder]) > G_aves[fder] * sum(root.rdn_Ht[fder]):
                # cluster link_H[-1]:
                graph_ = form_graph_(node_, pri_root_T_, fder, fd)
                sub_recursion_eval(root, graph_)  # sub+, eval last layer?
                if sum(root.val_Ht[fder]) > G_aves[fder] * sum(root.rdn_Ht[fder]):  # updated in sub+
                    agg_recursion(root, node_)  # agg+, replace root.node_ with new graphs, if any
                node_tt[fder][fd] = graph_
            elif root.root_T:  # if deeper agg+
                node_tt[fder][fd] = node_
                feedback(root, fd)  # update root.root..H, breadth-first

    node_[:] = node_tt  # replace local element of root.node_T


def form_graph_(node_, pri_root_T_, fder, fd):  # form fuzzy graphs of nodes per fder,fd, within root

    ave = G_aves[fder]  # val = (Val / len(links)) * med_decay * i?
    _Rdn_, Rdn_ = [1 for node in node_], [1 for node in node_]  # accum >vals of linked nodes to reduce graph overlap
    dRdn = ave+1

    while dRdn > ave:  # iteratively propagate lateral Rdn (soft non-max per node) through mediating links, to define ultimate maxes?

        # or compute local_max(val + sum(surround_vals*mediation_decay)), with incremental mediation rng?
        # similar to Gaussian blurring, but only to select graph pivots?
        
        for node, Rdn in zip(node_, Rdn_):
            if (sum(node.val_Ht[fder]) * Rdn/(Rdn+ave))  - ave * sum(node.rdn_Ht[fder]):  # potential graph init
                nodet_ = []
                for link in node.link_H[-1]:
                    _node = link.G1 if link.G0 is node else link.G0
                    nodet_ += [[_node, link.valt[fder], Rdn_[node_.index(_node)]]]
                # [[_node,link_val,_rdn_val]]:
                nodet_ = sorted(nodet_+[[node,1,1]], key=lambda _nodet:  # lower-val node rdn += higher val*medDecay, because links maybe pruned:
                         sum(_nodet[0].val_Ht[fder]) - ave * sum(_nodet[0].rdn_Ht[fder]) * _nodet[1]/(_nodet[1]+ave) * _nodet[2]/(_nodet[2]+ave),
                         reverse=True)
                __node, __val, __Rdn = nodet_[0]  # local max  # adjust overlap val by mediating link vals:
                Rdn = sum(__node.val_Ht[fder]) *__val/(__val+ave) *__Rdn/(__Rdn+ave)  # effective overlap *= connection strength?
                # add to weaker nodes Rdn_[i], accum for next:
                for _node, _val, _Rdn in nodet_[1:]:
                    Rdn_[node_.index(_node)] += Rdn  # stronger-node-init graphs would overlap current-node-init graph
                    Rdn += sum(_node.val_Ht[fder]) * _val/(_val+ave) * _Rdn/(_Rdn+ave)  # adjust overlap by medDecay

        dRdn = sum([abs(Rdn-_Rdn) for Rdn,_Rdn in zip(Rdn_,_Rdn_)])
        _Rdn_ = Rdn_; Rdn_ = [1 for node in node_]
    '''            
    no explicit layers, recursion to re-sort and re-sum with previously adjusted directly-linked _node Rdns? 
    or form longer-range more mediated soft maxes, Rdn += stronger (med_Val * med_decay * med_depth)?
    '''
    # not revised, we may need to check mediating links to sparsify maxes:
    while layers:
        new_layers = []
        for layer in layers:
            __node, __val, __Rdn = layer[0]  # 1st index: local max
            for (node, val, Rdn) in layer[1:]:  # each element in layer is [node, link's val, rdn]
                 if (sum(node.val_Ht[fder]) * Rdn/ (Rdn+ave))  - ave * sum(node.rdn_Ht[fder]):
                    layer = []
                    for link in node.link_H[-1]:  # mediated nodes
                        _node = link.G1 if link.G0 is node else link.G0
                        _Rdn = Rdn_[node_.index(_node)]
                        layer += [[_node, link.valt[fder], _Rdn]]
                    # sort, lower-val node rdn += higher (val * rel_med_val: links may be pruned)s:
                    layer = sorted(layer+[[node,1, 1]], key=lambda _nodet:  # add adjust by _Rdn here:
                            sum(_nodet[0].val_Ht[fder]) - ave * sum(_nodet[0].rdn_Ht[fder]) * (_nodet[1]/(_nodet[1]+ave)) * (_nodet[2]/(_nodet[2]+ave)) ,
                            reverse=True)
                    Rdn = sum(__node.val_Ht[fder]) *__val/(__val+ave) *__Rdn/(__Rdn+ave)
                    for _node, _val, _Rdn in layer:
                        Rdn_[node_.index(_node)] += Rdn
                        Rdn += sum(_node.val_Ht[fder]) * _val/(_val+ave) * _Rdn/(_Rdn+ave)
                    new_layers += [layer]
        layers = new_layers  # for next recursion

    segment_network(nodet_, pri_root_T_, fder, fd)

# not revised:
def segment_network(nodet_, pri_root_T_, fder, fd):

    # select local max nodes to initialize graphs, then prune links to add other nodes:
    # link val,rnd are combined with G0+G1 valH, rdnH for evaluation
    graph_ = []
    for nodet in nodet_:  # loop top-down, accumulate rdn per link from higher layers?

        max_nodes = []
        for i, (node, links, nodes, Nodes) in enumerate(layer):
            # get local max and max node based on rdn and val
            adjusted_val = [sum(node.val_Ht[fder]) * (1/sum(node.rdn_Ht[fder]))  for node in nodes]
            # adjust node.val_Ht[fder] by 1/rdn? So the higher rdn, the lower the graph's val?
            max_node = nodes[np.argmax(adjusted_val)]
            if max_node not in max_nodes: max_nodes += [max_node]

        for node in max_nodes:
            for (_node, links, nodes, Nodes) in layer:  # get max nodes' linksn nodes and etc
                if _node is node: break

            # below is not updated
            if not node.root_T[fder][fd]:  # not forming graph in prior loops
                graph = [[node], [pri_root_T_[node_.index(node)]], Val]
                node.root_T[fder][fd] = graph
                graph_ += [graph]
                # search links recursively
                nodes = [node]; links_ = [links]
                while links_:
                    node = nodes.pop()  # unpack node per links
                    links = links_.pop()
                    for link in links:
                        _node = link.G1 if link.G0 is node else link.G0
                        if _node not in graph[0] and link.Valt[fder] > ave * link.Rdnt[fder]:
                            for (__node, _Val, _links, __nodes, _Nodes) in layer:  # find _node in a same layer
                                if __node is _node: break
                            # pack _node into graph
                            graph[0] += [_node]
                            graph[1] += [pri_root_T_[node_.index(_node)]]
                            graph[2] += _Val
                            _node.root_T[fder][fd] = graph
                            links_ += [_links]
                            nodes += [_node]
    # evaluate links by val > ave*rdn, for fuzzy segmentation, no change in link vals
    return graph_

# not updated, ~segment_network:
def form_graph_direct(node_, pri_root_T_, fder, fd):  # form fuzzy graphs of nodes per fder,fd, initially fully overlapping

    graph_ = []
    nodet_ = []
    for node, pri_root_T in zip(node_, pri_root_T_):
        nodet = [node, pri_root_T, 0]
        nodet_ += [nodet]
        graph = [[nodet], [pri_root_T], 0]  # init graph per node?
        node.root_T[fder][fd] = [[graph, 0]]  # init with 1st root, node-specific val
    # use popping?:
    for nodet, graph in zip(nodet_, graph_):
        graph_ += [init_graph(nodet, nodet_, graph, fder, fd)]  # recursive depth-first GQ_+=[_nodet]
    # prune by rdn:
    regraph_ = graph_reval_(graph_, fder,fd)  # init reval_ to start
    if regraph_:
        graph_[:] = sum2graph_(regraph_, fder, fd)  # sum proto-graph node_ params in graph

    # add_alt_graph_(graph_t)  # overlap+contour, cluster by common lender (cis graph), combined comp?
    return graph_

def init_graph(nodet, nodet_, graph, fder, fd):  # recursive depth-first GQ_+=[_nodet]

    node, pri_root_T, val = nodet

    for link in node.link_H[-(1+fder)]:
        if link.valt[fd] > G_aves[fd]:
            # link is in node GQ
            _nodet = link.G1 if link.G0 is nodet else link.G0
            if _nodet in nodet_:  # not removed in prior loop
                _node,_pri_root_T,_val = _nodet
                graph[0] += [_node]
                graph[2] += _val
                if _pri_root_T not in graph[1]:
                    graph[1] += [_pri_root_T]
                nodet_.remove(_nodet)
                init_graph(_nodet, nodet_, graph, fder, fd)
    return graph
'''
initially links only: plain attention forming links to all nodes, then segmentation by backprop of 
(sum of root vals / max_root_val) - 1, summed from all nodes in graph? 
while dMatch per cluster > ave: 
- sum cluster match,
- sum in-cluster match per node root: containing cluster, positives only,
- sort node roots by in_cluster_match,-> index = cluster_rdn for node,
- prune root if in_cluster_match < ave * cluster_redundancy, 
- prune weak clusters, remove corresponding roots
'''

# draft:
def graph_reval_(graph_, fder,fd):

    regraph_ = []
    reval, overlap  = 0, 0
    for graph in graph_:
        for node in graph[0]:  # sort node root_(local) by root val (ascending)
            node.root_T[fder][fd] = sorted(node.root_T[fder][fd], key=lambda root:root[0][2], reverse=False)  # 2: links val
    while graph_:
        graph = graph_.pop()  # pre_graph or cluster
        node_, root_, val = graph
        remove_ = []  # sum rdn (overlap) per node root_T, remove if val < ave * rdn:
        for node in node_:
            rdn = 1 + len(node.root_T)
            if node.valt[fd] < G_aves[fd] * rdn:
            # replace with node.root_T[i][1] < G_aves[fd] * i:
            # val graph links per node, graph is node.root_T[i], i=rdn: index in fork root_ sorted by val
                remove_ += [node]       # to remove node from cluster
                for i, grapht in enumerate(node.root_T[fder][fd]):  # each root is [graph, Val]
                    if graph is grapht[0]:
                        node.root_T[fder][fd].pop(i)  # remove root cluster from node
                        break
        while remove_:
            remove_node = remove_.pop()
            node_.remove(remove_node)
            graph[2] -= remove_node.valt[fd]
            reval += remove_node.valt[fd]
        for node in node_:
            sum_root_val = sum([root[1] for root in node.root_T[fder][fd]])
            max_root_val = max([root[1] for root in node.root_T[fder][fd]])
            overlap += sum_root_val/ max_root_val
        regraph_ += [graph]

    if reval > ave and overlap > 0.5:  # 0.5 can be a new ave here
        regraph_ = graph_reval_(regraph_, fder,fd)

    return regraph_

# draft:
def graph_reval(graph, fd):  # exclusive graph segmentation by reval,prune nodes and links

    aveG = G_aves[fd]
    reval = 0  # reval proto-graph nodes by all positive in-graph links:

    for node in graph[0]:  # compute reval: link_Val reinforcement by linked nodes Val:
        link_val = 0
        _link_val = node.valt[fd]
        for derG in node.link_H[-1]:
            if derG.valt[fd] > [ave_Gm, ave_Gd][fd]:
                val = derG.valt[fd]  # of top aggH only
                _node = derG.G1 if derG.G0 is node else derG.G0
                # consider in-graph _nodes only, add to graph if link + _node > ave_comb_val?
                link_val += val + (_node.valt[fd]-val) * med_decay
        reval += _link_val - link_val  # _node.link_tH val was updated in previous round
    rreval = 0
    if reval > aveG:  # prune graph
        regraph, Val = [], 0  # reformed proto-graph
        for node in graph[0]:
            val = node.valt[fd]
            if val > G_aves[fd] and node in graph:
            # else no regraph += node
                for derG in node.link_H[-1][fd]:
                    _node = derG.G1 if derG.G0 is node else derG.G0  # add med_link_ val to link val:
                    val += derG.valt[fd] + (_node.valt[fd]-derG.valt[fd]) * med_decay
                Val += val
                regraph += [node]
        # recursion:
        if rreval > aveG and Val > aveG:
            regraph, reval = graph_reval([regraph,Val], fd)  # not sure about Val
            rreval += reval

    else: regraph, Val = graph

    return [regraph,Val], rreval
'''
In rng+, graph may be extended with out-linked nodes, merged with their graphs and re-segmented to fit fixed processors?
Clusters of different forks / param sets may overlap, else no use of redundant inclusion?
No centroid clustering, but cluster may have core subset.
'''

def comp_G_(G_, pri_G_=None, f1Q=1, fder=0):  # cross-comp in G_ if f1Q, else comp between G_ and pri_G_, if comp_node_?

    for G in G_:  # node_
        if fder:  # follow prior link_ layer
            _G_ = []
            for link in G.link_H[-2]:
                if link.valt[1] > ave_Gd:
                    _G_ += [link.G1 if G is link.G0 else link.G0]
        else:    _G_ = G_ if f1Q else pri_G_  # loop all Gs in rng+
        for _G in _G_:
            if _G in G.compared_:  # was compared in prior rng
                continue
            dy = _G.box[0]-G.box[0]; dx = _G.box[1]-G.box[1]
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity
            if distance < ave_distance * ((sum(_G.val_Ht[fder]) + sum(G.val_Ht[fder])) / (2*sum(G_aves))):
                G.compared_ += [_G]; _G.compared_ += [G]
                # same comp for cis and alt components:
                for _cG, cG in ((_G, G), (_G.alt_Graph, G.alt_Graph)):
                    if _cG and cG:  # alt Gs maybe empty
                        # form new layer of links:
                        comp_G(_cG, cG, distance, [dy,dx])
    '''
    combine cis,alt in aggH: alt represents node isolation?
    comp alts,val,rdn? cluster per var set if recurring across root: type eval if root M|D?
    '''

def comp_G(_G, G, distance, A):

    Mval,Dval = 0,0
    Mrdn,Drdn = 1,1
    # / P:
    mtuple, dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1)
    mval, dval = sum(mtuple), sum(dtuple)
    mrdn = dval>mval; drdn = dval<=mval
    derLay0 = [[mtuple,dtuple], [mval,dval], [mrdn,drdn]]
    Mval += mval; Dval += dval; Mrdn += mrdn; Drdn += drdn
    # / PP:
    dderH, valt, rdnt = comp_derH(_G.derH[0], G.derH[0], rn=1)
    mval,dval = valt
    Mval += dval; Dval += mval; Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval

    derH = [[derLay0]+dderH, [Mval,Dval],[Mrdn,Drdn]]  # appendleft derLay0 from comp_ptuple
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval], [Mrdn,Drdn])
    SubH = [der_ext, derH]  # two init layers of SubH, higher layers added by comp_aggH:
    # / G:
    if _G.aggH and G.aggH:  # empty in base fork
        subH, valt, rdnt = comp_aggH(_G.aggH, G.aggH, rn=1)
        SubH += subH  # append higher subLayers: list of der_ext | derH s
        mval,dval =valt
        Mval += valt[0]; Dval += valt[1]; Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval

    derG = CderG(G0=_G, G1=G, subH=SubH, valt=[Mval,Dval], rdnt=[Mrdn,Drdn], S=distance, A=A)
    if valt[0] > ave_Gm or valt[1] > ave_Gd:
        _G.link_H[-1] += [derG]; G.link_H[-1] += [derG]  # bilateral add links
        _G.val_Ht[0][-1] += Mval; _G.val_Ht[1][-1] += Dval; _G.rdn_Ht[0][-1] += Mrdn; _G.rdn_Ht[1][-1] += Drdn
        G.val_Ht[0][-1] += Mval;   G.val_Ht[1][-1] += Dval;  G.rdn_Ht[0][-1] += Mrdn;  G.rdn_Ht[1][-1] += Drdn

def sum2graph_(graph_, fder, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    Graph_ = []
    for graph in graph_:  # seq Gs
        if graph[2] < G_aves[fd]:  # form graph if val>min only
            continue
        pri_roots = graph[1]  # not sure how to pack this pri_roots into Graph since we have only fd here, so Graph.root_T[fd] = pri_roots?
        Graph = Cgraph(L=len(graph[0]))  # n nodes
        Link_ = []
        for G in graph[0]:
            sum_box(Graph.box, G.box)
            sum_ptuple(Graph.ptuple, G.ptuple)
            sum_derH(Graph.derH, G.derH, base_rdn=1)  # base_rdn?
            sum_aggH([Graph.aggH,Graph.valt,Graph.rdnt], [G.aggH,G.valt,G.rdnt], base_rdn=1)
            link_ = G.link_H[-1]
            Link_[:] = list(set(Link_ + link_))
            subH=[]; valt=[0,0]; rdnt=[1,1]
            for derG in link_:
                if derG.valt[fd] > G_aves[fd]:
                    sum_subH([subH,valt,rdnt], [derG.subH,derG.valt,derG.rdnt], base_rdn=1)
                    sum_box(G.box, derG.G0.box if derG.G1 is G else derG.G1.box)
            G.aggH += [[subH,valt,rdnt]]
            for i in 0,1:
                G.valt[i] += valt[i]; G.rdnt[i] += rdnt[i]
            Graph.node_T += [G]  # then converted to node_tt by feedback
        subH=[]; valt=[0,0]; rdnt=[1,1]
        for derG in Link_:  # sum unique links:
            sum_subH([subH,valt,rdnt], [derG.subH, derG.valt, derG.rdnt], base_rdn=1)
            Graph.A[0] += derG.A[0]; Graph.A[1] += derG.A[1]
            Graph.S += derG.S
        Graph.aggH += [[subH, valt, rdnt]]  # new aggLev
        for i in 0,1:
            Graph.valt[i] += valt[i]; Graph.rdnt[i] += rdnt[i]
        Graph_ += [Graph]

    return Graph_

''' if n roots: 
sum_aggH(Graph.uH[0][fd].aggH,root.aggH) or sum_G(Graph.uH[0][fd],root)? init if empty
sum_H(Graph.uH[1:], root.uH)  # root of Graph, init if empty
'''

def sum_box(Box, box):
    Y, X, Y0, Yn, X0, Xn = Box;  y, x, y0, yn, x0, xn = box
    Box[:] = [Y + y, X + x, min(X0, x0), max(Xn, xn), min(Y0, y0), max(Yn, yn)]

# old:
def add_alt_graph_(graph_t):  # mgraph_, dgraph_
    '''
    Select high abs altVal overlapping graphs, to compute value borrowed from cis graph.
    This altVal is a deviation from ave borrow, which is already included in ave
    '''
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            for node in graph.aggHs.H[-1].node_:
                for derG in node.link_.Q:  # contour if link.aggHs.val < aveGm: link outside the graph
                    for G in [derG.node0, derG.node1]:  # both overlap: in-graph nodes, and contour: not in-graph nodes
                        alt_graph = G.roott[1-fd]
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, CQ):  # not proto-graph or removed
                            graph.alt_graph_ += [alt_graph]
                            alt_graph.alt_graph_ += [graph]
                            # bilateral assign
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            if graph.alt_graph_:
                graph.alt_aggHs = CQ()  # players if fsub? der+: aggHs[-1] += player, rng+: players[-1] = player?
                for alt_graph in graph.alt_graph_:
                    sum_pH(graph.alt_aggHs, alt_graph.aggHs)  # accum alt_graph_ params
                    graph.alt_rdn += len(set(graph.aggHs.H[-1].node_).intersection(alt_graph.aggHs.H[-1].node_))  # overlap

# draft:
def sub_recursion_eval(root, graph_):  # eval per fork, same as in comp_slice, still flat aggH, add valt to return?

    termt = [1,1]
    for graph in graph_:
        node_ = copy(graph.node_T); sub_G_t = []
        fr = 0
        for fd in 0,1:
            # not sure int or/and ext:
            if graph.val_Ht[fd][-1] > G_aves[fd] * graph.rdn_Ht[fd][-1] and len(graph.node_T) > ave_nsubt[fd]:
                graph.rdnt[fd] += 1  # estimate, no node.rdnt[fd] += 1?
                termt[fd] = 0; fr = 1
                sub_G_t += [sub_recursion(graph, node_, fd)]  # comp_der|rng in graph -> parLayer, sub_Gs
            else:
                sub_G_t += [node_]
                if isinstance(root, Cgraph):
                    # merge_externals?
                    root.fback_ += [[graph.aggH, graph.valt, graph.rdnt]]  # fback_t vs. flat?
        if fr:
            graph.node_T = sub_G_t  # still node_ here
    for fd in 0,1:
        if termt[fd] and root.fback_:  # no lower layers in any graph
           feedback(root, fd)


def sub_recursion(graph, fder, fd):  # rng+: extend G_ per graph, der+: replace G_ with derG_, valt=[0,0]?

    comp_G_(graph.node_, pri_G_=None, f1Q=1, fd=fd)  # cross-comp all Gs in rng
    sub_G_t = form_graph_(graph, fder, fd)  # cluster sub_graphs via link_H

    for i, sub_G_ in enumerate(sub_G_t):
        if sub_G_:  # and graph.rdn > ave_sub * graph.rdn:  # from sum2graph, not last-layer valt,rdnt?
            for sub_G in sub_G_: sub_G.root = graph
            sub_recursion_eval(graph, sub_G_)

    return sub_G_t  # for 4 nested forks in replaced P_?

# not revised:
def feedback(root, fd):  # append new der layers to root

    Fback = deepcopy(root.fback_.pop())  # init with 1st fback: [aggH,valt,rdnt]
    while root.fback_:
        aggH, valt, rdnt = root.fback_.pop()
        sum_aggH(Fback, [aggH, valt, rdnt] , base_rdn=0)
    for i in 0, 1:
        sum_aggH([root.aggH[i], root.valt[i],root.rdnt[i]], [Fback[i],Fback[0][i],Fback[0][i]], base_rdn=0)

    if isinstance(root.root, Cgraph):  # not blob
        root = root.root
        root.fback_ += [Fback]
        if len(root.fback_) == len(root.node_[fd]):  # all nodes term, fed back to root.fback_
            feedback(root, fd)  # aggH/ rng layer in sum2PP, deeper rng layers are appended by feedback


def comp_ext(_ext, ext, Valt, Rdnt):  # comp ds:

    (_L,_S,_A),  (L,S,A) = _ext, ext
    dL=_L-L;      mL=ave-abs(dL)
    dS=_S/_L-S/L; mS=ave-abs(dS)
    if isinstance(A,list):
        mA, dA = comp_angle(_A,A)
    else:
        dA= _A-A; mA= ave-abs(dA)

    M = mL+mS+mA; D = dL+dS+dA
    Valt[0] += M; Valt[1] += D
    Rdnt[0] += D>M; Rdnt[1] += D<=M

    return [[mL,mS,mA], [dL,dS,dA]]


def sum_ext(Extt, extt):

    if isinstance(Extt[0], list):  # ext fd tuple
        for fd, (Ext,ext) in enumerate(zip(Extt,extt)):
            for i,(Par,par) in enumerate(zip(Ext,ext)):
                Ext[i] = Par+par
    else:  # single ext
        for i in 0,1: Extt[i]+=extt[i]  # sum L,S
        for j in 0,1: Extt[2][j]+=extt[2][j]  # sum dy,dx in angle

'''
derH: [[tuplet, valt, rdnt]]: default input from PP, for both rng+ and der+, sum min len?
subH: [[derH_t, valt, rdnt]]: m,d derH, m,d ext added by agg+ as 1st tuplet
aggH: [[subH_t, valt, rdnt]]: composition layers, ext per G
'''

def comp_subH(_subH, subH, rn):
    DerH = []
    Mval, Dval, Mrdn, Drdn = 0,0,1,1

    for _lay, lay in zip_longest(_subH, subH, fillvalue=[]):  # compare common lower layer|sublayer derHs
        if _lay and lay:  # also if lower-layers match: Mval > ave * Mrdn?
            if _lay[0] and isinstance(_lay[0][0],list):  # _lay[0][0] is derHt

                dderH, valt, rdnt = comp_derH(_lay[0], lay[0], rn)
                DerH += [[dderH, valt, rdnt]]  # for flat derH
                mval,dval = valt
                Mval += mval; Dval += dval; Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + dval <= mval
            else:  # _lay[0][0] is L, comp dext:
                DerH += comp_ext(_lay[1],lay[1], [Mval,Dval], [Mrdn,Drdn])

    return DerH, [Mval,Dval], [Mrdn,Drdn]  # new layer, 1/2 combined derH

def comp_aggH(_aggH, aggH, rn):  # no separate ext processing?
      SubH = []
      Mval, Dval, Mrdn, Drdn = 0,0,1,1

      for _lev, lev in zip_longest(_aggH, aggH, fillvalue=[]):  # compare common lower layer|sublayer derHs
          if _lev and lev:  # also if lower-layers match: Mval > ave * Mrdn?
              # compare dsubH only:
              dsubH, valt, rdnt = comp_subH(_lev[0], lev[0], rn)
              SubH += [[dsubH, valt, rdnt]]
              mval,dval = valt
              Mval += mval; Dval += dval; Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+mval<=dval

      return SubH, [Mval,Dval], [Mrdn,Drdn]


def sum_subH(T, t, base_rdn):

    SubH, Valt, Rdnt = T
    subH, valt, rdnt = t
    for i in 0, 1:
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i]
    if SubH:
        for Layer, layer in zip_longest(SubH,subH, fillvalue=[]):
            if layer:
                if Layer:
                    if layer[0] and isinstance(Layer[0][0], list):  # _lay[0][0] is derH
                        sum_derH(Layer, layer, base_rdn)
                    else: sum_ext(Layer, layer)
                else:
                    SubH += [deepcopy(layer)]  # _lay[0][0] is mL
    else:
        SubH[:] = deepcopy(subH)

def sum_aggH(T, t, base_rdn):

    AggH, Valt, Rdnt = T
    aggH, valt, rdnt = t
    for i in 0, 1:
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i]
    if aggH:
        if AggH:
            for Layer, layer in zip_longest(AggH,aggH, fillvalue=[]):
                if layer:
                    if Layer:
                        sum_subH(Layer, layer, base_rdn)
                    else:
                        AggH += [deepcopy(layer)]
        else:
            AggH[:] = deepcopy(aggH)