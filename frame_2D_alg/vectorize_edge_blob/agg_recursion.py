import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from .classes import Cgraph, CderG, CPP
from .filters import aves, ave, med_decay, ave_distance, G_aves, ave_Gm, ave_Gd
from .slice_edge import slice_edge, comp_angle
from .comp_slice import comp_P_, comp_ptuple, sum_ptuple, sum_derH, comp_derH
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

def vectorize_root(blob, verbose=False):  # vectorization pipeline: three composition levels of cross-comp->clustering:

    # lateral kernel cross-comp -> P clustering
    edge = slice_edge(blob, verbose=False)
    # vertical P cross-comp -> PP clustering
    comp_P_(edge)  # compare vertically-adjacent, laterally-overlapping Ps to form derPs, top-down
    # PP cross-comp -> graph clustering
    for fd in 0,1:
        node_ = edge.node_t[fd]  # only comp rng+ for edge
        if edge.val_Ht[0] * np.sqrt(len(node_)-1) if node_ else 0 > G_aves[0] * edge.rdn_Ht[0]:  # still valt and rdnt here
            G_ = []  # CPP -> Cgraph
            for PP in node_:
                derH, valt, rdnt = PP.derH, PP.valt, PP.rdnt  # init aggH is empty:
                G_ += [Cgraph(ptuple=PP.ptuple, derH=[derH, valt, rdnt], val_Ht=[[valt[0]], [valt[1]]], rdn_Ht=[[rdnt[0]], [rdnt[1]]],
                                 L=PP.ptuple[-1], box=[(PP.box[0] + PP.box[1]) / 2, (PP.box[2] + PP.box[3]) / 2] + list(PP.box))]
            edge.node_t[fd] = G_ # replace PPs with Gs
            while True:
                agg_recursion(edge, node_)  # node_[:] = new node_tt in sub+ feedback?
                if np.sum(edge.val_Ht[0]) * np.sqrt(len(node_)-1) if node_ else 0 <= G_aves[0] * np.sum(edge.rdn_Ht[0]):
                    break

def agg_recursion(root, node_):  # compositional recursion in root graph

    pri_root_tt_ = [node.root_tt for node in node_]  # merge node roots for new graphs, same for both comp forks
    node_tt = [[],[]]  # fill with comp fork if selected, else stays empty

    for fder in 0,1:
        # eval per comp fork, n comp graphs ~-> n matches, match rate decreases with distance:
        if np.sum(root.val_Ht[fder]) * np.sqrt(len(node_)-1) if node_ else 0 > G_aves[fder] * np.sum(root.rdn_Ht[fder]):
            if fder and len(node_[0].link_H) < 2:  # 1st call, no der+ yet
                continue
            root.val_Ht[fder] += [0]; root.rdn_Ht[fder] += [1]  # estimate, no node.rdnt[fder] += 1?
            for node in node_:
                node.root_tt[fder] = [[],[]]  # to replace node roots per comp fork
                node.val_Ht[fder] += [0]; node.rdn_Ht[fder] += [1]  # new layer, accum in comp_G_:
            if not fder:  # add layer of links if rng+
                for node in node_: node.link_H += [[]]

            comp_G_(node_, pri_G_=None, f1Q=1, fder=fder)  # cross-comp all Gs in (rng,der), nD array? form link_H per G
            node_tt[fder] = [[],[]]  # fill with clustering forks by default
            for fd in 0,1:
                # cluster link_H[-1] -> graph_,= node_ in node_tt, default, agg+ eval per graph:
                graph_ = form_graph_(root, node_, fder, fd, pri_root_tt_)
                node_tt[fder][fd] = graph_
    node_[:] = node_tt  # replace local element with new graph forks, possibly empty


def comp_G_(G_, pri_G_=None, f1Q=1, fder=0):  # cross-comp in G_ if f1Q, else comp between G_ and pri_G_, if comp_node_?

    for G in G_:  # node_
        if fder:  # follow prior link_ layer
            _G_ = []
            for link in G.link_H[-2]:
                if link.valt[1] > ave_Gd:
                    _G_ += [link.G1 if G is link.G0 else link.G0]
        else:  _G_ = G_ if f1Q else pri_G_  # loop all Gs in rng+
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

    Mval,Dval,Maxv = 0,0,0
    Mrdn,Drdn = 1,1

    # / P:
    mtuple, dtuple, Mtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1)
    mval, dval, maxv = sum(mtuple), sum(dtuple), sum(Mtuple)
    mrdn = dval>mval; drdn = dval<=mval
    derLay0 = [[mtuple,dtuple], [mval,dval,maxv], [mrdn,drdn]]
    Mval+=mval; Dval+=dval; Maxv+=maxv; Mrdn += mrdn; Drdn += drdn
    # / PP:
    dderH, valt, rdnt = comp_derH(_G.derH[0], G.derH[0], rn=1)
    mval, dval, maxv = valt
    Mval+=dval; Dval+= mval; Maxv+=maxv; Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval

    derH = [[derLay0]+dderH, [Mval,Dval,Maxv], [Mrdn,Drdn]]  # appendleft derLay0 from comp_ptuple
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval,Maxv], [Mrdn,Drdn])
    SubH = [der_ext, derH]  # two init layers of SubH, higher layers added by comp_aggH:
    # / G:
    if _G.aggH and G.aggH:  # empty in base fork
        subH, valt, rdnt = comp_aggH(_G.aggH, G.aggH, rn=1)
        SubH += subH  # append higher subLayers: list of der_ext | derH s
        mval, dval, maxv = valt
        Mval+=mval; Dval+=dval; Maxv+=maxv; Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval

    derG = CderG(G0=_G, G1=G, subH=SubH, valt=[Mval,Dval,Maxv], rdnt=[Mrdn,Drdn], S=distance, A=A)
    if valt[0] > ave_Gm or valt[1] > ave_Gd:
        _G.link_H[-1] += [derG]; G.link_H[-1] += [derG]  # bilateral add links
        _G.val_Ht[0][-1] += Mval; _G.val_Ht[1][-1] += Dval; _G.rdn_Ht[0][-1] += Mrdn; _G.rdn_Ht[1][-1] += Drdn
        G.val_Ht[0][-1] += Mval; G.val_Ht[1][-1] += Dval; G.rdn_Ht[0][-1] += Mrdn; G.rdn_Ht[1][-1] += Drdn


def form_graph_(root, Node_, fder, fd, pri_root_tt_):  # root function to form fuzzy graphs of nodes per fder,fd

    ave = G_aves[fder]
    max_ = select_max_(Node_, fder, ave)  # compute max of quasi-Gaussians: val + sum([_val * (link_val/max_val])

    full_graph_ = segment_node_(Node_, max_, fder, fd, pri_root_tt_)
    list_graph_ = prune_graph_(full_graph_, fder, fd)  # sort node roots and prune the weak

    graph_ = sum2graph_(list_graph_, fder, fd)  # convert list graphs to Cgraphs
    # tentative sub+:
    for graph in graph_:
        node_ = copy(graph.node_tt)  # still node_ here
        if sum(graph.val_Ht[fder]) * np.sqrt(len(node_) - 1) if node_ else 0 > G_aves[fder] * sum(graph.rdn_Ht[fder]):
            agg_recursion(graph, node_)  # replaces node_ formed above with new graphs in node_tt, recursive
        else:
            if not root.fback_tt: root.fback_tt = [[[],[]],[[],[]]]  # init empty
            for graph in graph_: root.fback_tt[fder][fd] += [[graph.aggH, graph.val_Ht, graph.rdn_Ht]]

    if root.fback_tt and root.fback_tt[fder][fd]:
        # sub+ is terminated in all root fork nodes, initiate feedback
        feedback(root, fder, fd)  # update root.root..aggH, breadth-first

    return graph_

def select_max_(node_, fder, ave):  # final maxes are graph-initializing nodes

    _Val_= [sum(node.val_Ht[fder]) for node in node_]
    dVal = ave+1  # adjustment of combined val per node per recursion

    while dVal > ave:  # iterative adjust Val by surround propagation, no direct increment mediation rng?
        Val_ = [0 for node in node_]
        for i, (node, Val) in enumerate(zip(node_, Val_)):

            if sum(node.val_Ht[fder]) - ave * sum(node.rdn_Ht[fder]):  # potential graph init
                for link in node.link_H[-1]:
                    _node = link.G1 if link.G0 is node else link.G0
                    # val + sum([_val * relative link val,= decay of max m|d: link.valt[2]:
                    Val_[i] += _Val_[node_.index(_node)] * (link.valt[fder] / link.valt[2]) - ave * sum(_node.rdn_Ht[fder])
                    # unilateral: simpler, parallelizable
        dVal = sum([abs(Val-_Val) for Val,_Val in zip(Val_,_Val_)])
        _Val_ = Val_
    # select local maxes of node quasi-Gaussian, not sure:
    max_, non_max_ = [],[]
    for node, Val in zip(node_, Val_):
        if Val<=0 or node in non_max_:
            continue
        fmax = 1
        for link in node.link_H[-1]:
            _node = link.G1 if link.G0 is node else link.G0
            if Val > Val_[node_.index(_node)]:
                non_max_ += [_node]  # skip in the future
            else:
                fmax = 0
                break
        if fmax: max_ += [node]
    return max_

def segment_node_(node_, max_, fder, fd, pri_root_tt_):

    graph_ = []  # initialize graphs with local maxes, then prune links to add other nodes:

    for max_node in max_:
        pri_root_tt = pri_root_tt_[node_.index(max_node)]
        graph = [[max_node], sum(max_node.val_Ht[fder]), [pri_root_tt]]
        max_node.root_tt[fder][fd] += [graph]
        _nodes = [max_node]  # current periphery of the graph
        while _nodes:  # search links outwards, recursively:
            nodes = []
            for node in _nodes:
                val = sum(node.val_Ht[fder]) - ave * sum(node.rdn_Ht[fder])
                for link in node.link_H[-1]:
                    _node = link.G1 if link.G0 is node else link.G0
                    if _node not in graph[0]:
                        _val = sum(_node.val_Ht[fder]) - ave * sum(_node.rdn_Ht[fder])
                        link_rel_val = link.valt[fder] / link.valt[1]
                        # tentative link eval to add _node in graph:
                        if (val+_val) * link_rel_val > 0:
                            graph[0] += [_node]
                            graph[1] += link_rel_val * _val
                            if pri_root_tt_:  # agg+
                                pri_root_tt = pri_root_tt_[node_.index(_node)]
                                graph[2] += [pri_root_tt]
                            _node.root_tt[fder][fd] += [graph]  # single root per fork?
                            nodes += [_node]
            _nodes = nodes
        graph_ += [graph]
    return graph_

def merge_root_tree(Root_tt, root_tt):  # not-empty fork layer is root_tt, each fork may be empty list:

    for Root_t, root_t in zip(Root_tt, root_tt):  # fder loop
        for Root_, root_ in zip(Root_t, root_t):  # fd loop
            for Root, root in zip(Root_, root_):
                if root.root_tt:  # not-empty root fork layer
                    if Root.root_tt: merge_root_tree(Root.root_tt, root.root_tt)
                    else: Root.root_tt[:] = root.root_tt
            Root_[:] = list( set(Root_+root_))  # merge root_, may be empty

def prune_graph_(graph_, fder, fd):

    for graph in graph_:
        for node in graph[0]:
            roots = sorted(node.root_tt[fder][fd], key=lambda root: root[1], reverse=True)
            for rdn, graph in enumerate(roots):
                graph[1] -= ave*rdn  # rdn to stronger overlapping graphs, + stronger forks, select param sets?
                # node is shared by multiple max-initialized graphs, pruning here still allows for some overlap
    pruned_graph_ = []
    for graph in graph_:
        if graph[1] > G_aves[fder]:  # eval adjusted Val to reduce graph overlap, for local sparsity?
            pruned_graph_ += [graph]
        else:
            for node in graph[0]:
                node.root_tt[fder][fd].remove(graph)

    return pruned_graph_


def sum2graph_(graph_, fder, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    Graph_ = []
    for graph in graph_:  # seq graphs
        Root_tt = copy(graph[2][0])  # merge pri_root_tt_ into Root_tt:
        [merge_root_tree(Root_tt, root_tt) for root_tt in graph[2][1:]]
        Graph = Cgraph(root_tt=Root_tt, L=len(graph[0]))  # n nodes
        Link_ = []
        for G in graph[0]:
            sum_box(Graph.box, G.box)
            sum_ptuple(Graph.ptuple, G.ptuple)
            sum_derH(Graph.derH, G.derH, base_rdn=1)  # base_rdn?
            sum_aggH([Graph.aggH,Graph.val_Ht,Graph.rdn_Ht], [G.aggH,G.val_Ht,G.rdn_Ht], base_rdn=1)
            link_ = G.link_H[-1]
            Link_[:] = list(set(Link_ + link_))
            subH=[]; valt=[0,0]; rdnt=[1,1]  # no max in G.valt?
            for derG in link_:
                if derG.valt[fd] > G_aves[fd]:
                    sum_subH([subH,valt,rdnt], [derG.subH,derG.valt,derG.rdnt], base_rdn=1)
                    sum_box(G.box, derG.G0.box if derG.G1 is G else derG.G1.box)
            G.aggH += [[subH,valt,rdnt]]
            for i in 0,1:
                G.val_Ht[i][-1] += valt[i]; G.rdn_Ht[i][-1] += rdnt[i]
            G.root_tt[fder][fd][G.root_tt[fder][fd].index(graph)] = Graph  # replace list graph in root_tt
            Graph.node_tt += [G]  # then converted to node_tt by feedback
        subH=[]; valt=[0,0]; rdnt=[1,1]
        for derG in Link_:  # sum unique links:
            sum_subH([subH,valt,rdnt], [derG.subH, derG.valt, derG.rdnt], base_rdn=1)
            Graph.A[0] += derG.A[0]; Graph.A[1] += derG.A[1]
            Graph.S += derG.S
        Graph.aggH += [[subH, valt, rdnt]]  # new aggLev
        for i in 0,1:
            # replace with Val: graph[1]?
            Graph.val_Ht[i][-1] += valt[i]; Graph.rdn_Ht[i][-1] += rdnt[i]
        Graph_ += [Graph]

    return Graph_

def sum_box(Box, box):
    Y,X,Y0,Yn,X0,Xn = Box; y,x,y0,yn,x0,xn = box
    Box[:] = [Y+y, X+x, min(X0,x0), max(Xn,xn), min(Y0,y0), max(Yn,yn)]


def comp_ext(_ext, ext, Valt, Rdnt):  # comp ds:

    (_L,_S,_A),  (L,S,A) = _ext, ext

    dL=_L-L;      mL=ave-abs(dL)
    dS=_S/_L-S/L; mS=ave-abs(dS)
    if isinstance(A,list):
        mA, dA = comp_angle(_A,A); max_mA = .5  # = ave_dangle
    else:
        dA= _A-A; mA= ave-abs(dA); max_mA = max(A,_A)
    M = mL+mS+mA
    D = dL+dS+dA
    maxv = max(L,_L)+max(S,_S)+max_mA

    Valt[0] += M; Valt[1] += D; Valt[2] += maxv
    Rdnt[0] += D>M; Rdnt[1] += D<=M

    return [[mL,mS,mA], [dL,dS,dA]]  # no Mtuple?


def sum_ext(Extt, extt):

    if isinstance(Extt[0], list):
        for Ext,ext in zip(Extt,extt):  # ext: m,d tuple
            for i,(Par,par) in enumerate(zip(Ext,ext)):
                Ext[i] = Par+par
    else:  # single ext
        for i in 0,1: Extt[i]+=extt[i]  # sum L,S
        for j in 0,1: Extt[2][j]+=extt[2][j]  # sum dy,dx in angle

'''
derH: [[tuplet, valt, rdnt]]: default input from PP, for both rng+ and der+, sum min len?
subH: [[derH_t, valt, rdnt]]: m,d derH, m,d ext added by agg+ as 1st tuplet
aggH: [[subH_t, valt, rdnt]]: composition levels, ext per G, 
'''

def comp_subH(_subH, subH, rn):
    DerH = []
    Mval, Dval, Maxv, Mrdn, Drdn = 0,0,0,1,1

    for _lay, lay in zip_longest(_subH, subH, fillvalue=[]):  # compare common lower layer|sublayer derHs
        if _lay and lay:  # also if lower-layers match: Mval > ave * Mrdn?
            if _lay[0] and isinstance(_lay[0][0],list):  # _lay[0][0] is derHt

                dderH, valt, rdnt = comp_derH(_lay[0], lay[0], rn)
                DerH += [[dderH, valt, rdnt]]  # for flat derH
                mval,dval,maxv = valt
                Mval += mval; Dval += dval; Maxv += maxv; Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + dval <= mval
            else:  # _lay[0][0] is L, comp dext:
                DerH += [comp_ext(_lay[1],lay[1], [Mval,Dval,Maxv], [Mrdn,Drdn])]  # pack as ptuple

    return DerH, [Mval,Dval,Maxv], [Mrdn,Drdn]  # new layer, 1/2 combined derH

def comp_aggH(_aggH, aggH, rn):  # no separate ext
    SubH = []
    Mval, Dval, Maxv, Mrdn, Drdn = 0,0,0,1,1

    for _lev, lev in zip_longest(_aggH, aggH, fillvalue=[]):  # compare common lower layer|sublayer derHs
        if _lev and lev:  # also if lower-layers match: Mval > ave * Mrdn?
            # compare dsubH only:
            dsubH, valt, rdnt = comp_subH(_lev[0], lev[0], rn)
            SubH += dsubH  # flatten to keep subH
            mval,dval,maxv = valt
            Mval += mval; Dval += dval; Maxv+=maxv; Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+mval<=dval

    return SubH, [Mval,Dval,Maxv], [Mrdn,Drdn]

def sum_subH(T, t, base_rdn):

    SubH, Valt, Rdnt = T
    subH, valt, rdnt = t

    for i in 0,1:  # link maxv is not summed in G.valt
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

    AggH, Val_Ht, Rdn_Ht = T
    aggH, val_Ht, rdn_Ht = t

    for i in 0,1:  # link maxv is not summed in G.valt
        for j, (Val, val) in enumerate(zip_longest(Val_Ht[i], val_Ht[i], fillvalue=None)):
            if val != None:
                if Val != None: Val_Ht[i][j] += val
                else:           Val_Ht[i] += [val]
        for j, (Rdn, rdn) in enumerate(zip_longest(Rdn_Ht[i], rdn_Ht[i], fillvalue=None)):
            if rdn != None:
                if Rdn != None: Rdn_Ht[i][j] += rdn
                else:           Rdn_Ht[i] += [rdn]
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


def feedback(root, fder, fd):  # called from form_graph_, append new der layers to root

    Fback = deepcopy(root.fback_tt[fder][fd][0])  # init with 1st [aggH,val_Ht,rdn_Ht]

    for aggH, val_Ht, rdn_Ht in root.fback_tt[fder][fd][1:]:
        sum_aggH(Fback, [aggH, val_Ht, rdn_Ht], base_rdn=0)
    sum_aggH([root.aggH, root.val_Ht, root.rdn_Ht], Fback, base_rdn=0)  # both fder forks sum into a same root?

    if isinstance(root, Cgraph):  # root has root_tt: is not CEdge
        rroot_ = root.root_tt[fder][fd]  # same fork, higher level
        for rroot in rroot_:  # fder rroot_ s are empty in top Gs
            rroot.fback_tt[fder][fd] += [Fback]
            if len(rroot.fback_tt[fder][fd]) == len(rroot.node_tt[fder][fd]):
                # all nodes terminated and fed back to rroot
                feedback(rroot, fder, fd)
                # sum2graph adds aggH per rng, feedback adds deeper sub+ layers