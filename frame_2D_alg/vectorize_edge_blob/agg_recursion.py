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


def agg_recursion(root, node_):  # compositional recursion in root graph

    for i in 0,1: root.rdn_Ht[i][0] += 1  # estimate, no node.rdnt[fder] += 1?
    pri_root_tt_ = [[[[],[]],[[],[]]] for _ in node_]

    for node, pri_root_tt in zip(node_, pri_root_tt_):
        merge_root_tree(pri_root_tt, node.root_tt)  # save node roots for new graphs
        node.root_tt = [[[],[]],[[],[]]]  # to replace node roots
        for i in 0,1:
            node.val_Ht[i]+=[0]; node.rdn_Ht[i]+=[1]  # new val,rdn layer, accum in comp_G_

    node_tt = [[[],[]],[[],[]]]  # fill with 4 clustering forks
    fr = 0
    for fder in 0,1:  # comp forks, each adds a layer of links
        if fder and len(node_[0].link_H) < 2:  # 1st call, no der+ yet
            continue
        comp_G_(node_, pri_G_=None, f1Q=1, fder=fder)  # cross-comp all Gs in (rng,der), nD array? form link_H per G

        for fd in 0,1:  # clustering forks, each adds graph_: new node_ in node_tt:
            if sum(root.val_Ht[fder]) > G_aves[fder] * sum(root.rdn_Ht[fder]):
                fr = 1  # cluster link_H[-1]:
                graph_ = form_graph_(node_, fder, fd, pri_root_tt_)
                sub_recursion_eval(root, graph_)  # sub+, eval last layer?
                if sum(root.val_Ht[fder]) > G_aves[fder] * sum(root.rdn_Ht[fder]):  # updated in sub+
                    agg_recursion(root, graph_)  # agg+, replace root.node_ with new graphs, if any
                node_tt[fder][fd] = graph_
            elif root.root_T:  # if deeper agg+
                node_tt[fder][fd] = node_
                feedback(root, fder, fd)  # update root.root..H, breadth-first
    if fr:
        node_[:] = node_tt  # replace local element if new graphs in any fork


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


def form_graph_(node_, fder, fd, pri_root_tt_):  # form fuzzy graphs of nodes per fder,fd, within root

    ave = G_aves[fder]
    max_ = select_max_(node_, fder, ave)  # compute max of quasi-Gaussians: val + sum([_val * (link_val/max_val])

    pre_graph_ = segment_node_(node_, max_, fder, fd, pri_root_tt_)
    graph_ = prune_graph_(pre_graph_, fder, fd)  # sort node roots and prune the weak

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

    max_, non_max_ = [],[]  # select local maxes of node quasi-Gaussian, not sure:
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

    for i, max_node in enumerate(max_):
        graph = [[max_node], sum(max_node.val_Ht[fder]), [pri_root_tt_[i]]]
        max_node.root_tt[fder][fd] += [graph]
        _nodes = [max_node]  # current periphery of the graph
        while _nodes:  # search links recursively outwards:
            nodes = []
            for node in _nodes:
                val = sum(node.val_Ht[fder]) - ave * sum(node.rdn_Ht[fder])
                for link in node.link_H[-1]:
                    _node = link.G1 if link.G0 is node else link.G0
                    if _node not in graph[0]:
                        _val = sum(_node.val_Ht[fder]) - ave * sum(_node.rdn_Ht[fder])
                        link_rel_val = link.valt[fder] / link.valt[1]
                        # tentative:
                        if (val+_val) * link_rel_val > 0:  # link eval to pack _node in graph
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
            for Root, root in zip(Root_, root_):  # not-empty fork layer is root_tt:
                if root.root_tt:
                    if Root.root_tt: merge_root_tree(Root.root_tt, root.root_tt)
                    else: Root.root_tt[:] = root.root_tt
            Root_[:] = list( set(Root_+root_))  # merge root_, may be empty

def prune_graph_(graph_, fder, fd):

    for graph in graph_:
        for node in graph[0]:
            roots = sorted(node.root_tt[fder][fd], key=lambda root: root[1], reverse=True)
            for rdn, graph in enumerate(roots):
                graph[1] -= ave*rdn  # rdn to stronger overlapping graphs, + rdn cross forks, select param sets?
                # node is shared by multiple max-initialized graphs, pruning here still allows for some overlap between them
    pruned_graph_ = []
    for graph in graph_:
        if graph[1] > G_aves[fder]:  # eval adjusted Val to reduce graph overlap, for local sparsity?
            pruned_graph_ += [graph]
        else:
            for node in graph[0]:
                node.root_tt[fder][fd].remove(graph)

    return sum2graph_(pruned_graph_, fder, fd)


def sum2graph_(graph_, fder, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    Graph_ = []
    for graph in graph_:  # seq graphs
        Graph = Cgraph(root_tt=graph[2], L=len(graph[0]))  # pri_root_tt_, n nodes
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

''' if n roots: 
sum_aggH(Graph.uH[0][fd].aggH,root.aggH) or sum_G(Graph.uH[0][fd],root)? init if empty
sum_H(Graph.uH[1:], root.uH)  # root of Graph, init if empty
'''

def sum_box(Box, box):
    Y, X, Y0, Yn, X0, Xn = Box;  y, x, y0, yn, x0, xn = box
    Box[:] = [Y + y, X + x, min(X0, x0), max(Xn, xn), min(Y0, y0), max(Yn, yn)]


# tentative:
def sub_recursion_eval(root, graph_):  # eval per fork, same as in comp_slice, still flat aggH, add valt to return?

    term_tt = [[],[]]  # term if both empty, not sure its needed, just check fback_tt?
    for graph in graph_:
        node_ = copy(graph.node_tt)  # still graph.node_
        sub_tt = []
        fr = 0
        for fder in 0,1:
            if graph.val_Ht[fder][-1] > G_aves[fder] * graph.rdn_Ht[fder][-1] and len(graph.node_tt) > ave_nsubt[fder]:
                graph.rdn_Ht[fder][-1] += 1  # estimate, no node.rdnt[fd]+=1?
                term_tt[fder] = [1,1]
                fr = 1
                sub_tt += [sub_recursion(root, graph, node_, fder, term_tt)]  # comp_der|rng in graph -> parLayer, sub_Gs
            else:
                sub_tt += [node_]
                # root.fback_tt[fder] += [[graph.aggH, graph.val_Ht, graph.rdn_Ht]]: feedback from sub_recursion only?
        if fr:
            graph.node_tt = sub_tt  # else still graph.node_
    for fder in 0,1:
        for fd in 0,1:
            # or no term_tt is needed: all forks must terminate here?
            if term_tt[fder][fd] and root.fback_tt[fder][fd]:  # no lower layers in any graph
                feedback(root, fder, fd)


def sub_recursion(root, graph, node_, fder, term_tt):  # rng+: extend G_ per graph, der+: replace G_ with derG_, valt=[0,0]?

    if not fder:  # add link layer:
        for node in node_: node.link_H += [[]]
    comp_G_(node_, pri_G_=None, f1Q=1, fder=fder)  # cross-comp all nodes in rng
    sub_t = []
    for fd in 0, 1:
        graph.rdn_Ht[fd][-1] += 1  # estimate
        pri_root_tt_ = []
        for node in node_:
            pri_root_tt_ += [[node.root_tt]]  # to be transferred to new graphs
            node.root_tt[fder][fd] = []  # fill with new graphs
        sub_G_ = form_graph_(node_, fder, fd, pri_root_tt_)  # cluster sub_graphs via link_H
        sub_recursion_eval(graph, sub_G_)
        for G in sub_G_:  # tentative, move to form_graph_?
            root.fback_tt[fder][fd] += [[G.aggH, G.val_Ht, G.rdn_Ht]]
        # not sure its needed:
        if sub_G_: term_tt[fder][fd] = 0
        sub_t += [sub_G_]

    return sub_t

# tentative
def feedback(root, fder, fd):  # append new der layers to root

    # not revised:
    Fback = deepcopy(root.fback_t[fder].pop())  # init with 1st fback: [aggH,val_Ht,rdn_Ht]
    while root.fback_t[fder]:
        aggH, val_Ht, rdn_Ht = root.fback_t[fder].pop()
        sum_aggH(Fback, [aggH, val_Ht, rdn_Ht], base_rdn=0)

    sum_aggH([root.aggH, root.val_Ht,root.rdn_Ht], Fback, base_rdn=0)  # both fder forks sum into a same root?

    for fder, root_t in enumerate(root.root_tt):
        for fd, root_ in enumerate(root_t):
                for rroot in root_:
                    rroot.fback_t[fder] += [Fback]
                    # it's not rroot.node_tt, we need to concat and check the deepest levels of node nesting?:
                    if len(rroot.fback_tt[fder][fd]) == len(rroot.node_tt[fder][fd]):  # all nodes term and fed back to root
                        feedback(rroot, fder, fd)  # aggH/rng in sum2PP, deeper rng layers are appended by feedback


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
aggH: [[subH_t, valt, rdnt]]: composition layers, ext per G
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