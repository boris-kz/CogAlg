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

    ave = G_aves[fder]
    # compute max of quasi-Gaussians: val + sum([_val * (link_val/max_val]):
    max_ = select_max_(node_, fder, ave)

    graph_ = segment_node_(node_, max_, pri_root_T_, fder, fd)
    prune_graphs(graph_, fder, fd)  # sort node roots and prune the weak


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


def segment_node_(node_, max_, pri_root_T_, fder, fd):

    graph_ = []  # initialize graphs with local maxes, then prune links to add other nodes:

    for max_node in max_:
        graph = [[max_node], [pri_root_T_[node_.index(max_node)]], sum(max_node.val_Ht[fder])]
        max_node.root_T[fder][fd] = graph
        _nodes = [max_node]  # current periphery of the graph
        # search links recursively outwards:
        while _nodes:
            nodes = []
            for node in _nodes:
                val = sum(node.val_Ht[fder]) - ave * sum(node.rdn_Ht[fder])
                for link in node.link_H[-1]:
                    _node = link.G1 if link.G0 is node else link.G0
                    if _node not in graph[0]:
                        _val = sum(_node.val_Ht[fder]) - ave * sum(_node.rdn_Ht[fder])
                        link_rel_val = link.valt[fder] / link.valt[2]
                        # link eval, tentative:
                        if (val+_val) * link_rel_val > 0:  # pack _node in graph:
                            graph[0] += [_node]
                            pri_root_T = pri_root_T_[node_.index(_node)]
                            if pri_root_T not in graph[1]:
                                # need ro unpack root tree and align forks?:
                                graph[1] += [pri_root_T]  # transfer node roots to new intermediate graph
                            graph[2] += link_rel_val * _val
                            _node.root_T[fder][fd] = graph  # single root per fork?
                            nodes += [_node]
            _nodes = nodes
        graph_ += [graph]
    return graph_

def prune_graphs(_graph_, fder, fd):

    graph_ = []
    for _node_, _pri_root_T_, _Val in _graph_:
        node_, pri_root_T_ = [],[]
        for node in _node_:
            roots = sorted(node.root_T[fder][fd], key=lambda root: root[2], reverse=True)
            rdn_Val = 1  # val of stronger inclusion in overlapping graphs, in same fork
            for root in roots:
                Val = sum(node.val_Ht) - ave * (sum(node.rdn_Ht) + rdn_Val/ave)  # not sure
                if Val > 0:
                    node_ += [node]
                    for link in node.link_Ht[fder]:
                        # tentative re-eval node links:
                        _node = link.G1 if link.G0 is node else link.G0
                        sum(_node.val_ht)  * (link.valt[fder]/link.valt[2]) - ave * (sum(_node.rdn_Ht[fder]) + node.Rdn)
                        # prune the value above is negative?
                    rdn_Val += root[2]
                    
            # not revised:
            else:
                node.root_T[fder][fd] = []  # reset root
                pri_root_T_.pop(j)  # remove pri_root_T
                Val -= Val_[j]  # reduce by link_rel_val * _val

        # evaluate and prune graphs
        if Val > ave:
            graph_ += [[new_nodes, pri_root_T_, Val]]

    return graph_

# replace with prune_graphs:
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


def sum2graph_(graph_, fder, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    Graph_ = []
    for graph in graph_:  # seq Gs
        if graph[2] < G_aves[fd]:  # form graph if val>min only
            continue
        Graph = Cgraph(root_T = graph[1], L=len(graph[0]))  # n nodes
        Link_ = []
        for G in graph[0]:
            sum_box(Graph.box, G.box)
            sum_ptuple(Graph.ptuple, G.ptuple)
            sum_derH(Graph.derH, G.derH, base_rdn=1)  # base_rdn?
            sum_aggH([Graph.aggH,Graph.valt,Graph.rdnt], [G.aggH,G.valt,G.rdnt], base_rdn=1)
            link_ = G.link_H[-1]
            Link_[:] = list(set(Link_ + link_))
            subH=[]; valt=[0,0]; rdnt=[1,1]  # no max in G.valt?
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
        mA, dA = comp_angle(_A,A); max_mA = 1.5
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
                DerH += comp_ext(_lay[1],lay[1], [Mval,Dval,Maxv], [Mrdn,Drdn])

    return DerH, [Mval,Dval,Maxv], [Mrdn,Drdn]  # new layer, 1/2 combined derH

def comp_aggH(_aggH, aggH, rn):  # no separate ext processing?
      SubH = []
      Mval, Dval, Maxv, Mrdn, Drdn = 0,0,0,1,1

      for _lev, lev in zip_longest(_aggH, aggH, fillvalue=[]):  # compare common lower layer|sublayer derHs
          if _lev and lev:  # also if lower-layers match: Mval > ave * Mrdn?
              # compare dsubH only:
              dsubH, valt, rdnt = comp_subH(_lev[0], lev[0], rn)
              SubH += [[dsubH, valt, rdnt]]
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

    AggH, Valt, Rdnt = T
    aggH, valt, rdnt = t

    for i in 0,1:  # link maxv is not summed in G.valt
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