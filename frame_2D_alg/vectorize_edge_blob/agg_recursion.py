import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from .classes import Cgraph, CderG
from .filters import aves, ave, ave_nsubt, ave_sub, ave_agg, G_aves, med_decay, ave_distance, ave_Gm, ave_Gd
from .comp_slice import comp_angle, comp_ptuple, sum_ptuple, comp_derH, sum_derH, comp_aangle
# from .sub_recursion import feedback  # temporary

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
This resembles neuron, which has dendritic tree as input and axonal tree as output.
But we have recursively structured param sets packed in each level of these trees, there is no such structure in neurons.
Diagram: 
https://github.com/boris-kz/CogAlg/blob/76327f74240305545ce213a6c26d30e89e226b47/frame_2D_alg/Illustrations/generic%20graph.drawio.png
-
Clustering criterion is G.M|D, summed across >ave vars if selective comp (<ave vars are not compared, so they don't add costs).
Fork selection should be per var or co-derived der layer or agg level. 
There are concepts that include same matching vars: size, density, color, stability, etc, but in different combinations.
Weak value vars are combined into higher var, so derivation fork can be selected on different levels of param composition.

agg+: [[new_aggLev]] += Node_aggH, where each lower aggLev is subH
separate G.derH, with len = min([len(node.derH) for node in G.node_])
'''

# not fully updated
def agg_recursion(root, node_):  # compositional recursion in root.PP_

    for i in 0,1: root.rdnt[i] += 1  # estimate, no node.rdnt[fder] += 1?

    for fder in 0, 1:
        comp_G_(node_, pri_G_=None, f1Q=1, fder=fder)  # cross-comp all Gs within rng
        for fd in 0, 1:
            graph_ = form_graph_(node_, fder, fd)  # clustering via link_t, select by fder
            # sub+, eval last layer?:
            if root.valt[fd] > ave_sub * root.rdnt[fd] and graph_:  # fixed costs and non empty graph_, same per fork
                sub_recursion_eval(root, graph_)
            # agg+, eval all layers?:
            if  sum(root.valt) > G_aves[fd] * ave_agg * sum(root.rdnt) and len(graph_) > ave_nsubt[fd]:
                agg_recursion(root, node_)  # replace root.node_ with new graphs
            elif root.root:  # if deeper agg+
                feedback(root, fd)  # update root.root..H, breadth-first
            root.node_tt[fder][fd] = graph_

# draft:
def comp_G_(G_, pri_G_=None, f1Q=1, fder=0):  # cross-comp in G_ if f1Q, else comp between G_ and pri_G_, if comp_node_?

    while G_:
        G = G_.pop()  # node_
        if fder: _G_ = [link.node_[1] if G is link.node_[0] else link.node_[0] for link in G.link_tH[-(1+fder)][1]]
        else:    _G_ = G_ if f1Q else pri_G_  # all Gs in rng+
        for _G in _G_:
            if _G in G.compared_:  # was compared in prior rng
                continue
            dy = _G.box[0]-G.box[0]; dx = _G.box[1]-G.box[1]
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity
            if distance < ave_distance * ((sum(_G.valt) + sum(G.valt)) / (2*sum(G_aves))):
                G.compared_ += [_G]; _G.compared_ += [G]
                # same comp for cis and alt components:
                for _cG, cG in ((_G, G), (_G.alt_Graph, G.alt_Graph)):
                    if _cG and cG:  # alt Gs maybe empty
                        # form new layer of links:
                        comp_G(_cG, cG, distance, [dy,dx])
    '''
    combine cis,alt in aggH? comp alts,val,rdn? cluster per var set if recurring across root: type eval if root M|D?
    '''
def comp_G(_G, G, distance, A):

    # / P:
    mtuple, dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1)
    mval, dval = sum(mtuple), sum(dtuple)
    Mval = mval; Dval = dval; Mrdn = 1+(dval>mval); Drdn = 1+dval<=mval
    # / PP:
    dderH, valt, rdnt = comp_derH(_G.derH, G.derH, rn=1)
    dderH = [[[mtuple,dtuple],[Mval,Dval],[Mrdn,Drdn]]] + dderH  # add der_tuple as 1st der order
    mval,dval =valt
    Mval += valt[0]; Dval += valt[1]; Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1] + dval<=mval
    # ext:
    _L,_S,_A = _G.L,_G.S,_G.A; L,S,A = G.L,G.S,G.A
    dL = _L-L; Dval+=dL; mL = ave-abs(dL); Mval+=mL
    dS = _S/_L-S/L; Dval+=dS; mS=ave-abs(dS); Mval+=mS
    mA, dA = comp_angle(_A,A); Mval+=mA; Dval+=dA
    dderH = [[[mL,mS,mA][dL,dS,dA]]] + dderH  # der_ext in dderH[0]
    # / G:
    if _G.aggH and G.aggH:  # empty in base fork
        subH, valt, rdnt = comp_aggH(_G.aggH, G.aggH, rn=1)
        subH = [dderH,[Mval,Dval],[Mrdn,Drdn]] + subH  # add dderH as 1st der order
        mval,dval =valt
        Mval += valt[0]; Dval += valt[1]; Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1] + dval<=mval
    else:
        subH = [[dderH,[Mval,Dval],[Mrdn,Drdn]]]  # add nesting

    derG = CderG(G1=_G, G2=G, subH=subH, valt=[Mval,Dval], rdnt=[Mrdn,Drdn], S=distance, A=A)
    # add links:
    if valt[0] > ave_Gm:
        _G.link_tH[-1][0] += [derG]; G.link_tH[-1][0] += [derG]  # bi-directional
    if valt[1] > ave_Gd:
        _G.link_tH[-1][1] += [derG]; G.link_tH[-1][1] += [derG]


def form_graph_(G_, fder, fd):  # form list graphs and their aggHs, G is node in GG graph

    node_ = []  # Gs with >0 +ve fork links:
    for G in G_:
        if G.link_tH[-(1+fder)][fd]: node_ += [G]  # all nodes with +ve links, not clustered in graphs yet
    graph_ = []
    # init graphs by link val:
    while node_:  # all Gs not removed in add_node_layer
        G = node_.pop(); gnode_ = [G]
        val = init_graph(gnode_, node_, G, fder, fd, val=0)  # recursive depth-first gnode_ += [_G]
        graph_ += [[gnode_,val]]
    # prune graphs by node val:
    regraph_ = graph_reval_(graph_, [G_aves[fder] for graph in graph_], fder)  # init reval_ to start
    if regraph_:
        graph_[:] = sum2graph_(regraph_, fder)  # sum proto-graph node_ params in graph

    # add_alt_graph_(graph_t)  # overlap+contour, cluster by common lender (cis graph), combined comp?
    return graph_


def init_graph(gnode_, G_, G, fder, fd, val):  # recursive depth-first gnode_+=[_G]

    for link in G.link_tH[-(1+fder)][fd]:
        # all positive links init graph, eval node.link_ in prune_node_layer
        _G = link.node_[1] if link.node_[0] is G else link.node_[0]
        if _G in G_:  # _G is not removed in prior loop
            gnode_ += [_G]
            G_.remove(_G)
            val += _G.valt[fd]  # interval
            val += init_graph(gnode_, G_, _G, fder, fd, val)
    return val


def graph_reval_(graph_, reval_, fd):  # recursive eval nodes for regraph, after pruning weakly connected nodes

    regraph_, rreval_ = [],[]
    aveG = G_aves[fd]

    while graph_:
        graph,val = graph_.pop()
        reval = reval_.pop()  # each link *= other_G.aggH.valt

        if val > aveG:  # else graph is not re-inserted
            if reval < aveG:  # same graph, skip re-evaluation:
                regraph_+=[[graph,val]]; rreval_+=[0]
            else:
                regraph, reval = graph_reval([graph,val], fd)  # recursive depth-first node and link revaluation
                if regraph[1] > aveG:
                    regraph_ += [regraph]; rreval_+=[reval]
    if rreval_:
        if max([reval for reval in rreval_]) > aveG:
            regraph_ = graph_reval_(regraph_, rreval_, fd)  # graph reval while min val reduction

    return regraph_

def graph_reval(graph, fd):  # exclusive graph segmentation by reval,prune nodes and links

    aveG = G_aves[fd]
    reval = 0  # reval proto-graph nodes by all positive in-graph links:

    for node in graph[0]:  # compute reval: link_Val reinforcement by linked nodes Val:
        lval = 0  # link value
        _lval = node.valt[0][fd]  # = sum([link.valt[fd] for link in node.link_tH[-1][fd]])?
        for derG in node.link_tH[-1][fd]:
            val = derG.valt[fd]  # of top aggH only
            _node = derG.node_[1] if derG.node_[0] is node else derG.node_[0]
            lval += val + (_node.valt[0][fd]-val) * med_decay
        reval += _lval - lval  # _node.link_tH val was updated in previous round
    rreval = 0
    if reval > aveG:
        # prune:
        regraph, Val = [], 0  # reformed proto-graph
        for node in graph[0]:
            val = node.valt[0][fd]
            if val < G_aves[fd] and node in graph:  # prune revalued node and its links
                for derG in node.link_tH[-1][fd]:
                    _node = derG.node_[1] if derG.node_[0] is node else derG.node_[0]
                    _link_ = _node.link_tH[-1][fd]
                    if derG in _link_: _link_.remove(derG)
                    rreval += derG.valt[fd] + (_node.valt[0][fd]-derG.valt[fd]) * med_decay  # else same as rreval += link_.val
            else:
                link_ = node.link_tH[-1][fd]  # prune node links only:
                remove_link_ = []
                for derG in link_:
                    _node = derG.node_[1] if derG.node_[0] is node else derG.node_[0]  # add med_link_ val to link val:
                    lval = derG.valt[fd] + (_node.valt[0][fd]-derG.valt[fd]) * med_decay
                    if lval < aveG:  # prune link, else no change
                        remove_link_ += [derG]
                        rreval += lval
                while remove_link_:
                    link_.remove(remove_link_.pop(0))
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

def sum2graph_(graph_, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    Graph_ = []
    for graph in graph_:  # seq Gs
        if graph[1] < G_aves[fd]:  # form graph if val>min only
            continue
        Graph = Cgraph()
        Link_ = []
        for G in graph[0]:
            sum_box(Graph.box, G.box)
            sum_aggH([Graph.aggH[1],Graph.valt[1],Graph.rdnt[1]], [G.aggH[1],G.valt[1],G.rdnt[1]], base_rdn=1)  # G internals
            link_ = G.link_tH[-1][fd]
            Link_[:] = list(set(Link_ + link_))
            aggH=[]; valt=[0,0]; rdnt=[1,1]
            for derG in link_:
                sum_aggH([aggH,valt,rdnt], [derG.aggH,derG.valt,derG.rdnt], base_rdn=1)  # node externals
                sum_box(G.box, derG.node_[0].box if derG.node_[1] is G else derG.node_[1].box)
            G.aggH[1]+=aggH  # internals+=externals after clustering:
            for i in 0,1:
                G.valt[1][i] += valt[i]; G.rdnt[1][i] += rdnt[i]
            Graph.node_ += [G]
        aggH=[]; valt=[0,0]; rdnt=[1,1]
        for derG in Link_:
            sum_aggH([aggH,valt,rdnt], [derG.aggH, derG.valt, derG.rdnt], base_rdn=1)  # sum unique links
        Graph.aggH[1] += aggH  # internals+=externals after clustering:
        for i in 0,1:
            Graph.valt[1][i] += valt[i]; Graph.rdnt[1][i] += rdnt[i]
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
        node_ = copy(graph.node_); sub_G_t = []
        fr = 0
        for fd in 0,1:
            # not sure int or/and ext:
            if graph.valt[1][fd] > G_aves[fd] * graph.rdnt[1][fd] and len(graph.node_) > ave_nsubt[fd]:
                graph.rdnt[1][fd] += 1  # estimate, no node.rdnt[fd] += 1?
                termt[fd] = 0; fr = 1
                sub_G_t += [sub_recursion(graph, node_, fd)]  # comp_der|rng in graph -> parLayer, sub_Gs
            else:
                sub_G_t += [node_]
                if isinstance(root, Cgraph):
                    # merge_externals?
                    root.fback_t[fd] += [[graph.aggH, graph.valt, graph.rdnt]]  # fback_t vs. flat?
        if fr:
            graph.node_ = sub_G_t
    for fd in 0,1:
        if termt[fd] and root.fback_t[fd]:  # no lower layers in any graph
           feedback(root, fd)


def sub_recursion(graph, node_, fd):  # rng+: extend G_ per graph, der+: replace G_ with derG_, valt=[0,0]?

    comp_G_(graph.node_, pri_G_=None, f1Q=1, fd=fd)  # cross-comp all Gs in rng
    sub_G_t = form_graph_(graph)  # cluster sub_graphs via link_tH

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
        sum_aggH([Fback[0],Fback[0],Fback[0]], [aggH, valt, rdnt] , base_rdn=0)
    for i in 0, 1:
        sum_aggH([root.aggH[i], root.valt[i],root.rdnt[i]], [Fback[0][i],Fback[0][i],Fback[0][i]], base_rdn=0)

    if isinstance(root.root, Cgraph):  # not blob
        root = root.root
        root.fback_ += [Fback]
        if len(root.fback_) == len(root.node_[fd]):  # all nodes term, fed back to root.fback_
            feedback(root, fd)  # aggH/ rng layer in sum2PP, deeper rng layers are appended by feedback


def comp_ext(_ext, ext):
    # comp ds:
    (_L,_S,_A), (L,S,A) = _ext, ext
    dL=_L-L; mL=ave-abs(dL); dS=_S/_L-S/L; mS=ave-abs(dS)
    if isinstance(A,list): mA, dA = comp_angle(_A,A)
    else:
        dA=_A-A; mA=ave-abs(dA)

    return [mL,mS,mA], [dL,dS,dA]

def sum_ext(_ext, ext):
    _dL,_dS,_dA = _ext[1]; dL,dS,dA = ext[1]
    if isinstance(dA,list):
        _dA[0]+=dA[0]; _dA[1]+=dA[1]
    else: _dA+=dA
    _ext[1][:] = _dL+dL,_dS+dS, _dA
    if ext[0]:
        for i, _par, par in enumerate(zip(_ext[0],ext[0])):
            _ext[i] = _par+par

'''
derH: [[tuplet, valt, rdnt]]: default input from PP, for both rng+ and der+, sum min len?
subH: [[derH_t, valt, rdnt]]: m,d derH, m,d ext added by agg+ as 1st tuplet
aggH: [[subH_t, valt, rdnt]]: composition layers, ext per G
'''
def comp_subH(_subH, subH, rn):
    dsubH = []
    Mval, Dval, Mrdn, Drdn = 0,0,1,1

    for _lay, lay in zip_longest(_subH, subH, fillvalue=[]):  # compare common lower layer|sublayer derHs
        if _lay and lay:  # also if lower-layers match: Mval > ave * Mrdn?
            _derH = _lay[0][1]; derH = lay[0][1]
            # comp dext:
            _L,_S,_A = _derH[0][1]; L,S,A = derH[0][1]
            dL = _L-L; Dval += dL; mL = ave-abs(dL); Mval += mL
            dS = _S/_L-S/L; Dval += dS; mS = ave - abs(dS); Mval += mS
            dA = _A-A;  Dval+=dA; mA = ave-abs(dL); Mval += mA
            # comp dderH:
            dderH, valt, rdnt = comp_derH(_derH[1:], derH[1:], rn)
            dderH = [[[mL,mS,mA],[dL,dS,dA]]] + dderH  # 1st element in each subLay is der_ext
            dsubH += [[dderH, valt, rdnt]]
            mval,dval = valt
            Mval += mval; Dval += dval; Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + dval <= mval

    return dsubH, [Mval,Dval], [Mrdn,Drdn]  # new layer, 1/2 combined derH


def comp_aggH(_aggH, aggH, rn):  # no separate ext processing?
    daggH = []
    Mval, Dval, Mrdn, Drdn = 0,0,1,1

    for _lev, lev in zip_longest(_aggH, aggH, fillvalue=[]):  # compare common lower layer|sublayer derHs
        if _lev and lev:  # also if lower-layers match: Mval > ave * Mrdn?
            # compare dsubH only:
            dsubH, valt, rdnt = comp_subH(_lev[0][1], lev[0][1], rn)
            daggH += [[dsubH, valt, rdnt]]
            mval,dval = valt
            mrdn = dval > mval; drdn = dval < mval
            Mrdn += rdnt[0] + mrdn; Drdn += rdnt[1] + drdn

    return daggH, [Mval,Dval], [Mrdn,Drdn]  # new layer, 1/2 combined derH

# not revised:
def sum_aggH(T, t, base_rdn):

    AggH, Valt, Rdnt = T
    aggH, valt, rdnt = t
    for i in 0, 1:
        Valt[i] += valt[i]
        Rdnt[i] += rdnt[i]

    if aggH:
        if AggH:
            for Ht, ht in zip_longest(AggH, aggH, fillvalue=None):
                if ht != None:
                    if Ht:
                        if len(Ht)>1 and Ht[1] and not isinstance(Ht[1][0], list):  # unpack each nested level
                            sum_aggH(Ht, ht, base_rdn)
                        elif Ht[0] and isinstance(Ht[0], list) and Ht[0][0] and isinstance(Ht[0][0], list) and not isinstance(Ht[0][0][0], list):  # not sure if there's a better way, check for derH in aggH
                            for Layer, layer in zip_longest(Ht,ht, fillvalue=None):
                                if layer != None:
                                    if Layer:
                                        for i, param in enumerate(layer):
                                            if i<2: sum_ptuple(Layer[i], param)  # mtuple | dtuple
                                            elif i<4: Layer[i] += param  # mval | dval
                                            else:     Layer[i] += param + base_rdn # | mrdn | drdn
                                    elif Layer!=None:
                                        Layer[:] = deepcopy(layer)
                                    else:
                                        Ht += [deepcopy(layer)]
                        else:
                            for H, h in zip(Ht, ht):  # recursively sum of each t of [t, valt, rdnt] (always 2 elements here)
                                sum_aggH(H, h, base_rdn)
                    elif Ht != None:
                        Ht[:] = deepcopy(ht)
                    else:
                        AggH += [deepcopy(ht)]
        else:
            AggH[:] = deepcopy(aggH)