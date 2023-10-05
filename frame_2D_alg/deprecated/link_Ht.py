import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from frame_2D_alg.vectorize_edge_blob.classes import Cgraph, CderG
from frame_2D_alg.vectorize_edge_blob.filters import aves, ave, ave_nsubt, ave_sub, ave_agg, G_aves, med_decay, ave_distance, ave_Gm, ave_Gd
from frame_2D_alg.vectorize_edge_blob.comp_slice import comp_angle, comp_ptuple, sum_ptuple, sum_derH, comp_derH, comp_aangle
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


def agg_recursion(root, node_):  # compositional recursion in root.PP_

    for i in 0,1: root.rdnt[i] += 1  # estimate, no node.rdnt[fder] += 1?

    for fder in 0, 1:
        if fder and len(node_[0].link_tH) < 2:  # 1st call, no der+ links yet?
            continue
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
            root.node_[fder][fd] = graph_


def comp_G_(G_, pri_G_=None, f1Q=1, fder=0):  # cross-comp in G_ if f1Q, else comp between G_ and pri_G_, if comp_node_?

    for G in G_:  # node_
        if fder:
            _G_ = [link.G1 if G is link.G0 else link.G0 for link in G.link_tH[-(1+fder)][1]]
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
                        comp_G(_cG, cG, distance, [dy,dx], fder)
    '''
    combine cis,alt in aggH: alt represents node isolation?
    comp alts,val,rdn? cluster per var set if recurring across root: type eval if root M|D?
    '''
def comp_G(_G, G, distance, A, fder):

    SubH = []
    Mval,Dval = 0,0; Mrdn,Drdn = 1,1

    comp_ext([_G.L,_G.S,_G.A], [G.L,G.S,G.A], [Mval,Dval], [Mrdn,Drdn], SubH)  # der_ext is 1st lay in SubH
    # / P:
    mtuple, dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1)
    mval, dval = sum(mtuple), sum(dtuple)
    Mval += mval; Dval += dval; Mrdn += dval>mval; Drdn += dval<=mval
    DerH = [[[mtuple,dtuple],[Mval,Dval],[Mrdn,Drdn]]]  # 1st lay in DerH
    # / PP:
    dderH, valt, rdnt = comp_derH(_G.derH[0], G.derH[0], rn=1)
    SubH += [DerH+ dderH]  # 2nd lay in SubH, both flat
    mval,dval =valt
    Mval += valt[0]; Dval += valt[1]; Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
    # / G:
    if _G.aggH and G.aggH:  # empty in base fork
        subH, valt, rdnt = comp_aggH(_G.aggH, G.aggH, rn=1)
        SubH += subH  # SubH is a flat list of der_ext | derH s
        mval,dval =valt
        Mval += valt[0]; Dval += valt[1]; Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
    else:
        subH = [[dderH,[Mval,Dval],[Mrdn,Drdn]]]  # add nesting

    derG = CderG(G0=_G, G1=G, subH=subH, valt=[Mval,Dval], rdnt=[Mrdn,Drdn], S=distance, A=A)
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
        _G = link.G1 if link.G0 is G else link.G0
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
        _lval = node.valt[fd]  # = sum([link.valt[fd] for link in node.link_tH[-1][fd]])?
        for derG in node.link_tH[-1][fd]:
            val = derG.valt[fd]  # of top aggH only
            _node = derG.G1 if derG.G0 is node else derG.G0
            lval += val + (_node.valt[fd]-val) * med_decay
        reval += _lval - lval  # _node.link_tH val was updated in previous round
    rreval = 0
    if reval > aveG:
        # prune:
        regraph, Val = [], 0  # reformed proto-graph
        for node in graph[0]:
            val = node.valt[fd]
            if val < G_aves[fd] and node in graph:  # prune revalued node and its links
                for derG in node.link_tH[-1][fd]:
                    _node = derG.G1 if derG.G0 is node else derG.G0
                    _link_ = _node.link_tH[-1][fd]
                    if derG in _link_: _link_.remove(derG)
                    rreval += derG.valt[fd] + (_node.valt[fd]-derG.valt[fd]) * med_decay  # else same as rreval += link_.val
            else:
                link_ = node.link_tH[-1][fd]  # prune node links only:
                remove_link_ = []
                for derG in link_:
                    _node = derG.G1 if derG.G0 is node else derG.G0  # add med_link_ val to link val:
                    lval = derG.valt[fd] + (_node.valt[fd]-derG.valt[fd]) * med_decay
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
            sum_ptuple(Graph.ptuple, G.ptuple)
            sum_derH(Graph.derH, G.derH, base_rdn=1)  # base_rdn?
            sum_aggH([Graph.aggH,Graph.valt,Graph.rdnt], [G.aggH,G.valt,G.rdnt], base_rdn=1)
            link_ = G.link_tH[-1][fd]
            Link_[:] = list(set(Link_ + link_))
            subH=[]; valt=[0,0]; rdnt=[1,1]
            for derG in link_:
                sum_subH([subH,valt,rdnt], [derG.subH,derG.valt,derG.rdnt], base_rdn=1)
                sum_box(G.box, derG.G0.box if derG.G1 is G else derG.G1.box)
            G.aggH += [subH]
            for i in 0,1:
                G.valt[i] += valt[i]; G.rdnt[i] += rdnt[i]
            Graph.node_ += [G]  # converted to node_tt by feedback
        subH=[]; valt=[0,0]; rdnt=[1,1]
        for derG in Link_:
            sum_subH([subH,valt,rdnt], [derG.subH, derG.valt, derG.rdnt], base_rdn=1)  # sum unique links
        Graph.aggH += [subH]
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


def comp_ext(_ext, ext, Valt, Rdnt, subH):  # comp ds:

    (_L,_S,_A),  (L,S,A) = _ext, ext
    dL=_L-L;      mL=ave-abs(dL)
    dS=_S/_L-S/L; mS=ave-abs(dS)
    if isinstance(A,list):
        mA, dA = comp_angle(_A,A)
    else:
        dA= _A-A; mA= ave-abs(dA)

    M = mL+mS+mA; D = dL+dS+dA
    Valt[0] += M; Valt[1] += D
    Rdnt[0] += D>M; Valt[1] += D<=M
    subH += [[[mL,mS,mA], [dL,dS,dA]]]


def sum_ext(Extt, extt):

    if isinstance(Extt[0], list):  # ext fd tuple
        for fd, (Ext,ext) in enumerate(zip(Extt,extt)):
            for i,(Par,par) in enumerate(zip(Ext,ext)):
                Ext[i] = Par+par
    else:  # single ext
        for i in 0,1: Extt[i]+=extt[i]  # L,S
        for j in 0,1: Extt[2][j]+=extt[2][j]  # dy,dx in angle

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
            if isinstance(_lay[0][0],list):  # _lay[0][0] is derHt

                dderH, valt, rdnt = comp_derH(_lay[0][1], lay[0][1], rn)
                DerH += [[dderH, valt, rdnt]]  # for flat derH
                mval,dval = valt
                Mval += mval; Dval += dval; Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + dval <= mval
            else:  # _lay[0][0] is L, comp dext:
                comp_ext(_lay[1],lay[1], [Mval,Dval], [Mrdn,Drdn], DerH)

    return DerH, [Mval,Dval], [Mrdn,Drdn]  # new layer, 1/2 combined derH

def comp_aggH(_aggH, aggH, rn):  # no separate ext processing?
      SubH = []
      Mval, Dval, Mrdn, Drdn = 0,0,1,1

      for _lev, lev in zip_longest(_aggH, aggH, fillvalue=[]):  # compare common lower layer|sublayer derHs
          if _lev and lev:  # also if lower-layers match: Mval > ave * Mrdn?
              # compare dsubH only:
              dsubH, valt, rdnt = comp_subH(_lev[0][1], lev[0][1], rn)
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
                    if isinstance(Layer[0][0], list):  # _lay[0][0] is derH
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

import numpy as np
from itertools import zip_longest
from copy import copy, deepcopy
from frame_2D_alg.vectorize_edge_blob.classes import CderP, CPP
from frame_2D_alg.vectorize_edge_blob.filters import ave, aves, vaves, ave_dangle, ave_daangle,med_decay, aveB, P_aves

'''
comp_slice traces edge blob axis by cross-comparing vertically adjacent Ps: slices across edge blob, along P.G angle.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)

P: any angle, connectivity in P_ is traced through root_s of derts adjacent to P.dert_, possibly forking. 
len prior root_ sorted by G is rdn of each root, to evaluate it for inclusion in PP, or starting new P by ave*rdn.
'''

def comp_slice(edge, verbose=False):  # high-G, smooth-angle blob, composite dert core param is v_g + iv_ga

    P_ = []
    for P in edge.node_:  # init P_, must be contiguous, gaps filled in scan_P_rim
        link_ = copy(P.link_tH[-1][0])  # init rng+
        P.link_tH[-1][0] = []  # fill with derPs in comp_P
        P_ +=[[P,link_]]
    for P, link_ in P_:
        for _P in link_:  # or spliced_link_ if active
            comp_P(_P,P, fder=0)  # replaces P.link_ Ps with derPs
    # convert node_ to node_tt:
    edge.node_ = [form_PP_t([Pt[0] for Pt in P_], PP_=None, base_rdn=2, fder=0), [[], []]]  # root fork is rng+ only


def comp_P(_P,P, fder=1, derP=None):  #  derP if der+, S if rng+

    aveP = P_aves[fder]
    rn = len(_P.dert_)/ len(P.dert_)

    if fder:  # der+: extend in-link derH
        rn *= len(_P.link_tH[-2][fder]) / len(P.link_tH[-2][fder])  # derH is summed from links
        dderH, valt, rdnt = comp_derH(_P.derH, P.derH, rn)  # += fork rdn
        derP = CderP(derH = derP.derH+dderH, valt=valt, rdnt=rdnt, P=P,_P=_P, S=derP.S)  # dderH valt,rdnt for new link
        mval,dval = valt; mrdn,drdn = rdnt
    else:  # rng+: add derH
        mtuple,dtuple = comp_ptuple(_P.ptuple, P.ptuple, rn)
        mval = sum(mtuple); dval = sum(dtuple)
        mrdn = 1+(dval>mval); drdn = 1+(1-(dval>mval))  # rdn = Dval/Mval?
        derP = CderP(derH=[[[mtuple,dtuple], [mval,dval],[mrdn,drdn]]], valt=[mval,dval], rdnt=[mrdn,drdn], P=P,_P=_P, S=derP)

    if mval > aveP*mrdn: P.link_tH[-1][0] += [derP]  # +ve links, for fork selection in form_PP_t, not fd as -ve links?
    if dval > aveP*drdn: P.link_tH[-1][1] += [derP]


def comp_derH(_derH, derH, rn):  # derH is a list of der layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    dderH = []  # or = not-missing comparand if xor?
    Mval, Dval, Mrdn, Drdn = 0,0,1,1

    for _lay, lay in zip_longest(_derH, derH, fillvalue=[]):  # compare common lower der layers | sublayers in derHs
        if _lay and lay:  # also if lower-layers match: Mval > ave * Mrdn?

            mtuple, dtuple = comp_dtuple(_lay[0][1], lay[0][1], rn)  # compare dtuples only, mtuples are for evaluation
            mval = sum(mtuple); dval = sum(dtuple)
            mrdn = dval > mval; drdn = dval < mval
            dderH += [[[mtuple,dtuple],[mval,dval],[mrdn,drdn]]]
            Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn

    return dderH, [Mval,Dval], [Mrdn,Drdn]  # new layer, 1/2 combined derH

def comp_dtuple(_ptuple, ptuple, rn):

    mtuple, dtuple = [],[]
    for _par, par, ave in zip(_ptuple, ptuple, aves):  # compare ds only?
        npar= par*rn
        mtuple += [min(_par, npar) - ave]
        dtuple += [_par - npar]

    return [mtuple, dtuple]


def form_PP_t(P_, PP_, base_rdn, fder):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = []
    for fd in 0,1:
        qPP_ = []  # initial sequence_PP s
        for P in P_:
            if not P.root_tt[fder][fd]:  # else already packed in qPP
                qPP = [[P]]  # init PP is 2D queue of (P,val)s of all layers?
                P.root_tt[fder][fd] = qPP; val = 0
                uplink_ = P.link_tH[-1][fd]
                uuplink_ = []  # next layer of uplinks
                while uplink_:  # later uuplink_
                    for derP in uplink_:
                        _P = derP._P
                        if _P not in P_:  # _P is outside of current PP, merge its root PP:
                            _PP = _P.root_tt[fder][fd]
                            if _PP:  # _P is already clustered
                                for _node in _PP.node_:
                                    if _node not in qPP[0]:
                                        qPP[0] += [_node]; _node.root_tt[fder][fd] = qPP  # reassign root
                                PP_.remove(_PP)
                        else:
                            _qPP = _P.root_tt[fder][fd]
                            if _qPP:
                                if _qPP is not qPP:  # _P may be added to qPP via other down-linked P
                                    val += _qPP[1]  # merge _qPP in qPP:
                                    for qP in _qPP[0]:
                                        qP.root_tt[fder][fd] = qPP
                                        qPP[0] += [qP]  # qP_+=[qP]
                                    qPP_.remove(_qPP)
                            else:
                                qPP[0] += [_P]  # pack bottom up
                                _P.root_tt[fder][fd] = qPP
                                val += derP.valt[fd]
                                uuplink_ += derP._P.link_tH[-1][fd]
                    uplink_ = uuplink_
                    uuplink_ = []
                qPP += [val, ave + 1]  # ini reval=ave+1, keep qPP same object for ref in P.roott
                qPP_ += [qPP]

        # prune qPPs by mediated links vals:
        rePP_ = reval_PP_(qPP_, fder, fd)  # PP = [qPP,valt,reval]
        CPP_ = [sum2PP(qPP, base_rdn, fder, fd) for qPP in rePP_]

        PP_t += [CPP_]  # least one PP in rePP_, which would have node_ = P_

    return PP_t  # add_alt_PPs_(graph_t)?


def reval_PP_(PP_, fder, fd):  # recursive eval / prune Ps for rePP

    rePP_ = []
    while PP_:  # init P__
        P_, val, reval = PP_.pop(0)
        # Ave = ave * 1+(valt[fd] < valt[1-fd])  # * PP.rdn for more selective clustering?
        if val > ave:
            if reval < ave:  # same graph, skip re-evaluation:
                rePP_ += [[P_,val,0]]  # reval=0
            else:
                rePP = reval_P_(P_,fd)  # recursive node and link revaluation by med val
                if val > ave:  # min adjusted val
                    rePP_ += [rePP]
                else:
                    for P in rePP: P.root_tt[fder][fd] = []
        else:  # low-val qPPs are removed
            for P in P_: P.root_tt[fder][fd] = []

    if rePP_ and max([rePP[2] for rePP in rePP_]) > ave:  # recursion if any min reval:
        rePP_ = reval_PP_(rePP_, fder, fd)

    return rePP_

def reval_P_(P_, fd):  # prune qPP by link_val + mediated link__val

    prune_=[]; Val=0; reval=0  # comb PP value and recursion value

    for P in P_:
        P_val = 0; remove_ = []
        for link in P.link_tH[-1][fd]:  # link val + med links val: single mediation layer in comp_slice:
            link_val = link.valt[fd] + sum([mlink.valt[fd] for mlink in link._P.link_tH[-1][0]]) * med_decay
            if link_val < vaves[fd]:
                remove_ += [link]; reval += link_val
            else: P_val += link_val
        for link in remove_:
            P.link_tH[-1][fd].remove(link)  # prune weak links
        if P_val * P.rdnt[fd] < vaves[fd]:
            prune_ += [P]
        else:
            Val += P_val * P.rdnt[fd]
    for P in prune_:
        for link in P.link_tH[-1][fd]:  # prune direct links only?
            _P = link._P
            _link_ = _P.link_tH[-1][fd]
            if link in _link_:
                _link_.remove(link); reval += link.valt[fd]

    if reval > aveB:
        P_, Val, reval = reval_P_(P_, fd)  # recursion
    return [P_, Val, reval]


def sum2PP(qPP, base_rdn, fder, fd):  # sum links in Ps and Ps in PP

    P_,_,_ = qPP  # proto-PP is a list
    PP = CPP(fd=fd, node_=P_)
    # accum:
    for i, P in enumerate(P_):
        P.root_tt[fder][fd] = PP
        sum_ptuple(PP.ptuple, P.ptuple)
        L = P.ptuple[-1]
        Dy = P.axis[0]*L/2; Dx = P.axis[1]*L/2; y,x =P.yx
        if i: Y0=min(Y0,(y-Dy)); Yn=max(Yn,(y+Dy)); X0=min(X0,(x-Dx)); Xn=max(Xn,(x+Dx))
        else: Y0=y-Dy; Yn=y+Dy; X0=x-Dx; Xn=x+Dx  # init

        for derP in P.link_tH[-1][fder]:
            derH, valt, rdnt = derP.derH, derP.valt, derP.rdnt
            sum_derH([P.derH,P.valt,P.rdnt], [derH,valt,rdnt], base_rdn)
            _P = derP._P  # bilateral summation:
            sum_derH([_P.derH,_P.valt,_P.rdnt], [derH,valt,rdnt], base_rdn)
        # excluding bilateral sums:
        sum_derH([PP.derH,PP.valt,PP.rdnt], [P.derH,P.valt,P.rdnt], base_rdn)

    PP.box =(Y0,Yn,X0,Xn)
    return PP


def sum_derH(T, t, base_rdn):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    DerH, Valt, Rdnt = T
    derH, valt, rdnt = t
    for i in 0, 1:
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i] + base_rdn
    if DerH:
        for Layer, layer in zip_longest(DerH,derH, fillvalue=[]):
            if layer:
                if Layer:
                    for i in range(0,1):
                        sum_ptuple(Layer[0][i], layer[0][i])  # ptuplet
                        Layer[1][i] += layer[1][i]  # valt
                        Layer[2][i] += layer[2][i] + base_rdn  # rdnt
                else:
                    DerH += [deepcopy(layer)]
    else:
        DerH[:] = deepcopy(derH)

def sum_ptuple(Ptuple, ptuple, fneg=0):

    for i, (Par, par) in enumerate(zip_longest(Ptuple, ptuple, fillvalue=None)):
        if par != None:
            if Par != None:
                if isinstance(Par, list):  # angle or aangle
                    for i,(P,p) in enumerate(zip(Par,par)):
                        Par[i] = P-p if fneg else P+p
                else:
                    Ptuple[i] += (-par if fneg else par)  # now includes n in ptuple[-1]?
            elif not fneg:
                Ptuple += [copy(par)]

def comp_ptuple(_ptuple, ptuple, rn):  # 0der

    mtuple, dtuple = [],[]
    # _n, n = _ptuple, ptuple: add to rn?
    for i, (_par, par, ave) in enumerate(zip(_ptuple, ptuple, aves)):
        if isinstance(_par, list):
            if len(_par)==2: m,d = comp_angle(_par, par)
            else:            m,d = comp_aangle(_par, par)
        else:  # I | M | Ma | G | Ga | L
            npar= par*rn  # accum-normalized par
            d = _par - npar
            if i: m = min(_par,npar)-ave
            else: m = ave-abs(d)  # inverse match for I, no mag/value correlation

        mtuple+=[m]; dtuple+=[d]
    return [mtuple, dtuple]


def comp_angle(_angle, angle):  # rn doesn't matter for angles

    _Dy, _Dx = _angle
    Dy, Dx = angle
    _G = np.hypot(_Dy,_Dx); G = np.hypot(Dy,Dx)
    sin = Dy / (.1 if G == 0 else G);     cos = Dx / (.1 if G == 0 else G)
    _sin = _Dy / (.1 if _G == 0 else _G); _cos = _Dx / (.1 if _G == 0 else _G)
    sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    # cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
    dangle = sin_da
    mangle = ave_dangle - abs(dangle)  # inverse match, not redundant as summed across sign

    return [mangle, dangle]

def comp_aangle(_aangle, aangle):

    _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _aangle
    sin_da0, cos_da0, sin_da1, cos_da1 = aangle

    sin_dda0 = (cos_da0 * _sin_da0) - (sin_da0 * _cos_da0)
    cos_dda0 = (cos_da0 * _cos_da0) + (sin_da0 * _sin_da0)
    sin_dda1 = (cos_da1 * _sin_da1) - (sin_da1 * _cos_da1)
    cos_dda1 = (cos_da1 * _cos_da1) + (sin_da1 * _sin_da1)
    # for 2D, not reduction to 1D:
    # aaangle = (sin_dda0, cos_dda0, sin_dda1, cos_dda1)
    # day = [-sin_dda0 - sin_dda1, cos_dda0 + cos_dda1]
    # dax = [-sin_dda0 + sin_dda1, cos_dda0 + cos_dda1]
    gay = np.arctan2((-sin_dda0 - sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in y?
    gax = np.arctan2((-sin_dda0 + sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in x?

    # daangle = sin_dda0 + sin_dda1?
    daangle = np.arctan2(gay, gax)  # diff between aangles, probably wrong
    maangle = ave_daangle - abs(daangle)  # inverse match, not redundant as summed

    return [maangle,daangle]

from itertools import zip_longest
import numpy as np
from copy import copy, deepcopy
from frame_2D_alg.vectorize_edge_blob.filters import PP_aves, ave_nsubt
from frame_2D_alg.vectorize_edge_blob.classes import CP, CPP
from frame_2D_alg.vectorize_edge_blob.comp_slice import comp_P, form_PP_t, sum_derH
from dataclasses import replace
'''
Nesting in dH, implicit or explicit, is added by each layer of comp, which forms dderH | dH.
That dH has the same nesting as compared G.H, but then G.H += [dH].

To preserve forks we encoded per layer, as fders in id_H (fd is clustering type, it doesn't matter here).
Those are fders are single sequence in upward H (which is not represented), but a fork tree if we trace it down.

We don't represent forks because they are merged in feedback, otherwise it's too complex:
we would need fback_T and Fback_T: last_layer_nforks = 2^n_higher_layers, to map to last-layer root forks.
'''

def sub_recursion_eval(root, PP_):  # fork PP_ in PP or blob, no derH in blob

    termt = [1,1]
    # PP_ in PP_t:
    for PP in PP_:
        sub_tt = []  # from rng+, der+
        fr = 0  # recursion in any fork
        for fder in 0,1:  # rng+ and der+:
            if len(PP.node_) > ave_nsubt[fder] and PP.valt[fder] > PP_aves[fder] * PP.rdnt[fder]:
                termt[fder] = 0
                if not fr:  # add link_tt and root_tt for both comp forks:
                    for P in PP.node_:
                        P.root_tt = [[None,None],[None,None]]
                        P.link_tH += [[[],[]]]  # form root_t, link_t in sub+:
                sub_tt += [sub_recursion(PP, PP_, fder)]  # comp_der|rng in PP->parLayer
                fr = 1
            else:
                sub_tt += [PP.node_]
                root.fback_ += [[PP.derH, PP.valt, PP.rdnt]]  # separate feedback per terminated comp fork
        if fr:
            PP.node_ = sub_tt  # nested PP_ tuple from 2 comp forks, each returns sub_PP_t: 2 clustering forks, if taken

    return termt

def sub_recursion(PP, PP_, fder):  # evaluate PP for rng+ and der+, add layers to select sub_PPs

    comp_der(PP.node_) if fder else comp_rng(PP.node_, PP.rng+1)  # same else new P_ and links
    # eval all| last layer?:
    PP.rdnt[fder] += PP.valt[fder] - PP_aves[fder]*PP.rdnt[fder] > PP.valt[1-fder] - PP_aves[1-fder]*PP.rdnt[1-fder]
    sub_PP_t = form_PP_t(PP.node_, PP_, base_rdn=PP.rdnt[fder], fder=fder)  # replace node_ with sub_PPm_, sub_PPd_

    for fd, sub_PP_ in enumerate(sub_PP_t):  # not empty: len node_ > ave_nsubt[fd]
        for sPP in sub_PP_: sPP.root_tt[fder][fd] = PP
        termt = sub_recursion_eval(PP, sub_PP_)

        if any(termt):  # fder: comp fork, fd: form fork:
            for fder in 0,1:
                if termt[fder] and PP.fback_:
                    feedback(PP, fder=fder, fd=fd)
                    # upward recursive extend root.derH, forward eval only
    return sub_PP_t

def feedback(root, fder, fd):  # append new der layers to root

    Fback = deepcopy(root.fback_.pop())
    # init with 1st fback: [derH,valt,rdnt], derH: [[mtuple,dtuple, mval,dval, mrdn, drdn]]
    while root.fback_:
        sum_derH(Fback,root.fback_.pop(), base_rdn=0)
    sum_derH([root.derH, root.valt,root.rdnt], Fback, base_rdn=0)

    if isinstance(root.root_tt[fder][fd], CPP):  # not blob
        root = root.root_tt[fder][fd]
        root.fback_ += [Fback]
        if len(root.fback_) == len(root.node_):  # all original nodes term, fed back to root.fback_t
            feedback(root, fder, fd)  # derH per comp layer in sum2PP, add deeper layers by feedback
        '''
        root.fback_t[fd] += [Fback] 
        if len(root.fback_t[fd]) == len(root.node_):  # all original nodes term, fed back to root.fback_t
            feedback(root, fder, fd)  # derH per comp layer in sum2PP, deeper layers are appended by feedback
        fback_t was to represent derH as a fork tree,
        but that should be fback_T and Fback_T: last_layer_nforks = 2^n_higher_layers, which seems too complex
        '''

def comp_rng(iP_, rng):  # form new Ps and links, switch to rng+n to skip clustering?

    P_ = []
    for P in iP_:
        for derP in P.link_tH[-2][0]:  # scan lower-layer mlinks
            _P = derP._P
            for _derP in _P.link_tH[-2][0]:  # next layer of all links, also lower-layer?
                __P = _derP._P  # next layer of Ps
                distance = np.hypot(__P.yx[1]-P.yx[1], __P.yx[0]-P.yx[0])   # distance between mid points
                if distance > rng:
                    comp_P(P,__P, fder=0, derP=distance)  # distance=S, mostly lateral, relative to L for eval?
        P_ += [P]
    return P_

def comp_der(P_):  # keep same Ps and links, increment link derH, then P derH in sum2PP

    for P in P_:
        for derP in P.link_tH[-2][1]:  # scan lower-layer dlinks
            # comp extended derH of the same Ps, to sum in lower-composition sub_PPs:
            comp_P(derP._P,P, fder=1, derP=derP)
    return P_

''' 
lateral splicing of initial Ps, not needed: will be spliced in PPs through common forks? 
spliced_link_ = []
__P = link_.pop()
for _P in link_.pop():
    spliced_link_ += lat_comp_P(__P, _P)  # comp uplinks, merge if close and similar, return merged __P
    __P = _P
'''

# draft, ignore for now:
def lat_comp_P(_P,P):  # to splice, no der+

    ave = P_aves[0]
    rn = len(_P.dert_)/ len(P.dert_)

    mtuple,dtuple = comp_ptuple(_P.ptuple[:-1], P.ptuple[:-1], rn)

    _L, L = _P.ptuple[-1], P.ptuple[-1]
    gap = np.hypot((_P.y - P.y), (_P.x, P.x))
    rL = _L - L
    mM = min(_L, L) - ave
    mval = sum(mtuple); dval = sum(dtuple)
    mrdn = 1+(dval>mval); drdn = 1+(1-(dval>mval))  # rdn = Dval/Mval?