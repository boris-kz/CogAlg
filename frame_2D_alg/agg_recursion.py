import sys
import numpy as np
from itertools import zip_longest
from copy import deepcopy, copy
from class_cluster import ClusterStructure, NoneType, comp_param, Cdert
import math as math
from comp_slice import *

'''
Blob edges may be represented by higher-composition patterns, etc., if top param-layer match,
in combination with spliced lower-composition patterns, etc, if only lower param-layers match.
This may form closed edge patterns around flat core blobs, which defines stable objects. 
Graphs are formed from blobs that match over <max distance. 
-
They are also assigned adjacent alt-fork graphs, which borrow predictive value from the graph. 
Primary value is match, so difference patterns don't have independent value. 
They borrow value from proximate or average match patterns, to the extent that they cancel their projected match. 
But alt match patterns would borrow borrowed value, which may be too tenuous to track, we can use the average instead.
-
cross_comp(alt_graph_) if high value?
'''
# aves defined for rdn+1:
ave_G = 6  # fixed costs per G
ave_Gm = 5  # for inclusion in graph
ave_Gd = 4
G_aves = [ave_Gm, ave_Gd]
ave_med = 3  # call cluster_node_layer
ave_rng = 3  # rng per combined val
ave_ext = 5  # to eval comp_plevel
ave_len = 3
ave_distance = 5
ave_sparsity = 2

class CQ(ClusterStructure):  # nodes or links' forks
    Q = list
    val = float

class Clink_(ClusterStructure):
    Q = list
    val = float
    Qm = list
    mval = float
    Qd = list
    dval = float

class CpH(ClusterStructure):  # hierarchy of params + associated vars

    H = list  # pplayers | players | ptuples
    val = int
    nval = int  # of open links: alt_graph_?
    forks = list  # m|d|am|ad list in H=plevels|pplayers, m|d list in H=players|ptuples
    link_ = lambda: Clink_()  # evaluated external links (graph=node), replace alt_node if open, direct only
    node_ = list  # sub-node_ s concatenated within root node_
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs; no alt_rdn: valt representation in alt_PP_ valts?
    rng = lambda: 1  # not for alt_graphs
    alt_graph_ = list  # contour + overlapping contrast graphs
    # extuple in graph: pplayers only, if S?
    L = list  # der L, init empty
    S = int  # sparsity: ave len link
    A = list  # area and axis: Dy,Dx
    x0 = float
    y0 = float  # center: box x0|y0 + L/2
    xn = float  # max distance from center
    yn = float

class Cgraph(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    root = object  # root graph
    wH = list  # down H: lower Levs feedback accum, H[i] is forks in 4^i fork tree?
    uH = list  # up H: higher Levs feedback accum, H[i] is CpH, H[0] is summed current Lev
    forks = list  # m|d|am|ad per uH[i]
    val = int  # summed across wH+uH? nval of open links: alt_graph_?
    pplayers = lambda: CpH  # current Lev with all associated params, not summed as in wH|uH
''' 
    Gtree: plevels ( forks ( pplayers ( players ( ptuples 
    H: [lev0, lev1, lev2..], forks/lev in wH for feedback; input: H[0 + 1/der+ sub_graph_, in sub+?
    new mutable node_n and immutable lev_n per composition level, lower-lev len_H = higher-lev len_H-1
'''

class CderG(ClusterStructure):  # graph links, within root node_

    node0 = lambda: CpH()  # converted to list in recursion
    node1 = lambda: CpH()
    mplevel = lambda: CpH()  # in alt/contrast if open
    dplevel = lambda: CpH()
    S = int  # sparsity: ave len link
    A = list  # area and axis: Dy,Dx


def agg_recursion(root, fseg):  # compositional recursion in root.PP_, pretty sure we still need fseg, process should be different

    for fork, pplayers in enumerate(root.H[0]):  # root graph 1st plevel forks: mpplayers, dpplayers, alt_mpplayers, alt_dpplayers
        if pplayers: # H: plevels ( forks ( pplayers ( players ( ptuples

            for G in pplayers.node_:  # init forks, rng+ H[1][fork][-1] is immutable, comp frng pplayers
                G.wH += [[[],[],[],[]]]
            mgraph_, dgraph_ = form_graph_(root, fork)  # cross-comp and graph clustering

            for fd, graph_ in enumerate([mgraph_,dgraph_]):  # eval graphs for sub+ and agg+:
                val = sum([graph.val for graph in graph_])
                # intra-graph sub+ comp node:
                if val > ave_sub * (root.rdn):  # same in blob, same base cost for both forks
                    pplayers.rdn+=1  # estimate
                    sub_recursion_g(graph_, val, fseg, fd)  # subdivide graph_ by der+|rng+
                # cross-graph agg+ comp graph:
                if val > G_aves[fd] * ave_agg * (root.rdn) and len(graph_) > ave_nsub:
                    pplayers.rdn += 1  # estimate
                    agg_recursion(root, fseg=fseg)


def form_graph_(root, fork): # form plevel in agg+ or player in sub+, G is node in GG graph; der+: comp_link if fderG, from sub+

    G_ = [root.root.cLev.node_,root.root.cLev.node_][fork%2]  # rng+: cluster higher root, der+: cluster current root
    comp_G_(G_, fork=fork)  # cross-comp all graphs in rng, graphs may be segs | fderGs, root G += link, link.node

    mnode_, dnode_ = [], []  # Gs with >0 +ve fork links:
    for G in G_:
        if G.link_.Qm: mnode_ += [G]  # all nodes with +ve links, not clustered in graphs yet
        if G.link_.Qd: dnode_ += [G]
    graph_t = []
    for fd, node_ in enumerate([mnode_, dnode_]):
        graph_ = []  # init graphs by link val:
        while node_:  # all Gs not removed in add_node_layer
            G = node_.pop(); gnode_ = [G]
            val = add_node_layer(gnode_, node_, G, fd, val=0)  # recursive depth-first gnode_+=[_G]
            graph_ += [CQ(Q=gnode_, val=val)]
        # reform graphs by node val:
        regraph_ = graph_reval(graph_, [ave_G for graph in graph_], fd)  # init reval_ to start
        if regraph_:
            graph_[:] = sum2graph_(regraph_, root, fd, fork)  # sum proto-graph node_ params in graph
            # root for feedback: sum val,node_, then selective unpack?
        graph_t += [graph_]

    # add_alt_graph_(graph_t)  # overlap+contour, cluster by common lender (cis graph), combined comp?
    return graph_t


def graph_reval(graph_, reval_, fd):  # recursive eval nodes for regraph, increasingly selective with reduced node.link_.val

    regraph_, rreval_ = [],[]
    Reval = 0
    while graph_:
        graph = graph_.pop()
        reval = reval_.pop()
        if reval < ave_G:  # same graph, skip re-evaluation:
            regraph_ += [graph]; rreval_ += [0]
            continue
        while graph.Q:  # some links will be removed, graph may split into multiple regraphs, init each with graph.Q node:
            regraph = CQ()
            node = graph.Q.pop()  # node_, not removed below
            val = [node.link_.mval, node.link_.dval][fd]  # in-graph links only
            if val > G_aves[fd]:  # else skip
                regraph.Q = [node]; regraph.val = val  # init for each node, then add _nodes
                readd_node_layer(regraph, graph.Q, node, fd)  # recursive depth-first regraph.Q+=[_node]
            reval = graph.val - regraph.val
            if regraph.val > ave_G:
                regraph_ += [regraph]; rreval_ += [reval]; Reval += reval
    if Reval > ave_G:
        regraph_ = graph_reval(regraph_, rreval_, fd)  # graph reval while min val reduction

    return regraph_

def readd_node_layer(regraph, graph_Q, node, fd):  # recursive depth-first regraph.Q+=[_node]

    for link in [node.link_.Qm, node.link_.Qd][fd]:  # all positive
        _node = link.node1 if link.node0 is node else link.node0
        _val = [_node.link_.mval, _node.link_.dval][fd]
        if _val > G_aves[fd] and _node in graph_Q:
            regraph.Q += [_node]
            graph_Q.remove(_node)
            regraph.val += _val
            readd_node_layer(regraph, graph_Q, _node, fd)

def add_node_layer(gnode_, G_, G, fd, val):  # recursive depth-first gnode_+=[_G]

    for link in G.link_.Q:  # all positive
        _G = link.node1 if link.node0 is G else link.node0
        if _G in G_:  # _G is not removed in prior loop
            gnode_ += [_G]
            G_.remove(_G)
            val += [_G.link_.mval,_G.link_.dval][fd]
            val += add_node_layer(gnode_, G_, _G, fd, val)

    return val

def comp_G_(G_, fork):  # cross-comp Gs (patterns of patterns): Gs, derGs, or segs inside PP, same process, no fderG?

    for i, _G in enumerate(G_):  # G: pplayers from H[fork] or root[fork]
        for G in G_[i+1:]:  # compare each G to other Gs in rng, bilateral link assign, val accum:

            if G in [node for link in _G.link_.Q for node in [link.node0,link.node1]]:
                continue  # this G pair was compared in prior rng+, add frng to skip?
            dx = _G.x0 - G.x0; dy = _G.y0 - G.y0  # center x0,y0
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity, proximity = ave-distance
            if distance < ave_distance * ((_G.val + G.val) / (2*sum(G_aves))):  # comp pplayers:

                mplevel, dplevel = comp_pH(_G, G)  # no need for fork in comp_pH?
                derG = CderG(node0=_G,node1=G, mplevel=mplevel,dplevel=dplevel, S=distance, A=[dy,dx])
                mval = mplevel.val; dval = dplevel.val
                tval = mval + dval
                _G.link_.Q += [derG]; _G.link_.val += tval  # val of combined-fork' +- links?
                G.link_.Q += [derG]; G.link_.val += tval
                if mval > ave_Gm:
                    _G.link_.Qm += [derG]; _G.link_.mval += mval  # no dval for Qm
                    G.link_.Qm += [derG]; G.link_.mval += mval
                if dval > ave_Gd:
                    _G.link_.Qd += [derG]; _G.link_.dval += dval  # no mval for Qd
                    G.link_.Qd += [derG]; G.link_.dval += dval


def sum2graph_(graph_, root, fd, fork):  # sum node and link params into graph, plevel in agg+ or player in sub+

    Graph_ = []  # CpHs
    for graph in graph_:
        if graph.val < ave_G:  # form graph if val>min only
            continue
        Glink_= []; X0,Y0 = 0,0
        UH = CpH
        # 1st pass: define center Y,X and Glink_:
        for node in graph.Q:
            Glink_ = list(set(Glink_ + [node.link_.Qm, node.link_.Qd][fd]))  # unique fork links over graph nodes
            X0 += node.x0; Y0 += node.y0
        L = len(graph.Q); X0/=L; Y0/=L; Xn,Yn = 0,0
        # 2nd pass: extend and sum nodes in graph:
        uH = []  # old levs
        for node in graph.Q:  # CQ(Q=gnode_, val=val)], define max distance,A, sum plevels:
            Xn = max(Xn, (node.x0 + node.xn) - X0)  # box xn = x0+xn
            Yn = max(Yn, (node.y0 + node.yn) - Y0)
            node_pplayers__ = [node.H[:-1],node.H][fd]  # plevels ( forks ( pplayers, skip last plevel if rng+
            sum_pH_(uH, node_pplayers__)
            new_lev = CpH(L=0,A=[0,0])
            link_ = [node.link_.Qm, node.link_.Qd][fd]  # fork link_
            # form quasi-gradient per node from variable-length links:
            for derG in link_:
                der_lev = [derG.mplevel,derG.dplevel][fd]
                sum_pH(new_lev,der_lev)
                new_lev.L+=1; new_lev.S+=derG.S; new_lev.A[0]+=derG.A[0]; new_lev.A[1]+=derG.A[1]
                new_lev.node_ += [derG.node0] if node is derG.node1 else [derG.node1]  # redundant to node.link_
            sum_pH(UH, uH)
            node.uH += [new_lev]
        new_Lev = CpH()
        for link in Glink_: sum_pH(new_Lev, [link.mplevel, link.dplevel][fd])

        Graph = Cgraph(cLev=new_Lev, uH=UH, wH=[[[],[],[],[]]], node_=graph.Q, A=[Xn*2,Yn*2], x0=X0,xn=Xn,y0=Y0,yn=Yn)
        sum_pH_(root.uH + [root.wH[0][fork]], UH + [new_Lev])  # sum combined Hs, u|w shift across levels

        Graph_ += [Graph]  # Cgraph, reduction: root fork += all link forks?
    return Graph_


def comp_pH(_pH, pH):  # recursive unpack plevels ( pplayer ( players ( ptuples -> ptuple:

    mpH, dpH = CpH(), CpH()  # new players in same top plevel?

    for i, (_spH, spH) in enumerate(zip(_pH.H, pH.H)):
        fork = pH.forks[i] if pH.forks else 0  # in plevels or players
        _fork = _pH.forks[i] if _pH.forks else 0
        if _fork == fork:
            if isinstance(_spH, Cptuple):
                mtuple, dtuple = comp_ptuple(_spH, spH, fork)
                mpH.H += [mtuple]; mpH.val += mtuple.val
                dpH.H += [dtuple]; dpH.val += dtuple.val

            elif isinstance(_spH, CpH):
                if spH.S:  # extuple is valid in graph: pplayer only?
                    comp_ext(_spH, spH, mpH, dpH)
                sub_mpH, sub_dpH = comp_pH(_spH, spH)
                mpH.H += [sub_mpH]; dpH.H += [sub_dpH]
                mpH.val += sub_mpH.val; dpH.val += sub_dpH.val

    return mpH, dpH


def comp_ext(_spH, spH, mpH, dpH):
    L, S, A = len(spH.node_), spH.S, spH.A
    _L, _S, _A = len(_spH.node_), _spH.S, _spH.A

    _sparsity = _S /(_L-1); sparsity = S /(L-1)  # average distance between connected nodes, single distance if derG
    dpH.S = _sparsity - sparsity; dpH.val += dpH.S
    mpH.S = min(_sparsity, sparsity); mpH.val += mpH.S
    if spH.L:  # dLs
        L = spH.L; _L = _spH.L
    dpH.L = _L - L; dpH.val += dpH.L
    mpH.L = ave_L - dpH.L; mpH.val += mpH.L

    if _A and A:  # axis: dy,dx only for derG or high-aspect Gs, both val *= aspect?
        if isinstance(_A, list):
            m, d = comp_angle(None, _A, A)
            mpH.A = m; dpH.A = d
        else:  # scalar mA or dA
            dpH.A = _A - A; mpH.A = min(_A, A)
    else:
        mpH.A = 1; dpH.A = 0  # no difference, matching low-aspect, only if both?
    mpH.val += mpH.A; dpH.val += dpH.A

# old
def sub_recursion_g(graph_, Sval, fseg, fd):  # rng+: extend G_ per graph, der+: replace G_ with derG_

    Mplevel, Dplevel = CpH(), CpH()  # per sub+
    for graph in graph_:
        mplevel, dplevel = CpH(), CpH()  # per graph
        node_ = graph.plevels.H[-1].node_  # get only latest pplayers?
        if graph.plevels.val > G_aves[fd] and len(node_) > ave_nsub:

            sub_mgraph_, sub_dgraph_ = form_graph_(graph)  # cross-comp and clustering cycle
            # rng+:
            # or get val from plevels.H[-1]?
            Rval = sum([sub_mgraph.plevels.val for sub_mgraph in sub_mgraph_])
            if Rval > ave_sub * graph.rdn:  # >cost of call:
                sub_mmplevel, sub_dmplevel = sub_recursion_g(sub_mgraph_, Rval, fseg=fseg, fd=0)
                Rval += sub_mmplevel.val; Rval += sub_dmplevel.val
                sum_pH(Mplevel, sub_mmplevel); sum_pH(Mplevel, sub_dmplevel)  # sum both?
                sum_pH(mplevel, sub_mmplevel); sum_pH(mplevel, sub_dmplevel)

            # der+:
            Dval = sum([sub_dgraph.plevels.val for sub_dgraph in sub_dgraph_])
            if Dval > ave_sub * graph.rdn:
                sub_mdplevel, sub_ddplevel = sub_recursion_g(sub_dgraph_, Dval, fseg=fseg, fd=1)
                Dval += sub_mdplevel.val; Dval += sub_ddplevel.val
                sum_pH(Dplevel, sub_mdplevel); sum_pH(Dplevel, sub_ddplevel)  # sum both?
                sum_pH(dplevel, sub_mdplevel); sum_pH(dplevel, sub_ddplevel)  # sum both?

            graph.plevels.mpH.H += [mplevel]; graph.plevels.dpH.H += [dplevel]  # add new sub+ pplayers
            Sval += Rval + Dval  # do we still need Sval here?

    return Mplevel, Dplevel


def add_alt_graph_(graph_t):  # mgraph_, dgraph_
    '''
    Select high abs altVal overlapping graphs, to compute value borrowed from cis graph.
    This altVal is a deviation from ave borrow, which is already included in ave
    '''
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            for node in graph.plevels.H[-1].node_:
                for derG in node.link_.Q:  # contour if link.plevels.val < ave_Gm: link outside the graph
                    for G in [derG.node0, derG.node1]:  # both overlap: in-graph nodes, and contour: not in-graph nodes
                        alt_graph = G.roott[1-fd]
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, CpH):  # not proto-graph or removed
                            graph.alt_graph_ += [alt_graph]
                            alt_graph.alt_graph_ += [graph]
                            # bilateral assign
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            if graph.alt_graph_:
                graph.alt_plevels = CpH()  # players if fsub? der+: plevels[-1] += player, rng+: players[-1] = player?
                for alt_graph in graph.alt_graph_:
                    sum_pH(graph.alt_plevels, alt_graph.plevels)  # accum alt_graph_ params
                    graph.alt_rdn += len(set(graph.plevels.H[-1].node_).intersection(alt_graph.plevels.H[-1].node_))  # overlap


def sum_pH_(PHt, pHt, fneg=0):
    for PH, pH in zip_longest(PHt, pHt, fillvalue=[]):
        if pH:
            if PH:
                if isinstance(pH, list): sum_pH_(PH, pH, fneg)
                else:                    sum_pH(PH, pH, fneg)
            else: PHt += [deepcopy(pH)]

def sum_pH(PH, pH, fneg=0):  # recursive unpack plevels ( pplayers ( players ( ptuples, no accum across fd: matched in comp_pH

    if isinstance(pH, list):
        if isinstance(PH, CpH):  # convert PH to list if pH is list
            PH = []
        sum_pH_(PH, pH)
    else:
        if pH.node_:  # valid extuple
            PH.node_ += [node for node in pH.node_ if node not in PH.node_]
            if pH.L:
                if pH.L:
                    if PH.L: PH.L += pH.L
                    else:    PH.L = pH.L
            PH.S += pH.S
            if PH.A:
                if isinstance(PH.A, list):
                    PH.A[0] += pH.A[0]; PH.A[1] += pH.A[1]
                else:
                    PH.A += pH.A
            else: PH.A = copy(pH.A)

        for SpH, spH in zip_longest(PH.H, pH.H, fillvalue=None):  # assume same forks
            if SpH:
                if spH:  # pH.H may be shorter than PH.H
                    if isinstance(spH, Cptuple):  # PH is ptuples, SpH_4 is ptuple
                        sum_ptuple(SpH, spH, fneg=fneg)
                    else:  # PH is players, H is ptuples
                        sum_pH(SpH, spH, fneg=fneg)

            else:  # PH.H is shorter than pH.H, extend it:
                PH.H += [deepcopy(spH)]
        PH.val += pH.val
    return PH