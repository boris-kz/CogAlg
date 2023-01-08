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

    H = list  # plevels | pplayers | players | ptuples
    fds = list  # (m|d)s, only in plevels and players, if fds[-1]: the nodes are links
    val = int
    nval = int  # of neg open links?
    node_ = list  # sub-node_ s concatenated within root node_
    L = list  # der L, init empty
    S = float # sparsity: ave len link
    A = list  # area and axis: Dy,Dx
    mpH = object
    dpH = object

class Cgraph(CPP):  # graph or generic PP of any composition

    link_ = lambda: Clink_()  # evaluated external links (graph=node), replace alt_node if open, direct only
    plevels_t = list  # 4 CpH|[]: mplevels, dplevels, alt_mplevels, alt_dplevels, each:
    # plevels, node_, mpH,dpH: dw append/ agg|sub+, uw replace/ sum nodes, der+/ new plevel
    alt_graph_ = list  # contour + overlapping contrast graphs
    alt_rdn = int  # node_, alt_node_s overlap
    x0 = float
    y0 = float  # center: box x0|y0 + L/2
    xn = float  # max distance from center
    yn = float
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs; no alt_rdn: valt representation in alt_PP_ valts?
    rng = lambda: 1  # not for alt_graphs
    roott = lambda: [None, None]  # higher-order segG or graphs of two forks

class CderG(ClusterStructure):  # graph links

    node0 = lambda: Cgraph()
    node1 = lambda: Cgraph()
    mplevel_t = list  # mmplevel, dpmlevel, alt_mmplevel, alt_dmplevel
    dplevel_t = list


def agg_recursion(root, fseg, fd=1):  # compositional recursion in root.PP_, pretty sure we still need fseg, process should be different

    mgraph_, dgraph_ = form_graph_(root, fd)  # PP cross-comp and clustering
    mval = sum([mgraph.plevels_t.val for mgraph in mgraph_])
    dval = sum([dgraph.plevels_t.val for dgraph in dgraph_])

    for fd, (graph_,val) in enumerate(zip([mgraph_,dgraph_],[mval,dval])):  # same graph_, val for sub+ and agg+
        # intra-graph sub+:
        if val > ave_sub * (root.rdn):  # same in blob, same base cost for both forks
            root.rdn+=1  # estimate
            sub_recursion_g(graph_, val, fseg, fd)  # subdivide graph_ by der+|rng+, accum val
        for graph in graph_:
            sum_pH([root.plevels.mpH, root.plevels.dpH][fd] , graph.plevels)
        # cross-graph agg+:
        if val > G_aves[fd] * ave_agg * (root.rdn) and len(graph_) > ave_nsub:
            root.rdn+=1  # estimate
            agg_recursion(root, fseg=fseg, fd=fd)  # cross-comp graphs


def form_graph_(root, ifd): # form plevel in agg+ or player in sub+, G is node in GG graph; der+: comp_link if fderG, from sub+

    G_ = root.plevels_t[ifd].H[0].node_  # root node_ for both forks is in fd higher plevel
    comp_G_(G_, ifd)  # cross-comp all graphs within rng, graphs may be segs | fderGs, G.roott += link, link.node

    mnode_, dnode_ = [], []  # Gs with >0 +ve fork links:
    for G in G_:
        if G.link_.Qm: mnode_ += [G]  # all nodes with +ve links, not clustered in graphs yet
        if G.link_.Qd: dnode_ += [G]
    graph_t = []
    if mnode_: root.plevels_t[ifd].mpH = CpH()  # if not empty?
    if dnode_: root.plevels_t[ifd].dpH = CpH()

    for fd, node_ in enumerate([mnode_, dnode_]):
        graph_ = []  # form graphs by link val:
        while node_:  # all Gs not removed in add_node_layer
            G = node_.pop()
            gnode_ = [G]
            val = add_node_layer(gnode_, node_, G, fd, val=0)  # recursive depth-first gnode_+=[_G]
            graph_ += [CQ(Q=gnode_, val=val)]
        # reform graphs by node val:
        regraph_ = graph_reval(graph_, [ave_G for graph in graph_], fd)  # init reval_ to start
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum proto-graph node_ params in graph
        graph_t += [graph_]
        # draft:
        Plevels = [root.plevels_t[ifd].mpH, root.plevels_t[ifd].dpH][fd]
        for graph in graph_:
            sum_pH(Plevels, graph.plevels)

    add_alt_graph_(graph_t)  # overlap+contour, cluster by common lender (cis graph), combined comp?
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
            regraph.val += _val; _node.roott[fd] = regraph
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

def comp_G_(G_, fd):  # cross-comp Gs (patterns of patterns): Gs, derGs, or segs inside PP, same process, no fderG?

    for i, _G in enumerate(G_):
        for G in G_[i+1:]:  # compare each G to other Gs in rng, bilateral link assign, val accum:
            if G in [node for link in _G.link_.Q for node in
                    [link.mplevel_t, link.dplevel_t][fd].H[0].node_ + [link.mplevel_t, link.dplevel_t][fd].H[1].node_]:
                # G,_G was compared in prior rng+, add frng to skip?
                continue
            dx = _G.x0 - G.x0; dy = _G.y0 - G.y0  # center x0,y0
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity
            # proximity = ave-distance
            if distance < ave_distance * (sum([plevels.val for plevels in _G.plevels_t]) + sum([plevels.val for plevels in G.plevels_t])) \
                    / (2*sum(G_aves)): # (G.val+_G.val)/2?
                mplevel_t, dplevel_t = [],[]
                for _plevels, plevels in zip(_G.plevels_t, G.plevels_t):
                    if _plevels and plevels:
                        mplevel, dplevel = comp_pH(_plevels, plevels) if fd else comp_pH(_plevels[:-1], plevels[:-1])  # replace last plevel
                        # optional comp_links, in|between nodes?
                        mplevel.node_, dplevel.node_ = [_G,G],[_G,G]  # double assign to sum_pH, only one is added
                        mplevel.S,dplevel.S = distance,distance; mplevel.A,dplevel.A = [dy,dx],[dy,dx]
                        mplevel_t += [mplevel]; dplevel_t += [dplevel]
                    else:
                        mplevel_t += []; dplevel_t += []
                derG = CderG(node0=_G, node1=G, mplevel_t=mplevel_t, dplevel_t=dplevel_t)

                mval, dval = sum([plevel.val for plevel in mplevel_t]), sum([plevel.val for plevel in dplevel_t])
                tval = mval + dval
                _G.link_.Q += [derG]; _G.link_.val += tval  # val of combined-fork' +- links?
                G.link_.Q += [derG]; G.link_.val += tval
                if mval > ave_Gm:
                    _G.link_.Qm += [derG]; _G.link_.mval += mval  # no dval for Qm
                    G.link_.Qm += [derG]; G.link_.mval += mval
                if dval > ave_Gd:
                    _G.link_.Qd += [derG]; _G.link_.dval += dval  # no mval for Qd
                    G.link_.Qd += [derG]; G.link_.dval += dval


def comp_pH(_pH, pH):  # recursive unpack plevels ( pplayer ( players ( ptuples -> ptuple:

    mpH, dpH = CpH(), CpH()  # new players in same top plevel?
    pri_fd = 0
    for i, (_spH, spH) in enumerate(zip(_pH.H, pH.H)):

        fd = pH.fds[i] if len(pH.fds) else 0  # in plevels or players
        _fd = _pH.fds[i] if len(_pH.fds) else 0
        if _fd == fd:
            if fd: pri_fd = 1  # all scalars
            if isinstance(spH, Cptuple):
                mtuple, dtuple = comp_ptuple(_spH, spH, pri_fd)
                mpH.H += [mtuple]; mpH.val += mtuple.val
                dpH.H += [dtuple]; dpH.val += dtuple.val
            else:
                if spH.node_:  # extuple is valid, in pplayers
                    comp_ext(_spH, spH, mpH, dpH)
                sub_mpH, sub_dpH = comp_pH(_spH, spH)
                if isinstance(spH.H[0], CpH):
                    mpH.H += [sub_mpH]; dpH.H += [sub_dpH]
                else:
                    # spH.H is ptuples, combine der_ptuples into single der_layer:
                    mpH.H += sub_mpH.H; dpH.H += sub_dpH.H
                mpH.val += sub_mpH.val; dpH.val += sub_dpH.val
        else:
            break
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


# tentative update
def sub_recursion_g(graph_, Sval, fseg, fd):  # rng+: extend G_ per graph, der+: replace G_ with derG_

    Mplevel, Dplevel = CpH(), CpH()  # per sub+
    for graph in graph_:
        mplevel, dplevel = CpH(), CpH()  # per graph
        node_ = graph.plevels.H[-1].node_  # get only latest pplayers?
        if graph.plevels.val > G_aves[fd] and len(node_) > ave_nsub:

            sub_mgraph_, sub_dgraph_ = form_graph_(graph, fd)  # cross-comp and clustering cycle
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


def sum2graph_(graph_, fd):  # sum node and link params into graph, plevel in agg+ or player in sub+

    Graph_ = []  # Cgraphs
    for graph in graph_:
        if graph.val < ave_G:  # form graph if val>min only
            continue
        Glink_= []; X0,Y0 = 0,0
        for node in graph.Q:  # first pass defines center Y,X and Glink_:
            Glink_ = list(set(Glink_ + node.link_.Q))  # unique links in graph nodes
            # or node.link_.Qd if fd else node.link_.Qm?
            X0 += node.x0; Y0 += node.y0
        L = len(graph.Q); X0/=L; Y0/=L; Xn,Yn = 0,0
        # sum m|dplevel_t:
        new_Plevel_t = [CpH,CpH,CpH,CpH]  # may be in links, same val for alts?
        for link in Glink_:
            for new_Plevel, der_plevel in zip(new_Plevel_t, [link.mplevel_t, link.dplevel_t][fd]):
                sum_pH(new_Plevel, der_plevel)
        Graph = Cgraph(plevels_t=[[],[],[],[]]); Graph.plevels_t[fd] = CpH
        Plevels = Graph.plevels_t[fd]

        for node in graph.Q:  # CQ(Q=gnode_, val=val)], define max distance,A, sum plevels:
            Xn = max(Xn,(node.x0+node.xn)-X0)  # box xn = x0+xn
            Yn = max(Yn,(node.y0+node.yn)-Y0)
            plevels = node.plevels_t[fd]
            new_plevel_t = [CpH,CpH,CpH,CpH]; new_val = 0  # may be in links, same val for alts?
            # form quasi-gradient per node from variable-length links:
            for derG in node.link_.Q:
                der_plevel_t = [derG.mplevel, derG.dplevel][fd]
                for new_plevel, der_plevel in zip(new_plevel_t, der_plevel_t):
                    sum_pH(new_plevel, der_plevel)
                    new_val += der_plevel.val
            if fd: plevels.val+=new_val; plevels.fds+=[fd]; plevels.H += [new_plevel_t]  # append new plevel
            else:  plevels.val+=new_val-plevels.H[-1].val; plevels.fds[-1]=fd; plevels.H[-1] = new_plevel_t  # replace last plevel
            # each plevel is plevel_t: 4 pplayers, not 1?
            for Plevel, plevel in zip(Plevels, plevels[:-1]):  # skip last plevel, redundant between nodes
                sum_pH(Plevel, plevel); Plevels.val += plevel.val
            node.roott[fd] = Graph

        Graph.x0=X0; Graph.xn=Xn; Graph.y0=Y0; Graph.yn=Yn
        new_Plevel.A = [Xn * 2, Yn * 2]
        Plevels.H += [new_Plevel]  # summed from unique links, not nodes
        Plevels.val += new_Plevel.val
        Plevels.fds = copy(plevels.fds) + [fd]
        Graph_ += [Graph]  # Cgraph

    return Graph_


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
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, Cgraph):  # not proto-graph or removed
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


def sum_pH(PH, pH, fneg=0):  # recursive unpack plevels ( pplayers ( players ( ptuples, no accum across fd: matched in comp_pH

    if pH.node_:  # valid extuple
        PH.node_ = list(set(PH.node_+pH.node_))
        if pH.L: PH.L += pH.L  # not sure
        PH.S += pH.S
        if PH.A:
            if isinstance(PH.A, list):
                PH.A[0] += pH.A[0]; PH.A[1] += pH.A[1]
            else:
                PH.A += pH.A
        else: PH.A = copy(pH.A)

    for SpH, spH in zip_longest(PH.H, pH.H, fillvalue=None):  # assume same fds
        if spH:
            if SpH:
                if isinstance(SpH, Cptuple):
                    sum_ptuple(SpH, spH, fneg=fneg)
                else:
                    sum_pH(SpH, spH, fneg=0)  # unpack sub-hierarchy, recursively
            elif not fneg:
                PH.H.append(spH)  # new Sub_pH
    PH.val += pH.val