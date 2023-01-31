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

class Cgraph(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    root = object  # root graph, root fork is local?
    G = object # mapping node in root?
    uH = list  # up H: higher Levs, no feedback, uH[i] is CpH, uH[-1] is current Lev
    wH = list  # down H: lower Levs of feedback, wH[i] is forks in 4^i fork tree
    # if higher CpH: 4forks if agg+, else []
    ufork__ = list  # taken fork_ per Lev, with one fork per tree layer
    wforkn_ = list  # nforks in tree lev, +/ fork-selective agg+
    val = int  # summed in uH+wH, or separate?
    nval = int  # of open links, alt_graph_?
    # G/uH, shared:
    link_ = lambda: Clink_()  # evaluated external links if node, alt_node if open
    node_ = list  # sub-node_ s concatenated within root node_
    rdn = int  # recursion count + irdn, for eval
    rng = lambda: 1  # not for alt_graphs
    alt_graph_ = list  # contour + overlapping contrast graphs
    # extuple if S:
    L = list  # der L, init empty
    S = int  # sparsity: ave len link
    A = list  # area and axis: Dy,Dx
    x0 = float
    y0 = float  # center: box x0|y0 + L/2
    xn = float  # max distance from center
    yn = float

class CpH(ClusterStructure):  # hierarchy of params + associated vars

    G = lambda: None  # root Cgraph in uH[-1], all params are shared by its uH, empty in uH[:-1]
    H = list  # pplayers | players | ptuples
    val = int
    nval = int  # of open links: alt_graph_?
    forks = list  # m|d|am|ad list in H=plevels|pplayers, m|d list in H=players|ptuples
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

    fork_ = root.ufork__[0]  # single-fork, always taken, H: plevels ( forks ( pplayers ( players ( ptuples
    root.wH = [[[],[],[],[]]]; root.wforkn_ = [1] # init, extend by feedback
    for G in root.node_:  # same for nodes:
        G.wH = [[[],[],[],[]]]; G.wforkn_ = [1]  # n forks = wforkn * 4

    mgraph_, dgraph_ = form_graph_(root, fork_)  # node.H cross-comp and graph clustering, comp frng pplayers

    for fd, graph_ in enumerate([mgraph_,dgraph_]):  # eval graphs for sub+ and agg+:
        val = sum([graph.val for graph in graph_])
        # intra-graph sub+ comp node:
        if val > ave_sub * root.rdn:  # same in blob, same base cost for both forks
            for graph in graph_: graph.rdn+=1  # estimate
            sub_recursion_g(graph_, val, fseg, fork_ + [fd])  # subdivide graph_ by der+|rng+
        # cross-graph agg+ comp graph:
        if val > G_aves[fd] * ave_agg * root.rdn and len(graph_) > ave_nsub:
            for graph in graph_: graph.rdn+=1   # estimate
            root.node_ = graph_  # update graph_? Else root's node_ stays the same throughout all the layers
            agg_recursion(root, fseg=fseg)
        # draft:
        else:  # bottom-up feedback to append graph wH, per fork termination?
            for graph in graph_:
                # initial draft:
                sum_pH(root.wH[1:], graph.wH, fork_)  # wH[1:]: higher wH[0] doesn't map to lower wH
            pass

def form_graph_(root, fork_): # form plevel in agg+ or sub-pplayer in sub+, G is node in GG graph

    G_ = root.node_
    comp_G_(G_)  # cross-comp all graph nodes in rng, graphs may be segs | fderGs, root G += link, link.node

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
            graph_[:] = sum2graph_(regraph_, root, fd, fork_)  # sum proto-graph node_ params in graph
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

def comp_G_(G_):  # cross-comp Graphs, or segs inside PP?

    for i, _G in enumerate(G_):  # G_ is node_ of root graph
        for G in G_[i+1:]:  # compare each G to other Gs in rng, bilateral link assign, val accum:
            # test if lev pair was compared in prior rng+, add frng to skip?:
            if G in [node for link in _G.link_.Q for node in [link.node0,link.node1]]:
                continue
            dx = _G.x0 - G.x0; dy = _G.y0 - G.y0  # center x0,y0
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity, proximity = ave-distance
            if distance < ave_distance * ((_G.val + G.val) / (2*sum(G_aves))):
                # comp pplayers_:
                mplevel, dplevel = comp_pH_(_G.uH, G.uH)  # compare taken forks only, no fork in comp_pH_?
                derG = CderG(node0=_G,node1=G, mplevel=mplevel,dplevel=dplevel, S=distance, A=[dy,dx])
                mplevel.G = derG; dplevel.G = derG  # to sum ext params
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


def sum2graph_(graph_, root, fd, fork_):  # sum node and link params into graph, plevel in agg+ or player in sub+

    Graph_ = []  # Cgraphs
    for graph in graph_:  # CQs
        if graph.val < ave_G:  # form graph if val>min only
            continue
        Glink_= []; X0,Y0 = 0,0
        # 1st pass: define center Y,X and Glink_:
        for G in graph.Q:
            Glink_ = list(set(Glink_ + [G.link_.Qm, G.link_.Qd][fd]))  # unique fork links in graph node_
            X0 += G.x0; Y0 += G.y0
        L = len(graph.Q); X0/=L; Y0/=L; Xn,Yn = 0,0
        UH = []
        node_ = []
        Graph = Cgraph( root=root, node_=node_, uH=UH, wH=[[]], # 4forks if agg+|sub+ only
                        ufork__=G.ufork__,wforkn_=[1])  # nodes have same ufork__ as graph, redundant?
        # 2nd pass: extend and sum nodes in graph:
        for G in graph.Q:  # CQ(Q=gnode_, val=val)], define new G and graph:
            Xn = max(Xn, (G.x0 + G.xn) - X0)  # box xn = x0+xn
            Yn = max(Yn, (G.y0 + G.yn) - Y0)
            sum_pH_(UH, G.uH)
            new_lev = CpH()  # add root: corresponding node in root graph?
            L=0; S=0; A=[0, 0]
            link_ = [G.link_.Qm, G.link_.Qd][fd]  # fork link_
            # form quasi-gradient per node from variable-length links:
            for derG in link_:
                der_lev = [derG.mplevel,derG.dplevel][fd]
                sum_pH(new_lev,der_lev)
                L+=1; S+=derG.S; A[0]+=derG.A[0]; A[1]+=derG.A[1]
            uH = deepcopy(G.uH); G.uH[-1].G=[]
            new_G = Cgraph( root=Graph, G=G, uH=uH+[new_lev], ufork__=G.ufork__+[fork_], node_ = copy(G.node_),
                            val=G.val+new_lev.val, L=L,S=S,A=A, x0=G.x0,xn=G.xn,y0=G.y0,yn=G.yn)
            new_lev.G = new_G  # G is immutable
            node_ += [new_G]
        new_Lev = CpH(G=Graph, val=graph.val, A=[Xn*2,Yn*2], x0=X0,xn=Xn,y0=Y0,yn=Yn)
        for link in Glink_: sum_pH(new_Lev, [link.mplevel, link.dplevel][fd])
        UH += [new_Lev]
        Graph.val = sum([lev.val for lev in UH])
        Graph_ += [Graph]

    return Graph_


def comp_pH_(_pH_, pH_):  # comp H-> nested MpH, DpH

    MpH, DpH = CpH(), CpH()  # lists of mpH,dpH, with implicit nesting

    for _pH, pH in zip(_pH_, pH_):
        # MpH and DpH doesn't have L,S,A, so i think we need to return L, S, A and pack it in derG later?
        if pH.G:  # extuple is valid in graph: pplayer only? (this should be here because PH is pplayers here)
            comp_ext(_pH, pH, MpH, DpH)

        mpH, dpH = comp_pH(_pH, pH)
        sum_pH(MpH, mpH)
        sum_pH(DpH, dpH)
    return MpH, DpH

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
                sub_mpH, sub_dpH = comp_pH(_spH, spH)
                mpH.H += [sub_mpH]; dpH.H += [sub_dpH]
                mpH.val += sub_mpH.val; dpH.val += sub_dpH.val

    return mpH, dpH


def comp_ext(_spH, spH, mpH, dpH):
    L, S, A = len(spH.G.node_), spH.G.S, spH.G.A
    _L, _S, _A = len(_spH.G.node_), _spH.G.S, _spH.G.A
    # below pending update
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

# rough redraft
def sub_recursion_g(graph_, fseg, fork_, RVal=0, DVal=0):  # rng+: extend G_ per graph, der+: replace G_ with derG_

    for graph in graph_:
        mplevel, dplevel = CpH(), CpH()  # per graph
        node_ = graph.node_
        if graph.uH[-1].val > G_aves[fork_[-1]] and len(node_) > ave_nsub:
            graph.wH = [[[],[],[],[]]]; graph.wforkn_ = [1]  # init, then extended through feedback, n forks = wforkn * 4
            for G in node_:
                G.wH = [[[],[],[],[]]]; G.wforkn_ = [1]  # same format

            sub_mgraph_, sub_dgraph_ = form_graph_(graph, fork_)  # cross-comp and clustering cycle
            # rng+:
            Rval = sum([sub_mgraph.uH[-1].val for sub_mgraph in sub_mgraph_])
            if RVal + Rval > ave_sub * graph.rdn:  # >cost of call:
                rval, dval = sub_recursion_g(sub_mgraph_, fseg=fseg, fork_=fork_+[0], RVal, DVal)
                RVal += rval+dval
            # der+:
            Dval = sum([sub_dgraph.uH[-1].val for sub_dgraph in sub_dgraph_])
            if DVal + Dval > ave_sub * graph.rdn:
                rval, dval = sub_recursion_g(sub_dgraph_, Dval, fseg=fseg, fork_=fork_+[1], RVal, DVal)
                Dval += rval+dval
            RVal += Rval
            DVal += Dval
        # rough draft:
        else: # bottom-up feedback to append root.wH, starting from wH[-1], breadth-first
            for node in node_:
                graph.uH[-1] += [node.wH[0]]
                for Lev, lev in zip_longest(graph.wH, node.wH[1:], fillvalue=[]):
                    for Fork, fork in zip_longest(Lev, lev, fillvalue=[]):
                        sum_pH(Fork, fork)  # add new sub+ pplayers

    return RVal, DVal  # or SVal= RVal+DVal, separate for each fork of sub+?


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


def sum_pH_(PH_, pH_, fneg=0):  # we need another function to sum graphs?

    for PH, pH in zip_longest(PH_, pH_, fillvalue=[]):
        if pH:
            if PH:
                if isinstance(pH, list): sum_pH_(PH, pH, fneg)
                else:                    sum_pH(PH, pH, fneg)
            else: PH_ += [deepcopy(pH)]

def sum_pH(PH, pH, fneg=0):  # recursive unpack plevels ( pplayers ( players ( ptuples, no accum across fd: matched in comp_pH

    if isinstance(pH, list):
        if isinstance(PH, CpH):  # convert PH to list if pH is list
            PH = []
        sum_pH_(PH, pH)
    else:

        if pH.G and PH.G:  # valid extuple
            if not isinstance(pH.G, CderG) and pH.G.L:  # derG doesn't have L
                if pH.G.L:
                    if PH.G.L: PH.G.L += pH.G.L
                    else:      PH.G.L = pH.G.L
            PH.G.S += pH.G.S
            if PH.G.A:
                if isinstance(PH.G.A, list):
                    PH.G.A[0] += pH.G.A[0]; PH.G.A[1] += pH.G.A[1]
                else:
                    PH.G.A += pH.G.A
            else: PH.G.A = copy(pH.G.A)


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