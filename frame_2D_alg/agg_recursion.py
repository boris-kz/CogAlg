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
    L = list  # init der L
    S = float # sparsity: ave len link
    A = list  # area and axis: Dy,Dx
    mpH = object
    dpH = object

class Cgraph(CPP):  # graph or generic PP of any composition

    link_ = lambda: Clink_()  # evaluated external links (graph=node), replace alt_node if open, direct only
    plevels_t = lambda: CQ()  # 4 CpH|[]: mplevels, dplevels, alt_mplevels, alt_dplevels, each:
    # plevels( pplayers( players( ptuples, node_, mpH,dpH: dw append/ agg|sub+, uw replace/ sum nodes, der+/ new plevel
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
    mplevel_t = lambda: CQ()  # mmplevel, dpmlevel, alt_mmplevel, alt_dmplevel
    dplevel_t = lambda: CQ()


def agg_recursion(root, G_, fseg, fd=1):  # compositional recursion in root.PP_, pretty sure we still need fseg, process should be different

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
            agg_recursion(root, graph_, fseg=fseg, fd=fd)  # cross-comp graphs


def form_graph_(root, ifd): # form plevel in agg+ or player in sub+, G is node in GG graph; der+: comp_link if fderG, from sub+

    G_ = root.plevels_t[ifd].H[0].node_  # root node_ for both forks is in fd higher plevel
    comp_G_(G_, ifd)  # cross-comp all graphs within rng, graphs may be segs | fderGs, G.roott += link, link.node

    mnode_, dnode_ = [], []  # Gs with >0 +ve fork links:
    for G in G_:
        if G.link_.Qm: mnode_ += [G]  # all nodes with +ve links, not clustered in graphs yet
        if G.link_.Qd: dnode_ += [G]
    graph_t = []
    # not needed, always empty?:
    if mnode_: root.plevels_t[ifd].mpH = CpH()  # reset new Plevel only if mnode_ or dnode_ is not empty
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
                    [link.mplevel_t, link.dplevel_t][fd].Q[0].node_ + [link.mplevel_t, link.dplevel_t][fd].Q[1].node_]:
                # G,_G was compared in prior rng+, add frng to skip?
                continue
            dx = _G.x0 - G.x0; dy = _G.y0 - G.y0  # center x0,y0
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity
            # proximity = ave-distance
            if distance < ave_distance * (_G.plevels_t.val + G.plevels_t.val) / (2*sum(G_aves)):  # comb G val
                mplevel_t, dplevel_t = CQ(),CQ()
                for _plevels, plevels in zip([_G.plevels_t.Q, G.plevels_t.Q]):
                    if _plevels and plevels:
                        mplevel, dplevel = comp_pH(_plevels[:-1-fd], plevels[:-1-fd])  # plevels[:-1] if rng+
                        # optional comp_links, in|between nodes?
                        mplevel.node_,dplevel.node_ = [_G,G],[_G,G]  # double assign to sum_pH, only one added
                        mplevel.S,dplevel.S = distance,distance; mplevel.A,dplevel.A = [dy,dx],[dy,dx]
                        mplevel_t.Q += [mplevel]; mplevel_t.val += mplevel.val
                        dplevel_t.Q += [dplevel]; dplevel_t.val += dplevel.val
                    else:
                        mplevel_t.Q += []; dplevel_t.Q += []
                derG = CderG(node0=_G, node1=G, mplevel_t=mplevel_t, dplevel_t=dplevel_t)

                mval, dval = mplevel_t.val, dplevel_t.val
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
# sorry, not reviewed
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

# pending update
def sum2graph_(G_, fd):  # sum node and link params into graph, plevel in agg+ or player in sub+: if fderG, also for alts?

    graph_ = []
    graph = Cgraph()
    # G_ should be G__ for a list of graphs, add for G_ in G__?

    for G in G_:
        X0,Y0, Xn,Yn = 0,0,0,0
        node_, val = G.Q, G.val  # form graph if val>min?
        link_ = []
        for node in node_:
            link_ = list(set(link_ + G.link_))  # unique links in graph
            X0+=node.x0; Y0+=node.y0  # first pass defines center
        L = len(node_); X0/=L; Y0/=L

        for node in node_:  # define max distance,A, sum plevels:
            Xn = max(Xn,(node.x0+node.xn)-X0)  # box xn = x0+xn
            Yn = max(Yn,(node.y0+node.yn)-Y0)
            # += ini new plevel:
            node.plevels.H += [CpH(mpH=CpH(),dpH=CpH())]; node.plevels.fds += [fd]
            for derG in node.link_.Q:  # form quasi-gradient from links of variable length:
                der_plevel = [derG.mplevel, derG.dplevel][fd]
                sum_pH(node.plevels.H[-1], der_plevel)  # new plevel: node_+=[G|_G], S=link.L, preA: [Xn,Yn], sum *= dangle?
                node.plevels.val += der_plevel.val

            for Plevel, plevel in zip(graph.plevels, node.plevels[:-1]):  # skip last plevel: it's redundant between nodes
                sum_pH(Plevel, plevel)  # plevels ( pplayers ( players ( ptuples
            node.roott[fd] = graph

        # need to check indent, something's wrong, we need for G_ in G__?
        # draft:
        new_plevel = CpH
        for link in link_: sum_pH(new_plevel, [link.mplevel_t, link.dplevel_t][fd])  # no alts for now
        graph.x0=X0; graph.xn=Xn; graph.y0=Y0; graph.yn=Yn
        new_plevel.A = [Xn * 2, Yn * 2]
        graph.plevels_t[fd].H += [new_plevel]  # new_plevel is summed from unique links, not nodes
        graph.plevels_t[fd].val += new_plevel.val
        graph.plevels_t[fd].fds = copy(node.plevels.fds) + [fd]
        graph_ += [graph]

    return graph_

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