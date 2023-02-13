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
Agg+ starts with cross-comp of bottom nodes: PPs, adding uH tree levels that fork upward.
Sub+ cross-comps node_ in select Gs, adding wH tree levels that fork downward.
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

class Clink_(ClusterStructure):
    Q = list
    val = float
    Qm = list
    mval = float
    Qd = list
    dval = float

class CpH(ClusterStructure):  # hierarchy of params + associated vars in pplayers | players | ptuples

    H = list  # hierarchy of pplayers | players | ptuples, can be Q: sequence
    val = float
    rdn = lambda: 1  # for all Qs?
    rng = lambda: 1
    fds = list  # m|d in pplayers,players,ptuples, m|d|None in levs?
    # in xpplayers, each m|d:
    L = int
    S = int
    A = int

class Cgraph(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    link_ = lambda: Clink_()  # evaluated external links per node if G.root
    link_pplayers = lambda: CpH()  # sum link_ pplayers, if G.root
    node_ = list  # or concatenated sub-node_s in uH levs
    pplayers = lambda: CpH()  # sum node_ pplayers
    val = int  # of all params
    rdn = lambda: 1
    uH = list  # upper Lev += lev per agg+|sub+: up-forking root tree, CpH lev, H=forks: G/agg+ or pplayers/sub+
    wH = list  # lower Lev += node__.. feedback: down-forking node tree, same syntax?
    # collaterals:
    nval = int  # open links: base alt rep, separate for node_ and link_
    alt_graph_ = list  # contour + overlapping contrast graphs
    alt_Graph = None  # conditional, summed and concatenated params of alt_graph_
    # or L,S,A in pplayers: already CpH? if S:
    L = list  # der L, init None
    S = int  # sparsity: ave len link
    A = list  # area|axis: Dy,Dx, ini None
    x0 = float
    y0 = float  # center: box x0|y0 + L/2
    xn = float  # max distance from center
    yn = float
    # sub+,agg+ roots:
    G = lambda: None  # same-scope lower der|rng G.G.G.., for all nodes beyond PP
    root = lambda: None  # root graph, not summed as in uH[-1][fd]


class CderG(ClusterStructure):  # graph links, within root node_

    node0 = lambda: Cgraph()  # converted to list in recursion
    node1 = lambda: Cgraph()
    mplevel = lambda: CpH()  # in alt/contrast if open
    dplevel = lambda: CpH()
    # new lev, includes shared node_?:
    S = int  # sparsity: ave len link
    A = list  # area and axis: Dy,Dx


def agg_recursion(root, fseg):  # compositional recursion in root.PP_, pretty sure we still need fseg, process should be different

    root.wH.insert(0, CpH(H=[Cgraph(),Cgraph()]))  # to sum feedback from new graphs
    for G in root.node_:
        G.uH.insert(0, CpH(H=[Cgraph(),Cgraph()]))  # not for sub+: lower node =G.G?

    fds = root.pplayers.fds
    mgraph_, dgraph_ = form_graph_(root, fds, fsub=0)  # node.H cross-comp and graph clustering, comp frng pplayers

    for fd, graph_ in enumerate([mgraph_,dgraph_]):  # eval graphs for sub+ and agg+:
        val = sum([graph.val for graph in graph_])
        # intra-graph sub+ comp node:
        if val > ave_sub * root.rdn:  # same in blob, same base cost for both forks
            for graph in graph_: graph.rdn+=1  # estimate
            sub_recursion_g(graph_, fseg, fds + [fd])  # subdivide graph_ by der+|rng+
            # feedback per selected graph in sub_recursion_g
        # cross-graph agg+ comp graph:
        if val > G_aves[fd] * ave_agg * root.rdn and len(graph_) > ave_nsub:
            for graph in graph_: graph.rdn+=1   # estimate
            agg_recursion(root, fseg=fseg)
        else: feedback(root, graph_, fds)  # bottom-up feedback: root.wH[0][fd].node_ = graph_, etc, breadth-first


def form_graph_(root, fds, fsub): # form plevel in agg+ or sub-pplayer in sub+, G is node in GG graph

    G_ = root.node_
    comp_G_(G_, fsub=fsub)  # cross-comp all graph nodes in rng, graphs may be segs | fderGs, root G += link, link.node

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
            graph_ += [CpH(H=gnode_, val=val)]
        # reform graphs by node val:
        regraph_ = graph_reval(graph_, [ave_G for graph in graph_], fd)  # init reval_ to start
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum proto-graph node_ params in graph
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
        while graph.H:  # some links will be removed, graph may split into multiple regraphs, init each with graph.Q node:
            regraph = CpH()
            node = graph.H.pop()  # node_, not removed below
            val = [node.link_.mval, node.link_.dval][fd]  # in-graph links only
            if val > G_aves[fd]:  # else skip
                regraph.H = [node]; regraph.val = val  # init for each node, then add _nodes
                readd_node_layer(regraph, graph.H, node, fd)  # recursive depth-first regraph.Q+=[_node]
            reval = graph.val - regraph.val
            if regraph.val > ave_G:
                regraph_ += [regraph]; rreval_ += [reval]; Reval += reval
    if Reval > ave_G:
        regraph_ = graph_reval(regraph_, rreval_, fd)  # graph reval while min val reduction

    return regraph_

def readd_node_layer(regraph, graph_H, node, fd):  # recursive depth-first regraph.Q+=[_node]

    for link in [node.link_.Qm, node.link_.Qd][fd]:  # all positive
        _node = link.node1 if link.node0 is node else link.node0
        _val = [_node.link_.mval, _node.link_.dval][fd]
        if _val > G_aves[fd] and _node in graph_H:
            regraph.H += [_node]
            graph_H.remove(_node)
            regraph.val += _val
            readd_node_layer(regraph, graph_H, _node, fd)

def add_node_layer(gnode_, G_, G, fd, val):  # recursive depth-first gnode_+=[_G]

    for link in G.link_.Q:  # all positive
        _G = link.node1 if link.node0 is G else link.node0
        if _G in G_:  # _G is not removed in prior loop
            gnode_ += [_G]
            G_.remove(_G)
            val += [_G.link_.mval,_G.link_.dval][fd]
            val += add_node_layer(gnode_, G_, _G, fd, val)
    return val


def comp_G_(G_, pri_G_=None, f1Q=1, fsub=0):  # cross-comp Graphs if f1Q else G_s, or segs inside PP?

    if not f1Q: mxpplayers_, dxpplayers_ = [],[]

    for i, _iG in enumerate(G_ if f1Q else pri_G_):  # G_ is node_ of root graph
        for iG in G_[i+1:] if f1Q else G_:  # compare each G to other Gs in rng, bilateral link assign, val accum:
            # if the pair was compared in prior rng+:
            if iG in [node for link in _iG.link_.Q for node in [link.node0,link.node1]]:  # if f1Q? add frng to skip?
                continue
            dx = _iG.x0 - iG.x0; dy = _iG.y0 - iG.y0  # center x0,y0
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity, proximity = ave-distance
            if distance < ave_distance * ((_iG.val + iG.val) / (2*sum(G_aves))):
                # same for cis and alt Gs:
                for _G, G in ((_iG, iG), (_iG.alt_Graph, iG.alt_Graph)):
                    if not _G or not G:  # or G.val
                        continue
                    mxpplayers, dxpplayers = comp_G(_G, G, fsub)  # comp pplayers, uH,wH, LSA, node_,link_? no fork in comp_pH_?
                    derG = CderG(node0=_G,node1=G, mplevel=mxpplayers,dplevel=dxpplayers, S=distance, A=[dy,dx])
                    mval = mxpplayers.val; dval = dxpplayers.val
                    tval = mval + dval
                    _G.link_.Q += [derG]; _G.link_.val += tval  # val of combined-fork' +- links?
                    G.link_.Q += [derG]; G.link_.val += tval
                    if mval > ave_Gm:
                        _G.link_.Qm += [derG]; _G.link_.mval += mval  # no dval for Qm
                        G.link_.Qm += [derG]; G.link_.mval += mval
                    if dval > ave_Gd:
                        _G.link_.Qd += [derG]; _G.link_.dval += dval  # no mval for Qd
                        G.link_.Qd += [derG]; G.link_.dval += dval

                    if not f1Q:  # implicit cis, alt pair nesting in xpplayers_
                        mxpplayers_ += [mxpplayers]; dxpplayers_ += [dxpplayers]
    if not f1Q: return mxpplayers_, dxpplayers_


def comp_G(_G, G, fsub):  # comp-> MpH, DpH, H = xpplayers: implicitly nested lists of all lower xpplayers

    MpH, DpH = CpH(), CpH()
    Pplayers, link_, node_ = G.pplayers, G.link_.Q, G.node_
    _Pplayers,_link_,_node_ = _G.pplayers, G.link_.Q, G.node_

    # comp core:
    for _pplayers, pplayers in zip(_Pplayers.H, Pplayers.H):  # Pplayers are implicitly nested ders of all lower Pplayers
        mpplayers, dpplayers = comp_pH(_pplayers, pplayers)
        MpH.H+=[mpplayers]; DpH.H+=[dpplayers]; MpH.val+=mpplayers.val; DpH.val+=dpplayers.val

    comp_ext(_Pplayers, Pplayers, MpH, DpH)  # comp LSA, added to the whole nested new pplayers in comp_G_
    Val = (MpH.val + DpH.val) * (_G.val + G.val)
    if Val > ave_G:  # selective specification:

        if Val * (len(_link_)+len(link_)) > ave_G:
            mlink_, dlink_ = comp_derG_(_link_, link_)  # new function?
            MpH.val += sum([mlink.val for mlink in mlink_])
            DpH.val += sum([dlink.val for dlink in dlink_])

        if Val * (len(_node_)+len(node_)) > ave_G:
            mxpplayers_, dxpplayers_ = comp_G_(_node_, node_, f1Q=0, fsub=fsub)
            MpH.val += sum([mxpplayers.val for mxpplayers in mxpplayers_])
            DpH.val += sum([dxpplayers.val for dxpplayers in dxpplayers_])

        # comp context: summed roots and nodes, but wH is empty in top-down comp?
        for _forks, forks in zip(_G.uH, G.uH):
            for _g, _fd in zip(forks.H, forks.fds):
                for g, fd in zip(forks.H, forks.fds):
                    if _fd == fd:
                        mpH, dpH = comp_pH(_g.pplayers, g.pplayers)
                        sum_pH(MpH, mpH); sum_pH(DpH, dpH)

        # comp alts, val, rdn?
    return MpH, DpH

def comp_ext(_pplayers, pplayers, mpH, dpH):
    _L,_S,_A, L,S,A = _pplayers.L, _pplayers.S, _pplayers.A, pplayers.L, pplayers.S, pplayers.A

    _sparsity = _S /(_L-1); sparsity = S /(L-1)  # average distance between connected nodes, single distance if derG
    dpH.S = _sparsity - sparsity; dpH.val += dpH.S
    mpH.S = min(_sparsity, sparsity); mpH.val += mpH.S
    dpH.L = _L - L; dpH.val += dpH.L
    mpH.L = min(_L, L); mpH.val += mpH.L

    if _A and A:  # axis: dy,dx only for derG or high-aspect Gs, both val *= aspect?
        if isinstance(_A, list):
            m, d = comp_angle(None, _A, A)
            mpH.A = m; dpH.A = d
        else:  # scalar mA or dA
            dpH.A = _A - A; mpH.A = min(_A, A)
    else:
        mpH.A = 1; dpH.A = 0  # no difference, matching low-aspect, only if both?
    mpH.val += mpH.A; dpH.val += dpH.A

# very initial draft
def comp_derG_(_derG_, derG_):

    mlink_, dlink_ = [], []
    for _derG in _derG_:
        for derG in derG_:
            mmpH, dmpH = comp_pH(_derG.mplevel, derG.mplevel)
            mdpH, ddpH = comp_pH(_derG.dplevel, derG.dplevel)
            mlink_ += [mmpH, dmpH]; dlink_ += [mdpH, ddpH]

    return mlink_, dlink_

def comp_pH(_pH, pH):  # recursive unpack plevels ( pplayer ( players ( ptuples -> ptuple:

    mpH, dpH = CpH(), CpH()  # new players in same top plevel?

    for i, (_spH, spH) in enumerate(zip(_pH.H, pH.H)):
        fd = pH.fds[i] if pH.fds else 0  # in plevels or players
        _fd = _pH.fds[i] if _pH.fds else 0
        if _fd == fd:
            if isinstance(_spH, Cptuple):
                mtuple, dtuple = comp_ptuple(_spH, spH, fd)
                mpH.H += [mtuple]; mpH.val += mtuple.val
                dpH.H += [dtuple]; dpH.val += dtuple.val

            elif isinstance(_spH, CpH):
                sub_mpH, sub_dpH = comp_pH(_spH, spH)
                mpH.H += [sub_mpH]; dpH.H += [sub_dpH]
                mpH.val += sub_mpH.val; dpH.val += sub_dpH.val

    return mpH, dpH

def sum_G(G, g):

    # sum pplayers:
    fd = g.pplayers.fds[-1]
    if G.uH:
        sum_pH(G.uH[-1].H[fd].pplayers, g.pplayers)
    else:  # add lev
        if fd: G.uH += [CpH(H=[Cgraph(), Cgraph(pplayers=deepcopy(g.pplayers))])]
        else:  G.uH += [CpH(H=[Cgraph(pplayers=deepcopy(g.pplayers)), Cgraph()])]

    # sum uH:
    i=0; fds = g.pplayers.fds  # g.pplayers.fds include g.G.pplayers.fds
    # indices: i/lev in uH, j/G in lev.H, k/fd in fds, len H = len fds
    while g.G:
        i += 1; j = sum(fd * (2**k) for k, fd in enumerate(fds[i:]))
        sum_pH(G.uH[i].H[j].pplayers, g.G.pplayers)
        g = g.G
    for Lev, lev in zip_longest(G.uH, g.uH, fillvalue=[]):
        i += 1
        if Lev:
            j = sum(fd * (2**k) for k, fd in enumerate(fds[i:]))
            sum_pH(Lev.H[j].pplayers, lev.H[j].pplayers)  # lev is CpH, lev.H is graphs
        else:
            G.uH += [copy(lev)]
    '''
    fds->index examples:
    0,1: 0*2^0 + 1*2^1->2: 1st G in 2nd 2tuple:  0,1, (2),3; 4,5, 6,7;; 8,9, 10,11; 12,13, 14,15
    1,0,1: 1+4->5: 2nd G in 1st 2t in 2nd 4t:    0,1, 2,3; 4,(5), 6,7;; 8,9, 10,11; 12,13, 14,15
    1,0,1,1: 1+4+8->13: 2'G, 1'2t, 2'4t, 2'8t:   0,1, 2,3; 4,5, 6,7;; 8,9, 10,11; 12,(13), 14,15
    '''
    # wH is summed in feedback
    # ext params
    G.L += g.L
    G.S += g.S
    if isinstance(g.A, list):
        if g.A:
            if G.A:
                G.A[0] += g.A[0]; G.A[1] += g.A[1]
            else: G.A = copy(g.A)
    else: G.A += g.A
    G.val += g.val
    G.rdn += g.rdn
    G.nval += g.nval
    # not sure if we need below for coordinates
    G.x0 = min(G.x0, g.x0)
    G.y0 = min(G.y0, g.y0)
    G.xn = max(G.xn, g.xn)
    G.yn = max(G.yn, g.yn)

    # node and links
    for node in g.node_:
        if node not in G.node_:
            G.node_ += [node]
    for link in g.link_.Q:
        if link not in G.link_.Q:
            G.link_.Q += [link]
    # alts
    for alt_graph in g.alt_graph_:
        if alt_graph not in G.alt_graph:
            G.alt_graph_ += [alt_graph]
    if g.alt_Graph:
        if G.alt_Graph:
            sum_pH(G.alt_Graph, g.alt_Graph)
        else:
            G.alt_Graph = deepcopy(g.alt_graph)

def sum_pH_(PH_, pH_, fneg=0):
    for H, h in zip_longest(PH_, pH_, fillvalue=[]):  # each H is CpH
        if h:
            if H:
                for G, g in zip_longest(H.H, h.H, fillvalue=[]):  # each G is Cgraph
                    if g:
                        if G: sum_pH(G.pplayers, g.pplayers, fneg)
                        else: H.H += [deepcopy(g)]  # copy and pack single fork G
            else:
                PH_ += [deepcopy(h)]  # copy and pack CpH

def sum_pH(PH, pH, fneg=0):  # recursive unpack plevels ( pplayers ( players ( ptuples, no accum across fd: matched in comp_pH

    for SpH, spH, Fd, fd in zip_longest(PH.H, pH.H, PH.fds, pH.fds, fillvalue=None):  # assume same forks
        if spH:  # check spH first because no point in summing PH if pH is shorter and spH is None
            if SpH:
                if isinstance(spH, Cptuple):  # PH is ptuples, SpH_ is ptuple
                    sum_ptuple(SpH, spH, fneg=fneg)
                else:  # PH is players, H is ptuples
                    sum_pH(SpH, spH, fneg=fneg)
            else:
                PH.fds += [fd]
                PH.H += [deepcopy(spH)]

    PH.val += pH.val
    PH.rdn += pH.rdn

    PH.L += pH.L
    PH.S += pH.S
    if isinstance(pH.A, list):
        if pH.A:
            if PH.A:
                PH.A[0] += pH.A[0]; PH.A[1] += pH.A[1]
            else: PH.A = copy(pH.A)
    else: PH.A += pH.A

    return PH


def sum2graph_(graph_, fd):  # sum node and link params into graph, plevel in agg+ or player in sub+

    Graph_ = []  # Cgraphs
    for graph in graph_:  # CpHs
        if graph.val < ave_G:  # form graph if val>min only
            continue
        X0,Y0 = 0,0
        for G in graph.H:  # CpH
            X0 += G.x0; Y0 += G.y0
        L = len(graph.H); X0/=L; Y0/=L; Xn,Yn = 0,0
        Graph = Cgraph(x0=X0, xn=Xn, y0=Y0, yn=Yn)
        # form G, keep iG:
        node_,Link_= [],[]
        for iG in graph.H:
            Xn = max(Xn, (iG.x0 + iG.xn) - X0)  # box xn = x0+xn
            Yn = max(Yn, (iG.y0 + iG.yn) - Y0)
            sum_G(Graph, iG)  # sum(Graph.uH[-1][fd], iG.pplayers), higher levs += G.G.pplayers | G.uH, lower scope than iGraph
            link_ = [iG.link_.Qm, iG.link_.Qd][fd]
            Link_ = list(set(Link_ + link_))  # unique links in node_
            G = Cgraph(G=iG, root=Graph)
            # sum quasi-gradient of G-external links in link_pplayers: redundant to Graph.pplayers, less valuable, optional?:
            for derG in link_:
                sum_pH(G.link_pplayers, [derG.mplevel, derG.dplevel][fd])
                G.link_pplayers.S += derG.S; G.link_pplayers.A += sum(derG.A)  # derA = der_mA + der_dA?
            l=len(link_); G.link_pplayers.L=l; G.link_pplayers.S/=l
            node_ += [G]
        Graph.node_ = node_ # lower nodes = G.G..; Graph.root = iG.root
        for Link in Link_:  # sum unique links
            sum_pH(Graph.pplayers, [Link.mplevel,Link.dplevel][fd])
            Graph.pplayers.S += Link.S
            # sum A from link_, or S from node_?
        # LSA per whole nested pplayers:
        Graph.pplayers.A = [Xn*2,Yn*2]; L=len(Link_); Graph.pplayers.L=L; Graph.pplayers.S/=L
        Graph.val = Graph.pplayers.val \
                  + sum([lev.val for lev in Graph.uH]) / max(1, sum([lev.rdn for lev in Graph.uH])) \
                  + sum([lev.val for lev in Graph.wH]) / max(1, sum([lev.rdn for lev in Graph.wH])) # if val > alt_val: rdn += len_Q?

        Graph_ += [Graph]
    return Graph_

# draft
def sub_recursion_g(graph_, fseg, fd_, RVal=0, DVal=0):  # rng+: extend G_ per graph, der+: replace G_ with derG_

    for graph in graph_:
        node_ = graph.node_
        Gval = graph.pplayers.val
        if Gval > G_aves[fd_[-1]] and len(node_) > ave_nsub:

            graph.wH.insert(0, CpH(H=[Cgraph(), Cgraph()]))  # to sum new graphs, no uH in CpH?
            sub_mgraph_, sub_dgraph_ = form_graph_(graph, fd_, fsub=1)  # cross-comp and clustering
            # rng+:
            Rval = sum([sub_mgraph.pplayers.val for sub_mgraph in sub_mgraph_])
            if RVal + Rval > ave_sub * graph.rdn:  # >cost of call:
                rval, dval = sub_recursion_g(sub_mgraph_, fseg=fseg, fd_=fd_+[0], RVal=Rval, DVal=DVal)
                RVal += rval+dval
            # der+:
            Dval = sum([sub_dgraph.pplayers.val for sub_dgraph in sub_dgraph_])
            if DVal + Dval > ave_sub * graph.rdn:
                rval, dval = sub_recursion_g(sub_dgraph_, fseg=fseg, fd_=fd_+[1], RVal=Rval, DVal=DVal)
                Dval += rval+dval
            RVal += Rval
            DVal += Dval
        # unpack?:
        else: feedback(graph, node_, fd_)  # bottom-up feedback to append root.uH[-1], root.wH, breadth-first

    return RVal, DVal  # or SVal= RVal+DVal, separate for each fork of sub+?

# in sub+ and agg+?
def feedback(graph, node_, fd_):  # bottom-up feedback to append root.uH[-1], root.wH, breadth-first

    for node in node_:
        if node.wH:
            sum_pH(graph.uH[0].H[fd_[-1]].pplayers, node.wH[0].H[fd_[-1]].pplayers)
            # the rest of node.wH maps to graph.wH:
            sum_pH_(graph.wH, node.wH[1:])

# old:
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