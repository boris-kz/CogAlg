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
Right now clustering criterion is single m|d, so we have one root per fork. But that m|d is summed across m|d per variable in Gs.
It should be summing only above-ave m|ds, which will be from different sets of compared vars, for different _G,G pairs. 
-
We obviously have different concepts that may include same components. What makes them different is the subset of component'params 
that matches within a cluster. For example, the same objects can be classified / clustered by size, density, color, stability, etc.
'''
# aves defined for rdn+1:
aveG = 6  # fixed costs per G
aveGm = 5  # for inclusion in graph
aveGd = 4
G_aves = [aveGm, aveGd]
ave_med = 3  # call cluster_node_layer
ave_rng = 3  # rng per combined val
ave_ext = 5  # to eval comp_inder_
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

    H = list  # hierarchy of pplayers | players | ptuples, can be sequence
    # convert pplayers and players to CpH?
    val = float
    rdn = lambda: 1  # for all Qs?
    rng = lambda: 1
    fds = list  # m|d in pplayers,players,ptuples, m|d|None in levs?
    nval = int  # of open links: base alt rep
    # in xpplayers and derGs, each m|d, or pack in H?
    L = list  # der L, init None
    S = int  # sparsity: ave len link
    A = list  # area|axis: Dy,Dx, ini None


class Cgraph(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    G = lambda: None  # same-scope lower-der|rng G.G.G., or [G0,G1] in derG, None in PP
    root = lambda: None  # root graph or inder_ G, element of ex.H[-1][fd]
    # uH: up-forking Levs if mult roots, not implemented yet
    inder_ = list  # inder_ ) node_) H: contents, Lev+= node tree slice: feedback, Lev/agg+, lev/sub+?
    node_ = list  # single-fork, conceptually H[0], concat sub-node_s in ex.H levs
    link_ = lambda: Clink_()  # temporary holder for der+ node_, then unique links within graph?
    fterm = lambda: 0  # G.node_ sub-comp was terminated
    H = list  # down-forking tree of Levs: slice of nodes
    val = int
    fds = list
    rdn = lambda: 1
    rng = lambda: 1
    nval = int  # of open links: base alt rep
    box = lambda: [0,0,0,0,0,0]  # ave y,x, y0,yn, x0,xn
    alt_graph_ = list  # contour + overlapping contrast graphs
    alt_Graph = None  # conditional, summed and concatenated params of alt_graph_


def agg_recursion(root, fseg):  # compositional recursion in root.PP_, pretty sure we still need fseg, process should be different

    mgraph_, dgraph_ = form_graph_(root, fsub=0)  # node.H cross-comp and graph clustering, comp frng pplayers

    for fd, graph_ in enumerate([mgraph_,dgraph_]):  # eval graphs for sub+ and agg+:
        val = sum([graph.val for graph in graph_])
        # intra-graph sub+ comp node:
        if val > ave_sub * root.rdn:  # same in blob, same base cost for both forks
            # estimate rdn, assign to the weaker per sub_graphs feedback:
            for graph in graph_: graph.rdn+=1
            sub_recursion_g(graph_, fseg, root.fds + [fd])  # divide graph_ in der+|rng+ sub_graphs
        else:
            root.fterm = 1; feedback(root)  # update root.root..H, breadth-first
        # cross-graph agg+ comp graph:
        if val > G_aves[fd] * ave_agg * root.rdn and len(graph_) > ave_nsub:
            for graph in graph_: graph.rdn+=1  # estimate
            agg_recursion(root, fseg=fseg)  # replaces root.H
        else:
            root.fterm = 1; feedback(root)  # update root.root..H, breadth-first


def form_graph_(root, fsub): # form inder_ in agg+ or sub-pplayer in sub+, G is node in GG graph

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
        regraph_ = graph_reval(graph_, [aveG for graph in graph_], fd)  # init reval_ to start
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd, fsub)  # sum proto-graph node_ params in graph
        graph_t += [graph_]

    # add_alt_graph_(graph_t)  # overlap+contour, cluster by common lender (cis graph), combined comp?
    return graph_t

def graph_reval(graph_, reval_, fd):  # recursive eval nodes for regraph, increasingly selective with reduced node.link_.val

    regraph_, rreval_ = [],[]
    Reval = 0
    while graph_:
        graph = graph_.pop()
        reval = reval_.pop()
        if reval < aveG:  # same graph, skip re-evaluation:
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
            if regraph.val > aveG:
                regraph_ += [regraph]; rreval_ += [reval]; Reval += reval
    if Reval > aveG:
        regraph_ = graph_reval(regraph_, rreval_, fd)  # graph reval while min val reduction

    return regraph_

def readd_node_layer(regraph, graph_H, node, fd):  # recursive depth-first regraph.Q+=[_node]

    for link in [node.link_.Qm, node.link_.Qd][fd]:  # all positive
        _node = link.G[1] if link.G[0] is node else link.G[0]
        _val = [_node.link_.mval, _node.link_.dval][fd]
        if _val > G_aves[fd] and _node in graph_H:
            regraph.H += [_node]
            graph_H.remove(_node)
            regraph.val += _val
            readd_node_layer(regraph, graph_H, _node, fd)

def add_node_layer(gnode_, G_, G, fd, val):  # recursive depth-first gnode_+=[_G]

    for link in G.link_.Q:  # all positive
        _G = link.G[1] if link.G[0] is G else link.G[0]
        if _G in G_:  # _G is not removed in prior loop
            gnode_ += [_G]
            G_.remove(_G)
            val += [_G.link_.mval,_G.link_.dval][fd]
            val += add_node_layer(gnode_, G_, _G, fd, val)
    return val


def comp_G_(G_, pri_G_=None, f1Q=1, fsub=0):  # cross-comp Graphs if f1Q, else G_s in comp_node_, or segs inside PP?

    if not f1Q: minder__,dinder__ = [],[]

    for i, _iG in enumerate(G_ if f1Q else pri_G_):  # G_ is node_ of root graph
        for iG in G_[i+1:] if f1Q else G_:  # compare each G to other Gs in rng, bilateral link assign, val accum:
            # if the pair was compared in prior rng+:
            if iG in [node for link in _iG.link_.Q for node in link.node_]:  # if f1Q? add frng to skip?
                continue
            dy = _iG.box[0]-iG.box[0]; dx = _iG.box[1]-iG.box[1]  # between center x0,y0
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity, proximity = ave-distance
            if distance < ave_distance * ((_iG.val + iG.val) / (2*sum(G_aves))):
                # same for cis and alt Gs:
                for _G, G in ((_iG, iG), (_iG.alt_Graph, iG.alt_Graph)):
                    if not _G or not G:  # or G.val
                        continue
                    minder_, dinder_, mval, dval, tval = comp_GQ(_G,G)  # comp_G while G.G, H/0G: GQ is one distributed node?
                    ext = [1,distance,[dy,dx]]
                    derG = Cgraph(G=[_G,G], inder_=[minder_+[[ext]], dinder_+[[ext]]], box=[])  # box is redundant to 2 nodes
                    # add links:
                    _G.link_.Q += [derG]; _G.link_.val += tval  # combined +-links val
                    G.link_.Q += [derG]; G.link_.val += tval
                    if mval > aveGm:
                        _G.link_.Qm += [derG]; _G.link_.mval += mval  # no dval for Qm
                        G.link_.Qm += [derG]; G.link_.mval += mval
                    if dval > aveGd:
                        _G.link_.Qd += [derG]; _G.link_.dval += dval  # no mval for Qd
                        G.link_.Qd += [derG]; G.link_.dval += dval

                    if not f1Q:  # comp G_s
                        minder__+= minder_; dinder__+= dinder_
                # implicit cis, alt pair nesting in minder_, dinder_
    if not f1Q:
        return minder__, dinder__  # else no return, packed in links


def comp_GQ(_G, G):  # compare lower-derivation G.G.s, pack results in minder__,dinder__

    minder__,dinder__ = [],[]; Mval,Dval = 0,0; Mrdn,Drdn = 1,1
    Tval= aveG+1
    while (_G and G) and Tval > aveG:  # same-scope if sub+, no agg+ G.G

        minder_, dinder_, mval, dval, mrdn, drdn = comp_G(_G, G)
        minder__+=minder_; dinder__+=dinder_; Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn  # rdn+=1: to inder_?
        _G = _G.G
        G = G.G
        Tval = (Mval+Dval) / (Mrdn+Drdn)

    return minder__, dinder__, Mval, Dval, Tval  # ext added in comp_G_, not within GQ

def comp_G(_G, G):  # in GQ

    minder_,dinder_ = [],[]  # ders of implicitly nested list of pplayers in inder_
    Mval, Dval = 0,0
    Mrdn, Drdn = 1,1
    if _G.box: _inder_, inder_ = _G.inder_, G.inder_
    else:
        _fd = _G.root.fds[-1] if _G.root.fds else 0; fd = G.root.fds[-1] if G.root.fds else 0
        _inder_, inder_ = _G.inder_[_fd], G.inder_[fd]  # derG if comp node_?

    minder_,dinder_, Mval,Dval, Mrdn,Drdn = comp_inder_(_inder_,inder_, minder_,dinder_, Mval,Dval, Mrdn,Drdn)
    # spec:
    _node_, node_ = _G.node_, G.node_  # link_ if fd, sub_node should be empty
    if (Mval+Dval)* _G.val*G.val * len(_node_)*len(node_) > aveG:  # / rdn?

        sub_minder_,sub_dinder_ = comp_G_(_node_, node_, f1Q=0)
        Mval += sum([mxpplayers.val for mxpplayers in sub_minder_])  # + rdn?
        Dval += sum([dxpplayers.val for dxpplayers in sub_dinder_])
        # pack m|dnode_ in m|dinder_: implicit?
    else: _G.fterm=1  # no G.fterm=1: it has it's own specification?
    '''
    comp alts,val,rdn?
    comp H in comp_G_?
    select >ave m|d vars only: addressable salient mset / dset in inder_? 
    cluster per var or set if recurring across root: type eval if root M|D?
    '''
    return minder_, dinder_, Mval, Dval, Mrdn, Drdn


def comp_inder_(_inder_, inder_, minder_,dinder_, Mval,Dval, Mrdn,Drdn):

    i=0; end=1; Tval = aveG+1; elev=0
    while end <= min(len(_inder_),len(inder_)) and Tval > aveG:

        _Lev, Lev = _inder_[i:end], inder_[i:end]  # each Lev of implicit nesting is inder_,ext formed by comp_G
        for _der,der in zip(_Lev,Lev):
            if der:
                if isinstance(der,CpH):  # pplayers, incr implicit nesting in m|dpplayers:
                    mpplayers, dpplayers = comp_pH(_der, der)
                    minder_ += [mpplayers]; Mval += mpplayers.val; Mrdn += mpplayers.rdn  # add rdn in form_?
                    dinder_ += [dpplayers]; Dval += dpplayers.val; Drdn += dpplayers.rdn
                else:
                    mext2, dext2 = [],[]
                    for _ext, ext in _der, der:  # list [node_ext, graph_ext], both are full|empty per der?
                        mext, dext = comp_ext(_ext,ext)
                        mext2+=[mext]; dext2+=[dext]; Mval+=sum(mext); Dval+=sum(dext)
                    minder_ += [mext2]; dinder_ += [dext2]
            else:
                minder_+=[[]]; dinder_+=[[]]
        Tval = (Mval+Dval) / (Mrdn+Drdn)  # eval if looping Levs
        i = end
        end = (end*2) + 1*elev
        elev = 1
    '''
    lenLev = (end*2)+1: 1, 1+1, 3+1, 7+1, 15+1.: +[der_ext, agg_ext] per G in GQ, levs | Levs? same fds till += [fd]?
    Lev1: pps: 1 pplayers  # 0der
    Lev2: pps,ext: lenLev = 2   
    Lev3: pps; pps,ext; ext: lenLev = 4
    Lev4: pps; pps,ext; pps,pps,ext,ext; ext: lenLev = 8
    '''
    return minder_,dinder_, Mval,Dval, Mrdn,Drdn

def comp_ext(_ext, ext):
    _L,_S,_A = _ext; L,S,A = ext

    dS = _S - S; mS = min(_S, S)  # average distance between connected nodes, single distance if derG
    dL = _L - L; mL = min(_L, L)
    if _A and A:  # axis: dy,dx only for derG or high-aspect Gs, both val *= aspect?
        if isinstance(_A, list):
            mA, dA = comp_angle(None, _A, A)
        else:  # scalar mA or dA
            dA = _A - A; mA = min(_A, A)
    else:
        mA,dA = 0,0
    return (mL,mS,mA), (dL,dS,dA)

def comp_pH(_pH, pH):  # recursive unpack inder_s ( pplayer ( players ( ptuples -> ptuple:

    mpH, dpH = CpH(), CpH()  # new players in same top inder_?

    for i, (_spH, spH) in enumerate(zip(_pH.H, pH.H)):  # s = sub
        fd = pH.fds[i] if pH.fds else 0  # in inder_s or players
        _fd = _pH.fds[i] if _pH.fds else 0
        if _fd == fd:
            if isinstance(_spH, Cptuple):
                mtuple, dtuple = comp_ptuple(_spH, spH, fd)
                mpH.H += [mtuple]; mpH.val += mtuple.val; mpH.fds += [0]  # mpH.rdn += mtuple.rdn?
                dpH.H += [dtuple]; dpH.val += dtuple.val; dpH.fds += [1]  # dpH.rdn += dtuple.rdn

            elif isinstance(_spH, CpH):
                smpH, sdpH = comp_pH(_spH, spH)
                mpH.H += [smpH]; mpH.val += smpH.val; mpH.rdn += smpH.rdn; mpH.fds += [smpH.fds]  # or 0 | fd?
                dpH.H += [sdpH]; dpH.val += sdpH.val; dpH.rdn += sdpH.rdn; dpH.fds += [sdpH.fds]

    return mpH, dpH


def sum2graph_(graph_, fd, fsub=0):  # sum node and link params into graph, inder_ in agg+ or player in sub+

    Graph_ = []  # Cgraphs
    for graph in graph_:  # CpHs
        if graph.val < aveG:  # form graph if val>min only
            continue
        Graph = Cgraph(fds=copy(graph.H[0].fds)+[fd])  # incr der
        ''' if mult roots: 
        sum_inder_(Graph.uH[0][fd].inder_,root.inder_) or sum_G(Graph.uH[0][fd],root)? init if empty
        sum_H(Graph.uH[1:], root.uH)  # root of Graph, init if empty
        '''
        node_,Link_ = [],[]  # form G, keep iG:
        for iG in graph.H:
            sum_G(Graph, iG, fmerge=0)  # local subset of lower Gs in new graph
            link_ = [iG.link_.Qm, iG.link_.Qd][fd]  # mlink_,dlink_
            Link_ = list(set(Link_ + link_))  # unique links in node_
            G = Cgraph(fds=copy(iG.fds)+[fd], root=Graph, node_=link_, box=copy(iG.box))  # no sub_nodes if fder, remove if <ave?
            for derG in link_:
                sum_box(G.box, derG.G[0].box if derG.G[1] is iG else derG.G[1].box)
                sum_inder_(G.inder_, derG.inder_[fd])  # two-fork derGs are not modified
            add_ext(G.box, len(link_), G.inder_[-1])  # composed node ext, not in derG.inder_
            # if mult roots: sum_H(G.uH[1:], Graph.uH)
            node_ += [G]
        Graph.root = iG.root  # same root, lower derivation is higher composition
        Graph.node_ = node_  # G| G.G| G.G.G..
        for derG in Link_:  # sum unique links, not box
            sum_inder_(Graph.inder_, derG.inder_[fd])
        Ext = [0,0,[0,0]]
        Ext = [sum_ext(Ext, G.inder_[-1][1]) for G in node_]  # add composed node Ext
        add_ext(Graph.box, len(node_), Ext) # add composed graph Ext
        Graph.inder_ += [Ext]  # [node Ext, graph Ext]
        # if Graph.uH: Graph.val += sum([lev.val for lev in Graph.uH]) / sum([lev.rdn for lev in Graph.uH])  # if val>alt_val: rdn+=len_Q?
        Graph_ += [Graph]

    return Graph_

def sum_G(G, g, fmerge=0):  # g is a node in G.node_

    sum_inder_(G.inder_, g.inder_)  # direct node representation
    # if g.uH: sum_H(G.uH, g.uH[1:])  # sum g->G
    # if g.H: sum_H(G.H[1:], g.H)  # not used yet
    G.val += g.val; G.rdn += g.rdn; G.nval += g.nval
    sum_box(G.box, g.box)
    if fmerge:
        for node in g.node_:
            if node not in G.node_: G.node_ += [node]
        for link in g.Link_.Q:  # redundant?
            if link not in G.Link_.Q: G.Link_.Q += [link]
        for alt_graph in g.alt_graph_:
            if alt_graph not in G.alt_graph: G.alt_graph_ += [alt_graph]
        if g.alt_Graph:
            if G.alt_Graph: sum_G(G.alt_Graph, g.alt_Graph)
            else:           G.alt_Graph = deepcopy(g.alt_graph)
    else: G.node_ += [g]

def sum_inder_(Inder_, inder_, fext=1):

    for i, (Inder, inder) in enumerate(zip_longest(Inder_, inder_, fillvalue=None)):
        if inder is not None:
            if Inder:
                if inder:
                    if isinstance(inder, CpH): sum_pH(Inder,inder)
                    else:
                        for Ext,ext in zip(Inder,inder):  # pair
                            sum_ext(Ext,ext)
                else: Inder_.insert(i,deepcopy(inder))  # for different-length Inder_, inder_

            elif Inder is None: Inder_ += [deepcopy(inder)]
            else:               Inder_[i] = deepcopy(inder)

def add_ext(box, L, ext_):  # add ext per composition level
    y,x, y0,yn, x0,xn = box
    dY = yn-y0; dX = xn-x0
    box[:2] = y/L, x/L  # norm to ave
    ext_ += [[L, L/ dY*dX, [dY,dX]]]  # composed L,S,A, norm S = nodes per area

def sum_ext(Ext, ext):
    for i, param in enumerate(ext):
        if i<2:  # L,S
            Ext[i] += param
        else:  # angle
            if isinstance(Ext[i], list):
                Ext[i][0] += param[0]; Ext[i][1] += param[1]
            else:
                Ext[i] += param

def sum_box(Box, box):
    Y, X, Y0, Yn, X0, Xn = Box;  y, x, y0, yn, x0, xn = box
    Box[:] = [Y + y, X + x, min(X0, x0), max(Xn, xn), min(Y0, y0), max(Yn, yn)]

def sum_H(H, h):  # add g.H to G.H, no eval but possible remove if weak?

    for i, (Lev, lev) in enumerate(zip_longest(H, h, fillvalue=[])):  # root.ex.H maps to node.ex.H[1:]
        if lev:
            if not Lev:  # init:
                Lev = CpH(H=[[] for fork in range(2**(i+1))])
            for j, (Fork, fork) in enumerate(zip(Lev.H, lev.H)):
                if fork:
                    if not Fork: Lev.H[j] = Fork = Cgraph()
                    sum_G(Fork, fork)

def sum_pH_(PH_, pH_, fneg=0):
    for PH, pH in zip_longest(PH_, pH_, fillvalue=[]):  # each is CpH
        if pH:
            if PH:
                for Fork, fork in zip_longest(PH.H, pH.H, fillvalue=[]):
                    if fork:
                        if Fork:
                            if fork.inder_:
                                for (Pplayers, Expplayers),(pplayers, expplayers) in zip(Fork.inder_, fork.inder_):
                                    if Pplayers:   sum_pH(Pplayers, pplayers, fneg)
                                    else:          Fork.inder_ += [[deepcopy(pplayers),[]]]
                                    if Expplayers: sum_pH(Expplayers, expplayers, fneg)
                                    else:          Fork.inder_[-1][1] = deepcopy(expplayers)
                        else: PH.H += [deepcopy(fork)]
            else:
                PH_ += [deepcopy(pH)]  # CpH

def sum_pH(PH, pH, fneg=0):  # recursive unpack inder_s ( pplayers ( players ( ptuples, no accum across fd: matched in comp_pH

    for SpH, spH, Fd, fd in zip_longest(PH.H, pH.H, PH.fds, pH.fds, fillvalue=None):  # assume same forks
        if spH:
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
    if not PH.L: PH.L = pH.L  # PH.L is empty list by default
    else:        PH.L += pH.L
    PH.S += pH.S
    if isinstance(pH.A, list):
        if pH.A:
            if PH.A:
                PH.A[0] += pH.A[0]; PH.A[1] += pH.A[1]
            else: PH.A = copy(pH.A)
    else: PH.A += pH.A

    return PH

# draft
def sub_recursion_g(graph_, fseg, fds, RVal=0, DVal=0):  # rng+: extend G_ per graph, der+: replace G_ with derG_

    for graph in graph_:
        node_ = graph.node_
        if graph.val > G_aves[fds[-1]] and len(node_) > ave_nsub:

            sub_mgraph_, sub_dgraph_ = form_graph_(graph, fsub=1)  # cross-comp and clustering
            # rng+:
            Rval = sum([sub_mgraph.val for sub_mgraph in sub_mgraph_])
            # eval if Val>cost of call, else feedback per sub_mgraph?
            if RVal + Rval > ave_sub * graph.rdn:
                rval, dval = sub_recursion_g(sub_mgraph_, fseg=fseg, fds=fds+[0], RVal=Rval, DVal=DVal)
                RVal += rval+dval
            # der+:
            Dval = sum([sub_dgraph.val for sub_dgraph in sub_dgraph_])
            if DVal + Dval > ave_sub * graph.rdn:
                rval, dval = sub_recursion_g(sub_dgraph_, fseg=fseg, fds=fds+[1], RVal=Rval, DVal=DVal)
                Dval += rval+dval
            RVal += Rval
            DVal += Dval
        else:
            graph.fterm = 1  # forward is terminated, graph.node_ is empty or weak
            feedback(graph)  # bottom-up feedback to update root.H

    return RVal, DVal  # or SVal= RVal+DVal, separate for each fork of sub+?

def feedback(root):  # bottom-up update root.H, breadth-first

    fbV = aveG+1
    while root and fbV > aveG:
        if all([[node.fterm for node in root.node_]]):  # forward was terminated in all nodes
            root.fterm = 1
            fbval, fbrdn = 0,0
            for node in root.node_:
                # node.node_ may empty when node is converted graph
                if node.node_ and not node.node_[0].box:  # link_ feedback is redundant, params are already in node.inder_
                    continue
                for sub_node in node.node_:
                    fd = sub_node.fds[-1] if sub_node.fds else 0
                    if not root.H: root.H = [CpH(H=[[],[]])]  # append bottom-up
                    if not root.H[0].H[fd]: root.H[0].H[fd] = Cgraph()
                    # sum nodes in root, sub_nodes in root.H:
                    sum_inder_(root.H[0].H[fd].inder_, sub_node.inder_)
                    sum_H(root.H[1:], sub_node.H)  # sum_G(sub_node.H forks)?
            for Lev in root.H:
                fbval += Lev.val; fbrdn += Lev.rdn
            fbV = fbval/max(1, fbrdn)
            root = root.root

# old:
def add_alt_graph_(graph_t):  # mgraph_, dgraph_
    '''
    Select high abs altVal overlapping graphs, to compute value borrowed from cis graph.
    This altVal is a deviation from ave borrow, which is already included in ave
    '''
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            for node in graph.inder_s.H[-1].node_:
                for derG in node.link_.Q:  # contour if link.inder_s.val < aveGm: link outside the graph
                    for G in [derG.node0, derG.node1]:  # both overlap: in-graph nodes, and contour: not in-graph nodes
                        alt_graph = G.roott[1-fd]
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, CpH):  # not proto-graph or removed
                            graph.alt_graph_ += [alt_graph]
                            alt_graph.alt_graph_ += [graph]
                            # bilateral assign
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            if graph.alt_graph_:
                graph.alt_inder_s = CpH()  # players if fsub? der+: inder_s[-1] += player, rng+: players[-1] = player?
                for alt_graph in graph.alt_graph_:
                    sum_pH(graph.alt_inder_s, alt_graph.inder_s)  # accum alt_graph_ params
                    graph.alt_rdn += len(set(graph.inder_s.H[-1].node_).intersection(alt_graph.inder_s.H[-1].node_))  # overlap