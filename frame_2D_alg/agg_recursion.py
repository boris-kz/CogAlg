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
All aves are defined for rdn+1 
'''

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

class CQ(ClusterStructure):  # nodes or links

    Q = list  # all tested links or +ve nodes
    val = lambda: [0,0]  # valt in link_, val in node_
    lQm = list  # positive links per mfork
    mval = float
    lQd = list  # positive links per dfork
    dval = float

class CpH(ClusterStructure):  # hierarchy of params + associated vars

    H = list  # plevels | pplayer | players | ptuples
    fds = list  # (m|d)s, only in plevels and players
    val = int
    nval = int  # of neg open links?
    # extuple in pplayer=plevel only, skip if L==0:
    L = float  # len_node_
    S = float  # sparsity: sum_len_links
    A = list   # axis: in derG or high-aspect G only?
    altTop = list  # optional lower-nested alt_CpH = cis_CpH.H[0] for agg+ sum,comp


class Cgraph(CPP):  # graph or generic PP of any composition

    link_ = lambda: CQ() # evaluated graph_as_node external links, replace alt_node if open, direct only?
    node_ = lambda: CQ() # fd-selected elements, common root of layers, levels:
    # agg,sub forks: [[],0] if called
    mlevels = list  # PPs) Gs) GGs.: node_ agg, each a flat list
    dlevels = list
    rlayers = list  # | mlayers: init from node_ for subdivision, micro relative to levels
    dlayers = list  # | alayers; init val = sum node_val, for sub_recursion
    # summed params in levels ( layers:
    mplevels = lambda: CpH()  # zipped with alt_plevels in comp_plevels
    dplevels = lambda: CpH()  # both include node_ params (layer0) and rlayers'players + dlayers'players?
    x0 = float
    y0 = float  # center: box x0|y0 + L/2
    xn = float  # max distance from center
    yn = float
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs; no alt_rdn: valt representation in alt_PP_ valts?
    rng = lambda: 1  # not for alt_graphs
    alt_graph_ = list  # contour graphs
    roott = lambda: [None, None]  # higher-order segG or graph


def agg_recursion(root, G_, fseg, ifd):  # compositional recursion in root.PP_, pretty sure we still need fseg, process should be different

    mgraph_, dgraph_ = form_graph_(root, G_, ifd=ifd)  # PP cross-comp and clustering
    mval = sum([mgraph.mplevels.val for mgraph in mgraph_])
    dval = sum([dgraph.dplevels.val for dgraph in dgraph_])
    root.mlevels += mgraph_; root.dlevels += dgraph_

    for fd, (graph_,val) in enumerate(zip([mgraph_,dgraph_],[mval,dval])):  # same graph_, val for sub+ and agg+
        # intra-graph sub+:
        if val > ave_sub * (root.rdn):  # same in blob, same base cost for both forks
            root.rdn+=1  # estimate
            sub_layers, val = sub_recursion_g(graph_, val, fseg, fd=fd)  # subdivide graph_ by der+|rng+, accum val
        else:
            sub_layers = []
        [root.rlayers, root.dlayers][fd] = [sub_layers, val]  # combined val of all new der orders
        [root.mplevels,root.dplevels][fd] = CpH()  # replace with summed params of higher-composition graphs:
        for graph in graph_:
            sum_pH([root.mplevels,root.dplevels][fd], [graph.mplevels,graph.dplevels][fd])  # not mplevels+dplevels?
        # cross-graph agg+:
        if val > G_aves[fd] * ave_agg * (root.rdn) and len(graph_) > ave_nsub:
            root.rdn+=1  # estimate
            agg_recursion(root, graph_, fseg=fseg, ifd=fd)  # cross-comp graphs


def form_graph_(root, G_, ifd): # form plevel in agg+ or player in sub+, G is node in GG graph; der+: comp_link if fderG, from sub+

    comp_G_(G_, ifd)  # cross-comp all graphs within rng, graphs may be segs | fderGs, G.roott += link, link.node
    mnode_, dnode_ = [], []  # Gs with >0 +ve fork links:
    for G in G_:
        if G.link_.Qm: mnode_ += [G]  # all nodes with +ve links, not clustered in graphs yet
        if G.link_.Qd: dnode_ += [G]
    graph_t = []
    Gm_, Gd_ = copy(G_), copy(G_)
    # form proto-graphs of linked nodes:
    for fd, (node_, G_) in enumerate(zip([mnode_, dnode_], [Gm_, Gd_])):
        graph_ = []
        while G_:  # all Gs not removed in add_node_layer
            G = G_.pop()
            gnode_=[G]; val=0
            val = add_node_layer(gnode_, val, G_, G, fd)  # recursive depth-first gnode_+=[_G]
            graph_ += CQ(Q=gnode_, val=val)

        regraph_ = graph_reval(graph_, fd)  # graphs recursively reformed by node.link_.val, reduced by pruning links
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum proto-graph node_ params in graph
            for graph in graph_:
                root_plevels = [root.mplevels, root.dplevels][fd]; plevels = [graph.mplevels, graph.dplevels][fd]
                if root_plevels.H or plevels.H:  # better init plevels=list?
                    sum_pH(root_plevels, plevels)
        graph_t += [graph_]

    return graph_t

def graph_reval(graph_, fd):  # recursive eval nodes for regraph, increasingly selective with reduced node.link_.val

    regraph_ = []
    while graph_:
        graph = graph_.pop()
        # some links will be lost in reval, the graph may split into multiple regraphs, init each with graph.Q node:
        while graph.Q:
            node = graph.Q.pop()  # node_, not removed below
            val = node.link_.valt[fd]  # in-graph links only
            if val > G_aves[fd]:  # else skip
                regraph = CQ(Q=[node], val=val)  # init for each node, then add _nodes
                node.roott[fd] = regraph
                readd_node_layer(regraph, graph.Q, node, fd)  # recursive depth-first regraph.Q+=[_node]
                regraph_ += regraph
        if graph.val-regraph.val > ave_G:  # graph reval while min val reduction
            regraph_ += [regraph]
    if regraph_: regraph_ = graph_reval(regraph_, fd)

    return regraph_

def readd_node_layer(regraph, graph_Q, node, fd):  # recursive depth-first regraph.Q+=[_node]

    for link in node.link_.Q:  # all positive
        _node = link.node_.Q[1] if link.node_.Q[0] is node else link.node_.Q[0]
        _val = _node.link_.valt[fd]
        if _val > G_aves[fd]:
            regraph.Q += [_node]; regraph.val += _val; _node.roott[fd] = regraph
            graph_Q.remove(_node)
            readd_node_layer(regraph, graph_Q, _node, fd)

def add_node_layer(gnode_, val, G_, G, fd):  # recursive depth-first gnode_+=[_G]

    for link in G.link_.Q:  # all positive
        _G = link.node_.Q[1] if link.node_.Q[0] is G else link.node_.Q[0]
        if _G in G_:  # _G is not removed in prior loop
            gnode_ += [_G]
            G_.remove(_G)
            val += add_node_layer(gnode_, val, G_, _G, fd)


def comp_G_(G_, ifd):  # cross-comp Gs (patterns of patterns): Gs, derGs, or segs inside PP, same process, no fderG?

    for i, _G in enumerate(G_):
        for G in G_[i+1:]:
            # compare each G to other Gs in rng, bilateral link assign, val accum:
            if G in [node for link in _G.link_.Q for node in link.node_.Q]:  # G,_G was compared in prior rng+, add frng to skip?
                continue
            _plevels = [_G.mplevels, _G.dplevels][ifd]; plevels = [G.mplevels, G.dplevels][ifd]
            dx = _G.x0 - G.x0; dy = _G.y0 - G.y0  # center x0,y0
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity
            # proximity = ave_rng - distance?
            if distance < ave_distance * ((_plevels.val+plevels.val) / (2*sum(G_aves))):
                # comb G eval
                mplevel, dplevel = comp_pH(_plevels, plevels)
                mplevel.L, dplevel.L = 1,1; mplevel.S, dplevel.S = distance,distance; mplevel.A, dplevel.A = [dy,dx],[dy,dx]
                mplevels = CpH(H=[mplevel], fds=[0], val=mplevel.val); dplevels = CpH(H=[dplevel], fds=[1], val=dplevel.val)
                # comp alts
                if _plevels.altTop and plevels.altTop:
                    altTopm,altTopd = comp_pH(_plevels.altTop, plevels.altTop)
                    mplevels.altTop = CpH(H=[altTopm],fds=[0],val=altTopm.val)  # for sum,comp in agg+, reduced weighting of altVs?
                    dplevels.altTop = CpH(H=[altTopd],fds=[1],val=altTopd.val)
                # new derG:
                y0 = (G.y0+_G.y0)/2; x0 = (G.x0+_G.x0)/2  # new center coords
                derG = Cgraph( node_=CQ(Q=[_G,G]), mplevels=mplevels, dplevels=dplevels, y0=y0, x0=x0, # compute max distance from center:
                               xn=max((_G.x0+_G.xn)/2 -x0, (G.x0+G.xn)/2 -x0), yn=max((_G.y0+_G.yn)/2 -y0, (G.y0+G.yn)/2 -y0))
                mval = derG.mplevels.val
                dval = derG.dplevels.val
                _G.link_.Q += [derG]; _G.link_.valt[0] += mval; _G.link_.valt[1] += dval  # +|- links
                G.link_.Q += [derG]; G.link_.valt[0] += mval; G.link_.valt[1] += dval
                if mval > ave_Gm:
                    _G.link_.Qm += [derG]; _G.link_.mval += mval  # no dval for Qm
                    G.link_.Qm += [derG]; G.link_.mval += mval
                if dval > ave_Gd:
                    _G.link_.Qd += [derG]; _G.link_.dval += dval  # no mval for Qd
                    G.link_.Qd += [derG]; G.link_.dval += dval


def comp_pH(_pH, pH):  # recursive unpack plevels ( pplayer ( players ( ptuples -> ptuple:
                       # if derG: compare mplevel or dplevel
    mpH, dpH = CpH(), CpH()
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
                if spH.L:  # extuple is valid, in pplayer only
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
    L, S, A = spH.L, spH.S, spH.A
    _L, _S, _A = _spH.L, _spH.S, _spH.A

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


def sub_recursion_g(graph_, Sval, fseg, fd):  # rng+: extend G_ per graph, der+: replace G_ with derG_

    comb_layers_t = [[],[]]
    for graph in graph_:
        if fd:
            graph_plevels = graph.dplevels
            node_ = []  # fill with positive links in graph:
            for node in graph.node_:
                for link in node.link_:
                    if link.dplevels.val>0 and link not in node_:
                        node_ += [link]
        else:
            graph_plevels = graph.mplevels
            node_ = graph.node_

        if graph_plevels.val > G_aves[fd] and len(node_) > ave_nsub:
            sub_mgraph_, sub_dgraph_ = form_graph_(graph, node_, ifd=fd)  # cross-comp and clustering cycle
            # rng+:
            Rval = sum([sub_mgraph.mplevels.val for sub_mgraph in sub_mgraph_])
            if Rval > ave_sub * graph.rdn:  # >cost of call:
                sub_rlayers, rval = sub_recursion_g(sub_mgraph_, Rval, fseg=fseg, fd=0)
                Rval+=rval; graph.rlayers = [[sub_mgraph_]+[sub_rlayers], Rval]
            else:
                graph.rlayers = [[]]  # this can be skipped if we init layers as [[]]
            # der+:
            Dval = sum([sub_dgraph.dplevels.val for sub_dgraph in sub_dgraph_])
            if Dval > ave_sub * graph.rdn:
                sub_dlayers, dval = sub_recursion_g(sub_dgraph_, Dval, fseg=fseg, fd=1)
                Dval+=dval; graph.dlayers = [[sub_dgraph_]+[sub_dlayers], Dval]
            else:
                graph.dlayers = [[]]

            for comb_layers, graph_layers in zip(comb_layers_t, [graph.rlayers[0], graph.dlayers[0]]):
                for i, (comb_layer, graph_layer) in enumerate(zip_longest(comb_layers, graph_layers, fillvalue=[])):
                    if graph_layer:
                        if i > len(comb_layers) - 1:
                            comb_layers += [graph_layer]  # add new r|d layer
                        else:
                            comb_layers[i] += graph_layer  # splice r|d PP layer into existing layer
            Sval += Rval + Dval

    return comb_layers_t, Sval


def sum2graph_(G_, fd):  # sum node and link params into graph, plevel in agg+ or player in sub+: if fderG?
                         # fd for clustering, same or fderG for alts?
    graph_ = []
    for G in G_:
        X0,Y0, Xn,Yn = 0,0,0,0
        node_, val = G.Q, G.valt[fd]
        L = len(node_)
        for node in node_: X0+=node.x0; Y0+=node.y0  # first pass defines center
        X0/=L; Y0/=L
        graph = Cgraph(L=L, node_=node_)
        graph_plevels = [graph.mplevels, graph.dplevels][fd]  # init with node_[0]?
        new_plevel = CpH(L=1)  # 1st link adds 2 nodes, other links add 1, one node is already in the graph

        for node in node_:  # define max distance,A, sum plevels:
            Xn = max(Xn,(node.x0+node.xn)-X0)  # box xn = x0+xn
            Yn = max(Yn,(node.y0+node.yn)-Y0)
            plevels = [node.mplevels, node.dplevels][fd]  # node: G|derG, sum plevels ( pplayers ( players ( ptuples:
            sum_pH(graph_plevels, plevels)
            rev=0
            while len(node.mplevels.H)==1 and len(node.dplevels.H)==1: # or plevels.H and plevels.H[0].L==1: node is derG:
                rev=1
                node = node.node_[0]  # get lower pplayers from node.node_[0]:
                node_plevels = node.mplevels if node.mplevels.H else node.dplevels  # prior sub+ fork
                if len(node.mplevels.H)==1 and len(node.dplevels.H)==1:  # node is derG in der++
                   sum_pH(graph_plevels, node_plevels)  # sum lower pplayers of the top plevel in reversed order
            if rev:
                i = 2**len(plevels.H); _i = -i  # n_players in implicit pplayer = n_higher_plevels ^2: 1|1|2|4...
                inp = graph_plevels.H[0]
                rev_pplayers = CpH(val=inp.val, L=inp.L, S=inp.S, A=inp.A)
                for players, fd in zip(inp.H[-i:-_i], inp.fds[-i:-_i]): # reverse pplayers to bottom-up, keep sequence of players in pplayer:
                    rev_pplayers.H += [players]; rev_pplayers.fds += [fd]
                    _i = i; i += int(np.sqrt(i))
                graph_plevels = CpH(H=node_plevels.H+[rev_pplayers], val=node_plevels.val+graph_plevels.val, fds=node_plevels.fds+[fd])
                # low plevels: last node_[0] in while derG, + top plevel: pplayers of node==derG
                plevels = graph_plevels  # for accum below?
            for derG in node.link_.Q:
                derG_plevels = [derG.mplevels, derG.dplevels][fd]
                sum_pH(new_plevel, derG_plevels.H[0])  # sum new_plevel across nodes, accum derG, += L=1, S=link.S, preA: [Xn,Yn]
                val += derG_plevels.val

            plevels.val += val  # new val in node
            graph_plevels.val += plevels.val

        new_plevel.A = [Xn*2,Yn*2]
        graph.x0=X0; graph.xn=Xn; graph.y0=Y0; graph.yn=Yn
        graph_plevels.H += [new_plevel]  # val is summed per node
        graph_plevels.fds += [fd]
        graph_ += [graph]

    for graph in graph_:  # 2nd pass: accum alt_graph_ params
        AltTop = CpH()  # Alt_players if fsub
        for node in graph.node_:
            for derG in node.link_.Q:
                for G in derG.node_.Q:
                    if G not in graph.node_:  # alt graphs are roots of not-in-graph G in derG.node_
                        alt_graph = G.roott[1-fd]
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, Cgraph):  # not proto-graph or removed
                            alt_plevels = [alt_graph.mplevels, alt_graph.dplevels][1-fd]
                            if AltTop.H:  # der+: plevels[-1] += player, rng+: players[-1] = player?
                                sum_pH(AltTop.H[0], alt_plevels.H[0])
                            else:
                                AltTop.H = [alt_plevels.H[0]]; AltTop.val = alt_plevels.H[0].val; AltTop.fds = alt_plevels.H[0].fds
                            graph.alt_graph_ += [alt_graph]
                            alt_graph.alt_graph_ += [graph]  # bilateral assign
        if graph.alt_graph_:
            graph.alt_graph_ += [AltTop]  # temporary storage

    for graph in graph_:  # 3rd pass: add alt fork to each graph plevel, separate to fix nesting in 2nd pass
        if graph.alt_graph_:
            AltTop = graph.alt_graph_.pop()
            add_alt_top(graph_plevels, AltTop)

    return graph_

# not revised:
def add_alt_top(plevels, AltTop):
    # plevels, not sure on fds yet
    plevels.altTop = AltTop; plevels.val = AltTop.val
    # pplayer
    for pplayer, AltTop_pplayer in zip(plevels.H, AltTop.H[:1]):  # get only 1st index
        pplayer.altTop = AltTop_pplayer; pplayer.val = AltTop_pplayer.val
        # players, not sure on fds yet
        for players, AltTop_players in zip(pplayer.H, AltTop_pplayer.H[:1]):
            players.altTop = AltTop_players; players.val = AltTop_players.val
            # ptuples
            for ptuples, AltTop_ptuples in zip(players.H, AltTop_players.H[:1]):
                ptuples.altTop = AltTop_ptuples; ptuples.val = AltTop_ptuples.val

# please revise:
def add_alts(cplevel, aplevel):

    cForks, cValt = cplevel; aForks, aValt = aplevel
    csize = len(cForks)
    cplevel[:] = [cForks + aForks, [sum(cValt), sum(aValt)]]  # reassign with alts

    for cFork, aFork in zip(cplevel[0][:csize], cplevel[0][csize:]):
        cplayers, cfvalt, cfds = cFork
        aplayers, afvalt, afds = aFork
        cFork[1] = [sum(cfvalt), afvalt[0]]

        for cplayer, aplayer in zip(cplayers, aplayers):
            cforks, cvalt = cplayer
            aforks, avalt = aplayer
            cplayer[:] = [cforks+aforks, [sum(cvalt), avalt[0]]]  # reassign with alts


def sum_pH(PH, pH, fneg=0):  # recursive unpack plevels ( pplayers ( players ( ptuples, no accum across fd: matched in comp_pH

    if PH.L:  # valid extuple
        PH.L += pH.L; PH.S += pH.S
        if PH.A:
            if isinstance(PH.A, list):
                PH.A[0] += pH.A[0]; PH.A[1] += pH.A[1]
            else:
                PH.A += pH.A
        else:
            PH.A = copy(pH.A)
    for SpH, spH in zip_longest(PH.H, pH.H, fillvalue=None):
        if spH:
            if SpH:
                if isinstance(SpH, Cptuple):
                    sum_ptuple(SpH, spH, fneg=fneg)
                else:
                    sum_pH(SpH, spH, fneg=0)  # unpack sub-hierarchy, recursively
            elif not fneg:
                PH.H.append(deepcopy(spH))  # new Sub_pH
    PH.val += pH.val