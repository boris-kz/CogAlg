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

    medG_ = list  # last checked mediating [mG, dir_link, G]s, from all nodes?
    link_ = list  # evaluated graph links, open links replace alt_node
    node_ = list  # elements, common root of layers, levels:
    # agg,sub forks:
    mlevels = list  # PPs) Gs) GGs.: node_ agg, each a flat list?
    dlevels = list
    rlayers = list  # | mlayers: top node_ subdivision, parallel to levels but micro
    dlayers = list  # | alayers; val=0 for sub_recursion
    # sum params in all nesting orders:
    mplevels = lambda: CpH()  # zipped with alt_plevels in comp_plevels
    dplevels = lambda: CpH()  # both include players from rlayers and dlayers?
    x0 = float
    y0 = float  # center: box x0|y0 + L/2
    xn = float  # max distance from center
    yn = float
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs; no alt_rdn: valt representation in alt_PP_ valts?
    rng = lambda: 1  # not for alt_graphs
    alt_graph_ = list  # contour graphs
    roott = lambda: [None, None]  # higher-order segG or graph


def agg_recursion(root, G_, fseg, ifd):  # compositional recursion in root.PP_, pretty sure we still need fseg, process should be different

    mgraph_, dgraph_ = form_graph_(root, G_, fderG=0, ifd=ifd)  # PP cross-comp and clustering

    # intra graph:
    if root.mplevels.val > ave_sub * root.rdn:  # same in blob?
        root.rdn += 1  # +select per mgraph
        sub_rlayers, rval = sub_recursion_g(mgraph_, fseg=0, fd=0)  # subdivide graph.node_ by der+|rng+, accum root.valt
        root.rlayers = [sub_rlayers, rval]  # combined match of all der orders?
    if root.dlayers.val > ave_sub * root.rdn:
        root.rdn += 1  # +select per dgraph
        sub_dlayers, dval = sub_recursion_g(dgraph_, fseg=0, fd=1)
        root.dlayers = [sub_dlayers, dval]
    # cross graph:
    root.mlevels += mgraph_; root.dlevels += dgraph_
    for fd, graph_ in enumerate([mgraph_, dgraph_]):
        root_val = root.dplevels.val if fd else root.mplevels.val
        # include lower-level der vals?
        if (root_val > G_aves[fd] * ave_agg * (root.rdn + 1)) and len(graph_) > ave_nsub:
            root.rdn += 1  # estimate
            agg_recursion(root, graph_, fseg=fseg, ifd=fd)  # cross-comp graphs


def form_graph_(root, G_, fderG, ifd):  # forms plevel in agg+ or player in sub+, G is potential node graph, in higher-order GG graph
                                   # der+: comp_link if fderG, from sub_recursion_g?
    for G in G_:  # roott: mgraph, dgraph
        for i in 0,1:
            graph = [[G], [], 0]  # proto-GG: [node_, medG_, val]
            G.roott[i] = graph  # initial overlap between graphs is eliminated in eval_med_layer

    comp_G_(G_, ifd)  # cross-comp all graphs within rng, graphs may be segs | fderGs?
    mgraph_, dgraph_ = [],[]  # initialize graphs with >0 positive links in graph roots:
    for G in G_:
        if len(G.roott[0][0])>1 and G.roott[0] not in mgraph_: mgraph_ += [G.roott[0]]  # root = [node_, val] for eval_med_layer, + link_nvalt?
        if len(G.roott[1][0])>1 and G.roott[1] not in dgraph_: dgraph_ += [G.roott[1]]

    for fd, graph_ in enumerate([mgraph_, dgraph_]):  # evaluate intermediate nodes to delete or merge their graphs:
        regraph_ = []
        while graph_:
            graph = graph_.pop(0)
            eval_med_layer(graph_= graph_, graph=graph, fd=fd)
            if graph[2][fd] > ave_agg: regraph_ += [graph]  # graph reformed by merges and removes above

        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum proto-graph node_ params in graph
            for graph in graph_:
                sum_pH(root.dplevels if fd else root.mplevels, graph.dplevels if fd else graph.mplevels)

    return mgraph_, dgraph_


def comp_G_(G_, ifd):  # cross-comp Gs (patterns of patterns): Gs, derGs, or segs inside PP, same process, no fderG?

    for i, _G in enumerate(G_):
        for G in G_[i+1:]:
            # compare each G to other Gs in rng, bilateral link assign:
            if G in [node for link in _G.link_ for node in link.node_]:  # G,_G was compared in prior rng+, add frng to skip?
                continue
            if ifd: _plevels = _G.dplevels; plevels = G.dplevels
            else:   _plevels = _G.mplevels; plevels = G.mplevels
            dx = _G.x0 - G.x0; dy = _G.y0 - G.y0  # center x0,y0
            distance = np.hypot(dy, dx)  # Euclidean distance between centroids, sum in sparsity
            # proximity = ave_rng - distance?
            if distance < ave_distance * ((_plevels.val+plevels.val) / (2*sum(G_aves))):
                # comb G eval
                mplevel, dplevel = comp_pH(_plevels, plevels)
                mplevel.L,dplevel.L = 1,1; mplevel.S,dplevel.S = distance,distance; mplevel.A,dplevel.A = [dy,dx],[dy,dx]
                if _plevels.altTop and plevels.altTop:
                    altTopm, altTopd = comp_pH(_plevels.altTop, plevels.altTop)
                    mplevel.altTop = altTopm; dplevel.altTop=altTopd  # for sum,comp in agg+, reduced weighting of altVs?
                # new derG:
                y0 = (G.y0+_G.y0)/2; x0 = (G.x0+_G.x0)/2  # new center coords
                derG = Cgraph( node_=[_G,G], mplevels=mplevel, dplevels=dplevel, y0=y0, x0=x0, # compute max distance from center:
                               xn=max((_G.x0+_G.xn)/2 -x0, (G.x0+G.xn)/2 -x0), yn=max((_G.y0+_G.yn)/2 -y0, (G.y0+G.yn)/2 -y0))
                _G.link_ += [derG]; G.link_ += [derG]  # any val
                for fd, val in enumerate([mplevel.val, dplevel.val]):  # alt fork is redundant, no support?
                    if val > 0:
                        for node, (graph, medG_, val) in zip([_G, G], [G.roott[fd], _G.roott[fd]]):  # bilateral inclusion
                            if node not in graph:
                                graph += [node]
                                for derG in node.link_:
                                    mG = derG.node_[0] if derG.node_[1] is node else derG.node_[1]
                                    if mG not in [medG[0] for medG in medG_] :
                                        # some derG.plevels.H is [mplevel, dplevel], others may be converted to CpH in der+?
                                        medG_ += [[mG, derG, _G]]  # derG is init dir_mderG

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
                    comp_ext(_spH, spH, mpH, dpH, pri_fd)
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

def comp_ext(_spH, spH, mpH, dpH, fd):
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


def eval_med_layer(graph_, graph, fd):  # recursive eval of reciprocal links from increasingly mediated nodes

    node_, medG_, val = graph  # node_ is not used here
    save_node_, save_medG_ = [], []
    adj_Val = 0  # adjustment in connect val in graph

    mmG__ = []  # not reviewed
    for mG, dir_mderG, G in medG_:  # assign G and shortest direct derG to each med node?
        if not mG.roott[fd]:  # root graph was deleted
            continue
        G_plevels = G.dplevels if fd else G.mplevels
        fmed = 1; mmG_ = []  # __Gs that mediate between Gs and _Gs
        mmG__ += [mmG_]
        for mderG in mG.link_:  # all evaluated links

            mmG = mderG.node_[1] if mderG.node_[0] is mG else mderG.node_[0]
            for derG in G.link_:
                try_mG = derG.node_[0] if derG.node_[1] is G else derG.node_[1]
                if mmG is try_mG:  # mmG is directly linked to G
                    if derG.mplevels.S > dir_mderG.mplevels.S:  # same S in m and d
                        dir_mderG = derG  # for next med eval if dir link is shorter
                        fmed = 0
                    break
            if fmed:  # mderG is not reciprocal, else explore try_mG links in the rest of medG_
                for mmderG in mmG.link_:
                    if G in mmderG.node_:  # mmG mediates between mG and G
                        adj_val = mmderG.dplevels.val if fd else mmderG.mplevels.val - ave_agg
                        # adjust nodes:
                        G_plevels.val += adj_val; mG.dplevels.val if fd else mG.mplevels.val += adj_val
                        val += adj_val; mG.roott[fd][2] += adj_val  # root is not graph yet
                        mmG = mmderG.node_[0] if mmderG.node_[0] is not mG else mmderG.node_[1]
                        if mmG not in mmG_:  # not saved via prior mG
                            mmG_ += [mmG]
                            adj_Val += adj_val

    for (mG, dir_mderG, G), mmG_ in zip(medG_, mmG__):  # new
        if G_plevels.val>0:
            if G not in save_node_:
                save_node_+= [G]  # G remains in graph
            for mmG in mmG_:  # may be empty
                if mmG not in save_medG_:
                    save_medG_ += [[mmG, dir_mderG, G]]

    add_medG_, add_node_ = [],[]
    for mmG, dir_mderG, G in save_medG_:  # eval graph merge after adjusting graph by mediating node layer
        _graph = mmG.roott[fd]
        if _graph in graph_ and _graph is not graph:  # was not removed or merged via prior _G
            _node_, _medG_, _val = _graph
            for _node, _medG in zip(_node_, _medG_):  # merge graphs, add direct links:
                for _node in _node_:
                    if _node not in add_node_ + save_node_:
                        _node.roott[fd]=graph; add_node_ += [_node]
                for _medG in _medG_:
                    if _medG not in add_medG_ + save_medG_: add_medG_ += [_medG]
                for _derG in _node.link_:
                    adj_Val += _derG.dplevels.val if fd else _derG.mplevels.val - ave_agg
            val += _val
            graph_.remove(_graph)

    if val > ave_G:
        graph[:] = [save_node_+ add_node_, save_medG_+ add_medG_, val]
        if adj_Val > ave_med:  # positive adj_Val from eval mmG_
            eval_med_layer(graph_, graph, fd)  # eval next med layer in reformed graph
    else:
        for node in save_node_+ add_node_: node.roott[fd] = []  # delete roots


def sub_recursion_g(graph_, fseg, fd):  # rng+: extend G_ per graph, der+: replace G_ with derG_

    comb_layers_t = [[],[]]
    sub_val = 0
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

        valSub = graph_plevels.val
        if valSub > G_aves[fd] and len(node_) > ave_nsub:  # graph.valt eval for agg+ only
            sub_mgraph_, sub_dgraph_ = form_graph_(graph, node_, fd, ifd=fd)  # cross-comp and clustering cycle
            # rng+:
            if graph.rval > ave_sub * graph.rdn:  # >cost of calling sub_recursion and looping:
                sub_rlayers, rval = sub_recursion_g(sub_mgraph_, fseg=fseg, fd=0)
                sub_val += rval
                graph.rlayers = [[sub_mgraph_] + [sub_rlayers], rval]  # draft
            # der+:
            if graph.dval > ave_sub * graph.rdn:
                sub_dlayers, dval = sub_recursion_g(sub_dgraph_, fseg=fseg, fd=1)
                sub_val += dval
                graph.dlayers = [[sub_dgraph_] + [sub_dlayers], dval]  # draft

            for comb_layers, graph_layers in zip(comb_layers_t, [graph.rlayers, graph.dlayers]):
                for i, (comb_layer, graph_layer) in enumerate(zip_longest(comb_layers, graph_layers, fillvalue=[])):
                    if graph_layer:
                        if i > len(comb_layers) - 1:
                            comb_layers += [graph_layer]  # add new r|d layer
                        else:
                            comb_layers[i] += graph_layer  # splice r|d PP layer into existing layer

    return comb_layers_t, sub_val


def sum2graph_(G_, fd):  # sum node and link params into graph, plevel in agg+ or player in sub+: if fderG?
                         # fd for clustering, same or fderG for alts?
    graph_ = []
    for G in G_:
        X0,Y0, Xn,Yn = 0,0,0,0
        node_, medG_, val = G
        L = len(node_)
        for node in node_: X0+=node.x0; Y0+=node.y0  # first pass defines center
        X0/=L; Y0/=L
        graph = Cgraph(L=L, node_=node_, medG_=medG_)
        new_plevel = CpH(L=1)  # 1st link adds 2 nodes, other links add 1, one node is already in the graph

        for node in node_:  # 2nd pass defines max distance and other params:
            Xn = max(Xn,(node.x0+node.xn)-X0)  # box xn = x0+xn
            Yn = max(Yn,(node.y0+node.yn)-Y0)
            if fd: graph_plevels = graph.dplevels; node_plevels = node.dplevels
            else:  graph_plevels = graph.mplevels; node_plevels = node.mplevels
            # accum params:
            sum_pH(graph_plevels, node_plevels)  # same for fder , # another fork's plevels will be empty
            for derG in node.link_:
                derG_plevels = derG.dplevels if fd else derG.mplevels
                sum_pH(new_plevel, derG_plevels)  # accum derG, += L=1, S=link.S, preA: [Xn,Yn]
                derG.roott[fd] = graph  # + link_ = [derG]?
                val += derG_plevels.val

            node_plevels.val += val  # derG valts summed in eval_med_layer added to lower-plevels valt
            graph_plevels.val += node_plevels.val

        new_plevel.A = [Xn*2,Yn*2]
        graph_plevels.H += [new_plevel]
        graph_plevels.fds = copy(node_plevels.fds) + [fd]
        graph.x0=X0; graph.xn=Xn; graph.y0=Y0; graph.yn=Yn
        graph_ += [graph]

    for graph in graph_:  # 2nd pass: accum alt_graph_ params
        AltTop = CpH()  # Alt_players if fsub
        for node in graph.node_:
            for derG in node.link_:
                for G in derG.node_:
                    if G not in graph.node_:  # alt graphs are roots of not-in-graph G in derG.node_
                        alt_graph = G.roott[1-fd]
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, Cgraph):  # not proto-graph or removed
                            alt_plevels = alt_graph.dplevels if fd else alt_graph.mplevels
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


def sum_pH(PH, pH, fneg=0):  # recursive unpack plevels ( pplayer ( players ( ptuples ( ptuple, no accum across fd: matched in comp_pH

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