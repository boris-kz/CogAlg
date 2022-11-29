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

    H = list  # plevels | players | PPlayers | ptuples
    val = int
    nval = int  # of neg open links?
    fds = list  # m|d per element, not in ptuples
    # extuple in pptuples only, skip if L==0:
    L = float  # len_node_
    S = float  # sparsity: sum_len_links
    A = list   # axis: in derG or high-aspect G only?

class Cgraph(CPP):  # graph or generic PP of any composition

    plevels = lambda: CpH()  # zipped with alt_plevels in comp_plevels
    alt_plevels = lambda: CpH()  # summed from alt_graph_, sub comp support, agg comp suppression?
    alt_graph_ = list
    x = float  # median: x0+L/2
    y = float
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs; no alt_rdn: valt representation in alt_PP_ valts?
    rng = lambda: 1  # not for alt_graphs
    medG_ = list  # last checked mediating [mG, dir_link, G]s, from all nodes?
    link_ = list  # evaluated graph links, open links replace alt_node
    node_ = list  # graph elements, root of layers, levels:
    rlayers = list  # | mlayers, top-down
    dlayers = list  # | alayers
    mlevels = list  # agg_PPs ) agg_PPPs ) agg_PPPPs.., bottom-up
    dlevels = list
    roott = lambda: [None, None]  # higher-order segG or graph


def agg_recursion(root, G_, fseg):  # compositional recursion in root.PP_, pretty sure we still need fseg, process should be different

    mgraph_, dgraph_ = form_graph_(root, G_, fder=0)  # PP cross-comp and clustering

    # intra graph:
    if root.plevels.val > ave_sub * root.rdn:
        sub_rlayers, rvalt = sub_recursion_g(mgraph_, root.valt, fd=0)  # subdivide graph.node_ by der+|rng+, accum root.valt
        root.valt[0] += sum(rvalt); root.rlayers = sub_rlayers
    if root.alt_plevels.val > ave_sub * root.rdn:
        sub_dlayers, dvalt = sub_recursion_g(dgraph_, root.valt, fd=1)
        root.valt[1] += sum(dvalt); root.dlayers = sub_dlayers
    # cross graph:
    root.mlevels += mgraph_; root.dlevels += dgraph_
    for fd, graph_ in enumerate([mgraph_, dgraph_]):
        # agg+ if new plevel valt:
        if (root.plevels.H[-1].val > G_aves[fd] * ave_agg * (root.rdn + 1)) and len(graph_) > ave_nsub:
            root.rdn += 1  # estimate
            agg_recursion(root, graph_, fseg=fseg)  # cross-comp graphs


def form_graph_(root, G_, fder):  # forms plevel in agg+ or player in sub+, G is potential node graph, in higher-order GG graph

    for G in G_:  # initialize mgraph, dgraph as roott per G, for comp_G_
        for i in 0,1:
            graph = [[G], [], [0,0]]  # proto-GG: [node_, medG_, valt]
            G.roott[i] = graph

    comp_G_(G_, fder)  # cross-comp all graphs within rng, graphs may be segs
    mgraph_, dgraph_ = [],[]  # initialize graphs with >0 positive links in graph roots:
    for G in G_:
        if len(G.roott[0][0])>1: mgraph_ += [G.roott[0]]  # root = [node_, valt] for cluster_node_layer eval, + link_nvalt?
        if len(G.roott[1][0])>1: dgraph_ += [G.roott[1]]

    for fd, graph_ in enumerate([mgraph_, dgraph_]):  # evaluate intermediate nodes to delete or merge graphs:
        regraph_ = []
        while graph_:
            graph = graph_.pop(0)
            eval_med_layer(graph_= graph_, graph=graph, fd=fd)
            if graph[2][fd] > ave_agg: regraph_ += [graph]  # graph reformed by merges and removes above

        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd, fder)  # sum proto-graph node_ params in graph
            plevels = deepcopy(graph_[0].plevels)  # same fds in all nodes?
            for graph in graph_[1:]:
                sum_pH(plevels, graph.plevels)
            root.plevels = plevels

    return mgraph_, dgraph_


def comp_G_(G_, fder):  # cross-comp Gs (patterns of patterns): Gs, derGs, or segs inside PP

    for i, _G in enumerate(G_):
        for G in G_[i+1:]:
            # compare each G to other Gs in rng, bilateral link assign:
            if G in [node for link in _G.link_ for node in link.node_]:  # G,_G was compared in prior rng+, add frng to skip?
                continue
            dx = _G.x - G.x; dy = _G.y - G.y
            distance = np.hypot(dy, dx)  # Euclidean distance between centroids, sum in sparsity
            # proximity = ave_rng - distance?
            if distance < ave_distance * ((_G.plevels.val+G.plevels.val) / (2*sum(G_aves))):
                # comb G eval
                mplevel, dplevel = comp_pH(_G.plevels, G.plevels)
                alt_mplevel, alt_dplevel = comp_pH(_G.alt_plevels, G.alt_plevels)
                # draft:
                mplevel.H=[CpH(L=1, S=distance, A=[dy,dx])]  # 1st player: L,S,A of derG, in G: summed derG S, redefined L,A?
                dplevel.H=[CpH(L=1, S=distance, A=[dy,dx])]  # external params or their derivatives in graph of any fd?
                plevels = CpH( H=[mplevel,dplevel])
                # in _G,G ) hl graph:
                derG = Cgraph( node_=[_G,G], plevels=plevels, val=mplevel.val + alt_mplevel.val,  # comb val?
                               y0=min(G.y0,_G.y0), yn=max(G.yn,_G.yn), x0=min(G.x0,_G.x0), xn=max(G.xn,_G.xn))
                               # or mean x0=_x+dx/2, y0=_y+dy/2?
                _G.link_ += [derG]; G.link_ += [derG]  # any val
                for fd, val in enumerate([mplevel.val, alt_mplevel.val]):  # alt fork is redundant, no support?
                    if val > 0:
                        for node, (graph, medG_, gvalt) in zip([_G, G], [G.roott[fd], _G.roott[fd]]):  # bilateral inclusion
                            if node not in graph:
                                graph += [node]
                                for derG in node.link_:
                                    mG = derG.node_[0] if derG.node_[1] is node else derG.node_[1]
                                    if mG not in medG_:
                                        medG_ += [[mG, derG, _G]]  # derG is init dir_mderG


def comp_pH(_pH, pH):  # recursive unpack plevels ( players ( PPlayers ( ptuples -> ptuple:

    mpH, dpH = CpH(), CpH()
    pri_fd = 0
    for i, _spH, spH in enumerate(zip(_pH.H, pH.H)):

        _fd = _pH.fds[i] if _pH.fds else 0
        fd = _pH.fds[i] if pH.fds else 0
        if _fd == fd:
            if fd:  # all scalars
                pri_fd = 1
            if isinstance(spH, Cptuple):
                mtuple, dtuple = comp_ptuple(_spH, spH, pri_fd)
                mpH.H += [mtuple]; mpH.val += mtuple.val
                dpH.H += [dtuple]; dpH.val += dtuple.val
            else:
                if spH.L:  # extuple is valid
                    comp_ext([_spH.L,_spH.S,_spH.A], [spH.L,spH.S,spH.A], [mpH.L,mpH.S,mpH.A], [dpH.L,dpH.S,dpH.A], mpH.val, dpH.val, pri_fd)
                sub_mpH, sub_dpH = comp_pH(_spH, spH)
                mpH.H += [sub_mpH]; mpH.val += sub_mpH.val
                dpH.H += [sub_dpH]; dpH.val += sub_dpH.val
        else:
            break

    return mpH, dpH

def comp_ext(_ext_, ext_, mext_, dext_, mval, dval, fd):

    for [_L, _S, _A], [L, S, A], [mL, mS, mA], [dL, dS, dA] in _ext_, ext_, mext_, dext_:

        _sparsity = _S / (_L - 1); sparsity = S / (L - 1)  # average distance between connected nodes
        dS[:] = _sparsity - sparsity; dval += dS
        mS[:] = min(_sparsity, sparsity); mval += mS
        dL[:] = _L - L; dval += dL
        mL[:] = min(_L, L); mval += mL
        if _A and A: # axis: dy,dx only for derG or high-aspect Gs, both val *= aspect?
            if fd:   # scalar mA or dA
                dA[:] = _A - A; mA[:] = min(_A, A)
            else:
                m, d = comp_angle(None, _A, A)
                mA[:] = m; dA[:] = d
        else:
            mA[:] = 1; dA[:] = 0  # no difference, matching low-aspect, only if both?
        mval += mA; dval += dA


# draft:
def eval_med_layer(graph_, graph, fd):   # recursive eval of reciprocal links from increasingly mediated nodes

    node_, medG_, valt = graph  # node_ is not used here
    save_node_, save_medG_ = [], []
    adj_Val = 0  # adjust connect val in graph

    for mG, dir_mderG, G in medG_:  # assign G and shortest direct derG to each med node?
        mmG_ = []  # __Gs that mediate between Gs and _Gs
        fmed = 1
        for mderG in mG.link_:  # all direct evaluated links

            mmG = mderG.node_[1] if mderG.node_[0] is mG else mderG.node_[0]
            for derG in G.link_:
                try_mG = derG.node_[0] if derG.node_[1] is G else derG.node_[1]
                if mmG is try_mG:  # mmG is directly linked to G
                    if derG.plevels.H[fd].S > dir_mderG.plevels.H[fd].S:
                        dir_mderG = derG  # for next med eval if dir link is shorter
                        fmed = 0
                    break
            if fmed:  # mderG is not reciprocal, else explore try_mG links in the rest of medG_
                for mmderG in mmG.link_:
                    if G in mmderG.node_:  # mmG mediates between mG and G
                        adj_val = mmderG.plevels.H[fd].val - ave_agg  # or increase ave per mediation depth
                        # adjust nodes:
                        G.plevels.val += adj_val; mG.plevels.val += adj_val  # valts are not updated
                        valt[fd] += adj_val; mG.roott[fd][2][fd] += adj_val  # root is not graph yet
                        mmG = mmderG.node_[0] if mmderG.node_[0] is not mG else mmderG.node_[1]
                        if mmG not in mmG_:  # not saved via prior mG
                            mmG_ += [mmG]
                            adj_Val += adj_val
        if G.plevels.val>0:
            save_node_+= [G]  # G remains in graph
            for mmG in mmG_:  # may be empty
                if mmG not in save_medG_:
                    save_medG_ += [[mmG, dir_mderG, G]]

    add_medG_, add_node_ = [],[]
    for mmG, dir_mderG, G in save_medG_:  # eval graph merge after adjusting graph by mediating node layer
        _graph = mmG.roott[fd]
        if _graph in graph_ and _graph is not graph:  # was not removed or merged via prior _G
            _node_, _medG_, _valt = _graph
            for _node, _medG in zip(_node_, _medG_):  # merge graphs, add direct links:
                for _node in _node_:
                    if _node not in add_node_ + save_node_: add_node_ += [_node]
                for _medG in _medG_:
                    if _medG not in add_medG_ + save_medG_: add_medG_ += [_medG]
                for _derG in _node.link_:
                    adj_Val += _derG.valt[fd] - ave_agg

            valt[fd] += _valt[fd]
            graph_.remove(_graph)

    graph[:] = [save_node_+ add_node_, save_medG_+ add_medG_, valt]
    if adj_Val > ave_med:  # positive adj_Val from eval mmG_
        eval_med_layer(graph_, graph, fd)  # eval next med layer in reformed graph


def sub_recursion_g(graph_, fseg, fd):  # rng+: extend G_ per graph, der+: replace G_ with derG_

    comb_layers_t = [[],[]]
    sub_valt = [0,0]
    for graph in graph_:
        if fd:
            node_ = []  # positive links within graph
            for node in graph.node_:
                for link in node.link_:
                    if link.valt[1]>0 and link not in node_:
                        if not link.fds:  # might be converted in other graph
                            link.fds = [fd]; link.valt = link.plevels[fd][1]; link.plevels = [link.plevels[fd]]  # single-plevel
                        node_ += [link]
        else: node_ = graph.node_

        valSub = graph.plevels[-1][1][fd]  # last plevel valt: ca = dm if fd else md: edge or core, or single player?
        if valSub > G_aves[fd] and len(node_) > ave_nsub:  # graph.valt eval for agg+ only

            sub_mgraph_, sub_dgraph_ = form_graph_(graph, node_, fder=fd)  # cross-comp and clustering cycle
            # rng+:
            if graph.valt[0] > ave_sub * graph.rdn:  # >cost of calling sub_recursion and looping:
                sub_rlayers, valt = sub_recursion_g(sub_mgraph_, graph.valt, fd=0)
                rvalt = sum(valt); graph.valt[0] += rvalt; sub_valt[0] += rvalt  # not sure
                graph.rlayers = [sub_mgraph_] + [sub_rlayers]
            # der+:
            if graph.valt[1] > ave_sub * graph.rdn:
                sub_dlayers, valt = sub_recursion_g(sub_dgraph_, graph.valt, fd=1)
                dvalt = sum(valt); graph.valt[1] += dvalt; sub_valt[1] += dvalt
                graph.dlayers = [sub_dgraph_] + [sub_dlayers]

            for comb_layers, graph_layers in zip(comb_layers_t, [graph.rlayers, graph.dlayers]):
                for i, (comb_layer, graph_layer) in enumerate(zip_longest(comb_layers, graph_layers, fillvalue=[])):
                    if graph_layer:
                        if i > len(comb_layers) - 1:
                            comb_layers += [graph_layer]  # add new r|d layer
                        else:
                            comb_layers[i] += graph_layer  # splice r|d PP layer into existing layer

    return comb_layers_t, sub_valt


def sum2graph_(G_, fd, fderG):  # sum node and link params into graph, plevel in agg+ or player in sub+: if fder

    graph_ = []
    for G in G_:
        node_, medG_, valt = G
        graph = Cgraph(x0=node_[0].x0, xn=node_[0].xn, y0=node_[0].y0, yn=node_[0].yn, node_=node_, medG_=medG_)
        new_plevel = CpH()
        sparsity, nlinks = 0, 0
        for node in node_:
            graph.x0=min(graph.x0, node.x0); graph.xn=max(graph.xn, node.xn); graph.y0=min(graph.y0, node.y0); graph.yn=max(graph.yn, node.yn)
            # accum params:
            sum_pH(graph.plevels, node.plevels)  # same for fder
            for derG in node.link_:
                sum_pH(new_plevel, derG.plevels.H[fd])  # accum derG
                derG.roott[fd] = graph  # + link_ = [derG]?
                valt[fd] += derG.plevels.H[fd].val
                nlinks += 1
            node.plevels.val += valt[fd]  # derG valts summed in eval_med_layer added to lower-plevels valt
            graph.plevels.val += node.plevels.val

        graph.plevels.fds = copy(node.plevels.fds) + [fd]
        graph_ += [graph]
        graph.plevels.H += [new_plevel]

    # below not updated yet
    for graph in graph_:  # 2nd pass: accum alt_graph_ params
        Alt_plevels = []  # Alt_players if fsub
        for node in graph.node_:
            for derG in node.link_:
                for G in derG.node_:
                    if G not in graph.node_:  # alt graphs are roots of not-in-graph G in derG.node_
                        alt_graph = G.roott[fd]
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, Cgraph):  # not proto-graph
                            # der+: plevels[-1] += player, rng+: players[-1] = player
                            # der+ if fder else agg+:
                            if Alt_plevels: sum_pH(Alt_plevels, alt_graph.plevels[-1] if fderG else alt_graph.plevels)
                            else:           Alt_plevels = deepcopy(alt_graph.plevels[-1] if fderG else alt_graph.plevels)
                            graph.alt_graph_ += [alt_graph]
                            alt_graph.alt_graph_ += [graph]  # bilateral assign
        if graph.alt_graph_:
            graph.alt_graph_ += [Alt_plevels]  # temporary storage

    for graph in graph_:  # 3rd pass: add alt fork to each graph plevel, separate to fix nesting in 2nd pass
        if graph.alt_graph_:
            Alt_plevels = graph.alt_graph_.pop()
            graph.valt[:] = [sum(graph.valt), 0]  # all existing graph forks are relatively cis(internal) vs new alt
            if fder:
                add_alts(graph.plevels, Alt_plevels)  # derG, plevels are actually caForks, no loop plevels
            else:
                for cplevel, aplevel in zip(graph.plevels, Alt_plevels):  # G
                    graph.valt[1] += sum(aplevel[1])  # alt valt
                    add_alts(cplevel, aplevel)  # plevel is caForks: caTree leaves
    return graph_


def sum_pH(PH, pH, fneg=0):  # recursive unpack plevels ( ptuples ( ptuple, no accum across fd: matched in comp_pH

    PH.L += pH.L; PH.S += pH.S; PH.A += pH.A
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