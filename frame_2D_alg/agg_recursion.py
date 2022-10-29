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
ave_len = 5  # ave for etuple.L

class Cgraph(CPP):  # graph or generic PP of any composition

    plevels = list  # plevel_t[1]s is summed from alt_graph_, sub comp support, agg comp suppression?
    fds = list  # prior forks in plevels, then player fds in plevel
    valt = lambda: [0, 0]
    nvalt = lambda: [0, 0]  # from neg open links
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs; no alt_rdn: valt representation in alt_PP_ valts?
    rng = lambda: 1  # not for alt_graphs
    link_ = list  # all evaluated external graph links, nested in link_layers? open links replace alt_node
    meds_ = list  # last checked mediating nodes
    node_ = list  # graph elements, root of layers and levels:
    rlayers = list  # | mlayers, top-down
    dlayers = list  # | alayers
    mlevels = list  # agg_PPs ) agg_PPPs ) agg_PPPPs.., bottom-up
    dlevels = list
    roott = lambda: [None, None]  # higher-order segG or graph
    alt_graph_ = list


def agg_recursion(root, G_, fseg):  # compositional recursion in root.PP_, pretty sure we still need fseg, process should be different

    mgraph_, dgraph_ = form_graph_(root, G_, fder=0)  # PP cross-comp and clustering

    # intra graph:
    if root.valt[0] > ave_sub * root.rdn:
        sub_rlayers, rvalt = sub_recursion_g(mgraph_, root.valt, fd=0)  # subdivide graph.node_ by der+|rng+, accum root.valt
        root.valt[0] += sum(rvalt); root.rlayers = sub_rlayers
    if root.valt[1] > ave_sub * root.rdn:
        sub_dlayers, dvalt = sub_recursion_g(dgraph_, root.valt, fd=1)
        root.valt[1] += sum(dvalt); root.dlayers = sub_dlayers

    # cross graph:
    root.mlevels += mgraph_; root.dlevels += dgraph_
    for fd, graph_ in enumerate([mgraph_, dgraph_]):
        # agg+ if new plevel valt:
        if (sum(root.plevels[-1][1]) > G_aves[fd] * ave_agg * (root.rdn + 1)) and len(graph_) > ave_nsub:
            root.rdn += 1  # estimate
            agg_recursion(root, graph_, fseg=fseg)  # cross-comp graphs


def form_graph_(root, G_, fder):  # forms plevel in agg+ or player in sub+, G is potential node graph, in higher-order GG graph

    for G in G_:  # initialize mgraph, dgraph as roott per G, for comp_G_
        for i in 0,1:
            graph = [[G], [], [0,0]]  # proto-GG: [node_, meds_, valt]
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
            plevels = deepcopy(graph_[0].plevels); fds = graph_[0].fds  # same for all nodes?
            for graph in graph_[1:]:
                sum_plevels(plevels, graph.plevels, fds, graph.fds)  # each plevel is (caTree, valt)
            root.plevels = plevels

    return mgraph_, dgraph_


def comp_G_(G_, fder):  # cross-comp Gs (patterns of patterns): Gs, derGs, or segs inside PP

    for i, _G in enumerate(G_):

        for G in G_[i+1:]:  # compare each G to other Gs in rng, bilateral link assign
            if G in [node for link in _G.link_ for node in link.node_]:  # G,_G was compared in prior rng+, add frng to skip?
                continue
            _x = (_G.xn +_G.x0)/2; _y = (_G.yn +_G.y0)/2; x = (G.xn + G.x0)/2; y = (G.yn + G.y0)/2
            dx = _x - x; dy = _y - y  # distances
            distance = np.hypot(dy, dx)
            # Euclidean distance between centroids, max depends on combined G value, if fder: eval distance /= dangle?
            if distance <= ave_rng * ((sum(_G.valt)+sum(G.valt)) / (2*sum(G_aves))):

                extra = [[[_x,_y], len(_G.node_)],[[x,y], len(G.node_)]]  # if fder: comp angle, else comp len?
                mplevel, dplevel = comp_plevels(_G.plevels, G.plevels, _G.fds, G.fds, extra)
                valt = [mplevel[1] - ave_Gm, dplevel[1] - ave_Gd]
                # valt is normalized, *= link rdn?
                derG = Cgraph(  # or mean x0=_x+dx/2, y0=_y+dy/2:
                    plevels=[mplevel,dplevel], x0=min(G.x0,_G.x0), xn=max(G.xn,_G.xn), y0=min(G.y0,_G.y0), yn=max(G.yn,_G.yn), valt=valt, node_=[_G,G])
                _G.link_ += [derG]; G.link_ += [derG]  # any val
                for fd in 0,1:
                    if valt[fd] > 0:  # alt fork is redundant, no support?
                        for node, (graph, meds_, gvalt) in zip([_G, G], [G.roott[fd], _G.roott[fd]]):  # bilateral inclusion
                            if node not in graph:
                                graph += [node]
                                meds_ += [[derG.node_[0] if derG.node_[1] is node else derG.node_[1] for derG in node.link_]]  # immediate links
                                gvalt[0] += node.valt[0]; gvalt[1] += node.valt[1]


def eval_med_layer(graph_, graph, fd):   # recursive eval of reciprocal links from increasingly mediated nodes

    node_, meds_, valt = graph
    save_node_, save_meds_ = [], []
    adj_Val = 0  # adjust connect val in graph

    for G, med_node_ in zip(node_, meds_):  # G: node or sub-graph
        mmed_node_ = []  # __Gs that mediate between Gs and _Gs
        for _G in med_node_:
            for derG in _G.link_:
                if derG not in G.link_:  # link_ includes all unique evaluated mediated links, flat or in layers?
                    # med_PP.link_:
                    med_link_ = derG.node_[0].link_ if derG.node_[0] is not _G else derG.node_[1].link_
                    for _derG in med_link_:
                        if G in _derG.node_ and _derG not in G.link_:  # __G mediates between _G and G
                            G.link_ += [_derG]
                            adj_val = _derG.valt[fd] - ave_agg  # or increase ave per mediation depth
                            # adjust nodes:
                            G.valt[fd] += adj_val; _G.valt[fd] += adj_val  # valts not updated
                            valt[fd] += adj_val; _G.roott[fd][2][fd] += adj_val  # root is not graph yet
                            __G = _derG.node_[0] if _derG.node_[0] is not _G else _derG.node_[1]
                            if __G not in mmed_node_:  # not saved via prior _G
                                mmed_node_ += [__G]
                                adj_Val += adj_val
        if G.valt[fd]>0:
            # G remains in graph
            save_node_ += [G]; save_meds_ += [mmed_node_]  # mmed_node_ may be empty

    for G, mmed_ in zip(save_node_, save_meds_):  # eval graph merge after adjusting graph by mediating node layer
        add_mmed_= []
        for _G in mmed_:
            _graph = _G.roott[fd]
            if _graph in graph_ and _graph is not graph:  # was not removed or merged via prior _G
                _node_, _meds_, _valt = _graph
                for _node, _meds in zip(_node_, _meds_):  # merge graphs, ignore _med_? add direct links:
                    for derG in _node.link_:
                        __G = derG.node_[0] if derG.node_[0] is not _G else derG.node_[1]
                        if __G not in add_mmed_ + mmed_:  # not saved via prior _G
                            add_mmed_ += [__G]
                            adj_Val += derG.valt[fd] - ave_agg
                valt[fd] += _valt[fd]
                graph_.remove(_graph)
        mmed_ += add_mmed_

    graph[:] = [save_node_,save_meds_,valt]
    if adj_Val > ave_med:  # positive adj_Val from eval mmed_
        eval_med_layer(graph_, graph, fd)  # eval next med layer in reformed graph

'''
plevel = caForks, valt
caFork = players, valt, fds
player = caforks, valt  # each is der Ptuples of all lower players, per new agg span
cafork = ptuples, valt:
valSub = sum([ptuple.val for caFork in graph.plevels[-1][0] for player in caFork[0] for cafork in player[0] for ptuple in cafork[0]])  
'''
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
                            link.fds = [fd]
                            link.valt = link.plevels[fd][1]
                            link.plevels = [link.plevels[fd]]  # single-plevel plevels
                        node_ += [link]
        else: node_ = graph.node_
        # draft:
        valSub = graph.plevels[-1][1][fd]  # cis if mG else alt?
        # last player valt: ca = dm if fd else md: edge or core, or single player?
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


def sum2graph_(G_, fd, fder):  # sum node and link params into graph, plevel in agg+ or player in sub+: if fder

    for (node_, _, _) in G_:  # for both derG and G: add new level of valt
        for node in node_:
            val2valt(node.plevels)
            for link in node.link_:
                for fd in 0,1:
                    val2valt([link.plevels[fd]])
    graph_ = []
    for G in G_:
        node_, meds_, valt = G
        graph = Cgraph(fds=deepcopy(node_[0].fds), x0=node_[0].x0, xn=node_[0].xn, y0=node_[0].y0, yn=node_[0].yn, node_=node_, meds_=meds_)
        new_plevel = [[], [0, 0]]
        for node in node_:
            graph.x0=min(graph.x0, node.x0); graph.xn=max(graph.xn, node.xn); graph.y0=min(graph.y0, node.y0); graph.yn=max(graph.yn, node.yn)
            # accum params:
            sum_plevels(graph.plevels, node.plevels, graph.fds, node.fds)  # same for fsub
            for derG in node.link_:
                sum_plevel(new_plevel, derG.plevels[fd])  # accum derG, add to graph when complete
                derG.roott[fd] = graph  # link_ = [derG]?
                valt[fd] += derG.valt[fd]
            node.valt[0] += valt[fd]  # derG valts summed in eval_med_layer added to lower-plevels valt
            graph.valt[0]+= node.valt[0]
            graph.valt[1]+= node.valt[1]
        graph_ += [graph]
        graph.plevels += [new_plevel]

    for graph in graph_:  # 2nd pass: accum alt_graph_ params
        Alt_plevels = []  # Alt_players if fsub
        for node in graph.node_:
            for derG in node.link_:
                for G in derG.node_:
                    if G not in graph.node_:  # alt graphs are roots of not-in-graph G in derG.node_
                        alt_graph = G.roott[fd]
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, Cgraph):  # not proto-graph
                            # der+: plevels[-1] += player, rng+: players[-1] = player
                            if fder:
                                if Alt_plevels: sum_plevel(Alt_plevels, alt_graph.plevels[-1])  # same-length caTrees?
                                else:           Alt_plevels = deepcopy(alt_graph.plevels[-1])
                            else:  # agg+
                                if Alt_plevels: sum_plevels(Alt_plevels, alt_graph.plevels, alt_graph.fds, alt_graph.fds)  # redundant fds
                                else:           Alt_plevels = deepcopy(alt_graph.plevels)
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

def val2valt(plevels):
    for plevel in plevels:
        if not isinstance(plevel[1], list): plevel[1] = [plevel[1], 0]  # plevel valt
        for caFork in plevel[0]:
            if not isinstance(caFork[1], list): caFork[1] = [caFork[1], 0]  # caFork valt
            for player in caFork[0]:
                if not isinstance(player[1], list): player[1] = [player[1],0]  # player valt

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
'''
    plevel = caForks, valt
    caFork = players, valt, fds 
    player = caforks, valt
    cafork = ptuples, valt      
'''
def comp_plevels(_plevels, plevels, _fds, fds, extra):

    mplevel, dplevel = [],[]  # fd plevels, each cis+alt, same as new_caT
    mval, dval = 0,0  # m,d in new plevel, else c,a
    iVal = ave_G  # to start loop:

    for _plevel, plevel, _fd, fd in zip(reversed(_plevels), reversed(plevels), _fds, fds):  # caForks (caTree)
        if iVal < ave_G or _fd != fd:  # top-down, comp if higher plevels and fds match, same agg+
            break
        mTree, dTree = [],[]; mtval, dtval = 0,0  # Fork trees
        _caForks, _valt = _plevel; caForks, valt = plevel

        for _caFork, caFork in zip(_caForks, caForks):  # bottom-up alt+, pass-through fds
            mplayers, dplayers = [],[]; mlval, dlval = 0,0
            if _caFork and caFork:
                mplayer, dplayer = comp_players(_caFork, caFork, extra)
                mplayers += [mplayer]; dplayers += [dplayer]
                mlval += mplayer[1]; dlval += dplayer[1]
            else:
                mplayers += [[]]; dplayers += [[]]  # to align same-length trees for comp and sum
            # pack fds:
            mTree += [[mplayers, mlval, caFork[2]]]; dTree += [[dplayers, dlval, caFork[2]]]
            mtval += mlval; dtval += dlval
        # merge trees:
        mplevel += mTree; dplevel += mTree  # merge Trees in candidate plevels
        mval += mtval; dval += dtval
        iVal = mval+dval  # after 1st loop

    return [mplevel,mval], [dplevel,dval]  # always single new plevel

def comp_players(_caFork, caFork, extra):  # unpack and compare layers from der+

    mplayer, dplayer = [], []  # flat lists of ptuples, nesting decoded by mapping to lower levels
    mVal, dVal = 0,0  # m,d in new player, else c,a
    _players, _val, _fds =_caFork; players, val, fds = caFork

    for _player, player in zip(_players, players):
        mTree, dTree = [],[]; mtval, dtval = 0,0
        _caforks,_ = _player; caforks,_ = player

        for _cafork, cafork in zip(_caforks, caforks):  # bottom-up alt+, pass-through fds
            if _cafork and cafork:
                _ptuples,_ = _cafork; ptuples,_ = cafork
                if _ptuples and ptuples:
                    mtuples, dtuples, mval, dval = comp_ptuples(ptuples, ptuples, _fds, fds, extra)
                    mTree += [[mtuples, mval]]; dTree += [[dtuples, dval]]
                    mtval += mval; dtval += dval
                else:
                    mTree += [[]]; dTree += [[]]
            else:
                mTree += [[]]; dTree += [[]]
        # merge Trees:
        mplayer += mTree; dplayer += dTree
        mVal += mtval; dVal += dtval

    return [mplayer,mVal], [dplayer,dVal]  # single new lplayer

# draft
def comp_ptuples(_ptuples, ptuples, _fds, fds, extra):  # unpack and compare der layers, if any from der+

    mptuples, dptuples = [],[]; MVal, DVal = 0,0

    for _Ptuple, Ptuple, _fd, fd in zip(ptuples, ptuples, _fds, fds):  # bottom-up der+, pass-through fds
        if _fd == fd:
            mtuple, dtuple = comp_ptuple(_Ptuple[0], Ptuple[0])
            mext__, dext__ = [],[]; mVal, dVal = 0,0
            
            for _ext__, ext__ in zip(_Ptuple[1], Ptuple[1]):
                mext_, dext_ = [],[]; mval, dval = 0,0

                for _ext_, ext_ in zip(_ext__, ext__):
                    for _etuple, etuple in zip(_ext_, ext_):  # loop ders from prior comps in each lower ext_
                        metuple, detuple = comp_etuple(_etuple, etuple)
                        mext_ += [metuple]; mext_ += [detuple]
                        mval += metuple.val; dval += metuple.val

                mext__ += [mext_]; mext__ += [dext_]
                mVal += mval; dVal += dval
            mptuples += [mext__ + [Cptuple(angle=extra[0][0], L = extra[0][1])]]; MVal += mVal
            dptuples += [dext__ + [Cptuple(angle=extra[1][0], L = extra[1][1])]]; DVal += dVal
        else:
            break  # comp same fds

    return [mtuple, mptuples], [dtuple, dptuples], MVal, DVal  # mPtuple, dPtuple

# not revised:
def comp_etuple(_etuple, etuple):

    metuple, detuple = Cptuple(), Cptuple()

    comp("L", _etuple.L, etuple.L, detuple.val, metuple.val, detuple, metuple, ave_len, finv=0)

    if isinstance(_etuple. angle, list):
        _Dy,_Dx = _etuple.angle[:]; Dy,Dx = etuple.angle[:]
        _G = np.hypot(_Dy,_Dx); G = np.hypot(Dy,Dx)
        sin = Dy / (.1 if G == 0 else G); cos = Dx / (.1 if G == 0 else G)
        _sin = _Dy / (.1 if _G == 0 else _G); _cos = _Dx / (.1 if _G == 0 else _G)
        sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
        cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
        # dangle is scalar now?
        dangle = np.arctan2(sin_da, cos_da)  # vertical difference between angles
        mangle = ave_dangle - abs(dangle)  # inverse match, not redundant as summed
        detuple.angle = dangle; metuple.angle= mangle
        detuple.val += dangle; metuple.val += mangle
    else:
        comp("angle", _etuple.angle, etuple.angle, detuple.val, metuple.val, detuple, metuple, ave_dangle, finv=0)

    return metuple, detuple

def sum_plevels(pLevels, plevels, Fds, fds):

    for Plevel, plevel, Fd, fd in zip_longest(pLevels, plevels, Fds, fds, fillvalue=[]):  # loop top-down, selective comp depth, same agg+?
        if Fd==fd:
            if plevel:
                if Plevel: sum_plevel(Plevel, plevel)
                else:      pLevels.append(deepcopy(plevel))
        else:
            break

# separate to sum derGs
def sum_plevel(Plevel, plevel):

    CaForks, lValt = Plevel; caForks, lvalt = plevel  # plevels tree

    for CaFork, caFork in zip_longest(CaForks, caForks, fillvalue=[]):
        if caFork:
            if CaFork:
                Players, Fds, Valt = CaFork; players, fds, valt = caFork
                Valt[0] += valt[0]; Valt[1] += valt[1]
                lValt[0] += valt[0]; lValt[1] += valt[1]

                for Player, player in zip_longest(Players, players, fillvalue=[]):
                    if player:
                        if Player: sum_player(Player, player, Fds, fds, fneg=0)
                        else:      Players += [deepcopy(player)]
            else:
                CaForks += [deepcopy(caFork)]
                lvalt[0] += caFork[1][0]; lvalt[1] += caFork[1][1]

# draft
def sum_player(Player, player, Fds, fds, fneg=0):  # accum layers while same fds

    Caforks, Valt = Player; caforks, valt = player
    Valt[0] += valt[0]; Valt[1] += valt[1]

    for Cafork, cafork in zip(Caforks, caforks):
        if Cafork and cafork:
            Ptuples, Val = Cafork
            ptuples, val = cafork
            for Ptuple, ptuple, Fd, fd in zip_longest(Ptuples, ptuples, Fds, fds, fillvalue=[]):
                if Fd==fd:
                    if ptuple:
                        if Ptuple: sum_ptuple(Ptuple[0], ptuple[0], fneg=0)
                        else:      Ptuples += [deepcopy(ptuple)]
                        Val += val
                else:
                    break