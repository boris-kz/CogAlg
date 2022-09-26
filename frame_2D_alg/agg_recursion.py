import sys
import numpy as np
from itertools import zip_longest
from copy import deepcopy, copy
from class_cluster import ClusterStructure, NoneType, comp_param, Cdert
import math as math
from comp_slice import *

'''
Blob edges may be represented by higher-composition PPPs, etc., if top param-layer match,
in combination with spliced lower-composition PPs, etc, if only lower param-layers match.
This may form closed edge patterns around flat core blobs, which defines stable objects.   
'''

ave_G = 6  # fixed costs per G
ave_Gm = 5  # for inclusion in graph
ave_Gd = 4
ave_med = 3  # call cluster_node_layer

class CderG(ClusterStructure):  # tuple of graph derivatives in graph link_

    plevel_t = lambda: [[],[]]  # lists of ptuples in current derivation layer per fork
    valt = lambda: [0,0]  # tuple of mval, dval
    alt_plevel_t = lambda: [[],[]]  # ders of summed adjacent alt-fork G params
    alt_valt = lambda: [0, 0]
    # below are not needed?
    box = list  # 2nd level, stay in PP?
    rdn = int  # mrdn, + uprdn if branch overlap?
    PPt = list  # PP,_PP comparands, the order is not fixed
    roott = lambda: [None, None]  # PP, not segment in sub_recursion
    uplink_layers = lambda: [[], []]  # init a layer of dderPs and a layer of match_dderPs
    downlink_layers = lambda: [[], []]
    # from comp_dx
    fdx = NoneType

class Cgraph(CPP, CderG):  # graph or generic PP of any composition

    plevels = list  # max n ptuples / level = n ptuples in all lower layers: 1, 1, 2, 4, 8...
    # plevel_t[1]s are from alt-fork graphs, support for sub comp, suppression for agg comp?
    valt = lambda: [0, 0]  # mval, dval from cis+alt forks
    nvalt = lambda: [0, 0]  # from neg (open) links
    fds = list  # prior fork sequence, map to mplayer or dplayer in lower (not parallel) player
    # or per plevel?
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs
    Rdn = int  # for accumulation only; no alt_rdn: valt representation in alt_PP_ valts?
    rng = lambda: 1  # rng starts with 1, not for alt_PPs
    box = list
    link_plevel_t = lambda: [[], []]  # one plevel, temporary accum per fork,
    # or in der_G only?
    link_ = list  # all evaluated external graph links, nested in link_layers? open links replace alt_node_
    node_ = list  # graph elements, root of layers and levels:
    rlayers = list  # | mlayers, top-down
    dlayers = list  # | alayers
    mlevels = list  # agg_PPs ) agg_PPPs ) agg_PPPPs.., bottom-up
    dlevels = list
    roott = lambda: [None, None]  # higher-order segG or graph
    # cPP_ = list  # co-refs in other PPPs


def agg_recursion(root, PP_, rng, fseg):  # compositional recursion in root.PP_, pretty sure we still need fseg, process should be different

    mgraph_, dgraph_ = form_graph_(root, PP_, rng, fd=1)  # PP cross-comp and clustering

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
        val = root.valt[fd]
        if (val > PP_aves[fd] * ave_agg * (root.rdn + 1)) and len(graph_) > ave_nsub:
            root.rdn += 1  # estimate
            agg_recursion(root, graph_, rng=val / ave_agg, fseg=fseg)  # cross-comp graphs


def form_graph_(root, PP_, rng, fd=1):

    for PP in PP_:  # initialize mgraph, dgraph as roott per PP, for comp_PP_
        for i in 0,1:
            graph = [[PP],[0,0]]  # [node_, valt]
            PP.roott[i] = graph

    comp_graph_(PP_, rng, fd)  # cross-comp all PPs within rng, PPs may be segs
    mgraph_, dgraph_ = [],[]  # initialize graphs with >0 positive links in PP roots:
    for PP in PP_:
        if len(PP.roott[0][0])>1: mgraph_ += [PP.roott[0]]  # root = [node_, valt] for cluster_node_layer eval, + link_nvalt?
        if len(PP.roott[1][0])>1: dgraph_ += [PP.roott[1]]

    for fd, graph_ in enumerate([mgraph_, dgraph_]):  # evaluate intermediate nodes to delete or merge graphs:
        regraph_ = []
        while graph_:
            graph = graph_.pop(0)
            med_node__ = [[derPP.PPt[1] if derPP.PPt[0] is PP else derPP.PPt[0] for derPP in PP.link_]for PP in graph[0]]
            ''' unpacked:
            med_node__ = []
            for PP in graph[0]:
                med_node_ = []
                for derPP in PP.link_:
                    if derPP.PPt[0] is PP:
                       med_node_ += [derPP.PPt[1]]
                    else:
                       med_node_ += [derPP.PPt[0]]
                med_node__ += [med_node_] 
                '''
            cluster_node_layer(graph_= graph_, graph=graph, med_node__= med_node__, fd=fd)
            if graph[1][fd] > ave_agg: regraph_ += [graph]  # graph reformed by merges and deletions in cluster_node_layer

        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum node_ params in graph
            # accum root plevels:
            plevels = deepcopy(graph_[0].plevels)  # each level is plevel_t: plevel, alt_plevel
            for graph in graph_[1:]:
                sum_plevel_ts(plevels, graph.plevels)
            root.plevels = plevels

    return mgraph_, dgraph_


def comp_graph_(PP_, rng, fd):  # cross-comp, same val,rng for both forks? PPs may be segs inside a PP

    for i, _PP in enumerate(PP_):  # compare patterns of patterns: _PP to other PPs in rng, bilateral link assign:
        for PP in PP_[i+1:]:

            area = PP.plevels[0][PP.fds[-1]][0][0][0].L; _area = _PP.plevels[0][_PP.fds[-1]][0][0][0].L
            # not revised: 1st plevel_t' fork' players' 1st player' 1st ptuple' L
            dx = ((_PP.xn - _PP.x0) / 2) / _area - ((PP.xn - PP.x0) / 2) / area
            dy = _PP.y / _area - PP.y / area
            distance = np.hypot(dy, dx)  # Euclidean distance between PP centroids
            # fork_rdnt = [1 + (root.valt[1] > root.valt[0]), 1 + (root.valt[0] >= root.valt[1])]  # not per graph_?

            val = _PP.valt[fd] / PP_aves[fd]  # rng+|der+ fork eval, init der+: single level
            if distance * val <= rng:
                # comp PPs, comb core,edge PP?
                mplevel_t, dplevel_t, mval_t, dval_t = comp_plevel_ts(_PP.plevels, PP.plevels)
                valt = [sum(mval_t) - ave_Gm, sum(dval_t) - ave_Gd]  # adjust for link rdn?
                derPP = CderG(plevel_t=[mplevel_t, dplevel_t], valt=valt, PPt=[PP,_PP])
                # any val:
                _PP.link_ += [derPP]
                PP.link_ += [derPP]
                for fd in 0,1:
                    if valt[fd] > 0:  # no alt fork support?
                        for node, (graph, gvalt) in zip([_PP, PP], [PP.roott[fd], _PP.roott[fd]]):  # bilateral inclusion
                            if node not in graph:
                                graph += [node]
                                gvalt[0] += node.valt[0]; gvalt[1] += node.valt[1]


def cluster_node_layer(graph_, graph, med_node__, fd):  # recursive eval of mutual links in increasingly mediated nodes

    node_, valt = graph
    save_node_, save_med__ = [],[]  # __PPs mediating between PPs and _PPs
    adj_Val = 0

    for PP, med_node_ in zip(node_, med_node__):
        save_med_ = []
        for _PP in med_node_:
            for derPP in _PP.link_:
                if derPP not in PP.link_:  # link_ includes all unique evaluated mediated links, flat or in layers?
                    # med_PP.link_:
                    med_link_ = derPP.PPt[0].link_ if derPP.PPt[0] is not _PP else derPP.PPt[1].link_
                    for _derPP in med_link_:
                        if PP in _derPP.PPt and _derPP not in PP.link_:  # __PP mediates between _PP and PP
                            PP.link_ += [_derPP]
                            adj_val = _derPP.valt[fd] - ave_agg  # or increase ave per mediation depth
                            # adjust nodes:
                            PP.valt[fd] += adj_val; _PP.valt[fd] += adj_val
                            valt[fd] += adj_val; _PP.roott[fd][0][fd] += adj_val  # root is not graph yet
                            # or batch recompute?
                            __PP = _derPP.PPt[0] if _derPP.PPt[0] is not _PP else _derPP.PPt[1]
                            if __PP not in save_med_:  # if not saved via prior _PP
                                save_med_ += [__PP]
                                adj_Val += adj_val  # save_med__ val?
        if PP.valt[fd]>0:
            # PP is connected enough to remain in the graph
            save_node_ += [PP]; save_med__ += [save_med_]  # save_med__ is nested, may be empty

    # re-eval full graph after adjusting it with mediating node layer:
    add_node_, add_med__ = [], []
    for PP, _PP_ in zip(save_node_, save_med__):
        for _PP in _PP_:
            _graph = _PP.roott[fd]
            if _graph in graph_ and _graph is not graph:  # was not removed or merged via prior _PP
                _node_, _valt = _graph
                for _node in _node_:  # merge graphs, add direct links:
                    add_node_ += [_node]
                    add_med__.append([derPP.PPt[1] if derPP.PPt[0] is _node else derPP.PPt[0] for derPP in _node.link_])
                valt[fd] += _valt[fd]
                graph_.remove(_graph)

    if valt[fd] > ave_G:  # adjusted graph value
        node_[:] = save_node_ + add_node_  # reassign as save_node_ (>0 after mediation) + add_node_ (mediated positive link nodes)
        med_node__[:] = save_med__ + add_med__

        if adj_Val > ave_med:  # eval next mediation layer in reformed graph: extra ops, no rdn+?
            cluster_node_layer(graph_, graph, med_node__, fd)


def sub_recursion_g(graph_, fseg, fd):  # rng+: extend PP_ per PPP, der+: replace PP with derPP in PPt

    comb_layers_t = [[],[]]
    sub_valt = [0,0]

    for graph in graph_:
        if graph.valt[fd] > PP_aves[fd] and len(graph.node_) > ave_nsub:

            sub_mgraph_, sub_dgraph_ = form_graph_(graph, graph.node_, graph.rng, fd)  # cross-comp and clustering cycle

            if graph.valt[0] > ave_sub * graph.rdn:  # rng +:
                sub_rlayers, valt = sub_recursion_g(sub_mgraph_, graph.valt, fd=0)
                rvalt = sum(valt); graph.valt[0] += rvalt; sub_valt[0] += rvalt  # not sure
                graph.rlayers = [sub_mgraph_] + [sub_rlayers]
            # if > cost of calling sub_recursion and looping:
            if graph.valt[1] > ave_sub * graph.rdn:  # der+:
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


def sum2graph_(igraph_, fd):  # sum node params into graph

    graph_ = []
    for igraph in igraph_:
        node_, valt = igraph
        # draft:
        graph = Cgraph( node_=node_, plevels=[], alt_plevels =[], link_valt=valt, fds=deepcopy(node_[0].fds),
                        rng=node_[0].rng+1, x0=node_[0].x0, xn=node_[0].xn, y0=node_[0].y0, yn=node_[0].yn)

        new_players_t, new_alt_players_t = [[],[]], [[],[]]
        new_val_t, new_alt_val_t = [[0, 0], [0, 0]], [[0, 0], [0, 0]]
        for node in node_:
            graph.valt[0] += node.valt[0]; graph.valt[1] += node.valt[1]

            graph.x0=min(graph.x0, node.x0)
            graph.xn=max(graph.xn, node.xn)
            graph.y0=min(graph.y0, node.y0)
            graph.yn=max(graph.yn, node.yn)
            # accum params:
            if not graph.plevels:  # empty plevels
                graph.plevels = deepcopy(node.plevels)
                graph.alt_plevels = deepcopy(node.alt_plevels)
            else:
                sum_plevel_ts (graph.plevels, node.plevels)
                sum_plevel_ts(graph.alt_plevels, node.alt_plevels)

            # below is draft
            for derG in node.link_:  # accum derG in new level
                sum_player_t(new_players_t, derG.plevel_t, fd)
                new_val_t[fd][0] += derG.valt[0]; new_val_t[fd][1] += derG.valt[1]
                new_alt_val_t[fd][0] += derG.alt_valt[0]; new_alt_val_t[fd][1] += derG.alt_valt[1]

        new_fds_t = [deepcopy(graph.plevels[-1][fd][1]) + [fd], deepcopy(graph.plevels[-1][fd][1]) + [1-fd]]
        # if alt_plvels is empty, their fds will be empty too?
        new_alt_fds_t = [deepcopy(graph.alt_plevels[-1][fd][1]) + [1-fd], deepcopy(graph.alt_plevels[-1][fd][1]) + [1-fd]]


        new_plevel_t = [[[new_players_t[0]], new_fds_t[0], new_val_t[0]], \
                        [[new_players_t[1]], new_fds_t[1], new_val_t[1]]]
        new_alt_plevel_t = [[[new_alt_players_t[0]], new_alt_fds_t[0], new_alt_val_t[0]], \
                            [[new_alt_players_t[1]], new_alt_fds_t[1], new_alt_val_t[1]]]

        # pack new level
        if any(new_players_t): graph.plevels += [new_plevel_t]
        if any(new_alt_players_t): graph.alt_plevels += [new_alt_plevel_t]

        graph_ += [graph]
    return graph_


def comp_plevel_ts(_plevels, plevels):

    mplevel_t, dplevel_t = [[],[]], [[],[]]
    mVal, dVal = 0,0

    for _plevel_t, plevel_t in zip(_plevels, plevels):
        for alt, (_plevel, plevel) in enumerate(zip(_plevel_t, plevel_t)):
            # cis | alt fork:
            if _plevel and plevel:
                _players, _fds, _valt = _plevel
                players, fds, valt = plevel
                mplayer, dplayer, mval, dval = comp_players(_players, players, _fds, fds)
                mplevel_t[alt] += [mplayer]; dplevel_t[alt] += [dplayer]
                mVal += mval; dVal += dval

    return mplevel_t, dplevel_t, mVal, dVal

def comp_plevels(_plevels, plevels, _fds, fds):

    mlevel, dlevel = [], []  # flat lists of ptuples, nesting decoded by mapping to lower levels
    mVal, dVal = 0,0

    for (_plevel, _lfds, _valt), (plevel, lfds, valt), _fd, fd in zip(_plevels, plevels, _fds, fds):
        if _fd==fd:
            mplayer, dplayer, mval, dval = comp_players(_plevel, plevel, _lfds, lfds)
            mlevel += mplayer; mVal += mval
            dlevel += dplayer; dVal += dval
        else:
            break  # only same-fd players are compared

    return mlevel, dlevel, mVal, dVal

def comp_players(_layers, layers, _fds, fds):  # unpack and compare der layers, if any from der+

    mplayer, dplayer = [], []  # flat lists of ptuples, nesting decoded by mapping to lower levels
    mval, dval = 0, 0

    for _player, player, _fd, fd in zip(_layers, layers, _fds, fds):
        if _fd==fd:
            for _ptuple, ptuple in zip(_player, player):
                mtuple, dtuple = comp_ptuple(_ptuple, ptuple)
                mplayer += [mtuple]; mval = mtuple.val
                dplayer += [dtuple]; dval = dtuple.val
        else:
            break  # only same-fd players are compared

    return mplayer, dplayer, mval, dval

def sum_plevel_ts(pLevels, plevels):

    for pLevel_t, plevel_t in zip_longest(pLevels, plevels, fillvalue=[]):
        for pLevel, plevel in zip_longest(pLevel_t, plevel_t, fillvalue=[]):
            # cis | alt forks
            if plevel and plevel[0]:
                if pLevel:
                    if pLevel[0]: sum_players(pLevel[0], plevel[0], pLevel[1], plevel[1])  # accum nodes' players
                    else:         pLevel[0] = deepcopy(plevel[0])  # append node's players
                    pLevel[1] = deepcopy(plevel[1])  # assign fds
                    pLevel[2][0] += plevel[2][0]
                    pLevel[2][1] += plevel[2][1]  # accumulate valt
                else:
                    pLevel.append(deepcopy(plevel))  # pack new plevel


def sum_player_t(pLayer_t, player_t, fd):

    for pLayer, player in zip_longest(pLayer_t, player_t, fillvalue=[]):
        if player[fd] and player[fd][0]:
            if pLayer:
                for ppLayer,pplayer in zip(pLayer, player[fd]):
                    sum_player(ppLayer,pplayer)
            else:
                pLayer += player[fd]  # pack new player

def sum_plevels(pLevels, plevels):

    for pLevel, plevel in zip_longest(pLevels, plevels, fillvalue=[]):
        if plevel and plevel[0]:
            if pLevel:
                if pLevel[0]: sum_players(pLevel[0], plevel[0], pLevel[1], plevel[1])  # accum nodes' players
                else:         pLevel[0] = deepcopy(plevel[0])  # append node's players
                pLevel[1] = deepcopy(plevel[1])  # assign fds
                pLevel[2][0] += plevel[2][0]; pLevel[2][1] += plevel[2][1]  # accumulate valt
            else:
                pLevels.append(deepcopy(plevel))  # pack new plevel

def sum_players(Layers, layers, Fds, fds, fneg=0):  # accum layers while same fds

    fbreak = 0
    for i, (Layer, layer, Fd, fd) in enumerate(zip(Layers, layers, Fds, fds)):
        if Fd==fd:
            sum_player(Layer, layer, fneg=fneg)
        else:
            fbreak = 1
            break
    Fds[:] = Fds[:i+1-fbreak]


# not revised, this is an alternative to form_graph, but may not be accurate enough to cluster:
def comp_centroid(PPP_):  # comp PP to average PP in PPP, sum >ave PPs into new centroid, recursion while update>ave

    update_val = 0  # update val, terminate recursion if low

    for PPP in PPP_:
        PPP_valt = [0 ,0]  # new total, may delete PPP
        PPP_rdn = 0  # rdn of PPs to cPPs in other PPPs
        PPP_players_t = [[], []]
        DerPP = CderG(player=[[], []])  # summed across PP_:
        Valt = [0, 0]  # mval, dval

        for i, (PP, _, fint) in enumerate(PPP.PP_):  # comp PP to PPP centroid, derPP is replaced, use comp_plevels?
            Mplayer, Dplayer = [],[]
            # both PP core and edge are compared to PPP core, results are summed or concatenated:
            for fd in 0, 1:
                if PP.players_t[fd]:  # PP.players_t[1] may be empty
                    mplayer, dplayer = comp_players(PPP.players_t[0], PP.players_t[fd], PPP.fds, PP.fds)  # params norm in comp_ptuple
                    player_t = [Mplayer + mplayer, Dplayer + dplayer]
                    valt = [sum([mtuple.val for mtuple in mplayer]), sum([dtuple.val for dtuple in dplayer])]
                    Valt[0] += valt[0]; Valt[1] += valt[1]  # accumulate mval and dval
                    # accum DerPP:
                    for Ptuple, ptuple in zip_longest(DerPP.player_t[fd], player_t[fd], fillvalue=[]):
                        if ptuple:
                            if not Ptuple: DerPP.player_t[fd].append(ptuple)  # pack new layer
                            else:          sum_players([[Ptuple]], [[ptuple]])
                    DerPP.valt[fd] += valt[fd]

            # compute rdn:
            cPP_ = PP.cPP_  # sort by derPP value:
            cPP_ = sorted(cPP_, key=lambda cPP: sum(cPP[1].valt), reverse=True)
            rdn = 1
            fint = [0, 0]
            for fd in 0, 1:  # sum players per fork
                for (cPP, CderG, cfint) in cPP_:
                    if valt[fd] > PP_aves[fd] and PP.players_t[fd]:
                        fint[fd] = 1  # PPs match, sum derPP in both PPP and _PPP:
                        sum_players(PPP.players_t[fd], PP.players_t[fd])
                        sum_players(PPP.players_t[fd], PP.players_t[fd])  # all PP.players in each PPP.players

                    if CderG.valt[fd] > Valt[fd]:  # cPP is instance of PP
                        if cfint[fd]: PPP_rdn += 1  # n of cPPs redundant to PP, if included and >val
                    else:
                        break  # cPP_ is sorted by value

            fnegm = Valt[0] < PP_aves[0] * rdn;  fnegd = Valt[1] < PP_aves[1] * rdn  # rdn per PP
            for fd, fneg, in zip([0, 1], [fnegm, fnegd]):

                if (fneg and fint[fd]) or (not fneg and not fint[fd]):  # re-clustering: exclude included or include excluded PP
                    PPP.PP_[i][2][fd] = 1 -  PPP.PP_[i][2][fd]  # reverse 1-0 or 0-1
                    update_val += abs(Valt[fd])  # or sum abs mparams?
                if not fneg:
                    PPP_valt[fd] += Valt[fd]
                    PPP_rdn += 1  # not sure
                if fint[fd]:
                    # include PP in PPP:
                    if PPP_players_t[fd]: sum_players(PPP_players_t[fd], PP.players_t[fd])
                    else: PPP_players_t[fd] = copy(PP.players_t[fd])  # initialization is simpler
                    # not revised:
                    PPP.PP_[i][1] = derPP   # no derPP now?
                    for i, cPPt in enumerate(PP.cPP_):
                        cPPP = cPPt[0].root
                        for j, PPt in enumerate(cPPP.cPP_):  # get PPP and replace their derPP
                            if PPt[0] is PP:
                                cPPP.cPP_[j][1] = derPP
                        if cPPt[0] is PP: # replace cPP's derPP
                            PPP.cPP_[i][1] = derPP
                PPP.valt[fd] = PPP_valt[fd]

        if PPP_players_t: PPP.players_t = PPP_players_t

        # not revised:
        if PPP_val < PP_aves[fPd] * PPP_rdn:  # ave rdn-adjusted value per cost of PPP

            update_val += abs(PPP_val)  # or sum abs mparams?
            PPP_.remove(PPP)  # PPPs are hugely redundant, need to be pruned

            for (PP, derPP, fin) in PPP.PP_:  # remove refs to local copy of PP in other PPPs
                for (cPP, _, _) in PP.cPP_:
                    for i, (ccPP, _, _) in enumerate(cPP.cPP_):  # ref of ref
                        if ccPP is PP:
                            cPP.cPP_.pop(i)  # remove ccPP tuple
                            break

    if update_val > sum(PP_aves):
        comp_centroid(PPP_)  # recursion while min update value

    return PPP_
'''
    1st and 2nd layers are single sublayers, the 2nd adds tuple pair nesting. Both are unpacked by func_pairs, not func_layers.  
    Multiple sublayers start on the 3rd layer, because it's derived from comparison between two (not one) lower layers. 
    4th layer is derived from comparison between 3 lower layers, where the 3rd layer is already nested, etc:
    initial 3-layer nesting: https://github.com/assets/52521979/ea6d436a-6c5e-429f-a152-ec89e715ebd6
    '''

# the below is out of date:
# draft, splice 2 PPs for now
def splice_PPs(PP_, frng):  # splice select PP pairs if der+ or triplets if rng+

    spliced_PP_ = []
    while PP_:
        _PP = PP_.pop(0)  # pop PP, so that we can differentiate between tested and untested PPs
        tested_segs = []  # new segs may be added during splicing, their links also need to be checked for splicing
        _segs = _PP.seg_levels[0]

        while _segs:
            _seg = _segs.pop(0)
            _avg_y = sum([P.y for P in _seg.P__]) / len(_seg.P__)  # y centroid for _seg

            for link in _seg.uplink_layers[1] + _seg.downlink_layers[1]:
                seg = link.P.root  # missing link of current seg

                if seg.root is not _PP:  # after merging multiple links may have the same PP
                    avg_y = sum([P.y for P in seg.P__]) / len(seg.P__)  # y centroid for seg

                    # test for y distance (temporary)
                    if (_avg_y - avg_y) < ave_splice:
                        if seg.root in PP_:
                            PP_.remove(seg.root)  # remove merged PP
                        elif seg.root in spliced_PP_:
                            spliced_PP_.remove(seg.root)
                        # splice _seg's PP with seg's PP
                        merge_PP(_PP, seg.root)

            tested_segs += [_seg]  # pack tested _seg
        _PP.seg_levels[0] = tested_segs
        spliced_PP_ += [_PP]

    return spliced_PP_


def merge_PP(_PP, PP, fPd):  # only for PP splicing

    for seg in PP.seg_levels[fPd][-1]:  # merge PP_segs into _PP:
        accum_PP(_PP, seg, fPd)
        _PP.seg_levels[fPd][-1] += [seg]

    # merge uplinks and downlinks
    for uplink in PP.uplink_layers:
        if uplink not in _PP.uplink_layers:
            _PP.uplink_layers += [uplink]
    for downlink in PP.downlink_layers:
        if downlink not in _PP.downlink_layers:
            _PP.down_layers += [downlink]