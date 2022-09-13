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

class CderG(ClusterStructure):  # tuple of graph derivatives in graph link_

    plevel_t = lambda: [[],[]]  # lists of ptuples in current derivation layer per fork
    valt = lambda: [0,0]  # tuple of mval, dval
    alt_plevel_t = lambda: [[],[]]  # ders of summed adjacent alt-fork G params
    alt_valt = lambda: [0, 0]
    # below are not needed?
    box = list  # 2nd level, stay in PP?
    rdn = int  # mrdn, + uprdn if branch overlap?
    _PP = object  # prior comparand  or always in PP_?
    PP = object  # next comparand
    root = lambda: None  # PP, not segment in sub_recursion
    uplink_layers = lambda: [[], []]  # init a layer of dderPs and a layer of match_dderPs
    downlink_layers = lambda: [[], []]
    # from comp_dx
    fdx = NoneType

class Cgraph(CPP, CderG):  # graph or generic PP of any composition

    alt_PP_ = list  # adjacent alt-fork gPPs, cross-support for sub, cross-suppression for agg?
    plevels = list  # max n ptuples / level = n ptuples in all lower layers: 1, 1, 2, 4, 8...
    alt_plevels = list
    valt = lambda: [0, 0]  # mval, dval
    nvalt = lambda: [0, 0]  # neg links
    alt_valt = lambda: [0, 0]  # mval, dval; no nvalt?
    fds = list  # prior fork sequence, map to mplayer or dplayer in lower (not parallel) player, none for players[0]
    alt_fds = list
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs
    Rdn = int  # for accumulation only
    alt_rdn = int  # valt representation in alt_PP_ valts
    rng = lambda: 1  # rng starts with 1, not for alt_PPs
    box = list
    link_plevel_t = lambda: [[], []]  # one plevel, accum per fork
    link_ = list  # lateral gPP connections: (PP, derPP, fint)_, + deeper layers?
    node_ = list  # gPP elements, root of layers and levels:
    rlayers = list  # | mlayers, top-down
    dlayers = list  # | alayers
    mlevels = list  # agg_PPs ) agg_PPPs ) agg_PPPPs.., bottom-up
    dlevels = list
    roott = lambda: [None, None]  # lambda: Cgraph()  # higher-order segP or PPP
    # cPP_ = list  # co-refs in other PPPs


def agg_recursion(root, PP_, rng, fseg=0):  # compositional recursion per blob.Plevel; P, PP, PPP are relative to each other

    mgraph_, dgraph_ = form_graph_(PP_, rng, fseg)  # PP cross-comp and clustering

    # intra graph:
    if sum(root.valt) > ave_agg * root.rdn:
        sub_rlayers, rvalt = sub_recursion_agg(mgraph_, root.valt, fd=0)  # subdivide graph.node_ by der+|rng+, accum root.valt
        root.valt[0] += sum(rvalt); root.rlayers = sub_rlayers
        sub_dlayers, dvalt = sub_recursion_agg(dgraph_, root.valt, fd=1)
        root.valt[1] += sum(dvalt); root.dlayers = sub_dlayers

    # cross graph:
    if fseg: root.mseg_levels += mgraph_; root.dseg_levels += dgraph_  # add bottom-up
    else:    root.mlevels += mgraph_; root.dlevels += dgraph_

    for fd, graph_ in enumerate([mgraph_, dgraph_]):
        val = root.valt[fd]
        if (val > PP_aves[fd] * ave_agg * (root.rdn + 1)) and len(graph_) > ave_nsub:
            root.rdn += 1  # estimate, replace by actual after agg_recursion?
            # cross-comp graphs:
            agg_recursion(root, graph_, rng=val / ave_agg, fseg=fseg)


def form_graph_(PP_, rng, fseg):

    for PP in PP_:  # initialize mgraph, dgraph as roott per PP, for comp_PP_
        for fd in 0,1:
            graph = [[PP],[0,0]]; PP.roott[fd] = graph  # [node_, valt]

    mgraph_, dgraph_ = comp_PP_(PP_, rng, fseg)
    # cross-comp all PPs within rng (PPs may be segs), compose graphs from +ve PP links

    graph_t = []
    for fd, igraph_ in enumerate([mgraph_, dgraph_]):
        graph_ = copy(igraph_)  # copy for popping
        while graph_:
            graph, link_valt = graph_.pop(0)  # eval intermediate nodes to extend / prune / merge graphs:
            cluster_node_layer(graph_= igraph_, graph=graph, node_ = graph[0], shared_Val=graph[1][fd] + link_valt[fd], fid=fd)

        graph_t.append(sum2graph_(igraph_, fd))  # sum node_ params into graph

    return graph_t


def comp_PP_(PP_, rng, fseg):  # cross-comp, same val,rng for both forks? PPs may be segs inside a PP

    graph_t = []
    for fd in 0,1:
        graph_ = []
        for i, _PP in enumerate(PP_):  # comp _PP to all other PPs in rng, bilateral assign:
            for PP in PP_[i+1:]:

                area = PP.plevels[0][0][0][0].L; _area = _PP.plevels[0][0][0][0].L  # 1st plevel' players' 1st player' 1st ptuple' L
                dx = ((_PP.xn - _PP.x0) / 2) / _area - ((PP.xn - PP.x0) / 2) / area
                dy = _PP.y / _area - PP.y / area
                distance = np.hypot(dy, dx)  # Euclidean distance between PP centroids
                # fork_rdnt = [1 + (root.valt[1] > root.valt[0]), 1 + (root.valt[0] >= root.valt[1])]  # not per graph_?

                val = _PP.valt[fd] / PP_aves[fd]  # no complimented val: cross-fork support if spread spectrum?
                if distance * val <= rng:
                    # comp PPs:
                    mplevel, dplevel = comp_plevels(_PP.plevels, PP.plevels, _PP.fds, PP.fds)
                    # players = plevel[0], draft:
                    valt = [sum([mtuple.val for plevel in mplevel[0] for player in plevel[0] for mtuple in player]),
                            sum([dtuple.val for plevel in dplevel[0] for player in plevel[0] for dtuple in player])]
                    derPP = CderG(plevel_t=[mplevel, dplevel], valt=valt)  # single-level
                    # add comp same-fds altPPs here
                    fint = []
                    for fdd in 0,1:  # sum players per fork
                        if valt[fdd] > PP_aves[fdd]:  # no cross-fork support?
                            fin = 1
                            for node, (graph, gvalt) in zip([_PP, PP], [PP.roott[fd], _PP.roott[fd]]):  # bilateral inclusion
                                graph += [node]
                                gvalt[0] += node.valt[0]; gvalt[1] += node.valt[1]
                        else:
                            fin = 0
                        fint += [fin]
                    _PP.link_ += [[PP, derPP, fint]]
                    PP.link_ += [[_PP, derPP, fint]]

            if len(_PP.roott[fd][0])>1:  # only positively linked PPs are stored in graphs
                graph_ += [_PP.roott[fd]]  # root = [node_, valt] for cluster_node_layer eval, add link_nvalt?

        graph_t += [graph_]
    return graph_t


def cluster_node_layer(graph_, graph, node_, shared_Val, fid):  # recursive eval of mutual links in increasingly mediated nodes

    for PP in node_:  # graph node_
        for (_PP, _derPP, _fint) in PP.link_:
            for (__PP, __derPP, __fint) in _PP.link_:
                for (___PP, ___derPP, ___fint) in __PP.link_:

                    if ___PP is PP:  # __PP is connected to PP and _PP
                        shared_Val += ___derPP.valt[fid]  # graph is reinforced by mediated matches
                        # if shared_Val > 0: cluster_node_layer?
            # not revised
            _graph = _PP.roott[fid]
            _node_, _valt = _graph
            if _graph is not graph:
                shared_Val += _derPP.valt[fid] - ave_agg  # * rdn: sum PP_ rdns / len PP_ + graphs overlap rate?
                if shared_Val > 0:  # merge graphs if mediating nodes match
                    for _node in _node_:  # merge node
                        if _node not in node_:
                            node_ += [_node]

                    graph_list = [graph for graph in graph_]  # get list of graph only, graph_ contain multiple [graph, valt] now
                    if _graph in graph_list:
                        graph_.pop(graph_list.index(_graph)) # remove graph from graph_
                        graph_list.remove(_graph)

                    # graph may be removed in prior merging, since graph may have multiple nodes
                    # recursively intermediated search for mutual connections
                    for __PP,_,_ in _PP.link_:
                        __graph = __PP.roott[fid]
                        if __graph is not graph and __graph in graph_list:  # graph is not graph and not merged in prior scan
                            cluster_node_layer(graph_, graph, __graph[0], shared_Val, fid)

# not reviewed:
def sub_recursion_agg(graph_, fseg, fd):  # rng+: extend PP_ per PPP, der+: replace PP with derPP in PPt

    comb_layers_t = [[],[]]
    sub_valt = [0,0]

    for graph in graph_:
        if graph.valt[fd] > PP_aves[fd] and len(graph.gPP_) > ave_nsub:

            sub_graph_t = form_graph_(graph.node_, graph.rng, fseg)
            if sum(graph.valt) > ave_agg * graph.rdn:

                sub_rlayers, valt = sub_recursion_agg(sub_graph_t[0], graph.valt, fd=0)
                rvalt = sum(valt); graph.valt[0] += rvalt; sub_valt[0] += rvalt  # not sure
                sub_dlayers, valt = sub_recursion_agg(sub_graph_t[1], graph.valt, fd=1)
                dvalt = sum(valt); graph.valt[1] += dvalt; sub_valt[1] += dvalt
                graph.rlayers = sub_rlayers; graph.dlayers = sub_dlayers

                for comb_layers, graph_layers in zip(comb_layers_t, [graph.rlayers, graph.dlayers]):
                    for i, (comb_layer, graph_layer) in enumerate(zip_longest(comb_layers, graph_layers, fillvalue=[])):
                        if graph_layer:
                            if i > len(comb_layers) - 1:
                                comb_layers += [graph_layer]  # add new r|d layer
                            else:
                                comb_layers[i] += graph_layer  # splice r|d PP layer into existing layer

    return comb_layers_t, sub_valt


def sum2graph_(igraph_, fd):  # sum nodes' params into graph

    graph_ = []
    for igraph in igraph_:
        node_, valt = igraph[0]
        # draft:
        graph = Cgraph(
            node_=node_, plevels=[[],[],[0,0]], valt=valt, fds=deepcopy(node_[0].fds), x0=node_[0].x0, xn=node_[0].xn, y0=node_[0].y0, yn=node_[0].yn)
        for node in node_:
            graph.valt[0] += node.valt[0]; graph.valt[1] += node.valt[1]

            graph.x0=min(graph.x0, node.x0)
            graph.xn=max(graph.xn, node.xn)
            graph.y0=min(graph.y0, node.y0)
            graph.yn=max(graph.yn, node.yn)

            # accumulate plevel and alt_plevel
            for gplevels, nplevels in zip([graph.plevels, graph.alt_plevels], [node.plevels, node.alt_plevels]):
                for gplevel, plevel in zip_longest(gplevels, nplevels, fillvalue=[]):
                    if plevel and plevel[0]:
                        if gplevel and gplevel[0]:
                            sum_players(gplevel[0], plevel[0])  # accum nodes' players
                            gplevel[1] = deepcopy(plevel[1])    # assign fds
                            gplevel[2][0] += plevel[2][0];gplevel[2][1] += plevel[2][1]  # accumulate valt
                        else:
                            gplevels.append(deepcopy(plevel))  # pack new plevel

        graph_ += [graph]

    return graph_


# not fully revised, this is an alternative to form_graph, but may not be accurate enough to cluster:
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


# for deeper agg_recursion:
def comp_levels(_levels, levels, der_levels, fsubder=0):  # only for agg_recursion, each param layer may consist of sub_layers

    # recursive unpack of nested param layers, each layer is ptuple pair_layers if from der+
    der_levels += [comp_players(_levels[0], levels[0])]

    # recursive unpack of deeper layers, nested in 3rd and higher layers, if any from agg+, down to nested tuple pairs
    for _level, level in zip(_levels[1:], levels[1:]):  # level = deeper sub_levels, stop if none
        der_levels += [comp_levels(_level, level, der_levels=[], fsubder=fsubder)]

    return der_levels  # possibly nested param layers


'''
    1st and 2nd layers are single sublayers, the 2nd adds tuple pair nesting. Both are unpacked by func_pairs, not func_layers.  
    Multiple sublayers start on the 3rd layer, because it's derived from comparison between two (not one) lower layers. 
    4th layer is derived from comparison between 3 lower layers, where the 3rd layer is already nested, etc:
    initial 3-layer nesting: https://github.com/assets/52521979/ea6d436a-6c5e-429f-a152-ec89e715ebd6
    '''


# the below is out of date:

def sum_levels(Params, params):  # Capitalized names for sums, as comp_levels but no separate der_layers to return

    if Params:
        sum_players(Params[0], params[0])  # recursive unpack of nested ptuple layers, if any from der+
    else:
        Params.append(deepcopy(params[0]))  # no need to sum

    for Level, level in zip_longest(Params[1:], params[1:], fillvalue=[]):
        if Level and level:
            sum_levels(Level, level)  # recursive unpack of higher levels, if any from agg+ and nested with sub_levels
        elif level:
            Params.append(deepcopy(level))  # no need to sum


def form_PPP_t(iPPP_t):  # form PPs from match-connected segs
    PPP_t = []

    for fPd, iPPP_ in enumerate(iPPP_t):
        # sort by value of last layer: derivatives of all lower layers:
        if fPd:
            iPPP_ = sorted(iPPP_, key=lambda iPPP: iPPP.dval, reverse=True)  # descending order
        else:
            iPPP_ = sorted(iPPP_, key=lambda iPPP: iPPP.mval, reverse=True)  # descending order

        PPP_ = []
        for i, iPPP in enumerate(iPPP_):
            if fPd:
                iPPP_val = iPPP.dval
            else:
                iPPP_val = iPPP.mval

            for mptuple, dptuple in zip(iPPP.mplayer, iPPP.dplayer):
                if mptuple and dptuple:  # could be None
                    if fPd:
                        iPPP.rdn += dptuple.val > mptuple.val
                    else:
                        iPPP.rdn += mptuple.val > dptuple.val
            '''
            for param_layer in iPPP.params:  # may need recursive unpack here
                iPPP.rdn += sum_named_param(param_layer, 'val', fPd=fPd)> sum_named_param(param_layer, 'val', fPd=1-fPd)
            '''

            ave = vaves[fPd] * iPPP.rdn * (i + 1)  # derPP is redundant to higher-value previous derPPs in derPP_
            if iPPP_val > ave:
                PPP_ += [iPPP]  # base derPP and PPP is CPP
                if iPPP_val > ave * 10:
                    comp_PP_(iPPP, fPd)  # replace with reclustering?
            else:
                break  # ignore below-ave PPs
        PPP_t.append(PPP_)
    return PPP_t


def form_segPPP_root(PP_, root_rdn, fPd):  # not sure about form_seg_root

    for PP in PP_:
        link_eval(PP.uplink_layers, fPd)
        link_eval(PP.downlink_layers, fPd)

    for PP in PP_:
        form_segPPP_(PP)


def form_segPPP_(PP):
    pass


# pending update
def splice_segs(seg_):  # in 1st run of agg_recursion
    pass


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
            _PP.downlink_layers += [downlink]