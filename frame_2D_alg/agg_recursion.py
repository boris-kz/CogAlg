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

    link_ = lambda: [[], 0]  # evaluated graph_as_node external links, replace alt_node if open, direct only?
    node_ = lambda: [[], 0]  # elements, common root of layers, levels:
    # agg,sub forks: [[],0] if called only
    mlevels = list  # PPs) Gs) GGs.: node_ agg, each a flat list
    dlevels = list
    rlayers = list  # | mlayers: init from node_ for subdivision, micro relative to levels
    dlayers = list  # | alayers; init val = sum node_val, for sub_recursion
    # sum params in all nesting orders:
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


def form_graph_(root, G_, ifd):  # forms plevel in agg+ or player in sub+, G is potential node graph, in higher-order GG graph
                                 # der+: comp_link if fderG, from sub_recursion_g?
    for G in G_:  # roott: mgraph, dgraph
        for i in 0,1:
            G.roott[i] = [[G],[],0]  # proto-graph is initialized for each [node, link_, val]

    comp_G_(G_, ifd)  # cross-comp all graphs within rng, graphs may be segs | fderGs?
    mgraph_, dgraph_ = [],[]  # initialize graphs with >0 positive links in graph roots:
    for G in G_:
        if len(G.roott[0][2])>ave_Gm and G.roott[0] not in mgraph_: mgraph_ += [G.roott[0]]  # root = [node_, link_, val], + link_nvalt?
        if len(G.roott[1][2])>ave_Gd and G.roott[1] not in dgraph_: dgraph_ += [G.roott[1]]

    for fd, graph_ in enumerate([mgraph_, dgraph_]):  # evaluate intermediate nodes to delete or merge their graphs:
        regraph_ = []
        while graph_:
            graph = graph_.pop(0)
            node_reval(graph_= graph_, graph=graph, fd=fd)
            if graph[2][fd] > ave_agg: regraph_ += [graph]  # graph reformed by merges and removes above
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum proto-graph node_ params in graph
            for graph in graph_:
                root_plevels = [root.mplevels, root.dplevels][fd]; plevels = [graph.mplevels, graph.dplevels][fd]
                if root_plevels.H or plevels.H:  # better init plevels=list?
                    sum_pH(root_plevels, plevels)

    return mgraph_, dgraph_


def comp_G_(G_, ifd):  # cross-comp Gs (patterns of patterns): Gs, derGs, or segs inside PP, same process, no fderG?

    for i, _G in enumerate(G_):
        for G in G_[i+1:]:
            # compare each G to other Gs in rng, bilateral link assign, val accum:
            if G in [node for link in _G.link_ for node in link.node_]:  # G,_G was compared in prior rng+, add frng to skip?
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
                derG = Cgraph( node_=[_G,G], mplevels=mplevels, dplevels=dplevels, y0=y0, x0=x0, # compute max distance from center:
                               xn=max((_G.x0+_G.xn)/2 -x0, (G.x0+G.xn)/2 -x0), yn=max((_G.y0+_G.yn)/2 -y0, (G.y0+G.yn)/2 -y0))
                _G.link_[0] += [derG]; _G.link_[1] += derG.mplevels.val
                G.link_[0] += [derG];   G.link_[1] += derG.dplevels.val  # any val
                for fd, val in enumerate([mplevel.val, dplevel.val]):  # alt fork is redundant, no support?
                    if val > 0:
                        for node, (graph, medG_, val) in zip([_G, G], [G.roott[fd], _G.roott[fd]]):  # bilateral inclusion
                            if node not in graph:
                                graph += [node]
                                for derG in node.link_:
                                    mG = derG.node_[0] if derG.node_[1] is node else derG.node_[1]
                                    if mG not in [medG[0] for medG in medG_]:
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

'''
draft: recursive eval of links per node, add/remove nodes and their reciprocal links
'''
def node_reval(graph_, graph, fd):

    node_, link_, val = graph
    adj_Val = 0  # adjustment in connect val in graph

    for node in node_:
        graph = node.roott[fd]
        if node.link_[1] > G_aves[fd]:
            for link in node.link_[0]:
                if [link.mplevels, link.dplevels][fd].val > G_aves[fd]:
                    _node = link.node_[0][0] if link.node_[0][1] is node else link.node_[0][1]
                    _graph = _node.roott[fd]
                    if _node.link_[1] > G_aves[fd] and _node not in graph.node_:
                        graph.node_[0] += [_node]; graph.node_[1] += _node.link_[1]  # val
                        _graph.node_[0] += [node]; _graph.node_[1] += node.link_[1]  # val

    # old:
    mmG__ = []  # not reviewed
    for mG, dir_mderG, G in medG_:  # assign G and shortest direct derG to each med node?
        if not mG.roott[fd]:  # root graph was deleted
            continue
        G_plevels = [G.mplevels, G.dplevels][fd]
        fmed = 1; mmG_ = []  # __Gs that mediate between Gs and _Gs
        mmG__ += [mmG_]
        for mderG in mG.link_:  # all evaluated links

            mmG = mderG.node_[1] if mderG.node_[0] is mG else mderG.node_[0]
            for derG in G.link_:
                try_mG = derG.node_[0] if derG.node_[1] is G else derG.node_[1]
                if mmG is try_mG:  # mmG is directly linked to G
                    if derG.mplevels.S > dir_mderG.mplevels.S:  # same Sd
                        dir_mderG = derG  # next med derG if dir link is shorter
                        fmed = 0
                    break
            if fmed:  # mderG is not reciprocal, else explore try_mG links in the rest of medG_
                for mmderG in mmG.link_:
                    if G in mmderG.node_:  # mmG mediates between mG and G: mmderG adds to connectivity of mG?
                        adj_val = [mmderG.mplevels, mmderG.dplevels][fd].val - ave_agg
                        # adjust nodes:
                        G_plevels.val += adj_val; [mG.mplevels, mG.dplevels][fd].val += adj_val
                        val += adj_val; mG.roott[fd][2] += adj_val  # root is not graph yet
                        mmG = mmderG.node_[0] if mmderG.node_[0] is not mG else mmderG.node_[1]
                        if mmG not in mmG_:  # not saved via prior mG
                            mmG_ += [mmG]; adj_Val += adj_val

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
                    adj_Val += [_derG.mplevels,_derG.dplevels][fd].val - ave_agg
            val += _val
            graph_.remove(_graph)

    if val > ave_G:
        graph[:] = [save_node_+ add_node_, save_medG_+ add_medG_, val]
        if adj_Val > ave_med:  # positive adj_Val from eval mmG_
            eval_med_layer(graph_, graph, fd)  # eval next med layer in reformed graph
    else:
        for node in save_node_+ add_node_: node.roott[fd] = []  # delete roots


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
            # der+:
            Dval = sum([sub_dgraph.dplevels.val for sub_dgraph in sub_dgraph_])
            if Dval > ave_sub * graph.rdn:
                sub_dlayers, dval = sub_recursion_g(sub_dgraph_, Dval, fseg=fseg, fd=1)
                Dval+=dval; graph.dlayers = [[sub_dgraph_]+[sub_dlayers], Dval]

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
        node_, medG_, val = G
        L = len(node_)
        for node in node_: X0+=node.x0; Y0+=node.y0  # first pass defines center
        X0/=L; Y0/=L
        graph = Cgraph(L=L, node_=node_, medG_=medG_)
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
                   sum_pH(graph_plevels, node_plevels)  # sum lower pplayers of the top plevel in reverse order
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
            for derG in node.link_:
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
            for derG in node.link_:
                for G in derG.node_:
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