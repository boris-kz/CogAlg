class CgPP(CPP, CderPP):  # generic PP, of any composition

    players_t = lambda: [[], []]  # mPlayers, dPlayers, max n ptuples / layer = n ptuples in all lower layers: 1, 1, 2, 4, 8...
    valt = lambda: [0, 0]  # mval, dval
    nvalt = lambda: [0, 0]  # neg links
    fds = list  # prior fork sequence, map to mplayer or dplayer in lower (not parallel) player, none for players[0]
    rng = lambda: 1  # rng starts with 1
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs
    Rdn = int  # for accumulation only
    oRdn = int  # overlapping redundancy
    nP = int  # len 2D derP__ in levels[0][fPd]?  ly = len(derP__), also x, y?
    uplink_layers = lambda: [[], []]  # likely not needed
    downlink_layers = lambda: [[], []]
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    box = list  # x0, xn, y0, yn
    mask__ = bool
    gPP_ = list  # (PP|gPP,derPP,fin)s, common root of layers and levels
    cPP_ = list  # co-refs in other PPPs
    rlayers = list  # | mlayers
    dlayers = list  # | alayers
    mlevels = list  # agg_PPs ) agg_PPPs ) agg_PPPPs..
    dlevels = list
    roott = lambda: [None, None]  # lambda: CgPP()  # higher-order segP or PPP


def agg_recursion(root, PP_, rng, fseg=0):  # compositional recursion per blob.Plevel; P, PP, PPP are relative to each other

    PPP_ = comp_PP_(copy(PP_), rng, fseg)  # cross-comp all PPs (which may be segs) within rng, same PPP_ for both forks
    mgraph_ = form_graph(PPP_, rng, fd=0)  # if top level miss, lower levels match: splice PPs vs form PPPs
    dgraph_ = form_graph(PPP_, rng, fd=1)
    # intra-graph:
    if sum(root.valt) > ave_agg * root.rdn:
        # draft:
        sub_rlayers, rvalt = sub_recursion_agg(mgraph_, root.valt, fd=0)  # re-form PPP.PP_ by der+ if PPP.fPd else rng+, accum root.valt
        root.valt[0] += sum(rvalt); root.rlayers = sub_rlayers
        sub_dlayers, dvalt = sub_recursion_agg(dgraph_, root.valt, fd=1)
        root.valt[1] += sum(dvalt); root.dlayers = sub_dlayers
    # cross-graph:
    for i, graph_ in enumerate([mgraph_, dgraph_]):
        # val = sum(root.valt)  # this eval will be per graph
        fork_rdnt = [1 + (root.valt[1] > root.valt[0]), 1 + (root.valt[0] >= root.valt[1])]

        if (root.valt[i] > PP_aves[i] * ave_agg * (root.rdn + 1) * fork_rdnt[i]) and len(graph_) > ave_nsub:
            root.rdn += 1  # estimate, replace by actual after agg_recursion?
            root.mlevels += mgraph_  # bottom-up
            root.dlevels += dgraph_
            agg_recursion(root, graph_, rng=val / ave_agg, fseg=0)  # cross-comp graphs


def comp_PP_(PP_, rng, fseg):  # 1st cross-comp, PPs may be segs inside a PP

    PPP_ = []
    iPPP_ = [copy_P(PP, iPtype=4) for PP in PP_]

    while PP_:  # compare _PP to all other PPs within rng
        _PP, _PPP = PP_.pop(), iPPP_.pop()
        _fid = _PP.fds[-1]
        _PP.roott[_fid] = _PPP

        for PPP, PP in zip(iPPP_, PP_):  # add selection by dy<rng if incremental y? accum derPPs in PPPs
            fid = PP.fds[-1]
            area = PP.players_t[fid][0][0].L; _area = _PP.players_t[_fid][0][0].L
            dx = ((_PP.xn - _PP.x0) / 2) / _area - ((PP.xn - PP.x0) / 2) / area
            dy = _PP.y / _area - PP.y / area
            distance = np.hypot(dy, dx)  # Euclidean distance between PP centroids

            val = sum(_PP.valt) / (ave_mPP+ave_dPP)  # combined PP eval
            if distance * val <= rng:
                # comp PPs:
                mplayer, dplayer = comp_players(_PP.players_t[_fid], PP.players_t[fid], _PP.fds, PP.fds)
                valt = [sum([mtuple.val for mtuple in mplayer]), sum([dtuple.val for dtuple in dplayer])]
                derPP = CderPP(player=[mplayer, dplayer], valt=valt)  # derPP is single-layer
                fint = []
                for fd in 0,1:  # sum players per fork
                    if valt[fd] > PP_aves[fd]:
                        fin = 1  # PPs match, sum derPP in both PPP and _PPP:
                        sum_players(_PPP.players_t[fd], PP.players_t[fd])
                        sum_players(PPP.players_t[fd], PP.players_t[fd])  # all PP.players in each PPP.players
                    else:
                        fin = 0
                    fint += [fin]
                _PPP.gPP_ += [[PP, derPP, fint]]
                _PP.cPP_ += [[PP, derPP, [1,1]]]  # rdn refs, initial fins=1, derPP is reversed
                PPP.gPP_ += [[_PP, derPP, fint]]
                PP.cPP_ += [[_PP, derPP, [1,1]]]  # bilateral assign to eval in centroid clustering, derPP is reversed
                '''
                if derPP.match params[-1]: form PPP
                elif derPP.match params[:-1]: splice PPs and their segs? 
                '''
        PPP_.append(_PPP)
    return PPP_

# below is not reviewed:

def form_graph(PPP_, rng, fd):  # cluster PPPs by match of mutual connections, which is summed across shared nodes and their roots

    graph_= []
    for PPP in PPP_:  # initialize graph for each PPP:
        graph=CgPP(gPP_=[PPP], fds = copy(PPP.fds), rng=rng, players_t = deepcopy(PPP.players_t), valt=PPP.valt)  # graph.gPP_: connected PPPs
        PPP.roott[fd]=graph  # set root of core part as graph
        graph_+=[graph]
    merged_graph_ = []
    while graph_:
        graph = graph_.pop(0)
        eval_ref_layer(graph_=graph_, graph=graph, PPP_=graph.gPP_, shared_Val=0, fid=fd)
        merged_graph_ += [graph]  # graphs merge (split?) in eval_ref_layer, each layer is intermediate PPPs
        # add|remove individual nodes+links: PPPs, not the links between PPs in PPP?

    return merged_graph_

# draft
def eval_ref_layer(graph_, graph, PPP_, shared_Val, fid):  # recursive eval of increasingly mediated nodes (graph_, graph, shared_M)?

    for PPP in PPP_:
        for (PP, derPP, fint) in PPP.gPP_:
            shared_Val += derPP.valt[fid]  # accum shared_M across mediating node layers

            # one of the PP.roott is empty now, we need to add d comp in comp_PP?
            for (_PP, _derPP, _fint) in PP.roott[fid].gPP_:  # _PP.PPP / PP.PPP reciprocal refs:
                if _PP is PP:  # mutual connection
                    shared_Val += _derPP.valt[fid] - ave_agg  # eval graph inclusion by match to mediating gPP
                    # * rdn: sum PP_ rdns / len PP_ + cross-PPP overlap rate + cross-graph overlap rate?
                    if shared_Val > 0:
                        _graph = _PP.roott[fd].roott[fid]  # PP.PPP.graph[m]: two nesting layers above PP
                        if _graph is not graph:
                            # merge graphs:
                            for fd in 0, 1:
                                if _graph.players_t[fd]:
                                    sum_players(graph.players_t[fd], _graph.players_t[fd])
                                    graph.valt[fd] += _graph.valt[fd]
                            for rPP in _graph.gPP_: rPP.roott[fid] = graph  # update graph reference
                            graph.gPP_ += _graph.gPP_
                            graph_.remove(_graph)

                            # recursive search of increasingly intermediated refs for mutual connections
                            for __PP in _PP.gPP_:
                                for (_root_PP,_,_) in __PP.root.gPP_:
                                    _root_PPP = _root_PP.roott[fid]
                                    if _root_PPP.roott[fid] is not graph:
                                        eval_ref_layer(graph_, graph, _root_PPP.roott[fid].gPP_, shared_Val, fid)


def comp_slice_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    segment_by_direction(blob, verbose=False)  # forms blob.dir_blobs
    for dir_blob in blob.dir_blobs:  # same-direction blob should be CBlob

        P__ = slice_blob(dir_blob, verbose=False)  # cluster dir_blob.dert__ into 2D array of blob slices
        comp_P_root(P__)  # scan_P_, comp_P | link_layer, adds mixed uplink_, downlink_ per P; comp_dx_blob(P__), comp_dx?

        # segments are stacks of (P,derP)s:
        segm_ = form_seg_root([copy(P_) for P_ in P__], fd=0, fds=[0])  # shallow copy: same Ps in different lists
        segd_ = form_seg_root([copy(P_) for P_ in P__], fd=1, fds=[0])
        # PPs are graphs of segs:
        PPm_, PPd_ = form_PP_root((segm_, segd_), base_rdn=2)  # not mixed-fork PP_, select by PP.fPd?
        # intra PP:
        dir_blob.rlayers = sub_recursion_eval(PPm_, fd=0)
        dir_blob.dlayers = sub_recursion_eval(PPd_, fd=1)  # add rlayers, dlayers, seg_levels to select PPs

        dir_blob.mlevels = [PPm_]; dir_blob.dlevels = [PPd_]  # agg levels
        M = dir_blob.M; G = dir_blob.G; dir_blob.valt = [M, G]; fork_rdnt = [1+(G>M), 1+(M>=G)]
        from agg_recursion import agg_recursion, CgPP
        # cross PP:
        for i, PP_ in enumerate([PPm_, PPd_]):
            if (dir_blob.valt[i] > PP_aves[i] * ave_agg * (dir_blob.rdn+1) * fork_rdnt[i]) and len(PP_) > ave_nsub:
                for j, PP in enumerate(PP_):  # convert to CgPP
                    alt_players = []
                    for altPP in PP.altPP_: sum_players(alt_players, altPP.players)
                    PP_[j] = CgPP(PP=PP, players_t=[PP.players,alt_players], fds=deepcopy(PP.fds), x0=PP.x0, xn=PP.xn, y0=PP.y0, yn=PP.yn)
                # cluster PPs into graphs:
                dir_blob.rdn += 1  # estimate, replace by actual after agg_recursion?
                agg_recursion(dir_blob, PP_, rng=2, fseg=0)

        # splice_dir_blob_(blob.dir_blobs)  # draft


def comp_PP_(PP_, rng, fseg):  # cross-comp + init clustering, same val,rng for both forks? PPs may be segs inside a PP

    graph_t = [[],[]]

    for PP in PP_:
        for f in 0,1:  # init mgraph, dgraph per PP, no new alt_players_t yet, draft:
            # actually, we can't initialize graphs with PPs, they should only contain positive links
            # comp_PP_ forms links, not graphs?
            graph = CgPP(node_=[PP], fds=copy(PP.fds), valt=PP.valt, x0=PP.x0, xn=PP.xn, y0=PP.y0, yn=PP.yn,
                         players_T = [deepcopy(PP.players_t),deepcopy(PP.alt_players_t)])  # nested lower-composition core,edge tuples
            PP.roott[f] = graph  # set root of core part as graph
            graph_t[f] += [graph]

    for i, _PP in enumerate(PP_):
        for PP in PP_[i:]:  # comp _PP to all other PPs in rng, select by dy<rng if incremental y? accum derPPs in link_player_t

            for fd in 0,1:  # add indefinite nesting:
                if PP.players_t[fd] and _PP.players_t[fd]:
                    continue
                area = PP.players_t[fd][0][0].L; _area = _PP.players_t[fd][0][0].L
                dx = ((_PP.xn - _PP.x0) / 2) / _area - ((PP.xn - PP.x0) / 2) / area
                dy = _PP.y / _area - PP.y / area
                distance = np.hypot(dy, dx)  # Euclidean distance between PP centroids
                #  fork_rdnt = [1 + (root.valt[1] > root.valt[0]), 1 + (root.valt[0] >= root.valt[1])]  # not per graph_?

                # separated val for evaluation now?
                val = _PP.valt[fd] / PP_aves[fd]  # complimented val: cross-fork support, if spread spectrum?
                if distance * val <= rng:
                    # comp PPs:
                    mplayer, dplayer = comp_players(_PP.players_t[fd], PP.players_t[fd], _PP.fds, PP.fds)
                    valt = [sum([mtuple.val for mtuple in mplayer]), sum([dtuple.val for dtuple in dplayer])]
                    derPP = CderPP(player_t=[mplayer, dplayer], valt=valt)  # single-layer
                    fint = []

                    for fdd in 0,1:  # sum players per fork
                        if valt[fdd] > PP_aves[fdd]:  # no cross-fork support?
                            fin = 1
                            for gPP in _PP, PP:  # bilateral inclusion
                                graph = gPP.roott[fd]
                                sum_players(graph.players_t[fdd], PP.players_t[fdd])
                                graph.node_ += [PP]
                                sum_player(gPP.link_player_t[fdd], derPP.player_t[fdd])
                        else:
                            fin = 0
                        fint += [fin]
                    _PP.link_ += [[PP, derPP, fint]]
                    PP.link_ += [[_PP, derPP, fint]]

    return graph_t
'''
_PP.cPP_ += [[PP, derPP, [1,1]]]  # rdn refs, initial fins=1, derPP is reversed
PP.cPP_ += [[_PP, derPP, [1,1]]]  # bilateral assign to eval in centroid clustering, derPP is reversed
if derPP.match params[-1]: form PPP
elif derPP.match params[:-1]: splice PPs and their segs? 

valt = [sum([mtuple.val for mtuple in mplevel]), sum([dtuple.val for dtuple in dplevel])]
_players = _PP.plevels[-1][0]; players = PP.plevels[-1][0]
_fds = _PP.plevels[-1][1]; fds = PP.plevels[-1][1]

                                all plevels accum when graph is complete, easier in batches, no sum/subtract at exclusion or merge 
                                we only need to reform node_ and accumulate link_valt to evaluate cluster_node_layer and reforming
                                # sum_player(node.link_plevel_t[fdd], derPP.plevel_t[fdd])  # accum links
'''

def sum_players(Layers, layers, fneg=0):  # no accum across fPd, that's checked in comp_players?

    if not Layers:
        if not fneg: Layers.append(deepcopy(layers[0]))
    else: accum_ptuple(Layers[0][0], layers[0][0], fneg)  # latuples, in purely formal nesting

    for Layer, layer in zip_longest(Layers[1:], layers[1:], fillvalue=[]):
        if layer:
            if Layer:
                for fork_Layer, fork_layer in zip(Layer, layer):
                    sum_player(fork_Layer, fork_layer, fneg=fneg)
            elif not fneg: Layers.append(deepcopy(layer))

def sum_player(Player, player, fneg=0):  # accum players in Players

    for i, (Ptuple, ptuple) in enumerate(zip_longest(Player, player, fillvalue=[])):
        if ptuple:
            if Ptuple: accum_ptuple(Ptuple, ptuple, fneg)
            elif Ptuple == None: Player[i] = deepcopy(ptuple)  # not sure
            elif not fneg: Player.append(deepcopy(ptuple))


def comp_players(_layers, layers):  # unpack and compare der layers, if any from der+

    mtuple, dtuple = comp_ptuple(_layers[0][0], layers[0][0])  # initial latuples, always present and nested
    mplayer=[mtuple]; dplayer=[dtuple]

    for _layer, layer in zip(_layers[1:], layers[1:]):
        for _ptuple, ptuple in zip(_layer, layer):

            mtuple, dtuple = comp_ptuple(_ptuple, ptuple)
            mplayer+=[mtuple]; dplayer+=[dtuple]

    return mplayer, dplayer

def sum_players(Layers, layers, Fds, fds, fneg=0):  # accum layers of same fds

    for i, (Layer, layer, Fd, fd) in enumerate(zip_longest(Layers, layers, Fds, fds, fillvalue=[])):
        if layer:
            if Layer:
                if Fd==fd: sum_player(Layer, layer, fneg=fneg)
            elif not fneg:
                Layers.append(deepcopy(layer))
        if Fd!=fd:
            break
    Fds[:]=Fds[:i]  # maybe cut short by the break
'''
    for Layer, layer, Fd, fd in zip_longest(Layers, layers, Fds, fds, fillvalue=[]):
        if layer:
            if Layer and Fd==fd:
                sum_player(Layer, layer, fneg=fneg)
            elif not fneg:
                Layers.append(deepcopy(layer))
'''

def comp_graph_(PP_, rng, fd):  # cross-comp, same val,rng for both forks? PPs may be segs inside a PP

    for i, _PP in enumerate(PP_):  # compare patterns of patterns: _PP to other PPs in rng, bilateral link assign:
        for PP in PP_[i+1:]:

            area = PP.plevels[0][0][0][0].L; _area = _PP.plevels[0][0][0][0].L  # 1st plevel' players' 1st player' 1st ptuple' L
            dx = ((_PP.xn - _PP.x0) / 2) / _area - ((PP.xn - PP.x0) / 2) / area
            dy = _PP.y / _area - PP.y / area
            distance = np.hypot(dy, dx)  # Euclidean distance between PP centroids
            # fork_rdnt = [1 + (root.valt[1] > root.valt[0]), 1 + (root.valt[0] >= root.valt[1])]  # not per graph_?

            val = _PP.valt[fd] / PP_aves[fd]  # rng+|der+ fork eval, init der+: single level
            if distance * val <= rng:
                # comp PPs:
                if fd:
                    mplevel, dplevel, mval, dval = \
                        comp_players(_PP.plevels[-1][0], PP.plevels[-1][0], _fds=_PP.plevels[-1][1], fds=PP.plevels[-1][1])
                    alt_mplevel, alt_dplevel, alt_mVal, alt_dVal = \
                        comp_players(_PP.alt_plevels[-1][0], PP.alt_plevels[-1][0], _fds=_PP.alt_plevels[-1][1], fds=PP.alt_plevels[-1][1])
                else:
                    mplevel, dplevel, mval, dval = \
                        comp_plevels(_PP.plevels[:-1], PP.plevels[:-1], _PP.fds[:-1], PP.fds[:-1])
                    alt_mplevel, alt_dplevel, alt_mVal, alt_dVal = \
                        comp_plevels(_PP.alt_plevels[:-1], PP.alt_plevels[:-1], _PP.alt_fds[:-1], PP.alt_fds[:-1])
                # so, I think we need to add this:
                # combine core,edge in PP: both define pattern?
                valt = [mval, dval]
                derPP = CderG(plevel_t=[mplevel, dplevel], valt=valt)
                fint = []
                for fd in 0,1:  # sum players per fork
                    if valt[fd] > PP_aves[fd]:  # no cross-fork support?
                        fin = 1
                        for node, (graph, gvalt) in zip([_PP, PP], [PP.roott[fd], _PP.roott[fd]]):  # bilateral inclusion
                            if node not in graph:
                                graph += [node]
                                gvalt[0] += node.valt[0]; gvalt[1] += node.valt[1]
                    else:
                        fin = 0
                    fint += [fin]
                _PP.link_ += [[PP, derPP, fint]]
                PP.link_ += [[_PP, derPP, fint]]


def cluster_node_layer(graph_, graph, med_node__, fd):  # recursive eval of mutual links in increasingly mediated nodes

    node_, valt = graph
    save_node_, save_med__ = [],[]  # __PPs mediating between PPs and _PPs, flat?

    for PP, med_node_ in zip(node_, med_node__):
        save_med_ = []
        for _PP in med_node_:
            for (__PP, _, _) in _PP.link_:
                if __PP is not PP:  # skip previously evaluated links, if __PP is not in node_:

                    for (___PP, ___derPP, ___fint) in __PP.link_:
                        if ___PP is PP:  # __PP mediates between _PP and PP
                            adj_val = ___derPP.valt[fd] - ave_agg  # or ave per mediation depth?
                            # adjust vals per node and graph:
                            PP.valt[fd] += adj_val; _PP.valt[fd] += adj_val; valt[fd] += adj_val
                            if __PP not in save_med_: save_med_ += [__PP]
                            # if not saved via prior _PP
        if save_med_ and PP.valt[fd]>0:
            save_node_ += [PP]; save_med__ += [save_med_]  # save_med__ is nested

    # re-eval full graph after adjusting it with mediating node layer:
    if valt[fd] > 0:
        add_node_, add_med__ = [], []
        for PP, _PP_ in zip(save_node_, save_med__):
            for _PP in _PP_:
                _graph = _PP.roott[fd]
                if _graph in graph_ and _graph is not graph:  # was not removed or merged via prior _PP
                    _node_, _valt = _graph
                    if _valt[fd] > 0:  # merge graphs:
                        for _node in _node_:
                            add_node_ += [_node]
                            add_med__.append([link[0] for link in _node.link_])  # initial mediation only
                        valt[fd] += _valt[fd]
                        graph_.remove(_graph)
                    else: graph_.remove(_graph)  # neg val

        node_[:] = save_node_ + add_node_  # reassign as save_node_ (>0 after mediation) + add_node_ (mediated positive nodes)
        med_node__[:] = save_med__ + add_med__

        if valt[fd] > ave_agg:  # extra ops, no rdn+?
            # eval next mediation layer in reformed graph:
            cluster_node_layer(graph_, graph, med_node__, fd)

'''
            plevels = deepcopy(graph_[0].plevels)  # initialization
            alt_plevels = deepcopy(graph_[0].alt_plevels)
            for graph in graph_[1:]:
                sum_plevels(plevels, graph.plevels)  # add players, fds
                sum_plevels(alt_plevels, graph.alt_plevels)
            root.plevels = plevels
            root.alt_plevels = alt_plevels

                sum_players(plevels[-1][0],graph.plevels[-1][0], plevels[-1][1],graph.plevels[-1][1], fneg=0)  # add players, fds
                plevels[-1][2][0] += graph.plevels[-1][2][0]; plevels[-1][2][1] += graph.plevels[-1][2][1]     # add valt
                
    # not per graph_: fork_rdnt = [1 + (root.valt[1] > root.valt[0]), 1 + (root.valt[0] >= root.valt[1])]?
'''
class CderG(ClusterStructure):  # tuple of graph derivatives in graph link_

    plevel_t = lambda: [[],[]]  # lists of ptuples in current derivation layer per fork
    valt = lambda: [0,0]  # tuple of mval, dval
    rdn = int
    PPt = list  # PP,_PP comparands, the order is not fixed
    roott = lambda: [None, None]  # PP, not segment in sub_recursion
    # not relevant: alt_plevel_t = lambda: [[],[]]; alt_valt = lambda: [0, 0]
    # not sure:
    box = list  # 2nd level, stay in PP?
    uplink_layers = lambda: [[], []]  # init a layer of dderPs and a layer of match_dderPs
    downlink_layers = lambda: [[], []]
    fdx = NoneType
    link_ = list

def form_graph_(root, PP_, rng):

    for PP in PP_:  # initialize mgraph, dgraph as roott per PP, for comp_PP_
        for i in 0,1:
            graph = [[PP],[0,0]]  # [node_, valt]
            PP.roott[i] = graph

    comp_graph_(PP_, rng)  # cross-comp all PPs within rng, PPs may be segs
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


def comp_graph_(PP_):  # cross-comp PPs: Gs, derGs if fd?, or segs inside PP. same val, ave_rng for both forks?

    for i, _PP in enumerate(PP_):  # compare patterns of patterns: _PP to other PPs in rng, bilateral link assign:
        for PP in PP_[i+1:]:

            area = PP.plevels[0][PP.fds[-1]][0][0][0].L; _area = _PP.plevels[0][_PP.fds[-1]][0][0][0].L
            # not revised: 1st plevel_t' fork' players' 1st player' 1st ptuple' L
            dx = ((_PP.xn - _PP.x0) / 2) / _area - ((PP.xn - PP.x0) / 2) / area
            dy = _PP.y / _area - PP.y / area
            distance = np.hypot(dy, dx)  # Euclidean distance between PP centroids
            # not per graph_: fork_rdnt = [1 + (root.valt[1] > root.valt[0]), 1 + (root.valt[0] >= root.valt[1])]?

            if distance <= ave_rng * (sum(_PP.valt) / sum(PP_aves)):  # max distance depends on combined val
                if isinstance(PP, Cgraph):
                    mplevel_t, dplevel_t, mval, dval = comp_plevel_ts(_PP.plevels, PP.plevels)
                else:  # draft for CderG, der+
                    mplevel_t, dplevel_t, mval, dval = comp_players(_PP.plevel_t, PP.plevel_t)

                valt = [mval - ave_Gm, dval - ave_Gd]  # adjust for link rdn?
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

