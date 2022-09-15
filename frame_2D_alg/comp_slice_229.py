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
