def eval_med_layer(graph_, graph, fd):  # recursive eval of reciprocal links from increasingly mediated nodes

    node_, medG_, val = graph  # node_ is not used here
    save_node_, save_medG_ = [], []
    adj_Val = 0  # adjustment in connect val in graph

    for mG, dir_mderG, G in medG_:  # assign G and shortest direct derG to each med node?
        fmed = 1; mmG_ = []  # __Gs that mediate between Gs and _Gs
        for mderG in mG.link_:  # all evaluated links

            mmG = mderG.node_[1] if mderG.node_[0] is mG else mderG.node_[0]
            for derG in G.link_:
                try_mG = derG.node_[0] if derG.node_[1] is G else derG.node_[1]
                if mmG is try_mG:  # mmG is directly linked to G
                    if derG.plevels[fd].S > dir_mderG.plevels[fd].S:
                        dir_mderG = derG  # for next med eval if dir link is shorter
                        fmed = 0
                    break
            if fmed:  # mderG is not reciprocal, else explore try_mG links in the rest of medG_
                for mmderG in mmG.link_:
                    if G in mmderG.node_:  # mmG mediates between mG and G
                        adj_val = mmderG.plevels[fd].val - ave_agg  # or increase ave per mediation depth
                        # adjust nodes:
                        G.plevels.val += adj_val; mG.plevels.val += adj_val  # valts are not updated
                        val += adj_val; mG.roott[fd][2] += adj_val  # root is not graph yet
                        mmG = mmderG.node_[0] if mmderG.node_[0] is not mG else mmderG.node_[1]
                        if mmG not in mmG_:  # not saved via prior mG
                            mmG_ += [mmG]
                            adj_Val += adj_val

    for mG, dir_mderG, G in medG_:  # new
        if G.plevels.val>0:
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
                    adj_Val += _derG.plevels[fd].val - ave_agg
            val += _val
            graph_.remove(_graph)

    if val > ave_G:
        graph[:] = [save_node_+ add_node_, save_medG_+ add_medG_, val]
        if adj_Val > ave_med:  # positive adj_Val from eval mmG_
            eval_med_layer(graph_, graph, fd)  # eval next med layer in reformed graph
    else:
        graph_.remove(graph)
        for node in save_node_+ add_node_: node.roott[fd] = []  # delete roots


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
            graph_plevels = [graph.mplevels,graph.dplevels][fd]; plevels = [node.mplevels, node.dplevels][fd]
            # graph.plevels.H is init empty, we need it to retrieve the players if node is derG
            sum_pH(graph_plevels, plevels)  # G or derG
            # sum node plevels(pplayers(players(ptuples:
            # plevels.H is empty if prior fork is different with fd
            # node is derG
            while plevels.H and plevels.H[0].L==1:  # node is derG, get lower pplayers from its nodes
                node = node.node_[0]
                plevels = node.mplevels if node.mplevels.H else node.dplevels  # prior sub+ fork
                i = 2 ** len(plevels.H); _i = -i  # n_players in implicit pplayer = n_higher_plevels ^2: 1|1|2|4...
                for graph_players, players in zip(graph_plevels.H[-1].H[-i:-_i], plevels.H[-1].H[-i:-_i]):
                    sum_pH(graph_players, players)  # sum pplayer' players
                    _i = i; i += int(np.sqrt(i))
            for derG in node.link_:
                derG_plevels = [derG.mplevels, derG.dplevels][fd]
                sum_pH(new_plevel, derG_plevels.H[0])  # the only plevel, accum derG, += L=1, S=link.S, preA: [Xn,Yn]
                val += derG_plevels.val
            plevels.val += val
            graph_plevels.val += plevels.val

        new_plevel.A = [Xn*2,Yn*2]
        graph_plevels.H += [new_plevel]
        graph_plevels.fds = copy(plevels.fds) + [fd]
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

graph_plevels = [node_[0].mplevels, node_[0].dplevels][fd]  # sum top-down


def form_graph_(root, G_, ifd):  # forms plevel in agg+ or player in sub+, G is potential node graph, in higher-order GG graph

    # der+: comp_link if fderG, from sub_recursion_g?
    for G in G_:  # roott: mgraph, dgraph
        for i in 0, 1:
            graph = [[G], [], 0]  # proto-GG: [node_, medG_, val]
            G.roott[i] = graph  # initial overlap between graphs is eliminated in eval_med_layer

    comp_G_(G_, ifd)  # cross-comp all graphs within rng, graphs may be segs | fderGs?
    mgraph_, dgraph_ = [], []  # initialize graphs with >0 positive links in graph roots:
    for G in G_:
        if len(G.roott[0][0]) > 1 and G.roott[0] not in mgraph_: mgraph_ += [G.roott[0]]  # root = [node_, val] for eval_med_layer, + link_nvalt?
        if len(G.roott[1][0]) > 1 and G.roott[1] not in dgraph_: dgraph_ += [G.roott[1]]

    for fd, graph_ in enumerate([mgraph_, dgraph_]):  # evaluate intermediate nodes to delete or merge their graphs:
        regraph_ = []
        while graph_:
            graph = graph_.pop(0)
            eval_med_layer(graph_=graph_, graph=graph, fd=fd)
            if graph[2][fd] > ave_agg: regraph_ += [graph]  # graph reformed by merges and removes above
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum proto-graph node_ params in graph
            for graph in graph_:
                root_plevels = [root.mplevels, root.dplevels][fd]; plevels = [graph.mplevels, graph.dplevels][fd]
                if root_plevels.H or plevels.H:  # better init plevels=list?
                    sum_pH(root_plevels, plevels)

    return mgraph_, dgraph_

for node in graph.Q:  # node_
        graph = node.roott[fd]
        if node.link_.valt[fd] > G_aves[fd]:
            for link in node.link_.Q:
                if [link.mplevels, link.dplevels][fd].val > G_aves[fd]:
                    _node = link.node_.Q[1] if link.node_.Q[0] is node else link.node_.Q[0]
                    _graph = _node.roott[fd]
                    val = _node.link_.val
                    if val > G_aves[fd]:
                        if _node not in graph.Q:
                            graph.Q += [_node]; graph.val += val
                            _graph.Q += [node]; _graph.val += val
                            adj_Val += val
                    elif _node in graph.Q:
                        graph.Q.remove(graph.Q.index(_node)); graph.val -= val
                        _graph.Q.remove(_graph.Q.index(node)); _graph.val -= val
                        adj_Val += val
        '''            
        else: remove pos reciprocal links 
        always for link in node.link_.Q: eval add|remove positive link
        links are added in comp_G_, separate if pos, remove if neg node, stays in tested node_: from tested link_?
        '''

medG_ = list  # last checked mediating [mG, dir_link, G]s, from all nodes?
'''
keep the shorter of direct or mediating links for both nodes, add value of mediated links: if shorter, unique for med eval:
recursive node eval of combined links, direct + mediated: not reciprocal, *= accum mediation coef, unless known direct?
or remove vs. med_eval, including reciprocal links: too complex, may be removed anyway, ave/ ave_rdn? 
remove links of removed nodes, add link_ val?
'''
def eval_med_layer(graph_, graph, fd):  # recursive eval of reciprocal links from increasingly mediated nodes,
                                        # links m,d of mediating node += reciprocal m,d?
    node_, medG_, val = graph  # node_ not used
    save_node_, save_medG_ = [], []
    adj_Val = 0  # adjustment in connect val in graph

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

def form_graph_(root, G_, ifd):  # form plevel in agg+ or player in sub+, G is node in GG graph; der+: comp_link if fderG, from sub+

    comp_G_(G_, ifd)  # cross-comp all graphs within rng, graphs may be segs | fderGs, G.roott += link, link.node
    mnode_, dnode_ = [],[]  # Gs with >0 +ve fork links:
    for G in G_:
        if G.link_.Qm: mnode_ += G  # all nodes with +ve links, not clustered in graphs yet
        if G.link_.Qd: dnode_ += G
    graph_t = []
    for fd, node_ in enumerate([mnode_, dnode_]):
        graph_ = []
        for G in G_.pop():  # form graphs of linked nodes: Gs not removed in add_node_layer
            gnode_ = [G]; val = 0
            add_node_layer(gnode_, val, G_, G, fd)  # recursive depth-first gnode_+=[_G]
            graph_ += CQ(Q=gnode_, val=val)

        regraph_ = graph_reval(graph_, fd)  # graphs recursively reformed by node re-evaluation
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum proto-graph node_ params in graph
            for graph in graph_:
                root_plevels = [root.mplevels, root.dplevels][fd]; plevels = [graph.mplevels, graph.dplevels][fd]
                if root_plevels.H or plevels.H:  # better init plevels=list?
                    sum_pH(root_plevels, plevels)
        graph_t += [graph_]

    return graph_t

def merge(_graph, graph, fd):

    for node in graph.Q:
        if node not in _graph.Q:
            _graph.Q += [node]
            _graph.valt[fd] += node.link_.valt[fd]

