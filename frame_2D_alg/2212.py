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

def form_graph_(root, G_): # form plevel in agg+ or player in sub+, G is node in GG graph; der+: comp_link if fderG, from sub+

    comp_G_(G_)  # cross-comp all graphs within rng, graphs may be segs | fderGs, G.roott += link, link.node
    mnode_, dnode_ = [], []  # Gs with >0 +ve fork links:
    for G in G_:
        if G.link_.Qm: mnode_ += [G]  # all nodes with +ve links, not clustered in graphs yet
        if G.link_.Qd: dnode_ += [G]
    graph_t = []
    Gm_, Gd_ = copy(G_), copy(G_)
    for fd, (node_, G_) in enumerate(zip([mnode_, dnode_], [Gm_, Gd_])):
        graph_ = []  # form graphs by link val:
        while G_:  # all Gs not removed in add_node_layer
            G = G_.pop()
            gnode_ = [G]
            val = add_node_layer(gnode_, G_, G, fd, val=0)  # recursive depth-first gnode_+=[_G]
            graph_ += [CQ(Q=gnode_, val=val)]
        # reform graphs by node val:
        regraph_ = graph_reval(graph_, [ave_G for graph in graph_], fd)  # init reval_ to start
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum proto-graph node_ params in graph
            for graph in graph_:
                if root.plevels.H or graph.plevels.H:  # or init plevels=list?
                    sum_pH(root.plevels, graph.plevels)
        graph_t += [graph_]

    add_alt_graph_(graph_t)  # overlap + contour, to compute borrowed value?
    return graph_t

def graph_reval(graph_, reval_, fd):  # recursive eval nodes for regraph, increasingly selective with reduced node.link_.val

    regraph_, rreval_ = [],[]
    Reval = 0
    while graph_:
        graph = graph_.pop()
        reval = reval_.pop()
        if reval < ave_G:  # same graph, skip re-evaluation:
            regraph_ += [graph]; rreval_ += [0]
            continue
        while graph.Q:  # some links will be removed, graph may split into multiple regraphs, init each with graph.Q node:
            regraph = CQ()
            node = graph.Q.pop()  # node_, not removed below
            val = [node.link_.mval, node.link_.dval][fd]  # in-graph links only
            if val > G_aves[fd]:  # else skip
                regraph.Q = [node]; regraph.val = val  # init for each node, then add _nodes
                node.roott[fd] = regraph
                readd_node_layer(regraph, graph.Q, node, fd)  # recursive depth-first regraph.Q+=[_node]
            reval = graph.val - regraph.val
            if regraph.val > ave_G:
                regraph_ += [regraph]; rreval_ += [reval]; Reval += reval
    if Reval > ave_G:
        regraph_ = graph_reval(regraph_, rreval_, fd)  # graph reval while min val reduction

    return regraph_

def merge(_graph, graph, fd):

    for node in graph.Q:
        if node not in _graph.Q:
            _graph.Q += [node]
            _graph.valt[fd] += node.link_.valt[fd]

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
                tval = mval + dval
                _G.link_.Q += [derG]; _G.link_.val += tval  # val of combined-fork' +- links?
                G.link_.Q += [derG]; G.link_.val += tval
                if mval > ave_Gm:
                    _G.link_.Qm += [derG]; _G.link_.mval += mval  # no dval for Qm
                    G.link_.Qm += [derG]; G.link_.mval += mval
                if dval > ave_Gd:
                    _G.link_.Qd += [derG]; _G.link_.dval += dval  # no mval for Qd
                    G.link_.Qd += [derG]; G.link_.dval += dval
'''
[gblob.node_.Q, gblob.alt_graph_][fd][:] = graph_  # node_=PPm_, alt_graph_=PPd_, [:] to enable to left hand assignment, not valid for object

if fseg: PP = PP.roott[PP.fds[-1]]  # seg root
PP_P_ = [P for P_ in PP.P__ for P in P_]  # PPs' Ps
for altPP in PP.altPP_:  # overlapping Ps from each alt PP
    altPP_P_ = [P for P_ in altPP.P__ for P in P_]  # altPP's Ps
    alt_rdn = len(set(PP_P_).intersection(altPP_P_))
    PP.alt_rdn += alt_rdn  # count overlapping PPs, not bilateral, each PP computes its own alt_rdn
    gblob.alt_rdn += alt_rdn  # sum across PP_
'''

def add_alt_graph_(graph_t):  # mgraph_, dgraph_

    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            for node in graph.node_:
                for derG in node.link_.Q:
                    for G in derG.node_.Q:
                        if G not in graph.node_:  # alt graphs are roots of not-in-graph G in derG.node_
                            alt_graph = G.roott[1-fd]  # never Cgraph here?
                            if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, Cgraph):  # not proto-graph or removed
                                graph.alt_graph_ += [alt_graph]
                                alt_graph.alt_graph_ += [graph]  # bilateral assign
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            if graph.alt_graph_:
                graph.alt_plevels = CpH()  # players if fsub? der+: plevels[-1] += player, rng+: players[-1] = player?
            for alt_graph in graph.alt_graph_:
                sum_pH(graph.alt_plevels, alt_graph.plevels)  # accum alt_graph_ params
                graph.alt_rdn += len(set(graph.node_).intersection(alt_graph.node_))  # overlap


def form_graph_link_(root, G_):  # form plevel in agg+ or player in sub+, G is node in GG graph; der+: comp_link if fderG, from sub+

    comp_G_(G_)  # cross-comp all graphs within rng, graphs may be segs | fderGs, G.roott += link, link.node
    node_, link_ = [], []  # Gs with >0 +ve fork links:
    for G in G_:
        if G.link_.Qm: node_ += [G]  # all nodes with +ve links, not clustered in graphs yet
        if G.link_.Qd:
            for link in G.link_.Qd:  # +dval, -mval?
                if link not in link_: link_ += [link]
    mgraph_=[]
    while node_:  # all Gs not removed in add_node_layer
        G = node_.pop()
        gnode_ = [G]
        val = add_node_layer(gnode_, node_, G, val=0)  # recursive depth-first gnode_+=[_G]
        mgraph_ += [CQ(Q=gnode_, val=val)]
    if mgraph_:
        regraph_ = graph_reval(mgraph_, [ave_G for graph in mgraph_])  # select by node val, init reval_ to start
        if regraph_:
            mgraph_[:] = sum2graph_(regraph_, fd=0)  # sum proto-graph node_ params in graph
            for mgraph in mgraph_:
                if root.plevels.H or mgraph.plevels.H:  # or init plevels=list?
                    sum_pH(root.plevels, mgraph.plevels)
    dgraph_=[]
    while link_:  # group links in dgraph
        link = link_.pop(0)
        if link.plevels[1].val > G_aves[1]:
            graph = CQ(Q=[link], val=link.plevels[1].val)
            add_link_layer(graph, link, link_)
            dgraph_ += [graph]
    if dgraph_:
        dgraph_[:] = sum2graph_(dgraph_, fd=1)  # sum proto-graph node_ params in graph
        for dgraph in dgraph_:
            if root.plevels.H or dgraph.plevels.H:
                sum_pH(root.plevels, dgraph.plevels)

    add_alt_graph_(mgraph_, dgraph_)  # overlap + contour, to compute value borrowed by specific vectors
    return [mgraph_, dgraph_]  # graph_t

def add_link_layer(graph, link, link_):  # cluster by borrowed val, link m is not known till sub+,
    # init in dnode: gradient-> graph_reval?

    for node in link.node_.Q:
        for adj_link in node.link_.Qd:
            if adj_link in link_ and adj_link not in graph.Q:
                graph.Q += [adj_link]
                graph.val += adj_link.plevels[1].val
                link_.remove(adj_link)
                add_link_layer(graph, link, link_)

def sum2graph_(G_, fd):  # sum node and link params into graph, plevel in agg+ or player in sub+: if fderG?
                         # fd for clustering, same or fderG for alts?
    graph_ = []
    for G in G_:
        X0, Y0, Xn, Yn = 0, 0, 0, 0
        node_, val = G.Q, G.val
        L = len(node_)
        for node in node_: X0 += node.x0; Y0 += node.y0  # first pass defines center
        X0 /= L; Y0 /= L
        graph = Cgraph(L=L, node_=node_)
        graph_plevels = CpH(); new_plevel = CpH(L=1)  # 1st link adds 2 nodes, other links add 1, one node is already in the graph

        for node in node_:  # define max distance,A, sum plevels:
            Xn = max(Xn, (node.x0 + node.xn) - X0)  # box xn = x0+xn
            Yn = max(Yn, (node.y0 + node.yn) - Y0)
            # node: G|derG, sum plevels ( pplayers ( players ( ptuples:
            if fd:
                sum_pH(graph_plevels, node.plevels[1])
            else:
                sum_pH(graph_plevels, node.plevels)
            # we are assigning object here, so i don't see other way except the one below
            if isinstance(node.node_, Cgraph):
                node.node_[0].root = graph
                # assign node.root too?
            else:
                node.root = graph
            # node.node[0].roott[1-fd] if fd else n  # in der+, converted link.node.roott is assigned instead

def sum2graph_(G_, fd):  # sum node and link params into graph, plevel in agg+ or player in sub+: if fderG?
                         # fd for clustering, same or fderG for alts?
    graph_ = []
    for G in G_:
        X0,Y0, Xn,Yn = 0,0,0,0
        node_, val = G.Q, G.val
        L = len(node_)
        for node in node_: X0+=node.x0; Y0+=node.y0  # first pass defines center
        X0/=L; Y0/=L
        graph = Cgraph(L=L, node_=node_)
        graph_plevels = CpH(); new_plevel = CpH(L=1)  # 1st link adds 2 nodes, other links add 1, one node is already in the graph

        for node in node_:  # define max distance,A, sum plevels:
            Xn = max(Xn,(node.x0+node.xn)-X0)  # box xn = x0+xn
            Yn = max(Yn,(node.y0+node.yn)-Y0)
            # node: G|derG, sum plevels ( pplayers ( players ( ptuples:
            sum_pH(graph_plevels, node.plevels)
            node.roott[fd] = graph
            rev=0
            while isinstance(node.plevels, list): # node is derG:
                rev=1
                Node = node.node_.Q[0]  # get lower pplayers from node.node_[0]:
                if isinstance(Node.plevels, list):  # node is derG in der++
                   sum_pH(graph_plevels, Node.plevels)  # sum lower pplayers of the top plevel in reversed order
            if rev:
                i = 2**len(node.plevels.H)  # n_players in implicit pplayer = n_higher_plevels ^2: 1|1|2|4...
                _i = -i; inp = graph_plevels.H[0]
                rev_pplayers = CpH(val=inp.val, L=inp.L, S=inp.S, A=inp.A)
                for players, fd in zip(inp.H[-i:-_i], inp.fds[-i:-_i]): # reverse pplayers to bottom-up, keep sequence of players in pplayer:
                    rev_pplayers.H += [players]; rev_pplayers.fds += [fd]
                    _i = i; i += int(np.sqrt(i))
                graph_plevels = CpH(H=Node.plevels.H+[rev_pplayers], val=Node.plevels.val+graph_plevels.val, fds=Node.plevels.fds+[fd])
                # low plevels in last node_[0] while derG, + top plevel: rev pplayers of node==derG
            for derG in node.link_.Q:
                sum_pH(new_plevel, derG.plevels[fd])  # sum new_plevel across nodes, accum derG, += L=1, S=link.S, preA: [Xn,Yn]

        new_plevel.A = [Xn*2,Yn*2]
        graph.x0=X0; graph.xn=Xn; graph.y0=Y0; graph.yn=Yn
        graph.plevels.H = graph_plevels.H + [new_plevel]  # currently empty
        graph.plevels.fds = copy(node.plevels.fds) + [fd]
        graph.plevels.val = graph_plevels.val
        graph_ += [graph]

    return graph_

def cluster_dnode_(_node, node_, link_):

    for link in link_:
        node = link.node_.Q[1] if link.node_.Q[0] is _node else link.node_.Q[0]

        if node in node_ and node.Ddplevel.val > 0:  # merge node into _node
            # accumulate plevels and alt_plevels
            sum_pH(_node.plevels, node.plevels)
            sum_pH(_node.alt_plevels, node.alt_plevels)
            # pack links
            for link in node.link_.Q:
                if link not in _node.link_.Q:
                    _node.link_.Q += [link]
            node_.remove(node)
            # search continuously with links
            next_link_ = copy(node.link_.Q)  # copy to prevent link added a same link_.Q while looping it in the next loop
            cluster_dnode_(_node, node_, next_link_)

def comp_links(node):  # forms quasi-gradient with variable link length

    Mdplevel = CpH(); Ddplevel = CpH()
    for i, _link in enumerate(node.link_.Qd):
        for link in node.link_.Q[i+1:]:
            mdplevel, ddplevel = comp_pH(_link.plevels[1], link.plevels[1])
            sum_pH(Mdplevel, mdplevel); sum_pH(Ddplevel, ddplevel)

    return Mdplevel, Ddplevel
'''
            # in form_graph if plevels per mlevels, dlevels:
            root.plevels.H=[]; root.plevels.val=0; root.plevels.fds=[]
            for graph in graph_:
                sum_pH(root.plevels, graph.plevels)  # replace root.plevels
                
            # in sub_recursion if last plevel is not added in sum2graph: 
            if fd:
                for node in node_:  # comp sum_node_link, not comp_links: revert explosion in links?
                                    # or selective comp_links?
                    Dplevel = CpH()
                    for link in node.link_.Q:  # form quasi-gradient from links of variable length:
                        sum_pH(Dplevel, link.plevels[1])  # adjust by dangle?
                    node.plevels.H = [Dplevel]; node.plevels.val = Dplevel.val; node.plevels.fds = [1]
                    # comp new plevel only:
'''