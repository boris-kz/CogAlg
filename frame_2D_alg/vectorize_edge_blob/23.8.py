def form_graph_(G_, pri_root_T_, fder, fd):  # form list graphs and their aggHs, G is node in GG graph

    node_ = []  # Gs with >0 +ve fork links:
    for G in G_:
        if G.link_H[-(1+fder)]: node_ += [G]  # node with +ve links, not clustered in graphs yet
        # der+: eval lower link_ layer, rng+: new link_ layer
    graph_ = []
    # init graphs by link val:
    while node_:  # all Gs not removed in add_node_layer
        G = node_.pop()
        qG = [[G],0]; G.root_T[fder][fd] = qG; gnode_.root_T = pri_root_T_
        val = init_graph(gnode_, node_, G, fder, fd, val=0)  # recursive depth-first gnode_ += [_G]
        graph_ += [[gnode_,val]]
    # prune graphs by node val:
    regraph_ = graph_reval_(graph_, [G_aves[fder] for graph in graph_], fder)  # init reval_ to start
    if regraph_:
        graph_[:] = sum2graph_(regraph_, fder)  # sum proto-graph node_ params in graph

    # add_alt_graph_(graph_t)  # overlap+contour, cluster by common lender (cis graph), combined comp?
    return graph_

def init_graph(qG, node_, fder, fd):  # recursive depth-first gnode_+=[_G]

    for link in G.link_H[-(1+fder)]:
        if link.valt[fd] > G_aves[fd]:
            # all positive links init graph, eval node.link_ in prune_node_layer:
            _G = link.G1 if link.G0 is G else link.G0
            if _G in G_:  # _G is not removed in prior loop
                gnode_ += [_G]
                G_.remove(_G)
                val += _G.valt[fd]  # interval
                val += init_graph(gnode_, G_, _G, fder, fd, val)
    return val

def graph_reval_(graph_, reval_, fd):  # recursive eval nodes for regraph, after pruning weakly connected nodes

    regraph_, rreval_ = [],[]
    aveG = G_aves[fd]

    while graph_:
        graph,val = graph_.pop()
        reval = reval_.pop()  # each link *= other_G.aggH.valt
        if val > aveG:  # else graph is not re-inserted
            if reval < aveG:  # same graph, skip re-evaluation:
                regraph_+=[[graph,val]]; rreval_+=[0]
            else:
                regraph, reval = graph_reval([graph,val], fd)  # recursive depth-first node and link revaluation
                if regraph[1] > aveG:
                    regraph_ += [regraph]; rreval_+=[reval]
    if rreval_:
        if max([reval for reval in rreval_]) > aveG:
            regraph_ = graph_reval_(regraph_, rreval_, fd)  # graph reval while min val reduction

    return regraph_
