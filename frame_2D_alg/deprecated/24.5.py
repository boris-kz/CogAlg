def segment_graph(root, Q, fd, nrng):  # eval rim links with summed surround vals for density-based clustering
    '''
    kernels = get_max_kernels(Q)  # parallelization of link tracing, not urgent
    grapht_ = select_merge(kernels)  # refine by match to max, or only use it to selectively merge?
    '''
    igraph_ = []; ave = G_aves[fd]
    # graph += node if >ave ingraph connectivity in recursively refined kernel, init per node|link:
    for e in Q:
        rim = e.rim if isinstance(e, CG) else e.rimt__[-1][-1][0] + e.rimt__[-1][-1][1]
        uprim = [link for link in rim if link.Et[fd] > ave]  # fork eval
        if uprim:  # skip nodes without add new added rim
            grapht = [[e,*e.kH], [uprim], [*e.DerH.Et], uprim]  # link_ = updated rim, +=e.DerH in sum2graph?
            e.root = grapht  # for merging
            igraph_ += [grapht]
        else: e.root = None
    _graph_ = copy(igraph_)

    while _graph_:  # grapht is nodes connected by kernels
        graph_= []  # graph_+= grapht if >ave dV: inclusion V update
        for grapht in _graph_:
            if grapht not in igraph_: continue  # skip merged graphs
            # add/remove in-graph links in node_ rims:
            G_, Link_, Et, Rim = grapht
            inVal, inRdn = 0,0  # new in-graph +ve
            new_Rim = []
            for link in Rim:  # newly external links, or all updated links?
                if link.node_[0] in G_: G,_G = link.node_  # one of the nodes is already clustered
                else:                   _G,G = link.node_
                if _G in G_: continue
                # eval links by combination direct and node-mediated connectivity, recursive refine by in-graph kernels:
                _val,_rdn = _G.Et[fd::2]; val,rdn = G.Et[fd::2]  # or DerH.Et?
                lval,lrdn = link.Et[fd::2]
                decay = (link.relt[fd] / (link.derH.n * 6)) ** nrng  # normalized decay at current mediation
                V = lval + ((val+_val) * decay) * .1  # med connectivity coef?
                R = lrdn + (rdn+_rdn) * .1  # no decay
                if V > ave * R:  # connect by rel match of nodes * match of node Vs: surround M|Ds,
                    # link.ExtH.add_(dderH)
                    # link.Et[fd] = V, link.Et[2+fd] = R
                # below is not updated
                # cval suggests how deeply inside the graph is G:
                cval = link.Et[fd] + get_match(_G.Et[fd], G.Et[fd])  # same coef for int and ext match?
                crdn = link.Et[2+fd] + (_G.Et[2+fd] + G.Et[2+fd]) / 2
                if cval > ave * crdn:  # _G and its root are effectively connected
                    # merge _G.root in grapht:
                    if _G.root:
                        _grapht = _G.root  # local and for feedback?
                        _G_,_Link_,_Et,_Rim = _grapht
                        if link not in Link_: Link_ += [link]
                        Link_[:] += [_link for _link in _Link_ if _link not in Link_]
                        for g in _G_:
                            g.root = grapht
                            if g not in G_: G_+=[g]
                        Et[:] = np.add(Et,_Et)
                        inVal += _Et[fd]; inRdn += _Et[2+fd]
                        igraph_.remove(_grapht)
                        new_Rim += [link for link in _Rim if link not in new_Rim+Rim+Link_]
                    else:  # _G doesn't have uprim and doesn't form any grapht
                        _G.root = grapht
                        G_ += [_G]
            # for next loop:
            if len(new_Rim) * inVal > ave * inRdn:
                grapht.pop(-1); grapht += [new_Rim]
                graph_ += [grapht]  # replace Rim
        # select graph expansion:
        if graph_: _graph_ = graph_
        else: break
    graph_ = []
    for grapht in igraph_:
        if grapht[2][fd] > ave * grapht[2][2+fd]:  # form Cgraphs if Val > ave* Rdn
            graph_ += [sum2graph(root, grapht[:3], fd, nrng)]

    return graph_

# non-recursive inclusion per single link
def segment_graph(root, Q, fd, nrng):  # eval rim links with summed surround vals for density-based clustering
    '''
    kernels = get_max_kernels(Q)  # parallelization of link tracing, not urgent
    grapht_ = select_merge(kernels)  # refine by match to max, or only use it to selectively merge?
    '''

    grapht_ = []
    # node_|link_
    for node in copy(Q):  # depth-first eval merge nodes connected via their rims|kernels:
        if node not in Q: continue  # already merged
        grapht = [[],[],[0,0,0,0],[]]  # G_, Link_, Et, adjacent nodes
        Q.remove(node)
        grapht_ += [grapht]
        merge_node(Q, grapht_, grapht, node, fd)  # default for initialization
    # form Cgraphs if Val > ave* Rdn:
    return [sum2graph(root, grapht[:3], fd, nrng) for grapht in grapht_ if  grapht[2][fd] > ave * grapht[2][2+fd]]

