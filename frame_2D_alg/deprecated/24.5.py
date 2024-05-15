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

'''
        # sparsity = 1/nlinks in rim? or summed node_ S?
        if _G.rim_t: _L = len(_G.rim_t[0][-1]) if _G.rim_t[0] else 0 + len(_G.rim_t[1][-1]) if _G.rim_t[1] else 0
        else:        _L = len(_G.node_[0].rim)+len(_G.node_[1].rim)
        if G.rim_t: L = len(G.rim_t[0][-1]) if G.rim_t[0] else 0 + len(G.rim_t[0][-1]) if G.rim_t[0] else 0
        else:       L = len(G.node_[0].rim)+len(G.node_[1].rim)
'''

def rng_recursion(PP, rng=1, fd=0):  # similar to agg+ rng_recursion, but looping and contiguously link mediated

    iP_, nP_  = PP.P_, []  # nP = new P with added links
    rrdn = 2  # cost of links added per rng+
    while True:
        P_ = []; V = 0
        for P in iP_:
            if not P.link_: continue
            prelink_ = []  # new prelinks per P
            if fd: _prelink_ = unpack_last_link_(P.link_)  # reuse links in der+
            else:  _prelink_ = P.link_.pop()  # old rng+ prelinks, including all links added in slice_edge
            for link in _prelink_:
                if link.distance <= rng:  # | rng * ((P.val+_P.val)/ ave_rval)?
                    _P = link.node_[0]
                    if fd and not (P.derH and _P.derH): continue  # nothing to compare
                    mlink = comp_P(link, fd)
                    if mlink:  # return if match
                        if P not in nP_: nP_ += [P]
                        V += mlink.derH.Et[0]
                        if rng > 1:  # test to add nesting to P.link_:
                            if rng == 2 and not isinstance(P.link_[0], list): P.link_[:] = [P.link_[:]]  # link_ -> link_H
                            if len(P.link_) < rng: P.link_ += [[]]  # add new link_
                        link_ = unpack_last_link_(P.link_)
                        if not fd: link_ += [mlink]
                        _link_ = unpack_last_link_(_P.link_ if fd else _P.link_[:-1])  # skip prelink_ if rng+
                        prelink_ += _link_  # connected __Ps links
            P.link_ += [prelink_]  # temporary pre-links, may be empty
            if prelink_: P_ += [P]
        rng += 1
        if V > ave * rrdn * len(P_) * 6:  #  implied val of all __P_s, 6: len mtuple
            iP_ = P_; fd = 0
            rrdn += 1
        else:
            for P in PP.P_:
                if P.link_: P.link_.pop()  # remove prelinks in rng+
            break
    # der++ in PPds from rng++, no der++ inside rng++: high diff @ rng++ termination only?
    PP.rng=rng  # represents rrdn

    return nP_

def unpack_last_link_(link_):  # unpack last link layer

    while link_ and isinstance(link_[-1], list): link_ = link_[-1]
    return link_
    # change to use nested link_, or
    # get higher-P mediated link_s while recursion_count < rng?
