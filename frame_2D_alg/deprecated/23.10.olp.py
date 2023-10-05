def form_graph_t(root, G_, _root_t_):  # root function to form fuzzy graphs of nodes per fder,fd

    graph_t = []
    for fd in 0,1:
        Gt_ = eval_node_connectivity(G_, fd)  # sum surround link values @ incr rng,decay: val += linkv/maxv * _node_val
        init_ = select_init_(Gt_, fd)  # select sparse max nodes to initialize graphs
        graph_ = segment_node_(init_, Gt_, fd, _root_t_)
        graph_ = prune_graph_(graph_, fd)  # sort node roots to add to graph rdn, prune weak graphs
        graph_ = sum2graph_(graph_, fd)  # convert to Cgraphs
        graph_t += [graph_]  # add alt_graphs?
    # sub+:
    for i in 0,1:
        root.val_Ht[i] += [0]; root.rdn_Ht[i] += [0]
    for fd, graph_ in enumerate(graph_t):  # breadth-first for in-layer-only roots
        for graph in graph_:  # external to agg+ vs internal in comp_slice sub+
            node_ = graph.node_t  # still flat?  eval fd comp_G_ in sub+:
            if sum(graph.val_Ht[fd]) * (len(node_)-1)*root.rng > G_aves[fd] * sum(graph.rdn_Ht[fd]):
                agg_recursion(root, graph, node_, fd)  # replace node_ with node_t, recursive
            else:  # feedback after graph sub+:
                root.fback_t[fd] += [[graph.aggH, graph.val_Ht, graph.rdn_Ht]]  # merge forks in root fork
                root.val_Ht[fd][-1] += graph.val_Ht[fd][-1]  # last layer, or all new layers via feedback?
                root.rdn_Ht[fd][-1] += graph.rdn_Ht[fd][-1]
            i = sum(graph.val_Ht[0]) > sum(graph.val_Ht[1])
            root.rdn_Ht[i][-1] += 1  # add fork rdn to last layer, representing all layers after feedback?
        # recursive feedback after all G_ sub+:
        if root.fback_t and root.fback_t[fd]:
            feedback(root, fd)  # update root.root.. aggH, val_Ht,rdn_Ht

    return graph_t  # root.node_t'node_ -> node_t: incr nested with each agg+?

def sum2graph_(graph_, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    Graph_ = []
    for graph in graph_:      # seq graphs
        Root_t = graph[2][0]  # merge _root_t_ into Root_t:
        [merge_root_tree(Root_t, root_t) for root_t in graph[2][1:]]
        Graph = Cgraph(fd=fd, root_t=Root_t, L=len(graph[0]))  # n nodes
        Link_ = []
        for G in graph[0]:
            sum_box(Graph.box, G.box)
            sum_ptuple(Graph.ptuple, G.ptuple)
            sum_derH(Graph.derH, G.derH, base_rdn=1)  # base_rdn?
            sum_aggH([Graph.aggH,Graph.val_Ht,Graph.rdn_Ht], [G.aggH,G.val_Ht,G.rdn_Ht], base_rdn=1)
            link_ = G.link_H[-1]
            subH=[]; valt=[0,0]; rdnt=[1,1]
            for derG in link_:
                if derG.valt[fd] > G_aves[fd] * derG.rdnt[fd]:  # sum positive links only:
                    if derG not in Link_: Link_ += [derG]
                    sum_subH([subH,valt,rdnt], [derG.subH,derG.valt,derG.rdnt], base_rdn=1, fneg = G is derG.G)  # fneg: reverse link sign
                    sum_box(G.box, derG.G.box if derG._G is G else derG._G.box)
            G.aggH += [[subH,valt,rdnt]]
            for i in 0,1:  # append external links val to internal links vals:
                G.val_Ht[i] += [valt[i]]; G.rdn_Ht[i] += [rdnt[i]]
            G.root_t[fd][G.root_t[fd].index(graph)] = Graph  # replace list graph in root_t
            Graph.node_t += [G]  # converted to node_t by feedback

        subH=[]; valt=[0,0]; rdnt=[1,1]  # sum unique positive links:
        for derG in Link_:
            sum_subH([subH,valt,rdnt], [derG.subH, derG.valt, derG.rdnt], base_rdn=1)
            Graph.A[0] += derG.A[0]; Graph.A[1] += derG.A[1]
            Graph.S += derG.S
        Graph.aggH += [subH]  # new aggLev, not summed from the nodes because their links overlap
        for i in 0,1:  # aggLev should not have valt,rdnt, it's redundant to G val_Ht, rdn_Ht:
            Graph.val_Ht[i] += [valt[i]]; Graph.rdn_Ht[i] += [rdnt[i]]
        Graph_ += [Graph]

    return Graph_

def merge_root_tree(Root_t, root_t):  # not-empty fork layer is root_t, each fork may be empty list:

    for Root_, root_ in zip(Root_t, root_t):
        for Root, root in zip(Root_, root_):
            if root.root_t:  # not-empty root fork layer
                if Root.root_t: merge_root_tree(Root.root_t, root.root_t)
                else: Root.root_t[:] = root.root_t
        Root_[:] = list( set(Root_+root_))  # merge root_, may be empty


def select_init_(Gt_, fd):  # local max selection for sparse graph init, if positive link

    init_, non_max_ = [],[]  # pick max in direct links, no recursively mediated links max: discontinuous?

    for node, val in Gt_:
        if node in non_max_: continue  # can't init graph
        if val<=0:  # no +ve links
            if sum(node.val_Ht[fd]) > ave * sum(node.rdn_Ht[fd]):
                init_+= [[node, 0]]  # single-node proto-graph
            continue
        fmax = 1
        for link in node.link_H[-1]:
            _node = link.G if link._G is node else link._G
            if val > Gt_[_node.it[fd]][1]:
                non_max_ += [_node]  # skip as next node
            else:
                fmax = 0; break  # break is not necessary?
        if fmax:
            init_ += [[node,val]]
    return init_

def segment_node_(init_, Gt_, fd, root_t_):  # root_t_ for fuzzy graphs if partial param sets: sub-forks?

    graph_ = []  # initialize graphs with local maxes, eval their links to add other nodes:

    for inode, ival in init_:
        iroot_t = root_t_[inode.it[fd]]  # same order as Gt_, assign node roots to new graphs, init with max
        graph = [[inode], ival, [iroot_t]]
        inode.root_t[fd] += [graph]
        _nodet_ = [[inode, ival, [iroot_t]]]  # current perimeter of the graph, init = graph

        while _nodet_:  # search links outwards recursively to form overlapping graphs:
            nodet_ = []
            for _node, _val, _root_t in _nodet_:
                for link in _node.link_H[-1]:
                    node = link.G if link._G is _node else link._G
                    if node in graph[0]: continue
                    val = Gt_[node.it[fd]][1]
                    root_t = root_t_[node.it[fd]]
                    if val * (link.valt[fd] / link.maxt[fd]) > ave:  # node val * link decay
                        if node.root_t[fd]:
                            merge(graph, node.root_t[fd])  # or use dict link_map?
                        else:
                            graph[0] += [node]
                            graph[1] += val
                            graph[2] += [root_t]
                            nodet_ += [node, val, root_t]  # new perimeter
            _nodet_ = nodet_
        graph_ += [graph]
    return graph_

def prune_graph_(graph_, fd):  # compute graph overlap to prune weak graphs, not nodes: rdn doesn't change the structure
                               # prune rootless nodes?
    for graph in graph_:
        for node in graph[0]:  # root rank = graph/node rdn:
            roots = sorted(node.root_t[fd], key=lambda root: root[1], reverse=True)  # sort by net val, if partial param sub-forks
            # or grey-scale rdn = root_val / sum_higher_root_vals?
            for rdn, graph in enumerate(roots):
                graph[1] -= ave * rdn  # rdn to >val overlapping graphs per node, also >val forks, alt sparse param sets?
                # nodes are shared by multiple max-initialized graphs, pruning here still allows for some overlap
        pruned_graph_ = []
        for graph in graph_:
            if graph[1] > G_aves[fd]:  # rdn-adjusted Val for local sparsity, doesn't affect G val?
                pruned_graph_ += [graph]
            else:
                for node in graph[0]:
                    node.root_t[fd].remove(graph)
    return pruned_graph_


def comp_G(link_, link, fd):

    _G, G = link._G, link.G
    maxM,maxD = 0,0  # max possible summed m|d, to compute relative summed m|d: V/maxV, link mediation coef
    Mval,Dval = 0,0; Mrdn,Drdn = 1,1

    # / P:
    mtuple, dtuple, Mtuple, Dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1, fagg=1)
    maxm, maxd = sum(Mtuple), sum(Dtuple)
    mval, dval = sum(mtuple), sum(abs(d) for d in dtuple)  # mval is signed, m=-min in comp x sign
    mrdn = dval>mval; drdn = dval<=mval
    derLay0 = [[mtuple,dtuple], [mval,dval], [mrdn,drdn]]
    Mval+=mval; Dval+=dval; Mrdn += mrdn; Drdn += drdn; maxM+=maxm; maxD+=maxd
    # / PP:
    dderH, valt, rdnt, maxt = comp_derH(_G.derH[0], G.derH[0], rn=1, fagg=1)
    mval,dval = valt; maxm,maxd = maxt
    Mval+=dval; Dval+=mval; maxM+=maxm; maxD+=maxd
    Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval

    derH = [[derLay0]+dderH, [Mval,Dval], [Mrdn,Drdn]]  # appendleft derLay0 from comp_ptuple
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval],[Mrdn,Drdn],[maxM,maxD])
    SubH = [der_ext, derH]  # two init layers of SubH, higher layers added by comp_aggH:
    # / G:
    if fd:  # else no aggH yet?
        subH, valt, rdnt, maxt = comp_aggH(_G.aggH, G.aggH, rn=1)
        SubH += subH  # append higher subLayers: list of der_ext | derH s
        mval,dval = valt; maxm,maxd = maxt
        Mval+=mval; Dval+=dval; maxM += maxm; maxD += maxd
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
        link_ += [link]

    elif Mval > ave_Gm or Dval > ave_Gd:  # or sum?
        link.subH = SubH; link.maxt = [maxM,maxD]; link.valt = [Mval,Dval]; link.rdnt = [Mrdn,Drdn]  # complete proto-link
        link_ += [link]

def comp_aggH(_aggH, aggH, rn):  # no separate ext
    SubH = []
    maxM,maxD, Mval,Dval, Mrdn,Drdn = 0,0,0,0,1,1

    for _lev, lev in zip_longest(_aggH, aggH, fillvalue=[]):  # compare common lower layer|sublayer derHs
        if _lev and lev:  # also if lower-layers match: Mval > ave * Mrdn?
            # compare dsubH only:
            dsubH, valt,rdnt,maxt = comp_subH(_lev[0], lev[0], rn)
            SubH += dsubH  # flatten to keep subH
            mval,dval = valt; maxm,maxd = maxt
            Mval += mval; Dval += dval; maxM += maxm; maxD += maxd
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + mval <= dval

    return SubH, [Mval,Dval],[Mrdn,Drdn],[maxM,maxD]

def comp_derH(_derH, derH, rn, fagg=0):  # derH is a list of der layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    dderH = []  # or = not-missing comparand if xor?
    Mval, Dval, Mrdn, Drdn = 0,0,1,1
    if fagg: maxM, maxD = 0,0

    for _lay, lay in zip_longest(_derH, derH, fillvalue=[]):  # compare common lower der layers | sublayers in derHs
        if _lay and lay:  # also if lower-layers match: Mval > ave * Mrdn?

            ret = comp_dtuple(_lay[0][1], lay[0][1], rn, fagg)  # compare dtuples only, mtuples are for evaluation
            mtuple, dtuple = ret[:2]
            mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
            mrdn = dval > mval; drdn = dval < mval
            derLay = [[mtuple,dtuple],[mval,dval],[mrdn,drdn]]
            Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn
            if fagg:
                Mtuple, Dtuple = ret[2:]
                derLay[0] += [Mtuple,Dtuple]
                maxm = sum(Mtuple); maxd = sum(Dtuple)
                maxM += maxm; maxD += maxd
            dderH += [derLay]

    ret = [dderH, [Mval,Dval], [Mrdn,Drdn]]  # new derLayer,= 1/2 combined derH
    if fagg: ret += [[maxM,maxD]]
    return ret
