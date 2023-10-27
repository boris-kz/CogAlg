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
    Mdec,Ddec = 0,0  # max possible summed m|d, to compute relative summed m|d: V/maxV, link mediation coef
    Mval,Dval = 0,0; Mrdn,Drdn = 1,1

    # / P:
    mtuple, dtuple, Mtuple, Dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1, fagg=1)
    mval, dval = sum(mtuple), sum(abs(d) for d in dtuple)  # mval is signed, m=-min in comp x sign
    mrdn = dval>mval; drdn = dval<=mval
    derLay0 = [[mtuple,dtuple], [mval,dval], [mrdn,drdn]]
    L = len(mtuple)
    Mdec += sum([par/max for par,max in zip(mtuple,Mtuple)]) / L
    Ddec += sum([par/max for par,max in zip(dtuple,Dtuple)]) / L
    Mval+=mval; Dval+=dval; Mrdn += mrdn; Drdn += drdn
    # / PP:
    _derH,derH = _G.derH,G.derH
    if _derH[0] and derH[0]:  # empty in single-node Gs
        L += min(len(_derH[0]),len(derH[0]))
        dderH, valt, rdnt, dect = comp_derH(_derH[0], derH[0], rn=1, fagg=1)
        mdec,ddec = dect; Mdec = (Mdec+mdec)/2; Ddec = (Ddec+ddec)/2  # averages
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
    else:
        dderH = []
    derH = [[derLay0]+dderH, [Mval,Dval], [Mrdn,Drdn], [Mdec, Ddec]]  # appendleft derLay0 from comp_ptuple
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval],[Mrdn,Drdn], [Mdec,Ddec])
    SubH = [der_ext, derH]  # two init layers of SubH, higher layers added by comp_aggH:
    # / G:
    if fd:  # else no aggH yet?
        subH, valt, rdnt, dect = comp_aggH(_G.aggH, G.aggH, rn=1)
        SubH += subH  # append higher subLayers: list of der_ext | derH s
        mdec,ddec = dect; link.dect = [(Mdec+mdec)/2,(Ddec+ddec)/2]  # averages
        mval,dval = valt; Mval+=mval; Dval+=dval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
        link_ += [link]

    elif Mval > ave_Gm or Dval > ave_Gd:  # or sum?
        link.subH = SubH; link.dect = [Mdec,Ddec]; link.valt = [Mval,Dval]; link.rdnt = [Mrdn,Drdn]  # complete proto-link
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

def agg_recursion(rroot, root, G_, fd):  # compositional agg+|sub+ recursion in root graph, clustering G_

    Val,Rdn = comp_G_(G_, fd)  # rng|der cross-comp all Gs, form link_H[-1] per G, sum in Val,Rdn
    if Val > ave*Rdn > ave:
        # else no clustering, same in agg_recursion?
        pP_v = cluster_params(parH=root.aggH, rVal=0,rRdn=0,rMax=0, fd=fd, G=root)  # pP_t: part_P_t
        root.valHt[fd] += [0 for pP in pP_v]; root.rdnHt[fd] += [1 for pP in pP_v]  # sum in form_graph_t feedback, +root.maxHt[fd]?

        GG_pP_t = form_graph_t(root, G_, pP_v)  # eval sub+ and feedback per graph
        # agg+ xcomp-> form_graph_t loop sub)agg+, vs. comp_slice:
        # sub+ loop-> eval-> xcomp
        for GG_ in GG_pP_t:  # comp_G_ eval: ave_m * len*rng - fixed cost, root update in form_t:
            if sum(root.valHt[0][-1]) * (len(GG_)-1)*root.rng > G_aves[fd] * sum(root.rdnHt[0][-1]):
                agg_recursion(rroot, root, GG_, fd=0)  # 1st xcomp in GG_

        G_[:] = GG_pP_t

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

def prune_graph_(root, graph_, fd):  # compute graph overlap to prune weak graphs, not nodes: rdn doesn't change the structure
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
                pruned_graph_ += [sum2graph(root, graph, fd)]
            else:
                for node in graph[0]:
                    node.root_t[fd].remove(graph)

    return pruned_graph_

def cluster_params(parH, rVal,rRdn,rMax, fd, G=None):  # G for parH=aggH

    part_P_ = []  # pPs: nested clusters of >ave param tuples, as below:
    part_ = []  # [[subH, sub_part_P_t], Val,Rdn,Max]
    Val, Rdn, Max = 0, 0, 0
    parH = copy(parH)
    i=1
    while parH:  # aggH | subH | derH, top-down
        subH = parH.pop(); fsub=1  # if parH is derH, subH is ptuplet
        if G:  # parH is aggH
            val=G.valHt[fd][-i]; rdn=G.rdnHt[fd][-i]; max=G.maxHt[fd][-i]; i+=1
        elif isinstance(subH[0][0], list):  # subH is subH | derH
            subH, valt, rdnt, maxt = subH   # unpack subHt
            val=valt[fd]; rdn=rdnt[fd]; max=maxt[fd]
        else:  # extt in subH or ptuplet in derH
            valP_t = [[cluster_vals(ptuple) for ptuple in subH if sum(ptuple)>ave]]
            if valP_t:
                part_ += [valP_t]  # params=vals, no sum-> Val,Rdn,Max?
            else:
                if Val:  # empty valP_ terminates root pP
                    part_P_ += [[part_,Val,Rdn,Max]]; rVal+=Val; rRdn+=Rdn; rMax+=Max  # root values
                part_=[]; Val,Rdn,Max = 0,0,0  # reset
            fsub=0
        if fsub:
            if val > ave:  # recursive eval,unpack
                Val+=val; Rdn+=rdn; Max+=max  # summed with sub-values:
                sub_part_P_t = cluster_params(subH, Val,Rdn,Max, fd)
                part_ += [[subH, sub_part_P_t]]
            else:
                if Val:  # empty sub_pP_ terminates root pP
                    part_P_ += [[part_,Val,Rdn,Max]]; rVal+=Val; rRdn+=Rdn; rMax+=Max  # root values
                part_=[]; Val,Rdn,Max = 0,0,0  # reset
    if part_:
        part_P_ += [[part_,Val,Rdn,Max]]; rVal+=Val; rRdn+=Rdn; rMax+=Max

    return [part_P_,rVal,rRdn,rMax]  # root values

def form_mediation_layers(layer, layers, fder):  # layers are initialized with same nodes and incrementally mediated links

    # form link layers to back-propagate overlap of root graphs to segment node_, pruning non-max roots?
    out_layer = []; out_val = 0   # new layer, val

    for (node, _links, _nodes, Nodes) in layer:  # higher layers have incrementally mediated _links and _nodes
        links, nodes = [], []  # per current-layer node
        Val = 0
        for _node in _nodes:
            for link in _node.link_H[-(1+fder)]:  # mediated links
                __node = link._G if link.G is _node else link.G
                if __node not in Nodes:  # not in lower-layer links
                    nodes += [__node]
                    links += [link]  # to adjust link.val in suppress_overlap
                    Val += link.valt[fder]
        # add fork val of link layer:
        node.val_Ht[fder] += [Val]
        out_layer += [[node, links, nodes, Nodes+nodes]]  # current link mediation order
        out_val += Val  # no permanent val per layer?

    layers += [out_layer]
    if out_val > ave:
        form_mediation_layers(out_layer, layers, fder)

def cluster_params(parHv, fderlay, fd):  # last v: value tuple valt,rdnt,maxt

    parH, rVal, rRdn, rMax = parHv  # compressed valt,rdnt,maxt per aggH replace initial summed G vals
    part_P_ = []  # pPs: nested clusters of >ave param tuples, as below:
    part_ = []  # [[subH, sub_part_P_t], Val,Rdn,Max]
    Val, Rdn, Max = 0, 0, 0
    parH = copy(parH)
    while parH:  # aggHv | subHv | derHv (ptupletv_), top-down
        subt = parH.pop()   # Hv | extt | ptupletv
        if fderlay:
            cluster_ptuplet(subt, [part_P_,rVal,rRdn,rMax], [part_,Val,Rdn,Max], v=1)  # subt is ptupletv
        else:
            for sub in subt:
                if isinstance(sub[0][0],list):
                    subH, valt, rdnt, maxt = sub  # subt is Hv
                    val, rdn, max = valt[fd],rdnt[fd],maxt[fd]
                    if val > ave:  # recursive eval,unpack
                        Val+=val; Rdn+=rdn; Max+=max  # summed with sub-values
                        # unpack tuple
                        sub_part_P_t = cluster_params([sub[0], sub[1][fd], sub[2][fd],sub[3][fd]], fderlay=1, fd=fd)
                        # each element in sub[0] is derLay
                        part_ += [[subH, sub_part_P_t]]
                    else:
                        if Val:  # empty sub_pP_ terminates root pP
                            part_P_ += [[part_,Val,Rdn,Max]]; rVal+=Val; rRdn+=Rdn; rMax+=Max  # root params
                            part_= [], Val,Rdn,Max = 0,0,0  # pP params (reset)
                else:
                    cluster_ptuplet(sub, [part_P_,rVal,rRdn,rMax], [part_,Val,Rdn,Max], v=0)  # sub is extt
    if part_:
        part_P_ += [[part_,Val,Rdn,Max]]; rVal+=Val; rRdn+=Rdn; rMax+=Max

    return [part_P_,rVal,rRdn,rMax]  # root values

def sum_link_tree_(G_,fd):  # sum surrounding link values to define connected nodes, with indirectly incr rng, to parallelize:
                            # link lower nodes via incr n of higher nodes, added till fully connected, n layers = med rng?
    ave = G_aves[fd]
    graph_ = []
    for i, G in enumerate(G_):
        G.it[fd] = i  # used here and segment_node_
        graph_ += [[G, G.valHt[fd][-1],G.rdnHt[fd][-1]]]  # init val,rdn
    # eval incr mediated links, sum decayed surround node Vals, while significant Val update:
    _Val,_Rdn = 0,0
    while True:
        DVal,DRdn = 0,0
        Val,Rdn = 0,0  # updated surround of all nodes
        for i, (G,val,rdn, node_,perimeter) in enumerate(graph_):
            for link in G.link_H[-1]:
                if link.valt[fd] < ave * link.rdnt[fd]: continue  # skip negative links
                _G = link.G if link._G is G else link._G
                if _G not in G_: continue
                _graph = graph_[_G.it[fd]]
                _val = _graph[1]; _rdn = _graph[2]
                try: decay = link.valt[fd]/link.maxt[fd]  # val rng incr per loop, per node?
                except ZeroDivisionError: decay = 1
                Val += _val * decay; Rdn += _rdn * decay  # link decay coef: m|d / max, base self/same
                # prune links in segment_node_
            graph_[i][1] = Val; graph_[i][2] = Rdn  # unilateral update, computed separately for _G
            DVal += Val-_Val  # update / surround extension, signed
            DRdn += Rdn-_Rdn

        if DVal < ave*DRdn:  # even low-Dval extension may be valuable if Rdn decreases?
            break
        _Val,_Rdn = Val,Rdn
    return graph_

def segment_node_par(root, G_,fd):  # sum surrounding link values to define connected nodes, incrementally mediated

    ave = G_aves[fd]
    graph_ = []
    for G in G_:
        graph_ += [[G, G.valHt[fd][-1],G.rdnHt[fd][-1], [G],[G]]]  # init val,rdn, node_,perimeter
    _Val, _Rdn = 0, 0

    while True:  # eval incr mediated links, sum perimeter Vals, append node_, while significant Val update:
        DVal,DRdn, Val,Rdn = 0,0, 0,0
        # update surround per node:
        for G, val,rdn, node_,perimeter in graph_:
            new_perimeter = []
            for node in perimeter:
                periVal, periRdn = 0,0
                for link in node.link_H[-1]:
                    _node = link.G if link._G is G else link._G
                    if _node not in G_ or _node in node_: continue
                    j = _node.i; _val = graph_[j][1]; _rdn = graph_[j][2]
                    # use relative link vals only:
                    try: decay = link.valt[fd]/link.maxt[fd]  # link decay coef: m|d / max, base self/same
                    except ZeroDivisionError: decay = 1
                    # sum mediated vals per node and perimeter node:
                    med_val = (val+_val) * decay; Val += med_val; periVal += med_val
                    med_rdn = (rdn+_rdn) * decay; Rdn += med_rdn; periRdn += med_rdn
                    new_perimeter += [_node]
                    node_ += [_node]
                k = node.i; graph_[k][1] += periVal; graph_[k][2] += periRdn

            i = G.i; graph_[i][1] = Val; graph_[i][2] = Rdn
            DVal += Val-_Val; DRdn += Rdn-_Rdn  # update / surround extension, signed
            perimeter[:] = new_perimeter
        if DVal < ave*DRdn:  # even low-Dval extension may be valuable if Rdn decreases?
            break
        _Val,_Rdn = Val,Rdn

    # prune non-max overlapping graphs:
    ipop_ = []
    for graph in graph_:
        node_ = graph[3]  # or Val-ave*Rdn:
        max_root_i = np.argmax([graph_[node.i][1] for node in node_])  # max root: graph nodes = graph roots, bilateral assign
        for i, root in enumerate(node_):  # reciprocal graph to graph_ refs
            if i != max_root_i:
                ipop_ += [root.i]  # index in graph_
    ipop_.sort(reverse=True)  # prevent skipping indices while popping
    [graph_.pop(i) for i in ipop_]  # graphs don't overlap, no need to remove individual nodes
    # prune weak graphs:
    cgraph_ = []
    for graph in graph_:
        if graph[1] > ave * graph[2]:  # Val > ave * Rdn
            cgraph_ += [sum2graph(root, graph, fd)]
    '''
    graph-parallel to cluster broad range of G_, same stop in overlapping Gs:
    graphs sum and buffer link tree Gs in their node_s, separate stopping

    runtime overlap | stop test, reuse PU for continuing | new_node: added to node_ from Node_?
    else: 
    - graphs send their i,vals to roots of all Gs in their node_
    - each G selects max val root, sends deletes to other root graphs
    '''
    return cgraph_

def segment_node_(root, Gt_, fd):  # fold sum_link_tree_, as in agg_parP_

    link_map = defaultdict(list)   # make default for root.node_t?
    ave = G_aves[fd]
    for G,_,_ in Gt_:
        for derG in G.link_H[-1]:
            if derG.valt[fd] > ave * derG.rdnt[fd]:  # or link val += node Val: prune +ve links to low Vals?
                link_map[G] += [derG._G]  # keys:Gs, vals: linked _G_s
                link_map[derG._G] += [G]
    graph_ = []
    # initialize proto-graphs with each node, eval links to add other nodes, skip added nodes next:
    for iG, iVal, iRdn in Gt_:
        if iVal > ave * iRdn and not iG.root[fd]:
            try: dec = iG.valHt[fd][-1] / iG.maxHt[fd][-1]
            except ZeroDivisionError: dec = 1  # add internal layers Val *= current-layer decay to init graph totals:
            tVal = iVal + sum(iG.valHt[fd]) * dec
            tRdn = iRdn + sum(iG.rdnHt[fd]) * dec
            cG_ = [iG]; iG.root[fd] = cG_  # clustered Gs
            perimeter = link_map[iG]       # recycle perimeter in breadth-first search, outward from iG:
            while perimeter:
                _G = perimeter.pop(0)
                for link in _G.link_H[-1]:
                    G = link.G if link._G is _G else link._G
                    if G in cG_ or G not in [Gt[0] for Gt in Gt_]: continue   # circular link
                    Gt = Gt_[G.it[fd]]; Val = Gt[1]; Rdn = Gt[2]
                    if Val > ave * Rdn:
                        try: decay = G.valHt[fd][-1] / G.maxHt[fd][-1]  # current link layer surround decay
                        except ZeroDivisionError: decay = 1
                        tVal += Val + sum(G.valHt[fd])*decay  # ext+ int*decay: proj match to distant nodes in higher graphs?
                        tRdn += Rdn + sum(G.rdnHt[fd])*decay
                        cG_ += [G]; G.root[fd] = cG_
                        perimeter += [G]
            if tVal > ave * tRdn:
                graph_ += [sum2graph(root, cG_, fd)]  # convert to Cgraphs

    return graph_

def sum2graph(root, cG_, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    graph = Cgraph(root=root, fd=fd, L=len(cG_))  # n nodes, transplant both node roots
    SubH = []; maxM,maxD, Mval,Dval, Mrdn,Drdn = 0,0, 0,0, 0,0
    Link_= []
    for G in cG_:
        # sum nodes in graph:
        sum_box(graph.box, G.box)
        sum_ptuple(graph.ptuple, G.ptuple)
        sum_derH(graph.derH, G.derH, base_rdn=1)
        sum_aggH(graph.aggH, G.aggH, base_rdn=1)
        sum_Hts(graph.valHt, graph.rdnHt, graph.maxHt, G.valHt, G.rdnHt, G.maxHt)

        subH=[]; mval,dval, mrdn,drdn, maxm,maxd = 0,0, 0,0, 0,0
        for derG in G.link_H[-1]:
            if derG.valt[fd] > G_aves[fd] * derG.rdnt[fd]:  # sum positive links only:
                (_mval,_dval),(_mrdn,_drdn),(_maxm,_maxd) = derG.valt, derG.rdnt, derG.maxt
                if derG not in Link_:
                    sum_subH(SubH, derG.subH, base_rdn=1)  # new aggLev, not from nodes: links overlap
                    Mval+=_mval; Dval+=_dval; Mrdn+=_mrdn; Drdn+=_drdn; maxM+=_maxm; maxD+=_maxd
                    graph.A[0] += derG.A[0]; graph.A[1] += derG.A[1]; graph.S += derG.S
                    Link_ += [derG]
                mval+=_mval; dval+=_dval; mrdn+=_mrdn; drdn+=_drdn; maxm+=_maxm; maxd+=_maxd
                sum_subH(subH, derG.subH, base_rdn=1, fneg = G is derG.G)  # fneg: reverse link sign
                sum_box(G.box, derG.G.box if derG._G is G else derG._G.box)
        # from G links:
        if subH: G.aggH += [subH]
        G.valHt[0]+=[mval]; G.valHt[1]+=[dval]; G.rdnHt[0]+=[mrdn]; G.rdnHt[1]+=[drdn]
        G.maxHt[0]+=[maxm]; G.maxHt[1]+=[maxd]
        G.root[fd] = graph  # replace cG_
        graph.node_t += [G]  # converted to node_t by feedback
    # + link layer:
    graph.valHt[0]+=[Mval]; graph.valHt[1]+=[Dval]; graph.rdnHt[0]+=[Mrdn]; graph.rdnHt[1]+=[Drdn]
    graph.maxHt[0]+=[maxM]; graph.maxHt[1]+=[maxD]

    return graph


def comp_ptuple(_ptuple, ptuple, rn, fagg=0):  # 0der params

    I, G, M, Ma, (Dy, Dx), L = _ptuple
    _I, _G, _M, _Ma, (_Dy, _Dx), _L = ptuple

    dI = _I - I*rn;  mI = ave-dI
    dG = _G - G*rn;  mG = min(_G, G*rn) - ave
    dL = _L - L*rn;  mL = min(_L, L*rn) - ave
    dM = _M - M*rn;  mM = match_func(_M, M*rn) - ave  # M, Ma may be negative
    dMa= _Ma- Ma*rn; mMa = match_func(_Ma, Ma*rn) - ave
    mAngle, dAngle = comp_angle((_Dy,_Dx), (Dy,Dx))

    mtuple = [mI, mG, mM, mMa, mAngle, mL]
    dtuple = [dI, dG, dM, dMa, dAngle, dL]
    if fagg:
        Mtuple = [max(_I,I), max(_G,G), max(_M,M), max(_Ma,Ma), 2, max(_L,L)]
        Dtuple = [abs(_I)+abs(I), abs(_G)+abs(G), abs(_M)+abs(M), abs(_Ma)+abs(Ma), 2, abs(_L)+abs(L)]

    ret = [mtuple, dtuple]
    if fagg: ret += [Mtuple, Dtuple]
    return ret