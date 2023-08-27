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

def add_nodes(node, GQt, link_, fder, fd):

    for link in link_:
        val = link.valt[fd]
        if val > G_aves[fd]:
            GQt[1] += val  # in-graph links val per node
            _node = link.G1 if link.G0 is node else link.G0
            if _node not in GQt[0][0]:
                _GQt = _node.root_T[fder][fd][0]   # [_GQ,_Val]
                # cross-assign nodes, pri_roots, accum val:
                if _GQt[0][1][0]:  # base fork pri_root is empty, and we can't use set with empty list
                    unique_pri_roots = list(set(_GQt[0][1] + GQt[0][1]))
                else:
                    unique_pri_roots = [[]]
                _GQt[0][1][:] = unique_pri_roots
                GQt[0][1] = unique_pri_roots
                link_ = _node.link_H[-(1+fder)]
                add_nodes(_node, GQt, link_, fder, fd)  # add indirect nodes and their roots recursively


def intra_blob_root(root_blob, render, verbose):  # recursive evaluation of cross-comp slice| range| per blob

    spliced_layers = []

    for blob in root_blob.rlayers[0]:
        # <--- extend der__t: cross-comp in larger kernels
        y0, yn, x0, xn = blob.box  # dert box
        Y, X = blob.root_der__t[0].shape  # root dert size
        # set pad size, e is for extended
        y0e = max(0, y0 - 1); yne = min(Y, yn + 1)
        x0e = max(0, x0 - 1); xne = min(X, xn + 1)
        # take extended der__t from part of root_der__t:
        blob.der__t = idert(
            *(par__[y0e:yne, x0e:xne] for par__ in blob.root_der__t))
        # extend mask__:
        blob.mask__ = np.pad(
            blob.mask__, ((y0 - y0e, yne - yn), (x0 - x0e, xne - xn)),
            constant_values=True, mode='constant')
        # extends blob box
        blob.box = (y0e, yne, x0e, xne)
        Ye = yne - y0e; Xe = xne - x0e
        # ---> end extend der__t
        # increment forking sequence: g -> r|v
        if Ye > 3 and Xe > 3:  # min blob dimensions: Ly, Lx
            # <--- r fork
            if (blob.G < aveR * blob.rdn and blob.sign and
                    Ye > root_blob.rng*2 + 3 and Xe > root_blob.rng*2 + 3):  # below-average G, eval for comp_r
                blob.rng = root_blob.rng + 1; blob.rdn = root_blob.rdn + 1.5  # sub_blob root values
                new_der__t, new_mask__ = comp_r(blob.der__t, blob.rng, blob.mask__)
                if new_mask__.shape[0] > 2 and new_mask__.shape[1] > 2 and False in new_mask__:
                    if verbose: print('fork: r')
                    sign__ = ave * (blob.rdn + 1) - new_der__t.g > 0  # m__ = ave - g__
                    # form sub_blobs:
                    sub_blobs, idmap, adj_pairs = flood_fill(
                        new_der__t, sign__, prior_forks=blob.prior_forks + 'r', verbose=verbose, mask__=new_mask__)
                    '''
                    adjust per average sub_blob, depending on which fork is weaker, or not taken at all:
                    sub_blob.rdn += 1 -|+ min(sub_blob_val, alt_blob_val) / max(sub_blob_val, alt_blob_val):
                    + if sub_blob_val > alt_blob_val, else -?  
                    adj_rdn = ave_nsub - len(sub_blobs)  # adjust ave cross-layer rdn to actual rdn after flood_fill:
                    blob.rdn += adj_rdn
                    for sub_blob in sub_blobs: sub_blob.rdn += adj_rdn
                    '''
                    assign_adjacents(adj_pairs)

                    sublayers = blob.rlayers
                    sublayers += [sub_blobs]  # next level sub_blobs, then add deeper layers of sub_blobs:
                    sublayers += intra_blob_root(blob, render, verbose)  # recursive eval cross-comp per blob
                    spliced_layers[:] += [spliced_layer + sublayer for spliced_layer, sublayer in
                                          zip_longest(spliced_layers, sublayers, fillvalue=[])]
            # ---> end r fork
            # <--- v fork
            if blob.G > aveG * blob.rdn and not blob.sign:  # above-average G, vectorize blob
                blob.rdn = root_blob.rdn + 1.5  # comp cost * fork rdn, sub_blob root values
                blob.prior_forks += 'v'
                if verbose: print('fork: v')
                # vectorize_root(blob, verbose=verbose)
            # ---> end v fork
    return spliced_layers


def comp_r(dert__, rng, mask__=None):
    '''
    If selective sampling: skipping current rim derts as kernel-central derts in following comparison kernels.
    Skipping forms increasingly sparse output dert__ for greater-range cross-comp, hence
    rng (distance between centers of compared derts) increases as 2^n, with n starting at 0:
    rng = 1: 3x3 kernel,
    rng = 2: 5x5 kernel,
    rng = 3: 9x9 kernel.
    Sobel coefficients to decompose ds into dy and dx:
    YCOEFs = np.array([-1, -2, -1, 0, 1, 2, 1, 0])
    XCOEFs = np.array([-1, 0, 1, 2, 1, 0, -1, -2])
        |--(clockwise)--+  |--(clockwise)--+
        YCOEF: -1  -2  -1  ¦   XCOEF: -1   0   1  ¦
                0       0  ¦          -2       2  ¦
                1   2   1  ¦          -1   0   1  ¦
    Scharr coefs:
    YCOEFs = np.array([-47, -162, -47, 0, 47, 162, 47, 0])
    XCOEFs = np.array([-47, 0, 47, 162, 47, 0, -47, -162])
    If skipping, configuration of input derts in next-rng kernel will always be 3x3, using Sobel coeffs:
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_comp_diagrams.png
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_comp_d.drawio
    '''
    ky, kx, km = compute_kernel(rng)
    # unmask all derts in kernels with only one masked dert (can be set to any number of masked derts),
    # to avoid extreme blob shrinking and loss of info in other derts of partially masked kernels
    # unmasked derts were computed in extend_dert()
    if mask__ is not None:
        majority_mask__ = convolve2d(mask__.astype(int), km, mode='valid') > 1
    else:
        majority_mask__ = None  # returned at the end of function
    # unpack dert__:
    i__, dy__, dx__, g__ = dert__  # i is pixel intensity, g is gradient
    # compare opposed pairs of rim pixels, project onto x, y:
    new_dy__ = dy__[rng:-rng, rng:-rng] + convolve2d(i__, ky, mode='valid')
    new_dx__ = dx__[rng:-rng, rng:-rng] + convolve2d(i__, kx, mode='valid')
    new_g__ = np.hypot(new_dy__, new_dx__)  # gradient, recomputed at each comp_r
    new_i__ = i__[rng:-rng, rng:-rng]

    return idert(new_i__, new_dy__, new_dx__, new_g__), majority_mask__


def form_graph_(node_, pri_root_T_, fder, fd):  # form fuzzy graphs of nodes per fder,fd, initially distributed across layers

    layer = []; Val = 0  # of new links

    for node in copy(node_):  # initial node_-> layer conversion
        links, _nodes = [],[node]
        val = 0  # sum all lower (link_val * med_coef)s per node, for node eval?
        for link in node.link_H[-(1+fder)]:
            _node = link.G1 if link.G0 is node else link.G0
            links += [link]; _nodes += [_node]
            val += link.valt[fder]
        # add fork val of direct (1st layer) links:
        node.val_Ht[fder] += [Val]
        layer += [[node, links, _nodes, copy(_nodes)]]
        Val += val
    if Val > ave:  # init hierarchical network:
        layers = [layer]
        # form layers of same nodes with incrementally mediated links:
        form_mediation_layers(layer, layers, fder=fder)
        # adjust link vals by stronger overlap per node across layers:
        suppress_overlap(layers, fder)
        # segment suppressed-overlap layers to graphs:
        return segment_network(layers, node_, pri_root_T_, fder, fd)
    else:
        return node_

def form_mediation_layers(layer, layers, fder):  # layers are initialized with same nodes and incrementally mediated links

    out_layer = []; out_val = 0   # new layer, val

    for (node, _links, _nodes, Nodes) in layer:  # higher layers have incrementally mediated _links and _nodes
        links, nodes = [], []  # per current-layer node
        Val = 0
        for _node in _nodes:
            for link in _node.link_H[-(1+fder)]:  # mediated links
                __node = link.G1 if link.G0 is _node else link.G0
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

# draft:
def suppress_overlap(layers, fder):  # adjust node vals by overlap, to combine with link val in segment

    # rdn: stronger overlap per node via NMS with linked nodes, direct for all-layers overlap,
    # or up to rng for partial overlap?
    Rdn = 0  # overlap: each positively linked node represents all others when segmented in graphs

    for i, layer in enumerate(layers):  # loop bottom-up, accum rdn per link from higher layers?
        for j, (node, links, nodes, Nodes) in enumerate(layer):
            if node.val_Ht[fder] > ave:
                # sort same-layer nodes to assign rdn: soft non-max?
                nodes = sorted(nodes, key=lambda node: sum(node.val_Ht[fder]), reverse=False)
                # or sort and rdn+ all lower-layer nodes by backprop: checked before but still add to rdn?
                for node in nodes:
                    val = sum(node.val_Ht[fder])  # to scale links for pruning in segment_network
                    node.Rdnt[fder] += val  # graph overlap: current+higher val links per node, all layers?
                    Rdn += val  # stronger overlap within layer
                # not updated:
                # backprop to more direct links per node:
                val = (Val / len(links)) * med_decay * i  # simplified redundancy and reinforcement by higher layer?
                for _layer in reversed(layers[:i]):  # loop down from current layer
                    _node, _links, _nodes, _Nodes = _layer[j]  # more direct links of same node in lower layer
                    for _link in _links:
                        _link.Valt[fder] += val  # direct links reinforced by ave mediated links val: higher layer,
                        # or segmentation eval per node' summed links, no link adjustment here?
                        if _link.valt[fder] < val:
                            _link.Rdnt[fder] += val  # average higher-layer val is rdn if greater?
    while Rdn > ave:
        suppress_overlap(layers, fder)

# old:
def add_alt_graph_(graph_t):  # mgraph_, dgraph_
    '''
    Select high abs altVal overlapping graphs, to compute value borrowed from cis graph.
    This altVal is a deviation from ave borrow, which is already included in ave
    '''
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            for node in graph.aggHs.H[-1].node_:
                for derG in node.link_.Q:  # contour if link.aggHs.val < aveGm: link outside the graph
                    for G in [derG.node0, derG.node1]:  # both overlap: in-graph nodes, and contour: not in-graph nodes
                        alt_graph = G.roott[1-fd]
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, CQ):  # not proto-graph or removed
                            graph.alt_graph_ += [alt_graph]
                            alt_graph.alt_graph_ += [graph]
                            # bilateral assign
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            if graph.alt_graph_:
                graph.alt_aggHs = CQ()  # players if fsub? der+: aggHs[-1] += player, rng+: players[-1] = player?
                for alt_graph in graph.alt_graph_:
                    sum_pH(graph.alt_aggHs, alt_graph.aggHs)  # accum alt_graph_ params
                    graph.alt_rdn += len(set(graph.aggHs.H[-1].node_).intersection(alt_graph.aggHs.H[-1].node_))  # overlap


def form_graph_(node_, pri_root_T_, fder, fd):  # form fuzzy graphs of nodes per fder,fd, within root

    ave = G_aves[fder]  # val = (Val / len(links)) * med_decay * i?
    _Rdn_, Rdn_ = [1 for node in node_], [1 for node in node_]  # accum >vals of linked nodes to reduce graph overlap
    dRdn = ave+1

    while dRdn > ave:  # iteratively propagate lateral Rdn (soft non-max per node) through mediating links, to define ultimate maxes?

        # or compute local_max(val + sum(surround_vals*mediation_decay)), with incremental mediation rng?
        # similar to Gaussian blurring, but only to select graph pivots?

        for node, Rdn in zip(node_, Rdn_):
            if (sum(node.val_Ht[fder]) * Rdn/(Rdn+ave))  - ave * sum(node.rdn_Ht[fder]):  # potential graph init
                nodet_ = []
                for link in node.link_H[-1]:
                    _node = link.G1 if link.G0 is node else link.G0
                    nodet_ += [[_node, link.valt[fder], Rdn_[node_.index(_node)]]]
                # [[_node,link_val,_rdn_val]]:
                nodet_ = sorted(nodet_+[[node,1,1]], key=lambda _nodet:  # lower-val node rdn += higher val*medDecay, because links maybe pruned:
                         sum(_nodet[0].val_Ht[fder]) - ave * sum(_nodet[0].rdn_Ht[fder]) * _nodet[1]/(_nodet[1]+ave) * _nodet[2]/(_nodet[2]+ave),
                         reverse=True)
                __node, __val, __Rdn = nodet_[0]  # local max  # adjust overlap val by mediating link vals:
                Rdn = sum(__node.val_Ht[fder]) *__val/(__val+ave) *__Rdn/(__Rdn+ave)  # effective overlap *= connection strength?
                # add to weaker nodes Rdn_[i], accum for next:
                for _node, _val, _Rdn in nodet_[1:]:
                    Rdn_[node_.index(_node)] += Rdn  # stronger-node-init graphs would overlap current-node-init graph
                    Rdn += sum(_node.val_Ht[fder]) * _val/(_val+ave) * _Rdn/(_Rdn+ave)  # adjust overlap by medDecay

        dRdn = sum([abs(Rdn-_Rdn) for Rdn,_Rdn in zip(Rdn_,_Rdn_)])
        _Rdn_ = Rdn_; Rdn_ = [1 for node in node_]
    '''            
    no explicit layers, recursion to re-sort and re-sum with previously adjusted directly-linked _node Rdns? 
    or form longer-range more mediated soft maxes, Rdn += stronger (med_Val * med_decay * med_depth)?
    '''
    # not revised, we may need to check mediating links to sparsify maxes:
    while True:
        for inode in node_:
            __node, __val, __Rdn = inode.link_H[-1][-1]  # local max
            for node, val, Rdn in inode.link_H[-1][1:]:  # mediated links
                 if (sum(node.val_Ht[fder]) * Rdn/ (Rdn+ave))  - ave * sum(node.rdn_Ht[fder]):
                    nodet_ = []
                    for link in node.link_H[-1]:  # mediated nodes
                        _node = link.G1 if link.G0 is node else link.G0
                        _Rdn = Rdn_[node_.index(_node)]
                        nodet_ += [[_node, link.valt[fder], _Rdn]]
                    # sort, lower-val node rdn += higher (val * rel_med_val: links may be pruned)s:
                    nodet_ = sorted(nodet_+[[node,1, 1]], key=lambda _nodet:  # add adjust by _Rdn here:
                            sum(_nodet[0].val_Ht[fder]) - ave * sum(_nodet[0].rdn_Ht[fder]) * (_nodet[1]/(_nodet[1]+ave)) * (_nodet[2]/(_nodet[2]+ave)) ,
                            reverse=True)
                    # check for new local max
                    if nodet_[0] > __val:
                        __node, __val, __Rdn = nodet_[0]
                        i = 1
                    else:
                        i = 0
                    Rdn = sum(__node.val_Ht[fder]) *__val/(__val+ave) *__Rdn/(__Rdn+ave)
                    for _node, _val, _Rdn in nodet_[i:]:
                        Rdn_[node_.index(_node)] += Rdn
                        Rdn += sum(_node.val_Ht[fder]) * _val/(_val+ave) * _Rdn/(_Rdn+ave)

                    # merge nodet_
                    for nodet in nodet_:
                        existing_node_ = [nodet[0] for nodet in inode.link_H[-1]]
                        if nodet[0] not in existing_node_:  # not packed in prior loop, prevent circular searching
                            inode.link_H[-1] += [nodet]

        # use dRdn to stop the recursion too?

    segment_network(nodet_, pri_root_T_, fder, fd)

# not updated, ~segment_network:
def form_graph_direct(node_, pri_root_T_, fder, fd):  # form fuzzy graphs of nodes per fder,fd, initially fully overlapping

    graph_ = []
    nodet_ = []
    for node, pri_root_T in zip(node_, pri_root_T_):
        nodet = [node, pri_root_T, 0]
        nodet_ += [nodet]
        graph = [[nodet], [pri_root_T], 0]  # init graph per node?
        node.root_T[fder][fd] = [[graph, 0]]  # init with 1st root, node-specific val
    # use popping?:
    for nodet, graph in zip(nodet_, graph_):
        graph_ += [init_graph(nodet, nodet_, graph, fder, fd)]  # recursive depth-first GQ_+=[_nodet]
    # prune by rdn:
    regraph_ = graph_reval_(graph_, fder,fd)  # init reval_ to start
    if regraph_:
        graph_[:] = sum2graph_(regraph_, fder, fd)  # sum proto-graph node_ params in graph

    # add_alt_graph_(graph_t)  # overlap+contour, cluster by common lender (cis graph), combined comp?
    return graph_

def init_graph(nodet, nodet_, graph, fder, fd):  # recursive depth-first GQ_+=[_nodet]

    node, pri_root_T, val = nodet

    for link in node.link_H[-(1+fder)]:
        if link.valt[fd] > G_aves[fd]:
            # link is in node GQ
            _nodet = link.G1 if link.G0 is nodet else link.G0
            if _nodet in nodet_:  # not removed in prior loop
                _node,_pri_root_T,_val = _nodet
                graph[0] += [_node]
                graph[2] += _val
                if _pri_root_T not in graph[1]:
                    graph[1] += [_pri_root_T]
                nodet_.remove(_nodet)
                init_graph(_nodet, nodet_, graph, fder, fd)
    return graph
'''
initially links only: plain attention forming links to all nodes, then segmentation by backprop of 
(sum of root vals / max_root_val) - 1, summed from all nodes in graph? 
while dMatch per cluster > ave: 
- sum cluster match,
- sum in-cluster match per node root: containing cluster, positives only,
- sort node roots by in_cluster_match,-> index = cluster_rdn for node,
- prune root if in_cluster_match < ave * cluster_redundancy, 
- prune weak clusters, remove corresponding roots
'''

def prune_graphs(_graph_, fder, fd):

    for _node_, _pri_root_T_, _Val in _graph_:
        node_, pri_root_T_ = [],[]
        prune_Val = -1
        graph_ = []
        for node in _node_:
            roots = sorted(node.root_T[fder][fd], key=lambda root: root[2], reverse=True)
            rdn_Val = 1  # val of stronger inclusion in overlapping graphs, in same fork

        # for _node_, _, _ in _graph_: I think this is not needed, we can just loop node_?
        for node in node_:
            roots = sorted(node.root_T[fder][fd], key=lambda root: root[2], reverse=True)
            # rdn_Val = 0  # val of stronger inclusion in overlapping graphs, in same fork (strongest root with 0 rdn)
            for root_rdn, root in enumerate(roots):
                Val = np.sum(node.val_Ht) - ave * (np.sum(node.rdn_Ht) + root_rdn)  # not sure
                if Val > 0:
                    prune_Val += Val
                    pruned_link_ = []  # per node
                    for link in node.link_H[fder]:
                        # tentative re-eval node links:
                        _node = link.G1 if link.G0 is node else link.G0
                        # prune links from node
                        if not np.sum(_node.val_Ht)  * (link.valt[fder]/link.valt[2]) - ave * (np.sum(_node.rdn_Ht[fder]) + root_rdn):
                            pruned_link_ += [link]
                    # prune links
                    for pruned_link in pruned_link_:
                        node.link_Ht[fder].remove(pruned_link)
                # if Val is < 0, increase their rdn so that we will not select them as max?
                else:
                    node.rdn_Ht[-1] += root_rdn * abs(Val)

        # reform graphs after pruning
        max_ = select_max_(node_, fder, ave)
        graph_ = segment_node_(node_, max_, pri_root_T_, fder, fd)

        if prune_Val < 0:
            break

    return graph_
# replace with prune_graphs:
def graph_reval_(graph_, fder,fd):

    regraph_ = []
    reval, overlap  = 0, 0
    for graph in graph_:
        for node in graph[0]:  # sort node root_(local) by root val (ascending)
            node.root_T[fder][fd] = sorted(node.root_T[fder][fd], key=lambda root:root[0][2], reverse=False)  # 2: links val
    while graph_:
        graph = graph_.pop()  # pre_graph or cluster
        node_, root_, val = graph
        remove_ = []  # sum rdn (overlap) per node root_T, remove if val < ave * rdn:
        for node in node_:
            rdn = 1 + len(node.root_T)
            if node.valt[fd] < G_aves[fd] * rdn:
            # replace with node.root_T[i][1] < G_aves[fd] * i:
            # val graph links per node, graph is node.root_T[i], i=rdn: index in fork root_ sorted by val
                remove_ += [node]       # to remove node from cluster
                for i, grapht in enumerate(node.root_T[fder][fd]):  # each root is [graph, Val]
                    if graph is grapht[0]:
                        node.root_T[fder][fd].pop(i)  # remove root cluster from node
                        break
        while remove_:
            remove_node = remove_.pop()
            node_.remove(remove_node)
            graph[2] -= remove_node.valt[fd]
            reval += remove_node.valt[fd]
        for node in node_:
            sum_root_val = sum([root[1] for root in node.root_T[fder][fd]])
            max_root_val = max([root[1] for root in node.root_T[fder][fd]])
            overlap += sum_root_val/ max_root_val
        regraph_ += [graph]

    if reval > ave and overlap > 0.5:  # 0.5 can be a new ave here
        regraph_ = graph_reval_(regraph_, fder,fd)

    return regraph_

# draft:
def graph_reval(graph, fd):  # exclusive graph segmentation by reval,prune nodes and links

    aveG = G_aves[fd]
    reval = 0  # reval proto-graph nodes by all positive in-graph links:

    for node in graph[0]:  # compute reval: link_Val reinforcement by linked nodes Val:
        link_val = 0
        _link_val = node.valt[fd]
        for derG in node.link_H[-1]:
            if derG.valt[fd] > [ave_Gm, ave_Gd][fd]:
                val = derG.valt[fd]  # of top aggH only
                _node = derG.G1 if derG.G0 is node else derG.G0
                # consider in-graph _nodes only, add to graph if link + _node > ave_comb_val?
                link_val += val + (_node.valt[fd]-val) * med_decay
        reval += _link_val - link_val  # _node.link_tH val was updated in previous round
    rreval = 0
    if reval > aveG:  # prune graph
        regraph, Val = [], 0  # reformed proto-graph
        for node in graph[0]:
            val = node.valt[fd]
            if val > G_aves[fd] and node in graph:
            # else no regraph += node
                for derG in node.link_H[-1][fd]:
                    _node = derG.G1 if derG.G0 is node else derG.G0  # add med_link_ val to link val:
                    val += derG.valt[fd] + (_node.valt[fd]-derG.valt[fd]) * med_decay
                Val += val
                regraph += [node]
        # recursion:
        if rreval > aveG and Val > aveG:
            regraph, reval = graph_reval([regraph,Val], fd)  # not sure about Val
            rreval += reval

    else: regraph, Val = graph

    return [regraph,Val], rreval

def prune_graph_(_graph_, fder, fd):

    graph_ = [[[],[],0] for graph in _graph_]  # init new graphs: [[node_, pri_root_tt_, Val]]

    for _node_, _pri_root_tt_, _Val in _graph_:
        for node, pri_root_tt  in zip(_node_, _pri_root_tt_):
            roots = sorted(node.root_tt[fder][fd], key=lambda root: root[2], reverse=True)
            for rdn, root in enumerate(roots):
                # rdn to stronger inclusion in same-fork overlapping graphs
                val = sum(node.val_Ht[fder]) - ave * (sum(node.rdn_Ht[fder]) + rdn)
                # no link pruning: they are not a component of graph, don't depend on rdn?
                if val > 0:
                    i = _graph_.index(root)  # current root graph of current node
                    graph_[i][0] += [node]
                    graph_[i][1] += [_pri_root_tt_]
                    graph_[i][2] += val
                else:
                    node.root_tt[fder][fd].remove(root)

    graph_ = [graph for graph in graph_ if graph[2] > G_aves[fder]]  # graph pruning
    graph_ = sum2graph_(graph_, fder, fd)
    return graph_