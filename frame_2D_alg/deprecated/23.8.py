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
