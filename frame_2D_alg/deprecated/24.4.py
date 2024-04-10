def node_connect(iG_):  # node connectivity = sum surround link vals, incr.mediated: Graph Convolution of Correlations
    '''
    Aggregate direct * indirect connectivity per node from indirect links via associated nodes, in multiple cycles.
    Each cycle adds contributions of previous cycles to linked-nodes connectivity, propagated through the network.
    Math: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/node_connect.png
    Replace mnode_, dnode_ with node_G_, link_G_: composed of links. Angle match | difference match in clustering links?
    Add backprop to revise G assignment: if M > (ave * len Rim) * (Rm / len Rim)?
    '''
    _G_ = iG_
    while True:
        # eval accumulated G connectivity with node-mediated range extension
        G_ = []  # next connectivity expansion, more selective by DV,Lent
        mediation = 1  # n intermediated nodes, increasing decay
        for G in _G_:
            uprim = []  # >ave updates of direct links
            for i in 0,1:
                val,rdn = G.Et[i::2]  # rng+ for both segment forks
                if not val: continue  # G has no new links
                ave = G_aves[i]
                for link in G.rim:
                    if len(link.dderH.H)<=len(G.extH.H): continue  # old links, else dderH+= in comp_G
                    # > ave derGs in new fd rim:
                    lval,lrdn,ldec = link.Et[i::2]  # step=2, graph-specific vals accumulated from surrounding nodes
                    decay =  (ldec/ (link.dderH.n * 6)) ** mediation  # normalized decay at current mediation
                    _G = link._node if link.node is G else link.node
                    _val,_rdn = _G.Et[i::2]
                    # current-loop vals and their difference from last-loop vals, before updating:
                    # += adjacent Vs * dec: lateral modulation of node value, vs. current vertical modulation by Kernel comp:
                    V = (val+_val) * decay; dv = V-lval
                    R = (rdn+_rdn)  # rdn doesn't decay
                    link.Et[i:4:2] = [V,R]  # last-loop vals for next loop | segment_node_, dect is not updated
                    if dv > ave * R:  # extend mediation if last-update val, may be negative
                        G.Et[i::2] = [V+v for V,v in zip(G.Et[i::2],[V,R])]  # last layer link vals
                        if link not in uprim: uprim += [link]
                    if V > ave * R:  # updated even if terminated
                        G.Et[i::2] = [V+v for V,v in zip(G.Et[i::2], [dv,R])]  # use absolute R?
            if uprim:
                G_ += [G]  # list of nodes to check in next loop
        if G_:
            mediation += 1  # n intermediated nodes in next loop
            _G_ = G_  # exclude weakly incremented Gs from next connectivity expansion loop
        else:
            break

def segment_node_(root, node_, fd, nrng, fagg):  # eval rim links with summed surround vals for density-based clustering

    # graph+= [node] if >ave (surround connectivity * relative value of link to any internal node)
    igraph_ = []; ave = G_aves[fd]

    for G in node_:  # init per node
        uprim = [link for link in G.rim if len(link.dderH.H)==len(G.extH.H)]
        if uprim:  # skip nodes without add new added rim
            grapht = [[G],[],[*G.Et], uprim]  # link_ = updated rim
            G.root = grapht  # for G merge
            igraph_ += [grapht]
        else:
            G.root = None
    _graph_ = copy(igraph_)

    while True:
        graph_ = []
        for grapht in _graph_:  # extend grapht Rim with +ve in-root links
            if grapht not in igraph_: continue  # skip merged graphs
            G_, Link_, Et, Rim = grapht
            inVal, inRdn = 0,0  # new in-graph +ve
            new_Rim = []
            for link in Rim:  # unique links
                if link.node in G_:  # one of the nodes is already clustered
                    G = link.node; _G = link._node
                else:
                    G = link._node; _G = link.node
                if _G in G_: continue
                # connect by rel match of nodes * match of node Vs: surround M|Ds,
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

def convolve_graph(node_, link_):  # revalue nodes and links by the value of their increasingly wide neighborhood:
    '''
    or inclusion adjustment by similarity to kernel, vs. rim similarity in all kernel elements:
    Reduce proximity bias by kernel centroid quasi-clustering: inclusion val adjustment per element.

    Sum connectivity per node|link from their links|nodes, extended in feedforward through the network, bottom-up.
    Then backprop adjusts node|link connect value by relative value of its higher-layer neighborhood, top-down.
    Math: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/node_connect.png
    '''
    rim_effect = 0.5  # impact of neighborhood inclusion on node|link inclusion

    for fd, e_ in zip((0,1), (node_,link_)):
        ave = G_aves[fd]
        iterations = 0
        # ff,fb through all layers, break if abs(_hV-hV) < ave or hV < ave:
        while True:
            kernels, lV, lR = [], 0, 0  # form/reform base layer of kernels per node|link:
            for e in e_:  # root node_|link_
                V=e.Et[fd]; R=e.Et[2+fd]
                kernels += [[[e], V, R]]  # init kernel / node|link, no neighborhood yet
                lV += V; lR += R
            layers = [[kernels, lV, lR]]  # init convo layers
            _hV = lV; _hR = lR

            while True:  # bottom-up feedforward, break if kernel == root E_
                # add higher layer of larger kernels: += adjacent nodes or dlinks:
                Kernels, lV,lR = [],0,0
                for e_,v,r in kernels:  # v,r for feedback only, overlap between kernels
                    E_,V,R = e_,v,r  # init with lower kernel
                    for e in e_:
                        if fd:  # += dlinks in link._node.rim:
                            for G in e._node, e.node:
                                for link in G.rim:
                                    if link not in E_ and link.Et[1] > ave * link.Et[3]:
                                        E_+=[link]; V+=link.Et[1]; R+=link.Et[3]
                        else:  # += linked nodes
                            V,R = e.Et[0],e.Et[2]
                            for link in e.rim:
                                node = link._node if link.node is e else link.node
                                if node not in E_ and node.Et[0] > ave * node.Et[2]:
                                    E_+=[node]; V+=node.Et[0]; R+=link.Et[3]
                    Kernels += [[E_,V,R]]; lV+=V; lR+=R
                layers += [[Kernels,lV,lR]]; hV=lV; hR=lR
                if len(Kernels[0]) == 1:
                    break  # stop if one Kernel covers the whole root node_|link_
                else:
                    kernels = Kernels
            # backprop per layer of Kernels to their sub-kernels in lower layer:
            elev = len(layers)
            while layers:
                Kernels,_,_,_ = layers.pop()  # unpack top-down
                elev -= 1  # elevation of the lower layer, to be adjusted
                for Kernel,V,R in Kernels:
                    rV = V / (ave * len(Kernel[0]))  # relative inclusion value, no use for R?
                    for i, (kernel,v,r) in enumerate(Kernel):
                        # adjust element val by wider Kernel relative val, rdn is not affected?
                        if elev:  # adjust lower kernel V:
                            Kernel[0][i][1] = v * (rV * rim_effect)  # this is wrong, should be lower-layer kernel that contains Kernel[0][i]
                        else:  # bottom layer, adjust node|link V:
                            Kernel[0][i].Et[fd] = v * (rV * rim_effect)

            iterations += 1
            if abs(_hV - hV) < ave or hV < ave*hR:  # low adjustment or net value?
                break
            else: _hV=hV; _hR=hR
