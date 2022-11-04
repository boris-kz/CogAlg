def comp_G_(G_, fder):  # cross-comp Gs (patterns of patterns): Gs, derGs, or segs inside PP

    for i, _G in enumerate(G_):

        for G in G_[i+1:]:  # compare each G to other Gs in rng, bilateral link assign
            if G in [node for link in _G.link_ for node in link.node_]:  # G,_G was compared in prior rng+, add frng to skip?
                continue
            # comp external params:
            _x = (_G.xn +_G.x0)/2; _y = (_G.yn +_G.y0)/2; x = (G.xn + G.x0)/2; y = (G.yn + G.y0)/2
            dx = _x - x; dy = _y - y
            distance = np.hypot(dy, dx)  # Euclidean distance between centroids, sum in G.sparsity
            proximity = ave_rng-distance  # coord match

            mang, dang = comp_angle(_G.angle, G.angle)  # dy,dx for derG or high-aspect Gs, both *= aspect?
            # n orders of len and sparsity, -= fill: sparsity inside nodes?
            if fder:
                mlen,dlen = 0,0; _sparsity = _G.sparsity; sparsity = G.sparsity  # single-link derGs
            else:
                _L = len(_G.node_); L = len(G.node_)
                dlen = _L - L; mlen = min(_L, L)
                if isinstance(G.node_[0][0], CP): _sparsity, sparsity = 1, 1
                else:
                    _sparsity = sum(node.sparsity for node in _G.node_) / _L
                    sparsity = sum(node.sparsity for node in G.node_) / L
            dspar = _sparsity-sparsity; mspar = min(_sparsity,sparsity)
            # draft:
            mext = [proximity, mang, mlen, mspar]; mVal = proximity + mang + mlen + mspar
            dext = [distance, dang, dlen, dspar];  dVal = distance + dang + dlen + dspar
            derext = [mext,dext,mVal,dVal]

            if mVal > ave_ext * ((sum(_G.valt)+sum(G.valt)) / (2*sum(G_aves))):  # max depends on combined G value
                mplevel, dplevel = comp_plevels(_G.plevels, G.plevels, _G.fds, G.fds, derext)
                valt = [mplevel[1] - ave_Gm, dplevel[1] - ave_Gd]  # norm valt: *= link rdn?
                derG = Cgraph(  # or mean x0=_x+dx/2, y0=_y+dy/2:
                    plevels=[mplevel,dplevel], y0=min(G.y0,_G.y0), yn=max(G.yn,_G.yn), x0=min(G.x0,_G.x0), xn=max(G.xn,_G.xn),
                    sparsity=distance, angle=[dy,dx], valt=valt, node_=[_G,G])
                _G.link_ += [derG]; G.link_ += [derG]  # any val
                for fd in 0,1:
                    if valt[fd] > 0:  # alt fork is redundant, no support?
                        for node, (graph, meds_, gvalt) in zip([_G, G], [G.roott[fd], _G.roott[fd]]):  # bilateral inclusion
                            if node not in graph:
                                for mderG in node.link_:
                                    mnode = mderG.node_[0] if mderG.node_[1] is node else mderG.node_[1]
                                    if mderG not in meds_:  # combined meds per node
                                        meds_ += [[derG.node_[0] if derG.node_[1] is node else derG.node_[1] for derG in node.link_]]
                                gvalt[0] += node.valt[0]; gvalt[1] += node.valt[1]
                                graph += [node, meds_, valt]  # meds per node


def eval_med_layer(graph_, graph, fd):   # recursive eval of reciprocal links from increasingly mediated nodes

    node_, meds_, valt = graph
    save_node_, save_meds_ = [], []
    adj_Val = 0  # adjust connect val in graph

    for G, med_node_ in zip(node_, meds_):  # G: node or sub-graph
        mmed_node_ = []  # __Gs that mediate between Gs and _Gs
        for _G in med_node_:
            for derG in _G.link_:
                if derG not in G.link_:  # link_ includes all unique evaluated mediated links, flat or in layers?
                    # med_PP.link_:
                    med_link_ = derG.node_[0].link_ if derG.node_[0] is not _G else derG.node_[1].link_
                    for _derG in med_link_:
                        if G in _derG.node_ and _derG not in G.link_:  # __G mediates between _G and G
                            G.link_ += [_derG]
                            adj_val = _derG.valt[fd] - ave_agg  # or increase ave per mediation depth
                            # adjust nodes:
                            G.valt[fd] += adj_val; _G.valt[fd] += adj_val  # valts not updated
                            valt[fd] += adj_val; _G.roott[fd][2][fd] += adj_val  # root is not graph yet
                            __G = _derG.node_[0] if _derG.node_[0] is not _G else _derG.node_[1]
                            if __G not in mmed_node_:  # not saved via prior _G
                                mmed_node_ += [__G]
                                adj_Val += adj_val
        if G.valt[fd]>0:
            # G remains in graph
            save_node_ += [G]; save_meds_ += [mmed_node_]  # mmed_node_ may be empty

    for G, mmed_ in zip(save_node_, save_meds_):  # eval graph merge after adjusting graph by mediating node layer
        add_mmed_= []
        for _G in mmed_:
            _graph = _G.roott[fd]
            if _graph in graph_ and _graph is not graph:  # was not removed or merged via prior _G
                _node_, _meds_, _valt = _graph
                for _node, _meds in zip(_node_, _meds_):  # merge graphs, ignore _med_? add direct links:
                    for derG in _node.link_:
                        __G = derG.node_[0] if derG.node_[0] is not _G else derG.node_[1]
                        if __G not in add_mmed_ + mmed_:  # not saved via prior _G
                            add_mmed_ += [__G]
                            adj_Val += derG.valt[fd] - ave_agg
                valt[fd] += _valt[fd]
                graph_.remove(_graph)
        mmed_ += add_mmed_

    graph[:] = [save_node_,save_meds_,valt]
    if adj_Val > ave_med:  # positive adj_Val from eval mmed_
        eval_med_layer(graph_, graph, fd)  # eval next med layer in reformed graph
