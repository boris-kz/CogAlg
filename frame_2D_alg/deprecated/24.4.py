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
    Sum connectivity per node|link from their links|nodes, extended in feedforward through the network, bottom-up.
    Then backprop adjusts node|link connect value by relative value of its higher-layer neighborhood, top-down.
    Math: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/node_connect.png
    Reduce proximity bias in inclusion value by kernel centroid quasi-clustering: cross-similarity among kernels
    '''
    rim_coef = 0.5  # impact of neighborhood inclusion on node|link inclusion

    for fd, e_ in zip((0,1), (node_,link_)):
        ave = G_aves[fd]
        Ce = [CG,Clink][fd]
        iterations = 0
        # ff,fb through all layers, break if abs(_hV-hV) < ave or hV < ave:
        while True:
            # bottom-up feedforward, 1st-layer kernels are [node|link + rim]s:
            kernels, lV,lR = [],0,0
            for e in e_:  # root node_|link_
                E_ = [e]; Et = copy(e.Et); n = 1  # E_ = e + rim/neighborhood:
                if fd: # += dlinks in link._node.rim:
                    for G in e._node, e.node:
                        for link in G.rim:
                            if link is not e and link.Et[1] > ave * link.Et[3]:
                                E_+=[link]; np.add(Et,link.Et); n += 1
                else:  # += linked nodes
                    for link in e.rim:
                        node = link._node if link.node is e else link.node
                        if node.Et[0] > ave * node.Et[2]:
                            E_+=[node]; Et = [V+v for V,v in zip(Et, link.Et)]; n += 1
                kernel = Ce(node_=E_,Et=Et,n=n); e.root = kernel  # replace in sum2graph
                kernels += [kernel]
                lV+=Et[fd]; lR+=Et[2+fd]
            layers = [[kernels,lV,lR]]  # init convo layers
            _hV=lV; _hR=lR
            while True:  # add higher layer Kernels: node_= new center + extended rim, break if kernel == root E_: no higher kernels
                Kernels, lV,lR = [],0,0
                for kernel in kernels:  # CG | Clink
                    Kernel = Ce(node_=[kernel],n=1); kernel.root=Kernel  # init with each lower kernel (central), add new rim from current rim roots:
                    '''
                    next layer wider Kernels: get root _Kernel of each _kernel in current rim, add _Kernel rim __kernels if not in current rim. 
                    Those rim _Kernels are a bridge between current rim and extended rim, they include both:
                    '''
                    for e in kernel.node_[1:]:  # current rim
                        _kernel = e.root
                        for _e in _kernel.node_[1:]:
                            __kernel = _e.root
                            if __kernel not in Kernel.node_ and __kernel not in kernel.node_:  # not in current rim, add to new rim:
                                Kernel.node_ += [__kernel]; Kernel.Et=[V+v for V,v in zip(Kernel.Et, __kernel.Et)]
                                if Kernel.derH: Kernel.derH.add_(__kernel.derH)
                                else:           Kernel.derH.append(__kernel.derH, flat=1)
                                Kernel.n += __kernel.n
                    Kernels += [Kernel]; lV+=Kernel.Et[fd]; lR+=Kernel.Et[2+fd]
                layers += [[Kernels,lV,lR]]; hV=lV; hR=lR
                if Kernels[0].n == len([node_,link_][fd]):
                    break  # each Kernel covers the whole root node_|link_
                else:
                    kernels = Kernels
            # backprop per layer of centroid Kernels to their sub-kernels in lower layer, draft:
            while layers:
                if len(layers) == 1: break  # skip if it's the bottom layer (nothing to compare?, their Et will be adjusted below, when len(layers) == 1)
                Kernels,_,_ = layers.pop()  # unpack top-down
                for Kernel in Kernels:
                    for kernel in Kernel.node_:
                        DderH = comp_kernel(Kernel, kernel, fd)
                        rV = DderH.Et[fd] / (ave * DderH.n) * rim_coef
                        kernel.Et[fd] *= rV  # adjust element inclusion value by relative value of Kernel, rdn is not affected?
                        if len(layers) == 1:  # bottom layer
                            for e in kernel.node_:  # adjust base node|link V:
                                dderH = comp_kernel(kernel, e, fd)
                                rv = dderH.Et[fd] / (ave * dderH.n) * rim_coef
                                e.Et[fd] *= rv
            iterations += 1
            if abs(_hV - hV) < ave or hV < ave*hR:  # low adjustment or net value?
                break
            else:
                _hV=hV; _hR=hR  # hR is not used?

def comp_kernel(_kernel, kernel, fd):

    if fd:
        dderH = CH()
        _kernel.comp_link(kernel, dderH)
    else:
        cy, cx = box2center(_kernel.box); _cy, _cx = box2center(kernel.box); dy = cy - _cy; dx = cx - _cx
        dist = np.hypot(dy, dx)  # distance between node centers
        dderH = comp_G([_kernel,kernel, dist, [dy,dx]],iEt=[0,0,0,0], fkernel=1)

    return dderH


def rng_recursion(rroot, root, node_, links, Et, nrng=1):  # rng++/G_, der+/link_ in sub+, -> rim_H

    for link in links:
        if isinstance(link.list):
            fd = 0; _G, G = link  # prelink in recursive rng+
            if isinstance(G,CG):
                cy, cx = box2center(G.box); _cy, _cx = box2center(_G.box)
                dy = cy-_cy; dx = cx-_cx
            else:
                (_y1,_x1),(_y2,_x2) = box2center(_G.node_[0].box), box2center(_G.node_[1].box)
                (y1,x1),(y2,x2)     = box2center(G.node_[0].box), box2center(G.node_[1].box)
                dy = (y1+y2)/2 -(_y1+_y2)/2; dx = (x1+x2)/2 -(_x1+_x2)/2
            dist = np.hypot(dy, dx)  # distance between node centers or link centers
        else:
            fd = 0; (_G, G), dist, (dy,dx) = link.node_, link.distance, link.angle  # Clink in 1st call from sub+
        if _G in G.compared_: continue
        # der+'rng+ is directional
        if nrng > 1:  # pair eval:
            M = (G.Et[0]+_G.Et[0])/2; R = (G.Et[2]+_G.Et[2])/2  # local
            for link in _G.rim:
                if comp_angle((dy,dx), link.angle)[0] > ave_mA:
                    for med_link in link.rim:  # if link is hyperlink
                        M += med_link.derH.H[-1].Et[0]
                        R += med_link.derH.H[-1].Et[2]
        if nrng==1:
            if dist<=ave_dist:
                G.compared_ += [_G]; _G.compared_ += [G]
                comp_G(link, Et) if fd else comp_G([_G,G, dist, [dy,dx]], Et)  # Clink in 1st call from sub+
        elif M / (dist/ave_dist) > ave * R:
           G.compared_ += [_G]; _G.compared_ += [G]
           comp_G([_G,G, dist, [dy,dx]], Et)
    if Et[0] > ave_Gm * Et[2]:  # rng+ eval per arg cluster because comp is bilateral, 2nd test per new pair
        nrng,_ = rng_recursion(rroot, root, node_, list(combinations(node_,r=2)), Et, nrng+1)

    return nrng, Et
