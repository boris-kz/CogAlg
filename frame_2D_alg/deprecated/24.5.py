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

    iP_  = PP.P_
    rng = 0  # cost of links added per rng+
    while True:
        rng += 1
        P_ = []; V = 0
        for P in iP_:
            if P.link_:
                if len(P.link)<rng: continue  # no current-rng link_
            else: continue  # top P_ in PP?
            _prelink_ = P.link_.pop()
            rng_link_, prelink_ = [],[]  # both per rng+
            for link in _prelink_:
                if link.distance <= rng:  # | rng * ((P.val+_P.val)/ ave_rval)?
                    _P = link.node_[0]
                    if fd and not (P.derH and _P.derH): continue  # nothing to compare
                    mlink = comp_P(link, fd)
                    if mlink: # return if match
                        V += mlink.derH.Et[0]
                        rng_link_ += [mlink]
                        prelink_ += _P.link_[-1]  # connected __Ps links (_P.link_[-1] is prelinks)
            if rng_link_:
                P.link_ += [rng_link_]
                if prelink_: P.link_ += [rng_link_]  # temporary pre-links
            if prelink_:
                P_ += [P]
        if V > ave * rng * len(P_) * 6:  #  implied val of all __P_s, 6: len mtuple
            iP_ = P_; fd = 0
        else:
            for P in PP.P_: P.link_.pop()  # remove prelinks
            break
    # der++ in PPds from rng++, no der++ inside rng++: high diff @ rng++ termination only?
    PP.rng=rng  # represents rrdn

    return P_

def unpack_last_link_(link_):  # unpack last link layer

    while link_ and isinstance(link_[-1], list): link_ = link_[-1]
    return link_
    # change to use nested link_, or
    # get higher-P mediated link_s while recursion_count < rng?

'''
parallelize by forward/feedback message-passing between two layers: roots with node_ and nodes with root_: 
while > ave dOV: compute link_ oV for each (node,root); then assign each node to max root

feedback refines link overlap: connected subset per node in higher nodes / clusters, 
different from fitting to the whole higher node in conventional backprop, as in GNN 

rng+/seg+: direct comp nodes (vs summed in krim) in proportion to graph size * M: likely distant match,
comp individual nodes in node-mediated krims, replacing krims and ExtH layers? 
adding links and overlap for segmentation.
'''
def segment_parallel(root, Q, fd, nrng):  # recursive eval node_|link_ rims for cluster assignment
    '''
    kernels = get_max_kernels(Q)  # for selective link tracing?
    grapht_ = select_merge(kernels)  # refine by match to max, or only use it to selectively merge?
    '''
    node_,root_ = [],[]
    for i, N in enumerate(Q):
        N.root = i  # index in root_,node_
        rim = get_rim(N, fd)
        _N_ = [link.node_[0] if link.node_[1] is N else link.node_[1] for link in rim]
        root_ += [[[],[N], [[0,0,0,0]]]]  # init link_=N.rim, node_=[N], oEt_
        node_ += [[N, rim,_N_, [i], []]]  # root_ indices, empty nEt_
    r = 0  # recursion count
    _OEt = [0,0,0,0]
    while True:
        OEt = [0,0,0,0]
        for N,rim,_N_, nroot_,_oEt_ in node_:  # update node roots
            olink_,oEt_ = [],[]  # recompute
            for link_, N_, nEt_ in root_:
                oEt = [0,0,0,0]
                for L,_N in zip(rim,_N_):
                    if _N in N_:  # both in nroot and in N.rim
                        oEt = np.add(oEt, L.Et)
                OEt = np.add(OEt,oEt)
                _oEt[:] = oEt
                if oEt[fd] > ave * oEt[2+fd]:  # N in root
                    if N not in N_:
                        N_ += [N]; nEt_ += [oEt]; link_[:] = list(set(link_).union(rim))  # not directional
                elif N in N_:
                    nEt_.pop(N_.index(N)); N_.remove(N); link_[:] = list(set(link_).difference(rim))
                olink_ = []
            _oEt_[:] = oEt_
        r += 1
        if OEt[fd] - _OEt[fd] < ave:  # low total overlap update
            break
        _OEt = OEt
    # old:
    while True:
        OEt = [0,0,0,0]
        for N,rim, root_,_oEt_ in node_:  # update node roots, inclusion vals
            oEt_ = []
            for link_,N_,_roEt_ in root_:  # update root links, nodes
                olink_ = []
                oEt = [0,0,0,0]
                for olink in olink_: oEt = np.add(oEt, olink.Et)
                OEt = np.add(OEt,oEt); oEt_+=[oEt]
                if oEt[fd] > ave * oEt[2+fd]:  # N in root
                    if N not in N_:
                        N_ += [N]; _roEt_ += [oEt]; link_[:] = list(set(link_).union(rim))  # not directional
                elif N in N_:
                    _roEt_.pop(N_.index(N)); N_.remove(N); link_[:] = list(set(link_).difference(rim))
            _oEt_[:] = oEt_
        r += 1
        if OEt[fd] - _OEt[fd] < ave:  # low total overlap update
            break
        _OEt = OEt

    for N,rim, Nroot_,oEt_ in node_:  # exclusive N assignment to clusters based on final oV

        Nroot_ = [root for root in Nroot_ if N in root[1]]
        if Nroot_:  # include isolated N?
            Nroot_ = sorted(Nroot_, key=lambda root: root[2][root[1].index(N)][fd])  # sort by NoV in roots
            N.root = Nroot_.pop()  # max root
            for root in Nroot_:  # remove N from other roots
                root[0][:] = list(set(root[0]).difference(rim))  # remove rim
                i = root[1].index(N); root[1].pop(i); root[2].pop(i)  # remove N and NoV

    return [sum2graph(root, grapht, fd, nrng) for grapht in root_ if grapht[1]]  # not-empty clusters

def get_rim(N, fd):
    if fd:  # N is Clink
        if isinstance(N.node_[0],CG):
            rim = [G.rim for G in N.node_]
        else:  # get link node rim_t dirs opposite from each other, else covered by the other link rim_t[1-dir]?
            rim = [(link.rim_t[1-dir][-1] if link.rim_t[dir] else []) for dir, link in zip((0,1), N.node_)]
    else:   rim = N.rim

    return rim
