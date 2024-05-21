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
def segment_recursive(root, Q, fd, nrng):  # recursive eval node_|link_ rims for cluster assignment
    '''
    kernels = get_max_kernels(Q)  # parallelization of link tracing, not urgent
    grapht_ = select_merge(kernels)  # refine by match to max, or only use it to selectively merge?
    '''
    _node_ = G_ = copy(Q)
    r = 0
    grapht_ = []
    while True:
        node_ = []
        for node in _node_:  # depth-first eval merge node_|link_ connected via their rims:
            if node not in G_: continue  # merged?
            if not node.root:  # init in 1st loop or empty root after root removal
                grapht = [[node],[],[0,0,0,0]]  # G_, Link_, Et
                grapht_ += [grapht]
                node.root = grapht
            upV, remaining_node_ = merge_node(grapht_, G_, node, fd, upV=0)  # G_ to remove merged node
            if upV > ave:  # graph update value, accumulate?
                node_ += [node]
            G_ += remaining_node_; node_ += remaining_node_    # re-add for next while loop
        if node_:
            _node_ = G_ = copy(node_)
        else:
            break  # no updates
        r += 1  # recursion count
    graph_ = [sum2graph(root, grapht[:3], fd, nrng) for grapht in grapht_ if  grapht[2][fd] > ave * grapht[2][2+fd]]

    return graph_

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
forward/feedback message-passing between two layers: roots with node_ and nodes with root_: 
while > ave dOV: compute link_ oV for each (node,root); then assign each node to max root
root feedback refines node x root link overlap, vs fitting to higher node in GNN backprop 

der+: correlation clustering of links through their nodes, forming dual trees of matching links in rng+?

add rng+/seg+: direct comp nodes (vs summed in krim) in proportion to graph size * M: likely distant match,
comp individual nodes in node-mediated krims, replacing krims and ExtH layers, 
adding links overlap for segmentation.
'''
def segment_parallel(root, Q, fd, nrng):  # recursive eval node_|link_ rims for cluster assignment
    '''
    kernels = get_max_kernels(Q)  # for selective link tracing?
    grapht_ = select_merge(kernels)  # refine by match to max, or only use it to selectively merge?
    '''
    node_,root_ = [],[]
    for i, N in enumerate(Q):
        N.root = i  # index in root_,node_
        rim = get_rim(N)
        _N_ = [link.node_[0] if link.node_[1] is N else link.node_[1] for link in rim]
        root_ += [[[],[N], [[0,0,0,0]]]]  # init link_=N.rim, node_=[N], oEt_
        node_ += [[N, rim,_N_, [i], []]]  # root_ indices, empty nEt_
    r = 0  # recursion count
    _OEt = [0,0,0,0]
    while True:
        OEt = [0,0,0,0]
        for N,rim,_N_,nroot_,_oEt_ in node_:  # update node roots
            oEt_ = []
            for i,(link_,N_,nEt_) in enumerate(root_):
                olink_,oEt = [],[0,0,0,0]
                for L,_N in zip(rim,_N_):  # get overlap between N rim and root link_
                    if _N in N_:  # in both N rim and root link_
                        oEt = np.add(oEt, L.Et)
                        olink_ += [L]
                if olink_:
                    OEt = np.add(OEt,oEt)
                    oEt_ += [oEt]
                # feedback to roots: add/remove N from rN_, adjusting next-loop overlap:
                if oEt[fd] > ave * oEt[2+fd]:  # N in root
                    if N not in N_:
                        nroot_ += [i]; N_ += [N]; nEt_ += [oEt]; link_[:] = list(set(link_).union(olink_))  # not directional
                elif N in N_ and len(N_)>1:
                    Ni = N_.index(N); nroot_.pop(Ni); nEt_.pop(Ni); N_.remove(N); link_[:] = list(set(link_).difference(olink_))
            _oEt_[:] = oEt_
        r += 1
        if OEt[fd] - _OEt[fd] < ave:  # low total overlap update
            break
        _OEt = OEt
    # exclusive N assignment to clusters based on final oV:
    for N,rim, _N_, nroot_,oEt_ in node_:
         if len(nroot_) == 1:
             continue  # keep isolated Ns
         nroot_ = [root_[i] for i in nroot_]
         nroot_ = sorted(nroot_, key=lambda root: root[2][root[1].index(N)][fd])  # sort by NoV in roots
         N.root = nroot_.pop()  # max root
         for root in nroot_:  # remove N from other roots
             root[0][:] = list(set(root[0]).difference(rim))  # remove rim
             i = root[1].index(N); root[1].pop(i); root[2].pop(i)  # remove N and NoV
    '''
    This pruning should actually be recursive too, it reduces overlap. May as well merge graphs via floodfill.
    Which is sequential, the only parallelization is reusing merged-graph processor for some other root graph. 
    Or subdividing processing of large graphs, but that's a lot of overhead.
    '''
    return [sum2graph(root, grapht, fd, nrng) for grapht in root_ if grapht[1]]  # not-empty clusters

def get_rim(N, fd):
    if fd:  # N is Clink
        if isinstance(N.node_[0],CG):
            rim = [G.rim for G in N.node_]
        else:  # get link node rim_t dirs opposite from each other, else covered by the other link rim_t[1-dir]?
            rim = [(link.rim_t[1-dir][-1] if link.rim_t[dir] else []) for dir, link in zip((0,1), N.node_)]
    else:   rim = N.rim

    return rim

def get_max_kernels(G_):  # use local-max kernels to init sub-graphs for segmentation

    kernel_ = []
    for G in copy(G_):
        _G_ = []; fmax = 1
        for link in G.rim:
            _G = link.node_[0] if link.node_[1] is G else link.node_[1]
            if _G.DerH.Et[0] > G.DerH.Et[0]:  # kernel-specific
                fmax = 0
                break
            _G_ += [_G]
        if fmax:
            kernel = [G]+_G_  # immediate kernel
            for k in kernel:
                k.root += [kernel]  # node in overlapped area may get more than 1 kernel
                if k in G_: G_.remove(k)
            kernel_ += [kernel]
    for G in G_:  # remaining Gs are not in any kernels, append to the nearest kernel
        _G_ = [link.node_[0] if link.node_[1] is G else link.node_[1] for link in G.rim]  # directly connected Gs already checked
        __G_ = []
        while _G_:
            while True:
                for _G in _G_:
                    for link in _G.rim:
                        __G = link.node_[0] if link.node_[1] is _G else link.node_[1]  # indirectly connected Gs
                        if __G not in G_:  # in some kernel, append G to it:
                            G.root = __G.root
                            __G.root[-1] += [G]
                            break
                        __G_ += [__G]
                _G_ = __G_
    # each kernel may still have overlapped nodes
    return kernel_

def rim_link_Et_x_olp(iQ, fd):
    # recursive link.Et += link.relt * (node_ rim overlap - link)
    _OEt = [0,0,0,0]
    _oEt_ = [[0,0,0,0] for n in iQ]
    _Q = copy(iQ)
    r = 1
    while True:
        OEt = [0,0,0,0]
        oEt_ = []
        DOV = 0
        Q = []
        for N, _oEt in zip(_Q, _oEt_):  # update node rim Ets
            oEt = [0,0,0,0]  # not sure
            DoV = 0
            rim = get_rim(N)
            for link in rim:
                _N = link.node_[0] if link.node_[1] is N else link.node_[1]
                _rim = get_rim(_N)
                olink_ = list(set(rim).intersection(set(_rim)))
                [np.add(oEt, L.Et) for L in olink_ if L is not link]
                V = link.Et[fd]
                doV = oEt[fd] - _oEt[fd]  # oV update to adjust V, not sure we need the rest of oEts:
                link.Et[fd] = V + link.relt[fd] * doV
                DoV += doV

            DOV += DoV; np.add(OEt, oEt); _oEt = oEt
            if DoV > ave:
                Q += [N]; oEt_ += [oEt]

        if DOV < ave: break  # low update value
        _Q = Q; _oEt_ = oEt_; _OEt = OEt
        r += 1

def merge_node(grapht_, iG_, G, fd, upV):

    if G in iG_: iG_.remove(G)
    G_, Link_, Et = G.root
    ave = G_aves[fd]
    remaining_node_ = []  # from grapht removed from grapht_
    fagg = isinstance(G,CG)

    for link in (G.rim if fagg else G.rim_t[0][-1] if G.rim_t[0] else [] + G.rim_t[1][-1] if G.rim_t[1] else []):  # be empty
        if link.Et[fd] > ave:  # fork eval
            if link not in Link_: Link_ += [link]
            _G = link.node_[0] if link.node_[1] is G else link.node_[1]
            if _G in G_:
                continue  # _G is already in graph
            # get link overlap between graph and node.rim:
            olink_ = list(set(Link_).intersection(_G.rim))
            oV = sum([olink.Et[fd] for olink in olink_])  # link overlap V
            if oV > ave and oV > _G.Et[fd]:  # higher inclusion value in new vs. current root
                upV += oV
                _G.Et[fd] = oV  # graph-specific, rdn?
                G_ += [_G]; Link_ += [link]; Et = np.add(Et,_G.Et)
                if _G.root:  # remove _G from root
                    _G_, _Link_, _Et = _G.root
                    _G_.remove(_G)
                    if link in _Link_: _Link_.remove(link)  # empty at grapht init in r==0
                    _Et = np.subtract(_Et, Et)
                    if _Et[fd] < ave*_Et[2+fd]:
                        grapht_.remove(_G.root)
                        for __G in _G_: __G.root = []   # reset root
                        remaining_node_ += _G_

                _G.root = G.root  # temporary
                upV, up_remaining_node_ = merge_node(grapht_, iG_, _G, fd, upV)
                remaining_node_ += up_remaining_node_

    return upV, [rnode for rnode in remaining_node_ if not rnode.root]  # skip node if they added to grapht during the subsequent merge_node process

def select_merge(kernel_):

    for kernel in copy(kernel_):
        for node in copy(kernel):
            for _kernel in copy(node.root):  # get overlapped _kernel of kernel
                if _kernel is not kernel and _kernel in kernel_:  # not a same kernel and not a merged kernel
                    for link in node.rim:  # get link between 2 centers
                        if kernel[0] in link.node_ and _kernel[0] in link.node_:
                            break
                    if link.ExtH.Et[0] > ave:  # eval by center's link's ExtH?
                        for _node in _kernel:  # merge _kernel into kernel
                            if _node not in kernel:
                                kernel += [_node]
                                _node.root = kernel
                        kernel_.remove(_kernel)  # remove merged _kernel
            node.root = kernel  # remove list kernel
    grapht_ = []
    for kernel in kernel_:
        Et =  [sum(node.Et[0] for node in kernel), sum(node.Et[1] for node in kernel)]
        rim = list(set([link for node in kernel for link in node.rim]))
        grapht = [kernel, [], Et, rim]  # not sure
        grapht_ += [grapht]

    return grapht_

def comp_P(link, fd):
    aveP = P_aves[fd]
    _P, P, distance, angle = link.node_[0], link.node_[1], link.distance, link.angle

    if fd:  # der+, comp derPs, no comp_latuple?
        rn = (_P.derH.n if P.derH else len(_P.dert_)) / P.derH.n
        derH = _P.derH.comp_(P.derH, CH(), rn=rn, flat=0)
        vm,vd,rm,rd = derH.Et
        rm += vd > vm; rd += vm >= vd
        He = link.derH  # append link derH:
        if He and not isinstance(He.H[0], CH): He = link.derH = CH(Et=[*He.Et], H=[He])  # nest md_ as derH
        He.Et = np.add(He.Et, (vm,vd,rm,rd))
        He.H += [derH]
    else:  # rng+, comp Ps
        rn = len(_P.dert_) / len(P.dert_)
        H = comp_latuple(_P.latuple, P.latuple, rn)
        vm = sum(H[::2]); vd = sum(abs(d) for d in H[1::2])
        rm = 1 + vd > vm; rd = 1 + vm >= vd
        n = (len(_P.dert_)+len(P.dert_)) / 2  # der value = ave compared n?
        _y,_x = _P.yx; y,x = P.yx
        angle = np.subtract([y,x], [_y,_x]) # dy,dx between node centers
        distance = np.hypot(*angle)      # distance between node centers
        link = Clink(node_=[_P, P], derH = CH(Et=[vm,vd,rm,rd],H=H,n=n),angle=angle,distance=distance)

    for cP in _P, P:  # to sum in PP
        link.latuple = [P+p for P,p in zip(link.latuple[:-1],cP.latuple[:-1])] + [[A+a for A,a in zip(link.latuple[-1],cP.latuple[-1])]]
        link.yx_ += cP.yx_
    if vm > aveP * rm:  # always rng+?
        return link
