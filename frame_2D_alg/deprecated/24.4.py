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


def rng_recursion(root, Et, fagg):  # comp Gs in agg+, links in sub+
    nrng = 1

    if fagg:  # distance eval, else fd: mangle eval
        _links = list(combinations(root.node_,r=2))
        while True:
            for link in _links:  # prelink in agg+
                _G, G = link
                if _G in G.compared_: continue
                cy, cx = box2center(G.box); _cy, _cx = box2center(_G.box)
                dy = cy-_cy; dx = cx-_cx;  dist = np.hypot(dy,dx)
                if nrng==1:  # eval distance between node centers
                    fcomp = dist <= ave_dist
                else:
                    M = (G.Et[0]+_G.Et[0])/2; R = (G.Et[2]+_G.Et[2])/2  # local
                    fcomp = M / (dist/ave_dist) > ave * R
                if fcomp:
                    G.compared_ += [_G]; _G.compared_ += [G]
                    Link = Clink(node_=[_G, G], distance=dist, angle=[dy, dx], box=extend_box(G.box, _G.box))
                    comp_G(Link, Et, fd=0)
            # reuse combinations
            if Et[0] > ave_Gm * Et[2] * nrng: nrng += 1
            else: break
    else:
        _links = root.link_  # der+'rng+: directional and node-mediated comp link
        while True:
            links = []
            for link in _links:
                for G in link.node_:  # search in both directions via Clink Gs
                    for _link in G.rim:
                        if _link in link.compared_: continue
                        (_y,_x),(y,x) = box2center(_link.box),box2center(link.box)
                        dy=_y-y; dx=_x-x; dist = np.hypot(dy,dx)  # distance between node centers
                        mA,dA = comp_angle((dy,dx), _link.angle)  # node-mediated, distance eval in agg+ only
                        if mA > ave_mA:
                            _link.compared_ += [link]; link.compared_ += [_link]
                            _derH = _link.derH; et = _derH.Et
                            Link = Clink(node_=[_link,link],distance=dist,angle=(dy,dx),box=extend_box(link.box,_link.box),
                                         derH=CH(H=deepcopy(_derH.H), Et=[et[0]+mA, et[1]+dA, et[2]+mA<dA, et[3]+dA<=mA]))
                            comp_G(Link, Et, fd=1)  # + comp med_links
                            if Link.derH.Et[0] > ave_Gm * Link.derH.Et[2] * nrng:
                                links += [Link]

            if Et[0] > ave_Gm * Et[2] * nrng:  # rng+ eval per arg cluster because comp is bilateral, 2nd test per new pair
                nrng += 1; _links = links
            else: break

    return nrng, Et


def comp_G(link, iEt, nrng=None):  # add dderH to link and link to the rims of comparands, which may be Gs or links

    dderH = CH()  # new layer of link.dderH
    if isinstance(link, Clink):  # always true now?
        # der+
        _G,G = link.node_; fd=1
    else:  # rng+
        _G,G, distance, [dy,dx] = link; fd=0
        link = Clink(node_=[_G, G], distance=distance, angle=[dy, dx])
    if isinstance(G,CG):
        fG = 1; rn = _G.n/G.n
    else:
        fG = 0; rn = min(_G.node_[0].n,_G.node_[1].n) / min(G.node_[0].n,G.node_[1].n)
    if not fd:  # form new Clink
        if fG:  # / P
            Et, relt, md_ = comp_latuple(_G.latuple, G.latuple, rn, fagg=1)
            dderH.n = 1; dderH.Et = Et; dderH.relt=relt
            dderH.H = [CH(nest=0, Et=copy(Et), relt=copy(relt), H=md_, n=1)]
            # / PP, if >1 Ps:
            if _G.iderH and G.iderH: _G.iderH.comp_(G.iderH, dderH, rn, fagg=1, flat=0)
            _L,L = len(_G.node_),len(G.node_); _S,S = _G.S, G.S/rn; _A,A = _G.A,G.A
            dderH.nest = 1  # packing md_
        else: _L,L = _G.distance,G.distance; _S,S = len(_G.rim),len(G.rim); _A,A = _G.angle,G.angle
        comp_ext(_L,L,_S,S,_A,A, distance, dderH)
    # / G, if >1 PPs | Gs:
    if _G.extH and G.extH: _G.extH.comp_(G.extH, dderH, rn, fagg=1, flat=1)  # always true in der+
    if _G.derH and G.derH: _G.derH.comp_(G.derH, dderH, rn, fagg=1, flat=0)  # append and sum new dderH to base dderH

    link.derH.append_(dderH, flat=0)  # append nested, higher-res lower-der summation in sub-G extH
    iEt[:] = np.add(iEt,dderH.Et)  # init eval rng+ and form_graph_t by total m|d?
    for i in 0,1:
        Val, Rdn = dderH.Et[i::2]
        if Val > G_aves[i] * Rdn:
            if not fd:  # else old links
                for node in _G,G:
                    if fG:
                        for _link in node.rim:  # +med_links for der+
                            if comp_angle(link.angle, _link.angle)[0] > ave_mA:
                                _link.rim += [link]  # med_links angle should also match
                        node.rim += [link]
                    else:  # node is Clink, all mediating links in link.rim should have matching angle:
                        if comp_angle(node.rim[-1].angle, link.angle)[0] > ave_mA:
                            node.rim += [link]
                                for node in _G,G:  # draft:
                    # or:
                    llink = node.rim[-1]  # last mediating link, doesn't matter for CG.rim
                    _node = llink.node_[1] if llink.node_[0] is node else llink.node_[0]
                    rim = []  # all mA links mediated by last-link _node
                    for _link in reversed.node.rim:  # +med_links for der+
                        if _node in _link.node_:
                            if comp_angle(link.angle, _link.angle)[0] > ave_mA:
                                rim += [link]
                        else:  # different mediating _node, different rim layer
                            node.rim += rim  # no op if empty
                            break  # for both fd?

                fd = 1  # to not add the same link twice
            _G.Et[i] += Val; G.Et[i] += Val
            _G.Et[2+i] += Rdn; G.Et[2+i] += Rdn  # per fork link in both Gs
            # if select fork links: iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dderH.Et[i::2])]
'''
        # comp G.link_ if fd else G.node_:
        for i in 0,1:  # comp co-mediating links between hyperlink nodes, while prior match per direction:
            new_rim = []
            for _mlink_,mlink_ in product(_G.rim_t[i],G.rim_t[i]):
                for _mlink, mlink in product(_mlink_,mlink_):  # use product here and above to get all possible pairs between 2 rims?
                    if _mlink in mlink.compared_: continue
                    (_y,_x),(y,x) = box2center(_mlink.box),box2center(mlink.box)
                    dy=_y-y; dx=_x-x; dist = np.hypot(dy,dx)  # distance between link centers, not evaluated?
                    mA,dA = comp_angle(_mlink.angle,mlink.angle)  # node-mediated, distance eval in agg+ only
                    if mA > ave_mA:
                        _mlink.compared_ += [mlink]; mlink.compared_ += [_mlink]
                        _derH = _mlink.derH; et = _derH.Et
                        mLink = Clink(node_=[_mlink,mlink],distance=dist,angle=(dy,dx),box=extend_box(mlink.box,_mlink.box),
                                      derH=CH(H=deepcopy(_derH.H), Et=[et[0]+mA, et[1]+dA, et[2]+mA<dA, et[3]+dA<=mA]))
                        comp_G(mLink, Et, fd=1, ri=i)
                        if et[0] > ave * et[2]: new_rim += [mLink]  # combined hyperlink, not in mlinks
                        else: break  # comp next mediated link if mediating match
            link.rim_t[i] += [new_rim]  # nest per rng+?
'''

def comp_ext(_L,L,_S,S,_A,A, dist,direction=None):  # compare non-derivatives: node_ L,S,A, dist,direction:

    dL = _L - L;      mL = min(_L,L) - ave_mL  # direct match
    dS = _S/_L - S/L; mS = min(_S,S) - ave_mL  # sparsity is accumulated over L
    mA, dA = comp_angle(_A,A)  # angle is not normalized

    prox = ave_dist - dist  # proximity = inverted distance (position difference), no prior accum to n
    M = prox + mL + mS + mA
    D = dist + abs(dL) + abs(dS) + abs(dA)  # signed dA?
    # dist and direction are currently external to comparands,
    # L,S,A: former dist and direction now internalized to compared nodes
    if direction:  # Link angle, direction alignment for link nodes only?
        Ma,Da = 0,0
        for angle in _A, A:
            ma,da = comp_angle(direction,angle)
            Ma+=ma; Da+=da
        M+=Ma; D+=Ma
        alignt = [Ma,Da]
    else: alignt = []

    mrdn = M > D; drdn = D<= M
    mdec = prox / max_dist + mL/ max(L,_L) + mS/ max(S,_S) if S or _S else 1 + mA  # Amax = 1
    ddec = dist / max_dist + mL/ (L+_L) + dS/ (S+_S) if S or _S else 1 + dA

    return [M,D,mrdn,drdn], [mdec,ddec], [prox,dist, alignt, mL,dL, mS,dS, mA,dA]


class CH(CBase):  # generic derivation hierarchy with variable nesting
    '''
    len layer +extt: 2, 3, 6, 12, 24,
    or without extt: 1, 1, 2, 4, 8..: max n of tuples per der layer = summed n of tuples in all lower layers:
    lay1: par     # derH per param in vertuple, layer is derivatives of all lower layers:
    lay2: [m,d]   # implicit nesting, brackets for clarity:
    lay3: [[m,d], [md,dd]]: 2 sLays,
    lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays
    '''
    def __init__(He, nest=0, n=0, Et=None, relt=None, H=None):
        super().__init__()
        He.nest = nest  # nesting depth: -1/ ext, 0/ md_, 1/ derH, 2/ subH, 3/ aggH
        He.n = n  # total number of params compared to form derH, summed in comp_G and then from nodes in sum2graph
        He.Et = [0,0,0,0] if Et is None else Et   # evaluation tuple: valt, rdnt
        He.relt = [0,0] if relt is None else relt  # m,d relative to max possible m,d
        He.H = [] if H is None else H  # hierarchy of der layers or md_

    def __bool__(H): return H.n != 0

    def add_(HE, He, irdnt=None):  # unpack down to numericals and sum them

        if irdnt is None: irdnt = []
        if HE:
            ddepth = abs(HE.nest-He.nest)  # compare nesting depth, nest lesser He: md_-> derH-> subH-> aggH:
            if ddepth:
                nHe = [HE,He][HE.nest > He.nest]  # He to be nested
                while ddepth > 0:
                    nHe.H = [CH(H=nHe.H, Et=copy(nHe.Et), nest=nHe.nest)]; ddepth -= 1

            if isinstance(HE.H[0], CH):
                H = []
                for Lay, lay in zip_longest(HE.H, He.H, fillvalue=None):
                    if lay:  # to be summed
                        if Lay is None: Lay = CH()
                        Lay.add_(lay, irdnt)  # recursive unpack to sum md_s
                    H += [Lay]
                HE.H = H
            else:
                HE.H = [V+v for V,v in zip_longest(HE.H, He.H, fillvalue=0)]  # both Hs are md_s
            # default:
            Et, et = HE.Et, He.Et
            HE.Et = np.add(HE.Et, He.Et); HE.relt = np.add(HE.relt, He.relt)
            if any(irdnt): Et[2:] = [E+e for E,e in zip(Et[2:], irdnt)]
            HE.n += He.n  # combined param accumulation span
            HE.nest = max(HE.nest, He.nest)
        else:
            HE.copy(He)  # initialization

    def append_(HE,He, irdnt=None, flat=0):

        if irdnt is None: irdnt = []
        if flat: HE.H += deepcopy(He.H)  # append flat
        else:  HE.H += [He]  # append nested

        Et, et = HE.Et, He.Et
        HE.Et = np.add(HE.Et, He.Et); HE.relt = np.add(HE.relt, He.relt)
        if irdnt: Et[2:4] = [E+e for E,e in zip(Et[2:4], irdnt)]
        HE.n += He.n  # combined param accumulation span
        # HE.nest = max(HE.nest, He.nest)

    def comp_(_He, He, dderH, rn=1, fagg=0, flat=1):  # unpack tuples (formally lists) down to numericals and compare them

        ddepth = abs(_He.nest - He.nest)
        n = 0
        if ddepth:  # unpack the deeper He: md_<-derH <-subH <-aggH:
            uHe = [He,_He][_He.nest>He.nest]
            while ddepth > 0:
                uHe = uHe.H[0]; ddepth -= 1  # comp 1st layer of deeper He:
            _cHe,cHe = [uHe,He] if _He.nest>He.nest else [_He,uHe]
        else: _cHe,cHe = _He,He

        if isinstance(_cHe.H[0], CH):  # _lay is He_, same for lay: they are aligned above
            Et = [0,0,0,0]  # Vm,Vd, Rm,Rd
            relt = [0,0]  # Dm,Dd
            dH = []
            for _lay,lay in zip(_cHe.H,cHe.H):  # md_| ext| derH| subH| aggH, eval nesting, unpack,comp ds in shared lower layers:
                if _lay and lay:  # ext is empty in single-node Gs
                    dlay = _lay.comp_(lay, CH(), rn, fagg=fagg, flat=1)  # dlay is dderH
                    Et = np.add(Et, dlay.Et)
                    relt = np.add(relt, dlay.relt)
                    dH += [dlay]; n += dlay.n
                else:
                    dH += [CH()]  # empty?
        else:  # H is md_, numerical comp:
            vm,vd,rm,rd, decm,decd = 0,0,0,0,0,0
            dH = []
            for i, (_d,d) in enumerate(zip(_cHe.H[1::2], cHe.H[1::2])):  # compare ds in md_ or ext
                d *= rn  # normalize by comparand accum span
                diff = _d-d
                match = min(abs(_d),abs(d))
                if (_d<0) != (d<0): match = -match  # if only one comparand is negative
                if fagg:
                    maxm = max(abs(_d), abs(d))
                    decm += abs(match) / maxm if maxm else 1  # match / max possible match
                    maxd = abs(_d) + abs(d)
                    decd += abs(diff) / maxd if maxd else 1  # diff / max possible diff
                vm += match - aves[i]  # fixed param set?
                vd += diff
                dH += [match,diff]  # flat
            Et = [vm,vd,rm,rd]; relt= [decm,decd]
            n = len(_cHe.H)/12  # unit n = 6 params, = 12 in md_

        dderH.append_(CH(nest=min(_He.nest,He.nest), Et=Et, relt=relt, H=dH, n=n), flat=flat)  # currently flat=1
        return dderH

    def copy(_H, H):
        for attr, value in H.__dict__.items():
            if attr != '_id' and attr in _H.__dict__.keys():  # copy only the available attributes and skip id
                setattr(_H, attr, deepcopy(value))


def rng_recursion(root, Et, fagg):  # comp Gs in agg+, links in sub+
    nrng = 1

    if fagg:  # distance eval, else fd: mangle eval
        while True:
            for link in list(combinations(root.node_,r=2)):
                _G, G = link
                if _G in G.compared_: continue
                cy, cx = box2center(G.box); _cy, _cx = box2center(_G.box)
                dy = cy-_cy; dx = cx-_cx;  dist = np.hypot(dy,dx)
                # eval distance between node centers:
                if nrng==1: fcomp = dist <= ave_dist
                else:
                    M = (G.Et[0]+_G.Et[0])/2; R = (G.Et[2]+_G.Et[2])/2  # local
                    fcomp = M / (dist/ave_dist) > ave * R
                if fcomp:
                    G.compared_ += [_G]; _G.compared_ += [G]
                    Link = Clink(node_=[_G, G], distance=dist, angle=[dy, dx], box=extend_box(G.box, _G.box))
                    if comp_G(Link, Et, fd=0):
                        for G in link: G.rim += [Link]
            # reuse combinations
            if Et[0] > ave_Gm * Et[2] * nrng: nrng += 1
            else: break
    else:
        _links = root.link_  # der+'rng+: directional and node-mediated comp link
        while True:
            links = []
            for link in _links:
                # revise:
                # link.rimt__ is a link-centered graph, that would overlap other connected-link - centered graphs.
                # Where link is an exemplar, select max connected exemplars per rng to reduce resolution of rng+

                for rim__ in link.rim_t:  # compare equimediated Clink nodes in hyperlink rims, if mediating links angle match?
                    if len(rim__[-1]) > nrng-1:  # rim is a hyperlink, nested by mediation / nrng
                        rim = rim__[-1][-1]  # link.rim is nested per der+( rng+
                        _link_ = []  # mA links in new rim layer
                    else: break  # med rng exhausted
                    for _link in rim:
                        if _link in link.compared_: continue
                        (_y,_x),(y,x) = box2center(link.box),box2center(_link.box)
                        dy=_y-y; dx=_x-x; dist = np.hypot(dy,dx)  # distance between link centers
                        mA,dA = comp_angle(_link.angle,link.angle)
                        if mA > ave_mA:
                            _link.compared_+=[link]; link.compared_+=[_link]
                            _derH = _link.derH; et = _derH.Et
                            Link = Clink(node_=[_link,link],distance=dist,angle=(dy,dx),box=extend_box(_link.box,link.box),
                                         derH=CH(H=deepcopy(_derH.H), Et=[et[0]+mA, et[1]+dA, et[2]+mA<dA, et[3]+dA<=mA]))
                            if comp_G(Link, Et, fd=1):
                                _link_ += [Link]  # append link.rim
                                links += [Link]  # for next layer
                    rim__[-1] += [_link_]
            if Et[0] > ave_Gm * Et[2] * nrng:  # rng+ eval per arg cluster because comp is bilateral, 2nd test per new pair
                nrng += 1; _links = links
            else: break

    return nrng, Et

def convolve_graph(iG_):  # node connectivity = sum surround link vals, incr.mediated: Graph Convolution of Correlations
    '''
    Aggregate direct * indirect connectivity per node from indirect links via associated nodes, in multiple cycles.
    Each cycle adds contributions of previous cycles to linked-nodes connectivity, propagated through the network.
    Math: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/node_connect.png
    Link-mediated in all iterations, no layers of kernels, update node Et.
    Add der+: cross-comp root.link_, update link Et.
    Not sure: sum and compare kernel params: reduced-resolution rng+, lateral link-mediated vs. vertical in agg_kernels?
    '''
    _G_ = iG_; fd = isinstance(iG_[0],Clink)  # ave = G_aves[fd]  # ilink_ if fd, very rare?
    while True:
        # eval accumulated G connectivity with node-mediated range extension
        G_ = []  # next connectivity expansion, use link_ instead? more selective by DV,Lent
        mediation = 1  # n intermediated nodes, increasing decay
        for G in _G_:
            uprim = []  # >ave updates of direct links
            for i in 0,1:
                val,rdn = G.Et[i::2]  # rng+ for both segment forks
                if not val: continue  # G has no new links
                ave = G_aves[i]
                for link in G.rim if isinstance(G,CG) else G.rimt__[-1][-1][0] + G.rimt__[-1][-1][1]:  # not updated
                    # > ave derGs in new fd rim:
                    lval,lrdn = link.Et[i::2]  # step=2, graph-specific vals accumulated from surrounding nodes, or use link.node_.Et instead?
                    decay =  (link.relt[i] / (link.derH.n * 6)) ** mediation  # normalized decay at current mediation
                    _G = link.node_[0] if link.node_[1] is G else link.node_[1]
                    _val,_rdn = _G.Et[i::2] # current-loop vals and their difference from last-loop vals, before updating:
                    V = (val+_val) * decay; dv = V-lval
                    R = (rdn+_rdn)  # rdn doesn't decay
                    link.Et[i::2] = [V,R]  # last-loop vals for next loop | segment_node_, dect is not updated
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

# not relevant with kernel-based rng+, use discrete rng+ to append link.rimt_ tree:
def add_rim_(link_):  # sub+: bidirectional rim_t += _links from last-layer node.rims, if they match link:

    for link in link_:
        new_rimt = [[],[]]
        if link.rimt__:
            rimt = [[],[]]
            for _rimt in link.rimt__[-1]:
                rimt[0] += _rimt[0]; rimt[1] += _rimt[1]  # flatten nested rimt_
        else:  rimt = [link.node_[0].rim,link.node_[1].rim]  # 1st sub+/ link
        for i, rim, new_rim in zip((0,1), rimt, new_rimt):
            for _link in rim:
                if _link is link or link in new_rim: continue
                angle = link.angle if i else [-d for d in link.angle]  # reverse angle direction for left link comp
                # or compare the whole links, which is the actual der+?
                if comp_angle(angle, _link.angle)[0] > ave_mA:
                    new_rim += [_link]  # from rng++/ last der+?
        if any(new_rimt): link.rimt__ += [[new_rimt]]  # double nesting for next rng+
        # this can be default? Else we will need a lot of checking for empty rimt__ later


