'''
            for G in link._G, link.G:
                # draft
                if len_root_HH > -1:  # agg_compress, increase nesting
                    if len(G.rim_t)==len_root_HH:  # empty rim layer, init with link:
                        if fd:
                            G.Vt=[0,Val]; G.Rt=[0,Rdn]; G.Dt=[0,Dec]  # temporary
                            G.rim_t = [[[[[],[link]]]],2]; G.rim_t = [[[[[],[link]]]],2]  # init rim_tHH, depth = 2
                        else:
                            G.Vt=[Val,0]; G.Rt=[Rdn,0]; G.Dt=[Dec,0]
                            G.rim_t = [[[[[link],[]]]],2]; G.rim_t = [[[[[link],[]]]],2]
                    else:
                        G.Vt[fd] += Val; G.Rt[fd] += Rdn; G.Dt[fd] += Dec  # accum rim layer with link
                        G.rim_t[0][-1][-1][fd] += [link]; G.rim_t[0][-1][-1][fd] += [link]  # append rim_tHH
                else:
                    if len(G.rim_t)==len_root_H:  # empty rim layer, init with link:
                        if fd:
                            G.Vt=[0,Val]; G.Rt=[0,Rdn]; G.Dt=[0,Dec]
                            G.rim_t = [[[[],[link]]],1]; G.rim_t = [[[[],[link]]],1]  # init rim_tH, depth = 1
                        else:
                            G.Vt=[Val,0]; G.Rt=[Rdn,0]; G.Dt=[Dec,0]
                            G.rim_t = [[[[link],[]]],1]; G.rim_t = [[[[link],[]]],1]
                    else:
                        G.Vt[fd] += Val; G.Rt[fd] += Rdn; G.Dt[fd] += Dec  # accum rim layer with link
                        G.rim_t[0][-1][fd] += [link]; G.rim_t[0][-1][fd] += [link]  # append rim_tH
'''

def rd_recursion(rroot, root, node_, lenH, lenHH, Et, nrng=1):  # rng,der incr over same G_,link_ -> fork tree, represented in rim_t

    Vt, Rt, Dt = Et
    for fd, Q, V,R,D in zip((0,1),(node_,root.link_), Vt,Rt,Dt):  # recursive rng+,der+

        ave = G_aves[fd]
        if fd and rroot == None:
            continue  # no link_ and der+ in base fork
        if V >= ave * R:  # true for init 0 V,R; nrng if rng+, else 0:
            if not fd:
                nrng += 1
            G_,(vt,rt,dt) = cross_comp(Q, lenH, lenHH, nrng*(1-fd))
            for inp in Q:
                if isinstance(inp,CderG): Gt = [inp._G, inp.G]  # link
                else: Gt = [inp]  # G, packed in list for looping:
                for G in Gt:
                    for i, link in enumerate(link_):  # add esubH layer, temporary?
                        if i:
                            sum_derHv(G.esubH[-1], link.subH[-1], base_rdn=link.Rt[fd])  # [derH, valt,rdnt,dect,extt,1]
                        else:
                            G.esubH += [deepcopy(link.subH[-1])]  # link.subH: cross-der+) same rng, G.esubH: cross-rng?

            # we should sum links' esubH per G instead?
            G_ = defaultdict(list)
            for link in link_:
                for G in (link.G, link._G): G_[G] += [link]  # pack links per G
            # sum links' esubH per G
            for G in G_.keys():
                for i, link in enumerate(G_[G]):
                    if i:
                        sum_derHv(G.esubH[-1], link.subH[-1], base_rdn=link.Rt[fd])  # [derH, valt,rdnt,dect,extt,1]
                    else:
                        G.esubH += [deepcopy(link.subH[-1])]  # link.subH: cross-der+) same rng, G.esubH: cross-rng?

            for i, v,r,d in zip((0,1), vt,rt,dt):
                Vt[i]+=v; Rt[i]+=rt[i]; Dt[i]+=d
                if v >= ave * r:
                    if i: root.link_+= link_  # rng+ links
                    # adds to root Et + rim_t, and Et per G:
                    rd_recursion(rroot, root, node_, lenH, lenHH, [vt,rt,dt], nrng)
    return nrng


def form_graph_t(root, lenH, lenHH, Et, nrng):

    rdepth = (lenH != None) + (lenHH != None)
    G_ = root.node_[not nrng] if isinstance(root.node_[0],list) else root.node_
    _G_ = []
    # not revised below:
    for G in G_:  # select Gs connected in current layer
        if G.rim_t[-1] > rdepth:  # if G is updated in comp_G, their depth should > rdepth
          _G_ += [G]
        else:
            if lenHH:  # check if lenHH is incremented
                if (G.rim_t[0] > lenHH):
                    _G_ += [G]
            else:  # check if lenH is incremented
                if (G.rim_t[0] > lenH):
                    _G_ += [G]

    node_connect(_G_)  # Graph Convolution of Correlations over init _G_
    node_t = []
    for fd in 0,1:
        if Et[0][fd] > ave * Et[1][fd]:  # eValt > ave * eRdnt, else no clustering, keep root.node_
            graph_ = segment_node_(root, _G_, fd, nrng)  # fd: node-mediated Correlation Clustering
            if not graph_: continue
            for graph in graph_:  # eval sub+ per node
                if graph.Vt[fd] * (len(graph.node_)-1)*root.rng > G_aves[fd] * graph.Rt[fd]:
                    # last sub+ val -> sub+:
                    if graph.node_[0].rim_t[-1] == 2:  # test root aggH?
                        lenH = len(graph.node_[0].rim_t[0][-1][0])
                        lenHH = len(graph.node_[0].rim_t[0])
                    else:  # rim_tH
                        lenH = len(graph.node_[0].rim_t[0])
                        lenHH = None
                    agg_recursion(root, graph, graph.node_[-1], lenH, lenHH, nrng+1*(1-fd))  # nrng+ if not fd
                    rroot = graph
                    rfd = rroot.fd
                    while isinstance(rroot.roott, list) and rroot.roott[rfd]:  # not blob
                        rroot = rroot.roott[rfd]
                        Val,Rdn = 0,0
                        if isinstance(rroot.node_[-1][0], list):  # node_ is node_t
                            node_ = rroot.node_[-1][rfd]
                        else: node_ = rroot.node_[-1]

                        for node in node_:  # sum vals and rdns from all higher nodes
                            Rdn += node.rdnt[rfd]
                            Val += node.valt[rfd]
                        # include rroot.Vt and Rt?
                        if Val * (len(rroot.node_[-1])-1)*rroot.rng > G_aves[fd] * Rdn:
                            # not sure about nrg here
                            agg_recursion(root, graph, rroot.node_[-1], len(rroot.aggH[-1][0]), rfd, nrng+1*(1-rfd))  # nrng+ if not fd
                else:
                    root.fback_t[root.fd] += [[graph.aggH, graph.valt, graph.rdnt, graph.dect]]
                    feedback(root,root.fd)  # update root.root..
            node_t += [graph_]  # may be empty
        else: node_t += []
    if any(node_t): G_[:] = node_t


def cross_comp(Q, lenH, lenHH, nrng):

    Et = [[0,0],[0,0],[0,0]]  # grapht link_' eValt, eRdnt, eDect(currently not used)
    fd = not nrng

    if fd:  # der+
        for link in Q:  # inp_= root.link_, reform links
            if (len(link.G.rim_t[0])==lenH  # the link was formed in prior rd+
                and link.Vt[1] > G_aves[1]*link.Rt[1]):  # >rdn incr
                comp_G(link, Et, lenH, lenHH)
    else:   # rng+
        for _G, G in combinations(Q, r=2):  # form new link_ from original node_
            dy = _G.box.cy - G.box.cy; dx = _G.box.cx - G.box.cx
            dist = np.hypot(dy,dx)
            # max distance between node centers, init=2
            if 2*nrng > dist > 2*(nrng-1):  # G,_G are within rng and were not compared in prior rd+
                link = CderG(_G=_G, G=G)
                comp_G(link, Et, lenH, lenHH)

    return Et
