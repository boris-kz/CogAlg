for part, epart in zip((valt, rdnt, dect), (evalt, erdnt, edect)):
    for i in 0, 1:  # internalize externals
        part[i] += epart[i]

Val, Rdn = 0, 0
for subH in graph.aggH[1:]:  # eval by sum of val,rdn of top subLays in lower aggLevs:
    if not subH[0]: continue  # empty rim if no prior comp
    Val += subH[0][-1][1][fd]  # subH: [derH, valt,rdnt,dect]
    Rdn += subH[0][-1][2][fd]

def sum_Hts(ValHt,RdnHt,DecHt, valHt,rdnHt,decHt):
    # loop m,d Hs, add combined decayed lower H/layer?
    for ValH,valH, RdnH,rdnH, DecH,decH in zip(ValHt,valHt, RdnHt,rdnHt, DecHt,decHt):
        ValH[:] = [V+v for V,v in zip_longest(ValH, valH, fillvalue=0)]
        RdnH[:] = [R+r for R,r in zip_longest(RdnH, rdnH, fillvalue=0)]
        DecH[:] = [D+d for D,d in zip_longest(DecH, decH, fillvalue=0)]
'''
derH: [[tuplet, valt, rdnt, dect]]: default input from PP, rng+|der+, sum min len?
subH: [derH_t]: m,d derH, m,d ext added by agg+ as 1st tuplet
aggH: [subH_t]: composition levels, ext per G, 
'''

def form_graph_t(root, G_, Et, fd):  # root_fd, form mgraphs and dgraphs of same-root nodes

    node_connect(copy(G_))  # init _G_ for Graph Convolution of Correlations
    graph_t = [[],[]]
    for i in 0,1:
        if Et[0][i] > ave * Et[1][i]:  # eValt > ave * eRdnt, else no clustering
            graph_t[i] = segment_node_(root, G_, fd)  # if fd: node-mediated Correlation Clustering
            # add alt_graphs?
    for fd, graph_ in enumerate(graph_t):  # breadth-first for in-layer-only roots
        for graph in graph_:
            # sub+ / last layer val dev, external to agg+ vs. internal in comp_slice sub+
            if graph.Vt[fd] * (len(graph.node_tH[-1])-1)*root.rng > G_aves[fd] * graph.Rt[fd]:
                # add empty layer:
                for G in graph.node_tH[-1]:  # node_
                    G.Vt, G.Rt, G.Dt = [0,0],[0,0],[0,0]
                    G.rim_tH += [[[],[]]]; G.Rim_tH += [[[],[]]]
                agg_recursion(root, graph, graph.node_tH[-1], fd)  # flat node_
            else:  # feedback after sub+
                root.fback_t[root.fd] += [[graph.aggH, graph.valt, graph.rdnt, graph.dect]]
                root.valt[root.fd] += graph.valt[fd]  # merge forks in root fork
                root.rdnt[root.fd] += graph.rdnt[fd]
                root.dect[root.fd] += graph.dect[fd]
        if root.fback_t and root.fback_t[fd]:  # recursive feedback after all G_ sub+
            feedback(root, fd)  # update root.root.. aggH, valHt,rdnHt


def vectorize_root(blob, verbose):  # vectorization pipeline is 3 composition levels of cross-comp,clustering:

    edge, adj_Pt_ = slice_edge(blob, verbose)  # lateral kernel cross-comp -> P clustering
    comp_P_(edge, adj_Pt_)  # vertical, lateral-overlap P cross-comp -> PP clustering
    # PP cross-comp -> discontinuous graph clustering:
    for fd, node_ in enumerate(edge.node_t):
        if edge.valt[fd] * (len(node_)-1) * (edge.rng+1) <= G_aves[fd] * edge.rdnt[fd]: continue
        G_,i = [],0
        for PP in node_:  # convert select CPPs to Cgraphs:
            if PP.valt[fd] * (len(node_)-1) * (PP.rng+1) <= G_aves[fd] * PP.rdnt[fd]: continue
            derH,valt,rdnt = PP.derH,PP.valt,PP.rdnt
            G_ += [Cgraph(ptuple=PP.ptuple, derH=derH, valt=copy(valt), rdnt=copy(rdnt), L=PP.ptuple[-1],
                          box=PP.box, link_=PP.link_, node_tH=[PP.node_t])]
            i += 1  # G index in node_
        if G_:
            node_[:] = G_  # replace  PPs with Gs
            agg_recursion(None, edge, node_, fd=0)  # edge.node_ = graph_t, micro and macro recursive

def form_graph_t(root, G_, Et, fd, nrng):  # root_fd, form mgraphs and dgraphs of same-root nodes

    _G_ = []  # init with not empty rims:
    for G in G_:
        if len(G.Rim_tH) != len(root.esubH):  # G was compared in this root
            link_ = list(set(G.Rim_tH[-1][0] + G.Rim_tH[-1][1]))  # links are shared across both forks
            if link_:  # not empty (it may empty if val's eval in comp_G is false)
                link = link_[0]
                G.Vt=copy(link.Vt); G.Rt=copy(link.Rt); G.Dt=copy(link.Dt)  # reset
                for link in link_[1:]:
                    for i in 0,1:
                        G.Vt[i]+=link.Vt[i]; G.Rt[i]+=link.Vt[i]; G.Dt[i]+=link.Dt[i]
            else:
                G.Vt = [0,0]; G.Rt = [0,0]; G.Dt = [0,0]  # we still need to reset them? Else they will be using prior layer vals
            _G_ += [G]
    ...

def vectorize_root(blob, verbose):  # vectorization pipeline is 3 composition levels of cross-comp,clustering:

    edge, adj_Pt_ = slice_edge(blob, verbose)  # lateral kernel cross-comp -> P clustering

    comp_P_(edge, adj_Pt_)  # vertical, lateral-overlap P cross-comp -> PP clustering
    edge.node_ = [edge.node_]  # convert to node_tH

    for fd, node_ in enumerate(edge.node_[-1]):  # node_ is generic for any nesting depth
        if edge.valt[fd] * (len(node_)-1) * (edge.rng+1) <= G_aves[fd] * edge.rdnt[fd]:
            continue  # else PP cross-comp -> discontinuous graph clustering:
        G_ = []
        for PP in node_:  # eval PP for agg+
            if PP.valt[fd] * (len(node_)-1) * (PP.rng+1) <= G_aves[fd] * PP.rdnt[fd]: continue
            PP.roott = [None,None]
            G_ += [PP]
        if G_:
            agg_recursion(None, edge, G_, fd=0)  # edge.node_ = graph_t, micro and macro recursive

    return edge

def agg_recursion(rroot, root, G_, fd, nrng=1):  # + fpar for agg_parP_? compositional agg|sub recursion in root graph, cluster G_

    Et = [[0,0],[0,0],[0,0]]  # eValt, eRdnt, eDect(currently not used)
    lenRoot = len(root.rim_tH)  # to init G.rim_tH

    if fd:  # der+
        for link in root.link_:  # reform links
            if link.Vt[1] < G_aves[1]*link.Rt[1]: continue  # maybe weak after rdn incr?
            comp_G(link._G,link.G,link, Et, lenRoot)
    else:   # rng+
        for i, _G in enumerate(G_):  # form new link_ from original node_
            for G in G_[i+1:]:
                dy = _G.box.cy - G.box.cy; dx = _G.box.cx - G.box.cx
                if np.hypot(dy,dx) < 2 * nrng:  # max distance between node centers, init=2
                    link = CderG(_G=_G, G=G)
                    comp_G(_G, G, link, Et, lenRoot)

    GG_t = form_graph_t(root, G_, Et, fd, nrng)  # root_fd, eval sub+, feedback per graph
    # agg+ xcomp-> form_graph_t loop sub)agg+, vs. comp_slice sub+ loop-> eval-> xcomp
    for GG_ in GG_t:
        if root.valt[0] * (len(GG_)-1)*root.rng > G_aves[fd] * root.rdnt[0]:
            # 1st xcomp in GG_, root update in form_t, max rng=2:
            agg_recursion(rroot, root, GG_, fd=0)

    if GG_t[0] or GG_t[1]:  # node_->node_t if no agg+, else sub+ is local to sub_G formed in agg+
        root.node_[:] = [GG_t]
    if rroot:  # base node_ agg+, fd=2
        rroot.fback_t[2] += [[root.aggH,root.valt,root.rdnt,root.dect]]
        feedback(rroot,2)  # update root.root..

def form_graph_t(root, G_, Et, fd, nrng):  # form Gm_,Gd_ of same-root nodes

    _G_ = [G for G in G_ if len(G.rim_tH)>len(root.rim_tH)]  # prune unconnected Gs

    node_connect(_G_)  # Graph Convolution of Correlations over init _G_
    graph_t = [[],[]]
    for i in 0,1:
        if Et[0][i] > ave * Et[1][i]:  # eValt > ave * eRdnt, else no clustering
            graph_t[i] = segment_node_(root, _G_, fd, nrng)  # fd: node-mediated Correlation Clustering

    for fd, graph_ in enumerate(graph_t):  # breadth-first for in-layer-only roots
        for graph in graph_:
            if isinstance(graph.node_[0], Cgraph): continue  # sub+ only for node_t:
            if graph.Vt[fd] * (len(graph.node_[fd])-1)*root.rng > G_aves[fd] * graph.Rt[fd]:
                # sub+, external to agg+, vs internal in comp_slice sub+:
                agg_recursion(root, graph, graph.node_[fd], fd, nrng+1*(1-fd))  # rng++ if not fd
            else:
                root.fback_t[root.fd] += [[graph.aggH, graph.valt, graph.rdnt, graph.dect]]
                feedback(root,root.fd)  # update root.root..

    return graph_t

def feedback(root, ifd):  # called from form_graph_, append new der layers to root

    # fback_ maps to node_: flat?
    AggH, Valt, Rdnt, Dect = deepcopy(root.fback_t[ifd].pop(0))
    # init with 1st tuple
    while root.fback_t[ifd]:
        aggH, valt, rdnt, dect = root.fback_t[ifd].pop(0)
        sum_aggHv(AggH, aggH, base_rdn=0)
        for j in 0,1:
            Valt[j] += valt[j]; Rdnt[j] += rdnt[j]; Dect[j] += dect[j]  # -> root.fback_t

    fd = 1 if ifd==1 else 0  # 2->0
    if Valt[fd] > G_aves[fd] * Rdnt[fd]:  # or compress each level?
        root.aggH += AggH  # higher levels are not affected
        for j in 0,1:
            root.valt[j] += Valt[j]; root.rdnt[j] += Rdnt[j]; root.dect[j] += Dect[j]  # both forks sum in same root

    if not isinstance(root.roott, CBlob):  # not Edge
        rroot = root.roott[fd]
        if rroot:
            rfd = ifd if ifd==2 else fd
            fback_ = rroot.fback_t[rfd]  # map to node_:
            rnode_ = rroot.node_ if isinstance(rroot.node_[0],Cgraph) else rroot.node_[rfd]  # node_t

            if fback_ and (len(fback_) == len(rnode_)):
                # after all rroot nodes terminate and feed back:
                feedback(rroot, ifd)  # sum2graph adds aggH per rng, feedback adds deeper sub+ layers