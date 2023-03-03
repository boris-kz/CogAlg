class Cgraph(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    G = lambda: None  # same-scope lower-der|rng G.G.G., in all nodes beyond PP
    root = lambda: None  # root graph or inder_ G, element of ex.H[-1][fd]
    # up,down trees:
    ex = object    # Inder_ ) link_) uH: context, Lev+= root tree slice: forward, comp summed up-forks?
    inder_ = list  # inder_ ) node_) wH: contents, Lev+= node tree slice: feedback, Lev/agg+, lev/sub+?
    # inder_ params:
    node_ = list  # single-fork, conceptually H[0], concat sub-node_s in ex.H levs
    fterm = lambda: 0  # G.node_ sub-comp or feedback was terminated
    Link_ = lambda: Clink_()  # unique links within node_, -> inder_?
    H = list  # Lev+= node'tree'slice, ~up-forking Levs in ex.H?
    val = int
    fds = list
    rdn = lambda: 1
    rng = lambda: 1
    nval = int  # of open links: base alt rep
    box = lambda: [0,0,0,0,0,0]  # center y,x, y0,x0, yn,xn, no need for max distance?
    x0 = float
    y0 = float  # center: box x0|y0 + L/2
    xn = float  # max distance from center
    yn = float
    L = list  # der L, init None
    S = int  # sparsity: ave len link
    A = list  # area|axis: Dy,Dx, ini None
    alt_graph_ = list  # contour + overlapping contrast graphs
    alt_Graph = None  # conditional, summed and concatenated params of alt_graph_


def fforward(root):  # top-down update node.ex.H, breadth-first

    for node in root.node_:
        for i, (rLev,nLev) in enumerate(zip_longest(root.ex.H, node.ex.H[1:], fillvalue=[])):  # root.ex.H maps to node.ex.H[1:]
            if rLev:
                j = sum(fd*(2**k) for k,fd in enumerate(rLev.fds[i:]))
                if not nLev:  # init:
                    nLev=CpH(H=[Cgraph() for fork in rLev.fds[i:]])  # fds=copy(rLev.fds)?
                sum_inder_(nLev.H[j].inder_, rLev.H[j].inder_)  # same fork
        ffV,ffR = 0,0
        for Lev in root.ex.H:
            ffV += Lev.val; ffR += Lev.rdn

        if node.fterm and ffV/ffR > aveG:
            node.fterm = 0
            fforward(node)


def sum2graph_(root, graph_, fd):  # sum node and link params into graph, inder_ in agg+ or player in sub+

    Graph_ = []  # Cgraphs
    for graph in graph_:  # CpHs
        if graph.val < aveG:  # form graph if val>min only
            continue
        X0,Y0 = 0,0
        for G in graph.H:  # CpH
            X0 += G.x0; Y0 += G.y0
        L = len(graph.H); X0/=L; Y0/=L; Xn,Yn = 0,0
        # conditional ex.inder_: remove if few links?
        Graph = Cgraph(fds=copy(G.fds), ex=Cgraph(node_=Clink_(),A=[0,0]), x0=X0, xn=Xn, y0=Y0, yn=Yn)
        # form G, keep iG:
        node_,Link_= [],[]
        for iG in graph.H:
            Xn = max(Xn, (iG.x0 + iG.xn) - X0)  # box xn = x0+xn
            Yn = max(Yn, (iG.y0 + iG.yn) - Y0)
            sum_G(Graph, iG)  # sum(Graph.uH[-1][fd], iG.pplayers), higher levs += G.G.pplayers | G.uH, lower scope than iGraph
            link_ = [iG.ex.node_.Qm, iG.ex.node_.Qd][fd]
            Link_ = list(set(Link_ + link_))  # unique links in node_
            G = Cgraph(fds=copy(iG.fds), G=iG, root=Graph, ex=Cgraph(node_=Clink_(),A=[0,0]))
            # sum quasi-gradient of links in ex.inder_: redundant to Graph.inder_, if len node_?:
            for derG in link_:
                sum_inder_(G.ex.inder_, [derG.minder_, derG.dinder_][fd]) # local feedback
                G.ex.S += derG.S; G.ex.A[0]+=derG.A[0]; G.ex.A[1]+=derG.A[1]
            l=len(link_); G.ex.L=l; G.ex.S/=l
            sum_ex_H(root, iG)
            node_ += [G]
        Graph.node_ = node_ # lower nodes = G.G..; Graph.root = iG.root
        for Link in Link_:  # sum unique links
            sum_inder_(Graph.inder_, [Link.minder_, Link.dinder_][fd])
            if Graph.inder_[-1]: # top ext
                Graph.inder_[-1][1]+=Link.S; Graph.inder_[-1][2][0]+=Link.A[0]; Graph.inder_[-1][2][1]+=Link.A[1]
            else: Graph.inder_[-1] = [1,Link.S,Link.A]
        L = len(Link_); Graph.inder_[-1][0] = L; Graph.inder_[-1][1] /= L  # last pplayers LSA per Link_
        # inder_ LSA per node_:
        Graph.A = [Xn*2,Yn*2]; L=len(node_); Graph.L=L; Graph.S = np.hypot(Xn-X0,Yn-Y0) / L
        if Graph.ex.H:
            Graph.val += sum([lev.val for lev in Graph.ex.H]) / sum([lev.rdn for lev in Graph.ex.H])  # if val>alt_val: rdn+=len_Q?
        Graph_ += [Graph]

    return Graph_
