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

class CderG(ClusterStructure):  # graph links, within root node_

    node0 = lambda: Cgraph()  # converted to list in recursion
    node1 = lambda: Cgraph()
    minder_ = list  # in alt/contrast if open
    dinder_ = list
    S = int  # sparsity: ave len link
    A = list  # area and axis: Dy,Dx

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

def comp_GQ(_G,G,fsub):  # compare lower-derivation G.G.s, pack results in minder__,dinder__

    minder__,dinder__ = [],[]; Mval,Dval = 0,0; Mrdn,Drdn = 1,1
    Tval= aveG+1  # start
    while (_G and G) and Tval > aveG:  # same-scope if sub+, no agg+ G.G

        minder_, dinder_, mval, dval, mrdn, drdn = comp_G(_G, G, fsub, fex=0)
        minder__+=minder_; dinder__+=dinder_
        Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn  # also /rdn+1: to inder_?
        # comp ex:
        if (Mval + Dval) * _G.ex.val * G.ex.val > aveG:
            mex_, dex_, mval, dval, mrdn, drdn = comp_G(_G.ex, G.ex, fsub, fex=1)
            minder__+=mex_; dinder__+=dex_
            Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn
        else:
            minder__+=[[]]; dinder__+=[[]]  # always after not-empty m|dinder_: list until sum2graph?
        _G = _G.G
        G = G.G
        Tval = (Mval+Dval) / (Mrdn+Drdn)

    return minder__, dinder__, Mval, Dval, Tval  # ext added in comp_G_, not within GQ

def comp_G(_G, G, fsub, fex):

    minder_,dinder_ = [],[]  # ders of implicitly nested list of pplayers in inder_
    Mval, Dval = 0,0; Mrdn, Drdn = 1,1

    minder_,dinder_, Mval,Dval, Mrdn,Drdn = comp_inder_(_G.inder_,G.inder_, minder_,dinder_, Mval,Dval, Mrdn,Drdn)
    # spec:
    _node_, node_ = _G.node_.Q if fex else _G.node_, G.node_.Q if fex else G.node_  # node_ is link_ if fex
    if (Mval+Dval) * _G.val*G.val * len(_node_)*len(node_) > aveG:
        if fex:  # comp link_
            sub_minder_,sub_dinder_ = comp_derG_(_node_, node_, G.fds[-1])
        else:    # comp node_
            sub_minder_,sub_dinder_ = comp_G_(_node_, node_, f1Q=0, fsub=fsub)
        Mval += sum([mxpplayers.val for mxpplayers in sub_minder_])  # + rdn?
        Dval += sum([dxpplayers.val for dxpplayers in sub_dinder_])
        # + sub node_?
        if (Mval+Dval) * _G.val*G.val * len(_G.ex.H)*len(G.ex.H) > aveG:
            # comp uH
            for _lev, lev in zip(_G.ex.H, G.ex.H):  # wH is empty in top-down comp
                for _fork, fork in zip(_lev.H, lev.H):
                    if _fork and fork:
                        fork_minder_, fork_dinder_, mval, dval, tval = comp_GQ(_fork, fork, fsub=0)
                        Mval+= mval; Dval+= dval
                    # else +[[]], if m|dexH:
    # pack m|dnode_ and m|dexH in minder_, dinder_: still implicit or nested?
    else: _G.fterm=1
    # no G.fterm=1: it has it's own specification?
    # comp alts,val,rdn?
    return minder_, dinder_, Mval, Dval, Mrdn, Drdn

def comp_derG_(_derG_, derG_, fd):

    mlink_,dlink_ = [],[]
    for _derG in _derG_:
        for derG in derG_:
            mlink, dlink,_,_,_,_,= comp_inder_([_derG.minder_,_derG.dinder_][fd], [derG.dinder_,_derG.dinder_][fd], [],[],0,0,1,1)
            # add comp fds: may be different?
            mext, dext = comp_ext(1,_derG.S,_derG.A, 1,derG.S,derG.A)
            mlink_ += [mlink] + [mext]  # not sure
            dlink_ += [dlink] + [dext]

    return mlink_, dlink_

def feedback(root):  # bottom-up update root.H, breadth-first

    fbV = aveG+1
    while root and fbV > aveG:
        if all([[node.fterm for node in root.node_]]):  # forward was terminated in all nodes
            root.fterm = 1
            fbval, fbrdn = 0,0
            for node in root.node_:
                # sum nodes in root, sub_nodes in root.H:
                for sub_node in node.node_:
                    fd = sub_node.fds[-1]
                    if not root.H: root.H = [CpH(H=[[],[]])]  # append bottom-up
                    if not root.H[0].H[fd]: root.H[0].H[fd] = Cgraph()
                    if isinstance(sub_node.G, list):
                        sub_inder_ = sub_node.inder_[fd]
                    else:
                        sub_inder_ = sub_node.inder_
                    sum_inder_(root.H[0].H[fd].inder_, sub_inder_)
                    # or sum_G?
                    # sum_H(root.H[1:], sub_node.H)
                    for i, (Lev,lev) in enumerate(zip_longest(root.H[1:], sub_node.H, fillvalue=[])):
                        if lev:
                            j = sum(fd*(2**k) for k,fd in enumerate(sub_node.fds[i:]))
                            if not Lev: Lev = CpH(H=[[] for fork in range(2**(i+1))])  # n forks *=2 per lev
                            if not Lev.H[j]: Lev.H[j] = Cgraph()
                            sum_inder_(Lev.H[j].inder_, lev.H[j].inder_)
                            # or sum_G?
            for Lev in root.H:
                fbval += Lev.val; fbrdn += Lev.rdn
            fbV = fbval/max(1, fbrdn)
            root = root.root

def sum_G(G, g, fmerge=0):  # g is a node in G.node_

    sum_inder_(G.inder_, g.inder_)  # direct node representation
    # if g.uH: sum_H(G.uH, g.uH[1:])  # sum g->G
    if g.H:
        sum_H(G.H[1:], g.H)  # not in sum2graph
    G.L += g.L; G.S += g.S
    if isinstance(g.A, list):
        if G.A:
            G.A[0] += g.A[0]; G.A[1] += g.A[1]
        else: G.A = copy(g.A)
    else: G.A += g.A
    G.val += g.val; G.rdn += g.rdn; G.nval += g.nval
    Y,X,Y0,Yn,X0,Xn = G.box[:]; y,x,y0,yn,x0,xn = g.box[:]
    G.box[:] = [Y+y, X+x, min(X0,x0), max(Xn,xn), min(Y0,y0), max(Yn,yn)]
    if fmerge:
        for node in g.node_:
            if node not in G.node_: G.node_ += [node]
        for link in g.Link_.Q:  # redundant?
            if link not in G.Link_.Q: G.Link_.Q += [link]
        for alt_graph in g.alt_graph_:
            if alt_graph not in G.alt_graph: G.alt_graph_ += [alt_graph]
        if g.alt_Graph:
            if G.alt_Graph: sum_G(G.alt_Graph, g.alt_Graph)
            else:           G.alt_Graph = deepcopy(g.alt_graph)
    else: G.node_ += [g]

