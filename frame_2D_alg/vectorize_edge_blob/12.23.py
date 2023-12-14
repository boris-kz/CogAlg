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

