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
