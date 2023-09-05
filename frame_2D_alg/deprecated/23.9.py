def sub_recursion_eval(root, graph_): # eval per fork, same as in comp_slice, still flat aggH, add valt to return?

    Sub_tt = [[[],[]],[[],[]]]  # graph_-wide, maps to root.fback_tt

    for graph in graph_:
        node_ = copy(graph.node_tt)  # still graph.node_
        sub_tt = []  # each new fork adds fback
        frt = [0,0]
        for fder in 0,1:
            if graph.val_Ht[fder][-1] * np.sqrt((len(node_)-1)) > G_aves[fder] * graph.rdn_Ht[fder][-1]:
                graph.rdn_Ht[fder][-1] += 1  # estimate, no node.rdnt[fd]+=1?
                frt[fder] = 1
                sub_tt += [sub_recursion(root, graph, node_, fder)]  # comp_der|rng in graph -> parLayer, sub_Gs
            else:
                sub_tt += [node_]
        for fder in 0,1:
            if frt[fder]:
                for fd in 0, 1:
                    Sub_tt[fder][fd] += [sub_tt[fder][fd]]  # graph_-wide, even if empty, nest to count graphs for higher feedback
                graph.node_tt[fder] = sub_tt[fder]  # else still graph.node_
    for fder in 0,1:
        for fd in 0,1:
            if Sub_tt[fder][fd]:  # new nodes, all terminated, all send feedback
                feedback(root, fder, fd)

def sub_recursion(root, graph, node_, fder):  # rng+: extend G_ per graph, der+: replace G_ with derG_, valt=[0,0]?

    for i in 0,1: root.rdn_Ht[i][0] += 1  # estimate, no node.rdnt[fder] += 1?
    pri_root_tt_ = []
    for node in node_:
        if not fder: node.link_H += [[]]  # add link layer:
        pri_root_tt_ += [node.root_tt]  # merge node roots for new graphs in segment_node_
        node.root_tt = [[[],[]],[[],[]]]  # replace node roots
        for i in 0,1:
            node.val_Ht[i]+=[0]; node.rdn_Ht[i]+=[1]  # new val,rdn layer, accum in comp_G_

    comp_G_(node_, pri_G_=None, f1Q=1, fder=fder)  # cross-comp all nodes in rng
    sub_t = []
    for fd in 0,1:
        graph.rdn_Ht[fd][-1] += 1  # estimate
        pri_root_tt_ = []
        for node in node_:
            pri_root_tt_ += [[node.root_tt]]  # to be transferred to new graphs
            node.root_tt[fder][fd] = []  # fill with new graphs
        sub_G_ = form_graph_(node_, fder, fd, pri_root_tt_)  # cluster sub_graphs via link_H
        sub_recursion_eval(graph, sub_G_)
        for G in sub_G_:
            root.fback_tt[fder][fd] += [[G.aggH, G.val_Ht, G.rdn_Ht]]
        sub_t += [sub_G_]

    return sub_t

# ignore: lateral splicing of initial Ps is not needed: will be spliced in PPs through common forks?
def lat_comp_P(_P,P):  # to splice, no der+

    ave = P_aves[0]
    rn = len(_P.dert_)/ len(P.dert_)

    mtuple,dtuple = comp_ptuple(_P.ptuple[:-1], P.ptuple[:-1], rn)
    _L, L = _P.ptuple[-1], P.ptuple[-1]
    gap = np.hypot((_P.y - P.y), (_P.x, P.x))
    rL = _L - L
    mM = min(_L, L) - ave
    mval = sum(mtuple); dval = sum(dtuple)
    mrdn = 1+(dval>mval); drdn = 1+(1-(dval>mval))  # rdn = Dval/Mval?