def sub_recursion_eval(root, graph_):  # eval per fork, same as in comp_slice, still flat aggH, add valt to return?

    Sub_tt = [[[],[]],[[],[]]]  # graph_-wide, maps to root.fback_tt

    for graph in graph_:
        node_ = copy(graph.node_tt)  # still graph.node_
        sub_tt = []  # each new fork adds fback
        fr = 0
        for fder in 0,1:
            if graph.val_Ht[fder][-1] * np.sqrt(len(node_)-1) if node_ else 0 > G_aves[fder] * graph.rdn_Ht[fder][-1]:
                graph.rdn_Ht[fder][-1] += 1  # estimate, no node.rdnt[fd]+=1?
                fr = 1
                sub_tt += [sub_recursion(root, graph, node_, fder)]  # comp_der|rng in graph -> parLayer, sub_Gs
            else:
                sub_tt += [node_]
        for fder in 0,1:
            for fd in 0,1:
                Sub_tt[fder][fd] += [sub_tt[fder][fd]]  # graph_-wide, even if empty, to count graphs for higher feedback
        if fr:
            graph.node_tt = sub_tt  # else still graph.node_
    for fder in 0,1:
        for fd in 0,1:
            if Sub_tt[fder][fd]:  # new nodes, all terminated, all send feedback
                feedback(root, fder, fd)
