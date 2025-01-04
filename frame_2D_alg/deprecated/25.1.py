def sum_G_(node_):
    G = CG()
    for n in node_:
        G.latuple += n.latuple; G.vert += n.vert; G.aRad += n.aRad; G.box = extend_box(G.box, n.box)
        if n.derH: G.derH.add_tree(n.derH, root=G)
        if n.extH: G.extH.add_tree(n.extH)
    return G
