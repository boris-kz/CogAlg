'''                    # convert edge to CG
                    cross_comp(edge, rc=1, iN_=[PP2G(PP)for PP in edge.node_])  # restricted range and recursion, no comp PPd_ and alts?
                    edge_ += [edge]
    # unpack edges
    [nG0,lG0] = edge_[0].H.pop(0)  # lev0 = [PPm_,PPd_], H,derH layered in cross_comp
    H, derH, baseT, derTT = nG0.H, nG0.derH, nG0.baseT, nG0.derTT
    if lG0:  derH, baseT, derTT = add_H(derH, lG0.derH, root=frame), np.add(baseT, lG0.baseT), np.add(derTT, lG0.derTT)
    for edge in edge_:
        H, derH, baseT, derTT = add_GH(H, edge.H, root=frame), add_H(derH, edge.derH, root=frame), np.add(baseT, edge.baseT), np.add(derTT, edge.derTT)
    # no frame2G?
    frame.H, frame.derH, frame.baseT, frame.derTT = H, derH, baseT, derTT
    return frame
'''
