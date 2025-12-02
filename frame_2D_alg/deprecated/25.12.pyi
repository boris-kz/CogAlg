def vect_edge(tile, rV=1, wTTf=[]):  # PP_ cross_comp and floodfill to init focal frame graph, no recursion:

    if np.any(wTTf):
        global ave, avd, arn, aveB, aveR, Lw, adist, amed, intw, compw, centw, contw, wM,wD,wc, wI,wG,wa, wL,wS,wA
        ave, avd, arn, aveB, aveR, Lw, adist, amed, intw, compw, centw, contw = (
            np.array([ave,avd, arn,aveB,aveR, Lw, adist, amed, intw, compw, centw, contw]) / rV)  # projected value change
        wTTf = np.multiply([[wM,wD,wc, wI,wG,wa, wL,wS,wA]], wTTf)  # or dw_ ~= w_/ 2?
    Edge_ = []
    for blob in tile.N_:
        if not blob.sign and blob.G > aveB:
            edge = slice_edge(blob, rV)
            if edge.G * ((len(edge.P_)-1)*Lw) > ave * sum([P.latT[4] for P in edge.P_]):
                PPm_ = comp_slice(edge, rV, wTTf)
                Edge = sum2G([PP2N(PPm) for PPm in PPm_], rc=1, root=None)
                if edge.link_:
                    L_,B_ = [],[]; Gd_ = [PP2N(PPd)for PPd in edge.link_]
                    [L_.append(Gd) if Gd.m > ave else B_.append(Gd) if Gd.d > avd else None for Gd in Gd_]
                    Lt = add_T_(B_,rc=2,root=Edge, nF='Lt'); Edge.L_=L_;
                    Bt = add_T_(B_,rc=2,root=Edge, nF='Bt'); Edge.B_=B_
                    form_B__(Edge)  # add Edge.Bt
                    if val_(Edge.dTT,3, mw=(len(PPm_)-1) *Lw) > 0:
                        trace_edge(Edge,3)  # cluster complemented G x G.B_, ?Edge.N_=G_, skip up
                        if B_ and val_(Bt.dTT,4, fi=0, mw=(len(B_)-1) *Lw) > 0:
                            Bt.N_ = B_; trace_edge(Bt,4) # ?Bt.N_= bG_
                            # for cross_comp in frame_H?
                Edge_ += [Edge]  # default?
    if Edge_:
        return sum2G(Edge_,2,None)

