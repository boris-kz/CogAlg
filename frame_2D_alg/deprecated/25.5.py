def agg_level(inputs):  # draft parallel

    frame, H, elevation = inputs
    lev_G = H[elevation]
    cross_comp(lev_G, rc=0, iN_=lev_G.node_, fi=1)  # return combined top composition level, append frame.derH
    if lev_G:
        # feedforward
        if len(H) < elevation+1: H += [lev_G]  # append graph hierarchy
        else: H[elevation+1] = lev_G
        # feedback
        if elevation > 0:
            if np.sum( np.abs(lev_G.aves - lev_G._aves)) > ave:  # filter update value, very rough
                m, d, n, o = lev_G.Et
                k = n * o
                m, d = m/k, d/k
                H[elevation-1].aves = [m, d]
            # else break?

def agg_H_par(focus):  # draft parallel level-updating pipeline

    frame = frame_blobs_root(focus)
    intra_blob_root(frame)
    vect_root(frame)
    if frame.node_:  # converted edges
        G_ = []
        for edge in frame.node_:
            comb_alt_(edge.node_, ave)
            # cluster_C_(edge, ave)  # no cluster_C_ in vect_edge
            G_ += edge.node_  # unpack edges
        frame.node_ = G_
        manager = Manager()
        H = manager.list([CG(node_=G_)])
        maxH = 20
        with Pool(processes=4) as pool:
            pool.map(agg_level, [(frame, H, e) for e in range(maxH)])

        frame.aggH = list(H)  # convert back to list

def cross_comp(root, rc, iN_, fi=1):  # rc: redundancy count; (cross-comp, exemplar selection, clustering), recursion

    N__,L_,Et = comp_node_(iN_,rc) if fi else comp_link_(iN_,rc)  # root.olp is in rc
    if N__:  # CLs if not fi
        nG, n__ = [],[]  # mfork
        for n in [N for N_ in N__ for N in N_]:
            n__ += [n]; n.sel = 0  # for cluster_N_
        if val_(Et, mw=(len(n__)-1)*Lw, aw=rc+loop_w) > 0:  # rc += is local
            E_,eEt = sel_exemplars(n__, rc+loop_w, fi)  # sel=1 for typical sparse nodes
            # call from sel_exemplars?:
            if val_(eEt, mw=(len(E_)-1)*Lw, aw=rc+clust_w) > 0:
                C_,cEt = cluster_C_(root, E_, rc+clust_w)  # refine _N_+_N__ by mutual similarity
                # internal cross_comp C_, recursion?
                if val_(cEt, mw=(len(C_)-1)*Lw, aw=rc+loop_w) > 0:  # refine exemplars by new et
                    S_,sEt = sel_exemplars([n for C in C_ for n in C.node_], rc+loop_w, fi, fC=1)
                else: S_,sEt = [],np.zeros(3)
                if val_(sEt+eEt, mw=(len(S_+E_)-1)*Lw, aw=rc+clust_w) > 0:
                    for rng, N_ in enumerate(N__, start=1):  # bottom-up, cluster via rng exemplars:
                        en_ = [n for n in N_ if n.sel]; eet = np.sum([n.et for n in en_])
                        if val_(eet, mw=(len(en_)-1)*Lw, aw=rc+clust_w*rng) > 0:
                            nG = cluster_N_(root, en_, rc+clust_w*rng, fi, rng)
                    if nG and val_(nG.Et, Et, mw=(len(nG.node_)-1)*Lw, aw=rc+clust_w*rng+loop_w) > 0:
                        nG = cross_comp(nG, rc+clust_w*rng+loop_w, nG.node_)
                        # top-composition, select unpack lower-rng nGs for deeper cross_comp
        lG = []  # dfork
        dval = val_(Et, mw=(len(L_)-1)*Lw, aw=rc+3+clust_w, fi=0)
        if dval > 0:
            if dval > ave:  # recursive derivation forms lH in each node_H level, Len banded?
                lG = cross_comp(sum_N_(L_), rc+loop_w*2, L2N(L_), fi=0)  # comp_link_, no centroids?
            else:  # lower res, dL_?
                lG = cluster_N_(sum_N_(L_),L2N(L_), rc+clust_w*2, fi=0, fnodet=1)  # overlaps the mfork above
            if nG: comb_alt_(nG.node_, rc+clust_w*3)
        if nG or lG:
            root.H += [[nG,lG]]  # current lev
            if nG: add_N(root,nG); add_node_H(root.H, nG.H, root)  # appends derH,H if recursion
            if lG: add_N(root,lG); root.lH += lG.H + [sum_N_(copy(lG.node_), root=lG)]  # lH: H within node_ level
        if nG:
            return nG
        # add_node_H(H, Fg.H, Fg)  # not needed?
