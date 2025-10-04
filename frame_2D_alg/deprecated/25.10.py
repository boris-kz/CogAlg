def form_B__(nG, lG):  # form and trace edge / boundary / background per node:

    for Lg in lG.N_:  # in frame or blob?
        rB_ = {n.root for L in Lg.N_ for n in L.N_ if n.root and n.root.root is not None}  # rdn core Gs, exclude frame
        if rB_:
            rB_ = {(core,rdn) for rdn,core in enumerate(sorted(rB_, key=lambda x:(x.Et[0]/x.Et[2]), reverse=True), start=1)}
            Lg.rB_ = [rB_, np.sum([i.Et for i,_ in rB_], axis=0)]  # reciprocal boundary
    def R(L):
        return L.root if L.root is None or L.root.root is lG else R(L.root)
    for Ng in nG.N_:
        Et, Rdn, B_ = np.zeros(3), 0, []  # core boundary clustering
        LR_ = {R(L) for n in Ng.N_ for L in n.rim}  # lGs for nGs, individual nodes and rims are too weak to bound
        for LR in LR_:
            if LR and LR.rB_:  # not None, eval Lg.B_[1]?
                for core, rdn in LR.rB_[0]:  # map contour rdns to core N:
                    if core is Ng:
                        B_ += [LR]; Et += core.Et; Rdn += rdn
        Ng.B_ = [B_,Et, Rdn]
