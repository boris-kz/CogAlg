def cluster_C_(root, rc):  # 0 nest gap from cluster_edge: same derH depth in root and top Gs

    def sum_C(dnode_, C=None):  # sum|subtract and average C-connected nodes

        if C is None:
            C = copy_(dnode_[0]); C.node_= copy(dnode_); dnode_.pop(0); C.fin = 1
            sign = 1  # add if new, else subtract
            C.M,C.L = 0,0  # centroid setattr
        else:
            sign = 0; C.node_ = [n for n in C.node_ if n.fin]  # not in -ve dnode_, may add +ve later

        sum_G_(dnode_, sign, fc=1, G=C)  # no extH, extend_box
        alt_ = [n.altG for n in dnode_ if n.altG]
        if alt_: sum_G_(alt_, sign, fc=0, G=C.altG)  # no m, M, L in altGs
        k = len(dnode_) + 1
        # get averages:
        for n in (C, C.altG):
            n.Et/=k; n.baseT/=k; n.derTT/=k; n.aRad/=k; n.yx /= k
            norm_H(n.derH, k)
        C.box = reduce(extend_box, (n.box for n in C.node_))

        return C

    def centroid_cluster(N, N_, C_, root):  # form and refine fully fuzzy C cluster around N, in root node_|link_
        # init:
        N.fin = 1; CN_ = [N]
        for n in N_:
            if not hasattr(n,'fin') or n.fin or n is N: continue  # in other C or in C.node_, or not in root
            radii = N.aRad + n.aRad
            dy, dx = np.subtract(N.yx, n.yx)
            dist = np.hypot(dy, dx)
            if dist < max_dist * ((radii * icoef**3) * (val_(N.Et,ave)+ val_(n.Et,ave))):
                n.fin = 1; CN_ += [n]
        # refine:
        C = sum_C(CN_)  # C.node_, add proximity bias for both match and overlap?
        while True:
            dN_, M, dM = [], 0, 0  # pruned nodes and values, or comp all nodes again?
            for _N in C.node_:
                m = sum( base_comp(C,_N)[0][0])  # derTT[0][0]
                if C.altG and _N.altG: m += sum( base_comp(C.altG,_N.altG)[0][0])  # or Et if proximity-weighted overlap?
                vm = m - ave
                if vm > 0:
                    M += m; dM += m - _N.m; _N.m = m  # adjust kept _N.m
                else:  # remove _N from C
                    _N.fin=0; _N.m=0; dN_+=[_N]; dM += -vm  # dM += abs m deviation
            if dM > ave and M > ave:  # loop update, break if low C reforming value
                if dN_:
                    C = sum_C(list(set(dN_)),C)  # subtract dN_ from C
                C.M = M  # with changes in kept nodes
            else:  # break
                if C.M > ave * 10:  # add proximity-weighted overlap?
                    for n in C.node_: n.root = C
                    C_ += [C]; C.root = root  # centroid cluster
                else:
                    for n in C.node_:  # unpack C.node_, including N
                        n.m = 0; n.fin = 0
                break

    C_t = [[],[]]  # concat exemplar/centroid nodes across top Gs for global frame cross_comp
    ave = globals()['ave'] * rc  # recursion count
    # Ccluster top node_|link_:
    for fn, C_,nest,_N_ in zip((1,0), C_t, [root.nnest,root.lnest], [root.node_,root.link_]):
        if not nest: continue
        N_ = [N for N in sorted([N for N in _N_[-1].node_], key=lambda n: n.Et[fn], reverse=True)]
        for N in N_:
            N.sign, N.m, N.fin = 1, 0, 0  # C update sign, inclusion m, inclusion flag
        for N in N_:
            if not N.fin:  # not in prior C
                if Val_(N.Et, root.Et, ave, coef=10) > 0:  # cross-similar in G
                    centroid_cluster(N,N_, C_, root)  # form centroid cluster around N, C_ +=[C]
                else:
                    break  # the rest of N_ is lower-M
        if len(C_) > ave_L:
            if fn:
                root.node_ += [sum_G_(C_)]; root.nnest += 1
            else:
                root.link_ += [sum_G_(C_)]; root.lnest += 1
            if not root.root:  # frame
                cross_comp(root, fn, rc+1)  # append derH, cluster_N_([root.node_,root.link_][fn][-1])

    # get centroid clusters of top Gs for next cross_comp
    C_t = [[],[]]
    ave = globals()['ave'] * rc  # recursion count
    # cluster top node_| link_:
    for fn, C_,nest,_N_ in zip((1,0), C_t, [root.nnest,root.lnest], [root.node_,root.link_]):
        if not nest: continue
        N_ = [N for N in sorted([N for N in _N_[-1].node_], key=lambda n: n.Et[fn], reverse=True)]
        for N in N_: N.Ct_ = []
        for N in N_:
            node_ = [N]; dist_ = [0]  # C node_
            for _N in N_:
                if N is _N: continue
                if Val_(N.Et, root.Et, ave, coef=1) < 0:  # the rest of N_ is lower-M
                    break
                dy,dx = np.subtract(_N.yx,N.yx); dist = np.hypot(dy,dx)
                if dist < max_dist:  # close enough to compare
                    node_ += [_N]; dist_ += [dist]
            C = sum_C(node_)
            for n, dist in zip(node_,dist_): n.Ct_ += [[C,0,dist]]  # empty m, same n in multiple Ns
            C_+= [C]
        refine_C_(C_)  # refine centroid clusters

        if _C is C:
            vm = m - ave * (max_dist / 2 / _dist) * (i + 1) * (_dist / max_dist) ** 2  # disc overlap
            # ave * inverse dist deviation (lower ave m) * redundancy * relative overlap between clusters


def add_lay(Lay, lay_, rev=0, fc=0):  # merge lays, including mlay + dlay

        if not isinstance(lay_,list): lay_ = [lay_]
        for lay in lay_:
            # rev = dir==-1, to sum/subtract numericals in m_ and d_:
            for fd, (F_, f_) in enumerate(zip(Lay.derTT, lay.derTT)):
                F_ += f_ * -1 if rev and (fd or fc) else f_  # m_|d_
            # concat node_,link_:
            Lay.node_ += [n for n in lay.node_ if n not in Lay.node_]
            Lay.link_ += lay.link_
            et = lay.Et * -1 if rev and fc else lay.Et
            Lay.Et += et
        return Lay
