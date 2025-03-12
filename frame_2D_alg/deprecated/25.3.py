def cluster_C_(root, rc):  # 0 nest gap from cluster_edge: same derH depth in root and top Gs

    def sum_C(node_):  # sum|subtract and average C-connected nodes

        C = copy_(node_[0]); C.node_= set(node_)  # add root and medoid / exemplar?
        C.M = 0
        sum_G_(node_[1:], G=C)  # no extH, extend_box
        alt_ = [n.altG for n in node_ if n.altG]
        if alt_:
            sum_G_(alt_, G=C.altG)  # no m, M, L in altGs
        k = len(node_)
        for n in (C, C.altG):
            n.Et/=k; n.baseT/=k; n.derTT/=k; n.aRad/=k; n.yx /= k
            norm_H(n.derH, k)
        return C

    def refine_C_(C_):  # refine weights in fuzzy C cluster around N, in root node_| link_
        '''
        - compare mean-node pairs, weigh match by center-node distance (inverse ave similarity),
        - suppress non-max cluster_sum in mean-node pairs: ave * cluster overlap | (dist/max_dist)**2
        - delete weak clusters, recompute cluster_sums of mean-to-node matches / remaining cluster overlap
        '''
        remove_ = []
        for C in C_:
            while True:
                C.M = 0; dM = 0  # pruned nodes and values, or comp all nodes again?
                for N in C.node_:
                    m = sum( base_comp(C,N)[0][0])  # derTT[0][0]
                    if C.altG and N.altG:
                        m += sum( base_comp(C.altG,N.altG)[0][0])
                    N.Ct_ = sorted(N.Ct_, key=lambda ct: ct[1], reverse=True)  # _N.M rdn = n stronger root Cts
                    for i, [_C,_m,_med] in enumerate(N.Ct_):
                        if _C is C:
                            vm = m - ave * (max_med/2 /_med) * (i+1) * (len(C.node_ & _C.node_) / (len(C.node_)+len(_C.node_)))
                            # ave * inverse med deviation (lower ave m) * redundancy * relative node_ overlap between clusters
                            dm = _m-vm; dM += dm
                            _C.M += dm
                            if _C.M > ave: N.Ct_[i][1] = vm
                            else:          N.Ct_.pop(i)
                            break  # CCt update only
                if C.M < ave:
                    for n in C.node_:
                        if C in n.Ct_: n.Ct_.remove(C)
                    remove_ += [C]  # delete weak | redundant cluster
                    break
                if dM > ave: C = sum_C(list(C.node_))  # recompute centroid, or ave * iterations: cost increase?
                else: break
        C_ = [C for C in C_ if C not in remove_]

    # get centroid clusters of top Gs for next cross_comp
    C_t = [[],[]]
    ave = globals()['ave'] * rc  # recursion count
    # cluster top node_| link_:
    for fn, C_,nest,_N_ in zip((1,0), C_t, [root.nnest,root.lnest], [root.node_,root.link_]):
        if not nest: continue
        N_ = [N for N in sorted([N for N in _N_[-1].node_], key=lambda n: n.Et[fn], reverse=True)]
        for N in N_: N.Ct_ = []
        for N in N_:
            med = 1; med_ = [1]; node_,_n_ = set([N]), set([N])
            while med <= max_med and _n_:  # fill init C.node_: _Ns connected to N by <=3 mediation degrees
                n_ = set()
                for _n in _n_:
                    rim_n_ = list(set([n for link in _n.rim for n in link.nodet]))
                    n_ += rim_n_; node_.update(rim_n_); med_ += [med]  # for med-weighted clustering
                med += 1; _n_ = n_  # mediated __Ns
            C = sum_C(list(node_))
            for n, med in zip(node_,med_): n.Ct_ += [[C,0,med]]  # empty m, same n in multiple Ns
            C_+= [C]
        # refine centroid clusters
        refine_C_(C_)
        if len(C_) > ave_L:
            if fn:
                root.node_ += [sum_G_(C_)]; root.nnest += 1
            else:
                root.link_ += [sum_G_(C_)]; root.lnest += 1
        # recursion in root cross_comp

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

def cross_comp(root, fn, rc):  # form agg_Level by breadth-first node_,link_ cross-comp, connect clustering, recursion
    # ave *= recursion count

    N_,L_,Et = comp_node_(root.node_[-1].node_ if fn else root.link_[-1].node_, ave*rc)  # cross-comp top-composition exemplars
    # mval -> lay
    if Val_(Et, Et, ave*(rc+1), fd=0) > 0:  # cluster eval
        derH = [[comb_H_(L_, root, fd=1)]]  # nested mlay
        pL_ = {l for n in N_ for l,_ in get_rim(n, fd=0)}
        if len(pL_) > ave_L:
            cluster_N_(root, pL_, ave*(rc+2), fd=0, rc=rc+2)  # form multiple distance segments, same depth
        # dval -> comp L_ for all dist segments, adds altGs
        if Val_(Et, Et, ave*(rc+2), fd=1) > 0:
            lN_,lL_,dEt = comp_link_(L2N(L_), ave*(rc+2))  # comp root.link_ forms root in alt clustering?
            if Val_(dEt, Et, ave*(rc+3), fd=1) > 0:
                derH[0] += [comb_H_(lL_, root, fd=1)]  # += dlay
                plL_ = {l for n in lN_ for l,_ in get_rim(n,fd=1)}
                if len(plL_) > ave_L:
                    cluster_N_(root, plL_, ave*(rc+4), fd=1, rc=rc+4)
                    # form altGs for cluster_C_, no new links between dist-seg Gs
        root.derH += derH  # feedback
        comb_altG_(root.node_[-1].node_, ave*(rc+4), rc=rc+4)  # comb node contour: altG_ | neg links sum, cross-comp -> CG altG
        cluster_C_(root, rc+5)  # -> mfork G,altG exemplars, +altG surround borrow, root.derH + 1|2 lays, agg++
        # no dfork cluster_C_, no ddfork
        # if val_: lev_G -> agg_H_seq
        return root.node_[-1]

    long_V = 0
    for n in N_:
        for l, _ in n.rim:
            co_nl_ = [nl for nl, _ in
                      (l.nodet[0].nrim + l.nodet[0].rim + l.nodet[1].nrim) + l.nodet[1].rim
                      if comp_angle(l.baseT[2:], nl.baseT[2:])[0] > .3 and nl.L < l.L]
            # shorter p+n links represent likely CC overlap to LCs
            if len(co_nl_) > 3: long_V += val_(n.Et, ave)
    # if long_V > ave:
''' 
 Hierarchical clustering should alternate between two phases: generative via connectivity and compressive via centroid.

 Connectivity clustering terminates at contour alt_Gs, with next-level cross-comp between new core + contour clusters.
 If strongly connected, these clusters may be sub-clustered (classified into) by centroids, with proximity bias. 

 Connectivity clustering is a generative learning phase, forming new derivatives and structured composition levels, 
 and centroid clustering is a compressive phase, reducing multiple similar comparands to a single exemplar. 
 
         else:
            if fi: node.nrim += [(Link,dir)]
            else:  node.nrimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
'''

