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
            
        # dval -> comp dPP_:
        if Val_(Et, Et, ave*2, fd=1) > 0:  # likely not from the same links
            lN_,lL_,dEt = comp_link_(L2N(L_), ave*2)
            if Val_(dEt, Et, ave*3, fd=1) > 0:
                lay += [sum_lay_(lL_, frame)]  # dfork
                lG_ = cluster_PP_(lN_, fd=1) if len(lN_) > ave_L else []
            else:
                lG_=[]  # empty dfork
        else: lG_=[]
        
                    if isinstance(LR.derH[0], list): 1Has a conversation. Original line has a conversation.
                         if len(LR.derH[0])==2: LR.derH[0][1].add_lay(lay)  # direct root only
                         else:                  LR.derH[0] += [lay.copy_(root=LR)]
                     else:  # L.nodet is CL
                         if len(LR.derH) == 2: LR.derH[1].add_lay(lay)  # direct root only
                         else:                 LR.derH += [lay.copy_(root=LR)]
                         
    L.derH = [reduce(lambda Lay, lay: Lay.add_lay(lay), L.derH, CLay(root=L))] # combine derH into single lay
'''

def centroid_M_(m_, M, ave):  # adjust weights on attr matches | diffs, recompute with sum

    _w_ = np.ones(len(m_))  # add cost attrs?
    while True:
        M /= np.sum(_w_)  # mean
        w_ = m_ / min(M, 1/M)  # rational deviations from the mean
        # in range 0:1, or 0:2: w = min(m/M, M/m) + mean(min(m/M, M/m))?
        Dw = np.sum( np.abs(w_-_w_))  # weight update
        m_[:] = (m_ * w_) / np.sum(m_)  # replace in each cycle?
        M = np.sum(m_)  # weighted M update
        if Dw > ave:
            _w_ = w_
        else:
            break
    return w_, M  # no need to return weights?

def cluster_N_(root, L_, ave, fi, rc):  # top-down segment L_ by >ave ratio of L.dists

    nest = root.nnest if fi else root.lnest  # same process for nested link_?

    L_ = sorted(L_, key=lambda x: x.L)  # short links first
    min_dist = 0; Et = root.Et
    while True:
        # each loop forms G_ of contiguous-distance L_ segment
        _L = L_[0]; N_, et = copy(_L.nodet), _L.Et
        for n in [n for l in L_ for n in l.nodet]:
            n.fin = 0
        for i, L in enumerate(L_[1:], start=1):
            rel_dist = L.L/_L.L  # >= 1
            if rel_dist < 1.2 or len(L_[i:]) < ave_L:  # ~= dist Ns or either side of L is weak: continue dist segment
                LV = Val_(et, Et, ave)  # link val
                _G,G = L.nodet  # * surround density: extH (_Ete[0]/ave + Ete[0]/ave) / 2, after cross_comp:
                surr_rV = (sum(_G.derTTe[0]) / (ave*_G.Et[2])) + (sum(G.derTTe[0]) / (ave*G.Et[2])) / 2
                if LV * surr_rV > ave:
                    _L = L; N_ += L.nodet; et += L.Et  # else skip weak link inside segment
            else:
                i -= 1; break  # terminate contiguous-distance segment
        G_ = []
        max_dist = _L.L
        for N in {*N_}:  # cluster current distance segment
            if N.fin: continue  # clustered from prior _N_
            _eN_,node_,link_,et, = [N],[],[], np.zeros(4)
            while _eN_:
                eN_ = []
                for eN in _eN_:  # cluster rim-connected ext Ns, all in root Gt
                    node_+=[eN]; eN.fin = 1  # all rim
                    for L,_ in get_rim(eN, fi=1):  # all +ve
                        if L not in link_:
                            eN_ += [n for n in L.nodet if not n.fin]
                            if L.L < max_dist:
                                link_+=[L]; et+=L.Et
                _eN_ = {*eN_}
            # if not link_: continue  # when the first _L breaks, all other L.L should have longer L and not link_ will be added
            link_ = list({*link_});  Lay = CLay()
            [Lay.add_lay(lay) for lay in sum_H(link_, root, fi=0)]
            derTT = Lay.derTT
            # weigh m_|d_ by similarity to mean m|d, replacing derTT:
            _,m_,M = centroid_M_(derTT[0], np.sum(derTT[0]), ave)
            _,d_,D = centroid_M_(derTT[1], np.sum(derTT[1]), ave)
            et[:2] = M,D; Lay.derTT = np.array([m_,d_])
            # cluster roots:
            if Val_(et, Et, ave) > 0:
                G_ += [sum2graph(root, [list({*node_}),link_, et, Lay], 1, min_dist, max_dist)]
        # longer links:
        L_ = L_[i + 1:]
        if L_: min_dist = max_dist  # next loop connects current-distance clusters via longer links
        # not updated:
        else:
            if G_:
                [comb_altG_(G.altG.node_, ave, rc) for G in G_]
                if fi:
                    root.node_ += [sum_G_(G_)]  # node_ is already nested
                    root.nnest += 1
                else:
                    if root.lnest: root.link_ += [sum_G_(G_)]
                    else: root.link_ = [sum_G_(root.link_), sum_G_(G_)]  # init nesting
                    root.lnest += 1
            break
    # draft:
    if (root.nnest if fi else root.lnest) > nest:  # if nested above
        node_ = root.node_[-1].node_ if fi else root.link_[-1].node_
        for n in node_:
            if Val_(n.Et, n.Et, ave*(rc+4), fi=0) > 0:
                # cross_comp new-G' flat link_:
                cross_comp(n, rc+4, fi=0)

def cross_comp1(root, rc, fi=1):  # recursion count, form agg_Level by breadth-first node_,link_ cross-comp, connect clustering, recursion

    nnest, lnest = root.nnest, root.lnest
    # default root is frame
    N_,L_,Et = comp_node_(root.node_[-1].node_, ave*rc) if fi else comp_link_(L2N(root.link_), ave*rc)  # nested node_ or flat link_

    if Val_(Et, Et, ave*(rc+1), fi=fi) > 0:
        lay = comb_H_(L_, root, fi=0)
        if fi: root.derH += [[lay]]  # [mfork] feedback
        else: root.derH[-1] +=[lay]  # dfork
        pL_ = {l for n in N_ for l,_ in get_rim(n, fi=fi)}
        lEt = np.sum([l.Et for l in pL_], axis=0)
        L = len(pL_)
        if L > ave_L and Val_(lEt, lEt, ave*rc+2) > 0:
            if fi:
                cluster_N_(root, pL_, ave*(rc+2), rc=rc+2)  # form multiple distance segments
                if lEt[0] -  ave*(rc+2)  *  ccoef * lEt[2] > 0:  # shorter links are redundant to LC above: rc+ 1 | LC_link_ / pL_?
                    cluster_C_(root, pL_, rc+3)  # mfork G,altG exemplars, +altG surround borrow, root.derH + 1|2 lays, agg++
                # if CC_V > LC_V: delete root.node_[-2]:LC_, [-1] is CC_?
            else:
                cluster_L_(root, N_, ave*(rc+2), rc=rc+2)  # CC links via llinks, no dist-nesting
                # no cluster_C_ for links, connectivity only
        # recursion:
        lev_Gt = []
        for fi, N_, nest, inest in zip((1,0), (root.node_,root.link_),(root.nnest,root.lnest),(nnest,lnest)):
            if nest > inest:  # nested above
                lev_G = N_[-1]  # cross_comp CGs in link_[-1].node_
                if Val_(lev_G.Et, lev_G.Et, ave*(rc+4), fi=fi) > 0:  # or global Et?
                    cross_comp(root, rc+4, fi=fi)  # this cross_comp is per node_ or link_ so we need to parse different fi here?
                lev_Gt += [lev_G]
            else: lev_Gt+=[[]]
        return lev_Gt

def cross_comp3(root, rc, ifi=0, iL_=[]):  # recursion count, form agg_Level by breadth-first node_,link_ cross-comp, connect clustering, recursion

    fi = not iL_
    N_,L_,Et = comp_node_(root.node_[-1].node_, ave*rc) if (fi or ifi) else comp_link_(L2N(root.link_[-1].node_ if ifi else iL_), ave*rc)   # nested node_ or flat link_

    if N_ and val_(Et, Et, ave*(rc+1), fi) > 0:
        lev_N, lev_L = [],[]
        lay = comb_H_(L_, root, fi=0)
        if fi: root.derH += [[lay]]  # [mfork] feedback
        else: root.derH[-1] +=[lay]  # dfork
        pL_ = {l for n in N_ for l,_ in get_rim(n, fi)}
        lEt = np.sum([l.Et for l in pL_], axis=0)
        # m_fork:
        if val_(lEt, lEt, ave*(rc+2), fi=1, coef=ccoef) > 0:  # or rc += 1?
            if fi:
                lev_N = cluster_N_(root, pL_, ave*(rc+2), rc=rc+2)  # combine distance segments
                if lEt[0] > ave*(rc+3) * lEt[3] * ccoef:  # short links already in LC: rc+ 1 | LC_link_ / pL_?
                    lev_C = cluster_C_(pL_, rc+3)  # mfork G,altG exemplars, +altG surround borrow, root.derH + 1|2 lays, agg++
                else: lev_C = []
                lev_N = comb_Gt(lev_N,lev_C,root)  # if CC_V > LC_V: delete root.node_[-2]:LC_, [-1] is CC_?
                root.node_ += [lev_N]
            else:
                lev_N = cluster_L_(root, N_, ave*(rc+2), rc=rc+2)  # via llinks, no dist-nesting, no cluster_C_
                root.link_ += [lev_N]
            if lev_N:
                if val_(lev_N.Et, lev_N.Et, ave*(rc+4), fi=1, coef=lcoef) > 0:  # or global _Et?
                    # m_fork recursion:
                    nG = cross_comp(root, rc=rc+4, ifi=fi)  # xcomp root.node_[-1]
                    if nG: lev_N = nG  # incr nesting
        # d_fork:
        if val_(lEt, lEt, ave*(rc+2), fi=0, coef=lcoef) > 0:
            # comp L_, d_fork recursion:
            lG = cross_comp(root, rc+4, iL_=L_)
            if lG: lev_L = lG
        # combine:
        lev_G = comb_Gt(lev_N, lev_L, root)  # L derH is already in the root?
        if lev_G:
            if lev_L:
                root.link_ += [lev_L]; root.lnest = lev_L.nnest
            if lev_N:
                root.node_ += [lev_N]; root.nnest = lev_N.nnest
            return lev_G

def comb_Gt(nG,lG, root):
    if nG:
       if lG: Gt = sum_G_([nG,lG], merge=1)  # merge forks
       else:  Gt = copy_(nG); nG.root=root; nG.node=[nG,[]]
    elif lG:  Gt = copy_(lG); nG.root=root; nG.node=[[],lG]
    else: Gt = []
    return Gt

def add_merge_H(H, h, root, rev=0):  # add derHs between level forks

    for i, (Lay,lay) in enumerate(zip_longest(H,h)):  # different len if lay-selective comp
        if lay:
            if isinstance(lay, list):  # merge forks
                for j, fork in zip((1,0), lay):
                    if j: layt = fork.copy_(root=fork.root, rev=rev)  # create
                    else: layt.add_lay(fork,rev=rev)  # merge
                lay = layt
            if Lay:
                if isinstance(Lay,list):  # merge forks
                    for k, fork in zip((1,0), Lay):
                        if k: layt = fork.copy_(root=fork.root, rev=rev)
                        else: layt.add_lay(fork,rev=rev)
                    Lay = layt
                    H[i] = Lay
                Lay.add_lay(lay,rev=rev)
            else:
                H += [lay.copy_(root=root,rev=rev)]
            root.derTTe += lay.derTT; root.Et += lay.Et

def comb_altG_(G_, ave, rc=1):  # combine contour G.altG_ into altG (node_ defined by root=G), for agg+ cross-comp
    # internal and external alts: different decay / distance?
    # background + contour?
    for G in G_:
        if isinstance(G,list): continue
        if G.altG:
            if G.altG.node_:
                G.altG = sum_N_(G.altG.node_)
                G.altG.node_ = [G.altG]  # formality for single-lev_G
                G.altG.root=G; G.altG.fi=0; G.altG.m=0
                if val_(G.altG.Et, G.Et, ave):  # alt D * G rM
                    cross_comp(G.altG, rc, G.altG.node_, fi=1)  # adds nesting
        else:  # sum neg links
            link_,node_,derH, Et = [],[],[], np.zeros(4)
            for link in G.link_:
                if val_(link.Et, G.Et, ave) > 0:  # neg link
                    link_ += [link]  # alts are links | lGs
                    node_ += [n for n in link.nodet if n not in node_]
                    Et += link.Et
            if link_ and val_(Et, G.Et, ave, coef=10) > 0:  # altG-specific coef for sum neg links (skip empty Et)
                altG = CG(root=G, Et=Et, node_=node_, link_=link_, fi=0); altG.m=0  # other attrs are not significant
                altG.derH = sum_H(altG.link_, altG, fi=0)   # sum link derHs
                altG.derTT = np.sum([link.derTT for link in altG.link_],axis=0)
                G.altG = altG

def cross_comp4(root, rc, iN_, fi=1):  # rc: recursion count, form agg_Level by breadth-first node_,link_ cross-comp, clustering, recursion

    N_,L_,Et = comp_node_(iN_, ave*rc) if fi else comp_link_(iN_, ave*rc)  # flat node_ or link_

    if N_ and val_(Et, Et, ave*(rc+1), fi) > 0:  # | np.add(Et[:2]) > (ave+ave_d) * np.multiply(Et[2:])?
        cG,nG,lG = [],[],[]

        lay = comb_H_(L_, root, fi=0)
        if fi: root.derH += [[lay]]  # [mfork] feedback
        else:  root.derH[-1] += [lay]  # dfork feedback
        pL_ = {l for n in N_ for l,_ in get_rim(n, fi)}
        lEt = np.sum([l.Et for l in pL_], axis=0)
        # m_fork:
        if val_(lEt,lEt, ave*(rc+2), 1, clust_w) > 0:  # or rc += 1?
            node_ = []
            if fi:
                cG = cluster_C_(pL_,rc+2)  # exemplar CCs, same derH, select to form partly overlapping LCs:
                if cG:
                    if val_(cG.Et, cG.Et, ave*(rc+3), 1, clust_w) > 0:  # link-cluster CC nodes via short rim Ls:
                        short_L_ = {L for C in cG.node_ for n in C.node_ for L,_ in n.rim if L.L < ave_dist}
                        nG = cluster_N_(cG, short_L_, ave*(rc+3), rc+3) if short_L_ else []
                        node_ = nG.node_ if nG else []
                    else: nG = []
            else:
                nG = cluster_L_(root, N_, ave*(rc+2), rc=rc+2)  # via llinks, no dist-nesting, no cluster_C_
                node_ = nG.node_ if nG else []
            if node_:
                nG.node_ = [copy_(nG)]  # add node_ nesting, nG may be accumulated in recursion:
                if val_(nG.Et, nG.Et, ave*(rc+4), fi=1, coef=loop_w) > 0:  # | global _Et?
                    cross_comp(nG, rc=rc+4, iN_=node_)  # recursive cross_comp N_
        if nG:
            root.cG = cG  # precondition for nG, pref cluster node.root LCs within CC, aligned and same nest as node_?
            root.node_ += [nG]; root.nnest = nG.nnest  # unpack node_[1] and cent_[1] nnest times:
        ''' node_[0].node_: base nodes,
            node_[1].node_[0].node_: next-level nodes,
            node_[1].node_[1].node_[0].node_: next-next level nodes...
            get Cs via node.rC_ or clearer with flat node_[0].cG, no root.cG '''
        # d_fork:
        if val_(lEt,lEt, ave*(rc+2), fi=0, coef=loop_w) > 0:
            L2N(L_)
            lG = sum_N_(L_, fi=0); lG.node_ = [copy_(lG)]
            cross_comp(lG, rc+4, iN_=L_, fi=0)  # recursive cross_comp L_
            root.link_ += [lG]; root.lnest = lG.nnest  # unpack link_[1] lnest times

