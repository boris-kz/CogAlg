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

from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_
from slice_edge import slice_edge, comp_angle
from comp_slice import comp_slice
from itertools import combinations, zip_longest
from functools import reduce
from copy import deepcopy, copy
import numpy as np

'''
This code is initially for clustering segments within edge: high-gradient blob, but too complex for that.
It's mostly a prototype for open-ended compositional recursion: clustering blobs, graphs of blobs, etc.
-
rng+ fork: incremental-range cross-comp nodes: edge segments at < max distance, cluster if they match. 
der+ fork: incremental-derivation cross-comp links, from node cross-comp, if abs_diff * rel_adjacent_match 
(variance patterns borrow value from co-projected match patterns because their projections cancel-out)
- 
So graphs should be assigned adjacent alt-fork (der+ to rng+) graphs, to which they lend predictive value.
But alt match patterns borrow already borrowed value, which is too tenuous to track, we use average borrowed value.
Clustering criterion within each fork is summed match of >ave vars (<ave vars are not compared and don't add comp costs).
-
Clustering is exclusive per fork,ave, with fork selected per variable | derLay | aggLay 
Fuzzy clustering can only be centroid-based, because overlapping connectivity-based clusters will merge.
Param clustering if MM, compared along derivation sequence, or combinatorial?
-
Summed-graph representation is nested in a dual tree of down-forking elements: node_, and up-forking clusters: root_.
That resembles neurons, with dendritic tree as input and axonal tree as output. 
But these graphs have recursively nested param sets mapping to each level of the trees, which don't exist in neurons.
-
diagrams: 
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/agg_recursion_unfolded.drawio.png
-
notation:
prefix  f denotes flag
postfix t denotes tuple, multiple ts is a nested tuple
prefix  _ denotes prior of two same-name variables, multiple _s for relative precedence
postfix _ denotes array of same-name elements, multiple _s is nested array
capitalized variables are usually summed small-case variables
'''

class CLay(CBase):  # layer of derivation hierarchy
    name = "lay"
    def __init__(l, **kwargs):
        super().__init__()
        l.Et = kwargs.get('Et', np.zeros(4))
        l.root = kwargs.get('root', None)  # higher node or link
        l.node_ = kwargs.get('node_', [])  # concat across fork tree
        l.link_ = kwargs.get('link_', [])
        l.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m_,d_ [M,D,n,o, I,G,A,L], sum across fork tree,
        # add weights for cross-similarity, along with vertical aves, for both m_ and d_?
        # altL = CLay from comp altG
        # i = kwargs.get('i', 0)  # lay index in root.node_, link_, to revise olp
        # i_ = kwargs.get('i_',[])  # priority indices to compare node H by m | link H by d
        # ni = 0  # exemplar in node_, trace in both directions?
    def __bool__(l): return bool(l.node_)

    def copy_(lay, root=None, rev=0, i=None):  # comp direction may be reversed to -1

        if i:  # reuse self
            C = lay; lay = i; C.node_=copy(i.node_); C.link_ = copy(i.link_); C.derTT=np.zeros((2,8)); C.root=root
        else:  # init new C
            C = CLay(root=root, node_=copy(lay.node_), link_=copy(lay.link_))
        C.Et = copy(lay.Et)
        for fd, tt in enumerate(lay.derTT):  # nested array tuples
            C.derTT[fd] += tt * -1 if (rev and fd) else deepcopy(tt)

        if not i: return C

    def add_lay(Lay, lay_, rev=0):  # merge lays, including mlay + dlay

        if not isinstance(lay_,list): lay_ = [lay_]
        for lay in lay_:
            # rev = dir==-1, to sum/subtract numericals in m_ and d_:
            for fd, (F_, f_) in enumerate(zip(Lay.derTT, lay.derTT)):
                F_ += f_ * -1 if (rev and fd ) else f_  # m_|d_
            # concat node_,link_:
            Lay.node_ += [n for n in lay.node_ if n not in Lay.node_]
            Lay.link_ += lay.link_
            Lay.Et += lay.Et
        return Lay

    def comp_lay(_lay, lay, rn, root, dir=1):  # unpack derH trees down to numericals and compare them

        i_ = lay.derTT[1] * rn * dir; _i_ = _lay.derTT[1]  # i_ is ds, scale and direction- normalized
        d_ = _i_ - i_
        a_ = np.abs(i_); _a_ = np.abs(_i_)
        m_ = np.minimum(_a_,a_) / reduce(np.maximum,[_a_,a_,1e-7])  # match = min/max comparands
        m_ *= np.where((_i_<0) != (i_<0), -1,1)  # match is negative if comparands have opposite sign
        derTT = np.array([m_,d_])
        node_ = list(set(_lay.node_+ lay.node_))  # concat
        link_ = _lay.link_ + lay.link_
        M = sum(m_ * w_t[0]); D = sum(d_ * w_t[1])
        Et = np.array([M, D, 8, (_lay.Et[3]+lay.Et[3])/2])  # n compared params = 8
        if root: root.Et += Et

        return CLay(Et=Et, root=root, node_=node_, link_=link_, derTT=derTT)

class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster
    # graph / node
    def __init__(G,  **kwargs):
        super().__init__()
        G.fi = kwargs.get('fi',0)  # or fd_: list of forks forming G, 1 if cluster of Ls | lGs, for feedback only?
        G.root = kwargs.get('root')  # may extend to list in cluster_N_, same nodes may be in multiple dist layers
        G.Et = kwargs.get('Et', np.zeros(4))  # sum all params M,D,n,o
        G.yx = kwargs.get('yx', np.zeros(2))  # init PP.yx = [(y+Y)/2,(x,X)/2], then ave node yx
        G.box = kwargs.get('box', np.array([np.inf,np.inf,-np.inf,-np.inf]))  # y,x,Y,X area: (Y-y)*(X-x)
        G.baseT = kwargs.get('baseT', np.zeros(4))  # I,G,Dy,Dx  # from slice_edge
        G.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m,d / Et,baseT: [M,D,n,o, I,G,A,L], summed across derH lay forks
        G.derTTe = kwargs.get('derTTe', np.zeros((2,8)))  # sum across link.derHs
        G.derH = kwargs.get('derH',[])  # each lay is [m,d]: Clay(Et,node_,link_,derTT), sum|concat links across fork tree
        G.extH = kwargs.get('extH',[])  # sum from rims, single-fork
        G.maxL = kwargs.get('maxL', 0)  # if dist-nested in cluster_N_
        G.aRad = 0  # average distance between graph center and node center
        # alt.altG is empty for now, needs to be more selective
        G.altG = CG(altG=[], fi=0) if kwargs.get('altG') is None else kwargs.get('altG')  # adjacent (contour) gap+overlap alt-fork graphs, converted to CG
        # G.fork_tree: list = z([[]])  # indices in all layers(forks, if no fback merge
        # G.fback_ = []  # node fb buffer, n in fb[-1]
        G.nnest = kwargs.get('nnest',0)  # node_H if > 0, node_[-1] is top G_
        G.lnest = kwargs.get('lnest',0)  # link_H if > 0, link_[-1] is top L_
        G.cG = kwargs.get('cG',[])  # single-lev node_-aligned centroids
        G.node_ = kwargs.get('node_',[])
        G.link_ = kwargs.get('link_',[])  # internal links
        G.rim = kwargs.get('rim',[])  # external links
        G.nrim = kwargs.get('nrim',[])
    def __bool__(G): return bool(G.node_)  # never empty

def copy_(N, root=None):
    C = CG(root=root)

    for name, value in N.__dict__.items():
        val = getattr(N, name)
        if name == '_id' or name == "Ct_": continue  # skip id and Ct_
        elif name == 'derH':
            for lay in N.derH:
                C.derH += [[fork.copy_(root=C) for fork in lay]] if isinstance(N, CG) else [lay.copy_(root=C)]  # CL
        elif name == 'extH':
            C.extH = [lay.copy_(root=C) for lay in N.extH]
        elif isinstance(value,list) or isinstance(value,np.ndarray):
            setattr(C, name, copy(val))  # Et,yx,box, node_,link_,rim_, altG, baseT, derTT
        else:
            setattr(C, name, val)  # numeric or object: fi, root, maxL, aRad, nnest, lnest
    return C

class CL(CBase):  # link or edge, a product of comparison between two nodes or links
    name = "link"
    def __init__(l,  **kwargs):
        super().__init__()
        l.nodet = kwargs.get('nodet',[])  # e_ in kernels, else replaces _node,node: not used in kernels
        l.L = kwargs.get('L',0)  # distance between nodes
        l.Et = kwargs.get('Et', np.zeros(4))
        l.fi = kwargs.get('fi',0)
        l.yx = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet
        l.box = kwargs.get('box', np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        l.baseT = kwargs.get('baseT', np.zeros(4))
        l.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m_,d_ [M,D,n,o, I,G,A,L], sum across derH
        l.derH  = kwargs.get('derH', [])  # list of single-fork CLays
        # add med, rimt, extH in der+
    def __bool__(l): return bool(l.nodet)

ave, avd, arn, aI, aveB, aveR, Lw, ave_dist, int_w, loop_w, clust_w = 10, 10, 1.2, 100, 100, 3, 5, 10, 2, 5, 10  # opportunity costs
wM, wD, wN, wO, wI, wG, wA, wL = 10, 10, 20, 20, 1, 1, 20, 20  # der params higher-scope weights = reversed relative estimated ave?
w_t = np.ones((2,8))  # fb weights per derTT, adjust in agg+

def vect_root(frame, rV=1, ww_t=[]):  # init for agg+:
    if np.any(ww_t):
        global ave, avd, arn, aveB, aveR, Lw, ave_dist, int_w, loop_w, clust_w, wM, wD, wN, wO, wI, wG, wA, wL, w_t
        ave, avd, arn, aveB, aveR, Lw, ave_dist, int_w, loop_w, clust_w = (
        np.array([ave,avd,arn,aveB, aveR,Lw,ave_dist,int_w,loop_w, clust_w]) / rV)  # projected value change
        w_t = np.array( [np.array([wM,wD,wN,wO,wI,wG,wA,wL]), np.array([wM,wD,wN,wO,wI,wG,wA,wL])]) * ww_t  # or dw_ ~= w_/ 2?
        # derTT w_
    blob_ = unpack_blob_(frame)
    frame2G(frame, derH=[CLay(root=frame)], node_=[blob_], root=None)
    edge_ = []  # cluster, unpack
    for blob in blob_:
        if not blob.sign and blob.G > aveB * blob.root.olp:
            edge = slice_edge(blob, rV)
            if edge.G*((len(edge.P_)-1)*Lw) > ave * sum([P.latuple[4] for P in edge.P_]):
                comp_slice(edge, rV, np.array([(*ww_t[0][:2],*ww_t[0][4:]),(*ww_t[0][:2],*ww_t[1][4:])]) if ww_t else [])  # to scale vert
                Et = edge.Et
                if Et[0] *((len(edge.node_)-1)*(edge.rng+1)*Lw) > ave*Et[2]*clust_w:  # eval PP:
                    G_ = [PP2G(PP)for PP in edge.node_ if PP[-1][0] > ave*PP[-1][2]]  # Et, no comp node_,link_,PPd_
                    edge_ += [cluster_edge(G_, frame)]  # 1layer derH, alt: converted adj_blobs of edge blob | alt_P_?
    # unpack edges:
    Lay = [CLay(root=frame), CLay(root=frame)]
    PP_,G_ = [],[]
    for edget in edge_:
        if edget:
            pp_,g_,lay = edget
            [F.add_lay(f) for F,f in zip(Lay,lay)]  # [mfork, dfork]
            PP_+= pp_; G_+= g_
    frame.derH = [Lay]
    frame.node_= [PP_]
    if G_:
        nG = sum_N_(G_); nG.node_ = [copy_(nG)]
        frame.node_ += [nG]
        frame.nnest = 1
    frame.baseT = np.sum([G.baseT for G in PP_ + G_], axis=0)
    frame.derTT = np.sum([G.derTT for G in PP_ + G_], axis=0)

    rave = []  # convert rave for comp_slice
    return frame, rave

def cluster_edge(iG_, frame):  # edge is CG but not a connectivity cluster, just a set of clusters in >ave G blob, unpack by default

    def cluster_PP_(N_, fi):
        G_ = []
        while N_:  # flood fill
            node_,link_, et = [],[], np.zeros(4)
            N = N_.pop(); _eN_ = [N]
            while _eN_:
                eN_ = []
                for eN in _eN_:  # rim-connected ext Ns
                    node_ += [eN]
                    for L,_ in get_rim(eN, fi):  # all +ve, * density: if L.Et[0]/ave_d * sum([n.extH.m * clust_w / ave for n in L.nodet])?
                        if L not in link_:
                            for eN in L.nodet:
                                if eN in N_:
                                    eN_ += [eN]; N_.remove(eN)  # merged
                            link_ += [L]; et += L.Et
                _eN_ = {*eN_}
            if val_(et, et, ave*2, fi=fi) > 0:
                Lay = CLay(); [Lay.add_lay(link.derH[0]) for link in link_]  # single-lay derH
                G_ += [sum2graph(frame, [node_,link_,et, Lay], fi)]
        return G_

    N_,L_,Et = comp_node_(iG_, ave)  # comp PP_
    # mval -> lay:
    if N_ and val_(Et, Et, ave, fi=1) > 0:
        lay = [sum_lay_(L_, frame)]  # [mfork]
        G_ = cluster_PP_(copy(N_), fi=1) if Et[0] * (len(N_)-1)*Lw > ave*Et[2]*clust_w else []

        return [N_,G_,lay]

def val_(Et, _Et, ave, coef=1, fi=1):  # m+d cluster | cross_comp eval, + cross|root _Et projection

    m, d, n, o = Et; _m,_d,_n,_o = _Et  # cross-fork induction of root Et alt, same overlap?

    d_loc = d * (_m - ave * coef * (_n/n))  # diff * co-projected m deviation, no bilateral deviation?
    d_ave = d - avd * ave  # d deviation, ave_d is always relative to ave m

    if fi: val = m + d_ave - d_loc  # match + proj surround val - blocking val, * decay?
    else:  val = d_ave + d_loc  # diff borrow val, generic + specific

    return val - ave * coef * n * o  # simplified: np.add(Et[:2]) > ave * np.multiply(Et[2:])


def comp_node_(_N_, ave, L=0):  # rng+ forms layer of rim and extH per N, appends N_,L_,Et, ~ graph CNN without backprop

    _Gp_ = []  # [G pair + co-positionals], for top-nested Ns, unless cross-nesting comp:
    if L: _N_ = filter(lambda N: len(N.derH)==L, _N_)  # if dist-nested
    for _G, G in combinations(_N_, r=2):  # if max len derH in agg+
        if _G.nnest != G.nnest:
            continue
        _n, n = _G.Et[2], G.Et[2]; rn = _n/n if _n>n else n/_n
        radii = G.aRad + _G.aRad
        dy,dx = np.subtract(_G.yx,G.yx); dist = np.hypot(dy,dx)
        _G.add, G.add = 0, 0
        _Gp_ += [(_G,G, rn, dy,dx, radii, dist)]
    rng = 1
    N_,L_,ET = set(),[],np.zeros(4)
    _Gp_ = sorted(_Gp_, key=lambda x: x[-1])  # sort by dist, closest pairs first
    while True:  # prior vM
        Gp_,Et = [],np.zeros(4)
        for Gp in _Gp_:
            _G,G, rn, dy,dx, radii, dist = Gp
            _nrim = {L.nodet[1] if L.nodet[0] is _G else L.nodet[0] for L,_ in _G.rim}
            nrim = {L.nodet[1] if L.nodet[0] is G else L.nodet[0] for L,_ in G.rim}
            if _nrim & nrim:  # indirectly connected Gs,
                continue     # no direct match priority?
            # dist vs. radii * induction, mainly / extH?
            (_m,_,_n,_),(m,_,n,_) = _G.Et,G.Et
            weighted_max = ave_dist * ((radii/aveR * int_w**3) * (_m/_n + m/n)/2 / (ave*(_n+n)))  # all ratios
            if dist < weighted_max:   # no density, ext V is not complete
                Link = comp_N(_G,G, ave, fi=1, angle=[dy,dx], dist=dist, fshort=dist<weighted_max/2)
                L_ += [Link]  # include -ve links
                if Link.Et[0] > ave * Link.Et[0] * loop_w:
                    N_.update({_G,G}); Et += Link.Et; _G.add,G.add = 1,1
            else:
                Gp_ += [Gp]  # re-evaluate not-compared pairs with one incremented N.M
        ET += Et
        if Et[0] > ave * Et[0] * loop_w:  # current-rng vM
            _Gp_ = [Gp for Gp in Gp_ if Gp[0].add or Gp[1].add]  # one incremented N.M
            rng += 1
        else:  # low projected rng+ vM
            break

    return  list(N_), L_, ET  # flat N__ and L__

def extend_box(_box, box):  # extend box with another box
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box

    return min(y0, _y0), min(x0, _x0), max(yn, _yn), max(xn, _xn)

def base_comp(_N, N, dir=1):  # comp Et, Box, baseT, derTT
    # comp Et:
    _M,_D,_n,_o = _N.Et; M,D,n,o = N.Et
    dn = _n - n; mn = min(_n,n) / max(_n,n)  # or multiplicative for ratios: min * rn?
    rn = _n / n
    o*=rn; do = _o - o; mo = min(_o,o) / max(_o,o)
    M*=rn; dM = _M - M; mM = min(_M,M) / max(_M,M)
    D*=rn; dD = _D - D; mD = min(_D,D) / max(_D,D)
    # comp baseT:
    _I,_G,_Dy,_Dx = _N.baseT; I,G,Dy,Dx = N.baseT  # I, G|D, angle
    I*=rn; dI = _I - I; mI = abs(dI) / aI
    G*=rn; dG = _G - G; mG = min(_G,G) / max(_G,G)
    mA, dA = comp_angle((_Dy,_Dx),(Dy*rn,Dx*rn))
    if isinstance(N, CL):  # dimension is distance
        _L,L = _N.L, N.L   # not cumulative
        mL,dL = min(_L,L)/ max(_L,L), _L - L
    else:  # dimension is box area
        _y0,_x0,_yn,_xn =_N.box; _A = (_yn-_y0) * (_xn-_x0)
        y0, x0, yn, xn = N.box;   A = (yn - y0) * (xn - x0)
        mL, dL = min(_A,A)/ max(_A,A), _A - A
        # mA, dA
    _m_,_d_ = np.array([mM,mD,mn,mo,mI,mG,mA,mL]), np.array([dM,dD,dn,do,dI,dG,dA,dL])
    # comp derTT:
    _i_ = _N.derTT[1]; i_ = N.derTT[1] * rn  # normalize by compared accum span
    d_ = (_i_ - i_ * dir)  # np.arrays
    _a_,a_ = np.abs(_i_),np.abs(i_)
    m_ = np.divide( np.minimum(_a_,a_), reduce(np.maximum, [_a_,a_,1e-7]))  # rms
    m_ *= np.where((_i_<0) != (i_<0), -1,1)  # match is negative if comparands have opposite sign

    # each [M,D,n,o, I,G,A,L]:
    return [m_+_m_, d_+_d_], rn

def comp_N(_N,N, ave, fi, angle=None, dist=None, dir=1, fshort=1):  # compare links, relative N direction = 1|-1, no need for angle, dist?
    dderH = []

    [m_,d_], rn = base_comp(_N, N, dir)
    baseT = np.array([(_N.baseT[0]+N.baseT[0])/2, (_N.baseT[1]+N.baseT[1])/2, *angle])  # link M,D,A
    derTT = np.array([m_, d_])
    M = np.sum(m_* w_t[0]); D = np.sum(np.abs(d_* w_t[1]))  # feedback-weighted sum
    Et = np.array([M,D, 8, (_N.Et[3]+N.Et[3]) /2])  # n comp vars, inherited olp
    _y,_x = _N.yx
    y, x = N.yx
    Link = CL(nodet=[_N,N], baseT=baseT, derTT=derTT, yx=np.add(_N.yx,N.yx)/2, L=dist, box=np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)]))
    # spec / lay:
    if fshort and M > ave and (len(N.derH) > 2 or isinstance(N,CL)):  # else derH is redundant to dext,vert
        dderH = comp_H(_N.derH, N.derH, rn, Link, Et, fi)  # comp shared layers, if any
        # spec/ comp_node_(node_|link_)
    Link.derH = [CLay(root=Link, Et=Et, node_=[_N,N],link_=[Link], derTT=copy(derTT)), *dderH]
    for lay in dderH: derTT += lay.derTT
    # spec / alt:
    if fi and _N.altG and N.altG:
        et = _N.altG.Et + N.altG.Et  # comb val
        if val_(et, et, ave*2, fi=0) > 0:  # eval Ds
            Link.altL = comp_N(_N.altG, N.altG, ave*2, fi=1, angle=angle)
            Et += Link.altL.Et
    Link.Et = Et
    if Et[0] > ave * Et[2]:  # | both forks: np.add(Et[:2]) > ave * np.multiply(Et[2:])
        for rev, node in zip((0,1), (N,_N)):  # reverse Link direction for
            if fi: node.rim += [(Link,dir)]
            else:  node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            add_H(node.extH, Link.derH, root=node, rev=rev, fi=0)
            node.Et += Et
    return Link

def get_rim(N,fi): return N.rim if fi else N.rimt[0] + N.rimt[1]  # add nesting in cluster_N_?

def sum2graph(root, grapht, fi, minL=0, maxL=None):  # sum node and link params into graph, aggH in agg+ or player in sub+

    node_, link_, Et, mfork = grapht  # Et and mfork are summed from link_
    n0=node_[0]
    graph = CG(
        fi=fi, Et=Et+n0.Et*int_w, box=n0.box, baseT=copy(n0.baseT), derTT=mfork.derTT, root=root, node_=[],link_=link_, maxL=maxL, nnest=root.nnest,
        derH = [[mfork]])  # higher layers are added by feedback, dfork added from comp_link_:
    for L in link_:
        L.root = graph  # reassign when L is node
        if not fi:  # add mfork as link.nodet(CL).root dfork
            LR_ = set([n.root for n in L.nodet if isinstance(n.root,CG)]) # skip frame, empty roots
            if LR_:
                dfork = reduce(lambda F,f: F.add_lay(f), L.derH, CLay())  # combine lL.derH
                for LR in LR_:  # lay0+= dfork
                    if len(LR.derH[0])==2: LR.derH[0][1].add_lay(dfork)  # direct root only
                    else:                  LR.derH[0] += [dfork.copy_(root=LR)]  # init by another node
                    LR.derTT += dfork.derTT
    N_, yx_ = [],[]
    for i, N in enumerate(node_):
        fc = 0
        if minL:  # max,min L.dist in graph.link_, inclusive, = lower-layer exclusive maxL, if G was dist-nested in cluster_N_
            while N.root.maxL and N.root is not graph and (minL != N.root.maxL):  # maxL=0 in edge|frame, not fd
                if N.root is graph:
                    fc=1; break  # graph was assigned as root via prior N
                else: N = N.root  # cluster prior-dist graphs vs nodes
        if fc: continue  # N.root was clustered in prior loop
        else: N_ += [N]  # roots if minL
        N.root = graph
        yx_ += [N.yx]
        if i:
            graph.Et+=N.Et*int_w; graph.baseT+=N.baseT; graph.box=extend_box(graph.box,N.box)
            # not in CL
    graph.node_= N_  # nodes or roots, link_ is still current-dist links only?
    yx = np.mean(yx_, axis=0)
    dy_,dx_ = (graph.yx - yx_).T; dist_ = np.hypot(dy_,dx_)
    graph.aRad = dist_.mean()  # ave distance from graph center to node centers
    graph.yx = yx
    if not fi:  # dgraph, no mGs / dG for now  # and val_(Et, _Et=root.Et) > 0:
        altG = []  # mGs overlapping dG
        for L in node_:
            for n in L.nodet:  # map root mG
                mG = n.root
                if isinstance(mG, CG) and mG not in altG:  # root is not frame
                    mG.altG.node_ += [graph]  # cross-comp|sum complete altG before next agg+ cross-comp, multi-layered?
                    altG += [mG]
    return graph

def sum_lay_(link_, root):
    lay0 = CLay(root=root)
    for link in link_:
        lay0.add_lay(link.derH[0]); root.derTTe += link.derH[0].derTT
    return lay0

def comb_H_(L_, root, fi):
    derH = sum_H(L_,root,fi=fi)
    Lay = CLay(root=root)
    for lay in derH:
        Lay.add_lay(lay); root.derTTe += lay.derTT
    return Lay

def sum_H(Q, root, rev=0, fi=1):  # sum derH in link_|node_
    DerH = []
    for e in Q: add_H(DerH, e.derH, root, rev, fi)
    return DerH

def add_H(H, h, root, rev=0, fi=1):  # add fork L.derHs

    for Lay,lay in zip_longest(H,h):  # different len if lay-selective comp
        if lay:
            if fi:  # two-fork lays
                if Lay:
                    for Fork,fork in zip_longest(Lay,lay):
                        if fork:
                            if Fork: Fork.add_lay(fork,rev=rev)
                            else:    Lay += [fork.copy_(root=root)]
                else:
                    Lay = []
                    for fork in lay:
                        Lay += [fork.copy_(root=root,rev=rev)]
                        root.derTT += fork.derTT; root.Et += fork.Et
                    H += [Lay]
            else:  # one-fork lays
                if Lay: Lay.add_lay(lay,rev=rev)
                else:   H += [lay.copy_(root=root,rev=rev)]
                root.derTTe += lay.derTT; root.Et += lay.Et

def comp_H(H,h, rn, root, Et, fi):  # one-fork derH if not fi, else two-fork derH

    derH = []
    for _lay,lay in zip_longest(H,h):  # different len if lay-selective comp
        if _lay and lay:
            if fi:  # two-fork lays
                dLay = []
                for _fork,fork in zip_longest(_lay,lay):
                    if _fork and fork:
                        dlay = _fork.comp_lay(fork, rn,root=root)
                        if dLay: dLay.add_lay(dlay)  # sum ds between input forks
                        else:    dLay = dlay
            else:  # one-fork lays
                 dLay = _lay.comp_lay(lay, rn, root=root)
            Et += dLay.Et
            derH += [dLay]
    return derH

def sum_N_(node_, root_G=None, root=None):  # form G

    fi = isinstance(node_[0],CG)
    if root_G: G = root_G
    else:
        g = node_.pop(0)
        G = copy_(g); G.node_=[g]; G.fi=fi; G.root=root  # no link_?
    for n in node_:
        add_N(G,n)
    if not fi:
        G.derH = [[lay] for lay in G.derH]  # nest
    return G

def add_N(N,n, fi=1, root=None):

    N.baseT+=n.baseT; N.derTT+=n.derTT; N.Et+=n.Et; N.yx+=n.yx; N.box=extend_box(N.box, n.box)
    if hasattr(n,'derTTe'):
        N.derTTe += n.derTTe; N.aRad += n.aRad
        if n.extH:
            add_H(N.extH, n.extH, root=N, fi=0)
        if fi:  # no H in CL?
            for i, (Lev,lev) in zip_longest(N.H,n.H):
                if lev:
                    if Lev:  # cG,nG,lG
                        for G,g in zip(Lev,lev): add_N(G,g)
                    else: N.H += [lev]
                    if root:
                        for g in lev: add_N(root,g)
        if n.derH:
            add_H(N.derH, n.derH, root=N, fi=fi)

def frame2G(G, **kwargs):
    blob2G(G, **kwargs)
    G.derH = kwargs.get('derH', [CLay(root=G, Et=np.zeros(4), derTT=[], node_=[],link_ =[])])
    G.Et = kwargs.get('Et', np.zeros(4))
    G.node_ = kwargs.get('node_', [])

def blob2G(G, **kwargs):
    # node_, Et stays the same:
    G.fi = 1  # fi=0 if cluster Ls|lGs
    G.root = kwargs.get('root')  # may extend to list in cluster_N_, same nodes may be in multiple dist layers
    G.link_ = kwargs.get('link_',[])
    G.nnest = kwargs.get('nnest',0)  # node_H if > 0, node_[-1] is top G_
    G.lnest = kwargs.get('lnest',0)  # link_H if > 0, link_[-1] is top L_
    G.derH = []  # sum from nodes, then append from feedback, maps to node_tree
    G.extH = []  # sum from rims
    G.baseT = np.zeros(4)  # I,G,Dy,Dx
    G.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m_,d_ base params
    G.derTTe = kwargs.get('derTTe', np.zeros((2,8)))
    G.box = kwargs.get('box', np.array([np.inf,np.inf,-np.inf,-np.inf]))  # y0,x0,yn,xn
    G.yx = kwargs.get('yx', np.zeros(2))  # init PP.yx = (y+Y)/2,(x+X)/2, then ave node yx
    G.rim = []  # flat links of any rng, may be nested in clustering
    G.maxL = 0  # nesting in nodes
    G.aRad = 0  # average distance between graph center and node center
    G.altG = []  # or altG? adjacent (contour) gap+overlap alt-fork graphs, converted to CG
    return G

def PP2G(PP):
    root, P_, link_, vert, latuple, A, S, box, yx, Et = PP

    baseT = np.array((*latuple[:2], *latuple[-1]))  # I,G,Dy,Dx
    [mM,mD,mI,mG,mA,mL], [dM,dD,dI,dG,dA,dL] = vert
    derTT = np.array([np.array([mM,mD,mL,0,mI,mG,mA,mL]), np.array([dM,dD,dL,0,dI,dG,dA,dL])])
    y,x,Y,X = box; dy,dx = Y-y,X-x
    # A = (dy,dx); L = np.hypot(dy,dx)
    G = CG(root=root, fi=1, Et=Et, node_=P_, link_=[], baseT=baseT, derTT=derTT, box=box, yx=yx, aRad=np.hypot(dy/2, dx/2),
           derH=[[CLay(node_=P_,link_=link_, derTT=deepcopy(derTT)), CLay()]])  # empty dfork
    return G

if __name__ == "__main__":
   # image_file = './images/raccoon_eye.jpeg'
    image_file = './images/toucan_small.jpg'
    image = imread(image_file)
    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vect_root(frame)