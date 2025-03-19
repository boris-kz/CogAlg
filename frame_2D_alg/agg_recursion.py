import numpy as np
from copy import copy, deepcopy
from functools import reduce
from itertools import zip_longest
from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, intra_blob_root, imread
from slice_edge import comp_angle
from vect_edge import L2N, base_comp, comp_N, sum_G_, comb_H_, sum_H, copy_, comp_node_, sum2graph, get_rim, CG,CL,CLay, vect_root, val_
'''
notation:
prefix f: flag
prefix _: prior of two same-name variables, multiple _s for relative precedence
postfix _: array of same-name elements, multiple _s is nested array
postfix t: tuple, multiple ts is a nested tuple
capitalized vars are summed small-case vars 

Current code is processing primary data, starting with images
Each agg+ cycle forms complemented graphs in cluster_N_ and refines them in cluster_C_: 
cross_comp -> cluster_N_ -> cluster_C -> cross_comp.., with incremental graph composition per cycle

Ultimate criterion is lateral match, with projecting sub-criteria to add distant | aggregate lateral match
If a property is found to be independently predictive its match is defined as min comparands: their shared quantity.
Else match is an inverted deviation of miss: instability of that property. 

After computing projected match in forward pass, the backprop will adjust filters to maximize next match. 
That includes coordinate filters, which select new input in current frame of reference

The process may start from arithmetic: inverse ops in cross-comp and direct ops in clustering, for pairwise and group compression. 
But there is a huge number of possible variations, so it seems a lot easier to design meaningful initial code manually.

Meta-code will generate/compress base code by process cross-comp (tracing function calls), and clustering by evaluated code blocks.
Meta-feedback must combine code compression and data compression values: higher-level match is still the ultimate criterion.

Code-coordinate filters may extend base code by cross-projecting and combining patterns found in the original base code
(which may include extending eval function with new match-projecting derivatives) 
Similar to cross-projection by data-coordinate filters, described in "imagination, planning, action" section of part 3 in Readme.
'''
ave, ave_L, max_med, icoef, lcoef, ccoef, ave_dist, med_cost = 5, 2, 3, .5, 3, 10, 2, 2

def cross_comp(root, rc, iL_=[]):  # recursion count, form agg_Level by breadth-first node_,link_ cross-comp, connect clustering, recursion

    fi = not iL_
    N_,L_,Et = comp_node_(root.node_[-1].node_, ave*rc) if fi else comp_link_(L2N(iL_), ave*rc)   # nested node_ or flat link_

    if val_(Et, Et, ave*(rc+1), fi) > 0:
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
                if lEt[0] > ave*(rc+3) * lEt[3] * ccoef:  # shorter links are redundant to LC above: rc+ 1 | LC_link_ / pL_?
                    lev_C = cluster_C_(pL_, rc+3)  # mfork G,altG exemplars, +altG surround borrow, root.derH + 1|2 lays, agg++
                    lev_N = sum_G_([lev_N,lev_C])  # two nodes?
                # if CC_V > LC_V: delete root.node_[-2]:LC_, [-1] is CC_?
            else:
                lev_N = cluster_L_(root, N_, ave*(rc+2), rc=rc+2)  # via llinks, no dist-nesting, no cluster_C_
            if lev_N:
                if val_(lev_N.Et, lev_N.Et, ave*(rc+4), fi=1, coef=lcoef) > 0:  # or global _Et?
                    # m_fork recursion:
                    nG = cross_comp(root, rc+4)  # xcomp root.node_[-1]
                    if nG: lev_N = nG[0]  # incr nesting
        # d_fork:
        if val_(lEt, lEt, ave*(rc+2), fi=0, coef=lcoef) > 0:
            # comp L_, d_fork recursion:
            lG = cross_comp(root, rc+4, iL_=L_)
            if lG: lev_L = lG[0]
        lev_G = []
        if lev_L:
            root.link_ += [lev_L]; root.lnest = lev_L.nnest
        if lev_N:
            root.node_ += [lev_N]; root.nnest = lev_N.nnest
            if lev_L: lev_G = sum_G_([lev_N,lev_L], merge=1)  # merge forks
            else:     lev_G = lev_N.copy_(root=root, node=[lev_N,[]])
        elif lev_L:   lev_G = lev_L.copy_(root=root, node=[[],lev_L])

        if lev_G:
            return lev_G

def comp_link_(iL_, ave):  # comp CLs via directional node-mediated link tracing: der+'rng+ in root.link_ rim_t node rims

    fi = isinstance(iL_[0].nodet[0], CG)
    for L in iL_:
        # init mL_t: bilateral mediated Ls per L:
        for rev, N, mL_ in zip((0,1), L.nodet, L.mL_t):
            for _L,_rev in N.rim if fi else N.rimt[0]+N.rimt[1]:
                if _L is not L and _L in iL_:  # nodet-mediated
                    if L.Et[0] > ave * L.Et[3] * lcoef:  # loop coef
                        mL_ += [(_L, rev ^_rev)]  # direction of L relative to _L
    _L_, out_L_, LL_, ET = iL_,set(),[],np.zeros(4)  # out_L_: positive subset of iL_, Et = np.zeros(4)?
    med = 1
    while True:  # xcomp _L_
        L_, Et = set(), np.zeros(4)
        for L in _L_:
            for mL_ in L.mL_t:
                for _L, rev in mL_:  # rev is relative to L
                    dy,dx = np.subtract(_L.yx,L.yx)
                    Link = comp_N(_L,L, ave, fi=0, angle=[dy,dx], dist=np.hypot(dy,dx), dir = -1 if rev else 1)  # d = -d if L is reversed relative to _L
                    Link.med = med
                    LL_ += [Link]  # include -ves, link order: nodet < L < rimt, mN.rim || L
                    if Link.Et[0] > ave * Link.Et[3] * lcoef:  # link induction
                        out_L_.update({_L,L}); L_.update({_L,L}); Et += Link.Et
        ET += Et
        if not any(L_): break
        # extend mL_t per last medL:
        if Et[0] > ave * Et[3] * lcoef + med*med_cost:  # project prior-loop value, med adds fixed cost
            _L_, ext_Et = set(), np.zeros(4)
            for L in L_:
                mL_t, lEt = [set(),set()], np.zeros(4)  # __Ls per L
                for mL_,_mL_ in zip(mL_t, L.mL_t):
                    for _L, rev in _mL_:
                        for _rev, N in zip((0,1), _L.nodet):
                            rim = N.rim if fi else N.rimt
                            if len(rim) == med:  # append in comp loop
                                for __L,__rev in rim if fi else rim[0]+rim[1]:
                                    if __L in L.visited_ or __L not in iL_: continue
                                    L.visited_ += [__L]; __L.visited_ += [L]
                                    if __L.Et[0] > ave * __L.Et[0] * lcoef:  # loop coef?
                                        mL_.add((__L, rev ^_rev ^__rev))  # combine reversals: 2 * 2 mLs, but 1st 2 are pre-combined
                                        lEt += __L.Et
                if lEt[0] > ave * lEt[3]:  # L'rng+, vs L'comp above, add coef
                    L.mL_t = mL_t; _L_.add(L); ext_Et += lEt
            # refine eval by extension D:
            if ext_Et[0] > ave * ext_Et[0] * lcoef + med*med_cost:
                med +=1
            else: break
        else: break
    return out_L_, LL_, ET

''' 
 Connectivity clustering (LC) by short links (less interference), with comp_N forming new derLay, partly overlapping:
 Centroid clustering (CC) by any links, regardless of local structure, long links form only match via base_comp?
 
 LC min distance is more restrictive than in cross-comp, due to density eval and optional use of resulting links in CC.
 LC terminates at contour alt_Gs, with next-level cross-comp between new core+contour clusters.

 LC is mainly generative: complexity of new derivatives and structured composition levels is greater than compression by LC 
 CC is strictly compressive, by similarity, no new diff representation. 
'''
def cluster_N_(root, L_, ave, rc):  # top-down segment L_ by >ave ratio of L.dists

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
                LV = val_(et, Et, ave)  # link val
                # add surround density term for clustering: extH (_Ete[0]/ave +Ete[0]/ave) /2:
                _G,G = L.nodet; surr_V = (sum(_G.derTTe[0])+ sum(G.derTTe[0])/2) / ave*G.Et[2]
                if LV * surr_V > ave:
                    _L = L; N_ += L.nodet; et += L.Et  # else skip weak link inside segment
            else:
                i -= 1; break  # terminate contiguous-distance segment
        max_dist = _L.L
        G_ = []  # cluster current distance segment N_:
        if len(N_) > ave_L and et[0] > ave*ccoef:
            for N in {*N_}:
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
                # Gt:
                link_ = list({*link_});  Lay = CLay()
                [Lay.add_lay(lay) for lay in sum_H(link_, root, fi=0)]
                derTT = Lay.derTT
                # weigh m_|d_ by similarity to mean m|d, weigh derTT:
                m_,M = centroid_M(derTT[0], ave=ave); d_,D = centroid_M(derTT[1], ave=ave)
                et[:2] = M,D; Lay.derTT = np.array([m_,d_])
                # cluster roots:
                if val_(et, Et, ave) > 0:
                    G_ += [sum2graph(root, [list({*node_}),link_, et, Lay], 1, min_dist, max_dist)]
            else:
                G_ += N_  # unclustered nodes
        # longer links:
        L_ = L_[i + 1:]
        if L_:
            min_dist = max_dist  # next loop connects current-distance clusters via longer links
        else:
            break
    if G_:
        # highest dist segment, includes all nodes
        [comb_altG_(G.altG.node_, ave, rc) for G in G_]
        return sum_G_(G_)

def cluster_L_(root, L_, ave, rc):  # CC links via direct llinks, no dist-nesting

    G_ = []  # flood-filled link clusters
    for L in L_: L.fin = 0
    for L in L_:
        if L.fin: continue
        L.fin = 1
        node_, link_, Et, Lay = [L], [], copy(L.Et), CLay()
        for lL, _ in L.rimt[0] + L.rimt[1]:
            # eval by directional density?
            link_ += [lL]; Et += lL.Et
            _L = lL.nodet[0] if lL.nodet[1] is L else lL.nodet[1]
            if not _L.fin:
                _L.fin = 1; node_ += [_L]
        if val_(Et, Et, ave) > 0:
            Lay = CLay()
            [Lay.add_lay(l) for l in sum_H(link_, root, fi=0)]
            G_ += [sum2graph(root, [list({*node_}), link_, Et, Lay], 0)]
    if G_:
        [comb_altG_(G.altG.node_, ave, rc) for G in G_]
        G_ = sum_G_(G_)

    return G_

def cluster_C_(L_, rc):  # 0 nest gap from cluster_edge: same derH depth in root and top Gs

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

    def refine_C_(C_):  # refine weights in fuzzy C cluster around N, in root node_|link_
        # comp mean-node pairs, use match as weight of node_cluster_sum,
        # delete weak clusters, recompute cluster_sums of mean_matches
        # ave *= cluster_overlap | (dist/max_dist)**2
        remove_ = []
        for C in C_:
            r = 0  # recursion count
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
                            dm = (_m-vm) if r else vm  # replace init 0
                            dM += dm
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
                r += 1
        C_[:] = [C for C in C_ if C not in remove_]

    ave = globals()['ave'] * rc  # recursion count
    C_ = []  # form centroid clusters for next cross_comp
    N_ = list(set([node for link in L_ for node in link.nodet]))
    for N in N_: N.Ct_ = []
    N_ = sorted(N_, key=lambda n: n.Et[0], reverse=True)
    for N in N_:
        if N.Et[0] < ave*N.Et[3]: break
        med = 1; med_ = [1]; node_,_n_ = [[N]],[N]  # node_ is nested
        while med <= max_med and _n_:  # fill init C.node_: _Ns connected to N by <=3 mediation degrees
            n_ = [n for _n in _n_ for link,_ in _n.rim for n in link.nodet]
            med += 1
            n_ = list(set(n_))
            node_ += [n_]; med_ += [med]
            _n_ = n_
        C = sum_C(list(set([med_n for med_n_ in node_ for med_n in med_n_])))  # nested by med
        for n_, med in zip(node_,med_):
            for n in n_:
                n.Ct_ += [[C,0,med]]  # empty m, same n in multiple Ns, for med-weighted clustering
        C_ += [C]
    refine_C_(C_)  # refine centroid clusters
    # draft:
    if len(C_) > ave_L: C_ = sum_G_(C_)
    else:  C_ = []
    return C_

def layer_C_(root, L_, rc):  # node-parallel cluster_C_ in mediation layers, prune Cs in top layer?

    # same nodes on all layers, hidden layers mediate links up and down, don't sum or comp anything?
    pass

def comb_altG_(G_, ave, rc=1):  # combine contour G.altG_ into altG (node_ defined by root=G), for agg+ cross-comp
    # internal and external alts: different decay / distance?
    # background + contour?
    for G in G_:
        if isinstance(G,list): continue
        if G.altG:
            if G.altG.node_:
                G.altG = sum_G_(G.altG.node_)
                G.altG.node_ = [G.altG]  # formality for single-lev_G
                G.altG.root=G; G.altG.fi=0; G.altG.m=0
                if val_(G.altG.Et, G.Et, ave):  # alt D * G rM
                    cross_comp(G.altG, fi=1, rc=rc)  # adds nesting
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

def norm_H(H, n):
    for lay in H:
        if lay:
            if isinstance(lay, CLay):
                for v_ in lay.derTT: v_ *= n  # array
                lay.Et *= n
            else:
                for fork in lay:
                    for v_ in fork.derTT: v_ *= n  # array
                    fork.Et *= n  # same node_, link_
# not used:
def sort_H(H, fi):  # re-assign olp and form priority indices for comp_tree, if selective and aligned

    i_ = []  # priority indices
    for i, lay in enumerate(sorted(H.node_, key=lambda lay: lay.Et[fi], reverse=True)):
        di = lay.i - i  # lay index in H
        lay.olp += di  # derR - valR
        i_ += [lay.i]
    H.i_ = i_  # H priority indices: node/m | link/d
    if fi:
        H.root.node_ = H.node_

def centroid_M(m_, ave):  # adjust weights on attr matches | diffs, recompute with sum
    _w_ = np.ones(len(m_))
    am_ = np.abs(m_)  # m|d are signed, but their contribution to mean and w_ is absolute
    M = np.sum(am_)
    while True:
        mean = max(M / np.sum(_w_), 1e-7)
        inverse_dev_ = np.minimum(am_/mean, mean/am_)  # rational deviation from mean rm in range 0:1, 1 if m=mean, 0 if one is 0?
        w_ = inverse_dev_/.5  # 2/ m=mean, 0/ inf max/min, 1/ mid_rng | ave_dev?
        w_ *= 8 / np.sum(w_)  # mean w = 1, M shouldn't change?
        if np.sum(np.abs(w_-_w_)) > ave:
            M = np.sum(am_* w_)
            _w_ = w_
        else:
            break
        # recursion if weights change
    return m_* w_, M  # no return w_?

def agg_level(inputs):  # draft parallel

    frame, H, elevation = inputs
    lev_G = H[elevation]
    Lev_G = cross_comp(frame, fi=1, rc=0)  # return combined top composition level, append frame.derH
    if lev_G:
        # feedforward
        if len(H) < elevation+1: H += [Lev_G]  # append graph hierarchy
        else: H[elevation+1] = Lev_G
        # feedback
        if elevation > 0:
            if np.sum( np.abs(Lev_G.aves - Lev_G._aves)) > ave:  # filter update value, very rough
                m, d, n, o = Lev_G.Et
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
            comb_altG_(edge.node_, ave)
            cluster_C_(edge, ave)  # no cluster_C_ in vect_edge
            G_ += edge.node_  # unpack edges
        frame.node_ = G_
        manager = Manager()
        H = manager.list([CG(node_=G_)])
        maxH = 20
        with Pool(processes=4) as pool:
            pool.map(agg_level, [(frame, H, e) for e in range(maxH)])

        frame.aggH = list(H)  # convert back to list

def agg_H_seq(focus, image, _nestt=(1,0), rV=1, _rv_t=[]):  # recursive level-forming pipeline, called from cluster_C_

    global ave, ave_L, icoef, ccoef, max_med, ave_dist  # cost params
    ave, ave_L, icoef, ccoef, max_med, ave_dist = np.array([ave, ave_L, icoef, ccoef, max_med, ave_dist]) / rV

    frame = frame_blobs_root(focus, rV)  # no _rv_t
    intra_blob_root(frame, rV)  # not sure
    vect_root(frame, rV, _rv_t)
    if not frame.nnest:
        return frame
    comb_altG_(frame.node_[-1].node_, ave*2)  # PP graphs in frame.node_[2]
    # forward agg+:
    lev_G = cross_comp(frame, rc=1)  # node_+= edge.node_
    frame_link_ = [lG for lG in lev_G.node_[1].node_ if lG.nnest==frame.lnest] if lev_G and lev_G.node_[1] else []
    rM,rD = 1,1  # sum derTT coefs: m_,d_ [M,D,n,o, I,G,A,L] / Et, baseT, dimension
    rv_t = np.ones((2,8))  # d val is borrowed from pair m in proportion to d mag, scaled by fb:
    # feedback weights:
    for fd, nest,_nest, Q in zip((0,1), (frame.nnest,frame.lnest), _nestt, (frame.node_[1:],frame_link_)):  # skip blob_
        if nest==_nest: continue  # no new nesting
        hG = Q[-1]  # top level, no feedback
        for lev_G in reversed(Q[:-1]):  # CG or CL
            _m,_d,_n,_ = hG.Et; m,d,n,_ = lev_G.Et
            rM += (_m/_n) / (m/n)  # no o eval?
            rD += (_d/_n) / (d/n)
            rv_t += np.abs((hG.derTT/_n) / (lev_G.derTT/n))
            hG = lev_G
    # combined adjust:
    rV = (rM + rD) / 2 / (frame.nnest + frame.lnest)  # min = 1, as min nesting levels in both forks = 3
    if rV > ave:  # normalized
        base = frame.node_[2]; Et,box,baseT = base.Et, base.box, base.baseT
        # project focus by bottom D_val:
        if val_(Et, Et, ave, coef=20) > 0:  # mean value shift within focus, bottom only, internal search per G
            # include temporal Dm_+ Ddm_?
            dy,dx = baseT[-2:]  # gA from summed Gs
            y,x,Y,X = box  # current focus?
            y = y+dy; x = x+dx; Y = Y+dy; X = X+dx  # alter focus shape, also focus size: +/m-, res decay?
            if y > 0 and x > 0 and Y < image.shape[0] and X < image.shape[1]:  # focus is inside the image
                # rerun agg+ with new focus and aves:
                agg_H_seq(image[y:Y,x:X], image, (frame.nnest,frame.lnest), rV, rv_t)
                # all aves *= rV, but ultimately differential backprop per ave?
    return frame

def max_g_window(i__, wsize=64):  # set min,max coordinate filters, updated by feedback to shift the focus
    dy__ = (
            (i__[2:, :-2] - i__[:-2, 2:]) * 0.25 +
            (i__[2:, 1:-1] - i__[:-2, 1:-1]) * 0.50 +
            (i__[2:, 2:] - i__[:-2, 2:]) * 0.25 )
    dx__ = (
            (i__[:-2, 2:] - i__[2:, :-2]) * 0.25 +
            (i__[1:-1, 2:] - i__[1:-1, :-2]) * 0.50 +
            (i__[2:, 2:] - i__[:-2, 2:]) * 0.25
    )
    g__ = np.hypot(dy__, dx__)
    nY = (image.shape[0] + wsize-1) // wsize
    nX = (image.shape[1] + wsize-1) // wsize  # n windows

    max_window = g__[0:wsize, 0:wsize]; max_g = 0
    for iy in range(nY):
        for ix in range(nX):
            y0 = iy * wsize; yn = y0 + wsize
            x0 = ix * wsize; xn = x0 + wsize
            g = np.sum(g__[y0:yn, x0:xn])
            if g > max_g:
                max_window = i__[y0:yn, x0:xn]
                max_g = g
    return max_window

if __name__ == "__main__":
    # image_file = './images/raccoon_eye.jpeg'
    image_file = './images/toucan.jpg'
    image = imread(image_file)
    focus = max_g_window(image)
    frame = agg_H_seq(focus, image)  # focus will be shifted by internal feedback