import numpy as np
from copy import copy, deepcopy
from functools import reduce
from itertools import zip_longest
from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, intra_blob_root, imread, aves, Caves
from vect_edge import L2N, base_comp, sum_G_, comb_H_, sum_H, add_H, comp_node_, comp_link_, sum2graph, get_rim, CG, CLay, vectorize_root, extend_box, Val_, val_
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
ave, ave_L, icoef, max_dist = aves.m, aves.L, aves.icoef, aves.max_dist

def cross_comp(root, fn):  # form agg_Level by breadth-first node_,link_ cross-comp, connect clustering, recursion

    N_,L_,Et = comp_node_(root.node_[-1].node_ if fn else root.link_[-1].node_)  # cross-comp top-composition exemplars
    # mfork
    if Val_(Et, _Et=Et, fd=0) > 0:  # cluster eval
        derH = [[comb_H_(L_, root, fd=1)]]  # nested mlay
        pL_ = {l for n in N_ for l,_ in get_rim(n, fd=0)}
        if len(pL_) > ave_L:
            cluster_N_(root, pL_,fd=0)  # form multiple distance segments, same depth
        if Val_(Et, _Et=Et, fd=0) > 0:
            L2N(L_,root)  # dfork, for all distance segments, adds altGs only
            lN_,lL_,dEt = comp_link_(L_,Et)  # same root for L_, root.link_ was compared in root-forming for alt clustering
            if Val_(dEt, _Et=Et, fd=1) > 0:
                derH[0] += [comb_H_(lL_, root, fd=1)]  # += dlay
                plL_ = {l for n in lN_ for l,_ in get_rim(n,fd=1)}
                if len(plL_) > ave_L:
                    cluster_N_(root, plL_, fd=1)  # form altGs for cluster_C_, no new links between dist-seg Gs
            else:
                derH[0] += [CLay()]  # empty dlay
        else: derH[0] += [CLay()]
        root.derH += derH  # feedback
        comb_altG_(top_(root))  # comb node contour: altG_ | neg links sum, cross-comp -> CG altG
        # agg eval +=derH,node_H:
        cluster_C_(root)  # -> mfork G,altG exemplars, +altG surround borrow, root.derH + 1|2 lays
        # no dfork cluster_C_, no ddfork
        # if val_: lev_G -> agg_H_seq
        return root.node_[-1]

def cluster_N_(root, L_, fd):  # top-down segment L_ by >ave ratio of L.dists

    L_ = sorted(L_, key=lambda x: x.dist)  # shorter links first
    min_dist = 0; Et = root.Et
    while True:
        # each loop forms G_ of contiguous-distance L_ segment
        _L = L_[0]; N_, et = copy(_L.nodet), _L.Et
        for n in [n for l in L_ for n in l.nodet]:
            n.fin = 0
        for i, L in enumerate(L_[1:], start=1):
            rel_dist = L.dist/_L.dist  # >= 1
            if rel_dist < 1.2 or Val_(et, _Et=Et) > 0 or len(L_[i:]) < ave_L:  # ~=dist Ns or either side of L is weak
                _L = L; N_ += L.nodet; et += L.Et
            else:
                i -= 1; break  # terminate contiguous-distance segment
        G_ = []
        max_dist = _L.dist
        for N in {*N_}:  # cluster current distance segment
            if N.fin: continue  # clustered from prior _N_
            _eN_,node_,link_,et, = [N],[],[], np.zeros(4)
            while _eN_:
                eN_ = []
                for eN in _eN_:  # cluster rim-connected ext Ns, all in root Gt
                    node_+=[eN]; eN.fin = 1  # all rim
                    for L,_ in get_rim(eN, fd):  # all +ve, * density: if L.Et[0]/ave_d * sum([n.extH.m * ccoef / ave for n in L.nodet])?
                        if L not in link_:
                            eN_ += [n for n in L.nodet if not n.fin]
                            if L.dist < max_dist:
                                link_+=[L]; et+=L.Et
                _eN_ = {*eN_}
            if Val_(et, _Et=Et) > 0:  # cluster node roots:
                G_ += [sum2graph(root, [list({*node_}),list({*link_}), et], fd, min_dist, max_dist)]
        # longer links:
        L_ = L_[i+1:]
        if L_: min_dist = max_dist  # next loop connects current-dist clusters via longer links
        else:
            nest,Q = (root.lnest, root.link_) if fd else (root.nnest, root.node_)
            if nest: Q += [sum_G_(G_)]
            else:  Q[:] = [sum_G_(Q[:]),sum_G_(G_)]  # init nesting if link_, node_ is already nested
            if fd: root.lnest += 1
            else:  root.nnest += 1
            break
''' 
Hierarchical clustering should alternate between two phases: generative via connectivity and compressive via centroid.

 Connectivity clustering terminates at effective contours: alt_Gs, beyond which cross-similarity is not likely to continue. 
 Next cross-comp is discontinuous and should be selective, for well-defined clusters: stable and likely recurrent.
 
 Such clusters should be compared and clustered globally: via centroid clustering, vs. local connectivity clustering.
 Only centroids (exemplars) need to be cross-compared on the next connectivity clustering level, representing their nodes.
 
 So connectivity clustering is a generative learning phase, forming new derivatives and structured composition levels, 
 while centroid clustering is a compressive phase, reducing multiple similar comparands to a single exemplar. '''

def cluster_C_(root):  # 0 nest gap from cluster_edge: same derH depth in root and top Gs

    def sum_C(dnode_, C=None):  # sum|subtract and average C-connected nodes

        if C is None:
            C,A = CG(node_=dnode_),CG(); C.fin = 1; sign=1  # add if new, else subtract (init C.fin here)
            C.M,C.L, A.M,A.L = 0,0,0,0  # centroid setattr
        else:
            A = C.altG; sign=0
            C.node_ = [n for n in C.node_ if n.fin]  # not in -ve dnode_, may add +ve later

        sum_G_(dnode_, sign, fc=1, G=C)  # no extH, extend_box
        sum_G_([n.altG for n in dnode_ if n.altG], sign, fc=0, falt=1, G=A)  # no m, M, L in altGs
        k = len(dnode_) + 1-sign
        for falt, n in zip((0,1), (C, A)):  # get averages
            n.Et/=k; n.derTT/=k; n.aRad/=k; n.yx /= k
            if np.any(n.baseT): n.baseT/=k
            norm_H(n.derH, k, fd=falt)  # alt has single layer
        C.box = reduce(extend_box, (n.box for n in C.node_))
        C.altG = A
        return C

    def centroid_cluster(N, C_, root):  # form and refine C cluster around N, in root node_|link_?
        # proximity bias for both match and overlap?
        # draft:
        N.fin = 1; CN_ = [N]
        for n in N_:
            if not hasattr(n,'fin') or n.fin or n is N: continue  # in other C or in C.node_, or not in root
            radii = N.aRad + n.aRad
            dy, dx = np.subtract(N.yx, n.yx)
            dist = np.hypot(dy, dx)
            # probably too complex:
            en = len(N.extH) * N.Et[2:]; _en = len(n.extH) * n.Et[2:]  # same n*o?
            GV = val_(N.Et) + val_(n.Et) + (sum(N.derTTe[0])-ave*en) + (sum(n.derTTe[0])-ave*_en)
            if dist > max_dist * ((radii * icoef**3) * GV): continue
            n.fin = 1; CN_ += [n]
        # same:
        C = sum_C(CN_)  # C.node_
        while True:
            dN_, M, dM = [], 0, 0  # pruned nodes and values, or comp all nodes again?
            for _N in C.node_:
                m = sum( base_comp(C,_N)[0][0])
                if C.altG and _N.altG: m += sum( base_comp(C.altG,_N.altG)[0][0])  # Et if proximity-weighted overlap?
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
    # C-cluster top node_|link_:
    C_t = [[],[]]  # concat exemplar/centroid nodes across top Gs for global frame cross_comp
    for fn, C_,nest,_N_ in zip((1,0), C_t, [root.nnest,root.lnest], [root.node_,root.link_]):
        if not nest: continue
        N_ = [N for N in sorted([N for N in _N_[-1].node_], key=lambda n: n.Et[fn], reverse=True)]
        for N in N_:
            N.sign, N.m, N.fin = 1, 0, 0  # C update sign, inclusion m, inclusion flag
        for N in N_:
            if not N.fin:  # not in prior C
                if Val_(N.Et, _Et=root.Et, coef=10) > 0:  # cross-similar in G
                    centroid_cluster(N, C_, root)  # form centroid around N, C_ +=[C]
                else:
                    break  # the rest of N_ is lower-M
        if len(C_) > ave_L:
            if fn:
                root.node_ += [sum_G_(C_)]; root.nnest += 1
            else:
                root.link_ += [sum_G_(C_)]; root.lnest += 1
            if not root.root:  # frame
                cross_comp(root, fn)  # append derH, cluster_N_([root.node_,root.link_][fn][-1])

def top_(G, fd=0):
    return (G.link_[-1] if G.lnest else G.link_) if fd else (G.node_[-1] if G.nnest else G.node_)

def comb_altG_(G_):  # combine contour G.altG_ into altG (node_ defined by root=G), for agg+ cross-comp
    # internal and external alts: different decay / distance?
    # background vs contour?
    for G in G_:
        if isinstance(G,list): continue
        if G.altG:
            if isinstance(G.altG, list):
                sum_G_(G.altG)
                G.altG = CG(root=G, node_= G.altG); G.altG.m=0  # was G.altG_
                if Val_(G.altG.Et, _Et=G.Et):  # alt D * G rM
                    cross_comp(G.altG, G.node_)
        else:  # sum neg links
            link_,node_,derH, Et = [],[],[], np.zeros(4)
            for link in G.link_:
                if Val_(link.Et, _Et=G.Et) > 0:  # neg link
                    link_ += [link]  # alts are links | lGs
                    node_ += [n for n in link.nodet if n not in node_]
                    Et += link.Et
            if Val_(Et, _Et=G.Et, coef=10) > 0:  # min sum neg links
                altG = CG(root=G, Et=Et, node_=node_, link_=link_); altG.m=0  # other attrs are not significant
                altG.derH = sum_H(altG.link_, altG, fd=1)   # sum link derHs
                G.altG = altG

def norm_H(H, n, fd=0):
    if fd: H = [H]  # L.derH is not nested
    for lay in H:
        if lay:
            for fork in lay:
                for v_ in fork.derTT: v_ *= n  # array
                fork.Et *= n  # same node_, link_
# not used:
def sort_H(H, fd):  # re-assign olp and form priority indices for comp_tree, if selective and aligned

    i_ = []  # priority indices
    for i, lay in enumerate(sorted(H.node_, key=lambda lay: lay.Et[fd], reverse=True)):
        di = lay.i - i  # lay index in H
        lay.olp += di  # derR - valR
        i_ += [lay.i]
    H.i_ = i_  # H priority indices: node/m | link/d
    if not fd:
        H.root.node_ = H.node_

def centroid_M_(m_, M, ave):  # adjust weights on attr matches, also add cost attrs
    _w_ = [1 for _ in m_]
    while True:
        w_ = [min(m/M, M/m) for m in m_]  # rational deviations from mean,
        # in range 0:1, or 0:2: w = min(m/M, M/m) + mean(min(m/M, M/m))
        Dw = sum([abs(w-_w) for w,_w in zip(w_,_w_)])  # weight update
        M = sum(m*w for m, w in zip(m_,w_)) / sum(w_)  # M update
        if Dw > ave:
            _w_ = w_
        else:
            break
    return w_, M

def agg_level(inputs):  # draft parallel

    frame, H, elevation = inputs
    lev_G = H[elevation]
    Lev_G = cross_comp(frame)  # return combined top composition level, append frame.derH
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
    vectorize_root(frame)
    if frame.node_:  # converted edges
        G_ = []
        for edge in frame.node_:
            comb_altG_(edge.node_)
            cluster_C_(edge)  # no cluster_C_ in vect_edge
            G_ += edge.node_  # unpack edges
        frame.node_ = G_
        manager = Manager()
        H = manager.list([CG(node_=G_)])
        maxH = 20
        with Pool(processes=4) as pool:
            pool.map(agg_level, [(frame, H, e) for e in range(maxH)])

        frame.aggH = list(H)  # convert back to list

def agg_H_seq(focus,_nestt=(1,0)):  # recursive level-forming pipeline, called from cluster_C_

    cluster_C_(frame)  # feedforward: recursive agg+ in edge)frame, both fork levs are lev_Gs
    dm_t = [[],[]]
    bottom_t = []
    for nest,_nest,Q in zip((frame.nnest,frame.lnest),_nestt, (frame.node_,frame.link_)):
        if nest==_nest: continue  # no new nesting
         # feedback, both Qs are lev_G_s:
        hG = Q[-1]  # init
        bottom = 1
        for lev_G in reversed(Q[:-1]):  # top level gets no feedback
            hm_ = hG.derTT[0] # + ave m-associated pars: len, dist, dcoords?
            hm_ = centroid_M_(hm_, sum(hm_)/8, ave)
            dm_ = hm_ - lev_G.aves
            if sum(dm_) > ave:  # update
                lev_G.aves = hm_  # proj agg+'m = m + dm?
                # project focus by val_* dy,dx: frame derTT dgA / baseT gA?
                # mean value shift within focus, bottom only, internal search per G
                hG = lev_G
            else:
                bottom = 0; break  # feedback did not reach the bottom level
        dm_t += [dm_]
        bottom_t += [bottom]
    if any(bottom_t) and sum(dm_t[0]) +sum(dm_t[1]) > ave:
        # bottom level is refocused, new aves, rerun agg+:
        agg_H_seq(focus,(frame.nnest,frame.lnest))

    return frame

if __name__ == "__main__":
    # image_file = './images/raccoon_eye.jpeg'
    image_file = './images/toucan.jpg'
    image = imread(image_file)
    # set min,max coordinate filters, updated by feedback to shift the focus within a frame:
    yn = xn = 64  # focal sub-frame size, = raccoon_eye, can be any number
    y0 = x0 = 300  # focal sub-frame start @ image center
    focus = image[y0:y0+yn, x0:x0+xn]
    frame = frame_blobs_root(focus)
    intra_blob_root(frame)
    vectorize_root(frame)
    if frame.node_[1]:  # unpack converted edges
        frame.node_[-1] = [[G for G in edge.node_[1]] for edge in frame.node_[1]]
        [comb_altG_(G.altG) for G in frame.node_[-1]]
        frame = agg_H_seq(focus)  # focus will be shifted by internal feedback