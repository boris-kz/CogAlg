import numpy as np
from copy import copy, deepcopy
from functools import reduce
from itertools import zip_longest
from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, intra_blob_root, imread, aves
from vect_edge import L2N, base_comp, sum_G_, comb_H_, sum_H, copy_, comp_node_, comp_link_, sum2graph, get_rim, CG, CLay, vectorize_root, extend_box, Val_, val_
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
ave, ave_L, icoef, max_dist = aves[-2], aves[6], aves[12], aves[9]

def cross_comp(root, fn, rc):  # form agg_Level by breadth-first node_,link_ cross-comp, connect clustering, recursion
    # rc: recursion count coef to ave

    N_,L_,Et = comp_node_(root.node_[-1].node_ if fn else root.link_[-1].node_, ave*rc)  # cross-comp top-composition exemplars
    # mval -> lay
    if Val_(Et, Et, ave*(rc+1), fd=0) > 0:  # cluster eval
        derH = [[comb_H_(L_, root, fd=1)]]  # nested mlay
        pL_ = {l for n in N_ for l,_ in get_rim(n, fd=0)}
        if len(pL_) > ave_L:
            cluster_N_(root, pL_, ave*(rc+2), fd=0)  # form multiple distance segments, same depth
        # dval -> comp L_ for all dist segments, adds altGs
        if Val_(Et, Et, ave*(rc+2), fd=1) > 0:
            lN_,lL_,dEt = comp_link_(L2N(L_), ave*(rc+2))  # comp root.link_ forms root in alt clustering?
            if Val_(dEt, Et, ave*(rc+3), fd=1) > 0:
                derH[0] += [comb_H_(lL_, root, fd=1)]  # += dlay
                plL_ = {l for n in lN_ for l,_ in get_rim(n,fd=1)}
                if len(plL_) > ave_L:
                    cluster_N_(root, plL_, ave*(rc+4), fd=1)  # form altGs for cluster_C_, no new links between dist-seg Gs
        # feedback:
        root.derH += derH
        comb_altG_(root.node_[-1].node_, ave*(rc+4))  # comb node contour: altG_ | neg links sum, cross-comp -> CG altG
        cluster_C_(root, rc+5)  # -> mfork G,altG exemplars, +altG surround borrow, root.derH + 1|2 lays, agg++
        # no dfork cluster_C_, no ddfork
        # if val_: lev_G -> agg_H_seq
        return root.node_[-1]

def cluster_N_(root, L_, ave, fd):  # top-down segment L_ by >ave ratio of L.dists

    L_ = sorted(L_, key=lambda x: x.dist)  # short links first
    min_dist = 0; Et = root.Et
    while True:
        # each loop forms G_ of contiguous-distance L_ segment
        _L = L_[0]; N_, et = copy(_L.nodet), _L.Et
        for n in [n for l in L_ for n in l.nodet]:
            n.fin = 0
        for i, L in enumerate(L_[1:], start=1):
            rel_dist = L.dist/_L.dist  # >= 1
            if rel_dist < 1.2 or len(L_[i:]) < ave_L:  # ~= dist Ns or either side of L is weak: continue dist segment
                LV = Val_(et, Et, ave)  # link val
                _G,G = L.nodet  # * surround density: extH (_Ete[0]/ave + Ete[0]/ave) / 2, after cross_comp:
                sur_rV = (sum(_G.derTTe[0]) / (ave*_G.Et[2])) + (sum(G.derTTe[0]) / (ave*G.Et[2])) / 2
                if LV * sur_rV > ave:
                    _L = L; N_ += L.nodet; et += L.Et  # else skip weak link inside segment
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
                    for L,_ in get_rim(eN, fd):  # all +ve
                        if L not in link_:
                            eN_ += [n for n in L.nodet if not n.fin]
                            if L.dist < max_dist:
                                link_+=[L]; et+=L.Et
                _eN_ = {*eN_}
            link_ = list({*link_});  Lay = CLay()
            [Lay.add_lay(lay) for lay in sum_H(link_, root, fd=1)]
            derTT = Lay.derTT
            # weigh m_|d_ by similarity to mean m|d, replacing derTT:
            _,M = centroid_M_(derTT[0], np.sum(derTT[0]), ave)
            _,D = centroid_M_(derTT[1], np.sum(derTT[1]), ave)
            et[:2] = M,D
            if Val_(et, Et, ave) > 0:  # cluster node roots:
                G_ += [sum2graph(root, [list({*node_}),link_, et, Lay], fd, min_dist, max_dist)]
        # longer links:
        L_ = L_[i + 1:]
        if L_: min_dist = max_dist  # next loop connects current-distance clusters via longer links
        else:
            if G_:
                [comb_altG_(G.altG, ave) for G in G_]
                if fd:
                    if root.lnest: root.link_ += [sum_G_(G_)]
                    else: root.link_ = [sum_G_(root.link_), sum_G_(G_)]  # init nesting
                    root.lnest += 1
                else:
                    root.node_ += [sum_G_(G_)]  # node_ is already nested
                    root.nnest += 1
            break
''' 
Hierarchical clustering should alternate between two phases: generative via connectivity and compressive via centroid.

 Connectivity clustering terminates at effective contours: alt_Gs, beyond which cross-similarity is not likely to continue. 
 Next cross-comp is discontinuous and should be selective, for well-defined clusters: stable and likely recurrent.
 
 Such clusters should be compared and clustered globally: via centroid clustering, vs. local connectivity clustering.
 Only centroids (exemplars) need to be cross-compared on the next connectivity clustering level, representing their nodes.
 
 So connectivity clustering is a generative learning phase, forming new derivatives and structured composition levels, 
 while centroid clustering is a compressive phase, reducing multiple similar comparands to a single exemplar. '''

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
            n.Et/=k; n.derTT/=k; n.aRad/=k; n.yx /= k
            if np.any(n.baseT): n.baseT/=k
            norm_H(n.derH, k)
        C.box = reduce(extend_box, (n.box for n in C.node_))

        return C

    def centroid_cluster(N, N_, C_, root):  # form and refine C cluster around N, in root node_|link_?
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
                m = sum( base_comp(C,_N)[0][0])  # derTT
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

def comb_altG_(G_, ave):  # combine contour G.altG_ into altG (node_ defined by root=G), for agg+ cross-comp
    # internal and external alts: different decay / distance?
    # background vs contour?
    for G in G_:
        if isinstance(G,list): continue
        if G.altG:
            if isinstance(G.altG, list):
                sum_G_(G.altG)
                G.altG = CG(root=G, node_= G.altG); G.altG.m=0  # was G.altG_
                if Val_(G.altG.Et, G.Et, ave):  # alt D * G rM
                    cross_comp(G.altG, G.node_, ave)  # need rc?
        else:  # sum neg links
            link_,node_,derH, Et = [],[],[], np.zeros(4)
            for link in G.link_:
                if Val_(link.Et, G.Et, ave) > 0:  # neg link
                    link_ += [link]  # alts are links | lGs
                    node_ += [n for n in link.nodet if n not in node_]
                    Et += link.Et
            if Val_(Et, G.Et, ave, coef=10) > 0:  # altG-specific coef for sum neg links
                altG = CG(root=G, Et=Et, node_=node_, link_=link_); altG.m=0  # other attrs are not significant
                altG.derH = sum_H(altG.link_, altG, fd=1)   # sum link derHs
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
def sort_H(H, fd):  # re-assign olp and form priority indices for comp_tree, if selective and aligned

    i_ = []  # priority indices
    for i, lay in enumerate(sorted(H.node_, key=lambda lay: lay.Et[fd], reverse=True)):
        di = lay.i - i  # lay index in H
        lay.olp += di  # derR - valR
        i_ += [lay.i]
    H.i_ = i_  # H priority indices: node/m | link/d
    if not fd:
        H.root.node_ = H.node_

def centroid_M_(m_, M, ave):  # adjust weights on attr matches | diffs, recompute with sum

    _w_ = np.ones(len(m_))  # add cost attrs?
    while True:
        M /= np.sum(_w_)  # mean
        w_ = m_ / min(M, 1/M)  # rational deviations from mean,
        # in range 0:1, or 0:2: w = min(m/M, M/m) + mean(min(m/M, M/m))?
        Dw = np.sum( np.abs(w_-_w_))  # weight update
        m_[:] = m_ * w_  # replace in each cycle?
        M = np.sum(m_)  # weighted M update
        if Dw > ave:
            _w_ = w_
        else:
            break
    return w_, M  # no need to return weights?

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

def agg_H_seq(focus, image, _nestt=(1,0)):  # recursive level-forming pipeline, called from cluster_C_

    global aves

    frame = frame_blobs_root(focus)
    intra_blob_root(frame)
    vectorize_root(frame)
    if not frame.nnest:
        return frame
    comb_altG_(frame.node_[-1].node_, ave*2)  # PP graphs in frame.node_[2]
    # feedforward agg+
    cluster_C_(frame, rc=1)  # ave *= recursion count
    rM,rD = 1,1  # sum derTT coefs: m_,d_ [M,D,n, I,G,gA, L,A] / Et,baseT,ext:
    rV_t = np.ones((2,8))  # d value is borrowed from corresponding ms in proportion to d mag, both scaled by fb
    # feedback to scale m,d aves:
    for fd, nest,_nest,Q in zip((0,1), (frame.nnest,frame.lnest), _nestt, (frame.node_[2:],frame.link_[1:])):  # skip blob_,PP_,link_PP_
        if nest==_nest: continue  # no new nesting
        hG = Q[-1]  # top level, no feedback
        for lev_G in reversed(Q[:-1]):
            _m,_d,_n,_ = hG.Et; m,d,n,_ = lev_G.Et
            rM += (_m/_n) / (m/n)  # no o eval?
            rD += (_d/_n) / (d/n)
            rV_t += (hG.derTT/_n) / (lev_G.derTT/n)
            hG = lev_G
    if rM > ave:  # base-level
        base = frame.node_[2]; Et,box,baseT = base.Et, base.box, base.baseT
        # project focus by bottom D_val:
        if Val_(Et, Et, ave, coef=20) > 0:  # mean value shift within focus, bottom only, internal search per G
            # include temporal Dm_+ Ddm_?
            dy,dx = baseT[-2:]  # gA from summed Gs
            y,x,Y,X = box  # current focus?
            y = y+dy; x = x+dx; Y = Y+dy; X = X+dx  # alter focus shape, also focus size: +/m-, res decay?
            if y > 0 and x > 0 and Y < image.shape[0] and X < image.shape[1]:  # focus is inside the image
                aves[:16] *= rV_t[:]
                frame.aves = aves  # adjust other aves too?
                # rerun agg+ with new bottom-level focus, aves:
                agg_H_seq(image[y:Y,x:X], image, (frame.nnest,frame.lnest))

    return frame

def max_g_window(i__, wsize=64):
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
    # set min,max coordinate filters, updated by feedback to shift the focus within a frame:
    frame = agg_H_seq(focus, image)  # recursion count, focus will be shifted by internal feedback