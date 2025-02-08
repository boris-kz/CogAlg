import numpy as np
from copy import copy, deepcopy
from functools import reduce
from itertools import zip_longest
from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, intra_blob_root, imread, aves, Caves
from comp_slice import comp_latuple, comp_md_
from vect_edge import L2N, sum_H, add_H, comp_H, comp_N, comp_node_, comp_link_, sum2graph, get_rim, CG, vectorize_root, comp_area, extend_box, Val_
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
ave, ave_L = aves.m, aves.L

def cross_comp(root):  # form agg_Level by breadth-first node_,link_ cross-comp, connect clustering, recursion

    N_,L_,Et = comp_node_(root.node_[-1])  # cross-comp top-composition exemplars in root.node_
    # mfork
    if Val_(Et, _Et=Et, fd=0) > 0:  # cluster eval
        derH = [[mlay] for mlay in sum_H(L_,root, fd=1)]  # nested mlay per layer
        pL_ = {l for n in N_ for l,_ in get_rim(n,fd=0)}
        if len(pL_) > ave_L:
            cluster_N_(root, pL_, fd=0)  # form multiple distance segments, same depth
        # dfork, one for all distance segments, adds altGs, no higher Gs:
        L2N(L_,root)
        lN_,lL_,dEt = comp_link_(L_,Et)  # same root for L_, root.link_ was compared in root-forming for alt clustering
        if Val_(dEt, _Et=Et, fd=1) > 0:
            dderH = sum_H(lL_, root, fd=1)
            for lay, dlay in zip(derH, dderH): lay += [dlay]
            derH += [[[], dderH[-1]]]  # dderH is longer
            plL_ = {l for n in lN_ for l,_ in get_rim(n,fd=1)}
            if len(plL_) > ave_L:
                cluster_N_(root, plL_, fd=1)  # form altGs for cluster_C_, no new links between dist-seg Gs
        else:
            for lay in derH:
                lay += [[]]  # empty dlay
        root.derH = derH  # replace lower derH, may not align to node_,link_H append in cluster_N_
        comb_altG_(top_(root))  # comb node contour: altG_ | neg links sum, cross-comp -> CG altG
        cluster_C_(root)  # -> mfork G,altG exemplars, +altG surround borrow, root.derH + 1|2 lays
        # no dfork cluster_C_, no ddfork
        # if val_: lev_G -> agg_H_pipe
        return sum_G_(top_(root))

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
            if nest: Q += [G_]
            else:  Q[:] = [Q[:],G_]  # init nesting if link_, node_ is already nested
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

def cluster_C_(root):  # 0 from cluster_edge: same derH depth in root and top Gs

    def sum_C(dnode_, C=None):  # sum|subtract and average C-connected nodes

        if C is None:
            C,A = CG(node_=dnode_),CG(); C.fin = 1; sign=1  # add if new, else subtract (init C.fin here)
            C.M,C.L, A.M,A.L = 0,0,0,0  # centroid setattr
        else:
            A = C.altG; sign=0
            C.node_ = [n for n in C.node_ if n.fin]  # not in -ve dnode_, may add +ve later

        sum_G_(dnode_, sign, fc=1, G=C)  # no extend_box, sum extH
        sum_G_([n.altG for n in dnode_ if n.altG], sign, fc=0, G=A)  # no m, M, L in altGs
        k = len(dnode_) + 1-sign
        for n in C, A:  # get averages
            n.Et/=k; n.latuple/=k; n.vert/=k; n.aRad/=k; n.yx /= k
            norm_H(n.derH, k)
        C.box = reduce(extend_box, (n.box for n in C.node_))
        C.altG = A
        return C

    def comp_C(C, N):  # compute match without new derivatives: global cross-comp is not directional

        mL = min(C.L, len(N.node_)) - ave_L
        mA = comp_area(C.box, N.box)[0]
        mLat = comp_latuple(C.latuple, N.latuple, C.Et[2], N.Et[2])[1][0]
        mVert = comp_md_(C.vert[1], N.vert[1])[1][0]
        M = mL + mA + mLat + mVert
        M += sum([fork.Et[0] for lay in comp_H(C.derH, N.derH, rn=1, root=None, Et=np.zeros(4),fd=0) for fork in lay if fork])
        if C.altG and N.altG:  # converted to altG
            M += comp_N(C.altG, N.altG, C.altG.Et[2] / N.altG.Et[2]).Et[0]
        # if fuzzy C:
        # Et = np.zeros(4)  # m,_,n,o: lateral proximity-weighted overlap, for sparse centroids
        # M /= np.hypot(*C.yx, *N.yx), comp node_?
        return M

    def centroid_cluster(N, C_, root):  # form and refine C cluster, in last G but higher root?
        # add proximity bias, for both match and overlap?

        N.fin = 1; _N_ = [N]; CN_ = [N]
        med = 0
        while med < 3 and _N_:  # fill init C.node_: _Ns connected to N by <=3 mediation degrees
            N_ = []
            for _N in _N_:
                for link, _ in _N.rim:
                    n = link.nodet[0] if link.nodet[1] is _N else link.nodet[1]
                    if n.fin: continue  # in other C or in C.node_
                    n.fin = 1; N_ += [n]; CN_ += [n]  # no eval
            _N_ = N_  # mediated __Ns
            med += 1
        C = sum_C(list(set(CN_))) # C.node_
        while True:
            dN_, M, dM = [], 0, 0  # pruned nodes and values, or comp all nodes again?
            for _N in C.node_:
                m = comp_C(C,_N)  # Et if proximity-weighted overlap?
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
    C_ = []  # concat exemplar/centroid nodes across top Gs, for higher cross_comp
    for G in root.node_[-1]:
        N_ = [N for N in sorted([N for N in G.node_], key=lambda n: n.Et[0], reverse=True)]  # new G.node_ is never nested
        for N in N_:
            N.sign, N.m, N.fin = 1, 0, 0  # C update sign, inclusion m, inclusion flag
        for i, N in enumerate(N_):  # replace some nodes by their centroid clusters
            if not N.fin:  # not in prior C
                if Val_(sum([l.Et for l in N.extH]), _Et=root.Et, coef=10) > 0:  # cross-similar in G
                    centroid_cluster(N, C_, root)  # search via N.rim, C_ +=[C]
                else:  # M is lower in the rest of N_
                    break
    root.node_ += [C_]; root.nnest += 1
    if len(C_) > ave_L and not root.root:  # frame
        cross_comp(root)  # append derH, cluster_N_(root.node_[-1])

def top_(G, fd=0):
    return (G.link_[-1] if G.lnest else G.link_) if fd else (G.node_[-1] if G.nnest else G.node_)

def sum_G_(node_, s=1, fc=0, G=None):
    if G is None:  # we need to create new G here, else all new G referencing a same CG
        G = CG(); G.ave = Caves()
    for n in node_:
        G.latuple += n.latuple * s
        G.vert = G.vert + n.vert*s if np.any(G.vert) else deepcopy(n.vert) * s
        G.Et += n.Et * s; G.aRad += n.aRad * s
        G.yx += n.yx * s
        if n.derH: add_H(G.derH, n.derH, root=G, rev = s==-1, fc=fc, fd=0)
        if fc:
            G.M += n.m * s; G.L += s
        else:
            if n.extH: add_H(G.extH, n.extH, root=G, rev = s==-1, fd=1)  # empty in centroid
            G.box = extend_box( G.box, n.box)  # extended per separate node_ in centroid
    return G

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

def norm_H(H, n, fd=1):
    for lay in H:
        if fd:  # L.derH or extH
            for fork in lay.m_d_t: fork *= n  # arrays
            lay.Et *= n  # same node_, link_
        else:
            for fork in lay:
                if fork:
                   for m_d_ in fork.m_d_t: m_d_ *= n  # arrays
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

def agg_H_seq(focus):  # sequential level-updating pipeline

    frame = frame_blobs_root(focus)
    intra_blob_root(frame)
    vectorize_root(frame)
    if frame.node_[1]:  # any converted edges, no frame.link_, edge.link_
        G_, Nnest = [], 0
        for edge in frame.node_[-1]:
            if edge.nnest:  # has higher graphs
                comb_altG_(edge.node_)
                cluster_C_(edge)  # recursive, within edge?
                G_ = [Lev + lev for Lev, lev in zip_longest(G_, edge.node_, fillvalue=[])]  # concat levels
                Nnest = max(Nnest, edge.nnest)
        if Nnest==2:  # no added levs
            return frame
        frame.nnest = Nnest
        frame.node_ = [frame.node_[0], *G_]  # replace edge_ with new node levels
        agg_H = []
        # feedforward:
        while len(frame.node_[-1]) > ave_L:  # draft
            lev_G = cross_comp(frame)  # return combined top composition level, append frame.derH
            if lev_G:
                agg_H += [lev_G]  # indefinite graph hierarchy, sum main params?
                if Val_(lev_G.Et, lev_G.Et) < 0: break
            else: break
        if agg_H:  # feedback
            hG = lev_G; agg_H = agg_H[:-1]  # local top graph, gets no feedback
            while agg_H:
                lev_G = agg_H.pop()
                drm_ = hG.vert[0] - lev_G.vert[0]
                if sum(drm_) > 0: # filter update value
                    lev_G.drm_ = drm_ # proj agg+'rm = rm + drm?
                    # Et[0] = np.sqrt(sum([m**2 for m in rm_]) / L)?
                    # d-projected box: min,max coord filters
                    # add cost params: distance?
                    hG = lev_G  # replace higher lev
                else: break
            frame.node_ = agg_H
    return frame

if __name__ == "__main__":
    # image_file = './images/raccoon_eye.jpeg'
    image_file = './images/toucan.jpg'
    image = imread(image_file)
    # set min,max coordinate filters, updated by feedback to shift the focus within a frame:
    yn = xn = 64  # focal sub-frame size, = raccoon_eye, can be any number
    y0 = x0 = 300  # focal sub-frame start @ image center
    focus = image[y0:y0+yn, x0:x0+xn]
    frame = agg_H_seq(focus)