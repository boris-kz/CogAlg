import numpy as np
from copy import copy, deepcopy
from functools import reduce
from frame_blobs import frame_blobs_root, intra_blob_root, imread
from comp_slice import comp_latuple, comp_md_
from vect_edge import icoef, norm_H, add_H, sum_H, comp_N, comp_node_, comp_link_, sum2graph, zrim, CLay, CG, ave, ave_L, vectorize_root, comp_area, extend_box, val_
'''
Cross-compare and cluster Gs within a frame, potentially unpacking their node_s first,
alternating agglomeration and centroid clustering.
notation:
prefix  f denotes flag
postfix t denotes tuple, multiple ts is a nested tuple
prefix  _ denotes prior of two same-name variables, multiple _s for relative precedence
postfix _ denotes array of same-name elements, multiple _s is nested array
capitalized vars are summed small-case vars 

Each agg+ cycle forms higher-composition complemented graphs G.altG_ in cluster_N_ and refines them in cluster_C_: 
cross_comp -> cluster_N_ -> cluster_C -> cross_comp...
'''

def cross_comp(root, nest=0):  # breadth-first node_,link_ cross-comp, connect.clustering, recursion

    N_,L_,Et = comp_node_(root.derH[-1].node_)  # cross-comp exemplars, extrapolate to their node_s
    # mfork
    if val_(Et, fo=1) > 0:
        root.derH += [sum_H(L_,root)]  # += [mlay]
        pL_ = {l for n in N_ for l,_ in zrim(n, fd=0)}
        if len(pL_) > ave_L:
            cluster_N_(root, pL_, nest, fd=0)  # nested distance clustering, calls centroid and higher connect.clustering
        # dfork,
        # replace with dist-layered link_ cross-comp inside cluster_N_?
        if val_(Et, _Et=Et, fo=1) > 0:  # same root for L_, root.link_ was compared in root-forming for alt clustering
            for L in L_:
                L.extH, L.root, L.Et, L.mL_t, L.rimt, L.aRad, L.visited_ = [],root,copy(L.derH.Et), [[],[]], [[],[]], 0,[L]
            lN_,lL_,dEt = comp_link_(L_,Et)
            if val_(dEt, _Et=Et, fo=1) > 0:
                root.derH[-1].add_lay( sum_H(lL_,root))  # mlay += dlay
                plL_ = {l for n in lN_ for l,_ in zrim(n, fd=1)}
                if len(plL_) > ave_L:
                    cluster_N_(root, plL_, nest, fd=1)

def cluster_N_(root, L_, nest, fd):  # top-down segment L_ by >ave ratio of L.dists

    L_ = sorted(L_, key=lambda x: x.dist)  # shorter links first
    min_dist = 0
    while True:
        # each loop forms G_ of contiguous-distance L_ segment
        _L = L_[0]; N_, et = copy(_L.nodet), _L.Et
        for n in [n for l in L_ for n in l.nodet]:
            n.fin = 0
        for i, L in enumerate(L_[1:], start=1):
            rel_dist = L.dist/_L.dist  # >= 1
            if rel_dist < 1.2 or val_(et)>0 or len(L_[i:]) < ave_L:  # ~=dist Ns or either side of L is weak
                _L = L; N_ += L.nodet; et += L.Et
            else:
                i -= 1; break  # terminate contiguous-distance segment
        G_ = []
        max_dist = _L.dist
        for N in {*N_}:  # cluster current distance segment
            if N.fin: continue  # clustered from prior eN_
            _eN_, node_,link_, et, = [N],[],[],np.zeros(4)
            while _eN_:
                eN_ = []
                for eN in _eN_:  # cluster rim-connected ext Ns, all in root Gt
                    node_+=[eN]; eN.fin = 1  # all rim
                    for L,_ in zrim(eN, fd):
                        if L not in link_:   # add density term: if L.derH.Et[0]/ave * n.extH m/ave or L.derH.Et[0] + n.extH m*.1?
                            eN_ += [n for n in L.nodet if not n.fin]
                            if L.dist < max_dist:
                                link_+=[L]; et+=L.Et
                _eN_ = {*eN_}
            if val_(et) > 0:  # cluster node roots:
                G_ += [sum2graph(root, [list({*node_}),list({*link_}), et], fd, min_dist, max_dist, nest)]
            else:  # unpack
                for n in {*node_}:
                    n.nest += 1; G_ += [n]
        # draft:
        # already in cross-comp, not dist-layered?:
        # root.derH += [CLay(Et=et, root=root, m_d_t = np.sum([l.derH[-1].m_d_t for l in L_],axis=0), node_=G_, link_=L_)]
        # dist-layered cross-comp link_[:i] here, not in root cross-comp?
        if val_(et, _Et=et, fo=1) > 0:  # same root for L_, root.link_ was compared in root-forming for alt clustering
            for L in L_:
                L.extH, L.root, L.Et, L.mL_t, L.rimt, L.aRad, L.visited_ = [],root,copy(L.derH.Et), [[],[]], [[],[]], 0,[L]
            lN_,lL_,dEt = comp_link_(L_,et)
            if val_(dEt, _Et=et, fo=1) > 0:
                root.derH[-1].add_lay( sum_H(lL_,root))  # mlay += dlay
                plL_ = {l for n in lN_ for l,_ in zrim(n, fd=1)}
                if len(plL_) > ave_L:
                    cluster_N_(root, plL_, nest, fd=1)

        comb_altG_(G_, root, fd)  # combine node contour: altG_ or neg links, by sum,cross-comp -> CG altG
        cluster_C_(root)  # get (G,altG) exemplars, altG surround borrow may reinforce G?  (fd is not needed?)
        # both fd Gs are complemented?
        L_ = L_[i+1:]
        if L_:
            nest += 1; min_dist = max_dist  # get longer links for next loop, to connect current-dist clusters
        else:
            break
''' 
Hierarchical clustering should alternate between two phases: generative via connectivity and compressive via centroid.

 Connectivity clustering terminates at effective contours: alt_Gs, beyond which cross-similarity is not likely to continue. 
 Next cross-comp is discontinuous and should be selective, for well-defined clusters: stable and likely recurrent.
 
 Such clusters should be compared and clustered globally: via centroid clustering, vs. local connectivity clustering.
 Only centroids (exemplars) need to be cross-compared on the next connectivity clustering level, representing their nodes.
 
 So connectivity clustering is a generative learning phase, forming new derivatives and structured composition levels, 
 while centroid clustering is a compressive phase, reducing multiple similar comparands to a single exemplar. '''

def cluster_C_(graph):

    def centroid(dnode_, node_, C=None):  # sum|subtract and average Rim nodes

        if C is None:
            C,A, fC = CG(),CG(), 0
            C.M,C.L, A.M,A.L = 0,0,0,0  # centroid setattr
        else:
            A, fC = C.altG, 1
        sum_G_(C, dnode_, fc=1)  # exclude extend_box and sum extH
        sum_G_(A, [n.altG for n in dnode_ if n.altG], fc=1)
        k = len(dnode_) + fC
        for n in C, A:  # get averages
            n.Et/=k; n.latuple/=k; n.vert/=k; n.aRad/=k; n.yx /= k
            norm_H(n.derH, k)
        C.box = reduce(extend_box, (n.box for n in node_))
        C.altG = A
        return C

    def comp_C(C, N):  # compute match without new derivatives: global cross-comp is not directional

        mL = min(C.L, len(N.derH[-1].node_)) - ave_L
        mA = comp_area(C.box, N.box)[0]
        mLat = comp_latuple(C.latuple, N.latuple, C.Et[2], N.Et[2])[1][0]
        mVert = comp_md_(C.vert[1], N.vert[1])[1][0]
        M = mL + mA + mLat + mVert
        M += sum([_lay.comp_lay(lay, rn=1,root=None).Et[0] for _lay,lay in zip(C.derH, N.derH)])
        if C.altG and N.altG:  # converted to altG
            M += comp_N(C.altG, N.altG, C.altG.Et[2] / N.altG.Et[2]).Et[0]
        # if fuzzy C:
        # Et = np.zeros(4)  # m,_,n,o: lateral proximity-weighted overlap, for sparse centroids
        # M /= np.hypot(*C.yx, *N.yx)
        # comp node_?
        return M

    def centroid_cluster(N):  # refine and extend cluster with extN_

        # add proximity bias, for both match and overlap?
        _N_ = {n for L,_ in N.rim for n in L.nodet if not n.fin}
        N.fin = 1; n_ = _N_| {N}  # include seed node
        C = centroid(n_, n_)
        while True:
            N_,dN_,extN_, M, dM, extM = [],[],[], 0,0,0  # included, changed, queued nodes and values
            for _N in _N_:
                m = comp_C(C,_N)  # Et if proximity-weighted overlap?
                vm = m - ave  # deviation
                if vm > 0:
                    N_ += [_N]; M += m
                    if _N.m: dM += m - _N.m  # was in C.node_, add adjustment
                    else:  dN_ += [_N]; dM += vm  # new node
                    _N.m = m  # to sum in C
                    for link, _ in _N.rim:
                        n = link.nodet[0] if link.nodet[1] is _N else link.nodet[1]
                        if n.fin or n.m: continue  # in other C or in C.node_
                        extN_ += [n]; extM += n.Et[0]  # external N for next loop
                elif _N.m:  # was in C.node_, subtract from C
                    _N.sign=-1; _N.m=0; dN_+=[_N]; dM += -vm  # dM += abs m deviation

            if dM > ave and M + extM > ave:  # update for next loop, terminate if low reform val
                if dN_:  # recompute C if any changes in node_
                    C = centroid(set(dN_), N_, C)
                _N_ = set(N_) | set(extN_)  # next loop compares both old and new nodes to new C
                C.M = M; C.node_ = N_
            else:
                if C.M > ave * 10:  # add proximity-weighted overlap
                    C.root = N.root  # C.nest = N.nest+1
                    for n in C.node_:
                        n.root = C; n.fin = 1; delattr(n,"sign")
                    return C  # centroid cluster
                else:  # unpack C.node_
                    for n in C.node_: n.m = 0
                    N.nest += 1
                    return N  # keep seed node

    # get representative centroids of complemented Gs: mCore + dContour, initially in unpacked edges
    N_ = sorted([N for N in graph.derH[-1].node_ if any(N.Et)], key=lambda n: n.Et[0], reverse=True)
    G_ = []
    for N in N_:
        N.sign, N.m, N.fin = 1, 0, 0  # setattr C update sign, inclusion val, C inclusion flag
    for i, N in enumerate(N_):  # replace some of connectivity cluster by exemplar centroids
        if not N.fin:  # not in prior C
            if val_(N.Et, coef=10) > 0:
                G_ += [centroid_cluster(N)]  # extend from N.rim, return C if packed else N
            else:  # the rest of N_ M is lower
                G_ += [N for N in N_[i:] if not N.fin]
                break
    graph.node_ = G_  # mix of Ns and Cs: exemplars of their network?
    if len(G_) > ave_L:
        cross_comp(graph)
        # selective connectivity clustering between exemplars, extrapolated to their node_

def sum_G_(G, node_, fc=0):
    for n in node_:
        if fc:
            s = n.sign; n.sign = 1  # single-use
        else: s = 1
        G.latuple += n.latuple * s
        G.vert = G.vert + n.vert*s if np.any(G.vert) else deepcopy(n.vert) * s
        G.Et += n.Et * s; G.aRad += n.aRad * s
        G.yx += n.yx * s
        if n.derH: add_H(G.derH, n.derH, root=G, rev = s==-1, fc=fc)
        if fc:
            G.M += n.m * s; G.L += s
        else:
            if n.extH: add_H(G.extH, n.extH, root=G, rev = s==-1)  # empty in centroid
            G.box = extend_box( G.box, n.box)  # extended per separate node_ in centroid

def comb_altG_(G):  # combine contour G.altG_ into altG (node_ defined by root=G), for agg+ cross-comp

    if G.altG:
        if isinstance(G.altG, list):
            sum_G_(G.altG[0], [a for a in G.altG[1:]])
            G.altG = CG(root=G, node_=G.altG); G.altG.sign = 1; G.altG.m = 0
            # alt D * G rM:
            if val_(G.altG.Et, _Et=G.Et):
                cross_comp(G.altG)
    else:
        # sum neg links into CG
        altG = CG(root=G, node_=[],link_=[]); altG.sign = 1; altG.m = 0
        derH = []
        for link in G.derH[-1].link_:
            if val_(link.Et, _Et=G.Et) > 0:  # neg link
                altG.link_ += [link]
                for n in link.nodet:
                    if n not in altG.node_:
                        altG.node_ += [n]  # altG.box = extend_box(altG.box, n.box)  same as in G?
                        altG.latuple += n.latuple; altG.vert += n.vert
                        if n.derH:
                            add_H(derH, n.derH, root=altG)
                        altG.Et += n.Et * icoef ** 2
        if altG.link_:
            altG.derH += [sum_H(altG.link_,altG)]  # sum link derH
            G.altG = altG

if __name__ == "__main__":
    image_file = './images/raccoon_eye.jpeg'
    image = imread(image_file)
    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vectorize_root(frame)
    if frame.node_:  # converted edges
        G_ = []
        for edge in frame.node_:
            comb_altG_(edge)
            cluster_C_(edge)  # no cluster_C_ in vect_edge
            G_ += edge.derH[-1].node_  # unpack edge, or keep if connectivity cluster, or in flat blob altG_?
        frame.node_ = G_
        cross_comp(frame)  # calls connectivity clustering