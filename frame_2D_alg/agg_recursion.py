
import numpy as np
from copy import copy, deepcopy
from functools import reduce
from frame_blobs import frame_blobs_root, intra_blob_root, imread
from comp_slice import comp_latuple, comp_md_
from vect_edge import feedback, comp_node_, comp_link_, sum2graph, get_rim, CH, CG, ave, ave_L, vectorize_root, comp_area, extend_box, val_
'''
Cross-compare and cluster Gs within a frame, potentially unpacking their node_s first,
alternating agglomeration and centroid clustering.
notation:
prefix  f denotes flag
postfix t denotes tuple, multiple ts is a nested tuple
prefix  _ denotes prior of two same-name variables, multiple _s for relative precedence
postfix _ denotes array of same-name elements, multiple _s is nested array
capitalized vars are summed small-case vars 
'''

def cross_comp(root):  # breadth-first node_,link_ cross-comp, connect.clustering, recursion

    N_,L_,Et = comp_node_(root.subG_)  # cross-comp exemplars, extrapolate to their node_s
    # mfork
    if val_(Et, fo=1) > 0:
        mlay = CH().add_tree([L.derH for L in L_]); H=root.derH; mlay.root=H; H.Et += mlay.Et; H.lft = [mlay]
        pL_ = {l for n in N_ for l,_ in get_rim(n, fd=0)}
        if len(pL_) > ave_L:
            cluster_N_(root, pL_, fd=0)  # nested distance clustering, calls centroid and higher connect.clustering
        # dfork
        if val_(Et, mEt=Et, fo=1) > 0:  # same root for L_, root.link_ was compared in root-forming for alt clustering
            for L in L_:
                L.extH, L.root, L.Et, L.mL_t, L.rimt, L.aRad, L.visited_ = CH(),root,copy(L.derH.Et), [[],[]], [[],[]], 0,[L]
            lN_,lL_,dEt = comp_link_(L_,Et)
            if val_(dEt, mEt=Et, fo=1) > 0:
                dlay = CH().add_tree([L.derH for L in lL_]); dlay.root=H; H.Et += dlay.Et; H.lft += [dlay]
                plL_ = {l for n in lN_ for l,_ in get_rim(n, fd=1)}
                if len(plL_) > ave_L:
                    cluster_N_(root, plL_, fd=1)

        feedback(root)  # add root derH to higher roots derH

def cluster_N_(root, L_, fd, nest=0):  # top-down segment L_ by >ave ratio of L.dists

    L_ = sorted(L_, key=lambda x: x.dist)  # shorter links first
    for n in [n for l in L_ for n in l.nodet]: n.fin = 0
    _L = L_[0]
    N_, et = copy(_L.nodet), _L.derH.Et
    L_ = L_[1:]
    while L_:  # longer links
        for i, L in enumerate(L_):  # short first
            rel_dist = L.dist / _L.dist  # >= 1
            if rel_dist < 1.2 or val_(et)>0 or len(L_[i:]) < ave_L:  # ~=dist Ns or either side of L is weak
                _L = L; N_ += L.nodet; et += L.derH.Et  # last L
            else:
                i -= 1; break  # terminate contiguous-distance segment
        G_ = []
        max_dist = _L.dist
        for N in {*N_}:  # cluster current distance segment
            _eN_, node_,link_, et, = [N], [],[], np.zeros(4)
            while _eN_:
                eN_ = []
                for eN in _eN_:  # cluster rim-connected ext Ns, all in root Gt
                    node_+=[eN]; eN.fin = 1  # all rim
                    for L,_ in get_rim(eN, fd):
                        if L not in link_:  # if L.derH.Et[0]/ave * n.extH m/ave or L.derH.Et[0] + n.extH m*.1: density?
                            eN_ += [n for n in L.nodet if nest or not n.fin]  # filter in 1st dist seg only
                            if L.dist < max_dist:
                                link_+=[L]; et+=L.derH.Et
                _eN_ = {*eN_}
            G_ += [sum2graph(root, [list({*node_}),list({*link_}), et], fd, max_dist, nest)]
            # cluster node roots if nest else nodes
        nest += 1
        if fd: root.link_ = G_  # replace with current-dist clusters
        else:  root.node_ = G_
        L_ = L_[i+1:]  # get longer links if any, may connect current-dist clusters
        N_ = []
    cluster_C_(root)

''' Hierarchical clustering should alternate between two phases: generative via connectivity and compressive via centroid.

 Connectivity clustering terminates at effective contours: alt_Gs, beyond which cross-similarity is not likely to continue. 
 Next cross-comp is discontinuous and should be selective, for well-defined clusters: stable and likely recurrent.
 
 Such clusters should be compared and clustered globally: via centroid clustering, vs. local connectivity clustering.
 Only centroids (exemplars) need to be cross-compared on the next connectivity clustering level, representing their nodes.
 
 So connectivity clustering is a generative learning phase, forming new derivatives and structured composition levels, 
 while centroid clustering is a compressive phase, reducing multiple similar comparands to a single exemplar. '''

def cluster_C_(graph):

    def centroid(dnode_, node_, C=None):  # sum|subtract and average Rim nodes

        if C is None:
            C = CG(); C.L = 0; C.M = 0  # setattr summed len node_ and match to nodes
        for n in dnode_:
            s = n.sign; n.sign=1  # single-use sign
            C.Et += n.Et * s; C.rng = n.rng * s; C.aRad += n.aRad * s
            C.L += len(n.node_) * s
            C.latuple += n.latuple * s
            C.vert += n.vert * s
            C.yx += n.yx
            if n.derH: C.derH.add_tree(n.derH, rev = s==-1, fc=1)
            if n.extH: C.extH.add_tree(n.extH, rev = s==-1, fc=1)
        # get averages:
        k = len(dnode_); C.Et/=k; C.latuple/=k; C.vert/=k; C.aRad/=k; C.yx /= k
        if C.derH: C.derH.norm_(k)  # derH/=k
        C.box = reduce(extend_box, (n.box for n in node_))
        return C

    def comp_C(C, N):  # compute match without new derivatives: global cross-comp is not directional

        mL = min(C.L,len(N.node_)) - ave_L
        mA = comp_area(C.box, N.box)[0]
        mLat = comp_latuple(C.latuple, N.latuple, C.Et[2], N.Et[2])[1][0]
        mVert = comp_md_(C.vert[1], N.vert[1])[1][0]
        mH = C.derH.comp_tree(N.derH).Et[0] if C.derH and N.derH else 0
        # comp node_, comp altG from converted adjacent flat blobs?
        return mL + mA + mLat + mVert + mH

    def centroid_cluster(N):  # refine and extend cluster with extN_

        _N_ = {n for L,_ in N.rim for n in L.nodet if not n.fin}
        N.fin = 1; n_ = _N_| {N}  # include seed node
        C = centroid(n_,n_)
        while True:
            N_,dN_,extN_, M, dM, extM = [],[],[], 0,0,0  # included, changed, queued nodes and values
            for _N in _N_:
                m = comp_C(C,_N)
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
                if dN_: # recompute C if any changes in node_
                    C = centroid(set(dN_), N_, C)
                _N_ = set(N_) | set(extN_)  # next loop compares both old and new nodes to new C
                C.M = M; C.node_ = N_
            else:
                if C.M > ave * 10:
                    C.root = N.root  # C.nest = N.nest+1
                    for n in C.node_:
                        n.root = C; n.fin = 1; delattr(n,"sign")
                    return C  # centroid cluster
                else:  # unpack C.node_
                    for n in C.node_: n.m = 0
                    return N  # keep seed node

    # get representative centroids of complemented Gs: mCore + dContour, initially in unpacked edges
    N_ = sorted([N for N in graph.node_ if any(N.Et)], key=lambda n: n.Et[0], reverse=True)
    G_ = []
    for N in N_:
        N.sign, N.m, N.fin = 1, 0, 0  # setattr: C update sign, inclusion val, prior C inclusion flag
    for i, N in enumerate(N_):  # replace some of connectivity cluster by exemplar centroids
        if not N.fin:  # not in prior C
            if N.Et[0] > ave * 10:
                G_ += [centroid_cluster(N)]  # extend from N.rim, return C if packed else N
            else:  # the rest of N_ M is lower
                G_ += [N for N in N_[i:] if not N.fin]
                break
    # draft:
    for G in G_:
        #  cross-comp | sum altG_ -> combined altG before next agg+ cross-comp
        if val_(np.sum([alt.Et for alt in G.altG_]), mEt=G.Et):
            G.altG_ = cross_comp(G, G.altG_)
        else:
            G.altG_ = reduce(sum_G_, [alt for alt in G.altG_])

    graph.node_ = G_  # mix of Ns and Cs: exemplars of their network?
    if len(G_) > ave_L:
        cross_comp(graph)  # selective connectivity clustering between exemplars, extrapolated to their node_

def sum_G_(node_):
    G = CG()
    for n in node_:
        G.rng = n.rng; G.latuple += n.latuple; G.vert += n.vert; G.aRad += n.aRad; G.box = extend_box(G.box, n.box)
        if n.derH: G.derH.add_tree(n.derH, root=G)
        if n.extH: G.extH.add_tree(n.extH)
    return G

if __name__ == "__main__":
    image_file = './images/raccoon_eye.jpeg'
    image = imread(image_file)
    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vectorize_root(frame)
    if frame.subG_:  # converted edges
        G_ = []
        for edge in frame.node_:
            cluster_C_(edge)  # no cluster_C_ in vect_edge
            G_ += edge.node_  # unpack edge, or keep if connectivity cluster, or in flat blob altG_?
        frame.node_ = G_
        cross_comp(frame)  # calls connectivity clustering