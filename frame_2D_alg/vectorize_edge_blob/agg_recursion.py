import sys
sys.path.append("..")
import numpy as np
from copy import copy, deepcopy
from functools import reduce
from itertools import combinations
from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_
from comp_slice import comp_latuple, comp_md_
from trace_edge import comp_N, comp_node_, comp_link_, sum2graph, get_rim, CH, CG, ave, ave_d, ave_L, vectorize_root, comp_area, extend_box, ave_rn
'''
Cross-compare and cluster edge blobs within a frame,
potentially unpacking their node_s first,
with recursive agglomeration
'''

def agg_cluster_(frame):  # breadth-first (node_,L_) cross-comp, clustering, recursion

    def cluster_eval(G, N_, fd):

        pL_ = {l for n in N_ for l,_ in get_rim(n, fd)}
        if len(pL_) > ave_L:
            sG_ = cluster_N_(G, pL_, fd)  # optionally divisive clustering
            frame.subG_ = sG_
            for sG in sG_:
                if len(sG.node_) > ave_L:
                    get_exemplar_(sG)  # centroid clustering in sG.node_
    '''
    cross-comp converted edges, then GGs, GGGs, etc, interlaced with exemplar selection 
    '''
    for edge in frame.subG_: get_exemplar_(edge)
    # also selectively unpack edges, cross-comp exemplars instead? re-comp within edge too?

    N_,L_,(m,d,r) = comp_node_(frame.subG_)  # exemplars, extrapolate to their Rims?
    if m > ave * r:
        mlay = CH().add_H([L.derH for L in L_])  # mfork, else no new layer
        frame.derH = CH(H=[mlay], md_t=deepcopy(mlay.md_t), n=mlay.n, root=frame); mlay.root=frame.derH
        frame.derH.Et = copy(mlay.Et)  # or make it default param again?
        vd = d - ave_d * r
        if vd > 0:
            for L in L_:
                L.root_ = [frame]; L.extH = CH(); L.rimt = [[],[]]
            lN_,lL_,md = comp_link_(L_)  # comp new L_, root.link_ was compared in root-forming for alt clustering
            vd *= md / ave
            frame.derH.append_(CH().add_H([L.derH for L in lL_]))  # dfork
            # recursive der+ eval_: cost > ave_match, add by feedback if < _match?
        else:
            frame.derH.H += [[]]  # empty to decode rng+|der+, n forks per layer = 2^depth
        # + aggLays and derLays, with exemplar selection:
        cluster_eval(frame, N_, fd=0)
        if vd > 0: cluster_eval(frame, lN_, fd=1)


def cluster_N_(root, L_, fd, nest=1):  # top-down segment L_ by >ave ratio of L.dists

    ave_rL = 1.2  # defines segment and cluster

    L_ = sorted(L_, key=lambda x: x.dist, reverse=True)  # lower-dist links
    _L = L_[0]
    N_, et = {*_L.nodet}, _L.derH.Et
    # current dist segment:
    for i, L in enumerate(L_[1:], start=1):  # long links first
        rel_dist = _L.dist / L.dist  # >1
        if rel_dist < ave_rL or et[0] < ave or len(L_[i:]) < ave_L:  # ~=dist Ns or either side of L is weak
            _L = L; et += L.derH.Et
            for n in L.nodet: N_.add(n)  # in current dist span
        else: break  # terminate contiguous-distance segment
    min_dist = _L.dist
    Gt_ = []
    for N in N_:  # cluster current distance segment
        if len(N.root_) > nest: continue  # merged, root_[0] = edge
        node_,link_, et = set(),set(), np.array([.0,.0,1.])
        Gt = [node_,link_,et, min_dist]; N.root_ += [Gt]
        _eN_ = {N}
        while _eN_:
            eN_ = set()
            for eN in _eN_:  # cluster rim-connected ext Ns, all in root Gt
                node_.add(eN)  # of all rim
                eN.root_ += [Gt]
                for L,_ in get_rim(eN, fd):
                    if L not in link_:
                        # if L.derH.Et[0]/ave * n.extH m/ave or L.derH.Et[0] + n.extH m*.1: density?
                        eN_.update([n for n in L.nodet if len(n.root_) <= nest])
                        if L.dist >= min_dist:
                            link_.add(L); et += L.derH.Et
            _eN_ = eN_
        sub_L_ = set()  # form subG_ from shorter-L seg per Gt, depth-first:
        for N in node_:  # cluster in N_
            sub_L_.update({l for l,_ in get_rim(N,fd) if l.dist < min_dist})
        if len(sub_L_) > ave_L:
            subG_ = cluster_N_(Gt, sub_L_, fd, nest+1)
        else: subG_ = []
        Gt += [subG_]; Gt_ += [Gt]
    G_ = []
    for Gt in Gt_:
        node_, link_, et, minL, subG_ = Gt; Gt[0] = list(node_)
        if et[0] > et[2] *ave *nest:  # rdn incr/ dist decr
            G_ += [sum2graph(root, Gt, fd, nest)]
        else:
            for n in node_: n.root_.pop()
    return G_

'''
 Connectivity clustering terminates at effective contours: alt_Gs, beyond which cross-similarity is not likely to continue. 
 Next cross-comp is discontinuous and should be selective for well-defined clusters: stable and likely recurrent.
 
 Such exemplar selection is by global similarity in centroid clustering, vs. transitive similarity in connectivity clustering.
 It's a compressive learning phase, while connectivity clustering is generative, forming new derivatives and composition levels.
 
 Centroid clusters may be extended, but only their exemplars will be cross-compared on the next connectivity clustering level.
 Other nodes in the cluster can be predicted from the exemplar, they are the same type of objects. 
'''
def get_exemplar_(graph):

    def centroid(dnode_, node_, C=None):  # sum|subtract and average Rim nodes

        if C is None: C = CG()
        for n in dnode_:
            s = n.sign; n.sign=1  # single-use sign
            C.n += n.n * s; C.Et += n.Et * s; C.rng = n.rng * s; C.aRad += n.aRad * s
            C.latuple += n.latuple * s; C.mdLay += n.mdLay * s
            if n.derH: C.derH.add_H(n.derH, sign=s)
            if n.extH: C.extH.add_H(n.extH, sign=s)
        # get averages:
        k = len(dnode_); C.n/=k; C.Et/=k; C.latuple/=k; C.mdLay/=k; C.aRad/=k
        if C.derH: C.derH.norm_(k)  # derH/=k
        C.box = reduce(extend_box, (n.box for n in node_))
        C.M = 0  # summed match to nodes
        C.L = 0  # ave len node_
        return C

    def comp_C(C, N):  # compute match without new derivatives: global cross-comp is not directional

        rn = C.n / N.n
        mL = min(C.L,len(N.node_)) - ave_L
        mA = comp_area(C.box, N.box)[0]
        mLat = comp_latuple(C.latuple, N.latuple, rn,fagg=1)[1][0]
        mLay = comp_md_(C.mdLay[0], N.mdLay[0], rn)[1][0]
        mH = C.derH.comp_H(N.derH, rn).Et[0] if C.derH and N.derH else 0
        # comp node_, comp altG from converted adjacent flat blobs?
        m = mL+mA+mLat+mLay+mH
        C.M += m
        return m

    def centroid_cluster(N, clustered_):  # refine and extend cluster with extN_

        _N_ = {n for L in N.Rim for n in L.nodet}
        n_ = _N_| {N}  # include seed node
        C = centroid(n_,n_)
        while True:
            N_, negN_, extN_, M, dM = [],[],[],0,0  # included, removed, extended nodes
            for _N in _N_:
                if _N in clustered_: continue
                m = comp_C(C,_N)
                dm = m - ave
                if dm > 0:
                    extN_ += [link.nodet[0] if link.nodet[1] is _N else link.nodet[1] for link in _N.rim]  # next comp to C
                    N_ += [_N]; M += m; _N.M = m  # to sum in C
                    if _N not in C.node_:
                        dM += dm; clustered_ += [_N]  # only new nodes
                elif _N in C.node_:
                    _N.sign=-1; negN_+=[_N]; dM += -dm  # to subtract from C, dM += abs dm
                    clustered_.remove(_N)  # if exclusive
            if dM > ave:  # new match, terminate (refine,extend) if low
                extN_ = set([eN for eN in extN_ if eN not in clustered_])
                C = centroid( extN_|set(negN_), N_, C)
                _N_ = set(N_)|extN_  # both old and new nodes will be compared to new C
                C.M = M; C.node_ = N_
            else:
                if C.M > ave * 10:
                    for n in C.node_:
                        n.root_ += [C]; delattr(n, "sign")
                    return C  # centroid cluster
                else:  # unpack C.node_
                    for n in C.node_:
                        clustered_.remove(n); n.M = 0
                    return N  # keep seed node

    N_ = graph.subG_  # complemented Gs: m-type core + d-type contour, initially edge
    N_ = sorted(N_, key=lambda n: n.Et[0], reverse=True)
    subG_, clustered_ = [], set()
    for N in N_: N.sign = 1
    for i, N in enumerate(N_):  # connectivity cluster may have exemplar centroids
        if N not in clustered_:
            if N.Et[0] > ave * 10:
                subG_ += [centroid_cluster(N, clustered_)]  # extend from N.rim, return C if packed else N
            else:  # the rest of N_ M is lower
                subG_ += N_[i:]
                break
    graph.subG_ = subG_  # mix of Ns and Cs: exemplars of their network?
    if len(graph.subG_) > ave_L:
        agg_cluster_(graph)  # selective connectivity clustering between exemplars, extrapolated to their node_


if __name__ == "__main__":
    image_file = '../images/raccoon_eye.jpeg'
    image = imread(image_file)
    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vectorize_root(frame)
    if frame.subG_:  # converted edges
        get_exemplar_(frame)  # selects connectivity-clustered edges for agg_cluster_