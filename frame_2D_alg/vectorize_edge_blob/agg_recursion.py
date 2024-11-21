import sys
sys.path.append("..")
import numpy as np
from copy import copy, deepcopy
from itertools import combinations
from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_
from comp_slice import comp_latuple, comp_md_
from vectorize_edge import comp_node_, comp_link_, sum2graph, get_rim, CH, CG, ave, ave_d, ave_L, vectorize_root, comp_area, extend_box, ave_rn
'''
Cross-compare and cluster edge blobs within a frame,
potentially unpacking their node_s first,
with recursive agglomeration
'''

def agg_recursion(frame):  # breadth-first (node_,L_) cross-comp, clustering, recursion

    def cluster_eval(frame, N_, fd):

        pL_ = {l for n in N_ for l,_ in get_rim(n, fd)}
        if len(pL_) > ave_L:
            G_ = cluster_N_(frame, pL_, fd)  # optionally divisive clustering
            frame.subG_ = G_
            if len(G_) > ave_L:
                agg_recursion(frame)  # cross-comp higher subGs
    '''
    cross-comp converted edges, then GGs, GGGs., 
    interlace with medoid clustering?
    '''
    iN_ = []  # eval to unpack top N:
    for N in frame.subG_: iN_ += [N] if N.derH.Et[0] < ave * N.derH.Et[2] else N.subG_

    N_,L_,(m,d,mr,dr) = comp_node_(iN_)
    if m > ave * mr:
        mlay = CH().add_H([L.derH for L in L_])  # mfork, else no new layer
        frame.derH = CH(H=[mlay], md_t=deepcopy(mlay.md_t), Et=copy(mlay.Et), n=mlay.n, root=frame); mlay.root=frame.derH  # init
        vd = d * (m/ave) - ave_d * dr  # proportional borrow from co-projected m: proj_m = m - vd/2: bidir, no rdn: d is secondary?
        if vd > 0:
            for L in L_:
                L.root_ = [frame]; L.extH = CH(); L.rimt = [[],[]]
            lN_,lL_,md = comp_link_(L_)  # comp new L_, root.link_ was compared in root-forming for alt clustering
            vd *= md / ave  # no full et?
            frame.derH.append_(CH().add_H([L.derH for L in lL_]))  # dfork
            # recursive der+ eval_: cost > ave_match, add by feedback if < _match?
        else:
            frame.derH.H += [[]]  # empty to decode rng+|der+, n forks per layer = 2^depth
        # + aggLays and derLays:
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
        node_,link_, et = set(),set(), np.zeros(4)
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
 Next cross-comp is discontinuous and global, thus centroid-based, for well-defined (likely stable and recurrent) clusters.
 They alternate: 
 connectivity clustering by transitive similarity is a generative phase, forming new derivatives in derH, 
 while centroid clustering purely by similarity is a compressive phase, no new parameters are formed:
'''
def MCluster_(frame):

    def comp_cN(_N, N):  # compute match without new derivatives: relation to the medoid is not directional

        rn = _N.n / N.n
        mL = min(len(_N.node_),len(N.node_)) - ave_L
        mA = comp_area(_N.box, N.box)[0]
        mLat = comp_latuple(_N.latuple,N.latuple,rn,fagg=1)[1][0]
        mLay = comp_md_(_N.mdLay[0], N.mdLay[0], rn, dir)[1][0]
        mH = _N.derH.comp_H(N.derH, rn, dir=dir).Et[0] if _N.derH and N.derH else 0

        return mL + mA + mLat + mLay + mH

    def medoid(N_):  # sum and average nodes

        N = CG()
        for n in N_:
            N.n += n.n; N.rng = n.rng; N.aRad += n.aRad
            N.box = extend_box(N.box, n.box)
            N.latuple += n.latuple; N.mdLay += n.mdLay
            N.derH.add_H(n.derH); N.extH.add_H(n.extH)
        # get averages:
        k = len(N_)
        N.n /= k; N.latuple /= k; N.mdLay /= k; N.aRad /= k; N.derH.norm(k)  # derH/= k
        return N

    def xcomp_(N_):  # initial global cross-comp

        for _G, G in combinations(N_, r=2):
            rn = _G.n/G.n
            if rn > ave_rn: continue  # scope disparity
            M = comp_cN(_G, G, rn)  # to be modified
            vM = M - ave
            for _g,g in (_G,G),(G,_G):
                if vM > 0:
                    g.perim.add(g)  # loose match, new param for now
                    if vM > ave: g.crim.add([G,M])  # strict match

    def cluster_(N_):  # initial flood fill via N.crim
        clust_ = []
        while N_:
            node_, peri_, M = set(), set(), 0
            N = N_.pop(); _ref_ = [[N, 0]]
            while _ref_:
                ref_ = set()
                for _ref in _ref_:
                    _eN = _ref[0]
                    node_.add(_ref)
                    peri_.update(_eN.perim)  # for comp to medoid in refine_
                    for ref in _eN.crim:
                        eN, eM = ref
                        if eN not in N_: continue
                        ref_.add(ref); M += eM
                        N_.remove(eN)  # merge
                _ref_ = ref_
            Ct = [node_,peri_,M]
            clust_ += [Ct]
        return clust_

    def refine_(clust):
        _node_, _peri_, _M = clust
        dM = ave + 1
        while dM > ave:
            node_, peri_, M = set(), set(), 0
            _N = medoid(node_)
            for N in peri_:
                M = comp_cN(_N, N)
                vM = M - ave
                for _n, n in (_N,N),(N,_N):
                    if vM > 0:
                        peri_.add(n)  # _n.perim.add(n)  # not needed?
                        if vM > ave:
                            node_.add(n)  # _n.crim.add([n, M])
            dM = M-_M
            _node_,_peri_,_M = node_,peri_,M

        clust[:] = [node_,peri_,M]  # final cluster

    # this is simplified, frame.subG_ should contain complemented graphs: m-type core + d-type contour
    N_ = frame.subG_
    xcomp_(N_)
    clust_ = cluster_(N_)
    for clust in clust_: refine_(clust)
        # re-cluster by node_ medoid
    frame.clust_ = clust_  # new param for now


if __name__ == "__main__":
    image_file = '../images/raccoon_eye.jpeg'
    image = imread(image_file)
    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vectorize_root(frame)
    if frame.subG_:  # converted edges
        agg_recursion(frame)