import sys
sys.path.append("..")
import numpy as np
from copy import copy, deepcopy
from itertools import combinations
from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_
from comp_slice import comp_latuple, comp_md_
from trace_edge import comp_node_, comp_link_, sum2graph, get_rim, CH, CG, ave, ave_d, ave_L, vectorize_root, comp_area, extend_box, ave_rn
'''
Cross-compare and cluster edge blobs within a frame,
potentially unpacking their node_s first,
with recursive agglomeration
'''

def agg_cluster_(frame):  # breadth-first (node_,L_) cross-comp, clustering, recursion

    def cluster_eval(frame, N_, fd):

        pL_ = {l for n in N_ for l,_ in get_rim(n, fd)}
        if len(pL_) > ave_L:
            G_ = cluster_N_(frame, pL_, fd)  # optionally divisive clustering
            frame.subG_ = G_
            if len(G_) > ave_L:
                med_cluster_(frame)  # alternating medoid clustering
    '''
    cross-comp converted edges, then GGs, GGGs., interlaced connectivity clustering and exemplar selection
    '''
    iN_ = []
    for N in frame.subG_:  # eval to unpack top N:
        iN_ += [N] if N.derH.Et[0] < ave * N.derH.Et[2] else N.subG_
        # use exemplar_ instead of full subG_, extend new graphs with exemplar.crim?

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
        # + aggLays and derLays, each calls exemplar selection:
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
 Next cross-comp is discontinuous and global, thus medoid-based, for well-defined (stable and likely recurrent) clusters.
 
 So connectivity clustering by transitive similarity is a generative phase, forming new graphs with derivatives in derH, 
 while medoid clustering purely by similarity is a compressive phase: no new params and composition levels are formed.
 
 Med_cluster may be extended, but only its exemplar node (with max m to medoid) will be a sample for next cross-comp.
 Other nodes interactions can be predicted from the exemplar, they are supposed to be the same type of objects. 
'''
def med_cluster_(frame):

    def comp_cN(_N, N):  # compute match without new derivatives: relation to the medoid is not directional

        rn = _N.n / N.n
        mL = min(len(_N.node_),len(N.node_)) - ave_L
        mA = comp_area(_N.box, N.box)[0]
        mLat = comp_latuple(_N.latuple,N.latuple,rn,fagg=1)[1][0]
        mLay = comp_md_(_N.mdLay[0], N.mdLay[0], rn)[1][0]
        mH = _N.derH.comp_H(N.derH, rn).Et[0] if _N.derH and N.derH else 0

        return mL + mA + mLat + mLay + mH

    def medoid(N_):  # sum and average nodes

        N = CG(root_=[None])
        for n in N_:
            N.n += n.n; N.rng = n.rng; N.aRad += n.aRad
            N.box = extend_box(N.box, n.box)
            N.latuple += n.latuple; N.mdLay += n.mdLay
            N.derH.add_H(n.derH); N.extH.add_H(n.extH)
        # get averages:
        k = len(N_)
        N.n /= k; N.latuple /= k; N.mdLay /= k; N.aRad /= k; N.derH.norm_(k)  # derH/= k
        return N

    def xcomp_(N_):  # initial global cross-comp

        for _G, G in combinations(N_, r=2):
            rn = _G.n/G.n
            if rn > ave_rn: continue  # scope disparity
            M = comp_cN(_G, G)
            vM = M - ave
            for _g,g in (_G,G),(G,_G):
                if vM > 0:
                    g.perim.add(_g)  # loose match
                    if vM > ave:
                        g.crim.add((_g,M))  # strict match
                        g.M += M

    def get_exemplar_(N_):  # select Ns with M > ave * Mr

        exemplar_ = []
        for N in N_:
            for _N, M in N.crim:
                if N.M >_N.M: _N.M -= M
                else:          N.M -= M
                # exclusive representation, or incr N.Mr?
            if N.M > ave:
                exemplar_ += [N]

        return exemplar_

    def refine_(exemplar):
        _node_, _peri_, _M = exemplar.crim, exemplar.perim, exemplar.M

        dM = ave + 1
        while dM > ave:
            node_, peri_, M = set(), set(), 0
            mN = medoid(_node_)
            for _N,_m in _peri_:
                m = comp_cN(mN,_N)
                if M > ave:
                    peri_.add((_N,m))
                    if M > ave*2: node_.add((_N,m))
            dM = M - _M
            _node_,_peri_,_M = node_,peri_,M

        exemplar.crim, exemplar.perim, exemplar.M = list(node_), list(peri_), M  # final cluster

    N_ = frame.subG_  # should be complemented graphs: m-type core + d-type contour
    for N in N_:
        N.perim = set(); N.crim = set(); N.root_ += [frame]
    xcomp_(N_)
    exemplar_ = get_exemplar_(N_)  # select strong Ns
    for N in exemplar_:
        if N.M > ave * 10:  # tentative, else keep N.crim
            refine_(N)  # N.crim = N.perim clustered by N.crim medoid
    frame.exemplar_ = exemplar_
    if len(frame.exemplar_) > ave_L:
        agg_cluster_(frame)  # alternating connectivity clustering per exemplar, more selective


if __name__ == "__main__":
    image_file = '../images/raccoon_eye.jpeg'
    image = imread(image_file)
    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vectorize_root(frame)
    if frame.subG_:  # converted edges
        # start from medoid clustering because edges are connectivity-clustered
        med_cluster_(frame)  # starts alternating agg_recursion