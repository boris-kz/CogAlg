import numpy as np
import sys
sys.path.append("..")
from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_
from slice_edge import CP, slice_edge, comp_angle, aveG
from functools import reduce
'''
comp_slice traces edge axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These are low-M high-Ma blobs, vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)
-
Vectorization is clustering of parameterized Ps + their derivatives (derPs) into PPs: patterns of Ps that describe edge blob.
This process is a reduced-dimensionality (2D->1D) version of cross-comp and clustering cycle, common across this project.

PP clustering in vertical (along axis) dimension is contiguous and exclusive because 
Ps are relatively lateral (cross-axis), their internal match doesn't project vertically. 

Primary clustering by match between Ps over incremental distance (rng++), followed by forming overlapping
Secondary clusters of match of incremental-derivation (der++) difference between Ps. 

As we add higher dimensions (3D and time), this dimensionality reduction is done in salient high-aspect blobs
(likely edges in 2D or surfaces in 3D) to form more compressed 'skeletal' representations of full-dimensional patterns.

comp_slice traces edge blob axis by cross-comparing vertically adjacent Ps: slices across edge blob, along P.G angle.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)

Connectivity in P_ is traced through root_s of derts adjacent to P.dert_, possibly forking. 
len prior root_ sorted by G is root.rdn, to eval for inclusion in PP or start new P by ave*rdn
'''

ave_dI = ave_inv = 20  # ave inverse m, change to Ave from the root intra_blob?
ave, ave_d = 5, 5  # ave direct m, change to Ave_min from the root intra_blob?
aves = ave, ave_mG, ave_mM, ave_mMa, ave_mA, ave_mL = 5, 10, 2, .1, .2, 2
PP_aves = ave_PPm, ave_PPd = 50, 50
P_aves = ave_Pm, ave_Pd = 10, 10
ave_Gm = 50
ave_L = 5

class CdP(CBase):  # produced by comp_P, comp_slice version of Clink
    name = "dP"

    def __init__(l, nodet, span, angle, yx, vertuple=None, latuple=None, Et=None, root=None):
        super().__init__()

        l.nodet = nodet  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.latuple = np.array([.0,.0,.0,.0,.0, np.zeros(2)], dtype=object) if latuple is None else latuple  # sum node_
        l.vertuple = np.array([np.zeros(6), np.zeros(6)]) if vertuple is None else vertuple  # m_,d_
        l.angle = angle  # dy,dx between node centers
        l.span = span  # distance between node centers
        l.yx = yx  # sum node_
        l.Et = np.zeros(2) if Et is None else Et
        l.root = root  # PPds containing dP
        l.lrim = []
        l.prim = []
        # l.med = 0  # comp rng: n of mediating Ps between node_ Ps
        # n = 1?
    def __bool__(l):
        return np.any(l.vertuple)

def vectorize_root(frame):

    blob_ = unpack_blob_(frame)
    for blob in blob_:
        if not blob.sign and blob.G > aveG * blob.root.rdn:
            edge = slice_edge(blob)
            if edge.G*(len(edge.P_) - 1) > ave_PPm:  # eval PP, rdn=1
                for P in edge.P_: P.mdLay = np.array([np.zeros(6), np.zeros(6)])
                comp_slice(edge)

def comp_slice(edge):  # root function

    edge.Et, edge.vertuple = np.zeros(2), np.array([np.zeros(6), np.zeros(6)])  # m_,d_
    for P in edge.P_:  # add higher links
        P.vertuple = np.array([np.zeros(6), np.zeros(6)])
        P.rim = []; P.lrim = []; P.prim = []

    comp_P_(edge)  # vertical P cross-comp -> PP clustering, if lateral overlap
    edge.node_ = form_PP_(edge, edge.P_, fd=0)

    for PPm in edge.node_:  # eval sub-clustering, not recursive
        if isinstance(PPm, list):  # PPt, not CP
            P_, link_, vertuple = PPm[1:4]
            if len(link_) > ave_L and sum(vertuple[1]) > ave_PPd:
                comp_dP_(PPm)
                PPm[2] = form_PP_(PPm, link_, fd=1)  # add PPds within PPm link_
            vertuple = PPm[3]
        else:
            vertuple = PPm.vertuple  # PPm is actually CP
        edge.vertuple += vertuple
        edge.Et[0] += sum(vertuple[0]); edge.Et[1] += sum(vertuple[1])

def comp_P_(edge):  # form links from prelinks

    edge.rng = 1
    for P, _pre_ in edge.pre__.items():
        for _P in _pre_:
            # prelinks
            dy,dx = np.subtract(P.yx,_P.yx)  # between node centers
            if abs(dy)+abs(dx) <= edge.rng * 2:
                # <max Manhattan distance
                angle=[dy,dx]; distance=np.hypot(dy,dx)
                derLay, et = comp_latuple(_P.latuple, P.latuple, len(_P.dert_), len(P.dert_))
                P.rim += [convert_to_dP(_P,P, derLay, angle, distance, et)]
    del edge.pre__

def comp_dP_(PP):  # node_- mediated: comp node.rim dPs, call from form_PP_

    link_ = PP[2]
    llink_ = []  # links between links
    for _dP in link_:
        M, n = PP[-2][0], PP[-1]
        if _dP.Et[1] * (M / n / ave) > aves[1]:
            for dP in P.rim:  # higher links
                if dP not in link_: continue  # skip removed node links
                _P, P = dP.nodet  # _P is lower
                rn = len(P.dert_) / len(_P.dert_)
                # do we need this:
                _node, node = (_dP, dP) if (*_dP.yx,) < (*dP.yx,) else (dP, _dP)
                mdVer, et = comp_md_(_node.vertuple[1], node.vertuple[1], rn)
                angle = np.subtract(node.yx,_node.yx)  # dy,dx of node centers
                distance = np.hypot(*angle)  # between node centers
                llink_ += [convert_to_dP(_node, node, mdVer, angle, distance, et)]
    return llink_

def comp_md_(_d_,d_, rn=.1, dir=1):  # dir may be -1

    d_ = d_*rn  # normalize by compared accum span
    dd_ = (_d_ - d_ * dir)  # np.arrays
    md_ = np.minimum(np.abs(_d_), np.abs(d_))
    md_[(_d_ < 0) != (d_ < 0)] *= -1  # negate if only one compared is negative

    return np.array([np.array(md_), np.array(dd_)]), np.array([md_.sum(), dd_.sum()])  # [m_,d_], Et

def convert_to_dP(_P,P, derLay, angle, distance, Et):

    link = CdP(nodet=[_P,P], Et=Et, vertuple=derLay, angle=angle, span=distance, yx=np.add(_P.yx, P.yx)/2)
    # regardless of clustering:
    _P.vertuple += link.vertuple; P.vertuple += link.vertuple
    _P.lrim += [link]; P.lrim += [link]
    _P.prim += [P];    P.prim +=[_P]  # all Ps are dPs if fd

    return link

def form_PP_(root, iP_, fd):  # form PPs of dP.valt[fd] + connected Ps val

    PPt_ = []
    for P in iP_: P.merged = 0
    for P in iP_:  # dP from link_ if fd
        if P.merged: continue
        if not P.lrim:
            PPt_ += [P]; continue
        _prim_ = P.prim; _lrim_ = P.lrim
        _P_ = {P}; link_ = set(); Et = np.zeros(2); n = 0
        while _prim_:
            prim_,lrim_ = set(),set()
            for _P,_L in zip(_prim_,_lrim_):
                if _L.Et[fd] < aves[fd] or _P.merged:
                    continue
                _P_.add(_P); link_.add(_L); Et += _L.Et; n += P.latuple[4]  # L
                prim_.update(set(_P.prim) - _P_)
                lrim_.update(set(_P.lrim) - link_)
                _P.merged = 1
            _prim_, _lrim_ = prim_, lrim_
        PPt = sum2PP(root, list(_P_), list(link_), Et, n)
        PPt_ += [PPt]

    return PPt_

def sum2PP(root, P_, dP_, Et, n):  # sum links in Ps and Ps in PP

    fd = isinstance(P_[0],CdP)
    if fd: latuple = np.sum([n.latuple for n in set([n for dP in P_ for n in  dP.nodet])], axis=0)
    else:  latuple = np.array([.0,.0,.0,.0,.0, np.zeros(2)], dtype=object)
    vertuple = np.array([np.zeros(6),np.zeros(6)])
    link_, A, S, box = [],[0,0],0, [np.inf,np.inf,0,0]
    # add uplinks:
    for dP in dP_:
        if dP.nodet[0] not in P_ or dP.nodet[1] not in P_: continue  # peripheral
        link_ += [dP]
        A = np.add(A,dP.angle)
        S += np.hypot(*dP.angle)  # links are contiguous but slanted
    # add Ps:
    for P in P_:
        if not fd:  # else summed from P_ nodets on top
            latuple += P.latuple
        for y,x in P.yx_ if isinstance(P, CP) else [P.nodet[0].yx, P.nodet[1].yx]:  # CdP
            box = accum_box(box,y,x)
    y0,x0,yn,xn = box
    PPt = [root, P_, link_, vertuple, latuple, A, S, box, [(y0+yn)/2,(x0+xn)/2], Et, n]  # n is the same as area
    for P in P_: P.root = PPt

    return PPt

def comp_latuple(_latuple, latuple, _n,n):  # 0der params

    _I, _G, _M, _Ma, _L, (_Dy, _Dx) = _latuple
    I, G, M, Ma, L, (Dy, Dx) = latuple
    rn = _n / n

    dI = _I - I*rn;  mI = ave_dI - dI            # vI = mI - ave
    dG = _G - G*rn;  mG = min(_G, G*rn)          # vG = mG - ave_mG
    dM = _M - M*rn;  mM = get_match(_M, M*rn)    # vM = mM - ave_mM  # M, Ma may be negative
    dMa= _Ma- Ma*rn; mMa = get_match(_Ma, Ma*rn) # vMa = mMa - ave_mMa
    dL = _L - L*rn;  mL = min(_L, L*rn)          # vL = mL - ave_mL
    mA, dA = comp_angle((_Dy,_Dx),(Dy,Dx))       # vA = mA - ave_mA

    d_ = np.array([dL, dI, dG, dM, dMa, dA])
    m_ = np.array([mL, mI, mG, mM, mMa, mA])
    et = np.array([np.sum(m_), np.sum(np.abs(d_))])

    return np.array([m_,d_]), et

def get_match(_par, par):
    match = min(abs(_par),abs(par))
    return -match if (_par<0) != (par<0) else match    # match = neg min if opposite-sign comparands

def accum_box(box, y, x):  # extend box with kernel
    y0, x0, yn, xn = box
    return min(y0,y), min(x0,x), max(yn,y), max(xn,x)

def min_dist(a, b, pad=0.5):
    if np.abs(a - b) < 1e-5:
        a -= 0.5
        b += 0.5
    return a, b

if __name__ == "__main__":

    # image_file = '../images//raccoon_eye.jpeg'
    image_file = '../images//toucan_small.jpg'
    image = imread(image_file)

    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vectorize_root(frame)
    # ----- verification -----
    # draw PPms as graphs of Ps and dPs
    # draw PPds as graphs of dPs and ddPs
    # dPs are shown in black, ddPs are shown in red
    import matplotlib.pyplot as plt
    from slice_edge import unpack_edge_
    num_to_show = 5
    edge_ = sorted(
        filter(lambda edge: hasattr(edge, "node_") and edge.node_, unpack_edge_(frame)),
        key=lambda edge: len(edge.yx_), reverse=True)
    for edge in edge_[:num_to_show]:
        yx_ = np.array(edge.yx_)
        yx0 = yx_.min(axis=0) - 1
        # show edge blob
        shape = yx_.max(axis=0) - yx0 + 2
        mask_nonzero = tuple(zip(*(yx_ - yx0)))
        mask = np.zeros(shape, bool)
        mask[mask_nonzero] = True
        # flatten
        P_, dP_, PPm_, PPd_ = [], [], [], []
        for node in edge.node_:
            if isinstance(node, CP): P_ += [node]
            else: PPm_ += [node]
        for PPm in PPm_:
            # root, P_, link_, mdLay, latuple, A, S, area, box, yx, n = PPm
            assert len(PPm[2]) > 0  # link_ can't be empty
            P_ += PPm[1]
            for link in PPm[2]:
                if isinstance(link, CdP): dP_ += [link]
                else: PPd_ += [link]
        for PPd in PPd_:
            assert len(PPd[2]) > 0  # link_ can't be empty
            dP_ += PPd[1] + PPd[2]  # dPs and ddPs

        plt.imshow(mask, cmap='gray', alpha=0.5)
        # plt.title("")
        print("Drawing Ps...")
        for P in P_:
            (y, x) = P.yx - yx0
            plt.plot(x, y, "ok")
        print("Drawing dPs...")
        nodet_set = set()
        for dP in dP_:
            _node, node = dP.nodet  # node is P or dP
            if (_node.id, node.id) in nodet_set:  # verify link uniqueness
                raise ValueError(
                    f"link not unique between {_node} and {node}. Duplicated links:\n" +
                    "\n".join([
                        f"dP.id={dP.id}, _node={dP.nodet[0]}, node={dP.nodet[1]}"
                        for dP in dP_
                        if dP.nodet[0] is _node and dP.nodet[1] is node
                    ])
                )
            nodet_set.add((_node.id, node.id))
            assert (*_node.yx,) < (*node.yx,)  # verify that link is up-link
            (_y, _x), (y, x) = _node.yx - yx0, node.yx - yx0
            style = "o-r" if isinstance(_node, CdP) else "-k"
            plt.plot([_x, x], [_y, y], style)

        print("Drawing PPm boxes...")
        for PPm in PPm_:
            _, _, _, _, _, _, _, _, (y0, x0, yn, xn), _ = PPm
            (y0, x0), (yn, xn) = ((y0, x0), (yn, xn)) - yx0
            y0, yn = min_dist(y0, yn)
            x0, xn = min_dist(x0, xn)
            plt.plot([x0, x0, xn, xn, x0], [y0, yn, yn, y0, y0], '-k', alpha=0.4)

        print("Drawing PPd boxes...")
        for PPd in PPd_:
            _, _, _, _, _, _, _, _, (y0, x0, yn, xn), _ = PPd
            (y0, x0), (yn, xn) = ((y0, x0), (yn, xn)) - yx0
            y0, yn = min_dist(y0, yn)
            x0, xn = min_dist(x0, xn)
            plt.plot([x0, x0, xn, xn, x0], [y0, yn, yn, y0, y0], '-r', alpha=0.4)

        plt.show()