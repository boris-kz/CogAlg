import numpy as np
from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_
from slice_edge import CP, slice_edge, comp_angle
from functools import reduce
from math import atan2, cos, floor, pi
from itertools import zip_longest
from copy import copy

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
len prior root_ sorted by G is root.olp, to eval for inclusion in PP or start new P by ave*olp
'''

ave, avd, aveB, ave_PPm, ave_PPd, ave_L, aI = 10, 10, 100, 50, 50, 4, 20
wM, wD, wI, wG, wA, wL = 10, 10, 1, 1, 20, 20  # dert:
w_t = np.ones((2,6))

class CdP(CBase):  # produced by comp_P, comp_slice version of Clink
    name = "dP"

    def __init__(l, nodet, span, angle, yx, Et, verT, latuple=None, root=None):
        super().__init__()

        l.nodet = nodet  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.L = 1  # min nodet
        l.verT = verT  # m_,d_,t_
        l.angl = angle  # dy,dx between node centers (for PP's internal L_)
        l.span = span  # distance between node centers
        l.yx = yx  # sum node_
        l.Et = Et
        l.root = root  # PPds containing dP
        l.rim = []
        l.lrim = []
        l.prim = []
    def __bool__(l): return l.nodet

def comp_slice_root(frame, rV=1, ww_t=None):

    blob_ = unpack_blob_(frame)
    for blob in blob_:
        if not blob.sign and blob.G > aveB * blob.root.olp:
            edge = slice_edge(blob, rV)  # ww_t[0][2:-1])  # wI,wG,wA?
            if edge.G * (len(edge.P_) - 1) > ave_PPm:  # eval PP, olp=1
                comp_slice(edge, rV, ww_t=ww_t)

def comp_slice(edge, rV=1, ww_t=None):  # root function

    global ave, avd, wM, wD, wI, wG, wA, wL, ave_L, ave_PPm, ave_PPd, w_t
    ave, avd, ave_L, ave_PPm, ave_PPd = np.array([ave, avd, ave_L, ave_PPm, ave_PPd]) / rV  # projected value change
    if np.any(ww_t):
        w_t = [[wM, wD, wI, wG, wA, wL]] * ww_t
        # der weights
    for P in edge.P_:  # add higher links
        P.verT = np.zeros((3,6))
        P.rim = []; P.lrim = []; P.prim = []
    edge.dP_ = []
    comp_P_(edge)  # vertical P cross-comp -> PP clustering, if lateral overlap
    PPt, mverT, mEt = form_PP_(edge.P_, fd=0)  # all Ps are converted to PPs
    if PPt:
        edge.node_ = PPt
        comp_dP_(edge, mEt)
        edge.link_, dverT, dEt = form_PP_(edge.dP_, fd=1)
        edge.verT = mverT + dverT
        edge.Et = mEt + dEt
    return PPt

def form_PP_(iP_, fd):  # form PPs of dP.valt[fd] + connected Ps val

    PPt_ = []; ET = np.zeros(4); VerT = np.zeros((3,6))

    for P in iP_: P.merged = 0
    for P in iP_:  # dP from link_ if fd
        if P.merged: continue
        _prim_ = P.prim; _lrim_ = P.lrim
        if fd: Et = P.Et  # summed verT, min L in dP
        else:  I,G,Dy,Dx,M,D,L = P.latuple; Et = np.array([M, G+abs(D), L, I])
        _P_ = {P}; link_ = set()
        verT = np.zeros((3,6))
        while _prim_:
            prim_,lrim_ = set(),set()
            for _P,_link in zip(_prim_,_lrim_):
                if _link.Et[fd] < [ave,avd][fd] or _P.merged:
                    continue
                _P_.add(_P); link_.add(_link)
                verT += _link.verT
                if fd: _Et = _P.Et
                else: _I,_G,_Dy,_Dx,_M,_D,_L = _P.latuple; _Et = np.array([_M,_G+abs(_D),_L,_I])
                Et += _Et  # intra-P similarity and variance
                prim_.update(set(_P.prim) - _P_)
                lrim_.update(set(_P.lrim) - link_)
                _P.merged = 1
            _prim_, _lrim_ = prim_, lrim_
        ET += Et; VerT += verT
        PPt_ += [sum2PP(list(_P_), list(link_), Et)]

    return PPt_, VerT, ET

def comp_P_(edge):  # form links from prelinks

    edge.rng = 1
    for _P, pre_ in edge.pre__.items():
        for P in pre_:  # prelinks
            dy,dx = np.subtract(P.yx,_P.yx)  # between node centers
            if abs(dy)+abs(dx) <= edge.rng * 2: # <max Manhattan distance
                angle=[dy,dx]; distance=np.hypot(dy,dx)
                verT, et = comp_latuple(_P.latuple, P.latuple, len(_P.dert_), len(P.dert_))
                dP = convert_to_dP(_P,P, verT, angle,distance, et)
                _P.rim += [dP]  # up only
                edge.dP_ += [dP]  # to form PPd_ by dval, separate from PPm_
    del edge.pre__

def comp_dP_(edge, mEt):  # node_- mediated: comp node.rim dPs, call from form_PP_

    M,_,n,_ = mEt
    rM = M/n / ave  # dP D borrows from normalized PP M
    for _dP in edge.dP_: _dP.prim = []; _dP.lrim = []
    for _dP in edge.dP_:
        _,D,_n,_ = _dP.Et
        if D/_n * rM > avd:
            _P, P = _dP.nodet  # _P is lower
            rn = n/_n; minn = min(_n,n)
            for dP in P.rim:  # higher links
                if dP not in edge.dP_: continue  # skip removed node links
                verT, et = comp_vert(_dP.verT[1], dP.verT[1], rn, minn)
                angle = np.subtract(dP.yx,_dP.yx)  # dy,dx of node centers
                distance = np.hypot(*angle)  # between node centers
                # up only:
                _dP.lrim += [convert_to_dP(_dP,dP, verT, angle, distance, et)]
                _dP.prim += [dP]

def convert_to_dP(_P,P, verT, angle, distance, Et):

    link = CdP(nodet=[_P,P], Et=Et, verT=verT, angle=angle, span=distance, yx=np.add(_P.yx, P.yx)/2)
    # Ps are dPs
    _P.verT+= link.verT; P.verT += link.verT
    _P.lrim += [link]; P.lrim += [link]
    _P.prim += [P];    P.prim +=[_P]
    link.L = min(_P.latuple[-1],P.latuple[-1]) if isinstance(_P,CP) else min(_P.L,P.L)  # P is CdP
    return link

def sum2PP(P_, dP_, Et):  # sum links in Ps and Ps in PP

    fd = isinstance(P_[0],CdP)
    if fd: latT = np.sum([n.latuple for n in set([n for dP in P_ for n in  dP.nodet])], axis=0)
    else:  latT = np.zeros(7)
    verT = np.zeros((3,6))
    link_ = []
    if dP_:  # add uplinks:
        S,A = 0,[0,0]
        for dP in dP_:
            if dP.nodet[0] not in P_ or dP.nodet[1] not in P_: continue  # peripheral link
            link_ += [dP]
            verT += dP.verT
            a = dP.angl; A = np.add(A,a); S += np.hypot(*a)  # span, links are contiguous but slanted
    else:  # single P PP
        S = P_[0].span if fd else 0  # no distance between nodes
        A = P_[0].angl if fd else P_[0].latuple[2:4]  # [I, G, Dy, Dx, M, D, L] or zeros?
    box = [np.inf,np.inf,0,0]
    for P in P_:
        if not fd:  # else summed from P_ nodets on top
            latT += P.latuple
        verT += P.verT
        for y,x in P.yx_ if isinstance(P, CP) else [P.nodet[0].yx, P.nodet[1].yx]:  # CdP
            box = accum_box(box,y,x)
    y0,x0,yn,xn = box
    PPt = [P_, link_, verT, latT, A, S, box, np.array([(y0+yn)/2,(x0+xn)/2]), Et]
    for P in P_: P.root = PPt
    return PPt

def comp_vert(_i_,i_, rn, minn, dir=1):  # i_ is ds, dir may be -1, ~ comp_lay

    i_ = i_ * rn  # normalize by compared accum span
    _a_,a_ = np.abs(_i_), np.abs(i_)  # d_ s
    d_ = (_i_- i_*dir)  # np.array d[I,G,A,M,D,L]
    m_ = np.minimum(_a_,a_)
    m_[(_i_<0) != (i_<0)] *= -1  # m is negative if comparands have opposite sign  # no np.where?
    t_ = np.maximum.reduce([_a_,a_, np.zeros(6)+1e-7])  # or signed?  I, G, A, M, D, L
    return (np.array([m_,d_,t_]),
            np.array([(m_/t_ +1)/2 @ w_t[0], (d_/t_ +1)/2 @ w_t[1], minn, t_@ w_t[0]]))  # Et

def comp_latuple(_latuple, latuple, _n,n):  # 0der params, add dir?

    _I,_G,_Dy,_Dx,_M,_D,_L = _latuple
    I, G, Dy, Dx, M, D, L = latuple
    rn = _n / n
    _pars = np.abs(np.array([_M,_D,_I,_G, np.array([_Dy,_Dx]),_L], dtype=object))
    pars  = np.abs(np.array([M, D, I, G, np.array([Dy,Dx]), L], dtype=object)) * rn
    pars[2] = [pars[2],aI]  # no avd*rn: d/=t
    m_,d_,t_ = comp(_pars,pars)
    return (np.array([m_,d_,t_]),  # verT
            np.array([(m_/t_ +1)/2 @ w_t[0], (d_/t_ +1)/2 @ w_t[1], min(L,_L), t_@ w_t[0]]))  # Et

def comp(_pars, pars):  # raw inputs or derivatives, norm to 0:1 in eval only

    m_,d_,t_ = [],[],[]
    for _p, p in zip(_pars, pars):
        if isinstance(_p, np.ndarray):
            mA, dA = comp_A(_p,p)  # both in -1:1
            m_ += [mA]; d_ += [dA]
            t_ += [1]  # norm already
        elif isinstance(p, list):  # massless I|S avd in p only
            p, avd = p
            d = _p - p; ad = abs(d)
            t_ += [max(avd, ad, 1e-7)]
            m_ += [avd-ad]  # +|-
            d_ += [d]
        else:  # massive
            t_ += [max(_p,p,1e-7)]
            m_ += [(min(_p,p) if _p<0 == p<0 else -min(_p,p))]
            d_ += [_p - p]
    return np.array(m_), np.array(d_), np.array(t_)

def comp_A(_A,A):

        dA = atan2(*_A) - atan2(*A)
        if   dA > pi: dA -= 2 * pi  # rotate CW
        elif dA <-pi: dA += 2 * pi  # rotate CCW

        return cos(dA), dA/pi  # mA, dA in -1:1

def accum_box(box, y, x):  # extend box with kernel
    y0, x0, yn, xn = box
    return min(y0,y), min(x0,x), max(yn,y), max(xn,x)

def min_dist(a, b, pad=0.5):
    if np.abs(a - b) < 1e-5:
        return a-pad, b+pad
    return a, b

if __name__ == "__main__":

    # image_file = './images//raccoon_eye.jpeg'
    image_file = './images//toucan_small.jpg'
    image = imread(image_file)

    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    comp_slice_root(frame)
    # ----- verification -----
    # draw PPms as graphs of Ps and dPs
    # draw PPds as graphs of dPs and ddPs
    # dPs are shown in black, ddPs are shown in red
    import matplotlib.pyplot as plt
    from frame_2D_alg.slice_edge import unpack_edge_
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
        assert all((isinstance(PPm, list) for PPm in edge.node_))
        PPm_, PPd_, P_, dP_ = edge.node_, edge.link_, edge.P_, edge.dP_

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
            assert (*_node.yx,) > (*node.yx,)  # verify that link is up-link
            (_y, _x), (y, x) = _node.yx - yx0, node.yx - yx0
            style = "o-r" if isinstance(_node, CdP) else "-k"
            plt.plot([_x, x], [_y, y], style)

        print("Drawing PPm boxes...")
        for PPm in PPm_:
            # P_, link_, vert, lat, A, S, box, yx, Et = PPm
            _, _, _, _, _, _, (y0, x0, yn, xn), _, _ = PPm
            (y0, x0), (yn, xn) = ((y0, x0), (yn, xn)) - yx0
            y0, yn = min_dist(y0, yn)
            x0, xn = min_dist(x0, xn)
            plt.plot([x0, x0, xn, xn, x0], [y0, yn, yn, y0, y0], '-k', alpha=0.4)

        print("Drawing PPd boxes...")
        for PPd in PPd_:
            _, _, _, _, _, _, (y0, x0, yn, xn), _, _ = PPd
            (y0, x0), (yn, xn) = ((y0, x0), (yn, xn)) - yx0
            y0, yn = min_dist(y0, yn)
            x0, xn = min_dist(x0, xn)
            plt.plot([x0, x0, xn, xn, x0], [y0, yn, yn, y0, y0], '-r', alpha=0.4)
        print("Max PPd size:", max((0, *(len(PPd[1]) for PPd in PPd_))))
        print("Max PPd link_ size:", max((0, *(len(PPd[2]) for PPd in PPd_))))

        plt.show()