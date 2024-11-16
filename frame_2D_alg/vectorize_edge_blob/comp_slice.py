import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
import sys
sys.path.append("..")
from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_
from slice_edge import CP, slice_edge, comp_angle, aveG

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
ave = 5  # ave direct m, change to Ave_min from the root intra_blob?
aves = ave_mI, ave_mG, ave_mM, ave_mMa, ave_mA, ave_mL = ave, 10, 2, .1, .2, 2
PP_aves = ave_PPm, ave_PPd = 50, 50
P_aves = ave_Pm, ave_Pd = 10, 10
ave_Gm = 50
ave_L = 5

class CdP(CBase):  # produced by comp_P, comp_slice version of Clink
    name = "dP"
    def __init__(l, nodet=None, mdLay=None, Et=None, root=None, span=None, angle=None, yx=None, latuple=None):
        super().__init__()

        l.nodet = [] if nodet is None else nodet  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.latuple = [] if latuple is None else latuple  # sum node_
        l.mdLay = [] if mdLay is None else mdLay  # same as md_C
        l.angle = [0,0] if angle is None else angle  # dy,dx between node centers
        l.span = span  # distance between node centers
        l.yx = [0,0] if yx is None else yx  # sum node_
        l.Et = [0,0,0,0] if Et is None else Et
        l.root = None if root is None else root  # PPds containing dP
        l.nmed = 0  # comp rng: n of mediating Ps between node_ Ps
        l.lrim = []
        l.prim = []
        # n = 1?
    def __bool__(l): return bool(l.mdLay[0])  # l.mdLay.H

def add_md_(HE, He,  irdnt=[]):  # p may be derP, sum derLays

    if HE:
        HE[0] = [V + v for V, v in zip_longest(HE[0], He[0], fillvalue=0)]  # H, np.array?
        HE[1] += He[1]  # Et
        HE[2] += He[2]  # n: combined param accumulation span
        if any(irdnt): HE[1][2:] = [E + e for E, e in zip(HE[1][2:], irdnt)]
    else:
        HE[:] = deepcopy(He)

def comp_md_(_H, H, rn=1, dir=1):  # replace dir with rev?

    vm, vd, rm, rd = 0,0,0,0
    derLay = []
    for i, (_d, d) in enumerate(zip(_H[1::2], H[1::2])):  # compare ds in md_ or ext
        d *= rn  # normalize by compared accum span
        diff = (_d - d) * dir
        match = min(abs(_d), abs(d))
        if (_d < 0) != (d < 0): match = -match  # negate if only one compared is negative
        vm += match - aves[i]  # fixed param set?
        vd += diff
        rm += vd > vm; rd += vm >= vd
        derLay += [match, diff]  # flat

    return [derLay, np.array([vm,vd,rm,rd], dtype='float'), 1]  # [H, Et, n]

def vectorize_root(frame):

    blob_ = unpack_blob_(frame)
    for blob in blob_:
        if not blob.sign and blob.G > aveG * blob.root.rdn:
            edge = slice_edge(blob)
            if edge.G*(len(edge.P_) - 1) > ave_PPm:  # eval PP, rdn=1
                comp_slice(edge)

def comp_slice(edge):  # root function

    edge.mdLay = [[], np.array([.0,.0,.0,.0]),0]  # H, Et, n
    for P in edge.P_:  # add higher links
        P.mdLay = [[],np.array([.0,.0,.0,.0]),0]  # for accumulation in sum2PP later (in lower P)
        P.rim = []; P.lrim = []; P.prim = []

    comp_P_(edge)  # vertical P cross-comp -> PP clustering, if lateral overlap
    edge.node_ = form_PP_(edge, edge.P_)

    for PPm in edge.node_:  # eval sub-clustering, not recursive
        if isinstance(PPm, list):  # PPt, not CP
            P_, link_, mdLay = PPm[1:4]
            Et = mdLay[1]
            if len(link_) > ave_L and Et[0] >PP_aves[0] * Et[2]:
                comp_link_(PPm)
                PPm[2] = form_PP_(PPm, link_)  # add PPds within PPm link_
            mdLay = PPm[3]
        else:
            mdLay = PPm.mdLay  # PPm is actually CP
        add_md_(edge.mdLay, mdLay)

def comp_P_(edge):  # form links from prelinks

    edge.rng = 1
    for P, _pre_ in edge.pre__.items():
        for _P in _pre_:  # prelinks
            dy,dx = np.subtract(P.yx,_P.yx) # dy,dx between node centers
            if abs(dy)+abs(dx) <= edge.rng * 2:  # <max Manhattan distance
                link = comp_P(_P,P, angle=[dy,dx], distance=np.hypot(dy,dx),fder=0)
                if link:
                    P.rim += [link]
    del edge.pre__

def comp_P(_P,P, angle=None, distance=None, fder=0):  # comp dPs if fd else Ps

    fd = isinstance(P,CdP)
    _y,_x = _P.yx; y,x = P.yx
    if fd:
        # der+: comp dPs
        rn = _P.mdLay[2] / P.mdLay[2]  # mdLay.n
        derLay = comp_md_(_P.mdLay[0], P.mdLay[0], rn=rn)  # comp md_latuple: H
        angle = np.subtract([y,x],[_y,_x])  # dy,dx of node centers
        distance = np.hypot(*angle)  # between node centers
    else:
        # rng+: comp Ps
        rn = len(_P.dert_) / len(P.dert_)
        H = comp_latuple(_P.latuple, P.latuple, rn)
        vm = sum(H[::2]); vd = sum(abs(d) for d in H[1::2])
        rm = 1 + vd > vm; rd = 1 + vm >= vd
        n = (len(_P.dert_)+len(P.dert_)) / 2  # der value = ave compared n?
        derLay = [H, [vm,vd,rm,rd], n]  # [H, Et, n]
    # get aves:
    latuple = [(P+p)/2 for P,p in zip(_P.latuple[:-1],P.latuple[:-1])] + [[(A+a)/2 for A,a in zip(_P.latuple[-1],P.latuple[-1])]]
    link = CdP(nodet=[_P,P], mdLay=derLay, angle=angle, span=distance, yx=[(_y+y)/2,(_x+x)/2], latuple=latuple)
    # if v > ave * r:
    Et = link.mdLay[1]
    if Et[fder] > aves[fder] * Et[fder+2]:
        P.lrim += [link]; _P.lrim += [link]
        P.prim +=[_P]; _P.prim +=[P]
        return link

def comp_link_(PP):  # node_- mediated: comp node.rim dPs, call from form_PP_

    link_ = PP[2]
    for dP in link_:
        if dP.mdLay[1][1] > aves[1]:
            for _dP in dP.nodet[0].rim:  # link.nodet is CP # for nmed, _rim_ in enumerate(dP.nodet[0].rim_):
                if _dP not in link_: continue  # skip removed node links
                comp_P(_dP,dP, fder=1)
                # if dlink: dlink.nmed = nmed  # link mediation order, not used?

def form_PP_(root, iP_):  # form PPs of dP.valt[fd] + connected Ps val

    PPt_ = []
    for P in iP_: P.merged = 0

    for P in iP_:  # for dP in link_ if fd
        if P.merged: continue
        if not P.lrim:
            PPt_ += [P]; continue
        _prim_ = P.prim; _lrim_ = P.lrim
        _P_ = {P}; link_ = set(); Et = np.array([0.0,0.0,0.0,0.0])
        while _prim_:
            prim_,lrim_ = set(),set()
            for _P,_L in zip(_prim_,_lrim_):
                if _P.merged: continue  # was merged
                _P_.add(_P); link_.add(_L); Et += _L.mdLay[1]  # _L.mdLay.Et
                prim_.update(set(_P.prim) - _P_)
                lrim_.update(set(_P.lrim) - link_)
                _P.merged = 1
            _prim_, _lrim_ = prim_, lrim_
        PPt = sum2PP(root, list(_P_), list(link_))
        PPt_ += [PPt]

    return PPt_

def sum2PP(root, P_, dP_):  # sum links in Ps and Ps in PP

    mdLay = [np.array, np.array([.0,.0,.0,.0]), 0]
    latuple, link_, A, S, area, n, box = [0,0,0,0,0,[0,0]], [],[0,0],0,0,0, [np.inf,np.inf,0,0]
    iRt = root[3][1] if isinstance(root,list) else root.mdLay[1][2:4]   # add to rdnt in root.mdLay.Et or root Et
    # add uplinks:
    for dP in dP_:
        if dP.nodet[0] not in P_ or dP.nodet[1] not in P_: continue
        add_md_(dP.nodet[1].mdLay, dP.mdLay, iRt)  # add to lower P
        link_ += [dP]  # link_
        A = np.add(A,dP.angle)
        S += np.hypot(*dP.angle)  # links are contiguous but slanted
    # add Ps:
    for P in P_:
        L = P.latuple[-2]
        area += L; n += L  # no + P.mdLay.n: current links only?
        add_lat(latuple, P.latuple)
        if P.mdLay:  # CdP or lower P has mdLay
            add_md_(mdLay, P.mdLay)
        for y,x in P.yx_ if isinstance(P, CP) else [P.nodet[0].yx, P.nodet[1].yx]:  # CdP
            box = accum_box(box,y,x)
    if mdLay:
        mdLay[1][2:4] = [R+r for R,r in zip(mdLay[1][2:4], iRt)]  # mdLay[1] = Et
    y0,x0,yn,xn = box
    # derH = [mdLay]
    PPt = [root, P_, link_, mdLay, latuple, A, S, area, box, [(y0+yn)/2,(x0+xn)/2], n]
    for P in P_: P.root = PPt
    return PPt

def add_lat(Lat,lat):
    Lat[:] = [P+p for P,p in zip(Lat[:-1],lat[:-1])] + [[A+a for A,a in zip(Lat[-1],lat[-1])]]
    return Lat

def comp_latuple(_latuple, latuple, rn, fagg=0):  # 0der params

    _I, _G, _M, _Ma, _L, (_Dy, _Dx) = _latuple
    I, G, M, Ma, L, (Dy, Dx) = latuple

    dI = _I - I*rn;  mI = ave_dI - dI
    dG = _G - G*rn;  mG = min(_G, G*rn) - aves[1]
    dL = _L - L*rn;  mL = min(_L, L*rn) - aves[2]
    dM = _M - M*rn;  mM = get_match(_M, M*rn) - aves[3]  # M, Ma may be negative
    dMa= _Ma- Ma*rn; mMa = get_match(_Ma, Ma*rn) - aves[4]
    mAngle, dAngle = comp_angle((_Dy,_Dx), (Dy,Dx))

    ret = [mL,dL,mI,dI,mG,dG,mM,dM,mMa,dMa,mAngle-aves[5],dAngle]
    if fagg:  # add norm m,d, ret=[ret,Ret]:
        mval, dval = sum(ret[::2]),sum(ret[1::2])
        mrdn, drdn = dval>mval, mval>dval
        ret = [ret, np.array([mval,dval,mrdn,drdn]), 1]  # if fagg only
    return ret

def get_match(_par, par):
    match = min(abs(_par),abs(par))
    return -match if (_par<0) != (par<0) else match    # match = neg min if opposite-sign comparands

def accum_box(box, y, x):
    """Box coordinate accumulation."""
    y0, x0, yn, xn = box
    return min(y0, y), min(x0, x), max(yn, y), max(xn, x)

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
            assert _node.yx < node.yx  # verify that link is up-link
            (_y, _x), (y, x) = _node.yx - yx0, node.yx - yx0
            style = "o-r" if isinstance(_node, CdP) else "-k"
            plt.plot([_x, x], [_y, y], style)

        print("Drawing PPm boxes...")
        for PPm in PPm_:
            _, _, _, _, _, _, _, _, (y0, x0, yn, xn), _, _ = PPm
            (y0, x0), (yn, xn) = ((y0, x0), (yn, xn)) - yx0
            y0, yn = min_dist(y0, yn)
            x0, xn = min_dist(x0, xn)
            plt.plot([x0, x0, xn, xn, x0], [y0, yn, yn, y0, y0], '-k', alpha=0.4)

        print("Drawing PPd boxes...")
        for PPd in PPd_:
            _, _, _, _, _, _, _, _, (y0, x0, yn, xn), _, _ = PPd
            (y0, x0), (yn, xn) = ((y0, x0), (yn, xn)) - yx0
            y0, yn = min_dist(y0, yn)
            x0, xn = min_dist(x0, xn)
            plt.plot([x0, x0, xn, xn, x0], [y0, yn, yn, y0, y0], '-r', alpha=0.4)

        plt.show()