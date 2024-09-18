import numpy as np
from collections import deque, defaultdict
from copy import deepcopy, copy
from itertools import zip_longest
import sys
sys.path.append("..")
from frame_blobs import CBase, imread
if __name__ == "__main__": from slice_edge import CP, comp_angle, CsliceEdge
else: from .slice_edge import CP, comp_angle, CsliceEdge

'''
Vectorize is a terminal fork of intra_blob.

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
ave_L = 5
ave_Gm = 50

class CcompSliceFrame(CsliceEdge):
    class CEdge(CsliceEdge.CEdge): # replaces CBlob

        def vectorize(edge):  # overrides in CsliceEdge.CEdge.vectorize
            edge.slice_edge()
            if edge.latuple[-1] * (len(edge.P_)-1) > ave_PPm:  # eval PP, rdn=1
                comp_slice(edge)
    CBlob = CEdge

class CdP(CBase):  # produced by comp_P, comp_slice version of Clink
    name = "dP"
    def __init__(l, nodet=None, mdLay=None, Et=None, Rt=None, root=None, span=None, angle=None, yx=None, latuple=None):
        super().__init__()

        l.nodet = [] if nodet is None else nodet  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.rim = []  # upper 2nd der links
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
    def __bool__(l): return bool(l.mdLay.H)

class CH(CBase):  # generic derivation hierarchy of variable nesting, depending on effective agg++(sub++ depth

    name = "H"
    def __init__(He, node_=None, md_t=None, n=0, Et=None, Rt=None, H=None, root=None, i=None, it=None):
        super().__init__()
        He.node_ = [] if node_ is None else node_  # concat, may be redundant to G.node_, lowest nesting order
        He.md_t = [] if md_t is None else md_t  # compared [mdlat,mdLay,mdext] per layer
        He.H = [] if H is None else H  # lower derLays or md_ in md_C, empty in bottom layer
        He.n = n  # number of params compared to form derH, sum in comp_G, from nodes in sum2graph
        He.Et = [0,0,0,0] if Et is None else Et  # evaluation tuple: valt, rdnt
        He.root = None if root is None else root  # N or higher-composition He
        He.i = 0 if i is None else i   # lay index in root.H, to revise rdn
        He.it = [0,0] if it is None else it  # max fd lay in He.H: init add,comp if deleted higher layers' H,md_t
        # He.ni = 0  # exemplar in node_, trace in both directions?
        # He.depth = 0  # nesting in H[0], -=i in H[Hi], added in agg++|sub++
        # He.nest = nest  # nesting depth: -1/ ext, 0/ md_, 1/ derH, 2/ subH, 3/ aggH?
    def __bool__(H): return H.n != 0

    def add_md_(HE, He, irdnt=[]):  # p may be derP, sum derLays

        # sum md_s:
        HE.H[:] = [V + v for V, v in zip_longest(HE.H, He.H, fillvalue=0)]
        HE.n += He.n  # combined param accumulation span
        HE.Et = np.add(HE.Et, He.Et)
        if any(irdnt): HE.Et[2:] = [E + e for E, e in zip(HE.Et[2:], irdnt)]
        return HE

    def add_md_t(HE, He, irdnt=[]):  # sum derLays

        for MD_C, md_C in zip(HE.md_t, He.md_t):
            MD_C.add_md_(md_C)
        HE.n += He.n
        HE.Et = np.add(HE.Et, He.Et)
        if any(irdnt): HE.Et[2:] = [E+e for E,e in zip(HE.Et[2:], irdnt)]

    def add_H(HE, He, irdnt=[]):  # unpack down to numericals and sum them

        if HE:
            for Lay,lay in zip_longest(HE.H, He.H, fillvalue=None):  # cross comp layer
                if lay:
                    if Lay: Lay.add_H(lay, irdnt)
                    else:
                        if Lay is None: HE.append_(CH().copy(lay))  # pack a copy of new lay in HE.H
                        else:           HE.H[HE.H.index(Lay)] = CH(root=HE).copy(lay)  # Lay was []
            # default
            HE.add_md_t(He)  # [ext_md_C, lat_md_C, lay_md_C]
            HE.Et = np.add(HE.Et, He.Et)
            HE.node_ += [node for node in He.node_ if node not in HE.node_]
            # node_ is empty in CL derH?
            if any(irdnt): HE.Et[2:] = [E+e for E,e in zip(HE.Et[2:], irdnt)]
            HE.n += He.n  # combined param accumulation span
        else:
            HE.copy(He)  # init
        # feedback, ideally buffered from all elements before summing in root, ultimately G|L:
        root = HE.root
        while root is not None:
            root.Et = np.add(root.Et, He.Et)
            if isinstance(root, CH):
                root.n += He.n
                root.node_ += [node for node in He.node_ if node not in HE.node_]
                root = root.root
            else: break  # root is G|L
        return HE

    def append_(HE,He, irdnt=None, flat=0):

        if irdnt is None: irdnt = []
        I = len(HE.H)  # min index
        if flat:
            for i, lay in enumerate(He.H):  # L.derH and root.derH.H should have different references?:
                lay = CH().copy(lay); lay.i = I+i; lay.root = HE; HE.H += [lay]
        else:
            He = CH().copy(He); He.i = I; He.root = HE; HE.H += [He]
        if HE.md_t: HE.add_md_t(He)  # accumulate [lat_md_C,lay_md_C,ext_md_C]
        else:       HE.md_t = [CH().copy(md_) for md_ in He.md_t]
        HE.n += He.n
        Et, et = HE.Et, He.Et
        HE.Et = np.add(HE.Et, He.Et)
        if irdnt: Et[2:4] = [E+e for E,e in zip(Et[2:4], irdnt)]
        root = HE
        while root is not None:
            root.Et = np.add(root.Et,He.Et)
            if isinstance(root, CH):
                root.node_ += [node for node in He.node_ if node not in HE.node_]
                root.n += He.n
                root = root.root
            else:
               break  # root is G|L
        return HE  # for feedback in agg+

    def comp_md_(_He, He, rn=1, fagg=0, frev=0):

        vm, vd, rm, rd = 0,0,0,0
        derLay = []
        for i, (_d, d) in enumerate(zip(_He.H[1::2], He.H[1::2])):  # compare ds in md_ or ext
            d *= rn  # normalize by compared accum span
            diff = _d - d
            if frev: diff = -diff  # from link with reversed dir
            match = min(abs(_d), abs(d))
            if (_d < 0) != (d < 0): match = -match  # negate if only one compared is negative
            vm += match - aves[i]  # fixed param set?
            vd += diff
            rm += vd > vm; rd += vm >= vd
            derLay += [match, diff]  # flat

        return CH(H=derLay, Et=[vm,vd,rm,rd], n=1)

    def comp_md_t(_He, He):

        der_md_t = []; Et = [0,0,0,0]; Rt = [0,0]
        for _md_C, md_C in zip(_He.md_t, He.md_t):

            der_md_C = _md_C.comp_md_(md_C, rn=1, fagg=0,frev=0)  # H is a list, use md_C
            der_md_t += [der_md_C]; Et = np.add(Et, der_md_C.Et)

        return CH(md_t=der_md_t, Et=Et, n=2.5)

    def comp_H(_He, He, rn=1, fagg=0, frev=0):  # unpack CHs down to numericals and compare them

        DLay = CH(node_=_He.node_+He.node_).add_H(_He.comp_md_t(He))
        # node_ is mediated comparands, default comp He.md_t per He,
        # H=[] in bottom | deprecated layer

        for _lay, lay in zip(_He.H, He.H):  # loop extHs or [mdlat,mdLay,mdext] rng tuples, flat
            if _lay and lay:
                dLay = _lay.comp_H(lay, rn, fagg, frev)  # comp He.md_t, comp,unpack lay.H
                DLay.append_(dLay, flat=0)               # subHH( subH?
        return DLay

    def copy(_He, He):
        for attr, value in He.__dict__.items():
            if attr != '_id' and attr != 'root' and attr in _He.__dict__.keys():  # copy attributes, skip id, root
                if attr == 'H':
                    if He.H:
                        _He.H = []
                        if isinstance(He.H[0], CH):
                            for lay in He.H: _He.H += [CH().copy(lay)]  # can't deepcopy CH.root
                        else: _He.H = deepcopy(He.H)  # md_
                elif attr == "md_t":
                    _He.md_t += [CH().copy(md_) for md_ in He.md_t]  # can't deepcopy CH.root
                elif attr == "node_":
                    _He.node_ = copy(He.node_)
                else:
                    setattr(_He, attr, deepcopy(value))
        return _He

def comp_slice(edge):  # root function

    edge.mdLay = CH()
    for P in edge.P_:  # add higher links
        P.mdLay = CH()  # for accumulation in sum2PP later (in lower P)
        P.rim_ = []; P.lrim = []; P.prim = []
    rng_recursion(edge)  # vertical P cross-comp -> PP clustering, if lateral overlap
    form_PP_(edge, edge.P_)
    for N in edge.node_:
        mdLay = N[3] if isinstance(N, list) else N.mdLay  # CP
        edge.mdLay.add_md_(mdLay)

def rng_recursion(edge):  # similar to agg+ rng_recursion, but looping and contiguously link mediated

    rng = 1  # cost of links added per rng+
    _Pt_ = edge.pre__.items() # includes prelink

    while True:  # extend mediated comp rng by adding prelinks
        Pt_ = []  # with new prelinks
        V = 0
        for P,_pre_ in _Pt_:
            if len(P.rim_) < rng-1: continue  # no _rng_link_ or top row
            rng_link_ = []; pre_ = []  # per rng+
            for _P in _pre_:  # prelinks
                _y,_x = _P.yx; y,x = P.yx
                dy,dx = np.subtract([y,x],[_y,_x]) # dy,dx between node centers
                if abs(dy)+abs(dx) <= rng*2:  # <max Manhattan distance
                    if len(_P.rim_) < rng-1: continue
                    link = comp_P(_P,P, angle=[dy,dx], distance=np.hypot(dy,dx),fder=0)
                    if link:  # mlink
                        V += link.mdLay.Et[0]
                        rng_link_ += [link]
                        if _P.rim_: pre_ += [dP.nodet[0] for dP in _P.rim_[-1]]  # connected __Ps
                        else:       pre_ += edge.pre__[_P]  # rng == 1
            # next P_ has prelinks:
            if pre_: Pt_ += [(P,pre_)]
            if rng_link_: P.rim_ += [rng_link_]

        if not Pt_ or V <= ave * rng * len(Pt_) * 6:  # implied val of all __P_s, 6: len mtuple
            break
        else:
            _Pt_ = Pt_
            rng += 1
    edge.rng=rng  # represents rrdn
    del edge.pre__

def comp_P(_P,P, angle=None, distance=None, fder=0):  # comp dPs if fd else Ps

    fd = isinstance(P,CdP)
    _y,_x = _P.yx; y,x = P.yx
    if fd:
        # der+: comp dPs
        rn = _P.mdLay.n / P.mdLay.n
        derLay = _P.mdLay.comp_md_(P.mdLay, rn=rn)
        angle = np.subtract([y,x],[_y,_x]) # dy,dx between node centers
        distance = np.hypot(*angle) # between node centers
    else:
        # rng+: comp Ps
        rn = len(_P.dert_) / len(P.dert_)
        H = comp_latuple(_P.latuple, P.latuple, rn)  # or remove fagg and output CH as default?
        vm = sum(H[::2]); vd = sum(abs(d) for d in H[1::2])
        rm = 1 + vd > vm; rd = 1 + vm >= vd
        n = (len(_P.dert_)+len(P.dert_)) / 2  # der value = ave compared n?
        derLay = CH(Et=[vm,vd,rm,rd], H=H, n=n)
    # get aves:
    latuple = [(P+p)/2 for P,p in zip(_P.latuple[:-1],P.latuple[:-1])] + [[(A+a)/2 for A,a in zip(_P.latuple[-1],P.latuple[-1])]]
    link = CdP(nodet=[_P,P], mdLay=derLay, angle=angle, span=distance, yx=[(_y+y)/2,(_x+x)/2], latuple=latuple)
    # if v > ave * r:
    if link.mdLay.Et[fder] > aves[fder] * link.mdLay.Et[fder+2]:
        P.lrim += [link]; P.prim +=[_P]; _P.prim +=[P]  # add Link as uplink in P.lrim
        return link

def form_PP_(root, iP_, fd=0):  # form PPs of dP.valt[fd] + connected Ps val

    for P in iP_: P.merged = 0
    PPt_ = []
    for P in iP_:
        if not P.lrim:
            PPt_ += [P]; continue
        _prim_ = P.prim; _lrim_ = P.lrim
        _P_ = {P}; link_ = set(); Et = [0,0,0,0]
        while _prim_:
            prim_,lrim_ = set(),set()
            for _P,_L in zip(_prim_,_lrim_):
                if _P.merged: continue  # was merged
                _P_.add(_P); link_.add(_L); Et = np.add(Et, _L.mdLay.Et)
                prim_.update(set(_P.prim) - _P_)
                lrim_.update(set(_P.lrim) - link_)
                _P.merged = 1
            _prim_, _lrim_ = prim_, lrim_
        PPt = sum2PP(root, list(_P_), list(link_), fd)
        PPt_ += [PPt]
        P_, link_, mdLay = PPt[1:4]
        if not fd and len(P_) > ave_L and mdLay.Et[fd] >PP_aves[fd] * mdLay.Et[2+fd]:
            comp_link_(PPt)
            form_PP_(PPt, link_, fd=1)  # form sub_PPd_ in select PPs, not recursive

    if isinstance(root, list): root[2] = PPt_  # PPt
    else: root.node_ = PPt_  # Cedge


def comp_link_(PP):  # node_- mediated: comp node.rim dPs, call from form_PP_

    link_ = PP[2]
    for dP in link_:
        if dP.mdLay.Et[1] > aves[1]:
            for nmed, _rim_ in enumerate(dP.nodet[0].rim_):  # link.nodet is CP
                for _dP in _rim_:
                    if _dP not in link_: continue  # skip those removed node's links
                    dlink = comp_P(_dP,dP,fder=1)
                    if dlink:
                        dP.rim += [dlink]  # in lower node uplinks
                        dlink.nmed = nmed  # link mediation order0

# draft:
def sum2PP(root, P_, dP_, fd):  # sum links in Ps and Ps in PP

    mdLay, latuple, link_, A, S, area, n, box = CH(), [0,0,0,0, 0, [0,0]], [], [0,0], 0, 0, 0, [0,0,0,0]
    iRt = root[3].Et if isinstance(root,list) else root.mdLay.Et[2:4]   # add to rdnt
    # add uplinks:
    for dP in dP_:
        if dP.nodet[0] not in P_ or dP.nodet[1] not in P_: continue
        dP.nodet[1].mdLay.add_md_(dP.mdLay, iRt)  # add to lower P
        link_ += [dP]  # link_
        A = np.add(A,dP.angle)
        S += np.hypot(*dP.angle)  # links are contiguous but slanted
    # add Ps:
    celly_,cellx_ = [],[]
    for P in P_:
        L = P.latuple[-2]
        area += L; n += L  # no + P.mdLay.n: current links only?
        add_lat(latuple, P.latuple)
        if P.mdLay:  # CdP or lower P has mdLay
            mdLay.add_md_(P.mdLay)
        for y,x in P.yx_ if isinstance(P, CP) else [P.nodet[0].yx, P.nodet[1].yx]:  # CdP
            y = int(round(y)); x = int(round(x))  # summed with float dy,dx in slice_edge?
            box = accum_box(box,y,x); celly_+=[y]; cellx_+=[x]
    if mdLay:
        mdLay.Et[2:4] = [R+r for R,r in zip(mdLay.Et[2:4], iRt)]
    y0,x0,yn,xn = box
    mask__ = np.zeros((yn-y0, xn-x0), bool)  # pixmap
    celly_ = np.array(celly_); cellx_ = np.array(cellx_)
    mask__[(celly_-y0, cellx_-x0)] = True
    # derH = [mdLay]
    PPt = [root, P_, link_, mdLay, latuple, A, S, area, box, [(y0+yn)/2,(y0+yn)/2], n]
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
        ret = CH(H=ret, Et=[mval,dval,mrdn,drdn], n=1)  # if fagg only
    return ret

def get_match(_par, par):
    match = min(abs(_par),abs(par))
    return -match if (_par<0) != (par<0) else match    # match = neg min if opposite-sign comparands

def negate(He):
    if isinstance(He.H[0], CH):
        for i,lay in enumerate(He.H):
            He.H[i] = negate(lay)
    else:  # md_
        He.H[1::2] = [-d for d in He.H[1::2]]
    return He

def accum_box(box, y, x):
    """Box coordinate accumulation."""
    y0, x0, yn, xn = box
    return min(y0, y), min(x0, x), max(yn, y+1), max(xn, x+1)

if __name__ == "__main__":

    image_file = '../images/raccoon_eye.jpeg'
    image = imread(image_file)

    # ----- verification -----
    frame = CcompSliceFrame(image).segment()
    import matplotlib.pyplot as plt
    from slice_edge import unpack_edge_
    num_to_show = 5

    edge_ = sorted(
        filter(lambda edge: hasattr(edge, "node_") and edge.node_, unpack_edge_(frame)),
        key=lambda edge: len(edge.yx_), reverse=True)
    for edge in edge_[:num_to_show]:
        yx_ = np.array(edge.yx_)
        yx0 = yx_.min(axis=0) - 1
        # show edge-blob
        shape = yx_.max(axis=0) - yx0 + 2
        mask_nonzero = tuple(zip(*(yx_ - yx0)))
        mask = np.zeros(shape, bool)
        mask[mask_nonzero] = True

        def draw_PP(_PP):
            if _PP.node_: print(_PP, "has node_")
            for fd, PP_ in enumerate(_PP.node_):
                if not PP_: continue
                plt.imshow(mask, cmap='gray', alpha=0.5)
                fork = 'der+' if isinstance(PP_[0].P_[0], CdP) else 'rng+'
                plt.title(f"Number of PP{'d' if fd else 'm'}s: {len(PP_)}, {fork}")
                for PP in PP_:
                    nodet_set = set()
                    for P in PP.P_:
                        (y, x) = P.yx - yx0
                        plt.plot(x, y, "ok")
                    for dP in PP.link_:
                        _node, node = dP.nodet
                        if (_node.id, node.id) in nodet_set:  # verify link uniqueness
                            raise ValueError(
                                "link not unique between {_node} and {node}. PP.link_:\n" +
                                "\n".join(map(lambda dP: f"dP.id={dP.id}, _node={dP.nodet[0]}, node={dP.nodet[1]}", PP.link_))
                            )
                        nodet_set.add((_node.id, node.id))
                        assert _node.yx < node.yx  # verify that link is up-link
                        (_y, _x), (y, x) = _node.yx - yx0, node.yx - yx0
                        plt.plot([_x, x], [_y, y], "-k")
                plt.show()
                for PP in PP_: draw_PP(PP)

        draw_PP(edge)