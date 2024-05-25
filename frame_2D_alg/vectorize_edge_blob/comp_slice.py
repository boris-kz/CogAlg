import numpy as np
from collections import deque, defaultdict
from copy import deepcopy, copy
from itertools import zip_longest, combinations
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

ave = 5  # ave direct m, change to Ave_min from the root intra_blob?
ave_dI = ave_inv = 20  # ave inverse m, change to Ave from the root intra_blob?
ave_Gm = 50
aves = ave_mI, ave_mG, ave_mM, ave_mMa, ave_mA, ave_mL = ave, 10, 2, .1, .2, 2
P_aves = ave_Pm, ave_Pd = 10, 10
PP_aves = ave_PPm, ave_PPd = 50, 50

class CcompSliceFrame(CsliceEdge):

    class CEdge(CsliceEdge.CEdge): # replaces CBlob

        def vectorize(edge):  # overrides in CsliceEdge.CEdge.vectorize

            edge.slice_edge()
            if edge.latuple[-1] * (len(edge.P_)-1) > ave_PPm:  # eval PP, rdn=1
                edge.iderH = CH(); edge.fback_ = []
                for P in edge.P_:
                    P.derH = CH()
                ider_recursion(None, edge)  # vertical, lateral-overlap P cross-comp -> PP clustering

    CBlob = CEdge

class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster

    def __init__(G, root=None, rng=1, fd=0, node_=None, link_=None, Et=None, n=0):  # we need P_ to init PP, Et in init graph
        super().__init__()
        # PP:
        G.root = [] if root is None else root  # mgraphs that contain this G, single-layer
        G.rng = rng
        G.S = 0  # sparsity: distance between node centers
        G.A = 0, 0  # angle: summed dy,dx in links
        G.area = 0
        G.Et = [0,0,0,0] if Et is None else Et  # external eval tuple, summed from rng++ before forming new graph and appending G.extH
        G.latuple = [0,0,0,0,0,[0,0]]  # lateral I,G,M,Ma,L,[Dy,Dx]
        G.iderH = CH() # summed from PPs
        G.derH = CH()  # nested derH in Gs: [[subH,valt,rdnt,dect]], subH: [[derH,valt,rdnt,dect]]: 2-fork composition layers
        G.DerH = CH()  # summed kernel rims
        G.node_ = [] if node_ is None else node_  # convert to node_t in sub_recursion
        G.link_ = [] if link_ is None else link_  # links per comp layer, nest in rng+)der+
        G.box = [np.inf, np.inf, -np.inf, -np.inf]  # y,x,y0,x0,yn,xn
        G.kH = []  # kernel: hierarchy of rim layers
        # graph-external, +level per root sub+:
        G.n = n  # external n (last layer n)
        G.rim = []  # direct links, depth, init rim_t, link_tH in base sub+ | cpr rd+, link_tHH in cpr sub+
        G.extH = CH()  # G-external daggH( dsubH( dderH, summed from rim links
        G.ExtH = CH()  # summed link.DerH
        G.alt_graph_ = []  # adjacent gap+overlap graphs, vs. contour in frame_graphs
        # dynamic attrs:
        G.Rim = []  # links to the most mediated nodes
        G.fback_ = []  # feedback [[aggH,valt,rdnt,dect]] per node layer, maps to node_H
        G.compared_ = []
        # Rdn: int = 0  # for accumulation or separate recursion count?
        # it: list = z([None,None])  # graph indices in root node_s, implicitly nested
        # depth: int = 0  # n sub_G levels over base node_, max across forks
        # nval: int = 0  # of open links: base alt rep
        # id_H: list = z([[]])  # indices in the list of all possible layers | forks, not used with fback merging
        # top aggLay: derH from links, lower aggH from nodes, only top Lay in derG:
        # top Lay from links, lower Lays from nodes, hence nested tuple?

    def __bool__(G): return G.n != 0  # to test empty

class CdP(CBase):  # comp_slice version of Clink
    name = "dP"
    def __init__(l, node_=None, derH=None, root=None, distance=None, angle=None, yx=None, latuple=None):
        super().__init__()

        l.node_ = [] if node_ is None else node_  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.angle = [0,0] if angle is None else angle  # dy,dx between node centers
        l.distance = distance  # distance between node centers
        l.latuple = [] if latuple is None else latuple  # sum node_
        l.yx = [0,0] if yx is None else yx  # sum node_
        l.rim = []
        l.derH = CH() if derH is None else derH
        l.root = None if root is None else root  # PPds containing dP
        l.nmed = 0  # comp rng: n of mediating Ps between node_ Ps

    def __bool__(l): return bool(l.derH.H)


class CH(CBase):  # generic derivation hierarchy with variable nesting
    '''
    len layer +extt: 2, 3, 6, 12, 24,
    or without extt: 1, 1, 2, 4, 8..: max n of tuples per der layer = summed n of tuples in all lower layers:
    lay1: par     # derH per param in vertuple, layer is derivatives of all lower layers:
    lay2: [m,d]   # implicit nesting, brackets for clarity:
    lay3: [[m,d], [md,dd]]: 2 sLays,
    lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays
    '''
    name = "H"
    def __init__(He, n=0, Et=None, relt=None, H=None):
        super().__init__()
        # He.nest = nest  # nesting depth: -1/ ext, 0/ md_, 1/ derH, 2/ subH, 3/ aggH
        He.n = n  # total number of params compared to form derH, summed in comp_G and then from nodes in sum2graph
        He.Et = [0,0,0,0] if Et is None else Et   # evaluation tuple: valt, rdnt
        He.relt = [0,0] if relt is None else relt  # m,d relative to max possible m,d
        He.H = [] if H is None else H  # hierarchy of der layers or md_

    def __bool__(H): return H.n != 0

    def add_(HE, He, irdnt=None):  # unpack down to numericals and sum them

        if irdnt is None: irdnt = []
        if HE:
            if isinstance(HE.H[0], CH):
                H = []
                for Lay, lay in zip_longest(HE.H, He.H, fillvalue=None):
                    if lay:  # to be summed
                        if Lay is None: Lay = CH()
                        Lay.add_(lay, irdnt)  # recursive unpack to sum md_s
                    H += [Lay]
                HE.H = H
            else:
                HE.H = [V+v for V,v in zip_longest(HE.H, He.H, fillvalue=0)]  # both Hs are md_s
            # default:
            Et, et = HE.Et, He.Et
            HE.Et = np.add(HE.Et, He.Et); HE.relt = np.add(HE.relt, He.relt)
            if any(irdnt): HE.Et[2:] = [E+e for E,e in zip(HE.Et[2:], irdnt)]
            HE.n += He.n  # combined param accumulation span
        else:
            HE.copy(He)  # initialization

    def append_(HE,He, irdnt=None, flat=0):

        if irdnt is None: irdnt = []
        if flat: HE.H += deepcopy(He.H)  # append flat
        else:    HE.H += [He]  # append nested
        Et, et = HE.Et, He.Et
        HE.Et = np.add(HE.Et, He.Et); HE.relt = np.add(HE.relt, He.relt)
        if irdnt: Et[2:4] = [E+e for E,e in zip(Et[2:4], irdnt)]
        HE.n += He.n  # combined param accumulation span


    def comp_(_He, He, dderH, rn=1, fagg=0, flat=1):  # unpack tuples (formally lists) down to numericals and compare them

        n = 0
        if isinstance(_He.H[0], CH):  # _lay and lay is He_, they are aligned
            Et = [0,0,0,0]  # Vm,Vd, Rm,Rd
            relt = [0,0]  # Dm,Dd
            dH = []
            for _lay,lay in zip(_He.H,He.H):  # md_| ext| derH| subH| aggH, eval nesting, unpack,comp ds in shared lower layers:
                if _lay and lay:  # ext is empty in single-node Gs
                    dlay = _lay.comp_(lay, CH(), rn, fagg=fagg, flat=1)  # dlay is dderH
                    Et = np.add(Et, dlay.Et)
                    relt = np.add(relt, dlay.relt)
                    dH += [dlay]; n += dlay.n
                else:
                    dH += [CH()]  # empty?
        else:  # H is md_, numerical comp:
            vm,vd,rm,rd, decm,decd = 0,0,0,0,0,0
            dH = []
            for i, (_d,d) in enumerate(zip(_He.H[1::2], He.H[1::2])):  # compare ds in md_ or ext
                d *= rn  # normalize by comparand accum span
                diff = _d-d
                match = min(abs(_d),abs(d))
                if (_d<0) != (d<0): match = -match  # if only one comparand is negative
                if fagg:
                    maxm = max(abs(_d), abs(d))
                    decm += abs(match) / maxm if maxm else 1  # match / max possible match
                    maxd = abs(_d) + abs(d)
                    decd += abs(diff) / maxd if maxd else 1  # diff / max possible diff
                vm += match - aves[i]  # fixed param set?
                vd += diff
                dH += [match,diff]  # flat
            Et = [vm,vd,rm,rd]; relt= [decm,decd]
            n = len(_He.H)/12  # unit n = 6 params, = 12 in md_

        dderH.append_(CH(Et=Et, relt=relt, H=dH, n=n), flat=flat)  # currently flat=1
        return dderH

    def copy(_H, H):
        for attr, value in H.__dict__.items():
            if attr != '_id' and attr in _H.__dict__.keys():  # copy only the available attributes and skip id
                setattr(_H, attr, deepcopy(value))


def ider_recursion(root, PP, fd=0):  # node-mediated correlation clustering: keep same Ps and links, increment link derH, then P derH in sum2PP

    Q = comp_link_(PP) if fd else rng_recursion(PP)
    # replace PP node_,link_, derH+=derLay by der+, or extend PP.link_ by rng++ cross-comp
    form_PP_t(PP, Q)  # calls der+

    if root is not None and PP.iderH: root.fback_ += [PP.iderH]  # feedback per PPd?

def rng_recursion(edge):  # similar to agg+ rng_recursion, but looping and contiguously link mediated

    iP_, i_P__ = edge.P_, edge._P__  # rng++ and prelink per edge, not PPs
    rng = 1  # cost of links added per rng+
    while True:
        P_ = []; V = 0
        _P__ = [[] for _ in edge.P_]
        for i, (P, i_P_) in enumerate(zip(iP_, i_P__)):
            if len(P.rim_) < rng: continue  # no _rng_link_ or top row
            rng_link_, _P_ = [],[]  # both per rng+
            for _P in i_P_:  # prelinks
                _y,_x = _P.yx; y,x = P.yx
                angle = np.subtract([y,x], [_y,_x]) # dy,dx between node centers
                distance = np.hypot(*angle)  # between node centers
                # or rng * ((P.val+_P.val)/ ave_rval)?:
                if distance <= rng:
                    if len(_P.rim_) < rng: continue
                    mlink = comp_P(_P,P, angle,distance)
                    if mlink:  # return if match
                        V += mlink.derH.Et[0]
                        rng_link_ += [mlink]
                        _P__[i] += [dP.node_[0] for dP in _P.rim_[-1]]  # connected __Ps
            if rng_link_:
                P.rim_ += [rng_link_]
                if _P__[i]: P_ += [P]  # Ps with prelinks for next rng+

        if V <= ave * rng * len(P_) * 6:  # implied val of all __P_s, 6: len mtuple
            break
        else: i_P__ = _P__
        rng += 1
    # der++ in PPds from rng++, no der++ inside rng++: high diff @ rng++ termination only?
    PP.rng=rng  # represents rrdn

    return iP_

def comp_link_(PP):  # node_- mediated: comp node.rim dPs

    dlink_ = []
    for dP in PP.link_:
       for nmed, _rim_ in enumerate(dP.node_[0].rim_):  # link.node_ is CP in 1st der+
           # add fork for CdP node_?
            for _dP in _rim_:
                dlink = comp_P(_dP,dP)
                if dlink:  # return if match
                    dP.rim += [dlink]
                    dlink_ += [dlink]
                    dlink.nmed = nmed
    return dlink_

def comp_P(_P,P, angle=None, distance=None):  # comp dPs if fd else Ps

    fd = isinstance(P,CdP)
    aveP = P_aves[fd]
    if fd:
        # der+: comp dPs
        rn = _P.derH.n / P.derH.n
        # H = comp_latuple(_P.latuple, P.latuple, rn)?
        derH = _P.derH.comp_(P.derH, CH(), rn=rn, flat=1)
        _y,_x = _P.yx; y,x = P.yx
        angle = np.subtract([y,x],[_y,_x]) # dy,dx between node centers
        distance = np.hypot(*angle) # between node centers
    else:
        # rng+: comp Ps
        rn = len(_P.dert_) / len(P.dert_)
        H = comp_latuple(_P.latuple, P.latuple, rn)
        vm = sum(H[::2]); vd = sum(abs(d) for d in H[1::2])
        rm = 1 + vd > vm; rd = 1 + vm >= vd
        n = (len(_P.dert_)+len(P.dert_)) / 2  # der value = ave compared n?
        derH = CH(Et=[vm,vd,rm,rd],H=H,n=n)
        _y,_x = _P.yx; y,x = P.yx
    # get aves:
    yx = [(_c+c)/2 for _c,c in zip((_y,_x),(y,x))]
    latuple = [(P+p)/2 for P,p in zip(_P.latuple[:-1],P.latuple[:-1])] + [[(A+a)/2 for A,a in zip(_P.latuple[-1],P.latuple[-1])]]

    link = CdP(node_=[_P,P],derH=derH, angle=angle,distance=distance,yx=yx,latuple=latuple)
    if link.derH.Et[0] > aveP * link.derH.Et[2]:  # always rng+? (vm > aveP * rm)
        return link

# not revised:
def form_PP_t(root, P_):  # form PPs of dP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    mLink_,_mP__,dLink_,_dP__ = [],[],[],[]  # per PP, !PP.link_?
    for P in P_:
        mlink_,_mP_,dlink_,_dP_ = [],[],[],[]  # per P
        mLink_+=[mlink_]; _mP__+=[_mP_]
        dLink_+=[dlink_]; _dP__+=[_dP_]
        if hasattr(P, "rim"): link_ = P.node_[0].rim  # CdP (get upper mediating link)
        else:                 link_ = [link for rim in P.rim_ for link in rim]# flatten P.link_ nested by rng
        for link in link_:
            if isinstance(link.derH.H[0],CH): m,d,mr,dr = link.derH.H[-1].Et  # last der+ layer vals
            else:                             m,d,mr,dr = link.derH.Et  # H is md_
            if m >= ave * mr:
                mlink_+= [link]; _mP_+= [link.node_[1] if link.node_[0] is P else link.node_[0]]
            if d > ave * dr:  # ?link in both forks?
                dlink_+= [link]; _dP_+= [link.node_[1] if link.node_[0] is P else link.node_[0]]
        # aligned
    for fd, (Link_,_P__) in zip((0,1),((mLink_,_mP__),(dLink_,_dP__))):
        CP_ = []  # all clustered Ps
        for P in P_:
            if P in CP_: continue  # already packed in some sub-PP
            cP_, clink_ = [P], []  # cluster per P
            if P in P_:
                P_index = P_.index(P)
                clink_ += Link_[P_index]
                perimeter = deque(_P__[P_index])  # recycle with breadth-first search, up and down:
                while perimeter:
                    _P = perimeter.popleft()
                    if _P in cP_ or _P in CP_ or _P not in P_: continue  # clustering is exclusive
                    cP_ += [_P]
                    clink_ += Link_[P_.index(_P)]
                    perimeter += _P__[P_.index(_P)]  # extend P perimeter with linked __Ps
            PP = sum2PP(root, cP_, clink_, fd)
            PP_t[fd] += [PP]
            CP_ += cP_

    # eval der+/ PP.link_: correlation clustering, after form_PP_t -> P.root
    for PP in PP_t[1]:
        if PP.iderH.Et[0] * len(PP.link_) > ave_PPd * PP.iderH.Et[2]:
            ider_recursion(root, PP, fd=1)
        if root.fback_:
            feedback(root)  # after der+ in all nodes, no single node feedback

    root.node_ = PP_t  # nested in der+, add_alt_PPs_?

# not revised
def sum2PP(root, P_, dP_, fd):  # sum links in Ps and Ps in PP

    PP = CG(fd=fd, root=root, rng=root.rng+1)
    PP.P_ = P_  # P_ is CdPs if fd, but summed in CG PP?
    iRt = root.iderH.Et[2:4] if root.iderH else [0,0]  # add to rdnt
    # += uplinks:
    for dP in dP_:
        if dP.node_[0] not in P_ or dP.node_[1] not in P_: continue
        if dP.derH:
            # dP.node_[1].derH nesting is lower than dP.derH, append them instead? But how about negate?
            if not isinstance(P_[0], CdP):
                dP.node_[1].derH.add_(dP.derH, iRt)
                dP.node_[0].derH.add_(negate(deepcopy(dP.derH)), iRt)  # negate reverses uplink ds direction
        PP.link_ += [dP]
        if fd: dP.root = PP
        PP.A = np.add(PP.A,dP.angle)
        PP.S += np.hypot(*dP.angle)  # links are contiguous but slanted
        PP.n += dP.derH.n  # *= ave compared P.L?
    # += Ps:
    celly_,cellx_ = [],[]
    for P in P_:
        L = P.latuple[-2]
        PP.area += L; PP.n += L  # no + P.derH.n: current links only?
        PP.latuple = [P+p for P,p in zip(PP.latuple[:-1],P.latuple[:-1])] + [[A+a for A,a in zip(PP.latuple[-1],P.latuple[-1])]]
        if P.derH:
            PP.iderH.add_(P.derH)  # no separate extH, the links are unique here
        if isinstance(P, CP):
            for y,x in P.yx_:
                y = int(round(y)); x = int(round(x))  # summed with float dy,dx in slice_edge?
                PP.box = accum_box(PP.box,y,x); celly_+=[y]; cellx_+=[x]
        if not fd: P.root = PP
    if PP.iderH:
        PP.iderH.Et[2:4] = [R+r for R,r in zip(PP.iderH.Et[2:4], iRt)]

    if isinstance(P_[0], CP):  # skip if P is CdP because it doens't have box and yx
        # pixmap:
        y0,x0,yn,xn = PP.box
        PP.mask__ = np.zeros((yn-y0, xn-x0), bool)
        celly_ = np.array(celly_); cellx_ = np.array(cellx_)
        PP.mask__[(celly_-y0, cellx_-x0)] = True

    return PP

# not revised
def feedback(root):  # in form_PP_, append new der layers to root PP, single vs. root_ per fork in agg+

    HE = deepcopy(root.fback_.pop(0))
    for He in root.fback_:
        HE.add_(He)
    root.iderH.add_(HE.H[-1] if isinstance(HE.H[0], CH) else HE)  # last md_ in H or sum md_

    if root.root and isinstance(root.root, CG):  # skip if root is Edge
        rroot = root.root  # single PP.root, can't be P
        fback_ = rroot.fback_
        node_ = rroot.node_[1] if rroot.node_ and isinstance(rroot.node_[0],list) else rroot.P_  # node_ is updated to node_t in sub+
        fback_ += [HE]
        if len(fback_)==len(node_):  # all nodes terminated and fed back
            feedback(rroot)  # sum2PP adds derH per rng, feedback adds deeper sub+ layers

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
    if fagg:  # add norm m,d, ret= [ret,Ret]:
        # max possible m,d per compared param
        Ret = [max(_L,L),abs(_L)+abs(L), max(_I,I),abs(_I)+abs(I), max(_G,G),abs(_G)+abs(G), max(_M,M),abs(_M)+abs(M), max(_Ma,Ma),abs(_Ma)+abs(Ma), 1,.5]
        mval, dval = sum(ret[::2]),sum(ret[1::2])
        mrdn, drdn = dval>mval, mval>dval
        mdec, ddec = 0, 0
        for fd, (ptuple,Ptuple) in enumerate(zip((ret[::2],ret[1::2]),(Ret[::2],Ret[1::2]))):
            for i, (par, maxv, ave) in enumerate(zip(ptuple, Ptuple, aves)):  # compute link decay coef: par/ max(self/same)
                if fd: ddec += abs(par)/ abs(maxv) if maxv else 1
                else:  mdec += (par+ave)/ (maxv+ave) if maxv else 1

        ret = [mval, dval, mrdn, drdn], [mdec, ddec], ret
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
    # verification:
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

        for fd, PP_ in enumerate(edge.node_):
            plt.imshow(mask, cmap='gray', alpha=0.5)
            plt.title(f"area={edge.area}, {'der+' if fd else 'rng+'}")
            for PP in PP_:
                for dP in PP.link_:
                    (_y, _x), (y, x) = dP.node_[0].yx - yx0, dP.node_[1].yx - yx0
                    plt.plot([_x, x], [_y, y], "o-k")
            plt.show()