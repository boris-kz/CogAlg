import numpy as np
from collections import deque, defaultdict
from copy import deepcopy, copy
from itertools import zip_longest, combinations
import sys
sys.path.append("..")
from frame_blobs import CBase, imread
if __name__ == "__main__": from slice_edge import comp_angle, CsliceEdge
else: from .slice_edge import comp_angle, CsliceEdge

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
                edge.iderH = CH()
                edge.fback_ = []
                for P in edge.P_:
                    P.derH = CH()   # create derH
                    P.rim_ = [[CdP([_P, P]) for _P in P.rim_]]  # prelinks for comp_slice
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
        G.kH = []
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

class CdP(CBase):
    name = "derP"
    def __init__(l, node_=None, derH=None, root=None, distance=0, angle=None):
        super().__init__()

        l.node_ = [] if node_ is None else node_  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.angle = [0,0] if angle is None else angle  # dy,dx between node centers
        l.distance = distance  # distance between node centers
        l.latuple = []  # sum node_
        l.yx = []  # sum node_
        l.derH = CH() if derH is None else derH
        l.root = None if root is None else root  # PPds containing dP

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

    Q = comp_link_(PP) if fd else rng_recursion(PP)  # extend PP.link_, derHs by same-der rng+ comp
    form_PP_t(PP, Q)  # calls der+

    if root is not None and PP.iderH: root.fback_ += [PP.iderH]  # feedback per PPd?


def rng_recursion(PP, fd=0):  # similar to agg+ rng_recursion, but looping and contiguously link mediated

    iP_ = PP.P_
    rng = 1  # cost of links added per rng+
    while True:
        P_ = []; V = 0
        for P in iP_:
            if not hasattr(P, "rim_") or len(P.rim_) < rng: continue  # no _rnglink_ or top row
            _prelink_ = P.rim_.pop()
            rnglink_, prelink_ = [],[]  # both per rng+
            for link in _prelink_:
                if fd:
                    if link.node_[0].rim_: _P_ = link.node_[0].rim_[-1]  # rng uplinks in _P
                    else: continue
                elif link.distance <= rng:  # | rng * ((P.val+_P.val)/ ave_rval)?
                    _P_ = [link.node_[0]]
                else: continue
                for _P in _P_:
                    if not hasattr(_P, "rim_") or len(_P.rim_) < rng: continue
                    mlink = comp_P(CdP(node_=[_P, P]) if fd else link, fd)
                    if mlink: # return if match
                        V += mlink.derH.Et[0]
                        rnglink_ += [mlink]
                        prelink_ += _P.rim_[-1]  # connected __Ps links (can't be prelinks?)
            if rnglink_:
                P.rim_ += [rnglink_]
                if prelink_:
                    P.rim_ += [prelink_]; P_ += [P]  # Ps with prelinks for next rng+

        if V <= ave * rng * len(P_) * 6:  # implied val of all __P_s, 6: len mtuple
            for P in P_: P.rim_.pop()  # remove prelinks
            break
        rng += 1
    # der++ in PPds from rng++, no der++ inside rng++: high diff @ rng++ termination only?
    PP.rng=rng  # represents rrdn

    return iP_


# draft
def comp_link_(PP):

    for link in PP.link_:
        if link.node_[0].rim_:  # empty in top row
            rim_ = copy(link.node_[0].rim_[-1])
            assert not hasattr(link, "rim_")  # no link_ yet
            link.rim_ = [rim_]  # add upper node uplinks as prelinks


def comp_P(link, fd):
    aveP = P_aves[fd]
    _P, P, distance, angle = link.node_[0], link.node_[1], link.distance, link.angle

    if fd:  # der+, comp derPs, comp_latuple is not significant?
        rn = (_P.derH.n if P.derH else len(_P.dert_)) / P.derH.n
        derH = _P.derH.comp_(P.derH, CH(), rn=rn, flat=0)
        link.derH = CH(Et=[*derH.Et], H=[derH])
        # distance, angle?
    else:  # rng+, comp Ps
        rn = len(_P.dert_) / len(P.dert_)
        H = comp_latuple(_P.latuple, P.latuple, rn)
        vm = sum(H[::2]); vd = sum(abs(d) for d in H[1::2])
        rm = 1 + vd > vm; rd = 1 + vm >= vd
        n = (len(_P.dert_)+len(P.dert_)) / 2  # der value = ave compared n?
        _y,_x = _P.yx; y,x = P.yx
        angle = np.subtract([y,x], [_y,_x]) # dy,dx between node centers
        distance = np.hypot(*angle)      # distance between node centers
        link = CdP(node_=[_P, P], derH = CH(Et=[vm,vd,rm,rd],H=H,n=n),angle=angle,distance=distance)

    # to sum in PP
    link.latuple = [P+p for P,p in zip(_P.latuple[:-1],P.latuple[:-1])] + [[A+a for A,a in zip(_P.latuple[-1],P.latuple[-1])]]
    link.yx_     = _P.yx_ + P.yx_

    if link.derH.Et[0] > aveP * link.derH.Et[2]:  # always rng+? (vm > aveP * rm)
        return link

# not revised:
def form_PP_t(root, P_):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    mLink_,_mP__,dLink_,_dP__ = [],[],[],[]  # per PP, !PP.link_?
    for P in P_:
        mlink_,_mP_,dlink_,_dP_ = [],[],[],[]  # per P
        mLink_+=[mlink_]; _mP__+=[_mP_]
        dLink_+=[dlink_]; _dP__+=[_dP_]
        if not (hasattr(P, "rim_") and P.rim_): continue
        for link in [L for rim in P.rim_ for L in rim]:  # flatten P.link_ nested by rng
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
            if P in CP_ or not (hasattr(P, "rim_") and P.rim_): continue  # already packed in some sub-PP
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


def sum2PP(root, P_, derP_, fd):  # sum links in Ps and Ps in PP

    PP = CG(fd=fd, root=root, rng=root.rng+1); PP.P_ = P_  # init P_
    iRt = root.iderH.Et[2:4] if root.iderH else [0,0]  # add to rdnt
    # += uplinks:
    for derP in derP_:
        if derP.node_[0] not in P_ or derP.node_[1] not in P_: continue
        if derP.derH:
            derP.node_[1].derH.add_(derP.derH, iRt)
            derP.node_[0].derH.add_(negate(deepcopy(derP.derH)), iRt)  # negate reverses uplink ds direction
        PP.link_ += [derP]
        if fd: derP.root = PP
        PP.A = np.add(PP.A,derP.angle)
        PP.S += np.hypot(*derP.angle)  # links are contiguous but slanted
        PP.n += derP.derH.n  # *= ave compared P.L?
    # += Ps:
    celly_,cellx_ = [],[]
    for P in P_:
        L = P.latuple[-2]
        PP.area += L; PP.n += L  # no + P.derH.n: current links only?
        PP.latuple = [P+p for P,p in zip(PP.latuple[:-1],P.latuple[:-1])] + [[A+a for A,a in zip(PP.latuple[-1],P.latuple[-1])]]
        if P.derH:
            PP.iderH.add_(P.derH)  # no separate extH, the links are unique here
        for y,x in P.yx_:
            y = int(round(y)); x = int(round(x))  # summed with float dy,dx in slice_edge?
        PP.box = accum_box(PP.box, y, x); celly_+=[y]; cellx_+=[x]
        if not fd: P.root = PP
    if PP.iderH:
        PP.iderH.Et[2:4] = [R+r for R,r in zip(PP.iderH.Et[2:4], iRt)]
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

    frame = CcompSliceFrame(image).segment()  # verification