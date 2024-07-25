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
ave_Gm = 50

class CcompSliceFrame(CsliceEdge):
    class CEdge(CsliceEdge.CEdge): # replaces CBlob

        def vectorize(edge):  # overrides in CsliceEdge.CEdge.vectorize
            edge.slice_edge()
            if edge.latuple[-1] * (len(edge.P_)-1) > ave_PPm:  # eval PP, rdn=1
                comp_slice(edge)
    CBlob = CEdge


class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster

    def __init__(G, root=None, rng=1, fd=0, node_=None, link_=None, Et=None, mdLay=None, derH=None, DerH=None, n=0):
        super().__init__()
        # PP:
        G.root = [] if root is None else root  # mgraphs that contain this G, single-layer
        G.rng = rng
        G.S = 0  # sparsity: distance between node centers
        G.A = 0, 0  # angle: summed dy,dx in links
        G.area = 0
        G.Et = [0,0,0,0] if Et is None else Et  # external eval tuple, summed from rng++ before forming new graph and appending G.extH
        G.latuple = [0,0,0,0,0,[0,0]]  # lateral I,G,M,Ma,L,[Dy,Dx]
        G.mdLay = CH() if derH is None else derH
        G.derH = CH() if derH is None else derH  # nested derH in Gs: [[subH,valt,rdnt,dect]], subH: [[derH,valt,rdnt,dect]]: 2-fork composition layers
        G.DerH = CH() if DerH is None else DerH  # _G.derH summed from krim
        G.node_ = [] if node_ is None else node_  # convert to node_t in sub_recursion
        G.link_ = [] if link_ is None else link_  # links per comp layer, nest in rng+)der+
        G.aRad = 0  # average distance between graph center and node center
        G.box = [np.inf, np.inf, -np.inf, -np.inf]  # y0,x0,yn,xn
        G.yx = [0,0]  # init PP.yx = [(y0+yn)/2,(x0,xn)/2], then ave node yx
        G.kH = []  # kernel: hierarchy of rim layers
        # graph-external, +level per root sub+:
        G.n = n  # external n (last layer n)
        G.rim_ = []  # direct links, depth, init rim_t, link_tH in base sub+ | cpr rd+, link_tHH in cpr sub+
        G.extH = CH()  # G-external daggH( dsubH( dderH, summed from rim links ) krim nodes
        G.alt_graph_ = []  # adjacent gap+overlap graphs, vs. contour in frame_graphs
        # dynamic attrs:
        G.Rim = []  # links to the most mediated nodes
        G.fback_t = [[],[]]  # dderHm_, dderHd_
        G.visited__ = []  # nested in rng++
        # Rdn: int = 0  # for accumulation or separate recursion count?
        # it: list = z([None,None])  # graph indices in root node_s, implicitly nested
        # depth: int = 0  # n sub_G levels over base node_, max across forks
        # nval: int = 0  # of open links: base alt rep
        # id_H: list = z([[]])  # indices in the list of all possible layers | forks, not used with fback merging
        # top aggLay: derH from links, lower aggH from nodes, only top Lay in derG:
        # top Lay from links, lower Lays from nodes, hence nested tuple?

    def __bool__(G): return G.n != 0  # to test empty

class CdP(CBase):  # produced by comp_P, comp_slice version of Clink
    name = "dP"
    def __init__(l, nodet=None, mdLay=None, Et=None, Rt=None, root=None, span=None, angle=None, yx=None, latuple=None):
        super().__init__()

        l.nodet = [] if nodet is None else nodet  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.angle = [0,0] if angle is None else angle  # dy,dx between node centers
        l.span = span  # distance between node centers
        l.latuple = [] if latuple is None else latuple  # sum node_
        l.yx = [0,0] if yx is None else yx  # sum node_
        l.rim = []  # upper 2nd der links
        l.Et = [0,0,0,0] if Et is None else Et
        l.mdLay = [] if mdLay is None else mdLay
        l.Rt = [] if Rt is None else Rt
        l.root = None if root is None else root  # PPds containing dP
        l.nmed = 0  # comp rng: n of mediating Ps between node_ Ps
        # n = 1?
    def __bool__(l): return bool(l.mdLay.H)


class CH(CBase):  # generic derivation hierarchy of variable nesting, depending on effective agg++(sub++ depth
    '''
    If nesting in derH.H may be deleted, we need to directly represent and compare deeper derH.H sub-layers:
    node_H) derH: each layer represents multiple sub-nodes,
    node_H) derH.H_) derH.D_ (nesting is from all prior xcomps, bottom 2D layer is [mdlat,mdLay,mdext])

    deleted H_[i] = n if low-variance individual layers, but D_[i] still has their summed Ders
    for indefinite nesting orders: CH( H = [Q__, Q_, Q...]), len Q__ = len H,
    where each Q is a list of CHs or len of deleted list
    '''
    name = "H"
    def __init__(He, n=0, Et=None, Rt=None, H=None, root=None):
        super().__init__()
        # He.nest = nest  # nesting depth: -1/ ext, 0/ md_, 1/ derH, 2/ subH, 3/ aggH?
        He.H = [] if H is None else H  # hierarchy of der layers or md_, may be [mdlat,mdLay,mdext]s
        He.n = n  # total number of params compared to form derH, summed in comp_G and then from nodes in sum2graph
        He.Et = [0,0,0,0] if Et is None else Et    # evaluation tuple: valt, rdnt
        He.Rt = [0,0] if Rt is None else Rt  # m,d relative to max possible m,d
        He.root = None if root is None else root

    def __bool__(H): return H.n != 0

    def add_md_(HE, He, irdnt=[]):  # p may be derP, sum derLays

        # sum md_s:
        HE.H[:] = [V+v for V,v in zip_longest(HE.H, He.H, fillvalue=0)]
        HE.n += He.n  # combined param accumulation span
        HE.Et = np.add(HE.Et, He.Et)
        if any(irdnt): HE.Et[2:] = [E + e for E, e in zip(HE.Et[2:], irdnt)]
        HE.Rt = np.add(HE.Rt, He.Rt)

    def add_H(HE, He, irdnt=[]):  # unpack down to numericals and sum them

        if HE:
            if isinstance(He.H, CH):  # tentative (H is CH)
                HE.H.add_H(He.H, irdnt)
            else:
                for Lay,lay in zip_longest(HE.H, He.H, fillvalue=None):  # cross comp layer
                    if lay is not None:
                        if Lay and lay.H:  # empty after removing H from rnglay
                            if isinstance(lay.H[0],CH):
                                Lay.add_H(lay, irdnt)  # unpack to add
                            else:
                                Lay.add_md_(lay, irdnt)  # lat md_| Lay md_| ext md_
                        else:
                            if Lay is None: Lay = CH(root=HE)
                            HE.H += [Lay.copy(lay) if lay else []]  # deleted kernel lays
            # default
            HE.Et = np.add(HE.Et, He.Et); HE.Rt = np.add(HE.Rt, He.Rt)
            if any(irdnt): HE.Et[2:] = [E+e for E,e in zip(HE.Et[2:], irdnt)]
            HE.n += He.n  # combined param accumulation span
        else:
            HE.copy(He)  # init
        while HE is not None:
            HE.Et = np.add(HE.Et, He.Et); HE.Rt = np.add(HE.Rt, He.Rt); HE.n += He.n
            HE = HE.root


    def append_(HE,He, irdnt=None, flat=0):

        if irdnt is None: irdnt = []
        if flat:
            for H in He.H: H.root = HE
            HE.H += He.H  # append flat
        else:
            He.root = HE
            HE.H += [He]  # append nested
        Et, et = HE.Et, He.Et
        HE.Et = np.add(HE.Et, He.Et); HE.Rt = np.add(HE.Rt, He.Rt)
        if irdnt: Et[2:4] = [E+e for E,e in zip(Et[2:4], irdnt)]
        HE.n += He.n
        root = HE.root
        while root is not None:
            root.Et = np.add(root.Et, He.Et); root.Rt = np.add(root.Rt, He.Rt); root.n += He.n
            root = root.root
        return HE  # for feedback in agg+


    def comp_md_(_He, He, rn=1, fagg=0, frev=0):

        vm, vd, rm, rd, decm, decd = 0, 0, 0, 0, 0, 0
        derLay = []
        for i, (_d, d) in enumerate(zip(_He.H[1::2], He.H[1::2])):  # compare ds in md_ or ext
            d *= rn  # normalize by comparand accum span
            diff = _d - d
            if frev: diff = -diff  # from link with reversed dir
            match = min(abs(_d), abs(d))
            if (_d < 0) != (d < 0): match = -match  # if only one comparand is negative
            if fagg:
                maxm = max(abs(_d), abs(d))
                decm += abs(match) / maxm if maxm else 1  # match / max possible match
                maxd = abs(_d) + abs(d)
                decd += abs(diff) / maxd if maxd else 1  # diff / max possible diff
            vm += match - aves[i]  # fixed param set?
            vd += diff
            rm += vd > vm; rd += vm >= vd
            derLay += [match, diff]  # flat

        return CH(H=derLay, Et=[vm,vd,rm,rd], Rt=[decm,decd], n=1)


    def comp_H(_He, He, rn=1, fagg=0, frev=0):  # unpack CHs down to numericals and compare them
        DLay = CH()  # merged dderH

        if isinstance(_He.H, CH):  # tentative (H is CH)
             dlay = _He.H.comp_H(He.H, rn, fagg, frev)
             DLay.H = dlay  # same structure when DLay.H = CH
             DLay.Et = copy(dlay.Et); DLay.Rt = copy(dlay.Rt); DLay.n = dlay.n; dlay.root = DLay
        else:
            for _Lay,Lay in zip(_He.H, He.H):  # loop extH s or [mdlat, mdLay, mdext] rng tuples
                if _Lay and Lay:
                    if isinstance(_Lay.H[0], CH):
                        dLay = _Lay.comp_H(Lay, rn, fagg, frev)
                        DLay.add_H(dLay)  # reduce resolution of derivation to fix Lays in derH
                    else:
                        dlay = _Lay.comp_md_(Lay, rn, fagg, frev)  # mdlat | mdLay | mdext
                        DLay.append_(dlay, flat=0)
        ''' 
        full:
        for _Lay,Lay in zip(_He.H, He.H):  # loop extH s
            if _Lay and Lay:
                dLay = CH()
                for _lay,lay in zip(_Lay.H,Lay.H):  # loop [mdlat, mdLay, mdext] rng tuples
                    if _lay and lay:
                        dlay = CH()
                        for E, e in zip(_lay.H, lay.H):  # mdlat | mdLay | mdext
                            dE = E.comp_md_(e, rn, fagg, frev)
                            dlay.append_(dE,flat=0)
                        dLay.append_(dlay, flat=0) '''
        return DLay

    def copy(_H, H):

        for attr, value in H.__dict__.items():
            if attr != '_id' and attr != 'root' and attr in _H.__dict__.keys():  # copy only the available attributes and skip id
                if attr == 'H':  # can't deepcopy CH.root
                    if isinstance(H.H, CH):  # H is CH
                        if not isinstance(_H.H, CH): _H.H = CH(root=_H)
                        H = H.H; _H = _H.H
                    else:
                        if H.H:
                            _H.H = []
                            if isinstance(H.H[0], CH):
                                for lay in H.H:
                                    Lay = CH()
                                    Lay.copy(lay)
                                    _H.H += [Lay]
                            else:  # md_
                                _H.H = deepcopy(H.H)
                else:
                    setattr(_H, attr, deepcopy(value))
        return _H


def comp_slice(edge):  # root function

    edge.mdLay = CH()
    for P in edge.P_:  # add higher links
        P.mdLay = CH()  # for accumulation in sum2PP later (in lower P)
        P.rim_ = []
    rng_recursion(edge)  # vertical P cross-comp -> PP clustering, if lateral overlap
    form_PP_t(edge, edge.P_)
    # der+ / PPd:
    for PP in edge.node_[1]:
        if PP.mdLay.Et[1] * len(PP.link_) > ave_PPd * PP.mdLay.Et[3]:
            comp_link_(PP)  # node-mediated correlation clustering, increment link derH, then P derH in sum2PP:
            form_PP_t(PP, PP.link_)
            edge.mdLay.add_md_(PP.mdLay)

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
                    link = comp_P(_P,P, angle=[dy,dx], distance=np.hypot(dy,dx))
                    if link and link.mdLay.Et[0] > aves[0]*link.mdLay.Et[2]:  # mlink
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

def comp_link_(PP):  # node_- mediated: comp node.rim dPs

    for dP in PP.link_:
        if dP.mdLay.Et[1] > aves[1]:
            for nmed, _rim_ in enumerate(dP.nodet[0].rim_):  # link.nodet is CP
                for _dP in _rim_:
                    dlink = comp_P(_dP,dP)
                    if dlink:
                        dP.rim += [dlink]  # in lower node uplinks
                        dlink.nmed = nmed  # link mediation order0

def comp_P(_P,P, angle=None, distance=None):  # comp dPs if fd else Ps

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
        derLay = CH(Et=[vm,vd,rm,rd],H=H,n=n)
    # get aves:
    latuple = [(P+p)/2 for P,p in zip(_P.latuple[:-1],P.latuple[:-1])] + [[(A+a)/2 for A,a in zip(_P.latuple[-1],P.latuple[-1])]]
    link = CdP(nodet=[_P,P], mdLay=derLay, angle=angle, span=distance, yx=[(_y+y)/2,(_x+x)/2], latuple=latuple)
    # if v > ave * r:
    if link.mdLay.Et[0] > aves[0] * link.mdLay.Et[2] or link.mdLay.Et[1] > aves[1] * link.mdLay.Et[3]:
        return link

def form_PP_t(root, P_):  # form PPs of dP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    mLink_,_mP__,dLink_,_dP__ = [],[],[],[]  # per PP, !PP.link_?
    for P in P_:
        mlink_,_mP_,dlink_,_dP_ = [],[],[],[]  # per P
        mLink_+=[mlink_]; _mP__+=[_mP_]
        dLink_+=[dlink_]; _dP__+=[_dP_]
        link_ = P.rim if hasattr(P,"rim") else [link for rim in P.rim_ for link in rim]
        # get upper links from all rngs of CP.rim_ | CdP.rim
        for link in link_:
            m,d,mr,dr = link.mdLay.H[-1].Et if isinstance(link.mdLay.H[0],CH) else link.mdLay.Et  # H is md_; last der+ layer vals
            _P = link.nodet[0]
            if m >= ave * mr:
                mlink_+= [link]; _mP_+= [_P]
            if d > ave * dr:  # ?link in both forks?
                dlink_+= [link]; _dP_+= [_P]
        # aligned
    for fd, (Link_,_P__) in zip((0,1),((mLink_,_mP__),(dLink_,_dP__))):
        CP_ = []  # all clustered Ps
        for P in P_:
            if P in CP_: continue  # already packed in some sub-PP
            P_index = P_.index(P)
            cP_, clink_ = [P], [*Link_[P_index]]  # cluster per P
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

    root.node_ = PP_t  # nested in der+, add_alt_PPs_?


def sum2PP(root, P_, dP_, fd):  # sum links in Ps and Ps in PP

    PP = CG(fd=fd, root=root, rng=root.rng+1)  # 1st layer of derH is mdLay
    PP.P_ = P_  # P_ is CdPs if fd, but summed in CG PP?
    iRt = root.mdLay.Et[2:4] if root.mdLay else [0,0]  # add to rdnt
    # += uplinks:
    for dP in dP_:
        if dP.nodet[0] not in P_ or dP.nodet[1] not in P_: continue
        dP.nodet[1].mdLay.add_md_(dP.mdLay, iRt)  # add to lower P
        PP.link_ += [dP]
        if fd: dP.root = PP
        PP.A = np.add(PP.A,dP.angle)
        PP.S += np.hypot(*dP.angle)  # links are contiguous but slanted
    # += Ps:
    celly_,cellx_ = [],[]
    for P in P_:
        L = P.latuple[-2]
        PP.area += L; PP.n += L  # no + P.mdLay.n: current links only?
        PP.latuple = [P+p for P,p in zip(PP.latuple[:-1],P.latuple[:-1])] + [[A+a for A,a in zip(PP.latuple[-1],P.latuple[-1])]]
        if P.mdLay:
            PP.mdLay.add_md_(P.mdLay)  # no separate extH, the links are unique here
        if isinstance(P, CP):
            for y,x in P.yx_:
                y = int(round(y)); x = int(round(x))  # summed with float dy,dx in slice_edge?
                PP.box = accum_box(PP.box,y,x); celly_+=[y]; cellx_+=[x]
        if not fd: P.root = PP
    if PP.mdLay:
        PP.mdLay.Et[2:4] = [R+r for R,r in zip(PP.mdLay.Et[2:4], iRt)]
    if isinstance(P_[0], CP):  # CdP has no box, yx
        # pixmap:
        y0,x0,yn,xn = PP.box
        PP.mask__ = np.zeros((yn-y0, xn-x0), bool)
        celly_ = np.array(celly_); cellx_ = np.array(cellx_)
        PP.mask__[(celly_-y0, cellx_-x0)] = True

    return PP

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
        ret = CH(H=ret, Et=[mval,dval,mrdn,drdn], Rt=[mdec,ddec], n=1)  # if fagg only
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
                                f"link not unique between {_node} and {node}. PP.link_:\n" +
                                "\n".join(map(lambda dP: f"dP.id={dP.id}, _node={dP.nodet[0]}, node={dP.nodet[1]}", PP.link_))
                            )
                        nodet_set.add((_node.id, node.id))
                        assert _node.yx < node.yx  # verify that link is up-link
                        (_y, _x), (y, x) = _node.yx - yx0, node.yx - yx0
                        plt.plot([_x, x], [_y, y], "-k")
                plt.show()
                for PP in PP_: draw_PP(PP)

        draw_PP(edge)