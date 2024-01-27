from __future__ import annotations

from itertools import count, zip_longest
from math import inf, hypot
from numbers import Real
from typing import Any, List, NamedTuple, Tuple

from class_cluster import CBase, CBaseLite, init_param as z
from frame_blobs import Cbox

from .filters import ave_dangle, ave_dI, ave_Pd, ave_Pm, aves
'''
    Conventions:
    prefix 'C'  denotes class
    postfix 't' denotes tuple, multiple ts is a nested tuple
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    1-3 letter names are normally scalars, except for P and similar classes, 
    capitalized variables are normally summed small-case variables,
    longer names are normally classes
'''

class Cangle(NamedTuple):
    dy: Real
    dx: Real
    # operators:
    def __abs__(self) -> Real: return hypot(self.dy, self.dx)
    def __pos__(self) -> Cangle: return self
    def __neg__(self) -> Cangle: return Cangle(-self.dy, -self.dx)
    def __add__(self, other: Cangle) -> Cangle: return Cangle(self.dy + other.dy, self.dx + other.dx)
    def __sub__(self, other: Cangle) -> Cangle: return self + (-other)

    def normalize(self) -> Cangle:
        dist = abs(self)
        return Cangle(self.dy / dist, self.dx / dist)

    def comp(self, other: Cangle) -> Cmd:  # rn doesn't matter for angles

        # angle = [dy,dx]
        _sin, sin = self.normalize()
        _cos, cos = other.normalize()

        dangle = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
        # cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
        mangle = ave_dangle - abs(dangle)  # inverse match, not redundant if sum cross sign

        return Cmd(mangle, dangle)

class Cmd(CBaseLite):     # m | d tuple
    m: Any
    d: Any


class Cptuple(CBaseLite):

    I: Real = 0
    G: Real = 0
    M: Real = 0
    Ma: Real = 0
    angle: Cangle = Cangle(0, 0)
    L: Real = 0

    # operators:
    def __pos__(self) -> Cptuple: return self
    def __neg__(self) -> Cptuple: return self.__class__(-self.I, -self.G, -self.M, -self.Ma, -self.angle, -self.L)

    def comp(self, other: Cptuple, rn: Real) -> Tuple[Cmd, Cmd, Cmd]:

        dI  = self.I  - other.I*rn;  mI  = ave_dI - dI
        dG  = self.G  - other.G*rn;  mG  = min(self.G, other.G*rn) - aves[1]
        dL  = self.L  - other.L*rn;  mL  = min(self.L, other.L*rn) - aves[2]
        dM  = self.M  - other.M*rn;  mM  = get_match(self.M, other.M*rn) - aves[3]  # M, Ma may be negative
        dMa = self.Ma - other.Ma*rn; mMa = get_match(self.Ma, other.Ma*rn) - aves[4]
        mAngle, dAngle = self.angle.comp(other.angle)

        mtuple = Cdertuple(mI, mG, mM, mMa, mAngle-aves[5], mL)
        dtuple = Cdertuple(dI, dG, dM, dMa, dAngle, dL)

        dertuplet = Cmd(mtuple, dtuple)
        valt = Cmd(sum(mtuple), sum(abs(d) for d in dtuple))
        rdnt = Cmd(1+(valt.d>valt.m), 1+(1-(valt.d>valt.m)))   # or rdn = Dval/Mval?

        return dertuplet, valt, rdnt

class Cdertuple(Cptuple):

    angle: Real = 0

    def comp(self, other: Cdertuple, rn: Real) -> Tuple[Cmd, Cmd, Cmd]:    # comp_dertuple

        mtuple, dtuple = [], []
        for _par, par, ave in zip(self, other, aves):  # compare ds only
            npar = par * rn
            mtuple += [get_match(_par, npar) - ave]
            dtuple += [_par - npar]

        ddertuplet = Cmd(Cdertuple(*mtuple), Cdertuple(*dtuple))
        valt = Cmd(sum(mtuple), sum(abs(d) for d in dtuple))
        rdnt = Cmd(valt.d > valt.m, valt.d < valt.m)

        return ddertuplet, valt, rdnt


class CderH(list):  # derH is a list of der layers or sub-layers, each = ptuple_tv
    __slots__ = []

    @classmethod
    def empty_layer(cls):
        return Cmd(Cdertuple(), Cdertuple())

    def __or__(self, other: CderH) -> CderH:
        return CderH([*self, *other])

    def __add__(self, other: CderH) -> CderH:
        return CderH((
            # sum der layers, dertuple is mtuple | dtuple
            Dertuplet + dertuplet for Dertuplet, dertuplet
            in zip_longest(self, other, fillvalue=self.empty_layer())  # mtuple,dtuple
        ))

    def __sub__(self, other: CderH) -> CderH:
        return CderH((
            # sum der layers, dertuple is mtuple | dtuple
            Dertuplet - dertuplet for Dertuplet, dertuplet
            in zip_longest(self, other, fillvalue=self.empty_layer())  # mtuple,dtuple
        ))

    def comp(self, other: CderH, rn: Real, n: Real = inf) -> Tuple[CderH, Cmd, Cmd]:

        dderH = CderH([])  # or not-missing comparand: xor?
        valt, rdnt = Cmd(0, 0), Cmd(1, 1)

        for i, _lay, lay in zip(count(), self, other):  # compare common lower der layers | sublayers in derHs
            if i >= n: break
            # if lower-layers match: Mval > ave * Mrdn?
            dertuplet, _valt, _rdnt = _lay.d.comp(lay.d, rn)  # compare dtuples only
            dderH |= [dertuplet]; valt += _valt; rdnt += _rdnt

        return dderH, valt, rdnt  # new derLayer,= 1/2 combined derH


class CP(CBase):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

    ptuple: Cptuple = z(Cptuple())  # latuple: I,G,M,Ma, angle(Dy,Dx), L
    rnpar_H: list = z([])
    derH: CderH = z(CderH())  # [(mtuple, ptuple)...] vertical derivatives summed from P links
    valt: Cmd = z(Cmd(0, 0))  # summed from the whole derH
    rdnt: Cmd = z(Cmd(1, 1))
    dert_: list = z([])  # array of pixel-level derts, ~ node_
    cells: set = z(set())  # pixel-level kernels adjacent to P axis, combined into corresponding derts projected on P axis.
    roott: list = z([None,None])  # PPrm,PPrd that contain this P, single-layer
    axis: Cangle = Cangle(0, 1)  # prior slice angle, init sin=0,cos=1
    yx: tuple = None
    ''' 
    add L,S,A from links?
    optional:
    link_H: list = z([[]])  # all links per comp layer, rng+ or der+
    dxdert_: list = z([])  # only in Pd
    Pd_: list = z([])  # only in Pm
    Mdx: int = 0  # if comp_dx
    Ddx: int = 0
    '''

    def comp(self, other: CP, link_: List[CderP], rn: Real, S: Real = None):
        dertuplet, valt, rdnt = self.ptuple.comp(other.ptuple, rn=rn)
        if valt.m > ave_Pm * rdnt.m or valt.d > ave_Pm * rdnt.d:
            derH = CderH([dertuplet])
            link_ += [CderP(derH=derH, valt=valt, rdnt=rdnt, _P=self, P=other, S=S)]


class CderP(CBase):  # tuple of derivatives in P link: binary tree with latuple root and vertuple forks

    _P: CP  # higher comparand
    P: CP  # lower comparand
    derH: CderH = z(CderH())  # [[[mtuple,dtuple],[mval,dval],[mrdn,drdn]]], single ptuplet in rng+
    valt: Cmd = z(Cmd(0, 0))  # replace with Vt?
    rdnt: Cmd = z(Cmd(1, 1))  # mrdn + uprdn if branch overlap?
    roott: list = z([None, None])  # PPdm,PPdd that contain this derP
    S: float = 0.0  # sparsity: distance between centers
    A: Cangle = None  # angle: dy,dx between centers
    # roott: list = z([None, None])  # for der++, if clustering is per link

    def comp(self, link_: List[CderP], rn: Real):
        dderH, valt, rdnt = self._P.derH.comp(self.P.derH, rn=rn, n=len(self.derH))
        if valt.m > ave_Pd * rdnt.m or valt.d > ave_Pd * rdnt.d:
            self.derH |= dderH; self.valt = valt; self.rdmt = rdnt  # update derP not form new one
            link_ += [self]

'''
max n of tuples per der layer = summed n of tuples in all lower layers: 1, 1, 2, 4, 8..:
lay1: par     # derH per param in vertuple, layer is derivatives of all lower layers:
lay2: [m,d]   # implicit nesting, brackets for clarity:
lay3: [[m,d], [md,dd]]: 2 sLays,
lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays
'''


class Cgraph(CBase):  # params of single-fork node_ cluster per pplayers

    fd: int = 0  # fork if flat layers?
    ptuple: Cptuple = z(Cptuple())  # default P
    derH: CderH = z(CderH())  # from PP, not derHv
    # graph-internal, generic:
    aggH: list = z([])  # [[subH,valt,rdnt,dect]], subH: [[derH,valt,rdnt,dect]]: 2-fork composition layers
    valt: Cmd = z(Cmd(0, 0))  # sum ptuple, derH, aggH
    rdnt: Cmd = z(Cmd(1, 1))
    dect: Cmd = z(Cmd(0, 0))
    link_: List[CderP | CderG] = z([])  # internal, single-fork
    node_: list = z([])  # base node_ replaced by node_t in both agg+ and sub+, deeper node-mediated unpacking in agg+
    # graph-external, +level per root sub+:
    fHH: int = 0  # nesting added per agg+
    rim_t: object = None  # direct links, depth, init rim_t, link_tH in base sub+ | cpr rd+, link_tHH in cpr sub+
    Rim_t: object = None  # links to furthest mediated evaluated nodes
    ext_H: list = z([])  # G-external daggH( dsubH( dderH, summed from rim links
    evalt: list = z([0,0])  # sum from esubH
    erdnt: list = z([1,1])
    edect: list = z([0,0])
    # ext params:
    L: int = 0 # len base node_; from internal links:
    S: float = 0.0  # sparsity: average distance to link centers
    A: list = z([0,0])  # angle: average dy,dx to link centers
    rng: int = 1
    box: Cbox = Cbox(inf,inf,-inf,-inf)  # y0,x0,yn,,xn
    # tentative:
    alt_graph_: list = z([])  # adjacent gap+overlap graphs, vs. contour in frame_graphs
    avalt: Cmd = z(Cmd(0, 0))  # sum from alt graphs to complement G aves?
    ardnt: Cmd = z(Cmd(1, 1))
    adect: Cmd = z(Cmd(0, 0))
    # PP:
    P_: list = z([])
    mask__: object = None
    # temporary:
    Vt: Cmd = z(Cmd(0, 0))  # last layer | last fork tree vals for node_connect and clustering
    Rt: Cmd = z(Cmd(1, 1))
    Dt: Cmd = z(Cmd(0, 0))
    it: list = z([None,None])  # graph indices in root node_s, implicitly nested
    roott: list = z([None,None])  # for feedback
    fback_t: list = z([[],[],[]])  # feedback [[aggH,valt,rdnt,dect]] per node fork, maps to node_H
    compared_: list = z([])
    Rdn: int = 0  # for accumulation or separate recursion count?
    # not used:
    depth: int = 0  # n sub_G levels over base node_, max across forks
    nval: int = 0  # of open links: base alt rep
    # id_H: list = z([[]])  # indices in the list of all possible layers | forks, not used with fback merging
    # top aggLay: derH from links, lower aggH from nodes, only top Lay in derG:
    # top Lay from links, lower Lays from nodes, hence nested tuple?


class CderG(CBase):  # params of single-fork node_ cluster per pplayers

    _G: Cgraph  # comparand + connec params
    G: Cgraph
    daggH: list = z([])  # optionally nested dderH ) dsubH ) daggH: top aggLev derived in comp_G
    Vt: Cmd = z(Cmd(0,0))  # last layer vals from comp_G
    Rt: Cmd = z(Cmd(1,1))
    Dt: Cmd = z(Cmd(0,0))
    S: float = 0.0  # sparsity: average distance to link centers
    A: Cangle = None  # angle: average dy,dx to link centers
    roott: list = z([None,None])
    # dir: bool  # direction of comparison if not G0,G1, only needed for comp link?


def get_match(_par, par):
    match = min(abs(_par),abs(par))
    return -match if (_par<0) != (par<0) else match    # match = neg min if opposite-sign comparands