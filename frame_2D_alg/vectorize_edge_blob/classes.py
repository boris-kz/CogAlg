import numpy as np
from itertools import count, zip_longest
from math import inf, hypot
from numbers import Real
from typing import Any, NamedTuple, Tuple
from copy import copy
from class_cluster import CBase, CBaseLite, init_param as z
from frame_blobs import Cbox

from .filters import ave_dangle, ave_dI, ave_Pd, ave_Pm, aves
'''
    Conventions:
    prefix 'C'  denotes class
    postfix 't' denotes tuple, multiple ts is a nested tuple
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior or not-self of two same-name variables
    prefix 'f'  denotes flag
    1-3 letter names are normally scalars, except for P and similar classes, 
    capitalized variables are normally summed small-case variables,
    longer names are normally classes
'''

class Cptuple(CBaseLite):

    I: Real = 0
    G: Real = 0
    M: Real = 0
    Ma: Real = 0
    angle: list = z([0,0])
    L: Real = 0

    # operators:
    def __pos__(self): return self
    def __neg__(self): return self.__class__(-self.I, -self.G, -self.M, -self.Ma, -self.angle, -self.L)

    def comp(self, other, rn):

        dI  = self.I  - other.I*rn;  mI  = ave_dI - dI
        dG  = self.G  - other.G*rn;  mG  = min(self.G, other.G*rn) - aves[1]
        dL  = self.L  - other.L*rn;  mL  = min(self.L, other.L*rn) - aves[2]
        dM  = self.M  - other.M*rn;  mM  = get_match(self.M, other.M*rn) - aves[3]  # M, Ma may be negative
        dMa = self.Ma - other.Ma*rn; mMa = get_match(self.Ma, other.Ma*rn) - aves[4]
        mAngle, dAngle = comp_angle(self.angle, other.angle)

        mtuple = [mI, mG, mM, mMa, mAngle-aves[5], mL]
        dtuple = [dI, dG, dM, dMa, dAngle, dL]

        tuplet = [mtuple, dtuple]
        valt = [sum(mtuple), sum(abs(d) for d in dtuple)]
        rdnt = [1+(valt.d>valt.m), 1+(1-(valt.d>valt.m))]   # or rdn = Dval/Mval?

        return tuplet, valt, rdnt


class CP(CBase):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

    ptuple: Cptuple = z(Cptuple())  # latuple: I,G,M,Ma, angle(Dy,Dx), L
    rnpar_H: list = z([])
    derH: CderH = z(CderH())  # [(mtuple, ptuple)...] vertical derivatives summed from P links
    vt: list = z([0,0])  # current rng, not used?
    rt: list = z([1,1])
    dert_: list = z([])  # array of pixel-level derts, ~ node_
    cells: set = z(set())  # pixel-level kernels adjacent to P axis, combined into corresponding derts projected on P axis.
    roott: list = z([None,None])  # PPrm,PPrd that contain this P, single-layer
    axis: list = z([0,0])  # prior slice angle, init sin=0,cos=1
    yx: list = z([0,0])
    link_: list = z([])  # uplinks per comp layer, nest in rng+)der+
    ''' 
    dxdert_: list = z([])  # only in Pd
    Pd_: list = z([])  # only in Pm
    Mdx: int = 0  # if comp_dx
    Ddx: int = 0
    '''

    def comp(self, other, link_, rn, A, S):

        dertuplet, valt, rdnt = self.ptuple.comp(other.ptuple, rn=rn)

        if valt.m > ave_Pm * rdnt.m or valt.d > ave_Pm * rdnt.d:
            derH = CderH([dertuplet])
            link_ += [CderP(derH=derH, valt=valt, rdnt=rdnt, _P=self, P=other, A=A, S=S)]


class CderP(CBase):  # tuple of derivatives in P link: binary tree with latuple root and vertuple forks

    _P: CP  # higher comparand
    P: CP  # lower comparand
    derH: CderH = z(CderH())  # [[[mtuple,dtuple],[mval,dval],[mrdn,drdn]]], single ptuplet in rng+
    vt: list = z([0,0])
    rt: list = z([1,1])  # mrdn + uprdn if branch overlap?
    roott: list = z([None, None])  # PPdm,PPdd that contain this derP
    S: float = 0.0  # sparsity: distance between centers
    A: list = z([0,0])  # angle: dy,dx between centers
    # roott: list = z([None, None])  # for der++, if clustering is per link

    def comp(self, link_, rn):
        dderH, valt, rdnt = self._P.derH.comp(self.P.derH, rn=rn, n=len(self.derH))

        if valt.m > ave_Pd * rdnt.m or valt.d > ave_Pd * rdnt.d:
            self.derH |= dderH; self.valt = valt; self.rdmt = rdnt  # update derP not form new one
            link_ += [self]
'''
len layers with ext: 2, 3, 6, 12, 24... 
max n of tuples per der layer = summed n of tuples in all lower layers: 1, 1, 2, 4, 8..:
lay1: par     # derH per param in vertuple, layer is derivatives of all lower layers:
lay2: [m,d]   # implicit nesting, brackets for clarity:
lay3: [[m,d], [md,dd]]: 2 sLays,
lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays
'''

class Cgraph(CBase):  # params of single-fork node_ cluster

    fd: int = 0  # fork if flat layers?
    ptuple: Cptuple = z(Cptuple())  # default P
    derH: CderH = z(CderH())  # from PP, not derHv
    # graph-internal, generic:
    aggH: list = z([])  # [[subH,valt,rdnt,dect]], subH: [[derH,valt,rdnt,dect]]: 2-fork composition layers
    valt: list = z([0,0])  # sum ptuple, derH, aggH
    rdnt: list = z([1,1])
    dect: list = z([0,0])
    link_: list = z([])  # internal, single-fork, incrementally nested
    node_: list = z([])  # base node_ replaced by node_t in both agg+ and sub+, deeper node-mediated unpacking in agg+
    # graph-external, +level per root sub+:
    rimH: list = z([])  # direct links, depth, init rim_t, link_tH in base sub+ | cpr rd+, link_tHH in cpr sub+
    RimH: list = z([])  # links to the most mediated nodes
    extH: list = z([])  # G-external daggH( dsubH( dderH, summed from rim links
    evalt: list = z([0,0])  # sum from esubH
    erdnt: list = z([1,1])
    edect: list = z([0,0])
    ext: list = z([0,0,[0,0]])  # L,S,A: L len base node_, S sparsity: average link len, A angle: average link dy,dx
    rng: int = 1
    box: Cbox = z(Cbox(inf,inf,-inf,-inf))  # y,x,y0,x0,yn,xn
    # tentative:
    alt_graph_: list = z([])  # adjacent gap+overlap graphs, vs. contour in frame_graphs
    avalt: list = z([0,0])  # sum from alt graphs to complement G aves?
    ardnt: list = z([1,1])
    adect: list = z([0,0])
    # PP:
    P_: list = z([])
    mask__: object = None
    # temporary, replace with Et:
    Vt: list = z([0,0])  # last layer | last fork tree vals for node_connect and clustering
    Rt: list = z([1,1])
    Dt: list = z([0,0])
    root: list = z([None])  # for feedback
    fback_: list = z([])  # feedback [[aggH,valt,rdnt,dect]] per node layer, maps to node_H
    compared_: list = z([])
    Rdn: int = 0  # for accumulation or separate recursion count?

    # it: list = z([None,None])  # graph indices in root node_s, implicitly nested
    # depth: int = 0  # n sub_G levels over base node_, max across forks
    # nval: int = 0  # of open links: base alt rep
    # id_H: list = z([[]])  # indices in the list of all possible layers | forks, not used with fback merging
    # top aggLay: derH from links, lower aggH from nodes, only top Lay in derG:
    # top Lay from links, lower Lays from nodes, hence nested tuple?


class CderG(CBase):  # params of single-fork node_ cluster per pplayers

    _G: Cgraph  # comparand + connec params
    G: Cgraph
    daggH: list = z([])  # optionally nested dderH ) dsubH ) daggH: top aggLev derived in comp_G
    Vt: list = z([0,0])  # last layer vals from comp_G
    Rt: list = z([1,1])
    Dt: list = z([0,0])
    S: float = 0.0  # sparsity: average distance to link centers
    A: list = z([0,0])  # angle: average dy,dx to link centers
    roott: list = z([None,None])
    # dir: bool  # direction of comparison if not G0,G1, only needed for comp link?


def get_match(_par, par):
    match = min(abs(_par),abs(par))
    return -match if (_par<0) != (par<0) else match    # match = neg min if opposite-sign comparands

def acc_(a_,b_, n=1): # accum iterables

    a_[:] = np.add(a_,b_) if a_ else copy(b_)
    np.divide(a_,n)

def add_(a_,b_, n=1):  # sum iterables
    return np.divide( np.add(a_,b_) if a_ else [b for b in b_], n)

def sub_(a_,b_):  # subtract iterables
    return [a - b for a, b in zip(a_, b_)]

def comp_(self, other, rn):  # compare tuples, include comp_ext and comp_ptuple?

    mtuple, dtuple = [], []
    for _par, par, ave in zip(self, other, aves):
        npar = par * rn
        mtuple += [get_match(_par, npar) - ave]
        dtuple += [_par - npar]

    tuplet = [mtuple,dtuple]
    valt = [sum(mtuple), sum(abs(d) for d in dtuple)]
    rdnt = [valt.d > valt.m, valt.d < valt.m]

    return tuplet, valt, rdnt

def comp_angle(self, other):  # angle = [dy,dx], rn doesn't matter

    _sin, sin = self.normalize()
    _cos, cos = other.normalize()

    dangle = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    # cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
    mangle = ave_dangle - abs(dangle)  # inverse match, not redundant if sum cross sign

    return [mangle, dangle]

def normalize(self):
    dist = abs(self)
    return self.__class__(self.dy / dist, self.dx / dist)


# draft
class CderH(CBase):  # derH is a list of der layers or sub-layers, each = ptuple_tv

    H: list = z([])
    valt: list = z([0,0])
    rdnt: list = z([1,1])
    dect: list = z([0,0])
    depth: int = 0

    @classmethod
    def empty_layer(cls): return list([[0,0,0,0,0,0], [0,0,0,0,0,0]])

    def __add__(Hv, hv, irdn, fneg=0, fagg=0):
        if hv:
            if Hv:
                H, Valt, Rdnt, Dect, Extt, Depth = Hv
                h, valt, rdnt, dect, extt, depth = hv
                Valt[:] = np.add(Valt,valt)
                Rdnt[:] = np.add( np.add(Rdnt,rdnt), [irdn,irdn])
                Rdnt[0] += Valt[1] > Valt[0]
                Rdnt[1] += Valt[0] > Valt[1]
                if fagg:
                    Dect[:] = np.divide( np.add(Dect,dect), 2)
                fC=0
                if isinstance(H, CderH):
                    fC=1
                    if isinstance(h, list):  # convert dertv to derH:
                        h = [CderH(H=h, valt=copy(hv.valt), rdnt=copy(hv.rdnt), dect=copy(hv.dect), ext=copy(hv.ext), depth=0)]
                elif isinstance(h, CderH):
                    fC=1; H = [CderH(H=H, valt=copy(Hv.valt), rdnt=copy(Hv.rdnt), dect=copy(Hv.dect), ext=copy(Hv.ext), depth=0)]

                if fC:  # both derH_:
                    H = [DerH + derH for DerH, derH in zip_longest(H,h)]
                else:  # both dertuplets:
                    H = [list(np.add(Dertuple,dertuple)) for Dertuple, dertuple in zip(H,h)]  # mtuple,dtuple
            else:
                Hv[:] = deepcopy(hv)
        return Hv

    def __iadd__(self, other): return self + other
    def __isub__(self, other): return self - other

    def __sub__(self, other):

        # pending update, same with __add__ above
        # subtract der layers, dertuple is mtuple | dtuple
        if other.H and other.H[0] and isinstance(other.H[0][0], list):
            H = [[ list(np.subtract(Dertuple,dertuple))  for Dertuple, dertuple in zip(Dertuplet, dertuplet)]
                 for Dertuplet, dertuplet in zip_longest(self.H, other.H, fillvalue=self.empty_layer())]  # mtuple,dtuple
        else:
            H = [list(np.subtract(Dertuple,dertuple)) for Dertuple, dertuple in zip_longest(self.H, other.H, fillvalue=[0,0,0,0,0,0])]  # mtuple,dtuple

        valt = np.subtract(self.valt, other.valt)
        rdnt = np.subtract(self.rdnt, other.rdnt); rdnt[0] -= valt[1] > valt[0]; rdnt[1] -= valt[0] > valt[1]
        dect = np.subtract(np.multiply(self.dect,2), other.dect)

        return CderH(H=H, valt=valt, rdnt=rdnt, dect=dect)



