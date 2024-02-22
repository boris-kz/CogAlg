import numpy as np
from itertools import count, zip_longest
from math import inf, hypot
from numbers import Real
from typing import Any, NamedTuple, Tuple
from copy import copy, deepcopy
from class_cluster import CBase, CBaseLite
from types import SimpleNamespace
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

# not revised:

def add_(HE, He):  # unpack tuples (formally lists) down to numericals and sum them

    if He:
        if HE:
            for i, (Lay,lay) in enumerate(zip_longest(HE, He, fillvalue=None)):  # always list He
                if lay:
                    if Lay:
                        if isinstance(Lay[2][0],list):  # and isinstance(lay[2][0],list):
                            # and par[0]==Par[0]: if same typ, or always aligned unless compressed: cpr version?
                            HE[i] = add_(Lay[1:], lay[1:])  # unpack and sum Ets and Hs
                        else:
                            HE[i] = np.add(Lay,lay)  # both have numericals in Hs
                    else:
                        HE += [deepcopy(lay)]  # skip deepcopy if numerical?
            return HE
        else:
            return deepcopy(He)
'''
add negate(t)
'''

def comp_(_H,H):  # unpack tuples (formally lists) down to numericals and compare them

    Et = 0,0,0,0,0,0  # Vm,Vd, Rm,Rd, Dm,Dd
    dH = []
    for _lay,lay in zip(_H,H):  # md_| ext | derH | subH | aggH, compare shared layers

        if isinstance(_lay[0], str):  # nested H, must be same for lay
            et, dlay = comp_(_lay,lay)  # unpack and comp params
        else:
            # lay Hs are md_s, numerical comp:
            vm,vd,rm,rd,dm,dd = 0,0,0,0,0,0
            dlay = []
            for _d,d in zip(_lay[2][1::2], lay[2][1::2]):  # compare ds in md_ or ext
                diff = _d-d
                match = min(abs(_d),abs(d))
                if (_d<0) != (d<0): match *= -1  # if only one comparand is negative
                vm += match; vd += diff
                dlay += [match,diff]  # flat list

            et = [vm,vd,rm,rd,dm,dd]
        for E,e in Et,et: E+=2
        dH += [dlay]

    return Et, dH


def comp_angle(_A, A):  # angle = [dy,dx], rn doesn't matter

    _dy,_dx = _A; _g = np.hypot(_dy,_dx); _sin = _dy/_g; sin = _dx/_g
    dy,dx = A;    g = np.hypot(dy,dx);    _cos =  dy/g;  cos = dx/g

    dangle = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    # cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
    mangle = ave_dangle - abs(dangle)  # inverse match, not redundant if sum cross sign

    return [mangle, dangle]


# not revised:
class z(SimpleNamespace):

    def __init__(self, **kwargs):  # this is not working here, their default value is still referencing a same id
        super().__init__(**kwargs)

    def __iadd__(T, t): return T + t
    def __isub__(T, t): return T - t

    def __add__(T, t):

        # sum ptuple
        if T.typ == "ptuple":
            for param in list(T.__dict__):
                V = getattr(T, param); v = getattr(t, param)
                if isinstance(V, list): add_(V, v)
                else:                   setattr(T, param, V+v)

        if T.typ == "derH":
            if t.H:
                if T.H:
                    H, Valt, Rdnt, Dect, Extt, Depth = T.H, T.valt, T.rdnt, T.dect, T.ext, t.depth
                    h, valt, rdnt, dect, extt, depth = t.H, t.valt, t.rdnt, t.dect, t.ext, t.depth
                    Valt[:] = np.add(Valt,valt)
                    Rdnt[:] = np.add( np.add(Rdnt,rdnt), [T.irdn,t.irdn])
                    Rdnt[0] += Valt[1] > Valt[0]
                    Rdnt[1] += Valt[0] > Valt[1]
                    if T.fagg:
                        Dect[:] = np.divide( np.add(Dect,dect), 2)
                    fC=0
                    if isinstance(H[0], z):
                        fC=1
                        if isinstance(h[0], list):  # convert dertv to derH:
                            h = [CderH(H=h, valt=copy(t.valt), rdnt=copy(t.rdnt), dect=copy(t.dect), ext=copy(t.ext), depth=0)]
                    elif isinstance(h[0], z):
                        fC=1; H = [CderH(H=H, valt=copy(T.valt), rdnt=copy(T.rdnt), dect=copy(T.dect), ext=copy(T.ext), depth=0)]

                    if fC:  # both derH_:
                        add_(H, h)
                        # H[:(len(h))] = [DerH + derH for DerH, derH in zip_longest(H,h)]  # if different length or always same? (could be different length)
                    else:  # both dertuplets:
                        H[:] = [list(np.add(Dertuple,dertuple)) for Dertuple, dertuple in zip(H,h)]  # mtuple,dtuple
                else:
                    T.H[:] = deepcopy(t.H)

        return T

# check and update new default value
def init_default(instance, params_set, default_value):
    for param, value in zip(params_set, default_value):
        if getattr(instance, param) is None: setattr(instance, param, deepcopy(value))  # deepcopy prevent list has a same reference

# predefined set of params for each instances
def Cptuple(typ="ptuple",I=None, G=None, M=None, Ma=None, angle=None, L=None):
    params_set = ("I", "G", "M", "Ma", "angle", "L")
    default_value = (0,0,0,0,[0,0], 0)
    instance = z(typ=typ,I=I, G=G, M=M, Ma=Ma, angle=angle, L=L)
    init_default(instance, params_set, default_value)
    return  instance

def CderH(typ="derH", H=None, valt=None, rdnt=None, dect=None, ext=None, depth=None,irdn=None, fagg=None):
    params_set = ("H", "valt", "rdnt", "dect", "ext", "depth", "irdn", "fagg")
    default_value = ([],[0,0],[1,1],[0,0],[], 0, 0, 0)
    instance = z(typ=typ, H=H, valt=valt, rdnt=rdnt, dect=dect, ext=ext, depth=depth, irdn=irdn, fagg=fagg)
    init_default(instance, params_set, default_value)
    return instance

def Cedge(typ="edge",root=None, node_=None, box=None, mask__=None, Rt=None, valt=None, rdnt=None, derH=None,fback_=None):
    params_set = ("root", "node_", "box", "mask__", "Rt", "valt", "rdnt", "derH", "fback_")
    default_value = (None,[[],[]],[inf,inf,-inf,-inf],None,[1,1],[0,0],[1,1], CderH(), [])
    instance = z(typ=typ, root=root, node_=node_, box=box, mask__=mask__, Rt=Rt, valt=valt, rdnt=rdnt, derH=derH,fback_=fback_)
    init_default(instance, params_set, default_value)
    return instance

def CP(typ="P", yx=None, axis=None, cells=None, dert_=None,derH=None,link_=None):
    params_set = ("yx", "axis", "cells", "dert_", "derH", "link_")
    default_value = ([0,0],[0,0],{},[],CderH(), [[]])
    instance = z(typ=typ, yx=yx,axis=axis,cells=cells,dert_=dert_, derH=derH, link_=link_)
    init_default(instance, params_set, default_value)
    return instance

def CderP(typ="derP", P=None,_P=None, derH=CderH(), vt=[0,0], rt=[1,1], S=0, A=[0,0], roott=[[],[]]):
    params_set = ("P", "_P", "derH", "vt", "rt", "S", "A", "roott")
    default_value = (None,None,CderH(),[0,0],[1,1], 0, 0, [[],[]])
    instance = z(typ=typ, P=P, _P=_P, derH=derH, vt=vt, rt=rt, S=S, A=A, roott=roott)
    init_default(instance, params_set, default_value)
    return instance

def Cgraph(typ="graph",
           fd = None,  # fork if flat layers?
           ptuple = None,  # default P
           derH = None,  # from PP, not derHv
           # graph-internal, generic:
           aggH = None,  # [[subH,valt,rdnt,dect]], subH: [[derH,valt,rdnt,dect]]: 2-fork composition layers
           valt = None,  # sum ptuple, derH, aggH
           rdnt = None,
           dect = None,
           link_ = None,  # internal, single-fork, incrementally nested
           node_ = None,  # base node_ replaced by node_t in both agg+ and sub+, deeper node-mediated unpacking in agg+
            # graph-external, +level per root sub+:
           rimH = None,  # direct links, depth, init rim_t, link_tH in base sub+ | cpr rd+, link_tHH in cpr sub+
           RimH = None,  # links to the most mediated nodes
           extH = None, # G-external daggH( dsubH( dderH, summed from rim links
           evalt = None,  # sum from esubH
           erdnt = None,
           edect = None,
           ext = None,  # L,S,A: L len base node_, S sparsity: average link len, A angle: average link dy,dx
           rng = None,
           box = None,  # y,x,y0,x0,yn,xn
           # tentative:
           alt_graph_ = None,  # adjacent gap+overlap graphs, vs. contour in frame_graphs
           avalt = None,  # sum from alt graphs to complement G aves?
           ardnt = None,
           adect = None,
           # PP:
           P_ = None,
           mask__ = None,
           # temporary, replace with Et:
           Vt = None, # last layer | last fork tree vals for node_connect and clustering
           Rt = None,
           Dt = None,
           root = None,  # for feedback
           fback_ = None, # feedback [[aggH,valt,rdnt,dect]] per node layer, maps to node_H
           compared_ = None,
           Rdn = None, # for accumulation or separate recursion count?

           it = None,  # graph indices in root node_s, implicitly nested
           depth = None,  # n sub_G levels over base node_, max across forks
           nval = None, # of open links: base alt rep
           id_H = None):  # indices in the list of all possible layers | forks, not used with fback merging

    params_set = ("fd","ptuple", "derH",
                  "aggH", "valt", "rdnt", "dect", "link_", "node_",
                  "rimH", "RimH", "extH", "evalt", "erdnt", "edect", "ext", "rng", "box",
                  "alt_graph_","avalt","ardnt","adect",
                  "P_","mask__",
                  "Vt","Rt","Dt","root","fback_","compared_","Rdn",
                  "it","depth","nval","id_H")

    default_value = (0,Cptuple(), CderH(),
                     [], [0,0], [1,1], [0,0], [], [],
                     [], [], [], [0,0], [1,1], [0,0], [0,0, [0,0]], 1, [inf,inf,-inf,-inf],
                     [],[0,0],[1,1],[0,0],
                     [],None,
                     [0,0],[1,1],[0,0],[None],[],[],0,
                     [None, None],0,0,[[]])

    instance = z(typ=typ, fd=fd,ptuple=ptuple, derH=derH,
                 aggH=aggH, valt=valt, rdnt=rdnt, dect=dect, link_=link_, node_=node_,
                 rimH=rimH, RimH=RimH, extH=extH, evalt=evalt, erdnt=erdnt, edect=edect, ext=ext, rng=rng, box=box,
                 alt_graph_=alt_graph_,avalt=avalt,ardnt=ardnt,adect=adect,
                 P_=P_,mask__=mask__,
                 Vt=Vt,Rt=Rt,Dt=Dt,root=root,fback_=fback_,compared_=compared_,Rdn=Rdn,
                 it=it,depth=depth,nval=nval,id_H=id_H)
    init_default(instance, params_set, default_value)

    return instance

# old, only to check for consistency:

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

def comp_d(self, other, rn):  # compare tuples, include comp_ext and comp_ptuple?

    mtuple, dtuple = [], []
    for _par, par, ave in zip(self, other, aves):
        npar = par * rn
        mtuple += [get_match(_par, npar) - ave]
        dtuple += [_par - npar]

    tuplet = [mtuple,dtuple]
    valt = [sum(mtuple), sum(abs(d) for d in dtuple)]
    rdnt = [valt.d > valt.m, valt.d < valt.m]

    return tuplet, valt, rdnt

def normalize(self):
    dist = abs(self)
    return self.__class__(self.dy / dist, self.dx / dist)


class CderH(CBase):  # derH is a list of der layers or sub-layers, each = ptuple_tv

    H: list = z([])
    valt: list = z([0,0])
    rdnt: list = z([1,1])
    dect: list = z([0,0])
    ext : list = z([])
    depth: int = 0

    irdn: int = 0
    # flags
    fneg: int = 0
    fagg: int = 0

    @classmethod
    def empty_layer(cls): return list([[0,0,0,0,0,0], [0,0,0,0,0,0]])

    def __add__(Hv, hv):
        if hv.H:
            if Hv.H:
                H, Valt, Rdnt, Dect, Extt, Depth = Hv.H, Hv.valt, Hv.rdnt, Hv.dect, Hv.ext, Hv.depth
                h, valt, rdnt, dect, extt, depth = hv.H, hv.valt, hv.rdnt, Hv.dect, Hv.ext, Hv.depth
                Valt[:] = np.add(Valt,valt)
                Rdnt[:] = np.add( np.add(Rdnt,rdnt), [Hv.irdn,Hv.irdn])
                Rdnt[0] += Valt[1] > Valt[0]
                Rdnt[1] += Valt[0] > Valt[1]
                if Hv.fagg:
                    Dect[:] = np.divide( np.add(Dect,dect), 2)
                fC=0
                if isinstance(H[0], CderH):
                    fC=1
                    if isinstance(h[0], list):  # convert dertv to derH:
                        h = [CderH(H=h, valt=copy(hv.valt), rdnt=copy(hv.rdnt), dect=copy(hv.dect), ext=copy(hv.ext), depth=0)]
                elif isinstance(h[0], CderH):
                    fC=1; H = [CderH(H=H, valt=copy(Hv.valt), rdnt=copy(Hv.rdnt), dect=copy(Hv.dect), ext=copy(Hv.ext), depth=0)]

                if fC:  # both derH_:
                    H[:(len(h))] = [DerH + derH for DerH, derH in zip_longest(H,h)]  # if different length or always same?
                else:  # both dertuplets:
                    H[:] = [list(np.add(Dertuple,dertuple)) for Dertuple, dertuple in zip(H,h)]  # mtuple,dtuple
            else:
                Hv.H[:] = deepcopy(hv.H)

        # return Hv

    def __iadd__(self, other): return self + other
    def __isub__(self, other): return self - other
