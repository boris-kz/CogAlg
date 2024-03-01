import numpy as np
from itertools import count, zip_longest
from math import inf, hypot
from numbers import Real
from typing import Any, NamedTuple, Tuple
from copy import copy, deepcopy
from class_cluster import CBase, CBaseLite
from types import SimpleNamespace

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
    
Convert to all-lists: cluster_params Cpar_:

Cpar_[0] is numeric typ: [[node|link], [nesting_depth]]:
depth is 0 in P | derP, 1 in PP | derPP, 2 in G | derG, blobs can be same as nodes.

Cpar_[1] is node [link_,sub_node_] or link nodet,
Cpar_[1:] is extensible param set, explained in top docstring per module,
Cpar_[-1] is instance id, set in initialization
We already have proto-G grapht in segment_node_, 195, just need to add params in sum2graph.
'''

def add_(HE, He, irdnt=[]):  # unpack tuples (formally lists) down to numericals and sum them

    if He:  # to be summed
        if HE:  # to sum in
            Depth, depth = HE[0], He[0]  # nesting depth, nest to the deeper He: md_-> derH-> subH-> aggH:
            ddepth = abs(Depth - depth)
            if ddepth:
                nHe = [HE,He][Depth > depth]  # nested He
                while ddepth > 0:
                   nHe[:] = [nHe[0]+1, [*nHe[1]], [deepcopy(nHe)]]; ddepth -= 1

            if isinstance(He[2][0], list):
                for Lay,lay in zip_longest(HE[2], He[2], fillvalue=[]):  # always list He
                    if lay:
                        if Lay:
                            if isinstance(Lay[2][0],list):  # and isinstance(lay[2][0],list), and par[0]==Par[0]: same typ, in cpr version?
                                add_(Lay, lay, irdnt)  # unpack and sum Ets and Hs
                            else:
                                Lay[2] = np.add(Lay[2],lay[2])  # both have numericals in H
                            Et, et = Lay[1], lay[1]  # sum lay[1], always numerical
                            Et[:] = [E+e for E,e in zip(Et,et)]
                            if irdnt: Et[2:4] = [E+e for E,e in zip(Et[2:4],irdnt)]
                        else:
                            HE[2] += [deepcopy(lay)]
            else:
                HE[2] = np.add(HE[2], He[2])  # sum flat lists: [m,d,m,d,m,d...]
        else:
            HE[:] = deepcopy(He)  # copy the 1st He to empty HE
        if irdnt:
            HE[1][2] += irdnt[0]; HE[1][3] += irdnt[1]

    return HE  # for summing

def nest(HE,He):

    Depth, depth = HE[0], He[0]  # nesting depth, nest to the deeper He: md_-> derH-> subH-> aggH:
    ddepth = abs(Depth - depth)
    if ddepth:
        nHe = [HE,He][Depth > depth]  # nested He
        while ddepth > 0:
           nHe[:] = [nHe[0]+1, [*nHe[1]], [deepcopy(nHe)]]; ddepth -= 1


def comp_(_He,He, rn=1, fagg=0):  # unpack tuples (formally lists) down to numericals and compare them

    _depth,depth = _He[0], He[0]
    ddepth = abs(_depth - depth)
    n = 0
    if ddepth:  # unpack the deeper He: md_<-derH <-subH <-aggH:
        uHe = [He,_He][_depth>depth]
        while ddepth > 0:
            uHe = uHe[2][0]; ddepth -= 1  # comp 1st layer of deeper He:
        _cHe,cHe = [uHe,He] if _depth>depth else [_He,uHe]
    else: _cHe,cHe = _He,He

    if isinstance(_cHe[2][0], list):  # _lay is He_, same for lay: they are aligned above
        Et = [0,0,0,0,0,0]  # Vm,Vd, Rm,Rd, Dm,Dd
        dH = []
        for _lay,lay in zip(_cHe[2],cHe[2]):  # md_| ext| derH| subH| aggH, eval nesting, unpack,comp ds in shared lower layers:
            dpth, et, dlay, dn = comp_(_lay,lay, rn, fagg)
            Et[:] = [E+e for E,e in zip(Et,et)]
            dH += [[dpth, et, dlay]]; n += dn
    else:  # H is md_, numerical comp:
        vm,vd,rm,rd, decm,decd = 0,0,0,0, 0,0
        dH = []
        for i, (_d,d) in enumerate(zip(_cHe[2][1::2], cHe[2][1::2])):  # compare ds in md_ or ext
            d *= rn  # normalize by accum span
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
        Et = [vm,vd,rm,rd]
        if fagg: Et += [decm, decd]
        n += 1 if len(_cHe[2]) == 12 else 0.5  # md_ += 1, ext += 0.5

    return min(_depth,depth), Et, dH, n


class z(SimpleNamespace):

    def __init__(self, **kwargs):  # this is not working here, their default value is still referencing a same id
        super().__init__(**kwargs)
    def __iadd__(T, t): return T + t
    def __isub__(T, t): return T - t

    def __add__(T, t):
        if T.typ == 'ptuple':
            for param in list(T.__dict__)[1:]:  # [1:] to skip typ
                V = getattr(T, param); v = getattr(t, param)
                if isinstance(V, list): setattr(T, param, [A + a for A, a in zip(V,v)])
                else:                   setattr(T, param, V+v)
        return T


def init_default(instance, params_set, default_value):
    for param, value in zip(params_set, default_value):
        if getattr(instance, param) is None: setattr(instance, param, deepcopy(value))  # deepcopy prevent list has a same reference

# C inits SimpleNamespace instance with typ-specific param set:

def Cptuple(typ='ptuple',I=None, G=None, M=None, Ma=None, angle=None, L=None):

    params_set = ('I', 'G', 'M', 'Ma', 'angle', 'L')
    default_value = (0,0,0,0,[0,0], 0)
    instance = z(typ=typ,I=I, G=G, M=M, Ma=Ma, angle=angle, L=L)
    init_default(instance, params_set, default_value)
    return  instance


def Cedge(typ='edge',root=None, node_=None, aggH=None, box=None, mask__=None, Et=None, et=None, He=None, fback_=None):
    params_set = ('root', 'node_', 'aggH', 'box', 'mask__', 'Et', 'et', 'He', 'fback_')
    default_value = (None,[[],[]],[],[inf,inf,-inf,-inf],None,[], [], [], [])
    instance = z(typ=typ, root=root, node_=node_, aggH=aggH, box=box, mask__=mask__, Et=Et, et=et, He=He, fback_=fback_)
    init_default(instance, params_set, default_value)
    return instance

def CP(typ='P', yx=None, axis=None, cells=None, dert_=None, He=None, link_=None):
    params_set = ('yx', 'axis', 'cells', 'dert_', 'He', 'link_')
    default_value = ([0,0],[0,0],{},[],[], [[]])
    instance = z(typ=typ, yx=yx,axis=axis,cells=cells,dert_=dert_, He=He, link_=link_)
    init_default(instance, params_set, default_value)
    return instance

def CderP(typ='derP', P=None,_P=None, He=None, et=None, S=None, A=None, n=None, roott=None):
    params_set = ('P', '_P', 'He', 'et', 'S', 'A', 'n', 'roott')
    default_value = (None,None,[],[], 0, 0, 0, [[],[]])
    instance = z(typ=typ, P=P, _P=_P, He=He, et=et, S=S, A=A, n=n,  roott=roott)
    init_default(instance, params_set, default_value)
    return instance

def CPP(typ='PP',
        fd = None,  # fork if flat layers?
        ptuple = None,  # default P
        He = None,  # md_| derH
        # graph-internal, generic:
        et = None,  # sum ptuple, derH, aggH
        link_ = None,  # internal, single-fork, incrementally nested
        node_ = None,  # base node_ replaced by node_t in both agg+ and sub+, deeper node-mediated unpacking in agg+
         # graph-external, +level per root sub+:
        ext = None,  # L,S,A: L len base node_, S sparsity: average link len, A angle: average link dy,dx
        rng = None,
        box = None,  # y,x,y0,x0,yn,xn
        # PP:
        P_ = None,
        mask__ = None,
        area = None,
        # temporary, replace with Et:
        Et = None, # last layer | last fork tree vals for node_connect and clustering
        root = None,  # for feedback
        fback_ = None): # feedback [[aggH,valt,rdnt,dect]] per node layer, maps to node_H

    params_set = ('fd','ptuple', 'He',
                  'et', 'link_', 'node_',
                  'ext', 'rng', 'box',
                  'P_','mask__', 'area',
                  'Et','root','fback_')

    default_value = (0,Cptuple(), [],
                     [], [], [],
                     [0,0, [0,0]], 1, [inf,inf,-inf,-inf],
                     [],None,0,
                     [],[None],[])

    instance = z(typ=typ, fd=fd,ptuple=ptuple, He=He,
                 et=et, link_=link_, node_=node_,
                 ext=ext, rng=rng, box=box,
                 P_=P_,mask__=mask__,area=area,
                 Et=Et,root=root,fback_=fback_)

    init_default(instance, params_set, default_value)

    return instance

def Cgraph(typ='graph',
           PP = None,  # for conversion
           fd = None,  # fork if flat layers?
           ptuple = None,  # default P
           He = None,  # from PP, not derHv
           # graph-internal, generic:
           aggH = None,  # [[subH,valt,rdnt,dect]], subH: [[derH,valt,rdnt,dect]]: 2-fork composition layers
           et = None,  # sum ptuple, derH, aggH
           link_ = None,  # internal, single-fork, incrementally nested
           node_ = None,  # base node_ replaced by node_t in both agg+ and sub+, deeper node-mediated unpacking in agg+
            # graph-external, +level per root sub+:
           rimH = None,  # direct links, depth, init rim_t, link_tH in base sub+ | cpr rd+, link_tHH in cpr sub+
           RimH = None,  # links to the most mediated nodes
           ext_He = None, # G-external He: daggH( dsubH( dderH, summed from rim links
           eet = None,  # sum from esubH
           ext = None,  # L,S,A: L len base node_, S sparsity: average link len, A angle: average link dy,dx
           rng = None,
           box = None,  # y,x,y0,x0,yn,xn
           # tentative:
           alt_graph_ = None,  # adjacent gap+overlap graphs, vs. contour in frame_graphs
           aet = None,  # sum from alt graphs to complement G aves?
           # PP:
           P_ = None,
           mask__ = None,
           area = None,
           # temporary, replace with Et:
           Et = None, # last layer | last fork tree vals for node_connect and clustering
           root = None,  # for feedback
           fback_ = None, # feedback [[aggH,valt,rdnt,dect]] per node layer, maps to node_H
           compared_ = None,
           Rdn = None, # for accumulation or separate recursion count?

           it = None,  # graph indices in root node_s, implicitly nested
           depth = None,  # n sub_G levels over base node_, max across forks
           nval = None, # of open links: base alt rep
           id_H = None):  # indices in the list of all possible layers | forks, not used with fback merging

    params_set = ('fd','ptuple', 'He',
                  'aggH', 'et', 'link_', 'node_',
                  'rimH', 'RimH', 'ext_He', 'eet', 'ext', 'rng', 'box',
                  'alt_graph_','aet',
                  'P_','mask__','area',
                  'Et','root','fback_','compared_','Rdn',
                  'it','depth','nval','id_H')

    default_value = (0,Cptuple(), [],
                     [], [], [], [],
                     [], [], [], [], [0,0, [0,0]], 1, [inf,inf,-inf,-inf],
                     [],[],
                     [],None,0,
                     [],[None],[],[],0,
                     [None, None],0,0,[[]])

    instance = z(typ=typ, fd=fd,ptuple=ptuple, He=He,
                 aggH=aggH, et=et, link_=link_, node_=node_,
                 rimH=rimH, RimH=RimH, ext_He=ext_He, eet=eet, ext=ext, rng=rng, box=box,
                 alt_graph_=alt_graph_,aet=aet,
                 P_=P_,mask__=mask__,area=area,
                 Et=Et,root=root,fback_=fback_,compared_=compared_,Rdn=Rdn,
                 it=it,depth=depth,nval=nval,id_H=id_H)

    init_default(instance, params_set, default_value)

    # convert PP to graph
    if PP is not None:
        for param_name in list(PP.__dict__)[1:-1]:  # [1:] to skip type, [:-1] to skip fback
            if param_name in params_set:
                setattr(instance, param_name, getattr(PP, param_name))

        # add Decay?
        if instance.et: instance.et += [0,0]
        if instance.Et: instance.Et += [0,0]

    return instance


def CderG(typ='derG', _G=None, G=None, He=None, et=None, S=None, A=None, n=None, roott=None):

    params_set = ('_G','G','He','et','S', 'A', 'n', 'roott')
    default_value = (None,None,[],[],0, [0,0], 0, [None, None])
    instance = z(typ=typ, _G=_G, G=G, He=He, et=et, S=S, A=A, n=n, roott=roott)
    init_default(instance, params_set, default_value)
    return instance

# separate classes:
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
 
    # dxdert_: list = z([])  # only in Pd
    # Pd_: list = z([])  # only in Pm
    # Mdx: int = 0  # if comp_dx
    # Ddx: int = 0
   

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
            
#  len layers with ext: 2, 3, 6, 12, 24... 
#  max n of tuples per der layer = summed n of tuples in all lower layers: 1, 1, 2, 4, 8..:
#  lay1: par     # derH per param in vertuple, layer is derivatives of all lower layers:
#  lay2: [m,d]   # implicit nesting, brackets for clarity:
#  lay3: [[m,d], [md,dd]]: 2 sLays,
#  lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays

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
'''

def get_match(_par, par):
    match = min(abs(_par),abs(par))
    return -match if (_par<0) != (par<0) else match    # match = neg min if opposite-sign comparands


def negate(He):
    if isinstance(He[2][0], list):
        for lay in He[2]:
            negate(lay)
    else:  # md_
        He[2][1::2] = [-d for d in He[2][1::2]]


def comp_angle(_A, A):  # rn doesn't matter for angles

    _angle, angle = [np.arctan2(Dy, Dx) for Dy, Dx in [_A, A]]

    dangle = _angle - angle  # difference between angles
    if dangle > np.pi: dangle -= 2 * np.pi  # rotate full-circle clockwise
    elif dangle < -np.pi: dangle += 2 * np.pi  # rotate full-circle counter-clockwise
    mangle = (np.cos(dangle) + 1) / 2  # angle similarity, scale to [0,1]
    dangle /= 2 * np.pi  # scale to the range of mangle, signed: [-.5,.5]

    return [mangle, dangle]