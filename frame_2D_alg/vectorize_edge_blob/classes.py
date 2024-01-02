from math import inf, hypot
from class_cluster import ClusterStructure, init_param as z
from frame_blobs import boxT
from collections import namedtuple
'''
    Conventions:
    postfix 't' denotes tuple, multiple ts is a nested tuple, 
    postfix 'T' is namedtuple
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    1-3 letter names are normally scalars, except for P and similar classes, 
    capitalized variables are normally summed small-case variables,
    longer names are normally classes
'''
ptupleT = namedtuple("ptupleT", "I G M Ma angle L")
ptupleT.__pos__ = lambda t: t
ptupleT.__neg__ = lambda t: ptupleT(-t.I, -t.G, -t.M, -t.Ma, -t.angle, -t.L)
ptupleT.__add__ = lambda _t, t: ptupleT(_t.I+t.I, _t.G+t.G, _t.M+t.M, _t.Ma+t.Ma, _t.angle+t.angle, _t.L+t.L)  # typo here
ptupleT.__sub__ = lambda _t, t: _t+(-t)

angleT = namedtuple('angleT', 'dy dx')
angleT.__abs__ = lambda a: hypot(a.dy, a.dx)
angleT.__pos__ = lambda a: a
angleT.__neg__ = lambda a: angleT(-a.dy,-a.dx)
angleT.__add__ = lambda _a,a: angleT(_a.dy+a.dy, _a.dx+a.dx)
angleT.__sub__ = lambda _a,a: _a+(-a)

class CEdge(ClusterStructure):  # edge blob

    fd : int = 0
    der__t_roots: object = None  # map to dir__t
    P_: list = z([])
    node_t : list = z(([],[]))  # default PP_t in select PP_ or G_ fder forks
    link_ : list = z([])
    fback_t : list = z([[],[]])  # for consistency, only fder=0 is used
    # for comp_slice:
    derH : list = z([])  # formed in PPs, inherited in graphs
    valt : list = z([0,0])
    rdnt : list = z([1,1])
    dect : list = z([0,0])  # consistent with graph
    # for agg+:
    aggH : list = z([])  # from Gs: [[subH,valt,rdnt]]: cross-fork composition layers
    rng = 1
    # initializing blob:
    blob : object = None
    M: float = 0.0  # summed PP.M, for both types of recursion?
    Ma: float = 0.0  # summed PP.Ma
    # access blob params as edge.blob.I:I: float = 0.0, etc.
    # Dy: float = 0.0
    # Dx: float = 0.0
    # G: float = 0.0
    # A: float = 0.0  # blob area
    # box: tuple = (0, 0, 0, 0)  # y0, yn, x0, xn
    # mask__ : object = None
    # der__t : object = None
    # adj_blobs: list = z([])  # adjacent blobs


class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

    ptuple : ptupleT = ptupleT(0,0,0,0,angleT(0,0),0)  # latuple: I,G,M,Ma, angle(Dy,Dx), L
    rnpar_H : list = z([])
    derH : list = z([])  # [[tuplet,valt,rdnt]] vertical derivatives summed from P links
    valt : list = z([0,0])  # summed from the whole derH
    rdnt : list = z([1,1])
    dert_ : list = z([])  # array of pixel-level derts, ~ node_
    cells : set = z(set())  # pixel-level kernels adjacent to P axis, combined into corresponding derts projected on P axis.
    roott : list = z([None,None])  # PPrm,PPrd that contain this P, single-layer
    axis : tuple = (0,1)  # prior slice angle, init sin=0,cos=1
    yx : tuple = None
    ''' 
    add L,S,A from links?
    optional:
    link_H : list = z([[]])  # all links per comp layer, rng+ or der+
    dxdert_ : list = z([])  # only in Pd
    Pd_ : list = z([])  # only in Pm
    Mdx : int = 0  # if comp_dx
    Ddx : int = 0
    '''

class CderP(ClusterStructure):  # tuple of derivatives in P link: binary tree with latuple root and vertuple forks

    derH : list = z([])  # [[[mtuple,dtuple],[mval,dval],[mrdn,drdn]]], single ptuplet in rng+
    valt : list = z([0,0])  # replace with Vt?
    rdnt : list = z([1,1])  # mrdn + uprdn if branch overlap?
    roott : list = z([None, None])  # PPdm,PPdd that contain this derP
    _P : object = None  # higher comparand
    P : object = None  # lower comparand
    S : float = 0.0  # sparsity: distance between centers
    A : list = z([0,0])  # angle: dy,dx between centers
    # roott : list = z([None, None])  # for der++, if clustering is per link
'''
max n of tuples per der layer = summed n of tuples in all lower layers: 1, 1, 2, 4, 8..:
lay1: par     # derH per param in vertuple, layer is derivatives of all lower layers:
lay2: [m,d]   # implicit nesting, brackets for clarity:
lay3: [[m,d], [md,dd]]: 2 sLays,
lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays
'''
class CPP(CderP):

    fd : int = 0  # PP is defined by m|d value
    ptuple : ptupleT = ptupleT(0,0,0,0,angleT(0,0),0)   # summed P__ ptuples, = 0th derLay
    derH : list = z([])  # [[mtuple,dtuple, mval,dval, mrdn,drdn]]: cross-fork composition layers
    valt : list = z([0,0])
    rdnt : list = z([1,1])
    mask__ : object = None
    root : list = z([None, None])  # edge / PP, PP / sub_PP: not shared between root forks
    P_: list = z([])
    node_t : list = z(([],[]))  # default PP_t in select PP_ or G_ fder forks
    link_ : list = z([])
    fback_t : list = z([[],[]])  # maps to node_tt: [derH,valt,rdnt] per node fork
    rng : int = 1  # sum of rng+: odd forks in last layer?
    box : boxT = boxT(inf,inf,-inf,-inf)  # y0,x0,yn,xn
    # temporary:
    coPP_ : list = z([])  # rdn reps in other PPPs, to eval and remove?
    Rdn : int = 0  # for accumulation or separate recursion count?
    # fdiv = NoneType  # if div_comp?


class Cgraph(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    fd: int = 0  # fork if flat layers?
    ptuple : ptupleT = ptupleT(0,0,0,0,angleT(0,0),0)  # default P
    derH : list = z([])  # from PP, not derHv
    # graph-internal, generic:
    aggH : list = z([])  # [[subH,valt,rdnt,dect]], subH: [[derH,valt,rdnt,dect]]: 2-fork composition layers
    valt : list = z([0,0])  # sum ptuple, derH, aggH
    rdnt : list = z([1,1])
    dect : list = z([0,0])
    link_ : list = z([])  # internal, single-fork
    node_ : list = z([])  # base node_ replaced by node_t in both agg+ and sub+, deeper node-mediated unpacking in agg+
    # graph-external, +level per root sub+:
    rim_t : list = z([])  # directly connected nodes, per fork ) layer (this should be init empty, new layer will be added in comp_G)
    Rim_t : list = z([])  # the most mediated evaluated nodes
    esubH : list = z([])  # external subH: [[daggH,valt,rdnt,dect]] of all der)rng rim links
    evalt : list = z([0,0])  # sum from esubH
    erdnt : list = z([1,1])
    edect : list = z([0,0])
    # ext params:
    L : int = 0 # len base node_; from internal links:
    S : float = 0.0  # sparsity: average distance to link centers
    A : list = z([0,0])  # angle: average dy,dx to link centers
    rng : int = 1
    box : boxT = boxT(inf,inf,-inf,-inf)  # y0,x0,yn,,xn
    # tentative:
    alt_graph_ : list = z([])  # adjacent gap+overlap graphs, vs. contour in frame_graphs
    avalt : list = z([0,0])  # sum from alt graphs to complement G aves?
    ardnt : list = z([1,1])
    adect : list = z([0,0])
    # PP:
    P_: list = z([])
    mask__ : object = None
    # temporary:
    Vt : list = z([0,0])  # last layer vals for node_connect and clustering
    Rt : list = z([1,1])
    Dt : list = z([0,0])
    it : list = z([None,None])  # graph indices in root node_s, implicitly nested
    roott : list = z([None,None])  # for feedback
    fback_t : list = z([[],[],[]])  # feedback [[aggH,valt,rdnt,dect]] per node fork, maps to node_H
    compared_ : list = z([])
    Rdn : int = 0  # for accumulation or separate recursion count?
    # not used:
    depth : int = 0  # n sub_G levels over base node_, max across forks
    nval : int = 0  # of open links: base alt rep
    # id_H : list = z([[]])  # indices in the list of all possible layers | forks, not used with fback merging
    # top aggLay: derH from links, lower aggH from nodes, only top Lay in derG:
    # top Lay from links, lower Lays from nodes, hence nested tuple?


class CderG(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    subH : list = z([])  # [[derH_t, valt, rdnt]]: top aggLev derived in comp_G, per rng, all der+
    Vt : list = z([0,0])  # last layer vals from comp_G
    Rt : list = z([1,1])
    Dt : list = z([0,0] )
    _G : object = None  # comparand + connec params
    G : object = None
    S : float = 0.0  # sparsity: average distance to link centers
    A : list = z([0,0])  # angle: average dy,dx to link centers
    roott : list = z([None,None])
    # dir : bool  # direction of comparison if not G0,G1, only needed for comp link?