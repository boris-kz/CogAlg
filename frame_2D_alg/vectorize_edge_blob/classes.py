from math import inf
from class_cluster import ClusterStructure, init_param as z
from frame_blobs import boxT

'''
    Conventions:
    postfix 't' denotes tuple, multiple ts is a nested tuple, 'T' for indefinite nesting
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    1-3 letter names are normally scalars, except for P and similar classes, 
    capitalized variables are normally summed small-case variables,
    longer names are normally classes
'''

class CEdge(ClusterStructure):  # edge blob

    der__t_roots: object = None  # map to dir__t
    node_t : list = z([])  # default P_, node_t in select PP_ or G_ fder forks
    fback_t : list = z([[],[]])  # for consistency, only fder=0 is used
    # for comp_slice:
    derH : list = z([])  # formed in PPs, inherited in graphs
    valt : list = z([0,0])
    rdnt : list = z([1,1])
    # for agg+:
    aggH : list = z([[]])  # formed in Gs: [[subH, valt, rdnt]]: cross-fork composition layers
    valHt : list = z([[0],[0]])  # Ht of link vals,rdns, decays per fder
    decHt : list = z([[0],[0]])
    rdnHt : list = z([[1],[1]])
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

    ptuple : tuple = (0,0,0,0,(0,0),0)  # latuple: I,G,M,Ma, angle(Dy,Dx), L
    derH : list = z([])  # [[tuplet,valt,rdnt]] vertical derivatives summed from P links
    valt : list = z([0,0])  # summed from the whole derH
    rdnt : list = z([1,1])
    dert_ : list = z([])  # array of pixel-level derts, ~ node_
    cells : set = z(set())  # pixel-level kernels adjacent to P axis, combined into corresponding derts projected on P axis.
    link_H : list = z([[]])  # +ve rlink_, dlink_ H from lower sub+
    root_t : list = z([None,None])  # PPrm,PPrd, PPdm,PPdd that contain this P, single-layer
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

    derH : list = z([])  # [[[mtuple,dtuple],[mval,dval],[mrdn,drdn]]], single in rng+
    valt : list = z([0,0])
    rdnt : list = z([1,1])  # mrdn + uprdn if branch overlap?
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
lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays:
'''
class CPP(CderP):

    fd : int = 0  # PP is defined by m|d value
    ptuple : list = z([0,0,0,0,[0,0],0])   # summed P__ ptuples, = 0th derLay
    derH : list = z([])  # [[mtuple,dtuple, mval,dval, mrdn,drdn]]: cross-fork composition layers
    valt : list = z([0,0])
    rdnt : list = z([1,1])
    mask__ : object = None
    root : object = None  # edge / PP, PP / sub_PP: not shared between root forks
    node_t : list = z([])  # init P_, -> node_tt if sub+: [rng+ [sub_PPm_,sub_PPd_], der+ [sub_PPm_,sub_PPd_]]
    fback_t : list = z([[],[]])  # maps to node_tt: [derH,valt,rdnt] per node fork
    rng : int = 1  # sum of rng+: odd forks in last layer?
    box : boxT = boxT(inf,inf,-inf,-inf)  # y0,x0,yn,xn
    # temporary:
    coPP_ : list = z([])  # rdn reps in other PPPs, to eval and remove?
    Rdn : int = 0  # for accumulation or separate recursion count?
    # fdiv = NoneType  # if div_comp?

class Cgraph(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    fd: int = 0  # graph is defined by m|d value
    ptuple : list = z([0,0,0,0,[0,0],0])  # default from P
    derH : list = z([[], [0,0], [1,1]])  # default from PP: [[tuplet,valt,rdnt]] from rng+| der+, sum min len?
    aggH : list = z([])  # [[sub_Ht, valt, rdnt]], subH: [[der_Ht, valt, rdnt]]; cross-fork composition layers
    valHt : list = z([[0],[0]])  # Ht of link vals,rdns, decays / fder:
    decHt : list = z([[0],[0]])
    rdnHt : list = z([[1],[1]])
    link_H : list = z([[]])  # added per rng+ comp_G_
    root : object = None  # ini graph, replace with mroot,droot for nodes in sub+, nest in up-forking tree: root_ fork / agg+
    node_t : list = z([])  # init G_-> Gm_,Gd_, nested in down-forking tree: node_ fork/ sub+
    fback_t : list = z([[],[]])  # maps to node_t: feedback [[aggH,valt,rdnt]] per node fork
    L : int = 0 # len base node_; from internal links:
    S : float = 0.0  # sparsity: average distance to link centers
    A : list = z([0,0])  # angle: average dy,dx to link centers
    rng : int = 1
    box : list = z([0,0,0,0,0,0])  # y,x, y0,yn, x0,xn
    # tentative:
    nval : int = 0  # of open links: base alt rep
    alt_graph_ : list = z([])  # adjacent gap+overlap graphs, vs. contour in frame_graphs
    alt_Graph : object = None  # conditional, summed and concatenated params of alt_graph_
    # temporary:
    it : list = z([0,0])  # indices in root.node_t, maybe nested?
    compared_ : list = z([])
    Rdn : int = 0  # for accumulation or separate recursion count?
    # id_H : list = z([[]])  # indices in the list of all possible layers | forks, not used with fback merging
    # top aggLay: derH from links, lower aggH from nodes, only top Lay in derG:
    # top Lay from links, lower Lays from nodes, hence nested tuple?

class CderG(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    subH : list = z([])  # [[derH_t, valt, rdnt]]: top aggLev derived in comp_G
    dect : list = z([0,0])  # m/maxm, d/maxd
    valt : list = z([0,0])  # m,d
    rdnt : list = z([1,1])
    _G : object = None  # comparand
    G: object = None  # comparand
    S : float = 0.0  # sparsity: average distance to link centers
    A : list = z([0,0])  # angle: average dy,dx to link centers
    # dir : bool  # direction of comparison if not G0,G1, only needed for comp link?
