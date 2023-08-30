from class_cluster import ClusterStructure, init_param as z

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

    I: float = 0.0
    Dy: float = 0.0
    Dx: float = 0.0
    G: float = 0.0
    A: float = 0.0  # blob area
    M: float = 0.0  # summed PP.M, for both types of recursion?
    # composite params:
    box: tuple = (0, 0, 0, 0)  # y0, yn, x0, xn
    mask__ : object = None
    der__t : object = None
    der__t_roots: object = None  # map to dir__t
    adj_blobs: list = z([])  # adjacent blobs
    node_ : list = z([])  # default P_, node_tt: list = z([[[],[]],[[],[]]]) in select PP_ or G_ forks
    root_ : object= None  # list root_ if fork overlap?
    derH : list = z([])  # formed in PPs, inherited in graphs
    aggH : list = z([[]])  # [[subH, valt, rdnt]]: cross-fork composition layers
    valt : list = z([0,0])
    rdnt : list = z([1,1])
    fback_ : list = z([])  # [feedback aggH,valt,rdnt per node]


class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

    ptuple : tuple = (0,0,0,0,(0,0),0)  # latuple: I,G,M,Ma, angle(Dy,Dx), L
    derH : list = z([])  # [[tuplet,valt,rdnt]] vertical derivatives summed from P links
    valt : list = z([0,0])  # summed from the whole derH
    rdnt : list = z([1,1])
    dert_ : list = z([])  # array of pixel-level derts, ~ node_
    link_H : list = z([[]])  # +ve rlink_, dlink_ H from lower sub+
    root_tt : list = z([[None,None],[None,None]])  # PPrm,PPrd, PPdm,PPdd that contain this P, single-layer
    olp_P_ : list = z([])  # overlaping Ps
    axis : tuple = (0, 1)  # prior slice angle, init sin=0,cos=1
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

    fd : int = 0  # PP is defined by combined-fork value per link: derP mtuple | dtuple; not fder?
    ptuple : list = z([0,0,0,0,[0,0],0])   # summed P__ ptuples, = 0th derLay
    derH : list = z([])  # [[mtuple,dtuple, mval,dval, mrdn,drdn]]: cross-fork composition layers
    valt : list = z([0,0])
    rdnt : list = z([1,1])
    mask__ : object = None
    node_tt : list = z([])  # P_,-> node_tt if sub+: [rng+ [sub_PPm_,sub_PPd_], der+ [sub_PPm_,sub_PPd_]]
    root_tt : list = z([[None,None],[None,None]])  # init edge, higher PPrm,PPrd, PPdm,PPdd that contain PP if sub+
    rng : int = 1  # sum of rng+: odd forks in last layer?
    box : list = z([0,0,0,0])  # y0,yn,x0,xn
    # temporary:
    fback_ : list = z([])  # [feedback derH,valt,rdnt per node]
    coPP_ : list = z([])  # rdn reps in other PPPs, to eval and remove?
    Rdn : int = 0  # for accumulation or separate recursion count?
    # fdiv = NoneType  # if div_comp?

class Cgraph(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    fd: int = 0  # not fder
    ptuple : list = z([])  # default from P
    derH : list = z([[], [0,0], [1,1]])  # [[tuplet, valt, rdnt]]: default from PP, for both rng+ and der+, sum min len?
    aggH : list = z([])  # [[sub_Ht, valt, rdnt]], subH: [[der_Ht, valt, rdnt]]; cross-fork composition layers
    link_H : list = z([[]])  # added per rng+ comp_G_
    val_Ht : list = z([[0],[0]])  # H of link vals per fder
    rdn_Ht : list = z([[1],[1]])
    node_tt : list = z([])  #-> Gmr_,Gdr_, Gmd_,Gdd_ by agg+/sub+, nested if wH: down-forking tree of node Levs
    root_tt : list = z([[],[]],[[],[]])  # up reciprocal to node_tt, nested if uH: up-forking tree of root Levs
    L : int = 0 # len base node_; from internal links:
    S : float = 0.0  # sparsity: average distance to link centers
    A : list = z([0,0])  # angle: average dy,dx to link centers
    rng : int = 1
    box : list = z([0,0,0,0,0,0])  # y,x, y0,yn, x0,xn
    nval : int = 0  # of open links: base alt rep
    alt_graph_: list = z([])  # adjacent gap+overlap graphs, vs. contour in frame_graphs
    alt_Graph : object = None  # conditional, summed and concatenated params of alt_graph_
    # temporary:
    compared_ : list = z([])
    fback_ : list = z([])  # [feedback aggH,valt,rdnt per node]
    Rdn : int = 0  # for accumulation or separate recursion count?

    # id_H : list = z([[]])  # indices in the list of all possible layers | forks, not used with fback merging
    # top aggLay: derH from links, lower aggH from nodes, only top Lay in derG:
    # top Lay from links, lower Lays from nodes, hence nested tuple?

class CderG(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    subH : list = z([])  # [[derH_t, valt, rdnt]]: top aggLev derived in comp_G
    valt : list = z([0,0,0])  # m,d,max
    rdnt : list = z([1,1])
    G0 : object = None  # comparand
    G1 : object = None
    S : float = 0.0  # sparsity: average distance to link centers
    A : list = z([0,0])  # angle: average dy,dx to link centers