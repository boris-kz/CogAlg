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

not needed:
class Cptuple(ClusterStructure):  # bottom-layer tuple of compared params in P, derH per par in derP, or PP

    I : int = 0  # [m,d] in higher layers:
    M : int = 0
    Ma : float = 0.0
    angle : list = z([0,0])  # in latuple only, replaced by float in vertuple
    aangle : list = z([0,0,0,0])
    G : float = 0.0  # for comparison, not summation:
    Ga : float = 0.0
    L : int = 0  # replaces n, still redundant to len dert_ in P, nlinks in PP or graph
'''

class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

    ptuple : list = z([0,0,0,0,0,[0,0],[0,0,0,0],0])  # latuple: I,G,Ga,M,Ma, angle(Dy,Dx), aangle(Dyy,Dyx,Dxy,Dxx), L
    derH : list = z([])  # [[tuplet,valt,rdnt]] vertical derivatives summed from P links
    valt : list = z([0,0])  # summed from the whole derH
    rdnt : list = z([1,1])
    dert_ : list = z([])  # array of pixel-level derts, ~ node_
    link_tH : list = z([[[],[]]])  # +ve rlink_, dlink_ H from lower sub+
    root_tt : list = z([[None,None],[None,None]])  # PPrm,PPrd, PPdm,PPdd that contain this P, single-layer
    dert_yx_ : list = z([])  # mappings to blob der_t
    dert_olp_: list = z(set())
    axis : list = z([0,1])  # prior slice angle, init sin=0,cos=1
    yx : list = z([])
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

    derH : list = z([])  # [[mtuple,dtuple,mval,dval,mrdn,drdn]], single in rng+
    valt : list = z([0,0])
    rdnt : list = z([1,1])  # mrdn + uprdn if branch overlap?
    _P : object = None  # higher comparand
    P : object = None  # lower comparand
    box : list = z([0,0,0,0])  # y0,yn, x0,xn: P.box+_P.box, or center+_center?
    S : float = 0.0  # sparsity: distance between centers
    A : list = z([0,0])  # angle: dy,dx between centers
    # roott : list = z([None, None])  # for der++, if clustering is per link
    # fdx : object = None  # if comp_dx

'''
max n of tuples per der layer = summed n of tuples in all lower layers: 1, 1, 2, 4, 8..:
lay1: par     # derH per param in vertuple, layer is derivatives of all lower layers:
lay2: [m,d]   # implicit nesting, brackets for clarity:
lay3: [[m,d], [md,dd]]: 2 sLays,
lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays:
'''
class CPP(CderP):

    fd : int = 0  # PP is defined by combined-fork value per link: derP mtuple | dtuple; not fder?
    ptuple : list = z([0,0,0,0,0,[0,0],[0,0,0,0],0])  # summed P__ ptuples, = 0th derLay
    derH : list = z([])  # [[mtuple,dtuple, mval,dval, mrdn,drdn]]: cross-fork composition layers
    valt : list = z([0,0])
    rdnt : list = z([1,1])
    mask__: object = None
    node_ : list = z([])  # P_, or node_tt: [rng+'[sub_PPm_,sub_PPd_], der+'[sub_PPm_,sub_PPd_]]
    root_tt : list = z([[None,None],[None,None]])  # higher PPrm,PPrd, PPdm,PPdd that contain this PP, for sub+ only
    rng : int = 1  # sum of rng+: odd forks in last layer?
    box : list = z([0,0,0,0])  # y0,yn,x0,xn
    # temporary:
    fback_ : list = z([])  # [feedback derH,valt,rdnt per node]
    coPP_ : list = z([])  # rdn reps in other PPPs, to eval and remove?
    Rdn : int = 0  # for accumulation or separate recursion count?
    # fdiv = NoneType  # if div_comp?

class Cgraph(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    fd: int = 0  # not fder
    link_tH : list = z([[[],[]]])  # +ve rlink_, dlink_ H, ~ derH layers
    derH : list = z([])  # [[derH, valt, rdnt]]: default input from PP, for both rng+ and der+, sum min len?
    aggH : list = z([])  # [[subH, valt, rdnt]]: cross-fork composition layers
    valt : list = z([0,0])
    rdnt : list = z([1,1])
    node_: list = z([])  # same-fork, incremental nesting if wH: down-forking tree of node Levs, 4 top forks if sub+:
    # node_tt: list = z([[[],[]],[[],[]]])  # rng+'Gm_,Gd_, der+'Gm_,Gd_, may not overlap
    root: object= None  # root_: list = z([])  # agg|sub+ mset forks, incr.nest if uH: up-forking tree of root Levs,
    # root_tt: list = z([[None,None],[None,None]])  # rng+'Gm,Gd, der+'Gm,Gd, if both comp and form are overlapping
    # external params, summed from links:
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
    ''' 
    ext / agg.sub.derH:
    L : list = z([])  # der L, init None
    S : int = 0  # sparsity: ave len link
    A : list = z([])  # area|axis: Dy,Dx, ini None
    
    # id_H : list = z([[]])  # indices in the list of all possible layers | forks, not used with fback merging
    # top aggLay: derH from links, lower aggH from nodes, only top Lay in derG:
    # top Lay from links, lower Lays from nodes, hence nested tuple?
    '''