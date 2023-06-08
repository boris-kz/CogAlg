from class_cluster import ClusterStructure, init_param as z
'''
    Conventions:
    postfix 't' denotes tuple, multiple ts is a nested tuple
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    1-3 letter names are normally scalars, except for P and similar classes, 
    capitalized variables are normally summed small-case variables,
    longer names are normally classes
'''

# not needed?
class CQ(ClusterStructure):  # generic links

    Q : list = z([])  # generic sequence or index increments in ptuple, derH, etc
    Qm : list = z([])  # in-graph only
    Qd : list = z([])
    ext : list = z([[], []])  # [ms,ds], per subH only
    valt : list = z([0,0])  # in-graph vals
    rdnt : list = z([1,1])  # none if represented m and d?
    out_valt : list = z([0,0])  # of non-graph links, as alt?
    fds : list = z([])  # not used?
    rng : int = 1  # not used?

class CH(ClusterStructure):  # generic hierarchy, or that's most of a node?
      pass

class Cptuple(ClusterStructure):  # bottom-layer tuple of compared params in P, derH per par in derP, or PP

    I : int = 0  # [m,d] in higher layers:
    M : int = 0
    Ma : float = 0.0
    angle : list = z([0,0])  # in latuple only, replaced by float in vertuple
    aangle : list = z([0,0,0,0])
    G : float = 0.0  # for comparison, not summation:
    Ga : float = 0.0
    L : int = 0  # replaces n, still redundant to len dert_ in P, nlinks in PP or graph


class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

    ptuple : list = z([])  # latuple: I, M, Ma, G, Ga, angle(Dy, Dx), aangle( Sin_da0, Cos_da0, Sin_da1, Cos_da1)
    derT : list = z([[],[]])  # ptuple) fork) layer) H)T; 1ptuple,1fork,1layer in comp_slice, extend in der+ and fback
    valT : list = z([0,0])
    rdnT : list = z([1,1])
    axis : list = z([0,1])  # prior slice angle, init sin=0,cos=1
    dert_ : list = z([])  # array of pixel-level derts, redundant to uplink_, only per blob?
    dert_ext_: list = z([])  # external params: roots and coords per dert
    link_ : list = z([])  # all links
    link_t : list = z([[],[]])  # +ve rlink_, dlink_
    roott : list = z([None, None])  # mPP,dPP that contain this P
    box : list = z([0,0,0,0])  # y0,yn, x0,xn
    # optional:
    dxdert_ : list = z([])  # only in Pd
    Pd_ : list = z([])  # only in Pm
    Mdx : int = 0  # if comp_dx
    Ddx : int = 0

class CderP(ClusterStructure):  # tuple of derivatives in P link: binary tree with latuple root and vertuple forks

    derT : list = z([])  # vertuple_ per layer, unless implicit? sum links / rng+, layers / der+?
    valT : list = z([0,0])  # also of derH
    rdnT : list = z([1,1])  # mrdn + uprdn if branch overlap?
    _P : object = None  # higher comparand
    P : object = None  # lower comparand
    roott : list = z([None, None])  # for der++
    box : list = z([0,0,0,0])  # y0,yn, x0,xn: P.box+_P.box, or center+_center?
    L : int = 0
    fdx : object = None  # if comp_dx

'''
max n of tuples per der layer = summed n of tuples in all lower layers: 1, 1, 2, 4, 8..:
lay1: par     # derH per param in vertuple, layer is derivatives of all lower layers:
lay2: [m,d]   # implicit nesting, brackets for clarity:
lay3: [[m,d], [md,dd]]: 2 sLays,
lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays:
'''
class CPP(CderP):

    fd : int = 0  # PP is defined by combined-fork value per link: derP mtuple | dtuple
    ptuple : list = z([])  # summed P__ ptuples, = 0th derLay
    derT : list = z([[],[]])  # ptuple) fork) layer) H)T: 1ptuple, 1fork, 1layer in comp_slice, extend in sub+ and fb
    valT : list = z([[],[]])  # nesting parallel to derT
    rdnT : list = z([[],[]])
    mask__ : object = None
    P_ : list = z([])  # array of nodes: Ps or sub-PPs
    link_ : list = z([])  # all links summed from Ps
    link_t: list = z([[],[]])  # +ve rlink_, dlink_
    roott : list = z([None, None])  # PPPm | PPPd containing this PP, for sub+ only
    rng : int = 1  # sum of rng+: odd forks in last layer?
    box : list = z([0,0,0,0])  # y0,yn,x0,xn
    # temporary:
    fback_ : list = z([])  # [feedback derT,valT,rdnT per node]
    coPP_ : list = z([])  # rdn reps in other PPPs, to eval and remove?
    Rdn : int = 0  # for accumulation or separate recursion count?
    # fdiv = NoneType  # if div_comp?

class Cgraph(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    G : object = None  # same-scope lower-der|rng G.G.G., or [G0,G1] in derG, None in PP
    fd: int = 0
    parT : list = z([[],[]])  # aggH( subH( derH H: Lev+= node tree slice/fb, Lev/agg+,lev/sub+? subH if derG
    id_T : list = z([[],[]])  # indices in the list of all possible layers|forks, for sparse representation
    valT : list = z([[],[]])
    rdnT : list = z([[],[]])
    # top md_sets vs md_T: mix of m|d per salient param, sum per set?
    node_: list = z([])  # same-fork: wH[0], variable nesting? concat sub-node_s in ex.H levs
    wH :   list = z([])  # down-forking tree of node Levs, forks in id_T?
    root_: list = z([])  # agg+|sub+ forks: msets?
    uH :   list = z([])  # up-forking tree of root Levs, if multiple roots, separate id_T?
    link_: list = z([])  # temporary holder for der+ node_, then unique links within graph?
    link_t: list = z([[],[]])  # +ve rlink_, dlink_
    rng : int = 1
    box : list = z([0,0,0,0,0,0])  # y,x, y0,yn, x0,xn
    nval : int = 0  # of open links: base alt rep
    alt_graph_ : list = z([])  # adjacent gap+overlap graphs, vs. contour in frame_graphs
    alt_Graph : object = None  # conditional, summed and concatenated params of alt_graph_
    # temporary:
    fback_ : list = z([])  # [feedback derT,valT,rdnT per node]
    Rdn : int = 0  # for accumulation or separate recursion count?
    ''' 
    ext / agg.sub.derH:
    L : list = z([])  # der L, init None
    S : int = 0  # sparsity: ave len link
    A : list = z([])  # area|axis: Dy,Dx, ini None
    '''
