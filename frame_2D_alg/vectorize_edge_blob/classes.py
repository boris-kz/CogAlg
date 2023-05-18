from frame_2D_alg.class_cluster import ClusterStructure, NoneType

class CQ(ClusterStructure):  # generic sequence or hierarchy

    Q = list  # generic sequence or index increments in ptuple, derH, etc
    Qm = list  # in-graph only
    Qd = list
    ext = lambda: [[],[]]  # [ms,ds], per subH only
    valt = lambda: [0,0]  # in-graph vals
    rdnt = lambda: [1,1]  # none if represented m and d?
    out_valt = lambda: [0,0]  # of non-graph links, as alt?
    fds = list
    rng = lambda: 1  # is it used anywhere?
    n = lambda: 1  # accum count, not needed?

class Cptuple(ClusterStructure):  # bottom-layer tuple of compared params in P, derH per par in derP, or PP

    I = int  # [m,d] in higher layers:
    M = int
    Ma = float
    axis = lambda: [1, 0]  # ini dy=1,dx=0, old angle after rotation
    angle = lambda: [0, 0]  # in latuple only, replaced by float in vertuple
    aangle = lambda: [0, 0, 0, 0]
    G = float  # for comparison, not summation:
    Ga = float
    x = int  # median: x0+L/2
    L = int  # len dert_ in P, area in PP
    n = lambda: 1  # not needed?


class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

    ptuple = Cptuple  # latuple: I, M, Ma, G, Ga, angle(Dy, Dx), aangle( Sin_da0, Cos_da0, Sin_da1, Cos_da1), ?[n, val, x, L, A]?
    derH = list  # 1vertuple / 1layer in comp_slice, extend in der+
    fds = list  # per derLay
    valt = lambda: [0,0]  # of fork links, represented in derH
    rdnt = lambda: [1,1]
    # n = lambda: 1
    x0 = int
    y0 = int  # for vertical gap in PP.P__
    dert_ = list  # array of pixel-level derts, redundant to uplink_, only per blob?
    link_ = list  # all links
    link_t = lambda: [[],[]]  # +ve rlink_, dlink_
    roott = lambda: [None,None]  # m,d PP that contain this P
    dxdert_ = list  # only in Pd
    Pd_ = list  # only in Pm
    # if comp_dx:
    Mdx = int
    Ddx = int

class CderP(ClusterStructure):  # tuple of derivatives in P link: binary tree with latuple root and vertuple forks

    derH = list  # vertuple_ per layer, unless implicit? sum links / rng+, layers / der+?
    fds = list
    valt = lambda: [0,0]  # also of derH
    rdnt = lambda: [1,1]  # mrdn + uprdn if branch overlap?
    _P = object  # higher comparand
    P = object  # lower comparand
    roott = lambda: [None,None]  # for der++
    x0 = int
    y0 = int
    L = int
    fdx = NoneType  # if comp_dx
'''
max ntuples / der layer = ntuples in lower layers: 1, 1, 2, 4, 8...
lay1: par     # derH per param in vertuple, each layer is derivatives of all lower layers:
lay2: [m,d]   # implicit nesting, brackets for clarity:
lay3: [[m,d], [md,dd]]: 2 sLays,
lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays:
'''

class CPP(CderP):

    ptuple = Cptuple  # summed P__ ptuples, = 0th derLay
    derH = list  # 1vert)1lay in comp_slice, extend in der+, extend to rngH in feedback, no comp rngH till agg+
    fds = list  # fd per derLay
    valt = lambda: [0,0]  # link Vals
    rdnt = lambda: [1,1]  # link Rdns
    Rdn = int  # for accumulation or separate recursion count?
    rng = lambda: 1
    box = lambda: [0,0,0,0]  # y0,yn, x0,xn
    fterm = int  # sub-comp was terminated
    fdiv = NoneType  # if div_comp?
    mask__ = bool
    P__ = list  # 2D array of nodes: Ps or sub-PPs
    link_ = list  # all links summed from Ps
    link_t = lambda: [[],[]]  # +ve rlink_, dlink_
    root = NoneType  # PPPm|PPPd containing this PP
    cPP_ = list  # rdn reps in other PPPs, to eval and remove?


class Cgraph(ClusterStructure):  # params of single-fork node_ cluster per pplayers
    ''' ext / agg.sub.derH:
    L = list  # der L, init None
    S = int  # sparsity: ave len link
    A = list  # area|axis: Dy,Dx, ini None
    '''
    G = NoneType  # same-scope lower-der|rng G.G.G., or [G0,G1] in derG, None in PP
    root = NoneType  # root graph or derH G, element of ex.H[-1][fd]
    pH = list  # aggH( subH( derH H: Lev+= node tree slice/fb, Lev/agg+, lev/sub+?  subH if derG
    H = list  # replace with node_ per pH[i]? down-forking tree of Levs: slice of nodes
    # uH: up-forking Levs if mult roots
    node_ = list  # single-fork, conceptually H[0], concat sub-node_s in ex.H levs
    link_ = CQ  # temporary holder for der+ node_, then unique links within graph?
    valt = lambda: [0,0]
    rdnt = lambda: [1,1]
    fterm = int  # node_ sub-comp was terminated
    rng = lambda: 1
    box = lambda: [0,0,0,0,0,0]  # y,x, y0,yn, x0,xn
    nval = int  # of open links: base alt rep
    alt_graph_ = list  # contour + overlapping contrast graphs
    alt_Graph = NoneType  # conditional, summed and concatenated params of alt_graph_