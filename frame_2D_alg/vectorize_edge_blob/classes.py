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

    ptuple = lambda: Cptuple()  # latuple: I, M, Ma, G, Ga, angle(Dy, Dx), aangle( Sin_da0, Cos_da0, Sin_da1, Cos_da1), ?[n, val, x, L, A]?
    derH = lambda: [[[[],[]]]]  # 1vertuple / 1layer in comp_slice, extend in der+
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

    ptuple = lambda: Cptuple()  # summed P__ ptuples, = 0th derLay
    P__ = list  # 2D array of nodes, may be sub-PPs
    derH = lambda: [[[[],[]]]]  # 1vertuple / 1layer in comp_slice, extend in der+
    fds = list  # fd per derLay
    valt = lambda: [0,0]  # summed fork links vals
    rdnt = lambda: [1,1]  # recursion count + Rdn / nderPs + mrdn + uprdn if branch overlap?
    Rdn = int  # for accumulation only?
    rng = lambda: 1
    nval = int  # of links to alt PPs?
    alt_rdn = int  # overlapping redundancy between core and edge
    alt_PP_ = list  # adjacent alt-fork PPs per PP, from P.roott[1] in sum2PP
    altuple = list  # summed from alt_PP_, sub comp support, agg comp suppression?
    box = lambda: [0,0,0,0]  # y0,yn, x0,xn
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    mask__ = bool
    link_ = list  # all links summed from Ps
    link_t = lambda: [[],[]]  # +ve rlink_, dlink_
    roott = lambda: [None,None]  # PPPm, PPPd that contain this PP
    cPP_ = list  # rdn reps in other PPPs, to eval and remove
