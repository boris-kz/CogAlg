'''
            for G in link._G, link.G:
                # draft
                if len_root_HH > -1:  # agg_compress, increase nesting
                    if len(G.rim_t)==len_root_HH:  # empty rim layer, init with link:
                        if fd:
                            G.Vt=[0,Val]; G.Rt=[0,Rdn]; G.Dt=[0,Dec]  # temporary
                            G.rim_t = [[[[[],[link]]]],2]; G.rim_t = [[[[[],[link]]]],2]  # init rim_tHH, depth = 2
                        else:
                            G.Vt=[Val,0]; G.Rt=[Rdn,0]; G.Dt=[Dec,0]
                            G.rim_t = [[[[[link],[]]]],2]; G.rim_t = [[[[[link],[]]]],2]
                    else:
                        G.Vt[fd] += Val; G.Rt[fd] += Rdn; G.Dt[fd] += Dec  # accum rim layer with link
                        G.rim_t[0][-1][-1][fd] += [link]; G.rim_t[0][-1][-1][fd] += [link]  # append rim_tHH
                else:
                    if len(G.rim_t)==len_root_H:  # empty rim layer, init with link:
                        if fd:
                            G.Vt=[0,Val]; G.Rt=[0,Rdn]; G.Dt=[0,Dec]
                            G.rim_t = [[[[],[link]]],1]; G.rim_t = [[[[],[link]]],1]  # init rim_tH, depth = 1
                        else:
                            G.Vt=[Val,0]; G.Rt=[Rdn,0]; G.Dt=[Dec,0]
                            G.rim_t = [[[[link],[]]],1]; G.rim_t = [[[[link],[]]],1]
                    else:
                        G.Vt[fd] += Val; G.Rt[fd] += Rdn; G.Dt[fd] += Dec  # accum rim layer with link
                        G.rim_t[0][-1][fd] += [link]; G.rim_t[0][-1][fd] += [link]  # append rim_tH
'''

def rd_recursion(rroot, root, node_, lenH, lenHH, Et, nrng=1):  # rng,der incr over same G_,link_ -> fork tree, represented in rim_t

    Vt, Rt, Dt = Et
    for fd, Q, V,R,D in zip((0,1),(node_,root.link_), Vt,Rt,Dt):  # recursive rng+,der+

        ave = G_aves[fd]
        if fd and rroot == None:
            continue  # no link_ and der+ in base fork
        if V >= ave * R:  # true for init 0 V,R; nrng if rng+, else 0:
            if not fd:
                nrng += 1
            G_,(vt,rt,dt) = cross_comp(Q, lenH, lenHH, nrng*(1-fd))
            for inp in Q:
                if isinstance(inp,CderG): Gt = [inp._G, inp.G]  # link
                else: Gt = [inp]  # G, packed in list for looping:
                for G in Gt:
                    for i, link in enumerate(link_):  # add esubH layer, temporary?
                        if i:
                            sum_derHv(G.esubH[-1], link.subH[-1], base_rdn=link.Rt[fd])  # [derH, valt,rdnt,dect,extt,1]
                        else:
                            G.esubH += [deepcopy(link.subH[-1])]  # link.subH: cross-der+) same rng, G.esubH: cross-rng?

            # we should sum links' esubH per G instead?
            G_ = defaultdict(list)
            for link in link_:
                for G in (link.G, link._G): G_[G] += [link]  # pack links per G
            # sum links' esubH per G
            for G in G_.keys():
                for i, link in enumerate(G_[G]):
                    if i:
                        sum_derHv(G.esubH[-1], link.subH[-1], base_rdn=link.Rt[fd])  # [derH, valt,rdnt,dect,extt,1]
                    else:
                        G.esubH += [deepcopy(link.subH[-1])]  # link.subH: cross-der+) same rng, G.esubH: cross-rng?

            for i, v,r,d in zip((0,1), vt,rt,dt):
                Vt[i]+=v; Rt[i]+=rt[i]; Dt[i]+=d
                if v >= ave * r:
                    if i: root.link_+= link_  # rng+ links
                    # adds to root Et + rim_t, and Et per G:
                    rd_recursion(rroot, root, node_, lenH, lenHH, [vt,rt,dt], nrng)
    return nrng


def form_graph_t(root, lenH, lenHH, Et, nrng):

    rdepth = (lenH != None) + (lenHH != None)
    G_ = root.node_[not nrng] if isinstance(root.node_[0],list) else root.node_
    _G_ = []
    # not revised below:
    for G in G_:  # select Gs connected in current layer
        if G.rim_t[-1] > rdepth:  # if G is updated in comp_G, their depth should > rdepth
          _G_ += [G]
        else:
            if lenHH:  # check if lenHH is incremented
                if (G.rim_t[0] > lenHH):
                    _G_ += [G]
            else:  # check if lenH is incremented
                if (G.rim_t[0] > lenH):
                    _G_ += [G]

    node_connect(_G_)  # Graph Convolution of Correlations over init _G_
    node_t = []
    for fd in 0,1:
        if Et[0][fd] > ave * Et[1][fd]:  # eValt > ave * eRdnt, else no clustering, keep root.node_
            graph_ = segment_node_(root, _G_, fd, nrng)  # fd: node-mediated Correlation Clustering
            if not graph_: continue
            for graph in graph_:  # eval sub+ per node
                if graph.Vt[fd] * (len(graph.node_)-1)*root.rng > G_aves[fd] * graph.Rt[fd]:
                    # last sub+ val -> sub+:
                    if graph.node_[0].rim_t[-1] == 2:  # test root aggH?
                        lenH = len(graph.node_[0].rim_t[0][-1][0])
                        lenHH = len(graph.node_[0].rim_t[0])
                    else:  # rim_tH
                        lenH = len(graph.node_[0].rim_t[0])
                        lenHH = None
                    agg_recursion(root, graph, graph.node_[-1], lenH, lenHH, nrng+1*(1-fd))  # nrng+ if not fd
                    rroot = graph
                    rfd = rroot.fd
                    while isinstance(rroot.roott, list) and rroot.roott[rfd]:  # not blob
                        rroot = rroot.roott[rfd]
                        Val,Rdn = 0,0
                        if isinstance(rroot.node_[-1][0], list):  # node_ is node_t
                            node_ = rroot.node_[-1][rfd]
                        else: node_ = rroot.node_[-1]

                        for node in node_:  # sum vals and rdns from all higher nodes
                            Rdn += node.rdnt[rfd]
                            Val += node.valt[rfd]
                        # include rroot.Vt and Rt?
                        if Val * (len(rroot.node_[-1])-1)*rroot.rng > G_aves[fd] * Rdn:
                            # not sure about nrg here
                            agg_recursion(root, graph, rroot.node_[-1], len(rroot.aggH[-1][0]), rfd, nrng+1*(1-rfd))  # nrng+ if not fd
                else:
                    root.fback_t[root.fd] += [[graph.aggH, graph.valt, graph.rdnt, graph.dect]]
                    feedback(root,root.fd)  # update root.root..
            node_t += [graph_]  # may be empty
        else: node_t += []
    if any(node_t): G_[:] = node_t


def cross_comp(Q, lenH, lenHH, nrng):

    Et = [[0,0],[0,0],[0,0]]  # grapht link_' eValt, eRdnt, eDect(currently not used)
    fd = not nrng

    if fd:  # der+
        for link in Q:  # inp_= root.link_, reform links
            if (len(link.G.rim_t[0])==lenH  # the link was formed in prior rd+
                and link.Vt[1] > G_aves[1]*link.Rt[1]):  # >rdn incr
                comp_G(link, Et, lenH, lenHH)
    else:   # rng+
        for _G, G in combinations(Q, r=2):  # form new link_ from original node_
            dy = _G.box.cy - G.box.cy; dx = _G.box.cx - G.box.cx
            dist = np.hypot(dy,dx)
            # max distance between node centers, init=2
            if 2*nrng > dist > 2*(nrng-1):  # G,_G are within rng and were not compared in prior rd+
                link = CderG(_G=_G, G=G)
                comp_G(link, Et, lenH, lenHH)

    return Et

from __future__ import annotations

from numbers import Real
from typing import NamedTuple
from math import inf, hypot

from class_cluster import CBase, init_param as z
from frame_blobs import boxT
'''
    Conventions:
    postfix 't' denotes tuple, multiple ts is a nested tuple, 
    postfix 'T' is namedtuple (?)
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    1-3 letter names are normally scalars, except for P and similar classes, 
    capitalized variables are normally summed small-case variables,
    longer names are normally classes
'''


class angleT(NamedTuple):
    dy: Real
    dx: Real

    # operators:
    def __abs__(self) -> angleT: return hypot(self.dy, self.dx)
    def __pos__(self) -> angleT: return self
    def __neg__(self) -> angleT: return angleT(-self.dy, -self.dx)
    def __add__(self, other: angleT) -> angleT: return angleT(self.dy + other.dy, self.dx + other.dx)
    def __sub__(self, other: angleT) -> angleT: return self + (-other)


class ptupleT(NamedTuple):
    I: Real
    G: Real
    M: Real
    Ma: Real
    angle: angleT
    L: Real

    # operators:
    def __pos__(self) -> ptupleT: return self
    def __neg__(self) -> ptupleT: return ptupleT(-self.I, -self.G, -self.M, -self.Ma, -self.angle, -self.L)

    def __sub__(self, other: ptupleT) -> ptupleT: return self + (-other)

    def __add__(self, other: ptupleT) -> ptupleT:
        return ptupleT(self.I+other.I, self.G+other.G, self.M+other.M, self.Ma+other.Ma, self.angle+other.angle, self.L+other.L)

class CEdge(CBase):  # edge blob

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


class CP(CBase):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

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

class CderP(CBase):  # tuple of derivatives in P link: binary tree with latuple root and vertuple forks

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


class Cgraph(CBase):  # params of single-fork node_ cluster per pplayers

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
    rim_t : list = z([[],0])  # direct links,depth, init rim_t, link_tH in base sub+ | cpr rd+, link_tHH in cpr sub+
    Rim_t : list = z([])  # links to the most mediated evaluated nodes
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
    Vt : list = z([0,0])  # last layer | last fork tree vals for node_connect and clustering
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


class CderG(CBase):  # params of single-fork node_ cluster per pplayers

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