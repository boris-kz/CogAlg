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


def sum_subHv(T, t, base_rdn, fneg=0):

    SubH,Valt,Rdnt,Dect,_ = T; subH,valt,rdnt,dect,_ = t
    for i in 0,1:
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i]+dect[i])/2
    if SubH:
        for Layer, layer in zip_longest(SubH,subH, fillvalue=[]):
            if layer:
                if Layer:
                    sum_derHv(Layer, layer, base_rdn, fneg)
                else:
                    SubH += [deepcopy(layer)]  # _lay[0][0] is mL
    else:
        SubH[:] = deepcopy(subH)


def sum_aggHv(AggH, aggH, base_rdn):

    if aggH:
        if AggH:
            for Layer, layer in zip_longest(AggH,aggH, fillvalue=[]):
                if layer:
                    if Layer:
                        sum_subHv(Layer, layer, base_rdn)
                    else:
                        AggH += [deepcopy(layer)]
        else:
            AggH[:] = deepcopy(aggH)

def sum_derHv(T,t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    DerH, Valt, Rdnt, Dect, Extt_,_ = T
    derH, valt, rdnt, dect, extt_,_ = t
    for Extt, extt in zip(Extt_,extt_):
        sum_ext(Extt, extt)
    for i in 0,1:
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i] + dect[i])/2
    DerH[:] = [
        [[sum_dertuple(Dertuple,dertuple, fneg*i) for i,(Dertuple,dertuple) in enumerate(zip(Tuplet,tuplet))],
          [V+v for V,v in zip(Valt,valt)], [R+r+base_rdn for R,r in zip(Rdnt,rdnt)], [(D+d)/2 for D,d in zip(Dect,dect)], 0
        ]
        for [Tuplet,Valt,Rdnt,Dect,_], [tuplet,valt,rdnt,dect,_]  # ptuple_tv
        in zip_longest(DerH, derH, fillvalue=[([0,0,0,0,0,0],[0,0,0,0,0,0]), (0,0),(0,0),(0,0),0])
    ]

def unpack_rim(rim_t, fd):  # if init rim_t is [[[],[]],0]

    if rim_t[1] == 2:  # depth=2 is rim_tH in agg++(agg_cpr)
        rim_t = rim_t[0][-1]  # last rim_t
    if rim_t[1] == 1:
        if isinstance(rim_t[0][-1],list):  # rim_t[0] is [rim_t2, rim_t2, ...], so we still need [-1] to select the last rim_t
              rim = rim_t[0][-1][fd][-1]  # rim_t in agg++
        else: rim = rim_t[0][-1][fd]  # rimtH in agg+
    else:
        rim = rim_t[fd]  # base rimt

    return rim


def agg_recursion(rroot, root, node_, nrng=1, lenH=0, lenHH=None):  # lenH = len(root.aggH[-1][0]), lenHH: same in agg_compress

    Et = [[0,0],[0,0],[0,0]]  # grapht link_' eValt, eRdnt, eDect(currently not used)
    fd = not nrng  # compositional agg|sub recursion in root graph:

    if fd:  # der+
        for link in root.link_:  # reform links
            if link.Vt[1] > G_aves[1]*link.Rt[1]:  # maybe weak after rdn incr?
                comp_G(link, Et, lenH, lenHH)
    else:   # rng+
        for _G, G in combinations(node_, r=2):  # form new link_ from original node_
            dy = _G.box.cy - G.box.cy; dx = _G.box.cx - G.box.cx
            if np.hypot(dy,dx) < 2 * nrng:  # max distance between node centers, init=2
                link = CderG(_G=_G, G=G)
                comp_G(link, Et, lenH, lenHH)

    form_graph_t(root, node_, Et, nrng)  # root_fd, eval sub+, feedback per graph
    if isinstance(node_[0],list):  # node_t was formed above

        for i, G_ in enumerate(node_):
            if root.valt[i] * (len(G_)-1)*root.rng > G_aves[i] * root.rdnt[i]:
                # agg+/ node_( sub)agg+/ node, vs sub+ only in comp_slice
                agg_recursion(rroot, root, G_, nrng=1)  # der+ if fd, else rng+ =2
                if rroot:
                    rroot.fback_t[i] += [[root.aggH,root.valt,root.rdnt,root.dect]]
                    feedback(rroot,i)  # update root.root..


def agg_recursion_old(rroot, root, node_, nrng=1, fagg=0, rfd=0):  # cross-comp and sub-cluster Gs in root graph node_:

    Et = [[0,0],[0,0],[0,0]]

    # agg+: der=1 xcomp of new Gs if fagg, else sub+: der+ xcomp of old Gs:
    nrng = rng_recursion(rroot, root, root.link_ if rfd else node_, Et, nrng, rfd)  # rng+ appends rim, link.derH

    GG_t = form_graph_t(root, node_, Et, nrng, fagg=fagg)  # may convert root.node_[-1] to node_t
    GGG_t = []  # add agg+ fork tree:

    while GG_t:  # unpack fork layers?
        # not sure:
        _GG_t, GGG_t = [],[]
        for fd, GG_ in enumerate(_GG_t):
            if not fd: nrng+=1
            if root.Vt[fd] * (len(GG_)-1) * nrng*2 > G_aves[fd] * root.Rt[fd]:
                # agg+ / node_t, vs. sub+ / node_:
                GGG_t, Vt, Rt  = agg_recursion(rroot, root, GG_, nrng=1, fagg=1, rfd=fd)  # for agg+, lenHH stays constant
                '''
                if rroot:
                    rroot.fback_t[fd] += [[root.aggH, root.valt, root.rdnt, root.dect]]
                    feedback(rroot,fd)  # update root.root..
                for i in 0,1:
                    if Vt[i] > G_aves[i] * Rt[i]:
                        GGG_t += [[i, GGG_t[fd][i]]]
                        # sparse agglomerated current layer of forks across GG_tree
                        GG_t += [[i, GG_t[fd][i],1]]  # i:fork, 1:packed sub_GG_t?
                        # sparse lower layer of forks across GG_tree
                    else:
                        GGG_t += [[i, GG_t[fd][i]]]  # keep lower-composition GGs
                '''
            GG_t = _GG_t  # for next loop

    return GGG_t  # should be tree nesting lower forks


def form_graph_t(root, G_, Et, nrng, lenH=0, lenHH=None):  # form Gm_,Gd_ from same-root nodes

    # select Gs connected in current layer:
    _G_ = [G for G in G_ if len(G.rim_t[0])>len(root.rim_t[0])]

    node_connect(_G_, lenHH)  # Graph Convolution of Correlations over init _G_
    node_t = []
    for fd in 0,1:
        if Et[0][fd] > ave * Et[1][fd]:  # eValt > ave * eRdnt: cluster
            graph_ = segment_node_(root, _G_, fd, nrng,lenH=0)  # fd: node-mediated Correlation Clustering
            for graph in graph_:
                # eval sub+ per node
                if graph.Vt[fd] * (len(graph.node_)-1)*root.rng > G_aves[fd] * graph.Rt[fd]:
                    node_ = graph.node_  # flat in sub+
                    if lenH: lenH = len(node_[0].esubH[-lenH:])  # in agg_compress
                    else:    lenH = len(graph.aggH[-1][0])  # in agg_recursion
                    agg_recursion(root, graph, node_, nrng, lenH, lenHH)
                else:
                    root.fback_t[root.fd] += [[graph.aggH, graph.valt, graph.rdnt, graph.dect]]
                    feedback(root,root.fd)  # update root.root..
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]
    if any(node_t):
        G_[:] = node_t  # else keep root.node_


def comp_G(link, Et, lenH=0, lenHH=None):  # lenH in sub+|rd+, lenHH in agg_compress sub+ only

    _G, G = link._G, link.G
    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0, 1,1, 0,0
    # keep separate P ptuple and PP derH, empty derH in single-P G, + empty aggH in single-PP G:

    # / P:
    mtuple, dtuple, Mtuple, Dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1, fagg=1)
    mval, dval = sum(mtuple), sum(abs(d) for d in dtuple)  # mval is signed, m=-min in comp x sign
    mrdn = dval>mval; drdn = dval<=mval
    dect = [0,0]
    for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
        for i, (par, max, ave) in enumerate(zip(ptuple, Ptuple, aves)):
            # compute link decay coef: par/ max(self/same)
            if fd: dect[fd] += par/max if max else 1
            else:  dect[fd] += (par+ave)/ max if max else 1
    dertv = [[mtuple,dtuple], [mval,dval],[mrdn,drdn],[dect[0]/6,dect[1]/6]]
    Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn; Mdec+=dect[0]/6; Ddec+=dect[1]/6  # ave of 6 params

    # / PP:
    dderH = []
    if _G.derH and _G.derH:  # empty in single-node Gs
        for _lay, lay in zip(_G.derH,_G.derH):
            mtuple,dtuple, Mtuple,Dtuple = comp_dtuple(_lay[1], lay[1], rn=1, fagg=1)
            mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
            mrdn = dval > mval; drdn = dval < mval
            mdec, ddec = 0, 0
            for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
                for (par, max, ave) in zip(ptuple, Ptuple, aves):  # different ave for comp_dtuple
                    if fd: ddec += par/max if max else 1
                    else:  mdec += (par+ave)/(max) if max else 1
            mdec /= 6; ddec /= 6
            Mval+=dval; Dval+=mval; Mdec=(Mdec+mdec)/2; Ddec=(Ddec+ddec)/2
            dderH += [[[mtuple,dtuple], [mval,dval],[mrdn,drdn],[mdec,ddec]]]

    # / G:
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec])
    derH = [[dertv]+dderH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec], der_ext]

    if _G.aggH and G.aggH:
        subH, valt,rdnt,dect = comp_aggHv(_G.aggH, G.aggH, rn=1)
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
        Mdec = (Mdec+dect[0])/2; Ddec = (Ddec+dect[1])/2
        # flat, appendleft:
        SubH = [[derH]+subH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]]
    else:
        SubH = derH

    link.Vt,link.Rt,link.Dt = Valt,Rdnt,Dect = [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]  # reset per comp_G
    # draft:
    fd = 1 if link.subH else 0  # test nesting top-down:

    if lenHH:  # root agg+
        if len(link.daggH) > lenHH:
            HHadd = 1  # append link rim_tH
            if lenH:  # root sub+
                rim_depth = 3  # 1: rimt, 2: rim_tH'rimt if lenHH else rim_t, 3: rim_tH'rim_t
                if len(link.daggH[-1]) > lenH:  # dsubH
                    Hadd = 1    # append rim_tH[-1][fd]
                    if fd: link.daggH[-1] += [[[SubH]]]  # link.subH is der+, new link in rng+
                else:
                    Hadd = 0
                    if fd: link.daggH[-1] += [[SubH]]
            else:
                rim_depth = 2; Hadd = None  # rim_tH[-1] = rimt
        else:
            HHadd = 0  # full daggH
            if fd: link.daggH[-1] += [[SubH]]
    else:
        HHadd = None  # use to distinguish between rim_tH[-1]'rimt and rim_t if depth=2
        if lenH:  # root sub+
            rim_depth = 2  # rim_t
            if len(link.daggH[-1]) > lenH:  # dsubH
                Hadd = 1
                if fd: link.daggH[-1][-1] += [[[SubH]]]  # link.subH is der+, new link in rng+
            else:
                Hadd = 0
                if fd: link.daggH[-1] += [[SubH]]
        else:
            rim_depth = 1; Hadd = None  # G.rim_t = rimt

    for fd, (Val,Rdn,Dec) in enumerate(zip(Valt,Rdnt,Dect)):
        if Val > G_aves[fd] * Rdn:
            # eval fork grapht in form_graph_t:
            Et[0][fd] += Val; Et[1][fd] += Rdn; Et[2][fd] += Dec
            append_rim(link, Val,Rdn,Dec, rim_depth, Hadd, HHadd, fd)

# not updated:

def append_rim(link, Val,Rdn,Dec, lenH, lenHH, fd):

    for G in link._G, link.G:
        if G.rim_t:
            link_depth = get_depth(G.rim_t)
        else:  # empty rim_t: init rim_t
            link_depth = 3
            G.rim_t = [[],[]]
        rim_t = G.rim_t

        if link_depth == 4:  # rim_tH
            if len(rim_t) > lenHH:  # rim_tH not incremented yet
                rim_t += [[[[]],[[link]]]] if fd else [[[[link]],[[]]]]  # add base rimt, no rd+ yet
            else:
                rim_ = rim_t[-1][fd]
                if len(rim_) == lenH: rim_ += [[]]  # add rim layer
                rim_[-1] += [link]  # append rim layer

        elif link_depth == 3:  # rim_t
            if len(rim_t[fd]) == lenH: rim_t[fd] += [[]]  # add rim layer (additional bracket to pack new layer)
            rim_t[fd][-1] += [link]

        elif link_depth == 2:  # rimt
            rim_t[fd] = link  # if init as None,None?

        G.Vt[fd] += Val; G.Rt[fd] += Rdn; G.Dt[fd] += Dec


        ''' old:
        root_depth = (lenH != 0) + (lenHH != 0)
        if lenHH == None:
            if len(rim_t) == root_depth:  # root_depth and rim_t not incremented yet
                # add link layer:
                if fd: G.rim_t = [[[],[link]]]; G.Vt=[0,Val]; G.Rt=[0,Rdn]; G.Dt=[0,Dec]
                else:  G.rim_t = [[[link],[]]]; G.Vt=[Val,0]; G.Rt=[Rdn,0]; G.Dt=[Dec,0]
            else:
                # append last link layer:
                rim_t[0][-1][fd] += [link];  G.Vt[fd] += Val; G.Rt[fd] += Rdn; G.Dt[fd] += Dec

        else:  # agg_compress rd+
            if fd: rim_t = [[[]],[[link]]]; G.Vt=[0,Val]; G.Rt=[0,Rdn]; G.Dt=[0,Dec]
            else:  rim_t = [[[link]],[[]]]; G.Vt=[Val,0]; G.Rt=[Rdn,0]; G.Dt=[Dec,0]
            if lenHH:  # depth = 2
                if len(G.rim_t[-1][fd]) == lenH:  # # init new link_ in rim_t[fd]
                    G.rim_t[-1][fd] += rim_t[fd]
                else:  # accumulate link
                    G.rim_t[-1][fd][-1] += [link]
            elif lenH:  # depth = 1
                if len(G.rim_t[fd]) == lenH:  # init new link_ in rim_t[fd]
                    G.rim_t[fd] += rim_t[fd]
                else:  # accumulate link
                    G.rim_t[fd][-1] += [link]
            else:  # depth = 0, init rim_t
                G.rim_t = rim_t
            '''

def sum_links_last_lay(G, fd, lenH):  # esubLay += last_lay/ link, lenH corresponds to len link subH in agg+ or link.subH[-1] in agg++

    esubLay = []  # accum from links
    lenHH = G.lenHH

    for link in unpack_rim(G.rim_t, fd, lenHH):  # links in rim
        subH = link.subH
        subH_depth = get_depth(G.rim_t)  # same nesting in link.subH
        frdH = 0
        if subH_depth > 2:
            frdH = 1 # sum rdH
            if   subH_depth==4: subH = subH[-1][fd][-1]
            elif subH_depth==3: subH = subH[fd][-1]
            elif subH_depth==2: subH = subH[fd]
        L = lenHH if lenHH != None else lenH
        if len(subH) <= L:
            continue  # was not appended in last sub+ of agg++
        if frdH:
            if len(subH) == 5:  # subH[0] is derHs, when G.aggH is empty
                sum_derHv(esubLay, subH, base_rdn=link.Rt[fd])
            else:
                for derH in subH[0][int(len(subH) / 2):]:  # derH_/ last xcomp: len subH *= 2
                    sum_derHv(esubLay, derH, base_rdn=link.Rt[fd])  # sum all derHs of link layer=rdH into esubH[-1]

        elif subH_depth == 2: sum_derHv(esubLay, subH[-1], base_rdn=link.Rt[fd])  # single layer of agg+'sub+, no rd+
        elif subH_depth == 1: sum_derHv(esubLay, subH, base_rdn=link.Rt[fd])  # single derH

        G.evalt[fd] += link.Vt[fd]; G.erdnt[fd] += link.Rt[fd]; G.edect[fd] += link.Dt[fd]
    G.esubH += [esubLay]
    '''
    link subH is appended per xcomp of either fork, with fd represented in root, nesting:
    agg+:  None | subH | subHH/ sub+, same for G.esubH, Cmd dertuples
    agg++: None | subH | subHH/ rd+, subHHH/ sub+
    '''

def unpack_rim(rim_t, fd, lenHH):
    # rim_t in agg+:  None| [mrim, drim]  | rimtH,
    # rim_t in agg++: None| [mrim_,drim_] | rim_tH

    rim_depth = get_depth(rim_t, 1)

    if rim_depth == 3:
        rim = rim_t[-1][fd][-1]  # in rim_tH
    elif rim_depth == 2:
        if lenHH==0: rim = rim_t[fd][-1]  # in rim_t, agg++
        else:        rim = rim_t[-1][fd]  # in rimtH, agg+
    elif rim_depth == 1:
        rim = rim_t[fd]  # in rimt
    else:
        rim = []  # depth = 0, empty rim

    return rim

def sum_links_last_lay(G, fd, lenH, lenHH):  # eLay += last_lay/ link, lenHH: daggH, lenH: dsubH

    eLay = []  # accum from links' dderH or dsubH; rim_t,link.daggH nesting per depth:

    for link in unpack_rim(G.rim_t, fd, lenHH):  # links in rim
        if link.daggH:
            dsubH = []
            daggH = link.daggH
            if G.fHH:  # from sub+
                if len(daggH) > lenHH:
                    if lenH: dsubH = daggH[-1]
                    else: dderH = daggH[-1]
            elif lenH >0: dsubH = daggH  # from der+
            else: dderH = daggH  # from non-recursive comp_G
            if dsubH:
                if len(dsubH) > lenH:
                    for dderH in dsubH[ int(len(dsubH)/2): ]:  # derH_/ last xcomp: len subH *= 2
                        sum_derHv(eLay, dderH, base_rdn=link.Rt[fd])  # sum all derHs of link layer=rdH into esubH[-1]
            else:  # empty dsubH
                sum_derHv(eLay, dderH, base_rdn=link.Rt[fd])

        G.evalt[fd] += link.Vt[fd]; G.erdnt[fd] += link.Rt[fd]; G.edect[fd] += link.Dt[fd]
    G.extH += [eLay]


def append_rim(link, dsubH, Val,Rdn,Dec, fd, lenH, lenHH):

    for G in link._G, link.G:
        if not G.rim_t: G.rim_t = [[],[]]  # must include link
        rim_t = G.rim_t

        if G.fHH:  # append existing G.rim_tH[-1] rimtH | rim_tH:
            if lenHH==-1 and rim_t:  # 1st sub+
                rim_t[:] = [rim_t]  # convert rimt|rim_t to rimtH|rim_tH
            if lenH == 0:  # 1st der+
                rim_t[-1][:] = [[rim_t[-1][0]],[rim_t[-1][1]]]  # convert last rimt to rim_t
            if len(rim_t)-1 == lenHH:
                if lenH == -1: rim_t += [[[],[]]]  # add rimt
                else:          rim_t += [[[[]],[[]]]]  # add rim_t
            if lenH == -1:
                rim_t[-1][fd] += [link]  # rimtH
            else:
                rim_t[-1][fd][-1] += [link]  # rim_tH
        else:
            # G.rimt | rim_t
            if lenH == -1: rim_t[fd] += [link]  # rimt
            else:
                if lenH == 0:  # 1st der+
                    rim_t[:] = [[rim_t[0]],[rim_t[1]]]  # convert rimt to rim_t
                if len(rim_t[fd]) - 1 == lenH:
                    rim_t[fd] += [[]]  # add der+ layer
                rim_t[fd][-1] += [link]

        if fd: # empty link.daggH in rng+
            if lenH == 0: dsubH = [dsubH]  # convert dderH to dsubH
            if lenHH== 0: dsubH = [dsubH]  # convert dsubH to daggH (-1 is default, so we need 0 here)
            link.daggH += [dsubH]

        G.Vt[fd] += Val; G.Rt[fd] += Rdn; G.Dt[fd] += Dec
'''
    for fd, (Val,Rdn,Dec) in enumerate(zip(Valt,Rdnt,Dect)):
        if Val > G_aves[fd] * Rdn:
            # eval fork grapht in form_graph_t:
            Et[0][fd] += Val; Et[1][fd] += Rdn; Et[2][fd] += Dec
            G.Vt[fd] += Val;  G.Rt[fd] += Rdn;  G.Dt[fd] += Dec
            for G in link.G, link._G:
                rim_t = G.rim_t
                if lenHH>0: rim_t = rim_t[-1]  # get rimt|rim_t from rimtH|rim_tH
                if lenH >0:  # der+
                    if len(rim_t[fd]) == lenH:
                        rim_t[fd] += [[link]]   # add new rim
                    else:
                        rim_t[fd][-1] += [link]  # rim_t
                else:
                    rim_t[fd] += [link]  # rimt
'''
def get_depth(rim_t, fneg=0):  # https://stackoverflow.com/questions/6039103/counting-depth-or-the-deepest-level-a-nested-list-goes-to

    list_depth = ((max(map(get_depth, rim_t))+1 if rim_t else 1) if isinstance(rim_t, list) else 0)   # -1 if input is not a list

    return list_depth - 1 if fneg else list_depth

def nest_rim(link):

    for G in link._G, link.G:
        rim_t = G.rim_t
        if not rim_t: G.rim_t = [[],[]]  # must include link
        if G.fHH:
            if lenHH==0:  # 1st sub+
                if len(G.rim_t)==2 and (G.rim_t[0] and isinstance(G.rim_t[0][0],CderG)) or (G.rim_t[1] and isinstance(G.rim_t[1][0],CderG)):
                    G.rim_t = [G.rim_t]  # convert rimt|rim_t to rimtH|rim_tH
            rim_t = G.rim_t[-1]  # for next der+ checking (if there's any)

        if lenH == 0:  # 1st der+
            if len(rim_t)==2 and (rim_t[0] and isinstance(rim_t[0][0],CderG)) or (rim_t[1] and isinstance(rim_t[1][0],CderG)):
                rim_t[:] = [[rim_t[0]],[rim_t[1]]]  # convert last rimt to rim_t

def unpack_rim(G, fd):  # rim_t =  [] | rimt|rim_t | rim_tH

    rim_t = G.rim_t
    if G.fHH: rim_t = rim_t[-1]  # nested rimtH | rim_tH
    rim = rim_t[fd]  # unpack mrim or drim
    # not sure:
    if rim and isinstance(rim[0],list):  rim = rim[-1]  # not CderG, rim is rim_
    return rim


def form_graph_t(root, G_, Et, nrng, fagg=0):  # form Gm_,Gd_ from same-root nodes

    node_connect(G_)  # Graph Convolution of Correlations over init _G_
    node_t = []
    for fd in 0, 1:
        if Et[0][fd] > ave * Et[1][fd]:  # eValt > ave * eRdnt: cluster
            graph_ = segment_node_(root, G_, fd, nrng)  # fd: node-mediated Correlation Clustering
            for graph in graph_:
                # add graph link_ in the evaluation instead of node? Because we need link in der+ sub later
                if graph.Vt[fd] * (len(graph.node_)-1)*root.rng * len(graph.link_) > G_aves[fd] * graph.Rt[fd]:
                    for node in graph.node_:
                        if node.rimH and isinstance(node.rimH[0],CderG):  # 1st sub+: convert rim to rimH
                            node.rimH = [node.rimH]
                        node.rimH += [[]]  # the simplest method is to add new rim layer here?
                    agg_recursion(root, graph, graph.node_, nrng, fagg=0)
                else:
                    root.fback_t[root.fd] += [[graph.aggH, graph.valt, graph.rdnt, graph.dect]]
                    # feedback(root,root.fd)  # update root.root..
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]
    if fagg:
        return node_t
    elif any(node_t):
        G_[:] = node_t  # else keep root.node_  (replacement only in sub+?)

