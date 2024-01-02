import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from collections import deque, defaultdict
from .classes import Cgraph, CderG, CPP
from .filters import ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd, ave_dI
from .slice_edge import slice_edge, comp_angle
from .comp_slice import comp_P_, comp_ptuple, comp_derH, sum_derH, sum_dertuple, get_match
from .agg_recursion import node_connect, segment_node_, comp_G, comp_aggHv, comp_derHv, sum_derHv, sum_ext, sum_subHv, sum_aggHv

'''
Implement sparse param tree in aggH: new graphs represent only high m|d params + their root params.
Compare layers in parallel by flipping H, comp forks independently. Flip HH,HHH.. in deeper processing? 

1st: cluster root params within param set, by match per param between previously cross-compared nodes
2nd: cluster root nodes per above-average param cluster, formed in 1st step. 

specifically, x-param comp-> p-clustering of root AggH( SubH( DerH( Vars, after all nodes cross comp, @xp rng 
param xcomp in derivation order, nested for lower level of hierarchical xcomp?  

compress aggH: param cluster nesting reflects root param set nesting (which is a superset of param clusters). 
exclusively deeper param cluster has empty (unpacked) higher param nesting levels.

Mixed-forks: connectivity cluster must be contiguous, not uniform, as distant nodes don't need to be similar?
Nodes are connected by m|d of different param sets in links, potentially clustered in pPs for compression.

Then combine graph with alt_graphs?
'''


def root(blob, verbose):  # vectorization pipeline is 3 composition levels of cross-comp,clustering
    edge = vectorize_root(blob, verbose)
    # temporary
    for fd, G_ in enumerate(edge.node_[-1]):
        if edge.aggH:
            agg_recursion_cpr(None, edge, G_, lenH=0, fd=0, nrng=1)


def vectorize_root(blob, verbose):  # vectorization in 3 composition levels of xcomp, cluster:

    edge, adj_Pt_ = slice_edge(blob, verbose)  # lateral kernel cross-comp -> P clustering

    comp_P_(edge, adj_Pt_)  # vertical, lateral-overlap P cross-comp -> PP clustering
    edge.node_ = [edge.node_]

    for fd, node_ in enumerate(edge.node_[-1]):  # always node_t
        if edge.valt[fd] * (len(node_) - 1) * (edge.rng + 1) > G_aves[fd] * edge.rdnt[fd]:
            for PP in node_: PP.roott = [None,None]
            agg_recursion(None, edge, lenH=0, lenHH=0, fd=0)
            # PP cross-comp -> discontinuous clustering, agg+ only, no Cgraph nodes

# draft:
def agg_recursion(rroot, root, lenH, lenHH, fd, nrng=0):  # compositional agg|sub recursion in root graph, cluster G_

    Et = [[0,0],[0,0],[0,0]]
    if isinstance(root.node_[-1][0], list):
        node_ = root.node_[-1][fd]  # agg+ / GG_ from sub+
    else:
        node_ = root.node_[-1]  # sub+ / G_ from sum2graph
    # init lenH = lenHH[-1] = 0:
    rd_recursion(rroot, root, 0, Et, fd, nrng=1)
    # may convert root.node_[-1] to node_t:
    _GG_t = form_graph_t(root, node_, Et, nrng)
    GGG_t = []  # agg+ fork tree
    rng = 2

    while _GG_t:  # fork layer, recursive unpack lower forks
        GG_t, GGG_t = [],[]
        for fd, GG_ in enumerate(_GG_t):
            if not fd: rng+=2
            if root.Vt[fd] * (len(GG_)-1)*rng > G_aves[fd] * root.Rt[fd]:
                # agg+/ node_( sub)agg+/ node, vs sub+ only in comp_slice
                GGG_t, Vt, Rt  = agg_recursion(rroot, root, lenH+1, fd=0)
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

        _GG_t = GG_t  # for next loop

    return GGG_t  # should be tree nesting lower forks


def rd_recursion(rroot, root, lenH, Et, fd, nrng=1):  # rng,der incr over same G_,link_ -> fork tree, represented in rim_t

    Vt, Rt, Dt = Et
    for fd, Q, V,R,D in zip((0,1),(root.node_,root.link_), Vt,Rt,Dt):  # recursive rng+,der+

        ave = G_aves[fd]
        if fd and rroot == None: continue  # no link_ and der+ in base fork

        if V >= ave * R:  # true for init 0 V,R; nrng if rng+, else 0:
            if not fd: nrng += 1
            link_,(vt,rt,dt) = cross_comp(Q, lenH, nrng*(1-fd))

            for i, v,r,d in zip((0,1), vt,rt,dt):
                Vt[i]+=v; Rt[i]+=rt[i]; Dt[i]+=d
                if v >= ave * r:
                    # draft, not sure:
                    if len(Q[0].rim_t)==lenH:  # init fork tree if empty:
                        for G in Q: G.rim_t += [[link_]]

                    if i: root.link_+= link_  # rng+ links
                    # adds to root Et + rim_t, and Et per G:
                    rd_recursion(rroot, root, lenH, [vt,rt,dt], nrng)

def cross_comp(Q, lenH, nrng):

    et = [[0,0],[0,0],[0,0]]
    link_ = []
    if nrng:  # rng+
        for i, _G in enumerate(Q):  # inp_= G_, form new link_ from original node_
            for G in Q[i+1:]:
                dy = _G.box.cy - G.box.cy; dx = _G.box.cx - G.box.cx
                if np.hypot(dy, dx) < 2 * nrng:  # max distance between node centers, init=2
                    link = CderG(_G=_G, G=G)
                    comp_G(link, et, lenH)
                    link_ += [link]
    else:  # der+
        for link in Q:  # inp_= root.link_, reform links
            if link.Vt[1] < G_aves[1] * link.Rt[1]: continue  # maybe weak after rdn incr?
            comp_G(link, et, lenH)
            link_ += [link]

    return link_, et


# not revised:
def form_graph_t(root, G_, Et, nrng):

    _G_ = [G for G in G_ if len(G.rim_t)>len(root.rim_t)]  # prune Gs unconnected in current layer

    node_connect(_G_)  # Graph Convolution of Correlations over init _G_
    node_t = []
    for fd in 0,1:
        if Et[0][fd] > ave * Et[1][fd]:  # eValt > ave * eRdnt, else no clustering, keep root.node_
            graph_ = segment_node_(root, _G_, fd, nrng)  # fd: node-mediated Correlation Clustering
            if not graph_: continue
            for graph in graph_:  # eval sub+ per node
                if graph.Vt[fd] * (len(graph.node_[-1])-1)*root.rng > G_aves[fd] * graph.Rt[fd]:
                    # last sub+ val -> sub+:
                    agg_recursion(root, graph, graph.node_[-1], len(graph.aggH[-1][0]), nrng+1*(1-fd))  # nrng+ if not fd
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


def feedback(root, fd):  # called from form_graph_, append new der layers to root

    AggH, ValHt, RdnHt, DecHt = deepcopy(root.fback_t[fd].pop(0))  # init with 1st tuple
    while root.fback_t[fd]:
        aggH, valHt, rdnHt, decHt = root.fback_t[fd].pop(0)
        sum_aggH(AggH, aggH, base_rdn=0)
    sum_aggH(root.aggH,AggH, base_rdn=0)

    if isinstance(root, Cgraph) and root.root:  # root is not CEdge, which has no roots
        rroot = root.root
        fd = root.fd  # current node_ fd
        fback_ = rroot.fback_t[fd]
        fback_ += [[AggH, ValHt, RdnHt, DecHt]]
        if fback_ and (len(fback_) == len(rroot.node_t)):  # flat, all rroot nodes terminated and fed back
            # getting cyclic rroot here not sure why it can happen, need to check further
            feedback(rroot, fd)  # sum2graph adds aggH per rng, feedback adds deeper sub+ layers


# more selective: only for parallel clustering?
def select_init_(Gt_, fd):  # local max selection for sparse graph init, if positive link

    init_, non_max_ = [],[]  # pick max in direct links, no recursively mediated links max: discontinuous?

    for node, val in Gt_:
        if node in non_max_: continue  # can't init graph
        if val<=0:  # no +ve links
            if sum(node.val_Ht[fd]) > ave * sum(node.rdn_Ht[fd]):
                init_+= [[node, 0]]  # single-node proto-graph
            continue
        fmax = 1
        for link in node.link_:
            _node = link.G if link._G is node else link._G
            if val > Gt_[_node.it[fd]][1]:
                non_max_ += [_node]  # skip as next node
            else:
                fmax = 0; break  # break is not necessary?
        if fmax:
            init_ += [[node,val]]
    return init_

def agg_recursion_cpr(rroot, root, G_, fd):  # compositional agg+|sub+ recursion in root graph, clustering G_

    parHv = [root.aggH,root.valt[fd],root.rdnt[fd],root.dect[fd]]
    form_pP_(pP_=[], parHv=parHv, fd=fd)  # sum is not needed here
    # compress aggH -> pP_,V,R,Y: select G' V,R,Y?

# to create 1st compressed layer
def init_parHv(parH, V, R, Y, fd):

    # 1st layer
    pP, pV,pR, pY = parH[0], parH[0][1][fd], parH[0][2][fd], parH[0][3][fd]
    pP_, part_ = [], []

    # compress 1st layer - always single element
    _,_,_, rV, rR, rY = compress_play_(pP_, [pP], part_, (0, 0, 0), (pV, pR, pY), fd)
    pP_ = [[part_,rV,rR,rY]]

    # rest of the layers
    parHv = [parH[1:], V-pV, R-pR, Y-pY]

    return pP_, parHv

# not updated:
def form_pP_(pP_, parHv, fd):  # fixed H nesting: aggH( subH( derH( parttv_ ))), pPs: >ave param clusters, nested
    '''
    p_sets with nesting depth, Hv is H, valt,rdnt,dect:
    aggHv: [aggH=subHv_, valt, rdnt, dect],
    subHv: [subH=derHv_, valt, rdnt, dect, 2],
    derHv: [derH=parttv_, valt, rdnt, dect, extt, 1]
    parttv: [[mtuple, dtuple],  valt, rdnt, dect, 0]
    '''

    # 1st layer initialization where pP_ is empty
    if not pP_:
        pP_, (parH, rV, rR, rY) = init_parHv(parHv[0], parHv[1], parHv[2], parHv[3], fd)
    else:
        parH, rV,rR,rY = parHv  # uncompressed H vals
        V,R,Y = 0,0,0  # compressed param sets:

    parH = copy(parH); part_ = []
    _play_ = pP_[-1]  # node_ + combined pars
    L = 1
    while len(parH) > L:  # get next player: len = sum(len lower lays): 1,1,2,4.: for subH | derH, not aggH?
        hL = 2 * L
        play_ = parH[L:hL]  # each player is [sub_pH, valt, rdnt, dect]
        # add conditionally compressed layers within layer_:
        pP_ += [form_pP_([], [play_,V,R,Y], fd)] if L > 2 else [play_]

        # compress current play_
        V,R,Y, rV, rR, rY = compress_play_(pP_, play_, part_, (V, R, Y ),(rV, rR, rY), fd)

        # compare compressed layer
        for _play in _play_:
            for play in play_:
                comp_pP(_play, play)
        L = hL

    if part_:
        pP_ += [[part_,V,R,Y]]; rV+=V; rR+=R; rY+=Y

    return [pP_,rV,rR,rY]  # root values


def compress_play_(pP_, play_, part_, rVals, Vals,  fd):

    V, R, Y = Vals
    rV, rR, rY = rVals

    for play in play_:  # 3-H unpack:
        if play[-1]:  # derH | subH
            if play[-1]>1:   # subH
                sspH,val,rdn,dec = play[0], play[1][fd], play[2][fd], play[3][fd]
                if val > ave:  # recursive eval,unpack
                    V+=val; R+=rdn; Y+=dec  # sum with sub-vals:
                    sub_pP_t = form_pP_([], [sspH,val,rdn,dec], fd)
                    part_ += [[sspH, sub_pP_t]]
                else:
                    if V:  # empty sub_pP_ terminates root pP
                        pP_ += [[part_,V,R,Y]]; rV+=V; rR+=R; rY+=Y  # root params
                        part_,V,R,Y = [],0,0,0  # pP params
                        # reset
            else:
                derH, val,rdn,dec,extt = play[0], play[1][fd], play[2][fd], play[3][fd], play[4]
                form_tuplet_pP_(extt, [pP_,rV,rR,rY], [part_,V,R,Y], v=0)
                sub_pP_t = form_pP_([], [derH,val,rdn,dec], fd)  # derH
                part_ += [[derH, sub_pP_t]]
        else:
            form_tuplet_pP_(play, [pP_,rV,rR,rY], [part_,V,R,Y], v=1)  # derLay

    return V, R, Y, rV, rR, rY


def comp_pP(_play, play):
    pass

def form_pP_recursive(parHv, fd):  # indefinite H nesting: (..HHH( HH( H( parttv_))).., init HH = [H] if len H > max

    parH, rV,rR,rY = parHv  # uncompressed summed G vals
    parP_ = []  # pPs: >ave param clusters, nested
    V,R,Y = 0,0,0
    parH = copy(parH); part_ = []; _player = parH[0]  # parP = [_player]  # node_ + combined pars
    L = 1
    while len(parH) > L:  # get next player of len = sum(len lower lays): 1,1,2,4., not for top H?
        hL = 2 * L
        play = parH[L:hL]  # [sparH, svalt, srdnt, sdect, depth]
        if play[-1]:       # recursive eval,unpack:
            subH,val,rdn,dec = play[0], play[1][fd], play[2][fd], play[3][fd]
            if val > ave:
                V+=val; R+=rdn; Y+=dec  # sum with sub-vals:
                sub_pP_t = form_pP_([subH,val,rdn,dec], fd)
                part_ += [[subH, sub_pP_t]]
            else:
                if V:  # empty sub_pP_ terminates root pP
                    parP_ += [[part_,V,R,Y]]; rV+=V; rR+=R; rY+=Y  # root params
                    part_ = [],V,R,Y = 0,0,0  # pP params
                    # reset parP
        else: form_tuplet_pP_(play, [parP_,rV,rR,rY], [part_,V,R,Y], v=1)  # derLay
        L = hL
    if part_:
        parP_ += [[part_,V,R,Y]]; rV+=V; rR+=R; rY+=Y
    return [parP_,rV,rR,rY]  # root values


def comp_parH(_lay,lay):  # nested comp between and then within same-fork dertuples?
    pass

def form_tuplet_pP_(ptuplet, part_P_v, part_v, v):  # ext or ptuple, params=vals

    part_P_,rVal,rRdn,rDec = part_P_v  # root params
    part_,Val,Rdn,Dec = part_v  # pP params
    if v: ptuplet, valt,rdnt,dect,_ = ptuplet

    valP_t = [[form_val_pP_(ptuple) for ptuple in ptuplet if sum(ptuple) > ave]]
    if valP_t:
        part_ += [valP_t]  # params=vals, no sum-> Val,Rdn,Max?
    else:
        if Val:  # empty valP_ terminates root pP
            part_P_ += [[part_, Val, Rdn, Dec]]
            part_P_v[1:] = rVal+Val, rRdn+Rdn, rDec+Dec  # root values
        part_v[:] = [],0,0,0  # reset

def form_val_pP_(ptuple):
    parP_ = []
    parP = [ptuple[0]] if ptuple[0] > ave else []  # init, need to use param type ave instead

    for par in ptuple[1:]:
        if par > ave: parP += [par]
        else:
            if parP: parP_ += [parP]  # terminate parP
            parP = []

    if parP: parP_ += [parP]  # terminate last parP
    return parP_  # may be empty