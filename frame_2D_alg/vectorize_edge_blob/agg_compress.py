import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from collections import deque, defaultdict
from .classes import Cgraph, CderG, CPP
from .filters import ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd, ave_dI
from .slice_edge import slice_edge, comp_angle
from .comp_slice import comp_P_, comp_ptuple, comp_derH, sum_derH, sum_dertuple, get_match
from .agg_recursion import comp_G, comp_aggHv, comp_derHv, vectorize_root, form_graph_t, sum_derHv, sum_ext

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
            agg_recursion(None, edge, G_, fd)


def agg_recursion(rroot, root, G_, fd):  # compositional agg+|sub+ recursion in root graph, clustering G_

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


def sum_aggH(AggH, aggH, base_rdn):

    if aggH:
        if AggH:
            for Layer, layer in zip_longest(AggH,aggH, fillvalue=[]):
                if layer:
                    if Layer:
                        sum_subH(Layer, layer, base_rdn)
                    else:
                        AggH += [deepcopy(layer)]
        else:
            AggH[:] = deepcopy(aggH)


def sum_subH(T, t , base_rdn, fneg=0):

    SubH, Valt,Rdnt,Dect = T; subH, valt,rdnt,dect = t
    for i in 0,1:
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+ base_rdn; Dect[i] += dect[i]
    if SubH:
        for Layer, layer in zip_longest(SubH,subH, fillvalue=[]):
            if layer:
                if Layer:
                    if layer[0] and isinstance(Layer[0][0], list):  # _lay[0][0] is derH
                        sum_derHv(Layer, layer, base_rdn, fneg)
                    else: sum_ext(Layer, layer)
                else:
                    SubH += [deepcopy(layer)]  # _lay[0][0] is mL
    else:
        SubH[:] = deepcopy(subH)


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