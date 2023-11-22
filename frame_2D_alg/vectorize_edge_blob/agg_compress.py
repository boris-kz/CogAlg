import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from collections import deque, defaultdict
from .classes import Cgraph, CderG, CPP
from .filters import ave_L, ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd, ave_dI, ave_G, ave_M, ave_Ma
from .slice_edge import slice_edge, comp_angle
from .comp_slice import comp_P_, comp_ptuple, sum_ptuple, comp_derH, sum_derH, sum_dertuple, get_match
from .agg_recursion import comp_G, comp_aggHv, comp_derHv, vectorize_root, form_graph_t, sum_Hts, sum_derHv, sum_ext

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
    vectorize_root(blob, verbose)

def agg_recursion(rroot, root, G_, fd):  # compositional agg+|sub+ recursion in root graph, clustering G_

    form_parP_(parHv = [root.aggH,sum(root.valHt[fd]),sum(root.rdnHt[fd]),sum(root.maxHt[fd])], fd=fd)
    # compress aggH-> pP_,V,R,M: select G V,R,M?
    Valt,Rdnt = comp_G_(G_,fd)  # rng|der cross-comp all Gs, form link_[-1] per G, sum in Val,Rdn

    root.valHt[fd]+=[0]; root.rdnHt[fd] += [1]; root.maxHt[fd] += [0]
    # combined forks sum in form_graph_t feedback
    GG_t = form_graph_t(root, Valt,Rdnt, G_)  # eval sub+ and feedback per graph
    # agg+ xcomp-> form_graph_t loop sub)agg+, vs. comp_slice:
    # sub+ loop-> eval-> xcomp
    for GG_ in GG_t:  # comp_G_ eval: ave_m * len*rng - fixed cost, root update in form_t:
        if root.valHt[0][-1] * (len(GG_)-1)*root.rng > G_aves[fd] * root.rdnHt[0][-1]:
            agg_recursion(rroot, root, GG_, fd=0)  # 1st xcomp in GG_

    G_[:] = GG_t

def form_parP_(parHv, fd):  # last v: value tuple valt,rdnt,maxt

    parH, rVal, rRdn, rDec = parHv  # compressed valt,rdnt,maxt per aggH replace initial summed G vals
    part_P_ = []  # pPs: nested clusters of >ave param tuples, as below:
    Val,Rdn,Dec = 0,0,0; parH = copy(parH)
    part_ = []
    while parH:  # aggHv( subHv( derHv( ptv_, top-down
        '''
        subt = Hv: >4-level list, | ptv: 3-level list, | extt: 2-level list:
        aggHv: [aggH=subHv_, valt, rdnt, dect],
        subHv: [subH=derHv_, valt, rdnt, dect],
        derHv: [derH=ptuple_tv_, valt, rdnt, dect] or extt, mixed in subH
        ptuple_tv: [[mtuple,dtuple], valt, rdnt, dect] 
        '''
        # partial draft:
        _lay = parH[0]; L = 1
        while len(parH) > L:  # len next Lay = len low Lays: 1,1,2,4.: for subH | derH, not aggH?
            hL = 2*L
            lay = parH[L:hL]  # [par_sH, valt, rdnt, dect]
            # comp or unpack?:
            if isinstance(subt[0][0],list):  # not extt
                if isinstance(subt[0][0][0],list):  # subt==Hv
                    subH,val,rdn,dec = subt[0], subt[1][fd], subt[2][fd], subt[3][fd]
                    if val > ave:  # recursive eval,unpack
                        Val+=val; Rdn+=rdn; Dec+=dec  # sum with sub-vals:
                        sub_part_P_t = form_parP_([subH,val,rdn,dec], fd)
                        part_ += [[subH, sub_part_P_t]]
                    else:
                        if Val:  # empty sub_pP_ terminates root pP
                            part_P_ += [[part_,Val,Rdn,Dec]]; rVal+=Val; rRdn+=Rdn; rDec+=Dec  # root params
                            part_= [], Val,Rdn,Dec = 0,0,0  # pP params
                            # reset
                else: form_tuplet_pP_(subt, [part_P_,rVal,rRdn,rDec], [part_,Val,Rdn,Dec], v=1)  # subt is derLay
            else:     form_tuplet_pP_(subt, [part_P_,rVal,rRdn,rDec], [part_,Val,Rdn,Dec], v=0)  # subt is extt
    if part_:
        part_P_ += [[part_,Val,Rdn,Dec]]; rVal+=Val; rRdn+=Rdn; rDec+Dec

    return [part_P_,rVal,rRdn,rDec]  # root values

def comp_parH(_lay,lay):  # nested comp between and then within same-fork dertuples?
    pass

def form_tuplet_pP_(ptuplet, part_P_v, part_v, v):  # ext or ptuple, params=vals

    part_P_,rVal,rRdn,rDec = part_P_v  # root params
    part_,Val,Rdn,Dec = part_v  # pP params
    if v: ptuplet, valt,rdnt,dect = ptuplet

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
        sum_Hts(ValHt,RdnHt,DecHt, valHt,rdnHt,decHt)
    sum_aggH(root.aggH,AggH, base_rdn=0)
    sum_Hts(root.valHt,root.rdnHt,root.decHt, ValHt,RdnHt,DecHt)  # both forks sum in same root

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