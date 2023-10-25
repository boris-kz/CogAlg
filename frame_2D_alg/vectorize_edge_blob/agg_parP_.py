import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from collections import deque, defaultdict
from .classes import Cgraph, CderG, CPP
from .filters import ave_L, ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd, ave_dI, ave_G, ave_M, ave_Ma
from .slice_edge import slice_edge, comp_angle
from .comp_slice import comp_P_, comp_ptuple, sum_ptuple, comp_derH, sum_derH, sum_dertuple, match_func
from .agg_recursion import comp_aggH, comp_ext, sum_box, sum_Hts, sum_derHv, comp_derHv, sum_ext

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

def vectorize_root(blob, verbose):  # vectorization pipeline is 3 composition levels of cross-comp,clustering:

    edge, adj_Pt_ = slice_edge(blob, verbose)  # lateral kernel cross-comp -> P clustering
    comp_P_(edge, adj_Pt_)  # vertical, lateral-overlap P cross-comp -> PP clustering
    # PP cross-comp -> discontinuous graph clustering:
    for fd in 0,1:
        node_ = edge.node_t[fd]  # always PP_t
        if edge.valt[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.rdnt[fd]:
            G_= []
            for i, PP in enumerate(node_):  # convert CPPs to Cgraphs:
                derH,valt,rdnt = PP.derH,PP.valt,PP.rdnt
                # convert to ptuple_tv_, not sure:
                derH[:] = [[[mtuple,dtuple],[sum(mtuple),sum(dtuple)],[],[]] for mtuple,dtuple in derH]
                # init aggH is empty:
                G_ += [Cgraph( ptuple=PP.ptuple, derH=[derH,valt,rdnt,[0,0]], valHt=[[valt[0]],[valt[1]]], rdnHt=[[rdnt[0]],[rdnt[1]]],
                               L=PP.ptuple[-1], i=i, box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box))]
            node_ = G_
            edge.valHt[0][0] = edge.valt[0]; edge.rdnHt[0][0] = edge.rdnt[0]  # copy
            agg_recursion(None, edge, node_, fd=0)  # edge.node_ = graph_t, micro and macro recursive


def agg_recursion(rroot, root, G_, fd):  # compositional agg+|sub+ recursion in root graph, clustering G_

    cluster_params(parHv = [root.aggH,sum(root.valHt[fd]),sum(root.rdnHt[fd]),sum(root.maxHt[fd])], fd=fd)
    # compress aggH-> pP_,V,R,M: select G V,R,M?
    Valt,Rdnt = comp_G_(G_,fd)  # rng|der cross-comp all Gs, form link_H[-1] per G, sum in Val,Rdn

    root.valHt[fd]+=[0]; root.rdnHt[fd] += [1]; root.maxHt[fd] += [0]
    # combined forks sum in form_graph_t feedback
    GG_t = form_graph_t(root, Valt,Rdnt, G_)  # eval sub+ and feedback per graph
    # agg+ xcomp-> form_graph_t loop sub)agg+, vs. comp_slice:
    # sub+ loop-> eval-> xcomp
    for GG_ in GG_t:  # comp_G_ eval: ave_m * len*rng - fixed cost, root update in form_t:
        if root.valHt[0][-1] * (len(GG_)-1)*root.rng > G_aves[fd] * root.rdnHt[0][-1]:
            agg_recursion(rroot, root, GG_, fd=0)  # 1st xcomp in GG_

    G_[:] = GG_t

def cluster_params(parHv, fd):  # last v: value tuple valt,rdnt,maxt

    parH, rVal, rRdn, rMax = parHv  # compressed valt,rdnt,maxt per aggH replace initial summed G vals
    part_P_ = []  # pPs: nested clusters of >ave param tuples, as below:
    part_ = []  # [[subH, sub_part_P_t], Val,Rdn,Max]
    Val,Rdn,Max = 0,0,0; parH = copy(parH)

    while parH:  # aggHv | subHv | derHv (ptv_), top-down
        subt = parH.pop()
        '''    subt = Hv: >4-level list, | ptv: 3-level list, | extt: 2-level list:
        aggHv: [aggH=subHv_, valt, rdnt, maxt],
        subHv: [subH=derHv_, valt, rdnt, maxt],
        derHv: [derH=ptuple_tv_, valt, rdnt, maxt] or extt, mixed in subH
        ptuple_tv: [[mtuple,dtuple], valt, rdnt, maxt] 
        '''
        if isinstance(subt[0][0],list):  # not extt
            if isinstance(subt[0][0][0],list):  # subt==Hv
                subH, val, rdn, max = subt[0], subt[1][fd], subt[2][fd], subt[3][fd]
                if val > ave:  # recursive eval,unpack
                    Val+=val; Rdn+=rdn; Max+=max  # sum with sub-vals:
                    sub_part_P_t = cluster_params([subH, val, rdn, max], fd)
                    part_ += [[subH, sub_part_P_t]]
                else:
                    if Val:  # empty sub_pP_ terminates root pP
                        part_P_ += [[part_,Val,Rdn,Max]]; rVal+=Val; rRdn+=Rdn; rMax+=Max  # root params
                        part_= [], Val,Rdn,Max = 0,0,0  # pP params
                        # reset
            else: cluster_ptuplet(subt, [part_P_,rVal,rRdn,rMax], [part_,Val,Rdn,Max], v=1)  # subt is derLay
        else:     cluster_ptuplet(subt, [part_P_,rVal,rRdn,rMax], [part_,Val,Rdn,Max], v=0)  # subt is extt
    if part_:
        part_P_ += [[part_,Val,Rdn,Max]]; rVal+=Val; rRdn+=Rdn; rMax+=Max

    return [part_P_,rVal,rRdn,rMax]  # root values

def cluster_ptuplet(ptuplet, part_P_v, part_v, v):  # ext or ptuple, params=vals

    part_P_,rVal,rRdn,rMax = part_P_v  # root params
    part_,Val,Rdn,Max = part_v  # pP params
    if v: ptuplet, valt,rdnt,maxt = ptuplet

    valP_t = [[cluster_vals(ptuple) for ptuple in ptuplet if sum(ptuple) > ave]]
    if valP_t:
        part_ += [valP_t]  # params=vals, no sum-> Val,Rdn,Max?
    else:
        if Val:  # empty valP_ terminates root pP
            part_P_ += [[part_, Val, Rdn, Max]]
            part_P_v[1:] = rVal+Val, rRdn+Rdn, rMax+Max  # root values
        part_v[:] = [],0,0,0  # reset

def cluster_vals(ptuple):
    parP_ = []
    parP = [ptuple[0]] if ptuple[0] > ave else []  # init, need to use param type ave instead

    for par in ptuple[1:]:
        if par > ave: parP += [par]
        else:
            if parP: parP_ += [parP]  # terminate parP
            parP = []

    if parP: parP_ += [parP]  # terminate last parP
    return parP_  # may be empty


def form_graph_t(root, Valt,Rdnt, G_):  # form mgraphs and dgraphs of same-root nodes

    graph_t = []
    for G in G_: G.root = [None,None]  # replace with mcG_|dcG_ in segment_node_, replace with Cgraphs in sum2graph
    for fd in 0,1:
        if Valt[fd] > ave * Rdnt[fd]:  # else no clustering
            graph_t += [segment_node_(root, G_,fd)]  # add alt_graphs?
        else:
            graph_t += [[]]  # or G_?
            # eval sub+, not in segment_node_: full roott must be replaced per node within recursion
    for fd, graph_ in enumerate(graph_t): # breadth-first for in-layer-only roots
        root.valHt[fd]+=[0]; root.rdnHt[fd]+=[1]  # remove if stays 0?

        for graph in graph_:  # external to agg+ vs internal in comp_slice sub+
            node_ = graph.node_t  # init flat
            if sum(graph.valHt[fd]) * (len(node_)-1)*root.rng > G_aves[fd] * sum(graph.rdnHt[fd]):  # eval fd comp_G_ in sub+
                agg_recursion(root, graph, node_, fd)  # replace node_ with node_t, recursive
            else:  # feedback after graph sub+:
                root.fback_t[fd] += [[graph.aggH, graph.valHt, graph.rdnHt, graph.maxHt]]
                root.valHt[fd][-1] += graph.valHt[fd][-1]  # last layer, or all new layers via feedback?
                root.rdnHt[fd][-1] += graph.rdnHt[fd][-1]  # merge forks in root fork
                root.maxHt[fd][-1] += graph.maxHt[fd][-1]
            i = sum(graph.valHt[0]) > sum(graph.valHt[1])
            root.rdnHt[i][-1] += 1  # add fork rdn to last layer, representing all layers after feedback

        if root.fback_t and root.fback_t[fd]:  # recursive feedback after all G_ sub+
            feedback(root, fd)  # update root.root.. aggH, valHt,rdnHt

    return graph_t  # root.node_t'node_ -> node_t: incr nested with each agg+?

'''
graphs add nodes if med_val = (val+_val) * decay > ave: connectivity is aggregate, individual link may be a fluke. 
this is density-based clustering, but much more subtle than conventional versions.
negative perimeter links may turn positive with increasing mediation: linked node val may grow faster than rdn
positive links are core, it's density can't decrease, evaluate perimeter: negative or new links
'''
# tentative
def segment_node_(root, G_, fd):  # sum surrounding link values to define connected nodes, incrementally mediated

    ave = G_aves[fd]
    graph_ = []
    # initialize proto-graphs with each node | max, eval perimeter links to add other nodes
    for G in G_:
        graph_ += [[G, G.valHt[fd][-1], G.rdnHt[fd][-1], [G], [G.link_H[-1]]]]
        # init graph = G,val,rdn,node_, perimeter: negative or new links
    for G in G_:
        for link in G.link_H[-1]:  # these are G-external links, not internal as in comp_slice
            _G = link.G if link._G is G else link._G
            # tentative convert link Gs to proto-graphs:
            link.G = graph_[G.i]; link._G = graph_[_G.i]
    _Val,_Rdn = 0,0
    # eval incr mediated links, sum perimeter Vals, append node_, while significant Val update:
    while True:
        DVal,DRdn, Val,Rdn = 0,0,0,0
        # update surround per graph:
        for graph in graph_:
            G, val, rdn, node_, perimeter = graph
            new_perimeter = []
            periVal, periRdn = 0,0
            for link in perimeter:
                _graph = link.G if link._G is G else link._G
                _G,_val,_rdn,_node_,_perimeter = _graph
                if _G not in G_ or _G in node_: continue
                # use relative link vals only:
                try: decay = link.valt[fd]/link.maxt[fd]  # link decay coef: m|d / max, base self/same
                except ZeroDivisionError: decay = 1
                med_val = (val+_val) * decay
                med_rdn = (rdn+_rdn) * decay
                # tentative:
                if med_val > ave * med_rdn:
                    Val += med_val; periVal += med_val
                    Rdn += med_rdn; periRdn += med_rdn
                    new_perimeter = set(new_perimeter+_perimeter)
                    # merge mediated _graph in graph:
                    val+=_val; rdn+=_rdn; node_ = set(node_+_node_); perimeter = set(perimeter+_perimeter)
                else:
                    new_perimeter = set(new_perimeter + link)  # for re-evaluation?
            # if links per node_link_: evaluate for inclusion in perimeter as a group?
            # k = node.i; graph_[k][1] += periVal; graph_[k][2] += periRdn
            # not updated:
            i = G.i; graph_[i][1] = Val; graph_[i][2] = Rdn
            DVal += Val-_Val; DRdn += Rdn-_Rdn  # update / surround extension, signed
            perimeter[:] = new_perimeter
        if DVal < ave*DRdn:  # even low-Dval extension may be valuable if Rdn decreases?
            break
        _Val,_Rdn = Val,Rdn

    return [sum2graph(root, graph, fd) for graph in graph_ if graph[1] > ave * graph[2]]  # Val > ave * Rdn


def sum2graph(root, cG_, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    graph = Cgraph(root=root, fd=fd, L=len(cG_))  # n nodes, transplant both node roots
    SubH = [[],[0,0],[1,1],[0,0]]; maxM,maxD, Mval,Dval, Mrdn,Drdn = 0,0, 0,0, 0,0
    Link_= []
    for i, G in enumerate(cG_[3]):
        # sum nodes in graph:
        sum_box(graph.box, G.box)
        sum_ptuple(graph.ptuple, G.ptuple)
        sum_derH(graph.derH, G.derH, base_rdn=1)
        sum_aggH(graph.aggH, G.aggH, base_rdn=1)
        sum_Hts(graph.valHt, graph.rdnHt, graph.maxHt, G.valHt, G.rdnHt, G.maxHt)

        subH=[[],[0,0],[1,1],[0,0]]; mval,dval, mrdn,drdn, maxm,maxd = 0,0, 0,0, 0,0
        for derG in G.link_H[-1]:
            if derG.valt[fd] > G_aves[fd] * derG.rdnt[fd]:  # sum positive links only:
                _subH = derG.subH
                (_mval,_dval),(_mrdn,_drdn),(_maxm,_maxd) = valt,rdnt,maxt = derG.valt, derG.rdnt, derG.maxt
                if derG not in Link_:
                    sum_subH(SubH, [_subH,valt,rdnt,maxt] , base_rdn=1)  # new aggLev, not from nodes: links overlap
                    Mval+=_mval; Dval+=_dval; Mrdn+=_mrdn; Drdn+=_drdn; maxM+=_maxm; maxD+=_maxd
                    graph.A[0] += derG.A[0]; graph.A[1] += derG.A[1]; graph.S += derG.S
                    Link_ += [derG]
                mval+=_mval; dval+=_dval; mrdn+=_mrdn; drdn+=_drdn; maxm+=_maxm; maxd+=_maxd
                sum_subH(subH, [_subH,valt,rdnt,maxt], base_rdn=1, fneg = G is derG.G)  # fneg: reverse link sign
                sum_box(G.box, derG.G.box if derG._G is G else derG._G.box)
        # from G links:
        if subH: G.aggH += [subH]
        G.i = i
        G.valHt[0]+=[mval]; G.valHt[1]+=[dval]; G.rdnHt[0]+=[mrdn]; G.rdnHt[1]+=[drdn]
        G.maxHt[0]+=[maxm]; G.maxHt[1]+=[maxd]
        G.root[fd] = graph  # replace cG_
        graph.node_t += [G]  # converted to node_t by feedback
    # + link layer:
    graph.valHt[0]+=[Mval]; graph.valHt[1]+=[Dval]; graph.rdnHt[0]+=[Mrdn]; graph.rdnHt[1]+=[Drdn]
    graph.maxHt[0]+=[maxM]; graph.maxHt[1]+=[maxD]

    return graph

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

    SubH, Valt,Rdnt,Maxt = T; subH, valt,rdnt,maxt = t
    for i in 0,1:
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+ base_rdn; Maxt[i] += maxt[i]
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


def comp_G_(G_, fd=0, oG_=None, fin=1):  # cross-comp in G_ if fin, else comp between G_ and other_G_, for comp_node_

    Mval,Dval, Mrdn,Drdn = 0,0,0,0
    if not fd:  # cross-comp all Gs in extended rng, add proto-links regardless of prior links
        for G in G_: G.link_H += [[]]  # add empty link layer, may remove if stays empty

        if oG_:
            for oG in oG_: oG.link_H += [[]]
        # form new links:
        for i, G in enumerate(G_):
            if fin: _G_ = G_[i+1:]  # xcomp G_
            else:   _G_ = oG_       # xcomp G_,other_G_, also start @ i? not used yet
            for _G in _G_:
                if _G in G.compared_: continue  # skip if previously compared
                dy = _G.box[0] - G.box[0]; dx = _G.box[1] - G.box[1]
                distance = np.hypot(dy, dx)  # Euclidean distance between centers of Gs
                if distance < ave_distance:  # close enough to compare
                    # * ((sum(_G.valHt[fd]) + sum(G.valHt[fd])) / (2*sum(G_aves)))):  # comp rng *= rel value of comparands?
                    G.compared_ += [_G]; _G.compared_ += [G]
                    G.link_H[-1] += [CderG( G=G, _G=_G, S=distance, A=[dy,dx])]  # proto-links, in G only
    for G in G_:
        link_ = []
        for link in G.link_H[-1]:  # if fd: follow links, comp old derH, else follow proto-links, form new derH
            if fd and link.valt[1] < G_aves[1]*link.rdnt[1]: continue  # maybe weak after rdn incr?
            mval,dval, mrdn,drdn = comp_G(link_,link, fd)
            Mval+=mval;Dval+=dval; Mrdn+=mrdn;Drdn+=drdn
        G.link_H[-1] = link_
        '''
        same comp for cis and alt components?
        for _cG, cG in ((_G, G), (_G.alt_Graph, G.alt_Graph)):
            if _cG and cG:  # alt Gs maybe empty
                comp_G(_cG, cG, fd)  # form new layer of links:
        combine cis,alt in aggH: alt represents node isolation?
        comp alts,val,rdn? cluster per var set if recurring across root: type eval if root M|D? '''

    return [Mval,Dval], [Mrdn,Drdn]

# draft
def comp_G(link_, link, fd):

    Mval,Dval, maxM,maxD, Mrdn,Drdn = 0,0, 0,0, 1,1
    _G, G = link._G, link.G
    # keep separate P ptuple and PP derH, empty derH in single-P G, + empty aggH in single-PP G:
    # / P:
    mtuple, dtuple, Mtuple, Dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1, fagg=1)
    maxm, maxd = sum(Mtuple), sum(Dtuple)
    mval, dval = sum(mtuple), sum(abs(d) for d in dtuple)  # mval is signed, m=-min in comp x sign
    mrdn = dval>mval; drdn = dval<=mval
    derLay0 = [[mtuple,dtuple],[mval,dval],[mrdn,drdn],[maxm,maxd]]
    Mval+=mval; Dval+=dval; Mrdn += mrdn; Drdn += drdn; maxM+=maxm; maxD+=maxd
    # / PP:
    _derH,derH = _G.derH,G.derH
    if _derH[0] and derH[0]:  # empty in single-node Gs
        dderH, valt, rdnt, maxt = comp_derH(_derH[0], derH[0], rn=1)
        maxM += maxt[0]; maxD += maxt[0]
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
    else:
        dderH = []
    derH = [[derLay0]+dderH, [Mval,Dval], [Mrdn,Drdn], [maxM,maxD]]  # appendleft derLay0 from comp_ptuple
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval],[Mrdn,Drdn], [maxM,maxD])
    SubH = [der_ext, derH]  # two init layers of SubH, higher layers added by comp_aggH:
    # / G:
    if fd:  # else no aggH yet?
        subH, valt, rdnt, maxt = comp_aggH(_G.aggH, G.aggH, rn=1)
        maxM += maxt[0]; maxD += maxt[0]
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
        link.subH = SubH+subH  # append higher subLayers: list of der_ext | derH s
        link.valt = [Mval,Dval]; link.rdnt = [Mrdn,Drdn]; link.maxt = [maxM,maxD]  # complete proto-link
        link_ += [link]

    elif Mval > ave_Gm or Dval > ave_Gd:  # or sum?
        link.subH = SubH
        link.valt = [Mval,Dval]; link.rdnt = [Mrdn,Drdn]; link.maxt = [maxM,maxD] # complete proto-link
        link_ += [link]

    return Mval,Dval, Mrdn,Drdn


def feedback(root, fd):  # called from form_graph_, append new der layers to root

    AggH, ValHt, RdnHt, MaxHt = deepcopy(root.fback_t[fd].pop(0))  # init with 1st tuple
    while root.fback_t[fd]:
        aggH, valHt, rdnHt, maxHt = root.fback_t[fd].pop(0)
        sum_aggH(AggH, aggH, base_rdn=0)
        sum_Hts(ValHt,RdnHt,MaxHt, valHt,rdnHt,maxHt)
    sum_aggH(root.aggH,AggH, base_rdn=0)
    sum_Hts(root.valHt,root.rdnHt,root.maxHt, ValHt,RdnHt,MaxHt)  # both forks sum in same root

    if isinstance(root, Cgraph) and root.root:  # root is not CEdge, which has no roots
        rroot = root.root
        fd = root.fd  # current node_ fd
        fback_ = rroot.fback_t[fd]
        fback_ += [[AggH, ValHt, RdnHt, MaxHt]]
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
        for link in node.link_H[-1]:
            _node = link.G if link._G is node else link._G
            if val > Gt_[_node.it[fd]][1]:
                non_max_ += [_node]  # skip as next node
            else:
                fmax = 0; break  # break is not necessary?
        if fmax:
            init_ += [[node,val]]
    return init_
