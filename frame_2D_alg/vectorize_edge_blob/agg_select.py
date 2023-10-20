import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from collections import deque, defaultdict
from .classes import Cgraph, CderG, CPP
from .filters import ave_L, ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd, ave_dI, ave_G, ave_M, ave_Ma
from .slice_edge import slice_edge, comp_angle
from .comp_slice import comp_P_, comp_ptuple, sum_ptuple, sum_dertuple, comp_derH, matchF
from .agg_recursion import sum2graph, comp_aggH, comp_ext, feedback

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

    edge = slice_edge(blob, verbose)  # lateral kernel cross-comp -> P clustering
    comp_P_(edge)  # vertical, lateral-overlap P cross-comp -> PP clustering
    # PP cross-comp -> discontinuous graph clustering:
    for fd in 0,1:
        node_ = edge.node_t[fd]  # always PP_t
        if edge.valt[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.rdnt[fd]:
            G_= []
            for PP in node_:  # convert CPPs to Cgraphs:
                derH,valt,rdnt = PP.derH,PP.valt,PP.rdnt  # init aggH is empty:
                for dderH in derH: dderH += [[0,0]]  # add maxt
                G_ += [Cgraph( ptuple=PP.ptuple, derH=[derH,valt,rdnt,[0,0]], valHt=[[valt[0]],[valt[1]]], rdnHt=[[rdnt[0]],[rdnt[1]]],
                               L=PP.ptuple[-1], box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box))]
            node_ = G_
            edge.valHt[0][0] = edge.valt[0]; edge.rdnHt[0][0] = edge.rdnt[0]  # copy
            agg_recursion(None, edge, node_, fd=0)  # edge.node_t = graph_t, micro and macro recursive


def agg_recursion(rroot, root, G_, fd):  # compositional agg+|sub+ recursion in root graph, clustering G_

    cluster_params(parHv = [root.aggH,sum(root.valHt[fd]),sum(root.rdnHt[fd]),sum(root.maxHt[fd])], fd=fd)
    # compress aggH-> pP_,V,R,M: select G V,R,M?
    Valt,Rdnt = comp_G_(G_,fd)  # rng|der cross-comp all Gs, form link_H[-1] per G, sum in Val,Rdn

    root.valHt[fd]+=[0]; root.rdnHt[fd] += [1]  # sum in form_graph_t feedback, +root.maxHt[fd]?
    # or per fork in form_graph_t?
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

    while parH:  # aggHv | subHv | derHv (ptupletv_), top-down
        subt = parH.pop()  # Hv: >4-level list, or ptupletv: 3-level list, or extt: 2-level list,
        # id / nesting:
        if isinstance(subt[-1][-1],list):  # subt is not extt
            if isinstance(subt[-1][-1][-1],list):  # subt==Hv
                subH, valt, rdnt, maxt = subt
                val, rdn, max = valt[fd],rdnt[fd],maxt[fd]
                if val > ave:  # recursive eval,unpack
                    Val+=val; Rdn+=rdn; Max+=max  # summed with sub-values:
                    sub_part_P_t = cluster_params(subt, fd)
                    part_ += [[subH, sub_part_P_t]]
                else:
                    if Val:  # empty sub_pP_ terminates root pP
                        part_P_ += [[part_,Val,Rdn,Max]]; rVal+=Val; rRdn+=Rdn; rMax+=Max  # root params
                        part_= [], Val,Rdn,Max = 0,0,0  # pP params
                        # reset
            else: cluster_ptuplet(subt, [part_P_,rVal,rRdn,rMax], [part_,Val,Rdn,Max], v=1)  # subt is ptupletv
        else:     cluster_ptuplet(subt, [part_P_,rVal,rRdn,rMax], [part_,Val,Rdn,Max], v=0)  # subt is extt
    if part_:
        part_P_ += [[part_,Val,Rdn,Max]]; rVal+=Val; rRdn+=Rdn; rMax+=Max

    return [part_P_,rVal,rRdn,rMax]  # root values

def cluster_ptuplet(ptuplet, part_P_v, part_v, v):  # ext or ptuple, params=vals

    part_P_,rVal,rRdn,rMax = part_P_v  # root params
    part_,Val,Rdn,Max = part_v  # pP params
    if v: ptuplet, valt,rdnt,maxt = ptuplet  # valt,rdnt,maxt

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

                           # tentative, replacing sum_link_tree_
def segment_node_(G_,fd):  # sum surrounding link values to define connected nodes, with indirectly incr rng, to parallelize:
                           # link lower nodes via incr n of higher nodes
    ave = G_aves[fd]
    graph_ = []
    for i, G in enumerate(G_):
        G.it[fd] = i
        graph_ += [[G, G.valHt[fd][-1],G.rdnHt[fd][-1]], [G],[G],i]  # init val,rdn,node_,perimeter
    _Val, _Rdn = 0, 0

    while True:  # eval incr mediated links, sum perimeter Vals, append node_, while significant Val update:
        DVal,DRdn = 0,0
        Val,Rdn = 0,0  # updated surround of all nodes

        for G, val,rdn, node_,perimeter,i in graph_:
            new_perimeter = []
            for node in perimeter:
                periVal, periRdn = 0,0
                for link in node.link_H[-1]:
                    _node = link.G if link._G is G else link._G
                    if _node not in G_ or _node in node_: continue
                    j = _node[-1]; _val = graph_[j][1]; _rdn = graph_[j][2]
                    # use relative link vals only:
                    try: decay = link.valt[fd]/link.maxt[fd]  # link decay coef: m|d / max, base self/same
                    except ZeroDivisionError: decay = 1
                    # sum mediated vals per node and perimeter node:
                    med_val = (val+_val) * decay; Val += med_val; periVal += med_val
                    med_rdn = (rdn+_rdn) * decay; Rdn += med_rdn; periRdn += med_rdn
                    new_perimeter += [_node]
                k = node[-1]; graph_[k][1] += periVal; graph_[k][2] += periRdn

            graph_[i][1] = Val; graph_[i][2] = Rdn  # unilateral update, computed separately for _node?
            DVal += Val-_Val  # update / surround extension, signed
            DRdn += Rdn-_Rdn
            perimeter[:] = new_perimeter

        if DVal < ave*DRdn:  # even low-Dval extension may be valuable if Rdn decreases?
            break
        _Val,_Rdn = Val,Rdn

    # prune non-max overlapping graphs:
    for graph in graph_:
        node_ = graph[4]  # or Val-ave*Rdn:
        max_root_i = np.argmax([node[1] for node in node_])  # max root: graph nodes = graph roots, bilateral assign
        for i,root in enumerate(node_):
            if i != max_root_i:
                graph_.remove(root)  # graphs don't overlap: we can remove non-max graph instead pruning its nodes
    # prune weak graphs:
    cgraph_ = []
    for graph in graph_:
        if graph[1] > ave * graph[2]:  # Val > ave * Rdn
            cgraph_ += [sum2graph(graph, fd)]

    return cgraph_


def form_graph_t(root, Valt,Rdnt, G_):  # form mgraphs and dgraphs of same-root nodes

    graph_t = []
    for G in G_: G.root = [None,None]  # replace with mcG_|dcG_ in segment_node_, replace with Cgraphs in sum2graph
    for fd in 0,1:
        if Valt[fd] > ave * Rdnt[fd]:  # else no clustering
            graph_t += [segment_node_(G_,fd)]  # add alt_graphs?
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

# not updated:

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
        dderH, valt, rdnt, maxt = comp_derH(_derH[0], derH[0], rn=1, fagg=1)
        maxM += maxt[0]; maxD += maxt[0]
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
    else:
        dderH = []
    derH = [[derLay0]+dderH, [Mval,Dval], [Mrdn,Drdn], [maxM, maxD]]  # appendleft derLay0 from comp_ptuple
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval],[Mrdn,Drdn], [maxM,maxD])
    SubH = [der_ext, derH]  # two init layers of SubH, higher layers added by comp_aggH:
    # / G:
    if fd:  # else no aggH yet?
        subH, valt, rdnt, maxt = comp_aggH(_G.aggH, G.aggH, rn=1)
        SubH += subH  # append higher subLayers: list of der_ext | derH s
        maxM += maxt[0]; maxD += maxt[0]
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
        link_ += [link]
    elif Mval > ave_Gm or Dval > ave_Gd:  # or sum?
        link.subH = SubH; link.maxt = [maxM,maxD]; link.valt = [Mval,Dval]; link.rdnt = [Mrdn,Drdn]  # complete proto-link
        link_ += [link]

    return Mval,Dval, Mrdn,Drdn

