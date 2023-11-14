import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from collections import defaultdict
from .classes import Cgraph, CderG
from .filters import ave_L, ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd
from .slice_edge import slice_edge, comp_angle
from .comp_slice import comp_P_, comp_derH, sum_derH, comp_ptuple, sum_dertuple, comp_dtuple, get_match

'''
Blob edges may be represented by higher-composition patterns, etc., if top param-layer match,
in combination with spliced lower-composition patterns, etc, if only lower param-layers match.
This may form closed edge patterns around flat core blobs, which defines stable objects. 
Graphs are formed from blobs that match over <max distance. 
-
They are also assigned adjacent alt-fork graphs, which borrow predictive value from the graph. 
Primary value is match, so difference patterns don't have independent value. 
They borrow value from proximate or average match patterns, to the extent that they cancel their projected match. 
But alt match patterns borrow already borrowed value, which may be too tenuous to track, we can use the average instead.
-
Graph is abbreviated to G below:
Agg+ cross-comps top Gs and forms higher-order Gs, adding up-forking levels to their node graphs.
Sub+ re-compares nodes within Gs, adding intermediate Gs, down-forking levels to root Gs, and up-forking levels to node Gs.
-
Generic graph is a dual tree with common root: down-forking input node Gs and up-forking output graph Gs. 
This resembles a neuron, which has dendritic tree as input and axonal tree as output. 
But we have recursively structured param sets packed in each level of these trees, which don't exist in neurons.
Diagram: 
https://github.com/boris-kz/CogAlg/blob/76327f74240305545ce213a6c26d30e89e226b47/frame_2D_alg/Illustrations/generic%20graph.drawio.png
-
Clustering criterion is G.M|D, summed across >ave vars if selective comp (<ave vars are not compared, so they don't add costs).
Fork selection should be per var or co-derived der layer or agg level. 
There are concepts that include same matching vars: size, density, color, stability, etc, but in different combinations.
Weak value vars are combined into higher var, so derivation fork can be selected on different levels of param composition.
'''

def vectorize_root(blob, verbose):  # vectorization pipeline is 3 composition levels of cross-comp,clustering:

    edge, adj_Pt_ = slice_edge(blob, verbose)  # lateral kernel cross-comp -> P clustering
    comp_P_(edge, adj_Pt_)  # vertical, lateral-overlap P cross-comp -> PP clustering
    # PP cross-comp -> discontinuous graph clustering:
    for fd, node_ in enumerate(edge.node_t):
        if edge.valt[fd] * (len(node_)-1) * (edge.rng+1) <= G_aves[fd] * edge.rdnt[fd]: continue
        G_ = []
        for PP in node_:  # convert select CPPs to Cgraphs:
            if PP.valt[fd] * (len(node_)-1) * (PP.rng+1) <= G_aves[fd] * PP.rdnt[fd]: continue
            derH,valt,rdnt = PP.derH,PP.valt,PP.rdnt
            derH[:] = [  # convert to ptuple_tv_: [[ptuplet,valt,rdnt,dect]]:
                [[mtuple,dtuple],
                 [sum(mtuple),sum(dtuple)],
                 [sum([m<d for m,d in zip(mtuple,dtuple)]), sum([d<=m for m,d in zip(mtuple,dtuple)])],
                 []] # empty PP.derH dects 
                for mtuple,dtuple in derH]
            G_ += [Cgraph( ptuple=PP.ptuple, derH=[derH,valt,rdnt,[]], valHt=[[valt[0]],[valt[1]]], rdnHt=[[rdnt[0]],[rdnt[1]]], L=PP.ptuple[-1],
                           box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box), link_=PP.link_, node_t=PP.node_t)]
        if G_:
            node_ = G_
            edge.valHt[0][0] = edge.valt[0]; edge.rdnHt[0][0] = edge.rdnt[0]  # copy
            agg_recursion(None, edge, node_, fd=0)  # edge.node_ = graph_t, micro and macro recursive


def agg_recursion(rroot, root, G_, fd):  # + fpar for agg_parP_? compositional agg|sub recursion in root graph, cluster G_

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,0,0,0,0
    link_ = defaultdict(list)
    if fd:
        for link in root.link_:
            if link.valt[1] < G_aves[1]*link.rdnt[1]: continue  # maybe weak after rdn incr?
            mval,dval, mrdn,drdn, mdec,ddec = comp_G(link_,link._G,link.G, fd)
            Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn; Mdec+=mdec; Ddec+=ddec
    else:
        for i, _node in enumerate(G_):  # original node_ for rng+
            for node in G_[i+1:]:
                dy = _node.box[0]-node.box[0]; dx = _node.box[1]-node.box[1]; distance = np.hypot(dy,dx)
                if distance < root.rng:  # <ave Euclidean distance between G centers; or rng * (_G_val+G_val)/ ave*2?
                    mval,dval, mrdn,drdn, mdec,ddec = comp_G(link_,_node,node, fd)
                    Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn; Mdec+=mdec; Ddec+=ddec

    root.valHt[fd] += [0]; root.rdnHt[fd] += [1]  # sum in feedback:
    GG_t = form_graph_t(root, [Mval,Dval],[Mrdn,Drdn], G_, link_)  # eval sub+, feedback per graph
    # agg+ xcomp-> form_graph_t loop sub)agg+, vs. comp_slice sub+ loop-> eval-> xcomp
    for GG_ in GG_t:  # comp_G_ eval: ave_m * len*rng - fixed cost, root update in form_t:
        if root.valHt[0][-1] * (len(GG_)-1)*root.rng > G_aves[fd] * root.rdnHt[0][-1]:
            agg_recursion(rroot, root, GG_, fd=0)  # 1st xcomp in GG_
    G_[:] = GG_t

def form_graph_t(root, Valt,Rdnt, G_, link_):  # form mgraphs and dgraphs of same-root nodes

    graph_t = []
    for G in G_: G.root = [None,None]  # replace with mcG_|dcG_ in segment_node_, replace with Cgraphs in sum2graph
    for fd in 0,1:
        if Valt[fd] > ave * Rdnt[fd]:  # else no clustering
            Gt_ = node_connect(G_, link_, fd)  # AKA Graph Convolution of Correlations
            graph_t += [segment_node_(root, Gt_, fd)]  # if fd: node-mediated Correlation Clustering; add alt_graphs?
        else:
            graph_t += [[]]  # or G_?
            # eval sub+, not in segment_node_: full roott must be replaced per node within recursion
    for fd, graph_ in enumerate(graph_t): # breadth-first for in-layer-only roots
        root.valHt[fd]+=[0]; root.rdnHt[fd]+=[1]  # remove if stays 0?

        for graph in graph_:  # external to agg+ vs internal in comp_slice sub+
            node_ = graph.node_t[fd]  # init flat?
            if sum(graph.valHt[fd]) * (len(node_)-1)*root.rng > G_aves[fd] * sum(graph.rdnHt[fd]):  # eval fd comp_G_ in sub+
                agg_recursion(root, graph, node_, fd)  # replace node_ with node_t, recursive
            else:  # feedback after graph sub+, not revised
                root.fback_t[fd] += [[graph.aggH, graph.valHt, graph.rdnHt, graph.decHt]]
                root.valHt[fd][-1] += graph.valHt[fd][-1]  # last layer, or all new layers via feedback?
                root.rdnHt[fd][-1] += graph.rdnHt[fd][-1]  # merge forks into root fork
                root.decHt[fd][-1] += graph.decHt[fd][-1]
            i = sum(graph.valHt[0]) > sum(graph.valHt[1])
            root.rdnHt[i][-1] += 1  # add fork rdn to last layer, representing all layers after feedback

        if root.fback_t and root.fback_t[fd]:  # recursive feedback after all G_ sub+
            feedback(root, fd)  # update root.root.. aggH, valHt,rdnHt

    return graph_t  # root.node_t'node_ -> node_t: incr nested with each agg+?


def node_connect(iG_,link_,fd):  # node connectivity = sum surround link vals, incr.mediated: Graph Convolution of Correlations
    '''
    aggregate indirect links by associated nodes (vs. individually), iteratively recompute connectivity in multiple cycles,
    effectively blending direct and indirect connectivity measures for each node over time.
    In each cycle, connectivity per node includes aggregated contributions from the previous cycles, propagated through the network.
    Math: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/node_connect.png
    '''
    Gt_ =[]; ave = G_aves[fd]
    for G in iG_:
        valt,rdnt,dect = [0,0],[0,0], [0,0]; rim = copy(link_[G])  # all links that contain G
        for link in rim:
            if link.valt[fd] > ave * link.rdnt[fd]:  # skip negative links
                for i in 0,1:
                    valt[i] += link.valt[i]; rdnt[i] += link.rdnt[i]; dect[i] += link.dect[i]  # sum direct link vals
        Gt_ += [[G, rim,valt,rdnt,dect]]
    _tVal,_tRdn = 0,0

    while True:  # eval same Gs,links, but with cross-accumulated node connectivity values
        tVal, tRdn = 0,0  # loop totals
        for G,rim,valt,rdnt,dect in Gt_:
            rim_val, rim_rdn = 0,0
            for link in rim:
                if link.valt[fd] < ave * link.rdnt[fd]: continue  # skip negative links
                _G = link.G if link._G is G else link._G
                if _G not in iG_: continue
                _Gt = Gt_[G.i]
                _G,_rim_,_valt,_rdnt,_dect = _Gt
                decay = link.dect[fd]  # node vals * relative link val:
                for i in 0,1:
                    linkV = _valt[i] * decay; valt[i]+=linkV
                    if fd==i: rim_val+=linkV
                    linkR = _rdnt[i] * decay; rdnt[i]+=linkR
                    if fd==i: rim_rdn+=linkR
                    dect[i] += link.dect[i]
            tVal += rim_val
            tRdn += rim_rdn
        if tVal-_tVal <= ave * (tRdn-_tRdn):
            break
        _tVal,_tRdn = tVal,tRdn

    return Gt_

def segment_node_(root, Gt_, fd):  # eval rim links with summed surround vals
    '''
    Graphs add nodes if their (surround connectivity * relative value of link to internal node) is above average
    It's a version of density-based clustering.
    '''
    igraph_ = []; ave = G_aves[fd]

    for Gt in Gt_:
        G,rim,valt,rdnt,dect = Gt
        subH = [[],[0,0],[1,1],[0,0]]
        Link_= []; A,S = [0,0],0
        for link in rim:
            if link.valt[fd] > G_aves[fd] * link.rdnt[fd]:
                sum_subHv(subH, [link.subH,link.valt,link.rdnt,link.dect], base_rdn=1)
                Link_ += [link]; A[0] += link.A[0]; A[1] += link.A[1]; S += link.S
        grapht = [[Gt],copy(rim),copy(valt),copy(rdnt),copy(dect),A,S,subH,Link_]
        G.root[fd] = grapht; igraph_ += [grapht]
    _tVal,_tRdn = 0,0
    _graph_ = igraph_  # prune while eval node rim links with surround vals for graph inclusion and merge:
    while True:
        tVal,tRdn = 0,0  # loop totals
        graph_ = []
        for grapht in _graph_:  # extend graph Rim
            nodet_,Rim, Valt,Rdnt,Dect, A,S, subH,link_ = grapht
            inVal,inRdn = 0,0  # in-graph: positive
            new_Rim = []
            for link in Rim:
                if link.G in grapht[0]:
                    Gt = Gt_[link.G.i]; _Gt = Gt_[link._G.i]
                else:
                    Gt = Gt_[link._G.i]; _Gt = Gt_[link.G.i]
                if _Gt in nodet_: continue
                # node match * surround M|D match: of potential in-graph position?
                comb_val = link.valt[fd] + get_match(Gt[2][fd],_Gt[2][fd])
                comb_rdn = link.rdnt[fd] + (Gt[3][fd] + _Gt[3][fd]) / 2
                if comb_val > ave*comb_rdn:
                    # sum links
                    _nodet_,_Rim,_Valt,_Rdnt,_Dect,_A,_S,_subH,_link_ = _Gt[0].root[fd]
                    sum_subHv(subH, _subH, base_rdn=1)
                    A[0] += _A[0]; A[1] += _A[1]; S += _S; link_ += _link_
                    for i in 0,1:
                        Valt[i] += _Valt[i]; Rdnt[i] += _Rdnt[i]; Dect[i] += _Dect[i]
                    inVal += _Valt[fd]; inRdn += _Rdnt[fd]
                    nodet_ += [__Gt for __Gt in _Gt[0].root[fd][0] if __Gt not in nodet_]
                    Rim = list(set(Rim + _Rim))
                    new_Rim = list(set(new_Rim + _Rim))

            tVal += inVal; tRdn += inRdn  # signed?
            if len(new_Rim) * inVal > ave * inRdn:  # eval new_Rim
                graph_ += [[nodet_,new_Rim,Valt,Rdnt,Dect,A,S, subH, link_]]

        if len(graph_) * (tVal-_tVal) <= ave * (tRdn-_tRdn):  # even low-Val extension may be valuable if Rdn decreases?
            break
        _graph_ = graph_
        _tVal,_tRdn = tVal,_tRdn
    # -> Cgraphs if Val > ave * Rdn:
    return [sum2graph(root, graph, fd) for graph in graph_ if graph[2] > ave * graph[3]]


def sum2graph(root, grapht, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    Gt_,Rim,(Mval,Dval),(Mrdn,Drdn),(Mdec,Ddec), A,S, subH,Link_ = grapht

    graph = Cgraph(fd=fd, L=len(Gt_),link_=Link_,A=A,S=S)  # n nodes
    for link in Link_:
        link.roott[fd]=graph
    for i, Gt in enumerate(Gt_):
        Gt += [root]  # Gt: [G,rim,valt,rdnt,dect,root]
        G = Gt[0]
        G.i = i
        sum_box(graph.box, G.box)
        graph.node_t[fd] += [G]
        graph.connec_t[fd] += [Gt[1:]]  # node connectivity params: rim,valt,rdnt,dect,n
        if (Mval,Dval)[fd] > ave * (Mrdn,Drdn)[fd]:  # redundant to nodes, only link_ params are necessary
            graph.ptuple += G.ptuple
            sum_derHv(graph.derH, G.derH, base_rdn=1)
            sum_aggHv(graph.aggH, G.aggH, base_rdn=1)
            sum_Hts(graph.valHt,graph.rdnHt,graph.decHt, G.valHt,G.rdnHt,G.decHt)
    # add derLay:
    graph.aggH += [subH]
    Y,X,Y0,X0,Yn,Xn = graph.box
    graph.box[:2] = [(Y0+Yn)/2, (X0+Xn)/2]
    graph.valHt[0]+=[Mval]; graph.valHt[1]+=[Dval]
    graph.rdnHt[0]+=[Mrdn]; graph.rdnHt[1]+=[Drdn]
    graph.decHt[0]+=[Mdec]; graph.decHt[1]+=[Ddec]

    return graph


def sum_Hts(ValHt,RdnHt,DecHt, valHt,rdnHt,decHt):
    # loop m,d Hs, add combined decayed lower H/layer?
    for ValH,valH, RdnH,rdnH, DecH,decH in zip(ValHt,valHt, RdnHt,rdnHt, DecHt,decHt):
        ValH[:] = [V+v for V,v in zip_longest(ValH, valH, fillvalue=0)]
        RdnH[:] = [R+r for R,r in zip_longest(RdnH, rdnH, fillvalue=0)]
        DecH[:] = [D+d for D,d in zip_longest(DecH, decH, fillvalue=0)]
'''
derH: [[tuplet, valt, rdnt, dect]]: default input from PP, rng+|der+, sum min len?
subH: [derH_t]: m,d derH, m,d ext added by agg+ as 1st tuplet
aggH: [subH_t]: composition levels, ext per G, 
'''

def comp_G(link_, _G, G, fd):

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0, 1,1, 0,0
    link = CderG( _G=_G, G=G)
    S = 6  # min Dect summation span = len ptuple
    # keep separate P ptuple and PP derH, empty derH in single-P G, + empty aggH in single-PP G
    # / P:
    mtuple, dtuple, Mtuple, Dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1, fagg=1)
    mval, dval = sum(mtuple), sum(abs(d) for d in dtuple)  # mval is signed, m=-min in comp x sign
    mrdn = dval>mval; drdn = dval<=mval
    dect = [0,0]
    for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,Mtuple),(dtuple,Dtuple))):
        for par,max in zip(ptuple,Ptuple):
            dect[fd] += par/max if max else 1  # link decay coef: m|d / max, base self/same
    derLay0 = [[mtuple,dtuple],[mval,dval],[mrdn,drdn],dect]
    Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn; Mdec+=dect[0]; Ddec+=dect[1]
    # / PP:
    _derH,derH = _G.derH,G.derH
    if _derH[0] and derH[0]:  # empty in single-node Gs
        S += len(derH[0]) * 6
        dderH, (mval,dval),(mrdn,drdn),(mdec,ddec) = comp_derHv(_derH[0], derH[0], rn=1)
        Mval+=dval; Dval+=mval; Mdec+=mdec; Ddec+=ddec
        Mrdn+= mrdn+ dval>mval; Drdn+= drdn+ dval<=mval
    else: dderH = []
    derH = [[derLay0]+dderH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]]  # appendleft derLay0 from comp_ptuple
    S+=3  # len ext
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec])
    SubH = [der_ext, derH]  # two init layers of SubH, higher layers added by comp_aggH:
    # / G:
    if fd:  # else no aggH yet?
        subH, valt,rdnt,dect, s = comp_aggHv(_G.aggH, G.aggH, rn=1)
        S += s
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
        Mdec += dect[0]; Ddec += dect[1]
        link.subH = SubH+subH  # append higher subLayers: list of der_ext | derH s
        link.valt = [Mval,Dval]; link.rdnt = [Mrdn,Drdn]; link.dect = [Mdec/S,Ddec/S]  # complete proto-link
        link_[G] += [link]
        link_[_G]+= [link]
    elif Mval > ave_Gm or Dval > ave_Gd:  # or sum?
        link.subH = SubH
        link.valt = [Mval,Dval]; link.rdnt = [Mrdn,Drdn]; link.dect = [Mdec/S,Ddec/S] # complete proto-link
        link_[G] += [link]
        link_[_G]+= [link]
        # dict: key=G, values=derGs
    return Mval,Dval, Mrdn,Drdn, Mdec,Ddec

# draft:
def comp_aggHv(_aggH, aggH, rn):  # no separate ext

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    SubH = []; S = min(len(_aggH),len(aggH))

    for _lev, lev in zip_longest(_aggH, aggH, fillvalue=[]):  # compare common subHs, if lower-der match?
        if _lev and lev:
            dsubH, valt,rdnt,dect, s = comp_subHv(_lev[0], lev[0], rn)  # skip valt,rdnt,dect
            SubH += dsubH; S += s  # flatten to keep as subH
            Mdec += dect[0]; Ddec += dect[1]
            mval,dval = valt; Mval += mval; Dval += dval
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + mval <= dval

    return SubH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec], S

def comp_subHv(_subH, subH, rn):

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    dsubH =[]; S = min(len(_subH),len(subH))

    for _lay, lay in zip(_subH, subH):  # compare common lower layer|sublayer derHs
        # if lower-layers match: Mval > ave * Mrdn?
        if _lay[0] and isinstance(_lay[0][0],list):
            S += min(len(_lay[0]),len(lay[0])) * 6  # _lay[0]: derH, 6: len dtuple
            dderH, valt,rdnt,dect = comp_derHv(_lay[0], lay[0], rn)
            dsubH += [[dderH, valt,rdnt,dect]]  # flat
            Mdec += dect[0]; Ddec += dect[1]
            mval,dval = valt; Mval += mval; Dval += dval
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + dval <= mval
        else:  # _lay[0][0] is L, comp dext:
            dsubH += [comp_ext(_lay[1],lay[1],[Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec])]
            S+=3  # pack extt as ptuple
    return dsubH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec], S  # new layer,= 1/2 combined derH


def comp_derHv(_derH, derH, rn):  # derH is a list of der layers or sub-layers, each is ptuple_tv

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    dderH =[]
    for _lay, lay in zip(_derH, derH):  # compare common lower der layers | sublayers in derHs, if lower-layers match?
        # comp dtuples, eval mtuples
        mtuple, dtuple, Mtuple, Dtuple = comp_dtuple(_lay[0][1], lay[0][1], rn, fagg=1)
        # sum params:
        mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
        mrdn = dval > mval; drdn = dval < mval
        dect = [0,0]
        for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,Mtuple),(dtuple,Dtuple))):
            for par, max in zip(ptuple, Ptuple):
                dect[fd] += par/max if max else 1  # link decay coef: m|d / max, base self/same
        dderH += [[[mtuple,dtuple],[mval,dval],[mrdn,drdn],dect]]
        Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn; Mdec+=dect[0]; Ddec+=dect[1]

    return dderH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]  # new derLayer,= 1/2 combined derH


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

def sum_subHv(T, t, base_rdn, fneg=0):

    SubH,Valt,Rdnt,Dect = T; subH,valt,rdnt,dect = t
    for i in 0,1:
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] += dect[i]
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

def sum_derHv(T,t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    DerH,Valt,Rdnt,Dect = T; derH,valt,rdnt,dect = t
    for i in 0,1:
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] += dect[i]
    DerH[:] = [
        [ [sum_dertuple(Dertuple,dertuple, fneg*i) for i,(Dertuple,dertuple) in enumerate(zip(Tuplet,tuplet))],
          [V+v for V,v in zip(Valt,valt)], [R+r+base_rdn for R,r in zip(Rdnt,rdnt)], [D+d for D,d in zip(Dect,dect)]
        ]
        for [Tuplet,Valt,Rdnt,Dect], [tuplet,valt,rdnt,dect]  # ptuple_tv
        in zip_longest(DerH, derH, fillvalue=[([0,0,0,0,0,0],[0,0,0,0,0,0]), (0,0),(0,0),(0,0)])
    ]

def comp_ext(_ext, ext, Valt, Rdnt, Dect):  # comp ds:

    (_L,_S,_A),(L,S,A) = _ext,ext
    dL = _L-L

    if isinstance(A,list):
        if any(A) and any(_A):
            mA,dA = comp_angle(_A,A); adA=dA
        else:
            mA,dA,adA = 0,0,0
        max_mA = max_dA = .5  # = ave_dangle
        dS = _S/_L - S/L  # S is summed over L, dS is not summed over dL
    else:
        mA = get_match(_A,A)- ave_dangle; dA = _A-A; adA = abs(dA); _aA=abs(_A); aA=abs(A)
        max_dA = _aA + aA; max_mA = max(_aA, aA)
        dS = _S - S
    mL = get_match(_L,L) - ave_L
    mS = get_match(_S,S) - ave_L

    m = mL+mS+mA; d = abs(dL)+ abs(dS)+ adA
    Valt[0] += m; Valt[1] += d
    Rdnt[0] += d>m; Rdnt[1] += d<=m
    _aL = abs(_L); aL = abs(L)
    _aS = abs(_S); aS = abs(S)
    Dect[0] += mL/ max(aL,_aL) + (mS/ max(aS,_aS)) if aS or _aS else 1 + (mA/ max_mA) if max_mA else 1
    Dect[1] += dL / (_aL+aL) + (dS / max(_aS+aS)) if aS or _aS else 1 + (dA / max_dA) if max_mA else 1

    return [[mL,mS,mA], [dL,dS,dA]]

def sum_ext(Extt, extt):

    if isinstance(Extt[0], list):
        for Ext,ext in zip(Extt,extt):  # ext: m,d tuple
            for i,(Par,par) in enumerate(zip(Ext,ext)):
                Ext[i] = Par+par
    else:  # single ext
        for i in 0,1: Extt[i]+=extt[i]  # sum L,S
        for j in 0,1: Extt[2][j]+=extt[2][j]  # sum dy,dx in angle

def sum_box(Box, box):
    (_,_,Y0,X0,Yn,Xn), (_,_,y0,x0,yn,xn) = Box, box  # unpack
    Box[2:] = [min(X0,x0), min(Y0,y0), max(Xn,xn), max(Yn,yn)]


def feedback(root, fd):  # called from form_graph_, append new der layers to root

    AggH, ValHt, RdnHt, DecHt = deepcopy(root.fback_t[fd].pop(0))  # init with 1st tuple
    while root.fback_t[fd]:
        aggH, valHt, rdnHt, decHt = root.fback_t[fd].pop(0)
        sum_aggHv(AggH, aggH, base_rdn=0)
        sum_Hts(ValHt,RdnHt,DecHt, valHt,rdnHt,decHt)
    sum_aggHv(root.aggH,AggH, base_rdn=0)
    sum_Hts(root.valHt,root.rdnHt,root.decHt, ValHt,RdnHt,DecHt)  # both forks sum in same root

    if isinstance(root, Cgraph) and root.connec_t[fd]:  # root is not CEdge, which has no roots
        rroot = root.connec_t[fd][0][-1]  # get root from 1st connect [rim,val,rdn,root]
        fd = root.fd  # current node_ fd
        fback_ = rroot.fback_t[fd]
        fback_ += [[AggH, ValHt, RdnHt, DecHt]]
        if fback_ and (len(fback_) == len(rroot.node_t)):  # flat, all rroot nodes terminated and fed back
            # getting cyclic rroot here not sure why it can happen, need to check further
            feedback(rroot, fd)  # sum2graph adds aggH per rng, feedback adds deeper sub+ layers