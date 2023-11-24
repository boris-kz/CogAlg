import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from collections import defaultdict
from .classes import Cgraph, CderG
from .filters import aves, ave_mL, ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd
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
        G_,i = [],0
        for PP in node_:  # convert select CPPs to Cgraphs:
            if PP.valt[fd] * (len(node_)-1) * (PP.rng+1) <= G_aves[fd] * PP.rdnt[fd]: continue
            derH,valt,rdnt = PP.derH,PP.valt,PP.rdnt
            G_ += [Cgraph( ptuple=PP.ptuple, derH=derH, valHt=[[valt[0]],[valt[1]]], rdnHt=[[rdnt[0]],[rdnt[1]]], L=PP.ptuple[-1], it=[i,i],
                           box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box), link_=PP.link_, node_t=PP.node_t )]
            i+=1  # G index in node_
        if G_:
            node_[:] = G_  # replace  PPs with Gs
            edge.valHt[0][0] = edge.valt[0]; edge.rdnHt[0][0] = edge.rdnt[0]  # copy
            agg_recursion(None, edge, node_, fd=0)  # edge.node_ = graph_t, micro and macro recursive


def agg_recursion(rroot, root, G_, fd):  # + fpar for agg_parP_? compositional agg|sub recursion in root graph, cluster G_

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,0,0,0,0
    link_ = defaultdict(list)  # lists of link per node, should be named node2link_ instead ?
    if fd:
        for link in get_link_(root.link_):
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
    GG_t = form_graph_t(root, [Mval,Dval],[Mrdn,Drdn], G_, link_, fd)  # eval sub+, feedback per graph
    # agg+ xcomp-> form_graph_t loop sub)agg+, vs. comp_slice sub+ loop-> eval-> xcomp
    for GG_ in GG_t:  # comp_G_ eval: ave_m * len*rng - fixed cost, root update in form_t:
        if root.valHt[0][-1] * (len(GG_)-1)*root.rng > G_aves[fd] * root.rdnHt[0][-1]:
            agg_recursion(rroot, root, GG_, fd=0)  # 1st xcomp in GG_
    G_[:] = GG_t

def form_graph_t(root, Valt,Rdnt, G_, link_, fd):  # form mgraphs and dgraphs of same-root nodes

    for G in G_: G.root = [None,None]  # replace with mcG_|dcG_ in segment_node_, replace with Cgraphs in sum2graph

    Gt_ = node_connect(G_, link_, fd)  # AKA Graph Convolution of Correlations
    graph_t = []
    for i in 0,1:
        if Valt[i] > ave * Rdnt[i]:  # else no clustering
            graph_t += [segment_node_(root, Gt_, link_, i, fd)]  # if fd: node-mediated Correlation Clustering; add alt_graphs?
        else:
            graph_t += [[]]  # or G_?
    # sub+, external to agg+ vs. internal in comp_slice sub+:
    for fd, graph_ in enumerate(graph_t):  # breadth-first for in-layer-only roots
        root.valHt[fd]+=[0]; root.rdnHt[fd]+=[1]  # remove if stays 0?
        for graph in graph_:
            node_ = graph.node_t[fd]  # flat here?
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

    iGt_ = []
    for i, G in enumerate(iG_):
        valt,rdnt,dect = [0,0],[0,0],[0,0]
        rimt = [defaultdict(list),defaultdict(list)]
        uprimt = [[],[]]
        for link in link_[G]:  # dict, all links containing G
            for i in 0,1:
                if link.valt[i] > G_aves[i] * link.rdnt[i]:  # skip negative, init with direct connectivity vals:
                    valt[i] += link.valt[i]; rdnt[i] += link.rdnt[i]; dect[i] += link.dect[i]
                    link.Vt[i][0] = link.Vt[i][1] = link.valt[i]
                    rimt[i][G] += [link]; uprimt[i] += [link]  # same
        iGt_ += [[G, rimt,valt,rdnt,dect,uprimt,[None,None]]]  # roott
    _Gt_ = copy(iGt_)  # for selective connectivity expansion, not affecting return iGt_
    '''
    Aggregate direct * indirect connectivity per node from indirect links via associated nodes, in multiple cycles. 
    Each cycle adds contributions of previous cycles to linked-nodes connectivity, propagated through the network.
    Math: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/node_connect.png 
    '''
    while True:  # eval same Gs,links, but with cross-accumulated node connectivity values, indirectly extending their range
        Gt_ = []
        DVt, Lent = [0,0],[0,0]  # _Gt_ updates per loop
        for Gt in _Gt_:
            G, rimt, valt,rdnt,dect,_uprimt,_ = Gt
            uprimt = [[],[]]  # for >ave updates
            dVt = [0,0]  # dRt?
            for i in 0,1:
                ave = G_aves[i]
                for link in _uprimt[i]:  # eval former >ave updates, +ve only?
                    _G, j = (link.G, 1) if link._G is G else (link._G, 0)
                    if _G not in iG_: continue  # outside root graph
                    _Gt = iGt_[_G.it[fd]]  # may represent Gt indirectly?
                    _G,_rimt,_valt,_rdnt,_dect,_,_ = _Gt
                    if _valt[i] < ave: continue  # _valt is updated after _linkV?
                    decay = link.dect[i]
                    dect[i] += link.dect[i]
                    rdnt[i] += _rdnt[i] * decay  # for segment_node_, else if fd==i: rimR += linkR and link.Rt?
                    linkV = _valt[i] * decay  # _node connect val * relative link val
                    dv = linkV - link.Vt[i][j]; valt[i] += dv; link.Vt[i][j] = linkV
                    if dv > ave * rdnt[i]:
                        dVt[i] += dv; uprimt[i]+= [link]
                L = len(uprimt[i]); Lent[i] += L
                if dVt[i] > ave * rdnt[i] * L:
                    _uprimt[i][:] = uprimt[i]  # pruned for next loop
                    DVt[i] += dVt[i]
                    if Gt not in Gt_: Gt_ += [Gt]
        if DVt[0] <= ave_Gm * Lent[0] and DVt[1] <= ave_Gd * Lent[1]:  # scale by rdn too?
            break
        _Gt_ = Gt_  # exclude weakly incremented Gts from next connectivity expansion loop

    return iGt_

def segment_node_(root, Gt_, fd, root_fd):  # eval rim links with summed surround vals for density-based clustering

    # graph += [node] if >ave (surround connectivity * relative value of link to any internal node)
    igraph_ = []; ave = G_aves[fd]

    for Gt in Gt_:
        G,rimt,valt,rdnt,dect, uprimt, *_ = Gt
        subH = [[],[0,0],[1,1],[0,0]]
        Link_ = []; A,S = [0,0],0
        for link in rimt[fd][G]:
            if link.valt[fd] > G_aves[fd] * link.rdnt[fd]:
                sum_subHv(subH, [link.subH,link.valt,link.rdnt,link.dect], base_rdn=1)
                Link_ += [link]; A[0] += link.A[0]; A[1] += link.A[1]; S += link.S
        grapht = [[Gt],Link_, copy(valt),copy(rdnt),copy(dect),A,S,subH, copy(Link_)]
        G.root[fd] = grapht; igraph_ += [grapht]

    _graph_ = igraph_; _tVal,_tRdn = 0,0
    while True:
        tVal,tRdn = 0,0  # loop totals
        graph_ = []
        while _graph_:  # extend graph Rim
            grapht = _graph_.pop()
            nodet_,Rim, Valt,Rdnt,Dect, A,S, subH,_upRim = grapht
            inVal,inRdn = 0,0  # in-graph: positive
            upRim = []
            for link in get_link_(Rim):  # unique links
                if link.G is nodet_[0][0]:
                    Gt = Gt_[link.G.it[root_fd]]; _Gt = Gt_[link._G.it[root_fd]] if Gt[0] in root.node_t[root_fd] else None
                else:
                    Gt = Gt_[link._G.it[root_fd]]; _Gt = Gt_[link.G.it[root_fd]] if Gt[0] in root.node_t[root_fd] else None
                if _Gt is None: continue  # not in root.node_
                if _Gt in nodet_: continue
                # node match * surround M|D match: of potential in-graph position?
                comb_val = link.valt[fd] + get_match(Gt[2][fd],_Gt[2][fd])
                comb_rdn = link.rdnt[fd] + (Gt[3][fd] + _Gt[3][fd]) / 2
                if comb_val > ave*comb_rdn:
                    # merge node.root:
                    _nodet_,_Rim,_Valt,_Rdnt,_Dect,_A,_S,_subH,__upRim = _Gt[0].root[fd]
                    if _Gt[0].root[fd] in grapht:  # grapht is not graphts?
                        grapht.remove(_Gt[0].root[fd])   # remove overlapping root
                    for _nodet in _nodet_: _nodet[0].root[fd] = grapht  # assign new merged root
                    sum_subHv(subH, _subH, base_rdn=1)
                    A[0] += _A[0]; A[1] += _A[1]; S += _S
                    list(set(upRim +__upRim))  # not sure, also need to exclude Rim?
                    for i in 0,1:
                        Valt[i] += _Valt[i]; Rdnt[i] += _Rdnt[i]; Dect[i] += _Dect[i]
                    inVal += _Valt[fd]; inRdn += _Rdnt[fd]
                    nodet_ += [__Gt for __Gt in _Gt[0].root[fd][0] if __Gt not in nodet_]
            tVal += inVal
            tRdn += inRdn  # signed?
            if len(Rim) * inVal > ave * inRdn:
                graph_ += [[nodet_,Rim, Valt,Rdnt,Dect,A,S, subH,upRim]]  # eval Rim for extension

        if len(graph_) * (tVal-_tVal) <= ave * (tRdn-_tRdn):  # even low-Val extension may be valuable if Rdn decreases?
            break
        _graph_ = graph_
        _tVal,_tRdn = tVal,_tRdn
    # -> Cgraphs if Val > ave * Rdn:
    return [sum2graph(root, graph, fd) for graph in graph_ if graph[2] > ave * graph[3]]


def sum2graph(root, grapht, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    Gt_,Rim,(Mval,Dval),(Mrdn,Drdn),(Mdec,Ddec), A,S, subH,Link_ = grapht

    graph = Cgraph(fd=fd, L=len(Gt_),link_=Link_,A=A,S=S)  # n nodes
    for link in get_link_(Link_):
        link.roott[fd]=graph
    for i, Gt in enumerate(Gt_):
        Gt[-1][fd] = root  # Gt: [G,rimt,valt,rdnt,dect,roott]
        G = Gt[0]
        G.it[fd] = i
        sum_box(graph.box, G.box)
        graph.node_t[fd] += [G]
        graph.connec_t[fd] += [Gt[1:]]  # node connectivity params: rim,valt,rdnt,dect,n
        if (Mval,Dval)[fd] > ave * (Mrdn,Drdn)[fd]:  # redundant to nodes, only link_ params are necessary
            graph.ptuple += G.ptuple
            sum_derH([graph.derH,[0,0],[1,1]], [G.derH,[0,0],[1,1]], base_rdn=1)
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
    # keep separate P ptuple and PP derH, empty derH in single-P G, + empty aggH in single-PP G
    # / P:
    mtuple, dtuple, Mtuple, Dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1, fagg=1)
    mval, dval = sum(mtuple), sum(abs(d) for d in dtuple)  # mval is signed, m=-min in comp x sign
    mrdn = dval>mval; drdn = dval<=mval
    dect = [0,0]
    for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):  # the prior zipping elements are wrong
        for i, (par, max, ave) in enumerate(zip(ptuple, Ptuple, aves)):  # compute link decay coef: par/ max(self/same)
            if fd: dect[fd] += par/max if max else 1
            else:  dect[fd] += (par+ave)/ max if max else 1
    derLay0 = [[mtuple,dtuple],[mval,dval],[mrdn,drdn],[dect[0]/6, dect[1]/6]]  # ave of 6 params
    Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn; Mdec+=dect[0]/6; Ddec+=dect[1]/6
    # / PP:
    _derH,derH = _G.derH,G.derH
    if _derH and derH:  # empty in single-node Gs
        dderH, [mval,dval],[mrdn,drdn],[mdec,ddec]  = comp_derHv(_derH, derH, rn=1)
        Mval+=dval; Dval+=mval; Mdec=(Mdec+mdec)/2; Ddec=(Ddec+ddec)/2
        Mrdn+= mrdn+ dval>mval; Drdn+= drdn+ dval<=mval
    else: dderH = []
    derH = [[derLay0]+dderH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]]  # appendleft derLay0 from comp_ptuple
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec])
    SubH = [der_ext, derH]  # two init layers of SubH, higher layers added by comp_aggH:
    # / G:
    if fd:  # else no aggH yet?
        subH, valt,rdnt,dect = comp_aggHv(_G.aggH, G.aggH, rn=1)
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
        Mdec = (Mdec+dect[0])/2; Ddec = (Ddec+dect[1])/2
        link.subH = SubH+subH  # append higher subLayers: list of der_ext | derH s
        link.valt = [Mval,Dval]; link.rdnt = [Mrdn,Drdn]; link.dect = [Mdec,Ddec]  # complete proto-link
        # dict key:G,vals:derGs
        link_[G] += [link]
        link_[_G]+= [link]
    elif Mval > ave_Gm or Dval > ave_Gd:  # or sum?
        link.subH = SubH
        link.valt = [Mval,Dval]; link.rdnt = [Mrdn,Drdn]; link.dect = [Mdec,Ddec] # complete proto-link
        link_[G] += [link]
        link_[_G]+= [link]

    return Mval,Dval, Mrdn,Drdn, Mdec,Ddec

# draft:
def comp_aggHv(_aggH, aggH, rn):  # no separate ext

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    SubH = []; S = min(len(_aggH), len(aggH))

    for _lev, lev in zip(_aggH, aggH):  # compare common subHs, if lower-der match?
        if _lev and lev:
            dsubH, valt,rdnt,dect = comp_subHv(_lev[0], lev[0], rn)  # skip valt,rdnt,dect
            SubH += dsubH  # flatten to keep as subH
            Mdec += dect[0]; Ddec += dect[1]
            mval,dval = valt; Mval += mval; Dval += dval
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + mval <= dval
    if S: Mdec/= S; Ddec /= S
    return SubH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]

def comp_subHv(_subH, subH, rn):

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    dsubH =[]; S = min(len(_subH), len(subH))

    for _lay, lay in zip(_subH, subH):  # compare common lower layer|sublayer derHs
        # if lower-layers match: Mval > ave * Mrdn?
        if _lay[0] and isinstance(_lay[0][0],list):
            dderH, valt,rdnt,dect = comp_derHv(_lay[0], lay[0], rn)
            dsubH += [[dderH, valt,rdnt,dect]]  # flat
            Mdec += dect[0]; Ddec += dect[1]
            mval,dval = valt; Mval += mval; Dval += dval
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + dval <= mval
        else:  # _lay[0][0] is L, comp dext:
            dsubH += [comp_ext(_lay[1],lay[1],[Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec])]
    if S: Mdec/= S; Ddec /= S  # normalize
    return dsubH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]  # new layer,= 1/2 combined derH


def comp_derHv(_derH, derH, rn):  # derH is a list of der layers or sub-layers, each is ptuple_tv

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    dderH =[]; S = min(len(_derH), len(derH))

    for _lay, lay in zip(_derH, derH):  # compare common lower der layers | sublayers in derHs, if lower-layers match?
        # comp dtuples, eval mtuples
        if isinstance(lay[0][0], list):  _dlay = _lay[0][1]; dlay = lay[0][1]
        else:                            _dlay = _lay[1]; dlay = lay[1]  # converted derlay with no valt and rdnt
        mtuple, dtuple, Mtuple, Dtuple = comp_dtuple(_dlay, dlay, rn, fagg=1)
        # sum params:
        mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
        mrdn = dval > mval; drdn = dval < mval
        dect = [0,0]
        for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):  # the zipping sequence is wrong
            for (par, max, ave) in zip(ptuple, Ptuple, aves):  # different ave for comp_dtuple
                if fd: dect[fd] += par/max if max else 1
                else:  dect[fd] += (par+ave)/(max) if max else 1

        dderH += [[[mtuple,dtuple],[mval,dval],[mrdn,drdn],[dect[0]/6,dect[1]/6]]]
        Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn; Mdec+=dect[0]/6; Ddec+=dect[1]/6
    if S: Mdec /= S; Ddec /= S  # normalize when non zero layer
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
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i]+dect[i])/2
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
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i] + dect[i])/2
    # why fneg*i in sum_dertuple?
    DerH[:] = [
        [ [sum_dertuple(Dertuple,dertuple, fneg*i) for i,(Dertuple,dertuple) in enumerate(zip(Tuplet,tuplet))],
          [V+v for V,v in zip(Valt,valt)], [R+r+base_rdn for R,r in zip(Rdnt,rdnt)], [(D+d)/2 for D,d in zip(Dect,dect)]
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
    mL = get_match(_L,L) - ave_mL
    mS = get_match(_S,S) - ave_mL

    m = mL+mS+mA; d = abs(dL)+ abs(dS)+ adA
    Valt[0] += m; Valt[1] += d
    Rdnt[0] += d>m; Rdnt[1] += d<=m
    _aL = abs(_L); aL = abs(L)
    _aS = abs(_S); aS = abs(S)
    Dect[0] += (mL/ max(aL,_aL) + mS/ max(aS,_aS) if aS or _aS else 1 + (mA / max_mA) if max_mA else 1) /3  # ave dec of 3 vals
    Dect[1] += (dL/ (_aL+aL) + dS / max(_aS+aS) if aS or _aS else 1 + (dA / max_dA) if max_mA else 1) /3

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
        rroot = root.connec_t[fd][0][-1][fd]  # get root from 1st connect [rim,val,rdn,roott]
        fd = root.fd  # current node_ fd
        fback_ = rroot.fback_t[fd]
        fback_ += [[AggH, ValHt, RdnHt, DecHt]]
        if fback_ and (len(fback_) == len(rroot.node_t)):  # flat, all rroot nodes terminated and fed back
            # getting cyclic rroot here not sure why it can happen, need to check further
            feedback(rroot, fd)  # sum2graph adds aggH per rng, feedback adds deeper sub+ layers

def get_link_(linkmap):
    return list(set([link for node, link_ in linkmap.items() for link in link_]))