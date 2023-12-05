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
            G_ += [Cgraph(ptuple=PP.ptuple, derH=derH, valt=copy(valt), rdnt=copy(rdnt), L=PP.ptuple[-1],
                          box=PP.box, link_=PP.link_, node_tH=[PP.node_t])]
            i += 1  # G index in node_
        if G_:
            node_[:] = G_  # replace  PPs with Gs
            agg_recursion(None, edge, node_, fd=0)  # edge.node_ = graph_t, micro and macro recursive


def agg_recursion(rroot, root, G_, fd):  # + fpar for agg_parP_? compositional agg|sub recursion in root graph, cluster G_

    Et = [[0,0],[0,0],[0,0]]  # eValt, eRdnt, eDect
    link_ = []
    # for G in G_: G.it = [None, None]  # reassign (this is not needed now)
    if fd:  # der+
        for link in root.link_:  # reform links
            if link.valt[1] < G_aves[1]*link.rdnt[1]: continue  # maybe weak after rdn incr?
            comp_G(link._G,link.G,link, link_, Et)
    else:   # rng+
        for i, _node in enumerate(G_):  # form new link_ from original node_
            for node in G_[i+1:]:
                dy = _node.box.cy - node.box.cy; dx = _node.box.cx - node.box.cx
                distance = np.hypot(dy, dx)  # distance between node centers, init ave_rng = 3:
                if distance < 3 * ((_node.valt[fd] + node.valt[fd]) / ave * (_node.rdnt[fd] + node.rdnt[fd])):
                    comp_G(_node, node, CderG(_G=_node,G=node), link_, Et)

    GG_t = form_graph_t(root, G_, Et, fd)  # eval sub+, feedback per graph
    # agg+ xcomp-> form_graph_t loop sub)agg+, vs. comp_slice sub+ loop-> eval-> xcomp
    for GG_ in GG_t:  # comp_G_ eval: ave_m * len*rng - fixed cost, root update in form_t:
        if root.valt[0] * (len(GG_)-1)*root.rng > G_aves[fd] * root.rdnt[0]:
            agg_recursion(rroot, root, GG_, fd=0)  # 1st xcomp in GG_

    if isinstance(root, Cgraph): root.node_tH += [GG_t]  # Cgraph
    else: root.node_t[fd][:] = GG_t   # Cedge


def comp_G(_G, G, link, link_, Et):

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0, 1,1, 0,0
    # keep separate P ptuple and PP derH, empty derH in single-P G, + empty aggH in single-PP G
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
    fadd = 0
    if link.subH:  # old link, not empty aggH
        subH, valt,rdnt,dect = comp_aggHv(_G.aggH, G.aggH, rn=1)
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
        Mdec = (Mdec+dect[0])/2; Ddec = (Ddec+dect[1])/2
        link.subH = SubH+subH  # append higher subLayers: list of der_ext | derH s
        if Mval > ave_Gm or Dval > ave_Gd:
            fadd = 1
    elif Mval > ave_Gm or Dval > ave_Gd:  # new link
        link.subH = SubH
        fadd = 1
    if fadd:
        for typ, (Part, lpart, Lpart, part) in enumerate(zip(
            Et,[link.valt,link.rdnt,link.dect],[link.Vt,link.Rt,link.Dt],[[Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]])):
            for fd, par in enumerate(part):
                Part[fd]+=par; lpart[fd]=par; Lpart[fd]=par
                # link.Vt,link.Rt,link.Dt and node external params to initialize node_connect:
                for node in link._G, link.G:
                    if link.valt[fd] < G_aves[fd] * link.rdnt[fd]: continue  # exclude neg links
                    if typ: continue  # once per link
                    node.rim_tH[-1][fd] += [link]
                    node.Rim_tH[-1][fd] += [link]
                    sum_subHv([node.esubH, node.evalt,node.erdnt,node.edect], [link.subH, link.valt,link.rdnt,link.dect],link.rdnt[fd])
        link_ += [link]  # in some graph


def form_graph_t(root, G_, Et, fd):  # form mgraphs and dgraphs of same-root nodes

    node_connect(G_)  # AKA Graph Convolution of Correlations
    graph_t = [[],[]]
    for i in 0,1:
        if Et[0][i] > ave * Et[1][i]:  # eValt > ave * eRdnt, else no clustering
            graph_t[i] = segment_node_(root,G_, i,fd) # if fd: node-mediated Correlation Clustering
            # add alt_graphs?
    for fd, graph_ in enumerate(graph_t):  # breadth-first for in-layer-only roots
        # root.valt[fd]+=[0]; root.rdnt[fd]+=[1]  # remove if stays 0? (why we need this?)
        for graph in graph_:
            # sub+ is external to agg+ vs. internal in comp_slice sub+,
            Val,Rdn = 0,0
            for subH in graph.aggH[1:]:  # eval by sum of val,rdn of top subLays in lower aggLevs:
                if not subH[0]: continue  # empty subH when there's no rim because rim may empty in one of the fork
                Val+=subH[0][-1][1][fd]; Rdn+=subH[0][-1][2][fd]  # subH is [derH, valt,rdnt, dect]
            if Val * (len(graph.node_tH[-1])-1)*root.rng > G_aves[fd] * Rdn:
                for G in graph.node_tH[-1]:  # still node_
                    G.rim_tH += [[[],[]]]; G.Rim_tH += [[[],[]]]  # add layer
                agg_recursion(root, graph, graph.node_tH[-1], fd)  # for flat node_
            else:  # feedback after sub+
                root.fback_t[fd] += [[graph.aggH, graph.valt, graph.rdnt, graph.dect]]
                root.valt[fd] += graph.valt[fd]  # merge forks in root fork
                root.rdnt[fd] += graph.rdnt[fd]
                root.dect[fd] += graph.dect[fd]
        if root.fback_t and root.fback_t[fd]:  # recursive feedback after all G_ sub+
            feedback(root, fd)  # update root.root.. aggH, valHt,rdnHt

    return graph_t  # root.node_t'node_ -> node_t: incr nested with each agg+?

def node_connect(iG_):  # node connectivity = sum surround link vals, incr.mediated: Graph Convolution of Correlations
    '''
    Aggregate direct * indirect connectivity per node from indirect links via associated nodes, in multiple cycles.
    Each cycle adds contributions of previous cycles to linked-nodes connectivity, propagated through the network.
    Math: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/node_connect.png '''
    _G_ = copy(iG_)  # for selective connectivity expansion, not affecting return iGt_

    while True:  # eval same Gs,links, but with cross-accumulated node connectivity values, indirectly extending their range
        G_ = []  # DVt, Lent = [0,0],[0,0]  # _G_ updates per loop, for more selective eval?
        for G in _G_:
            uprimt = [[],[]]  # >ave updates of direct links
            for i in 0,1:
                valt,rdnt,dect = G.esubH[-1][1:]; val,rdn,dec = valt[i],rdnt[i],dect[i]  # last layer only
                ave = G_aves[i]
                for link in G.Rim_tH[-1][i]:
                    lval,lrdn,ldec = link.valt[i], link.rdnt[i], link.dect[i]
                    _G = link._G if link.G is G else link.G
                    _valt,_rdnt,_dect = G.esubH[-1][1:]; _val,_rdn,_dec = valt[i],rdnt[i],dect[i]
                    # for segment_node_:
                    linkV = ldec * (val+_val); dv = linkV-link.Vt[i]; link.Vt[i] = linkV
                    linkR = ldec * (rdn+_rdn); dr = linkR-link.Rt[i]; link.Rt[i] = linkR
                    linkD = ldec * (dec+_dec); dd = linkD-link.Dt[i]; link.Dt[i] = linkD
                    if dv > ave * dr:
                        uprimt[i] += [link]  # dVt[i] += dv; L = len(uprimt[i]); Lent[i] += L for more selective eval
                    if linkV > ave * linkR:
                        G.evalt[i] += dv; G.erdnt[i] += dr; G.edect[i] += dd
            if any(uprimt):  # pruned for next loop
                G.Rim_tH[-1] = uprimt

        if G_: _G_ = G_  # exclude weakly incremented Gs from next connectivity expansion loop
        else:  break


def segment_node_(root, G_, fd, root_fd):  # eval rim links with summed surround vals for density-based clustering

    # graph += [node] if >ave (surround connectivity * relative value of link to any internal node)
    igraph_ = []; ave = G_aves[fd]
    for i, G in enumerate(G_):
        subH = [[],[0,0],[1,1],[0,0]]
        A,S = [0,0], 0
        Link_ = []
        for link in G.rim_tH[-1][fd]:
            sum_subHv(subH, [link.subH,link.valt,link.rdnt,link.dect], base_rdn=1)
            A[0] += link.A[0]; A[1] += link.A[1]; S += link.S
            Link_ += [link]
        # grapht int += node int+ext:
        cvalt = [val+eval for val+eval in zip(G.valt,G.evalt)]
        crdnt = [rdn+erdn for rdn+erdn in zip(G.rdnt,G.erdnt)]
        cdect = [dec+edec for dec+edec in zip(G.dect,G.edect)]
        grapht = [[G],Link_, cvalt,crdnt,cdect, A,S, copy(G.subH),copy(Link_)]
        G.roott[fd] = grapht  # for feedback
        igraph_ += [grapht]
    _graph_ = igraph_
    while True:
        graph_ = []
        for grapht in _graph_:  # extend grapht Rim with +ve in-root links
            G_, Rim, Valt, Rdnt, Dect, A,S, subH,_upRim = grapht
            inVal, inRdn = 0,0  # new in-graph: positive
            upRim = []
            for link in Rim:  # unique links
                if link.G in G_:
                    G = link.G; _G = link._G
                else:
                    G = link._G; _G = link.G
                if _G in G_: continue
                # node match * surround M|D match: of potential in-graph position?
                comb_val = link.valt[fd] + get_match(G.evalt[fd],_G.evalt[fd])
                comb_rdn = link.rdnt[fd] + (G.erdnt[fd] + _G.erdnt[fd]) / 2
                if comb_val > ave * comb_rdn:
                    # merge node.root:
                    _G_,_Rim,_Valt,_Rdnt,_Dect,_A,_S,_subH,__upRim = _G.roott[fd]
                    for __G in _G_:  # assign merged root:
                        __G.roott[fd] = grapht
                    sum_subHv([subH, Valt, Rdnt, Dect], [_subH,_Valt,_Rdnt,_Dect], base_rdn=1)
                    A[0] += _A[0]; A[1] += _A[1]; S += _S
                    upRim = list(set([rim for rim in upRim +__upRim if rim not in Rim]))
                    inVal += _Valt[fd]
                    inRdn += _Rdnt[fd]
                    G_ += [__G for __G in _G_ if __G not in G_]
            if len(Rim) * inVal > ave * inRdn:
                graph_ += [[G_,Rim, Valt,Rdnt,Dect, A,S, subH,upRim]]  # eval Rim for extension

        if graph_: _graph_ = graph_
        else: break
    # -> Cgraphs if Val > ave * Rdn:
    return [sum2graph(root, graph, fd) for graph in igraph_ if graph[2][fd]+graph[5][fd] > ave * (graph[3][fd] + graph[6][fd])]


def sum2graph(root, grapht, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    G_,Rim,valt,rdnt,dect, A,S, subH, Link_ = grapht
    graph = Cgraph(fd=fd,L=len(G_),link_=Link_,A=A,S=S, valt=valt,rdnt=rdnt,dect=dect)
    graph.roott[fd] = root
    for link in Link_: link.roott[fd]=graph
    node_ = []  # flat here
    # not updated:
    for i, G in enumerate(G_):
        graph.box += G.box
        node_ += [G]
        if (valt[fd]+evalt[fd]) > ave * (rdnt[fd]+erdnt[fd]):  # redundant to nodes, only link_ params are necessary
            graph.ptuple += G.ptuple
            sum_derH([graph.derH,[0,0],[1,1]], [G.derH,[0,0],[1,1]], base_rdn=1)
            sum_aggHv(graph.aggH, G.aggH, base_rdn=1)
            sum_ts(graph.valt, graph.rdnt, graph.dect, G.valt, G.rdnt, G.dect)
            sum_subHv([graph.esubH,graph.evalt,graph.erdnt,graph.edect],[G.esubH,G.evalt,G.erdnt,G.edect], base_rdn=1)  # or concatenate them?
    # add derLay:
    graph.aggH += [subH]
    graph.node_tH += [node_]

    return graph

def sum_ts(Valt,Rdnt,Dect, valt,rdnt,dect):

    for fd in 0,1:
        Valt[fd] += valt[fd]
        Rdnt[fd] += rdnt[fd]
        Dect[fd] += dect[fd]


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


def feedback(root, fd):  # called from form_graph_, append new der layers to root

    AggH, Valt, Rdnt, Dect = deepcopy(root.fback_t[fd].pop(0))  # init with 1st tuple
    while root.fback_t[fd]:
        aggH, valt, rdnt, dect = root.fback_t[fd].pop(0)
        sum_aggHv(AggH, aggH, base_rdn=0)
        sum_ts(Valt, Rdnt, Dect, valt, rdnt,rdnt)
    sum_aggHv(root.aggH,AggH, base_rdn=0)
    sum_ts(root.valt,root.rdnt,root.dect, Valt,Rdnt,Dect)  # both forks sum in same root

    if isinstance(root, Cgraph):  # root is not CEdge, which has no roots

        rroot = root.roott[fd]  # nodec_H is a flat list of nodec
        if rroot:
            fd = root.fd  # node_ fd
            fback_ = rroot.fback_t[fd]
            fback_ += [[AggH, Valt, Rdnt, Dect]]
            if isinstance(rroot, Cgraph):  # Cgraph
                if isinstance(rroot.node_tH[-1][0], list):
                    len_node_ = len(rroot.node_tH[-1][fd])  # # node_tH[-1] is updated with node_t
                else:
                    len_node_ = len(rroot.node_tH[-1])  # node_tH[-1] is still flat
            else:                         len_node_ = len(rroot.node_t[fd])   # Cedge
            if fback_ and (len(fback_) == len_node_):  # flat, all rroot nodes terminated and fed back
                # getting cyclic rroot here not sure why it can happen, need to check further
                feedback(rroot, fd)  # sum2graph adds aggH per rng, feedback adds deeper sub+ layers