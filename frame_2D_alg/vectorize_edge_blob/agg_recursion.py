import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from collections import deque, defaultdict
from .classes import Cgraph, CderG, CPP
from .filters import ave_L, ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd
from .slice_edge import slice_edge, comp_angle
from .comp_slice import comp_P_, comp_ptuple, sum_ptuple, sum_derH, comp_derH, matchF
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

    edge = slice_edge(blob, verbose)  # lateral kernel cross-comp -> P clustering
    comp_P_(edge)  # vertical, lateral-overlap P cross-comp -> PP clustering
    # PP cross-comp -> discontinuous graph clustering:
    for fd in 0,1:
        node_ = edge.node_t[fd]  # always PP_t
        if edge.valt[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.rdnt[fd]:
            G_= []
            for PP in node_:  # convert CPPs to Cgraphs:
                derH, valt, rdnt = PP.derH, PP.valt, PP.rdnt  # init aggH is empty:
                G_ += [Cgraph( ptuple=PP.ptuple, derH=[derH,valt,rdnt], valHt=[[valt[0]],[valt[1]]], rdnHt=[[rdnt[0]],[rdnt[1]]],
                               L=PP.ptuple[-1], box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box))]
            node_ = G_
            edge.valHt[0][0] = edge.valt[0]; edge.rdnHt[0][0] = edge.rdnt[0]  # copy
            agg_recursion(None, edge, node_, fd=0)  # edge.node_t = graph_t, micro and macro recursive


def agg_recursion(rroot, root, G_, fd):  # compositional agg+|sub+ recursion in root graph, clustering G_

    comp_G_(G_, fd)  # rng|der cross-comp all Gs, nD array? form link_H per G
    root.valHt[fd] += [0]; root.rdnHt[fd] += [1]  # sum in form_graph_t feedback

    GG_t = form_graph_t(root, G_)  # eval sub+ and feedback per graph
    # agg+ xcomp-> form_graph_t loop sub)agg+, vs. comp_slice:
    # sub+ loop-> eval-> xcomp
    for GG_ in GG_t:  # comp_G_ eval: ave_m * len*rng - fixed cost, root update in form_t:
        if root.valHt[0][-1] * (len(GG_)-1)*root.rng > G_aves[fd] * root.rdnHt[0][-1]:
            agg_recursion(rroot, root, GG_, fd=0)  # 1st xcomp in GG_

    G_[:] = GG_t

def form_graph_t(root, G_):  # form mgraphs and dgraphs of same-root nodes

    graph_t = []
    for G in G_: G.root = [None,None]  # replace with mcG_|dcG_ in segment_node_, replace with Cgraphs in sum2graph
    for fd in 0,1:
        Gt_ = eval_node_connectivity(G_, fd)  # sum surround link values @ incr rng,decay
        graph_t += [segment_node_(root, Gt_, fd)]  # add alt_graphs?

    # eval sub+, not in segment_node_: full roott must be replaced per node within recursion
    for fd, graph_ in enumerate(graph_t): # breadth-first for in-layer-only roots
        root.valHt[fd]+=[0]; root.rdnHt[fd]+=[1]  # remove if stays 0?

        for graph in graph_:  # external to agg+ vs internal in comp_slice sub+
            node_ = graph.node_t  # init flat
            if sum(graph.valHt[fd]) * (len(node_)-1)*root.rng > G_aves[fd] * sum(graph.rdnHt[fd]):  # eval fd comp_G_ in sub+
                agg_recursion(root, graph, node_, fd)  # replace node_ with node_t, recursive
            else:  # feedback after graph sub+:
                root.fback_t[fd] += [[deepcopy(graph.aggH), deepcopy(graph.valHt), deepcopy(graph.rdnHt)]]  # merge forks in root fork
                root.valHt[fd][-1] += graph.valHt[fd][-1]  # last layer, or all new layers via feedback?
                root.rdnHt[fd][-1] += graph.rdnHt[fd][-1]
            i = sum(graph.valHt[0]) > sum(graph.valHt[1])
            root.rdnHt[i][-1] += 1  # add fork rdn to last layer, representing all layers after feedback

        if root.fback_t and root.fback_t[fd]:  # recursive feedback after all G_ sub+
            feedback(root, fd)  # update root.root.. aggH, valHt,rdnHt

    return graph_t  # root.node_t'node_ -> node_t: incr nested with each agg+?


def eval_node_connectivity(node_, fd):  # sum surrounding link values to define connected nodes

    ave = G_aves[fd]
    Gt_ = []
    for i,G in enumerate(node_):
        G.it[fd] = i  # used here, in segment_node_
        Gt_ += [[G,0,1]]  # init surround val,rdn
    # iterative eval rng expansion by summing decayed surround node Vals, prune links and cluster per rng?:
    while True:
        DVal = 0
        for i, (_G,_Val,_Rdn) in enumerate(Gt_):
            Val,Rdn = 0,0  # updated surround
            for link in _G.link_H[-1]: # prune links by +Val-ave?
                if link.valt[fd] < ave * link.rdnt[fd]: continue  # skip negative links
                G = link.G if link._G is _G else link._G
                Gt = Gt_[G.it[fd]]
                Gval = Gt[1]; Grdn = Gt[2]; decay = link.dect[fd]
                Val += Gval * decay
                Rdn += Grdn * decay  # link decay coef: m|d / max, base self/same
            Gt_[i][1] = Val
            Gt_[i][2] = Rdn  # unilateral update, computed separately for _G
            DVal += abs(_Val-Val)  # _Val update / surround extension
        if DVal < ave:  # also if low Val?
            break

    return Gt_

def segment_node_(root, Gt_, fd):

    link_map = defaultdict(list)  # make default for root.node_t?
    ave = G_aves[fd]
    for G,_,_ in Gt_:
        for derG in G.link_H[-1]:
            if derG.valt[fd] > ave * derG.rdnt[fd]:  # or link val += node Val: prune +ve links to low Vals?
                link_map[G] += [derG._G]  # keys:Gs, vals: linked _G_s
                link_map[derG._G] += [G]
    graph_ = []
    for iG, iVal, iRdn in Gt_:  # initialize proto-graphs with each node, eval links to add other nodes
        if iG.root[fd]: continue
        tVal = sum(iG.valHt[fd]) * iG.decHt[fd][-1] + iVal  # graph totals, Val,Rdn *= decay for mediated links
        tRdn = sum(iG.rdnHt[fd]) * iG.decHt[fd][-1] + iRdn
        cG_ = [iG]; iG.root[fd] = cG_  # clustered Gs
        perimeter = link_map[iG]  # recycle in breadth-first search

        while perimeter:  # search links outwards recursively to form overlapping graphs:
            _G = perimeter.pop(0)
            if _G in cG_: continue
            Gt = Gt_[_G.it[fd]]; _Val,_Rdn = Gt[1],Gt[2]
            for link in _G.link_H[-1]:
                G = link.G if link._G is _G else link._G
                if G in cG_: continue    # circular link
                Gt = Gt_[G.it[fd]]; Val, Rdn = Gt[1], Gt[2]
                val, rdn, decay = link.valt[fd], link.rdnt[fd], link.dect[fd]
                rVal=(_Val+Val)*decay; rRdn=(_Rdn+Rdn)*decay  # * link decay coef -> relative Val,Rdn
                if rVal > ave * rRdn:
                    # add val and rdn from link too?
                    tVal += sum(G.valHt[fd]) * G.decHt[fd][-1] + Val  # internal+external vals
                    tRdn += sum(G.rdnHt[fd]) * G.decHt[fd][-1] + Rdn
                    cG_ += [G]; G.root[fd] = cG_
                    perimeter += [G for G in link_map[G] if G not in perimeter]  # extend perimeter, skip if added by another _G
        if tVal > ave * tRdn:
            graph_ += [sum2graph(root, cG_, fd)]  # convert to Cgraphs

    return graph_

def sum2graph(root, cG_, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    graph = Cgraph(root=root, fd=fd, L=len(cG_))  # n nodes, transplant both node roots
    SubH = []; Mdec,Ddec, Mval,Dval, Mrdn,Drdn = 0,0, 0,0, 0,0
    Link_= []
    for G in cG_:
        # sum nodes in graph:
        sum_box(graph.box, G.box)
        sum_ptuple(graph.ptuple, G.ptuple)
        sum_derH(graph.derH, G.derH, base_rdn=1)
        sum_aggH(graph.aggH, G.aggH, base_rdn=1)
        sum_Hts(graph.valHt, graph.rdnHt, graph.decHt, G.valHt, G.rdnHt, G.decHt)
        link_ = G.link_H[-1]
        subH = []; mval,dval, mrdn,drdn, mdec,ddec = 0,0, 0,0, 0,0
        for derG in link_:
            if derG.valt[fd] > G_aves[fd] * derG.rdnt[fd]:  # sum positive links only:
                (_mval,_dval),(_mrdn,_drdn),(_mdec,_ddec) = derG.valt, derG.rdnt, derG.dect
                if derG not in Link_:
                    sum_subH(SubH, derG.subH, base_rdn=1)  # new aggLev, not from nodes: links overlap
                    Mval+=_mval; Dval+=_dval; Mrdn+=_mrdn; Drdn+=_drdn; Mdec+=_mdec; Ddec+=_ddec
                    graph.A[0] += derG.A[0]; graph.A[1] += derG.A[1]; graph.S += derG.S
                    Link_ += [derG]
                mval+=_mval; dval+=_dval; mrdn+=_mrdn; drdn+=_drdn; mdec+=_mdec; ddec+=_ddec
                sum_subH(subH, derG.subH, base_rdn=1, fneg = G is derG.G)  # fneg: reverse link sign
                sum_box(G.box, derG.G.box if derG._G is G else derG._G.box)
        # add params of external G links:
        if subH: G.aggH += [subH]
        G.valHt[0]+=[mval]; G.valHt[1]+=[dval]; G.rdnHt[0]+=[mrdn]; G.rdnHt[1]+=[drdn]
        L = len(link_); G.decHt[0]+=[mdec/L]; G.decHt[1]+=[ddec/L]
        G.root[fd] = graph  # replace cG_
        graph.node_t += [G]  # converted to node_t by feedback
    # add layer from links:
    graph.valHt[0]+=[Mval]; graph.valHt[1]+=[Dval]; graph.rdnHt[0]+=[Mrdn]; graph.rdnHt[1]+=[Drdn]
    L = len(cG_)
    for mdec,ddec in zip(graph.decHt[0],graph.decHt[1]):
        mdec/=L; ddec/=L
    L = len(Link_); graph.decHt[0]+=[Mdec/L]; graph.decHt[1]+=[Ddec/L]

    return graph

def sum_Hts(ValHt, RdnHt, DecHt, valHt, rdnHt, decHt):
    for ValH,valH, RdnH,rdnH, DecH,decH in zip(ValHt,RdnHt, valHt,rdnHt, DecHt,decHt):

        ValH[:] = [Val + val for Val, val in zip_longest(ValH, valH, fillvalue=0)]
        DecH[:] = [Dec + dec for Dec, dec in zip_longest(DecH, decH, fillvalue=0)]
        RdnH[:] = [max(1,Rdn+rdn) for Rdn,rdn in zip_longest(RdnH, rdnH, fillvalue=0)]

'''
derH: [[tuplet, valt, rdnt]]: default input from PP, rng+|der+, sum min len?
subH: [derH_t]: m,d derH, m,d ext added by agg+ as 1st tuplet
aggH: [subH_t]: composition levels, ext per G, 
'''

def comp_G_(G_, fd=0, oG_=None, fin=1):  # cross-comp in G_ if fin, else comp between G_ and other_G_, for comp_node_

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
            comp_G(link_, link, fd)
        G.link_H[-1] = link_
        '''
        same comp for cis and alt components?
        for _cG, cG in ((_G, G), (_G.alt_Graph, G.alt_Graph)):
            if _cG and cG:  # alt Gs maybe empty
                comp_G(_cG, cG, fd)  # form new layer of links:
        combine cis,alt in aggH: alt represents node isolation?
        comp alts,val,rdn? cluster per var set if recurring across root: type eval if root M|D? '''

# draft
def comp_G(link_, link, fd):

    _G, G = link._G, link.G
    Mdec,Ddec = 0,0  # max possible summed m|d, to compute relative summed m|d: V/maxV, link mediation coef
    Mval,Dval = 0,0; Mrdn,Drdn = 1,1

    # / P:
    mtuple, dtuple, Mtuple, Dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1, fagg=1)
    mval, dval = sum(mtuple), sum(abs(d) for d in dtuple)  # mval is signed, m=-min in comp x sign
    mrdn = dval>mval; drdn = dval<=mval
    derLay0 = [[mtuple,dtuple], [mval,dval], [mrdn,drdn]]
    L = len(mtuple)
    Mdec += sum([par/max for par,max in zip(mtuple,Mtuple)]) / L
    Ddec += sum([par/max for par,max in zip(dtuple,Dtuple)]) / L
    Mval+=mval; Dval+=dval; Mrdn += mrdn; Drdn += drdn

    # / PP:
    dderH, valt, rdnt, dect = comp_derH(_G.derH[0], G.derH[0], rn=1, fagg=1)
    mdec,ddec = dect; Mdec = (Mdec+mdec)/2; Ddec = (Ddec+ddec)/2  # averages
    mval,dval = valt; Mval+=dval; Dval+=mval
    Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval

    derH = [[derLay0]+dderH, [Mval,Dval], [Mrdn,Drdn], [Mdec, Ddec]]  # appendleft derLay0 from comp_ptuple
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval],[Mrdn,Drdn], [Mdec,Ddec])
    SubH = [der_ext, derH]  # two init layers of SubH, higher layers added by comp_aggH:
    # / G:
    if fd:  # else no aggH yet?
        subH, valt, rdnt, dect = comp_aggH(_G.aggH, G.aggH, rn=1)
        SubH += subH  # append higher subLayers: list of der_ext | derH s
        mdec,ddec = dect; link.dect = [(Mdec+mdec)/2,(Ddec+ddec)/2]  # averages
        mval,dval = valt; Mval+=mval; Dval+=dval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
        link_ += [link]

    elif Mval > ave_Gm or Dval > ave_Gd:  # or sum?
        link.subH = SubH; link.dect = [Mdec,Ddec]; link.valt = [Mval,Dval]; link.rdnt = [Mrdn,Drdn]  # complete proto-link
        link_ += [link]

# draft:
def comp_aggH(_aggH, aggH, rn):  # no separate ext
    SubH = []
    Mdec,Ddec, Mval,Dval, Mrdn,Drdn = 0,0,0,0,1,1

    for _lev, lev in zip_longest(_aggH, aggH, fillvalue=[]):  # compare common lower layer|sublayer derHs
        if _lev and lev:  # also if lower-layers match: Mval > ave * Mrdn?
            # compare dsubH only:
            dsubH, valt,rdnt,dect = comp_subH(_lev[0], lev[0], rn)
            SubH += dsubH  # flatten to keep subH
            mdec,ddec = dect; Mdec += mdec; Ddec += ddec
            mval,dval = valt; Mval += mval; Dval += dval
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + mval <= dval
    L = len(aggH)
    return SubH, [Mval,Dval],[Mrdn,Drdn],[Mdec/L,Ddec/L]

def comp_subH(_subH, subH, rn):
    DerH = []
    Mdec,Ddec, Mval,Dval, Mrdn,Drdn = 0,0,0,0,1,1

    for _lay, lay in zip_longest(_subH, subH, fillvalue=[]):  # compare common lower layer|sublayer derHs
        if _lay and lay:  # also if lower-layers match: Mval > ave * Mrdn?
            if _lay[0] and isinstance(_lay[0][0],list):  # _lay[0][0] is derHt

                dderH, valt, rdnt, dect = comp_derH(_lay[0], lay[0], rn, fagg=1)
                DerH += [[dderH, valt, rdnt, dect]]  # flat derH
                mdec,ddec = dect; Mdec += mdec; Ddec += ddec
                mval,dval = valt; Mval += mval; Dval += dval
                Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + dval <= mval
            else:  # _lay[0][0] is L, comp dext:
                DerH += [comp_ext(_lay[1],lay[1],[Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec])]
                # pack extt as ptuple
    L = len(subH)
    return DerH, [Mval,Dval],[Mrdn,Drdn],[Mdec/L,Ddec/L]  # new layer,= 1/2 combined derH

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

def sum_subH(SubH, subH, base_rdn, fneg=0):

    if SubH:
        for Layer, layer in zip_longest(SubH,subH, fillvalue=[]):
            if layer:
                if Layer:
                    if layer[0] and isinstance(Layer[0][0], list):  # _lay[0][0] is derH
                        sum_derH(Layer, layer, base_rdn, fneg)
                    else: sum_ext(Layer, layer)
                else:
                    SubH += [deepcopy(layer)]  # _lay[0][0] is mL
    else:
        SubH[:] = deepcopy(subH)

''' replace with:
    SubH[:] = [
        # sum subLays: [derHt,valt,rdnt]s, fneg*i: for dsubH, poss sum_ext?
        [ [sum_derH(DerH,derH, fneg*i) for i,(DerH,derH) in enumerate(zip(Tuplet,tuplet))],
          [Val+val for Val, val in zip(Valt,valt)], [Rdn+rdn + base_rdn for Rdn, rdn in zip(Rdnt,rdnt)]
        ]
        for [Tuplet,Valt,Rdnt], [tuplet,valt,rdnt]
        in zip_longest(SubH, subH, fillvalue=[([],[]), (0,0),(0,0)])  # SubHt, valt, rdnt
    ]
'''

def comp_ext(_ext, ext, Valt, Rdnt, Dect):  # comp ds:

    (_L,_S,_A),(L,S,A) = _ext,ext
    dL = _L-L
    dS = _S/_L - S/L
    if isinstance(A,list):
        mA, dA = comp_angle(_A,A); adA=dA; max_mA = max_dA = .5  # = ave_dangle
    else:
        mA = matchF(_A,A)- ave_dangle; dA = _A-A; adA = abs(dA); _aA=abs(_A); aA=abs(A)
        max_dA = _aA + aA; max_mA = max(_aA, aA)
    mL = matchF(_L,L) - ave_L
    mS = matchF(_S,S) - ave_L
    m = mL + mS + mA
    d = abs(dL) + abs(dS) + adA
    Valt[0] += m; Valt[1] += d
    Rdnt[0] += d>m; Rdnt[1] += d<=m

    _aL = abs(_L); aL = abs(L); _aS = abs(_S); aS = abs(S)
    Dect[0] = (Dect[0] + (mL+mS+mA) / (max(aL,_aL) + max(aS,_aS) + max_mA)) / 2
    Dect[1] = (Dect[1] + (dL+dS+dA) / (_aL+aL + _aS+aS + max_dA)) / 2

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
    Y,X,Y0,Yn,X0,Xn = Box; y,x,y0,yn,x0,xn = box
    Box[:] = [Y+y, X+x, min(X0,x0), max(Xn,xn), min(Y0,y0), max(Yn,yn)]


def feedback(root, fd):  # called from form_graph_, append new der layers to root

    AggH, ValHt, RdnHt = root.fback_t[fd].pop(0)  # init with 1st [aggH,valHt,rdnHt]
    while root.fback_t[fd]:
        aggH, valHt, rdnHt, = root.fback_t[fd].pop(0)
        sum_aggH(AggH, aggH, base_rdn=0); sum_Hts(ValHt,valHt, RdnHt,rdnHt)  # we need to sum decHt here?
    sum_aggH(root.aggH,AggH, base_rdn=0); sum_Hts(root.valHt, ValHt,root.rdnHt,RdnHt)  # both fder forks sum into a same root?

    if isinstance(root, Cgraph):  # root is not CEdge, which has no roots
        for fd, rroot_ in enumerate(root.root):  # should be roott here?
            for rroot in rroot_:
                fd = root.fd  # current node_ fd
                fback_ = rroot.fback_t[fd]
                fback_ += [[AggH, ValHt, RdnHt]]
                if fback_ and (len(fback_) == len(rroot.node_t)):  # flat, all rroot nodes terminated and fed back
                    # getting cyclic rroot here not sure why it can happen, need to check further
                    feedback(rroot, fd)  # sum2graph adds aggH per rng, feedback adds deeper sub+ layers