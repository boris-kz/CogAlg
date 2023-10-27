import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
from collections import deque, defaultdict
from .classes import Cgraph, CderG, CPP
from .filters import ave_L, ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd
from .slice_edge import slice_edge, comp_angle
from .comp_slice import comp_P_, comp_derH, sum_derH, comp_ptuple, sum_ptuple, sum_dertuple, comp_dtuple, match_func


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
    for fd in 0,1:
        node_ = edge.node_[fd]  # always PP_t
        if edge.valt[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.rdnt[fd]:
            G_= []
            for PP in node_:  # convert CPPs to Cgraphs:
                derH,valt,rdnt = PP.derH,PP.valt,PP.rdnt  # init aggH is empty:
                for dderH in derH: dderH += [[0,0]]  # add maxt
                G_ += [Cgraph( ptuple=PP.ptuple, derH=[derH,valt,rdnt,[0,0]], valHt=[[valt[0]],[valt[1]]], rdnHt=[[rdnt[0]],[rdnt[1]]],
                               L=PP.ptuple[-1], box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box))]
            node_ = G_
            edge.valHt[0][0] = edge.valt[0]; edge.rdnHt[0][0] = edge.rdnt[0]  # copy
            agg_recursion(None, edge, node_, fd=0)  # edge.node_ = graph_t, micro and macro recursive


def agg_recursion(rroot, root, G_, fd):  # compositional agg+|sub+ recursion in root graph, clustering G_

    Valt,Rdnt, Glink__ = comp_G_(G_,fd)  # rng|der cross-comp all Gs, nD array? form link_ per G
    root.valHt[fd] += [0]; root.rdnHt[fd] += [1]  # sum in form_graph_t feedback

    GG_t = form_graph_t(root, Valt,Rdnt, G_, Glink__)  # eval sub+ and feedback per graph
    # agg+ xcomp-> form_graph_t loop sub)agg+, vs. comp_slice:
    # sub+ loop-> eval-> xcomp
    for GG_ in GG_t:  # comp_G_ eval: ave_m * len*rng - fixed cost, root update in form_t:
        if root.valHt[0][-1] * (len(GG_)-1)*root.rng > G_aves[fd] * root.rdnHt[0][-1]:
            agg_recursion(rroot, root, GG_, fd=0)  # 1st xcomp in GG_

    G_[:] = GG_t

def form_graph_t(root, Valt,Rdnt, G_, Glink__):  # form mgraphs and dgraphs of same-root nodes

    graph_t = []
    for G in G_: G.root = [None,None]  # replace with mcG_|dcG_ in segment_node_, replace with Cgraphs in sum2graph
    for fd in 0,1:
        if Valt[fd] > ave * Rdnt[fd]:  # else no clustering
            graph_t += [segment_node_(root, G_, Glink__,fd)]  # add alt_graphs?
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
Graphs add nodes if med_val = (val+_val) * decay > ave: connectivity is aggregate, individual link may be a fluke. 
This is density-based clustering, but much more subtle than conventional versions.
evaluate perimeter: negative or new links
negative perimeter links may turn positive with increasing mediation: linked node val may grow faster than rdn
positive links are within the graph, it's density can't decrease, so they don't change 
'''
# tentative
def segment_node_(root, iG_, Glink__, fd):  # local external Glink_ per G is formed in comp_G_

    # sum surrounding link values to define incrementally mediated connected nodes:
    graph_, Gt_ = [],[]; ave = G_aves[fd]

    for G, Glink_ in zip(iG_,Glink__):
        # internal ival,irdn are currently not used:
        ival, irdn, perimeter = sum(G.valHt[fd]), sum(G.rdnHt[fd]), Glink_

        Gt = [G, 0,0, ival,irdn, perimeter]  # perimeter: negative or new links, separate termination per node?
        grapht = [[Gt], 0,0, ival,irdn, copy(perimeter)]  # nodet_, Val,Rdn, iVal,iRdn, Perimeter
        G.root[fd] = grapht
        graph_ += [grapht]; Gt_ += [Gt]  # to access Gt by G.i

    _tVal, _tRdn = 0,0
    # eval incr mediated links, sum perimeter Vals, append node_, while significant Val update:
    while True:
        tVal, tRdn = 0,0  # iG_ loop totals
        # update graph surround:
        for grapht in graph_:
            nodet_, Val,Vdn, iVal,iRdn, Perimeter = grapht
            periVal, periRdn, inVal, inRdn = 0,0,0,0  # in: new in-graph values, net-positive
            new_Perimeter = []
            for link in Perimeter:
                if link.G in grapht[0]:  # access Gt by G.i
                    Gt = Gt_[link.G.i]; _Gt = Gt_[link._G.i]
                else:
                    Gt = Gt_[link._G.i]; _Gt = Gt_[link.G.i]
                if _Gt[0] not in iG_ or _Gt in nodet_:
                    continue
                G, val, rdn, ival, irdn, perimeter = Gt  # not sure we actually need perimeter per node
                _G,_val,_rdn,_ival,_irdn,_perimeter = _Gt
                # use relative link vals only:
                try: decay = link.valt[fd]/link.maxt[fd]  # link decay coef: m|d / max, base self/same
                except ZeroDivisionError: decay = 1
                med_val = (rdn+_val) * decay; periVal += med_val
                med_rdn = (rdn+_rdn) * decay; periRdn += med_rdn
                # tentative:
                if med_val > ave * med_rdn:
                    inVal += med_val; Gt[1] += med_val; _Gt[1] += med_val  # also sum per node
                    inRdn += med_rdn; Gt[2] += med_val; _Gt[2] += med_val
                    # merge _graph in graph:
                    _nodet_,_Val,_Vdn,_iVal,_iRdn,_Perimeter = _G.root[fd]
                    Val += _val; rdn += _rdn
                    nodet_ = list(set(nodet_ + _Gt[0].root[fd][0]))  # append _nodet_
                    Perimeter = list(set(Perimeter + _Perimeter))
                    new_Perimeter = list(set(new_Perimeter + _Perimeter))
                else:  # negative link, for reevaluation:
                    new_Perimeter = list(set(new_Perimeter + [link]))
            # separate node perimeter extension?
            # update per graph extension, signed:
            grapht[1] += inVal; tVal += inVal  # if eval per graph: DVal += inVal -_inVal?
            grapht[2] += inRdn; tRdn += inRdn  # DRdn += inRdn -_inRdn

        if (tVal-_tVal) < ave * (tRdn-_tRdn):  # even low-Val extension may be valuable if Rdn decreases?
            break
        _tVal,_tRdn = tVal,_tRdn

    return [sum2graph(root, graph, fd) for graph in graph_ if graph[1] > ave * graph[2]]  # Val > ave * Rdn


def sum2graph(root, cG_, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    graph = Cgraph(root=root, fd=fd, L=len(cG_))  # n nodes, transplant both node roots
    SubH = [[],[0,0],[1,1],[0,0]]; maxM,maxD, Mval,Dval, Mrdn,Drdn = 0,0,0,0,0,0
    Link_= []
    for i, (G,Glink_) in enumerate(cG_[3]):
        # sum nodes in graph:
        sum_box(graph.box, G.box)
        sum_ptuple(graph.ptuple, G.ptuple)
        sum_derHv(graph.derH, G.derH, base_rdn=1)
        sum_aggH(graph.aggH, G.aggH, base_rdn=1)
        sum_Hts(graph.valHt, graph.rdnHt, graph.maxHt, G.valHt, G.rdnHt, G.maxHt)
        # sum external links:
        subH=[[],[0,0],[1,1],[0,0]]; mval,dval, mrdn,drdn, maxm,maxd = 0,0, 0,0, 0,0
        for derG in Glink_:
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
                sum_box(G.box, derG.G[0].box if derG._G[0] is G else derG._G[0].box)  # derG.G is proto-graph
        # from G links:
        if subH: G.aggH += [subH]
        G.i = i
        G.valHt[0]+=[mval]; G.valHt[1]+=[dval]; G.rdnHt[0]+=[mrdn]; G.rdnHt[1]+=[drdn]
        G.maxHt[0]+=[maxm]; G.maxHt[1]+=[maxd]
        G.root[fd] = graph  # replace cG_
        graph.node_t += [G]  # converted to node_t by feedback
    # + link layer:
    graph.link_ = Link_  # use in sub_recursion, as with PPs, no link_?
    graph.valHt[0]+=[Mval]; graph.valHt[1]+=[Dval]; graph.rdnHt[0]+=[Mrdn]; graph.rdnHt[1]+=[Drdn]
    graph.maxHt[0]+=[maxM]; graph.maxHt[1]+=[maxD]

    return graph


def sum_Hts(ValHt, RdnHt, MaxHt, valHt, rdnHt, maxHt):
    # loop m,d Hs, add combined decayed lower H/layer?

    for ValH,valH, RdnH,rdnH, MaxH,maxH in zip(ValHt,valHt, RdnHt,rdnHt, MaxHt,maxHt):
        ValH[:] = [V+v for V,v in zip_longest(ValH, valH, fillvalue=0)]
        MaxH[:] = [M+m for M,m in zip_longest(MaxH, maxH, fillvalue=0)]
        RdnH[:] = [R+r for R,r in zip_longest(RdnH, rdnH, fillvalue=0)]
'''
derH: [[tuplet, valt, rdnt]]: default input from PP, rng+|der+, sum min len?
subH: [derH_t]: m,d derH, m,d ext added by agg+ as 1st tuplet
aggH: [subH_t]: composition levels, ext per G, 
'''

def comp_G_(G_, fd=0, oG_=None, fin=1):  # cross-comp in G_ if fin, else comp between G_ and other_G_, for comp_node_

    Mval,Dval, Mrdn,Drdn = 0,0,0,0
    if not fd:  # cross-comp all Gs in extended rng, add proto-links regardless of prior links
        for G in G_: G.link_ += [[]]  # add empty link layer, may remove if stays empty

        if oG_:
            for oG in oG_: oG.link_ += [[]]
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
                    # old:
                    G.link_ += [CderG( G=G, _G=_G, S=distance, A=[dy,dx])]  # proto-links, in G only
    link__ = []
    for G in G_:
        link_ = []
        for link in G.link_:  # if fd: follow links, comp old derH, else follow proto-links, form new derH
            if fd and link.valt[1] < G_aves[1]*link.rdnt[1]: continue  # maybe weak after rdn incr?
            mval,dval, mrdn,drdn = comp_G(link_,link, fd)
            Mval+=mval;Dval+=dval; Mrdn+=mrdn;Drdn+=drdn
        link__ = [link_]
        '''
        same comp for cis and alt components?
        for _cG, cG in ((_G, G), (_G.alt_Graph, G.alt_Graph)):
            if _cG and cG:  # alt Gs maybe empty
                comp_G(_cG, cG, fd)  # form new layer of links:
        combine cis,alt in aggH: alt represents node isolation?
        comp alts,val,rdn? cluster per var set if recurring across root: type eval if root M|D? '''

    return [Mval,Dval], [Mrdn,Drdn], link__

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
        dderH, valt, rdnt, maxt = comp_derHv(_derH[0], derH[0], rn=1)
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

# draft:
def comp_aggH(_aggH, aggH, rn):  # no separate ext
    SubH = []
    maxM,maxD, Mval,Dval, Mrdn,Drdn = 0,0,0,0,1,1

    for _lev, lev in zip_longest(_aggH, aggH, fillvalue=[]):  # compare common lower layer|sublayer derHs
        if _lev and lev:  # also if lower-layers match: Mval > ave * Mrdn?
            # compare dsubH only:
            dsubH, valt,rdnt,maxt = comp_subH(_lev, lev, rn)  # no more valt and rdnt in subH now
            SubH += dsubH  # flatten to keep subH
            maxM += maxt[0]; maxD += maxt[1]
            mval,dval = valt; Mval += mval; Dval += dval
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + mval <= dval

    return SubH, [Mval,Dval],[Mrdn,Drdn],[maxM,maxD]

def comp_subH(_subH, subH, rn):
    DerH = []
    maxM,maxD, Mval,Dval, Mrdn,Drdn = 0,0,0,0,1,1

    for _lay, lay in zip_longest(_subH, subH, fillvalue=[]):  # compare common lower layer|sublayer derHs
        if _lay and lay:  # also if lower-layers match: Mval > ave * Mrdn?
            if _lay[0] and isinstance(_lay[0][0],list):
                # _lay[0] is derHv
                dderH, valt, rdnt, maxt = comp_derHv(_lay[0], lay[0], rn)
                DerH += [[dderH, valt, rdnt, maxt]]  # flat derH
                maxM += maxt[0]; maxD += maxt[1]
                mval,dval = valt; Mval += mval; Dval += dval
                Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + dval <= mval
            else:  # _lay[0][0] is L, comp dext:
                DerH += [comp_ext(_lay[1],lay[1],[Mval,Dval],[Mrdn,Drdn],[maxM,maxD])]
                # pack extt as ptuple
    return DerH, [Mval,Dval],[Mrdn,Drdn],[maxM,maxD]  # new layer,= 1/2 combined derH


def comp_derHv(_derH, derH, rn):  # derH is a list of der layers or sub-layers, each is ptuple_tv

    dderH = []  # or not-missing comparand: xor?
    Mval,Dval, Mrdn,Drdn, maxM,maxD = 0,0,1,1,0,0

    for _lay, lay in zip_longest(_derH, derH, fillvalue=[]):  # compare common lower der layers | sublayers in derHs
        if _lay and lay:  # also if lower-layers match: Mval > ave * Mrdn?
            # compare dtuples only, mtuples are for evaluation:
            mtuple, dtuple, Mtuple, Dtuple = comp_dtuple(_lay[0][1], lay[0][1], rn, fagg=1)
            # sum params:
            mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
            mrdn = dval > mval; drdn = dval < mval
            maxm = sum(Mtuple); maxd = sum(Dtuple)
            Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn; maxM+= maxm; maxD+= maxd
            ptuple_tv = [[mtuple,dtuple],[mval,dval],[mrdn,drdn],[maxm,maxd]]  # or += [Mtuple,Dtuple] for future comp?
            dderH += [ptuple_tv]  # derLay

    return dderH, [Mval,Dval],[Mrdn,Drdn],[maxM,maxD]  # new derLayer,= 1/2 combined derH


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
                        sum_derHv(Layer, layer, base_rdn, fneg)
                    else: sum_ext(Layer, layer)
                else:
                    SubH += [deepcopy(layer)]  # _lay[0][0] is mL
    else:
        SubH[:] = deepcopy(subH)

def sum_derHv(T, t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    DerH, Valt,Rdnt,Maxt = T; derH, valt,rdnt,maxt = t
    for i in 0,1:
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+ base_rdn; Maxt[i] += maxt[i]
    DerH[:] = [
        [ [sum_dertuple(Dertuple,dertuple, fneg*i) for i,(Dertuple,dertuple) in enumerate(zip(Tuplet,tuplet))],
          [V+v for V,v in zip(Valt,valt)], [R+r+base_rdn for R,r in zip(Rdnt,rdnt)], [M+m for M,m in zip(Maxt,maxt)],
        ]
        for [Tuplet, Valt,Rdnt,Maxt], [tuplet, valt,rdnt,maxt]
        in zip_longest(DerH, derH, fillvalue=[([0,0,0,0,0,0],[0,0,0,0,0,0]), (0,0),(0,0),(0,0)])  # ptuple_tv
    ]

def comp_ext(_ext, ext, Valt, Rdnt, Maxt):  # comp ds:

    (_L,_S,_A),(L,S,A) = _ext,ext
    dL = _L-L
    dS = _S/_L - S/L
    if isinstance(A,list):
        mA, dA = comp_angle(_A,A); adA=dA; max_mA = max_dA = .5  # = ave_dangle
    else:
        mA = match_func(_A,A)- ave_dangle; dA = _A-A; adA = abs(dA); _aA=abs(_A); aA=abs(A)
        max_dA = _aA + aA; max_mA = max(_aA, aA)
    mL = match_func(_L,L) - ave_L
    mS = match_func(_S,S) - ave_L

    m = mL + mS + mA
    d = abs(dL) + abs(dS) + adA
    Valt[0] += m; Valt[1] += d
    Rdnt[0] += d>m; Rdnt[1] += d<=m
    _aL = abs(_L); aL = abs(L)
    _aS = abs(_S); aS = abs(S)
    Maxt[0] += max(aL,_aL) + max(aS,_aS) + max_mA
    Maxt[1] += _aL+aL + _aS+aS + max_dA

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