import numpy as np
from copy import deepcopy, copy
from itertools import zip_longest
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
        if edge.valt[fd] * np.sqrt(len(node_)-1) if node_ else 0 > G_aves[fd] * edge.rdnt[fd]:
            G_ = []  # convert CPPs to Cgraphs:
            for PP in node_:
                derH, valt, rdnt = PP.derH, PP.valt, PP.rdnt  # init aggH is empty:
                G_ += [Cgraph( ptuple=PP.ptuple, derH=[derH,valt,rdnt], val_Ht=[[valt[0]],[valt[1]]], rdn_Ht=[[rdnt[0]],[rdnt[1]]],
                               L=PP.ptuple[-1], box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box))]
            node_ = G_
            edge.val_Ht[0][0] = edge.valt[0]; edge.rdn_Ht[0][0] = edge.rdnt[0]  # copy
            agg_recursion(None, edge, node_, fd=0)  # edge.node_t = graph_t, micro and macro recursive


def agg_recursion(rroot, root, G_, fd):  # compositional agg+|sub+ recursion in root graph, clustering G_

    comp_G_(G_, fd)  # cross-comp all Gs in (rng,der), nD array? form link_H per G

    root.val_Ht[fd] += [0]; root.rdn_Ht[fd] += [1]  #  estimate, no node.rdn += 1
    ave = G_aves[fd]  # add current fork rdn:
    root.rdn_Ht[fd][-1] += (root.val_Ht[fd][-1] - ave*root.rdn_Ht[fd][-1] > root.val_Ht[1-fd][-1] - G_aves[1-fd]*root.rdn_Ht[1-fd][-1])
    _root_t_ = []
    for G in G_:
        _root_t_+= [G.root_t]  # merge G roots in)between forks into GG roots: between Gs and root
        G.root_t = [[],[]]  # fill with GGs:
    GG_t = form_graph_t(root, G_, _root_t_)  # internal eval sub+/graph and feedback
    # agg+ cross-comp-> form_graph_t loop sub+) recursive agg+, vs. comp_slice sub+ looping-> evaluation-> cross-comp

    for GG_ in GG_t:  # comp_G_ eval: ave_m * len*rng - fixed cost, root update in form_t:
        if root.val_Ht[0][-1] * (len(GG_)-1)*root.rng > ave * root.rdn_Ht[0][-1]:  
            agg_recursion(rroot, root, GG_, fd=0) # 1st xcomp in GG_

    G_[:] = GG_t

def form_graph_t(root, G_, _root_t_):  # root function to form fuzzy graphs of nodes per fder,fd

    graph_t = []
    for fd in 0,1:
        Gt_ = eval_node_connectivity(G_, fd)  # sum surround link values @ incr rng,decay: val += linkv/maxv * _node_val
        init_ = select_init_(Gt_, fd)  # select sparse max nodes to initialize graphs
        graph_ = segment_node_(init_, Gt_, fd, _root_t_)
        graph_ = prune_graph_(graph_, fd)  # sort node roots to add to graph rdn, prune weak graphs
        graph_ = sum2graph_(graph_, fd)  # convert to Cgraphs
        graph_t += [graph_]  # add alt_graphs?
    # sub+:
    for fd, graph_ in enumerate(graph_t):  # breadth-first for in-layer-only roots
        for graph in graph_:  # external to agg+ vs internal in comp_slice sub+
            node_ = graph.node_t  # still flat?  eval fd comp_G_ in sub+:
            if sum(graph.val_Ht[fd]) * (len(node_)-1)*root.rng > G_aves[fd] * sum(graph.rdn_Ht[fd]):
                agg_recursion(root, graph, node_, fd)  # replace node_ with node_t, recursive
            else:  # feedback after graph sub+:
                root.fback_t[fd] += [[graph.aggH, graph.val_Ht, graph.rdn_Ht]]  # merge forks in root fork
        # recursive feedback after all G_ sub+:
        if root.fback_t and root.fback_t[fd]:
            feedback(root, fd)  # update root.root.. aggH, val_Ht,rdn_Ht

    return graph_t  # root.node_t'node_ -> node_t: incr nested with each agg+?


def comp_G_(G_, fd=0, oG_=None, fin=1):  # cross-comp in G_ if fin, else comp between G_ and other_G_, for comp_node_

    if not fd:  # cross-comp all Gs in extended rng, add proto-links regardless of prior links
        for G in G_: G.link_H += [[]]  # add empty link layer, may remove if stays empty
        if oG_:
            for oG in oG_: oG.link_H += [[]]
        # form new links:
        for i, G in enumerate(G_):
            if fin: _G_ = G_[i+1:]  # xcomp in G_
            else:   _G_ = oG_  # xcomp G_,other_G_, also start @ i? not used yet
            for _G in _G_:
                if _G in G.compared_: continue  # skip if previously compared
                dy = _G.box[0] - G.box[0]; dx = _G.box[1] - G.box[1]
                distance = np.hypot(dy, dx)  # Euclidean distance between centers of Gs
                if distance < ave_distance * ((sum(_G.val_Ht[fd]) + sum(G.val_Ht[fd])) / (2*sum(G_aves))):  # very tentative
                    # close enough to compare:
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

def comp_G(link_, link, fd):

    maxM,maxD = 0,0  # max possible summed m|d, to compute relative summed m|d: V/maxV, link mediation coef
    Mval,Dval = 0,0
    Mrdn,Drdn = 1,1
    _G, G = link._G, link.G

    # / P:
    mtuple, dtuple, Mtuple, Dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1, fagg=1)
    maxm, maxd = sum(Mtuple), sum(Dtuple)
    mval, dval = sum(mtuple), sum(abs(d) for d in dtuple)  # mval is signed, m=-min in comp x sign
    mrdn = dval>mval; drdn = dval<=mval
    derLay0 = [[mtuple,dtuple], [mval,dval], [mrdn,drdn]]
    Mval+=mval; Dval+=dval; Mrdn += mrdn; Drdn += drdn; maxM+=maxm; maxD+=maxd
    # / PP:
    dderH, valt, rdnt, maxt = comp_derH(_G.derH[0], G.derH[0], rn=1, fagg=1)
    mval,dval = valt; maxm,maxd = maxt
    Mval+=dval; Dval+=mval; maxM+=maxm; maxD+=maxd
    Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval

    derH = [[derLay0]+dderH, [Mval,Dval], [Mrdn,Drdn]]  # appendleft derLay0 from comp_ptuple
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval],[Mrdn,Drdn],[maxM,maxD])
    SubH = [der_ext, derH]  # two init layers of SubH, higher layers added by comp_aggH:
    # / G:
    if fd:  # else no aggH yet?
        subH, valt, rdnt, maxt = comp_aggH(_G.aggH, G.aggH, rn=1)
        SubH += subH  # append higher subLayers: list of der_ext | derH s
        mval,dval = valt; maxm,maxd = maxt
        Mval+=mval; Dval+=dval; maxM += maxm; maxD += maxd
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
        link_ += [link]

    elif Mval > ave_Gm or Dval > ave_Gd:  # or sum?
        link.subH = SubH; link.maxt = [maxM,maxD]; link.valt = [Mval,Dval]; link.rdnt = [Mrdn,Drdn]  # complete proto-link
        link_ += [link]


def eval_node_connectivity(node_, fd):  # sum surrounding link values to select nodes, to initialize graphs

    ave = G_aves[fd]
    Gt_ = []
    for i,G in enumerate(node_):
        G.it[fd] = i  # only used here, in select_init_, segment_node_?
        Gt_ += [[G, G.val_Ht[fd][-1] - ave * G.rdn_Ht[fd][-1]]]  # Gt = [G,_Val], ave is normalized for circular links?

    while True:  # iterative Val range expansion by summing decayed surround node Vals, via same direct links
        DVal = 0  # node_Val update
        for i, (_G, _Val) in enumerate(Gt_):
            Val = 0  # updated _G surround value
            for link in _G.link_H[-1]:
                G = link.G if link._G is _G else link._G
                GVal = Gt_[G.it[fd]][1]
                Val += GVal * (link.valt[fd] / link.maxt[fd])  # _G Val * link decay (m|d / max: self=100%?)
            Gt_[i][1] = Val  # _G Val update, unilateral for simplicity: computed separately for _G
            DVal += abs(_Val-Val)  # node_Val update / surround extension, eval in init
        if DVal < ave:  # low node_Val update, also if low node_Val?
            break

    return Gt_

def select_init_(Gt_, fd):  # local max selection for sparse graph init:

    max_, non_max_ = [],[]  # pick max in direct links,
    # max mediated links and _node_ buffer for more sparsity?
    for _node, _Val in Gt_:
        if _Val<=0 or _node in non_max_: continue  # can't init graph
        fmax=1
        for link in _node.link_H[-1]:
            node = link.G if link._G is _node else link._G
            if _Val > Gt_[node.it[fd]][1]:
                non_max_ += [node]  # skip as next node
            else:
                fmax=0; break  # break is not necessary
        if fmax:
            max_ += [[_node,_Val]]
    return max_


def segment_node_(init_, Gt_, fd, root_t_):

    graph_ = []  # initialize graphs with local maxes, eval their links to add other nodes:
    for inode, ival in init_:

        iroot_t = root_t_[inode.it[fd]]  # same order as Gt_, assign node roots to new graphs, init with max
        graph = [[inode],ival,[iroot_t]]
        inode.root_t[fd] += [graph]
        _nodet_ = [[inode,ival,[iroot_t]]]  # current perimeter of the graph, init = graph

        while _nodet_:  # search links outwards recursively to form overlapping graphs:
            nodet_ = []
            for _node,_val,_root_t in _nodet_:
                for link in _node.link_H[-1]:
                    node = link.G if link._G is _node else link._G
                    if node in graph[0]: continue
                    val = Gt_[node.it[fd]][1]
                    root_t = root_t_[node.it[fd]]
                    if val * (link.valt[fd] / link.maxt[fd]) > ave:  # node val * link decay
                        graph[0] += [node]
                        graph[1] += val
                        graph[2] += [root_t]
                        nodet_ += [node,val,root_t]  # new perimeter
                    # else: node not in this graph, but may be in another
            _nodet_ = nodet_
        graph_ += [graph]
    return graph_


def prune_graph_(graph_, fd):  # compute graph overlap to prune weak graphs, not nodes: rdn doesn't change the structure
                               # prune rootless nodes?
    for graph in graph_:
        for node in graph[0]:  # root rank = graph/node rdn:
            roots = sorted(node.root_t[fd], key=lambda root: root[1], reverse=True)  # sort by net val
            # or grey-scale rdn = root_val / sum_higher_root_vals?
            for rdn, graph in enumerate(roots):
                graph[1] -= ave*rdn  # rdn to >val overlapping graphs per node, also >val forks, alt sparse param sets?
                # nodes are shared by multiple max-initialized graphs, pruning here still allows for some overlap
    pruned_graph_ = []
    for graph in graph_:
        if graph[1] > G_aves[fd]:  # rdn-adjusted Val for local sparsity, doesn't affect G val?
            pruned_graph_ += [graph]
        else:
            for node in graph[0]:
                node.root_t[fd].remove(graph)

    return pruned_graph_


def sum2graph_(graph_, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    Graph_ = []
    for graph in graph_:      # seq graphs
        Root_t = graph[2][0]  # merge _root_t_ into Root_t:
        [merge_root_tree(Root_t, root_t) for root_t in graph[2][1:]]
        Graph = Cgraph(fd=fd, root_t=Root_t, L=len(graph[0]))  # n nodes
        Link_ = []
        for G in graph[0]:
            sum_box(Graph.box, G.box)
            sum_ptuple(Graph.ptuple, G.ptuple)
            sum_derH(Graph.derH, G.derH, base_rdn=1)  # base_rdn?
            sum_aggH([Graph.aggH,Graph.val_Ht,Graph.rdn_Ht], [G.aggH,G.val_Ht,G.rdn_Ht], base_rdn=1)
            link_ = G.link_H[-1]
            Link_[:] = list(set(Link_ + link_))
            subH=[]; valt=[0,0]; rdnt=[1,1]
            for derG in link_:
                if derG.valt[fd] > G_aves[fd]:
                    sum_subH([subH,valt,rdnt], [derG.subH,derG.valt,derG.rdnt], base_rdn=1, fneg = G is derG.G)  # fneg: reverse link sign
                    sum_box(G.box, derG.G.box if derG._G is G else derG._G.box)
            G.aggH += [[subH,valt,rdnt]]
            for i in 0,1:
                G.val_Ht[i][-1] += valt[i]; G.rdn_Ht[i][-1] += rdnt[i]
            G.root_t[fd][G.root_t[fd].index(graph)] = Graph  # replace list graph in root_tt
            Graph.node_t += [G]  # then converted to node_t by feedback
        subH=[]; valt=[0,0]; rdnt=[1,1]

        for derG in Link_:  # sum unique links:
            sum_subH([subH,valt,rdnt], [derG.subH, derG.valt, derG.rdnt], base_rdn=1)
            Graph.A[0] += derG.A[0]; Graph.A[1] += derG.A[1]
            Graph.S += derG.S
        Graph.aggH += [[subH, valt, rdnt]]  # new aggLev
        for i in 0,1:
            Graph.val_Ht[i][-1] += valt[i]; Graph.rdn_Ht[i][-1] += rdnt[i]
        Graph_ += [Graph]

    return Graph_

def merge_root_tree(Root_t, root_t):  # not-empty fork layer is root_t, each fork may be empty list:

    for Root_, root_ in zip(Root_t, root_t):
        for Root, root in zip(Root_, root_):
            if root.root_t:  # not-empty root fork layer
                if Root.root_t: merge_root_tree(Root.root_t, root.root_t)
                else: Root.root_t[:] = root.root_t
        Root_[:] = list( set(Root_+root_))  # merge root_, may be empty


def sum_box(Box, box):
    Y,X,Y0,Yn,X0,Xn = Box; y,x,y0,yn,x0,xn = box
    Box[:] = [Y+y, X+x, min(X0,x0), max(Xn,xn), min(Y0,y0), max(Yn,yn)]

'''
derH: [[tuplet, valt, rdnt]]: default input from PP, for both rng+ and der+, sum min len?
subH: [[derH_t, valt, rdnt]]: m,d derH, m,d ext added by agg+ as 1st tuplet
aggH: [[subH_t, valt, rdnt]]: composition levels, ext per G, 
'''

def comp_subH(_subH, subH, rn):
    DerH = []
    maxM, maxD, Mval, Dval, Mrdn, Drdn = 0,0,0,0,1,1

    for _lay, lay in zip_longest(_subH, subH, fillvalue=[]):  # compare common lower layer|sublayer derHs
        if _lay and lay:  # also if lower-layers match: Mval > ave * Mrdn?
            if _lay[0] and isinstance(_lay[0][0],list):  # _lay[0][0] is derHt

                dderH, valt, rdnt, maxt = comp_derH(_lay[0], lay[0], rn, fagg=1)
                DerH += [[dderH, valt, rdnt, maxt]]  # for flat derH
                mval,dval = valt; maxm,maxd = maxt
                Mval += mval; Dval += dval; maxM += maxm; maxD += maxd
                Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + dval <= mval
            else:  # _lay[0][0] is L, comp dext:
                DerH += [comp_ext(_lay[1],lay[1],[Mval,Dval],[Mrdn,Drdn],[maxM,maxD])]  # pack extt as ptuple

    return DerH, [Mval,Dval],[Mrdn,Drdn],[maxM,maxD]  # new layer,= 1/2 combined derH

def comp_aggH(_aggH, aggH, rn):  # no separate ext
    SubH = []
    maxM,maxD, Mval,Dval, Mrdn,Drdn = 0,0,0,0,1,1

    for _lev, lev in zip_longest(_aggH, aggH, fillvalue=[]):  # compare common lower layer|sublayer derHs
        if _lev and lev:  # also if lower-layers match: Mval > ave * Mrdn?
            # compare dsubH only:
            dsubH, valt,rdnt,maxt = comp_subH(_lev[0], lev[0], rn)
            SubH += dsubH  # flatten to keep subH
            mval,dval = valt; maxm,maxd = maxt
            Mval += mval; Dval += dval; maxM += maxm; maxD += maxd
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + mval <= dval

    return SubH, [Mval,Dval],[Mrdn,Drdn],[maxM,maxD]

def sum_subH(T, t, base_rdn, fneg=0):

    SubH, Valt, Rdnt = T; subH, valt, rdnt = t

    for i in 0,1:
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i]
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

def sum_aggH(T, t, base_rdn):

    AggH, Val_Ht, Rdn_Ht = T; aggH, val_Ht, rdn_Ht = t

    for i in 0,1:  # link maxv is not summed in G.valt
        for j, (Val, val) in enumerate(zip_longest(Val_Ht[i], val_Ht[i], fillvalue=None)):
            if val != None:
                if Val != None: Val_Ht[i][j] += val
                else:           Val_Ht[i] += [val]
        for j, (Rdn, rdn) in enumerate(zip_longest(Rdn_Ht[i], rdn_Ht[i], fillvalue=None)):
            if rdn != None:
                if Rdn != None: Rdn_Ht[i][j] += rdn
                else:           Rdn_Ht[i] += [rdn]
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


def comp_ext(_ext, ext, Valt, Rdnt, Maxt):  # comp ds:

    (_L,_S,_A),(L,S,A) = _ext,ext

    dL = _L-L
    dS = _S/_L - S/L
    if isinstance(A,list):
        mA, dA = comp_angle(_A,A); adA=dA; max_mA = max_dA = .5  # = ave_dangle
    else:
        mA = matchF(_A,A) - ave_dangle; dA = _A-A; adA = abs(dA); _aA=abs(_A); aA=abs(A)
        max_dA = _aA + aA; max_mA = max(_aA, aA)
    mL = matchF(_L,L) - ave_L
    mS = matchF(_S,S) - ave_L
    m = mL + mS + mA
    d = abs(dL) + abs(dS) + adA
    Valt[0] += m; Valt[1] += d
    Rdnt[0] += d>m; Rdnt[1] += d<=m

    _aL = abs(_L); aL = abs(L); _aS = abs(_S); aS = abs(S)
    Maxt[0] += max(aL,_aL) + max(aS,_aS) + max_mA
    Maxt[1] += _aL+aL + _aS+aS + max_dA

    return [[mL,mS,mA], [dL,dS,dA]]  # no Mtuple, Dtuple?

def sum_ext(Extt, extt):

    if isinstance(Extt[0], list):
        for Ext,ext in zip(Extt,extt):  # ext: m,d tuple
            for i,(Par,par) in enumerate(zip(Ext,ext)):
                Ext[i] = Par+par
    else:  # single ext
        for i in 0,1: Extt[i]+=extt[i]  # sum L,S
        for j in 0,1: Extt[2][j]+=extt[2][j]  # sum dy,dx in angle


def feedback(root, fd):  # called from form_graph_, append new der layers to root

    Fback = root.fback_t[fd].pop(0)  # init with 1st [aggH,val_Ht,rdn_Ht]
    while root.fback_t[fd]:
        sum_aggH(Fback, root.fback_t[fd].pop(0), base_rdn=0)
    sum_aggH([root.aggH, root.val_Ht, root.rdn_Ht], Fback, base_rdn=0)  # both fder forks sum into a same root?

    if isinstance(root, Cgraph):  # root is not CEdge, which has no roots
        for fd, rroot_ in enumerate(root.root_t):
            for rroot in rroot_:
                fd = root.fd  # current node_ fd
                fback_ = rroot.fback_t[fd]
                fback_ += [Fback]
                if fback_ and (len(fback_) == len(rroot.node_t)):  # flat, all rroot nodes terminated and fed back
                    # getting cyclic rroot here not sure why it can happen, need to check further
                    feedback(rroot, fd)  # sum2graph adds aggH per rng, feedback adds deeper sub+ layers