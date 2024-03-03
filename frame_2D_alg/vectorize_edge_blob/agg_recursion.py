import numpy as np
from copy import deepcopy, copy
from itertools import combinations, zip_longest
from .classes import z, CderP, Cgraph, CderG, add_, comp_, get_match, nest
from .filters import aves, ave_mL, ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd
from .slice_edge import slice_edge, comp_angle
from .comp_slice import der_recursion, comp_ptuple
from utils import box2center, extend_box

'''
Blob edges may be represented by higher-composition patterns, etc., if top param-layer match,
in combination with spliced lower-composition patterns, etc, if only lower param-layers match.
This may form closed edge patterns around flat core blobs, which defines stable objects. 
Graphs are formed from blobs that match over < max distance. 
-
They are also assigned adjacent alt-fork graphs, which borrow predictive value from the graph. 
Primary value is match, diff.patterns borrow value from proximate match patterns, canceling their projected match. 
But alt match patterns borrow already borrowed value, which may be too tenuous to track, we use average borrowed value.
-
Clustering criterion is M|D, summed across >ave vars if selective comp (<ave vars are not compared and don't add costs).
Fork selection should be per var| der layer| agg level. Clustering is exclusive per fork,ave, overlapping between fork,aves.  
(same fork,ave fuzzy clustering is possible if centroid-based, connectivity-based clusters merge)

There are concepts that include same matching vars: size, density, color, stability, etc, but in different combinations.
Weak value vars are combined into higher var, so derivation fork can be selected on different levels of param composition.
Clustering by variance: lend|borrow, contribution or counteraction to similarity | stability, such as metabolism? 
-
graph G:
Agg+ cross-comps top Gs and forms higher-order Gs, adding up-forking levels to their node graphs.
Sub+ re-compares nodes within Gs, adding intermediate Gs, down-forking levels to root Gs, and up-forking levels to node Gs.
-
Generic graph is a dual tree with common root: down-forking input node Gs and up-forking output graph Gs. 
This resembles a neuron, which has dendritic tree as input and axonal tree as output. 
But we have recursively structured param sets packed in each level of these trees, which don't exist in neurons.

Diagrams: 
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/agg_recursion_unfolded.drawio.png
'''

def vectorize_root(blob):  # vectorization in 3 composition levels of xcomp, cluster:

    edge = slice_edge(blob)  # lateral kernel cross-comp -> P clustering
    der_recursion(None, edge)  # vertical, lateral-overlap P cross-comp -> PP clustering

    for fd, node_ in enumerate(edge.node_):  # always node_t
        if edge.et:
            val,rdn = edge.et[fd], edge.et[2+fd]
            if  val * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * rdn:
                for i, PP in enumerate(node_):  # CPP -> CG:
                    node_[i] = Cgraph(PP=PP); PP.root = None
                # discontinuous PP rng+ cross-comp, cluster -> G_t
                agg_recursion(None, edge, node_, nrng=1, fagg=1)  # agg+, no Cgraph nodes

    return edge


def agg_recursion(rroot, root, node_, nrng=1, fagg=0):  # lenH = len(root.aggH[-1][0]), lenHH: same in agg_compress

    Et = [0,0,0,0,0,0]  # need real init too: G-external vals summed from grapht link_ Et

    # agg+ der=1 xcomp of new Gs if fagg, else sub+: der+ xcomp of old Gs:
    nrng = rng_recursion(rroot, root, combinations(node_,r=2) if fagg else root.link_, Et, nrng)  # rng+ appends rim, link.derH

    form_graph_t(root, node_, Et, nrng, fagg)  # root_fd, eval der++ and feedback per Gd, not sub-recursion in Gms

    if node_ and isinstance(node_[0], list):
        for fd, G_ in enumerate(node_):
            if root.et[fd] * (len(G_)-1)*root.rng > G_aves[fd] * root.et[2+fd]:
                # agg+ / node_t, vs. sub+ / node_:
                agg_recursion(rroot, root, G_, nrng=1, fagg=1)
                if rroot and fd:  # der+ only
                    rroot.fback_ += [[root.aggH,root.et]]
                    feedback(rroot)  # update root.root..


def rng_recursion(rroot, root, Q, Et, nrng=1):  # rng++/ G_, der+/ link_ if called from sub+ fork of agg_recursion, -> rimH

    et = [0,0,0,0,0,0]
    _G_,_link_ = set(),set()  # for next rng+, der+
    fd = isinstance(Q,list)  # link_ is list, node_ is combinations

    if fd:
        for link in Q:  # init rng+ from der+, extend root links
            G = link.G; _G = link._G
            if _G in G.compared_: continue
            if link.Vt[1] > G_aves[1] * link.Rt[1]:
                G.compared_+=[_G]; _G.compared_+=[G]
                comp_G(link, et)
                comp_rim(_link_,link,nrng)  # add matching-direction rim links for next rng+?
    else:
        Gt_ = Q  # prelinks for init or recursive rng+, form new link_, or make it general?
        for (_G, G) in Gt_:
            if _G in G.compared_: continue
            dy, dx = box2center(G.box)  # compute distance between node centers:
            dist = np.hypot(dy, dx)
            # combined pairwise eval (directional val?) per rng+:
            if nrng==1 or ((G.Vt[0]+_G.Vt[0])/ (dist/ave_distance) > ave*(G.Rt[0]+_G.Rt[0])):
                link = CderG(_G=_G, G=G, A=[dy,dx], S=dist)
                G.compared_+=[_G]; _G.compared_+=[G]
                comp_G(link, et)
            else:
                _G_.add((_G, G))  # for next rng+

    if et[0] > ave_Gm * et[2]:  # rng+ eval per arg cluster because comp is bilateral, 2nd test per new pair
        Et[:] = [V+v for V, v in zip(Et, et)]  # Vt[i]+=v; Rt[i]+=rt[i]; Dt[i]+=d
        _Q = _link_ if fd else list(_G_)
        if _Q:
            nrng = rng_recursion(rroot, root, _Q, Et, nrng+1)  # eval rng+ for der+ too

    return nrng

def comp_rim(_link_, link, nrng):  # for next rng+:

    for G in link._G, link.G:
        for _link in G.rimH[-1]:
            _G = _link.G if _link.G in [link._G,link.G] else _link.G  # new to link
            dy = _G.box.cy - G.box.cy; dx = _G.box.cx - G.box.cx
            dist = np.hypot(dy, dx)  # distance between node centers
            # or use compared?
            if 2*nrng > dist > 2*(nrng-1):  # init max comparison distance = 2
                # potentially connected G,_G: within rng, no comp in prior rng
                if comp_angle(link.A,_link.A)[0] > ave:  # link direction matches in G|_G rimH[-1]
                    _link_.add(CderG(G=G, _G=_G, A=[dy,dx], S=dist))  # to compare in rng+


def form_graph_t(root, G_, Et, nrng, fagg=0):  # form Gm_,Gd_ from same-root nodes

    node_connect(G_)  # Graph Convolution of Correlations over init _G_
    node_t = []
    for fd in 0, 1:
        if Et[fd] > ave * Et[2+fd]:  # eVal > ave * eRdn
            graph_ = segment_node_(root, G_, fd, nrng, fagg)  # fd: node-mediated Correlation Clustering
            if fd:  # der+ only, rng++ term by high diffs, can't be extended much in sub Gs
                for graph in graph_:
                    if graph.link_ and graph.Et[1] > G_aves[1] * graph.Et[3]:
                        graph.Et = [0,0,0,0,0,0]  # reset
                        node_ = graph.node_
                        if isinstance(node_[0].rimH[0], z):  # 1st sub+, same rim nesting?
                            for node in node_: node.rimH = [node.rimH]  # rim -> rimH
                        agg_recursion(root, graph, graph.node_, nrng, fagg=0)
                    else:
                        root.fback_ += [[graph.aggH, graph.Et]]
                        feedback(root)  # update root.root.. per sub+
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]
    if any(node_t):
        G_[:] = node_t  # else keep root.node_


def node_connect(_G_):  # node connectivity = sum surround link vals, incr.mediated: Graph Convolution of Correlations
    '''
    Aggregate direct * indirect connectivity per node from indirect links via associated nodes, in multiple cycles.
    Each cycle adds contributions of previous cycles to linked-nodes connectivity, propagated through the network.
    Math: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/node_connect.png
    '''
    while True:
        # eval accumulated G connectivity vals, indirectly extending their range
        G_ = []  # next connectivity expansion, more selective by DVt,Lent = [0,0],[0,0]?
        for G in _G_:
            uprim = []  # >ave updates of direct links
            rim = G.rimH[-1] if G.rimH and isinstance(G.rimH[0], list) else G.rimH
            for i in 0,1:
                val,rdn,dec = G.Et[i::2]  # connect by last layer
                ave = G_aves[i]
                for link in rim:
                    # >ave derG in fd rim
                    lval,lrdn,ldec = link.et[i::2]; ldec /= link.n
                    _G = link._G if link.G is G else link.G
                    _val,_rdn,_dec = _G.Et[i::2]
                    # Vt.. for segment_node_:
                    V = ldec * (val+_val); dv = V-lval
                    R = ldec * (rdn+_rdn); dr = R-lrdn
                    D = ldec * (dec+_dec); dd = D-ldec
                    link.et[i::2] = [V, R, D]
                    if dv > ave * dr:
                        G.Et[i::2] = [V+v for V,v in zip(G.Et[i::2], [V, R, D])]  # add link last layer vals
                        if link not in uprim: uprim += [link]
                        # more selective eval: dVt[i] += dv; L=len(uprim); Lent[i] += L
                    if V > ave * R:
                        G.eet[i::2] = [V+v for V, v in zip(G.eet[i::2], [dv, dr, dd])]
            if uprim:  # prune rim for next loop
                rim[:] = uprim
                G_ += [G]
        if G_: _G_ = G_  # exclude weakly incremented Gs from next connectivity expansion loop
        else:  break


def segment_node_(root, root_G_, fd, nrng, fagg):  # eval rim links with summed surround vals for density-based clustering

    # graph += [node] if >ave (surround connectivity * relative value of link to any internal node)
    igraph_ = []; ave = G_aves[fd]

    for G in root_G_:   # init per node, last-layer Vt,Vt,Dt:
        grapht = [[G],[],G.Et, copy(G.rimH[-1] if G.rimH and isinstance(G.rimH[0],list) else G.rimH)]  # link_ = last rim
        G.root = grapht  # for G merge
        igraph_ += [grapht]
    _graph_ = igraph_

    while True:
        graph_ = []
        for grapht in _graph_:  # extend grapht Rim with +ve in-root links
            G_, Link_, Et, Rim = grapht
            inVal, inRdn = 0,0  # new in-graph +ve
            new_Rim = []
            for link in Rim:  # unique links
                if link.G in G_:
                    G = link.G; _G = link._G
                else:
                    G = link._G; _G = link.G
                if _G in G_: continue
                # connect by rel match of nodes * match of node Vs: surround M|Ds,
                # V suggests how deeply inside the graph is G
                cval = link.et[fd] + get_match(G.Et[fd],_G.Et[fd])  # same coef for int and ext match?
                crdn = link.et[2+fd] + (G.Et[2+fd] + _G.Et[2+fd]) / 2
                if cval > ave * crdn:  # _G and its root are effectively connected
                    # merge _root:
                    _grapht = _G.root
                    _G_,_Link_,_Et,_Rim = _grapht
                    # can't use set with simplenamespace anymore
                    if link not in Link_: Link_ += [link]
                    Link_[:] += [_link for _link in _Link_ if _link not in Link_]
                    for g in _G_:
                        g.root = grapht
                        if g not in G_: G_+=[g]
                    for i in 0,1:
                        Et[:] = [V+v for V,v in zip(Et, _Et)]
                        inVal += _Et[fd]; inRdn += _Et[2+fd]
                    if _grapht in igraph_:
                        igraph_.remove(_grapht)
                    new_Rim += [link for link in _Rim if link not in new_Rim+Rim+Link_]
            # for next loop:
            if len(new_Rim) * inVal > ave * inRdn:
                graph_ += [[G_,Link_, Et, new_Rim]]

        if graph_: _graph_ = graph_  # selected graph expansion
        else: break

    graph_ = []
    for grapht in igraph_:
        if grapht[2][fd] > ave * grapht[2][2+fd]:  # form Cgraphs if Val > ave* Rdn
            graph_ += [sum2graph(root, grapht[:3], fd, nrng, fagg)]

    return graph_


def sum2graph(root, grapht, fd, nrng, fagg):  # sum node and link params into graph, aggH in agg+ or player in sub+

    G_,Link_, Et = grapht

    graph = Cgraph(fd=fd, node_=G_,link_=Link_, Et=Et, rng=nrng)
    if fd: graph.root = root
    for link in Link_:
        link.root = graph
        graph.ext[1] += link.S
        graph.ext[2] += link.A
    if graph.ext[1] == 0: graph.ext[1] = 1   # prevent zero division for single node's graph
    ext_He, et, eet = [], [0,0,0,0,0,0], [0,0,0,0,0,0]
    # grapht int = node int+ext
    for G in G_:
        graph.ext[0] += 1
        graph.area += G.area
        sum_last_lay(G, fd)
        graph.box = extend_box(graph.box, G.box)
        graph.ptuple += G.ptuple
        add_(graph.He, G.He,irdnt=[1,1])
        ext_He = add_(ext_He, G.ext_He, irdnt=G.et[2:4])
        graph.aggH = [add_(aggH, GaggH, irdnt=[1,1]) for aggH, GaggH in zip_longest(graph.aggH, G.aggH, fillvalue=[])]
        eet = [V+v for V,v in zip(eet, G.eet)]
        et = [V+v for V,v in zip(et, G.et)]
    graph.aggH += [ext_He]  # dsubH|daggH
    graph.et = [V+v+ev for V,v,ev in zip(graph.et, et, eet)]  # graph internals = G Internals + Externals

    if fagg:
        if graph.aggH and graph.aggH[0][0] == 1:  # if depth == 1 (derH), convert to subH (depth =2)
            # 1st agg+, init aggH = [subHv]:
            graph.aggH =  [[2, [*graph.et], graph.aggH]]

    if fd:  # assign alt graphs from d graph, after both linked m and d graphs are formed
        for link in graph.link_:
            mgraph = link.roott[0]
            if mgraph:
                for fd, (G, alt_G) in enumerate(((mgraph,graph), (graph,mgraph))):  # bilateral assign:
                    if G not in alt_G.alt_graph_:
                        G.alt_graph_ += [alt_G]
                        G.aet = [V+v for V, v in zip_longest(G.aet, alt_G.et, fillvalue=0)]
    return graph


def sum_last_lay(G, fd):  # eLay += last layer of link.daggH (dsubH|ddaggH)

    ext_He = [1, [0,0,0,0,0,0], []]
    for link in G.rimH[-1] if G.rimH and isinstance(G.rimH[0],list) else G.rimH:
        if link.He:
            H = link.He[2]
            if G.aggH:
                G_depth = G.aggH[0][0]
                if G_depth == 2 and G.ext_He and G.ext_He[0] == 2:
                    H = H[2][-1][2]  # last subH
                else:
                    H = []
            # derH_/ last xcomp: len subH *= 2, maybe single dderH
            # with ext interlaced with H, this int(len(H)/2) is no longer correct?
            ext_He[2] = [add_(eHE, eHe, irdnt=link.et[2:4]) for eHE, eHe in zip_longest(H[int(len(H)/2): ], ext_He[2], fillvalue=[])]
            # sum all derHs of link layer=rdH into esubH[-1]

    if ext_He[2]: add_(G.ext_He, ext_He)


  # draft
def comp_G(link, iEt):  # add flat dderH to link and link to the rims of comparands

    _G, G = link._G, link.G  # comp: default P.ptuple and G.ext, PP.He if>1 P, G.HHe if>1 PP:
    rn = _G.n / G.n; Depth = 0
    N = 1.5  # default
    # / P:
    Et, md_ = comp_ptuple(_G.ptuple, G.ptuple, rn, fagg=1)
    dderH = [[0, [*Et], md_]]  # dderH.H
    # / PP, if >1 Ps:
    if _G.derH and G.derH:
        depth, et, dHe, n = comp_(_G.derH, G.derH, rn, fagg=1)  # dderH is generic now
        dderH += dHe  # add flat
        Et = [E+e for E,e in zip(Et,et)]  # evaluation tuple: valt, rdnt, dect
        N += n; Depth = depth
    # / G, if >1 PPs:
    if _G.aggH and G.aggH:  # as above
        depth, et, daggH, n = comp_(_G.aggH, G.aggH, rn, fagg=1)
        dderH += daggH  # still flat
        Et = [E+e for E,e in zip(Et,et)]
        N += n; Depth = depth

    et, extt = comp_ext(_G.ext, G.ext, rn)
    link.et = [E+e for E,e in zip(Et,et)]  # reset per comp_G
    link.n = N
    dderH = [Depth, Et, dderH + [[-1,et,extt]]]  # extt must be last
    if link.dderH:
        add_(link.dderH, dderH)  # existing He in der+
    else:
        link.dderH = dderH
    for fd in 0, 1:
        Val, Rdn, Dec = Et[fd::2]
        if Val > G_aves[fd] * Rdn:
            iEt[fd::2] = [V+v for V,v in zip(iEt[fd::2], Et[fd::2])]  # to eval grapht in form_graph_t
            G.Et[fd::2] = [V+v for V,v in zip(G.Et[fd::2], Et[fd::2])]
            if not fd:
                for G in link.G, link._G:
                    rimH = G.rimH
                    if rimH and isinstance(rimH[0],list):  # rim is converted to rimH in 1st sub+
                        if len(rimH) == len(G.RimH): rimH += [[]]  # no new rim layer yet
                        rimH[-1] += [link]  # rimH
                    else:
                        rimH += [link]  # rim


def comp_ext(_ext, ext, rn):  # primary ext only

    _L,_S,_A = _ext; L,S,A = ext; L/=rn; S/=rn  # angle is not normalized

    mA, dA = comp_angle(_A, A)
    mS, dS = min(_S,S)-ave_mL, _S/_L - S/L  # S is summed over L, dS is not summed over dL
    mL, dL = min(_L,L)-ave_mL, _L -L
    M = mL + mS + mA
    D = abs(dL) + abs(dS) + dA
    mrdn = M > D
    drdn = D<= M
    # all comparands are positive: maxm = maxd
    Lmax = max(L,_L); Smax = max(S,_S); Amax = .5  # max_mA = max_dA = ave_dangle
    mdec = mL/Lmax + mS/Smax + mA/Amax
    ddec = dL/Lmax + dS/Smax + dA/Amax

    return [M,D,mrdn,drdn,mdec,ddec], [mL,dL,mS,dS,mA,dA]


def feedback(root):  # called from form_graph_, append new der layers to root

    AggH, eT = deepcopy(root.fback_.pop(0))  # init
    while root.fback_:
        aggH, et = root.fback_.pop(0)
        add_(AggH, aggH)
        AggH = [add_(HE, He) for HE,He in zip_longest(AggH, aggH, fillvalue=[])]
        eT = [V+v for V,v in zip_longest(eT, et, fillvalue=0)]

    if eT[1] > G_aves[1] * eT[3]:  # compress levels?
        root.aggH = [add_(rHe, He) for rHe,He in zip_longest(root.aggH, AggH, fillvalue=[])]
        root.et = [V+v for V,v in zip_longest(root.et, eT, fillvalue=0)]

    if root.typ != 'edge':  # not Edge
        rroot = root.root
        if rroot:
            fback_ = rroot.fback_  # always node_t if feedback
            if fback_ and len(fback_) == len(rroot.node_[1]):  # after all nodes sub+
                feedback(rroot)    # sum2graph adds higher aggH, feedback adds deeper aggH layers