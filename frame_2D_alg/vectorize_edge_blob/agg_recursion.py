import numpy as np
from copy import deepcopy, copy
from itertools import combinations, zip_longest
from .classes import Cgraph, CderG, Cmd, CderH
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
Clustering criterion is G M|D, summed across >ave vars if selective comp (<ave vars are not compared, so they don't add costs).
Fork selection should be per var or co-derived der layer or agg level. 
There are concepts that include same matching vars: size, density, color, stability, etc, but in different combinations.
Weak value vars are combined into higher var, so derivation fork can be selected on different levels of param composition.

Clustering by variance: lend|borrow, contribution or counteraction to similarity | stability, such as metabolism? 
-
G is graph:
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

def vectorize_root(blob, verbose):  # vectorization in 3 composition levels of xcomp, cluster:

    edge, adj_Pt_ = slice_edge(blob, verbose)  # lateral kernel cross-comp -> P clustering

    comp_P_(edge, adj_Pt_)  # vertical, lateral-overlap P cross-comp -> PP clustering

    for fd, node_ in enumerate(edge.node_):  # always node_t
        if edge.valt[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.rdnt[fd]:

            for PP in node_: PP.roott = [None, None]
            # discontinuous PP cross-comp, cluster -> G_t
            agg_recursion(None, edge, node_, nrng=1)  # agg+, no Cgraph nodes

    return edge


def agg_recursion(rroot, root, node_, nrng=0, lenHH=-1):  # compositional agg|sub recursion in root graph, cluster G_

    Et = [[0,0],[0,0],[0,0]]
    lenH = -1  # no empty append lenHH[-1] = 0?

    nrng = rd_recursion(rroot, root, node_, Et, nrng, lenH, lenHH)  # rng+, adds rim_ as rim_t[-1][0]
    if root.link_ and isinstance(root.link_[0], CderG):  # else CderP in edge before agg+
        rd_recursion(rroot, root, root.link_, Et, 0, lenH, lenHH)  # der+, adds link_, rim_ as rim_t[-1][1]

    _GG_t = form_graph_t(root, node_, Et, nrng, lenH, lenHH)  # may convert root.node_[-1] to node_t
    GGG_t = []  # add agg+ fork tree:

    while _GG_t:  # unpack fork layers?
        GG_t, GGG_t = [],[]
        for fd, GG_ in enumerate(_GG_t):

            if not fd: nrng+=1  # also nrng+ in rd+
            if root.Vt[fd] * (len(GG_)-1)*nrng*2 > G_aves[fd] * root.Rt[fd]:
                # agg+/ node_t, vs. sub+/ node_:
                GGG_t, Vt, Rt  = agg_recursion(rroot, root, root.node[fd], nrng=1)
                if rroot:
                    rroot.fback_t[fd] += [[root.aggH, root.valt, root.rdnt, root.dect]]
                    feedback(rroot,fd)  # update root.root..
                for i in 0,1:
                    if Vt[i] > G_aves[i] * Rt[i]:
                        GGG_t += [[i, GGG_t[fd][i]]]
                        # sparse agglomerated current layer of forks across GG_tree
                        GG_t += [[i, GG_t[fd][i],1]]  # i:fork, 1:packed sub_GG_t?
                        # sparse lower layer of forks across GG_tree
                    else:
                        GGG_t += [[i, GG_t[fd][i]]]  # keep lower-composition GGs

        _GG_t = GG_t  # for next loop

    return GGG_t  # should be tree nesting lower forks


def form_graph_t(root, G_, Et, nrng, lenH=-1, lenHH=-1):  # form Gm_,Gd_ from same-root nodes

    fd = not nrng
    _G_ = []
    for G in G_:  # select Gs connected in current layer:
        if G.rim_t:  # without depth, rim_t is init as an empty list
            if lenHH >= 0: rim_tH = G.rim_t[-1][fd]  # sub+'H
            else:     rim_tH = G.rim_t[fd]  # rim_
            if len(rim_tH) > lenH: _G_ += [G]

    node_connect(_G_, lenHH)  # Graph Convolution of Correlations over init _G_
    node_t = []
    for fd in 0,1:
        if Et[0][fd] > ave * Et[1][fd]:  # eValt > ave * eRdnt: cluster
            graph_ = segment_node_(root, _G_, fd, nrng, lenH)  # fd: node-mediated Correlation Clustering
            for graph in graph_:
                if graph.Vt[fd] * (len(graph.node_)-1)*root.rng > G_aves[fd] * graph.Rt[fd]:
                    graph.fsub = 1
                    agg_recursion(root, graph, graph.node_, nrng)
                else:
                    root.fback_t[root.fd] += [[graph.aggH, graph.valt, graph.rdnt, graph.dect]]
                    # feedback(root,root.fd)  # update root.root..
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]
    if any(node_t):
        G_[:] = node_t  # else keep root.node_


def node_connect(_G_, lenHH):  # node connectivity = sum surround link vals, incr.mediated: Graph Convolution of Correlations
    '''
    Aggregate direct * indirect connectivity per node from indirect links via associated nodes, in multiple cycles.
    Each cycle adds contributions of previous cycles to linked-nodes connectivity, propagated through the network.
    Math: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/node_connect.png
    '''
    frd = lenHH != None
    while True:
        # eval accumulated G connectivity vals, indirectly extending their range
        G_ = []  # next connectivity expansion, more selective by DVt,Lent = [0,0],[0,0]?
        for G in _G_:
            uprimt = [[],[]]  # >ave updates of direct links
            for i in 0,1:
                val,rdn,dec = G.Vt[i],G.Rt[i],G.Dt[i]  # connect by last layer
                ave = G_aves[i]
                for link in unpack_rim(G.rim_t, i, lenHH):
                    # >ave derG in fd rim
                    lval,lrdn,ldec = link.Vt[i],link.Rt[i],link.Dt[i]
                    _G = link._G if link.G is G else link.G
                    _val,_rdn,_dec = _G.Vt[i],_G.Rt[i],_G.Dt[i]
                    # Vt.. for segment_node_:
                    V = ldec * (val+_val); dv = V-link.Vt[i]; link.Vt[i] = V
                    R = ldec * (rdn+_rdn); dr = R-link.Rt[i]; link.Rt[i] = R
                    D = ldec * (dec+_dec); dd = D-link.Dt[i]; link.Dt[i] = D
                    if dv > ave * dr:
                        G.Vt[i]+=V; G.Rt[i]+=R; G.Dt[i]+=D  # add link last layer vals
                        uprimt[i] += [link]
                        # more selective eval: dVt[i] += dv; L = len(uprimt[i]); Lent[i] += L
                    if V > ave * R:
                        G.evalt[i] += dv; G.erdnt[i] += dr; G.edect[i] += dd
            if any(uprimt):  # prune rim for next loop
                if frd:
                    for i in 0, 1: unpack_rim(G.rim_t,i,lenHH)[:] = uprimt[i]
                else: G.rim_t[:] = uprimt
                G_ += [G]
        if G_: _G_ = G_  # exclude weakly incremented Gs from next connectivity expansion loop
        else:  break


def unpack_rim(rim_t, fd, lenHH):
    # rim_t in agg+:  None| [mrim, drim]  | rimtH,
    # rim_t in agg++: None| [mrim_,drim_] | rim_tH

    rim_depth = get_depth(rim_t, 1)

    if rim_depth == 3:
        rim = rim_t[-1][fd][-1]  # in rim_tH
    elif rim_depth == 2:
        if lenHH==0: rim = rim_t[fd][-1]  # in rim_t, agg++
        else:        rim = rim_t[-1][fd]  # in rimtH, agg+
    elif rim_depth == 1:
        rim = rim_t[fd]  # in rimt
    else:
        rim = []  # depth = 0, empty rim

    return rim

def get_depth(rim_t, fneg=0):  # https://stackoverflow.com/questions/6039103/counting-depth-or-the-deepest-level-a-nested-list-goes-to

    list_depth = ((max(map(get_depth, rim_t))+1 if rim_t else 1) if isinstance(rim_t, list) else 0)   # -1 if input is not a list

    return list_depth - 1 if fneg else list_depth


def segment_node_(root, root_G_, fd, nrng, lenH=-1, lenHH=-1):  # eval rim links with summed surround vals for density-based clustering

    # graph += [node] if >ave (surround connectivity * relative value of link to any internal node)
    igraph_ = []; ave = G_aves[fd]

    for G in root_G_:   # init per node,  last-layer Vt,Vt,Dt:
        grapht = [[G],[], G.Vt,G.Rt,G.Dt, copy(unpack_rim(G.rim_t, fd, lenHH))]  # init link_ with last rim
        G.roott[fd] = grapht  # roott for feedback
        igraph_ += [grapht]
    _graph_ = igraph_

    while True:
        graph_ = []
        for grapht in _graph_:  # extend grapht Rim with +ve in-root links
            G_, Link_, Valt, Rdnt, Dect, Rim = grapht
            inVal, inRdn = 0,0  # new in-graph +ve
            new_Rim = []
            for link in Rim:  # unique links
                if link.G in G_:
                    G = link.G; _G = link._G
                else:
                    G = link._G; _G = link.G
                if _G in G_: continue
                # connect by rel match of nodes * match of node Vs: surround M|Ds,
                # V = how deeply inside the graph is G
                cval = link.Vt[fd] + get_match(G.Vt[fd],_G.Vt[fd])  # same coef for int and ext match?
                crdn = link.Rt[fd] + (G.Rt[fd] + _G.Rt[fd]) / 2
                if cval > ave * crdn:  # _G and its root are effectively connected
                    # merge _root:
                    _grapht = _G.roott[fd]
                    _G_,_Link_,_Valt,_Rdnt,_Dect,_Rim = _grapht
                    Link_[:] = list(set(Link_+_Link_)) + [link]
                    for g in _G_:
                        g.roott[fd] = grapht
                        if g not in G_: G_+=[g]
                    for i in 0,1:
                        Valt[i]+=_Valt[i]; Rdnt[i]+=_Rdnt[i]; Dect[i]+=_Dect[i]
                        inVal += _Valt[fd]; inRdn += _Rdnt[fd]
                    if _grapht in igraph_:
                        igraph_.remove(_grapht)
                    new_Rim += [link for link in _Rim if link not in new_Rim+Rim+Link_]
            # for next loop:
            if len(new_Rim) * inVal > ave * inRdn:
                graph_ += [[G_,Link_, Valt,Rdnt,Dect, new_Rim]]

        if graph_: _graph_ = graph_  # selected graph expansion
        else: break

    # -> Cgraphs if Val > ave * Rdn:
    return [sum2graph(root, graph, fd, nrng, lenH, lenHH) for graph in igraph_ if graph[2][fd] > ave * graph[3][fd]]


def sum2graph(root, grapht, fd, nrng, lenH=-1, lenHH=-1):  # sum node and link params into graph, aggH in agg+ or player in sub+

    G_,Link_,Vt,Rt,Dt,_ = grapht  # last-layer vals only; depth 0:derLay, 1:derHv, 2:subHv

    graph = Cgraph(fd=fd, node_=G_, L=len(G_),link_=Link_,Vt=Vt, Rt=Rt, Dt=Dt, rng=nrng)
    graph.roott[fd] = root
    for link in Link_:
        link.roott[fd]=graph
    eH, valt,rdnt,dect, evalt,erdnt,edect = [], [0,0],[0,0],[0,0], [0,0],[0,0],[0,0]  # grapht int = node int+ext
    A0, A1, S = 0,0,0
    for G in G_:
        sum_links_last_lay(G, fd, lenH, lenHH)
        graph.box += G.box
        graph.ptuple += G.ptuple
        sum_derH([graph.derH,[0,0],[1,1]], [G.derH,[0,0],[1,1]], base_rdn=1)
        sum_subHv([eH,evalt,erdnt,edect,2], [G.extH,G.evalt,G.erdnt,G.edect,2], base_rdn=G.erdnt[fd])
        sum_aggHv(graph.aggH, G.aggH, base_rdn=1)
        A0 += G.A[0]; A1 += G.A[1]; S += G.S
        for j in 0,1:
            evalt[j] += G.evalt[j]; erdnt[j] += G.erdnt[j]; edect[j] += G.edect[j]
            valt[j] += G.valt[j]; rdnt[j] += G.rdnt[j]; dect[j] += G.dect[j]

    graph.aggH += [[eH,evalt,erdnt,edect,2]]  # new derLay
    # graph internals = G Internals + Externals:
    graph.valt = Cmd(*valt) + evalt
    graph.rdnt = Cmd(*rdnt) + erdnt
    graph.dect = Cmd(*dect) + edect
    graph.A = [A0,A1]; graph.S = S

    if fd:  # assign alt graphs from d graph, after both linked m and d graphs are formed
        for link in graph.link_:
            mgraph = link.roott[0]
            if mgraph:
                for fd, (G, alt_G) in enumerate(((mgraph,graph), (graph,mgraph))):  # bilateral assign:
                    if G not in alt_G.alt_graph_:
                        G.alt_graph_ += [alt_G]
                        for i in 0,1:
                            G.avalt[i] += alt_G.valt[i]; G.ardnt[i] += alt_G.rdnt[i]; G.adect[i] += alt_G.dect[i]
    return graph

# draft:
def sum_links_last_lay(G, fd, lenH, lenHH):  # eLay += last_lay/ link, lenHH: daggH, lenH: dsubH

    eLay = []  # accum from links' dderH or dsubH; rim_t,link.daggH nesting per depth:

    for link in unpack_rim(G.rim_t, fd, lenHH):  # links in rim
        if link.daggH:
            dsubH = []  # default
            daggH = link.daggH
            if G.fHH:
                if len(daggH) > lenHH:
                    if lenH: dsubH = daggH[-1]
                    else: dderH = daggH[-1]
            elif lenH: dsubH = daggH
            else: dderH = daggH
            if dsubH:
                if len(dsubH) > lenH:
                    for dderH in dsubH[ int(len(dsubH)/2): ]:  # derH_/ last xcomp: len subH *= 2
                        sum_derHv(eLay, dderH, base_rdn=link.Rt[fd])  # sum all derHs of link layer=rdH into esubH[-1]
            else:
                sum_derHv(eLay, dderH, base_rdn=link.Rt[fd])

        G.evalt[fd] += link.Vt[fd]; G.erdnt[fd] += link.Rt[fd]; G.edect[fd] += link.Dt[fd]
    G.extH += [eLay]


def comp_G(link, Et, lenH=0, lenHH=None):  # lenH in sub+|rd+, lenHH in agg_compress sub+ only

    _G, G = link._G, link.G
    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0, 1,1, 0,0
    # keep separate P ptuple and PP derH, empty derH in single-P G, + empty aggH in single-PP G:

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
    dertv = [[mtuple,dtuple], [mval,dval],[mrdn,drdn],[dect[0]/6,dect[1]/6]]
    Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn; Mdec+=dect[0]/6; Ddec+=dect[1]/6  # ave of 6 params

    # / PP:
    dderH = []
    if _G.derH and _G.derH:  # empty in single-node Gs
        for _lay, lay in zip(_G.derH,_G.derH):
            mtuple,dtuple, Mtuple,Dtuple = comp_dtuple(_lay[1], lay[1], rn=1, fagg=1)
            mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
            mrdn = dval > mval; drdn = dval < mval
            mdec, ddec = 0, 0
            for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
                for (par, max, ave) in zip(ptuple, Ptuple, aves):  # different ave for comp_dtuple
                    if fd: ddec += par/max if max else 1
                    else:  mdec += (par+ave)/(max) if max else 1
            mdec /= 6; ddec /= 6
            Mval+=dval; Dval+=mval; Mdec=(Mdec+mdec)/2; Ddec=(Ddec+ddec)/2
            dderH += [[[mtuple,dtuple], [mval,dval],[mrdn,drdn],[mdec,ddec]]]

    # / G:
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec])
    dderH = [[dertv]+dderH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec], der_ext]

    if _G.aggH and G.aggH:
        daggH, valt,rdnt,dect = comp_aggHv(_G.aggH, G.aggH, rn=1)
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
        Mdec = (Mdec+dect[0])/2; Ddec = (Ddec+dect[1])/2
        # flat, appendleft:
        dsubH = [[dderH]+daggH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]]
    else:
        dsubH =  dderH
    link.Vt,link.Rt,link.Dt = Valt,Rdnt,Dect = [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]  # reset per comp_G

    for fd, (Val,Rdn,Dec) in enumerate(zip(Valt,Rdnt,Dect)):
        if Val > G_aves[fd] * Rdn:
            # eval fork grapht in form_graph_t:
            Et[0][fd] += Val; Et[1][fd] += Rdn; Et[2][fd] += Dec
            append_rim(link, dsubH, Val,Rdn,Dec, fd, lenH, lenHH)

# draft
def append_rim(link, dsubH, Val,Rdn,Dec, fd, lenH, lenHH):

    for G in link._G, link.G:
        rim_t = G.rim_t
        if rim_t:
            if G.fHH:  # rim_t= rim_tH
                if len(rim_t) < lenHH:  # rim_tH += rim_t
                    if fd:
                        dsubH = [dsubH]  # convert dsubH to daggH?
                        rim_t += [[[],[link]]]  # add rimt ! rim_t: no der+ yet
                    else:
                        rim_t += [[[link],[]]]
                else:  # append last rim_t[fd]
                    if len(rim_t[-1][fd]) < lenH:  # append rim_ vs. rim
                        if lenH==2:  # 1st der+,
                            # this is done in rd_recursion?:
                            if fd: dsubH = [dsubH]  # convert dderH to dsubH?
                            rim_t[:] = [[rim_t[0]],[rim_t[1]]]  # convert rimt to rim_t
                        rim_t[fd][-1] += [link]  # init empty
                    else:
                        rim_t[fd] += [link]  # rimt
            else:  # rimt | rim_t
                # same as above, make function:
                if len(rim_t[-1][fd]) < lenH:  # append rim_ vs. rim
                    if lenH == 2:  # 1st der+,
                        dsubH = [dsubH]  # convert dsubH to daggH?
                        rim_t[:] = [[rim_t[0]], [rim_t[1]]]  # convert rimt to rim_t
                    rim_t[fd][-1] += [link]  # init empty
                else:
                    rim_t[fd] += [link]  # rimt

            if fd: link.daggH += [dsubH]  # new layer with added nesting
            G.Vt[fd] += Val; G.Rt[fd] += Rdn; G.Dt[fd] += Dec


def comp_aggHv(_aggH, aggH, rn):  # no separate ext

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    SubH = []

    for _lev, lev in zip(_aggH, aggH):  # compare common subHs, if lower-der match?
        if _lev and lev:
            dsubH, valt,rdnt,dect = comp_subHv(_lev[0],lev[0], rn)
            SubH += dsubH  # concat
            Mdec += dect[0]; Ddec += dect[1]
            mval,dval = valt; Mval += mval; Dval += dval
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + mval <= dval
    if SubH:
        S = min(len(_aggH),len(aggH)); Mdec/= S; Ddec /= S  # normalize

    return SubH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]

# pending update on subHv_t
def comp_subHv(_subH, subH, rn):

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    dsubH =[]

    for _lay, lay in zip(_subH, subH):  # compare common lower layer|sublayer derHs, if prior match?

        dderH, valt,rdnt,dect = comp_derHv(_lay[0],lay[0], rn)  # derHv: [derH, valt, rdnt, dect, extt, 1]:
        dextt = [comp_ext(_ext,ext,[Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]) for _ext,ext in zip(_lay[-2],lay[-2])]

        dsubH += [[dderH, valt,rdnt,dect,dextt, 1]]  # flat
        Mdec += dect[0]; Ddec += dect[1]
        mval,dval = valt; Mval += mval; Dval += dval
        Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + dval <= mval
    if dsubH:
        S = min(len(_subH),len(subH)); Mdec/= S; Ddec /= S  # normalize

    return dsubH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]  # new layer,= 1/2 combined derH


def comp_derHv(_derH, derH, rn):  # derH is a list of der layers or sub-layers, each is ptuple_tv

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    dderH =[]

    for _lay, lay in zip(_derH,derH):
        # comp dtuples, eval mtuples:
        mtuple,dtuple, Mtuple,Dtuple = comp_dtuple(_lay[0][1], lay[0][1], rn, fagg=1)
        mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
        mrdn = dval > mval; drdn = dval < mval
        dect = [0,0]
        for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
            for (par, max, ave) in zip(ptuple, Ptuple, aves):  # different ave for comp_dtuple
                if fd: dect[fd] += par/max if max else 1
                else:  dect[fd] += (par+ave)/(max) if max else 1
        dect[0]/=6; dect[1]/=6
        dderH += [[[mtuple,dtuple], [mval,dval],[mrdn,drdn],dect, 0]]
        Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn
        Mdec+=dect[0]; Ddec+=dect[1]

    if dderH:
        S = min(len(_derH),len(derH)); Mdec /= S; Ddec /= S  # normalize

    return dderH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]  # new derLayer,= 1/2 combined derH


def sum_aggHv(AggH, aggH, base_rdn):

    if aggH:
        if AggH:
            for Layer, layer in zip_longest(AggH,aggH, fillvalue=[]):
                sum_subHv(Layer, layer, base_rdn)
        else:
            AggH[:] = deepcopy(aggH)

# pending update on subHv_t
def sum_subHv(T, t, base_rdn, fneg=0):

    if t:
        if T:
            SubH,Valt,Rdnt,Dect = T; subH,valt,rdnt,dect = t
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i]+dect[i])/2
            if SubH:
                for Layer, layer in zip_longest(SubH,subH, fillvalue=[]):
                    sum_derHv(Layer, layer, base_rdn, fneg)  # _lay[0][0] is mL
            else:
                SubH[:] = deepcopy(subH)
        else:
            T[:] = deepcopy(t)


def sum_derHv(T,t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    if t:
        if T:
            DerH, Valt, Rdnt, Dect, Extt_ = T
            derH, valt, rdnt, dect, extt_ = t
            for Extt, extt in zip(Extt_,extt_):
                sum_ext(Extt, extt)
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i] + dect[i])/2
            DerH[:] = [
                [[sum_dertuple(Dertuple,dertuple, fneg*i) for i,(Dertuple,dertuple) in enumerate(zip(Tuplet,tuplet))],
                  [V+v for V,v in zip(Valt,valt)], [R+r+base_rdn for R,r in zip(Rdnt,rdnt)], [(D+d)/2 for D,d in zip(Dect,dect)],
                ]
                for [Tuplet,Valt,Rdnt,Dect], [tuplet,valt,rdnt,dect]  # ptuple_tv
                in zip_longest(DerH, derH, fillvalue=[([0,0,0,0,0,0],[0,0,0,0,0,0]), (0,0),(0,0),(0,0)])
            ]

        else:
            T[:] = deepcopy(t)

def sum_ext(Extt, extt):

    if isinstance(Extt[0], list):
        for Ext,ext in zip(Extt,extt):  # ext: m,d tuple
            for i,(Par,par) in enumerate(zip(Ext,ext)):
                Ext[i] = Par+par
    else:  # single ext
        for i in range(3): Extt[i]+=extt[i]  # sum L,S,A


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

    # ave dec = ave (ave dec, ave L,S,A dec):
    Dect[0] = ((mL / max(aL,_aL) if aL or _aL else 1 +
                mS / max(aS,_aS) if aS or _aS else 1 +
                mA / max_mA if max_mA else 1) /3
                + Dect[0]) / 2
    Dect[1] = ((dL / (_aL+aL) if aL+_aL else 1 +
                dS / (_aS+aS) if aS+_aS else 1 +
                dA / max_dA if max_mA else 1) /3
                + Dect[1]) / 2

    return [[mL,mS,mA], [dL,dS,dA]]


def feedback(root, fd):  # called from form_graph_, append new der layers to root

    AggH, Valt, Rdnt, Dect = deepcopy(root.fback_t[fd].pop(0))
    # init with 1st tuple
    while root.fback_t[fd]:
        aggH, valt, rdnt, dect = root.fback_t[fd].pop(0)
        sum_aggHv(AggH, aggH, base_rdn=0)
        for i in 0,1:
            Valt[i] += valt[i]; Rdnt[i] += rdnt[i]; Dect[i] += dect[i]
            #-> root.fback_t
    if Valt[fd] > G_aves[fd] * Rdnt[fd]:  # or compress each level?
        root.aggH += AggH  # higher levels are not affected
        for j in 0,1:  # sum both in same root fork
            root.valt[fd] += Valt[j]; root.rdnt[fd] += Rdnt[j]; root.dect[fd] += Dect[j]

    if isinstance(root.roott, list):  # not Edge
        rroot = root.roott[fd]
        if rroot:
            fback_ = rroot.fback_t[fd]  # always node_t for feedback
            if fback_ and len(fback_) == len(rroot.node_[fd]):  # all nodes sub+ terminated
                feedback(rroot, fd)  # sum2graph adds higher aggH, feedback adds deeper aggH layers


# draft:
def rd_recursion(rroot, root, Q, Et, nrng=1, lenH=0, lenHH=0):  # rng,der incr over same G_,link_ -> fork tree, represented in rim_t

    fd = not nrng; link_ = []; ave = G_aves[fd]  # this ave can be rmeoved now?
    et = [[0,0],[0,0],[0,0]]  # grapht link_' eValt, eRdnt, eDect(currently not used)
    '''
    link.subH[-1][fd] += [layer] per rng+|der+, after fd recursion is terminated: 
    G.subH[-1][fd] += consecutive same-fd link layers in single clustering layer
    '''
    if fd:  # der+
        G_ = []
        for link in Q:  # inp_= root.link_, reform links
            if link.Vt[1] > G_aves[1] * link.Rt[1]:  # >rdn incr
                if lenH==1:  # 1st der+: dderH-> dsubH
                    link.subH = [link.subH]
                comp_G(link, Et, lenH+1, lenHH)
                if link.G not in G_: G_ += [link.G]
                if link._G not in G_: G_ += [link._G]
    else:  # rng+
        G_ = Q
        for _G, G in combinations(G_, r=2):  # form new link_ from original node_
            dy = _G.box.cy - G.box.cy; dx = _G.box.cx - G.box.cx
            dist = np.hypot(dy, dx)
            # max distance between node centers, init=2
            if 2*nrng > dist > 2*(nrng-1):  # G,_G are within rng and were not compared in prior rd+
                link = CderG(_G=_G, G=G)
                comp_G(link, et, lenH, lenHH)

    if et[0][fd] > ave_Gm * et[1][fd]:  # single layer accum
        for Part, part in zip(Et, et):
            for i, par in enumerate(part):  # Vt[i]+=v; Rt[i]+=rt[i]; Dt[i]+=d:
                Part[i] += par
        if fd:
            for G in G_:
                for link in unpack_rim(G.rim_t, fd, lenHH):
                    if len(link.subH[0][-1]) > lenH:  # link.subH was appended in this rd cycle
                        link_ += [link]  # for next rd cycle
        else:
            pruned_G_ = []
            for G in G_:
                if G.rim_t:
                    rim_t = G.rim_t
                    if lenHH>=0:
                        G_lenH = len(rim_t[-1][fd])  # agg++ : G.rim_t is rim_tH or rimtH
                    if lenH>=0:  # G.rim_t is rim_t
                        G_lenH = len(rim_t[fd])
                    else:  # lenH == 0: G.rim_t is rimt
                        G_lenH = len(rim_t[fd])
                    if G_lenH > lenH:
                        pruned_G_ += [G]  # remove if empty rim_t

        rd_recursion(rroot, root, link_ if fd else pruned_G_, Et, 0 if fd else nrng+1, lenH+1, lenHH)

    return nrng