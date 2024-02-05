import numpy as np
from copy import deepcopy, copy
from itertools import combinations, zip_longest
from .classes import Cgraph, CderG, Cmd, CderH, Cangle
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

def vectorize_root(blob, verbose):  # vectorization in 3 composition levels of xcomp, cluster:

    edge, adj_Pt_ = slice_edge(blob, verbose)  # lateral kernel cross-comp -> P clustering

    comp_P_(edge, adj_Pt_)  # vertical, lateral-overlap P cross-comp -> PP clustering

    for fd, node_ in enumerate(edge.node_):  # always node_t
        if edge.valt[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.rdnt[fd]:

            for PP in node_: PP.roott = [None, None]
            # discontinuous PP cross-comp, cluster -> G_t
            agg_recursion(None, edge, node_, nrng=1, fagg=1)  # agg+, no Cgraph nodes

    return edge


def agg_recursion(rroot, root, node_, nrng=1, fagg=0):  # lenH = len(root.aggH[-1][0]), lenHH: same in agg_compress

    Et = [[0,0],[0,0],[0,0]]  # grapht link_ Valt,Rdnt,Dect(currently not used)

    # agg+ der=1 xcomp of new Gs if fagg, else sub+: der+ xcomp of old Gs:
    nrng = rng_recursion(rroot, root, combinations(node_,r=2) if fagg else root.link_, Et, nrng)  # rng+ appends rim, link.derH

    form_graph_t(root, node_, Et, nrng, fagg)  # root_fd, eval sub+, feedback per graph

    if node_ and isinstance(node_[0], list):
        for i, G_ in enumerate(node_):
            if root.valt[i] * (len(G_)-1)*root.rng > G_aves[i] * root.rdnt[i]:
                # agg+ / node_t, vs. sub+ / node_:
                agg_recursion(rroot, root, G_, nrng=1, fagg=1)
                if rroot:
                    rroot.fback_t[i] += [[root.aggH,root.valt,root.rdnt,root.dect]]
                    feedback(rroot,i)  # update root.root..

def rng_recursion(rroot, root, Q, Et, nrng=1):  # rng++/ G_, der+/ link_ if called from sub+ fork of agg_recursion, -> rimH

    et = [[0,0],[0,0],[0,0]]  # grapht link_ Valt,Rdnt,Dect
    _G_,_link_ = set(),set()  # for next rng+ | sub+
    fd = isinstance(Q,set)  # link_ is a set

    if fd:  # der+, but recursion is still rng+
        for link in Q:  # inp_= root.link_, reform links
            if link.Vt[1] > G_aves[1] * link.Rt[1]:  # >rdn incr
                comp_G(link, Et)
                comp_rim(_link_, link, nrng)  # add matching-direction rim links for next rng+
    else:  # rng+, only before sub+
        Gt_ = Q
        for (_G, G) in Gt_:  # form new link_ from original node_
            dy = _G.box.cy - G.box.cy; dx = _G.box.cx - G.box.cx
            dist = np.hypot(dy, dx)
            if 2*nrng >= dist > 2*(nrng-1):  # G,_G are within rng and were not compared at prior rng
                # pairwise eval in rng++, or directional?
                if nrng==1 or (G.Vt[0]+_G.Vt[0]) > ave * (G.Rt[0]+_G.Rt[0]):
                    link = CderG(_G=_G, G=G, A=Cangle(dy,dx), S=dist)
                    comp_G(link, et)
                else:
                    _G_.add((_G, G))  # for next rng+
    '''
    recursion eval per original cluster because comp eval is bilateral, we need to test all pairs?
    '''
    if et[0][0] > ave_Gm * et[1][0]:  # eval single layer accum, for rng++ only
        for Part, part in zip(Et, et):
            for i, par in enumerate(part):  # Vt[i]+=v; Rt[i]+=rt[i]; Dt[i]+=d:
                Part[i] += par
        _Q = _link_ if fd else list(_G_)
        if _Q:
            nrng = rng_recursion(rroot, root, _Q, Et, nrng+1)  # eval rng+ for der+ too

    return nrng

def comp_rim(_link_, link, nrng):  # for next rng+:

    for G in link._G, link.G:
        for _link in G.rimH[-1]:
            _G = _link.G if _link.G in [link._G,link.G] else _link.G  # new to link

            dy = _G.box.cy - G.box.cy; dx = _G.box.cx - G.box.cx
            dist = np.hypot(dy, dx)  # max distance between node centers, init=2
            if 2*nrng > dist > 2*(nrng-1):  # potentially connected G,_G: within rng and not compared in prior rng

                # compare direction of link to all links in link.G.rimH[-1] and link._G.rimH[-1]:
                if comp_angle(link.A,_link.A)[0] > ave:
                    _link_.add(CderG(G=G, _G=_G, A=Cangle(dy,dx), S=dist))  # to compare in rng+


def form_graph_t(root, G_, Et, nrng, fagg=0):  # form Gm_,Gd_ from same-root nodes

    node_connect(G_)  # Graph Convolution of Correlations over init _G_
    node_t = []
    for fd in 0, 1:
        if Et[0][fd] > ave * Et[1][fd]:  # eValt > ave * eRdnt
            graph_ = segment_node_(root, G_, fd, nrng, fagg)  # fd: node-mediated Correlation Clustering
            if fd:  # der+ only, rng+ exhausted before sub+, can't be effectively extended in sub Gs?
                for graph in graph_:
                    if graph.link_ and graph.Vt[1] > G_aves[1] * graph.Rt[1]:
                        graph.Vt = [0,0]; graph.Rt = [0,0]; graph.Dt = [0,0]  # reset
                        node_ = graph.node_
                        if isinstance(node_[0].rimH[0],CderG):  # 1st sub+, same rim nesting?
                            for node in node_: node.rimH = [node.rimH]  # rim -> rimH
                        agg_recursion(root, graph, graph.node_, nrng, fagg=0)
                    else:
                        root.fback_t[root.fd] += [[graph.aggH, graph.valt, graph.rdnt, graph.dect]]
                        feedback(root,root.fd)  # update root.root.. per sub+
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
                val,rdn,dec = G.Vt[i],G.Rt[i],G.Dt[i]  # connect by last layer
                ave = G_aves[i]
                for link in rim:
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
                        if link not in uprim: uprim += [link]
                        # more selective eval: dVt[i] += dv; L=len(uprim); Lent[i] += L
                    if V > ave * R:
                        G.evalt[i] += dv; G.erdnt[i] += dr; G.edect[i] += dd
            if uprim:  # prune rim for next loop
                rim[:] = uprim
                G_ += [G]
        if G_: _G_ = G_  # exclude weakly incremented Gs from next connectivity expansion loop
        else:  break


def segment_node_(root, root_G_, fd, nrng, fagg):  # eval rim links with summed surround vals for density-based clustering

    # graph += [node] if >ave (surround connectivity * relative value of link to any internal node)
    igraph_ = []; ave = G_aves[fd]

    for G in root_G_:   # init per node,  last-layer Vt,Vt,Dt:
        grapht = [[G],[], G.Vt,G.Rt,G.Dt, copy(G.rimH[-1] if G.rimH and isinstance(G.rimH[0],list) else G.rimH)]  # link_ = last rim
        G.roott[fd] = grapht  # roott for feedback
        igraph_ += [grapht]
    _graph_ = igraph_

    while True:
        graph_ = []
        for grapht in _graph_:  # extend grapht Rim with +ve in-root links
            G_, Link_, Vt, Rt, Dt, Rim = grapht
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
                cval = link.Vt[fd] + get_match(G.Vt[fd],_G.Vt[fd])  # same coef for int and ext match?
                crdn = link.Rt[fd] + (G.Rt[fd] + _G.Rt[fd]) / 2
                if cval > ave * crdn:  # _G and its root are effectively connected
                    # merge _root:
                    _grapht = _G.roott[fd]
                    _G_,_Link_,_Vt,_Rt,_Dt,_Rim = _grapht
                    Link_[:] = list(set(Link_+_Link_)) + [link]
                    for g in _G_:
                        g.roott[fd] = grapht
                        if g not in G_: G_+=[g]
                    for i in 0,1:
                        Vt[i]+=_Vt[i]; Rt[i]+=_Rt[i]; Dt[i]+=_Dt[i]
                        inVal += _Vt[fd]; inRdn += _Rt[fd]
                    if _grapht in igraph_:
                        igraph_.remove(_grapht)
                    new_Rim += [link for link in _Rim if link not in new_Rim+Rim+Link_]
            # for next loop:
            if len(new_Rim) * inVal > ave * inRdn:
                graph_ += [[G_,Link_, Vt,Rt,Dt, new_Rim]]

        if graph_: _graph_ = graph_  # selected graph expansion
        else: break

    graph_ = []
    for grapht in igraph_:
        if grapht[2][fd] > ave * grapht[3][fd]:  # form Cgraphs if Val > ave* Rdn
            graph_ += [sum2graph(root, grapht[:5], fd, nrng, fagg)]

    return graph_


def sum2graph(root, grapht, fd, nrng, fagg):  # sum node and link params into graph, aggH in agg+ or player in sub+

    G_,Link_,Vt,Rt,Dt = grapht

    graph = Cgraph(fd=fd, node_=G_,link_=set(Link_), Vt=Vt,Rt=Rt,Dt=Dt, rng=nrng)
    graph.roott[fd] = root
    for link in Link_:
        link.roott[fd]=graph
        graph.ext[1] += link.S
        graph.ext[2] += link.A
    extH, valt,rdnt,dect, evalt,erdnt,edect = [], [0,0],[0,0],[0,0], [0,0],[0,0],[0,0]
    # grapht int = node int+ext
    for G in G_:
        graph.ext[0] += 1
        sum_last_lay(G, fd)
        graph.box += G.box
        graph.ptuple += G.ptuple
        sum_derH([graph.derH,[0,0],[1,1]], [G.derH,[0,0],[1,1]], base_rdn=1)
        sum_Hv([extH,evalt,erdnt,edect,2], [G.extH,G.evalt,G.erdnt,G.edect,2], base_rdn=G.erdnt[fd])
        sum_H(graph.aggH, G.aggH, depth=2, base_rdn=1)
        for j in 0,1:
            evalt[j] += G.evalt[j]; erdnt[j] += G.erdnt[j]; edect[j] += G.edect[j]
            valt[j] += G.valt[j]; rdnt[j] += G.rdnt[j]; dect[j] += G.dect[j]
    graph.aggH += extH  # concat dsubH or daggH
    # graph internals = G Internals + Externals:
    valt = Cmd(*valt) + evalt; graph.valt = valt
    rdnt = Cmd(*rdnt) + erdnt; graph.rdnt = rdnt
    dect = Cmd(*dect) + edect; graph.dect = dect
    if fagg:
        if not G.aggH or (G_[0].aggH and G_[0].aggH[-1] == 1):
            # 1st agg+, init aggH = [subHv]:
            graph.aggH = [[graph.aggH,valt,rdnt,dect,2]]

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


def sum_last_lay(G, fd):  # eLay += last layer of link.daggH (dsubH|ddaggH)
    eLay = []

    for link in G.rimH[-1] if G.rimH and isinstance(G.rimH[0],list) else G.rimH:
        if link.daggH:
            daggH = link.daggH
            if G.aggH:
                if (G.aggH[-1] == 2) and (len(daggH) > len(G.extH)):
                    dsubH = daggH[-1][0]  # last subH
                else:
                    dsubH = []
            else:  # from sub+
                dsubH = daggH
            for dderH in dsubH[ int(len(dsubH)/2): ]:  # derH_/ last xcomp: len subH *= 2, maybe single dderH
                sum_Hv(eLay, dderH, base_rdn=link.Rt[fd])  # sum all derHs of link layer=rdH into esubH[-1]

    if eLay: G.extH += [eLay]


# below is not revised:

def comp_G(link, Et):

    _G, G = link._G, link.G
    Mval,Dval, Mrdn,Drdn, Mdec,Ddec, Ext = 0,0, 1,1, 0,0, []
    # keep separate P ptuple and PP derH, empty derH in single-P G, + empty aggH in single-PP G:

    # / P:
    mtuple, dtuple, Mtuple, Dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1, fagg=1)
    mval, dval = sum(mtuple), sum(abs(d) for d in dtuple)  # mval is signed, m=-min in comp x sign
    mrdn = dval>mval; drdn = dval<=mval
    dect = [0,0]
    for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
        for i, (par, max, ave) in enumerate(zip(ptuple, Ptuple, aves)):
            # compute link decay coef: par/ max(self/same)
            if fd: dect[fd] += abs(par)/ abs(max) if max else 1
            else:  dect[fd] += (par+ave)/ (max+ave) if max else 1
    dertv = CderH([[mtuple,dtuple], [mval,dval],[mrdn,drdn],[dect[0]/6,dect[1]/6],0])  # no ext in dertv
    Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn;  Mdec+=dect[0]/6; Ddec+=dect[1]/6  # ave of 6 params
    # not updated:
    # / PP:
    dderH = []
    if _G.derH and _G.derH:  # empty in single-P Gs?
        for _lay, lay in zip(_G.derH,_G.derH):
            mtuple,dtuple, Mtuple,Dtuple = comp_dtuple(_lay[1], lay[1], rn=1, fagg=1)
            mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
            mrdn = dval > mval; drdn = dval < mval
            mdec, ddec = 0, 0
            for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
                for (par, max, ave) in zip(ptuple, Ptuple, aves):  # different ave for comp_dtuple
                    if fd: ddec += abs(par)/ abs(max) if max else 1
                    else:   mdec += (par+ave)/ (max+ave) if max else 1
            mdec /= 6; ddec /= 6
            Mval+=dval; Dval+=mval; Mdec=(Mdec+mdec)/2; Ddec=(Ddec+ddec)/2
            dderH += [CderH([[mtuple,dtuple], [mval,dval],[mrdn,drdn],[mdec,ddec],0])]

    # / G:
    der_ext,(mval, dval),(mrdn, drdn),(mdec, ddec) = comp_ext(_G.ext,G.ext)
    Mval += mval; Dval += dval; Mrdn += mrdn; Drdn += drdn; Mdec += mdec; Ddec += ddec
    dderH = [[dertv]+dderH+[der_ext], [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec], 1]
    sum_ext(Ext, der_ext)

    if _G.aggH and G.aggH:
        daggH, valt,rdnt,dect = comp_Hv([_G.aggH,_G.valt,_G.rdnt,_G.dect,2], [G.aggH,G.valt,G.rdnt,G.dect,2], rn=1)
        # aggH is subH before agg+
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0] + dval>mval; Drdn += rdnt[1] + dval<=mval
        Mdec = (Mdec+dect[0])/2; Ddec = (Ddec+dect[1])/2
        # flat, appendleft:
        daggH = [[dderH]+daggH+[Ext], [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec],2]
    else:
        daggH = dderH
    link.daggH += [daggH]

    link.Vt,link.Rt,link.Dt = Valt,Rdnt,Dect = [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]  # reset per comp_G

    for fd, (Val,Rdn,Dec) in enumerate(zip(Valt,Rdnt,Dect)):
        if Val > G_aves[fd] * Rdn:
            # to eval fork grapht in form_graph_t:
            Et[0][fd] += Val; Et[1][fd] += Rdn; Et[2][fd] += Dec
            G.Vt[fd] += Val;  G.Rt[fd] += Rdn;  G.Dt[fd] += Dec
            if not fd:
                for G in link.G, link._G:
                    rimH = G.rimH
                    if rimH and isinstance(rimH[0],list):  # rim is converted to rimH in 1st sub+
                        if len(rimH) == len(G.RimH): rimH += [[]]  # no new rim layer yet
                        rimH[-1] += [link]  # rimH
                    else:
                        rimH += [link]  # rim


def comp_Hv(_Hv, Hv, rn):
    _H, H = _Hv[0], Hv[0]
    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    DH = []

    for _lev, lev in zip(_H, H):  # compare common subHs, if lower-der match?
        if _lev and lev:
            if lev[-1] == 0:  # derlay with ptuplet
                mtuple,dtuple, Mtuple,Dtuple = comp_dtuple(_lev[0][1], lev[0][1], rn, fagg=1)
                mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
                mrdn = dval > mval; drdn = dval < mval
                dect = [0,0]
                for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
                    for (par, max, ave) in zip(ptuple, Ptuple, aves):  # different ave for comp_dtuple
                        if fd: dect[fd] += abs(par)/abs(max) if max else 1
                        else:  dect[fd] += (par+ave)/abs(max)+ave if max else 1
                dect[0]/=6; dect[1]/=6
                DH += [[[mtuple,dtuple], [mval,dval],[mrdn,drdn],dect,0]]
            elif lev[-1] == 1:  # derHv with depth == 1
                dH,(mval,dval),(mrdn,drdn),dect = comp_Hv(_lev,lev, rn)
                DH += [[dH, (mval,dval),(mrdn,drdn),dect,1]]
            elif lev[-1] == 2:  # subHv with depth == 2
                dH,(mval,dval),(mrdn,drdn),dect = comp_Hv(_lev,lev, rn)
                DH += dH  # concat
            else:  # ext
                dextt,(mval,dval),(mrdn,drdn),dect = comp_ext_(_lev,lev)  # Valt ,Rdnt and Dect are summed in comp_ext
                DH += [dextt]

            Mdec += dect[0]; Ddec += dect[1]
            Mval += mval; Dval += dval
            Mrdn += mrdn + dval > mval; Drdn += drdn + mval <= dval

    if DH:
        S = min(len(_H),len(H)); Mdec/= S; Ddec /= S  # normalize

    return DH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]

'''
def comp_aggHv(_aggHv, aggHv, rn):  # no separate ext

    _aggH, aggH = _aggHv[0], aggHv[0]
    Mval,Dval, Mrdn,Drdn, Mdec,Ddec,Ext = 0,0,1,1,0,0,[]
    SubH = []

    for _lev, lev in zip(_aggH, aggH):  # compare common subHs, if lower-der match?
        if _lev and lev:

            if lev[-1] == 1:  # derHv with depth == 1
                dderH, valt,rdnt,dect, dextt = comp_derHv(_lev,lev, rn)
                SubH += [[dderH, valt,rdnt,dect,dextt,1]]  # should be 1 here on derHv
            else:  # subHv with depth == 2
                dsubH, valt,rdnt,dect,dextt = comp_subHv(_lev,lev, rn)
                SubH += dsubH  # concat
            Mdec += dect[0]; Ddec += dect[1]
            mval,dval = valt; Mval += mval; Dval += dval
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + mval <= dval
            sum_ext(Ext, dextt)

    if SubH:
        S = min(len(_aggH),len(aggH)); Mdec/= S; Ddec /= S  # normalize

    dextt, Valt, Rdnt, Dect = comp_ext_(_aggHv[-2],aggHv[-2], [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec])
    sum_ext(Ext, dextt)

    return SubH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec], Ext


def comp_subHv(_subHv, subHv, rn):

    _subH, subH = _subHv[0], subHv[0]
    Mval,Dval, Mrdn,Drdn, Mdec,Ddec, Ext = 0,0,1,1,0,0, []
    dsubH =[]

    for _lay, lay in zip(_subH, subH):  # compare common lower layer|sublayer derHs, if prior match?

        dderH, valt,rdnt,dect, dextt = comp_derHv(_lay,lay, rn)  # derHv: [derH, valt, rdnt, dect, extt]:
        dsubH += [[dderH, valt,rdnt,dect,dextt, 1]]  # flat
        Mdec += dect[0]; Ddec += dect[1]
        mval,dval = valt; Mval += mval; Dval += dval
        Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + dval <= mval
        sum_ext(Ext, dextt)
    if dsubH:
        S = min(len(_subH),len(subH)); Mdec/= S; Ddec /= S  # normalize

    dextt, Valt, Rdnt, Dect = comp_ext_(_subHv[-2],subHv[-2], [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec])
    sum_ext(Ext, dextt)

    return dsubH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec], Ext  # new layer,= 1/2 combined derH


def comp_derHv(_derHv, derHv, rn):  # derH is a list of der layers or sub-layers, each is ptuple_tv

    _derH, derH = _derHv[0], derHv[0]
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
                if fd: dect[fd] += abs(par)/abs(max) if max else 1
                else:  dect[fd] += (par+ave)/abs(max)+ave if max else 1
        dect[0]/=6; dect[1]/=6
        dderH += [[[mtuple,dtuple], [mval,dval],[mrdn,drdn],dect,0]]
        Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn
        Mdec+=dect[0]; Ddec+=dect[1]

    if dderH:
        S = min(len(_derH),len(derH)); Mdec /= S; Ddec /= S  # normalize

    dextt, Valt, Rdnt, Dect = comp_ext_(_derHv[-2],derHv[-2], [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec])

    return dderH, Valt,Rdnt,Dect, dextt  # new derLayer,= 1/2 combined derH
'''

# tentative
def sum_Hv(Hv, hv, base_rdn, fneg=0):

    if hv:
        if Hv:
            H,Valt,Rdnt,Dect,depth = Hv; h,valt,rdnt,dect,depth = hv
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i]+dect[i])/2
            sum_H(H, h, depth, base_rdn)
        else:
           Hv[:] = deepcopy(hv)


def sum_H(H, h, depth, base_rdn, fneg=0):

    if h:
        if H:
            for Layer, layer in zip_longest(H,h, fillvalue=None):
                if layer:
                    if Layer:
                        if isinstance(layer[-1], list):  # ext
                            sum_ext(Layer, layer)

                        elif layer[-1] == 0:  # derHv with depth == 1
                            Tuplet,Valt,Rdnt,Dect,_ = Layer
                            tuplet,valt,rdnt,dect,_ = layer
                            for i in 0,1:
                                sum_dertuple(Tuplet[i],tuplet[i], fneg*i)
                                Valt[i] += valt[i]
                                Rdnt[i] += rdnt[i]
                                Dect[i] += dect[i]
                        else:  # subHv, aggHv
                             sum_Hv(Layer, layer, base_rdn, fneg)
                    elif Layer != None:
                        Layer[:] = deepcopy(layer)
                    else:
                        H += [deepcopy(layer)]
        else:
            H[:] = deepcopy(h)



'''
# looks like we only need sum_aggH instead of sum_aggHv
def sum_aggH(AggH, aggH, base_rdn):

    if aggH:
        if AggH:
            for Layer, layer in zip_longest(AggH,aggH, fillvalue=[]):
                if layer[-1] == 1:  # derHv with depth == 1
                    sum_derHv(Layer, layer, base_rdn)
                else:  # subHv
                    sum_subHv(Layer, layer, base_rdn)
        else:
            AggH[:] = deepcopy(aggH)


def sum_aggHv(T, t, base_rdn):

    if t:
        if T:
            AggH,Valt,Rdnt,Dect,_ = T; aggH,valt,rdnt,dect,_ = t
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i]+dect[i])/2
            if AggH:
                for Layer, layer in zip_longest(AggH,aggH, fillvalue=[]):
                    if layer[-1] == 1:  # derHv with depth == 1
                        sum_derHv(Layer, layer, base_rdn)
                    else:  # subHv
                        sum_subHv(Layer, layer, base_rdn)
            else:
                AggH[:] = deepcopy(aggH)
            sum_ext(Ext,ext)
        else:
           T[:] = deepcopy(t)


def sum_subHv(T, t, base_rdn, fneg=0):

    if t:
        if T:
            SubH,Valt,Rdnt,Dect,Ext,_ = T; subH,valt,rdnt,dect,ext,_ = t
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i]+dect[i])/2
            if SubH:
                for Layer, layer in zip_longest(SubH,subH, fillvalue=[]):
                    sum_derHv(Layer, layer, base_rdn, fneg)  # _lay[0][0] is mL
                    sum_ext(Layer[-2], layer[-2])
            else:
                SubH[:] = deepcopy(subH)
            sum_ext(Ext,ext)
        else:
            T[:] = deepcopy(t)


def sum_derHv(T,t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    if t:
        if T:
            DerH, Valt, Rdnt, Dect, Extt_,_ = T; derH, valt, rdnt, dect, extt_,_ = t
            for Extt, extt in zip(Extt_,extt_):
                sum_ext(Extt, extt)
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i] + dect[i])/2
            DerH[:] = [
                [[sum_dertuple(Dertuple,dertuple, fneg*i) for i,(Dertuple,dertuple) in enumerate(zip(Tuplet,tuplet))],
                  [V+v for V,v in zip(Valt,valt)], [R+r+base_rdn for R,r in zip(Rdnt,rdnt)], [(D+d)/2 for D,d in zip(Dect,dect)], 0
                ]
                for [Tuplet,Valt,Rdnt,Dect,_], [tuplet,valt,rdnt,dect,_]  # ptuple_tv
                in zip_longest(DerH, derH, fillvalue=[([0,0,0,0,0,0],[0,0,0,0,0,0]), (0,0),(0,0),(0,0),0])
            ]
            sum_ext(Extt_, extt_)

        else:
            T[:] = deepcopy(t)
'''

def sum_ext(Extt, extt):

    if Extt:
        if isinstance(Extt[0], list):
            for Ext,ext in zip(Extt,extt):  # ext: m,d tuple
                for i,(Par,par) in enumerate(zip(Ext,ext)):
                    Ext[i] = Par+par
        else:  # single ext
            for i in range(3): Extt[i]+=extt[i]  # sum L,S,A

    else:
        Extt[:] = deepcopy(extt)

# not reviewed:
def comp_ext_(_ext_, ext_):

    Valt, Rdnt, Dect = [0,0],[1,1],[0,0]
    dextt = []
    if isinstance(_ext_[0], list):  # der_extt: [[L,S,A], [L,S,A],...]
        for _der_ext, der_ext in zip(_ext_, ext_):
             dder_ext, valt, rdnt, dect = comp_ext(_der_ext,der_ext)  # pack them flat, an array where each element is [L,S,A], decode using ,1,2,4,8,... ?
             dextt += [dder_ext]
             for i in 0,1:
                 Valt[i] += valt[i]
                 Rdnt[i] += rdnt[i]
                 Dect[i] += dect[i]
    else:  # ext: [L,S,A]
        dextt, valt, rdnt, dect = comp_ext(_ext_,ext_)
        for i in 0,1:
            Valt[i] += valt[i]
            Rdnt[i] += rdnt[i]
            Dect[i] += dect[i]


    return dextt, Valt, Rdnt, Dect

def comp_ext(_ext, ext):  # comp ds:

    (_L,_S,_A),(L,S,A) = _ext,ext
    dL = _L-L
    Mval, Dval, Mrdn, Drdn = 0,0,1,1

    if isinstance(A,Cangle):
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
    Mval += m; Dval += d
    Mrdn += d>m; Drdn += d<=m
    _aL = abs(_L); aL = abs(L)
    _aS = abs(_S); aS = abs(S)

    # ave dec = ave (ave dec, ave L,S,A dec):
    Mdec = (mL / max(aL,_aL) if aL or _aL else 1 +
                mS / max(aS,_aS) if aS or _aS else 1 +
                mA / max_mA if max_mA else 1) /3

    Ddec = (dL / (_aL+aL) if aL+_aL else 1 +
                dS / (_aS+aS) if aS+_aS else 1 +
                dA / max_dA if max_mA else 1) /3


    return [[mL,mS,mA], [dL,dS,dA]],[Mval, Dval],[Mrdn, Drdn],[Mdec, Ddec]


def feedback(root, fd):  # called from form_graph_, append new der layers to root

    AggH, Valt, Rdnt, Dect = deepcopy(root.fback_t[fd].pop(0))
    # init with 1st tuple
    while root.fback_t[fd]:
        aggH, valt, rdnt, dect = root.fback_t[fd].pop(0)
        sum_aggH(AggH, aggH, base_rdn=0)
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