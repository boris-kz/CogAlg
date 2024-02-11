import numpy as np
from copy import deepcopy, copy
from itertools import combinations, zip_longest
from .classes import add_, CderP, Cgraph, CderG, Ct, CderH
from .filters import aves, ave_mL, ave_dangle, ave, ave_distance, G_aves, ave_Gm, ave_Gd
from .slice_edge import slice_edge, comp_angle
from .comp_slice import der_recursion, comp_derH, sum_derH, comp_ptuple, sum_dertuple, comp_dtuple, get_match

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
Clustering criterion is G M|D, summed across >ave vars if selective comp (<ave vars are not compared, so they don't add costs).
Fork selection should be per var or co-derived der layer or agg level. 
Exclusive clustering per (fork,ave), the nodes may overlap between clusters of different forks.  

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
        if edge.valt[fd] * (len(node_)-1)*(edge.rng+1) > G_aves[fd] * edge.rdnt[fd]:

            for PP in node_: PP.root = None
            # discontinuous PP cross-comp, cluster -> G_t
            agg_recursion(None, edge, node_, nrng=1, fagg=1)  # agg+, no Cgraph nodes

    return edge


def agg_recursion(rroot, root, node_, nrng=1, fagg=0):  # lenH = len(root.aggH[-1][0]), lenHH: same in agg_compress

    Et = [[0,0],[0,0],[0,0]]  # G-external vals summed from grapht link_ Valt, Rdnt, Dect(currently not used)

    # agg+ der=1 xcomp of new Gs if fagg, else sub+: der+ xcomp of old Gs:
    nrng = rng_recursion(rroot, root, combinations(node_,r=2) if fagg else root.link_, Et, nrng)  # rng+ appends rim, link.derH

    form_graph_t(root, node_, Et, nrng, fagg)  # root_fd, eval sub+, feedback per graph

    if node_ and isinstance(node_[0], list):
        for fd, G_ in enumerate(node_):
            if root.valt[fd] * (len(G_)-1)*root.rng > G_aves[fd] * root.rdnt[fd]:
                # agg+ / node_t, vs. sub+ / node_:
                agg_recursion(rroot, root, G_, nrng=1, fagg=1)
                if rroot and fd:  # der+ only
                    rroot.fback_ += [[root.aggH,root.valt,root.rdnt,root.dect]]
                    feedback(rroot)  # update root.root..


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
                    link = CderG(_G=_G, G=G, A=Ct(dy,dx), S=dist)
                    comp_G(link, et)
            else:
                _G_.add((_G, G))  # for next rng+

    if et[0][0] > ave_Gm * et[1][0]:  # rng+ eval per arg cluster because comp is bilateral, 2nd test per new pair
        for Part, part in zip(Et,et): add_(Part,part)  # Vt[i]+=v; Rt[i]+=rt[i]; Dt[i]+=d
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
                    _link_.add(CderG(G=G, _G=_G, A=Ct(dy,dx), S=dist))  # to compare in rng+


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
                        root.fback_ += [[graph.aggH, graph.valt, graph.rdnt, graph.dect]]
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

    for G in root_G_:   # init per node, last-layer Vt,Vt,Dt:
        grapht = [[G],[],G.Vt,G.Rt,G.Dt, copy(G.rimH[-1] if G.rimH and isinstance(G.rimH[0],list) else G.rimH)]  # link_ = last rim
        G.root = grapht  # for G merge
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
                    _grapht = _G.root
                    _G_,_Link_,_Vt,_Rt,_Dt,_Rim = _grapht
                    Link_[:] = list(set(Link_+_Link_)) + [link]
                    for g in _G_:
                        g.root = grapht
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
    if fd: graph.root = root
    for link in Link_:
        link.root = graph
        graph.ext[1] += link.S
        graph.ext[2] += link.A
    extH, valt,rdnt,dect, evalt,erdnt,edect = [], [0,0],[0,0],[0,0], [0,0],[0,0],[0,0]
    # grapht int = node int+ext
    for G in G_:
        graph.ext[0] += 1
        sum_last_lay(G, fd)
        graph.box.extend(G.box)
        graph.ptuple += G.ptuple
        sum_derH([graph.derH,[0,0],[1,1]], [G.derH,[0,0],[1,1]], base_rdn=1)
        sum_Hv([extH,evalt,erdnt,edect,2], [G.extH,G.evalt,G.erdnt,G.edect,2], base_rdn=G.erdnt[fd])
        sum_H(graph.aggH, G.aggH, depth=2, base_rdn=1)
        for j in 0,1:
            evalt[j] += G.evalt[j]; erdnt[j] += G.erdnt[j]; edect[j] += G.edect[j]
            valt[j] += G.valt[j]; rdnt[j] += G.rdnt[j]; dect[j] += G.dect[j]
    graph.aggH.append(extH)  # dsubH| daggH
    # graph internals = G Internals + Externals:
    valt = Ct(*valt) + evalt; graph.valt = valt
    rdnt = Ct(*rdnt) + erdnt; graph.rdnt = rdnt
    dect = Ct(*dect) + edect; graph.dect = dect
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


def comp_G(link, Et):

    _G, G = link._G, link.G
    Valt, Rdnt, Dect, Extt = Ct(0,0), Ct(1,1), Ct(0,0), []
    # keep separate P ptuple and PP derH, empty derH in single-P G, + empty aggH in single-PP G:

    # / P:
    mtuple, dtuple, Mtuple, Dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1, fagg=1)
    valt = Ct(sum(mtuple), sum(abs(d) for d in dtuple))  # mval is signed, m=-min in comp x sign
    rdnt = Ct(valt[1]>valt[0], valt[1]<=valt[0])
    dect = Ct(0,0)
    for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
        for i, (par, max, ave) in enumerate(zip(ptuple, Ptuple, aves)):
            # compute link decay coef: par/ max(self/same)
            if fd: dect[1] += abs(par)/ abs(max) if max else 1
            else:  dect[0] += (par+ave)/ (max+ave) if max else 1
    dect[0] = dect[0]/6; dect[1] = dect[1]/6  # ave of 6 params
    Valt += valt; Rdnt += rdnt; Dect += dect
    dertv = CderH(H=[mtuple,dtuple], valt=valt,rdnt=rdnt,dect=dect,depth=0)  # no ext in dertvs

    # / PP:
    extt,valt,rdnt,dect = comp_ext(_G.ext,G.ext)
    Valt += valt; Rdnt += rdnt; (Dect[0] + dect[0]) /2; Dect[1] = (Dect[1] + dect[1]) /2
    for Ext,ext in zip(Extt,extt): sum_ext(Ext,ext)

    dderH = [dertv, extt]
    if _G.derH and _G.derH:  # empty in single-P Gs?
        for _lay, lay in zip(_G.derH.H,_G.derH.H):
            mtuple,dtuple, Mtuple,Dtuple = comp_dtuple(_lay[1], lay[1], rn=1, fagg=1)
            valt = Ct(sum(mtuple),sum(abs(d) for d in dtuple))
            rdnt = Ct(valt[1] > valt[0], valt[1] <= valt[0])
            dect = Ct(0,0)
            for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
                for (par, max, ave) in zip(ptuple, Ptuple, aves):  # different ave for comp_dtuple
                    if fd: dect[1] += abs(par)/ abs(max) if max else 1
                    else:  dect[0] += (par+ave)/ (max+ave) if max else 1
            dect[0] = dect[0]/6; dect[1] = dect[1]/6  # ave of 6 params
            Valt += valt; Rdnt += rdnt; Dect[0] = (Dect[0] + dect[0]) /2; Dect[1] = (Dect[1] + dect[1]) /2
            dderH += [CderH(H=[mtuple,dtuple], valt=valt,rdnt=rdnt,dect=dect,depth=0)]  # new dderH layer
    dderH = CderH(H=dderH,valt=copy(Valt),rdnt=copy(Rdnt),dect=copy(Dect), depth=1)
    # / G:

    if _G.aggH and G.aggH:
        H = [dderH]
        dHv = comp_Hv(CderH(_G.aggH,_G.valt,_G.rdnt,_G.dect,2), CderH(G.aggH,G.valt,G.rdnt,G.dect,2), rn=1, depth=2)
        dH = dHv.H; valt=dHv.valt; rdnt = dHv.rdnt; ext = dHv.ext

        # aggH is subH before agg+
        Valt += valt
        Rdnt[0] += rdnt[0] + (valt[1]>valt[0]); Rdnt[1] += rdnt[1] + (valt[1]<=valt[0])
        Dect[0] = (Dect[0] + dect[0]) /2; Dect[1] = (Dect[1] + dect[1]) /2
        # flat, appendleft:
        daggH = CderH(H+dH+[Extt], copy(Valt),copy(Rdnt),copy(Dect),2)
    else:
        daggH = dderH
    link.daggH += [daggH]

    link.Vt,link.Rt,link.Dt = Valt,Rdnt,Dect  # reset per comp_G

    for fd, (Val,Rdn,Dec) in enumerate(zip(Valt,Rdnt,Dect)):
        if Val > G_aves[fd] * Rdn:
            Et[0][fd] += Val; Et[1][fd] += Rdn; Et[2][fd] += Dec  # to eval grapht in form_graph_t
            G.Vt[fd] += Val;  G.Rt[fd] += Rdn;  G.Dt[fd] += Dec
            if not fd:
                for G in link.G, link._G:
                    rimH = G.rimH
                    if rimH and isinstance(rimH[0],list):  # rim is converted to rimH in 1st sub+
                        if len(rimH) == len(G.RimH): rimH += [[]]  # no new rim layer yet
                        rimH[-1] += [link]  # rimH
                    else:
                        rimH += [link]  # rim


def comp_Hv(_Hv, Hv, rn,depth):  # for derH, subH, or aggH

    DHv = CderH(depth=depth)

    for _lev, lev in zip_longest(_Hv.H, Hv.H, fillvalue=0):  # compare common subHs, if lower-der match?
        if _lev and lev:
            if lev[-1] == 0:  # ptuplet
                mtuple,dtuple, Mtuple,Dtuple = comp_dtuple(_lev[0][1], lev[0][1], rn, fagg=1)
                mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
                mrdn = dval > mval; drdn = dval < mval
                dect = Ct(0,0)
                for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
                    for (par, max, ave) in zip(ptuple, Ptuple, aves):
                        if fd: dect[1] += abs(par)/abs(max) if max else 1
                        else:  dect[0] += (par+ave)/abs(max)+ave if max else 1
                dect[0]/=6; dect[1]/=6
                valt = Ct(mval, dval); rdnt = Ct(mrdn, drdn)
                dhv = CderH([mtuple,dtuple],valt,rdnt,dect,0)
                DHv.H += [dhv]
                DHv.valt += dhv.valt; DHv.rdnt += dhv.rdnt; DHv.dect += dhv.dect
            elif lev[-1] == 1:  # derHv
                dHv = comp_Hv(_lev,lev, rn, 1)
                DHv += [dHv]
                DHv.valt += dhv.valt;DHv.rdnt += dhv.rdnt;DHv.dect += dhv.dect
            elif lev[-1] == 2:  # subHv
                dHv = comp_Hv(_lev,lev, rn, 2)
                DHv.H += dHv.H  # concat
                DHv.valt += dhv.valt;DHv.rdnt += dhv.rdnt;DHv.dect += dhv.dect
            else:  # ext
                dext,valt,rdnt,dect = comp_ext(_lev[1],lev[1])  # comp dext only
                DHv.H += [dext]
                DHv.valt += valt
                DHv.rdnt += rdnt; DHv.rdnt[0] += valt[1] > valt[0]; DHv.rdnt[1] += valt[0] > valt[1]
                DHv.dect += dect
    if DHv.H:
        S = min(len(_Hv.H),len(Hv.H)); DHv.dect[0]/= S; DHv.dect[1]/= S   # normalize

    return DHv


def comp_ext(_ext, ext):  # comp ds:

    (_L,_S,_A),(L,S,A) = _ext,ext
    dL = _L-L
    Mval, Dval, Mrdn, Drdn = 0,0,1,1

    if isinstance(A,Ct):
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

    return [[mL,mS,mA], [dL,dS,dA]],Ct(Mval, Dval),Ct(Mrdn, Drdn),Ct(Mdec, Ddec)


# tentative
def sum_Hv(Hv, hv, base_rdn, fneg=0):

    if hv:
        if Hv:
            H,Valt,Rdnt,Dect,Depth = Hv; h,valt,rdnt,dect,depth = hv
            for i in 0,1:
                Valt += valt; Rdnt += rdnt; Dect[0] = (Dect[0]+dect[0])/2; Dect[1] = (Dect[1]+dect[1])/2
            sum_H(H, h, depth, base_rdn)
        else:
           Hv[:] = deepcopy(hv)


def sum_H(H, h, depth, base_rdn, fneg=0):

    if h:
        if H:
            for Layer, layer in zip_longest(H,h, fillvalue=None):
                if layer:
                    if Layer:
                        if isinstance(layer[-1], list):  # extt
                            for Ext,ext in zip (Layer, layer):
                                sum_ext(Ext,ext)
                        elif layer[-1] == 0:  # derHv
                            Tuplet,Valt,Rdnt,Dect,_ = Layer; tuplet,valt,rdnt,dect,_ = layer
                            for i in 0,1:
                                sum_dertuple(Tuplet[i],tuplet[i], fneg*i)
                            Valt += valt; Rdnt += rdnt; Dect += dect
                        else:  # subHv| aggHv
                             sum_Hv(Layer, layer, base_rdn, fneg)
                    elif Layer != None:
                        Layer[:] = deepcopy(layer)
                    else:
                        H.append(deepcopy(layer))
        else:
            H[:] = deepcopy(h)


def sum_ext(Ext, ext):  # ext: m|d L,S,A

    for i,(Par,par) in enumerate(zip(Ext,ext)):

        if isinstance(Par,list): add_(Par,par)  # sum angle
        else: Ext[i] = Par+par


def feedback(root):  # called from form_graph_, append new der layers to root

    AggH, Valt, Rdnt, Dect = deepcopy(root.fback_.pop(0))  # init
    while root.fback_:
        aggH, valt, rdnt, dect = root.fback_.pop(0)
        sum_Hv(AggH, aggH, base_rdn=0); Valt += valt; Rdnt += rdnt; Dect += dect

    if Valt[1] > G_aves[1] * Rdnt[1]:  # compress levels?
        root.aggH += AggH; root.valt += Valt; root.rdnt += Rdnt; root.dect += Dect

    if isinstance(root.root, list):  # not Edge
        rroot = root.root
        if rroot:
            fback_ = rroot.fback_  # always node_t if feedback
            if fback_ and len(fback_) == len(rroot.node_[1]):  # after all nodes sub+
                feedback(rroot)    # sum2graph adds higher aggH, feedback adds deeper aggH layers