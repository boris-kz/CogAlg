class Cgraph(CPP):  # graph or generic PP of any composition

    link_ = lambda: Clink_()  # evaluated external links (graph=node), replace alt_node if open, direct only
    node_ = lambda: CQ()  # input | fd Gs from sub+ or secondary agg+, redundant to layers,levels
    # agg,sub forks: [[],0] if called
    mlevels = list  # for comp len and caching
    dlevels = list
    rlayers = list  # | mlayers: init from node_ for subdivision, micro relative to levels
    dlayers = list  # | alayers; val->sub_recursion, init = plevels.val?
    # summed params in levels ( layers:
    plevels = lambda: CpH()  # summed node_ plevels ( pplayers ( players ( ptuples: primary sub+, secondary agg+?
    alt_plevels = list  # optional summed contour params, also compared by comp_pH
    alt_graph_ = list  # contour + overlapping contrast graphs
    alt_rdn = int  # node_, alt_node_s overlap
    x0 = float
    y0 = float  # center: box x0|y0 + L/2
    xn = float  # max distance from center
    yn = float
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs; no alt_rdn: valt representation in alt_PP_ valts?
    rng = lambda: 1  # not for alt_graphs
    roott = lambda: [None, None]  # higher-order segG or graphs of two forks

# compare depth of nesting:
Mlen, Dlen = 0, 0
for _Hm, _Hd, Hm, Hd in zip([_G.mlevels, _G.dlevels, G.mlevels, G.dlevels], [_G.rlayers[0], _G.dlayers[0], G.rlayers[0], G.dlayers[0]]):
    mmlen, dmlen, mdlen, ddlen = 0, 0, 0, 0
    for _H, H, mlen, dlen in zip([_Hm, _Hd], [Hm, Hd], [mmlen, dmlen], [mdlen, ddlen]):  # H: hierarchy
        for _level, level in zip(_H, H):
            _L, L = len(_level), len(level)
            dlen += _L - L; mlen += ave-dlen
        Mlen += mlen
        Dlen += dlen
mplevel.val += Mlen; dplevel.val += Dlen  # combined cis match and diff


def agg_recursion(root, fseg):  # compositional recursion in root.PP_, pretty sure we still need fseg, process should be different

    for fork, pplayers in enumerate(root.wH[0]):  # root graph 1st plevel forks: mpplayers, dpplayers, alt_mpplayers, alt_dpplayers
        # each is still an individual cLev here, no feedback yet?
        if pplayers: # H: plevels ( forks ( pplayers ( players ( ptuples

            for G in pplayers.node_:  # init forks, rng+ H[1][fork][-1] is immutable, comp frng pplayers
                G.wH += [[[],[],[],[]]]
            mgraph_, dgraph_ = form_graph_(root, fork)  # cross-comp and graph clustering

            for fd, graph_ in enumerate([mgraph_,dgraph_]):  # eval graphs for sub+ and agg+:
                val = sum([graph.val for graph in graph_])
                # intra-graph sub+ comp node:
                if val > ave_sub * (root.rdn):  # same in blob, same base cost for both forks
                    pplayers.rdn+=1  # estimate
                    sub_recursion_g(graph_, val, fseg, fd)  # subdivide graph_ by der+|rng+
                # cross-graph agg+ comp graph:
                if val > G_aves[fd] * ave_agg * (root.rdn) and len(graph_) > ave_nsub:
                    pplayers.rdn += 1  # estimate
                    agg_recursion(root, fseg=fseg)

def form_graph_(root, G_, fd): # form plevel in agg+ or player in sub+, G is node in GG graph; der+: comp_link if fderG, from sub+

    comp_G_(G_, fd)  # cross-comp all graphs within rng, graphs may be segs | fderGs, G.roott += link, link.node
    mnode_, dnode_ = [], []  # Gs with >0 +ve fork links:
    for G in G_:
        if G.link_.Qm: mnode_ += [G]  # all nodes with +ve links, not clustered in graphs yet
        if G.link_.Qd: dnode_ += [G]
    graph_t = []
    for fd, node_ in enumerate([mnode_, dnode_]):
        graph_ = []  # form graphs by link val:
        while node_:  # all Gs not removed in add_node_layer
            G = node_.pop()
            gnode_ = [G]
            val = add_node_layer(gnode_, node_, G, fd, val=0)  # recursive depth-first gnode_+=[_G]
            graph_ += [CQ(Q=gnode_, val=val)]
        # reform graphs by node val:
        regraph_ = graph_reval(graph_, [ave_G for graph in graph_], fd)  # init reval_ to start
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum proto-graph node_ params in graph
        graph_t += [graph_]
        [root.mlevels,root.dlevels][fd][0].insert(0, graph_)  # add top agg fork level
        [root.mlevels,root.dlevels][fd][1].insert(0, CpH())
        for graph in graph_:
            sum_pH([root.mlevels, root.dlevels][fd][1][-1], graph.plevels)

    add_alt_graph_(graph_t)  # overlap+contour, cluster by common lender (cis graph), combined comp?
    return graph_t

def graph_reval(graph_, reval_, fd):  # recursive eval nodes for regraph, increasingly selective with reduced node.link_.val

    regraph_, rreval_ = [],[]
    Reval = 0
    while graph_:
        graph = graph_.pop()
        reval = reval_.pop()
        if reval < ave_G:  # same graph, skip re-evaluation:
            regraph_ += [graph]; rreval_ += [0]
            continue
        while graph.Q:  # some links will be removed, graph may split into multiple regraphs, init each with graph.Q node:
            regraph = CQ()
            node = graph.Q.pop()  # node_, not removed below
            val = [node.link_.mval, node.link_.dval][fd]  # in-graph links only
            if val > G_aves[fd]:  # else skip
                regraph.Q = [node]; regraph.val = val  # init for each node, then add _nodes
                readd_node_layer(regraph, graph.Q, node, fd)  # recursive depth-first regraph.Q+=[_node]
            reval = graph.val - regraph.val
            if regraph.val > ave_G:
                regraph_ += [regraph]; rreval_ += [reval]; Reval += reval
    if Reval > ave_G:
        regraph_ = graph_reval(regraph_, rreval_, fd)  # graph reval while min val reduction

    return regraph_

def readd_node_layer(regraph, graph_Q, node, fd):  # recursive depth-first regraph.Q+=[_node]

    for link in [node.link_.Qm, node.link_.Qd][fd]:  # all positive
        _node = link.node_.Q[1] if link.node_.Q[0] is node else link.node_.Q[0]
        _val = [_node.link_.mval, _node.link_.dval][fd]
        if _val > G_aves[fd] and _node in graph_Q:
            regraph.Q += [_node]
            graph_Q.remove(_node)
            regraph.val += _val; _node.roott[fd] = regraph
            readd_node_layer(regraph, graph_Q, _node, fd)

def add_node_layer(gnode_, G_, G, fd, val):  # recursive depth-first gnode_+=[_G]

    for link in G.link_.Q:  # all positive
        _G = link.node_.Q[1] if link.node_.Q[0] is G else link.node_.Q[0]
        if _G in G_:  # _G is not removed in prior loop
            gnode_ += [_G]
            G_.remove(_G)
            val += [_G.link_.mval,_G.link_.dval][fd]
            val += add_node_layer(gnode_, G_, _G, fd, val)

    return val

def comp_G_(G_, fd):  # cross-comp Gs (patterns of patterns): Gs, derGs, or segs inside PP, same process, no fderG?

    for i, _G in enumerate(G_):
        for G in G_[i+1:]:
            # compare each G to other Gs in rng, bilateral link assign, val accum:
            if G in [node for link in _G.link_.Q for node in link.node_.Q]:  # G,_G was compared in prior rng+, add frng to skip?
                continue
            dx = _G.x0 - G.x0; dy = _G.y0 - G.y0  # center x0,y0
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity
            # proximity = ave-distance
            if distance < ave_distance * ((_G.plevels.val+_G.alt_plevels.val + G.plevels.val+G.alt_plevels.val) / (2*sum(G_aves))):  # comb G val
                if not fd:  # rng+
                    _G.plevels.val-=_G.plevels.H[-1].val; _G.plevels.H.pop(); _G.plevels.fds.pop()
                    G.plevels.val-=G.plevels.H[-1].val; G.plevels.H.pop(); G.plevels.fds.pop()
                    # replace last plevel:
                mplevel, dplevel = comp_pH(_G.plevels, G.plevels)  # optional comp_links, in|between nodes?
                mplevel.L, dplevel.L = 1,1; mplevel.S, dplevel.S = distance,distance; mplevel.A, dplevel.A = [dy,dx],[dy,dx]
                # comp contour+overlap:
                if _G.alt_plevels and G.alt_plevels:  # or if comb plevels.val > ave * alt_rdn
                    alt_mplevel, alt_dplevel = comp_pH(_G.alt_plevels, G.alt_plevels)
                    alt_plevels = [alt_mplevel, alt_dplevel]  # no separate extuple?
                else: alt_plevels = []
                # new derG:
                y0 = (G.y0+_G.y0)/2; x0 = (G.x0+_G.x0)/2  # new center coords
                derG = Cgraph( node_=CQ(Q=[_G,G]), plevels=[mplevel,dplevel], alt_plevels=alt_plevels, y0=y0, x0=x0, # max distance from center:
                               xn=max((_G.x0+_G.xn)/2 -x0, (G.x0+G.xn)/2 -x0), yn=max((_G.y0+_G.yn)/2 -y0, (G.y0+G.yn)/2 -y0))
                mval, dval = mplevel.val, dplevel.val
                tval = mval + dval
                _G.link_.Q += [derG]; _G.link_.val += tval  # val of combined-fork' +- links?
                G.link_.Q += [derG]; G.link_.val += tval
                if mval > ave_Gm:
                    _G.link_.Qm += [derG]; _G.link_.mval += mval  # no dval for Qm
                    G.link_.Qm += [derG]; G.link_.mval += mval
                if dval > ave_Gd:
                    _G.link_.Qd += [derG]; _G.link_.dval += dval  # no mval for Qd
                    G.link_.Qd += [derG]; G.link_.dval += dval

def comp_pH(_pH, pH):  # recursive unpack plevels ( pplayer ( players ( ptuples -> ptuple:

    mpH, dpH = CpH(), CpH()
    pri_fd = 0
    for i, (_spH, spH) in enumerate(zip(_pH.H, pH.H)):

        fd = pH.fds[i] if len(pH.fds) else 0  # in plevels or players
        _fd = _pH.fds[i] if len(_pH.fds) else 0
        if _fd == fd:
            if fd: pri_fd = 1  # all scalars
            if isinstance(spH, Cptuple):
                mtuple, dtuple = comp_ptuple(_spH, spH, pri_fd)
                mpH.H += [mtuple]; mpH.val += mtuple.val
                dpH.H += [dtuple]; dpH.val += dtuple.val
            else:
                if spH.L:  # extuple is valid, in pplayer only
                    comp_ext(_spH, spH, mpH, dpH)
                sub_mpH, sub_dpH = comp_pH(_spH, spH)
                if isinstance(spH.H[0], CpH):
                    mpH.H += [sub_mpH]; dpH.H += [sub_dpH]
                else:
                    # spH.H is ptuples, combine der_ptuples into single der_layer:
                    mpH.H += sub_mpH.H; dpH.H += sub_dpH.H
                mpH.val += sub_mpH.val; dpH.val += sub_dpH.val
        else:
            break
    return mpH, dpH

def comp_ext(_spH, spH, mpH, dpH):
    L, S, A = spH.L, spH.S, spH.A
    _L, _S, _A = _spH.L, _spH.S, _spH.A

    _sparsity = _S /(_L-1); sparsity = S /(L-1)  # average distance between connected nodes, single distance if derG
    dpH.S = _sparsity - sparsity; dpH.val += dpH.S
    mpH.S = min(_sparsity, sparsity); mpH.val += mpH.S
    dpH.L = _L - L; dpH.val += dpH.L
    mpH.L = min(_L, L); mpH.val += mpH.L

    if _A and A:  # axis: dy,dx only for derG or high-aspect Gs, both val *= aspect?
        if isinstance(_A, list):
            m, d = comp_angle(None, _A, A)
            mpH.A = m; dpH.A = d
        else:  # scalar mA or dA
            dpH.A = _A - A; mpH.A = min(_A, A)
    else:
        mpH.A = 1; dpH.A = 0  # no difference, matching low-aspect, only if both?
    mpH.val += mpH.A; dpH.val += dpH.A


def sub_recursion_g(graph_, Sval, fseg, fd):  # rng+: extend G_ per graph, der+: replace G_ with derG_

    comb_layers_t = [[],[]]
    for graph in graph_:
        node_ = graph.node_
        if graph.plevels.val > G_aves[fd] and len(node_) > ave_nsub:

            sub_mgraph_, sub_dgraph_ = form_graph_(graph, node_, fd)  # cross-comp and clustering cycle
            # rng+:
            Rval = sum([sub_mgraph.plevels.val for sub_mgraph in sub_mgraph_])
            if Rval > ave_sub * graph.rdn:  # >cost of call:
                sub_rlayers, rval = sub_recursion_g(sub_mgraph_, Rval, fseg=fseg, fd=0)
                Rval+=rval; graph.rlayers = [[sub_mgraph_]+[sub_rlayers], Rval]
            else:
                graph.rlayers = [[]]  # this can be skipped if we init layers as [[]]
            # der+:
            Dval = sum([sub_dgraph.plevels.val for sub_dgraph in sub_dgraph_])
            if Dval > ave_sub * graph.rdn:
                sub_dlayers, dval = sub_recursion_g(sub_dgraph_, Dval, fseg=fseg, fd=1)
                Dval+=dval; graph.dlayers = [[sub_dgraph_]+[sub_dlayers], Dval]
            else:
                graph.dlayers = [[]]

            for comb_layers, graph_layers in zip(comb_layers_t, [graph.rlayers[0], graph.dlayers[0]]):
                for i, (comb_layer, graph_layer) in enumerate(zip_longest(comb_layers, graph_layers, fillvalue=[])):
                    if graph_layer:
                        if i > len(comb_layers) - 1:
                            comb_layers += [graph_layer]  # add new r|d layer
                        else:
                            comb_layers[i] += graph_layer  # splice r|d PP layer into existing layer
            Sval += Rval + Dval

    return comb_layers_t, Sval


def sum2graph_(graph_, fd): # sum node and link params into graph, plevel in agg+ or player in sub+

    Graph_ = []  # Cgraphs
    for graph in graph_:
        if graph.val < ave_G:  # form graph if val>min only
            continue
        Glink_= []; X0,Y0 = 0,0
        for node in graph.Q:
            Glink_ = list(set(Glink_ + node.link_))  # unique links in graph
            X0 += node.x0; Y0 += node.y0
        L = len(graph.Q); X0/=L; Y0/=L; Xn,Yn = 0,0  # first pass defines center and Glink_
        new_Plevel = CpH
        new_Plevel = [sum_pH(new_Plevel, [link.mplevel_t,link.dplevel_t][fd][fd]) for link in Glink_]  # no alts for now
        Graph = Cgraph(plevels_t=[[],[],[],[]])
        Graph.plevels_t[fd] = CpH; Plevels = Graph.plevels_t[fd]

        for node in graph.Q:  # CQ(Q=gnode_, val=val)]
            # define max distance,A, sum plevels:
            Xn = max(Xn,(node.x0+node.xn)-X0)  # box xn = x0+xn
            Yn = max(Yn,(node.y0+node.yn)-Y0)
            plevels = node.plevels_t[fd]
            if fd: plevels.H += [CpH(mpH=CpH(),dpH=CpH())]; plevels.fds += [fd]  # append new plevel
            else:  plevels.H[-1] = CpH() # replace last plevel
            # form quasi-gradient per node from variable-length links:
            for derG in node.link_.Q:
                der_plevel = [derG.mplevel, derG.dplevel][fd][fd]  # or ifd?
                sum_pH(plevels.H[-1], der_plevel)  # new plevel: node_+=[G|_G], S=link.L, preA: [Xn,Yn], sum *= dangle?
                plevels.val += der_plevel.val
            for Plevel, plevel in zip(Plevels, plevels[:-1]):  # skip last plevel, redundant between nodes
                sum_pH(Plevel, plevel); Plevels.val += plevel.val
            node.roott[fd] = Graph

        Graph.x0=X0; Graph.xn=Xn; Graph.y0=Y0; Graph.yn=Yn
        new_Plevel.A = [Xn * 2, Yn * 2]
        Plevels.H += [new_Plevel]  # new_plevel is summed from unique links, not nodes
        Plevels.val += new_Plevel.val
        Plevels.fds = copy(plevels.fds) + [fd]
        Graph_ += [Graph]  # Cgraph

    return Graph_


def add_alt_graph_(graph_t):  # mgraph_, dgraph_
    '''
    Select high abs altVal overlapping graphs, to compute value borrowed from cis graph.
    This altVal is a deviation from ave borrow, which is already included in ave
    '''
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            for node in graph.node_:
                for derG in node.link_.Q:  # contour if link.plevels.val < ave_Gm: link outside the graph
                    for G in derG.node_.Q:  # both overlap: in-graph nodes, and contour: not in-graph nodes
                        alt_graph = G.roott[1-fd]
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, Cgraph):  # not proto-graph or removed
                            graph.alt_graph_ += [alt_graph]
                            alt_graph.alt_graph_ += [graph]
                            # bilateral assign
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            if graph.alt_graph_:
                graph.alt_plevels = CpH()  # players if fsub? der+: plevels[-1] += player, rng+: players[-1] = player?
                for alt_graph in graph.alt_graph_:
                    sum_pH(graph.alt_plevels, alt_graph.plevels)  # accum alt_graph_ params
                    graph.alt_rdn += len(set(graph.node_).intersection(alt_graph.node_))  # overlap

def copy_G(G):
    link_ = copy(G.link_)
    node_ = copy(G.node_)
    rlayers = copy(G.rlayers)
    dlayers = copy(G.rlayers)
    mlevels = copy(G.mlevels)
    dlevels = copy(G.dlevels)

    G_copy = Cgraph(plevels = deepcopy(G.plevels), alt_plevels = deepcopy(G.alt_plevels),
                    link_=link_, node_=node_, alt_graph_=G.alt_graph_,
                    mlevels=mlevels, dlevels=dlevels, rlayers=rlayers, dlayers=dlayers
                    )
    return G_copy

def sum_pH(PH, pH, fneg=0):  # recursive unpack plevels ( pplayers ( players ( ptuples, no accum across fd: matched in comp_pH

    if PH.L:  # valid extuple
        PH.L += pH.L; PH.S += pH.S
        if PH.A:
            if isinstance(PH.A, list):
                PH.A[0] += pH.A[0]; PH.A[1] += pH.A[1]
            else:
                PH.A += pH.A
        else:
            PH.A = copy(pH.A)
#   elif node not in PH.node_: PH.node_ += [node]  # node that has the pH

    for SpH, spH in zip_longest(PH.H, pH.H, fillvalue=None):  # assume same fds
        if spH:
            if SpH:
                if isinstance(SpH, Cptuple):
                    sum_ptuple(SpH, spH, fneg=fneg)
                else:
                    sum_pH(SpH, spH, fneg=0)  # unpack sub-hierarchy, recursively
            elif not fneg:
                PH.H.append(deepcopy(spH))  # new Sub_pH
    PH.val += pH.val

def form_graph_(root, ifd): # form plevel in agg+ or player in sub+, G is node in GG graph; der+: comp_link if fderG, from sub+

    G_ = root.plevels_t[ifd].H[0].node_  # root node_ for both forks is in fd higher plevel
    comp_G_(G_, ifd)  # cross-comp all graphs within rng, graphs may be segs | fderGs, G.roott += link, link.node

    mnode_, dnode_ = [], []  # Gs with >0 +ve fork links:
    for G in G_:
        if G.link_.Qm: mnode_ += [G]  # all nodes with +ve links, not clustered in graphs yet
        if G.link_.Qd: dnode_ += [G]
    graph_t = []
    if mnode_: root.plevels_t[ifd].mpH = CpH()  # if not empty?
    if dnode_: root.plevels_t[ifd].dpH = CpH()
    for fd, node_ in enumerate([mnode_, dnode_]):
        graph_ = []  # form graphs by link val:
        while node_:  # all Gs not removed in add_node_layer
            G = node_.pop()
            gnode_ = [G]
            val = add_node_layer(gnode_, node_, G, fd, val=0)  # recursive depth-first gnode_+=[_G]
            graph_ += [CQ(Q=gnode_, val=val)]
        # reform graphs by node val:
        regraph_ = graph_reval(graph_, [ave_G for graph in graph_], fd)  # init reval_ to start
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd, ifd)  # sum proto-graph node_ params in graph
        graph_t += [graph_]
        # draft:
        Plevels = [root.plevels_t[ifd].mpH, root.plevels_t[ifd].dpH][fd]
        for graph in graph_:
            for plevels in graph.plevels_t:
                sum_pH(Plevels, plevels)

    add_alt_graph_(graph_t)  # overlap+contour, cluster by common lender (cis graph), combined comp?
    return graph_t

# draft:
def sum2graph_(graph_, fd, ifd):  # sum node and link params into graph, plevel in agg+ or player in sub+

    Graph_ = []  # Cgraphs
    for graph in graph_:
        if graph.val < ave_G:  # form graph if val>min only
            continue
        Glink_= []; X0,Y0 = 0,0
        for node in graph.Q:  # first pass defines center Y,X and Glink_:
            Glink_ = list(set(Glink_ + node.link_.Q))  # or node.link_.Qd if fd else node.link_.Qm? unique links across graph nodes
            X0 += node.x0; Y0 += node.y0
        L = len(graph.Q); X0/=L; Y0/=L; Xn,Yn = 0,0
        Graph = Cgraph()
        Plevels_4 = [CpH(),CpH(),CpH(),CpH()]; Val = 0

        for [node, plevels_4] in graph.Q:  # CQ(Q=gnode_, val=val)], define max distance,A, sum plevels:
            Xn = max(Xn,(node.x0+node.xn)-X0)  # box xn = x0+xn
            Yn = max(Yn,(node.y0+node.yn)-Y0)
            new_plevel_4 = [CpH(),CpH(),CpH(),CpH()]; new_val = 0  # may be in links, same val for alts?
            # form quasi-gradient per node from variable-length links:
            for derG in node.link_.Q:
                der_plevel_t = [derG.mplevel_t, derG.dplevel_t][fd]
                for new_plevel, der_plevel in zip(new_plevel_4, der_plevel_t):  # plevel is 4 pplayers
                    sum_pH(new_plevel, der_plevel)
                    new_val += der_plevel.val
            # draft:
            if fd: # append new plevel
                plevels_4[ifd].val+=new_val; plevels_4[ifd].fds+=[fd]; plevels_4[ifd].plevels += [new_plevel_4]
            else:  # replace last plevel
                plevels_4[ifd].val+=new_val-plevels_4[ifd].plevels.val; plevels_4[ifd].fds[-1]=fd; plevels_4[ifd].plevels = new_plevel_4

            Graph.node_ += [[node, [new_plevel_4]]]  # separate, to not modify and copy nodes

            sum_pHt(Plevels_4, plevels_4, fRng=0, frng=1)  # skip last plevel, redundant between nodes
            Plevels.val += plevels.val
            plevels.root = Graph[1][fd]  # add root per plevels?

        Graph.x0=X0; Graph.xn=Xn; Graph.y0=Y0; Graph.yn=Yn
        new_Plevel_4 = [CpH(),CpH(),CpH(),CpH()]; new_Val = 0  # may be in links, same val for alts?
        for link in Glink_:
            for new_Plevel, der_plevel in zip(new_Plevel_4, [link.mplevel_t, link.dplevel_4][fd]):
                sum_pH(new_Plevel, der_plevel)
                new_Val += der_plevel.val
        new_Plevel_4[fd].A = [Xn*2,Yn*2]  # not sure
        Plevels.val += new_Val
        Plevels.fds = copy(plevels.fds) + [fd]
        # draft:
        for plevels in graph[1]:  # plevels_4
            for Plevel, plevel in zip(graph.root[1], plevels):
                sum_pH(Plevel, plevel)
            graph.root.val += plevels.val
        Graph_ += [[Graph, [Plevels_4]]]  # Cgraph, reduction: root fork += all link forks?

    return Graph_
''' 
    nested new_node = [old_node, lev_n], where old_node = [oold_node, lev_n-1], oold_node = [ooold_node, lev_n-2]...

for Lev, lev in zip_longest(Graph, node_pplayers__, fillvalue=[]):  # lev: 4 fork pplayers
    for Pplayers, pplayers in zip_longest(Lev, lev, fillvalue=CpH()):
        Graph += [sum_pH(Pplayers, pplayers)]  # sum old plevels
        
 H = list  # plevels, higher H[:c+1]=fork CpH_, lower H[c:]=forks_: 4^i fork tree, from [m,d,am,ad], for feedback accum
'''
