import numpy as np
from copy import deepcopy, copy
from .classes import CQ, Cgraph
from .filters import aves, ave, ave_nsub, ave_sub, ave_agg, G_aves, med_decay, ave_distance, ave_Gm, ave_Gd
from .comp_slice import comp_angle, comp_aangle

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
This resembles neuron, which has dendritic tree as input and axonal tree as output.
But we have recursively structured param sets packed in each level of these trees, there is no such structure in neurons.
Diagram: 
https://github.com/boris-kz/CogAlg/blob/76327f74240305545ce213a6c26d30e89e226b47/frame_2D_alg/Illustrations/generic%20graph.drawio.png
-
Clustering criterion is G.M|D, summed across >ave vars if selective comp (<ave vars are not compared, so they don't add costs).
Fork selection should be per var or co-derived der layer or agg level. 
There are concepts that include same matching vars: size, density, color, stability, etc, but in different combinations.
Weak value vars are combined into higher var, so derivation fork can be selected on different levels of param composition.
'''


def agg_recursion(root):  # compositional recursion in root.PP_

    mgraph_, dgraph_ = form_graph_(root, fsub=0)  # node.H cross-comp and graph clustering, comp frng pplayers

    Rdnt = [np.sum(root.rdnT[i]) for i in [0,1]]  # Valt = [np.sum(root.valT[i]) for i in [0,1]]
    # eval graphs for sub+, ~sub_recursion_eval?:
    for fd, graph_ in enumerate([mgraph_,dgraph_]):
        val = sum([np.sum(graph.valT[fd]) for graph in graph_])
        rdn = sum([np.sum(graph.rdnT[fd]) for graph in graph_]) + Rdnt[fd]
        # intra-graph sub+:
        if val > ave_sub * rdn:  # same in blob, same base cost for both forks
            for graph in graph_: graph.rdnT[-1][fd] += 1  # estimate, assign to the weaker in feedback
            sub_recursion(graph_, fd)  # divide graph_ in der+|rng+ sub_graphs
        else:
            feedback(root)  # update root.root..H, breadth-first
    # eval graph_ for agg+: ~agg_recursion_eval?:
    Rdnt = [np.sum(root.rdnT[i]) for i in [0,1]]; Valt = [np.sum(root.valT[i]) for i in [0,1]]  # updated in sub+

    for fd, graph_ in enumerate([mgraph_, dgraph_]):  # eval graphs for sub+ and agg+:
        if Valt[fd] > G_aves[fd] * ave_agg * Rdnt[fd] and len(graph_) > ave_nsub:
            for graph in graph_: graph.rdnT[-1][fd] += 1  # estimate
            agg_recursion(root)  # replaces root.H
        else:
            feedback(root)  # update root.root..H, breadth-first


def form_graph_(root, fsub): # form derH in agg+ or sub-pplayer in sub+, G is node in GG graph

    G_ = root.node_
    comp_G_(G_, fsub=fsub)  # cross-comp all graph nodes in rng, graphs may be segs | fderGs, root G += link, link.node

    mnode_,dnode_ = [],[]  # Gs with >0 +ve fork links:
    for G in G_:
        if G.link_t[0]: mnode_ += [G]  # all nodes with +ve links, not clustered in graphs yet
        if G.link_t[1]: dnode_ += [G]
    graph_t = []
    for fd, node_ in enumerate([mnode_, dnode_]):
        graph_ = []  # init graphs by link val:
        while node_:  # all Gs not removed in add_node_layer
            G = node_.pop(); gnode_ = [G]
            val = init_graph(gnode_, node_, G, fd, val=0)  # recursive depth-first gnode_+=[_G]
            graph_+=[gnode_,val]
        # prune graphs by node val:
        regraph_ = graph_reval_(graph_, [G_aves[fd] for graph in graph_], fd)  # init reval_ to start
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum proto-graph node_ params in graph
        graph_t += [graph_]

    # add_alt_graph_(graph_t)  # overlap+contour, cluster by common lender (cis graph), combined comp?
    return graph_t

# not revised
def init_graph(gnode_, G_, G, fd, val):  # recursive depth-first gnode_+=[_G]

    for link in G.link_:
        # all positive links init graph, eval node.link_ in prune_node_layer
        _G = link.G[1] if link.G[0] is G else link.G[0]
        if _G in G_:  # _G is not removed in prior loop
            gnode_ += [_G]
            G_.remove(_G)
            val += _G.link_.valt[fd]
            val += init_graph(gnode_, G_, _G, fd, val)
    return val


def graph_reval_(graph_, reval_, fd):  # recursive eval nodes for regraph, after pruning weakly connected nodes

    regraph_, rreval_ = [],[]
    aveG = G_aves[fd]

    while graph_:
        graph,val = graph_.pop()
        reval = reval_.pop()  # each link *= other_G.aggH.valt
        if val > aveG:
            if reval < aveG:  # same graph, skip re-evaluation:
                regraph_+=[[graph,val]]; rreval_+=[0]
            else:
                regraph, reval, rval = graph_reval(graph, val, fd)  # recursive depth-first node and link revaluation
                if rval>aveG:
                    regraph_+=[[regraph,rval]]; rreval_+=[reval]
        # else remove graph
    if max([reval for reval in rreval_]) > aveG:
        regraph_ = graph_reval_(regraph_, rreval_, fd)  # graph reval while min val reduction

    return regraph_

def graph_reval(graph, Val, fd):  # exclusive graph segmentation by reval,prune nodes and links

    aveG = G_aves[fd]
    reval = 0  # reval proto-graph nodes by all positive in-graph links:

    for node,val in graph.Q:
        link_val = 0
        val = node.link_.valt[fd]
        for link in node.link_.Qd if fd else node.link_.Qm:  # Qm=Qr: rng+
            _node = link.G[1] if link.G[0] is node else link.G[0]
            link_val += val + _node.link_.valt[fd]*med_decay - val*med_decay
        reval += val - link_val  # _node.link_.valt was updated in previous round
        node.link_.valt[fd] = link_val  # update
    rreval = 0
    if reval > aveG:  # prune:
        regraph = CQ()  # reformed proto-graph
        for node in graph.Q:
            val = node.link_.valt[fd]
            if val < G_aves[fd] and node in graph.Q:  # prune revalued node and its links
                for link in node.link_.Qd if fd else node.link_.Qm:
                    _node = link.G[1] if link.G[0] is node else link.G[0]
                    _link_ = _node.link_.Qd if fd else node.link_.Qm
                    if link in _link_: _link_.remove(link)
                    rreval += link.pH.valt[fd] + val*med_decay - link.pH.valt[fd]*med_decay  # else same as rreval += link_.val
            else:
                link_ = node.link_.Qd if fd else node.link_.Qm  # prune node links only:
                remove_link_ = []
                for link in link_:
                    _node = link.G[1] if link.G[0] is node else link.G[0]  # add med_link_ val to link val:
                    link_val = link.pH.valt[fd] + _node.link_.valt[fd]*med_decay - link.pH.valt[fd]*med_decay
                    if link_val < aveG:  # prune link, else no change
                        remove_link_ += [link]
                        rreval += link_val
                while remove_link_:
                    link_.remove(remove_link_.pop(0))
                Val+=val  #?
                regraph.Q += [node]; regraph.valt[fd] += node.link_.valt[fd]
        # recursion:
        if rreval > aveG and Val > aveG:
            regraph, reval, Val = graph_reval(graph, Val, fd)  # not sure about Val
            rreval+= reval

    else: regraph = graph
    return regraph, rreval, Val
'''
In rng+, graph may be extended with out-linked nodes, merged with their graphs and re-segmented to fit fixed processors?
Clusters of different forks / param sets may overlap, else no use of redundant inclusion?
No centroid clustering, but cluster may have core subset.
'''

def comp_G_(G_, pri_G_=None, f1Q=1, fsub=0):  # cross-comp Graphs if f1Q, else G_s in comp_node_, or segs inside PP?

    if not f1Q: dpH_ = []

    for i, _iG in enumerate(G_ if f1Q else pri_G_):  # G_ is node_ of root graph
        for iG in G_[i+1:] if f1Q else G_:  # compare each G to other Gs in rng, bilateral link assign, val accum:
            # if the pair was compared in prior rng+:
            if iG in [node for link in _iG.link_.Q for node in link.node_]:  # if f1Q? add frng to skip?
                continue
            dy = _iG.box[0]-iG.box[0]; dx = _iG.box[1]-iG.box[1]  # between center x0,y0
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity, proximity = ave-distance
            if distance < ave_distance * ((sum(_iG.pH.valt) + sum(iG.pH.valt)) / (2*sum(G_aves))):
                # same for cis and alt Gs:
                for _G, G in ((_iG, iG), (_iG.alt_Graph, iG.alt_Graph)):
                    if not _G or not G:  # or G.val
                        continue
                    dpH = op_parH(_G.pH, G.pH, fcomp=1)  # comp layers while lower match?
                    dpH.ext[1] = [1,distance,[dy,dx]]  # pack in ds
                    mval, dval = dpH.valt
                    derG = Cgraph(valt=[mval,dval], G=[_G,G], pH=dpH, box=[])  # box is redundant to G
                    # add links:
                    _G.link_.Q += [derG]; G.link_.Q += [derG]  # no didx, no ext_valt accum?
                    if mval > ave_Gm:
                        _G.link_.Qm += [derG]; _G.link_.valt[0] += mval
                        G.link_.Qm += [derG]; G.link_.valt[0] += mval
                    if dval > ave_Gd:
                        _G.link_.Qd += [derG]; _G.link_.valt[1] += dval
                        G.link_.Qd += [derG]; G.link_.valt[1] += dval

                    if not f1Q: dpH_+= dpH  # comp G_s
                # implicit cis, alt pair nesting in mderH, dderH
    if not f1Q:
        return dpH_  # else no return, packed in links

    '''
    comp alts,val,rdn? cluster per var set if recurring across root: type eval if root M|D?
    '''

def op_parH(_parH, parH, fcomp, fneg=0):  # unpack aggH( subH( derH -> ptuples

    if fcomp: dparH = CQ()
    elev, _idx, d_didx, last_i, last_idx = 0,0,0,-1,-1

    for _i, _didx in enumerate(_parH.Q):  # i: index in Qd (select param set), idx: index in ptypes (full param set)
        _idx += _didx; idx = last_idx+1; _fd = _parH.fds[elev]; _val = _parH.Qd[_i].valt[_fd]
        for i, didx in enumerate(parH.Q[last_i+1:]):  # start after last matching i and idx
            idx += didx; fd = _parH.fds[elev]; val = parH.Qd[_i+i].valt[fd]
            if _idx==idx:
                if _fd==fd:
                    _sub = _parH.Qd[_i]; sub = parH.Qd[_i+i]
                    if fcomp:
                        if _val > G_aves[fd] and val > G_aves[fd]:
                            if sub.n:  # sub is ptuple
                                dsub = op_ptuple(_sub, sub, fcomp, fd, fneg)  # sub is vertuple | ptuple | ext
                            else:  # sub is pH
                                dsub = op_parH(_sub, sub, 0, fcomp)  # keep unpacking aggH | subH | derH
                                if sub.ext[1]: comp_ext(_sub.ext[1],sub.ext[1], dsub)
                            dparH.valt[0]+=dsub.valt[0]; dparH.valt[1]+=dsub.valt[1]  # add rdnt?
                            dparH.Qd += [dsub]; dparH.Q += [_didx+d_didx]
                            dparH.fds += [fd]
                    else:  # no eval: no new dparH
                        if sub.n: op_ptuple(_sub, sub, fcomp, fd, fneg)  # sub is vertuple | ptuple | ext
                        else:
                            op_parH(_sub, sub, fcomp)  # keep unpacking aggH | subH | derH
                            if sub.ext[1]: sum_ext(_sub.ext, sub.ext)
                last_i=i; last_idx=idx  # last matching i,idx
                break
            elif fcomp:
                if _idx < idx: d_didx+=didx  # += missing didx
            else:
                _parH.Q.insert[idx, didx+d_didx]
                _parH.Q[idx+1] -= didx+d_didx  # reduce next didx
                _parH.Qd.insert[idx, deepcopy(parH.Qd[idx])]
                d_didx = 0
            if _idx < idx: break  # no par search beyond current index
            # else _idx > idx: keep searching
            idx += 1  # 1 sub/loop
        _idx += 1
        if elev in (0,1) or not _i%(2**elev):  # first 2 levs are single-element, higher levs are 2**elev elements
            elev+=1  # elevation
    if fcomp:
        return dparH
    else:
        _parH.valt[0] += parH.valt[0]; _parH.valt[1] += parH.valt[1]
        _parH.rdnt[0] += parH.rdnt[0]; _parH.rdnt[1] += parH.rdnt[1]


def op_ptuple(_ptuple, ptuple, fcomp, fd=0, fneg=0):  # may be ptuple, vertuple, or ext

    aveG = G_aves[fd]
    if fcomp:
        dtuple=CQ(n=_ptuple.n)  # + ptuple.n / 2: average n?
        rn = _ptuple.n/ptuple.n  # normalize param as param*rn for n-invariant ratio: _param/ param*rn = (_param/_n)/(param/n)
    _idx, d_didx, last_i, last_idx = 0,0,-1,-1

    for _i, _didx in enumerate(_ptuple.Q):  # i: index in Qd: select param set, idx: index in full param set
        _idx += _didx; idx = last_idx+1
        for i, didx in enumerate(ptuple.Q[last_i+1:]):  # start after last matching i and idx
            idx += didx
            if _idx == idx:
                _par = _ptuple.Qd[_i]; par = ptuple.Qd[_i+i]
                if fcomp:  # comp ptuple
                    if ptuple.Qm: val =_par+par if fd else _ptuple.Qm[_i]+ptuple.Qm[_i+i]
                    else:         val = aveG+1  # default comp for 0der pars
                    if val > aveG:
                        if isinstance(par,list):
                            if len(par)==4: m,d = comp_aangle(_par,par)
                            else: m,d = comp_angle(_par,par)
                        else: m,d = comp_par(_par, par*rn, aves[idx], finv = not i and not ptuple.Qm)
                            # finv=0 if 0der I
                        dtuple.Qm+=[m]; dtuple.Qd+=[d]; dtuple.Q+=[d_didx+_didx]
                        dtuple.valt[0]+=m; dtuple.valt[1]+=d  # no rdnt, rdn = m>d or d>m?)
                else:  # sum ptuple
                    D, d = _ptuple.Qd[_i], ptuple.Qd[_i+i]
                    if isinstance(d, list):  # angle or aangle
                        for j, (P,p) in enumerate(zip(D,d)): D[j] = P-p if fneg else P+p
                    else: _ptuple.Qd[i] += -d if fneg else d
                    if _ptuple.Qm:
                        mpar = ptuple.Qm[_i+i]; _ptuple.Qm[i] += -mpar if fneg else mpar
                last_i=i; last_idx=idx  # last matching i,idx
                break
            elif fcomp:
                if _idx < idx: d_didx+=didx  # no dpar per _par
            else: # insert par regardless of _idx < idx:
                _ptuple.Q.insert[idx, didx+d_didx]
                _ptuple.Q[idx+1] -= didx+d_didx  # reduce next didx
                _ptuple.Qd.insert[idx, ptuple.Qd[idx]]
                if _ptuple.Qm: _ptuple.Qm.insert[idx, ptuple.Qm[idx]]
                d_didx = 0
            if _idx < idx: break  # no par search beyond current index
            # else _idx > idx: keep searching
            idx += 1
        _idx += 1
    if fcomp: return dtuple

def comp_ext(_ext, ext, dsub):
    # comp ds:
    (_L,_S,_A), (L,S,A) = _ext, ext
    dL=_L-L; mL=ave-abs(dL); dS=_S/_L-S/L; mS=ave-abs(dS)
    if isinstance(A,list): mA, dA = comp_angle(_A,A)
    else:
        dA=_A-A; mA=ave-abs(dA)
    dsub.ext[0][:]= mL,mS,mA; dsub.ext[1][:]= dL,dS,dA
    dsub.valt[0] += mL+mS+mA; dsub.valt[1] += dL+dS+dA

def sum_ext(_ext, ext):
    _dL,_dS,_dA = _ext[1]; dL,dS,dA = ext[1]
    if isinstance(dA,list):
        _dA[0]+=dA[0]; _dA[1]+=dA[1]
    else: _dA+=dA
    _ext[1][:] = _dL+dL,_dS+dS, _dA
    if ext[0]:
        for i, _par, par in enumerate(zip(_ext[0],ext[0])):
            _ext[i] = _par+par

''' if n roots: 
sum_derH(Graph.uH[0][fd].derH,root.derH) or sum_G(Graph.uH[0][fd],root)? init if empty
sum_H(Graph.uH[1:], root.uH)  # root of Graph, init if empty
'''
# draft:
def sum2graph_(graph_, fd):  # sum node and link params into graph, derH in agg+ or player in sub+

    Graph_ = []  # Cgraphs
    for graph in graph_:  # CQs
        if graph.valt[fd] < G_aves[fd]:  # form graph if val>min only
            continue
        Graph = Cgraph(pH=deepcopy(graph.Q[0].pH))
        Link_ = []
        for i, iG in enumerate(graph.Q):  # form G, keep iG as local subset of lower Gs
            if i: op_parH(Graph.pH, iG.pH, fcomp=0)
            # if g.uH: sum_H(G.uH, g.uH[1:])  # sum g->G
            # if g.H: sum_H(G.H[1:], g.H)  # not used yet
            for i in 0, 1:
                Graph.pH.valt[i] += iG.pH.valt[i]; Graph.pH.rdnt[i] += iG.pH.rdnt[i]
            sum_box(Graph.box, iG.box)
            link_ = [iG.link_.Qm, iG.link_.Qd][fd]  # mlink_|dlink_
            Link_[:] = list(set(Link_ + link_))
            subH = deepcopy(link_[0].pH)  # init, fd = new Cgraph.aggH fd
            G = Cgraph(pH=subH, root=Graph, node_=link_, box=copy(iG.box))
            for i, derG in enumerate(link_):
                if i: op_parH(G.pH, derG.pH, fcomp=0)
                sum_box(G.box, derG.G[0].box if derG.G[1] is iG else derG.G[1].box)
                Graph.nval += iG.nval
            Graph.node_ += [G]
        # if mult roots: sum_H(G.uH[1:], Graph.uH)
        Graph.root = iG.root  # same root, lower derivation is higher composition
        SubH = deepcopy(Link_[0].pH) # init
        for derG in Link_[1:]:  # sum unique links only
            op_parH(SubH, derG.pH, fcomp=0)
        # if Graph.uH: Graph.val += sum([lev.val for lev in Graph.uH]) / sum([lev.rdn for lev in Graph.uH])  # if val>alt_val: rdn+=len_Q?
        Graph.pH.Qd += [SubH]; Graph.pH.Q+=[0]
        Graph.pH.valt[0]+=SubH.valt[0]; Graph.pH.valt[1]+=SubH.valt[1]
        Graph.pH.rdnt[0]+=SubH.rdnt[0]; Graph.pH.rdnt[1]+=SubH.rdnt[1]
        Graph.pH.fds+=[fd]
        Graph_ += [Graph]

    return Graph_

def sum_box(Box, box):
    Y, X, Y0, Yn, X0, Xn = Box;  y, x, y0, yn, x0, xn = box
    Box[:] = [Y + y, X + x, min(X0, x0), max(Xn, xn), min(Y0, y0), max(Yn, yn)]

# draft
def sub_recursion(graph_, RVal=0, DVal=0):  # rng+: extend G_ per graph, der+: replace G_ with derG_

    for graph in graph_:
        node_ = graph.node_
        if graph.valt[0] > G_aves[0] and graph.valt[1] > G_aves[1] and len(node_) > ave_nsub:

            sub_mgraph_, sub_dgraph_ = form_graph_(graph, fsub=1)  # cross-comp and clustering
            # rng+:
            Rval = sum([sum(sub_mgraph.valt) for sub_mgraph in sub_mgraph_])
            # eval if Val>cost of call, else feedback per sub_mgraph?
            if RVal + Rval > ave_sub * graph.rdn:
                rval, dval = sub_recursion(sub_mgraph_, RVal=Rval, DVal=DVal)
                RVal += rval+dval
            # der+:
            Dval = sum([sum(sub_dgraph.valt) for sub_dgraph in sub_dgraph_])
            if DVal + Dval > ave_sub * graph.rdn:
                rval, dval = sub_recursion(sub_dgraph_, RVal=Rval, DVal=DVal)
                Dval += rval+dval
            RVal += Rval
            DVal += Dval
        else:
            feedback(graph)  # bottom-up feedback to update root.H

    return RVal, DVal  # or SVal= RVal+DVal, separate for each fork of sub+?

# old:
def feedback(root):  # bottom-up update root.H, breadth-first

    fbV = aveG+1
    while root and fbV > aveG:
        if all([[node.fterm for node in root.node_]]):  # forward was terminated in all nodes
            root.fterm = 1
            fbval, fbrdn = 0,0
            for node in root.node_:
                # node.node_ may empty when node is converted graph
                if node.node_ and not node.node_[0].box:  # link_ feedback is redundant, params are already in node.derH
                    continue
                for sub_node in node.node_:
                    fd = sub_node.fds[-1] if sub_node.fds else 0
                    if not root.H: root.H = [CQ(H=[[],[]])]  # append bottom-up
                    if not root.H[0].H[fd]: root.H[0].H[fd] = Cgraph()
                    # sum nodes in root, sub_nodes in root.H:
                    sum_parH(root.H[0].H[fd].derH, sub_node.derH)
                    sum_H(root.H[1:], sub_node.H)  # sum_G(sub_node.H forks)?
            for Lev in root.H:
                fbval += Lev.val; fbrdn += Lev.rdn
            fbV = fbval/max(1, fbrdn)
            root = root.root
        else:
            break

# old:
def add_alt_graph_(graph_t):  # mgraph_, dgraph_
    '''
    Select high abs altVal overlapping graphs, to compute value borrowed from cis graph.
    This altVal is a deviation from ave borrow, which is already included in ave
    '''
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            for node in graph.derHs.H[-1].node_:
                for derG in node.link_.Q:  # contour if link.derHs.val < aveGm: link outside the graph
                    for G in [derG.node0, derG.node1]:  # both overlap: in-graph nodes, and contour: not in-graph nodes
                        alt_graph = G.roott[1-fd]
                        if alt_graph not in graph.alt_graph_ and isinstance(alt_graph, CQ):  # not proto-graph or removed
                            graph.alt_graph_ += [alt_graph]
                            alt_graph.alt_graph_ += [graph]
                            # bilateral assign
    for fd, graph_ in enumerate(graph_t):
        for graph in graph_:
            if graph.alt_graph_:
                graph.alt_derHs = CQ()  # players if fsub? der+: derHs[-1] += player, rng+: players[-1] = player?
                for alt_graph in graph.alt_graph_:
                    sum_pH(graph.alt_derHs, alt_graph.derHs)  # accum alt_graph_ params
                    graph.alt_rdn += len(set(graph.derHs.H[-1].node_).intersection(alt_graph.derHs.H[-1].node_))  # overlap