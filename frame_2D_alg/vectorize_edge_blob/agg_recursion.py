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
        node_ = edge.node_
        if edge.valt[fd] * (len(node_)-1)*(edge.rng+1) <= G_aves[fd] * edge.rdnt[fd]:   continue
        G_= []
        for PP in node_:  # always PP_t, convert CPPs to Cgraphs:
            derH,valt,rdnt = PP.derH,PP.valt,PP.rdnt  # init aggH is empty:
            # convert derH to ptuple_tv_: [[ptuplet, valt, maxt, rdnt]]:
            derH[:] = [
                [[mtuple,dtuple], maxtuplet, [sum(mtuple),sum(dtuple)],
                 [sum([0 if m>d else 1 for m,d in zip(mtuple,dtuple)]), sum([0 if d > m else 1 for m, d in zip(mtuple,dtuple)])]
                ] for (mtuple,dtuple), maxtuplet in zip(derH, reform_maxtuplet_([PP.node_]))
            ]
            G_ += [Cgraph( ptuple=PP.ptuple, derH=[derH,valt,rdnt,[0,0]], valHt=[[valt[0]],[valt[1]]], rdnHt=[[rdnt[0]],[rdnt[1]]],
                           L=PP.ptuple[-1], box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box))]
        for PP in node_:
            for link in PP.link_:  # convert link_ to derGs:
                G = G_[node_.index(link.roott[fd])]  # some link'P's root not in node_, is it due to sub+ where
                _G = G_[node_.index(link.roott[fd])]
                derG = CderG( G=G, _G=_G, S=link.S, A=link.A)
                edge.link_ += [derG]
        node_ = G_
        edge.valHt[0][0] = edge.valt[0]; edge.rdnHt[0][0] = edge.rdnt[0]  # copy
        agg_recursion(None, edge, node_, fd=0)  # edge.node_ = graph_t, micro and macro recursive

# draft
def reform_maxtuplet_(node_t_):   # form maxt per derLayer in PP.derH, from PP.node_
    maxtuplet_ = []
    '''
    compute relative m|d per param in zip (dertuple,maxtuple): m/maxm or m/maxd,
    sum them in Decay across derH) aggH, then decay = Decay / (len(dertuple)*len(derH)*len(aggH))
    '''
    while True:
        # nesting is not revised:
        sub_node_t_ = []  # deeper layer of node_
        maxtuplet = [[0,0,0,0,0,0],[0,0,0,0,0,0]]  # current layer, 0 per param type
        for node_t in node_t_:
            # check for CP
            if node_t[0] and isinstance(node_t[0],list) and (isinstance(node_t[0][0],CPP) or (node_t[1] and isinstance(node_t[1][0],CPP))):
                # PP.node_ is [subPPm_,subPPd_], unpack each
                for fd, PP_ in enumerate(node_t):  # [sub_PPm_,sub_PPd_]
                    for PP in PP_:
                        for link in PP.link_:  # get lower-der comp pairs and find their max
                            _P, P = link._P, link.P
                            rn = len(_P.dert_) / len(P.dert_)
                            for i, (_par, par) in enumerate(zip(_P.ptuple, P.ptuple)):
                                if hasattr(par, '__len__'): maxm = maxd = 2  # angle
                                else:
                                    maxm = max(abs(_par),abs(par*rn)); maxd = abs(_par)+abs(par*rn)
                                maxtuplet[0][i] += maxm
                                maxtuplet[1][i] += maxd
                                if PP.node_[0] and isinstance(PP.node_[0],list) and (isinstance(PP.node_[0][0],CPP)
                                    or (PP.node_[1] and isinstance(PP.node_[1][0],CPP))):
                                    sub_node_t_ += [PP.node_]
        maxtuplet_ += [maxtuplet]  # pack current layer maxt

        if sub_node_t_: node_t_ = sub_node_t_  # for next layer checking
        else:           break
    return maxtuplet_

# tentative
def agg_recursion(rroot, root, G_, fd):  # compositional agg+|sub+ recursion in root graph, clustering G_

    Mval,Dval, Mrdn,Drdn = 0,0,0,0
    link_ = defaultdict(list)
    if fd:
        for link in root.link_:
            if link.valt[1] < G_aves[1]*link.rdnt[1]: continue  # maybe weak after rdn incr?
            mval,dval, mrdn,drdn = comp_G(link_,link._G,link.G, fd)
            Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn
    else:
        for i, _node in enumerate(root.node_):  # original node_ for rng+
            for node in root.node_[i+1:]:
                dy = _node.box[0]-node.box[0]; dx = _node.box[1]-node.box[1]
                distance = np.hypot(dy, dx)  # Euclidean distance between centers of Gs
                if distance < root.rng:  # close enough to compare
                    # * ((sum(_G.valHt[fd]) + sum(G.valHt[fd])) / (2*sum(G_aves)))):  # comp rng *= rel G value?
                    mval,dval, mrdn,drdn = comp_G(link_,_node,node, fd)
                    Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn

    root.valHt[fd] += [0]; root.rdnHt[fd] += [1]
    # sum in feedback:
    GG_t = form_graph_t(root, [Mval,Dval],[Mrdn,Drdn], G_, link_)  # eval sub+ and feedback per graph
    # agg+ xcomp-> form_graph_t loop sub)agg+, vs. comp_slice sub+ loop-> eval-> xcomp
    for GG_ in GG_t:  # comp_G_ eval: ave_m * len*rng - fixed cost, root update in form_t:
        if root.valHt[0][-1] * (len(GG_)-1)*root.rng > G_aves[fd] * root.rdnHt[0][-1]:
            agg_recursion(rroot, root, GG_, fd=0)  # 1st xcomp in GG_
    G_[:] = GG_t

def form_graph_t(root, Valt,Rdnt, G_, link_):  # form mgraphs and dgraphs of same-root nodes

    graph_t = []
    for G in G_: G.root = [None,None]  # replace with mcG_|dcG_ in segment_node_, replace with Cgraphs in sum2graph
    for fd in 0,1:
        if Valt[fd] > ave * Rdnt[fd]:  # else no clustering
            Gt_ = node_connect(G_, link_, fd)
            graph_t += [segment_node_(root, Gt_, fd)]  # add alt_graphs?
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

def node_connect(iG_, link_, fd):  # sum surround values to define node connectivity, over incr.mediated links

    ave = G_aves[fd]
    Gt_ = []
    for G in iG_:
        ival,irdn, rim = sum(G.valHt[fd]), sum(G.rdnHt[fd]), link_[G]   # all links that contain G, copy(link_[G])?
        Gt_ += [G, 0,0, ival,irdn, rim]  # perimeter: negative or new links, separate termination per node?
    _tVal, _tRdn = 0,0

    while True:  # eval same Gs,links with cumulative mediated values
        tVal, tRdn = 0,0  # loop totals
        for G,val,rdn, ival,irdn, rim in Gt_:  # also update links?
            rim_val, rim_rdn = 0,0
            for link in rim:
                if link.valt[fd] < ave * link.rdnt[fd]: continue  # skip negative links
                _G = link.G if link._G is G else link._G
                if _G not in iG_: continue
                _Gt = Gt_[G.i]
                _G,_val,_rdn,_ival,_irdn,_rim = _Gt
                try: decay = link.valt[fd]/link.maxt[fd]  # link decay coef: m|d / max, base self/same
                except ZeroDivisionError: decay = 1
                # node vals * relative link val:
                linkV = (val+_val) * decay; val+=linkV; _val+=linkV; rim_val += linkV  # bilateral accum
                linkR = (rdn+_rdn) * decay; rdn+=linkR; _rdn+=linkR; rim_rdn += linkR
                link.Vt[fd] = linkV  # for segment_node_
            tVal += rim_val
            tRdn += rim_rdn
        if tVal-_tVal <= ave * (tRdn-_tRdn):
            break
        _tVal,_tRdn = tVal,tRdn

    return Gt_

def segment_node_(root, Gt_, fd):  # eval rim links with summed surround vals

    graph_=[]; ave = G_aves[fd]

    for Gt in Gt_:
        G, val,rdn,ival,irdn,rim = Gt  # ival,irdn are currently not used
        grapht = [[Gt],val,rdn, ival,irdn, copy(rim)]  # nodet_,Val,Rdn, iVal,iRdn, Rim: negative or new links
        G.root[fd] = grapht; graph_ += [grapht]
    _tVal, _tRdn = 0,0

    while True:  # eval node rim links with surround vals for graph inclusion and merge:
        tVal, tRdn = 0,0  # iG_ loop totals
        # update graph surround:
        for grapht in graph_:
            nodet_, Val,Rdn, iVal,iRdn, Rim = grapht
            periVal,periRdn, inVal,inRdn = 0,0,0,0  # in: new in-graph values, net-positive
            new_Rim = []
            for link in Rim:
                if link.Vt[fd] > 0:  # combined net val from node_connect, merge linked graphs:
                    if link.G in grapht[0]: _Gt = Gt_[link._G.i]
                    else:                   _Gt = Gt_[link.G.i]
                    if _Gt in nodet_: continue
                    _nodet_,_Val,_Rdn,_iVal,_iRdn,_Rim = _Gt[0].root[fd]
                    Val+=_Val; Rdn+=_Rdn; iVal+=_iVal; iRdn+=_iRdn
                    nodet_ += [__Gt for __Gt in _Gt[0].root[fd][0] if __Gt not in nodet_]
                    Rim = list(set(Rim + _Rim))
                    new_Rim = list(set(new_Rim + _Rim))
                else:  # negative link, for reevaluation:
                    new_Rim = list(set(new_Rim + [link]))
            # graph extension:
            grapht[1] += inVal; tVal += inVal  # if eval per graph: DVal += inVal -_inVal?
            grapht[2] += inRdn; tRdn += inRdn  # DRdn += inRdn -_inRdn, signed

        if (tVal-_tVal) <= ave * (tRdn-_tRdn):  # even low-Val extension may be valuable if Rdn decreases?
            break
        _tVal,_tRdn = tVal,_tRdn

    return [sum2graph(root, graph, fd) for graph in graph_ if graph[1] > ave * graph[2]]  # Val > ave * Rdn

# not updated
def sum2graph(root, cG_, fd):  # sum node and link params into graph, aggH in agg+ or player in sub+

    graph = Cgraph(root=root, fd=fd, L=len(cG_))  # n nodes, transplant both node roots
    SubH = [[],[0,0],[1,1],[0,0]]; maxM,maxD, Mval,Dval, Mrdn,Drdn = 0,0,0,0,0,0
    Link_= []
    for i, (G,link_) in enumerate(cG_[3]):
        # sum nodes in graph:
        sum_box(graph.box, G.box)
        sum_ptuple(graph.ptuple, G.ptuple)
        sum_derHv(graph.derH, G.derH, base_rdn=1)
        sum_aggHv(graph.aggH, G.aggH, base_rdn=1)
        sum_Hts(graph.valHt, graph.rdnHt, graph.maxHt, G.valHt, G.rdnHt, G.maxHt)
        # sum external links:
        subH=[[],[0,0],[1,1],[0,0]]; mval,dval, mrdn,drdn, maxm,maxd = 0,0, 0,0, 0,0
        for derG in link_:
            if derG.valt[fd] > G_aves[fd] * derG.rdnt[fd]:  # sum positive links only:
                _subH = derG.subH
                (_mval,_dval),(_mrdn,_drdn),(_maxm,_maxd) = valt,rdnt,maxt = derG.valt, derG.rdnt, derG.maxt
                if derG not in Link_:
                    sum_subHv(SubH, [_subH,valt,rdnt,maxt] , base_rdn=1)  # new aggLev, not from nodes: links overlap
                    Mval+=_mval; Dval+=_dval; Mrdn+=_mrdn; Drdn+=_drdn; maxM+=_maxm; maxD+=_maxd
                    graph.A[0] += derG.A[0]; graph.A[1] += derG.A[1]; graph.S += derG.S
                    Link_ += [derG]
                mval+=_mval; dval+=_dval; mrdn+=_mrdn; drdn+=_drdn; maxm+=_maxm; maxd+=_maxd
                sum_subHv(subH, [_subH,valt,rdnt,maxt], base_rdn=1, fneg = G is derG.G)  # fneg: reverse link sign
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
    Y,X,Y0,X0,Yn,Xn = graph.box
    graph.box[:2] = [(Y0+Yn)/2, (X0+Xn)/2]

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
# old
def comp_G_(root_link_, fd=0):  # cross-comp in G_, comp between G_ and other_G_ for comp_node_ is not implemented

    Mval,Dval, Mrdn,Drdn = 0,0,0,0
    link_ = defaultdict(list)

    for link in root_link_:  # always form new link, maybe empty?
        if fd and link.valt[1] < G_aves[1]*link.rdnt[1]:
            continue  # maybe weak after rdn incr?

        mval,dval, mrdn,drdn = comp_G(link_,link, fd)
        Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn
        '''
        same comp for cis and alt components?
        for _cG, cG in ((_G, G), (_G.alt_Graph, G.alt_Graph)):
            if _cG and cG:  # alt Gs maybe empty
                comp_G(_cG, cG, fd)  # form new layer of links:
        combine cis,alt in aggH: alt represents node isolation?
        comp alts,val,rdn? cluster per var set if recurring across root: type eval if root M|D? '''

    return link_, [Mval,Dval], [Mrdn,Drdn]


def comp_G(link_, _G, G, fd):

    Mval,Dval, maxM,maxD, Mrdn,Drdn = 0,0, 0,0, 1,1
    link = CderG( _G=_G, G=G)
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
        subH, valt, rdnt, maxt = comp_aggHv(_G.aggH, G.aggH, rn=1)
        maxM += maxt[0]; maxD += maxt[0]
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0]+dval>mval; Drdn += rdnt[1]+dval<=mval
        link.subH = SubH+subH  # append higher subLayers: list of der_ext | derH s
        link.valt = [Mval,Dval]; link.rdnt = [Mrdn,Drdn]; link.maxt = [maxM,maxD]  # complete proto-link
        link_[G] += [link]
        link_[_G]+= [link]
    elif Mval > ave_Gm or Dval > ave_Gd:  # or sum?
        link.subH = SubH
        link.valt = [Mval,Dval]; link.rdnt = [Mrdn,Drdn]; link.maxt = [maxM,maxD] # complete proto-link
        link_[G] += [link]
        link_[_G]+= [link]
        # dict: key = G, values = derGs
    return Mval,Dval, Mrdn,Drdn

# draft:
def comp_aggHv(_aggH, aggH, rn):  # no separate ext
    SubH = []
    maxM,maxD, Mval,Dval, Mrdn,Drdn = 0,0,0,0,1,1

    for _lev, lev in zip_longest(_aggH, aggH, fillvalue=[]):  # compare common lower layer|sublayer derHs
        if _lev and lev:  # also if lower-layers match: Mval > ave * Mrdn?
            # compare dsubH only:
            dsubH, valt,rdnt,maxt = comp_subHv(_lev, lev, rn)  # no more valt and rdnt in subH now
            SubH += dsubH  # flatten to keep subH
            maxM += maxt[0]; maxD += maxt[1]
            mval,dval = valt; Mval += mval; Dval += dval
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + mval <= dval

    return SubH, [Mval,Dval],[Mrdn,Drdn],[maxM,maxD]

def comp_subHv(_subH, subH, rn):
    DerH = []
    maxM,maxD, Mval,Dval, Mrdn,Drdn = 0,0,0,0,1,1

    for _lay, lay in zip(_subH, subH):  # compare common lower layer|sublayer derHs
        # if lower-layers match: Mval > ave * Mrdn?
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

    for _lay, lay in zip(_derH, derH):  # compare common lower der layers | sublayers in derHs, if lower-layers match?
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

def sum_subHv(SubH, subH, base_rdn, fneg=0):

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
    (_,_,Y0,X0,Yn,Xn), (_,_,y0,x0,yn,xn) = Box, box  # unpack
    Box[2:] = [min(X0,x0), min(Y0,y0), max(Xn,xn), max(Yn,yn)]


def feedback(root, fd):  # called from form_graph_, append new der layers to root

    AggH, ValHt, RdnHt, MaxHt = deepcopy(root.fback_t[fd].pop(0))  # init with 1st tuple
    while root.fback_t[fd]:
        aggH, valHt, rdnHt, maxHt = root.fback_t[fd].pop(0)
        sum_aggHv(AggH, aggH, base_rdn=0)
        sum_Hts(ValHt,RdnHt,MaxHt, valHt,rdnHt,maxHt)
    sum_aggHv(root.aggH,AggH, base_rdn=0)
    sum_Hts(root.valHt,root.rdnHt,root.maxHt, ValHt,RdnHt,MaxHt)  # both forks sum in same root

    if isinstance(root, Cgraph) and root.root:  # root is not CEdge, which has no roots
        rroot = root.root
        fd = root.fd  # current node_ fd
        fback_ = rroot.fback_t[fd]
        fback_ += [[AggH, ValHt, RdnHt, MaxHt]]
        if fback_ and (len(fback_) == len(rroot.node_t)):  # flat, all rroot nodes terminated and fed back
            # getting cyclic rroot here not sure why it can happen, need to check further
            feedback(rroot, fd)  # sum2graph adds aggH per rng, feedback adds deeper sub+ layers