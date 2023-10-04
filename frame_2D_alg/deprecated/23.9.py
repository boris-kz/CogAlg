def sub_recursion_eval(root, graph_): # eval per fork, same as in comp_slice, still flat aggH, add valt to return?

    Sub_tt = [[[],[]],[[],[]]]  # graph_-wide, maps to root.fback_tt

    for graph in graph_:
        node_ = copy(graph.node_tt)  # still graph.node_
        sub_tt = []  # each new fork adds fback
        frt = [0,0]
        for fder in 0,1:
            if graph.val_Ht[fder][-1] * np.sqrt((len(node_)-1)) > G_aves[fder] * graph.rdn_Ht[fder][-1]:
                graph.rdn_Ht[fder][-1] += 1  # estimate, no node.rdnt[fd]+=1?
                frt[fder] = 1
                sub_tt += [sub_recursion(root, graph, node_, fder)]  # comp_der|rng in graph -> parLayer, sub_Gs
            else:
                sub_tt += [node_]
        for fder in 0,1:
            if frt[fder]:
                for fd in 0, 1:
                    Sub_tt[fder][fd] += [sub_tt[fder][fd]]  # graph_-wide, even if empty, nest to count graphs for higher feedback
                graph.node_tt[fder] = sub_tt[fder]  # else still graph.node_
    for fder in 0,1:
        for fd in 0,1:
            if Sub_tt[fder][fd]:  # new nodes, all terminated, all send feedback
                feedback(root, fder, fd)

def sub_recursion(root, graph, node_, fder):  # rng+: extend G_ per graph, der+: replace G_ with derG_, valt=[0,0]?

    for i in 0,1: root.rdn_Ht[i][0] += 1  # estimate, no node.rdnt[fder] += 1?
    pri_root_tt_ = []
    for node in node_:
        if not fder: node.link_H += [[]]  # add link layer:
        pri_root_tt_ += [node.root_tt]  # merge node roots for new graphs in segment_node_
        node.root_tt = [[[],[]],[[],[]]]  # replace node roots
        for i in 0,1:
            node.val_Ht[i]+=[0]; node.rdn_Ht[i]+=[1]  # new val,rdn layer, accum in comp_G_

    comp_G_(node_, pri_G_=None, f1Q=1, fder=fder)  # cross-comp all nodes in rng
    sub_t = []
    for fd in 0,1:
        graph.rdn_Ht[fd][-1] += 1  # estimate
        pri_root_tt_ = []
        for node in node_:
            pri_root_tt_ += [[node.root_tt]]  # to be transferred to new graphs
            node.root_tt[fder][fd] = []  # fill with new graphs
        sub_G_ = form_graph_(node_, fder, fd, pri_root_tt_)  # cluster sub_graphs via link_H
        sub_recursion_eval(graph, sub_G_)
        for G in sub_G_:
            root.fback_tt[fder][fd] += [[G.aggH, G.val_Ht, G.rdn_Ht]]
        sub_t += [sub_G_]

    return sub_t

# ignore: lateral splicing of initial Ps is not needed: will be spliced in PPs through common forks?
def lat_comp_P(_P,P):  # to splice, no der+

    ave = P_aves[0]
    rn = len(_P.dert_)/ len(P.dert_)

    mtuple,dtuple = comp_ptuple(_P.ptuple[:-1], P.ptuple[:-1], rn)
    _L, L = _P.ptuple[-1], P.ptuple[-1]
    gap = np.hypot((_P.y - P.y), (_P.x, P.x))
    rL = _L - L
    mM = min(_L, L) - ave
    mval = sum(mtuple); dval = sum(dtuple)
    mrdn = 1+(dval>mval); drdn = 1+(1-(dval>mval))  # rdn = Dval/Mval?

def sub_recursion_eval_PP(root, PP_):  # fork PP_ in PP or blob, no derH in blob

    termt = [1,1]
    # PP_ in PP_t:
    for PP in PP_:
        sub_tt = []  # from rng+, der+
        fr = 0  # recursion in any fork
        for fder in 0,1:  # rng+ and der+:
            if len(PP.node_tt) > ave_nsubt[fder] and PP.valt[fder] > PP_aves[fder] * PP.rdnt[fder]:
                termt[fder] = 0
                if not fr:  # add link_tt and root_tt for both comp forks:
                    for P in PP.node_tt:
                        P.root_tt = [[None,None],[None,None]]
                        P.link_H += [[]]  # form root_t, link_t in sub+:
                sub_tt += [sub_recursion(PP, PP_, fder)]  # comp_der|rng in PP->parLayer
                fr = 1
            else:
                sub_tt += [PP.node_tt]
                # we have a separated fback for PPs here, not sure if this should be merged into fback_tt? Because this is derH instead of aggH here
                root.fback_ += [[PP.derH, PP.valt, PP.rdnt]]  # separate feedback per terminated comp fork
        if fr:
            PP.node_tt = sub_tt  # nested PP_ tuple from 2 comp forks, each returns sub_PP_t: 2 clustering forks, if taken

    return termt

def vectorize_root(edge, verbose=False):

    slice_edge(edge, verbose=False)
    edge = comp_slice(edge, verbose=verbose)  # scan rows top-down, compare y-adjacent, x-overlapping Ps to form derPs
    # not revised:
    for fd, PP_ in enumerate(edge.node_tt[0]):  # [rng+ PPm_,PPd_, der+ PPm_,PPd_]
        # sub+, intra-PP:
        sub_recursion_eval(edge, PP_)
        # agg+, inter-PP, 1st layer is two forks only:
        if sum([PP.valt[fd] for PP in PP_]) > ave * sum([PP.rdnt[fd] for PP in PP_]):
            node_ = [] # G_
            for PP in PP_: # CPP -> Cgraph:
                derH,valt,rdnt = PP.derH,PP.valt,PP.rdnt
                node_ += [Cgraph(ptuple=PP.ptuple, derH=[derH,valt,rdnt], val_Ht=[[valt[0]],[valt[1]]], rdn_Ht=[[rdnt[0]],[rdnt[1]]],
                                 L=PP.ptuple[-1], box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box))]  # use PP.ptuple[-1] for L because node might be updated with sub_tt
                                 # init aggH is empty
                sum_derH([edge.derH,edge.valt,edge.rdnt], [derH,valt,rdnt], 0)
            edge.node_tt[0][fd][:] = node_

            # form incrementally higher-composition graphs, graphs-of-graphs, etc., rng+ comp only because they are not linked yet:
            while sum(edge.val_Ht[0]) * np.sqrt(len(node_)-1) if node_ else 0 > G_aves[0] * sum(edge.rdn_Ht[0]):
                agg_recursion(edge, node_)  # node_[:] = new node_tt in the end, with sub+
                # no feedback of node_ in agg+ here?

def agg_recursion(root, node_):  # compositional recursion in root graph

    pri_root_tt_ = [node.root_tt for node in node_]  # merge node roots for new graphs, same for both comp forks
    node_tt = [[],[]]  # fill with comp fork if selected, else stays empty

    for fder in 0,1:
        # eval per comp fork, n comp graphs ~-> n matches, match rate decreases with distance:
        if np.sum(root.val_Ht[fder]) * np.sqrt(len(node_)-1) if node_ else 0 > G_aves[fder] * np.sum(root.rdn_Ht[fder]):
            if fder and len(node_[0].link_H) < 2:  # 1st call, no der+ yet
                continue
            root.val_Ht[fder] += [0]; root.rdn_Ht[fder] += [1]  # estimate, no node.rdnt[fder] += 1?
            for node in node_:
                node.root_tt[fder] = [[],[]]  # to replace node roots per comp fork
                node.val_Ht[fder] += [0]; node.rdn_Ht[fder] += [1]  # new layer, accum in comp_G_:
            if not fder:  # add layer of links if rng+
                for node in node_: node.link_H += [[]]

            comp_G_(node_, pri_G_=None, f1Q=1, fder=fder)  # cross-comp all Gs in (rng,der), nD array? form link_H per G
            node_tt[fder] = [[],[]]  # fill with clustering forks by default
            for fd in 0,1:
                # cluster link_H[-1] -> graph_,= node_ in node_tt, default, agg+ eval per graph:
                graph_ = form_graph_(root, node_, fder, fd, pri_root_tt_)
                node_tt[fder][fd] = graph_
    node_[:] = node_tt  # replace local element with new graph forks, possibly empty


def comp_G_(G_, pri_G_=None, f1Q=1, fder=0):  # cross-comp in G_ if f1Q, else comp between G_ and pri_G_, if comp_node_?

    for G in G_:  # node_
        if fder:  # follow prior link_ layer
            _G_ = []
            for link in G.link_H[-2]:
                if link.valt[1] > ave_Gd:
                    _G_ += [link.G1 if G is link.G0 else link.G0]
        else:  _G_ = G_ if f1Q else pri_G_  # loop all Gs in rng+
        for _G in _G_:
            if _G in G.compared_:  # was compared in prior rng
                continue
            dy = _G.box[0]-G.box[0]; dx = _G.box[1]-G.box[1]
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity
            if distance < ave_distance * ((sum(_G.val_Ht[fder]) + sum(G.val_Ht[fder])) / (2*sum(G_aves))):
                G.compared_ += [_G]; _G.compared_ += [G]
                # same comp for cis and alt components:
                for _cG, cG in ((_G, G), (_G.alt_Graph, G.alt_Graph)):
                    if _cG and cG:  # alt Gs maybe empty
                        # form new layer of links:
                        comp_G(_cG, cG, distance, [dy,dx])
    '''
    combine cis,alt in aggH: alt represents node isolation?
    comp alts,val,rdn? cluster per var set if recurring across root: type eval if root M|D?
    '''

def agg_recursion(rroot, root, G_, fini):  # compositional recursion in root graph, rng+ only if fini

    pri_root_tt_ = []
    for G in deepcopy(G_):  # copy for reversibility
        pri_root_tt_ += [G.root_tt]  # merge node roots for new graphs, same for both comp forks
        G.root_tt = (([],[]),([],[]))  # replace with new graphs
    fr = 0  # comp of either fork

    for fder in 0,1:  # der+| rng+ comp forks
        if fder and fini:  # 1st call, no der+ yet, or test len(G_[0].link_H) < 2?
            break
        # eval last layer, per comp fork only, n comp graphs ~-> n matches, match rate decreases with distance:
        if root.val_Ht[fder][-1] * np.sqrt(len(G_)-1) if G_ else 0 > G_aves[fder] * root.rdn_Ht[fder][-1]:
            fr = 1
            comp_G_(G_, pri_G_=None, f1Q=1, fder=fder)  # cross-comp all Gs in (rng,der), nD array? form link_H per G
            root.val_Ht[fder] += [0]; root.rdn_Ht[fder] += [1]  # estimate, no node.rdnt[fder] += 1?
            # or root.rdn_t[fder][-1] += (root.val_Ht[fder][-1] - G_aves[fder] * root.rdn_Ht[fder][-1] > root.val_Ht[1-fder][-1] - G_aves[1-fder] * root.rdn_Ht[1-fder][-1])
            for fd in 0,1:
                # cluster link_H[-1] -> graph_,= node_ in node_tt, default, agg+ eval per graph:
                form_graph_(root, G_, fder, fd, pri_root_tt_)
                if rroot:
                    rroot.fback_tt[fder][fd] += [[root.aggH, root.val_Ht, root.rdn_Ht]]  # merge across forks
        if not fr:
            root.node_tt = G_  # revert if empty node_tt


def form_PP_t(root, P_, base_rdn):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = []
    for fd in 0,1:
        qPP_ = []  # initial pre_PPs are in list format
        for P in P_:
            if P.root_t[fd]:  continue  # skip if already packed in some qPP
            qPP = [[P]]  # append with _Ps, then Val in the end
            P.root_t[fd] = qPP
            val = 0  # sum of in-graph link vals, added to qPP in the end
            uplink_ = P.link_H[-1] # 1st layer of uplinks
            uuplink_ = []  # next layer of uplinks, or uplink_ = deque(P.link_H[-1]) # queue in breadth first search

            while uplink_:  # test for next-line uuplink_, set at loop's end
                for derP in uplink_:
                    if derP.valt[fd] <= P_aves[fd]*derP.rdnt[fd]: continue  # link _P should not be in qPP
                    # else add link val, always unique
                    val += derP.valt[fd]
                    _P = derP._P
                    _qPP = _P.root_t[fd]
                    if _qPP :  # _P was clustered in different qPP in prior loops
                        if _qPP is qPP: continue
                        for __P in _qPP[0]:  # merge _qPP in qPP
                            qPP[0] += [__P]; __P.root_t[fd] = qPP
                        val += _qPP[1]  # _qPP Val
                        qPP_.remove(_qPP)
                    else:  # add _P
                        qPP[0] += [_P]; _P.root_t[fd] = qPP
                    # pack bottom up
                    uuplink_ += derP._P.link_H[-1]
                uplink_ = uuplink_
                uuplink_ = []
            qPP += [val, ave+1]  # ini reval=ave+1, keep qPP same object for ref in P.
            qPP_ += [qPP]

        rePP_ = reval_PP_(qPP_, fd)  # prune qPPs by mediated links vals, PP = [qPP,valt,reval]
        PP_t += [[sum2PP(root, qPP, base_rdn, fd) for qPP in rePP_]]

    for fd in 0,1:  # after form_PP_t, to fill root_t per sub+ layer
        sub_recursion(root, PP_t[fd], fd)  # eval P_ rng+ per PPm or der+ per PP
        if root.fback_t and root.fback_t[fd]:
            feedback(root, fd)  # feedback after sub+ is terminated in all root fork nodes, not individual through multiple layers

    root.node_t = PP_t  # PPs maybe nested in sub+, add_alt_PPs_?

def reval_PP(P_, fd):   # recursive eval / prune Ps for rePP

    val = sum((derP.valt[fd] for P in P_ for derP in P.link_H[-1]
               if derP.valt[fd] > P_aves[fd] * derP.rdnt[fd]))
    # add rdn: Ave = ave * 1+(valt[fd] < valt[1-fd])  # * PP.rdn for more selective clustering?

    if val <= ave: return   # check val first, return None if val doesn't pass ave
    reval = ave+1  # to start reval

    while reval > ave:
        P_, val, reval = reval_P_(P_, fd)  # recursive node and link re-evaluation by med val
        if val <= ave: return  # always check val after reval

    rePP = [P_, val, reval]     # pack rePP
    for P in P_:
        P.root_t = rePP         # assign root_t

    return [P_, val, reval]

def reval_PP_(PP_, fd):  # recursive eval / prune Ps for rePP

    rePP_ = []
    while PP_:  # init P__
        P_, val, reval = PP_.pop(0)
        # Ave = ave * 1+(valt[fd] < valt[1-fd])  # * PP.rdn for more selective clustering?
        if val > ave:
            if reval < ave:  # same graph, skip re-evaluation:
                rePP_ += [[P_,val,0]]  # reval=0
            else:
                rePP = reval_P_(P_,fd)  # recursive node and link re-evaluation by med val
                if val > ave:  # min adjusted val
                    rePP_ += [rePP]
                else:
                    for P in rePP: P.root_t[fd] = []
        else:  # low-val qPPs not added to rePP_
            for P in P_: P.root_t[fd] = []

    if rePP_ and max([rePP[2] for rePP in rePP_]) > ave:  # recursion if any min reval:
        rePP_ = reval_PP_(rePP_, fd)

    return rePP_
'''
            while _Val > Ave *_Rdn:
                Val = reval_cP_(cP_, ave, fd)  # recursive link revaluation by mediated link vals
                if abs(_Val-Val) < Ave/2:  # lower filter for reval vs. eval
                    PP_t[fd] += sum2PP(root, cP_, base_rdn, fd)
                    break
'''
def reval_cP_(cP_, ave, fd):  # reval cP_ by link_val + mediated link_vals
    Val = 0  # pre PP val

    for P in cP_:
        P_val = 0
        for link in P.link_H[-1]:
            _val,_rdn = 0,0
            for _link in link._P.link_H[-1]:
                _val += _link.valt[fd]; _rdn += _link.rdnt[fd]
            # link val += med links val, single mediation layer in comp_slice:
            link_val = (link.valt[fd]+ _val* med_decay) - ave * (link.rdnt[fd]+ _rdn* med_decay)
            if link_val > ave/2:  # lower filter for links vs Ps
                P_val += link_val  # consider positive med-adjusted links only

        if P_val > ave:
            Val += P_val

    return Val  # same cP_: pruning Ps would split it into multiple sub-cP_ s?

def sum_ptuple(Ptuple, ptuple, fneg=0):

    for i, (Par, par) in enumerate(zip_longest(Ptuple, ptuple, fillvalue=None)):
        if par != None:
            if Par != None:
                if isinstance(Par, list) or isinstance(Par, tuple):  # angle or aangle
                    for i,(P,p) in enumerate(zip(Par,par)):
                        Par[i] = P-p if fneg else P+p
                else:
                    Ptuple[i] += (-par if fneg else par)  # now includes n in ptuple[-1]?
            elif not fneg:
                Ptuple += [copy(par)]

    nodet_ = [[node, (node.val_Ht[fd][-1] - ave * node.rdn_Ht[fd][-1])] for node in node_]

    while dVal > ave:  # iterative adjust Val by surround propagation, no direct increment mediation rng?

        dVal = 0
        for i, [node,Val] in enumerate(nodet_):
            if Val>0:  # potential graph init
                for link in node.link_H[-1]:
                    _node = link.G1 if link.G0 is node else link.G0   # val + sum([_val * link_rel_val: max m|d decay]):
                    dval = _Val_[node_.index(_node)] * (link.valt[fd] / link.valt[2]) - ave * sum(_node.rdn_Ht[fd])

def sum_aggH(T, t, base_rdn):

    AggH, Val_Ht, Rdn_Ht = T; aggH, val_Ht, rdn_Ht = t

    for i in 0,1:
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
