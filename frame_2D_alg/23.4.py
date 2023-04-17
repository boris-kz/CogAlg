def comp_pH(_pH, pH):  # recursive unpack derHs ( pplayer ( players ( ptuples -> ptuple:

    mpH, dpH = CQ(), CQ()  # new players in same top derH?

    for i, (_spH, spH) in enumerate(zip(_pH.H, pH.H)):  # s = sub
        fd = pH.fds[i] if pH.fds else 0  # in derHs or players
        _fd = _pH.fds[i] if _pH.fds else 0
        if _fd == fd:
            if isinstance(_spH, Cptuple):
                mtuple, dtuple = comp_ptuple(_spH, spH, fd)
                # not sure here, one of the val is always 0?
                mpH.H += [mtuple]; mpH.valt[0] += mtuple.val; mpH.fds += [0]  # mpH.rdn += mtuple.rdn?
                dpH.H += [dtuple]; dpH.valt[1] += dtuple.val; dpH.fds += [1]  # dpH.rdn += dtuple.rdn

            elif isinstance(_spH, CQ):
                smpH, sdpH = comp_pH(_spH, spH)
                mpH.H +=[smpH]; mpH.valt[0]+=smpH.valt[0]; mpH.valt[1]+=smpH.valt[1]; mpH.rdn+=smpH.rdn; mpH.fds +=[smpH.fds]  # or 0 | fd?
                dpH.H +=[sdpH]; dpH.valt[0]+=sdpH.valt[0]; dpH.valt[1]+=sdpH.valt[1]; dpH.rdn+=sdpH.rdn; dpH.fds +=[sdpH.fds]

    return mpH, dpH

def sum_pH_(PH_, pH_, fneg=0):
    for PH, pH in zip_longest(PH_, pH_, fillvalue=[]):  # each is CQ
        if pH:
            if PH:
                for Fork, fork in zip_longest(PH.H, pH.H, fillvalue=[]):
                    if fork:
                        if Fork:
                            if fork.derH:
                                for (Pplayers, Expplayers),(pplayers, expplayers) in zip(Fork.derH, fork.derH):
                                    if Pplayers:   sum_pH(Pplayers, pplayers, fneg)
                                    else:          Fork.derH += [[deepcopy(pplayers),[]]]
                                    if Expplayers: sum_pH(Expplayers, expplayers, fneg)
                                    else:          Fork.derH[-1][1] = deepcopy(expplayers)
                        else: PH.H += [deepcopy(fork)]
            else:
                PH_ += [deepcopy(pH)]  # CQ

def sum_pH(PH, pH, fneg=0):  # recursive unpack derHs ( pplayers ( players ( ptuples, no accum across fd: matched in comp_pH

    for SpH, spH, Fd, fd in zip_longest(PH.H, pH.H, PH.fds, pH.fds, fillvalue=None):  # assume same forks
        if spH:
            if SpH:
                if isinstance(spH, Cptuple):  # PH is ptuples, SpH_ is ptuple
                    sum_ptuple(SpH, spH, fneg=fneg)
                else:  # PH is players, H is ptuples
                    sum_pH(SpH, spH, fneg=fneg)
            else:
                PH.fds += [fd]
                PH.H += [deepcopy(spH)]
    PH.valt[0] += pH.valt[0]; PH.valt[1] += pH.valt[1]
    PH.rdn += pH.rdn
    if not PH.L: PH.L = pH.L  # PH.L is empty list by default
    else:        PH.L += pH.L
    PH.S += pH.S
    if isinstance(pH.A, list):
        if pH.A:
            if PH.A:
                PH.A[0] += pH.A[0]; PH.A[1] += pH.A[1]
            else: PH.A = copy(pH.A)
    else: PH.A += pH.A

    return PH

def op_derH(_derH, derH, op, Mval,Dval, Mrdn,Drdn, idx_=[]):  # idx_: derH indices, op: comp|sum, lenlev: 1, 1, 2, 4, 8...

    op(_derH[0], derH[0], idx_+[0])  # single-element 1st lev
    if len(_derH)>1 and len(derH)>1:
        op(_derH[1], derH[1], idx_+[1])  # single-element 2nd lev
        i,idx = 2,2; last=4  # multi-element 2nd+ levs, init incr elevation = i

        while last<len(derH) and last<len(derH):
            op_derH(_derH[i:last], derH[i:last], op, Mval,Dval, Mrdn,Drdn, idx_+[idx])  # _lev, lev: incrementally nested
            i=last; last+=i  # last=i*2
            idx+=1  # elevation in derH

    elif _derH or derH:
        pass  # fill into DerH if sum or dderH if comp?

def comp_derH(_derH, derH, Mval,Dval, Mrdn,Drdn):

    dderH = []
    for _Lev, Lev in zip_longest(derH, derH, fillvalue=[]):  # each Lev or subLev is [CpQ|list, ext, valt, rdnt]:
        if _Lev and Lev:
            if isinstance(Lev,CpQ):  # players, same for both or test? incr nesting in dplayers?
                # use extended comp_ptuple instead?
                dplayers = comp_pH(_Lev, Lev)
                Mval += dplayers.valt[0]; Mrdn += dplayers.rdnt[0]  # add rdn in form_?
                Dval += dplayers.valt[1]; Drdn += dplayers.rdnt[1]
                dderH += [dplayers]
            else:  # [sub derH, ext, valt, rdnt]
                dder, Mval, Dval, Mrdn, Drdn = comp_derH(_Lev[0],Lev[0], Mval,Dval, Mrdn,Drdn)  # all not-empty
                mext, dext = comp_ext(_Lev[1],Lev[1])
                Mval+=sum(mext); Dval+=sum(dext)
                dderH += [[dder, [mext,dext]], [Mval,Dval], [Mrdn,Drdn]]
            if (Mval+Dval) / (Mrdn+Drdn) < ave_G:
                break
        else:
            dderH += [_Lev if _Lev else -Lev]  # difference from null comparand, not sure

    return dderH, Mval,Dval, Mrdn,Drdn
'''
    generic for pTree, including each element of pPP (ptuple extended to 2D)?
    Lev1: lays: CpQ players, 1st ext is added per G.G, 2nd ext per Graph, add subLev nesting per Lev:
    Lev2: [dlays, ext]: 1 subLev
    Lev3: [[dlays, [ddlays,dextp]], ext]: 2 sLevs, 1 ssLev
    Lev4: [[dlays, [ddlays,dextp], [[[dddlays,ddextp]],dextp]], ext]: 3 sLevs, 2 ssLevs, 1 sssLev
'''

def comp_derH(_derH, derH, Mval, Dval, Mrdn, Drdn, _fds, fds):  # idx_: derH indices, op: comp|sum, lenlev: 1, 1, 2, 4, 8...

    dderH = []
    if _fds[0]==fds[0]:  # else higher fds won't match either
        dderH += [comp_ptuple(_derH[0], derH[0])]  # single-element 1st lev
        if (len(_derH)>1 and len(derH)>1) and _fds[1]==fds[1]:
            dderH += [comp_ptuple(_derH[1], derH[1])]  # single-element 2nd lev
            i,idx = 2,2; last=4  # multi-element 2nd+ levs, init incr elevation = i
            # append Mval, Dval, Mrdn, Drdn?
            while last < len(derH) and last < len(derH):  # loop _lev, lev, may be nested
                dderH += comp_derH(_derH[i:last], derH[i:last], comp_ptuple, Mval, Dval, Mrdn, Drdn, idx_ + [idx])
                i=last; last+=i  # last=i*2
                idx+=1  # elevation in derH

def comp_derH(_derH, derH):  # no need to check fds in comp_slice

    dderH = []; valt = [0,0]; rdnt = [1,1]
    for i, (_ptuple,ptuple) in enumerate(zip(_derH, derH)):

        dtuple = comp_vertuple(_ptuple,ptuple) if isinstance(_ptuple, CQ) else comp_ptuple(_ptuple,ptuple)
        # dtuple = comp_vertuple(_ptuple,ptuple) if i else comp_ptuple(_ptuple,ptuple)
        dderH += [dtuple]
        for j in 0,1:
            valt[j] += dtuple.valt[j]; rdnt[j] += dtuple.rdnt[j]

    return dderH, valt, rdnt

def prune_node_layer(regraph, graph_H, node, fd):  # recursive depth-first regraph.Q+=[_node]
    relink_=[]
    for link in node.link_.Qd if fd else node.link_.Qm:  # all positive in-graph links, Qm is actually Qr: rng+
        _node = link.G[1] if link.G[0] is node else link.G[0]
        _val = [_node.link_.mval, _node.link_.dval][fd]
        # ave / link val + linked node val:
        if _val > G_aves[fd] and _node in graph_H:
            regraph.Q += [_node]
            graph_H.remove(_node)
            regraph.valt[fd] += _val
            prune_node_layer(regraph, graph_H, _node, fd)
            # adjust link val by _node.val, adjust node.val in next round:
            link.valt[fd] += (_val / len(_node.link_) * med_decay) - link.valt[fd]
            relink_+=[link]
    [node.link_.Qd,node.link_.Qm][fd][:] = relink_  # contains links to graph nodes only

def comp_derH(_derH, derH, j,k):

    dderH = CQ()
    # we need the same nested looping and if _idx==idx as in comp_vertuple, test if Cptuple for comp_ptuple?
    # old:
    dtuple = comp_ptuple(_derH.Q[0], derH.Q[0])  # all compared pars are in Qd, including 0der
    add_dtuple(dderH, dtuple)
    elev = 0
    for i, (_ptuple,ptuple) in enumerate(zip(_derH.Q[1:], derH.Q[1:])):

        if _derH.fds[elev]!=derH.fds[elev]:  # fds start from 2nd lay
            break
        if not i%(2**elev):  # first 2 levs are single-element, higher levs are 2**elev elements
            elev += 1
        if j: dtuple = comp_vertuple(_ptuple, ptuple)  # local comps pack results in dderH
        else: dtuple = comp_ext(_ptuple, ptuple, k)  # if 1st derH in subH, comp_angle if 1st subH in aggH?

        dderH.fds += _derH.fds[elev]
        add_dtuple(dderH, dtuple)

    return dderH

def comp_vertuple(_vertuple, vertuple):

    dtuple=CQ(n=_vertuple.n)
    rn = _vertuple.n/vertuple.n  # normalize param as param*rn for n-invariant ratio: _param/ param*rn = (_param/_n)/(param/n)
    _idx, idx, d_didx = 0,0,0

    for _i, _didx in enumerate(_vertuple.Q):  # i: index in Qd (select param set), idx: index in pnames (full param set)
        for i, didx in enumerate(vertuple.Q[_i:]): # idx at i<_i won't match _idx
            if _idx==idx:
                m,d = comp_par(_vertuple.Qd[_i], vertuple.Qd[i+_i]*rn, aves[idx])
                dtuple.Qm += [m]; dtuple.Qd += [d]
                dtuple.Q += [d_didx + _didx]
                break
            elif _idx < idx:  # no dpar per _par
                d_didx += _didx
                break  # no par search beyond current index
            # else _idx > idx: continue search
            idx += didx
        _idx +=_didx

    return dtuple

# replaced by comp_ptuple
def comp_ext(_ext, ext, k):  # comp ds only, add Qn?
    _L,_S,_A = _ext; L,S,A = ext

    dS = _S - S; mS = min(_S, S)  # average distance between connected nodes, single distance if derG
    dL = _L - L; mL = min(_L, L)
    if _A and A:
        # axis: dy,dx only for derG or high-aspect Gs, both val *= aspect?
        if k: dA = _A[1] - A[1]; mA = min(_A[1], A[1])  # scalar mA,dA
        else: mA, dA = comp_angle(_A, A)
    else:
        mA,dA = 0,0

    return CQ(Qm=[mL,mS,mA],Qd=[mL,mS,mA], valt=[mL+mS+mA,dL+dS+dA])

def graph_reval(graph_, reval_, fd):  # recursive eval nodes for regraph, after pruning weakly connected nodes
    '''
    extend with comp_centroid to adjust all links, so centrally similar nodes are less pruned?
    or centroid match is case-specific, else scale links by combined valt of connected nodes' links, in their aggQ
    '''
    regraph_, rreval_ = [],[]
    Reval = 0
    while graph_:
        graph = graph_.pop()
        reval = reval_.pop()  # each link *= other_G.aggQ.valt
        if reval < aveG:  # same graph, skip re-evaluation:
            regraph_ += [graph]; rreval_ += [0]
            continue
        while graph.Q:  # links may be revalued and removed, splitting graph to regraphs, init each with graph.Q node:
            regraph = CQ()
            node = graph.Q.pop()  # node_, not removed below
            val = [node.link_.mval, node.link_.dval][fd]  # in-graph links only
            if val > G_aves[fd]:  # else skip
                regraph.Q = [node]; regraph.valt[fd] = val  # init for each node, then add _nodes
                prune_node_layer(regraph, graph.Q, node, fd)  # recursive depth-first regraph.Q+=[_node]
            reval = graph.valt[fd] - regraph.valt[fd]
            if regraph.valt[fd] > aveG:
                regraph_ += [regraph]; rreval_ += [reval]; Reval += reval
    if Reval > aveG:
        regraph_ = graph_reval(regraph_, rreval_, fd)  # graph reval while min val reduction

    return regraph_

def prune_node_layer(graph_H, node, fd):  # recursive depth-first regraph.Q+=[_node]

    link_ = node.link_.Qd if fd else node.link_.Qm
    for link in link_:
        # all positive in-graph links, Qm is actually Qr: rng+
        _node = link.G[1] if link.G[0] is node else link.G[0]
        _val = [_node.link_.mval, _node.link_.dval][fd]
        link.valt[fd] += (_val / len([_node.link_.Qm, _node.link_.Qd][fd]) * med_decay) - link.valt[fd]
        # link val += norm _node.val, adjust node.val in next round
    regraph = CQ()  # init for each node, then add _nodes
    relink_=[]
    for link in link_:
        # prune revalued nodes and links:
        _node = link.G[1] if link.G[0] is node else link.G[0]
        _val = [_node.link_.mval, _node.link_.dval][fd]
        # ave / link val + linked node val:
        if _val > G_aves[fd] and _node in graph_H:
            regraph.Q += [_node]
            graph_H.remove(_node)
            regraph.valt[fd] += _val
            relink_ += [link]  # remove link?
    # recursion:
    if regraph.valt[fd] > G_aves[fd]:
        prune_node_layer(regraph, graph_H, _node, fd)  # not sure about _node

    [node.link_.Qd,node.link_.Qm][fd][:] = relink_  # links to in-graph nodes only

def sum_ext(Ext, ext):
    for i, param in enumerate(ext):
        if i<2:  # L,S
            Ext[i] += param
        else:  # angle
            if isinstance(Ext[i], list):
                Ext[i][0] += param[0]; Ext[i][1] += param[1]
            else:
                Ext[i] += param

def sum_H(H, h):  # add g.H to G.H, no eval but possible remove if weak?

    for i, (Lev, lev) in enumerate(zip_longest(H, h, fillvalue=[])):  # root.ex.H maps to node.ex.H[1:]
        if lev:
            if not Lev:  # init:
                Lev = CQ(H=[[] for fork in range(2**(i+1))])
            for j, (Fork, fork) in enumerate(zip(Lev.H, lev.H)):
                if fork:
                    if not Fork: Lev.H[j] = Fork = Cgraph()
                    sum_G(Fork, fork)

def comp_parH(_parH, parH):  # unpack aggH( subH( derH -> ptuples

    dparH = CQ(); elev, _idx, d_didx, last_i, last_idx = 0,0,0,-1,-1

    for _i, _didx in enumerate(_parH.Q):  # i: index in Qd (select param set), idx: index in ptypes (full param set)
        _idx += _didx; idx = last_idx+1
        for i, didx in enumerate(parH.Q[last_i+1:]):  # start after last matching i and idx
            idx += didx
            if _idx==idx:
                _fd = _parH.fds[elev]; fd = parH.fds[elev]  # fd per lev, not sub
                if _fd==fd and _parH.Qd[_i].valt[fd] + parH.Qd[_i+i].valt[fd] > aveG:  # same-type eval
                    _sub = _parH.Qd[_i]; sub = parH.Qd[_i+i]
                    if sub.n:
                        dsub = comp_ptuple(_sub, sub, fd)  # sub is vertuple, ptuple, or ext
                    else:
                        dsub = comp_parH(_sub, sub)  # keep unpacking aggH | subH | derH
                    dparH.valt[0]+=dsub.valt[0]; dparH.valt[1]+=dsub.valt[1]  # add rdnt?
                    dparH.Qd += [dsub]; dparH.Q += [_didx + d_didx]
                    dparH.fds += [fd]
                    last_i=i; last_idx=idx  # last matching i,idx
                    break
            elif _idx < idx:  # no dsub / _sub
                d_didx += didx  # += missing didx
                break  # no parH search beyond _idx
            # else _idx>idx: keep searching
            idx += 1  # 1 sub/loop
        _idx += 1
        if elev in (0,1) or not (_i+1)%(2**elev):  # first 2 levs are single-element, higher levs are 2**elev elements
            elev+=1  # elevation

    return dparH

def comp_ptuple(_ptuple, ptuple, fd):  # may be ptuple, vertuple, or ext

    dtuple=CQ(n=_ptuple.n)  # combine with ptuple.n?
    rn = _ptuple.n/ptuple.n  # normalize param as param*rn for n-invariant ratio: _param/ param*rn = (_param/_n)/(param/n)
    _idx, d_didx, last_i, last_idx = 0,0,-1,-1

    for _i, _didx in enumerate(_ptuple.Q):  # i: index in Qd: select param set, idx: index in full param set
        _idx += _didx; idx = last_idx+1
        for i, didx in enumerate(ptuple.Q[last_i+1:]):  # start after last matching i and idx
            idx += didx
            if _idx == idx:
                if ptuple.Qm: val = _ptuple.Qd[_i]+ptuple.Qd[_i+i] if fd else _ptuple.Qm[_i]+ptuple.Qm[_i+i]
                else:         val = aveG+1  # default comp for 0der pars
                if val > aveG:
                    _par, par = _ptuple.Qd[_i], ptuple.Qd[_i+i]
                    if isinstance(par,list):
                        if len(par)==4: m,d = comp_aangle(_par,par)
                        else: m,d = comp_angle(_par,par)
                    else:
                        m,d = comp_par(_par, par*rn, aves[idx], finv = not i and not ptuple.Qm)  # finv=0 for 0der I only
                    dtuple.Qm+=[m]; dtuple.Qd+=[d]; dtuple.Q+=[d_didx+_didx]
                    dtuple.valt[0]+=m; dtuple.valt[1]+=d  # no rdnt, rdn = m>d or d>m?)
                last_i=i; last_idx=idx  # last matching i,idx
                break
            elif _idx < idx:  # no dpar per _par
                d_didx += didx
                break  # no par search beyond current index
            # else _idx > idx: keep searching
            idx += 1
        _idx += 1
    return dtuple

def add_ext(box, L, extt):  # add ext per composition level
    y,x, y0,yn, x0,xn = box
    dY = yn-y0; dX = xn-x0
    box[:2] = y/L, x/L  # norm to ave
    extt += [[L, L/ dY*dX, [dY,dX]]]  # composed L,S,A, norm S = nodes per area

def sum2graph_(graph_, fd, fsub=0):  # sum node and link params into graph, derH in agg+ or player in sub+

    Graph_ = []  # Cgraphs
    for graph in graph_:  # CQs

        if graph.valt[fd] < aveG:  # form graph if val>min only
            continue
        Graph = Cgraph(fds=copy(graph.Q[0].fds)+[fd])  # incr der
        ''' if n roots: 
        sum_derH(Graph.uH[0][fd].derH,root.derH) or sum_G(Graph.uH[0][fd],root)? init if empty
        sum_H(Graph.uH[1:], root.uH)  # root of Graph, init if empty
        '''
        node_,Link_ = [],[]  # form G, keep iG:
        for iG in graph.Q:
            sum_G(Graph, iG, fmerge=0)  # local subset of lower Gs in new graph
            link_ = [iG.link_.Qm, iG.link_.Qd][fd]  # mlink_,dlink_
            Link_ = list(set(Link_ + link_))  # unique links in node_
            G = Cgraph(fds=copy(iG.fds)+[fd], root=Graph, node_=link_, box=copy(iG.box))  # no sub_nodes in derG, remove if <ave?
            for derG in link_:
                sum_box(G.box, derG.G[0].box if derG.G[1] is iG else derG.G[1].box)
                op_parH(G.aggH, derG.aggH, fcomp=0)  # two-fork derGs are not modified
                Graph.valt[0] += derG.valt[0]; Graph.valt[1] += derG.valt[1]
            # add_ext(G.box, len(link_), G.derH[-1])  # composed node ext, not in derG.derH (this should be not needed now)
            # if mult roots: sum_H(G.uH[1:], Graph.uH)
            node_ += [G]
        Graph.root = iG.root  # same root, lower derivation is higher composition
        Graph.node_ = node_  # G| G.G| G.G.G..
        for derG in Link_:  # sum unique links, not box
            op_parH(Graph.aggH, derG.aggH, fcomp=0)
            Graph.valt[0] += derG.valt[0]; Graph.valt[1] += derG.valt[1]
        # we already have ext assigned in comp_G_ (derG.aggH.Qd[0], which is Graph.aggH.Qd[0]), so there's no need to pack Ext here again?
        Ext = deepcopy(G.aggH.Qd[0])  # 1st Ext
        for G in node_[1:]:
            op_parH(Ext,G.aggH.Qd[0], fcomp=0)
        Graph.aggH.Qd.insert(0, Ext);  Graph.aggH.Q.insert(0,0)

        # if Graph.uH: Graph.val += sum([lev.val for lev in Graph.uH]) / sum([lev.rdn for lev in Graph.uH])  # if val>alt_val: rdn+=len_Q?
        Graph_ += [Graph]

    return Graph_

def sum_ptuple(Ptuple, ptuple, fneg=0):

    for pname, ave in zip(pnames, aves):
        Par = getattr(Ptuple, pname); par = getattr(ptuple, pname)

        if pname in ("angle","axis") and isinstance(Par, list):
            sin_da0 = (Par[0] * par[1]) + (Par[1] * par[0])  # sin(A+B)= (sinA*cosB)+(cosA*sinB)
            cos_da0 = (Par[1] * par[1]) - (Par[0] * par[0])  # cos(A+B)=(cosA*cosB)-(sinA*sinB)
            Par = [sin_da0, cos_da0]
        elif pname == "aangle" and isinstance(Par, list):
            _sin_da0, _cos_da0, _sin_da1, _cos_da1 = Par
            sin_da0, cos_da0, sin_da1, cos_da1 = par
            sin_dda0 = (_sin_da0 * cos_da0) + (_cos_da0 * sin_da0)
            cos_dda0 = (_cos_da0 * cos_da0) - (_sin_da0 * sin_da0)
            sin_dda1 = (_sin_da1 * cos_da1) + (_cos_da1 * sin_da1)
            cos_dda1 = (_cos_da1 * cos_da1) - (_sin_da1 * sin_da1)
            Par = [sin_dda0, cos_dda0, sin_dda1, cos_dda1]
        else:
            Par += (-par if fneg else par)
        setattr(Ptuple, pname, Par)

    Ptuple.valt[0] += ptuple.valt[0]; Ptuple.valt[1] += ptuple.valt[1]
    Ptuple.n += 1
