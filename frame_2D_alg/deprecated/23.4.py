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

def comp_P(_P, P):  # forms vertical derivatives of params per P in _P.uplink, conditional ders from norm and DIV comp

    if isinstance(_P, CP):
        vertuple = comp_ptuple(_P.ptuple, P.ptuple)
        derQ = [vertuple]; Valt=copy(vertuple.valt); Rdnt=copy(vertuple.rdnt)
        L = len(_P.dert_)
    else:  # P is derP
        derQ=[]; Valt=[0,0]; Rdnt=[1,1]
        for _ptuple, ptuple in zip(_P.derQ, P.derQ):
            dtuple, rdnt, valt = comp_vertuple(_ptuple, ptuple)
            derQ+=[dtuple]; Valt[0]+=valt[0]; Valt[1]+=valt[1]; Rdnt[0]+=rdnt[0]; Rdnt[1]+=rdnt[1]
        L = _P.L

    # derP is single-layer, links are compared individually, but higher layers have multiple vertuples?
    return CderP(derQ=derQ, valt=Valt, rdnt=Rdnt, P=P, _P=_P, x0=_P.x0, y0=_P.y0, L=L)

def comp_GQ(_G, G):  # compare lower-derivation G.G.s, pack results in mderH_,dderH_

    dpH_ = CQ(); Tval= aveG+1

    while (_G and G) and Tval > aveG:  # same-scope if sub+, no agg+ G.G
        dpH = comp_G(_G, G)
        dpH_.Qd += [dpH]; dpH_.Q += [0]
        for i in 0,1:
            dpH_.valt[i] += dpH.valt[i]; dpH_.rdnt[i] += dpH.rdnt[i]
        _G = _G.G; G = G.G
        Tval = sum(dpH_.valt) / sum(dpH_.rdnt)

    return dpH_  # ext added in comp_G_, not within derH

def comp_G(_G, G):  # in derH

    Mval, Dval = 0,0
    Mrdn, Drdn = 1,1
    _pH, pH = _G.pH, G.pH  # same for G or derG now

    dpH = op_parH(_pH, pH, fcomp=1)
    # spec:
    _node_, node_ = _G.node_, G.node_  # link_ if fd, sub_node should be empty
    # below is not updated
    if (Mval+Dval)* sum(_G.pH.valt)*sum(G.pH.valt) * len(_node_)*len(node_) > aveG:  # / rdn?

        sub_dderH, mval, dval, mrdn, drdn = comp_G_(_node_, node_, f1Q=0)
        Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn
        # pack m|dnode_ in m|dderH: implicit?

    else: _G.fterm=1  # no G.fterm=1: it has it's own specification?
    return dpH

def sub_recursion_eval(root):  # for PP or dir_blob

    root_PPm_, root_PPd_ = root.rlayers[-1], root.dlayers[-1]
    for fd, PP_ in enumerate([root_PPm_, root_PPd_]):
        mcomb_layers, dcomb_layers, PPm_, PPd_ = [], [], [], []

        for PP in PP_:
            # fd = _P.valt[1]+P.valt[1] > _P.valt[0]+_P.valt[0]  # if exclusive comp fork per latuple in P| vertuple in derP?
            if fd:  # add root to derP for der+:
                for P_ in PP.P__[1:-1]:  # skip 1st and last row
                    for P in P_:
                        for derP in P.uplink_layers[-1][fd]:
                            derP.roott[fd] = PP
                comb_layers = dcomb_layers; PP_layers = PP.dlayers; PPd_ += [PP]
            else:
                comb_layers = mcomb_layers; PP_layers = PP.rlayers; PPm_ += [PP]

            val = PP.valt[fd]; alt_val = sum([alt_PP.valt[fd] for alt_PP in PP.alt_PP_]) if PP.alt_PP_ else 0   # for fork rdn:
            ave = PP_aves[fd] * (PP.rdnt[fd] + 1 + (alt_val > val))
            if val > ave and len(PP.P__) > ave_nsub:
                sub_recursion(PP)  # comp_P_der | comp_P_rng in PPs -> param_layer, sub_PPs
                ave*=2  # 1+PP.rdn incr
                # splice deeper layers between PPs into comb_layers:
                for i, (comb_layer, PP_layer) in enumerate(zip_longest(comb_layers, PP_layers, fillvalue=[])):
                    if PP_layer:
                        if i > len(comb_layers) - 1: comb_layers += [PP_layer]  # add new r|d layer
                        else: comb_layers[i] += PP_layer  # splice r|d PP layer into existing layer
            # segs:
            agg_recursion_eval(PP, [copy(PP.mseg_levels[-1]), copy(PP.dseg_levels[-1])])
            # include empty comb_layers:
            if fd:
                PPmm_ = [PPm_] + mcomb_layers; mVal = sum([PP.valt[0] for PP_ in PPmm_ for PP in PP_])
                PPmd_ = [PPm_] + dcomb_layers; dVal = sum([PP.valt[1] for PP_ in PPmd_ for PP in PP_])
                root.dlayers = [PPmd_,PPmm_]
            else:
                PPdm_ = [PPm_] + mcomb_layers; mVal = sum([PP.valt[0] for PP_ in PPdm_ for PP in PP_])
                PPdd_ = [PPd_] + dcomb_layers; dVal = sum([PP.valt[1] for PP_ in PPdd_ for PP in PP_])
                root.rlayers = [PPdm_, PPdd_]
            # or higher der val?
            if isinstance(root, CPP):  # root is CPP
                for i in 0,1:
                    root.valt[i] += PP.valt[i]  # vals
                    root.rdnt[i] += PP.rdnt[i]  # ad rdn too?
            else:  # root is CBlob
                if fd: root.G += sum([alt_PP.valt[fd] for alt_PP in PP.alt_PP_]) if PP.alt_PP_ else 0
                else:  root.M += PP.valt[fd]


def sub_recursion(PP):  # evaluate PP for rng+ and der+

    P__  = [P_ for P_ in reversed(PP.P__)]  # revert bottom-up to top-down
    P__ = comp_P_der(P__) if PP.fds[-1] else comp_P_rng(P__, PP.rng + 1)   # returns top-down
    PP.rdnt[PP.fds[-1] ] += 1  # two-fork rdn, priority is not known?  rotate?

    sub_segm_ = form_seg_root([copy(P_) for P_ in P__], fd=0, fds=PP.fds)
    sub_segd_ = form_seg_root([copy(P_) for P_ in P__], fd=1, fds=PP.fds)  # returns bottom-up
    # sub_PPm_, sub_PPd_:
    sub_PPm_, sub_PPd_ = form_PP_root((sub_segm_, sub_segd_), PP.rdnt[PP.fds[-1]] + 1)
    PP.rlayers[:] = [sub_PPm_]; PP.dlayers[:] = [sub_PPd_]

    sub_recursion_eval(PP)  # add rlayers, dlayers, seg_levels to select sub_PPs

def comp_P_rng(P__, rng):  # rng+ sub_recursion in PP.P__, switch to rng+n to skip clustering?

    for P_ in P__:
        for P in P_:  # add 2 link layers: rng_derP_ and match_rng_derP_:
            P.uplink_layers += [[],[[],[]]]; P.downlink_layers += [[],[[],[]]]

    for y, _P_ in enumerate(P__[:-rng]):  # higher compared row, skip last rng: no lower comparand rows
        for x, _P in enumerate(_P_):
            # get linked Ps at dy = rng-1:
            for pri_derP in _P.downlink_layers[-3][0]:  # fd always = 0 here
                pri_P = pri_derP.P
                # compare linked Ps at dy = rng:
                for ini_derP in pri_P.downlink_layers[0]:
                    P = ini_derP.P
                    # add new Ps, their link layers and reset their roots:
                    if P not in [P for P_ in P__ for P in P_]:
                        append_P(P__, P)
                        P.uplink_layers += [[],[[],[]]]; P.downlink_layers += [[],[[],[]]]
                    derP = comp_P(_P, P)
                    P.uplink_layers[-2] += [derP]
                    _P.downlink_layers[-2] += [derP]

    return P__

def comp_P_der(P__):  # der+ sub_recursion in PP.P__, compare P.uplinks to P.downlinks

    if isinstance(P__[0][0].ptuple, Cptuple):
        for P_ in P__:
            for P in P_: P.ptuple = [P.ptuple]

    dderPs__ = []  # derP__ = [[] for P_ in P__[:-1]]  # init derP rows, exclude bottom P row
    for P_ in P__[1:-1]:  # higher compared row, exclude 1st: no +ve uplinks, and last: no +ve downlinks
        # not revised:
        dderPs_ = []  # row of dderPs
        for P in P_:
            if isinstance(P.ptuple, Cptuple): P.ptuple = [P.ptuple]
            dderPs = []  # dderP for each _derP, derP pair in P links
            for _derP in P.uplink_layers[-1][1]:  # fd=1
                for derP in P.downlink_layers[-1][1]:
                    # there maybe no x overlap between recomputed Ls of _derP and derP, compare anyway,
                    # mderP * (ave_olp_L / olp_L)? or olp(_derP._P.L, derP.P.L)?
                    # gap: neg_olp, ave = olp-neg_olp?
                    dderP = comp_P(_derP, derP)  # form higher vertical derivatives of derP.players,
                    # or comp derP.players[1] only: it's already diffs of all lower players?
                    derP.uplink_layers[0] += [dderP]  # pre-init layer per derP
                    _derP.downlink_layers[0] += [dderP]
                    dderPs_ += [dderP]  # actually it could be dderPs_ ++ [derPP]
                # compute x overlap between dderP'__P and P, in form_seg_ or comp_layer?
            dderPs_ += dderPs  # row of dderPs
        dderPs__ += [dderPs_]
    return dderPs__

def comp_der(derP):  # forms vertical derivatives of params per P in _P.uplink, conditional ders from norm and DIV comp
    _P, P = derP._P, derP.P
    # tentative:
    elev = int( np.sqrt( len(derP.derQ)-1))
    i = elev if elev < 3 else 2 ** (elev - 1)  # elev counts from 0, init index = 0,1,2,4,8...
    j = elev + 1 if elev < 2 else 2 ** elev  # last index
    if len(_P.ptuple)>j-1 and len(P.ptuple)>j-1: # ptuples are derHs, extend derP.derQ:
        comp_layer(derP, i,j)

def comp_rng(derP):  # forms vertical derivatives of params per P in _P.uplink, conditional ders from norm and DIV comp
    _P, P = derP._P, derP.P
    if isinstance(P.ptuple, Cptuple):
        vertuple = comp_ptuple(_P.ptuple, P.ptuple)
        derP.derQ = [vertuple]; derP.valt = copy(vertuple.valt); derP.rdnt = copy(vertuple.rdnt)
    else:
        comp_layer(derP, 0, min(len(_P.derH)-1,len(P.derH)-1))  # this is actually comp derH, works the same here
    derP.L = len(_P.dert_)


def form_seg_root(P__, fd, fds):  # form segs from Ps

    for P_ in P__[1:]:  # scan bottom-up, append link_layers[-1] with branch-rdn adjusted matches in link_layers[-2]:
        for P in P_: link_eval(P.uplink_layers, fd)  # uplinks_layers[-2] matches -> uplinks_layers[-1]
                     # forms both uplink and downlink layers[-1]
    seg_ = []
    for P_ in reversed(P__):  # get a row of Ps bottom-up, different copies per fPd
        while P_:
            P = P_.pop(0)
            if P.uplink_layers[-1][fd]:  # last matching derPs layer is not empty
                form_seg_(seg_, P__, [P], fd, fds)  # test P.matching_uplink_, not known in form_seg_root
            else:
                seg_.append( sum2seg([P], fd, fds))  # no link_s, terminate seg_Ps = [P]
    return seg_

def link_eval(link_layers, fd):
    # sort derPs in link_layers[-2] by their value param:
    derP_ = sorted( link_layers[-2], key=lambda derP: derP.valt[fd], reverse=True)

    for i, derP in enumerate(derP_):
        if not fd:
            rng_eval(derP, fd)  # reset derP.valt, derP.rdn
        mrdn = derP.valt[1-fd] > derP.valt[fd]  # sum because they are valt
        derP.rdnt[fd] += not mrdn if fd else mrdn

        if derP.valt[fd] > vaves[fd] * derP.rdnt[fd] * (i+1):  # ave * rdn to stronger derPs in link_layers[-2]
            link_layers[-1][fd].append(derP)
            derP._P.downlink_layers[-1][fd] += [derP]
            # misses = link_layers[-2] not in link_layers[-1], sum as PP.nvalt[fd] in sum2seg and sum2PP
# ?
def rng_eval(derP, fd):  # compute value of combined mutual derPs: overlap between P uplinks and _P downlinks

    _P, P = derP._P, derP.P
    common_derP_ = []

    for _downlink_layer, uplink_layer in zip(_P.downlink_layers[1::2], P.uplink_layers[1::2]):
        # overlap between +ve P uplinks and +ve _P downlinks:
        common_derP_ += list( set(_downlink_layer[fd]).intersection(uplink_layer[fd]))
    rdn = 1
    olp_val = 0
    nolp = len(common_derP_)
    for derP in common_derP_:
        rdn += derP.valt[fd] > derP.valt[1-fd]
        olp_val += derP.valt[fd]  # olp_val not reset for every derP?
        derP.valt[fd] = olp_val / nolp
    '''
    for i, derP in enumerate( sorted( link_layers[-2], key=lambda derP: derP.params[fPd].val, reverse=True)):
    if fPd: derP.rdn += derP.params[fPd].val > derP.params[1-fPd].val  # mP > dP
    else: rng_eval(derP, fPd)  # reset derP.val, derP.rdn
    if derP.params[fPd].val > vaves[fPd] * derP.rdn * (i+1):  # ave * rdn to stronger derPs in link_layers[-2]
    '''
#    derP.rdn += (rdn / nolp) > .5  # no fractional rdn?

def form_seg_(seg_, P__, seg_Ps, fd, fds):  # form contiguous segments of vertically matching Ps

    if len(seg_Ps[-1].uplink_layers[-1][fd]) > 1:  # terminate seg
        seg_.append( sum2seg( seg_Ps, fd, fds))  # convert seg_Ps to CPP seg
    else:
        uplink_ = seg_Ps[-1].uplink_layers[-1][fd]
        if uplink_ and len(uplink_[0]._P.downlink_layers[-1][fd])==1:
            # one P.uplink AND one _P.downlink: add _P to seg, uplink_[0] is sole upderP:
            P = uplink_[0]._P
            [P_.remove(P) for P_ in P__ if P in P_]  # remove P from P__ so it's not inputted in form_seg_root
            seg_Ps += [P]  # if P.downlinks in seg_down_misses += [P]
            if seg_Ps[-1].uplink_layers[-1][fd]:
                form_seg_(seg_, P__, seg_Ps, fd, fds)  # recursive compare sign of next-layer uplinks
            else:
                seg_.append( sum2seg(seg_Ps, fd, fds))
        else:
            seg_.append( sum2seg(seg_Ps, fd, fds))  # terminate seg at 0 matching uplink
'''
    mseg_levels = list  # from 1st agg_recursion[fPd], seg_levels[0] is seg_, higher seg_levels are segP_..s
    dseg_levels = list
    uplink_layers = lambda: [[]]  # the links here will be derPPs from discontinuous comp, not in layers?
    downlink_layers = lambda: [[]]

'''

