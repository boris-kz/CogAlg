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

def sub_recursion_eval(root, PP_, fd):  # fork PP_ in PP or blob, no derH in blob

    # add RVal=0, DVal=0 to return?
    term = 1
    for PP in PP_:
        if PP.valt[fd] > PP_aves[fd] * PP.rdnt[fd] and len(PP.P_) > ave_nsub:
            term = 0
            sub_recursion(PP)  # comp_der|rng in PP -> parLayer, sub_PPs
        elif isinstance(root, CPP):
            root.fback_ += [[PP.derH, PP.valt, PP.rdnt]]
    if term and isinstance(root, CPP):
        feedback(root, fd)  # upward recursive extend root.derT, forward eval only
        '''        
        for i, sG_ in enumerate(sG_t):
            val,rdn = 0,0
            for sub_G in sG_:
                val += sum(sub_G.valt); rdn += sum(sub_G.rdnt)
            Val = Valt[fd]+val; Rdn = Rdnt[fd]+rdn
        or
        Val = Valt[fd] + sum([sum(G.valt) for G in sG_])
        Rdn = Rdnt[fd] + sum([sum(G.rdnt) for G in sG_])
        '''
# old:
def op_parT(_graph, graph, fcomp, fneg=0):  # unpack aggH( subH( derH -> ptuples

    _parT, parT = _graph.parT, graph.parT

    if fcomp:
        dparT,valT,rdnT = comp_unpack(_parT, parT, rn=1)
        return dparT,valT,rdnT
    else:
        _valT, valT = _graph.valT, graph.valT
        _rdnT, rdnT = _graph.rdnT, graph.rdnT
        for i in 0,1:
            sum_unpack([_parT[i], _valT[i], _rdnT[i]], [parT[i], valT[i],rdnT[i]])

# same as comp|sum unpack?:
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
