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
