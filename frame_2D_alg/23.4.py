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


