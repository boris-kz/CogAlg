def comp_N(Link, iEt, rng, rev=None):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = rev is not None  # compared links have binary relative direction?
    _N, N = Link.nodet; rn = _N.n / N.n

    if fd:  # CLs, form single layer:
        DLay = _N.derH.comp_H(N.derH, rn, fagg=1)  # new link derH = local dH
        N.mdext = comp_ext(2,2, _N.S,N.S/rn, _N.angle, N.angle if rev else [-d for d in N.angle])  # reverse for left link
        N.Et = np.array(N.Et) + DLay.Et + N.mdext.Et
        DLay.root = Link
    else:   # CGs
        mdlat = comp_latuple(_N.latuple, N.latuple, rn,fagg=1)
        mdLay = _N.mdLay.comp_md_(N.mdLay, rn, fagg=1)  # not in CG of links?
        mdext = comp_ext(len(_N.node_), len(N.node_), _N.S, N.S / rn, _N.A, N.A)
        DLay = CH(  # 1st derLay
            md_t=[mdlat,mdLay,mdext], Et=np.array(mdlat.Et)+mdLay.Et+mdext.Et, Rt=np.array(mdlat.Rt)+mdLay.Rt+mdext.Rt, n=2.5)
        mdlat.root = DLay; mdLay.root = DLay; mdext.root = DLay
        if _N.derH and N.derH:
            dLay = _N.derH.comp_H(N.derH,rn,fagg=1,frev=rev)
            DLay.add_H(dLay)  # also append discrete higher subLays in dLay.H_[0], if any?
            # no comp extH: current ders
    if fd: Link.derH.append_(DLay)
    else:  Link.derH = DLay
    iEt[:] = np.add(iEt,DLay.Et)  # init eval rng+ and form_graph_t by total m|d?
    for i in 0,1:
        Val, Rdn = DLay.Et[i::2]
        if Val > G_aves[i] * Rdn * (rng+1): Link.ft[i] = 1  # fork inclusion tuple
        _N.Et[i] += Val; N.Et[i] += Val  # not selective
        _N.Et[2+i] += Rdn; N.Et[2+i] += Rdn  # per fork link in both Gs
        # if select fork links: iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dH.Et[i::2])]
    if any(Link.ft):
        Link.n = min(_N.n,N.n)  # comp shared layers
        Link.yx = np.add(_N.yx, N.yx) / 2
        Link.S += (_N.S + N.S) / 2
        for rev, node in zip((0,1),(_N,N)):  # ?reversed Link direction
            if fd:
                if len(node.rimt_) == rng:
                    node.rimt_[-1][1-rev] += [[Link,rev]]  # add in last rng layer, opposite to _N,N dir
                else: node.rimt_ += [[[[Link,rev]],[]]] if dir else [[[],[[Link,rev]]]]  # add rng layer
            else:
                if len(node.extH.H) == rng:
                    node.rim_[-1] += [[Link, rev]]  # accum last rng layer
                else: node.rim_ += [[[Link, rev]]]  # init rng layer

        return True

def sum_N_(N_, fd=0):  # sum partial grapht in merge

    N = N_[0]
    n = N.n; S = N.S
    L, A = (N.span, N.angle) if fd else (len(N.node_), N.A)
    if not fd:
        latuple = deepcopy(N.latuple)
        mdLay = CH().copy(N.mdLay)
        extH = CH().copy(N.extH)
    derH = CH().copy(N.derH)
    # Et = copy(N.Et)
    for N in N_[1:]:
        if not fd:
            add_lat(latuple, N.latuple)
            if N.mdLay: mdLay.add_md_(N.mdLay)
        n += N.n; S += N.S
        L += N.span if fd else len(N.node_)
        A = [Angle+angle for Angle,angle in zip(A, N.angle if fd else N.A)]
        if N.derH: derH.add_H(N.derH)
        if N.extH: extH.add_H(N.extH)

    if fd: return n, L, S, A, derH, extH
    else:  return n, L, S, A, derH, extH, latuple, mdLay  # no comp Et

def comp_core(_node_, node_, fmerge=1):  # compare partial graphs in merge or kLay in rng_kern_

    dderH = CH()
    fd = isinstance(_node_[0], CL)
    _pars = sum_N_(_node_,fd) if fmerge else _node_
    pars = sum_N_(node_,fd) if fmerge else node_
    n, L, S, A, derH = pars[:5]; _n,_L,_S,_A,_derH = _pars[:5]
    rn = _n/n
    mdext = comp_ext(_L,L, _S,S/rn, _A,A)
    dderH.n = mdext.n;  dderH.Et = np.array(mdext.Et); dderH.Rt = np.array(mdext.Rt)
    if fd:
        dderH.H = [[mdext]]
    else:
        _latuple, _mdLay = _pars[5:]; latuple, mdLay = pars[5:]
        if any(_latuple[:5]) and any(latuple[:5]):  # latuple is empty in CL
            mdlat = comp_latuple(_latuple, latuple, rn, fagg=1)
            dderH.n+=mdlat.n; dderH.Et+=mdlat.Et; dderH.Rt+=mdlat.Rt
        if _mdLay and mdLay:
            mdlay = _mdLay.comp_md_(mdLay, rn, fagg=1)
            dderH.n+=mdlay.n; dderH.Et+=mdlay.Et; dderH.Rt+=mdlay.Rt
    if _derH and derH:
        ddderH = _derH.comp_H(derH, rn, fagg=1)  # append and sum new dderH to base dderH
        dderH.H += ddderH.H  # merge derlay
        dderH.n+=ddderH.n; dderH.Et+=ddderH.Et; dderH.Rt+=ddderH.Rt

    return dderH

def comp_L(_pars, pars):  # compare partial graphs in merge

    _n,_L,_S,_A,_derH = _pars; n, L, S, A, derH = pars
    rn = _n / n
    dderH = derH.comp_H(derH, rn, fagg=1)  # new link derH = local dH
    mdext = comp_ext(_L,L, _S,S/rn, _A,A)
    return dderH, mdext

