# Link fork eval, projected by M of surrounding cluster:

vD = D * (M / ave)  # project by borrow from rel M
vM = M - vD / 2  # cancel by lend to D


def add_H(HE, He_, root=None, ri=None, sign=1):  # unpack derHs down to numericals and sum them

    if not isinstance(He_, list): He_ = [He_]
    for He in He_:
        if HE:
            for i, (Lay, lay) in enumerate(
                    zip_longest(HE.H, He.H, fillvalue=None)):  # cross comp layer
                if lay:
                    if Lay:
                        Lay.add_H(lay)
                    else:
                        if Lay is None:
                            HE.append_(lay.copy_(root=HE))  # pack a copy of new lay in HE.H
                        else:
                            HE.H[i] = lay.copy_(root=HE)  # Lay was []
            HE.add_lay(He, sign)
            HE.node_ += [node for node in He.node_ if
                         node not in HE.node_]  # node_ is empty in CL derH?
        elif root:
            if ri is None:  # root is node
                root.derH = He.copy_(root=root)
            else:
                root.H[ri] = He.copy_(root=root)
            root.Et += He.Et
        else:
            HE = He.copy_(root=root)

    return HE  # root should be updated by returned HE, we don't need it here?

def comp_N(_N,N, rn, angle=None, dist=None, dir=None):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = dir is not None  # compared links have binary relative direction
    dir = 1 if dir is None else dir  # convert to numeric
    # comp externals:
    if fd:
        _L, L = _N.dist, N.dist;  dL = _L-L; mL = min(_L,L) - ave_L  # direct match
        mA,dA = comp_angle(_N.angle, [d*dir for d in N.angle])  # rev 2nd link in llink
        # comp med if LL: isinstance(>nodet[0],CL), higher-order version of comp dist?
    else:
        _L, L = len(_N.node_),len(N.node_); dL = _L-L; mL = min(_L,L) - ave_L
        mA,dA = comp_area(_N.box, N.box)  # compare area in CG vs angle in CL
    n = .3
    M = mL + mA; D = (abs(dL)+abs(dA)) * (M/ave); M = M - D/2
    Et = np.array([M,D])
    md_t = [np.array([np.array([mL,mA]), np.array([dL,dA]), Et, n], dtype=object)]  # init as [mdExt]
    if not fd:  # CG
        mdLat = comp_latuple(_N.latuple,N.latuple, _N.n, N.n)  # not sure about ns
        mdLay = comp_md_(_N.mdLay[1], N.mdLay[1], rn, dir)
        md_t += [mdLat,mdLay]; Et += mdLat[2] + mdLay[2]; n += mdLat[3] + mdLay[3]
    # | n = (_n+n)/2?
    Et = np.append(Et, (_N.Et[2]+N.Et[2])/2 )  # Et[0] += ave_rn - rn?
    subLay = CH(n=n, md_t=md_t, Et=Et)
    eLay = CH(H=[subLay], n=n, md_t=deepcopy(md_t), Et=copy(Et))
    if _N.derH and N.derH:
        dderH = _N.derH.comp_H(N.derH, rn, dir=dir)  # comp shared layers
        eLay.append_(dderH, flat=1)
    elif _N.derH: eLay.append_(_N.derH.copy_(root=eLay))  # one empty derH
    elif  N.derH: eLay.append_(N.derH.copy_(root=eLay,rev=1))
    # spec: comp_node_(node_|link_), combinatorial, node_ may be nested with rng-)agg+, graph similarity search?
    Et = copy(eLay.Et)
    if not fd and _N.altG and N.altG:  # not for CL, eval M?
        altLink = comp_N(_N.altG, N.altG, _N.altG.n/N.altG.n)  # no angle,dist, init alternating PPds | dPs?
        eLay.altH = altLink.derH
        Et += eLay.altH.Et
    Link = CL(nodet=[_N,N],derH=eLay, n=min(_N.n,N.n),yx=np.add(_N.yx,N.yx)/2, angle=angle,dist=dist,box=extend_box(N.box,_N.box))
    if Et[0] > ave * Et[2]:
        eLay.root = Link
        for rev, node in zip((0,1), (N,_N)):  # reverse Link direction for _N
            if fd: node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            else:  node.rim += [(Link, rev)]
            node.extH.add_H(Link.derH, root=node.extH)
            # flat
    return Link

def copy_(_He, root, rev=0):
        # comp direction may be reversed
        He = CH(root=root, node_=copy(_He.node_), n=_He.n, i=_He.i, i_=copy(_He.i_))
        He.Et = copy(_He.Et)
        He.md_t = deepcopy(_He.md_t)
        if rev:
            for _,d_,_,_ in He.md_t:  # mdExt, possibly mdLat, mdLay
               d_ *= -1   # negate ds
        for he in _He.H:
            He.H += [he.copy_(root=He, rev=rev)] if he else [[]]
        return He

def comp_lay(_md_t, md_t, rn, dir=1):  # replace dir with rev?

    der_md_t = []
    tM, tD = 0, 0
    for i, (_md_, md_) in enumerate(zip(_md_t, md_t)):  # [mdExt, possibly mdLat, mdLay], default per layer
        if i == 2:  # nested mdLay
            sub_md_t, (m,d) = comp_lay(_md_[0], md_[0], rn, dir=dir)  # nested mdLay is [md_t, [tM, tD]], skip [tM and tD]
            der_md_t += [[sub_md_t, (m,d)]]
            tM += m; tD += d

    def comp_H(_He, He, dir=1):  # unpack each layer of CH down to numericals and compare each pair

        der_md_t, et = comp_md_t(_He.md_t, He.md_t, _He.n / He.n, dir=1)
        # ini:
        DLay = CH(md_t=der_md_t, Et=np.append(et,(_He.Et[2]+He.Et[2])/2), n=.3 if len(der_md_t)==1 else 2.3)  # .3 in default comp ext
        # empty H in bottom | deprecated layer:
        for rev, _lay, lay in zip((0,1), _He.H, He.H):  #  fork & layer CH / rng+|der+, flat
            if _lay and lay:
                dLay = _lay.comp_H(lay, dir)  # comp He.md_t, comp,unpack lay.H
                DLay.append_(dLay)  # DLay.H += subLay
            else:
                l = _lay if _lay else lay  # only one not empty lay = difference between lays:
                if l: DLay.append_(l.copy_(root=DLay, rev=rev))
                else: DLay.H += [CH()]  # to trace fork types
            # nested subHH ( subH?
        return DLay

    def comp_He(_He, He, rn, root, dir=1):

        derH = CH(root=root)
        # lay.H maps to higher Hs it was derived from, len lay.H = 2 ^ lay_depth (unpacked root H[:i])
        for _he, he in zip(_He.H, He.H):
            if he:
                if _he:  # maybe empty CH to trace fork types
                    if isinstance(he.H[0], CH):
                        Lay = _he.comp_He(he, rn, root=derH, dir=dir)  # deeper unpack -> comp_md_t
                    else:
                        Lay = _he.comp_md_t(he, rn=rn, root=derH, dir=dir)  # comp shared layers
                    derH.append_(Lay, flat=1)
                else:
                    derH.append_(he.copy_(root=derH, dir=1))  # not sure we need to copy?
        # old comp_H:
        for dir, _he, he in zip((1,-1), _He.H, He.H):  #  fork & layer CH / rng+|der+, flat
            if _he and he:
                Lay = _he.comp_He(he, root=derH, dir=dir)  # unpack lay.H
                derH.append_(Lay)
            else:
                l = _he if _he else he  # only one not empty lay = difference between lays:
                if l: derH.append_(l.copy_(root=derH, dir=dir))
                else: derH.H += [CH()]  # to trace fork types

        return derH

    def add_H(HE, He_, sign=1):  # unpack derHs down to numericals and sum them

        if not isinstance(He_,list): He_ = [He_]
        for He in He_:
            for i, (Lay,lay) in enumerate(zip_longest(HE.H, He.H, fillvalue=None)):  # cross comp layer
                if lay:
                    if Lay is None: HE.append_(lay.copy_(root=HE))  # pack a copy of new lay in HE.H
                    else:           Lay.add_H(lay)  # depth-first, He in add_lay has summed all the nested lays
            HE.add_lay(He, sign)
            HE.node_ += [node for node in He.node_ if node not in HE.node_]  # node_ is empty in CL derH?

        return HE  # root should be updated by returned HE
