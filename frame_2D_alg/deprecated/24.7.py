def rng_link_(N_):  # comp Clinks: der+'rng+ in root.link_ rim_t node rims: directional and node-mediated link tracing

    _mN_t_ = [[[N.nodet[0]],[N.nodet[1]]] for N in N_]  # rim-mediating nodes
    rng = 1; L_ = N_[:]
    Et = [0,0,0,0]
    while True:
        mN_t_ = [[[],[]] for _ in L_]
        for L, _mN_t, mN_t in zip(L_, _mN_t_, mN_t_):
            for rev, _mN_, mN_ in zip((0,1), _mN_t, mN_t):
                # comp L, _Ls: nodet mN 1st rim, -> rng+ _Ls/ rng+ mm..Ns:
                rim_ = [n.rim if isinstance(n,CG) else n.rimt_[0][0] + n.rimt_[0][1] for n in _mN_]
                for rim in rim_:
                    for _L,_rev in rim:  # _L is reversed relative to its 2nd node
                        if _L is L or _L in L.compared_: continue
                        if not hasattr(_L,"rimt_"): add_der_attrs(link_=[_L])  # _L not in root.link_, same derivation
                        L.compared_ += [_L]; _L.compared_ += [L]
                        dy,dx = np.subtract(_L.yx, L.yx)
                        Link = Clink(nodet=[_L,L], span=2, angle=[dy,dx], box=extend_box(_L.box, L.box))
                        # L.rim_t += new Link
                        if comp_N(Link, Et, rng, rev^_rev):  # negate ds if only one L is reversed
                            # add rng+ mediating nodes to L, link order: nodet < L < rim_t, mN.rim || L
                            mN_ += _L.nodet  # get _Ls in mN.rim
                            if _L not in L_:  # not in root
                                L_ += [_L]; mN_t_ += [[[],[]]]
                            mN_t_[L_.index(_L)][1-rev] += L.nodet
        _L_, _mN_t = [],[]
        for L, mN_t in zip(L_, mN_t_):
            if any(mN_t):
                _L_ += [L]; _mN_t_ += [mN_t]
        if _L_:
            L_ = _L_; rng += 1
        else:
            break
        # Lt_ = [(L, mN_t) for L, mN_t in zip(L_, mN_t_) if any(mN_t)]
        # if Lt_: L_,_mN_t_ = map(list, zip(*Lt_))  # map list to convert tuple from zip(*)

    return N_, Et, rng

def comp_N(Link, iEt, rng, rev=None):  # dir if fd, Link+=dderH, comparand rim+=Link

    fd = rev is not None  # compared links have binary relative direction?
    dH = CH(); _N, N = Link.nodet; rn = _N.n / N.n

    if fd:  # Clink Ns
        _N.derH.comp_(N.derH, dH, rn, fagg=1, flat=1, frev=rev)  # dH += [*dderH]
        # reverse angle direction for left link:
        _A, A = _N.angle, N.angle if rev else [-d for d in N.angle]
        Et, rt, md_ = comp_ext(2,2, _N.S,N.S/rn, _A,A)  # 2 nodes in nodet
        dH.append_(CH(Et=Et,relt=rt,H=md_,n=0.5,root=dH),flat=0)  # dH += [dext]
    else:  # CG Ns
        et, rt, md_ = comp_latuple(_N.latuple, N.latuple, rn,fagg=1)
        dH.append_(CH(Et=et,relt=rt,H=md_,n=1,root=dH),flat=0)  # dH = [dlatuple], or also pack in derH? then same sequence as in fd?
        _N.derH.comp_(N.derH, dH,rn,fagg=1,flat=1,frev=rev)     # dH += [*dderH]
        Et, Rt, Md_ = comp_ext(len(_N.node_),len(N.node_), _N.S,N.S/rn,_N.A,N.A)
        dH.append_(CH(Et=Et,relt=Rt,H=Md_,n=0.5,root=dH),flat=0)  # dH += [dext]
    # / N, if >1 PPs | Gs:
    if _N.extH and N.extH:
        _N.extH.comp_(N.extH, dH, rn, fagg=1, flat=1, frev=rev)
    # link.derH += dH:
    if fd: Link.derH.append_(dH, flat=1)
    else:  Link.derH = dH
    iEt[:] = np.add(iEt,dH.Et)  # init eval rng+ and form_graph_t by total m|d?
    fin = 0
    for i in 0,1:
        Val, Rdn = dH.Et[i::2]
        if Val > G_aves[i] * Rdn: fin = 1
        _N.Et[i] += Val; N.Et[i] += Val  # not selective
        _N.Et[2+i] += Rdn; N.Et[2+i] += Rdn  # per fork link in both Gs
        # if select fork links: iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dH.Et[i::2])]
    if fin:
        Link.n = min(_N.n,N.n)  # comp shared layers
        Link.yx = np.add(_N.yx, N.yx) / 2
        Link.S += (_N.S + N.S) / 2
        for rev, node in zip((0,1),(_N,N)):  # ?reversed Link direction
            if fd:
                if len(node.rimt_) == rng:
                    node.rimt_[-1][1-rev] += [[Link,rev]]  # add in last rng layer, opposite to _N,N dir
                else:
                    node.rimt_ += [[[[Link,rev]],[]]] if dir else [[[],[[Link,rev]]]]  # add rng layer
            else:
                node.rim += [[Link,rev]]
                if len(node.extH.H)==rng:
                    node.extH.H[-1].H[-1].add_(Link.derH)  # accum last layer
                else:
                    rngLay = CH()
                    rngLay.append_(Link.derH, flat=0)
                    node.extH.append_(rngLay, flat=0)  # init last layer
        return True

class CH(CBase):  # generic derivation hierarchy with variable nesting
    '''
    len layer +extt: 2, 3, 6, 12, 24,
    or without extt: 1, 1, 2, 4, 8..: max n of tuples per der layer = summed n of tuples in all lower layers:
    lay1: par     # derH per param in vertuple, layer is derivatives of all lower layers:
    lay2: [m,d]   # implicit nesting, brackets for clarity:
    lay3: [[m,d], [md,dd]]: 2 sLays,
    lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLays, <=2 ssLays
    '''
    name = "H"
    def __init__(He, n=0, Et=None, relt=None, H=None, root=None):
        super().__init__()
        # He.nest = nest  # nesting depth: -1/ ext, 0/ md_, 1/ derH, 2/ subH, 3/ aggH
        He.n = n  # total number of params compared to form derH, summed in comp_G and then from nodes in sum2graph
        He.Et = [0,0,0,0] if Et is None else Et   # evaluation tuple: valt, rdnt
        He.relt = [0,0] if relt is None else relt  # m,d relative to max possible m,d
        He.H = [] if H is None else H  # hierarchy of der layers or md_
        #| HT: [[],..] with same root?
        He.root = None if root is None else root

    def __bool__(H): return H.n != 0

    # below is not updated for nested H structure:

    def add_(HE, He, irdnt=None):  # unpack down to numericals and sum them

        if irdnt is None: irdnt = []
        if HE:
            if isinstance(HE.H[0], CH):
                H = []
                for Lay,lay in zip_longest(HE.H, He.H, fillvalue=None):
                    if lay is not None:  # to be summed
                        if Lay:
                            if lay: Lay.add_(lay, irdnt)  # recursive unpack to sum md_s
                        else:       Lay = deepcopy(lay) if lay else []  # deleted kernel lays
                    if Lay:  # may be empty
                        Lay.root = HE
                    H += [Lay]
                HE.H = H
            else:
                HE.H = [V+v for V,v in zip_longest(HE.H, He.H, fillvalue=0)]  # both Hs are md_s
            # default:
            HE.Et = np.add(HE.Et, He.Et); HE.relt = np.add(HE.relt, He.relt)
            if any(irdnt): HE.Et[2:] = [E+e for E,e in zip(HE.Et[2:], irdnt)]
            HE.n += He.n  # combined param accumulation span
        else:
            HE.copy(He)  # initialization
        if HE.root is not None: HE.root.update_root(He)

    def append_(HE,He, irdnt=None, flat=0):

        if irdnt is None: irdnt = []
        if flat:
            for H in He.H:
                if isinstance(H, CH): H.root = HE
            HE.H += He.H  # append flat
        else:
            He.root = HE
            HE.H += [He]  # append nested
        Et, et = HE.Et, He.Et
        HE.Et = np.add(HE.Et, He.Et); HE.relt = np.add(HE.relt, He.relt)
        if irdnt: Et[2:4] = [E+e for E,e in zip(Et[2:4], irdnt)]
        HE.n += He.n
        if HE.root is not None: HE.root.update_root(He)

        return HE  # for feedback in agg+

    def update_root(root, He):

        while root is not None:
            root.Et = np.add(root.Et, He.Et)
            root.relt = np.add(root.relt, He.relt)
            root.n += He.n
            root = root.root

    def comp_(_He, He, DH, rn=1, fagg=0, flat=1, frev=0):  # unpack tuples (formally lists) down to numericals and compare them

        n = 0
        if isinstance(_He.H[0], CH):  # _lay and lay is He_, they are aligned
            Et = [0,0,0,0]  # Vm,Vd, Rm,Rd
            relt = [0,0]  # Dm,Dd
            dH = []
            for _lay,lay in zip(_He.H,He.H):  # md_| ext| derH| subH| aggH, eval nesting, unpack,comp ds in shared lower layers:
                if _lay and lay:  # ext is empty in single-node Gs
                    dlay = _lay.comp_(lay, CH(), rn, fagg=fagg, flat=1, frev=frev)  # dlay is dderH, frev in agg+ only
                    Et = np.add(Et,dlay.Et)
                    relt = np.add(relt,dlay.relt)
                    dH += [dlay]; n += dlay.n
                else:
                    dH += [CH()]  # empty?
        else:  # H is md_, numerical comp:
            vm,vd,rm,rd, decm,decd = 0,0,0,0,0,0
            dH = []
            for i, (_d,d) in enumerate(zip(_He.H[1::2], He.H[1::2])):  # compare ds in md_ or ext
                d *= rn  # normalize by comparand accum span
                diff = _d - d
                if frev: diff = -diff  # from link with reversed dir
                match = min(abs(_d),abs(d))
                if (_d<0) != (d<0): match = -match  # if only one comparand is negative
                if fagg:
                    maxm = max(abs(_d), abs(d))
                    decm += abs(match) / maxm if maxm else 1  # match / max possible match
                    maxd = abs(_d) + abs(d)
                    decd += abs(diff) / maxd if maxd else 1  # diff / max possible diff
                vm += match - aves[i]  # fixed param set?
                vd += diff
                rm += vd > vm; rd += vm >= vd
                dH += [match,diff]  # flat
            Et = [vm,vd,rm,rd]; relt= [decm,decd]
            n = len(_He.H)/12  # unit n = 6 params, = 12 in md_

        return DH.append_(CH(Et=Et, relt=relt, H=dH, n=n), flat=flat)  # currently flat=1
