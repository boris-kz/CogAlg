def form_PP_t(root, P_):  # form PPs of dP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    mLink_,_mP__,dLink_,_dP__ = [],[],[],[]  # per PP, !PP.link_?
    for P in P_:
        mlink_,_mP_,dlink_,_dP_ = [],[],[],[]  # per P
        mLink_+=[mlink_]; _mP__+=[_mP_]
        dLink_+=[dlink_]; _dP__+=[_dP_]
        link_ = P.rim if hasattr(P,"rim") else [link for rim in P.rim_ for link in rim]
        # get upper links from all rngs of CP.rim_ | CdP.rim
        for link in link_:
            m,d,mr,dr = link.mdLay.H[-1].Et if isinstance(link.mdLay.H[0],CH) else link.mdLay.Et  # H is md_; last der+ layer vals
            _P = link.nodet[0]
            if m >= ave * mr:
                mlink_+= [link]; _mP_+= [_P]
            if d > ave * dr:  # ?link in both forks?
                dlink_+= [link]; _dP_+= [_P]
        # aligned
    for fd, (Link_,_P__) in zip((0,1),((mLink_,_mP__),(dLink_,_dP__))):
        CP_ = []  # all clustered Ps
        for P in P_:
            if P in CP_: continue  # already packed in some sub-PP
            P_index = P_.index(P)
            cP_, clink_ = [P], [*Link_[P_index]]  # cluster per P
            perimeter = deque(_P__[P_index])  # recycle with breadth-first search, up and down:
            while perimeter:
                _P = perimeter.popleft()
                if _P in cP_ or _P in CP_ or _P not in P_: continue  # clustering is exclusive
                cP_ += [_P]
                clink_ += Link_[P_.index(_P)]
                perimeter += _P__[P_.index(_P)]  # extend P perimeter with linked __Ps
            PP = sum2PP(root, cP_, clink_, fd)
            PP_t[fd] += [PP]
            CP_ += cP_

    root.node_ = PP_t  # nested in der+, add_alt_PPs_?

def add_lays(root, Q, fd):  # add link.derH if fd, else new eLays of higher-composition nodes:
    # init:
    root.derH.append_(Q[0].derH if fd else Q[0].derH.H[1:])
    # accum:
    for n in Q[1:]: root.derH.H[-1].add_H_(n.derH if fd else n.derH.H[1:])

def comp_N(Link, iEt, rng, rev=None):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = rev is not None  # compared links have binary relative direction?
    _N, N = Link.nodet; _S,S = _N.S,N.S; _A,A = _N.A,N.A
    if fd:  # CL
        if rev: A = [-d for d in A]  # reverse angle direction if N is left link?
        _L=2; L=2; _lat,lat,_lay,lay = None,None,None,None
    else:   # CG
        _L,L,_lat,lat,_lay,lay = len(_N.node_),len(_N.node_),_N.latuple,N.latuple,_N.mdLay,N.mdLay
    # form dlay:
    derH = comp_pars([_L,_S,_A,_lat,_lay,_N.derH], [L,S,A,lat,lay,N.derH], rn=_N.n/N.n)
    Et = derH.Et
    iEt[:] = np.add(iEt,Et)  # init eval rng+ and form_graph_t by total m|d?
    for i in 0,1:
        Val, Rdn = Et[i::2]
        if Val > G_aves[i] * Rdn * (rng+1): Link.ft[i] = 1  # fork inclusion tuple
        _N.Et[i] += Val; N.Et[i] += Val  # not selective
        _N.Et[2+i] += Rdn; N.Et[2+i] += Rdn  # per fork link in both Gs
        # if select fork links: iEt[i::2] = [V+v for V,v in zip(iEt[i::2], dH.Et[i::2])]
    if any(Link.ft):
        Link.derH = derH; derH.root = Link; Link.Et = Et; Link.n = min(_N.n,N.n)  # comp shared layers
        Link.nodet = [_N,N]; Link.yx = np.add(_N.yx,N.yx) /2
        # S,A set before comp_N
        for rev, node in zip((0,1),(_N,N)):  # ?reversed Link direction
            if fd:
                if len(node.rimt_)==rng: node.rimt_[-1][1-rev] += [[Link,rev]]  # add in last rng layer, opposite to _N,N dir
                else:                    node.rimt_ = [[[[Link,rev]],[]]] if dir else [[[],[[Link,rev]]]]  # add rng layer
            else:
                if len(node.rim_)==rng: node.rim_[-1] += [[Link, rev]]
                else:                   node.rim_ += [[[Link, rev]]]
            # elay += derH in rng_kern_
        return True

def rng_kern_(N_, rng):  # comp Gs summed in kernels, ~ graph CNN without backprop, not for CLs

    _G_ = []; Et = [0,0,0,0]
    # comp_N:
    for _G,G in combinations(N_,r=2):
        if _G in [g for visited_ in G.visited__ for g in visited_]:  # compared in any rng++
            continue
        dy,dx = np.subtract(_G.yx,G.yx)
        dist = np.hypot(dy,dx)
        aRad = (G.aRad+_G.aRad) / 2  # ave G radius
        # eval relative distance between G centers:
        if dist / max(aRad,1) <= (max_dist * rng):
            for _g,g in (_G,G),(G,_G):
                if len(g.visited__) == rng:
                    g.visited__[-1] += [_g]
                else: g.visited__ += [[_g]]  # init layer
            Link = CL(nodet=[_G,G], S=2,A=[dy,dx], box=extend_box(G.box,_G.box))
            comp_N(Link, Et, rng)
            if Link.Et[0] > ave * Link.Et[2] * (rng+1):
                for g in _G,G:
                    if g not in _G_: _G_ += [g]
    G_ = []  # init conv kernels:
    for G in (_G_):
        krim = []
        for link,rev in G.rim_[-1]:  # form new krim from current-rng links
            if link.Et[0] > link.Et[2] * ave * (rng+1):  # mlink
                _G = link.nodet[0] if link.nodet[1] is G else link.nodet[1]
                krim += [_G]
                G.elay.add_H(link.derH)
                G._kLay = sum_kLay(G,_G); _G._kLay = sum_kLay(_G,G)  # next krim comparands
        if krim:
            if rng>1: G.kHH[-1] += [krim]  # kH = lays(nodes
            else:     G.kHH += [[krim]]
            G_ += [G]
    Gd_ = copy(G_)  # Gs with 1st layer kH, dLay, _kLay
    _G_ = G_; n=0  # n higher krims
    # convolution: Def,Sum,Comp kernel rim, in separate loops for bilateral G,_G assign:
    while True:
        G_ = []
        for G in _G_:  # += krim
            G.kHH[-1] += [[]]; G.visited__ += [[]]
        for G in _G_:
            #  append G.kHH[-1][-1]:
            for _G in G.kHH[-1][-2]:
                for link, rev in _G.rim_[-1]:
                    __G = link.nodet[0] if link.nodet[1] is G else link.nodet[1]
                    if __G in _G_:
                        if __G not in G.kHH[-1][-1] + [g for visited_ in G.visited__ for g in visited_]:
                            # bilateral add layer of unique mediated nodes
                            G.kHH[-1][-1] += [__G]; __G.kHH[-1][-1] += [G]
                            for g,_g in zip((G,__G),(__G,G)):
                                g.visited__[-1] += [_g]
                                if g not in G_:  # in G_ only if in visited__[-1]
                                    G_ += [g]
        for G in G_: G.visited__ += [[]]
        for G in G_: # sum kLay:
            for _G in G.kHH[-1][-1]:  # add last krim
                if _G in G.visited__[-1] or _G not in _G_:
                    continue  # Gs krim appended when _G was G
                G.visited__[-1] += [_G]; _G.visited__[-1] += [G]
                # sum alt G lower kLay:
                G.kLay = sum_kLay(G,_G); _G.kLay = sum_kLay(_G,G)
        for G in G_: G.visited__[-1] = []
        for G in G_:
            for _G in G.kHH[-1][0]:  # convo in direct kernel only
                if _G in G.visited__[-1] or _G not in G_: continue
                G.visited__[-1] += [_G]; _G.visited__[-1] += [G]
                # comp G kLay -> rng derLay:
                rlay = comp_pars(_G._kLay, G._kLay, _G.n/G.n)  # or/and _G.kHH[-1][-1] + G.kHH[-1][-1]?
                if rlay.Et[0] > ave * rlay.Et[2] * (rng+n):  # layers add cost
                    _G.elay.add_H(rlay); G.elay.add_H(rlay)  # bilateral
        _G_ = G_; G_ = []
        for G in _G_:  # eval dLay
            G.visited__.pop()  # loop-specific layer
            if G.elay.Et[0] > ave * G.elay.Et[2] * (rng+n+1):
                G_ += [G]
        if G_:
            for G in G_: G._kLay = G.kLay  # comp in next krng
            _G_ = G_; n += 1
        else:
            for G in Gd_:
                G.visited__.pop()  # kH - specific layer
                delattr(G,'_kLay')
                if hasattr(G,'kLay'): delattr(G,'kLay')
            break

    return Gd_, Et  # all Gs with dLay added in 1st krim

def sum_kLay(G, g):  # sum next-rng kLay from krim of current _kLays, init with G attrs

    KLay = (G.kLay if hasattr(G,"kLay")
                   else (G._kLay if hasattr(G,"_kLay")  # init conv kernels, also below:
                              else (len(G.node_),G.S,G.A,deepcopy(G.latuple),CH().copy(G.mdLay),CH().copy(G.derH) if G.derH else CH())))  # init DerH if empty
    kLay = (G._kLay if hasattr(G,"_kLay")
                    else (len(g.node_),g.S,g.A,deepcopy(g.latuple),CH().copy(g.mdLay),CH().copy(g.derH) if g.derH else None))  # in init conv kernels
    L,S,A,Lat,MdLay,DerH = KLay
    l,s,a,lat,mdLay,derH = kLay
    return [
            L+l, S+s, [A[0]+a[0],A[1]+a[1]], # L,S,A
            add_lat(Lat,lat),                # latuple
            MdLay.add_md_(mdLay),            # mdLay
            DerH.add_H(derH) if derH else DerH
    ]

def rng_node_(_N_):  # each rng+ forms rim_ layer per N, appends N__,L__,Et:

    N__ = []; L__ = []; HEt = [0,0,0,0]
    rng = 1
    while True:
        N_,Et = rng_kern_(_N_,rng)
        # adds link_ to N.rim_ and _N_H to N.kHH, but elay and Et are summed across rng++
        L_ = [Lt[0] for N in N_ for Lt in N.rim_[-1]]
        if Et[0] > ave * Et[2] * rng:
            L__+= L_; HEt = np.add(HEt, Et)
            N__ += N_; _N_ = N_; rng += 1
        else:
            break
    return list(set(N__)), L__, HEt, rng


def comp_pars(_pars, pars, rn):  # compare Ns, kLays or partial graphs in merging

    _L,_S,_A,_latuple,_mdLay,_derH = _pars
    L, S, A, latuple, mdLay, derH = pars

    mdext = comp_ext(_L,L,_S,S/rn,_A,A)
    if mdLay:  # CG
        mdLay = _mdLay.comp_md_(mdLay, rn, fagg=1)
        mdlat = comp_latuple(_latuple, latuple, rn, fagg=1)
        n = mdlat.n + mdLay.n + mdext.n
        md_t = [mdlat, mdLay, mdext]
        Et = np.array(mdlat.Et) + mdLay.Et + mdext.Et
        Rt = np.array(mdlat.Rt) + mdLay.Rt + mdext.Rt
    else:  # += CL nodes
        n = mdext.n; md_t = [mdext]; Et = mdext.Et; Rt = mdext.Rt
    # init H = [lay0]:
    dlay = CH( H=[CH(n=n, md_t=md_t, Et=Et, Rt=Rt)],  # or n = (_n+n) / 2?
               n=n, md_t=[CH().copy(md_) for md_ in md_t], Et=copy(Et),Rt=copy(Rt))
    if _derH and derH:
        dderH = _derH.comp_H(derH, rn, fagg=1)  # new link derH = local dH
        dlay.append_(dderH, flat=1)

    return dlay

def segment0(root, Q, fd, rng):  # cluster iN_(G_|L_) by weight of shared links, initially single linkage

    N_, max_ = [],[]
    # init Gt per G|L node:
    for N in Q:
        Lrim = [Lt[0] for Lt in (N.rimt_[-1][0] + N.rimt_[-1][1] if fd else N.rim_[-1])]  # external links
        Lrim = [L for L in Lrim if L.Et[fd] > ave * (L.Et[2+fd]) * rng]  # to merge if sL_ match
        Nrim = [_N for L in Lrim for _N in L.nodet if _N is not N]  # external nodes
        Gt = [[N],[], Lrim, Nrim, [0,0,0,0]]
        N.root = Gt
        N_ += [Gt]
        # select exemplar maxes to segment clustering:
        emax_ = [eN for eN in Nrim if eN.Et[fd] >= N.Et[fd] or eN in max_]  # _N if _N == N
        if not emax_: max_ += [Gt]  # N.root, if no higher-val neighbors
        # extended rrim max: V * k * max_rng?
    for Gt in N_: Gt[3] = [_N.root for _N in Gt[3]]  # replace eNs with Gts
    for Gt in max_ if max_ else N_:
        node_,link_,Lrim,Nrim, Et = Gt
        while True:  # while Nrim
            _Nrim_,_Lrim_ = [],[]  # recursively merge Gts with >ave shared +ve external links
            for _Gt,_L in zip(Nrim,Lrim):
                if _Gt not in N_: continue  # was merged
                sL_ = set(Lrim).intersection(set(_Gt[2])).union([_L])  # shared external links + potential _L
                Et = np.sum([L.Et for L in sL_], axis=0)
                if Et[0] > ave * (Et[2]) * rng:  # mval of shared links, not relative overlap: sL.V / Lrim.V?
                    # or any +ve links, same as per N?
                    link_ += [_L]
                    merge(Gt,_Gt, _Nrim_,_Lrim_)
                    N_.remove(_Gt)
            if _Nrim_:
                Nrim[:],Lrim[:] = _Nrim_,_Lrim_  # for clustering, else break, contour = term rims?
            else: break
    return [sum2graph(root, Gt, fd, rng) for Gt in N_]

def comp_ext(_L,L,_S,S,_A,A):  # compare non-derivatives:

    dL = _L - L; mL = min(_L,L) - ave_L  # direct match
    dS = _S/_L - S/L; mS = min(_S,S) - ave_L  # sparsity is accumulated over L
    mA, dA = comp_angle(_A,A)  # angle is not normalized
    M = mL + mS + mA
    D = abs(dL) + abs(dS) + abs(dA)  # signed dA?
    mrdn = M > D; drdn = D<= M
    mdec = mL/ max(L,_L) + mS/ max(S,_S) if S or _S else 1 + mA  # Amax = 1
    ddec = max_dist + mL/ (L+_L) + dS/ (S+_S) if S or _S else 1 + dA

    return CH(H=[mL,dL, mS,dS, mA,dA], Et=[M,D,mrdn,drdn], Rt=[mdec,ddec], n=0.5)

def segment1(root, Q, fd, rng):  # cluster Q: G_|L_, by value density of +ve links per node
    '''
    convert to bottom-up:
    '''
    for N in Q: N.merged = 0  # reset if sub-clustering
    N_ = []
    for N in Q:
        if not N.lrim:
            N_ += [N]; continue
        _nrim_ = N.nrim; _lrim_ = N.lrim
        node_ = {N}; link_ = set(); Et = [0,0,0,0]
        while _nrim_:
            nrim_,lrim_ = set(),set()  # eval,merge _nrim_, replace with extended nrim_
            for _N,_L in zip(_nrim_,_lrim_):
                if _N.merged: continue
                int_N = _L.nodet[0] if _L.nodet[1] is _N else _L.nodet[1]
                # cluster by sum N_rim_Ms * L_rM, mildly negating if include neg links:
                if (int_N.Et[0]+_N.Et[0]) * (_L.Et[0]/ave) > ave:
                    node_.add(_N); link_.add(_L); Et = np.add(Et, _L.Et)
                    nrim_.update(set(_N.nrim) - node_)
                    lrim_.update(set(_N.lrim) - link_)
                    _N.merged = 1
            _nrim_, _lrim_ = nrim_, lrim_
        G = sum2graph(root, [list(node_), list(link_), Et], fd, rng)
        # low-rng sub-clustering in G:
        sub_rng = rng-1; H = G.extH.H
        while len(H) > sub_rng and H[sub_rng].Et[0] > ave * H[sub_rng].Et[2] * (sub_rng+1):
            # segment sub_node_:
            H[sub_rng].node_[:] = segment(G, H[sub_rng].node_, fd, rng)
        N_ += [G]
    return N_ # Gs and isolated Ns

def comp_latuple(_latuple, latuple, rn, fagg=0):  # 0der params

    _I, _G, _M, _Ma, _L, (_Dy, _Dx) = _latuple
    I, G, M, Ma, L, (Dy, Dx) = latuple

    dI = _I - I*rn;  mI = ave_dI - dI
    dG = _G - G*rn;  mG = min(_G, G*rn) - aves[1]
    dL = _L - L*rn;  mL = min(_L, L*rn) - aves[2]
    dM = _M - M*rn;  mM = get_match(_M, M*rn) - aves[3]  # M, Ma may be negative
    dMa= _Ma- Ma*rn; mMa = get_match(_Ma, Ma*rn) - aves[4]
    mAngle, dAngle = comp_angle((_Dy,_Dx), (Dy,Dx))

    ret = [mL,dL,mI,dI,mG,dG,mM,dM,mMa,dMa,mAngle-aves[5],dAngle]
    if fagg:  # add norm m,d, ret=[ret,Ret]:
        # get max possible m and d per compared param to compute relt:
        Mx_ = [max(_L,L),abs(_L)+abs(L), max(_I,I),abs(_I)+abs(I), max(_G,G),abs(_G)+abs(G), max(_M,M),abs(_M)+abs(M), max(_Ma,Ma),abs(_Ma)+abs(Ma), 1,.5]
        mval, dval = sum(ret[::2]),sum(ret[1::2])
        mrdn, drdn = dval>mval, mval>dval
        mdec, ddec = 0, 0
        for fd, (ptuple,Ptuple) in enumerate(zip((ret[::2],ret[1::2]),(Mx_[::2],Mx_[1::2]))):
            for i, (par, maxv, ave) in enumerate(zip(ptuple, Ptuple, aves)):
                # compute link decay coef: par/ max(self/same):
                if fd: ddec += abs(par)/ abs(maxv) if maxv else 1
                else:  mdec += (par+ave)/ (maxv+ave) if maxv else 1
        ret = CH(H=ret, Et=[mval,dval,mrdn,drdn], Rt=[mdec,ddec], n=1)  # if fagg only
    return ret

def comp_md_(_He, He, rn=1, fagg=0, frev=0):

        vm, vd, rm, rd, decm, decd = 0, 0, 0, 0, 0, 0
        derLay = []
        for i, (_d, d) in enumerate(zip(_He.H[1::2], He.H[1::2])):  # compare ds in md_ or ext
            d *= rn  # normalize by compared accum span
            diff = _d - d
            if frev: diff = -diff  # from link with reversed dir
            match = min(abs(_d), abs(d))
            if (_d < 0) != (d < 0): match = -match  # negate if only one compared is negative
            if fagg:
                maxm = max(abs(_d), abs(d))
                decm += abs(match) / maxm if maxm else 1  # match / max possible match
                maxd = abs(_d) + abs(d)
                decd += abs(diff) / maxd if maxd else 1  # diff / max possible diff
            vm += match - aves[i]  # fixed param set?
            vd += diff
            rm += vd > vm; rd += vm >= vd
            derLay += [match, diff]  # flat

        return CH(H=derLay, Et=[vm,vd,rm,rd], Rt=[decm,decd], n=1)


def rng_node_(_N_):  # rng+ forms layer of rim_ and extH per N, appends N__,L__,Et, ~ graph CNN without backprop

    N__ = []; L__ = []; ET = [0,0,0,0]
    rng = 1
    while True:
        N_ = []; Et = [0,0,0,0]
        # full search, no mediation
        for _G,G in combinations(_N_,r=2):  # or set rim_ for all Gs in one loop?
            fcont = 0
            for g in G.visited_:
                if g is _G:
                    fcont = 1; break  # compared in any rng
                elif G in g.nrim_[-1] and _G in g.nrim_[-1]:  # shorter match-mediated match
                    fcont = 1; break  # or longer direct match priority?
            if fcont: continue
            dy,dx = np.subtract(_G.yx,G.yx); dist = np.hypot(dy,dx)
            aRad = (G.aRad +_G.aRad) / 2  # ave G radius
            # eval relative distance between G centers:
            if dist / max(aRad,1) <= (max_dist * rng):
                for _g, g in (_G,G),(G,_G): g.visited_ += [_g]
                Link = CL(nodet=[_G,G], S=2, A=[dy,dx], box=extend_box(G.box,_G.box))
                comp_N(Link, Et, rng)
                if Link.Et[0] > ave * Link.Et[2] * (rng+1):
                    for g in _G,G:
                        if g not in N_: N_ += [g]
        if Et[0] > ave * Et[2] * rng:
            ET = np.add(ET, Et)
            L__ += [list(set([Lt[0] for N in N_ for Lt in N.rim_[-1]]))]
            N__ += [N_]  # nest to sub-cluster?
            _N_ = N_
            rng += 1
        else:
            break
    return N__,L__,ET,rng

def merge(Gt, gt):

    N_, L_, Et, Lrim, Nrim = Gt
    n_, l_, et, lrim, nrim = gt
    for N in n_:
        N.root_ += [Gt]
        N.merged = 1
    Et[:] = np.add(Et, et)
    N_.update(n_); Nrim = set(Nrim) - set(nrim)  # we need .update to concatenate set
    L_.update(l_); Lrim = set(Lrim) - set(lrim)

    Nrim.update(set(nrim) - set(N_))
    Lrim.update(set(lrim) - set(L_))

def par_segment(root, Q, fd, rng):  # parallelizable by merging Gts initialized with each N
    # mostly old
    N_, max_ = [],[]
    # init Gt per G|L node:
    for N in Q:
        Lrim = [Lt[0] for Lt in (N.rimt_[-1][0] + N.rimt_[-1][1] if fd else N.rim_[-1])]  # external links
        Lrim = [L for L in Lrim if L.Et[fd] > ave * (L.Et[2+fd]) * rng]  # +ve to merge Gts
        Nrim = [_N for L in Lrim for _N in L.nodet if _N is not N]  # external nodes
        Gt = [[N],[],[0,0,0,0], Lrim,Nrim]
        N.root = Gt
        N_ += [Gt]
        # select exemplar maxes to segment clustering:
        emax_ = [eN for eN in Nrim if eN.Et[fd] >= N.Et[fd] or eN in max_]  # _N if _N == N
        if not emax_: max_ += [Gt]  # N.root, if no higher-val neighbors
        # extended rrim max: V * k * max_rng?
    for Gt in N_: Gt[3] = [_N.root for _N in Gt[3]]  # replace eNs with Gts
    for Gt in max_ if max_ else N_:
        node_,link_,Et, Lrim,Nrim = Gt
        while True:  # while Nrim, not revised
            _Nrim_,_Lrim_ = [],[]  # recursive merge connected Gts
            for _Gt,_L in zip(Nrim,Lrim):  # always single N unless parallelized
                if _Gt not in N_: continue  # was merged
                for L in _Gt[3]:
                    if L in Lrim and (len(_Lrim_) > ave_L or len(Lrim) > ave_L):  # density-based
                        merge(Gt,_Gt, _Nrim_,_Lrim_); N_.remove(_Gt)
                        break  # merge if any +ve shared external links
            if _Nrim_:
                Nrim[:],Lrim[:] = _Nrim_,_Lrim_  # for clustering, else break, contour = term rims?
            else: break
    return [sum2graph(root, Gt[:3], fd, rng) for Gt in N_]


def segment(root, iN__, fd, irng):  # cluster Q: G__|L__, by value density of +ve links per node

    # rng=1: cluster connected Gs in Gts, then rng+: merge connected Gts into higher Gts
    for G in iN__[0]:
        G.merged = 0
    N_,_re_N_ = [],[]
    for G in iN__[0]:  # not need for iN__[1:]?
        if not G.nrim_:
            N_ += [G]; continue
        node_,link_,Et, _nrim,_lrim = {G}, set(), np.array([.0,.0,.0,.0]), {G.nrim_[0]}, {G.lrim_[0]}
        while _lrim:
            nrim, lrim = set(), set()
            for _G,_L in zip(_nrim,_lrim):
                if _G.merged: continue
                for g in node_:  # compare external _G to all internal nodes, include if any of them match
                    L_ = g.lrim_[0].intersect(_G.lrim_[0])
                    if L_:
                        L = L_[0]  # <= 1 link between two nodes
                        if (g.Et[0]+_G.Et[0]) * (L.Et[0]/ave) > ave:  # cluster by sum G_rim_Ms * L_rM, neg if neg link
                            node_.add(_G); link_.add(_L); Et += _L.Et
                            nrim.update(set(_G.nrim_[0])-node_)
                            lrim.update(set(_G.lrim_[0])-link_)
                            _G.merged = 1
                            break
            _nrim, _lrim = nrim,lrim  # replace with extended rims
        Gt = [node_,link_, Et,0]
        for n in node_: n.root_ = [Gt]
        _re_N_ += [Gt]  # selective
        N_ += [Gt]
    N__ = [N_]
    rng = 1
    # rng+: merge Gts connected via G.lrim_[rng] in their node_s into higher Gts
    while True:
        re_N_, N_ = [],[]
        for Gt in _re_N_:
            for G in Gt[0]: G.merged = 0
        for node_,link_,Et, mrg in _re_N_:
            if mrg: continue
            Node_,Link_,ET = node_.copy(),link_.copy(),Et.copy()  # rng GT
            for G in node_:
                if not G.merged and len(G.nrim_) > rng:
                    _nrim = G.nrim_[rng] - Node_
                    _lrim = G.lrim_[rng] - Link_
                    while _lrim:
                        nrim,lrim = set(),set()
                        for _G,_L in zip(_nrim,_lrim):
                            if _G.merged: continue  # or if _G.root_[-1][3]: same?
                            if (G.Et[0]+_G.Et[0]) * (_L.Et[0]/ave) > ave:
                                n_ = _G.root_[-1][0]; Node_.add(n_)
                                for g in n_: g.merged = 0
                                Link_.add(_L); Link_.add(_G.root_[-1][1])
                                ET += _L.Et + _G.root_[-1][2]
                                nrim.update(set(_G.nrim_[rng])-Node_)
                                lrim.update(set(_G.lrim_[rng])-Link_)
                            _G.merged = 1
                            _G.root_[-1][3] = 1  # redundant?
                        _nrim, _lrim = nrim,lrim  # replace with extended rims
                Gt = [Node_,Link_, ET,0]
                for n in Node_: n.root_ += [Gt]
                re_N_ += [Gt]  # if len(G.nrim_) > rng
        if re_N_:
            rng += 1; N__ += [re_N_]; _re_N_ = re_N_
        else:
            break
    for i, N_ in enumerate(N__):  # batch conversion of Gts to CGs
        for ii, N in enumerate(N_):
            if isinstance(N, list):  # not CG
                N_[ii] = sum2graph(root, [list(N[0]), list(N[1]), N[2]], fd, rng=i)
                # [node_,link_,Et], higher-rng Gs are supersets
    iN__[:] = N__  # Gs and isolated Ns

def negate(He):  # negate is no longer useful?
    if isinstance(He.H[0], CH):
        for i,lay in enumerate(He.H):
            He.H[i] = negate(lay)
    else:  # md_
        He.H[1::2] = [-d for d in He.H[1::2]]
    return He


def form_PP_(root, iP_, fd=0):  # form PPs of dP.valt[fd] + connected Ps val

    for P in iP_: P.merged = 0
    PPt_ = []

    for P in iP_:  # for dP in link_ if fd
        if P.merged: continue
        if not P.lrim:
            PPt_ += [P]; continue
        _prim_ = P.prim; _lrim_ = P.lrim
        _P_ = {P}; link_ = set(); Et = np.array([.0,.0,.0,.0])
        while _prim_:
            prim_,lrim_ = set(),set()
            for _P,_L in zip(_prim_,_lrim_):
                if _P.merged: continue  # was merged
                _P_.add(_P); link_.add(_L); Et += _L.mdLay[1]  # _L.mdLay.Et
                prim_.update(set(_P.prim) - _P_)
                lrim_.update(set(_P.lrim) - link_)
                _P.merged = 1
            _prim_, _lrim_ = prim_, lrim_
        PPt = sum2PP(root, list(_P_), list(link_), fd)
        PPt_ += [PPt]

    if fd:  # terminal fork
        root[2] = PPt_  # replace PPm link_ with a mix of CdPs and PPds
    else:
        for PPt in PPt_:  # eval sub-clustering, not recursive
            if isinstance(PPt, list):  # a mix of CPs and PPms
                P_, link_, [_, Et, _] = PPt[1:4]
                if len(link_) > ave_L and Et[fd] >PP_aves[fd] * Et[2+fd]:
                    comp_link_(PPt)
                    form_PP_(PPt, link_, fd=1)
                    # += PPds within PPm link_
        root.node_ = PPt_  # edge.node_

def reset_merged(Gt_,rng):

    for Gt in Gt_:
        for G in Gt[0]:
            G.merged = 0
            if len(G.nrim_) > rng:
                for n in G.nrim_[rng]:
                    if len(n.nrim_) > rng:  # we need to check their nrim too?
                        n.merged = 0
