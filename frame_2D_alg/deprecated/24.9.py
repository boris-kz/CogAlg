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