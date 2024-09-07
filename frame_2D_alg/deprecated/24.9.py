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
