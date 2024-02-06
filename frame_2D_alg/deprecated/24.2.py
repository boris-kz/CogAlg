def comp_G(link, Et):

    _G, G = link._G, link.G
    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0, 1,1, 0,0
    # keep separate P ptuple and PP derH, empty derH in single-P G, + empty aggH in single-PP G:

    # / P:
    mtuple, dtuple, Mtuple, Dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1, fagg=1)
    mval, dval = sum(mtuple), sum(abs(d) for d in dtuple)  # mval is signed, m=-min in comp x sign
    mrdn = dval>mval; drdn = dval<=mval
    dect = [0,0]
    for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
        for i, (par, max, ave) in enumerate(zip(ptuple, Ptuple, aves)):
            # compute link decay coef: par/ max(self/same)
            if fd: dect[fd] += abs(par)/ abs(max) if max else 1
            else:  dect[fd] += (par+ave)/ (max+ave) if max else 1
    dertv = [[mtuple,dtuple], [mval,dval],[mrdn,drdn],[dect[0]/6,dect[1]/6]]
    Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn; Mdec+=dect[0]/6; Ddec+=dect[1]/6  # ave of 6 params

    # / PP:
    dderH = []
    if _G.derH and _G.derH:  # empty in single-P Gs?
        for _lay, lay in zip(_G.derH,_G.derH):
            mtuple,dtuple, Mtuple,Dtuple = comp_dtuple(_lay[1], lay[1], rn=1, fagg=1)
            mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
            mrdn = dval > mval; drdn = dval < mval
            mdec, ddec = 0, 0
            for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
                for (par, max, ave) in zip(ptuple, Ptuple, aves):  # different ave for comp_dtuple
                    if fd: ddec += abs(par)/ abs(max) if max else 1
                    else:   mdec += (par+ave)/ (max+ave) if max else 1
            mdec /= 6; ddec /= 6
            Mval+=dval; Dval+=mval; Mdec=(Mdec+mdec)/2; Ddec=(Ddec+ddec)/2
            dderH += [[[mtuple,dtuple], [mval,dval],[mrdn,drdn],[mdec,ddec]]]

    # / G:
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec])
    dderH = [[dertv]+dderH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec], der_ext]

    if _G.aggH and G.aggH:
        daggH, valt,rdnt,dect = comp_aggHv(_G.aggH, G.aggH, rn=1) if G.fHH else comp_subHv(_G.aggH, G.aggH, rn=1)
        # aggH is subH before agg+
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0] + dval>mval; Drdn += rdnt[1] + dval<=mval
        Mdec = (Mdec+dect[0])/2; Ddec = (Ddec+dect[1])/2
        # flat, appendleft:
        daggH = [[dderH]+daggH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]]
    else:
        daggH = dderH
    link.daggH += [daggH]

    link.Vt,link.Rt,link.Dt = Valt,Rdnt,Dect = [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]  # reset per comp_G

    for fd, (Val,Rdn,Dec) in enumerate(zip(Valt,Rdnt,Dect)):
        if Val > G_aves[fd] * Rdn:
            # to eval fork grapht in form_graph_t:
            Et[0][fd] += Val; Et[1][fd] += Rdn; Et[2][fd] += Dec
            G.Vt[fd] += Val;  G.Rt[fd] += Rdn;  G.Dt[fd] += Dec
            if not fd:
                for G in link.G, link._G:
                    rimH = G.rimH
                    if rimH and isinstance(rimH[0],list):  # rim is converted to rimH in 1st sub+
                        if len(rimH) == len(G.RimH): rimH += [[]]  # no new rim layer yet
                        rimH[-1] += [link]  # rimH
                    else:
                        rimH += [link]  # rim


def comp_aggHv(_aggH, aggH, rn):  # no separate ext

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    SubH = []

    for _lev, lev in zip(_aggH, aggH):  # compare common subHs, if lower-der match?
        if _lev and lev:

            if len(lev) == 5:  # derHv with additional ext
                dderH, valt,rdnt,dect, dextt = comp_derHv(_lev,lev, rn)
                SubH += [[dderH, valt,rdnt,dect,dextt]]
            else:
                dsubH, valt,rdnt,dect = comp_subHv(_lev[0],lev[0], rn)
                SubH += dsubH  # concat
            Mdec += dect[0]; Ddec += dect[1]
            mval,dval = valt; Mval += mval; Dval += dval
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + mval <= dval
    if SubH:
        S = min(len(_aggH),len(aggH)); Mdec/= S; Ddec /= S  # normalize

    return SubH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]


def comp_subHv(_subH, subH, rn):

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    dsubH =[]

    for _lay, lay in zip(_subH, subH):  # compare common lower layer|sublayer derHs, if prior match?

        dderH, valt,rdnt,dect, dextt = comp_derHv(_lay,lay, rn)  # derHv: [derH, valt, rdnt, dect, extt]:
        dsubH += [[dderH, valt,rdnt,dect,dextt]]  # flat
        Mdec += dect[0]; Ddec += dect[1]
        mval,dval = valt; Mval += mval; Dval += dval
        Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + dval <= mval
    if dsubH:
        S = min(len(_subH),len(subH)); Mdec/= S; Ddec /= S  # normalize

    return dsubH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]  # new layer,= 1/2 combined derH


def sum_derHv(T,t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    if t:
        if T:
            DerH, Valt, Rdnt, Dect, Extt_ = T
            derH, valt, rdnt, dect, extt_ = t
            for Extt, extt in zip(Extt_,extt_):
                sum_ext(Extt, extt)
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i] + dect[i])/2
            DerH[:] = [
                [[sum_dertuple(Dertuple,dertuple, fneg*i) for i,(Dertuple,dertuple) in enumerate(zip(Tuplet,tuplet))],
                  [V+v for V,v in zip(Valt,valt)], [R+r+base_rdn for R,r in zip(Rdnt,rdnt)], [(D+d)/2 for D,d in zip(Dect,dect)],
                ]
                for [Tuplet,Valt,Rdnt,Dect], [tuplet,valt,rdnt,dect]  # ptuple_tv
                in zip_longest(DerH, derH, fillvalue=[([0,0,0,0,0,0],[0,0,0,0,0,0]), (0,0),(0,0),(0,0)])
            ]
        else:
            T[:] = deepcopy(t)

def sum_ext(Extt, extt):

    if isinstance(Extt[0], list):
        for Ext,ext in zip(Extt,extt):  # ext: m,d tuple
            for i,(Par,par) in enumerate(zip(Ext,ext)):
                Ext[i] = Par+par
    else:  # single ext
        for i in range(3): Extt[i]+=extt[i]  # sum L,S,A


def comp_ext(_ext, ext, Valt, Rdnt, Dect):  # comp ds:

    (_L,_S,_A),(L,S,A) = _ext,ext
    dL = _L-L

    if isinstance(A,list):
        if any(A) and any(_A):
            mA,dA = comp_angle(_A,A); adA=dA
        else:
            mA,dA,adA = 0,0,0
        max_mA = max_dA = .5  # = ave_dangle
        dS = _S/_L - S/L  # S is summed over L, dS is not summed over dL
    else:
        mA = get_match(_A,A)- ave_dangle; dA = _A-A; adA = abs(dA); _aA=abs(_A); aA=abs(A)
        max_dA = _aA + aA; max_mA = max(_aA, aA)
        dS = _S - S
    mL = get_match(_L,L) - ave_mL
    mS = get_match(_S,S) - ave_mL

    m = mL+mS+mA; d = abs(dL)+ abs(dS)+ adA
    Valt[0] += m; Valt[1] += d
    Rdnt[0] += d>m; Rdnt[1] += d<=m
    _aL = abs(_L); aL = abs(L)
    _aS = abs(_S); aS = abs(S)

    # ave dec = ave (ave dec, ave L,S,A dec):
    Dect[0] = ((mL / max(aL,_aL) if aL or _aL else 1 +
                mS / max(aS,_aS) if aS or _aS else 1 +
                mA / max_mA if max_mA else 1) /3
                + Dect[0]) / 2
    Dect[1] = ((dL / (_aL+aL) if aL+_aL else 1 +
                dS / (_aS+aS) if aS+_aS else 1 +
                dA / max_dA if max_mA else 1) /3
                + Dect[1]) / 2

    return [[mL,mS,mA], [dL,dS,dA]]

'''
Each call to comp_rng | comp_der forms dderH: new layer of derH. Both forks are merged in feedback to contain complexity
(deeper layers are appended by feedback, if nested we need fback_tree: last_layer_nforks = 2^n_higher_layers)
'''
def sub_recursion(root, PP, fd):  # called in form_PP_, evaluate PP for rng+ and der+, add layers to select sub_PPs
    rng = PP.rng+(1-fd)

    if fd: link_ = comp_der(PP.link_)  # der+: reform old links
    else:  link_ = comp_rng(PP.link_, rng)  # rng+: form new links, should be recursive?

    rdn = (PP.valt[fd] - PP_aves[fd] * PP.rdnt[fd]) > (PP.valt[1-fd] - PP_aves[1-fd] * PP.rdnt[1-fd])
    PP.rdnt += (0, rdn) if fd else (rdn, 0)     # a little cumbersome, will be revised later
    for P in PP.P_: P.root = [None,None]  # fill with sub_PPm_,sub_PPd_ between nodes and PP:

    form_PP_t(PP, link_, base_rdn=PP.rdnt[fd])
    root.fback_t[fd] += [[PP.derH, PP.valt, PP.rdnt]]  # merge in root.fback_t fork, else need fback_tree


def comp_P_(edge: Cgraph, adj_Pt_: List[Tuple[CP, CP]]):  # cross-comp P_ in edge: high-gradient blob, sliced in Ps in the direction of G

    for _P, P in adj_Pt_:  # scan, comp contiguously uplinked Ps, rn: relative weight of comparand
        # initial comp is rng+
        distance = np.hypot(_P.yx[1]-P.yx[1],_P.yx[0]-P.yx[0])
        comp_P(edge.link_, _P, P, rn=len(_P.dert_)/len(P.dert_), derP=distance, fd=0)

    form_PP_t(edge, edge.link_, base_rdn=2)

def comp_P(link_, _P, P, rn, derP=None, S=None, V=0):  #  derP if der+, S if rng+

    if derP is not None:  # der+: extend in-link derH, in sub+ only
        dderH, valt, rdnt = comp_derH(_P.derH, P.derH, rn=rn)  # += fork rdn
        derH = derP.derH | dderH; S = derP.S  # dderH valt,rdnt for new link
        aveP = P_aves[1]

    elif S is not None:  # rng+: add derH
        mtuple, dtuple = comp_ptuple(_P.ptuple, P.ptuple, rn, fagg=0)
        valt = Cmd(sum(mtuple), sum(abs(d) for d in dtuple))
        rdnt = Cmd(1+(valt.d>valt.m), 1+(1-(valt.d>valt.m)))   # or rdn = Dval/Mval?
        derH = CderH([Cmd(mtuple, dtuple)])
        aveP = P_aves[0]
        V += valt[0] + sum(mtuple)
    else:
        raise ValueError("either derP (der+) or S (rng+) should be specified")

    A = Cangle(*(_P.yx - P.yx))
    derP = CderP(derH=derH, valt=valt, rdnt=rdnt, P=P,_P=_P, S=S, A=A)

    if valt.m > aveP*rdnt.m or valt.d > aveP*rdnt.d:
        link_ += [derP]

    return V


def form_PP_t(root, root_link_, base_rdn):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    for fd in 0,1:
        P_Ps = defaultdict(list)
        derP_ = []
        for derP in root_link_:
            if derP.valt[fd] > P_aves[fd] * derP.rdnt[fd]:
                P_Ps[derP.P] += [derP._P]  # key:P, vals:linked _Ps, up and down
                P_Ps[derP._P] += [derP.P]
                derP_ += [derP]  # filtered derP
        inP_ = []  # clustered Ps and their val,rdn s for all Ps
        for P in root.P_:
            if P in inP_: continue  # already packed in some PP
            cP_ = [P]  # clustered Ps and their val,rdn s
            perimeter = deque(P_Ps[P])  # recycle with breadth-first search, up and down:
            while perimeter:
                _P = perimeter.popleft()
                if _P in cP_: continue
                cP_ += [_P]
                perimeter += P_Ps[_P]  # append linked __Ps to extended perimeter of P
            PP = sum2PP(root, cP_, derP_, base_rdn, fd)
            PP_t[fd] += [PP]  # no if Val > PP_aves[fd] * Rdn:
            inP_ += cP_  # update clustered Ps

    for fd, PP_ in enumerate(PP_t):  # after form_PP_t -> P.roott
        for PP in PP_:
            if PP.valt[1] * len(PP.link_) > PP_aves[1] * PP.rdnt[1]:  # val*len*rng: sum ave matches - fixed PP cost
                der_recursion(root, PP, fd)  # eval rng+/PPm or der+/PPd
        if root.fback_t and root.fback_t[fd]:
            feedback(root, fd)  # after sub+ in all nodes, no single node feedback up multiple layers

    root.node_ = PP_t  # nested in sub+, add_alt_PPs_?

def sum_aggH(AggH, aggH, base_rdn):

    if aggH:
        if AggH:
            for Layer, layer in zip_longest(AggH,aggH, fillvalue=[]):
                if layer[-1] == 1:  # derHv with depth == 1
                    sum_derHv(Layer, layer, base_rdn)
                else:  # subHv
                    sum_subHv(Layer, layer, base_rdn)
        else:
            AggH[:] = deepcopy(aggH)


def sum_aggHv(T, t, base_rdn):

    if t:
        if T:
            AggH,Valt,Rdnt,Dect,_ = T; aggH,valt,rdnt,dect,_ = t
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i]+dect[i])/2
            if AggH:
                for Layer, layer in zip_longest(AggH,aggH, fillvalue=[]):
                    if layer[-1] == 1:  # derHv with depth == 1
                        sum_derHv(Layer, layer, base_rdn)
                    else:  # subHv
                        sum_subHv(Layer, layer, base_rdn)
            else:
                AggH[:] = deepcopy(aggH)
            sum_ext(Ext,ext)
        else:
           T[:] = deepcopy(t)


def sum_subHv(T, t, base_rdn, fneg=0):

    if t:
        if T:
            SubH,Valt,Rdnt,Dect,Ext,_ = T; subH,valt,rdnt,dect,ext,_ = t
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i]+dect[i])/2
            if SubH:
                for Layer, layer in zip_longest(SubH,subH, fillvalue=[]):
                    sum_derHv(Layer, layer, base_rdn, fneg)  # _lay[0][0] is mL
                    sum_ext(Layer[-2], layer[-2])
            else:
                SubH[:] = deepcopy(subH)
            sum_ext(Ext,ext)
        else:
            T[:] = deepcopy(t)


def sum_derHv(T,t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    if t:
        if T:
            DerH, Valt, Rdnt, Dect, Extt_,_ = T; derH, valt, rdnt, dect, extt_,_ = t
            for Extt, extt in zip(Extt_,extt_):
                sum_ext(Extt, extt)
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i] + dect[i])/2
            DerH[:] = [
                [[sum_dertuple(Dertuple,dertuple, fneg*i) for i,(Dertuple,dertuple) in enumerate(zip(Tuplet,tuplet))],
                  [V+v for V,v in zip(Valt,valt)], [R+r+base_rdn for R,r in zip(Rdnt,rdnt)], [(D+d)/2 for D,d in zip(Dect,dect)], 0
                ]
                for [Tuplet,Valt,Rdnt,Dect,_], [tuplet,valt,rdnt,dect,_]  # ptuple_tv
                in zip_longest(DerH, derH, fillvalue=[([0,0,0,0,0,0],[0,0,0,0,0,0]), (0,0),(0,0),(0,0),0])
            ]
            sum_ext(Extt_, extt_)

        else:
            T[:] = deepcopy(t)
