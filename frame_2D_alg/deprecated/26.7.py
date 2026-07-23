def split_oF_():  # divisive clustering
    sF_, rF_ = [], []
    for oF in oF_:
        # or if oF.w * len(gated_segment_AST): approximate gain from encapsulating the segment? same for merge but with component oFs?
        if (len(oF.body)-1) * wL > ave:  # * split w,c;  need to add callers to raw code forks too?
            _n= oF.body[0]; grp=[_n]; grp_=[]
            for n in oF.body[1:]:
                if comp_prim(_n,n): grp+=[n]
                else: grp_+= [grp]; grp =[n]
                _n=n
            grp_ += [grp]  # last
            if len(grp_) > 1:  # -1 * wL > ave?
                for grp in grp_:  # single refinement
                    fc = sum([get_fc(prim) for prim in grp])
                    sub = CoF(root=oF,fc=fc,body=grp); sub.caller_ = copy(oF.caller_)
                    sF_ += [sub]
        else: rF_ += [oF]
    return sF_, rF_

def clust_oF_(oF_):  # cluster Ts with similar body and callers

    grp_ = {}   # group same-typ oFs:
    for T in oF_:  # updated in split_oF_
        grp_.setdefault(T.typ, []).append(T)
    grp_ = list(grp_.values())
    C_ = [sum2O(g) for g in grp_]  # initial composites
    Ln,Lc = len(oF_),len(C_); L = Ln*Lc  # -1?
    _v__ = np.zeros((Ln, Lc))
    while True:
        v__ = np.zeros((Ln, Lc))
        for j,T in enumerate(oF_):  # refine by comp T x G_, cluster_P analog
            for i,C in enumerate(C_):
                v__[j,i] = (comp_callers(C,T) + comp_body(C,T)) * min(C.fc,T.fc)  # x centroid V
        C_ = [sum2O(oF_, w_=v__[:,i]) for i in range(Lc)]  # weighted re-aggr
        V, dV = v__.sum(), np.abs(v__-_v__).sum()
        if V*dV*(wcO*L) <= ave * (Ln+ ccO*L): break  # convergence
        _v__ = v__
    smF_,rF_ = [],[]  # final hard assign
    for i in range(Lc):
        t_ = [T for j,T in enumerate(oF_) if v__[j].argmax() == i]
        if t_:
            if (gV := v__[:,i].sum()) * ((len(t_)-1)*wL) > ave:
                nT = sum2O(t_)
                nT.memb = gV; fC = sum(t.fc for t in t_); nT.cmpr = fC - fC/len(t_)
                smF_ += [nT]
            else: rF_ += t_  # unpack if weak
    return smF_, rF_

def clust_oF_old():  # cluster Ts if called together, global only

        grp_ = {}  # group same-typ oFs?
        for T in oF_:  # updated in split_oF_
            grp_.setdefault(T.typ, []).append(T)
        grp_ = list(grp_.values())
        C_ = [sum2O(g) for g in grp_]  # initial composites
        Ln, Lc = len(oF_), len(C_);
        L = Ln * Lc  # -1?
        _v__ = np.zeros((Ln, Lc))
        while True:
            v__ = np.zeros((Ln, Lc))
            for j, T in enumerate(oF_):  # refine by comp T x G_, cluster_P analog
                for i, C in enumerate(C_):
                    v__[j, i] = (comp_callers(C, T) + comp_body(C, T)) * min(C.fc, T.fc)  # x centroid V
            C_ = [sum2O(oF_, w_=v__[:, i]) for i in range(Lc)]  # weighted re-aggr
            V, dV = v__.sum(), np.abs(v__ - _v__).sum()
            if V * dV * (wcO * L) <= ave * (Ln + ccO * L): break  # convergence
            _v__ = v__
        nT_, rF_ = [], []  # final hard assign
        for i in range(Lc):
            t_ = [T for j, T in enumerate(oF_) if v__[j].argmax() == i]
            if t_:
                if (gV := v__[:, i].sum()) * ((len(t_) - 1) * wL) > ave:
                    nT = sum2O(t_)
                    nT.memb = gV;
                    fC = sum(t.fc for t in t_);
                    nT.cmpr = fC - fC / len(t_)
                    nT.fdef = fdef;
                    fdef.name = f"oF{nT.id}"  # unique under removal: len(nF_) may repeat after pops
                    nT_ += [nT]
                else:
                    rF_ += t_  # unpack if weak
        if nT_:
            merged_oF_ = {t for nT in nT_ for t in nT.N_}  # list of merged oFs
            new_oF_ = [(F, nF_[F.nF]) for F in oF_ if F not in merged_oF_] + [(nT, nT.fdef) for nT in nT_]
            oF_[:] = [F for F, _ in new_oF_]
            nF_[:] = [fd for _, fd in new_oF_]
            iF_.clear();
            iF_.update({fd.name: j for j, (_, fd) in enumerate(new_oF_)})
            for i, (F, _) in enumerate(new_oF_): F.nF = i  # re-update nF index

def inject_oF_(oF_, g):  # inject AST in g, recompile g[name]

    for oF in oF_:
        oF.fdef = nF_[oF.nF]; oF.g = g
        oF.body=[]; oF.fc=0; oF.g_=[]; oF.gV_=[]
        for node in ast.iter_child_nodes(oF.fdef):
            if r := build(oF,node): t,fc = r; oF.body += [t]; oF.fc += fc
        ast.fix_missing_locations(oF.fdef)
        exec( compile( ast.Module(body=[oF.fdef], type_ignores=[]), '<oF>', 'exec'), oF.g)
        oF.g[oF.fdef.name] = CoF.traced(oF.g[oF.fdef.name])


def emit_oF(F, g, name='M'):  # rebuild: aligned bodies, no tails | nesting | gv_ rebinding
    def get_mm(F):  # emission-time resolver: leaf members via N_ flattening
        return {m for n in F.N_ for m in get_mm(n)} if F.N_ else {F}

    mm = sorted(get_mm(F), key=lambda m: m.id)  # leaf members, source donors
    body_ = []
    for j, t in enumerate(F.body):
        if is_fork(t):
            (a, b), o_ = t[1], t[-1]
            t_ = frozenset().union(*[get_mm(o) for o in o_]);
            g[f'_o{j}'] = t_  # resolve o_ to leafs, plant in oF.g
            rep_t = next(m for m in mm if m in t_)  # branch source: representative of assigned oFs
            rep_e = next(m for m in mm if m not in t_)  # else source: representative of complement
            body_ += [ast.If(test=ast.Compare(left=ast.Name('o', ast.Load()), ops=[ast.In()],
                                              comparators=[ast.Name(f'_o{j}', ast.Load())]),
                             body=[deepcopy(rep_t.fdef.body[j])], orelse=[deepcopy(rep_e.fdef.body[j])])]
        else:
            body_ += [deepcopy(mm[0].fdef.body[j])]  # shared op: any member's source
    fdef = ast.parse(f"def {name}(): pass").body[0]
    fdef.args = deepcopy(mm[0].fdef.args);
    fdef.args.args.insert(0, ast.arg(arg='o'))  # caller passes its old oF
    fdef.body = body_;
    ast.fix_missing_locations(fdef)
    exec(compile(ast.Module(body=[fdef], type_ignores=[]), '<oF>', 'exec'), g)
    return g[name], fdef

def clust_oF_1():  # exemplar-seeded merge for body compression

    F_ = copy(oF_); mrg_F_ = []
    for _F,F in combinations(F_,2):
        if _F.typ == F.typ:  # or all typs?
            w = comp_body(_F,F) * min(_F.fc,F.fc)
            _F.rim += [(F,w)]; F.rim += [(_F,w)]
            if w > ave: _F.w += w; F.w += w
    for _F in sorted(F_, key=lambda F: F.w, reverse=True):
        if _F.w <= ave: break
        if _F.fin: continue
        T = CoF(N_=[_F], typ=_F.typ, body=copy(_F.body), fc=_F.fc, caller_=copy(_F.caller_))
        for F,w in sorted(_F.rim, key=lambda t: t[1], reverse=True):
            if w <= ave: break
            if F.fin: continue
            merge_oF(T,F); T.N_ += [F]; T.caller_ |= F.caller_; F.fin = 1
        if len(T.N_) > 1:
            T.fc = sum(get_fc(t) for t in T.body); T.cmpr = sum(f.fc for f in T.N_) - T.fc
            if T.cmpr > ave: _F.fin = 1; mrg_F_ += [T]
            else:
                for F in T.N_[1:]: F.fin = 0
    for T in mrg_F_: oF_.append(T); nF_.append(None); T.nF = len(oF_)-1

def val_(TT, wTT=wTT, fd = 0):  # base eval: multi-variate rel match for membership

    m_,d_ = TT; ad_ = np.abs(d_); t_ = eps_(m_+ad_)  # ~ max comparand
    return (m_/t_) @ wTT[0]  # /= total = rdn, borrow in G.Bt only?

def comp_prim(_n,n):
    if isinstance(_n,CoF) or isinstance(n,CoF):
        return comp_callers(_n,n) > ave if isinstance(_n,CoF) and isinstance(n,CoF) and n.nF==_n.nF else 0
    else:
        return _n[0]==n[0] and not (is_fork(_n) or is_fork(n)) if isinstance(_n,tuple) and isinstance(n,tuple) else 0 if isinstance(_n,tuple) or isinstance(n,tuple) else type(_n)==type(n)

def comp_body(_n, n, C=0):  # estimated n-merge cost compression, init mean C=3, accum from recursive unpack

    if isinstance(n, CoF):
        if isinstance(_n, CoF):
            if n is _n: C += n.fc
            elif n.fc > ave_C:  # mean compression value
                C -= 2  # fork compression cost, ast.IfExp
                for _sub, sub in zip(_n.body, n.body): C = comp_body(_sub, sub, C)
                if len(n.body) > len(_n.body): C -= 2  # single tail fork cost
            else: C -= 2
        else: C -= 2
    elif isinstance(n, tuple):  # AST fork
        if isinstance(_n, tuple) and n[0] is _n[0]:
            C += costs.get(n[0], 0)
            for _sub, sub in zip(_n[1], n[1]): C = comp_body(_sub, sub, C)
            if len(n[1]) > len(_n[1]): C -= 2
        else: C -= 2
    elif type(_n) is type(n):
        C += costs.get(type(n), 0)  # AST leaf
    else: C -= 2
    return C

def comp_body1(_n, n, _f_=None, f=None):  # compare == merge: compression C + merged form (pure, replaces merge_oF)

    def gate(f_, sub): return              # 1-branch: these members do op, others skip

    def merge_seq(_q,q, _n_,n):  # merge aligned op sequences (bodies | node children) -> C, merged_
        C, Q = 0, []
        for _t, t in zip(_q, q):
            c,  = comp_body(_t, t, _n_,n); C += c; Q += [mt]
        for _t in _q[len(q):]:
            Q += [_t if isinstance(_t,tuple) and _t[0] is ast.IfExp else gate(_m_, _t)]; C -= 2  # tail: new member m skips these
        for  t in  q[len(_q):]:
            Q += [(ast.IfExp, (f_, sub))([m], t)]; C -= 2  # tail: prior members skip these
        return C, Q

    if _f_ is None: return merge_seq(_n.body, n.body, [_n], n)  # top level
    if isinstance(_n,tuple) and _n[0] is ast.IfExp and isinstance(_n[1][0],list):
        # existing fork, append it or closest sub_fork
        sub_ = list(_n[1:])  # [(m_, op), ...]
        r_ = [comp_body(op, n, bf_, f) for bf_,op in sub_]
        i  = int(np.argmax([c for c,_ in r_])); c, op = r_[i]
        if c > 0: sub_[i] = (sub_[i][0]+[f], op)  # n merges into branch i
        else:     sub_ += [([f], n)]; c = -2         # n becomes a new branch
        return c, (ast.IfExp, *sub_)
    if _n is n: return get_fc(n), _n  # same ref, skip

    if isinstance(_n,tuple) and isinstance(n,tuple) and _n[0] is n[0]:  # same ast node, merge children
        C, sub_ = merge_seq(_n[1], n[1], _f_, f)
        return C + costs.get(_n[0],0), (_n[0], tuple(sub_))
    elif type(_n) is type(n) and not isinstance(_n,(tuple,CoF)):  # other than tuple and CoF
        return costs.get(type(n),0), _n
    else:
        return -2, (ast.IfExp, (_f_,_n), ([f],n))  # 2-branch fork from one divergent op pair

def clust_oF_2():  # centroid form: exemplar seeds, fuzzy membership by mean link C, reform to convergence

    F_ = copy(oF_); T_ = []
    for F in F_: F.rim = []; F.w = 0; F.root_ = []
    for _F,F in combinations(F_,2):
        if _F.typ == F.typ:
            if (w := comp_body(_F.body, F.body)) > ave:  # compression estimate
                L = (_F,F,w); _F.rim += [L]; F.rim += [L]; _F.w += w; F.w += w
    E_ = []
    for F in sorted(F_, key=lambda F: F.w, reverse=True):  # exemplars: NMS in similarity space
        if F.w <= ave: break
        if not any((n if _n is F else _n) in E_ for _n,n,w in F.rim): E_ += [F]
    for E in E_:
        T = CoF(N_=[E], typ=E.typ); E.root_ = [(T,E.w)]
        T_ += [T]  # seed C = E.w: pass-1 stand-in
    while True:  # reform to convergence
        upd = 0
        for F in F_:
            root_ = []
            for T in T_:
                if F.typ == T.typ:
                    n_ = [G for G in T.N_ if G is not F]
                    C = sum(w for _n,n,w in F.rim if (n if _n is F else _n) in n_) / len(n_) if n_ else F.w
                    if C > ave: root_ += [(C,T)]
            root_.sort(key=lambda t: t[0], reverse=True)
            root_ = [(T,C) for i,(C,T) in enumerate(root_) if C > ave*(i+1)]  # rdn-scaled gate per extra root
            if [t for t,_ in root_] != [t for t,_ in F.root_]: upd = 1
            F.root_ = root_
        if not upd:
            break  # all memberships stable
        for T in T_: T.N_ = [F for F in F_ if any(t is T for t,_ in F.root_)]  # rebuild once per pass
        T_ = [T for T in T_ if T.N_]

    for T in sorted(T_, key=lambda T: sum(C for F in T.N_ for t,C in F.root_ if t is T), reverse=True):
        if len(T.N_) > 1:
            fc, cmpr, bod = form_body(T)
            C = cmpr - sum(F.fc for F in T.N_ if F.root_[0][0] is not T)  # member fc credited once, at primary root
            if C > ave:
                T.body = bod; T.fc = fc; T.w = C; T.cmpr = cmpr
                T.caller_ = set().union(*[F.caller_ for F in T.N_])
                oF_.append(T); nF_.append(None); T.nF = len(oF_)-1
                continue
        for F in T.N_: F.root_ = [r for r in F.root_ if r[0] is not T]  # release failed | singleton: next roots promote

def clust_oF_3():  # flood-fill by rim links, cluster_N form: frontier expansion + cluster contact-merge

    F_ = copy(oF_); T_ = []
    for F in F_: F.rim = []; F.root_ = []; F.rw_ = []
    for _F,F in combinations(F_,2):
        if (w := comp_body(_F.body, F.body)) > ave:  # compression, *= F.w?
            _F.rim += [(_F,F,w)]; F.rim += [(_F,F,w)]; F.w += w; _F.w += w
    for _F in sorted(F_, key=lambda F: F.w, reverse=True):
        w_, N_,L_ = [_F.fc], [_F], set(_F.rim)
        for _f,f, w in sorted(_F.rim, key=lambda l: l[2], reverse=True):
            F = f if _f in N_ else _f
            if w > ave * (len([rw for rw in F.rw_ if rw>w])-1):  # rdn = n stronger memberships, but not final in this loop?
                w_+=[w]; N_+=[F]; L_.update(F.rim)
        if len(N_) > 1 and (W :=sum(w_)) > ave:
            T = CoF(w=W, N_=N_, L_=L_)
            for f,w in zip(N_,w_): f.root_+=[T]; f.rw_+=[w]
            T_ += [T]
    # only if rdn can be reassigned?
    for T in T_:
        if len(T.N_) > 1:  # skip Ts emptied by contact-merge
            form_body(T)
            if T.cmpr > ave:
                oF_.append(T); nF_.append(None); T.nF = len(oF_)-1
            else:
                for F in T.N_: i = F.root_.index(T); F.root_.pop(i); F.rw_.pop(i)

def clust_oF_dense():  # simplified oF rim-mediated centroid clustering

    F_ = copy(oF_)
    for F in F_: F.rim = []
    for _F, F in combinations(F_,2):
        if (w := comp_body(_F.body, F.body)) > ave: _F.rim += [F,w]; F.rim += [_F,w]; _F.w += w; F.w += w
    _T_ = []
    for _F in F_:
        _F.w = sum([w * (_F.w/(_F.w+F.w)) for F,w in _F.rim])  # /= rdn to stronger F in rim
        if _F.w > ave:
            N_ = [_F]+ [L[0] for L in _F.rim]; T = CoF(N_=N_, L_= [0 for _ in N_])  # no T.L_,T.w yet
            form_body(T); _T_ += [T]
    while _T_:
        T_,nT_ = [],[]; W = DW = 0
        root__ = [[] for f in F_]  # dense, replace in parallel
        for i,T in enumerate(_T_):
            Tw = Dw = 0; N_,L_ = [],[]
            for j, w in enumerate(T.L_):  # aligned with F_ in all Ts
                w = comp_body(T.body, F.body)
                N_+= [F]; L_+= [w]; Tw += w; Dw += F_[j].root_[i]
            DW += Dw; W += Tw
            if Dw > ave:
                T.N_= N_; T.L_= L_; T.w = W  # add adjust each w for rdn to stronger roots?
                T_+= [T]; nT_+=[T]
            else: nT_ += [[]]
        if any(nT_) and DW > ave:   # continue refinement as long there's changes in the members
            for T,nT in zip(T_,nT_):
                if nT: form_body(T)  # rebuild from new N_
        else: break
    for T in T_:
        if T.w > ave: oF_.append(T); nF_.append(None); T.nF = len(oF_) - 1
        else:
            for F in T.N_: i = F.root_.index(T); F.root_.pop(i); F.rw_.pop(i)

def sum2G(ft_, fTT, root=None, init=1):  # core clustering function

    if not init:
        N_,_,ntt,nc,nr = ft_[0]; N_+=root.N_; ntt+=root.Nt.dTT; nc+=root.Nt.c; nr+=root.Nt.r; ft_[0] = N_,_,ntt,nc,nr
        if len(ft_)>1: L_,_,ltt,lc,lr=ft_[1]; L_+=root.L_; ltt+=root.Lt.dTT; lc+=root.Lt.c; lr+=root.Lt.r; ft_[1]=L_,_,ltt,lc,lr
    Ft_ = []
    for ft, nF in zip_longest(ft_,('Nt','Lt','Bt')):
        if ft: n_,_,tt,c,r = ft; Ft_+= [CF(N_=n_,nF=nF,dTT=tt,m=(vt:=val_(tt,wTT,1))[0],d=vt[1],c=c,r=r)]
        else:  Ft_ += [CF()]
    C_ = [c for N in ft_[0][0] for c in N.C_]  # splice centroids
    Ft_ += [sum2F(list(set(C_)), root.Ct) if C_ else CF()]  # add multiple root_ in Cs?
    G = comb_Ft(*Ft_, root, wTT=fTT)
    N_ = G.N_; N=N_[0]; G.sub = N.sub+1 if G.L_ else N.sub; r=G.r
    if G.Lt:  # sub+
        Lt = G.Lt; L_,lm,ld,lr = Lt.N_,Lt.m,Lt.d,Lt.r; L=len(L_)-1; Av = ave+avd
        if gv_(Vn := (lm+ld)*wcN - Av* (lr+1+ccN*L)):  # for cluster_N
            c = G.Lt.c; E_ = get_exemplars({N for link in L_ for N in link.N_}, r,c)
            if gv_(Vn* (wcC-wcN)* (mdecay(L_)-decay) - Av* (lr+1+(ccC-ccN)*L)):
                r +=1; G_,r = cluster_C(G.Nt,E_,r,c)  # higher V, low decay, eval cluster_P
            else:      G_,r = cluster_N(G.Nt,E_,r,c)  # updates G
            if G_ and gv_(val_(G.Nt.dTT,G.wTT*ttA) * ((G.c+wAgg) /(G.r+r+cAgg)) * ((len(G_)-1)**2 *wL) - ave):  # if full cross_comp?
                cross_comp(G.Nt,r)
    if G.Bt:
        Bt = G.Bt; bd,br,L = Bt.d,Bt.r,len(Bt.N_); rroot = root.root if root.root else 0
        if N.typ!=1 and bd*(wAgg*L) > avd*(br+cAgg*L): [F2N(L) for L in Bt.N_]; cross_comp(Bt, br)  # no ddfork
        if rroot: Bt.brrw = Bt.m * (rroot.m * (decay * (rroot.span/G.span)))  # external lend only, need to subtract from root?
    FV_(CoF.get(), G.dTT, G.c, G.r)
    return G
'''
        G__ = [G for T in tile_ for G in T.N_ for sub in G.N_ if sub.sub == G.sub]  # sub+'s N.sub should be == G
        L_,TT,lc,lr,_ = comp_N_(combinations(G__,2),R); lm, ld = val_(TT, fd=1)
        L=len(L_)-1; Av = ave+avd
        E_ = get_exemplars({N for link in L_ for N in link.N_}, lr,lc)
        Fn = sum2F(G__); CC_ = []
        Vn = (lm+ld)*wcN - Av* (lr+1+ccN*L)
        if gv_(Vn* wcC* (mdecay(L_)-decay) - Av* (lr+1+ccC*L)):
            CC_,r = cluster_C(Fn.Nt,E_,lr,lc)
            
    while elev < max_elev:
        tile_,C,R = fill_frame(iY,iX, elev, T)  # project from seed tile
        if tile_: # sparse,2D
            Fr = sum2F(tile_)  # higher-scope tile( oH( aH
            if cross_comp(Fr.Nt, rr=0):  # spec-> tN_,tC_,tL_, proj comb N_'L_?
                if elev and ffb:  # ffb =1 in main, no ffeedback in added tiles
                    Fr,aTT,oTT,aH,oH = ffeedback(Fr, aTT,oTT,aH,oH)  # term,form oH(aH
                elev +=1; T=Fr  # next-extension seed
                
    cross_comp(Ct,r)  # all distant Cs, seq C_ in eigenvector = argmax(root.wTT)?
    
                        oy_ = _y_.intersect(_y_,Y_-1); oY_ = _Y_.intersect(y_+1); ox_ = _x_.intersect(_x_,X_-1); oX_ = _X_.intersect(x_+1)  # adjacent
                        if oy_ or oY_ or ox_ or oX_:
                            if gv_(val_( base_comp(_N,N)[0])) > ave:
                                add2F(_N,N, 1)
                                add_Nt(_N); _N.m,_N.d = val_(_N.dTT, fd=1)
                                tile_[i].cont_[j][0] = _N   
'''
def frame_H(image, iY,iX, Ly,Lx, Y,X, rV, max_elev=4, ffb=0):

    def fill_frame(_iy,_ix, elev, T):  # expand level_frame from pixel-level seed tile

        frame = np.full((Ly,Lx), None, dtype=object)  # level scope
        cy,cx = (Ly-1)//2,(Lx-1)//2; y,x = cy,cx  # start=mean
        PV__ = np.zeros([Ly,Lx])
        T_ = []
        while T and gv_(val_(T.dTT*T.wTT*ttFrm) * ((T.c+wFrm)/(T.r+cFrm)) - ave):
            frame[y,x]=T; T_+=[T]; dy_dx = T.box[2:] -T.box[:2]
            pTT, pc = proj_N(T, np.hypot(*dy_dx), dy_dx, elev, T.c)  # no proj r?
            if gv_(ave - val_(pTT*T.wTT*ttFrm) * ((pc+wFrm)/(T.r+elev+cFrm))):  # inverted val
                proj_focus(PV__,y,x,T)
                pv__ = PV__.copy(); pv__[frame != None] = 0
                y,x = np.unravel_index(pv__.argmax(), PV__.shape)
                if gv_(PV__[y,x] - ave):
                    iy = _iy+ (y-cy)*(T.box[2]-T.box[0]); ix = _ix+ (x-cx)*(T.box[3]-T.box[1])
                    T = frame_H(image, iy,ix, Ly,Lx, Y,X, rV, max_elev=elev)[0]  # expand new tile to current level, no fb
                else: break
            else: break
        if T_:
            TT,C,R = sum_vt(T_, wTT=ttFrm); R += elev
            if val_(TT*ttFrm) * ((C+wFrm)/(R+cFrm)) > ave:
                return T_,C,R
        return [], 0, 0

    global ave,avd; aTT=oTT=np.zeros((2,9)); aH,oH = [],[]  # regime refs across levs / ffeedback
    elev,Fr = 0,[]
    if T := vect_edge(frame_blobs_root(comp_pixel(image[iY:iY+Ly, iX:iX+Lx]), rV), rV):  # initial pixel tile
        T.yx = np.array([iY+Ly//2, iX+Lx//2]); T.box = np.array([iY,iX, min(iY+Ly,Y), min(iX+Lx,X)]); T.span = np.hypot(Ly,Lx)/2
        if not cross_comp(T.Nt, rr=0):
            T = []
    if not T or not max_elev: return T, ([], np.zeros((2,9)),0,0,0)  # frame_H(0) = pixel tile
    while elev < max_elev:
        tile_,C,R = fill_frame(iY,iX, elev, T)  # project from seed tile
        if tile_: # sparse,2D
            Fr = sum2F(tile_)  # higher-scope tile( oH( aH
            if cross_comp(Fr.Nt, rr=0):  # spec-> tN_,tC_,tL_, proj comb N_'L_?
                if elev and ffb:  # ffb =1 in main, no ffeedback in added tiles
                    Fr,aTT,oTT,aH,oH = ffeedback(Fr, aTT,oTT,aH,oH)  # term,form oH(aH
                elev +=1; T=Fr  # next-extension seed
            else: break
        else: break
    if Fr: FV_(CoF.get(), Fr.dTT, Fr.c, Fr.r)
    return Fr  # intra-lev feedback

def vect_edge(tile, rV=1):  # PP_ cross_comp and floodfill to init focal frame graph, no recursion:

    global ave,avd,Fw_,Fc_  # /= projected V change:
    def PP2N(PP):
        P_,L_,B_,verT,latT,A,S,box,yx, m,d,c = PP
        kern = np.array(latT[:4])
        [mM, mD, mI, mG, mA, mL], [dM, dD, dI, dG, dA, dL] = verT  # re-pack in dTT:
        dTT = np.array([ np.array([mM, mD, mL, mI, mG, mA, mL, mL / 2, 0]),  # extA=0
                         np.array([dM, dD, dL, dI, dG, dA, dL, dL / 2, 0])])
        y,x,Y,X = box; dy,dx = Y+1-y, X+1-x
        A = [np.array(A), np.sign(dTT[1] @ ttVct[1])]  # append sign
        PP = CL(typ=0, dTT=dTT,m=m,d=d,c=c,r=1, kern=kern,yx=yx,angl=A,span=np.hypot(dy/2,dx/2))  # set root in trace_edge
        m_, d_ = np.zeros(6), np.zeros(6); PP.B_ = B_
        for B in B_: m_ += B.verT[0]; d_ += B.verT[1];
        ad_ = np.abs(d_); t_ = m_ + ad_  # ~ max comparand
        m = m_/eps_(t_) @ w_t[0] - ave*2; d = ad_/eps_(t_) @ w_t[1] - avd*2
        PP.Bt = CF(N_=B_, m=m, d=d, r=2, root=PP,nF='Bt')
        for P in P_: P.root = PP
        if hasattr(P,'nt'):  # typ=1?
            PP.root_ = []  # Gd.root_: cores, no centroids? multiple PPms may share same PPd?
            for dP in P_:
                for P in dP.nt: PP.root_ += [P.root]  # PPm
        return PP
    blob_ = tile.N_; G_,TT,C,R = [],np.zeros((2,9)),0,0
    for blob in blob_:
        if not blob.sign:
            if gv_(blob.G * wVct - ave * cVct):  # proxy for comp_slice and slice_edge
                edge = slice_edge(blob, rV); L = len(edge.P_)-1
                if gv_(edge.G * (wVct*L) - sum([P.latT[4] for P in edge.P_]) * (cVct*L)):
                    PPm_ = comp_slice(edge, rV, ttVct)  # add comp_slice's weights?
                    N_ = [PP2N(PPm) for PPm in PPm_]
                    c = sum([PPm.c for PPm in N_]); C += c
                    for PPd in edge.link_: PP2N(PPd)  # we don't form Gds?
                    for N in N_:
                        if N.B_:
                            PPd_ = [B.root for B in N.B_]; sum2F(PPd_,N.Bt)
                            N.Bt.N_ = PPd_; [setattr(B,'root',N.Bt) for B in PPd_]
                    tt,c,r = sum_vt(N_)
                    if gv_(val_(tt*ttVct) * ((c+wVct)/ (3+cVct)) * ((len(PPm_)-1)*wL) - ave):
                        G_,TT,c,R = trace_edge([F2N(n) for n in N_], G_,TT,c,3,tile); C += c  # flatten B_-mediated Gs
    if G_:
        Nt = CF(nF='Nt', root=tile); Nt.N_=G_; Nt.dTT=TT; Nt.c=C; Nt.r=1; Nt.H=tile.H; tile.Nt=Nt; tile.dTT=TT; tile.c=C
        if val_(TT*ttVct) * ((C+wFrm)/ cFrm) * ((len(G_)-1)*wL) > ave:  # L for trans-comp only?
            A_ = [G.angl[0] for G in G_ if G.angl]
            tile.angl = [np.sum(A_, axis=0) if A_ else np.zeros(2), np.sign(tile.dTT[1] @ ttVct[1])]
            FV_(CoF.get(), TT,C,1)
            return tile

    def splice_T_(tile_):  # splice not-terminated Ns

        for i, _T in enumerate(tile_):
            for T in tile_[i:]:
                for i, (_C,C,_N_,N_) in enumerate(zip(_T.box, T.box, (_T.y_,_T.Y_,_T.x_,_T.X_), (T.y_,T.Y_,T.x_,T.X_))):
                    if abs(_C-C)==1:  # adjacent sides
                        for _N,N in product(_N_,N_):
                            if N is _N: continue
                            if val_( proj_V(_N,N)):
                                if gv_(val_( base_comp(_N,N)[0])) > ave:
                                    add2F(_N,N, 1)
                                    add_Nt(_N); _N.m,_N.d = val_(_N.dTT, fd=1)
                                    N_ = _N  # need to to index this

def cross_comp(root, rr, fC=0):  # core function mediating recursive rng+ and der+ cross-comp and clustering

    N_,G_ = root.N_,[]  # root is Ft, converted below, rc=rdn+olp, comp N_| B_| C_:
    if Lt := comp_C_(N_,rr,fC=1) if fC else comp_N_(combinations(N_,2),rr):
        L_,TT,c,r, cV = Lt
        oF_[CoF.get().nF].V_ += [cV]  # combined comp_ results
        root.L_ = L_  # val=m+d /clust, m/comp
        if gv_(val_(TT*ttcN) * ((c+wcN)/(r+ccN)) * ((len(L_)-1)*wL) - ave):  # if +ve, store neg gate values
            E_ = get_exemplars({N for L in L_ for N in L.N_}, r,c)
            G_,r = cluster_N(root, E_,r,c)  # cluster_C, _P, eval?
            if G_:
                if not root.typ: F2N(root)  # promote at 1st sub+ or agg+
                root.H += [sum2F(L_,root,froot=1)]  # dLev per L_
                root.Nt = sum2F(G_,root,froot=2)  #| C_?
                if Ct := root.Ct:
                    xcomp(Ct,r,root) # sub+'agg+
    return G_

def xcomp(N_, merge=1):  # light version for contiguous tiles, also Bs?
    out_ = []
    for N in N_:
        if N.fin: continue  # merged
        for _N in N.rim:
            if _N.fin: continue  # contiguous only, maybe in 2D and deep
            m = val_(base_comp(_N,N)[0])
            if m > ave:
                add2F(_N.N, merge); N.rim += _N.rim; _N.fin = 1
        out_ += [N]
    return out_

def comp_N_(_pairs, r, tnF=None, root=2):  # incremental-distance cross_comp, max dist depends on prior match

    def proj_V(_N, N, dist, dy_dx, dec, r):  # _N x N induction
        Dec = dec or decay ** ((dist / ((_N.span + N.span) / 2)))
        iTT = (_N.dTT + N.dTT) * Dec
        eTT = (_N.Rt.dTT + N.Rt.dTT) * Dec
        C = min(_N.c, N.c); R = (_N.r + N.r) / 2
        if val_((eTT + iTT) * ttPrj) * (C / (cPrj + r + R)) * wPrj > ave:  # not oF, spec / link:
            eTT += proj_N(N, dist, dy_dx, r, N.c, dec)[0]  # pTT/ L_,B_,rim, if pV >0
            eTT += proj_N(_N, dist, -dy_dx, r, _N.c, dec)[0]  # reverse direction
        return iTT + eTT

    pairs, olp_ = [],[]  # no olp_?
    for _N,N in _pairs:  # get all-to-all pre-links
        if _N.sub != N.sub: continue  # or comp x composition?
        if N is _N: olp_ += [N]  # overlap = unit match, no miss
        else: dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx); c = min(_N.c,N.c); pairs += [[dist, dy_dx, _N,N, c]]
    N_, L_ = [],[]
    for pL in sorted(pairs, key=lambda x: x[0]):  # proximity prior, test compared?
        dist, dy_dx, _N,N, lc = pL  # rim angl is not canonic
        pTT = proj_V(_N,N, dist, dy_dx, root.m if root!=2 else decay** (dist/((_N.span+N.span)/2)), r)  # based on current rim
        m,d = val_(pTT,ttN_,1); lr = r+ (N.r+_N.r)/2  # +|-match certainty
        if m > 0:
            if gv_(m*(lc/lr)*wN - ave*(r+cN)):  # comp if marginally predictable, update N.Rt pair eval, ave / proj surprise value?
                Link = comp_N(_N,N, lr,lc, full=not tnF, A=dy_dx, span=dist, rL=root)
                Link.rTT = np.abs(pTT - Link.dTT) / eps_(Link.dTT)  # relative prediction error to fit oF, direction-agnostic
                L_+= [Link]; N_+= [_N,N]
            else:
                pL = CL(typ=-1, N_=[_N,N],dTT=pTT,m=m,d=d,c=lc,r=lr, angl=[dy_dx,1],span=dist)
                L_+= [pL]; N.rim+=[pL]; _N.rim += [pL]; N_+=pL.N_  # add neg C?
                # oF.N_ is +-Ls
        else: break  # beyond initial induction range, re-sort by proj_V?
    if L_:
        for N in set(N_):
            if N.rim: N.Rt = sum2F(N.rim)
        cV = FV_(CoF.get(), *sum_vt(L_))  # +-ve Ls for oF, no oF.N_?
        pL_ = [L for L in L_ if (L.m > ave or L.typ == -1)]
        return pL_,*sum_vt(pL_), cV  # +ve only, redundant +-ve for oF

