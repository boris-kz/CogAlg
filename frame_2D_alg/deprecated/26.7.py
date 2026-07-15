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
