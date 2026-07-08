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

def get_fc(n):
    return n.fc if isinstance(n,CoF) else costs.get(n[0],0)+sum(get_fc(c) for c in n[1]) if isinstance(n,tuple) else costs.get(type(n),0)

def comp_prim(_n,n):
    if isinstance(_n,CoF) or isinstance(n,CoF):
        return comp_callers(_n,n) > ave if isinstance(_n,CoF) and isinstance(n,CoF) and n.nF==_n.nF else 0
    else:
        return _n[0]==n[0] and not (is_fork(_n) or is_fork(n)) if isinstance(_n,tuple) and isinstance(n,tuple) else 0 if isinstance(_n,tuple) or isinstance(n,tuple) else type(_n)==type(n)


