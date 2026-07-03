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

def clust_oF_(oF_):  # cluster Ts if called together, global only

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
    ''' with average_linkage:
    for _T,T in combinations(oF_,2): V = (comp_callers(_T,T) + comp_body(_T,T)) * min(_T.fc,T.fc); _T.V_[T] = V; T.V_[_T] = V 
    for t in C.N_: if t is not T: v__[j,i] += t.V_[T]  # += pairwise Vs '''
    return smF_, rF_

def inject_oF_(oF_, g):  # inject AST in g, recompile g[name]

    for oF in oF_:
        oF.fdef = nF_[oF.nF]; oF.g = g
        oF.body=[]; oF.fc=0; oF.g_=[]; oF.gV_=[]
        for node in ast.iter_child_nodes(oF.fdef):
            if r := build(oF,node): t,fc = r; oF.body += [t]; oF.fc += fc
        ast.fix_missing_locations(oF.fdef)
        exec( compile( ast.Module(body=[oF.fdef], type_ignores=[]), '<oF>', 'exec'), oF.g)
        oF.g[oF.fdef.name] = CoF.traced(oF.g[oF.fdef.name])
