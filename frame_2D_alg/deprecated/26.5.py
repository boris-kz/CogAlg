
class CoF(CF):  # oF/ code fork, N_,dTT: data scope, w = vt_(wTT)[0]?
    name = "func"
    _cur = contextvars.ContextVar('oF')
    def __init__(f, **kw):
        super().__init__(**kw)
        f.call_ = kw.get('call_',[])  # call tree
        f.typ_ = kw.get('typ_',[])  # call_ flattened and nested with tFs
        f.fw,f.fc,f.fr = [kw.get(x,0) for x in ('fw','fc','fr')]  # call_ gain, cost, rdn: multiple oFs on same data?
        f.gate_ = kw.get('gate_', [])  # each gate eval within the function
    @staticmethod
    def get(): return CoF._cur.get(Z)
    @staticmethod
    def traced(func):
        if getattr(func,'wrapped',False): return func
        @wraps(func)
        def inner(*a, **kw):
            _CoF = CoF._cur.get()  # oF.N_ = call_:
            oF = CoF(nF=onF_.index(func.__name__),root=_CoF); oF.wTT = FTT_[oF.nF]; oF.fw = Fw_[oF.nF]; oF.fc = Fc_[oF.nF]; _CoF.call_+=[oF]
            CoF._cur.set(oF); out = func(*a, **kw)
            if oF.call_:  # complete at this point
                call_ = flat_(oF); L=(len(call_)-1)  # flat call tree
                if oF.fw*L > ave*(oF.fc*L):  # eval summarize, fc*fr if multiple oFs on the same data?
                    sum2F(call_,oF)  # sum data only
            oF.wTT= cent_TT(getattr(oF,'rTT',oF.dTT), oF.r)  # rTT covers cluster compression
            oF.w += np.mean(oF.wTT)
            CoF._cur.set(_CoF)
            return out
        inner.wrapped = True
        return inner
    def __bool__(f): return bool(f.N_)

Z = CoF(nF='Z'); Z.gF = CoF(nF='Z', root=Z, fo=0); CoF._cur.set(Z)

class CoF(CF):
    name = "func"
    _cur = contextvars.ContextVar('oF')
    def __init__(f, fo=1, **kw):
        super().__init__(**kw)
        f.fo = fo
        f.call_ = kw.get('call_',[])  # sub-oFs
        f.typ_  = kw.get('typ_', [])
        f.fw,f.fc,f.fr = [kw.get(x,0) for x in ('fw','fc','fr')]
        f.gF = CoF(nF=kw.get('nF',0), root=f, fo=0) if fo else None  # embedded gate aggregate
    @staticmethod
    def get(): return CoF._cur.get(Z)
    @staticmethod
    def traced(func):  # traverse call_|| gate_
        if getattr(func,'wrapped',False): return func
        @wraps(func)
        def inner(*a, **kw):
            _CoF = CoF._cur.get()
            oF = CoF(nF=onF_.index(func.__name__),root=_CoF, fo=1)
            oF.gF = CoF(nF=oF.nF, root=oF, fo=0); gF = oF.gF  # always aligned
            oF.wTT = FTT_[oF.nF]; oF.fw = Fw_[oF.nF]; oF.fc = Fc_[oF.nF]; _CoF.call_+= [oF]
            if fo:
                gF.wTT = GTT_[gF.nF]; gF.fw = Gw_[gF.nF]; gF.fc = Gc_[oF.nF]; _CoF.gF.call_+= [gF]  # call_= gate_
            CoF._cur.set(oF); out = func(*a, **kw)
            if oF.call_:
                tree = flat_(oF); L=len(tree)-1  # flatten call_
                if oF.fw*L > ave*(oF.fc*L): sum2F(tree,oF); oF.wTT = cent_TT(getattr(oF,'rTT', oF.dTT), oF.r); oF.w += np.mean(oF.wTT)
                if fo:
                    tree = flat_(gF); L=len(tree)-1  # flatten gate_
                    if gF.fw*L > ave*(oF.fc*L): sum2F(tree,gF); gF.w += np.mean(gF.wTT)  # no cent_TT?
            CoF._cur.set(_CoF)
            return out
        inner.wrapped = True
        return inner
    @staticmethod
    def gate(V, cost, dTT=None, name=''):
        _CoF = CoF._cur.get()
        g = CoF(nF=name, root=_CoF, fo=0); g.fc=cost; g.w = V - ave*cost; g.span = len(_CoF.call_)
        if dTT is not None: g.dTT = dTT
        _CoF.gF.call_ += [g]; _CoF.gF.fw += g.w; _CoF.gF.fc += cost
        return g.w > 0
    def __bool__(f): return bool(f.N_)

def merge(_Q,Q, root=None):  # combine aligned ops, if-fork per miss, no inline recursion

    mrg = CoF(nF=Q.nF, N_=[_Q,Q], root=root or Q.root)  # add Q to N_ for unpacking purpose during eval
    call_, Q_ = [],set()
    for _f,f in zip_longest(_Q.call_, Q.call_, fillvalue=None):  # call_: op sequence in func or block
        if _f is not None and f is not None:
            if _f.nF==f.nF:
                mrg.call_ += [f]  # match: copy the sequence
            elif sum_vt([_f,f],fm=1)[0]>ave:  # miss: add gate to select forks (temporary)
                # i think we need to add eval func here, else all combinations are getting gates if their nF is different
                gate = CoF(nF='E', call_=[_f,f], c=_f.c+f.c, root=mrg, fo=0)  # fc=Ec_[eval.nF]? root is not reassigned
                mrg.call_ += [gate]  # need to add eval func?
        else:
            call_ += [_f if f is None else f]  # leftover, if their call_ size is different
    if mrg.call_:
        mrg.c +=_Q.c+Q.c; mrg.fw +=_Q.fw+Q.fw; mrg.fc+= Fc_[Q.nF]  # included for final mrg eval, gates are transparent?
        if call_:  # leftover calls, pack them into gate? or repack them into call_?
            mrg.call_ += call_
    else:  # there is no merged fs or gate in mrg, return and recycle the input typs?
        Q_ = set([_Q, Q])

    return mrg, Q_

def CopyF(F, root=None, r=1):  # F = CF
    C = CF(dTT=F.dTT*r, m=F.m, d=F.d, c=F.c, r=F.r, root=root or F.root, nF=F.nF)
    C.N_ = [(N if isinstance(N, CN) else (CopyF(N,root=C) if isinstance(N, CF) else [])) for N in F.N_]  # flat
    return C

def Copy_(N, root=None, init=0, typ=None):

    if typ is None: typ = 3 if init<2 else N.typ  # G.typ = 3, C.typ=2
    C = [CC, CN][typ-2](dTT=deepcopy(N.dTT),typ=typ); C.root = N.root if root is None else root
    for attr in ['m','d','c','r']: setattr(C,attr, getattr(N,attr))
    if not init and C.typ==N.typ: C.Nt = CopyF(N.Nt,root=C)
    if typ:
        for attr in ['fin','span','mang','sub','exe']: setattr(C,attr, getattr(N,attr))
        for attr in ['nt','kern','box','compared','dTT','m','d','c','r']: setattr(C,attr, copy(getattr(N,attr)))
        for attr in ['Nt','Lt','Bt','Ct','Xt','Rt']: setattr(C, attr, CopyF(getattr(N,attr), root=C))
        if init:  # new G
            C.yx = [N.yx]
            if N.angl is not None: C.angl = [copy(N.angl[0]), N.angl[1]]  # get mean
            C.L_ = [l for l in N.rim if l.m>ave]; N.root=C; C.fin = 0  # else centroid
            C.N_ = [N]
        else:
            C.Lt=CopyF(N.Lt); C.Bt=CopyF(N.Bt)  # empty in init G
            C.angl = copy(N.angl); C.yx = copy(N.yx)
        if typ > 1: C.Rt = CopyF(N.Rt)
    return C

def split(Q, root=None):  # invert merge at most cost-unamortized gate

    g = max(Q.gF.call_, key=lambda g: max(b.fc*b.c - b.fw*b.c for b in g.call_))
    Qs = []
    for b in g.call_:
        S = CoF(nF=Q.nF, root=root or Q.root); S.call_=[b if f is g else f for f in Q.call_]
        S.gF.call_=[x for x in Q.gF.call_ if x is not g]
        S.c=b.c; S.fw=sum(f.fw for f in S.call_ if isinstance(f,CoF)); S.fc=Fc_[Q.nF]
        Qs += [S]
    return Qs

def eval(mrg):  # after adding multiple oFs in mrg.call_, not just the pair in single merge()

    fW = eW = 0
    call_ = []
    for F in mrg.call_:  # or typ_: nFs?
        if F.nF=='E': eW += F.w*F.c  # vs plain count
        else: call_+= [F]; fW += F.fc
    Z.typ_ += [mrg if fW/eW >ave else call_[:]]
    # unpack mrg if high forking cost ratio

    def comp_call_(_C, C):  # draft, need to include identity - cost of forking?

        # common aligned calls
        m = sum(1 for _c, c in zip(_C.call_, C.call_) if _c.nF == c.nF)
        d = min(len(_C.call_), len(C.call_)) - m
        # common callers, single / call
        _caller_, caller_ = set(_C.root) if isinstance(_C.root, list) else set([C.root]), set(C.root) if isinstance(C.root, list) else set([C.root]),
        olp = _caller_ & caller_
        off = list(_caller_-olp) + list(caller_-olp)
        M = sum([c.w * c.c for c in olp])
        D = sum([c.w * c.c for c in off])
        return m, d, M, D

def add_typ_(oF):  # record oF vals for weighting, mapped to global FTT_

    typ_ = [[] for _ in range(len(FTT_))]
    for F in flat_(oF): typ_[F.nF] += [F]  # flattened call tree
    for i, F_ in enumerate(typ_):
        if F_:
            T = sum2F(F_,CoF()); T.nF=i; T.wTT=cent_TT(getattr(T,'rTT',T.dTT),T.r)
            T.N_ = T.call_; T.call_ = F_[0].call_; typ_[i]=T  # N_=instances, call_=callees
            T.root = [F.root for F in F_]  # all callers per typ
    oF.typ_ = typ_
    if any(typ_): add2F(oF,sum2F([t for t in typ_ if t],CoF()))  # refine summed call_?

def cluster_AST(Q):  # primitives are triggers for next oF, wrap them in CoF, for frame only?

    seg_, seg = [], []
    for c in Q:
        seg += [c]
        if isinstance(c,CoF): seg_ += [CoF(call_=seg)]; seg = []
    return seg_

TYP_SKEL = None

def build_typ_skel(root_F='frame_H', modules=('agg_recursion','comp_slice','slice_edge')):
    import importlib, inspect
    funcs = {}
    for m in modules:
        mod = importlib.import_module(m)
        for n in onF_+[root_F]:
            fn = getattr(mod, n, None)
            if fn and n not in funcs: funcs[n] = ast.parse(inspect.getsource(fn)).body[0]
    def items(fn):
        seq = []
        for stmt in fn.body:
            names = [getattr(c.func,'id',None) or getattr(c.func,'attr',None)
                     for c in ast.walk(stmt) if isinstance(c,ast.Call)]
            nF_ = [n for n in names if n in onF_]
            seq += [nF_.index(nF) for nF in nF_] if nF_ else [stmt]
        return seq
    oF_ = [items(funcs[n]) if n in funcs else [] for n in onF_]
    op_ = items(funcs[root_F]) if root_F in funcs else []
    return oF_, op_

class CoF1(CF):
    name = "func"
    _cur = contextvars.ContextVar('oF')
    def __init__(f, fo=1, **kw):
        super().__init__(**kw)
        f.call_ = kw.get('call_',[])  # top-level AST items
        f.typ_ = kw.get('typ_',[])  # unique oFs in call_
        f.fw,f.fc,f.fr = [kw.get(x,0) for x in ('fw','fc','fr')]
        f.gF = CoF(nF=kw.get('nF',0), root=f, fo=0) if fo else None  # oF gate, if any
    @staticmethod
    def get(): return CoF._cur.get(Z)
    @staticmethod
    def traced(func):
        if getattr(func,'wrapped',False): return func
        @wraps(func)
        def inner(*a, **kw):
            _CoF = CoF._cur.get()
            oF = CoF(nF=onF_.index(func.__name__), root=_CoF); gF=oF.gF  # auto-created
            oF.wTT = FTT_[oF.nF]; oF.fw = Fw_[oF.nF]; oF.fc = Fc_[oF.nF]; _CoF.call_ += [oF]
            # gF.wTT = ETT_[gF.nF]; gF.fw = Ew_[gF.nF]; gF.fc = Ec_[gF.nF]  # in gate | add_gF?
            CoF._cur.set(oF); out = func(*a, **kw)
            if oF.call_:
                for i, F in zip((1,0),(oF, oF.gF)):
                    tree = flat_(oF); L=len(tree)-1
                    if oF.fw*L > ave*(oF.fc*L):
                        sum2F(tree, oF); wtt = getattr(oF,'rTT',oF.dTT); oF.wTT = cent_TT(wtt,oF.r) if i else wtt  # oF only?
                        oF.w += np.mean(oF.wTT)  # not affected by cent_TT?
            CoF._cur.set(_CoF)
            return out
        inner.wrapped = True
        return inner
    @staticmethod
    def gate(gain,cost, dTT=None):  # gain = evaled_block_w - default_block_w
        _CoF = CoF._cur.get(); gF = _CoF.gF
        Ec = Ec_[gF.nF]; T = ave*cost  # threshold
        if gain>T: gw = -Ec  # pass: cost only, body runs anyway
        else:      gw = cost-gain - Ec  # fail: saved body cost - proj gain - eval cost
        g = CoF(nF=gF.nF, root=_CoF, fo=0); g.w = gw; g.fc = Ec; g.span = len(_CoF.call_)
        if dTT is not None: g.dTT = dTT
        gF.N_+=[g]; gF.fw += gw; gF.fc += Ec
        return gain>T
    def __bool__(f): return bool(f.call_)

def typ_AST(nF, typ_):  # static body scan: typ refs + primitive stmts

    seq = []
    for stmt in ast.parse(inspect.getsource(globals()[onF_[nF]])).body[0].body:  # .body[0]: FunctionDef, .body: function stmts
        oF_ = [n for n in (getattr(c.func, 'id', None) or getattr(c.func, 'attr', None)
                 for c in ast.walk(stmt) if isinstance(c, ast.Call)) if n in onF_]
        seq += [typ_[onF_.index(n)] for n in oF_] if oF_ else [stmt]
    return seq

def add_gF(T, stmts):  # per typ.call_, pack primitives in next oF.gF.call_

    C_, gate = [],[]
    for c in stmts:
        if isinstance(c,CoF):
            c.gF.call_ = gate
            c.gF.fc = sum(costs.get(type(n),0) for p in gate for n in ast.walk(p))
            C_ += [c]; gate = []
        else: gate += [c]
    return C_

def add_call_typ_(T):

    typ_ = [[] for _ in range(len(FTT_))]
    for F in flat_(T): typ_[F.nF] += [F]  # bin runtime instances, T.call_ still trace tree
    for i, F_ in enumerate(typ_):
        if F_:
            t = sum2F(F_,CoF()); t.nF=i; t.wTT=cent_TT(getattr(t,'rTT',t.dTT), t.r)
            t.N_ = t.call_  # instances → N_
            t.call_ = add_gF(t, typ_AST(i))  # static AST structure
            typ_[i] = t
    T.typ_ = typ_
    T.call_ = add_gF(T, typ_AST(T.nF))  # T's own static structure
    if any(typ_): add2F(T, sum2F([t for t in typ_ if t], CoF()))

def cluster_call_():  # cluster T callees if called together, for global Z

    def seg_eval(seg, seg_):
        if sum(c.fc for c in seg) > ave * Ec_[0]:  # base eval cost
            T = sum2F(seg); nF = T.nF = seg[0].nF  # 1st onF slot
            onF_[nF+1: nF+len(seg)+1].pop()  # remove nested oFs
            for i, F in enumerate(seg): F.nF = i  # nested indices
            seg_ += [T]
        else: seg[:] = []
    T_= []
    for T in Z.typ_:
        if not T: continue
        seg_, seg, _C = [],[],[]
        for C in T.call_:
            if isinstance(C, CoF):
                if isinstance(_C, list): _C = C; seg += [C]; continue  # init
                _t,t = Z.typ_[_C.nF], Z.typ_[C.nF]
                if comp_callers(_t,t) > ave: seg+= [C]
                else:  # init seg with consecutive externally missing call
                    seg_ += [seg_oF] if (seg_oF := seg_eval(seg)) else []
                    seg = [T]
        if seg:  # last
            seg_ += [seg_oF] if (seg_oF := seg_eval(seg)) else []
        T_ += seg_ or T  # eval segmented T?
        _C = C
    return CoF(typ_ = T_)  # new Z for new input, no other attrs yet?

def cluster_call_1():  # cluster T callees if called together, for global Z only?

    _T_ = []
    for t in Z.typ_:
        if t: t.grp = [t]; t.caller_ = set(t.caller_); _T_ += [t]; t.V = 0  # pairwise accum of membership value?
    V = 0; Olp,Off = set(),set()
    C = -Fc_[0]  # default cost of new representation, placeholder
    for _T,T in combinations(_T_,2):
        if _T.grp is T.grp: continue  # the pair was merged
        v,olp,off = comp_callers(_T,T)
        if v > ave:
            V += v; Olp.update(olp); Off.update(off)  # caller overlap and offset, not sure we need it
            _T.grp += T.grp; T.grp = _T.grp; T.V+=v; _T.V+=v  # concat new roots of externally matching typs
    grp_ = set(t.grp for t in _T_)
    for T_ in grp_:
        v,c = 0,-Fc_[0]
        for t in T_: v+= t.v; c+= t.fc  # external match and compression of element representation
    if V > ave - C:
        G_ = []
        for T in _T_:
            if T.grp in G_: continue
            grp = T.grp; gV = sum(t.v for t in grp); gC = sum(t.fc for t in grp) - 1  # if G.fc = 1
            if gV > ave- gC:  # caller overlap gain - (ave - grp_compression)
                G_ += [grp]
            else: G_ += grp; V-= gV; C-= gC  # revert grouping
        if V > ave - C:
            G_ = [sum2F(grp) if isinstance(grp,list) else grp for grp in G_]  # keep discrete Ts
            # form new Z for new input, vals added in CoF traced:
            return CoF(typ_=G_)  # or Z.w = V+ C*ave?

def cluster_calls():  # cluster Ts if called together, for global Z only?

    def comp_callers(_T, T):
        # compute value of callers_overlap + calls_overlap
        olp = _T.caller_ & T.caller_
        off = list(_T.caller_- olp) + list(T.caller_- olp)
        M = sum([c.w * c.c for c in olp])
        D = sum([c.w * c.c for c in off])
        return M / (D or eps)
        # match if same_callers / diff_callers > ave?
    _T_ = []
    global Z
    for t in Z.typ_:
        if t: t.grp = [t]; t.caller_ = set(t.caller_); t.V = 0; _T_ += [t]
    for _T,T in combinations(_T_,2):
        if _T.grp is T.grp: continue  # already merged
        v = comp_callers(_T,T)
        if v > ave:
            _g,g = _T.grp,T.grp
            for t in g: t.grp = _g  # repoint T.grp
            _g += g; _T.V += v; T.V += v  # pairwise accum of membership value
    G_= []
    V = 0; C = -Fc_[0]  # grp cost if grp.nF=comp_N_
    for t in _T_:
        if t.grp[0] is not t: continue  # eval each group once, at its seed
        grp = t.grp; gV = 0; gC = -Fc_[0]  # membership, compression / grp
        for t in grp: gV += t.V; gC += t.fc  # add links?
        gV /= 2  # norm for pairwise redundancy
        if gV > ave-gC: G_ += [[grp,gV,gC]]; V += gV; C += gC
        else:           G_ += grp
    if V > ave - C:
        typ_ = []  # form new Z.typ_ for new input, vals added in CoF traced:
        for grp in G_:
            if isinstance(grp,list):
                g,v,c = grp; T=sum2F(g); T.memb=v; T.cmpr=c  # downward reciprocals of V,C
                typ_ += [T]
            else: typ_ += [grp]  # keep old T
        Z = CoF(typ_=typ_); Z.memb, Z.cmpr = V,C  # membership and compression summed / Z?
        return Z

    def comp_body(_n, n):  # estimate AST-merge cost compression, non-recursive

        if isinstance(n,CoF):
            if isinstance(_n,CoF):
                if n.nF==_n.nF: return Fc_[n.nF]  # or F.fc? cost compression
                else: return -2  # fork cost = ast.IfExp, unpack in recursion
            else: return -2
        elif type(_n) is type(n): return costs.get(type(n))  # cost compression
        else: return -2

def add_typ_1(R):  # root oF, always Z?

    typ_ = [[] for _ in range(len(FTT_))]
    for F in flat_(R): typ_[F.nF] += [F] # bin runtime instances, T.call_ still trace tree
    for i, F_ in enumerate(typ_):
        if F_:
            T = sum2F(F_,CoF()); T.nF=i; T.wTT=cent_TT(getattr(T,'rTT',T.dTT), T.r)
            T.N_ = T.call_; root_ = []  # instances → N_
            T.caller_ = [F.root for F in F_]  # for comp_callers only?
            T.fc = Fc_[T.nF]  # get_ops(ast.parse(inspect.getsource(getattr(agg_recursion, oF_[i]))).body[0])  # body costs
            typ_[i] = T
    if any(typ_):
        add2F( R, sum2F([t for t in typ_ if t], CoF()))
    R.typ_ = typ_

def get_Fc_(paths): # compute weights based on operations in function, independent of deeper callees

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in oF_:
                funcs[node.name] = node

    return round(sum(costs.get(type(p),0) for name in oF_ for p in ast.walk(funcs[name])))

def get_nF_(paths):
    global oF_
    for path in paths:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in tree.body:  # only function definitions
            if isinstance(node, ast.FunctionDef):
                dict_oF_[node.name] = node

    return list(dict_oF_.keys())  # nF_: func names
'''
# func | block cost,gain,distribution, cost = oF_complexity / vt_complexity, ||oF_, *=data:
Fc_ = [13,11,18,5,4,22,20,12,3,13,14,11,3,5,4,3]; cN_,cC_,cN,cF, cE,ccN,ccC,ccP, cX,cFrm,cVct,cTrc,cBac,cPrj,cCS,cSE = Fc_
Fw_ = copy(Fc_); wN_,wC_,wN,wF, wE,wcN,wcC,wcP, wX,wFrm,wVct,wTrc,wBac,wPrj,wCS,wSE = Fw_  # ave gain/call, init = cost
FTT_= [deepcopy(wTT) for _ in range(16)]; ttN_,ttC_,ttN,ttF, ttE,ttcN,ttcC,ttcP, ttX,ttFrm,ttVct,ttTrc,ttBac,ttPrj,ttCs,ttSE = FTT_
ETT_ is local, add in oF_?
eval V = Ew - Ec * ave:
Ec_ = [3,3,4,2,1,5,5,4,1,4,4,4,2,2,1.1]  # complexity placeholders || oF_, same evals for different oFs?
Ew_ = copy(Ec_)  # ave gain/eval, init = cost, then evaled_block_w - default_block_w
'''
def add_typ_(R):

    def build_body(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in oF_:
            T = typ_[oF_.index(node.func.id)]
            if T: return T
        sub_ = [r for t in ast.iter_child_nodes(node) if (r := build_body(t)) is not None]
        if sub_: return (type(node), sub_)
        if type(node) in costs: return node

    def set_fc(body):
        fc = 0
        for n in body:
            if isinstance(n, CoF): fc += 3
            elif isinstance(n, tuple): fc += costs.get(n[0],0) + set_fc(n[1])
            else: fc += costs.get(type(n),0)
        return fc

    typ_ = [[] for _ in range(len(oF_))]
    for F in flat_(R): typ_[F.nF] += [F]
    for i, F_ in enumerate(typ_):
        if F_:
            T = sum2F(F_,CoF()); T.nF=i; T.wTT=cent_TT(getattr(T,'rTT',T.dTT), T.r)
            T.N_ = T.call_
            T.caller_ = [F.root for F in F_]
            typ_[i] = T
    for i, T in enumerate(typ_):  # second pass: static body refs other Ts
        if T:
            T.body = build_body(Fname_[oF_[i]])
            T.fc = set_fc(T.body)
    if any(typ_):
        add2F(R, sum2F([t for t in typ_ if t], CoF()))
    R.typ_ = typ_
