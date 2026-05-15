
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


