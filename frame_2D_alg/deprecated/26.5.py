
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
