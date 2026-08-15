import numpy as np, inspect, contextvars, weakref
from contextlib import contextmanager
from functools import wraps
from copy import copy, deepcopy
import ast; from itertools import combinations
eps = 1e-7
def eps_(a): return np.where(a==0, eps, a)
ave = decay = .3
avd = 20  # mean sum( abs(dTT[1]) * wTT[1]), the borrower
wM,wD,wi, wG,wI,wa, wL,wS,wA = 10, 10, 20, 20, 5, 20, 2, 1, 1  # dTT weights = reversed relative ave, update from wTT_ after feedback
wT = np.array([wM,wD,wi, wG,wI,wa, wL,wS,wA])
wTT = np.array([wT,wT*avd])
'''
code modification: compare aligned ops between oF_ AST sequences, cluster/split/merge matches into higher oF typs
'''
def val_(TT, wTT=wTT, fd=0):  # multi-variate rel match for membership, rel diff for boundary

    m_,d_ = TT; ad_ = np.abs(d_)
    vm = (m_/ eps_(m_+ad_)) @ wTT[0]  # /= total | max = rdn
    if fd:
        vd = (ad_@ wTT[1] / avd) * (vm/2)  # rat_dev_vd- proportional borrow from (vm + neutral trans-vm)/2?
        return vm, vd
    else: return vm

def sum_vt(N_, fr=0, fm=0, wTT=wTT, fdiv=1):  # basic weighted sum of CN|CF list

    C = sum(n.c for n in N_); R = 0; TT = np.zeros((2,9))
    for n in N_:
        w = n.c/C; TT += (n.rTT if fr else n.dTT)*w; R += n.r*w
    if fm:
        m,d = val_(TT, wTT,1)
        if fdiv: m/= ave*R; d/= avd*R  # in 0-inf for summation
        else:    m-= ave*R; d-= avd*R  # in -1:1 without r
        return   m,d,TT,C,R
    else: return TT,C,R

class CBase:
    refs = []
    def __init__(obj):
        obj._id = len(obj.refs)
        obj.refs.append(weakref.ref(obj))
    def __hash__(obj): return obj.id
    @property
    def id(obj): return obj._id
    @classmethod
    def get_instance(cls, _id):
        inst = cls.refs[_id]()
        if inst is not None and inst.id == _id:
            return inst
    def __repr__(obj): return f"{obj.__class__.__name__}(id={obj.id})"
    '''
    def __getattribute__(ave,name):
        coefs =   object.__getattribute__(ave, "coefs")
        if name == "coefs":
            return object.__getattribute__(ave, name)
        elif name == "md":
            return [ave.m * coefs["m"], ave.d *  coefs["d"]]  # get updated md
        else:
            return object.__getattribute__(ave, name)  * coefs[name]  # always return ave * coef
    '''
class CF(CBase):  # clustering fork: rim, Nt,Ct, Bt,Lt: ext|int- defined nodes, ext|int- defining links, Lt/Ft, Ct/lev, Bt/G
    name ="fork"  # add sub-forks by F2N
    def __init__(f, **kw):
        super().__init__()
        if not hasattr(f,'N_'): f.N_ = kw.get('N_',[])  # flat top lev, calls in oF, all sub-forks added conditionally
        if not hasattr(f,'L_'): f.L_ = kw.get('L_',[])  # +-Ls in levs or cLs in C
        f.H = kw.get('H',[])  # hierarchy = packed N_|L_s: lower CFs/ Nt||Ct, nestable, H[0]= redundant f.N_ if not empty
        f.nF = kw.get('nF','Nt')
        f.dTT = kw.get('dTT',np.zeros((2,9))); f.m, f.d, f.c, f.r = [kw.get(x,0) for x in ('m','d','c','r')]  # rdpTT in oF?
        f.wTT = kw.get('wTT',copy(wTT))  # mean wTT = 1?
        f.typ = kw.get('typ',0)  # blocks sub_comp
        f.root = kw.get('root',None)  # convert to list in typ oFs?
        f.root_ = kw.get('root_', [])  # [[C,m,d]...] per root centroid, add in cluster_C only?
    def __bool__(f): return bool(f.c)  # N_ may be empty?

class CL(CF):  # typ=1, add kern+positionals for base comp, Rt,Nt,Bt,Ct from comp_sub F2N
    name = "link"
    def __init__(l, **kw):
        super().__init__(**kw)
        l.kern = kw.get('kern',np.zeros(4))  # I,G,A diffs in links
        l.span = kw.get('span',1)  # distance in nodet or aRad, comp with kern or len(N_)
        l.angl = kw.get('angl',None)  # (dy,dx),dir, sum from L_, rarely?
        l.typ  = kw.get('typ',1)
        l.rim  = kw.get('rim',[])  # comp PPs and Ls
        l.yx   = kw.get('yx', np.zeros(2))  # mean nodet? comp box is not meaningful?

def prop_F_(F, attr='N_'):  # factory function to get and update top-composition fork.N_|H
    def get(N): return getattr(getattr(N,F), attr)
    def set(N, new_val): setattr(getattr(N,F), attr, new_val)
    return property(get,set)

class CN(CL):  # full node | graph fork set
    name = "node"
    N_,L_,B_,X_,rim,H = prop_F_('Nt'),prop_F_('Lt'),prop_F_('Bt'),prop_F_('Xt'),prop_F_('Rt'),prop_F_('Nt','H')  # ext|int -defined Ns,Ls
    def __init__(n, **kw):
        n.Nt,n.Bt,n.Lt,n.Xt,n.Rt = ((kw.get(f) if f in kw else CF(root=n) for f in ('Nt','Bt','Lt','Xt','Rt')))  # CN if nest
        super().__init__(**kw)
        n.mang = kw.get('mang',1) # ave match of angles in L_, =1 in links
        n.box = kw.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.sub = kw.get('sub',0)  # composition depth relative to top-composition peers?
        n.exe = kw.get('exe',0)  # exemplar, temporary
        n.typ = kw.get('typ',3)  # full comp
        # ftree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def __bool__(n): return bool(n.c)

class CoF(CF):
    name = "func"
    _cur = contextvars.ContextVar('oF')
    def __init__(f, **kw):
        super().__init__(**kw)
        f.call_ = kw.get('call_',[])  # called oFs only
        f.body = kw.get('body',[])  # static AST ops + CoF refs in source order
        f.fc,f.fr = [kw.get(x,0) for x in ('fc','fr')]  # fw: code compression, fc: costs, fr if nested oF?
        f.w = kw.get('w',0)  # membership weight
        f.caller_ = kw.get('caller_', set())
        f.g_ = kw.get('g_', [])  # callee gates
        f.gV_= kw.get('gv_',[])  # sum gating vals
        f.V_ = kw.get('val_',[])  # sum(val_(TT) * c/r
    @staticmethod
    def get(): return CoF._cur.get()  # Frm?
    @staticmethod
    def traced(func):
        if getattr(func, 'wrapped', False): return func
        @wraps(func)
        def inner(*a, **kw):
            _CoF = CoF._cur.get(None)
            oF = CoF(nF=iF_[func.__name__], root = _CoF)
            i = iF_[func.__name__]; oF_[i].call_ += [oF]  # F_call_T_[i][oF.nF] += oF.dTT
            if _CoF is not None:
                _CoF.call_ += [oF]
                oF_[iF_[func.__name__]].caller_.add(_CoF)  # for comp_caller_
            _oF = CoF._cur.set(oF)
            out = func(*a, **kw)
            CoF._cur.reset(_oF)
            return out
        inner.wrapped = True
        return inner
    def __bool__(f): return bool(f.call_)

def gv_(v, i=None):

    if v > 0: return v  # if +ve Fv triggers oF recursion, invert sign at the caller?
    else: oF_[CoF.get().nF].gV_[i] -= v  # ave Fv - neg v: +ve val of eval

def build(func, node):  # AST → CoF | (type,sub_) | ast_leaf | None

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id=='gv_':
            func.gV_+=[0]; func.g_+=[node]; l=len(func.g_)  # func.g_[i] <-> oF.gV_[i]
            if len(node.args)==1: node.args.append(ast.Constant(value=l-1))  # add gv_(i)
            else: node.args[-1] = ast.Constant(value=l-1)  # re-build: replace index, don't accumulate
            sub_ = [r for t in ast.iter_child_nodes(node) if (r := build(func,t)) is not None]
            sub_,fc_ = zip(*sub_) if sub_ else ((),())
            return (('gv_',l), sub_, node), sum(fc_)+costs.get(ast.Call,0)
        if (i := iF_.get(node.func.id)) is not None: return oF_[i],3
    sub_ = [rett for t in ast.iter_child_nodes(node) if (rett := build(func,t)) is not None]
    if sub_:
        sub_,fc_= zip(*sub_); return (type(node), sub_, node), sum(fc_)+costs.get(type(node), 0)
    if type(node) in costs: return node, costs.get(type(node),0)

def F_body_():  # form function body by AST tracing

    for func,name in zip(oF_,nF_):
        for node in ast.iter_child_nodes(name):  # skip top function definition
            rett = build(func, node)
            if rett:
                t, fc = rett; func.body += [t]; func.fc += fc

def parse_funcs(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in iF_:
                nF_[iF_.get(node.name)] = node

def clust_oF_():  # simplified oF rim-mediated centroid clustering

    global oF_;  F_ = [F for F in oF_ if F]  # uncalled Fs, don't cluster, pack in _F_ if the loop is not representative?
    for F in F_: F.rim = []; F.w = 0; F.root_ = []
    for _F, F in combinations(F_,2):  # w = relative compression: shared / min cost, ave-commensurate
        if (w := comp_body(_F.body, F.body) / min(_F.fc, F.fc)) > ave:
            _F.rim += [(F,w)]; F.rim += [(_F,w)]; _F.w += w; F.w += w
    FC_,_F_ = [],[]
    w_ = [sum([w * F.w / (F.w+_F.w) for _F, w in F.rim]) for F in F_]  # need to review
    for F,w in zip(F_,w_):
        F.w = w
        if F.w > ave:
            C = CoF(N_= [F]+[f for f,_ in F.rim], L_= [0 for _ in F_]); C.fin=0  # L_: dense prior w_, aligned with F_ in all Ts
            form_body(C); FC_ += [C]  # function clusters
    _w__ = [[0 for f in F_] for t in FC_]
    while True:  # refine Ts
        Dv = 0; w__ = [[0 for F in C.N_] for C in FC_]
        for i,C in enumerate(FC_):  # first pass to compute w and assign F's root_
            if C.fin: continue
            for j,F in enumerate(C.N_):  # rim-local candidates: members prune, never join
                w = (comp_body(C.body, F.body) / min(C.fc, F.fc) if F else 0)
                F.root_ += [C]; w__[i][j] = w
        for i,C in enumerate(FC_):  # second pass
            if C.fin: continue
            N_,w_ = [],[]
            for j,F in enumerate(C.N_):  # rim-local candidates: members prune, never join
                w = w__[i][j]
                if F.root_:
                    rw_ = [w__[FC_.index(r)][r.N_.index(F)] for r in F.root_ if r is not C]; sr_,wr_ = [.5],[]  # self
                    for rw in rw_:
                        if rw > w: sr_ = [rw]  # stronger or weaker alt root
                        else: wr_ += [rw]
                    r = np.mean(wr_)/np.mean(sr_)
                    F.r = r; C.r += r
                    Dv += abs(w -_w__[i][j]) - ave*F.r
                    w *= r; w__[i][j] = w
                if w>ave: N_ += [F]; w_ += [w]
                else:     F.root_.remove(C)
            C.N_ = N_; C.L_ = w_
            if w_: C.w = np.mean(w_)
            if sum(w_) > ave*C.r: form_body(C)  # rebuild from remaining members, refine
            else: C.fin = 1  # converged | weak, filtered below
        if Dv > 0:  _w__ = w__
        else: break
    _C_ = [C for  C in FC_ if C.w > ave]  # not only converged?
    _F_ = [F for F in oF_ if F not in (_F for C in _C_ for _F in C.N_)]; out_ = []  # recycle singletons
    for i,F in enumerate(_F_ + _C_):
        F.nF=i
        if F not in oF_: F.fdef = get_fdef(F, name=f'oF{i}')   # reinit the fdef: ast node
        out_ += [F]  # rename by index
    return out_

# from fable, not reviewed yet
def get_fdef(F, name=None):  # real FunctionDef from F.body; members define branch dispatch + arg spec

    def mk_args(names=()):  # full ast.arguments, compile-safe
        return ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in names], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])

    def body2ast(body, fi_map=None):  # meta body -> list of real ast stmts; fi_map: member CoF -> branch index
        def fork(t): return (isinstance(t,tuple) and t[0] is ast.IfExp and len(t)>1
                             and isinstance(t[1],tuple) and len(t[1])==2 and isinstance(t[1][0],list))  # form_body marker
        def stmt(n):
            n = deepcopy(n)
            return n if isinstance(n, ast.stmt) else ast.Expr(value=n)
        out = []
        for t in body:
            if isinstance(t, ast.AST): out += [stmt(t)]  # leaf: original stmt
            elif isinstance(t, CoF):   out += [stmt(ast.Expr(ast.Call(func=ast.Name(f'oF{t.nF}',ast.Load()),args=[],keywords=[])))]  # split-out sub, free vars unresolved
            elif fork(t):  # (ast.IfExp, ([f_],op)...) -> dispatch on fi selector, branch per member set
                node = None
                for k,(f_,op) in enumerate(t[1:]):
                    op_ = op if isinstance(op,list) else [op]
                    test = ast.Compare(left=ast.Name('fi',ast.Load()), ops=[ast.In()],
                                       comparators=[ast.Tuple(elts=[ast.Constant((fi_map or {}).get(f,k)) for f in f_], ctx=ast.Load())])
                    br = ast.If(test=test, body=body2ast(op_, fi_map) or [ast.Pass()], orelse=[])
                    if node is None: node = br; out += [node]
                    else: node.orelse = [br]; node = br
            elif isinstance(t,tuple) and len(t)>2 and isinstance(t[2],(ast.stmt,ast.expr)): out += [stmt(t[2])]  # packed node
            elif isinstance(t,tuple) and len(t)>2 and isinstance(t[2],ast.AST): pass  # arguments etc: cost-counted, not body ops
            else: out += [ast.Pass()]  # unrecoverable: pre-pack tree | synthetic without node
        return out

    mem_ = [f for f in F.N_ if isinstance(f,CoF)]
    fi_map = {f:k for k,f in enumerate(mem_)}
    args = deepcopy(nF_[mem_[0].nF].args) if mem_ and mem_[0].nF < len(nF_) and nF_[mem_[0].nF] is not None else mk_args()
    if mem_: args.kwonlyargs += [ast.arg(arg='fi')]; args.kw_defaults += [ast.Constant(0)]  # member selector
    fd = ast.FunctionDef(name=name or f'oF{F.nF}', args=args, body=body2ast(F.body, fi_map) or [ast.Pass()],
                         decorator_list=[], returns=None, type_comment=None)
    if 'type_params' in ast.FunctionDef._fields: fd.type_params = []  # py3.12+
    return ast.fix_missing_locations(fd)

def comp_body(_n, n):  # compare only: compression estimate C; construction in form_body

    if isinstance(_n,CoF) or isinstance(n,CoF):
        if _n is n:
            return get_fc(n)  # same ref: saved call site
        if isinstance(_n,CoF) and isinstance(n,CoF) and min(_n.fc,n.fc) > ave_C:
            return comp_body(_n.body, n.body) - 2  # cross-ref: body overlap - fork cost; also top-level pair entry
        return -2
    if isinstance(_n,list):  # op sequences: bodies | tuple children
        C = -2 * abs(len(_n)-len(n))
        for _t,t in zip(_n,n): C += comp_body(_t,t)
        return C
    if isinstance(_n,tuple) and isinstance(n,tuple) and _n[0] is n[0]:
        return costs.get(_n[0],0) + comp_body(list(_n[1]), list(n[1]))
    if type(_n) is type(n) and not isinstance(_n,tuple):
        return costs.get(type(n),0)  # leaf
    return -2

def form_body(F):

    def form_forks(bod_):
        # merge IfExp forks, bod is ([f], t)
        forks = [bod for bod in bod_ if isinstance(bod[1], tuple) and bod[1][0] is ast.IfExp]  # retrieve ifExp forks only
        if len(forks)>1:  # merge IfExp forks
            _fork = list(forks[0])  # _fork is [[f], t]
            _ifnode = list(_fork[1])  # _ifnode is [ast.ifExp, ([f],t),([f],t)...]
            for fork in forks[1:]:
                _fork[0] += fork[0]    # merge fs
                ifnode = list(fork[1])
                _ifnode += ifnode[1:]  # merge ([f],t) of subsequent forks into main _fork
                bod_.remove(fork)  # remove the merged fork
            _fork[1] = (ast.IfExp, *form_forks(_ifnode[1:]))  # recursively form_forks in t1,t2.. of ([f],t,t2...)
            bod_[bod_.index(forks[0])] = tuple(_fork)  # reassign the merged fork
        return bod_  # bod_ is [([f], t),([f], t),...]
    _body_ = [f.body for f in F.N_ if f]
    Bod = []
    for i in range(max(len(b) for b in _body_)):
        ibod_ = [([F.N_[j]], body[i]) for j, body in enumerate(_body_) if len(body) > i]  # i's index bodies and their F from all Fs
        if len(ibod_) > 1:
            form_forks(ibod_)  # ibod_ is [([f],t),([f],t)...]
            Bod += [(ast.IfExp, *ibod_)]  # in the format of (ast.IfExp, ([f],t)...)
        else:
            Bod += [ibod_[0][1]]  # single body, direct append and no additional fork (their f won't be preserved here?)
    F.fc = sum(get_fc(n) for n in Bod)
    F.body = Bod

def _form_body(F, fassign=1):  # render composite body: n-way positional columns, cluster ops per column

    def fork(t): return isinstance(t,tuple) and t[0] is ast.IfExp and isinstance(t[1][0],list)
    def form(col, Fn):  # col: [(op,F)] -> bare op | merged op | fork
        grp = {}
        for fk,F in col:
            k = fk if isinstance(fk,CoF) else id(fk[0]) if isinstance(fk,tuple) else type(fk)  # ref | head | leaf type
            grp.setdefault(k,[]).append((fk,F))
        branch_ = []
        for g in grp.values():
            t0 = g[0][0]; f_ = [F for _,F in g]
            if len(g)==1 or not isinstance(t0,tuple) or all(t is t0 for t,_ in g):
                op = t0  # single | ref,leaf | shared
            else:  # same-head: children columns
                op = (t0[0], tuple(form([(t[1][j],F) for t,F in g if len(t[1])>j], len(g)) for j in range(max(len(t[1]) for t,_ in g))))
            branch_ += [(f_,[op])]
        if len(branch_)==1 and len(branch_[0][0])==Fn: return branch_[0][1][0]  # universal: bare op
        return (ast.IfExp, *branch_)  # always form branch when there's at least 2 same length Fs
    Bod = []
    N_ = F.N_; bod_= [F.body for F in N_]
    for j in range(max(len(b) for b in bod_)):
        fk = form([(b[j],F) for b,F in zip(bod_,N_) if len(b)>j], len(N_))
        if fork(fk) and Bod and fork(Bod[-1]) and [b[0] for b in Bod[-1][1:]]==[b[0] for b in fk[1:]]:  # same func partition: concat
            Bod[-1] = (ast.IfExp, *[(f_, op_+_op_) for (f_,op_),(__,_op_) in zip(Bod[-1][1:], fk[1:])])
        else:
            Bod += [fk]
    fc = sum(get_fc(n) for n in Bod)
    cmpr = sum(f.fc for f in F.N_) - fc  # compression = original member - merged?
    if fassign:
        F.fc=fc; F.cmpr=cmpr; F.body = Bod
    else: return fc,cmpr,Bod

def comp_callers(_T, T):  # compute value of callers_overlap + calls_overlap

    olp = _T.caller_ & T.caller_
    off = list(_T.caller_- olp) + list(T.caller_- olp)
    M = sum([c.w * c.c for c in olp])
    D = sum([c.w * c.c for c in off])
    return M / (D or 1e-7) # match if same_callers / diff_callers > ave?
    # refine by comp results: -ve return may remove the callee?

def get_fc(n):
    return n.fc if isinstance(n,CoF) else costs.get(n[0],0)+sum(get_fc(c) for c in n[1]) if isinstance(n,tuple) else costs.get(type(n),0)

# divisive clustering:
def split_oF(t, oF):  # pack sub gate from gates
    if (isinstance(t,tuple) and t[0] in (ast.If,ast.IfExp) and isinstance(h:=t[1][0],tuple) and isinstance(h[0],tuple) and h[0][0]=='gv_'
        and oF.w * sum(get_fc(p) for p in t[1]) > ave):
        sub = CoF(root=oF, fc=get_fc(t), body=[t], caller_={oF})
        oF_.append(sub); nF_.append(None); sub.nF = len(oF_)-1
        return sub
    if isinstance(t,tuple) and t[1]:  # return body and split nested node recursively
        return (t[0], tuple(split_oF(s,oF) for s in t[1]),*t[2:])  # t[2] could be prior ast node
    return t

def inject_oF_(oF_, g):  # inject AST in g, recompile g[name]

    for oF in oF_:
        oF.fdef = nF_[oF.nF]; oF.g = g
        oF.body=[]; oF.fc=0; oF.g_=[]; oF.gV_=[]
        for node in ast.iter_child_nodes(oF.fdef):
            if r := build(oF,node): t,fc = r; oF.body += [t]; oF.fc += fc
        ast.fix_missing_locations(oF.fdef)
        exec( compile( ast.Module(body=[oF.fdef], type_ignores=[]), '<oF>', 'exec'), oF.g)
        oF.g[oF.fdef.name] = CoF.traced(oF.g[oF.fdef.name])

def sum2O(F_, root=None, w_=None, fcall_=0):  # for fw,fc,fr only (dTT,r,w are from val_)

    fc_ = np.array([n.fc for n in F_], dtype=float); fC = fc_.sum()
    if w_ is None:  w_ = fc_/fC; w=0
    else:
        w_ = w_ / (w_.sum() or eps)
        for N, v in zip(F_,w_): N.w = v
        w = w_
    F = CoF(N_=F_, root=root, fc=fC, w=w, fr=F_[0].fr+1)  # unfinished, use w_ for N summing?
    if fcall_:
        c_ = np.array([n.c for n in F_], dtype=float); C = c_.sum(); w_ = c_/C; F.call_ = F_
    else:
        F.body = copy(F_[0].body)  # shallow copy
        # for f in F_[1:]: merge_oF(F, f) use comp_body?
    if hasattr(F_[0],'caller_'): F.caller_ = set([caller for N in F_ for caller in N.caller_])  # for comp_caller between centroids
    return F   # fw = Fw

def add2O(F, n, nested=0):

    if F.fc:  # sum subtree gain, fixed cost, extensive: no weighting?
        C=F.fc+n.fc; _w,w = F.fc/C, n.fc/C; F.fr = F.fr*_w + n.fr*w; F.fc = C; F.w += n.w
    else:
        F.fc=n.fc; F.fr=n.fr; F.fw=n.fw
    if nested: F.call_ += [n]  # else add forks?
    return F

def trace_func(module_dict, module_name=None):

    if module_name is None: module_name = module_dict.get('__name__')
    for name, obj in list(module_dict.items()):
        if name in iF_.keys():
            if not inspect.isfunction(obj): continue
            if obj.__module__ != module_name: continue
            if getattr(obj, 'wrapped', False): continue
            module_dict[name] = CoF.traced(obj)

wcO, ccO = 5,5  # temporary
ave_C, wL = 3,3
costs = {  # types
    ast.Assign: 2,  # bind name: trivial
    ast.Attribute: 5,  # single dict lookup on object
    ast.UnaryOp: 2,  # apply one operator to one operand
    ast.BoolOp: 1,  # short-circuit decision between already-evaluated values
    ast.Compare: 2,  # compare already-evaluated operands
    ast.If: 1,  # test + pick branch, body ops counted separately
    ast.IfExp: 2,  # same as If
    ast.BinOp: 2,  # apply operator to two already-evaluated operands
    ast.AugAssign: 2,  # read + op + store, but op and target counted separately
    ast.Subscript: 5,  # index resolution into container
    ast.GeneratorExp: 3,  # lazy wrapper, inner loop body counted as child nodes
    ast.While: 1,  # condition re-evaluation overhead per iteration, body counted separately
    ast.For: 1,  # iterator protocol: __iter__ + __next__ overhead, body counted separately
    ast.ListComp: 1,  # same iteration overhead as For + list.append + allocation
    ast.SetComp: 2,  # same as ListComp + hashing per element
    ast.Call: 3,  # frame creation + arg binding + return: overhead beyond the callee body itself
}
_names = ['frame_H','cross_comp','trace_edge',  # root_, oF_[0] = frame_H, adds level per call
          'comp_N_','comp_N','comp_F',          # comp_: incrementally distant, nested
          'get_exemplars','cluster_N','cluster_C','cluster_P','sum2G', # clus_: incrementally fuzzy, parallel
          'ffeedback','proj_N',                                        # fbac_: update filters) coords) funcs
          'vect_edge']                                                 # prep_
typ_= ['root_','root_','root_','comp_','comp_','comp_','clus_','clus_','clus_','clus_','clus_','fbac_','fbac_','prep_']
nF_ = [None]*len(_names)  # FunctionDefs
iF_ = {n: i for i,n in enumerate(_names)}  # indices name → nF, static
oF_ = [CoF(nF=i,typ=typ) for i,typ in enumerate(typ_)]
parse_funcs(["agg_recursion.py"])  # populate nF_
# AST -> F.body:
for func,fdef in zip(oF_,nF_):
    for node in ast.iter_child_nodes(fdef):
        if r := build(func,node): t,fc = r; func.body += [t]; func.fc += fc
def call_sites(fd):  # FunctionDef
    return [n for n in ast.walk(fd) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in iF_]
F_call_T_ = [[np.zeros((2,9)) for _ in call_sites(fd)] for fd in nF_]  # dTT computed per callee
F_call_i_ = [{n.lineno: j for j,n in enumerate(call_sites(fd))} for fd in nF_]