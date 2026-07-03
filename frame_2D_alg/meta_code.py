import numpy as np, inspect, contextvars, weakref
from contextlib import contextmanager
from functools import wraps
from copy import copy, deepcopy
import ast; from itertools import combinations
'''
code modification: compare aligned ops between oF_ AST sequences, cluster/split/merge matches into higher oF typs
'''
eps = 1e-7
def eps_(a): return np.where(a==0, eps, a)

ave,avd = .3,.5; decay = ave/(ave+avd)  # ave m,d / unit dist, recomputed from dTT*wTT?
wM,wD,wi, wG,wI,wa, wL,wS,wA = 10, 10, 20, 20, 5, 20, 2, 1, 1  # dTT weights = reversed relative ave, update from wTT_ after feedback
wT = np.array([wM,wD,wi, wG,wI,wa, wL,wS,wA])
wTT = np.array([wT,wT*avd])

def vt_(TT, wTT=wTT):  # base eval: multi-variate rel match, rel diff for membership

    m_,d_ = TT; ad_ = np.abs(d_); t_ = eps_(m_+ad_)  # ~ max comparand
    m = (m_/t_) @ wTT[0]; d = (ad_/t_) @ wTT[1]  # /= total = rdn, borrow in G.Bt only?
    return m, d

def sum_vt(N_, fr=0, fm=0, wTT=wTT, fdiv=1):  # basic weighted sum of CN|CF list

    C = sum(n.c for n in N_); R = 0; TT = np.zeros((2,9))
    for n in N_:
        w = n.c/C; TT += (n.rTT if fr else n.dTT)*w; R += n.r*w
    if fm:
        m,d = vt_(TT, wTT)
        if fdiv: m/= ave*R; d/= avd*R  # in 0-inf for summation
        else:    m-= ave*R; d-= avd*R  # in -1:1 without r
        return   m,d,TT,C,R
    else: return TT,C,R

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
        f.wTT = kw.get('wTT',wTT)  # mean wTT = 1?
        f.w = kw.get('w',0)  # membership weight
        f.typ = kw.get('typ',0)  # blocks sub_comp
        f.root = kw.get('root',None)  # convert to list in typ oFs?
    def __bool__(f): return bool(f.c)  # N_ may be empty?

class CL(CF):  # typ=1, add kern+positionals for base comp, Rt,Nt,Bt,Ct from comp_sub F2N
    name = "link"
    def __init__(l, **kw):
        super().__init__(**kw)
        l.kern = kw.get('kern',np.zeros(4))  # I,G,A diffs in links
        l.span = kw.get('span',1)  # distance in nodet or aRad, comp with kern or len(N_)
        l.angl = kw.get('angl',None)  # (dy,dx),dir, sum from L_, rarely?
        l.typ  = kw.get('typ',1)
        l.yx   = kw.get('yx', np.zeros(2))  # mean nodet? comp box is not meaningful?

class CC(CL):  # typ=2, adds arrays per N_
    name = "cent"
    def __init__(n, **kw):
        super().__init__(**kw)
        n.m_ = kw.get('m_',[])  # add _m_,_d_? also in C.N_, may conflict with promoted C m_,d_?
        n.d_ = kw.get('d_',[])
        n.typ = kw.get('typ',2)

def prop_F_(F, attr='N_'):  # factory function to get and update top-composition fork.N_|H
    def get(N): return getattr(getattr(N,F), attr)
    def set(N, new_val): setattr(getattr(N,F), attr, new_val)
    return property(get,set)

class CN(CL):  # full node | graph fork set
    name = "node"
    N_,C_,L_,B_,X_,rim,H = prop_F_('Nt'),prop_F_('Ct'),prop_F_('Lt'),prop_F_('Bt'),prop_F_('Xt'),prop_F_('Rt'),prop_F_('Nt','H')  # ext|int -defined Ns,Ls
    def __init__(n, **kw):
        n.Nt,n.Bt,n.Ct,n.Lt,n.Xt,n.Rt = ((kw.get(f) if f in kw else CF(root=n) for f in ('Nt','Bt','Ct','Lt','Xt','Rt')))  # CN if nest, Ct||Nt
        super().__init__(**kw)
        n.box = kw.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.mang= kw.get('mang',1) # ave match of angles in L_, =1 in links
        n.sub = kw.get('sub',0)  # composition depth relative to top-composition peers?
        n.exe = kw.get('exe',0)  # exemplar, temporary
        n.fin = kw.get('fin',0)  # clustered, temporary
        n.compared = kw.get('compared',set())
        n.root_ = kw.get('root_',[])  # reciprocal roots, Cs not Bs?
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
        f.fw,f.fc,f.fr = [kw.get(x,0) for x in ('fw','fc','fr')]  # fr if nested oF?
        f.caller_ = kw.get('caller_', set())
        f.g_ = kw.get('g_', [])  # callee gates
        f.gV_= kw.get('gv_',[])  # sum gating vals
        f.V_ = kw.get('vt_',[])  # sum(vt_(TT) * c/r
    @staticmethod
    def get(): return CoF._cur.get()  # Frm?
    @staticmethod
    def traced(func):
        if getattr(func, 'wrapped', False): return func
        @wraps(func)
        def inner(*a, **kw):
            _CoF = CoF._cur.get(None)
            oF = CoF(nF=iF_[func.__name__], root=_CoF)
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

    if v > 0: return v
    else: oF_[CoF.get().nF].gV_[i] -= v  # double -ve -> +ve

def build(func, node):  # AST → CoF | (type,sub_) | ast_leaf | None

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id=='gv_':
            func.gV_+=[0]; func.g_+=[node]; l=len(func.g_)  # func.g_[i] <-> oF.gV_[i]
            node.args.append(ast.Constant(value=l-1))  # add gv_(i)
            sub_ = [r for t in ast.iter_child_nodes(node) if (r := build(func,t)) is not None]
            sub_,fc_ = zip(*sub_) if sub_ else ((),())
            return (('gv_',l), sub_), sum(fc_)+costs.get(ast.Call,0)
        if (i := iF_.get(node.func.id)) is not None: return oF_[i],3
    sub_ = [rett for t in ast.iter_child_nodes(node) if (rett := build(func,t)) is not None]
    if sub_:
        sub_,fc_= zip(*sub_); return (type(node), sub_), sum(fc_)+costs.get(type(node), 0)
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

_names = ['frame_H','cross_comp','trace_edge',                         # root_, oF_[0] = frame_H, adds level per call
          'comp_N_','comp_C_','comp_N','comp_F',                       # comp_: incrementally distant, nested
          'get_exemplars','cluster_N','cluster_C','cluster_P','sum2G', # clus_: incrementally fuzzy, parallel
          'ffeedback','proj_N',                                        # fbac_: update filters) coords) funcs
          'vect_edge']                                                 # prep_
typ_= ['root_','root_','root_','comp_','comp_','comp_','comp_','clus_','clus_','clus_','clus_','clus_','fbac_','fbac_','prep_']
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

def comp_callers(_T, T):  # compute value of callers_overlap + calls_overlap

    olp = _T.caller_ & T.caller_
    off = list(_T.caller_- olp) + list(T.caller_- olp)
    M = sum([c.w * c.c for c in olp])
    D = sum([c.w * c.c for c in off])
    return M / (D or 1e-7) # match if same_callers / diff_callers > ave?
    # refine by comp results: -ve return may remove the callee?

def comp_prim(_n,n):
    if isinstance(_n,CoF) or isinstance(n,CoF):
        return comp_callers(_n,n) > ave if isinstance(_n,CoF) and isinstance(n,CoF) and n.nF==_n.nF else 0
    else:
        return _n[0]==n[0] if isinstance(_n,tuple) and isinstance(n,tuple) else 0 if isinstance(_n,tuple) or isinstance(n,tuple) else type(_n)==type(n)

def get_fc(n):
    return n.fc if isinstance(n,CoF) else costs.get(n[0],0)+sum(get_fc(c) for c in n[1]) if isinstance(n,tuple) else costs.get(type(n),0)

def split_oF_():  # divisive clustering

    def is_gate_l(t):  # check ast.type instead
        if isinstance(t,tuple) and not isinstance(t[0],tuple) and isinstance(t[1],tuple) and t[1]:
            h = t[1][0]
            if isinstance(h,tuple) and isinstance(h[0],tuple) and h[0][0]=='gv_': return h[0][1]
        return 0
    def is_fork(t):  # check type = ast.IfExp instead?
        return isinstance(t,tuple) and not isinstance(t[0],tuple) and t[0] in (ast.If, ast.IfExp) and t[1]

    def extract(t, oF, sF_):  # pack sub gate from gv or ifexp
        l = is_gate_l(t); fork = l or is_fork(t);oF.w=1
        if fork and oF.w * sum(get_fc(p) for p in t[1]) if isinstance(t,tuple) else 1 > ave:
            sub = CoF(root=oF, fc=get_fc(t), body=[t], caller_={oF})
            if not l: oF.gV_ += [0]; oF.g_ += [t]  # this is not correct yet, we need to identify the gv_ index of new ifexp gate
            sF_ += [sub]
            return sub
        if isinstance(t,tuple) and not isinstance(t[0],tuple) and isinstance(t[1],tuple):  # return body
            return (t[0], tuple(extract(s,oF,sF_) for s in t[1]))
        return t

    sF_,rF_ = [],[]
    for oF in oF_:
        oF.body = [extract(t,oF,sF_) for t in oF.body]
        if oF.body: rF_ += [oF]  # pack to rF_ if there's leftover unsplit body
    return sF_, rF_

def clust_oF_(oF_site_):  # cluster Ts if called together, global only

    # not revised:
    def emit_oF(nT):  # nT.body -> ast.FunctionDef, inverse of build; None if not emittable
        fd_ = [nF_[f.nF] for f in nT.N_ if isinstance(f.nF,int)]
        if len(fd_) < len(nT.N_): return  # member without fdef: split sub | unregistered composite
        a_ = [[a.arg for a in fd.args.args] for fd in fd_]
        if any(a != a_[0] for a in a_[1:]): return  # rewritten sites keep caller args: members must share signature
        stmts = []
        for t in nT.body:
            if isinstance(t,CoF) and isinstance(t.nF,int):
                stmts += [ast.Expr(ast.Call(func=ast.Name(nF_[t.nF].name,ctx=ast.Load()), args=[], keywords=[]))]  # arg binding in bind_oF_?
            else: return  # (typ,sub_) skeleton | (ast.IfExp,(Sub,sub)) fork: no runnable AST form, accounting-only
        if stmts:
            fdef = ast.parse(f"def {nT.name}(): pass").body[0]
            fdef.args = deepcopy(fd_[0].args); fdef.body = stmts
            return ast.fix_missing_locations(fdef)

    grp_ = {}   # group same-typ oFs?
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
    nT_,rF_ = [],[]  # final hard assign
    for i in range(Lc):
        t_ = [T for j,T in enumerate(oF_) if v__[j].argmax() == i]
        if t_:
            if (gV := v__[:,i].sum()) * ((len(t_)-1)*wL) > ave:
                nT = sum2O(t_)
                nT.memb = gV; fC = sum(t.fc for t in t_); nT.cmpr = fC - fC/len(t_)
                nT_ += [nT]
            else: rF_ += t_  # unpack if weak
    smF_ = []  # final smF
    for nT in nT_:  # create ast node with fdef
        fdef = emit_oF(nT)
        if fdef is None: continue
        name = f"oF{len(nF_)}"; fdef.name = name  # temporary named by length of nF_, for eg: oF11,oF12,oF13...
        nF_.append(fdef); iF_[name] = len(nF_)-1; oF_.append(nT); nT.nF = len(oF_)-1
        smF_ += [nT]
    for nT in smF_:
        for fork in nT.N_:
            for (caller_fd, site) in oF_site_.get(fork, []):   # point site to the updated merged oF
                site.func.id = nF_[nT.nF].name
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

def merge_oF(F,f, fsel=1):  # combine aligned ops, if-fork per miss, no inline recursion, f can only be added as a whole

    body,add_ = [],[]  # replace F.body: op sequence in func
    C = 0  # compression
    for i, (Sub,sub) in enumerate(zip(F.body,f.body)):
        if isinstance(Sub,CoF) and Sub.nF=='E':  # gate oF, none yet
            fin=0
            if isinstance(sub,CoF) and sub.nF=='E':
                subt = merge_oF(Sub, sub)  # add2F(Sub,ssub_)?
                if isinstance(subt, tuple):
                    sub = subt[1]; C+=sub.fc; body+=[sub]; add_+=[sub]
                    continue  # skip if merged:
            for _sub in Sub.body:
                if comp_prim(_sub,sub): fin=1; break  # no new fork?
            if not fin:
                if isinstance(sub, CoF): add2O(Sub,sub)
                C+=get_fc(sub); add_+=[sub]
            body += [Sub]
        else:  # default
            if comp_prim(Sub,sub): body += [Sub]  # add count?
            else: body += [(ast.IfExp,(Sub,sub))]; C += get_fc(sub); add_+=[sub]
            # new fork
    if len(f.body) > len(F.body): offs = f.body[i+1:]; body+=offs; add_+=offs
    elif len(F.body)>len(f.body): body+= F.body[i+1:]
    F.body = body
    if fsel and (f.fc / (C or 1e-7) > ave):  # high individual_cost / forking_Cost, else keep old F,f
        if add_: add2O(F, sum2O(add_))
        return F,f
    else: return F

def sum2O(F_, root=None, w_=None, fcall_=0):  # for fw,fc,fr only (dTT,r,w are from vt_)

    fc_ = np.array([n.fc for n in F_], dtype=float); fC = fc_.sum()
    if w_ is None:  w_ = fc_/fC; fw=0
    else:
        w_ = w_ / (w_.sum() or eps)
        for N, v in zip(F_,w_): N.fw = v
        fw = w_
    F = CoF(N_=F_, root=root, fc=fC, fw=fw, fr=F_[0].fr+1)  # unfinished, use w_ for N summing?
    if fcall_:
        c_ = np.array([n.c for n in F_], dtype=float); C = c_.sum(); w_ = c_/C; F.call_ = F_
    else:
        F.body = copy(F_[0].body)  # shallow copy
        for f in F_[1:]: merge_oF(F, f, fsel=0)
    if hasattr(F_[0],'caller_'): F.caller_ = set([caller for N in F_ for caller in N.caller_])  # for comp_caller between centroids
    return F   # fw = Fw

def add2O(F, n, nested=0):

    if F.fc:  # sum subtree gain, fixed cost, extensive: no weighting?
        C=F.fc+n.fc; _w,w = F.fc/C, n.fc/C; F.fr = F.fr*_w + n.fr*w; F.fc = C; F.fw += n.fw
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