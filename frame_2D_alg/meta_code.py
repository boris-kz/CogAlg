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

ave = decay = .3; avd = 20  # mean sum( abs(dTT[1]) * wTT[1]), the borrower
wM,wD,wi, wG,wI,wa, wL,wS,wA = 10, 10, 20, 20, 5, 20, 2, 1, 1  # dTT weights = reversed relative ave, update from wTT_ after feedback
wT = np.array([wM,wD,wi, wG,wI,wa, wL,wS,wA])
wTT = np.array([wT,wT*avd])

def val_(TT, wTT=wTT, fd=0):  # multi-variate rel match for membership, rel diff for boundary

    m_,d_ = TT; ad_ = np.abs(d_)
    vm = (m_/ eps_(m_+ad_)) @ wTT[0]  # /= total | max = rdn
    if fd:
        vd = (ad_@ wTT[1] / avd) * (vm/2)  # rat_dev_vd- proportional borrow from (vm + neutral trans-vm)/2?
        return vm,vd
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
        f.fc,f.fr = [kw.get(x,0) for x in ('fc','fr')]  # fw: code compression, fc: costs, fr if nested oF?
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

def comp_body_claude(_n, n, _m_, m):  # compare = merge: realized compression C + merged form, pure
    # _n: composite side, may hold 'E' blocks | '|' forks; n: member side, raw ops + registered refs

    def alt(m_,t_): return CoF(nF='A', N_=m_, body=t_, fc=sum(get_fc(t) for t in t_))  # member-subset block
    def fork(a_):   return CoF(nF='|', N_=a_, fc=2 + sum(a.fc for a in a_))  # alts in N_; 2: ast.IfExp

    def seq(_q, q):  # aligned op sequences
        C = 0; Q = []
        for c,t in [comp_body(_t,t,_m_,m) for _t,t in zip(_q,q)]: C += c; Q += [t]
        for _t in _q[len(q):]: Q += [_t if isinstance(_t,CoF) and _t.nF=='|' else fork([alt(copy(_m_),[_t])])]; C -= 2  # m skips
        for t in q[len(_q):]:  Q += [fork([alt([m],[t])])]; C -= 2  # priors skip
        return C, Q

    if isinstance(n,list): return seq(_n, n)  # top bodies
    if isinstance(_n,CoF) and _n.nF=='|':  # fork: merge n into best alt | append alt
        r_ = [comp_body(a.body, [n], a.N_, m) for a in _n.N_]  # nested forks tag in alt-member terms
        i = int(np.argmax([c for c,_ in r_])); c, body = r_[i]
        a_ = copy(_n.N_)
        if c > 0: a_[i] = alt(_n.N_[i].N_+[m], body); return c, fork(a_)
        else:     return -2, fork(a_ + [alt([m],[n])])
    if isinstance(n,CoF) or isinstance(_n,CoF):
        if n is _n: return get_fc(n), _n  # same registered ref | shared block: match = saved call site
        if isinstance(n,CoF) and isinstance(_n,CoF) and isinstance(n.nF,str) and isinstance(_n.nF,str) and n.fc > ave_C:
            C, body = seq(_n.body, n.body)  # inline blocks: recursive body merge
            if C > 0: return C-2, CoF(nF='E', N_=copy(_m_)+[m], body=body, fc=sum(get_fc(t) for t in body),
                                      caller_=_n.caller_|n.caller_)
        return -2, fork([alt(copy(_m_),[_n]), alt([m],[n])])  # diff refs | mixed: fork here, merge at def level
    if isinstance(_n,tuple) and isinstance(n,tuple) and n[0] is _n[0]:
        C, sub_ = seq(_n[1], n[1]); return C + costs.get(n[0],0), (n[0], tuple(sub_))  # deep partial merge
    if type(_n) is type(n) and not isinstance(n,tuple): return costs.get(type(n),0), _n  # leaf
    return -2, fork([alt(copy(_m_),[_n]), alt([m],[n])])  # fork per miss


def comp_body_claude2(_n, n, _m_=None, m=None):  # compare == merge: compression C + merged form (pure, replaces merge_oF)
    
    def gate(f_, sub):             return (ast.IfExp, (f_, sub))              # 1-branch: these members do op, others skip
    def new_fork(_f_,_sub, m,sub): return (ast.IfExp, (_f_,_sub), ([m],sub))  # 2-branch fork from one divergent op pair
    
    def add_fork(F, n, _m_, m):  # route member op n into existing fork F: join its best branch, else append a branch
        branch_ = list(F[1:])  # [(m_, op), ...]
        r_ = [comp_body(op, n, bm_, m) for bm_,op in branch_]
        i  = int(np.argmax([c for c,_ in r_])); c, op = r_[i]
        if c > 0: branch_[i] = (branch_[i][0]+[m], op)  # n merges into branch i
        else:     branch_ += [([m], n)]; c = -2         # n becomes a new branch
        return c, (ast.IfExp, *branch_)
    
    def merge_seq(_q, q, _m_, m):  # merge aligned op sequences (bodies | node children) -> C, merged_
        C, Q = 0, []
        for _t, t in zip(_q, q):
            c, mt = comp_body(_t, t, _m_, m); C += c; Q += [mt]
        for _t in _q[len(q):]:  Q += [_t if isinstance(_t,tuple) and _t[0] is ast.IfExp else gate(_m_, _t)]; C -= 2  # tail: new member m skips these
        for  t in  q[len(_q):]: Q += [gate([m], t)]; C -= 2  # tail: prior members skip these
        return C, Q
    
    if _m_ is None: return merge_seq(_n.body, n.body, [_n], n)  # top level
    if isinstance(_n,tuple) and _n[0] is ast.IfExp and isinstance(_n[1][0],list):  # existing fork
        return add_fork(_n, n, _m_, m)
    if _n is n: return get_fc(n), _n  # same ref, skip
    
    if isinstance(_n,tuple) and isinstance(n,tuple) and _n[0] is n[0]:  # same ast node, merge children
        C, sub_ = merge_seq(_n[1], n[1], _m_, m); return C + costs.get(_n[0],0), (_n[0], tuple(sub_))
    if type(_n) is type(n) and not isinstance(_n,(tuple,CoF)):  # other than tuple and CoF
        return costs.get(type(n),0), _n
    return -2, new_fork(_m_, _n, m, n)  # miss, create fork


def comp_callers(_T, T):  # compute value of callers_overlap + calls_overlap

    olp = _T.caller_ & T.caller_
    off = list(_T.caller_- olp) + list(T.caller_- olp)
    M = sum([c.w * c.c for c in olp])
    D = sum([c.w * c.c for c in off])
    return M / (D or 1e-7) # match if same_callers / diff_callers > ave?
    # refine by comp results: -ve return may remove the callee?

def get_fc(n):
    if isinstance(n,CoF): return n.fc if isinstance(n.nF,str) else costs.get(ast.Call,0)  # inline 'E'|'|' block | registered ref: body counted at def
    if isinstance(n,tuple): return costs.get(n[0],0) + sum(get_fc(c) for c in n[1])
    return costs.get(type(n),0)

def split_oF_():  # divisive clustering

    def is_gate_l(t):  # check ast.type instead
        if isinstance(t,tuple) and t[0] in (ast.If,ast.IfExp) and isinstance(h:=t[1][0],tuple) and isinstance(h[0],tuple) and h[0][0] == 'gv_': return 1
        return 0

    def split(t, oF):  # pack sub gate from gates
        if is_gate_l(t) and oF.w * sum(get_fc(p) for p in t[1]) > ave:  # w is membership weight, it should be fw?
            sub = CoF(root=oF, fc=get_fc(t), body=[t], caller_={oF})
            oF_.append(sub); nF_.append(None); sub.nF = len(oF_)-1
            return sub
        if isinstance(t,tuple) and t[1]:  # return body and split nested node recursively
            return (t[0], tuple(split(s,oF) for s in t[1]))
        return t

    for oF in copy(oF_):  # copy because we append new sub during
        oF.body = [split(t,oF) for t in oF.body]
''' 
with average_linkage:
for _T,T in combinations(oF_,2): V = (comp_callers(_T,T) + comp_body(_T,T)) * min(_T.fc,T.fc); _T.V_[T] = V; T.V_[_T] = V 
for t in C.N_: if t is not T: v__[j,i] += t.V_[T]  # += pairwise Vs 
'''
def clust_oF_():  # exemplar-seeded merge for body compression

    F_ = copy(oF_); T_ = []
    for F in F_: F.fin = 0; F.rim = []
    for _F,F in combinations(F_,2):
        if _F.typ == F.typ:
            w,_ = comp_body(_F,F)
            # w *= min(_F.fc,F.fc)  # this is not needed?
            if w > ave:
                _F.w += w; F.w += w  # projected compression: above-ave w only
                _F.rim += [(F,w)]; F.rim += [(_F,w)]
                
    for _F in sorted(F_, key=lambda F: F.w, reverse=True):
        if _F.fin: continue
        if _F.w <= ave: break  # that _F.w*wL is not relevant here?
        T = CoF(N_=[_F], typ=_F.typ, body=_F.body, fc=_F.fc)
        for F,w in sorted(_F.rim, key=lambda t: t[1], reverse=True):
            if F.fin: continue
            C, body = comp_body(T.body,F.body, T.N_,F)
            if C>ave: T.body = body; T.w += C; T.N_ += [F]; F.fin = 1
        if len(T.N_) > 1:
            T.fc = sum(get_fc(t) for t in T.body); T.cmpr = sum(f.fc for f in T.N_) - T.fc
            if T.cmpr > ave: _F.fin = 1; T_ += [T]
            else:
                for F in T.N_[1:]: F.fin = 0
    for T in T_: oF_.append(T); nF_.append(None); T.nF = len(oF_)-1

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