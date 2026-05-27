import numpy as np, inspect, contextvars, weakref
from contextlib import contextmanager
from functools import wraps
from copy import copy, deepcopy
import ast; from itertools import combinations
'''
code modification: compare aligned ops between oF_ AST sequences, cluster/merge matches into higher oF typs
'''
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
        f.nF = kw.get('nF','Nt')
        f.dTT = kw.get('dTT',np.zeros((2,9))); f.m, f.d, f.c, f.r = [kw.get(x,0) for x in ('m','d','c','r')]  # rdpTT in oF?
        f.wTT = kw.get('wTT',wTT)  # mean wTT = 1?
        f.w = kw.get('w',0)  # membership weight
        f.fb_ = kw.get('fb_',[])
        f.typ = kw.get('typ',0)  # blocks sub_comp
        f.root = kw.get('root',None)  # convert to list in typ oFs?
    def __bool__(f): return bool(f.c)  # N_ may be empty?

class CL(CF):  # typ=1, add kern+positionals for base comp, Rt,Nt,Bt,Ct from comp_sub F2N
    name = "link"
    def __init__(l, **kw):
        super().__init__(**kw)
        l.nt = kw.get('nt',[])  # nodet
        l.yx = kw.get('yx', np.zeros(2))  # mean nodet? comp box is not meaningful?
        l.kern = kw.get('kern',np.zeros(4))  # I,G,A diffs in links
        l.span = kw.get('span',1)  # distance in nodet or aRad, comp with kern or len(N_)
        l.angl = kw.get('angl',None)  # (dy,dx),dir, sum from L_, rarely?
        l.typ  = kw.get('typ',1)

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
        n.H = kw.get('H',[])  # hierarchy: lower CF levs/ Nt||Ct
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

# process selection:
def flat_(oF, call_=None):  # all nested call_ s

    if call_ is None: call_ = []
    for sub in oF.call_:
        call_ += [sub]
        if sub.call_: flat_(sub,call_)
    return call_

class CoF(CF):
    name = "func"
    _cur = contextvars.ContextVar('oF')
    def __init__(f, **kw):
        super().__init__(**kw)
        f.call_ = kw.get('call_',[])  # called oFs only
        f.body = kw.get('body',[])  # static AST ops + CoF refs in source order
        f.fw,f.fc,f.fr = [kw.get(x,0) for x in ('fw','fc','fr')]  # fr if nested oF?
    @staticmethod
    def get(): return CoF._cur.get()  # Frm?
    @staticmethod
    def traced(func):
        if getattr(func, 'wrapped', False): return func
        @wraps(func)
        def inner(*a, **kw):
            _CoF = CoF._cur.get(None)
            oF = CoF(nF=iF_[func.__name__], root=_CoF)
            oF_[iF_[func.__name__]].call_ += [oF]
            if _CoF is not None:
                _CoF.call_ += [oF]
                oF_[iF_[func.__name__]].caller_.add(_CoF)  # caller_ for comp_caller_
            _oF = CoF._cur.set(oF)
            out = func(*a, **kw)
            if oF.call_:
                tree = flat_(oF)  # if len(tree)-1?
                sum2O(tree,oF,fdata=1); wtt = getattr(oF,'rTT',oF.dTT); oF.wTT = cent_TT(wtt,oF.r)
            CoF._cur.reset(_oF)
            return out
        inner.wrapped = True
        return inner
    def __bool__(f): return bool(f.call_)

eps = 1e-7
def eps_(a): return np.where(a==0, eps, a)

ave,avd = .3,.5; decay = ave/(ave+avd)  # ave m,d / unit dist, recomputed from dTT*wTT?
wM,wD,wi, wG,wI,wa, wL,wS,wA = 10, 10, 20, 20, 5, 20, 2, 1, 1  # dTT weights = reversed relative ave, update from wTT_ after feedback
wT = np.array([wM,wD,wi, wG,wI,wa, wL,wS,wA])
wTT = np.array([wT,wT*avd])
wcO, ccO = 5,5  # temporary
ave_C = 3
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
_names = ['frame_H','cross_comp','trace_edge',                 # root_, oF_[0] = frame_H, adds level per call
          'comp_N_','comp_C_','comp_N','comp_F',               # comp_
          'get_exemplars','cluster_N','cluster_C','cluster_P', # clus_
          'ffeedback','proj_N',                                # proj_
          'vect_edge','comp_slice','slice_edge']               # prep_
          # deep eval Fs?
typ_= ['root_','root_','root_','comp_','comp_','comp_','comp_','clus_','clus_','clus_','clus_','proj_','proj_','prep_','prep_','prep_']
# need to add oF-specific body:
oF_ = [CoF(nF=i,typ=typ) for i,typ in enumerate(typ_)]
iF_ = {n: i for i,n in enumerate(_names)}  # indices name → nF, static
nF_ = [None] * len(_names)  # FunctionDef s

# old:
def merge(F,f):  # combine aligned ops, if-fork per miss, no inline recursion, f can only be added as a whole

    call_, add_ = [], []  # replace F.call_ if eval
    C = 0; cost = f.fc  # total and fixed-fork costs
    for i, (Sub,sub) in enumerate(zip(F.body,f.body)):  # call_: op sequence in func or block, init F per f?
        fork = []
        # add comp primitives
        if Sub.nF=='E':  # previously added gate, use gF_
            fork = Sub; fin=0
            if sub.nF=='E':
                subt = merge(Sub, sub)  # add2F(Sub,ssub_)?
                if isinstance(subt, tuple):
                    sub = subt[1]; C+=sub.fc; call_+=[sub]; add_+=[sub]
                    continue  # skip if merged:
            for _sub in Sub.call_:
                if _sub.nF==sub.nF: fin=1; break  # no new fork cost?
            if not fin:
                add2O(Sub,sub); C+=sub.fc; add_+=[sub]
        elif Sub.nF != sub.nF:
            fork = CoF(nF='E',call_=[Sub,sub]); C += cost; add_+=[sub]
        call_ += [fork or Sub]
    if len(f.call_) > len(F.call_): offs = f.call_[i+1:]; call_+=offs; add_+=offs
    elif len(F.call_)>len(f.call_): call_+=F.call_[i+1:]

    if f.fc / (C or 1e-7) > ave:  # high individual_cost / forking_Cost, else keep old F,f
        if add_: add2O(F, sum2O(add_))
        F.call_= call_
        return F,f
    else: return F

def cluster_oF_():  # cluster Ts if called together, global only

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
        elif type(_n) is type(n): C += costs.get(type(n),0)  # AST leaf
        else: C -= 2
        return C

    def comp_callers(_T, T):
        # compute value of callers_overlap + calls_overlap
        olp = _T.caller_ & T.caller_
        off = list(_T.caller_- olp) + list(T.caller_- olp)
        M = sum([c.w * c.c for c in olp])
        D = sum([c.w * c.c for c in off])
        return M / (D or 1e-7)  # match if same_callers / diff_callers > ave?

    for t in oF_: t.caller_ = set(t.caller_); t.V_ = {}  # pairwise V to every other oF
    for _T,T in combinations(oF_,2):
        V = (comp_callers(_T,T) + comp_body(_T,T)) * min(_T.fc, T.fc)
        _T.V_[T] = V
        T.V_[_T] = V
    grp_= []; _T = oF_[0]; grp=[_T]
    for T in oF_[1:]:
        if T.typ==_T.typ and _T.V_[T] > ave: grp+=[T]  # include _T.V_[T]? Else the pairwise T.V_ section above is actually redundant
        else: grp_+=[grp]; grp=[T]
        _T=T
    grp_ += [grp]
    C_ = [sum2O(g) for g in grp_]  # initial composites
    Ln,Lc = len(oF_),len(C_); L = Ln*Lc  # -1?
    _v__ = np.zeros((Ln, Lc))
    # cluster_P analog:
    while True:
        v__ = np.zeros((Ln, Lc))
        for j,T in enumerate(oF_):  # refine by comp T x G_
            for i,C in enumerate(C_):
                v__[j,i] = (comp_callers(C,T) + comp_body(C,T)) * min(C.fc, T.fc)
        C_ = [sum2O(oF_, w_=v__[:,i]) for i in range(Lc)]  # weighted re-aggregation
        V, dV = v__.sum(), np.abs(v__-_v__).sum()
        if V*dV*(wcO*L) <= ave * (Ln+ ccO*L): break  # convergence
        _v__ = v__
    new_oF_ = []  # final hard assign
    for i in range(Lc):
        t_ = [T for j,T in enumerate(oF_) if v__[j].argmax() == i]
        if t_:
            if (gV := v__[:,i].sum()) * ((len(t_)-1)*wL) > ave:
                nT = sum2O(t_); nT.memb = gV; nT.cmpr = sum(t.fc for t in t_)  # compression should be just sum of fc?
                new_oF_ += [nT]
            else: new_oF_ += t_  # unpack if weak
    return new_oF_


def split_oF_():  # divisive clustering
        
    def comp_prim(_P,P): 
        match = 0
        if isinstance(_P, CoF):
            if isinstance(P, CoF):
                match = P.typ == _P.typ  # match by type
        elif isinstance(_P, tuple):
            if isinstance(P, tuple): 
                match = _P[0] == P[0]  # match by sub type
        elif not isinstance(P, CoF) and not isinstance(P, tuple):  # _P is primitive node
            match = type(_P) == type(P)  # match by node type
        
        return match

    def get_fc(P):  # get fc of single prim, similar with build
        if isinstance(P, CoF): return P.fc
        if isinstance(P, tuple):
            return costs.get(P[0], 0) + sum(get_fc(c) for c in P[1])
        return costs.get(P, 0)

    out = []
    for oF in oF_:
        if len(oF.body)>1:  # skip if less than 2 primitives
            grp_= []; _P = oF.body[0]; grp=[_P]
            for P in oF.body[1:]:
                if comp_prim(_P, P): grp+=[P]
                else:                grp_+=[grp]; grp=[P]
                _P=P
            grp_ += [grp]
            
            # no refinement iterations yet
            for grp in grp_:
                fc = sum([get_fc(prim) for prim in grp])
                sub = CoF(root=oF, fc=fc,  body=grp)  # not sure on fr, fw, caller_ and call_
                out += [sub]    
        else:
            out += [oF]
    oF_[:] = out


def trace_func(module_dict, module_name=None):

    if module_name is None: module_name = module_dict.get('__name__')
    for name, obj in list(module_dict.items()):
        if name in iF_.keys():
            if not inspect.isfunction(obj): continue
            if obj.__module__ != module_name: continue
            if getattr(obj, 'wrapped', False): continue
            module_dict[name] = CoF.traced(obj)

def parse_funcs(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in iF_:
                nF_[iF_.get(node.name)] = node

def F_body_():  # get function bodies from their AST

    def build(node):  # AST → CoF | (type,sub_) | ast_leaf | None

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if (i:= iF_.get(node.func.id)) is not None and oF_[i].body:  # wrong bracket, else i = boolean
                return oF_[i], 3
        sub_ = [rett for t in ast.iter_child_nodes(node) if (rett:= build(t)) is not None]
        if sub_:
            sub_, fc_ = zip(*sub_); return (type(node), sub_), sum(fc_)+costs.get(type(node), 0)
        if type(node) in costs:
            return node, costs.get(type(node), 0)

    for i, F in enumerate(oF_):
        F.caller_ = set()
        for node in ast.iter_child_nodes(nF_[i]):  # skip top function definition
            rett = build(node)
            if rett:
                t, fc = rett; F.body += [t]; F.fc += fc

def sum2O(N_, root=None, w_=None, fdata=0):  # for w,c,r, fw,fc,fr only?

    if fdata:
        c_ = np.array([n.c for n in N_], dtype=float); C = c_.sum(); w_ = c_/C; body = []
    else:
        body = [prim for N in N_ for prim in N.body]  # add body for centroid (not sure but splice oFs' body?)
    fc_ = np.array([n.fc for n in N_], dtype=float); fC = fc_.sum()
    if w_ is None:  w_ = fc_/fC; fw=0  # N = N_[0]
    else:           
        w_ = w_ / (w_.sum() or eps)
        for N, v in zip(N_,w_): N.fw = v
        fw = w_
    # unfinished, use w_ for N summing?
    oF = CoF(call_=N_, root=root, fc=fC, fw=fw, fr=N_[0].fr+1, body=body)
    # for centroids
    if hasattr(N_[0], 'caller_'): oF.caller_ = set([caller for N in N_ for caller in N.caller_])  # for comp_caller between centroids
    return oF   # fw = Fw

def add2O(F, n, nested=0):

    if F.fc:  # sum subtree gain, fixed cost, extensive: no weighting?
        C=F.fc+n.fc; _w,w = F.fc/C, n.fc/C; F.fr = F.fr*_w + n.fr*w; F.fc = C; F.fw += n.fw
    else:
        F.fc=n.fc; F.fr=n.fr; F.fw=n.fw
    if nested: F.call_ += [n]  # else add forks?
    return F

def cent_TT(dTT, r):  # EM-like weight attr matches | diffs by their match to the sum, recompute to convergence

    wTT,_wTT = [],np.ones((2,9)); coT = np.abs(dTT[0])+np.abs(dTT[1])  # complemented vals
    while True:
        for fd, _wT, dT in zip((0,1), _wTT, dTT):
            vT = np.abs(dT)  # if -m: wrong, or surprise value?
            rvT = vT / eps_(coT) * _wT  # weighted normalized vals
            mean = rvT.mean() or eps  # scalar
            invdev_ = np.minimum(rvT / mean, mean / eps_(rvT))
            wT = invdev_ / (invdev_.mean() or eps)   # mean(wT)=1
            wTT += [wT]
        if np.sum(np.abs(wTT-_wTT)) < ave * r:  # if np.linalg.norm(wT - _wT, 1) < r?
            break
        _wTT = np.array(wTT); wTT = []
    return _wTT  # single-mode dTT, extend to 2D-3D lev cycles in H, cross-level param max / centroid?