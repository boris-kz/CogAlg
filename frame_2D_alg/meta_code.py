import numpy as np, inspect, contextvars
from copy import copy, deepcopy
import ast; from itertools import combinations
import agg_recursion
from agg_recursion import (sum2F, add2F, CoF, Copy_, cent_TT, vt_, Z,wTT, ave, avd, eps, flat_, frame_H, imread)
'''
code modification: compare aligned ops between Z.typ_[i] AST sequences, cluster/merge matches into higher oF typs
'''
oF_ = ['comp_N_','comp_C_','comp_N','comp_F',  # comp_ functions
       'get_exemplars','cluster_N','cluster_C','cluster_P',  # clust_ functions
       'cross_comp','frame_H','vect_edge','trace_edge','ffeedback','proj_N','comp_slice','slice_edge']  # combined, ancillary
gF_ = []  # add gating oFs, was nF='E'
# func | block cost,gain,distribution, cost = oF_complexity / vt_complexity, ||oF_, *=data:
Fc_ = [13,11,18,5,4,22,20,12,3,13,14,11,3,5,4,3]; cN_,cC_,cN,cF, cE,ccN,ccC,ccP, cX,cFrm,cVct,cTrc,cBac,cPrj,cCS,cSE = Fc_
Fw_ = copy(Fc_); wN_,wC_,wN,wF, wE,wcN,wcC,wcP, wX,wFrm,wVct,wTrc,wBac,wPrj,wCS,wSE = Fw_  # ave gain/call, init = cost
FTT_= [deepcopy(wTT) for _ in range(16)]; ttN_,ttC_,ttN,ttF, ttE,ttcN,ttcC,ttcP, ttX,ttFrm,ttVct,ttTrc,ttBac,ttPrj,ttCs,ttSE = FTT_
# eval V = Ew - Ec * ave:
Ec_ = [3,3,4,2,1,5,5,4,1,4,4,4,2,2,1.1]  # complexity placeholders || oF_, same evals for different oFs?
Ew_ = copy(Ec_)  # ave gain/eval, init = cost, then evaled_block_w - default_block_w
ETT_= [deepcopy(wTT) for _ in range(16)]  # for more precise eeval?
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
def merge(F,f):  # combine aligned ops, if-fork per miss, no inline recursion, f can only be added as a whole

    call_, add_ = [], []  # replace F.call_ if eval
    C = 0; cost = Ec_[0]  # total and fixed-fork costs
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
                add2F(Sub,sub, fo=1); C+=sub.fc; add_+=[sub]
        elif Sub.nF != sub.nF:
            fork = CoF(nF='E',call_=[Sub,sub]); C += cost; add_+=[sub]
        call_ += [fork or Sub]
    if len(f.call_) > len(F.call_): offs = f.call_[i+1:]; call_+=offs; add_+=offs
    elif len(F.call_)>len(f.call_): call_+=F.call_[i+1:]

    if f.fc / (C or eps) > ave:  # high individual_cost / forking_Cost, else keep old F,f
        if add_: add2F(F, sum2F(add_), fo=1)
        F.call_= call_
        return F,f
    else: return F

# work in progress
def cluster_calls():  # cluster Ts if called together, for global Z only?

    def comp_body(_n, n, C=4):  # estimate AST-merge cost compression, init mean C=3, accum from recursive unpack
        # C0 = 4; Cmin = 3; fork = 2
        if isinstance(n, CoF):
            if isinstance(_n, CoF):
                if n.nF==_n.nF: C+= Fc_[n.nF]
                elif C > 3:  # mean compression value
                    for _sub,sub in zip(_n.body,n.body):  # eval bodies before merging?
                        C = comp_body(_sub, sub, C)
                    if len(_n.body) != len(n.body): C-=2  # single tail fork cost
                else: C-=2  # fork compression cost, ast.IfExp
            else: C-=2
        elif type(_n) is type(n): C+= costs.get(type(n),0)
        else: C-=2
        return C

    def comp_body_claude(_n, n, C=0):  # compression of n given _n: cost of n absorbed by merge
        if isinstance(n, CoF) and isinstance(_n, CoF):
            if n is _n: C += Fc_[n.nF]  # full reuse, save whole body
            elif n.fc > ave_C:  # mean compression value
                for i, sub in enumerate(n.body):
                    if i < len(_n.body): C = comp_body(_n.body[i], sub, C)
                    else: C -= 2  # n has tail past _n
            else: C -= 2
        elif type(_n) is type(n): C += costs.get(type(n),0)
        else: C -= 2
        return C

    def comp_callers(_T, T):
        # compute value of callers_overlap + calls_overlap
        olp = _T.caller_ & T.caller_
        off = list(_T.caller_- olp) + list(T.caller_- olp)
        M = sum([c.w * c.c for c in olp])
        D = sum([c.w * c.c for c in off])
        return M / (D or eps)
        # match if same_callers / diff_callers > ave?
    global Z
    _T_, pairs = [],[]
    for t in Z.typ_:
        if t: t.grp = [t]; t.caller_ = set(t.caller_); t.V = 0; _T_ += [t]
    for _T,T in combinations(_T_,2):
        v,c = comp_callers(_T,T), comp_body(_T,T)
        pairs += [[v,c,_T,T]]

    # draft: complete-linkage agglomeration on pairs
    qual = {frozenset({p[2], p[3]}) for p in pairs if p[0] > ave}  # qualifying pairs
    pairs.sort(key=lambda p: -p[0])  # best v first
    for v, c, _T, T in pairs:
        if v <= ave: break
        if _T.grp is T.grp: continue
        if all(frozenset({a, b}) in qual for a in _T.grp for b in T.grp):
            _g, g = _T.grp, T.grp
            for t in g: t.grp = _g
            _g += g
    ''' Claude:
    Qualifying criterion is just v > ave; if you want c to gate qualification too, change the filter (e.g. p[0] > ave and p[1] < some_bound).
    Sort key is v only — c could be folded in (-(p[0] - p[1]/k)) for joint ordering.
    The complete-linkage check is O(|_g|·|g|) per candidate merge — unavoidable for the criterion; the all-pairs qual lookup keeps it constant per check.
    if _T.grp is T.grp: continue in the first loop fires never (no merges happen during pair collection); harmless but vestigial. 
    Same for t.V = 0 now that the pair loop doesn't accumulate t.V.
    '''
    # not revised:
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

def trace_func(module_dict, module_name=None):

    if module_name is None: module_name = module_dict.get('__name__')
    for name, obj in list(module_dict.items()):
        if name in oF_:
            if not inspect.isfunction(obj): continue
            if obj.__module__ != module_name: continue
            if getattr(obj, 'wrapped', False): continue
            module_dict[name] = CoF.traced(obj)

def ffeedback(frame):  # adjust filters: all aves *= rV, ultimately differential backprop per ave?

    rTT = np.divide(frame.H[0].wTT, frame.H[1].wTT)  # wTT_ is not relevant now
    _wTT = frame.H[1].wTT
    for lev in frame.H[2:]:  # sum ratios between consecutive-level TTs, top-down in frame H, not lev-selective or sub-lev recursive
        rTT += np.divide(_wTT,lev.wTT)
        _wTT = lev.wTT
    rM = rD = 0
    rm, rd = vt_(rTT,wTT)
    return rM+rD, rTT  # add rm,rd?

# old: start with flat oFs
def get_Fc_(paths): # compute weights based on operations in function, independent of deeper callees

    funcs = {}  # oF_ names and function objects
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and (node.name in oF_ or node.name == 'vt_'):
                funcs[node.name] = node

    return round(sum(costs.get(type(p),0) for name in oF_ for p in ast.walk(funcs[name])))
# or:
funcs = {}  # name → FunctionDef, populated by parse_funcs(paths)

def parse_funcs(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in oF_:
                funcs[node.name] = node

def add_typ_(R):
    typ_ = [[] for _ in range(len(FTT_))]
    for F in flat_(R): typ_[F.nF] += [F]
    for i, F_ in enumerate(typ_):
        if F_:
            T = sum2F(F_,CoF()); T.nF=i; T.wTT=cent_TT(getattr(T,'rTT',T.dTT), T.r)
            T.N_ = T.call_
            T.caller_ = [F.root for F in F_]
            T.fc = Fc_[T.nF]
            typ_[i] = T
    for i, T in enumerate(typ_):  # second pass: static body refs other Ts
        if T: T.body = build_body(funcs[oF_[i]], typ_)
    if any(typ_):
        add2F(R, sum2F([t for t in typ_ if t], CoF()))
    R.typ_ = typ_

# draft
def build_body(node, typ_):
    out = []
    children = node.body if isinstance(node, ast.FunctionDef) else ast.iter_child_nodes(node)
    for child in children:
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in oF_:
            T = typ_[oF_.index(child.func.id)]
            if T: out += [T]
            continue
        if type(child) in costs: out += [child]
        out += build_body(child, typ_)
    return out

if __name__ == "__main__":
    # move to agg_recursion
    trace_func(vars(agg_recursion))  # add oF tracing
    Y,X = imread('./images/toucan.jpg').shape
    frame_H(image=imread('./images/toucan.jpg'), iY=Y//2-31, iX=X//2-31, Ly=64,Lx=64, Y=Y,X=X, rV=1)
    add_typ_(Z)
    # add fffeedback to reform Z for next frame_H
    Z = cluster_calls()  # new Z, before or after merge?
    typ_, mrg_ = [],[]
    for i, t in enumerate(Z.typ_):  # vs. combinations(Z.typ_,2)?
        if t in mrg_: continue
        F = Copy_(t, cls=CoF)
        for f in Z.typ_[i+1:]: mrg_ += [merge(F,f)]
        typ_ += [F]
    for i, F in typ_: F.nF = i
    Z.typ_ = typ_