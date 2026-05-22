import numpy as np, inspect, contextvars
from copy import copy, deepcopy
import ast; from itertools import combinations
import agg_recursion
from agg_recursion import (sum2F, add2F, CoF, Copy_, cent_TT, vt_, Z,wTT, ave, avd, eps, flat_, frame_H, imread)
'''
code modification: compare aligned ops between Z.typ_[i] AST sequences, cluster/merge matches into higher oF typs
'''
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

def cluster_calls():  # cluster Ts if called together, for global Z only?

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
        return M / (D or eps)  # match if same_callers / diff_callers > ave?

    _T_, pairs = [],[]
    for t in oF_:
        if t: t.grp = [t]; t.caller_ = set(t.caller_); t.V = 0; _T_ += [t]
    for _T,T in combinations(_T_,2):
        v,c = comp_callers(_T,T), comp_body(_T,T)
        pairs += [[v,c,_T,T]]

    # complete-linkage agglomeration on pairs, replace with centroids
    # not revised:
    Pairs = {frozenset({p[2], p[3]}) for p in pairs if p[0]+p[1] > ave}  # qualifying pairs
    Pairs.sort(key=lambda p: -p[0]-p[1])  # best v first
    for v, c, _T,T in pairs:
        if v <= ave: break
        if _T.grp is T.grp: continue
        if all(frozenset({a,b}) in Pairs for a in _T.grp for b in T.grp):
        # match = sum([1 for a in _T.grp for b in T.grp if frozenset({a, b}) in Pairs])
        # if match/len(T.grp)*len(T.grp) > 0.75:  # using all is too strict?
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
    # draft:
    G_= []
    V = C = 0  # recompute grp cost from AST?
    for t in _T_:
        if t.grp[0] is not t: continue  # eval each group once, at its seed
        grp = t.grp; gV = gC = 0  # membership, compression / grp
        for t in grp: gV += t.V; gC += t.fc  # add links?
        gV /= 2  # norm for pairwise redundancy
        if gV > ave-gC: G_ += [[grp,gV,gC]]; V += gV; C += gC
        else:           G_ += grp
    if V > ave - C:
        new_oF_ = []  # for new input, vals added in CoF traced:
        for grp in G_:
            if isinstance(grp,list):
                g,v,c = grp; T=sum2F(g); T.memb=v; T.cmpr=c  # downward reciprocals of V,C
                new_oF_ += [T]
            else: new_oF_ += [grp]  # keep old T
        Z = CoF(); Z.memb, Z.cmpr = V,C  # membership and compression summed / Z?
        return Z

_names = ['comp_N_','comp_C_','comp_N','comp_F',  # comp_ functions
          'get_exemplars','cluster_N','cluster_C','cluster_P',  # clust_ functions
          'cross_comp','frame_H','vect_edge','trace_edge','ffeedback','proj_N','comp_slice','slice_edge']  # combined, ancillary
          # + eval oFs?
oF_ = [CoF(nF=i) for i in range(len(_names))]
iF_ = {n: i for i,n in enumerate(_names)}  # indices name → nF, static
nF_ = [None] * len(_names)   # FunctionDef per nF

def trace_func(module_dict, module_name=None):

    if module_name is None: module_name = module_dict.get('__name__')
    for name, obj in list(module_dict.items()):
        if name in oF_:
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
                nF_[iF_[node.name]] = node

def add_typ_():
    for i, T in enumerate(oF_):
        if nF_[i] is not None:
            T.caller_ = []
            T.body = build_body(nF_[i])
            T.fc = set_fc(T.body)

def build_body(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        i = iF_.get(node.func.id)
        if i is not None and oF_[i].body: return oF_[i]
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

if __name__ == "__main__":
    # move to agg_recursion
    parse_funcs(["agg_recursion.py"])  # populate Fname_
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