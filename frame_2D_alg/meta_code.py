import numpy as np, inspect, contextvars
import ast; from itertools import combinations, zip_longest
import agg_recursion
from agg_recursion import (sum2F, add2F, CoF, Copy_, cent_TT, vt_, Z, Fw_, Fc_, FTT_, wTT, Ew_, Ec_, ETT_, ave, avd, eps, flat_, frame_H, imread)
'''
code modification: compare aligned ops between Z.typ_[i] AST sequences, cluster/merge matches into higher oF typs
'''
costs = {
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

def cluster_calls():  # cluster Ts if called together, for global Z only?

    def comp_body(_n, n):  # project AST-merge cost, recursive, no commit
        '''
        # draft
        if type(_n) is not type(n): return get_ops(_n) + get_ops(n)
        C = 0
        _ch, ch = list(ast.iter_child_nodes(_n)), list(ast.iter_child_nodes(n))
        for a, b in zip(_ch, ch): C += comp_body(a, b)
        for x in _ch[len(ch):] + ch[len(_ch):]: C += get_ops(x)
        return C
        '''
        
        def get_node_c(node):  # cost of a single AST node, including onF_
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):      name = node.func.id
                if isinstance(node.func, ast.Attribute): name = node.func.attr
                if name in oF_: return get_ops(ast.parse(inspect.getsource(getattr(agg_recursion, oF_[oF_.index(name)]))).body[0])
            else: return costs.get(type(node), 0)
       
        # get ast of each typ
        if isinstance(_n, CoF):  # first call
            _ast_n, ast_n = get_ast(_n).body, get_ast(n).body
        else:  # recursive node
            _ast_n, ast_n = ast.iter_child_nodes(_n), ast.iter_child_nodes(n)
        
        #  Not sure on this, return native cost when both calls have different nF? I think we still can comp if it's different?
        # if _n.nF != n.nF: return get_ops(_ast_n) + get_ops(ast_n)
        
        C = 0
        for _node, node in zip_longest(_ast_n, ast_n):
            if _node is None:                   C += get_node_c(node)   # n is longer
            elif node is None:                  C += get_node_c(_node)  # _n is longer          
            elif type(_node) is not type(node): C += get_node_c(_node) + get_node_c(node)  # both nodes are different
            # else:                               C += comp_body(_node, node)   # same type, run the proces recursively (not sure)
        return C

    def comp_callers(_T, T):
        # compute value of callers_overlap + calls_overlap (add comp_body)
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
        if _T.grp is T.grp: continue  # already merged
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

oF_ = ['comp_N_','comp_C_','comp_N','comp_F',  # comp_ functions
       'get_exemplars','cluster_N','cluster_C','cluster_P',  # clust_ functions
       'cross_comp','frame_H','vect_edge','trace_edge','ffeedback','proj_N','comp_slice','slice_edge']  # combined,ancillary

def get_ops(node):
    return sum(costs.get(type(n), 0) for n in ast.walk(node))

def get_ast(T):  # get ast primitives of T, and T.nF must be in onF_
    return ast.parse(inspect.getsource(getattr(agg_recursion, oF_[T.nF]))).body[0]

def init_wc(paths): # compute weights based on operations in function, independent of deeper callees
    # oF_+ vt_ names and function objects:
    funcs = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and (node.name in oF_ or node.name == 'vt_'):
                funcs[node.name] = node

    base = get_ops(funcs.pop('vt_'))
    return [round(get_ops(funcs[name]) / base) for name in oF_], base


def add_typ_(R):  # root oF, always Z?

    typ_ = [[] for _ in range(len(FTT_))]
    for F in flat_(R): typ_[F.nF] += [F] # bin runtime instances, T.call_ still trace tree
    for i, F_ in enumerate(typ_):
        if F_:
            T = sum2F(F_,CoF()); T.nF=i; T.wTT=cent_TT(getattr(T,'rTT',T.dTT), T.r)
            T.N_ = T.call_; root_ = []  # instances → N_
            T.caller_ = [F.root for F in F_]  # for comp_callers only?
            T.fc = get_ops(ast.parse(inspect.getsource(getattr(agg_recursion, oF_[i]))).body[0])  # body costs
            typ_[i] = T
    if any(typ_):
        add2F( R, sum2F([t for t in typ_ if t], CoF()))
    R.typ_ = typ_

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

    fc_, base = init_wc(("agg_recursion.py", "comp_slice.py", "slice_edge.py"))
    print(f"Weights = {fc_}")