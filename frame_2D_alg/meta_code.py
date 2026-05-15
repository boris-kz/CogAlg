import numpy as np
from itertools import combinations
from agg_recursion import (sum2F, add2F, CoF, Copy_, cent_TT, vt_, Z, Fw_, Fc_, FTT_,
                           Ew_, Ec_, ETT_, ave, avd, eps, onF_, flat_)
'''
code modification: compare aligned ops between Z.typ_[i] AST sequences, cluster/merge matches into higher oF typs
'''
def merge(F,f):  # combine aligned ops, if-fork per miss, no inline recursion, f can only be added as a whole

    call_, add_ = [], []  # replace F.call_ if eval
    C = 0; cost = Ec_[0]  # total and fixed-fork costs
    for i, (Sub,sub) in enumerate(zip(F.call_,f.call_)):  # call_: op sequence in func or block, init F per f?
        fork = []
        if Sub.nF=='E':  # previously added gate
            fork = Sub; fin=0
            if sub.nF=='E':
                sub = merge(Sub, sub)  # add2F(Sub,ssub_)?
                if sub: C+=sub.fc; call_+=[sub]; add_+=[sub]  # merge failed
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

def comp_callers(_T, T):

    caller_, _caller_ = set(T.root), set(_T.root)
    olp = _caller_ & caller_
    off = list(_caller_-olp) + list(caller_-olp)
    M = sum([c.w * c.c for c in olp])
    D = sum([c.w * c.c for c in off])
    return M/(D or eps)  # match if same_callers / diff_callers > ave?

def cluster_call_(Z):  # cluster T callees if called together?

    def seg_eval(seg, seg_):
        if sum(c.fc for c in seg) > ave * Ec_[0]:  # base eval cost
            T = sum2F(seg); T.nF = [call.nF for call in T.call_]  # new_nF = old_nF_?
            seg_ += [T]
    T_= []
    for T in Z.typ_:
        if not T: continue
        seg_, seg, _C = [],[],[]
        for C in T.call_:
            if isinstance(C, CoF):
                if not _C: _C = C; seg += [C]; continue
                _t, t = Z.typ_[_C.nF], Z.typ_[C.nF]
                if comp_callers(_t,t)>ave: seg+= [C]
                else: seg_eval(seg, seg_); seg = [C]
                _C = C
            else: seg += [C]
        if seg: seg_eval(seg, seg_)  # last seg, add eval?
        if seg_: T_ += seg_  # segmented T, add eval?
        else:    T_ += [T]   # recycled T
    return CoF(typ_ = T_)  # new Z for new input, no other attrs yet?

# draft, may not be needed
def cluster_AST(Q):  # initial wrap all primitives with oFs for Z.call_

    seg_, C = [],[]
    for c in Q:
        if C:
            if isinstance(c, CoF): seg_ += [C]; C = c
            else: C.call_ += [c]
        elif isinstance(c, CoF): C=c
        else: C = CoF(call_=[c])
    if C: seg_ += [C]
    return seg_

if __name__ == "__main__":

    import agg_recursion
    from agg_recursion import frame_H, imread, trace_func, add_typ_

    trace_func(vars(agg_recursion))  # add oF tracing
    Y,X = imread('./images/toucan.jpg').shape
    frame_H(image=imread('./images/toucan.jpg'), iY=Y//2-31, iX=X//2-31, Ly=64,Lx=64, Y=Y,X=X, rV=1)
    add_typ_(Z)  # each call_ in Z.typ_ is the flatten calls of same typ
    # add fffeedback to reform Z for next frame_H:
    Z = cluster_call_(Z)  # new Z, before or after merge?
    typ_, mrg_ = [],[]
    for i, t in enumerate(Z.typ_):  # vs. combinations(Z.typ_,2)?
        if t in mrg_: continue
        F = Copy_(t, cls=CoF)
        for f in Z.typ_[i+1:]: mrg_ += [merge(F,f)]
        typ_ += [F]

import ast
onF_ = ['comp_N_','comp_C_','comp_N','comp_F',  # comp_ functions
        'get_exemplars','cluster_N','cluster_C','cluster_P',  # clust_ functions
        'cross_comp','frame_H','vect_edge','trace_edge','ffeedback','proj_N','comp_slice','slice_edge']  # combined,ancillary

weights = {
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

def get_ops(node):
    return sum(weights.get(type(n), 0) for n in ast.walk(node))

def init_wc(paths): # compute weights based on operations in function, independent of deeper callees
    # onF_+ vt_ names and function objects:
    funcs = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and (node.name in onF_ or node.name == 'vt_'):
                funcs[node.name] = node

    base = get_ops(funcs.pop('vt_'))
    return [round(get_ops(funcs[name]) / base) for name in onF_], base

def get_wc(path, func=None, block=None, base=None):

    # get function
    def get_func_node(tree, name):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node

    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src, filename=path)
    if not base: base = get_ops(get_func_node(tree, 'vt_'))
    if block:
        start, end = block  # inclusive line numbers
        ops_count = 0
        for n in ast.walk(tree): # walk the tree and sum weights for nodes that fall within the line range
            if hasattr(n, 'lineno'):
                if start <= n.lineno <= end:  # make sure node's line number falls within the block
                    ops_count += weights.get(type(n), 0)
        return round(ops_count / base)
    elif func:
        node = get_func_node(tree, func)
    return round(get_ops(node) / base)

TYP_SKEL = None

def build_typ_skel(paths=("agg_recursion.py","comp_slice.py","slice_edge.py")):
    nodes = {}
    for p in paths:
        with open(p,encoding="utf-8") as f: tree = ast.parse(f.read(),filename=p)
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name in onF_: nodes[n.name] = n
    call_seq = [[] for _ in onF_]
    for name, fn in nodes.items():
        i = onF_.index(name)
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                c = getattr(n.func,'id',None) or getattr(n.func,'attr',None)
                if c in onF_: call_seq[i] += [onF_.index(c)]
    callers = [[] for _ in onF_]
    for i, seq in enumerate(call_seq):
        for c in seq:
            if i not in callers[c]: callers[c] += [i]
    return call_seq, callers

def add_typ_(oF):  # moved from agg_recursion

    global TYP_SKEL
    if TYP_SKEL is None: TYP_SKEL = build_typ_skel()
    call_seq, callers = TYP_SKEL
    typ_ = [[] for _ in range(len(FTT_))]
    for F in flat_(oF): typ_[F.nF] += [F]
    for i, F_ in enumerate(typ_):
        if F_:
            T = sum2F(F_,CoF()); T.nF=i; T.wTT=cent_TT(getattr(T,'rTT',T.dTT),T.r)
            T.N_ = T.call_; typ_[i] = T  # instances → N_
    for i, T in enumerate(typ_):  # wire AST structure after all T's exist
        if T:
            T.call_ = [typ_[k] for k in call_seq[i] if typ_[k]]
            T.root  = [typ_[k] for k in callers[i]  if typ_[k]]
    oF.typ_ = typ_
    if any(typ_): add2F(oF, sum2F([t for t in typ_ if t], CoF()))

if __name__ == "__main__":
    fc_, base = init_wc(("agg_recursion.py", "comp_slice.py", "slice_edge.py"))
    print(f"Weights = {fc_}")

    fc_cross_comp = get_wc("agg_recursion.py", func="cross_comp")
    print(f"Weight of cross_comp = {fc_cross_comp}")

    fc_block = get_wc("agg_recursion.py", block=(225,243))
    print(f"Weights for line 225 to 243 = {fc_block}")