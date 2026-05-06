# init code modification
import numpy as np
from itertools import combinations
from agg_recursion import (CoF, comp_F, cent_TT, vt_, Z, Fw_, Fc_, FTT_, Ew_, Ec_, ETT_, ave, avd, ttF, eps_, eps)

# draft by claude:
# pairwise similarity over function-type summaries:
def comp_typ_(typ_):
    L_ = []
    for _T,T in combinations(typ_, 2):
        dF = comp_F(_T,T)  # already gives m,d via comp_derT(dTT) (most of the oF.dTT is empty, and we only fill rTT?)
        L_ += [(_T,T,dF)]
    return L_

# similarity clusters: flood-fill on m > ave links
def cluster_O(typ_):
    L_ = comp_typ_(typ_); Clust_ = []; seen = set()
    for typ in typ_:
        if typ in seen: continue
        Clust = {typ}; front = [typ]
        while front:
            T = front.pop()
            for T1,T2,dF in L_:
                _T = T2 if T1 is T else (T1 if T2 is T else None)
                if _T and _T not in Clust and dF.m > ave*dF.r:
                    Clust.add(_T); front += [_T]
        seen |= Clust; Clust_ += [list(Clust)]
    return Clust_

# nF similarity → "alike", gain → "useful"
def gain_rate(T): return Ew_[T.nF] / (Ec_[T.nF] or eps)  # use only global?

# combined decision per cluster
def decisions(typ_):
    out = []
    typ_ = [t for t in typ_ if isinstance(t, CoF)]  # remove empty
    clust_ = cluster_O(typ_)
    for clust in clust_:
        ranked = sorted(clust, key=gain_rate, reverse=True)
        rep = ranked[0]; rest = ranked[1:]
        if len(rest):
            high = [t for t in rest if gain_rate(t) > ave]
            low  = [t for t in rest if gain_rate(t) <= ave]
            if high: out += [('merge_factor', rep.nF, [t.nF for t in high])]  # both pay off      (similar and high gain, merge them?)
            if low:  out += [('drop_dup', rep.nF, [t.nF for t in low])]       # rep does the work (similar but low gain, leave them?)
    # split: per-T multimodality (placeholder — needs per-call dTT, see below)
    # drop: per-T gain
    for T in typ_:
        if gain_rate(T) < ave: out += [('drop', T.nF)]  # shouldn't be split here? Drop means remove?
    return out

if __name__ == "__main__":
    import agg_recursion
    from agg_recursion import frame_H, imread, trace_func, add_typ_
    
    trace_func(vars(agg_recursion))
    Y,X = imread('./images/toucan.jpg').shape
    frame_H(image=imread('./images/toucan.jpg'), iY=Y//2-31, iX=X//2-31, Ly=64,Lx=64, Y=Y,X=X, rV=1)
    add_typ_(Z)
    for d in decisions(Z.typ_): print(d)
'''
Split detection needs per-call dTT, not just summary. Current add_typ_ does T = sum2F(F_, CoF()) which collapses N calls into one mean. 
To detect multimodal input distributions (the split signal), we need to keep the F_ list — either preserve T.N_ = F_ in add_typ_, 
or accumulate without collapsing. Then cent_TT over those individual dTTs reveals modes. Without this, "split" recommendations are guesses.

comp_F's sub-comp branch fires when _F.nF == F.nF. In meta_code, typ_ entries have distinct nF (one per function type), 
so sub-comp never fires — we get the bare dTT comparison only. That's what we want here, but worth knowing: 
if you ever feed two summaries of the same function (e.g. across runs or across sub-trees), comp_F goes deep. May or may not be desirable.

No connectivity over gFs yet. This sketch clusters oFs only. To get the dual mechanism (similarity-by-data + similarity-by-control-flow), 
add a parallel cluster_O pass over the gF-side of typ_ entries (T.gF if it exists on summary). 
Two cluster sets, possibly with conflicting groupings, that's OK — they answer different questions.

Position in the system: this naturally goes above agg_recursion, importing from it, no upward dependency. ffeedback's role would extend slightly: 
instead of just updating filters, it could call decisions() and surface them. 
# '''

'''
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


if __name__ == "__main__":
    fc_, base = init_wc(("agg_recursion.py", "comp_slice.py", "slice_edge.py"))
    print(f"Weights = {fc_}")

    fc_cross_comp = get_wc("agg_recursion.py", func="cross_comp")
    print(f"Weight of cross_comp = {fc_cross_comp}")

    fc_block = get_wc("agg_recursion.py", block=(225,243))
    print(f"Weights for line 225 to 243 = {fc_block}")
'''