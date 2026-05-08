import numpy as np
from itertools import combinations
from agg_recursion import (CoF, comp_F, cent_TT, vt_, Z, Fw_, Fc_, FTT_, Ew_, Ec_, ETT_, ave, avd, ttF, eps_, eps)
'''
code modification: compare aligned ops between Z.typ_[i] AST sequences, cluster/merge matches into higher oF typs
'''
def merge(_Q,Q, root=None):  # combine aligned ops, if-fork per miss, no inline recursion

    mrg = CoF(nF=Q.nF, root=root or Q.root)
    for _f,f in zip(_Q.call_, Q.call_):  # call_: op sequence in func or block
        if _f.nF==f.nF:
            mrg.call_ += [f.nF]  # match: copy the sequence
        else:  # miss: add gate to select forks
            gate = CoF(nF='E', call_=[_f,f], c=_f.c+f.c, root=mrg, fo=0)  # fc=Ec_[eval.nF]? root is not reassigned
            mrg.call_ += [gate]  # need to add eval func?
    mrg.c +=_Q.c+Q.c; mrg.fw +=_Q.fw+Q.fw; mrg.fc+= Fc_[Q.nF]  # included for final mrg eval, gates are transparent?
    return mrg

def eval(mrg):  # after adding multiple oFs in mrg.call_, not just the pair in single merge()

    fW = eW = 0
    call_ = []
    for F in mrg.call_:  # or typ_: nFs?
        if F.nF=='E': eW += F.w*F.c  # vs plain count
        else: call_+= [F]; fW += F.fc
    Z.typ_ += [mrg if fW/eW >ave else call_[:]]  # unpack mrg if high forking cost ratio

# not revised:
def split(Q, root=None):  # invert merge at most cost-unamortized gate

    g = max(Q.gF.call_, key=lambda g: max(b.fc*b.c - b.fw*b.c for b in g.call_))
    Qs = []
    for b in g.call_:
        S = CoF(nF=Q.nF, root=root or Q.root); S.call_=[b if f is g else f for f in Q.call_]
        S.gF.call_=[x for x in Q.gF.call_ if x is not g]
        S.c=b.c; S.fw=sum(f.fw for f in S.call_ if isinstance(f,CoF)); S.fc=Fc_[Q.nF]
        Qs += [S]
    return Qs

if __name__ == "__main__":

    import agg_recursion
    from agg_recursion import frame_H, imread, trace_func, add_typ_

    trace_func(vars(agg_recursion))  # add oF tracing
    Y,X = imread('./images/toucan.jpg').shape
    frame_H(image=imread('./images/toucan.jpg'), iY=Y//2-31, iX=X//2-31, Ly=64,Lx=64, Y=Y,X=X, rV=1)
    add_typ_(Z)  # each call_ in Z.typ_ is the flatten calls of same typ
    mrg_, spl_ = [],[]
    # add fffeedback to reform Z for next frame_H
    for F,_F in combinations(Z.typ_,2): mrg_ += [merge(F,_F)]
    for F in Z.typ_: spl_ += [split(F)]

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