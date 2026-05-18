import numpy as np, inspect, contextvars
import ast; from itertools import combinations
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
    for i, (Sub,sub) in enumerate(zip(F.call_,f.call_)):  # call_: op sequence in func or block, init F per f?
        fork = []
        if Sub.nF=='E':  # previously added gate
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

def comp_callers(_T, T):

    caller_, _caller_ = set(T.caller_), set(_T.caller_)
    olp = _caller_ & caller_
    off = list(_caller_-olp) + list(caller_-olp)
    M = sum([c.w * c.c for c in olp])
    D = sum([c.w * c.c for c in off])
    return M/(D or eps)  # match if same_callers / diff_callers > ave?

def cluster_call_():  # cluster T callees if called together? (Z is global and can be retrieved within the func anyway)

    def seg_eval(seg, seg_):
        if sum(c.fc for c in seg) > ave * Ec_[0]:  # base eval cost
            T = sum2F(seg); T.nF = [call.nF for call in T.call_]  # new_nF = old_nF_? (only need list when there's different typ in call_?)
            seg_ += [T]
    T_= []
    for T in Z.typ_:
        if not T: continue
        seg_, seg, _C = [],[],[]
        for C in T.call_:
            if isinstance(C, CoF):
                if isinstance(_C, list):  # init
                    _C = C; seg += [C]; continue
                _t, t = Z.typ_[_C.nF], Z.typ_[C.nF]
                if comp_callers(_t,t)>ave: seg+= [C]
                else: seg_eval(seg, seg_); seg = [C]
                _C = C
            else: seg += [C]
        if seg: seg_eval(seg, seg_)  # last seg, add eval?
        if seg_: T_ += seg_  # segmented T, add eval?
        else:    T_ += [T]   # recycled T (seg is always filled, so T_ may not be relevant?)
    return CoF(typ_ = T_)  # new Z for new input, no other attrs yet?

def trace_func(module_dict, module_name=None):
    if module_name is None: module_name = module_dict.get('__name__')
    for name, obj in list(module_dict.items()):
        if name in onF_:
            if not inspect.isfunction(obj): continue
            if obj.__module__ != module_name: continue
            if getattr(obj, 'wrapped', False): continue
            module_dict[name] = CoF.traced(obj)

def ffeedback(frame):  # adjust filters: all aves *= rV, ultimately differential backprop per ave?

    rTT = np.divide(frame.H[0].wTT, frame.H[1].wTT)  # wTT_ is not relevant now
    _wTT = frame.H[1].wTT
    for lev in frame.H[2:]:  # sum ratios between consecutive-level TTs, top-down frame expansion levels, not lev-selective or sub-lev recursive
        rTT += np.divide(_wTT,lev.wTT)
        _wTT = lev.wTT
    rM = rD = 0
    rm, rd = vt_(rTT,wTT)
    return rM+rD, rTT  # add rm,rd?

onF_ = ['comp_N_','comp_C_','comp_N','comp_F',  # comp_ functions
        'get_exemplars','cluster_N','cluster_C','cluster_P',  # clust_ functions
        'cross_comp','frame_H','vect_edge','trace_edge','ffeedback','proj_N','comp_slice','slice_edge']  # combined,ancillary

def get_ops(node):
    return sum(costs.get(type(n), 0) for n in ast.walk(node))

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


def add_typ_(R):  # root oF, always Z?

    typ_ = [[] for _ in range(len(FTT_))]
    for F in flat_(R): typ_[F.nF] += [F] # bin runtime instances, T.call_ still trace tree
    for i, F_ in enumerate(typ_):
        if F_:
            T = sum2F(F_,CoF()); T.nF=i; T.wTT=cent_TT(getattr(T,'rTT',T.dTT), T.r)
            T.N_ = T.call_; root_ = []  # instances → N_
            T.caller_ = [F.root for F in F_]  # for comp_callers only?
            T.fc = get_ops(ast.parse(inspect.getsource(getattr(agg_recursion, onF_[i]))).body[0])
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
    Z = cluster_call_()  # new Z, before or after merge?
    typ_, mrg_ = [],[]
    for i, t in enumerate(Z.typ_):  # vs. combinations(Z.typ_,2)?
        if t in mrg_: continue
        F = Copy_(t, cls=CoF)
        for f in Z.typ_[i+1:]: mrg_ += [merge(F,f)]
        typ_ += [F]

    fc_, base = init_wc(("agg_recursion.py", "comp_slice.py", "slice_edge.py"))
    print(f"Weights = {fc_}")