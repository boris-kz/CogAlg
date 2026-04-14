# for code modification, just a stab now

import ast
from agg_recursion import onF_


def init_wc(path):
    # compute weights based on operations in function, independent of deeper callees
    def get_ops(node):
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
        return sum(weights.get(type(n), 0) for n in ast.walk(node))

    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    # get only onF_'s function name and their function object, and also vt_
    funcs = {func.name: func for func in ast.walk(tree) if isinstance(func, ast.FunctionDef) and (func.name in onF_ or func.name == 'vt_')}
    # get base vt_'s operations weight
    base = get_ops(funcs.pop('vt_'))
    return [(get_ops(funcs[name]) / base) for name in onF_]


if __name__ == "__main__":
    fc_ = init_wc("agg_recursion.py")
    print(f"Weights = {fc_}")