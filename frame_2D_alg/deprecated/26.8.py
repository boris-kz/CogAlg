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

def cluster_P(_C_, _c, root):  # multi-seed mean shift: parallel centroid refine, _C_ varies via split/merge

    iC_ = _C_; cnt = 0; Ln = len(N_:= list(set([N for C in _C_ for N in C.N_])))  # all Ns are in all Cs
    _md__ = np.zeros((Ln, len(_C_), 2))  # NxC, cols aligned to _C_
    for i, N in enumerate(N_):
        for c,m,d in N.root_:
            if c in _C_: _md__[i,_C_.index(c)] = m,d
    while True:
        for N in N_: N.root_ = []  # reset, append in sum2F
        Lc = len(_C_); L = Lc*Ln  # Lc,L,md__ per cycle since _C_ varies
        md__ = np.zeros((Ln,Lc,2)); O = 0
        for j,N in enumerate(N_):
            for i,C in enumerate(_C_): md__[j,i] = val_(base_comp(C,N)[0], ttcP)
            m_ = md__[j,:,0]; O += m_.sum() - m_.max()  # cross-C ambiguity, gates split/merge
        C_ = [sum2F(N_, root, md__[:,i,0], md__[:,i,1]) for i in range(Lc)]  # mean shift, aligned to md__
        Mt = md__[:,:,0].sum()  # total V
        conv = md__.shape==_md__.shape and Mt* np.abs(md__-_md__).sum()* (wcP*L) <= ave*(root.r+ccP*L)  # convergence
        removed = []
        if gv_(O*wcP - ave*(root.r+ccP*L)):  # merge redundant Cs
            for i,_C in enumerate(C_):
                if _C in removed: continue
                for C in C_[i+1:]:
                    if C in removed: continue
                    l = comp_N(C,_C,(C.r+_C.r)/2, min(C.c,_C.c), A=(a:=_C.yx-C.yx), span=np.hypot(*a))
                    if l.m*wF > ave*(l.r+cF):
                        add2F(_C,C,1); removed += [C]
        new_ = [Copy_(N,root,init=1,cls=CL) for j,N in enumerate(N_)
                if np.sort(md__[j,:,0])[-2:].min()*wcP > ave*(N.r+ccP)]  # seed overlap Ns
        _C_ = [c for c in C_ if c not in removed and c.m*wcP > ave*c.r*ccP] + new_  # survive+prune, +seeds
        if conv and not (removed) and len(_C_)==Lc:
            break
        _md__ = md__; cnt += 1
    out_ = []
    for N in N_: N.root_ = []  # replace with out_ Cs:
    for i, _C in enumerate(C_):
        if _C.m > ave * _C.r:  # prune, add olp as stronger ms?
            N_,m_,d_ = [],[],[]
            for N, m,d in zip(_C.N_, md__[:,i,0], md__[:,i,1]):
                if m*N.c > ave*N.r: N_+=[N]; m_+=[m]; d_+=[d]
            if N_:
                C = sum2F(N_,root, m_,d_)
                for N in N_:
                    L = CN(typ=1, dTT=N.dTT,c=N.c,r=N.r,m=N.m,d=N.d, span=np.hypot(*(dy_dx:=C.yx-N.yx)), angl=[dy_dx,np.sign(N.dTT[1]@ttcN[1])])
                    L.N_ = [N,C]; C.L_ += [L]
                out_ += [C]
    if out_:
        iTT, iC, iR = sum_vt(iC_); oTT,oC,oR = sum_vt(out_)
        FV_(CoF.get(), iTT-oTT, iC-oC, iR-oR)
    return out_
'''
                        _C.m_ = [(_m+m)/2 for _m,m in zip(_C.m_,C.m_)]; _C.d_ = [(_d+d)/2 for _d,d in zip(_C.d_,C.d_)]
                        for N in N_:
                            N.root_[i][1] += N.root_[j][1]; N.root_[i][2] += N.root_[j][2]  # sum m and d reference of merged _C
                            N.root_.pop(j)
'''