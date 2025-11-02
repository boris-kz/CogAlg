class CH(CBase):  # nesting hierarchy or a level thereof

    name = "H"    # from top-composition = bottom derivation
    def __init__(n, **kwargs):
        super().__init__()
        n.dTT = kwargs.get('dTT',np.zeros((2,9)))  # m_,d_ [M,D,n, I,G,a, L,S,A]: single or sum H
        n.H   = kwargs.get('H',[])  # for nesting, empty if single layer represented by F_,dTT
        n.F_  = kwargs.get('F_',[])  # N_,B_,C_|Nt,Bt,Ct, each [n_,m,d,c, rc], empty if H
        n.rc  = kwargs.get('rc',0)  # complement to root.rc, for lev ranking
        n.m   = kwargs.get('m',0); n.d = kwargs.get('d',0); n.c = kwargs.get('c',0)  # to set rc
        n.root = kwargs.get('root',[])  # to pass vals?
        # n.depth = 0  # max nesting depth in H
    def __bool__(n): return bool(n.rc)

def comp_H(_H, H, rc, root, TT=None):  # unpack derH trees down to numericals and compare them

    spec = 1  # default
    if TT is None:  # recursive call, else dTT is passed from comp_N, _H.dTT and H.dTT are redundant
        TT = comp_derT(_H.dTT[1], H.dTT[1]*rc)
        if not _H.H and H.H and val_(TT, rc, mw=(1-min(len(H),len(_H)) *Lw)) < 0:
            spec = 0  # spec eval for recursive call only, 2 or more levs/H, different Lw for H?
    dH = []
    if spec:
        for Lev,lev in zip(_H.H, H.H):
            tt = np.zeros((2,9)); fork_ = [[],[],[]]  # or 6?
            for i, (F,f) in enumerate(zip_longest(Lev, lev)):
                if F and f:
                    # add comp_derT(fork dTTs), spec eval?
                    N_,L_,mTT,B_,dTT = comp_N_(F[0],rc,f[0]) if i<2 else comp_C_(F[0],rc,f[0])
                    fTT = mTT + dTT; tt += fTT
                    fork_[i] = [[N_,fTT]]  # do we need B_,L_ per fork?
            TT += tt; dH += [[fork_,tt]]
            # add fork_,dH sort and rc assign
    return CN(H=dH, dTT=TT, root=root, rc=rc, m=sum(TT[0]), d=sum(TT[1]), c=min(_H.c, H.c))

