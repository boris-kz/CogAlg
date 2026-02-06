def comp_N(_N,N, rc, A=np.zeros(2), span=None):  # compare links, optional angl,span,dang?

    TT,_ = base_comp(_N, N)
    yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])  # ext
    angl = [A, np.sign(TT[1] @ wTTf[1])]  # canonic direction
    m, d = vt_(TT,rc)
    Link = CN(typ=1,exe=1, nt=[_N,N], dTT=TT,m=m,d=d,c=min(N.c,_N.c),rc=rc, yx=yx,box=box,span=span,angl=angl, baseT=(_N.baseT+N.baseT)/2)
    # trans-root comp:
    Nt_, Ct_ = zip(*N.Nt.N_) if N.Nt.N_ else [],[]; _Nt_,_Ct_= zip(_N.Nt.N_) if _N.Nt.N_ else [],[]
    for _F_,F_,nF in zip((_Nt_,_N.Bt.N_,_Ct_),(Nt_,N.Bt.N_,Ct_),('Nt','Bt','Ct')):
        if _F_ and F_:
            comp_F_(_F_,F_,'t'+nF,rc, Link)  # deeper trans_comp in comp_C_'comp_N, unpack|reref levs?
            rc += 1  # default fork redundancy
        for n, _n in (_N,N), (N,_N):  # if rim-mediated comp: reverse dir in _N.rim: rev^_rev?
            n.rim += [Link]; n.eTT += TT; n.ec += Link.c; n.compared.add(_n)
    return Link

def comp_F_(_F_,F_,nF, rc, root):  # root is nG, unpack node trees down to numericals and compare them

    L_,TTm,C,TTd,Cd = [],np.zeros((2,9)),0,np.zeros((2,9)),0; Rc=cc=0  # comp count
    if isinstance(F_[0],CN):
        # same for nested top level: N_[-1], if aligned?
        for _N, N in product(F_,_F_):
            if _N is N: dtt = np.array([N.dTT[1], np.zeros(9)]); TTm += dtt; C=1; Cd=0  # overlap is pure match
            else:       cm,cd = comp_n(_N,N, TTm,TTd,C,Cd,rc,L_); C+=cm; Cd+=cd
            Rc+=_N.rc+N.rc; cc += 1
        if L_: sum2f(L_,nF,root)  # always flat tFt
    else:
        for _lev,lev in zip(_F_,F_):  # L_=H, bottom-up, or top-down if root!node selective?
            dlev = []; rc += 1  # redundant higher levs and Cts
            for _ft, ft in zip(_lev,lev):  # lev:[nt,ct], no ct in B_?
                if not _ft or not ft: continue
                lTT = comp_derT(_ft.dTT[1],ft.dTT[1]); lRc= lC= lcc= 1  # min per dTT?
                _sN_,sN_ = set(_lev.N_), set(lev.N_)
                iN_ = list(_sN_ & sN_)  # intersect = match
                for n in iN_: lTT+=n.dTT; lC+=n.c; lRc+=n.rc; lcc+=1
                _oN_= _sN_-sN_; oN_= sN_-_sN_; dN_= []
                for _n,n in product(_oN_,oN_):
                    cm,_ = comp_n(_n,n, lTT,TTd,C,Cd,rc, dN_)  # comp offsets
                    lRc += _n.rc+n.rc; lC+=cm; lcc+=1
                lRc /= len(dN_); m,d = vt_(lTT,lRc)
                dlev += [CF(N_=dN_,nF=ft.nF, root=root,dTT=lTT,m=m,d=d,c=lC,rc=lRc)]
                TTm+= lTT; C+=lC; Rc+=lRc; cc+=lcc
            if dlev:
                L_ += [dlev]  # (nt+ct)/2: fork rdn?
        if L_:
            Rc/=cc; m,d=vt_(TTm,Rc); setattr(root,nF, CF(N_=L_,nF=nF,dTT=TTm,m=m,d=d,c=C,rc=Rc,root=root))

def sum2F(N_,nF, root, TT=np.zeros((2,9)), C=0, Rc=0, fset=1, fCF=0):  # -> Ft

    H = []; ff = nF=='Nt'  # unpack,concat,resum existing node'levs, sum,append to new N_'lev
    for F in N_:  # fork N_, lev = [nt,ct], if Nt only?
        if not F.N_: continue
        if not C: TT += F.dTT; C += F.c; Rc += F.rc
        if isinstance(F.Nt.N_[0], CN):
            if H: (H[-1][0] if ff else H[-1]).extend(F.N_)  # flat
            else: H = [[list(F.N_),[]]] if ff else [list(F.N_)]
        else:  # H aligned bottom-up
            if H:
                for Lev,lev in zip_longest(H, F.Nt.N_):
                    if lev:
                        if Lev:
                            for _ft,ft in zip_longest(Lev if ff else [Lev], lev if ff else [lev]):
                                if ft: _ft += ft.N_  #  concat nt,ct
                        else: H += [[list(ft.N_) for ft in lev] if ff else list(lev.N_)]
            else:
                H = [[(list(ft.N_) if ft else []) for ft in lev] if ff else list(lev.N_) for lev in F.Nt.N_]
    m,d = vt_(TT); rc = Rc/len(N_); Cx = (CN,CF)[fCF]
    Ft = Cx(dTT=TT,m=m,d=d,c=C,rc=rc,root=root)
    Ft.nF = nF  # splice N_ H:
    if H:
        Ft.N_ = [[(sum2f(n_,nF,Ft) if n_ else CF()) for n_ in lev] if ff else sum2f(lev,nF,Ft) for lev in H]  # skip empty list
        topNt = CF(N_=N_,nF=nF,dTT=TT,m=m,d=d,c=C,rc=rc,root=Ft)
        Ft.N_+= [[topNt,CF()]] if ff else [topNt]
    else:
        Ft.N_ = N_  # no C_ in lev0: init fsub=0?
    if fset:
        root_update(root, Ft)
        if not fCF: Ft.Nt.c = Ft.c  # init only?
    return Ft

def sub_comp(_N, N, rc, Link):  # root is nG, unpack node trees down to numericals and compare them

    if _N.Nt and N.Nt:  # trans-root comp
        Nt_,Ct_= zip(*N.Nt.N_); _Nt_,_Ct_= zip(*_N.Nt.N_)
        for _Ft, Ft, nF in zip((_Nt_,Nt_), (_Ct_,Ct_), ('Nt','Ct')):
            rc += 1
            comp_Ft(_Ft, Ft, nF, rc, Link)  # deeper trans_comp in comp_n, unpack|reref levs?
    if _N.Bt and N.Bt:
        comp_Ft(_N.Bt, N.Bt, 'Bt', rc + 1, Link)

def prop_F_(F):  # factory function, sets property+setter to get and update top-composition fork.N_
    def Nf_(N):  # CN Nt | Lt | Bt | Ct
        Ft = getattr(N,F)
        if Ft: return Ft if Ft.typ==4 else Ft.Nt.N_[-1]  # or Ft.N_?
        else:  return Ft
    def get(N): return getattr(Nf_(N),'N_')
    def set(N, new_N): setattr(Nf_(N),'N_',new_N)
    return property(get,set)

class CN(CBase):
    name = "node"
    N_,C_, B_,L_ = prop_F_('Nt'),prop_F_('Ct'), prop_F_('Bt'), prop_F_('Lt')
    # ext| int- defined nodes, ext|int- defining links, Lt/Ft, Ct/lev, Bt/G

def comp_Ft(_Ft, Ft, nF, rc, root):  # root is nG, unpack node trees down to numericals and compare them

    L_,TTm,C,TTd,Cd = [],np.zeros((2,9)),0,np.zeros((2,9)),0; Rc=cc=0  # comp count
    # add eval for nested levs and Ct:
    for _N, N in product(_Ft.N_,Ft.N_):  # top lev, spec eval in comp_n:
        if _N is N: dtt = np.array([N.dTT[1], np.zeros(9)]); TTm += dtt; C=1; Cd=0  # overlap is pure match
        else:       cm,cd = comp_n(_N,N, TTm,TTd,C,Cd,rc,L_); C+=cm; Cd+=cd
        Rc += _N.rc+N.rc; cc += 1  # not edited
    if L_:
        if Ft.N_[0].typ > 3: # N_=H, lev.typ=5 if nested
            for _lev,lev in zip( reversed(_Ft.N_[:-1]), reversed(Ft.N_[:-1])):  # top-1 - down
                TTm += comp_derT(_lev.dTT[1],lev.dTT[1]); C+=min(_lev.c,lev.c)  # lrc?
        sum2F(L_,'t'+nF, root, TTm,C, rc, fCF=1)  # rc is wrong
        # Rc/=cc; m,d=vt_(TTm,Rc); setattr(root,nF, CF(N_=L_,nF=nF,dTT=TTm,m=m,d=d,c=C,rc=Rc,root=root))

