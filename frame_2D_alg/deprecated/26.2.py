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

def prop_F_(F):  # factory function, sets property+setter to get and update top-composition fork.N_
    def Nf_(N): return getattr(N,F).N_
    def get(N): return getattr(Nf_(N),'N_')
    def set(N, new_N): setattr(Nf_(N),'N_',new_N)
    return property(get,set)

class CN(CBase):
    name = "node"
    N_,C_,B_ = prop_F_('Nt'),prop_F_('Ct'),prop_F_('Bt')  # ext|int- defined nodes,links, Lt/Ft, Ct/lev, Bt/G
    @property
    def L_(N):
        if isinstance(N.Nt.Lt, list): N.Nt.Lt =CF(root=N.Nt)
        return N.Nt.Lt.N_
    @L_.setter
    def L_(N,v):
        if isinstance(N.Nt.Lt, list): N.Nt.Lt =CF(root=N.Nt)
        N.Nt.Lt.N_ = v
    def __init__(n, **kwargs):
        super().__init__()
        n.typ = kwargs.get('typ', 0)
        # 0= PP: block trans_comp, etc?
        # 1= L:  typ,nt,dTT, m,d,c,rc, root,rng,yx,box,span,angl,fin,compared, Nt,Bt,Ct from comp_sub, tNt,tBt,tCt from comp_F_
        # 2= G:  + rim, eTT, em,ed,ec, baseT,mang,sub,exe
        # 3= Cn: + m_,d_,r_,o_ in C.rN_
        n.m,  n.d, n.c = kwargs.get('m',0), kwargs.get('d',0), kwargs.get('c',0)  # sum forks to borrow
        n.dTT = kwargs.get('dTT',np.zeros((2,9)))  # Nt+Lt dTT: m_,d_ [M,D,n, I,G,a, L,S,A]
        n.rim = kwargs.get('rim',[])  # external links, rng-nest?
        n.em, n.ed, n.ec = kwargs.get('em',0),kwargs.get('ed',0),kwargs.get('ec',0)  # sum dTT
        n.eTT = kwargs.get('eTT',np.zeros((2,9)))  # sum rim dTT
        n.rc  = kwargs.get('rc', 1)  # redundancy to ext Gs, ave in links?
        n.Nt,n.Bt,n.Ct,n.Lt = ((kwargs.get(fork) if fork in kwargs else CF(root=n) for fork in ('Nt','Bt','Ct','Lt')))
        # ini fork tuples, ->CN / nesting, L.Lt if comp L, Ct || Nt
        n.baseT = kwargs.get('baseT',np.zeros(4))  # I,G,A: not ders, in links for simplicity, mostly redundant
        n.nt    = kwargs.get('nt', [])  # nodet, links only
        n.yx    = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.box   = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.span  = kwargs.get('span',1) # distance in nodet or aRad, comp with baseT and len(N_), not additive?
        n.angl  = kwargs.get('angl',[np.zeros(2),0])  # (dy,dx),dir, sum from L_
        n.mang  = kwargs.get('mang',1)  # ave match of angles in L_, =1 in links
        n.root  = kwargs.get('root',None)  # immediate
        n.sub = 0  # composition depth relative to top-composition peers?
        n.fin = kwargs.get('fin',0)  # clustered, temporary
        n.exe = kwargs.get('exe',0)  # exemplar, temporary
        n.rN_ = kwargs.get('rN_',[]) # reciprocal root nG_ for bG|cG, nG has Bt.N_,Ct.N_ instead?
        n.compared = set()
        n.nF = kwargs.get('nF', 'Nt')  # to set attr in root_update
        n.fb_ = kwargs.get('fb_', [])
        # ftree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def __bool__(n): return bool(n.c)

class CF(CBase):  # Nt,Ct, Bt,Lt: ext|int- defined nodes, ext|int- defining links, Lt/Ft, Ct/lev, Bt/G
    name = "fork"
    def __init__(f, **kwargs):
        super().__init__()
        f.nF = kwargs.get('nF','Nt')
        f.N_ = kwargs.get('N_',[])  # flat
        f.H  = kwargs.get('H', [])  # CF levs
        f.Lt = kwargs.get('Lt',[])  # from Ft cross_comp, Ct is parallel to Nt
        f.dTT = kwargs.get('dTT',np.zeros((2,9)))
        f.m, f.d, f.c, f.rc = [kwargs.get(x,0) for x in ('m','d','c','rc')]
        f.root = kwargs.get('root',None)
        f.fb_ = kwargs.get('fb_',[])
    def __bool__(f): return bool(f.c)

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

def sum2F_old(N_,nF, root, TT=np.zeros((2,9)), C=0, Rc=0, fset=1, fCF=0):  # -> Ft

    H = []  # unpack,concat,resum existing node'levs, sum,append to new N_'lev
    for F in N_:  # fork N_, lev=Nt
        if not F.N_: continue
        if not C: TT += F.dTT; C += F.c; Rc += F.rc
        if F.Nt.typ==4:  # flat N_
            if H: H[-1] += F.N_
            else: H = [list(F.N_)]
        else:
            if H:  # aligned bottom-up?
                for Lev,lev in zip_longest(H, F.Nt.Nt.N_):
                    if lev:
                        if Lev: Lev += lev.N_  # keep lev nesting if any, separate concat for lev.Ct.N_?
                        else: H += [list(lev.N_)]
            else: H = [list(lev.N_) for lev in F.Nt.Nt.N_]
    m,d = vt_(TT); rc = Rc/len(N_)
    Ft = (CN,CF)[fCF](dTT=TT,m=m,d=d,c=C,rc=rc,root=root,typ=4 if fCF else 5); Ft.nF = nF
    if not fCF: Ft.Nt = CF(typ=5)
    if H: Ft.N_ = [sum2f(lev,nF,Ft) for lev in H] + [CF(N_=N_,nF=nF,dTT=TT,m=m,d=d,c=C,rc=rc,root=Ft)]  # top lev
    else: Ft.N_ = N_  # no C_ in lev0: init fsub=0?
    if fset:
        root_update(root, Ft)
        if not fCF: Ft.Nt.c = Ft.c  # init only?
    return Ft

def comp_N(_N,N, rc, A=np.zeros(2), span=None):  # compare links, optional angl,span,dang?

    TT,_ = base_comp(_N, N)
    yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])  # ext
    angl = [A, np.sign(TT[1] @ wTTf[1])]  # canonic direction
    m, d = vt_(TT,rc)
    Link = CN(typ=1,exe=1, nt=[_N,N], dTT=TT,m=m,d=d,c=min(N.c,_N.c),rc=rc, yx=yx,box=box,span=span,angl=angl, baseT=(_N.baseT+N.baseT)/2)
    if m > ave * nw:
        for _Ft,Ft,nF in zip((_N.Nt,_N.Ct,_N.Bt), (N.Nt,N.Ct,N.Bt), ('Nt','Ct','Bt')):
            if _Ft and Ft:  # add eval?
                rc+=1; comp_Ft(_Ft,Ft, nF,rc,Link)
    for n, _n in (_N,N), (N,_N):  # if rim-mediated comp: reverse dir in _N.rim: rev^_rev?
        n.rim += [Link]; n.eTT += TT; n.ec += Link.c; n.compared.add(_n)
    return Link
'''
                if not hasattr(G, nF): setattr(G, nF, CF(nF=nF))  # init root.tFt
                FH=[]; dTT=np.zeros((2,9)); c=0; rc=0
                for n_ in _FH:
                    tlev = sum2F(n_, nF, getattr(G, nF)); FH+=[tlev]; dTT+=tlev.dTT; c+=tlev.c; rc+=tlev.rc
                rc/= len(FH); m,d = vt_(dTT,rc)
                setattr(G,nF, CF(N_=FH,dTT=dTT,r=r,rc=rc,m=m,d=d,root=G))
                '''
def trans_cluster(G): # trans_links mediate re-order in sort_H?
        FH_ = [[],[],[]]  # draft:
        for L in G.L_:    # splice trans_links from base links
            for FH, Ft in zip(FH_, (getattr(L,'tNt',[]),getattr(L,'tBt',[]),getattr(L,'tCt',[]))):
                if Ft:
                    if isinstance(Ft.N_[0], CF):
                        for Lev, lev in zip_longest(FH, Ft.N_):
                            if lev:
                                if Lev: Lev += lev.N_  # concat for sum2F
                                else: FH[:] = [copy(lev.N_)]
                            else:  FH += [list(lev.N_)]
                    else:          FH[0] += Ft.N_
        # merge tL_ nt root G|C?
        for FH, nF in zip(FH_, ('tNt','tBt','tCt')):
            if FH:  # merge Lt.fork.nt.roots
                for lev in reversed(FH):  # bottom-up to get incrementally higher roots
                    for tL in lev:  # trans_link
                        rt0 = tL.nt[0].root.root; rt1 = tL.nt[1].root.root  # merge Ft.Gs?
                        if rt0 != rt1: add_N(rt0, rt1, merge=1)  # concat in higher G
                # set tFt:
                if not hasattr(G,nF): setattr(G,nF, CF(nF=nF))  # init root.tFt
                setattr(G,nF, sum2f( [sum2f(n_, nF, getattr(G,nF)) for n_ in FH], nF, G))

def root_update(root, Ft, ini=1):

    _c,c = root.c,Ft.c; C = _c+c; root.c = C  # c is not weighted, min(_lev.c,lev.c) if root is link?
    root.rc = (root.rc*_c + Ft.rc*c) / C
    if isinstance(root,CF) or (hasattr(Ft,'nF') and (Ft.nF=='Nt' or Ft.nF=='Lt')):  # core forks
        root.dTT = (root.dTT*_c + Ft.dTT*c) /C
    else:  # borrow alt-fork deviations:
        root.m = (root.m*_c+Ft.m*c) /C; root.d = (root.d*_c+Ft.d*c) /C
    if ini:
        setattr(root,'t'+Ft.nF if ini==2 else Ft.nF, Ft)
    if root.root: root_update(root.root, Ft, ini=0)
    # upward recursion, we need to batch in root.fb_?

def comp_N1(_N,N, rc, full=1, A=np.zeros(2),span=None, rL=[]):

    def comp_H(_Nt,Nt, Link):
        dH, tt,C,Rc = [],np.zeros((2,9)),0,0
        for _lev,lev in zip(reversed([_Nt]+_Nt.H),reversed([Nt]+Nt.H)):  # or always top-down?
            ltt = comp_derT(_lev.dTT[1],lev.dTT[1]); lc = min(_lev.c,lev.c)
            lrc = (_lev.rc+lev.rc)/2; m,d=vt_(ltt,lrc)
            dH += [CF(dTT=ltt,m=m,d=d,c=lc,rc=lrc,root=Link)]
            tt+=ltt; C+=lc; Rc += lrc
        return dH,tt,C,Rc

    def link_update(rL):
        for ft_,nF,i in zip(rL.fb_T, ('Nt','Ct'),(0,1)):
            C = sum([ft.c if ft else 0 for ft in ft_])
            if C:
                Ft = CF(nF='t'+nF, c=C, root=rL)
                for ft in ft_:
                    if ft: cr = ft.c/C; Ft.dTT += ft.dTT*cr; Ft.rc += ft.rc*cr; Ft.N_ += ft.N_  # trans-links
                Ft.m,Ft.d = vt_(Ft.dTT,Ft.rc)
                setattr(rL,'t'+nF, Ft)
                if rL.root: rL.root.fb_T[i] += [Ft]
            elif rL.root: rL.root.fb_T[i] += [[]]
        if rL.root:
            rL.root.fb_T = [[], []]
            if len(rL.root.fb_T[0]) == len(rL.root.N_):
                link_update(rL.root)

    def comp_Ft(_Ft, Ft, tnF, rc, Link):  # root is nG, unpack node trees down to numericals and compare them

        L_=[]; TT=np.zeros((2,9)); C= Rc= 0
        for _N, N in product(_Ft.N_,Ft.N_):  # top lev, direct or via prop_F if nest->CN, spec eval in comp_n:
            if _N is N:
                dtt= np.array([N.dTT[1],np.zeros(9)]); TT+=dtt; C+=1  # overlap = pure match, not weighted?
            else:
                L = comp_N(_N,N, rc,full=0, rL=Link)  # trans-links
                if L.m > ave * (connw+rc):
                    L_+= [L]; TT+=L.dTT*L.c; Rc+=L.rc*L.c; C+=L.c
        if C:
            tF = getattr(Link, tnF); tF.N_+= L_; _c=tF.c; C+=_c; tF.c=C
            tF.dTT = dTT = (tF.dTT*_c +TT) / C
            tF.rc = rc = (tF.rc *_c +Rc) / C;  tF.m,tF.d = vt_(dTT,rc)

    TT = base_comp(_N,N)[0] if full else comp_derT(_N.dTT[1],N.dTT[1])
    m,d = vt_(TT,rc)
    Link = CN(typ=1,exe=1, nt=[_N,N], dTT=TT,m=m,d=d,c=min(N.c,_N.c),rc=rc)
    Link.tNt,Link.tCt,Link.tBt = CF(nF='tNt',root=Link),CF(nF='tCt',root=Link),CF(nF='tBt',root=Link)  # typ 1 only
    if N.typ and m > ave*nw:
        dH,tt,C,Rc = comp_H(_N.Nt, N.Nt, Link)  # tentative comp
        if m + vt_(tt,Rc/C)[0] > ave * nw:
            Link.H = dH  # hm,hd are temporary placeholders
            for _Ft, Ft, tnF in zip((_N.Nt,_N.Ct,_N.Bt), (N.Nt,N.Ct,N.Bt), ('tNt','tCt','tBt')):
                if _Ft and Ft:
                    rc+=1; comp_Ft(_Ft,Ft, tnF,rc, Link)  # sub-comps
    if full:
        if span is None: span = np.hypot(*_N.yx - N.yx)
        yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx
        box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
        angl = [A, np.sign(TT[1] @ wTTf[1])]
        Link.yx=yx; Link.box=box; Link.span=span; Link.angl=angl; Link.baseT=(_N.baseT+N.baseT)/2
    else:
        Link.root = rL  # terminal sub-comp, rL:root_L, Link: tL+ sub-tL layers, batched feedback
        rL.fb_T[0] += [Link.tNt if Link.tNt else []]
        rL.fb_T[1] += [Link.tBt if Link.tBt else []]  # not sure
        rL.fb_T[2] += [Link.tCt if Link.tCt else []]
        link_update(rL)
    for n, _n in (_N,N),(N,_N):
        n.rim+=[Link]; n.eTT+=TT; n.ec+=Link.c; n.compared.add(_n)  # or all comps are unique?

    return Link

def up_update(rL, rnF):  # upward recursion

        tNt = sum2f(rL.tNt.fb_,'tNt',rL) if rL.tNt.fb_ else []; rL.tNt.fb_ = []
        tBt = sum2f(rL.tBt.fb_,'tBt',rL) if rL.tBt.fb_ else []; rL.tBt.fb_ = []
        tCt = sum2f(rL.tCt.fb_,'tCt',rL) if rL.tCt.fb_ else []; rL.tCt.fb_ = []
        if tBt:  # tNt is never empty?
            tNt.typ = tBt.typ = 0
            tNt = comp_N(tNt, tBt, rL.rc, full=0, rL=rL,rnF=rnF)
        if tCt:
            tCt.typ = 0  # no further conversion?
            tNt = comp_N(tNt, tCt, rL.rc, full=0, rL=rL,rnF=rnF)
        getattr(rL.root, rnF).fb_ += [tNt]  # combined FtT
        if all([len(F.fb_) == len(F.N_) for F in (rL.tNt, rL.tBt, rL.tCt)]):
            link_update(rL.root, rnF)

def trans_cluster(G): # trans_links mediate re-order in sort_H?
        tF_ = [[],[],[]]  # draft:
        for L in G.L_:    # splice trans_links from base links
            for tL_, Ft in zip(tF_, (getattr(L,'tNt',[]),getattr(L,'tBt',[]),getattr(L,'tCt',[]))):
                if Ft: tL_ += Ft.N_  # flat
        # merge tL nt Gs|Cs, not Bs?
        for tL_, nF in zip(tF_, ('tNt','tBt','tCt')):
            for tL in tL_:  # merge trans_link.nt.roots
                # tL.nt[i].N_[0].root.root: N_:[Nt,rBt,rCt]?
                rt0 = tL.nt[0].root.root; rt1 = tL.nt[1].root.root  # CNs
                if rt0 != rt1: add_N(rt0, rt1, merge=1)
# partial draft:
def sum2G(Ft_,tt,c,r, root=None, init=1, typ=None, fsub=1):  # updates root if not init

    Nt = Ft_[0] ; ft_ = [Nt]
    for ft in zip(Ft_[1:], [[],[],[]]):
        ft_ += [ft]
    N_,_,ntt, nc, nr = Nt
    G = comb_Ft(*Ft_, root, N_)
    if not init: N_+=root.N_; ntt+=root.Nt.dTT; nc+=root.Nt.c  # add,norm nr? batch root updates?
    if typ is None: typ = N_[0].typ
    for N in N_[1:]:
        add_N(G,N, coef=N.c/c)  # skip forks
    Nt = sum2F(N_,'Nt',G, ntt, nc, nr)
    if len(Ft_) > 1 and Ft_[1]:  # Nt.Lt, empty if singleton, starting from trace_edge
        L_,_,ltt,lc,lr= Ft_[1]
        if init:  # else same ext
            yx_ = np.array([n.yx for n in N_]); G.yx=yx= yx_.mean(axis=0); dy_,dx_= (yx_-yx).T
            G.span = np.hypot(dy_,dx_).mean()  # N centers dist to G center
        else: L_+=root.L_; ltt+=root.Lt.dTT; lc+=root.Lt.c; lr+=root.Lt.r
        A = np.sum([l.angl[0] for l in L_], axis=0) if L_ else np.zeros(2)
        G.angl = np.array([A, np.sign(G.dTT[1] @ wTTf[1])], dtype=object)  # angle dir = d sign
        m, d = vt_(ltt,lr)
        Lt = CF(N_=L_,nF='Lt',root=G.Nt,dTT=ltt,m=m,d=d,c=lc,r=lr)  # in Nt only?
        if m > ave*specw:  # comp typ -1 pre-links
            L_,pL_= [],[]; [L_.append(L) if L.typ==1 else pL_.append(L) for L in G.L_]
            if pL_:
                if sum_vt(pL_,r)[0] > ave * specw:
                    for L in pL_:
                        link = comp_N(*L.nt, r, 1, L.angl[0], L.span)  # full = 1
                        Lt.dTT+= link.dTT-L.dTT; L_+=[link]
                    Lt.m, Lt.d = vt_(Lt.dTT, r)
        if fsub and Lt.m * Lt.d * ((len(N_)-1)*Lw) > ave*avd * (r+1) * cw:  # Variance * borrowed Match?
            V = G.m - ave*(r+1) * connw  # centw-=connw?
            if V > 0:  # cent|conn divisive clustering:
                if (mdecay(Lt.N_)- decay) * V > ave*centw:  # comb eval, Lt.N_ is L_, G.L_ is not assigned yet
                    cluster_C(G.Nt, N_,r+1)  # cent cluster: N_->Ct.N_, higher than G.C_
                else: cluster_N(G.Nt, N_,r+1)  # conn cluster/ rc+1: N_-> Nt.N_| N_
        Nt.Lt = Lt
    else: G.yx = G.yx[0]  # remove list
    if N_[0].typ==2 and G.L_:  # else mang = 1
        G.mang = np.mean([comp_A(G.angl[0], l.angl[0])[0] for l in G.L_])
    G.Nt = Nt  # Nt vals are pre-summed in Gt
    G.m, G.d = vt_(G.dTT,r)
    G.sub = N.sub+1 if G.L_ else N.sub  # remain the same sub value for singleton
    if len(Ft_) > 2:  # from cluster_N
        B_,_,btt,bc,br = Ft_[2]; m, d = vt_(btt,br)
        Bt = CN(dTT=btt,m=m,d=d,c=bc,r=br,root=G); Bt.N_=B_; Bt.nF='Bt' # use as root:
        if typ!=1 and d > avd*br*nw:  # no ddfork
            cross_comp(Bt, bc,'Bt')
        Bt.brrw = Bt.m * (root.root.m * (decay * (root.root.span/G.span)))  # trans-Bt contrast is local; subtract from root.m?
        G.Bt = Bt  # for sub-comp only, doesn't affect G vals?
    G.rN_ = sorted(G.rN_, key=lambda x: (x.m/x.c), reverse=True)  # if lG?
    # no Ct yet?
    return G

def add_N(G, N, rc=1, merge=0):  # sum Fts if merge

    N.fin = 1; N.root = G
    if hasattr(G,'m_'): G.r = np.sum([o*rc for o in N.o_]); G.rN_+=N.rN_; G.m_+=N.m_; G.o_+=N.o_  # not sure
    if N.typ and N.rN_:
        _c=G.c; c=N.c; C=_c+c; G.c=C  # weigh contribution of intensive params
        G.dTT = (G.dTT*_c + N.dTT*c) /C; G.r = (G.r*_c + N.r*c) /C
        G.m,G.d = vt_(G.dTT, G.r)
        G.C_ += N.rN_; G.Ct.dTT += N.Ct.dTT*rc; G.Ct.c += N.Ct.c*rc; G.Ct.r += N.Ct.r*rc  # for trans-cluster and cluster_C
    G.span = (G.span*_c+N.span*c*rc) / C
    A,a = G.angl[0],N.angl[0]; A[:] = (A*_c+a*c*rc) /C  # vect only
    if isinstance(G.yx, list): G.yx += [N.yx]  # weigh by C?
    if N.typ > 1:
        G.baseT = (G.baseT*_c + N.baseT*c*rc) /C
        G.mang = (G.mang*_c + N.mang*c*rc) /C
        G.box = extend_box(G.box, N.box)
    if merge: merge_f(G, N, cc=G.c/N.c)  # if not batched
    # if N is Fg: margin = Ns of proj max comp dist > min _Fg point dist: cross_comp Fg_?
    return N

def _add_F(T, Ft, nF, r=1, merge=1):

    G = T.root; Ft.r -= Ft.m > T.m  # lower than T.m by default
    def sum_H(H, h, r, root):
        for Lev, lev in zip_longest(H, h):  # bottom-up
            if lev:
                if Lev: add_F(Lev, lev, nF, r)
                else:   H.append(CopyF(lev, root))
        return list(H)
    _c = T.c; c = Ft.c; C = _c + c; T.c = C; rc = C / c
    T.dTT = (T.dTT * _c + Ft.dTT * c) / C
    T.r = (T.r * _c + Ft.r * c) / C
    T.m, T.d = vt_(T.dTT, T.r)
    if merge:
        for N in Ft.N_:
            N.fin = 1; N.root = T
            if hasattr(T,'m_'): T.r = np.sum([o * r for o in N.o_]); T.rN_ += N.rN_; T.m_ += N.m_; T.o_ += N.o_
            T.C_ += N.rN_
            T.box = extend_box(T.box, N.box)
            A, a = T.angl[0], N.angl[0]; A[:] = (A * _c + a * c * r) / C
            T.span = (T.span * _c + N.span * c * r) / C
            if N.typ > 1:
                T.baseT = (T.baseT * _c + N.baseT * c) / C
                T.mang = (T.mang * _c + N.mang * c) / C
            if isinstance(T.yx, list): T.yx += [N.yx]
        if G.typ==2 and G.L_:
            G.mang = np.mean([comp_A(G.angl[0], l.angl[0])[0] for l in G.L_])
            G.yx = [n.yx for n in T.N_]
            if len(T.N_) > 1:
                yx_ = np.array(G.yx); G.yx = yx_.mean(axis=0); dy_, dx_ = (yx_ - G.yx).T
                G.span = np.hypot(dy_, dx_).mean()
            elif T.N_:
                G.yx = copy(T.N_[0].yx); G.span = T.N_[0].span
            if nF=='Lt':
                A = np.sum([l.angl[0] for l in T.N_], axis=0) if T.N_ else np.zeros(2)
                G.angl = np.array([A, np.sign(G.dTT[1] @ wTTf[1])], dtype=object)
                if T.m > ave * specw:  # comp typ -1 pre-links
                    pL_ = [L for L in G.L_ if L.typ!=1]
                    if pL_ and sum_vt(pL_,r)[0] > ave * specw:
                        for L in pL_:
                            link = comp_N(*L.nt,r, 1, L.angl[0], L.span)
                            T.dTT += link.dTT-L.dTT; G.L_.append(link)
                        T.m, T.d = vt_(T.dTT, r)
        if isinstance(Ft, CN): Ft = Ft.Nt  # handling for comb_Ft
        if T.H and Ft.H:
            T.H = sum_H(T.H, Ft.H, r, T)

def add_N_pars(T, N, nF, r=1, merge=0):  # draft, sum Fts if merge

    N.fin = 1; N.root = T
    _,_,_,C,R= sum_vt([T,N],T); r += R
    if hasattr(T,'m_'): T.r = np.sum([o*r for o in N.o_]); T.rN_+=N.rN_; T.m_+=N.m_; T.o_+=N.o_
    if N.typ:
        T.C_ += N.rN_; _c,c = T.c,N.c  # C_ overlaps between Gs
        if nF != 'Lt':
            A,a = T.angl[0],N.angl[0]; A[:] = (A*_c+a*c) /C  # vect only
        T.span = (T.span*_c + N.span*c) / C
        T.box = extend_box(T.box,N.box)
        if N.typ > 1:
            T.baseT = (T.baseT*_c + N.baseT*c) /C
            T.mang = (T.mang*_c + N.mang*c) /C
    if isinstance(T.yx, list): T.yx += [N.yx]  # weigh by C?
    if merge: merge_f(T, N, cc=T.c/N.c)  # if not batched
    # if N is Fg: margin = Ns of proj max comp dist > min _Fg point dist: cross_comp Fg_?

def vT_(_T,T, root=None):
    _c = _T.c; c = T.c; C = _c + c
    R = (_T.r*_c + T.r*c) / C
    TT = (_T.dTT*_c + T.dTT*c) /C; m,d = vt_(TT,R)
    if root: root.dTT=TT; root.r=R; root.c=C; root.m=m; root.d=d
    return R,C, TT, m,d

def merge_f(N,n, cc=1):
    for Ft, ft in zip((N.Nt, N.Bt, N.Lt), (n.Nt, n.Bt, n.Lt)):
        if ft:
            add_F(Ft, ft, (n.r + n.r*cc) / 2)  # ft*cc?
            setattr(N, Ft.nF, Ft)  # not sure about N.root update

def add_F(T, Ft, nF, r=1, merge=1):

    Ft.r -= Ft.m > T.m  # lower than T.m by default
    T.m,T.d,T.dTT,T.c,T.r = sum_vt([T,Ft])
    C = T.c; r += T.r
    def sum_H(H, h, r, root):
        for Lev,lev in zip_longest(H, h):  # bottom-up
            if lev:
                if Lev: add_F(Lev,lev, nF, r)
                else:   H.append(CopyF(lev, root))
        return list(H)
    G = T.root if isinstance(T,CF) else T  # add ext attrs in G from Nt + Lt:
    if nF =='Lt':  # or abs(typ)==1?
        L_ = Ft.N_
        if Ft.m > ave*specw:  # comp typ -1 pre-links
            L_,pL_ =[],[]; [L_.append(L) if L.typ==1 else pL_.append(L) for L in Ft.N_]
            if pL_ and sum_vt(pL_,r)[0] > ave*specw:
                for L in pL_:
                    link = comp_N(*L.nt, r,1, L.angl[0], L.span); Ft.dTT+= link.dTT-L.dTT; L_+=[link]
                Ft.m, Ft.d = vt_(Ft.dTT, r)  # + update G?
        A = np.sum([l.angl[0] for l in L_], axis=0) if L_ else np.zeros(2)
        G.angl = np.array([A, np.sign(G.dTT[1] @ wTTf[1])], dtype=object)
        G.mang = np.mean([comp_A(G.angl[0], l.angl[0])[0] for l in G.L_])  # Ls only?
    elif nF =='Nt':
        N_ = Ft.N_; yx_ = []
        T.N_ += N_ if merge else [Ft]  # also merge L_?
        if isinstance(T, CF) and T.H and Ft.H: T.H = sum_H(T.H,Ft.H, r,T)
        for N in N_:
            N.fin = 1; N.root = T; c = N.c
            if hasattr(T,'m_'): T.r = np.sum([o*T.r for o in N.o_]); T.rN_+=N.rN_; T.m_+=N.m_; T.o_+=N.o_
            G.C_ += N.rN_  # Ct||Nt;  A,a = G.angl[0],N.angl[0]; A[:]= (A*C+a*c)/C # vect only, if in Nt?
            G.baseT = (T.baseT*C + N.baseT*c) /C
            G.box = extend_box(T.box, N.box)
            yx_ += [N.yx]
        G.yx = yx = np.mean(yx_, axis=0); dy_,dx_= (np.array(yx_)-yx).T
        G.span = np.hypot(dy_,dx_).mean() if len(N_)>1 else N.span
        # if merge?
    return T



