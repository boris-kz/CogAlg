def cross_comp(root, rc, fcon=1, fL=0):  # core function, mediates rng+ and der+ cross-comp and clustering, rc = rdn + olp

    iN_,L_,TT,c,TTd,cd = comp_N_((root.N_,root.B_)[fL], rc) if fcon else comp_C_(root.N_,rc); nG_=[]  # nodes| centroids
    if L_:
        cr = cd / (c+cd) *.5  # dfork borrow ratio, .5 for one direction
        if val_(TT, rc+compw, TTw(root), (len(L_)-1)*Lw,1,TTd,cr) > 0 or fL:  # default for links
            root.L_=L_; sum2T(L_,rc,root,'Lt')  # new ders, no Lt.N_, root.B_,Bt if G
            for n in iN_: n.em = sum([l.m for l in n.rim]) / len(n.rim)  # pre-val_
            nG_,rc = Cluster(root, L_,rc,fcon,fL)  # Bt in cluster_N, sub+ in sum2G
        # agg+:
        if nG_ and val_(TT, rc+connw, TTw(root), (len(nG_)-1)*Lw) > 0:  # mval only
            nG_,rc = trace_edge(root.N_,rc,root)  # comp Ns x N.Bt|B_.nt, with/out mfork?
        if nG_ and val_(root.dTT, rc+compw, TTw(root), (len(root.N_)-1)*Lw,1, TTd,cr) > 0:
            nG_,rc = cross_comp(root, rc)

    return nG_,rc  # nG_: recursion flag

def Cluster(root, iL_, rc, fcon=1, fL=0):  # generic clustering root

    def trans_cluster(root, iL_, rc):  # called from cross_comp(Fg_), others?

        dN_,dB_,dC_ = [],[],[]  # splice specs from links between Fgs in Fg cluster
        for Link in iL_:
            dN_+= Link.N_; dB_+= Link.B_; dC_+= Link.C_
        for tL_,nf_,nft,fC in (dN_,'tN_','tNt',0), (dB_,'tB_','tBt',0), (dC_,'tC_','tCt',1):
            if tL_:
                Ft = sum2T(tL_,rc, root,nft); N_ = list({n for L in tL_ for n in L.nt})
                for N in N_: N.exe=1
                cluster_N(Ft, N_,rc)  # default fork redundancy
                if val_(Ft.dTT, rc, TTw(root), (len(Ft.N_)-1)*Lw) > 0:
                    cross_comp(Ft, rc)  # unlikely, doesn't add rc?
                setattr(root,nf_, tL_)
                rc += 1
    def get_exemplars(N_, rc):  # multi-layer non-maximum suppression -> sparse seeds for diffusive centroid clustering
        E_ = set()
        for rdn, N in enumerate(sorted(N_, key=lambda n:n.em, reverse=True), start=1):  # strong-first
            oL_ = set(N.rim) & {l for e in E_ for l in e.rim}
            roV = vt_(sum([l.dTT for l in oL_]), rc)[0] if oL_ else 0 / (N.em or eps)  # relative rim olp V
            if N.em * N.c > ave * (rc+rdn+compw+ roV):  # ave *= rV of overlap by stronger-E inhibition zones
                E_.update({n for l in N.rim for n in l.nt if n is not N and N.em > ave*rc})  # selective nrim
                N.exe = 1  # in point cloud of focal nodes
            else: break  # the rest of N_ is weaker, trace via rims
        return E_
    G_ = []
    if fcon:  # connect-cluster, no exemplars or recursive centroid clustering
        N_,L_,i = [],[],0  # add pL_, pN_ for pre-links?
        if fcon > 1:  # merge similar Cs, not dCs, no recomp
            link_ = sorted(list({L for L in iL_}), key=lambda link: link.d)  # from min D
            for i, L in enumerate(link_):
                if val_(L.dTT, rc+compw, wTTf,fi=0) < 0:
                    _N, N = L.nt; root.L_.remove(L)  # merge nt, remove link
                    if _N is not N:  # not yet merged
                        for n in N.N_: add_N(_N, n); _N.N_ += [n]
                        for l in N.rim: l.nt = [_N if n is N else n for n in l.nt]
                        if N in N_: N_.remove(N)
                        N_ += [_N]
                else: L_ = link_[i:]; break
        else: L_ = iL_
        N_ = list(set([n for L in L_ for n in L.nt] + N_))  # include merged Cs
        if N_ and (val_(root.dTT, rc+connw, TTw(root), mw=(len(N_)-1)*Lw) > 0 or fcon>1):
            G_,rc = cluster_N(root, N_,rc, fL)  # contig in feature space if centroids, no B_,C_?
            if G_:  # top root.N_
                tL_ = [tl for n in root.N_ for l in n.L_ for tl in l.N_]  # trans-links
                if sum(tL.m for tL in tL_) * ((len(tL_)-1)*Lw) > ave*(rc+connw):  # use tL.dTT?
                    trans_cluster(root, tL_, rc+1)  # sets tTs
                    mmax_ = []
                    for F,tF in zip((root.Nt,root.Bt,root.Ct), (root.tNt,root.tBt,root.tCt)):
                        if F and tF:
                            maxF,minF = (F,tF) if F.m>tF.m else (tF,F)
                            mmax_+= [max(F.m,tF.m)]; minF.rc+=1  # rc+=rdn
                    sm_ = sorted(mmax_, reverse=True)
                    for m, (Ft, tFt) in zip(mmax_,((root.Nt, root.tNt),(root.Bt, root.tBt),(root.Ct, root.tCt))): # +rdn in 3 fork pairs
                        r = sm_.index(m); Ft.rc+=r; tFt.rc+=r  # rc+=rdn
    else:
        # primary centroid clustering:
        N_ = list({N for L in iL_ for N in L.nt if N.em})  # newly connected only
        E_ = get_exemplars(N_,rc)
        if E_ and val_(np.sum([g.dTT for g in E_],axis=0), rc+centw, TTw(root), (len(E_)-1)*Lw) > 0:
            G_,rc = cluster_C(E_,root,rc)  # forms root.Ct, may call cross_comp-> cluster_N, incr rc
    return  G_,rc

def cluster_N(root, iN_, rc, fL=0):  # flood-fill node | link clusters, flat

    def rroot(n): return rroot(n.root) if n.root and n.root!=root else n
    def nt_vt(n,_n):
        M, D = 0,0  # exclusive match, contrast
        for l in set(n.rim+_n.rim):
            if l.m > 0:   M += l.m
            elif l.d > 0: D += l.d
        return M, D
    # root attrs:
    G_, N__,L__,Lt_ = [],[],[],[]; TT,nTT,lTT = np.zeros((2,9)),np.zeros((2,9)),np.zeros((2,9)); C,nC,lC = 0,0,0; in_= set()
    for N in iN_: N.fin = 0
    for N in iN_:  # form G per remaining N
        if N.fin or (root.root and not N.exe): continue  # no exemplars in Fg
        N_=[N]; L_,B_,C_=[],[],[]; N.fin=1  # init G
        __L_ = N.rim  # spliced rim
        while __L_:
            _L_ = []
            for L in __L_: # flood-fill via frontier links
                _N = L.nt[0] if L.nt[1].fin else L.nt[1]; in_.add(L)
                if not _N.fin and _N in iN_:
                    m,d = nt_vt(*L.nt)
                    if m > ave * (rc-1):  # cluster nt, L,C_ by combined rim density:
                        N_ += [_N]; _N.fin = 1
                        L_ += [L]; C_ += _N.C_
                        _L_+= [l for l in _N.rim if l not in in_ and (l.nt[0].fin ^ l.nt[1].fin)]   # new frontier links, +|-?
                    elif d > avd * (rc-1):  # contrast value, exclusive?
                        B_ += [L]
            __L_ = list(set(_L_))
        if N_:
            Ft_ = ([list(set(N_)),np.zeros((2,9)),0,0], [list(set(L_)),np.zeros((2,9)),0,0], [list(set(B_)),np.zeros((2,9)),0,0], [list(set(C_)),np.zeros((2,9)),0,0])
            for i, (F_,tt,c,r) in enumerate(Ft_):
                for F in F_:
                    tt += F.dTT; Ft_[i][2] += F.c
                    if i>1: Ft_[i][3] += F.rc  # Bt,Ct rdn only?
            (N_,nt,nc,_),(L_,lt,lc,_), (B_,bt,bc,br),(C_,ct,cc,cr) = Ft_
            tt = nt*nc + lt*lc; c= nc+lc
            if B_: tt += bt*bc*(br/len(B_)); c+=bc
            if C_: tt += ct*cc*(cr/len(C_)); c+=cc
            if val_(tt, rc, TTw(root), (len(N_)-1)*Lw) > 0 or fL:
                G = sum2G(((N_,nt,nc),(L_,lt,lc),(B_,bt,bc),(C_,ct,cc)), rc,root)  # br,cr?
                if not fL and G.Bt and G.Bt.d > avd * rc * nw:  # no ddfork
                    lg_,_rc = cross_comp(G, rc, fL=1)  # proximity-prior comp B_
                    if lg_: sum2T(lg_,_rc,G,'Bt')
                N__+= N_; L__+=L_; Lt_+=[n.Lt for n in N_]; TT+=tt; nTT+=nt; lTT+=lt; C+=c; nC+=nc; lC+=lc  # G.TT * cr * rcr?
                G_ += [G]
    if G_ and (fL or val_(TT, rc+1, TTw(root), (len(G_)-1)*Lw)):  # include singleton lGs
        rc += 1
        root_replace(root,rc, G_,N__,L__,Lt_,TT,nTT,lTT,C,nC,lC)
    return G_, rc

def cluster_C(E_, root, rc):  # form centroids by clustering exemplar surround via rims of new member nodes, within root

    _C_, _N_ = [],[]
    for E in E_:
        C = cent_TT( Copy_(E, root, init=2, typ=3), rc, init=1)  # all rims are within root, sequence along max w attr?
        C._N_ = [n for l in E.rim for n in l.nt if n is not E]  # core members + surround -> comp_C
        _N_ += C._N_; _C_ += [C]
        for N in C._N_: N.M, N.D, N.C, N.DTT = 0,0,0,np.zeros((2,9))  # init, they might be added to Ct.N_ later
    # reset:
    for n in set(root.N_+_N_ ): n.Ct.N_,n.mo_, n._C_,n._mo_ = [],[], [],[]  # aligned pairs, in cross_comp root
    # reform C_, refine C.N_s
    while True:
        C_,cnt,olp, mat, dif, DTT,Dm,Do = [],0,0,0,0,np.zeros((2,9)),0,eps; Ave = ave * (rc + nw)
        _Ct_ = [[c, c.m/c.c if c.m !=0 else eps, c.rc] for c in _C_]
        for _C,_m,_o in sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True):
            if _m > Ave * _o:
                C = cent_TT( sum2G([[_C.N_,np.sum([N.dTT for N in _C.N_],axis=0),sum([N.c for N in _C.N_])]], rc, root, fsub=0), rc, init=1)
                # C update lags behind N_; non-local C.rc += N.mo_ os?
                _N_,_N__,mo_,M,D,O,comp,dTT,dm,do = [],[],[],0,0,0,0,np.zeros((2,9)),0,0  # per C
                for n in _C._N_:  # core+ surround
                    if C in n.Ct.N_: continue
                    dtt,_ = base_comp(C,n); comp += 1
                    m,d = vt_(dtt,rc); dTT += dtt; nm = m * n.c  # rm,olp / C
                    odm = np.sum([mo[0]-nm for mo in n._mo_ if mo[0]>m])  # overlap = higher-C inclusion vals - current C val
                    if m > 0 and nm > Ave * odm:
                        _N_+=[n]; M+=m; O+=odm; mo_ += [np.array([nm,odm])]  # n.o for convergence eval
                        _N__ += [_n for l in n.rim for _n in l.nt if _n is not n]  # +|-Ls
                        if _C not in n._C_: dm+=m; do+=odm  # not in extended _N__
                    else:
                        if _C in n._C_: __m,__o = n._mo_[n._C_.index(_C)]; dm+=__m; do+=__o
                        D += abs(d)  # distinctive from excluded nodes (background)
                mat+=M; dif+=D; olp+=O; cnt+=comp  # from all comps?
                DTT += dTT
                if M > Ave * len(_N_) * O and val_(dTT, rc+O, TTw(C),(len(C.N_)-1)*Lw):
                    for n, mo in zip(_N_,mo_): n.mo_+=[mo]; n.Ct.N_+=[C]; C.Ct.N_+=[n]  # reciprocal root assign
                    C.M += M; C.D += D; C.C += comp; C.DTT += dTT
                    C.N_+=_N_; C._N_ = list(set(_N__))  # core, surround elements
                    C_ += [C]; Dm+=dm; Do+=do  # new incl or excl
                else:
                    for n in _C._N_:
                        n.exe = n.m/n.c > 2 * ave  # refine exe
                        for i, c in enumerate(n.Ct.N_):
                            if c is _C: # remove mo mapping to culled _C
                                n.mo_.pop(i); n.Ct.N_.pop(i); break
            else: break  # the rest is weaker
        if Dm/Do > Ave:  # dval vs. dolp, overlap increases as Cs may expand in each loop
            _C_ = C_
            for n in root.N_: n._C_=n.Nt.N_; n._mo_=n.mo_; n.Nt.N_,n.mo_ = [],[]  # new n.Ct.N_s, combine with vo_ in Ct_?
        else:  # converged
            break
    C_ = [C for C in C_ if val_(C.DTT, rc, TTw(C))]; fcon = 0  # prune C_
    if C_:
        for n in [N for C in C_ for N in C.N_]:
            n.exe = (n.D if n.typ==1 else n.M) + np.sum([mo[0]-ave*mo[1] for mo in n.mo_]) - ave
            # exemplar V + sum n match_dev to Cs, m * ||C rvals?
        if val_(DTT, rc+olp, TTw(root), (len(C_)-1)*Lw) > 0:
            Ct = sum2T(C_,rc,root, nF='Ct')
            Ct.tNt, Ct.tBt, Ct.tCt = CF(),CF(),CF()
            # re-order C_ along argmax(root.wTT): eigenvector?
            _,rc = cross_comp(Ct, rc, fcon=2)  # distant Cs, different attr weights, no cross_comp dCs?
            root.C_=C_; root.Ct=Ct; root_update(root,Ct)
            fcon = 1
    return  fcon, rc

def merge_C_(_L_):  # for similar centroids only, they are not supposed to be local
        N_,xN_,L_ = [],[],[]
        _L_ = sorted( set(_L_), key=lambda link: link.d)  # from min D
        for i, L in enumerate(_L_):
            if val_(L.dTT, rc+nw, wTTf,fi=0) < 0:
                _N, N = L.nt
                if _N is N or N in xN_: continue  # not yet merged
                for n in N.N_: add_N(_N, n); _N.N_ += [n]
                for l in N.rim: l.nt = [_N if n is N else n for n in l.nt]
                N_ += [_N]; xN_+=[N]
                if N in N_: N_.remove(N)
            else: L_ = _L_[i:]; break
        if xN_: root.N_ = set(root.N_) - set(xN_);
        return list(set(N_)), L_

def cross_comp1(root, rc, fL=0):  # core function mediating recursive rng+ and der+ cross-comp and clustering, rc=rdn+olp

    N_ = (root.N_,root.B_)[fL]; nG_ = []
    iN_, L_,TT,c,TTd,cd = comp_N_(N_,rc) if N_[0].typ<3 else comp_C_(N_,rc, fC=1)  # nodes | centroids
    if L_:
        for n in iN_: n.em, n.ed = vt_(np.sum([l.dTT for l in n.rim],axis=0), rc)
        cr = cd/ (c+cd) *.5  # dfork borrow ratio, .5 for one direction
        fcon = fL or bool(root.root) or mdecay(L_)>decay  # conditional spec, must cluster B_?
        if val_(TT, rc+(centw,connw)[fcon], TTw(root), (len(L_)-1)*Lw,1, TTd,cr) > 0 or fL:
            root.L_ = L_
            sum2T(L_,rc,root,'Lt')  # new ders, root.B_,Bt if G
            E_ = get_exemplars({N for L in L_ for N in L.nt if N.em}, rc)  # exemplar N_| C_
            if fcon:
                nG_,rc = cluster_N(root, E_,rc, fL)  # form Bt, sub+ in sum2G
            else:  # centroid clustering if sub+ & dm/ddist, but cluster_C_par is global?
                nG_,rc = cluster_C(root, E_,rc)
        if nG_ and val_(root.dTT, rc+(cw,nw)[fcon], TTw(root), (len(root.N_)-1)*Lw,1, TTd,cr) > 0:
            nG_,rc = cross_comp(root,rc)  # agg+
    # nG_: recursion flag:
    return nG_,rc

class CN(CBase):
    name = "node"
    def __init__(n, **kwargs):
        super().__init__()
        n.typ = kwargs.get('typ', 0)
        # 0=PP: block trans_comp, etc?
        # 1= L: typ,nt,dTT, m,d,c,rc, root,rng,yx,box,span,angl,fin,compared, N_,B_,C_,L_,Nt,Bt,Ct from comp_sub?
        # 2= G: + rim, eTT, em,ed,ec, baseT,mang,sub,exe, Lt, tNt, tBt, tCt?
        # 3= C: + m_,o_,M,D,C, DTT?
        n.m,  n.d, n.c = kwargs.get('m',0), kwargs.get('d',0), kwargs.get('c',0)  # sum forks to borrow
        n.dTT = kwargs.get('dTT',np.zeros((2,9)))  # Nt+Lt dTT: m_,d_ [M,D,n, I,G,a, L,S,A]
        n.rim = kwargs.get('rim',[])  # external links, rng-nest?
        n.em, n.ed, n.ec = kwargs.get('em',0),kwargs.get('ed',0),kwargs.get('ec',0)  # sum dTT
        n.eTT = kwargs.get('eTT',np.zeros((2,9)))  # sum rim dTT
        n.rc  = kwargs.get('rc', 1)  # redundancy to ext Gs, ave in links?
        n.Nt, n.Bt, n.Ct, n.Lt = kwargs.get('Nt',CF()), kwargs.get('Bt',CF()), kwargs.get('Ct',CF()), kwargs.get('Lt',CF())
        # Fork tuple: [N_,dTT,m,d,c,rc,nF,root], N_ may be H: [N_,dTT] per level, nest=len(N_)
        n.baseT = kwargs.get('baseT',np.zeros(4))  # I,G,A: not ders, in links for simplicity, mostly redundant
        n.nt    = kwargs.get('nt', [])  # nodet, links only
        n.yx    = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.box   = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.span  = kwargs.get('span',1) # distance in nodet or aRad, comp with baseT and len(N_), not additive?
        n.angl  = kwargs.get('angl',[np.zeros(2),0])  # (dy,dx),dir, sum from L_
        n.mang  = kwargs.get('mang',1)  # ave match of angles in L_, =1 in links
        n.root  = kwargs.get('root',None)  # immediate
        n.sub = 0 # composition depth relative to top-composition peers?
        n.fin = kwargs.get('fin',0)  # clustered, temporary
        n.exe = kwargs.get('exe',0)  # exemplar, temporary
        n.rN_ = kwargs.get('rN_',[]) # reciprocal root nG_ for bG | cG, nG has Bt.N_,Ct.N_ instead
        n.compared = set()
        # ftree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    @property
    def N_(n): return n.Nt.N_[-1].N_ if (n.Nt.N_ and isinstance(n.Nt.N_[0],CF)) else n.Nt.N_
    @N_.setter
    def N_(n, n_):
        if n.Nt.N_ and isinstance(n.Nt.N_[0],CF): n.Nt.N_[-1].N_ = n_
        else: n.Nt.N_ = n_
    @property
    def L_(n): return n.Lt.N_[-1].N_ if (n.Lt.N_ and isinstance(n.Lt.N_[0],CF)) else n.Lt.N_
    @L_.setter
    def L_(n, n_):
        if n.Lt.N_ and isinstance(n.Lt.N_[0],CF): n.Lt.N_[-1].N_ = n_
        else: n.Lt.N_ = n_
    @property
    def B_(n): return n.Bt.N_[-1].N_ if (n.Bt.N_ and isinstance(n.Bt.N_[0],CF)) else n.Bt.N_
    @B_.setter
    def B_(n, n_):
        if n.Bt.N_ and isinstance(n.Bt.N_[0],CF): n.Bt.N_[-1].N_ = n_
        else: n.Bt.N_ = n_
    @property
    def C_(n): return n.Ct.N_[-1].N_ if (n.Ct.N_ and isinstance(n.Ct.N_[0],CF)) else n.Ct.N_
    @C_.setter
    def C_(n, n_):
        if n.Ct.N_ and isinstance(n.Ct.N_[0],CF): n.Lt.N_[-1].N_ = n_
        else: n.Ct.N_ = n_

class CF_:
    def __init__(s, attr): s.Ft = attr
    def oN_(s,o):
        ft = getattr(o, s.Ft)
        return (ft.N_[-1],'N_') if (ft.N_ and isinstance(ft.N_[0],CF)) else (ft,'N_')
    def __get__(s, o, c):
        if o is None: return s
        _ft,ft = s.oN_(o); return getattr(_ft,ft)
    def __set__(s, o, v):
        N_,Ft = s.oN_(o); setattr(N_,Ft, v)

class CN(CBase):
    name = "node"
    N_,L_,B_,C_ = CF_('Nt'), CF_('Lt'), CF_('Bt'), CF_('Ct')  # n.Ft.N_[-1] if n.Ft.N_ and isinstance(n.Ft.N_[-1],CF) else n.Ft.N_

    def __init__(n, **kwargs):
        super().__init__()
        n.typ = kwargs.get('typ', 0)
        n.Nt, n.Bt, n.Ct, n.Lt = (kwargs.get(fork,CF()) for fork in ('Nt','Bt','Ct','Lt'))

class CF_:  # to get fork N_
    def __init__(s, attr): s.Ft = attr
    def ret_N_(s, o):
        ft = getattr(o, s.Ft)
        return (ft.N_[-1], 'N_') if (ft.N_ and isinstance(ft.N_[-1], CF)) else (ft, 'N_')
    def __get__(s, o, c):
        if o is None: return s
        _ft, ft = s.ret_N_(o); return getattr(_ft, ft)
    def __set__(s, o, v):
        N_, Ft = s.ret_N_(o); setattr(N_, Ft, v)

def trans_comp(_N,N, rc, root):  # unpack node trees down to numericals and compare them

    for _F_,F_,nF_,nFt in zip((_N.N_,_N.B_,_N.C_),(N.N_,N.B_,N.C_), ('N_','B_','C_'),('Nt','Bt','Ct')):
        if _F_ and F_:  # eval?
            N_,dF_,TT,c,_,_ = comp_C_(_F_,rc, F_)  # callee comp_N may call deeper trans_comp, batch root_update?
            if dF_:  # match trans-links, !N_?
                setattr(root, nF_,dF_); setattr(root,nFt, CF(dTT=TT,c=c,root=root,nF=nFt))
                rc += 1  # default fork redundancy?
    tFt3_ = []  # form dtFt3H by comp tFt3H:
    for _Ft3, Ft3 in zip(_N.Lt.N_, N.Lt.N_):  # Lt.N_ is derH of tFt triples
        tFt3 = []  # dLay of trans-forks
        for _F_,F_,nFt in zip(_Ft3, Ft3, ('Nt','Bt','Ct')):
            if _F_ and F_:  # eval?
                _,dF_,TT,c,_,_= comp_C_(_F_,rc,F_)
                if dF_: tFt3 += [CF(N_=dF_,dTT=TT,c=c,root=root,nF=nFt)]; rc += 1
                else: tFt3 += [[]]
            else: tFt3 += [[]]
        if any(tFt3): tFt3_ += [tFt3]  # fixed tFt order
    if tFt3_: root.Lt.N_ += tFt3_  # extend Lt.N_ derH with ders of old Lays, not nested?
    # node H:
    for _lev,lev in zip(_N.Nt.N_, N.Nt.N_):  # no comp Bt,Ct: external to N,_N
        rc += 1  # deeper levels are redundant
        tt = comp_derT(_lev.dTT[1],lev.dTT[1]); m,d = vt_(tt,rc)
        oN_= set(_lev.N_) & set(lev.N_)  # intersect, or offset, or comp_len?
        dlev = CF(N_=oN_,nF='Nt',dTT=tt, m=m,d=d,c=min(_lev.c,lev.c),rc=rc,root=root)  # min: only shared elements are compared
        root.Nt.N_ += [dlev]  # root:link, nt.N_:derH
        root_update(root.Nt, dlev)  # root is Link

def add2F(Ft, nF, N, coef):  # unpack in sum2F?
    NH = N.Nt.N_ if isinstance(N.Nt.N_[0],CF) else [N.Nt.N_]  # nest if flat, replace Nt with Ft=nF?
    for Lev,lev in zip_longest(Ft.N_, NH, fillvalue=None):  # G.Nt.N_=H, top-down
        if lev:
            if Lev is None: Ft.N_ += [CopyT(lev, root=Ft)]
            else: Lev.N_ += lev.N_; Lev.dTT+=lev.dTT*coef; Lev.c+=lev.c*coef

def root_replace(root, rc, G_, N_,L_,Lt_,TT,nTT,lTT,C,nc,lc):

    root.dTT=TT; root.c=C
    root.rc =rc  # not sure
    if hasattr(root,'wTT'): cent_TT(root, root.rc)
    sum2F(L_,'Lt', root,1,lTT,lc)
    lTT,lc = np.zeros((2,9)),0  # reset for top nested lev
    for Lt in Lt_: lTT+=Lt.dTT; lc+=Lt.c
    m,d = vt_(lTT,rc)
    _lev_ = []  # existing levels
    if root.Nt.N_:
        if isinstance(root.Nt.N_[0], CF):   _lev_ = root.Nt.N_[:]  # existing levels
        elif isinstance(root.Nt.N_[0], CN): _lev_ = sum2F(root.Nt.N_,'Nt',root, fset=0)  # create level from Ns
    l0 = CF(N_=N_,dTT=lTT,m=m,d=d,c=lc, root=root)  # l0
    l1 = sum2F(G_,'Nt', root,nTT,nc,fset=0)  # l1
    root.Nt.N_ = [l1, l0] + _lev_

def cluster_N1(root, _N_, rc, fL=0):  # flood-fill node | link clusters, flat, replace iL_ with E_?

    def trans_cluster(root, iL_, rc):  # connectivity only?

        tN_,tB_,tC_ = [],[],[]  # splice trans-links from +ve links, skip redundant centroids?
        for Link in iL_: tN_+= Link.N_; tB_+= Link.B_; tC_+= Link.C_
        for tL_, nF in (tN_,'Nt'), (tB_,'Bt'), (tC_,'Ct'):
            if tL_:
                Ft = sum2F(tL_,nF,root.Lt,fset=0)  # updates root
                N_ = list({n for L in tL_ for n in L.nt})
                for N in N_: N.exe=1
                cluster_N(Ft, N_,rc)
                if val_(Ft.dTT, rc, TTw(root), (len(Ft.N_)-1)*Lw) > 0:
                    cross_comp(Ft, rc)  # unlikely, doesn't add rc?
                setattr(root, nF, Ft); rc+=1  # or append F_?
        ''' 
        if recursive trans_comp:
        for lev in zip_longest(Link.Nt.N_, Link.Bt.N_, Link.Ct.N_):  # each fork.N_ is H
            tF_ += [lev]  # Lt.N_ is trans-H 
        '''
    def rroot(n): return rroot(n.root) if n.root and n.root!=root else n
    def nt_vt(n,_n):
        M, D = 0,0  # exclusive match, contrast
        for l in set(n.rim+_n.rim):
            if l.m > 0:   M += l.m
            elif l.d > 0: D += l.d
        return M, D

    G_, N_,L_ = [],[],[]  # add prelink pL_,pN_? include merged Cs, in feature space for Cs
    if _N_ and (val_(root.dTT, rc+connw, TTw(root), mw=(len(_N_)-1)*Lw) > 0 or fL):
        for N in _N_: N.fin=0; N.exe=1  # not sure
        G_, L__= [],[]; TT,nTT,lTT = np.zeros((2,9)),np.zeros((2,9)),np.zeros((2,9)); C,nC,lC = 0,0,0; in_= set()  # root attrs
        for N in _N_: # form G per remaining N
            if N.fin or (root.root and not N.exe): continue  # no exemplars in Fg
            N_=[N]; L_,B_,C_=[],[],[]; N.fin=1  # init G
            __L_ = N.rim  # spliced rim
            while __L_:
                _L_ = []
                for L in __L_:  # flood-fill via frontier links
                    _N = L.nt[0] if L.nt[1].fin else L.nt[1]; in_.add(L)
                    if not _N.fin and _N in _N_:
                        m,d = nt_vt(*L.nt)
                        if m > ave * (rc-1):  # cluster nt, L,C_ by combined rim density:
                            N_ += [_N]; _N.fin = 1
                            L_ += [L]; C_ += _N.C_
                            _L_+= [l for l in _N.rim if l not in in_ and (l.nt[0].fin ^ l.nt[1].fin)]   # new frontier links, +|-?
                        elif d > avd * (rc-1):  # contrast value, exclusive?
                            B_ += [L]
                __L_ = list(set(_L_))
            if N_:
                Ft_ = ([list(set(N_)),np.zeros((2,9)),0,0], [list(set(L_)),np.zeros((2,9)),0,0], [list(set(B_)),np.zeros((2,9)),0,0], [list(set(C_)),np.zeros((2,9)),0,0])
                for i, (F_,tt,c,r) in enumerate(Ft_):
                    for F in F_:
                        tt += F.dTT; Ft_[i][2] += F.c
                        if i>1: Ft_[i][3] += F.rc  # define Bt,Ct rc /= ave node rc?
                (N_,nt,nc,_),(L_,lt,lc,_),(B_,bt,bc,br),(C_,ct,cc,cr) = Ft_
                c = nc + lc + bc*br + cc*cr
                tt = (nt*nc + lt*lc + bt*bc*br + ct*cc*cr) / c
                if val_(tt,rc, TTw(root), (len(N_)-1)*Lw) > 0 or fL:
                    G_ = [sum2G(((N_,nt,nc),(L_,lt,lc),(B_,bt,bc),(C_,ct,cc)), tt,c, rc,root)]  # calls sub+/ 3 forks, br,cr?
                    L__+=L_; TT+=tt; nTT+=nt; lTT+=lt; C+=c; nC+=nc; lC+=lc
                    # G.TT * cr * rcr?
        if G_ and (fL or val_(TT, rc+1, TTw(root), (len(G_)-1)*Lw)):  # include singleton lGs
            rc += 1
            root_replace(root,rc, TT,C, G_,nTT,nC,L__,lTT,lC)
        if G_:  # top root.N_
            tL_ = [tl for n in root.N_ for l in n.L_ for tl in l.N_]  # trans-links
            if sum(tL.m for tL in tL_) * ((len(tL_)-1)*Lw) > ave*(rc+connw):  # use tL.dTT?
                trans_cluster(root, tL_, rc+1)  # sets tTs
                mmax_ = []
                for F,tF in zip((root.Nt,root.Bt,root.Ct), (root.Lt.N_[-1])):
                    if F and tF:
                        maxF,minF = (F,tF) if F.m>tF.m else (tF,F)
                        mmax_+= [max(F.m,tF.m)]; minF.rc+=1  # rc+=rdn
                sm_ = sorted(mmax_, reverse=True)
                tNt, tBt, tCt = root.Lt.N_[-1]
                for m, (Ft, tFt) in zip(mmax_,((root.Nt,tNt),(root.Bt,tBt),(root.Ct,tCt))): # +rdn in 3 fork pairs
                    r = sm_.index(m); Ft.rc+=r; tFt.rc+=r  # rc+=rdn
    return G_, rc

def trans_cluster(root, rc):  # may create nested levs, re-order in sort_H?

    for lev in root.Lt.N_:  # Lt.N_ = H, nested in trans_comp
        for Ft, nF in [[getattr(lev,nF), nF] for nF in ('Nt','Bt','Ct')]:
            if Ft and isinstance(Ft.N_[0],CF):
                H = []
                for i, lev in enumerate(getattr(root,nF).N_[1:]): # deeper levs only
                    if isinstance(lev, CN):  # lev was converted to CN in trans_comp
                        _N_ = list({n for t in lev.N_ for n in t.nt})  # trans_links
                        for n in _N_: n.exe = 1
                        t_,rc = cluster_N(root,_N_,rc+i-1,nF=nF,lev=i); T_=[]  # or merge n.roots, this is not agg+?
                        if t_:
                            if val_(lev.dTT, rc, TTw(root), (len(t_)-1) * Lw) > 0:
                                T_,_ = cross_comp(lev, rc)  # nest, root_update
                            lev.N_ = T_ or t_  # no other changes?
                    H += [lev]; rc += 1

def slope(link_):  # get ave 2nd rate of change with distance in cluster or frame?

    Link_ = sorted(link_, key=lambda x: x.span)
    dists = np.array([l.span for l in Link_])
    diffs = np.array([l.d/l.c for l in Link_])
    rates = diffs / dists
    return (np.diff(rates) / np.diff(dists)).mean()  # ave d(d_rate) / d(unit_distance)

def comp_F_(_F_,F_,nF, rc, root):  # root is link, unpack node trees down to numericals and compare them

    L_,TTm,cm,TTd,cd = [],np.zeros((2,9)),0,np.zeros((2,9)),0
    if isinstance(F_[0],CN):
        for _N, N in product(F_,_F_):
            if _N is N: dtt = np.array([N.dTT[1], np.zeros(9)]); TTm += dtt; cm=1; cd=0  # overlap is pure match
            else:       cm,cd = comp_n(_N,N, TTm,TTd,cm,cd,rc,L_)
    else:
        for _lev,lev in zip(_F_,F_):
            rc += 1  # deeper levels are redundant
            tt = comp_derT(_lev.dTT[1],lev.dTT[1]); m,d = vt_(tt,rc)
            _sN_,sN_ = set(_lev.N_), set(lev.N_)
            iN_ = _sN_ & sN_; _oN_= _sN_-sN_; oN_= sN_-_sN_; dN_=[]  # intersect, offsets in lev
            for n in iN_: m += n.m  # all match, not in dN_?
            if _oN_ and oN_ and m > ave:  # offset comp:
                for _n,n in product(_oN_,oN_): comp_n(_n,n, TTm,TTd,cm,cd,rc, dN_); nm,nd= vt_(TTm,rc); m+=nm; d+=nd
            L_ += [CF(nF='tF',N_=dN_,dTT=tt,m=m,d=d,c=min(_lev.c,lev.c),rc=rc,root=root)]
            # L_ = H
    if L_: setattr(root,nF, sum2F(L_,nF,root,TTm,cm, fCF=0))  # root is Link or trans_link

def add_F(F,f, cr=1, merge=1):

    cc = F.c / f.c
    H = isinstance(F.N_[0],CF)  # flag for splicing
    if merge:
        if hasattr(F,'Nt'): merge_f(F,f, cc)
        else: F.N_.extend(f.N_)
    else: F.N_.append(f)
    F.dTT+=f.dTT*cc*cr; F.c+=f.c; F.rc+=cr  # -> ave cr?

def trans_cluster1(tFt, root, rc):  # trans_links mediate re-order in sort_H?

    def rroot(n): return rroot(n.root) if n.root and n.root != root else n  # root is nG
    FH_ = [[],[],[]]
    for tL in tFt.N_:  # splice from base links
        for FH,Ft in zip(FH_, (getattr(L,'tNt',[]),getattr(L,'tBt',[]),getattr(L,'tCt',[]))):  # trans_links
            if not Ft: continue
            for Lev,lev in zip_longest(FH, Ft.N_):  # always H?
                if lev:
                    if Lev: Lev += lev.N_  # concat for sum2F
                    else:   FH += [list(lev.N_)]
    # merge tL_ nt roots:
    for FH, nF in zip(FH_, ('tNt','tBt','tCt')):
        if FH:  # merge Lt.fork.nt.roots
            for lev in reversed(FH):  # bottom-up to get incrementally higher roots
                for tL in lev:  # trans_link
                    rt0 = tL.nt[0].root; rt1 = tL.nt[1].root
                    if rt0 is rt1: continue
                    merge = rt0 is root == rt1 is root  # else append
                    if not merge and rt0 is root: rt0,rt1 = rt1,rt0  # concat in higher G
                    add_N(rt0, rt1, merge)
            # set tFt:
            if not hasattr(root, nF): setattr(root, nF, CF(nF=nF))  # init root.tFt
            FH = [sum2F(n_,nF,getattr(root, nF)) for n_ in FH]; sum2F(FH, nF, root)

        ''' reval rc:
        tL_ = [tL for n in root.N_ for l in n.L_ for tL in l.N_]  # trans-links
        if sum(tL.m for tL in tL_) * ((len(tL_)-1)*Lw) > ave*(rc+connw):  # use tL.dTT?
            mmax_ = []
            for F,tF in zip((root.Nt,root.Bt,root.Ct), (root.Lt.N_[-1])):
                if F and tF:
                    maxF,minF = (F,tF) if F.m>tF.m else (tF,F)
                    mmax_+= [max(F.m,tF.m)]; minF.rc+=1  # rc+=rdn
            sm_ = sorted(mmax_, reverse=True)
            tNt, tBt, tCt = root.Lt.N_[-1]
            for m, (Ft, tFt) in zip(mmax_,((root.Nt,tNt),(root.Bt,tBt),(root.Ct,tCt))): # +rdn in 3 fork pairs
                r = sm_.index(m); Ft.rc+=r; tFt.rc+=r  # rc+=rdn
        '''
def root_replace(root, rc, TT,C, N_,nTT,nc,L_):

    root.dTT=TT; root.c=C; root.rc = rc  # not sure
    if hasattr(root,'wTT'): cent_TT(root, root.rc)
    sum2F(N_,'Nt',root, nTT,nc)
    sum2f(L_,'Lt',root)


def rroot(n):
    return rroot(n.root) if n.root and n.root != Ft else n  # root is nG

def F2N(f):

        F = CN(dTT=f.dTT, Lt=f.Lt, m=f.m, d=f.d, c=f.c, rc=f.rc, root=f.root, typ=4)
        F.nF = f.nF
        setattr(f.root, f.nF, F)  # CN Ft
        for N in f.N_: N.root = F; F.Nt.N_ += [N]  # the rest of Nt is redundant
        return F

def sum2F(N_,nF, root, TT=np.zeros((2,9)), C=0, Rc=0, fset=1, fCF=1):  # -> Ft

    H = []  # unpack,concat,resum existing node'levs, sum,append to new N_'lev
    for F in N_:  # fork-specific N_
        if not C: TT += F.dTT; C += F.c; Rc += F.rc
        if isinstance(F.Nt.N_[0],CF):  # H, top-down, eval sort_H?
            if H:
                for Lev,lev in zip_longest(H, reversed(F.Nt.N_)):  # align bottom-up
                    if lev and lev.N_:
                        if Lev: Lev += lev.N_
                        else:   H += [list(lev.N_)]
            else: H = [list(lev.N_) for lev in F.Nt.N_]
        elif H: H[0] += F.N_  # flat
        else:   H = [list(F.N_)]
    m,d = vt_(TT); rc = Rc/len(N_); Cx = (CN,CF)[fCF]
    Ft = Cx(dTT=TT,m=m,d=d,c=C,rc=rc,root=root)
    Ft.nF = nF  # splice N_ H:
    if H: Ft.N_ = [CF(N_=N_,nF=nF,dTT=TT,m=m,d=d,c=C,rc=rc,root=Ft)] + [sum2f(n_,nF,Ft) for n_ in reversed(H)]
    else: Ft.N_ = N_
    if fset: root_update(root, Ft)
    return Ft

def comp_F_(_F_,F_,nF, rc, root):  # root is nG, unpack node trees down to numericals and compare them

    L_,TTm,C,TTd,Cd = [],np.zeros((2,9)),0,np.zeros((2,9)),0; Rc=cc=0  # comp count
    if isinstance(F_[0],CN):
        for _N, N in product(F_,_F_):
            if _N is N: dtt = np.array([N.dTT[1], np.zeros(9)]); TTm += dtt; C=1; Cd=0  # overlap is pure match
            else:       cm,cd = comp_n(_N,N, TTm,TTd,C,Cd,rc,L_); C+=cm; Cd+=cd
            Rc+=_N.rc+N.rc; cc += 1
        if L_: sum2f(L_,nF,root)  # always flat tFt
    else:
        for _lev,lev in zip(_F_,F_):  # L_ = H, lev = [nt,ct]
            rc += 1  # deeper levels are redundant, also Cts?
            # for _ft,ft in zip(_lev,lev):
            lTT = comp_derT(_lev.dTT[1],lev.dTT[1]); lRc= lC= lcc= 1  # min per dTT?
            _sN_,sN_ = set(_lev.N_), set(lev.N_)
            iN_ = list(_sN_ & sN_)  # intersect = match
            for n in iN_: lTT+=n.dTT; lC+=n.c; lRc+=n.rc; lcc+=1
            _oN_= _sN_-sN_; oN_= sN_-_sN_; dN_= []
            for _n,n in product(_oN_,oN_):
                cm,_ = comp_n(_n,n, lTT,TTd,C,Cd,rc, dN_)  # comp offsets
                lRc += _n.rc+n.rc; lC+=cm; lcc+=1
            lRc /= len(dN_); m,d = vt_(lTT,lRc)
            L_ += [CF(N_=dN_,nF='tF',root=root,dTT=lTT,m=m,d=d,c=lC,rc=lRc)]
            TTm+= lTT; C+=lC; Rc+=lRc; cc+=lcc
        if L_:
            Rc/=cc; m,d=vt_(TTm,Rc); setattr(root,nF,CF(N_=L_,nF=nF,dTT=TTm,m=m,d=d,c=C,rc=Rc,root=root))

