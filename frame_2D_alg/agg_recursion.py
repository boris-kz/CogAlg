import numpy as np, inspect, contextvars
from copy import copy, deepcopy
from math import atan2, cos, pi  # from functools import reduce
from itertools import zip_longest, combinations, product  # from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, imread, comp_pixel, CBase
from slice_edge import slice_edge
from comp_slice import comp_slice, w_t
from functools import wraps

'''
This is a main module of open-ended clustering algorithm, designed to discover empirical patterns of indefinite complexity. 
Lower modules cross-comp and cluster image pixels and blob slices(Ps), the input here is resulting PPs: segments of matching Ps.
Cycles of (generative cross-comp, compressive clustering, filter-adjusting feedback) should form hierarchical model of input stream: 

Cross-comp forms Miss and Match (min: shared_quantity for directly predictive params, else inverse deviation of miss or variation), in 2 forks:
rng+: incremental-range cross-comp nodes: edge segments at < max distance, cluster if they match. 
der+: incremental-derivation cross-comp links, from node cross-comp, if abs_diff * relative_adjacent_match

Clustering compressively groups the elements into compositional hierarchy, initially by pair-wise similarity or density thereof.
High-contrast links are correlation clustered to form contours of adjacent node connectivity clusters.
Each cycle goes through <=4 incrementally fuzzy and parallelizable stages:

- select sparse exemplars to seed the clusters, top k for parallelization? (get_exemplars),
- connectivity/ density-based agglomerative clustering, followed by divisive clustering (cluster_N), 
- sequential centroid-based fuzzy clustering with iterative refinement, start in divisive phase (cluster_C),
- centroid-parallel frame refine by two-layer EM if min global overlap, prune for next cros_comp cycle (cluster_P).

That forms hierarchical graph representation: dual tree of down-forking elements: node_H, and up-forking clusters: root_H:
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
Similar to neurons: dendritic input tree and axonal output tree, but with lateral cross-comp and nested param sets per layer.

Overall fitness function is predictive value of the model, estimated through multiple orders of projection:
initially summed match, refined by external projection in space and time (where match is combined with directional diffs),
then by comparing projected to actual match, testing the accuracy of prior cross_comp and clustering process, etc. 
Higher projection orders should be generated recursively from lower-order cross-comp, per sufficient number of newly formed clusters.

Feedback: projected match adjusts filters to maximize next marginal match, with coord filters selecting new input (in ffeedback()),
to be refined by cross_comp of co-projected patterns: "imagination, planning, action" in part 3)   
This is similar to backprop but sparse, and the filters only control operations, they are not weights on input parameters.

Higher-order feedback should modify the code by adding weights on code elements according to their contribution to projected match.
Code weights, currently nw,cw,Nw,Cw,specw, control skipping / recursing over corresponding functions ( blocks ( operations.
Also cross-comp and cluster (compress) code elements and function calls, real and projected, though much more coarsely than data?

notation:
prefix  f denotes flag
postfix t denotes tuple, multiple ts is a nested tuple
prefix  _ denotes prior of two same-name vars, multiple _s for relative precedence
postfix _ denotes array of same-name elements, multiple _s is nested array
capitalized vars are summed small-case vars
'''
eps = 1e-7
def eps_(a): return np.where(a==0, eps, a)

def prop_F_(F):  # factory function to get and update top-composition fork.N_
    def get(N): return getattr(N,F).N_
    def set(N, new_N): setattr(getattr(N,F),'N_', new_N)
    return property(get,set)

class CN(CBase):
    name = "node"
    N_,C_,L_,B_,X_,rim = prop_F_('Nt'),prop_F_('Ct'),prop_F_('Lt'),prop_F_('Bt'),prop_F_('Xt'),prop_F_('Rt')  # ext|int -defined Ns,Ls
    def __init__(n, **kwargs):
        super().__init__()
        n.typ = kwargs.get('typ', 0)
        # 0= PP: block trans_comp, etc?
        # 1= L:  typ,nt,dTT, m,d,c,r, root,rng,yx,box,span,angl,fin,compared, Nt,Bt,Ct from comp_sub
        # 2= G:  + rim,kern,mang,sub,exe
        # 3= C | Cn: +rN_| m_,d_,r_,o_
        n.dTT, n.TTn, n.TTc = [np.copy(kwargs.get(x,np.zeros((2,9)))) for x in ('dTT','TTn','TTc')]  # +wTTC per Ct.dTT, all m_,d_ [M,D,n, I,G,a, L,S,A]
        n.wTT_ = kwargs.get('wTT_',[np.ones((2,9)) for _ in range(4)])  # dTT: Nt+Lt+Bt, TTC, TTn, TTc
        n.m, n.d, n.c, n.r = [kwargs.get(x,0) for x in ('m','d','c','r')]  # combined-fork vals
        n.Nt,n.Bt,n.Ct,n.Lt,n.Xt,n.Rt = ((kwargs.get(f) if f in kwargs else CF(root=n) for f in ('Nt','Bt','Ct','Lt','Xt','Rt'))) # CN if nest, Ct||Nt
        n.kern = kwargs.get('kern',np.zeros(4))  # I,G,A: not ders, in links for simplicity, mostly redundant
        n.span = kwargs.get('span',1)  # distance in nodet or aRad, comp with kern or len(N_)
        n.angl = kwargs.get('angl',[np.zeros(2),0])  # (dy,dx),dir, sum from L_
        n.mang = kwargs.get('mang',1)  # ave match of angles in L_, =1 in links
        n.box  = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.yx   = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.root = kwargs.get('root',None)  # immediate
        n.sub = 0  # composition depth relative to top-composition peers?
        n.exe = kwargs.get('exe',0)  # exemplar, temporary
        n.fin = kwargs.get('fin',0)  # clustered, temporary
        n.compared = set()
        n.nt = kwargs.get('nt',[])  # L nodet
        n.rN_= kwargs.get('rN_',[])  # reciprocal root nG_ for bG|cG, nG has Bt.N_,Ct.N_ instead?
        n.nF = kwargs.get('nF','Nt')  # to set attr in root_update
        n.fb_= kwargs.get('fb_',[])
        # ftree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def __bool__(n): return bool(n.c)

class CF(CBase): # iF/ data: rim, Nt,Ct, Bt,Lt: ext|int- defined nodes, ext|int- defining links, Lt/Ft, Ct/lev, Bt/G
    name = "fork"
    def __init__(f, **kwargs):
        super().__init__()
        f.N_ = kwargs.get('N_',[])  # flat top lev, calls in oF
        f.H  = kwargs.get('H', [])  # lower CF levs / Nt||Ct
        f.nF = kwargs.get('nF','Nt')
        f.Lt = kwargs.get('Lt',[])  # +|-/ cross_comp | data scope in oF?
        f.Ct = kwargs.get('Ct',[])  # promoted | generalized lower types, oF only?
        f.dTT = kwargs.get('dTT',np.zeros((2,9))); f.m, f.d, f.c, f.r = [kwargs.get(x,0) for x in ('m','d','c','r')]  # rdpTT in oF?
        f.wTT = kwargs.get('wTT',np.zeros((2,9))); f.w, f.wc, f.wr = [kwargs.get(x,0) for x in ('w','wc','wr')]  # or wT, fork coefs, no wd?
        f.fb_ = kwargs.get('fb_',[])
        f.typ = kwargs.get('typ',0)  # blocks sub_comp
        f.root = kwargs.get('root',None)
    def __bool__(f): return bool(f.c)  # N_ may be empty?

class CoF(CF):  # oF/ code fork, Ct: types, dTT=results, w=m or separate sum wTT?
    name = "func"
    _cur = contextvars.ContextVar('oF')
    _typ = {}  # func_name -> typ_oF singleton in Z.Ct
    def __init__(f, **kw):
        super().__init__(**kw)
        f.call_ = kw.get('call_',[])  # function call trace, H: nested call tree or clustered typs only?
    @staticmethod
    def get(): return CoF._cur.get(Z)
    @staticmethod
    def traced(func):
        if getattr(func, 'wrapped', False):
            return func
        @wraps(func)
        def inner(*a, **kw):
            name = func.__name__
            if name not in CoF._typ:
                t = CoF(nF=name); CoF._typ[name] = t  # not root in typ CoF until clustering?
                Z.Ct.N_ += [t]  # register type, or in ffeedback only?
            typ = CoF._typ[name]
            _CoF = CoF._cur.get()
            oF = CoF(nF=name, root=_CoF, typ=typ)  # call instance
            _CoF.call_ += [oF]  # add Q2R(call_) before evals? nest by data scope in frame.H[i].H?
            CoF._cur.set(oF); result = func(*a, **kw)
            CoF._cur.set(_CoF)  # restore
            return result
        inner.wrapped = True  # flag
        return inner
# Z:
# singletons, no wTT:
ave,avd = .3,.5; Ave,Avd = CF(nF='ave',w=ave), CF(nF='avd',w=avd)  # ave m,d / unit dist, oFs: updatable weights version
wY, wX = 64, 64; WY,WX = CF(nF='wY',w=wY), CF(nF='wX',w=wX)
decay, wYX = ave/(ave+avd), np.hypot(wY,wX)
aveB,Lw,distw,intw = 100,.5,.5,.5; AveB,LW,Distw,Intw = CF(nF='aveB',w=aveB), CF(nF='Lw',w=Lw), CF(nF='distw',w=distw), CF(nF='intw',w=intw)
# attr weight tree:
wM,wD,wi, wG,wI,wa, wL,wS,wA = 10, 10, 20, 20, 5, 20, 2, 1, 1  # dTT weights = reversed relative ave, update from wTT_ after feedback
wT = np.array([wM,wD,wi, wG,wI,wa, wL,wS,wA])
wN,wC,wn,wc = 10,20,5,10  # fork weights
wTTN, wTTC, wTTn, wTTc = np.array([wT,wT*avd])*wN, np.array([wT,wT*avd])*wC, np.array([wT,wT*avd])*wn, np.array([wT,wT*avd])*wc
wTT_ = [wTTN,wTTC, wTTn,wTTc]  # || Nt,Ct,Lt, no Bt: no call, no info?
''' call tree, draft:
    WN,WC, Wn,Wc = CF(nF='wN',wTT=wTTN,w=wN), CF(nF='wC',wTT=wTTC,w=wC), CF(nF='wn',wTT=wTTn,w=wn), CF(nF='wc',wTT=wTTc,w=wc)
    WTT_ = CF(nF='wTT_',Ct= CF(N_=[WN,WC,Wn,Wc]))  # add sub-forks
    Z.Ct.N_ = [  
        # cluster oFs by call,type, as in AST, if V: compression or rel prediction error, oF.wTT*= rpdTT, _oF.wTT/= rpdTT:
        CF(nF='comp_',Ct= CF(N_= [CF(nF='comp_N_'), CF(nF='comp_C_'), CF(nF='comp_N'), CF(nF='comp_F')])),  # finer comps downstream?
        CF(nF='clust',Ct= CF(N_= [CF(nF='exemplars'), CF(nF='cluster_N'), CF(nF='cluster_C'), CF(nF='cluster_P')])),  # Q2R,cent_TT, sum functions?
        CF(nF='eval_',Ct= CF(N_= [CF(nF='vt_'), CF(nF='val_'), CF(nF='proj_TT'), CF(nF='proj_N'), CF(nF='ffeedback')])),  # or ffeedback is a clust type?
        CF(nF='attr_',Ct= CF(N_= [Ave,Avd,WTT_, WY,WX, AveB, LW,Distw,Intw]))  # all modifiable weights
        ] 
    if  projecting_root: F.root.wTT *= rdpTT * (F.c/ F.root.c)  # c-weighted feedback
    elif selecting_root: (F.m - F.root.Lt.m) * (F.root.c-Ft.c)  # clustering value = loss reduction: root.Lt.m < selective F.m, add dval? 
'''
Z = CoF(nF ='Z', Ct=CoF(nF='Ct'))  # global meta code, Z.N_= call trace, root=caller, Ct: function sub-types in ffeedback
CoF._cur.set(Z)  # root context

def vt_(TT, r, wTT=wTTn):  # brief val_ to get m,d, rc=0 to return raw vals, Wn for comp_N

    m_,d_= TT; ad_ = np.abs(d_); t_ = eps_(m_+ad_)  # ~ max comparand
    return m_/t_ @ wTT[0] - ave*r, ad_/t_ @ wTT[1] - avd*r  # norm by co-derived variance

def val_(TT, r, wTT=wTTn, mw=1.0,fi=1, _TT=None, cr=.5):  # m,d eval per cluster, cr = cd / cm+dc, default half-weight?

    t_ = eps_(TT[0] + np.abs(TT[1]))  # comb val/attr, match can be negative?
    rv = TT[0] / t_ @ wTT[0] if fi else TT[1] / t_ @ wTT[1]  # fork / total per scalar
    if _TT is not None:
        _t_ = eps_(np.abs(_TT[0]) + np.abs(_TT[1]))
        _rv = _TT[0] / _t_ @ wTT[0] if fi else _TT[1] / _t_@ wTT[1]
        rv  = rv * (1-cr) + _rv * cr  # + borrowed alt fork val, cr: d count ratio, must be passed with _TT?
    return rv*mw - (ave if fi else avd) * r

def sum_vt(N_, fr=0, fm=0, wTT=wTTn):  # basic weighted sum of CN|CF list, wTTn for comp_N

    C = sum(n.c for n in N_); R = 0; TT = np.zeros((2,9))
    for n in N_:
        rc = n.c / C; TT += (n.rTT if fr else n.dTT)*rc; R += n.r*rc  # * weight
    return (*vt_(TT,R,wTT), TT,C,R) if fm else (TT,C,R)

def Q2R(N_, R=None, merge=1, froot=1, fN=0, fr=0):  # update root with N_

    if R is None: R = CN() if fN else CF()  # root
    R.m, R.d, R.dTT, R.c, R.r = sum_vt(N_, fr,fm=1)  # fr: val = local surprise
    if merge:
        for N in [n for N in N_ for n in N.N_] if merge==3 else (N_[1].N_ if merge == 2 else N_):  # 2: pair merge
            R.N_ += [N]
            if froot: N.root=R
            if fN: R.C_ += N.C_; R.TTn += N.TTn; R.TTc += N.TTc  # frame_H only?
    if fN and R.C_: R.Ct = Q2R(R.C_)
    return R

def TTw(G, wi=0): return wTT_[wi] * G.wTT_[wi]  # * secondary wweights, average=1
''' 
  Agg cycle:
- Cross-comp nodes, evaluate incremental-derivation cross-comp of new >ave difference links, recursively. 

- Connectivity-cluster select exemplars/centroids by >ave match links, correlation-cluster links by >ave diff,
- Form complemented (core+contour) clusters -> divisive sub-clustering, higher-composition cross_comp. 

- Centroid-cluster from exemplars, spreading via their rim within a frame, increasing overlap between centroids.
- Switch to globally parallel EM-like fuzzy centroid refining if overlap makes lateral expansion less efficient.

- Forward: extend cross-comp and clustering of top clusters across frames, re-order centroids by eigenvalues.
- Feedback coords to bottom level or prior-level in parallel pipelines, filter updates in more coarse cycles 
'''

def cross_comp(Ft, rr, nF='Nt'):  # core function mediating recursive rng+ and der+ cross-comp and clustering

    N_,G_ = Ft.N_,[]; fC = N_[0].typ==3  # rc=rdn+olp, comp N_|B_|C_:
    L_, TT,c,r,TTd,cd,rd = comp_C_(N_,rr,fC=1) if fC else comp_N_(combinations(N_,2),rr)
    if L_:  # Lm_, no +|- Ft.Lt?
        M, D = vt_(TT, r, TTw(Ft.root,fC+2))
        if M * ((len(L_)-1)*Lw) * wn > ave:  # global cw,nw,Nw,Cw / ffeedback
            # vs TTn = root.Lt.dTT: +ve links between initial nodes, does not represent -ves?
            setattr( Ft.root,('TTn','TTc')[fC], TT+TTd)  # comp->TT, clust->dTT, compression = TT-dTT, in sum2G
            E_ = get_exemplars({N for L in L_ for N in L.nt}, r)
            G_,r = cluster_N(Ft, E_, r)  # -> cluster_C, _P
            if G_:
                Ft = sum2F(G_, nF, Ft.root); fC = G_[0].N_[0].typ==3  # not sure, G.N_= spliced C_ if more valuable?
                if val_(TT*wn, r, TTw(Ft.root,fC+2), (len(G_)-1)*Lw,1,TTd, rd/(r+rd)) > 0:
                    G_,r = cross_comp(Ft,r,nF)  # agg+, trans-comp
    return G_, r  # G_ is recursion flag

def comp_N_(_pairs, r, tnF=None, root=2):  # incremental-distance cross_comp, max dist depends on prior match

    pairs, TT,cm,rm = [],np.zeros((2,9)), 0,0
    for pair in _pairs:  # get all-to-all pre-links
        _N, N = pair
        if _N.sub != N.sub: continue  # or comp x composition?
        if N is _N:  # overlap = unit match, no miss, or skip?
            tt = np.array([N.dTT[1],np.zeros(9)]); TT+=tt; c=min(N.c,_N.c); cm+=c; rm+=(N.r+_N.r)/2 *c
        else:
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx)
            pairs += [[dist, dy_dx,_N,N]]
    if not pairs: return 0,0,0,r,0,0,0

    def proj_V(_N,N, dist, dy_dx, dec):  # _N x N induction
        Dec = dec or decay ** ((dist/((_N.span+N.span)/2)))
        iTT = (_N.dTT + N.dTT) * Dec
        eTT = (_N.Rt.dTT + N.Rt.dTT) * Dec
        if abs(vt_(eTT,r)[0]) > ave*wn*r*(_N.r+N.r)/2:  # spec N links
            eTT = proj_N(N, dist, dy_dx, r, dec)  # proj N L_,B_,rim, if pV>0: eTT += pTT?
            eTT+= proj_N(_N,dist, -dy_dx, r, dec)  # reverse direction
        return iTT+eTT

    N_,L_,TTd,cd,rd = [],[],np.zeros((2,9)),0,0  # any global use of dLs, rd?
    acc = [TT,cm,rm, TTd,cd,rd]
    for pL in sorted(pairs, key=lambda x: x[0]):  # proximity prior, test compared?
        dist,dy_dx,_N,N = pL # rim angl is not canonic
        pTT = proj_V(_N,N, dist, dy_dx, root.m if root!=2 else decay** (dist/((_N.span+N.span)/2)))  # based on current rim
        lr = r+ (N.r+_N.r)/2; m,d = vt_(pTT,lr)  # +|-match certainty
        if m > 0:
            if abs(m) < ave*wn:  # comp if marginally predictable, update N.Rt pair eval, ave / proj surprise value?
                Link = comp_N(_N,N, lr, full=not tnF, A=dy_dx, span=dist, rL=root,L_=L_,N_=N_, acc=acc)
                Link.rTT = np.abs(pTT - Link.dTT) / eps_(Link.dTT)  # relative prediction error to fit oF, direction-agnostic
            else:
                pL = CN(typ=-1, nt=[_N,N], dTT=pTT,m=m,d=d,c=min(N.c,_N.c),r=lr, angl=np.array([dy_dx,1],dtype=object),span=dist)
                L_+= [pL]; N.rim+=[pL]; _N.rim += [pL]; N_+=pL.nt; acc[0]+=pTT*pL.c; acc[1]+=pL.c; acc[2]+=pL.r*pL.c  # all +ve
                # no oF val, ~= links in clustering
        else: break  # beyond initial induction range, re-sort by proj_V?
    for N in set(N_):
        if N.rim: Q2R(N.rim, N.Rt, merge=0, froot=0)
    # call trace:
    L_ = [n for n in L_ if n.typ>-1]
    if L_: Q2R(L_, R=CoF.get(),fr=1)  # skip pLs, oF.call_+=[oF], oF.nF='comp_N_', adds data
    TT,cm,rm, TTd,cd,rd = acc
    return L_,TT,cm,rm/(cm or eps), TTd,cd,rd/(cd or eps)

def comp_C_(C_, rr,_C_=[], fall=1, fC=0):  # simplified for centroids, trans-N_s, levels
    N_,L_ = [],[]; acc = [np.zeros((2,9)),0,0,np.zeros((2,9)),0,0]
    if fall:
        pairs = product(C_,_C_) if _C_ else combinations(C_,r=2)  # comp between | within list
        for _C, C in pairs:
            if _C is C:
                dtt = np.array([C.dTT[1],np.zeros(9)]); acc[0]+=dtt; acc[1]+=1; acc[2]+=1  # overlap=match
            else:
                dy_dx = _C.yx-C.yx; dist = np.hypot(*dy_dx)
                comp_N(_C,C, rr, A=dy_dx,span=dist,L_=L_,N_=N_,acc=acc)
        if fC:
            # merge similar distant centroids, they are non-local
            L_ = sorted(L_, key=lambda dC: dC.d); _C_= C_; C_, merg_ = [],[]  # from min D
            for i, L in enumerate(L_):
                if val_(L.dTT, rr+wn, wTTc,fi=0) < 0:
                    C0,C1 = L.nt
                    if C0 is C1 or C1 in merg_ or C0 in merg_: continue  # not merged
                    Q2R([C0,C1], C0,merge=2)
                    add_Nt(C0, C1.Nt); merg_ += [C1]
                    for l in C1.rim: l.nt = [C0 if n is C1 else n for n in l.nt]
                else: L_ = L_[i:]; break
            if merg_: N_ = list(set(N_) - set(merg_))
    else:
        # adjacent | distance-constrained cross_comp along eigenvector: sort by max attr, or original yx in sub+, proj C.L_?
        for C in C_: C.compared=set()
        C_ = sorted(C_, key=lambda C: C.dTT[0][np.argmax(wTTc[0])])  # max weight defines eigenvector
        for j in range( len(C_)-1):
            _C = C_[j]; C = C_[j+1]
            if _C in C.compared: continue
            dy_dx = _C.yx-C.yx; dist = np.hypot(*dy_dx)
            comp_N(_C,C, rr,A=dy_dx,span=dist,L_=L_,N_=N_,acc=acc)  # simplified for typ=3
    for N in list(set(N_)):
        if N.rim: Q2R(N.rim, N.Rt, merge=0, froot=0)
    if L_: Q2R(L_,R=CoF.get())  # 0 projection?
    TTm,cm,rm,TTd,cd,rd = acc
    return L_,TTm,cm, rm/(cm or eps), TTd,cd, rd/(cd or eps)  # L_ is Lm now

def comp_N(_N,N, r, full=1, A=np.zeros(2),span=None, rL=None, L_=None, N_=None, acc=None):

    def comp_H(_Nt,Nt, Link):  # tentative pre-comp
        dH, tt, C,R = [],np.zeros((2,9)),0,0
        for _lev, lev in zip([_Nt]+_Nt.H, [Nt]+Nt.H):  # should be top-down
            ltt = comp_derT(_lev.dTT[1],lev.dTT[1])
            lc = min(_lev.c,lev.c); lr = (_lev.r+lev.r)/2; m,d = vt_(ltt,lr)
            dH += [CF(dTT=ltt,m=m,d=d,c=lc,r=lr,root=Link)]
            tt += ltt*lc; C+=lc; R+=lr*lc
        return dH,tt,C, (r* Link.c+R)/ (Link.c+C)  # same norm for tt?

    TT= base_comp(_N,N)[0] if full else comp_derT(_N.dTT[1],N.dTT[1]); m,d = vt_(TT,r)
    L = CN(typ=1, nt=[_N,N], dTT=TT, m=m, d=d, c=min(N.c,_N.c), r=r, root=rL, exe=1)
    if N.typ > 0 and m > ave*wn:
        dH,htt,C,R = comp_H(_N.Nt, N.Nt, L); r=R  # tentative subcomp
        if m + vt_(htt,R/C)[0] > ave*wn:  # subcomp -> tFs in L:
            if abs(N.typ) ==1:
                for _n,n in product(_N.nt,N.nt): L.Nt.fb_ += [comp_N(_n,n,r)]  # link sub-comp
            else:
                for i,(_Ft,Ft,tnF) in enumerate(zip((_N.Nt,_N.Lt,_N.Bt,_N.Ct),(N.Nt,N.Lt,N.Bt,N.Ct),('Nt','Lt','Bt','Ct'))):
                    if _Ft and Ft:  # sub-comp
                        dFt = comp_F(_Ft,Ft, r,L); getattr(L,tnF).fb_ += dFt.N_  # tFt feedback
                        r = (i or 1) -1  # Nt,Lt are core, not redundant
            for fb_, nF in zip((L.Nt.fb_,L.Lt.fb_,L.Bt.fb_,L.Ct.fb_), ('Nt','Lt','Bt','Ct')):
                if fb_: Ft = Q2R(fb_,getattr(L,nF)); Q2R([L,Ft], L,merge=0)
                    # +=trans-links, python-batched bottom-up
    if full:
        if span is None: span = np.hypot(*_N.yx - N.yx)
        yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx
        box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
        angl = [A, np.sign(TT[1] @ wTTn[1])]
        L.yx=yx; L.box=box; L.span=span; L.angl=angl; L.kern=(_N.kern+N.kern)/2
    for n, _n in (_N,N),(N,_N):
        n.rim += [L]; n.compared.add(_n)  # or unique comps?
    if L_ is not None:
        N_+= [_N,N]  # for sum_vt(rim)
        if   L.m > ave: acc[0]+=L.dTT*L.c; acc[1]+=L.c; acc[2]+=L.r*L.c; L_ += [L];
        elif L.typ==1 and L.d > avd: acc[3]+=L.dTT*L.c; acc[4]+=L.c; acc[5]+=L.r*L.c  # no pLs?
    # add call trace?
    return L

def comp_F(_F, F, ir=0, rL=None):

    ddTT = comp_derT(_F.dTT[1],F.dTT[1]); r=(_F.r+F.r)/2; m,d = vt_(ddTT,r); r+=ir
    dF = CF(dTT=ddTT,m=m,d=d,c=min(_F.c,F.c),r=r)
    if _F.nF == F.nF:  # sub-comp
        _N_,N_=_F.N_,F.N_; nF=F.nF; f = nF =='Lt'
        if  _N_ and N_:
            if f: Np_ = [[_n,n] for _n,n in zip(_N_,N_) if _n and n]  # same forks
            else: Np_ = list(product(_N_,N_))  # pairs
            if (rL.m+m)/2 * ((len(Np_)-1)*Lw) > ave * wn:
                if f: L_= [comp_F(*Np, r,rL=dF) for Np in Np_]; TT,C,R = sum_vt(L_)
                else: L_,TT,C,R,_,_,_= comp_N_(Np_,r,nF,rL)
                if L_:
                    Q2R([dF,CF(N_=L_,dTT=TT,c=C,r=R)], dF, merge=2)
                    Q2R([rL,dF], rL, merge=0)
    return dF  # no cross-fork N_, no L ext updates?

def base_comp(_N,N):  # comp Et, kern, extT, dTT

    _M,_D,_C = _N.m,_N.d,_N.c; _I,_G,_Dy,_Dx =_N.kern; _L = len(_N.N_)  # N_: density within span
    M, D, C = N.m, N.d, N.c; I, G, Dy, Dx = N.kern; L = len(N.N_)
    rn = C/_C
    _pars = np.array([_M*rn,_D*rn,_C,_I*rn,_G*rn, [_Dy,_Dx],_L*rn,_N.span], dtype=object)  # Et, kern, extT
    pars  = np.array([M,D,C, (I,wI),G, [Dy,Dx], L,(N.span,wS)], dtype=object)
    mA,dA = comp_A(_N.angl[0]*_N.angl[1], N.angl[0]*N.angl[1])
    m_,d_ = comp(_pars,pars, mA,dA)  # M,D,n, I,G,a, L,S,A
    dm_,dd_ = comp_derT(_N.dTT[1], N.dTT[1])

    return np.array([m_+dm_,d_+dd_]), rn  # or rm, rv?
'''
    if np.hypot(*_N.angl[0])*_N.mang + np.hypot(*N.angl[0])*N.mang > ave*wA:  # aligned L_'As, mang *= (len_H)+fi+1
    mang = (rn*_N.mang + N.mang) / (1+rn)  # ave, weight each side by rn
    align = 1 - mang* (1-mA)  # in 0:1, weigh mA 
'''
def comp_derT(_i_,i_):

    d_ = _i_ - i_  # both arrays, no angles or massless params
    m_ = np.minimum( np.abs(_i_), np.abs(i_))
    m_[(_i_<0) != (i_<0)] *= -1  # negate opposite signs in-place
    return np.array([m_,d_])

def comp(_pars, pars, meA,deA):  # compute m_,d_ from inputs or derivatives

    m_,d_ = [],[]
    for _p, p in zip(_pars, pars):
        if isinstance(_p, list):  # vector angle
            mA, dA = comp_A(_p, p)
            m_ += [mA]; d_ += [dA]
        elif isinstance(p, tuple):  # massless: I|S avd in p only
            p, avd = p
            d = _p - p
            m_ += [avd - abs(d)]  # + complement max(avd,ad)?
            d_ += [d]
        else:  # massive
            _a,a = abs(_p), abs(p)
            m_ += [min(_a,a) if (_p<0)==(p<0) else -min(_a,a)]  # + complement max(_a,a) for +ves?
            d_ += [_p - p]
    # for general mass in 0:1, m = dir_m * mass + inv_m * (1-mass)?
    return np.array(m_+[meA]), np.array(d_+[deA])

def comp_A(_A,A):

    dA = atan2(*_A)- atan2(*A)  # * direction
    if   dA > pi: dA -= 2 * pi  # rotate CW
    elif dA <-pi: dA += 2 * pi  # rotate CCW
    '''  or 
    den = np.hypot(*_A) * np.hypot(*A) + eps
    mA = (_A @ A / den +1) / 2  # cos_da in 0:1, no rot = 1 if dy * _dx - dx * _dy >= 0 else -1  # +1 CW, −1 CCW, ((1-cos_da)/2) * rot?
    '''
    return (cos(dA)+1) /2, dA/pi  # mA in 0:1, dA in -1:1, or invert dA, may be negative?

def get_exemplars(N_,rr):  # multi-layer non-maximum suppression -> sparse seeds for diffusive clustering, cluster_N too?

    N_= sorted(N_, key=lambda n: n.Rt.m*n.c, reverse=True); E_,Inh_ = set(),set()
    # strong-first
    for rdn, N in enumerate(N_, start=1):
        inh_ = list(Inh_ & set(N.rim))  # stronger Es in N.rim
        oM = sum_vt(inh_,fm=1)[0] if inh_ else 0
        oV = oM / (N.Rt.m or eps)  # relative olp V
        if N.Rt.m * N.c > ave * (rr+rdn+wn + oV):
            E_.add(N); N.exe = 1  # in point cloud of focal nodes
            Inh_.update(set(N.rim))  # extend inhibition zone
        else:
            break  # the rest of N_ is weaker, trace via rims
    return E_ or [N_[0]]

def cluster_N(Ft, _N_, r):  # flood-fill node | link clusters, flat, replace iL_ with E_?

    def nt_vt(n,_n):
        M, D = 0,0  # exclusive match, contrast
        for l in set(n.rim+_n.rim):
            if l.m > 0:   M += l.m
            elif l.d > 0: D += l.d
        return M, D
    def trans_cluster(G):
        for L in G.L_:
            for tFt in L.Nt,L.Bt,L.Ct:  # Lt doesn't form trans-links, Ct is not root-constrained?
                for tL in tFt.N_:
                    if tL.m > ave*wN:  # merge trans_link.nt.roots
                        rt0 = getattr(tL.nt[0].root,'root',None); rt1 = getattr(tL.nt[1].root,'root',None)  # CNs
                        if rt0 and rt1 and rt0 !=rt1: add_Nt(rt0, rt1, merge=1)  # concat in higher G
            L.Nt,L.Bt,L.Ct = CF(),CF(),CF()
            # merge roots
    G_ =[]  # add prelink pL_,pN_? include merged Cs, in feature space for Cs
    if _N_ and val_(Ft.dTT, r+wN, TTw(Ft.root), mw=(len(_N_)-1)*Lw) > 0:  #| fL?
        for N in _N_:
            N.fin=0; N.exe=1; Q2R(N.rim, N.Rt, merge=0, froot=0)  # only if N was added in trans-cluster?
        G_= []; TT=np.zeros((2,9)); C=0; in_= set()  # root attrs
        for N in _N_:  # form G per remaining N
            if N.fin or (Ft.root.root and not N.exe): continue  # no exemplars in Fg
            N_ = [N]; L_,B_,C_ = [],[],[]; N.fin=1  # init G, no C_?
            __L_= N.rim  # spliced rim
            while __L_:
                _L_ = []
                for L in set(__L_) - in_:  # flood-fill via frontier links
                    _N = L.nt[0] if L.nt[1].fin else L.nt[1]; in_.add(L)
                    if not _N.fin and _N in Ft.N_:
                        m,d = nt_vt(*L.nt)
                        if m > ave * (r-1):  # cluster nt, L,C_ by combined rim density:
                            N_ += [_N]; _N.fin = 1
                            L_ += [L]; C_ += _N.rN_
                            _L_+= [l for l in _N.rim if l not in in_ and (l.nt[0].fin ^ l.nt[1].fin)]   # new frontier links, +|-?
                        elif d > avd * (r-1): B_ += [L]  # contrast value, exclusive?
                __L_ = list(set(_L_))
            if N_:
                ft_ = []
                for i,(F_,nF) in enumerate(zip((N_,L_,B_),('Nt','Lt','Bt'))):  # no Ct till sub+?
                    F_ = list(set(F_)) or []; tt,fc,fr = sum_vt(F_) if F_ else (np.zeros((2,9)),0,0)
                    ft_+= [[F_,nF,tt,fc,fr]]
                (_,_,nt,nc,_),(_,_,lt,lc,_),(_,_,bt,bc,br) = ft_
                c = nc + lc + bc*br  # redundant B_,C_
                tt = (nt*nc + lt*lc + bt*bc*br) /c  # tentative
                if val_(tt, r, TTw(Ft.root), (len(N_)-1)*Lw) > 0:
                    G_ += [sum2G(ft_, Ft)]; TT+=tt; C+=c  # tt*(C*r)? sub+ in sum2G
        if G_:
            for G in G_: trans_cluster(G)  # splice trans_links, merge L.nt.roots
            if val_(TT, r+1, TTw(Ft.root), (len(G_)-1)*Lw) > 0:
                sum2F(G_, Ft.nF, Ft.root, TT,C)  # reform Nt, Ft.Lt is empty till cross_comp?
                r += 1
        # clust_val = selectivity vs root.Lt: all comps, add dval?
        sel_TT = (Ft.dTT*Ft.c - Ft.root.Lt.dTT*Ft.root.Lt.c) / eps_(Ft.root.Lt.dTT * Ft.root.Lt.c)
        oF = CoF.get(); oF.N_=G_; oF.dTT=sel_TT; oF.c = Ft.root.c - Ft.c  # add data
        # combine C_:
        Q2R([C for N in (G_ if G_ else _N_) for C in N.Ct.N_], Ft.root.Ct, froot=0)
    return G_, r

def cluster_C(Ft, E_, r):  # form centroids by clustering exemplar surround via rims of new member nodes, within root

    C_,_C_ = [],[]  # form root.Ct, may call cross_comp-> cluster_N, incr rc
    for n in Ft.N_: n._C_,n.m_,n._m_,n.o_,n._o_,n.rN_ = [],[],[],[],[],[]
    for E in E_:
        C = Copy_(E,Ft,init=1,typ=3)  # all rims in root, sequence along eigenvector?
        for TT, wTT, ww in zip([C.dTT,C.Ct.dTT,C.TTn,C.TTc], C.wTT_, [wN,wC,wn,wc]):
            wTT[:] = cent_TT(TT,r) * ww
        C._N_ = list({n for l in E.rim for n in l.nt if (n is not E and n in Ft.N_)})
        C._L_ = set(E.rim)  # init peer links
        for n in C._N_+C.N_: n._m_+=[C.m*(n.c/C.c)]; n._o_+=[1]; n._C_+=[C]
        _C_ += [C]
    oC_ = []  # output stable Cs
    while True:  # reform C_, add direct in-C_ cross-links for membership?
        C_,cnt,olp, mat,dif, DTT,Dm,Do = [],0,0,0,0,np.zeros((2,9)),0,0; Ave = ave*(r+wn); Avd = avd*(r+wn)
        _Ct_ = [[c, c.m/c.c, c.r] for c in _C_]
        for cr, (_C,_m,_o) in enumerate(sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True),start=1):
            if _m > Ave *_o:
                L_, N_,N__,m_,o_,M,D,O,cc, dTT,dm,do = [],[],[],[],[],0,0,0,0, np.zeros((2,9)),0,0  # /C
                for n in set(_C.N_+_C._N_):  # current + frontier
                    dtt,_ = base_comp(_C, n); cc+=1  # or comp_N, decay?
                    m,d = vt_(dtt, cr, wTTC); dTT += dtt; m *= n.c  # rm,olp / C
                    odm = np.sum([_m-m for _m in n._m_ if _m>m])  # higher-m overlap
                    oL_ = set(n.rim) & _C._L_  # replace peer rim overlap with more precise m
                    if oL_: _m,_d = sum_vt(oL_,fm=1)[:2]; m+=_m; d+=_d  # abs m?
                    m_+=[m]; o_+= [odm]  # from all comps
                    M += m; D += abs(d)
                    if m > 0 and m > Ave * odm:
                        N_+=[n]; L_+=n.L_; O+=odm  # convergence val
                        for _n in [_n for l in n.rim for _n in l.nt if _n is not n]:
                            if not hasattr(_n,'_m_'): _n._C_,_n.m_,_n._m_,_n.o_,_n._o_,_n.rN_ = [],[],[],[],[],[]
                            N__ += [_n]  # +|-Ls
                        if _C not in n._C_: dm+=m; do+=odm  # not in extended _N__
                    else:
                        if _C in n._C_: i = n._C_.index(_C); dm+=n._m_[i]; do+=n._o_[i]
                DTT+=dTT; mat+=M; dif+=D; olp+=O; cnt+=cc
                if M > Ave*O and val_(dTT, cr+O, TTw(_C,1),(len(N_)-1)*Lw) > 0:  # dTT * wTTC is more precise?
                    C = sum2C(N_, _C, _i=None, root=Ft)
                    for n,m,o in zip(N_,m_,o_):
                        n.rN_ += [C]; n.m_+=[m]; n.o_+=[o]
                    C._N_ = list(set(N__)-set(N_))  # new frontier
                    C._L_ = set(L_)  # peer links
                    if D< Avd*O: oC_+= [C]  # output if stable, actually if val_(DTT, fi=0) + D?
                    else:        C_ += [C]  # reform
                else:
                    for n in _C._N_+_C.N_:
                        n.exe = n.m/n.c > 2 * ave
                        for i, c in enumerate(n.rN_):
                            if c is _C:  # remove _C-mapping m,o:
                                n.rN_.pop(i); n.m_.pop(i);n.o_.pop(i); break
                Dm+=dm; Do+=do
            else: break  # the rest is weaker
        for n in Ft.N_:
            n._C_ = n.rN_; n._m_= n.m_; n._o_= n.o_; n.rN_,n.m_,n.o_ = [],[],[]  # new n.Ct.N_s, combine with v_ in Ct_?
        if oC_+C_ and mat*dif*olp > ave*wC*2:  # if val_(DTT,len(oC_+C_)?
            oC_ = cluster_P(oC_+C_, Ft.N_, Ft)  # refine all memberships in parallel by global backprop|EM
            break
        if Do and Dm/Do > Ave: _C_=C_  # dval vs. dolp: overlap increases with Cs expansion
        else: oC_ += C_; break  # converged
    if oC_:
        for n in [N for C in oC_ for N in C.N_]:  # exemplar V + sum n match_dev to Cs, m* ||C rvals:
            n.exe = (n.d if n.typ==1 else n.m) + np.sum([m-ave*o for m, o in zip(n.m_, n.o_)]) - ave
        if val_(DTT, r+olp, TTw(Ft.root,1), (len(oC_)-1)*Lw) > 0:
            Ct = sum2F(oC_,'Ct', Ft.root, fCF=0)
            _, r = cross_comp(Ct,r)  # all distant Cs, seq C_ in eigenvector = argmax(root.wTT)?
            sel_TT = (Ct.dTT*Ct.c - Ft.dTT*Ft.c) / eps_(Ct.dTT * Ct.c)
            oF = CoF.get(); oF.N_=C_; oF.dTT=sel_TT; oF.c = Ft.c-Ct.c  # data, select in Nt.N_?
    return oC_,r

def cluster_P(__C_,N_,root):  # Parallel centroid refining, _C_ from cluster_C, N_= root.N_, if global val*overlap > min

    for N in N_: N._m_,N._o_,N._d_,N._r_ = [],[],[],[]
    for C in __C_: C.rN_= N_  # soft assign all Ns per C
    _C_=__C_; cnt = 0
    while True:
        M = O = dM = dO = 0
        for N in N_:
            N.m_,N.d_,N.r_,N.rN_ = map(list,zip(*[vt_(base_comp(C,N)[0],C.r,wTTC) + (C.r,C) for C in _C_]))  # distance-weight match?
            N.o_ = list(np.argsort(np.argsort(N.m_)[::-1])+1)  # rank of each C_[i] = rdn of C in C_
            dM+= sum(abs(m-_m) for _m,m in zip(N._m_,N.m_)) if cnt else sum(N.m_)
            dO+= sum(abs(o-_o) for _o,o in zip(N._o_,N.o_)) if cnt else sum(N.o_)
            M += sum(N.m_); O += sum(N.o_)
        C_ = [sum2C(N_,_C, i, root=root) for i, _C in enumerate(_C_)]  # update with new coefs, _C.m, N->C refs
        Ave = ave * (root.r+wC)
        cnt += 1
        if M > Ave*O and dM > Ave*dO:  # strong update
            for N in N_: N._m_,N._o_,N._d_,N._r_ = N.m_,N.o_,N.d_,N.r_
            _C_ = C_
        else: break
    out_ = []
    for i, C in enumerate(C_):
        keep = C.m > ave*(wC+C.r)
        if keep:
            C.N_ = [n for n in C.N_ if n.m_[i] * C.m > ave * n.r_[i] * n.o_[i]]
            keep = bool(C.N_)
        out_ += [C if keep else []]
    for N in N_:
        for _v_,v_ in zip((N._m_,N._d_,N._r_,N._o_), (N.m_,N.d_,N.r_,N.o_)):
            v_[:] = [v for v,c in zip(v_,out_) if c]; _v_[:] = []
        N.rN_ = [c for c,keep in zip(N.rN_,out_) if keep]

    C_ = [c for c in out_ if c]
    dCt= Q2R(set(__C_)-set(C_))  # compression
    oF = CoF.get(); oF.N_=C_; oF.dTT=dCt.dTT; oF.c = dCt.c  # data
    return C_  # or out_?
''' next order: level-parallel cluster_H / multiple agg+? compress as autoencoder? '''

def add_H(H,h, root, fN=0):
    for Lev,lev in zip_longest(H, h):  # bottom-up
        if lev:
            if Lev: Q2R([Lev,lev], Lev, merge=2,froot=0, fN=fN)
            else: H.append((CopyF,Copy_)[fN](lev, root))
def sum_H(N_, Ft):
    H = []
    for N in N_:
        if H: H[0]+= N.N_ # top lev
        elif  N.N_: H = [list(N.N_)]
        if hasattr(N, 'Nt'): add_H(Ft.H, N.Nt.H, Ft)
    Ft.H = [sum2F(lev,'lev',Ft,fset=0) for lev in H] if H else []

def sum2F(N_, nF, root, TT=np.zeros((2,9)), C=0, R=0, fset=1, fCF=1):  # -> CF/CN

    if C: m,d = vt_(TT,R)
    else: m,d,TT,C,R = sum_vt(N_,fm=1)
    Ft = (CN,CF)[fCF](nF=nF, dTT=TT,m=m,d=d,c=C,r=R, root=root); setattr(Ft,'N_',N_); Ft.H = []   # root Bt|Ct ->CN
    if any([n.N_ for n in N_]):
        sum_H(N_,Ft)  # sum lower levels, if any
    if fset:
        setattr(root, Ft.nF,Ft)
        for N in N_: N.root = Ft
    return Ft

def sum2C(N_, _C, _i=None, root=None):  # fuzzy sum + base attrs for centroids

    cc_ = []
    for N in N_:
        if _i is None: i = N._C_.index(_C); m,o = N._m_[i], N._o_[i]  # cluster_C
        else:          m,o = N.m_[_i],N.o_[_i]  # current m_,o_ in cluster_P
        cc_ += [N.c * (m/(ave*o) * _C.m)]  # *_C.m: proj survival?
    Cc = sum(cc_)
    R = 0; TT = np.zeros((2,9)); kern = np.zeros(4); span = 0; yx = np.zeros(2)
    for N, c in zip(N_,cc_):
        rc = c/ Cc; TT += N.dTT*rc; kern += N.kern*rc; span += N.span*rc; yx += N.yx*rc; R += N.r*rc
    m,d = vt_(TT,R, TTw(root.root,1))  # wTTC
    C = CN(typ=3, Nt= CF(N_=N_), dTT=TT, m=m, d=d, c=Cc, r=R, yx=yx, kern=kern, span=span, root=root)
    # set param correlation wws per comp and clustering:
    for i, TT, wTT, ww in zip([0,1,2,3], [C.dTT, C.Ct.dTT, C.TTn, C.TTc], C.wTT_, [wN,wC,wn,wc]):
        C.wTT_[i] = cent_TT(TT, R) * ww
    return C

def sum2G(ft_, root=None, init=1, typ=None):

    if not init:
        N_,_,ntt,nc,nr = ft_[0]; N_+=root.N_; ntt+=root.Nt.dTT; nc+=root.Nt.c; nr+=root.Nt.r; ft_[0] = N_,_,ntt,nc,nr
        if len(ft_)>1: L_,_,ltt,lc,lr=ft_[1]; L_+=root.L_; ltt+=root.Lt.dTT; lc+=root.Lt.c; lr+=root.Lt.r; ft_[1]=L_,_,ltt,lc,lr
    Ft_ = []
    for ft, nF in zip_longest(ft_,('Nt','Lt','Bt')):
        if ft: n_,_,tt,c,r = ft; Ft_+= [CF(N_=n_,nF=nF,dTT=tt,m=(vt:=vt_(tt,r))[0],d=vt[1],c=c,r=r)]
        else:  Ft_ += [CF()]
    C_ = [c for N in ft_[0][0] for c in N.Ct.N_]
    Ft_ += [Q2R(list(set(C_))) if C_ else CF()]  # add multiple root_ in Cs?
    G = comb_Ft(*Ft_, root)
    N_ = G.N_; N=N_[0]; G.sub = N.sub+1 if G.L_ else N.sub
    if typ is None: typ = N.typ  # obsolete?
    G.typ=typ; r=G.r
    if G.Lt:  # sub+
        Lt = G.Lt; L_,lm,ld,lr = Lt.N_,Lt.m,Lt.d,Lt.r
        if lm*ld* ((len(L_)-1)*Lw) > ave*avd* (lr+1)*wN:
            V = lm - ave*(lr+1) * wN
            if V > 0:
                r+=1; E_= get_exemplars({N for L in L_ for N in L.nt},r)
                G_,r= cluster_C(G.Nt,E_,r) if (mdecay(L_)-decay)*V > ave*wC else cluster_N(G.Nt, E_,r)
                if G_ and val_(G.Nt.dTT, r+wn, TTw(G,0), mw=(len(G_)-1)*Lw) > 0:
                    cross_comp(G.Nt,r,'Nt')
    if G.Bt:
        Bt = G.Bt; bd,br = Bt.d,Bt.r; rroot = root.root if root.root else 0
        if bd > avd*br*wn and N.typ!=1: cross_comp(Bt, br,'Bt')  # no ddfork
        if rroot: Bt.brrw = Bt.m * (rroot.m * (decay * (rroot.span/G.span)))  # external lend only, subtract from root?
    G.rN_ = sorted(G.rN_, key=lambda x: (x.m/x.c), reverse=True)
    return G

def comb_Ft(Nt, Lt, Bt, Ct, root):  # from sum2G, default Nt

    G = CN(Nt=Nt,Lt=Lt,Bt=Bt,Ct=Ct, root=root); Nt.root=G; Lt.root=G; Bt.root=G; Ct.root=G
    T = CopyF(Nt)  # temporary accumulator
    dF_ = []
    for Ft in Lt, Bt:  # connectivity forks, Ct is not directly combined and compared, rdn only?
        if Ft: dF_ += [comp_F(T,Ft, root.r,G)]; T.dTT,T.c,T.r = sum_vt([T,Ft])  # Bt*brrw?
        else:  dF_ += [CF()]
    Q2R([G,T], G, merge=0)
    if any(dF_):
        Q2R(dF_, G.Xt)  # cross-fork covariance
        Q2R([G,G.Xt], G,merge=0)
    add_Nt(G, Nt)  # add H,kern,ext, doesn't affect comp_F
    if Lt: add_Lt(G, Lt)
    return G

def add_Nt(G, Nt, merge=0):  # addition to Q2R

    if isinstance(Nt,CF) and G.Nt.H and Nt.H:  # also Ct.H if separate?
        add_H(G.Nt.H if merge else G.Nt.H[:-1], Nt.H, G)  # else top lev = Nt.N_
    N_ = Nt.N_
    if merge: G.N_ += N_ # never 2
    else:     G.Nt.H += [Q2R(N_+ (G.Nt.H.pop().N_ if G.Nt.H else []))]  # extend or init top level
    yx_ = []; C = G.c + Nt.c  # G is empty?
    for N in N_:
        N.fin = 1; N.root = G; c = N.c
        if hasattr(G,'m_'): G.r += np.sum([o*N.r for o in N.o_]); G.rN_ += N.rN_; G.m_ += N.m_; G.o_ += N.o_
        G.C_ += N.rN_  # Ct||Nt;  A,a = G.angl[0],N.angl[0]; A[:]= (A*C+a*c)/C # vect only, if in Nt?
        G.kern = G.kern = (G.kern*(C-c) + N.kern*c) / C  # massive?
        G.box = extend_box(G.box, N.box)
        yx_ += [N.yx]
    G.yx = yx = np.mean(yx_, axis=0); dy_,dx_ = (np.array(yx_)-yx).T
    G.span = np.hypot(dy_,dx_).mean() if len(N_)>1 else N.span

def add_Lt(G, Lt):  # addition to Q2R

    L_ = Lt.N_
    if Lt.m > ave*wn*Lt.r:  # comp typ -1 pre-links
        L_,pL_ = [],[]; [L_.append(L) if L.typ==1 else pL_.append(L) for L in Lt.N_]
        if pL_ and sum_vt(pL_,fm=1)[0] > ave*wn * np.mean([L.r for L in pL_]):
            for L in pL_: L_ += [comp_N(*L.nt, G.r,1, L.angl[0], L.span)]
            Q2R(L_,Lt)
            Q2R([G,Lt], G, merge=0)
    A = np.sum([l.angl[0] for l in L_], axis=0) if L_ else np.zeros(2)
    G.angl = np.array([A, np.sign(G.dTT[1] @ wTTN[1])], dtype=object)  # add weighting?
    G.mang = np.mean([comp_A(G.angl[0], l.angl[0])[0] for l in G.L_])  # Ls only?

def cent_TT(dTT, r):  # weight attr matches | diffs by their match to the sum, recompute to convergence

    wTT,_wTT = [],np.ones((2,9)); coT = np.abs(dTT[0]) + np.abs(dTT[1])  # complemented vals
    while True:
        for fd, _wT, dT in zip((0,1), _wTT, dTT):
            vT = np.abs(dT)  # if -m: wrong, or surprise value?
            rvT = vT / eps_(coT) * _wT  # weighted normalized vals
            mean = rvT.mean() or eps  # scalar
            invdev_ = np.minimum(rvT / mean, mean / eps_(rvT))
            wT = invdev_ / (invdev_.mean() or eps)   # mean(wT)=1
            wTT += [wT]
        if np.sum(np.abs(wTT-_wTT)) < ave * r:  # if np.linalg.norm(wT - _wT, 1) < r?
            break
        _wTT = np.array(wTT); wTT = []
    return _wTT  # single-mode dTT, extend to 2D-3D lev cycles in H, cross-level param max / centroid?

def mdecay(L_):  # slope function
    L_ = sorted(L_, key=lambda l: l.span)
    dm_ = np.diff([l.m/l.c for l in L_])
    ddist_ = np.diff([l.span for l in L_])
    return - (dm_/ eps_(ddist_)).mean()  # -dm/ddist

def CopyF(F, root=None, r=1):  # F = CF
    C = CF(dTT=F.dTT*r, m=F.m, d=F.d, c=F.c, r=F.r, root=root or F.root, nF=F.nF)
    C.N_ = [(N if isinstance(N, CN) else (CopyF(N,root=C) if isinstance(N, CF) else [])) for N in F.N_]  # flat
    return C

def Copy_(N, root=None, init=0, typ=None):

    if typ is None: typ = 2 if init<2 else N.typ  # G.typ = 2, C.typ=0
    C = CN(dTT=deepcopy(N.dTT),typ=typ); C.root = N.root if root is None else root
    for attr in ['m','d','c','r']: setattr(C,attr, getattr(N,attr))
    if not init and C.typ==N.typ: C.Nt = CopyF(N.Nt,root=C)
    if typ:
        for attr in ['fin','span','mang','sub','exe']: setattr(C,attr, getattr(N,attr))
        for attr in ['nt','kern','box','compared','dTT','TTn','TTc','m','d','c','r']: setattr(C,attr, copy(getattr(N,attr)))
        for attr in ['Nt','Lt','Bt','Ct','Xt','Rt']: setattr(C, attr, CopyF(getattr(N,attr), root=C))
        if init:  # new G
            C.yx = [N.yx]; C.angl = np.array([copy(N.angl[0]), N.angl[1]],dtype=object)  # get mean
            C.L_ = [l for l in N.rim if l.m>ave]; N.root=C; C.fin = 0  # else centroid
            C.N_ = [N]; C.m_=[]; C._m_=[]; C.o_=[]; C._o_=[]
        else:
            C.Lt=CopyF(N.Lt); C.Bt=CopyF(N.Bt)  # empty in init G
            C.angl = copy(N.angl); C.yx = copy(N.yx)
        if typ > 1: C.Rt = CopyF(N.Rt)
    return C

def extend_box(_box, box):
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return np.array((min(y0,_y0), min(x0,_x0), max(yn,_yn), max(xn,_xn)))

def sort_H(H, fi):  # lev.rc = complementary to root.rc and priority index in H, if selective and aligned

    i_ = []  # priority indices
    for i, lay in enumerate(sorted(H.node_, key=lambda lay: [lay.m,lay.d][fi], reverse=True)):
        di = lay.i - i  # lay index in H
        lay.r += di  # derR - valR
        i_ += [lay.i]
    H.i_ = i_  # H priority indices: node/m | link/d
    if fi > 1:
        H.root.node_ = H.node_
    # more advanced ordering: dH | H as medoid cluster of layers, nested cent_TT across layers?

def eval(V, weights):  # conditional progressive eval, with default ave in weights[0]
    W = 1
    for w in weights:
        W *= w
        if V < W: return 0
    return 1

def proj_focus(PV__, y,x, tile):  # radial accum of projected focus value in PV__

    m,d,n = tile.m, tile.d, tile.c  # add r?
    V = (m-ave*n) + (d-avd*n)
    dy,dx = tile.angl[0]; a = dy / (dx or eps)  # average link_ orientation, projection
    Dec = decay * (wYX / np.hypot(y-tile.yx[0], x-tile.yx[1]))  # unit_decay *rel_dist?
    H, W = PV__.shape  # = win__
    n = 1  # radial distance
    while y-n>=0 and x-n>=0 and y+n<H and x+n<W:  # rim is within frame
        dec = Dec * n
        pV__= np.array([
        V * dec * 1.4, V * dec * a, V * dec * 1.4,  # a = aspect = dy/dx, affects axial directions only
        V * dec / a,                V * dec / a,
        V * dec * 1.4, V * dec * a, V * dec * 1.4
        ], dtype=float)
        if np.sum(pV__) < ave * 8:
            break  # < min adjustment
        rim_coords = np.array([
        (y-n,x-n), (y-n,x), (y-n,x+n),
        (y, x-n),           (y, x+n),
        (y+n,x-n), (y+n,x), (y+n,x+n)
        ], dtype=int)
        row,col = rim_coords[:,0], rim_coords[:,1]
        PV__[row,col] += pV__  # in-place accum pV to rim
        n += 1

def proj_TT(L, cos_d, dist, r, pTT, wTT, fdec=0, frec=0, dec=1):  # accumulate link pTT with iTT or eTT internally, L may be N?

    dec = dist if fdec else ave ** (1 + (dist * dec) / L.span)  # not fully revised, ave = match decay rate / unit distance
    TT = np.array([L.dTT[0] * dec, L.dTT[1] * cos_d * dec])
    cert = abs(val_(TT,r,wTT) - ave)  # approximation
    if cert > ave:
        pTT+=TT; return  # certainty margin = ave
    if not frec:  # non-recursive
        for lev in L.Nt.N_:  # refine pTT
            proj_TT(lev, cos_d, dec, r+1, pTT, wTT, fdec=1, frec=1)
    pTT += TT  # L.dTT is redundant to H, neither is redundant to Bt,Ct
    if L.Bt:  # + trans-link tNt, tBt, tCt?
        TT = L.Bt.dTT
        if TT is not None:
            pTT += np.array([TT[0] * dec, TT[1] * cos_d * dec])

def proj_N(N, dist, A, r, dec=1):  # arg rc += N.rc+Nw, recursively specify N projection val, add pN if comp_pN?

    cos_d = (N.angl[0].dot(A) / ((np.hypot(*N.angl[0]) * dist) or eps)) * N.angl[1]  # internal x external angle alignment
    iTT, eTT = np.zeros((2,9)), np.zeros((2,9))
    wTT = TTw(N,0)
    for L in N.L_+N.B_: proj_TT(L, cos_d, dist, L.r+r, iTT, wTT, dec=dec)  # accum TT internally
    for L in N.rim:  proj_TT(L, cos_d, dist, L.r+r, eTT, wTT, dec=dec)
    pTT = iTT + eTT  # projected int,ext links, work the same?

    return pTT  # val_(N.dTT,rc) * (1- val_(iTT+eTT, rc))  # info_gain = N.m * average link uncertainty, should be separate
'''
add comp_prj_nt?
def comp_prj_dH(_N, N, ddH, rn, link, angl, span, dec):
        
    # comp proj dH to actual dH-> surprise, not used
    _cos_da = angl.dot(_N.angl) / (span * _N.span)  # .dot for scalar cos_da
    cos_da = angl.dot(N.angl) / (span * N.span)
    _rdist = span / _N.span
    rdist  = span / N.span
    prj_DH = add_H(proj_H(_N.derH, _cos_da, _rdist * dec),
                    proj_H(N.derH, cos_da, rdist * dec))  # comb proj dHs
    # add imagination: cross_comp proj derHs?
    # Et+= confirm:
    dddH = comp_H(prj_DH, ddH, rn, link)
    link.m += dddH.m; link.d += dddH.d; link.c += dddH.c; link.dTT += dddH.dTT
    add_H(ddH, dddH)    
'''

def vect_edge(tile, rV=1, wTT_=None):  # PP_ cross_comp and floodfill to init focal frame graph, no recursion:

    global ave,avd, aveB, Lw,distw,intw, wN,wC,wn,wc, wTTn,wTTc,wTTN,wTTC  # /= projected V change:
    if wTT_ is None: wTTn/=rV; wTTc/=rV; wTTN/=rV; wTTC/=rV  # also cw,nw,Cw,Nw?
    else: wTTN,wTTC, wTTn,wTTc = wTT_  # detailed feedback
    def PP2N(PP):
        P_,L_,B_,verT,latT,A,S,box,yx, m,d,c = PP
        kern = np.array(latT[:4])
        [mM, mD, mI, mG, mA, mL], [dM, dD, dI, dG, dA, dL] = verT  # re-pack in dTT:
        dTT = np.array([ np.array([mM, mD, mL, mI, mG, mA, mL, mL / 2, 0]),  # extA=0
                         np.array([dM, dD, dL, dI, dG, dA, dL, dL / 2, 0])])
        y,x,Y,X = box; dy,dx = Y+1-y, X+1-x
        A = np.array([np.array(A), np.sign(dTT[1] @ wTTn[1])], dtype=object)  # append sign
        PP = CN(typ=0, dTT=dTT,m=m,d=d,c=c,r=1, kern=kern,box=box,yx=yx,angl=A,span=np.hypot(dy/2,dx/2))  # set root in trace_edge
        m_, d_ = np.zeros(6), np.zeros(6)
        for B in B_: m_ += B.verT[0]; d_ += B.verT[1];
        ad_ = np.abs(d_); t_ = m_ + ad_  # ~ max comparand
        m = m_/eps_(t_) @ w_t[0] - ave*2; d = ad_/eps_(t_) @ w_t[1] - avd*2
        PP.Bt = CF(N_=B_, m=m, d=d, r=2, root=PP,nF='Bt')
        for P in P_: P.root = PP
        if hasattr(P,'nt'):  # PPd, assign rN_:
            for dP in P_:
                for P in dP.nt: PP.rN_ += [P.root]  # PPm
        return PP
    blob_ = tile.N_; G_,TT,C = [],np.zeros((2,9)),0
    for blob in blob_:
        if not blob.sign and blob.G > aveB:
            edge = slice_edge(blob, rV)
            if edge.G * ((len(edge.P_)-1)*Lw) > ave * sum([P.latT[4] for P in edge.P_]):
                PPm_ = comp_slice(edge, rV, wTTn)
                N_ = [PP2N(PPm) for PPm in PPm_]
                for PPd in edge.link_: PP2N(PPd)
                for N in N_:
                    if N.B_:
                        PPd_ = [B.root for B in N.B_]; Q2R(PPd_, N.Bt)
                        N.Bt.N_ = PPd_; [setattr(B,'root',N.Bt) for B in PPd_]
                if val_(sum_vt(N_)[0],3,TTw(tile), (len(PPm_)-1)*Lw) > 0:
                    G_,TT,C = trace_edge(N_,G_,TT,C,3,tile)  # flatten B_-mediated Gs
    if G_:
        setattr(tile,'Nt', sum2F(G_,'Nt',tile,TT,C,R=1))  # update tile.wTT?
        tile.dTT = TT; tile.c = C
        if vt_(TT, tile.r)[0] > ave:
            return tile

def trace_edge(N_,_G_,_TT,_C, r,root):  # cluster contiguous shapes via PPs in edge blobs or lGs in boundary/skeleton?

    L_, cT_, lTT, lc = [],set(),np.zeros((2,9)),0  # comp co-mediated Ns:
    for N in N_: N.fin = 0
    for N in N_:
        _N_ = [rN for B in N.B_ for rN in B.rN_ if rN is not N]   # + node-mediated
        for _N in list(set(_N_)):  # share boundary or cores if lG with N, same val?
            cT = tuple(sorted((N.id,_N.id)))
            if cT in cT_: continue
            cT_.add(cT)
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx)  # Rc = r+ (N.r+_N.r)/2
            Link = comp_N(_N,N, r, A=dy_dx, span=dist)
            if vt_(Link.dTT,Link.r)[0] > ave*r:  L_+=[Link]  # r = 1|2, add Bt?
            if L_: lTT,lc,_ = sum_vt(L_)
    Gt_ = []
    for N in N_:  # flood-fill G per seed N
        if N.fin: continue
        N.fin=1; _N_=[N]; Gt=[]; N.root=Gt; n_,ntt,nc, l_,ltt,lc = [N],N.dTT.copy(),N.c or 1,[],np.zeros((2,9)),0  # Gt
        while _N_:
            _N = _N_.pop(0)
            for L in _N.rim:
                if L in L_:
                    n = L.nt[0] if L.nt[1] is _N else L.nt[1]
                    if n in N_:
                        if n.root is Gt: continue
                        if n.fin:  # merge n root
                            _root = n.root; n_+=_root[0];l_+=_root[3]; _root[6]=1
                            for _n in _root[0]: _n.root = Gt
                        else: n.fin=1; _N_+=[n]; n_+=[n]; l_+=[L]  # add single n
                        n.root = Gt
        ntt,nc,_ = sum_vt(n_)
        if l_: ltt,lc,_= sum_vt(l_); ltt*=lc/nc;
        Gt += [n_,ntt,nc, l_,ltt,lc, 0]; Gt_+=[Gt]
    G_, TT,C = [],np.zeros((2,9)),0
    for n_,ntt,nc,l_,ltt,lc,merged in Gt_:
        if not merged:
            if vt_(ntt+ltt,r)[0] > ave*r:  # wrap singletons, use Gt.r?
                TT += ntt+ltt; C += nc+lc  # add Bt?
                G_ += [sum2G([(n_,'Nt',ntt,nc,r)] + ([(l_,'Lt',ltt,lc,r)] if l_ else []),root,typ=2)]
            else:
                for N in n_: N.fin=0; N.root=root
    if val_(TT,r+1,TTw(root)) > 0: _G_+=G_; _TT+=TT;_C+=C  # eval per edge, concat in tile?
    return _G_,_TT,_C

# frame expansion per level: cross_comp lower-window N_,C_, forward results to next lev, project feedback to scan new lower windows

def frame_H(image, iY,iX, Ly,Lx, Y,X, rV, max_elev=4):  # all initial args set manually

    def base_tile(y,x):  # 1st level, higher levels get Tg s
        T = frame_blobs_root( comp_pixel( image[y:y+Ly, x:x+Lx]), rV)
        T = vect_edge(T, rV)  # form, trace PP_
        if T: cross_comp(T.Nt,T.r)
        return T

    def expand_lev(_iy, _ix, elev, T):  # seed tile is pixels in 1st lev, or Fg in higher levs

        frame = np.full((Ly, Lx), None, dtype=object)  # level scope
        iy,ix = _iy,_ix; cy,cx = (Ly-1)//2, (Lx-1)//2; y,x = cy,cx  # start at mean
        T_, PV__ = [], np.zeros([Ly, Lx])  # maps to level frame
        while True:
            # each loop adds one tile to lev_frame
            if not elev: T = base_tile(iy, ix)
            if T and val_(T.dTT,T.r+wn+elev, wTT_[0], mw=(len(T.N_)-1)*Lw) > 0:
                frame[y,x] = T; T_ += [T]
                dy_dx = np.array([T.yx[0]-y, T.yx[1]-x]); pTT = proj_N(T, np.hypot(*dy_dx), dy_dx, elev)
                if 0 < val_(pTT, elev, wTT_[0]) < ave:  # extend lev by combined proj T_
                    proj_focus(PV__,y,x, T)  # PV__+= pV__
                    pv__ = PV__.copy(); pv__[frame != None] = 0  # exclude processed
                    y, x = np.unravel_index(pv__.argmax(), PV__.shape)
                    if PV__[y,x] > ave:
                        iy = _iy + (y-cy) * Ly**elev; ix = _ix + (x-cx) * Lx**elev  # feedback to shifted coords
                        if elev:
                            T = frame_H(image, iy, ix, Ly, Lx, Y,X, rV, elev)  # up to current level
                    else: break
                else: break
            else: break
        if T_:
            TT,C,R = sum_vt(T_); R += elev
            if val_(TT, R/len(T_) + elev, wTT_[0], mw=(len(T_)-1)*Lw) > 0:
                return T_
    global ave,avd, Lw, intw, wN,wC,wn,wc, distw, WN,WC,Wn,Wc, wTT_
    ave,avd, Lw, intw, wN,wC,wn,wc, distw = np.array([ave,avd, Lw, intw, wN,wC,wn,wc, distw]) / rV

    elev = 0; F, tile = [],[]  # frame, seed lower tile, if any
    while elev < max_elev:
        tile_ = expand_lev(iY,iX, elev, tile)
        if tile_:  # sparse higher-scope tile, if expanded
            F = CN(box=np.array([0,0,Y,X]), yx=np.array([Y//2, X//2]))  # same center on all levels
            F.H = []; F.N_ = tile_  # or Nt = Q2R(tile_)?
            if elev: [add_H(F.H,T.H, F,fN=1) for T in tile_]
            F.H += [lev := Q2R([n for N in tile_ for n in N.N_],fN=1)]; Q2R([F,lev], merge=0,R=F)  # concat prior top lev
            if cross_comp(F.Nt, rr=elev)[0]:  # spec->tN_,tC_,tL_, proj comb N_'L_?
                elev += 1
                if rV > ave:
                    for i, dTT, wTT, ww in zip((0,1,2,3), (F.dTT,F.Ct.dTT,F.TTn,F.TTc), F.wTT_, (wN,wC,wn,wc)):
                        F.wTT_[i] = cent_TT(dTT,2) * ww  # max-scope correlation weights
                if elev == max_elev:  # fb / top lev
                    rV, wTT_ = ffeedback(F)  # update globals, rV is not used?
                for wTT, wwTT in zip(wTT_, F.wTT_): wTT *= wwTT   # F.wTT_ is effectively global
                tile = F  # lev tile_ is next extension seed
            else: break
        else: break
    return F  # for intra-lev feedback

def ffeedback(frame):  # adjust filters: all aves *= rV, ultimately differential backprop per ave?

    def init_C_(N_):
        typ_ = []  # group calls by nF type
        for oF in N_:
            tF = next((fork for fork in typ_ if fork.nF==oF.nF), None)
            if tF: tF.N_ += [oF]  # calls / function type
            else:  typ_ += [CF(nF=oF.nF, N_=[oF])]
        return typ_

    # obsolete, there is no Z.Nt.H, just flat Z.call_( indiv oF.call_s)?
    sum2F(Z.call_,'Nt', Z)  # fset Z.Nt, sum nested calls in H
    Z.Ct.H = [init_C_(lev.N_) for lev in Z.Nt.H]  # maps to Z.H
    H, _prom_ = [],[]; r = frame.r+1
    for lev in Z.Ct.H:  # bottom-up merge nested typ_F calls? it's flat at his point
        for tF in _prom_:
            _tF = next((fork for fork in tF.N_ if fork.nF==tF.nF), None)
            if _tF: Q2R(list(set(_tF.N_+tF.N_)), _tF, merge=0,froot=0)
            else:   lev += [tF]
        H+= [Q2R([tF for tF in lev if tF.m > ave*r], froot=0) if lev else CF()]
        _prom_ = [tF for tF in lev if tF.m > ave*(r+1)]  # promote to next level
        r += 1
    Z.Ct.H = H  # lower levs
    Z.Ct.N_ = init_C_(Z.N_ + [oF for tF in _prom_ for oF in tF.N_])  # top lev

    Q2R(Z.Ct.N_, Z.Ct, froot=0)
    # sum ratios between consecutive-level TTs:
    _N,_C,_n,_c = frame.dTT, frame.Ct.dTT, frame.TTn, frame.TTc; rTT_ = [np.ones((2,9)) for _ in range(4)]
    for lev in frame.H:
        # top-down frame expansion levels, not lev-selective or sub-lev recursive
        N,C,n,c = lev.dTT, lev.Ct.dTT, lev.TTn, lev.TTc
        rN, rC, rn, rc = [np.abs(_tt / eps_(tt)) for _tt,tt in zip((_N,_C,_n,_c), (N,C,n,c))]
        for rTT,rtt in zip(rTT_,(rN,rC,rn,rc)): rTT += rtt
        _N,_C,_n,_c = N,C,n,c
    rM = rD = 0
    for i, rTT in enumerate(rTT_):
        rm, rd = vt_(rTT, frame.r, wTT_[i]); rM+=rm; rD+=rd
    return rM+rD, rTT_

if __name__ == "__main__":  # './images/toucan_small.jpg' './images/raccoon_eye.jpeg', add larger global image

    def trace_func(module_dict, module_name=None, exclude=()):
        if module_name is None: module_name = module_dict.get('__name__')
        exclude = set(exclude)
        for name, obj in list(module_dict.items()):
            if name in exclude: continue
            if not inspect.isfunction(obj): continue
            if obj.__module__ != module_name: continue
            if getattr(obj, 'wrapped', False): continue
            module_dict[name] = CoF.traced(obj)

    trace_func(vars(),exclude= {'trace_functions','eps_','prop_F_','extend_box','TTw','Copy_','CopyF'})
    Y,X = imread('./images/toucan.jpg').shape
    # frame = agg_frame(0, image=imread('./images/toucan.jpg'), iY=Y, iX=X)
    frame = frame_H(image=imread('./images/toucan.jpg'), iY=Y//2 -31, iX=X//2 -31, Ly=64,Lx=64, Y=Y, X=X, rV=1)
    # search frames ( tiles inside image, at this size it should be 4K, or 256K panorama, won't actually work on toucan