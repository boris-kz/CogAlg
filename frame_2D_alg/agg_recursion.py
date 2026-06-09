import numpy as np, inspect, contextvars
from copy import copy, deepcopy
from math import atan2, cos, pi  # from functools import reduce
from itertools import zip_longest, combinations, product  # from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, imread, comp_pixel, CBase
from slice_edge import slice_edge
from comp_slice import comp_slice, w_t
from meta_code import oF_,CF,CL,CC,CN,CoF, wT,wTT, eps_,eps,ave,avd,decay, trace_func,parse_funcs, split_oF_,cluster_oF_,F_body_
'''
This is a main module of open-ended clustering algorithm, designed to discover empirical patterns of indefinite complexity. 
Lower modules cross-comp and cluster image pixels and blob slices(Ps), the input here is resulting PPs: segments of matching Ps.
Cycles of (generative cross-comp, compressive clustering, filter-adjusting feedback) should form hierarchical model of input stream: 

Cross-comp forms Miss and Match (min: shared_quantity for directly predictive params, else inverse deviation of miss or variation), in 2 forks:
rng+: incremental-range cross-comp nodes: edge segments at < max distance, cluster if they match. 
der+: incremental-derivation cross-comp links, from node cross-comp, if abs_diff * relative_adjacent_match

Clustering compressively groups the elements into compositional hierarchy, initially by pair-wise similarity or density thereof.
High-contrast links are correlation clustered to form a boundary per node connectivity cluster.
This is followed by centroid-based expansion and divisive sub-clustering.

each level may extend clustering through 4 increasingly fuzzy stages, each seeded by prior-stage output:
- select sparse exemplars to seed the clusters, top k for parallelization (get_exemplars), no clustering?
- connectivity-based agglomerative ( divisive clustering (cluster_N), 
- centroid-based marginally fuzzy and extensible clustering, in divisive phase (cluster_C), or ave_linkage?
- centroid-parallel fully fuzzy FCM-like refine, if significant global overlap (cluster_P), no higher stage?

Clustering forms hierarchical graphs, each a dual tree of down-forking elements: node_H, and up-forking clusters: root_H:
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
Similar to neural dendritic input tree and axonal output tree, but with lateral cross-comp and nested param sets per layer.

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
wM,wD,wi, wG,wI,wa, wL,wS,wA = wT
cFrm, cN_,cC_,cN,cF, cE,ccN,ccC,ccP, cAgg, cVct,cTrc,cBac,cPrj,cCS,cSE = (      # function complexity
wFrm, wN_,wC_,wN,wF, wE,wcN,wcC,wcP, wAgg, wVct,wTrc,wBac,wPrj,wCS,wSE ) = [F.fc for F in oF_]  # ave gain/call, init = cost
ttFrm, ttN_,ttC_,ttN,ttF, ttE,ttcN,ttcC,ttcP, tAgg,ttVct,ttTrc,ttBac,ttPrj,ttCs,ttSE = [F.dTT for F in oF_]

def vt_(TT, wTT=wTT):  # base eval: multi-variate rel match, rel diff for membership

    m_,d_ = TT; ad_ = np.abs(d_); t_ = eps_(m_+ad_)  # ~ max comparand
    m = m_/t_ @ wTT[0]; d = ad_/t_ @ wTT[1]  # norm by co-derived val
    return m, d

def sum_vt(N_, fr=0, fm=0, wTT=wTT, fdiv=1):  # basic weighted sum of CN|CF list

    C = sum(n.c for n in N_); R = 0; TT = np.zeros((2,9))
    for n in N_:
        w = n.c/C; TT += (n.rTT if fr else n.dTT)*w; R += n.r*w
    if fm:
        m,d = vt_(TT, wTT)
        if fdiv: m/= ave*R; d/= avd*R  # in 0-inf for summation
        else:    m-= ave*R; d-= avd*R  # in -1:1 without r
        return   m,d,TT,C,R
    else: return TT,C,R
''' 
  cycle:
- cross-comp nodes, evaluate incremental-derivation cross-comp of new >ave difference links, recursively. 
- connectivity-cluster from exemplars by >ave match links, correlation-cluster links by >ave diff
- form core+contour objects, divisive sub-clustering, higher-composition cross_comp
- centroid-cluster, from exemplars via their rim within a frame, increasing overlap
- parallel fully-fuzzy centroid refining if >min overlap
- forward: selective extend cross-comp, clustering across tiles, re-order centroids by eigenvalues
- feedback filter updates 
'''
def cross_comp(root, rr):  # core function mediating recursive rng+ and der+ cross-comp and clustering

    N_,G_ = root.N_,[]; fC = N_[0].typ==2  # root is Ft, converted below, rc=rdn+olp, comp N_|B_|C_:
    # comp_ s update root oF:
    L_,TT,c,r,TTd,cd,rd = comp_C_(N_,rr,fC=1) if fC else comp_N_(combinations(N_,2),rr)
    if L_: # Lm_, no +|- Ft.Lt?
        root.L_ = L_; L= len(L_)-1  # val=m+d /clust, m/comp
        if sum(vt_(TT,ttE)) * (wE*L) > (ave+avd) * (r+cE*L):  # or oF.gF evals?
            E_ = get_exemplars({N for L in L_ for N in L.N_}, r,c)
            G_,r = cluster_N(root, E_,r,c)  # cluster_C, _P, eval?
            if G_:
                if not root.typ: F2N(root)  # promote @ 1st sub+ or agg+
                root.H+= [sum2F(L_,root,froot=1)]  # lev: L_+ derivatives
                root.Nt = sum2F(G_,root,froot=2); L=len(G_)-1  # or C_ s?
                if vt_(TT,tAgg)[0]*(wAgg*L) > ave* (r+cAgg*L):  # root brrw| rdn?
                    G_,r = cross_comp(root.Nt,r)  # agg+
            oF = CoF.get(); oF.N_ = G_ or N_; oF.rTT=TT; oF.c+=c; oF.r+=r
    return G_, r  # G_ is recursion flag
'''
oF.m = val dTT, or
if proj comp: F.root.wTT *= rdpTT * (F.c/ F.root.c)  # c-weighted feedback
if select clust: (F.m - F.root.Lt.m) * (F.root.c-Ft.c)  # clustering value = loss reduction: root.Lt.m < selective F.m, add dval? 
'''
def comp_N_(_pairs, r, tnF=None, root=2):  # incremental-distance cross_comp, max dist depends on prior match

    pairs, TT,cm,rm = [],np.zeros((2,9)), 0,0
    for pair in _pairs:  # get all-to-all pre-links
        _N, N = pair
        c = min(N.c,_N.c)  # comp weight
        if _N.sub != N.sub: continue  # or comp x composition?
        if N is _N:  # overlap = unit match, no miss, or skip?
            tt = np.array([N.dTT[1],np.zeros(9)]); TT+=tt; cm+=c; rm+=(N.r+_N.r)/2 *c
        else:
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx)
            pairs += [[dist, dy_dx,_N,N,c]]
    if not pairs: return 0,0,0,r,0,0,0

    def proj_V(_N,N, dist, dy_dx, dec):  # _N x N induction
        Dec = dec or decay ** ((dist/((_N.span+N.span)/2)))
        iTT = (_N.dTT + N.dTT) * Dec
        eTT = (_N.Rt.dTT + N.Rt.dTT) * Dec
        cNt = min(_N.c,N.c); rNt = (_N.r+N.r) / 2
        if abs(vt_(eTT,ttN_)[0])* wPrj* cNt > ave* (cPrj+r+rNt):  # spec per N link:
            eTT+= proj_N(N, dist, dy_dx, r, N.c, dec)  # pTT/ L_,B_,rim, if pV>0
            eTT+= proj_N(_N,dist, -dy_dx, r, _N.c, dec)  # reverse direction
        return iTT+eTT

    N_,L_,C,R,TTd,cd,rd = [],[],0,0,np.zeros((2,9)),0,0  # any global use of dLs, rd?
    acc = [TT,cm,rm, TTd,cd,rd]
    for pL in sorted(pairs, key=lambda x: x[0]):  # proximity prior, test compared?
        dist,dy_dx,_N,N, c = pL  # rim angl is not canonic
        pTT = proj_V(_N,N, dist, dy_dx, root.m if root!=2 else decay** (dist/((_N.span+N.span)/2)))  # based on current rim
        lr = r+ (N.r+_N.r)/2; m,d = vt_(pTT,ttN_)  # +|-match certainty
        if m > 0:
            C += c; R += lr * c  # from all Ls+ pLs?
            if abs(m)*wN < ave*(r+cN):  # comp if marginally predictable, update N.Rt pair eval, ave / proj surprise value?
                Link = comp_N(_N,N, lr,c,full=not tnF, A=dy_dx, span=dist, rL=root,L_=L_,N_=N_, acc=acc)
                Link.rTT = np.abs(pTT-Link.dTT) / eps_(Link.dTT)  # relative prediction error to fit oF, direction-agnostic
            else:
                pL = CN(typ=-1, dTT=pTT,m=m,d=d,c=c,r=lr, angl=[dy_dx,1],span=dist); pL.N_ = [_N,N]  # Nt.N_ needs separate assignment
                L_+= [pL]; N.rim+=[pL]; _N.rim += [pL]; N_+=pL.N_; acc[0]+=pTT*pL.c; acc[1]+=pL.c; acc[2]+=pL.r*pL.c  # all +ve
                # no oF val, ~= links in clustering
        else: break  # beyond initial induction range, re-sort by proj_V?
    for N in set(N_):
        if N.rim: sum2F(N.rim, N.Rt)
    # call trace:
    if L_ := [n for n in L_ if n.typ>-1]: sum2F(L_,CoF.get())  # skip pLs, oF.call_+=[oF], adds data
    TT,cm,rm, TTd,cd,rd = acc
    oF = CoF.get(); oF.N_=N_; oF.dTT=TT; oF.c+=c; oF.r+=r
    return L_,TT,cm,rm/(cm or eps), TTd,cd,rd/(cd or eps)

def comp_C_(C_,_r, _C_=[], fall=1, fC=0):  # simplified for centroids, trans-N_s, levels

    N_,L_,tc,tr = [],[],0,0; acc = [np.zeros((2,9)),0,0,np.zeros((2,9)),0,0]
    if fall:
        pairs = product(C_,_C_) if _C_ else combinations(C_,r=2)  # comp between | within list
        for _C, C in pairs:
            c = min(C.c,_C.c); r = (C.r+_C.r)/2  # comp weight
            tc += c; tr += r*c
            if _C is C:
                dtt = np.array([C.dTT[1],np.zeros(9)]); acc[0]+=dtt; acc[1]+=c; acc[2]+=1  # overlap=match
            else:
                dy_dx = _C.yx-C.yx; dist = np.hypot(*dy_dx)
                comp_N(_C,C,_r,c, A=dy_dx,span=dist,L_=L_,N_=N_,acc=acc)
        if fC:
            # merge similar distant centroids, they are non-local
            L_ = sorted(L_, key=lambda dC: dC.d); _C_= C_; C_, merg_ = [],[]  # from min D
            for i, L in enumerate(L_):
                if vt_(L.dTT,ttF)[0]*wF < avd*(_r+cF):
                    C0,C1 = L.N_
                    if C0 is C1 or C1 in merg_ or C0 in merg_: continue  # not merged
                    add2F(C0,C1,1); merg_ += [C1]
                    for l in C1.rim: l.N_ = [C0 if n is C1 else n for n in l.N_]
                else: L_ = L_[i:]; break
            if merg_: N_ = list(set(N_) - set(merg_))
    else:
        # adjacent | distance-constrained cross_comp along eigenvector: sort by max attr, or original yx in sub+, proj C.L_?
        for C in C_: C.compared=set()
        C_ = sorted(C_, key=lambda C: C.dTT[0][np.argmax(ttC_[0])])  # max weight defines eigenvector
        for j in range( len(C_)-1):
            _C = C_[j]; C = C_[j+1]
            if _C in C.compared: continue
            dy_dx = _C.yx-C.yx; dist = np.hypot(*dy_dx); c = min(C.c,_C.c); r = (C.r+_C.r)/2  # tc += c; acc[1]+=c
            tc += c; tr += r*c
            comp_N(_C,C,_r, c ,A=dy_dx,span=dist,L_=L_,N_=N_,acc=acc)  # simplified for typ=3
    for N in list(set(N_)):
        if N.rim: sum2F(N.rim, N.Rt)
    if L_: sum2F(L_,CoF.get())
    TTm,cm,rm,TTd,cd,rd = acc
    oF = CoF.get(); oF.N_ = N_; oF.dTT=TTm; oF.c+=c; oF.r+=r
    return L_,TTm,cm, rm/(cm or eps), TTd,cd, rd/(cd or eps)  # L_ is Lm now

def comp_N(_N,N, r,_c, full=1, A=np.zeros(2),span=None, rL=None, L_=None, N_=None, acc=None):

    def comp_H(_Nt,Nt, Link):  # tentative pre-comp
        dH, tt, C,R = [],np.zeros((2,9)),0,0
        for _lev, lev in zip([_Nt]+_Nt.H, [Nt]+Nt.H):  # should be top-down
            ltt = comp_derT(_lev.dTT[1],lev.dTT[1])
            lc = min(_lev.c,lev.c); lr = (_lev.r+lev.r)/2; m,d = vt_(ltt,ttN_)
            dH += [CF(dTT=ltt,m=m,d=d,c=lc,r=lr,root=Link)]
            tt += ltt*lc; C+=lc; R+=lr*lc
        return dH,tt,C, (r* Link.c+R)/ (Link.c+C)  # same norm for tt?

    TT= base_comp(_N,N)[0] if full else comp_derT(_N.dTT[1],N.dTT[1]); m,d = vt_(TT, ttN_)
    L = CL(N_=[_N,N], dTT=TT,m=m,d=d,c=_c,r=r, root=rL)
    if N.typ==3 and _N.typ==3 and m*wF > ave*(r+cF):
        dH,htt,C,R = comp_H(_N,N, L); r+=R; m+= vt_(htt,ttN_)[0]  # tentative dH, replace if subcomp?
    if N.typ and m*wN_ > ave*(r+cN_):  # skip PPs
        F2N(L)  # add tFs for subcomp
        if N.typ==1:
            for _n,n in product(_N.N_,N.N_): L.Nt.fb_ += [comp_N(_n,n,r,_c)]  # link sub-comp
        else:
            for i,(_Ft,Ft, tnF) in enumerate(zip((_N.Nt,_N.Lt,_N.Bt,_N.Ct),(N.Nt,N.Lt,N.Bt,N.Ct),('Nt','Lt','Bt','Ct'))):
                if _Ft and Ft:  # sub-comp
                    dFt = comp_F(_Ft,Ft,r,L); getattr(L,tnF).fb_ += dFt.N_  # tFt feedback
                    r += (i or 1) -1  # Nt,Lt are not redundant
        for fb_, nF in zip((L.Nt.fb_,L.Lt.fb_,L.Bt.fb_,L.Ct.fb_), ('Nt','Lt','Bt','Ct')):
            if fb_: # L+= trans-Ls, python-batched bottom-up:
                sum2F(fb_,getattr(L,nF))
        sum2F([L.Nt,L.Lt,L.Bt,L.Ct], CoF.get())  # data should be L.Fts?
    if full:
        if span is None: span = np.hypot(*_N.yx - N.yx)
        yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx
        box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
        angl = [A, np.sign(TT[1] @ ttN_[1])]
        L.yx=yx; L.box=box; L.span=span; L.angl=angl; L.kern=(_N.kern+N.kern)/2
    for n, _n in (_N,N),(N,_N):
        n.rim += [L]; n.compared.add(_n)  # or unique comps?
    if L_ is not None:
        N_+= [_N,N]  # for sum_vt(rim)
        if   L.m > ave: acc[0]+=L.dTT*L.c; acc[1]+=L.c; acc[2]+=L.r*L.c; L_ += [L];
        elif L.typ==1 and L.d > avd: acc[3]+=L.dTT*L.c; acc[4]+=L.c; acc[5]+=L.r*L.c  # no pLs?

    oF = CoF.get(); oF.N_=[_N,N]; oF.dTT=L.dTT; oF.c+=L.c; oF.r+=L.r
    return L

def comp_F(_F, F, ir=0, rL=None):

    ddTT = comp_derT(_F.dTT[1],F.dTT[1]); r=(_F.r+F.r)/2; m,d = vt_(ddTT,ttF); r+=ir
    dF = CF(dTT=ddTT, m=m,d=d,r=r,c=min(_F.c,F.c))
    if _F.nF == F.nF:  # sub-comp
        _N_,N_=_F.N_,F.N_; nF=F.nF; l = nF=='Lt'
        if  _N_ and N_:
            if l: Np_ = [[_n,n] for _n,n in zip(_N_,N_) if _n and n]  # same forks
            else: Np_ = list(product(_N_,N_))  # pairs
            L = len(Np_)-1
            if np.mean([rL.m,m])* (wF*L) > ave* (r+cF*L):
                if l: L_= [L for Np in Np_ for L in comp_F(*Np, r,rL=dF).N_]; TT,C,R = sum_vt(L_, wTT=ttF)
                else: L_,TT,C,R,_,_,_= comp_N_(Np_,r,nF,rL)
                if L_:
                    add2F(dF,CF(N_=L_,dTT=TT,c=C,r=R),merge=1); add2F(rL,dF,merge=2); sum2F(L_,CoF.get())
    oF = CoF.get(); oF.N_=[_F,F]; oF.dTT=dF.dTT; oF.c+=dF.c; oF.r+=dF.r
    return dF  # no cross-fork N_, no L ext updates?

def base_comp(_N,N):  # comp Et, kern, extT, dTT

    _M,_D,_C = _N.m,_N.d,_N.c; _I,_G,_Dy,_Dx =_N.kern; _L = len(_N.N_)  # N_: density within span
    M, D, C = N.m, N.d, N.c; I, G, Dy, Dx = N.kern; L = len(N.N_)
    rn = C/_C; mA=dA=0
    _pars = np.array([_M*rn,_D*rn,_C,_I*rn,_G*rn, [_Dy,_Dx],_L*rn,_N.span], dtype=object)  # Et, kern, extT
    pars  = np.array([M,D,C, (I,wI),G, [Dy,Dx], L,(N.span,wS)], dtype=object)
    if hasattr(_N,'angl') and _N.angl and N.angl:  # mostly Ls, some Gs
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

def get_exemplars(N_,_r,_c):  # multi-layer non-maximum suppression -> sparse seeds for diffusive clustering, cluster_N too?

    N_= sorted(N_, key=lambda n: n.Rt.m*n.c, reverse=True); E_,Inh_ = set(),set()
    for rdn, N in enumerate(N_, start=1):  # strong-first
        inh_ = list(Inh_ & set(N.rim))  # stronger Es in N.rim
        oM = sum_vt(inh_,fm=1, wTT=ttE)[0] if inh_ else 0
        oV = oM / (N.Rt.m or eps)  # relative olp V
        if N.Rt.m * N.c * wE > ave* (_r+rdn+cE+oV):
            E_.add(N); N.exe = 1  # in point cloud of focal nodes
            Inh_.update(set(N.rim))  # extend inhibition zone
        else:
            break  # the rest of N_ is weaker, trace via rims
    oF = CoF.get(); oF.N_ = N_
    if E_:  sum2F(E_,oF)  # or default with wc cost?
    else: E_ = [N_[0]]  # no inhibition, any N can be seed
    return E_, (N_, np.sum([E.dTT for E in E_],axis=0), sum([E.c for E in E_]), sum([E.r for E in E_]))

def cluster_N(Ft, _N_, r,_c):  # flood-fill node | link clusters, flat, replace iL_ with E_?

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
                    if tL.m*wcN> ave*ccN:  # merge trans_link.N_.roots
                        rt0 = getattr(tL.N_[0].root,'root',None); rt1 = getattr(tL.N_[1].root,'root',None)  # CNs
                        if rt0 and rt1 and rt0 !=rt1: add_Nt(rt0, rt1, merge=1)  # concat in higher G
            L.Nt,L.Bt,L.Ct = CF(),CF(),CF()
            # merge roots
    G_= []  # add prelink pL_,pN_? include merged Cs, in feature space for Cs
    if _N_ and sum(vt_(Ft.dTT, Ft.root.wTT * ttcN)) * wcN * (len(_N_) - 1) > (ave + avd) * (r + ccN *  (len(_N_) - 1))  > 0:  #| fL?
        for N in _N_:
            N.fin=0; N.exe=1; sum2F(N.rim,N.Rt)  # only if N was added in trans-cluster?
        G_=[]; Gt_=[]; in_= set()  # root attrs
        for N in _N_:  # form G per remaining N
            if N.fin or (Ft.root.root and not N.exe): continue  # no exemplars in Fg
            N_ = [N]; L_,B_ = [],[]; N.fin=1  # init G
            __L_= N.rim  # spliced rim
            while __L_:
                _L_ = []
                for L in set(__L_) - in_:  # flood-fill via frontier links
                    _N = L.N_[0] if L.N_[1].fin else L.N_[1]; in_.add(L)
                    if not _N.fin and _N in Ft.N_:
                        m,d = nt_vt(*L.N_)
                        if m > ave * (r-1):  # cluster nt, L,C_ by combined rim density:
                            span = np.sqrt(len(N_))  # approx span
                            if span > 3:  # refine by rim connectivity / norm span
                                iM = sum([L.m for L in _N.rim if (L.N_[0] if L.N_[1] is _N else L.N_[1]) in N_])
                                if iM / (span*decay) < ave * (r-1): continue  # normalized N-to-N_ match, sum for sum2G?
                            N_ += [_N]; L_ += [L]; _N.fin = 1
                            _L_+= [l for l in _N.rim if l not in in_ and (l.N_[0].fin ^ l.N_[1].fin)]   # new frontier links, +|-?
                        elif d > avd * (r-1): B_ += [L]  # contrast value, exclusive?
                __L_ = list(set(_L_))
            if N_:
                ft_ = []
                for i,(F_,nF) in enumerate(zip((N_,L_,B_),('Nt','Lt','Bt'))):  # no Ct till sub+?
                    F_ = list(set(F_)) or []; tt,fc,fr = sum_vt(F_,wTT=ttcN) if F_ else (np.zeros((2,9)),0,0)
                    ft_+= [[F_,nF,tt,fc,fr]]
                (_,_,nt,nc,nr),(_,_,lt,lc,lr),(_,_,bt,bc,br) = ft_
                c = nc + lc + bc
                r = (nr*nc + lr*lc + br*bc) /c  # br includes overlap?
                tt= (nt*nc + lt*lc + bt*bc) /c  # tentative
                if sum(vt_(tt, Ft.root.wTT*ttcN))*wcN*(len(N_)-1) > (ave+avd)*(r+ccN*(len(N_)-1)):  # apply Fw_ and Fc_ in every eval_?
                    G_ += [sum2G(ft_,ttcN, CN())]; Gt_+= [[tt,c,r]]  # evals cluster_C, cluster_P internally
                    # _C=C+c; _rc=C/_C; rc=c/_C; TT=TT*_rc+tt*rc; R=R*_rc+r*rc; C=_C
        if G_:
            for G in G_: trans_cluster(G)  # splice trans_links, merge L.N_.roots
            C = sum([g[1] for g in Gt_]); TT=np.zeros((2,9)); R=0; r+=1  # + wC?
            for tt,c,gr in Gt_: w=c/C; TT+=tt*w; R+=gr*w
            L = len(G_)-1
            if sum(vt_(TT, Ft.root.wTT*ttcN))*(wcN*L) > (ave+avd)*(r+R+ccN*L):  # reform root,Nt, no other forks yet:
                rG = Ft.root; Nt=rG.Nt; Nt.N_=G_; Nt.dTT=TT; Nt.r=R
                rG.dTT=TT; rG.c=C; rG.r=R
        # clust_val = selectivity vs root.Lt: all comps, add dval? add Gt vals:
        sel_TT = (Ft.dTT*Ft.c - Ft.root.Lt.dTT*Ft.root.Lt.c) / eps_(Ft.root.Lt.dTT * Ft.root.Lt.c)
        oF = CoF.get(); oF.N_=copy(G_); oF.rTT=sel_TT; oF.c = Ft.root.c - Ft.c; oF.r=sum([G.r * G.c for G in G_])/oF.c  # sum_vt?
        # combine C_:
        C_ = [C for N in (G_ if G_ else _N_) for C in N.Ct.N_]
        if C_:
            sum2F(C_, root=Ft.root.Ct)  # Ct.r includes overlap?
            L_ = [L for C in C_ for L in C.L_]  # C-to-N links
            if L_: Ft.root.Ct.Lt = sum2F(L_,root=Ft.root.Ct)
    return G_, r

def cluster_C(Ft, E_,_r,_c):  # form centroids by clustering exemplar surround via rims of new member nodes, within root

    N_= copy(Ft.N_); _C_=[]  # revert if 0 clusters?
    for n in N_: n.root_,n.m_,n.d_,n._root_,n._m_,n._d_ = [],[],[],[],[],[]
    for i,E in enumerate(E_):
        C = Copy_(E, Ft,init=1,typ=CC)
        C.N_,C.L_,C.m_,C.d_ = [E],[],[1],[0]
        E._root_+=[C]; E._m_+=[1]; E._d_+=[0]  # self m,d
        C._N_= list({n for l in E.rim for n in l.N_ if n is not E})  # init w for first loop eval
        _C_ += [C]
    out_ = []
    while True:  # reform C_
        C_, cnt,mat,dif,rdn,DTT,Up = [],0,0,0,0,np.zeros((2,9)),0; Ave = ave*(_r+ccC); Avd = avd*(_r+ccC)
        for _C in _C_:  # C.m,d /rTT? sort / sum(_C.m_)?
            N__,n_,m_,d_,M,D,T,R,dTT,up = [],[],[],[],0,0,0,0, np.zeros((2,9)),0  # /C
            for n in _C.N_+_C._N_:  # current + frontier
                if n not in N_: N_+=[n]; n.root_,n.m_,n.d_,n._root_,n._m_,n._d_ = [],[],[],[],[],[]
                dtt,_ = base_comp(_C,n)  # or comp_N, decay?
                m,d = vt_(dtt,ttcC); dTT+=dtt
                n_+=[n]; m_+=[m]; d_+=[d]; c=n.c; T+=c; M+=m*c; D+=abs(d)*c; R+=n.r*c  # scale totals only?
                if _C in n._root_:
                    k=n._root_.index(_C); up += abs(n._m_[k]-m) + abs(n._d_[k]-d)
                else: up += m+abs(d)
            r = _r+ R/T  # loop-local, not ave?
            if M*(_C.m+ wC_*len(n_)+wC_*len(n_)) > Ave*(r+_C.r+ cC_*len(n_)):  # else: Up+= sum(_C._m_)+ sum([abs(d) for d in _C._d_])?
                for n in [_n for n in n_ for l in n.rim for _n in l.N_ if _n is not n]:
                    N__ += [n]  # +|-Ls
                C = sum2F(n_, Ft,m_,d_)  # C.N_ = n_
                C._N_ = list(set(N__)- set(n_))  # new frontier
                if D<Avd: out_+=[C]  # output if stable
                else:     C_ += [C]  # reform
                DTT+=dTT; mat+=M; dif+=D; cnt+=T; rdn+=R; Up+=up
        r = _r+ rdn/(cnt or eps)
        L = len(out_+ C_); olp = sum([len(N.root_) for N in N_])  # rdn+=olp, prioritize stronger?
        if (mat+dif)* (wcP*L) > Ave* (r+olp+ ccP*L):
            out_+=C_; T=sum([f.c for f in out_])
            out_ = cluster_P(out_,T, Ft)  # refine all memberships in parallel by global backprop
            break
        if Up * (wcC*len(C_)) > Avd * (r+ (ccC*len(C_))):
            for C in C_: C._m_ = C.m_; C._d_ = C.d_
            for n in N_: n._root_ = n.root_; n._m_ = n.m_; n._d_ = n.d_
            for n in set([_n for _C in _C_ for _n in _C.N_ + _C._N_]): n.root_,n.m_,n.d_ = [],[],[]
            _C_ = C_
        else: out_+=C_; break  # converged
    if out_:
        for n in [N for C in out_ for N in C.N_]:  # exemplar V + sum n match_dev to Cs, m* ||C rvals:
            n.exe = (n.d if n.typ==1 else n.m) + np.sum(n.m_) - ave
        if vt_(DTT,Ft.root.wTT*ttcC)[0]*wcC*(len(out_)-1) > ave*(r+ccC*(len(out_)-1)):
            Ct = sum2F([F2N(C) for C in out_]); Ft.root.Ct = Ct; Ct.root = Ft.root
            _,r = cross_comp(Ct,r)  # all distant Cs, seq C_ in eigenvector = argmax(root.wTT)?
            sel_TT = (Ct.dTT*Ct.c - Ft.dTT*Ft.c) / eps_(Ct.dTT * Ct.c)
            oF = CoF.get(); oF.N_=out_; oF.rTT=sel_TT; oF.r=Ct.r; oF.c = Ft.c-Ct.c  # data, select in Nt.N_?
    return out_, r

def cluster_P(_C_, _c, root):  # FCM-style parallel centroid refine, may add proj_C

    cnt = 0  # r = root.r+1?:
    N_ = list(set([N for C in _C_ for N in C.N_]))  # all Ns are in all Cs
    Ln,Lc = len(N_),len(_C_); L=Lc*Ln  # localy constant
    _md__ = np.zeros((Ln, Lc, 2))  # NxC
    for j, N in enumerate(N_):
        for c,m,d in zip(N.root_,N.m_,N.d_):
            if c: i= _C_.index(c); _md__[j,i] = m,d
    while True:
        md__ = np.zeros_like(_md__)
        for j,N in enumerate(N_):
            for i,C in enumerate(_C_):
                TT,_ = base_comp(C,N); md__[j,i] = vt_(TT, ttcP)  # use rc?
        C_ = [sum2F(N_, root, md__[:,i,0], md__[:,i,1]) for i in range(Lc)]
        Mt = md__[:,:,0].sum()  # total V
        dM = np.abs(md__[:,:,0] -_md__[:,:,0]).sum()
        dD = np.abs(md__[:,:,1] -_md__[:,:,1]).sum()  # updates
        cnt += 1
        if Mt* (dM+dD)* (wcP*L) > ave* (root.r+ccP*L):
            _C_ = C_; _md__ = md__
        else: break  # weak * converged
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
                    N.root_ += [C]  # form C-to-N links:
                    C.L_= [CN(typ=1,N_=[N,C], dTT=N.dTT,c=N.c,r=N.r,m=N.m,d=N.d, span=np.hypot(*(dy_dx:=C.yx-N.yx)), angl=[dy_dx,np.sign(N.dTT[1]@ttcN[1])])]
                out_ += [C]
    if out_:
        dCt = sum2F(list(set(_C_)-set(out_)))  # compress
        oF = CoF.get(); oF.N_=C_; oF.rTT=dCt.dTT; oF.r=dCt.r; oF.c=_c-dCt.c
        return out_

def sum2F(N_, root=None, m_=[],d_=[], merge=0, froot=0, nF=None):  # -> CF/CL/CC/CN

    c_ = np.array([n.c for n in N_], dtype=float); C = c_.sum(); w_ = c_/C; N = N_[0]
    fC = np.any(m_)
    typ = 2 if fC else N.typ
    cls_ = [CF,CL,CC,CN]
    for i, (n,w) in enumerate(zip(N_,w_)):
        if i:
            TT+=n.dTT*w; R+=n.r*w; n_ += (n.N_ if merge else [n])
            if typ:  # links and higher
                kern+=n.kern*w; span+=n.span*w; yx+=n.yx*w
                if n.angl is not None: angl = copy(n.angl[0]) if angl is None else angl+n.angl[0]
                if typ==3: box=extend_box(box,n.box)
        else:  # init
            TT = n.dTT*w; R=n.r*w; n_ = copy(n.N_ if merge else [n])
            if typ:
                kern=n.kern*w; span=n.span*w; yx=n.yx*w; angl = copy(n.angl[0]) if n.angl is not None else None
                if typ==3: box=copy(n.box)
    F = (cls_[typ])(dTT=TT, c=C, r=R, nF=nF)
    if isinstance(F, CN): F.Nt.dTT = copy(TT); F.Nt.c = C; F.Nt.r = R  # update Nt's params
    if typ:
        F.N_ = n_; F.kern=kern; F.span=span; F.yx=yx
        if angl is not None: F.angl = [angl, np.sign(F.dTT[1] @ ttcP[1])]
        if typ == 3:
            if not F.Nt:  # direct H, frame only?
                for N in N_: add_H(F.H, N.H, F, fN = 1 if (N.H and isinstance(N.H[0],CN)) else 0)  # concat lower levs (follow existing H's type)
                F.H += [sum2F([n for N in N_ for n in N.N_], F)]  # add top lev
            F.m_,F.d_ = m_,d_; F.m, F.d = sum(m_),sum(d_)
            F.box = box
        else:
            F.m, F.d = vt_(TT)  # link or centroid
    else: F.N_=n_; F.m,F.d = vt_(TT)
    if root:
        F.wTT = root.wTT
        if typ!=2: add2F(root,F,2)  # skip centroids
        if typ==3 and getattr(root,'C_', None): root.Ct = sum2F(root.C_)
    if froot == 1:
        for n in N_: n.root = root or F
    elif froot == 2: F.root = root
    return F

def add2F(F, n, merge=0, fr=0, fo=0):  # unpack for batching in sum2F

    a = 'rTT' if fr else 'dTT'  # or wTT?
    if F.c:
        C=F.c+n.c; _w,w = F.c/C, n.c/C; F.r = F.r*_w + n.r*w; F.c = C
        if isinstance(F,CoF) and isinstance(n,CoF): F.fw += n.m  # sum subtree gain, fixed cost, extensive: no weighting?
        setattr(F, a, getattr(F,a)*_w + getattr(n,a)*w)
    else:
        setattr(F,a,getattr(n,a)); F.c=n.c; F.r=n.r
        if isinstance(F,CoF) and isinstance(n,CoF): F.fw=n.m
    if merge <2:
        if fo: F.typ_+= [n]
        else: F.N_ += (n.N_ if merge else [n])
    if hasattr(F,'H') and getattr(n,'H',None): add_H(F.H, n.H, F)
    if hasattr(n,'C_'): F.C_ = getattr(F,'C_',[]) + n.C_  # same for L_?
    return F

def add_H(H,h, root, fN=0):
    for Lev,lev in zip_longest(H, h):  # bottom-up
        if lev:
            if Lev: add2F(Lev,lev,1)  # unpack nested Hs in add2H
            else: H.append(Copy_(lev, root, cls=(CF,CN)[fN]))

def sum2G(ft_, fTT, root=None, init=1):  # core clustering function

    if not init:
        N_,_,ntt,nc,nr = ft_[0]; N_+=root.N_; ntt+=root.Nt.dTT; nc+=root.Nt.c; nr+=root.Nt.r; ft_[0] = N_,_,ntt,nc,nr
        if len(ft_)>1: L_,_,ltt,lc,lr=ft_[1]; L_+=root.L_; ltt+=root.Lt.dTT; lc+=root.Lt.c; lr+=root.Lt.r; ft_[1]=L_,_,ltt,lc,lr
    Ft_ = []
    for ft, nF in zip_longest(ft_,('Nt','Lt','Bt')):
        if ft: n_,_,tt,c,r = ft; Ft_+= [CF(N_=n_,nF=nF,dTT=tt,m=(vt:=vt_(tt,wTT))[0],d=vt[1],c=c,r=r)]
        else:  Ft_ += [CF()]
    C_ = [c for N in ft_[0][0] for c in N.C_]  # splice centroids
    Ft_ += [sum2F(list(set(C_)), root.Ct) if C_ else CF()]  # add multiple root_ in Cs?
    G = comb_Ft(*Ft_, root, wTT=fTT)
    N_ = G.N_; N=N_[0]; G.sub = N.sub+1 if G.L_ else N.sub; r=G.r
    if G.Lt:  # sub+
        Lt = G.Lt; L_,lm,ld,lr = Lt.N_,Lt.m,Lt.d,Lt.r; L=len(L_)
        Vn = (lm+ld)* (wcN*L) - (ave+avd)* (lr+1+ccN*L)
        if Vn > 0:
            c = G.Lt.c; E_ = get_exemplars({N for L in L_ for N in L.N_}, r,c)
            if Vn *(mdecay(L_)-decay) > (ave+avd)*(lr+1+ccC*L):
                r+=1; G_,r = cluster_C(G.Nt,E_,r,c)  # higher V, low decay, eval cluster_P
            else:     G_,r = cluster_N(G.Nt,E_,r,c)  # updates G
            L= (len(G_)-1) **2  # if full cross_comp?
            if G_ and vt_(G.Nt.dTT,G.wTT*tAgg)[0]*(wAgg*L) > ave*(r+cAgg*L) > 0:
                cross_comp(G.Nt,r)
    if G.Bt:
        Bt = G.Bt; bd,br,L = Bt.d,Bt.r,len(Bt.N_); rroot = root.root if root.root else 0
        if N.typ!=1 and bd*(wAgg*L) > avd*(br+cAgg*L): [F2N(L) for L in Bt.N_]; cross_comp(Bt, br)  # no ddfork
        if rroot: Bt.brrw = Bt.m * (rroot.m * (decay * (rroot.span/G.span)))  # external lend only, subtract from root?
    return G

def comb_Ft(Nt, Lt, Bt, Ct, root,wTT):  # from sum2G, default Nt

    G = CN(Nt=Nt,Lt=Lt,Bt=Bt,Ct=Ct, root=root); Nt.root=G; Lt.root=G; Bt.root=G; Ct.root=G
    T = Copy_(Nt)  # temporary accumulator
    dF_ = []
    for Ft in Lt, Bt:  # connectivity forks, Ct is not directly combined and compared, rdn only?
        if Ft: dF_ += [comp_F(T, Ft, root.r,G)]; T.dTT,T.c,T.r = sum_vt([T,Ft],wTT=wTT)  # Bt*brrw?
        else:  dF_ += [CF()]
    add2F(G,T,merge=2)
    if any(dF_): sum2F(dF_,G.Xt)  # cross-fork covariance
    add_Nt(G,Nt)  # add H,kern,ext, doesn't affect comp_F
    if Lt: add_Lt(G, Lt,wTT)
    G.m,G.d = vt_(G.dTT, wTT)  # recompute m and d after add_Nt and add_Lt above
    return G

def add_Nt(G, Nt, merge=0):  # in sum2G

    if isinstance(Nt,CN) and G.H and Nt.H:  # also Ct.H if separate?
        add_H(G.H if merge else G.H[:-1], Nt.H, G)  # else top lev = Nt.N_
    N_ = Nt.N_
    if merge: G.N_ += N_
    # i don't think we need this else now. G.Nt.N_ is always = Nt.N_ here, we don't need to add top level into H
    # else: G.H += [sum2F(N_+ (G.H.pop().N_ if G.H else []), G)]  # init or extend top level
    yx_ = []; C = G.c + Nt.c  # G is empty?
    for N in N_:
        N.fin = 1; N.root = G; c = N.c
        if hasattr(N,'m_'):
            if not hasattr(G,'m_'): G.root_,G.m_,G.d_ = [],[],[]  # or G.rm_,G.rd_?
            G.C_ += N.root_; G.m_+= N.m_; G.d_+= N.d_  # Ct || Nt
        G.kern = (G.kern*(C-c) + N.kern*c) / C  # massive?
        G.box = extend_box(G.box, N.box)
        yx_ += [N.yx]
    G.yx = yx = np.mean(yx_, axis=0); dy_,dx_ = (np.array(yx_)-yx).T  # weigh by c?
    G.span = np.hypot(dy_,dx_).mean() if len(N_)>1 else N.span

def add_Lt(G, Lt,wTT):  # addition to Q2R

    L_ = Lt.N_
    if Lt.m * wN > ave*(cN+Lt.r):  # comp typ -1 pre-links
        L_,pL_ = [],[]; [L_.append(L) if L.typ==1 else pL_.append(L) for L in Lt.N_]
        if pL_ and sum_vt(pL_,fm=1,wTT=wTT)[0]* ave*wN > ave*(cN * np.mean([L.r for L in pL_])):
            for L in pL_: L_ += [comp_N(*L.N_, G.r,L.c,1, L.angl[0], L.span)]
            sum2F(L_,Lt); add2F(G, Lt, merge=2)
    A = np.sum([l.angl[0] for l in L_], axis=0) if L_ else np.zeros(2)
    G.angl = [A, np.sign(G.dTT[1] @ ttcN[1])]
    G.mang = np.mean([comp_A(G.angl[0], l.angl[0])[0] for l in G.L_])  # Ls only?

# utilities:
def F2N(F):  # convert for cross_comp

    Nt = Copy_(F, root=F, cls=CF,typ=0); Nt.N_=F.N_; L_=F.L_  # replace in cross_comp
    F.__class__ = CN; F.Nt = Nt
    Na_ = dict(H=[], mang=1, box=np.array([np.inf,np.inf,-np.inf,-np.inf]), sub=0, exe=0, fin=0, root_=[], compared = set())
    if F.typ==0:  # CF | PP, no overlap for Cs
        Na_.update(kern=np.zeros(4), span=1, angl=None, yx=np.zeros(2))
    for k,v in Na_.items(): setattr(F, k, copy(v))
    if L_: F.H += [sum2F(L_, F)]
    for ft in ('Lt','Ct','Bt','Xt','Rt'): setattr(F, ft, CF(root=F))
    return F

def Copy_(N, root=None, r=1, cls=None, init=0, typ=None, froot=0):
    cls = cls or type(N)
    a = dict(dTT=N.dTT*r,m=N.m,d=N.d,c=N.c,r=N.r,fb_=list(N.fb_), root=root or N.root, nF=N.nF,wTT=copy(N.wTT),w=N.w, typ=N.typ if typ is None else typ,
             N_=copy(N.N_), L_=copy(N.L_))
    if isinstance(N, CL): a.update(yx=copy(N.yx), kern=copy(N.kern), span=N.span, angl=[copy(N.angl[0]), N.angl[1]] if N.angl else None)
    if isinstance(N, CC): a.update(m_=list(N.m_), d_=list(N.d_))
    if isinstance(N,CoF): a.update(call_=copy(N.call_), typ_=copy(N.typ_), fw=N.fw, fc=N.fc, fr=N.fr)
    C = cls(**a)
    if isinstance(N, CN):
        if init:
            C.yx=[N.yx]; C.angl=[copy(N.angl[0]), N.angl[1]] if N.angl is not None else None
            C.L_=[l for l in N.rim if l.m>ave]; N.root=C; C.fin=0; C.N_=[N]
        else:
            for f in ('Nt','Lt','Bt','Ct','Xt','Rt'): setattr(C, f, Copy_(getattr(N,f), root=C))
            C.H = [Copy_(lev, root=C) for lev in N.H]
            C.angl=deepcopy(N.angl); C.yx=copy(N.yx); C.box=copy(N.box); C.mang=N.mang; C.exe=N.exe; C.fin=N.fin; C.root_=list(N.root_)
    if froot:
        for n in C.N_+C.L_: n.root = C  # reassign for feedback, or root_+= if multiple roots?
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

def mdecay(L_):  # slope function
    L_ = sorted(L_, key=lambda l: l.span)
    dm_ = np.diff([l.m/l.c for l in L_])
    ddist_ = np.diff([l.span for l in L_])
    return - (dm_/ eps_(ddist_)).mean()  # -dm/ddist

# comp_slice hand-off:
def vect_edge(tile, rV=1):  # PP_ cross_comp and floodfill to init focal frame graph, no recursion:

    global ave,avd,Fw_,Fc_  # /= projected V change:
    def PP2N(PP):
        P_,L_,B_,verT,latT,A,S,box,yx, m,d,c = PP
        kern = np.array(latT[:4])
        [mM, mD, mI, mG, mA, mL], [dM, dD, dI, dG, dA, dL] = verT  # re-pack in dTT:
        dTT = np.array([ np.array([mM, mD, mL, mI, mG, mA, mL, mL / 2, 0]),  # extA=0
                         np.array([dM, dD, dL, dI, dG, dA, dL, dL / 2, 0])])
        y,x,Y,X = box; dy,dx = Y+1-y, X+1-x
        A = [np.array(A), np.sign(dTT[1] @ ttVct[1])]  # append sign
        PP = CL(typ=0, dTT=dTT,m=m,d=d,c=c,r=1, kern=kern,yx=yx,angl=A,span=np.hypot(dy/2,dx/2))  # set root in trace_edge
        m_, d_ = np.zeros(6), np.zeros(6); PP.B_ = B_
        for B in B_: m_ += B.verT[0]; d_ += B.verT[1];
        ad_ = np.abs(d_); t_ = m_ + ad_  # ~ max comparand
        m = m_/eps_(t_) @ w_t[0] - ave*2; d = ad_/eps_(t_) @ w_t[1] - avd*2
        PP.Bt = CF(N_=B_, m=m, d=d, r=2, root=PP,nF='Bt')
        for P in P_: P.root = PP
        if hasattr(P,'nt'):  # typ=1?
            PP.root_ = []  # Gd.root_: cores, no centroids? multiple PPms may share same PPd?
            for dP in P_:
                for P in dP.nt: PP.root_ += [P.root]  # PPm
        return PP
    blob_ = tile.N_; G_,TT,C,R = [],np.zeros((2,9)),0,0
    for blob in blob_:
        if not blob.sign and blob.G * wSE > ave * cSE:
            edge = slice_edge(blob, rV); L = len(edge.P_)-1
            if edge.G * (wCS*L) > sum([P.latT[4] for P in edge.P_]) * (cCS*L):
                PPm_ = comp_slice(edge, rV, ttVct)  # add comp_slice's weights?
                N_ = [PP2N(PPm) for PPm in PPm_]
                c = sum([PPm.c for PPm in N_]); C += c
                for PPd in edge.link_: PP2N(PPd)  # we don't form Gds?
                for N in N_:
                    if N.B_:
                        PPd_ = [B.root for B in N.B_]; sum2F(PPd_,N.Bt)
                        N.Bt.N_ = PPd_; [setattr(B,'root',N.Bt) for B in PPd_]
                L = len(PPm_)-1
                if sum(vt_(sum_vt(N_,wTT=ttVct)[0])) * (wVct*L) > (ave+avd) * (3+cVct*L):
                    G_,TT,C, R = trace_edge([F2N(n) for n in N_], G_,TT,c,3,tile)  # flatten B_-mediated Gs
    if G_:
        Nt = CF(nF='Nt', root=tile); Nt.N_=G_; Nt.dTT=TT; Nt.c=C; Nt.r=1; Nt.H=tile.H; tile.Nt=Nt; tile.dTT=TT; tile.c=C; L=len(G_)
        if vt_(TT,ttVct)[0]*(wFrm*L) > ave * (cFrm*L):  # L for trans-comp only?
            oF = CoF.get(); oF.N_=[N for G in G_ for N in G.N_]; oF.dTT=TT; oF.c+=C; oF.r+=R
            return tile

def trace_edge(N_,_G_,_TT,_C, r,root):  # cluster contiguous shapes via PPs in edge blobs or lGs in boundary/skeleton?

    L_, cT_, lTT, lc = [],set(),np.zeros((2,9)),0  # comp co-mediated Ns:
    for N in N_: N.fin = 0
    for N in N_:
        _N_ = [rN for B in N.B_ for rN in B.root_ if rN is not N]   # + node-mediated
        for _N in list(set(_N_)):  # share boundary or cores if lG with N, same val?
            cT = tuple(sorted((N.id,_N.id)))
            if cT in cT_: continue
            cT_.add(cT)
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx)  # Rc = r+ (N.r+_N.r)/2
            Link = comp_N(_N,N, r,_C,A=dy_dx, span=dist)
            if vt_(Link.dTT,ttTrc)[0] > ave*r:  L_+=[Link]  # r = 1|2, add Bt?
            if L_: lTT,lc,_ = sum_vt(L_,wTT=ttTrc)
    Gt_ = []
    for N in N_:  # flood-fill G per seed N
        if N.fin: continue
        N.fin=1; _N_=[N]; Gt=[]; N.root=Gt; n_,ntt,nc, l_,ltt,lc = [N],N.dTT.copy(),N.c or 1,[],np.zeros((2,9)),0  # Gt
        while _N_:
            _N = _N_.pop(0)
            for L in _N.rim:
                if L in L_:
                    n = L.N_[0] if L.N_[1] is _N else L.N_[1]
                    if n in N_:
                        if n.root is Gt: continue
                        if n.fin:  # merge n root
                            _root = n.root; n_+=_root[0];l_+=_root[3]; _root[6]=1
                            for _n in _root[0]: _n.root = Gt
                        else: n.fin=1; _N_+=[n]; n_+=[n]; l_+=[L]  # add single n
                        n.root = Gt
        ntt,nc,_ = sum_vt(n_, wTT=ttTrc)
        if l_: ltt,lc,_= sum_vt(l_, wTT=ttTrc); ltt*=lc/nc;
        Gt += [n_,ntt,nc, l_,ltt,lc, 0]; Gt_+=[Gt]
    G_, TT,C,R = [],np.zeros((2,9)),0,0
    for n_,ntt,nc,l_,ltt,lc,merged in Gt_:
        if not merged:
            if vt_(ntt+ltt,ttTrc)[0] > ave*r:  # wrap singletons, use Gt.r?
                TT += ntt+ltt; C += nc+lc; R += r*(nc+lc)  # add Bt?
                G_ += [sum2G([(n_,'Nt',ntt,nc,r)]+([(l_,'Lt',ltt,lc,r)] if l_ else []), ttTrc, root)]
            else:
                for N in n_: N.fin=0; N.root=root
    if sum(vt_(TT,root.wTT*ttTrc))*wTrc > (ave+avd)*(r+1+cTrc): _G_+=G_; _TT+=TT;_C+=C  # eval per edge, concat in tile?
    oF = CoF.get(); oF.N_= G_ or N_; oF.dTT=_TT; oF.c+=_C; oF.r+=r+R/_C
    return _G_, _TT, _C, r+R/_C

# frame expansion: cross_comp lower-tile N_,C_, forward results to next lev, project feedback to scan new lower windows
wY, wX = 64, 64; wYX = np.hypot(wY,wX)

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

def proj_TT(L, cos_d, dist, r, pTT, wTT, dec=1, fdec=0, frec=0):
    # accumulate link pTT with iTT | eTT internally, L may be N?

    dec = dist if fdec else ave ** (1 + (dist * dec) / L.span)  # not fully revised, ave = match decay rate / unit distance
    TT = np.array([L.dTT[0] * dec, L.dTT[1] * cos_d * dec])  # IxE angle alignment * decay?
    cert = abs(sum(vt_(TT,wTT)) * wPrj)  # approximation
    if cert > (ave+avd)*(r+cPrj): # certainty margin
        pTT+=TT; return
    if not frec:  # non-recursive
        for lev in L.Nt.N_:  # refine pTT
            proj_TT(lev, cos_d, dec, r+1, pTT, wTT, fdec=1, frec=1)
    pTT += TT  # L.dTT is redundant to H, neither is redundant to Bt,Ct
    if L.Bt:  # + trans-link tNt, tBt, tCt?
        TT = L.Bt.dTT
        if TT is not None:
            pTT += np.array([TT[0] * dec, TT[1] * cos_d * dec])

def proj_N(N, dist, A, r, _c, dec=1):  # arg rc += N.rc+Nw, recursively specify N projection val, add pN if comp_pN?

    cos_d = (N.angl[0].dot(A) / ((np.hypot(*N.angl[0]) * dist) or eps)) * N.angl[1] if N.angl else 0  # int x ext angle alignment, mean=0
    iTT,eTT = np.zeros((2,9)),np.zeros((2,9))
    wTT = CoF.get().wTT*ttPrj
    for L in N.L_+ N.B_: proj_TT(L, cos_d, dist, L.r+r, iTT, wTT, dec)  # accums TT internally
    if hasattr(N.Ct,'Lt') and N.Ct.Lt: [proj_TT(L,cos_d,dist,L.r+r,iTT,wTT,dec) for L in N.Ct.Lt.N_]  # C-to-N links
    for L in N.rim: proj_TT(L,cos_d,dist,L.r+r,eTT,wTT,dec)
    pTT = iTT + eTT  # projected int,ext links, work the same?
    oF = CoF.get(); oF.N_=[N]; oF.dTT=pTT; oF.c+=_c; oF.r+=r
    return pTT  # info_gain = N.m * average link uncertainty, should be separate
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
def frame_H(image, iY,iX, Ly,Lx, Y,X, rV, max_elev=4):  # fH=0: tiles, 1:same_filter_levs, 2:same_oF_levs

    def base_tile(y,x, elev):  # lower T if elev else pixels
        if elev:
            T = frame_H(image, y,x, Ly,Lx, Y,X, rV, max_elev=elev)  # smaller-scope unit
        else:
            T = vect_edge(frame_blobs_root(comp_pixel(image[y:y+Ly, x:x+Lx]), rV), rV)
            if T: T.yx=np.array([y+Ly//2,x+Lx//2]); T.box=np.array([y,x,min(y+Ly,Y),min(x+Lx,X)]); T.span=np.hypot(Ly,Lx)/2
        return T

    def expand_lev(_iy,_ix, elev, T):  # seed tile is pixels in 1st lev, or Fg in higher levs

        frame = np.full((Ly, Lx), None, dtype=object)  # level scope
        iy,ix = _iy,_ix; cy,cx = (Ly-1)//2, (Lx-1)//2; y,x = cy,cx  # start=mean
        T_, PV__,C,R = [],np.zeros([Ly,Lx]),0,0  # tiles, maps to level frame
        while True:
            if not elev: T = base_tile(iy,ix,0)  # T=0
            if T and sum(vt_(T.dTT, T.wTT*ttFrm)) * wFrm > (ave+avd)*(T.r+cFrm):  # complex gate, make incremental?
                frame[y,x] =T; T_+=[T]  # loop adds one tile to level
                dy_dx = np.array([T.yx[0]-y, T.yx[1]-x])
                pTT = proj_N(T, np.hypot(*dy_dx), dy_dx, elev,T.c)
                if 0 < sum(vt_(pTT, T.wTT*ttFrm)) * wFrm < ave * cFrm:  # extend lev by combined proj T_
                    proj_focus(PV__,y,x, T)  # PV__+= pV__
                    pv__ = PV__.copy(); pv__[frame != None] = 0  # exclude processed
                    y, x = np.unravel_index(pv__.argmax(), PV__.shape)
                    if PV__[y,x] > ave:
                        iy = _iy+ (y-cy)*(T.box[2]-T.box[0]); ix = _ix+ (x-cx)*(T.box[3]-T.box[1])  # step by sub-frame footprint
                        if elev:
                            T = base_tile(iy,ix, elev)  # lower frame_H, up to current level (this may update filters and oF too? So that feedback should be for top layer only?)
                    else: break
                else: break
            else: break
        if T_:
            TT,C,R = sum_vt(T_, wTT=ttFrm); R+= elev
            if sum(vt_(TT,ttFrm))*(C*wFrm) > (ave+avd)*(R+cFrm):   # cancel weak frame, null T_,C,R
                return T_,C,R
        return [], 0, 0
    elev = 0
    aTT = oTT = np.zeros((2,9))  # regime refs across levs
    tile, Fr = [],[]; global ave,avd  # update / ffeedback
    while elev < max_elev:
        tile_,C,R = expand_lev(iY,iX, elev, tile)  # project from seed tile
        if tile_: # sparse,2D
            Fr = sum2F(tile_)  # higher-scope tile( oH( aH
            if cross_comp(Fr.Nt, rr=elev)[0]:  # spec-> tN_,tC_,tL_, proj comb N_'L_?
                if elev:
                    aTT,oTT = ffeedback(Fr, aTT,oTT)  # term,form oH(aH
                elev += 1; tile = Fr  # next-extension seed
            else: break
        else: break
    if Fr: oF = CoF.get(); oF.N_=[Fr]; oF.dTT=Fr.dTT; oF.c+=Fr.c; oF.r+=Fr.r
    return Fr  # intra-lev feedback

def ffeedback(frame, aTT,oTT):  # recompute filters from regime drift; fork: reform oF_ on cross-regime drift

    global ave,avd
    if aH := pack_seg(frame,'aH',wBac, cBac, aTT):
        ave, avd = vt_(aH.dTT)  # filters *= ave
        aTT = aH.dTT
        if oH := pack_seg(frame,'oH', wBac, cBac**2, oTT):
            split_oF_(); cluster_oF_()  # reform oF_
            oTT = oH.dTT
    oF = CoF.get(); oF.N_=frame.H; oF.dTT+=frame.dTT; oF.c+=frame.c; oF.r+=frame.r
    return aTT,oTT

def pack_seg(frame, nF, w, c, _dTT):  # drift-gated regime termination for aH and oH

    H = frame.H
    n = ('aH','oH') if nF=='aH' else ('oH',)
    i = next((j+1 for j in reversed(range(len(H))) if H[j].nF in n), 0)  # packed levs
    if tail := H[i:] if nF=='aH' else [l for l in H[i:] if l.nF=='aH']:  # lev_| aH_
        D = sum(np.sum(np.abs(t.dTT-_dTT) * wTT) for t in tail)  # drift
        if (vD := D*w - ave*c) > 0:  # update value
            seg = CN(nF=nF, H=tail, root=frame); seg.dTT = tail[-1].dTT; seg.c = sum(l.c for l in tail)  # default regime summary
            if vD > ave: seg.dTT,seg.c,seg.r = sum_vt(tail); seg.m,seg.d = vt_(seg.dTT)  # deep summary
            frame.H = H[:i]+[seg]  # append
            return seg

if __name__ == "__main__":  # './images/toucan_small.jpg' './images/raccoon_eye.jpeg', add larger global image
    trace_func(vars())
    comp_slice = CoF.traced(comp_slice); slice_edge = CoF.traced(slice_edge)  # add traced to comp_slice and slice_edge

    parse_funcs(["agg_recursion.py","comp_slice.py","slice_edge.py"])  # populate nF_
    F_body_()  # fill with AST
    Y, X = imread('./images/toucan.jpg').shape
    frame = frame_H(image=imread('./images/toucan.jpg'), iY=Y//2 -31, iX=X//2 -31, Ly=64,Lx=64, Y=Y, X=X, rV=1)
    # search frames ( tiles inside image, at this size it should be 4K, or 256K panorama, won't actually work on toucan