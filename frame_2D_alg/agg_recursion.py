import numpy as np, weakref
from copy import copy, deepcopy
from math import atan2, cos, floor, pi  # from functools import reduce
from itertools import zip_longest, combinations, chain, product;  from multiprocessing import Pool, Manager
from frame_blobs import frame_blobs_root, imread, unpack_blob_, comp_pixel, CBase
from slice_edge import slice_edge
from comp_slice import comp_slice
'''
Lower modules start with cross-comp and clustering image pixels, here the initial input is PPs: segments of matching blob slices or Ps.

This is a main module of open-ended dual clustering algorithm, designed to discover empirical patterns of indefinite complexity. 
It's a recursive 4-stage cycle, forming agglomeration levels: 
generative cross-comp, compressive clustering, filter-adjusting feedback, code-extending forward (not yet implemented): 

Cross-comp forms Miss, Match: min= shared_quantity for directly predictive params, else inverse deviation of miss=variation, 2 forks:
rng+: incremental-range cross-comp nodes: edge segments at < max distance, cluster if they match. 
der+: incremental-derivation cross-comp links, from node cross-comp, if abs_diff * rel_adjacent_match

Clustering is compressively grouping the elements, by direct similarity to centroids or transitive similarity in graphs, 2 forks:
nodes: connectivity clustering / >ave M, progressively reducing overlap by exemplar selection, centroid clustering, floodfill.
links: correlation clustering if >ave D, forming contours that complement adjacent connectivity clusters.

That forms hierarchical graph representation: dual tree of down-forking elements: node_H, and up-forking clusters: root_H:
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
Similar to neurons: dendritic input tree and axonal output tree, but with lateral cross-comp and nested param sets per layer.

Induction / pre- comp_N projection:
feedback of projected match adjusts filters to maximize next match, including coordinate filters that select new inputs.
(initially ffeedback, can be refined by cross_comp of co-projected patterns: "imagination, planning, action" section of part 3)

Deduction:
in feedforward, code += [ops] that formed new layer of syntax, extended to process cross-layer diffs, if any? 
or cross-comp function calls and cluster code blocks of past and simulated processes? (not yet implemented)

notation:
prefix  f denotes flag
postfix t denotes tuple, multiple ts is a nested tuple
prefix  _ denotes prior of two same-name vars, multiple _s for relative precedence
postfix _ denotes array of same-name elements, multiple _s is nested array
capitalized vars are summed small-case vars
'''
eps = 1e-7
class CLay(CBase):  # layer of derivation hierarchy, subset of CG
    name = "lay"
    def __init__(l, **kwargs):
        super().__init__()
        l.Et = kwargs.get('Et', np.zeros(2))
        l.rc = kwargs.get('rc', 1)  # ave nodet overlap
        l.node_ = kwargs.get('node_', [])  # concat across fork tree
        l.link_ = kwargs.get('link_', [])
        l.derTT = kwargs.get('derTT', np.zeros((2,9)))  # sum m_,d_ [M,D,n, I,G,A, L,S,eA] across fork tree
        # i: lay index in root node_,link_, to revise olp; i_: m,d priority indices in comp node|link H
        # ni: exemplar in node_, trace in both directions?
    def __bool__(l): return bool(l.node_)

    def copy_(lay, rev=0, i=None):  # comp direction may be reversed to -1, currently not used
        if i:  # reuse self
            C = lay; lay = i; C.node_=copy(i.node_); C.link_ = copy(i.link_); C.derTT=np.zeros((2,9))
        else:  # init new C
            C = CLay(node_=copy(lay.node_), link_=copy(lay.link_))
        C.Et = copy(lay.Et)
        for fd, tt in enumerate(lay.derTT):  # nested array tuples
            C.derTT[fd] += tt * -1 if (rev and fd) else deepcopy(tt)

        if not i: return C

    def add_lay(Lay, lay, rev=0, rn=1):  # no rev, merge lays, including mlay + dlay

        # rev = dir==-1, to sum/subtract numericals in m_,d_
        for fd, Fork_, fork_ in zip((0,1), Lay.derTT, lay.derTT):
            Fork_ += (fork_ * -1 if (rev and fd) else fork_) * rn  # m_| d_| t_
        # concat node_,link_:
        Lay.node_ += [n for n in lay.node_ if n not in Lay.node_]
        Lay.link_ += lay.link_
        Lay.Et += lay.Et * rn
        Lay.rc = (Lay.rc + lay.rc * rn) /2
        return Lay

    def comp_lay(_lay, lay, rn, root):  # unpack derH trees down to numericals and compare them

        derTT = comp_derT(_lay.derTT[1], lay.derTT[1] * rn)  # ext A align replaced dir/rev
        Et = np.array([derTT[0] @ wTTf[0], np.abs(derTT[1]) @ wTTf[1]])
        if root: root.Et[:2] += Et  # no separate n
        node_ = list(set(_lay.node_+ lay.node_))  # concat, or redundant to nodet?
        link_ = _lay.link_ + lay.link_
        return CLay(Et=Et, rc=(_lay.rc+lay.rc*rn)/2, node_=node_, link_=link_, derTT=derTT)

class CN(CBase):
    name = "node"
    def __init__(n, **kwargs):
        super().__init__()
        n.fi = kwargs.get('fi', 1)  # if G else 0, fd_: list of forks forming G?
        n.N_ = kwargs.get('N_',[])  # nodes, or ders in links
        n.L_ = kwargs.get('L_',[])  # links if fi else len nodet.N_s?
        n.nH = kwargs.get('nH',[])  # top-down hierarchy of sub-node_s: CN(sum_N_(Nt_))/ lev, with single added-layer derH, empty nH
        n.lH = kwargs.get('lH',[])  # bottom-up hierarchy of L_ graphs: CN(sum_N_(Lt_))/ lev, within each nH lev
        n.Et = kwargs.get('Et',np.zeros(3))  # sum from L_, cent_?
        n.et = kwargs.get('et',np.zeros(3))  # sum from rim, altg_?
        n.rc = kwargs.get('rc',1)  # redundancy to ext Gs, ave in links? separate rc for rim, or internally overlapping?
        n.baseT = kwargs.get('baseT', np.zeros(4))  # I,G,A: not ders
        n.derTT = kwargs.get('derTT',np.zeros((2,9)))  # sum derH -> m_,d_ [M,D,n, I,G,A, L,S,eA], dertt: comp rims + overlap test?
        n.derH  = kwargs.get('derH',[])  # sum from L_ or rims
        n.yx   = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.rng  = kwargs.get('rng',1)  # or med: loop count in comp_node_|link_
        n.box  = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.span = kwargs.get('span',0) # distance in nodet or aRad, comp with baseT and len(N_) but not additive?
        n.angl = kwargs.get('angl',np.zeros(2))  # dy,dx, sum from L_
        n.mang = kwargs.get('mang',1)  # ave match of angles in L_, = identity in links
        n.rim = kwargs.get('rim',[])  # node-external links, rng-nested? set?
        n.root  = kwargs.get('root', [])  # immediate
        n.altg_ = kwargs.get('altg_',[])  # ext contour Gs, replace/combine rim?
        n.C_    = kwargs.get('C_',[])  # int centroids, replace/combine N_?
        n.R_    = kwargs.get('R_', [])  # root centroids
        n.fin = kwargs.get('fin',0)  # clustered, temporary
        n.exe = kwargs.get('exe',0)  # exemplar, temporary
        n.compared = set()
        # n.fork_tree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def __bool__(n): return bool(n.N_)

def Copy_(N, root=None, init=0):

    C = CN(root=root)
    if init:  # init G|C with N
        C.N_ = [N]; C.nH, C.lH, N.root = [],[],C
        C.L_ = N.rim if N.fi else N.L_
    else:
        C.N_,C.nH,C.lH = list(N.N_),list(N.nH),list(N.lH); N.root = root or N.root
        C.L_ = list(N.L_) if N.fi else N.L_
    C.derH  = [lay.copy_() for lay in N.derH]
    C.derTT = deepcopy(N.derTT)
    for attr in ['Et','baseT','yx','box','angl','rim','altg_','C_']: setattr(C, attr, copy(getattr(N, attr)))
    for attr in ['rc','rng', 'fi', 'fin', 'span', 'mang']: setattr(C, attr, getattr(N, attr))  # add centroid attrs?
    return C

ave, avd, arn, aI, aS, aveB, aveR, Lw, intw, loopw, centw, contw = 10, 10, 1.2, 100, 5, 100, 3, 5, 2, 5, 10, 15  # value filters + weights
adist, amed, distw, medw = 10, 3, 2, 2  # cost filters + weights, add alen?
wM, wD, wN, wO, wG, wL, wI, wS, wa, wA = 10, 10, 20, 10, 20, 5, 20, 2, 1, 1  # der params higher-scope weights = reversed relative estimated ave?
mW = dW = 9; wTTf = np.ones((2,9))  # fb weights per derTT, adjust in agg+
wY = wX = 64; wYX = np.hypot(wY,wX)  # focus dimensions
'''
initial PP_ cross_comp and connectivity clustering to initialize focal frame graph, no recursion:
'''
def vect_root(Fg, rV=1, wTTf=[]):  # init for agg+:
    if np.any(wTTf):
        global ave, avd, arn, aveB, aveR, Lw, adist, amed, intw, loopw, centw, contw, wM, wD, wN, wO, wG, wL, wI, wS, wa, wA
        ave, avd, arn, aveB, aveR, Lw, adist, amed, intw, loopw, centw, contw = (
            np.array([ave,avd, arn,aveB,aveR, Lw, adist, amed, intw, loopw, centw, contw]) / rV)  # projected value change
        wTTf = np.multiply([[wM, wD, wN, wO, wG, wL, wI, wS, wa, wA]], wTTf)  # or dw_ ~= w_/ 2?
        wTTf = np.delete(wTTf,(2,3), axis=1)  #-> comp_slice, = np.array([(*wTTf[0][:2],*wTTf[0][4:]),(*wTTf[0][:2],*wTTf[1][4:])])
    blob_ = unpack_blob_(Fg); N_ = []
    for blob in blob_:
        if not blob.sign and blob.G > aveB:
            edge = slice_edge(blob, rV)
            if edge.G * ((len(edge.P_)-1)* Lw) > ave * sum([P.latT[4] for P in edge.P_]):
                N_ += comp_slice(edge, rV, wTTf)
    Fg.N_ = [PP2N(PP, Fg) for PP in N_]

def val_(Et, fi=1, mw=1, aw=1, _Et=np.zeros(3)):  # m,d eval per cluster or cross_comp

    if mw <= 0: return (0, 0) if fi == 2 else 0
    am = ave * aw  # includes olp, M /= max I | M+D? div comp / mag disparity vs. span norm
    ad = avd * aw  # dval is borrowed from co-projected or higher-scope mval
    m, d, n = Et
    # m,d may be negative, but deviation is +ve?
    if fi==2: val = np.array([m-am, d-ad]) * mw
    else:     val = (m-am if fi else d-ad) * mw  # m: m/ (m+d), d: d/ (m+d)?
    if _Et[2]:
        _m,_d,_n = _Et
        rn =_n/n  # borrow rational deviation of contour if fi else root Et:
        if fi==2: val *= np.array([_d/ad, _m/am]) * mw * rn
        else:     val *= (_d/ad if fi else _m/am) * mw * rn
    return val

''' Core process per agg level, as described in top docstring:

- Cross-comp nodes, evaluate incremental-derivation cross-comp of new >ave difference links, recursively. 
- Select exemplar nodes, links, centroid-cluster their rim nodes, selectively spreading out within a frame. 

- Connectivity-cluster select exemplars/centroids by >ave match links, correlation-cluster links by >ave diff.
- Form complemented (core+contour) clusters for recursive higher-composition cross_comp. 

- Forward: extend cross-comp and clustering of top clusters across frames, re-order centroids by eigenvalues.
- Feedback coords to bottom level or prior-level in parallel pipelines, filter updates in more coarse cycles '''

def cross_comp(root, rc, fC=0):  # rng+ and der+ cross-comp and clustering

    N_,L_,Et = comp_Q(root.N_, rc, fC)  # rc: redundancy+olp, lG.N_ is Ls
    if len(L_) > 1:
        mV,dV = val_(Et,2, (len(L_)-1)*Lw, rc+loopw); lG = []
        if dV > 0:
            if root.L_: root.lH += [sum_N_(root.L_)]  # replace L_ with agg+ L_:
            root.L_=L_; root.Et += Et
            if fC < 2 and dV > avd:  # no comp ddCs
                lG = CN(cent_=L_) if fC else CN(N_=L_)
                lG = cross_comp(lG, rc+1, fC*2)  # dfork
                if lG:  # batched lH extension
                    rc+=lG.rc; root.lH += [lG]+lG.nH; root.Et+=lG.Et; root.derH+=lG.derH  # new lays
        if mV > 0:
            nG = Cluster(root, N_, rc, fC)  # get_exemplars, cluster_C, rng connectivity cluster
            if nG:  # batched nH extension
                rc+=nG.rc  # redundant clustering layers
                if lG: comb_altg_(nG,lG, rc+1)  # assign G contours (skip if lG is empty)
                if val_(nG.Et,1, (len(nG.N_)-1)*Lw, rc+loopw, Et) > 0:
                    nG = cross_comp(nG, rc) or nG  # agg+, -> cross-frame? cross-comp C_?
                _H = root.nH; root.nH = []
                nG.nH = _H + [root] + nG.nH  # pack root in Nt.nH, has own L_,lH
                return nG  # recursive feedback, add to root

def comb_altg_(nG, lG, rc):  # cross_comp contour/background per node:

    for Lg in lG.N_:
        altg_ = {n.root for L in Lg.N_ for n in L.N_ if n.root and n.root.root}  # rdn core Gs, exclude frame
        if altg_:
            altg_ = {(core,rdn) for rdn,core in enumerate(sorted(altg_, key=lambda x:(x.Et[0]/x.Et[2]), reverse=True), start=1)}
            Lg.altg_ = [altg_, np.sum([i.Et for i,_ in altg_], axis=0)]
    def R(L):
        if L.root: return L.root if L.root.root is lG else R(L.root)
        else:      return None
    for Ng in nG.N_:
        Et, Rdn, altl_ = np.zeros(3), 0, []  # contour/core clustering
        LR_ = {R(L) for n in Ng.N_ for L in n.rim}  # lGs, individual rims are too weak
        for LR in LR_:
            if LR and LR.altg_:  # not None, eval Lg.altg_[1]?
                for core, rdn in LR.altg_[0]:  # map contour rdns to core N:
                    if core is Ng:
                        altl_ += [LR]; Et += core.Et; Rdn += rdn  # add to Et[2]?
        if altl_:
            aG = CN(N_=altl_, Et=Et)
            if val_(Et,0,(len(altl_)-1)*Lw, rc+Rdn+loopw) > 0:  # norm by core_ rdn
                aG = cross_comp(aG, rc)
            Ng.altg_ = [list(set(aG.N_)), aG.Et]

def proj_V(_N, N, angle, dist, fC=0):  # estimate cross-induction between N and _N before comp

    def proj_L_(L_, int_w=1):
        V = 0
        for L in L_:
            cos = (L.angl @ angle) / (np.hypot(*L.angl) * np.hypot(*angle))  # angl is [dy,dx]
            mang = (cos+ abs(cos)) / 2  # = max(0, cos): 0 at 90°/180°, 1 at 0°
            V += L.Et[1-fi] * mang * int_w * mW * (L.span/dist * av)  # decay = link.span / l.span * cosine ave?
            # += proj rim-mediated nodes?
        return V
    V = 0; fi = N.fi; av = ave if fi else avd
    for node in _N,N:
        if node.et[2]:  # mainly if fi?
            v = node.et[1-fi]  # external, already weighted?
            V+= proj_L_(node.rim) if v*mW > av*node.rc else v  # too low for indiv L proj
        v = node.Et[1-fi]  # internal, lower weight
        V+= proj_L_(node.L_,intw) if (fi and v*mW > av*node.rc) else v  # empty link L_
    if fC:
        V += np.sum([i[0] for i in _N.mo_]) + np.sum([i[0] for i in N.mo_])
        # project matches to centroids?
    return V

def comp_Q(iN_, rc, fC):  # comp pairs of nodes or links within max_dist

    N__,L_,ET = [],[], np.zeros(3); rng,olp_,_N_ = 1,[],copy(iN_)
    while True:  # _vM, rng in rim only?
        N_,Et = [],np.zeros(3)
        for _N, N in combinations(_N_, r=2):  #| proximity-order for min ders?
            if _N in N.compared or len(_N.nH) != len(N.nH):  #| comp top nodes | depth?
                continue
            if fC==2: # dCs
                m_,d_ = comp_derT(_N.derTT[1], N.derTT[1])
                ad_ = np.abs(d_); t_ = m_+ ad_+ eps  # ~ max comparand
                et = np.array([m_/t_ @ wTTf[0], ad_/t_ @ wTTf[1], min(_N.Et[2], N.Et[2])])  # signed
                dC = CN(N_= [_N,N], Et=Et); L_ += [dC]; Et += et
                for n in _N,N: N_+= [n]; n.rim += [dC]; n.et+=et
            else:
                dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx); olp = (N.rc +_N.rc) / 2
                if fC or ({l for l in _N.rim if l.Et[0]>ave} & {l for l in N.rim if l.Et[0]>ave}):  # +ve
                    fcomp = 1  # x all Cs, or connected by common match, which means prior bilateral proj eval
                else:
                    V = proj_V(_N,N, dy_dx, dist)  # eval _N,N cross-induction for comp
                    fcomp = adist * V/olp > dist  # min induction
                if fcomp:
                    Link = comp_N(_N,N, olp, rc, lH=L_, angl=dy_dx, span=dist, rng=rng)
                    if val_(Link.Et, aw=loopw+olp) > 0:
                        N_ += [_N,N]; Et += Link.Et; olp_ += [olp]
        N_ = list(set(N_))
        if fC:
            N__ = N_; ET = Et; break  # no rng-banding
        elif N_:
            N__ += [N_]; ET += Et  # rng+ eval:
            if not fC and val_(Et, mw=(len(N_)-1)*Lw, aw= loopw+ (sum(olp_) if olp_ else 1)) > 0:  # current-rng vM
                _N_ = N_; rng += 1; olp_ = []  # reset, for eval only?
            else: break  # low projected rng+ vM
        else: break
    return N__, L_, ET

def comp_N(_N,N, olp,rc, lH=None, angl=np.zeros(2), span=None, fdeep=0, rng=1):  # compare links, optional angl,span,dang?

    derTT, Et, rn = min_comp(_N,N)  # -1 if _rev else 1, d = -d if L is reversed relative to _L, obsoleted by angle?
    baseT = (rn*_N.baseT+ N.baseT) /2  # not derived
    yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])  # primary ext attrs
    fi = N.fi
    Link = CN(Et=Et,rc=olp, N_=[_N,N], L_=len(_N.N_+N.N_), et=_N.et+N.et, baseT=baseT,derTT=derTT, yx=yx,box=box, span=span,angl=angl, rng=rng, fi=0)
    Link.derH = [CLay(Et=Et[:2], node_=[_N,N],link_=[Link],derTT=copy(derTT), root=Link)]
    if fdeep:
        if val_(Et,1, len(N.derH)-2, olp+rc) > 0 or fi==0:  # else derH is dext,vert
            Link.derH += comp_H(_N.derH,N.derH, rn, Et, derTT, Link)  # append
        if fi:
            for falt, _N_,N_ in (0,1), (_N.N_,N.N_), (_N.altg_,N.altg_):  # cent_s overlap, use for cross-ref only?
                if falt: _N_,N_ = _N_[0],N_[0]  # eval Et?
                if (_N_ and N_) and isinstance(N_[0],CN) and isinstance(_N_[0],CN):  # not PP
                    spec(_N_,N_, olp,rc,Et, Link.lH)  # for dspe?
    if fdeep==2: return Link  # or Et?
    if lH is not None:  lH += [Link]
    for n, _n, rev in zip((N,_N),(_N,N),(0,1)):  # if rim-mediated comp: reverse dir in _N.rim: rev^_rev
        n.rim += [Link]; n.et += Et
        n.compared.add(_n)
    return Link

def min_comp(_N,N):  # comp Et, baseT, extT, derTT

    fi = N.fi
    _M,_D,_n =_N.Et; _I,_G,_Dy,_Dx =_N.baseT; _L = len(_N.N_) if fi else _N.L_  # len nodet.N_s
    M, D, n  = N.Et; I, G, Dy, Dx = N.baseT; L = len(N.N_) if fi else N.L_
    _pars = np.array([_M,_D,_n,_I,_G, np.array([_Dy,_Dx]),_L,_N.span], dtype=object)  # Et, baseT, extT
    pars = np.array([M,D,n, (I,aI),G, np.array([Dy,Dx]), L,(N.span,aS)], dtype=object)
    rn = _n/n
    mA, dA = comp_A(rn*_N.angl, N.angl)
    m_, d_ = comp(rn*_pars,pars, mA,dA)  # -> M,D,n, I,G,A, L,S,eA
    md_,dd_= comp_derT(rn*_N.derTT[1], N.derTT[1])
    m_+= md_; d_+= dd_
    DerTT = np.array([m_,d_])
    ad_ = np.abs(d_); t_ = m_+ ad_+ eps  # max comparand
    Et = np.array([m_* (1, mA)[fi] /t_ @ wTTf[0],  # norm but signed?
                   ad_* (1, 2-mA)[fi] /t_ @ wTTf[1], min(_n,n)])  # shared scope?
    '''
    if np.hypot(*_N.angl)*_N.mang + np.hypot(*N.angl)*N.mang > ave*wA:  # aligned L_'As, mang *= (len_nH)+fi+1
    mang = (rn*_N.mang + N.mang) / (1+rn)  # ave, weight each side by rn
    align = 1 - mang* (1-mA)  # in 0:1, weigh mA '''
    return DerTT, Et, rn

def comp_derT(_i_,i_):

    d_ = _i_ - i_
    _a_, a_ = np.abs(_i_), np.abs(i_)
    m_ = np.minimum(_a_, a_)
    m_[(_i_<0)!=(i_<0)] *= -1  # negate opposite signs in-place
    # m_ += np.maximum(_a_, a_)
    return np.array([m_,d_])

def comp(_pars, pars, meA=0, deA=0):  # compute +ve m_, signed d_ from raw inputs or derivatives

    m_,d_ = [],[]
    for _p, p in zip(_pars, pars):
        if isinstance(_p, np.ndarray):
            mA, dA = comp_A(_p, p)
            m_ += [mA]; d_ += [dA]
        elif isinstance(p, tuple):  # massless I|S avd in p only
            p, avd = p
            d = _p - p; ad = abs(d)
            m_ += [avd-ad]  # + complement max(avd,ad)?
            d_ += [d]
        else:  # massive
            _a,a = abs(_p), abs(p)
            m_ += [min(_a,a) if _p<0==p<0 else -min(_a,a)]  # + complement max(_a,a) for +ves?
            d_ += [_p - p]
    # add ext A:
    return np.array(m_+[meA]), np.array(d_+[deA])

def comp_A(_A,A):

    dA = atan2(*_A) - atan2(*A)
    if   dA > pi: dA -= 2 * pi  # rotate CW
    elif dA <-pi: dA += 2 * pi  # rotate CCW
    '''  or 
    den = np.hypot(*_A) * np.hypot(*A) + eps
    mA = (_A @ A / den +1) / 2  # cos_da in 0:1, no rot = 1 if dy * _dx - dx * _dy >= 0 else -1  # +1 CW, −1 CCW, ((1-cos_da)/2) * rot?
    '''
    return (cos(dA)+1) /2, dA/pi  # mA in 0:1, dA in -1:1

def spec(_spe,spe, olp,rc, Et, dspe=None, fdeep=0):  # for N_|cent_ | altg_
    for _N in _spe:
        for N in spe:
            if _N is not N:
                dN = comp_N(_N, N, olp,rc, fdeep=2); Et += dN.Et  # may be d=cent_?
                if dspe is not None: dspe += [dN]  # this problem is still persists, dspe is int for links
                if fdeep:
                    for L,l in [(L,l) for L in _N.rim for l in N.rim]:  # l nested in L
                        if L is l: Et += l.Et  # overlap val
                    if _N.altg_ and N.altg_: spec(_N.altg_,N.altg_, olp,rc, Et)

def rolp(N, _N_, R=0): # rel V of L_|N.rim overlap with _N_: inhibition|shared zone, oN_ = list(set(N.N_) & set(_N.N_)), no comp?

    n_ = N.N_ if R else {n for l in N.rim for n in l.N_ if n is not N}  # nrim
    olp_ = n_ & _N_
    if olp_:
        oEt = np.sum([i.Et for i in olp_], axis=0)
        _Et = N.Et if R else N.et  # not sure
        rV = (oEt[0]/oEt[2]) / (_Et[0]/_Et[2])
        return rV * val_(N.et,1, aw=centw)  # contw for cluster?
    else:
        return 0

def get_exemplars(N_, rc):  # get sparse nodes by multi-layer non-maximum suppression

    E_ = set()
    for rdn, N in enumerate(sorted(N_, key=lambda n: n.et[0]/n.et[2], reverse=True),start=1):  # strong-first
        roV = rolp(N, E_)
        if val_(N.et,1, aw = rc + rdn + loopw + roV) > 0:  # ave *= relV of overlap by stronger-E inhibition zones
            E_.update({n for l in N.rim for n in l.N_ if n is not N and val_(n.Et,1,aw=rc) > 0})  # selective nrim
            N.exe = 1  # in point cloud of focal nodes
        else:
            break  # the rest of N_ is weaker, trace via rims
    return E_

def Cluster(root, N_, rc, fC):  # generic root for clustering

    nG = []
    if fC:  # centroids -> primary connectivity clustering
        L_, nG = [], []
        dC_ = sorted(list({L for C in N_ for L in  C.rim}), key=lambda dC: dC.Et[1])  # from min D
        for i, dC in enumerate(dC_):
            if val_(dC.Et, fi=0, aw=rc+loopw) < 0:  # merge similar centroids, no recomp
                _C,C = dC.N_
                if _C is not C:  # not merged
                    add_N(_C,C, fmerge=1,froot=1)  # fin,root.rim
                    for l in C.rim: l.N_ = [_C if n is C else n for n in l.N_]
            else:
                root.L_ = [l for l in root.L_ if l not in dC_[:i]]  # cleanup
                L_ = dC_[i:]; C_ = list({n for L in L_ for n in L.N_})  # remaining Cs and dCs
                if L_ and val_(np.sum([l.Et for l in L_], axis=0), mw=(len(L_)-1)*Lw, aw=rc+contw) > 0:
                    nG = cluster_n(root, C_,rc)  # by connectivity in feature space
                if not nG: nG = CN(N_=C_,L_=L_)
                break
    else:
        F_ = list(set([N for n_ in N_ for N in n_]))  # flat
        E_ = get_exemplars(F_, rc)
        if E_ and val_(np.sum([g.Et for g in E_],axis=0), F_[0].fi, (len(E_)-1)*Lw, rc+centw, root.Et) > 0:  # any rng
            cluster_C(E_, root, rc)
        for rng, rN_ in enumerate(N_, start=1):  # bottom-up rng-banded clustering
            aw = rc + rng + contw
            if rN_ and val_(np.sum([n.Et for n in rN_], axis=0),1, (len(rN_)-1)*Lw, aw) > 0:
                nG = cluster_N(root, F_, rN_, aw, rng) or nG
    # top valid nG:
    return nG

def cluster_N(root, iN_, rN_, rc, rng=1):  # flood-fill node | link clusters

    def rroot(n):
        if n.root and n.root.rng > n.rng: return rroot(n.root) if n.root.root else n.root
        else:                             return None

    def extend_Gt(_link_, node_, cent_, link_, long_, in_):
        for L in _link_:  # spliced rim
            if L in in_: continue  # already clustered
            in_.add(L)
            for _N in L.N_:
                if _N not in iN_ or _N.fin: continue
                if rng==1 or (not _N.root or _N.root.rng==1):  # not rng-banded
                    node_ += [_N]; cent_ += _N.R_; _N.fin = 1
                    for l in _N.rim:
                        if l in in_: continue  # cluster by link+density:
                        if l.rng==rng and val_(l.Et+ett(l), aw=rc) > 0:
                            link_ += [l]
                        elif l.rng > rng: long_ += [l]
                else:  # cluster top-rng roots
                    _n = _N; _R = rroot(_n)
                    if _R and not _R.fin:
                        if rolp(N, link_, R=1) > ave * rc:
                            node_ += [_R]; _R.fin = 1; _N.fin = 1
                            link_ += _R.L_; long_ += _R.hL_; cent_ += _R.R_
    G_, in_ = [], set()
    for n in rN_: n.fin = 0
    for N in rN_:  # form G per remaining rng N
        if not N.exe or N.fin: continue
        node_,cent_,Link_,_link_,long_ = [N],[],[],[],[]
        if rng==1 or (not N.root or N.root.rng==1):  # not rng-banded
            cent_ = N.R_[:]
            for l in N.rim:
                if val_(l.Et+ett(l), aw=rc+1) > 0:
                    if l.rng==rng: _link_ += [l]
                    elif l.rng>rng: long_ += [l]
        else:  # N is rng-banded, cluster top-rng roots
            n = N; R = rroot(n)
            if R and not R.fin: node_,_link_,long_,cent_ = [R], R.L_[:], R.hL_[:], R.R_[:]; R.fin = 1
        N.fin = 1; link_ = []
        while _link_:
            Link_ += _link_
            extend_Gt(_link_, node_, cent_, link_, long_, in_)
            if link_: _link_ = list(set(link_)); link_ = []  # extended rim
            else:     break
        if node_:
            N_, L_, long_ = list(set(node_)), list(set(Link_)), list(set(long_))
            Et, olp = np.zeros(3), 0
            for n in node_: olp += n.rc  # from Ns, vs. Et from Ls?
            for l in Link_: Et += l.Et
            if val_(Et, 1, (len(node_)-1)*Lw, rc+olp, root.Et) > 0:
                G_ += [sum2graph(root, N_,L_, long_, set(cent_), Et, olp, rng)]
            elif n.fi:  # L_ is preserved anyway
                G_ += node_
    if G_: return sum_N_(G_, root)  # nG

def cluster_n(root, C_, rc, rng=1):  # simplified flood-fill for C_, etc

    def extend_G(_link_, node_,cent_,link_,in_):
        for L in _link_:  # spliced rim
            if L in in_: continue  # already clustered
            in_.add(L)
            for _N in L.N_:
                if not _N.fin and _N in C_:
                    node_ += [_N]; cent_ += _N.R_; _N.fin = 1
                    for l in _N.rim:
                        if l in in_: continue  # cluster by link+density:
                        if val_(l.Et + ett(l), aw=rc) > 0: link_ += [l]
    G_, in_ = [],set()
    for C in C_: C.fin = 0
    for C in C_:  # form G per remaining C
        node_,cent_,Link_,link_ = [C],C.C_[:],[],[]
        _link_ = [l for l in C.rim if val_(l.Et+ett(l), aw=rc) > 0]
        while _link_:
            extend_G(_link_, node_, cent_, link_, in_)  # _link_: select rims to extend G
            if link_: Link_ += _link_; _link_ = list(set(link_)); link_ = []
            else:     break
        if node_:
            N_= list(set(node_)); L_= list(set(Link_))
            Et, olp = np.zeros(3), 0
            for n in N_: olp += n.rc  # from Ns, vs. Et from Ls?
            for l in L_: Et += l.Et
            if val_(Et,1, (len(N_)-1)*Lw, rc+olp, root.Et) > 0:
                G_ += [sum2graph(root, N_,L_,[], set(cent_), Et,olp, rng)]
            else:
                G_ += N_
    if G_: return sum_N_(G_,root)  # nG

def cluster_C(E_, root, rc):  # form centroids by clustering exemplar surround via rims of new member nodes, within root

    _C_, _N_, cG = [],[],[]
    for E in E_:
        C = cent_attr( Copy_(E,root, init=2), rc); C.N_ = [E]   # all rims are within root
        C._N_ = [n for l in E.rim for n in l.N_ if n is not E]  # core members + surround -> comp_C
        _N_ += C._N_; _C_ += [C]
    # reset:
    for n in set(root.N_+_N_ ): n.R_,n.mo_, n._C_,n._mo_ = [],[], [],[]  # aligned pairs, in cross_comp root
    # reform C_, refine C.N_s
    while True:
        C_, cnt,mat,olp, Dm,Do = [],0,0,0,0,0; Ave = ave * (rc+loopw)
        _Ct_ = [[c, c.Et[0]/c.Et[2] if c.Et[0] !=0 else eps, c.rc] for c in _C_]
        for _C,_m,_o in sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True):
            if _m > Ave * _o:
                C = cent_attr( sum_N_(_C.N_, root=root, fC=1), rc); C.R_ = []  # C update lags behind N_; non-local C.rc += N.mo_ os?
                _N_,_N__, mo_, M,O, dm,do = [],[],[],0,0,0,0  # per C
                for n in _C._N_:  # core+ surround
                    if C in n.R_: continue
                    m = min_comp(C,n)[1][0]  # val,olp / C:
                    o = np.sum([mo[0] /m for mo in n._mo_ if mo[0]>m])  # overlap = higher-C inclusion vals / current C val
                    cnt += 1  # count comps per loop
                    if m > Ave * o:
                        _N_+=[n]; M+=m; O+=o; mo_ += [np.array([m,o])]  # n.o for convergence eval
                        _N__ += [_n for l in n.rim for _n in l.N_ if _n is not n]  # +|-Ls
                        if _C not in n._C_: dm+=m; do+=o  # not in extended _N__
                    elif _C in n._C_:
                        __m,__o = n._mo_[n._C_.index(_C)]; dm+=__m; do+=__o
                if M > Ave * len(_N_) * O:
                    for n, mo in zip(_N_,mo_): n.mo_+=[mo]; n.R_+=[C]; C.R_+=[n]  # bilateral assign
                    C.M = M; C.N_+= _N_; C._N_= list(set(_N__))  # core, surround elements
                    C_+=[C]; mat+=M; olp+=O; Dm+=dm; Do+=do  # new incl or excl
                else:
                    for n in _C._N_:
                        n.exe = n.Et[0]/n.Et[2] > 2 * ave  # refine exe, Et vals are already normalized, Et[2] no longer needed for eval?
                        for i, c in enumerate(n.R_):
                            if c is _C: n.mo_.pop(i); n.R_.pop(i); break  # remove mo mapping to culled _C
            else: break  # the rest is weaker
        if Dm > Ave * cnt * Do:  # dval vs. dolp, overlap increases as Cs may expand in each loop?
            _C_ = C_
            for n in root.N_: n._C_=n.R_; n._mo_=n.mo_; n.R_,n.mo_ = [],[]  # new n.R_s, combine with vo_ in Ct_?
        else:  # converged
            break
    if C_:
        for n in [N for C in C_ for N in C.N_]:
            # exemplar V increased by summed n match to root C * rel C Match:
            n.exe = n.et[1-n.fi]+ np.sum([n.mo_[i][0] * (C.M / (ave*n.mo_[i][0])) for i, C in enumerate(n.R_)]) > ave
        if mat > ave:
            cG = cross_comp(sum_N_(C_), rc, fC=1)
            # Cs may be distant, different attrs
        root.C_ = [cG.N_ if cG else C_, mat]

def cent_attr(C, rc):  # weight attr matches | diffs by their match to the sum, recompute to convergence

    wTT = []  # Cs can be fuzzy only to the extent that their correlation weights are different?
    t_ = C.derTT[0] + np.abs(C.derTT[1])  # m_* align, d_* 2-align for comp only?

    for fd, derT, wT in zip((0,1), C.derTT, wTTf):
        if fd: derT = np.abs(derT)  # ds
        _w_ = np.ones(9)  # weigh by feedback:
        val_ = derT / t_ * wT  # signed ms, abs ds
        V = np.sum(val_)
        while True:
            mean = max(V / max(np.sum(_w_), eps), eps)
            inverse_dev_ = np.minimum(val_/mean, mean/val_)  # rational deviation from mean rm in range 0:1, if m=mean
            w_ = inverse_dev_ / .5  # 2/ m=mean, 0/ inf max/min, 1 / mid_rng | ave_dev?
            w_ *= 9 / np.sum(w_)  # mean w = 1, M shouldn't change?
            if np.sum(np.abs(w_-_w_)) > ave*rc:
                V = np.sum(val_ * w_)
                _w_ = w_
            else: break  # weight convergence
        wTT += [_w_]
    C.wTT = np.array(wTT)  # replace wTTf
    return C

def ett(L): return (L.N_[0].et + L.N_[1].et) * intw

def sum2graph(root, node_,link_,long_,cent_, Et, olp, rng):  # sum node,link attrs in graph, aggH in agg+ or player in sub+

    n0 = Copy_(node_[0]); derH = n0.derH; fi = n0.fi
    graph = CN(root=root, fi=1,rng=rng, N_=node_,L_=link_,cent_=cent_, Et=Et,rc=olp, box=n0.box, baseT=n0.baseT, derTT=n0.derTT)
    graph.hL_ = long_
    n0.root = graph; yx_ = [n0.yx]; fg = fi and isinstance(n0.N_[0],CN)   # not PPs
    Nt = Copy_(n0); DerH = [] # CN, add_N(Nt,Nt.Lt)?
    for N in node_:
        add_H(derH,N.derH,graph); graph.baseT+=N.baseT; graph.derTT+=N.derTT; graph.box=extend_box(graph.box,N.box); yx_+=[N.yx]; N.root = graph
        if fg: add_N(Nt,N)  # froot = 0
    for L in link_:
        add_H(DerH,L.derH,graph); graph.baseT+=L.baseT; graph.derTT+=L.derTT
    if DerH: add_H(derH,DerH, graph)  # * rn?
    graph.derH = derH
    if fg: graph.nH = Nt.nH + [Nt]  # pack prior top level
    yx = np.mean(yx_, axis=0); dy_,dx_ = (yx_-yx).T; dist_ = np.hypot(dy_,dx_)
    graph.span = dist_.mean()  # node centers distance to graph center
    graph.angl = np.sum([l.angl for l in link_], axis=0)
    if fi and len(link_) > 1:  # else default mang = 1
        graph.mang = np.sum([comp_A(graph.angl, l.angl)[0] for l in link_]) / len(link_)
    graph.yx=yx
    return graph

def slope(link_):  # get ave 2nd rate of change with distance in cluster or frame?

    Link_ = sorted(link_, key=lambda x: x.span)
    dists = np.array([l.span for l in Link_])
    diffs = np.array([l.Et[1]/l.Et[2] for l in Link_])
    rates = diffs / dists
    # ave d(d_rate) / d(unit_distance):
    return (np.diff(rates) / np.diff(dists)).mean()

def comp_H(H,h, rn, ET=None, DerTT=None, root=None):  # one-fork derH

    derH, derTT, Et = [], np.zeros((2,9)), np.zeros(3)
    for _lay, lay in zip_longest(H,h):  # selective
        if _lay and lay:
            dlay = _lay.comp_lay(lay, rn, root=root)
            derH += [dlay]; derTT = dlay.derTT; Et[:2] += dlay.Et
    if Et[2]: DerTT += derTT; ET += Et
    return derH

def add_H(H, h, root=0, rn=1, rev=0):  # layer-wise add|append derH

    for Lay, lay in zip_longest(H,h):  # different len if lay-selective comp
        if lay:
            if Lay: Lay.add_lay(lay, rn)
            else:   H += [lay.copy_(rev)]  # * rn?
            if root: root.derTT += lay.derTT*rn; root.Et[:2] += lay.Et*rn
    return H

def add_sett(Sett,sett):
    if Sett: N_,Et = Sett; n_ = sett[0]; N_.update(n_); Et += np.sum([t.Et for t in n_-N_])
    else:    Sett += [copy(par) for par in sett]  # altg_, Et

def sum_N_(node_, root=None, fC=0):  # form cluster G

    G = Copy_(node_[0], root, init = 0 if fC else 1)
    if fC:
        G.N_= [node_[0]]; G.L_ = [] if G.fi else len(node_); G.rim = []
    for n in node_[1:]:
        add_N(G,n,0, fC, froot=1)
    G.rc /= len(node_)
    if not fC and G.fi:
        for L in G.L_: G.Et += L.Et  # avoid redundant Ls in rims
    return G  # no rim

def add_N(N,n, fmerge=0, fC=0, froot=0):

    if fmerge:  # different in altg_?
        for node in n.N_: node.root=N; N.N_ += [node]
        N.L_ += n.L_; N.rim += n.rim  # L is list or len, no L.root assign, rims can't overlap
    else:
        N.N_ += [n]
        if not fC: N.L_ += [l for l in n.rim if l.Et[0]>ave] if n.fi else n.L_  # len
    if froot:
        n.fin = 1; n.root = N
    if n.altg_: add_sett(N.altg_,n.altg_)  # ext clusters
    if n.C_: add_sett(N.C_,n.C_)  # int clusters
    if n.nH: add_NH(N.nH,n.nH, root=N)
    if n.lH: add_NH(N.lH,n.lH, root=N)
    en,_en = n.Et[2],N.Et[2]; rn = en/_en
    for Par,par in zip((N.angl,N.mang,N.baseT,N.derTT,N.Et), (n.angl,n.mang,n.baseT,n.derTT,n.Et)):
        Par += par  # *rn for comp
    if n.derH: add_H(N.derH,n.derH, N, rn)
    if fC: N.rc += np.sum([mo[1] for mo in n._mo_])
    else:  N.rc = (N.rc*_en + n.rc*en) / (en+_en)  # ave of aves
    N.yx = (N.yx*_en + n.yx*en) /(en+_en)
    N.span = max(N.span,n.span)
    N.box = extend_box(N.box, n.box)
    if hasattr(n,'mo_') and hasattr(N,'mo_'):
        N.R_ += n.R_; N.mo_ += n.mo_  # not sure
    return N

def add_NH(H, h, root, rn=1):

    for Lev, lev in zip_longest(H, h, fillvalue=None):  # always aligned?
        if lev:
            if Lev: add_N(Lev,lev)  # lev.root shouldn't be updated to Lev, so froot = 0
            else:   H += [Copy_(lev, root)]

def extend_box(_box, box):
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return min(y0,_y0), min(x0,_x0), max(yn,_yn), max(xn,_xn)

def PP2N(PP, frame):

    P_, link_, verT, latT, A, S, box, yx, Et = PP
    baseT = np.array(latT[:4])
    [mM,mD,mI,mG,mA,mL], [dM,dD,dI,dG,dA,dL] = verT  # re-pack in derTT:
    derTT = np.array([
        np.array([mM,mD,mL, mI,mG,mA, mL,mL/2, eps]), # extA=eps
        np.array([dM,dD,dL, dI,dG,dA, dL,dL/2, eps]) ])
    derH = [CLay(node_=P_,link_=link_, derTT=deepcopy(derTT), Et=np.array([sum(derTT[0]), sum(np.abs(derTT[1]))]) )]
    y,x,Y,X = box; dy,dx = Y-y,X-x

    return CN(root=frame, fi=1, Et=Et, N_=P_, L_=link_, baseT=baseT, derTT=derTT, derH=derH, box=box, yx=yx, angl=A, span=np.hypot(dy/2,dx/2))

# not used, make H a centroid of layers, same for nH?
def sort_H(H, fi):  # re-assign olp and form priority indices for comp_tree, if selective and aligned

    i_ = []  # priority indices
    for i, lay in enumerate(sorted(H.node_, key=lambda lay: lay.Et[fi], reverse=True)):
        di = lay.i - i  # lay index in H
        lay.rc += di  # derR - valR
        i_ += [lay.i]
    H.i_ = i_  # H priority indices: node/m | link/d
    if fi:
        H.root.node_ = H.node_

def eval(V, weights):  # conditional progressive eval, with default ave in weights[0]
    W = 1
    for w in weights:
        W *= w
        if V < W: return 0
    return 1

def val_H(H):
    derTT = np.zeros((2,9)); Et = np.zeros(3)
    for lay in H:
        for fork in lay:
            if fork: derTT += fork.derTT; Et += fork.Et
    return derTT, Et

def prj_TT(_Lay, proj, dec):
    Lay = _Lay.copy_(); Lay.derTT[1] *= proj * dec  # only ds are projected?
    return Lay

def prj_dH(_H, proj, dec):
    H = []
    for lay in _H:
        H += [[lay[0].copy_(), prj_TT(lay[1],proj,dec) if lay[1] else []]]  # two-fork
    return H

def comp_prj_dH(_N,N, ddH, rn, link, angl, span, dec):  # comp combined int proj to actual ddH, as in project_N_

    # include prj derTT?
    _cos_da= angl.dot(_N.angl) / (span *_N.span)  # .dot for scalar cos_da
    cos_da = angl.dot(N.angl) / (span * N.span)
    _rdist = span/_N.span
    rdist  = span/ N.span
    prj_DH = add_H( prj_dH(_N.derH[1:], _cos_da *_rdist, _rdist*dec),  # derH[0] is positionals
                    prj_dH( N.derH[1:], cos_da * rdist, rdist*dec),
                    link)  # comb proj dHs | comp dH ) comb ddHs?
    # Et+= confirm:
    dddH = comp_H(prj_DH, ddH[1:], rn, link.Et, link.derTT, link)  # ddH[1:] maps to N.derH[1:]
    add_H(ddH, dddH, link, rn)

def project_N_(Fg, yx):

    dy,dx = Fg.yx - yx
    Fdist = np.hypot(dy,dx)  # external dist
    rdist = Fdist / Fg.span
    Angle = np.array([dy,dx]) # external angle
    angle = np.sum([L.angl for L in Fg.L_], axis=0)
    cos_d = angle.dot(Angle) / (np.hypot(*angle) * Fdist)
    # difference between external and internal angles, *= rdist
    ET = np.zeros(3); DerTT = np.zeros((2,9))
    N_ = []
    for _N in Fg.N_:  # sum _N-specific projections for cross_comp
        if len(_N.derH) < 2: continue
        M,D,n = _N.Et
        dec = rdist * (M/(M+D))  # match decay rate, * ddecay for ds?
        prj_H = prj_dH(_N.derH[1:], cos_d * rdist, dec)  # derH[0] is positionals
        prjTT, pEt = val_H(prj_H)  # sum only ds here?
        pD = pEt[1]*dec; dM = M*dec
        pM = dM - pD * (dM/(ave*n))  # -= borrow, regardless of surprise?
        pEt = np.array([pM, pD, n])
        if val_(pEt, aw=contw):
            ET+=pEt; DerTT+=prjTT
            N_ += [CN(N_=_N.N_, Et=pEt, derTT=prjTT, derH=prj_H, root=CN())]  # same target position?
    # proj Fg:
    if val_(ET, mw=len(N_)*Lw, aw=contw):
        return CN(N_=N_,L_=Fg.L_,Et=ET, derTT=DerTT)  # proj Fg, add Prj_H?

def ffeedback(root):  # adjust filters: all aves *= rV, ultimately differential backprop per ave?

    wTTf = np.ones((2,9))  # sum derTT coefs: m_,d_ [M,D,n, I,G,A, L,S,eA] / Et, baseT, dimension
    rM, rD, rVd = 1, 1, 0
    hLt = sum_N_(root.L_)  # links between top nodes
    _derTT = np.sum([l.derTT for l in hLt.N_])  # _derTT[np.where(derTT==0)] = eps
    for lev in reversed(root.nH):  # top-down
        if not lev.lH: continue
        Lt = lev.lH[-1]  # dfork
        _m, _d, _n = hLt.Et; m, d, n = Lt.Et
        rM += (_m / _n) / (m / n)  # relative higher val, | simple relative val?
        rD += (_d / _n) / (d / n)
        derTT = np.sum([l.derTT for l in Lt.N_])  # top link_ is all comp results
        wTTf += np.abs((_derTT / _n) / (derTT / n))
        if Lt.lH:  # ddfork only, not recursive?
            # intra-level recursion in dfork
            rVd, wTTfd = ffeedback(Lt)
            wTTf = wTTf + wTTfd

    return rM+rD+rVd, wTTf

def project_focus(PV__, y,x, Fg):  # radial accum of projected focus value in PV__

    m,d,n = Fg.Et
    V = (m-ave*n) + (d-avd*n)
    dy,dx = Fg.angl; a = dy/ max(dx,eps)  # average link_ orientation, projection
    decay = (ave / (Fg.baseT[0]/n)) * (wYX / adist)  # base decay = ave_match / ave_template * rel dist (ave_dist is a placeholder)
    H, W = PV__.shape  # = win__
    n = 1  # radial distance
    while y-n>=0 and x-n>=0 and y+n<H and x+n<W:  # rim is within frame
        dec = decay * n
        pV__ = np.array([
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

def agg_frame(foc, image, iY, iX, rV=1, wTTf=[], fproj=0):  # search foci within image, additionally nested if floc

    if foc: dert__ = image  # focal img was converted to dert__
    else:
        dert__ = comp_pixel(image) # global
        global ave, Lw, intw, loopw, centw, contw, adist, amed, medw, mW, dW
        ave, Lw, intw, loopw, centw, contw, adist, amed, medw = np.array([ave, Lw, intw, loopw, centw, contw, adist, amed, medw]) / rV
        # fb rws: ~rvs
    nY,nX = dert__.shape[-2] // iY, dert__.shape[-1] // iX  # n complete blocks
    Y, X  = nY * iY, nX * iX  # sub-frame dims
    frame = CN(box=np.array([0,0,Y,X]), yx=np.array([Y//2,X//2]))
    dert__= dert__[:,:Y,:X]  # drop partial rows/cols
    win__ = dert__.reshape(dert__.shape[0], iY,iX, nY,nX).swapaxes(1,2)  # dert=5, wY=64, wX=64, nY=13, nX=20
    PV__  = win__[3].sum( axis=(0,1)) * intw  # init prj_V = sum G, shape: nY=13, nX=20
    aw = contw * 20
    while np.max(PV__) > ave * aw:  # max G * int_w + pV, add prj_V foci, only if not foc?
        # max win index:
        y,x = np.unravel_index(PV__.argmax(), PV__.shape)
        PV__[y,x] = -np.inf  # to skip?
        if foc:
            Fg = frame_blobs_root( win__[:,:,:,y,x], rV)  # [dert, iY, iX, nY, nX]
            vect_root(Fg, rV,wTTf); Fg.L_=[]  # focal dert__ clustering
            cross_comp(Fg, rc=frame.rc)
        else:
            Fg = agg_frame(1, win__[:,:,:,y,x], wY,wX, rV=1, wTTf=[])  # use global wY,wX in nested call
            if Fg and Fg.L_:  # only after cross_comp(PP_)
                rV, wTTf = ffeedback(Fg)  # adjust filters
                Fg = cent_attr(Fg,2)  # compute Fg.wTT: correlation weights in frame derTT?
                wTTf *= Fg.wTT; mW = np.sum(wTTf[0]); dW = np.sum(wTTf[1])  # global?
                wTTf[0] *= 9/mW; wTTf[1] *= 9/dW  # Fg.wTT is redundant?
                # re-norm weights
        if Fg and Fg.L_:
            if fproj and val_(Fg.Et,1, (len(Fg.N_)-1)*Lw, Fg.rc+loopw*20):
                pFg = project_N_(Fg, np.array([y,x]))
                if pFg:
                    cross_comp(pFg, rc=Fg.rc)
                    if val_(pFg.Et,1, (len(pFg.N_)-1)*Lw, pFg.rc+contw*20):
                        project_focus(PV__, y,x, Fg)  # += proj val in PV__
            # no target proj
            frame = add_N(frame, Fg, fmerge=1, froot=1) if frame else Copy_(Fg)
            aw = contw *20 * frame.Et[2] * frame.rc

    if frame.N_ and val_(frame.Et, (len(frame.N_)-1)*Lw, frame.rc+loopw+20) > 0:
        # recursive xcomp
        Fn = cross_comp(frame, rc=frame.rc+loopw)
        if Fn: frame.N_=Fn.N_; frame.Et+=Fn.Et  # other frame attrs remain
        # spliced foci cent_:
        if frame.C_ and val_(frame.C_[1], (len(frame.N_)-1)*Lw, frame.rc+centw) > 0:
            Fc = cross_comp(frame, rc=frame.rc+loopw, fC=1)
            if Fc: frame.C_=Fn.N_; frame.Et+=Fn.Et
        if not foc:
            return frame  # foci are unpacked
'''
frame expansion per level: cross_comp lower-window N_,C_, forward results to next lev, project feedback to scan new lower windows
'''
def frame_H(image, iY,iX, Ly,Lx, Y,X, rV, max_elev=4, wTTf=np.ones(9,dtype="float")):  # all initial args are set manually

    def base_win(y,x):  # 1st level, higher levels get Fg s

        Fg = frame_blobs_root( comp_pixel( image[y:y+Ly, x:x+Lx]), rV)
        vect_root(Fg, rV, wTTf); Fg.L_=[]  # clustering
        return cross_comp(Fg, rc=frame.rc)

    def cross_comp_(win, rc):  # top-composition xcomp, add margin search extend and splice?

        n_,l_,c_ = [],[],[]
        for g in win.flat:
            if g: n_ += g.N_; l_ += g.L_; c_ += g.C_

        nG, lG, cG = cross_comp(CN(N_=n_),rc), cross_comp(CN(N_=l_),rc+1), cross_comp(CN(N_=c_),rc+2,fC=1)
        Et = np.sum([ g.Et for g in (nG,lG,cG) if g])
        rc = np.mean([g.rc for g in (nG,lG,cG) if g])

        return CN(Et=Et, rc=rc, N_= nG.N_ if nG else [], L_= lG.N_ if lG else [], C_= cG.N_ if cG else [])

    def expand_lev(iy,ix, elev, win):  # N_,C_,L_ window vs. frame in higher levels

        PV__ = np.zeros([Ly,Lx])  # maps to level window
        Win = np.empty((Ly,Lx), dtype=object)  # output G__
        y,x = iy // Ly ** elev, ix // Lx ** elev  # init Win tile coords
        Fg_ = []
        while True:
            if elev: Fg = cross_comp_(win, rc=elev)  # cross_comp per N_,C_,L_ in the window?
            else:    Fg = base_win(iy,ix)  # level 0
            if Fg and val_(Fg.Et,1, (len(Fg.N_)-1)*Lw, Fg.rc+loopw+elev) > 0:
                PV__[y,x] = -np.inf  # skip
                Win[y,x] = Fg; Fg_ += [Fg]
                pFg = project_N_(Fg, np.array([y,x]))  # extend lev by feedback in Win
                if pFg and val_(pFg.Et,1,(len(pFg.N_)-1)*Lw, pFg.rc+elev) > 0:
                    project_focus(PV__,y,x, Fg)  # add proj vals into PV__
                    y, x = np.unravel_index(PV__.argmax(),PV__.shape)
                    if PV__[y,x] > ave:
                        iy = y* Ly**elev; ix = x* Lx**elev  # new win by feedback to image, scale y,x with elevation
                        win = frame_H(image, iy,ix, Ly,Lx, Y,X, rV, elev)  # up to current level
                    else: break
                else: break
            else: break
        if Fg_ and val_(np.sum([g.Et for g in Fg_]),1,(len(Fg_)-1)*Lw, np.mean([g.rc for g in Fg_])+elev) > 0:
            return Win
        else: return []

    frame = CN(box=np.array([0,0,Y,X]), yx=np.array([Y//2,X//2])); frame.H=[]
    elev = 0; _win = []
    while True and elev < max_elev:
        win = expand_lev(iY,iX, elev, _win)  # fixed focal point
        if win:
            frame.H += [win]; _win=win; elev+=1  # feedforward extends H
        else: break
    return _win
''' 
    global ave, Lw, intw, loopw, centw, contw, adist, amed, medw, mW, dW
    ave, Lw, intw, loopw, centw, contw, adist, amed, medw = np.array([ave, Lw, intw, loopw, centw, contw, adist, amed, medw]) / rV
    rV, wTTf = ffeedback(Fg)  # adjust filters
    Fg = cent_attr(Fg,2)  # compute Fg.wTT: correlation weights in frame derTT?
    wTTf *= Fg.wTT; mW = np.sum(wTTf[0]); dW = np.sum(wTTf[1])  # global?
    wTTf[0] *= 9/mW; wTTf[1] *= 9/dW  # Fg.wTT is redundant?
'''

if __name__ == "__main__":  # './images/toucan_small.jpg' './images/raccoon_eye.jpeg', add larger global image

    Y,X = imread('./images/toucan_small.jpg').shape
    # frame = agg_frame(0, image=imread('./images/toucan.jpg'), iY=Y, iX=X)
    # slicing must use int instead of float
    frame = frame_H(image=imread('./images/toucan.jpg'), iY=int(Y/2), iX=int(X/2), Ly=64,Lx=64, Y=Y, X=X, rV=1)
    # search frames ( foci inside image