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
class CLay(CBase):  # layer of derivation hierarchy, subset of CG
    name = "lay"
    def __init__(l, **kwargs):
        super().__init__()
        l.Et = kwargs.get('Et', np.zeros(4))
        l.olp = kwargs.get('olp', 1)  # ave nodet overlap
        l.node_ = kwargs.get('node_', [])  # concat across fork tree
        l.link_ = kwargs.get('link_', [])
        l.derTT = kwargs.get('derTT', np.zeros((3,10)))  # sum m_,d_ [M,D,n,o, I,G,A, L,S,eA] across fork tree
        # i: lay index in root node_,link_, to revise olp; i_: m,d priority indices in comp node|link H
        # ni: exemplar in node_, trace in both directions?
    def __bool__(l): return bool(l.node_)

    def copy_(lay, rev=0, i=None):  # comp direction may be reversed to -1

        if i:  # reuse self
            C = lay; lay = i; C.node_=copy(i.node_); C.link_ = copy(i.link_); C.derTT=np.zeros((3,10))
        else:  # init new C
            C = CLay(node_=copy(lay.node_), link_=copy(lay.link_))
        C.Et = copy(lay.Et)
        for fd, tt in enumerate(lay.derTT):  # nested array tuples
            C.derTT[fd] += tt * -1 if (rev and fd) else deepcopy(tt)

        if not i: return C

    def add_lay(Lay, lay, rev=0, rn=1):  # merge lays, including mlay + dlay

        # rev = dir==-1, to sum/subtract numericals in m_,d_
        for fd, Fork_, fork_ in zip((0,1), Lay.derTT, lay.derTT):
            Fork_ += (fork_ * -1 if (rev and fd) else fork_) * rn  # m_| d_
        # concat node_,link_:
        Lay.node_ += [n for n in lay.node_ if n not in Lay.node_]
        Lay.link_ += lay.link_
        Lay.Et += lay.Et * rn
        Lay.olp = (Lay.olp + lay.olp * rn) /2
        return Lay

    def comp_lay(_lay, lay, rn, root, dir=1):  # unpack derH trees down to numericals and compare them

        derTT, Et = comp_derT(_lay.derTT[1], lay.derTT[1] * rn * dir)  # is dir still used?
        if root: root.Et += Et
        node_ = list(set(_lay.node_+ lay.node_))  # concat, or redundant to nodet?
        link_ = _lay.link_ + lay.link_
        return CLay(Et=Et, olp=(_lay.olp+lay.olp*rn)/2, node_=node_, link_=link_, derTT=derTT)

def comp_derT(_dT, dT):  # all normalized diffs

    _a_, a_ = np.abs(_dT), np.abs(dT)
    d_ = _dT - dT  # signed
    m_ = np.minimum(_a_, a_)
    m_ *= np.where((_dT < 0) != (dT < 0), -1, 1)  # match is negative if comparands have opposite sign
    t_ = np.maximum.reduce([_a_, a_, np.zeros(10) + 1e-7])
    # or max signed ds?
    return (np.array([m_, d_, t_]),  # derTT
            np.array([(m_/t_ +1)/2 @ wTTf[0], (d_/t_ +1)/2 @ wTTf[1], 10, t_ @ wTTf[0]]))  # Et: M, D, n=10, T

class CN(CBase):
    name = "node"
    def __init__(n, **kwargs):
        super().__init__()
        n.fi = kwargs.get('fi', 1)  # if G else 0, fd_: list of forks forming G?
        n.N_ = kwargs.get('N_',[])  # nodes, or ders in links
        n.L_ = kwargs.get('L_',[])  # links if fi else len nodet.N_s?
        n.nH = kwargs.get('nH',[])  # top-down hierarchy of sub-node_s: CN(sum_N_(Nt_))/ lev, with single added-layer derH, empty nH
        n.lH = kwargs.get('lH',[])  # bottom-up hierarchy of L_ graphs: CN(sum_N_(Lt_))/ lev, within each nH lev
        n.Et = kwargs.get('Et',np.zeros(4))  # sum from L_, cent_?
        n.et = kwargs.get('et',np.zeros(4))  # sum from rim, altg_?
        n.olp = kwargs.get('olp',1)  # overlap to ext Gs, ave in links? separate olp for rim, or internally overlapping?
        n.rim = kwargs.get('rim',[])  # node-external links, rng-nested? set?
        n.derH  = kwargs.get('derH',[])  # sum from L_ or rims
        n.derTT = kwargs.get('derTT',np.zeros((3,10)))  # sum derH -> m_,d_ [M,D,n,o, I,G,A, L,S,extA]
        n.baseT = kwargs.get('baseT',np.zeros(4))  # I,G,A not ders
        n.yx    = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.rng   = kwargs.get('rng',1)  # or med: loop count in comp_node_|link_
        n.box   = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.span  = kwargs.get('span',0) # distance in nodet or aRad, comp with baseT and len(N_) but not additive?
        n.angl  = kwargs.get('angl',np.zeros(2))  # dy,dx, sum from L_
        n.mang  = kwargs.get('mang',1)  # ave match of angles in L_, = identity in links
        n.root  = kwargs.get('root', [])  # immediate only
        n.cent_ = kwargs.get('cent_',[])  # int centroid Gs, replace/combine N_?
        n.altg_ = kwargs.get('altg_',[])  # ext contour Gs, replace/combine rim?
        n.fin = kwargs.get('fin',0)  # clustered, temporary
        n.exe = kwargs.get('exe',1)  # exemplar, temporary
        n.seen_ = set()
        # n.fork_tree: list =z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def __bool__(n): return bool(n.N_)

def Copy_(N, root=None, init=0):

    C = CN(root=root)
    if init:  # init G|C with N
        C.N_ = [N]; C.nH, C.lH, N.root = [],[],C
        if init==1: N.L_ += N.rim  # empty in centroid
    else:
        C.N_,C.nH,C.lH, N.root = (list(N.N_),list(N.nH),list(N.lH), root if root else N.root)
        C.L_ = list(N.L_) if N.fi else N.L_
    C.derH  = [lay.copy_() for lay in N.derH]
    C.derTT = deepcopy(N.derTT)
    for attr in ['Et', 'baseT','yx','box','angl','rim','altg_','cent_']: setattr(C, attr, copy(getattr(N, attr)))
    for attr in ['olp','rng', 'fi', 'fin', 'span', 'mang']: setattr(C, attr, getattr(N, attr))
    return C

ave, avd, arn, aI, aS, aveB, aveR, Lw, intw, loopw, centw, contw = 10, 10, 1.2, 100, 5, 100, 3, 5, 2, 5, 10, 15  # value filters + weights
adist, amed, distw, medw = 10, 3, 2, 2  # cost filters + weights, add alen?
wM, wD, wN, wO, wG, wL, wI, wS, wa, wA = 10, 10, 20, 10, 20, 5, 20, 2, 1, 1  # der params higher-scope weights = reversed relative estimated ave?
mW = dW = 10; wTTf = np.ones((2,10))  # fb weights per derTT, adjust in agg+
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
            if edge.G * ((len(edge.P_) - 1) * Lw) > ave * sum([P.latuple[4] for P in edge.P_]):
                N_ += comp_slice(edge, rV, wTTf)
    Fg.N_ = [PP2N(PP, Fg) for PP in N_]

def val_(Et, fi=1, mw=1, aw=1, _Et=np.zeros(4)):  # m,d eval per cluster or cross_comp

    if mw <= 0: return 0
    am = ave * aw  # includes olp, M /= max I | M+D? div comp / mag disparity vs. span norm
    ad = avd * aw  # dval is borrowed from co-projected or higher-scope mval
    m,d,n,_ = Et
    if fi==2: val = np.array([m-am, d-ad]) * mw
    else:     val = (m-am if fi else d-ad) * mw  # m: m/ (m+d), d: d/ (m+d)?
    if _Et[2]:
        # borrow rational deviation of contour if fi else root Et: multiple? not circular
        _m,_d,_n,_ = _Et; _mw = mw*(_n/n)
        if fi==2: val *= np.array([_d/ad, _m/am]) * _mw
        else:     val *= (_d/ad if fi else _m/am) * _mw
    return val

''' Core process per agg level, as described in top docstring:
Cross-compare nodes, evaluate incremental-derivation cross-comp of new >ave difference links, recursively.
 
Select exemplar nodes, connectivity-cluster by >ave match links, correlation-cluster links by >ave difference.
Form complemented (core+contour) clusters for recursive higher-composition cross_comp, reorder by eigenvalues. 

Centroid-cluster nodes, selectively spreading from exemplar seeds via their rim, across connectivity clusters.
Feedback coords to bottom level or prior-level in parallel pipelines, filter updates in more coarse cycles '''

def cross_comp(root, rc):  # rng+ and der+ cross-comp and clustering

    N_,L_,Et = comp_(root.N_, rc)  # rc: redundancy+olp, lG.N_ is Ls
    if len(L_) > 1:
        mV,dV = val_(Et,2, (len(L_)-1)*Lw, rc+loopw); lG = []
        if dV > 0:
            if root.L_: root.lH += [sum_N_(root.L_)]  # replace L_ with agg+ L_:
            root.L_ =L_; root.Et += Et
            if dV > avd:
                lG = cross_comp(CN(N_=L_), rc+contw)  # link clustering, +2 der layers
                if lG: root.lH += [lG]+lG.nH; root.Et+=lG.Et; root.derH+=lG.derH  # new lays
        if mV > 0:
            nG = Cluster(root, N_, rc+loopw)  # get_exemplars, rng-band, local cluster_C_
            if nG:
                if val_(np.sum([g.Et for g in nG.cent_]),1,(len(nG.N_)-1)*Lw, rc+centw+nG.rng, Et) > 0:
                    cluster_C_(nG, rc)  # cluster nG.cent_ exemplars, of any rng
                if lG: comb_altg_(nG,lG, rc+2)  # both fork alt Ns are clustered
                if val_(nG.Et,1, (len(nG.N_)-1)*Lw, rc+loopw+nG.rng, Et) > 0:
                    nG = cross_comp(nG, rc+loopw) or nG  # agg+
                _H = root.nH; root.nH = []
                nG.nH = _H + [root] + nG.nH  # pack root in Nt.nH, has own L_,lH
                return nG  # recursive feedback

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
        Et, Rdn, altl_ = np.zeros(4), 0, []  # contour/core clustering
        LR_ = {R(L) for n in Ng.N_ for L in n.rim}  # lGs, individual rims are too weak
        for LR in LR_:
            if LR and LR.altg_:  # not None, eval Lg.altg_[1]?
                for core, rdn in LR.altg_[0]:  # map contour rdns to core N:
                    if core is Ng:
                        altl_ += [LR]; Et += core.Et; Rdn += rdn  # add to Et[2]?
        if altl_:
            aG = CN(N_=altl_, Et=Et)
            if val_(Et,0,(len(altl_)-1)*Lw, rc+Rdn+loopw) > 0:  # norm by core_ rdn
                aG = cross_comp(aG, rc) or aG
            Ng.altg_ = (aG.N_,aG.Et)

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
            V+= proj_L_(node.rim) if v*mW > av*node.olp else v  # too low for indiv L proj
        v = node.Et[1-fi]  # internal, lower weight
        V+= proj_L_(node.L_,intw) if (fi and v*mW > av*node.olp) else v  # empty link L_
    if fC:
        V += np.sum([i[0] for i in _N.mo_]) + np.sum([i[0] for i in N.mo_])
        # project matches to centroids?
    return V

def comp_(iN_, rc):  # comp pairs of nodes or links within max_dist

    N__,L_, ET = [],[], np.zeros(4); rng,olp_,_N_ = 1,[],copy(iN_)
    while True:  # _vM, rng in rim only?
        N_,Et = [],np.zeros(4)
        for _N, N in combinations(_N_, r=2):  # sort-> proximity-order ders?
            if _N in N.seen_ or len(_N.nH) != len(N.nH):  # | root.nH: comp top nodes only, or comp depth
                continue
            dy_dx = _N.yx-N.yx; dist = np.hypot(*dy_dx); olp = (N.olp+_N.olp) / 2
            if {l for l in _N.rim if l.Et[0]>ave} & {l for l in N.rim if l.Et[0]>ave}:  # +ve rim
                fcomp = 1  # connected by common match, which means prior bilateral proj eval
            else:
                V = proj_V(_N,N, dy_dx, dist)  # eval _N,N cross-induction for comp
                fcomp = adist * V/olp > dist  # min induction
            if fcomp:
                Link = comp_N(_N,N, olp, rc, L_=L_, angl=dy_dx, span=dist, rng=rng)
                if val_(Link.Et, aw=loopw*olp) > 0:
                    Et += Link.Et; olp_ += [olp]  # link.olp is the same with o
                    for n in _N,N:
                        if n not in N_ and val_(n.et, aw=rc+rng-1+loopw+olp) > 0:  # cost+ / rng?
                            N_ += [n]  #-> rng+ eval
        if N_:
            N__ += [N_]; ET += Et
            if val_(Et, mw=(len(N_)-1)*Lw, aw=loopw * (sum(olp_) if olp_ else 1)) > 0:  # current-rng vM
                _N_ = N_; rng += 1; olp_ = []  # reset
            else: break  # low projected rng+ vM
        else: break
    return N__, L_, ET

def comp_N(_N,N, o,rc, L_=None, angl=np.zeros(2), span=None, fdeep=0, rng=1):  # compare links, optional angl,span,dang?

    derTT, Et,rn = min_comp(_N,N, rc)  # -1 if _rev else 1, d = -d if L is reversed relative to _L, obsoleted by sign in angle?
    baseT = (_N.baseT+ N.baseT*rn) /2  # not derived
    yx = np.add(_N.yx,N.yx) /2; _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])  # primary ext attrs
    fi = N.fi
    Link = CN(Et=Et,olp=o, N_=[_N,N], L_=len(_N.N_+N.N_), et=_N.et+N.et, baseT=baseT,derTT=derTT, yx=yx,box=box, span=span,angl=angl, rng=rng, fi=0)
    Link.derH = [CLay(Et=copy(Et), node_=[_N,N],link_=[Link],derTT=copy(derTT), root=Link)]
    if fdeep:
        if val_(Et,1, len(N.derH)-2, o+rc) > 0 or fi==0:  # else derH is dext,vert
            Link.derH += comp_H(_N.derH,N.derH, rn, Et, derTT, Link)  # append
        if fi:
            _N_,N_ = (_N.cent_,N.cent_) if (_N.cent_ and N.cent_) else (_N.N_,N.N_)  # or both? no rim,altg_ in top nodes
            if isinstance(N_[0],CN) and isinstance(_N_[0],CN):  # not PP
                spec(_N_,N_,o,rc,Et, Link.L_)  # use L_ for dspe?
    if fdeep==2: return Link  # or Et?
    if L_ is not None:
        L_ += [Link]
    for n, _n, rev in zip((N,_N),(_N,N),(0,1)):  # if rim-mediated comp: reverse dir in _N.rim: rev^_rev
        n.rim += [Link]; n.et += Et; n.seen_.add(_n)  # or extT += Link?
    return Link

def min_comp(_N,N, rc):  # comp Et, baseT, extT, derTT

    fi = N.fi
    _M,_D,_n,_t =_N.Et; _I,_G,_Dy,_Dx =_N.baseT; _L = len(_N.N_) if fi else _N.L_  # len nodet.N_s
    M, D, n, t  = N.Et;  I, G, Dy, Dx = N.baseT;  L = len(N.N_) if fi else N.L_
    rn = _n / n  # size ratio, add _o/o?
    _pars = np.abs(np.array([_M,_D,_n,_t,_I,_G, np.array([_Dy,_Dx]),_L,_N.span], dtype=object))  # Et, baseT, extT
    pars  = np.abs(np.array([ M, D, n, t, I, G, np.array([Dy,Dx]), L, N.span], dtype=object)) * rn
    pars[4] = [pars[4],aI]; pars[8] = [pars[8],aS]  # no avd*rn: d/=t
    m_,d_,t_ = comp(_pars,pars)
    if (np.any(_N.angl) and np.any(N.angl)) and (_N.mang and N.mang):
        mA,dA = comp_A(_N.angl, N.angl*rn)
        conf = _N.mang * (N.mang/rn)  # fractional
        np.append(m_, mA*conf)  # only affects Et, no rot = 1 if dy * _dx - dx * _dy >= 0 else -1  # +1 CW, −1 CCW, ((1-cos_da)/2) * rot?
        np.append(d_, dA*conf); np.append(t_, mA+abs(dA))
    else:
        m_=np.append(m_,0); d_=np.append(d_,0); t_=np.append(t_,0)
    # 3 x M,D,n,t, I,G,A, L,S,eA:
    (md_,dd_,td_), (M,D,_,T) = comp_derT(_N.derTT[1], N.derTT[1] * rn)  # no dir?
    DerTT = np.array([m_+md_, d_+dd_, t_+td_])
    Et = np.array([
        (m_/t_ +1)/2 @ wTTf[0] +M, # include prior M,D, norm m,d to +ve 0:1 vals
        (d_/t_ +1)/2 @ wTTf[1] +D, min(_n,n), t_@ wTTf[0] +T])  # n: shared scope?
    return DerTT, Et, rn

def comp(_pars, pars):  # raw inputs or derivatives, norm to 0:1 in eval only

    m_,d_,t_ = [],[],[]
    for _p, p in zip(_pars, pars):
        if isinstance(_p, np.ndarray):
            mA, dA = comp_A(_p,p)  # both in -1:1
            m_ += [mA]; d_ += [dA]
            t_ += [1]  # norm already
        elif isinstance(p, list):  # massless I|S avd in p only
            p, avd = p
            d = _p - p; ad = abs(d)
            t_ += [max(avd, ad, 1e-7)]
            m_ += [avd-ad]  # +|-
            d_ += [d]
        else:  # massive
            t_ += [max(_p,p,1e-7)]
            m_ += [(min(_p,p) if _p<0 == p<0 else -min(_p,p))]
            d_ += [_p - p]
    return np.array(m_), np.array(d_), np.array(t_)

def comp_A(_A,A):

        dA = atan2(*_A) - atan2(*A)
        if   dA > pi: dA -= 2 * pi  # rotate CW
        elif dA <-pi: dA += 2 * pi  # rotate CCW

        return cos(dA), dA/pi  # mA, dA in -1:1

def spec(_spe,spe, o,rc, Et, dspe=None, fdeep=0):  # for N_|cent_ | altg_
    for _N in _spe:
        for N in spe:
            if _N is not N:
                dN = comp_N(_N, N, o,rc, fdeep=2); Et += dN.Et  # may be d=cent_?
                if dspe is not None: dspe += [dN]
                if fdeep:
                    for L,l in [(L,l) for L in _N.rim for l in N.rim]:  # l nested in L
                        if L is l: Et += l.Et  # overlap val
                    if _N.altg_ and N.altg_: spec(_N.altg_,N.altg_, o,rc, Et)

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

    _E_ = set()
    # stronger-first
    for rdn, N in enumerate(sorted(N_, key=lambda n: n.et[0]/n.et[2], reverse=True), start=1):
        # ave *= relV of overlap by stronger-E inhibition zones
        roV = rolp(N, _E_)
        if val_(N.et,1, aw = rc + rdn + loopw + roV) > 0:  # cost
            _E_.update({n for l in N.rim for n in l.N_ if n is not N and val_(n.Et,1,aw=rc) > 0})  # selective nrim
            N.exe = 1  # in point cloud of focal nodes
        else:
            break  # the rest of N_ is weaker, trace via rims

def Cluster(root, N_, rc):  # clustering root

    if isinstance(N_[0],CN): Nf_ = N_; N_ = [N_]; Et = root.Et  # nest N_ as rng-banded
    else:                    Nf_ = list(set([N for n_ in N_ for N in n_])); Et = None
    get_exemplars(Nf_,rc)  # set n.exe, cluster rN_ via rng exemplars
    nG = []
    for rng, rN_ in enumerate(N_, start=1):  # bottom-up rng-banded clustering
        aw = rc * rng +contw
        Et = Et if Et is not None else np.sum([n.Et for n in rN_], axis=0)
        if rN_ and val_(Et,1, (len(rN_)-1)*Lw, aw) > 0:
            nG = cluster(root, Nf_, rN_, aw, rng) or nG
    # top valid nG:
    return nG

def cluster(root, iN_, rN_, rc, rng=1):  # flood-fill node | link clusters

    def rroot(n):
        if n.root and n.root.rng > n.rng: return rroot(n.root) if n.root.root else n.root
        else:                             return None
    G_, cent_ = [], []  # exclusive per fork,ave, only centroids can be fuzzy
    for n in iN_: n.fin = 0
    for N in rN_:  # init
        if not N.exe or N.fin: continue  # exemplars or all N_
        N.fin = 1; seen_,N_,link_,long_ = [],[],[],[]
        if rng==1 or (not N.root or N.root.rng==1):  # not rng-banded, L.root is empty
            for l in N.rim:
                seen_ += [l]
                if l.rng==rng and val_(l.Et+ett(l), aw=rc+1) > 0: link_+=[l]; N_ += [N]  # l.Et potentiated by density term
                elif l.rng>rng: long_+=[l]
        else: # N is rng-banded, cluster top-rng roots
            n = N; R = rroot(n)
            if R and not R.fin: N_,link_,long_ = [R], R.L_, R.hL_; R.fin = 1; seen_+=R.link_
        Seen_ = set(link_)  # all visited
        L_ = []
        while link_:  # extend clustered N_ and L_
            L_+=link_; _link_=link_; link_=[]; seen_=[]
            for L in _link_:  # buffer
                for _N in L.N_:
                    if _N not in iN_ or _N.fin: continue  # or always in iN_ now?
                    if rng==1 or (not _N.root or _N.root.rng==1):  # not rng-banded
                        N_ += [_N]; _N.fin = 1  # conditional
                        for l in N.rim:
                            if l in seen_: continue
                            seen_+=[l]
                            if l.rng==rng and val_(l.Et+ett(l), aw=rc) > 0: link_+=[l]
                            elif l.rng>rng: long_ += [l]
                    else:  # cluster top-rng roots
                        _n = _N; _R = rroot(_n)
                        if _R and not _R.fin:
                            seen_+=_R.link_
                            if rolp(N, link_, R=1) > ave*rc:
                                N_+=[_R]; _R.fin=1; _N.fin=1; link_+=_R.link_; long_+=_R.hL_
            link_ = list(set(link_)-Seen_)
            Seen_.update(set(seen_))
        if N_:
            N_, long_ = list(set(N_)), list(set(long_))
            Et, olp = np.zeros(4),0  # sum node_:
            for n in N_:
                Et += n.et; olp += n.olp  # any fork
            if val_(Et,1, (len(N_)-1)*Lw, rc+olp, root.Et) > 0:
                G = sum2graph(root, N_,L_, long_,Et, olp, rng)
                G_ += [G]   # dval:
                if val_(Et, fi=0, mw=G.span*2* slope(L_), aw=olp+centw) > 0:
                    cent_ += [n for n in N_ if n.exe]  # exemplars for cluster_C_ per frame?
    if G_:
        nG = sum_N_(G_,root); nG.cent_ = cent_
        return nG

def cluster_C_(root, rc):  # form centroids by clustering exemplar surround via rims of new member nodes, within root

    _C_, _N_ = [], []
    while root.cent_:  # replace with C_ in the end
        E = root.cent_.pop()
        C = cent_attr( Copy_(E,root, init=2), rc); C.N_ = [E]   # all rims are within root
        C._N_ = [n for l in E.rim for n in l.N_ if n is not E]  # core members + surround -> comp_C
        _N_ += C._N_; _C_ += [C]
    # reset:
    for n in set(root.root.N_+_N_ ): n.C_,n.mo_, n._C_,n._mo_ = [],[], [],[]  # aligned pairs, in cross_comp root
    # reform C_, refine C.N_s
    while True:
        C_, cnt,mat,olp, Dm,Do = [],0,0,0,0,0; Ave = ave * rc * loopw
        _Ct_ = [[c, c.Et[0]/c.Et[2] if c.Et[0] !=0 else 1e-7, c.olp] for c in _C_]
        for _C,_m,_o in sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True):
            if _m > Ave*_o:
                C = cent_attr( sum_N_(_C.N_, root=root, fC=1), rc); C.C_ = []  # C update lags behind N_; non-local C.olp += N.mo_ os?
                _N_,_N__, mo_, M,O, dm,do = [],[],[],0,0,0,0  # per C
                for n in _C._N_:  # core+ surround
                    if C in n.C_: continue
                    m = min_comp(C,n, rc)[1][0]  # val,olp / C:
                    o = np.sum([mo[0] /m for mo in n._mo_ if mo[0]>m])  # overlap = higher-C inclusion vals / current C val
                    cnt += 1  # count comps per loop
                    if m > Ave * o:
                        _N_+=[n]; M+=m; O+=o; mo_ += [np.array([m,o])]  # n.o for convergence eval
                        _N__ += [_n for l in n.rim for _n in l.N_ if _n is not n]  # +|-ls for comp C
                        if _C not in n._C_: dm += m; do += o  # not in extended _N__
                    elif _C in n._C_:
                        __m,__o = n._mo_[n._C_.index(_C)]; dm +=__m; do +=__o
                if M > Ave * len(_N_) * O:
                    for n, mo in zip(_N_,mo_): n.mo_ += [mo]; n.C_ += [C]; C.C_ += [n]  # bilateral assign
                    C.M = M; C.N_+= _N_; C._N_= list(set(_N__))  # core, surround elements
                    C_+=[C]; mat+=M; olp+=O; Dm+=dm; Do+=do   # new incl or excl
                else:
                    for n in _C._N_:
                        for i, c in enumerate(n.C_):
                            if c is _C: n.mo_.pop(i); n.C_.pop(i); break  # remove mo mapping to culled _C
            else: break  # the rest is weaker
        if Dm > Ave * cnt * Do:  # dval vs. dolp, overlap increases as Cs may expand in each loop?
            _C_ = C_
            for n in root.root.N_: n._C_ = n.C_; n._mo_= n.mo_; n.C_,n.mo_ = [],[]  # new n.C_s, combine with vo_ in Ct_?
        else:  # converged
            break
    if  mat > Ave * cnt * olp:
        root.cent_ = (set(C_), mat)
        # cross_comp, low overlap eval in comp_node_?

def cent_attr(C, rc):  # weight attr matches | diffs by their match to the sum, recompute to convergence

    wTT = []  # Cs can be fuzzy only to the extent that their correlation weights are different?

    for derT, wT in zip(C.derTT, wTTf):
        # weigh values by feedback before cross-correlation
        _w_ = np.ones(10)
        val_ = np.abs(derT) * wT  # m|d are signed, but their contribution is absolute
        V = np.sum(val_)
        while True:
            mean = max(V / max(np.sum(_w_), 1e-7), 1e-7)
            inverse_dev_ = np.minimum(val_/mean, mean/val_)  # rational deviation from mean rm in range 0:1, if m=mean
            w_ = inverse_dev_ / .5  # 2/ m=mean, 0/ inf max/min, 1 / mid_rng | ave_dev?
            w_ *= 10 / np.sum(w_)  # mean w = 1, M shouldn't change?
            if np.sum(np.abs(w_-_w_)) > ave*rc:
                V = np.sum(val_ * w_)
                _w_ = w_
            else:  break  # weight convergence
        wTT += [_w_]
    C.wTT = np.array(wTT)  # replace wTTf
    return C

def ett(L): return (L.N_[0].et + L.N_[1].et) * intw

def sum2graph(root, node_,link_,long_, Et, olp, rng):  # sum node,link attrs in graph, aggH in agg+ or player in sub+

    n0 = Copy_(node_[0]); derH = n0.derH; fi = n0.fi
    graph = CN(root=root, fi=1,rng=rng, N_=node_,L_=link_,olp=olp, Et=Et, box=n0.box, baseT=n0.baseT, derTT=n0.derTT)
    graph.hL_ = long_
    n0.root = graph; yx_ = [n0.yx]; fg = fi and isinstance(n0.N_[0],CN)   # not PPs
    Nt = Copy_(n0); DerH = []  #->CN, comb forks: add_N(Nt,Nt.Lt)?
    for N in node_[1:]:
        add_H(derH,N.derH,graph); graph.baseT+=N.baseT; graph.derTT+=N.derTT; graph.box=extend_box(graph.box,N.box); yx_+=[N.yx]; N.root = graph
        if fg: add_N(Nt,N)
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

    derH, derTT, Et = [], np.zeros((3,10)), np.zeros(4)
    for _lay, lay in zip_longest(H,h):  # selective
        if _lay and lay:
            dlay = _lay.comp_lay(lay, rn, root=root)
            derH += [dlay]; derTT = dlay.derTT; Et += dlay.Et
    if Et[2]: DerTT += derTT; ET += Et
    return derH

def sum_H_(Q):  # sum derH in link_|node_, not used
    H = Q[0]; [add_H(H,h) for h in Q[1:]]
    return H

def add_H(H, h, root=0, rn=1, rev=0):  #  layer-wise add|append derH

    for Lay, lay in zip_longest(H,h):  # different len if lay-selective comp
        if lay:
            if Lay: Lay.add_lay(lay, rn)
            else:   H += [lay.copy_(rev)]  # * rn?
            if root: root.derTT += lay.derTT*rn; root.Et += lay.Et*rn
    return H

def add_sett(Sett,sett):
    if Sett: N_,Et = Sett; n_ = sett[0]; N_.update(n_); Et += np.sum([t.Et for t in n_-N_])
    else:    Sett += [copy(par) for par in sett]  # altg_, Et

def sum_N_(node_, root=None, fC=0):  # form cluster G

    G = Copy_(node_[0], root, init = 0 if fC else 1)
    if fC:
        G.N_= [node_[0]]; G.L_ = [] if G.fi else len(node_)
    for n in node_[1:]:
        add_N(G,n,0, fC)
    G.olp /= len(node_)
    if not fC:
        for L in G.L_: G.Et += L.Et
    return G   # no rim

def add_N(N,n, fmerge=0, fC=0):

    if fmerge:  # different in altg_?
        for node in n.N_: node.root=N; N.N_ += [node]
        N.L_ += n.L_  # no L.root assign
    else:
        n.root=N; N.N_ += [n]
        if hasattr(N,"wTT"): N.L_ += n.L_  # for extend C, wTT is recomputed
        else:                N.L_ += [l for l in n.rim if val_(l.Et)>0]
    if n.altg_: add_sett(N.altg_,n.altg_)  # ext clusters
    if n.cent_: add_sett(N.cent_,n.cent_)  # int clusters
    if n.nH: add_NH(N.nH,n.nH, root=N)
    if n.lH: add_NH(N.lH,n.lH, root=N)
    en,_en = n.Et[2],N.Et[2]; rn = en/_en
    for Par,par in zip((N.angl,N.mang,N.baseT,N.derTT,N.Et), (n.angl,n.mang,n.baseT,n.derTT,n.Et)):
        Par += par  # *rn for comp
    if n.derH: add_H(N.derH,n.derH, N, rn)
    if fC: N.olp += np.sum([mo[1] for mo in n._mo_])
    else:  N.olp = (N.olp*_en + n.olp*en) / (en+_en)  # ave of aves
    N.yx = (N.yx*_en + n.yx*en) /(en+_en)
    N.span = max(N.span,n.span)
    N.box = extend_box(N.box, n.box)
    if hasattr(n,'C_') and hasattr(N,'C_'):
        N.C_ += n.C_; N.mo_ += n.mo_  # not sure
    return N

def add_NH(H, h, root, rn=1):

    for Lev, lev in zip_longest(H, h, fillvalue=None):  # always aligned?
        if lev:
            if Lev: add_N(Lev,lev)
            else:   H += [Copy_(lev, root)]

def extend_box(_box, box):
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return min(y0,_y0), min(x0,_x0), max(yn,_yn), max(xn,_xn)

def PP2N(PP, frame):

    P_, link_, verT, latT, A, S, box, yx, Et = PP
    baseT = np.array(latT[:4])
    [mM,mD,mI,mG,mA,mL], [dM,dD,dI,dG,dA,dL], [tM,tD,tI,tG,tA,tL] = verT  # re-pack in derTT:
    derTT = np.array([
        np.array([mM,mD,mL,mM+abs(mD), mI,mG,mA, mL,mL/2, 1e-7]),  # 0 extA
        np.array([dM,dD,dL,dM+abs(dD), dI,dG,dA, dL,dL/2, 1e-7]),
        np.array([tM,tD,tL,tM+abs(tD), tI,tG,tA, tL,tL/2, 1e-7])
    ])
    derH = [CLay(node_=P_,link_=link_, derTT=deepcopy(derTT))]
    y,x,Y,X = box; dy,dx = Y-y,X-x

    return CN(root=frame, fi=1, Et=Et, N_=P_, L_=link_, baseT=baseT, derTT=derTT, derH=derH, box=box, yx=yx, angl=A, span=np.hypot(dy/2,dx/2))

# not used, make H a centroid of layers, same for nH?
def sort_H(H, fi):  # re-assign olp and form priority indices for comp_tree, if selective and aligned

    i_ = []  # priority indices
    for i, lay in enumerate(sorted(H.node_, key=lambda lay: lay.Et[fi], reverse=True)):
        di = lay.i - i  # lay index in H
        lay.olp += di  # derR - valR
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
    derTT = np.zeros((3,10)); Et = np.zeros(4)
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
    Fdist = np.hypot(dy,dx)   # external dist
    rdist = Fdist / Fg.span
    Angle = np.array([dy,dx]) # external angle
    angle = np.sum([L.angl for L in Fg.L_])
    cos_d = angle.dot(Angle) / (np.hypot(*angle) * Fdist)
    # difference between external and internal angles, *= rdist
    ET = np.zeros(4); DerTT = np.zeros((3,10))
    N_ = []
    for _N in Fg.N_:  # sum _N-specific projections for cross_comp
        if len(_N.derH) < 2: continue
        M,D,n,t = _N.Et
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

    wTTf = np.ones((2,10))  # sum derTT coefs: m_,d_ [M,D,n,o, I,G,A,L] / Et, baseT, dimension
    rM, rD, rVd = 1, 1, 0
    hLt = sum_N_(root.L_)  # links between top nodes
    _derTT = np.sum([l.derTT for l in hLt.N_])  # _derTT[np.where(derTT==0)] = 1e-7
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
    dy,dx = Fg.angl; a = dy/ max(dx,1e-7)  # average link_ orientation, projection
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
    PV__  = win__[3].sum( axis=(0,1)) * intw  # init proj vals = sum G in dert[3],        shape: nY=13, nX=20
    aw = contw * 20
    while np.max(PV__) > ave * aw:  # max G * int_w + pV
        # max win index:
        y,x = np.unravel_index(PV__.argmax(), PV__.shape)
        PV__[y,x] = -np.inf  # to skip, | separate in__?
        if foc:
            Fg = frame_blobs_root( win__[:,:,:,y,x], rV)  # [dert, iY, iX, nY, nX]
            vect_root(Fg, rV,wTTf); Fg.L_=[]  # focal dert__ clustering
            cross_comp(Fg, rc=frame.olp)
        else:
            Fg = agg_frame(1, win__[:,:,:,y,x], wY,wX, rV=1, wTTf=[])  # use global wY,wX in nested call
            if Fg and Fg.L_:  # only after cross_comp(PP_)
                rV, wTTf = ffeedback(Fg)  # adjust filters
                Fg = cent_attr(Fg,2)  # compute Fg.wTT: correlation weights in frame derTT?
                wTTf *= Fg.wTT; mW = np.sum(wTTf[0]); dW = np.sum(wTTf[1])  # global?
                wTTf[0] *= 10/mW; wTTf[1] *= 10/dW  # Fg.wTT is redundant?
                # re-norm weights
        if Fg and Fg.L_:
            if fproj and val_(Fg.Et,1, (len(Fg.N_)-1)*Lw, Fg.olp+loopw*20):
                pFg = project_N_(Fg, np.array([y,x]))
                if pFg:
                    pFg = cross_comp(pFg, rc=Fg.olp)  # skip compared_ in FG cross_comp
                    if pFg and val_(pFg.Et,1, (len(pFg.N_)-1)*Lw, pFg.olp+contw*20):
                        project_focus(PV__, y,x, Fg)  # += proj val in PV__
            # no target proj
            frame = add_N(frame, Fg, fmerge=1) if frame else Copy_(Fg)
            aw = contw *20 * frame.Et[2] * frame.olp

    if frame.N_ and val_(frame.Et, (len(frame.N_)-1)*Lw, frame.olp+loopw*20) > 0:

        F = cross_comp(frame, rc=frame.olp+loopw)  # recursive xcomp Fg.N_s
        if F and not foc:
            return F  # foci are not preserved

if __name__ == "__main__":  # './images/toucan_small.jpg' './images/raccoon_eye.jpeg', add larger global image

    iY,iX = imread('./images/toucan_small.jpg').shape
    frame = agg_frame(0, image=imread('./images/toucan.jpg'), iY=iY, iX=iX)
    # search frames ( foci inside image