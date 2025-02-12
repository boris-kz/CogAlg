from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_, aves, Caves
from slice_edge import slice_edge, comp_angle
from comp_slice import comp_slice, comp_latuple, comp_vert
from itertools import combinations, zip_longest
from functools import reduce
from copy import deepcopy, copy
import numpy as np

'''
This code is initially for clustering segments within edge: high-gradient blob, but too complex for that.
It's mostly a prototype for open-ended compositional recursion: clustering blobs, graphs of blobs, etc.
-
rng+ fork: incremental-range cross-comp nodes: edge segments at < max distance, cluster if they match. 
der+ fork: incremental-derivation cross-comp links, from node cross-comp, if abs_diff * rel_adjacent_match 
(variance patterns borrow value from co-projected match patterns because their projections cancel-out)
- 
So graphs should be assigned adjacent alt-fork (der+ to rng+) graphs, to which they lend predictive value.
But alt match patterns borrow already borrowed value, which is too tenuous to track, we use average borrowed value.
Clustering criterion within each fork is summed match of >ave vars (<ave vars are not compared and don't add comp costs).
-
Clustering is exclusive per fork,ave, with fork selected per variable | derLay | aggLay 
Fuzzy clustering can only be centroid-based, because overlapping connectivity-based clusters will merge.
Param clustering if MM, compared along derivation sequence, or combinatorial?
-
Summed-graph representation is nested in a dual tree of down-forking elements: node_, and up-forking clusters: root_.
That resembles neurons, with dendritic tree as input and axonal tree as output. 
But these graphs have recursively nested param sets mapping to each level of the trees, which don't exist in neurons.
-
diagrams: 
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/generic%20graph.drawio.png
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/agg_recursion_unfolded.drawio.png
-
notation:
prefix  f denotes flag
postfix t denotes tuple, multiple ts is a nested tuple
prefix  _ denotes prior of two same-name variables, multiple _s for relative precedence
postfix _ denotes array of same-name elements, multiple _s is nested array
capitalized variables are usually summed small-case variables
'''
ave, ave_d, ave_L, ave_G, max_dist, ave_rn, ccoef, icoef, med_cost, ave_dI = \
aves.B, aves.d, aves.L, aves.G, aves.max_dist, aves.rn, aves.ccoef, aves.icoef, aves.med_cost, aves.dI

class CLay(CBase):  # flat layer if derivation hierarchy
    name = "lay"
    def __init__(l, **kwargs):
        super().__init__()
        l.Et = kwargs.get('Et', np.zeros(4))
        l.root = kwargs.get('root', None)  # higher node or link
        l.node_ = kwargs.get('node_', [])  # concat across fork tree
        l.link_ = kwargs.get('link_', [])
        l.m_d_t = kwargs.get('m_d_t', [[np.zeros(2),np.zeros(6)],[np.zeros(2),np.zeros(6)]])  # [[mext,mver],[dext,dver]], sum across fork tree
        # altL = CLay from comp altG
        # i = kwargs.get('i', 0)  # lay index in root.node_, link_, to revise olp
        # i_ = kwargs.get('i_',[])  # priority indices to compare node H by m | link H by d
        # ni = 0  # exemplar in node_, trace in both directions?
    def __bool__(l): return bool(l.node_)

    def copy_(lay, root=None, rev=0, fc=0, i=None):  # comp direction may be reversed to -1

        if i:  # reuse self
            C = lay; lay = i; C.node_=copy(i.node_); C.link_ = copy(i.link_); C.m_d_t=[]; C.root=root
        else:  # init new C
            C = CLay(root=root, node_=copy(lay.node_), link_=copy(lay.link_))
        C.Et = lay.Et * -1 if (fc and rev) else copy(lay.Et)

        for fd, tt in enumerate(lay.m_d_t):  # nested array tuples
            C.m_d_t += [tt * -1 if rev and (fd or fc) else deepcopy(tt)]

        if not i: return C

    def add_lay(Lay, lay_, rev=0, fc=0):  # merge lays, including mlay + dlay

        if not isinstance(lay_,list): lay_ = [lay_]
        for lay in lay_:
            # rev = dir==-1, to sum/subtract numericals in m_t and d_t:
            for fd, (F_t, f_t) in enumerate(zip(Lay.m_d_t, lay.m_d_t)):
                for F_,f_ in zip(F_t, f_t):
                    F_ += f_ * -1 if rev and (fd or fc) else f_  # m_|d_ in [dext,dver]
            # concat node_,link_:
            Lay.node_ += [n for n in lay.node_ if n not in Lay.node_]
            Lay.link_ += lay.link_
            et = lay.Et * -1 if rev and fc else lay.Et
            Lay.Et += et
        return Lay

    def comp_lay(_lay, lay, rn, root, dir=1):  # unpack derH trees down to numericals and compare them

        _i_t = _lay.m_d_t[1]
        i_t = [i_ * rn * dir for i_ in lay.m_d_t[1]]  # i_ is ds, scale- and direction- normalized
        d_t = [_i_ - i_ for _i_,i_ in zip(_i_t,i_t)]  # [dext,dver])

        _a_t = [np.abs(_i_) for _i_ in _i_t]; a_t = [np.abs(i_) for i_ in i_t]
        m_t = [np.minimum(_a_,a_) / reduce(np.maximum,[_a_,a_,1e-7]) for _a_,a_ in zip(_a_t,a_t)]  # match = min/max comparands
        for f, (_i_,i_) in enumerate(zip(_i_t, i_t)):  # [dext,vert]
            m_t[f][(_i_<0) != (i_<0)] *= -1  # m is negative if comparands have opposite sign
        m_d_t = [m_t,d_t]  # [[mext,mvert],[dext,dvert]]
        node_ = list(set(_lay.node_+ lay.node_))  # concat
        link_ = _lay.link_ + lay.link_
        M = sum(m_t[0]) + sum(m_t[1])
        D = sum(d_t[0]) + sum(d_t[1])
        Et = np.array([M, D, 2 if len(i_t)==1 else 8, (_lay.Et[3]+lay.Et[3])/2])  # n comp params = 2 in dext, 6 in vert
        if root: root.Et += Et

        return CLay(Et=Et, root=root, node_=node_, link_=link_, m_d_t=m_d_t)

class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster
    # graph / node
    def __init__(G,  **kwargs):
        super().__init__()
        G.Et = kwargs.get('Et', np.zeros(4))  # sum all param Ets
        G.fd_ = kwargs.get('fd_',[])  # list of forks forming G, 1 if cluster of Ls | lGs, for feedback only?
        G.root = kwargs.get('root')  # may extend to list in cluster_N_, same nodes may be in multiple dist layers
        G.latuple = kwargs.get('latuple', np.array([.0,.0,.0,.0,.0, np.zeros(2)], dtype=object))  # lateral I,G,M,D,L,[Dy,Dx]
        G.dext = kwargs.get('dext',np.array([np.zeros(2),np.zeros(2)]))  # [mext,dext], sum across fork tree
        G.vert = kwargs.get('vert',np.array([np.zeros(6),np.zeros(6)]))  # vertical m_d_ of latuple, sum across fork tree
        G.derH = kwargs.get('derH',[])  # each lay is [m,d]: Clay(Et,node_,link_,m_d_t), sum|concat links across fork tree
        G.extH = kwargs.get('extH',[])  # sum from rims, single-fork
        G.box = kwargs.get('box', np.array([np.inf,np.inf,-np.inf,-np.inf]))  # y0,x0,yn,xn
        G.yx = kwargs.get('yx', np.zeros(2))  # init PP.yx = [(y0+yn)/2,(x0,xn)/2], then ave node yx
        G.maxL = kwargs.get('maxL', 0)  # if dist-nested in cluster_N_
        G.aRad = 0  # average distance between graph center and node center
        G.altG = []  # adjacent (contour) gap+overlap alt-fork graphs, converted to CG
        # G.fork_tree: list = z([[]])  # indices in all layers(forks, if no fback merge
        # G.fback_ = []  # node fb buffer, n in fb[-1]
        G.nnest = kwargs.get('nnest',0)  # node_H if > 0, node_[-1] is top G_
        G.lnest = kwargs.get('lnest',0)  # link_H if > 0, link_[-1] is top L_
        G.node_ = kwargs.get('node_',[])
        G.link_ = kwargs.get('link_',[])  # internal links
        G.rim = kwargs.get('rim',[])  # external links
    def __bool__(G): return bool(G.node_)  # never empty

class CL(CBase):  # link or edge, a product of comparison between two nodes or links
    name = "link"
    def __init__(l,  **kwargs):
        super().__init__()
        l.nodet = kwargs.get('nodet',[])  # e_ in kernels, else replaces _node,node: not used in kernels
        l.Et = kwargs.get('Et', np.zeros(4))
        l.fd = kwargs.get('fd',0)
        l.derH = kwargs.get('derH',[])  # list of single-fork CLays
        l.angle = kwargs.get('angle',[])  # dy,dx between nodet centers
        l.dist = kwargs.get('dist',0)  # distance between nodet centers
        l.box = kwargs.get('box',[])  # sum nodet, not needed?
        l.yx = kwargs.get('yx',[])
        # add med, rimt, extH in der+
    def __bool__(l): return bool(l.nodet)

def vectorize_root(frame):
    # init for agg+:
    blob_ = unpack_blob_(frame)
    frame2G(frame, derH=[CLay(root=frame, Et=np.zeros(4), m_d_t=[], node_=[],link_=[])], node_=[frame.blob_,[]], root=None)  # distinct from base blob_
    for blob in blob_:
        if not blob.sign and blob.G > ave_G * blob.root.olp:
            edge = slice_edge(blob, frame.aves)
            if edge.G * (len(edge.P_)-1) > ave:  # eval PP
                comp_slice(edge)
                if edge.Et[0] * (len(edge.node_)-1)*(edge.rng+1) > ave:
                    lat = np.array([.0,.0,.0,.0,.0,np.zeros(2)],dtype=object); vert = np.array([np.zeros(6), np.zeros(6)])
                    for PP in edge.node_:
                        vert += PP[3]; lat += PP[4]
                    y_,x_ = zip(*edge.dert_.keys()); box = [min(y_),min(x_),max(y_),max(x_)]
                    blob2G(edge, root=frame, vert=vert,latuple=lat, box=box, yx=np.divide([edge.latuple[:2]], edge.area))  # node_, Et stay the same
                    G_ = []
                    for PP in edge.node_:  # no comp node_, link_ | PPd_ for now
                        P_,link_,vert,lat, A,S,box,[y,x],Et = PP[1:]  # PPt
                        if Et[0] > ave:  # no altG until cross-comp
                            G = CG(root=edge,fd_=[0],Et=Et, node_=P_,link_=[], vert=copy(vert), latuple=lat, box=box, yx=np.array([y,x]),
                                   derH=[[CLay(), CLay(m_d_t = [[np.zeros(2),vert[0]],[np.zeros(2),vert[1]]])]]
                                   )  # 1st layer = [empty dfork, mfork = [empty dext, PP vert]
                            y0,x0,yn,xn = box; G.aRad = np.hypot((yn-y0)/2,(xn-x0)/2)  # approx
                            G_ += [G]
                    if len(G_) > ave_L:
                        edge.node_ = G_; frame.node_[-1] += [edge]
                        cluster_edge(edge)
                        # alt: converted adj_blobs of edge blob?
    frame.derH = sum_H(frame.node_[-1],frame)  # single layer


def val_(Et, coef=1):  # comparison / inclusion eval by m only, no contextual projection

    m, d, n, _ = Et  # skip overlap
    return m - ave * coef * n  # coef per compared param type

def Val_(Et, _Et, coef=1, fd=0):  # m|d cluster|batch eval, + cross|root projection

    m, d, n, o = Et; _m,_d,_n,_o = _Et  # cross-fork induction of root Et alt, same overlap?

    d_loc = d * (_m / (ave * coef * _n))  # diff * co-projected m deviation, no bilateral deviation?
    d_ave = (d / ave_d) * ave  # scale rational deviation of d?

    if fd: val = d_ave + d_loc  # diff borrow val, generic + specific
    else: val = m + d_ave - d_loc  # proj match: += surround val - blocking val, * decay?

    return val - ave * coef * n * o

def cluster_edge(edge):  # edge is CG but not a connectivity cluster, just a set of clusters in >ave G blob, unpack by default?

    def cluster_PP_(N_, fd):
        G_ = []
        while N_:  # flood fill
            node_,link_, et = [],[], np.zeros(4)
            N = N_.pop(); _eN_ = [N]
            while _eN_:
                eN_ = []
                for eN in _eN_:  # rim-connected ext Ns
                    node_ += [eN]
                    for L,_ in get_rim(eN, fd):  # all +ve, * density: if L.Et[0]/ave_d * sum([n.extH.m * ccoef / ave for n in L.nodet])?
                        if L not in link_:
                            for eN in L.nodet:
                                if eN in N_:
                                    eN_ += [eN]; N_.remove(eN)  # merged
                            link_ += [L]; et += L.Et
                _eN_ = {*eN_}
            if Val_(et, _Et=et, fd=fd) > 0:  # cluster eval
                G_ = [sum2graph(edge, [node_,link_,et], fd)]
        if G_:
            edge.node_[:] = [edge.node_[:], G_]  # init nesting in node_|link_
            edge.nnest += 1
    # comp PP_:
    N_,L_,Et = comp_node_(edge.node_)
    edge.link_ += L_
    if Val_(Et, _Et=Et, fd=0) > 0:  # cluster eval
        derH = [[sum_lay_(L_,edge,)]]  # single nested mlay
        if len(N_) > ave_L:
            cluster_PP_(N_, fd=0)
        if Val_(Et, _Et=Et, fd=0) > 0:  # likely not from the same links
            L2N(L_,edge)  # comp dPP_:
            lN_,lL_,dEt = comp_link_(L_,Et)
            if Val_(dEt, _Et=Et, fd=1) > 0:
                derH[0] += [comb_H_(lL_, edge,fd=1)]  # dlay, or sum_lay_?
                if len(lN_) > ave_L:
                    cluster_PP_(lN_, fd=1)
            else:
                derH[0] += [[]]  # empty dlay
        else: derH[0] += [[]]
        edge.derH = derH

def comp_node_(_N_, L=0):  # rng+ forms layer of rim and extH per N, appends N_,L_,Et, ~ graph CNN without backprop

    _Gp_ = []  # [G pair + co-positionals], for top-nested Ns, unless cross-nesting comp:
    if L: _N_ = filter(lambda N: len(N.derH)==L, _N_)
    for _G, G in combinations(_N_, r=2):  # if max len derH in agg+
        _n, n = _G.Et[2], G.Et[2]; rn = _n/n if _n>n else n/_n
        if rn > ave_rn:  # scope disparity or _G.depth != G.depth
            continue
        radii = G.aRad + _G.aRad
        dy,dx = np.subtract(_G.yx,G.yx)
        dist = np.hypot(dy,dx)
        _G.add, G.add = 0, 0
        _Gp_ += [(_G,G, rn, dy,dx, radii, dist)]
    rng = 1
    N_,L_,ET = set(),[],np.zeros(4)
    _Gp_ = sorted(_Gp_, key=lambda x: x[-1])  # sort by dist, shortest pairs first
    while True:  # prior vM
        Gp_,Et = [],np.zeros(4)
        for Gp in _Gp_:
            _G,G, rn, dy,dx, radii, dist = Gp
            _nrim = {L.nodet[1] if L.nodet[0] is _G else L.nodet[0] for L,_ in _G.rim}
            nrim = {L.nodet[1] if L.nodet[0] is G else L.nodet[0] for L,_ in G.rim}
            if _nrim & nrim:  # indirectly connected Gs,
                continue     # no direct match priority?
            # dist vs. radii * induction, mainly / extH?
            GV = val_(_G.Et) + val_(G.Et) + sum([val_(l.Et) for l in _G.extH]) + sum([val_(l.Et) for l in G.extH])
            if dist < max_dist * ((radii * icoef**3) * GV):
                Link = comp_N(_G,G, rn, angle=[dy,dx], dist=dist)
                L_ += [Link]  # include -ve links
                if val_(Link.Et) > 0:
                    N_.update({_G,G}); Et += Link.Et; _G.add,G.add = 1,1
            else:
                Gp_ += [Gp]  # re-evaluate not-compared pairs with one incremented N.M
        ET += Et
        if val_(Et) > 0:  # current-rng vM
            _Gp_ = [Gp for Gp in Gp_ if Gp[0].add or Gp[1].add]  # one incremented N.M
            rng += 1
        else:  # low projected rng+ vM
            break

    return  list(N_), L_, ET  # flat N__ and L__

def comp_link_(iL_, iEt):  # comp CLs via directional node-mediated link tracing: der+'rng+ in root.link_ rim_t node rims

    fd = isinstance(iL_[0].nodet[0], CL)
    for L in iL_:
        # init mL_t: bilateral mediated Ls per L:
        for rev, N, mL_ in zip((0,1), L.nodet, L.mL_t):
            for _L,_rev in N.rimt[0]+N.rimt[1] if fd else N.rim:
                if _L is not L and _L in iL_:  # nodet-mediated
                    if val_(_L.Et) > 0:
                        mL_ += [(_L, rev ^_rev)]  # direction of L relative to _L
    _L_, out_L_, LL_, ET = iL_,set(),[],np.zeros(4)  # out_L_: positive subset of iL_, Et = np.zeros(4)?
    med = 1
    while True:  # xcomp _L_
        L_, Et = set(), np.zeros(4)
        for L in _L_:
            for mL_ in L.mL_t:
                for _L, rev in mL_:  # rev is relative to L
                    rn = _L.Et[2] / L.Et[2]
                    if rn > ave_rn: continue  # scope disparity, no diff nesting?
                    dy,dx = np.subtract(_L.yx,L.yx)
                    Link = comp_N(_L,L, rn,angle=[dy,dx],dist=np.hypot(dy,dx), dir = -1 if rev else 1)  # d = -d if L is reversed relative to _L
                    Link.med = med
                    LL_ += [Link]  # include -ves, link order: nodet < L < rimt, mN.rim || L
                    if val_(Link.Et) > 0:  # link induction
                        out_L_.update({_L,L}); L_.update({_L,L}); Et += Link.Et
        ET += Et
        if not any(L_): break
        # extend mL_t per last medL:
        if val_(Et) - med * med_cost > 0:  # project prior-loop value, med adds fixed cost
            _L_, ext_Et = set(), np.zeros(4)
            for L in L_:
                mL_t, lEt = [set(),set()], np.zeros(4)  # __Ls per L
                for mL_,_mL_ in zip(mL_t, L.mL_t):
                    for _L, rev in _mL_:
                        for _rev, N in zip((0,1), _L.nodet):
                            rim = N.rimt if fd else N.rim
                            if len(rim) == med:  # append in comp loop
                                for __L,__rev in rim[0]+rim[1] if fd else rim:
                                    if __L in L.visited_ or __L not in iL_: continue
                                    L.visited_ += [__L]; __L.visited_ += [L]
                                    if val_(__L.Et) > 0:  # add coef for loop induction?
                                        mL_.add((__L, rev ^_rev ^__rev))  # combine reversals: 2 * 2 mLs, but 1st 2 are pre-combined
                                        lEt += __L.Et
                if val_(lEt) > 0:  # L'rng+, vs L'comp above
                    L.mL_t = mL_t; _L_.add(L); ext_Et += lEt
            # refine eval by extension D:
            if val_(ext_Et) - med * med_cost > 0:
                med +=1
            else: break
        else: break
    return out_L_, LL_, ET

def extend_box(_box, box):  # extend box with another box
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box
    return min(y0, _y0), min(x0, _x0), max(yn, _yn), max(xn, _xn)

def comp_area(_box, box):
    _y0,_x0,_yn,_xn =_box; _A = (_yn - _y0) * (_xn - _x0)
    y0, x0, yn, xn = box;   A = (yn - y0) * (xn - x0)
    return _A-A, min(_A,A) - ave_L**2  # mA, dA

def comp_dext(_dext, dext, rn, dir=1):
    (_dL, _dA), (dL, dA) = _dext,dext

    ddL = _dL - dL * rn * dir; mdL = min(_dL, dL*rn) / max(_dL, dL*rn) - ave_L  # m/mag
    if _dL < 0 != dL < 0: mdL = -mdL  # m is negative for comparands of opposite sign
    ddA = _dA - dA * rn * dir; mdA = min(_dA, dA*rn) / max(_dA, dA*rn) - 2
    if _dA < 0 != dA < 0: mdA = -mdA

    return np.array([np.array([mdL,mdA]),np.array([ddL,ddA])])

def comp_N(_N,N, rn, angle=None, dist=None, dir=1):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = isinstance(N,CL); dderH=[]  # compare links, relative N direction = 1|-1
    # comp externals:
    if fd:
        _L, L = _N.dist, N.dist; L*=rn; dL = _L - L; mL = min(_L,L) / max(_L,L) - ave_L  # rm
        mA,dA = comp_angle(_N.angle, [d*dir *rn for d in N.angle])  # rev 2nd link in llink
        # comp med if LL: isinstance(>nodet[0],CL), higher-order version of comp dist?
    else:
        _L, L = len(_N.node_),len(N.node_); dL = _L- L*rn; mL = min(_L, L*rn) - ave_L
        mA,dA = comp_area(_N.box, N.box)  # compare area in CG vs angle in CL
    # init
    dext = np.array( [np.array([mL,mA]), np.array([dL,dA])])
    M = mL+mA; D = abs(dL)+abs(dA); _o,o = _N.Et[3],N.Et[3]; olp=(_o+o)/2  # inherited
    Et = np.array([M,D, 2 if fd else 8, olp])  # n comp vars
    Link = CL(fd=fd, nodet=[_N,N], yx=np.add(_N.yx,N.yx)/2, angle=angle, dist=dist, box=extend_box(N.box,_N.box))
    if fd:
        vert = np.array([np.zeros(6),np.zeros(6)])
    else:  # default vert:
        vert, (Mv,Dv) = comp_vert(_N.vert[1], N.vert[1])
        if np.any(N.dext):
            ddext = comp_dext(_N.dext[1], N.dext[1], rn)  # combine der_ext across all layers, same as vert
            dext += ddext; M += np.sum(ddext[0]); D += np.sum(ddext[1])
        M+= Mv; D+= Dv; Et[:2] = M,D
        if M > ave:  # specification
            dLat,lEt = comp_latuple(_N.latuple, N.latuple, _o,o) # lower value
            Et += np.array([lEt[0], lEt[1], 2, 0])  # same olp?
            vert += dLat
    if M > ave and (len(N.derH) > 2 or isinstance(N,CL)):  # else derH is redundant to dext,vert
        dderH = comp_H(_N.derH, N.derH, rn, Link, Et, fd)  # comp shared layers, if any
        # comp_node_(node_|link_)
    Link.derH = [CLay(root=Link,Et=Et,node_=[_N,N],link_=[Link], m_d_t=[[dext[0],vert[0]],[dext[1],vert[1]]]), *dderH]
    # spec:
    if not fd and _N.altG and N.altG:  # if alt M?
        Link.altL = comp_N(_N.altG, N.altG, _N.altG.Et[2] / N.altG.Et[2])
        Et += Link.altL.Et
    Link.Et = Et
    if val_(Et) > 0:
        for rev, node in zip((0,1), (N,_N)):  # reverse Link direction for _N
            if fd: node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            else:  node.rim += [(Link,dir)]
            add_H(node.extH, Link.derH, root=node, rev=rev, fd=1)
            node.Et += Et
    return Link

def get_rim(N,fd): return N.rimt[0] + N.rimt[1] if fd else N.rim  # add nesting in cluster_N_?

def sum2graph(root, grapht, fd, minL=0, maxL=None):  # sum node and link params into graph, aggH in agg+ or player in sub+

    node_, link_, Et = grapht
    graph = CG(fd_=node_[0].fd_+[fd], Et=Et*icoef, root=root, node_=[],link_=link_, maxL=maxL, nnest=root.nnest, lnest=root.lnest)
    # arg Et is weaker if internal, maxL,minL: max and min L.dist in graph.link_
    N_, yx_ = [],[]
    for N in node_:
        fc = 0
        if minL:  # > 0, inclusive, = lower-layer exclusive maxL, if G was distance-nested in cluster_N_
            while N.root.maxL and N.root is not graph and (minL != N.root.maxL):  # maxL=0 in edge|frame
                if N.root is graph:
                    fc=1; break  # graph was assigned as root via prior N
                else: N = N.root  # cluster prior-dist graphs vs. nodes
        if fc: continue  # N.root was clustered in prior loop
        else: N_ += [N]  # roots if minL
        graph.box = extend_box(graph.box, N.box)  # pre-compute graph.area += N.area?
        yx_ += [N.yx]
        if isinstance(node_[0],CG): graph.latuple += N.latuple
        graph.Et += N.Et * icoef ** 2  # deeper, lower weight
        N.root = graph
    graph.node_= N_  # nodes or roots, link_ is still current-dist links only?
    graph.derH = [[CLay(root=graph), lay] for lay in sum_H(link_, graph, fd=1)]  # sum and nest link derH
    for lay in graph.derH:
        for fork in lay:
            if fork.m_d_t:
                graph.dext += fork.m_d_t[1][0]; graph.vert += fork.m_d_t[1][1]
    yx = np.mean(yx_, axis=0)
    dy_,dx_ = (graph.yx - yx_).T; dist_ = np.hypot(dy_,dx_)
    graph.aRad = dist_.mean()  # ave distance from graph center to node centers
    graph.yx = yx
    if fd:  # dgraph, no mGs / dG for now  # and val_(Et, _Et=root.Et) > 0:
        altG = []  # mGs overlapping dG
        for L in node_:
            for n in L.nodet:  # map root mG
                mG = n.root
                if mG not in altG:
                    mG.altG += [graph]  # cross-comp|sum complete altG before next agg+ cross-comp
                    altG += [mG]
    return graph

def sum_lay_(link_, root):
    lay0 = CLay(root=root)
    for link in link_: lay0.add_lay(link.derH[0])
    return lay0

def comb_H_(L_, root, fd):

    derH = sum_H(L_,root,fd=fd)
    Lay = CLay(root=root)
    for lay in derH: Lay.add_lay(lay)
    return Lay

def sum_H(Q, root, rev=0, fc=0, fd=0):  # sum derH in link_|node_
    DerH = []
    for e in Q: add_H(DerH, e.derH, root, rev, fc, fd)
    return DerH

def add_H(H, h, root, rev=0, fc=0, fd=0):  # add fork L.derHs

    for Lay,lay in zip_longest(H,h):  # different len if lay-selective comp
        if lay:
            if fd: # one-fork lays
                if Lay: Lay.add_lay(lay,rev=rev,fc=fc)
                else: H += [lay.copy_(root=root,rev=rev,fc=fc)]
                root.Et += lay.Et
            else:  # two-fork lays
                if Lay:
                    for i, (Fork,fork) in enumerate(zip_longest(Lay,lay)):
                        if fork:
                            if Fork:
                                if fork: Fork.add_lay(fork, rev=rev,fc=fc)
                            else: Lay[i] = fork.copy_(root=root)
                else:
                    Lay = []
                    for fork in lay:
                        if fork:
                            Lay += [fork.copy_(root=root,rev=rev,fc=fc)]; root.Et += fork.Et
                        else: Lay += [[]]
                    H += [Lay]

def comp_H(H,h, rn, root, Et, fd):  # one-fork derH if fd, else two-fork derH

    derH = []
    for _lay,lay in zip_longest(H,h):  # different len if lay-selective comp
        if _lay and lay:
            if fd:  # one-fork lays
                dLay = _lay.comp_lay(lay, rn, root=root)
            else:  # two-fork lays
                dLay = []
                for _fork,fork in zip_longest(_lay,lay):
                    if _fork and fork:
                        dlay = _fork.comp_lay(fork, rn,root=root)
                        if dLay: dLay.add_lay(dlay, root=root)  # sum ds between input forks
                        else:    dLay = dlay
            Et += dLay.Et
            derH += [dLay]
    return derH

def L2N(link_,root):
    for L in link_:
        L.root = root; L.fd_=copy(L.nodet[0].fd_); L.mL_t,L.rimt = [[],[]],[[],[]]; L.aRad=0; L.visited_,L.extH = [],[]

def frame2G(G, **kwargs):
    blob2G(G, **kwargs)
    G.derH = kwargs.get('derH', [CLay(root=G, Et=np.zeros(4), m_d_t=[], node_=[],link_ =[])])
    G.Et = kwargs.get('Et', np.zeros(4))
    G.node_ = kwargs.get('node_', [])
    G.aves = Caves()  # per frame's aves

def blob2G(G, **kwargs):
    # node_, Et stays the same:
    G.fd_ = []  # fd=1 if cluster of Ls|lGs?
    G.root = kwargs.get('root')  # may extend to list in cluster_N_, same nodes may be in multiple dist layers
    G.link_ = kwargs.get('link_',[])
    G.nnest = kwargs.get('nnest',0)  # node_H if > 0, node_[-1] is top G_
    G.lnest = kwargs.get('lnest',0)  # link_H if > 0, link_[-1] is top L_
    G.derH = []  # sum from nodes, then append from feedback, maps to node_tree
    G.extH = []  # sum from rims
    G.latuple = kwargs.get('latuple', np.array([.0,.0,.0,.0,.0,np.zeros(2)],dtype=object))  # lateral I,G,M,D,L,[Dy,Dx]
    G.vert = kwargs.get('vert', np.array([np.zeros(6), np.zeros(6)]))  # vertical m_d_ of latuple
    G.dext = kwargs.get('dext', np.array([np.zeros(2), np.zeros(2)]))
    G.box = kwargs.get('box', np.array([np.inf,np.inf,-np.inf,-np.inf]))  # y0,x0,yn,xn
    G.yx = kwargs.get('yx', np.zeros(2))  # init PP.yx = [(y0+yn)/2,(x0,xn)/2], then ave node yx
    G.rim = []  # flat links of any rng, may be nested in clustering
    G.maxL = 0  # nesting in nodes
    G.aRad = 0  # average distance between graph center and node center
    G.altG = []  # or altG? adjacent (contour) gap+overlap alt-fork graphs, converted to CG
    return G

if __name__ == "__main__":
   # image_file = './images/raccoon_eye.jpeg'
    image_file = './images/toucan_small.jpg'
    image = imread(image_file)
    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vectorize_root(frame)