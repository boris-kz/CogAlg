from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_, aves
from slice_edge import slice_edge, comp_angle
from comp_slice import comp_slice
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
ave      = aves[-2]
ave_d    = aves[1]
ave_L    = aves[6]
ave_G    = aves[4]
max_dist = aves[9]
ave_rn   = aves[8]
icoef    = aves[12]
med_cost = aves[13]
ave_dI   = aves[14]

class CLay(CBase):  # flat layer if derivation hierarchy
    name = "lay"
    def __init__(l, **kwargs):
        super().__init__()
        l.Et = kwargs.get('Et', np.zeros(4))
        l.root = kwargs.get('root', None)  # higher node or link
        l.node_ = kwargs.get('node_', [])  # concat across fork tree
        l.link_ = kwargs.get('link_', [])
        l.derTT = kwargs.get('derTT', np.zeros((2,8)))  # [[mBase,mEt,mExt],[dBase,dEt,dExt]], sum across fork tree
        # altL = CLay from comp altG
        # i = kwargs.get('i', 0)  # lay index in root.node_, link_, to revise olp
        # i_ = kwargs.get('i_',[])  # priority indices to compare node H by m | link H by d
        # ni = 0  # exemplar in node_, trace in both directions?
    def __bool__(l): return bool(l.node_)

    def copy_(lay, root=None, rev=0, fc=0, i=None):  # comp direction may be reversed to -1

        if i:  # reuse self
            C = lay; lay = i; C.node_=copy(i.node_); C.link_ = copy(i.link_); C.derTT=[]; C.root=root
        else:  # init new C
            C = CLay(root=root, node_=copy(lay.node_), link_=copy(lay.link_))
        C.Et = lay.Et * -1 if (fc and rev) else copy(lay.Et)

        for fd, tt in enumerate(lay.derTT):  # nested array tuples
            C.derTT += [tt * -1 if rev and (fd or fc) else deepcopy(tt)]

        if not i: return C

    def add_lay(Lay, lay_, rev=0, fc=0):  # merge lays, including mlay + dlay

        if not isinstance(lay_,list): lay_ = [lay_]
        for lay in lay_:
            # rev = dir==-1, to sum/subtract numericals in m_ and d_:
            for fd, (F_, f_) in enumerate(zip(Lay.derTT, lay.derTT)):
                F_ += f_ * -1 if rev and (fd or fc) else f_  # m_|d_
            # concat node_,link_:
            Lay.node_ += [n for n in lay.node_ if n not in Lay.node_]
            Lay.link_ += lay.link_
            et = lay.Et * -1 if rev and fc else lay.Et
            Lay.Et += et
        return Lay

    def comp_lay(_lay, lay, rn, root, dir=1):  # unpack derH trees down to numericals and compare them

        i_ = lay.derTT[1] * rn * dir; _i_ = _lay.derTT[1]
        # i_ is ds, scale and direction- normalized
        d_ = _i_ - i_
        a_ = np.abs(i_); _a_ = np.abs(_i_)
        m_ = np.minimum(_a_,a_) / reduce(np.maximum,[_a_,a_,1e-7])  # match = min/max comparands
        m_[(_i_<0) != (i_<0)] *= -1  # match is negative if comparands have opposite sign
        derTT = np.array([m_,d_])
        node_ = list(set(_lay.node_+ lay.node_))  # concat
        link_ = _lay.link_ + lay.link_
        Et = np.array([sum(m_),sum(d_), 8, (_lay.Et[3]+lay.Et[3])/2])  # n compared params = 8
        if root: root.Et += Et

        return CLay(Et=Et, root=root, node_=node_, link_=link_, derTT=derTT)


class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster
    # graph / node
    def __init__(G,  **kwargs):
        super().__init__()
        G.fd = kwargs.get('fd',0)  # or fd_: list of forks forming G, 1 if cluster of Ls | lGs, for feedback only?
        G.root = kwargs.get('root')  # may extend to list in cluster_N_, same nodes may be in multiple dist layers
        G.Et = kwargs.get('Et', np.zeros(4))  # sum all params M,D,n,o
        G.yx = kwargs.get('yx', np.zeros(2))  # init PP.yx = [(y+Y)/2,(x,X)/2], then ave node yx
        G.box = kwargs.get('box', np.array([np.inf,-np.inf,np.inf,-np.inf]))  # y,Y,x,X, area: (Y-y)*(X-x),
        G.baseT = kwargs.get('baseT',[])  # I,G,Dy,Dx
        G.derTT = kwargs.get('derTT',np.zeros((2,8)))  # m,d / baseT,Et,box, summed across derH lay forks
        G.derTTe = kwargs.get('derTTe',np.zeros((2,8)))  # sum across link.derHs
        G.derH = kwargs.get('derH',[])  # each lay is [m,d]: Clay(Et,node_,link_,derTT), sum|concat links across fork tree
        G.extH = kwargs.get('extH',[])  # sum from rims, single-fork
        G.maxL = kwargs.get('maxL', 0)  # if dist-nested in cluster_N_
        G.aRad = 0  # average distance between graph center and node center
        G.altG = CG(altG=G,fd=1) if kwargs.get('altG') is None else kwargs.get('altG')  # adjacent (contour) gap+overlap alt-fork graphs, converted to CG
        # G.fork_tree: list = z([[]])  # indices in all layers(forks, if no fback merge
        # G.fback_ = []  # node fb buffer, n in fb[-1]
        G.nnest = kwargs.get('nnest',0)  # node_H if > 0, node_[-1] is top G_
        G.lnest = kwargs.get('lnest',0)  # link_H if > 0, link_[-1] is top L_
        G.node_ = kwargs.get('node_',[])
        G.link_ = kwargs.get('link_',[])  # internal links
        G.rim = kwargs.get('rim',[])  # external links
    def __bool__(G): return bool(G.node_)  # never empty

def copy_(N):
    C = CG(); fd = N.fd
    for name, value in N.__dict__.items():
        val = getattr(N, name)
        if name == 'derH':
            for lay in N.derH:
                if fd: C.derH += [lay.copy_(root=C)]  # CLay
                else:  C.derH += [[fork.copy_(root=C) for fork in lay]]
        elif name == 'extH':
            C.extH = [lay.copy_(root=C) for lay in N.extH]
        elif isinstance(value,list) or isinstance(value,np.ndarray):
            setattr(C, name, copy(val))  # Et,yx,box, node_,link_,rim_, altG, baseT, derTT
        else:
            setattr(C, name, val)  # numeric or object: fd, root, maxL, aRad, nnest, lnest
    return C

class CL(CBase):  # link or edge, a product of comparison between two nodes or links
    name = "link"
    def __init__(l,  **kwargs):
        super().__init__()
        l.nodet = kwargs.get('nodet',[])  # e_ in kernels, else replaces _node,node: not used in kernels
        l.Et = kwargs.get('Et', np.zeros(4))
        l.fd = kwargs.get('fd',0)
        l.yx = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet
        l.box = kwargs.get('box', np.array([np.inf,-np.inf,np.inf,-np.inf]))  # y,Y,x,X: angle=(Y-y,X-x); dist=hypot(angle),
        l.derH = kwargs.get('derH',[])  # list of single-fork CLays
        l.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m,d (baseT,Et,box), summed across derH
        l.dist = kwargs.get('dist',0)  # distance between nodet centers, redundant to box
        l.angle = kwargs.get('angle',[])  # dy,dx between nodet centers, redundant to box
        # add med, rimt, extH in der+
    def __bool__(l): return bool(l.nodet)

def vectorize_root(frame):  # init for agg+:

    blob_ = unpack_blob_(frame)
    frame2G(frame, derH=[CLay(root=frame)], node_=[blob_], root=None)
    edge_ = []  # cluster, unpack
    for blob in blob_:
        if not blob.sign and blob.G > ave_G * blob.root.olp:
            edge = slice_edge(blob)
            if edge.G * (len(edge.P_)-1) > ave:  # eval PP
                comp_slice(edge)
                if edge.Et[0] * (len(edge.node_)-1)*(edge.rng+1) > ave:
                    G_ = [PP2G(PP)for PP in edge.node_ if PP[-1][0] > ave]  # Et, no altGs
                    if len(G_) > ave_L:  # no comp node_,link_,PPd_
                        edge_ += [cluster_edge(G_, frame)]  # 1layer derH, alt: converted adj_blobs of edge blob | alt_P_?
    # unpack edges:
    Lay = [CLay(root=frame), CLay(root=frame)]
    PP_,G_,L_,lG_ = [],[],[],[]
    for edget in edge_:
        if edget:
            pp_,g_,l_,lg_,lay = edget
            [F.add_lay(f) for F,f in zip(Lay,lay)]  # [mfork, dfork]
            PP_+= pp_; G_+= g_; L_+= l_; lG_+= lg_
    frame.derH = [Lay]
    frame.link_ = [L_,sum_G_(lG_)]
    frame.node_ += [PP_,sum_G_(G_)]
    frame.baseT = np.sum([G.baseT for G in PP_+ G_], axis=0)
    frame.derTT = np.sum([L.derTT for L in L_+ lG_], axis=0)
    if G_: frame.nnest = 1
    if lG_: frame.lnest = 1
    return frame

def cluster_edge(iG_, frame):  # edge is CG but not a connectivity cluster, just a set of clusters in >ave G blob, unpack by default?

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
            if Val_(et, et, ave*2, fd=fd) > 0:
                Lay = CLay()
                [[Lay.add_lay(lay) for lay in link.derH[0]] for link in link_]
                G_ = [sum2graph(frame, [node_,link_,et, Lay], fd)]
        return G_

    N_,L_,Et = comp_node_(iG_)  # comp PP_
    # mval -> lay:
    if Val_(Et, Et, ave, fd=0) > 0:
        lay = [sum_lay_(L_, frame)]  # [mfork]
        G_ = cluster_PP_(copy(N_), fd=0) if len(N_) > ave_L else []
        # dval -> comp dPP_:
        if Val_(Et, Et, ave*2, fd=1) > 0:  # likely not from the same links
            lN_,lL_,dEt = comp_link_(L2N(L_),Et)
            if Val_(dEt, Et, ave*2, fd=1) > 0:
                lay += [sum_lay_(lL_, frame)]  # dfork
                lG_ = cluster_PP_(lN_, fd=1) if len(lN_) > ave_L else []
            else:
                lay += [CLay()]; lG_=[]  # empty dfork
        else: lay += [CLay()]; lG_=[]

        return [N_,G_, L_,lG_, lay]

def val_(Et, ave, coef=1):  # comparison / inclusion eval by m only, no contextual projection

    m, d, n, _ = Et  # skip overlap
    return m - ave * coef * n  # coef per compared param type

def Val_(Et, _Et, ave, coef=1, fd=0):  # m|d cluster|batch eval, + cross|root projection

    m, d, n, o = Et; _m,_d,_n,_o = _Et  # cross-fork induction of root Et alt, same overlap?

    d_loc = d * (_m * ave * coef * _n)  # diff * co-projected m deviation, no bilateral deviation?
    d_ave = d * ave_d * ave  # coef aves, scale rational deviation of d?

    if fd: val = d_ave + d_loc  # diff borrow val, generic + specific
    else:  val = m + d_ave - d_loc  # proj match: += surround val - blocking val, * decay?

    return val - ave * coef * n * o

def comp_node_(_N_, ave, L=0):  # rng+ forms layer of rim and extH per N, appends N_,L_,Et, ~ graph CNN without backprop

    _Gp_ = []  # [G pair + co-positionals], for top-nested Ns, unless cross-nesting comp:
    if L: _N_ = filter(lambda N: len(N.derH)==L, _N_)  # if dist-nested
    for _G, G in combinations(_N_, r=2):  # if max len derH in agg+
        _n, n = _G.Et[2], G.Et[2]; rn = _n/n if _n>n else n/_n
        if rn > ave * ave_rn:  # scope disparity or _G.depth != G.depth, not needed?
            continue
        radii = G.aRad + _G.aRad
        dy,dx = np.subtract(_G.yx,G.yx)
        dist = np.hypot(dy,dx)
        _G.add, G.add = 0, 0
        _Gp_ += [(_G,G, rn, dy,dx, radii, dist)]
    rng = 1
    N_,L_,ET = set(),[],np.zeros(4)
    _Gp_ = sorted(_Gp_, key=lambda x: x[-1])  # sort by dist, closest pairs first
    while True:  # prior vM
        Gp_,Et = [],np.zeros(4)
        for Gp in _Gp_:
            _G,G, rn, dy,dx, radii, dist = Gp
            _nrim = {L.nodet[1] if L.nodet[0] is _G else L.nodet[0] for L,_ in _G.rim}
            nrim = {L.nodet[1] if L.nodet[0] is G else L.nodet[0] for L,_ in G.rim}
            if _nrim & nrim:  # indirectly connected Gs,
                continue     # no direct match priority?
            # dist vs. radii * induction, mainly / extH?
            if dist < max_dist * ((radii * icoef**3) * (val_(_G.Et, ave)+val_(G.Et, ave))):  # ext V is not complete
                Link = comp_N(_G,G, ave, angle=[dy,dx], dist=dist)
                L_ += [Link]  # include -ve links
                if val_(Link.Et, ave) > 0:
                    N_.update({_G,G}); Et += Link.Et; _G.add,G.add = 1,1
            else:
                Gp_ += [Gp]  # re-evaluate not-compared pairs with one incremented N.M
        ET += Et
        if val_(Et, ave) > 0:  # current-rng vM
            _Gp_ = [Gp for Gp in Gp_ if Gp[0].add or Gp[1].add]  # one incremented N.M
            rng += 1
        else:  # low projected rng+ vM
            break

    return  list(N_), L_, ET  # flat N__ and L__

def comp_link_(iL_, ave):  # comp CLs via directional node-mediated link tracing: der+'rng+ in root.link_ rim_t node rims

    fd = isinstance(iL_[0].nodet[0], CL)
    for L in iL_:
        # init mL_t: bilateral mediated Ls per L:
        for rev, N, mL_ in zip((0,1), L.nodet, L.mL_t):
            for _L,_rev in N.rimt[0]+N.rimt[1] if fd else N.rim:
                if _L is not L and _L in iL_:  # nodet-mediated
                    if val_(_L.Et, ave) > 0:
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
                    Link = comp_N(_L,L, ave, angle=[dy,dx],dist=np.hypot(dy,dx), dir = -1 if rev else 1)  # d = -d if L is reversed relative to _L
                    Link.med = med
                    LL_ += [Link]  # include -ves, link order: nodet < L < rimt, mN.rim || L
                    if val_(Link.Et, ave) > 0:  # link induction
                        out_L_.update({_L,L}); L_.update({_L,L}); Et += Link.Et
        ET += Et
        if not any(L_): break
        # extend mL_t per last medL:
        if val_(Et, ave) - med * med_cost > 0:  # project prior-loop value, med adds fixed cost
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
                                    if val_(__L.Et, ave) > 0:  # add coef for loop induction?
                                        mL_.add((__L, rev ^_rev ^__rev))  # combine reversals: 2 * 2 mLs, but 1st 2 are pre-combined
                                        lEt += __L.Et
                if val_(lEt, ave) > 0:  # L'rng+, vs L'comp above
                    L.mL_t = mL_t; _L_.add(L); ext_Et += lEt
            # refine eval by extension D:
            if val_(ext_Et, ave) - med * med_cost > 0:
                med +=1
            else: break
        else: break
    return out_L_, LL_, ET

def extend_box(_box, box):  # extend box with another box
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box

    return min(y0, _y0), min(x0, _x0), max(yn, _yn), max(xn, _xn)

def base_comp(_N, N, dir=1, fd=0):  # comp Et, Box, baseT, derTT

    _M,_D,_n = _N.Et[:-1]; M,D,n = N.Et[:-1]
    # comp Et:
    rn = _n/n; mn = (ave_rn-rn) / max(rn, ave_rn)  # ? * priority coef?
    nM = M*rn; dM = _M - nM; mM = min(_M,nM) / max(_M,nM)
    nD = D*rn; dD = _D - nD; mD = min(_D,nD) / max(_D,nD)
    # comp baseT:
    if N.fd:
        mI,mG,mgA, dI,dG,dgA = .0,.0,.0,.0,.0,.0  # empty in CL
    else:
        _I,_G,_Dy,_Dx = _N.baseT; I,G,Dy,Dx = N.baseT   # I,G,Angle
        I*=rn; dI = _I - I; mI = abs(dI) / ave_dI
        G*=rn; dG = _G - G; mG = min(_G,G) / max(_G,G)
        mgA, dgA = comp_angle((_Dy,_Dx),(Dy*rn,Dx*rn))
    # comp ext:
    _y,_x,_Y,_X = _N.box; y,x,Y,X = np.array(N.box) * rn
    _dy,_dx, dy, dx = _Y-_y, _X-_x, Y-y, X-x
    mA, dA = comp_angle((_dy,_dx),(dy,dx))
    _L = _dy * _dx; L = dy * dx  # area
    dL = _L - L; mL = min(_L,L) / max(_L,L)

    _m_, _d_ = np.array([mM,mD,mn, mI,mG,mgA, mL,mA]), np.array([dM,dD,rn, dI,dG,dgA, dL,dA])
    # comp derTT:
    _i_ = _N.derTT[1]; i_ = N.derTT[1] * rn  # 8 params, normalize by compared accum span
    d_ = (_i_ - i_ * dir)  # np.arrays
    _a_,a_ = np.abs(_i_),np.abs(i_)
    m_ = np.divide( np.minimum(_a_,a_), reduce(np.maximum, [_a_,a_,1e-7]))  # rms
    m_[(_i_<0) != (i_<0)] *= -1  # m is negative if comparands have opposite sign

    # each [M,D,n, I,G,gA, L,A]: L is area
    return [m_+_m_, d_+_d_], rn

def comp_N(_N,N, ave, angle=None, dist=None, dir=1):  # compare links, relative N direction = 1|-1, no need for angle, dist?
    fd = isinstance(N, CL);  dderH = []

    [m_,d_], rn = base_comp(_N, N, dir, fd)
    M = sum(m_); D = sum(np.abs(d_))
    Et = np.array([M,D, 8, (_N.Et[3]+ N.Et[3]) /2])  # n comp vars, inherited olp
    derTT = np.array([m_,d_])
    Link = CL(fd=fd,nodet=[_N,N], derTT=derTT, yx=np.add(_N.yx,N.yx)/2, angle=angle, dist=dist, box=extend_box(N.box,_N.box))
    # spec / lay:
    if M > ave and (len(N.derH) > 2 or isinstance(N,CL)):  # else derH is redundant to dext,vert
        dderH = comp_H(_N.derH, N.derH, rn, Link, Et, fd)  # comp shared layers, if any
        # spec/ comp_node_(node_|link_)
    Link.derH = [CLay(root=Link,Et=Et,node_=[_N,N],link_=[Link], derTT=copy(derTT)), *dderH]
    for lay in dderH: derTT += lay.derTT
    # spec / alt:
    if not fd and _N.altG and N.altG:  # if alt M?
        Link.altL = comp_N(_N.altG, N.altG, _N.altG.Et[2] / N.altG.Et[2])
        Et += Link.altL.Et
    Link.Et = Et
    if val_(Et, ave) > 0:
        for rev, node in zip((0,1), (N,_N)):  # reverse Link direction for _N
            if fd: node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            else:  node.rim += [(Link,dir)]
            add_H(node.extH, Link.derH, root=node, rev=rev, fd=1)
            node.Et += Et
    return Link

def get_rim(N,fd): return N.rimt[0] + N.rimt[1] if fd else N.rim  # add nesting in cluster_N_?

def sum2graph(root, grapht, fd, minL=0, maxL=None):  # sum node and link params into graph, aggH in agg+ or player in sub+

    node_, link_, Et, Lay = grapht  # Et and Lay summed from link_
    graph = CG(
        fd=fd, Et=Et, root=root, node_=[],link_=link_, maxL=maxL, nnest=root.nnest, lnest=root.lnest, baseT=copy(node_[0].baseT),
        derTT=Lay.derTT, derH = [Lay] if fd else [[Lay]])  # higher layers are added by feedback, dfork added from comp_link_
    if fd:
        for L in link_:  # fd mfork = link.nodet.root dfork
            for LR in set([n.root for n in L.nodet if n.root]):
                if len(LR.derH[0])==2: LR.derH[0][1].add_lay(Lay)
                else:                  LR.derH[0] += [L.copy_(root=LR)]
                LR.derTT += Lay.derTT
    N_, yx_ = [],[]
    for i, N in enumerate(node_):
        fc = 0
        if minL:  # max,min L.dist in graph.link_, inclusive, = lower-layer exclusive maxL, if G was dist-nested in cluster_N_
            while N.root.maxL and N.root is not graph and (minL != N.root.maxL):  # maxL=0 in edge|frame
                if N.root is graph:
                    fc=1; break  # graph was assigned as root via prior N
                else: N = N.root  # cluster prior-dist graphs vs nodes
        if fc: continue  # N.root was clustered in prior loop
        else: N_ += [N]  # roots if minL
        N.root = graph
        yx_ += [N.yx]
        graph.box = extend_box(graph.box, N.box)  # pre-compute graph.area += N.area?
        graph.Et += N.Et * icoef  # deeper, lower weight
        if i and not fd: graph.baseT += N.baseT
        # not in CL
    graph.node_= N_  # nodes or roots, link_ is still current-dist links only?
    yx = np.mean(yx_, axis=0)
    dy_,dx_ = (graph.yx - yx_).T; dist_ = np.hypot(dy_,dx_)
    graph.aRad = dist_.mean()  # ave distance from graph center to node centers
    graph.yx = yx
    if fd:  # dgraph, no mGs / dG for now  # and val_(Et, _Et=root.Et) > 0:
        altG = []  # mGs overlapping dG
        for L in node_:
            for n in L.nodet:  # map root mG
                mG = n.root
                if isinstance(mG, CG) and mG not in altG:  # root is not frame
                    mG.altG.node_ += [graph]  # cross-comp|sum complete altG before next agg+ cross-comp, multi-layered?
                    altG += [mG]
    return graph

def sum_lay_(link_, root):
    lay0 = CLay(root=root)
    for link in link_:
        lay0.add_lay(link.derH[0]); root.derTTe += link.derH[0].derTT
    return lay0

def comb_H_(L_, root, fd):
    derH = sum_H(L_,root,fd=fd)
    Lay = CLay(root=root)
    for lay in derH:
        Lay.add_lay(lay); root.derTTe += lay.derTT
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
                root.derTTe += lay.derTT; root.Et += lay.Et
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
                        Lay += [fork.copy_(root=root,rev=rev,fc=fc)]
                        root.derTT += fork.derTT; root.Et += fork.Et
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
                        if dLay: dLay.add_lay(dlay)  # sum ds between input forks
                        else:    dLay = dlay
            Et += dLay.Et
            derH += [dLay]
    return derH

def sum_G_(node_, s=1, fc=0, G=None):

    if G is None:
        G = copy_(node_[0]); G.node_ = [node_[0]]; G.link_ = []; node_=node_[1:]
    for n in node_:
        G.node_ += [n]
        if not G.fd:
            G.baseT += n.baseT * s; G.derTTe += n.derTTe * s  # G.Ete?
        G.derTT += n.derTT * s; G.Et += n.Et * s; G.aRad += n.aRad * s; G.yx += n.yx * s
        if n.derH:
            add_H(G.derH, n.derH, root=G, rev=s==-1, fc=fc, fd=G.fd)
        if fc:
            G.M += n.m * s; G.L += s
        else:
            if n.extH: add_H(G.extH, n.extH, root=G, rev = s==-1, fd=1)  # empty in centroid, flat in extH?
            G.box = extend_box( G.box, n.box)  # extended per separate node_ in centroid
    return G

def L2N(link_):
    for L in link_:
         L.fd=1; L.mL_t,L.rimt=[[],[]],[[],[]]; L.aRad=0; L.visited_,L.extH=[],[]; L.baseT=[]; L.derTTe=np.zeros((2,8))
    return link_

def frame2G(G, **kwargs):
    blob2G(G, **kwargs)
    G.derH = kwargs.get('derH', [CLay(root=G, Et=np.zeros(4), derTT=[], node_=[],link_ =[])])
    G.Et = kwargs.get('Et', np.zeros(4))
    G.node_ = kwargs.get('node_', [])
    G.aves = aves

def blob2G(G, **kwargs):
    # node_, Et stays the same:
    G.fd = 0  # fd=1 if cluster Ls|lGs
    G.root = kwargs.get('root')  # may extend to list in cluster_N_, same nodes may be in multiple dist layers
    G.link_ = kwargs.get('link_',[])
    G.nnest = kwargs.get('nnest',0)  # node_H if > 0, node_[-1] is top G_
    G.lnest = kwargs.get('lnest',0)  # link_H if > 0, link_[-1] is top L_
    G.derH = []  # sum from nodes, then append from feedback, maps to node_tree
    G.extH = []  # sum from rims
    G.baseT = []  # I,G,Dy,Dx
    G.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m_,d_ base params
    G.derTTe = kwargs.get('derTTe', np.zeros((2,8)))
    G.box = kwargs.get('box', np.array([np.inf,np.inf,-np.inf,-np.inf]))  # y0,x0,yn,xn
    G.yx = kwargs.get('yx', np.zeros(2))  # init PP.yx = (y+Y)/2,(x+X)/2, then ave node yx
    G.rim = []  # flat links of any rng, may be nested in clustering
    G.maxL = 0  # nesting in nodes
    G.aRad = 0  # average distance between graph center and node center
    G.altG = []  # or altG? adjacent (contour) gap+overlap alt-fork graphs, converted to CG
    return G

def PP2G(PP):
    root, P_, link_, vert, latuple, A, S, box, yx, Et = PP

    baseT = np.array((*latuple[:2], *latuple[-1]))  # I,G,Dy,Dx
    derTT = np.hstack((vert, np.zeros((2,2))))  # [I,G,A, M,D,L] -> [I,G,gA, M,D,n, empty L,A]
    y,x,Y,X = box; dy,dx = Y-y,X-x              # A = (dy,dx); L = np.hypot(dy,dx)
    G = CG(root=root, fd=0, Et=Et, node_=P_, link_=[], baseT=baseT, derTT=derTT, box=box, yx=yx, aRad=np.hypot(dy/2, dx/2),
           derH=[[CLay(node_=P_,link_=link_, derTT=deepcopy(derTT)), CLay()]])  # empty dfork
    return G

if __name__ == "__main__":
   # image_file = './images/raccoon_eye.jpeg'
    image_file = './images/toucan_small.jpg'
    image = imread(image_file)
    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vectorize_root(frame)