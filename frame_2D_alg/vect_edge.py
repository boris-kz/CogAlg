from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_
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

class CLay(CBase):  # layer of derivation hierarchy
    name = "lay"
    def __init__(l, **kwargs):
        super().__init__()
        l.Et = kwargs.get('Et', np.zeros(4))
        l.root = kwargs.get('root', None)  # higher node or link
        l.node_ = kwargs.get('node_', [])  # concat across fork tree
        l.link_ = kwargs.get('link_', [])
        l.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m_,d_ [M,D,n,o, I,G,A,L], sum across fork tree,
        # add weights for cross-similarity, along with vertical aves, for both m_ and d_?
        # altL = CLay from comp altG
        # i = kwargs.get('i', 0)  # lay index in root.node_, link_, to revise olp
        # i_ = kwargs.get('i_',[])  # priority indices to compare node H by m | link H by d
        # ni = 0  # exemplar in node_, trace in both directions?
    def __bool__(l): return bool(l.node_)

    def copy_(lay, root=None, rev=0, i=None):  # comp direction may be reversed to -1

        if i:  # reuse self
            C = lay; lay = i; C.node_=copy(i.node_); C.link_ = copy(i.link_); C.derTT=np.zeros((2,8)); C.root=root
        else:  # init new C
            C = CLay(root=root, node_=copy(lay.node_), link_=copy(lay.link_))
        C.Et = copy(lay.Et)

        for fd, tt in enumerate(lay.derTT):  # nested array tuples
            C.derTT[fd] += tt * -1 if (rev and fd) else deepcopy(tt)

        if not i: return C

    def add_lay(Lay, lay_, rev=0):  # merge lays, including mlay + dlay

        if not isinstance(lay_,list): lay_ = [lay_]
        for lay in lay_:
            # rev = dir==-1, to sum/subtract numericals in m_ and d_:
            for fd, (F_, f_) in enumerate(zip(Lay.derTT, lay.derTT)):
                F_ += f_ * -1 if (rev and fd ) else f_  # m_|d_
            # concat node_,link_:
            Lay.node_ += [n for n in lay.node_ if n not in Lay.node_]
            Lay.link_ += lay.link_
            Lay.Et += lay.Et
        return Lay

    def comp_lay(_lay, lay, rn, root, dir=1):  # unpack derH trees down to numericals and compare them

        i_ = lay.derTT[1] * rn * dir; _i_ = _lay.derTT[1]  # i_ is ds, scale and direction- normalized
        d_ = _i_ - i_
        a_ = np.abs(i_); _a_ = np.abs(_i_)
        m_ = np.minimum(_a_,a_) / reduce(np.maximum,[_a_,a_,1e-7])  # match = min/max comparands
        m_ *= np.where((_i_<0) != (i_<0), -1,1)  # match is negative if comparands have opposite sign
        derTT = np.array([m_,d_])
        node_ = list(set(_lay.node_+ lay.node_))  # concat
        link_ = _lay.link_ + lay.link_
        M = sum(m_ * w_t[0]); D = sum(d_ * w_t[1])
        Et = np.array([M, D, 8, (_lay.Et[3]+lay.Et[3])/2])  # n compared params = 8
        if root: root.Et += Et

        return CLay(Et=Et, root=root, node_=node_, link_=link_, derTT=derTT)

class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster
    # graph / node
    def __init__(G,  **kwargs):
        super().__init__()
        G.fi = kwargs.get('fi',0)  # or fd_: list of forks forming G, 1 if cluster of Ls | lGs, for feedback only?
        G.root = kwargs.get('root')  # may extend to list in cluster_N_, same nodes may be in multiple dist layers
        G.Et = kwargs.get('Et', np.zeros(4))  # sum all params M,D,n,o
        G.yx = kwargs.get('yx', np.zeros(2))  # init PP.yx = [(y+Y)/2,(x,X)/2], then ave node yx
        G.box = kwargs.get('box', np.array([np.inf,np.inf,-np.inf,-np.inf]))  # y,x,Y,X area: (Y-y)*(X-x)
        G.baseT = kwargs.get('baseT', np.zeros(4))  # I,G,Dy,Dx  # from slice_edge
        G.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m,d / Et,baseT: [M,D,n,o, I,G,A,L], summed across derH lay forks
        G.derTTe = kwargs.get('derTTe', np.zeros((2,8)))  # sum across link.derHs
        G.derH = kwargs.get('derH',[])  # each lay is [m,d]: Clay(Et,node_,link_,derTT), sum|concat links across fork tree
        G.extH = kwargs.get('extH',[])  # sum from rims, single-fork
        G.maxL = kwargs.get('maxL', 0)  # if dist-nested in cluster_N_
        G.aRad = 0  # average distance between graph center and node center
        # alt.altG is empty for now, needs to be more selective
        G.altG = CG(altG=[], fi=0) if kwargs.get('altG') is None else kwargs.get('altG')  # adjacent (contour) gap+overlap alt-fork graphs, converted to CG
        # G.fork_tree: list = z([[]])  # indices in all layers(forks, if no fback merge
        # G.fback_ = []  # node fb buffer, n in fb[-1]
        G.nnest = kwargs.get('nnest',0)  # node_H if > 0, node_[-1] is top G_
        G.lnest = kwargs.get('lnest',0)  # link_H if > 0, link_[-1] is top L_
        G.node_ = kwargs.get('node_',[])
        G.link_ = kwargs.get('link_',[])  # internal links
        G.rim = kwargs.get('rim',[])  # external links
        G.nrim = kwargs.get('nrim',[])
    def __bool__(G): return bool(G.node_)  # never empty

def copy_(N):
    C = CG()
    for name, value in N.__dict__.items():
        val = getattr(N, name)
        if name == '_id' or name == "Ct_": continue  # skip id and Ct_
        elif name == 'derH':
            for lay in N.derH:
                if N.fi: C.derH +=[[fork.copy_(root=C) for fork in lay]]
                else:    C.derH +=[lay.copy_(root=C)]  # CLay
        elif name == 'extH':
            C.extH = [lay.copy_(root=C) for lay in N.extH]
        elif isinstance(value,list) or isinstance(value,np.ndarray):
            setattr(C, name, copy(val))  # Et,yx,box, node_,link_,rim_, altG, baseT, derTT
        else:
            setattr(C, name, val)  # numeric or object: fi, root, maxL, aRad, nnest, lnest
    return C

class CL(CBase):  # link or edge, a product of comparison between two nodes or links
    name = "link"
    def __init__(l,  **kwargs):
        super().__init__()
        l.nodet = kwargs.get('nodet',[])  # e_ in kernels, else replaces _node,node: not used in kernels
        l.L = kwargs.get('L',0)  # distance between nodes
        l.Et = kwargs.get('Et', np.zeros(4))
        l.fi = kwargs.get('fi',0)
        l.yx = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet
        l.box = kwargs.get('box', np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        l.baseT = kwargs.get('baseT', np.zeros(4))
        l.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m_,d_ [M,D,n,o, I,G,A,L], sum across derH
        l.derH  = kwargs.get('derH', [])  # list of single-fork CLays
        # add med, rimt, extH in der+
    def __bool__(l): return bool(l.nodet)

ave, avd, arn, aI, aveB, ave_L, max_dist, icoef = 10, 10, 1.2, 100, 100, 5, 10, 2  # opportunity costs
wM, wD, wN, wO, wI, wG, wA, wL = 10, 10, 20, 20, 1, 1, 20, 20  # der params higher-scope weights = reversed relative estimated ave?
w_t = np.ones((2,8))  # fb weights per derTT, adjust in agg+

def vect_root(frame, rV=1, ww_t=[]):  # init for agg+:
    if np.any(ww_t):
        global ave, avd, arn, aveB, ave_L, max_dist, icoef, wM, wD, wN, wO, wI, wG, wA, wL, w_t
        ave, avd, arn, aveB, ave_L, max_dist, icoef = np.array([ave,avd,arn,aveB,ave_L,max_dist,icoef]) / rV  # projected value change
        w_t = np.array( [np.array([wM,wD,wN,wO,wI,wG,wA,wL]), np.array([wM,wD,wN,wO,wI,wG,wA,wL])]) * ww_t  # or dw_ ~= w_/ 2?
        # derTT w_
    blob_ = unpack_blob_(frame)
    frame2G(frame, derH=[CLay(root=frame)], node_=[blob_], root=None)
    edge_ = []  # cluster, unpack
    for blob in blob_:
        if not blob.sign and blob.G > aveB * blob.root.olp:
            edge = slice_edge(blob, rV)
            if edge.G * (len(edge.P_)-1) > ave:  # eval PP
                comp_slice(edge, rV, np.array([(*ww_t[0][:2],*ww_t[0][4:]),(*ww_t[0][:2],*ww_t[1][4:])]) if ww_t else [])  # to scale vert
                if edge.Et[0] * (len(edge.node_)-1)*(edge.rng+1) > ave:
                    G_ = [PP2G(PP)for PP in edge.node_ if PP[-1][0] > ave]  # Et, no altGs
                    if len(G_) > ave_L:  # no comp node_,link_,PPd_
                        edge_ += [cluster_edge(G_, frame)]  # 1layer derH, alt: converted adj_blobs of edge blob | alt_P_?
    # unpack edges:
    Lay = [CLay(root=frame), CLay(root=frame)]
    PP_,G_ = [],[]
    for edget in edge_:
        if edget:
            pp_,g_,lay = edget
            [F.add_lay(f) for F,f in zip(Lay,lay)]  # [mfork, dfork]
            PP_+= pp_; G_+= g_
    frame.derH = [Lay]
    frame.node_= [PP_]
    if G_:
        frame.node_ += [sum_G_(G_)]
        frame.nnest = 1
    frame.baseT = np.sum([G.baseT for G in PP_ + G_], axis=0)
    frame.derTT = np.sum([G.derTT for G in PP_ + G_], axis=0)

    rave = []  # convert rave for comp_slice
    return frame, rave

def cluster_edge(iG_, frame):  # edge is CG but not a connectivity cluster, just a set of clusters in >ave G blob, unpack by default

    def cluster_PP_(N_, fi):
        G_ = []
        while N_:  # flood fill
            node_,link_, et = [],[], np.zeros(4)
            N = N_.pop(); _eN_ = [N]
            while _eN_:
                eN_ = []
                for eN in _eN_:  # rim-connected ext Ns
                    node_ += [eN]
                    for L,_ in get_rim(eN, fi):  # all +ve, * density: if L.Et[0]/ave_d * sum([n.extH.m * ccoef / ave for n in L.nodet])?
                        if L not in link_:
                            for eN in L.nodet:
                                if eN in N_:
                                    eN_ += [eN]; N_.remove(eN)  # merged
                            link_ += [L]; et += L.Et
                _eN_ = {*eN_}
            if Val_(et, et, ave*2, fi=fi) > 0:
                Lay = CLay(); [Lay.add_lay(link.derH[0]) for link in link_]  # single-lay derH
                G_ += [sum2graph(frame, [node_,link_,et, Lay], fi)]
        return G_

    N_,L_,Et = comp_node_(iG_, ave)  # comp PP_
    # mval -> lay:
    if Val_(Et, Et, ave, fi=1) > 0:
        lay = [sum_lay_(L_, frame)]  # [mfork]
        G_ = cluster_PP_(copy(N_), fi=1) if len(N_) > ave_L else []
        '''
        # dval -> comp dPP_:
        if Val_(Et, Et, ave*2, fd=1) > 0:  # likely not from the same links
            lN_,lL_,dEt = comp_link_(L2N(L_), ave*2)
            if Val_(dEt, Et, ave*3, fd=1) > 0:
                lay += [sum_lay_(lL_, frame)]  # dfork
                lG_ = cluster_PP_(lN_, fd=1) if len(lN_) > ave_L else []
            else:
                lG_=[]  # empty dfork
        else: lG_=[]
        '''
        return [N_,G_,lay]

def val_(Et, ave, coef=1):  # comparison / inclusion eval by m only, no contextual projection

    m, d, n, _ = Et  # skip overlap
    return m - ave * coef * n  # coef per compared param type

def Val_(Et, _Et, ave, coef=1, fi=1):  # m|d cluster|batch eval, + cross|root _Et projection

    m, d, n, o = Et; _m,_d,_n,_o = _Et  # cross-fork induction of root Et alt, same overlap?

    d_loc = d * (_m - ave * coef * (_n/n))  # diff * co-projected m deviation, no bilateral deviation?
    d_ave = d - avd * ave  # d deviation, ave_d is always relative to ave m

    if fi: val = m + d_ave - d_loc  # match + proj surround val - blocking val, * decay?
    else:  val = d_ave + d_loc  # diff borrow val, generic + specific

    return val - ave * coef * n * o

def comp_node_(_N_, ave, L=0):  # rng+ forms layer of rim and extH per N, appends N_,L_,Et, ~ graph CNN without backprop

    _Gp_ = []  # [G pair + co-positionals], for top-nested Ns, unless cross-nesting comp:
    if L: _N_ = filter(lambda N: len(N.derH)==L, _N_)  # if dist-nested
    for _G, G in combinations(_N_, r=2):  # if max len derH in agg+
        _n, n = _G.Et[2], G.Et[2]; rn = _n/n if _n>n else n/_n
        if rn > arn:  # scope disparity or _G.depth != G.depth, not needed?
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
            if dist < max_dist * ((radii * icoef**3) * (val_(_G.Et,ave)+ val_(G.Et,ave))):  # ext V is not complete
                Link = comp_N(_G,G, ave, fi=1, angle=[dy,dx], dist=dist)
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

def extend_box(_box, box):  # extend box with another box
    y0, x0, yn, xn = box; _y0, _x0, _yn, _xn = _box

    return min(y0, _y0), min(x0, _x0), max(yn, _yn), max(xn, _xn)

def base_comp(_N, N, dir=1):  # comp Et, Box, baseT, derTT
    # comp Et:
    _M,_D,_n,_o = _N.Et; M,D,n,o = N.Et
    dn = _n - n; mn = min(_n,n) / max(_n,n)  # or multiplicative for ratios: min * rn?
    rn = _n / n
    o*=rn; do = _o - o; mo = min(_o,o) / max(_o,o)
    M*=rn; dM = _M - M; mM = min(_M,M) / max(_M,M)
    D*=rn; dD = _D - D; mD = min(_D,D) / max(_D,D)
    # comp baseT:
    _I,_G,_Dy,_Dx = _N.baseT; I,G,Dy,Dx = N.baseT  # I, G|D, angle
    I*=rn; dI = _I - I; mI = abs(dI) / aI
    G*=rn; dG = _G - G; mG = min(_G,G) / max(_G,G)
    mA, dA = comp_angle((_Dy,_Dx),(Dy*rn,Dx*rn))
    if isinstance(N, CL):  # dimension is distance
        _L,L = _N.L, N.L   # not cumulative
        mL,dL = min(_L,L)/ max(_L,L), _L - L
    else:  # dimension is box area
        _y0,_x0,_yn,_xn =_N.box; _A = (_yn-_y0) * (_xn-_x0)
        y0, x0, yn, xn = N.box;   A = (yn - y0) * (xn - x0)
        mL, dL = min(_A,A)/ max(_A,A), _A - A
        # mA, dA
    _m_,_d_ = np.array([mM,mD,mn,mo,mI,mG,mA,mL]), np.array([dM,dD,dn,do,dI,dG,dA,dL])
    # comp derTT:
    _i_ = _N.derTT[1]; i_ = N.derTT[1] * rn  # normalize by compared accum span
    d_ = (_i_ - i_ * dir)  # np.arrays
    _a_,a_ = np.abs(_i_),np.abs(i_)
    m_ = np.divide( np.minimum(_a_,a_), reduce(np.maximum, [_a_,a_,1e-7]))  # rms
    m_ *= np.where((_i_<0) != (i_<0), -1,1)  # match is negative if comparands have opposite sign

    # each [M,D,n,o, I,G,A,L]:
    return [m_+_m_, d_+_d_], rn

def comp_N(_N,N, ave, fi, angle=None, dist=None, dir=1, fshort=1):  # compare links, relative N direction = 1|-1, no need for angle, dist?
    dderH = []

    [m_,d_], rn = base_comp(_N, N, dir)
    baseT = np.array([(_N.baseT[0]+N.baseT[0])/2, (_N.baseT[1]+N.baseT[1])/2, *angle])  # link M,D,A
    derTT = np.array([m_, d_])
    M = np.sum(m_* w_t[0]); D = np.sum(np.abs(d_* w_t[1]))  # feedback-weighted sum
    Et = np.array([M,D, 8, (_N.Et[3]+N.Et[3]) /2])  # n comp vars, inherited olp
    _y,_x = _N.yx
    y, x = N.yx
    Link = CL(nodet=[_N,N], baseT=baseT, derTT=derTT, yx=np.add(_N.yx,N.yx)/2, L=dist, box=np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)]))
    # spec / lay:
    if fshort and M > ave and (len(N.derH) > 2 or isinstance(N,CL)):  # else derH is redundant to dext,vert
        dderH = comp_H(_N.derH, N.derH, rn, Link, Et, fi)  # comp shared layers, if any
        # spec/ comp_node_(node_|link_)
    Link.derH = [CLay(root=Link,Et=Et,node_=[_N,N],link_=[Link], derTT=copy(derTT)), *dderH]
    for lay in dderH: derTT += lay.derTT
    # spec / alt:
    if fi and _N.altG and N.altG:
        et = _N.altG.Et + N.altG.Et  # comb val
        if Val_(et, et, ave*2, fi=0) > 0:  # eval Ds
            Link.altL = comp_N(_N.altG, N.altG, ave*2, fi=0)
            Et += Link.altL.Et
    Link.Et = Et
    if val_(Et, ave) > 0:
        for rev, node in zip((0,1), (N,_N)):  # reverse Link direction for
            if fi: node.rim += [(Link,dir)]
            else:  node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            add_H(node.extH, Link.derH, root=node, rev=rev, fi=0)
            node.Et += Et
    return Link

def get_rim(N,fi): return N.rim if fi else N.rimt[0] + N.rimt[1]  # add nesting in cluster_N_?

def sum2graph(root, grapht, fi, minL=0, maxL=None):  # sum node and link params into graph, aggH in agg+ or player in sub+

    node_, link_, Et, Lay = grapht  # Et and Lay are summed from link_
    graph = CG(
        fi=fi, Et=Et, root=root, node_=[],link_=link_, maxL=maxL, nnest=root.nnest, lnest=root.lnest, baseT=copy(node_[0].baseT),
        derTT=Lay.derTT, derH = [[Lay]] if fi else [Lay])  # higher layers are added by feedback, dfork added from comp_link_:
    for L in link_:
        L.root = graph  # reassign when L is node
        if not fi:  # add mfork as link.nodet(CL).root dfork
            LR_ = set([n.root for n in L.nodet if isinstance(n.root, CG)])  # skip frame, empty roots
            if LR_:
                lay = reduce(lambda Lay, lay: Lay.add_lay(lay), L.derH, CLay())  # combine lL.derH
                for LR in LR_:  # lay0+= dfork
                    if len(LR.derH[0])==2: LR.derH[0][1].add_lay(lay)  # direct root only
                    else:                  LR.derH[0] += [lay.copy_(root=LR)]
                    LR.derTT += lay.derTT
    N_, yx_ = [],[]
    for i, N in enumerate(node_):
        fc = 0
        if minL:  # max,min L.dist in graph.link_, inclusive, = lower-layer exclusive maxL, if G was dist-nested in cluster_N_
            while N.root.maxL and N.root is not graph and (minL != N.root.maxL):  # maxL=0 in edge|frame, not fd
                if N.root is graph:
                    fc=1; break  # graph was assigned as root via prior N
                else: N = N.root  # cluster prior-dist graphs vs nodes
        if fc: continue  # N.root was clustered in prior loop
        else: N_ += [N]  # roots if minL
        N.root = graph
        yx_ += [N.yx]
        graph.box = extend_box(graph.box, N.box)
        graph.Et += N.Et * icoef  # deeper, lower weight
        if i: graph.baseT += N.baseT
        # not in CL
    graph.node_= N_  # nodes or roots, link_ is still current-dist links only?
    yx = np.mean(yx_, axis=0)
    dy_,dx_ = (graph.yx - yx_).T; dist_ = np.hypot(dy_,dx_)
    graph.aRad = dist_.mean()  # ave distance from graph center to node centers
    graph.yx = yx
    if not fi:  # dgraph, no mGs / dG for now  # and val_(Et, _Et=root.Et) > 0:
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

def comb_H_(L_, root, fi):
    derH = sum_H(L_,root,fi=fi)
    Lay = CLay(root=root)
    for lay in derH:
        Lay.add_lay(lay); root.derTTe += lay.derTT
    return Lay

def sum_H(Q, root, rev=0, fi=1):  # sum derH in link_|node_
    DerH = []
    for e in Q: add_H(DerH, e.derH, root, rev, fi)
    return DerH

def add_H(H, h, root, rev=0, fi=1):  # add fork L.derHs

    for Lay,lay in zip_longest(H,h):  # different len if lay-selective comp
        if lay:
            if fi:  # two-fork lays
                if Lay:
                    for Fork,fork in zip_longest(Lay,lay):
                        if fork:
                            if Fork: Fork.add_lay(fork,rev=rev)
                            else:    Lay += [fork.copy_(root=root)]
                else:
                    Lay = []
                    for fork in lay:
                        Lay += [fork.copy_(root=root,rev=rev)]
                        root.derTT += fork.derTT; root.Et += fork.Et
                    H += [Lay]

            else:  # one-fork lays
                if Lay: Lay.add_lay(lay,rev=rev)
                else:   H += [lay.copy_(root=root,rev=rev)]
                root.derTTe += lay.derTT; root.Et += lay.Et

def comp_H(H,h, rn, root, Et, fi):  # one-fork derH if not fi, else two-fork derH

    derH = []
    for _lay,lay in zip_longest(H,h):  # different len if lay-selective comp
        if _lay and lay:
            if fi:  # two-fork lays
                dLay = []
                for _fork,fork in zip_longest(_lay,lay):
                    if _fork and fork:
                        dlay = _fork.comp_lay(fork, rn,root=root)
                        if dLay: dLay.add_lay(dlay)  # sum ds between input forks
                        else:    dLay = dlay
            else:  # one-fork lays
                 dLay = _lay.comp_lay(lay, rn, root=root)
            Et += dLay.Et
            derH += [dLay]
    return derH

def sum_G_(node_, G=None):

    if G is None:
        G = copy_(node_[0]); G.node_ = [node_[0]]; G.link_ = []; node_=node_[1:]
    for n in node_:
        if n not in G.node_: G.node_ += [n]  # prevent packing same n in alts
        G.baseT += n.baseT; G.derTT += n.derTT; G.Et += n.Et;  G.yx += n.yx
        if G.fi:
            G.derTTe += n.derTTe; G.aRad += n.aRad
            if n.extH: add_H(G.extH, n.extH, root=G, fi=0)
            # empty in centroid?
        if n.derH: add_H(G.derH, n.derH, root=G, fi=G.fi)
        G.box = extend_box( G.box, n.box)  # extended per separate node_ in centroid
    return G

def L2N(link_):
    for L in link_:
        L.fi=0; L.mL_t,L.rimt=[[],[]],[[],[]]; L.aRad=0; L.visited_,L.extH=[],[]; L.derTTe=np.zeros((2,8))
        if not hasattr(L,'root'): L.root=[]
    return link_

def frame2G(G, **kwargs):
    blob2G(G, **kwargs)
    G.derH = kwargs.get('derH', [CLay(root=G, Et=np.zeros(4), derTT=[], node_=[],link_ =[])])
    G.Et = kwargs.get('Et', np.zeros(4))
    G.node_ = kwargs.get('node_', [])

def blob2G(G, **kwargs):
    # node_, Et stays the same:
    G.fi = 1  # fi=0 if cluster Ls|lGs
    G.root = kwargs.get('root')  # may extend to list in cluster_N_, same nodes may be in multiple dist layers
    G.link_ = kwargs.get('link_',[])
    G.nnest = kwargs.get('nnest',0)  # node_H if > 0, node_[-1] is top G_
    G.lnest = kwargs.get('lnest',0)  # link_H if > 0, link_[-1] is top L_
    G.derH = []  # sum from nodes, then append from feedback, maps to node_tree
    G.extH = []  # sum from rims
    G.baseT = np.zeros(4)  # I,G,Dy,Dx
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
    [mM,mD,mI,mG,mA,mL], [dM,dD,dI,dG,dA,dL] = vert
    derTT = np.array([np.array([mM,mD,mL,0,mI,mG,mA,mL]), np.array([dM,dD,dL,0,dI,dG,dA,dL])])
    y,x,Y,X = box; dy,dx = Y-y,X-x
    # A = (dy,dx); L = np.hypot(dy,dx)
    G = CG(root=root, fi=1, Et=Et, node_=P_, link_=[], baseT=baseT, derTT=derTT, box=box, yx=yx, aRad=np.hypot(dy/2, dx/2),
           derH=[[CLay(node_=P_,link_=link_, derTT=deepcopy(derTT)), CLay()]])  # empty dfork
    return G

if __name__ == "__main__":
   # image_file = './images/raccoon_eye.jpeg'
    image_file = './images/toucan_small.jpg'
    image = imread(image_file)
    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vect_root(frame)