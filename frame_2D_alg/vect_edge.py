from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_
from slice_edge import slice_edge, comp_angle, ave_G
from comp_slice import comp_slice, comp_latuple, comp_md_
from itertools import combinations, zip_longest
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
That resembles a neuron, which has dendritic tree as input and axonal tree as output. 
But we have recursively nested param sets packed in each level of the trees, which don't exist in neurons.
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
ave = 3
ave_d = 4
ave_L = 4
max_dist = 2
ave_rn = 1000  # max scope disparity
ccoef = 10  # scaling match ave to clustering ave
icoef = .15  # internal M proj_val / external M proj_val
med_cost = 10

class CLay(CBase):  # flat layer if derivation hierarchy
    name = "lay"
    def __init__(l, root, Et, node_, link_, m_d_t):
        super().__init__()
        l.root = root   # higher node or link
        l.Et = Et
        l.node_ = node_  # concat across fork tree
        l.link_ = link_
        l.m_d_t = m_d_t  # [[mext,mlat,mver],[dext,dlat,dver]], sum across fork tree
        # altL = CLay from comp altG
        # i = kwargs.get('i', 0)  # lay index in root.node_, link_, to revise olp
        # i_ = kwargs.get('i_',[])  # priority indices to compare node H by m | link H by d
        # ni = 0  # exemplar in node_, trace in both directions?
    def __bool__(l): return bool(l.m_d_t)

    def copy_(lay, root=None, rev=0, fc=0, i=None):  # comp direction may be reversed to -1

        if i:  # reuse self
            C = lay; lay = i; C.node_=copy(i.node_); C.link_ = copy(i.link_); C.m_d_t=[]; C.root=root
        else:  # init new C
            C = CLay(root=root, Et=np.zeros(4), node_=copy(lay.node_), link_=copy(lay.link_), m_d_t=[])
        C.Et = lay.Et * -1 if (fc and rev) else copy(lay.Et)

        for fd, tt in enumerate(lay.m_d_t):  # nested array tuples
            C.m_d_t += [tt * -1 if rev and (fd or fc) else deepcopy(tt)]

        if not i: return C

    def add_lay(Lay, lay_, rev=0, fc=0):  # rev = dir==-1, unpack derH trees down to numericals and sum/subtract them
        if not isinstance(lay_,list): lay_ = [lay_]
        for lay in lay_:
            # sum der forks:
            for fd, (F_t, f_t) in enumerate(zip(Lay.m_d_t, lay.m_d_t)):  # m_t and d_t
                for F_,f_ in zip(F_t, f_t):
                    F_ += f_ * -1 if rev and (fd or fc) else f_  # m_|d_ in [dext,dlat,dver]
            # concat node_,link_:
            Lay.node_ += [n for n in lay.node_ if n not in Lay.node_]
            Lay.link_ += lay.link_
            et = lay.Et * -1 if rev and fc else lay.Et
            Lay.Et += et
        return Lay

    def comp_lay(_lay, lay, rn, root, dir=1):  # unpack derH trees down to numericals and compare them

        _d_t, d_t = _lay.m_d_t[1], lay.m_d_t[1]  # norm,comp np.array d_t:
        dd_t = _d_t - d_t * rn * dir
        md_t = np.array([np.minimum(_d_,d_) for _d_,d_ in zip(_d_t, d_t)], dtype=object)
        for i, (_d_,d_) in enumerate(zip(_d_t, d_t)):
            md_t[i][(_d_<0) != (d_<0)] *= -1  # negate if only one of _d_ or d_ is negative
        M = sum([sum(md_) for md_ in md_t])
        D = sum([sum(dd_) for dd_ in dd_t])
        n = .3 if len(d_t)==1 else 2.3  # n comp params / 6 (2/ext, 6/Lat, 6/Ver)

        m_d_t = [np.array(md_t),np.array(dd_t)]
        node_ = list(set(_lay.node_+ lay.node_))  # concat
        link_ = _lay.link_ + lay.link_
        Et = np.array([M, D, n, (_lay.Et[3] + lay.Et[3]) / 2])
        if root: root.Et += Et

        return CLay(Et=Et, root=root, node_=node_, link_=link_, m_d_t=m_d_t)  # no comp_derH yet

    # not used:
    def sort_H(He, fd):  # re-assign olp and form priority indices for comp_tree, if selective and aligned

        i_ = []  # priority indices
        for i, lay in enumerate(sorted(He.node_, key=lambda lay: lay.Et[fd], reverse=True)):
            di = lay.i - i  # lay index in H
            lay.olp += di  # derR- valR
            i_ += [lay.i]
        He.i_ = i_  # H priority indices: node/m | link/d
        if not fd:
            He.root.node_ = He.node_

class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster
    # graph / node
    def __init__(G,  **kwargs):
        super().__init__()
        G.Et = kwargs.get('Et', np.zeros(4))  # sum all param Ets
        G.fd_ = kwargs.get('fd_',[])  # list of forks forming G, 1 if cluster of Ls | lGs
        G.root = kwargs.get('root')  # may extend to list in cluster_N_, same nodes may be in multiple dist layers
        G.derH = kwargs.get('derH',[])  # each layer is Clay(Et, node_, link_, m_d_t), summed|concat from links across fork tree
        G.extH = kwargs.get('extH',[])  # sum from rims
        G.vert = kwargs.get('vert', np.array([np.zeros(6), np.zeros(6)])) # vertical m_d_ of latuple
        G.latuple = kwargs.get('latuple', np.array([.0,.0,.0,.0,.0,np.zeros(2)],dtype=object))  # lateral I,G,M,D,L,[Dy,Dx]
        G.box = kwargs.get('box', np.array([np.inf,np.inf,-np.inf,-np.inf]))  # y0,x0,yn,xn
        G.yx = kwargs.get('yx', np.zeros(2))  # init PP.yx = [(y0+yn)/2,(x0,xn)/2], then ave node yx
        G.maxL = kwargs.get('maxL', 0)  # if dist-nested in cluster_N_
        G.aRad = 0  # average distance between graph center and node center
        G.altG = []  # adjacent (contour) gap+overlap alt-fork graphs, converted to CG
        # G.depth = 0  # n missing higher agg layers
        # G.fork_tree: list = z([[]])  # indices in all layers(forks, if no fback merge
        # G.fback_ = []  # node fb buffer, n in fb[-1]?
        G.node_ = kwargs.get('node_',[])
        G.link_ = kwargs.get('link_',[])  # internal links
        G.rim = kwargs.get('rim',[])  # external links
    def __bool__(G): return bool(G.node_)  # never empty

class CL(CBase):  # link or edge, a product of comparison between two nodes or links
    name = "link"
    def __init__(l,  **kwargs):
        super().__init__()
        l.Et = kwargs.get('Et', np.zeros(4))
        l.fd = kwargs.get('fd',0)
        l.derH = kwargs.get('derH',[])  # list of CLay s
        l.nodet = kwargs.get('nodet',[])  # e_ in kernels, else replaces _node,node: not used in kernels
        l.angle = kwargs.get('angle',[])  # dy,dx between nodet centers
        l.dist = kwargs.get('dist',0)  # distance between nodet centers
        l.box = kwargs.get('box',[])  # sum nodet, not needed?
        l.yx = kwargs.get('yx',[])
        # add med, rimt, extH in der+
    def __bool__(l): return bool(l.nodet)

def vectorize_root(frame):
    # init for agg+:
    blob_ = unpack_blob_(frame)
    frame2CG(frame, derH=[CLay(root=frame, Et=np.zeros(4), m_d_t=[], node_=[], link_=[])], node_=[], root=None)  # distinct from base blob_
    for blob in blob_:
        if not blob.sign and blob.G > ave_G * blob.root.olp:
            edge = slice_edge(blob)
            if edge.G * (len(edge.P_)-1) > ave:  # eval PP
                comp_slice(edge)
                if edge.Et[0] * (len(edge.node_)-1)*(edge.rng+1) > ave:
                    lat = np.array([.0,.0,.0,.0,.0,np.zeros(2)],dtype=object); vert = np.array([np.zeros(6), np.zeros(6)])
                    for PP in edge.node_:
                        vert += PP[3]; lat += PP[4]
                    y_,x_ = zip(*edge.dert_.keys()); box = [min(y_),min(x_),max(y_),max(x_)]
                    blob2CG(edge, root=frame, vert=vert,latuple=lat, box=box, yx=np.divide([edge.latuple[:2]], edge.area))  # node_, Et stay the same
                    G_ = []
                    for PP in edge.node_:  # no comp node_, link_ | PPd_ for now
                        P_,link_,vert,lat, A,S,box,[y,x],Et = PP[1:]  # PPt
                        if Et[0] > ave:  # no altG until cross-comp
                            G = CG(root=edge, fd_=[0], Et=Et, node_=P_, link_=[], vert=vert, latuple=lat, box=box, yx=np.array([y,x]))
                            y0,x0,yn,xn = box; G.aRad = np.hypot((yn-y0)/2,(xn-x0)/2)  # approx
                            G_ += [G]
                    if len(G_) > ave_L:
                        frame.node_ += [edge]; edge.node_ = G_
                        cluster_edge(edge)  # add altG: summed converted adj_blobs of converted edge blob?

def val_(Et, _Et=[], fo=0, coef=1, fd=1):  # compute projected match in mfork or borrowed match in dfork

    m, d, n, o = Et
    if any(_Et):  # get alt fork in root Et
        _m,_d,_n,_o = _Et  # cross-fork induction, same overlap?
        if fd:  # proj diff *= co-match lend
            val = d * (_m / (ave * coef * _n)) - ave_d * coef * n * (o if fo else 1)  # ave *= coef: more specific rel aves
        else:  # proj match -= co-diff borrow
            val = m - (_d - (ave_d * coef * _n)) - ave * coef * n * (o if fo else 1)
            # this diff borrow is not m-specific, same as match lend in fd, can be positive for current m?
    else:
        val = m - ave * coef * n * (o if fo else 1)  # * overlap in cluster eval, not comp eval
    return val

def cluster_edge(edge):  # edge is CG but not a connectivity cluster, just a set of clusters in >ave G blob, unpack by default?

    def cluster_PP_(N_, fd):
        G_,deep_ = [],[]
        while N_:  # flood fill
            node_,link_, et = [],[], np.zeros(4)
            N = N_.pop(); _eN_ = [N]
            while _eN_:
                eN_ = []
                for eN in _eN_:  # rim-connected ext Ns
                    node_ += [eN]
                    for L,_ in zrim(eN, fd):
                        if L not in link_:
                            for eN in L.nodet:  # eval by link.derH.Et + extH.Et * ccoef > ave?
                                if eN in N_:
                                    eN_ += [eN]; N_.remove(eN)  # merged
                            link_ += [L]; et += L.derH[0].Et  # add density term: if L.derH.Et[0]/ave * n.extH m/ave or L.derH.Et[0] + n.extH m*.1?
                _eN_ = {*eN_}
            if val_(et) > 0: G_ += [sum2graph(edge, [node_,link_,et], fd)]
            else:
                for n in node_: deep_ += [n]  # unpack weak Gts
        if fd: edge.node_ = G_+ [deep_]
        else:  edge.link_ = G_+ [deep_]
    # comp PP_:
    N_,L_,Et = comp_node_(edge.node_ if isinstance(edge.node_[-1],CG) else edge.node_[:1])
    edge.link_ += L_
    if val_(Et, fo=1) > 0:  # cancel by borrowing d?
        lay = sum_H(L_,edge)  # mlay
        if len(N_) > ave_L:
            cluster_PP_(N_, fd=0)
        if val_(Et, _Et=Et, fo=1) > 0:  # likely not from the same links
            L2N(L_,edge)  # comp dPP_:
            lN_,lL_,dEt = comp_link_(L_,Et)
            if val_(dEt, fo=1) > 0:
                lay.add_lay(sum_H(lL_,edge))  # mlay += dlay
                if len(lN_) > ave_L:
                    cluster_PP_(lN_, fd=1)
        # one layer / cluster_edge:
        edge.derH += [lay]

def comp_node_(_N_):  # rng+ forms layer of rim and extH per N, appends N_,L_,Et, ~ graph CNN without backprop

    _Gp_ = []  # [G pair + co-positionals]
    for _G, G in combinations(_N_, r=2):  # skip G if list?
        rn = _G.Et[2] / G.Et[2]
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
            # dist vs. radii * induction:
            GV = val_(_G.Et) + val_(G.Et) + sum([val_(l.Et) for l in _G.extH]) + sum([val_(l.Et) for l in _G.extH])
            if dist < max_dist * ((radii * icoef**3) * GV):
                Link = comp_N(_G,G, rn, angle=[dy,dx], dist=dist)
                L_ += [Link]  # include -ve links
                if val_(Link.Et) > 0:
                    N_.update({_G,G}); Et += Link.Et; _G.add,G.add = 1,1
            else:
                Gp_ += [Gp]  # re-evaluate not-compared pairs with one incremented N.M
        ET += Et
        if val_(Et) > 0:  # current-rng vM, -= borrowing d?
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
                    if val_(_L.Et, _Et=iEt) > 0:  # proj val = compared d * rel root M
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
                                    if val_(__L.Et, _Et=Et) > 0:  # compared __L.derH mag * loop induction
                                        mL_.add((__L, rev ^_rev ^__rev))  # combine reversals: 2 * 2 mLs, but 1st 2 are pre-combined
                                        lEt += __L.Et
                if val_(lEt) > 0: # L'rng+, vs L'comp above
                    L.mL_t = mL_t; _L_.add(L); ext_Et += lEt
            # refine eval by extension D:
            if val_(ext_Et, _Et=Et) - med * med_cost > 0:
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

def comp_N(_N,N, rn, angle=None, dist=None, dir=1):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = isinstance(N,CL)  # compare links, relative N direction = 1|-1
    # comp externals:
    if fd:
        _L, L = _N.dist, N.dist;  dL = _L - L*rn; mL = min(_L, L*rn) - ave_L  # direct match
        mA,dA = comp_angle(_N.angle, [d*dir *rn for d in N.angle])  # rev 2nd link in llink
        # comp med if LL: isinstance(>nodet[0],CL), higher-order version of comp dist?
    else:
        _L, L = len(_N.node_),len(N.node_); dL = _L- L*rn; mL = min(_L, L*rn) - ave_L
        mA,dA = comp_area(_N.box, N.box)  # compare area in CG vs angle in CL
    # der ext
    m_t = np.array([mL,mA], dtype=float); d_t = np.array([dL,dA], dtype=float)
    _o,o = _N.Et[3],N.Et[3]; olp = (_o+o) / 2  # inherit from comparands?
    Et = np.array([mL+mA, abs(dL)+abs(dA), .3, olp])  # n = compared vars / 6
    if fd:  # CL
        m_t = np.array([m_t, np.zeros(6), np.zeros(6)], dtype=object)  # empty lat, ver
        d_t = np.array([d_t, np.zeros(6), np.zeros(6)], dtype=object)
    else:   # CG
        (mLat, dLat), L_et = comp_latuple(_N.latuple, N.latuple, _o,o)
        (mVer, dVer), V_et = comp_md_(_N.vert[1], N.vert[1], dir)
        m_t = np.array([m_t, mLat, mVer], dtype=object)
        d_t = np.array([d_t, dLat, dVer], dtype=object)
        Et += np.array([L_et[0]+V_et[0], L_et[1]+V_et[1], 2, 0])
        # same olp?
    Link = CL(fd=fd, nodet=[_N,N], yx=np.add(_N.yx,N.yx)/2, angle=angle, dist=dist, box=extend_box(N.box,_N.box))
    lay0 = CLay(root=Link, Et=Et, m_d_t=[m_t,d_t], node_=_N.node_+N.node_, link_=_N.link_+N.link_)  # remove overlap later
    derH = [_lay.comp_lay(lay,rn, root=Link) for _lay,lay in zip(_N.derH, N.derH)]  # comp shared layers, if any
    for lay in derH: Et += lay.Et
    Link.derH = [lay0, *derH]
    # spec: comp_node_(node_|link_), combinatorial, node_ nested / rng-)agg+?
    if not fd and _N.altG and N.altG:  # if alt M?
        Link.altL = comp_N(_N.altG, N.altG, _N.altG.Et[2] / N.altG.Et[2])
        Et += Link.altL.Et
    Link.Et = Et
    if val_(Et) > 0:
        for rev, node in zip((0,1), (N,_N)):  # reverse Link direction for _N
            if fd: node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            else:  node.rim += [(Link,dir)]
            add_H(node.extH, Link.derH, root=node, rev=rev)
            node.Et += Et
    return Link

def zrim(N,fd): return N.rimt[0] + N.rimt[1] if fd else N.rim  # add nesting in cluster_N_?

def sum2graph(root, grapht, fd, minL=0, maxL=None):  # sum node and link params into graph, aggH in agg+ or player in sub+

    node_, link_, Et = grapht
    graph = CG(fd_=node_[0].fd_+[fd], Et=Et*icoef, root=root, node_=[], link_=link_, maxL=maxL)
    # arg Et is weaker if internal, maxL,minL: max and min L.dist in graph.link_
    yx = np.array([0,0]); yx_ = []
    N_ = []
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
        yx = np.add(yx, N.yx)
        yx_ += [N.yx]
        if isinstance(node_[0],CG):
            graph.latuple += N.latuple; graph.vert += N.vert
        graph.Et += N.Et * icoef ** 2  # deeper, lower weight
        N.root = graph
    graph.node_= N_  # nodes or roots, link_ is still current-dist links only?
    graph.derH = [sum_H(link_,graph)]  # sum, comb link derHs
    L = len(node_)
    yx = np.divide(yx,L)
    dy,dx = np.divide( np.sum([ np.abs(yx-_yx) for _yx in yx_], axis=0), L)
    graph.aRad = np.hypot(dy,dx)  # ave distance from graph center to node centers
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


def sum_H(Q, root, rev=0, fc=0, fmerge=1):  # sum derH in link_|node_

    DerH = [lay.copy_(root=root,rev=rev,fc=fc) for lay in Q[0].derH]
    for e in Q[1:]:
        for Lay, lay in zip_longest(DerH, e.derH, fillvalue=[]):
            if lay:
                if Lay: Lay.add_lay(lay,rev=rev,fc=fc)
                else: DerH += [lay.copy_(root=root,rev=rev,fc=fc)]
    if fmerge:
        Lay = DerH[0].copy_(root=root)
        for lay in DerH[1:]: Lay.add_lay(lay,rev=rev,fc=fc)
        return Lay  # CLay derH
    else:
        return DerH  # list, currently not used

def add_H(H, h, root, rev=0, fc=0):
    for Lay, lay in zip_longest(H, h, fillvalue=[]):
        if lay:
            if Lay: Lay.add_lay(lay,rev=rev,fc=fc)
            else:   H += [lay.copy_(root=root,rev=rev,fc=fc)]
            root.Et += lay.Et

def norm_H(H, n):
    for lay in H:
       for fork in lay.m_d_t: fork *= n  # arrays
       lay.Et *= n  # same node_, link_

def L2N(link_,root):
    for L in link_:
        L.root=root; L.fd_=copy(L.nodet[0].fd_); L.mL_t,L.rimt = [[],[]],[[],[]]; L.aRad=0; L.visited_,L.node_,L.link_,L.extH = [],[],[],[]

def frame2CG(G, **kwargs):
    blob2CG(G, **kwargs)
    G.derH = kwargs.get('derH', [CLay(root=G, Et=np.zeros(4), m_d_t=[], node_=[],link_ =[])])
    G.Et = kwargs.get('Et', np.zeros(4))
    G.node_ = kwargs.get('node_', [])

def blob2CG(G, **kwargs):
    # node_, Et stays the same:
    G.fd_ = []  # fd=1 if cluster of Ls|lGs?
    G.root = kwargs.get('root')  # may extend to list in cluster_N_, same nodes may be in multiple dist layers
    G.link_ = []
    G.derH = []  # sum from nodes, then append from feedback, maps to node_tree
    G.extH = []  # sum from rims
    G.latuple = kwargs.get('latuple', np.array([.0,.0,.0,.0,.0,np.zeros(2)],dtype=object))  # lateral I,G,M,D,L,[Dy,Dx]
    G.vert = kwargs.get('vert', np.array([np.zeros(6), np.zeros(6)]))  # vertical m_d_ of latuple
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