import sys
sys.path.append("..")
from frame_blobs import CBase, frame_blobs_root, intra_blob_root, imread, unpack_blob_
from slice_edge import slice_edge, comp_angle, ave_G
from comp_slice import comp_slice, comp_latuple, comp_md_
from itertools import combinations, zip_longest
from functools import reduce
from copy import deepcopy, copy
import numpy as np

'''
This code is initially for clustering segments within edge: high-gradient blob, but too complex for that.
It's mostly a prototype for open-ended compositional recursion: clustering blobs, graphs of blobs, etc.
-
rng+ fork: incremental-range cross-comp nodes, cluster edge segments, initially PPs, that match over < max distance. 
der+ fork: incremental-derivation cross-comp links from node cross-comp, if abs_diff * rel_match > ave 
(variance patterns borrow value from co-projected match patterns because their projections cancel-out)
- 
Thus graphs should be assigned adjacent alt-fork (der+ to rng+) graphs, to which they lend predictive value.
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

class CH(CBase):  # generic derivation hierarchy of variable nesting: extH | derH, their layers and sub-layers

    name = "H"
    def __init__(He, layt=None, Et=None, node_=None, root=None, i=None, i_=None, altH=None):
        super().__init__()
        # He.n = n  # total number of params compared to form derH, to normalize in next comp
        He.layt = [] if layt is None else layt  # nested CH forks, each mediates its own layt, or md_tC in top layer
        He.Et = np.zeros(4) if Et is None else Et  # += links (n is added to et now)
        He.node_ = [] if node_ is None else node_  # concat bottom nesting order if CG, may be redundant to G.node_
        He.root = None if root is None else root  # N or higher-composition He
        He.i = 0 if i is None else i  # lay index in root.H, to revise olp
        He.i_ = [] if i_ is None else i_  # priority indices to compare node H by m | link H by d
        He.altH = CH(altH=object) if altH is None else altH   # summed altLays, prevent cyclic
        He.depth = 0  # max depth of fork tree?
        # He.fd = 0 if fd is None else fd  # 0: sum CGs, 1: sum CLs
        # He.ni = 0  # exemplar in node_, trace in both directions?
        # He.deep = 0 if deep is None else deep  # nesting in root H
        # He.nest = 0 if nest is None else nest  # nesting in H

    def __bool__(H): return bool(H.Et[3] != 0)

    def copy_md_C(he, root, dir=1, fc=0):  # dir is sign if called from centroid, which doesn't use dir

        md_t = [np.array([m_ * dir if fc else copy(m_), d_ * dir]) for m_, d_ in he.H]
        # m_ * dir only if called from centroid()
        Et = he.Et * dir if fc else copy(he.Et)

        return CH(root=root, H=md_t, node_=copy(he.node_), Et=Et, i=he.i, i_=he.i_)

    def copy_(He, root, dir=1, fc=0):  # comp direction may be reversed to -1

        C = CH(root=root, node_=copy(He.node_), Et=copy(He.Et), i=He.i, i_=He.i_)
        for he in He.H:
            C.H += [he.copy_(root=C, dir=dir, fc=fc) if isinstance(he.H[0],CH) else
                    he.copy_md_C(root=C, dir=dir, fc=fc)]
        return C

    def add_md_C(Lay, lay, dir=1, fc=0):

        for Md_, md_ in zip(Lay.H, lay.H):  # [mdext, ?vert, mdVer]
            Md_ += np.array([md_[0]*dir if fc else md_[0].copy(), md_[1]*dir])

        Lay.Et += lay.Et * dir if fc else copy(lay.Et)

    def add_H(HE, He_, dir=1, fc=0):  # unpack derHs down to numericals and sum them, may subtract from centroid
        if not isinstance(He_,list): He_ = [He_]

        for He in He_:
            for Lay, lay in zip_longest(HE.H, He.H, fillvalue=None):
                if lay:
                    if Lay:  # unpack|add, same nesting in both lays
                        Lay.add_H(lay,dir,fc) if isinstance(lay.H[0],CH) else Lay.add_md_C(lay,dir,fc)
                    elif Lay is None:
                        HE.append_( lay.copy_(root=HE, dir=dir, fc=fc) if isinstance(lay.H[0],CH) else lay.copy_md_C(root=HE, dir=dir, fc=fc))
                elif lay is not None and Lay is None:
                    HE.H += [CH()]

            HE.node_ += [node for node in He.node_ if node not in HE.node_]  # empty in CL derH?
            HE.Et += He.Et * dir

        return HE  # root should be updated by returned HE

    def append_(HE,He, flat=0):

        if flat:
            for i, lay in enumerate(He.H):
                if lay:
                    lay = lay.copy_(root=HE) if isinstance(lay.H[0],CH) else lay.copy_md_C(root=HE)
                    lay.i = len(HE.H) + i
                HE.H += [lay]  # lay may be empty to trace forks
        else:
            He.i = len(HE.H); He.root = HE; HE.H += [He]  # He can't be empty
            HE.H += [He]
        HE.Et += He.Et
        return HE

    def comp_md_C(_md_C, md_C, rn, root, olp=1., dir=1):

        der_md_t = []
        Et = np.zeros(2)
        for _md_, md_ in zip(_md_C.H, md_C.H):  # [mdext, ?vert, mdvert]
            # comp ds:
            der_md_, et = comp_md_(_md_[1], md_[1], rn, dir=dir)
            der_md_t += [der_md_]
            Et += et

        return CH(root=root, H=der_md_t, Et=np.append(Et,[olp, .3 if len(der_md_t)==1 else 2.3]))  # .3 in default comp ext)

    def comp_H(_He, He, rn, root):

        derH = CH(root=root)  # derH.H ( flat lay.H, or nest per higher lay.H for selective access?
        # lay.H maps to higher Hs it was derived from, len lay.H = 2 ^ lay_depth (unpacked root H[:i])

        for _lay, lay in zip(_He.H, He.H):  # both may be empty CH to trace fork types
            if _lay and lay:  # same depth
                if isinstance(lay.H[0], CH):
                    dLay = _lay.comp_H(lay, rn, root=derH) # deeper unpack -> comp_md_t
                else:
                    dLay = _lay.comp_md_C(lay, rn=rn, root=derH, olp=(_He.Et[3]+He.Et[3]) /2)  # comp shared layers, add n to olp?
                derH.append_(dLay)
            elif _lay or lay:
                derH.append_(CH())
        return derH

    def norm_(He, n):

        for lay in He.H:   # not empty list
            if lay:
                if isinstance(lay.H[0], CH):
                    lay.norm_C(n)
                else:
                    for md_ in lay.H: md_ *= n
                    lay.Et *= n
        He.Et *= n

    # not updated:
    def sort_H(He, fd):  # re-assign olp and form priority indices for comp_H, if selective and aligned

        i_ = []  # priority indices
        for i, lay in enumerate(sorted(He.H, key=lambda lay: lay.Et[fd], reverse=True)):
            di = lay.i - i  # lay index in H
            lay.olp += di  # derR- valR
            i_ += [lay.i]
        He.i_ = i_  # comp_H priority indices: node/m | link/d
        if not fd:
            He.root.node_ = He.H[i_[0]].node_  # no He.node_ in CL?


class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster

    def __init__(G, fd=0, rng=1, root=[], node_=[], link_=[], subG_=[], subL_=[],
                 Et=None, latuple=None, vert=None, derH=None, extH=None, altG=None, box=None, yx=None):
        super().__init__()
        G.fd = fd  # 1 if cluster of Ls | lGs?
        G.Et = np.zeros(4) if Et is None else Et  # sum all param Ets
        G.rng = rng
        G.root = root  # may extend to list in cluster_N_, same nodes may be in multiple dist layers
        G.node_ = node_  # convert to GG_ or node_H in agg++
        G.link_ = link_  # internal links per comp layer in rng+, convert to LG_ in agg++
        G.subG_ = subG_  # selectively clustered node_
        G.subL_ = subL_  # selectively clustered link_
        G.latuple = np.array([.0,.0,.0,.0,.0,np.zeros(2)],dtype=object) if latuple is None else latuple  # lateral I,G,M,D,L,[Dy,Dx]
        G.vert = np.array([np.zeros(6), np.zeros(6)]) if vert is None else vert  # vertical md_ of latuple
        # maps to node_H / agg+|sub+:
        G.derH = CH() if derH is None else derH  # sum from nodes, then append from feedback
        G.extH = CH() if extH is None else extH  # sum from rim_ elays, H maybe deleted
        G.rim = []  # flat links of any rng, may be nested in clustering
        G.aRad = 0  # average distance between graph center and node center
        G.box = [np.inf, np.inf, -np.inf, -np.inf] if box is None else box  # y0,x0,yn,xn
        G.yx = np.array([0,0]) if yx is None else yx  # init PP.yx = [(y0+yn)/2,(x0,xn)/2], then ave node yx
        G.altG = CG(altG=G) if altG is None else altG  # adjacent gap+overlap graphs, vs. contour in frame_graphs (prevent cyclic)
        # G.fback_ = []  # always from CGs with fork merging, no dderHm_, dderHd_
        # id_H: list = z([[]])  # indices in all layers(forks, if no fback merge
        # depth: int = 0  # n sub_G levels over base node_, max across forks
        # nval: int = 0  # of open links: base alt rep
    def __bool__(G): return bool(G.Et[3] != 0)  # to test empty

class CL(CBase):  # link or edge, a product of comparison between two nodes or links
    name = "link"

    def __init__(l, nodet=None, dist=None, derH=None, angle=None, box=None, H_=None, yx=None):
        super().__init__()
        # CL = binary tree of Gs, depth+/der+: CL nodet is 2 Gs, CL + CLs in nodet is 4 Gs, etc., unpack sequentially
        l.derH = CH(root=l) if derH is None else derH
        l.nodet = [] if nodet is None else nodet  # e_ in kernels, else replaces _node,node: not used in kernels
        l.angle = np.zeros(2) if angle is None else angle  # dy,dx between nodet centers
        l.dist = 0 if dist is None else dist  # distance between nodet centers
        l.box = [] if box is None else box  # sum nodet, not needed?
        l.yx = [0,0] if yx is None else yx
        l.H_ = [] if H_ is None else H_  # if agg++| sub++?
        # add med, rimt, elay | extH in der+
    def __bool__(l): return bool(l.derH.H)

def vectorize_root(frame):

    blob_ = unpack_blob_(frame)
    for blob in blob_:
        if not blob.sign and blob.G > ave_G * blob.root.olp:
            blob = slice_edge(blob)
            if blob.G * (len(blob.P_)-1) > ave:  # eval PP
                comp_slice(blob)
                if blob.Et[0] * (len(blob.node_)-1)*(blob.rng+1) > ave:
                    # init for agg+:
                    if not hasattr(frame, 'derH'):
                        frame.derH = CH(root=frame); frame.root = None; frame.subG_ = []
                    Y,X,_,_,_,_ = blob.latuple
                    lat = np.array([.0,.0,.0,.0,.0,np.zeros(2)],dtype=object); vert = np.array([np.zeros(6), np.zeros(6)])
                    for PP in blob.node_:
                        vert += PP[3]; lat += PP[4]
                    edge = CG(root=frame, node_=blob.node_, vert=vert,latuple=lat, box=[Y,X,0,0],yx=[Y/2,X/2], Et=blob.Et)
                    G_ = []
                    for N in edge.node_:  # no comp node_, link_ | PPd_ for now
                        P_, link_, vert, lat, A, S, box, [y,x], Et = N[1:]  # PPt
                        if Et[0] > ave:   # no altG until cross-comp
                            PP = CG(fd=0, Et=Et,root=edge, node_=P_,link_=link_, vert=vert,latuple=lat, box=box,yx=[y,x])
                            y0,x0,yn,xn = box
                            PP.aRad = np.hypot(*np.subtract(PP.yx,(yn,xn)))
                            G_ += [PP]
                    edge.subG_ = G_
                    if len(G_) > ave_L:
                        cluster_edge(edge); frame.subG_ += [edge]; frame.derH.add_H(edge.derH)
                        # add altG: summed converted adj_blobs of converted edge blob
                        # if len(edge.subG_) > ave_L: agg_recursion(edge)  # unlikely

def cluster_edge(edge):  # edge is CG but not a connectivity cluster, just a set of clusters in >ave G blob, unpack by default?

    def cluster_PP_(edge, fd):
        Gt_ = []
        N_ = copy(edge.link_ if fd else edge.subG_)
        while N_:  # flood fill
            node_,link_, et = [],[], np.zeros(4)
            N = N_.pop(); _eN_ = [N]
            while _eN_:
                eN_ = []
                for eN in _eN_:  # rim-connected ext Ns
                    node_ += [eN]
                    for L,_ in get_rim(eN, fd):
                        if L not in link_:
                            for eN in L.nodet:  # eval by link.derH.Et + extH.Et * ccoef > ave?
                                if eN in N_:
                                    eN_ += [eN]; N_.remove(eN)  # merged
                            link_+= [L]; et += L.derH.Et
                _eN_ = eN_
            Gt = [node_,link_,et]; Gt_ += [Gt]
        # convert select Gt+minL+subG_ to CGs:
        subG_ = [sum2graph(edge, [node_,link_,et], fd, nest=1) for node_,link_,et in Gt_ if et[0] > ave * et[2]]
        if subG_:
            if fd: edge.subL_ = subG_
            else:  edge.subG_ = subG_  # higher aggr, mediated access to init edge.subG_
    # comp PP_:
    N_,L_,Et = comp_node_(edge.subG_)
    edge.subG_ = N_
    edge.link_ = L_
    if val_(Et, fo=1) > 0:  # cancel by borrowing d?
        mlay = CH().add_H([L.derH for L in L_])
        edge.derH = CH(layt=[mlay], root=edge, Et=copy(mlay.Et))
        mlay.root = edge.derH  # init
        if len(N_) > ave_L:
            cluster_PP_(edge, fd=0)
        # borrow from misprojected m: proj_m -= proj_d, comp instead of link eval:
        if val_(Et, mEt=Et, fo=1) > 0:  # likely not from the same links  (not using val_ here?)
            for L in L_:
                L.extH, L.root, L.mL_t, L.rimt, L.aRad, L.visited_, L.Et = CH(), [edge], [[], []], [[], []], 0, [L], copy(L.derH.Et)
            # comp dPP_:
            lN_,lL_,_ = comp_link_(L_, Et)
            dlay = CH().add_H([L.derH for L in lL_])
            edge.derH.append_(dlay)
            if len(lN_) > ave_L:
                cluster_PP_(edge, fd=1)

def val_(Et, mEt=[], fo=0):

    m, d, n, o = Et
    if any(mEt):
        mm,_,mn,_ = mEt  # cross-induction from root G, not affected by overlap
        val = d * (mm / (ave * mn)) - ave_d * n * (o if fo else 0)
    else:
        val = m - ave * n * (o if fo else 1)  # * overlap in cluster eval, not comp eval
    return val

def comp_node_(_N_):  # rng+ forms layer of rim and extH per N, appends N_,L_,Et, ~ graph CNN without backprop

    _Gp_ = []  # [G pair + co-positionals]
    for _G, G in combinations(_N_, r=2):
        rn = _G.Et[2] / G.Et[2]
        if rn > ave_rn: continue  # scope disparity
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
            if dist < max_dist * ((radii * icoef**3) * (val_(_G.Et)+val_(G.Et) + val_(_G.extH.Et)+val_(G.extH.Et))):
                Link = comp_N(_G,G, rn, angle=[dy,dx],dist=dist)
                L_ += [Link]  # include -ve links
                if val_(Link.derH.Et) > 0:
                    N_.update({_G,G}); Et += Link.derH.Et; _G.add,G.add = 1,1
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
                if _L is not L:
                    if val_(_L.derH.Et, mEt=iEt) > 0: # proj val = compared d * rel root M
                        mL_ += [(_L, rev ^_rev)]  # direction of L relative to _L
    _L_, out_L_, LL_, ET = iL_,set(),[],np.zeros(4)  # out_L_: positive subset of iL_, Et = np.zeros(4)?
    med = 1
    while True:  # xcomp _L_
        L_, Et = set(), np.zeros(4)
        for L in _L_:
            for mL_ in L.mL_t:
                for _L, rev in mL_:  # rev is relative to L
                    rn = _L.Et[2] / L.Et[2]
                    if rn > ave_rn: continue  # scope disparity
                    dy,dx = np.subtract(_L.yx,L.yx)
                    Link = comp_N(_L,L, rn,angle=[dy,dx],dist=np.hypot(dy,dx), dir = -1 if rev else 1)  # d = -d if L is reversed relative to _L
                    Link.med = med
                    LL_ += [Link]  # include -ves, link order: nodet < L < rimt, mN.rim || L
                    if val_(Link.derH.Et) > 0:  # link induction
                        out_L_.update({_L,L}); L_.update({_L,L}); Et += Link.derH.Et
        ET += Et
        if not any(L_): break
        # else extend mL_t per last medL:
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
                                    if val_(__L.derH.Et, mEt=Et) > 0:  # compared __L.derH mag * loop induction
                                        mL_.add((__L, rev ^_rev ^__rev))  # combine reversals: 2 * 2 mLs, but 1st 2 are pre-combined
                                        lEt += __L.derH.Et
                if val_(lEt) > 0: # L'rng+, vs L'comp above
                    L.mL_t = mL_t; _L_.add(L); ext_Et += lEt
            # refine eval by extension D:
            if val_(ext_Et, mEt=Et) - med * med_cost > 0:
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
    # new layer:
    mdext = np.array([np.array([mL,mA]), np.array([dL,dA])])
    _o, o = _N.Et[3], N.Et[3]
    olp = (_o+o) / 2  # inherit from comparands?
    if fd:  # CL
        md_t = np.array([mdext])
        Et = np.array([mL+mA, abs(dL)+abs(dA), .3, olp])  # n = compared vars / 6
    else:  # CG
        vert, et1 = comp_latuple(_N.latuple, N.latuple, _o,o)
        md_vert, et2 = comp_md_(_N.vert[1], N.vert[1], dir)
        md_t = [mdext, vert, md_vert]
        Et = np.array([mL+mA +et1[0]+et2[0], abs(dL)+abs(dA) +et1[1]+et2[1], 2.3, olp])
    # 1st lay:
    md_C = CH(H = md_t, Et=Et)
    derH = CH(H=[md_C], Et=copy(Et))  # no lay = CH(H=[md_C], Et=copy(Et), n=n)
    if _N.derH and N.derH:
        dderH = _N.derH.comp_H(N.derH, rn, root=derH)  # comp shared layers
        derH.append_(dderH, flat=1)
    # spec: comp_node_(node_|link_), combinatorial, node_ nested / rng-)agg+?
    Et = copy(derH.Et)
    if not fd and _N.altG and N.altG:  # not for CL, eval M?
        alt_Link = comp_N(_N.altG, N.altG, _N.altG.Et[2]/N.altG.Et[2])  # no angle,dist, init alternating PPds | dPs?
        derH.altH = alt_Link.derH
        Et += derH.altH.Et
    Link = CL(nodet=[_N,N],derH=derH, yx=np.add(_N.yx,N.yx)/2, angle=angle,dist=dist,box=extend_box(N.box,_N.box))
    if val_(Et)>0:
        derH.root = Link
        for rev, node in zip((0,1), (N,_N)):  # reverse Link direction for _N
            if fd: node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            else:  node.rim += [(Link,dir)]
            node.extH.add_H(Link.derH)
            node.Et += Et
    return Link

def get_rim(N,fd): return N.rimt[0] + N.rimt[1] if fd else N.rim  # add nesting in cluster_N_?

def sum2graph(root, grapht, fd, nest):  # sum node and link params into graph, aggH in agg+ or player in sub+

    node_, link_, Et = grapht[:3]
    Et *= icoef  # is internal now
    graph = CG(fd=fd, Et=Et, root = [root]+[node_[0].root], node_=node_, link_=link_, rng=nest)
    if len(grapht) == 5:  # called from cluster_N
        minL, subG_ = grapht[3:]
        if fd: graph.subL_ = subG_
        else:  graph.subG_ = subG_
        graph.minL = minL
    yx = np.array([0,0])
    fg = isinstance(node_[0],CG)
    if fg: M,D = 0,0
    derH = CH(root=graph)
    for N in node_:
        graph.box = extend_box(graph.box, N.box)  # pre-compute graph.area += N.area?
        yx = np.add(yx, N.yx)
        if N.derH: derH.add_H(N.derH)
        if fg:
            vert = N.vert; M += np.sum(vert[0]); D += np.sum(vert[1]); graph.vert += vert
            graph.latuple += N.latuple
        N.root[-1] = graph  # replace Gt, if single root, else N.root[-1][-1] = graph
    if fg:
        graph.Et[:2] += np.array([M,D]) * icoef**2
    if derH:
        graph.derH = derH  # lower layers
    derLay = CH().add_H([link.derH for link in link_])
    for lay in derLay.H:
        lay.root = graph; graph.derH.H += [lay]  # concat new layer, add node_? higher layers are added by feedback
    graph.derH.Et += Et # arg Et
    L = len(node_)
    yx = np.divide(yx,L); graph.yx = yx
    # ave distance from graph center to node centers:
    graph.aRad = sum([np.hypot( *np.subtract(yx, N.yx)) for N in node_]) / L
    # for CG nodes only:
    if isinstance(N,CG) and fd:
        # assign alt graphs from d graph, after both m and d graphs are formed
        for node in node_:
            mgraph = node.root_[-1]
            # altG summation below is still buggy with current add_H
            if mgraph:
                mgraph.altG = sum_G_([mgraph.altG, graph])  # bilateral sum?
                graph.altG = sum_G_([graph.altG, mgraph])
    return graph

def sum_G_(node_):
    G = CG()
    for n in node_:
        G.rng = n.rng; G.latuple += n.latuple; G.vert += n.vert; G.aRad += n.aRad; G.box = extend_box(G.box, n.box)
        if n.derH: G.derH.add_H(n.derH)
        if n.extH: G.extH.add_H(n.extH)
    return G

if __name__ == "__main__":
    image_file = '../images/raccoon_eye.jpeg'
    image = imread(image_file)
    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vectorize_root(frame)