def rng_trace_rim(N_, Et):  # comp Clinks: der+'rng+ in root.link_ rim_t node rims: directional and node-mediated link tracing

    _L_ = N_
    rng = 1
    while _L_:
        L_ = []
        for L in _L_:
            for dir, rim_ in zip((0,1), L.rim_t):  # Link rng layers
                for mL in rim_[-1]:
                    for N in mL.nodet:  # N in L.nodet, _N in _L.nodet
                        if N.root is not L.nodet:  # N.root = nodet
                            _L = N.root[-1]     # nodet[-1] = L
                            if _L is L or _L in L.compared_:
                                continue
                            if not hasattr(_L,"rimt"):
                                add_der_attrs( link_=[_L])  # _L is outside root.link_, still same derivation
                            L.compared_ += [_L]
                            _L.compared_ += [L]
                            Link = Clink(nodet =[_L,L], box=extend_box(_L.box, L.box))
                            comp_N(Link, L_, Et, rng, dir) # L_+=nodet, L.rim_t+=Link
                            break  # only one N mediates _L
    # or
            for dir, rim_ in zip((0,1), L.rim_t):  # Link rng layers
                for mL in rim_[-1]: # prior _L
                    # mL has Links in rim_t 1st layer:
                    for link in mL.rim_t[0][0] + mL.rim_t[0][1]:
                        for _L in link.nodet:
                            if _L is L or _L in L.compared_:  # if _L is mL
                                continue
                            if not hasattr(_L,"rim_t"):
                                add_der_attrs( link_=[_L])  # _L is outside root.link_, still same derivation
                            L.compared_ += [_L]
                            _L.compared_ += [L]
                            Link = Clink(nodet =[_L,L], box=extend_box(_L.box, L.box))
                            comp_N(Link, L_, Et, rng, dir) # L_+=nodet, L.rim_t+=Link
                            break  # only one N mediates _L
        _L_=L_; rng+=1
    return N_, rng, Et

def rng_trace_rim(N_, Et):  # comp Clinks: der+'rng+ in root.link_ rim_t node rims: directional and node-mediated link tracing

    _L_ = N_
    rng = 1
    while _L_:
        L_ = []
        for L in _L_:
            for nodet in L.nodet[-1]:  # variable nesting _L-mediating nodes, init Gs
                for dir, N_ in zip((0,1),nodet):  # Link direction
                    if not isinstance(N_,list): N_ = [N_]
                    for N in N_:
                        rim = N.rim if isinstance(N,CG) else N.rimt_[-1][0]+N.rimt_[-1][1]  # concat dirs
                        for _L in rim:
                            if _L is L or _L in L.compared_: continue
                            if not hasattr(_L,"rimt_"):
                                add_der_attrs( link_=[_L])  # _L is outside root.link_, still same derivation
                            L.compared_ += [_L]; _L.compared_ += [L]
                            # not needed?:
                            if rng > 1:  # draft
                                coSpan = L.span * np.cos( np.arctan2(*np.subtract(L.angle, (dy,dx))))
                                _coSpan = _L.span * np.cos( np.arctan2(*np.subtract(_L.angle, (dy,dx))))
                                if dist / ((coSpan +_coSpan) / 2) > max_dist * rng:  # Ls are too distant
                                    continue
                            Link = Clink(nodet =[_L,L], box=extend_box(_L.box, L.box))
                            comp_N(Link, L_, Et, rng, dir)  # L_+=nodet, L.rim_t+=Link
        _L_=L_; rng+=1
    return N_, rng, Et

def rng_node_(N_, Et, rng=1):  # comp Gs|kernels in agg+, links | link rim_t node rims in sub+
                               # similar to graph convolutional network but without backprop
    _G_ = N_
    G_ = []  # eval comp_N -> G_:
    for link in list(combinations(_G_,r=2)):
        _G, G = link
        if _G in G.compared_: continue
        dy,dx = np.subtract(_G.yx,G.yx)
        dist = np.hypot(dy,dx)
        aRad = (G.aRad+_G.aRad) /2  # ave radius to eval relative distance between G centers:
        if dist / max(aRad,1) <= max_dist * rng:
            G.compared_ += [_G]; _G.compared_ += [G]
            Link = Clink(nodet=[_G,G], span=dist, angle=[dy,dx], box=extend_box(G.box,_G.box))
            if comp_N(Link, Et):
                G_ += [_G,G]
    _G_ = list(set(G_))
    for G in _G_:  # init kernel as [krim]
        krim =[]; DerH =CH() # layer/krim
        for link in G.rim:
            DerH.add_(link.derH)
            krim += [link.nodet[0] if link.nodet[1] is G else link.nodet[1]]
        if DerH: G.derH.append_(DerH, flat=0)
        G.kH = [krim]
    rng = 1  # aggregate rng+: recursive center node DerH += linked node derHs for next-loop cross-comp
    while len(_G_) > 2:
        G_ = []
        for G in _G_:
            if len(G.rim) < 2: continue  # one link is always overlapped
            for link in G.rim:
                if link.derH.Et[0] > ave:  # link.Et+ per rng
                    comp_krim(link, G_, rng)  # + kernel rim / loop, sum in G.extH, derivatives in link.extH?
        _G_ = list(set(G_))
        rng += 1
    for G in _G_:
        for i, link in enumerate(G.rim):
            G.extH.add_(link.DerH) if i else G.extH.append_(link.DerH, flat=1)  # for segmentation

    return N_, rng, Et
'''
    _Gt_ = []
    for G in G_:  # def kernel rim per G:
        _Gi_,M = [],0
        for link in G.rim:
            _G = [link.nodet[0] if link.nodet[1] is G else link.nodet[1]]
            _Gi_ += [G_.index(_G)]
            M += link.derH.Et[0]
        _Gt_ += [[G,_Gi_,M,[]]]  # _Gi_: krim indices, []: local compared_
    rng = 1
    while _Gt_:  # aggregate rng+ cross-comp: recursive center node DerH += linked node derHs for next loop
        Gt_ = []
        for i, [G,_Gi_,_M,_compared_] in enumerate(_Gt_):
            Gi_, M, compared_ = [],0,[]
            for _i in _Gi_:  # krim indices
                if _i in _compared_: continue
                _G,__Gi_,__M,__compared_ = _Gt_[_i]
                compared_ += [i]; __compared_ += [_i]  # bilateral assign
                dderH = _G.derH.comp_(G.derH)
                m = dderH.Et[0]
                if m > ave * dderH.Et[2]:
                    M += m; __M += m
                    Gi_ += [_i]; __Gi_+=[i]
                    for g in _G,G:
                        g.DerH.H[-1].add_(dderH, irdnt=dderH.H[-1].Et[2:]) if len(g.DerH.H)==rng else g.DerH.append_(dderH,flat=0)
            if M-_M > ave:
                Gt_ += [[G, Gi_, M, compared_]]
'''
def comp_G(_G, G):  # comp without forming links

    dderH = CH()
    _n,_L,_S,_A,_derH,_extH,_iderH,_latuple = _G.n,len(_G.node_),_G.S,_G.A,_G.derH,_G.extH,_G.iderH,_G.latuple
    n, L, S, A, derH, extH, iderH, latuple = G.n,len(G.node_), G.S, G.A, G.derH, G.extH, G.iderH, G.latuple
    rn = _n/n
    et, rt, md_ = comp_ext(_L,L, _S,S/rn, _A,A)
    Et, Rt, Md_ = comp_latuple(_latuple, latuple, rn, fagg=1)
    dderH.n = 1; dderH.Et = np.add(Et,et); dderH.relt = np.add(Rt,rt)
    dderH.H = [CH(Et=et,relt=rt,H=md_,n=.5),CH(Et=Et,relt=Rt,H=Md_,n=1)]
    _iderH.comp_(iderH, dderH, rn, fagg=1, flat=0)

    if _derH and derH: _derH.comp_(derH, dderH, rn, fagg=1, flat=0)  # append and sum new dderH to base dderH
    if _extH and extH: _extH.comp_(extH, dderH, rn, fagg=1, flat=1)

    return dderH

'''
G.DerH sums krim _G.derHs, not from links, so it's empty in the first loop.
_G.derHs can't be empty in comp_krim: init in loop link.derHs
link.DerH is ders from comp G.DerH in comp_krim
G.extH sums link.DerHs: '''

def comp_krim(link, G_, nrng, fd=0):  # sum rim _G.derHs, compare to form link.DerH layer

    _G,G = link.nodet  # same direction
    ave = G_aves[fd]
    for node in _G, G:
        if node in G_: continue  # new krim is already added
        krim = []  # kernel rim
        for _node in node.kH[-1]:
            for _link in _node.rim:
                __node = _link.nodet[0] if _link.nodet[1] is _node else _link.nodet[1]
                krim += [_G for _G in __node.kH[-1] if _G not in krim]
                if node.DerH: node.DerH.add_(__node.derH, irdnt=_node.Et[2:])
                else:         node.DerH = deepcopy(__node.derH)  # init
        node.kH += [krim]
    _skrim = set(_G.kH[-1]); skrim = set(G.kH[-1])
    _xrim = list(_skrim - skrim)
    xrim = list(skrim - _skrim)  # exclusive kernel rims
    if _xrim and xrim:
        dderH = comp_N_(_xrim, xrim)
        if dderH.Et[0] > ave * dderH.Et[2]:
            G_ += [_G,G]  # update node_, use nested link.derH vs DerH?
            link.DerH.H[-1].add_(dderH, irdnt=dderH.H[-1].Et[2:]) if len(link.DerH.H)==nrng else link.DerH.append_(dderH,flat=1)

    # connectivity eval in segment_graph via decay = (link.relt[fd] / (link.derH.n * 6)) ** nrng  # normalized decay at current mediation

import numpy as np
from collections import defaultdict
from math import atan2, cos, floor, pi
import sys
sys.path.append("..")
from frame_blobs import CBase, CFrame, imread, unpack_blob_

'''
In natural images, objects look very fuzzy and frequently interrupted, only vaguely suggested by initial blobs and contours.
Potential object is proximate low-gradient (flat) blobs, with rough / thick boundary of adjacent high-gradient (edge) blobs.
These edge blobs can be dimensionality-reduced to their long axis / median line: an effective outline of adjacent flat blob.
-
Median line can be connected points that are most equidistant from other blob points, but we don't need to define it separately.
An edge is meaningful if blob slices orthogonal to median line form some sort of a pattern: match between slices along the line.
These patterns effectively vectorize representation: they represent match and change between slice parameters along the blob.
-
This process is very complex, so it must be selective. Selection should be by combined value of gradient deviation of edge blobs
and inverse gradient deviation of flat blobs. But the latter is implicit here: high-gradient areas are usually quite sparse.
A stable combination of a core flat blob with adjacent edge blobs is a potential object.
'''
# filters:
octant = 0.3826834323650898  # radians per octant
aveG = 10  # for vectorize
ave_g = 30  # change to Ave from the root intra_blob?
ave_mL = 2
ave_dist = 3
max_dist = 15
ave_dangle = .95  # vertical difference between angles: -1->1, abs dangle: 0->1, ave_dangle = (min abs(dangle) + max abs(dangle))/2,


class CsliceEdge(CFrame):

    class CEdge(CFrame.CBlob): # replaces CBlob

        def term(blob):  # extension of CsubFrame.CBlob.term(), evaluate for vectorization right after rng+ in intra_blob
            super().term()
            if not blob.sign and blob.G > aveG * blob.root.rdn:
                blob.vectorize()

        def vectorize(blob):  # overridden in comp_slice, agg_recursion
            blob.slice_edge()

        def slice_edge(edge):
            axisd = edge.select_max()   # select max
            yx_ = sorted(axisd.keys(), key=lambda yx: edge.dert_[yx][-1])  # sort by g

            # form P per non-overlapped max yx
            edge.P_ = []; edge.rootd = {}
            while yx_:
                yx = yx_.pop(); axis = axisd[yx]  # get max of maxes (highest g)
                edge.P_ += [CP(edge, yx, axis)]   # form P
                yx_ = [yx for yx in yx_ if yx not in edge.rootd]    # remove merged maxes if any

            edge.P_.sort(key=lambda P: P.yx, reverse=True)
            edge.trace()
            # del edge.rootd    # still being used for visual verification
            return edge

        def select_max(edge):
            axisd = {}  # map yx to axis
            for (y, x), (i, gy, gx, g) in edge.dert_.items():
                sa, ca = gy/g, gx/g
                # check neighbors
                new_max = True
                for dy, dx in [(-sa, -ca), (sa, ca)]:
                    _y, _x = round(y+dy), round(x+dx)
                    if (_y, _x) not in edge.dert_: continue  # skip if pixel not in edge blob
                    _i, _gy, _gx, _g = edge.dert_[_y, _x]  # get g of neighbor
                    if g < _g:
                        new_max = False
                        break
                if new_max: axisd[y, x] = sa, ca
            return axisd

        def trace(edge):  # fill and trace across slices

            adjacent_ = [(P, y,x) for P in edge.P_ for y,x in edge.rootd if edge.rootd[y,x] is P]
            edge.pre__ = defaultdict(list)  # prelinks
            while adjacent_:
                _P, _y,_x = adjacent_.pop(0)  # also pop _P__
                _pre_ = edge.pre__[_P]
                for y,x in [(_y-1,_x),(_y,_x+1),(_y+1,_x),(_y,_x-1)]:
                    try:  # if yx has _P, try to form link
                        P = edge.rootd[y,x]
                        pre_ = edge.pre__[P]
                        if _P is not P and _P not in pre_ and P not in _pre_:
                            if _P.yx < P.yx: pre_ += [_P]  # edge.P_'s uplinks
                            else:            _pre_ += [P]
                    except KeyError:    # if yx empty, keep tracing
                        if (y,x) not in edge.dert_: continue   # stop if yx outside the edge
                        edge.rootd[y,x] = _P
                        adjacent_ += [(_P, y,x)]
    CBlob = CEdge

class CP(CBase):

    def __init__(P, edge, yx, axis):
        super().__init__()
        y, x = yx
        P.axis = ay, ax = axis
        pivot = i,gy,gx,g = edge.dert_[y,x]  # dert is None if (_y, _x) not in edge.dert_: return` in `interpolate2dert`
        ma = ave_dangle  # ? max value because P direction is the same as dert gradient direction
        m = ave_g - g
        pivot += ma,m
        edge.rootd[y, x] = P
        I,G,M,Ma,L,Dy,Dx = i,g,m,ma,1,gy,gx
        P.yx_, P.dert_ = [yx], [pivot]

        for dy,dx in [(-ay,-ax),(ay,ax)]:  # scan in 2 opposite directions to add derts to P
            P.yx_.reverse(); P.dert_.reverse()
            (_y,_x), (_,_gy,_gx,*_) = yx, pivot  # start from pivot
            y,x = _y+dy, _x+dx  # 1st extension
            while True:
                # scan to blob boundary or angle miss:
                ky, kx = round(y), round(x)
                if (round(y),round(x)) not in edge.dert_: break
                try: i,gy,gx,g = interpolate2dert(edge, y, x)
                except TypeError: break  # out of bound (TypeError: cannot unpack None)
                mangle, dangle = comp_angle((_gy,_gx), (gy, gx))
                if mangle < ave_dangle: break  # terminate P if angle miss
                # update P:
                edge.rootd[ky, kx] = P
                m = ave_g - g
                I += i; Dy += dy; Dx += dx; G += g; Ma += ma; M += m; L += 1
                P.yx_ += [(y,x)]; P.dert_ += [(i,gy,gx,g,ma,m)]
                # for next loop:
                y += dy; x += dx
                _y,_x,_gy,_gx = y,x,gy,gx

        P.yx = tuple(np.mean([P.yx_[0], P.yx_[-1]], axis=0))
        P.latuple = I, G, M, Ma, L, (Dy, Dx)


# --------------------------------------------------------------------------------------------------------------
# utility functions

def interpolate2dert(edge, y, x):
    if (y, x) in edge.dert_: return edge.dert_[y, x]  # if edge has (y, x) in it

    # get coords surrounding dert:
    y_ = [fy] = [floor(y)]; x_ = [fx] = [floor(x)]
    if y != fy: y_ += [fy+1]    # y is non-integer
    if x != fx: x_ += [fx+1]    # x is non-integer

    I, Dy, Dx, G = 0, 0, 0, 0
    K = 0
    for _y in y_:
        for _x in x_:
            if (_y, _x) not in edge.dert_: continue
            i, dy, dx, g = edge.dert_[_y, _x]
            k = (1 - abs(_y-y)) * (1 - abs(_x-x))
            I += i*k; Dy += dy*k; Dx += dx*k; G += g*k
            K += k

    if K != 0:
        return I/K, Dy/K, Dx/K, G/K


def comp_angle(_A, A):  # rn doesn't matter for angles

    _angle, angle = [atan2(Dy, Dx) for Dy, Dx in [_A, A]]
    dangle = _angle - angle  # difference between angles

    if dangle > pi: dangle -= 2*pi  # rotate full-circle clockwise
    elif dangle < -pi: dangle += 2*pi  # rotate full-circle counter-clockwise

    mangle = (cos(dangle)+1)/2  # angle similarity, scale to [0,1]
    dangle /= 2*pi  # scale to the range of mangle, signed: [-.5,.5]

    return [mangle, dangle]

def unpack_edge_(frame):
    return [blob for blob in unpack_blob_(frame) if hasattr(blob, "P_")]

if __name__ == "__main__":

    image_file = '../images/raccoon_eye.jpeg'
    image = imread(image_file)

    frame = CsliceEdge(image).segment()
    # verification:
    import matplotlib.pyplot as plt

    # settings
    num_to_show = 5
    show_gradient = True
    show_slices = True

    # unpack and sort edges
    edge_ = sorted(unpack_edge_(frame), key=lambda edge: len(edge.yx_), reverse=True)

    # show first largest n edges
    for edge in edge_[:num_to_show]:
        yx_ = np.array(edge.yx_)
        yx0 = yx_.min(axis=0) - 1

        # show edge-blob
        shape = yx_.max(axis=0) - yx0 + 2
        mask_nonzero = tuple(zip(*(yx_ - yx0)))
        mask = np.zeros(shape, bool)
        mask[mask_nonzero] = True
        plt.imshow(mask, cmap='gray', alpha=0.5)
        plt.title(f"area = {edge.area}")

        # show gradient
        if show_gradient:
            vu_ = [(-gy/g, gx/g) for i, gy, gx, g in edge.dert_.values()]
            y_, x_ = zip(*(yx_ - yx0))
            v_, u_ = zip(*vu_)
            plt.quiver(x_, y_, u_, v_, scale=100)

        # show slices
        if show_slices:
            for P in edge.P_:
                y_, x_ = zip(*(P.yx_ - yx0))
                if len(P.yx_) == 1:
                    v, u = P.axis
                    y_ = y_[0]-v/2, y_[0]+v/2
                    x_ = x_[0]-u/2, x_[0]+u/2
                plt.plot(x_, y_, "k-", linewidth=3)
                yp, xp = P.yx - yx0
                pre_set = set()
                for _P in edge.pre__[P]:
                    assert _P.id not in pre_set     # verify pre-link uniqueness
                    pre_set.add(_P.id)
                    assert _P.yx < P.yx     # verify up-link
                    _yp, _xp = _P.yx - yx0
                    plt.plot([_xp, xp], [_yp, yp], "ko--", alpha=0.5)

                yx_ = [yx for yx in edge.rootd if edge.rootd[yx] is P]
                if yx_:
                    y_, x_ = zip(*(yx_ - yx0))
                    plt.plot(x_, y_, 'o', alpha=0.5)

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()

def feedback(root, root_fd, fsub=1):  # called from form_graph_, append new der layers to root

    DerH = deepcopy(root.fback_t[0].pop(0))  # init DerH merged from both forks
    for fd in 0,1:
        while root.fback_t[fd]:
            derH = root.fback_t[fd].pop(0)
            DerH.add_(derH)
        if DerH.Et[fd] > G_aves[fd] * DerH.Et[fd+2]:  # merge combined DerH into root.derH
            if fsub: root.derH.append_(DerH, flat=1)  # append higher layers
            else:    root.derH.add_(DerH)  # sum shared layers, append the rest

    # recursive feedback, propagated when sub+ ends in all nodes of both forks:
    if root.root and isinstance(root.root, CG):  # not Edge
        rroot = root.root
        if rroot:
            rroot.fback_t[root_fd] += [DerH]
            if all(len(f_) == len(rroot.node_) for f_ in rroot.fback_t):  # both forks of sub+ end for all nodes
                feedback(rroot, root_fd=root_fd, fsub=fsub)  # sum2graph adds higher aggH, feedback adds deeper aggH layers

def form_graph_t(root, N_, Et, rng, root_fd):  # segment N_ to Nm_, Nd_

    node_t = []
    for fd in 0,1:
        # der+: comp link via link.node_ -> dual trees of matching links in der+rng+, more likely but complex: higher-order links
        # rng+: distant M / G.M, less likely: term by >d/<m, but less complex
        if Et[fd] > ave * Et[2+fd]:
            if not fd:
                for G in N_: G.root = []  # only nodes have roots?
            graph_ = segment_N_(root, N_, fd, rng)
            for graph in graph_:
                Q = graph.link_ if fd else graph.node_
                if len(Q) > ave_L and graph.Et[fd] > G_aves[fd] * graph.Et[fd]:
                    if fd: add_der_attrs(Q)
                    # else sub+rng+: comp Gs at distance < max_dist * rng+1:
                    agg_recursion(root, graph, Q, rng+1, fagg=1-fd)  # graph.node_ is not node_t yet, rng for rng+ only
                else:
                    root.fback_t[fd] += [graph.derH]  # sub+ fb -> root formed by intermediate agg+, -> original root
                    if all(len(f_) == len(graph_) for f_ in root.fback_t):  # both forks of sub+ end for all nodes
                        feedback(root, root_fd=root_fd)  # graph_ is new root.node_:
            node_t += [graph_]  # may be empty
        else:
            node_t += [[]]

    return node_t
