import numpy as np
from collections import defaultdict
from math import atan2, cos, floor, pi
import sys
sys.path.append("..")
from frame_blobs import CBase, CH, imread   # for CP
from intra_blob import CsubFrame
from utils import box2center
from .filters import  ave_mL, ave_dangle, ave_dist, max_dist
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
octant = 0.3826834323650898  # radians per octant
aveG = 10  # for vectorize
ave_g = 30  # change to Ave from the root intra_blob?
ave_dangle = .8  # vertical difference between angles: -1->1, abs dangle: 0->1, ave_dangle = (min abs(dangle) + max abs(dangle))/2,

class CsliceEdge(CsubFrame):

    class CEdge(CsubFrame.CBlob):     # replaces CBlob after definition

        def term(blob):     # an extension to CsubFrame.CBlob.term(), evaluate for vectorization right after rng+ in intra_blob
            super().term()
            if not blob.sign and blob.G > aveG * blob.root.rdn:
                blob.vectorize()

        def vectorize(blob):        # to be overridden in higher modules (comp_slice, agg_recursion)
            blob.slice_edge()

        def slice_edge(edge):
            edge.rootd = defaultdict(list)
            edge.P_ = sorted([CP(edge, yx, axis) for yx, axis in edge.select_max()], key=lambda P: P.yx, reverse=True)
            # scan to update link_:
            for P in edge.P_:
                fill_ = [yx for yx in edge.rootd if P in edge.rootd[yx]]
                yx_ = list(edge.rootd.keys())
                while fill_:
                    y, x = fill_.pop(0)
                    if (y, x) not in yx_: continue
                    yx_.remove((y, x))
                    term = False
                    for _P in edge.rootd[y, x]:
                        if _P is not P:
                            term = True
                            if _P not in P.link_[0]: P.link_[0] += [_P]
                    if not term: fill_ += [(y-1,x-1),(y-1,x),(y-1,x+1),(y,x+1),(y+1,x+1),(y+1,x),(y+1,x-1),(y,x-1)]

                P.link_[0] = [Clink(node=P, _node=_P) for _P in P.link_[0]]

            del edge.rootd

        def select_max(edge):
            max_ = []
            for (y, x), (i, gy, gx, g) in edge.dert_.items():
                # sin_angle, cos_angle:
                sa, ca = gy/g, gx/g
                # get neighbor direction
                dy = 1 if sa > octant else -1 if sa < -octant else 0
                dx = 1 if ca > octant else -1 if ca < -octant else 0
                new_max = True
                for _y, _x in [(y-dy, x-dx), (y+dy, x+dx)]:
                    if (_y, _x) not in edge.dert_: continue  # skip if pixel not in edge blob
                    _i, _gy, _gx, _g = edge.dert_[_y, _x]  # get g of neighbor
                    if g < _g:
                        new_max = False
                        break
                if new_max: max_ += [((y, x), (sa, ca))]
            return max_

    CBlob = CEdge


class Clink(CBase):  # the product of comparison between two nodes

    def __init__(l,_node=None, node=None, dderH= None, roott=None, distance=0, angle=None ):
        super().__init__()
        # for P only, use box for Gs?:
        l.angle = np.subtract(node.yx, _node.yx) if angle is None else angle  # dy,dx between node centers
        l.distance = np.hypot(*l.angle) if distance is None else distance  # distance between node centers
        l.Et = [0,0,0,0]  # graph-specific, accumulated from surrounding nodes in node_connect
        l.relt = [0,0]
        l.node_ = []  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.link_ = []  # list of mediating Clinks in hyperlink in roughly the same direction, as in hypergraph
        l.dderH = CH() if dderH is None else dderH
        l.roott = [None, None] if roott is None else roott  # clusters that contain this link
        # l.dir: bool  # direction of comparison if not G0,G1, only needed for comp link?

    def __bool__(l): return bool(l.dderH.H)

    def comp_link(Link, dderH, fagg=0):  # use in der+ and comp_kernel

        if isinstance(Link,list):  # if der+'rng+: form new Clink
            _link,link = Link
            Link = Clink(node_=[_link,link])
        else: _link,link = Link.node_  # higher-der link

        _y1,_x1 = box2center(_link.node_[0].box)
        _y2,_x2 = box2center(_link.node_[1].box)
        y1,x1 = box2center(link.node_[0].box)
        y2,x2 = box2center(link.node_[1].box)
        dy = (y1+y2)/2 - (_y1+_y2)/2
        dx = (x1+x2)/2 - (_x1+_x2)/2
        Link.angle = (dy,dx)
        Link.distance = np.hypot(dy, dx)  # distance between link centers
        (_G1,_G2),(G1,G2) = _link.node_,link.node_
        rn = min(_G1.n,_G2.n) / min(G1.n,G2.n) # min: only shared layers are compared
        _link.dderH.comp_(link.dderH, dderH, rn, fagg=0, flat=1)
        ddderH = CH()
        for _med_link,med_link in zip(_link.link_,link.link_):
            _med_link.comp_link(med_link, ddderH)
        dderH.add_(ddderH)
        # comp_ext:
        _L,L,_S,S,_A,A = _link.distance,link.distance, len(_link.link_),len(link.link_), _link.angle,link.angle
        L/=rn; S/=rn
        dL = _L - L;      mL = min(_L,L) - ave_mL  # direct match
        dS = _S/_L - S/L; mS = min(_S,S) - ave_mL  # sparsity is accumulated over L
        mA, dA = comp_angle(_A, A)  # angle is not normalized
        dist = Link.distance
        prox = ave_dist-dist  # proximity = inverted distance (position difference), no prior accum to n
        M = prox + mL + mS + mA
        D = dist + abs(dL) + abs(dS) + abs(dA)  # signed dA?
        mrdn = M > D; drdn = D<= M
        mdec = prox / max_dist + mL/ max(L,_L) + mS/ max(S,_S) if S or _S else 1 + mA  # Amax = 1
        ddec = dist / max_dist + mL/ (L+_L) + dS/ (S+_S) if S or _S else 1 + dA

        dderH.append_(CH(Et=[M,D,mrdn,drdn],relt=[mdec,ddec], H=[prox,dist, mL,dL, mS,dS, mA,dA], n=2/3), flat=0)  # 2/3 of 6-param unit


class CP(CBase):
    def __init__(P, edge, yx, axis):  # form_P:

        super().__init__()
        y, x = yx
        pivot = i, gy, gx, g = edge.dert_[y, x]
        ma = ave_dangle  # max value because P direction is the same as dert gradient direction
        m = ave_g - g
        pivot += ma, m   # pack extra ders

        I, G, M, Ma, L, Dy, Dx = i, g, m, ma, 1, gy, gx
        P.axis = ay, ax = axis
        P.yx_, P.dert_, P.link_ = [yx], [pivot], [[]]

        for dy, dx in [(-ay, -ax), (ay, ax)]: # scan in 2 opposite directions to add derts to P
            P.yx_.reverse(); P.dert_.reverse()
            (_y, _x), (_, _gy, _gx, *_) = yx, pivot  # start from pivot
            y, x = _y+dy, _x+dx  # 1st extension
            while True:
                # scan to blob boundary or angle miss:
                try: i, gy, gx, g = interpolate2dert(edge, y, x)
                except TypeError: break  # out of bound (TypeError: cannot unpack None)

                mangle,dangle = comp_angle((_gy,_gx), (gy, gx))
                if abs(mangle*2-1) < ave_dangle: break  # terminate P if angle miss
                # update P:
                if P not in edge.rootd[round(y), round(x)]: edge.rootd[round(y), round(x)] += [P]
                m = ave_g - g
                I += i; Dy += dy; Dx += dx; G += g; Ma += ma; M += m; L += 1
                P.yx_ += [(y, x)]; P.dert_ += [(i, gy, gx, g, ma, m)]
                # for next loop:
                y += dy; x += dx
                _y, _x, _gy, _gx = y, x, gy, gx

        P.yx = P.yx_[L // 2]
        P.latuple = I, G, M, Ma, L, (Dy, Dx)
        P.derH = CH()

    def __repr__(P): return f"P({', '.join(map(str, P.latuple))})"  # or return f"P(id={P.id})" ?

def interpolate2dert(edge, y, x):
    if (y, x) in edge.dert_: return edge.dert_[y, x]  # if edge has (y, x) in it

    # get nearby coords:
    y_ = [fy] = [floor(y)]; x_ = [fx] = [floor(x)]
    if y != fy: y_ += [fy+1]    # y is non-integer
    if x != fx: x_ += [fx+1]    # x is non-integer

    I, Dy, Dx, G = 0, 0, 0, 0
    for _y in y_:
        for _x in x_:
            if (_y, _x) not in edge.dert_: return
            i, dy, dx, g = edge.dert_[_y, _x]
            k = (1 - abs(_y-y)) * (1 - abs(_x-x))
            I += i*k; Dy += dy*k; Dx += dx*k; G += g*k

    return I, Dy, Dx, G


def comp_angle(_A, A):  # rn doesn't matter for angles

    _angle, angle = [atan2(Dy, Dx) for Dy, Dx in [_A, A]]
    dangle = _angle - angle  # difference between angles

    if dangle > pi: dangle -= 2*pi  # rotate full-circle clockwise
    elif dangle < -pi: dangle += 2*pi  # rotate full-circle counter-clockwise

    mangle = (cos(dangle)+1)/2  # angle similarity, scale to [0,1]
    dangle /= 2*pi  # scale to the range of mangle, signed: [-.5,.5]

    return [mangle, dangle]

def project(y, x, s, c, r):
    dist = s*y + c*x - r
    # Subtract left and right side by dist:
    # 0 = s*y + c*x - r - dist
    # 0 = s*y + c*x - r - dist*(s*s + c*c)
    # 0 = s*(y - dist*s) + c*(x - dist*c) - r
    # therefore, projection of y, x onto the line is:
    return y - dist*s, x - dist*c

if __name__ == "__main__":

    image_file = '../images/raccoon_eye.jpeg'
    image = imread(image_file)

    frame = CsliceEdge(image).segment()
    # verification:
    import numpy as np
    import matplotlib.pyplot as plt

    # show first largest n edges
    edge_, edgeQue = [], list(frame.blob_)
    while edgeQue:
        blob = edgeQue.pop(0)
        if hasattr(blob, "P_"): edge_ += [blob]
        elif hasattr(blob, "rlay"): edgeQue += blob.rlay.blob_

    num_to_show = 5
    show_gradient = False
    sorted_edge_ = sorted(edge_, key=lambda edge: len(edge.yx_), reverse=True)
    for edge in sorted_edge_[:num_to_show]:
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
            plt.quiver(x_, y_, u_, v_)

        # show slices
        edge.P_.sort(key=lambda P: len(P.yx_), reverse=True)
        for P in edge.P_:
            yx1, yx2 = P.yx_[0], P.yx_[-1]
            y_, x_ = zip(*(P.yx_ - yx0))
            yp, xp = P.yx - yx0
            plt.plot(x_, y_, "g-", linewidth=2)
            for link in P.link_[-1]:
                _yp, _xp = link._node.yx - yx0
                plt.plot([_xp, xp], [_yp, yp], "ko-")

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()