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

    class CEdge(CsubFrame.CBlob): # replaces CBlob

        def term(blob):  # extension of CsubFrame.CBlob.term(), evaluate for vectorization right after rng+ in intra_blob
            super().term()
            if not blob.sign and blob.G > aveG * blob.root.rdn:
                blob.vectorize()

        def vectorize(blob):  # overridden in comp_slice, agg_recursion
            blob.slice_edge()

        def slice_edge(edge):
            hyx_ = edge.slice()  # horizontal map
            edge.trace(hyx_)  # vertical map

        def slice(edge):  # scan along direction of gradient
            # use max G in kernel to get direction, immediately form partial Ps?
            hyx_ = defaultdict(list)
            for (y,x), (i,gy,gx,g) in edge.dert_.items():
                sa,ca = gy/g, gx/g  # sin_angle, cos_angle
                # get neighbor direction
                dy = 1 if sa > octant else -1 if sa < -octant else 0
                dx = 1 if ca > octant else -1 if ca < -octant else 0
                for _y,_x in [(y-dy, x-dx), (y+dy, x+dx)]:
                    if (_y,_x) not in edge.dert_: continue  # skip if pixel not in edge blob
                    if (y,x) not in hyx_[_y,_x]: hyx_[_y,_x] += [(y,x)]
                    if (_y,_x) not in hyx_[y,x]: hyx_[y,x] += [(_y,_x)]
            return hyx_

        def trace(edge, hyx_):  # fill and trace across slices
            edge.P_ = []
            fill_yx_ = list(hyx_.keys())  # set of pixel coordinates to be filled (fill_yx_)
            hdert_ = defaultdict(list)  # map derts to blob
            vadjacent_ = []  # derts vertically adjacent to P
            while fill_yx_:  # fill_yx_ is popped per filled pixel, in form_blob
                if not vadjacent_:  # init blob
                    P = CP(); vadjacent_ += [fill_yx_[0]]
                P.form(edge, fill_yx_, vadjacent_, hdert_, hyx_)
                if not vadjacent_:  # scan P
                    P.term(edge)
                    edge.P_ += [P]
    CBlob = CEdge

class CP(CBase):
    def __init__(P):
        super().__init__()
        P.dert_ = {}
        P.link_ = [[]]

    def form(P, edge, fill_yx_, vadjacent_, hdert_, hyx_):
        y,x = vadjacent_.pop()
        if (y,x) not in fill_yx_: return
        fill_yx_.remove((y,x))
        P.dert_[y,x] = edge.dert_[y,x]
        hdert_[y,x] += [P]
        for _y,_x in [(y-1,x-1),(y-1,x),(y-1,x+1),(y,x+1),(y+1,x+1),(y+1,x),(y+1,x-1),(y,x-1)]:
            if (_y,_x) in hyx_[y,x]:
                vadjacent_ += [(_y,_x)]
            else:  # get _Ps vertically adjacent to P
                for _P in hdert_[_y,_x]:
                    if _P not in P.link_[0]:
                        P.link_[0] += [_P]

    def term(P, edge):
        y, x = P.yx = map(sum, zip(*P.dert_.keys()))
        ay,ax = P.axis = map(sum, zip(*((gy/g, gx/g) for i,gy,gx,g in P.dert_.values())))

        pivot = i,gy,gx,g = interpolate2dert(edge, y,x)
        ma = ave_dangle  # max value because P direction is the same as dert gradient direction
        m = ave_g - g
        pivot += ma,m

        I,G,M,Ma,L,Dy,Dx = i,g,m,ma,1,gy,gx
        dert_ = P.dert_  # DEBUG
        P.yx_, P.dert_, P.link_ = [P.yx], [pivot], [[]]

        # this rotation should be recursive, use P.latuple Dy,Dx to get secondary direction, no need for axis?
        for dy,dx in [(-ay,-ax),(ay,ax)]:  # scan in 2 opposite directions to add derts to P
            P.yx_.reverse(); P.dert_.reverse()
            (_y,_x), (_,_gy,_gx,*_) = P.yx, pivot  # start from pivot
            y,x = _y+dy, _x+dx  # 1st extension
            while True:
                # scan to blob boundary or angle miss:
                try: i,gy,gx,g = interpolate2dert(edge, y,x)
                except TypeError: break  # out of bound (TypeError: cannot unpack None)

                mangle,dangle = comp_angle((_gy,_gx), (gy, gx))
                if mangle < ave_dangle: break  # terminate P if angle miss
                # update P:
                m = ave_g - g
                I += i; Dy += dy; Dx += dx; G += g; Ma += ma; M += m; L += 1
                P.yx_ += [(y,x)]; P.dert_ += [(i,gy,gx,g,ma,m)]
                # for next loop:
                y += dy; x += dx
                _y,_x,_gy,_gx = y,x,gy,gx

        P.yx = P.yx_[L // 2]
        P.latuple = I, G, M, Ma, L, (Dy, Dx)
        P.derH = CH()

    def __repr__(P): return f"P({', '.join(map(str, P.latuple))})"  # or return f"P(id={P.id})" ?


class Clink(CBase):  # the product of comparison between two nodes

    def __init__(l, node_=None,rim=None, derH=None, extH=None, roott=None, distance=0, angle=None ):
        super().__init__()
        if hasattr(node_[0],'yx'): _y,_x = node_[0].yx; _y,_x = node_[1].yx # CP
        else:                      _y,_x = box2center(node_[0].box); y,x = box2center(node_[1].box) # CG
        l.angle = np.subtract([y,x], [_y, _x]) if angle is None else angle  # dy,dx between node centers
        l.distance = np.hypot(*l.angle) if distance is None else distance  # distance between node centers
        l.Et = [0,0,0,0]  # graph-specific, accumulated from surrounding nodes in node_connect
        l.relt = [0,0]
        l.node_ = [] if node_ is None else node_  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.rim = []  # for der+, list of mediating Clinks in hyperlink in roughly the same direction, as in hypergraph
        l.derH = CH() if derH is None else derH
        l.extH = CH() if extH is None else extH  # for der+
        l.roott = [None, None] if roott is None else roott  # clusters that contain this link
        # dir: bool  # direction of comparison if not G0,G1, only needed for comp link?
        # n: always min(node_.n)?

    def __bool__(l): return bool(l.dderH.H)

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

    # get coords surrounding dert:
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
    import matplotlib.pyplot as plt

    # show first largest n edges
    edge_, edgeQue = [], list(frame.blob_)
    while edgeQue:
        blob = edgeQue.pop(0)
        if hasattr(blob, "P_"): edge_ += [blob]
        elif hasattr(blob, "rlay"): edgeQue += blob.rlay.blob_

    num_to_show = 5
    show_gradient = True
    show_slices = True
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
        if show_slices:
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