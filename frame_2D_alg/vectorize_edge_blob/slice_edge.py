import numpy as np
from math import atan2, cos, floor, pi
import sys
sys.path.append("..")
from frame_blobs import CBase, CH, imread   # for CP
from intra_blob import CsubFrame

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
ave_dangle = .2  # vertical difference between angles: -1->1, abs dangle: 0->1, ave_dangle = (min abs(dangle) + max abs(dangle))/2,

class CsliceEdge(CsubFrame):

    class CEdge(CsubFrame.CBlob):     # replaces CBlob after definition

        def term(blob):     # an extension to CsubFrame.CBlob.term(), evaluate for vectorization right after rng+ in intra_blob
            super().term()
            if not blob.sign and blob.G > aveG * blob.root.rdn:
                blob.vectorize()

        def vectorize(blob):        # to be overridden in higher modules (comp_slice, agg_recursion)
            blob.slice_edge()

        def slice_edge(edge):
            root__ = {}  # map max yx to P, like in frame_blobs
            dert__ = {}
            for yx, axis in edge.select_max():
                P = CP(edge, yx, axis, root__, dert__)
                if P.yx_:
                    edge.P_ += [P]  # P_ is added dynamically, only edge-blobs have P_
            edge.P_ = sorted(edge.P_, key=lambda P: P.yx[0], reverse=True)  # sort Ps in descending order (bottom up)
            # scan to update link_:
            for i, P in enumerate(edge.P_):
                y, x = P.yx  # pivot, change to P center
                for _P in edge.P_[i+1:]:  # scan all higher Ps to get links to adjacent / overlapping Ps in P_ sorted by y
                    _y, _x = _P.yx
                    # get max possible y,x extension from P centers:
                    Dy = abs(P.yx_[0][0] - P.yx_[-1][0])/2; _Dy = abs(_P.yx_[0][0] - _P.yx_[-1][0])/2
                    Dx = abs(P.yx_[0][1] - P.yx_[-1][1])/2; _Dx = abs(_P.yx_[0][1] - _P.yx_[-1][1])/2
                    # min gap = distance between centers - combined extension,
                    # max overlap is negative min gap:
                    ygap = (_P.yx[0] - P.yx[0]) - (Dy+_Dy)
                    xgap = abs(_P.yx[1]-P.yx[1]) - (Dx+_Dx)
                    # overlapping | adjacent Ps:
                    if ygap <= 0 and xgap <= 0:
                        angle = np.subtract((y,x),(_y,_x))
                        P.link_[0] += [Clink(node=P, _node=_P, distance=np.hypot(*angle), angle=angle)]  # prelinks

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
            max_.sort(key=lambda itm: itm[0])  # sort by yx
            return max_

    CBlob = CEdge


class Clink(CBase):  # the product of comparison between two nodes

    def __init__(l,_node=None, node=None, dderH= None, roott=None, distance=0.0, angle=None):
        super().__init__()
        l.Et = [0,0,0,0,0,0]  # graph-specific, accumulated from surrounding nodes in node_connect
        l.node_ = []  # e_ in kernels, else replaces _node,node: not used in kernels?
        l.link_ = []  # list of mediating Clinks in hyperlink
        l.dderH = CH() if dderH is None else dderH
        l.roott = [None, None] if roott is None else roott  # clusters that contain this link
        l.distance = distance  # distance between node centers
        l.angle = [0,0] if angle is None else angle  # dy,dx between node centers
        # dir: bool  # direction of comparison if not G0,G1, only needed for comp link?
        # deprecated:
        l._node = _node  # prior comparand
        l.node = node
        l.med_Gl_ = []  # replace by link_, intermediate nodes and links in roughly the same direction, as in hypergraph edges

    def __bool__(l): bool(l.dderH.H)
    # draft:
    def comp_link(_link, link, dderH, rn=1, fagg=0, flat=1):  # use in der+ and comp_kernel

        dderH = comp_(_link.dderH, link.dderH, dderH, rn=1, fagg=0, flat=1)
        mA,dA = comp_angle(_link.angle, link.angle)
        # draft:
        for _med_link,med_link in zip(_link.link_,link.link_):
            comp_link(_med_link, med_link)


class CP(CBase):
    def __init__(P, edge, yx, axis, root__, dert__):  # form_P:

        super().__init__()
        y, x = yx
        pivot = i, gy, gx, g = interpolate2dert(edge, y, x)  # pivot dert
        ma = ave_dangle  # max value because P direction is the same as dert gradient direction
        m = ave_g - g
        pivot += ma, m   # pack extra ders

        I, G, M, Ma, L, Dy, Dx = i, g, m, ma, 1, gy, gx
        P.axis = ay, ax = axis
        P.yx_, P.dert_, P.link_ = [yx], [pivot], [[]]
        f_overlap = 0  # to determine if there's overlap

        for dy, dx in [(-ay, -ax), (ay, ax)]: # scan in 2 opposite directions to add derts to P
            P.yx_.reverse(); P.dert_.reverse()
            (_y, _x), (_, _gy, _gx, *_) = yx, pivot  # start from pivot
            y, x = _y+dy, _x+dx  # 1st extension
            while True:
                # scan to blob boundary or angle miss:
                try: i, gy, gx, g = interpolate2dert(edge, y, x)
                except TypeError: break  # out of bound (TypeError: cannot unpack None)

                mangle,dangle = comp_angle((_gy,_gx), (gy, gx))
                if mangle < ave_dangle: break  # terminate P if angle miss
                # update P:
                m = ave_g - g
                I += i; Dy += dy; Dx += dx; G += g; Ma += ma; M += m; L += 1
                # not sure about round
                if (round(y),round(x)) in dert__:   # stop forming P if any overlapping dert
                    f_overlap = 1
                    P.yx_= []
                    break
                P.yx_ += [(y, x)]; P.dert_ += [(i, gy, gx, g, ma, m)]
                # for next loop:
                y += dy; x += dx
                _y, _x, _gy, _gx = y, x, gy, gx
            if f_overlap: break

        if not f_overlap:
            for yx in P.yx_:
                dert__[round(yx[0]), round(yx[1])] = P  # update dert__

            P.yx = P.yx_[L // 2]
            root__[yx[0], yx[1]] = P    # update root__
            P.latuple = I, G, M, Ma, L, (Dy, Dx)
            P.derH = CH()

    def __repr__(P): return f"P({', '.join(map(str, P.latuple))})"  # or return f"P(id={P.id})" ?

def interpolate2dert(edge, y, x):
    if (y, x) in edge.dert_: return edge.dert_[y, x]  # if edge has (y, x) in it

    # get nearby coords:
    y_ = [fy] = [floor(y)]; x_ = [fx] = [floor(x)]
    if y != fy: y_ += [fy+1]    # y is non-integer
    if x != fx: x_ += [fx+1]    # x is non-integer
    n, I, Dy, Dx, G = 0, 0, 0, 0, 0
    for _y in y_:
        for _x in x_:
            if (_y, _x) in edge.dert_:
                _i, _dy, _dx, _g = edge.dert_[_y, _x]
                I += _i; Dy += _dy; Dx += _dx; G += _g; n += 1

    if n >= 2: return I/n, Dy/n, Dx/n, G/n


def comp_angle(_A, A):  # rn doesn't matter for angles

    _angle, angle = [atan2(Dy, Dx) for Dy, Dx in [_A, A]]
    dangle = _angle - angle  # difference between angles

    if dangle > pi: dangle -= 2*pi  # rotate full-circle clockwise
    elif dangle < -pi: dangle += 2*pi  # rotate full-circle counter-clockwise

    mangle = (cos(dangle)+1)/2  # angle similarity, scale to [0,1]
    dangle /= 2*pi  # scale to the range of mangle, signed: [-.5,.5]

    return [mangle, dangle]

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
    sorted_edge_ = sorted(edge_, key=lambda edge: len(edge.yx_), reverse=True)
    for edge in sorted_edge_[:num_to_show]:
        yx_ = np.array(edge.yx_)
        yx0 = yx_.min(axis=0) - 1
        shape = yx_.max(axis=0) - yx0 + 2
        mask_nonzero = tuple(zip(*(yx_ - yx0)))
        mask = np.zeros(shape, bool)
        mask[mask_nonzero] = True
        plt.imshow(mask, cmap='gray', alpha=0.5)
        plt.title(f"area = {edge.area}")

        for P in edge.P_:
            yx1, yx2 = P.yx_[0], P.yx_[-1]
            y_, x_ = zip(yx1 - yx0, yx2 - yx0)
            yp, xp = P.yx - yx0
            plt.plot(x_, y_, "b-", linewidth=2)
            for _P in P.link_:
                _yp, _xp = _P.yx - yx0
                plt.plot([_xp, xp], [_yp, yp], "ko-")

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()