from math import atan2, cos, floor, pi
import sys
sys.path.append("..")
from frame_blobs import CBase, imread   # for CP
from intra_blob import CIntraBlobFrame

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

class CSliceEdgeFrame(CIntraBlobFrame):

    def evaluate(frame):
        super().evaluate()
        frame.edge_ = []
        edgeQue = list(frame.blob_)
        while edgeQue:
            blob = edgeQue.pop(0)
            try: rdn = blob.root.rdn
            except AttributeError: rdn = 1

            if not blob.sign and blob.G > aveG * rdn:  frame.edge_ += [blob.slice_edge()]    # slice edge
            elif hasattr(blob, "lay"):            edgeQue += blob.lay.blob_             # flatten/unpack deeper blobs
        return frame

    class CEdge(CIntraBlobFrame.CBlob):
        def slice_edge(edge):
            root__ = {}  # map max yx to P, like in frame_blobs
            edge.P_ = [CP(edge, yx, axis, root__) for yx, axis in edge.select_max()]  # max = (yx, axis)
            return edge

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

    CBlob = CEdge   # Replace CBlob with CEdge

class CP(CBase):
    def __init__(P, edge, yx, axis, root__):  # form_P:

        super().__init__()
        y, x = yx
        pivot = i, gy, gx, g = interpolate2dert(edge, y, x)  # pivot dert
        ma = ave_dangle  # max value because P direction is the same as dert gradient direction
        m = ave_g - g
        pivot += ma, m   # pack extra ders

        I, G, M, Ma, L, Dy, Dx = i, g, m, ma, 1, gy, gx
        P.axis = ay, ax = axis
        P.yx_, P.dert_, P.link_ = [yx], [pivot], []

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
                P.yx_ += [(y, x)]; P.dert_ += [(i, gy, gx, g, ma, m)]
                # for next loop:
                y += dy; x += dx
                _y, _x, _gy, _gx = y, x, gy, gx

        # scan for neighbor P pivots, update link_:
        y, x = yx   # pivot
        for _y, _x in [(y-1,x-1), (y-1,x), (y-1,x+1), (y,x-1), (y,x+1), (y+1,x-1), (y+1,x), (y+1,x+1)]:
            if (_y, _x) in root__:  # neighbor has P
                P.link_ += [root__[_y, _x]]
        root__[y, x] = P    # update root__

        P.yx = P.yx_[L // 2]  # center
        P.latuple = I, G, M, Ma, L, (Dy, Dx)

    def __repr__(P):
        return f"P({', '.join(map(str, P.latuple))})"  # or return f"P(id={P.id})" ?

def interpolate2dert(edge, y, x):
    if (y, x) in edge.dert_:   # if edge has (y, x) in it
        return edge.dert_[y, x]

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

    frame = CSliceEdgeFrame(image).evaluate()
    # verification:
    import numpy as np
    import matplotlib.pyplot as plt

    # show first largest n edges
    num_to_show = 5
    sorted_edge_ = sorted(frame.edge_, key=lambda edge: len(edge.yx_), reverse=True)
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