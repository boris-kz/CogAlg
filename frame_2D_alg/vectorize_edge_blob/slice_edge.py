import numpy as np
from collections import defaultdict
from itertools import combinations
from math import atan2, cos, floor, pi
import sys
sys.path.append("..")
from frame_blobs import frame_blobs_root, intra_blob_root, CBase, imread, unpack_blob_
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
ave_olp = 5

class CP(CBase):

    def __init__(P, edge, yx, axis):
        super().__init__()
        y, x = yx
        P.axis = ay, ax = axis
        pivot = i,gy,gx,g = edge.dert_[y,x]  # dert is None if _y,_x not in edge.dert_: return` in `interpolate2dert`
        ma = ave_dangle  # max if P direction = dert g direction
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
                if edge.rootd.get((ky,kx)) is not None: break  # skip overlapping P
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

def vectorize_root(frame):

    blob_ = unpack_blob_(frame)
    for blob in blob_:
        if not blob.sign and blob.G > aveG * blob.root.rdn:
            slice_edge(blob)

def slice_edge(edge):

    axisd = select_max(edge)
    yx_ = sorted(axisd.keys(), key=lambda yx: edge.dert_[yx][-1])  # sort by g
    edge.P_ = []; edge.rootd = {}
    # form P/ local max yx:
    while yx_:
        yx = yx_.pop(); axis = axisd[yx]  # get max of g maxes
        edge.P_ += [CP(edge, yx, axis)]   # form P
        yx_ = [yx for yx in yx_ if yx not in edge.rootd]    # remove merged maxes if any
    edge.P_.sort(key=lambda P: P.yx, reverse=True)
    trace(edge)
    # del edge.rootd  # for visual verification
    return edge

def select_max(edge):
    axisd = {}  # map yx to axis
    for (y, x), (i, gy, gx, g) in edge.dert_.items():
        sa, ca = gy/g, gx/g
        new_max = True
        for dy, dx in [(-sa, -ca), (sa, ca)]:  # check neighbors
            _y, _x = round(y+dy), round(x+dx)
            if (_y, _x) not in edge.dert_: continue  # skip if pixel not in edge blob
            _i, _gy, _gx, _g = edge.dert_[_y, _x]  # neighboring g
            if g < _g:
                new_max = False
                break
        if new_max: axisd[y, x] = sa, ca
    return axisd

def trace(edge):  # fill and trace across slices

    adjacent_ = [(P, y,x) for P in edge.P_ for y,x in edge.rootd if edge.rootd[y,x] is P]
    bi__ = defaultdict(list)  # prelinks (bi-lateral)
    while adjacent_:
        _P, _y,_x = adjacent_.pop(0)  # also pop _P__
        _pre_ = bi__[_P]
        for y,x in [(_y-1,_x),(_y,_x+1),(_y+1,_x),(_y,_x-1)]:
            try:  # if yx has _P, try to form link
                P = edge.rootd[y,x]
                pre_ = bi__[P]
                if _P is not P and _P not in pre_ and P not in _pre_:
                    pre_ += [_P]
                    _pre_ += [P]
            except KeyError:    # if yx empty, keep tracing
                if (y,x) not in edge.dert_: continue   # stop if yx outside the edge
                edge.rootd[y,x] = _P
                adjacent_ += [(_P, y,x)]
    # remove redundant links
    for P in edge.P_:
        yx = P.yx
        for __P, _P in combinations(bi__[P], r=2):
            if __P not in bi__[P] or _P not in bi__[P]: continue
            __yx, _yx = __P.yx, _P.yx   # center coords
            # start -> end:
            __yx1 = np.subtract(__P.yx_[0], __P.axis)
            __yx2 = np.add(__P.yx_[-1], __P.axis)
            _yx1 = np.subtract(_P.yx_[0], _P.axis)
            _yx2 = np.add(_P.yx_[-1], _P.axis)
            # remove link(_P,P) crossing __P:
            if xsegs(yx, _yx, __yx1, __yx2):
                bi__[P].remove(_P)
                bi__[_P].remove(P)
            # remove link(__P,P) crossing _P):
            elif xsegs(yx, __yx, _yx1, _yx2):
                bi__[P].remove(__P)
                bi__[__P].remove(P)
    for P in edge.P_:
        for _P in bi__[P]:
            if P in bi__[_P]:
                if _P.yx < P.yx: bi__[_P].remove(P)
                else:            bi__[P].remove(_P)

    edge.pre__ = bi__  # prelinks

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

def xsegs(yx1, yx2, yx3, yx4):
    # return True if segments (yx1, yx2) & (yx3, yx4) crossed
    # https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    (y1, x1), (y2, x2), (y3, x3), (y4, x4) = yx1, yx2, yx3, yx4

    v1 = (y2 - y1)*(x3 - x2) - (x2 - x1)*(y3 - y2)
    v2 = (y2 - y1)*(x4 - x2) - (x2 - x1)*(y4 - y2)

    v3 = (y4 - y3)*(x1 - x4) - (x4 - x3)*(y1 - y4)
    v4 = (y4 - y3)*(x2 - x4) - (x4 - x3)*(y2 - y4)

    return (v1*v2 <= 0 and v3*v4 <= 0)

if __name__ == "__main__":
    image_file = '../images/raccoon_eye.jpeg'
    image = imread(image_file)

    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    vectorize_root(frame)
    # verification:
    import matplotlib.pyplot as plt
    # settings
    num_to_show = 5
    show_gradient = True
    show_slices = True
    # unpack and sort edges:
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
        if show_gradient:
            vu_ = [(-gy/g, gx/g) for i, gy, gx, g in edge.dert_.values()]
            y_, x_ = zip(*(yx_ - yx0))
            v_, u_ = zip(*vu_)
            plt.quiver(x_, y_, u_, v_, scale=100)
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