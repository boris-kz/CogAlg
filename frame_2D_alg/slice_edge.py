import numpy as np
from collections import defaultdict
from itertools import combinations
from math import atan2, cos, floor, pi
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
ave_I, ave_G, ave_dangle  = 100, 100, 0.95

class CP(CBase):
    def __init__(P, yx, axis):
        super().__init__()
        P.yx = yx
        P.axis = axis
        P.dert_ = []

def slice_edge_root(frame, rM=1):

    blob_ = unpack_blob_(frame)
    for blob in blob_:
        if not blob.sign and blob.G > ave_G * blob.area * rM:
            slice_edge(blob, rM)

def slice_edge(edge, rV=1):
    if rV != 1:
        global ave_I, ave_G, ave_dangle
        ave_I, ave_G, ave_dangle = np.array([ave_I, ave_G, ave_dangle]) / rV  # projected value change

    axisd = select_max(edge)
    yx_ = sorted(axisd.keys(), key=lambda yx: edge.dert_[yx][-1])  # sort by g
    edge.P_ = []; edge.rootd = {}
    # form P/ local max yx:
    while yx_:
        yx = yx_.pop(); axis = axisd[yx]  # get max of g maxes
        P = form_P(CP(yx, axis), edge)
        edge.P_ += [P]
        yx_ = [yx for yx in yx_ if yx not in edge.rootd]    # remove merged maxes if any
    edge.P_.sort(key=lambda P: P.yx, reverse=True)
    trace_P_adjacency(edge)
    if __name__ != "__main__": del edge.rootd   # keep for visual verification in slice_edge only
    return edge

def select_max(edge):
    axisd = {}  # map yx to axis
    for (y, x), (i, gy, gx, g) in edge.dert_.items():
        sa, ca = gy/max(g,1e-7), gx/max(g,1e-7)
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

def form_P(P, edge):
    y, x = P.yx
    ay, ax = P.axis
    center_dert = i,g,gy,gx = edge.dert_[y,x]  # dert is None if _y,_x not in edge.dert_: return` in `interpolate2dert`
    edge.rootd[y, x] = P
    I,G,Dy,Dx,M,D,L = i,g,gy,gx, 0,0,1
    P.yx_ = [P.yx]
    P.dert_ += [center_dert]

    for dy,dx in [(-ay,-ax),(ay,ax)]:  # scan in 2 opposite directions to add derts to P
        P.yx_.reverse(); P.dert_.reverse()
        (_y,_x), (_i,_g,_gy,_gx) = P.yx, center_dert  # start from center_dert
        y,x = _y+dy, _x+dx  # 1st extension
        while True:
            # scan to blob boundary or angle miss:
            ky, kx = round(y), round(x)
            if (round(y),round(x)) not in edge.dert_: break
            try: i,g,gy,gx = interpolate2dert(edge, y, x)
            except TypeError: break  # out of bound (TypeError: cannot unpack None)
            if edge.rootd.get((ky,kx)) is not None:
                break  # skip overlapping P
            mangle, dangle = comp_angle((_gy,_gx),(gy,gx))
            if mangle < ave_dangle: break  # terminate P if angle miss
            m = min(_i,i) + min(_g,g) + mangle
            d = abs(-i-i) + abs(_g-g) + dangle
            if m < ave_I + ave_G + ave_dangle: break  # terminate P if total miss, blob should be more permissive than P
            # update P:
            edge.rootd[ky, kx] = P
            I+=i; Dy+=dy; Dx+=dx; G+=g; M+=m; D+=d; L+=1
            P.yx_ += [(y,x)]; P.dert_ += [(i,g,gy,gx)]
            # for next loop:
            y += dy; x += dx
            _y,_x,i,g,_gy,_gx = y,x,i,g,gy,gx

    P.yx = tuple(np.mean([P.yx_[0], P.yx_[-1]], axis=0))    # new center
    P.latuple = np.array([I, G, Dy, Dx, M, D, L])

    return P

def trace_P_adjacency(edge):  # fill and trace across slices

    margin_rim = [(P, y,x) for P in edge.P_ for y,x in edge.rootd if edge.rootd[y,x] is P]
    prelink__ = defaultdict(list)
    # bilateral
    while margin_rim:   # breadth-first search for neighbors
        _P, _y,_x = margin_rim.pop(0)  # also pop _P__
        _margin = prelink__[_P]  # empty list per _P
        for y,x in [(_y-1,_x),(_y,_x+1),(_y+1,_x),(_y,_x-1)]:  # adjacent pixels
            if (y,x) not in edge.dert_: continue   # yx is outside the edge
            if (y,x) not in edge.rootd:  # assign root, keep tracing
                edge.rootd[y, x] = _P
                margin_rim += [(_P, y, x)]
                continue
            # form link if yx has _P
            P = edge.rootd[y,x]
            margin = prelink__[P]  # empty list per P
            if _P is not P and _P not in margin and P not in _margin:
                margin += [_P]; _margin += [P]
    # remove crossed links
    for _P in edge.P_:
        _yx = _P.yx
        for P, __P in combinations(prelink__[_P], r=2):
            if {__P,P}.intersection(prelink__[_P]) != {__P, P}: continue   # already removed
            yx, __yx = P.yx, __P.yx
            # get aligned line segments:
            yx1 = np.subtract(P.yx_[0], P.axis)
            yx2 = np.add(P.yx_[-1], P.axis)
            __yx1 = np.subtract(__P.yx_[0], __P.axis)
            __yx2 = np.add(__P.yx_[-1], __P.axis)
            # remove crossed uplinks:
            if xsegs(yx, _yx, __yx1, __yx2):
                prelink__[P].remove(_P); prelink__[_P].remove(P)
            elif xsegs(_yx, __yx, yx1, yx2):
                prelink__[_P].remove(__P); prelink__[__P].remove(_P)
    # for comp_slice:
    edge.pre__ = {_P:[P for P in prelink__[_P] if _P.yx > P.yx] for _P in prelink__}

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

    # image_file = './images//raccoon_eye.jpeg'
    image_file = './images//toucan_small.jpg'
    image = imread(image_file)

    frame = frame_blobs_root(image)
    intra_blob_root(frame)
    slice_edge_root(frame)
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
        assert len(edge.P_) == len({P.yx for P in edge.P_})     # verify that P.yx is unique
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
            for _P in edge.P_:
                _y_, _x_ = zip(*(_P.yx_ - yx0))
                if len(_P.yx_) == 1:
                    v, u = _P.axis
                    _y_ = _y_[0]-v/2, _y_[0]+v/2
                    _x_ = _x_[0]-u/2, _x_[0]+u/2
                plt.plot(_x_, _y_, "k-", linewidth=3)
                _yp, _xp = _P.yx - yx0
                assert len(set(edge.pre__[_P])) == len(edge.pre__[_P])   # verify pre-link uniqueness
                for P in edge.pre__[_P]:
                    assert _P.yx > P.yx     # verify up-link
                    yp, xp = P.yx - yx0
                    plt.plot([_xp, xp], [_yp, yp], "ko--", alpha=0.5)
                _cyx_ = [_yx for _yx in edge.rootd if edge.rootd[_yx] is _P]
                if _cyx_:
                    _cy_, _cx_ = zip(*(_cyx_ - yx0))
                    plt.plot(_cx_, _cy_, 'o', alpha=0.5)

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()