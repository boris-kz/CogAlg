import sys
import numpy as np
from math import floor
from collections import namedtuple, deque, defaultdict
from itertools import product, combinations
from .classes import CEdge, CP
from .filters import ave_g, ave_dangle, ave_daangle

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

ptupleT = namedtuple("ptupleT", "I G M Ma angle L")
octant = 0.3826834323650898

def slice_edge(blob, verbose=False):
    max_mask__ = max_selection(blob)  # mask of local directional maxima of dy, dx, g
    # form slices (Ps) from max_mask__ and form links by tracing max_mask__:
    edge = trace_edge(blob, max_mask__, verbose=verbose)
    return edge

def max_selection(blob):

    Y, X = blob.mask__.shape
    g__ = blob.der__t.g
    # compute direction of gradient
    with np.errstate(divide='ignore', invalid='ignore'):
        sin__, cos__ = [blob.der__t.dy, blob.der__t.dx] / g__

    # round angle to one of eight directions
    up__, lft__, dwn__, rgt__ = (sin__< -octant), (cos__< -octant), (sin__> octant), (cos__> octant)
    mdly__, mdlx__ = ~(up__ | dwn__), ~(lft__ | rgt__)
    # merge in 4 bilateral axes
    axes_mask__ = [
        mdly__ & (rgt__ | lft__), (dwn__ & rgt__) | (up__ & lft__),  #  0,  45 deg
        (dwn__ | up__) & mdlx__,  (dwn__ & lft__) | (up__ & rgt__),  # 90, 135 deg
    ]
    max_mask__ = np.zeros_like(blob.mask__, dtype=bool)
    # local max from cross-comp within axis, use kernel max for vertical sparsity?
    for axis_mask__, (ydir, xdir) in zip(axes_mask__, ((0,1),(1,1),(1,0),(1,-1))):  # y,x direction per axis
        # axis AND mask:
        mask__ = axis_mask__ & blob.mask__
        y_, x_ = mask__.nonzero()
        # neighbors:
        yn1_, xn1_ = y_ + ydir, x_ + xdir
        yn2_, xn2_ = y_ - ydir, x_ - xdir
        # computed vals
        axis1_ = (0 <= yn1_) & (yn1_ < Y) & (0 <= xn1_) & (xn1_ < X)
        axis2_ = (0 <= yn2_) & (yn2_ < Y) & (0 <= xn2_) & (xn2_ < X)
        # compare values
        not_max_ = np.zeros_like(y_, dtype=bool)
        not_max_[axis1_] |= (g__[y_[axis1_], x_[axis1_]] < g__[yn1_[axis1_], xn1_[axis1_]])
        not_max_[axis2_] |= (g__[y_[axis2_], x_[axis2_]] < g__[yn2_[axis2_], xn2_[axis2_]])
        # select maxes
        mask__[y_[not_max_], x_[not_max_]] = False
        # add to max_mask__
        max_mask__ |= mask__

    return max_mask__

def trace_edge(blob, mask__, verbose=False):

    edge = CEdge(blob=blob)
    blob.dlayers = [[edge]]
    max_ = {*zip(*mask__.nonzero())}  # convert mask__ into a set of (y,x)

    if verbose:
        step = 100 / len(max_)  # progress % percent per pixel
        progress = 0.0; print(f"\rTracing max... {round(progress)} %", end="");  sys.stdout.flush()
    edge.node_t = []
    while max_:  # queue of (y,x,P)s
        y,x = max_.pop()
        maxQue = deque([(y,x,None)])
        while maxQue:  # trace max_
            # initialize pivot dert
            y,x,_P = maxQue.popleft()
            i = blob.i__[blob.ibox.slice()][y, x]
            dy, dx, g = blob.der__t.get_pixel(y, x)
            ma = ave_dangle  # max value because P direction is the same as dert gradient direction
            assert g > 0, "g must be positive"
            P = form_P(blob, CP(yx=(y,x), axis=(dy/g, dx/g), cells={(y,x)}, dert_=[(y,x,i,dy,dx,g,ma)]))
            edge.node_t += [P]
            if _P is not None:
                P.link_H[0] += [_P]  # add up links only
            # search in max_ path
            adjacents = max_ & {*product(range(y-1,y+2), range(x-1,x+2))}   # search neighbors
            maxQue.extend(((_y, _x, P) for _y, _x in adjacents))
            max_ -= adjacents   # set difference = first set AND not both sets: https://www.scaler.com/topics/python-set-difference/
            max_ -= P.cells     # remove all maxes in the way

            if verbose:
                progress += step; print(f"\rTracing max... {round(progress)} %", end=""); sys.stdout.flush()

    if verbose: print("\r" + " " * 79, end=""); sys.stdout.flush(); print("\r", end="")

    return edge

def form_P(blob, P):

    scan_direction(blob, P, fleft=1)  # scan left
    scan_direction(blob, P, fleft=0)  # scan right
    # init:
    _, _, I, Dy, Dx, G, Ma = map(sum, zip(*P.dert_))
    L = len(P.dert_)
    M = ave_g*L - G
    G = np.hypot(Dy, Dx)  # recompute G
    P.ptuple = ptupleT(I, G, M, Ma, (Dy,Dx), L)
    P.yx = P.dert_[L//2][:2]  # new center

    return P

def scan_direction(blob, P, fleft):  # leftward or rightward from y,x

    sin,cos = _dy,_dx = P.axis
    _y, _x = P.yx  # pivot
    r = cos*_y - sin*_x  # P axial line: cos*y - sin*x = r = constant
    _cy,_cx = round(_y), round(_x)  # keep initial cell
    y, x = (_y-sin,_x-cos) if fleft else (_y+sin, _x+cos)  # first dert in the direction of axis

    while True:  # scan to blob boundary or angle miss

        dert = interpolate2dert(blob, y, x)
        if dert is None: break  # blob boundary
        i, dy, dx, g = dert
        cy, cx = round(y), round(x)  # nearest cell of (y, x)
        if not blob.mask__[cy, cx]: break
        if abs(cy-_cy) + abs(cx-_cx) == 2:  # mask of cell between (y,x) and (_y,_x)
            my = (_cy+cy) / 2  # midpoint cell, P axis is above, below or over it
            mx = (_cx+cx) / 2
            _my_cos = sin * mx + r  # _my*cos at mx in P, to avoid division
            my_cos = my * cos       # new cell
            if cos < 0: my_cos, _my_cos = -my_cos, -_my_cos   # reverse sign for comparison because of cos
            if abs(my_cos-_my_cos) > 1e-5:
                adj_y, adj_x = (  # deviation from P axis: above/_y>y, below/_y<y, over/_y~=y, with reversed y:
                    ((_cy, cx) if _cy < cy else (cy, _cx)) if _my_cos < my_cos else
                    ((_cy, cx) if _cy > cy else (cy, _cx)))
                if not blob.mask__[adj_y, adj_x]: break    # if the cell is masked, stop
                P.cells |= {(adj_y, adj_x)}

        mangle,dangle = comp_angle((_dy,_dx), (dy, dx))
        if mangle < 0:  # terminate P if angle miss
            break
        P.cells |= {(cy, cx)}  # add current cell to overlap
        _cy, _cx, _dy, _dx = cy, cx, dy, dx
        if fleft:
            P.dert_ = [(y,x,i,dy,dx,g,mangle)] + P.dert_  # append left
            y -= sin; x -= cos  # next y,x
        else:
            P.dert_ = P.dert_ + [(y,x,i,dy,dx,g,mangle)]  # append right
            y += sin; x += cos  # next y,x

def interpolate2dert(blob, y, x):

    Y, X = blob.mask__.shape    # boundary
    x0, y0 = floor(x), floor(y) # floor
    x1, y1 = x0 + 1, y0 + 1     # ceiling
    if x0 < 0 or x1 >= X or y0 < 0 or y1 >= Y: return None  # boundary check
    kernel = [  # cell weighing by inverse distance from float y,x:
        # https://www.researchgate.net/publication/241293868_A_study_of_sub-pixel_interpolation_algorithm_in_digital_speckle_correlation_method
        (y0, x0, (y1 - y) * (x1 - x)),
        (y0, x1, (y1 - y) * (x - x0)),
        (y1, x0, (y - y0) * (x1 - x)),
        (y1, x1, (y - y0) * (x - x0))]
    ider__t = (blob.i__[blob.ibox.slice()],) + blob.der__t

    return (sum((par__[ky, kx] * dist for ky, kx, dist in kernel)) for par__ in ider__t)


def comp_angle(_angle, angle):  # rn doesn't matter for angles

    # angle = [dy,dx]
    (_sin, sin), (_cos, cos) = [*zip(_angle, angle)] / np.hypot(*zip(_angle, angle))

    dangle = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    # cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
    mangle = ave_dangle - abs(dangle)  # inverse match, not redundant if sum cross sign

    return [mangle, dangle]

def comp_aangle(_aangle, aangle):  # currently not used, just in case we need it later

    _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _aangle
    sin_da0, cos_da0, sin_da1, cos_da1 = aangle

    sin_dda0 = (cos_da0 * _sin_da0) - (sin_da0 * _cos_da0)
    cos_dda0 = (cos_da0 * _cos_da0) + (sin_da0 * _sin_da0)
    sin_dda1 = (cos_da1 * _sin_da1) - (sin_da1 * _cos_da1)
    cos_dda1 = (cos_da1 * _cos_da1) + (sin_da1 * _sin_da1)
    # for 2D, not reduction to 1D:
    # aaangle = (sin_dda0, cos_dda0, sin_dda1, cos_dda1)
    # day = [-sin_dda0 - sin_dda1, cos_dda0 + cos_dda1]
    # dax = [-sin_dda0 + sin_dda1, cos_dda0 + cos_dda1]
    gay = np.arctan2((-sin_dda0 - sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in y?
    gax = np.arctan2((-sin_dda0 + sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in x?

    # daangle = sin_dda0 + sin_dda1?
    daangle = np.arctan2(gay, gax)  # diff between aangles, probably wrong
    maangle = ave_daangle - abs(daangle)  # inverse match, not redundant as summed

    return [maangle,daangle]