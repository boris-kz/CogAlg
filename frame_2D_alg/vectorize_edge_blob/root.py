import sys
import numpy as np
from copy import copy, deepcopy
from itertools import product
from frame_blobs import UNFILLED, EXCLUDED
from .classes import CEdge, CP, CPP, CderP, Cgraph
from .filters import ave, ave_g, ave_ga, ave_rotate
from .comp_slice import comp_slice, comp_angle, sum_derH
from .hough_P import new_rt_olp_array, hough_check
from .agg_recursion import agg_recursion, sum_aggH
from .sub_recursion import sub_recursion_eval

'''
Vectorize is a terminal fork of intra_blob.
-
In natural images, objects look very fuzzy and frequently interrupted, only vaguely suggested by initial blobs and contours.
Potential object is proximate low-gradient (flat) blobs, with rough / thick boundary of adjacent high-gradient (edge) blobs.
These edge blobs can be dimensionality-reduced to their long axis / median line: an effective outline of adjacent flat blob.
-
Median line can be connected points that are most equidistant from other blob points, but we don't need to define it separately.
An edge is meaningful if blob slices orthogonal to median line form some sort of a pattern: match between slices along the line.
In simplified edge tracing we cross-compare among blob slices in x along y, where y is the longer dimension of a blob / segment.
Resulting patterns effectively vectorize representation: they represent match and change between slice parameters along the blob.
-
This process is very complex, so it must be selective. Selection should be by combined value of gradient deviation of edge blobs
and inverse gradient deviation of flat blobs. But the latter is implicit here: high-gradient areas are usually quite sparse.
A stable combination of a core flat blob with adjacent edge blobs is a potential object.
-
So, comp_slice traces blob axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)
-
Vectorization is clustering of parameterized Ps + their derivatives (derPs) into PPs: patterns of Ps that describe edge blob.
This process is a reduced-dimensionality (2D->1D) version of cross-comp and clustering cycle, common across this project.
As we add higher dimensions (3D and time), this dimensionality reduction is done in salient high-aspect blobs
(likely edges in 2D or surfaces in 3D) to form more compressed "skeletal" representations of full-dimensional patterns.
'''

oct_sep = 0.3826834323650898

def vectorize_root(blob, verbose=False):

    max_mask__ = non_max_suppression(blob)  # mask of local max dy,dx,g
    # form Ps from max_mask__ and form links by tracing max_mask__:
    edge = slice_blob_ortho(blob, max_mask__, verbose=verbose)

    comp_slice(edge, verbose=verbose)  # scan rows top-down, compare y-adjacent, x-overlapping Ps to form derPs
    # rng+ in comp_slice adds edge.node_T[0]:
    for fd, PP_ in enumerate(edge.node_T[0]):  # [rng+ PPm_,PPd_, der+ PPm_,PPd_]
        # sub+, intra PP:
        sub_recursion_eval(edge, PP_)
        # agg+, inter-PP, 1st layer is two forks only:
        if sum([PP.valt[fd] for PP in PP_]) > ave * sum([PP.rdnt[fd] for PP in PP_]):
            node_= []
            for PP in PP_: # CPP -> Cgraph:
                derH,valt,rdnt = PP.derH,PP.valt,PP.rdnt
                node_ += [Cgraph(ptuple=PP.ptuple, derH=[derH,valt,rdnt], valt=valt,rdnt=rdnt, L=len(PP.node_),
                                 box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box))]
                sum_derH([edge.derH,edge.valt,edge.rdnt], [derH,valt,rdnt], 0)
            edge.node_T[0][fd][:] = node_
            # node_[:] = new node_tt in the end:
            agg_recursion(edge, node_)


def non_max_suppression(blob):
    Y, X = blob.mask__.shape
    g__ = blob.der__t.g

    # compute direction of gradient
    with np.errstate(divide='ignore', invalid='ignore'):
        s__, c__ = [blob.der__t.dy, blob.der__t.dx] / g__

    # round angle to one of eight directions
    up__, lft__, dwn__, rgt__ = (s__ < -oct_sep), (c__ < -oct_sep), (s__ > oct_sep), (c__ > oct_sep)
    mdly__, mdlx__ = ~(up__ | dwn__), ~(lft__ | rgt__)

    # assign directions, reduced to four
    dir_mask___ = [
        mdly__ & (rgt__ | lft__), (dwn__ & rgt__) | (up__ & lft__),     #  0,  45 deg
        (dwn__ | up__) & mdlx__,  (dwn__ & lft__) | (up__ & rgt__),     # 90, 135 deg
    ]
    ryx_ = [(0, 1), (1, 1), (1, 0), (1, -1)]

    # for each direction, find local maximum by comparing with neighboring pixels
    max_mask__ = np.zeros_like(blob.mask__, dtype=bool)
    for dir_mask__, (ry, rx) in zip(dir_mask___, ryx_):
        # get indices of pixels in blob with corresponding direction
        mask__ = dir_mask__ & (~blob.mask__)    # and with blob mask
        y_, x_ = mask__.nonzero()

        # get neighbor pixel indices
        yn1_, xn1_ = y_ + ry, x_ + rx
        yn2_, xn2_ = y_ - ry, x_ - rx

        # choose valid neighbor indices
        valid1_ = (0 <= yn1_) & (yn1_ < Y) & (0 <= xn1_) & (xn1_ < X)
        valid2_ = (0 <= yn2_) & (yn2_ < Y) & (0 <= xn2_) & (xn2_ < X)

        # compare values
        not_max_ = np.zeros_like(y_, dtype=bool)
        not_max_[valid1_] |= (g__[y_[valid1_], x_[valid1_]] < g__[yn1_[valid1_], xn1_[valid1_]])
        not_max_[valid2_] |= (g__[y_[valid2_], x_[valid2_]] < g__[yn2_[valid2_], xn2_[valid2_]])

        # suppress non-maximum points
        mask__[y_[not_max_], x_[not_max_]] = False

        # add to max_mask__
        max_mask__ |= mask__

    return max_mask__


def slice_blob_ortho(blob, max_mask__, verbose=False):
    pass

def form_P(P, dir__t, mask__, axis):

    rdert_, dert_yx_ = [P.dert_[len(P.dert_)//2]],[P.yx]      # include pivot
    dert_olp_ = {(round(P.yx[0]), round(P.yx[1]))}
    rdert_,dert_yx_,dert_olp_ = scan_direction(rdert_,dert_yx_,dert_olp_, P.yx, axis, dir__t,mask__, fleft=1)  # scan left
    rdert_,dert_yx_,dert_olp_ = scan_direction(rdert_,dert_yx_,dert_olp_, P.yx, axis, dir__t,mask__, fleft=0)  # scan right
    # initialization
    rdert = rdert_[0]
    I, Dy, Dx, G = rdert; M=ave_g-G; dert_=[rdert]
    # accumulation:
    for rdert in rdert_[1:]:
        i, dy, dx, g = rdert
        I+=i; M+=ave_g-g; Dy+=dy; Dx+=dx
        dert_ += [rdert]
    L = len(dert_)
    P.dert_ = dert_; P.dert_yx_ = dert_yx_  # new dert and dert_yx
    P.yx = P.dert_yx_[L//2]              # new center
    G = np.hypot(Dy, Dx)  # recompute G
    P.ptuple = [I,G,M,[Dy,Dx], L]
    P.axis = axis
    P.dert_olp_ = dert_olp_
    return P

def scan_direction(rdert_,dert_yx_,dert_olp_, yx, axis, dir__t,mask__, fleft):  # leftward or rightward from y,x
    Y, X = mask__.shape # boundary
    y, x = yx
    sin,cos = axis      # unpack axis
    r = cos*y - sin*x   # from P line equation: cos*y - sin*x = r = constant
    _cy,_cx = round(y), round(x)  # keep previous cell
    y, x = (y-sin,x-cos) if fleft else (y+sin, x+cos)   # first dert position in the direction of axis
    while True:                   # start scanning, stop at boundary or edge of blob
        x0, y0 = int(x), int(y)   # floor
        x1, y1 = x0 + 1, y0 + 1   # ceiling
        if x0 < 0 or x1 >= X or y0 < 0 or y1 >= Y: break  # boundary check
        kernel = [  # cell weighing by inverse distance from float y,x:
            # https://www.researchgate.net/publication/241293868_A_study_of_sub-pixel_interpolation_algorithm_in_digital_speckle_correlation_method
            (y0, x0, (y1 - y) * (x1 - x)),
            (y0, x1, (y1 - y) * (x - x0)),
            (y1, x0, (y - y0) * (x1 - x)),
            (y1, x1, (y - y0) * (x - x0))]
        cy, cx = round(y), round(x)                         # nearest cell of (y, x)
        if mask__[cy, cx]: break                            # mask check of (y, x)
        if abs(cy-_cy) + abs(cx-_cx) == 2:                  # mask check of intermediate cell between (y, x) and (_y, _x)
            # Determine whether P goes above, below or crosses the middle point:
            my, mx = (_cy+cy) / 2, (_cx+cx) / 2             # Get middle point
            myc1 = sin * mx + r                             # my1: y at mx on P; myc1 = my1*cos
            myc = my*cos                                    # multiply by cos to avoid division
            if cos < 0: myc, myc1 = -myc, -myc1             # reverse sign for comparison because of cos
            if abs(myc-myc1) > 1e-5:                        # check whether myc!=myc1, taking precision error into account
                # y is reversed in image processing, so:
                # - myc1 > myc: P goes below the middle point
                # - myc1 < myc: P goes above the middle point
                # - myc1 = myc: P crosses the middle point, there's no intermediate cell
                ty, tx = (
                    ((_cy, cx) if _cy < cy else (cy, _cx))
                    if myc1 < myc else
                    ((_cy, cx) if _cy > cy else (cy, _cx))
                )
                if mask__[ty, tx]: break    # if the cell is masked, stop
                dert_olp_ |= {(ty,tx)}
        ptuple = [
            sum((par__[ky, kx] * dist for ky, kx, dist in kernel))
            for par__ in dir__t]
        dert_olp_ |= {(cy, cx)}  # add current cell to overlap
        _cy, _cx = cy, cx
        if fleft:
            rdert_ = [ptuple] + rdert_          # append left
            dert_yx_ = [(y,x)] + dert_yx_     # append left coords per dert
            y -= sin; x -= cos  # next y,x
        else:
            rdert_ = rdert_ + [ptuple]  # append right
            dert_yx_ = dert_yx_ + [(y,x)]
            y += sin; x += cos  # next y,x

    return rdert_,dert_yx_,dert_olp_


# -------------- Alternatives

def otsu(g_):
    # Mean and variance of g_, with probability distribution p_:
    # mu = sum([p*g for p, g in zip(p_, g_)])
    # var = sum([p*(g - mu)**2 for p, g in zip(p_, g_)])
    # Expand the terms:
    # var = sum([(p*g**2 - 2*p*g*mu + p*mu**2) for p, g in zip(p_, g_)])
    # Split the sum:
    # var = sum([p*g**2 for p, g in zip(p_, g_)])
    #     - 2*mu*sum([p*g for p, g in zip(p_, g_)])
    #     + mu**2*sum([p for p in p_])
    # Since p_ sums up to 1 and mu = sum([p*g for p, g in zip(p_, g_)]):
    # var = [p*g**2 for g, p in zip(g_, p_)] - 2*mu*mu + mu**2
    # Simplify:
    # var = [p*g**2 for g, p in zip(g_, p_)] - mu**2
    # Intra-class variance of Otsu's method:
    # var = var0*wei0 + var1*wei1
    # With:
    # var0 = sum([g**2 for g in g0_])/len(g0_) - (sum(g0_)/len(g0_))**2
    # Factor out (1/len(g0_)):
    # var0 = (sum([g**2 for g in g0_]) - sum(g0_)**2/len(g0_))/len(g0_)
    # Weight of class 0:
    # wei0 = len(g0_)/len(g_)
    # The product of the two:
    # var0*wei0 = (sum([g**2 for g in g0_]) - sum(g0_)**2/len(g0_))/len(g_)
    # Same with var1, wei1. Substitute var0, wei0, var1, wei1 into var and factor out (1/len(g_)):
    # var = (sum([g**2 for g in g0_]) + sum([g**2 for g in g1_])
    #      - sum(g0_)**2/len(g0_) - sum(g1_)**2/len(g1_)) / len(g_)
    # Simplify:
    # var = (sum([g**2 for g in g_]) - sum(g0_)**2/len(g0_) - sum(g1_)**2/len(g1_)) / len(g_)
    #
    # sum([g**2 for g in g_]) and len(g_) is the same for all classes. Therefore, we can simplify the formula:
    # ┌───────────────────────────────────────────────────┐
    # │ val = sum(g0_)**2/len(g0_) + sum(g1_)**2/len(g1_) │
    # └───────────────────────────────────────────────────┘
    # with val = sg_sqr - var * lg, where lg = len(g_), sg = sum([g*g for g in g_])
    # The threshold at which var is minimal is the threshold at which val is maximal.

    if len(g_) <= 1: return g_[0]
    g_ = np.sort(g_.reshape(-1))

    # lengths of g0_ and g1_ through all possible thresholds:
    lg0_ = np.arange(1, len(g_))
    lg1_ = len(g_) - lg0_

    # sums of g0_ and g1_ through all possible thresholds:
    sg0_ = np.cumsum(g_)[:-1]
    sg1_ = g_.sum() - sg0_

    # value for threshold selection:
    val_ = sg0_*sg0_/lg0_ + sg1_*sg1_/lg1_

    return g_[val_.argmax()]

# replace form_edge_ with comp_slice and form_PP_
