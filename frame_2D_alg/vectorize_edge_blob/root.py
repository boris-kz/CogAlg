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

    max_mask__ = non_max_suppression(blob)  # mask of local directional maxima of dy, dx, g
    import matplotlib.pyplot as plt
    Y, X = blob.mask__.shape
    p, q = (2, 1) if X > Y else (1, 2)
    plt.subplot(p, q, 1)
    plt.imshow(~blob.mask__, cmap='gray')
    plt.subplot(p, q, 2)
    plt.imshow(max_mask__, cmap='gray')
    plt.show()
    # Otsu's method to determine ave: https://en.wikipedia.org/wiki/Otsu%27s_method
    # ave = otsu(blob.der__t, blob.der__t.g[max_mask__])
    # st_mask__ = (ave - max_der__t.g > 0) & max_mask__   # mask of strong edges
    # wk_mask__ = ((ave/2) - max_der__t.g > 0) & max_mask__   # mask of weak edges
    #
    # # Edge tracking by hysteresis, forming edge structure:
    # edge_ = form_edge_(st_mask__, wk_mask__)
    #
    # comp_slice(edge_, verbose=verbose)  # scan rows top-down, compare y-adjacent, x-overlapping Ps to form derPs
    # # rng+ in comp_slice adds edge.node_T[0]:
    # for edge in edge_:
    #     for fd, PP_ in enumerate(edge.node_T[0]):  # [rng+ PPm_,PPd_, der+ PPm_,PPd_]
    #         # sub+, intra PP:
    #         sub_recursion_eval(edge, PP_)
    #         # agg+, inter-PP, 1st layer is two forks only:
    #         if sum([PP.valt[fd] for PP in PP_]) > ave * sum([PP.rdnt[fd] for PP in PP_]):
    #             node_= []
    #             for PP in PP_: # CPP -> Cgraph:
    #                 derH,valt,rdnt = PP.derH,PP.valt,PP.rdnt
    #                 node_ += [Cgraph(ptuple=PP.ptuple, derH=[derH,valt,rdnt], valt=valt,rdnt=rdnt, L=len(PP.node_),
    #                                  box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box))]
    #                 sum_derH([edge.derH,edge.valt,edge.rdnt], [derH,valt,rdnt], 0)
    #             edge.node_T[0][fd][:] = node_
    #             # node_[:] = new node_tt in the end:
    #             agg_recursion(edge, node_)


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


def otsu(der__t, mask__):
    pass

def form_edge_(sedge_mask__, wedge_mask__):
    pass