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

    max_der__t = non_max_suppression(blob)  # local max of dy, dx, g
    # Otsu's method to determine ave: https://en.wikipedia.org/wiki/Otsu%27s_method
    ave = otsu(max_der__t.g)
    max_mask__ = ave - max_der__t.g > 0   # mask of strong edges
    suppressed_mask__ = (ave/2) - max_der__t.g > 0   # mask of weak edges

    # Edge tracking by hysteresis, forming edge structure:
    edge_ = form_edge_(max_mask__, suppressed_mask__)

    comp_slice(edge_, verbose=verbose)  # scan rows top-down, compare y-adjacent, x-overlapping Ps to form derPs
    # rng+ in comp_slice adds edge.node_T[0]:
    for edge in edge_:
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
    pass

def otsu(g):
    pass

def form_edge_(sedge_mask__, wedge_mask__):
    pass
