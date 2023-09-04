import sys
import numpy as np
from collections import namedtuple, deque, defaultdict
from itertools import product
from .slice_edge import slice_edge
from .classes import CEdge, CP, CPP, CderP, Cgraph
from .filters import ave, ave_g, ave_dangle
from .comp_slice import comp_slice, comp_angle, sum_derH
from .hough_P import new_rt_olp_array, hough_check
from .agg_recursion import agg_recursion, sum_aggH
from .sub_recursion import sub_recursion_eval

'''
Vectorize is a terminal fork of intra_blob.

comp_slice traces edge axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These are low-M high-Ma blobs, vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)
-
Vectorization is clustering of parameterized Ps + their derivatives (derPs) into PPs: patterns of Ps that describe edge blob.
This process is a reduced-dimensionality (2D->1D) version of cross-comp and clustering cycle, common across this project.
As we add higher dimensions (3D and time), this dimensionality reduction is done in salient high-aspect blobs
(likely edges in 2D or surfaces in 3D) to form more compressed "skeletal" representations of full-dimensional patterns.
'''


def vectorize_root(edge, verbose=False):

    slice_edge(edge, verbose=False)
    # not revised:
    comp_slice(edge, verbose=verbose)  # scan rows top-down, compare y-adjacent, x-overlapping Ps to form derPs
    # rng+ in comp_slice adds edge.node_T[0]:
    for fd, PP_ in enumerate(edge.node_tt[0]):  # [rng+ PPm_,PPd_, der+ PPm_,PPd_]
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
            edge.node_tt[0][fd][:] = node_
            # node_[:] = new node_tt in the end:
            agg_recursion(edge, node_)
