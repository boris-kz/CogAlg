import sys
import numpy as np
from collections import namedtuple, deque, defaultdict
from itertools import product
from frame_blobs import Tdert
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

Tptuple = namedtuple("Tptuple", "I Dy Dx G M L")

oct_sep = 0.3826834323650898

def vectorize_root(blob, verbose=False):

    max_mask__ = max_selection(blob)  # mask of local directional maxima of dy, dx, g

    # form slices (Ps) from max_mask__ and form links by tracing max_mask__:
    slice_blob_ortho(blob, max_mask__, verbose=verbose)

    form_link_(blob, max_mask__)

    '''
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
    '''

def max_selection(blob):
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

    max_mask__ = np.zeros_like(blob.mask__, dtype=bool)
    # local max by comparing neighboring pixels per direction:
    for dir_mask__, (ry, rx) in zip(dir_mask___, ryx_):
        # direction pixels AND blob mask:
        mask__ = dir_mask__ & blob.mask__
        y_, x_ = mask__.nonzero()
        # neighbors:
        yn1_, xn1_ = y_ + ry, x_ + rx
        yn2_, xn2_ = y_ - ry, x_ - rx
        # computed vals:
        valid1_ = (0 <= yn1_) & (yn1_ < Y) & (0 <= xn1_) & (xn1_ < X)
        valid2_ = (0 <= yn2_) & (yn2_ < Y) & (0 <= xn2_) & (xn2_ < X)

        # compare values
        not_max_ = np.zeros_like(y_, dtype=bool)
        not_max_[valid1_] |= (g__[y_[valid1_], x_[valid1_]] < g__[yn1_[valid1_], xn1_[valid1_]])
        not_max_[valid2_] |= (g__[y_[valid2_], x_[valid2_]] < g__[yn2_[valid2_], xn2_[valid2_]])
        # select maxes
        mask__[y_[not_max_], x_[not_max_]] = False
        # add to max_mask__
        max_mask__ |= mask__

    return max_mask__


def slice_blob_ortho(blob, mask__, verbose=False):

    y_, x_ = mask__.nonzero()
    der_t = blob.der__t.get_pixel(y_, x_)
    deryx_ = sorted(zip(y_, x_, *der_t), key=lambda t: t[-1]) # sort by g
    filled = set()
    if verbose:
        step = 100 / len(deryx_)  # progress % percent per pixel
        progress = 0.0; print(f"\rSlicing... {round(progress)} %", end="");  sys.stdout.flush()

    for y, x, dy, dx, g in deryx_:
        i = blob.i__[blob.ibox.slice()][y, x]
        assert g > 0, "g must be positive"
        P = form_P(CP(yx=(y, x), axis=(dy/g, dx/g), dert_olp_={(y,x)}, dert_=[(y, x, i, dy, dx, g)]), blob)
        # exclude >=50% overlap:
        if len(filled & P.dert_olp_) / len(P.dert_olp_) >= 0.5:
            continue
        filled.update(P.dert_olp_)
        blob.P_ += [P]

        if verbose:
            progress += step; print(f"\rSlicing... {round(progress)} %", end=""); sys.stdout.flush()
    if verbose: print("\r" + " " * 79, end=""); sys.stdout.flush(); print("\r", end="")


def form_P(P, blob):

    scan_direction(P, blob, fleft=1)  # scan left
    scan_direction(P, blob, fleft=0)  # scan right
    # init:
    _, _, I, Dy, Dx, G = map(sum, zip(*P.dert_))
    L = len(P.dert_)
    M = ave_g*L - G
    G = np.hypot(Dy, Dx)  # recompute G
    P.ptuple = Tptuple(I, Dy, Dx, G, M, L)
    P.yx = P.dert_[L//2][:2]  # new center

    return P

def scan_direction(P, blob, fleft):  # leftward or rightward from y,x

    Y, X = blob.mask__.shape    # boundary
    sin,cos = _dy,_dx = P.axis  # unpack axis
    _y, _x = P.yx               # start with pivot
    r = cos*_y - sin*_x   # from P line equation: cos*y - sin*x = r = constant
    _cy,_cx = round(_y), round(_x)  # keep previous cell
    y, x = (_y-sin,_x-cos) if fleft else (_y+sin, _x+cos)   # first dert position in the direction of axis
    while True:                   # start scanning, stop at boundary or edge of blob
        x0, y0 = int(x), int(y)   # floor
        x1, y1 = x0 + 1, y0 + 1   # ceiling
        if x0 < 0 or x1 >= X or y0 < 0 or y1 >= Y: break   # boundary check
        kernel = [  # cell weighing by inverse distance from float y,x:
            # https://www.researchgate.net/publication/241293868_A_study_of_sub-pixel_interpolation_algorithm_in_digital_speckle_correlation_method
            (y0, x0, (y1 - y) * (x1 - x)),
            (y0, x1, (y1 - y) * (x - x0)),
            (y1, x0, (y - y0) * (x1 - x)),
            (y1, x1, (y - y0) * (x - x0))]
        cy, cx = round(y), round(x)  # nearest cell of (y, x)
        if not blob.mask__[cy, cx]:
            break
        if abs(cy-_cy) + abs(cx-_cx) == 2:  # mask check of intermediate cell between (y, x) and (_y, _x)
            my = (_cy+cy) / 2
            mx = (_cx+cx) / 2    # cell midpoint, P axis may be above, below or over
            _myc = sin * mx + r  # y at mx in P; myc1 = my1*cos
            myc = my * cos       # new cell, multiply by cos to avoid division
            if cos < 0: myc, _myc = -myc, -_myc   # reverse sign for comparison because of cos
            if abs(myc-_myc) > 1e-5:
                # deviation from P axis, y is reversed in image processing:
                # myc1 > myc: above axis, myc1 < myc: below axis, myc1 = myc: over axis, no intermediate cell
                ty, tx = (
                    ((_cy, cx) if _cy < cy else (cy, _cx))
                    if _myc < myc else
                    ((_cy, cx) if _cy > cy else (cy, _cx))
                )
                if not blob.mask__[ty, tx]: break    # if the cell is masked, stop
                P.dert_olp_ |= {(ty,tx)}

        ider__t = (blob.i__[blob.ibox.slice()],) + blob.der__t
        dert = (sum((par__[ky, kx] * dist for ky, kx, dist in kernel)) for par__ in ider__t)
        i,dy,dx,g = dert
        mangle,dangle = comp_angle((_dy,_dx), (dy, dx))
        if mangle < 0:  # terminate P if angle miss
            break
        P.dert_olp_ |= {(cy, cx)}  # add current cell to overlap
        _cy, _cx, _dy, _dx = cy, cx, dy, dx
        if fleft:
            P.dert_ = [[y,x,*dert]] + P.dert_  # append left
            y -= sin; x -= cos  # next y,x
        else:
            P.dert_ = P.dert_ + [[y,x,*dert]]  # append right
            y += sin; x += cos  # next y,x

# not revised:
def form_link_(blob, mask__):

    max_ = set(zip(*mask__.nonzero()))  # mask__ coordinates

    dert_root_ = defaultdict(set)
    for P in blob.P_:
        for y, x in P.dert_olp_ & max_:
            dert_root_[y, x].add(P)

    # trace edge from each P
    blob.P_link_ = set()    # clear P_link_
    for P in blob.P_:
        traceq_ = deque(P.dert_olp_ & max_)  # start with dert_olp_ & max_
        traced_ = set(traceq_)
        while traceq_:   # trace adjacent through max_
            _y, _x = traceq_.popleft()
            # check for root
            stop = False
            for _P in (dert_root_[_y, _x] - {P}):
                link = (P, _P) if P.id < _P.id else (_P, P)
                blob.P_link_.add(link)
                stop = True     # stop when a root is reached
            if not stop:    # continue
                yx_ = {*product(range(_y-1,_y+2), range(_x-1,_x+2))}
                yx_ = (yx_ & max_) - traced_
                traceq_.extend(yx_)
                traced_.add(yx_)