import sys
import numpy as np
from copy import copy, deepcopy
from itertools import product
from frame_blobs import recompute_dert, recompute_adert, CEdge
from .classes import CP, CPP, CderP, Cgraph
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
'''
convert to up_ and down_ for angle=0:
octants = lambda: [
    [sin < -0.38, cos < -0.38],
    [sin < -0.38, -0.38 ≤ cos ≤ 0.38],
    [sin < -0.38, cos > 0.38],
    [-0.38 ≤ sin ≤ 0.38, cos > 0.38],
    [sin > 0.38, cos > 0.38],
    [sin > 0.38, -0.38 ≤ cos ≤ 0.38],
    [sin > 0.38, cos < -0.38],
    [-0.38 ≤ sin ≤ 0.38, cos < -0.38]
]   '''
oct_sep = 0.3826834323650898


def vectorize_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    # convert Cblob to Cedge: (temporary)
    edge = CEdge(dir__t=blob.der__t, dir__t_roots=[[[] for col in row] for row in blob.der__t[0]], mask__=blob.mask__,
                 I=blob.I)

    slice_edge(edge, verbose)  # form 2D array of Ps: horizontal blob slices in dir__t
    rotate_P_(edge, verbose)  # re-form Ps around centers along P.G, P sides may overlap, if sum(P.M s + P.Ma s)?
    cP_ = set(edge.node_)  # to pop here
    while cP_:
        form_link_(cP_.pop(), cP_, edge)  # trace adjacent Ps, fill|prune if missing or redundant, add them to P.link_

    comp_slice(edge, verbose=verbose)  # scan rows top-down, compare y-adjacent, x-overlapping Ps to form derPs
    # rng+ in comp_slice adds edge.node_tt[0]:
    for fd, PP_ in enumerate(edge.node_[0]):  # [rng+ PPm_,PPd_, der+ PPm_,PPd_]
        # sub+, intra PP:
        sub_recursion_eval(edge, PP_)
        # agg+, inter-PP:
        if sum([PP.valt[fd] for PP in PP_]) > ave * sum([PP.rdnt[fd] for PP in PP_]):
            node_= []
            for PP in PP_:  # convert PPs to graphs:
                node_ += [Cgraph(ptuple=PP.ptuple, derH=[PP.derH, PP.valt, PP.rdnt], L=len(PP.node_), # empty aggH, etc.
                                 box=[(PP.box[0]+PP.box[1])/2, (PP.box[2]+PP.box[3])/2] + list(PP.box))]
                sum_derH([edge.derH, edge.valt, edge.rdnt], [PP.derH, PP.valt, PP.rdnt], 0)
            # node_: edge.node_tt[0][fd]
            agg_recursion(edge, node_)

'''
or only compute params needed for rotate_P_?
'''
def slice_edge(edge, verbose=False):  # form blob slices nearest to slice Ga: Ps, ~1D Ps, in select smooth edge (high G, low Ga) blobs

    P_ = []
    height, width = edge.mask__.shape

    for y in range(height):  # iterate through lines, each may have multiple slices -> Ps
        if verbose: print(f"\rConverting to image... Processing line {y + 1}/{height}", end=""); sys.stdout.flush()
        _mask = True  # mask -1st dert
        x = 0
        while x < width:  # iterate through pixels in a line
            mask = edge.mask__[y, x]
            dert = [par__[y, x] for par__ in edge.dir__t]   # exclude i
            g, ga, ri, dy, dx, dyy, dyx, dxy, dxx = dert
            if not mask:  # masks: if 0,_1: P initialization, if 0,_0: P accumulation, if 1,_0: P termination
                if _mask:  # ini P params with first unmasked dert
                    Pdert_ = [dert]
                    I = ri; M = ave_g - g; Ma = ave_ga - ga; Dy = dy; Dx = dx
                    Dyy, Dyx, Dxy, Dxx = dyy, dyx, dxy, dxx
                else:
                    # dert and _dert are not masked, accumulate P params:
                    I +=ri; M+=ave_g-g; Ma+=ave_ga-ga; Dy+=dy; Dx+=dx; Dyy+=dyy; Dyx+=dyx; Dxy+=dxy; Dxx+=dxx
                    Pdert_ += [dert]
            elif not _mask:
                # _dert is not masked, dert is masked, pack P:
                P_ += [term_P(I, M, Ma, Dy, Dx, Dyy, Dyx, Dxy, Dxx, y,x-1, Pdert_)]
            _mask = mask
            x += 1
        if not _mask:  # pack last P:
            P_ += [term_P(I, M, Ma, Dy, Dx, Dyy, Dyx, Dxy, Dxx, y,x-1, Pdert_)]

    if verbose: print("\r" + " " * 79, end=""); sys.stdout.flush(); print("\r", end="")
    edge.node_ = P_
    return P_

def term_P(I, M, Ma, Dy, Dx, Dyy, Dyx, Dxy, Dxx, y,x, Pdert_):

    G = recompute_dert(Dy, Dx)  # recompute G,Ga, it can't reconstruct M,Ma
    Ga, Dyy, Dyx, Dxy, Dxx = recompute_adert(Dyy, Dyx, Dxy, Dxx)
    L = len(Pdert_)  # params.valt = [params.M+params.Ma, params.G+params.Ga]?
    P = CP(ptuple=[I, G, Ga, M, Ma, [Dy, Dx], [Dyy, Dyx, Dxy, Dxx], L], dert_=Pdert_)
    P.dert_yx_ = [(y, kx) for kx in range(x-L+1, x+1)]  # +1 to compensate for x-1 in slice_blob
    P.yx = P.dert_yx_[L//2]
    P.dert_olp_ = set(P.dert_yx_)
    return P

def rotate_P_(edge, verbose=False):  # rotate each P to align it with direction of P or dert gradient

    dir__t = edge.dir__t; mask__= edge.mask__
    if verbose: i = 0
    P_ = []
    for P in edge.node_:
        G = P.ptuple[1]
        daxis = P.ptuple[5][0] / G  # dy: deviation from horizontal axis
        _daxis = 0
        if verbose: i += 1
        while abs(daxis) * G > ave_rotate:  # recursive reform P in blob.dir__t along new G angle:
            if verbose: print(f"\rRotating... {i}/{len(P_)}: {round(np.degrees(np.arctan2(*P.axis)))}°", end=" " * 79); sys.stdout.flush()
            _axis = P.axis
            P = form_P(P, dir__t, mask__, axis=np.divide(P.ptuple[5], np.hypot(*P.ptuple[5])))  # pivot to P angle
            maxis, daxis = comp_angle(_axis, P.axis)
            ddaxis = daxis +_daxis  # cancel-out if opposite-sign
            _daxis = daxis
            G = P.ptuple[1]  # rescan in the direction of ave_a, P.daxis if future reval:
            if ddaxis * G < ave_rotate:  # terminate if oscillation
                axis = np.add(_axis, P.axis)
                axis = np.divide(axis, np.hypot(*axis))  # normalize
                P = form_P(P, dir__t, mask__, axis=axis)  # not pivoting to dert G
                break
        for cy,cx in P.dert_olp_:  # assign roots in dir__t
            edge.dir__t_roots[cy][cx] += [P]  # final rotated P

        P_ += [P]
    edge.node_ = P_

    if verbose: print("\r", end=" " * 79); sys.stdout.flush(); print("\r", end="")

def form_P(P, dir__t, mask__, axis):

    rdert_, dert_yx_ = [P.dert_[len(P.dert_)//2]],[P.yx]      # include pivot
    dert_olp_ = {(round(P.yx[0]), round(P.yx[1]))}
    rdert_,dert_yx_,dert_olp_ = scan_direction(rdert_,dert_yx_,dert_olp_, P.yx, axis, dir__t,mask__, fleft=1)  # scan left
    rdert_,dert_yx_,dert_olp_ = scan_direction(rdert_,dert_yx_,dert_olp_, P.yx, axis, dir__t,mask__, fleft=0)  # scan right
    # initialization
    rdert = rdert_[0]
    G, Ga, I, Dy, Dx, Dyy, Dyx, Dxy, Dxx = rdert; M=ave_g-G; Ma=ave_ga-Ga; dert_=[rdert]
    # accumulation:
    for rdert in rdert_[1:]:
        g, ga, i, dy, dx, dyy, dyx, dxy, dxx = rdert
        I+=i; M+=ave_g-g; Ma+=ave_ga-ga; Dy+=dy; Dx+=dx; Dyy+=dyy; Dyx+=dyx; Dxy+=dxy; Dxx+=dxx
        dert_ += [rdert]
    L = len(dert_)
    P.dert_ = dert_; P.dert_yx_ = dert_yx_  # new dert and dert_yx
    P.yx = P.dert_yx_[L//2]              # new center
    G = recompute_dert(Dy, Dx); Ga, Dyy, Dyx, Dxy, Dxx = recompute_adert(Dyy, Dyx, Dxy, Dxx)  # recompute G,Ga
    P.ptuple = [I,G,Ga,M,Ma, [Dy,Dx], [Dyy,Dyx,Dxy,Dxx], L]
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

# draft
def form_link_(P, cP_, edge):  # trace adj Ps up and down by adj dert roots, fill|prune if missing or redundant, add to P.link_ if >ave*rdn
    Y, X = edge.mask__.shape

    # rim as a dictionary with key=(y,x) and value=roots
    rim_ = {(rim_y,rim_x):edge.dir__t_roots[rim_y][rim_x]                   # unique key-value pair
            for iy, ix in P.dert_olp_                                       # overlap derts
            for rim_y,rim_x in product(range(iy-1,iy+2),range(ix-1,ix+2))   # rim loop of iy, ix
            if (0 <= rim_y < Y) and (0 <= rim_x < X)                        # boundary check
            and not edge.mask__[rim_y,rim_x]}                               # blob boundary check
    # scan rim roots:
    link_ = {*sum(rim_.values(), start=[])} & cP_   # intersect with cP_ to prevent duplicate links and self linking (P not in cP_)
    # form links:
    for _P in link_:
        P.link_tH[-1][0] += [_P]
        _P.link_tH[-1][0] += [P]  # bidirectional assign maybe needed in ortho version, else uplinks only?
    # check empty link_:
    if not P.link_tH[-1][0]:
        # filter non-empty roots and get max-G dert coord:
        y, x = max([(y, x) for y, x in rim_ if not rim_[y, x]],     # filter non-empty roots
                     key=lambda yx: edge.dir__t[1][yx])             # get max-G dert coord
        # get max-G dert:
        dert = [par__[y,x] for par__ in edge.dir__t]       # get max-G dert
        # form new P
        _P = form_P(CP(dert, dert_=[dert], dert_yx_=[(y,x)], dert_olp_={(y,x)}, yx=(y, x)),
                    edge.dir__t, edge.mask__,
                    axis=np.divide(dert[3:5], dert[0]))
        # link _P:
        P.link_tH[-1][0] += [_P]; _P.link_tH[-1][0] += [P]    # form link with P first to avoid further recursion
        _cP_ = set(edge.node_) - {P}           # exclude P
        form_link_(_P, _cP_, edge)          # call form_link_ for the newly formed _P
        edge.node_ += [_P]                     # add _P to blob.P_ for further linking with remaining cP_


def slice_edge_ortho(edge, verbose=False):  # slice_blob with axis-orthogonal Ps

    Y, X = edge.mask__.shape
    yx_ = sort_cell_by_da(edge)
    idmap = np.full(edge.mask__.shape, UNFILLED, dtype=int)
    idmap[edge.mask__] = EXCLUDED_ID

    edge.P_ = []
    adj_pairs = set()
    for y, x in yx_:
        if idmap[y, x] == UNFILLED:
            # form P along axis
            dert = [par__[y, x] for par__ in edge.dir__t[1:]]
            axis = np.divide(dert[3:5], dert[0])

            P = form_P(
                CP(dert, dert_=[dert], dert_yx_=[(y,x)], dert_olp_={(y,x)}, yx=(y, x)),
                edge.dir__t, edge.mask__, axis=axis)
            edge.P_ += [P]

            for _y, _x in P.dert_olp_:
                for __y, __x in product(range(_y-1,y+2), range(x-1,x+2)):
                    if (0 <= __y < Y) and (0 <= __x < X) and idmap[__y, __x] not in (UNFILLED, EXCLUDED_ID):
                        adj_pairs.add((idmap[__y, __x], P.id))

            # check for adjacent Ps using idmap
            for _y, _x in P.dert_olp_:
                if idmap[_y, _x] == UNFILLED:
                    idmap[_y, _x] = P.id

    return adj_pairs

def sort_cell_by_da(edge):   # sort derts by angle deviation of derts
    with np.errstate(divide='ignore', invalid='ignore'):  # suppress numpy RuntimeWarning
        included = np.pad(~edge.mask__, 1, 'constant', constant_values=False)

        uv__ = np.pad(edge.dir__t[4:6] / edge.dir__t.g,
                      [(0, 0), (1, 1), (1, 1)], 'constant', constant_values=0)
        if np.isnan(uv__[:, included]).any():
            raise ValueError("g = 0 in edge blob")

        # Get angle deviation of derts
        rim_slices = (
            ks.tl, ks.tc, ks.tr,
            ks.ml, ks.mr,
            ks.bl, ks.bc, ks.br)
        rim___ = [included[sl] for sl in rim_slices]
        rim_da___ = [(1 - (uv__[ks.mc] * uv__[sl]).sum(axis=0))
                     for sl in rim_slices]

        # Set irrelevant rim = 0
        for rim__, rim_da__ in zip(rim___, rim_da___):
            rim_da__[~rim__] = 0

        # Get average angle deviation
        da__ = sum(rim_da___) / sum(rim___)

        # Get in-blob cells
        yx_ = sorted(zip(*np.indices(da__.shape)[:, included[ks.mc]]),
                     key=lambda yx: da__[yx[0], yx[1]])

    return yx_

def slice_blob_hough(blob, verbose=False):  # slice_blob with axis-orthogonal Ps, using Hough transform

    Y, X = blob.mask__.shape
    # Get thetas and positions:
    dy__, dx__ = blob.dir__t[4:6]  # Get blob derts' angle
    y__, x__ = np.indices((Y, X))  # Get blob derts' position
    theta__ = np.arctan2(dy__, dx__)  # Compute theta

    if verbose:
        step = 100 / (~blob.mask__).sum()  # progress % percent per pixel
        progress = 0.0; print(f"\rFilling... {round(progress)} %", end="");  sys.stdout.flush()
    # derts with same rho and theta lies on the same line
    # floodfill derts with similar rho and theta
    P_ = []
    filled = blob.mask__.copy()
    zy, zx = Y/2, X/2
    for y in y__[:, 0]:
        for x in x__[0]:
            # initialize P at first unfilled dert found
            if not filled[y, x]:
                M = 0; Ma = 0; I = 0; Dy = 0; Dx = 0; Dyy = 0; Dyx = 0; Dxy = 0; Dxx = 0
                dert_ = []
                dert_yx_ = []
                to_fill = [(y, x)]                  # dert indices to floodfill
                rt_olp__ = new_rt_olp_array((Y, X)) # overlap of rho theta (line) space
                while to_fill:                      # floodfill for one P
                    y2, x2 = to_fill.pop()          # get next dert index to fill
                    if x2 < 0 or x2 >= X or y2 < 0 or y2 >= Y:  # skip if out of bounds
                        continue
                    if filled[y2, x2]:              # skip if already filled
                        continue
                    # check if dert is almost on the same line and have similar gradient angle
                    new_rt_olp__ = hough_check(rt_olp__, y2, x2, theta__[y2, x2], zx, zy)
                    if not new_rt_olp__.any():
                        continue

                    filled[y2, x2] = True       # mark as filled
                    rt_olp__[:] = new_rt_olp__  # update overlap
                    # accumulate P params:
                    dert = tuple(param__[y2, x2] for param__ in blob.dir__t[1:])
                    g, ga, ri, dy, dx, dyy, dyx, dxy, dxx = dert  # skip i
                    M += ave_g - g; Ma += ave_ga - ga; I += ri; Dy+=dy; Dx+=dx; Dyy+=dyy; Dyx+=dyx; Dxy+=dxy; Dxx+=dxx
                    dert_ += [dert]  # unpack x, y, add dert to P
                    dert_yx_ += [(y2, x2)]

                    # add neighbors to fill
                    to_fill += [*product(range(y2-1, y2+2), range(x2-1, x2+2))]
                if not rt_olp__.any():
                    raise ValueError
                G = recompute_dert(Dy, Dx)
                Ga, Dyy, Dyx, Dxy, Dxx = recompute_adert(Dyy, Dyx, Dxy, Dxx)
                L = len(dert_)
                axis = (0, 1) if G == 0 else (Dy / G, Dx / G)
                P_ += [CP(ptuple=[I, M, Ma, [Dy, Dx], [Dyy, Dyx, Dxy, Dxx], G, Ga, L],
                          yx=sorted(dert_yx_, key=lambda yx: yx[1])[L//2], axis=axis,
                          dert_=dert_, dert_yx_=dert_yx_, dert_olp_=dert_yx_)]
                if verbose:
                    progress += L * step; print(f"\rFilling... {round(progress)} %", end=""); sys.stdout.flush()
    blob.P_ = P_
    if verbose: print("\r" + " " * 79, end=""); sys.stdout.flush(); print("\r", end="")
    return P_


def slice_blob_flow(blob, verbose=False):  # version of slice_blob_ortho

    # find the derts with gradient pointing at current dert:
    _yx_ = np.indices(blob.mask__.shape)[:, ~blob.mask__].T  # blob derts' position
    with np.errstate(divide='ignore', invalid='ignore'):  # suppress numpy RuntimeWarning
        sc_ = np.divide(blob.dir__t[4:6], blob.dir__t[1])[:, ~blob.mask__].T    # blob derts' angle
    uv_ = np.zeros_like(sc_)        # (u, v) points to one of the eight neighbor cells
    u_, v_ = uv_.T                  # unpack u, v
    s_, c_ = sc_.T                  # unpack sin, cos
    u_[oct_sep <= s_] = 1              # down, down left or down right
    u_[(-oct_sep < s_) & (s_ < oct_sep)] = 0  # left or right
    u_[s_ <= -oct_sep] = -1              # up, up-left or up-right
    v_[oct_sep <= c_] = 1              # right, up-right or down-right
    v_[(-oct_sep < c_) & (c_ < oct_sep)] = 0  # up or down
    v_[c_ <= -oct_sep] = -1              # left, up-left or down-left
    yx_ = _yx_ + uv_                # compute target cell position
    m__ = (yx_.reshape(-1, 1, 2) == _yx_).all(axis=2)   # mapping from _yx_ to yx_
    def get_p(a):
        nz = a.nonzero()[0]
        if len(nz) == 0:    return -1
        elif len(nz) == 1:  return nz[0]
        else:               raise ValueError
    p_ = [*map(get_p, m__)]       # reduced mapping from _yx_ to yx_
    n_ = m__.sum(axis=0) # find n, number of gradient sources per cell

    # cluster Ps, start from cells without any gradient source
    blob.P_ = []
    for i in range(len(n_)):
        if n_[i] == 0:                  # start from cell without any gradient source
            I = 0; M = 0; Ma = 0; Dy = 0; Dx = 0; Dyy = 0; Dyx = 0; Dxy = 0; Dxx = 0
            dert_ = []
            dert_yx_ = []
            j = i
            while True:      # while there is a dert to follow
                y, x = _yx_[j]      # get dert position
                dert = tuple(par__[y, x] for par__ in blob.dir__t[1:])  # dert params at _y, _x, skip i
                g, ga, ri, dy, dx, dyy, dyx, dxy, dxx = dert
                I+=i; M+=ave_g-g; Ma+=ave_ga-ga; Dy+=dy; Dx+=dx; Dyy+=dyy; Dyx+=dyx; Dxy+=dxy; Dxx+=dxx
                dert_ += [dert]
                dert_yx_ += [(y, x)]

                # remove all gradient sources from the cell
                while True:
                    try:
                        k = p_.index(j)
                        p_[k] = -1
                    except ValueError as e:
                        if "is not in list" not in str(e):
                            raise e
                        break
                if p_[j] != -1:
                    j = p_[j]
                else:
                    break
            G = recompute_dert(Dy, Dx); Ga, Dyy, Dyx, Dxy, Dxx = recompute_adert(Dyy, Dyx, Dxy, Dxx)
            L = len(dert_)
            blob.P_ += [CP(ptuple=[I,M,Ma,G,Ga,[Dy,Dx],[Dyy,Dyx,Dxy,Dxx],L],
                           yx=dert_yx_[L//2], axis=(Dy/G, Dx/G),
                           dert_=dert_, dert_yx_=dert_yx_, dert_olp_=dert_yx_)]

    return blob.P_

def append_P(P__, P):  # pack P into P__ in top down sequence

    current_ys = [P_[0].y0 for P_ in P__]  # list of current-layer seg rows
    if P.y0 in current_ys:
        if P not in P__[current_ys.index(P.y0)]:
            P__[current_ys.index(P.y0)].append(P)  # append P row
    elif P.y0 > current_ys[0]:  # P.y0 > largest y in ys
        P__.insert(0, [P])
    elif P.y0 < current_ys[-1]:  # P.y0 < smallest y in ys
        P__.append([P])
    elif P.y0 < current_ys[0] and P.y0 > current_ys[-1]:  # P.y0 in between largest and smallest value
        for i, y in enumerate(current_ys):  # insert y if > next y
            if P.y0 > y: P__.insert(i, [P])  # PP.P__.insert(P.y0 - current_ys[-1], [P])


def copy_P(P, Ptype=None):  # Ptype =0: P is CP | =1: P is CderP | =2: P is CPP | =3: P is CderPP | =4: P is CaggPP

    if not Ptype:  # assign Ptype based on instance type if no input type is provided
        if isinstance(P, CPP):     Ptype = 2
        elif isinstance(P, CderP): Ptype = 1
        else:                      Ptype = 0  # CP

    uplink_layers, downlink_layers = P.uplink_layers, P.downlink_layers  # local copy of link layers
    P.uplink_layers, P.downlink_layers = [], []  # reset link layers
    roott = P.roott  # local copy
    P.roott = [None, None]
    if Ptype == 1:
        P_derP, _P_derP = P.P, P._P  # local copy of derP.P and derP._P
        P.P, P._P = None, None  # reset
    elif Ptype == 2:
        mseg_levels, dseg_levels = P.mseg_levels, P.dseg_levels
        P__ = P.P__
        P.mseg_levels, P.dseg_levels, P.P__ = [], [], []  # reset
    elif Ptype == 3:
        PP_derP, _PP_derP = P.PP, P._PP  # local copy of derP.P and derP._P
        P.PP, P._PP = None, None  # reset
    elif Ptype == 4:
        gPP_, cPP_ = P.gPP_, P.cPP_
        mlevels, dlevels = P.mlevels, P.dlevels
        P.gPP_, P.cPP_, P, P.mlevels, P.dlevels = [], [], [], []  # reset

    new_P = deepcopy(P)  # copy P with empty root and link layers, reassign link layers:
    new_P.uplink_layers += copy(uplink_layers)
    new_P.downlink_layers += copy(downlink_layers)

    # shallow copy to create new list
    P.uplink_layers, P.downlink_layers = copy(uplink_layers), copy(downlink_layers)  # reassign link layers
    P.roott = roott  # reassign root
    # reassign other list params
    if Ptype == 1:
        new_P.P, new_P._P = P_derP, _P_derP
        P.P, P._P = P_derP, _P_derP
    elif Ptype == 2:
        P.mseg_levels, P.dseg_levels = mseg_levels, dseg_levels
        new_P.P__ = copy(P__)
        new_P.mseg_levels, new_P.dseg_levels = copy(mseg_levels), copy(dseg_levels)
    elif Ptype == 3:
        new_P.PP, new_P._PP = PP_derP, _PP_derP
        P.PP, P._PP = PP_derP, _PP_derP
    elif Ptype == 4:
        P.gPP_, P.cPP_ = gPP_, cPP_
        P.roott = roott
        new_P.gPP_, new_P.cPP_ = [], []
        new_P.mlevels, new_P.dlevels = copy(mlevels), copy(dlevels)

    return new_P