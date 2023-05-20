# warnings.filterwarnings('error')
# import warnings  # to detect overflow issue, in case of infinity loop
from itertools import zip_longest
import sys
import numpy as np
from copy import copy, deepcopy
from .classes import Cptuple, CP, CPP, CderP
from .filters import ave, ave_g, ave_ga, ave_rotate
from .comp_slice import comp_slice, comp_angle
from .agg_convert import agg_recursion_eval
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

def vectorize_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    slice_blob(blob, verbose=verbose)  # form 2D array of Ps: horizontal blob slices in dert__
    # if Daxis?:
    rotate_P_(blob)  # re-form Ps around centers along P.G, P sides may overlap
    # if sum(P.M s + P.Ma s)?:
    comp_slice(blob, verbose=verbose)  # scan rows top-down, compare y-adjacent, x-overlapping Ps to form derPs
    # re compare PPs, cluster in graphs:
    for fd, PP_ in enumerate([blob.PPm_, blob.PPd_]):
        sub_recursion_eval(blob, PP_, fd=fd)  # intra PP
        # no feedback to blob?
        if sum([PP.valt[fd] for PP in PP_]) > ave * sum([PP.rdnt[fd] for PP in PP_]):
            agg_recursion_eval(blob, copy(PP_), fd=fd)  # comp sub_PPs, form intermediate PPs

'''
this is too involved for initial Ps, most of that is only needed for final rotated Ps?
'''
def slice_blob(blob, verbose=False):  # form blob slices nearest to slice Ga: Ps, ~1D Ps, in select smooth edge (high G, low Ga) blobs

    mask__ = blob.mask__  # same as positive sign here
    dert__ = zip(*blob.dert__)  # convert 10-tuple of 2D arrays into 1D array of 10-tuple blob rows
    dert__ = [zip(*dert_) for dert_ in dert__]  # convert 1D array of 10-tuple rows into 2D array of 10-tuples per blob
    P__ = []
    height, width = mask__.shape
    if verbose: print("Converting to image...")

    for y, (dert_, mask_) in enumerate(zip(dert__, mask__)):  # unpack lines, each may have multiple slices -> Ps:
        if verbose: print(f"\rConverting to image... Processing line {y + 1}/{height}", end=""); sys.stdout.flush()
        P_ = []
        _mask = True  # mask -1st dert
        for x, (dert, mask) in enumerate(zip(dert_, mask_)):
            g, ga, ri, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = dert[1:]  # skip i
            if not mask:  # masks: if 0,_1: P initialization, if 0,_0: P accumulation, if 1,_0: P termination
                if _mask:  # ini P params with first unmasked dert
                    Pdert_ = [dert]
                    params = Cptuple(I=ri,M=ave_g-g,Ma=ave_ga-ga, angle=[dy,dx], aangle=[sin_da0, cos_da0, sin_da1, cos_da1])
                else:
                    # dert and _dert are not masked, accumulate P params:
                    params.M+=ave_g-g; params.Ma+=ave_ga-ga; params.I+=ri; params.angle[0]+=dy; params.angle[1]+=dx
                    params.aangle = [_par+par for _par,par in zip(params.aangle,[sin_da0,cos_da0,sin_da1,cos_da1])]
                    Pdert_ += [dert]
            elif not _mask:
                # _dert is not masked, dert is masked, terminate P:
                params.G = np.hypot(*params.angle)  # Dy,Dx  # recompute G,Ga, it can't reconstruct M,Ma
                params.Ga = (params.aangle[1]+1) + (params.aangle[3]+1)  # Cos_da0, Cos_da1
                L = len(Pdert_)
                params.L = L; params.x = x-L/2  # params.valt = [params.M+params.Ma, params.G+params.Ga]
                P_+=[CP(ptuple=params, x0=x-L, y0=y, dert_=Pdert_)]
            _mask = mask
        # pack last P, same as above:
        if not _mask:
            params.G = np.hypot(*params.angle); params.Ga = (params.aangle[1]+1) + (params.aangle[3]+1)
            L = len(Pdert_); params.L = L; params.x = x-L/2  # params.valt=[params.M+params.Ma,params.G+params.Ga]
            P_ += [CP(ptuple=params, x0=x-L, y0=y, dert_=Pdert_)]
        P__ += [P_]

    if verbose: print("\r", end="")
    blob.P__ = P__
    return P__


def rotate_P_(blob):  # rotate each P to align it with direction of P gradient

    P__, dert__, mask__ = blob.P__, blob.dert__, blob.mask__

    yn, xn = dert__[0].shape[:2]
    for P_ in P__:
        for P in P_:
            daxis = P.ptuple.angle[0] / P.ptuple.L  # dy: deviation from horizontal axis
            # recursive reform P along new G angle in blob.dert__:
            # P.daxis for future reval?
            while P.ptuple.G * abs(daxis) > ave_rotate:
                rotate_P(P, dert__, mask__, yn, xn)
                maxis, daxis = comp_angle(P.ptuple.angle, P.axis)


def rotate_P(P, dert__t, mask__, yn, xn):

    L = len(P.dert_)
    rdert_ = [P.dert_[L//2]]  # init rotated dert_ with old central dert
    ycenter = P.y0  # P center coords
    xcenter = P.x0 + L//2
    sin = P.ptuple.angle[0] / P.ptuple.G
    cos = P.ptuple.angle[1] / P.ptuple.G
    if cos < 0: sin,cos = -sin,-cos
    # dx always >= 0, dy can be < 0
    assert abs(sin**2 + cos**2 - 1) < 1e-5  # hypot(dy,dx)=1: each dx,dy adds one rotated dert|pixel to rdert_
    P.axis = (sin,cos)  # last P angle
    # scan left:
    rx=xcenter; ry=ycenter
    while True:  # terminating condition is in form_rdert()
        rdert = form_rdert(rx,ry, dert__t, mask__)
        if rdert is None: break  # dert is not in blob: masked or out of bound
        rdert_ = [rdert] + rdert_   # add to left
        rx-=cos; ry-=sin  # next rx,ry
    P.x0 = rx; yleft = ry
    # scan right:
    rx=xcenter; ry=ycenter
    while True:
        rdert = form_rdert(rx,ry, dert__t, mask__)
        if rdert is None: break  # dert is not in blob: masked or out of bound
        rdert_ += [rdert]
        rx+=cos; ry+=sin  # next rx,ry
    P.y0 = min(yleft, ry) # P may go up-right or down-right
    # form rP:
    # initialization:
    rdert = rdert_[0]; _, G, Ga, I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1 = rdert; M=ave_g-G; Ma=ave_ga-Ga; dert_=[rdert]
    # accumulation:
    for rdert in rdert_[1:]:
        _, g, ga, i, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = rdert
        I+=i; M+=ave_g-g; Ma+=ave_ga-ga; Dy+=dy; Dx+=dx; Sin_da0+=sin_da0; Cos_da0+=cos_da0; Sin_da1+=sin_da1; Cos_da1+=cos_da1
        dert_ += [rdert]
    # re-form gradients:
    G = np.hypot(Dy,Dx);  Ga = (Cos_da0 + 1) + (Cos_da1 + 1)
    ptuple = Cptuple(I=I, M=M, G=G, Ma=Ma, Ga=Ga, angle=(Dy,Dx), aangle=(Sin_da0, Cos_da0, Sin_da1, Cos_da1))
    # replace P:
    P.ptuple = ptuple; P.dert_ = dert_


def form_rdert(rx,ry, dert__t, mask__):

    Y, X = mask__.shape
    # coord, distance of four int-coord derts, overlaid by float-coord rdert in dert__, int for indexing
    # always in dert__ for intermediate float rx,ry:
    x1 = int(np.floor(rx)); dx1 = abs(rx - x1)
    x2 = int(np.ceil(rx));  dx2 = abs(rx - x2)
    y1 = int(np.floor(ry)); dy1 = abs(ry - y1)
    y2 = int(np.ceil(ry));  dy2 = abs(ry - y2)
    # terminate scan_left | scan_right:
    if (x1 < 0 or x1 >= X or x2 < 0 or x2 >= X) or (y1 < 0 or y1 >= Y or y2 < 0 or y2 >= Y):
        return None
    # scale all dert params in proportion to inverted distance from rdert, sum(distances) = 1
    # approximation, square of rpixel is rotated, won't fully match not-rotated derts
    k1 = 1 - np.hypot(dx1, dy1)
    k2 = 1 - np.hypot(dx1, dy2)
    k3 = 1 - np.hypot(dx2, dy1)
    k4 = 1 - np.hypot(dx2, dy2)
    K = k1 + k2 + k3 + k4
    mask = (
        mask__[y1, x1] * k1 +
        mask__[y2, x1] * k2 +
        mask__[y1, x2] * k3 +
        mask__[y2, x2] * k4) / K

    if round(mask):  # summed mask is fractional, round to 1|0
        return None  # return rdert if inside the blob
    ptuple = []
    for dert__ in dert__t:  # 10 params in dert: i, g, ga, ri, dy, dx, day0, dax0, day1, dax1
        param = (
            dert__[y1, x1] * k1 +
            dert__[y2, x1] * k2 +
            dert__[y1, x2] * k3 +
            dert__[y2, x2] * k4) / K
        ptuple += [param]

    return ptuple


# slice_blob with axis-orthogonal Ps, but P centers may overlap or be missed?
def slice_blob_ortho(blob):

    P_ = []
    while blob.dert__:
        dert = blob.dert__.pop()
        P = CP(dert_= [dert])  # init cross-P per dert
        # need to find/combine adjacent _dert in the direction of gradient:
        _dert = blob.dert__.pop()
        mangle,dangle = comp_angle(dert.angle, _dert.angle)
        if mangle > ave:
            P.dert_ += [_dert]  # also sum ptuple, etc.
        else:
            P_ += [P]
            P = CP(dert_= [_dert])  # init cross-P per missing dert
            # add recursive function to find/combine adjacent _dert in the direction of gradient:
            _dert = blob.dert__.pop()
            mangle, dangle = comp_angle(dert.angle, _dert.angle)
            if mangle > ave:
                P.dert_ += [_dert]  # also sum ptuple, etc.
            else:
                pass  # add recursive slice_blob


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