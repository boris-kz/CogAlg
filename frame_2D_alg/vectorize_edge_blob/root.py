# warnings.filterwarnings('error')
# import warnings  # to detect overflow issue, in case of infinity loop
'''
Comp_slice is a terminal fork of intra_blob.
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
import sys
import numpy as np
from itertools import zip_longest
from copy import copy

from .classes import CQ, Cptuple, CP, CderP, CPP
from .filters import (
    aves, vaves, PP_vars, PP_aves,
    ave_inv, ave, ave_g, ave_ga,
    flip_ave, flip_ave_FPP,
    div_ave,
    ave_rmP, ave_ortho, aveB, ave_dI, ave_M, ave_Ma, ave_G, ave_Ga, ave_L,
    ave_x, ave_dx, ave_dy, ave_daxis, ave_dangle, ave_daangle,
    ave_mval, ave_mPP, ave_dPP, ave_splice,
    ave_nsub, ave_sub, ave_agg, ave_overlap, ave_rotate,
    med_decay,
)
from .comp_slice import comp_slice

def vectorize_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    slice_blob(blob, verbose=False)  # form 2D array of Ps: horizontal blob slices in dert__
    rotate_P_(blob)  # reform Ps around centers along G, sides may overlap
    comp_slice(blob, verbose=verbose)  # scan rows top-down, comp y-adjacent, x-overlapping Ps to form derPs

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
        P_ = []
        _mask = True  # mask the cell before 1st dert
        for x, (dert, mask) in enumerate(zip(dert_, mask_)):
            if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()

            g, ga, ri, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = dert[1:]  # skip i
            if not mask:  # masks: if 0,_1: P initialization, if 0,_0: P accumulation, if 1,_0: P termination
                if _mask:  # ini P params with first unmasked dert
                    Pdert_ = [dert]
                    params = Cptuple(M=ave_g-g,Ma=ave_ga-ga,I=ri, angle=[dy,dx], aangle=[sin_da0, cos_da0, sin_da1, cos_da1])
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
                P_+=[CP(ptuple=params, x0=x-(L-1), y0=y, dert_=Pdert_)]
            _mask = mask
        # pack last P, same as above:
        if not _mask:
            params.G = np.hypot(*params.angle); params.Ga = (params.aangle[1]+1) + (params.aangle[3]+1)
            L = len(Pdert_); params.L = L; params.x = x-L/2  # params.valt=[params.M+params.Ma,params.G+params.Ga]
            P_ += [CP(ptuple=params, x0=x-(L-1), y0=y, dert_=Pdert_)]
        P__ += [P_]

    blob.P__ = P__
    return P__


def rotate_P_(blob):  # rotate each P to align it with direction of P gradient

    P__, dert__, mask__ = blob.P__, blob.dert__, blob.mask__

    yn, xn = dert__[0].shape[:2]
    for P_ in P__:
        for P in P_:
            daxis = P.ptuple.angle[0] / P.ptuple.L  # dy: deviation from horizontal axis
            if P.ptuple.G * abs(daxis) > ave_rotate:
                P.ptuple.axis = P.ptuple.angle
                rotate_P(P, dert__, mask__, yn, xn)  # recursive reform P along new axis in blob.dert__
                _, daxis = comp_angle(P.ptuple.axis, P.ptuple.angle)
            # store P.daxis to adjust params?

def rotate_P(P, dert__t, mask__, yn, xn):

    L = len(P.dert_)
    rdert_ = [P.dert_[int(L/2)]]  # init rotated dert_ with old central dert

    ycenter = int(P.y0 + P.ptuple.axis[0]/2)  # can be negative
    xcenter = int(P.x0 + abs(P.ptuple.axis[1]/2))  # always positive
    Dy, Dx = P.ptuple.angle
    dy = Dy/L; dx = abs(Dx/L)  # hypot(dy,dx)=1: each dx,dy adds one rotated dert|pixel to rdert_
    # scan left:
    rx=xcenter-dx; ry=ycenter-dy; rdert=1  # to start while:
    while rdert and rx>=0 and ry>=0 and np.ceil(ry)<yn:
        rdert = form_rdert(rx,ry, dert__t, mask__)
        # terminate the sequence if dert is outside the blob (masked or out of bound)
        if rdert is None: break
        rdert_.insert(0, rdert)
        rx += dx; ry += dy  # next rx, ry
    P.x0 = rx+dx; P.y0 = ry+dy  # revert to leftmost
    # scan right:
    rx=xcenter+dx; ry=ycenter+dy; rdert=1  # to start while:
    while rdert and ry>=0 and np.ceil(rx)<xn and np.ceil(ry)<yn:
        rdert = form_rdert(rx,ry, dert__t, mask__)
        # terminate the sequence if dert is outside the blob (masked or out of bound)
        if rdert is None: break
        rdert_ += [rdert]
        rx += dx; ry += dy  # next rx,ry
    # form rP:
    # initialization:
    rdert = rdert_[0]; _, G, Ga, I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1 = rdert; M=ave_g-G; Ma=ave_ga-Ga; ndert_=[rdert]
    # accumulation:
    for rdert in rdert_[1:]:
        _, g, ga, i, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = rdert
        I+=i; M+=ave_g-g; Ma+=ave_ga-ga; Dy+=dy; Dx+=dx; Sin_da0+=sin_da0; Cos_da0+=cos_da0; Sin_da1+=sin_da1; Cos_da1+=cos_da1
        ndert_ += [rdert]
    # re-form gradients:
    G = np.hypot(Dy,Dx);  Ga = (Cos_da0 + 1) + (Cos_da1 + 1)
    ptuple = Cptuple(I=I, M=M, G=G, Ma=Ma, Ga=Ga, angle=(Dy,Dx), aangle=(Sin_da0, Cos_da0, Sin_da1, Cos_da1))
    # add n,val,L,x,axis?
    # replace P:
    P.ptuple=ptuple; P.dert_=ndert_

def form_rdert(rx,ry, dert__t, mask__):

    # coord, distance of four int-coord derts, overlaid by float-coord rdert in dert__, int for indexing
    # always in dert__ for intermediate float rx,ry:
    x1 = int(np.floor(rx)); dx1 = abs(rx - x1)
    x2 = int(np.ceil(rx));  dx2 = abs(rx - x2)
    y1 = int(np.floor(ry)); dy1 = abs(ry - y1)
    y2 = int(np.ceil(ry));  dy2 = abs(ry - y2)

    try:
        # scale all dert params in proportion to inverted distance from rdert, sum(distances) = 1?
        # approximation, square of rpixel is rotated, won't fully match not-rotated derts
        mask = mask__[y1, x1] * (1 - np.hypot(dx1, dy1)) \
             + mask__[y2, x1] * (1 - np.hypot(dx1, dy2)) \
             + mask__[y1, x2] * (1 - np.hypot(dx2, dy1)) \
             + mask__[y2, x2] * (1 - np.hypot(dx2, dy2))
        mask = round(mask)  # summed mask is fractional, round to 1|0
    except IndexError:
        # out of dert__ => out of blob. Treat as if masked
        mask = 1
    if mask:
        return None
    # if rdert is still inside the blob, return it
    ptuple = []
    for dert__ in dert__t:  # 10 params in dert: i, g, ga, ri, dy, dx, day0, dax0, day1, dax1
        param = dert__[y1, x1] * (1 - np.hypot(dx1, dy1)) \
              + dert__[y2, x1] * (1 - np.hypot(dx1, dy2)) \
              + dert__[y1, x2] * (1 - np.hypot(dx2, dy1)) \
              + dert__[y2, x2] * (1 - np.hypot(dx2, dy2))
        ptuple += [param]
    return ptuple

'''
replace rotate_P_ with directly forming axis-orthogonal Ps:
'''
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


