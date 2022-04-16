'''
Comp_slice is a terminal fork of intra_blob.
-
It traces blob axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat (high internal match) blobs.
(high match or match of angle: M | Ma, roughly corresponds to low gradient: G | Ga)
-
Vectorization is clustering of parameterized Ps + their derivatives (derPs) into PPs: patterns of Ps that describe edge blob.
This process is a reduced-dimensionality (2D->1D) version of cross-comp and clustering cycle, common across this project.
As we add higher dimensions (2D alg, 3D alg), this dimensionality reduction is done in salient high-aspect blobs
(likely edges / contours in 2D or surfaces in 3D) to form more compressed "skeletal" representations of full-D patterns.
'''

from collections import deque
import sys
import numpy as np
from itertools import zip_longest
from copy import deepcopy
from class_cluster import ClusterStructure, NoneType, comp_param, Cdert
from segment_by_direction import segment_by_direction
# import warnings  # to detect overflow issue, in case of infinity loop
# warnings.filterwarnings('error')

ave_inv = 20  # ave inverse m, change to Ave from the root intra_blob?
ave = 5  # ave direct m, change to Ave_min from the root intra_blob?
ave_g = 30  # change to Ave from the root intra_blob?
ave_ga = 0.78  # ga at 22.5 degree
flip_ave = .1
flip_ave_FPP = 0  # flip large FPPs only (change to 0 for debug purpose)
div_ave = 200
ave_rmP = .7  # the rate of mP decay per relative dX (x shift) = 1: initial form of distance
ave_ortho = 20
aveB = 50
# comp_param coefs:
ave_I = ave_inv
ave_M = ave  # replace the rest with coefs:
ave_Ma = 10
ave_G = 10
ave_Ga = 2  # related to dx?
ave_L = 10
ave_dx = 5  # difference between median x coords of consecutive Ps
ave_dangle = 2  # vertical difference between angles
ave_daangle = 2
ave_mP = 10
ave_dP = 10
ave_mPP = 10
ave_dPP = 10
ave_splice = 10

param_names = ["x", "I", "M", "Ma", "L", "angle", "aangle"]  # angle = Dy, Dx; aangle = sin_da0, cos_da0, sin_da1, cos_da1; recompute Gs for comparison?
aves = [ave_dx, ave_I, ave_M, ave_Ma, ave_L, ave_G, ave_Ga, ave_mP, ave_dP]

class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP

    params = list  # 9 compared horizontal params: x, L, I, M, Ma, G, Ga, Ds( Dy, Dx, Sin_da0), Das( Cos_da0, Sin_da1, Cos_da1)
    # I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1 are summed from dert[3:], M, Ma from ave- g, ga
    # G, Ga are recomputed from Ds, Das; M, Ma are not restorable from G, Ga
    x0 = int
    x = float  # median x
    L = int
    sign = NoneType  # g-ave + ave-ga sign
    # all the above are redundant to params
    # rdn = int  # blob-level redundancy, ignore for now
    y = int  # for vertical gap in PP.P__
    # if comp_dx:
    Mdx = int
    Ddx = int
    # composite params:
    dert_ = list  # array of pixel-level derts, redundant to upconnect_, only per blob?
    upconnect_ = list
    downconnect_ = list
    # only in Pd:
    Pm = object  # reference to root P
    dxdert_ = list
    # only in Pm:
    Pd_ = list

class CderP(ClusterStructure):  # tuple of derivatives in P upconnect_ or downconnect_

    dP = int
    mP = int
    params = list  # P derivation layer, n_params = 9 * 2**der_cnt, flat, decoded by mapping each m,d to lower-layer param
    x0 = int  # redundant to params:
    x = float  # median x
    L = int
    sign = NoneType  # g-ave + ave-ga sign
    y = int  # for vertical gaps in PP.P__, replace with derP.P.y?
    P = object  # lower comparand
    _P = object  # higher comparand
    PP = object  # FPP if flip_val, contains this derP
    segment = object
    # higher derivatives
    rdn = int  # mrdn + uprdn, no need for separate mrdn?
    upconnect_ = list  # tuples of higher-row higher-order derivatives per derP
    downconnect_ = list
   # from comp_dx
    fdx = NoneType

class CPP(CP, CderP):  # derP params are inherited from P

    params = list  # derivation layer += derP params, param L is actually Area
    nderP = int  # ly = len(derP__), also x, y?
    sign = bool
    rng = lambda: 1  # rng starts with 1
    rdn = int  # for PP evaluation, recursion count + Rdn / n_derPs
    Rdn = int  # for accumulation only
    upconnect_ = list
    downconnect_ = list
    segments = list  # non-branching stacks of lower-composition Ps, in PP only
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    box = list  # for visualization only, original box before flipping
    mask__ = bool
    P__ = list
    derP__ = list  # replaces dert__
    Plevels = list  # replaces levels
    sublayers = list
    root = object  # segP or higher-order PP

# Functions:

def comp_slice_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    segment_by_direction(blob, verbose=False)  # forms blob.dir_blobs
    for dir_blob in blob.dir_blobs:  # dir_blob should be CBlob

        P__ = slice_blob(dir_blob, verbose=False)  # cluster dir_blob.dert__ into 2D array of blob slices
        # comp_dx_blob(P__), comp_dx?

        derP__ = comp_P_root(P__, rng=1, frng=0)  # scan_P_, comp_P, or comp_layers if called from sub_recursion
        (segm__, segd__) = form_PP_root(derP__, root_rdn=2, fseg=1)  # PP is a stack of (P, derP)s from comp_P, redundant to root blob
        # 2D seg__: cross-sign (same-sign), to be converted to respectively PP_ and PP

        splice_PPs(segm__, frng=1)  # actually splicing segs here
        splice_PPs(segd__, frng=0)
        sub_recursion([], segm__, frng=1)  # rng+ comp_P in PPms, -> param_layer, form sub_PPs
        sub_recursion([], segd__, frng=0)  # der+ comp_P in PPds, -> param_layer, form sub_PPs

        agglo_recursion((segm__, segd__), fseg=1)  # higher-composition comp_seg -> segPs... per seg__[n], forms PPs with PP.levels
        agglo_recursion(dir_blob, fseg=0)  # 2nd call per dir_blob.PP_s formed in 1st call, forms PPP..s and dir_blob.levels

    splice_dir_blob_(blob.dir_blobs)


def slice_blob(blob, verbose=False):  # forms horizontal blob slices: Ps, ~1D Ps, in select smooth edge (high G, low Ga) blobs

    mask__ = blob.mask__  # same as positive sign here
    dert__ = zip(*blob.dert__)  # convert 10-tuple of 2D arrays into 1D array of 10-tuple blob rows
    dert__ = [zip(*dert_) for dert_ in dert__]  # convert 1D array of 10-tuple rows into 2D array of 10-tuples per blob

    height, width = mask__.shape
    if verbose: print("Converting to image...")
    P__ = []  # blob of Ps

    for y, (dert_, mask_) in enumerate(zip(dert__, mask__)):  # unpack lines
        P_ = []  # line of Ps
        _mask = True
        for x, (dert, mask) in enumerate(zip(dert_, mask_)):  # unpack derts: tuples of 10 params
            if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()

            if not mask:  # masks: if 0,_1: P initialization, if 0,_0: P accumulation, if 1,_0: P termination
                if _mask:  # initialize P params with first unmasked dert:
                    Pdert_ = []
                    params = [ave_g-dert[1], ave_ga-dert[2], *dert[3:]]  # m, ma, dert[3:]: i, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1
                else:
                    # dert and _dert are not masked, accumulate P params from dert params:
                    params[1] += ave_g-dert[1]; params[2] += ave_ga-dert[2]  # M, Ma
                    for i, (Param, param) in enumerate(zip(params[2:], dert[3:]), start=2):  # I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1
                        params[i] = Param + param
                    Pdert_.append(dert)
            elif not _mask:
                # _dert is not masked, dert is masked, terminate P:
                L = len(Pdert_)
                P_.append( CP(params= [x-(L-1)/2, L] + list(params), x0=x-(L-1), L=L, y=y, dert_=Pdert_))

            _mask = mask
        if not _mask:  # pack last P:
            L = len(Pdert_)
            P_.append( CP(params = [x-(L-1)/2, L] + list(params), x0=x-(L-1), L=L, y=y, dert_=Pdert_))
        P__ += [P_]

    blob.P__ = P__
    return P__

def comp_P_root(P__, rng, frng):  # vertically compares y-adjacent and x-overlapping Ps: blob slices, forming 2D derP__

    # if der+: P__ is last-call derP__, derP__=[], form new derP__
    # if rng+: P__ is last-call P__, accumulate derP__ with new_derP__
    derP__ = []  # tuples of derivatives from P__, lower derP__ in recursion
    for P_ in P__:
        for P in P_:
            P.upconnect_, P.downconnect_ = [],[]  # reset connects and PP refs in the last layer only
            if isinstance(P, CderP): P.PP = None

    for i, _P_ in enumerate(P__):  # higher compared row
        derP_ = []
        if i+rng < len(P__):  # rng=1 unless rng+ fork
            P_ = P__[i+rng]   # lower compared row
            for P in P_:
                if frng: cP = P.P  # rng+, compared P is lower derivation
                else:    cP = P  # der+, compared P is top derivation
                for _P in _P_:  # upper compared row
                    if frng: _cP = _P.P
                    else:    _cP = _P
                    # test for x overlap between P and _P in 8 directions, all Ps are from +derts, form sub_Pds for comp_dx?
                    if (cP.x0 - 1 < (_cP.x0 + _cP.L) and (cP.x0 + cP.L) + 1 > _cP.x0):

                        if isinstance(cP, CPP) or isinstance(cP, CderP):
                            derP = comp_layer(_cP, cP)  # form vertical derivatives of horizontal P params
                        else:
                            derP = comp_P(_cP, cP)  # form higher vertical derivatives of derP or PP params
                        derP.y=P.y
                        if frng:  # accumulate derP through rng+ recursion:
                            accum_layer(derP.params, P.params)
                        if not P.downconnect_:  # initial row per root PP, then follow upconnect_
                            derP_.append(derP)
                        P.upconnect_.append(derP)  # per P for form_PP
                        _P.downconnect_.append(derP)

                    elif (cP.x0 + cP.L) < _cP.x0:  # no P xn overlap, stop scanning lower P_
                        break
        if derP_:
            derP__ += [derP_]  # rows in blob or PP
        _P_ = P_

    return derP__


def comp_P(_P, P):  # forms vertical derivatives of params per P in _P.upconnect, conditional ders from norm and DIV comp

    # compared P params:
    x, L, M, Ma, I, Dx, Dy, sin_da0, cos_da0, sin_da1, cos_da1 = P.params
    _x, _L, _M, _Ma, _I, _Dx, _Dy, _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _P.params

    dx = _x - x;  mx = ave_dx - abs(dx)  # mean x shift, if dx: rx = dx / ((L+_L)/2)? no overlap, offset = abs(x0 -_x0) + abs(xn -_xn)?
    dI = _I - I;  mI = ave_I - abs(dI)
    dM = _M - M;  mM = min(_M, M)
    dMa = _Ma - Ma;  mMa = min(_Ma, Ma)  # dG, dM are directional, re-direct by dx?
    dL = _L - L * np.hypot(dx, 1); mL = min(_L, L)  # if abs(dx) > ave: adjust L as local long axis, no change in G,M
    # G, Ga:
    G = np.hypot(Dy, Dx); _G = np.hypot(_Dy, _Dx)  # compared as scalars
    dG = _G - G;  mG = min(_G, G)
    Ga = (cos_da0 + 1) + (cos_da1 + 1); _Ga = (_cos_da0 + 1) + (_cos_da1 + 1)  # gradient of angle, +1 for all positives?
    # or Ga = np.hypot( np.arctan2(*Day), np.arctan2(*Dax)?
    dGa = _Ga - Ga;  mGa = min(_Ga, Ga)

    # comp angle:
    _sin = _Dy / (1 if _G==0 else _G); _cos = _Dx / (1 if _G==0 else _G)
    sin  = Dy / (1 if G==0 else G); cos = Dx / (1 if G==0 else G)
    sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
    dangle = np.arctan2(sin_da, cos_da)  # vertical difference between angles
    mangle = ave_dangle - abs(dangle)  # indirect match of angles, not redundant as summed

    # comp angle of angle: forms daa, not gaa?
    sin_dda0 = (cos_da0 * _sin_da0) - (sin_da0 * _cos_da0)
    cos_dda0 = (cos_da0 * _cos_da0) + (sin_da0 * _sin_da0)
    sin_dda1 = (cos_da1 * _sin_da1) - (sin_da1 * _cos_da1)
    cos_dda1 = (cos_da1 * _cos_da1) + (sin_da1 * _sin_da1)

    daangle = (sin_dda0, cos_dda0, sin_dda1, cos_dda1)
    # day = [-sin_dda0 - sin_dda1, cos_dda0 + cos_dda1]
    # dax = [-sin_dda0 + sin_dda1, cos_dda0 + cos_dda1]
    gay = np.arctan2( (-sin_dda0 - sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in y?
    gax = np.arctan2( (-sin_dda0 + sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in x?
    daangle = np.arctan2( gay, gax)  # probably wrong
    maangle = ave_daangle - abs(daangle)  # match between aangles, not redundant as summed

    dP = abs(dx)-ave_dx + abs(dI)-ave_I + abs(G)-ave_G + abs(Ga)-ave_Ga + abs(dM)-ave_M + abs(dMa)-ave_Ma + abs(dL)-ave_L
    # sum to evaluate for der+, abs diffs are distinct from directly defined matches:
    mP = mx + mI + mG + mGa + mM + mMa + mL + mangle + maangle

    params = [dx, mx, dL, mL, dI, mI, dG, mG, dGa, mGa, dM, mM, dMa, mMa, dangle, mangle, daangle, maangle]
    # or summable params only, all Gs are computed at termination?

    x0 = min(_P.x0, P.x0)
    xn = max(_P.x0+_P.L, P.x0+P.L)
    L = xn-x0

    derP = CderP(x0=x0, L=L, y=_P.y, mP=mP, dP=dP, params=params, P=P, _P=_P)
    return derP


def form_PP_root(iderP__, root_rdn, fseg):  # initialize segs per derP in iderP__ or PPs per  in iderP__

    PP_t = []
    for fPd in 0, 1:
        PP_ = []
        derP__ = deepcopy(iderP__)
        for derP_ in derP__:  # scan bottom-up
            for derP in derP_:
                if not isinstance(derP.segment, CPP):  # no derP.segment yet
                    # derP.rdn = (rng+|der+ rdn), ?+ sum([1 for upderP in derP.P.upconnect_ if upderP.dP >= derP.dP]):
                    if fPd:
                        derP.rdn = (derP.mP > derP.dP); sign = derP.dP >= ave_dP * derP.rdn  # PPd / vD sign, distinct from directly defined match:
                    else:
                        derP.rdn = (derP.dP >= derP.mP); sign = derP.mP > ave_mP * derP.rdn
                    if derP._P.upconnect_:
                        form_PP_(derP, PP_, fPd, fseg)  # form PP over P upconnects, if len upconnect_>1: form segments

                    accum_PP(PP, derP, fseg=1)  # accum PP with derP, including rdn, derP.P.downconnect_cnt = 0
    '''
                    if PP_:  # seg_derP_ if fseg, else PP_segments, for batch-mode accum_PP at termination?
                        PP = CPP(sign=sign, x0=derP.x0)
                        terminate_segments(segments, PP)  # terminate segments after checking through upconnects
                        PP_.append(PP)
        for PP in PP_:  # all PPs are terminated
            PP.rdn += root_rdn + PP.Rdn / PP.nderP  # PP rdn is recursion rdn + average (forks + upconnects) rdn
        PP_t.append(PP_)
    '''
    return PP_t  # PPm_, PPd_


def form_PP_(iderP, iPP, PP_, fPd):  # compare lower-layer iderP sign to upconnects sign, form same-contiguous-sign PPs

    # form vertically contiguous segments or patterns of patterns by derP sign, in dir_blob
    # rdn may be sub_PP.rdn, recursion is per sub_PP, rng+|der+ overlap is derP.rdn?

    matching_upconnect_ = []  # always 1 element now
    for derP in iderP._P.upconnect_:  # get lower-der upconnects?
        iderP__ = [pri_derP for segment in segments for derP_ in segment.derP__ for pri_derP in derP_]

        if derP not in iderP__:  # may be added in Pp merging
            segment = isegment
            if fPd:
                derP.rdn = (derP.mP > derP.dP) #+ sum([1 for upderP in derP.P.upconnect_ if upderP.dP >= derP.dP])
                sign = derP.dP >= ave_dP * derP.rdn
            else:
                derP.rdn = (derP.dP >= derP.mP) #+ sum([1 for upderP in derP.P.upconnect_ if upderP.mP > derP.mP])
                sign = derP.mP > ave_mP * derP.rdn
            # same sign, continue in same PP
            if segment.sign == sign:
                # derP is having existing segment, merge it with current PP
                if isinstance(derP.segment, CPP):
                    if segment not in segments: segments.append(derP.segment)  # similar with merge PP
                # derP is not having segment or PP
                else:
                    # more than 1 matching upconnect in segment, terminate current segment and reinit segment
                    if segment.upconnect_:
                        if segment not in segments: segments.append(segment)  # we need this check because there is multiple upconnects, it may terminated in previous upconnect
                        segment = CPP(sign=sign, x0=derP.x0)  # initialize new segment
                        accum_PP(segment, derP, fseg=1)  # accumulate derP into segment
                    # no upconnect in segment, pack derP as upconnect
                    else:
                        segment.upconnect_ += [derP]
                        accum_PP(segment, derP, fseg=1)

                matching_upconnect_ += [derP]  # matching upconnect per PP
                if derP._P.upconnect_:
                    form_PP_(derP, segment, segments, PP_, fPd)  # recursive compare sign of next-layer upconnects

            # different sign and non matching derP, terminate segment, initialize PP and segment
            else:
                # terminate segments into PP
                if segment not in segments: segments += [segment]
                PP = CPP(sign=sign, x0=derP.x0)
                terminate_segments(segments, PP)
                PP_.append(PP)
                segments = []  # reinit segments

                # derP is not having segment, init segment
                if not isinstance(derP.segment, CPP):
                    segment = CPP(sign=sign, x0=derP.x0)
                    accum_derP(segment, derP, fsegment=1)
                if derP._P.upconnect_:
                    upconnect_2_PP_(derP, derP.segment, segments, PP_, fPd)  # recursive compare sign of next-layer upconnects

                # terminate segments after scanning all upconnects
                if segments:
                    PP = CPP(sign=sign, x0=derP.x0)
                    terminate_segments(segments, PP)
                    PP_.append(PP)


    iderP._P.upconnect_ = matching_upconnect_


def merge_PP(_PP, PP, PP_, derP__):  # merge PP into _PP

    for derP_ in PP.derP__:
        for derP in derP_:
            _derP__ = [_pri_derP for _pri_derP_ in _PP.derP__ for _pri_derP in _pri_derP_]  # accum_PP may append new derP
            if derP not in _derP__:
                accum_PP(_PP, derP, 0)  # accumulate params
    for up_PP in PP.upconnect_:
        if up_PP not in _PP.upconnect_:  # PP may have multiple downconnects
            _PP.upconnect_.append(up_PP)

    for derP_ in derP__:  # update derP P,_P
        for derP in derP_:
            if derP.P is PP:  derP.P = _PP
            if derP._P is PP: derP._P = _PP

    for i, down_PP in enumerate(PP.downconnect_):
        if PP in down_PP.upconnect_:
            down_PP.upconnect_[down_PP.upconnect_.index(PP)] = _PP  # update lower PP's upconnect from PP to _PP
            if down_PP not in _PP.downconnect_:
                _PP.downconnect_ += [down_PP]

    for segment in PP.segments:  # add segments from PP
        _PP.segments += [segment]

    if PP in PP_:
        PP_.remove(PP)  # merged PP


def accum_PP(PP, derP, fseg):  # convert to batch mode: accumulate derPs into segment or segs into PP

    if not PP.params: PP.params = derP.params.copy()
    else:             accum_layer(PP.params, derP.params)
    PP.x0 = min(PP.x0, derP.x0)
    PP.nderP += 1
    PP.mP += derP.mP
    PP.dP += derP.dP
    PP.Rdn += derP.rdn
    PP.y = max(derP.y, PP.y)  # or pass local y arg instead of derP.y?

    if not PP.derP__:
        PP.derP__.append([derP])
        PP.P__.append([derP.P])
    else:
        current_ys = [derP_[0].P.y for derP_ in PP.derP__]  # list of current-layer derP rows
        if derP.P.y in current_ys:
            PP.derP__[current_ys.index(derP.P.y)].append(derP)  # append derP row
            PP.P__[current_ys.index(derP.P.y)].append(derP.P)  # append P row
        elif derP.P.y > current_ys[-1]:  # derP.y > largest y in ys
            PP.derP__.append([derP]); PP.P__.append([derP.P])
        elif derP.P.y < current_ys[0]:  # derP.y < smallest y in ys
            PP.derP__.insert(0, [derP]); PP.P__.insert(0, [derP.P])
        elif derP.P.y > current_ys[0] and derP.P.y < current_ys[-1] :  # derP.y in between largest and smallest value
            PP.derP__.insert(derP.P.y-current_ys[0], [derP])
            PP.P__.insert(derP.P.y-current_ys[0], [derP.P])

    PP.L = len(PP.derP__)  # PP.L is Ly

    if fsegment: derP.segment = PP
    else: derP.PP = PP

def accum_layer(top_layer, der_layer):

    for i, (_param, param) in enumerate(zip(top_layer, der_layer)):
        if isinstance(_param, tuple):
            if len(_param) == 2:  # (sin_da, cos_da)
                _sin_da, _cos_da = _param
                sin_da, cos_da = param
                sum_sin_da = (cos_da * _sin_da) + (sin_da * _cos_da)  # sin(α + β) = sin α cos β + cos α sin β
                sum_cos_da = (cos_da * _cos_da) - (sin_da * _sin_da)  # cos(α + β) = cos α cos β - sin α sin β
                top_layer[i] = (sum_sin_da, sum_cos_da)
            else:  # (sin_da0, cos_da0, sin_da1, cos_da1)
                _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _param
                sin_da0, cos_da0, sin_da1, cos_da1 = param
                sum_sin_da0 = (cos_da0 * _sin_da0) + (sin_da0 * _cos_da0)  # sin(α + β) = sin α cos β + cos α sin β
                sum_cos_da0 = (cos_da0 * _cos_da0) - (sin_da0 * _sin_da0)  # cos(α + β) = cos α cos β - sin α sin β
                sum_sin_da1 = (cos_da1 * _sin_da1) + (sin_da1 * _cos_da1)
                sum_cos_da1 = (cos_da1 * _cos_da1) - (sin_da1 * _sin_da1)
                top_layer[i] = (sum_sin_da0, sum_cos_da0, sum_sin_da1, sum_cos_da1)
        else:  # scalar
            top_layer[i] += param


def sub_recursion(root_sublayers, PP_, frng):  # compares param_layers of derPs in generic PP, form or accum top derivatives

    comb_sublayers = []
    for PP in PP_:  # PP is generic higher-composition pattern, P is generic lower-composition pattern
                    # both P and PP may be recursively formed higher-derivation derP and derPP, etc.

        if frng: PP_V = PP.mP - ave_mPP * PP.rdn; rng = PP.rng+1; min_L = rng * 2  # V: value of sub_recursion per PP
        else:    PP_V = PP.dP - ave_dPP * PP.rdn; rng = PP.rng; min_L = 3  # need 3 Ps to compute layer2, etc.
        if PP_V > 0 and PP.nderP > min_L:

            PP.rdn += 1  # rdn to prior derivation layers
            PP.rng = rng
            sub_derP__ = comp_P_root(PP.derP__, rng, frng)  # scan_P_, comp_P layer0;  splice PPs across dir_blobs?
            sub_PPm_, sub_PPd_ = form_PP_(sub_derP__, PP.rdn)  # each PP is a stack of (P, derP)s from comp_P

            PP.sublayers = [(sub_PPm_, sub_PPd_)]
            if sub_PPm_:   # or rng+n to reduce clustering costs?
                sub_recursion(PP.sublayers, sub_PPm_, frng=1)  # rng+ comp_P in PPms, form param_layer, sub_PPs
            if sub_PPd_:
                sub_recursion(PP.sublayers, sub_PPd_, frng=0)  # der+ comp_P in PPds, form param_layer, sub_PPs

            if PP.sublayers:  # pack added sublayers:
                new_comb_sublayers = []
                for (comb_sub_PPm_, comb_sub_PPd_), (sub_PPm_, sub_PPd_) in zip_longest(comb_sublayers, PP.sublayers, fillvalue=([], [])):
                    comb_sub_PPm_ += sub_PPm_
                    comb_sub_PPd_ += sub_PPd_
                    new_comb_sublayers.append((comb_sub_PPm_, comb_sub_PPd_))  # add sublayer
                comb_sublayers = new_comb_sublayers

    if comb_sublayers: root_sublayers += comb_sublayers


def agglo_recursion(blob):  # compositional recursion per blob.Plevel. P, PP, PPP are relative terms, each may be of any composition order

    PP_t = blob.levels[-1]  # input-level composition Ps, initially PPs
    PPP_t = []  # next-level composition Ps, initially PPPs  # for fiPd, PP_ in enumerate(PP_t): fiPd = fiPd % 2  # dir_blob.M += PP.M += derP.m

    n_extended = 0
    for i, PP_ in enumerate(PP_t):   # fiPd = fiPd % 2
        fiPd = i % 2
        if fiPd: ave_PP = ave_dPP
        else:    ave_PP = ave_mPP

        M = ave-abs(blob.G)
        if M > ave_PP * blob.rdn and len(PP_)>1:  # >=2 comparands
            n_extended += 1

            derPP_ = comp_aggloP_root(PP_, rng=1)  # PP is generic for lower-level composition
            PPPm_, PPPd_ = form_PP_(derPP_, root_rdn=2)  # PPP is generic next-level composition

            splice_PPs(PPPm_, frng=1)
            splice_PPs(PPPd_, frng=0)
            PPP_t += [PPPm_, PPPd_]  # flat version

            if PPPm_: sub_recursion([], PPPm_, frng=1)  # rng+
            if PPPd_: sub_recursion([], PPPd_, frng=0)  # der+
        else:
            PPP_t += [[], []]  # replace with neg PPPs?

    blob.levels.append(PPP_t)  # levels of dir_blob are Plevels

    if n_extended/len(PP_t) > 0.5:  # mean ratio of extended PPs
        agglo_recursion(blob)


def comp_aggloP_root(PP_, rng):

    for PP in PP_: PP.downconnect_ = []  # new downconnect will be recomputed for derPP
    derPP__ = []

    for PP in PP_:
        for i, _PP in enumerate(PP.upconnect_):
            if isinstance(_PP, CPP):  # _PP could be replaced by derPP

                derPP = comp_layer(_PP, PP)  # cross-sign if PPd?
                PP.upconnect_[i] = derPP  # replace PP with derPP
                _PP.downconnect_ += [derPP]

                if not derPP__: derPP__.append([derPP])
                else:
                    # pack derPP in row at derPP.y:
                    current_ys = [derP_[0].P.y for derP_ in derPP__]  # list of current-layer derP rows
                    if derPP.P.y in current_ys:
                        derPP__[current_ys.index(derPP.P.y)].append(derPP)  # append derPP row
                    elif derPP.P.y > current_ys[-1]:  # derPP.y > largest y in ys
                        derPP__.append([derPP])
                    elif derPP.P.y < current_ys[0]:  # derPP.y < smallest y in ys
                        derPP__.insert(0, [derPP])
                    elif derPP.P.y > current_ys[0] and derPP.P.y < current_ys[-1] :  # derPP.y in between largest and smallest value
                        derPP__.insert(derPP.P.y-current_ys[0], [derPP])

    return derPP__

# draft
def splice_PPs(PPP_, frng):  # splice select PP pairs if der+ or triplets if rng+

    for PPP in PPP_:
        if frng:  # rng fork
            for __PP_ in PPP.P__:  # top-down
                for i, __PP in enumerate(__PP_):
                    spliced_downconnect_ = []
                    for j, _downconnect in enumerate(__PP.downconnect_):
                        if isinstance(_downconnect, CderP): _PP = _downconnect.PP  # get PP reference if downconnect is derP
                        else:                               _PP = _downconnect
                        fbreak = 0
                        for k, downconnect in enumerate(_PP.downconnect_):
                            if isinstance(downconnect, CderP): PP = downconnect.PP
                            else:                              PP = downconnect
                            if __PP is not _PP and __PP is not PP and _PP is not PP:
                                max_rng = max(__PP.rng, PP.rng)
                                if max_rng > _PP.L:
                                    spliced_downconnect_ += [_downconnect]
                                    # higher row of Ps that form add_derPs:
                                    _P_ = [P for P_ in (__PP.P__[-max_rng:] + PP.P__[:max_rng]) for P in P_]
                                    # multiple lower-row Ps, [:] to not reference PP.P__, to be packed in accum_PP:
                                    P__ = [__P_[:] for __P_ in __PP.P__[-max_rng-1:]] + [P_[:] for P_ in PP.P__[:max_rng]]
                                    # comp Ps x gap _PP:
                                    for P_ in P__:  # lower row
                                        for _P in _P_[:]:  # higher row
                                            for P in P_[:]:  # lower Ps per _P
                                                if isinstance(P, CPP): add_derP = comp_layer(_P, P)
                                                else:                  add_derP = comp_P(_P, P)
                                                accum_PP(__PP, add_derP)
                                                # we may be adding multiple PPs here, one per x-overlap?
                                        _P_ = P_

                                    # merge PP into __PP
                                    for derP_ in PP.derP__:
                                        for derP in derP_:
                                            _derP__ = [_pri_derP for _pri_derP_ in __PP.derP__ for _pri_derP in _pri_derP_]  # accum_PP may append new derP
                                            if derP not in _derP__:
                                                accum_PP(__PP, derP)  # accumulate params

                                    # update upconnect and downconnect reference for _PP, pack into function?
                                    for up_PP in _PP.upconnect_:
                                        if up_PP not in __PP.upconnect_:  # PP may have multiple downconnects
                                            __PP.upconnect_.append(up_PP)
                                            Rdn = 0
                                            n = 0
                                            for up_PP in _PP.upconnect_:
                                                if up_PP not in __PP.upconnect_:  # PP may have multiple downconnects
                                                    __PP.upconnect_.append(up_PP)
                                                    Rdn += up_PP.dP > __PP.mP
                                                    n += 1
                                            __PP.rdn += Rdn / n
                                            __PP.rdn += up_PP.dP > __PP.mP  # ?

                                    for i, down_PP in enumerate(_PP.downconnect_):
                                        if _PP in down_PP.upconnect_:
                                            down_PP.upconnect_[down_PP.upconnect_.index(_PP)] = __PP  # update lower PP's upconnect from PP to _PP
                                            if down_PP not in _PP.downconnect_:
                                                __PP.downconnect_ += [down_PP]
                                                __PP.rdn += down_PP.dP > __PP.mP

                                    # update upconnect and downconnect reference for PP
                                    for up_PP in PP.upconnect_:
                                        if up_PP not in __PP.upconnect_:  # PP may have multiple downconnects
                                            __PP.upconnect_.append(up_PP)
                                            __PP.rdn += up_PP.dP > __PP.mP
                                    for i, down_PP in enumerate(PP.downconnect_):
                                        if PP in down_PP.upconnect_:
                                            down_PP.upconnect_[down_PP.upconnect_.index(PP)] = __PP  # update lower PP's upconnect from PP to _PP
                                            if down_PP not in _PP.downconnect_:
                                                __PP.downconnect_ += [down_PP]
                                                __PP.rdn += down_PP.dP > __PP.mP

                                    fbreak = 1
                                    break  # break PP_ loop
                        if fbreak: break   # break  _PP loop

                    for downconnect in spliced_downconnect_:  # PP__.downconnect could be PP or derP
                        __PP.downconnect_.remove(downconnect)

        else:  # der+ fork
            pass


def comp_dx(P):  # cross-comp of dx s in P.dert_

    Ddx = 0
    Mdx = 0
    dxdert_ = []
    _dx = P.dert_[0][2]  # first dx
    for dert in P.dert_[1:]:
        dx = dert[2]
        ddx = dx - _dx
        if dx > 0 == _dx > 0: mdx = min(dx, _dx)
        else: mdx = -min(abs(dx), abs(_dx))
        dxdert_.append((ddx, mdx))  # no dx: already in dert_
        Ddx += ddx  # P-wide cross-sign, P.L is too short to form sub_Ps
        Mdx += mdx
        _dx = dx
    P.dxdert_ = dxdert_
    P.Ddx = Ddx
    P.Mdx = Mdx


def comp_layer(_derP, derP):

    nparams = len(_derP.params)
    derivatives = []
    hyps = []
    mP = 0  # for rng+ eval
    dP = 0  # for der+ eval

    for i, (_param, param) in enumerate(zip(_derP.params, derP.params)):
        # get param type:
        param_type = int(i/ (2 ** (nparams-1)))  # for 9 compared params, but there are more in higher layers?

        if param_type == 0:  # x
            _x = param; x = param
            dx = _x - x; mx = ave_dx - abs(dx)
            derivatives.append(dx); derivatives.append(mx)
            hyps.append(np.hypot(dx, 1))
            dP += dx; mP += mx

        elif param_type == 1:  # I
            _I = _param; I = param
            dI = _I - I; mI = ave_I - abs(dI)
            derivatives.append(dI); derivatives.append(mI)
            dP += dI; mP += mI

        elif param_type == 2:  # G
            hyp = hyps[i%param_type]
            _G = _param; G = param
            dG = _G - G/hyp;  mG = min(_G, G)  # if comp_norm: reduce by hypot
            derivatives.append(dG); derivatives.append(mG)
            dP += dG; mP += mG

        elif param_type == 3:  # Ga
            _Ga = _param; Ga = param
            dGa = _Ga - Ga;  mGa = min(_Ga, Ga)
            derivatives.append(dGa); derivatives.append(mGa)
            dP += dGa; mP += mGa

        elif param_type == 4:  # M
            hyp = hyps[i%param_type]
            _M = _param; M = param
            dM = _M - M/hyp;  mM = min(_M, M)
            derivatives.append(dM); derivatives.append(mM)
            dP += dM; mP += mM

        elif param_type == 5:  # Ma
            _Ma = _param; Ma = param
            dMa = _Ma - Ma;  mMa = min(_Ma, Ma)
            derivatives.append(dMa); derivatives.append(mMa)
            dP += dMa; mP += mMa

        elif param_type == 6:  # L
            hyp = hyps[i%param_type]
            _L = _param; L = param
            dL = _L - L/hyp;  mL = min(_L, L)
            derivatives.append(dL); derivatives.append(mL)
            dP += dL; mP += mL

        elif param_type == 7:  # angle, (sin_da, cos_da)
            if isinstance(_param, tuple):  # (sin_da, cos_da)
                 _sin_da, _cos_da = _param; sin_da, cos_da = param
                 sin_dda = (cos_da * _sin_da) - (sin_da * _cos_da)  # sin(α - β) = sin α cos β - cos α sin β
                 cos_dda = (cos_da * _cos_da) + (sin_da * _sin_da)  # cos(α - β) = cos α cos β + sin α sin β
                 dangle = (sin_dda, cos_dda)  # da
                 mangle = ave_dangle - abs(np.arctan2(sin_dda, cos_dda))  # ma is indirect match
                 derivatives.append(dangle); derivatives.append(mangle)
                 dP += np.arctan2(sin_dda, cos_dda); mP += mangle
            else: # m or scalar
                _mangle = _param; mangle = param
                dmangle = _mangle - mangle;  mmangle = min(_mangle, mangle)
                derivatives.append(dmangle); derivatives.append(mmangle)
                dP += dmangle; mP += mmangle

        elif param_type == 8:  # dangle   (sin_da0, cos_da0, sin_da1, cos_da1)
            if isinstance(_param, tuple):  # (sin_da, cos_da)
                _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _param
                sin_da0, cos_da0, sin_da1, cos_da1 = param

                sin_dda0 = (cos_da0 * _sin_da0) - (sin_da0 * _cos_da0)
                cos_dda0 = (cos_da0 * _cos_da0) + (sin_da0 * _sin_da0)
                sin_dda1 = (cos_da1 * _sin_da1) - (sin_da1 * _cos_da1)
                cos_dda1 = (cos_da1 * _cos_da1) + (sin_da1 * _sin_da1)
                daangle = (sin_dda0, cos_dda0, sin_dda1, cos_dda1)
                # day = [-sin_dda0 - sin_dda1, cos_dda0 + cos_dda1]
                # dax = [-sin_dda0 + sin_dda1, cos_dda0 + cos_dda1]
                gay = np.arctan2( (-sin_dda0 - sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in y?
                gax = np.arctan2( (-sin_dda0 + sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in x?
                maangle = ave_dangle - abs(np.arctan2(gay, gax))  # match between aangles, probably wrong
                derivatives.append(daangle); derivatives.append(maangle)
                dP += daangle; mP += maangle

            else:  # m or scalar
                _maangle = _param; maangle = param
                dmaangle = _maangle - maangle;  mmaangle = min(_maangle, maangle)
                derivatives.append(dmaangle); derivatives.append(mmaangle)
                dP += dmaangle; mP += mmaangle

    x0 = min(_derP.x0, derP.x0)
    xn = max(_derP.x0+_derP.L, derP.x0+derP.L)
    L = xn-x0

    return CderP(x0=x0, L=L, y=_derP.y, mP=mP, dP=dP, params=derivatives, P=derP, _P=_derP)

# old draft
def splice_dir_blob_(dir_blobs):

    for i, _dir_blob in enumerate(dir_blobs):
        for fPd in 0, 1:
            PP_ = _dir_blob.levels[0][fPd]

            if fPd: PP_val = sum([PP.mP for PP in PP_])
            else:   PP_val = sum([PP.dP for PP in PP_])

            if PP_val - ave_splice > 0:  # high mPP pr dPP

                _top_P_ = _dir_blob.P__[0]
                _bottom_P_ = _dir_blob.P__[-1]

                for j, dir_blob in enumerate(dir_blobs):
                    if _dir_blob is not dir_blob:

                        top_P_ = dir_blob.P__[0]
                        bottom_P_ = dir_blob.P__[-1]
                        # test y adjacency
                        if (_top_P_[0].y-1 == bottom_P_[0].y) or (top_P_[0].y-1 == _bottom_P_[0].y):
                            # tet x overlap
                             _x0 = min([_P.x0 for _P_ in _dir_blob.P__ for _P in _P_])
                             _xn = min([_P.x0+_P.L for _P_ in _dir_blob.P__ for _P in _P_])
                             x0 = min([P.x0 for P_ in dir_blob.P__ for P in P_])
                             xn = min([P.x0+_P.L for P_ in dir_blob.P__ for P in P_])
                             if (x0 - 1 < _xn and xn + 1 > _x0) or  (_x0 - 1 < xn and _xn + 1 > x0) :
                                 splice_dir_blobs(_dir_blob, dir_blob)  # splice dir_blob into _dir_blob
                                 dir_blobs[j] = _dir_blob

def splice_dir_blobs(_blob, blob):
    # merge blob into _blob here
    pass