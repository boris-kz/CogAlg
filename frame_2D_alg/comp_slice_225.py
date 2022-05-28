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
    downconnect_cnt = int
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
    # higher derivatives
    rdn = int  # mrdn + uprdn, no need for separate mrdn?
    upconnect_ = list  # tuples of higher-row higher-order derivatives per derP
    downconnect_cnt = int
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
    downconnect_cnt = int
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    box = list  # for visualization only, original box before flipping
    mask__ = bool
    derP__ = list  # replaces dert__
    Plevels = list  # replaces levels
    sublayers = list

# Functions:

def comp_slice_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    segment_by_direction(blob, verbose=False)  # need to revise, it should form blob.dir_blobs, not FPPs
    for dir_blob in blob.dir_blobs:  # dir_blob should be CBlob, splice PPs across dir_blobs?

        P__ = slice_blob(dir_blob, verbose=False)  # cluster dir_blob.dert__ into 2D array of blob slices
        # comp_dx_blob(P__), comp_dx?

        derP__ = comp_P_root(P__, rng=1)  # scan_P_, comp_P, or comp_layers if called from sub_recursion
        (PPm_, PPd_) = form_PP_(derP__, root_rdn=2)  # each PP is a stack of (P, derP)s from comp_P, redundant to root blob

        sub_recursion([], PPm_, rng=2)  # rng+ comp_P in PPms, -> param_layer, form sub_PPs
        sub_recursion([], PPd_, rng=1)  # der+ comp_P in PPds, -> param_layer, form sub_PPs

        dir_blob.levels = [[PPm_, PPd_]]  # 1st composition level, each PP_ may be multi-layer from sub_recursion
        agglo_recursion(dir_blob)  # higher-composition comp_PP in blob -> derPPs, form PPP., appends dir_blob.levels


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

def comp_P_root(P__, rng):  # vertically compares y-adjacent and x-overlapping Ps: blob slices, forming 2D derP__

    # if der+: P__ is last-call derP__, derP__=[], form new derP__
    # if rng+: P__ is last-call P__, accumulate derP__ with new_derP__
    derP__ = []  # tuples of derivatives from P__, lower derP__ in recursion
    for P_ in P__:
        for P in P_:
            P.upconnect_ = []; P.downconnect_cnt = 0  # reset connects and PP refs in the last layer only
            if isinstance(P, CderP):
                P.PP = None
    ''' if rng += n:
    for P_ in P__[i+1:]:  # vertical gap <= rng, for rng+
        if (rng>1 and P_[0].P.y - _P_[0].P.y <= rng) or (rng == 1 and P_[0].y - _P_[0].y <= rng): 
    '''
    for i, _P_ in enumerate(P__):  # upper row, top-down
        derP_ = []
        for P_ in P__[i + rng:]:  # lower row,  # if multiple rng increment:

            if rng>1: cP = P.P  # rng+, compared P is lower derivation
            else:     cP = P    # der+, compared P is top derivation
            for _P in _P_:  # upper row
                if rng>1: _cP = _P.P
                else:     _cP = _P
                # test for x overlap between P and _P in 8 directions, all Ps are from +derts, form sub_Pds for comp_dx?
                if (cP.x0 - 1 < (_cP.x0 + _cP.L) and (cP.x0 + cP.L) + 1 > _cP.x0) \
                    and cP.y - _cP.y < rng:  # vertical gap <= rng, for rng+

                    if isinstance(cP, CderP): derP = comp_layer(_cP, cP)  # form higher derivatives of vertical derivatives
                    else:                     derP = comp_P(_cP, cP)  # form vertical derivatives of horizontal P params
                    # if multiple rng increment:
                    # derP.y = P.y
                    if rng>1:  # accumulate derP through rng+ recursion:
                        accum_layer(derP.params, P.params)
                    if not P.downconnect_cnt:  # initial row per root PP, then follow upconnect_
                        derP_.append(derP)
                    P.upconnect_.append(derP)  # per P for form_PP
                    _P.downconnect_cnt += 1

                elif (cP.x0 + cP.L) < _cP.x0:  # no P xn overlap, stop scanning lower P_
                    break
        if derP_: derP__ += [derP_]  # rows in blob or PP
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
    mP = mx + mI + mG + mGa + mM + mMa + mL + mangle + maangle

    params = [dx, mx, dL, mL, dI, mI, dG, mG, dGa, mGa, dM, mM, dMa, mMa, dangle, mangle, daangle, maangle]
    # or summable params only, all Gs are computed at termination?

    x0 = min(_P.x0, P.x0)
    xn = max(_P.x0+_P.L, P.x0+P.L)
    L = xn-x0

    derP = CderP(x0=x0, L=L, y=_P.y, mP=mP, dP=dP, params=params, P=P, _P=_P)

    return derP


def form_PP_(derP__, root_rdn):  # form vertically contiguous patterns of patterns by derP sign, in dir_blob

    # rdn may be sub_PP.rdn, recursion is per sub_PP, rng+|der+ overlap is derP.rdn?
    PP_t = []
    for fPd in 0, 1:
        PP_ = []
        for derP_ in deepcopy(derP__):  # scan bottom-up
            for derP in derP_:
                if not derP.P.downconnect_cnt and not isinstance(derP.PP, CPP):  # no derP.PP yet
                    # derP.rdn = fork rdn + rdn to stronger upconnects, forming overlapping PPs:
                    if fPd:
                        derP.rdn = (derP.mP > derP.dP) + sum([1 for upderP in derP.P.upconnect_ if upderP.dP >= derP.dP])
                        sign = derP.dP >= ave_dP * derP.rdn
                    else:
                        derP.rdn = (derP.dP >= derP.mP) + sum([1 for upderP in derP.P.upconnect_ if upderP.mP > derP.mP])
                        sign = derP.mP > ave_mP * derP.rdn

                    PP = CPP(sign=sign)
                    accum_PP(PP, derP)  # accum PP with derP, including rdn, derP.P.downconnect_cnt = 0
                    PP_.append(PP)
                    if derP.P.upconnect_:
                        upconnect_2_PP_(derP, PP_, fPd)  # form PPs over P upconnects

        for PP in PP_:  # all PPs are terminated
            PP.rdn += root_rdn + PP.Rdn / PP.nderP  # PP rdn is recursion rdn + average fork rdn + upconnects rdn

        PP_t.append(PP_)

    return PP_t  # PPm_, PPd_


def upconnect_2_PP_(iderP, PP_, fPd):  # compare lower-layer iderP sign to upconnects sign, form same-contiguous-sign PPs

    matching_upconnect_ = []
    for derP in iderP._P.upconnect_:  # get lower-der upconnects?
        derP__ = [pri_derP for derP_ in iderP.PP.derP__ for pri_derP in derP_]

        if derP not in derP__:  # may be added in Pp merging
            if fPd:
                derP.rdn = (derP.mP > derP.dP) + sum([1 for upderP in derP.P.upconnect_ if upderP.dP >= derP.dP])
                sign = derP.dP >= ave_dP * derP.rdn
            else:
                derP.rdn = (derP.dP >= derP.mP) + sum([1 for upderP in derP.P.upconnect_ if upderP.mP > derP.mP])
                sign = derP.mP > ave_mP * derP.rdn

            if iderP.PP.sign == sign:  # upconnect is same-sign, or if match only, no neg PPs?
                if isinstance(derP.PP, CPP):
                    if (derP.PP is not iderP.PP):  # upconnect has PP, merge it
                        merge_PP(iderP.PP, derP.PP, PP_)
                else:
                    accum_PP(iderP.PP, derP)  # accumulate derP in current PP
                matching_upconnect_.append(derP)
            else:
                # sign changed
                if not isinstance(derP.PP, CPP):
                    PP = CPP(sign=sign)
                    PP_.append(PP)
                    accum_PP(PP, derP)
                    derP.P.downconnect_cnt = 0

                iderP.PP.upconnect_ += [derP.PP]  # for comp_PP_root, or comp_Pn_root in agglo_recursion
                derP.PP.downconnect_cnt += 1

            if derP._P.upconnect_:
                upconnect_2_PP_(derP, PP_, fPd)  # recursive compare sign of next-layer upconnects

    iderP._P.upconnect_ = matching_upconnect_


def merge_PP(_PP, PP, PP_):  # merge PP into _PP

    for derP_ in PP.derP__:
        for derP in derP_:
            _derP__ = [_pri_derP for _pri_derP_ in _PP.derP__ for _pri_derP in _pri_derP_]  # accum_PP may append new derP
            if derP not in _derP__:
                accum_PP(_PP, derP)  # accumulate params
    for up_PP in PP.upconnect_:
        if up_PP not in _PP.upconnect_:  # single PP may have multiple downconnects
            _PP.upconnect_.append(up_PP)

    if PP in PP_:
        PP_.remove(PP)  # merged PP


def accum_PP(PP, derP):  # accumulate params in PP

    if not PP.params: PP.params = derP.params.copy()
    else:             accum_layer(PP.params, derP.params)
    PP.nderP += 1
    PP.mP += derP.mP
    PP.dP += derP.dP
    PP.Rdn += derP.rdn

    if not PP.derP__: PP.derP__.append([derP])
    else:
        current_ys = [derP_[0].P.y for derP_ in PP.derP__]  # list of current-layer derP rows
        if derP.P.y in current_ys:
            PP.derP__[current_ys.index(derP.P.y)].append(derP)  # append derP row
        elif derP.P.y > current_ys[-1]:  # derP.y > largest y in ys
            PP.derP__.append([derP])
        elif derP.P.y < current_ys[0]:  # derP.y < smallest y in ys
            PP.derP__.insert(0, [derP])
        elif derP.P.y > current_ys[0] and derP.P.y < current_ys[-1] :  # derP.y in between largest and smallest value
            PP.derP__.insert(derP.P.y-current_ys[0], [derP])

    derP.PP = PP


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


def sub_recursion(root_sublayers, PP_, rng):  # compares param_layers of derPs in generic PP, form or accum top derivatives

    comb_sublayers = []
    for PP in PP_:  # PP is generic higher-composition pattern, P is generic lower-composition pattern
                    # both P and PP may be recursively formed higher-derivation derP and derPP, etc.

        if rng > 1: PP_V = PP.mP - ave_mPP * PP.rdn; min_L = rng * 2  # V: value of sub_recursion per PP
        else:       PP_V = PP.dP - ave_dPP * PP.rdn; min_L = 3  # need 3 Ps to compute layer2, etc.
        if PP_V > 0 and PP.nderP > min_L:

            PP.rdn += 1  # rdn to prior derivation layers
            sub_derP__ = comp_P_root(PP.derP__, rng)  # scan_P_, comp_P layer0;  splice PPs across dir_blobs?
            sub_PPm_, sub_PPd_ = form_PP_(sub_derP__, PP.rdn)  # each PP is a stack of (P, derP)s from comp_P

            PP.sublayers = [(sub_PPm_, sub_PPd_)]
            if sub_PPm_:
                sub_recursion(PP.sublayers, sub_PPm_, rng+1)  # rng+ comp_P in PPms, form param_layer, sub_PPs
            if sub_PPd_:
                sub_recursion(PP.sublayers, sub_PPd_, rng=1)  # der+ comp_P in PPds, form param_layer, sub_PPs

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

    nextended = 0
    for i, PP_ in enumerate(PP_t):   # fiPd = fiPd % 2
        fiPd = i % 2
        if fiPd: ave_PP = ave_dPP
        else:    ave_PP = ave_mPP

        M = ave-abs(blob.G)
        if M > ave_PP * blob.rdn and len(PP_)>1:  # >=2 comparands
            nextended += 1
            derPP_ = comp_aggloP_root(PP_)  # PP is generic for lower-level composition
            PPPm_, PPPd_ = form_aggloP_(derPP_, root_rdn=2)  # PPP is generic next-level composition
            PPP_t += [PPPm_, PPPd_]  # flat version
        else:
            PPP_t += [[], []]

    blob.levels.append(PPP_t)  # levels of dir_blob are Plevels
    if nextended/len(PP_t)>0.5:  # temporary
        agglo_recursion(blob)

def comp_aggloP_root(PP_):

    for PP in PP_:
        PP.downconnect_cnt = 0

    derPP_ = []
    for PP in PP_:
        for _PP in PP.upconnect_:
            if isinstance(_PP, CPP):  # _PP could be the added derPP
                derPP = comp_layer(_PP, PP)
                derPP_.append(derPP)
                PP.upconnect_.append(derPP)
                _PP.downconnect_cnt += 1
    return derPP_


def comp_PP(_PP, PP):  # comp_aggloP
    # loop each param to compare: the number of params is not known?
    derPP = CderP(_P=_PP, P=PP)
    return derPP

# same as form_PP_?
def form_aggloP_(derPP_, root_rdn):  # initially forms PPPs

    PPP_t = []
    for fPpd in 0, 1:
        PPP_ = []
        for derPP in deepcopy(derPP_):
            if not derPP.P.downconnect_cnt and not isinstance(derPP.PP, CPP):
                if fPpd:
                    derPP.rdn = (derPP.mP > derPP.dP) + sum([1 for upderPP in derPP.P.upconnect_ if upderPP.dP >= derPP.dP])
                    sign = derPP.dP >= ave_dP * derPP.rdn
                else:
                    derPP.rdn = (derPP.dP >= derPP.mP) + sum([1 for upderPP in derPP.P.upconnect_ if upderPP.mP > derPP.mP])
                    sign = derPP.mP > ave_mP * derPP.rdn

                PPP = CPP(sign=sign)
                accum_PPP(PPP, derPP)
                PPP_.append(PPP)
                if derPP._P.upconnect_:
                    upconnect_2_PPP_(derPP, PPP_, fPpd)  # form PPPs across _P upconnects

        for PPP in PPP_:  # all PPPs are terminated
            PPP.rdn += root_rdn + PPP.Rdn / PPP.nderP  # PP rdn is recursion rdn + average fork rdn + upconnects rdn
        PPP_t.append(PPP_)

    return PPP_t  # PPPm_, PPPd_


def upconnect_2_PPP_(iderPP, PPP_, fPd):  # compare lower-layer derP sign to upconnects sign, form same-contiguous-sign PPs

    matching_upconnect_ = []

    for derPP in iderPP._P.upconnect_:  # lower-der upconnects
        derPP_ = [pri_derP for pri_derP in iderPP.PP.derP__]

        if derPP not in derPP_:  # may be added in Pp merging
            if fPd:
                derPP.rdn = (derPP.mP > derPP.dP) + sum([1 for upderPP in derPP.P.upconnect_ if isinstance(upderPP, CderP) and upderPP.dP >= derPP.dP])
                sign = derPP.dP >= ave_dP * derPP.rdn
            else:
                derPP.rdn = (derPP.dP >= derPP.mP) + sum([1 for upderPP in derPP.P.upconnect_ if isinstance(upderPP, CderP) and upderPP.mP > derPP.mP])
                sign = derPP.mP > ave_mP * derPP.rdn

            if iderPP.PP.sign == sign:  # upconnect is same-sign, or if match only, no neg PPs?
                # This test won't work, they are all CPPs now:
                if isinstance(derPP.PP, CPP):
                    if (derPP.PP is not iderPP.PP):  # upconnect has PP, merge it
                        merge_PPP(iderPP.PP, derPP.PP, PPP_)
                else:  # accumulate derP in current PP
                    accum_PPP(iderPP.PP, derPP)
                matching_upconnect_.append(derPP)
            else:  # sign changed
                if not isinstance(derPP.PP, CPP):
                    PPP = CPP(sign=sign)
                    PPP_.append(PPP)
                    accum_PPP(PPP, derPP)
                    derPP.P.downconnect_cnt = 0

                iderPP.PP.upconnect_ += [derPP.PP]  # for comp_PP_root, or comp_Pn_root in agglo_recursion
                derPP.PP.downconnect_cnt += 1

            if derPP._P.upconnect_:
                upconnect_2_PPP_(derPP, PPP_, fPd)  # recursive compare sign of next-layer upconnects

    derPP._P.upconnect_ = matching_upconnect_


def merge_PPP(_PPP, PPP, PPP_):  # merge PPP into _PPP

    for derPP in PPP.derP__:  # PP is 1 dimensional now, there's no y?
        _derPP__ = [_pri_derPP for _pri_derPP_ in _PPP.derP__ for _pri_derPP in _pri_derPP_]  # accum_PP may append new derP
        if derPP not in _derPP__:
            accum_PPP(_PPP, derPP)

    for up_PPP in PPP.upconnect_:
        if up_PPP not in _PPP.upconnect_:  # single PP may have multiple downconnects
            _PPP.upconnect_.append(up_PPP)

    if PPP in PPP_:
        PPP_.remove(PPP)  # merged PPP


def accum_PPP(PPP, derPP):  # accumulate params in PP

    if not PPP.params: PPP.params = derPP.params.copy()
    else:             accum_layer(PPP.params, derPP.params)
    PPP.nderP += 1
    PPP.mP += derPP.mP
    PPP.dP += derPP.dP
    PPP.Rdn += derPP.rdn
    derPP.PP = PPP
    PPP.derP__.append(derPP)  # PP is a group of x ooverlapping Ps, should be having no y?


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

# 5.11:

def comp_P_root(P__, rng, frng):  # vertically compares y-adjacent and x-overlapping Ps: blob slices, forming 2D derP__

    # if der+: P__ is last-call derP__, derP__=[], form new derP__
    # if rng+: P__ is last-call P__, accumulate derP__ with new_derP__
    # 2D array of derivative tuples from P__[n], P__[n-rng], sub-recursive:
    for P_ in P__:
        for P in P_:
            P.uplink_t, P.downlink_t = [[],[]],[[],[]]  # reset links and PP refs in the last layer only
            P.root = object

    for i, _P_ in enumerate(P__):  # higher compared row
        if i+rng < len(P__):  # rng=1 unless rng+ fork
            P_ = P__[i+rng]   # lower compared row
            for P in P_:
                if frng:
                    scan_P_(P, _P_, frng)  # rng+, compare at input derivation, which is single P
                else:
                    for derP in P.uplink_t[0]:  # der+, compare at new derivation, which is derP_
                        scan_P_(derP, _P_, frng)
            _P_ = P_

def scan_P_(P, _P_, frng):

    for _P in _P_:  # higher compared row
        if frng:
            fbreak = comp_olp(P, _P, frng)
        else:  # P is derP
            for _derP in _P.uplink_t[0]:
                fbreak = comp_olp(P, _derP, frng)  # comp derPs
        if fbreak:
            break

def comp_olp(P, _P, frng):  # P, _P can be derP, _derP;  also form sub_Pds for comp_dx?

    fbreak=0
    if P.x0 - 1 < (_P.x0 + _P.L) and (P.x0 + P.L) + 1 > _P.x0:  # test x overlap(_P,P) in 8 directions, all Ps of +ve derts:

        if isinstance(P, CPP) or isinstance(P, CderP):
            derP = comp_layer(_P, P)  # form vertical derivatives of horizontal P params
        else:
            derP = comp_P(_P, P)  # form higher vertical derivatives of derP or PP params
            derP.y = P.y
            if frng:  # accumulate derP through rng+ recursion:
                accum_layer(derP.params, P.params)
                P.uplink_t[1].append(derP)  # per P for form_PP
                _P.downlink_t[1].append(derP)

    elif (P.x0 + P.L) < _P.x0:  # no P xn overlap, stop scanning lower P_
        fbreak = 1

    return fbreak


def comp_P_rng(PP, rng, fPd):  # compare Ps over incremented range: P__[n], P__[n-rng], sub-recursive

    reversed_P__ = []  # to rescan P__ bottom-up
    for P_ in reversed(PP.P__):
        reversed_P__ += [P_]
        for P in P_:
            P.downlink_layers = [[],[]]  # reset only downlinks, uplinks will be updated below
            P.root = object  # reset links and PP refs in the last sub_P layer

    for i, P_ in enumerate(reversed_P__):  # scan bottom up
        if (i+rng) <= len(reversed_P__)-1:
            _P_ = reversed_P__[i+rng]
            for P in P_:
                new_mixed_uplink_ = []
                for _P in _P_:
                    derP = comp_P(_P, P)  # forms vertical derivatives of P params
                    new_mixed_uplink_ += [derP]       # new uplink per P
                    _P.downlink_layers[1] += [derP]        # add downlink per _P
                P.uplink_layers = [[], new_mixed_uplink_]  # update with new uplinks


# draft, combine with comp_P_rng?
def comp_P_der(PP, fPd):

    reversed_P__ = []
    for P_ in reversed(PP.P__):
        reversed_P__ += [P_]
        for P in P_:
            for derP in P.uplink_layers[1] + P.downlink_layers[1]:  # reset all derP:
                derP.uplink_layers = [[], []]
                derP.downlink_layers = [[], []]
                derP.root = object

    for i, P_ in enumerate(reversed_P__):  # scan bottom up
        if (i+1) <= len(reversed_P__)-1:
            _P_ = reversed_P__[i+1]  # upper row's P
            for P in P_:
                if P.uplink_layers[0][0]:  # non empty derP
                    derP = P.uplink_layers[0][0]
                    for _P in _P_:
                        if _P.uplink_layers[0][0]:  # non empty _derP
                           _derP = _P.uplink_layers[0][0]
                           dderP = comp_layer(_derP, derP)  # forms vertical derivatives of P params
                           derP.uplink_layers[1] += [dderP]
                           _derP.downlink_layers[1] += [dderP]

def comp_branches(P, _P_, frng):

    for _P in _P_:  # higher compared row
        if frng:
            if isinstance(P, CPP) or isinstance(P, CderP):  # rng+ fork for derPs, very unlikely
                comp_derP(_P, P)  # form higher vertical derivatives of derP or PP params
            else:
                comp_P(_P, P)  # form vertical derivatives of horizontal P params
        else:
            for _derP in _P.uplink_layers[-1]:
                comp_derP(_derP, P)  # P is actually derP, form higher vertical derivatives of derP or PP params


def comp_P_sub(iP__, frng):  # sub_recursion in PP, if frng: rng+ fork, else der+ fork

    P__ = [P_ for P_ in reversed(iP__)]  # revert to top-down
    if frng:
        uplinks__ = [[ [] for P in P_] for P_ in P__ ]  # init links per P
        downlinks__ = deepcopy(uplinks__)  # same format, all empty
    else:
        derP__ = [[] for P_ in P__[:-1]]  # init derP rows, exclude bottom P row

    for y, _P_ in enumerate( P__):  # always top-down, higher compared row
        for x, _P in enumerate(_P_):
            if frng:
                for derP in _P.downlink_layers[-1]:  # lower comparands are linked Ps at dy = rng
                    if derP.P in P__[y-1]:  # derP.P may not be in P__, which mean it is a branch and it is in another PP
                        P = derP.P
                        if isinstance(P, CPP) or isinstance(P, CderP):  # rng+ fork for derPs, very unlikely
                            derP = comp_derP(P, _P)  # form higher vertical derivatives of derP or PP params
                        else:
                            derP = comp_P(P, _P)  # form vertical derivatives of horizontal P params
                        # += links:
                        downlinks__[y][x] += [derP]
                        up_x = P__[y-1].index(P)  # index of P in P_ at y-1
                        uplinks__[y-1][up_x] += [derP]
            elif y < len(P__)-1:  # exclude last bottom P's derP
                for _derP in _P.downlink_layers[-1]:  # der+, compare at current derivation, which is derPs
                    for derP in _derP.P.downlink_layers[-1]:
                        dderP = comp_derP(_derP, derP)  # form higher vertical derivatives of derP or PP params
                        derP.uplink_layers[0] += [dderP]  # pre-init layer per derP
                        _derP.downlink_layers[0] += [dderP]
                        derP__[y].append(derP)
    if frng:
        for P_, uplinks_,downlinks_ in zip( P__, uplinks__, downlinks__):  # always top-down
            for P, uplinks, downlinks in zip_longest(P_, uplinks_, downlinks_, fillvalue=[]):
                P.uplink_layers += [uplinks]  # add link_layers to each P
                P.downlink_layers += [downlinks]
        return iP__
    else:
        return derP__

