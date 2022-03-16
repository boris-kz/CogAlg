'''
Comp_slice is a terminal fork of intra_blob.
-
It traces blob axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat or high-M blobs.
(high match: M / Ma, roughly corresponds to low gradient: G / Ga)
-
Vectorization is clustering of Ps + their derivatives (derPs) into PPs: patterns of Ps that describe an edge.
This process is a reduced-dimensionality (2D->1D) version of cross-comp and clustering cycle, common across this project.
As we add higher dimensions (2D alg, 3D alg), this dimensionality reduction is done in salient high-aspect blobs
(likely edges / contours in 2D or surfaces in 3D) to form more compressed "skeletal" representations of full-D patterns.
Most functions should be replaced by casting generic Search, Compare, Cluster functions
'''

from collections import deque
import sys
import numpy as np
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
ave_dangle = 2
ave_daangle = 2
ave_mP = 10
ave_dP = 10
ave_mPP = 10
ave_dPP = 10

param_names = ["x", "I", "M", "Ma", "L", "angle", "aangle"]  # angle = Dy, Dx; aangle = sin_da0, cos_da0, sin_da1, cos_da1; recompute Gs for comparison?
aves = [ave_dx, ave_I, ave_M, ave_Ma, ave_L, ave_G, ave_Ga, ave_mP, ave_dP]

class CP(ClusterStructure):

    layer0 = list  # 9 compared params: x, L, I, M, Ma, G, Ga, Ds( Dy, Dx, Sin_da0), Das( Cos_da0, Sin_da1, Cos_da1)
    # I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1 are summed from dert[3:], M, Ma from ave- g, ga
    # G, Ga are recomputed from Ds, Das; M, Ma are not restorable from G, Ga
    L = int  # redundant for convenience
    Rdn = int
    # if comp_dx:
    Mdx = int
    Ddx = int
    # new:
    x0 = int
    x = float  # median x
    y = int  # for visualization only
    sign = NoneType  # g-ave + ave-ga sign
    dert_ = list  # array of pixel-level derts, redundant to upconnect_, only per blob?
    upconnect_ = list
    downconnect_cnt = int
    # only in Pd:
    Pm = object  # reference to root P
    dxdert_ = list
    # only in Pm:
    Pd_ = list

class CderP(ClusterStructure):  # dert per CP param, please revise

    d = int
    m = int
    param_layers = list  # each call to slice_level_root compares all param_layers' params: i_, and adds a new param_layer: (m,d) per i
    # layer is flat but decoded by mapping (m,d)s to params in all lower layers
    # lower params are re-compared because they are summed in recursion / composition, so their value is different
    P = object   # lower comparand
    _P = object  # higher comparand
    PP = object  # FPP if flip_val, contains this derP
    # from comp_dx
    fdx = NoneType
    distance = int  # d_ave_x

class CPP(CP, CderP):  # derP params are inherited from P

    D = int  # summed derP.d
    M = int  # summed derP.m
    A = int  # summed from P.L s
    sign = bool
    upconnect_ = list
    upconnect_PP_ = list
    downconnect_cnt = int
    downconnect_cnt_PP = int
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    box = list   # for visualization only, original box before flipping
    mask__ = bool
    derP_ = list  # replaces dert__
    P_ = list  # replaces P__?
    Plevels = list  # replaces levels?
    param_layers = list
    sublayers = list

# Functions:

def comp_slice_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    segment_by_direction(blob, verbose=False)  # need to revise, it should form blob.dir_blobs, not FPPs
    for dir_blob in blob.dir_blobs:  # dir_blob should be Cblob, splice PPs across dir_blobs?

        P__ = slice_blob(dir_blob, verbose=False)  # cluster dir_blob.dert__ into 2D array of blob slices
        # comp_dx_blob(P__), comp_dx?

        derP_ = comp_P_blob(P__)  # scan_P_, comp_slice
        (PPm_, PPd_) = form_PP_(derP_)  # each PP is a stack of (P, derP)s from comp_P

        sub_recursion([PPm_, PPd_])  # rng+ comp_P in PPms, der+ comp_P in PPds, -> param_layer, form sub_PPs
        dir_blob.levels = [[PPm_, PPd_]]  # 1st composition level, each PP_ may be multi-layer from sub_recursion

        agglo_recursion(dir_blob)  # higher composition comp_PP in blob -> derPPs, form PPP., appends dir_blob.levels


def slice_blob(blob, verbose=False):  # forms horizontal blob slices: Ps, ~1D Ps, in select smooth-edge (high G, low Ga) blobs

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
                    layer0 = [ave_g-dert[1], ave_ga-dert[2], *dert[3:]]  # m, ma, dert[3:]: i, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1
                else:
                    # dert and _dert are not masked, accumulate P params from dert params:
                    layer0[1] += ave_g-dert[1]; layer0[2] += ave_ga-dert[2]  # M, Ma
                    for i, (Param, param) in enumerate(zip(layer0[2:], dert[3:]), start=2): # I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1
                        layer0[i] = Param + param
                    Pdert_.append(dert)
            elif not _mask:
                # _dert is not masked, dert is masked, terminate P:
                L = len(Pdert_)
                P_.append( CP(layer0= [x-(L-1)/2, L] + list(layer0), x0=x-(L-1), L=L, y=y, dert_=Pdert_))

            _mask = mask
        if not _mask:  # pack last P:
            L = len(Pdert_)
            P_.append( CP(layer0 = [x-(L-1)/2, L] + list(layer0), x0=x-(L-1), L=L, y=y, dert_=Pdert_))
        P__ += [P_]

    return P__


def comp_P_blob(P__):  # vertically compares y-adjacent and x-overlapping blob slices, forming derP__t

    derP_ = []
    _P_ = P__[0]  # upper row, local, no need for blob.P__?

    for P_ in P__[1:]:
        for P in P_:  # lower row
            for _P in _P_: # upper row
                # test for x overlap between P and _P in 8 directions, all Ps here are positive
                if (P.x0 - 1 < (_P.x0 + _P.L) and (P.x0 + P.L) + 1 > _P.x0):
                    # if P was not compared yet:
                    if not [1 for derP in P.upconnect_ if _P is derP._P]:
                        # form a tuple of vertical derivatives per P param tuple:
                        derP = comp_P(_P, P)
                        derP_.append(derP)  # per blob, for comp_slice_recursive only?
                        P.upconnect_.append(derP)  # per P, eval in form_PP
                        _P.downconnect_cnt += 1
                elif (P.x0 + P.L) < _P.x0:  # no P xn overlap, stop scanning lower P_
                    break
        _P_ = P_  # update prior _P_ to current P_

    return derP_


def comp_P(_P, P):  # forms vertical derivatives of params per P in _P.upconnect, conditional ders from norm and DIV comp

    # compared P params:
    x, L, M, Ma, I, Dx, Dy, sin_da0, cos_da0, sin_da1, cos_da1 = P.layer0
    _x, _L, _M, _Ma, _I, _Dx, _Dy, _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _P.layer0

    dx = _x - x;  mx = ave_dx(ave) - abs(dx)  # mean x shift, if dx: rx = dx / ((L+_L)/2)? no overlap, offset = abs(x0 -_x0) + abs(xn -_xn)?
    dI = _I - I;  mI = ave_I - abs(dI)
    dM = _M - M;  mM = min(_M, M)
    dMa = _Ma - Ma;  mMa = min(_Ma, Ma)  # dG, dM are directional, re-direct by dx?
    dL = _L - L * np.hypot(dx, 1); mL = min(_L, L)  # if abs(dx) > ave: adjust L as local long axis, no change in G,M
    # G, Ga:
    G = np.hypot(Dy, Dx); _G = np.hypot(_Dy, _Dx)  # compared as scalars
    dG = _G - G;  mG = min(_G, G)
    Ga = (cos_da0 + 1) + (cos_da1 + 1); _Ga = (_cos_da0 + 1) + (_cos_da1 + 1)  # gradient of angle, not sure about: +1 for all positives?
    # or Ga = np.hypot( np.arctan2(*Day), np.arctan2(*Dax)?
    dGa = _Ga - Ga;  mGa = min(_Ga, Ga)

    # comp angle:
    _sin = _Dy / _G; _cos = _Dx / _G
    sin  = Dy / G; cos = Dx / G
    sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
    dangle = np.arctan2(sin_da, cos_da)  # difference of angles
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
    maangle = ave_daangle - abs(daangle)  # match between aangles

    dlayer = abs(dx)-ave_dx + abs(dI)-ave_I + abs(G)-ave_G + abs(Ga)-ave_Ga + abs(dM)-ave_M + abs(dMa)-ave_Ma + abs(dL)-ave_L
    mlayer = mx + mI + mG + mGa + mM + mMa + mL + mangle + maangle

    param_layers = [[x, L, I, G, Ga, M, Ma, (sin_da, cos_da), (sin_da0, cos_da0, sin_da1, cos_da1)], # copy of layer0, the original in P
                    # layer0 is only for comp_slice_recursive, append if called, not here?
                    [dx, mx, dL, mL, dI, mI, dG, mG, dGa, mGa, dM, mM, dMa, mMa, dangle, mangle, daangle, maangle]]  # layer1
                    # or summable params only, all Gs are computed at termination?

    derP = CderP(m=mlayer, d=dlayer, param_layers=param_layers[1:], P=P, _P=_P)
    return derP


def form_PP_(derP_):  # form vertically contiguous patterns of patterns by derP sign, in blob or FPP

    PP_t = []
    for fPd in 0, 1:
        PP_ = []
        for derP in reversed(deepcopy(derP_)):  # scan bottom-up
            if not derP.P.downconnect_cnt and not isinstance(derP.PP, CPP):
                rdn = derP.P.Rdn + len(derP.P.upconnect_)
                # multiple upconnects form partially overlapping PPs, rnd needs to be proportional to overlap?
                if fPd: sign = derP.d > ave_dP * rdn
                else:   sign = derP.m > ave_mP * rdn
                PP = CPP(sign=sign)
                accum_PP(PP, derP)  # ?
                PP_.append(PP)  # initialized with derP.P.downconnect_cnt = 0
                if derP._P.upconnect_:
                    upconnect_2_PP_(derP, PP_, fPd)  # form PPs across _P upconnects
        PP_t.append(PP_)

    return PP_t  # PPm_, PPd_

def upconnect_2_PP_(iderP, PP_, fPd):
    '''
    compare sign of lower-layer iderP to the sign of its upconnects to form contiguous same-sign PPs
    '''
    confirmed_upconnect_ = []

    for derP in iderP._P.upconnect_:  # potential upconnects from previous call
        if derP not in iderP.PP.derP_:  # this may occur after Pp merging
            rdn = derP.P.Rdn + len(derP.P.upconnect_)
            if fPd: sign = derP.d > ave_dP * rdn
            else: sign = derP.m > ave_mP * rdn

            if iderP.PP.sign == sign:  # upconnect is having same sign
                if isinstance(derP.PP, CPP) and (derP.PP is not iderP.PP):  # upconnect is having existing PP, merge them
                    merge_PP(iderP.PP, derP.PP, PP_)
                else:  # accumulate derP in current PP
                    accum_PP(iderP.PP, derP)
                    confirmed_upconnect_.append(derP)
            else:  # sign changed
                if not isinstance(derP.PP, CPP):  # derP is root derP unless it already has FPP/PP
                    PP = CPP(sign=sign)  # param layer will be accumulated in accum_PP anyway
                    PP_.append(PP)  # pack every new PP initialized with derP.P with 0 downconnect count
                    accum_PP(PP, derP)
                    derP.P.downconnect_cnt = 0  # reset downconnect count for root derP
                # add connectivity between PPs
                iderP.PP.upconnect_PP_.append(derP.PP) # add new initialized PP as upconnect of current PP
                derP.PP.downconnect_cnt_PP += 1  # add downconnect count to newly initialized PP

            if derP._P.upconnect_:
                upconnect_2_PP_(derP, PP_, fPd)  # recursive compare sign of next-layer upconnects

    iderP._P.upconnect_ = confirmed_upconnect_


def merge_PP(_PP, PP, PP_):  # merge PP into _PP

    for derP in PP.derP_:
        if derP not in _PP.derP_:
            accum_PP(_PP, derP)  # accumulate params
    _PP.Rdn += PP.Rdn
    if PP in PP_:
        PP_.remove(PP)  # remove merged PP

def accum_PP(PP, derP):  # accumulate params in PP

    if not PP.param_layers:  # if param layer is empty, copy over the derP's param layers
        PP.param_layers = derP.param_layers.copy()
    else:
        for _param_layer, param_layer in zip(PP.param_layers, derP.param_layers):
            for i, (_param, param) in enumerate(zip(_param_layer, param_layer)):

                if isinstance(_param, tuple):
                    if len(_param) == 2:  # (sin_da, cos_da)
                        _sin_da, _cos_da = _param
                        sin_da, cos_da = param
                        sum_sin_da = (cos_da * _sin_da) + (sin_da * _cos_da)  # sin(α + β) = sin α cos β + cos α sin β
                        sum_cos_da = (cos_da * _cos_da) - (sin_da * _sin_da)  # cos(α + β) = cos α cos β - sin α sin β
                        _param_layer[i] = (sum_sin_da, sum_cos_da)
                    else:  # (sin_da0, cos_da0, sin_da1, cos_da1)
                        _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _param
                        sin_da0, cos_da0, sin_da1, cos_da1 = param
                        sum_sin_da0 = (cos_da0 * _sin_da0) + (sin_da0 * _cos_da0)  # sin(α + β) = sin α cos β + cos α sin β
                        sum_cos_da0 = (cos_da0 * _cos_da0) - (sin_da0 * _sin_da0)  # cos(α + β) = cos α cos β - sin α sin β
                        sum_sin_da1 = (cos_da1 * _sin_da1) + (sin_da1 * _cos_da1)
                        sum_cos_da1 = (cos_da1 * _cos_da1) - (sin_da1 * _sin_da1)
                        _param_layer[i] = (sum_sin_da0, sum_cos_da0, sum_sin_da1, sum_cos_da1)
                else:  # scalar
                    _param_layer[i] += param
    PP.Rdn += derP.P.Rdn  # add rdn, add derP._P.Rdn too?
    PP.L += 1
    PP.derP_.append(derP)  # add derP to Pp
    derP.PP = PP           # update reference


# draft
def sub_recursion(PP_t):  # compares param_layers of consecutive derPs inside generic PP, forming higher param_layer

    # der+:
    for PP in PP_t[1]:  # PP is generic higher-composition pattern, P is generic lower-composition pattern
        der_incr(PP)
        # param layers der+:
        if PP.D > aveB*PP.Rdn and PP.L > len(PP.param_layers):  # (need 3 Ps compute layer2, etc.)
            for derP in PP.derP_:
                if derP.P.downconnect_cnt == 0:  # lowest derP in PP, else it was scanned in lower derP.P.upconnect_
                    for _derP in derP._P.upconnect_:
                        if _derP.m > ave_mP:
                            new_layer = []
                            for _layer, layer in zip(_derP.param_layers, derP.param_layers):
                                # compare next layer:
                                mlayer = comp_layer(_layer, layer, new_layer)  # append new_layer with lower-layer derivatives
                                _derP.m += mlayer
                                if _derP.m < ave_mP:
                                    break
                            if new_layer:
                                _derP.param_layers += [new_layer]; derP.param_layers += [new_layer]  # layer0 remains in P
                        derP = _derP
    # rng+:
    for PP in PP_t[0]:  # PP is generic higher-composition pattern, P is generic lower-composition pattern
        rng_incr(PP)
        # rng is number of Ps compared per _P, top param_layer is accumulated per comp, vs. compared in der+
        rng = PP.M / ave_mPP*PP.Rdn
        if rng > PP.rng and PP.L > rng*2:
            pass

 # draft
def rng_incr(PP):

    # to pack derP.Ps into P__, i think it should be simpler if we pack Ps per PP?
    ys = np.unique([derP.P.y for derP in PP.derP_])
    min_y = min(ys)
    P__ = [[] for _ in range(len(ys))]
    # to pack P into different P_ based on their y location:
    for derP in PP.derP_:
        y = derP.P.y - min_y
        if derP.P not in P__[y]:  # prevent same P, it is possible with multiple upcoinnects
            P__[y].append(derP.P)

    sub_derP_ = comp_P_blob(P__)  # scan_P_, comp_slice;  splice PPs across dir_blobs?
    sub_PP_t = form_PP_(sub_derP_)  # each PP is a stack of (P, derP)s from comp_P
    PP.sublayers = [sub_PP_t]
    PP.sublayers += sub_recursion(sub_PP_t)  # rng+ comp_P in PPms, der+ comp_P in PPds, -> param_layer, form sub_PPs


def der_incr(PP):

    # to pack derP.Ps into P__, i think it should be simpler if we pack Ps per PP?
    ys = np.unique([derP.P.y for derP in PP.derP_])
    min_y = min(ys)
    P__ = [[] for _ in range(len(ys))]
    # to pack P into different P_ based on their y location:
    for derP in PP.derP_:
        y = derP.P.y - min_y
        if derP.P not in P__[y]:  # prevent same P, it is possible with multiple upcoinnects
            P__[y].append(derP.P)

    sub_derP_ = comp_P_blob(P__)  # scan_P_, comp_slice;  splice PPs across dir_blobs?
    sub_PP_t = form_PP_(sub_derP_)  # each PP is a stack of (P, derP)s from comp_P
    PP.sublayers = [sub_PP_t]
    PP.sublayers += sub_recursion(sub_PP_t)  # rng+ comp_P in PPms, der+ comp_P in PPds, -> param_layer, form sub_PPs


def agglo_recursion(blob):  # compositional recursion, per blob.Plevel

    PP_ = blob.levels[-1]
    PPP_ = []
    # for fiPd, PP_ in enumerate(PP_t): fiPd = fiPd % 2
    if len(PP_)>1:  # at least 2 comparands
        derPP_ = comp_Plevel(PP_)
        PPP_ = form_Plevel(derPP_)

    blob.levels.append(PPP_)  # levels of dir_blob are Plevels
    if len(PPP_) > 4:
        agglo_recursion(blob)

def comp_Plevel(PP_):

    derPP_ = []
    for PP in PP_:
        for _PP in PP.upconnect_PP_:
            # upconnect is PP
            if not [1 for derPP in PP.upconnect_ if PP is derPP.P]:

                derPP = comp_PP(_PP, PP)
                derPP_.append(derPP)
                PP.upconnect_.append(derPP)
                _PP.downconnect_cnt += 1
    return derPP_


def comp_PP(_PP, PP):
    # loop each param and compare it?
    # not possible without loop, since number of param is getting bigger with each new param layer depth

    derPP = CderP(_P=_PP, P=PP)
    return derPP


def form_Plevel(derPP_):

    PPP_ = []
    for derPP in deepcopy(derPP_):
        # last-row derPPs downconnect_cnt == 0
        rdn = derPP.P.Rdn + len(derPP.P.upconnect_)
        if not derPP.P.downconnect_cnt and not isinstance(derPP.PP, CPP) and derPP.m > ave_mP * rdn:  # root derPP was not terminated in prior call
            PPP = CPP()
            accum_PP(PPP,derPP)
            if derPP._P.upconnect_:
                upconnect_2_PP_(derPP, PPP_)  # form PPPs across _P upconnects
    return PPP_


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

# old:
def comp_slice(_P, P):  # forms vertical derivatives of derP params, and conditional ders from norm and DIV comp

    x0, Dx, Dy, L, = P.x0, P.Dx, P.Dy, P.L
    # params per comp branch, add angle params
    _x0, _Dx, _Dy,_dX, _L = _P.x0, _P.Dx, _P.Dy, _P.dX, _P.L

    dX = (x0 + (L-1) / 2) - (_x0 + (_L-1) / 2)  # x shift: d_ave_x, or from offsets: abs(x0 - _x0) + abs(xn - _xn)?

    if dX > ave_dX:  # internal comp is higher-power, else two-input comp not compressive?
        xn = x0 + L - 1
        _xn = _x0 + _L - 1
        mX = min(xn, _xn) - max(x0, _x0)  # overlap = abs proximity: summed binary x match
        rX = dX / mX if mX else dX*2  # average dist / prox, | prox / dist, | mX / max_L?

    ddX = dX - _dX  # long axis curvature, if > ave: ortho eval per P, else per PP_dX?
    mdX = min(dX, _dX)  # dX is inversely predictive of mP?

    # is this looks better? or it would better if we stick to the old code?
    difference = P.difference(_P)   # P - _P
    match = P.min_match(_P)         # min of P and _P
    abs_match = P.abs_min_match(_P) # min of abs(P) and abs(_P)

    dL = difference['L'] # L: positions / sign, dderived: magnitude-proportional value
    mL = match['L']
    dM = difference['M'] # use abs M?  no Mx, My: non-core, lesser and redundant bias?
    mM = match['M']

    # min is value distance for opposite-sign comparands, vs. value overlap for same-sign comparands
    dDy = difference['Dy']  # Dy per sub_P by intra_comp(dx), vs. less vertically specific dI
    mDy = abs_match['Dy']
    # no comp G: Dy, Dx are more specific:
    dDx = difference['Dx']  # same-sign Dx if Pd
    mDx = abs_match['Dx']

    if dX * P.G > ave_ortho:  # estimate params of P locally orthogonal to long axis, maximizing lateral diff and vertical match
        # diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/orthogonalization.png
        # Long axis is a curve of connections between ave_xs: mid-points of consecutive Ps.

        # Ortho virtually rotates P to connection-orthogonal direction:
        hyp = np.hypot(dX, 1)  # ratio of local segment of long (vertical) axis to dY = 1
        L = L / hyp  # orthogonal L
        # combine derivatives in proportion to the contribution of their axes to orthogonal axes:
        # contribution of Dx should increase with hyp(dX,dY=1), this is original direction of Dx:
        Dy = (Dy / hyp + Dx * hyp) / 2  # estimated along-axis D
        Dx = (Dy * hyp + Dx / hyp) / 2  # estimated cross-axis D
        '''
        alternatives:
        oDy = (Dy * hyp - Dx / hyp) / 2;  oDx = (Dx / hyp + Dy * hyp) / 2;  or:
        oDy = hypot( Dy / hyp, Dx * hyp);  oDx = hypot( Dy * hyp, Dx / hyp)
        '''
        # recompute difference and match
        dL = _L - L
        mL = min(_L, L)
        dDy = _Dy - Dy
        mDy = min(abs(_Dy), abs(Dy))
        dDx = _Dx - Dx
        mDx = min(abs(_Dx), abs(Dx))

    if (Dx > 0) != (_Dx > 0): mDx = -mDx
    if (Dy > 0) != (_Dy > 0): mDy = -mDy

    dDdx, dMdx, mDdx, mMdx = 0, 0, 0, 0
    if P.dxdert_ and _P.dxdert_:  # from comp_dx
        fdx = 1
        dDdx = difference['Ddx']
        mDdx = abs_match['Ddx']
        if (P.Ddx > 0) != (_P.Ddx > 0): mDdx = -mDdx
        # Mdx is signed:
        dMdx = match['Mdx']
        mMdx = -abs_match['Mdx']
        if (P.Mdx > 0) != (_P.Mdx > 0): mMdx = -mMdx
    else:
        fdx = 0
    # coeff = 0.7 for semi redundant parameters, 0.5 for fully redundant parameters:
    dP = ddX + dL + 0.7*(dM + dDx + dDy)  # -> directional PPd, equal-weight params, no rdn?
    # correlation: dX -> L, oDy, !oDx, ddX -> dL, odDy ! odDx? dL -> dDx, dDy?
    if fdx: dP += 0.7*(dDdx + dMdx)

    mP = mdX + mL + 0.7*(mM + mDx + mDy)  # -> complementary PPm, rdn *= Pd | Pm rolp?
    if fdx: mP += 0.7*(mDdx + mMdx)
    mP -= ave_mP * ave_rmP ** (dX / L)  # dX / L is relative x-distance between P and _P,

    derP = CderP(P=P, _P=_P, mP=mP, dP=dP, dX=dX, mL=mL, dL=dL, mDx=mDx, dDx=dDx, mDy=mDy, dDy=dDy)
    P.derP = derP

    if fdx:
        derP.fdx=1; derP.dDdx=dDdx; derP.mDdx=mDdx; derP.dMdx=dMdx; derP.mMdx=mMdx

    '''
    min comp for rotation: L, Dy, Dx, no redundancy?
    mParam weighting by relative contribution to mP, /= redundancy?
    div_f, nvars: if abs dP per PPd, primary comp L, the rest is normalized?
    '''
    return derP

# not revised, remove ifs, etc:
def comp_layer(_layer, layer, new_layer):

    param_num = len(_layer)
    layer_num = param_num / 9  # 9 compared params
    param_layer = []
    hyps = []
    mP = 0
    dP = 0  # not useful?
    for i, (_param, param) in enumerate(zip(_layer, layer)):

        param_type = int(i/ (2 ** (layer_num-1)))  # param type, range from 0 - 8 for 9 compared params

        if param_type == 0:  # x
            _x = param; x = param
            dx = _x - x; mx = ave_dx - abs(dx)
            param_layer.append(dx); param_layer.append(mx)
            hyps.append(np.hypot(dx, 1))
            dP += dx; mP += mx

        elif param_type == 1:  # I
            _I = _param; I = param
            dI = _I - I; mI = ave_dI - abs(dI)
            param_layer.append(dI); param_layer.append(mI)
            dP += dI; mP += mI

        elif param_type == 2:  # G
            hyp = hyps[i%param_type]
            _G = _param; G = param
            dG = _G - G/hyp;  mG = min(_G, G)  # if comp_norm: reduce by hypot
            param_layer.append(dG); param_layer.append(mG)
            dP += dG; mP += mG

        elif param_type == 3:  # Ga
            _Ga = _param; Ga = param
            dGa = _Ga - Ga;  mGa = min(_Ga, Ga)
            param_layer.append(dGa); param_layer.append(mGa)
            dP += dGa; mP += mGa

        elif param_type == 4:  # M
            hyp = hyps[i%param_type]
            _M = _param; M = param
            dM = _M - M/hyp;  mM = min(_M, M)
            param_layer.append(dM); param_layer.append(mM)
            dP += dM; mP += mM

        elif param_type == 5:  # Ma
            _Ma = _param; Ma = param
            dMa = _Ma - Ma;  mMa = min(_Ma, Ma)
            param_layer.append(dMa); param_layer.append(mMa)
            dP += dMa; mP += mMa

        elif param_type == 6:  # L
            hyp = hyps[i%param_type]
            _L = _param; L = param
            dL = _L - L/hyp;  mL = min(_L, L)
            param_layer.append(dL); param_layer.append(mL)
            dP += dL; mP += mL

        elif param_type == 7:  # angle, (sin_da, cos_da)
            if isinstance(_param, tuple):  # (sin_da, cos_da)
                 _sin_da, _cos_da = _param; sin_da, cos_da = param
                 sin_dda = (cos_da * _sin_da) - (sin_da * _cos_da)  # sin(α - β) = sin α cos β - cos α sin β
                 cos_dda = (cos_da * _cos_da) + (sin_da * _sin_da)  # cos(α - β) = cos α cos β + sin α sin β
                 dangle = (sin_dda, cos_dda)  # da
                 mangle = ave_dangle - abs(np.arctan2(sin_dda, cos_dda))  # ma is indirect match
                 param_layer.append(dangle); param_layer.append(mangle)
                 dP += np.arctan2(sin_dda, cos_dda); mP += mangle
            else: # m or scalar
                _mangle = _param; mangle = param
                dmangle = _mangle - mangle;  mmangle = min(_mangle, mangle)
                param_layer.append(dmangle); param_layer.append(mmangle)
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
                param_layer.append(daangle); param_layer.append(maangle)
                dP += daangle; mP += maangle

            else:  # m or scalar
                _maangle = _param; maangle = param
                dmaangle = _maangle - maangle;  mmaangle = min(_maangle, maangle)
                param_layer.append(dmaangle); param_layer.append(mmaangle)
                dP += dmaangle; mP += mmaangle

    new_layer += [param_layer]

    return mP