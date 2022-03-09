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

ave_inv = 20 # ave inverse m, change to Ave from the root intra_blob?
ave_min = 5  # ave direct m, change to Ave_min from the root intra_blob?
ave_g = 30  # change to Ave from the root intra_blob?
flip_ave = .1
flip_ave_FPP = 0  # flip large FPPs only (change to 0 for debug purpose)
div_ave = 200
ave_rmP = .7  # the rate of mP decay per relative dX (x shift) = 1: initial form of distance
ave_ortho = 20
ave_ga = 0.78  # ga at 22.5 degree
# comp_param:
ave_dI = 10  # same as ave_inv
ave_dM = 10  # same as ave_min, replace the rest with coefs:
ave_dG = 10
ave_dangle = 10  # related to dx?
ave_dL = 10
ave_dx = 10  # difference between median x coords of consecutive Ps

param_names = ["x", "I", "angle", "G", "M", "L"]  # angle = (Dy, Dx)
aves = [ave_dx, ave_dI, ave_dangle, ave_dG, ave_dM, ave_dL]

class CP(ClusterStructure):

    # dert params summed in slice:
    I = int
    Dy = int
    Dx = int
    G = int
    M = int  # summed ave - abs(dx), not restorable from Dx.
    L = int
    # comp_angle:
    Ddyy = int
    Ddxy = int
    Da = int
    # comp_dx:
    Mdx = int
    Ddx = int
    # new:
    x0 = int
    x = int  # median x
    dX = int  # shift of average x between P and _P, if any
    y = int  # for visualization only
    sign = NoneType  # sign of gradient deviation
    dert_ = list   # array of pixel-level derts: (p, dy, dx, g, m), extended in intra_blob
    upconnect_ = list
    downconnect_cnt = int
    derP = object # derP object reference
    # only in Pd:
    Pm = object  # reference to root P
    dxdert_ = list
    # only in Pm:
    Pd_ = list

class CderP(ClusterStructure):  # dert per CP param, please revise

    d = int
    m = int
    param_layer0 = lambda:{"x":0, "I":0, "G":0, "M":0, "L":0, "DyDx":[0,0]}
    param_layer1 = lambda:{"x":[0,0], "I":[0,0], "G":[0,0], "M":[0,0], "L":[0,0], "angle":[0,0]}
    P = object   # lower comparand
    _P = object  # higher comparand
    PP = object  # FPP if flip_val, contains this derP
    # from comp_dx
    fdx = NoneType
    distance = int  # d_ave_x
    # for higher level:
    dmm = int
    mmm = int
    ddm = int
    mdm = int
    param_layer01 = lambda:{"x":[0,0], "I":[0,0], "G":[0,0], "M":[0,0], "L":[0,0], "angle":[0,0], "DdyDdx":[0,0]}
    param_layer11 = lambda:{"x":[0,0,0,0], "I":[0,0,0,0], "G":[0,0,0,0], "M":[0,0,0,0], "L":[0,0,0,0], "angle":[0,0], "DdyDdx":[0,0]}
    '''
    Replace with param_layers: each call to slice_level_root compares all param_layers' params: i_, and adds a new param_layer: (m,d) per i.
    (lower params are re-compared because they are summed in recursion / composition, so their value is different)
    This new layer is formally flat, but implicit nesting can be decoded by mapping (m,d)s to i params in lower param layers.
    '''

class CPP(CP, CderP):  # derP params are inherited from P

    A = int  # summed from P.L s
    upconnect_PP_ = list
    upconnect_ = list
    downconnect_cnt = int
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    box = list   # for visualization only, original box before flipping
    dert__ = list
    mask__ = bool
    # Pp params
    derP_ = list
    P__ = list
    param_layer0 = lambda:{"x":0, "I":0, "G":0, "M":0, "L":0, "DyDx":[0,0]}
    param_layer1 = lambda:{"x":[0,0], "I":[0,0], "G":[0,0], "M":[0,0], "L":[0,0], "angle":[0,0]}

class CderPP(ClusterStructure):
    # obsolete
    param_layer01 = dict
    param_layer1 = dict
    param_layer11 = dict
    PP = object
    _PP = object
    mmPP = int
    dmPP = int
    mdPP = int
    ddPP = int

# Functions:
'''-
workflow:
intra_blob -> slice_blob(blob) -> derP_ -> PP,
if flip_val(PP is FPP): pack FPP in blob.PP_ -> flip FPP.dert__ -> slice_blob(FPP) -> pack PP in FPP.PP_
else       (PP is PP):  pack PP in blob.PP_
'''

def comp_slice_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    segment_by_direction(blob, verbose=False)
    P__ = slice_blob(blob, verbose=False)  # 2D array of blob slices
    derP_ = comp_slice_blob(P__)  # scan_P_, comp_slice
    form_PP_(blob, derP_)
    # higher comp orders:
    slice_level_root(blob)


def slice_blob(blob, verbose=False):  # forms horizontal blob slices: Ps, ~1D Ps, in select smooth-edge (high G, low Ga) blobs

    mask__ = blob.mask__  # same as positive sign here
    dert__ = zip(*blob.dert__)  # convert 10-tuple of 2D arrays into 1D array of 10-tuple blob rows
    dert__ = [zip(*dert_) for dert_ in dert__]  # convert 1D array of 10-tuple rows into 2D array of 10-tuples per blob

    height, width = mask__.shape
    if verbose: print("Converting to image...")
    P__ = []  # blob of Ps

    for y, (dert_, mask_) in enumerate( zip(dert__, mask__)):  # unpack lines
        P_ = []  # line of Ps
        _mask = True
        for x, (dert, mask) in enumerate( zip(dert_, mask_)):  # unpack derts: tuples of 10 params
            if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()
            # masks: if 0,_1: P initialization, if 0,_0: P accumulation, if 1,_0: P termination:
            if not mask:
                if _mask:  # initialize P with first unmasked dert:
                    P = CP(I=dert[0], Dy=dert[1], Dx=dert[2], G=dert[3], Dydy=dert[5], Dxdy=dert[6], Dydx=dert[7], Dxdx=dert[8], Ga=dert[9],
                           x0=x, L=1, y=y, dert_=[dert])  # sign is always positive, else masked
                else:
                    # dert and _dert are not masked, accumulate P params:
                    P.accumulate(I=dert[0], Dy=dert[1], Dx=dert[2], G=dert[3],Dydy=dert[5], Dxdy=dert[6], Dydx=dert[7], Dxdx=dert[8], Ga=dert[9], L=1)
                    P.dert_.append(dert)
            elif not _mask:
                # _dert is not masked, dert is masked, terminate P:
                P.x = P.x0 + (P.L - 1) /2
                P_.append(P)
            _mask = mask

        if not _mask: P_.append(P)  # pack last P
        P__ += [P_]
    return P__

def comp_slice_blob(P__):  # vertically compares y-adjacent and x-overlapping blob slices, forming derP__t

    derP_ = []  # derPs per blob
    _P_ = P__[0]  # upper row

    for P_ in P__[1:]:
        for P in P_:  # lower row
            for _P in _P_: # upper row
                # test for x overlap between P and _P in 8 directions, all Ps here are positive
                if (P.x0 - 1 < (_P.x0 + _P.L) and (P.x0 + P.L) + 1 > _P.x0):
                    # upconnect is derP or dirP:
                    if not [1 for derP in P.upconnect_ if P is derP.P]:
                        # P was not compared before
                        derP = comp_slice(_P, P)  # form vertical derivatives per param
                        derP_.append(derP)
                        P.upconnect_.append(derP)
                        _P.downconnect_cnt += 1
                elif (P.x0 + P.L) < _P.x0:  # no P xn overlap, stop scanning lower P_
                    break
        _P_ = P_  # update prior _P_ to current P_
    return derP_


def comp_slice(_P, P):  # forms vertical derivatives of P params, conditional ders from norm and DIV comp
    # compared params:
    x, L, I, G, M, Ga, Ma, Dx, Dy, Sda0, Cda0, Sda1, Cda1 = P.x, P.L, P.I, P.G, P.M, P.Ga, P.Ma, P.Dx, P.Dy, P.Sda0, P.Cda0, P.Sda1, P.Cda1
    _x, _L, _I, _G, _M, _Ga, _Ma, _Dx, _Dy, _Sda0, _Cda0, _Sda1, _Cda1 = _P.x, _P.L, _P.I, _P.G, _P.M, _P.Ga, _P.Ma, _P.Dx, _P.Dy, _P.Sda0, _P.Cda0, _P.Sda1, _P.Cda1

    dx = _x - x;  mx = ave_dx - abs(dx)  # mean x shift, if dx: rx = dx / ((L+_L)/2)? no overlap, offset = abs(x0 -_x0) + abs(xn -_xn)?
    dI = _I - I;  mI = ave_dI - abs(dI)
    dG = _G - G;  mG = min(_G, G)  # dG, dM are directional, re-direct by dx?
    dM = _M - M;  mM = min(_M, M)
    dL = _L - L*np.hypot(dx, dy=1); mL = min(_L, L)  # if abs(dx) > ave: L is local long axis, no change in G,M
    # comp angle:
    _norm_G = np.hypot(_Dy, _Dx) - (ave_dG * _L)
    norm_G  = np.hypot(Dy, Dx) - (ave_dG * L)
    _absG = max(1, _norm_G + (ave_dG * _L))
    absG  = max(1, norm_G + (ave_dG * L))
    _sin = _Dy / _absG; _cos = _Dx / _absG
    sin  = Dy / absG; cos = Dx / absG
    sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
    dangle = (sin_da, cos_da)  # da
    mangle = ave_dangle - abs(np.arctan2(sin_da, cos_da))  # ma is indirect match

    # add comp angle_diff here

    mP = mx + mI + mG + mM + mL + mangle
    dP = dx + dI + dG + dM + dL + dangle  # placeholder for now
    # pseudo:
    param_layer0 = {"x":x, "I":mI, "G":mG, "M":mM, "L":mL, "DyDx":(Dy,Dx)}
    param_layer1 = {"x":(mx,dx), "I":(mI,dI), "G":(mG,dG), "M":(mM,dM), "L":(mL,dL), "angle":(mangle,dangle)}

    derP = CderP(m=mP, d=dP, param_layer0=param_layer0, param_layer1=param_layer1, P=P, _P=_P)
    return derP


def form_PP_(blob, derP_):  # form vertically contiguous patterns of patterns by derP sign, in blob or FPP

    blob.derP_ = derP_
    PP_t = []  # flat version

    for fPpd in 0, 1:
        PP_ = []
        for derP in deepcopy(derP_):
            # last-row derPs downconnect_cnt == 0
            if not derP.P.downconnect_cnt and not isinstance(derP.PP, CPP):  # root derP was not terminated in prior call
                PP = CPP()
                accum_PP(PP,derP)
                if derP._P.upconnect_:
                    upconnect_2_Pp_(derP, PP_, fPpd)  # form PPs across _P upconnects
                else:
                    PP_.append(derP.Pp)  # terminate PP
        PP_t.append(PP_)
    blob.slice_levels.append(PP_t)

def upconnect_2_Pp_(iderP, PP_, fPpd):
    '''
    compare sign of lower-layer iderP to the sign of its upconnects to form contiguous same-sign PPs
    '''
    confirmed_upconnect_ = []

    for derP in iderP._P.upconnect_:  # potential upconnects from previous call
        if derP not in iderP.PP.derP_:  # this may occur after Pp merging

            if fPpd: same_sign = (iderP.d>0) == (derP.d>0)
            else: same_sign = (iderP.m>0) == (derP.m>0)

            if same_sign:  # upconnect derP has different Pp, merge them
                if isinstance(derP.PP, CPP) and (derP.PP is not iderP.PP):
                    merge_PP(iderP.PP, derP.PP, PP_)
                else:  # accumulate derP in current PP
                    accum_PP(iderP.PP, derP)
                    confirmed_upconnect_.append(derP)
            else:
                if not isinstance(derP.PP, CPP):  # sign changed, derP is root derP unless it already has FPP/PP
                    PP = CPP()
                    accum_PP(PP, derP)
                    derP.P.downconnect_cnt = 0  # reset downconnect count for root derP
                iderP.PP.upconnect_PP_.append(derP.PP) # add new initialized PP as upconnect of current PP
                derP.PP.downconnect_cnt += 1  # add downconnect count to newly initialized PP

            if derP._P.upconnect_:
                upconnect_2_Pp_(derP, PP_, fPpd)  # recursive compare sign of next-layer upconnects
            elif derP.PP is not iderP.PP and derP.P.downconnect_cnt == 0:
                PP_.append(derP.PP)  # terminate PP (not iPP) at the sign change

    iderP._P.upconnect_ = confirmed_upconnect_

    if iderP.P.downconnect_cnt == 0:
        PP_.append(iderP.PP)  # iPp is terminated after all upconnects are checked


def merge_PP(_PP, PP, PP_):  # merge PP into _PP

    for derP in PP.derP_:
        if derP not in _PP.derP_:
            accum_PP(_PP, derP)     # accumulate params
    if PP in PP_:
        PP_.remove(PP)  # remove merged PP


def accum_PP(PP, derP):  # accumulate params in PP

    # extended from accum_Dert:
    # def accum_Dert(Dert: dict, **params) -> None:
    #     Dert.update({param: Dert[param] + value for param, value in params.items()})
    for param, value in derP.param_layer0.items():  # param_layer0
        if param == "DyDx":
            PP.param_layer0[param][0] += value[0]  # Dy
            PP.param_layer0[param][1] += value[1]  # Dx
        else:
            PP.param_layer0[param] += value
    for param, value in derP.param_layer1.items():  # param_layer1
        PP.param_layer1[param][0] += value[0]  # accumulate m
        PP.param_layer1[param][1] += value[1]  # accumulate d

    PP.L += 1
    PP.derP_.append(derP)  # add derP to Pp
    derP.PP = PP           # update reference


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


def slice_level_root(blob):

    PP_t = blob.slice_levels[-1]
    PPP_, PPP_t = [], []
    nextended = 0  # number of extended-depth

    for fiPd, PP_ in enumerate(PP_t):
        fiPd = fiPd % 2
        if len(PP_)>1:  # at least 2 comparands
            nextended += 1
            for fPd in 0, 1:
                derPP_ = comp_PP_blob(PP_)
                PPP_ = form_PPP_(derPP_, fPd)
                PPP_t.append(PPP_)
        else:
            PPP_t += [[] for _ in range(2)]  # align indexing, replace with count of missing

    blob.slice_levels.append(PPP_t)
    if len(PPP_) / max(nextended,1) < 4:
        slice_level_root(blob)


def comp_PP_blob(PP_):

    derPP_ = []
    for PP in PP_:
        for _PP in PP.upconnect_PP_:
            # upconnect is derP or dirP:
            if not [1 for derPP in PP.upconnect_ if PP is derPP.P]:
                derPP = comp_PP(_PP, PP)
                derPP_.append(derPP)
                PP.upconnect_.append(derPP)
                _PP.downconnect_cnt += 1
    return derPP_

# replace with operations on param_layers:

def comp_PP(_PP, PP):  # forms vertical derivatives of PP params, conditional ders from norm and DIV comp

    # compute param_layer01 from param_layer0
    _x, _I, _G, _M, _L, (_Dy, _Dx) = _PP.param_layer0.values
    x, I, G, M, L, (Dy, Dx) = PP.param_layer0.values

    dx = _x - x  # mean x shift, or from offsets: abs(x0 - _x0) + abs(xn - _xn)?
    mx = ave_dx - abs(dx)  # if mx+dx: rx = dx / ((L+_L)/2), overlap and offset don't matter?
    # if abs(dx) > ave: comp_norm:
    hyp = np.hypot(dx, 1)  # ratio of local segment of long (vertical) axis to dy = 1
    dI = _I - I;  mI = ave_dI - abs(dI)
    dG = _G - G/hyp;  mG = min(_G, G)  # if comp_norm: reduce by hypot
    dM = _M - M/hyp;  mM = min(_M, M)
    dL = _L - L/hyp;  mL = min(_L, L)

    # comp_a, not sure, please check:
    _norm_G = np.hypot(_Dy, _Dx) - (ave_dG * _L)
    norm_G  = np.hypot(Dy, Dx) - (ave_dG * L)
    _absG = max(1, _norm_G + (ave_dG * _L))
    absG  = max(1, norm_G + (ave_dG * L))
    _sin = _Dy / _absG; _cos = _Dx / _absG
    sin  = Dy / absG; cos = Dx / absG
    # angle = [sin, cos]; _angle = [_sin, _cos]
    sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
    dangle = np.arctan2(sin_da, cos_da)  # da
    mangle = ave_dangle - abs(dangle)  # indirect match, ma

    mP = mx + mI + mG + mM + mL + mangle
    dP = dx + dI + dG + dM + dL + dangle  # placeholder for now

    param_layer01 = {"x":(mx,dx), "I":(mI,dI), "G":(mG,dG), "M":(mM,dM), "L":(mL,dL), "angle":(mangle,dangle), "DdyDdx":[sin_da,cos_da]}

    # compute param_layer11 from param_layer1
    (_mx, _dx), (_mI, _dI), (_mG, _dG), (_mM, _dM), (_mL,_dL), (_mangle, _dangle), (_sin_da, _cos_da) = _PP.param_layer1.values
    (mx, dx), (mI, dI), (mG, dG), (mM, dM), (mL,dL), (mangle, dangle), (sin_da, cos_da) = PP.param_layer1.values
    # x
    dmx = _mx - mx
    mmx = ave_dx - abs(dmx)
    ddx = _dx - dx
    mdx = ave_dx - abs(ddx)
    mhyp = np.hypot(dmx, 1)  # ratio of local segment of long (vertical) axis to dy = 1
    dhyp = np.hypot(ddx, 1)
    # I
    dmI = _mI - mI; mmI = ave_dI - abs(dmI)
    ddI = _dI - dI; mdI = ave_dI - abs(ddI)
    # G
    dmG = _mG - mG/mhyp
    mmG = min(_mG, mG)
    ddG = _dG - dG/dhyp
    mdG = min(_dG, dG)
    # M
    dmM = _mM - mM/mhyp
    mmM = min(_mM, mM)
    ddM = _dM - dM/dhyp
    mdM = min(_dM, dM)
    # L
    dmL = _mL - mL/mhyp
    mmL = min(_mL, mL)
    ddL = _dL - dL/dhyp
    mdL = min(_dL, dL)
    # angle (not sure, there's no cos_ma, sin_ma)
    sin_dda = (cos_da * _sin_da) - (sin_da * _cos_da)  # sin(α - β) = sin α cos β - cos α sin β
    cos_dda = (cos_da * _cos_da) + (sin_da * _sin_da)  # cos(α - β) = cos α cos β + sin α sin β
    ddangle = np.arctan2(sin_dda, cos_dda)  # da
    mdangle = ave_dangle - abs(ddangle)     # indirect match, ma

    dmPP = dmx + dmI + dmG + dmM + dmL
    mmPP = mmx + mmI + mmG + mmM + mmL
    ddPP = ddx + ddI + ddG + ddM + ddL + ddangle
    mdPP = mdx + mdI + mdG + mdM + mdL + mdangle

    param_layer11 = {"x":(dmx,mmx,ddx,mdx), "I":(dmI,mmI,ddI,mdI), "G":(dmG,mmG,ddG,mdG), "M":(dmM,mmM,ddM,mdM), "L":(dmL,mmL,ddL,mdL),
                     "angle":(mdangle,ddangle), "DdyDdx":[sin_dda,cos_dda]}

    derPP = CderP(dP=dP, mP=mP, dmPP=dmPP, mmPP=mmPP, ddPP=ddPP, mdPP=mdPP, param_layer01=param_layer01, param_layer1=PP.param_layer1, param_layer11=param_layer11,
                  P=PP, _P=_PP)

    return derPP


def form_PPP_(derPP_, fiPd):

    PPP_ = []
    for derPP in deepcopy(derPP_):
        # last-row derPPs downconnect_cnt == 0
        if not derPP.P.downconnect_cnt and not isinstance(derPP.PP, CPP):  # root derPP was not terminated in prior call
            PPP = CPP()
            accum_PP(PPP,derPP)
            if derPP._P.upconnect_:
                upconnect_2_Pp_(derPP, PPP_, fiPd)  # form PPPs across _P upconnects
            else:
                PPP_.append(derPP.PP)  # terminate PPP
    return PPP_

# old:

def comp_slice_full(_P, P):  # forms vertical derivatives of derP params, and conditional ders from norm and DIV comp

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