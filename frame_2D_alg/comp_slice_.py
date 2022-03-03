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
ave_dx = 10  # difference between median x coords of consecutive Ps
ave_dI = 10
ave_dangle = 10
ave_dG = 10
ave_dM = 10
ave_dL = 10

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
    Ddy = int
    Ddx = int
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

class CderP(ClusterStructure):  # dert per CP param

    i = int
    p = int
    d = int
    m = int
    rdn = int
    P = object   # lower comparand
    _P = object  # higher comparand
    PP = object  # FPP if flip_val, contains this derP
    # from comp_dx
    fdx = NoneType
    distance = int  # d_ave_x

class CPp(CP, CderP):

    layer1 = dict
    # between PPs:
    upconnect_ = list
    downconnect_cnt = int
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    box = list   # for visualization only, original box before flipping
    dert__ = list
    mask__ = bool
    # PP params
    derP__ = list
    P__ = list
    PPmm_ = list
    PPdm_ = list
    # PPd params
    derPd__ = list
    Pd__ = list
    # comp_dx params
    PPmd_ = list
    PPdd_ = list
    # comp_PP
    derPPm_ = list
    derPPd_ = list
    distance = int
    mmPP = int
    dmPP = int
    mdPP = int
    ddPP = int
    PPPm = object
    PPPd = object
    neg_mmPP = int
    neg_mdPP = int

class CderPP(ClusterStructure):

    layer01 = dict
    layer1 = dict
    layer11 = dict
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

def comp_slice_root(blob, verbose=False):  # always angle blob, core params are g and ga, forming Pg and Pga?

    segment_by_direction(blob, verbose=False)
    P_ = slice_blob(blob, verbose=False)  # convert to batch processing, whole blob per function
    comp_slice_blob(blob, P_, verbose=False)  # calls scan_P_ and comp_slice
    # add higher comp orders

def slice_blob(blob, verbose=False):  # forms horizontal blob slices: Ps, ~1D Ps, in select smooth-edge (high G, low Ga) blobs

    mask__ = blob.mask__  # same as positive sign here
    dert__ = zip(*blob.dert__)  # convert tuple of arrays to array of tuples
    height, width = dert__[0].shape
    if verbose: print("Converting to image...")
    P__ = []  # blob Ps

    for y, (dert_, mask_) in enumerate( zip(dert__, mask__)):  # unpack lines
        P_ = []  # line Ps
        _mask = 1
        for x, (dert, mask) in  enumerate( zip(dert_, mask_)):
            if verbose:
                print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()
            # masks: if 0,_1: P initialization, if 0,_0: P accumulation, if 1,_0: P termination:
            if ~mask:
                if _mask:  # initialize P with first unmasked dert:
                    P = CP(I=dert[0], Dy=dert[1], Dx=dert[2], G=dert[3], Dydy=dert[4], Dxdy=dert[5], Dydx=dert[6], Dxdx=dert[7], Ga=dert[8],
                           x0=x, L=1, y=y, dert_=dert_)  # sign is always positive, else masked?
                else:
                    # dert and _dert are not masked, accumulate P params:
                    P.accumulate(I=dert[0], Dy=dert[1], Dx=dert[2], G=dert[3], Dydy=dert[5], Dxdy=dert[6], Dydx=dert[7], Dxdx=dert[8], Ga=dert[9], L=1)
                    P.dert_.append(dert)
            elif ~_mask:
                # _dert is not masked, dert is masked, terminate P:
                P.x = P.x0 + (P.L - 1) // 2  # x can't be fractional?
                P_.append(P)

            _mask = mask
            P__ += [P_]
    return P__

def scan_P_(P_, _P_):  # test for x overlap between Ps, call comp_slice
    '''
    derP_ = scan_P_(P_, _P_)  # tests for x overlap between Ps, calls comp_slice
    derP__ += [derP_] # frame of derPs
    '''
    derP_ = []
    for P in P_:  # lower row
        if P.sign:  # positive P only
            for _P in _P_:  # upper row
                # test for x overlap between P and _P in 8 directions
                if _P.sign and (P.x0 - 1 < (_P.x0 + _P.L) and (P.x0 + P.L) + 1 > _P.x0):  # all Ps here are positive

                    fcomp = [1 for derP in P.upconnect_ if P is derP.P]  # upconnect could be derP or dirP
                    if not fcomp:
                        derP = comp_slice(_P, P)  # form vertical and directional derivatives
                        derP_.append(derP)
                        P.upconnect_.append(derP)
                        _P.downconnect_cnt += 1

                elif not _P.sign or (P.x0 + P.L) < _P.x0:  # stop scanning the rest of lower P_ if there is no overlap
                    break
    return derP_


def form_PP_root(blob, derP__, P__, derPd__, Pd__, fPPd):
    '''
    form vertically contiguous patterns of patterns by the sign of derP, in blob or in FPP
    '''
    blob.derP__ = derP__; blob.P__ = P__
    blob.derPd__ = derPd__; blob.Pd__ = Pd__
    if fPPd:
        derP_2_PP_(blob.derP__, blob.PPdm_,  1)   # cluster by derPm dP sign
        derP_2_PP_(blob.derPd__, blob.PPdd_,  1)  # cluster by derPd dP sign, not used
    else:
        derP_2_PP_(blob.derP__, blob.PPmm_, 0)   # cluster by derPm mP sign
        derP_2_PP_(blob.derPd__, blob.PPmd_, 0)  # cluster by derPd mP sign, not used


def derP_2_PP_(derP_, PP_,  fPPd):
    '''
    first row of derP_ has downconnect_cnt == 0, higher rows may also have them
    '''
    for derP in reversed(derP_):  # bottom-up to follow upconnects, derP is stored top-down
        if not derP.P.downconnect_cnt and not isinstance(derP.PP, CPP):  # root derP was not terminated in prior call
            PP = CPP()  # init
            accum_PP(PP,derP)

            if derP._P.upconnect_:  # derP has upconnects
                upconnect_2_PP_(derP, PP_, fPPd)  # form PPs across _P upconnects
            else:
                PP_.append(derP.PP)


def upconnect_2_PP_(iderP, PP_,  fPPd):
    '''
    compare sign of lower-layer iderP to the sign of its upconnects to form contiguous same-sign PPs
    '''
    confirmed_upconnect_ = []

    for derP in iderP._P.upconnect_:  # potential upconnects from previous call
        if derP not in iderP.PP.derP__:  # this may occur after PP merging

            if fPPd: same_sign = (iderP.dP > 0) == (derP.dP > 0)  # comp dP sign
            else: same_sign = (iderP.mP > 0) == (derP.mP > 0)  # comp mP sign

            if same_sign:  # upconnect derP has different PP, merge them
                if isinstance(derP.PP, CPP) and (derP.PP is not iderP.PP):
                    merge_PP(iderP.PP, derP.PP, PP_)
                else:  # accumulate derP in current PP
                    accum_PP(iderP.PP, derP)
                    confirmed_upconnect_.append(derP)
            else:
                if not isinstance(derP.PP, CPP):  # sign changed, derP is root derP unless it already has FPP/PP
                    PP = CPP()
                    accum_PP(PP,derP)
                    derP.P.downconnect_cnt = 0  # reset downconnect count for root derP

                iderP.PP.upconnect_.append(derP.PP) # add new initialized PP as upconnect of current PP
                derP.PP.downconnect_cnt += 1        # add downconnect count to newly initialized PP

            if derP._P.upconnect_:
                upconnect_2_PP_(derP, PP_, fPPd)  # recursive compare sign of next-layer upconnects

            elif derP.PP is not iderP.PP and derP.P.downconnect_cnt == 0:
                PP_.append(derP.PP)  # terminate PP (not iPP) at the sign change

    iderP._P.upconnect_ = confirmed_upconnect_

    if not iderP.P.downconnect_cnt:
        PP_.append(iderP.PP)  # iPP is terminated after all upconnects are checked


def merge_PP(_PP, PP, PP_):  # merge PP into _PP

    for derP in PP.derP__:
        if derP not in _PP.derP__:
            _PP.derP__.append(derP) # add derP to PP
            derP.PP = _PP           # update reference
            _PP.accum_from(derP)    # accumulate params
    if PP in PP_:
        PP_.remove(PP)  # remove merged PP


def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})

def accum_PP(PP, derP):  # accumulate params in PP

    PP.accum_from(derP)    # accumulate params
    PP.accum_from(derP.P)  # accum derP's P base param to PP
    PP.derP__.append(derP) # add derP to PP
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


def comp_slice(_P, P):  # forms vertical derivatives of derP params, and conditional ders from norm and DIV comp

    derP_t = []  # 6 tuples of derP for I, Dy, Dx, G, M, L
    for param_name, ave in zip(param_names, aves):
        # retrieve param from param_name
        if param_name == "L" or param_name == "M":
            hyp = np.hypot(P.x, 1)  # ratio of local segment of long (vertical) axis to dY = 1
            _param = getattr(_P,param_name)
            param = getattr(P,param_name) / hyp # orthogonal L & M are reduced by hyp
        elif param_name == "Dy" or param_name == "Dx":
            _G = np.hypot(_P.Dy, _P.Dx) - (ave_G * _P.L)
            G = np.hypot(P.Dy, P.Dx) - (ave_G * P.L)
            _absG = max(1,_G + (ave_G*_P.L))
            absG = max(1,G + (ave_G*P.L))
            sin  = P.Dy/absG  ;  cos = P.Dx/absG
            _sin = _P.Dy/_absG; _cos = _P.Dx/_absG
            param = [sin, cos]
            _param = [_sin, _cos]
        else:
            param = getattr(P, param_name)
            _param = getattr(_P, param_name)

        # compute d and m
        if param_name == "I" or param_name == "L":
            d = param - _param   # difference
            m = ave - abs(d)     # indirect match
            i = param
            p = param+_param
        elif param_name == "Dy" or param_name == "Dx":
            sin, cos = param[0], param[1]
            _sin, _cos = _param[0], _param[1]
            # difference of dy and dx
            sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
            cos_da= (cos * _cos) + (sin * _sin)   # cos(α - β) = cos α cos β + sin α sin β
            d = np.arctan2(sin_da, cos_da)        # da
            m = ave - abs(d)                      # indirect match, ma
            if param_name == "Dy":
                i = sin_da  # Ddy
                p = (cos * _sin) + (sin * _cos)  # sin(α - β) = sin α cos β + cos α sin β
            elif param_name == "Dx":
                i = cos_da  # Ddx
                p = (cos * _cos) - (sin * _sin)  # cos(α + β) = cos α cos β - sin α sin β
        elif param_name == "G" or param_name == "M":
            d = param - _param                      # difference
            m = min(param,_param) - abs(d)/2 - ave  # direct match
            i = param
            p = param+_param

        derP = CderP(i=i, p=p, d=d, m=m, _P=_P, P=P)
        derP_t.append(derP)

    return derP_t

# obsolete
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

''' radial comp extension for co-internal blobs:
    != sign comp x sum( adj_blob_) -> intra_comp value, isolation value, cross-sign merge if weak, else:
    == sign comp x ind( adj_adj_blob_) -> same-sign merge | composition:
    borrow = adj_G * rA: default sum div_comp S -> relative area and distance to adjj_blob_
    internal sum comp if mA: in thin lines only? comp_norm_G or div_comp_G -> rG?
    isolation = decay + contrast:
    G - G * (rA * ave_rG: decay) - (rA * adj_G: contrast, = lend | borrow, no need to compare vG?)
    if isolation: cross adjj_blob composition eval,
    else:         cross adjj_blob merge eval:
    blob merger if internal match (~raG) - isolation, rdn external match:
    blob compos if external match (~rA?) + isolation,
    Also eval comp_slice over fork_?
    rng+ should preserve resolution: rng+_dert_ is dert layers,
    rng_sum-> rng+, der+: whole rng, rng_incr-> angle / past vs next g,
    rdn Rng | rng_ eval at rng term, Rng -= lost coord bits mag, always > discr?
    
    Add comp_PP_recursive
'''

# draft of comp_PP, following structure of comp_blob
def comp_PP_(blob, fPPd):

    for fPd in [0,1]:
        if fPPd: # cluster by d sign
            if fPd: # using derPd (PPdd)
                PP_ = blob.PPdd_
            else: # using derPm (PPdm)
                PP_ = blob.PPdm_
            for PP in PP_:
                if len(PP.derPPd_) == 0: # PP doesn't perform any searching in prior function call
                    comp_PP_recursive(PP, PP.upconnect_, derPP_=[], fPPd=fPPd)

            form_PPP_(PP_, fPPd)

        else: # cluster by m sign
            if fPd: # using derPd (PPmd)
                PP_ = blob.PPmd_
            else: # using derPm (PPmm)
                PP_ = blob.PPmm_
            for PP in PP_:
                if len(PP.derPPm_) == 0: # PP doesn't perform any searching in prior function call
                    comp_PP_recursive(PP, PP.upconnect_, derPP_=[], fPPd=fPPd)

            form_PPP_(PP_, fPPd)

def comp_PP_recursive(PP, upconnect_, derPP_, fPPd):

    derPP_pair_ = [ [derPP.PP, derPP._PP]  for derPP in derPP_]

    for _PP in upconnect_:
        if [_PP, PP] in derPP_pair_ : # derPP.PP = _PP, derPP._PP = PP
            derPP = derPP_[derPP_pair_.index([_PP,PP])]

        elif [PP, _PP] not in derPP_pair_ : # same pair of PP and _PP doesn't checked prior this function call
            derPP = comp_PP(PP, _PP) # comp_PP
            derPP_.append(derPP)

        if "derPP" in locals(): # derPP exists
            accum_derPP(PP, derPP, fPPd)    # accumulate derPP
            if fPPd:                  # PP cluster by d
                mPP = derPP.mdPP      # match of PPs' d
            else:                     # PP cluster by m
                mPP = derPP.mmPP      # match of PPs' m

            if mPP>0: # _PP replace PP to continue the searching
                comp_PP_recursive(_PP, _PP.upconnect_, derPP_, fPPd)

            elif fPPd and PP.neg_mdPP + PP.mdPP > ave_mPP: # evaluation to extend PPd comparison
                PP.distance += len(_PP.Pd__) # approximate using number of Py, not so sure
                PP.neg_mdPP += derPP.mdPP
                comp_PP_recursive(PP, _PP.upconnect_, derPP_, fPPd)

            elif not fPPd and PP.neg_mmPP + PP.mmPP > ave_mPP: # evaluation to extend PPm comparison
                PP.distance += len(_PP.P__) # approximate using number of Py, not so sure
                PP.neg_mmPP += derPP.mmPP
                comp_PP_recursive(PP, _PP.upconnect_, derPP_, fPPd)


'''   
def form_Pd_(P_):  # form Pds from Pm derts by dx sign, otherwise same as form_P
    Pd__ = []
    for iP in P_:
        if (iP.downconnect_cnt>0) or (iP.upconnect_):  # form Pd s if at least one connect in P, else they won't be compared
            P_Ddx = 0  # sum of Ddx across Pd s
            P_Mdx = 0  # sum of Mdx across Pd s
            Pd_ = []   # Pds in P
            _dert = iP.dert_[0]  # 1st dert
            dert_ = [_dert]
            _sign = _dert[2] > 0
            # initialize P with first dert
            P = CP(I=_dert[0], Dy=_dert[1], Dx=_dert[2], M=_dert[3],
               Dydy=_dert[4], Dxdy=_dert[5], Dydx=_dert[6], Dxdx=_dert[7], Ma=_dert[8],
                   x0=iP.x0, dert_=dert_, L=1, y=iP.y, sign=_sign, Pm=iP)
            x = 1  # relative x within P
            for dert in iP.dert_[1:]:
                sign = dert[2] > 0
                if sign == _sign: # same Dx sign
                    # accumulate P params with (p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma) = dert
                    P.accumulate(I=_dert[0], Dy=_dert[1], Dx=_dert[2], M=_dert[3],
                                 Dydy=_dert[4], Dxdy=_dert[5], Dydx=_dert[6], Dxdx=_dert[7], Ma=_dert[8], L=1)
                    P.dert_.append(dert)
                else:  # sign change, terminate P
                    if P.Dx > ave_Dx:
                        # cross-comp of dx in P.dert_
                        comp_dx(P); P_Ddx += P.Ddx; P_Mdx += P.Mdx
                    P.x = P.x0 + (P.L-1) // 2
                    Pd_.append(P)
                    # reinitialize params
                    P = CP(I=_dert[0], Dy=_dert[1], Dx=_dert[2], M=_dert[3],
                           Dydy=_dert[4], Dxdy=_dert[5], Dydx=_dert[6], Dxdx=_dert[7], Ma=_dert[8],
                           x0=iP.x0+x, dert_=[dert], L=1, y=iP.y, sign=sign, Pm=iP)
                _sign = sign
                x += 1
            # terminate last P
            if P.Dx > ave_Dx:
                comp_dx(P); P_Ddx += P.Ddx; P_Mdx += P.Mdx
            P.x = P.x0 + (P.L-1) // 2
            Pd_.append(P)
            # update Pd params in P
            iP.Pd_ = Pd_; iP.Ddx = P_Ddx; iP.Mdx = P_Mdx
            Pd__ += Pd_
    return Pd__
    
def scan_Pd_(P_, _P_):  # test for x overlap between Pds
    derPd_ = []
    for P in P_:  # lower row
        for _P in _P_:  # upper row
            for Pd in P.Pd_: # lower row Pds
                for _Pd in _P.Pd_: # upper row Pds
                    # test for same sign & x overlap between Pd and _Pd in 8 directions
                    if (Pd.x0 - 1 < (_Pd.x0 + _Pd.L) and (Pd.x0 + Pd.L) + 1 > _Pd.x0) and (Pd.sign == _Pd.sign):
                        fcomp = [1 for derPd in Pd.upconnect_ if Pd is derPd.P]  # upconnect could be derP or dirP
                        if not fcomp:
                            derPd = comp_slice(_Pd, Pd)
                            derPd_.append(derPd)
                            Pd.upconnect_.append(derPd)
                            _Pd.downconnect_cnt += 1
                    elif (Pd.x0 + Pd.L) < _Pd.x0:  # stop scanning the rest of lower P_ if there is no overlap
                        break
    return derPd_
'''
