'''
Comp_slice is a terminal fork of intra_blob.
It traces blob axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat or low-G blobs.
Vectorization is clustering of Ps + their derivatives into PPs: patterns of Ps that describe an edge.
'''

from collections import deque
import sys
import numpy as np
from class_cluster import ClusterStructure, NoneType
from slice_utils import draw_PP_

import warnings  # to detect overflow issue, in case of infinity loop
warnings.filterwarnings('error')

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback, not needed here
aveG = 50  # filter for comp_g, assumed constant direction
flip_ave = .1
flip_ave_FPP = 5  # flip large FPPs only
div_ave = 200
ave_dX = 10  # difference between median x coords of consecutive Ps
ave_Dx = 10
ave_mP = 20  # just a random number right now.

class CP(ClusterStructure):
    # Dert: summed pixel values and pixel-level derivatives:
    I = int
    Dy = int
    Dx = int
    G = int
    M = int
    Dyy = int
    Dyx = int
    Dxy = int
    Dxx = int
    Ga = int
    Ma = int
    L = int
    x0 = int
    y = int  # for visualization only
    sign = NoneType  # sign of gradient deviation
    dert_ = list   # array of pixel-level derts: (p, dy, dx, g, m), extended in intra_blob
    upconnect_ = list
    downconnect_cnt = int
    # only in in Pd:
    Pm = object  # reference to Pm in Pd s
    dxdert_ = list
    Mdx = int  # replaced dxP_Mdx
    Ddx = int  # replaced dxP_Ddx
    # only on Pm:
    Pd_ = list


class CderP(ClusterStructure):
    ## derP
    mP = int
    dP = int
    mx = int
    dx = int
    mL = int
    dL = int
    mDx = int
    dDx = int
    mDy = int
    dDy = int
    P = object  # lower comparand
    _P = object  # higher comparand
    PP = object  # FPP if flip_val, contains this derP
    flip_val = int


class CPP(ClusterStructure):

    derPP = object  # set of derP params accumulated in PP
    # between PPs:
    upconnect_ = list
    downconnect_cnt = int
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_?
    fdiv = NoneType
    box = list # for visualization only, original box before flipping
    # FPP params
    flip_val = int  # Ps are vertically biased
    dert__ = list
    mask__ = bool
    # PP params
    derP__ = list
    PP_ = list
    P__ = list
    # FPP params
    derPf__ = list
    PPf_ = list
    Pf__ = list
    # PPd params
    derPd__ = list
    PPd_ = list
    Pd__ = list
    # FPP params
    derPdf__ = list
    PPdf_ = list
    Pdf__ = list
    # comp_dx params
    Ddx = int
    Mdx = int

# Functions:
'''
leading '_' denotes higher-line variable or structure, vs. same-type lower-line variable or structure
trailing '_' denotes array name, vs. same-name elements of that array. '__' is a 2D array
leading 'f' denotes flag
-
workflow, needs an update:
-
intra_blob -> slice_blob(blob) -> derP_ -> PP,
if flip_val(PP is FPP): pack FPP in blob.PP_ -> flip FPP.dert__ -> slice_blob(FPP) -> pack PP in FPP.PP_
else       (PP is PP):  pack PP in blob.PP_
-
please see scan_P_ diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/comp_slice.drawio
'''

def slice_blob(blob, verbose=False):
    '''
    Slice_blob converts selected smooth-edge blobs (high G, low Ga) into sliced blobs,
    adding horizontal blob slices: Ps or 1D patterns
    '''
    if not isinstance(blob, CPP):  # input is blob, else FPP, no flipping
        flip_eval_blob(blob)

    dert__ = blob.dert__
    mask__ = blob.mask__
    height, width = dert__[0].shape
    if verbose: print("Converting to image...")
    P__ , derP__, Pd__, derPd__ = [], [], [], []

    zip_dert__ = zip(*dert__)
    _P_ = form_P_(list(zip(*next(zip_dert__))), mask__[0], 0)  # 1st upper row
    P__ += _P_  # frame of Ps

    for y, dert_ in enumerate(zip_dert__, start=1):  # scan top down
        if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()

        P_ = form_P_(list(zip(*dert_)), mask__[y], y)  # horizontal clustering - lower row
        derP_ = scan_P_(P_, _P_)  # tests for x overlap between Ps, calls comp_slice

        Pd_ = form_Pd_(P_)  # form Pds across Ps
        derPd_ = scan_Pd_(P_, _P_)  # adds upconnect_ in Pds and calls derPd_2_PP_derPd_, same as derP_2_PP_?

        derP__ += derP_ ; derPd__ += derPd_ # frame of derPs
        P__ += P_ ; Pd__ += Pd_
        _P_ = P_  # set current lower row P_ as next upper row _P_

    form_PP_shell(blob, derP__, P__, derPd__, Pd__)  # form PPs in blob or in FPP
    # draw PPs and FPPs
    if not isinstance(blob, CPP):
        draw_PP_(blob)


def form_PP_shell(blob, derP__, P__, derPd__, Pd__):
    '''
    form PPs in blob or in FPP
    '''
    if not isinstance(blob, CPP):  # input is blob
        blob.derP__ = derP__; blob.P__ = P__
        blob.derPd__ = derPd__; blob.Pd__ = Pd__
        derP_2_PP_(blob.derP__, blob.PP_,  blob.fflip)  # form vertically contiguous patterns of patterns
        derP_2_PP_(blob.derPd__, blob.PPd_, blob.fflip)
    else: # input is FPP
        blob.derPf__ = derP__; blob.Pf__ = P__
        blob.derPdf__ = derPd__; blob.Pdf__ = Pd__
        derP_2_PP_(blob.derPf__, blob.PPf_, 0)  # form vertically contiguous patterns of patterns
        derP_2_PP_(blob.derPdf__, blob.PPdf_, 0)


def form_P_(idert_, mask_, y):  # segment dert__ into P__, in horizontal ) vertical order
    '''
    sums dert params within Ps and increments L: horizontal length.
    '''
    P_ = []  # rows of derPs
    dert_ = [list(idert_[0])]  # get first dert from idert_ (generator/iterator)
    _mask = mask_[0]  # mask bit per dert
    if ~_mask:
        I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = dert_[0]; L = 1; x0 = 0  # initialize P params with first dert

    for x, dert in enumerate(idert_[1:], start=1):  # left to right in each row of derts
        mask = mask_[x]  # pixel mask

        if mask:  # masks: if 1,_0: P termination, if 0,_1: P initialization, if 0,_0: P accumulation:
            if ~_mask:  # _dert is not masked, dert is masked, terminate P:
                P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, dert_=dert_, y=y)
                P_.append(P)
        else:  # dert is not masked
            if _mask:  # _dert is masked, initialize P params:
                I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = dert; L = 1; x0 = x; dert_ = [dert]
            else:
                I += dert[0]  # _dert is not masked, accumulate P params with (p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma) = dert
                Dy += dert[1]
                Dx += dert[2]
                G += dert[3]
                M += dert[4]
                Dyy += dert[5]
                Dyx += dert[6]
                Dxy += dert[7]
                Dxx += dert[8]
                Ga += dert[9]
                Ma += dert[10]
                L += 1
                dert_.append(dert)
        _mask = mask

    if ~_mask:  # terminate last P in a row
        P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, dert_=dert_, y=y)
        P_.append(P)

    return P_


def scan_P_(P_, _P_):  # test for x overlap between Ps, call comp_slice

    derP_ = []
    for P in P_:  # lower row
        for _P in _P_:  # upper row
            # test for x overlap between P and _P in 8 directions
            if (P.x0 - 1 < (_P.x0 + _P.L) and (P.x0 + P.L) + 1 > _P.x0):  # all Ps here are positive

                fcomp = [1 for derP in P.upconnect_ if P is derP.P]  # upconnect could be derP or dirP
                if not fcomp:
                    derP = comp_slice(_P, P)  # form vertical and directional derivatives
                    derP_.append(derP)
                    P.upconnect_.append(derP)
                    _P.downconnect_cnt += 1

            elif (P.x0 + P.L) < _P.x0:  # stop scanning the rest of lower P_ if there is no overlap
                break

    return derP_


def form_Pd_(P_):
    '''
    form Pd s across P's derts using Dx sign
    '''
    Pd__ = []
    for iP in P_:
        if (iP.downconnect_cnt>0) or (iP.upconnect_):  # form Pd if at least one connect in P, else Pd s won't be compared
            P_Ddx = 0  # sum of Ddx across Pd s
            P_Mdx = 0  # sum of Mdx across Pd s
            Pd_ = []   # Pds in P
            _dert = iP.dert_[0]  # 1st dert
            dert_ = [_dert]
            I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = _dert; L = 1; x0 = iP.x0  # initialize P params with first dert
            _sign = _dert[2] > 0
            x = 1  # relative x within P

            for dert in iP.dert_[1:]:
                sign = dert[2] > 0
                if sign == _sign: # same Dx sign
                    I += dert[0]  # accumulate P params with (p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma) = dert
                    Dy += dert[1]
                    Dx += dert[2]
                    G += dert[3]
                    M += dert[4]
                    Dyy += dert[5]
                    Dyx += dert[6]
                    Dxy += dert[7]
                    Dxx += dert[8]
                    Ga += dert[9]
                    Ma += dert[10]
                    L += 1
                    dert_.append(dert)

                else:  # sign change, terminate P
                    P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, dert_=dert_, y=iP.y, sign=_sign, Pm=iP)
                    if Dx > ave_Dx:
                        comp_dx(P); P_Ddx += P.Ddx; P_Mdx += P.Mdx
                    Pd_.append(P)
                    # reinitialize param
                    I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = dert; x0 = iP.x0+x ;L = 1 ; dert_ = [dert]

                _sign = sign
                x += 1
            # terminate last P
            P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, dert_=dert_, y=iP.y, sign=_sign, Pm=iP)
            if Dx > ave_Dx:
                comp_dx(P); P_Ddx += P.Ddx; P_Mdx += P.Mdx
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
                    if (Pd.x0 - 1 < (_Pd.x0 + _Pd.L) and (Pd.x0 + Pd.L) + 1 > _Pd.x0) and (Pd.sign == _Pd.sign) : # all Ps here are positive

                        fcomp = [1 for derPd in Pd.upconnect_ if Pd is derPd.P]  # upconnect could be derP or dirP
                        if not fcomp:
                            derPd = comp_slice(_Pd, Pd) # the operations is the same between Pd and Pm in comp_slice?
                            derPd_.append(derPd)
                            Pd.upconnect_.append(derPd)
                            _Pd.downconnect_cnt += 1

                    elif (Pd.x0 + Pd.L) < _Pd.x0:  # stop scanning the rest of lower P_ if there is no overlap
                        break

    return derPd_


def comp_slice(_P, P):  # forms vertical derivatives of derP params, and conditional ders from norm and DIV comp

    s, x0, Dx, Dy, G, M, L = P.sign, P.x0, P.Dx, P.Dy, P.G, P.M, P.L
    # params per comp branch, add angle params, ext: X, new: L,
    # no input I comp in top dert?
    _s, _x0, _Dx, _Dy, _G, _M, _L = _P.sign, _P.x0, _P.Dx, _P.Dy, _P.G, _P.M, _P.L
    '''
    redefine Ps by dx in dert_, rescan dert by input P dX / d_ave_x: skip if not in blob?
    '''
    xn = x0 + L-1;  _xn = _x0 + _L-1
    _dX = (x0 + L-1 / 2) - (_x0 + _L-1 / 2)  # d_ave_x, alternatively:
    dX = abs(x0 - _x0) + abs(xn - _xn)  # diff of ave x, by offset, or max_L - overlap: abs distance?

    if dX > ave_dX:  # internal comp is higher-power, else two-input comp not compressive?
        mX = min(xn, _xn) - max(x0, _x0)  # overlap = abs proximity: summed binary positional match | miss:
        rX = dX / mX if mX else dX*2  # average dist / prox, | prox / dist, | mX / max_L?

    ddX = dX - _dX  # long axis curvature, if > ave: ortho eval per P, else per PP_dX?
    '''
    if ortho:  # estimate params of P locally orthogonal to long axis, maximizing lateral diff and vertical match

        Long axis is a curve, consisting of connections between mid-points of consecutive Ps.
        Ortho virtually rotates each P to make it orthogonal to its connection:
        hyp = hypot(dX, 1)  # long axis increment (vertical distance), to adjust params of orthogonal slice:
        L /= hyp
        # re-orient derivatives by combining them in proportion to their decomposition on new axes:
        Dx = (Dx * hyp + Dy / hyp) / 2  # no / hyp: kernel doesn't matter on P level?
        Dy = (Dy / hyp - Dx * hyp) / 2  # estimated D over vert_L

        param correlations: dX-> L, ddX-> dL, neutral to Dx: mixed with anti-correlated oDy?
    '''
    dL = L - _L; mL = min(L, _L)  # L: positions / sign, dderived: magnitude-proportional value
    dM = M - _M; mM = min(M, _M)  # no Mx, My: non-core, lesser and redundant bias?

    dDx = abs(Dx) - abs(_Dx); mDx = min(abs(Dx), abs(_Dx))  # same-sign Dx in vxP
    dDy = Dy - _Dy; mDy = min(Dy, _Dy)  # Dy per sub_P by intra_comp(dx), vs. less vertically specific dI

    # no comp G: Dx, dDy are more specific?
    dP = ddX + dL + dM + dDx + dDy  # -> directional PPd, equal-weight params, no rdn?
    # correlation: dX -> L, oDy, !oDx, ddX -> dL, odDy ! odDx? dL -> dDx, dDy?
    mP = mL + mM + mDx + mDy   # -> complementary PPm, rdn *= Pd | Pm rolp?

    mP -= ave_mP / 2^(dX / L)  # just a rough draft
    ''' Positional miss is positive: lower filters, no match: always inverse miss? '''

    flip_val = (dX * (P.Dy / (P.Dx+.001)) - flip_ave)  # avoid division by zero

    derP = CderP(P=P, _P=_P, flip_val=flip_val,
                 mP=mP, dP=dP, mX=mX, dX=dX, mL=mL, dL=dL, mDx=mDx, dDx=dDx, mDy=mDy, dDy=dDy)
    # div_f, nvars

    return derP


def derP_2_PP_(derP_, PP_, fflip):
    '''
    first row of derP_ has downconnect_cnt == 0, higher rows may also have them
    '''
    for derP in reversed(derP_):  # bottom-up to follow upconnects, derP is stored top-down
        if not derP.P.downconnect_cnt and not isinstance(derP.PP, CPP):  # root derP was not terminated in prior call
            PP = CPP(derPP=CderP())  # init
            accum_PP(PP,derP)

            if derP._P.upconnect_:  # derP has upconnects
                upconnect_2_PP_(derP, PP_, fflip)  # form PPs across _P upconnects
            else:
                if (derP.PP.derPP.flip_val > flip_ave_FPP) and fflip:
                    flip_FPP(derP.PP)
                PP_.append(derP.PP)


def upconnect_2_PP_(iderP, PP_, fflip):
    '''
    compare sign of lower-layer iderP to the sign of its upconnects to form contiguous same-sign PPs
    '''
    confirmed_upconnect_ = []

    for derP in iderP._P.upconnect_:  # potential upconnects from previous call
        if derP not in iderP.PP.derP__:  # derP should not in current iPP derP_ list, but this may occur after the PP merging

            if (derP.flip_val>0 and iderP.flip_val>0 and iderP.PP.derPP.flip_val>0):
                # upconnect derP has different FPP, merge them
                if isinstance(derP.PP, CPP) and (derP.PP is not iderP.PP):
                    merge_PP(iderP.PP, derP.PP, PP_)
                else: # accumulate derP to current FPP
                    accum_PP(iderP.PP, derP)
                    confirmed_upconnect_.append(derP)

            # same sign and not FPP
            elif ((iderP.mP > 0) == (derP.mP > 0)) and not (iderP.flip_val>0) and not (derP.flip_val>0):
                # upconnect derP has different PP
                if isinstance(derP.PP, CPP) and (derP.PP is not iderP.PP):
                    merge_PP(iderP.PP, derP.PP, PP_)
                else: # accumulate derP to current PP
                    accum_PP(iderP.PP, derP)
                    confirmed_upconnect_.append(derP)

            elif not isinstance(derP.PP, CPP):  # sign changed, derP is root derP unless it already has FPP/PP
                PP = CPP(derPP=CderP())
                accum_PP(PP,derP)
                derP.P.downconnect_cnt = 0  # reset downconnect count for root derP

            if derP._P.upconnect_:
                upconnect_2_PP_(derP, PP_, fflip)  # recursive compare sign of next-layer upconnects

            elif derP.PP is not iderP.PP and derP.P.downconnect_cnt == 0:
                if (derP.PP.derPP.flip_val > flip_ave_FPP) and fflip:
                    flip_FPP(derP.PP)
                PP_.append(derP.PP)  # terminate PP (not iPP) at the sign change

    iderP._P.upconnect_ = confirmed_upconnect_

    if not iderP.P.downconnect_cnt:
        if (iderP.PP.derPP.flip_val > flip_ave_FPP) and fflip:
            flip_FPP(iderP.PP)
        PP_.append(iderP.PP)  # iPP is terminated after all upconnects are checked


def flip_eval_blob(blob):

    # L_bias (Lx / Ly) * G_bias (Gy / Gx), blob.box = [y0,yn,x0,xn], ddirection: preferential comp over low G
    horizontal_bias = (blob.box[3] - blob.box[2]) / (blob.box[1] - blob.box[0])  \
                    * (abs(blob.Dy) / abs(blob.Dx))

    if horizontal_bias > 1 and (blob.G * blob.Ma * horizontal_bias > flip_ave / 10):
        blob.fflip = 1  # rotate 90 degrees for scanning in vertical direction
        # swap blob Dy and Dx:
        Dy=blob.Dy; blob.Dy = blob.Dx; blob.Dx = Dy
        # rotate dert__:
        blob.dert__ = tuple([np.rot90(dert) for dert in blob.dert__])
        blob.mask__ = np.rot90(blob.mask__)
        # swap dert dys and dxs:
        '''
        blob.dert__[1] swap doesn't affect blob.dert__[2] swap?
        '''
        blob.dert__ = list(blob.dert__)  # convert to list since param in tuple is immutable
        blob.dert__[1], blob.dert__[2] = \
        blob.dert__[2], blob.dert__[1]


def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})


def accum_PP(PP, derP):  # accumulate derP params in PP

    PP.derPP.accumulate(flip_val=derP.flip_val, mP=derP.mP, dP=derP.dP, mx=derP.mx, dx=derP.dx, mL=derP.mL, dL=derP.dL, mDx=derP.mDx, dDx=derP.dDx,
                        mDy=derP.mDy, dDy=derP.dDy)
    PP.derP__.append(derP)
    derP.PP = PP  # update reference


def merge_PP(_PP, PP, PP_):  # merge PP into _PP

    for derP in PP.derP__:
        if derP not in _PP.derP__:
            _PP.derP__.append(derP)
            derP.PP = _PP  # update reference
            # accumulate if PP' derP not in _PP
            _PP.derPP.accumulate(flip_val=derP.flip_val, mP=derP.mP, dP=derP.dP, mx=derP.mx, dx=derP.dx,
                                 mL=derP.mL, dL=derP.dL, mDx=derP.mDx, dDx=derP.dDx,
                                 mDy=derP.mDy, dDy=derP.dDy)
    if PP in PP_:
        PP_.remove(PP)  # remove merged PP


def flip_FPP(FPP):
    '''
    flip derts of FPP and call again slice_blob to get PPs of FPP
    '''
    # get box from P and P
    x0 = min(min([derP.P.x0 for derP in FPP.derP__]), min([derP._P.x0 for derP in FPP.derP__]))
    xn = max(max([derP.P.x0+derP.P.L for derP in FPP.derP__]), max([derP._P.x0+derP._P.L for derP in FPP.derP__]))
    y0 = min(min([derP.P.y for derP in FPP.derP__]), min([derP._P.y for derP in FPP.derP__]))
    yn = max(max([derP.P.y for derP in FPP.derP__]), max([derP._P.y for derP in FPP.derP__])) +1  # +1 because yn is not inclusive
    FPP.box = [y0,yn,x0,xn]
    # init empty derts, 11 params each: p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma
    dert__ = [np.zeros((yn-y0, xn-x0)) for _ in range(11)]
    mask__ = np.ones((yn-y0, xn-x0)).astype('bool')

    # fill empty dert with current FPP derts
    for derP in FPP.derP__:
        # _P
        for _x, _dert in enumerate(derP._P.dert_):
            for i, _param in enumerate(_dert):
                dert__[i][derP._P.y-y0, derP._P.x0-x0+_x] = _param
                mask__[derP._P.y-y0, derP._P.x0-x0+_x] = False
        # P
        for x, dert in enumerate(derP.P.dert_):
            for j, param in enumerate(dert):
                dert__[j][derP.P.y-y0, derP.P.x0-x0+x] = param
                mask__[derP.P.y-y0, derP.P.x0-x0+x] = False
    # flip dert__
    flipped_dert__ = [np.rot90(dert) for dert in dert__]
    flipped_mask__ = np.rot90(mask__)
    flipped_dert__[1],flipped_dert__[2] = \
    flipped_dert__[2],flipped_dert__[1]  # swap dy and dx in derts, always flipped in FPP
    FPP.dert__ = flipped_dert__
    FPP.mask__ = flipped_mask__
    # form PP_ in flipped FPP
    slice_blob(FPP, verbose=True)

def comp_dx(P):  # cross-comp of dx s in P.dert_

    Ddx = 0
    Mdx = 0
    dxdert_ = []
    _dx = P.dert_[0][2]  # first dx
    for dert in P.dert_[1:]:
        dx = dert[2]
        ddx = dx - _dx
        mdx = min(dx, _dx)
        dxdert_.append((ddx, mdx))  # no dx: already in dert_
        Ddx += ddx  # P-wide cross-sign, P.L is too short to form sub_Ps
        Mdx += mdx
        _dx = dx
    P.dxdert_ = dxdert_
    P.Ddx = Ddx
    P.Mdx = Mdx

'''
    radial comp extension for co-internal blobs:
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
'''