'''
    comp_slice_ is a terminal fork of intra_blob.
    It traces blob axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
    These high-G high-Ma blobs are vectorized into outlines of adjacent flat or low-G blobs.
    Vectorization is clustering of Ps + derivatives into PPs: patterns of Ps that describe an edge.
    Double edge lines: assumed match between edges of high-deviation intensity, no need for cross-comp?
    secondary cross-comp of low-deviation blobs?   P comb -> intra | inter comp eval?
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

from time import time
from collections import deque
from class_cluster import ClusterStructure, NoneType
from math import hypot
import numpy as np
import warnings  # to detect overflow issue, in case of infinity loop
warnings.filterwarnings('error')

ave = 20
div_ave = 200
flip_ave = 1000
ave_dX = 10  # difference between median x coords of consecutive Ps

class CderP(ClusterStructure):

    Pi = object  # P instance, accumulation: CderP.Pi.I += 1, etc.
    Pm = int
    Pd = int
    mx = int
    dx = int
    mL = int
    dL = int
    mDx = int
    dDx = int
    mDy = int
    dDy = int
    mDg = int
    dDg = int
    mMg = int
    dMg = int
    upconnect_ = list
    downconnect_cnt = int
    # stack and PP object reference
    stack = object
    PP = object

class CPP(ClusterStructure):

    derP_ = list
    derPi = object
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_?
    fdiv = NoneType
    # between PPs:
    upconnect_ = list
    downconnect_cnt = int
    upconnect_cnt = int


def comp_slice_(stack_, _P):
    '''
    cross-compare connected Ps of stack_, including Ps of adjacent stacks (upconnects)
    '''
    derP_ = []
    DdX = 0
    for stack in reversed(stack_):  # bottom-up

        if not stack.f_checked :  # else this stack has been scanned as some other upconnect
            stack.f_checked = 1
            downconnect_cnt = stack.downconnect_cnt

            if not _P:  # stack is from blob.stack_
                _P = stack.Py_.pop()
                derP_.append(CderP(Pi=_P, downconnect_cnt = downconnect_cnt, stack = stack))
                # no derivatives, assign stack.downconnect_cnt to derP.downconnect_cnt

            for P in reversed(stack.Py_):
                derP = comp_slice(P, _P, DdX)  # ortho and other conditional operations are evaluated per PP
                derP.downconnect_cnt = downconnect_cnt
                if derP_:
                    derP_[-1].upconnect_.append(derP)  # next
                derP_.append(derP)  # derP is converted to CderP in comp_slice
                _P = P
                downconnect_cnt = 1  # always 1 inside Py_

            if stack.upconnect_:
                derP_ += comp_slice_(stack.upconnect_, _P)  # recursive compare _P to all upconnected Ps

    return derP_

'''
Special value of branching points: at least one angle miss, or that's corner?
'''

def comp_slice(P, _P, DdX):  # forms vertical derivatives of P params, and conditional ders from norm and DIV comp

    s, x0, G, M, Dx, Dy, L, Dg, Mg = P.sign, P.x0, P.G, P.M, P.Dx, P.Dy, P.L, P.Dg, P.Mg
    # params per comp branch, add angle params, ext: X, new: L,
    # no input I comp in top dert?
    _s, _x0, _G, _M, _Dx, _Dy, _L, _Dg, _Mg = _P.sign, _P.x0, _P.G, _P.M, _P.Dx, _P.Dy, _P.L, _P.Dg, _P.Mg
    '''
    redefine Ps by dx in dert_, rescan dert by input P d_ave_x: skip if not in blob?
    '''
    xn = x0 + L-1;  _xn = _x0 + _L-1
    mX = min(xn, _xn) - max(x0, _x0)  # overlap: abs proximity, cumulative binary positional match | miss:
    _dX = (xn - L/2) - (_xn - _L/2)
    dX = abs(x0 - _x0) + abs(xn - _xn)  # offset, or max_L - overlap: abs distance?

    if dX > ave_dX:  # internal comp is higher-power, else two-input comp not compressive?
        rX = dX / (mX+.001)  # average dist/prox, | prox/dist, | mX / max_L?

    ave_dx = (x0 + (L-1)//2) - (_x0 + (_L-1)//2)  # d_ave_x, median vs. summed, or for distant-P comp only?

    ddX = dX - _dX  # long axis curvature
    DdX += ddX  # if > ave: ortho eval per P, else per PP_dX?
    # param correlations: dX-> L, ddX-> dL, neutral to Dx: mixed with anti-correlated oDy?
    '''
    if ortho:  # estimate params of P locally orthogonal to long axis, maximizing lateral diff and vertical match
        Long axis is a curve, consisting of connections between mid-points of consecutive Ps.
        Ortho virtually rotates each P to make it orthogonal to its connection:
        hyp = hypot(dX, 1)  # long axis increment (vertical distance), to adjust params of orthogonal slice:
        L /= hyp
        # re-orient derivatives by combining them in proportion to their decomposition on new axes:
        Dx = (Dx * hyp + Dy / hyp) / 2  # no / hyp: kernel doesn't matter on P level?
        Dy = (Dy / hyp - Dx * hyp) / 2  # estimated D over vert_L
    '''
    dL = L - _L; mL = min(L, _L)  # L: positions / sign, dderived: magnitude-proportional value
    dM = M - _M; mM = min(M, _M)  # no Mx, My: non-core, lesser and redundant bias?

    dDx = abs(Dx) - abs(_Dx); mDx = min(abs(Dx), abs(_Dx))  # same-sign Dx in vxP
    dDy = Dy - _Dy; mDy = min(Dy, _Dy)  # Dy per sub_P by intra_comp(dx), vs. less vertically specific dI

    # gdert param comparison, if not fPP, values would be 0
    dMg = Mg - _Mg; mMg = min(Mg, _Mg)
    dDg = Dg - _Dg; mDg = min(Dg, _Dg)

    Pd = ddX + dL + dM + dDx + dDy + dMg + dDg # -> directional dPP, equal-weight params, no rdn?
    # correlation: dX -> L, oDy, !oDx, ddX -> dL, odDy ! odDx? dL -> dDx, dDy?  G = hypot(Dy, Dx) for 2D structures comp?
    Pm = mX + mL + mM + mDx + mDy + mMg + mDg # -> complementary vPP, rdn *= Pd | Pm rolp?

    derP = CderP(Pi=P, Pm=Pm, Pd=Pd, mX=mX, dX=dX, mL=mL, dL=dL, mDx=mDx, dDx=dDx, mDy=mDy, dDy=dDy, mDg=mDg, dDg=dDg, mMg=mMg, dMg=dMg)
    # div_f, nvars

    return derP


def accum_PP(PP, derP):

    PP.derPi.accumulate(Pm=derP.Pm, Pd=derP.Pd, mx=derP.mx, dx=derP.dx, mL=derP.mL, dL=derP.dL, mDx=derP.mDx, dDx=derP.dDx,
                          mDy=derP.mDy, dDy=derP.dDy, mDg=derP.mDg, dDg=derP.dDg, mMg=derP.mMg, dMg=derP.dMg)
    PP.derP_.append(derP)

def merge_PP(_PP, PP, PP_):  # merge PP into _PP

    _PP.derP_.extend(PP.derP_)
    _PP.dert_Pi.accumulate(Pm=PP.dert_Pi.Pm, Pd=PP.dert_Pi.Pd, mx=PP.dert_Pi.mx, dx=PP.dert_Pi.dx,
                           mL=PP.dert_Pi.mL, dL=PP.dert_Pi.dL, mDx=PP.dert_Pi.mDx, dDx=PP.dert_Pi.dDx,
                           mDy=PP.dert_Pi.mDy, dDy=PP.dert_Pi.dDy, mDg=PP.dert_Pi.mDg,
                           dDg=PP.dert_Pi.dDg, mMg=PP.dert_Pi.mMg, dMg=PP.dert_Pi.dMg)

    for derP in PP.derP_: # update PP reference
        derP.PP = _PP

    if PP in PP_:
        PP_.remove(PP)  # remove the merged PP


def derP_2_PP_(derP_, PP_):
    '''
    first row of derP_ has downconnect_cnt == 0, higher rows may also have them
    '''
    for derP in derP_:  # bottom-up to follow upconnects

        if derP.downconnect_cnt == 0:  # root derP
            PP = CPP(derPi=derP, derP_= [derP])  # init
            derP.PP = PP
            upconnect_2_PP_(derP, PP_)  # form PPs across dertP upconnects

    return PP_


def upconnect_2_PP_(iderP, PP_):
    '''
    check derPs' upconnects to form continuous same-sign PPs
    '''
    # if _derP.upconnect_cnt: _derP.upconnect_cnt -= 1  # remaining upconnects
    # this is done by: PP.upconnect_cnt += len(confirmed_upconnect_)
    # or not needed at all: each prior upconnect_ is followed until termination?

    confirmed_upconnect_ = []
    for derP in iderP.upconnect_:  # upconnects from previous call

        if (iderP.Pm > 0) == (derP.Pm > 0):
            if derP.PP and (derP.PP is not iderP.PP):
                # if derP.PP may be sufficient: checked derP.PP must be different from iderP.PP?
                merge_PP(iderP.PP, derP.PP, PP_)
            else:
                derP.PP = iderP.PP
            confirmed_upconnect_.append(derP)
            accum_PP(iderP.PP, derP)
        else:
            # unconfirmed: derP is root derP
            derP.downconnect_cnt = 0
            derP.PP = CPP(derPi=derP, derP_=[derP])  # init

        if derP.upconnect_:  # need to make it conditional, else an infinite loop?
            upconnect_2_PP_(derP, PP_)  # form PPs across dertP upconnects

    iderP.upconnect_ = confirmed_upconnect_
    iderP.PP.upconnect_cnt += len(confirmed_upconnect_)
    # PP.upconnect_cnt should not be needed: each prior upconnect_ is followed until termination,
    # so initial upconnect_cnt per upconnect is always 0?

    if iderP.PP.upconnect_cnt == 0:  # if above, replace with if not confirmed_upconnect_:
        PP_.append(iderP.PP)  # PP termination;  upconnects are called from derP, not PP

'''
    if len(PP.upconnect_) > 0:
        _derP = PP.derP_[-1]

        for i, derP in enumerate(upconnect_):  # bottom-up to follow upconnects
            if not isinstance(derP.PP, CPP):  # checked derP should have PP object instance
                if (_derP.Pm > 0) != (derP.Pm > 0):
                    PP = CPP(derPi=CderP(), derP_= [derP])
                    derP.PP = PP

                accum_PP(PP, derP)  # regardless of termination
                derP_2_PP_(derP.upconnect_, PP, PP_ )  # form PPs across upconnects

            elif derP.PP is not PP: # if there is PP in derP and they are not the same PP, merge them
                merge_PP(PP, derP.PP, PP_)

                PP.upconnect_cnt -= 1 # reduce upconnect count and check termination, because this section will not call the recursvie function to further check on upconnect
                if PP.upconnect_cnt == 0: PP_.append(PP)

    # terminate PP when upconnect = 0, should be at the end of the upconnect
    elif PP.upconnect_cnt == 0: PP_.append(PP)

    # form PP again with the non upconnect's derPs
    derP_2_PP_(derP_, PP_)
'''