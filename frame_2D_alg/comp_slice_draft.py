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

class Cdert_P(ClusterStructure):

    Pi = object  # P instance, accumulation: Cdert_P.Pi.I += 1, etc.
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
    sstack = object

class CPP(ClusterStructure):

    stack_ = list
    sstack_ = list
    sstacki = object
    # between PPs:
    upconnect_ = list
    in_upconnect_cnt = int # tentative upconnect count before getting the confirmed upconnect
    downconnect_cnt = int
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_?
    fdiv = NoneType


class CSstack(ClusterStructure):
    # an element of sstack in PP
    upconnect_ = list
    upconnect_cnt = int
    downconnect_cnt = int
    #+ other params of CStack?
    Py_ = list
    dert_Pi = object
    fdiv = NoneType

    ## PP
    PP = object
    PP_id = int
    # sstack params = accumulated dert_P params:
    # sPM, sPD, sMX, sDX, sML, sDL, sMDx, sDDx, sMDy, sDDy, sMDg, sDDg, sMMg, sDMg
    # add to blob:
    # PPm_ = list
    # PPd_ = list  # these are now primary blob-level structures


def comp_slice_(stack_, _P):
    '''
    cross-compare connected Ps of stack_, including Ps of adjacent stacks (upconnects)
    '''
    for stack in reversed(stack_):  # bottom-up

        if not stack.f_checked :  # else this stack has been scanned as some other upconnect
            stack.f_checked = 1
            DdX = 0  # accumulate across stacks?

            dert_Py_ = []
            if not _P:  # stack is from blob.stack_
                _P = stack.Py_.pop()
                dert_Py_.append(Cdert_P(Pi=_P))  # _P only, no derivatives in 1st dert_P

            for P in reversed(stack.Py_):
                dert_P = comp_slice(P, _P, DdX)  # ortho and other conditional operations are evaluated per PP
                dert_Py_.append(dert_P)  # dert_P is converted to Cdert_P in comp_slice
                _P = P

            stack.Py_ = dert_Py_
            if stack.upconnect_:
                comp_slice_(stack.upconnect_, _P)  # recursive compare _P to all upconnected Ps


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

    dert_P = Cdert_P(Pi=P, Pm=Pm, Pd=Pd, mX=mX, dX=dX, mL=mL, dL=dL, mDx=mDx, dDx=dDx, mDy=mDy, dDy=dDy, mDg=mDg, dDg=dDg, mMg=mMg, dMg=dMg)
    # div_f, nvars

    return dert_P


def comp_slice_old(blob, AveB):  # comp_slice eval per blob, simple stack_

        for stack in blob.stack_:
            if stack.G * stack.Ma - AveB / 10 > 0:  # / 10: ratio AveB to AveS, or not needed?
                # or default (L, Dy, Dx, G) min comp for rotation,
                # primary comp L, the rest is normalized?
                # overlap vs. shift:
                # init cross-dimension div_comp: Dx/Dy, but separate val for comp G, no default G/L?
                # comp -> min Dx/Dy for rotation, min G for comp_g?
                # also default min comp to upconnect_ Ps -> forking / merging PPs -> stack_ per PP!
                # stack.f_stackPP = 1  # scan Py_ -> comp_slice -> form_PP -> 2D PP_: clusters of same-sign dP | mP
                DdX = 0

                if stack.G * (stack.Ly / stack.A) * (abs(stack.Dy) / abs((stack.Dx) + 1)) > ave:  # G_bias * L_bias -> virt.rotation:
                    # or default min comp, eval per PP?
                    ortho = 1  # estimate params of P orthogonal to long axis at P' y and ave_x, to increase mP
                else:
                    ortho = 0
                dert_P_ = []
                _P = stack.Py_[0]

                for P in stack.Py_[1:]:
                    dert_P = comp_slice(ortho, P, _P, DdX)
                    dert_P_.append(dert_P)
                    _P = P

def stack_2_PP_(stack_, PP_):
    '''
    first stack_ call, then sign-unconfirmed upconnect_ calls
    '''
    for i, stack in enumerate(stack_):  # bottom-up to follow upconnects

        if stack.downconnect_cnt == 0:  # root stacks were not checked, upconnects always checked
            _dert_P = stack.Py_[0]
            sstack = CSstack(dert_Pi=_dert_P, Py_=[_dert_P])  # sstack: secondary stack, dert_P sigh-confirmed
            _dert_P.sstack = sstack
            PP = CPP(sstacki=sstack,sstack_=[sstack])
            sstack.PP = PP  # initialize upward reference

            for dert_P in stack.Py_[1:]:
                if (_dert_P.Pm > 0) != (dert_P.Pm > 0):
                    stack.sstack_.append(sstack)
                    stack.PP = PP
                    PP_.append(PP)  # terminate sstack and PP
                    sstack = CSstack(dert_Pi=Cdert_P()) # init sstack
                    PP = CPP(sstacki=sstack,sstack_=[sstack])  # init PP with the initialized sstack
                    sstack.PP = PP

                accum_sstack(sstack, dert_P)  # regardless of termination
                _dert_P = dert_P

            upconnect_2_PP_(stack.upconnect_, PP_, PP)  # form PPs across upconnects

    return PP_


def upconnect_2_PP_(stack_, PP_, iPP):  # terminate, initialize, increment blob PPs: clusters of same-sign mP dert_Ps

    if iPP.in_upconnect_cnt > 0:  # to track connects across function calls
        iPP.in_upconnect_cnt -= 1  # decreased by current upconnect
    upconnect_ = []
    isstack = iPP.sstack_[-1]
    _dert_P = isstack.Py_[-1]

    for i, stack in enumerate(stack_):  # breadth-first, upconnect_ is not reversed
        dert_P = stack.Py_[0]
        if (_dert_P.Pm > 0) == (dert_P.Pm > 0):  # upconnect has same sign
            upconnect_.append(stack_.pop(i))  # separate stack_ into sign-connected stacks: upconnect_, and unconnected stacks: popped stack_

    # iPP may still having non-terminated upconnects in other loops, we need add them instead of reassigning
    iPP.in_upconnect_cnt += len(upconnect_)

    if len(upconnect_) == 1:  # 1 same-sign upconnect per PP
        if not upconnect_[0].sstack_:  # sstack_ is empty until stack is scanned over
            accum_sstack(isstack, _dert_P)  # accumulate the input _dert_P
            PP = iPP  # no difference in single stack

            for dert_P in upconnect_[0].Py_:
                if (_dert_P.Pm > 0) != (dert_P.Pm > 0):
                    upconnect_[0].sstack_.append(isstack)
                    upconnect_[0].PP = PP
                    PP.in_upconnect_cnt -= 1
                    if PP.in_upconnect_cnt == 0:
                        PP_.append(PP)  # terminate sstack and PP
                    isstack = CSstack(dert_Pi=Cdert_P())  # init empty sstack, then accum_sstack
                    PP = CPP(sstacki=isstack, sstack_=[isstack])
                    isstack.PP = PP
                # else isstack is not terminated, no need to update connects
                accum_sstack(isstack, dert_P)  # regardless of termination
                _dert_P = dert_P

            upconnect_2_PP_(upconnect_[0].upconnect_, PP_, PP)
        else:
            merge_PP(iPP, upconnect_[0].sstack_[0].PP, PP_)  # merge connected PPs
            iPP.in_upconnect_cnt -= 1
            if iPP.in_upconnect_cnt <= 0:
                PP_.append(iPP)

    elif upconnect_:  # >1 same-sign upconnects per PP

        idert_P = _dert_P  # downconnected dert_P
        confirmed_upconnect_ = []  # same dert_P sign

        for upconnect in upconnect_:  # form PPs across stacks
            sstack = isstack  # downconnected sstack
            PP = iPP  # then redefined per stack
            _dert_P = idert_P
            ffirst = 1  # first dert_P in Py_

            if not upconnect.sstack_:
                for dert_P in upconnect.Py_:
                    if (_dert_P.Pm > 0) != (dert_P.Pm > 0):
                        # accum_PP(upconnect, sstack, PP)  # term. sstack
                        upconnect.sstack_.append(sstack)
                        upconnect.PP = PP
                        # separate iPP termination test
                        if PP is iPP:
                            PP.in_upconnect_cnt -= 1
                            if PP.in_upconnect_cnt == 0:
                                PP_.append(PP)
                        else:  # terminate stack-local PP
                            PP_.append(PP)
                        sstack = CSstack(dert_Pi=Cdert_P())  # init empty PP, regardless of iPP termination
                        PP = CPP(sstacki=sstack,sstack_=[sstack])  # we don't know if PP will fork at stack term
                        sstack.PP = PP

                    accum_sstack(sstack, dert_P)  # regardless of termination
                    if (PP is iPP) and ffirst and ((_dert_P.Pm > 0) == (dert_P.Pm > 0)):
                        confirmed_upconnect_.append(dert_P)  # to access dert_P.sstack
                    _dert_P = dert_P
                    ffirst = 0

                upconnect_2_PP_(upconnect.upconnect_, PP_, PP)

            else:
                merge_PP(iPP, upconnect.sstack_[0].PP, PP_)
                iPP.in_upconnect_cnt -= 1
                if iPP.in_upconnect_cnt <= 0:
                    PP_.append(iPP)

        # after all upconnects are checked:
        if confirmed_upconnect_:  # at least one first (_dert_P.Pm > 0) == (dert_P.Pm > 0) in upconnect_

            if len(confirmed_upconnect_) == 1:  # sstacks merge:
                dert_P = confirmed_upconnect_[0]
                merge_sstack(iPP.sstack_[-1], dert_P.sstack)
            else:
                for dert_P in confirmed_upconnect_:
                    # iPP is accumulated and isstack is downconnect of new sstack
                    iPP.sstack_[-1].upconnect_.append(dert_P.sstack)
                    dert_P.sstack.downconnect_cnt += 1

    else:
        if iPP.in_upconnect_cnt:
            iPP.in_upconnect_cnt -= 1
        if iPP.in_upconnect_cnt <= 0:  # 0 same-sign upconnects per PP:
            PP_.append(iPP)

    stack_2_PP_(stack_, PP_)  # stack_ now contains only stacks unconnected to isstack


def merge_PP(_PP, PP, PP_):  # merge PP into _PP
    _PP.sstack_.extend(PP.sstack_)
    _PP.sstacki.dert_Pi.accumulate(Pm=PP.sstacki.dert_Pi.Pm, Pd=PP.sstacki.dert_Pi.Pd, mx=PP.sstacki.dert_Pi.mx, dx=PP.sstacki.dert_Pi.dx,
                                     mL=PP.sstacki.dert_Pi.mL, dL=PP.sstacki.dert_Pi.dL, mDx=PP.sstacki.dert_Pi.mDx, dDx=PP.sstacki.dert_Pi.dDx,
                                     mDy=PP.sstacki.dert_Pi.mDy, dDy=PP.sstacki.dert_Pi.dDy, mDg=PP.sstacki.dert_Pi.mDg,
                                     dDg=PP.sstacki.dert_Pi.dDg, mMg=PP.sstacki.dert_Pi.mMg, dMg=PP.sstacki.dert_Pi.dMg)

    for sstack in PP.sstack_: # update PP reference
        sstack.PP = _PP

    if PP in PP_:
        PP_.remove(PP)  # remove the merged PP


def accum_sstack(sstack, dert_P):  # accumulate dert_P into sstack

    # accumulate dert_P params into sstack
    sstack.dert_Pi.accumulate(Pm=dert_P.Pm, Pd=dert_P.Pd, mx=dert_P.mx, dx=dert_P.dx,
                              mL=dert_P.mL, dL=dert_P.dL, mDx=dert_P.mDx, dDx=dert_P.dDx,
                              mDy=dert_P.mDy, dDy=dert_P.dDy, mDg=dert_P.mDg, dDg=dert_P.dDg,
                              mMg=dert_P.mMg, dMg=dert_P.dMg)

    sstack.Py_.append(dert_P)
    dert_P.sstack = sstack # update sstack reference in dert_P


def accum_PP(stack, sstack, PP):  # accumulate PP

    # accumulate sstack params into PP
    PP.sstacki.dert_Pi.accumulate(Pm=sstack.dert_Pi.Pm, Pd=sstack.dert_Pi.Pd, mx=sstack.dert_Pi.mx, dx=sstack.dert_Pi.dx,
                                  mL=sstack.dert_Pi.mL, dL=sstack.dert_Pi.dL, mDx=sstack.dert_Pi.mDx, dDx=sstack.dert_Pi.dDx,
                                  mDy=sstack.dert_Pi.mDy, dDy=sstack.dert_Pi.dDy, mDg=sstack.dert_Pi.mDg, dDg=sstack.dert_Pi.dDg,
                                  mMg=sstack.dert_Pi.mMg, dMg=sstack.dert_Pi.dMg)

    PP.sstack_.append(sstack)
    sstack.PP = PP
    sstack.PP_id = PP.id

    # add sstack to stack
    if stack:
        stack.sstack_.append(sstack)
        stack.PP = PP


def accum_gstack(gsstack, istack, sstack):   # accumulate istack and sstack into stack
    '''
    This looks wrong, accum_nested_stack should be an add-on to accum_sstack
    only called if istack.f_sstack:
    accum_sstack accumulates dert_P into sstack
    # accum_gstack accumulates sstack into gsstack?
    '''

    if istack.f_sstack:  # input stack is sstack
        # sstack params
        dert_Pi, mPP_, dPP_, dert_P_, fdiv = sstack.unpack()

        # need to accumulate dert_P params here, from sstack.dert_P params
        # accumulate sstack params
        gsstack.sstack.mPP_.extend(mPP_)
        gsstack.sstack.dPP_.extend(dPP_)
        gsstack.sstack.dert_P_.extend(dert_P_)
        gsstack.sstack.fdiv = fdiv

    # istack params
    I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, A, Ly, x0, xn, y0, Py_, sign, _, _, _, _, _, _, _  = istack.unpack()

    # accumulate istack param into stack_sstack
    gsstack.I += I
    gsstack.Dy += Dy
    gsstack.Dx += Dx
    gsstack.G += G
    gsstack.M += M
    gsstack.Dyy += Dyy
    gsstack.Dyx += Dyx
    gsstack.Dxy += Dxy
    gsstack.Dxx += Dxx
    gsstack.Ga += Ga
    gsstack.Ma += Ma
    gsstack.A += A
    gsstack.Ly += Ly
    if gsstack.x0 > x0: gsstack.x0 = x0
    if gsstack.xn < xn: gsstack.xn = xn
    if gsstack.y0 > y0: gsstack.y0 = y0
    gsstack.Py_.extend(Py_)
    gsstack.sign = sign  # sign should be same across istack

'''
    Pd and Pm are ds | ms per param summed in P. Primary comparison is by subtraction, div if par * rL compression:
    DL * DS > min: must be both, eval per dPP PD, signed? comp d?
    - resulting vertically adjacent dPPs and vPPs are evaluated for cross-comparison, to form PPPs and so on
    - resulting param derivatives form par_Ps, which are evaluated for der+ and rng+ cross-comparison
    | default top+ P level: if PD | PM: add par_Ps: sub_layer, rdn ele_Ps: deeper layer?
    aS compute if positive eV (not qD?) = mx + mL -ave? :
    aI = I / L; dI = aI - _aI; mI = min(aI, _aI)
    aD = D / L; dD = aD - _aD; mD = min(aD, _aD)
    aM = M / L; dM = aM - _aM; mM = min(aM, _aM)
    d_aS comp if cs D_aS, iter dS - S -> (n, M, diff): var precision or modulo + remainder?
    pP_ eval in +vPPs only, per rdn = alt_rdn * fork_rdn * norm_rdn, then cost of adjust for pP_rdn?
    eval_div(PP):
    if dL * Pm > div_ave:  # dL = potential compression by ratio vs diff, or decremental to Pd and incremental to Pm?
        rL  = L / _L  # DIV comp L, SUB comp (summed param * rL) -> scale-independent d, neg if cross-sign:
        nDx = Dx * rL; ndDx = nDx - _Dx; nmDx = min(nDx, _Dx)  # vs. nI = dI * rL or aI = I / L?
        nDy = Dy * rL; ndDy = nDy - _Dy; nmDy = min(nDy, _Dy)
        Pnm = mX + nmDx + nmDy  # defines norm_mPP, no ndx: single, but nmx is summed
        if Pm > Pnm: nmPP_rdn = 1; mPP_rdn = 0  # added to rdn, or diff alt, olp, div rdn?
        else: mPP_rdn = 1; nmPP_rdn = 0
        Pnd = ddX + ndDx + ndDy  # normalized d defines norm_dPP or ndPP
        if Pd > Pnd: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
        else: dPP_rdn = 1; ndPP_rdn = 0
        div_f = 1
        nvars = Pnm, nmDx, nmDy, mPP_rdn, nmPP_rdn,  Pnd, ndDx, ndDy, dPP_rdn, ndPP_rdn
    else:
        div_f = 0  # DIV comp flag
        nvars = 0  # DIV + norm derivatives
    '''

def term_PP2(typ, PP):  # eval for orient (as term_blob), incr_comp_slice, scan_par_:

    s, L2, I2, D2, Dy2, M2, My2, G2, Olp2, Py_, PM, PD, Mx, Dx, ML, DL, MI, DI, MD, DD, MDy, DDy, MM, DM, MMy, DMy, nVars = PP

    rdn = Olp2 / L2  # rdn per PP, alt Ps (if not alt PPs) are complete?

    # if G2 * Dx > ave * 9 * rdn and len(Py_) > 2:
    # PP, norm = orient(PP) # PP norm, rescan relative to parent blob, for incr_comp, comp_sliceP, and:

    if G2 + PM > ave * 99 * rdn and len(Py_) > 2:
       PP = incr_range_comp_slice(typ, PP)  # forming incrementally fuzzy PP

    if G2 + PD > ave * 99 * rdn and len(Py_) > 2:
       PP = incr_deriv_comp_slice(typ, PP)  # forming incrementally higher-derivation PP

    if G2 + PM > ave * 99 * rdn and len(Py_) > 2:  # PM includes results of incr_comp_slice
       PP = scan_params(0, PP)  # forming vpP_ and S_p_ders

    if G2 + PD > ave * 99 * rdn and len(Py_) > 2:  # PD includes results of incr_comp_slice
       PP = scan_params(1, PP)  # forming dpP_ and S_p_ders

    return PP

''' incr_comp() ~ recursive_comp() in line_POC(), with Ps instead of pixels?
    with rescan: recursion per p | d (signed): frame(meta_blob | blob | PP)? '''

def incr_range_comp_slice(typ, PP):
    return PP

def incr_deriv_comp_slice(typ, PP):
    return PP

def scan_params(typ, PP):  # at term_network, term_blob, or term_PP: + P_ders and nvars?
    '''
    Aves (integer filters) and coefs (ratio filters) per parameter type trigger formation of parameter_Ps,
    after full-blob comp_slice_ sums match and miss per parameter.
    Also coefs per sub_blob from comp_blob_: potential parts of a higher object?
    '''
    P_ = PP[11]
    Pars = [ (0,0,0,[]) ]

    for P in P_:  # repack ders into par_s by parameter type:

        s, ix, x, I, D, Dy, M, My, G, oG, Olp, t2_, Pm, Pd, mx, dx, mL, dL, mI, dI, mD, dD, mDy, dDy, mM, dM, mMy, dMy, div_f, nvars = P
        pars_ = [(x, mx, dx), (len(t2_), mL, dL), (I, mI, dI), (D, mD, dD), (Dy, mDy, dDy), (M, mM, dM), (My, mMy, dMy)]  # no nvars?

        for par, Par in zip(pars_, Pars): # PP Par (Ip, Mp, Dp, par_) += par (p, mp, dp):

            p, mp, dp = par
            Ip, Mp, Dp, par_ = Par

            Ip += p; Mp += mp; Dp += dp; par_.append((p, mp, dp))
            Par = Ip, Mp, Dp, par_  # how to replace Par in Pars_?

    for Par in Pars:  # select form_par_P -> Par_vP, Par_dP: combined vs. separate: shared access and overlap eval?
        Ip, Mp, Dp, par_ = Par

        if Mp + Dp > ave * 9 * 7 * 2 * 2:  # ave PP * ave par_P rdn * rdn to PP * par_P typ rdn?
            par_vPS, par_dPS = form_par_P(0, par_)
            par_Pf = 1  # flag
        else:
            par_Pf = 0; par_vPS = Ip, Mp, Dp, par_; par_dPS = Ip, Mp, Dp, par_

        Par = par_Pf, par_vPS, par_dPS
        # how to replace Par in Pars_?

    return PP

def form_par_P(typ, param_):  # forming parameter patterns within par_:

    p, mp, dp = param_.pop()  # initial parameter
    Ip = p, Mp = mp, Dp = dp, p_ = []  # Par init

    _vps = 1 if mp > ave * 7 > 0 else 0  # comp cost = ave * 7, or rep cost: n vars per par_P?
    _dps = 1 if dp > 0 else 0

    par_vP = Ip, Mp, Dp, p_  # also sign, typ and par olp: for eval per par_PS?
    par_dP = Ip, Mp, Dp, p_
    par_vPS = 0, 0, 0, []  # IpS, MpS, DpS, par_vP_
    par_dPS = 0, 0, 0, []  # IpS, MpS, DpS, par_dP_

    for par in param_:  # all vars are summed in incr_par_P
        p, mp, dp = par
        vps = 1 if mp > ave * 7 > 0 else 0
        dps = 1 if dp > 0 else 0

        if vps == _vps:
            Ip, Mp, Dp, par_ = par_vP
            Ip += p; Mp += mp; Dp += dp; par_.append(par)
            par_vP = Ip, Mp, Dp, par_
        else:
            par_vP = term_par_P(0, par_vP)
            IpS, MpS, DpS, par_vP_ = par_vPS
            IpS += Ip; MpS += Mp; DpS += Dp; par_vP_.append(par_vP)
            par_vPS = IpS, MpS, DpS, par_vP_
            par_vP = 0, 0, 0, []

        if dps == _dps:
            Ip, Mp, Dp, par_ = par_dP
            Ip += p; Mp += mp; Dp += dp; par_.append(par)
            par_dP = Ip, Mp, Dp, par_
        else:
            par_dP = term_par_P(1, par_dP)
            IpS, MpS, DpS, par_dP_ = par_dPS
            IpS += Ip; MpS += Mp; DpS += Dp; par_dP_.append(par_dP)
            par_vPS = IpS, MpS, DpS, par_dP_
            par_dP = 0, 0, 0, []

        _vps = vps; _dps = dps

    return par_vPS, par_dPS  # tuples: Ip, Mp, Dp, par_P_, added to Par

    # LIDV per dx, L, I, D, M? also alt2_: fork_ alt_ concat, for rdn per PP?
    # fpP fb to define vpPs: a_mx = 2; a_mw = 2; a_mI = 256; a_mD = 128; a_mM = 128

def term_par_P(typ, par_P):  # from form_par_P: eval for orient, re_comp? or folded?
    return par_P

def scan_par_P(typ, par_P_):  # from term_PP, folded in scan_par_? pP rdn per vertical overlap?
    return par_P_

def comp_slice_P(par_P, _par_P):  # with/out orient, from scan_pP_
    return par_P

def scan_PP_(PP_):  # within a blob, also within a segment?
    return PP_

def comp_sliceP(PP, _PP):  # compares PPs within a blob | segment, -> forking PPP_: very rare?
    return PP

'''
    horiz_dim_val = ave_Lx - |Dx| / 2  # input res and coord res are adjusted so mag approximates predictive value,
    vertical_dim_val  = Ly - |Dy| / 2  # or proj M = M - (|D| / M) / 2: no neg?
    core params G and M represent value of all others, no max Pm = L + |V| + |Dx| + |Dy|: redundant and coef-filtered?
    no * Ave_blob / Ga: angle match rate, already represented by hforks' position + mag' V+G -> comp( d | ortho_d)?
    eval per blob, too expensive for seg? no abs_Dx, abs_Dy for comp dert eval: mostly redundant?
    colors will be defined as color / sum-of-colors, color Ps are defined within sum_Ps: reflection object?
    relative colors may match across reflecting objects, forming color | lighting objects?
    comp between color patterns within an object: segmentation?
'''