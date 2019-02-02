from collections import deque
import math as math
from time import time

# this component is a draft and mostly out of date

def comp_P(norm, P, _P):  # forms vertical derivatives of P vars, also conditional ders from DIV comp

    (s, x0, L, I, G, Dx, Dy, dert_), xd = P  # + optional A, sDa
    (_s, _x0, _L, _I, _G, _Dx, _Dy, _dert_), _xd = P
    xdd = 0  # optional, signs of xdd and dL correlate, signs of xd (position) and dL (dimension) don't?

    mx = (x0 + L-1) - _x0   # x olp, ave - xd -> vxP: low partial distance, or relative: olp_L / min_L (dderived)?
    if x0 > _x0: mx -= x0 - _x0   # vx only for !olp distant-P comp?

    # no if mx > ave:  # >min rolp, else Py_ termination? by full comp only?
    if norm:  # if xD / Ly * (Dx + Dy) > ave: derivatives are xd-normalized before comp:

        hyp = math.hypot(xd, 1)  # Ly increment = hyp / 1 (vert distance)
        L /= hyp; I /= hyp; Dx /= hyp; Dy /= hyp  # est orthogonal slice is reduced from P

        Dx = (Dx * hyp + Dy / hyp) / 2 / hyp  # est D over ver_L, Ders summed in ver / lat ratio
        Dy = (Dy / hyp - Dx * hyp) / 2 * hyp  # est D over lat_L

    dL = L - _L; mL = min(L, _L)  # ext miss: Ddx + DL? L, I, G, A: int, I, dif, no S = I + G + A: too few?
    dI = I - _I; mI = min(I, _I)  # L and I are not xd-normalized

    G = math.hypot(Dy, Dx)  # re-formed vs. accumulated: lost cross-sign Ds, and normalized?
    dG = G - _G; mG = min(G, _G)  # primary to Ds?

    dDx = Dx - _Dx; mDx = min(Dx, _Dx)  # conditional or primary if norm?
    dDy = Dy - _Dy; mDy = min(Dy, _Dy)  # lat sum of y_ders also indicates P match and orientation?

    Pd = xdd + dL + dI + dG + dDx + dDy  # defines dPP, no dS-to-xd correlation?
    Pm = mx +  mL + mI + mG + mDx + mDy  # defines mPP; comb rep value = Pm * 2 + Pd, for intra_blob?

    if dI * dL > div_ave:  # L defines P, I indicates potential ratio vs diff compression?

        rL = L / _L  # DIV comp L, SUB comp (summed param * rL) -> scale-independent d, neg if cross-sign:
        nI = I * rL; ndI = nI - _I; nmI = min(nI, _I)  # vs. nI = dI * nrL?

        nDx = Dx * rL; ndDx = nDx - _Dx; nmDx = min(nDx, _Dx)
        nDy = Dy * rL; ndDy = nDy - _Dy; nmDy = min(nDy, _Dy)

        Pnm = mx + nmI + nmDx + nmDy  # defines norm_mPP, no ndx: single, but nmx is summed

        if Pm > Pnm: nmPP_rdn = 1; mPP_rdn = 0  # added to rdn, or diff alt, olp, div rdn?
        else: mPP_rdn = 1; nmPP_rdn = 0

        Pnd = xdd + ndI + ndDx + ndDy  # normalized d defines norm_dPP or ndPP

        if Pd > Pnd: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
        else: dPP_rdn = 1; ndPP_rdn = 0

        div_f = 1
        nvars = Pnm, nmI, nmDx, nmDy, mPP_rdn, nmPP_rdn, \
            Pnd, ndI, ndDx, ndDy, dPP_rdn, ndPP_rdn

    else:
        div_f = 0  # DIV comp flag
        nvars = 0  # DIV + norm derivatives

    P_ders = Pm, Pd, mx, xd, mL, dL, mI, dI, mDx, dDx, mDy, dDy, div_f, nvars

    vs = 1 if Pm > ave * 7 > 0 else 0  # comp cost = ave * 7, or rep cost: n vars per P?
    ds = 1 if Pd > 0 else 0

    return (P, P_ders), vs, ds


''' no comp_q_(q_, _q_, yP_): vert comp by ycomp, ortho P by orientation?
    comp_P is not fuzzy: x, y vars are already fuzzy?
    
    no DIV comp(L): match is insignificant and redundant to mS, mLPs and dLPs only?:

    if dL: nL = len(q_) // len(_q_)  # L match = min L mult
    else: nL = len(_q_) // len(q_)
    fL = len(q_) % len(_q_)  # miss = remainder 

    no comp aS: m_aS * rL cost, minor cpr / nL? no DIV S: weak nS = S // _S; fS = rS - nS  
    or aS if positive eV (not qD?) = mx + mL -ave:

    aI = I / L; dI = aI - _aI; mI = min(aI, _aI)  
    aD = D / L; dD = aD - _aD; mD = min(aD, _aD)  
    aM = M / L; dM = aM - _aM; mM = min(aM, _aM)

    d_aS comp if cs D_aS, iter dS - S -> (n, M, diff): var precision or modulo + remainder? 
    pP_ eval in +vPPs only, per rdn = alt_rdn * fork_rdn * norm_rdn, then cost of adjust for pP_rdn? '''


def form_PP(typ, P, PP):  # increments continued vPPs or dPPs (not pPs): incr_blob + P_ders?

    P, P_ders, S_ders = P
    s, ix, x, I, D, Dy, M, My, G, oG, Olp, t2_ = P
    L2, I2, D2, Dy2, M2, My2, G2, OG, Olp2, Py_ = PP

    L2 += len(t2_)
    I2 += I
    D2 += D; Dy2 += Dy
    M2 += M; My2 += My
    G2 += G
    OG += oG
    Olp2 += Olp

    Pm, Pd, mx, dx, mL, dL, mI, dI, mD, dD, mDy, dDy, mM, dM, mMy, dMy, div_f, nvars = P_ders
    _dx, Ddx, \
    PM, PD, Mx, Dx, ML, DL, MI, DI, MD, DD, MDy, DDy, MM, DM, MMy, DMy, div_f, nVars = S_ders

    Py_.appendleft((s, ix, x, I, D, Dy, M, My, G, oG, Olp, t2_, Pm, Pd, mx, dx, mL, dL, mI, dI, mD, dD, mDy, dDy, mM, dM, mMy, dMy, div_f, nvars))

    ddx = dx - _dx  # no ddxP_ or mdx: olp of dxPs?
    Ddx += abs(ddx)  # PP value of P norm | orient per indiv dx: m (ddx, dL, dS)?

    # summed per PP, then per blob, for form_pP_ or orient eval?

    PM += Pm; PD += Pd  # replace by zip (S_ders, P_ders)
    Mx += mx; Dx += dx; ML += mL; DL += dL; ML += mI; DL += dI
    MD += mD; DD += dD; MDy += mDy; DDy += dDy; MM += mM; DM += dM; MMy += mMy; DMy += dMy

    return s, L2, I2, D2, Dy2, M2, My2, G2, Olp2, Py_, PM, PD, Mx, Dx, ML, DL, MI, DI, MD, DD, MDy, DDy, MM, DM, MMy, DMy, nVars


def term_PP(typ, PP):  # eval for orient (as term_blob), incr_comp_P, scan_par_:

    s, L2, I2, D2, Dy2, M2, My2, G2, Olp2, Py_, PM, PD, Mx, Dx, ML, DL, MI, DI, MD, DD, MDy, DDy, MM, DM, MMy, DMy, nVars = PP

    rdn = Olp2 / L2  # rdn per PP, alt Ps (if not alt PPs) are complete?

    # if G2 * Dx > ave * 9 * rdn and len(Py_) > 2:
    # PP, norm = orient(PP) # PP norm, rescan relative to parent blob, for incr_comp, comp_PP, and:

    if G2 + PM > ave * 99 * rdn and len(Py_) > 2:
       PP = incr_range_comp_P(typ, PP)  # forming incrementally fuzzy PP

    if G2 + PD > ave * 99 * rdn and len(Py_) > 2:
       PP = incr_deriv_comp_P(typ, PP)  # forming incrementally higher-derivation PP

    if G2 + PM > ave * 99 * rdn and len(Py_) > 2:  # PM includes results of incr_comp_P
       PP = scan_params(0, PP)  # forming vpP_ and S_p_ders

    if G2 + PD > ave * 99 * rdn and len(Py_) > 2:  # PD includes results of incr_comp_P
       PP = scan_params(1, PP)  # forming dpP_ and S_p_ders

    return PP

''' incr_comp() ~ recursive_comp() in line_POC(), with Ps instead of pixels?
    with rescan: recursion per p | d (signed): frame(meta_blob | blob | PP)? '''

def incr_range_comp_P(typ, PP):
    return PP

def incr_deriv_comp_P(typ, PP):
    return PP

def scan_params(typ, PP):  # at term_network, term_blob, or term_PP: + P_ders and nvars?

    P_ = PP[11]
    Pars = [(0,0,0,[]), (0,0,0,[]), (0,0,0,[]), (0,0,0,[]), (0,0,0,[]), (0,0,0,[]), (0,0,0),[]]

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

def comp_par_P(par_P, _par_P):  # with/out orient, from scan_pP_
    return par_P

def scan_PP_(PP_):  # within a blob, also within a segment?
    return PP_

def comp_PP(PP, _PP):  # compares PPs within a blob | segment, -> forking PPP_: very rare?
    return PP

def flip(blob):  # vertical-first run of form_P and deeper functions over blob's ders__
    return blob


def comp_Py_(val_PP_, norm, blob, xD, rdn):  # scan of vertical Py_ -> comp_P -> 2D mPPs and dPPs
    s, [min_x, max_x, min_y, max_y, xD, abs_xD, Ly], [L, I, G, Dx, Dy, abs_Dx, abs_Dy], root_ = blob

    if val_PP_ * ((max_x - min_x + 1) / (max_y - min_y + 1)) * (max(abs_Dx, abs_Dy) / min(abs_Dx, abs_Dy)) > flip_ave:
        # | (max(Dx, Dy) / min(Dx, Dy): cumulative?
        flip(blob)  # vertical blob rescan -> comp_Px_

    # flip if PM gain projected by D-bias <-> L-bias: width / height, vs abs(xD) / height for oriented blobs?
    # or flip_eval(positive xd_dev_P (>>90)), after scan_Py_-> xd_dev_P?

    mPP = 0,[],[]  # pattern of patterns, defined by deviation of M_params, dderived: match is a minimum, per P | param?
    dPP = 0,[],[]  # sub PP within negative mPP: min value of signed dS
    mPP_, dPP_, CmPP_, CdPP_, Cm_, Cd_ = [],[],[],[],[],[]  # comparable params and their derivatives

    Py_ = blob[3][last]  # per segment, also flip eval per seg?
    _P = Py_.popleft()  # initial comparand

    while Py_:  # comp_P starts from 2nd P, top-down
        P = Py_.popleft()
        _P, _ms, _ds = comp_P(norm, P, _P)

        while Py_:  # form_PP starts from 3rd P
            P = Py_.popleft()
            P, ms, ds = comp_P(norm, P, _P)  # P: S_vars += S_ders in comp_P
            if ms == _ms:
                mPP = form_PP(1, P, mPP)
            else:
                mPP = term_PP(1, mPP)  # SPP += S, PP eval for orient, incr_comp_P, scan_par..?
                mPP_.append(mPP)
                for par, C in zip(mPP[1], CmPP_):  # blob-wide summation of 16 S_vars from incr_PP
                    C += par
                    Cm_.append(C)  # or S is directly modified in SvPP?
                CmPP_ = Cm_  # but SPP is redundant, if len(PP_) > ave?
                mPP = ms, [], []  # s, PP, Py_ init
            if ds == _ds:
                dPP = form_PP(0, P, dPP)
            else:
                dPP = term_PP(0, dPP)
                dPP_.append(dPP)
                for var, C in zip(dPP[1], CdPP_):
                    C += var
                    Cd_.append(C)
                CdPP_ = Cd_
                dPP = ds,[],[]
            _P = P; _ms = ms; _ds = ds

    return blob, CmPP_, mPP_, CdPP_, dPP_  # blob | PP_? comp_P over fork_, after comp_segment?


flip_ave = 1000