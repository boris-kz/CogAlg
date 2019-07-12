from collections import deque
from math import hypot
from time import time
'''
    comp_P_ is a fork of intra_blob, currently a draft
    it will cross-compare vertically adjacent Ps (representing horizontally contiguous slices across a blob)
    and form dPPs and vPPs: vertically contiguous sets of Ps with same-sign vertical difference or match deviation of P params
    (difference | match deviation per param is summed between all compared params in P)
    
    par_coefs per level: match|dev rate fb, also per pattern, for form par_P, after full-blob comp_P_ 
    sub_coefs per sub_blob, if recurrent in super_blob from comp_blob 
 
    comp_P is potentially micro and macro recursive: 
    - resulting param derivatives are evaluated for inc_deriv and inc_range cross-comparison, to form par_Ps and so on
    - resulting vertically adjacent dPPs and vPPs are evaluated for cross-comparison, to form PPPs and so on

    comp_P blob formed by intra_comp(dx, no g calc) -> sub_blob per dx in der+ blob | v in rng+ blob -=ig: different dim?
    core comp within P -> vdP, ddP, ortho?  then blobs redefine per ddx, dvx, vd, vv, etc.? recursively -> nested PPs?
    
    map seg,P -> sub-seg,P for nested intra_comp(dx_g)  
    -> comp_P_ eval by combined overlapped-blob elongation:  
    
    val_PP_ = 
    V + G: core params represent value of all others, no max Pm = L + |V| + |Dx| + |Dy|: redundant and coef filtered? 
    # orientation coefs: 
    * max(ave_Lx, Ly) / min(ave_Lx, Ly)  # dim bias: g match rate, max comp_P rng?    
    * max(Dy, Dx) / min(Dy, Dx)    # variation bias (no Mx, My: low bias?); both biases are signed and may cancel-out 
    
    * Ave_blob / Ga  # angle match rate: stability of variation bias -> secondary comp( d | ortho_d)?   
    eval per blob, too expensive for seg?  no abs_Dx, abs_Dy for comp dert eval: mostly redundant?
'''
ave = 20
div_ave = 200
flip_ave = 1000


def comp_P(ortho, P, _P, DdX):  # forms vertical derivatives of P params, also conditional ders from norm and DIV comp

    s, x0, G, A, M, Dx, Dy, L, derts_ = P  # ext: X, new: L, dif: Dx, Dy -> G, no comp of inp I in top dert?
    _s, _x0, _G, _A, _M, _Dx, _Dy, _L, _derts_, _dX = _P  # params per comp_branch, S x branch if min n?

    xn = x0 + L-1;  _xn = _x0 + _L-1
    overlap = min(xn, _xn) - max(x0, _x0)
    offset = abs(x0 - _x0) + abs(xn - _xn)

    mX = overlap / offset  # mX is L-normalized, individual x m|d is binary
    dX = (x0 + (L-1)//2) - (_x0 + (_L-1)//2)  # d_ave_x, vX = mX - ave_mX -> P inclusion, or distant-P comp only?

    ddX = dX - _dX  # for ortho eval if first-run ave_DdX * Pm: += compensated angle change,
    DdX += ddX  # mag correlation: dX-> L, ddX-> dL, neutral to Dx: mixed with anti-correlated oDy?

    if ortho:  # if ave_dX * val_PP_: estimate params of P orthogonal to long axis, to maximize lat diff and vert match

        hyp = hypot(dX, 1)  # long axis increment = hyp / 1 (vertical distance), to estimate params of orthogonal slice:
        L /= hyp
        Dx = (Dx * hyp + Dy / hyp) / 2 / hyp
        Dy = (Dy / hyp - Dx * hyp) / 2 / hyp  # est D over vert_L, Ders summed in vert / lat ratio?

    dL = L - _L; mL = min(L, _L)  # comp Derts[1] -> abs match, dderived: magnitude-proportional value
    dM = M - _M; mM = min(M, _M)  # or V / G? no Mx, My: non-core, lesser and redundant bias?

    dDx = abs(Dx) - abs(_Dx); mDx = min(abs(Dx), abs(_Dx))  # same-sign Dx in vxP
    dDy = Dy - _Dy; mDy = min(Dy, _Dy)  # Dy per sub_P by intra_comp(dx), vs. less vertically specific dI

    Pd = ddX + dL, dM, + dDx + dDy  # -> directional dPP, equal-weight params, no rdn?
    # correlation: dX -> L, oDy, !oDx, ddX -> dL, odDy ! odDx? dL -> dDx, dDy?  G = hypot(Dy, Dx) for 2D structures comp?
    Pm = mX + mL + mM, mDx + mDy  # -> complementary vPP, rdn *= Pd | Pm rolp?

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

    P_ders = Pm, Pd, mX, dX, mL, dL, mDx, dDx, mDy, dDy, div_f, nvars

    vs = 1 if Pm > ave * 7 > 0 else 0  # comp cost = ave * 7, or rep cost: n vars per P?
    ds = 1 if Pd > 0 else 0

    return (P, P_ders), vs, ds


def comp_P_(val_PP_, blob, Ave, xD):  # scan of vertical Py_ -> comp_P -> 2D mPPs and dPPs, recursive?

    # differential Pd -> dPP and Pm -> vPP: dderived params magnitude is the only proxy to predictive value

    G, Dy, Dx, N, L, Ly, sub_ = blob.Dert[0]  # G will be redefined from Dx, Dy, or only per blob for 2D comp?
    max_y, min_y, max_x, min_x = blob.box
    DdX = 0

    if val_PP_ * ((max_x - min_x + 1) / (max_y - min_y + 1)) * (max(abs(Dx), abs(Dy)) / min(abs(Dx), abs(Dy))) > flip_ave:
        flip(blob)
        # vertical blob rescan -> comp_Px_ if PM gain: D-bias <-> L-bias: width / height, vs abs(xD) / height for oriented blobs?
        # or flip_eval(positive xd_dev_P (>>90)), after scan_Py_-> xd_dev_P?

    if xD / Ly * val_PP_ > Ave: ort = 1  # estimate params of Ps orthogonal to long axis, seg-wide for same-syntax comp_P
    else: ort = 0  # to max ave ders, or if xDd to min Pd?

    mPP_, dPP_, CmPP_, CdPP_, Cm_, Cd_ = [],[],[],[],[],[]  # C for combined comparable params and their derivatives

    mPP = 0, [], []  # per dev of M_params, dderived: match = min, G+=Ave?
    dPP = 0, [], []  # per dev of D_params: abs or co-signed?

    Py_ = blob[3][-1]  # per segment, also flip eval per seg?
    _P = Py_.popleft()  # initial comparand

    while Py_:  # comp_P starts from 2nd P, top-down
        P = Py_.popleft()
        _P, _ms, _ds = comp_P(ort, P, _P, DdX)

        while Py_:  # form_PP starts from 3rd P
            P = Py_.popleft()
            P, ms, ds = comp_P(ort, P, _P, DdX)  # P: S_vars += S_ders in comp_P
            if ms == _ms:
                mPP = form_PP(1, P, mPP)
            else:
                mPP = term_PP(1, mPP)  # SPP += S, PP eval for orient, incr_comp_P, scan_par..?
                mPP_.append(mPP)
                for par, C in zip(mPP[1], CmPP_):  # blob-wide summation of 16 summed vars from incr_PP
                    C += par
                    Cm_.append(C)  # or C is directly modified in CvPP?
                CmPP_ = Cm_  # but CPP is redundant, if len(PP_) > ave?
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
                dPP = ds, [], []

            _P = P; _ms = ms; _ds = ds
    return blob, CmPP_, mPP_, CdPP_, dPP_  # blob | PP_? comp_P over fork_, after comp_segment?


'''
    selection by dx: cross-dimension in oriented blob, recursive 1D alg -> nested Ps?
    G redefined by Dx, Dy for alt comp, or only per blob for 2D comp?

    aS compute if positive eV (not qD?) = mx + mL -ave? :
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

'''
    colors will be defined as color / sum-of-colors, color Ps are defined within sum_Ps: reflection object?
    relative colors may match across reflecting objects, forming color | lighting objects?     
    comp between color patterns within an object: segmentation?
'''