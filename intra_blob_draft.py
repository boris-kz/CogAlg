from collections import deque
import math as math
from time import time
import frame_blobs

'''
    intra_blob() is an extension to frame_blobs, it performs evaluation for comp_P and recursive frame_blobs within each blob.
    Currently it's mostly a draft, combined with frame_blobs it will form a 2D version of first-level algorithm
    inter_blob() will be second-level 2D algorithm, and a prototype for meta-level algorithm
    
    colors will be defined as color / sum-of-colors, color Ps are defined within sum_Ps: reflection object?
    relative colors may match across reflecting objects, forming color | lighting objects?     
    comp between color patterns within an object: segmentation?
    
    inter_olp_blob: scan alt_typ_ ) alt_color, rolp * mL > ave * max_L?   
    intra_blob rdn is eliminated by merging blobs, reduced by full inclusion: mediated access?
'''

def blob_eval(blob):
    s, [min_x, max_x, min_y, max_y, xD, abs_xD, Ly], [L, I, G, Dx, Dy, abs_Dx, abs_Dy], root_ = blob

    if L > A_cost:  # fixed per blob: ini params + ini params * (added params / ini params), converted to min L?
        if G > (ave + a_cost) * L:  # comp_a delay - comp delay per dert,

            blob = comp_angle(blob)  # angle comp, ablob def; a, da, sda accum in higher-composition reps
            sDa = blob[2][7]

            if G * -sDa > ave * L * 2:  # only after comp_a, -sDa indicates likely d match
                blob = incr_deriv(blob)  # recursion over ds: dderived?

    if abs_Dx - abs_Dy > flip_ave:  # blob Ds are horizontally oriented, projecting flipped PM > ave * 5?
        blob = flip(blob)  # vertical rescan -> Pys for comp_P
        # vs. scan_Py_ -> xdP, flip_eval(xdP)?  if < 90: param *= angle? or immediate comp_P if flip?

    rMy = (ave * L) / (blob[0][5] * 1.4)  # vertical M coef: inverted Dy / ave_D
    P_sum = L + I + G + Dx + Dy  # or abs_Dx, abs_Dy: sum between Ps, more accurate but not needed for most blobs?

    proj_PM = P_sum * rMy * math.hypot(Ly, abs_xD / Ly)  # projected match between Ps
    #  P_sum: max P match, hypot (long axis): span of Der summation, to justify added syntax

    if proj_PM > ave * 5:  # evaluate blob for comp_P along Py_, 5 params * comp cost, primary comp_P | recursion eval?
        scan_Py_(0, blob, xD)  # leading to comp_P, etc.

    if G > ave * L * 2:   # sG is absolute variation, indicating likely d reversal and match among distant pixels
        blob = incr_range(blob)  # recursion over +distant ps, including diagonal?

    '''
    dCx = max_x - min_x + 1;  dCy = max_y - min_y + 1
    rC = dCx / dCy  # width / height, vs shift / height: abs(xD) / Ly for oriented blobs only?
    rD = max(abs_Dx, abs_Dy) / min(abs_Dx, abs_Dy)  # lateral variation / vertical variation, for flip and comp_P eval
    '''
    return blob

def comp_angle(blob):
    # compute and compare angle, define ablobs, accumulate a, da, sda in higher-composition reps within input blob

    s, [min_x, max_x, min_y, max_y, xD, abs_xD, Ly], [L, I, G, Dx, Dy, abs_Dx, abs_Dy], root_ = blob
    A, Da, sDa = 0, 0, 0

    for i, seg in enumerate(range(len(root_))):
        [min_xs, max_xs, min_ys, xDs, ave_x], [Ls, Is, Gs, Dxs, Dys], Py_, fork_, roots, blob_ref = seg[1:]  # ignore s
        # first P of seg: scan higher-line _Ps in fork_
        P, xd = Py_[0]
        lateral_comp_a(P)
        _P_ = []
        for fork in fork_:
            _P_.append(fork[3][-1][0])  # get a list of _P from fork_

        P = vertical_comp_a(P, _P_)  # reconstruct P
        Py_[0] = P, xd
        As, Das, sDas = P[2][-3:]  # P[2]: P's params
        for ii, P in enumerate(range(len(Py_[1:]))):
            _P = Py_[ii-1][0]
            P, xd = P
            lateral_comp_a(P)
            P = vertical_comp_a(P, _P)
            Py_[ii] = P, xd
            As += P[2][-3]
            Das += P[2][-2]
            sDas += P[2][-1]
        root_[i] = s, (min_xs, max_xs, min_ys, xDs), (Ls, Is, Gs, Dxs, Dys, As, Das, sDas), tuple(Py_), fork_, roots
        A += As
        Da += Das
        sDa += sDas

    return s, (min_x, max_x, min_y, xD, Ly), (L, I, G, Dx, Dy, A, Da, sDa), tuple(root_)

def lateral_comp_a(P):

    dert_ = P[3]
    dx, dy = dert_[0][-2:]  # first dert
    _a = int((math.atan2(dy, dx)) * degree) + 128  # angle from 0 -> 255
    da = ave
    dert_[0] += _a, da
    for i, dert in enumerate(dert_[1:]):
        dx, dy = dert[-2:]
        a = int((math.atan2(dy, dx)) * degree) + 128
        da = abs(a - _a)
        dert_[i] += a, da
        # aP = form_P(dert, _dert)  # i/o must be extended
        _a = a
    P[3] = dert_

def vertical_comp_a(P, *_P_):
    s, [min_x, max_x], [L, I, G, Dx, Dy], dert_ = P
    x = min_x; i = 0
    for _P in _P_:
        [_min_x, _max_x], _dert_ = _P[1], _P[3]
        if x < _min_x:
            i += _min_x - x
            x = _min_x
            _i = 0
        else:
            _i = x - min_x
        while _dert_[_i] and dert_[i]:
            _a = dert_[i][-2]
            p, g, dx, dy, a, da = dert_[i]
            da += abs(a - _a)
            sda = 2 * ave - da
            dert_[i] = p, g, dx, dy, a, da, sda
            x += 1; i += 1; _i += 1

    A, Da, sDa = 0, 0, 0
    for i, dert in enumerate(dert_):
        p, g, dx, dy, a, da = dert
        if dert[6]:
            sda = dert[6]
        else:
            sda = ave - da  # da += ave; sda = 2 * ave - da <=> sda = ave - da ?
        dert_[i] = (p, g, dx, dy, a, da, sda)
        A += a
        Da += da
        sDa += sda

    return s, (min_x, max_x), (L, I, G, Dx, Dy, A, Da, sDa), tuple(dert_)


def incr_range(blob):  # frame_blobs recursion if sG
    return blob

def incr_deriv(blob):  # frame_blobs recursion if Dx + Dy: separately, or abs_Dx + abs_Dy: directional, but for both?
    return blob

def flip(blob):  # vertical-first run of form_P and deeper functions over blob's ders__
    return blob

def scan_Py_(norm, blob, xD):  # scan of vertical Py_ -> comp_P -> 2D mPPs and dPPs

    vPP = 0,[],[]  # s, PP (with S_ders), Py_ (with P_ders and e_ per P in Py)
    dPP = 0,[],[]  # PP: L2, I2, D2, Dy2, M2, My2, G2, Olp2

    SvPP, SdPP, Sv_, Sd_ = [],[],[],[]
    vPP_, dPP_, yP_ = [],[],[]

    Py_ = blob[2]  # unless oriented?
    _P = Py_.popleft()  # initial comparand

    while Py_:  # comp_P starts from 2nd P, top-down

        P = Py_.popleft()
        _P, _vs, _ds = comp_P(norm, P, _P, xD)  # per blob, before orient

        while Py_:  # form_PP starts from 3rd P

            P = Py_.popleft()
            P, vs, ds = comp_P(norm, P, _P, xD)  # P: S_vars += S_ders in comp_P

            if vs == _vs:
                vPP = form_PP(1, P, vPP)
            else:
                vPP = term_PP(1, vPP)  # SPP += S, PP eval for orient, incr_comp_P, scan_par..?
                vPP_.append(vPP)
                for par, S in zip(vPP[1], SvPP):  # blob-wide summation of 16 S_vars from incr_PP
                    S += par
                    Sv_.append(S)  # or S is directly modified in SvPP?
                SvPP = Sv_  # but SPP is redundant, if len(PP_) > ave?
                vPP = vs, [], []  # s, PP, Py_ init

            if ds == _ds:
                dPP = form_PP(0, P, dPP)
            else:
                dPP = term_PP(0, dPP)
                dPP_.append(dPP)
                for var, S in zip(dPP[1], SdPP):
                    S += var
                    Sd_.append(S)
                SdPP = Sd_
                dPP = ds,[],[]

            _P = P; _vs = vs; _ds = ds

    ''' S_ders | S_vars eval for PP ) blob ) network orient, incr distance | derivation comp_P
        redun alt P ) pP) PP ) blob ) network? '''

    return blob, SvPP, vPP_, SdPP, dPP_  # blob | PP_? comp_P over fork_, after comp_segment?


def comp_P(norm, P, _P, xD):  # forms vertical derivatives of P vars, also conditional ders from DIV comp

    (s, x0, L, I, D, Dy, M, My, Alt0, Alt1, Alt2, ders_), xd = P
    (_s, _x0, _L, _I, _D, _Dy, _M, _My, _Alt0, _Alt1, _Alt2, _ders_), _xd = P

    ddx = 0  # optional, 2Le norm / D? s_ddx and s_dL correlate, s_dx position and s_dL dimension don't?

    mx = (x0 + L-1) - _x0  # vx = ave_xd - xd: distance (cost) decrease vs. benefit incr? or:
    if x0 > _x0: mx -= x0 - _x0  # mx = x olp, - ave_mx -> vxP, distant P mx = -(ave_xd - xd)?

    dL = L - _L; mL = min(L, _L)  # relative olp = mx / L? ext_miss: Ddx + DL?
    dI = I - _I; mI = min(I, _I)  # L and I are dims vs. ders, not rdn | select, I per quad, no norm?

    if norm:  # if xD: derivatives are xd- normalized before comp:
        hyp = math.hypot(xD, 1)  # len incr = hyp / 1 (vert distance == 1)

        D = (D * hyp + Dy / hyp) / 2 / hyp  # est D over ver_L, Ders summed in ver / lat ratio
        Dy= (Dy / hyp - D * hyp) / 2 * hyp  # est D over lat_L
        M = (M * hyp + My / hyp) / 2 / hyp  # est M over ver_L
        My= (My / hyp + M * hyp) / 2 * hyp  # est M over lat_L; G is combined: not adjusted

    dD = D - _D; mD = min(D, _D)
    dM = M - _M; mM = min(M, _M)

    dDy = Dy - _Dy; mDy = min(Dy, _Dy)  # lat sum of y_ders also indicates P match and orientation?
    dMy = My - _My; mMy = min(My, _My)

    # oG in Pm | Pd: lat + vert- quantified e_ overlap (mx)?  no G comp: redundant to ders

    Pd = ddx + dL + dI + dD + dDy + dM + dMy  # defines dPP, dx does not correlate
    Pm = mx + mL + mI + mD + mDy + mM + mMy  # defines vPP; comb rep value = Pm * 2 + Pd?

    if dI * dL > div_ave:  # potential d compression, vs. ave * 21(7*3)?

        # DIV comp: cross-scale d, neg if cross-sign, no ndx: single, yes nmx: summed?
        # for S: summed vars I, D, M: nS = S * rL, ~ rS,rP: L defines P?

        rL = L / _L  # L defines P, SUB comp of rL-normalized nS:
        nI = I * rL; ndI = nI - _I; nmI = min(nI, _I)  # vs. nI = dI * nrL?

        nD = D * rL; ndD = nD - _D; nmD = min(nD, _D)
        nM = M * rL; ndM = nM - _M; nmM = min(nM, _M)

        nDy = Dy * rL; ndDy = nDy - _Dy; nmDy = min(nDy, _Dy)
        nMy = My * rL; ndMy = nMy - _My; nmMy = min(nMy, _My)

        Pnm = mx + nmI + nmD + nmDy + nmM + nmMy  # normalized m defines norm_vPP, if rL

        if Pm > Pnm: nvPP_rdn = 1; vPP_rdn = 0  # added to rdn, or diff alt, olp, div rdn?
        else: vPP_rdn = 1; nvPP_rdn = 0

        Pnd = ddx + ndI + ndD + ndDy + ndM + ndMy  # normalized d defines norm_dPP or ndPP

        if Pd > Pnd: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
        else: dPP_rdn = 1; ndPP_rdn = 0

        div_f = 1
        nvars = Pnm, nmI, nmD, nmDy, nmM, nmMy, vPP_rdn, nvPP_rdn, \
                Pnd, ndI, ndD, ndDy, ndM, nmMy, dPP_rdn, ndPP_rdn

    else:
        div_f = 0  # DIV comp flag
        nvars = 0  # DIV + norm derivatives

    P_ders = Pm, Pd, mx, xd, mL, dL, mI, dI, mD, dD, mDy, dDy, mM, dM, mMy, dMy, div_f, nvars

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


def intra_blob(frame):  # evaluate blobs for orthogonal flip, incr_rng_comp, incr_der_comp, comp_P
    [neg_mL, neg_myL, I, D, Dy, M, My], [xD, Ly, blob_] = frame

    _blob_ = []
    for blob in blob_:
        if blob[0]:  # positive g sign
            _blob_.append( blob_eval(blob) )
    frame[1][2] = _blob_

    return frame  # frame of 2D patterns, to be outputted to level 2


# ************ MAIN FUNCTIONS END ***************************************************************************************

# ************ PROGRAM BODY *********************************************************************************************

# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:

ave = 15                # g value that coincides with average match: gP filter
div_ave = 1023          # filter for div_comp(L) -> rL, summed vars scaling
flip_ave = 10000        # cost of form_P and deeper?
ave_rate = 0.25         # match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)
dim = 2                 # number of dimensions
rng = 2                 # number of pixels compared to each pixel in four directions
min_coord = rng * 2 - 1 # min x and y for form_P input: ders2 from comp over rng*2 (bidirectional: before and after pixel p)
degree = 128 / math.pi  # coef to convert radian to 256 degrees
A_cost = 1000
a_cost = 15

# Main ---------------------------------------------------------------------------
start_time = time()
frame = intra_blob(frame_blobs)
end_time = time() - start_time
print(end_time)
