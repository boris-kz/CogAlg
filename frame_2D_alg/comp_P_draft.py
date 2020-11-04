'''
   comp_P_ is a terminal fork of intra_blob.
    
   comp_P_ traces blob axis by cross-comparing vertically adjacent Ps: laterally contiguous slices across edge blob.
   It should vectorize edges: high-G low Ga blobs, into outlines of adjacent flats: low-G blobs.
   This is a form of dimensionality reduction.

   Double edge lines: assumed match between edges of high-deviation intensity, no need for cross-comp?
   secondary cross-comp of low-deviation blobs?   P comb -> intra | inter comp eval?
   radial comp extension for co-internal blobs:
   != sign comp x sum( adj_blob_) -> intra_comp value, isolation value, cross-sign merge if weak, else:
   == sign comp x ind( adj_adj_blob_) -> same-sign merge | composition:
   
   borrow = adj_G * rS: default sum div_comp S -> relative area and distance to adjj_blob_
   internal sum comp if mS: in thin lines only? comp_norm_G or div_comp_G -> rG?
   isolation = decay + contrast: 
   G - G * (rS * ave_rG: decay) - (rS * adj_G: contrast, = lend | borrow, no need to compare vG?)

   if isolation: cross adjj_blob composition eval, 
   else:         cross adjj_blob merge eval:
   blob merger if internal match (~raG) - isolation, rdn external match:  
   blob compos if external match (~rS?) + isolation, 

   Also eval comp_P over fork_?
   rng+ should preserve resolution: rng+_dert_ is dert layers, 
   rng_sum-> rng+, der+: whole rng, rng_incr-> angle / past vs next g, 
   rdn Rng | rng_ eval at rng term, Rng -= lost coord bits mag, always > discr? 
'''
from time import time
from collections import deque
from class_cluster import ClusterStructure, NoneType
from math import hypot
import numpy as np

ave = 20
div_ave = 200
flip_ave = 1000
ave_dX = 10  # difference between median x coords of consecutive Ps

class CP(ClusterStructure):
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
    sign = NoneType
    dert_ = list
    gdert_ = list
    Dg = int
    Mg = int

class Cstack(ClusterStructure):
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
    S = int
    Ly = int
    y0 = int
    Py_ = list
    blob = NoneType
    down_connect_cnt = int
    sign = NoneType
    fPP = bool  # PPy_ if 1, else Py_

class Cdert_P(ClusterStructure):
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
    dert_ = list
    ms = bool
    ds = bool

class CPP(ClusterStructure):
    PM = int
    PD = int
    MX = int
    DX = int
    ML = int
    DL = int
    MDx = int
    DDx = int
    MDy = int
    DDy = int
    fdiv = bool
    P_ = list


def cluster_Py_(stack, Ave):
    # scan of vertical Py_ -> comp_P -> form_PP -> 2D PPd_, PPm_: clusters of same-sign Pd | Pm deviation
    DdX = 0
    y0 = stack.y0
    yn = stack.y0 + stack.Ly
    x0 = min([P.x0 for P in stack.Py_])
    xn = max([P.x0 + P.L for P in stack.Py_])

    L_bias = (xn - x0 + 1) / (yn - y0 + 1)  # elongation: width / height, pref. comp over long dimension
    G_bias = min(abs(stack.Dx), abs(stack.Dy)) / max(abs(stack.Dx), abs(stack.Dy))
    # ddirection: max(Gy,Gx) / min(Gy,Gx), pref. comp over low G
    # or y/x (L_bias * G_bias) max / min?

    if stack.G * L_bias * G_bias > flip_ave:
        flip_yx(stack)  # 90 degree rotation, vertical blob rescan -> comp_Px_ if projected PM gain
    '''
       if orientation < 1: 
          orientation = 1 / orientation; flip_cost = flip_ave  # no separate L, D orientation?
       else: flip_cost = 0
       comp_P_ if (G + M) * orientation - flip_cost > Ave_comp_P? '''

    if stack.G * (stack.Dx / stack.Dy) * stack.Ly_  > Ave: ort = 1
    # virtual rotation: if G * L_bias * L_bias after any rescan: estimate params of Ps as orthogonal to long axis, to increase PM
    else: ort = 0

    mPP_, dPP_, CmPP_, CdPP_, Cm_, Cd_ = [], [], [], [], [], []  # "C" is for combined comparable params and their derivatives
    mPP = CPP()
    dPP = CPP()

    while stack.Py_:  # comp_P starts from 2nd P, top-down
        P = stack.Py_.pop(0)
        _dert_P = comp_P(ort, P, _P, DdX)

        while stack.Py_:  # form_PP starts from 3rd P
            P = stack.Py_.pop(0)
            dert_P = comp_P(ort, P, _P, DdX)  # P: S_vars += S_ders in comp_P
            if dert_P.ms == _dert_P.ms:
                mPP = form_PP(1, P, mPP)
            else:
            # under review, disregard
                mPP = term_PP(1, mPP)  # SPP += S, PP eval for orient, incr_comp_P, scan_par..?
                mPP_.append(mPP)
                for par, C in zip(mPP[1], CmPP_):  # blob-wide summation of 16 summed vars from incr_PP
                    C += par
                    Cm_.append(C)  # or C is directly modified in CvPP?
                CmPP_ = Cm_  # but CPP is redundant, if len(PP_) > ave?
                mPP = dert_P.ms, [], []  # s, PP, Py_ init
            if dert_P.ds == _ds:
                dPP = form_PP(0, P, dPP)
            else:
                dPP = term_PP(0, dPP)
                dPP_.append(dPP)
                for var, C in zip(dPP[1], CdPP_):
                    C += var
                    Cd_.append(C)
                CdPP_ = Cd_
                dPP = dert_P.ds, [], []

            _P = P; _ms = dert_P.ms; _ds = dert_P.ds

    return stack, CmPP_, mPP_, CdPP_, dPP_


def flip_yx(Py_):  # vertical-first run of form_P and deeper functions over blob's ders__

    y0 = 0
    yn = len(Py_)
    x0 = min([P.x0 for P in Py_])
    xn = max([P.x0 + P.L for P in Py_])

    # initialize list containing y and x size, number of sublist = number of params
    dert__ = [(np.zeros((yn - y0, xn - x0)) - 1) for _ in range(len(Py_[0].dert_[0]))]
    mask__ = np.zeros((yn - y0, xn - x0)) > 0

    # insert Py_ value into dert__
    for y, P in enumerate(Py_):
        for x, idert in enumerate(P.dert_):
            for i, (param, dert) in enumerate(zip(idert, dert__)):
                dert[y, x] = param

    # create mask and set masked area = True
    mask__[np.where(dert__[0] == -1)] = True

    # rotate 90 degree clockwise, anti-clockwise is better for consistency?
    dert__flip = tuple([np.rot90(dert) for dert in dert__])
    mask__flip = np.rot90(mask__)

    Py_flip = []
    # form vertical patterns after rotation
    from P_blob import form_P_
    for y, dert_ in enumerate(zip(*dert__flip)):
        crit_ = dert_[3] > 0  # compute crit from G? dert_[3] is G
        P_ = form_P_(zip(*dert_), crit_, mask__flip[y])

        if len([P for P in P_]) > 0:  # empty P, when mask is masked for whole row or column, need check further on this
            Py_flip.append([P for P in P_][0])  # change deque of P_ into list

    return Py_flip

'''
    Pd and Pm are ds | ms per param summed in P. Primary comparison is by subtraction, div if par * rL compression: 
    DL * DS > min: must be both, eval per dPP PD, signed? comp d?
    
    - resulting vertically adjacent dPPs and vPPs are evaluated for cross-comparison, to form PPPs and so on
    - resulting param derivatives form par_Ps, which are evaluated for der+ and rng+ cross-comparison
    | default top+ P level: if PD | PM: add par_Ps: sub_layer, rdn ele_Ps: deeper layer? 
'''

def comp_P(ortho, P, _P, DdX):  # forms vertical derivatives of P params, and conditional ders from norm and DIV comp

    s, x0, (G, M, Dx, Dy, L), dert_ = P  # ext: X, new: L, dif: Dx, Dy -> G, no comp of inp I in top dert?
    _s, _x0, (_G, _M, _Dx, _Dy, _L), _dert_, _dX = _P  # params per comp_branch, S x branch if min n?
    '''
    redefine Ps by dx in dert_, rescan dert by input P d_ave_x: skip if not in blob?
    '''
    xn = x0 + L-1;  _xn = _x0 + _L-1
    mX = min(xn, _xn) - max(x0, _x0)  # overlap: abs proximity, cumulative binary positional match | miss:
    dX = abs(x0 - _x0) + abs(xn - _xn)  # offset, or max_L - overlap: abs distance?

    if dX > ave_dX:  # internal comp is higher-power, else two-input comp not compressive?
       rX = dX / mX  # average dist/prox, | prox/dist, | mX / max_L?
    ave_dx = (x0 + (L-1)//2) - (_x0 + (_L-1)//2)  # d_ave_x, median vs. summed, or for distant-P comp only?

    ddX = dX - _dX  # for ortho eval if first-run ave_DdX * Pm: += compensated angle change,
    DdX += ddX  # mag correlation: dX-> L, ddX-> dL, neutral to Dx: mixed with anti-correlated oDy?

    if ortho:  # if ave_dX * val_PP_: estimate params of P orthogonal to long axis, maximizing lat diff, vert match

        hyp = hypot(dX, 1)  # long axis increment (vertical distance), to adjust params of orthogonal slice:
        L /= hyp
        Dx = (Dx * hyp + Dy / hyp) / 2 / hyp
        Dy = (Dy / hyp - Dx * hyp) / 2 / hyp  # recompute? est D over vert_L, Ders summed in vert / lat ratio?

    dL = L - _L; mL = min(L, _L)  # L: positions / sign, dderived: magnitude-proportional value
    dM = M - _M; mM = min(M, _M)  # no Mx, My: non-core, lesser and redundant bias?

    dDx = abs(Dx) - abs(_Dx); mDx = min(abs(Dx), abs(_Dx))  # same-sign Dx in vxP
    dDy = Dy - _Dy; mDy = min(Dy, _Dy)  # Dy per sub_P by intra_comp(dx), vs. less vertically specific dI

    Pd = ddX + dL + dM + dDx + dDy  # -> directional dPP, equal-weight params, no rdn?
    # correlation: dX -> L, oDy, !oDx, ddX -> dL, odDy ! odDx? dL -> dDx, dDy?  G = hypot(Dy, Dx) for 2D structures comp?
    Pm = mX + mL + mM, mDx + mDy  # -> complementary vPP, rdn *= Pd | Pm rolp?

    P_ders = Pm, Pd, mX, dX, mL, dL, mDx, dDx, mDy, dDy  # div_f, nvars

    vs = 1 if Pm > ave * 7 > 0 else 0  # comp cost = ave * 7, or rep cost: n vars per P?
    ds = 1 if Pd > 0 else 0

    return (P, P_ders), vs, ds

'''
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
    '''
    Aves (integer filters) and coefs (ratio filters) per parameter type trigger formation of parameter_Ps,
    after full-blob comp_P_ sums match and miss per parameter.
    Also coefs per sub_blob from comp_blob_: potential parts of a higher object?
    '''
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

'''  
    rL: elongation = max(ave_Lx, Ly) / min(ave_Lx, Ly): match rate in max | min dime, also max comp_P rng?    
    rD: ddirection = max(Dy, Dx) / min(Dy, Dx);  low bias, indirect Mx, My: = M/2 *|/ ddirection?

    horiz_dim_val = ave_Lx - |Dx| / 2  # input res and coord res are adjusted so mag approximates predictive value,
    vertical_dim_val  = Ly - |Dy| / 2  # or proj M = M - (|D| / M) / 2: no neg? 
    
    core params G and M represent value of all others, no max Pm = L + |V| + |Dx| + |Dy|: redundant and coef-filtered?
    no * Ave_blob / Ga: angle match rate, already represented by hforks' position + mag' V+G -> comp( d | ortho_d)?   
    eval per blob, too expensive for seg? no abs_Dx, abs_Dy for comp dert eval: mostly redundant?
    
    # Dert is None if len | Var < min, for blob_, fork_, and layers?  

    colors will be defined as color / sum-of-colors, color Ps are defined within sum_Ps: reflection object?
    relative colors may match across reflecting objects, forming color | lighting objects?     
    comp between color patterns within an object: segmentation?
'''