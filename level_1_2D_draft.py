from scipy import misc
from collections import deque
import numpy as np

''' Level 1 with patterns defined by the sign of quadrant gradient: modified core algorithm of levels 1+2.

    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match.
    
    Pixels are discrete samples of continuous image, so rightward and downward derivatives per pixel are 
    equally representative samples of 0-90 degree quadrant gradient: minimal unique unit of 2D gradient. 
    Such gradient is computed as the average of these two orthogonally diverging derivatives.
   
    2D blobs are defined by same-sign quadrant gradient, of value for vP or difference for dP.
    Level 1 has 5 steps of encoding, incremental per line defined by vertical coordinate y:

    y:   comp(p_): lateral comp -> tuple t,
    y-1: ycomp(t_): vertical comp -> quadrant t2,
    y-1: form_P(t2_): lateral combination -> 1D pattern P,  
    y-2: form_P2(P_): vertical scan_P_, fork_eval, form_blob, comp_P, form_PP -> 2D P2 
    y-3: term_P2(P2_): P2s are evaluated for termination, re-orientation, and re-consolidation 

    All 2D functions (ycomp, scan_P_, etc.) input two lines: relatively higher and lower.
    Higher-line patterns include additional variables, derived while they were lower-line patterns
    
    postfix '_' denotes array name, vs. same-name elements of that array 
    prefix '_' denotes higher-line variable or pattern '''


def comp(p_):  # comparison of consecutive pixels within line forms tuples: pixel, match, difference

    t_ = []  # complete fuzzy tuples: summation range = rng
    it_ = deque(maxlen=rng)  # incomplete fuzzy tuples: summation range < rng

    for p in p_:
        index = 0

        for it in it_:  # incomplete tuples, with summation range from 0 to rng
            pri_p, fd, fm = it

            d = p - pri_p  # difference between pixels
            m = min(p, pri_p)  # match between pixels

            fd += d  # fuzzy d: sum of ds between p and all prior ps within it_
            fm += m  # fuzzy m: sum of ms between p and all prior ps within it_

            it = pri_p, fd, fm
            it_[index] = it
            index += 1

        if len(it_) == rng:

            t = pri_p, fd, fm
            t_.append(t)  # completed tuple is transferred from it_ to t_

        it = p, 0, 0  # fd and fm are directional, initialized per new p
        it_.appendleft(it)  # new prior tuple

    t_ += it_  # last number = rng of tuples that remain incomplete
    return t_


def ycomp(t_, t2__, _vP_, _dP_):  # vertical comparison between pixels, forms 2D t2: p, d, dy, m, my

    vP_ = []; vP = [0,0,0,0,0,0,0,0,[]]  # pri_s, I, D, Dy, M, My, G, Olp, e_
    dP_ = []; dP = [0,0,0,0,0,0,0,0,[]]  # pri_s, I, D, Dy, M, My, G, Olp, e_

    x = 0; new_t2__ = []   # t2_ buffer: 2D array
    olp, ovG, odG = 0,0,0  # len of overlap between vP and dP, and gs summed over olp, all shared

    for t, t2_ in zip(t_, t2__):  # compares vertically consecutive pixels, forms quadrant gradients

        p, d, m = t
        index = 0
        x += 1

        for t2 in t2_:
            pri_p, _d, fdy, _m, fmy = t2

            dy = p - pri_p  # vertical difference between pixels
            my = min(p, pri_p)  # vertical match between pixels

            fdy += dy  # fuzzy dy: sum of dys between p and all prior ps within t2_
            fmy += my  # fuzzy my: sum of mys between p and all prior ps within t2_

            t2 = pri_p, _d, fdy, _m, fmy
            t2_[index] = t2
            index += 1

        if len(t2_) == rng:  # 2D tuple is completed and moved from t2_ to form_P:

            dg = _d + fdy  # d gradient
            vg = _m + fmy - ave  # v gradient
            t2 = pri_p, _d, fdy, _m, fmy  # 2D tuple

            # form 1D patterns vP and dP: horizontal spans of same-sign vg or dg, with associated vars:

            olp, ovG, odG, vP, dP, vP_, _vP_ = form_P(1, t2, vg, dg, olp, ovG, odG, vP, dP, vP_, _vP_, x)
            olp, odG, ovG, dP, vP, dP_, _dP_ = form_P(0, t2, dg, vg, olp, odG, ovG, dP, vP, dP_, _dP_, x)

        t2 = p, d, 0, m, 0  # fdy and fmy are initialized at 0
        t2_.appendleft(t2)  # new prior tuple is added to t2_, replaces completed one
        new_t2__.append(t2_)
        
    # line ends, vP and dP term, no init, inclusion per incomplete lateral fd and fm:

    if olp:  # if vP x dP overlap len > 0, incomplete vg - ave / (rng / X-x)?

        odG *= ave_k; odG = odG.astype(int)  # ave_k = V / I, to project V of odG

        if ovG > odG:  # comp of olp vG and olp dG, == goes to vP: secondary pattern?
            dP[7] += olp  # overlap of lesser-oG vP or dP, or P = P, Olp?
        else:
            vP[7] += olp  # to form rel_rdn = alt_rdn / len(e_)

    if y + 1 > rng:  # starting with the first line of complete t2s

        vP_, _vP_ = scan_P_(0, vP, vP_, _vP_, x)  # empty _vP_
        dP_, _dP_ = scan_P_(1, dP, dP_, _dP_, x)  # empty _dP_

    return new_t2__, vP_, dP_  # extended in scan_P_, renamed as arguments _vP_, _dP_

    # ? alt_: top P alt = Olp, oG, alt_oG: to remove if hLe demotion and alt_oG < oG?
    # P_ redefined as np.array ([P, alt_, roots, forks): to increment without init?


def form_P(typ, t2, g, alt_g, olp, oG, alt_oG, P, alt_P, P_, _P_, x):  # forms 1D dP or vP

    p, d, dy, m, my = t2  # 2D tuple of quadrant variables per pixel
    pri_s, I, D, Dy, M, My, G, Olp, e_ = P

    s = 1 if g > 0 else 0  # g = 0 is negative?
    if s != pri_s and x > rng + 2:  # P is terminated

        if typ: alt_oG *= ave_k; alt_oG = alt_oG.astype(int)  # ave V / I, to project V of odG
        else: oG *= ave_k; oG = oG.astype(int)               # same for h_der and h_comp eval?

        if oG > alt_oG:  # comp between overlapping vG and dG
            Olp += olp  # olp is assigned to the weaker of P | alt_P, == -> P: local access
        else:
            alt_P[7] += olp

        P = pri_s, I, D, Dy, M, My, G, Olp, e_  # no A = ave * alt_rdn / e_: dA < cost?
        P_, _P_ = scan_P_(typ, P, P_, _P_, x)  # scan over contiguous higher-level _Ps

        I, D, Dy, M, My, G, Olp, e_ = 0,0,0,0,0,0,0,[]  # P and olp initialization
        olp, oG, alt_oG = 0,0,0

    # continued or initialized vars are accumulated:

    olp += 1  # len of overlap to stronger alt-type P, accumulated until P or _P terminates
    oG += g; alt_oG += alt_g  # for eval to assign olp to alt_rdn of vP or dP

    I += p    # pixels summed within P
    D += d    # lateral D, for P comp and P2 orientation
    Dy += dy  # vertical D, for P2 normalization
    M += m    # lateral D, for P comp and P2 orientation
    My += my  # vertical M, for P2 orientation
    G += g    # d or v gradient summed to define P value, or V = M - 2a * W?

    if typ:
        pri = p, g, alt_g  # g = v gradient
        e_.append(pri)  # pattern element: prior same-level quadrant, for selective incremental range
    else:
        e_.append(g)  # g = d gradient and pattern element, for selective incremental derivation

    P = [s, I, D, Dy, M, My, G, Olp, e_]
    return olp, oG, alt_oG, P, alt_P, P_, _P_  # accumulated in ycomp


def scan_P_(typ, P, P_, _P_, x):  # P scans overlapping _Ps in _P_ for inclusion into attached P2s

    A = ave  # initialization before accumulation per rdn fork, cached | in P, no component adjust?
    buff_ = [] # _P_ buffer for next P; alt_ -> rolp, alt2_ -> rolp2

    fork_, fork_vP_, fork_dP_ = deque(),deque(),deque()  # refs per P to compute and transfer forks
    s, I, D, Dy, M, My, G, Olp, e_ = P  # Olp: 1D overlap by stronger alt Ps

    ix = x - len(e_)  # initial x of P
    _ix = 0  # initialized ix of _P displaced from _P_ by last scan_P_

    while x >= _ix:  # P to _P match eval, while horizontal overlap between P and _P_:

        oG = 0  # fork overlap gradient: oG += g
        ex = x  # ex is lateral coordinate of loaded P element
        _P = _P_.popleft()  # _P = _P in y-2, root_ in y-1, forks_P2 in y-3

        if s == _P[0][0]:  # if s == _s: vg or dg sign match: temporary fork_.append for fork_eval

            while ex > _P[0][1]: # _ix = _P[0][1]
        
                for e in e_:  # oG accumulation per P (Pm, Pd from comp_P only)

                    if typ: oG += e[1]  # if vP: e = p, g, alt_g
                    else: oG += e  # if dP: e = g
                    ex += 1

            fork_.append((oG, _P))  # or Pm, Pd in comp_P, vs. re-packing _P, rdn = sort order

        if _P[0][2] > ix:  buff_.append(_P)  # if _x > ix: _P is buffered for next-P comp

        else:  # no horizontal overlap between _P and next P, _P is removed from _P_

            if (_P[1] == 0 and y > rng + 3) or y == Y - 1:  # P2 term if root_ == 0

                blob_ = _P[2][2]
                for blob in blob_:

                    blob, vPP, dPP = blob  # <= one _vPP and _dPP per higher-level blob
                    term_P2(blob)  # eval for 2D P re-orient and re-scan, then recursion

                    if vPP: term_P2(vPP)  # if comp_P in fork_eval(blob)
                    if dPP: term_P2(dPP)  # not for dPP in dPP_: only to eval for rdn?

    P = s, ix, x, I, D, Dy, M, My, G, Olp, e_  # no x overlap between P and next _P

    if fork_:  # P is evaluated for inclusion into fork _Ps, appending y-2 fork_ and y-3 root_

        bA = A  # P eval for _P blob inclusion and comp_P, rdn vs. A?
        fork_, bA = fork_eval(2, P, fork_, bA, x)  # bA *= blob rdn

        if fork_vP_: # lateral len(dPP_): from comp_P over same forks, during fork_eval of blob_

            vA = bA  # eval for inclusion in vPPs (2D value patterns), rdn alt_ = blobs:
            fork_vP_, vA = fork_eval(0, P, fork_vP_, vA, x)

            dA = vA  # eval for inclusion in dPPs (2D difference patterns), rdn alt_ = vPPs:
            fork_dP_, dA = fork_eval(1, P, fork_dP_, dA, x)

            # individual vPPs and dPPs are also modified in their fork

    root_ = []  # same-type root Ps for future _P term eval, after displacement from _P_
    forks = fork_, fork_vP_, fork_dP_  # matching P2 on current y-2

    P_.append((P, root_, forks))  # _P_ = P_ for next-line scan_P_()
    buff_ += _P_  # excluding displaced _Ps

    return P_, buff_  # _P_ = buff_ for scan_P_(next P)

    # y-2: P + root_, forks_P2 <- form_P2 (to form_PP)
    # y-3: P2 + root_P2_, forks_seg_P2 <- segment if root_>1 or term if root_=0
    # y-n: segment P2s + root_seg_P2_, forks_seg_P2, term if root_-> term_, full term at last P


def fork_eval(typ, P, fork_, A, x):  # Ps evaluated for form_blob, comp_P, form_PP with fork _Ps

    fork_.sort(key=lambda fork: fork[0], reverse=True)  # max-to-min crit, or both sort & select

    ini = 1; select_ = deque()
    fork = fork_.pop()  # _P adds fork_blob | fork_vPP | fork_dPP
    crit, _P = fork   # criterion: oG for blob, Pm for fork_vPP, Pd for fork_dPP

    while fork_ and (crit > A or ini == 1):  # _P -> P2 inclusion if contiguous sign match

        ini = 0
        fA = A * (1 + P[9][0] / P[10])  # rel olp: 1 + alt_rdn / len(e_), or P adjust < cost?
        fork = fA, _P

        select_.appendleft(fork)
        A += A  # or olp_rdn += 1, then A * comb rdn: local and adjustable by hLe selection?

        fork = fork_.pop()  # repeat
        crit, _P = fork  # _P = _P, _root_, _forks

    init = 0 if select_ == 1 else 1
    for fork in select_:

        if typ == 2:  # fork = blob

            fork = form_blob(P, fork, init)  # crit is summed in _G, alt_rdn in fA?
            P = comp_P(P, fork, x)  # comp_P if form_blob? also typ for P' e_ comp?

        else:
            fork = form_PP(typ, P, fork, init)  # fork = vPP or dPP, after comp_P

        fork_.appendleft(fork)  # not-selected forks are out of fork_

    return fork_  # A or rdn is packed for higher P2-type accumulation?


def form_blob(P, fork, init):  # P inclusion into blob (initialized or continuing) of selected fork

    blob, root_, forks = fork  # forks: y-3 blob_, vPP_, dPP_, are not changed?
    s, ix, x, I, D, Dy, M, My, G, Olp, e_ = P  # or rdn = e_ / Olp + blob_rdn from fork_eval?

    if init:  # single fork_blob, fork_P2s are neither extended nor terminated

        L2 = len(e_)  # no separate e2_: Py_( P( e_? overlap / comp_P only?
        I2 = I
        D2 = D; Dy2 = Dy
        M2 = M; My2 = My
        G2 = G  # oG: vertical contiguity for fork_eval, also for comp_P?
        Olp2 = Olp  # + blob_rdn, before PP rdn?
        Py_ = [P]  # vertical array of patterns within a blob

    else:  # single fork continues, max fork if multi-A select?

        L2, I2, D2, Dy2, M2, My2, G2, Olp2, Py_ = blob

        L2 += len(e_)
        I2 += I
        D2 += D; Dy2 += Dy
        M2 += M; My2 += My
        G2 += G
        Olp2 += Olp
        Py_.append(P)

    blob = L2, G2, I2, D2, Dy2, M2, My2, Olp2, Py_
    root_.appendleft(P)  # adds P@ in input order, same-type roots only

    fork = blob, root_, forks  # P is replacing _P, which was summed into P2s
    return fork


def comp_P(P, _P, x):  # forms vertical derivatives of P vars, also from conditional DIV comp

    s, I, D, Dy, M, My, G, alt, e_ = P  # select alt_ per fork, no olp: = mx? no oG: fork sel
    _s, _ix, _x, _I, _D, _Dy, _M, _My, _G, Olp, _e_ = _P

    ddx = 0  # optional, 2Le norm / D? s_ddx and s_dL correlate, s_dx position and s_dL dimension don't?
    ix = x - len(e_)  # initial coordinate of P; S is generic for summed vars I, D, M:

    dx = x - len(e_)/2 - _x - len(_e_)/2  # Dx? comp(dx), ddx = Ddx / h? dS *= cos(ddx), mS /= cos(ddx)?
    mx = x - _ix; if ix > _ix: mx -= ix - _ix  # mx = x olp, - a_mx -> vxP, distant P mx = -(a_dx - dx)?

    dL = len(e_) - len(_e_); mL = min(len(e_), len(_e_))  # relative olp = mx / L? ext_miss: Ddx + DL?
    dI = I - _I; mI = min(I, _I)
    dD = D - _D; mD = min(D, _D)
    dM = M - _M; mM = min(M, _M)  # no G comp: y-derivatives are incomplete, no alt_ comp: rdn only?

    Pd = ddx + dL + dI + dD + dM  # defines dPP; var_P form if PP form, term if var_P or PP term;
    Pm = mx + mL + mI + mD + mM   # defines vPP; comb rep value = Pm * 2 + Pd? group by y_ders?

    if dI * dL > div_a: # DIV comp: cross-scale d, neg if cross-sign, nS = S * rL, ~ rS,rP: L defines P
                        # no ndx, yes nmx: summed?

        rL = len(e_) / len(_e_)  # L defines P, SUB comp of rL-normalized nS:
        nI = I * rL; ndI = nI - _I; nmI = min(nI, _I)  # vs. nI = dI * nrL?
        nD = D * rL; ndD = nD - _D; nmD = min(nD, _D)
        nM = M * rL; ndM = nM - _M; nmM = min(nM, _M)

        Pnm = mx + nmI + nmD + nmM  # normalized m defines norm_vPP, as long as rL is computed
        if Pm > Pnm: nvPP_rdn = 1; vPP_rdn = 0 # added to rdn, or diff alt, olp, div rdn?
        else: vPP_rdn = 1; nvPP_rdn = 0

        Pnd = ddx + ndI + ndD + ndM  # normalized d defines norm_dPP or ndPP
        if Pd > Pnd: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
        else: dPP_rdn = 1; ndPP_rdn = 0

        div_f = 1
        nvars = Pnm, nmI, nmD, nmM, vPP_rdn, nvPP_rdn, \
                Pnd, ndI, ndD, ndM, dPP_rdn, ndPP_rdn

    else:
        div_f = 0  # DIV comp flag
        nvars = 0  # DIV + norm derivatives

    ''' 
    no DIV comp(L): match is insignificant and redundant to mS, mLPs and dLPs only?
    
    if dL: nL = len(e_) // len(_e_)  # L match = min L mult
    else: nL = len(_e_) // len(e_)
    fL = len(e_) % len(_e_)  # miss = remainder 
    
    form_PP at fork_eval after full rdn: A = a * alt_rdn * fork_rdn * norm_rdn, 
    form_pP (parameter pattern) in +vPPs only, then cost of adjust for pP_rdn?
    
    comp_P is not fuzzy: x & y vars are already fuzzy?  
    eval per fork, PP, or yP, not per comp

    no comp aS: m_aS * rL cost, minor cpr / nL? no DIV S: weak nS = S // _S; fS = rS - nS  
    or aS if positive eV (not eD?) = mx + mL -ave:
    
    aI = I / len(e_); dI = aI - _aI; mI = min(aI, _aI)  
    aD = D / len(e_); dD = aD - _aD; mD = min(aD, _aD)  
    aM = M / len(e_); dM = aM - _aM; mM = min(aM, _aM)

    d_aS comp if cs D_aS, _aS is aS stored in _P, S preserved to form hP SS?
    iter dS - S -> (n, M, diff): var precision or modulus + remainder? '''

    P_ders = Pd, Pm, mx, dx, mL, dL, mI, dI, mD, dD, mM, dM, div_f, nvars
    P = P, P_ders

    return P  # for inclusion in vPP_, dPP_ by form_PP in fork_eval, P -> P_ in scan_P_


def form_PP(typ, P, fork, init):  # forms vPPs, dPPs, and pPs within each

    P, P_ders = P
    s, ix, x, I, D, Dy, M, My, G, rdn, e_ = P
    Pd, Pm, mx, dx, mL, dL, mI, dI, mD, dD, mM, dM, div_f, nvars = P_ders

    PP, root_, forks = fork

    if typ: crit = Pm # total match per pattern
    else: crit = Pd * ave_k  # projected match of difference per pattern

    if crit > ave * 5 * rdn:  # comp cost, or * number of vars per P: rep cost?

        if init:  # new PP and pP_ initialization:

            dxP_ = []; dx2 = dx, dxP_ # or ddx_P: known x match?
            mxP_ = []; mx2 = mx, mxP_

            LP_ = []; L2 = len(e_), LP_  # no rL, fL?
            IP_ = []; I2 = I, IP_
            DP_ = []; D2 = D, DP_
            MP_ = []; M2 = M, MP_
            Dy2 = Dy
            My2 = My
            G2 = G
            rdn2 = rdn
            Py_ = [P]

        else:  # increments current PP

            crit, rdn, L2, I2, D2, Dy2, M2, My2, G2, rdn2, Py_ = PP

            L2 += len(e_); I2 += I; D2 += D; Dy2 += Dy; M2 += M; My2 += My; G2 += G
            rdn_contents = rdn  # alt_P, alt_PP, fork, alt_pP?
            Py_.append(P)

            # mx, mL, mI, mD, mM; ddx, dL, dI, dD, dM;
            # also norm pPs?

        form_pP(dx, dxP_)
        form_pP(len(e_), LP_)
        form_pP(I, IP_)
        form_pP(D, DP_)
        form_pP(M, MP_)

    PP = L2, I2, D2, Dy2, M2, My2, G2, rdn2, Py_  # pP_s are packed in corr. parameters
    root_.appendleft(P)  # adds P@ in input order, same-type roots only

    fork = PP, root_, forks
    return fork


def form_pP(par, pP_):  # forming parameter patterns within PP

    # a_mx = 2; a_mw = 2; a_mI = 256; a_mD = 128; a_mM = 128: feedback to define vpPs: parameter value patterns
    # a_PM = a_mx + a_mw + a_mI + a_mD + a_mM  or A * n_vars, rdn accum per pP, alt eval per vertical overlap?

    # LIDV per dx, L, I, D, M? select per term?
    # alt2_: fork_ alt_ concat, to re-compute redundancy per PP


def term_P2(P2):  # blob | vPP | dPP eval for rotation, re-scan, re-comp, recursion, accumulation

    P2_ = []
    ''' 
    conversion of root to term, sum into wider fork, also sum per frame?
    
    dimensionally reduced axis: vP PP or contour: dP PP; dxP is direction pattern

    PP = PP, root_, blob_, _vPP_, _dPP_?
    vPP and dPP included in selected forks, rdn assign and form_PP eval after fork_ term in form_blob?

    blob= 0,0,0,0,0,0,0,0,0,0,[],[]  # crit, rdn, W, I2, D2, Dy2, M2, My2, G2, rdn2, alt2_, Py_
    vPP = 0,0,0,0,0,0,0,0,0,0,[],[]
    dPP = 0,0,0,0,0,0,0,0,0,0,[],[]  # P2s are initialized at non-matching P transfer to _P_?

    np.array for direct accumulation, or simply iterator of initialization?

    P2_ = np.array([blob, vPP, dPP],
        dtype=[('crit', 'i4'), ('rdn', 'i4'), ('W', 'i4'), ('I2', 'i4'), ('D2', 'i4'), ('Dy2', 'i4'),
        ('M2', 'i4'), ('My2', 'i4'), ('G2', 'i4'), ('rdn2', 'i4'), ('alt2_', list), ('Py_', list)]) 
    
    mean_dx = 1  # fractional?
    dx = Dx / H
    if dx > a: comp(abs(dx))  # or if dxP Dx: fixed ddx cost?  comp of same-sign dx only

    vx = mean_dx - dx  # normalized compression of distance: min. cost decrease, not min. benefit?
    
    
    eval of d,m adjust | _var adjust | x,y adjust if projected dS-, mS+ for min.1D Ps over max.2D

        if dw sign == ddx sign and min(dw, ddx) > a: _S /= cos (ddx)  # to angle-normalize S vars for comp

    if dw > a: div_comp (w): rw = w / _w, to width-normalize S vars for comp: 

        if rw > a: pn = I/w; dn = D/w; vn = V/w; 

            comp (_n) # or default norm for redun assign, but comp (S) if low rw?

            if d_n > a: div_comp (_n) -> r_n # or if d_n * rw > a: combined div_comp eval: ext, int co-variance?

        comp Dy and My, /=cos at PP term?  default div and overlap eval per PP? not per CP: sparse coverage?
        
    rrdn = 1 + rdn_w / len(e_)  # redundancy rate / w, -> P Sum value, orthogonal but predictive
    
    S = 1 if abs(D) + V + a * len(e_) > rrdn * aS else 0  # rep M = a*w, bi v!V, rdn I?
    '''


def root_2D(f):  # postfix '_' denotes array vs. element, prefix '_' denotes higher-line variable

    global rng; rng = 1
    global ave; ave = 127  # filters, ultimately set by separate feedback, then ave *= rng

    global div_a; div_a = 127  # not justified
    global ave_k; ave_k = 0.25  # average V / I

    global _vP2_; _vP2_ = []
    global _dP2_; _dP2_ = []  # 2D Ps terminated on line y-3

    global Y; global X; Y, X = f.shape  # Y: frame height, X: frame width
    global y; y = 0

    _vP_, _dP_, frame_ = [], [], []

    t2_ = deque(maxlen=rng)  # vertical buffer of incomplete pixel tuples, for fuzzy ycomp
    t2__ = []  # 2D (vertical buffer + horizontal line) array of 2D tuples, also deque for speed?
    p_ = f[0, :]  # first line of pixels
    t_ = comp(p_)

    for t in t_:
        p, d, m = t
        t2 = p, d, 0, m, 0  # fdy and fmy initialized at 0
        t2_.append(t2)  # only one tuple per first-line t2_
        t2__.append(t2_)  # in same order as t_

    for y in range(1, Y):  # vertical coordinate y is index of new line p_

        p_ = f[y, :]
        t_ = comp(p_)  # lateral pixel comparison
        t2__, _vP_, _dP_ = ycomp(t_, t2__, _vP_, _dP_) # vertical pixel comp, P and P2 form

        P2_ = _vP2_, _dP2_  # arrays of blobs terminated per line, adjusted by term_P2
        frame_.append(P2_)  # line of patterns is added to frame of patterns
        _vP2_, _dP2_ = [],[]

    return frame_  # frame of 2D patterns is outputted to level 2

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
root_2D(f)

