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
    y-2: form_P2(P_): vertical scan_P_, fork_eval, form_blob, comp_P, form_PP -> 2D pattern P2 
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

            it_[index] = pri_p, fd, fm
            index += 1

        if len(it_) == rng:
            t_.append((pri_p, fd, fm))  # completed tuple is transferred from it_ to t_

        it_.appendleft((p, 0, 0))  # new prior tuple, fd and fm are initialized at 0

    t_ += it_  # last number = rng of tuples that remain incomplete
    return t_


def ycomp(t_, t2__, _vP_, _dP_):  # vertical comparison between pixels, forms 2D t2: p, d, dy, m, my

    vP_ = []; vP = [0,0,0,0,0,0,0,0,[]]  # pri_s, I, D, Dy, M, My, G, Olp, e_
    dP_ = []; dP = [0,0,0,0,0,0,0,0,[]]  # pri_s, I, D, Dy, M, My, G, Olp, e_

    x = 0; new_t2__ = []   # t2_ buffer: 2D array
    olp, ovG, odG = 0,0,0  # len of overlap between vP and dP, and gs summed over olp, all shared

    for t, t2_ in zip(t_, t2__):  # compares vertically consecutive pixels, forms quadrant gradients

        p, d, m = t; index = 0; x += 1

        for t2 in t2_:
            pri_p, _d, fdy, _m, fmy = t2

            dy = p - pri_p  # vertical difference between pixels
            my = min(p, pri_p)  # vertical match between pixels

            fdy += dy  # fuzzy dy: sum of dys between p and all prior ps within t2_
            fmy += my  # fuzzy my: sum of mys between p and all prior ps within t2_

            t2_[index] = pri_p, _d, fdy, _m, fmy
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
        
    # line ends, vP and dP term, no init, inclusion with incomplete lateral fd and fm:

    if olp:  # if vP x dP overlap len > 0, incomplete vg - ave / (rng / X-x)?

        odG *= ave_k; odG = odG.astype(int)  # ave_k = V / I, to project V of odG

        if ovG > odG:  # comp of olp vG and olp dG, == goes to vP: secondary pattern?
            dP[7] += olp  # overlap of lesser-oG vP or dP, or P = P, Olp?
        else:
            vP[7] += olp  # to form rel_rdn = alt_rdn / len(e_)

    if y + 1 > rng:  # starting with the first line of complete t2s

        vP_, _vP_ = scan_P_(0, vP, vP_, _vP_, x)  # returns empty _vP_
        dP_, _dP_ = scan_P_(1, dP, dP_, _dP_, x)  # returns empty _dP_

    return new_t2__, vP_, dP_  # extended in scan_P_, renamed as arguments _vP_, _dP_

    # poss alt_: top P alt = Olp, oG, alt_oG: to remove if hLe demotion and alt_oG < oG?
    # P_ can be redefined as np.array ([P, alt_, roots, forks) to increment without init?


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
        e_.append((p, g, alt_g))  # g = v gradient, for selective incremental range comp
    else:
        e_.append(g)  # g = d gradient and pattern element, for selective incremental derivation

    P = [s, I, D, Dy, M, My, G, Olp, e_]
    return olp, oG, alt_oG, P, alt_P, P_, _P_  # accumulated in ycomp


def scan_P_(typ, P, P_, _P_, x):  # P scans overlapping _Ps in _P_ for inclusion into attached P2s

    buff_ = [] # _P_ buffer for scan_P_(next_P)
    fork_ = deque()  # forks: _P_ references from P; _P, root_, fork_ = _P

    s, I, D, Dy, M, My, G, Olp, e_ = P  # Olp: 1D overlap by stronger alt Ps; or no need to unpack?
    ix = x - len(e_)  # initial x of P
    _ix = 0  # initialized ix of _P displaced from _P_ by last scan_P_

    while x >= _ix:  # P to _P match eval, while horizontal overlap between P and _P_:

        oG = 0  # fork overlap gradient: oG += g (distinct from alt P oG)
        ex = x  # ex is lateral coordinate of loaded P element
        _P = _P_.popleft()  # _P = _P in y-2, root_ in y-1, forks_P2 in y-3

        if s == _P[0][0]:  # if s == _s: vg or dg sign match: temporary fork_.append for fork_eval:

            while ex > _P[0][1]: # _ix = _P[0][1]
        
                for e in e_:  # oG accumulation per P (Pm, Pd from comp_P only)

                    if typ: oG += e[1]  # if vP: e = p, g, alt_g
                    else: oG += e  # if dP: e = g
                    ex += 1

            fork_.append((oG, _P))  # or Pm, Pd in comp_P, vs. re-packing _P, rdn = sort order

        if _P[0][2] > ix:  # if _x > ix:
            buff_.append(_P)  # _P is buffered for next-P comp

        else:  # no overlap between _P and next P, _P is out of _P_ and evaluated for blob term:

            if (_P[1] == 0 and _P[2] == 0 and y > rng) or y == Y - 1:

                term_blob(_P) # if root_== 0 and fork_== 0

    # no overlap between P and next _P, P eval for inclusion into fork _Ps:

    fork_.sort(key=lambda fork: fork[0], reverse=True)  # max-to-min oG, or both sort & select?

    select_ = deque(); rdn = 1  # number of select forks per P
    fork = fork_.pop()

    while fork_ and (fork[0] > ave * rdn):  # fork[0] = crit, latter summed in form?

        select_.appendleft(fork); rdn += 1  # inclusion if match, no neg forks?
        fork = fork_.pop()  # no: fork = rdn + alt_rdn / len(e_), _P: float adj < cost?

    init = 0 if select_ == 1 else 1
    for fork in select_:

        fork = form_blob(P, fork, rdn, init)  # P is added to fork blob
        fork_.appendleft(fork)  # not-selected forks are out of fork_

    P = s, ix, x, I, D, Dy, M, My, G, Olp, rdn, e_  # for conversion to _P

    P_.append((P, [], fork_))  # root_ initialized, _P_ = P_ for next-line scan_P_()
    buff_ += _P_  # excluding displaced _Ps

    return P_, buff_  # _P_ = buff_ for scan_P_(next P)

    # y-2: P + root_, fork_b_ <- form_P2 (to form_PP)
    # y-3: P2 + root_b_, fork_seg_b_ <- segment if root_>1 or term if root_=0
    # y-n: segment blobs + root_seg_b_, fork_seg_b_, term if root_-> term_, full term at last P


def form_blob(P, fork, rdn, init):  # P inclusion into blob (selected fork), _P is not preserved

    blob, root_, fork_ = fork  # _fork_ in y-3, not changed
    s, ix, x, I, D, Dy, M, My, G, Olp, e_ = P  # or rdn = e_ / Olp + blob_rdn

    if init:  # forming new _P

        L2 = len(e_)  # no separate e2_: Py_( P( e_? overlap / comp_P only?
        I2 = I
        D2 = D; Dy2 = Dy
        M2 = M; My2 = My
        G2 = G  # also oG: vertical contiguity for comp_P eval?
        rdn2 = rdn
        Olp2 = Olp
        Py_ = [P]  # vertical array of patterns within a blob

    else:  # extending matching _P

        L2, I2, D2, Dy2, M2, My2, G2, Olp2, rdn2, Py_ = blob

        L2 += len(e_)
        I2 += I
        D2 += D; Dy2 += Dy
        M2 += M; My2 += My
        G2 += G
        rdn2 += rdn
        Olp2 += Olp
        Py_.append(P)

    blob = s, L2, G2, I2, D2, Dy2, M2, My2, rdn2, Olp2, Py_
    root_.appendleft(P)  # adds P@ in input order, same-type roots only

    fork = blob, root_, fork_  # P is replacing _P, which was summed into blob
    return fork


def term_blob(_P):  # blob eval for comp_P, only if complete term: root_ and fork_ == 0?

    '''
    for blob in blob_:

        blob, vPP, dPP = blob  # <= one _vPP and _dPP per higher-level blob
        term_P2(blob)  # eval for 2D P re-orient and re-scan, then recursion

        if vPP: term_P2(vPP)  # if comp_P in fork_eval(blob)
        if dPP: term_P2(dPP)  # not for dPP in dPP_: only to eval for rdn?

        if fork_vP_: # from comp_P over select forks in fork_eval(blob_)

            v_rdn = b_rdn  # eval for inclusion in vPPs (2D value patterns):
            fork_vP_, v_rdn = fork_eval(0, P, fork_vP_, v_rdn, x)

            d_rdn = v_rdn  # eval for inclusion in dPPs (2D difference patterns), rdn alt_ = vPPs:
            fork_dP_, d_rdn = fork_eval(1, P, fork_dP_, d_rdn, x)

            comp_P -> vPP and dPP per select forks blob, after term: cost div, before orient?
            P = comp_P(P, fork, x)  # comp_P if form_blob? also typ for P' e_ comp?
            fork = form_PP(typ, P, fork, rdn, init)  # fork = vPP or dPP, after comp_P

            if typ: crit = Pm  # total match per pattern
            else: crit = Pd * ave_k  # projected match of difference per pattern
            if crit > ave * 5 * rdn:  # comp cost, or * number of vars per P: rep cost?
    '''

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


def form_PP(typ, P, fork, rdn, init):  # forms vPPs, dPPs, and pPs within each

    P, P_ders = P
    s, ix, x, I, D, Dy, M, My, G, Olp, e_ = P
    Pd, Pm, mx, dx, mL, dL, mI, dI, mD, dD, mM, dM, div_f, nvars = P_ders

    PP, root_, forks = fork

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
        Olp2 = Olp
        rdn2 = rdn
        Py_ = [P]

    else:  # increments current PP, if same sign?

        crit, rdn, L2, I2, D2, Dy2, M2, My2, G2, Olp2, rdn2, Py_ = PP

        L2 += len(e_); I2 += I; D2 += D; Dy2 += Dy; M2 += M; My2 += My; G2 += G
        Olp2 += Olp; rdn2 += rdn  # rdn: alt_P, alt_PP, fork, alt_pP?
        Py_.append(P)

        # mx, mL, mI, mD, mM; ddx, dL, dI, dD, dM;
        # also norm pPs?

    form_pP(dx, dxP_); form_pP(mx, mxP_)  # same form_pP?
    form_pP(len(e_), LP_)
    form_pP(I, IP_)
    form_pP(D, DP_)
    form_pP(M, MP_)

    PP = s, L2, I2, D2, Dy2, M2, My2, G2, Olp2, rdn2, Py_  # pP_s are packed in parameters
    root_.appendleft(P)  # adds P@ in input order, same-type roots only

    fork = PP, root_, forks
    return fork


def form_pP(par, pP_):  # forming parameter patterns within PP

    # a_mx = 2; a_mw = 2; a_mI = 256; a_mD = 128; a_mM = 128: feedback to define vpPs: parameter value patterns
    # a_PM = a_mx + a_mw + a_mI + a_mD + a_mM  or A * n_vars, rdn accum per pP, alt eval per vertical overlap?

    # LIDV per dx, L, I, D, M? select per term?
    # alt2_: fork_ alt_ concat, to re-compute redundancy per PP


def term_PP(PP):  # vPP | dPP eval for rotation, re-scan, re-comp, recursion, accumulation

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

