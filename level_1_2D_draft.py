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

    Level 1 performs several steps of incremental encoding, per line defined by vertical coordinate y:

    y: comp(p_): lateral comp -> tuple t,
    y- 1: ycomp(t_): vertical comp -> quadrant t2,
    y- 1: form_P(P): lateral combination -> 1D pattern P,  
    y- 2: scan_P_(P, _P) -> fork_, root_: vertical continuity between 1D Ps of adjacent lines 
    y- 3+: form_blob: merges y-2 P into 2D blob
    y- 3+: term_blob: terminated blobs are evaluated for comp_P and form_PP, -> 2D patterns PPs,
           PPs are evaluated for blob re-orientation, re-scan, consolidation and comparison 

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

        if len(it_) == rng:  # or while x < rng: icomp(){ p = pop(p_).., no t_.append?
            t_.append((pri_p, fd, fm))  # completed tuple is transferred from it_ to t_

        it_.appendleft((p, 0, 0))  # new prior tuple, fd and fm are initialized at 0

    t_ += it_  # last number = rng of tuples remain incomplete
    return t_


def ycomp(t_, t2__, _vP_, _dP_):  # vertical comparison between pixels, forms 2D tuples t2

    vP_ = []; vP = [0,0,0,0,0,0,0,0,[]]  # pri_s, I, D, Dy, M, My, G, Olp, e_
    dP_ = []; dP = [0,0,0,0,0,0,0,0,[]]  # pri_s, I, D, Dy, M, My, G, Olp, e_

    vg_blob_, dg_blob_ = [],[]  # output line of blobs, vertical concat -> frame in frame()

    x = 0; new_t2__ = []   # t2_ buffer: 2D array
    olp, ovG, odG = 0,0,0  # len of overlap between vP and dP, gs summed over olp, all shared

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

        if len(t2_) == rng:  # or while y < rng: i_ycomp(): t2_ = pop(t2__), t = pop(t_)., no form_P?

            dg = _d + fdy  # d gradient
            vg = _m + fmy - ave  # v gradient
            t2 = pri_p, _d, fdy, _m, fmy  # completed 2D tuple moved from t2_ to form_P:

            # form 1D patterns vP and dP: horizontal spans of same-sign vg or dg, with associated vars:

            olp, ovG, odG, vP, dP, vP_, _vP_, vg_blob_ = \
            form_P(1, t2, vg, dg, olp, ovG, odG, vP, dP, vP_, _vP_, vg_blob_, x)

            olp, odG, ovG, dP, vP, dP_, _dP_, dg_blob_ = \
            form_P(0, t2, dg, vg, olp, odG, ovG, dP, vP, dP_, _dP_, dg_blob_, x)

        t2_.appendleft((p, d, 0, m, 0))  # initial fdy and fmy = 0, new t2 replaces completed t2 in t2_
        new_t2__.append(t2_)
        
    # line ends, vP and dP term, no init, inclusion with incomplete lateral fd and fm:

    if olp:  # if vP x dP overlap len > 0, incomplete vg - ave / (rng / X-x)?

        odG *= ave_k; odG = odG.astype(int)  # ave_k = V / I, to project V of odG

        if ovG > odG:  # comp of olp vG and olp dG, == goes to vP: secondary pattern?
            dP[7] += olp  # overlap of lesser-oG vP or dP, or P = P, Olp?
        else:
            vP[7] += olp  # to form rel_rdn = alt_rdn / len(e_)

    if y + 1 > rng:  # starting with the first line of complete t2s

        vP_, _vP_, vg_blob_ = scan_P_(0, vP, vP_, _vP_, vg_blob_, x)  # returns empty _vP_
        dP_, _dP_, dg_blob_ = scan_P_(1, dP, dP_, _dP_, dg_blob_, x)  # returns empty _dP_

    return new_t2__, _vP_, _dP_, vg_blob_, dg_blob_  # extended in scan_P_

    # poss alt_: top P alt = Olp, oG, alt_oG: to remove if hLe demotion and alt_oG < oG?
    # P_ can be redefined as np.array ([P, alt_, roots, forks) to increment without init?


def form_P(typ, t2, g, alt_g, olp, oG, alt_oG, P, alt_P, P_, _P_, blob_, x):

    # forms 1D dP or vP, then scan_P_ adds forks in _P fork_s and accumulates blob_

    p, d, dy, m, my = t2  # 2D tuple of quadrant variables per pixel
    pri_s, I, D, Dy, M, My, G, Olp, e_ = P  # initial pri_ vars = 0, or skip form?

    s = 1 if g > 0 else 0  # g = 0 is negative: no selection?
    if s != pri_s and x > rng + 2:  # P is terminated

        if typ:
            alt_oG *= ave_k; alt_oG = alt_oG.astype(int)  # ave V / I, to project V of odG
        else:
            oG *= ave_k; oG = oG.astype(int)  # same for h_der and h_comp eval?

        if oG > alt_oG:  # comp between overlapping vG and dG
            Olp += olp  # olp is assigned to the weaker of P | alt_P, == -> P: local access
        else:
            alt_P[7] += olp

        P = (pri_s, I, D, Dy, M, My, G, Olp, e_), [], []  # no ave * alt_rdn / e_: adj < cost?
        P_, _P_, blob_ = scan_P_(typ, P, P_, _P_, blob_, x)  # scan over contiguous higher-level _Ps

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
    return olp, oG, alt_oG, P, alt_P, P_, _P_, blob_  # accumulated in ycomp


def scan_P_(typ, P, P_, _P_, blob_, x):  # P scans overlapping _Ps in _P_, forms overlapping Gs

    buff_ = []
    (s, I, D, Dy, M, My, G, Olp, e_), root_, alt_root_  = P  # roots are to find unique fork Ps

    ix = x - len(e_)  # initial x of P
    _ix = 0  # initialized ix of _P displaced from _P_ by last scan_P_

    while x >= _ix:  # P to _P match eval, while horizontal overlap between P and _P_:

        ex = x  # ex is lateral coordinate of loaded P element
        oG = 0  # fork gradient overlap: oG += g (distinct from alt_P oG)
        _P, blob, fork_, alt_fork_ = _P_.popleft()  # _P in y-2, blob in y-3, forks in y-1

        if s == _P[0]:  # if s == _s: vg or dg sign match, fork_.append eval

            while ex > _P[1]: # ex > _ix
                for e in e_:  # accumulation of oG between P and _P:

                    if typ: oG += e[1]  # if vP: e = p, g, alt_g
                    else: oG += e  # if dP: e = g
                    ex += 1  # odG is adjusted: *= ave_k in form_P

            if oG > ave * 16: # if mult _P: cost of fork and blob in term_blob, unless fork_==1

                root_.append((oG, _P)) # _Ps connected to P, term if root_!= 1
                fork_.append((oG, P))  # Ps connected to _P, term if fork_!= 1

            elif oG > ave * 4: # if one _P: > cost of summation in form_blob, unless root_!=1?

                alt_root_.append((oG, _P)) # _Ps connected to P, select root_.append at P output
                alt_fork_.append((oG, P))  # Ps connected to _P, select fork_.append at P output

        if _P[2] > ix:  # if _x > ix:
            buff_.append(_P)  # _P is buffered for scan_P_(next P)

        elif fork_== 0 and alt_fork_== 0: # no overlap between _P and next P, and blob term

            blob = incr_blob((oG, _P), blob)  # default _P incl, empty init at final P root_!= 1

            if blob[8] > ave * 9 and blob[10] > 2:  # blob value OG > cost: ave * 9 vars | ave_OG?
               blob = scan_blob(typ, blob)

            blob_.append((blob, fork_))  # terminated blobs: input line y - 3+, record layer 5+
            # no term delay unless _P is buffered in fork Ps root_| alt_root_, waiting for P output:

    # no overlap between P and next _P: delayed blob += _P for root_ of P if fork_ != 0

    if root_== 1 and root_[0][3] > 1:  # select single alt fork for single root, else split

        root_= [max(alt_root_)]  # same as root = max(alt_root_, key= lambda alt_root: alt_root[0])
        root_[0][1].append((root_[0][0][0], P))  # _P(oG, P) is added to fork_ of max root _P

    for _P, blob, fork_, alt_fork_ in root_:  # final fork assignment and blob increment per _P

        blob = incr_blob(_P, blob)  # default per root, blob is modified in root _P?

        if fork_ == 1 and alt_root_ > 1:  # select single max root for single fork, else merge

            fork_ = [max(alt_fork_)]
            fork_[0][2].append(_P)  # _P(oG, _P) is added to root_ of max fork P

        if fork_!= 1 or root_!= 1:  # split | merge / current count, also if y == Y - 1 in frame()?

            if blob[8] > ave * 9 and blob[10] > 2:  # if OG > Ave and Py_ > 2, cost: comp, PP, PP_?
               blob = scan_blob(typ, blob)  # top-down fork_, no root_: rdn

            blob_.append((blob, fork_))  # terminated blob_ is input line y - 3+ | record layer 5+

    if root_== 1 and root_[0][3] == 1:  # blob assign if final P' root_==1 & root' fork_==1
        blob = root_[0][1]  # root' fork' blob
    else:
        blob = (0,0,0,0,0,0,0,0,0,0,[]) # init s, L2, I2, D2, Dy2, M2, My2, G2, OG, Olp2, Py_[]

    P = s, ix, x, I, D, Dy, M, My, G, Olp, e_  # P becomes _P, oG is per new P in fork_?

    P_.append((P, blob, [], []))  # blob assign, forks init, _P_ = P_ for next-line scan_P_()
    buff_ += _P_  # excluding displaced _Ps

    return P_, buff_, blob_  # _P_ = buff_ for scan_P_(next P)

''' sequential displacement and higher-layer (L) inclusion at record layer's end:

    y, 1st 1L: p_ -> t_
    y- 1,  2L: t_, t2_ -> P_
    y- 2,  3L: P_, _P_ -> fork_ between _Ps and Ps
    y- 3+, 4L: fork_, blob_: blob segments of variable depth 
    y- 3+, 5+: blob_, term_: layers of terminated segments, global term if root_==0, same OG eval? '''


def incr_blob(_P, blob):  # continued or initialized blob is incremented by attached _P

    s, L2, I2, D2, Dy2, M2, My2, G2, OG, Olp2, Py_ = blob
    oG, _P = _P  # oG was buffered in root_| fork_
    s, ix, x, I, D, Dy, M, My, G, Olp, e_ = _P  # s is re-assigned

    L2 += len(e_)  # no separate e2_: Py_( P( e_, overlap / comp_P only
    I2 += I
    D2 += D; Dy2 += Dy
    M2 += M; My2 += My
    G2 += G
    OG += oG  # vertical contiguity for comp_P eval?
    Olp2 += Olp
    Py_.append((s, ix, x, I, D, Dy, M, My, G, oG, Olp, e_))

    blob = s, L2, I2, D2, Dy2, M2, My2, G2, OG, Olp2, Py_
    return blob # _P summed into blob


def scan_blob(typ, blob):  # vertical scan of Ps in Py_ for comp_P, incr_PP, form_pP_?

    vPP = 0,[],[],[],[],[],[],[],[],[],[]  # s, L2, I2, D2, Dy2, M2, My2, G2, Olp2, Py_
    dPP = 0,[],[],[],[],[],[],[],[],[],[]  # with pP_ per parameter
    vPP_, dPP_ = [],[]  # 2D value P and difference P
    vS_ders = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    dS_ders = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    Py_ = blob[10]
    _P = Py_.popleft()  # initial comparand

    while Py_: # comp_P starts from 2nd P, top-down

        P = Py_.popleft()
        _P, _vs, _ds = comp_P(typ, P, _P)  # per blob, before orient

        while Py_:  # form_PP starts from 3rd P

            P = Py_.popleft()
            P, vs, ds = comp_P(typ, P, _P)

            if vs ==_vs:
                vPP, vS_ders = incr_PP(typ, P, vPP, vS_ders)  # no delay
            else:
                vPP_.append(vPP)  # or PP eval for comp_pP?
                vPP = 0,[],[],[],[],[],[],[],[],[],[]

                for p in vS_ders:  # form_pP select per S_der
                    p, p_ = p
                    if p > ave * 9 * 5 * 2:  # Vp > ave PP * ave pP rdn * rdn to PP
                        pP_= []
                        pP_= form_pP_(typ, p, p_, pP_)

                vS_ders = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

            if ds ==_ds:
                dPP, dS_ders = incr_PP(typ, P, dPP, dS_ders)
            else:
                dPP_.append(dPP)
                dPP = 0,[],[],[],[],[],[],[],[],[],[]
                dS_ders = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

            _P = P; _vs = vs; _ds = ds

    # if PP_ > ave: sum_PP += PP?
    # eval for comp_PP, reorient, rescan, recursion:

    blob = orient(blob, vPP_, dPP_)

    return blob  # blob or PP_, comp_P may continue over fork_, after comp_segment?


def comp_P(typ, P, _P):  # forms vertical derivatives of P vars, also conditional ders from DIV comp

    s, ix, x, I, D, Dy, M, My, G, oG, Olp, e_ = P
    _s, _ix, _x, _I, _D, _Dy, _M, _My, _G, _oG, _Olp, _e_ = _P

    ddx = 0  # optional, 2Le norm / D? s_ddx and s_dL correlate, s_dx position and s_dL dimension don't?
    ix = x - len(e_)  # initial coordinate of P; S is generic for summed vars I, D, M;

    dx = x - len(e_)/2 - _x - len(_e_)/2  # Dx? comp(dx), ddx = Ddx / h? dS *= cos(ddx), mS /= cos(ddx)?
    mx = x - _ix; if ix > _ix: mx -= ix - _ix  # mx = x olp, - a_mx -> vxP, distant P mx = -(a_dx - dx)?

    dL = len(e_) - len(_e_); mL = min(len(e_), len(_e_))  # relative olp = mx / L? ext_miss: Ddx + DL?
    dI = I - _I; mI = min(I, _I)
    dD = D - _D; mD = min(D, _D)
    dM = M - _M; mM = min(M, _M)

    # lat S of y_ders also indicates P match and orientation, even though y_ders are incomplete?
    # oG in Pm | Pd: ~ e_ match | overlap (mx), but lat + vert?  no G comp: redundant to ders

    Pd = ddx + dL + dI + dD + dM  # defines dPP; var_P form if PP form, term if var_P or PP term;
    Pm = mx + mL + mI + mD + mM  # defines vPP; comb rep value = Pm * 2 + Pd?

    if dI * dL > div_a: # DIV comp: cross-scale d, neg if cross-sign, nS = S * rL, ~ rS,rP: L defines P

        # no ndx: single, yes nmx: summed?

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

    vs = 1 if Pm > ave * 5 > 0 else 0  # comp cost = ave * 5, or rep cost: n vars per P?
    ds = 1 if Pd > 0 else 0

    P_ders = Pm, Pd, mx, dx, mL, dL, mI, dI, mD, dD, mM, dM, div_f, nvars
    P = P, P_ders

    return P, vs, ds  # for inclusion in vPP_, dPP_ by form_PP:

''' 
    no DIV comp(L): match is insignificant and redundant to mS, mLPs and dLPs only?:

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
    iter dS - S -> (n, M, diff): var precision or modulo + remainder? '''


def incr_PP(typ, P, PP, S_ders):  # increments continued vPPs or dPPs, not pPs within each

    P, P_ders = P
    s, ix, x, I, D, Dy, M, My, G, oG, Olp, e_ = P
    L2, I2, D2, Dy2, M2, My2, G2, OG, Olp2, Py_ = PP

    L2 += len(e_)
    I2 += I
    D2 += D; Dy2 += Dy
    M2 += M; My2 += My
    G2 += G
    OG += oG
    Olp2 += Olp
    Py_.appendleft((P, P_ders))

    Pm, Pd, mx, dx, mL, dL, mI, dI, mD, dD, mM, dM, div_f, nvars = P_ders
    # or ddx: predicts dS?  added eval for nvars pPs, or before div_comp?

    Mx, Dx, ML, DL, MI, DI, MD, DD, MM, DM = S_ders  # only criterion ps comp?
    Mx += mx; Dx += dx; ML += mL; DL += dL; ML += mI; DL += dI; MD += mD; DD += dD; MM += mM; DM += dM

    # comp_P eval by blob OG: default form,
    # form_pP eval by PP' |Vp| or |Dp|: + ave rdn = 5 (2.5 * 2), or cum rdn for !max ps?

    PP = s, L2, I2, D2, Dy2, M2, My2, G2, Olp2, Py_

    return PP, S_ders

    # fork_ per blob, rdn comp_P(_P, P_) between blobs?
    # rdn: alt_P, alt_PP, comp fork, alt_pP?

''' 
        if typ: alt_oG *= ave_k; alt_oG = alt_oG.astype(int)  # ave V / I, to project V of odG
        else: oG *= ave_k; oG = oG.astype(int)               # same for h_der and h_comp eval?

        if oG > alt_oG:  # comp between overlapping vG and dG
            Olp += olp  # olp is assigned to the weaker of P | alt_P, == -> P: local access
        else:
            alt_P[7] += olp 

        olp, oG, alt_oG = 0,0,0
        P_, _P_, blob_ = scan_P_(typ, P, P_, _P_, blob_, x)  # no scan over hLe _Ps 

        olp += 1  # len of overlap to stronger alt-type P, accumulated until P or _P terminates
        G += g; alt_oG += alt_g  # for eval to assign olp to alt_rdn of vP or dP 

        after orient:

        if typ: e_.append((P, Pm, Pd))  # selective incremental-range comp_P, if min Py_? 
        else: e_.append(P)  # selective incremental-derivation comp_P? '''


def form_pP_(typ, par, par_, pP_):  # forming parameter patterns within PP

    vps = 1 if mp > ave * 5 > 0 else 0  # comp cost = ave * 5, or rep cost: n vars per P?
    dps = 1 if dp > 0 else 0

    if vps == _vps:
        vpP = incr_pP(typ, p, P_ders, vpP)  # no delay
    else:
        vpP_.append(vpP)  # comp_pP eval per PP?
        vpP = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []

    incr_pP(typ, P_ders, dx, dxP)
    incr_pP(typ, P_ders, mx, mxP)

    incr_pP(typ, P_ders, len(e_), LP)
    incr_pP(typ, P_ders, I, IP)
    incr_pP(typ, P_ders, D, DP)
    incr_pP(typ, P_ders, M, MP)

    return pP_

    # scan_pP_ eval at PP term in scan_blob?
    # a_mx = 2; a_mw = 2; a_mI = 256; a_mD = 128; a_mM = 128: feedback to define vpPs: parameter value patterns
    # a_PM = a_mx + a_mw + a_mI + a_mD + a_mM  or A * n_vars, rdn accum per pP, alt eval per vertical overlap?

    # LIDV per dx, L, I, D, M? select per term?
    # alt2_: fork_ alt_ concat, to re-compute redundancy per PP

def scan_pP_():
    P2 = []
    return P2

def comp_pP():
    P2 = []
    return P2

def orient(blob, PP_, alt_PP_):  # vPP | dPP eval for blob | PP rotation, re-scan, re-comp, recursion, accumulation
    P2 = []
    return P2

def comp_PP():
    P2 = []
    return P2

def comp_blob():
    P2 = []
    return P2

''' 
    Orient = m_dx, value of re-orient = m(ddx, (dL, dS)): accumulated locally?
    dimensionally reduced axis: vP PP or contour: dP PP; dxP is direction pattern

    blob_-> term_, forming layers 5+, regardless of line, sum into wider fork, also sum per frame?
    layer 3 is cont blobs, single ref to layer 2 _Ps
    
    vPP or dPP rdn assign at blob eval?
    
    PP = 0,0,0,0,0,0,0,0,0,0,[],[]  # initialized at non-matching P transfer to _P_?
    np.array for direct accumulation, vs. iterator of initialization?:

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


def frame(f):  # postfix '_' denotes array vs. element, prefix '_' denotes higher-line variable

    global ave; ave = 127  # filters, ultimately set by separate feedback, then ave *= rng
    global rng; rng = 1

    global div_a; div_a = 127  # not justified
    global ave_k; ave_k = 0.25  # average V / I initialization

    global Y; global X; Y, X = f.shape  # Y: frame height, X: frame width
    global y; y = 0

    _vP_, _dP_, frame_ = [], [], []

    t2_ = deque(maxlen=rng)  # vertical buffer of incomplete pixel tuples, for fuzzy ycomp
    t2__ = []  # vertical buffer + horizontal line: 2D array of 2D tuples, deque for speed?
    p_ = f[0, :]  # first line of pixels
    t_ = comp(p_)  # after part_comp (pop, no t_.append) while x < rng?

    for t in t_:
        p, d, m = t
        t2 = p, d, 0, m, 0  # fdy and fmy initialized at 0
        t2_.append(t2)  # only one tuple per first-line t2_
        t2__.append(t2_)  # in same order as t_

    # part_ycomp (pop, no form_P) while y < rng?

    for y in range(1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?

        p_ = f[y, :]  # vertical coordinate y is index of new line p_
        t_ = comp(p_)  # lateral pixel comparison
        t2__, _vP_, _dP_, vg_blob_, dg_blob_ = ycomp(t_, t2__, _vP_, _dP_) # vertical pixel comp

        frame_.append((vg_blob_, dg_blob_))  # line of blobs is added to frame of blobs

    return frame_  # frame of 2D patterns is outputted to level 2

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
frame(f)

