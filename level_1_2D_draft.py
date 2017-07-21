from scipy import misc
from collections import deque

'''
    Level 1 with patterns defined by the sign of vertex gradient: modified core algorithm of levels 1 + 2.

    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match. 
    Minimal and unique unit of 2D gradient is a vertex of rightward and downward derivatives per pixel.

    Vertex gradient is computed as an average of these two equally representative sample derivatives. 
    2D patterns are blobs of same-sign vertex gradient, of value for vP or difference for dP.
    Level 1 has 4 steps of encoding, incremental per line defined by coordinate y:

    y:   comp()    p_ array of pixels, lateral comp -> p,m,d,
    y-1: ycomp()   t_ array of tuples, vertical comp, der.comb -> 1D P,
    y-2: comp_P()  P_ array of 1D patterns, vertical comp, eval, comb -> PP ) CP
    y-3: cons_P2() P2_ array of 2D patterns, fork overlap, eval, PP or CP consolidation:
'''

def comp(p_):  # comparison of consecutive pixels in a scan line forms tuples: pixel, match, difference

    t_ = []
    pri_p = p_[0]  # no d, m at x=0, lagging t_.append(t)

    for p in p_:  # compares laterally consecutive pixels, vs. for x in range(1, X)

        d = p - pri_p  # difference between consecutive pixels
        m = min(p, pri_p)  # match between consecutive pixels
        t = pri_p, d, m
        t_.append(t)
        pri_p = p

    t = pri_p, 0, 0; t_.append(t)  # last pixel is not compared
    return t_


def ycomp(t_, _t_, fd, fv, y, Y, r, a, _vP_, _dP_):

    # vertical comparison between pixels, forms vertex tuples t2: p, d, dy, m, my, separate fd, fv
    # last "_" denotes array vs. element, first "_" denotes higher-line array, pattern, or variable

    x, valt_, dalt_, vP_, dP_, term_vP_, term_dP_ = 0,[],[],[],[],[],[]  # term_P_: terminated _Ps
    pri_s, I, D, Dy, M, My, G, olp, e_ = 0,0,0,0,0,0,0,0,[]  # also _G: interference | redundancy?
    vP = pri_s, I, D, Dy, M, My, G, olp, e_
    dP = pri_s, I, D, Dy, M, My, G, olp, e_  # alt_ included at term, rdn from alt_ eval in form_PP?

    A = a * r

    for t, _t in zip(t_, _t_):  # compares vertically consecutive pixels, forms gradients

        x += 1
        p, d, m = t
        _p, _d, _m = _t

        dy = p - _p   # vertical difference between pixels, summed -> Dy
        dg = _d + dy  # gradient of difference, formed at prior-line pixel _p, -> dG: variation eval?
        fd += dg      # all shorter + current- range dg s within extended quadrant

        my = min(p, _p)   # vertical match between pixels, summed -> My
        vg = _m + my - A  # gradient of predictive value (relative match) at prior-line _p, -> vG
        fv += vg          # all shorter + current- range vg s within extended quadrant

        t2 = p, d, dy, m, my  # fd, fv -> type-specific g, _g; all are accumulated within P:

        # forms 1D slice of value pattern vP: horizontal span of same-sign vg s with associated vars:

        sv, valt_, dalt_, vP, vP_, _vP_, term_vP_ = \
        form_P(0, t2, fv, fd, valt_, dalt_, vP, vP_, _vP_, term_vP_, x, y, Y, r, A)

        # forms 1D slice of difference pattern dP: horizontal span of same-sign dg s with associated vars:

        sd, dalt_, valt_, dP, dP_, _dP_, term_dP_ = \
        form_P(1, t2, fd, fv, dalt_, valt_, dP, dP_, _dP_, term_dP_, x, y, Y, r, A)

    # line's end, last ycomp t: lateral d = 0, m = 0, inclusion per incomplete gradient?
    # vP, dP term, no initialization:

    dolp = dP[7]; dalt = len(vP_), dolp; dalt_.append(dalt)
    olp = vP[7]; valt = len(dP_), olp; valt_.append(valt)

    vP_, _vP_, term_vP_ = comp_P(valt_, vP, vP_, _vP_, term_vP_, x, y, Y, r, A)  # empty _vP_?
    dP_, _dP_, term_dP_ = comp_P(dalt_, dP, dP_, _dP_, term_dP_, x, y, Y, r, A)  # empty _dP_?

    return vP_, dP_, term_vP_, term_dP_  # with refs to vPPs, dPPs, vCPs, dCPs from comp_P, adjusted by cons_P2


def form_P(type, t2, g, _g, alt_, _alt_, P, P_, _P_, term_P_, x, y, Y, r, A):  # forms 1D slices of 2D patterns

    p, d, dy, m, my = t2
    pri_s, I, D, Dy, M, My, G, olp, e_ = P  # unpacked to increment or initialize vars, +_G to eval alt_P rdn?

    s = 1 if g > 0 else 0
    if s != pri_s and x > r + 2:  # P (span of same-sign gs) is terminated and compared to overlapping _Ps:

        P_, _P_, term_P_ = comp_P(alt_, P, P_, _P_, term_P_, x, y, Y, r, A)  # P_ becomes _P_ at line end
        _alt = len(P_), olp # index len(P_) and overlap of P are buffered in _P' _alt_:
        _alt_.append(_alt)
        I, D, Dy, M, My, G, olp, e_, alt_ = 0,0,0,0,0,0,0,[],[]  # initialized P and alt_

    # continued or initialized P vars are accumulated:

    olp += 1  # P overlap to concurrent alternative-type P, accumulated till either P or _P is terminated
    I += p    # p s summed within P
    D += d; Dy += dy  # lat D for vertical vP comp, + vert Dy for P2 orient adjust eval and gradient
    M += m; My += my  # lateral and vertical M for P2 orient, vs V gradient eval, V = M - 2a * W?
    G += g  # fd | fv summed to define P value, with directional resolution loss

    if type == 0:
        pri = p, g, _g  # also d, dy, m, my, for fuzzy accumulation within P-specific r?
        e_.append(pri)  # prior same-line vertex, buffered for selective inc_rng comp
    else:
        e_.append(g)  # prior same-line difference gradient, buffered for inc_der comp

    P = s, I, D, Dy, M, My, G, olp, e_

    return s, alt_, _alt_, P, P_, _P_, term_P_

    # draft below:

def comp_P(alt_, P, P_, _P_, term_P_, x, y, Y, r, A):  # _x: x of _P displaced from _P_ by last comb_P

    # vertical comparison between 1D slices, for selective inclusion in 2D patterns vPP and dPP?

    buff_, CP_, = deque(), deque()
    root_, _fork_, Fork_ = deque(), deque(), deque()  # olp root_: same-sign higher _Ps, fork_: same-sign lower Ps

    ddx = 0; nvar = 1  # same nvar for dPP and vPP: var comp -> d_vars and m_vars, depth incr if PM + PD?
    # or full comp till min number of lower-D | higher-d: -> S? var_P form within PP only, redundant?

    _x = 0  # coordinate of _P
    _n = 0  # index of _P, for addressing root Ps in root_

    W, I2, D2, Dy2, M2, My2, G2, rdn2, alt2_, Py_ = 0,0,0,0,0,0,0,0,[],[]  # PP vars (pattern of patterns) per fork
    WC, IC, DC, DyC, MC, MyC, GC, rdnC, altC_, PP_ = 0,0,0,0,0,0,0,0,[],[]  # CP vars (connected PPs) at first Fork

    a_mx = 2; a_mw = 2; a_mI = 256; a_mD = 128; a_mM = 128  # feedback to define var_vPs (variable value patterns)
    a_PM = 512  # or sum of a_m_vars? rdn accum per var_P, alt eval per vertical overlap?

    s, I, D, Dy, M, My, G, e_ = P  # also alt_, root_: doesn't need to be returned?
    w = len(e_); ix = x - w  # w: P width, ix: P initial coordinate

    while x >= _x:  # P is compared to next _P in _P_ while there remains some horizontal overlap between them

        _P = _P_.popleft(); _n += 1  # _n is _P counter to sync Fork_ with _P_, or len(P_) - len(_P_)?
        _s, _ix, _x, _w, _I, _D, _Dy, _M, _My, _G, _r, _e_, _rdn, _alt_, _root_ = _P

        if s == _s:  # P var comp -> var_dP and var_vP, to always eval for internal and external comp?
            ''' 
            PP def per combined-var vertical der sign, lower-vars comp while minimal higher-vars M or |D|
            conditional next-var comp -> var_P?, nvar ++, M|D += m|d *= rdn to prior vars: 
            
            external x ( higher dim w ( lower der I ( lower res D,M ( e_., vs. per P?
            if value of var eval -> var_Ps: smaller than or redundant to pixels? 
            
            incr Le group: L, L D V, IDV DDV VDV: new D, d_ and V, p_ per input variable, higher-Le comp priority? or
            all- Le comp, cross-derivative comp, eval of higher-power re-comp and higher-derivation specification?
            
            proximity (pri neg) vs. match (pri pos) relative value adjusts coordinate vs input resolution?
            var_Ps for spec in PP? recursion per var_P2 | PP? mx and wx are subsets of w, trigger S(summed) vars comp?
            
            '''

            dx = x - w/2 - _x - _w/2  # form_P(dxP), Dx > ave? comp(dx), ddx = Ddx / h? dS *= cos(ddx), mS /= cos(ddx)?
            mx = x - _ix; if ix > _ix: mx -= ix - _ix  # mx - a_mx -> form_P(vxP), vs. mx = -(a_dx - dx): if discont.?

            dw = w - _w  # -> dwP, var_P or PP' Ddx + Dw (higher-Dim D) triggers adjustment of derivatives or _vars?
            mw = min(w, _w)  # -> vwP, mx + mw (higher-Dim m) triggers comp(S | aS(norm to assign redun))? or full comb?

            dI = I - _I; mI = min(I, _I)  # eval of MI vs. Mh rdn at term PP | var_P, not per slice?
            dD = D - _D; mD = min(D, _D)
            dM = M - _M; mM = min(M, _M)  # no G comp: y-derivatives are incomplete. len(alt_) comp?

            PD = ddx + dw + dI + dD + dM  # defines dPP, or per ddx + dw: signs correlate, dx (direction) and dw (dimension) signs don't?
            PM = mx + mw + mI + mD + mM  # defines vPP, or per mx + mw, proximity value = mx / w, not redundant to mw: similarity?

            # if PM * 2(rep value) + PD > A: combined spec eval?

            P = PM, PD, x, mx, dx, w, mw, dw, I, mI, dI, D, mD, dD, M, mM, dM, Dy, My, G, e_

            root_.append(P)  # root temporarily includes current P and its P comp derivatives, as well as _P and PP

        rdn = 0  # redundancy (recreated vs. stored): number of stronger-PM root Ps in root_ + alt Ps in alt_
                 # vs.vars *= overlap ratio: prohibitive cost?  average redundancy: no ind.var eval, only per var_P?

        while len(root_) > 0:  # rdn assignment within root_, separate for vPP and dPP?

            root = root_.pop(); PM = root[0]  # PM (P match: sum of var matches between Ps) is first variable of root P

            for i in range(len(root_)):  # remaining roots are reused by while len(root_)

                _root = root_[i]; _PM = _root[0]  # lateral PM comp, neg v count -> rdn for PP inclusion eval:
                if PM > _PM: _root[1] += 1; root_[i] = _root  # root_rdn increment, separate alt_rdn?
                else: rdn += 1  # redundancy = len(stronger_root_) + len(stronger_alt_):

        for i in range(len(alt_)):  # refs within alt_P_, dP vs. vP PM comp, P rdn coef = neg v Olp / w?

            ialt_P = alt_[_alt_ + i]; alt_P = alt_[ialt_P]; _PM = alt_P[0]  # _alt_P_ + i: composite address of _P?
            if PM > _PM: alt_[ialt_P[1]] += 1; alt_[_alt_ + i] = ialt_P  # alt_P rdn increment???
            else: rdn += 1

        # combined P match (PM) eval for P inclusion into PP, and then all connected PPs into CP
        # also form_dPP per PD, eval for internal and external comp? spec eval?

        if PM > A * nvar * rdn:  # P inclusion by combined match, unique representation tracing through max PM PPs?

            W +=_w; I2 +=_I; D2 +=_D; Dy2 +=_Dy; M2 +=_M; My2 +=_My; G2 += G; alt2_ += alt_, P_.append(_P)  # PP vars
            PP = W, I2, D2, Dy2, M2, My2, G2, alt2_, P_  # Alt_: root_ alt_ concat, to re-compute redundancy per PP

            root = len(_P_), PP; root_.append(root)  # _P index and PP per root, possibly multiple roots per P
            _fork_.appendleft(_n)  # index of connected P in future term_P_, to be buffered in Fork_ of CP

        if _x <= ix:  # _P and attached PP output if no horizontal overlap between _P and next P:

            PP = W, I2, D2, Dy2, M2, My2, G2, alt2_, Py_  # PP per _root P in _root_
            Fork_ += _fork_  # all continuing _Ps of CP, referenced from its first fork _P: CP flag per _P?

            if (len(_fork_) == 0 and y > r + 3) or y == Y - 1:  # no continuation per _P, term of PP, accum of CP:

                cons_P2(PP)  # separate for vPP: summed var_vs, and dPP: summed var_ds?
                # PP eval for rotation, re-scan, re-comp, recursion, accumulation per _root PP, _root_ rdn adjust, eval?

                WC += W; IC += I2; DC += D2; DyC += Dy2; MC += M2; MyC += My2; GC += G2; altC_ += alt2_; PP_.append(PP)

            else:
                _P = _s, _ix, _x, _w, _I, _D, _Dy, _M, _My, _G, _r, _e_, _alt_, _fork_, _root_  # PP index per root
                # old _root_, new _fork_ (old _fork_ is displaced with old _P_?)
                buff_.appendleft(_P)  # _P is re-inputted for next-P comp

            CP = WC, IC, DC, DyC, MC, MyC, GC, altC_, PP_, Fork_

            if (len(Fork_) == 0 and y > r + 3) or y == Y - 1:  # no continuation per CP:

                cons_P2(CP)  # eval for rotation, re-scan, cross-comp of P2_? also sum per frame?

            elif len(_P_) == last_Fork_nP:  # CP_ to _P_ sync for PP inclusion and cons(CP) trigger by Fork_' last _P?

                CP_.append(CP)  # PP may include len(CP_): CP index

            Py_.append(P)  # vertical inclusion, per P per root?

        P = s, w, I, D, Dy, M, My, G, r, e_, alt_, root_  # each root is new, includes P2 if unique cont:
        P_.append(P)  # _P_ = P_ for next-line comp, if no horizontal overlap between P and next _P

        _P_ += buff_  # first to pop() in _P_ for next-P comb_P()

    return P_, _P_, term_P_  # _P_ and term_P_ include _P and ref PP, root_ is accumulated within comp_P
    

def cons_P2(P2):  # sub-level 4: eval for rotation, re-scan, re-comp, recursion, accumulation, at PP or CP term

    ''' 
    :param P2: 
    :return: 
    ''''''
    cons_P2(PP): eval of d,m adjust | _var adjust | x,y adjust if projected dS-, mS+ for min.1D Ps over max.2D

        if dw sign == ddx sign and min(dw, ddx) > a: _S /= cos (ddx)  # to angle-normalize S vars for comp

    if dw > a: div_comp (w): rw = w / _w, to width-normalize S vars for comp: 

        if rw > a: pn = I/w; dn = D/w; vn = V/w; 

            comp (_n) # or default norm for redun assign, but comp (S) if low rw?

            if d_n > a: div_comp (_n) -> r_n # or if d_n * rw > a: combined div_comp eval: ext, int co-variance?

        comp Dy and My, /=cos at PP term?  default div and overlap eval per PP? not per CP: sparse coverage?
        
    rrdn = 1 + rdn_w / len(e_)  # redundancy rate / w, -> P Sum value, orthogonal but predictive
    S = 1 if abs(D) + V + a * len(e_) > rrdn * aS else 0  # rep M = a*w, bi v!V, rdn I?
    '''

    mean_dx = 1  # fractional?
    dx = Dx / H
    if dx > a: comp(abs(dx))  # or if dxP Dx: fixed ddx cost?  comp of same-sign dx only

    vx = mean_dx - dx  # normalized compression of distance: min. cost decrease, not min. benefit?


def Le1(f):  # last "_" denotes array vs. element, first "_" denotes higher-line array, pattern, variable

    r = 1; a = 127  # feedback filters
    Y, X = f.shape  # Y: frame height, X: frame width
    fd, fv, y, _vP_, _dP_, term_vP_, term_dP_, F_ = 0,0,0,[],[],[],[],[]

    p_ = f[0, :]  # y is index of new line p_
    _t_= comp(p_)  # _t_ includes ycomp() results, with Dy, My, dG, vG initialized at 0

    for y in range(1, Y):

        p_ = f[y, :]
        t_ = comp(p_)  # lateral pixel comp, then vertical pixel comp:
        _vP_, _dP_, term_vP_, term_dP_ = ycomp(t_, _t_, fd, fv, y, Y, r, a, _vP_, _dP_)
        _t_ = t_

        PP_ = term_vP_, term_dP_  # PP term by comp_P, adjust by cons_P2, after P ) PP ) CP termination
        F_.append(PP_)  # line of patterns is added to frame of patterns, y = len(F_)

    return F_  # output to level 2

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
Le1(f)

