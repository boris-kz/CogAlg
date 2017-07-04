from scipy import misc
from collections import deque

'''
    Level 1 with patterns defined by the sign of quadrant gradient: modified core algorithm of levels 1 + 2.

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
    pri_p = p_.poplelf()  # no d, m at x=0, lagging t_.append(t)

    for p in p_:  # new pixel, comp to prior pixel, x += len(e_) at term? vs. for x in range(1, X)

        d = p - pri_p  # difference between consecutive pixels
        m = min(p, pri_p)  # match between consecutive pixels
        t = pri_p, d, m
        t_.append(t)
        pri_p = p

    t = pri_p, 0, 0; t_.append(t)  # last pixel is no compared
    return t_


def ycomp(t_, _t_, fd, fv, y, Y, r, a, _vP_, _dP_):

    # vertical comparison between pixels forms vertex tuples t2: p, d, dy, m, my, fd, fv
    # last "_" denotes array vs. element, first "_" denotes higher-line array, pattern, or variable

    x, alt_, dalt_, vP_, dP_, next_, dnext_ = 0,[],[],[],[], deque(), deque()
    pri_s, I, D, Dy, M, My, G, olp, e_ = 0,0,0,0,0,0,0,0,[]  # also _G?
    vP = pri_s, I, D, Dy, M, My, G, olp, e_
    dP = pri_s, I, D, Dy, M, My, G, olp, e_  # alt_ included at term, rdn from alt_ eval in form_PP?

    A = a * r

    for t, _t in zip(t_, _t_):  # compares vertically consecutive pixels

        x += 1
        p, d, m = t
        _p, _d, _m = _t

        dy = p - _p   # vertical difference between pixels, summed -> Dy
        dg = _d + dy  # gradient of difference, formed at prior-line pixel _p, -> dG: variation eval?
        fd += dg      # all shorter + current- range dg s within extended quadrant

        my = min(p, _p)   # vertical match between pixels, summed -> My
        vg = _m + my - A  # gradient of predictive value (relative match) at prior-line _p, -> vG
        fv += vg          # all shorter + current- range vg s within extended quadrant

        t2 = p, d, dy, m, my  # all of which are accumulated within P:

        # forms 1D slice of value pattern vP: horizontal span of same-sign vg s with associated vars:

        sv, alt_, dalt_, vP, vP_, next_ = \
        form_P(0, t2, fv, fd, alt_, dalt_, vP, vP_, _vP_, next_, x, y, Y, r, A)

        # forms 1D slice of difference pattern dP: horizontal span of same-sign dg s with associated vars:

        sd, dalt_, alt_, dP, dP_, dnext_ = \
        form_P(1, t2, fd, fv, dalt_, alt_, dP, dP_, _dP_, dnext_, x, y, Y, r, A)

    # line end, last ycomp: lateral d = 0, m = 0, inclusion per incomplete gradient?
    # vP term, no initialization:

    dolp = dP[7]; dalt = len(vP_), dolp; dalt_.append(dalt)
    root_, _vP_, next_ = comp_P(vP, len(vP_), _vP_, dalt_, next_, x, y, Y, r, A)
    vP = vP, alt_, root_; vP_.append(vP)

    # dP term, no initialization:

    olp = vP[7]; alt = len(dP_), olp; alt_.append(alt)
    droot_, _dP_, dnext_ = comp_P(dP, len(dP_), _dP_, alt_, dnext_, x, y, Y, r, A)
    dP = dP, dalt_, droot_; dP_.append(dP)

    return _vP_, _dP_  # with references to vPPs, dPPs, vCPs, dCPs, from comp_P and adjusted by cons_P2


def form_P(type, t2, g, _g, alt_, _alt_, P, P_, _P_, next_, x, y, Y, r, A):  # forms 1D slices of 2D patterns

    p, d, dy, m, my = t2
    pri_s, I, D, Dy, M, My, G, olp, e_ = P  # unpacked to increment or initialize, + _G to eval alt_P rdn?

    s = 1 if g > 0 else 0
    if s != pri_s and x > r + 2:  # P is terminated and compared to overlapping _Ps:

        root_, _P_, next_ = comp_P(P, len(P_), _P_, alt_, next_, x, y, Y, r, A)
        P = P, alt_, root_  # vs. packed in P? or root_ is accumulated within comp_P only?
        P_.append(P)  # Ps include root_ added by comp_P, olps are packed in alt_:

        _alt = len(P_), olp  # len(P_) is index of current P, olp accumulated by form_P() for
        _alt_.append(_alt)  # index and rdn of terminated P is buffered at current alt_P
        I, D, Dy, M, My, G, olp, e_, alt_ = 0,0,0,0,0,0,0,[],[]  # initialized P and alt_

    # P (representing span of same-sign gs) is incremented regardless of termination:

    olp += 1  # P overlap to concurrent alternative-type P
    I += p  # p s summed within P
    D += d; Dy += dy  # lat D for vertical vP comp, + vert Dy for P2 orient adjust eval and gradient
    M += m; My += my  # lateral and vertical, M vs V: eval is per gradient, V = M - 2a * W?
    G += g  # fd|fv summed to define P value, with directional resolution loss

    if type == 0:
        pri = p, g, _g  # also d, dy, m, my, for fuzzy accumulation within r?
        e_.append(pri)  # prior same-line vertex, buffered for selective inc_rng comp
    else:
        e_.append(g)  # prior same-line fuzzy difference gradient, buffered for inc_der comp?

    P = s, I, D, Dy, M, My, G, olp, e_

    return s, alt_, _alt_, P, P_, next_

    # draft below:

def comp_P(P, n, _P_, alt_, next_, x, y, Y, r, A):  # _x: x of _P displaced from _P_ by last comb_P

    # vertical comparison between 1D slices, for selective inclusion in 2D patterns

    buff_, CP_, _x, _n = [],[], 0,0  # n: index of P, _n: index of _P
    root_, _fork_, Fork_ = [],[],[]  # refs to overlapping root_: same-sign higher _Ps, fork_: same-sign lower Ps

    W, IP, DP, DyP, MP, MyP, GP, Rdn, Alt_, P_ = 0,0,0,0,0,0,0,0,[],[]  # PP vars (pattern of patterns), per fork
    WC, IC, DC, DyC, MC, MyC, GC, RdnC, AltC_, PP_ = 0,0,0,0,0,0,0,0,[],[]  # CP vars (connected PPs) at first Fork

    s, I, D, Dy, M, My, G, e_ = P  # also alt_, root_: doesn't to be returned?
    w = len(e_); ix = x - w  # w: P width, ix: P initial coordinate

    while x >= _x:  # P scans over remaining _P_ while there is some horizontal overlap between P and next _P

        _P = _P_.pop(); _n += 1  # _n is _P counter to sync Fork_ with _P_, better than len(P_) - len(_P_)?
        _s, _ix, _x, _w, _I, _D, _Dy, _M, _My, _G, _r, _e_, _rdn, _alt_, _root_ = _P

        if s == _s:  # P comp, combined P match (PM) eval: P -> PP inclusion if PM > A * len(stronger_root_)?

           dx = x - w/2 - _x - _w/2  # mx = mean_dx - dx: signed, or w overlap: match is partial x identity?
           # dxP term: Dx > ave? comp(dx)?

           dw = w -_w; mw = min(w, _w)  # orientation if difference decr / match incr for min.1D Ps over max.2D
           # ddxP term: dw sign == ddx sign? comp(dw, ddx), match -> w*cos match: _w *= cos(ddx), comp(w, _w)?

           '''             
           comp of lateral D and M, /=cos?  default div and overlap eval per P2? not per CP: sparse coverage?

           if mx+mw > a: # input vars norm and comp, also at P2 term: rotation if match (-DS, Ddx), div_comp if rw?  

           comp (dw, ddx) -> m_dw_ddx # to angle-normalize S vars for comp:

           if m_dw_ddx > a: _S /= cos (ddx)

           if dw > a: div_comp (w) -> rw # to width-normalize S vars for comp: 

               if rw > a: pn = I/w; dn = D/w; vn = V/w; 

                  comp (_n) # or default norm for redun assign, but comp (S) if low rw?

                  if d_n > a: div_comp (_n) -> r_n # or if d_n * rw > a: combined div_comp eval: ext, int co-variance?

           else: comp (S) # even if norm for redun assign?
           '''

           root_.append(P)  # root temporarily includes current P and its P comp derivatives, as well as prior P

        rdn = 0  # redundancy (recreated vs. stored): number of stronger-PM root Ps in root_ + alt Ps in alt_
                 # vs.vars *= overlap ratio: prohibitive cost?

        while len(root_) > 0:

            root = root_.pop(); PM = root[0]  # PM (P match: sum of var matches between Ps) is first variable of root P

            for i in range(len(root_)):  # remaining roots are reused by while len(root_)

                _root = root_[i]; _PM = _root[0]  # lateral PM comp, neg v count -> rdn for PP inclusion eval:
                if PM > _PM: _root[1] += 1; root_[i] = _root  # _root P rdn increment: added var?
                else: rdn += 1  # redundancy within root_ and alt_:

        for i in range(len(alt_)):  # refs within alt_P_, dP vs. vP PM comp, P rdn coef = neg v Olp / w?

            ialt_P = alt_[_alt_ + i]; alt_P = alt_[ialt_P]; _PM = alt_P[0]  # _alt_P_ + i: composite address of _P?
            if MP > _PM: alt_[ialt_P[1]] += 1; alt_[_alt_ + i] = ialt_P  # alt_P rdn increment???
            else: rdn += 1

        # combining matching _Ps into PP, and then all connected PPs into CP:
        # selective, no form_dPP?

        if PM > A*8 * rdn:  # P inclusion by combined-P match value

            W +=_w; IP +=_I; DP +=_D; DyP +=_Dy; MP +=_M; MyP +=_My; GP += G; Alt_ += alt_, P_.append(_P)  # PP vars
            PP = W, IP, DP, DyP, MP, MyP, GP, Alt_, P_  # Alt_: root_ alt_ concat, to re-compute redundancy per PP

            root = len(_P_), PP; root_.append(root)  # _P index and PP per root, possibly multiple roots per P
            _fork_.append(_n)  # index of connected P in future next_P_, to be buffered in Fork_ of CP

        if _x <= ix:  # _P and PP output if no horizontal overlap between _P and next P:

            PP = W, IP, DP, DyP, MP, MyP, GP, Alt_, P_  # PP per _root P in _root_
            Fork_ += _fork_  # all continuing _Ps of CP, referenced from its first fork _P: CP flag per _P?

            if (len(_fork_) == 0 and y > r + 3) or y == Y - 1:  # no continuation per _P, term of PP, accum of CP:

                cons_P2(PP)  # term' PP eval for rotation, re-scan, re-comp, recursion, accumulation per _root PP,
                # then _root_ eval at adjusted redundancy? CP vars:
                WC += W; IC += IP; DC += DP; DyC += DyP; MC += MP; MyC += MyP; GC += GP; AltC_ += Alt_; PP_.append(PP)

            else:
                _P = _s, _ix, _x, _w, _I, _D, _Dy, _M, _My, _G, _r, _e_, _alt_, _fork_, _root_  # PP index per root
                # old _root_, new _fork_ (old _fork_ is displaced with old _P_?)
                buff_.append(_P)  # _P is re-inputted for next-P comp

            CP = WC, IC, DC, DyC, MC, MyC, GC, AltC_, PP_, Fork_

            if (len(Fork_) == 0 and y > r + 3) or y == Y - 1:  # no continuation per CP:

                cons_P2(CP)  # eval for rotation, re-scan, cross-comp of P2_? also sum per frame?

            elif n == last_Fork_nP:  # CP_ to _P_ sync for PP inclusion and cons(CP) trigger by Fork_' last _P?

                CP_.append(CP)  # PP may include len(CP_): CP index

            P_.append(P)  # per P per root?

        P = s, w, I, D, Dy, M, My, G, r, e_, alt_, root_  # each root is new, includes P2 if unique cont:
        next_.append(P)  # _P_ = for next line comp, if no horizontal overlap between P and next _P

        buff_.reverse(); _P_ += buff_  # first to pop() in _P_ for next-P comb_P()

    return root_, _P_, next_  # root includes _P and ref PP
    

def cons_P2(P2):  # sub-level 4: eval for rotation, re-scan, re-comp, recursion, accumulation, at PP or CP term

    # rrdn = 1 + rdn_w / len(e_)  # redundancy rate / w, -> P Sum value, orthogonal but predictive
    # S = 1 if abs(D) + V + a * len(e_) > rrdn * aS else 0  # rep M = a*w, bi v!V, rdn I?

    mean_dx = 1  # fractional?
    dx = Dx / H
    if dx > a: comp(abs(dx))  # or if dxP Dx: fixed ddx cost?  comp of same-sign dx only

    vx = mean_dx - dx  # normalized compression of distance: min. cost decrease, not min. benefit?


def Le1(f):  # last "_" denotes array vs. element, first "_" denotes higher-line array, pattern, variable

    r = 1; a = 127  # feedback filters
    Y, X = f.shape  # Y: frame height, X: frame width
    fd, fv, y, vP_, dP_, _vP_, _dP_, F_  = 0,0,0,[],[], deque(), deque(), []

    p_ = f[0, :]  # y is index of new line p_
    _t_= comp(p_)  # _t_ includes ycomp() results: My, Dy, Vq, initialized = 0

    for y in range(1, Y):

        p_ = f[y, :]
        t_ = comp(p_)  # lateral comp of pixels
        _vP_, _dP_ = ycomp(t_, _t_, fd, fv, y, Y, r, a, _vP_, _dP_)  # vertical comp of pixels
        # accumulates vP, dP, vP_, dP_; P ) PP ) CP termination triggers comp_P, cons_P2()
        _t_ = t_

    P_ = vP_, dP_
    F_.append(P_)  # line of patterns is added to frame of patterns, y = len(F_)

    return F_  # output to level 2

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
Le1(f)

