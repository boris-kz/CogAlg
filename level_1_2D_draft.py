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

def comp(p_, X):  # comparison of consecutive pixels in a scan line forms tuples: pixel, match, difference

    t_ = deque()
    pri_p = p_[0]  # no d, m at x=0
    t = pri_p; t_.appendleft(t)

    for x in range(1, X):  # cross-compares consecutive pixels

        p = p_[x]  # new pixel, comp to prior pixel:
        d = p - pri_p  # lateral difference between consecutive pixels
        m = min(p, pri_p)  # lateral match between consecutive pixels
        t = p, d, m; t_.appendleft(t)
        pri_p = p

    return t_

def ycomp(t_, _t_, fd, fv, _x, y, X, Y, r, a, _vP_, _dP_):

    # vertical comparison between pixels, forming 1D slices of 2D patterns
    # last "_" denotes array vs. element, first "_" denotes higher-line array, pattern, or variable

    vP_, dP_, next_vP_, next_dP_ = [],[], deque(), deque()
    pri_s, I, D, Dy, M, My, g, e_, = 0, 0, 0, 0, 0, 0, 0, []
    vP = pri_s, I, D, Dy, M, My, g, e_
    dP = pri_s, I, D, Dy, M, My, g, e_
    alt_, dalt_ = [],[]  # filled by alt_P, inclusion in P at term, rdn from alt_ eval in form_PP()?

    A = a * r
    pri_p = t_[0]  # no d, m at x=0

    for x in range(1, X):  # compares vertically consecutive tuples, resulting derivatives end with 'y' and 'g':

        t = t_[x];  p, d, m = t
        _t = _t_[x]; _p, _d, _m = _t  # _my, _dy, fd, fv are accumulated within current P

        dy = p - _p   # vertical difference between pixels, -> Dy
        dg = _d + dy  # gradient of difference, formed at prior-line pixel _p, -> Dg: variation eval?
        fd += dg      # all shorter + current- range dg s within extended quadrant

        my = min(p, _p)   # vertical match between pixels, -> My
        vg = _m + my - A  # gradient of predictive value (relative match) at prior-line _p, -> Mg?
        fv += vg          # all shorter + current- range vg s within extended quadrant

        t2 = pri_p, d, dy, m, my, vg, dg

        sv, alt_, dalt_, vP, vP_ = form_P(t2, fv, alt_, dalt_, vP, vP_, _vP_, x, _x, y, X, Y, r, A)
        # forms 1D value pattern vP: horizontal span of same-sign vg s with associated vars

        sd, dalt_, alt_, dP, dP_ = form_P(t2, fd, dalt_, alt_, dP, dP_, _dP_, x, _x, y, X, Y, r, A)
        # forms 1D difference pattern dP: horizontal span of same-sign dg s with associated vars

        pri_p = _p   # for inclusion into vP and dP by laterally-next p' ycomp()

    return _vP_, _dP_  # with references to vPPs, dPPs, vCPs, dCPs formed by comb_P and adjusted by cons_P2


def form_P(t2, f, alt_, _alt_, P, P_, _P_, x, _x, y, X, Y, r, A):

    rdn = 0; next_ = deque()
    p, d, dy, m, my = t2
    pri_s, I, D, Dy, M, My, g, p_ = P

    s = 1 if g > 0 else 0  # s: positive sign of vg
    if x > r + 2 and (s != pri_s or x == X - 1):  # if vg sign miss or line ends, vP is terminated

        if y > 1:  # comp_P() or separate comb_vP() and comb_dP()?

           P = pri_s, I, D, Dy, M, My, g, p_, alt_  # M vs V: eval per vertex, V = M - 2a * W?
           root_, _vP_, next_vP_ = comp_P(P, len(P_), _P_, _alt_, next_, x, _x, y, Y, r, A)
           P = P, root_
           P_.append(P)  # vPs include root_ formed by comb_P

           o = len(P_), rdn  # len(vP_) is index of current vP, alt formed by comb_P()
           alt_.append(o)  # index and alt of terminated vP is buffered at current dP

           I, D, Dy, M, My, Vg, p_, alt_, _alt_ = 0,0,0,0,0,0,[],[],[]  # init. vP and dalt
           # vP (representing span of same-sign vg s) is incremented:

        rdn += 1  # P overlap to concurrent alternative-type P
        I += p  # p s summed within vP
        D += d; Dy += dy  # lat D for vertical vP comp, + vert Dy for P2 orient adjust eval and gradient
        M += m; My += my  # lateral and vertical summation within vP and vP2
        g += f    # fvs summed to define vP value, but directional res.loss for orient eval

        if P==vP:
            pri = p, fd, fv
            
        p_.append(pri)  # prior same-line quadrant vertex, buffered for selective inc_rng comp

    return s, alt_, _alt_, P, P_

    # draft below:


def comp_P(P, n, _P_, _alt_P, next_P, x, _x, y, Y, r, A):  # _x: x of _P displaced from _P_ by last comb_P

    # combines matching _Ps into PP and then PPs into CP, _alt_P_: address only

    buff_, CP_, _n = [],[], 0  # n: index of P, _n: index of _P
    root_, _fork_, Fork_ = [],[],[]  # refs to overlapping root_: same-sign higher _Ps, fork_: same-sign lower Ps

    W, IP, DP, DyP, MP, MyP, GP, Rdn, Alt_, P_ = 0,0,0,0,0,0,0,0,[],[]  # PP vars (pattern of patterns), per fork
    WC, IC, DC, DyC, MC, MyC, GC, RdnC, AltC_, PP_ = 0,0,0,0,0,0,0,0,[],[]  # CP vars (connected PPs) at first Fork

    s, I, D, Dy, M, My, G, e_, rdn, alt_ = P  # rdn: relative overlap to stronger alt_Ps?
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

        for i in range(len(alt_)):  # refs within alt_P_, dP vs. vP PM comp, neg v count -> rdn for PP inclusion eval:

            ialt_P = alt_[_alt_P_ + i]; alt_P = alt_[ialt_P]; _PM = alt_P[0]  # _alt_P_ + i: composite address of _P?
            if MP > _PM: alt_[ialt_P[1]] += 1; alt_[_alt_P_ + i] = ialt_P  # alt_P rdn increment???
            else: rdn += 1

        if PM > A*10 * rdn:  # P inclusion by combined-P match value

            W +=_w; IP +=_I; DP +=_D; DyP +=_Dy; MP +=_M; MyP +=_My; GP += G; Alt_ += alt_, P_.append(_P)  # PP vars
            PP = W, IP, DP, DyP, MP, MyP, GP, Alt_, P_  # Alt_: root_ alt_s concat, to recreate PP redundancy?

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
    next_P_.append(P)  # _P_ = for next line comp, if no horizontal overlap between P and next _P

    buff_.reverse(); _P_ += buff_  # first to pop() in _P_ for next-P comb_P()

    return root_, _P_, next_P_  # root includes _P and ref PP
    

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

    fd, fv, _x, y, vP_, dP_, _vP_, _dP_, F_  = 0,0,0,0,[],[], deque(), deque(), []
    s, I, D, Dy, M, My, g, p_, rdn, alt_ = 0, 0, 0, 0, 0, 0, 0, [], 0, []
    vP = s, I, D, Dy, M, My, g, p_, rdn, alt_
    dP = s, I, D, Dy, M, My, g, p_, rdn, alt_

    p_ = f[0, :]  # y is index of new line p_
    _t_= comp(p_, X)  # _t_ includes ycomp() results: My, Dy, Vq, initialized = 0

    for y in range(1, Y):

        p_ = f[y, :]  # y is index of new line ip_
        t_ = comp(p_, X)

        _vP_, _dP_ = ycomp(t_, _t_, fd, fv, _x, y, X, Y, r, a, vP, dP, vP_, dP_, _vP_, _dP_)

        # vP, dP, vP_, dP_ for local accumulation?
        # form_P() within ycomp(), comp_P and cons_P2() are triggered by P ) PP ) CP termination

        _t_ = t_

    P_ = vP_, dP_
    F_.append(P_)  # line of patterns is added to frame of patterns, y = len(F_)

    return F_  # output to level 2

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
Le1(f)

