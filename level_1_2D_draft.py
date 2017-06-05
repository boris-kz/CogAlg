from scipy import misc

'''

Level 1 with 2D gradient: modified combination of core algorithm levels 1 and 2 

Initial 2D comparison forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. Both lateral and vertical comparison is
performed on the same level because average lateral match ~ average vertical match. These derivatives form quadrant gradients:
average of rightward and downward match or difference per pixel (they are equally representative samples of quadrant).
Quadrant gradient is a minimal unit of 2D gradient, so 2D pattern (blob) is defined by matching sign of 
quadrant gradient of value for vP, or quadrant gradient of difference for dP.

'''

def comp(p, pri_p, _p, _d, _m, fd, fv, dy, my, x, y, _x, X, r, A, # input variables, x and y from higher-scope for loop w, h
         pri_s, I, D, V, rv, p_, ow, alt_,  # variables of vP
         pri_sd, Id, Dd, Vd, rd, d_, owd, dalt_,  # variables of dP
         _ip_, vP_, dP_, _vP_, _dP_):  # template array and output pattern arrays

    # last "_" denotes array vs. element, first "_" denotes array, pattern, or variable from higher line

    d = p - pri_p    # difference between laterally consecutive pixels
    if y > 0: dy = p - _p  # difference between vertically consecutive pixels, else dy = 0

    dq = _d + dy     # quadrant gradient of difference per prior-line pixel _p
    fd += dq         # all shorter + current- range dq s within extended quadrant?

    m = min(p, pri_p)  # match between laterally consecutive pixels
    if y > 0: my = min(p, _p)  # match between vertically consecutive pixels, else my = 0

    vq = _m + my - A # quadrant gradient of predictive value (relative match) per prior-line _p
    fv += vq         # all shorter + current- range vq s within extended quadrant


    # formation of 1D value pattern vP: horizontal span of same-sign vq s:

    s = 1 if vq > 0 else 0  # s: positive sign of vq
    if x > r+2 and (s != pri_s or x == X-1):  # vP is terminated if vq sign miss or line ends

        vP = pri_s, I, D, V, rv, p_, alt_  # no default div: redundancy eval per P2 on Le2?
        vP_.append(vP)  # or vP_ is within vP2, after comp_vP:

        if y > 1: comp_P(vP, _vP_, x, _x, y)  # called from comp() if P term, separate comp_vP and comp_dP?

        alt = len(vP_), ow  # len(P_) is an index of last overlapping vP
        dalt_.append(alt)  # addresses of overlapping vPs and ow are buffered at current dP
        I, D, V, rv, ow, owd, p_, alt_ = (0, 0, 0, 0, 0, 0, [], [])  # initialization of new vP and ow

    pri_s = s   # vP (representing span of same-sign vq s) is incremented:
    ow += 1     # overlap to current dP
    I += pri_p  # ps summed within P width
    D += dq     # fds summed within P width
    V += fv     # fvs summed within P width

    pri = pri_p, fd, fv  # same-line prior tuple,
    p_.append(pri)  # buffered within P for selective inc_rng comp

    high = p, d, m  # new higher-line tuple, m sum \ P, v sum \ P2: compressed?
    _ip_.append(high)  # buffered for next-line comp


    # formation of difference pattern dP: horizontal span of same-sign dq s:

    sd = 1 if d > 0 else 0  # sd: positive sign of d;
    if x > r+2 and (sd != pri_sd or x == X-1):  # if derived pri_sd miss, dP is terminated

        dP = pri_sd, Id, Dd, Vd, rd, d_, dalt_
        dP_.append(dP)  # output of dP

        if y > 1: comp_P(dP, _dP_, x, _x, y)  # called from comp() if P term, separate comp_vP and comp_dP?

        alt = len(dP_), owd  # len(P_) is an address of last overlapping dP
        alt_.append(alt)  # addresses of overlapping dPs and owds are buffered at current vP
        Id, Dd, Vd, rd, ow, owd, d_, dalt_ = (0, 0, 0, 0, 0, 0, [], [])  # initialization of new dP and ow

    pri_sd = sd  # dP (representing span of same-sign dq s) is incremented:
    owd += 1     # overlap to current vP
    Id += pri_p  # ps summed within P width
    Dd += fd     # fds summed within P width
    Vd += fv     # fvs summed within P width
    d_.append(fd)  # prior fds buffered in P for select inc_der, no associated derivatives, no lateral high?


    return pri_s, I, D, V, rv, p_, ow, alt_, \
           pri_sd, Id, Dd, Vd, rd, d_, owd, dalt_, \
           _x, _ip_, vP_, dP_, _vP_, _dP_ # for next p comparison and P increment

'''

incremental depth of comp per prior line, up to 4 levels:

- array of pixels on current y line: lateral comp ->
- array of p,m,d tuples on y-1 line: vertical comp ->
- array of 1D patterns on y -2 line: vertical comp_P()->
- array of 2D patterns on y -3 line: term(), comp_P2 is on Le2: higher-composition inputs

summation per 1D P, then comp, 2D sum if s cont. at comp_P?
1D alt_ are combined into P2 alt_:?

'''

def comp_P(P, _P_, x, _x, y):  # called from comp() if P term, _x is from last comp_P() within line

    s, I, D, M, r, e_, alt_ = P  # M vs. V: no lateral eval, V = M - 2a * W?
    w = len(e_)
    ix = x - w  # initial x coordinate?
    buff_ = []  # _Ps re-inputted for next P comp
    next_ = []  # Ps re-inputted for next line comp
    term_ = []  # discontinued _Ps outputted to term()
    P2_ = {}    # summed _Ps, potentially multiple per P
    n, W, I2, D2, M2, rP_ = (0, 0, 0, 0, 0, [])  # number and sum of _P inclusions per P

    while x > _x:  # horizontal overlap between P and next _P, no redundancy and eval till P2?

        _P, _P2_ = _P_.pop()  # root P2s, inclusion by sign only: ~dP, !comp?
        _s, _n, _ix, _x, _I, _D, _M, _r, _e_, _alt_ = _P
        _w = len(_e_)

        if s == _s: # inclusion into _P, then comp?

            n += 1; W += w; I2 += _I; D2 += _D; M2 += _M  # P2 = n, w, I2, D2, M2, P_

            dx = x - w/2 - _x - _w/2  # comp (x, _x), or mx = overlap?
            
            comp (w, _w) # -> mw, dw, re-orientation if projected difference decr. / match incr. for minimized 1D Ps over maximized 2D:
 
               comp (dw, ddx);  if match (dw, ddx) > a: _w *= cos (ddx); comp (w, _w) # proj w*cos match, if same-sign dw, ddx, not |.|

            ''' 
            if mx+mw > a: # conditional input vars norm and comp, also at P2 term: rotation if match (-DS, Ddx),  div_comp if rw?  
 
               comp (dw, ddx) -> m_dw_ddx # to angle-normalize S vars for comp:
 
            if m_dw_ddx > a: _S /= cos (ddx)
 
               if dw > a: div_comp (w) -> rw # to width-normalize S vars for comp: 
 
                  if rw > a: pn = I/w; dn = D/w; vn = V/w; 
 
                     comp (_n) # or default norm for redun assign, but comp (S) if low rw?
 
                     if d_n > a: div_comp (_n) -> r_n # or if d_n * rw > a: combined div_comp eval: ext, int co-variance?
 
            else: comp (S) # even if norm for redun assign?
            '''

        if _x <= ix:  # no horizontal overlap between _P and next P
            out = _P, _P2_

            if _n == 0: term_.append(out) # or immediate term(out)?
            else: buff_.append(out) # also: P_ += _P_: transfer to P?

    out = P, P2_; next_.append(out) # no horizontal overlap between P and next _P

    _P_.reverse(); _P_ += buff_; _P_.reverse() # front concat: re-input to next_P patt()?


        if y > r + 3 and (_n == 0 or y == Y - 1):

            term()  # called from comp_P(): no new input needed

            # P2 is terminated if all exposed Ps are displaced (no lower-line continuation),
            # and evaluated for rotation, 1D re-scan and re-comp:

            # rrdn = 1 + rdn_w / len(e_)  # redundancy rate / w, -> P Sum value, orthogonal but predictive
            # S = 1 if abs(D) + V + a * len(e_) > rrdn * aS else 0  # rep M = a*w, bi v!V, rdn I?

def term():

    mean_dx = 1  # fractional?
    dx = Dx / H
    if dx > a: comp(abs(dx))  # or if dxP Dx: fixed ddx cost?  comp of same-sign dx only

    vx = mean_dx - dx  # normalized compression of distance: min. cost decrease, not min. benefit?


def Le1(Fp_): # last '_' distinguishes array name from element name

    FP_ = []  # output frame of vPs: relative match patterns, and dPs: difference patterns
    Y, X = Fp_.shape  # Y: frame height, X: frame width

    a = 127  # minimal filter for vP inclusion
    min_r=0  # default range of fuzzy comparison, initially 0

    for y in range(Y):

        dy = 0; my = 0  # used by comp within 1st line
        ip_ = Fp_[y, :]  # y is index of input line ip_ (vs. p_ of P)
        if y > 0: _ip_ = Fp_[y-1, :]  # else dq = d + 0, vq = m + 0 - A

        if min_r == 0: A = a
        else: A = 0;

        _p, _m, _d, fd, fv, r, x, vP_, dP_ = (0, 0, 0, 0, 0, 0, 0, [], [])  # nP tuple
        pri_s, I, D, V, rv, ow, p_, alt_ = (0, 0, 0, 0, 0, 0, [], [])  # vP tuple
        pri_sd, Id, Dd, Vd, rd, owd, d_, dalt_ = (0, 0, 0, 0, 0, 0, [], [])  # dP tuple

        for x in range(X):  # cross-compares consecutive pixels, outputs sequence of d, m, v:

            p = ip_[x]  # p_ within P only, could use pop()? comp to prior pixel:
            if y > 0: _p, _m, _d = _ip_[x]  # replaced with patterns by comp:

            if x > 0:
                pri_s, I, D, V, rv, p_, ow, alt_, \
                pri_sd, Id, Dd, Vd, rd, d_, owd, dalt_, \
                _x, _ip_, vP_, dP_, _vP_, _dP_= \
                comp(p, pri_p, _p, _d, _m, fd, fv, dy, my, x, y, _x, X, r, A, # input variables
                     pri_s, I, D, V, rv, p_, ow, alt_,  # variables of vP
                     pri_sd, Id, Dd, Vd, rd, d_, owd, dalt_,  # variables of dP
                     _ip_, vP_, dP_, _vP_, _dP_)  # template array and output pattern arrays

            pri_p = p  # prior pixel, pri_ values are always derived before use

        LP_ = vP_, dP_
        FP_.append(LP_)  # line of patterns P_ is added to frame of patterns, y = len(FP_)

    return FP_  # or return FP_  # frame of patterns (vP or dP): output to level 2;

'''

1D orientation is initially arbitrary but comparison of average coordinate x and width w is necessary to re-orient it for maximized 1D P match.

- initial 1D Ps are not nested, and are not defined by horizontal sign,

- elements of p_ (but not d_) contain both leftward and downward derivatives,

- overlap () records area of overlap per alt_P2s, assigned to both and selected at 2Le eval 


2D P includes array of initial 1D Ps and their variables, external vars x and w (new first) are cross-compared within 2D P (P2) by default:


comp (x, _x) -> dx; mx = mean_dx - dx // normalized compression of distance: min cost decrease, not min benefit?

  if dx > a: comp (|dx|) // or if dxP Dx: fixed ddx cost?  comp of same-sign dx only

comp (w, _w) -> mw, dw // default, re-orientation if projected difference decr. / match incr. for minimized 1D Ps over maximized 2D:

  comp (dw, ddx);  if match (dw, ddx) > a: _w *= cos (ddx); comp (w, _w) // proj w * cos match, if same-sign dw, ddx, not |.|


if mx+mw > a: // conditional input vars norm and comp, also at P2 term: rotation if match (-DS, Ddx), div_comp if rw?  

   comp (dw, ddx) -> m_dw_ddx // to angle-normalize S vars for comp:

   if m_dw_ddx > a: _S /= cos (ddx)

   if dw > a: div_comp (w) -> rw // to width-normalize S vars for comp: 

      if rw > a: pn = I/w; dn = D/w; vn = V/w; 

         comp (_n) // or default norm for redun assign, but comp (S) if low rw?

         if d_n > a: div_comp (_n) -> r_n // or if d_n * rw > a: combined div_comp eval: ext, int co-variance?

   else: comp (S) // even if norm for redun assign?


P2 term: orient and norm at a fixed cost of *cos and div_comp: two-level selection?


L = h (height) / cos (Dx / h);  w = W/ L // max L, min ave w, within level 1?  

if yD * M_dw_ddx > a: scan angle * cos (Dx) // or if match (-DS, Ddx)? 

   coord stretch | compress \ ddx, or rotation: shift and re-scan \ Ddx, or re-definition \ frame Ddx,  to min difference and max match?

   potentially recursive 1D ) 2D comp within P2: generic levels 1 and 2 


if Mx + Mw > a: // /+ extv_P2 of any power, value of summed-variables S (xI, xD, xM) norm and comp between 1D Ps:

   if SS * Dw > a: div_comp (w, S) // initial P comp override?

   else: comp (S, _S)

'''

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
Le1(f)
