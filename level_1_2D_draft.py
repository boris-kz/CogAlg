from scipy import misc

'''

Level 1 with 2D gradient: modified combination of core algorithm levels 1 and 2 

Initial 2D comparison forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. Both lateral and vertical comparison is
performed on the same level because average lateral match ~ average vertical match. These derivatives form quadrant gradients:
average of rightward and downward match or difference per pixel (these are equally representative samples of quadrant).
Quadrant gradient is a minimal unit of 2D gradient, so 2D pattern (blob) is defined by matching sign of 
quadrant gradient of value for vP, or quadrant gradient of difference for dP.

'''

def comp(p, pri_p, _p, _d, _m, fd, fv, dy, my, _ip_,  # input variables
         pri_s, I, D, V, p_, olp, olp_,  # variables of vP
         pri_sd, Id, Dd, Vd, d_, dolp, dolp_,  # variables of dP
         x, _x, X, y, Y, r, A, vP_, dP_, _vP_, _dP_): # parameters and output

    # last "_" denotes array vs. element, first "_" denotes template: higher-line array, pattern, or variable

    d = p - pri_p    # lateral difference between pixels, -> lat.D for vert.comp, orient min, max cross-cancel in dq?
    if y > 0: dy = p - _p  # vertical difference between pixels, -> vert.D, comb with lat.D for orient eval per P2

    dq = _d + dy     # quadrant gradient of difference, formed at prior-line pixel _p, -> Dq for variation eval?
    fd += dq         # all shorter + current- range dq s within extended quadrant, for pixel inclusion

    m = min(p, pri_p)  # lateral match between pixels, -> lat.M for vert.comp, directional res.loss in vq?
    if y > 0: my = min(p, _p)  # vertical match between pixels, else my = 0

    vq = _m + my -A  # quadrant gradient of predictive value (relative match) at prior-line _p, -> Mq for P2 eval?
    fv += vq         # all shorter + current- range vq s within extended quadrant?


    # formation of 1D value pattern vP: horizontal span of same-sign vq s:

    s = 1 if vq > 0 else 0  # s: positive sign of vq
    if x > r+2 and (s != pri_s or x == X-1):  # if vq sign miss or line ends, vP is terminated

        vP = pri_s, I, D, V, p_, olp_
        if y > 1:
           n = len(vP_); comp_P(vP, _vP_, x, _x, y, Y, n)  # or comp_vP and comp_dP, with vP_.append(vP)

        o = len(vP_), olp  # len(vP_) is index of current vP
        dolp_.append(o)  # index and olp of terminated vP is buffered at current dP

        I, D, V, olp, dolp, p_, olp_ = 0, 0, 0, 0, 0, [], []  # initialization of new vP and olp

    pri_s = s   # vP (representing span of same-sign vq s) is incremented:
    olp += 1    # overlap to current dP
    I += pri_p  # p s summed within vP
    D += dq     # fds summed within vP
    V += fv     # fvs summed within vP

    # S denotes summed vars above, except olp: relative to w
    # lateral S for vertical comp -> potential adjustment by orientation?
    # vertical S for complementary adjustment and P2 gradient, no dq or vq sum: redundant?

    pri = pri_p, fd, fv  # same-line prior 2D tuple, buffered for selective inc_rng comp
    p_.append(pri)
    high = p, d, m  # next-higher-line tuple, buffered for spec of next-line vertical D, M comp?
    _ip_.append(high)

    # formation of difference pattern dP: horizontal span of same-sign dq s:

    sd = 1 if d > 0 else 0  # sd: positive sign of d;
    if x > r+2 and (sd != pri_sd or x == X-1):  # if dq sign miss or line ends, dP is terminated

        dP = pri_sd, Id, Dd, Vd, d_, dolp_
        if y > 1:
           n = len(dP_); comp_P(dP, _dP_, x, _x, y, Y, n)  # or comp_vP and comp_dP, with dP_.append(dP)

        o = len(dP_), dolp  # len(dP_) is index of current dP
        olp_.append(o)  # index and dolp of terminated dP is buffered at current vP

        Id, Dd, Vd, olp, dolp, d_, dolp_ = 0, 0, 0, 0, 0, [], []  # initialization of new dP and olp

    pri_sd = sd  # dP (representing span of same-sign dq s) is incremented:
    dolp += 1    # overlap to current vP
    Id += pri_p  # p s summed within dP
    Dd += fd     # fds summed within dP
    Vd += fv     # fvs summed within dP
    d_.append(fd)  # same fds as in p_ but within dP, to spec vert.Dd comp, no associated derivatives


    return _x, _ip_, vP_, dP_, _vP_, _dP_, \
           pri_s, I, D, V, p_, olp, olp_, \
           pri_sd, Id, Dd, Vd, d_, dolp, dolp_ # for next p comparison and P increment

'''
4 levels of incremental depth of comp per higher line:

y:    p_ array of pixels, lateral comp -> p,m,d,
y-1: _p_ array of tuples, vertical comp -> 1D P,
y-2:  P_ array of 1D patterns, vertical comp, sum -> P2,
y-3: _P_ array of 2D patterns, overlap, eval? P2 consolidation;
'''

def comp_P(P, _P_, x, _x, y, Y, n):  # _x: of last P within line

    x_buff_, y_buff_, CP2_, _n = [],[],[],0  # output arrays and template (prior comparand) counter
    root_, _fork_, cfork_ = [],[],[]  # arrays of same-sign lower- or higher- line Ps

    W, I2, D2, M2, P_ = 0,0,0,0,[]  # variables of P2, poss. per root
    CW, CI2, CD2, CM2, P2_ = 0,0,0,0,[]  # variables of CP2: connected P2s, per cfork_

    s, I, D, M, r, e_, olp_ = P  # M vs. V: no lateral eval, V = M - 2a * W?
    w = len(e_); ix = x - w  # w: width, ix: initial coordinate of a P

    while x > _x:  # horizontal overlap between P and next _P, forks are not redundant, P2 if multiple 1-forks only?

        _P = _P_.pop(); _n += 1  # to sync with cfork_, better than len(in_P_) - len(_P_)?
        _s, _ix, _x, _w, _I, _D, _M, _r, _e_, _olp_, _root_, __fork_ = _P  # __fork_, _root_: trunk tracing in CP2?

        if s == _s:  # !eval, ~dP? -> P2 at y_buff_.append(P) if vertically cont. 1-forks, known after while x > _x?

            root_.append(len(_P_))  # index of connected _P within _P_
            _fork_.append(n)  # future index of connected P within y_buff_, not yet displaced

            dx = x - w/2 - _x - _w/2  # mx = mean_dx - dx, signed, or unsigned overlap?
            dw = w -_w; mw = min(w, _w)  # orientation if difference decr / match incr for min.1D Ps over max.2D:

            # comp(dx), comp(dw, ddx) at dxP term?
            # if match(dw, ddx) > a: _w *= cos(ddx); comp(w, _w)  # proj w*cos match, if same-sign dw, ddx, not |.|

            # comp of separate lateral D and M
            # default div and overlap eval per P2? not CP2: sparse coverage?

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

        if len(__fork_) == 1 and len(_fork_) == 1: # then _P includes trunk P2:
           t = 1; W +=_w; I2 +=_I; D2 +=_D; M2 +=_M; P_.append(_P)
        else: t = 0 

        if _x <= ix:  # _P output if no horizontal overlap between _P and next P:

            P2 = W, I2, D2, M2, P_  # ?
            cfork_.append(_fork_)  # all continuing _Ps of CP2

            if (len(_fork_) == 0 and y > r + 3) or y == Y - 1:  # no continuation for current _P and its P2:

                if t > 0: cons(P2)  # including _P? eval for rotation, re-scan, re-comp
                CW += W; CI2 += I2; CD2 += D2; CM2 += M2; P2_.append(P2)  # forming variables of CP2

            else:
                _P = _s, _ix, _x, _w, _I, _D, _M, _r, _e_, _olp_, _fork_, _root_, P2
                # old _root_, new _fork_, old _fork_ is displaced with old P2?
                x_buff_.append(_P)  # _P is re-inputted for next-P comp

            CP2 = cfork_, CW, CI2, CI2, CD2, CM2, P2_

            if (len(cfork_) == 0 and y > r + 3) or y == Y - 1:  # no continuation per CP2:

                cons(CP2)  # eval for rotation, re-scan, cross-comp of P2_? also sum per frame?

            elif n == len(cfork_):  # CP2_ to _P_ sync for P2 inclusion and cons(CP2) trigger by cfork_?
            # or last fork in cfork_, possibly disc.?

                CP2_.append(CP2)

    P = s, w, I, D, M, r, e_, olp_, root_  # each root is new, includes P2 if unique cont:
    y_buff_.append(P)  # _P_ = for next line comp, if no horizontal overlap between P and next _P

    _P_.reverse(); _P_ += x_buff_; _P_.reverse() # front concat for next P comp


def cons(P): # at CP2 term?

    # rrdn = 1 + rdn_w / len(e_)  # redundancy rate / w, -> P Sum value, orthogonal but predictive
    # S = 1 if abs(D) + V + a * len(e_) > rrdn * aS else 0  # rep M = a*w, bi v!V, rdn I?

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
        ip_= Fp_[y, :]  # or ip_ = Fp_[0, :] lat comp before vert comp?
        if y > 0:
           _ip_ = Fp_[y-1, :]  # else dq = d + 0, vq = m + 0 - A

        if min_r == 0: A = a
        else: A = 0;

        _p, _m, _d, fd, fv, r, x, vP_, dP_ = 0, 0, 0, 0, 0, 0, 0, [], []  # i/o tuple
        pri_s, I, D, V, olp, p_, olp_ = 0, 0, 0, 0, 0, [], []  # vP tuple
        pri_sd, Id, Dd, Vd, dolp, d_, dolp_ = 0, 0, 0, 0, 0, [], []  # dP tuple

        for x in range(X):  # cross-compares consecutive pixels, outputs sequence of d, m, v:

            if x > 0:  # no pri_p = ip_[0]: vertical comp anyway?
                p = ip_[x]
                _p, _m, _d  = _ip_[x]  # replaced with patterns by comp:

                _x, _ip_, vP_, dP_, _vP_, _dP_, \
                pri_s, I, D, V, p_, olp, olp_, \
                pri_sd, Id, Dd, Vd, d_, dolp, dolp_ = \
                \
                comp(p, pri_p, _p, _d, _m, fd, fv, dy, my, _ip_,  # input variables
                     pri_s, I, D, V, p_, olp, olp_,  # variables of vP
                     pri_sd, Id, Dd, Vd, d_, dolp, dolp_,  # variables of dP
                     x, _x, X, y, r, A, vP_, dP_, _vP_, _dP_)  # parameters and output

            pri_p = p  # prior pixel, pri_ values are always derived before use

        LP_ = vP_, dP_
        FP_.append(LP_)  # line of patterns P_ is added to frame of patterns, y = len(FP_)

    return FP_  # frame of patterns (vP or dP) is output to level 2

'''

1D orientation is initially arbitrary but comparison of average coordinate x and width w is necessary to re-orient it for maximized 1D P match.

- initial 1D Ps are not nested, and are not defined by horizontal sign,

- elements of p_ (but not d_) contain both leftward and downward derivatives,

- overlap () records area of overlap per olp_P2s, assigned to both and selected at 2Le eval 


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
