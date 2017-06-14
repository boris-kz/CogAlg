from scipy import misc

'''
    Level 1 with 2D gradient: modified core algorithm levels 1 and 2 

    2D pixel comparison forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    Lateral and vertical comparison is performed on the same level because average lateral match ~ average vertical match. 
    These derivatives form quadrant gradients: average of rightward and downward match or difference per pixel 
    (they are equally representative samples of quadrant). Quadrant gradient is a minimal unit of 2D gradient, so 
    2D pattern is defined by matching sign of quadrant gradient of value for vP, or quadrant gradient of difference for dP.

    4 levels of incremental encoding per line:

    y:   comp()    p_ array of pixels, lateral comp -> p,m,d,
    y-1: ycomp()   t_ array of tuples, vertical comp, der.comb -> 1D P,
    y-2: comb_P()  P_ array of 1D patterns, vertical comb eval, comp -> P2 ) C2
    y-3: cons_P2() C_ array of 2D connected patterns, overlap, eval, P2 consolidation:
    
'''

def comp(p_, X):  # comparison between consecutive pixels in a row, forming tuples: pixel, match, difference

    t_ = []
    pri_p = p_[0]
    t = pri_p, 0, 0  # ignored anyway?
    t_.append(t)

    for x in range(1, X):  # cross-compares consecutive pixels

        p = p_[x]  # new pixel, comp to prior pixel, could use pop()?
        d = p - pri_p  # lateral difference between consecutive pixels
        m = min(p, pri_p)  # lateral match between consecutive pixels
        t = p, d, m
        t_.append(t)
        pri_p = p

    return t_

def ycomp(t_, _t_, fd, fv, _x, y, X, Y, r, A, vP_, dP_, _vP_, _dP_,  # i/o variables, output is Ps, comp to _Ps?
          pri_s, I, D, Dy, M, My, Vq, p_, olp, olp_,  # vP variables
          pri_sd, Id, Dd, Dyd, Md, Myd, Dq, d_, dolp, dolp_):  # dP variables

    # last "_" denotes array vs. element, first "_" denotes higher-line array, pattern, or variable

    pri_p = 0

    for x in range(X):  # compares vertically consecutive tuples

        t = t_[x];  p, d, m = t
        _t = _t_[x]; _p, _d, _m = _t  # My, Dy are per P?

        dy = p - _p   # vertical difference between pixels, -> Dy
        dq = _d + dy  # quadrant gradient of difference, formed at prior-line pixel _p, -> Dq: variation eval?
        fd += dq      # all shorter + current- range dq s within extended quadrant

        my = min(p, _p)   # vertical match between pixels, -> My
        vq = _m + my - A  # quadrant gradient of predictive value (relative match) at prior-line _p, -> Mq?
        fv += vq          # all shorter + current- range vq s within extended quadrant

        # lat D, M: vertical comp, + Dy, My for orient adjust eval and P2 gradient,
        # vs. Dq, Vq: P2 cons eval? or directional res.loss: *cos, /cos cross-cancellation?

        # formation of 1D value pattern vP: horizontal span of same-sign vq s:

        s = 1 if vq > 0 else 0  # s: positive sign of vq
        if x > r + 2 and (s != pri_s or x == X - 1):  # if vq sign miss or line ends, vP is terminated

            vP = pri_s, I, D, Dy, M, My, Vq, p_, olp_  # 2D P
            if y > 1:
               n = len(vP_)
               comb_P(vP, _vP_, _x, x, y, Y, n)  # or comb_vP and comb_dP, with vP_.append(vP)

            o = len(vP_), olp  # len(vP_) is index of current vP
            dolp_.append(o)  # index and olp of terminated vP is buffered at current dP

            I, D, V, olp, dolp, p_, olp_ = 0, 0, 0, 0, 0, [], []  # initialization of new vP and olp

        pri_s = s   # vP (representing span of same-sign vq s) is incremented:
        olp += 1    # overlap to current dP
        I += pri_p  # p s summed within vP
        D += d; Dy += dy  # summed within vP and vP2
        M += m; My += my  # summed within vP and vP2
        Vq += fv  # fvs summed within vP

        pri = pri_p, fd, fv  # same-line prior 2D tuple, buffered for selective inc_rng comp
        t_.append(pri)
        high = p, d, m  # next-higher-line tuple, buffered for spec of next-line vertical D, M comp?
        _t_.append(high)

        # formation of difference pattern dP: horizontal span of same-sign dq s:

        sd = 1 if d > 0 else 0  # sd: positive sign of d;
        if x > r + 2 and (sd != pri_sd or x == X - 1):  # if dq sign miss or line ends, dP is terminated

            dP = pri_sd, Id, Dd, Dyd, Md, Myd, Dq, d_, dolp_  # 1D P
            if y > 1:
               n = len(dP_)
               comb_P(dP, _dP_, _x, x, y, Y, n)  # or comb_vP and comb_dP, with dP_.append(dP)

            o = len(dP_), dolp  # len(dP_) is index of current dP
            olp_.append(o)  # index and dolp of terminated dP is buffered at current vP

            Id, Dd, Vd, olp, dolp, d_, dolp_ = 0, 0, 0, 0, 0, [], []  # initialization of new dP and olp

        pri_sd = sd  # dP (representing span of same-sign dq s) is incremented:
        dolp += 1  # overlap to current vP
        Id += pri_p  # p s summed within dP
        Dd += d; Dyd += dy  # summed within dP and dP2
        Md += m; Myd += my  # summed within dP and dP2
        Dq += fd  # fds summed within dP
        d_.append(fd)  # same fds as in p_ but within dP, to spec vert.Dd comp, no associated derivatives

        pri_p = _p

    return vP_, dP_  # or vC2_, dC2_ formed by comb_P and then cons_P2?

''' 
    _p, vP_, dP_, _vP_, _dP_,  pri_s, I, D, V, p_, olp, olp_,  pri_sd, Id, Dd, Vd, d_, dolp, dolp_  
    for accumulation at comb_P, not cons_P2? 
'''

def comb_P(P, _P_, _x, x, y, Y, n):  # _x of last _P displaced from _P_ by last comb_P(), initially = 0

    x_buff_, y_buff_, CP2_, _n = [],[],[],0  # output arrays and template (prior comparand) counter
    root_, _fork_, cfork_ = [],[],[]  # arrays of same-sign lower- or higher- line Ps

    W, I2, D2, M2, P_ = 0,0,0,0,[]  # variables of P2, poss. per root
    CW, CI2, CD2, CM2, P2_ = 0,0,0,0,[]  # variables of CP2: connected P2s, per cfork_

    s, I, D, M, r, e_, olp_ = P  # M vs. V: no lateral eval, V = M - 2a * W?
    w = len(e_); ix = x - w  # w: width, ix: initial coordinate of a P

    while x >= _x:  # horizontal overlap between P and next _P, forks are not redundant, P2 if multiple 1-forks only?

        _P = _P_.pop(); _n += 1  # to sync with cfork_, better than len(in_P_) - len(_P_)?
        _s, _ix, _x, _w, _I, _D, _M, _r, _e_, _olp_, _root_, __fork_ = _P  # __fork_, _root_: connection tracing in CP2?

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

                if t > 0: cons_P2(P2)  # including _P? eval for rotation, re-scan, re-comp
                CW += W; CI2 += I2; CD2 += D2; CM2 += M2; P2_.append(P2)  # forming variables of CP2

            else:
                _P = _s, _ix, _x, _w, _I, _D, _M, _r, _e_, _olp_, _fork_, _root_, P2
                # old _root_, new _fork_, old _fork_ is displaced with old P2?
                x_buff_.append(_P)  # _P is re-inputted for next-P comp

            CP2 = cfork_, CW, CI2, CI2, CD2, CM2, P2_

            if (len(cfork_) == 0 and y > r + 3) or y == Y - 1:  # no continuation per CP2:

                cons_P2(CP2)  # eval for rotation, re-scan, cross-comp of P2_? also sum per frame?

            elif n == len(cfork_):  # CP2_ to _P_ sync for P2 inclusion and cons(CP2) trigger by cfork_?
            # or last fork in cfork_, possibly disc.?

                CP2_.append(CP2)

    P = s, w, I, D, M, r, e_, olp_, root_  # each root is new, includes P2 if unique cont:
    y_buff_.append(P)  # _P_ = for next line comp, if no horizontal overlap between P and next _P

    _P_.reverse(); _P_ += x_buff_; _P_.reverse() # front concat for next P comp


def cons_P2(P): # at CP2 term, sub-level 4?

    # rrdn = 1 + rdn_w / len(e_)  # redundancy rate / w, -> P Sum value, orthogonal but predictive
    # S = 1 if abs(D) + V + a * len(e_) > rrdn * aS else 0  # rep M = a*w, bi v!V, rdn I?

    mean_dx = 1  # fractional?
    dx = Dx / H
    if dx > a: comp(abs(dx))  # or if dxP Dx: fixed ddx cost?  comp of same-sign dx only

    vx = mean_dx - dx  # normalized compression of distance: min. cost decrease, not min. benefit?


def Le1(Fp_): # last '_' distinguishes array name from element name

    FP_ = []  # output frame and line of template tuples
    Y, X = Fp_.shape  # Y: frame height, X: frame width

    fd, fv, y, r, A, vP_, dP_, _vP_, _dP_ = 0,0,0,0,0,[],[],[],[]  # i/o variables
    pri_s, I, D, V, p_, olp, olp_ = 0,0,0,0,[],0,[]  # vP variables
    pri_sd, Id, Dd, Vd, d_, dolp, dolp_ = 0,0,0,0,[],0,[]  # dP variables

    p_ = Fp_[0, :]  # y is index of new line ip_
    _t_= comp(p_, X)

    for y in range(1, Y):

        p_ = Fp_[y, :]  # y is index of new line ip_
        t_ = comp(p_, X)
        ycomp(t_, _t_, fd, fv, y, X, Y, r, A, vP_, dP_, _vP_, _dP_,  # i/o variables, output is Ps, comp to _Ps?
              pri_s, I, D, V, p_, olp, olp_,  # vP variables
              pri_sd, Id, Dd, Vd, d_, dolp, dolp_)  # dP variables
        # internal comb_P()) cons_P2(): triggered by P2) C2 termination
        _t_ = t_

    FP_.append(P_)  # line of patterns is added to frame of patterns, y = len(Ft_)

    return FP_  # output to level 2

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
Le1(f)

