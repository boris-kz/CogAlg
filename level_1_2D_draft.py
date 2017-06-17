from scipy import misc

'''
    Level 1 with 2D gradient: modified core algorithm of levels 1 + 2. 

    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match. 
    Minimal unit of 2D is quadrant defined by 4 pixels. 
    
    Derivatives in a given quadrant have two equally representative samples, unique per pixel: 
    right-of-pixel and down-of-pixel. Hence, quadrant gradient is computed as an average of the two.  
    2D pattern is defined by matching sign of quadrant gradient of value for vP and of difference for dP.

    Level 1 has 4 steps of incremental encoding per added scan line, defined by coordinate y:

    y:   comp()    p_ array of pixels, lateral comp -> p,m,d,
    y-1: ycomp()   t_ array of tuples, vertical comp, der.comb -> 1D P,
    y-2: comb_P()  P_ array of 1D patterns, vertical comb eval, comp -> P2 ) C2
    y-3: cons_P2() C_ array of 2D connected patterns, overlap, eval, P2 consolidation:
    
'''

def comp(p_, X):  # comparison of consecutive pixels in a scan line forms tuples: pixel, match, difference

    t_ = []
    pri_p = p_[0]
    t = pri_p, 0, 0  # ignored anyway?
    t_.append(t)

    for x in range(1, X):  # cross-compares consecutive pixels

        p = p_[x]  # new pixel, comp to prior pixel, pop() is faster?
        d = p - pri_p  # lateral difference between consecutive pixels
        m = min(p, pri_p)  # lateral match between consecutive pixels
        t = p, d, m
        t_.append(t)
        pri_p = p

    return t_

def ycomp(t_, _t_, fd, fv, _x, y, X, Y, a, r, vP, dP, vP_, dP_, _vP_, _dP_):

    # vertical comparison between pixels, forming 1D slices of 2D patterns
    # last "_" denotes array vs. element, first "_" denotes higher-line array, pattern, or variable

    A = a * r; pri_p = 0

    for x in range(X):  # compares vertically consecutive tuples, resulting derivatives end with 'y' and 'q':

        t = t_[x];  p, d, m = t
        _t = _t_[x]; _p, _d, _m = _t  # _my, _dy, fd, fv are accumulated within current P

        dy = p - _p   # vertical difference between pixels, -> Dy
        dq = _d + dy  # quadrant gradient of difference, formed at prior-line pixel _p, -> Dq: variation eval?
        fd += dq      # all shorter + current- range dq s within extended quadrant

        my = min(p, _p)   # vertical match between pixels, -> My
        vq = _m + my - A  # quadrant gradient of predictive value (relative match) at prior-line _p, -> Mq?
        fv += vq          # all shorter + current- range vq s within extended quadrant


        # formation of 1D value pattern vP: horizontal span of same-sign vq s with associated vars:

        s = 1 if vq > 0 else 0  # s: positive sign of vq
        pri_s, I, D, Dy, M, My, Vq, p_, olp, olp_ = vP  # vP tuple initialized as list, vars re-assigned by dP?
        dolp_ = dP[8]

        if x > r + 2 and (s != pri_s or x == X - 1):  # if vq sign miss or line ends, vP is terminated

            if y > 1:
               n = len(vP_)  # vP is  packed in ycomp declaration:
               comb_P(vP, _vP_, _x, x, y, Y, n)  # or comb_vP and comb_dP, with vP_.append(vP)

            o = len(vP_), olp  # len(vP_) is index of current vP, olp formed by comb_P()
            dolp_.append(o)  # index and olp of terminated vP is buffered at current dP

            I, D, Dy, M, My, Vq, p_, olp, olp_, dolp = 0,0,0,0,0,0,[],0,[],0  # init. vP and dolp

        pri_s = s   # vP (representing span of same-sign vq s) is incremented:
        olp += 1    # overlap to current dP
        I += pri_p  # p s summed within vP
        D += d; Dy += dy  # lat D for vertical vP comp, + vert Dy for P2 orient adjust eval and gradient
        M += m; My += my  # lateral and vertical summation within vP and vP2
        Vq += fv  # fvs summed to define vP value, but directional res.loss for orient eval
        p_.append(p) # pri = pri_p, fd, fv: same-line prior quadrant, buffered for selective inc_rng comp


        # formation of difference pattern dP: horizontal span of same-sign dq s with associated vars:

        sd = 1 if d > 0 else 0  # sd: positive sign of d;
        pri_sd, Id, Dd, Ddy, Md, Mdy, Dq, d_, dolp = dP  # dP tuple

        if x > r + 2 and (sd != pri_sd or x == X - 1):  # if dq sign miss or line ends, dP is terminated

            if y > 1:
               n = len(dP_)
               comb_P(dP, _dP_, _x, x, y, Y, n)  # or comb_vP and comb_dP, with dP_.append(dP)

            o = len(dP_), dolp  # len(dP_) is index of current dP, dolp formed by comb_P()
            olp_.append(o)  # index and dolp of terminated dP is buffered at current vP

            Id, Dd, Ddy, Md, Mdy, Dq, d_, dolp , dolp_, olp = 0,0,0,0,0,0,[],0,[],0  # init. dP and olp

        pri_sd = sd  # dP (representing span of same-sign dq s) is incremented:
        dolp += 1  # overlap to current vP
        Id += pri_p  # p s summed within dP
        Dd += d; Ddy += dy  # lateral and vertical summation within dP and dP2
        Md += m; Mdy += my  # lateral and vertical summation within dP and dP2
        Dq += fd  # fds summed to define dP value, for cons_P2 and level 2 eval
        d_.append(fd)  # same fds as in p_ but within dP, to spec vert.Dd comp, no associated derivatives

        dP = pri_sd, Id, Dd, Ddy, Md, Mdy, Dq, d_, dolp, dolp_
        vP = pri_s, I, D, Dy, M, My, Vq, p_, olp, olp_

        pri_p = _p  # for laterally-next p' ycomp() inclusion into vP and dP

    return vP_, dP_  # or vCP_, dCP_ formed by comb_P and then cons_P2?


def comb_P(P, _P_, _x, x, y, Y, n):  # _x of last _P displaced from _P_ by last comb_P(), initially = 0

    x_buff_, y_buff_, CP_, _n = [],[],[],0  # output arrays and template (prior comparand) counter
    root_, _fork_, Cfork_ = [],[],[]  # arrays of same-sign lower- or higher- line Ps

    W, I2, D2, Dy2, M2, My2, Q2, P_ = 0,0,0,0,0,0,0,[]  # variables of root P2, spec if len(P_) > 1
    WC, IC, DC, MC, DyC, MyC, QC, P2_ = 0,0,0,0,0,0,0,[]  # variables of CP: connected P2s, per Cfork_

    # olp2_: adjusted and preserved post eval, for hLe eval?

    s, I, D, Dy, M, My, Q, r, e_, olp_ = P  # M vs. V: eval per quadrant only, V = M - 2a * W?
    w = len(e_); ix = x - w  # w: width, ix: initial coordinate of a P

    while x >= _x:  # horizontal overlap between P and next _P, forks don't overlap but full P2s do?

        _P = _P_.pop(); _n += 1  # to sync with Cfork_, better than len(in_P_) - len(_P_)?
        _s, _ix, _x, _w, _I, _D, _Dy, _M, _My, _Q, _r, _e_, _olp_, _root_ = _P

        # _root_ to trace redundancy of adjusted olp_, buffered _fork_ s trace connections within CP,
        # no __fork_: sequential _fork_ s access?

        if s == _s:  # accumulation of P2, ~dP?

            root_.append(len(_P_))  # index of connected _P within _P_
            _fork_.append(n)  # future index of connected P within y_buff_, which is not formed yet

            dx = x - w/2 - _x - _w/2  # mx = mean_dx - dx, signed, or unsigned overlap?
            dw = w -_w; mw = min(w, _w)  # orientation if difference decr / match incr for min.1D Ps over max.2D:

            ''' 
            comp(dx), comp(dw, ddx) at dxP term?
            if match(dw, ddx) > a: _w *= cos(ddx); comp(w, _w)  # proj w*cos match, if same-sign dw, ddx, not |.|

            comp of separate lateral D and M
            default div and overlap eval per P2? not CP: sparse coverage?

            if mx+mw > a: # input vars norm and comp, also at P2 term: rotation if match (-DS, Ddx), div_comp if rw?  

            comp (dw, ddx) -> m_dw_ddx # to angle-normalize S vars for comp:

            if m_dw_ddx > a: _S /= cos (ddx)

            if dw > a: div_comp (w) -> rw # to width-normalize S vars for comp: 

                if rw > a: pn = I/w; dn = D/w; vn = V/w; 

                    comp (_n) # or default norm for redun assign, but comp (S) if low rw?

                    if d_n > a: div_comp (_n) -> r_n # or if d_n * rw > a: combined div_comp eval: ext, int co-variance?

            else: comp (S) # even if norm for redun assign?
            
            if len(__fork_) == 1 and len(_fork_) == 1:  # then _P includes trunk P2?
            '''

        # P2 accumulation per fork, vars *= overlap ratio?  including P comp derivatives

        W +=_w; I2 +=_I; D2 +=_D; Dy2 +=_Dy; M2 +=_M; My2 +=_My; Q2 += Q; P_.append(_P)


        if _x <= ix:  # _P output if no horizontal overlap between _P and next P:

            P2 = W, I2, D2, Dy2, M2, My2, Q2, P_  # for output?
            Cfork_.append(_fork_)  # all continuing _Ps of CP

            if (len(_fork_) == 0 and y > r + 3) or y == Y - 1:  # no continuation for current _P and its P2:

                if t > 0: cons_P2(P2)  # eval for rotation, re-scan, re-comp
                WC += W; IC += I2; DC += D2; DyC += Dy2; MC += M2; MyC += My2; QC += Q2; P2_.append(P2) # CP vars

            else:
                _P = _s, _ix, _x, _w, _I, _D, _Dy, _M, _My, Q, _r, _e_, _olp_, _fork_, _root_, P2
                # old _root_, new _fork_, old _fork_ is displaced with old P2?
                x_buff_.append(_P)  # _P is re-inputted for next-P comp

            CP = Cfork_, WC, IC, DC, DyC, MC, MyC, QC, P2_

            if (len(Cfork_) == 0 and y > r + 3) or y == Y - 1:  # no continuation per CP2:

                cons_P2(CP)  # eval for rotation, re-scan, cross-comp of P2_? also sum per frame?

            elif n == len(Cfork_):  # CP_ to _P_ sync for P2 inclusion and cons(CP) trigger by Cfork_?
            # or last fork in Cfork_, possibly disc.?

                CP_.append(CP)

    P = s, w, I, D, Dy, M, My, Q, r, e_, olp_, root_  # each root is new, includes P2 if unique cont:
    y_buff_.append(P)  # _P_ = for next line comp, if no horizontal overlap between P and next _P

    _P_.reverse(); _P_ += x_buff_; _P_.reverse() # front concat for next P comp


def cons_P2(P): # at CP2 term, sub-level 4?

    # rrdn = 1 + rdn_w / len(e_)  # redundancy rate / w, -> P Sum value, orthogonal but predictive
    # S = 1 if abs(D) + V + a * len(e_) > rrdn * aS else 0  # rep M = a*w, bi v!V, rdn I?

    mean_dx = 1  # fractional?
    dx = Dx / H
    if dx > a: comp(abs(dx))  # or if dxP Dx: fixed ddx cost?  comp of same-sign dx only

    vx = mean_dx - dx  # normalized compression of distance: min. cost decrease, not min. benefit?


def Le1(f): # last "_" denotes array vs. element, first "_" denotes higher-line array, pattern, variable

    r = 1; a = 127  # feedback filters
    Y, X = f.shape  # Y: frame height, X: frame width

    fd, fv, _x, y, vP_, dP_, _vP_, _dP_, F_  = 0,0,0,0,[],[],[],[],[]

    I, D, Dy, M, My, Vq, p_, olp, olp_ = 0,0,0,0,0,0,[],0,[]
    vP = I, D, Dy, M, My, Vq, p_, olp, olp_
    Id, Dd, Ddy, Md, Mdy, Dq, d_, dolp, dolp_ = 0,0,0,0,0,0,[],0,[]
    dP = Id, Dd, Ddy, Md, Mdy, Dq, d_, dolp, dolp_

    p_ = f[0, :]  # y is index of new line ip_
    _t_= comp(p_, X)  # _t_ includes ycomp() results: My, Dy, Vq, initialized = 0?

    for y in range(1, Y):

        p_ = f[y, :]  # y is index of new line ip_
        t_ = comp(p_, X)
        ycomp(t_, _t_, fd, fv, _x, y, X, Y, a, r, vP, dP, vP_, dP_, _vP_, _dP_)
        # comb_P() and cons_P2() are triggered by P2 ) CP termination within ycomp()
        _t_ = t_

    F_.append(P_)  # line of patterns is added to frame of patterns, y = len(F_)

    return F_  # output to level 2

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
Le1(f)

