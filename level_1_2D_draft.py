from scipy import misc

'''
    Level 1 with 2D gradient: modified core algorithm of levels 1 + 2. 

    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match. 
    Minimal unit of 2D is quadrant defined by 4 pixels. 
    
    Each quadrant is represented by two unique sample derivatives: rightward and downward from its first pixel. 
    So, quadrant gradient is computed as the average of these two derivatives. 
    2D patterns are blobs of same-sign quadrant gradient: of value for vP or difference for dP.

    Level 1 has 4 steps of incremental encoding per added scan line, defined by coordinate y:

    y:   comp()    p_ array of pixels, lateral comp -> p,m,d,
    y-1: ycomp()   t_ array of tuples, vertical comp, der.comb -> 1D P,
    y-2: comb_P()  P_ array of 1D patterns, vertical comp, eval, comb -> PP ) CP
    y-3: cons_P2() P2_ array of 2D patterns, fork overlap, eval, PP or CP consolidation:
    
'''

def comp(p_, X):  # comparison of consecutive pixels in a scan line forms tuples: pixel, match, difference

    t_ = []
    pri_p = p_[0]
    t = pri_p, 0, 0  # initial _d and _m are ignored anyway?
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

    _P_, next_P_ = [],[]
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
        pri_s, I, D, Dy, M, My, Vq, p_, olp, olp_ = vP  # vP tuple, vars maybe re-assigned to dP tuple?
        dolp_ = dP[9]  # vs. dP = dP.astype([]); dolp_ = dP.pop()

        if x > r + 2 and (s != pri_s or x == X - 1):  # if vq sign miss or line ends, vP is terminated

            if y > 1:  # separate comb_vP and comb_dP?
               _P_, next_P_ = comb_P(vP, len(vP_), _vP_, A, _x, x, y, Y, _P_, next_P_)

            o = len(vP_), olp  # len(vP_) is index of current vP, olp formed by comb_P()
            dolp_.append(o)  # index and olp of terminated vP is buffered at current dP

            I, D, Dy, M, My, Vq, p_, olp, olp_, dolp = 0,0,0,0,0,0,[],0,[],0  # init. vP and dolp

        pri_s = s   # vP (representing span of same-sign vq s) is incremented:
        olp += 1    # overlap to current dP
        I += pri_p  # p s summed within vP
        D += d; Dy += dy  # lat D for vertical vP comp, + vert Dy for P2 orient adjust eval and gradient
        M += m; My += my  # lateral and vertical summation within vP and vP2
        Vq += fv    # fvs summed to define vP value, but directional res.loss for orient eval
        p_.append(p)  # pri = pri_p, fd, fv: prior same-line quadrant, buffered for selective inc_rng comp


        # formation of difference pattern dP: horizontal span of same-sign dq s with associated vars:

        sd = 1 if d > 0 else 0  # sd: positive sign of d;
        pri_sd, Id, Dd, Ddy, Md, Mdy, Dq, d_, dolp, dolp_ = dP  # dP tuple

        if x > r + 2 and (sd != pri_sd or x == X - 1):  # if dq sign miss or line ends, dP is terminated

            if y > 1:  # separate comb_vP and comb_dP?
               _P_, next_P_ = comb_P(dP, len(dP_), _dP_, A, _x, x, y, Y, _P_, next_P_)

            o = len(dP_), dolp  # len(dP_) is index of current dP, dolp formed by comb_P()
            olp_.append(o)  # index and dolp of terminated dP is buffered at current vP

            Id, Dd, Ddy, Md, Mdy, Dq, d_, dolp , dolp_, olp = 0,0,0,0,0,0,[],0,[],0  # init. dP and olp

        pri_sd = sd  # dP (representing span of same-sign dq s) is incremented:
        dolp += 1  # overlap to current vP
        Id += pri_p  # p s summed within dP
        Dd += d; Ddy += dy  # lateral and vertical summation within dP and dPP
        Md += m; Mdy += my  # lateral and vertical summation within dP and dPP
        Dq += fd  # fds summed to define dP value, for cons_P2 and level 2 eval
        d_.append(fd)  # same fds as in p_ but within dP for selective inc_der comp, no other derivatives

        dP = pri_sd, Id, Dd, Ddy, Md, Mdy, Dq, d_, dolp, dolp_
        vP = pri_s, I, D, Dy, M, My, Vq, p_, olp, olp_

        pri_p = _p  # for inclusion into vP and dP by laterally-next p' ycomp()

    return vP_, dP_  # also attached vPP_, dPP_ and vCP_, dCP_ formed by comb_P and adjusted by cons_P2

    # draft below:

def comb_P(P, nP, _P_, A, _x, x, y, Y, P_, next_P_):  # combines matching _Ps into PP, and then PPs into CP
                                                      # _x: x of _P displaced from _P_ by last comb_P

    buff_, CP_, n_P = [],[], 0  # output arrays and _P counter
    root_, _fork_, Fork_ = [],[],[]  # root_: same-sign higher Ps, fork_: same-sign lower Ps, overlapping P

    W, IP, DP, DyP, MP, MyP, QP = 0,0,0,0,0,0,0  # variables of PP (pattern of patterns), multiple per fork
    WC, IC, DC, DyC, MC, MyC, QC, PP_ = 0,0,0,0,0,0,0,[]  # variables of CP (connected PPs), at last Fork

    s, I, D, Dy, M, My, Q, r, e_, olp_ = P  # M vs. V: eval per quadrant only, V = M - 2a * W?
    w = len(e_); ix = x - w  # w: P' width, ix: P' initial coordinate

    while x >= _x:  # horizontal overlap between P and next _P

        _P = _P_.pop(); n_P += 1  # n_P is _P counter to sync Fork_ with _P_, better than len(P_) - len(_P_)?
        _s, _ix, _x, _w, _I, _D, _Dy, _M, _My, _Q, _r, _e_, _olp_, _root_ = _P

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

            root_.append(PP)  # root temporarily includes current P and its P comp derivatives, as well as prior PP

        rdn = 0; _rdn = 0  # compute rdn: number of stronger-PM roots per root of root_, vs. vars *= overlap ratio?

        while len(root_) > 0: # vs. for i in range(len(root_)): root = root_[i]

            PP = root_.pop(); PM = PP[n]

            while len(root_) > 0: # a copy?

                if PM > _PM: _rdn+=1 # PM comp, neg v count -> rdn for PP incl eval:
                else: rdn+=1

        if PM > A*10 * rdn:  # PP inclusion by combined-P match value

            W +=_w; IP +=_I; DP +=_D; DyP +=_Dy; MP +=_M; MyP +=_My; QP += Q; P_.append(_P)
            PP = W, IP, DP, DyP, MP, MyP, QP, P_  # also olP_: concat of roots? no rolP_: same as root_

            root = len(_P_), PP; root_.append(root)  # _P index and PP per root, possibly multiple roots per P
            _fork_.append(n_P)  # index of connected P in future next_P_, to be buffered in Fork_ of CP

        if _x <= ix:  # _P output if no horizontal overlap between _P and next P:

            PP = W, IP, DP, DyP, MP, MyP, QP, P_  # PP per _root in _root_
            Fork_ += _fork_  # all continuing _Ps of CP, attached to its first fork _P: CP flag?

            if (len(_fork_) == 0 and y > r + 3) or y == Y - 1:  # no continuation per _P, term of PP, accum of CP:

                cons_P2(PP)  # term' PP eval for rotation, re-scan, re-comp, recursion, accumulation per _root PP?
                # then _root_ eval at adjusted redundancy?
                WC += W; IC += IP; DC += DP; DyC += DyP; MC += MP; MyC += MyP; QC += QP; PP_.append(PP)  # CP vars

            else:
                _P = _s, _ix, _x, _w, _I, _D, _Dy, _M, _My, _Q, _r, _e_, _olp_, _fork_, _root_  # PP per root
                # old _root_, new _fork_. old _fork_ is displaced with old _P?
                buff_.append(_P)  # _P is re-inputted for next-P comp

            CP = WC, IC, DC, DyC, MC, MyC, QC, PP_, Fork_

            if (len(Fork_) == 0 and y > r + 3) or y == Y - 1:  # no continuation per CP:

                cons_P2(CP)  # eval for rotation, re-scan, cross-comp of P2_? also sum per frame?

            elif nP == last_Fork_nP:  # CP_ to _P_ sync for PP inclusion and cons(CP) trigger by Fork_' last _P?
                CP_.append(CP)

    P = s, w, I, D, Dy, M, My, Q, r, e_, olp_, root_  # each root is new, includes P2 if unique cont:
    next_P_.append(P)  # _P_ = for next line comp, if no horizontal overlap between P and next _P

    _P_.reverse(); _P_ += buff_; _P_.reverse() # front concat for next-P comp_P()

    return _P_, next_P_


def cons_P2(P2):  # sub-level 4: eval for rotation, re-scan, re-comp, recursion, accumulation, at PP or CP term

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

    pri_s, I, D, Dy, M, My, Vq, p_, olp, olp_ = 0,0,0,0,0,0,0,[],0,[]
    vP = pri_s, I, D, Dy, M, My, Vq, p_, olp, olp_
    pri_sd, Id, Dd, Ddy, Md, Mdy, Dq, d_, dolp, dolp_ = 0,0,0,0,0,0,0,[],0,[]
    dP = pri_sd, Id, Dd, Ddy, Md, Mdy, Dq, d_, dolp, dolp_

    p_ = f[0, :]  # y is index of new line ip_
    _t_= comp(p_, X)  # _t_ includes ycomp() results: My, Dy, Vq, initialized = 0?

    for y in range(1, Y):

        p_ = f[y, :]  # y is index of new line ip_
        t_ = comp(p_, X)
        vP_, dP_ = ycomp(t_, _t_, fd, fv, _x, y, X, Y, a, r, vP, dP, vP_, dP_, _vP_, _dP_)
        # comb_P() and cons_P2() are triggered by PP ) CP termination within ycomp()
        _t_ = t_

    P_ = vP_, dP_
    F_.append(P_)  # line of patterns is added to frame of patterns, y = len(F_)

    return F_  # output to level 2

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
Le1(f)

