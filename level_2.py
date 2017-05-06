FP_ = Le1(Fp_) # input frame of 1D patterns: vP or dP

def Le2(FP_, H, a, aV, aD, A, AV, AD):  # or Le2-specific filters?

    FP2_, P2_, P2 = ([],[],{})  # output frame, line, and 2D patterns: vP2 or dP2

    for y in range(H):

        iP_ = FP_[y, :]  # y is index of new line iP_
        vP_, dP_= overlap(iP_)  # forms rdn_w to define S: filter *= 1 + w / rdn_w

        if y > 0: # patterning vertically adjacent and horizontal aligned iPs into sPs and SPs

           _vP_= patt(vP_,_vP_); patt(vP_,_vP_) # vP_s -> vsP_ and vSP_
           _dP_= patt(dP_,_dP_); patt(dP_,_dP_) # dP_s -> dsP_ and dSP_

        else: _vP_= vP_; _dP_= dP_ # prior line initialization, y>1 lines get _P_ from patt()

        FP2_.append(P2_)  # or FP2_[y] = P2_: line of patterns P2_[n] is outputted to FP2_[y]

    return FP2_  # frame of 2D patterns (vP2 or dP2) is outputted to level 3

def overlap(iP_): # computes rdn_w: total width of stronger alt.Ps overlap to P

    xd, xv, rdn_w, alt_P = (0, 0, 0, {})
    buff_ = []  # alt_Ps re-inputted into vP_ or dP_ for next-P overlap()
    disp_ = []  # alt_Ps outputted into ovP_ or odP_ for:
    ovP_, odP_= ([],[]) # output to patt()

    for i in range(len(iP_)):  # row of Le1 patterns: dPs or vPs, min_r or r per root e_?

        dP_, vP_ = ([],[]) # initialized empty for each line of 1D patterns, why numbers?

        type, s, w, I, D, V, r, e_ = iP_[i]  # type: flag indicating alternative vP or dP
        p = I / w; d = D / w; v = V / w  # default to eval overlap, poss. for div.comp?

        if type > 0:
            ix = xd  # first x coordinate of dP
            x = xd + w  # last x coordinate of dP
            alt_P_= vP_ # alternative-type array and pattern to compute overlap
        else:
            ix = xv  # first x coordinate of vP
            x = xv + w  # last x coordinate of vP
            alt_P_= dP_

        if len(alt_P_) > 0: # skip and alt_P_ initialization if len(alt_P_) == 0

            alt_P = alt_P_.pop() # reassigned buff_, why number? accessed only for rdn_w, x, w, d, v?
            _rdn_w, _s, _ix, _x, _w, _I, _p, _D, _d, _V, _v, _r, _e_ = alt_P # first "_" denotes alt_P var

            dx = x -_x  # always > 0
            rem_ow = w - dx  # remaining width of overlap between P and uP_

            while rem_ow > 0:
                ow = min(rem_ow, _w)  # w of overlap between P and uP

                if abs(d) + v > abs(_d) +_v:  # relative evaluation, uni ave v, no + p: redundant to d,v?
                    _rdn_w += ow  # alternative redundancy allocation
                else: rdn_w += ow

                if rem_ow > _w:
                    disp_.append(alt_P) # including overlap per uP, from forward and backward overlap() runs
                else:
                    buff_.append(alt_P) # overlap accumulated in forward (as P) and backward (as alt_P) runs

                rem_ow -= ow  # remaining overlap, possibly negative
                if rem_ow > 0:
                   alt_P = alt_P_.pop()
                   _rdn_w, _s, _ix, _x, _w, _I, _p, _D, _d, _V, _v, _r, _e_ = alt_P  # no rem_ow = w - dx

            P = rdn_w, s, ix, x, w, I, p, D, d, V, v, r, e_
            # overlap() splits iP_ into vP_ and dP_, adds rdn_w, ix, x, averages to each P

        if type > 0:
            dP_.append(P); vP_+= buff_; ovP_+= disp_
        else:
            vP_.append(P); dP_+= buff_; odP_+= disp_  # disp_ if no next P overlap, concat in right order?

    return ovP_, odP_  # vP2 are formed over complete output ovP_ or odP_:


def patt(P_, _P_): # patterning vertically adjacent and horizontal aligned vPs or dPs
                   # first "_" denotes pri_ pattern or variable

    P, _P, sP_, SP_, _sP_, _SP_, dP2_, vP2_, buff_, term_, next_ = ({},{},[],[],[],[],[],[],[],[],[])
    _x = 0; a = 127; aS = 511; aSS = 2047 # Le1_a * aw: vert patt() and comp() cost, A = a*r: vert comp range?

    sw, sI, sD, sV, sn = (0, 0, 0, 0, 0)
    Sw, SI, SD, SV, Sn, S = (0, 0, 0, 0, 0, 0) # also at the end of while?

    for i in range(len(P_)):

        P = P_.pop()
        rdn_w, s, ix, x, w, I, p, D, d, V, v, r, e_ = P

        while x >_x: # horizontal overlap between P and next _P, initially from overlap() in Le2

            _P, _sP_, _SP_ = _P_.pop() # SPs may include dP2_ and vP2_: arrays of root P2s
            _rrdn, _s, _sn, _S, _Sn, _ix, _x, _w, _I, _p, _D, _d, _V, _v, _r, _e_ = _P

            # _rrdn is kept for indexed overlap adjustment, and combined sP_ x SP_ rdn eval?

            rrdn = 1 + rdn_w / w  # redundancy rate / w, -> P Sum value, orthogonal but predictive
            S = 1 if abs(D) + V + a * w > rrdn * aS else 0  # rep M = a*w, bi v!V, rdn I?

            if s ==_s: # sP inc by sign (~dP,!comp), sP and SP overlap, but comp eval is combined

               sw+=_w; sI+=_I; sD+=_D; sV+=_V; sn += 1 # sn: number of P inclusions in _sP_
               sP = sw, sI, sD, sV, sn
               sP_.append(sP); sP_+=_sP_ # _sP_(and x?) of matching-s _P is transferred to P

            if S ==_S: # SP inc by vSum (~vP,!comp), far more selective than sP

               Sw+=_w; SI+=_I; SD+=_D; SV+=_V; Sn += 1 # Sn: number of P inclusions in _SP_
               SP = Sw, SI, SD, SV, Sn
               SP_.append(SP); SP_+=_SP_ # _SP_(and x?) of matching-S _P is transferred to P

            if _x <= ix: # no horizontal overlap between _P and next P

               if sn == 0 and Sn == 0: o = _P, _sP_, _SP_; term_.append(o) # terminated _P output
               elif sn == 0: o = _P, _sP_; term_.append(o) # rdn P, separate _sP_, _SP_ inc|out
               elif Sn == 0: o = _P, _SP_; term_.append(o) # _sP_ = []?

            else: o = _P, _sP_, _SP_; buff_.append(o)
                
        _P_ += buff_ # re-input to next_P patt()
        o = P, sP_, SP_; next_.append(o)  # prior-line _sP_ and _SP_ were transferred at match

        if sn or Sn == 0 and S and (abs(sD) + sV + a * sw) - (_sn +_Sn) * aSS > 0:
           # intra - +SP comp eval at sP or SP termination, (_sn +_Sn) * rrdn?

           comp_P(_P_)  # after _P term?

    cons(term_)  # consolidation at line end: rdn allocation within x_, rearranged as max-to-min?
                 # last inclusion per P contains x_: indices of co-including _Ps

    return next_ # re-inputted as _P_

def comp_P(_P_)

'''
    mw = min(w, _w); dw = w - _w
    ax = x - w/2; _ax = _x - _w/2  # median x, more stable than last x?

    d_ax = ax - _ax # or d_ax = x - w/2;
    m_ax = mean_d_ax - d_ax: relative compression of distance, benefit = min cost reduction?

    overlap: component of comp_p value, independent of comp_P match?:

    if d_ax < dw/2: ow = mw
    else: dw = w-_w; ow = mw - (d_ax - dw / 2)

    div_comp_vP():
        div_comp_t(); form_r_vAP (); form_rv_vAP (); // forming ratio- and ratio-value patterns

    else: comp_vP()
        comp_t (); form_d_vAP (); form_v_vAP ();

    if: mI > 2a for spec, or if I * a/ap > 2a for spliced e_ comp, before P comp?

    else: APs include pixels with min match over past pixel + aligned pixel of past line:

    comp_p_() form_p_dAP (); form_p_vAP ()

    term_d_vAP(); term_v_vAP): output to 3rd level if no current-line matches

likely n wd / 1 wv, strong dPs /-vP, weak dPs /+vP -> alt +vP2s, gaussian: vP edge = dP axis, vP axis = dP edge?
+|D| in vP: value of partial shorter-range dPs within wv? comp(x,!u_ax): point contrast?
no comp(P,uP): x and w miss, known I D V match while overlap

'''

