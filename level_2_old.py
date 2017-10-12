
FP_, H = Le1(Fp_)  # input frame of 1D patterns: vP or dP

'''

Level 2:

Clustering (patterning) and selective cross-comparison between vertically adjacent
and horizontally overlapping 1D patterns, formed by level 1.
These patterns are interlaced and overlapping dPs and vPs, clustered and
cross-compared separately, forming 2D patterns (blobs).

'''

def cont_ycomp(t_, _t_, _vP_, _dP_):  # non-fuzzy vertical comparison between pixels

    vP_, dP_, valt_, dalt_ = [],[],[],[]  # append by form_P, alt_-> alt2_, packed in scan_P_

    vP = 0,0,0,0,0,0,0,0,[]  # pri_s, I, D, Dy, M, My, G, rdn_olp, e_
    dP = 0,0,0,0,0,0,0,0,[]  # pri_s, I, D, Dy, M, My, G, rdn_olp, e_

    olp = 0,0,0  # olp_len, olp_vG, olp_dG: common for current vP and dP
    x = 0

    for t, _t in zip(t_, _t_):  # compares vertically consecutive pixels, forms quadrant gradients

        x += 1
        p, d, m = t
        _p, _d, _m = _t

        # non-fuzzy pixel comparison:

        dy = p - _p  # vertical difference between pixels, summed -> Dy
        dg = _d + dy  # gradient of difference, formed at prior-level pixel _p, -> dG: variation eval?

        my = min(p, _p)   # vertical match between pixels, summed -> My
        vg = _m + my - ave  # gradient of predictive value (relative match) at prior-level _p, -> vG

        t2 = p, d, dy, m, my  # 2D tuple, -> type-specific g, _g, all accumulated in P:

        # form 1D value pattern vP: horizontal span of same-sign vg s with associated vars:

        sv, olp, valt_, dalt_, vP, dP, vP_, _vP_ = \
        form_P(1, t2, vg, dg, olp, valt_, dalt_, vP, dP, vP_, _vP_, x)

        # form 1D difference pattern dP: horizontal span of same-sign dg s + associated vars:

        sd, olp, dalt_, valt_, dP, vP, dP_, _dP_ = \
        form_P(0, t2, dg, vg, olp, dalt_, valt_, dP, vP, dP_, _dP_, x)

    # level ends, olp term, vP term, dP term, no init, inclusion per incomplete lateral fd and fm:

    dalt_.append(olp); valt_.append(olp)  # if any?
    olp_len, olp_vG, olp_dG = olp  # same for vP and dP;     incomplete vg - ave / (rng / X-x)?

    if olp_vG > olp_dG:  # comp of olp_vG to olp_dG, == goes to alt_P or to vP: primary?
       vP[7] += olp_len  # olp_len is added to redundant overlap of lesser-oG-  vP or dP
    else:
       dP[7] += olp_len  # no P[8] /= P[7]: rolp = rdn_olp / len(e_): P to alt_Ps rdn ratio

    if y+1 > rng:

        vP_, _vP_ = scan_P_(0, vP, valt_, vP_, _vP_, x)  # empty _vP_ []
        dP_, _dP_ = scan_P_(1, dP, dalt_, dP_, _dP_, x)  # empty _dP_ []

    return vP_, dP_  # extended for P_ append in scan_P_, renamed as arguments _vP_, _dP_


def Le2(FP_, H,):  # also a, aV, aD, *= Le2-specific multiples?

    FP2_, P2_, P2 = ([],[],{})  # output frame, level, and 2D patterns: vP2 or dP2

    for y in range(H):

        ivP_, idP_ = FP_[y, :]  # y is index of new level iP_
        vP_= overlap(ivP_)  # forms rdn_w to define S: filter *= 1 + w / rdn_w
        dP_= overlap(idP_)

        if y > 0: # patterning vertically adjacent and horizontal aligned iPs into sPs and SPs

           _vP_ = patt(vP_, _vP_); patt(vP_, _vP_)  # vP_s -> vsP_ and vSP_
           _dP_ = patt(dP_, _dP_); patt(dP_, _dP_)  # dP_s -> dsP_ and dSP_

        else: _vP_= vP_; _dP_= dP_  # prior level initialization, y>1 lines get _P_ from patt()

        FP2_.append(P2_)  # or FP2_[y] = P2_: level of patterns P2_[n] is outputted to FP2_[y]

    return FP2_  # frame of 2D patterns (vP2 or dP2) is outputted to level 3


def overlap(iP_):  # computes ix, x, rdn_w: total width of stronger alt.Ps overlap to P

    x, _rdn_w, rdn_w, P_ = (0, 0, 0, [])

    for i in range(len(iP_)):  # row of Le1 patterns: dPs or vPs, min_r or r per root e_?

        s, p, I, d, D, v, V, r, e_, alt_ = iP_[i]
        ix = x  # first x coordinate of P, computed on Le2 to avoid transfer
        x += len(e_)  # last x coordinate of P

        for ii in range(len(alt_)):  # alternative patterns' buffer

            _i, ow = alt_[ii]
            _s, _p, _I, _d, _D, _v, _V, _r, _e_, _alt_ = iP_[_i]  # if present: full frame input?

            if abs(d) + v > abs(_d) + _v:  # relative evaluation, unilateral ave v, p is redundant to d,v
                _rdn_w += ow  # alternative overlap (redundancy) assignment
            else:
                rdn_w += ow

            P = rdn_w, s, ix, x, p, I, d, D, v, V, r, e_
            P_.append(P)  # full frame process before pattern()

    return P_  # overlap() adds rdn_w, ix, x, to each P


def patt(P_, _P_):  # patterning (clustering) vertically adjacent and horizontally overlapping vPs or dPs
                    # first "_" denotes array, pattern, or variable from higher level

    P, _P, sP_, SP_, _sP_, _SP_, dP2_, vP2_, buff_, term_, next_ = ({},{},[],[],[],[],[],[],[],[],[])
    _x = 0; a = 127; aS = 511; aSS = 2047  # Le1_a * aw: vert patt() and comp() cost, A = a*r: vert comp range?

    sw, sI, sD, sV, sn = (0, 0, 0, 0, 0)
    Sw, SI, SD, SV, Sn, S = (0, 0, 0, 0, 0, 0)  # also at the end of while?

    for i in range(len(P_)):

        P = P_.pop()  # after P_ order reversal?
        rdn_w, s, ix, x, p, I, d, D, v, V, r, e_ = P

        while x >_x: # horizontal overlap between P and next _P, initially from overlap() in Le2

            _P, _sP_, _SP_ = _P_.pop()  # SPs may include dP2_ and vP2_: arrays of root P2s
            _rrdn, _s, _sn, _S, _Sn, _ix, _x, _I, _p, _D, _d, _V, _v, _r, _e_ = _P

            # _rrdn is kept for indexed overlap adjustment, and combined sP_ x SP_ rdn eval?

            rrdn = 1 + rdn_w / len(e_)  # redundancy rate / w, -> P Sum value, orthogonal but predictive
            S = 1 if abs(D) + V + a * len(e_) > rrdn * aS else 0  # rep M = a*w, bi v!V, rdn I?

            if s ==_s: # sP inc by sign (~dP,!comp), sP and SP overlap, but comp eval is combined

               sw += len(_e_); sI+=_I; sD+=_D; sV+=_V; sn += 1  # sn: number of P inclusions in _sP_
               sP = sw, sI, sD, sV, sn
               sP_.append(sP); sP_+=_sP_ # _sP_(and x?) of matching-s _P is transferred to P

            if S ==_S: # SP inc by vSum (~vP,!comp), far more selective than sP

               Sw += len(_e_); SI+=_I; SD+=_D; SV+=_V; Sn += 1 # Sn: number of P inclusions in _SP_
               SP = Sw, SI, SD, SV, Sn
               SP_.append(SP); SP_+=_SP_ # _SP_(and x?) of matching-S _P is transferred to P

            if _x <= ix: # no horizontal overlap between _P and next P

               if sn == 0 and Sn == 0: o = _P, _sP_, _SP_; term_.append(o) # terminated _P output
               elif sn == 0: o = _P, _sP_; term_.append(o)  # rdn P, separate _sP_, _SP_ inc|out
               elif Sn == 0: o = _P, _SP_; term_.append(o)  # _sP_ = []?

            else: o = _P, _sP_, _SP_; buff_.append(o)

        _P_.reverse(); _P_ += buff_; _P_.reverse() # front concat: re-input to next_P patt()?
        o = P, sP_, SP_; next_.append(o)  # prior-level _sP_ and _SP_ were transferred at match

        if sn or Sn == 0 and S and (abs(sD) + sV + a * sw) - (_sn +_Sn) * aSS > 0:

           # intra - +SP comp eval at sP or SP termination, (_sn +_Sn) * rrdn?

           comp_P(_P_)  # after _P term?

    cons(term_)  # consolidation at level end: rdn allocation within x_, rearranged as max-to-min?
                 # last inclusion per P contains x_: indices of co-including _Ps

    return next_ # re-inputted as _P_

def comp_P(_P_):
        """
        :param _P_:
        :return:
        """
        pass

def cons(term_):
        """
        :param term_:
        :return:
        """
        pass
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

    else: APs include pixels with min match over past pixel + aligned pixel of past level:

    comp_p_() form_p_dAP (); form_p_vAP ()

    term_d_vAP(); term_v_vAP): output to 3rd level if no current-level matches

likely n wd / 1 wv, strong dPs /-vP, weak dPs /+vP -> alt +vP2s, gaussian: vP edge = dP axis, vP axis = dP edge?
+|D| in vP: value of partial shorter-range dPs within wv? comp(x,!u_ax): point contrast?
no comp(P,uP): x and w miss, known I D V match while overlap

'''

