'''
2nd level cross-compares resulting patterns Ps (s, L, I, D, M, dert_, layers) and evaluates them for deeper cross-comparison.
Depth of cross-comparison (discontinuous if generic) is increased in lower-recursion e_, then between same-recursion e_s:

comp (s): if same-sign (cross-sign comp is borrow)

    comp (L, I, D, M): equal-weight, select redundant I | (D,M),  div L if V_var * D_vars, and same-sign d_vars?
        comp (dert_):  lower composition than layers, if any
    comp (layers):  same-derivation elements
        comp (P_):  sub patterns

This 2nd level alg should be extended to a recursive meta-level algorithm
'''

ave_dI = 20
div_ave = 50
ave_Pm = 50

def comp_P(P_):

    dert_P_ = []  # array of Ps with added derivatives
    _P = P_[0]  # 1st P
    oP = P_[1]  # 1st opposite-sign P

    for i, P in enumerate(P_, start=2):
        sign, L, I, D, M, dert_, sub_H = P
        _sign, _L, _I, _D, _M, _dert_, _sub_H = _P

        roL = oP[1] / L  # relative_distance,
        # roM = oP[4] / M: contrast?
        dL = L - _L
        mL = min(L, _L)  # L: positions / sign, dderived: magnitude-proportional value
        dI = I - _I
        mI = abs(dI) - ave_dI
        dD = abs(D) - abs(_D)
        mD = min(abs(D), abs(_D))  # same-sign D in dP?
        dM = M - _M;  mM = min(M, _M)

        Pm = mL + mM + mD  # Pm *= roL * decay: contrast to global decay rate?
        ms = 1 if Pm > ave_Pm * 7 > 0 else 0  # comp cost = ave * 7, or rep cost: n vars per P?
        # in form_dPP:
        # Pd = dL + dM + dD  # -> directional dPP, equal-weight params, no rdn?
        # ds = 1 if Pd > 0 else 0

        dert_P_.append( (ms, Pm, roL, mL, dL, mI, dI, mD, dD, mM, dM, 0, [], P))  # div_f = 0, nvars = []
        _P = oP
        oP = P  # next P will have opposite sign

    return dert_P_


def form_mPP(dert_P_):  # cluster dert_Ps by Pm sign

    mPP_ = []
    for dert_P in dert_P_:

    ''' evaluation for comp by division is per PP, not per P: results must be comparable between consecutive Ps  

        if M + abs(dL + dI + dD + dM) > div_ave:  
            
            rL = L / _L  # DIV comp L, SUB comp (summed param * rL) -> scale-independent d, neg if cross-sign:
            norm_D = D * rL
            norm_dD = norm_D - _norm_D
            nmDx = min(nDx, _Dx)  # vs. nI = dI * rL or aI = I / L?
            nDy = Dy * rL;
            ndDy = nDy - _Dy;
            nmDy = min(nDy, _Dy)

            Pnm = mX + nmDx + nmDy  # defines norm_mPP, no ndx: single, but nmx is summed

            if Pm > Pnm:
                nmPP_rdn = 1; mPP_rdn = 0  # added to rdn, or diff alt, olp, div rdn?
            else:
                mPP_rdn = 1; nmPP_rdn = 0

            Pnd = ddX + ndDx + ndDy  # normalized d defines norm_dPP or ndPP

            if Pd > Pnd:
                ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
            else:
                dPP_rdn = 1; ndPP_rdn = 0

            div_f = 1
            nvars = Pnm, nmD, mPP_rdn, nmPP_rdn, Pnd, ndDx, ndDy, dPP_rdn, ndPP_rdn

        else:
            div_f = 0  # DIV comp flag
            nvars = 0  # DIV + norm derivatives

        '''
    return mPP_
