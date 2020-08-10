'''
line_PPs is a 2nd level 1D algorithm.

It cross-compares line_patterns output Ps (s, L, I, D, M, dert_, layers) and evaluates them for deeper cross-comparison.
Depth of cross-comparison: range+ and deriv+, is increased in lower-recursion element_,
then between same-recursion element_s:

comp (s): if same-sign,
          cross-sign comp is borrow, also default L and M (core param) comp?
          discontinuous comp up to max rel distance +|- contrast borrow, with bi-directional selection?

    comp (L, I, D, M): equal-weight, select redundant I | (D,M),  div L if V_var * D_vars, and same-sign d_vars?
        comp (dert_):  lower composition than layers, if any
    comp (layers):  same-derivation elements
        comp (P_):  sub patterns

Increment of 2nd level alg over 1st level alg should be made recursive, forming relative-level meta-algorithm.

Comparison distance is extended to first match or maximal accumulated miss over compared dert_Ps, measured by roL*roM?
Match or miss may be between Ps of either sign, but comparison of lower P layers is conditional on higher-layer match

Comparison between two Ps is of variable-depth P hierarchy, with sign at the top, until max higher-layer miss.
This is vertical induction: results of higher-layer comparison predict results of next-layer comparison,
similar to lateral induction: variable-range comparison among Ps, until first match or max prior-Ps miss.

Resulting PPs will be more like graphs (in 1D), with explicit distances between nearest element Ps.
This is different from 1st level connectivity clustering, where all distances between nearest elements = 1.
'''

ave_dI = 20
div_ave = 50
ave_mP = 50
max_miss = 50

def comp_P(P_):
    dert_P_ = []  # array of alternating-sign Ps with derivatives from comp_P

    for i, P in enumerate(P_):
        sign, L, I, D, M, dert_, sub_H = P
        oL = omP = 0
        for _P in (P_[i+1 :]):  # no last-P displacement, just shifting first _P for variable-range comp
            _sign, _L, _I, _D, _M, _dert_, _sub_H = _P
            roL = oL / L  # relative distance to _P
            roM = omP / (abs(M)+1)  # relative miss or contrast between _P: also search blocker, roD for dPP

            if roL * roM > max_miss:  # accumulated over -mPs before first +mP, no select by M sign
                dL = L - _L
                mL = min(L, _L)  # L: positions / sign, derived: magnitude-proportional value
                dI = I - _I
                mI = abs(dI) - ave_dI  # not derived, match is inverse miss
                dD = abs(D) - abs(_D)
                mD = min(abs(D), abs(_D))  # same-sign D in dP?
                dM = M - _M;  mM = min(M, _M)

                mP = mL + mM + mD  # Pm *= roL * decay: contrast to global decay rate?
                ms = 1 if mP > ave_mP * 7 > 0 else 0  # comp cost = ave * 7, or rep cost: n vars per P?
                if ms:
                    # add comp over deeper layers, adjust and evaluate updated mP
                    dert_P_.append( (ms, mP, roL, roM, mL, dL, mI, dI, mD, dD, mM, dM, P))
                    break  # nearest-neighbour search is terminated by first match
                else:
                    oL += _L
                    omP += mP  # other derivatives and oP_ are not significant if neg mP, optional in dert_P?
            else:
                dert_P_.append((ms, mP, roL, roM, mL, dL, mI, dI, mD, dD, mM, dM, P))
                # at least one comp per loop, derivatives preserved if +mP only
                break  # reached maximal accumulated miss, stop search

    return dert_P_


def form_PPm(dert_P_):  # cluster dert_Ps by mP sign

    PPm_ = []
    for ms, mP, roL, roM, mL, dL, mI, dI, mD, dD, mM, dM, P in dert_P_:
        # in form_PPd:
        # dP = dL + dM + dD  # -> directional PPd, equal-weight params, no rdn?
        # ds = 1 if Pd > 0 else 0

    ''' evaluation for comp by division is per PP, not per P: results must be comparable between consecutive Ps  

        fdiv = 0, nvars = []
        if M + abs(dL + dI + dD + dM) > div_ave:  
            
            rL = L / _L  # DIV comp L, SUB comp (summed param * rL) -> scale-independent d, neg if cross-sign:
            norm_D = D * rL
            norm_dD = norm_D - _norm_D
            nmDx = min(nDx, _Dx)  # vs. nI = dI * rL or aI = I / L?
            nDy = Dy * rL;
            ndDy = nDy - _Dy;
            nmDy = min(nDy, _Dy)

            Pnm = mX + nmDx + nmDy  # defines norm_mPP, no ndx: single, but nmx is summed

            if Pm > Pnm: nmPP_rdn = 1; mPP_rdn = 0  # added to rdn, or diff alt, olp, div rdn?
            else:        mPP_rdn = 1; nmPP_rdn = 0
            Pnd = ddX + ndDx + ndDy  # normalized d defines norm_dPP or ndPP

            if Pd > Pnd: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
            else:        dPP_rdn = 1; ndPP_rdn = 0
            fdiv = 1
            nvars = Pnm, nmD, mPP_rdn, nmPP_rdn, Pnd, ndDx, ndDy, dPP_rdn, ndPP_rdn

        else:
            fdiv = 0  # DIV comp flag
            nvars = 0  # DIV + norm derivatives
        '''
    return PPm_
