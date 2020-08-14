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

Resulting PPs will be more like 1D graphs, with explicit distances between nearest element Ps.
This is different from 1st level connectivity clustering, where all distances between nearest elements = 1.
'''

ave_dI = 20
div_ave = 50
ave_mP = 20
max_miss = 1

def comp_P(P_):
    dert_P_ = []  # array of alternating-sign Ps with derivatives from comp_P

    for i, P in enumerate(P_):
        sign, L, I, D, M, dert_, sub_H = P
        neg_M = mP = smP = _smP = neg_L = 0  # initialization

        for _P in (P_[i+1 :]):  # no last-P displacement, just shifting first _P for variable-range comp
            _sign, _L, _I, _D, _M, _dert_, _sub_H = _P

            if neg_M < max_miss:  # miss accumulated while mP < max miss: search continues, no select by M sign
                dL = L - _L
                mL = min(L, _L)  # L: positions / sign, derived: magnitude-proportional value
                dI = I - _I
                mI = ave_dI - abs(dI)  # not derived, match is inverse deviation of miss
                dD = D - _D      # sum if opposite-sign
                mD = min(D, _D)  # same-sign D in dP?
                dM = M - _M      # sum if opposite-sign
                mM = min(M, _M)  # negative if opposite-sign, M += adj_M + deep_M: combined value only?

                mP = mL + mM + mD - ave_mP  # mP *= (nL/L) * decay: contrast to global decay rate?
                smP = mP > 0  # ave_mP = ave * 3: comp cost, or rep cost: n vars per P?

                if smP:  # dert_P sign is positive if match is found, else negative
                    _smP = True  # backward match
                    # add comp over deeper layers, adjust and evaluate updated mP
                    dert_P_.append( (smP, mP, neg_M, neg_L, mL, dL, mI, dI, mD, dD, mM, dM, P))
                    break  # nearest-neighbour search, terminated by first match
                else:
                    # accumulate negative-mP params:
                    neg_M += mP  # both are negative here  # roD / dPP?
                    neg_L += _L  # distance
                    '''                     
                    no contrast borrowing: initial opposite-sign P miss is expected
                    no rnM = -nmP / (abs(M) + 1): simple vs relative miss to _Ps
                    no neg dert_P derivatives: not significant, also no contrast value in neg PPms? 
                    no neg_rL = neg_L / L: distance * ave_decay is obviated by actual decay: neg_rM '''
            else:
                dert_P_.append((smP or _smP, mP, neg_M, neg_L, 0, 0, 0, 0, 0, 0, 0, 0, P))
                # smP is ORed bilaterally, distinct from mP, negative for single (weak) dert_Ps only?
                # at least one comp per loop, 0-d derivatives will be ignored in -ve dert_Ps and PPs?
                break  # > max_miss, stop search

    return dert_P_


def form_PPm(dert_P_):  # cluster dert_Ps by mP sign, positive only: no contrast in overlapping comp?
    PPm_ = []
    # initialize PPm with first dert_P:
    _smP, MP, Neg_M, Neg_L, ML, DL, MI, DI, MD, DD, MM, DM, _P = dert_P_[0]  # positive only, no contrast?
    P_ = [_P]

    for i, dert_P in enumerate(dert_P_, start=1):
        smP = dert_P[0]
        if smP != _smP:
            PPm_.append([_smP, MP, Neg_M, Neg_L, ML, DL, MI, DI, MD, DD, MM, DM, P_])
            # initialize PPm with current dert_P:
            _sm, MP, Neg_M, Neg_L, ML, DL, MI, DI, MD, DD, MM, DM, _P = dert_P[1:]
            P_ = [_P]
        else:
            # accumulate params
            sm, mP, neg_M, neg_L, mL, dL, mI, dI, mD, dD, mM, dM, P = dert_P
            MP+=mP; Neg_M+=neg_M; Neg_L+=neg_L; ML+=mL; DL+=dL; MI+=mI; DI+=dI; MD+=mD; DD+=dD; MM+=mM; DM+=dM
            P_.append(P)
        _smP = smP

    PPm_.append([_smP, MP, Neg_M, Neg_L, ML, DL, MI, DI, MD, DD, MM, DM, P_])  # pack last PP

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
