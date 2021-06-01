'''
line_PPs is a 2nd-level 1D algorithm, its input is Ps formed by the 1st-level line_patterns.
It cross-compares Ps (s, L, I, D, M, dert_, layers) and evaluates them for deeper cross-comparison.
Range or derivation of cross-comp is selectively increased if the match from prior-order cross-comp is above threshold,
search up to max rel distance +|- contrast borrow, with bi-directional selection?
-
comp s: if same-sign, cross-sign comp is borrow, also default L and M (core param) comp?
comp (L, I, D, M): equal-weight, select redundant I | (D,M),  div L if V_var * D_vars, and same-sign d_vars?
comp (dert_):  lower composition than layers, if any
comp (layers):  same-derivation elements
comp (P_):  sub patterns
-
Increment of 2nd level alg over 1st level alg should be made recursive, forming relative-level meta-algorithm.
Comparison distance is extended to first match or maximal accumulated miss over compared derPs, measured by roL*roM?
Match or miss may be between Ps of either sign, but comparison of lower P layers is conditional on higher-layer match
-
Comparison between two Ps is of variable-depth P hierarchy, with sign at the top, until max higher-layer miss.
This is vertical induction: results of higher-layer comparison predict results of next-layer comparison,
similar to lateral induction: variable-range comparison among Ps, until first match or max prior-Ps miss.
-
Resulting PPs will be more like 1D graphs, with explicit distances between nearest element Ps.
This is different from 1st level connectivity clustering, where all distances between nearest elements = 1.
Negative PPms and PPds are not formed: contrast value can't be assumed because all Ps are terminated.

Still, form PPd for strong Pm s?
also deeper backward search for strong Pm s?
'''

import numpy as np
from line_patterns import *
from class_cluster import ClusterStructure, NoneType

class CderP(CP):
    sign = bool
    mP = int
    neg_M = int
    neg_L = int
    P = object

class CPP(CderP):
    P_ = list

ave = 100  # ave dI -> mI, * coef / var type
'''
no ave_mP: deviation computed via rM  # ave_mP = ave*3: comp cost, or n vars per P: rep cost?
'''
ave_div = 50
ave_rM = .5  # average relative match per input magnitude, at rl=1 or .5?
ave_M = 100  # search stop
ave_sub_M = 50  # sub_H comp filter
ave_Ls = 3

def search_P_(P_):  # cross-compare patterns within horizontal line

    derP_ = []  # comp_P_ forms array of alternating-sign derPs (derivatives + P): output of pair-wise comp_P

    for i, P in enumerate(P_):
        neg_M = vmP = sign = _sign = neg_L = 0  # initialization

        for j, _P in enumerate(P_[i + 1:]):  # variable-range comp, no last-P displacement, just shifting first _P
            if P.M + neg_M > 0:  # search while net_M > ave_M * nparams or 1st _P, no selection by M sign
               # add ave_M decay with distance?

                derP, _L, _sign = comp_P(P, _P, neg_M, neg_L)
                sign, vmP, neg_M, neg_L, P = derP.sign, derP.mP, derP.neg_M, derP.neg_L, derP.P
                if sign:
                    P_[i + 1 + j].sign = True  # backward match per P: __sign = True
                    derP_.append(derP)
                    break  # nearest-neighbour search is terminated by first match
                else:
                    neg_M += vmP  # accumulate contiguous miss: negative mP
                    neg_L += _L   # accumulate distance to match
                    if j == len(P_):
                        # last P is a singleton derP, derivatives are ignored:
                        derP_.append(CderP(sign=sign or _sign, mP=vmP, neg_M=neg_M, neg_L=neg_L, P=P ))
                    '''                     
                    no contrast value in neg derPs and PPs: initial opposite-sign P miss is expected
                    neg_derP derivatives are not significant; neg_M obviates distance * decay_rate * M '''
            else:
                derP_.append(CderP(sign=sign or _sign, mP=vmP, neg_M=neg_M, neg_L=neg_L, P=P))
                # sign is ORed bilaterally, negative for singleton derPs only
                break  # neg net_M: stop search

    return derP_


def comp_P(P, _P, neg_M, neg_L):  # multi-variate cross-comp, _sign = 0 in line_patterns

    dC_ave = ave_M * ave_rM ** (1 + neg_L / P.L)  # average match projected at current distance: neg_L, add coef / var?

    dm_layer = P.comp_param(_P, ave=dC_ave)  # comp_param may need to be edited

    mP = dm_layer.I.m + dm_layer.L.m + dm_layer.M.m + dm_layer.D.m  # match(P, _P), no I: overlap, for regression to 0der-representation?
    if P.sign == _P.sign: mP *= 2  # sign is MSB, value of sign match = full magnitude match?

    sign = mP > 0
    if sign:  # positive forward match, compare sub_layers between P.sub_H and _P.sub_H:
        dert_sub_H = []  # sub hierarchy, abbreviation for new sub_layers

        if P.sub_layers and _P.sub_layers:  # not empty sub layers
            for sub_P, _sub_P in zip(P.sub_layers, _P.sub_layers):

                if P and _P:  # both forks exist
                    Ls, fdP, fid, rdn, rng, sub_P_ = sub_P[0]
                    _Ls, _fdP, _fid, _rdn, _rng, _sub_P_ = _sub_P[0]
                    # fork comparison:
                    if fdP == _fdP and rng == _rng and min(Ls, _Ls) > ave_Ls:
                        dert_sub_P_ = []
                        sub_mP = 0
                        # compare all sub_Ps to each _sub_P, form dert_sub_P per compared pair
                        for sub_P in sub_P_:  # note name recycling in nested loop
                            for _sub_P in _sub_P_:
                                dert_sub_P, _, _ = comp_P(sub_P, _sub_P, neg_M=0, neg_L=0)  # ignore _sub_L, _sub_sign?
                                sub_mP += dert_sub_P.mP  # sum sub_vmPs in derP_layer
                                dert_sub_P_.append(dert_sub_P)

                        dert_sub_H.append((fdP, fid, rdn, rng, dert_sub_P_))  # add only layers that have been compared
                        mP += sub_mP  # of compared H, no specific mP?
                        if sub_mP < ave_sub_M:
                            # potentially mH: trans-layer induction?
                            break  # low vertical induction, deeper sub_layers are not compared
                    else:
                        break  # deeper P and _P sub_layers are from different intra_comp forks, not comparable?

    derP = CderP(sign=sign, mP=mP, neg_M=neg_M, neg_L=neg_L, P=P, dm_layer=dm_layer)

    return derP, _P.L, _P.sign


def form_PPm(derP_):  # cluster derPs into PPm s by mP sign, eval for div_comp per PPm

    PPm_ = []
    derP = derP_[0]
    PP_P_ = [derP.P]  # initialize PPm with first derP (positive PPms only, no contrast: miss over discontinuity is expected)

    for i, derP in enumerate(derP_, start=1):
        sign = derP.sign
        if sign != _sign:
            # terminate PPm:
            PPm_.append(CPP(sign=sign, mP=derP.mP, neg_M=derP.neg_M, neg_L=derP.neg_L, P_=derP.P_, dm_layer=derP.dm_layer))
            # initialize PPm with current derP:
            _sign, mP, neg_M, neg_L, _P, mL, dL, mI, dI, mD, dD, mM, dM = \
            derP.sign, derP.mP, derP.neg_M, derP.Neg_L, derP.P, dm_layer=derP.dm_layer
            P_ = [_P]
        else:
            # accumulate PPm with current derP:
            mP += derP.mP
            neg_M += derP.neg_M
            neg_L += derP.neg_L
            for PP_param, derP_param in zip(PP.dm_layer, derP.dm_layer):
                PP_param += derP_param  # just a draft
            P_.append(derP.P)
        _sign = sign
    # pack last PP:
    PPm_.append(CPP(sign=_sign, mP=mP, neg_M=neg_M, neg_L=neg_L, P_=P_, dm_layer=derP.dm_layer))

    return PPm_

''' 
    Each PP is evaluated for intra-processing, not per P: results must be comparable between consecutive Ps): 
    - incremental range and derivation as in line_patterns intra_P, but over multiple params, 
    - x param div_comp: if internal compression: rm * D * L, * external compression: PP.L * L-proportional coef? 
    - form_par_P if param Match | x_param Contrast: diff (D_param, ave_D_alt_params: co-derived co-vary? neg val per P, else delete?
    
    form_PPd: dP = dL + dM + dD  # -> directional PPd, equal-weight params, no rdn?  
    if comp I -> dI ~ combined d_derivatives, then project ave_d?
    
    L is summed sign: val = S val, core ! comb?  
'''

def div_comp_P(PP_):  # draft, check all PPs for x-param comp by division between element Ps
    '''
    div x param if projected div match: compression per PP, no internal range for ind eval.
    ~ (L*D + L*M) * rm: L=min, positive if same-sign L & S, proportional to both but includes fractional miss
    + PPm' DL * DS: xP difference compression, additive to x param (intra) compression: S / L -> comp rS
    also + ML * MS: redundant unless min or converted?
    vs. norm param: Var*rL-> comp norm param, simpler but diffs are not L-proportional?
    '''
    for PP in PP_:
        if PP.M / (PP.L + PP.I + abs(PP.D) + abs(PP.dM)) * (abs(PP.dL) + abs(PP.dI) + abs(PP.dD) + abs(PP.dM)) > ave_div:
            # if irM * D_vars: match rate projects der and div match,
            # div if scale invariance: comp x dVars, signed
            ''' 
            | abs(dL + dI + dD + dM): div value ~= L, Vars correlation: stability of density, opposite signs cancel-out?
            | div_comp value is match: min(dL, dI, dD, dM) * 4, | sum of pairwise mins?
            '''
            _derP = PP.derP_[0]
            # sign, vmP, neg_M, neg_L, iP, mL, dL, mI, dI, mD, dD, mM, dM = P,
            _sign, _L, _I, _D, _M, _dert_, _sub_H, __sign = _derP[4]

            for i, derP in enumerate(PP.derP_[1:]):
                sign, L, I, D, M, dert_, sub_H, _sign = derP[4]
                # DIV comp L, SUB comp (summed param * rL) -> scale-independent d, neg if cross-sign:
                rL = L / _L
                # mL = whole_rL * min_L?
                dI = I * rL - _I  # rL-normalized dI, vs. nI = dI * rL or aI = I / L
                mI = ave - abs(dI)  # I is not derived, match is inverse deviation of miss
                dD = D * rL - _D  # sum if opposite-sign
                mD = min(D, _D)   # same-sign D in dP?
                dM = M * rL - _M  # sum if opposite-sign
                mM = min(M, _M)   # - ave_rM * M?  negative if x-sign, M += adj_M + deep_M: P value before layer value?

                mP = mI + mM + mD  # match(P, _P) for derived vars, defines norm_PPm, no ndx: single, but nmx is summed
                if mP > derP[1]:
                    rrdn = 1  # added to rdn, or diff alt, olp, div rdn?
                else:
                    rrdn = 2
                if mP > ave * 3 * rrdn:
                    rvars = mP, mI, mD, mM, dI, dD, dM  # redundant vars: dPP_rdn, ndPP_rdn, assigned in each fork?
                else:
                    rvars = []
                # append rrdn and ratio variables to current derP:
                PP.derP_[i] += [rrdn, rvars]
                # P vars -> _P vars:
                _sign = sign, _L = L, _I = I, _D = D, _M = M, _dert_ = dert_, _sub_H = sub_H, __sign = _sign
                '''
                m and d from comp_rate is more accurate than comp_norm?
                rm, rd: rate value is relative? 
                
                also define Pd, if strongly directional? 
                   
                if dP > ndP: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
                else:        dPP_rdn = 1; ndPP_rdn = 0
                '''
    return PP_
