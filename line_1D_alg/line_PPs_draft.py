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

form PPd if min(PP.(M+mP), dP) > ave_dP, then sub-recursion as div_comp ) der_comp?
also deeper backward search for strong Pm s?
Sub-recursion if no template replacement by nearest-match: no strict continuity
different from range_comp in that elements can be distant, but also always positive?
'''

# add ColAlg folder to system path
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname("CogAlg"), '..')))

import numpy as np
from line_patterns import CP
from frame_2D_alg.class_cluster import ClusterStructure, NoneType, comp_param, Cdm

class CderP(ClusterStructure):

    sign = bool
    mP = int
    dP = int
    neg_M = int
    neg_L = int
    P = object
    layer1 = list  # d, m per comparand
    add_comparands = list
    ileft = int  # index of left-most _P that P was compared to in line_PPs, initialize at max X
    PP = object  # PP that derP belongs to, for merging PPs in back_search_extend

class CPP(CP, CderP):

    layer1 = list
    add_comparands = list
    sign = bool
    derP_ = list  # consituents, maybe sub_PPm_
    replace = {'ileft': (None, None)}


ave = 100  # ave dI -> mI, * coef / var type
# no ave_mP: deviation computed via rM  # ave_mP = ave* n_comp_params: comp cost, or n vars per P: rep cost?
ave_div = 50
ave_rM = .5  # average relative match per input magnitude, at rl=1 or .5?
ave_M = 100  # search stop
ave_sub_M = 50  # sub_H comp filter
ave_Ls = 3
ave_PPM = 200
base_comparands=['L', 'I', 'D', 'M']


def search(P_):  # cross-compare patterns within horizontal line

    derP_ = []  # search forms array of derPs (P + P'derivatives): combined output of pair-wise comp_P

    for i, P in enumerate(P_):
        neg_M = neg_L = sign = mP =_smP = 0  # initialization

        for j, _P in enumerate(P_[i + 1:]):  # variable-range comp, no last-P displacement, just shifting first _P
            if P.M + neg_M > 0:  # search while net_M > ave_M * nparams or 1st _P, no selection by M sign
                # P.M decay with distance: * ave_rM ** (1 + neg_L / P.L): only for abs P.M?

                derP, _L, _smP = comp_P(P, _P, neg_M, neg_L)
                if i < _P.derP.ileft: _P.derP.ileft = i  # index of leftmost P that _P was compared to, for back_search_extend()

                sign, mP, neg_M, neg_L, P = derP.sign, derP.mP, derP.neg_M, derP.neg_L, derP.P
                if sign:
                    P_[i + 1 + j]._smP = True  # backward match per P, or set _smP in derP_ with empty CderPs?
                    derP_.append(derP)
                    break  # nearest-neighbour search is terminated by first match
                else:
                    neg_M += mP  # accumulate contiguous P miss, or all derivatives?
                    neg_L += _L  # accumulate distance to matching P
                    if j == len(P_):  # needs review
                        # last P has a singleton derP
                        derP_.append( CderP(sign=sign or _smP, mP=mP, neg_M=neg_M, neg_L=neg_L, P=P, ileft=1024 ))
                    '''                     
                    no contrast value in neg derPs and PPs: initial opposite-sign P miss is expected
                    neg_derP derivatives are not significant; neg_M obviates distance * decay_rate * M '''
            else:
                derP_.append( CderP(sign=sign or _smP, mP=mP, neg_M=neg_M, neg_L=neg_L, P=P, ileft=1024 ))
                # sign is ORed bilaterally, negative for singleton derPs only
                break  # neg net_M: stop search

    PPm_ = form_PPm_(derP_)  # cluster derPs into PPms by the sign of mP

    PPm_ = back_search_extend( PPm_, P_)  # evaluate for 1st P in each PP, merge with _P.derP.PP if any

    return PPm_


def comp_P(P, _P, neg_M, neg_L):  # multi-variate cross-comp, _smP = 0 in line_patterns
    mP = dP = 0
    layer1 = []

    for (param, _param) in zip([P.I, P.L, P.D, P.M], [_P.I, _P.L, _P.D, _P.M]):
        dm = comp_param(param, _param, [], ave)
        layer1.append([dm.d, dm.m])
        mP += dm.m; dP += dm.d

    mP -= ave_M * ave_rM ** (1 + neg_L / P.L)  # average match projected at current distance: neg_L, add coef / var?
    # match(P,_P), ave_M is addition to ave? or abs for projection in search?
    if P.sign == _P.sign: mP *= 2  # sign is MSB, value of sign match = full magnitude match?

    sign = mP > 0
    if sign:  # positive forward match, compare sub_layers between P.sub_H and _P.sub_H:
        der_sub_H = []  # sub hierarchy, abbreviation for new sub_layers

        if P.sub_layers and _P.sub_layers:  # not empty sub layers
            for sub_P, _sub_P in zip(P.sub_layers, _P.sub_layers):

                if P and _P:  # both forks exist
                    Ls, fdP, fid, rdn, rng, sub_P_ = sub_P[0]
                    _Ls, _fdP, _fid, _rdn, _rng, _sub_P_ = _sub_P[0]
                    # fork comparison:
                    if fdP == _fdP and rng == _rng and min(Ls, _Ls) > ave_Ls:
                        der_sub_P_ = []
                        sub_mP = 0
                        # compare all sub_Ps to each _sub_P, form dert_sub_P per compared pair
                        for sub_P in sub_P_:  # note name recycling in nested loop
                            for _sub_P in _sub_P_:
                                der_sub_P, _, _ = comp_P(sub_P, _sub_P, neg_M=0, neg_L=0)  # ignore _sub_L, _sub_sign?
                                sub_mP += der_sub_P.mP  # sum sub_vmPs in derP_layer
                                der_sub_P_.append(der_sub_P)

                        der_sub_H.append((fdP, fid, rdn, rng, der_sub_P_))  # add only layers that have been compared
                        mP += sub_mP  # of compared H, no specific mP?
                        if sub_mP < ave_sub_M:
                            # potentially mH: trans-layer induction?
                            break  # low vertical induction, deeper sub_layers are not compared
                    else:
                        break  # deeper P and _P sub_layers are from different intra_comp forks, not comparable?

    derP = CderP(sign=sign, mP=mP, neg_M=neg_M, neg_L=neg_L, P=P, layer1=layer1, ileft=1024)

    return derP, _P.L, _P.sign


def form_PPm_(derP_):  # cluster derPs into PPm s by mP sign, eval for div_comp per PPm

    PPm_ = []
    derP = derP_[0]  # 1st derP
    PP = CPP( derP_=[derP], inherit=([derP.P],[derP]) )  # initialize PP with 1st derP params
    derP.PP = PP  # PP that derP belongs to, for merging PPs in back_search_extend
    # positive PPms only, miss over discontinuity is expected, contrast dP -> PPd: if PP.mP * abs(dP) > ave_dP: explicit borrow only?

    for i, derP in enumerate(derP_, start=1):
        if derP.sign != PP.sign:  # sign != _sign: same-sign derPs in PP
            # terminate PPm:
            PPm_.append(PP)
            PP = CPP( derP_=[derP], inherit=([derP.P],[derP]) )  # reinitialize PPm with current derP
            derP.PP = PP  # PP that derP belongs to, for merging PPs in back_search_extend
        else:
            PP.accum_from(derP.P)
            PP.accum_from(derP)  # accumulate PPm numerical params with same-name current derP params, exclusions?

    PPm_.append(PP)  # pack last PP

    return PPm_


def back_search_extend( PPm_, P_):  # each P should form derP, evaluate for the 1st P in PP, merge with _P.PP if any

    # search by PP.derP.P_[0] over P_, starting from illeft, as in forward search() but with decreasing indices.

    for PP in PPm_:
        neg_M = neg_L = mP = sign = _sign = 0
        derP = PP.derP_[0]
        illeft = derP.ileft - 1

        while illeft > 0:
            _P = P_[illeft]
            while (derP.P.M + neg_M > 0):
                _derP, _L, _sign = comp_P(derP.P, P_[illeft], neg_M, neg_L)

                if _sign:  # last mP sign is positive: P_[illeft] has PP
                    # merge derP into _P.derP:
                    _P.derP.neg_M += derP.neg_M; _P.derP.neg_L += derP.neg_L  # accumulated derivatives
                    _P.derP.mP = derP.mP; _P.derP.dP = derP.dP  # derivatives between newly adjacent Ps in PP:
                    _derP.PP.layer1 = _derP.layer1
                    # merge derP.PP into _derP.PP:
                    _derP.PP.accum_from(PP)
                    for i, dm in enumerate(PP.layer1):  # accumulate layer1
                        _derP.PP.layer1[i] += dm
                    PPm_.remove(PP)
                    break  # nearest-neighbour search is terminated by first match

                else:
                    _derP.neg_M += _derP.mP  # accumulate contiguous P miss, or all derP vars?
                    _derP.neg_L += _L   # accumulate distance to match
                    illeft -= 1

            break  # search ends without a match, no change in derP and PP.


def form_PPd(PPm_,derP_):
    '''
    Compare dPs when they have high value than ave. diff doesn't have independent value
    contrastive borrow from adjacent M i-e _PP.mP(_PP.mP+_PP.M)? How much raw value abs(dP) can borrow from adjacent match
    Criteria of forming PPd - PP.mP(PP.mP+PP.M)*abs(derP.dP) > ave ?
    terminate PPd when ?
    '''
    PPd_ = []


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
            # old:
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


def intra_PPm_(PPm_, rdn):
    '''
    Each PP is evaluated for intra-processing, non-recursive here:

    - incremental range and derivation, as in line_patterns intra_P but over multiple params,
    - x param div_comp: if internal compression: rm * D * L, * external compression: PP.L * L-proportional coef?
    - form_par_P if param Match | x_param Contrast: diff (D_param, ave_D_alt_params: co-derived co-vary? neg val per P, else delete?

    form_PPd: dP = dL + dM + dD  # -> directional PPd, equal-weight params, no rdn?
    if comp I -> dI ~ combined d_derivatives, then project ave_d?
    '''

    for PP in PPm_:
        if len(PP.P_) > 8 and PP.mP + PP.M > ave_PPM:
            # calls rnd_derP:
            sub_PPm_ = rng_search(PP.P_, (ave_M + PP.M / len(PP.P_)) / 2 * rdn)  # ave_M is average of local and global match

            return sub_PPm_


def rng_search(P_, ave):
    comb_layers = []  # if recursive only
    sub_PPm_ = []
    rderP_ = []

    for i, P in enumerate(P_):
        neg_M = vmP = sign = _smP = neg_L = 0

        for j, _P in enumerate(P_[i + 2:]):  # i+2: skip previously compared adjacent Ps, i+3 for sparse comp?
            # variable-range comp, no last-P displacement, just shifting first _P
            if P.M * (neg_L/P.L * ave_rM) + neg_M > ave:  # search while net_M > ave

                rderP, _L, _smP = comp_P(P, _P, neg_M, neg_L)
                sign, vmP, neg_M, neg_L, P = rderP.sign, rderP.mP, rderP.neg_M, rderP.neg_L, rderP.P
                if sign:
                    P_[i + 1 + j]._smP = True  # backward match per compared _P
                    rderP_.append(rderP)
                    break  # nearest-neighbour search is terminated by first match
                else:
                    neg_M += vmP  # accumulate contiguous miss: negative mP
                    neg_L += _L   # accumulate distance to match
                    if j == len(P_):
                        # last P is a singleton derP, derivatives are ignored:
                        rderP_.append(CderP(sign=sign or _smP, mP=vmP, neg_M=neg_M, neg_L=neg_L, P=P ))
            else:
                rderP_.append(CderP(sign=sign or _smP, mP=vmP, neg_M=neg_M, neg_L=neg_L, P=P))
                # sign is ORed bilaterally, negative for singleton derPs only
                break  # neg net_M: stop search

        sub_PPm_ = form_PPm_(rderP_)  # cluster derPs into PPms by the sign of mP, same form_PPm_?

    return sub_PPm_
