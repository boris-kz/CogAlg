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

from line_patterns import CP
from frame_2D_alg.class_cluster import ClusterStructure, comp_param, Cdm_

class CderP(ClusterStructure):

    sign = bool  # bilateral? or local?
    rrdn =int
    mP = int
    dP = int
    neg_M = int
    neg_L = int
    adj_mP = int
    P = object
    layer1 = dict  # d, m per comparand
    der_sub_H = list  # sub hierarchy of derivatives, from comp sublayers
    PP = object  # PP that derP belongs to, currently not used

class CPP(CP, CderP):

    derP_ = list  # constituents, maybe sub_PPm_
    layer1 = dict  # d,m per comparand (dict is not inherited automatically)

ave = 100  # ave dI -> mI, * coef / var type
# no ave_mP: deviation computed via rM  # ave_mP = ave* n_comp_params: comp cost, or n vars per P: rep cost?
ave_div = 50
ave_rM = .5  # average relative match per input magnitude, at rl=1 or .5?
ave_M = 100  # search stop
ave_D = 100
ave_sub_M = 500  # sub_H comp filter
ave_Ls = 3
ave_PPM = 200
ave_merge = -50  # merge adjacent Ps
ave_inv = 20 # ave inverse m, change to Ave from the root intra_blob?
ave_min = 5  # ave direct m, change to Ave_min from the root intra_blob?

layer0_rdn = {'L': .25, 'I': .5, 'D': .25, 'M': .5}  # M is doubled because it represents both comparands


def search(P_):  # cross-compare patterns within horizontal line

    sub_search_recursive(P_, fderP=0)  # search in sublayers first: proceed with incremental distance

    # search in P_:
    derP_ = []  # search forms array of derPs (P + P'derivatives): combined output of pair-wise comp_P
    derP_d_ = []; PPm_ = []; PPd_ = []; remove_index = []

    for i, _P in enumerate(P_):
        if i not in remove_index:
            neg_M = neg_L = mP = sign = dP =_smP = 0  # initialization

            for j, P in enumerate(P_[i + 1:], start=1):  # j starts at 1, variable-range comp, no last-P displace, just shifting first _P
                if (j+i) not in remove_index:
                    if _P.M + neg_M > 0:  # search while net_M > ave_M * nparams or 1st P, no selection by M sign
                        # P.M decay with distance: * ave_rM ** (1 + neg_L / P.L): only for abs P.M?

                        derP, _L, _smP = merge_comp_P(P_, _P, P, i, j+i, neg_M, neg_L, remove_index)
                        if derP:
                            sign, mP, dP, neg_M, neg_L = derP.sign, derP.mP, derP.dP, derP.neg_M, derP.neg_L
                            derP_d_.append(derP)  # at each comp_P: induction = lend value for form_PPd_
                            if sign:
                                P_[j+i]._smP = True  # backward match per P, or set _smP in derP_ with empty CderPs?
                                derP_.append(derP)
                                derP=[]
                                break  # nearest-neighbour search is terminated by first match
                            else:
                                neg_M += mP  # accumulate contiguous P miss, or all derivatives?
                                neg_L += _L  # accumulate distance to matching P
                                '''                     
                                no contrast value in neg derPs and PPs: initial opposite-sign P miss is expected
                                neg_derP derivatives are not significant; neg_M obviates distance * decay_rate * M '''
                    else:
                        if derP:  # current _P has been compared before
                            derP.sign=sign or _smP  # sign is ORed bilaterally, negative for singleton derPs only
                            derP_.append(derP)  # vs. derP_.append( CderP(sign=sign or _smP, mP=mP,dP=dP, neg_M=neg_M, neg_L=neg_L, P=_P, layer1={}))
                        break  # neg net_M: stop search

    for index in sorted(remove_index, reverse=True):
        del P_[index]  # delete the merged Ps

    if derP_:
        PPm_ = form_PP_(derP_, fPd=False)  # cluster derPs into PPms by the sign of mP
        eval_params(PPm_)

    if len(derP_d_)>1:
        derP_d_ = form_adjacent_mP(derP_d_)
        PPd_ = form_PP_(derP_d_, fPd=True)  # cluster derP_ds into PPds by the sign of vdP
        eval_params(PPd_)

    return PPm_, PPd_


def sub_search_recursive(P_, fderP):  # search in top sublayer per P / sub_P

    for P in P_:
        if P.sublayers:
            sublayer = P.sublayers[0][0]  # top sublayer has one element
            sub_P_ = sublayer[5]
            if len(sub_P_) > 2:
                PM = P.M; PD = P.D
                if fderP:
                    PM += P.derP.mP; PD += P.derP.mP
                    # include match added by last search
                if P.fPd:
                    if abs(PD) > ave_D:  # better use sublayers.D|M, but we don't have it yet
                        sub_PPm_, sub_PPd_ = search(sub_P_)
                        sublayer[6].append(sub_PPm_); sublayer[7].append(sub_PPd_)
                        sub_search_recursive(sub_P_, fderP=1)  # deeper sublayers search is selective per sub_P
                elif PM > ave_M:
                    sub_PPm_, sub_PPd_ = search(sub_P_)
                    sublayer[6].append(sub_PPm_); sublayer[7].append(sub_PPd_)
                    sub_search_recursive(sub_P_, fderP=1)  # deeper sublayers search is selective per sub_P


def merge_comp_P(P_, _P, P, i, j, neg_M, neg_L, remove_index):  # multi-variate cross-comp, _smP = 0 in line_patterns

    mP = dP = 0
    layer1 = dict({'L': .0, 'I': .0, 'D': .0, 'M': .0})
    dist_coef = ave_rM ** (1 + neg_L / _P.L)  # average match projected at current distance from P: neg_L, separate for min_match, add coef / var?

    for param_name in layer1:
        if param_name == "I":
            dist_ave = ave_inv * dist_coef
        else: dist_ave = ave_min * dist_coef
        param = getattr(_P, param_name) / _P.L  # swap _ convention here, it's different in search
        _param = getattr(P, param_name) / P.L
        dm = comp_param(_param, param, [], dist_ave)
        rdn = layer0_rdn[param_name]
        mP += dm.m*rdn
        dP += dm.d*rdn
        layer1[param_name] = dm

    if neg_L == 0:
        mP += _P.dert_[0].m; dP += _P.dert_[0].d

    rel_distance = neg_L / _P.L

    if mP / max(rel_distance, 1) > ave_merge:  # merge(_P, P): splice proximate and param/L- similar Ps:
        _P.accum_from(P)
        _P.dert_+= P.dert_
        remove_index.append(j)
        if _P.fPd: _P.sign = P.D > 0
        else: P.sign = P.M > 0

        if (i-1) >=0 and (i-1) not in remove_index and (i) not in remove_index:
            derP, _L, _smP = merge_comp_P(P_, P_[i-1], _P, i-1, i, neg_M, neg_L, remove_index)  # backward re-comp_P

        elif (j+1) <= len(P_)-1 and (j+1) not in remove_index and (i) not in remove_index:
            derP, _L, _smP = merge_comp_P(P_, _P, P_[j+1], i, j+1, neg_M, neg_L, remove_index)  # forward comp_P
        else:
            derP = None

    else:  # form derP:
        derP, L, _smP = comp_P(_P, P, neg_L, neg_M)

    return derP, _P.L, _P.sign


def comp_P(_P, P, neg_L, neg_M):  # multi-variate cross-comp, _smP = 0 in line_patterns

    mP = dP = 0
    layer1 = dict({'L':.0,'I':.0,'D':.0,'M':.0})
    dist_coef = ave_rM ** (1 + neg_L / _P.L)  # average match projected at current distance:

    for param_name in layer1:
        if param_name == "I":
            dist_ave = ave_inv * dist_coef
        else:
            dist_ave = ave_min * dist_coef
        param = getattr(_P, param_name)
        _param = getattr(P, param_name)
        dm = comp_param(_param, param, [], dist_ave)
        rdn = layer0_rdn[param_name]
        mP += dm.m * rdn
        dP += dm.d * rdn
        layer1[param_name] = dm
        '''
        main comp is between summed params, with an option for div_comp, etc.
        mP -= ave_M * ave_rM ** (1 + neg_L / P.L)  # average match projected at current distance: neg_L, add coef / var?
        match(P,_P), ave_M is addition to ave? or abs for projection in search?
        '''
        if P.sign == _P.sign: mP *= 2  # sign is MSB, value of sign match = full magnitude match?
        sign = mP > 0
        if sign:  # positive forward match, compare sublayers between P.sub_H and _P.sub_H:
            comp_sublayers(_P, P, mP)

        if isinstance(_P.derP, CderP):  # derP is created in comp_sublayers
            _P.derP.sign = sign
            _P.derP.layer1 = layer1
            _P.derP.accumulate(mP=mP, neg_M=neg_M, neg_L=neg_L, P=_P)
            derP = _P.derP
        else:
            derP = CderP(sign=sign, mP=mP, neg_M=neg_M, neg_L=neg_L, P=_P, layer1=layer1)
            _P.derP = derP

    return derP, _P.L, _P.sign


def comp_sublayers(_P, P, mP):  # also add dP?

    if P.sublayers and _P.sublayers:  # not empty sub layers
        for _sub_layer, sub_layer in zip(_P.sublayers, P.sublayers):

            if _sub_layer and sub_layer:
                _Ls, _fdP, _fid, _rdn, _rng, _sub_P_, [], [] = _sub_layer[0]
                Ls, fdP, fid, rdn, rng, sub_P_, [], [] = sub_layer[0]
                # fork comparison:
                if fdP == _fdP and rng == _rng and min(Ls, _Ls) > ave_Ls:
                    der_sub_P_ = []
                    sub_mP = 0
                    # compare all sub_Ps to each _sub_P, form dert_sub_P per compared pair:
                    remove_index = []
                    for m, _sub_P in enumerate(_sub_P_):  # note name recycling in nested loop
                        for n, sub_P in enumerate(sub_P_):
                            if n not in remove_index:
                                # -1 for i, because comparing different sub_P_
                                der_sub_P, _, _ = merge_comp_P(_sub_P_, _sub_P, sub_P, -1, n, 0, 0, remove_index)
                                sub_mP += der_sub_P.mP  # sum sub_vmPs in derP_layer
                                der_sub_P_.append(der_sub_P)

                    # delete the merged sub_Ps at last
                    for index in sorted(remove_index, reverse=True):
                        del sub_P_[index]

                    if not isinstance(_P.derP, CderP): _P.derP = CderP(P=_P)  # _P had no derP
                    _P.derP.der_sub_H.append((fdP, fid, rdn, rng, der_sub_P_))  # add only layers that have been compared

                    mP += sub_mP  # of compared H, no specific mP?
                    if sub_mP < ave_sub_M:
                        # potentially mH: trans-layer induction: if mP + sub_mP < ave_sub_M: both local and global values of mP.
                        break  # low vertical induction, deeper sublayers are not compared
                else:
                    break  # deeper P and _P sublayers are from different intra_comp forks, not comparable?


def form_PP_(derP_, fPd):  # cluster derPs into PP s by derP sign

    PP_ = []
    derP = derP_[0]

    if fPd: _sign = (derP.adj_mP + derP.P.M) * abs(derP.dP) > ave  # value of difference is contrast: borrowed from co-projected match
    else:   _sign = derP.mP > 0

    PP = CPP( derP_=[derP], inherit=([derP.P,derP]) )  # initialize PP with 1st derP params
    # positive PPs only, miss over discontinuity is expected, contrast dP -> PPd: if PP.mP * abs(dP) > ave_dP: explicit borrow only?

    for derP in derP_[1:]:
        if fPd: sign = (derP.adj_mP + derP.P.M) * abs(derP.dP) > ave
        else:   sign = derP.mP > 0

        if sign != sign:  # sign != _sign: same-sign derPs in PP
            # terminate PPm:
            PP_.append(PP)
            PP = CPP( derP_=[derP], inherit=([derP.P, derP]) )  # reinitialize PPm with current derP
            derP.PP = PP  # PP that derP belongs to, not used
        else:
            PP.accum_from(derP.P)
            PP.accum_from(derP)  # accumulate PPm numerical params with same-name current derP params, exclusions?
            PP.derP_.append(derP)

        _sign = sign

    PP_.append(PP)  # pack last PP

    return PP_


def div_comp_P(PP_):  # draft, check all PPs for x-param comp by division between element Ps
    '''
    div x param if projected div match: compression per PP, no internal range for ind eval.
    ~ (L*D + L*M) * rm: L=min, positive if same-sign L & S, proportional to both but includes fractional miss
    + PPm' DL * DS: xP difference compression, additive to x param (intra) compression: S / L -> comp rS
    also + ML * MS: redundant unless min or converted?
    vs. norm param: Var*rL-> comp norm param, simpler but diffs are not L-proportional?
    '''
    for PP in PP_:
        vdP = (PP.adj_mP + PP.P.M) * abs(PP.dP) - ave_div
        if vdP > 0:
            # if irM * D_vars: match rate projects der and div match,
            # div if scale invariance: comp x dVars, signed
            ''' 
            | abs(dL + dI + dD + dM): div value ~= L, Vars correlation: stability of density, opposite signs cancel-out?
            | div_comp value is match: min(dL, dI, dD, dM) * 4, | sum of pairwise mins?
            '''
            _derP = PP.derP_[0]
            # smP, vmP, neg_M, neg_L, iP, mL, dL, mI, dI, mD, dD, mM, dM = P,
            #_sign, _L, _I, _D, _M, _dert_, _sub_H, __smP = _derP.P
            _P = _derP.P
            for i, derP in enumerate(PP.derP_[1:]):
                P = derP.P
                # DIV comp L, SUB comp (summed param * rL) -> scale-independent d, neg if cross-sign:
                rL = P.L / _P.L
                # mL = whole_rL * min_L?
                '''
                dI = I * rL - _I  # rL-normalized dI, vs. nI = dI * rL or aI = I / L
                mI = ave - abs(dI)  # I is not derived, match is inverse deviation of miss
                dD = D * rL - _D  # sum if opposite-sign
                mD = min(D, _D)   # same-sign D in dP?
                dM = M * rL - _M  # sum if opposite-sign
                mM = min(M, _M)   # - ave_rM * M?  negative if x-sign, M += adj_M + deep_M: P value before layer value?
                mP = mI + mM + mD  # match(P, _P) for derived vars, defines norm_PPm, no ndx: single, but nmx is summed
                '''
                for (param, _param) in zip([P.I, P.D, P.M], [_P.I, _P.D, _P.M]):
                    dm = comp_param(param, _param, [], ave, rL)
                    layer1.append([dm.d, dm.m])
                    mP += dm.m; dP += dm.d

                if dP > P.derP.dP:
                    ndP_rdn = 1; dP_rdn = 0  #Not sure what to do with these
                else:
                    dP_rdn = 1; ndP_rdn = 0

                if mP > derP.mP:
                    rrdn = 1  # added to rdn, or diff alt, olp, div rdn?
                else:
                    rrdn = 2
                if mP > ave * 3 * rrdn:
                    #rvars = mP, mI, mD, mM, dI, dD, dM  # redundant vars: dPP_rdn, ndPP_rdn, assigned in each fork?
                    rvars = layer1
                else:
                    rvars = []
                # append rrdn and ratio variables to current derP:
                #PP.derP_[i] += [rrdn, rvars]
                PP.derP_[i].rrdn = rrdn; PP.derP_[i].layer1 = rvars
                # P vars -> _P vars:
                _P = P
                '''
                m and d from comp_rate is more accurate than comp_norm?
                rm, rd: rate value is relative? 
                also define Pd, if strongly directional? 
                if dP > ndP: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
                else:        dPP_rdn = 1; ndPP_rdn = 0
                '''
    return PP_


def form_adjacent_mP(derP_d_):

    pri_mP = derP_d_[0].mP
    mP = derP_d_[1].mP
    derP_d_[0].adj_mP = derP_d_[1].mP

    for i, derP in enumerate(derP_d_[2:]):
        next_mP = derP.mP
        derP_d_[i+1].adj_mP = (pri_mP + next_mP)/2
        pri_mP = mP
        mP = next_mP

    return derP_d_


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

        sub_PPm_ = form_PP_(rderP_, fPd=False)  # cluster derPs into PPms by the sign of mP, same form_PPm_?

    return sub_PPm_


def eval_params(PPm_):

    for PP in PPm_:  # interae ove PPm_ in search of strong Ps
        if PP.layer1:  # how can layer1 be empty?
            for param_name in PP.layer1:

                M = PP.layer1[param_name].m - ave_M
                D = PP.layer1[param_name].d - ave_D
                if M > 0 or D > 0:
                    if M > D: D -= ave_D
                    else: M -= ave_M  # filter*2 if redundant
                    if M > ave_M: form_Pp_(PP, param_name, fPd = False)
                    if D > ave_D: form_Pp_(PP, param_name, fPd = True)


def form_Pp_(PP, param_name, fPd):

    rdn = layer0_rdn[param_name]
    if fPd: _sign = PP.derP_[0].layer1[param_name].d > 0
    else:   _sign = PP.derP_[0].layer1[param_name].m > 0

    Pp = CP(_smP=False, sign=_sign)  # both probably not needed

    for derP in PP.derP_[1:]:
        d = derP.layer1[param_name].d
        m = derP.layer1[param_name].m
        if fPd: sign = d > 0
        else:   sign = m > 0

        if sign != _sign:
            if fPd: PP.layer1[param_name].Ppd_.append(Pp)
            else:   PP.layer1[param_name].Ppm_.append(Pp)
            Pp = CP(_smP=False, sign=_sign)  # both probably not needed
        else:
            # accumulate Pp params
            Pp.L += 1
            Pp.D += d
            Pp.M += m
            Pp.d_.append(d)
            Pp.m_.append(m)

        _sign = sign

'''
    comp_P draft - Needs further changes
    
    if _P.M/_P.L > ave_aM: #ave_aM = ?
        _P.I *= 2*rdn ; _P.layer00['p_'] = [p*2*rdn for p in p_]    #rdn = ?
        #(L,D,M, d_,m_) *= 1 should remain the same
    else:
        _P.L *= 2*rdn;_P.D *= 2*rdn;_P.M *= 2*rdn
        _P.layer00['d_'] = [d*2*rdn for d in d_]
        _P.layer00['m_'] = [m*2*rdn for m in m_]
    if P.M/P.L > ave_aM: #should be computed for both _P and P?
        P.I *= 2*rdn ; P.layer00['p_'] = [p*2*rdn for p in p_]
        #(L,D,M, d_,m_) *= 1 should remain the same
    else:
        P.L *= 2*rdn;P.D *= 2*rdn;P.M *= 2*rdn
        P.layer00['d_'] = [d*2*rdn for d in d_]
        P.layer00['m_'] = [m*2*rdn for m in m_]
if i < _P.derP.ileft: _P.derP.ileft = i  # index of leftmost P that _P was compared to, for back_search_extend()
PPm_ = back_search_extend( PPm_, P_)  # evaluate for 1st P in each PP, merge with _P.derP.PP if any
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
'''