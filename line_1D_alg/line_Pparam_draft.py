'''
Just an initialization at this point
line_Pps is a 2nd-level 1D algorithm, its input is Ps formed by the 1st-level line_patterns.
It cross-compares Ps (s, L, I, D, M, dert_, layers) and forms Pparam_ for each compared param
'''

# add ColAlg folder to system path
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname("CogAlg"), '..')))

from line_patterns import CP
from frame_2D_alg.class_cluster import ClusterStructure, comp_param, Cdert

class CderP(ClusterStructure):

    sign = bool
    rrdn =int
    neg_M = int
    neg_L = int
    adj_mP = int
    P = object
    layer1 = dict  # dert per compared param
    der_sub_H = list  # sub hierarchy of derivatives, from comp sublayers

class Cdert(ClusterStructure):
    i = int  # input for range_comp only
    p = int  # accumulated in rng
    d = int  # accumulated in rng
    m = int  # distinct in deriv_comp only
    negL = int  # in mdert only
    negM = int  # in mdert only

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
# layer0_rdn = {'L': .25, 'I': .5, 'D': .25, 'M': .5}


def search(P_):  # cross-compare patterns within horizontal line

    sub_search_recursive(P_, fderP=0)  # search with incremental distance: first inside sublayers
    merge_P_draft(P_)  # merge I- or D- similar and weakly separated Ps

    if len(P_) > 2:  # at least 2 comparands, unpack P_:
        layer0 = {'L_': ([],.25), 'I_': ([],.5), 'D_': ([],.25), 'M_': ([],.5)}  # M is doubled because it represents both comparands
        for P in P_:
            layer0['L_'][0].append((P.L, P.L, P.x0))  # L # (2 L entry for code consistent processing later)
            layer0['I_'][0].append((P.I, P.L, P.x0))  # I
            layer0['D_'][0].append((P.D, P.L, P.x0))  # D
            layer0['M_'][0].append((P.M, P.L, P.x0))  # M

        for param in layer0:  # loop L_, I_, D_, M_
            search_param_(param)


def search_param_(param):

    ddert_, mdert_ = [], []  # line-wide (i, p, d, m)_, + (negL, negM) in mdert: variable-range search
    name = param.key
    rdn = param[1]
    param_ = param[0]
    _param, _L, _x0 = param_[0]

    for i, (param, L, x0) in enumerate(param_[1:]):
        dert = comp_param(_param, param, name, ave/rdn)  # param is compared to prior-P param

        ddert_.append(dert)
        negL = negM = 0
        comb_M = dert.m
        while comb_M > 0:
            i+=1
            ext_param, ext_L, ext_x0 = param_[i]  # extend search beyond next param
            dert = comp_param(param, ext_param, name, ave)
            comb_M = dert.m + negM - ave / (1 - (negL / (negL + L + _L)))
            negM += dert.m
            negL += ext_L

        dert.negL = negL; dert.negM = negM
        mdert_.append(dert)  # only after extended search is terminated

        _param = param

    Ppm_ = form_Pp_(mdert_)  # draft
    Ppd_ = form_Pp_(ddert_)

    return Ppm_, Ppd_

# below is not revised:

def merge_P_draft(P_):
    neg_M = neg_L = 0
    remove_index = []

    for i, _P in enumerate(P_):
        if i not in remove_index:
            for j, P in enumerate(P_[i + 1:], start=1):
                # j starts at 1, variable-range comp, no last-P displace, just shifting first _P
                if (j + i) not in remove_index:
                    # P.M decay with distance: * ave_rM ** (1 + neg_L / P.L), for abs P.M?
                    if _P.M + neg_M > 0:  # search while net_M > ave_M * nparams or 1st P, no selection by M sign

                        neg_M, neg_L, merge_val = merge_eval_draft(P_, _P, P, i, j + i, neg_M, neg_L, remove_index)
                        if sign:
                            P_[j + i]._smP = True  # backward match per P, or set _smP in derP_ with empty CderPs?
                            break  # nearest-neighbour search is terminated by 1st matching P, which latter searches as _P in line 81
                        else:
                            neg_M += mP  # accumulate contiguous P miss, or all derivatives?
                            neg_L += _L  # accumulate distance to matching P
                    else:
                        break  # neg net_M: stop search to merge

    for index in sorted(remove_index, reverse=True):
        del P_[index]  # delete the merged Ps


def merge_eval_draft(P_, _P, P, i, j, neg_M, neg_L, remove_index):  # comp to merge

    '''
    For 3 Pms, same-sign P1 and P3, opposite-sign P2, it's something like this:
    rel_proximity = abs( (M1+M3) / M2):
    rel_similarity = match( M1/L1, M3/L3) / miss: (match( M1/L1, M2/L2) + match( M3/L3, M2/L2) ) # both should be negative
    merge_value = rel_proximity * rel_similarity - ave_merge_value
'''
    mP = 0
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
            derP, _L, _smP = comp_param(P_, P_[i-1], _P, i-1, i, neg_M, neg_L, remove_index)  # backward re-comp_P

        elif (j+1) <= len(P_)-1 and (j+1) not in remove_index and (i) not in remove_index:
            derP, _L, _smP = comp_param(P_, _P, P_[j+1], i, j+1, neg_M, neg_L, remove_index)  # forward comp_P
        else:
            derP = None


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


def form_Pp_(derP_, fPd):  # this is a draft, please check

    # rdn = layer0_rdn[param_name]
    param_names = derP_[0].layer1.key  #?

    for param_name in param_names:
        if fPd:
            derP_[0].layer1.Ppd_[0] = CP(sign = derP_[0].layer1[param_name].d > 0)  # not sure this is correct?
        else:
            derP_[0].layer1.Ppm_[0] = CP(sign = derP_[0].layer1[param_name].m > 0)

    for derP in derP_:
        for param_name in param_names:  # this is not correct, we two loops
            if fPd:
                _sign = derP_[0].layer1.param_name.Ppd_[0].sign
                sign = derP.layer1[param_name].d > 0
            else:
                _sign = derP_[0].layer1.param_name.Ppm_[0].sign
                sign = derP.layer1[param_name].m > 0

            for derP in derP_[1:]:
                d = derP.layer1[param_name].d
                m = derP.layer1[param_name].m
                if fPd: sign = d > 0
                else:   sign = m > 0

            if sign != _sign:
                if fPd: derP.layer1[param_name].Ppd_.append(Pp)
                else:   derP.layer1[param_name].Ppm_.append(Pp)
                Pp = CP(sign=_sign)
            else:
                # accumulate Pp params
                Pp.L += 1
                Pp.D += d
                Pp.M += m
                Pp.d_.append(d)
                Pp.m_.append(m)

    merge_Pp_Kelvin(derP_, fPd)

def merge_Pp_Kelvin(derP_, fPd):  # merge low value Pps into negative Pp
    neg_Pp_ = []
    remove_index = []

    for i, derP in enumerate(derP_):
        for param_name in derP.layer1:
                if fPd: Pp_ = derP.layer1[param_name].Ppm_
                else:   Pp_ = derP.layer1[param_name].Ppd_

                for j, Pp in enumerate(Pp_):
                    if fPd: Pparam = Pp.M
                    else:   Pparam = Pp.D
                    if abs(Pparam) < ave:  # get low value negative Pps
                        neg_Pp_.append(Pp)
                        remove_index.append(j)
                if neg_Pp_:
                    _neg_Pp = neg_Pp_[0]
                    for i, neg_Pp in enumerate(neg_Pp_, start=1):
                        _neg_Pp.accum_from(neg_Pp)  # accumulate all neg_Pps
                    if fPd:
                        derP.layer1[param_name].Ppm_[0] = _neg_Pp  # replace first Pp in Pp_ with accumulated neg_Pp
                    else:
                        derP.layer1[param_name].Ppd_[0] = _neg_Pp
                    _neg_Pp = None

                for index in sorted(remove_index, reverse=True):  # remove all merged Ppd from original Pp_
                    if fPd:
                        del derP.layer1[param_name].Ppm_[index]
                    else:
                        del derP.layer1[param_name].Ppd_[index]
                remove_index = []

