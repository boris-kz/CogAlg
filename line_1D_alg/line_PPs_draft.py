'''
line_PPs is a 2nd-level 1D algorithm, its input is Ps formed by the 1st-level line_patterns.
It cross-compares P params (initially L, I, D, M) and forms Pparam_ for each of them
'''

import sys  # add CogAlg folder to system path
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname("CogAlg"), '..')))

from line_patterns import *
from frame_2D_alg.class_cluster import ClusterStructure, comp_param


class Cpdert(ClusterStructure):
    # P param dert
    i = int  # input for range_comp only
    p = int  # accumulated in rng
    d = int  # accumulated in rng
    m = int  # distinct in deriv_comp only
    negL = int  # in mdert only
    negM = int  # in mdert only
    x0 = int  # pixel-level
    L = int  # pixel-level

class CPp(CP):
    pdert_ = list
    iL = int  # length of Pp in pixels
    ix0 = int  # x starting pixel coordinate
    fPd = bool  # P is Pd if true, else Pm; also defined per layer
    negL = int  # in mdert only
    negM = int  # in mdert only
    sublayers = list

class CderPp(ClusterStructure):
    mPp = int
    dPp = int
    rrdn = int
    negM = int
    negL = int
    adj_mP = int
    _Pp = object
    Pp = object
    layer1 = dict  # dert per compared param
    der_sub_H = list  # sub hierarchy of derivatives, from comp sublayers

class CPP(CPp):
    layer1 = dict
    derPp_ = list


ave = 100  # ave dI -> mI, * coef / var type
# no ave_mP: deviation computed via rM  # ave_mP = ave* n_comp_params: comp cost, or n vars per P: rep cost?
ave_div = 50
ave_rM = .5  # average relative match per input magnitude, at rl=1 or .5?
ave_negM = 10  # or rM?
ave_M = 100  # search stop
ave_D = 100  # search stop
ave_sub_M = 500  # sub_H comp filter
ave_Ls = 3
ave_PPM = 200
ave_splice = 50  # merge a kernel of 3 adjacent Pps
ave_inv = 20 # ave inverse m, change to Ave from the root intra_blob?
ave_min = 5  # ave direct m, change to Ave_min from the root intra_blob?
ave_rolp = .5  # ave overlap ratio for comp_Pp


def search(P_):  # cross-compare patterns within horizontal line

    # sub_search_recursive(P_, fderP=0)  # search with incremental distance: first inside sublayers
    PP_ = []
    layer0 = {'L_': [[],.25], 'I_': [[],.5], 'D_': [[],.25], 'M_': [[],.5]}  # M is doubled because it represents both comparands
    if len(P_) > 1:
        # at least 2 comparands, unpack P_:
        for P in P_:
            layer0['L_'][0].append((P.L, P.L, P.x0))  # L: (2 Ls for code-consistent processing later)
            layer0['I_'][0].append((P.I, P.L, P.x0))  # I
            layer0['D_'][0].append((P.D, P.L, P.x0))  # D
            layer0['M_'][0].append((P.M, P.L, P.x0))  # M

        for param_name in layer0:  # loop L_, I_, D_, M_
            search_param_(param_name, layer0[param_name])  # layer0[param_name][0].append((Ppm_, Ppd_))

        PP_ = comp_overlaps(layer0, fPd=0)  # calls comp_Pp_ and form_PP_ for overlapping derPp___s

    return PP_

def search_param_(param_name, iparam):

    ddert_, mdert_ = [], []  # line-wide (i, p, d, m)_, + (negL, negM) in mdert: variable-range search
    rdn = iparam[1]  # iparam = (param_, P.L, P.x0), rdn
    param_ = iparam[0]
    _param, _L, _x0 = param_[0]

    for i, (param, L, x0) in enumerate(param_[1:], start=1):
        # param is compared to prior-P param:
        dert = comp_param(_param, param, param_name, ave/rdn)
        # or div_comp(L), norm_comp(I, D, M) -> splice or higher composition?
        # negL, negM stay 0:
        ddert_.append( Cpdert( i=dert.i, p=dert.p, d=dert.d, m=dert.m, x0=x0, L=L) )
        negL=negM=0
        comb_M = dert.m
        j = i
        while comb_M > 0 and j+1 < len(param_):
            j += 1
            ext_param, ext_L, ext_x0 = param_[j]  # extend search beyond next param
            dert = comp_param(param, ext_param, param_name, ave)
            if dert.m > 0:
                break  # 1st matching param takes over connectivity search from _param, in the next loop
            else:
                comb_M = dert.m + negM - ave_M  # adjust ave_M for relative continuity and similarity?
                negM += dert.m
                negL += ext_L
        # after extended search, if any:
        mdert_.append( Cpdert( i=dert.i, p=dert.p, d=dert.d, m=dert.m, x0=x0, L=L, negL=negL, negM=negM))
        _param = param

    Ppm_ = form_Pp_(mdert_, param_name, rdn, fPd=0)
    Ppd_ = form_Pp_(ddert_, param_name, rdn, fPd=1)

    iparam[0] = (Ppm_, Ppd_)


def form_Pp_(dert_, param_name, rdn, fPd):  # almost the same as line_patterns form_P_ for now
    # initialization:
    Pp_ = []
    x = 0
    _sign = None  # to initialize 1st P, (None != True) and (None != False) are both True

    for dert in dert_:  # segment by sign
        if fPd: sign = dert.d > 0
        else:   sign = dert.m > 0  # adjust by ave projected at distance=negL and contrast=negM, if significant:
                # m + ddist_ave = ave - ave * (ave_rM * (1 + negL / ((param.L + _param.L) / 2))) / (1 + negM / ave_negM)?
        if sign != _sign:
            # sign change, initialize P and append it to P_
            Pp = CPp(L=1, iL=dert.L, I=dert.p, D=dert.d, M=dert.m, negL=dert.negL, negM=dert.negM, x0=x, ix0=dert.x0, pdert_=[dert], sublayers=[], fPd=fPd)
            Pp_.append(Pp)  # updated with accumulation below
        else:
            # accumulate params:
            Pp.L += 1; Pp.iL += dert.L; Pp.I += dert.p; Pp.D += dert.d; Pp.M += dert.m; Pp.negL+=dert.negL; Pp.negM+=dert.negM
            Pp.pdert_ += [dert]
        x += 1
        _sign = sign

    if len(Pp_) > 2:
        splice_P_(Pp_, fPd=0)  # merge mean_I- or mean_D- similar and weakly separated Ps; move to comp_param instead?
        intra_Ppm_(Pp_, param_name, rdn, fPd)  # evaluates for sub-recursion per Pm

    return Pp_


def comp_overlaps(layer0, fPd):  # find Pps that overlap across 4 Pp_s, compute overlap ratio, call comp_Pp_ and form_PP_
    PP_ = []
    derPp____ = []  # from comp_Pp across all params
    # it seems there are 4 layers of nesting, not 3?

    for i, _param_name in enumerate(layer0): # loop 1st param
        if fPd: _Pp_ = layer0[_param_name][0][1]  # _Ppd
        else:   _Pp_ = layer0[_param_name][0][0]  # _Ppm
        start_Pp_ = [0,0,0,0]  # {'L_': 0, 'I_': 0, 'D_': 0, 'M_': 0}
        derPp___ = []  # from comp_Pp of current param' overlapping Pps

        for _Pp in _Pp_:  # Pps of current param, _Pp of 1st param may overlap with multiple Pps of the other param
            derPp__ = []  # from comp_Pp of _Pp to other params

            for j, param_name in enumerate(layer0):
                if i>j:  # Pp pair is unique: https://stackoverflow.com/questions/16691524/calculating-the-overlap-distance-of-two-1d-line-segments
                    if fPd: Pp_ = layer0[param_name][0][1]  # Ppd
                    else:   Pp_ = layer0[param_name][0][0]  # Ppm
                    derPp_ = []  # from comp_Pp of _Pp to overlapping Pps per param

                    for k, Pp in enumerate(Pp_[start_Pp_[j]:], start=start_Pp_[j]):  # check Pps of the other param for x overlap:
                        if (Pp.ix0 - 1 < (_Pp.ix0 + _Pp.iL) and (Pp.ix0 + Pp.iL) + 1 > _Pp.ix0):

                            olpL = max(0, min(_Pp.ix0+_Pp.iL, Pp.ix0+Pp.iL) - max(_Pp.ix0, Pp.ix0))  # L of overlap
                            rolp = olpL / ((_Pp.iL + Pp.iL) / 2)  # mean of Ls
                            if rolp > ave_rolp:
                                derPp = comp_Pp(_Pp, Pp, layer0)
                                derPp_.append((derPp, _param_name, param_name)) # pack derPp bottom up
                        else:
                            start_Pp_[j] = k  # next Pp starting index for current param_name
                            break  # if current Pp doesn't overlap _Pp, the next one won't either, but next _Pp overlaps current Pp

                    if derPp_:
                        derPp__.append(derPp_)  # derPp_s from comp_Pp (_Pp, other param Pp_)
            if derPp__:
                derPp___.append(derPp__)  # derPp_s from comp_Pp (_Pp, other params)
        if derPp___:
            derPp____.append(derPp___)  # derPp_s from comp_Pp (param_Pp_, other params)
    if derPp____:
        form_PP_(PP_, derPp____)  # all unique derPp_s from comp_Pp(cross-params)

    return PP_


def form_PP_(PP_, derPp____):  # unpack derPp____ top down, pack matching derPps into PPs bottom-up
    '''
    Draft: form PP from all overlapping derPps
    '''
    rdn = {'L_': .25, 'I_': .5, 'D_': .25, 'M_': .5}
    fPP = 0
    PPm = PPd = CPP()  # initialize PPs

    for _param, derPp___ in derPp____:  # from comp_Pp (across params)
        for _Pp, derPp__ in derPp___:  # from comp_Pp (param_Pp_, other params)
            for param, derPp_ in derPp__:  # from comp_Pp (_Pp, other params)
                for Pp, derPp in derPp_:  # from comp_Pp (_Pp, other param' Pp_)

                    rdn = (rdn[_param] + rdn[param]) / 2  # mean rdn pof compared params
                    if derPp.mPp * rdn > ave_M:
                        PPm.accum_from(derPp); PPm.derPp_.append(derPp)
                        fPP = 1
                    if derPp.dPp * rdn > ave_D:
                        PPd.accum_from(derPp); PPd.derPp_.append(derPp)
                        fPP = 1

        # below ia not reviewed

        for derPp__ in derPp__:  # derPp__ = list of overlapping Pp's derPps for different param_names
            for (derPp, _param_name, param_name) in derPp_: # derPp_ = list of overlapping Pp's derPps per param_name

                # mean of match  and difference based on both compared params?
                mPp = (derPp.mPp * rdn[_param_name] + derPp.mPp * rdn[param_name])/2
                dPp = (derPp.dPp * rdn[_param_name] + derPp.dPp * rdn[param_name])/2

                if mPp > ave_M and dPp > ave_D :  # ave is just a placeholder
                    fPP = 1
                    PP.accum_from(derPp)
                    PP.derPp_.append(derPp)
        if fPP: PP_.append(PP)
'''
We should get 4-layer nesting in derPp____: names ( _Pp_ ( names ( Pp_ ))), and form PP_ out of it.
PP should combine all matching overlapping Pps, with multiple matches in each dimension.

def form_PP_(PP_, derPp__):  # Draft: form PP from derPps formed from different overlapping Pps
    for derPp_ in derPp__:
        fPP = 0
        PP = CPP() # initialize 1st PP
        for derPp in derPp_:
            if derPp.mPp > -ave_M*100 and derPp.dPp > -ave_D*100:
                fPP = 1
                PP.accum_from(derPp)
                PP.derPp_.append(derPp)
        if fPP: PP_.append(PP)
'''

def comp_Pp(_Pp, Pp, layer0):
    '''
    almost same as old comp_P
    '''
    mPp = dPp = 0
    layer1 = dict({'L':.0,'I':.0,'D':.0,'M':.0})
    dist_coef = ave_rM * (1 + _Pp.negL / _Pp.L)  # average match projected at current distance

    for param_name in layer1:
        if param_name == "I":
            dist_ave = ave_inv * dist_coef
        else:
            dist_ave = ave_min * dist_coef
        param = getattr(_Pp, param_name)
        _param = getattr(Pp, param_name)
        dert = comp_param(_param, param, [], dist_ave)
        rdn = layer0[param_name+'_'][1] # index 1 =rdn
        mPp += dert.m * rdn
        dPp += dert.d * rdn
        layer1[param_name] = dert

    negM = _Pp.negM - Pp.negM
    negL = _Pp.L - Pp.negL
    '''
    options for div_comp, etc.    
    if P.sign == _P.sign: mP *= 2  # sign is MSB, value of sign match = full magnitude match?
    if mP > 0
        # positive forward match, compare sublayers between P.sub_H and _P.sub_H:
       comp_sublayers(_P, P, mP)
    if isinstance(_P.derP, CderP):  # derP is created in comp_sublayers
        _P.derP.sign = sign
        _P.derP.layer1 = layer1
        _P.derP.accumulate(mP=mP, neg_M=neg_M, neg_L=neg_L, P=_P)
        derP = _P.derP
    else:
        derP = CderP(sign=sign, mP=mP, neg_M=neg_M, neg_L=neg_L, P=_P, layer1=layer1)
        _P.derP = derP
    '''
    derPp = CderPp( mPp=mPp, dPp=dPp, negM=negM, negL=negL, _Pp=_Pp, Pp=Pp, layer1=layer1)

    return derPp

# below is not revised

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

def form_adjacent_mP(derPp_):

    pri_mP = derPp_[0].mP
    mP = derPp_[1].mP
    derPp_[0].adj_mP = derPp_[1].mP

    for i, derP in enumerate(derPp_[2:]):
        next_mP = derP.mP
        derPp_[i+1].adj_mP = (pri_mP + next_mP)/2
        pri_mP = mP
        mP = next_mP

    return derPp_

def intra_Ppm_(Pp_, param_name, rdn, fPd):
    '''
    Draft
    Each Pp is evaluated for sub-recursion: incremental range and derivation, as in line_patterns but via adjusted ave_M,
    - x param div_comp: if internal compression: rm * D * L, * external compression: PP.L * L-proportional coef?
    - form_par_P if param Match | x_param Contrast: diff (D_param, ave_D_alt_params: co-derived co-vary? neg val per P, else delete?
    form_PPd: dP = dL + dM + dD  # -> directional PPd, equal-weight params, no rdn?
    if comp I -> dI ~ combined d_derivatives, then project ave_d?
    '''
    for Pp in Pp_:

        if (Pp.L > 2) and (Pp.M > -ave_M and Pp.M / Pp.L > -ave):
            sub_param_ = []

            for pdert in Pp.pdert_:
                if fPd:
                    param_name = "D_"  # comp d
                    sub_param_.append((pdert.d, pdert.x0, pdert.L))
                else:
                    param_name = "I_"  # comp i @ local ave_M
                    sub_param_.append((pdert.i, pdert.x0, pdert.L))

            Pp.sublayers = search_param_(param_name, [sub_param_, rdn] )  # iparam = [sub_param_, rdn], where sub_param_ = (param1, x01, L1),(param2, x02, L2),...

            # add: ave_M + (Pp.M / len(Pp.P_)) / 2 * rdn: ave_M is average of local and global match
            # extended search needs to be restricted to ave_M-terminated derts