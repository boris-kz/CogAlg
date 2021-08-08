'''
line_PPs is a 2nd-level 1D algorithm, its input is Ps formed by the 1st-level line_patterns.
It cross-compares P params (initially L, I, D, M) and forms param_Ps (Pps) for each of them.
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
    negiL = int
    negL = int  # in mdert only
    negM = int  # in mdert only
    x0 = int  # pixel-level
    L = int  # pixel-level
    _P = object  # input P reference

class CPp(CP):
    pdert_ = list
    Rdn = int  # cross-param rdn accumulated from pderts
    rval = int  # Pp value (M | abs D) adjusted for cross-param Rdn
    iL = int  # length of Pp in pixels
    ix0 = int  # x starting pixel coordinate
    fPd = bool  # P is Pd if true, else Pm; also defined per layer
    negiL = int
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

class CPP(CPp, CderPp):
    layer1 = dict


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


def search(P_, fPd):  # cross-compare patterns within horizontal line
    # sub_search_recursive(P_, fderP=0)  # search with incremental distance: first inside sublayers

    layer0 = {'L_':[],'I_':[],'D_':[],'M_':[]}  # param_name: [params]
    if len(P_) > 1:
        # at least 2 comparands, unpack P_:
        for P in P_:
            layer0['L_'].append((P.L, P.L, P.x0))  # L: (2 Ls for code-consistent processing later)
            layer0['I_'].append((P.I, P.L, P.x0))  # I
            layer0['D_'].append((P.D, P.L, P.x0))  # D
            layer0['M_'].append((P.M, P.L, P.x0))  # M

        mdert__ = []; ddert__ = []

        for j, param_name in enumerate(layer0):  # loop L_, I_, D_, M_
            param_ = layer0[param_name]
            _par_= param_[:-1]; par_= param_[1:]  # compared vectors

            if param_name == "I_" and not fPd:  # project by D  # I=D in deriv_comp sub_Ps
                _par_[:][0] -= layer0["D_"][:-1][0] /2  # _I in (I,L,x0) is forward projected by _D in (D,L,x0)
                par_ [:][0] += layer0["D_"][1:][0] / 2  # I in (I,L,x0) is backward projected by D in (D,L,x0)

            mdert_, ddert_ = search_param_(_par_, par_, param_name)  # layer0[param_name][0].append((Ppm_, Ppd_))
            mdert__.append(mdert_)
            ddert__.append(ddert_)

        mrdn__ = sum_rdn_(layer0, mdert__, fPd=True)
        drdn__ = sum_rdn_(layer0, ddert__, fPd=False)

        for param_name, mdert_, ddert_, mrdn_, drdn_ in zip(layer0, mdert__, ddert__, mrdn__, drdn__):

            Ppm_ = form_Pp_(mdert_, param_name, mrdn_, fPd=0)
            rdn_Ppm_ = form_rdn_Pp_(Ppm_, param_name, fPd=0)
            Ppd_ = form_Pp_(ddert_, param_name, drdn_, fPd=1)
            rdn_Ppd_ = form_rdn_Pp_(Ppd_, param_name, fPd=1)

        return (rdn_Ppm_, rdn_Ppd_)
'''
next level line_PPPs:
PPm_ = search_Pp_(layer0, fPd=0)  # calls comp_Pp_ and form_PP_ per param
PPd_ = search_Pp_(layer0, fPd=1)
'''

def search_param_(_param_, param_, param_name):

    ddert_, mdert_ = [], []  # line-wide (i, p, d, m)_, + (negL, negM) in mdert: variable-range search

    for i, (_param, _L, _x0), (param, L, x0) in enumerate( zip( _param_, param_)):
        dert = comp_param(_param, param, param_name, ave)  # param is compared to prior-P _param

        # or div_comp(L), norm_comp(I, D, M): simpler than mean, but never sub_comp?
        # if I,!fPd or D, fPd: splice eval in form_rdn_Pp_
        ddert_.append( Cpdert( i=dert.i, p=dert.p, d=dert.d, m=dert.m, x0=x0, L=L) )
        negiL=negL=negM=0  # comp next only
        comb_M = dert.m
        j = i
        while comb_M > 0 and j+1 < len(param_):
            j += 1
            ext_param, ext_L, ext_x0 = param_[j]  # extend search beyond next param
            dert = comp_param(_param, ext_param, param_name, ave)
            if dert.m > 0:
                break  # 1st matching param takes over connectivity search from _param, in the next loop
            else:
                comb_M = dert.m + negM - ave_M  # adjust ave_M for relative continuity and similarity?
                negM += dert.m
                negiL += ext_L
                negL += 1
        # after extended search, if any:
        mdert_.append( Cpdert( i=dert.i, p=dert.p, d=dert.d, m=dert.m, x0=x0, L=L, negiL=negiL,negL=negL, negM=negM))

    return mdert_, ddert_


def form_Pp_(dert_, param_name, rdn_, fPd):  # almost the same as line_patterns form_P_ for now
    # initialization:
    Pp_ = []
    x = 0
    _sign = None  # to initialize 1st P, (None != True) and (None != False) are both True

    for dert, rdn in zip(dert_, rdn_):  # segment by sign
        if fPd: sign = dert.d > 0
        else:   sign = dert.m > 0  # adjust by ave projected at distance=negL and contrast=negM, if significant:
                # m + ddist_ave = ave - ave * (ave_rM * (1 + negL / ((param.L + _param.L) / 2))) / (1 + negM / ave_negM)?
        if sign != _sign:
            # sign change, initialize P and append it to P_
            Pp = CPp(L=1, iL=dert.L, I=dert.p, D=dert.d, M=dert.m, Rdn=rdn, negiL=dert.negiL, negL=dert.negL, negM=dert.negM,
                     x0=x, ix0=dert.x0, pdert_=[dert], sublayers=[], fPd=fPd)
            Pp_.append(Pp)  # updated with accumulation below
        else:
            # accumulate params:
            Pp.L += 1; Pp.iL += dert.L; Pp.I += dert.p; Pp.D += dert.d; Pp.M += dert.m; Pp.Rdn+=rdn; Pp.negiL+=dert.negiL; Pp.negL+=dert.negL; Pp.negM+=dert.negM
            Pp.pdert_ += [dert]
        x += 1
        _sign = sign

    intra_Ppm_(Pp_, param_name, fPd)  # evaluates for sub-recursion per Pm

    return Pp_

def form_rdn_Pp_(Pp_, param_name, fPd):  # cluster Pps by cross-param redundant value sign, re-evaluate them for cross-level rdn
    rPp_ = []
    x = 0
    _sign = None  # to initialize 1st rdn Pp, (None != True) and (None != False) are both True

    for Pp in Pp_:
        if fPd: Pp.rval = abs(Pp.D) - Pp.Rdn * ave_D * Pp.L
        else:   Pp.rval = Pp.M - Pp.Rdn * ave_D * Pp.L
        sign =  Pp.rval > 0
        if sign != _sign:  # sign change, initialize rPp and append it to rPp_

            rPp = CPp(L=1, iL=Pp.iL, I=Pp.I, D=Pp.D, M=Pp.M, Rdn=Pp.Rdn, rval = Pp.rval, negiL=Pp.negiL, negL=Pp.negL, negM=Pp.negM,
                      x0=x, ix0=Pp.ix0, pdert_=[Pp], sublayers=[], fPd=fPd)
                      # or rPp is sign, Pp_?
            if _sign:  # -rPps are not processed?
                for i, Pp in enumerate( rPp.pdert_ ):
                    # assign cross-level rdn, re-eval Pp and pdert_:
                    Pp_val = Pp.rval / Pp.L - ave     # / Pp.L: resolution reduction, but lower rdn:
                    pdert_val = Pp.rval - ave * Pp.L  # * Pp.L: ave cost * number of representations

                    if Pp_val > pdert_val: pdert_val -= ave * Pp.Rdn
                    else:                  Pp_val -= ave * Pp.Rdn  # ave scaled by rdn
                    if Pp_val <= 0:
                        rPp.pdert_[i] = CPp(pdert_=Pp.pdert_)  # Pp remove: reset Pp vars to 0
                    elif pdert_val <= 0:
                        if ((param_name == "I_") and not fPd) or ((param_name == "D_") and fPd):  # dert-level P-defining params
                            P=CP()
                            for pdert in Pp.pdert_:
                                P.accum_from(pdert._P, excluded=["x0"])  # different from Pp params
                                P.dert_ += [pdert._P.dert_]  # splice dert_s, eval intra_P?
                        else:
                            Pp.pdert_ = []  # pdert_ remove

            rPp_.append(rPp)  # updated with accumulation below
        else:
            # accumulate params:
            rPp.L+=1; rPp.iL+=Pp.iL; rPp.I+=Pp.I; rPp.D+=Pp.D; rPp.M+=Pp.M; rPp.Rdn+=Pp.Rdn; rPp.rval+=Pp.rval
            rPp.negiL+=Pp.negiL; rPp.negL+=Pp.negL; rPp.negM+=Pp.negM
            rPp.pdert_ += [Pp]
        x += 1
        _sign = sign

    return rPp_


def sum_rdn_(layer0, pdert__, fPd):
    '''
    access same-index pderts of all params, assign redundancy to lesser-magnitude m|d per param pair.
    if other-param same-P_-index pdert is missing, rdn doesn't change.
    '''
    if fPd: alt='M'
    else: alt='D'
    name_pairs = (('I','L'), ('I','D'), ('I','M'), ('L',alt), ('D','M'))
    pderts_Rdn = [[], [], [], []]  # L_, I_, D_, M_' Rdns, as in pdert__

    for L_dert, I_dert, D_dert, M_dert in zip(pdert__[0], pdert__[1], pdert__[2], pdert__[3]):  # there is one pdert for each _P
        rdn_pairs = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

        for rdn_pair, name_pair in zip(rdn_pairs, name_pairs):
            # assign rdn in each rdn_pair using partial name replacement: https://www.w3schools.com/python/ref_func_eval.asp
            if fPd:
                if eval("abs("+name_pair[0]+"_dert.d) >  abs("+name_pair[1]+"_dert.d)"): # (param_name)_dert.d|m
                    rdn_pair[1] = 1
                else: rdn_pair[0] = 1  # weaker pair rdn=1
            else:
                if eval(name_pair[0]+"_dert.m > " + name_pair[1] + "_dert.m"):
                    rdn_pair[1] = 1
                else: rdn_pair[0] = 1  # weaker pair rdn=1

        for i, param_name in enumerate(layer0):  # sum param rdn from all pairs it is in, flatten pair_names, pair_rdns?
            Rdn = 0
            for name_in_pair, rdn in zip(name_pairs, rdn_pairs):
                if param_name[0] == name_in_pair[0]:  # param_name = "L_", param_name[0] = "L"
                    Rdn += rdn[0]  # M*=2: represents both comparands?
                elif param_name[0] == name_in_pair[1]:
                    Rdn += rdn[1]

            pderts_Rdn[i].append(Rdn)  # same length as pdert_

    return pderts_Rdn


def intra_Ppm_(Pp_, param_name, fPd):
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
            fPd_ = []

            for pdert in Pp.pdert_:
                if fPd:
                    param_name = "D_"  # comp d
                    sub_param_.append((pdert.d, pdert.x0, pdert.L))
                    fPd_.append(1)
                else:
                    param_name = "I_"  # comp i @ local ave_M
                    sub_param_.append(((pdert.i, pdert.d), pdert.x0, pdert.L))
                    fPd_.append(0)

            Pp.sublayers = search_param_(param_name, sub_param_, fPd_)  # iparam = sub_param_: (param1, x01, L1),(param2, x02, L2),...
            # add: ave_M + (Pp.M / len(Pp.P_)) / 2 * rdn: ave_M is average of local and global match?
            # extended search needs to be restricted to ave_M-terminated derts

# below is obsolete:

def form_PP_(params_derPp____, fPd):  # Draft:
    '''
    unpack 4-layer derPp____: _names ( _Pp_ ( names ( Pp_ ))),
    pack derPps with overlapping match: sum of concurrent mPps > ave_M * rolp, into PPs of PP_
    '''
    rdn = [.25,.5,.25,.5]  # {'L_': .25, 'I_': .5, 'D_': .25, 'M_': .5}
    names = ['L_', 'I_', 'D_', 'M_']
    Rolp = 0
    PP_ = []
    _sign = None
    # init new empty derPp____ with the same list structure as params_derPp____, for [i][j][k] indexing later
    derPp____ = [[[[]  for param_derPp_   in param_derPp__] \
                       for param_derPp__  in param_derPp___] \
                       for param_derPp___ in params_derPp____]
    param_name_ = derPp____.copy()

    for i, _param_derPp___ in enumerate( params_derPp____):  # derPp___ from comp_Pp (across params)
        for j, _Pp_derPp__ in enumerate( _param_derPp___):  # from comp_Pp (param_Pp_, other params)
            for k, param_derPp_ in enumerate( _Pp_derPp__):  # from comp_Pp (_Pp, other params)
                for (derPp, rolp, _name, name) in param_derPp_:  # from comp_Pp (_Pp, other param' Pp_)
                    # debugging
                    if names[i] != _name: raise ValueError("Wrong _name")
                    if names[k] != name: raise ValueError("Wrong name")

                    if "pre_PP" not in locals(): pre_PP = CPP(derPp____=derPp____.copy())
                    # if fPd: derPp_val = derPp.dPp; ave = ave_D
                    # else:   derPp_val = derPp.mPp; ave = ave_M
                    # mean_rdn = (rdn[i] + rdn[k]) / 2  # of compared params
                    # if derPp_val * mean_rdn > ave:
                    # else: pre_PP = CPP(derPp____=derPp____.copy())
                    # accum either sign, no eval or sub_PP_ per layer:
                    Rolp += rolp
                    pre_PP.accum_from(derPp)
                    pre_PP.derPp____[i][j][k].append(derPp)
                    pre_PP.param_name_.append((names[i], names[k]))
        '''    
        We can't evaluate until the top loop because any overlap may form sufficient match. 
        Then we only define pre_PPs by overlap of any element of any layer to any other element of any other layer.
        But there are so many possible overlaps that pre_PP may never terminate.
        Another way to define them is by minimal combined-layers' match per x (iP). 
        But then we are back to immediate multi-param comp_P_, which is pointless because different derivatives anti-correlate.
         
                # inclusion into higher layer of pre_PP by the sum of concurrent mPps > ave_M * Rolp, over all lower layers:
                if "pre_PP" in locals() and pre_PP.derPp____[i][j][k] and not pre_PP.mPp > ave_M * Rolp:
                    pre_PP = CPP(derPp____=derPp____.copy())
            # pre_PP.derPp____[i][j] is a nested list, we need to check recursively to determine whether there is any appended derPp
            if "pre_PP" in locals() and not emptylist(pre_PP.derPp____[i][j]) and not pre_PP.mPp > ave_M * Rolp:
                pre_PP = CPP(derPp____=derPp____.copy())
        '''
        if "pre_PP" in locals() and not emptylist(pre_PP.derPp____[i]):
            if pre_PP.mPp > ave_M * Rolp:
                PP_.append(pre_PP)  # no negative PPs?
                _sign = True
            else:
                _sign=False
                pre_PP = CPP(derPp____=derPp____.copy())

    return PP_

# https://stackoverflow.com/questions/1593564/python-how-to-check-if-a-nested-list-is-essentially-empty
def emptylist(in_list):
    '''
    check if nested list is totally empty
    '''
    if isinstance(in_list, list): # Is a list
        return all( map(emptylist, in_list) )
    return False # Not a list

def comp_Pp(_Pp, Pp, layer0):
    '''
    almost same as old comp_P
    '''
    mPp = dPp = 0
    layer1 = dict({'L':.0,'I':.0,'D':.0,'M':.0})
    dist_coef = ave_rM * (1 + _Pp.negL / _Pp.L)
    # average match projected at current distance, needs a review
    for param_name in layer1:
        if param_name == "I":
            ave = ave_inv # * dist_coef
        else:
            ave = ave_min # * dist_coef
        param = getattr(_Pp, param_name)
        _param = getattr(Pp, param_name)
        dert = comp_param(_param, param, [], ave)
        rdn = layer0[param_name+'_'][1]  # index 1 =rdn
        mPp += dert.m * rdn
        dPp += dert.d * rdn
        layer1[param_name] = dert

    negM = _Pp.negM - Pp.negM
    negL = _Pp.L - Pp.negL
    negiL = _Pp.iL - Pp.negiL

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
    derPp = CderPp( mPp=mPp, dPp=dPp, negM=negM, negL=negL, negiL=negiL, _Pp=_Pp, Pp=Pp, layer1=layer1)

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


# to be updated
def draw_PP_(image, frame_PP_):
    # init every possible combinations
    img_mparams = {'L_I_': np.zeros_like(image), 'L_D_': np.zeros_like(image),
                   'L_M_': np.zeros_like(image), 'I_D_': np.zeros_like(image),
                   'I_M_': np.zeros_like(image), 'D_M_': np.zeros_like(image)}

    img_dparams = {'L_I_': np.zeros_like(image), 'L_D_': np.zeros_like(image),
                   'L_M_': np.zeros_like(image), 'I_D_': np.zeros_like(image),
                   'I_M_': np.zeros_like(image), 'D_M_': np.zeros_like(image)}

    for y, (PPm_, PPd_) in enumerate(frame_PP_):  # draw each line
        if PPm_ or PPd_:
            for PPm, PPd in zip_longest(PPm_, PPd_,  fillvalue=[]):

                if PPm:
                    draw_PP(img_mparams, PPm, y)
                if PPd:
                    draw_PP(img_dparams, PPd, y)
    # plot diagram of each pair PPs
    plt.figure()
    for i, param in enumerate(img_mparams):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img_mparams[param], vmin=0, vmax=255)
        plt.title("pair = " + param + " m param")

    plt.figure()
    for i, param in enumerate(img_dparams):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img_dparams[param], vmin=0, vmax=255)
        plt.title("pair = " + param + " d param")


def draw_PP(img_params, PP, y):

    for (derPp___) in PP.derPp____:
        for derPp__ in derPp___:
            for (derPp_) in derPp__:
                for derPp in derPp_:

                    Pp = derPp.Pp
                    _Pp = derPp._Pp
                    _name, name = PP.param_name_.pop(0)
                    # values draw
                    img_params[_name + name][y, _Pp.ix0:_Pp.ix0 + _Pp.iL] += 32
                    img_params[_name + name][y, Pp.ix0:Pp.ix0 + Pp.iL] += 32