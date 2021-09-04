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
    negM = int  # in mdert only
    negL = int  # in mdert only
    negiL = int
    Pp = object  # Pp that pdert is in, for merging in form_Pp_rng, temporary?

class CPp(CP):
    pdert_ = list
    P_ = list  # zip with pdert_
    Rdn = int  # cross-param rdn accumulated from pderts
    rval = int  # Pp value (M | abs D) adjusted for cross-param Rdn
    iL = int  # length of Pp in pixels
    fPd = bool  # P is Pd if true, else Pm; also defined per layer
    negM = int  # in mdert only
    negL = int  # in mdert only
    negiL = int
    sublayers = list

class CderPp(ClusterStructure):  # for line_PPPs only, if PPP comb x Pps?
    mPp = int
    dPp = int
    rrdn = int
    negM = int
    negL = int
    adj_mP = int  # not needed?
    _Pp = object
    Pp = object
    layer1 = dict  # dert per compared param
    der_sub_H = list  # sub hierarchy of derivatives, from comp sublayers

class CPP(CPp, CderPp):
    layer1 = dict

ave = 1  # ave dI -> mI, * coef / var type
# no ave_mP: deviation computed via rM  # ave_mP = ave* n_comp_params: comp cost, or n vars per P: rep cost?
ave_div = 50
ave_rM = .5  # average relative match per input magnitude, at rl=1 or .5?
ave_negM = 10  # or rM?
ave_M = 10  # search stop
ave_D = 10  # search stop
ave_sub_M = 500  # sub_H comp filter
ave_Ls = 3
ave_PPM = 200
ave_splice = 50  # merge Ps within core-param Pp
ave_inv = 20  # ave inverse m, change to Ave from the root intra_blob?
ave_min = 5  # ave direct m, change to Ave_min from the root intra_blob?
ave_rolp = .5  # ave overlap ratio for comp_Pp
ave_mL = 2  # needs to be tuned
ave_mI = 5  # needs to be tuned
ave_mD = 5  # needs to be tuned
ave_mM = 5  # needs to be tuned


def search(P_, fPd):  # cross-compare patterns within horizontal line

    sub_search_recursive(P_, fPd)  # search with incremental distance: first inside sublayers?
    layer0 = {'L_': [], 'I_': [], 'D_': [], 'M_': []}  # param_name: [param values]

    Ldert_ = []
    for P in P_:  # unpack P params
        L = P.L
        if "_P" in locals():  # not the 1st P
            _L = _P.L
            rL = L / _L  # div_comp L: higher-scale, not accumulated: no search
            # add ave_Ls to L param match computation?
            mL = int(max( rL, 1 / rL)) * min(L, _L) - ave_mL  # div_comp match is additive compression, not directional
            Ldert_.append( Cdert( i=L, p=L + _L, d=rL, m=mL))
        _P = P
        layer0['I_'].append(P.I / L)  # mean values for comp_param
        layer0['D_'].append(P.D / L)
        layer0['M_'].append(P.M / L)
    Pdert__ = [Ldert_]  # no search for L, step=1 only

    if fPd:  # comp | search per param type, separate dert1_ and dert2_ (step=1 and step=2 comp) for P splice only
        # I
        dert1_I = [ comp_param(_par, par, "I_", ave_mI) for _par, par in zip(layer0["I_"][:-1], layer0["I_"][1:]) ]
        # D: Pd-defining param
        dert1_D = [ comp_param(_par, par, "D_", ave_mD) for _par, par in zip(layer0["D_"][:-1], layer0["D_"][1:]) ]
        dert2_D = [ comp_param(_par, par, "D_", ave_mD) for _par, par in zip(layer0["D_"][:-2], layer0["D_"][2:]) ]
        # M
        dert1_M = [ comp_param(_par, par, "M_", ave_mM) for _par, par in zip(layer0["M_"][:-1], layer0["M_"][1:]) ]
        # generic
        Pdert__ += [dert1_I, dert1_D, dert1_M]
        dert1_ = dert1_D  # for Pd splicing
        dert2_ = dert2_D
    else:
        # I: Pm-defining param
        pdert_I = search_param_(layer0["I_"], layer0["D_"], P_, ave_mI, rave=1)  # forms variable-negL pderts
        dert1_I = [comp_param(__par, par, "I_", ave_mI) for __par, par in zip(layer0["I_"][:-1], layer0["I_"][1:])]
        dert2_I = [comp_param(__par, par, "I_", ave_mI) for __par, par in zip(layer0["I_"][:-2], layer0["I_"][2:])]
        # D
        dert1_D = [comp_param(_par, par, "D_", ave_mD) for _par, par in zip(layer0["D_"][:-1], layer0["D_"][1:]) ]
        # M
        dert2_M = [comp_param(_par, par, "M_", ave_mM) for _par, par in zip(layer0["M_"][:-2], layer0["M_"][2:]) ]
        # generic
        Pdert__ += [pdert_I, dert1_D, dert2_M]
        dert1_ = dert1_I  # for Pm splicing
        dert2_ = dert2_I
    '''
    old:
    for param_name in ["I_", "D_", "M_"]:
        param_ = layer0[param_name]  # param values
        # if dert-level P-defining param:
        if ((param_name == "I_") and not fPd) or ((param_name == "D_") and fPd):
            if not fPd:
                Pdert__ += [search_param_(param_, layer0["D_"], P_, ave_mI, rave=1)]  # pdert_ if "I_"
            # step=2 comp for P splice only:
            dert2_ = [comp_param(__par, par, param_name[0], ave) for __par, par in zip( param_[:-2], param_[2:])]
        # else step=1 per param only:
        dert1_ = [comp_param(_par, par, param_name[0], ave) for _par, par in zip( param_[:-1], param_[1:])]
        dert1__ += [dert1_]
        if not param_name == "I_":
            Pdert__ += [dert1_]  # clustered into Pps in form_Pp_
    '''

    rdn__ = sum_rdn_(layer0, Pdert__, fPd=1)  # assign redundancy to lesser-magnitude m|d in param pair for same-_P Pderts
    rval_Pp__ = []
    Ppm__ = []  # for visualization

    for param_name, Pdert_, rdn_ in zip(layer0, Pdert__, rdn__):  # segment Pdert__ into Pps
        if param_name == "I_" and not fPd:
            Ppm_ = form_Pp_rng(None, Pdert_, rdn_, P_)
        else:
            Ppm_ = form_Pp_(None, Pdert_, param_name, rdn_, P_, fPd=0)  # Ppd_ is formed in -Ppms only, in intra_Ppm_
        # rval_Pp_s per param:
        rval_Pp__ += [ form_rval_Pp_(Ppm_, param_name, dert1_, dert2_, fPd=0)]  # evaluates for compact()
        Ppm__ += [Ppm_]

    return rval_Pp__, Ppm__


def search_param_(I_, D_, P_, ave, rave):  # variable-range search in mdert_, only if param is core param?

    # higher local ave for extended rng: -> lower m and term by match, and higher proj_M?
    mdert_ = []  # line-wide (i, p, d, m, negL, negM, negiL)

    for i, (_I, _D, _P) in enumerate( zip(I_[:-1], D_[:-1], P_[:-1])):
        proj_M = 1
        negiL = negL = negM = 0
        _pI = _I - (_D / 2)  # forward project by _D
        j = i + 1

        while proj_M > 0 and j < len(I_):
            I = I_[j]; D = D_[j]; P = P_[j]
            pI = I - (D / 2)  # backward project by D
            dert = comp_param(_pI, pI, "I_", ave)  # param is compared to prior-P _param
            if dert.m > 0:
                if dert.m + _P.M > 0:
                    comp_sublayers(P_[i], P_[j], dert.m, dert.d)
                break  # 1st matching param takes over connectivity search from _param, in the next loop
            else:
                proj_M = dert.m * rave + negM - ave_M  # lower ave_M instead of projection?
                negM += dert.m * rave  # or abs m only?
                negiL += P.L
                negL += 1
                j += 1

        # after extended search, if any:
        mdert_.append( Cpdert(i=dert.i, p=dert.p, d=dert.d, m=dert.m, negiL=negiL, negL=negL, negM=negM))

    return mdert_


def sum_rdn_(layer0, Pdert__, fPd):
    '''
    access same-index pderts of all P params, assign redundancy to lesser-magnitude m|d in param pair.
    if other-param same-P_-index pdert is missing, rdn doesn't change.
    '''
    if fPd: alt = 'M'
    else:   alt = 'D'
    name_pairs = (('I', 'L'), ('I', 'D'), ('I', 'M'), ('L', alt), ('D', 'M'))  # pairs of params redundant to each other
    pderts_Rdn = [[], [], [], []]  # L_, I_, D_, M_' Rdns, as in pdert__

    for Ldert, Idert, Ddert, Mdert in zip_longest(Pdert__[0], Pdert__[1], Pdert__[2], Pdert__[3], fillvalue=Cdert()):
        # pdert per _P in P_, 0: Ldert_, 1: Idert_, 2: Ddert_, 3: Mdert_
        rdn_pairs = [[fPd, 0], [fPd, 1-fPd], [fPd, fPd], [0, 1], [1-fPd, fPd]]  # rdn in olp Pms, Pds: if fPd: I, M rdn+=1, else: D rdn+=1
        # names:    ('I','L'), ('I','D'),    ('I','M'),  ('L',alt), ('D','M'))  # I.m + P.M: value is combined across P levels?

        for rdn_pair, name_pair in zip(rdn_pairs, name_pairs):
            # assign rdn in each rdn_pair using partial name substitution: https://www.w3schools.com/python/ref_func_eval.asp
            if fPd:
                if eval("abs(" + name_pair[0] + "dert.d) > abs(" + name_pair[1] + "dert.d)"):  # (param_name)dert.d|m
                    rdn_pair[1] += 1
                else: rdn_pair[0] += 1  # weaker pair rdn+1
            else:
                if eval(name_pair[0] + "dert.m > " + name_pair[1] + "dert.m"):
                    rdn_pair[1] += 1
                else: rdn_pair[0] += 1  # weaker pair rdn+1

        for i, param_name in enumerate(layer0):  # sum param rdn from all pairs it is in, flatten pair_names, pair_rdns?
            Rdn = 0
            for name_in_pair, rdn in zip(name_pairs, rdn_pairs):
                if param_name[0] == name_in_pair[0]:  # param_name = "L_", param_name[0] = "L"
                    Rdn += rdn[0]
                elif param_name[0] == name_in_pair[1]:
                    Rdn += rdn[1]

            pderts_Rdn[i].append(Rdn)  # same length as Pdert_

    return pderts_Rdn  # rdn__


def form_Pp_(rootPp, dert_, param_name, rdn_, P_, fPd):
    # initialization:
    Pp_ = []
    x = 0
    _sign = None  # to initialize 1st P, (None != True) and (None != False) are both True

    for dert, rdn, P in zip(dert_, rdn_, P_):  # segment by sign
        if fPd: sign = dert.d > 0
        else:   sign = dert.m > 0
        # adjust by ave projected at distance=negL and contrast=negM, if significant:
        # m + ddist_ave = ave - ave * (ave_rM * (1 + negL / ((param.L + _param.L) / 2))) / (1 + negM / ave_negM)?
        if sign != _sign:
            # sign change, initialize Pp and append it to Pp_
            Pp = CPp( L=1, iL=P_[x].L, I=dert.p, D=dert.d, M=dert.m, Rdn=rdn, x0=x, ix0=P_[x].x0, pdert_=[dert], P_=[P_[x]], sublayers=[], fPd=fPd)
            Pp_.append(Pp)  # updated by accumulation below
        else:
            # accumulate params:
            Pp.L += 1; Pp.iL += P_[x].L; Pp.I += dert.p; Pp.D += dert.d; Pp.M += dert.m; Pp.Rdn += rdn; Pp.pdert_ += [dert]; Pp.P_ += [P]
        x += 1
        _sign = sign

    if param_name == "M_" and not fPd:
        Pp.P_ += P_[-2:]  # last 2 Ps for M param (step 2)
    else:
        Pp.P_ += [P_[-1]]  # last P

    if rootPp:
        # call from intra_Pp_; sublayers brackets: 1st: param set, 2nd: sublayer concatenated from n root_Ps, 3rd: layers depth hierarchy
        rootPp.sublayers = [[( fPd, Pp_ )]]  # 1st sublayer is one element, sub_Ppm__=[], + Dert=[]
        if len(P_) > 4:  # 2 * (rng+1) = 2*2 =4
            rootPp.sublayers += intra_Pp_(Pp_, param_name, rdn_, fPd)  # deeper comb_layers feedback, sum params for comp_sublayers?
    else:
        # call from search
        intra_Pp_(Pp_, param_name, rdn_, fPd)   # evaluates for sub-recursion and forming Ppd_ per Pm
        return Pp_  # else packed in P.sublayers


def form_Pp_rng(rootPp, dert_, rdn_, P_):  # cluster Pps by cross-param redundant value sign, eval for cross-level rdn
    # multiple Pps may overlap within _dert.negL
    Pp_ = []

    for i, (_dert, _P, _rdn) in enumerate(zip(dert_, P_, rdn_)):
        if _dert.m + _P.M > ave*_rdn:  # positive Pps only, else too much overlap? + _P.M: value is combined across P levels?
            # initialize Pp:
            if not isinstance(_dert.Pp, CPp):  # _dert is not in any Pp
                Pp = CPp(L=1, iL=_P.L, I=_dert.p, D=_dert.d, M=_dert.m, Rdn=_rdn, negiL=_dert.negiL, negL=_dert.negL, negM=_dert.negM,
                         x0=i, ix0=_P.x0, pdert_=[_dert], P_=[_P], sublayers=[], fPd=0)
                _dert.Pp = Pp
                Pp_.append(Pp)  # params will be accumulated
            else:
                break  # _dert already searched forward
            j = i + _dert.negL + 1

            while (j <= len(dert_)-1):
                dert = dert_[j]; P = P_[j]; rdn = rdn_[j]  # no pop: maybe used by other _derts
                if dert.m + P.M > ave*rdn:
                    if isinstance(dert.Pp, CPp):  # unique Pp per dert in row Pdert_
                        # merge Pp with dert.Pp, if any:
                        Pp.accum_from(dert.Pp,excluded=['x0'])
                        Pp.P_ += dert.Pp.P_
                        Pp.pdert_ += dert.Pp.pdert_
                        Pp.sublayers += dert.Pp.sublayers
                        Pp_.remove(dert.Pp)
                        dert.Pp = Pp
                        break  # dert already searched forward
                    else:  # accumulate params:
                        Pp.L += 1; Pp.iL += P.L; Pp.I += dert.p; Pp.D += dert.d; Pp.M += dert.m; Pp.Rdn += rdn; Pp.negiL += dert.negiL
                        Pp.negL += dert.negL; Pp.negM += dert.negM; Pp.pdert_ += [dert]; Pp.P_ += [P]
                        # no dert.Pp, only 1st dert in pdert_ is checked; Pp derts already searched dert_, they won't be new _derts
                        j += dert.negL+1
                else:
                    break  # Pp is terminated

    if rootPp:
        # call from intra_Pp_; sublayers brackets: 1st: param set, 2nd: sublayer concatenated from n root_Ps, 3rd: layers depth hierarchy
        rootPp.sublayers = [[( 0, Pp_ )]]  # 1st sublayer is one element, sub_Ppm__=[], + Dert=[]
        if len(P_) > 4:  # 2 * (rng+1) = 2*2 =4
            rootPp.sublayers += intra_Pp_(Pp_, "I_", rdn_, fPd=0)  # deeper comb_layers feedback, sum params for comp_sublayers?
    else:
        # call from search
        intra_Pp_(Pp_, "I_", rdn_, fPd=0)   # evaluates for sub-recursion and forming Ppd_ per Pm
        return Pp_  # else packed in P.sublayers


def form_rval_Pp_(iPp_, param_name, pdert1_, pdert2_, fPd):
    # cluster Pps by cross-param redundant value sign, re-evaluate them for cross-level rdn
    rval_Pp_ = []
    _sign = None  # to initialize 1st rdn Pp, (None != True) and (None != False) are both True

    for iPp in iPp_:
        if fPd: rval = abs(iPp.D) - iPp.Rdn * ave_D * iPp.L
        else:   rval = iPp.M - iPp.Rdn * ave_M * iPp.L
        sign = rval>0

        if sign != _sign:  # sign change, initialize rPp and append it to rPp_
            if _sign:  # -rPps are not processed?
                compact(rval_Pp_, pdert1_, pdert2_, param_name, fPd)  # re-eval Pps, Pp.pdert_s for redundancy, eval splice Ps
            Rval=rval; Pp_=[iPp]  # updated by accumulation below
        else:
            # accumulate params:
            Rval += rval; Pp_.append(iPp)
        _sign = sign

    return (Rval, Pp_)  # rval_Pp_
'''
    rPp_ = []
    x = 0
    _sign = None  # to initialize 1st rdn Pp, (None != True) and (None != False) are both True
    for Pp in Pp_:
        if fPd: Pp.rval = abs(Pp.D) - Pp.Rdn * ave_D * Pp.L
        else:   Pp.rval = Pp.M - Pp.Rdn * ave_M * Pp.L
        sign = Pp.rval > 0
        if sign != _sign:  # sign change, initialize rPp and append it to rPp_
            rPp = CPp(L=1, iL=Pp.iL, I=Pp.I, D=Pp.D, M=Pp.M, Rdn=Pp.Rdn, rval=Pp.rval, negiL=Pp.negiL, negL=Pp.negL, negM=Pp.negM,
                      x0=x, ix0=Pp.x0, pdert_=[Pp], sublayers=[], fPd=fPd)
            # or rPp is sign, Pp_?
            if _sign:  # -rPps are not processed?
                compact(rPp, pdert1__, pdert2__, param_name, fPd)  # re-eval Pps, Pp.pdert_s for redundancy, eval splice Ps
            rPp_.append(rPp)  # updated by accumulation below
        else:
            # accumulate params:
            rPp.L += 1; rPp.iL += Pp.iL; rPp.I += Pp.I; rPp.D += Pp.D; rPp.M += Pp.M; rPp.Rdn += Pp.Rdn; rPp.rval += Pp.rval
            rPp.negiL += Pp.negiL; rPp.negL += Pp.negL; rPp.negM += Pp.negM
            rPp.pdert_ += [Pp]
        x += 1
        _sign = sign
'''
def compact(rval_Pp_, pdert1_, pdert2_, param_name, fPd):  # re-eval Pps, Pp.pdert_s for redundancy, eval splice Ps

    for i, (rval, Pp) in enumerate(rval_Pp_):
        # assign cross-level rdn (Pp vs. pdert_), re-evaluate Pp and pdert_:
        Pp_val = rval / Pp.L - ave  # / Pp.L: resolution reduction, but lower rdn:
        pdert_val = rval - ave * Pp.L  # * Pp.L: ave cost * number of representations

        if Pp_val > pdert_val: pdert_val -= ave * Pp.Rdn
        else:                  Pp_val -= ave * Pp.Rdn  # ave scaled by rdn
        if Pp_val <= 0:
            rval_Pp_[i] = (rval, CPp(pdert_=Pp.pdert_))
            # Pp remove: reset Pp vars to 0, do we need reset their x0 and L too? or we should keep the position information?
            # do you mean P remove? no, Ls are already merged, and initial x0 is fine

        elif ((param_name == "I_") and not fPd) or ((param_name == "D_") and fPd):  # P-defining params, else no separation
            M2 = M1 = 0
            # param match over step=2 and step=1:
            for pdert2 in pdert2_: M2 += pdert2.m  # match(I, __I or D, __D)
            for pdert1 in pdert1_: M1 += pdert1.m  # match(I, _I or D, _D)

            if M2 / abs(M1) > -ave_splice:  # similarity / separation: splice Ps in Pp, also implies weak Pp.pdert_?
                _P = CP()
                for P in Pp.P_:
                    _P.accum_from(P, excluded=["x0"])  # different from Pp params
                    _P.dert_ += [P.dert_]  # splice dert_s within Pp
                # rerun form_P_(P.dert_)?
                rval_Pp_[i] = (rval, _P)  # replace Pp with spliced P

        if pdert_val <= 0:
            Pp.pdert_ = []  # remove pdert_
    '''
    for i, Pp in enumerate(rPp.pdert_):
        # assign cross-level rdn (Pp vs. pdert_), re-evaluate Pp and pdert_:
        Pp_val = Pp.rval / Pp.L - ave  # / Pp.L: resolution reduction, but lower rdn:
        pdert_val = Pp.rval - ave * Pp.L  # * Pp.L: ave cost * number of representations
        if Pp_val > pdert_val: pdert_val -= ave * Pp.Rdn
        else:                  Pp_val -= ave * Pp.Rdn  # ave scaled by rdn
        if Pp_val <= 0:
            rPp.pdert_[i] = CPp(pdert_=Pp.pdert_)  # Pp remove: reset Pp vars to 0
        elif ((param_name == "I_") and not fPd) or ((param_name == "D_") and fPd):  # P-defining params, else no separation
            M2 = M1 = 0
            # param match over step=2 and step=1:
            for pdert2 in pdert2_: M2 += pdert2.m  # match(I, __I or D, __D)
            for pdert1 in pdert1_: M1 += pdert1.m  # match(I, _I or D, _D)
            if M2 / abs(M1) > -ave_splice:  # similarity / separation: splice Ps in Pp, also implies weak Pp.pdert_?
                _P = CP()
                for P in Pp.P_:
                    _P.accum_from(P, excluded=["x0"])  # different from Pp params
                    _P.dert_ += [P.dert_]  # splice dert_s within Pp
                # rerun form_P_(P.dert_)?
                rPp.pdert_[i] = _P  # replace Pp with spliced P
        if pdert_val <= 0:
            Pp.pdert_ = []  # remove pdert_
    '''

def intra_Pp_(Pp_, param_name, rdn_, fPd):  # evaluate for sub-recursion in line Pm_, pack results into sub_Pm_

    comb_layers = []  # combine into root P sublayers[1:]
    # each Pp is evaluated for incremental range and derivation xcomp, as in line_patterns but with local aves

    for Pp, rdn in zip( Pp_, rdn_):  # each sub_layer is nested to depth = sublayers[n]
        if Pp.L > 2:  # no rel_adj_M = adj_M / -P.M: discontinuous search?
            mean_M = Pp.M / Pp.L  # for internal Pd eval, +opposite-side mean_M?

            if fPd:  # Pp is Ppd
                if abs(Pp.D) * mean_M > ave_D * rdn and Pp.L > 3:  # mean_M from adjacent +ve Ppms
                    rdn_ = [rdn + 1 for rdn in rdn_]
                    ddert_ = []
                    for _pdert, pdert in zip( Pp.pdert_[:-1], Pp.pdert_[1:]):  # Pd.pdert_ is dert1_
                        _param = _pdert.d; param = pdert.d
                        dert = comp_param(_param, param, param_name[0], ave)  # cross-comp of ds in dert1_, !search, local aves?
                        ddert_ += [ Cdert( i=dert.i, p=dert.p, d=dert.d, m=dert.m)]
                    # cluster Pd derts by md sign:
                    form_Pp_(Pp, ddert_, param_name, rdn_, Pp.P_, fPd=True)
            else:  # Pp is Ppm
                # +Ppm -> sub_Ppm_: low-variation span, eval rng_comp:
                if Pp.M > 0 and Pp.M > ave_M * rdn and param_name=="I_":  # and if variable cost: Pp.M / Pp.L? -lend to contrast?
                    rdn_ = [rdn+1 for rdn in rdn_]
                    I_ = [pdert.i for pdert in (Pp.pdert_[:-1])]
                    D_ = [pdert.d for pdert in (Pp.pdert_[:-1])]
                    # search range is extended by higher ave: less replacing by match,
                    # and by higher proj_P = dert.m * rave ((Pp.M / Pp.L) / ave): less term by miss:
                    P_ave = Pp.M / Pp.L
                    rpdert_ = search_param_(I_, D_, Pp.P_, (ave + P_ave) / 2, rave = P_ave / ave )
                    form_Pp_(Pp, rpdert_, param_name, rdn_, Pp.P_, fPd=False)  # cluster by m sign, eval intra_Pm_
                # -Ppm -> sub_Ppd:
                elif -Pp.M > ave_D * rdn:  # high-variation span, -M is contrast borrowed from adjacent +Ppms: or abs D: likely sign match span?
                    rdn_ = [rdn+1 for rdn in rdn_]
                    form_Pp_(Pp, Pp.pdert_, param_name, rdn_, Pp.P_, fPd=True)  # cluster by d sign: partial d match, eval intra_Pm_(Pdm_)

            if Pp.sublayers:  # splice sublayers from all sub_Pp calls in Pp:
                comb_layers = [comb_layer + sublayer for comb_layer, sublayer in
                               zip_longest(comb_layers, Pp.sublayers, fillvalue=[])
                               ]
    return comb_layers


def sub_search_recursive(P_, fPd):  # search in top sublayer per P / sub_P

    for P in P_:
        if P.sublayers:
            sublayer = P.sublayers[0][0]  # top sublayer has one array
            sub_P_ = sublayer[3]
            if len(sub_P_) > 2:
                if fPd:
                    if abs(P.D) > ave_D:  # better use sublayers.D|M, but we don't have it yet
                        sub_rdn_Pp__ = search(sub_P_, fPd)
                        sublayer[4].append(sub_rdn_Pp__)
                        sub_search_recursive(sub_P_, fPd)  # deeper sublayers search is selective per sub_P
                elif P.M > ave_M:
                    sub_rdn_Pp__ = search(sub_P_, fPd)
                    sublayer[4].append(sub_rdn_Pp__)
                    sub_search_recursive(sub_P_, fPd)  # deeper sublayers search is selective per sub_P


def comp_sublayers(_P, P, mP, dP):  # not revised; also add dP?

    if P.sublayers and _P.sublayers:  # not empty sub layers
        for _sub_layer, sub_layer in zip(_P.sublayers[0], P.sublayers[0]):

            if _sub_layer and sub_layer:
                _fPd, _rdn, _rng, _sub_P_, _sub_Pp__, = _sub_layer
                fPd, rdn, rng, sub_P_, sub_Pp__ = sub_layer
                # fork comparison:
                if fPd == _fPd and rng == _rng and min(_P.L, P.L) > ave_Ls:
                    sub_mP = sub_dP = 0
                    # compare all sub_Ps to each _sub_P:
                    for _sub_P in _sub_P_:
                        for sub_P in sub_P_:
                            sub_dert = comp_param(_sub_P.I, sub_P.I, "I_", ave)
                    sub_mP += sub_dert.m  # of compared H, no specific mP?
                    sub_dP += sub_dert.d
                    if sub_mP + mP < ave_sub_M:
                        # potentially mH: trans-layer induction: if mP + sub_mP < ave_sub_M: both local and global values of mP.
                        break  # low vertical induction, deeper sublayers are not compared
                else:
                    break  # deeper P and _P sublayers are from different intra_comp forks, not comparable?


def draw_PP_(image, frame_Pp__):

    from matplotlib import pyplot as plt
    import numpy as np

    # initialization
    img_rval_Pp_ = [np.zeros_like(image) for _ in range(4)]

    img_Pp_ = [np.zeros_like(image) for _ in range(4)]
    img_Pp_pdert_ = [np.zeros_like(image) for _ in range(4)]

    param_names = ['L', 'I', 'D', 'M']

    for y, (rval_Pp__, Pp__) in enumerate(frame_Pp__):  # loop each line
        for i, (rval_Pp_, Pp_) in enumerate(zip(rval_Pp__, Pp__)): # loop each rdn_Pp or Pp
            # rval_Pp
            for j, rval_Pps in enumerate(rval_Pp_):
                for k, (rval, Pp) in enumerate(rval_Pps):
                    for m, P in enumerate(Pp.P_):

                        if rval>0:
                            img_rval_Pp_[i][y,P.x0:P.x0+P.L] = 255 # + sign
                        else:
                            img_rval_Pp_[i][y,P.x0:P.x0+P.L] = 128 # - sign
            # Pp
            for j, Pp in enumerate(Pp_): # each Pp
                for k, P in enumerate(Pp.P_): # each P or pdert

                    if Pp.M>0:
                        img_Pp_[i][y,P.x0:P.x0+P.L] = 255 # + sign
                    else:
                        img_Pp_[i][y,P.x0:P.x0+P.L] = 128 # - sign

                    if P.M>0:
                        img_Pp_pdert_[i][y,P.x0:P.x0+P.L] = 255 # + sign
                    else:
                        img_Pp_pdert_[i][y,P.x0:P.x0+P.L] = 128 # - sign


    # plot diagram of params
    plt.figure()
    for i, param in enumerate(param_names):
        plt.subplot(2, 2, i + 1)
        plt.imshow(img_rval_Pp_[i], vmin=0, vmax=255)
        plt.title("Rval Pps, Param = " + param)

    plt.figure()
    for i, param in enumerate(param_names):
        plt.subplot(2, 2, i + 1)
        plt.imshow(img_Pp_[i], vmin=0, vmax=255)
        plt.title("Pps, Param = " + param)

    plt.figure()
    for i, param in enumerate(param_names):
        plt.subplot(2, 2, i + 1)
        plt.imshow(img_Pp_pdert_[i], vmin=0, vmax=255)
        plt.title("Pderts, Param = " + param)