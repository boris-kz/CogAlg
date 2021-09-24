'''
line_PPs is a 2nd-level 1D algorithm, its input is P_ formed by the 1st-level line_patterns.
It cross-compares P params (initially L, I, D, M) and forms param_Ps: Pp_ for each param type per image row.
-
Subsequent cross-comp between Pps of different params is exclusive of x overlap, where the relationship is already known.
Thus it should be on 3rd level: no Pp overlap means comp between Pps: higher composition, same-type ) cross-type?
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
    sub_M = int  # match from comp_sublayers, if any
    sub_D = int  # diff from comp_sublayers, if any
    Pp = object  # Pp that pdert is in, for merging in form_Pp_rng, temporary?

class CPp(CP):
    pdert_ = list  # components, "p" for param
    P_ = list  # zip with pdert_
    Rdn = int  # cross-param rdn accumulated from pderts
    rdn_ = list # for sub's workflow
    rval = int  # Pp value (M | abs D) adjusted for cross-param Rdn
    iL = int  # length of Pp in pixels
    fPd = bool  # P is Pd if true, else Pm; also defined per layer
    negM = int  # in mdert only
    negL = int  # in mdert only
    negiL = int
    sublayers = list
    subDerts = list

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

class CPP(CPp, CderPp):  # obsolete, no Pp grouping in PPs?
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
ave_mL = 5  # needs to be tuned
ave_mI = 5  # needs to be tuned
ave_mD = 1  # needs to be tuned
ave_mM = 5  # needs to be tuned
ave_sub = 20  # for comp_sub_layers

ave_Dave = 100  # summed feedback filter
ave_dave = 20   # mean feedback filter

def norm_feedback(P_, fPd):

    for P in P_:
        fbM = fbL = 0
        fbM += P.M; fbL += P.L
        if fbM > ave_Dave:
            if fbM / fbL > ave_dave:
                pass  # eventually feedback: line_patterns' cross_comp(frame_of_pixels_, ave + fbM / fbL)
                # also terminate Fspan: same-filter frame_ with summed params
        P.I /= P.L; P.D /= P.L; P.M /= P.L  # immediate normalization to a mean
    search(P_, fPd)


def search(P_, fPd):  # cross-compare patterns within horizontal line

    param_names = ["L_", "I_", "D_", "M_"]
    Ldert_, Idert_, Ddert_, Mdert_, dert1_, dert2_ = [], [], [], [], [], []

    for _P, P, P2 in zip(P_, P_[1:], P_[2:] + [CP()]):
        _L, _I, _D, _M, *_ = _P.unpack()  # *_: skip remaining params
        L, I, D, M, *_ = P.unpack()
        D2, M2 = P2.D, P2.M
        # div_comp for L:
        rL = L / _L  # higher order of scale, not accumulated: no search, rL is directional
        int_rL = int( max(rL, 1/rL))
        frac_rL = max(rL, 1/rL) - int_rL
        mL = int_rL * min(L, _L) - (int_rL*frac_rL) / 2 - ave_mL  # div_comp match is additive compression: +=min, not directional
        Ldert_.append(Cdert( i=L, p=L + _L, d=rL, m=mL))
        # summed params comp:
        if fPd:
            Idert_ += [comp_param(_I, I, "I_", ave_mI)]
            Ddert = comp_param(_D, D2, "D_", ave_mD)  # step=2 for same-D-sign comp
            Ddert_ += [Ddert]
            dert2_ += [Ddert.copy()]
            dert1_ += [comp_param(_D, D, "D_", ave_mD)]  # to splice Pds
            Mdert_ += [comp_param(_M, M, "M_", ave_mM)]
        else:
            Ddert_ += [comp_param(_D, D, "D_", ave_mD)]
            Mdert = comp_param(_M, M2, "M_", ave_mM)  # step=2 for same-M-sign comp
            Mdert_ += [Mdert]
            dert2_ += [Mdert.copy()]
            dert1_ += [comp_param(_M, M, "M_", ave_mM)]  # to splice Pms
        _L, _I, _D, _M = L, I, D, M

    LP_ = P_[:-1]
    dert2_ = dert2_[:-1]  # due to zip_longest in line 98
    if not fPd:
        Idert_, IP_ = search_param_(P_, ave_mI, rave=1)  # comp x variable range, depending on M of Is
        Mdert_ = Mdert_[:-1]  # due to zip_longest in search_param
        DP_, MP_ = P_[:-1], P_[:-2]
    else:
        IP_, DP_, MP_ = P_[:-1], P_[:-2], P_[:-1]

    Pdert__ = [(Ldert_, LP_), (Idert_, IP_), (Ddert_, DP_), (Mdert_, MP_)]

    rdn__ = sum_rdn_(param_names, Pdert__, fPd=1)  # assign redundancy to lesser-magnitude m|d in param pair for same-_P Pderts
    rval_Pp__ = []
    Ppm__ = []  # for visualization

    for param_name, (Pdert_, P_), rdn_ in zip(param_names, Pdert__, rdn__):  # segment Pdert__ into Pps
        if param_name == "I_" and not fPd:
            Ppm_ = form_Pp_rng(None, Pdert_, rdn_, P_)
        else:
            Ppm_ = form_Pp_(None, Pdert_, param_name, rdn_, P_, fPd=0)  # Ppd_ is formed in -Ppms only, in intra_Ppm_
        # redundant-value sign-spans of Pps per param:
        rval_Pp__ += [ form_rval_Pp_(Ppm_, param_name, dert1_, dert2_, fPd=0)]  # evaluates for compact()
        Ppm__ += [Ppm_]

    return rval_Pp__, Ppm__


def search_param_(P_, ave, rave):  # variable-range search in mdert_, only if param is core param?

    # higher local ave for extended rng: -> lower m and term by match, and higher proj_M?
    Idert_, _P_ = [], []  # line-wide (i, p, d, m, negL, negM, negiL)

    for i, _P in enumerate( P_[:-1]):
        negM = 0
        _pI = _P.I - (_P.D / 2)  # forward project by _D
        j = i + 1  # init with positive-M Is only: internal match projects xP I match:

        while _P.M + negM > ave_M and j < len(P_):  # starts with positive _P.Ms, but continues over negM if > ave_M
            P = P_[j]
            pI = P.I + (P.D / 2)  # backward project by D
            dert = comp_param(_pI, pI, "I_", ave)  # param is compared to prior-P _param
            pdert = Cpdert(i=dert.i, p=dert.p, d=dert.d, m=dert.m)  # convert Cdert to Cpdert
            curr_M = pdert.m * rave + (_P.M + P.M) / 2  # P.M is bidirectional

            if curr_M > ave_sub:  # comp all sub_P_ params, for core I only?
                comp_sublayers_draft(P_[i], P_[j], pdert)  # should set dert.sub_M: between sublayers
            if curr_M + pdert.sub_M > ave_M:  # or > ave_cM? else: pdert.sub_M is negative
                break  # 1st match takes over connectivity search in the next loop
            else:
                pdert.negM += curr_M - ave_M  # known to be negative, accum per dert
                pdert.negiL += P.L
                pdert.negL += 1
                negM = pdert.negM
                j += 1

        if "pdert" in locals():  # after extended search, if any:
            Idert_.append(pdert)
            _P_.append(_P)
            del pdert # prevent reuse of same pdert in multiple loops

    return Idert_, _P_


def sum_rdn_(param_name_, Pdert__, fPd):
    '''
    access same-index pderts of all P params, assign redundancy to lesser-magnitude m|d in param pair.
    if other-param same-P_-index pdert is missing, rdn doesn't change.
    '''
    if fPd: alt = 'M'
    else:   alt = 'D'
    name_pairs = (('I', 'L'), ('I', 'D'), ('I', 'M'), ('L', alt), ('D', 'M'))  # pairs of params redundant to each other
    pderts_Rdn = [[], [], [], []]  # L_, I_, D_, M_' Rdns, as in pdert__

    for Ldert, Idert, Ddert, Mdert in zip_longest(Pdert__[0][0], Pdert__[1][0], Pdert__[2][0], Pdert__[3][0], fillvalue=Cdert()):
        # pdert per _P in P_, 0: Ldert_, 1: Idert_, 2: Ddert_, 3: Mdert_
        rdn_pairs = [[fPd, 0], [fPd, 1-fPd], [fPd, fPd], [0, 1], [1-fPd, fPd]]  # rdn in olp Ps: if fPd: I, M rdn+=1, else: D rdn+=1
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

        for i, param_name in enumerate(param_name_):  # sum param rdn from all pairs it is in, flatten pair_names, pair_rdns?
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
            Pp = CPp( L=1, iL=P_[x].L, I=dert.p, D=dert.d, M=dert.m, Rdn=rdn, rdn_ = [rdn], x0=x, ix0=P_[x].x0, pdert_=[dert], P_=[P_[x]], sublayers=[], fPd=fPd)
            Pp_.append(Pp)  # updated by accumulation below
        else:
            # accumulate params:
            Pp.L += 1; Pp.iL += P_[x].L; Pp.I += dert.p; Pp.D += dert.d; Pp.M += dert.m; Pp.Rdn += rdn; Pp.rdn_ += [rdn]; Pp.pdert_ += [dert]; Pp.P_ += [P]
        x += 1
        _sign = sign

    if rootPp:
        Dert = [0,0,0,0]  # P.L, I, D, M summed within a layer
        # call from intra_Pp_; sublayers brackets: 1st: param set, 2nd: sublayer concatenated from n root_Ps, 3rd: layers depth hierarchy
        rootPp.sublayers = [[( fPd, Pp_ )]]  # 1st sublayer is one element, sub_Ppm__=[], + Dert=[]
        rootPp.subDerts = [Dert]
        if (len(Pp_) > 4) and (rootPp.M * len(Pp_) > ave_M): # 2 * (rng+1) = 2*2 =4
            Dert[:] = [0, 0, 0, 0]  # P.L, I, D, M summed within a layer
            for Pp in Pp_:
                Dert[0] += Pp.L; Dert[1] += Pp.I; Dert[2] += Pp.D; Dert[3] += Pp.M
            comb_sublayers, comb_subDerts = intra_Pp_(Pp_, param_name, fPd)  # deeper comb_layers feedback, subDerts is selective
            rootPp.sublayers += comb_sublayers
            rootPp.subDerts += comb_subDerts
    else:
        # call from search
        intra_Pp_(Pp_, param_name, fPd)   # evaluates for sub-recursion and forming Ppd_ per Pm
        return Pp_  # else packed in P.sublayers


def form_Pp_rng(rootPp, dert_, rdn_, P_):  # cluster Pps by cross-param redundant value sign, eval for cross-level rdn
    # multiple Pps may overlap within _dert.negL
    Pp_ = []

    for i, (_dert, _P, _rdn) in enumerate(zip(dert_, P_, rdn_)):
        if _dert.m + _P.M > ave*_rdn:  # positive Pps only, else too much overlap? +_P.M: value is combined across P levels?
            # initialize Pp:
            if not isinstance(_dert.Pp, CPp):  # _dert is not in any Pp
                Pp = CPp(L=1, iL=_P.L, I=_dert.p, D=_dert.d, M=_dert.m, Rdn=_rdn, rdn_ = [_rdn], negiL=_dert.negiL, negL=_dert.negL, negM=_dert.negM,
                         x0=i, ix0=_P.x0, pdert_=[_dert], P_=[_P], sublayers=[], fPd=0)
                _dert.Pp = Pp
                Pp_.append(Pp)  # params will be accumulated
            else:
                break  # this _dert already searched forward
            j = i + _dert.negL + 1

            while (j <= len(dert_)-1):
                dert = dert_[j]; P = P_[j]; rdn = rdn_[j]  # no pop: maybe used by other _derts
                if dert.m + P.M > ave*rdn:
                    if isinstance(dert.Pp, CPp):  # unique Pp per dert in row Pdert_
                        # merge Pp with dert.Pp, if any:
                        Pp.accum_from(dert.Pp,excluded=['x0'])
                        Pp.P_ += dert.Pp.P_
                        Pp.pdert_ += dert.Pp.pdert_
                        Pp.rdn_ += dert.Pp.rdn_
                        Pp.sublayers += dert.Pp.sublayers
                        Pp_.remove(dert.Pp)
                        dert.Pp = Pp
                        break  # this dert already searched forward
                    else:  # accumulate params:
                        Pp.L += 1; Pp.iL += P.L; Pp.I += dert.p; Pp.D += dert.d; Pp.M += dert.m; Pp.Rdn += rdn; Pp.rdn_ += [rdn]; Pp.negiL += dert.negiL
                        Pp.negL += dert.negL; Pp.negM += dert.negM; Pp.pdert_ += [dert]; Pp.P_ += [P]
                        # no dert.Pp, only 1st dert in pdert_ is checked; Pp derts already searched dert_, they won't be new _derts
                        j += dert.negL+1
                else:
                    break  # Pp is terminated

    if rootPp:
        Dert = [0,0,0,0]  # P.L, I, D, M summed within a layer
        # call from intra_Pp_; sublayers brackets: 1st: param set, 2nd: sublayer concatenated from n root_Ps, 3rd: layers depth hierarchy
        rootPp.sublayers = [[( fPd, Pp_ )]]  # 1st sublayer is one element, sub_Ppm__=[], + Dert=[]
        rootPp.subDerts = [Dert]
        if (len(Pp_) > 4) and (rootPp.M * len(Pp_) > ave_M): # 2 * (rng+1) = 2*2 =4
            Dert[:] = [0, 0, 0, 0]  # P.L, I, D, M summed within a layer
            for Pp in Pp_:
                Dert[0] += Pp.L; Dert[1] += Pp.I; Dert[2] += Pp.D; Dert[3] += Pp.M
            comb_sublayers, comb_subDerts = intra_Pp_(Pp_, "I_", fPd=0)  # deeper comb_layers feedback, subDerts is selective
            rootPp.sublayers += comb_sublayers
            rootPp.subDerts += comb_subDerts
    else:
        # call from search
        intra_Pp_(Pp_, "I_", fPd=0)   # evaluates for sub-recursion and forming Ppd_ per Pm
        return Pp_  # else packed in P.sublayers


def form_rval_Pp_(iPp_, param_name, pdert1_, pdert2_, fPd):
    # cluster Pps by cross-param redundant value sign, re-evaluate them for cross-level rdn
    rval_Pp__ = []
    Rval=0
    _sign = None  # to initialize 1st rdn Pp, (None != True) and (None != False) are both True

    for Pp in iPp_:
        if fPd: rval = abs(Pp.D) - Pp.Rdn * ave_D * Pp.L
        else:   rval = Pp.M - Pp.Rdn * ave_M * Pp.L
        sign = rval>0

        if sign != _sign:  # sign change, initialize rPp and append it to rPp_
            rval_Pp_ = [(rval, Pp)]
            Rval = rval
            if _sign:  # -rPps are not processed?
                compact(rval_Pp_, pdert1_, pdert2_, param_name, fPd)  # re-eval Pps, Pp.pdert_s for redundancy, eval splice Ps
            rval_Pp__.append(( Rval, rval_Pp_))  # updated by accumulation below
        else:
            # accumulate params:
            Rval += rval
            rval_Pp_ += [(rval, Pp)]
        _sign = sign

    return rval_Pp__


def compact(rval_Pp_, pdert1_, pdert2_, param_name, fPd):  # re-eval Pps, Pp.pdert_s for redundancy, eval splice Ps

    for i, (rval, Pp) in enumerate(rval_Pp_):
        # assign cross-level rdn (Pp vs. pdert_), re-evaluate Pp and pdert_:
        Pp_val = rval / Pp.L - ave  # / Pp.L: resolution reduction, but lower rdn:
        pdert_val = rval - ave * Pp.L  # * Pp.L: ave cost * number of representations

        if Pp_val > pdert_val: pdert_val -= ave * Pp.Rdn
        else:                  Pp_val -= ave * Pp.Rdn  # ave scaled by rdn
        if Pp_val <= 0:
            rval_Pp_[i] = (rval, CPp(pdert_=Pp.pdert_))

        elif ((param_name == "I_") and not fPd) or ((param_name == "D_") and fPd):  # P-defining params, else no separation
            M2 = M1 = 0
            # param match over step=2 and step=1:
            for pdert2 in pdert2_: M2 += pdert2.m  # match(I, __I) or (D, __D)
            for pdert1 in pdert1_: M1 += pdert1.m  # match(I, _I) or (D, _D)

            if abs(M2) / abs(M1) > ave_splice:  # similarity / separation: splice Ps in Pp, also implies weak Pp.pdert_?
                _P = CP()
                for P in Pp.P_:
                    _P.accum_from(P, excluded=["x0"])  # different from Pp params
                    _P.dert_ += [P.dert_]  # splice dert_s within Pp

                form_P_(_P, _P.dert_, rdn=1, rng=1, fPd=fPd)  # rerun on spliced Ps
                rval_Pp_[i] = (rval, _P)  # replace Pp with spliced P,
                # or rerun search(spliced_P_) if len(spliced_P_) / len(P_) > ave?

        if pdert_val <= 0:
            Pp.pdert_ = []  # remove pdert_


def intra_Pp_(Pp_, param_name, fPd):  # evaluate for sub-recursion in line Pm_, pack results into sub_Pm_

    comb_sublayers = []  # combine into root P sublayers[1:]
    comb_subDerts=[]
    # each Pp is evaluated for incremental range and derivation xcomp, as in line_patterns but with local aves

    for Pp in Pp_:  # each sub_layer is nested to depth = sublayers[n]
        if Pp.L > 2:  # no rel_adj_M = adj_M / -P.M: discontinuous search?
            mean_M = Pp.M / Pp.L  # for internal Pd eval, +opposite-side mean_M?

            if fPd:  # Pp is Ppd
                if abs(Pp.D) * mean_M > ave_D * Pp.Rdn and Pp.L > 3:  # mean_M from adjacent +ve Ppms
                    # search in top sublayer, eval by pdert.d:
                    sub_search_draft(Pp.P_, fPd)
                    rdn_ = [rdn + 1 for rdn in Pp.rdn_[:-1]]
                    ddert_ = []
                    # higher derivation comp, if sublayers.D?
                    for _pdert, pdert in zip( Pp.pdert_[:-1], Pp.pdert_[1:]):  # Pd.pdert_ is dert1_
                        _param = _pdert.d; param = pdert.d
                        dert = comp_param(_param, param, param_name[0], ave)  # cross-comp of ds in dert1_, !search, local aves?
                        ddert_ += [ Cdert( i=dert.i, p=dert.p, d=dert.d, m=dert.m)]
                    # cluster Pd derts by md sign:
                    form_Pp_(Pp, ddert_, param_name, rdn_, Pp.P_, fPd=True)

            else:  # Pp is Ppm
                if Pp.M > 0 and Pp.M > ave_M * Pp.Rdn and param_name=="I_":  # and if variable cost: Pp.M / Pp.L? -lend to contrast?
                    # search in top sublayer, eval by pdert.m:
                    sub_search_draft(Pp.P_, fPd)
                    # +Ppm -> sub_Ppm_: low-variation span, eval rng_comp:
                    rdn_ = [rdn+1 for rdn in Pp.rdn_[:-1]]
                    # range extended by incr ave: less term by match, and decr proj_P = dert.m * rave ((M/L) / ave): less term by miss
                    rpdert_, rP_ = search_param_(Pp.P_, (ave + Pp.M) / 2, rave = Pp.M / ave )  # rpdert_len-=1 in search_param:
                    form_Pp_(Pp, rpdert_, param_name, rdn_[:-1], rP_, fPd=False)  # cluster by m sign, eval intra_Pm_

                elif -Pp.M > ave_D * Pp.Rdn:  # high-variation span, -M is contrast borrowed from adjacent +Ppms: or abs D: likely sign match span?
                    # -Ppm -> sub_Ppd:
                    rdn_ = [rdn+1 for rdn in Pp.rdn_]
                    form_Pp_(Pp, Pp.pdert_, param_name, rdn_, Pp.P_, fPd=True)  # cluster by d sign: partial d match, eval intra_Pm_(Pdm_)


            if Pp.sublayers: # splice sublayers from all sub_Pp calls in Pp:
                # combine sublayers
                new_sublayers = []
                for comb_subset_, subset_ in zip_longest(comb_sublayers, Pp.sublayers, fillvalue=([])):
                    # append combined subset_ (array of sub_P_ param sets):
                    new_sublayers.append(comb_subset_ + subset_)
                comb_sublayers = new_sublayers
                # combine subDerts
                new_subDerts = []
                for comb_Dert_, Dert_ in zip_longest(comb_subDerts, Pp.subDerts, fillvalue=([])):
                    if Dert_ or comb_Dert_: # at least one is not empty
                        new_Dert_ = [comb_param + param
                                    for comb_param, param in
                                    zip_longest(comb_Dert_, Dert_, fillvalue=0)]
                        new_subDerts.append(new_Dert_)
                comb_subDerts = new_subDerts

    return comb_sublayers, comb_subDerts


def sub_search_draft(P_, fPd):  # search in top sublayer per P / sub_P, after P_ search: top-down induction,
    # called from intra_Pp_, especially MPp: init select by P.M, then combined Pp match?

    for P in P_:
        if P.sublayers: # not empty sublayer
            if P.sublayers[0]:  # not empty 1st layer subset, P.sublayers[0][0] is Dert
                subset = P.sublayers[0][0]  # top sublayer subset_ is one array
                # if pdert.m, eval per P, Idert or Ddert only?
                sub_P_ = subset[3]
                if len(sub_P_) > 2:
                    if fPd:
                        if abs(P.D) > ave_D:  # or if P.D + pdert.d + sublayer.Dert.D: P.sublayers[0][0][2]?
                            sub_rdn_Pp__ = search(sub_P_, fPd)
                            subset[5].append(sub_rdn_Pp__)
                            # recursion via form_P_
                    elif P.M > ave_M:  # or if P.M + pdert.m + sublayer.Dert.M: P.sublayers[0][0][3]?
                        sub_rdn_Pp__ = search(sub_P_, fPd)
                        subset[5].append(sub_rdn_Pp__)
                        # recursion via form_P_: deeper sublayers search is selective per sub_P


def comp_sublayers_draft(_P, P, pdert):  # if pdert.m -> if summed params m -> if positional m: mx0?

    param_names = ["L_", "I_", "D_", "M_"]
    aves = [ave_mL, ave_mI, ave_mD, ave_mM]

    for _subDert, subDert in zip(_P.subDerts, P.subDerts):
        # comp Derts, accumulate dert.sub_M:
        for _param, param, param_name, ave in zip(_subDert, subDert, param_names, aves):
            dert = comp_param(_param, param, param_name, ave)
            pdert.sub_M += dert.m  # higher-value mL?
            # add dertDert per _subDert, also a copy for subDert?

    if pdert.sub_M > ave_M * 4:  # or pdert.sub_M + pdert.m + P.M?
        # comp sub_Ps between sub_P_s in 1st sublayer:
        _fPd, _rdn, _rng, _sub_P_, _sub_pdert_, _sub_Pp__ = _P.sublayers[0]
        fPd, rdn, rng, sub_P_, sub_pdert_, sub_Pp__ = P.sublayers[0]
        # if same intra_comp fork, else sub_Ps are not comparable:
        if fPd == _fPd and rng == _rng and min(_P.L, P.L) > ave_Ls:
            if pdert.sub_M > 0:  # compare sub_Ps to each _sub_P within max relative distance, comb_M- proportional:
                _SL = SL = 0  # summed Ls
                start_index = next_index = 0  # index of starting sub_P for current _sub_P
                _sub_pdert_.append([])  # per sub_P_, selective nesting: [_sub_P [ sub_P_ [dir_dert_ [sub_pdert]]]]:

                for _sub_P in _sub_P_:  # doesn't form Pps: short range and long distance? #  _sub_pdert_right and _sub_pdert_left?
                    _SL += _sub_P.L  # ix0 of next _sub_P
                    # search right:
                    for sub_P in sub_P_[start_index:]:  # index_ix0 > _ix0
                        right_pdert_, fbreak = search_dir(_sub_P, sub_P, pdert, param_names, aves)
                        _sub_pdert_[-1].append( right_pdert_)  # preserve nesting
                        if fbreak:
                            break  # only sub_Ps with relatively proximate position in sub_P_|_sub_P_ are compared
                        if SL < _SL:
                            next_index += 1  # if next ix overlap: ix0 of next _sub_P < ix0 of current sub_P
                        SL += sub_P.L  # ix0 of next sub_P
                    # search left:
                    for sub_P in reversed( sub_P_[ len(sub_P_) - start_index:]):  # index_ix0 <= _ix0
                        left_pdert_, fbreak = search_dir(_sub_P, sub_P, pdert, param_names, aves)
                        _sub_pdert_[-1].append(left_pdert_)  # preserve nesting
                        if fbreak:
                            break  # only sub_Ps with relatively proximate position in sub_P_|_sub_P_ are compared
                    # for next _sub_P:
                    start_index = next_index


def search_dir(_sub_P, sub_P, pdert, param_names, aves):

    fbreak = 0
    dir_pdert_ = []
    dist_decay = 2  # decay of projected match with relative distance between sub_Ps

    distance = (sub_P.x0 + sub_P.L / 2) - (_sub_P.x0 + _sub_P.L / 2)  # distance between mean xs
    rel_distance = distance / (_sub_P.L + sub_P.L) / 2  # mean L; gap and overlap are edge-specific?
    # or comp all, then filter by distance?
    if ((_sub_P.M + sub_P.M) / 2 + pdert.m) * rel_distance * dist_decay > ave_M:
        sub_pdert = Cpdert()
        # 1x1 comp, add search_param, sum_rdn, etc, as in search?
        for param_name, ave in zip([param_names], aves):
            _param = getattr(_sub_P, param_name[0])
            param = getattr(sub_P, param_name[0])

            if param_name != "I_" or _sub_P.fPd:
                dert = comp_param(_param, param, param_name, ave)
                sub_pdert.sub_M += dert.m  # high-value mL: macro-param?
            else:
                # if param_name == "I_": sub_pdert = search_param_(param_)
                sub_dert = comp_param(_sub_P.I, sub_P.I, "I_", ave_mI)
                sub_pdert.sub_M += sub_dert.m  # between whole compared sub_Hs
                sub_pdert.sub_D += sub_dert.d

        dir_pdert_.append(sub_pdert)  # _sub_pdert_right or _sub_pdert_left
    else:
        fbreak = 1  # only sub_Ps with relatively proximate position in sub_P_|_sub_P_ are compared

    return dir_pdert_, fbreak


def draw_PP_(image, frame_Pp__):

    from matplotlib import pyplot as plt
    import numpy as np

    # initialization
    img_rval_Pp_ = [np.zeros_like(image) for _ in range(4)]

    img_Pp_ = [np.zeros_like(image) for _ in range(4)]
    img_Pp_pdert_ = [np.zeros_like(image) for _ in range(4)]

    img_Pp_layer_ = [np.zeros_like(image) for _ in range(4)]
    draw_layer = 1 # draw certain layer, start from 1, the higher the number, the deeper the layer, 0 = root layer so not applicable here

    param_names = ['L', 'I', 'D', 'M']

    for y, (rval_Pp__, Pp__) in enumerate(frame_Pp__):  # loop each line
        for i, (rval_Pp_, Pp_) in enumerate(zip(rval_Pp__, Pp__)): # loop each rdn_Pp or Pp
            # rval_Pp
            for j, (Rval, rval_Pps) in enumerate(rval_Pp_):
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

                # sub_Pps
                for k, sub_P_layers in enumerate(Pp.sublayers): # each layer
                    if k+1 == draw_layer:
                        for (_, Pp_) in enumerate(sub_P_layers[0]): # each sub_P's Pps
                            for n, P in enumerate(Pp.P_): # each P or pdert
                                if Pp.M>0:
                                    img_Pp_layer_[i][y,P.x0:P.x0+P.L] = 255 # + sign
                                else:
                                    img_Pp_layer_[i][y,P.x0:P.x0+P.L] = 128 # - sign
                        break # draw only selected layer

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
        plt.imshow(img_Pp_layer_[i], vmin=0, vmax=255)
        plt.title("Sub Pps, Layer = "+str(draw_layer)+", Param = " + param)

    plt.figure()
    for i, param in enumerate(param_names):
        plt.subplot(2, 2, i + 1)
        plt.imshow(img_Pp_pdert_[i], vmin=0, vmax=255)
        plt.title("Pderts, Param = " + param)
    pass