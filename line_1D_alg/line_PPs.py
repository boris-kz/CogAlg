'''
line_PPs is a 2nd-level 1D algorithm, its input is P_ formed by the 1st-level line_patterns.
It cross-compares P params (initially L, I, D, M) and forms param_Ps: Pp_ for each param type per image row.
-
Subsequent cross-comp between Pps of different params is exclusive of x overlap, where the relationship is already known.
Thus it should be on 3rd level: no Pp overlap means comp between Pps: higher composition, same-type ) cross-type?
-
Next, line_PPPs should be formed as a cross-level increment: line_PPPs = increment (line_PPs).
This increment will be made recursive: we should be able to automatically generate 3rd and any n+1- level alg:
line_PPPPs = increment (line_PPPs), etc. That will be the hardest and most important part of this project
'''

import sys  # add CogAlg folder to system path
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname("CogAlg"), '..')))
import numpy as np
from line_patterns import *
from frame_2D_alg.class_cluster import ClusterStructure, comp_param

class Cpdert(ClusterStructure):
    # P param dert
    i = int  # input for range_comp only
    p = int  # accumulated in rng
    d = int  # accumulated in rng
    m = int  # distinct in deriv_comp only
    rdn = int   # P.Rdn + param rdn?
    negM = int  # in mdert only
    negL = int  # in mdert only
    negiL = int
    sub_M = int  # match from comp_sublayers, if any
    sub_D = int  # diff from comp_sublayers, if any
    Pp = object  # Pp that pdert is in, for merging in form_Pp_rng, temporary?

class CPp(CP):
    pdert_ = list  # Pp elements, "p" for param
    P_ = list  # zip with pdert_
    Rdn = int  # cross-param rdn accumulated from pderts
    rdn_ = list # for sub's workflow
    rval = int  # not needed?  Pp value (M | abs D) adjusted for cross-param Rdn
    iL = int  # length of Pp in pixels
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
ave_d = 10  # redefine as ave_coef
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

# used in search, form_Pp_root, comp_sublayers_, draw_PP_:
param_names = ["L_", "I_", "D_", "M_"]
aves = [ave_mL, ave_mI, ave_mD, ave_mM]


def line_PPs_root(rootPp, rdn, root_P_t):  # root_P_t: Pm_, Pd_; higher-level input is nested to the depth = 2*elevation (level counter), or 2^elevation?

    Pp_ttt = []  # 3-level nested tuple per line: Pm_, Pd_( Ppm_, Ppd_( LPp_, IPp_, DPp_, MPp_)))
    if not rootPp: norm_feedback(root_P_t)  # before processing

    for i, P_ in enumerate(root_P_t):  # fPd = i: Pm_| Pd_
        if len(P_) > 1:
            # form Pp_ tuple, as in line_patterns cross_comp:
            Pdert_t, dert1_, dert2_ = search(P_, rdn, i)  # returns LPp_, IPp_, DPp_, MPp_
            Ppm_t = form_Pp_root(rootPp, Pdert_t, dert1_, dert2_, fPd=False)  # eval intra_Pp_ (calls form_Pp_)
            Ppd_t = form_Pp_root(rootPp, Pdert_t, dert1_, dert2_, fPd=True)
            # each element of Ppm_t and Ppd_t is LPp_, IPp_, DPp_, MPp_
            Pp_ttt.append((Ppm_t, Ppd_t))
        else:
            Pp_ttt.append(P_)

    return Pp_ttt  # 3-level nested tuple per line: Pm_, Pd_( Ppm_, Ppd_( LPp_, IPp_, DPp_, MPp_)))


def search(P_, rdn, fPd):  # cross-compare patterns within horizontal line

    Ldert_, Idert_, Ddert_, Mdert_, dert1_, dert2_, LP_, IP_, DP_, MP_ = [], [], [], [], [], [], [], [], [], []

    for _P, P, P2 in zip(P_, P_[1:], P_[2:] + [CP()]):  # for P_ cross-comp over step=1 and step=2
        _L, _I, _D, _M, *_ = _P.unpack()  # *_: skip remaining params
        L, I, D, M, *_ = P.unpack()
        D2, M2 = P2.D, P2.M
        # special div_comp for L:
        rL = L / _L  # higher order of scale, not accumulated: no search, rL is directional
        int_rL = int( max(rL, 1/rL))
        frac_rL = max(rL, 1/rL) - int_rL
        mL = int_rL * min(L, _L) - (int_rL*frac_rL) / 2 - ave_mL  # div_comp match is additive compression: +=min, not directional
        Ldert_.append(Cdert( i=L, p=L + _L, d=rL, m=mL))  # add mrdn if default form Pd?
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
    dert2_ = dert2_[:-1]  # due to filled CP() in P2 ( for loop above )
    if not fPd:
        Idert_, IP_ = search_param_(P_, ave_mI, rdn, rave=1)  # comp x variable range, depending on M of Is
        Mdert_ = Mdert_[:-1]  # due to filled CP() in P2 ( for loop above )
        DP_, MP_ = P_[:-1], P_[:-2]
    else:
        IP_, DP_, MP_ = P_[:-1], P_[:-2], P_[:-1]

    Pdert_t = (Ldert_, LP_), (Idert_, IP_), (Ddert_, DP_), (Mdert_, MP_)

    return Pdert_t, dert1_, dert2_


def search_param_(P_, ave, rdn, rave):  # variable-range search in mdert_, only if param is core param?

    # higher local ave for extended rng: -> lower m and term by match, and higher proj_M?
    Idert_, _P_ = [], []  # line-wide (i, p, d, m, negL, negM, negiL)

    for i, _P in enumerate( P_[:-1]):
        negM = 0
        _pI = _P.I - (_P.D / 2)  # forward project by _D
        j = i + 1  # init with positive-M Is only: internal match projects xP I match:

        while _P.M + negM > ave_M and j < len(P_):  # starts with positive _P.Ms, but continues over negM if > ave
            P = P_[j]
            pI = P.I + (P.D / 2)  # back-project by D
            dert = comp_param(_pI, pI, "I_", ave)  # param is compared to prior-P _param
            pdert = Cpdert(i=dert.i, p=dert.p, d=dert.d, m=dert.m)  # convert Cdert to Cpdert
            curr_M = pdert.m * rave + (_P.M + P.M) / 2  # P.M is bilateral, no fPd in search_param

            if curr_M > ave_sub * rdn and _P.sublayers[0] and P.sublayers[0]:  # comp sub_P_s, for core I only?
                comp_sublayers(P_[i], P_[j], pdert.m)  # forms pdert.sub_M:
            if curr_M + pdert.sub_M > ave_M * rdn * 4:  # ave_cM
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
            del pdert  # prevent reuse of same pdert in multiple loops

    return Idert_, _P_


def form_Pp_root(rootPp, Pdert_t, pdert1_, pdert2_, fPd):  # add rootPp for form_Pp_, if called from intra_Pp_

    rdn__ = sum_rdn_(param_names, Pdert_t, fPd=fPd)  # assign redundancy to lesser-magnitude m|d in param pair for same-_P Pderts
    Pp_t = []

    for param_name, (Pdert_, P_), rdn_ in zip(param_names, Pdert_t, rdn__):  # segment Pdert__ into Pps
        if param_name == "I_" and not fPd:
            Pp_ = form_Pp_rng(rootPp, P_, Pdert_, pdert1_, pdert2_, rdn_)  # eval splice P_ by match induction, no I mag induction
        else:
            Pp_ = form_Pp_(rootPp, P_, Pdert_, param_name, rdn_, fPd=0)
        Pp_t.append(Pp_)  # Ppm | Ppd

    return Pp_t

def form_Pp_(rootPp, P_, pdert_, param_name, rdn_, fPd):  # lay_rdn: layer rdn, as in line_patterns
    # initialization:
    Pp_ = []
    x = 0
    _sign = None  # to initialize 1st P, (None != True) and (None != False) are both True

    for pdert, rdn, P in zip(pdert_, rdn_, P_):  # segment by sign
        if fPd: sign = pdert.d > 0
        else:   sign = pdert.m > 0
        # adjust by ave projected at distance=negL and contrast=negM, if significant:
        # m + ddist_ave = ave - ave * (ave_rM * (1 + negL / ((param.L + _param.L) / 2))) / (1 + negM / ave_negM)?
        if sign != _sign:
            # sign change, compact terminated Pp, initialize Pp and append it to Pp_
            if Pp_:  # empty at 1st sign change
                Pp = Pp_[-1]; Pp.I /= Pp.L; Pp.D /= Pp.L; Pp.M /= Pp.L  # immediate normalization
            Pp = CPp( L=1, iL=P_[x].L, I=pdert.p, D=pdert.d, M=pdert.m, Rdn=rdn, rdn_ = [rdn], x0=x, ix0=P_[x].x0, pdert_=[pdert], P_=[P_[x]], sublayers=[])
            Pp_.append(Pp)  # updated by accumulation below
        else:
            # accumulate params:
            Pp.L += 1; Pp.iL += P_[x].L; Pp.I += pdert.p; Pp.D += pdert.d; Pp.M += pdert.m; Pp.Rdn += rdn; Pp.rdn_ += [rdn]; Pp.pdert_ += [pdert]; Pp.P_ += [P]
        x += 1
        _sign = sign

    intra_Pp_(rootPp, Pp_, param_name, fPd)

    return Pp_


def form_Pp_rng(rootPp, P_, pdert_, pdert1_, pdert2_, rdn_):  # cluster Pps by cross-param redundant value sign, eval for cross-level rdn

    Pp_ = []  # multiple Pps may overlap within _dert.negL

    for i, (_pdert, _P, _rdn) in enumerate(zip(pdert_, P_, rdn_)):
        if _pdert.m + _P.M > ave *_rdn:  # positive Pps only, else too much overlap? +_P.M: value is combined across P levels?
            # initialize Pp:
            if not isinstance(_pdert, Cpdert): _pdert = Cpdert(i=_pdert.i, p=_pdert.p, d=_pdert.d, m=_pdert.m)  # convert Cdert to Cpdert
            if not isinstance(_pdert.Pp, CPp):  # _pdert is not in any Pp
                Pp = CPp(L=1, iL=_P.L, I=_pdert.p, D=_pdert.d, M=_pdert.m, Rdn=_rdn, rdn_ = [_rdn], negiL=_pdert.negiL, negL=_pdert.negL, negM=_pdert.negM,
                         x0=i, ix0=_P.x0, pdert_=[_pdert], P_=[_P], sublayers=[])
                _pdert.Pp = Pp
                Pp_.append(Pp)  # params will be accumulated
            else:
                break  # this _dert already searched forward
            j = i + _pdert.negL + 1

            while (j <= len(pdert_)-1):
                pdert = pdert_[j]; P = P_[j]; rdn = rdn_[j]  # no pop: maybe used by other _derts
                if not isinstance(pdert, Cpdert): pdert = Cpdert(i=pdert.i, p=pdert.p, d=pdert.d, m=pdert.m)  # convert Cdert to Cpdert
                if pdert.m + P.M > ave*rdn:
                    if isinstance(pdert.Pp, CPp):  # unique Pp per dert in row Pdert_
                        # merge Pp with dert.Pp, if any:
                        Pp.accum_from(pdert.Pp,excluded=['x0'])
                        Pp.P_ += pdert.Pp.P_
                        Pp.pdert_ += pdert.Pp.pdert_
                        Pp.rdn_ += pdert.Pp.rdn_
                        Pp.sublayers += pdert.Pp.sublayers
                        Pp_.remove(pdert.Pp)
                        pdert.Pp = Pp
                        break  # this dert already searched forward
                    else:  # accumulate params:
                        Pp.L += 1; Pp.iL += P.L; Pp.I += pdert.p; Pp.D += pdert.d; Pp.M += pdert.m; Pp.Rdn += rdn; Pp.rdn_ += [rdn]; Pp.negiL += pdert.negiL
                        Pp.negL += pdert.negL; Pp.negM += pdert.negM; Pp.pdert_ += [pdert]; Pp.P_ += [P]
                        # no dert.Pp, only 1st dert in pdert_ is checked; Pp derts already searched dert_, they won't be new _derts
                        j += pdert.negL+1
                else:
                    # Pp is terminated
                    if Pp_:  # empty at 1st sign change
                        Pp = Pp_[-1]  # eval splice Ps, if search_param only: match induction, I is not value, no mag induction
                        Pp.I /= Pp.L; Pp.D /= Pp.L; Pp.M /= Pp.L  # immediate normalization
                        if Pp.M > ave_M * 4:  # splice_ave
                            compact(Pp_, pdert1_, pdert2_, "I", rdn, fPd=0)  # re-eval Pps, Pp.pdert_s for redundancy, eval splice Ps
                    break

    intra_Pp_(rootPp, Pp_, "I_", fPd=0)  # deep layers feedback, add total_rdn?

    return Pp_


def sum_rdn_(param_name_, Pdert__, fPd):
    '''
    access same-index pderts of all P params, assign redundancy to lesser-magnitude m|d in param pair.
    if other-param same-P_-index pdert is missing, rdn doesn't change.
    To include layer_rdn: initialize each rdn at 1 vs. 0, then increment here and in intra_Pp_?
    '''
    if fPd: alt = 'M'
    else:   alt = 'D'
    name_pairs = (('I', 'L'), ('I', 'D'), ('I', 'M'), ('L', alt), ('D', 'M'))  # pairs of params redundant to each other
    pderts_Rdn = [[], [], [], []]  # L_, I_, D_, M_' Rdns, as in pdert__

    for Ldert, Idert, Ddert, Mdert in zip_longest(Pdert__[0][0], Pdert__[1][0], Pdert__[2][0], Pdert__[3][0], fillvalue=Cdert()):
        # pdert per _P in P_, 0: Ldert_, 1: Idert_, 2: Ddert_, 3: Mdert_
        # P M|D rdn + dert m|d rdn:
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

            pderts_Rdn[i].append(Rdn)  # same length as Pdert_, sum with P.Rdn?

    return pderts_Rdn  # rdn__


def compact(Pp_, pdert1_, pdert2_, param_name, fPd):  # re-eval Pps, Pp.pdert_s for redundancy, eval splice Ps

    Pp = Pp_[-1]
    for i, P, rdn in zip( enumerate(Pp.P_, Pp.rdn_)):
        # assign cross-level rdn (Pp vs. pdert_), re-evaluate Pp and pdert_:
        # wrong, need a review:
        P_val = P.Rdn / P.L * rdn * ave  # / Pp.L: resolution reduction, but lower rdn:
        dert_val = P.Rdn * rdn * ave  # * Pp.L: ave cost * number of representations

        if P_val > dert_val: dert_val -= ave * P.Rdn
        else: P_val -= ave * P.Rdn  # ave scaled by rdn
        if P_val <= 0:
            Pp = CPp(pdert_=Pp.pdert_)  # Pp remove: reset Pp vars to 0

        elif ((param_name == "I_") and not fPd) or ((param_name == "D_") and fPd):  # P-defining params, else no separation
            M2 = M1 = 0
            # param match over step=2 and step=1:
            for pdert2 in pdert2_: M2 += pdert2.m  # match(I, __I or D, __D)
            for pdert1 in pdert1_: M1 += pdert1.m  # match(I, _I or D, _D)

            if M2 / abs(M1) > ave_splice:  # similarity / separation: splice Ps in Pp, also implies weak Pp.pdert_?
                _P = CP()
                for P in Pp.P_:
                    _P.accum_from(P, excluded=["x0"])  # different from Pp params
                    _P.dert_ += [P.dert_]  # splice dert_s within Pp
                Pp_[-1].P_ = _P  # replace with spliced P

        # if pdert_val <= 0:
        #    Pp.pdert_ = []  # remove pdert_


def intra_Pp_(rootPp, Pp_, param_name, fPd):  # evaluate for sub-recursion in line Pm_, pack results into sub_Pm_

    comb_sublayers = []  # combine into root P sublayers[1:]
    # each Pp may be compared over incremental range and derivation, as in line_patterns but with local aves

    for Pp in Pp_:  # each sub_layer is nested to depth = sublayers[n]
        if Pp.L > 1:
            if fPd:  # Pp is Ppd
                if abs(Pp.D) + Pp.M > ave_D * Pp.Rdn and Pp.L > 2:  # + Pp.M: borrow potential, regardless of Rdn?
                    # search in top sublayer, eval by pdert.d:
                    sub_search_draft(Pp, Pp.P_, fPd)
                    rdn_ = [rdn + 1 for rdn in Pp.rdn_[:-1]]  # that obviates layer_rdn += 1
                    sub_Ppm_, sub_Ppd_ = [], []
                    Pp.sublayers = [[( fPd, sub_Ppm_, sub_Ppd_ )]]
                    ddert_ = []  # higher derivation comp, if sublayers.D?
                    for _pdert, pdert in zip( Pp.pdert_[:-1], Pp.pdert_[1:]):  # Pd.pdert_ is dert1_
                        _param = _pdert.d; param = pdert.d
                        dert = comp_param(_param, param, param_name[0], ave)  # cross-comp of ds in dert1_, !search, local aves?
                        ddert_ += [ Cdert( i=dert.i, p=dert.p, d=dert.d, m=dert.m)]
                    # cluster ddert_:
                    sub_Ppm_[:] = form_Pp_(Pp, Pp.P_, ddert_, param_name, rdn_, fPd=False)
                    sub_Ppd_[:] = form_Pp_(Pp, Pp.P_, ddert_, param_name, rdn_, fPd=True)
                else:
                    Pp.sublayers += [[]]  # empty subset to preserve index in sublayer, or increment index of subset?

                # replace the above with:
                line_PPs_root(rootPp, rdn_, root_P_t)
                '''
                root_P_t consists of two ((Ldert_, LP_), (Idert_, IP_), (Ddert_, DP_), (Mdert_, MP_))s, for Pds and Pms.
                Instead of comparing P params, search will be comparing dparams in pderts, for each param.
                '''
            else:  # Pp is Ppm
                if Pp.M > ave_M * Pp.Rdn and param_name=="I_":  # variable costs, add fixed costs? -lend to contrast?
                    # search in top sublayer, eval by pdert.m:
                    sub_search_draft(Pp, Pp.P_, fPd)
                    # +Ppm -> sub_Ppm_: low-variation span, eval rng_comp:
                    rdn_ = [rdn+1 for rdn in Pp.rdn_[:-1]]  # that obviates lay_rdn += 1
                    sub_Ppm_, sub_Ppd_ = [], []
                    Pp.sublayers = [[( fPd, sub_Ppm_, sub_Ppd_ )]]
                    # range extended by incr ave: less term by match, and decr proj_P = dert.m * rave ((M/L) / ave): less term by miss
                    rpdert_, rP_ = search_param_(Pp.P_, Pp.rdn_, (ave + Pp.M) / 2, rave = Pp.M / ave )
                    # rpdert_len-=1 in search_param, cluster rpdert_ by rm:
                    if param_name == "I_" and not fPd:
                        sub_Ppm_[:] = form_Pp_rng(Pp, Pp.P_, rpdert_, [], [], rdn_)
                    else:
                        sub_Ppm_[:] = form_Pp_(Pp, Pp.P_, rpdert_, param_name, rdn_, fPd=False)  # cluster by m sign, eval intra_Pm_
                    sub_Ppd_[:] = form_Pp_(Pp, Pp.P_, rpdert_, param_name, rdn_, fPd=True)  # cluster by d sign: partial d match
                else:
                    Pp.sublayers += [[]]  # empty subset to preserve index in sublayer

                # replace the above with:
                line_PPs_root(rootPp, rdn_, root_P_t)
                '''
                root_P_t consists of two ((Ldert_, LP_), (Idert_, IP_), (Ddert_, DP_), (Mdert_, MP_))s, for Pds and Pms.
                Search will be comparing params themselves at incremental rng, +1 relative to previous range, also for each param. 
                For search_param, initial comp range is neg_L +1, with ave adjusted by Pp.M
                '''
            if rootPp and Pp.sublayers:
                comb_sublayers = [comb_subset_ + subset_ for comb_subset_, subset_ in
                                  zip_longest(comb_sublayers, Pp.sublayers, fillvalue=[])
                                  ]
    if rootPp:
        rootPp.sublayers += comb_sublayers
    else:
        return Pp_  # each Pp has new sublayers, comb_sublayers is not needed


def sub_search_draft(rootPp, P_, fPd):  # search in top sublayer per P / sub_P, after P_ search: top-down induction,
    # called from intra_Pp_, especially MPp: init select by P.M, then combined Pp match?
    for P in P_:
        if P.sublayers[0]:  # not empty sublayer
            subset = P.sublayers[0][0]  # top sublayer subset_ is one array
            root_P_t = [subset[2], subset[3]]
            line_PPs_root(rootPp, rootPp.rdn_, root_P_t)

            # mostly obsolete:
            for i, sub_P_ in enumerate([subset[2], subset[3]]):  # fPd = i: rval_Pm_| rval_Pd_
                if len(sub_P_) > 2:
                    sub_P_ = splice(sub_P_, fPd)  # for discontinuous search
                     # or if P.D + pdert.d + sublayer.Dert.D
                    if (fPd and abs(P.D) > ave_D) or (P.M > ave_M): # or if P.M + pdert.m + sublayer.Dert.M
                        sub_Pdert_t, sub_dert1_, sub_dert2_ = search(sub_P_, rdn, fPd=i)
                        sub_Ppm_t = form_Pp_root(rootPp, sub_Pdert_t, sub_dert1_, sub_dert2_, rdn, fPd=False)
                        sub_Ppd_t = form_Pp_root(rootPp, sub_Pdert_t, sub_dert1_, sub_dert2_, rdn, fPd=True)
                        subset[6].append(sub_Ppm_t)
                        subset[7].append(sub_Ppd_t)
                    '''
                    same as recursion via line_PPs_root:
                    Pdert_t, dert1_, dert2_ = search(P_, rdn, i)  # returns LPp_, IPp_, DPp_, MPp_
                    Ppm_t = form_Pp_root(rootPp, Pdert_t, dert1_, dert2_, rdn, fPd=False)  # eval intra_Pp_ (calls form_Pp_)
                    Ppd_t = form_Pp_root(rootPp, Pdert_t, dert1_, dert2_, rdn, fPd=True)
                    Pp_ttt.append((Ppm_t, Ppd_t))
                else:
                    Pp_ttt.append(P_)

                    '''
                        #  deeper sublayers search is selective per sub_P


def comp_sublayers(_P, P, root_v):  # if pdert.m -> if summed params m -> if positional m: mx0?
    # replace root_v with root_vt: separate evaluations?

    xDert_vt = form_comp_Derts(_P, P, root_v)  # value tuple: separate vm, vd for comp sub_mP_ or sub_dP_
    # comp sub_Ps between 1st sublayers of _P and P:
    # eval @ call
    _rdn, _rng = _P.sublayers[0][0][:2]  # 2nd [0] is the only subset
    rdn, rng  = P.sublayers[0][0][:2]  # P.sublayers[0][0]: rdn, rng, sub_Pm_, sub_Pd_, xsub_pmdertt_, xsub_pddertt_

    for i, xDert_v in enumerate(xDert_vt):  # eval m, d:
        if xDert_v + root_v > ave_M * (i * ave_D) * rdn:  # ave_D should be defined as coef
            _sub_P_, _,  _xsub_pdertt_ = _P.sublayers[0][0][2+i: 4+i+1]
            sub_P_, _, xsub_pdertt_ = P.sublayers[0][0][2+i: 4+i+1]
            if rng == _rng and min(_P.L, P.L) > ave_Ls:
                # if same intra_comp fork: compare sub_Ps to each _sub_P within max relative distance, comb_V- proportional:
                _SL = SL = 0  # summed Ls
                start_index = next_index = 0  # index of starting sub_P for current _sub_P
                _xsub_pdertt_ += [[]]  # array of cross-sub_P pdert tuples, inner brackets for subset_
                xsub_pdertt_ += [[]]  # append xsub_dertt per _sub_P_ and sub_P_, sparse?

                for _sub_P in _sub_P_:
                    P_ = []  # to form xsub_Pps
                    _xsub_pdertt = [[], [], [], []]  # tuple of L, I, D, M xsub_pderts
                    _SL += _sub_P.L  # ix0 of next _sub_P
                    # search right:
                    for sub_P in sub_P_[start_index:]:  # index_ix0 > _ix0, comp sub_Ps at proximate relative positions in sub_P_
                        if comp_sub_P(_sub_P, sub_P, _xsub_pdertt, P_, root_v, i):
                            break
                        # if next ix overlap: ix0 of next _sub_P < ix0 of current sub_P
                        if SL < _SL: next_index += 1
                        SL += sub_P.L  # ix0 of next sub_P
                    # search left:
                    for sub_P in reversed(sub_P_[len(sub_P_) - start_index:]):  # index_ix0 <= _ix0
                        if comp_sub_P(sub_P, _sub_P, _xsub_pdertt, P_, root_v, i):  # invert sub_P, _sub_P positions
                            break
                    # not implemented: if param_name == "I_" and not fPd: sub_pdert = search_param_(param_)
                    start_index = next_index  # for next _sub_P

                    if _xsub_pdertt[0]:  # at least 1 sub_pdert, real min length ~ 8, very unlikely
                        sub_Pdertt_ = [(_xsub_pdertt[0], P_), (_xsub_pdertt[1], P_), (_xsub_pdertt[2], P_), (_xsub_pdertt[3], P_)]
                        # form 4-tuple of xsub_Pp_s:
                        xsub_Pp_t = form_Pp_root(None, sub_Pdertt_, [], [], rdn+1, fPd=i)
                        _xsub_pdertt_[-1][:] = xsub_Pp_t
                        xsub_pdertt_[-1][:] = _xsub_pdertt_[-1]  # bilateral assignment
                    else:
                        _xsub_pdertt_[-1].append(_xsub_pdertt)  # preserve nesting


def form_comp_Derts(_P, P, root_v):

    subDertt_t = [[],[]]  # two comparands, each preserved for future processing
    # form Derts:
    for _sublayer, sublayer in zip(_P.sublayers, P.sublayers):
        for i, isublayer in enumerate([_sublayer, sublayer]):  # list of subsets: index 0 = _P.sublayers, index 1 = P.sublayers
            nsub = 0
            for subset in isublayer:
                nsub += len(subset[2]) + len(subset[3])
                # or nsub_t, for separate form mDert | dDert eval?
            if root_v * nsub > ave_M * 5:  # ave_Dert, separate eval for m and d?
                subDertt = [[0, 0, 0, 0], [0, 0, 0, 0]]  # Pm,Pd' L,I,D,M summed in sublayer
                for rdn, rng, sub_Pm_, sub_Pd_, xsub_pmdertt_, xsub_pddertt_, _, _ in isublayer: # current layer subsets
                    for j, sub_P_ in enumerate([sub_Pm_, sub_Pd_]):  # j: Pm | Pd
                        for sub_P in sub_P_:
                            subDertt[j][0] += sub_P.L
                            subDertt[j][1] += sub_P.I
                            subDertt[j][2] += sub_P.D
                            subDertt[j][3] += sub_P.M
                subDertt_t[i].append(subDertt)  # 2-tuple, index 0 = _P, index 1 = P
            else:
                break  # deeper Derts are not formed
    _P.subDertt_= subDertt_t[0]
    P.subDertt_ = subDertt_t[1]
    # comp Derts, sum xlayer:
    xDert_vt = [0,0]
    derDertt = [[],[]]
    for (_mDert, _dDert), (mDert, dDert) in zip(subDertt_t[0], subDertt_t[1]):
        for i, (_iDert, iDert) in enumerate( zip((_mDert, _dDert), (mDert, dDert))):
            derDert = []
            for _param, param, param_name, ave in zip(_iDert, iDert, param_names, aves):
                dert = comp_param(_param, param, param_name, ave)
                if i:  # higher value for L?
                    vd = abs(dert.d) - ave_d
                    if vd > 0: xDert_vt[i] += vd  # no neg value sum: -Ps won't be processed
                elif dert.m > 0:
                    xDert_vt[i] += dert.m
                derDert.append(dert)  # dert per param, derDert_ per _subDert, copy in subDert?
            derDertt[i].append(derDert)  # m, d derDerts per sublayer

        _P.derDertt_.append(derDertt)  # derDert_ per _P, P pair
        P.derDertt_.append(derDertt)  # bilateral assignment?

    return xDert_vt
'''
xsub_pderts (cross-sub_P): xsub_pdertt_[ xsub_pdertt [ xsub_pdert_[ sub_pdert]]]: each bracket is a level of nesting
the above forms Pps across xsub_dertt, then also form Pps across _xsub_pdertt_ and xsub_pdertt_? 
'''

def comp_sub_P(_sub_P, sub_P, xsub_pdertt, P_, root_v, fPd):
    fbreak = 0
    xsub_P_vt = [0,0]  # vm,vd combined across params
    dist_decay = 2  # decay of projected match with relative distance between sub_Ps

    distance = (sub_P.x0 + sub_P.L / 2) - (_sub_P.x0 + _sub_P.L / 2)  # distance between mean xs
    mean_L = (_sub_P.L + sub_P.L) / 2
    rel_distance = distance / mean_L  # not gap and overlap: edge-specific?
    if fPd:
        sub_V = abs((_sub_P.D + sub_P.D) / 2) - ave_d * mean_L
    else:
        sub_V = (_sub_P.M + sub_P.M) / 2
    V = sub_V + root_v
    # or comp all, then filter by distance?
    if V * rel_distance * dist_decay > ave_M * (fPd * ave_D):
        # 1x1 comp:
        for i, (param_name, ave) in enumerate(zip(param_names, aves)):
            _param = getattr(_sub_P, param_name[0])  # can be simpler?
            param = getattr(sub_P, param_name[0])
            dert = comp_param(_param, param, param_name, ave)
            sub_pdert = Cpdert(i=dert.i, p=dert.p, d=dert.d, m=dert.m)  # convert Cdert to Cpdert
            # no negative-value sum: -Ps won't be processed:
            if dert.m > 0: xsub_P_vt[0] += dert.m
            vd = abs(dert.d) - ave_d  # convert to coef
            if vd > 0: xsub_P_vt[1] += vd
            xsub_pdertt[i].append(sub_pdert)  # per L, I, D, M' xsub_pdert
            P_.append(sub_P)  # same sub_P for all xsub_Pps

        V += xsub_P_vt[0] + xsub_P_vt[1]  # or two separate values for comp_sublayers?
        if V > ave_M * 4 and _sub_P.sublayers[0] and sub_P.sublayers[0]:  # 4: ave_comp_sublayers coef
            # if sub_P.sublayers is not empty, then sub_P.sublayers[0] can't be empty
            comp_sublayers(_sub_P, sub_P, V)  # recursion for deeper layers
    else:
        fbreak = 1  # only sub_Ps with relatively proximate position in sub_P_|_sub_P_ are compared

    return fbreak


def norm_feedback(P_t):

    fbM = fbL = 0
    for P in P_t[0]:  # Pm_
        fbM += P.M; fbL += P.L
        if abs(fbM) > ave_Dave:
            if abs(fbM / fbL) > ave_dave:
                fbM = fbL = 0
                pass  # eventually feedback: line_patterns' cross_comp(frame_of_pixels_, ave + fbM / fbL)
                # also terminate Fspan: same-filter frame_ with summed params, re-init at all 0s
        P.I /= P.L; P.D /= P.L; P.M /= P.L  # immediate normalization to a mean
    for P in P_t[1]:  # Pd_
        P.I /= P.L; P.D /= P.L; P.M /= P.L

def splice(P_, fPd):  # currently not used, replaced by compact() in line_PPs
    '''
    Initial P termination is by pixel-level sign change, but resulting separation may be insignificant on a pattern level.
    That is, separating opposite-sign pattern is weak relative to separated same-sign patterns.
    The criterion to re-evaluate separation is similarity of P-defining param: M/L for Pm, D/L for Pd, among the three Ps
    If relative similarity > merge_ave: all three Ps are merged into one.
    '''
    splice_val_ = [splice_eval(__P, _P, P, fPd)  # compute splice values
                   for __P, _P, P in zip(P_, P_[1:], P_[2:])]
    sorted_splice_val_ = sorted(enumerate(splice_val_),
                                key=lambda k: k[1],
                                reverse=True)   # sort index by splice_val_
    if sorted_splice_val_[0][1] <= ave_splice:  # exit recursion
        return P_

    folp_ = np.zeros(len(P_), bool)  # if True: P is included in another spliced triplet
    spliced_P_ = []
    for i, splice_val in sorted_splice_val_:  # loop through splice vals
        if splice_val <= ave_splice:  # stop, following splice_vals will be even smaller
            break
        if folp_[i : i+3].any():  # skip if overlap
            continue
        folp_[i : i+3] = True     # splice_val > ave_splice: overlapping Ps folp=True
        __P, _P, P = P_[i : i+3]  # triplet to splice
        # merge _P and P into __P:
        __P.accum_from(_P, excluded=['x0', 'ix0'])
        __P.accum_from(P, excluded=['x0', 'ix0'])

        if hasattr(__P, 'pdert_'):  # for splice_Pp_ in line_PPs
            __P.pdert_ += _P.pdert_ + P.pdert_
        else:
            __P.dert_ += _P.dert_ + P.dert_
        spliced_P_.append(__P)

    # add remaining Ps into spliced_P
    spliced_P_ += [P_[i] for i, folp in enumerate(folp_) if not folp]
    spliced_P_.sort(key=lambda P: P.x0)  # back to original sequence

    if len(spliced_P_) > 4:
        splice(spliced_P_, fPd)

    return spliced_P_

def splice_eval(__P, _P, P, fPd):  # only for positive __P, P, negative _P triplets, needs a review
    '''
    relative continuity vs separation = abs(( M2/ ( M1+M3 )))
    relative similarity = match (M1/L1, M3/L3) / miss (match (M1/L1, M2/L2) + match (M3/L3, M2/L2)) # both should be negative
    or P2 is reinforced as contrast - weakened as distant -> same value, not merged?
    splice P1, P3: by proj mean comp, ~ comp_param, ave / contrast P2
    also distance / meanL, if 0: fractional distance = meanL / olp? reduces ave, not m?
    '''
    if fPd:
        if _P.D==0: _P.D =.1  # prevents /0
        rel_continuity = abs((__P.D + P.D) / _P.D)
        __mean= __P.D/__P.L; _mean= _P.D/_P.L; mean= P.D/P.L
    else:
        if _P.M == 0: _P.M =.1  # prevents /0
        rel_continuity = abs((__P.M + P.M) / _P.M)
        __mean= __P.M/__P.L; _mean= _P.M/_P.L; mean= P.M/P.L

    m13 = min(mean, __mean) - abs(mean-__mean)/2    # inverse match of P1, P3
    m12 = min(_mean, __mean) - abs(_mean-__mean)/2  # inverse match of P1, P2, should be negative
    m23 = min(_mean, mean) - abs(_mean- mean)/2     # inverse match of P2, P3, should be negative

    miss = abs(m12 + m23) if not 0 else .1
    rel_similarity = (m13 * rel_continuity) / miss  # * rel_continuity: relative value of m13 vs m12 and m23
    # splice_value = rel_continuity * rel_similarity

    return rel_similarity
