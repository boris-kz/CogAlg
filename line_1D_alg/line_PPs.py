'''
line_PPs is a 2nd-level 1D algorithm, its input is P_ formed by the 1st-level line_patterns.
It cross-compares P params (initially L, I, D, M) and forms param_Ps: Pp_ for each param type per row in the image.
In Pp: P stands for pattern and p for "partial" or "param": L|I|D|M of input P the Pp is formed from.
-
So Pp is a pattern of a specific input-P param type: span of same-sign match | diff between same-type params in P_ array.
This module is line_PPs vs. line_Pps because it forms Pps of all param types, which in combination represent patterns of patterns: PPs.
Conversion of line_Ps into line_PPs is manual: initial input formatting does not apply on higher levels
(initial inputs are filter-defined, vs. mostly comparison-defined for higher levels)
-
Cross-comp between Pps of different params is exclusive of x overlap, where the relationship is already known.
Thus it should be on 3rd level: no Pp overlap means comp between Pps: higher composition, same-type ) cross-type
'''

import sys  # add CogAlg folder to system path
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname("CogAlg"), '..')))
import numpy as np
from copy import deepcopy
from collections import deque
from frame_2D_alg.class_cluster import ClusterStructure, comp_param
from line_Ps import *
# from line_PPPs import line_PPPs_root

class Cpdert(ClusterStructure):
    # P param dert
    i = int  # input for range_comp only
    p = int  # accumulated in rng
    d = int  # accumulated in rng
    m = int  # distinct in deriv_comp only
    rdn = int  # summed param rdn
    sub_M = int  # match from comp_sublayers, if any
    sub_D = int  # diff from comp_sublayers, if any
    P = object  # P of i param
    roots = lambda: [[], []]  # [Ppm,Ppd]: Pps that pdert is in, to join rdert_s, or Rdert for rdert
    # for Rderts:
    rdert_ = list  # fixed-rng
    adert = object  # anchor dert
    olp_M = int  # dert.m accumulated in rng from all orders of overlap

class CPp(CP):

    dert_ = list  # already in CP? if not empty: Pp is primarily a merged P, other params are optional
    pdert_ = list  # Pp elements, "p" for param
    flay_rdn = bool  # Pp is layer-redundant to Pp.pdert_
    sublayers = list  # lambda: [([],[])]  # nested Ppm_ and Ppd_
    subDerts = list  # for comp sublayers
    levels = list  # levels of composition: Ps ) Pps ) Ppps..
    root = object  # higher Pp, to replace locals for merging
    rng = int
    # layer1: iL, iI, iD, iM, iRdn: summed P params

ave = 1  # ave dI -> mI, * coef / var type:
# no ave_mP: deviation computed via rM  # ave_mP = ave* n_comp_params: comp cost, or n vars per P: rep cost?
# all below should be coefs to ave?
ave_d = 2  # redefine as ave_coef
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
ave_mL = 0.5  # needs to be tuned
ave_mI = 1.2  # needs to be tuned, base ave_param?
ave_mD = 1.2  # needs to be tuned
ave_mM = 1.2  # needs to be tuned
ave_sub = 20  # for comp_sub_layers

ave_Dave = 100  # summed feedback filter
ave_dave = 20   # mean feedback filter

ave_dir_m = 10  # temorary, ave to evaluate negative match between anchor pdert.Ps

# used in search, form_Pp_root, comp_sublayers_, draw_PP_:
param_names = ["L_", "I_", "D_", "M_"]  # not really needed, we can just use indices?
aves = [ave_mL, ave_mI, ave_mD, ave_mM]
'''
    Conventions:
    postfix 't' denotes tuple, multiple ts is a nested tuple
    (or flat list if selective access from sublayers[0]?)
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    capitalized variables are normally summed small-case variables
'''

def line_PPs_root(P_t):  # P_T is P_t = [Pm_, Pd_];  higher-level input is nested to the depth = 1 + 2*elevation (level counter)

    norm_feedback(P_t)
    # 1st sublayer: implicit 3-layer nested P_tuple, P_ttt: (Pm_, Pd_, each:( Lmd, Imd, Dmd, Mmd, each: ( Ppm_, Ppd_)))
    # deep sublayers: implicit 2-layer nested tuples: Ppm_(Ppmm_), Ppd_(Ppdm_,Ppdd_)
    sublayer0 = []
    root = CPp(levels=[P_t], sublayers=[sublayer0])

    for fPd, P_ in enumerate(P_t):  # fPd: Pm_ or Pd_
        if len(P_) > 2:
            Pdert_t, dert1_, dert2_ = cross_comp(P_, fPd)  # Pdert_t: Ldert_, Idert_, Ddert_, Mdert_ (tuples of derivatives per P param)
            sum_rdn_(param_names, Pdert_t, fPd)  # sum cross-param redundancy per pdert, to evaluate for deeper processing
            for param_name, Pdert_ in zip(param_names, Pdert_t):  # Pdert_ -> Pps:
                for fPpd in 0, 1:  # 0-> Ppm_, 1-> Ppd_: more anti-correlated than Pp_s of different params
                    Pp_ = form_Pp_(deepcopy(Pdert_), fPpd)
                    sublayer0 += [Pp_]
                    if (fPpd and param_name == "D_") or (not fPpd and param_name == "I_"):
                        if not fPpd:
                            splice_Ps(Pp_, dert1_, dert2_, fPd, fPpd)  # splice eval by Pp.M in Ppm_, for Pms in +IPpms or Pds in +DPpm
                        rng_incr(root, Pp_, 1, 1)  # eval rng+ comp,form per Pp
                        der_incr(root, Pp_, 1)  # eval der+ comp,form per Pp
        else:
            sublayer0 += [[] for _ in range(8)]  # 8 empty [] to preserve index, 8 for each fPd

    root.levels.append(root.sublayers)  # to contain 1st and 2nd levels
    return root


def cross_comp(P_, fPd):  # cross-compare patterns within horizontal line

    Ldert_, Idert_, Ddert_, Mdert_, dert1_, dert2_ = [], [], [], [], [], []

    for _P, P, P2 in zip(P_, P_[1:], P_[2:] + [CP()]):  # for P_ cross-comp over step=1 and step=2
        _L, _I, _D, _M, *_ = _P.unpack()  # *_: skip remaining params
        L, I, D, M, *_ = P.unpack()
        D2, M2 = P2.D, P2.M

        Ldert_ += [comp_par(_P, _L, L, "L_", ave_mL)]  # div_comp L, sub_comp summed params:
        Idert_ += [comp_par(_P, _I, I, "I_", ave_mI)]
        if fPd:
            Ddert = comp_par(_P, _D, D2, "D_", ave_mD)  # step=2 for same-D-sign comp?
            Ddert_ += [Ddert]
            dert2_ += [Ddert.copy()]
            dert1_ += [comp_par(_P, _D, D, "D_", ave_mD)]  # to splice Pds
            Mdert_ += [comp_par(_P, _M, M, "M_", ave_mM)]
        else:
            Ddert_ += [comp_par(_P, _D, D, "D_", ave_mD)]
            Mdert = comp_par(_P, _M, M2, "M_", ave_mM)  # step=2 for same-M-sign comp?
            Mdert_ += [Mdert]
            dert2_ += [Mdert.copy()]
            dert1_ += [comp_par(_P, _M, M, "M_", ave_mM)]  # to splice Pms
        _L, _I, _D, _M = L, I, D, M

    if not fPd: Mdert_ = Mdert_[:-1]  # remove CP() filled in P2

    return (Ldert_, Idert_, Ddert_, Mdert_), dert1_, dert2_[:-1]  # remove CP() filled in P2


def comp_par(_P, _param, param, param_name, ave):

    if param_name == 'L_':  # special div_comp for L:
        d = param / _param  # higher order of scale, not accumulated: no search, rL is directional
        int_rL = int(max(d, 1 / d))
        frac_rL = max(d, 1 / d) - int_rL
        m = int_rL * min(param, _param) - (int_rL * frac_rL) / 2 - ave  # div_comp match is additive compression: +=min, not directional
    else:
        d = param - _param  # difference
        if param_name == 'I_': m = ave - abs(d)  # indirect match
        else: m = min(param, _param) - abs(d) / 2 - ave  # direct match

    return Cpdert(P=_P, i=_param, p=param + _param, d=d, m=m)


def form_Pp_(root_pdert_, fPd):
    # initialization:
    Pp_ = []
    x = 0
    _pdert = root_pdert_[0]
    if fPd: _sign = _pdert.d > 0
    else:   _sign = _pdert.m > 0
    # init Pp params:
    L=1; I=_pdert.p; D=_pdert.d; M=_pdert.m; Rdn=_pdert.rdn+_pdert.P.Rdn; x0=x; ix0=_pdert.P.x0; pdert_=[_pdert]

    for pdert in root_pdert_[1:]:  # segment by sign
        # proj = decay, or adjust by ave projected at distance=negL and contrast=negM, if significant:
        # m + ddist_ave = ave - ave * (ave_rM * (1 + negL / ((param.L + _param.L) / 2))) / (1 + negM / ave_negM)?
        if fPd: sign = pdert.d > 0
        else:   sign = pdert.m > 0

        if sign != _sign:  # sign change, pack terminated Pp, initialize new Pp
            term_Pp( Pp_, L, I, D, M, Rdn, x0, ix0, pdert_, fPd)
            # re-init Pp params:
            L=1; I=pdert.p; D=pdert.d; M=pdert.m; Rdn=pdert.rdn+pdert.P.Rdn; x0=x; ix0=pdert.P.x0; pdert_=[pdert]
        else:
            # accumulate params:
            L += 1; I += pdert.p; D += pdert.d; M += pdert.m; Rdn += pdert.rdn+pdert.P.Rdn; pdert_ += [pdert]
        _sign = sign; x += 1

    term_Pp( Pp_, L, I, D, M, Rdn, x0, ix0, pdert_, fPd)  # pack last Pp
    return Pp_

def term_Pp(Pp_, L, I, D, M, Rdn, x0, ix0, pdert_, fPd):

    if fPd: value = abs(D)
    else: value = M
    Pp_value = value / (L *.7)  # .7: ave intra-Pp-match coef, for value reduction with resolution, still non-negative
    pdert_V  = value - L * ave_M * (ave_D * fPd)  # cost incr per pdert representations
    flay_rdn = Pp_value < pdert_V
    # Pp vs Pdert_ rdn
    Pp = CPp(L=L, I=I, D=D, M=M, Rdn=Rdn+L+L*flay_rdn, x0=x0, ix0=ix0, flay_rdn=flay_rdn, rng=1, pdert_=pdert_)

    for pdert in Pp.pdert_: pdert.roots[fPd] = Pp  # root Pp refs

    Pp_.append(Pp)
    # no immediate normalization: Pp.I /= Pp.L; Pp.D /= Pp.L; Pp.M /= Pp.L; Pp.Rdn /= Pp.L


def sum_rdn_(param_names, Pdert_t, fPd):
    '''
    access same-index pderts of all P params, assign redundancy to lesser-magnitude m|d in param pair.
    if other-param same-P_-index pdert is missing, rdn doesn't change.
    To include layer_rdn: initialize each rdn at 1 vs. 0, then increment here and in intra_Pp_?
    '''
    if fPd: alt = 'M'
    else:   alt = 'D'
    name_pairs = (('I', 'L'), ('I', 'D'), ('I', 'M'), ('L', alt), ('D', 'M'))  # pairs of params redundant to each other
    # rdn_t = [[], [], [], []] is replaced with pdert.rdn

    for i, (Ldert, Idert, Ddert, Mdert) in enumerate( zip_longest(Pdert_t[0], Pdert_t[1], Pdert_t[2], Pdert_t[3], fillvalue=Cpdert())):
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

        for j, param_name in enumerate(param_names):  # sum param rdn from all pairs it is in, flatten pair_names, pair_rdns?
            Rdn = 0
            for name_in_pair, rdn in zip(name_pairs, rdn_pairs):
                if param_name[0] == name_in_pair[0]:  # param_name = "L_", param_name[0] = "L"
                    Rdn += rdn[0]
                elif param_name[0] == name_in_pair[1]:
                    Rdn += rdn[1]

            if len(Pdert_t[j]) >i:  # if fPd: Ddert_ is step=2, else: Mdert_ is step=2
                Pdert_t[j][i].rdn = Rdn  # [Ldert_, Idert_, Ddert_, Mdert_]

    # no need to return since rdn is updated in each pdert

def splice_Ps(Ppm_, pdert1_, pdert2_, fPd, fPpd):  # re-eval Pps, Pp.pdert_s for redundancy, eval splice Ps
    '''
    Initial P termination is by pixel-level sign change, but resulting separation may not be significant on a pattern level.
    That is, separating opposite-sign patterns are weak relative to separated same-sign patterns, especially if similar.
     '''
    for i, Pp in enumerate(Ppm_):
        if fPpd: value = abs(Pp.D)  # DPpm_ if fPd, else IPpm_
        else: value = Pp.M  # add summed P.M|D?

        if value > ave_M * (ave_D*fPpd) * Pp.Rdn * 4 and Pp.L > 4:  # min internal xP.I|D match in +Ppm
            M2 = M1 = 0
            for pdert2 in pdert2_: M2 += pdert2.m  # match(I, __I or D, __D): step=2
            for pdert1 in pdert1_: M1 += pdert1.m  # match(I, _I or D, _D): step=1

            if M2 / max( abs(M1), 1) > ave_splice:  # similarity / separation(!/0): splice Ps in Pp, also implies weak Pp.pdert_?
                # replace Pp params with summed P params, Pp is now primarily a spliced P:
                Pp.L = sum([pdert.P.L for pdert in Pp.pdert_])
                Pp.I = sum([pdert.P.I for pdert in Pp.pdert_])
                Pp.D = sum([pdert.P.D for pdert in Pp.pdert_])
                Pp.M = sum([pdert.P.M for pdert in Pp.pdert_])
                Pp.Rdn = sum([pdert.P.Rdn for pdert in Pp.pdert_])

                for pdert in Pp.pdert_:
                    Pp.dert_ += pdert.P.dert_  # if Pp.dert_: spliced P, summed P params are primary, other Pp params are low-value
                intra_P(Pp, rdn=1, rng=1, fPd=fPd)  # fPd, not fPpd, re-eval line_Ps' intra_P per spliced P
        '''
        no splice(): fine-grained eval per P triplet is too expensive?
        '''

def intra_P(P, rdn, rng, fPd):  # this is a rerun of line_Ps
    comb_sublayers = []
    if not fPd:
        if P.M - P.Rdn * ave_M * P.L > ave_M * rdn and P.L > 2:  # M value adjusted for xP and higher-layers redundancy
            rdn+=1; rng+=1
            P.subset = rdn, rng, [],[],[],[]  # 1st sublayer, []s: xsub_pmdertt_, _xsub_pddertt_, sub_Ppm_, sub_Ppd_
            sub_Pm_, sub_Pd_ = [], []  # initialize layers top-down, concatenate by intra_P_ in form_P_
            P.sublayers = [(sub_Pm_, sub_Pd_)]
            rdert_ = range_comp(P.dert_)  # rng+, skip predictable next dert, local ave? rdn to higher (or stronger?) layers
            sub_Pm_[:] = form_P_(P, rdert_, rdn, rng, fPd=False)  # cluster by rm sign
            sub_Pd_[:] = form_P_(P, rdert_, rdn, rng, fPd=True)  # cluster by rd sign
        # else: Pp.sublayers = []  # reset after the above converts it to [([],[])]?
    else:  # P is Pd
        if abs(P.D) - (P.L - P.Rdn) * ave_D * P.L > ave_D * rdn and P.L > 1:  # high-D span, level rdn, vs. param rdn in dert
            rdn+=1; rng+=1
            P.subset = rdn, rng, [],[],[],[]  # 1st sublayer, []s: xsub_pmdertt_, _xsub_pddertt_, sub_Ppm_, sub_Ppd_
            sub_Pm_, sub_Pd_ = [], []
            P.sublayers = [(sub_Pm_, sub_Pd_)]
            ddert_ = deriv_comp(P.dert_)  # i is d
            sub_Pm_[:] = form_P_(P, ddert_, rdn, rng, fPd=False)  # cluster by mm sign
            sub_Pd_[:] = form_P_(P, ddert_, rdn, rng, fPd=True)  # cluster by md sign

    if P.sublayers:
        new_comb_sublayers = []
        for (comb_sub_Pm_, comb_sub_Pd_), (sub_Pm_, sub_Pd_) in zip_longest(comb_sublayers, P.sublayers, fillvalue=([],[])):
            comb_sub_Pm_ += sub_Pm_  # remove brackets, they preserve index in sub_Pp root_
            comb_sub_Pd_ += sub_Pd_
            new_comb_sublayers.append((comb_sub_Pm_, comb_sub_Pd_))  # add sublayer
        comb_sublayers = new_comb_sublayers

    P.sublayers += comb_sublayers  # no return


def der_incr(rootPp, Pp_, hlayers):  # evaluate each Pp for incremental derivation, as in line_patterns der_comp but with higher local ave,
    # pack results into sub_Pm_
    comb_sublayers = []  # combine into root P sublayers[1:], each nested to depth = sublayers[n]

    for i, Pp in enumerate(Pp_):
        if Pp.L > 1:
            loc_ave_M = ave_M * Pp.Rdn * hlayers * ((Pp.M / ave_M) / 2) * ave_D  # ave_D=ave_d?
            iM = sum( [pdert.P.M for pdert in Pp.pdert_])
            loc_ave = (ave + iM) / 2 * ave_d * Pp.Rdn * hlayers  # cost per cross-comp

            if abs(Pp.D) / Pp.L > loc_ave_M * 4:  # 4: search cost, + Pp.M: local borrow potential?
                sub_search(Pp, fPd=True)  # search in top sublayer, eval by pdert.d
                sub_Ppm_, sub_Ppd_ = [], []
                Pp.sublayers = [(sub_Ppm_, sub_Ppd_)]
                ddert_ = []
                for _pdert, pdert in zip( Pp.pdert_[:-1], Pp.pdert_[1:]):  # or comp abs d, or step=2 for sign match?
                    ddert_ += [comp_par(_pdert.P, _pdert.d, pdert.d, "D_", loc_ave * ave_mD)]  # cross-comp of ds
                sub_Ppm_[:] = form_Pp_(ddert_, fPd=False)
                sub_Ppd_[:] = form_Pp_(ddert_, fPd=True)
                if sub_Ppd_ and abs(Pp.D) + Pp.M > loc_ave_M * 4:  # looping search cost, diff val per Pd_'DPpd_, +Pp.iD?
                    der_incr(Pp, sub_Ppd_, hlayers+1)  # recursive der+, no need for Pdert_, no rng+: Pms are redundant?
                else:
                    Pp.sublayers = []  # reset after the above converts it to [([],[])]

        if Pp.sublayers:  # pack added sublayers, same as in rng_incr:
            new_comb_sublayers = []
            for (comb_sub_Ppm_, comb_sub_Ppd_), (sub_Ppm_, sub_Ppd_) in zip_longest(comb_sublayers, Pp.sublayers, fillvalue=([], [])):
                comb_sub_Ppm_ += sub_Ppm_  # use brackets for nested P_ttt
                comb_sub_Ppd_ += sub_Ppd_
                new_comb_sublayers.append((comb_sub_Ppm_, comb_sub_Ppd_))  # add sublayer
            comb_sublayers = new_comb_sublayers

    if rootPp and comb_sublayers: rootPp.sublayers += comb_sublayers  # new sublayers


def rng_incr(rootPp, Pp_, hlayers, rng):  # evaluate each Pp for incremental range, fixed: parallelizable, rng-selection gain < costs
    # pack results into sub_Pm_
    comb_sublayers = []  # combine into root P sublayers[1:], each nested to depth = sublayers[n]

    for i, Pp in enumerate(Pp_):
        if Pp.L > 1:
            loc_ave_M = ((Pp.M + ave_M) / 2) * Pp.Rdn * hlayers  # cost per loop = (ave of global and local aves) * redundancy
            iM = sum( [pdert.P.M for pdert in Pp.pdert_])
            loc_ave = (ave + iM) / 2 * Pp.Rdn * hlayers  # cost per cross-comp

            if Pp.M / Pp.L > loc_ave_M + 4:  # 4: search cost, + Pp.iM?
                sub_search(Pp, fPd=False)
                sub_Ppm_, sub_Ppd_ = [], []
                Pp.sublayers = [(sub_Ppm_, sub_Ppd_)]
                Rdert_ = comp_rng(Pp, loc_ave * ave_mI, rng)  # accumulates Rderts over fixed-rng rdert_
                rng+=1  # init=1 in form_Pp_
                sub_Ppm_[:] = form_rPp_(Rdert_, rng, depth=1)  # draft
                if sub_Ppm_ and Pp.M > loc_ave_M * 4 and not Pp.dert_:  # 4: looping cost, if Pm_'IPpm_.M, +Pp.iM?
                    rng_incr(Pp, sub_Ppm_, hlayers+1, rng)  # recursive rng+, no der+ in redundant Pds?
                else:
                    Pp.sublayers = []  # reset after the above converts it to [([],[])]

        if Pp.sublayers:  # pack added sublayers, same as in der_incr:
            new_comb_sublayers = []
            for (comb_sub_Ppm_, comb_sub_Ppd_), (sub_Ppm_, sub_Ppd_) in zip_longest(comb_sublayers, Pp.sublayers, fillvalue=([],[])):
                comb_sub_Ppm_ += sub_Ppm_  # use brackets for nested P_ttt
                comb_sub_Ppd_ += sub_Ppd_
                new_comb_sublayers.append((comb_sub_Ppm_, comb_sub_Ppd_))  # add sublayer
            comb_sublayers = new_comb_sublayers

    if rootPp and comb_sublayers: rootPp.sublayers += comb_sublayers  # new sublayers
    # no return, Pp_ is changed in-place

def comp_rng(rootPp, loc_ave, rng):  # extended fixed-rng search-right for core I at local ave: lower m

    if rng==1:  # 1st call, initialize Rdert_ with aderts:
        Rdert_ = [Cpdert(adert=pdert) for pdert in rootPp.pdert_]
    else:  # rderts are left and right from adert, evaluation per search:
        Rdert_ = rootPp.pdert_.copy()  # copy to avoid overwriting pdert.roots:

    for Rdert in enumerate(Rdert_): Rdert.roots = []  # reset

    for i, _Rdert in enumerate(Rdert_[:-rng+1]):  # extend consecutive fixed-rng Rderts, overlapping within rng-1
        adert = _Rdert.anchor
        # cross-comp at current rng:
        Rdert = Rdert_[i+rng]  # Rdert in which cdert is an anchor
        rdert = Cpdert(P=adert.P, roots=Rdert)  # rng dert
        cdert = Rdert.anchor  # compared-P dert
        rdert.p = cdert.i + adert.i  # -> ave i
        rdert.d = cdert.i - adert.i  # difference
        rdert.m = loc_ave - abs(adert.d)  # indirect match
        if rdert.m > ave_M * 4 and rdert.P.sublayers and cdert.P.sublayers:  # 4: init ave_sub coef
            comp_sublayers(rdert.P, cdert.P, rdert.m)  # deeper cross-comp between high-m sub_Ps, separate from ndert.m
        # left Rdert assign:
        _Rdert.accum_from(rdert, ignore_capital=True)  # Pp params += pdert params
        _Rdert.rdert_ += [rdert]  # extend _Rdert to the right
        # right Rdert assign:
        Rdert.accum_from(rdert, ignore_capital=True)  # Pp params += pdert params
        Rdert.rdert_.insert(0, rdert.copy())  # extend Rdert to the left
        Rdert.rdert_[0].roots = _Rdert

    return Rdert_


def form_rPp_(Rdert_, rng, depth):  # evaluate direct and mediated match between adert.P and olp_dert.Ps
    '''
    Each rng+ adds a layer of pdert_ mediation to form extended graphs of Rderts,
    including Rdert_ pdert.Rderts with peri-negative direct match but positive mediated match into rPp.
    It also adds mediated match to positive pderts for accuracy. But there will be fewer overlapping pderts with mediation?
    '''
    # not revised yet:
    rPp_ = []
    for _Rdert in Rdert_:
        if not isinstance(_Rdert.roots, CPp):  # if no rPp is formed from prior merging
            _rPp = CPp(pdert_=[_Rdert])
            _Rdert.roots = _rPp
            rPp_.append(_rPp)

            olp_M = 0
            for olp_dert in _Rdert.rdert_:  # breadth-first evaluation:
                rel_m = olp_dert.m / max(1, olp_dert.i)  # ratio of adert m to olp_dert mag
                olp_M += olp_dert.m + olp_dert.roots.m * rel_m  # match to olp pdert.Ps is estimated from direct m ratio

            if olp_M > ave_M * 4:  # total M of adert to olp_dert_ is a precondition for rPp extension
                for olp_dert in _Rdert.rdert_:
                    Rdert = olp_dert.roots
                    olp_dert.olp_M += Rdert.rdert_[0].m  # _Rdert olp added per rng+, always 2 rderts per Rdert
                    Rdert.rdert_[0].olp_M += olp_dert.m  # reciprocal left olp extension

                    if olp_dert.m + olp_dert.olp_M > ave_dir_m:  # ave_dir_m < ave, negative here,
                        # eval by match of anchor + mediated Rdert overlap
                        eval_merge_rPp(_rPp, Rdert)

                        # test to include oolp_Rdert in rPp, also higher-order overlaps
                        oolp_Rdert = Rdert.rdert_[0].roots
                        # this _rPp is actually rPp after the merging above
                        if oolp_Rdert not in _rPp.pdert_ and oolp_Rdert.m + olp_dert.olp_M > ave_dir_m:
                            eval_merge_rPp(_rPp, oolp_Rdert)
    return rPp_

def eval_merge_rPp(_rPp, Rdert):  # merge overlapping rPp in _rPp

    rPp = Rdert.roots
    if isinstance(rPp, CPp):  # Rdert has rPp, merge it
        for Rdert in rPp.pdert_:
            if Rdert not in _rPp.pdert_:
                _rPp.accum_from(Rdert.roots, excluded=['x0'])
                _rPp.pdert_.append(Rdert)
                Rdert.roots = _rPp  # update reference of rdert' Rdert
    else:  # no rPp
        _rPp.accum_from(Rdert)
        _rPp.pdert_.append(Rdert)  # pdert_ is Rderts
        Rdert.roots = _rPp

'''   
    We know that "other" pderts in rPps are the same within overlap, but "anchor" pderts are different, because overlap is between different rPps. 
    So, the difference between same-other-P pdert.m s within overlap should be proportional to relative m between anchor (other) pdert Ps
                    
    local proximity sub-clustering, before and after merge, 
    merge by overlapping M, vs. cluster by rng m in cross comp: 
    second-order cross-central vs. primary center-edge similarity?
 '''
# same level: same comparison, but not adjacent, that was at rng-? still a priority?:

def reform_rPp_(rPp_, root, rng):  # cluster rng-overlapping directional rPps by M sign
    re_rPp_ = []  # output clusters
    distance = 1

    for rPp in rPp_:
        if rPp.M > 0:
            if "re_rPp" in locals():
                # additions and exclusions, exclude overlap? or individual vars accum and init is clearer?
                re_rPp.accum_from(rPp, ignore_capital=True)  # both Rdert and any of Rdert_[-rng:-1] are positive
                re_rPp.L += rPp.L ; re_rPp.pdert_ += rPp.pdert_  # extend Rderts
            else:
                re_rPp = CPp(pdert_=rPp.pdert_, root=root)
                re_rPp.L = rPp.L; re_rPp.accum_from(re_rPp)
            distance = 1
        else:
            if "re_rPp" in locals() and distance==rng:
                term_re_rPp(re_rPp, re_rPp_)  # rPp.pdert_ is Rdert_
                del re_rPp  # exceeded comp rng, remove from locals
            distance += 1  # from next pre_rPp (add distance after checking distance == rng)
    if "re_rPp" in locals():  # terminate last rPp
        term_re_rPp(re_rPp, re_rPp_)

    return rPp_


def term_re_rPp(rPp, rPp_):  # Pp_, L, I, D, M, Rdn, x0, ix0, rPp_):
    # add conditional cross-comp between pre_rPp_s?

    Pp_M = rPp.M / (rPp.L *.7)  # .7: ave intra-Pp-match coef, for value reduction with resolution, still non-negative
    rPp_M  = rPp.M - rPp.L * ave_M  # cost incr per pdert representations
    rPp.flay_rdn = Pp_M < rPp_M  # Pp vs rPp_ rdn
    # rPp = CPp(L=L, I=I, D=D, M=M, Rdn=Rdn+L+L*flay_rdn, x0=x0, ix0=ix0, flay_rdn=flay_rdn, pdert_=Rdert_)
    for Rdert in rPp.pdert_:  # rPp.pdert_ is Rdert_
        Rdert.roots = rPp  # Rdert is CPp, so we use root
    rPp_.append(rPp)
    # no immediate normalization: Pp.I /= Pp.L; Pp.D /= Pp.L; Pp.M /= Pp.L; Pp.Rdn /= Pp.L


def sub_search(rootPp, fPd):  # ~line_PPs_root: cross-comp sub_Ps in top sublayer of high-M Pms | high-D Pds, called from intra_Pp_

    for pdert in rootPp.pdert_:  # vs. Idert_
        P = pdert.P
        if P.sublayers:  # not empty sublayer
            subset = P.sublayers[0]
            for fsubPd, sub_P_ in enumerate([subset[0], subset[1]]):  # sub_Pm_, sub_Pd_
                if len(sub_P_) > 2 and ((fPd and abs(P.D) > ave_D * rootPp.Rdn) or (P.M > ave_M * rootPp.Rdn)):
                    # + pdert.m + sublayer.Dert.M?
                    sub_Pdert_t, dert1_, dert2_ = cross_comp(sub_P_, fsubPd)  # Pdert_t: Ldert_, Idert_, Ddert_, Mdert_
                    sum_rdn_(param_names, sub_Pdert_t, fsubPd)
                    paramset = []
                    # Pdert_-> Pp_:
                    for param_name, sub_Pdert_ in zip(param_names, sub_Pdert_t):
                        param_md = []
                        for fPpd in 0, 1:
                            sub_Pp_ = form_Pp_(sub_Pdert_, fPpd)
                            param_md.append(sub_Pp_)
                            if (fPpd and param_name == "D_") or (not fPpd and param_name == "I_"):
                                if not fPpd:
                                    splice_Ps(sub_Pp_, dert1_, dert2_, fPd, fPpd)  # splice eval by Pp.M in Ppm_, for Pms in +IPpms or Pds in +DPpm
                                rootPp = CPp(pdert_=rootPp.pdert_, sublayers=[param_md])
                                intra_Pp_(rootPp, param_md[fPpd], sub_Pdert_, 1, fPpd)  # der+ or rng+
                        paramset.append(param_md)
                    P.subset[4 + fsubPd].append(paramset)  # sub_Ppm_tt and sub_Ppd_tt
                    # deeper comp_sublayers is selective per sub_P


def comp_sublayers(_P, P, root_v):  # if pdert.m -> if summed params m -> if positional m: mx0?

    # replace root_v with root_vt: separate evaluations?
    sub_M, sub_D = 0, 0
    _rdn, _rng = _P.subset[:2]
    rdn, rng = P.subset[:2]
    xDert_vt = form_comp_Derts(_P, P, root_v)  # value tuple: separate vm, vd for comp sub_mP_ or sub_dP_

    # comp sub_Ps between 1st sublayers of _P and P:
    for fPd, xDert_v in enumerate(xDert_vt):  # eval m, d:
        if fPd: sub_D += xDert_v
        else: sub_M += xDert_v
        if xDert_v + root_v > ave_M * (fPd * ave_D) * rdn:
            _sub_P_ = _P.sublayers[0][fPd]
            sub_P_ = P.sublayers[0][fPd]
            _xsub_pdertt_ = _P.subset[2+fPd]  # array of cross-sub_P pdert tuples
            xsub_pdertt_ = P.subset[2+fPd]
            if rng == _rng and min(_P.L, P.L) > ave_Ls:  # if same fork: comp sub_Ps to each _sub_P in max relative distance per comb_V:
                _SL = SL = 0  # summed Ls
                start_index = next_index = 0  # index of starting sub_P for current _sub_P
                for _sub_P in _sub_P_:
                    _xsub_pdertt_ += [[]]  # append xsub_dertt per _sub_P, sparse, inner brackets for subset_?
                    xsub_pdertt_ += [[]]
                    _xsub_pdertt = [[], [], [], []]  # L, I, D, M xsub_pdert_s
                    _SL += _sub_P.L  # ix0 of next _sub_P;  no xsub_pdertt_ per sub_P_, summed only?
                    # search right:
                    for sub_P in sub_P_[start_index:]:  # index_ix0 > _ix0, comp sub_Ps at proximate relative positions in sub_P_
                        fbreak, xsub_P_M, xsub_P_D = comp_sub_P(_sub_P, sub_P, _xsub_pdertt, root_v, fPd)
                        sub_M += xsub_P_M ; sub_D += xsub_P_D
                        if fbreak:
                            break
                        # if next ix overlap: ix0 of next _sub_P < ix0 of current sub_P
                        if SL < _SL: next_index += 1
                        SL += sub_P.L  # ix0 of next sub_P
                    # search left:
                    for sub_P in reversed(sub_P_[len(sub_P_) - start_index:]):  # index_ix0 <= _ix0
                        fbreak, xsub_P_M, xsub_P_D = comp_sub_P(_sub_P, sub_P, _xsub_pdertt, root_v, fPd)  # invert sub_P, _sub_P positions
                        sub_M += xsub_P_M ; sub_D += xsub_P_D
                        if fbreak:
                            break
                    # not implemented: if param_name == "I_" and not fPd: sub_pdert = search_param_(param_)
                    start_index = next_index  # for next _sub_P

                    if any(_xsub_pdertt):  # at least 1 sub_pdert, real min length ~ 8, very unlikely
                        xsub_Pp_t = []  # LPpm_, IPpm_, DPpm_, MPpm_
                        sum_rdn_(param_names, _xsub_pdertt, fPd)  # no return from sum_rdn now
                        for param_name, xsub_Pdert_ in zip(param_names, _xsub_pdertt):
                            xsub_Pp_ = form_Pp_(xsub_Pdert_, fPd=0)
                            # no step=2 for splice: xsub_pdertts are not consecutive, and their Ps are not aligned?
                            if param_name == "I_":
                                # no root: Pd_ is always empty as comp_sublayers can be called from I param only:
                                intra_Pp_(None, xsub_Pp_, xsub_Pdert_, 1, fPd=0)  # rng+ only?
                            xsub_Pp_t.append(xsub_Pp_)

                        _xsub_pdertt_[-1][:] = xsub_Pp_t
                        xsub_pdertt_[-1][:] = xsub_Pp_t  # bilateral assignment?

    return sub_M, sub_D


def comp_sub_P(_sub_P, sub_P, xsub_pdertt, root_v, fPd):
    fbreak = 0
    xsub_P_M, xsub_P_D = 0,0  # vm,vd combined across params
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
            # if same-sign only?
            sub_pdert = comp_par(sub_P, _param, param, param_name, ave)
            # no negative-value sum: -Ps won't be processed:
            if sub_pdert.m > 0: xsub_P_M += sub_pdert.m
            vd = abs(sub_pdert.d) - ave_d  # convert to coef
            if vd > 0: xsub_P_D += vd
            xsub_pdertt[i].append(sub_pdert)  # per L, I, D, M' xsub_pdert

        V += xsub_P_M + xsub_P_D  # separate eval for form_Pm_ and form_Pd_?

        if V > ave_M * 4 and _sub_P.sublayers and sub_P.sublayers:  # 4: ave_comp_sublayers coef
            # update sub_M in Ipdert(xsub_pdertt[1]) only?
            sub_M, sub_D = comp_sublayers(_sub_P, sub_P, V)  # recursion for deeper layers
            xsub_P_M += sub_M; xsub_P_D += sub_M
    else:
        fbreak = 1  # only sub_Ps with relatively proximate position in sub_P_|_sub_P_ are compared

    return fbreak, xsub_P_M, xsub_P_D


def form_comp_Derts(_P, P, root_v):

    subDertt_t = [[],[]]  # two comparands, each preserved for future processing
    # form Derts:
    for _sublayer, sublayer in zip(_P.sublayers, P.sublayers):
        for i, isublayer in enumerate([_sublayer, sublayer]):  # list of subsets: index 0 = _P.sublayers, index 1 = P.sublayers
            nsub = len(isublayer[0]) + len(isublayer[1])
            if root_v * nsub > ave_M * 5:  # ave_Dert, separate eval for m and d?
                subDertt = [[0, 0, 0, 0], [0, 0, 0, 0]]  # Pm,Pd' L,I,D,M summed in sublayer
                for j, sub_P_ in enumerate(isublayer):  # j: Pm | Pd
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
        for fPd, (_iDert, iDert) in enumerate( zip((_mDert, _dDert), (mDert, dDert))):
            derDert = []
            for _param, param, param_name, ave in zip(_iDert, iDert, param_names, aves):
                dert = comp_par(_P, _param, param, param_name, ave)
                if fPd:
                    vd = abs(dert.d) - ave_d  # higher value for L?
                    if vd > 0: xDert_vt[fPd] += vd  # no neg value sum: -Ps won't be processed
                elif dert.m > 0:
                    xDert_vt[fPd] += dert.m
                derDert.append(dert)  # dert per param, derDert_ per _subDert, copy in subDert?
            derDertt[fPd].append(derDert)  # m, d derDerts per sublayer

        _P.derDertt_.append(derDertt)  # derDert_ per _P, P pair
        P.derDertt_.append(derDertt)  # bilateral assignment?

    return xDert_vt
'''
xsub_pderts (cross-sub_P): xsub_pdertt_[ xsub_pdertt [ xsub_pdert_[ sub_pdert]]]: each bracket is a level of nesting
the above forms Pps across xsub_dertt, then also form Pps across _xsub_pdertt_ and xsub_pdertt_? 
'''

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