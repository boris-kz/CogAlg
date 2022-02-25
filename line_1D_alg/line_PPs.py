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

class Cderp(ClusterStructure):  # dert per P param

    i = int  # input for range_comp only
    p = int  # accumulated in rng
    d = int  # accumulated in rng
    m = int  # distinct in deriv_comp only
    rdn = int  # summed param rdn
    sub_M = int  # match from comp_sublayers, if any
    sub_D = int  # diff from comp_sublayers, if any
    P = object  # P of i param
    roots = lambda: [[], []]  # [Ppm,Ppd]: Pps that derp is in, to join rderp_s, or Rderp for rderp
    # in Rderts:
    rderp_ = list  # fixed rng of comparands
    aderp = object  # anchor derp

class CPp(CP):  # "p" stands for param

    P = object  # if not empty: Pp is primarily a spliced P, all params below are optional
    derp_ = list  # Pp elements, replaced by single P if spliced?
    flay_rdn = bool  # Pp is layer-redundant to Pp.derp_
    sublayers = list  # lambda: [([],[])]  # nested Ppm_ and Ppd_
    subDerts = list  # for comp sublayers
    levels = list  # levels of composition: Ps ) Pps ) Ppps..
    root = object  # higher Pp, to replace locals for merging
    rng = int  # range of P.I cross-comparison
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

ave_dir_m = 10  # temorary, ave to evaluate negative match between anchor derp.Ps

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
    sublayer0 = []
    root = CPp(levels=[P_t], sublayers=[sublayer0])
    '''
    1st sublayer: implicit 3-layer nested P_tuple, P_ttt: (Pm_, Pd_, each:( Lmd, Imd, Dmd, Mmd, each: ( Ppm_, Ppd_)))
    deep sublayers: implicit 2-layer nested tuples: Ppm_(Ppmm_), Ppd_(Ppdm_,Ppdd_)
    '''
    for fPd, P_ in enumerate(P_t):  # fPd: Pm_ or Pd_
        if len(P_) > 2:
            derp_t, derp1_, derp2_ = cross_comp(P_, fPd)  # derp_t: Lderp_, Iderp_, Dderp_, Mderp_ (tuples of derivatives per P param)
            sum_rdn_(param_names, derp_t, fPd)  # sum cross-param redundancy per derp, to evaluate for deeper processing
            for param_name, derp_ in zip(param_names, derp_t):  # derp_ -> Pps:
                for fPpd in 0, 1:  # 0-> Ppm_, 1-> Ppd_: more anti-correlated than Pp_s of different params
                    Pp_ = form_Pp_(deepcopy(derp_), fPpd)
                    sublayer0 += [Pp_]
                    if (fPpd and param_name == "D_") or (not fPpd and param_name == "I_"):
                        if not fPpd:
                            splice_Ps(Pp_, derp1_, derp2_, fPd, fPpd)  # splice eval by Pp.M in Ppm_, for Pms in +IPpms or Pds in +DPpm
                        range_incr(root, Pp_, hlayers=1, rng=2)  # eval rng+ comp,form per Pp, parallel to:
                        deriv_incr(root, Pp_, hlayers=1)  # eval der+ comp,form per Pp
        else:
            sublayer0 += [[] for _ in range(8)]  # 8 empty [] to preserve index, 8 for each fPd

    root.levels.append(root.sublayers)  # to contain 1st and 2nd levels
    return root


def cross_comp(P_, fPd):  # cross-compare patterns within horizontal line

    Lderp_, Iderp_, Dderp_, Mderp_, derp1_, derp2_ = [], [], [], [], [], []

    for _P, P, P2 in zip(P_, P_[1:], P_[2:] + [CP()]):  # for P_ cross-comp over step=1 and step=2
        _L, _I, _D, _M, *_ = _P.unpack()  # *_: skip remaining params
        L, I, D, M, *_ = P.unpack()
        D2, M2 = P2.D, P2.M

        Lderp_ += [comp_par(_P, _L, L, "L_", ave_mL)]  # div_comp L, sub_comp summed params:
        Iderp_ += [comp_par(_P, _I, I, "I_", ave_mI)]
        if fPd:
            Dderp = comp_par(_P, _D, D2, "D_", ave_mD)  # step=2 for same-D-sign comp?
            Dderp_ += [Dderp]
            derp2_ += [Dderp.copy()]
            derp1_ += [comp_par(_P, _D, D, "D_", ave_mD)]  # to splice Pds
            Mderp_ += [comp_par(_P, _M, M, "M_", ave_mM)]
        else:
            Dderp_ += [comp_par(_P, _D, D, "D_", ave_mD)]
            Mderp = comp_par(_P, _M, M2, "M_", ave_mM)  # step=2 for same-M-sign comp?
            Mderp_ += [Mderp]
            derp2_ += [Mderp.copy()]
            derp1_ += [comp_par(_P, _M, M, "M_", ave_mM)]  # to splice Pms
        _L, _I, _D, _M = L, I, D, M

    if not fPd: Mderp_ = Mderp_[:-1]  # remove CP() filled in P2

    return (Lderp_, Iderp_, Dderp_, Mderp_), derp1_, derp2_[:-1]  # remove CP() filled in P2


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

    return Cderp(P=_P, i=_param, p=param + _param, d=d, m=m)


def form_Pp_(root_derp_, fPd):
    # initialization:
    Pp_ = []
    x = 0
    _derp = root_derp_[0]
    if fPd: _sign = _derp.d > 0
    else:   _sign = _derp.m > 0
    # init Pp params:
    L=1; I=_derp.p; D=_derp.d; M=_derp.m; Rdn=_derp.rdn+_derp.P.Rdn; x0=x; ix0=_derp.P.x0; derp_=[_derp]

    for derp in root_derp_[1:]:  # segment by sign
        # proj = decay, or adjust by ave projected at distance=negL and contrast=negM, if significant:
        # m + ddist_ave = ave - ave * (ave_rM * (1 + negL / ((param.L + _param.L) / 2))) / (1 + negM / ave_negM)?
        if fPd: sign = derp.d > 0
        else:   sign = derp.m > 0

        if sign != _sign:  # sign change, pack terminated Pp, initialize new Pp
            term_Pp( Pp_, L, I, D, M, Rdn, x0, ix0, derp_, fPd)
            # re-init Pp params:
            L=1; I=derp.p; D=derp.d; M=derp.m; Rdn=derp.rdn+derp.P.Rdn; x0=x; ix0=derp.P.x0; derp_=[derp]
        else:
            # accumulate params:
            L += 1; I += derp.p; D += derp.d; M += derp.m; Rdn += derp.rdn+derp.P.Rdn; derp_ += [derp]
        _sign = sign; x += 1

    term_Pp( Pp_, L, I, D, M, Rdn, x0, ix0, derp_, fPd)  # pack last Pp
    return Pp_

def term_Pp(Pp_, L, I, D, M, Rdn, x0, ix0, derp_, fPd):

    if fPd: value = abs(D)
    else: value = M
    Pp_value = value / (L *.7)  # .7: ave intra-Pp-match coef, for value reduction with resolution, still non-negative
    derp_V  = value - L * ave_M * (ave_D * fPd)  # cost incr per derp representations
    flay_rdn = Pp_value < derp_V
    # Pp vs derp_ rdn
    Pp = CPp(L=L, I=I, D=D, M=M, Rdn=Rdn+L+L*flay_rdn, x0=x0, ix0=ix0, flay_rdn=flay_rdn, rng=1, derp_=derp_)

    for derp in Pp.derp_: derp.roots[fPd] = Pp  # root Pp refs

    Pp_.append(Pp)
    # no immediate normalization: Pp.I /= Pp.L; Pp.D /= Pp.L; Pp.M /= Pp.L; Pp.Rdn /= Pp.L


def sum_rdn_(param_names, derp_t, fPd):
    '''
    access same-index derps of all P params, assign redundancy to lesser-magnitude m|d in param pair.
    if other-param same-P_-index derp is missing, rdn doesn't change.

    This is current-level Rdn, summed in resulting P, added to sum of lower-derivation Rdn of element Ps?
    To include layer_rdn: initialize each rdn at 1 vs. 0, then increment here and in intra_Pp_?
    '''
    if fPd: alt = 'M'
    else:   alt = 'D'
    name_pairs = (('I', 'L'), ('I', 'D'), ('I', 'M'), ('L', alt), ('D', 'M'))  # pairs of params redundant to each other
    # rdn_t = [[], [], [], []] is replaced with derp.rdn

    for i, (Lderp, Iderp, Dderp, Mderp) in enumerate( zip_longest(derp_t[0], derp_t[1], derp_t[2], derp_t[3], fillvalue=Cderp())):
        # derp per _P in P_, 0: Lderp_, 1: Iderp_, 2: Dderp_, 3: Mderp_
        # P M|D rdn + derp m|d rdn:
        rdn_pairs = [[fPd, 0], [fPd, 1-fPd], [fPd, fPd], [0, 1], [1-fPd, fPd]]  # rdn in olp Ps: if fPd: I, M rdn+=1, else: D rdn+=1
        # names:    ('I','L'), ('I','D'),    ('I','M'),  ('L',alt), ('D','M'))  # I.m + P.M: value is combined across P levels?

        for rdn_pair, name_pair in zip(rdn_pairs, name_pairs):
            # assign rdn in each rdn_pair using partial name substitution: https://www.w3schools.com/python/ref_func_eval.asp
            if fPd:
                if eval("abs(" + name_pair[0] + "derp.d) > abs(" + name_pair[1] + "derp.d)"):  # (param_name)dert.d|m
                    rdn_pair[1] += 1
                else: rdn_pair[0] += 1  # weaker pair rdn+1
            else:
                if eval(name_pair[0] + "derp.m > " + name_pair[1] + "derp.m"):
                    rdn_pair[1] += 1
                else: rdn_pair[0] += 1  # weaker pair rdn+1

        for j, param_name in enumerate(param_names):  # sum param rdn from all pairs it is in, flatten pair_names, pair_rdns?
            Rdn = 0
            for name_in_pair, rdn in zip(name_pairs, rdn_pairs):
                if param_name[0] == name_in_pair[0]:  # param_name = "L_", param_name[0] = "L"
                    Rdn += rdn[0]
                elif param_name[0] == name_in_pair[1]:
                    Rdn += rdn[1]

            if len(derp_t[j]) >i:  # if fPd: Dderp_ is step=2, else: Mderp_ is step=2
                derp_t[j][i].rdn = Rdn  # [Lderp_, Iderp_, Dderp_, Mderp_]

    # rdn is updated in each derp, no need to return

def splice_Ps(Ppm_, derp1_, derp2_, fPd, fPpd):  # re-eval Pps, Pp.derp_s for redundancy, eval splice Ps
    '''
    Initial P termination is by pixel-level sign change, but resulting separation may not be significant on a pattern level.
    That is, separating opposite-sign patterns are weak relative to separated same-sign patterns, especially if similar.
     '''
    for i, Pp in enumerate(Ppm_):
        if fPpd: value = abs(Pp.D)  # DPpm_ if fPd, else IPpm_
        else: value = Pp.M  # add summed P.M|D?

        if value > ave_M * (ave_D*fPpd) * Pp.Rdn * 4 and Pp.L > 4:  # min internal xP.I|D match in +Ppm
            M2 = M1 = 0
            for derp2 in derp2_: M2 += derp2.m  # match(I, __I or D, __D): step=2
            for derp1 in derp1_: M1 += derp1.m  # match(I, _I or D, _D): step=1

            if M2 / max( abs(M1), 1) > ave_splice:  # similarity / separation(!/0): splice Ps in Pp, also implies weak Pp.derp_:
                # Pp is now primarily a spliced P, or higher Pp a spliced lower Pp:
                if isinstance(derp1.P, CP):
                    P = CP()
                else:
                    P = CPp()  # if called from line_recursive
                P.x0 = Pp.derp_[0].P.x0
                P.I = sum([derp.P.I for derp in Pp.derp_])
                P.D = sum([derp.P.D for derp in Pp.derp_])
                P.M = sum([derp.P.M for derp in Pp.derp_])
                P.Rdn = sum([derp.P.Rdn for derp in Pp.derp_])
                for derp in Pp.derp_: P.dert_ += derp.P.dert_
                P.L = len(P.dert_)
                # re-run line_P sub-recursion per spliced P, or lower spliced Pp in line_recursive:
                range_incr_P_([], [P], rdn=1, rng=1)
                deriv_incr_P_([], [P], rdn=1, rng=1)
                Pp.P = P
        '''
        no splice(): fine-grained eval per P triplet is too expensive?
        '''

def deriv_incr(rootPp, Pp_, hlayers):  # evaluate each Pp for incremental derivation, as in line_patterns der_comp but with higher local ave,
    # pack results into sub_Pm_
    comb_sublayers = []  # combine into root P sublayers[1:], each nested to depth = sublayers[n]

    for i, Pp in enumerate(Pp_):
        if Pp.L > 1 and not isinstance(Pp.P, CP):  # else Pp is a spliced P
            loc_ave_M = ave_M * Pp.Rdn * hlayers * ((Pp.M / ave_M) / 2) * ave_D  # ave_D=ave_d?
            iM = sum( [derp.P.M for derp in Pp.derp_])
            loc_ave = (ave + iM) / 2 * ave_d * Pp.Rdn * hlayers  # cost per cross-comp

            if abs(Pp.D) / Pp.L > loc_ave_M * 4:  # 4: search cost, + Pp.M: local borrow potential?
                sub_search(Pp, fPd=True)  # search in top sublayer, eval by derp.d
                dderp_ = []
                for _derp, derp in zip( Pp.derp_[:-1], Pp.derp_[1:]):  # or comp abs d, or step=2 for sign match?
                    dderp_ += [comp_par(_derp.P, _derp.d, derp.d, "D_", loc_ave * ave_mD)]  # cross-comp of ds
                    cD = sum( abs(dderp.d) for dderp in dderp_)
                if cD > loc_ave_M * 4:  # fixed costs of clustering per Pp.derp_
                    sub_Ppm_, sub_Ppd_ = [], []
                    Pp.sublayers = [(sub_Ppm_, sub_Ppd_)]
                    sub_Ppm_[:] = form_Pp_(dderp_, fPd=False)
                    sub_Ppd_[:] = form_Pp_(dderp_, fPd=True)
                    if sub_Ppd_ and abs(Pp.D) + Pp.M > loc_ave_M * 4:  # looping search cost, diff val per Pd_'DPpd_, +Pp.iD?
                        deriv_incr(Pp, sub_Ppd_, hlayers+1)  # recursive der+, no need for derp_, no rng+: Pms are redundant?
                    else:
                        Pp.sublayers = []  # reset after the above converts it to [([],[])]

        if Pp.sublayers:  # pack added sublayers, same as in range_incr:
            new_comb_sublayers = []
            for (comb_sub_Ppm_, comb_sub_Ppd_), (sub_Ppm_, sub_Ppd_) in zip_longest(comb_sublayers, Pp.sublayers, fillvalue=([], [])):
                comb_sub_Ppm_ += sub_Ppm_  # use brackets for nested P_ttt
                comb_sub_Ppd_ += sub_Ppd_
                new_comb_sublayers.append((comb_sub_Ppm_, comb_sub_Ppd_))  # add sublayer
            comb_sublayers = new_comb_sublayers

    if rootPp and comb_sublayers: rootPp.sublayers += comb_sublayers  # new sublayers


def range_incr(rootPp, Pp_, hlayers, rng):  # evaluate each Pp for incremental range, fixed: parallelizable, rng-selection gain < costs
    # pack results into sub_Pm_
    comb_sublayers = []  # combine into root P sublayers[1:], each nested to depth = sublayers[n]

    for i, Pp in enumerate(Pp_):
        if Pp.L > rng+1 and not isinstance(Pp.P, CP):  # else Pp is a spliced P
            loc_ave_M = ((Pp.M + ave_M) / 2) * Pp.Rdn * hlayers  # cost per loop = (ave of global and local aves) * redundancy
            iM = sum( [derp.P.M for derp in Pp.derp_])
            loc_ave = (ave + iM) / 2 * Pp.Rdn * hlayers  # cost per cross-comp

            if Pp.M / Pp.L > loc_ave_M + 4:  # 4: search cost, + Pp.iM?
                sub_search(Pp, fPd=False)
                Rderp_, cM = comp_rng(Pp, loc_ave * ave_mI, rng)  # accumulates Rderps from fixed-rng rderp_
                if cM > loc_ave_M * 4:  # current-rng M > fixed costs of clustering per Pp.derp_, else reuse it for multiple rmg+?
                    sub_Ppm_, sub_Ppd_ = [], []
                    Pp.sublayers = [(sub_Ppm_, sub_Ppd_)]
                    sub_Ppm_[:] = form_rPp_(Rderp_)
                    if sub_Ppm_ and Pp.M > loc_ave_M * 4:  # 4: looping cost, if Pm_'IPpm_.M, +Pp.iM?
                        range_incr(Pp, sub_Ppm_, hlayers+1, rng+1)  # recursive rng+, no der+ in redundant Pds?
                    else:
                        Pp.sublayers = []  # reset after the above converts it to [([],[])]

        if Pp.sublayers:  # pack added sublayers, same as in deriv_incr:
            new_comb_sublayers = []
            for (comb_sub_Ppm_, comb_sub_Ppd_), (sub_Ppm_, sub_Ppd_) in zip_longest(comb_sublayers, Pp.sublayers, fillvalue=([],[])):
                comb_sub_Ppm_ += sub_Ppm_  # use brackets for nested P_ttt
                comb_sub_Ppd_ += sub_Ppd_
                new_comb_sublayers.append((comb_sub_Ppm_, comb_sub_Ppd_))  # add sublayer
            comb_sublayers = new_comb_sublayers

    if rootPp and comb_sublayers: rootPp.sublayers += comb_sublayers  # new sublayers
    # Pp_ is changed in-place

def comp_rng(rootPp, loc_ave, rng):  # extended fixed-rng search-right for core I at local ave (lower m)

    if rng==2:  # 1st call, initialize Rderp_ with aderps
        Rderp_ = [Cderp( P=CP(), aderp=derp, rdn = derp.P.Rdn + derp.rdn)  # rdn accum from derps ) levels
                  for derp in rootPp.derp_]
    else:  # rderps are left and right from aderp, evaluate per rng+1:
        Rderp_ = rootPp.derp_.copy()  # copy to avoid overwriting derp.roots
    cM = 0
    for Rderp in Rderp_: Rderp.roots = []  # reset for future rPps

    for i, _Rderp in enumerate(Rderp_[:-rng]):  # cross-comp at rng to extend Rderps, exclude last rng Rderps (no comparands)

        _aderp = _Rderp.aderp
        Rderp = Rderp_[i + rng]  # rderp in which aderp is an anchor
        rderp = Cderp(P=_aderp.P, roots=Rderp)  # rng dert
        aderp = Rderp.aderp  # compared dert
        rderp.p = aderp.i + _aderp.i  # -> ave i
        rderp.d = aderp.i - _aderp.i  # difference
        rderp.m = loc_ave - abs(_aderp.d)  # indirect match
        cM += rderp.m  # to eval form_rPp_
        if rderp.m > ave_M * 4 and rderp.P.sublayers and aderp.P.sublayers:  # 4: init ave_sub coef
            # deeper cross-comp between high-m sub_Ps, separate from rderp.m:
            comp_sublayers(rderp.P, aderp.P, rderp.m)
        # left Rderp assign:
        _Rderp.accum_from(rderp)  # Pp params += derp params, including rdn: rdert-specific
        _Rderp.P.accum_from(rderp.P, excluded=["x0"])
        _Rderp.rderp_ += [rderp]  # extend _Rderp to the right
        # right Rdert assign:
        Rderp.accum_from(rderp)  # Pp params += derp params
        Rderp.P.accum_from(rderp.P, excluded=["x0"])
        Rderp.rderp_.insert(0, rderp.copy())  # extend Rderp to the left
        Rderp.rderp_[0].roots = _Rderp

    return Rderp_, cM


def form_rPp_(Rderp_):  # evaluate inclusion in _rPp of accumulated Rderts, by mutual olp_M within comp rng
    '''
    this is a version of agglomerative clustering, similar to https://en.wikipedia.org/wiki/Single-linkage_clustering,
    but between cluster-level representations vs. their elements (which is done in splice_Pp):
    primary clustering is by rderp.m: direct match between Rderp.aderps
    secondary clustering is by merging rPp of matching Rderts: clustered nodes don't need to directly match each other
    '''
    rPp_ = []
    for _Rderp in Rderp_:
        # initialize _rPp to merge all matching rPps in _Rderp.rderp_:
        if isinstance(_Rderp.roots, CPp):
            _rPp = _Rderp.roots
        else:
            _rPp = CPp(derp_=[_Rderp], L=1)
            _Rderp.roots = _rPp
        _rPp.accum_from(_Rderp, ignore_capital=True)  # no Rderp.rdn *= rng: rderp-specific accum, along with m?

        olp_M = 0
        i_=[]
        for i, rderp in enumerate(_Rderp.rderp_):  # sum in olp_M to evaluate rderp.Rderp inclusion in _rPp
            if rderp.m > 0:
                olp_M += rderp.m  # match between anchor derts
                i_.append(i)
        if olp_M / max(1, len(i_)) > ave_M * 4:  # clustering by variable cost of process in +rPp, vs mean M of overlap
            for i in i_:
                rderp = _Rderp.rderp_[i]
                Rderp = rderp.roots
                rPp = Rderp.roots
                # merge _rPp:
                if isinstance(rPp, CPp):
                    for cRderp in rPp.derp_:
                        if cRderp not in _rPp.derp_:
                            cRderp.roots = _rPp
                            _rPp.accum_from(cRderp, ignore_capital=True)
                            _rPp.derp_.append(cRderp)
                            _rPp.L += 1
                    rPp_.remove(rPp)
                else:
                    rderp.roots.roots = _rPp
                    _rPp.accum_from(Rderp, ignore_capital=True)
                    _rPp.derp_.append(Rderp)
                    _rPp.L += 1

        # else: _rPp is single Rderp, include in rPp_ anyway?:
        rPp_.append(_rPp)

    return rPp_  # no term_rPp


def sub_search(rootPp, fPd):  # ~line_PPs_root: cross-comp sub_Ps in top sublayer of high-M Pms | high-D Pds, called from intra_Pp_

    for derp in rootPp.derp_:  # vs. Idert_
        P = derp.P
        if P.sublayers:  # not empty sublayer
            subset = P.sublayers[0]
            for fsubPd, sub_P_ in enumerate([subset[0], subset[1]]):  # sub_Pm_, sub_Pd_
                if len(sub_P_) > 2 and ((fPd and abs(P.D) > ave_D * rootPp.Rdn) or (P.M > ave_M * rootPp.Rdn)):
                    # + derp.m + sublayer.Dert.M?
                    sub_derp_t, dert1_, dert2_ = cross_comp(sub_P_, fsubPd)  # derp_t: Ldert_, Idert_, Ddert_, Mdert_
                    sum_rdn_(param_names, sub_derp_t, fsubPd)
                    paramset = []
                    # derp_-> Pp_:
                    for param_name, sub_derp_ in zip(param_names, sub_derp_t):
                        param_md = []
                        for fPpd in 0, 1:
                            sub_Pp_ = form_Pp_(deepcopy(sub_derp_), fPpd)
                            param_md.append(sub_Pp_)
                            if (fPpd and param_name == "D_") or (not fPpd and param_name == "I_"):
                                if not fPpd:
                                    splice_Ps(sub_Pp_, dert1_, dert2_, fPd, fPpd)  # splice eval by Pp.M in Ppm_, for Pms in +IPpms or Pds in +DPpm
                                range_incr([], sub_Pp_, hlayers=1, rng=2)  # eval rng+ comp,form per Pp
                                deriv_incr([], sub_Pp_, hlayers=1)  # eval der+ comp,form per Pp
                        paramset.append(param_md)
                    P.subset[4 + fsubPd].append(paramset)  # sub_Ppm_tt and sub_Ppd_tt
                    # deeper comp_sublayers is selective per sub_P


def comp_sublayers(_P, P, root_v):  # if derp.m -> if summed params m -> if positional m: mx0?

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
            _xsub_derpt_ = _P.subset[2+fPd]  # array of cross-sub_P derp tuples
            xsub_derpt_ = P.subset[2+fPd]

            if rng == _rng and min(_P.L, P.L) > ave_Ls:  # if same fork: comp sub_Ps to each _sub_P in max relative distance per comb_V:
                _SL = SL = 0  # summed Ls
                start_index = next_index = 0  # index of starting sub_P for current _sub_P
                for _sub_P in _sub_P_:
                    _xsub_derpt_ += [[]]  # append xsub_dertt per _sub_P, sparse, inner brackets for subset_?
                    xsub_derpt_ += [[]]
                    _xsub_derpt = [[], [], [], []]  # L, I, D, M xsub_derp_s
                    _SL += _sub_P.L  # ix0 of next _sub_P;  no xsub_derpt_ per sub_P_, summed only?
                    # search right:
                    for sub_P in sub_P_[start_index:]:  # index_ix0 > _ix0, comp sub_Ps at proximate relative positions in sub_P_
                        fbreak, xsub_P_M, xsub_P_D = comp_sub_P(_sub_P, sub_P, _xsub_derpt, root_v, fPd)
                        sub_M += xsub_P_M ; sub_D += xsub_P_D
                        if fbreak:
                            break
                        # if next ix overlap: ix0 of next _sub_P < ix0 of current sub_P
                        if SL < _SL: next_index += 1
                        SL += sub_P.L  # ix0 of next sub_P
                    # search left:
                    for sub_P in reversed(sub_P_[len(sub_P_) - start_index:]):  # index_ix0 <= _ix0
                        fbreak, xsub_P_M, xsub_P_D = comp_sub_P(_sub_P, sub_P, _xsub_derpt, root_v, fPd)  # invert sub_P, _sub_P positions
                        sub_M += xsub_P_M ; sub_D += xsub_P_D
                        if fbreak:
                            break
                    start_index = next_index  # for next _sub_P

                    if any(_xsub_derpt):  # at least 1 sub_derp, real min length ~ 8, very unlikely
                        xsub_Pp_t = []  # LPpm_, IPpm_, DPpm_, MPpm_
                        sum_rdn_(param_names, _xsub_derpt, fPd)  # no return from sum_rdn now
                        for param_name, xsub_derp_ in zip(param_names, _xsub_derpt):
                            xsub_Pp_ = form_Pp_(deepcopy(xsub_derp_), fPd=0)
                            # no step=2 for splice: xsub_derpts are not consecutive, and their Ps are not aligned?
                            if param_name == "I_":
                                splice_Ps(xsub_Pp_, [], [], fPd, fPpd=0)  # splice eval by Pp.M in Ppm_, for Pms in +IPpms or Pds in +DPpm
                                range_incr([], xsub_Pp_, hlayers=1, rng=2)  # eval rng+ comp,form per Pp
                                deriv_incr([], xsub_Pp_, hlayers=1)  # eval der+ comp,form per Pp
                            xsub_Pp_t.append(xsub_Pp_)

                        _xsub_derpt_[-1][:] = xsub_Pp_t
                        xsub_derpt_[-1][:] = xsub_Pp_t  # bilateral assignment?

    return sub_M, sub_D


def comp_sub_P(_sub_P, sub_P, xsub_derpt, root_v, fPd):
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
            sub_derp = comp_par(sub_P, _param, param, param_name, ave)
            # no negative-value sum: -Ps won't be processed:
            if sub_derp.m > 0: xsub_P_M += sub_derp.m
            vd = abs(sub_derp.d) - ave_d  # convert to coef
            if vd > 0: xsub_P_D += vd
            xsub_derpt[i].append(sub_derp)  # per L, I, D, M' xsub_derp

        V += xsub_P_M + xsub_P_D  # separate eval for form_Pm_ and form_Pd_?

        if V > ave_M * 4 and _sub_P.sublayers and sub_P.sublayers:  # 4: ave_comp_sublayers coef
            # update sub_M in Iderp(xsub_derpt[1]) only?
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
xsub_derps (cross-sub_P): xsub_derpt_[ xsub_derpt [ xsub_derp_[ sub_derp]]]: each bracket is a level of nesting
the above forms Pps across xsub_dertt, then also form Pps across _xsub_derpt_ and xsub_derpt_? 
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

# to avoid intermediate clustering costs and selection, but better to just skip clustering phase with comp_rng:
# if multiple rng extension cycle:

def search_rng(rootPp, loc_ave, rng):  # extended fixed-rng search-right for core I at local ave: lower m

    idert_ = rootPp.derp_.copy()  # copy to avoid overwriting idert.roots:
    Rderp_ = idert_.copy()  # each bilateral, evaluation per search, also symmetry per evaluated P|dert

    for i, (Rderp, idert) in enumerate( zip(Rderp_, idert_)):
        Rderp.aderp=idert  # initialize Rderp for each idert, which is Rderp' anchor dert
        Rderp.roots = []  # reset

    for i, (idert, _Rderp) in enumerate(zip( idert_, Rderp_)):  # form consecutive fixed-rng Rderps, overlapping within rng-1
        j = i + rootPp.x0 + 1  # comparand index in idert_, step=1 was in cross-comp, start at step=2 or 1 + prior rng

        while j - (i + rootPp.x0 + 1) < rng and j < len(idert_) - 1:
            # cross-comp within rng:
            Rderp = Rderp_[j]  # Rderp in which cdert is an anchor
            rderp = Cderp(P=idert.P, roots=Rderp)  # rng dert
            cdert = idert_[j]  # compared-P dert
            rderp.p = cdert.i + idert.i  # -> ave i
            rderp.d = cdert.i - idert.i  # difference
            rderp.m = loc_ave - abs(idert.d)  # indirect match
            j += 1
            if rderp.m > ave_M * 4 and rderp.P.sublayers and cdert.P.sublayers:  # 4: init ave_sub coef
                comp_sublayers(rderp.P, cdert.P, rderp.m)  # deeper cross-comp between high-m sub_Ps, separate from ndert.m
            # left Rderp assign:
            _Rderp.accum_from(rderp, ignore_capital=True)  # Pp params += derp params
            _Rderp.rderp_ += [rderp]  # extend _rderp to the right
            # right rderp assign:
            Rderp.accum_from(rderp, ignore_capital=True)  # Pp params += derp params
            Rderp.rderp_.insert(0, rderp.copy())  # extend Rderp to the left
            Rderp.rderp_[0].roots = _Rderp

    return Rderp_


# only if oolp_Rderp not in _rPp.derp_, which is never true:

def eval_olp_recursive(Rderp, olp_dert, _rPp):

    if olp_dert.m + olp_dert.olp_M > ave_dir_m:  # ave_dir_m < ave, negative here, eval by anchor m + overlap m
        # merge overlapping rPp in _rPp:
        rPp = Rderp.roots
        if isinstance(rPp, CPp):  # Rderp has rPp, merge it
            for Rderp in rPp.derp_:
                if Rderp not in _rPp.derp_:
                    _rPp.accum_from(Rderp.roots, excluded=['x0'])
                    _rPp.derp_.append(Rderp)
                    Rderp.roots = _rPp  # update reference
        else:  # no rPp
            _rPp.accum_from(Rderp)
            _rPp.derp_.append(Rderp)  # derp_ is Rderps
            Rderp.roots = _rPp

        oolp_Rderp = rderp.rderp_[0].roots
        if oolp_Rderp not in _rPp.derp_:  # this is never true, oolp_Rderp is always in rPp.derp_
            eval_olp_recursive(oolp_Rderp, Rderp.rderp_[0], _rPp)  # check olp of olp


def form_rPp_old(Rdert_, rng, depth):  # evaluate direct and mediated match between aderp.P and olp_dert.Ps
    '''
    Each rng+ adds a layer of derp_ mediation to form extended graphs of Rderts,
    including Rdert_ derp.Rderts with peri-negative direct match but positive mediated match into rPp.
    It also adds mediated match to positive derps for accuracy. But there will be fewer overlapping derps per mediation
    '''
    rPp_ = []
    for _Rdert in Rdert_:
        if not isinstance(_Rdert.roots, CPp):  # if no rPp is formed from prior merging
            _rPp = CPp(derp_=[_Rdert])
            _Rdert.roots = _rPp
            rPp_.append(_rPp)
            olp_M = 0
            for olp_dert in _Rdert.rdert_:  # breadth-first evaluation:
                rel_m = olp_dert.m / max(1, olp_dert.i)  # ratio of aderp m to olp_dert mag
                olp_M += olp_dert.m + olp_dert.roots.m * rel_m  # match to olp derp.Ps is estimated from direct m ratio
                '''  
                estimated m to olp derp: should be proportional to m between aderps because 
                within overlap between Rderts, rderts are same but compared aderps are different.               
                '''
            if olp_M > ave_M * 4:  # total M of aderp to olp_dert_ is a precondition for rPp extension
                for olp_dert in _Rdert.rdert_:
                    Rdert = olp_dert.roots
                    olp_dert.olp_M += Rdert.rdert_[0].m  # _Rdert olp added per rng+, always 2 rderts per Rdert
                    Rdert.rdert_[0].olp_M += olp_dert.m  # reciprocal left olp extension
                    # test to add olp_Rdert in rPp, then same for higher-order overlaps:
                    eval_olp_recursive(Rdert, olp_dert, _rPp)
    return rPp_