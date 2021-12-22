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
    negM = int  # in mdert only
    negL = int  # in mdert only
    sub_M = int  # match from comp_sublayers, if any
    sub_D = int  # diff from comp_sublayers, if any
    P = object  # P of i param
    Ppt = lambda: [object, object]  # tuple [Ppm,Ppd]: Pps that pdert is in, to join rdert_s

class CPp(CP):
    dert_ = list  # if not empty: Pp is primarily a merged P, other params are optional
    pdert_ = list  # Pp elements, "p" for param
    flay_rdn = bool  # Pp is layer-redundant to Pp.pdert_
    negM = int  # in rng_Pps only
    negL = int  # in rng_Pps only, summed in L, no need to be separate?
    _negM = int  # for search left, within adjacent neg Ppm only?
    _negL = int  # left-most compared distance from Pp.x0
    sublayers = list
    subDerts = list
    sublevels = list  # levels of composition per generic Pp: P ) Pp ) Ppp...
    rootPp = object  # to replace locals for merging
    # layer1: iL, iI, iD, iM, iRdn: summed P params

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

# used in search, form_Pp_root, comp_sublayers_, draw_PP_:
param_names = ["L_", "I_", "D_", "M_"]  # not really needed, we can just use indices?
aves = [ave_mL, ave_mI, ave_mD, ave_mM]
'''
    Conventions:
    postfix 't' denotes tuple, multiple ts is a nested tuple
    (usually the nesting is implicit, actual structure is flat list)
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    capitalized variables are normally summed small-case variables
'''

def line_PPs_root(P_t):  # P_T is P_t = [Pm_, Pd_];  higher-level input is implicitly nested to the depth = 1 + 2*elevation (level counter)

    norm_feedback(P_t)  # before processing

    P_ttt = []  # output is 16-tuple of Pp_s per line, implicitly nested into 3 levels
    for fPd, P_ in enumerate(P_t):  # fPd: Pm_ or Pd_, wrong order?
        if len(P_) > 1:
            Pdert_t, dert1_, dert2_ = cross_comp(P_, fPd)  # Pdert_t: Ldert_, Idert_, Ddert_, Mdert_ (tuples of derivatives per P param)
            sum_rdn_(param_names, Pdert_t, fPd)  # sum cross-param redundancy per pdert, to evaluate for deeper processing
            # P_tt=[] if nested
            for param_name, Pdert_ in zip(param_names, Pdert_t):  # Pdert_ -> Pps:
                for fPpd in 0, 1:  # 0-> Ppm_, 1-> Ppd_: more anti-correlated than Pp_s of different params
                    Pp_ = form_Pp_(Pdert_, fPpd)
                    if (fPpd and param_name == "D_") or (not fPpd and param_name == "I_"):
                        if not fPpd:
                            splice_Ps(Pp_, dert1_, dert2_, fPd, fPpd)  # splice eval by Pp.M in Ppm_, for Pms in +IPpms or Pds in +DPpm
                        intra_Pp_(None, Pp_, Pdert_, 1, fPpd)  # der+ or rng+ per Pp
                    P_ttt.append(Pp_)  # implicit nesting
        else:
            P_ttt.append( [[] for _ in range(8)])  # pack 8 empty P_s to preserve index

    return [P_t, P_ttt]  # P_T_, contains 1st level and 2nd level outputs


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
        m = int_rL * min(param, _param) - (int_rL * frac_rL) / 2 - ave
        # div_comp match is additive compression: +=min, not directional
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

    if fPd: V = abs(D)
    else: V = M
    PpV = V / (L *.7)  # .7: ave intra-Pp-match coef, for value reduction with resolution, still non-negative
    pdert_V = V - L * ave_M * (ave_D * fPd)  # cost incr per pdert representations
    flay_rdn = PpV < pdert_V
    # Pp vs Pdert_ rdn
    Pp = CPp(L=L, I=I, D=D, M=M, Rdn=Rdn+L+L*flay_rdn, x0=x0, ix0=ix0, flay_rdn=flay_rdn, pdert_=pdert_, sublayers=[[]])
    for pdert in Pp.pdert_: pdert.Ppt[fPd] = Pp  # root Pp refs
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
        if fPpd: V = abs(Pp.D)  # DPpm_ if fPd, else IPpm_
        else: V = Pp.M  # add summed P.M|D?

        if V > ave_M * (ave_D*fPpd) * Pp.Rdn * 4 and Pp.L > 4:  # min internal xP.I|D match in +Ppm
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

def intra_P(P, rdn, rng, fPd):  # this really a return to line_Ps
    comb_sublayers = []

    if not fPd:
        if P.M - P.Rdn * ave_M * P.L > ave_M * rdn and P.L > 2:  # M value adjusted for xP and higher-layers redundancy
            rdn+=1; rng+=1
            sub_Pm_, sub_Pd_ = [], []
            P.sublayers += [[(rdn, rng, sub_Pm_, sub_Pd_, [], [], [], [] )]]  # 4[]: xsub_pmdertt_, _xsub_pddertt_, sub_Ppm_, sub_Ppd_
            rdert_ = range_comp(P.dert_)  # rng+, skip predictable next dert, local ave? rdn to higher (or stronger?) layers
            sub_Pm_[:] = form_P_(P, rdert_, rdn, rng, fPm=True)  # cluster by rm sign
            sub_Pd_[:] = form_P_(P, rdert_, rdn, rng, fPm=False)  # cluster by rd sign
        else:
            P.sublayers += [[]]  # empty subset to preserve index in sublayer
    else:  # P is Pd
        if abs(P.D) - (P.L - P.Rdn) * ave_D * P.L > ave_D * rdn and P.L > 1:  # high-D span, level rdn, vs. param rdn in dert
            rdn+=1; rng+=1
            sub_Pm_, sub_Pd_ = [], []  # initialize layers top-down, concatenate by intra_P_ in form_P_
            P.sublayers += [[(rdn, rng, sub_Pm_, sub_Pd_, [], [], [], [] )]]  # 4[]: xsub_pmdertt_, _xsub_pddertt_, sub_Ppm_, sub_Ppd_
            ddert_ = deriv_comp(P.dert_)  # i is d
            sub_Pm_[:] = form_P_(P, ddert_, rdn, rng, fPm=True)  # cluster by mm sign
            sub_Pd_[:] = form_P_(P, ddert_, rdn, rng, fPm=False)  # cluster by md sign
        else:
            P.sublayers += [[]]  # empty subset to preserve index in sublayer

    if P.sublayers:
        comb_sublayers = [comb_subset_ + subset_ for comb_subset_, subset_ in
                          zip_longest(comb_sublayers, P.sublayers, fillvalue=[])
                          ]

    P.sublayers += comb_sublayers  # no return


def intra_Pp_(rootPp, Pp_, Pdert_, hlayers, fPd):  # evaluate for sub-recursion in line Pm_, pack results into sub_Pm_
    '''
    each Pp may be compared over incremental range or derivation, as in line_patterns but with higher local ave
    '''
    comb_sublayers = []  # combine into root P sublayers[1:], each nested to depth = sublayers[n]

    for i, Pp in enumerate(Pp_):
        loc_ave_M = ave_M * Pp.Rdn * hlayers
        if Pp.L > 1 and Pp.M > loc_ave_M:  # min for both forks

            loc_ave_M *= (Pp.M / ave_M) / 2
            iM = sum( [pdert.P.M for pdert in Pp.pdert_])
            loc_ave = (ave + iM) / 2 * Pp.Rdn * hlayers  # cost per comp
            if fPd:
                loc_ave *= ave_d; loc_ave_M *= ave_D  # =ave_d?
                # der+ fork
                if abs(Pp.D) / Pp.L > loc_ave_M * 4:  # 4: search cost, + Pp.M: local borrow potential?
                    sub_search(Pp, fPd=True)  # search in top sublayer, eval by pdert.d
                    sub_Ppm_, sub_Ppd_ = [], []
                    Pp.sublayers = [[(sub_Ppm_, sub_Ppd_)]]
                    ddert_ = []
                    for _pdert, pdert in zip( Pp.pdert_[:-1], Pp.pdert_[1:]):  # or comp abs d, or step=2 for sign match?
                        ddert_ += [comp_par(_pdert.P, _pdert.d, pdert.d, "D_", loc_ave * ave_mD)]  # cross-comp of ds
                    sub_Ppm_[:] = form_Pp_(ddert_, fPd=False)
                    sub_Ppd_[:] = form_Pp_(ddert_, fPd=True)
                    if abs(Pp.D) + Pp.M > loc_ave_M * 4:  # 4: looping search cost, diff induction per Pd_'DPpd_, +Pp.iD?
                        intra_Pp_(Pp, sub_Ppd_, None, hlayers+1, fPd)  # recursive der+, no rng+: Pms are redundant?
                else:
                    Pp.sublayers += [[]]  # empty subset to preserve index in sublayer, or increment index of subset?
            else:
                # rng+ fork
                if Pp.M / Pp.L > loc_ave_M + 4:  # 4: search cost, + Pp.iM?
                    sub_search(Pp, True)  # search in top sublayer, eval by pdert.d
                    sub_Ppm_, sub_Ppd_ = [], []
                    Pp.sublayers = [[(sub_Ppm_, sub_Ppd_)]]
                    # higher ave -> distant match, higher ave_negM -> extend Pp
                    rdert_ = search_Idert_(Pp, Pdert_, loc_ave * ave_mI)  # comp x variable range, while curr_M
                    sub_Ppm_[:] = join_rng_pdert_s(rdert_.copy())  # rdert_ contains P+pdert_s that form rng_Pps

                    if Pp.M > loc_ave_M * 4 and not Pp.dert_:  # 4: looping cost, not spliced Pp, if Pm_'IPpm_.M, +Pp.iM?
                        intra_Pp_(Pp, sub_Ppm_, rdert_, hlayers+1, fPd)  # recursive rng+ per joined cluster, no der+ in redundant Pds?
                else:
                    Pp.sublayers += [[]]  # empty subset to preserve index in sublayer, or increment index of subset?

        if isinstance(rootPp, CPp) and Pp.sublayers[0]:
            comb_sublayers = [comb_subset_ + subset_ for comb_subset_, subset_ in
                              zip_longest(comb_sublayers, Pp.sublayers, fillvalue=[])
                             ]
    if isinstance(rootPp, CPp):
        rootPp.sublayers += comb_sublayers

    # no return, Pp_ is changed in-place

def search_Idert_(root_Pp, Idert_, loc_ave):  # extended fixed-rng search-right for core I at local ave: lower m
    # fixed because it's parallelizable and individual extensions are not worth it

    rng = int( root_Pp.M / root_Pp.L / 4)  # ave_rng
    Pp_ = []
    idert_ = root_Pp.pdert_.copy()
    for idert in idert_: idert.Ppt = [[],[]]  # clear higher-level ref for current level

    for i, idert in enumerate(idert_):  # overlapping pderts and +Pps, no -Pps
        j = i + root_Pp.x0 + 1  # start at step=2, step=1 was in cross-comp
        Pp = CPp()
        while j-i < rng and j < len(Idert_) - 1:
            # cross-comp:
            cdert = Idert_[j]  # current dert with compared P
            idert.p = cdert.i + idert.i  # -> ave i
            idert.d = cdert.i - idert.i  # difference
            idert.m = loc_ave - abs(idert.d)  # indirect match
            if idert.m > 0:
                if idert.m > ave_M * 4 and idert.P.sublayers[0] and cdert.P.sublayers[0]:  # 4: init ave_sub coef
                    comp_sublayers(idert.P, cdert.P, idert.m)
                Pp.accum_from(idert, excluded=['x0'])  # Pp params += pdert params
                Pp.pdert_ += [idert]; idert.Ppt[0] += [Pp]
                idert = Cpdert()
            else:  # idert miss, represent discontinuity:
                idert.negL += 1
                idert.negM += idert.m
            j += 1
        if idert.m <= 0:  # add last idert if negative:
            Pp.accum_from(idert, excluded=['x0'])  # Pp params += pdert params
            Pp.pdert_ += [idert]; idert.Ppt[0] += [Pp]
        Pp_.append(Pp)

    return Pp_  # vs. rng_dert_
'''
    for variable rng:
    for i, idert in enumerate(Pp.pdert_):  # overlapping pderts and +Pps, no -Pps
        # search right:
        j = i + Pp.x0 + 1  # start at step=2, step=1 was in cross-comp
        search_direction(Pp, idert, rng_dert_, Idert_, j, loc_ave, fleft=0)
        # search left:
        j = i + Pp.x0 - 1  # start at step=2, step=1 was in cross-comp
        search_direction(Pp, idert, rng_dert_, Idert_, j, loc_ave, fleft=1)
'''

# draft
def join_rng_pdert_s(Pp_):  # vs. merge, also removes redundancy, no need to adjust?
    _Pp = Pp_[0]
    for Pp in Pp_[1:]:

        Pp.pdert_ = [Pp.pdert_]  # convert into nested list
        for pdert in Pp.pdert_:
            if pdert.Ppt[0][0] is Pp:  # common Pp, single-element Ppt[0] at this point?
                # compare initial Pp params:
                xPp_m = 0  # total match between Pps
                ppdert_ = []  # xparam derts
                for param_name in param_names:
                    _param = getattr(_Pp, param_name[1:])  # skip L: = rng, I, D, M only
                    param = getattr(Pp, param_name[1:])
                    d = param - _param  # difference
                    if param_name == 'I_': m = ave - abs(d)  # indirect match
                    else: m = min(param, _param) - abs(d) / 2 - ave  # direct match
                    xPp_m += m
                    ppdert_.append(Cpdert(P=Pp, i=_param, p=param + _param, d=d, m=m))
                if xPp_m > ave_M * 4:
                    Pp.accum_from(_Pp)
                    Pp.pdert_ += pdert._Pp.pdert_  # should be nested, make it recursive:
                    # while Pp is list (or is not CPp):
                    #   for rdert in Pp.rdert_:...
                    Pp_.remove(_Pp)  # redundant to clustered representation, remove with all nesting
        _Pp = Pp

def sub_search(rootPp, fPd):  # ~line_PPs_root: cross-comp sub_Ps in top sublayer of high-M Pms | high-D Pds, called from intra_Pp_

    for pdert in rootPp.pdert_:
        P = pdert.P
        if P.sublayers[0]:  # not empty sublayer
            subset = P.sublayers[0][0]  # single top sublayer subset
            for fsubPd, sub_P_ in enumerate([subset[2], subset[3]]):  # sub_Pm_, sub_Pd_

                if len(sub_P_) > 1 and ((fPd and abs(P.D) > ave_D * rootPp.Rdn) or (P.M > ave_M * rootPp.Rdn)):
                    # + pdert.m + sublayer.Dert.M?
                    sub_Pdert_t, dert1_, dert2_ = cross_comp(sub_P_, fsubPd)  # Pdert_t: Ldert_, Idert_, Ddert_, Mdert_
                    sub_Pp_tt = []  # Ppm_t, Ppd_t, each: [LPp_, IPp_, DPp_, MPp_]

                    for fPpd in 0, 1:  # 0-> Ppm_t, 1-> Ppd_t
                        sub_Pp_t = []  # [LPp_, IPp_, DPp_, MPp_]
                        sub_rdn_t = sum_rdn_(param_names, sub_Pdert_t, fPd)
                        # Pdert_-> Pp_:
                        for param_name, sub_Pdert_, rdn_ in zip(param_names, sub_Pdert_t, sub_rdn_t):
                            sub_Pp_ = form_Pp_(sub_Pdert_, fPpd)
                            if (fPpd and param_name == "D_") or (not fPpd and param_name == "I_"):
                                if not fPpd:
                                    splice_Ps(sub_Pp_, dert1_, dert2_, fPd)  # splice eval by Pp.M in Ppm_, for Pms in +IPpms or Pds in +DPpm
                                sub_Pp_ = intra_Pp_(None, sub_Pp_, sub_Pdert_, 1, fPpd)  # der+ or rng+
                            sub_Pp_t.append(sub_Pp_)
                        sub_Pp_tt.append(sub_Pp_t)
                    subset[6 + fsubPd].append(sub_Pp_tt)  # sub_Ppm_tt and sub_Ppd_tt
                    # deeper comp_sublayers is selective per sub_P


def comp_sublayers(_P, P, root_v):  # if pdert.m -> if summed params m -> if positional m: mx0?
    # replace root_v with root_vt: separate evaluations?

    sub_M, sub_D = 0, 0
    xDert_vt = form_comp_Derts(_P, P, root_v)  # value tuple: separate vm, vd for comp sub_mP_ or sub_dP_
    # comp sub_Ps between 1st sublayers of _P and P:
    _rdn, _rng = _P.sublayers[0][0][:2]
    rdn, rng = P.sublayers[0][0][:2]  # 2nd [0] is a sole subset: rdn, rng, sub_Pm_, sub_Pd_, xsub_pmdertt_, xsub_pddertt_

    for fPd, xDert_v in enumerate(xDert_vt):  # eval m, d:
        if fPd: sub_D += xDert_v
        else: sub_M += xDert_v
        if xDert_v + root_v > ave_M * (fPd * ave_D) * rdn:
            _sub_P_, _, _xsub_pdertt_ = _P.sublayers[0][0][2+fPd: 4+fPd+1]
            sub_P_, _, xsub_pdertt_ = P.sublayers[0][0][2+fPd: 4+fPd+1]

            if rng == _rng and min(_P.L, P.L) > ave_Ls:  # if same fork: compare sub_Ps to each _sub_P within max relative distance per comb_V:
                _SL = SL = 0  # summed Ls
                start_index = next_index = 0  # index of starting sub_P for current _sub_P
                _xsub_pdertt_ += [[]]  # array of cross-sub_P pdert tuples, inner brackets for subset_
                xsub_pdertt_ += [[]]  # append xsub_dertt per _sub_P_ and sub_P_, sparse?

                for _sub_P in _sub_P_:
                    _xsub_pdertt = [[], [], [], []]  # L, I, D, M xsub_pdert_s
                    _SL += _sub_P.L  # ix0 of next _sub_P
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

                    if _xsub_pdertt[0]:  # at least 1 sub_pdert, real min length ~ 8, very unlikely
                        xsub_Pp_t = []  # LPpm_, IPpm_, DPpm_, MPpm_
                        rdn_t = sum_rdn_(param_names, _xsub_pdertt, fPd)
                        for param_name, xsub_Pdert_, rdn_ in zip(param_names, _xsub_pdertt, rdn_t):
                            xsub_Pp_ = form_Pp_(xsub_Pdert_, fPd=0)
                            # no step=2 for splice: xsub_pdertts are not consecutive, and their Ps are not aligned?
                            if param_name == "I_":
                                xsub_Pp_ = intra_Pp_(None, xsub_Pp_, xsub_Pdert_, 1, fPd=0)  # rng+ only?
                            xsub_Pp_t.append(xsub_Pp_)

                        _xsub_pdertt_[-1][:] = xsub_Pp_t
                        xsub_pdertt_[-1][:] = xsub_Pp_t  # bilateral assignment?

    return sub_M, sub_D


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

        if V > ave_M * 4 and _sub_P.sublayers[0] and sub_P.sublayers[0]:  # 4: ave_comp_sublayers coef
            # if sub_P.sublayers is not empty, then sub_P.sublayers[0] can't be empty
            # update sub_M in Ipdert(xsub_pdertt[1]) only?
            sub_M, sub_D = comp_sublayers(_sub_P, sub_P, V)  # recursion for deeper layers
            xsub_P_M += sub_M; xsub_P_D += sub_M
    else:
        fbreak = 1  # only sub_Ps with relatively proximate position in sub_P_|_sub_P_ are compared

    return fbreak, xsub_P_M, xsub_P_D


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
