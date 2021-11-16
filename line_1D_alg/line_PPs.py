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
-
Pp is a graph consisting of pderts: each has a node (P) + right edge (derivatives).
pdert is P + 1st right match, diff, + intermediate negM, negL. If 1st right match: pdert is positive, else: negative.
So, positive Pp has negative-m 1st and last pderts, and positive pderts-m in between.
In negative pderts that edge doesn't connect to another node, but we can extend the range of search for it.
'''

import sys  # add CogAlg folder to system path
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname("CogAlg"), '..')))
import numpy as np
from frame_2D_alg.class_cluster import ClusterStructure, comp_param
from line_patterns import *

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
    Ppt = lambda: [object, object]  # tuple [Ppm,Ppd]: Pps that pdert is in, for merging in form_Pp_rng, temporary?

class CPp(CP):
    dert_ = list  # if not empty: Pp is primarily a merged P, other params are optional
    pdert_ = list  # Pp elements, "p" for param
    flay_rdn = bool  # Pp is layer-redundant to Pp.pdert_
    negM = int  # in mdert only
    negL = int  # in mdert only  # summed in L, no need for separate sum?
    _negM = int  # for search left, within adjacent neg Ppm only?
    _negL = int  # left-most compared distance from Pp.x0
    sublayers = list
    subDerts = list
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

class CPP(CPp, CderPp):  # obsolete, no Pp grouping in PPs?
    layer1 = dict

ave = 1  # ave dI -> mI, * coef / var type
# no ave_mP: deviation computed via rM  # ave_mP = ave* n_comp_params: comp cost, or n vars per P: rep cost?
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
'''
    Conventions:
    postfix 't' denotes tuple
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    capitalized variables are normally summed small-case variables
'''

def line_PPs_root(P_t):  # P_t= Pm_, Pd_;  higher-level input is nested to the depth = 2*elevation (level counter), or 2^elevation?
    Pp_ttt = []
    norm_feedback(P_t)  # before processing

    for fPd, P_ in enumerate(P_t):  # fPd: Pm_| Pd_
        if len(P_) > 1:
            Pdert_t, dert1_, dert2_ = cross_comp(P_, fPd)  # Pdert_t: Ldert_, Idert_, Ddert_, Mdert_
            Pp_tt = form_Pp_root(Pdert_t, dert1_, dert2_, fPd)  # -> Ppm_t, Ppd_t, each: LPp_, IPp_, DPp_, MPp_
            Pp_ttt.append(Pp_tt)
        else:
            Pp_ttt.append(P_)

    return Pp_ttt  # 3-level nested tuple per line: Pm_, Pd_( Ppm_, Ppd_( LPp_, IPp_, DPp_, MPp_)))

def form_Pp_root(Pdert_t, dert1_, dert2_, fPd):  # Ppm_t and Ppd_t forks, can be combined but it would be more opaque

    Ppm_t = []  # [LPpm_, IPpm_, DPpm_, MPpm_]
    rdn_t = sum_rdn_(param_names, Pdert_t, fPd=0)  # assign redundancy to lesser-magnitude m|d in param pair for same-_P Pderts
    for param_name, Pdert_, rdn_ in zip(param_names, Pdert_t, rdn_t):
        Ppm_ = form_Pp_(Pdert_, fPd=0)  # segment Pdert__ into Ppms
        # Ppm_ only:
        if (param_name == "I_" or param_name=="D_"):
            rootM = sum(Pp.M for Pp in Ppm_) + sum(pdert.P.M for pdert in Pdert_)  # input match in two overlapping layers
            if rootM > ave_M * 4:
                if fPd:
                    if param_name=="D_": splice_Ps(Ppm_, dert1_, dert2_, fPd)  # eval splice Pds in each +DPpm
                elif param_name=="I_": splice_Ps(Ppm_, dert1_, dert2_, fPd)  # eval splice Pms in each +IPpm
        if param_name=="I_": Ppm_ = intra_Pp_(None, Ppm_, Pdert_, fPd)  # no intra_Pp_ in spliced Pps
        Ppm_t.append(Ppm_)

    Ppd_t = []  # [LPpd_, IPpd_, DPpd_, MPpd_]
    rdn_t = sum_rdn_(param_names, Pdert_t, fPd=1)
    for param_name, Pdert_, rdn_ in zip(param_names, Pdert_t, rdn_t):
        Ppd_ = form_Pp_(Pdert_, fPd=0)  # segment Pdert__ into Ppds, no splice eval: rdn to more M-specific Ppm_
        if param_name=="D_": Ppd_ = intra_Pp_(None, Ppd_, Pdert_, fPd)
        Ppm_t.append(Ppd_)

    return [Ppm_t, Ppd_t]  # Pp_tt


def cross_comp(P_, fPd):  # cross-compare patterns within horizontal line

    Ldert_, Idert_, Ddert_, Mdert_, dert1_, dert2_ = [], [], [], [], [], []

    for _P, P, P2 in zip(P_, P_[1:], P_[2:] + [CP()]):  # for P_ cross-comp over step=1 and step=2
        _L, _I, _D, _M, *_ = _P.unpack()  # *_: skip remaining params
        L, I, D, M, *_ = P.unpack()
        D2, M2 = P2.D, P2.M

        Ldert_ += [comp_par(P, _L, L, "L_", ave_mL)] # div_comp L, sub_comp summed params:
        Idert_ += [comp_par(P, _I, I, "I_", ave_mI)]
        if fPd:
            Ddert = comp_par(P, _D, D2, "D_", ave_mD)  # step=2 for same-D-sign comp
            Ddert_ += [Ddert]
            dert2_ += [Ddert.copy()]
            dert1_ += [comp_par(P, _D, D, "D_", ave_mD)]  # to splice Pds
            Mdert_ += [comp_par(P, _M, M, "M_", ave_mM)]
        else:
            Ddert_ += [comp_par(P, _D, D, "D_", ave_mD)]
            Mdert = comp_par(P, _M, M2, "M_", ave_mM)  # step=2 for same-M-sign comp
            Mdert_ += [Mdert]
            dert2_ += [Mdert.copy()]
            dert1_ += [comp_par(P, _M, M, "M_", ave_mM)]  # to splice Pms
        _L, _I, _D, _M = L, I, D, M

    if not fPd: Mdert_ = Mdert_[:-1]  # remove CP() filled in P2

    return (Ldert_, Idert_, Ddert_, Mdert_), dert1_, dert2_[:-1]  # remove CP() filled in P2


def comp_par(P, _param, param, param_name, ave):

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

    return Cpdert(P=P, i=param, p=param + _param, d=d, m=m)


def form_Pp_(pdert_, fPd):
    # initialization:
    Pp_ = []
    x = 0
    _sign = None  # to initialize 1st P, (None != True) and (None != False) are both True

    for pdert in pdert_:  # segment by sign
        if fPd: sign = pdert.d > 0
        else:   sign = pdert.m > 0
        # adjust by ave projected at distance=negL and contrast=negM, if significant:
        # m + ddist_ave = ave - ave * (ave_rM * (1 + negL / ((param.L + _param.L) / 2))) / (1 + negM / ave_negM)?
        if sign != _sign:
            # sign change, compact terminated Pp, initialize Pp and append it to Pp_
            if Pp_:  # empty at 1st sign change
                Pp = Pp_[-1]  # no immediate normalization: Pp.I /= Pp.L; Pp.D /= Pp.L; Pp.M /= Pp.L; Pp.Rdn /= Pp.L
                Pp.Rdn += Pp.L  # redundancy to higher root Pp layers, not normalized
                # Pp vs Pdert_ redundancy:
                if fPd: V = abs(Pp.D)  # derived values
                else: V = Pp.M
                PpV = V / (Pp.L *.7)  #.7: ave intra-Pp-match coef, for value reduction with resolution, still non-negative
                pdert_V = V - Pp.L * ave_M * (ave_D * fPd)  # cost incr per pdert representations
                if PpV <= pdert_V:
                    Pp.Rdn += Pp.L; Pp.flay_rdn = 1  # Pp or pdert_ are layer-redundant, use in next-level eval

            Pp = CPp( L=1, I=pdert.p, D=pdert.d, M=pdert.m, Rdn = pdert.rdn+pdert.P.Rdn, x0=x, ix0=pdert.P.x0, pdert_=[pdert], sublayers=[[]])
            Pp_.append(Pp)  # Pp params will be updated by accumulation:
        else:
            # accumulate params:
            Pp.L += 1; Pp.I += pdert.p; Pp.D += pdert.d; Pp.M += pdert.m; Pp.Rdn += pdert.rdn+pdert.P.Rdn; Pp.pdert_ += [pdert]

        pdert.Ppt[fPd] = Pp  # Ppm | Ppd that pdert is in, replace root_Pp if any
        _sign = sign; x += 1

    return Pp_


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

    return Pdert_t


def splice_Ps(Ppm_, pdert1_, pdert2_, fPd):  # re-eval Pps, Pp.pdert_s for redundancy, eval splice Ps
    '''
    Initial P termination is by pixel-level sign change, but resulting separation may not be significant on a pattern level.
    That is, separating opposite-sign patterns are weak relative to separated same-sign patterns, especially if similar.
     '''
    for i, Pp in enumerate(Ppm_):
        if fPd: V = abs(Pp.D)  # DPpm_ if fPd, else IPpm_
        else: V = Pp.M  # add summed P.M|D?

        if V > ave_M * (ave_D*fPd) * Pp.Rdn * 4 and Pp.L > 4:  # min internal xP.I|D match in +Ppm
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
        '''
        no splice(): fine-grain eval per P triplet is too expensive?
        '''

def intra_Pp_(rootPp, Pp_, Pdert_, fPd):  # evaluate for sub-recursion in line Pm_, pack results into sub_Pm_
    '''
    each Pp may be compared over incremental range or derivation, as in line_patterns but with higher local ave
    '''
    comb_sublayers = []  # combine into root P sublayers[1:], each nested to depth = sublayers[n]

    for i, Pp in enumerate(Pp_):
        if Pp.L > 1:
            if fPd:
                if abs(Pp.D) + Pp.M > ave_M * ave_D * Pp.Rdn:  # + Pp.M: borrow potential, regardless of Rdn?
                    sub_search(Pp, True)  # search in top sublayer, eval by pdert.d
                    sub_Ppm_, sub_Ppd_ = [], []
                    Pp.sublayers = [[(sub_Ppm_, sub_Ppd_)]]
                    ddert_ = []
                    for _pdert, pdert in zip( Pp.pdert_[:-1], Pp.pdert_[1:]):
                        ddert_ += [comp_par(_pdert.P, _pdert.d, pdert.d, "D", ave)]  # higher derivation cross-comp of ds in dert1_, local aves?
                    sub_Ppm_[:] = form_Pp_(ddert_, fPd=True)  # cluster in ddert_
                    sub_Ppd_[:] = form_Pp_(ddert_, fPd=True)
                    if abs(Pp.D) > ave_M * ave_D * 4:  # diff induction per Pd_'DPpd_, add Pp.iD?
                        intra_Pp_(Pp, sub_Ppd_, None, fPd=True)  # recursive der+, no rng+: Pms are redundant?
                else:
                    Pp.sublayers += [[]]  # empty subset to preserve index in sublayer, or increment index of subset?
            else:
                # rng+ by more selective nearest match @ higher ave, in addition to extension by higher ave_negM in extra_Pp_
                if Pp.M > ave_M * Pp.Rdn:  # + Pp.iM?
                    sub_search(Pp, True)  # search in top sublayer, eval by pdert.d
                    sub_Ppm_, sub_Ppd_ = [], []
                    Pp.sublayers = [[(sub_Ppm_, sub_Ppd_)]]
                    rdert_ = search_Idert_(Pp, Pdert_, ave_mI, rave=1)  # comp x variable range, depending on I.M
                    sub_Ppm_[:] = form_Pp_rng(rdert_)  # cluster in rdert_
                    if Pp.M > ave_M * 4:  # intensity induction per Pm_'IPpm_, add Pp.iM?
                        intra_Pp_(Pp, sub_Ppd_, rdert_, fPd=False)  # recursive rng+, no der+: Pds are redundant?
                else:
                    Pp.sublayers += [[]]  # empty subset to preserve index in sublayer, or increment index of subset?

        if isinstance(rootPp, CPp) and Pp.sublayers[0]:
            comb_sublayers = [comb_subset_ + subset_ for comb_subset_, subset_ in
                              zip_longest(comb_sublayers, Pp.sublayers, fillvalue=[])
                             ]
    if isinstance(rootPp, CPp):
        rootPp.sublayers += comb_sublayers
    else:
        return Pp_  # each Pp has new sublayers, comb_sublayers is not needed


def search_Idert_(Pp, Idert_, ave_d, rave):  # extended variable-range search for core I at local ave: lower m and term by match

    rng_dert_ = Pp.pdert_  # may be extended
    flmiss_ = [1 for _ in Pp.pdert_]  # ones the length of Pp.pdert_
    for i, idert in enumerate(Pp.pdert_):  # overlapping pderts & +Pps, if pdert.m>0: negL is intra-Pp reference, no -Pps

        comb_M = idert.m
        j = i + Pp.x0 + 1
        # search right:
        while comb_M > 0 and j+1 < len(Idert_):  # from step=2, may continue outside Pp.pdert_, no merge: Pps overlap?
            j += 1
            rdert = Idert_[j]  # right_dert, extend search beyond next param
            idert.p = rdert.i + idert.i
            idert.d = rdert.i - idert.i  # difference
            idert.m = ave_d - abs(idert.d)  # indirect match
            if idert.m > 0:
                if j >= len(Pp.pdert_):  # rdert is outside Pp
                    rng_dert_.append(rdert); flmiss_.append(0)
                else:
                    flmiss_[j - Pp.x0] = 0  # =index of rdert in rng_dert_
                break  # 1st matching param takes over connectivity search from _param, in the next loop
            else:
                idert.negL += 1
                idert.negM += idert.m
                comb_M = idert.m - ave_M
                if j >= len(Pp.pdert_):  # missing searched derts added for form_Pp_rng to represent discontinuity
                    rng_dert_.append(rdert)
                    flmiss_.append(1)
        # search left:
        if flmiss_[i]:  # if idert was not matched as rdert in search right, else it's replaced as idert by left-matching dert
            comb_M = idert.m
            j = i + Pp.x0 - 1
            while comb_M > 0 and j-1 >= 0:  # from step=2, may continue outside Pp.pdert_, no merge: Pps overlap?
                j -= 1
                ldert = Idert_[j]  # left dert
                ldert.p = idert.p + ldert.i
                ldert.d = idert.i - ldert.i  # difference
                ldert.m = ave_d - abs(ldert.d)  # indirect match
                if ldert.m > 0:
                    if j < Pp.x0:  # ldert is outside Pp
                        rng_dert_.insert(0, ldert); flmiss_.insert(0, 0)
                    else:
                        flmiss_[j - Pp.x0] = 0  # =index of ldert in rng_dert_
                    break  # 1st matching dert takes over connectivity search in the next loop
                else:
                    ldert.negL += 1
                    ldert.negM += ldert.m
                    comb_M = ldert.m - ave_M
                    if j < Pp.x0:  # missing searched derts added for form_Pp_rng to represent discontinuity
                        rng_dert_.insert(0, ldert); flmiss_.insert(0, 1)

    return rng_dert_  # no return for flmiss_?

def form_Pp_rng(rdert_):  # rng_derts -> Ppms only, still a draft

    for pdert in rdert_: pdert.Ppt[0]=[]  # clear Ppms to replace with rng_Ppms
    Pp_ = []
    x = 0
    for i, _rdert in enumerate(rdert_):  # form +Pp from +rderts
        if not _rdert.Ppt[0]:  # _rdert is not in any rng_Pp formed in prior loops, else skip all:

            Pp = CPp(L=1, I=_rdert.p, D=_rdert.d, M=_rdert.m, Rdn=_rdert.rdn+_rdert.P.Rdn, x0=x, rdert_=[_rdert], sublayers=[[]])
            cm = 1  # initialize current m to start the loop
            j = i + 1 + _rdert.negL
            while cm > 0 and j < len(rdert_):
                rdert = rdert_[j]
                # accumulate params:
                Pp.L += 1; Pp.I += rdert.p; Pp.D += rdert.d; Pp.M += rdert.m; Pp.Rdn += rdert.rdn+rdert.P.Rdn+Pp.L  # rdn to root layers, not normalized
                Pp.pdert_ += [rdert]
                rdert.Ppt[0].append(Pp)  # Pps may overlap, skip as _rdert in the next loop
                cm = rdert.m
                if cm > 0:
                    Pp.negL += _rdert.negL; Pp.negM += _rdert.negM
                j += 1 + rdert.negL

            Pp_.append(Pp)
    return Pp_

# obsolete:

def extra_Pp_(Pp_, Idert_, ave, rave):  # incremental-range search for core I for +IPpms in Idert_: maps to adj IPpms

    # higher local ave for extended rng -> lower m and term by match, higher proj_M?
    Rdn, AddM = 0, 0
    for Pp in Pp_:
        addM = 0
        iM = sum([pdert.P.M for pdert in Pp.pdert_])  # not precomputed because it's very selective
        if Pp.M + iM > ave_M * Pp.Rdn * -4:  # rng_coef, this is fixed costs, add variable costs? -lend to contrast?
            # search left:
            i = Pp.x0  # Idert mapped to 1st pdert
            j = i - Pp._negL - 1  # 1st-left not-compared Idert
            if j > 0:
                addM = search_Pdert_(Pp_, Pp, Idert_, i, j, ave, rave, fleft=True)
                if Pp not in Pp_:  # if merged, Pp is removed from Pp_
                    Pp = Pp.pdert_[0].Ppt[0]  # merging Pp
            # search right:
            i = Pp.x0 + Pp.L - 1  # Idert mapped to last pdert
            j = i + Pp.pdert_[-1].negL + 1  # 1st-right not-compared Idert
            if j < len(Idert_):
                addM += search_Pdert_(Pp_, Pp, Idert_, i, j, ave, rave, fleft=False)
            # else: next Pp can search left but not right
        Rdn += Pp.Rdn
        AddM += addM
    # recursion:
    if AddM > ave_M * (Rdn / len(Pp_)) * 4:  # extra_Pp_ coef
        extra_Pp_(Pp_, Idert_, ave, rave+10)

    # no return Pp_: changed in place

def search_Pdert_(Pp_, iPp, Idert_, i, j, ave, rave, fleft):

    iIdert = Idert_[i]; iI = iIdert.i; iD = iIdert.d  # Idert searches Idert_ left or right from j
    addM = 0  # added to Pp by search
    if fleft: negM = iPp._negM; ipI = iI + (iD / 2)  # back-project by _D, if P.M? negL is compensated by decay?
    else:   negM = iIdert.negM; ipI = iI - (iD / 2)  # forward-project by _D, if P.M?

    while(iIdert.P.M + iIdert.m + negM > ave_M) and ((not fleft and j < len(Idert_)) or (fleft and j >= 0)):

        cIdert = Idert_[j]  # c for current comparand, left OR right from iIdert
        cI = cIdert.i; cD = cIdert.d
        if fleft:
            Idert = iIdert; _Idert = cIdert  # rename by direction
            pI = ipI; _pI = cI - (cD / 2)  # forward-project by cD
        else:
            Idert = cIdert; _Idert = iIdert  # rename by direction
            _pI = ipI; pI = cI + (cD / 2)  # back-project by cD

        # comp(_Idert, Idert):
        _Idert.p = pI + _pI; _Idert.d = _pI - pI; _Idert.m = ave - abs(_Idert.d)
        curr_M = _Idert.m * rave + (iIdert.P.M + cIdert.P.M) / 2  # P.M is bilateral
        P = Idert.P; _P = _Idert.P

        if curr_M > ave_sub * P.Rdn and _P.sublayers[0] and P.sublayers[0]:
            _Idert.sub_M, _Idert.sub_D = comp_sublayers(_P, P, _Idert.m)
        # comb.match(_P, P):
        if curr_M + _Idert.sub_M > ave_M * P.Rdn * 4:
            cPp = cIdert.Ppt[0]  # Pp to merge if positive only:

            if cPp.M > 0:  # +Pp: match to any Idert in Pp.pdert_-> all consecutive matches
                if fleft: addM += iPp.M; merge(Pp_, cPp, iPp)  # merge iPp in cPp
                else:     addM += cPp.M; merge(Pp_, iPp, cPp)  # merge cPp in iPp
            else:
                # transfer cIdert from -cPp to +Pp, Idert.Ppt[0] is updated but cPp remains as defined in form_Pp_:
                if fleft:
                    iPp.x0 -= 1 + iPp._negL
                    iPp.L += 1 + iPp._negL  # or len(iPp.pdert_) += iPp.negL at termination?
                    iPp.negL += iPp._negL; iPp.negM += iPp._negM  # summed from all pderts
                    iPp._negL, iPp._negM = 0, 0
                    Idert.P = P; Idert.i = P.I  # pderts represent initial P and i: the last on the left
                    addM += _Idert.m
                    _Idert.Ppt[0] = iPp
                    iPp.pdert_.insert(0, _Idert)  # appendleft
                else:
                    iPp.L += 1 + iPp.pdert_[-1].negL  # last pdert negL, negM are not yet in Pp
                    iPp.negL += iPp.pdert_[-1].negL; iPp.negM += iPp.pdert_[-1].negM  # summed from all pderts
                    addM += Idert.m
                    Idert.Ppt[0] = iPp
                    iPp.pdert_.append(Idert)
                iPp.iL += cIdert.P.L; iPp.I += cIdert.i; iPp.D += cIdert.d; iPp.M += cIdert.m; iPp.Rdn += cIdert.rdn
            break  # matching pdert or merged Pp takes over connectivity search in the next extra_Pp_

        else:  # Iderts miss
            if fleft:
                iPp._negL += 1; iPp._negM += _Idert.m
                j -= 1
            else:
                last_pdert = iPp.pdert_[-1]
                last_pdert.negM += curr_M - ave_M  # known to be negative, accum per dert
                last_pdert.negiL += P.L
                last_pdert.negL += 1
                negM = last_pdert.negM
                j += 1

    return addM

def merge(Pp_, _Pp, Pp):  # merge Pp with dert.Pp, if any:
    '''
    direction-specific:
    '''
    _Pp.accum_from(Pp, excluded=['x0'])
    # merge pderts
    for pdert in Pp.pdert_:
        _Pp.pdert_.append(pdert)
        pdert.Ppt[0] = _Pp  # pdert.Ppm
    # previously summed from all pderts except for the last negative one:
    _Pp.negL += Pp._negL  # left, right: _Pp.negL += _Pp.pdert_[-1].negL?
    _Pp.negM += Pp._negM  # left, right: _Pp.negM += _Pp.pdert_[-1].negM?
    _Pp.L = len(_Pp.pdert_) + _Pp.negL  #?

    # merge sublayers
    _Pp.sublayers += Pp.sublayers
    Pp_.remove(Pp)


def sub_search(rootPp, fPd):  # search inside top sublayer per P / sub_P, after P_ search: top-down induction,
    # called from intra_Pp_, especially IPp: init select by P.M, combined in Pp match?

    for pdert in rootPp.pdert_:
        P = pdert.P
        if P.sublayers[0]:  # not empty sublayer
            subset = P.sublayers[0][0]  # single top sublayer subset
            for i, sub_P_ in enumerate([subset[2], subset[3]]):  # sub_Pm_, sub_Pd_
                '''
                for sub_P_ in sub_P_t:  # not sure about this
                    if len(sub_P_) > 2: sub_P_ = splice(sub_P_, fPd)  # for discontinuous search?
                '''
                if (fPd and abs(P.D) > ave_D * rootPp.Rdn) or (P.M > ave_M * rootPp.Rdn):  # or if P.M + pdert.m + sublayer.Dert.M
                    if len(sub_P_) > 1: # at least 2 comparands in sub_P_?
                        sub_Pdert_t, dert1_, dert2_ = cross_comp(sub_P_, fPd=i)
                        sub_Ppm_t = form_Pp_root(sub_Pdert_t, dert1_, dert2_, fPd=False)
                        sub_Ppd_t = form_Pp_root(sub_Pdert_t, dert1_, dert2_, fPd=True)
                        subset[6].append(sub_Ppm_t)
                        subset[7].append(sub_Ppd_t)
                        #  deeper sublayers search is selective per sub_P


def comp_sublayers(_P, P, root_v):  # if pdert.m -> if summed params m -> if positional m: mx0?
    # replace root_v with root_vt: separate evaluations?

    sub_M, sub_D = 0, 0
    xDert_vt = form_comp_Derts(_P, P, root_v)  # value tuple: separate vm, vd for comp sub_mP_ or sub_dP_
    # comp sub_Ps between 1st sublayers of _P and P:
    _rdn, _rng = _P.sublayers[0][0][:2]
    rdn, rng  = P.sublayers[0][0][:2]  # 2nd [0] is a sole subset: rdn, rng, sub_Pm_, sub_Pd_, xsub_pmdertt_, xsub_pddertt_

    for i, xDert_v in enumerate(xDert_vt):  # eval m, d:
        if i: sub_D += xDert_v
        else: sub_M += xDert_v
        if xDert_v + root_v > ave_M * (i * ave_D) * rdn:  # ave_D is coef
            _sub_P_, _, _xsub_pdertt_ = _P.sublayers[0][0][2+i: 4+i+1]
            sub_P_, _, xsub_pdertt_ = P.sublayers[0][0][2+i: 4+i+1]
            if rng == _rng and min(_P.L, P.L) > ave_Ls:
                # if same intra_comp fork: compare sub_Ps to each _sub_P within max relative distance, comb_V- proportional:
                _SL = SL = 0  # summed Ls
                start_index = next_index = 0  # index of starting sub_P for current _sub_P
                _xsub_pdertt_ += [[]]  # array of cross-sub_P pdert tuples, inner brackets for subset_
                xsub_pdertt_ += [[]]  # append xsub_dertt per _sub_P_ and sub_P_, sparse?

                for _sub_P in _sub_P_:
                    _xsub_pdertt = [[], [], [], []]  # L, I, D, M xsub_pdert_s
                    _SL += _sub_P.L  # ix0 of next _sub_P
                    # search right:
                    for sub_P in sub_P_[start_index:]:  # index_ix0 > _ix0, comp sub_Ps at proximate relative positions in sub_P_
                        fbreak, xsub_P_M, xsub_P_D = comp_sub_P(_sub_P, sub_P, _xsub_pdertt, root_v, i)
                        sub_M += xsub_P_M ; sub_D += xsub_P_D
                        if fbreak:
                            break
                        # if next ix overlap: ix0 of next _sub_P < ix0 of current sub_P
                        if SL < _SL: next_index += 1
                        SL += sub_P.L  # ix0 of next sub_P
                    # search left:
                    for sub_P in reversed(sub_P_[len(sub_P_) - start_index:]):  # index_ix0 <= _ix0
                        fbreak, xsub_P_M, xsub_P_D = comp_sub_P(_sub_P, sub_P, _xsub_pdertt, root_v, i)  # invert sub_P, _sub_P positions
                        sub_M += xsub_P_M ; sub_D += xsub_P_D
                        if fbreak:
                            break
                    # not implemented: if param_name == "I_" and not fPd: sub_pdert = search_param_(param_)
                    start_index = next_index  # for next _sub_P

                    if _xsub_pdertt[0]:  # at least 1 sub_pdert, real min length ~ 8, very unlikely
                        # form 4-tuple of xsub_Pp_s:
                        xsub_Pp_t = form_Pp_root(_xsub_pdertt, [], [], fPd=i)
                        _xsub_pdertt_[-1][:] = xsub_Pp_t
                        xsub_pdertt_[-1][:] = _xsub_pdertt_[-1]  # bilateral assignment
                    else:
                        _xsub_pdertt_[-1].append(_xsub_pdertt)  # preserve nesting

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
        for i, (_iDert, iDert) in enumerate( zip((_mDert, _dDert), (mDert, dDert))):
            derDert = []
            for _param, param, param_name, ave in zip(_iDert, iDert, param_names, aves):
                dert = comp_par(_P, _param, param, param_name, ave)
                if i:  # =fPd  # higher value for L?
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

def splice(P_, fPd):  # currently not used, replaced by compact() in line_PPs
    '''
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