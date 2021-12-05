'''
3rd-level operations forming Ppps in Ppp_ttttt (5-level nested tuple of arrays of output patterns: 1 + 2 * 2(elevation-1)),
and cross-level recursion in comp_P_recursive, forming Pps (param patterns) of incremental scope and depth,
in output P_T of depth = 1 + 2 * elevation-1 (last T denotes nested tuple of unknown depth)
'''

from line_patterns import *
from line_PPs import *
from itertools import zip_longest


def line_recursive(p_):  # draft for level-recursive processing, starting with line_patterns

    oP_T = line_PPPs_start( line_PPs_root( line_Ps_root(p_)))
    # oPp_T = line_PPs_root(iP_T); oPpp_T = line_PPPs_root(oPp_T)
    # if pipeline: concatenate p_s across frames for indefinite recursion

    return oP_T


def line_PPPs_start(Pp_ttt):  # starts level-recursion, higher-level input is nested to the depth = 1 + 2*elevation (level counter)

    norm_feedback(Pp_ttt)  # before processing
    Ppp_ttttt = []  # add 4-tuple of Pp vars ( 2-tuple of Pppm, Pppd ) to the input
    for Pp_tt, fPd in zip(Pp_ttt, [0, 1]):  # fPd: Pm_ | Pd_
        Ppp_tttt = []
        # cross_core_comp(Pp_tt)  # not needed on this level
        for param_name, Pp_t in zip(param_names, Pp_tt):  # LPp_ | IPp_ | DPp_ | MPp_
            Ppp_ttt = []
            if isinstance(Pp_t, list):  # Ppt is not P
                for Pp_, fPpd in zip(Pp_t, [0,1]):  # fPpd: Ppm_ | Ppd_
                    Ppp_tt = []
                    if len(Pp_) > 1:
                        comp_P_recursive(Pp_ttt, [], Pp_, [], fPpd)  # no iM_ttt and iM_ input yet
                        # Pp_[:] = Ppp_tt: LPpp(m_,d_), IPpp(m_,d_), DPpp(m_,d_), MPpp(m_,d_), deeper with recursion
                        '''
                        below is not needed, Pp_ttt nesting is extended in place?
                        '''
                    else: Ppp_tt.append([])  # keep index
                Ppp_ttt.append(Pp_tt)  # Ppm_, Ppd_
            else: Ppp_ttt = Pp_t       # Pp_t is P, actual nesting is variable
            Ppp_tttt.append(Ppp_ttt)   # LPp_, IPp_, DPp_, MPp_
        Ppp_ttttt.append(Ppp_tttt)     # Pm_, Pd_

    return Ppp_ttttt  # 5 or greater- level nested tuple of arrays per line:
    # (Pm_, Pd_( LPp_, IPp_, DPp_, MPp_( Ppm_, Ppd_( LPpp_, IPpp_, DPpp_, MPpp_( Pppm_, Pppd_ )))))


def comp_P_recursive(iP_T, iM_T, iP_, iM_, fPd):  # cross_comp_Pp_, sum_rdn, splice, intra, comp_P_recursive

    norm_feedback(iP_)
    Pdert_t, pdert1_, pdert2_ = cross_comp_Pp_(iP_, fPd)
    sum_rdn(param_names, Pdert_t, fPd)
    oP_tt, oM_tt = [], []  # Pp_tt or deeper if recursion, added per comp_P_recursive

    for param_name, Pdert_ in zip(param_names, Pdert_t):  # param_name: LPp_ | IPp_ | DPp_ | MPp_
        oP_t, oM_t = [], []  # Ppm, Ppd_
        for fPpd in 0, 1:  # 0: Ppm_, 1: Ppd_
            if Pdert_:
                oP_ = form_Pp_(Pdert_, fPpd)
                if (fPd and param_name == "D_") or (not fPd and param_name == "I_"):
                    if not fPpd:
                        splice_Ps(oP_, pdert1_, pdert2_, fPd)  # splice eval by Pp.M in Ppm_, for Pms in +IPpms or Pds in +DPpm
                    intra_Pp_(None, oP_, Pdert_, 1, fPpd)  # der+ | rng+
                oP_t.append(oP_)
                oM_t.append( [sum( [Pp.M for Pp in oP_] )] )  # to evaluate for cross_core_comp and recursion
            else:
                oP_t.append([]); oM_t.append([])  # preserve index
        oP_tt.append(oP_t); oM_tt.append(oM_t)
    iP_[:] = oP_tt; iM_[0] = oM_tt  # nesting added per comp_P_recursive, reflected in iP_T, iM_T; iM_ is single-element list for mutability?

    cross_core_comp(iP_T, iM_T)  # evaluate recursion with results of cross_core_comp

    for param_name, oP_t, oM_t in zip(param_names, oP_tt, oM_tt):  # param_name: LPpp_ | IPpp_ | DPpp_ | MPpp_
        for fPpd in 0, 1:
            oP_ = oP_t[fPpd]  # fPpd 0: Ppm_, 1: Ppd_
            oM = oM_t[fPpd][0]
            if (fPd and param_name == "D_") or (not fPd and param_name == "I_"):
                if len(oP_) > 4 and oM > ave_M * 4:  # 1st 4: ave_len_oP_, 2nd 4: recursion coef
                    comp_P_recursive(iP_T, iM_T, oP_, [oM], fPpd)  # oP nesting increases in recursion

    # return iP_T, iM_T  # for cross_core_comp only?


# draft, need further discussion
def cross_core_comp(oP_tt, oPM_tt):  # P_T, M_T

    xPp_ttt = [] # cross compare between 4 params, always = 6 elements
    for j, (_P_t, _PM_t) in enumerate(zip(oP_tt, oPM_tt)):
        if j+1 < 4: #
            for P_t, PM_t in zip(oP_tt[j+1:], oPM_tt[j+1:]):
                xPp_tt = [] # xPp between params
                for fPd in 0, 1:
                    _P_ = _P_t[fPd]
                    P_ = P_t[fPd]
                    xPp_t = []
                    for i,(param_name, ave) in enumerate(zip(param_names, aves)):
                        xpdert_ = []
                        for _P in _P_:
                            for P in P_:
                                # probably wrong but we need this evaluation, add in PM for evaluation?
                                if _P.M + P.M > (_P.Rdn + P.Rdn) * ave:
                                    _param = getattr(_P,param_name[0])
                                    param = getattr(P,param_name[0])
                                    xpdert = comp_par(_P, _param, param, param_name, ave)
                                    xpdert_.append(xpdert)
                        if len(xpdert_)>1:
                            xPp_ = form_Pp_(xpdert_, fPd)
                        else: xPp_ = []
                        xPp_t.append(xPp_)
                    xPp_tt.append(xPp_tt)
                xPp_ttt.append(xPp_tt)
        # eval by PM_s for cross_core_comp between P_s, need to compute one-to-one rdn?



def line_PPPs_root(Pp_ttt):  # higher-level input is nested to the depth = 2+elevation (level counter), or 2*elevation?

    norm_feedback(Pp_ttt)  # before processing
    Ppp_ttttt = []  # add 4-tuple of Pp vars ( 2-tuple of Pppm, Pppd )

    for Pp_tt, fPd in zip(Pp_ttt, [0, 1]):  # fPd: Pm_ | Pd_
        Ppp_tttt = []
        for Pp_t in Pp_tt:  # LPp_ | IPp_ | DPp_ | MPp_
            Ppp_ttt = []
            if isinstance(Pp_t, list):  # Pp_t is not a spliced P: if IPp_ only?
                Ppp_tt = []
                for Pp_, fPpd in zip(Pp_t, [0,1]):  # fPpd: Ppm_ | Ppd_
                    if len(Pp_)>1:
                        Ppdert_t, Ppdert1_, Ppdert2_ = cross_comp_Pp_(Pp_, fPpd)
                        sum_rdn(param_names, Ppdert_t, fPpd)
                        for param_name, Ppdert_ in zip(param_names, Ppdert_t):  # param_name: LPpp_ | IPpp_ | DPpp_ | MPpp_
                            Ppp_t = []
                            for fPppd in 0,1:  # fPppd 0: Pppm_, 1: Pppd_
                                if Ppdert_:
                                    Ppp_ = form_Pp_(Ppdert_, fPppd)
                                    if (fPpd and param_name == "D_") or (not fPpd and param_name == "I_"):
                                        if not fPppd:
                                            splice_Ps(Ppp_, Ppdert1_, Ppdert2_, fPpd)  # splice eval by Pp.M in Ppm_, for Pms in +IPpms or Pds in +DPpm
                                        intra_Pp_(None, Ppp_, Ppdert_, 1, fPppd)  # der+ or rng+
                                else: Ppp_ = []     # keep index
                                Ppp_t.append(Ppp_)  # Pppm_, Pppd_
                            Ppp_tt.append(Ppp_t)    # LPpp_, IPpp_, DPpp_, MPpp_
                    else: Ppp_tt = []
                Ppp_ttt.append(Ppp_tt)              # Ppm_, Ppd_
            else: Ppp_ttt = Pp_t  # spliced P
            Ppp_tttt.append(Ppp_ttt)                # LPp_, IPp_, DPp_, MPp_
        Ppp_ttttt.append(Ppp_tttt)                  # Pm_, Pd_

    return Ppp_ttttt  # 5-level nested tuple of arrays per line:
    # (Pm_, Pd_( LPp_, IPp_, DPp_, MPp_( Ppm_, Ppd_ ( LPpp_, IPpp_, DPpp_, MPpp_( Pppm_, Pppd_ )))))

''' 
Next: 
- check if line_PPs functions work with line_PPPs,
- explore possible cross_comp between iP_s of different core params, to the extent that they don't anticorrelate
'''

def norm_feedback(Pp_ttt):
    # probably recursive norm_feedback here depends on the depth
    pass

def cross_comp_Pp_(Pp_, fPpd):  # cross-compare patterns of params within horizontal line

    LPpdert_, IPpdert_, DPpdert_, MPpdert_, Ppdert1_, Ppdert2_ = [], [], [], [], [], []

    for _Pp, Pp, Pp2 in zip(Pp_, Pp_[1:], Pp_[2:] + [CPp()]):  # for P_ cross-comp over step=1 and step=2
        _L, _I, _D, _M = _Pp.L, _Pp.I, _Pp.D, _Pp.M
        L, I, D, M, = Pp.L, Pp.I, Pp.D, Pp.M
        D2, M2 = Pp2.D, Pp2.M

        LPpdert_ += [comp_par(_Pp, _L, L, "L_", ave_mL)]  # div_comp L, sub_comp summed params:
        IPpdert_ += [comp_par(_Pp, _I, I, "I_", ave_mI)]
        if fPpd:
            DPpdert = comp_par(_Pp, _D, D2, "D_", ave_mD)  # step=2 for same-D-sign comp?
            DPpdert_ += [DPpdert]
            Ppdert2_ += [DPpdert.copy()] # to splice Ppds
            Ppdert1_ += [comp_par(_Pp, _D, D, "D_", ave_mD)]  # to splice Pds
            MPpdert_ += [comp_par(_Pp, _M, M, "M_", ave_mM)]
        else:
            DPpdert_ += [comp_par(_Pp, _D, D, "D_", ave_mD)]
            MPpdert = comp_par(_Pp, _M, M2, "M_", ave_mM)  # step=2 for same-M-sign comp?
            MPpdert_ += [MPpdert]
            Ppdert2_ += [MPpdert.copy()]
            Ppdert1_ += [comp_par(_Pp, _M, M, "M_", ave_mM)]  # to splice Ppms

        _L, _I, _D, _M = L, I, D, M

    if not fPpd: MPpdert_ = MPpdert_[:-1]  # remove CPp() filled in P2

    return (LPpdert_, IPpdert_, DPpdert_, MPpdert_), Ppdert1_, Ppdert2_[:-1]  # remove CPp() filled in dert2


def sum_rdn(param_names, Ppdert_t, fPd):
    '''
    access same-index pderts of all Pp params, assign redundancy to lesser-magnitude m|d in param pair.
    if other-param same-Pp_-index pdert is missing, rdn doesn't change.
    '''
    if fPd: alt = 'M'
    else:   alt = 'D'
    name_pairs = (('I', 'L'), ('I', 'D'), ('I', 'M'), ('L', alt), ('D', 'M'))  # pairs of params redundant to each other
    # rdn_t = [[], [], [], []] is replaced with pdert.rdn

    for i, (LPpdert, IPpdert, DPpdert, MPpdert) in enumerate( zip_longest(Ppdert_t[0], Ppdert_t[1], Ppdert_t[2], Ppdert_t[3], fillvalue=Cpdert())):
        # pdert per _P in P_, 0: Ldert_, 1: Idert_, 2: Ddert_, 3: Mdert_
        # P M|D rdn + dert m|d rdn:
        rdn_pairs = [[fPd, 0], [fPd, 1-fPd], [fPd, fPd], [0, 1], [1-fPd, fPd]]  # rdn in olp Ps: if fPd: I, M rdn+=1, else: D rdn+=1
        # names:    ('I','L'), ('I','D'),    ('I','M'),  ('L',alt), ('D','M'))  # I.m + P.M: value is combined across P levels?

        for rdn_pair, name_pair in zip(rdn_pairs, name_pairs):
            # assign rdn in each rdn_pair using partial name substitution: https://www.w3schools.com/python/ref_func_eval.asp
            if fPd:
                if eval("abs(" + name_pair[0] + "Ppdert.d) > abs(" + name_pair[1] + "Ppdert.d)"):  # (param_name)dert.d|m
                    rdn_pair[1] += 1
                else: rdn_pair[0] += 1  # weaker pair rdn+1
            else:
                if eval(name_pair[0] + "Ppdert.m > " + name_pair[1] + "Ppdert.m"):
                    rdn_pair[1] += 1
                else: rdn_pair[0] += 1  # weaker pair rdn+1

        for j, param_name in enumerate(param_names):  # sum param rdn from all pairs it is in, flatten pair_names, pair_rdns?
            Rdn = 0
            for name_in_pair, rdn in zip(name_pairs, rdn_pairs):
                if param_name[0] == name_in_pair[0]:  # param_name = "L_", param_name[0] = "L"
                    Rdn += rdn[0]
                elif param_name[0] == name_in_pair[1]:
                    Rdn += rdn[1]

            if len(Ppdert_t[j]) >i:  # if fPd: Ddert_ is step=2, else: Mdert_ is step=2
                Ppdert_t[j][i].rdn = Rdn  # [Ldert_, Idert_, Ddert_, Mdert_]


def comp_par(_Pp, _param, param, param_name, ave):

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

    return Cpdert(P=_Pp, i=_param, p=param + _param, d=d, m=m)

# draft
def form_Pp_(root_Ppdert_, fPpd):

    Ppp_ = []
    x = 0
    _Ppdert = root_Ppdert_[0]
    if fPpd: _sign = _Ppdert.d > 0
    else:   _sign = _Ppdert.m > 0
    # init Ppp params:
    L=1; I=_Ppdert.p; D=_Ppdert.d; M=_Ppdert.m; Rdn=_Ppdert.rdn; x0=x; Ppdert_=[_Ppdert]

    for Ppdert in root_Ppdert_[1:]:  # segment by sign
        if fPpd: sign = Ppdert.d > 0
        else:   sign = Ppdert.m > 0
        # sign change, pack terminated Ppp, initialize new Ppp:
        if sign != _sign:
            term_Pp( Ppp_, L, I, D, M, Rdn, x0, Ppdert_, fPpd)
            # re-init Pp params:
            L=1; I=Ppdert.p; D=Ppdert.d; M=Ppdert.m; Rdn=Ppdert.rdn; x0=x; Ppdert_=[Ppdert]
        else:  # accumulate params:
            L += 1; I += Ppdert.p; D += Ppdert.d; M += Ppdert.m; Rdn += Ppdert.rdn; Ppdert_ += [Ppdert]
        _sign = sign; x += 1

    term_Pp( Ppp_, L, I, D, M, Rdn, x0, Ppdert_, fPpd)  # pack last Ppp

    return Ppp_


def term_Pp(Ppp_, L, I, D, M, Rdn, x0, Ppdert_, fPpd):

    Ppp = CPp(L=L, I=I, D=D, M=M, Rdn=Rdn+L, x0=x0, pdert_=Ppdert_, sublayers=[[]])
    for Ppdert in Ppp.pdert_: Ppdert.Ppt[fPpd] = Ppp  # root Ppp refs
    Ppp_.append(Ppp)


def intra_Pp_(rootPpp, Ppp_, Ppdert_, hlayers, fPd):  # evaluate for sub-recursion
    pass

def splice_Ps(Pppm_, Ppdert1_, Ppdert2_, fPd):  # re-eval Ppps, pPp.pdert_s for redundancy, eval splice Pps
    '''
    Initial P termination is by pixel-level sign change, but resulting separation may not be significant on a pattern level.
    That is, separating opposite-sign patterns are weak relative to separated same-sign patterns, especially if similar.
     '''
    for i, Ppp in enumerate(Pppm_):
        if fPd: V = abs(Ppp.D)  # DPpm_ if fPd, else IPpm_
        else: V = Ppp.M  # add summed P.M|D?

        if V > ave_M * (ave_D*fPd) * Ppp.Rdn * 4 and Ppp.L > 4:  # min internal xP.I|D match in +Ppm
            M2 = M1 = 0
            for Ppdert2 in Ppdert2_: M2 += Ppdert2.m  # match(I, __I or D, __D): step=2
            for Ppdert1 in Ppdert1_: M1 += Ppdert1.m  # match(I, _I or D, _D): step=1

            if M2 / max( abs(M1), 1) > ave_splice:  # similarity / separation(!/0): splice Ps in Pp, also implies weak Pp.pdert_?
                # replace Pp params with summed P params, Pp is now primarily a spliced P:
                Ppp.L = sum([Ppdert.P.L for Ppdert in Ppp.pdert_]) # In this case, Ppdert.P is Pp
                Ppp.I = sum([Ppdert.P.I for Ppdert in Ppp.pdert_])
                Ppp.D = sum([Ppdert.P.D for Ppdert in Ppp.pdert_])
                Ppp.M = sum([Ppdert.P.M for Ppdert in Ppp.pdert_])
                Ppp.Rdn = sum([Ppdert.P.Rdn for Ppdert in Ppp.pdert_])

                for Ppdert in Ppp.pdert_:
                    Ppp.dert_ += Ppdert.P.pdert_
        '''
        no splice(): fine-grain eval per P triplet is too expensive?
        '''