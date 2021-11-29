'''
cross-level (xlevel) increment is supposed to recursively generate next-level code from current-level code:
next_level_code = cross_level_increment_code (current_level_code).
.
1st increment would only convert line_Ps into line_PPs: line_PPs_code = 1st_increment_code (line_Ps_code).
It's an example of xlevel increment, but due to initial input formatting most increments won't be recursive
(initial inputs are filter-defined, vs. mostly comparison-defined for higher levels):
.
cross_comp_incr(cross_comp):
- replace input frame with FIFO P_
- replace input with P, feeding 3 cross-comps,
- and selective variable-range search_param, comp_sublayers
.
form_P_incr(form_P_):
- add form_Pp_root, form_Pp_rng
.
intra_P_incr(intra_P_):
- add comb_sublayers, comb_subDerts
.
range_comp_incr(range_comp):
- combine with deriv_comp, for comp_param only?
.
form_rval_P_(P_):
- pack sum_rdn_incr,
- add xlevel_rdn, compact
.
2nd increment converts line_PPs into line_PPs: line_PPPs_code = 2nd_increment_code (line_PPs_code).
We will then try to convert it into fully recursive xlevel_increment
Separate increment for each root function to accommodate greater input nesting and value variation:
'''

from line_patterns import *
from line_PPs import *
from itertools import zip_longest

def line_root(iP_t):  # generic for line_PPs_root, line_PPPs_root, etc, by evaluating iP_t nesting to unpack
    '''
    for i, P_t in enumerate(root_P_t):  # fPd = i
    for P_, param_name, ave in zip(P_t, param_names, aves):
        norm_feedback(P_, i)
        Pdert_t, dert1_, dert2_ = search(P_, i)
        rval_Pp_t, Pp_t = form_Pp_root( Pdert_t, dert1_, dert2_, i)
    '''
    pass

def search_incr(search):
    '''
    Increase maximal comparison depth, according to greater input pattern depth
    still specific to search_param(I): reinforced by M?
    Also increase max search range, according to max accumulated value:
    incr P eval depth?
    Add same-iP xparam search?
    '''
    pass

def form_Pp_incr(form_Pp_):
    '''
    Add composition level,
    Add redundancy according to that in the input?
    '''
    pass


def line_PPPs_simplified(Pp_ttt):  # higher-level input is nested to the depth = 1 + 2*elevation (level counter)?

    norm_feedback(Pp_ttt)  # before processing
    Ppp_ttttt = []  # add  4-tuple of Pp vars ( 2-tuple of Pppm, Pppd )

    for Pp_tt, fPd in zip(Pp_ttt, [0,1]):  # fPd: Pm_ | Pd_
        Ppp_tttt = []
        for Pp_t in Pp_tt:  # LPp_ | IPp_ | DPp_ | MPp_
            Ppp_ttt = []
            for Pp_, fPpd in zip(Pp_t, [0,1]):  # fPpd: Ppm_ | Ppd_
                Ppp_tt = []
                Ppdert_t, Ppdert1_, Ppdert2_ = cross_comp(Pp_, fPpd)
                sum_rdn_Pp(param_names, Ppdert_t, fPpd)  # sum cross-param redundancy per Ppdert
                for param_name, Ppdert_ in zip( param_names, Ppdert_t):
                    Ppp_t = []
                    for fPppd in 0, 1:  # 0: Pppm_, 1: Pppd_:
                        Ppp_ = form_Ppp_(Ppdert_, fPppd)
                        if (fPpd and param_name == "D_") or (not fPpd and param_name == "I_"):
                            if not fPppd:
                                splice_Ps(Ppp_, Ppdert1_, Ppdert2_, fPpd)  # splice eval by Ppp.M, for Ppms in +IPppms or Ppds in +DPppm
                            intra_Pp_(None, Pp_, Ppdert_, 1, fPppd)  # der+ or rng+
                        Ppp_t.append(Ppp_)  # Pppm_, Pppd_
                    Ppp_tt.append(Pp_t)   # LPpp_, IPpp_, DPpp_, MPpp_
                Ppp_ttt.append(Pp_tt)   # Ppm_, Ppd_
            Ppp_tttt.append(Ppp_ttt)  # LPp_, IPp_, DPp_, MPp_
        Ppp_ttttt.append(Ppp_tttt)  # Pm_, Pd_

    return Ppp_ttttt  # 5-level nested tuple of arrays per line:
    # (Pm_, Pd_( LPp_, IPp_, DPp_, MPp_( Ppm_, Ppd_( LPpp_, IPpp_, DPpp_, MPpp_( Pppm_, Pppd_)))))


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
                        Ppdert_t, Ppdert1_, Ppdert2_ = cross_comp_Pp(Pp_, fPpd)
                        sum_rdn_Pp(param_names, Ppdert_t, fPpd)
                        for param_name, Ppdert_ in zip(param_names, Ppdert_t):  # param_name: LPpp_ | IPpp_ | DPpp_ | MPpp_
                            Ppp_t = []
                            for fPppd in 0,1:  # fPppd 0: Pppm_, 1: Pppd_
                                if Ppdert_:
                                    Ppp_ = form_Ppp_(Ppdert_, fPppd)
                                    if (fPpd and param_name == "D_") or (not fPpd and param_name == "I_"):
                                        if not fPppd:
                                            splice_Pps(Ppp_, Ppdert1_, Ppdert2_, fPpd)  # splice eval by Pp.M in Ppm_, for Pms in +IPpms or Pds in +DPpm
                                        intra_Ppp_(None, Ppp_, Ppdert_, 1, fPppd)  # der+ or rng+
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
- start combining line_PPs_root and line_PPP_root into generic line_root, evaluating nesting in input P_t for unpacking and repacking,
- explore possible cross_comp between iP_s of different core params, to the extent that they don't anticorrelate
'''

def norm_feedback(Pp_ttt):
    # probably recursive norm_feedback here depends on the depth
    pass

def cross_comp_Pp(Pp_, fPpd):  # cross-compare patterns of params within horizontal line

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


def sum_rdn_Pp(param_names, Ppdert_t, fPd):
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
def form_Ppp_(Ppdert_, fPpd):

    Ppp_ = []
    x = 0
    _Ppdert = Ppdert_[0]
    if fPpd: _sign = _Ppdert.d > 0
    else:   _sign = _Ppdert.m > 0
    # init Ppp params:
    L=1; I=_Ppdert.p; D=_Ppdert.d; M=_Ppdert.m; Rdn=_Ppdert.rdn; x0=x; Ppdert_=[_Ppdert]

    for Ppdert in Ppdert_[1:]:  # segment by sign
        if fPpd: sign = Ppdert.d > 0
        else:   sign = Ppdert.m > 0
        # sign change, pack terminated Ppp, initialize new Ppp:
        if sign != _sign:
            term_Ppp( Ppp_, L, I, D, M, Rdn, x0, Ppdert_, fPpd)
            # re-init Pp params:
            L=1; I=Ppdert.p; D=Ppdert.d; M=Ppdert.m; Rdn=Ppdert.rdn; x0=x; Ppdert_=[Ppdert]
        else:  # accumulate params:
            L += 1; I += Ppdert.p; D += Ppdert.d; M += Ppdert.m; Rdn += Ppdert.rdn; Ppdert_ += [Ppdert]
        _sign = sign; x += 1

    term_Ppp( Ppp_, L, I, D, M, Rdn, x0, Ppdert_, fPpd)  # pack last Ppp

    return Ppp_


def term_Ppp(Ppp_, L, I, D, M, Rdn, x0, Ppdert_, fPpd):

    Ppp = CPp(L=L, I=I, D=D, M=M, Rdn=Rdn+L, x0=x0, pdert_=Ppdert_, sublayers=[[]])
    for Ppdert in Ppp.pdert_: Ppdert.Ppt[fPpd] = Ppp  # root Ppp refs
    Ppp_.append(Ppp)


def intra_Ppp_(rootPpp, Ppp_, Ppdert_, hlayers, fPd):  # evaluate for sub-recursion
    pass

def splice_Pps(Pppm_, Ppdert1_, Ppdert2_, fPd):  # re-eval Ppps, pPp.pdert_s for redundancy, eval splice Pps
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