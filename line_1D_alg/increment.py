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


def line_root_incr(line_PPs_root):
    # convert line_PPs_root into line_PPPs search_root by adding a layer of nesting to unpack:
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

def line_PPPs_draft(Pp_ttt):  # higher-level input is nested to the depth = 1 + 2*elevation (level counter)?
    '''
    draft, mostly to show nesting at this point. Need to add conditions, sum_rdn, splice, intra_Ppp_
    '''
    norm_feedback(Pp_ttt)  # before processing
    Ppp_ttttt = []  # add  4-tuple of Pp vars ) 2-tuple of Pppm, Pppd per var

    for Pp_tt, fPd in zip(Pp_ttt, [0,1]):  # fPd: Pm_ | Pd_
        Ppp_tttt = []  # Pppm_, Pppd_
        for Pp_t, fPpd in zip(Pp_tt, [0,1]):  # fPpd: Ppm_ | Ppd_
            Ppp_ttt = []
            for param_name, Pp_ in zip( param_names, Pp_t):  # param_name: LPp_ | IPp_ | DPp_ | MPp_
                Ppdert_t = cross_comp(Pp_, fPpd)
                Ppp_tt = []
                for fPpd in 0, 1:  # 0: Pppm_, 1: Pppd_
                    Ppp_t = []
                    for Ppdert_ in Ppdert_t:  # Lpdert_, Ipdert_, Dpdert_, Mpdert_
                        Ppp_ = form_Ppp_(Ppdert_, fPpd)
                        Ppp_t.append(Ppp_)  # LPpp_, IPpp_, DPpp_, MPpp_
                    Ppp_tt.append(Pp_t)   # Pppm_, Pppd_
                Ppp_ttt.append(Pp_tt)   # LPp_, IPp_, DPp_, MPp_
            Ppp_tttt.append(Ppp_ttt)  # Ppm_, Ppd_
        Ppp_ttttt.append(Ppp_tttt)  # Pm_, Pd_

    return Ppp_ttttt  # 5-level nested tuple of arrays per line:
    # (Pm_, Pd_( Ppm_, Ppd_( LPp_, IPp_, DPp_, MPp_ ( Pppm_, Pppd_ ( LPpp_, IPpp_, DPpp_, MPpp_)))))

# draft
def line_PPPs_root(Pp_ttt):  # higher-level input is nested to the depth = 2+elevation (level counter), or 2*elevation?

    norm_feedback(Pp_ttt)  # before processing
    Ppp_ttttt = []  # add  4-tuple of Pp vars ) 2-tuple of Pppm, Pppd per var

    for Pp_tt, fPd in zip(Pp_ttt, [0,1]):  # fPd: root Pm_ or root Pd_
        Ppp_tttt = []  # each element is L, I, D, M's Ppp_ of different fPpd

        for Pp_t, fPpd in zip(Pp_tt, [0,1]):  # fPpd: Ppm_ or Ppd_
            if isinstance(Pp_t, list):  # Ppt is not P
                Ppp_tt = []
                for Pp_ in Pp_t:  # LPp_, IPp_, DPp_, MPp_
                    Ppp_t = []
                    if len(Pp_)>1:
                        Ppdert_t = cross_comp(Pp_, fPpd)  # or it should be fPd here?
                        for Ppdert_ in Ppdert_t:  # L, I, D, M, Ppps
                            Ppp_ = form_PPP_(Ppdert_, fPpd)
                            Ppp_t.append(Ppp_)

                        Ppp_tt.append(Ppp_t)
                Ppp_ttt.append(Ppp_tt)

        Ppp_tttt.append(Ppp_ttt)

    return Ppp_ttttt  # 5-level nested tuple of arrays per line:
    # (Pm_, Pd_( Ppm_, Ppd_( LPp_, IPp_, DPp_, MPp_ (LPpp_, IPpp_, DPpp_, MPpp_))))



def norm_feedback(Pp_ttt):
    # probably recursive norm_feedback here depends on the depth
    pass

def cross_comp(Pp_, fPpd):  # cross-compare patterns of params within horizontal line

    LPpdert_, IPpdert_, DPpdert_, MPpdert_ = [], [], [], []

    for _Pp, Pp, Pp2 in zip(Pp_, Pp_[1:], Pp_[2:] + [CPp()]):  # for P_ cross-comp over step=1 and step=2
        _L, _I, _D, _M = _Pp.L, _Pp.I, _Pp.D, _Pp.M
        L, I, D, M, = Pp.L, Pp.I, Pp.D, Pp.M
        D2, M2 = Pp2.D, Pp2.M

        LPpdert_ += [comp_par(_Pp, _L, L, "L_", ave_mL)]  # div_comp L, sub_comp summed params:
        IPpdert_ += [comp_par(_Pp, _I, I, "I_", ave_mI)]
        if fPpd:
            DPpdert_ += [comp_par(_Pp, _D, D2, "D_", ave_mD)]  # step=2 for same-D-sign comp?
            MPpdert_ += [comp_par(_Pp, _M, M, "M_", ave_mM)]
        else:
            DPpdert_ += [comp_par(_Pp, _D, D, "D_", ave_mD)]
            MPpdert_ += [comp_par(_Pp, _M, M2, "M_", ave_mM)]  # step=2 for same-M-sign comp?
        _L, _I, _D, _M = L, I, D, M

    if not fPpd: MPpdert_ = MPpdert_[:-1]  # remove CPp() filled in P2

    return LPpdert_, IPpdert_, DPpdert_, MPpdert_




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

        if sign != _sign:  # sign change, pack terminated Ppp, initialize new Ppp
            Ppp_ = term_Ppp( Ppp_, L, I, D, M, Rdn, x0, Ppdert_, fPpd)
            # re-init Pp params:
            L=1; I=Ppdert.p; D=Ppdert.d; M=Ppdert.m; Rdn=Ppdert.rdn; x0=x; Ppdert_=[Ppdert]
        else:
            # accumulate params:
            L += 1; I += Ppdert.p; D += Ppdert.d; M += Ppdert.m; Rdn += Ppdert.rdn; Ppdert_ += [Ppdert]
        _sign = sign; x += 1

    Ppp_ = term_Ppp( Ppp_, L, I, D, M, Rdn, x0, Ppdert_, fPpd)  # pack last Ppp

    return Ppp_


def term_Ppp(Ppp_, L, I, D, M, Rdn, x0, Ppdert_, fPpd):

    Ppp = CPp(L=L, I=I, D=D, M=M, Rdn=Rdn+L, x0=x0, pdert_=Ppdert_, sublayers=[[]])
    for Ppdert in Ppp.pdert_: Ppdert.Ppt[fPpd] = Ppp  # root Ppp refs
    Ppp_.append(Ppp)


# draft
def form_PPP_recursive(Pp_ttt):
    Ppp_tttttt = form_PPP_root(Pp_ttt)

    if f_recursive:
        return form_PPP_recursive(PPP_ttt)
    else:
        return PPP_ttt

