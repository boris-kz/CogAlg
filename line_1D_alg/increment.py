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


# draft
def line_PPP_root(Pp_ttt):  # higher-level input is nested to the depth = 2*elevation (level counter), or 2^elevation?

    Pp_ttttt = []  # add Pp vars tuple ) Pppm, Pppd tuple

    norm_feedback(P_t)  # before processing

    for fPd, P_ in enumerate(P_t):  # fPd: Pm_ or Pd_
        if len(P_) > 1:
            Pdert_t, dert1_, dert2_ = cross_comp(P_, fPd)  # Pdert_t: Ldert_, Idert_, Ddert_, Mdert_
            Pp_tt = []  # Ppm_t, Ppd_t, each: [LPp_, IPp_, DPp_, MPp_]

            for fPpd in 0, 1:  # 0-> Ppm_t, 1-> Ppd_t
                Pp_t = []  # [LPp_, IPp_, DPp_, MPp_]
                rdn_t = sum_rdn_(param_names, Pdert_t, fPd)
                # Pdert_-> Pps:
                for param_name, Pdert_, rdn_ in zip(param_names, Pdert_t, rdn_t):
                    Pp_ = form_Pp_(Pdert_, fPpd)
                    if (fPpd and param_name == "D_") or (not fPpd and param_name == "I_"):
                        if not fPpd:
                            splice_Ps(Pp_, dert1_, dert2_, fPd)  # splice eval by Pp.M in Ppm_, for Pms in +IPpms or Pds in +DPpm
                        Pp_ = intra_Pp_(None, Pp_, Pdert_, 1, fPpd)  # der+ or rng+
                    Pp_t.append(Pp_)
                Pp_tt.append(Pp_t)
            Pp_ttt.append(Pp_tt)
        else:
            Pp_ttt.append(P_)  # Pps are not formed

    return Pp_ttt  # 3-level nested tuple per line: Pm_, Pd_( Ppm_, Ppd_( LPp_, IPp_, DPp_, MPp_)))


# draft
def fom_PPP_(Pp_):
    # form PPP_ from Pp_ here
    pass

# draft
def form_PPP_recursive(Pp_ttt):
    Ppp_tttttt = form_PPP_root(Pp_ttt)

    if f_recursive:
        return form_PPP_recursive(PPP_ttt)
    else:
        return PPP_ttt



