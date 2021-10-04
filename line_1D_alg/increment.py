'''
cross-level (xlevel) increment is supposed to recursively generate next-level code from current-level code:
next_level_code = cross_level_increment_code (current_level_code).
.
1st increment only converts line_Ps into line_PPs: line_PPs_code = 1st_increment_code (line_Ps_code).
It's a test for generic xlevel increment, due to original input formatting most increments won't be recursive
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
We will then try to convert it into fully recursive xlevel_increment:
'''
ave_mL = 5  # needs to be tuned
ave_mI = 5  # needs to be tuned
ave_mD = 1  # needs to be tuned
ave_mM = 5  # needs to be tuned

param_names = ["L_", "I_", "D_", "M_"]
aves = [ave_mL, ave_mI, ave_mD, ave_mM]


def search_root_incr(search_root):
    # convert line_PPs search_root into line_PPPs search_root
    pass

def search_root(root_P_t, feedback, elevation=1):
    # each P_ in P_t is FIFO
    # input is P_ tuple, starting from Pm_, Pd_, then nested to the depth = elevation (level counter)
    # elevation and feedback are not used in non-recursive 2nd increment

    for i, P_t in enumerate(root_P_t):
        for P_, param_name, ave in zip(P_t, param_names, aves):
            search(P_, i, param_name, ave)
            # currently in search_pattern(), need to update
            # in 2nd increment: 1st layer i = fPd, 2nd layer j = L | I | D | M
    pass



