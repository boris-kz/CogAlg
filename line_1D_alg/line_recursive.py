'''
3rd-level operations forming Ppps in Ppp_ttttt (5-level nested tuple of arrays of output patterns: 1 + 2 * 2(elevation-1)),
and cross-level recursion in level_recursion, forming Pps (param patterns) of incremental scope and depth,
in output P_T of depth = 1 + 2 * elevation-1 (last T denotes nested tuple of unknown depth)
'''

from collections import deque
from line_Ps import *
from line_PPs import *
from itertools import zip_longest
import math

class CderPp(ClusterStructure):  # should not be different from derp? PPP comb x Pps?
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

'''
    Conventions:
    postfix 't' denotes tuple, multiple ts is a nested tuple, 'T' is a nested tuple of unknown depth
    (usually the nesting is implicit, actual structure is flat list)
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    capitalized variables are normally summed small-case variables
'''

def line_recursive(p_):  # redundant to main in line_Ps

    P_t = line_Ps_root(p_)
    root = line_PPs_root(P_t)
    types_ = []
    for i in range(16):  # len(root.sublayers[0]
        types = [i%2, int(i%8 / 2), int(i/8) % 2]  # 2nd level output types: fPpd, param, fPd
        types_.append(types)

    return line_level_root(root, types_)


def line_level_root(root, types_):  # recursively adds higher levels of pattern composition and derivation
    '''
    Specific outputs: P_t = line_Ps_root(), Pp_ttt = line_PPs_root(), Ppp_ttttt = line_PPPs_root()
    if pipeline: output per P termination, append till min iP_ len, concatenate across frames
    '''
    sublayer0 = root.levels[-1][0]  # input is 1st sublayer of the last level
    new_sublayer0 = []  # 1st sublayer: (Pm_, Pd_( Lmd, Imd, Dmd, Mmd ( Ppm_, Ppd_))), deep sublayers: Ppm_(Ppmm_), Ppd_(Ppdm_,Ppdd_)
    root.sublayers = [new_sublayer0]  # will become new level, reset from last-level sublayers

    nextended = 0  # number of extended-depth P_s
    new_types_ = []
    new_M = 0
    '''
    - unpack and decode input: implicit tuple of P_s, nested to depth = 1 + 2*(elevation-1): 2Le: 2 P_s, 3Le: 16 P_s, 4Le: 128 P_s..
    - cross-comp and clustering of same-type P params: core params of new Pps
    '''
    for P_, types in zip(sublayer0, types_):

        if len(P_) > 2 and sum([P.M for P_ in sublayer0 for P in P_]) > ave_M:  # 2: min aveN, will be higher
            nextended += 1  # nesting depth of this P_ will be extended
            fiPd = types[0]  # or not just the last one, OR all fPds in types to switch to direct match?

            derp_t, dert1_, dert2_ = cross_comp(P_, fiPd)  # derp_t: Lderp_, Iderp_, Dderp_, Mderp_, same as in line_PPs
            sum_rdn_(param_names, derp_t, fiPd)  # sum cross-param redundancy per derp
            for param, derp_ in enumerate(derp_t):  # derp_ -> Pps:

                for fPd in 0, 1:  # 0-> Ppm_, 1-> Ppd_:
                    new_types = types.copy()
                    new_types.insert(0, param)  # add param index
                    new_types.insert(0, fPd)  # add fPd
                    new_types_.append(new_types)
                    Pp_ = form_Pp_(deepcopy(derp_), fPd)
                    new_sublayer0 += [Pp_]  # Ppm_| Ppd_
                    if (fPd and param == 2) or (not fPd and param == 1):  # 2: "D_", 1: "I_"
                        if not fPd:
                            splice_Ps(Pp_, dert1_, dert2_, fiPd, fPd)  # splice eval by Pp.M in Ppm_, for Pms in +IPpms or Pds in +DPpm
                        range_incr(root, Pp_, hlayers=1, rng=2)  # evaluate greater-range cross-comp and clustering per Pp
                        deriv_incr(root, Pp_, hlayers=1)  # evaluate higher-derivation cross-comp and clustering per Pp
                    new_M += sum([Pp.M for Pp in Pp_])  # Pp.M includes rng+ and der+ Ms
        else:
            new_types_ += [[] for _ in range(8)]  # align indexing with sublayer, replace with count of missing prior P_s, or use nested tuples?
            new_sublayer0 += [[] for _ in range(8)]

    if len(sublayer0) / max(nextended,1) < 4 and new_M > ave_M * 4:  # ave_extend_ratio and added M, will be default if pipelined
        # cross_core_comp(new_sublayer0, new_types_)
        root.levels.append(root.sublayers)  # levels represent all lower hierarchy

        if len(sublayer0) / max(nextended,1) < 8 and new_M > ave_M * 8:  # higher thresholds for recursion than for cross_core_comp?
            line_level_root(root, new_types_)  # try to add new level

    norm_feedback(root.levels)  # +dfilters: adjust all independent filters on lower levels, for pipelined version only

# functions below are not used:

def cross_core_comp(iP_T, types_):  # currently not used because:
    # correlation is predetermined by derivation: rdn coefs, multiplied across derivation hierarchy, no need to compare?
    '''
    compare same-type new params across different-type input Pp_s, separate from convertable dimensions|modalities: filter patterns
    if increasing correlation between higher derivatives, of pattern-summed params,
    similar to rng+, if >3 nesting levels in iP_T: root_depth - comparand_depth >3, which maps to the distance of >16 Pp_s?
    '''
    xPp_t_ = []  # each element is from one elevation of nesting
    ntypes = 1 + 2 * math.log(len(iP_T) / 2, 8)  # number of types per P_ in iP_T, with (fPd, param_name) n_pairs = math.log(len(iP_T)/2, 8)

    for elevation in range(int(ntypes)):  # each loop is an elevation of nesting
        if elevation % 2:  # params
            LP_t, IP_t, DP_t, MP_t = [], [], [], []
            # get P_ of each param for current elevation (compare at each elevation?)
            for i, types in enumerate(types_):
                if types:  # else empty set
                    if types[elevation] == 0:
                        LP_t += [iP_T[i]]
                    elif types[elevation] == 1:
                        IP_t += [iP_T[i]]
                    elif types[elevation] == 2:
                        DP_t += [iP_T[i]]
                    elif types[elevation] == 3:
                        MP_t += [iP_T[i]]
            P_tt = [LP_t, IP_t, DP_t, MP_t]

            xPp_t = [] # cross compare between 4 params, always = 8 elements if call from root function
            for j, _P_t in enumerate(P_tt):
                if j+1 < 4:  # 4 params
                    for P_t in P_tt[j+1:]:
                        xPp_ = []
                        for _P_ in _P_t:
                            for P_ in P_t:
                                if _P_ and P_:  # not empty _P_ and P_
                                    if len(P_)>2 and len(_P_)>2:
                                        _M = sum([_P.M for _P in _P_])
                                        M = sum([P.M for P in P_])
                                        for i,(param_name, ave) in enumerate(zip(param_names, aves)):
                                            for fPd in 0,1:
                                                xderp_ = []  # contains result from each _P_ and P_ pair
                                                for _P in _P_:
                                                    for P in P_:
                                                        # probably wrong but we need this evaluation, add in PM for evaluation?
                                                        if _P.M + P.M + _M + M > (_P.Rdn + P.Rdn) * ave:
                                                            _param = getattr(_P,param_name[0])
                                                            param = getattr(P,param_name[0])
                                                            xderp = comp_par(_P, _param, param, param_name, ave)
                                                            xderp_.append(xderp)
                                                xPp_ += form_Pp_(xderp_, fPd)  # add a loop to form xPp_ with fPd = 0 and fPd = 1? and intra_Pp?
                        xPp_t.append(xPp_)
            xPp_t_.append(xPp_t)


def norm_feedback(levels):
    # adjust all independent filters on lower levels by corresponding mean deviations (Ms), for pipelined version only
    pass

def P_type_assign(iP_T):  # P_T_: 2P_, 16P_, 128P_., each level is nested to the depth = 1 + 2*elevation

    ntypes = 1 + 2 * math.log(len(iP_T) / 2, 8)  # number of types per P_ in iP_T, with (fPd, param_name) n_pairs = math.log(len(iP_T)/2, 8)
    types_ = []  # parallel to P_T, for zipping

    for i, iP_ in enumerate(iP_T):  # last-level-wide comp_form_P__

        types = []  # list of fPds and names of len = ntypes
        step = len(iP_T) / 2  # implicit nesting, top-down
        nsteps = 1
        while (len(types) < ntypes):  # decode unique set of alternating types per P_: [fPd,name,fPd,name..], from index in iP_T:
            if len(types) % 2:
                types.insert(0, int(i % step / (step / 4)))  # add name index: 0|1|2|3
            else:
                types.insert(0, int((i / step)) % 2)  # add fPd: 0|1. This is the 1st increment because len(types) starts from 0
            nsteps += 1
            if nsteps % 2:
                step /= 8
            ''' Level 1 
            types.append( int((i/8))  % 2 )     # fPd
            types.append( int( i%8 / 2 ))       # param
            types.append( int((i/1))  % 2 )     # fPd
                Level 2
            types.append( int((i/64)) % 2 )     # fPd
            types.append( int( i%64/16 ))       # param 
            types.append( int((i/8))  % 2 )     # fPd    
            types.append( int( i%8/2 ))         # param
            types.append( int((i/1))  % 2 )     # fPd

            bottom-up scheme:
            _step = 1  # n of indices per current level of type
            for i, iP_ in enumerate( iP_T ):  # last-level-wide comp_form_P__
                while( len(types) < ntypes):  # decode unique set of alternating types per P_: [fPd,name,fPd,name..], from index in iP_T:
                    if len(types) % 2:
                       step = _step*4  # add name index: 0|1|2|3
                    else:
                    step = _step*2  # add fPd: 0|1. This is the 1st increment because len(types) starts from 0
                types.append( int( (i % step) / _step))  # int to round down: type should not change within step
                _step = step
            '''
        types_.append(types)  # parallel to P_T, for zipping
    return types_, ntypes


def line_PPPs_root(root):  # test code only, some obsolete

    sublayer0 = []  # 1st sublayer: (Pm_, Pd_( Lmd, Imd, Dmd, Mmd ( Ppm_, Ppd_))), deep sublayers: Ppm_(Ppmm_), Ppd_(Ppdm_,Ppdd_)
    root.sublayers = [sublayer0]  # reset from last-level sublayers
    P_ttt = root.levels[-1][0]  # input is 1st sublayer of the last level, always P_ttt? Not really, it depends on the level
    elevation = len(root.levels)
    level_M = 0

    for fiPd, paramset in enumerate(P_ttt):
        for param_name, param_md in zip(param_names, paramset):
            for fiPpd, P_ in enumerate(param_md):  # fiPpd: Ppm_ or Ppd_

                if len(P_) > 2:  # aveN, actually will be higher
                    derp_t, dert1_, dert2_ = cross_comp_Pp_(P_, fiPpd)  # derp_t: Ldert_, Idert_, Ddert_, Mdert_
                    sum_rdn_(param_names, derp_t, fiPpd)  # sum cross-param redundancy per derp
                    paramset = []
                    for param_name, derp_ in zip(param_names, derp_t):  # derp_ -> Pps:
                        param_md = []
                        for fPpd in 0, 1:  # 0-> Ppm_, 1-> Ppd_:
                            Pp_ = form_Pp_(derp_, fPpd)
                            param_md += [Pp_]  # -> [Ppm_, Ppd_]
                            if (fPpd and param_name == "D_") or (not fPpd and param_name == "I_"):
                                if not fPpd:
                                    splice_Pps(Pp_, dert1_, dert2_, fiPpd, fPpd)  # splice eval by Pp.M in Ppm_, for Pms in +IPpms or Pds in +DPpm
                                intra_Pp_(root, param_md[fPpd], 1, fPpd)  # eval der+ or rng+ per Pp
                            level_M += sum([Pp.M for Pp in Pp_])
                        paramset += [param_md]  # -> [Lmd, Imd, Dmd, Mmd]
                    sublayer0 += [paramset]  # -> [Pm_, Pd_]
                else:
                    # additional brackets to preserve the whole index, else the next level output will not be correct since some of them are empty
                    sublayer0 += [[[[], []] for _ in range(4)]]  # empty paramset to preserve index in [Pm_, Pd_]
    # add nesting here
    root.levels.append(root.sublayers)

    if any(sublayer0) and level_M > ave_M:  # evaluate for next level recursively
        line_level_root(root)

    return root
