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

class CderPp(ClusterStructure):  # if different, PPP comb x Pps?
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

def line_recursive(p_):
    '''
    Specific outputs: P_t = line_Ps_root(), Pp_ttt = line_PPs_root(), Ppp_ttttt = line_PPPs_root()
    if pipeline: output per P termination, append till min iP_ len, concatenate across frames
    '''
    P_t = line_Ps_root(p_)
    root = line_PPs_root(P_t)
    return line_level_root(root)


def line_level_root(root):

    sublayer0 = []  # 1st sublayer: (Pm_, Pd_( Lmd, Imd, Dmd, Mmd ( Ppm_, Ppd_))), deep sublayers: Ppm_(Ppmm_), Ppd_(Ppdm_,Ppdd_)
    root.sublayers=[sublayer0]  # reset from last-level sublayers
    P_ttt = root.levels[-1][0]  # input is 1st sublayer of the last level, always P_ttt?

    for fiPd, pramset in enumerate(P_ttt):  # fiPd: Pm_ or Pd_
        for param_name, param_md in zip(param_names, pramset):
            for fiPpd, P_ in enumerate(param_md):  # fiPpd: Ppm_ or Ppd_
                # below is not fully revised:
                if len(P_) > 1:
                    Pdert_t, dert1_, dert2_ = cross_comp(P_, fiPpd)  # Pdert_t: Ldert_, Idert_, Ddert_, Mdert_ (tuples of derivatives per P param)
                    sum_rdn_(param_names, Pdert_t, fiPpd)  # sum cross-param redundancy per pdert, to evaluate for deeper processing
                    paramset = []
                    for param_name, Pdert_ in zip(param_names, Pdert_t):  # Pdert_ -> Pps:
                        param_md = []
                        for fPpd in 0, 1:  # 0-> Ppm_, 1-> Ppd_: more anti-correlated than Pp_s of different params
                            Pp_ = form_Pp_(Pdert_, fPpd)
                            param_md += [Pp_]  # -> [Ppm_, Ppd_]
                            if (fPpd and param_name == "D_") or (not fPpd and param_name == "I_"):
                                if not fPpd:
                                    splice_Ps(Pp_, dert1_, dert2_, fiPpd, fPpd)  # splice eval by Pp.M in Ppm_, for Pms in +IPpms or Pds in +DPpm
                                intra_Pp_(root, param_md, Pdert_, 1, fPpd)  # eval der+ or rng+ per Pp
                        paramset += [param_md]  # -> [Lmd, Imd, Dmd, Mmd]
                    sublayer0 += [paramset]  # -> [Pm_, Pd_]
                else: sublayer0 += [[]]  # empty paramset to preserve index in [Pm_, Pd_]
    # add nesting here

    root.levels.append(root.sublayers)  # to contain 1st and 2nd levels
    return root


def line_PPPs_root(Pp_t):

    Ppp_ttt = []
    for fPpd, Pp_ in enumerate(Pp_t):
        Ppp_tt = []
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
                    else:
                        Ppp_ = []     # keep index
                    Ppp_t.append(Ppp_)  # Pppm_, Pppd_
                Ppp_tt.append(Ppp_t)    # LPpp_, IPpp_, DPpp_, MPpp_
        Ppp_ttt.append(Ppp_tt)

    return Ppp_ttt


def level_recursion(P_T_):  # P_T_: 2P_, 16P_, 128P_., each level is implicitly nested to the depth = 1 + 2*elevation

    nextended = 0  # number of P_s with extended depth
    oP_T = []  # new level: flat list of P_s, to preserve input level
    iP_T = P_T_[-1]
    ntypes = 1 + 2 * math.log( len(iP_T)/2, 8)  # number of types per P_ in iP_T, with (fPd, param_name) n_pairs = math.log(len(iP_T)/2, 8)
    types_ = []  # parallel to P_T, for zipping

    for i, iP_ in enumerate( iP_T ):  # last-level-wide comp_form_P__

        types = []  # list of fPds and names of len = ntypes
        step = len(iP_T) / 2  # implicit nesting, top-down
        nsteps = 1
        while( len(types) < ntypes):  # decode unique set of alternating types per P_: [fPd,name,fPd,name..], from index in iP_T:
            if len(types) % 2:
                types.insert(0, int( i%step / (step/4) ))  # add name index: 0|1|2|3
            else:
                types.insert(0, int( (i/step)) % 2)  # add fPd: 0|1. This is the 1st increment because len(types) starts from 0
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

        if len(iP_) > 1 and sum([P.M for P in iP_]) > ave_M:
            nextended += 1
            oP_T += line_level_root(iP_T, iP_, types)  # comp Ps and form higher Ps, adds two nesting levels: oP_tt_, 8 P_s:
        else:
            oP_T += [[] for _ in range(8)]  # add 8 empty P_s, better to add count of missing prior P_s to each P_?

    cross_core_comp(oP_T, types_, ntypes)  # eval cross-comp of Pp_s in last sublevel iP_T, implicitly nested by all lower hierarchy
    # oP_T starts at 4th level: len(iP_T)>16, form xM per iP_, append to xM_ per oP_T?
    P_T_.append(oP_T)  # hierarchy of levels

    if len(iP_T) / max(nextended,1) < 4:  # ave_extend_ratio
        level_recursion(P_T_)  # increased implicit nesting in oP_T

'''
def line_level_root(iP_T, P_, types):  # cross_comp_Pp_, sum_rdn, splice, intra, comp_P_recursive
    norm_feedback(P_)  # before processing
    fPd = types[0]
    Pdert_t, pdert1_, pdert2_ = cross_comp_Pp_(P_, fPd)  # iP_: fully unpacked element in iP_T deepest 2-tuple (always Pm_, Pd_)
    sum_rdn(param_names, Pdert_t, fPd)
    oP_tt = []  # two new nesting levels added to iP_T per comp_P_recursive
    for Pdert_, param_name in zip(Pdert_t, param_names):  # param_name: LPp_ | IPp_ | DPp_ | MPp_
        for fPpd in 0, 1:  # 0: Ppm_, 1: Ppd_
            if Pdert_:
                oP_ = form_Pp_(Pdert_, fPpd)  # forms 8 Pp_s per iP_, but P sublevels is still missing one level?
                if (fPd and param_name == "D_") or (not fPd and param_name == "I_"):
                    if not fPpd:
                        splice_Ps(oP_, pdert1_, pdert2_, fPd)  # splice eval by Pp.M in Ppm_, for Pms in +IPpms or Pds in +DPpm
                    intra_Pp_(None, oP_, Pdert_, 1, fPpd)  # internal der+ | rng+
                oP_tt.append(oP_)  # nesting is implicit
            else:
                oP_tt.append([])  # preserve index
    return oP_tt
'''

def cross_core_comp(iP_T, types_, ntypes):  # draft, need further discussion and update
    '''
    comp Pp_s across >3 nesting levels in iP_T: common_root_depth - comparands_depth >3, which maps to the distance of >16 Pp_s
    '''
    xPp_t = []
    for elevation in range(int(ntypes)):
        if elevation % 2:  # params

            LP_t, IP_t, DP_t, MP_t = [], [], [], []
            # get P_ of each param for current elevation (compare at each elevation?)
            for i, types in enumerate(types_):
                if types[elevation] == 0:
                    LP_t.append(iP_T[i])
                elif types[elevation] == 1:
                    IP_t.append(iP_T[i])
                elif types[elevation] == 2:
                    DP_t.append(iP_T[i])
                elif types[elevation] == 3:
                    MP_t.append(iP_T[i])
                P_tt = [LP_t, IP_t, DP_t, MP_t]

                xPp_ = [] # cross compare between 4 params, always = 6 elements if call from root function
                for j, _P_t in enumerate(P_tt):
                    if j+1 < 4: # always < 4 due to there are 4 params
                        for P_t in P_tt[j+1:]:
                            for _P_ in _P_t:
                                for P_ in P_t:
                                    if _P_ and P_:  # not empty _P_ and P_
                                        _M = sum([_P.M for _P in _P_])
                                        M = sum([P.M for P in P_])
                                        for i,(param_name, ave) in enumerate(zip(param_names, aves)):
                                            xpdert_ = []
                                            for _P in _P_:
                                                for P in P_:
                                                    # probably wrong but we need this evaluation, add in PM for evaluation?
                                                    if _P.M + P.M + _M + M > (_P.Rdn + P.Rdn) * ave:
                                                        _param = getattr(_P,param_name[0])
                                                        param = getattr(P,param_name[0])
                                                        xpdert = comp_par(_P, _param, param, param_name, ave)
                                                        xpdert_.append(xpdert)
                                            if len(xpdert_)>1:
                                                fPd = 1
                                                xPp_.append(form_Pp_(xpdert_, fPd))  # add a loop to form xPp_ with fPd = 0 and fPd = 1?
                                            else:
                                                xPp_.append([])
            xPp_t.append(xPp_)


def intra_Pp(Pp, fPd):  # this is a rerun of line_PPs
    pass

# not needed:

def line_PPPs_start(P_ttt):  # starts level-recursion, higher-level input is nested to the depth = 1 + 2*elevation (level counter)
    '''
    # unpack for comp_Pp_recursive, not needed
    for Pp_tt, fPd in zip(P_ttt, [0, 1]):  # fPd: Pm_ | Pd_
        for Pp_t, param_name in zip(Pp_tt, param_names):  # LPp_ | IPp_ | DPp_ | MPp_
            for Pp_, fPpd in zip(Pp_t, [0, 1]):  # fPpd: Ppm_ | Ppd_
                if len(Pp_) > 1 and sum([Pp.M for Pp in Pp_]) > ave_M:
    '''
    [P_ttt].append( level_recursion(P_ttt) )  # add oP_tt_ as two new nesting levels in P_T

def compute_depth(l):
    """
    Get maximum number of depth from input list:  https://stackoverflow.com/questions/6039103/counting-depth-or-the-deepest-level-a-nested-list-goes-to
    """
    if isinstance(l, list):
        if l:
            return 1 + max(compute_depth(item) for item in l)
        else:
            return 1
    else:
        return 0


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