'''
line_PPs is a 2nd-level 1D algorithm, its input is Ps formed by the 1st-level line_patterns.
It cross-compares P params (initially L, I, D, M) and forms param_Ps (Pps) for each of them.
'''

import sys  # add CogAlg folder to system path
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname("CogAlg"), '..')))
from line_patterns import *
from frame_2D_alg.class_cluster import ClusterStructure, comp_param

class Cpdert(ClusterStructure):
    # P param dert
    i = int  # input for range_comp only
    p = int  # accumulated in rng
    d = int  # accumulated in rng
    m = int  # distinct in deriv_comp only
    negM = int  # in mdert only
    negL = int  # in mdert only
    negiL = int
    Pp = object  # for Pp merging in form_Pp_rng, temporary?

class CPp(CP):
    pdert_ = list
    P_ = list  # zip with pdert_
    Rdn = int  # cross-param rdn accumulated from pderts
    rval = int  # Pp value (M | abs D) adjusted for cross-param Rdn
    iL = int  # length of Pp in pixels
    fPd = bool  # P is Pd if true, else Pm; also defined per layer
    negM = int  # in mdert only
    negL = int  # in mdert only
    negiL = int
    sublayers = list

class CderPp(ClusterStructure):
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

class CPP(CPp, CderPp):
    layer1 = dict

ave = 100  # ave dI -> mI, * coef / var type
# no ave_mP: deviation computed via rM  # ave_mP = ave* n_comp_params: comp cost, or n vars per P: rep cost?
ave_div = 50
ave_rM = .5  # average relative match per input magnitude, at rl=1 or .5?
ave_negM = 10  # or rM?
ave_M = 100  # search stop
ave_D = 100  # search stop
ave_sub_M = 500  # sub_H comp filter
ave_Ls = 3
ave_PPM = 200
ave_splice = 50  # merge Ps within core-param Pp
ave_inv = 20  # ave inverse m, change to Ave from the root intra_blob?
ave_min = 5  # ave direct m, change to Ave_min from the root intra_blob?
ave_rolp = .5  # ave overlap ratio for comp_Pp


def search(P_, fPd):  # cross-compare patterns within horizontal line

    sub_search_recursive(P_, fPd)  # search with incremental distance: first inside sublayers
    layer0 = {'L_': [], 'I_': [], 'D_': [], 'M_': []}  # param_name: [params]

    if len(P_) > 1:  # at least 2 comparands
        Ldert_ = []
        for P in P_:  # unpack Ps
            L = P.L
            if "_P" in locals():  # not the 1st P
                _L = _P.L
                rL = L / _L  # div_comp L: higher-scale, not accumulated: no search
                mL = int(max(rL, 1 / rL)) * min(L, _L)  # match in comp by division as additive compression, not directional
                Ldert_.append(Cdert(i=L, p=L + _L, d=rL, m=mL))
            _P = P
            layer0['I_'].append(P.I / L)  # mean values for comp_param
            layer0['D_'].append(P.D / L)
            layer0['M_'].append(P.M / L)
        dert1__ = [Ldert_]
        Pdert__ = [Ldert_]  # no search for L, step=1 only, contains derts. Pp elements are pderts if param is core I

        for param_name in ["I_", "D_", "M_"]:
            param_ = layer0[param_name]  # param values
            # if dert-level P-defining param:
            if ((param_name == "I_") and not fPd) or ((param_name == "D_") and fPd):
                if not fPd:
                    Pdert__ += [search_param_(param_, layer0["D_"], P_, ave, rave=1)]  # pdert_ if "I_"
                # step=2 comp for P splice, one param: (I and not fPd) or (D and fPd):
                dert2_ = [comp_param(__par, par, param_name[0], ave) for __par, par in zip( param_[:-2], param_[2:])]
            # else step=1 per param only:
            dert1_ = [comp_param(_par, par, param_name[0], ave) for _par, par in zip( param_[:-1], param_[1:])]
            dert1__ += [dert1_]
            if not param_name == "I_": Pdert__ += [dert1_]  # dert_ = comp_param_

        rdn__ = sum_rdn_(layer0, Pdert__, fPd=1)  # assign redundancy to lesser-magnitude m|d in param pair for same-_P Pderts
        rdn_Ppm__ = []

        for param_name, Pdert_, rdn_ in zip(layer0, Pdert__, rdn__):  # segment Pdert__ into Pps
            if param_name == "I_" and not fPd:  # = isinstance(Pdert_[0], Cpdert)
                Ppm_ = form_Pp_rng(Pdert_, rdn_, P_)
            else:
                Ppm_ = form_Pp_(Pdert_, param_name, rdn_, P_, fPd=0)  # Ppd_ is formed in -Ppms only, in intra_Ppm_
            # list of param rdn_Ppm_s:
            rdn_Ppm__ += [form_rdn_Pp_(Ppm_, param_name, dert1__, dert2_, fPd=0)]

    return rdn_Ppm__


def search_param_(I_, D_, P_, ave, rave):  # variable-range search in mdert_, only if param is core param?

    # higher local ave for extended rng: -> lower m and term by match, and higher proj_M?
    mdert_ = []  # line-wide (i, p, d, m, negL, negM, negiL)

    for i, (_I, _D, _P) in enumerate( zip(I_[:-1], D_[:-1], P_[:-1])):
        proj_M = 1
        negiL = negL = negM = 0
        _pI = _I - (_D / 2)  # forward project by _D
        j = i + 1

        while proj_M > 0 and j < len(I_):
            I = I_[j]; D = D_[j]; P = P_[j]
            pI = I - (D / 2)  # backward project by D
            dert = comp_param(_pI, pI, "I_", ave)  # param is compared to prior-P _param
            if dert.m > 0:
                comp_sublayers(P_[i], P_[j], dert.m, dert.d)
                break  # 1st matching param takes over connectivity search from _param, in the next loop
            else:
                proj_M = dert.m * rave + negM - ave_M  # lower ave_M instead of projection?
                negM += dert.m * rave  # or abs m only?
                negiL += P.L
                negL += 1
                j += 1

        # after extended search, if any:
        mdert_.append( Cpdert(i=dert.i, p=dert.p, d=dert.d, m=dert.m, negiL=negiL, negL=negL, negM=negM))

    return mdert_


def sum_rdn_(layer0, Pdert__, fPd):
    '''
    access same-index pderts of all P params, assign redundancy to lesser-magnitude m|d in param pair.
    if other-param same-P_-index pdert is missing, rdn doesn't change.
    '''
    if fPd: alt = 'M'
    else:   alt = 'D'
    name_pairs = (('I', 'L'), ('I', 'D'), ('I', 'M'), ('L', alt), ('D', 'M'))  # pairs of params redundant to each other
    pderts_Rdn = [[], [], [], []]  # L_, I_, D_, M_' Rdns, as in pdert__

    for Ldert, Idert, Ddert, Mdert in zip(Pdert__[0], Pdert__[1], Pdert__[2], Pdert__[3]):  # 0: Ldert_, 1: Idert_, 2: Ddert_, 3: Mdert_
        # pdert per _P
        rdn_pairs = [[fPd, 0], [fPd, 1-fPd], [fPd, fPd], [0, 1], [1-fPd, fPd]]  # rdn in olp Pms, Pds: if fPd: I, M rdn+=1, else: D rdn+=1
        # names:    ('I','L'), ('I','D'),    ('I','M'),  ('L',alt), ('D','M'))
        # I *= M: comp value is combined?

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

        for i, param_name in enumerate(layer0):  # sum param rdn from all pairs it is in, flatten pair_names, pair_rdns?
            Rdn = 0
            for name_in_pair, rdn in zip(name_pairs, rdn_pairs):
                if param_name[0] == name_in_pair[0]:  # param_name = "L_", param_name[0] = "L"
                    Rdn += rdn[0]
                elif param_name[0] == name_in_pair[1]:
                    Rdn += rdn[1]

            pderts_Rdn[i].append(Rdn)  # same length as Pdert_

    return pderts_Rdn  # rdn__


def form_Pp_(dert_, param_name, rdn_, P_, fPd):
    # initialization:
    Pp_ = []
    x = 0
    _sign = None  # to initialize 1st P, (None != True) and (None != False) are both True

    for dert, rdn, P in zip(dert_, rdn_, P_):  # segment by sign
        if fPd: sign = dert.d > 0
        else:   sign = dert.m > 0  # adjust by ave projected at distance=negL and contrast=negM, if significant:
        # m + ddist_ave = ave - ave * (ave_rM * (1 + negL / ((param.L + _param.L) / 2))) / (1 + negM / ave_negM)?
        if sign != _sign:
            # sign change, initialize P and append it to P_
            Pp = CPp(L=1, iL=P_[x].L, I=dert.p, D=dert.d, M=dert.m, Rdn=rdn, x0=x, ix0=P_[x].x0, pdert_=[dert], P_=[P_[x]], sublayers=[], fPd=fPd)

            if hasattr(dert, "negL"):  # easier with is instance dert, Cpdert?
                Pp.accumulate(negiL=dert.negiL, negL=dert.negL, negM=dert.negM)
            Pp_.append(Pp)  # updated by accumulation below
        else:
            # accumulate params:
            Pp.L += 1; Pp.iL += P_[x].L; Pp.I += dert.p; Pp.D += dert.d; Pp.M += dert.m; Pp.Rdn += rdn; Pp.pdert_ += [dert]; Pp.P_ += [P]
            if hasattr(dert, "negL"):
                Pp.accumulate(negiL=dert.negiL, negL=dert.negL, negM=dert.negM)
        x += 1
        _sign = sign

    intra_Ppm_(Pp_, param_name, rdn_, fPd)  # evaluates for sub-recursion and forming Ppd_ per Pm
    # rng_search and der_comp for core param only: depending on M?
    return Pp_


def form_Pp_rng(dert_, rdn_, P_):  # multiple Pps may overlap within _dert.negL

    Pp_ = []
    merged_idx_ = []  # indices of merged derts P in P_

    for i, (_dert, _P, _rdn) in enumerate(zip(dert_, P_, rdn_)):
        # initialize Pp for positive derts only, else too much overlap?
        if _dert.m > ave*_rdn:
            if not isinstance(_dert.Pp, CPp):
                Pp = CPp(L=1, iL=_P.L, I=_dert.p, D=_dert.d, M=_dert.m, Rdn=_rdn, negiL=_dert.negiL, negL=_dert.negL, negM=_dert.negM,
                         x0=i, ix0=_P.x0, pdert_=[_dert], P_=[_P], sublayers=[], fPd=0)
                _dert.Pp = Pp
            else:
                Pp = _dert.Pp
            j = i + _dert.negL + 1
            while (j <= len(dert_)-1) and (j not in merged_idx_):
                dert = dert_[j]; P = P_[j]; rdn = rdn_[j]  # no pop: maybe used by other _derts
                if dert.m > ave*rdn:
                    if isinstance(dert.Pp, CPp):
                        # merge Pp with dert.Pp, if any:
                        Pp.accum_from(dert.Pp,excluded=['x0'])
                        Pp.P_ += dert.Pp.P_
                        Pp.pdert_ += dert.Pp.pdert_
                        Pp.sublayers += dert.Pp.sublayers
                        break
                    else:  # accumulate params:
                        Pp.L += 1; Pp.iL += P.L; Pp.I += dert.p; Pp.D += dert.d; Pp.M += dert.m; Pp.Rdn += rdn; Pp.negiL += dert.negiL
                        Pp.negL += dert.negL; Pp.negM += dert.negM; Pp.pdert_ += [dert]; Pp.P_ += [P]
                        dert.Pp = Pp
                        # Pp derts already searched through dert_, so they won't be used as _derts:
                        merged_idx_.append(j)
                        j += dert.negL
                else:
                    Pp_.append(Pp)  # even if single-dert
                    break  # Pp is terminated

    intra_Ppm_(Pp_, "I_", rdn_, fPd=0)  # evaluates for sub-recursion and forming Ppd_ per Pm
    # rng_search and der_comp for core param only: depends on M?
    return Pp_


def form_rdn_Pp_(Pp_, param_name, pdert1__, pdert2__, fPd):
    # cluster Pps by cross-param redundant value sign, re-evaluate them for cross-level rdn
    rPp_ = []
    x = 0
    _sign = None  # to initialize 1st rdn Pp, (None != True) and (None != False) are both True

    for Pp in Pp_:
        if fPd: Pp.rval = abs(Pp.D) - Pp.Rdn * ave_D * Pp.L
        else:   Pp.rval = Pp.M - Pp.Rdn * ave_D * Pp.L
        sign = Pp.rval > 0
        if sign != _sign:  # sign change, initialize rPp and append it to rPp_

            rPp = CPp(L=1, iL=Pp.iL, I=Pp.I, D=Pp.D, M=Pp.M, Rdn=Pp.Rdn, rval=Pp.rval, negiL=Pp.negiL, negL=Pp.negL, negM=Pp.negM,
                      x0=x, ix0=Pp.x0, pdert_=[Pp], sublayers=[], fPd=fPd)
            # or rPp is sign, Pp_?
            if _sign:  # -rPps are not processed?
                compact(rPp, pdert1__, pdert2__, param_name, fPd)  # re-eval Pps, Pp.pdert_s for redundancy, eval splice Ps
            rPp_.append(rPp)  # updated by accumulation below
        else:
            # accumulate params:
            rPp.L += 1; rPp.iL += Pp.iL; rPp.I += Pp.I; rPp.D += Pp.D; rPp.M += Pp.M; rPp.Rdn += Pp.Rdn; rPp.rval += Pp.rval
            rPp.negiL += Pp.negiL; rPp.negL += Pp.negL; rPp.negM += Pp.negM
            rPp.pdert_ += [Pp]
        x += 1
        _sign = sign

    return rPp_


def compact(rPp, pdert1__, pdert2_, param_name, fPd):  # re-eval Pps, Pp.pdert_s for redundancy, eval splice Ps

    for i, Pp in enumerate(rPp.pdert_):
        # assign cross-level rdn (Pp vs. pdert_), re-evaluate Pp and pdert_:
        Pp_val = Pp.rval / Pp.L - ave  # / Pp.L: resolution reduction, but lower rdn:
        pdert_val = Pp.rval - ave * Pp.L  # * Pp.L: ave cost * number of representations

        if Pp_val > pdert_val: pdert_val -= ave * Pp.Rdn
        else:                  Pp_val -= ave * Pp.Rdn  # ave scaled by rdn
        if Pp_val <= 0:
            rPp.pdert_[i] = CPp(pdert_=Pp.pdert_)  # Pp remove: reset Pp vars to 0

        elif ((param_name == "I_") and not fPd) or ((param_name == "D_") and fPd):  # P-defining params, else no separation
            M2 = M1 = 0
            # param match over step=2 and step=1:
            for pdert2 in pdert2_: M2 += pdert2.m  # match(I, __I or D, __D): only one pdert2_
            if fPd:
                for pdert1 in pdert1__[2]: M1 += pdert1.m  # match(D, _D)
            else:
                for pdert1 in pdert1__[1]: M1 += pdert1.m  # match(I, _I)

            if M2 / abs(M1) > -ave_splice:  # similarity / separation: splice Ps in Pp, also implies weak Pp.pdert_?
                _P = CP()
                for P in Pp.P_:
                    _P.accum_from(P, excluded=["x0"])  # different from Pp params
                    _P.dert_ += [P.dert_]  # splice dert_s, eval intra_P?
                rPp.pdert_[i] = _P  # replace Pp with spliced P

        if pdert_val <= 0:
            Pp.pdert_ = []  # remove pdert_


def intra_Ppm_(Pp_, param_name, rdn_, fPd):  # evaluate for sub-recursion in line Pm_, pack results into sub_Pm_

    comb_layers = []  # combine into root P sublayers[1:]
    # each Pp is evaluated for incremental range and derivation xcomp, as in line_patterns but via localized aves

    for Pp, rdn in zip( Pp_, rdn_):  # each sub_layer is nested to depth = sublayers[n]
        if Pp.L > 2:
            if Pp.M > 0:  # low-variation span, eval rng_comp
                if Pp.M > ave_M * rdn and param_name=="I_":  # and if variable cost: Pp.M / Pp.L? reduced by lending to contrast?
                    rdn_ = [rdn+1 for rdn in rdn_]
                    I_ = [pdert.i for pdert in (Pp.pdert_[:-1])]
                    D_ = [pdert.d for pdert in (Pp.pdert_[:-1])]
                    # search range is extended by higher ave: less replacing by match,
                    # and by higher proj_P = dert.m * rave ((Pp.M / Pp.L) / ave): less term by miss:
                    P_ave = Pp.M / Pp.L
                    rpdert_ = search_param_(I_, D_, Pp.P_, (ave + P_ave) / 2, rave = P_ave / ave )

                    sub_Ppm_ = form_Pp_(rpdert_, param_name, rdn_, Pp.P_, fPd=False)  # cluster by m sign, eval intra_Pm_
                    Pp.sublayers += [[[fPd, sub_Ppm_]]]  # 1st sublayer is single element
                    if len(sub_Ppm_) > 4:
                        Pp.sublayers += intra_Ppm_(sub_Ppm_, param_name, rdn_, fPd)  # feedback, add sum params for comp_sublayers?
                        comb_layers = [ comb_layers + sublayers for comb_layers, sublayers in
                                        zip_longest(comb_layers, Pp.sublayers, fillvalue=[])  # splice sublayers across sub_Pps
                                      ]
            else:  # neg Ppm: high-variation span, min neg M is contrast value, borrowed from adjacent +Ppms:
                if -Pp.M > ave_D * rdn:  # or abs D: likely sign match span?
                    rdn_ = [rdn+1 for rdn in rdn_]
                    mean_M = Pp.M / Pp.L  # for internal Pd eval, +opposite-side mean_M?

                    sub_Ppd_ = form_Pp_(Pp.pdert_, param_name, rdn_, Pp.P_, fPd=True)  # cluster by d sign: partial d match, eval intra_Pm_(Pdm_)
                    Pp.sublayers += [[[True, sub_Ppd_]]]  # 1st layer, Dert=[], fill if Ls > min?
                    Pp.sublayers += intra_Ppd_(sub_Ppd_, param_name, mean_M, rdn_)  # der_comp eval, rdn_?
                    comb_layers = [ comb_layers + sublayers for comb_layers, sublayers in
                                    zip_longest(comb_layers, Pp.sublayers, fillvalue=[])  # splice sublayers across sub_Pps
                                  ]
    return comb_layers


def intra_Ppd_(Pd_, param_name, mean_M, rdn_):  # evaluate for sub-recursion in line P_, packing results in sub_P_

    comb_layers = []
    for Pd, rdn in zip( Pd_, rdn_):  # each sub in sub_ is nested to depth = sub_[n]

        if abs(Pd.D) * mean_M > ave_D * rdn and Pd.L > 3:  # mean_M from adjacent +ve Ppms
            rdn_ = [rdn + 1 for rdn in rdn_]
            ddert_ = []
            for _pdert, pdert in zip( Pd.pdert_[:-1], Pd.pdert_[1:]):  # Pd.pdert_ is dert1_
                _param = _pdert.d; param = pdert.d
                dert = comp_param(_param, param, param_name[0], ave)  # cross-comp of ds in dert1_, !search, also local aves?
                ddert_ += [ Cdert( i=dert.i, p=dert.p, d=dert.d, m=dert.m)]
            # cluster Pd derts by md sign:
            sub_Pm_ = form_Pp_(ddert_, param_name, rdn_, Pd.P_, fPd=True)
            Pd.sublayers += [[[True, True, sub_Pm_ ]]]  # 1st layer: fid, fPd, rdn, rng, sub_P_
            if len(sub_Pm_) > 3:
                Pd.sublayers += intra_Ppm_(sub_Pm_, param_name, rdn_, fPd=True)
                comb_layers = [ comb_layers + sublayers for comb_layers, sublayers in
                                zip_longest(comb_layers, Pd.sublayers, fillvalue=[])  # splice sublayers across sub_Pps
                              ]
    return comb_layers


# draft and tentative
def sub_search_recursive(P_, fPd):  # search in top sublayer per P / sub_P

    for P in P_:
        if P.sublayers:
            sublayer = P.sublayers[0][0]  # top sublayer has one array
            sub_P_ = sublayer[4]
            if len(sub_P_) > 2:
                if fPd:
                    if abs(P.D) > ave_D:  # better use sublayers.D|M, but we don't have it yet
                        sub_rdn_Pp__ = search(sub_P_, fPd)
                        sublayer[5].append(sub_rdn_Pp__)
                        sub_search_recursive(sub_P_, fPd)  # deeper sublayers search is selective per sub_P
                elif P.M > ave_M:
                    sub_rdn_Pp__ = search(sub_P_, fPd)
                    sublayer[5].append(sub_rdn_Pp__)
                    sub_search_recursive(sub_P_, fPd)  # deeper sublayers search is selective per sub_P


def comp_sublayers(_P, P, mP, dP):  # not revised; also add dP?

    if P.sublayers and _P.sublayers:  # not empty sub layers
        for _sub_layer, sub_layer in zip(_P.sublayers[0], P.sublayers[0]):

            if _sub_layer and sub_layer:
                _fid, _fPd, _rdn, _rng, _sub_P_, _sub_Pp__, = _sub_layer
                fid, fPd, rdn, rng, sub_P_, sub_Pp__ = sub_layer
                # fork comparison:
                if fPd == _fPd and rng == _rng and min(_P.L, P.L) > ave_Ls:
                    sub_mP = sub_dP = 0
                    # compare all sub_Ps to each _sub_P:
                    for _sub_P in _sub_P_:
                        for sub_P in sub_P_:
                            sub_dert = comp_param(_sub_P.I, sub_P.I, "I_", ave)
                    sub_mP += sub_dert.m  # of compared H, no specific mP?
                    sub_dP += sub_dert.d
                    if sub_mP + mP < ave_sub_M:
                        # potentially mH: trans-layer induction: if mP + sub_mP < ave_sub_M: both local and global values of mP.
                        break  # low vertical induction, deeper sublayers are not compared
                else:
                    break  # deeper P and _P sublayers are from different intra_comp forks, not comparable?

# below is obsolete:
'''     
param_ = [[ getattr( P, param_name[0]), P.L, P.x0] for P in P_]  # param values
D_ = [ getattr(P, "D") for P in P_]
_par_= []
for (I, L, x0),(D,_,_) in zip(param_[:-1], layer0["D_"][:-1]):  # _I in (I,L,x0) is forward projected by _D in (D,L,x0)
    _par_.append((I-(D/2), L, x0))
par_= []
for (I, L, x0),(D,_,_) in zip(param_[1:], layer0["D_"][1:]): # I in (I,L,x0) is backward projected by D in (D,L,x0)
    par_.append((I+(D/2), L, x0))
    
rL_ = [_L[0]/L[0] for _L, L in zip(layer0["L_"][:-1],layer0["L_"][1:])] # _L = L_[:-1], L = L_[1:], div_comp L, no search?
mL_ = [int(rL) * min(_L[0],L[0]) for _L, L, rL in zip(layer0["L_"][:-1],layer0["L_"][1:], rL_)] # definition of div_match
'''

def search_old(P_, fPd):  # cross-compare patterns within horizontal line

    sub_search_recursive(P_, fPd)  # search with incremental distance: first inside sublayers
    layer0 = {'L_': [], 'I_': [], 'D_': [], 'M_': []}  # param_name: [params]

    if len(P_) > 1:  # at least 2 comparands
        Ldert_ = []; rL_ = []
        # unpack Ps:
        for P in P_:
            if "_P" in locals():  # not the 1st P
                L=P.L; _L=_P.L
                rL = L /_L  # div_comp L: higher-scale, not accumulated: no search
                mL = int(max(rL, 1/rL)) * min(L,_L)  # match in comp by division as additive compression, not directional
                Ldert_.append(Cdert( i=L, p=L+_L, d=rL, m=mL ))
                rL_.append(rL)
            _P = P
            layer0['I_'].append([P.I, P.L, P.x0])  # I tuple
            layer0['D_'].append([P.D, P.L, P.x0])  # D tuple
            layer0['M_'].append([P.M, P.L, P.x0])  # M tuple

        dert1__ = [Ldert_]  # no search for L, step=1 only, contains derts vs. pderts
        Pdert__ = [Ldert_]  # Pp elements: pderts if param is core m, else derts

        for param_name in ["I_", "D_", "M_"]:
            param_ = layer0[param_name]  # param values
            par_ = param_[1:]  # compared vectors:
            _par_ = [[ _par*rL, L, x0] for [_par,L,x0], rL in zip(param_[:-1], rL_) ]  # normalize by rL

            if ((param_name == "I_") and not fPd) or ((param_name == "D_") and fPd):  # dert-level P-defining params
                if not fPd:
                    # project I by D, or D by Dd in deriv_comp sub_Ps:
                    _par_ = [[_par - (D / 2), L, x0] for [_par, L, x0], [D, _, _] in zip(_par_, layer0["D_"][:-1])]
                    # _I in (I,L,x0) is  - (D / 2): forward projected by _D in (D,L,x0)
                    par_ = [[ par + (D / 2), L, x0] for [ par, L, x0], [D, _, _] in zip(par_, layer0["D_"][1:])]
                    # I in (I,L,x0) is backward projected by D in (D,L,x0)
                    Pdert__ += [ search_param_(_par_, par_, P_[:-1], ave, rave=1) ]  # pdert_ if "I_"
                del _P
                _rL_=[]
                for P in P_:  # form rLs to normalize cross-comp of same-M-sign Ps in pdert2_
                    if "_P" in locals():  # not the 1st P
                        if "__P" in locals():  # not the 2nd P
                            _rL_.append(P.L / __P.L)
                        __P = _P
                    _P = P
                __par_ = [[__par * _rL, L, x0] for [__par, L, x0], _rL in zip(param_[:-2], _rL_)]  # normalize by _rL
                # step=2 comp for P splice, one param: (I and not fPd) or (D and fPd):
                dert2_ = [ comp_param(__par, par, param_name[0], ave) for __par, par in zip(__par_, par_[1:]) ]
            # else step=1 only:

            dert1_ = [ comp_param(_par, par, param_name[0], ave) for _par, par in zip(_par_, par_) ]  # append pdert1_ per param_
            dert1__ += [dert1_]
            if not param_name=="I_": Pdert__ += [dert1_]  # dert_ = comp_param_

        rdn__ = sum_rdn_(layer0, Pdert__, fPd=1)  # assign redundancy to lesser-magnitude m|d in param pair for same-_P Pderts
        rdn_Ppm__ = []

        for param_name, Pdert_, rdn_ in zip(layer0, Pdert__, rdn__):  # segment Pdert__ into Pps
            if param_name=="I_" and not fPd:  # = isinstance(Pdert_[0], Cpdert)
                Ppm_ = form_Pp_rng(Pdert_, rdn_, P_)
            else:
                Ppm_ = form_Pp_(Pdert_, param_name, rdn_, P_, fPd=0)  # Ppd_ is formed in -Ppms only, in intra_Ppm_
            # list of param rdn_Ppm_s:
            rdn_Ppm__ += [ form_rdn_Pp_(Ppm_, param_name, dert1__, dert2_, fPd=0) ]

    return rdn_Ppm__


def form_PP_(params_derPp____, fPd):  # Draft:
    '''
    unpack 4-layer derPp____: _names ( _Pp_ ( names ( Pp_ ))),
    pack derPps with overlapping match: sum of concurrent mPps > ave_M * rolp, into PPs of PP_
    '''
    rdn = [.25, .5, .25, .5]  # {'L_': .25, 'I_': .5, 'D_': .25, 'M_': .5}
    names = ['L_', 'I_', 'D_', 'M_']
    Rolp = 0
    PP_ = []
    _sign = None
    # init new empty derPp____ with the same list structure as params_derPp____, for [i][j][k] indexing later
    derPp____ = [[[[] for param_derPp_ in param_derPp__] \
                  for param_derPp__ in param_derPp___] \
                 for param_derPp___ in params_derPp____]
    param_name_ = derPp____.copy()

    for i, _param_derPp___ in enumerate(params_derPp____):  # derPp___ from comp_Pp (across params)
        for j, _Pp_derPp__ in enumerate(_param_derPp___):  # from comp_Pp (param_Pp_, other params)
            for k, param_derPp_ in enumerate(_Pp_derPp__):  # from comp_Pp (_Pp, other params)
                for (derPp, rolp, _name, name) in param_derPp_:  # from comp_Pp (_Pp, other param' Pp_)
                    # debugging
                    if names[i] != _name: raise ValueError("Wrong _name")
                    if names[k] != name: raise ValueError("Wrong name")

                    if "pre_PP" not in locals(): pre_PP = CPP(derPp____=derPp____.copy())
                    # if fPd: derPp_val = derPp.dPp; ave = ave_D
                    # else:   derPp_val = derPp.mPp; ave = ave_M
                    # mean_rdn = (rdn[i] + rdn[k]) / 2  # of compared params
                    # if derPp_val * mean_rdn > ave:
                    # else: pre_PP = CPP(derPp____=derPp____.copy())
                    # accum either sign, no eval or sub_PP_ per layer:
                    Rolp += rolp
                    pre_PP.accum_from(derPp)
                    pre_PP.derPp____[i][j][k].append(derPp)
                    pre_PP.param_name_.append((names[i], names[k]))
        '''    
        We can't evaluate until the top loop because any overlap may form sufficient match. 
        Then we only define pre_PPs by overlap of any element of any layer to any other element of any other layer.
        But there are so many possible overlaps that pre_PP may never terminate.
        Another way to define them is by minimal combined-layers' match per x (iP). 
        But then we are back to immediate multi-param comp_P_, which is pointless because different derivatives anti-correlate.
                # inclusion into higher layer of pre_PP by the sum of concurrent mPps > ave_M * Rolp, over all lower layers:
                if "pre_PP" in locals() and pre_PP.derPp____[i][j][k] and not pre_PP.mPp > ave_M * Rolp:
                    pre_PP = CPP(derPp____=derPp____.copy())
            # pre_PP.derPp____[i][j] is a nested list, we need to check recursively to determine whether there is any appended derPp
            if "pre_PP" in locals() and not emptylist(pre_PP.derPp____[i][j]) and not pre_PP.mPp > ave_M * Rolp:
                pre_PP = CPP(derPp____=derPp____.copy())
        '''
        if "pre_PP" in locals() and not emptylist(pre_PP.derPp____[i]):
            if pre_PP.mPp > ave_M * Rolp:
                PP_.append(pre_PP)  # no negative PPs?
                _sign = True
            else:
                _sign = False
                pre_PP = CPP(derPp____=derPp____.copy())

    return PP_

# https://stackoverflow.com/questions/1593564/python-how-to-check-if-a-nested-list-is-essentially-empty
def emptylist(in_list):
    '''
    check if nested list is totally empty
    '''
    if isinstance(in_list, list):  # Is a list
        return all(map(emptylist, in_list))
    return False  # Not a list


def comp_Pp(_Pp, Pp, layer0):
    '''
    next level line_PPPs:
    PPm_ = search_Pp_(layer0, fPd=0)  # calls comp_Pp_ and form_PP_ per param
    PPd_ = search_Pp_(layer0, fPd=1)
    '''
    mPp = dPp = 0
    layer1 = dict({'L': .0, 'I': .0, 'D': .0, 'M': .0})
    dist_coef = ave_rM * (1 + _Pp.negL / _Pp.L)
    # average match projected at current distance, needs a review
    for param_name in layer1:
        if param_name == "I":
            ave = ave_inv  # * dist_coef
        else:
            ave = ave_min  # * dist_coef
        param = getattr(_Pp, param_name)
        _param = getattr(Pp, param_name)
        dert = comp_param(_param, param, [], ave)
        rdn = layer0[param_name + '_'][1]  # index 1 =rdn
        mPp += dert.m * rdn
        dPp += dert.d * rdn
        layer1[param_name] = dert

    negM = _Pp.negM - Pp.negM
    negL = _Pp.L - Pp.negL
    negiL = _Pp.iL - Pp.negiL

    '''
    options for div_comp, etc.    
    if P.sign == _P.sign: mP *= 2  # sign is MSB, value of sign match = full magnitude match?
    if mP > 0
        # positive forward match, compare sublayers between P.sub_H and _P.sub_H:
       comp_sublayers(_P, P, mP)
    if isinstance(_P.derP, CderP):  # derP is created in comp_sublayers
        _P.derP.sign = sign
        _P.derP.layer1 = layer1
        _P.derP.accumulate(mP=mP, neg_M=neg_M, neg_L=neg_L, P=_P)
        derP = _P.derP
    else:
        derP = CderP(sign=sign, mP=mP, neg_M=neg_M, neg_L=neg_L, P=_P, layer1=layer1)
        _P.derP = derP
    '''
    derPp = CderPp(mPp=mPp, dPp=dPp, negM=negM, negL=negL, negiL=negiL, _Pp=_Pp, Pp=Pp, layer1=layer1)

    return derPp

def div_comp_P(PP_):  # draft, check all PPs for x-param comp by division between element Ps
    '''
    div x param if projected div match: compression per PP, no internal range for ind eval.
    ~ (L*D + L*M) * rm: L=min, positive if same-sign L & S, proportional to both but includes fractional miss
    + PPm' DL * DS: xP difference compression, additive to x param (intra) compression: S / L -> comp rS
    also + ML * MS: redundant unless min or converted?
    vs. norm param: Var*rL-> comp norm param, simpler but diffs are not L-proportional?
    '''
    for PP in PP_:
        vdP = (PP.adj_mP + PP.P.M) * abs(PP.dP) - ave_div
        if vdP > 0:
            # if irM * D_vars: match rate projects der and div match,
            # div if scale invariance: comp x dVars, signed
            ''' 
            | abs(dL + dI + dD + dM): div value ~= L, Vars correlation: stability of density, opposite signs cancel-out?
            | div_comp value is match: min(dL, dI, dD, dM) * 4, | sum of pairwise mins?
            '''
            _derP = PP.derP_[0]
            # smP, vmP, neg_M, neg_L, iP, mL, dL, mI, dI, mD, dD, mM, dM = P,
            # _sign, _L, _I, _D, _M, _dert_, _sub_H, __smP = _derP.P
            _P = _derP.P
            for i, derP in enumerate(PP.derP_[1:]):
                P = derP.P
                # DIV comp L, SUB comp (summed param * rL) -> scale-independent d, neg if cross-sign:
                rL = P.L / _P.L
                # mL = whole_rL * min_L?
                '''
                dI = I * rL - _I  # rL-normalized dI, vs. nI = dI * rL or aI = I / L
                mI = ave - abs(dI)  # I is not derived, match is inverse deviation of miss
                dD = D * rL - _D  # sum if opposite-sign
                mD = min(D, _D)   # same-sign D in dP?
                dM = M * rL - _M  # sum if opposite-sign
                mM = min(M, _M)   # - ave_rM * M?  negative if x-sign, M += adj_M + deep_M: P value before layer value?
                mP = mI + mM + mD  # match(P, _P) for derived vars, defines norm_PPm, no ndx: single, but nmx is summed
                '''
                for (param, _param) in zip([P.I, P.D, P.M], [_P.I, _P.D, _P.M]):
                    dm = comp_param(param, _param, [], ave, rL)
                    layer1.append([dm.d, dm.m])
                    mP += dm.m; dP += dm.d

                if dP > P.derP.dP:
                    ndP_rdn = 1; dP_rdn = 0  # Not sure what to do with these
                else:
                    dP_rdn = 1;  ndP_rdn = 0

                if mP > derP.mP:
                    rrdn = 1  # added to rdn, or diff alt, olp, div rdn?
                else:
                    rrdn = 2
                if mP > ave * 3 * rrdn:
                    # rvars = mP, mI, mD, mM, dI, dD, dM  # redundant vars: dPP_rdn, ndPP_rdn, assigned in each fork?
                    rvars = layer1
                else:
                    rvars = []
                # append rrdn and ratio variables to current derP:
                # PP.derP_[i] += [rrdn, rvars]
                PP.derP_[i].rrdn = rrdn
                PP.derP_[i].layer1 = rvars
                # P vars -> _P vars:
                _P = P
                '''
                m and d from comp_rate is more accurate than comp_norm?
                rm, rd: rate value is relative? 
                also define Pd, if strongly directional? 
                if dP > ndP: ndPP_rdn = 1; dPP_rdn = 0  # value = D | nD
                else:        dPP_rdn = 1; ndPP_rdn = 0
                '''
    return PP_


def form_adjacent_mP(derPp_):  # not used in discontinuous search?
    pri_mP = derPp_[0].mP
    mP = derPp_[1].mP
    derPp_[0].adj_mP = derPp_[1].mP

    for i, derP in enumerate(derPp_[2:]):
        next_mP = derP.mP
        derPp_[i + 1].adj_mP = (pri_mP + next_mP) / 2
        pri_mP = mP
        mP = next_mP

    return derPp_


# to be updated
def draw_PP_(image, frame_PP_):
    # init every possible combinations
    img_mparams = {'L_I_': np.zeros_like(image), 'L_D_': np.zeros_like(image),
                   'L_M_': np.zeros_like(image), 'I_D_': np.zeros_like(image),
                   'I_M_': np.zeros_like(image), 'D_M_': np.zeros_like(image)}

    img_dparams = {'L_I_': np.zeros_like(image), 'L_D_': np.zeros_like(image),
                   'L_M_': np.zeros_like(image), 'I_D_': np.zeros_like(image),
                   'I_M_': np.zeros_like(image), 'D_M_': np.zeros_like(image)}

    for y, (PPm_, PPd_) in enumerate(frame_PP_):  # draw each line
        if PPm_ or PPd_:
            for PPm, PPd in zip_longest(PPm_, PPd_, fillvalue=[]):

                if PPm:
                    draw_PP(img_mparams, PPm, y)
                if PPd:
                    draw_PP(img_dparams, PPd, y)
    # plot diagram of each pair PPs
    plt.figure()
    for i, param in enumerate(img_mparams):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img_mparams[param], vmin=0, vmax=255)
        plt.title("pair = " + param + " m param")

    plt.figure()
    for i, param in enumerate(img_dparams):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img_dparams[param], vmin=0, vmax=255)
        plt.title("pair = " + param + " d param")


def draw_PP(img_params, PP, y):
    for (derPp___) in PP.derPp____:
        for derPp__ in derPp___:
            for (derPp_) in derPp__:
                for derPp in derPp_:
                    Pp = derPp.Pp
                    _Pp = derPp._Pp
                    _name, name = PP.param_name_.pop(0)
                    # values draw
                    img_params[_name + name][y, _Pp.ix0:_Pp.ix0 + _Pp.iL] += 32
                    img_params[_name + name][y, Pp.ix0:Pp.ix0 + Pp.iL] += 32