def comp_ptuple(_params, params):  # compare 2 lataples or vertuples, similar operations for m and d params

    tuple_ds, tuple_ms = Cptuple(), Cptuple()
    dtuple, mtuple = 0, 0
    _x, _L, _M, _Ma, _I  = _params.x, _params.L, _params.M, _params.Ma, _params.I
    x, L, M, Ma, I = params.x, params.L, params.M, params.Ma, params.I
    # x
    dx = _x - x; mx = ave_dx - abs(dx)
    tuple_ds.x = dx; tuple_ms.x = mx
    dtuple += abs(dx); mtuple += mx
    hyp = np.hypot(dx, 1)
    # L
    dL = _L - L/hyp;  mL = min(_L, L)
    tuple_ds.L = dL; tuple_ms.L = mL
    dtuple += abs(dL); mtuple += mL
    # I
    dI = _I - I; mI = ave_I - abs(dI)
    tuple_ds.I = dI; tuple_ms.I = mI
    dtuple += abs(dI); mtuple += mI
    # M
    dM = _M - M/hyp;  mM = min(_M, M)
    tuple_ds.M = dM; tuple_ms.M = mM
    dtuple += abs(dM); mtuple += mM
    # Ma
    dMa = _Ma - Ma;  mMa = min(_Ma, Ma)
    tuple_ds.Ma = dMa; tuple_ms.Ma = mMa
    dtuple += abs(dMa); mtuple += mMa

    if isinstance(_params.angle, tuple):
        # lataple:
        _G= _params.G; G = params.G
        dG = _params.G - params.G/hyp;  mG = min(_G, G)  # if comp_norm: reduce by hypot
        tuple_ds.G = dG; tuple_ms.G = mG
        dtuple += abs(dG); mtuple += mG

        _Ga = _params.Ga; Ga = params.Ga
        dGa = _Ga - Ga;  mGa = min(_Ga, Ga)
        tuple_ds.Ga = dGa; tuple_ms.Ga = mGa
        dtuple += abs(dGa); mtuple += mGa

        # angle = (sin_da, cos_da)
        _Dy,_Dx = _params.angle[:]; Dy,Dx = params.angle[:]
        _G = np.hypot(_Dy,_Dx); G = np.hypot(Dy,Dx)
        sin = Dy / (.1 if G == 0 else G); cos = Dx / (.1 if G == 0 else G)
        _sin = _Dy / (.1 if _G == 0 else _G); _cos = _Dx / (.1 if _G == 0 else _G)

        sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
        cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β
        dangle = np.arctan2(sin_da, cos_da)  # vertical difference between angles
        mangle = ave_dangle - abs(dangle)  # indirect match of angles, not redundant as summed
        tuple_ds.angle = dangle; tuple_ms.angle= mangle
        dtuple += mangle  # actually dangle
        mtuple += ave - abs(mangle)

        # aangle
        _sin_da0, _cos_da0, _sin_da1, _cos_da1 = _params.aangle
        sin_da0, cos_da0, sin_da1, cos_da1 = params.aangle
        sin_dda0 = (cos_da0 * _sin_da0) - (sin_da0 * _cos_da0)
        cos_dda0 = (cos_da0 * _cos_da0) + (sin_da0 * _sin_da0)
        sin_dda1 = (cos_da1 * _sin_da1) - (sin_da1 * _cos_da1)
        cos_dda1 = (cos_da1 * _cos_da1) + (sin_da1 * _sin_da1)
        daangle = (sin_dda0, cos_dda0, sin_dda1, cos_dda1)
        # day = [-sin_dda0 - sin_dda1, cos_dda0 + cos_dda1]; dax = [-sin_dda0 + sin_dda1, cos_dda0 + cos_dda1]

        gay = np.arctan2( (-sin_dda0 - sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in y?
        gax = np.arctan2( (-sin_dda0 + sin_dda1), (cos_dda0 + cos_dda1))  # gradient of angle in x?
        maangle = abs(np.arctan2(gay, gax))  # match between aangles, probably wrong
        tuple_ds.aangle = daangle; tuple_ms.aangle = maangle
        dtuple += maangle  # actually daangle
        mtuple += ave - abs(maangle)

    else:
        # vertuple
        _angle = _params.angle; angle = params.angle
        dangle = _angle - angle;  mangle = min(_angle, angle)
        tuple_ds.aangle = dangle; tuple_ms.angle = mangle
        dtuple += abs(dangle); mtuple += mangle

        _aangle = _params.aangle; aangle = params.aangle
        daangle = _aangle - aangle; maangle = min(_aangle, aangle)
        tuple_ds.aangle = daangle; tuple_ms.aangle = maangle
        dtuple += abs(daangle); mtuple += maangle

        _val=_params.val; val=params.val
        dval = _val - val; mval = min(_val, val)
        tuple_ds.val = dval; tuple_ms.val = mval
        dtuple += abs(dval); mtuple += mval

    tuple_ds.val = mtuple; tuple_ms.val = dtuple

    return tuple_ds, tuple_ms

def ave_ptuple(ptuple, n):

    for i, param in enumerate(ptuple):  # make ptuple an iterable?
        ptuple[i] = param / n

        if isinstance(param, tuple):
            for j, sub_param in enumerate(param):
                param[j] = sub_param / n  # angle or aangle
        else:
            ptuple[i] = param / n
'''
sum_pairs.x /= n; sum_pairs.L /= n; sum_pairs.M /= n; sum_pairs.Ma /= n; sum_pairs.G /= n; sum_pairs.Ga /= n; sum_pairs.val /= n
if isinstance(sum_pairs.angle, tuple):
    sin_da, cos_da = sum_pairs.angle[0]/n, sum_pairs.angle[1]/n
    sin_da0, cos_da0, sin_da1, cos_da1 = sum_pairs.aangle[0]/n, sum_pairs.aangle[1]/n, sum_pairs.aangle[2]/n, sum_pairs.aangle[3]/n
    sum_pairs.angle = (sin_da, cos_da)
    sum_pairs.aangle = (sin_da0, cos_da0, sin_da1, cos_da1)
else:
    sum_pairs.angle /= n
    sum_pairs.aangle /= n
    
accum_ptuple:
        # angle
        _sin_da, _cos_da = Ptuple.angle
        sin_da, cos_da = ptuple.angle
        sum_sin_da = (cos_da * _sin_da) + (sin_da * _cos_da)  # sin(α + β) = sin α cos β + cos α sin β
        sum_cos_da = (cos_da * _cos_da) - (sin_da * _sin_da)  # cos(α + β) = cos α cos β - sin α sin β
        Ptuple.angle = (sum_sin_da, sum_cos_da)
        # aangle
        _sin_da0, _cos_da0, _sin_da1, _cos_da1 = Ptuple.aangle
        sin_da0, cos_da0, sin_da1, cos_da1 = ptuple.aangle
        sum_sin_da0 = (cos_da0 * _sin_da0) + (sin_da0 * _cos_da0)  # sin(α + β) = sin α cos β + cos α sin β
        sum_cos_da0 = (cos_da0 * _cos_da0) - (sin_da0 * _sin_da0)  # cos(α + β) = cos α cos β - sin α sin β
        sum_sin_da1 = (cos_da1 * _sin_da1) + (sin_da1 * _cos_da1)
        sum_cos_da1 = (cos_da1 * _cos_da1) - (sin_da1 * _sin_da1)
        Ptuple.aangle = (sum_sin_da0, sum_cos_da0, sum_sin_da1, sum_cos_da1)
'''
