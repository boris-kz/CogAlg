def cross_comp(frame_of_pixels_):  # converts frame_of_pixels to frame_of_patterns, each pattern maybe nested

    Y, X = image.shape  # Y: frame height, X: frame width
    frame_of_patterns_ = []
    for y in range(ini_y + 1, Y):
        # initialization per row:
        pixel_ = frame_of_pixels_[y, :]  # y is index of new line pixel_
        P_ = []  # row of patterns
        __p, _p = pixel_[0:2]  # each prefix '_' denotes prior
        _d = _p - __p  # initial comp
        _m = ave - abs(_d)
        _bi_d = _d * 2  # __d and __m are back-projected as = _d or _m
        _bi_m = _m * 2
        # initialize P with dert_[0]:
        P = _bi_m > 0, [], 1, __p, _bi_d, _bi_m, [(__p, _bi_d, _bi_m, None)], []  # sign, depth, L, I, D, M, dert_, sub_

        for p in pixel_[2:]:  # pixel p is compared to prior pixel _p in a row
            d = p - _p
            m = ave - abs(d)  # initial match is inverse deviation of |difference|
            bi_d = d + _d  # ave bilateral difference
            bi_m = m + _m  # ave bilateral match
            dert = _p, bi_d, bi_m, _d
            # accumulate or terminate mP: span of pixels forming same-sign m:
            P, P_ = form_P(P, P_, dert)
            _p, _d, _m = p, d, m
        # terminate last P in row:
        dert = _p, _d * 2, _m * 2, _d  # last d and m are forward-projected to bilateral values
        P, P_ = form_P(P, P_, dert)
        P_ += [P]  # adds incomplete P
        # evaluate sub-recursion per P:
        intra_P(P_, fid=False, rdn=1, rng=1, fseg=False)  # recursive
        frame_of_patterns_ += [P_]  # line of patterns is added to frame of patterns

    return frame_of_patterns_  # frame of patterns will be output to level 2

def intra_P(P_, fid, rdn, rng, fseg):  # evaluate for sub-recursion in line P_, filling its sub_P_ with the results

    deep_sub_ = []  # sub_ extension feedback from intra_P

    for sign, LL, L, I, D, M, dert_, sub_ in P_:  # sub_: list of lower pattern layers, nested to depth = sub_[n]

        if sign and not fseg:  # positive m: low-variation P
            if M > ave_M * rdn and L > 4:  # rng+ eval vs. fixed cost = ave_M
                sub_rng = rng + 1  # n of extensions, comp range = rng**2: 1, 2, 4.., kernel = rng * 2 + 1: 3, 5, 9
                lateral_sub_ = rng_comp(dert_, fid)  # form sub_Ps within dert_
                lL = len(lateral_sub_); sub_rdn = rdn
                if rdn > 1:  # += 1.2 per recursion: 1 + (1 / ave_nP): rdn to higher derts + ave rdn to higher sub_
                    sub_rdn += 1 / lL - 0.2  # adjust distributed rdn estimated in prior intra_P
                sub_ += [[( sign, lL, fseg, fid, sub_rdn, sub_rng, lateral_sub_ )]]  # 1st layer, add Dert?
                sub_ += intra_P(lateral_sub_, fid, sub_rdn+1.2, sub_rng, fseg=False) # recursion eval, feedback

        if fseg:  # P is seg_P: d sign match is partial d match and pre-condition for der_comp, for any d sign
        # or elif to avoid overlap
            if (abs(D) > ave_D * rdn) and L > 3:  # der+ fixed cost eval
                lateral_sub_ = der_comp(dert_)  # cross-comp uni_ds in dert[3]
                lL = len(lateral_sub_); sub_rdn = rdn
                if rdn > 1: sub_rdn += 1 / lL - 0.2  # adjust distributed rdn estimated in prior intra_P
                sub_ += [[( sign, lL, fseg, True, sub_rdn, rng, lateral_sub_ )]]  # 1st layer, add Dert?
                # sign of input m, not md,
                sub_ += intra_P(lateral_sub_, True, sub_rdn+1.2, rng, fseg=False) # deep layers feedback

        elif L > ave_L * rdn:  # high variation, segment by d sign, filtered by L because d sign may change
            lateral_sub_ = segment(dert_)  # segment dert_ by ds: sign match covers variable cost of der+?
            lL = len(lateral_sub_); sub_rdn = rdn
            if rdn > 1: sub_rdn += 1 / lL - 0.2  # adjust distributed rdn estimated in prior intra_P
            sub_ += [[( sign, lL, fseg, True, sub_rdn, rng, lateral_sub_ )]]  # 1st layer, add Dert?
            sub_ += intra_P(lateral_sub_, True, sub_rdn+1.2, rng, fseg=True)  # will trigger der+ eval

        # each: else merge non-selected sub_Ps within P, if in max recursion depth? Eval per P_: same op, !layer
        LL[:] = [len(sub_)]

        for i, sub in enumerate(sub_):
            if sub == []: break
            try: deep_sub_[i].extend(sub)
            except IndexError: deep_sub_.append(sub)

    return deep_sub_  # add return of Dert and hypers, same in sub_[0]? [] fill if min_nP: L, LL?

def form_deep_sub_(dert_, sign, fdP, fid, rdn, rng, rsub_, dsub_):
    # rdn was estimated as rdn += 1.2: 1 (rdn to higher derts) + 1 / ave_nP (ave rdn to higher sub_)

    sub_mP_ = form_P_(dert_, False)
    lL = len(sub_mP_)
    if rdn > 1: rdn += 1 / lL - 0.2  # adjust distributed part of estimated rdn: sub_rdn += 1 / lL - 0.2
    dsub_ += [[(fdP, sign, lL, fid, rdn, rng, sub_mP_)]]  # 1st layer, Dert[] fill if lL > min?

    sub_dP_ = form_P_(dert_, True)
    lL = len(sub_dP_)
    if rdn > 1: rdn += 1 / lL - 0.2
    rsub_ += [[(fdP, sign, lL, True, rdn, rng, sub_dP_)]]  # 1st layer, Dert[] fill if lL > min?

    _rsub_, _dsub_ = intra_P(sub_dP_, True, True, rdn + 1.2, rng)  # deep layers feedback
    dsub_ += [rsub + dsub for rsub, dsub in zip_longest(_rsub_, _dsub_, fillvalue=[])]

    _rsub_, _dsub_ = intra_P(sub_mP_, False, fid, rdn + 1.2, rng + 1)
    rsub_ += [rsub + dsub for rsub, dsub in zip_longest(_rsub_, _dsub_, fillvalue=[])]
    # splice temp _rsub_ and _dsub_ into single r|d sub_ hierarchy

    # not needed?:
    # sub_ hierarchy extension by feedback from intra_P: dsub_ if fdP, else rsub_:
    # deep_rsub_ = [rsub + dsub for rsub, dsub in zip_longest(deep_rsub_, rsub_, fillvalue=[])]
    # deep_dsub_ = [rsub + dsub for rsub, dsub in zip_longest(deep_dsub_, dsub_, fillvalue=[])]
    return rsub_, dsub_


def segment(dert_):  # P segmentation by same d sign: initialization, accumulation, termination

    sub_ = []  # becomes lateral_sub_
    _p, _d, _m, _uni_d = dert_[0]  # prefix '_' denotes prior
    try:
        _sign = _uni_d > 0; ini = 1
    except:
        _p, _d, _m, _uni_d = dert_[1]  # skip dert_[0] if uni_d is None: 1st dert in comp sequence
        _sign = _uni_d > 0; ini = 2
    LL, L, I, D, M, seg_dert_ = [], 1, _p, _uni_d, _m, [(_p, _d, _m, _uni_d)]  # initialize seg_P, same as P

    for p, d, m, uni_d in dert_[ini:]:
        sign = uni_d > 0
        if _sign != sign:
            sub_.append((_sign, LL, L, I, D, M, seg_dert_, []))  # terminate seg_P, same as P
            LL, L, I, D, M, seg_dert_, sub_ = [], 0, 0, 0, 0, [], []  # reset accumulated seg_P params
        _sign = sign
        L += 1; I += p; D += uni_d; M += m  # D += uni_d to eval for comp uni_d
        seg_dert_.append((p, d, m, uni_d))

    sub_.append((_sign, LL, L, I, D, M, seg_dert_, []))  # pack last segment, nothing to accumulate
    # also Dert in sub_ [], fill if min lLL?
    return sub_  # becomes lateral_sub_


def rng_comp(dert_, fid):  # skip odd derts for sparse rng+ comp: 1 skip / 1 add, to maintain 2x overlap

    sub_P_ = []   # prefix '_' denotes the prior of same-name variables, initialization:
    (__i, __short_bi_d, __short_bi_m, _), _, (_i, _short_bi_d, _short_bi_m, _) = dert_[0:3]
    _d = _i - __i
    if fid: _m = min(__i, _i) - ave_min;
    else:   _m = ave - abs(_d)  # no ave * rng: actual m and d value is cumulative?
    _bi_d = _d * 2 + __short_bi_d
    _bi_m = _m * 2 + __short_bi_m  # back-project _m and d
    # initialize P with dert_[0]:
    sub_P = _bi_m > 0, [], 1, __i, _bi_d, _bi_m, [(__i, _bi_d, _bi_m, None)], []  # sign, LL, L, I, D, M, dert_, sub_

    for n in range(4, len(dert_), 2):  # backward comp, ave | cumulative ders and filters?
        i, short_bi_d, short_bi_m = dert_[n][:3]  # shorter-rng dert
        d = i - _i
        if fid: m = min(i, _i) - ave_min  # match = min: magnitude of derived vars correlates with stability
        else:   m = ave - abs(d)  # inverse match: intensity doesn't correlate with stability
        bi_d = _d + d + _short_bi_d  # bilateral difference, accum in rng
        bi_m = _m + m + _short_bi_m  # bilateral match, accum in rng
        dert = _i, bi_d, bi_m, _d
        _i, _d, _m, _short_bi_d, _short_bi_m = i, d, m, short_bi_d, short_bi_d
        # P accumulation or termination:
        sub_P, sub_P_ = form_P(sub_P, sub_P_, dert)

    # terminate last sub_P in dert_:
    dert = _i, _d * 2, _m * 2, _d  # forward-project unilateral to bilateral d and m values
    sub_P, sub_P_ = form_P(sub_P, sub_P_, dert)
    sub_P_ += [sub_P]
    return sub_P_  # becomes lateral_sub_


def der_comp(dert_):  # cross-comp consecutive uni_ds in same-sign dert_: sign match is partial d match
    # dd and md may match across d sign, but likely in high-match area, spliced by spec in comp_P?

    sub_P_ = []   # initialization:
    (_, _, _, __i), (_, _, _, _i) = dert_[1:3]  # each prefix '_' denotes prior
    __i = abs(__i); _i = abs(_i)
    _d = _i - __i  # initial comp
    _m = min(__i, _i) - ave_min
    _bi_d = _d * 2  # __d and __m are back-projected as = _d or _m
    _bi_m = _m * 2
    # initialize P with dert_[1]:
    sub_P = _bi_m > 0, [], 1, __i, _bi_d, _bi_m, [(__i, _bi_d, _bi_m, None)], []  # sign, LL, L, I, D, M, dert_, sub_

    for dert in dert_[3:]:
        i = abs(dert[3])  # unilateral d in same-d-sign seg, no sign comp
        d = i - _i   # d is dd
        m = min(i, _i) - ave_min  # md = min: magnitude of derived vars corresponds to predictive value
        bi_d = _d + d  # bilateral d-difference per _i
        bi_m = _m + m  # bilateral d-match per _i
        dert = _i, bi_d, bi_m, _d
        _i, _d, _m = i, d, m
        # P accumulation or termination:
        sub_P, sub_P_ = form_P(sub_P, sub_P_, dert)
    # terminate last sub_P in dert_:
    dert = _i, _d * 2, _m * 2, _d  # forward-project unilateral to bilateral d and m values
    sub_P, sub_P_ = form_P(sub_P, sub_P_, dert)
    sub_P_ += [sub_P]
    return sub_P_  # becomes lateral_sub


def cross_comp(frame_of_pixels_):  # converts frame_of_pixels to frame_of_patterns, each pattern maybe nested

    Y, X = image.shape  # Y: frame height, X: frame width
    frame_of_patterns_ = []
    for y in range(ini_y + 1, Y):
        # initialization per row:
        pixel_ = frame_of_pixels_[y, :]  # y is index of new line pixel_
        P_ = []  # row of patterns
        __p, _p = pixel_[0:2]  # each prefix '_' denotes prior
        _d = _p - __p  # initial comp
        _m = ave - abs(_d)
        _bi_d = _d * 2  # __d and __m are back-projected as = _d or _m
        _bi_m = _m * 2
        if _bi_m > 0:
            if _bi_m > ave_m: sign = 0  # low variation
            else: sign = 1  # medium variation
        else: sign = 2  # high variation
        # initialize P with dert_[0]:
        P = sign, __p, _bi_d, _bi_m, [(__p, _bi_d, _bi_m, None)], []  # sign, I, D, M, dert_, sub_

        for p in pixel_[2:]:  # pixel p is compared to prior pixel _p in a row
            d = p - _p
            m = ave - abs(d)  # initial match is inverse deviation of |difference|
            bi_d = d + _d  # bilateral difference
            bi_m = m + _m  # bilateral match
            dert = _p, bi_d, bi_m, _d
            # accumulate or terminate mP: span of pixels forming same-sign m:
            P, P_ = form_P(P, P_, dert)
            _p, _d, _m = p, d, m  # uni_d is not used in comp
        # terminate last P in row:
        dert = _p, _d * 2, _m * 2, _d  # last d and m are forward-projected to bilateral values
        P, P_ = form_P(P, P_, dert)
        P_ += [P]
        # evaluate sub-recursion per P:
        P_ = intra_P(P_, fid=False, rdn=1, rng=1, sD=0)  # recursive
        frame_of_patterns_ += [P_]  # line of patterns is added to frame of patterns

    return frame_of_patterns_  # frame of patterns will be output to level 2

''' Recursion extends pattern structure to 1d hierarchy and then 2d hierarchy, to be adjusted by macro-feedback:
    P_:
    fid,   # flag: input is derived: magnitude correlates with predictive value: m = min-ave, else m = ave-|d|
    rdn,   # redundancy to higher layers, possibly lateral overlap of rng+, seg_d, der+, rdn += 1 * typ coef?
    rng,   # comp range, + frng | fder?  
    P:
    sign,  # ternary: 0 -> rng+, 1 -> segment_d, 2 -> der+ 
    Dert = I, D, M,  # L = len(dert_) * rng
    dert_, # input for sub_segment or extend_comp
           # conditional 1d array of next layer:
    sub_,  # seg_P_ from sub_segment or sub_P_ from extend_comp
           # conditional 2d array of deeper layers, each layer maps to higher layer for feedback:
    layer_,  # each layer has Dert and array of seg_|sub_s, each with crit and fid, nested to different depth
             # for layer-parallel access and comp, similar to frequency domain representation
    root_P   # reference for layer-sequential feedback 

    orders of composition: 1st: dert_, 2nd: seg_|sub_(derts), 3rd: P layer_(sub_P_(derts))? 
    line-wide layer-sequential recursion and feedback, for clarity and slice-mapped SIMD processing? 
'''

def form_P(P, P_, dert):  # initialization, accumulation, termination, recursion

    _sign, I, D, M, dert_, sub_ = P  # each sub in sub_ is nested to depth = sub_[n]
    p, d, m, uni_d = dert
    if m > 0:
        if m > ave_m: sign = 0  # low variation: eval comp rng+ per P, ternary sign
        else: sign = 1  # medium variation: segment P.dert_ by d sign
    else: sign = 2  # high variation: eval comp der+ per P

    if sign != _sign:  # sign change: terminate P
        P_.append(P)
        I, D, M, dert_ = 0, 0, 0, []  # reset accumulated params
    # accumulate params with bilateral values:
    I += p; D += d; M += m
    dert_ += [(p, d, m, uni_d)]  # uni_d for der_comp and segment
    P = sign, I, D, M, dert_, sub_  # sub_, layer_ accumulated in intra_P

    return P, P_


def intra_P(P_, fid, rdn, rng, sD):  # evaluate for sub-recursion in line P_, filling its sub_P_ with the results

    feedback = []
    if m > 0:
        if m > ave_m: sign = 0  # low variation: eval comp rng+ per P, ternary sign
        else: sign = 1  # medium variation: segment P.dert_ by d sign
    else: sign = 2  # high variation: eval comp der+ per P

    for sign, L, I, D, M, dert_, sub_ in P_:  # sub_: list of lower pattern layers, nested to depth = sub_[n]

        if sign == 0:  # low-variation P, rdn+1.2: 1 + (1 / ave_nP): rdn to higher derts + ave rdn to higher sub_
            if M > ave_M * rdn and L > 4:  # rng+ eval vs. fixed cost = ave_M
                rng += 1  # n of extensions, mapped comp rng = rng**2: 1, 2, 4.., kernel = rng * 2 + 1: 3, 5, 9
                lat_sub_ = rng_comp(dert_, fid)
                lL = len(lat_sub_)  # lateral L, for visibility
                rdn += 1 / lL - 0.2  # adjust distributed rdn, estimated in intra_P
                sub_ += [[( 0, lL, fid, rdn, rng, lat_sub_ )]]  # 1st layer, fork = 0, may differ from P sign
                sub_ += intra_P(lat_sub_, fid, rdn+1.2, rng, 0)  # recursion eval and deeper layers feedback

        elif sign == 1 and not sD:  # mid-variation P:
            if L > ave_L * rdn:  # low |M|, filter by L only
                lat_sub_, sD = segment(dert_)  # segment dert_ by ds: sign match covers variable cost?
                lL = len(lat_sub_); rdn += 1 / lL - 0.2
                sub_ += [[( 1, lL, True, rdn, rng, lat_sub_)]]  # 1st layer, fork = 1
                sub_ += intra_P(lat_sub_, True, rdn+1.2, rng, sD)  # will trigger fork 2:

        elif sign == 2 or sD:  # high-variation P or any seg_P? ds match is necessary?
            if sD: vD = sD  # called after segment(), or filter by L: n of ms?
            else:  vD = -M
            if (vD > ave_D * rdn) and len(dert_) > 4:  # der+ eval, full-P der_comp obviates seg_dP_
                lat_sub_ = der_comp(dert_)  # cross-comp uni_ds in dert[3]
                lL = len(lat_sub_); rdn += 1 / lL - 0.2
                sub_ += [[( 2, lL, True, rdn, rng, lat_sub_)]]  # 1st layer, fork = 2
                sub_ += intra_P(lat_sub_, True, rdn+1.2, rng, 0)  # deeper layers feedback

        # each: else merge non-selected sub_Ps within P, if in max recursion depth? Eval per P_: same op, !layer
        for i, sub in enumerate(sub_):
            if sub == []:
                break
            try_extend(feedback, i, sub)

    return feedback  # add return of Dert and hypers, same in sub_[0]? [] fill if min_nP: L, LL?


def segment(dert_):  # P segmentation by same d sign: initialization, accumulation, termination

    sub_ = []  # replaces P.sub_
    sub_D = 1  # bias to trigger 3rd fork in next intra_P
    _p, _d, _m, _uni_d = dert_[1]  # skip dert_[0]: no uni_d; prefix '_' denotes prior
    _sign = _uni_d > 0
    I =_p; D =_d; M =_m; seg_dert_= [(_p, _d, _m, _uni_d)]  # initialize seg_P, same as P

    for p, d, m, uni_d in dert_[2:]:
        sign = uni_d > 0
        if _sign != sign:
            sub_D += abs(D)
            sub_.append((_sign, I, D, M, seg_dert_, []))  # terminate seg_P, same as P
            I, D, M, dert_, = 0, 0, 0, []  # reset accumulated seg_P params
        _sign = sign
        I += p; D += d; M += m  # accumulate seg_P params, or D += uni_d?
        dert_.append((p, d, m, uni_d))
    sub_D += abs(D)
    sub_.append((_sign, I, D, M, seg_dert_, []))  # pack last segment, nothing to accumulate

    return sub_, sub_D  # replace P.sub_


def rng_comp(dert_, fid):  # sparse comp, 1 pruned dert / 1 extended dert to maintain 2x overlap

    sub_P_ = []  # replaces P.sub_; prefix '_' denotes the prior of same-name variables, initialization:
    (__i, __short_bi_d, __short_bi_m, _), _, (_i, _short_bi_d, _short_bi_m, _) = dert_[0:3]
    _d = _i - __i  # initial comp
    if fid: _m = min(__i, _i) - ave_m + __short_bi_m
    else:   _m = ave - abs(_d) + __short_bi_m
    _bi_d = _d * 2  # __d and __m are back-projected as = _d or _m
    _bi_m = _m * 2
    if _bi_m > 0:
        if _bi_m > ave_m: sign = 0  # low variation
        else: sign = 1  # medium variation
    else: sign = 2  # high variation
    # initialize P with dert_[0]:
    sub_P = sign, __i, _bi_d, _bi_m, [(__i, _bi_d, _bi_m, None)], []  # sign, I, D, M, dert_, sub_

    for n in range(4, len(dert_), 2):  # backward comp, skip 1 dert to maintain overlap rate, that defines ave
        i, short_bi_d, short_bi_m = dert_[n][:3]  # shorter-rng dert
        d = i - _i
        if fid:  # match = min: magnitude of derived vars correlates with stability
            m = min(i, _i) - ave_m + _short_bi_m   # m accum / i number of comps
        else:  # inverse match: intensity doesn't correlate with stability
            m = ave - abs(d) + _short_bi_m
        d += _short_bi_d  # _d and _m combine bi_d | bi_m at rng-1
        bi_d = _d + d  # bilateral difference, accum in rng
        bi_m = _m + m  # bilateral match, accum in rng
        dert = _i, bi_d, bi_m, _d
        _i, _d, _m, _short_bi_d, _short_bi_m = i, d, m, short_bi_d, short_bi_d
        # P accumulation or termination:
        sub_P, sub_P_ = form_P(sub_P, sub_P_, dert)

    # terminate last sub_P in dert_:
    dert = _i, _d * 2, _m * 2, _d  # forward-project unilateral to bilateral d and m values
    sub_P, sub_P_ = form_P(sub_P, sub_P_, dert)
    sub_P_ += [sub_P]
    return sub_P_  # replaces P.sub_


def der_comp(dert_):  # comp of consecutive uni_ds in dert_, dd and md may match across d sign

    sub_P_ = []  # return to replace P.sub_, initialization:
    (_, _, _, __i), (_, _, _, _i) = dert_[1:3]  # each prefix '_' denotes prior
    _d = _i - __i  # initial comp
    _m = min(__i, _i) - ave_m
    _bi_d = _d * 2  # __d and __m are back-projected as = _d or _m
    _bi_m = _m * 2
    if _bi_m > 0:
        if _bi_m > ave_m: sign = 0
        else: sign = 1
    else: sign = 2
    # initialize P with dert_[1]:
    sub_P = sign, __i, _bi_d, _bi_m, [(__i, _bi_d, _bi_m, None)], []  # sign, I, D, M, dert_, sub_

    for dert in dert_[3:]:
        i = dert[3]  # unilateral d
        d = i - _i   # d is dd
        m = min(i, _i) - ave_m  # md = min: magnitude of derived vars corresponds to predictive value
        bi_d = _d + d  # bilateral d-difference per _i
        bi_m = _m + m  # bilateral d-match per _i
        dert = _i, bi_d, bi_m, _d
        _i, _d, _m = i, d, m
        # P accumulation or termination:
        sub_P, sub_P_ = form_P(sub_P, sub_P_, dert)

    # terminate last sub_P in dert_:
    dert = _i, _d * 2, _m * 2, _d  # forward-project unilateral to bilateral d and m values
    sub_P, sub_P_ = form_P(sub_P, sub_P_, dert)
    sub_P_ += [sub_P]
    return sub_P_  # replaces P.sub_

'''
no ave * rdn, bi_m ave_m * rng-2: comp cost is separate from mP definition for comp_P

def intra_P(P_, fid, rdn, rng):  # evaluate for sub-recursion in line P_, filling sub_ | seg_, len(sub_)- adjusted rdn?

    for n, (_, _, _, _, _, _, layer_) in enumerate(P_):  # pack fid, rdn, rng in P_?
        _, dert__ = layer_[-1]  # Dert and derts of newly formed layer

        for _sign, I, D, M, dert_ in dert__:  # dert_s of all sub_Ps in a layer
            if _sign == 0:  # low-variation P: segment by mm, segment(ds) eval/ -mm seg, extend_comp(rng) eval/ +mm seg:
                
                if M > ave_M * rdn and len(dert_) > 2:  # ave * rdn is fixed cost of comp_rng forming len(sub_P_) = ave_nP
                    ssub_, rdn = extend_comp(dert_, True, fid, rdn + 1, rng)  # frng=1
                    seg_[n][6][2][:], rdn = intra_P(ssub_, fid, rdn + 1, rng)  # eval per seg_, sub_ = frng=0, fid=1, sub_

                if len(dert_) > ave_Lm * rdn:  # fixed costs of new P_ are translated to ave L
                    sub_, rdn = sub_segment(dert_, True, fid, rdn+1, rng)
                    P_[n][5][2][:], rdn = intra_seg(sub_, True, fid, rdn+1, rng)  # eval per seg_P, P.seg_ = fmm=1, fid, seg_
            else:
                if -M > ave_D * rdn and len(dert_) > rng * 2:  # -M > fixed costs of full-P comp_d -> d_mP_, obviates seg_dP_
                    fid = True
                    sub_, rdn = extend_comp(dert_, False, fid, rdn+1, rng=1)
                    P_[n][5][2][:], rdn = intra_P(sub_, fid, rdn+1, rng)  # eval per sub_P, P.sub_ = frng=0, fid=1, sub_

                # else merge short sub_Ps between Ps, if allowed by max recursion depth?

    intra_P(P_, fid, rdn, rng)  # recursion eval per new layer, in P_.layer_[-1]: array of mixed seg_s and sub_s
    return P_

def intra_seg(seg_, fmm, fid, rdn, rng):  # evaluate for sub-recursion in P_, filling sub_ | seg_, use adjusted rdn

    for n, (sign, I, D, M, dert_, sseg_, ssub_) in enumerate(seg_):
        if fmm:  # comp i over rng incremented as 2**n: 1, 2, 4, 8:

            if M > ave_M * rdn and len(dert_) > 2:  # ave * rdn is fixed cost of comp_rng forming len(sub_P_) = ave_nP
                ssub_, rdn = extend_comp(dert_, True, fid, rdn+1, rng)  # frng=1
                seg_[n][6][2][:], rdn = intra_P(ssub_, fid, rdn+1, rng)  # eval per seg_, sub_ = frng=0, fid=1, sub_

        elif len(dert_) > ave_Ld * rdn:  # sub-segment by d sign, der+ eval per seg, merge neg segs into nSeg for rng+?
            fid = True
            sseg_, rdn = sub_segment(dert_, False, fid, rdn+1, rng)  # d-sign seg_ = fmm=0, fid, seg_
            for sign_d, Id, Dd, Md, dert_d_, seg_d_, sub_d_ in sseg_[2]:  # seg_P in sub_seg_

                if Dd > ave_D * rdn and len(dert_) > 2:  # D accumulated in same-d-sign segment may be higher that P.D
                    sub_d_, rdn = extend_comp(dert_, False, fid, rdn+1, rng)  # frng = 0, fid = 1: cross-comp d

    layer cycle or mixed-fork: breadth-first beyond same-fork sub_?
    rng+, der+, seg_d( der+ | merge-> rng+): fixed cycle? 
    also merge initial non-selected rng+ | der+?
    sseg__ += [intra_seg(sseg_, False, fid, rdn, rng)]  # line-wide for caching only?
    return seg_

def sub_segment(P_dert_, fmm, fid, rdn, rng):  # mP segmentation by mm or d sign: initialization, accumulation, termination

    P_seg_ = []  # replaces P.seg_
    _p, _d, _m, _uni_d = P_dert_[0]  # prefix '_' denotes prior
    if fmm: _sign = _m - ave > 0  # flag: segmentation criterion is sign of mm, else sign of uni_d
    else:   _sign = _uni_d > 0
    I =_p; D =_d; M =_m; dert_= [(_p, _d, _m, _uni_d)]; seg_= []; sub_= []; layer_=[]  # initialize seg_P, same as P

    for p, d, m, uni_d in P_dert_[1:]:
        if fmm:
            sign = m - ave > 0  # segmentation crit = mm sign
        else:
            sign = uni_d > 0  # segmentation crit = uni_d sign
        if _sign != sign:
            seg_.append((_sign, I, D, M, dert_, seg_, sub_, layer_))  # terminate seg_P, same as P
            I, D, M, dert_, seg_, sub_ = 0, 0, 0, [], [], []  # reset accumulated seg_P params
        _sign = sign
        I += p; D += d; M += m; dert_.append((p, d, m, uni_d))  # accumulate seg_P params, not uni_d

    P_seg_.append((_sign, I, D, M, dert_, seg_, sub_, layer_))  # pack last segment, nothing to accumulate
    rdn *= len(P_seg_) / ave_nP  # cost per seg?
    intra_seg(P_seg_, fmm, fid, rdn+1, rng)  # evaluate for sub-recursion, different fork of intra_P, pass fmm?

    return (fmm, fid, P_seg_), rdn  # replace P.seg_, rdn
'''