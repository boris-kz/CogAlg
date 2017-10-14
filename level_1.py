from scipy import misc

'''
Level 1:

Cross-comparison between consecutive pixels within horizontal scan level (row).
Resulting difference patterns dPs (spans of pixels forming same-sign differences)
and relative match patterns vPs (spans of pixels forming same-sign predictive value)
are redundant representations of each level of pixels.

I don't like to pack arguments, this code is optimized for visibility rather than speed 
'''


def pre_comp(typ, e_, A, r):  # prepares for comp recursion

    A += a * 2  # to compensate for redundancy + fv accumulation?
    X = len(e_)  # initialized t_ is inside new vP only

    olp, vP_, dP_ = 0, [], []   # olp is common for both:
    vP = 0, 0, 0, 0, 0, [], []  # pri_s, I, D, V, rv, t_, olp_
    dP = 0, 0, 0, 0, 0, [], []  # pri_sd, Id, Dd, Vd, rd, d_, dolp_

    if typ:  # comparison range incr within e_ = t_

        r += 1  # comp range counter, recorded within new Ps
        for x in range(r+1, X):

            p, ifd, ifv = e_[x]  # ifd, ifv not used: directional accum for pri_p only
            pri_p, fd, fv = e_[x-r]  # for comparison of r-pixel-distant pixels:

            fd, fv, vP, dP, vP_, dP_, olp = \
            re_comp(x, p, pri_p, fd, fv, vP, dP, vP_, dP_, olp, X, A, r)

    else:  # comparison derivation incr within e_ = d_: not tuples within range incr?

        pri_d = e_[0]  # no deriv_incr while r < min_r, only more fuzzy?
        fd, fv = 0, 0

        for x in range(1, X):
            d = e_[x]

            fd, fv, vP, dP, vP_, dP_, olp = \
            re_comp(x, d, pri_d, fd, fv, vP, dP, vP_, dP_, olp, X, A, r)

            pri_d = d

    return vP_, dP_  # local vP_ + dP_ replaces t_ or d_


def form_P(typ, P, alt_P, P_, alt_P_, olp, pri_p, fd, fv, x, X, A, r):

    if typ: s = 1 if fv >= 0 else 0  # sign of fd, 0 is positive?
    else:   s = 1 if fd >= 0 else 0  # sign of fv, 0 is positive?

    pri_s, I, D, V, rf, e_, olp_ = P  # debug: 0 values in P?

    if x > r + 2 and (s != pri_s or x == X - 1):  # P is terminated and evaluated

        if typ:
            if len(e_) > r + 3 and pri_s == 1 and V > A + aV:  # minimum of 3 tuples:
                rf = 1  # incr range flag
                e_.append(pre_comp(1, e_, A, r))  # comparison range incr within e_ = t_

        else:
            if len(e_) > 3 and abs(D) > A + aD:  # minimum of 3 ds:
                rf = 1  # incr derivation flag
                e_.append(pre_comp(0, e_, A, r))  # comp derivation incr within e_ = d_

        P = pri_s, I, D, V, rf, e_, olp_
        P_.append(P)  # output

        o = len(P_), olp  # index of current P and terminated olp are buffered in alt_olp_
        alt_P[6].append(o)
        o = len(alt_P_), olp  # index of current alt_P and terminated olp buffered in olp_
        olp_.append(o)

        olp, I, D, V, rf, e_, olp_ = 0, 0, 0, 0, 0, [], []  # initialized P and olp

    pri_s = s   # vP (span of pixels forming same-sign v) is incremented:
    I += pri_p  # ps summed within vP
    D += fd     # fuzzy ds summed within vP
    V += fv     # fuzzy vs summed within vP

    if typ:
        t = pri_p, fd, fv  # inputs for inc_rng comp are tuples, vs. pixels for initial comp
        e_.append(t)
    else:
        e_.append(fd)  # prior fds of the same sign are buffered within dP

    P = pri_s, I, D, V, rf, e_, olp_

    return P, alt_P, P_, alt_P_, olp  # alt_ and _alt_ accumulated per level


def re_comp(x, p, pri_p, fd, fv, vP, dP, vP_, dP_, olp, X, A, r):  # recursive comp

    d = p - pri_p      # difference between consecutive pixels
    m = min(p, pri_p)  # match between consecutive pixels
    v = m - A          # relative match (predictive value) between consecutive pixels

    fd += d  # fuzzy d accumulates ds between p and all prior ps in r via range_incr()
    fv += v  # fuzzy v; lower-r fv and fd are in lower Ps, different for p and pri_p

    # formation of value pattern vP: span of pixels forming same-sign fv s:

    vP, dP, vP_, dP_, olp = \
    form_P(1, vP, dP, vP_, dP_, olp, pri_p, fd, fv, x, X, A, r)

    # formation of difference pattern dP: span of pixels forming same-sign fd s:

    dP, vP, dP_, vP_, olp = \
    form_P(0, dP, vP, dP_, vP_, olp, pri_p, fd, fv, x, X, A, r)

    olp += 1  # overlap between concurrent vP and dP, to be buffered in olp_

    return fd, fv, vP, dP, vP_, dP_, olp  # for next-p comp, vP and dP increment, output


def comp(x, p, it_, vP, dP, vP_, dP_, olp, X, A, r):

    # comparison of consecutive pixels within line forms tuples: pixel, difference, match

    for it in it_:  # incomplete tuples with summation range from 0 to r
        pri_p, fd, fm = it

        d = p - pri_p  # difference between pixels
        m = min(p, pri_p)  # match between pixels

        fd += d  # fuzzy d: sum of ds between p and all prior ps within it_
        fm += m  # fuzzy m: sum of ms between p and all prior ps within it_

    if len(it_) == r:  # current tuple fd and fm are accumulated over range = r
        fv = fm - A

        #  formation of value pattern vP: span of pixels forming same-sign fv s:

        vP, dP, vP_, dP_, olp = \
        form_P(1, vP, dP, vP_, dP_, olp, pri_p, fd, fv, x, X, A, r)

        # formation of difference pattern dP: span of pixels forming same-sign fd s:

        dP, vP, dP_, vP_, olp = \
        form_P(0, dP, vP, dP_, vP_, olp, pri_p, fd, fv, x, X, A, r)

        olp += 1  # overlap between vP and dP, represented in both and terminated with either
        del it_[0]  # complete-r tuple is removed

    it = p, 0, 0  # fd and fm are directional: initialized at 0 for a new tuple
    it_.append(it)  # new tuple is added

    return it_, vP, dP, vP_, dP_, olp  # for next-p comparison, vP and dP increment, output


def root_1D(Fp_):  # last '_' distinguishes array name from element name

    FP_ = []  # output frame of vPs: relative-match patterns, and dPs: difference patterns
    Y, X = Fp_.shape  # Y: frame height, X: frame width
    min_r = 2  # min fuzzy comparison range

    global a; a = 63
    global aV; aV = 63 * min_r  # min V for initial incremental-range comp(t_)
    global aD; aD = 63 * min_r  # min |D| for initial incremental-derivation comp(d_)

    A = a * min_r  # initial min match for positive vP inclusion, += a per recursion

    for y in range(Y):
        p_ = Fp_[y, :]   # y is index of new line p_

        r, x, olp, vP_, dP_ = min_r, 0, 0, [], []  # initialized at each level
        vP = 0, 0, 0, 0, 0, [], []  # pri_s, I, D, V, rv, t_, olp_
        dP = 0, 0, 0, 0, 0, [], []  # pri_sd, Id, Dd, Vd, rd, d_, dolp_

        it_ = []  # incomplete fuzzy tuples: summation range < rng
        pri_t = p_[0], 0, 0  # no d, m at x = 0
        it_.append(pri_t)

        for x in range(1, X):  # cross-compares consecutive pixels
            p = p_[x]  # new pixel, fuzzy comp to it_ for r <= min_r:

            it_, vP, dP, vP_, dP_, olp = \
            comp(x, p, it_, vP, dP, vP_, dP_, olp, X, A, r)

        LP_ = vP_, dP_
        FP_.append(LP_)  # level of patterns is added to frame of patterns, y = len(FP_)

    return FP_  # output to level 2

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
root_1D(f)

# at vP term: print ('type', 0, 'pri_s', pri_s, 'I', I, 'D', D, 'V', V, 'rv', rv, 'p_', p_)
# at dP term: print ('type', 1, 'pri_sd', pri_sd, 'Id', Id, 'Dd', Dd, 'Vd', Vd, 'rd', rd, 'd_', d_)

