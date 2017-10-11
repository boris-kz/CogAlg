from scipy import misc

'''
Level 1:

Cross-comparison between consecutive pixels within horizontal scan line (row).
Resulting difference patterns dPs (spans of pixels forming same-sign differences)
and relative match patterns vPs (spans of pixels forming same-sign predictive value)
are redundant representations of each line of pixels.

I don't like to pack arguments, this code is optimized for visibility rather than speed 
'''

def range_incr(a, aV, aD, A, AV, AD, r, t_):

    A += a  # A, AV, AD incr. to adjust for redundancy to patterns formed by prior comp:
    AV += aV  # aV: min V for initial comp(t_), AV: min V for higher recursions
    AD += aD  # aD: min |D| for initial comp(d_), AD: min |D| for higher recursions

    X = len(t_)  # initialized t_ is inside new vP only

    olp, vP_, dP_ = 0, [], []   # olp is common for both:
    vP = 0, 0, 0, 0, 0, [], []  # pri_s, I, D, V, rv, t_, olp_
    dP = 0, 0, 0, 0, 0, [], []  # pri_sd, Id, Dd, Vd, rd, d_, dolp_

    for x in range(r+1, X):

        p, fd, fv = t_[x]       # compared to a pixel at x-r-1:
        pp, pfd, pfv = t_[x-r]  # previously compared p, its fd, fv to next p

        fv += pfv  # fuzzy v is summed over extended-comp range
        fd += pfd  # fuzzy d is summed over extended-comp range

        pfv += fv  # x-r is complete, while current fv and fd are not?
        pfd += fd  # bilateral accum, then form() of two completes:

        pri_p, pri_fd, pri_fv = t_[x-r-1]  # for comp(p, pri_p):

        olp, fd, fv, vP, dP, vP_, dP = \
        re_comp(x, p, pri_p, fd, fv, olp, vP, dP, vP_, dP_, X, a, aV, aD, A, AV, AD, r)

    return vP_, dP_  # local vPs and dPs replace t_


def deriv_incr(a, aV, aD, A, AV, AD, r, d_):

    A += a; AV += aV; AD += aD  # no deriv_incr while r < min_r, only more fuzzy?

    X = len(d_)  # not tuples even within range_incr()?
    id_ = d_  # to differentiate from initialized d_:

    fd, fv, olp, vP_, dP_ = 0, 0, 0, [], []  # fuzzy values initialized per derivation order
    vP = 0, 0, 0, 0, 0, [], []  # pri_s, I, D, V, rv, t_, olp_
    dP = 0, 0, 0, 0, 0, [], []  # pri_sd, Id, Dd, Vd, rd, d_, dolp_

    pri_p = id_[0]

    for x in range(1, X):
        p = id_[x]

        olp, fd, fv, vP, dP, vP_, dP = \
        re_comp(x, p, pri_p, fd, fv, olp, vP, dP, vP_, dP_, X, a, aV, aD, A, AV, AD, r)

        pri_p = p

    return vP_, dP_  # local vPs and dPs replace d_


def form_P(typ, P, alt_P, P_, alt_P_, olp, pri_p, fd, fv, x, X,  # input variables
         a, aV, aD, A, AV, AD, r):  # filter variables

    if typ:
        s = 1 if fv >= 0 else 0  # sign of fd, 0 is positive?
    else:
        s = 1 if fd >= 0 else 0  # sign of fv, 0 is positive?

    pri_s, I, D, V, rf, e_, olp_ = P

    if x > r + 2 and (s != pri_s or x == X - 1):  # if derived pri_s miss, P is terminated

        if typ:
            if len(e_) > r + 3 and pri_s == 1 and V > AV:  # min 3 comp over extended distance within p_:
                r += 1  # comp range counter
                rf = 1  # incr range flag
                e_.append(range_incr(a, aV, aD, A, AV, AD, r, e_))
        else:
            if len(e_) > 3 and abs(D) > AD:  # min 3 comp within d_:
                rf = 1  # incr derivation flag
                e_.append(deriv_incr(a, aV, aD, A, AV, AD, r, e_))

        P = pri_s, I, D, V, rf, e_, olp_
        P_.append(P)  # output

        o = len(P_), olp  # index of current P and terminated olp are buffered in alt_olp_
        alt_P[6].append(o)
        o = len(alt_P_), olp  # index of current alt_P and terminated olp are buffered in olp_
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

    return olp, P, alt_P, P_, alt_P_  # alt_ and _alt_ accumulated per line


def re_comp(x, p, pri_p, fd, fv, olp, vP, dP, vP_, dP_,  # inputs and output patterns
            X, a, aV, aD, A, AV, AD, r):  # filters      # recursive comp

    d = p - pri_p      # difference between consecutive pixels
    m = min(p, pri_p)  # match between consecutive pixels
    v = m - A          # relative match (predictive value) between consecutive pixels

    fd += d  # fuzzy d accumulates ds between p and all prior ps within min_r, via range_incr()
    fv += v  # fuzzy v accumulates vs between p and all prior ps within min_r, via range_incr()

    # fv and fd at r < min_r are lost, full fv and fd are different for p and pri_p
    # formation of value pattern vP: span of pixels forming same-sign fv s:

    olp, vP, dP, vP_, dP_ = \
    form_P(1, vP, vP_, dP, dP_, olp, pri_p, fd, fv, x, X, a, aV, aD, A, AV, AD, r)

    # formation of difference pattern dP: span of pixels forming same-sign fd s:

    olp, dP, vP, dP_, vP_ = \
    form_P(0, dP, dP_, vP, vP_, olp, pri_p, fd, fv, x, X, a, aV, aD, A, AV, AD, r)

    olp += 1  # overlap between concurrent vP and dP, to be buffered in olp_

    return olp, fd, fv, vP, dP, vP_, dP  # for next p comparison, vP and dP increment, and output


def comp(x, p, it_, olp, vP, dP, vP_, dP_,  # inputs and output patterns
         X, a, aV, aD, min_r, A, AV, AD, r):  # filters

    # comparison of consecutive pixels within line forms tuples: pixel, match, difference

    for it in it_:  # incomplete tuples with summation range from 0 to rng
        pri_p, fd, fm = it

        d = p - pri_p  # difference between pixels
        m = min(p, pri_p)  # match between pixels

        fd += d  # fuzzy d: sum of ds between p and all prior ps within it_
        fm += m  # fuzzy m: sum of ms between p and all prior ps within it_

    if len(it_) == min_r:

        fv = fm - A; del it_[0]  # completed tuple is removed from it_
        # formation of value pattern vP: span of pixels forming same-sign fv s:

        olp, vP, dP, vP_, dP_ = \
        form_P(1, vP, vP_, dP, dP_, olp, pri_p, fd, fv, x, X, a, aV, aD, A, AV, AD, r)

        # formation of difference pattern dP: span of pixels forming same-sign fd s:

        olp, dP, vP, dP_, vP_ = \
        form_P(0, dP, dP_, vP, vP_, olp, pri_p, fd, fv, x, X, a, aV, aD, A, AV, AD, r)

        olp += 1  # overlap between concurrent vP and dP, to be represented in both?

    it = p, fd, fm
    it_.append(it)  # new prior tuple

    return it_, olp, vP, dP, vP_, dP  # for next p comparison, vP and dP increment, and output


def level_1(Fp_):  # last '_' distinguishes array name from element name

    FP_ = []  # output frame of vPs: relative match patterns, and dPs: difference patterns
    Y, X = Fp_.shape  # Y: frame height, X: frame width

    a = 127  # minimal filter for vP inclusion
    aV = 63  # minimal filter for incremental-range comp
    aD = 63  # minimal filter for incremental-derivation comp
    min_r=1  # initial fuzzy comparison range

    for y in range(Y):

        p_ = Fp_[y, :]   # y is index of new line ip_
        A = a * min_r    # initial filters, incremented with r
        AV = aV * min_r  # min V for initial comp(t_)
        AD = aD * min_r  # min |D| for initial comp(d_)

        r, x, olp, vP_, dP_ = 0, 0, 0, [], []  # initialized at each line
        vP = 0, 0, 0, 0, 0, [], []  # pri_s, I, D, V, rv, t_, olp_
        dP = 0, 0, 0, 0, 0, [], []  # pri_sd, Id, Dd, Vd, rd, d_, dolp_

        it_ = []  # incomplete fuzzy tuples: summation range < rng
        pri_t = p_[0], 0, 0  # no d, m at x = 0
        it_.append(pri_t)

        for x in range(1, X):  # cross-compares consecutive pixels
            p = p_[x]  # new pixel, fuzzy comp to it_ for r <= min_r:

            it_, olp, vP, dP, vP_, dP_ = \
            comp(x, p, it_, olp, vP, dP, vP_, dP_,  # inputs and output patterns
                 X, a, aV, aD, min_r, A, AV, AD, r)  # filters

        LP_ = vP_, dP_
        FP_.append(LP_)  # line of patterns is added to frame of patterns, y = len(FP_)

    return FP_  # output to level 2

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
level_1(f)

# at vP term: print ('type', 0, 'pri_s', pri_s, 'I', I, 'D', D, 'V', V, 'rv', rv, 'p_', p_)
# at dP term: print ('type', 1, 'pri_sd', pri_sd, 'Id', Id, 'Dd', Dd, 'Vd', Vd, 'rd', rd, 'd_', d_)

