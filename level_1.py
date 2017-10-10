from scipy import misc

'''
Level 1:

Cross-comparison between consecutive pixels within horizontal scan line (row).
Resulting difference patterns dPs (spans of pixels forming same-sign differences)
and relative match patterns vPs (spans of pixels forming same-sign predictive value)
are redundant representations of each line of pixels.

I don't like to pack arguments, this code is optimized for visibility rather than speed 
'''

def range_incr(a, aV, aD, min_r, A, AV, AD, r, t_):

    A += a  # A, AV, AD incr. to adjust for redundancy to patterns formed by prior comp:
    AV += aV  # aV: min V for initial comp(t_), AV: min V for higher recursions
    AD += aD  # aD: min |D| for initial comp(d_), AD: min |D| for higher recursions

    X = len(t_)
    it_ = t_  # to differentiate from initialized t_:

    olp, vP_, dP_ = 0, [], []   # olp is common for both:
    vP = 0, 0, 0, 0, 0, [], []  # pri_s, I, D, V, rv, t_, olp_
    dP = 0, 0, 0, 0, 0, [], []  # pri_sd, Id, Dd, Vd, rd, d_, dolp_

    for x in range(r+1, X):

        p, fd, fv = it_[x]       # compared to a pixel at x-r-1:
        pp, pfd, pfv = it_[x-r]  # previously compared p(ignored), its fd, fv to next p

        fv += pfv  # fuzzy v is summed over extended-comp range
        fd += pfd  # fuzzy d is summed over extended-comp range

        pfv += fv  # for pp, complete, while current fv and fd are not?
        pfd += fd  # bilateral accum, then form() of two completes:

        pri_p, pri_fd, pri_fv = it_[x-r-1]  # for comp(p, pri_p):

        olp, vP, dP, vP_, dP = \
        comp(p, pri_p, fd, fv, olp, vP, dP, vP_, dP_, x, X, a, aV, aD, min_r, A, AV, AD, r)

    return vP_, dP_  # local vPs and dPs replace t_


def deriv_incr(a, aV, aD, min_r, A, AV, AD, r, d_):

    A += a; AV += aV; AD += aD  # no deriv_incr while r < min_r, only more fuzzy?

    X = len(d_)  # d_= tuples if deriv_incr() within range_incr()? or still ds?
    id_ = d_  # to differentiate from initialized d_:

    fd, fv, olp, vP_, dP_ = 0, 0, 0, [], []  # fuzzy values initialized per derivation order
    vP = 0, 0, 0, 0, 0, [], []  # pri_s, I, D, V, rv, t_, olp_
    dP = 0, 0, 0, 0, 0, [], []  # pri_sd, Id, Dd, Vd, rd, d_, dolp_

    pri_p = id_[0]

    for x in range(1, X):
        p = id_[x]

        olp, vP, dP, vP_, dP = \
        comp(p, pri_p, fd, fv, olp, vP, dP, vP_, dP_, x, X, a, aV, aD, min_r, A, AV, AD, r)

        pri_p = p

    return vP_, dP_  # local vPs and dPs replace d_


def form_P(typ, P_, P, alt_P, olp, pri_p, fd, fv, x, X,  # input variables
         a, aV, aD, min_r, A, AV, AD, r):  # filter variables

    if typ:
        pri_s, I, D, V, rf, e_, olp_, alt_olp_ = P
        s = 1 if fv >= 0 else 0  # sign of fd, 0 is positive?
    else:
        pri_s, I, D, V, rf, e_, olp_, alt_olp_ = alt_P
        s = 1 if fd >= 0 else 0  # sign of fv, 0 is positive?

    if x > r + 2 and (s != pri_s or x == X - 1):  # if derived pri_s miss, P is terminated

        if typ:
            if len(e_) > r + 3 and pri_s == 1 and V > AV:  # min 3 comp over extended distance within p_:
                r += 1  # comp range counter
                rf = 1  # incr range flag
                e_.append(range_incr(a, aV, aD, min_r, A, AV, AD, r, e_))

        else:
            if len(e_) > 3 and abs(D) > AD:  # min 3 comp within d_:
                rf = 1  # incr derivation flag
                e_.append(deriv_incr(a, aV, aD, min_r, A, AV, AD, r, e_))

        p = I / len(e_); d = D / len(e_); v = V / len(e_)  # default to eval overlap, poss. div.comp
        P = pri_s, p, I, d, D, v, V, rf, e_, olp_
        P_.append(P)  # output of P, related to P_ by overlap only, no distant comp till level 3

        o = len(P_), olp  # len(P_) is index of current P
        alt_olp_.append(o)  # indexes of overlapping vPs and olp are buffered at current alt_P

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

    P = pri_s, I, D, V, rf, e_, olp_, alt_olp_

    return olp, P, alt_P, P_  # alt_ and _alt_ accumulated per line


def comp(p, pri_p, fd, fv, olp, vP, dP, vP_, dP_, x, X,  # input variables
        a, aV, aD, min_r, A, AV, AD, r):  # filter variables and output patterns

    d = p - pri_p      # difference between consecutive pixels
    m = min(p, pri_p)  # match between consecutive pixels
    v = m - A          # relative match (predictive value) between consecutive pixels

    fd += d  # fuzzy d accumulates ds between p and all prior ps within min_r, via range_incr()
    fv += v  # fuzzy v accumulates vs between p and all prior ps within min_r, via range_incr()

    # fv and fd at r < min_r are lost, full fv and fd are different for p and pri_p
    # formation of value pattern vP: span of pixels forming same-sign fv s:

    olp, vP, dP, vP_ = \
    form_P(1, vP_, vP, dP, olp, pri_p, fd, fv, x, X, a, aV, aD, min_r, A, AV, AD, r)

    # formation of difference pattern dP: span of pixels forming same-sign fd s:

    olp, dP, vP, dP_ = \
    form_P(0, dP_, dP, vP, olp, pri_p, fd, fv, x, X, a, aV, aD, min_r, A, AV, AD, r)

    olp += 1  # overlap between concurrent vP and dP

    return olp, vP, dP, vP_, dP  # for next p comparison, vP and dP increment, and output


def ini_comp(
        p, pri_p, fd, olp, x,  # input variables
        pri_s, I, D, V, rv, t_, olp_,  # vP variables
        pri_sd, Id, Dd, Vd, rd, d_, dolp_,  # dP variables
        a, aV, aD, min_r, A, AV, AD, r, X, vP_, dP_):  # filters and outputs

    # comparison of consecutive pixels within line forms tuples: pixel, match, difference

    it_ = []  # incomplete fuzzy tuples: summation range < rng
    d,m = 0,0  # no d, m at x = 0

    for it in it_:  # incomplete tuples with summation range from 0 to rng
        pri_p, fd, fm = it

        d = p - pri_p  # difference between pixels
        m = min(p, pri_p)  # match between pixels

        fd += d  # fuzzy d: sum of ds between p and all prior ps within it_
        fm += m  # fuzzy m: sum of ms between p and all prior ps within it_

    if len(it_) == min_r + 1:

        del it_[0]  # completed tuple is transferred from it_ to form_P
        fv = fm - A

        s = 1 if fv >= 0 else 0  # sign of fv, 0 is positive?
        if x > r + 2 and (s != pri_s or x == X - 1):  # if derived pri_s miss, vP is terminated

            if len(t_) > r + 3 and pri_s == 1 and V > AV:  # min 3 comp over extended distance within t_:

                r += 1  # r: incremental range-of-comp counter
                rv = 1  # rv: incremental range flag
                t_.append(range_incr(a, aV, aD, min_r, A, AV, AD, r, t_))

            p = I / len(t_); d = D / len(t_); v = V / len(t_)  # default to eval overlap, poss. div.comp
            vP = pri_s, p, I, d, D, v, V, rv, t_, olp_
            vP_.append(vP)  # output of vP, related to dP_ by overlap only, no distant comp till level 3

            o = len(vP_), olp  # len(P_) is index of current vP
            dolp_.append(o)  # indexes of overlapping vPs and olp are buffered at current dP

            olp, I, D, V, rv, t_, olp_ = 0, 0, 0, 0, 0, [], []  # initialized olp and vP

        pri_s = s  # vP (span of pixels forming same-sign v) is incremented:
        I += pri_p  # ps summed within vP
        D += fd  # fuzzy ds summed within vP
        V += fv  # fuzzy vs summed within vP
        t = pri_p, fd, fv  # inputs for inc_rng comp are tuples, vs. pixels for initial comp
        t_.append(t)  # tuples (pri_p, fd, fv) are buffered within each vP

        # formation of difference pattern dP: span of pixels forming same-sign fd s:

        sd = 1 if fd >= 0 else 0  # sd: positive sign of fd
        if x > r + 2 and (sd != pri_sd or x == X - 1):  # if derived pri_sd miss, dP is terminated

            if len(d_) > 3 and abs(Dd) > AD:  # min 3 comp within d_:

                rd = 1  # rd: incremental derivation flag:
                d_.append(deriv_incr(a, aV, aD, min_r, A, AV, AD, r, d_))

            pd = Id / len(d_); dd = Dd / len(d_); vd = Vd / len(d_)  # to evaluate olp Ps directly
            dP = pri_sd, pd, Id, dd, Dd, vd, Vd, rd, d_, dolp_
            dP_.append(dP)  # output of dP

            o = len(dP_), olp  # len(P_) is index of current dP
            olp_.append(o)  # indexes of overlapping dPs and olps are buffered at current vP

            olp, Id, Dd, Vd, rd, d_, dolp_ = 0, 0, 0, 0, 0, [], []  # initialized olp and dP

        pri_sd = sd  # dP (span of pixels forming same-sign d) is incremented:
        Id += pri_p  # ps summed within dP
        Dd += fd  # fuzzy ds summed within dP
        Vd += fv  # fuzzy vs summed within dP
        d_.append(fd)  # prior fds of the same sign are buffered within dP

        olp += 1  # shared overlap between concurrent vP and dP

    it = p, d, m
    it_.append(it)  # new prior tuple

    vP = pri_s, I, D, V, rv, t_, olp_, dolp_
    dP = pri_sd, Id, Dd, Vd, rd, d_, dolp_, olp_

    return olp, vP, dP, vP_, dP_


def level_1(Fp_):  # last '_' distinguishes array name from element name

    FP_ = []  # output frame of vPs: relative match patterns, and dPs: difference patterns
    Y, X = Fp_.shape  # Y: frame height, X: frame width

    a = 127  # minimal filter for vP inclusion
    aV = 63  # minimal filter for incremental-range comp
    aD = 63  # minimal filter for incremental-derivation comp
    min_r=0  # default range of fuzzy comparison, initially 0

    for y in range(Y):

        p_ = Fp_[y, :]  # y is index of new line ip_

        if min_r == 0: A = a; AV = aV  # actual filters, incremented per comp recursion
        else: A = 0; AV = 0  # if r > min_r

        if min_r <= 1: AD = aD
        else: AD = 0

        fd, fv, r, x, olp, vP_, dP_ = 0, 0, 0, 0, 0, [], []  # i/o tuple
        pri_s, I, D, V, rv, t_, olp_ = 0, 0, 0, 0, 0, [], []  # vP tuple
        pri_sd, Id, Dd, Vd, rd, d_, dolp_ = 0, 0, 0, 0, 0, [], []  # dP tuple

        pri_p = p_[0]

        for x in range(1, X):  # cross-compares consecutive pixels

            p = p_[x]  # new pixel for comp to prior pixel, could use pop()?

            olp, vP, dP, vP_, dP_ = \
            ini_comp(  # initial fuzzy comp for r <= min_r (not used elsewhere)

                p, pri_p, fd, olp, x,  # input variables
                pri_s, I, D, V, rv, t_, olp_,  # vP variables
                pri_sd, Id, Dd, Vd, rd, d_, dolp_,  # dP variables
                a, aV, aD, min_r, A, AV, AD, r, X, vP_, dP_)  # filters and outputs

            pri_p = p  # prior pixel, pri_ values are always derived before use

        LP_ = vP_, dP_
        FP_.append(LP_)  # line of patterns is added to frame of patterns, y = len(FP_)

    return FP_  # output to level 2

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
level_1(f)

# at vP term: print ('type', 0, 'pri_s', pri_s, 'I', I, 'D', D, 'V', V, 'rv', rv, 'p_', p_)
# at dP term: print ('type', 1, 'pri_sd', pri_sd, 'Id', Id, 'Dd', Dd, 'Vd', Vd, 'rd', rd, 'd_', d_)

