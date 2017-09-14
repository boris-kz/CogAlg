from scipy import misc
from collections import deque

'''
    Level 1 with patterns defined by the sign of vertex gradient: modified core algorithm of levels 1 + 2.

    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match.
    
    Pixels are discrete samples of continuous image, so rightward and downward derivatives per pixel are 
    equally representative samples of continuous 90-degree gradient: minimal unique unit of 2D gradient. 
    Hence, such vertex gradient is computed as average of these two orthogonally diverging derivatives.
   
    2D patterns are blobs of same-sign vertex gradient, of value for vP or difference for dP.
    Level 1 has 4 steps of encoding, incremental per line defined by vertical coordinate y:

    y:   comp()    p_ array of pixels, lateral comp -> p,m,d,
    y-1: ycomp()   t_ array of tuples, vertical comp, 1D comb in form_P,
    y-2: comp_P()  P_ array of 1D patterns, vertical comp, 2D comb in form_B, form_PP
    y-3: term_PP() PP_ array of 2D patterns PPs, evaluated for termination and consolidation
'''

def comp(p_):  # comparison of consecutive pixels in a scan line forms tuples: pixel, match, difference

    t_ = []
    pri_p = p_[0]  # no d, m at x=0, lagging t_.append(t)

    for p in p_:  # compares laterally consecutive pixels, vs. for x in range(1, X)

        d = p - pri_p  # difference between consecutive pixels
        m = min(p, pri_p)  # match between consecutive pixels
        t = pri_p, d, m
        t_.append(t)
        pri_p = p

    t = pri_p, 0, 0
    t_.append(t)  # last pixel is not compared
    return t_


def ycomp(t_, _t_, fd, fv, y, Y, r, a, _vP_, _dP_):

    # vertical comparison between pixels, forms vertex tuples t2: p, d, dy, m, my, separate fd, fv
    # last "_" denotes array vs. element, first "_" denotes higher-line array, pattern, or variable

    x, valt_, dalt_, vP_, dP_, term_vP_, term_dP_ = 0,[],[],[],[],[],[]  # term_P_ accumulated in ycomp
    pri_s, I, D, Dy, M, My, G, olp, e_ = 0,0,0,0,0,0,0,0,[]  # also _G: interference | redundancy?

    vP = pri_s, I, D, Dy, M, My, G, olp, e_  # alt_ and rdn over alt_ are included in P by form_blob
    dP = pri_s, I, D, Dy, M, My, G, olp, e_

    A = a * r

    for t, _t in zip(t_, _t_):  # compares vertically consecutive pixels, forms vertex gradients

        x += 1
        p, d, m = t
        _p, _d, _m = _t

        dy = p - _p   # vertical difference between pixels, summed -> Dy
        dg = _d + dy  # gradient of difference, formed at prior-line pixel _p, -> dG: variation eval?
        fd += dg      # fuzzy d-gradient: all shorter + current- range dg s within extended quadrant

        my = min(p, _p)   # vertical match between pixels, summed -> My
        vg = _m + my - A  # gradient of predictive value (relative match) at prior-line _p, -> vG
        fv += vg          # fuzzy v-gradient: all shorter + current- range vg s in extended quadrant

        t2 = p, d, dy, m, my  # 2D tuple, fd, fv -> type-specific g, _g; all accumulated within P:

        sv, valt_, dalt_, vP, vP_, _vP_, term_vP_ = \
        form_P(0, t2, fv, fd, valt_, dalt_, vP, vP_, _vP_, term_vP_, x, y, Y, r, A)

        # forms 1D slice of value pattern vP: horizontal span of same-sign vg s with associated vars

        sd, dalt_, valt_, dP, dP_, _dP_, term_dP_ = \
        form_P(1, t2, fd, fv, dalt_, valt_, dP, dP_, _dP_, term_dP_, x, y, Y, r, A)

        # forms 1D slice of difference pattern dP: horizontal span of same-sign dg s + associated vars

    # line ends, last ycomp t: lateral d = 0, m = 0, inclusion per incomplete gradient?
    # vP, dP term, no initialization:

    dolp = dP[7]; dalt = len(vP_), dolp; dalt_.append(dalt)  # olp: summed overlap by stronger alt_Ps
    volp = vP[7]; valt = len(dP_), volp; valt_.append(valt)

    vP_, _vP_, term_vP_ = form_blob(valt_, vP, vP_, _vP_, term_vP_, x, y, Y, r, A)  # empty _vP_
    dP_, _dP_, term_dP_ = form_blob(dalt_, dP, dP_, _dP_, term_dP_, x, y, Y, r, A)  # empty _dP_

    return vP_, dP_, term_vP_, term_dP_  # also alt_ return for fork_eval? for comp_P, form_PP, term_PP


def form_P(type, t2, g, _g, alt_, _alt_, P, P_, _P_, term_P_, x, y, Y, r, A):  # forms 1D slices of a blob

    p, d, dy, m, my = t2  # 2D tuple per pixel
    pri_s, I, D, Dy, M, My, G, olp, e_ = P  # to increment or initialize vars, also _G to eval alt_P rdn?

    s = 1 if g > 0 else 0
    if s != pri_s and x > r + 2:  # P (span of same-sign gs) is terminated and merged with overlapping _Ps

        P_, _P_, term_P_ = form_blob(type, alt_, P, P_, _P_, term_P_, x, y, Y, r, A)  # line end: _P_ = P_
        _alt = len(P_), olp # index len(P_) and overlap of P are buffered in _P_alt_, total olp = len(e_):
        _alt_.append(_alt)
        I, D, Dy, M, My, G, e_, alt_ = 0,0,0,0,0,0,[],[]  # P and alt_ are initialized

    # continued or initialized P vars are accumulated:

    olp += 1  # P overlap to concurrent alternative-type P, accumulated till either P or _P is terminated
    I += p    # p s summed within P
    D += d    # lateral D for vertical P comp
    Dy += dy  # vertical Dy, for P2 comp only
    M += m    # lateral D for vertical P comp
    My += my  # vertical My, for P2 comp only
    G += g    # fd or fv gradient summed to define P value, vs. V = M - 2a * W?

    if type == 0:
        pri = p, g, _g  # v gradient, also d, dy, m, my for fuzzy accumulation within P-specific r?
        e_.append(pri)  # pattern element: prior same-line vertex, buffered for selective inc_rng comp
    else:
        e_.append(g)  # pattern element: prior same-line d gradient, buffered for selective inc_der comp

    P = s, I, D, Dy, M, My, G, olp, e_  # alt_ is accumulated in ycomp, for PP cost eval before comp_P?

    return s, alt_, _alt_, P, P_, _P_, term_P_

    # draft below:


def form_blob(type, alt_, P, P_, _P_, term_P_, x, y, Y, r, A):  # P over _P_ scan, inclusion, displacement

    # merges vertically contiguous and horizontally overlapping same-type and same-sign Ps into blobs
    # alt_ -> rdn for fork_eval, then alt2_ += alt_ for blob eval

    fork_ = deque() # higher-line matches per P, to assign redundancy and move term _P to next _P
    root_, blob_, buff_ = deque(), deque(), deque()

    vPP_, dPP_, _vPP_, _dPP_ = [],[],[],[]
    s, I, D, Dy, M, My, G, olp, e_ = P[0] # P = P, alt_ fork_, vPP_, dPP_ before P_.append(P)

    _ix = 0  # initial coordinate of _P displaced from _P_ by last comp_P
    ix = x - len(e_)  # initial coordinate of P

    while x >= _ix:  # P /_P connection eval, while horizontal overlap between P and _P:

        oG = 0  # overlapping gradient: oG += g, approx: oG = G * mw / len(e_)
        ex = x  # coordinate of current P element

        _P = _P_.popleft()  # _P = _P, _alt_, blob _fork_, _vPP_, _dPP_, root_, no exposed Ps in y-3

        _ix = P[0][1]  # sub- _P = _s, _ix, _x, _I, _D, _Dy, _M, _My, _G, _e_, _rdn, r
        I2, D2, Dy2, M2, My2, G2, e2_, alt2_, olp2 = _P[2]  # summed blob

        if P[0] == _P[0][0]:  # if s == _s: v or d sign match

            while ex > _ix:

                for e in e_:  # oG accumulation  # PM, PD from comp_P only
                    if type == 0: oG += e[1]  # if vP: e = p, g, _g
                    else: oG += e  # if dP: e = g
                    ex += 1

            fork = oG, _P
            fork_.append(fork)  # _P inclusion in P
            _P[5].append(P)  # root_.append(P), to track continuing roots in form_PP

            I2 += I
            D2 += D; Dy2 += Dy  
            M2 += M; My2 += My
            G2 += G  # possibly fuzzy
            e2_ += e_ # blob area = len(e2_), or no e2_: Py_( P( e_?
            alt2_ += alt_ # or replaced by alt_blob_, with olp2 per alt2 (P2)? 
            olp2 += olp   # area overlap to stronger alt2_, after alt blobs eval?

            # sum of all roots per blob, to evaluate it, assigned to last fork at partial term?

        if _P[0][8] > ix:  # if _x > ix:

            buff_.append(_P)  # _P with updated root_ is buffered for popleft to next-P comp

        else: # no horizontal overlap between _P and next P, _fork_s are evaluated for term_PP?

            if (len(root_) == 0 and y > r + 3) or y == Y - 1:  # _P or frame is terminated
                # trunk term if split: len(root_) > 1, part blob?

                for blob in blob_:
                    blob, _vPP, _dPP = blob  # <= one _vPP and _dPP per higher-line blob:

                    term_PP(blob)  # possible re-orient and re-scan, but no direct recursion
                    if _vPP > 0: term_PP(_vPP)  # not for _vPP in _vPP_: only to eval for rdn?
                    if _dPP > 0: term_PP(_dPP)

            buff_ += _P_  # for form_blob (next P)

    # no horizontal overlap between P and _P, evaluation for P comp to fork_, then buffered:

    for alt in alt_:  # rdn = P[9]: number of greater-criterion alt Ps in alt_, =0 in ycomp?

        if alt[6] > P[6]: # comp of criterion: vG for vP | dG for dP
            P[9] += 1  # increment weaker-alt rdn
        else: alt[9] += 1

    if len(fork_) > 0:  # fork_ eval for comp_P, returning _vPP_, _dPP_, vPP_, dPP_
       fork_ = fork_eval(0, fork_, A)

    if len(vPP_) > 0:  # vPP_ eval for form_PP -> 2D value pattern, rdn alt_ dPPs:
        vPP_ = fork_eval(1, vPP_, A)  # vPP_buff_ is recycled in form_PP?

    if len(dPP_) > 0:  # dPP_ eval for form_PP -> 2D difference pattern, rdn alt_ vPPs:
        dPP_ = fork_eval(2, dPP_, A)

    # these vPPs and dPPs are also modified in corresponding forks?

    P = P, alt_, fork_, vPP_, dPP_  # adding root_ (lower-line matches) at P_ -> _P_ conversion
    P_.append(P)  # P buffered in P_, terminated root Ps are stored in term_?

    # no term and no buff: _P is already included in its root Ps

    return P_, buff_, term_P_  # _P and term_P include _P, alt_, blob_, _vPP_, _dPP_: formed in comp_P


def fork_eval(type, fork_, A):  # fork_ eval for comp_P and _vPP_, _dPP_ append, then eval for form_PP

    # fork eval per P slice or at blob term | split: variation accumulated in 1D or 2D?

    select_ = [] # for fork comp eval, or for _PP form eval

    for fork in fork_: # fork_ is preserved for assignment to last fork

        crit = fork[0]  # selection criterion: oG for fork | PM for vPP | PD for dPP
        if crit > A:  # comp to _crit in select_,  inclusion in select_

            for select in select_:  # forks are re-evaluated at each rdn increment

                _crit = select[0]
                if crit > _crit:  # criterion comp

                    select[0][9] += 1  # increment weaker-fork rdn
                    if _crit < A * select[0][9]: del (select)  # delete select from select_

                else: fork[0][9] += 1

            if crit > A * fork[0][9]:  # inclusion after full select_ rdn assignment
                select_.append(fork)

    while len(select_) > 0:
        fork = select_.pop  # no eval, already selected

        if type == 0: comp_P(fork)  # that also changes fork in original fork_?
        else: form_PP (type, fork)  # separate _vPP_, _dPP_ per _P were appended by comp_P

        # fork access by increasing rdn (max rdn for non-select) or coordinate, no separate max?

    return fork_


def comp_P(P, P_, _P, _P_, x):  # forms 2D derivatives of 1D P vars to define vPP and dPP:

    ddx = 0  # optional;

    s, I, D, Dy, M, My, G, e_, oG, rdn, alt_  = P  # select alt_ per fork, no olp: = mx?
    _s, _ix, _x, _I, _D, _Dy, _M, _My, _G, _e_, _rdn, r, _alt_, blob_ = _P  # fork = r, _alt_, blob_, P?

    ix = x - len(e_)  # len(e_) or w: P width, initial coordinate of P, for output only?

    dx = x - len(e_)/2 - _x - len(_e_)/2  # Dx? comp(dx), ddx = Ddx / h? dS *= cos(ddx), mS /= cos(ddx)?
    mx = x - _ix
    if ix > _ix: mx -= ix - _ix  # x overlap, mx - a_mx, form_P(vxP), vs. discont: mx = -(a_dx - dx)?

    dw = len(e_) - len(_e_)  # -> dwP: higher dim? Ddx + Dw triggers adjustment of derivatives or _vars?
    mw = min(len(e_), len(_e_))  # w: P width = len(e_), relative overlap: mx / w, similarity: mw?

    # ddx and dw signs correlate, dx (position) and dw (dimension) signs don't correlate?
    # full input CLIDV comp, or comp(S| aS(L rdn norm) in positive eM = mx+mw, more predictive than eD?

    dI = I - _I; mI = min(I, _I)  # eval of MI vs. Mh rdn at term PP | var_P, not per slice?
    dD = D - _D; mD = min(D, _D)
    dM = M - _M; mM = min(M, _M)  # no G comp: y-derivatives are incomplete. also len(alt_) comp?

    PD = ddx + dw + dI + dD + dM  # defines dPP; var_P form if PP form, term if var_P or PP term;
    PM = mx + mw + mI + mD + mM   # defines vPP; comb rep value = PM * 2 + PD?  group by y_ders?

    # vPP and dPP included in selected forks, rdn assign and form_PP eval after fork_ term in form_blob?

    crit, rdn, W, I2, D2, Dy2, M2, My2, G2, rdn2, alt2_, Py_ = 0,0,0,0,0,0,0,0,0,0,[],[]  # PP vars declaration

    P2  = crit, rdn, W, I2, D2, Dy2, M2, My2, G2, rdn2, alt2_, Py_
    vPP = crit, rdn, W, I2, D2, Dy2, M2, My2, G2, rdn2, alt2_, Py_
    dPP = crit, rdn, W, I2, D2, Dy2, M2, My2, G2, rdn2, alt2_, Py_

    blob = P2, vPP, dPP  # PPs are initialized at non-matching P transfer to _P_, as generic element of _fork_
    blob_.append(blob)  # not really append, only _fork type declaration, re-used per lower P?

    P = s, I, D, Dy, M, My, G, r, e_, alt_, blob_  # _fork_ is empty, similar to tuple declaration?
    # returned Ps also include current derivatives per var?

    P_.append(P)  # _P_ = P_ for next-line comp, if no horizontal overlap between P and next _P

    return P_, _P_, term_P_  # fork_ is accumulated within blob_eval


def form_PP(PP, fork, root_, _fork_, _P_, term_P_, _x, y, Y, r, A):  # forms vPPs, dPPs, and their var Ps

    # dimensionally reduced axis: vP'PP or contour: dP'PP; dxP is direction pattern

    a_mx = 2; a_mw = 2; a_mI = 256; a_mD = 128; a_mM = 128  # feedback to define var_vPs (variable value patterns)
    # a_PM = a_mx + a_mw + a_mI + a_mD + a_mM  or A * n_vars, rdn accum per var_P, alt eval per vertical overlap?

    crit, rdn, W, I2, D2, Dy2, M2, My2, G2, rdn2, alt2_, Py_ = PP  # initialized at P re-input in comp_P

    mx, dx, mw, dw, mI, dI, mD, dD, mM, dM, P, _P = fork  # current derivatives, to be included if match
    s, ix, x, I, D, Dy, M, My, G, r, e_, alt_ = P  # input P or _P: no inclusion of last input?

    # criterion eval, P inclusion in PP, then all connected PPs in CP, unique tracing of max_crit PPs:

    if crit > A * 5 * rdn:  # PP vars increment, else empty fork ref?

        W += len(alt_); I2 += I; D2 += D; Dy2 += Dy; M2 += M; My2 += My; G2 += G; alt2_ += alt_
        Py_.append(P)

        # also var_P form: LIDV per dx, w, I, D, M? select per term?
        PP = W, I2, D2, Dy2, M2, My2, G2, alt2_, Py_  # alt2_: fork_ alt_ concat, to re-compute redundancy per PP

        fork = len(_P_), PP
        fork_.append(fork)  # _P index and PP per fork, possibly multiple forks per P

        root_.appendleft(P)  # connected Ps in future term_ and term_P_
        # all continuing _Ps of CP, referenced from its first root _P: CP flag per _P?

    return _P_, term_P_  # laced term_vP_ and term_dP_? + refs to vPPs, dPPs, vCPs, dCPs?


def term_PP(P2):  # sub-level 4: eval for rotation, re-scan, re-comp, recursion, accumulation, at PP or CP term?

    term_ = []  # terminated root Ps per next fork P in Py_, last fork (CP) is stored in term_PP_?
                # accumulated per term_PP within root?  incomplete inclusion: no-match Ps -> _P' cont_?

    WC, IC, DC, DyC, MC, MyC, GC, rdnC, altC_, PP_ = 0,0,0,0,0,0,0,0,[],[]  # CP vars (connected PPs) at first Fork
    WC += W; IC += I2; DC += D2; DyC += Dy2; MC += M2; MyC += My2; GC += G2; altC_ += alt2_; PP_.append(PP)

    # orientation (D reduction -> axis | outline), consolidation, P2 rdn fb?

    if (len(fork_) == 0 and y > r + 3) or y == Y - 1:  # no continuation per CP (CP and returned by term):

       # eval for rotation, re-scan, cross-comp of P2_? also sum per frame?

    elif len(_P_) == last_croot_nP:  # CP_ to _P_ sync for PP inclusion and cons(CP) trigger by Fork_' last _P?

        CP_.append(CP)  # PP may include len(CP_): CP index

''' 
    :param P2: 
    :return: 
    ''''''
    eval of d,m adjust | _var adjust | x,y adjust if projected dS-, mS+ for min.1D Ps over max.2D

        if dw sign == ddx sign and min(dw, ddx) > a: _S /= cos (ddx)  # to angle-normalize S vars for comp

    if dw > a: div_comp (w): rw = w / _w, to width-normalize S vars for comp: 

        if rw > a: pn = I/w; dn = D/w; vn = V/w; 

            comp (_n) # or default norm for redun assign, but comp (S) if low rw?

            if d_n > a: div_comp (_n) -> r_n # or if d_n * rw > a: combined div_comp eval: ext, int co-variance?

        comp Dy and My, /=cos at PP term?  default div and overlap eval per PP? not per CP: sparse coverage?
        
    rrdn = 1 + rdn_w / len(e_)  # redundancy rate / w, -> P Sum value, orthogonal but predictive
    S = 1 if abs(D) + V + a * len(e_) > rrdn * aS else 0  # rep M = a*w, bi v!V, rdn I?
    '''

    mean_dx = 1  # fractional?
    dx = Dx / H
    if dx > a: comp(abs(dx))  # or if dxP Dx: fixed ddx cost?  comp of same-sign dx only

    vx = mean_dx - dx  # normalized compression of distance: min. cost decrease, not min. benefit?


def Le1(f):  # last "_" denotes array vs. element, first "_" denotes higher-line array, pattern, variable

    r = 1; a = 127  # feedback filters
    Y, X = f.shape  # Y: frame height, X: frame width
    fd, fv, y, _vP_, _dP_, term_vP_, term_dP_, frame_ = 0,0,0,[],[],[],[],[]

    p_ = f[0, :]   # first line / row of pixels
    _t_= comp(p_)  # _t_ includes ycomp() results, with Dy, My, dG, vG initialized at 0

    for y in range(1, Y):  # y is index of new line p_

        p_ = f[y, :]
        t_ = comp(p_)  # lateral pixel comp, then vertical pixel comp:
        _vP_, _dP_, term_vP_, term_dP_ = ycomp(t_, _t_, fd, fv, y, Y, r, a, _vP_, _dP_)
        _t_ = t_

        PP_ = term_vP_, term_dP_  # PP term by comp_P, adjust by cons_P2, after P ) PP ) CP termination
        frame_.append(PP_)  # line of patterns is added to frame of patterns, y = len(F_)

    return frame_  # output to level 2

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
Le1(f)

