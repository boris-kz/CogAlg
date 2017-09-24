from scipy import misc
from collections import deque
import numpy as np

'''
    Level 1 with patterns defined by the sign of vertex gradient: modified core algorithm of levels 1 + 2.

    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match.
    
    Pixels are discrete samples of continuous image, so rightward and downward derivatives per pixel are 
    equally representative samples of continuous 0-90 degree gradient: minimal unique unit of 2D gradient 
    Hence, such vertex gradient is computed as average of these two orthogonally diverging derivatives.
   
    2D patterns are blobs of same-sign vertex gradient, of value for vP or difference for dP.
    Level 1 has 5 steps of encoding, incremental per line defined by vertical coordinate y:

    y:   comp(p_):  lateral comp -> tuple t,
    y-1: ycomp (t_): vertical comp -> vertex t2,
    y-1: form_P(t2_): lateral combination -> 1D pattern P,
    y-2: form_P2 (P_): vertical comb | comp -> 2D pattern P2,
    y-2: term_P2 (P2_): P2s are terminated and evaluated for recursion
'''

def comp(p_):  # comparison of consecutive pixels within line forms tuples: pixel, match, difference

    t_ = []  # complete fuzzy tuples: summation range = rng
    it_ = []  # incomplete fuzzy tuples: summation range < rng
    d, m = 0, 0  # no d, m at x = 0

    for p in p_:

        for it in it_:  # incomplete tuples with summation range from 0 to rng
            pri_p, fd, fm = it

            d = p - pri_p  # difference between pixels
            m = min(p, pri_p)  # match between pixels

            fd += d  # fuzzy d: sum of ds between p and all prior ps within it_
            fm += m  # fuzzy m: sum of ms between p and all prior ps within it_

        if len(it_) == rng:

            t = pri_p, fd, fm
            t_.append(t)
            del it_[0]  # completed tuple is transferred from it_ to t_

        it = p, d, m
        it_.append(it)  # new prior tuple

    t_ += it_  # last number = rng of tuples remain incomplete
    return t_


def ycomp(t_, _t_):  # vertical comparison between pixels, forms vertex tuples t2: p, fd, fdy, fm, fmy

    # last "_" denotes array vs. element, first "_" denotes higher-line array, pattern, or variable

    global _vP_; global _dP_  # converted from vP_ and dP_ here, then used by form_blob
    global vP_;  global dP_  # appended by form_P

    global valt_; valt_ = []  # appended by form_P, included in P by form_blob, to form alt2_?
    global dalt_; dalt_ = []

    vP = 0,0,0,0,0,0,0,[],0  # pri_s, I, D, Dy, M, My, G, e_, rdn_olp
    dP = 0,0,0,0,0,0,0,[],0  # pri_s, I, D, Dy, M, My, G, e_, rdn_olp

    olp = 0,0,0  # olp_len, olp_vG, olp_dG: common for current vP and dP
    x = 0

    for t, _t in zip(t_, _t_):  # compares vertically consecutive pixels, forms vertex gradients

        x += 1
        p, d, m = t
        _p, _d, _m = _t

        # non-fuzzy pixel comparison:

        dy = p - _p  # vertical difference between pixels, summed -> Dy
        dg = _d + dy  # gradient of difference, formed at prior-line pixel _p, -> dG: variation eval?

        my = min(p, _p)   # vertical match between pixels, summed -> My
        vg = _m + my - ave  # gradient of predictive value (relative match) at prior-line _p, -> vG

        t2 = p, d, dy, m, my  # 2D tuple, fd, fv -> type-specific g, _g; all accumulated within P:

        sv, olp, valt_, dalt_, vP, dP, vP_ = \
        form_P(0, t2, vg, dg, olp, valt_, dalt_, vP, dP, vP_, x)

        # forms 1D value pattern vP: horizontal span of same-sign vg s with associated vars

        sd, olp, dalt_, valt_, dP, vP, dP_ = \
        form_P(1, t2, dg, vg, olp, dalt_, valt_, dP, vP, dP_, x)

        # forms 1D difference pattern dP: horizontal span of same-sign dg s + associated vars

    _vP_ = vP_; _dP_ = dP_  # P_ is appended and returned by form_P within a line

    # line ends, t2s have incomplete lateral fd and fm, inclusion per vg - ave / (rng / X-x)?
    # olp term, vP term, dP term, no initialization:

    dalt_.append(olp); valt_.append(olp)  # if any?
    olp_len, olp_vG, olp_dG = olp

    if olp_vG > olp_dG:  # comp of olp_vG to olp_dG, == goes to alt_P or to vP: primary?
        vP[8] += olp_len  # accumulate redundant overlap in current vP or dP with weaker oG
    else:
        dP[8] += olp_len

    vP[8] /= vP[7]  # rolp = rdn_olp / len(e_): redundancy ratio of P to overlapping alt_Ps
    dP[8] /= dP[7]  # or A = ave * dP[8] /= dP[7]?

    if y+1 > rng:

        vP_, _vP_ = scan_high(0, vP, vP_, valt_, x)  # empty _vP_
        dP_, _dP_ = scan_high(1, dP, dP_, dalt_, x)  # empty _dP_

    # no return of vP_, dP_: converted to global _vP_, _dP_ at line end


def form_P(typ, t2, g, alt_g, olp, alt_, _alt_, P, alt_P, P_, x):  # forms 1D Ps

    p, d, dy, m, my = t2  # 2D tuple of vertex per pixel
    pri_s, I, D, Dy, M, My, G, e_, rdn_olp = P

    if typ == 0:
        olp_len, oG, alt_oG = olp  # overlap between current vP and dP, accumulated in ycomp
    else:
        olp_len, alt_oG, oG = olp

    s = 1 if g > 0 else 0
    if s != pri_s and x > rng + 2:  # P (span of same-sign gs) is terminated

        if alt_oG > oG:  # comp of olp_vG to olp_dG, == goes to alt_P or to vP: primary?
            rdn_olp += olp_len  # accumulate redundant overlap in current vP or dP with weaker oG
        else:
            alt_P[8] += olp_len

        rolp = rdn_olp / len(e_)  # redundancy ratio of P to overlapping alt_Ps
        A = ave * rolp
        P = pri_s, I, D, Dy, M, My, G, e_, A  # or rolp vs. A: passed to combine rdn?

        P_ = scan_high(typ, P, P_, alt_, x)  # for all forks, _P_ = P_ at line end

        alt = alt_P, olp_len, oG, alt_oG  # or P index len(P_): faster than P?  for P eval in form_blob
        alt_.append(alt)
        _alt = P, olp_len, alt_oG, oG  # redundant olp repr in concurrent alt_P, formed by terminated P
        _alt_.append(_alt)

        I, D, Dy, M, My, G, e_, alt_ = 0,0,0,0,0,0,[],[]
        olp = 0,0,0  # P, alt_, olp are initialized

    # continued or initialized P vars are accumulated:

    olp_len += 1  # alt P overlap: olp_len, oG, alt_oG are accumulated till either P or _P is terminated
    oG += g; alt_oG += alt_g

    I += p    # p s summed within P
    D += d    # lateral D, for P comp and P2 normalization
    Dy += dy  # vertical D, for P2 normalization
    M += m    # lateral D, for P comp and P2 normalization
    My += my  # vertical M, for P2 normalization
    G += g    # d or v gradient summed to define P value, or V = M - 2a * W?

    if typ == 0:
        pri = p, g, alt_g  # v gradient, also d, dy, m, my for fuzzy accumulation within P-specific r?
        e_.append(pri)  # pattern element: prior same-line vertex, buffered for selective inc_rng comp
    else:
        e_.append(g)  # pattern element: prior same-line d gradient, for selective inc_der comp

    P = s, I, D, Dy, M, My, G, e_  # incomplete P
    olp = olp, oG, alt_oG

    return s, olp, alt_, _alt_, P, alt_P, P_  # alt_ and _alt_ are accumulated in ycomp over full line


def scan_high(typ, P, P_, alt_, x):  # P scans over higher-line _P_ for inclusion, _P displacement

    fork_ = deque()  # higher-line matches per P, to assign redundancy and move term _P to next _P
    root_, blob_, buff_ = deque(), deque(), deque()

    vPP_, dPP_, _vPP_, _dPP_ = [],[],[],[]
    _ix = 0  # initial coordinate of _P displaced from _P_ by last comp_P

    s, I, D, Dy, M, My, G, e_, A = P
    ix = x - len(e_)  # initial coordinate of P

    if typ == 0: _P_ = _vP_
    else: _P_ = _dP_

    while x >= _ix:  # P to _P connection eval, while horizontal overlap between P and _P:

        fork_oG = 0  # fork overlap gradient: oG += g, approx: oG = G * mw / len(e_)
        ex = x  # coordinate of current P element

        _P = _P_.popleft()   # _P = _P, _alt_, root_, blob_, _vPP_, _dPP_
        _ix = _P[0][1]  # sub- _P = _s, _ix, _x, _I, _D, _Dy, _M, _My, _G, _e_

        if P[0] == _P[0][0]:  # if s == _s: v or d sign match

            while ex > _ix:
                for e in e_:  # oG accumulation (PM, PD from comp_P only)

                    if typ == 0: fork_oG += e[1]  # if vP: e = p, g, alt_g
                    else: fork_oG += e  # if dP: e = g
                    ex += 1

            fork = fork_oG, 0, _P  # rdn is initialized at 0
            fork_.append(fork)  # _P inclusion in P
            _P[2].append(P)  # root_.append(P), to track continuing roots in form_PP

        if _P[0][2] > ix:  # if _x > ix:

            buff_.append(_P)  # _P with updated root_ is buffered for next-P comp

        else:  # no horizontal overlap between _P and next P, _P is evaluated for termination

            if (len(root_) == 0 and y > rng + 3) or y == Y - 1:  # _P or frame is terminated

                for blob in blob_:
                    blob, _vPP, _dPP = blob  # <= one _vPP and _dPP per higher-line blob:

                    term_P2(blob, A)  # possible 2D P re-orient and re-scan, but no direct recursion
                    if _vPP > 0: term_P2(_vPP, A)  # not for _vPP in _vPP_: only to eval for rdn?
                    if _dPP > 0: term_P2(_dPP, A)

            buff_ += _P_  # for scan_high(next P)

    # no more horizontal overlap between P and _P:

    if len(fork_) > 0:  # fork_ evaluation for P inclusion and comparison
        bA = A
        fork_, bA = fork_eval(0, P, fork_, bA)  # bA *= blob rdn

        if len(vPP_) > 0:  # = lateral len(dPP_): formed by comp_P over same forks

            vA = bA  # eval for inclusion in vPPs (2D value patterns), rdn alt_ = blobs:
            vPP_, vA = fork_eval(1, P, vPP_, vA)

            dA = vA  # eval for inclusion in dPPs (2D difference patterns), rdn alt_ = vPPs:
            dPP_, dA = fork_eval(2, P, dPP_, dA)

            # individual vPPs and dPPs are also modified in their fork

    P = P, alt_, fork_, vPP_, dPP_  # adding root_ (lower-line matches) at P_ -> _P_ conversion
    P_.append(P)  # P is buffered in P_, terminated root Ps are stored in term_?

    if typ == 0: _vP_ = buff_  # modifying global _vP_
    else: _dP_ = buff_ # _P = _P, alt_, blob_, _vPP_, _dPP_

    return P_  # with added fork_... per P


def fork_eval(typ, P, fork_, A):  # _Ps eval for init_blob, incr_blob, comp_P, init_PP, incr_PP

    select_ = []  # for fork comp eval, or for _PP form eval

    for fork in fork_:  # fork | select = crit, rdn, _P

        if fork[0] > A:  # comp to _crit (oG for fork | PM for vPP | PD for dPP) in select_

            for select in select_:  # forks are re-evaluated at each rdn increment

                if fork[0] > select[0]:  # criterion comp

                    select[1] += 1  # increment weaker-fork rdn, initialized per fork?
                    if select[0] < A * select[1]: del select  # delete from select_ if _rdn = max

                else: fork[1] += 1  # increment weaker-fork rdn

            if fork[0] > A * fork[1]:  # inclusion after full select_ comp, also A *= rdn?
                select_.append(fork)

    for select in select_:  # no re-eval for select forks, rdn = max for non-select forks

        # merges vertically contiguous and horizontally overlapping same- type and sign Ps into P2s
        # P2: blob | vPP | dPP, alt_ -> rolp and alt2_, -> rolp2: area overlap?

        if typ == 0:  # select fork = blob

            fork = form_blob(P, select)  # same min oG for blob inclusion and comp_P?:
            vPP, dPP = comp_P(P, select)
            fork = fork, vPP, dPP

        else:  # select fork = vPP or dPP

            fork = form_PP(typ, P, select)

        fork_.append(fork)
        del select  # from select_, preserved in fork_

    return fork_, A  # or includes A?

    # terminated root Ps contain 1 blob_, each blob is transferred to corresponding fork?
    # and are summed into fork's blob at its term
    # blob init if multiple forks, non-selected forks may be terminated?

    # fork = crit, rdn, _P;  # formed by scan_high, A formed per eval: a * rolp * rdn..?
    # _P = _P, _alt_, root_, blob_, _vPP_, _dPP_
    # _P = s, _ix, _x, _I, _D, _Dy, _M, _My, _G, _e_


def form_blob(P, fork):  # P inclusion into selected fork's blob, initialized or continuing

    s, I, D, Dy, M, My, G, e_, rdn, alt_ = P

    if fork[1] > 0:  # rdn > 0: new blob initialization, then terminated if not of max fork?

        I2 = I
        D2 = D; Dy2 = Dy
        M2 = M; My2 = My
        G2 = G
        area = len(e_)  # initialized with new fork?
        e2_ = e_  # or no separate e2_: Py_( P( e_?
        alt2_ = alt_ # or replaced by alt_blob_?
        Py_ = P  # vertical array of patterns within a blob

        blob = I2, D2, Dy2, M2, My2, G2, area, e2_, alt2_, Py_
        fork[2][3].append(blob) # blob_.append

    else:  # increments axis: max _fork's blob in blob_ of max fork: first or separate?

        I2, D2, Dy2, M2, My2, G2, area, e2_, alt2_, Py_ = fork[3]

        I2 += I
        D2 += D; Dy2 += Dy
        M2 += M; My2 += My
        G2 += G
        area += len(e_)  # initialized with new fork?
        e2_.append(e_)  # or no separate e2_: Py_( P( e_?
        alt2_ += alt_  # or replaced by alt_blob_?
        Py_.append(fork[0])  # vertical array of patterns within a blob

        blob = I2, D2, Dy2, M2, My2, G2, area, e2_, alt2_, Py_

    return fork


def comp_P(P, P_, _P, _P_, x):  # forms 2D derivatives of 1D P vars to define vPP and dPP:

    ddx = 0  # optional;

    s, I, D, Dy, M, My, G, e_, oG, rdn, alt_ = P  # select alt_ per fork, no olp: = mx?
    _s, _ix, _x, _I, _D, _Dy, _M, _My, _G, _e_, _rdn, _alt_, blob_ = _P  # fork = P, _alt_, blob_?

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

    blob= 0,0,0,0,0,0,0,0,0,0,[],[]  # crit, rdn, W, I2, D2, Dy2, M2, My2, G2, rdn2, alt2_, Py_
    vPP = 0,0,0,0,0,0,0,0,0,0,[],[]
    dPP = 0,0,0,0,0,0,0,0,0,0,[],[]  # P2s are initialized at non-matching P transfer to _P_?

    ''' np.array for direct accumulation, or simply iterator of initialization?

    P2_ = np.array([blob, vPP, dPP],
        dtype=[('crit', 'i4'), ('rdn', 'i4'), ('W', 'i4'), ('I2', 'i4'), ('D2', 'i4'), ('Dy2', 'i4'),
        ('M2', 'i4'), ('My2', 'i4'), ('G2', 'i4'), ('rdn2', 'i4'), ('alt2_', list), ('Py_', list)])
    '''

    P = s, I, D, Dy, M, My, G, e_, alt_, blob_  # _fork_ is empty, similar to tuple declaration?
    # returned Ps also include current derivatives per var?

    P_.append(P)  # _P_ = P_ for next-line comp, if no horizontal overlap between P and next _P

    return P_, _P_


def form_PP(PP, fork, root_, blob_, _P_, _P2_, _x, A):  # forms vPPs, dPPs, and their var Ps

    if fork[1] > 0:  # rdn > 0: new PP initialization?

    else:  # increments PP of max fork, cached by fork_eval(), also sums all terminated PPs?

        crit, rdn, W, I2, D2, Dy2, M2, My2, G2, rdn2, alt2_, Py_ = PP  # initialized at P re-input in comp_P

        W += len(alt_); I2 += I; D2 += D; Dy2 += Dy; M2 += M; My2 += My; G2 += G; alt2_ += alt_
        Py_.append(P)

    # fork = crit, rdn, _P
    # _P = _P, _alt_, root_, blob_, _vPP_, _dPP_
    # _P = s, _ix, _x, _I, _D, _Dy, _M, _My, _G, _e_

    # dimensionally reduced axis: vP'PP or contour: dP'PP; dxP is direction pattern

    a_mx = 2; a_mw = 2; a_mI = 256; a_mD = 128; a_mM = 128  # feedback to define var_vPs (variable value patterns)
    # a_PM = a_mx + a_mw + a_mI + a_mD + a_mM  or A * n_vars, rdn accum per var_P, alt eval per vertical overlap?


    mx, dx, mw, dw, mI, dI, mD, dD, mM, dM, P, _P = fork  # current derivatives, to be included if match
    s, ix, x, I, D, Dy, M, My, G, r, e_, alt_ = P  # input P or _P: no inclusion of last input?

    # criterion eval, P inclusion in PP, then all connected PPs in CP, unique tracing of max_crit PPs:

    if crit > A * 5 * rdn:  # PP vars increment, else empty fork ref?

        W += len(alt_); I2 = I; D2 = D; Dy2 = Dy; M2 = M; My2 = My; G2 = G; alt2_ = alt_; Py_ = P
        # also var_P form: LIDV per dx, w, I, D, M? select per term?

        PP = W, I2, D2, Dy2, M2, My2, G2, alt2_, Py_  # alt2_: fork_ alt_ concat, to re-compute redundancy per PP

        fork = len(_P_), PP
        blob_.append(fork)  # _P index and PP per fork, possibly multiple forks per P

        root_.append(P)  # connected Ps in future blob_ and _P2_

    return PP


def term_P2(P2, A):  # blob | vPP | dPP eval for rotation, re-scan, re-comp, recursion, accumulation

    P2_ = []
    ''' 
    conversion of root to term, sum into wider fork, also sum per frame?
    
    mean_dx = 1  # fractional?
    dx = Dx / H
    if dx > a: comp(abs(dx))  # or if dxP Dx: fixed ddx cost?  comp of same-sign dx only

    vx = mean_dx - dx  # normalized compression of distance: min. cost decrease, not min. benefit?
    
    
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


def level_1(f):  # last "_" denotes array vs. element, first "_" denotes higher-line variable or pattern

    global rng; rng = 1
    global ave; ave = 127  # filters, ultimately set by separate feedback, then ave *= rng

    global Y; global X
    Y, X = f.shape  # Y: frame height, X: frame width

    global _vP2_; _vP2_ = []
    global _dP2_; _dP2_ = []  # 2D Ps terminated on line y-3

    _vP_, _dP_, frame_ = [], [], []

    global y; y = 0

    p_ = f[0, :]   # first line of pixels
    _t_= comp(p_)  # _t_ includes ycomp() results and Dy, My, dG, vG = 0

    for y in range(1, Y):  # vertical coordinate y is index of new line p_

        p_ = f[y, :]
        t_ = comp(p_)  # lateral pixel comp
        ycomp(t_, _t_) # vertical pixel comp, _vP2_, _dP2_ are appended internally
        _t_ = t_

        P2_ = _vP2_, _dP2_  # arrays of blobs terminated on current line, adjusted by term_P2
        frame_.append(P2_)  # line of patterns is added to frame of patterns
        _vP2_, _dP2_ = [],[]

    return frame_  # frame of 2D patterns is output to level 2

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
level_1(f)

