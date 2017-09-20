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

    y:   comp(p_): lateral comp -> tuple t,
    y-1: ycomp (t_): vertical comp -> vertex t2,
    y-2: form_P(t2_): lateral combination -> 1D pattern P,
    y-3: form_P2 (P_): vertical comb | comp -> 2D pattern P2,
    y-4: term_P2 (P2_): P2s are terminated and evaluated for recursion
'''

def comp(p_):  # comparison of consecutive pixels in a scan line forms tuples: pixel, match, difference

    t_ = []  # complete fuzzy tuples: summation rng = r
    it_ = deque()  # incomplete fuzzy tuples: summation rng < r
    d, m = 0, 0  # no d, m at x = 0

    for p in p_:
        rng = 1  # summation range of current tuple in it_:

        for it in it_:
            pri_p, fd, fm = it  # tuples with summation range from 1 to r

            d = p - pri_p  # difference between consecutive pixels
            m = min(p, pri_p)  # match between consecutive pixels

            fd += d  # fuzzy d: sum of ds between p and prior ps within rng
            fm += m  # fuzzy m: sum of ms between p and prior ps within rng

            rng += 1
            if rng == r:

                t = pri_p, fd, fm
                t_.append(t)
                del(it)

        it = p, d, m
        it_.appendleft(it)  # new prior tuple, last r tuples remain incomplete

    return t_


def ycomp(t_, _t_):  # vertical comparison between pixels, forms vertex tuples t2: p, fd, fdy, fm, fmy

    # last "_" denotes array vs. element, first "_" denotes higher-line array, pattern, or variable

    global _vP_; global _dP_  # converted from vP_ and dP_ below
    global vP_; global dP_  # appended by form_P, used by form_blob

    vP_, dP_ = [],[]

    global valt_; global dalt_  # appended by form_P,
    valt_, dalt_ = [],[]

    vP = 0,0,0,0,0,0,0,[],0  # pri_s, I, D, Dy, M, My, G, e_, rdn_olp
    dP = 0,0,0,0,0,0,0,[],0  # pri_s, I, D, Dy, M, My, G, e_, rdn_olp

    olp = 0,0,0  # olp_len, olp_vG, olp_dG: common for current vP and dP
    x = 1  # ycomp starts from 2nd line

    for t, _t in zip(t_, _t_):  # compares vertically consecutive pixels, forms vertex gradients

        x += 1
        p, d, m = t
        _p, _d, _m = _t

        dy = p - _p   # vertical difference between pixels, summed -> Dy
        dg = _d + dy  # gradient of difference, formed at prior-line pixel _p, -> dG: variation eval?
        fd += dg  # to be revised!  fuzzy d-gradient: all shorter + current- range dg s within extended quadrant

        my = min(p, _p)   # vertical match between pixels, summed -> My
        vg = _m + my - a  # gradient of predictive value (relative match) at prior-line _p, -> vG
        fv += vg  # to be revised!  fuzzy v-gradient: all shorter + current- range vg s in extended quadrant

        t2 = p, d, dy, m, my  # 2D tuple, fd, fv -> type-specific g, _g; all accumulated within P:

        sv, olp, valt_, dalt_, vP, dP, vP_ = \
        form_P(0, t2, fv, fd, olp, valt_, dalt_, vP, dP, vP_, x)

        # forms 1D value pattern vP: horizontal span of same-sign vg s with associated vars

        sd, olp, dalt_, valt_, dP, vP, dP_ = \
        form_P(1, t2, fd, fv, olp, dalt_, valt_, dP, vP, dP_, x)

        # forms 1D difference pattern dP: horizontal span of same-sign dg s + associated vars

    _vP_ = vP_; _dP_ = dP_ # formed by form_P

    # line ends, last ycomp t: lateral d = 0, m = 0, inclusion per incomplete gradient,
    # olp term, vP term, dP term, no initialization:

    dalt_.append(olp); valt_.append(olp)  # if any?
    olp_len, olp_vG, olp_dG = olp

    if olp_vG > olp_dG:  # comp of olp_vG to olp_dG, == goes to alt_P or to vP: primary?
        vP[8] += olp_len  # accumulate redundant overlap in current vP or dP with weaker oG
    else:
        dP[8] += olp_len

    vP[8] /= vP[7]  # rolp = rdn_olp / len(e_):
    dP[8] /= dP[7]  # relative rdn_olp: redundancy ratio of P to overlapping alt_Ps

    vP_, _vP_ = form_blob(0, vP, vP_, valt_, x)  # empty _vP_
    dP_, _dP_ = form_blob(1, dP, dP_, dalt_, x)  # empty _dP_

    return vP_, dP_  # also alt_ return for fork_eval?


def form_P(type, t2, g, alt_g, olp, alt_, _alt_, P, alt_P, P_, x):  # forms 1D Ps

    p, d, dy, m, my = t2  # 2D tuple of vertex per pixel
    pri_s, I, D, Dy, M, My, G, e_, rdn_olp = P

    if type == 0:
        olp_len, oG, alt_oG = olp  # overlap between current vP and dP, accumulated in ycomp
    else:
        olp_len, alt_oG, oG = olp  # no oG, alt_oG div: same olp_len, oG comp -> rdn_olp at term

    s = 1 if g > 0 else 0
    if s != pri_s and x > r + 2:  # P(span of same-sign gs) is terminated and evaluated to form blob

        if alt_oG > oG:  # comp of olp_vG to olp_dG, == goes to alt_P or to vP: primary?
            rdn_olp += olp_len  # accumulate redundant overlap in current vP or dP with weaker oG
        else:
            alt_P[8] += olp_len

        rolp = rdn_olp / len(e_)  # relative rdn_olp: redundancy ratio of P to overlapping alt_Ps
        A = a * rolp  # filter is adjusted for redundancy, rolp is not passed?

        P = pri_s, I, D, Dy, M, My, G, e_, A

        P_ = form_blob(type, P, P_, alt_, x)  # _P_ = P_ at line end

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
    D += d    # lateral D for vertical P comp
    Dy += dy  # vertical D, for blob normalization
    M += m    # lateral D for vertical P comp
    My += my  # vertical M, for blob normalization
    G += g    # fd or fv gradient summed to define P value, vs. V = M - 2a * W?

    if type == 0:
        pri = p, g, alt_g  # v gradient, also d, dy, m, my for fuzzy accumulation within P-specific r?
        e_.append(pri)  # pattern element: prior same-line vertex, buffered for selective inc_rng comp
    else:
        e_.append(g)  # pattern element: prior same-line d gradient, buffered for selective inc_der comp

    P = s, I, D, Dy, M, My, G, e_
    olp = olp, oG, alt_oG

    return s, olp, alt_, _alt_, P, alt_P, P_

    # alt_ and _alt_ are accumulated in ycomp over full line, eval in form_blob for fork_eval -> comp_P
    # draft below:


def form_blob(type, P, P_, alt_, x):  # P over _P_ scan, inclusion, displacement

    # merges vertically contiguous and horizontally overlapping same-type and same-sign Ps into P2s,
    # P2: 2D P, generic for blob | vPP | dPP;  alt_-> rdn for fork_eval, then alt2_+= alt_ for P2 eval

    fork_ = deque()  # higher-line matches per P, to assign redundancy and move term _P to next _P
    root_, blob_, buff_ = deque(), deque(), deque()

    vPP_, dPP_, _vPP_, _dPP_ = [],[],[],[]

    if type == 0: _P_ = _vP_
    else: _P_ = _dP_

    s, I, D, Dy, M, My, G, e_, A = P[0] # P_.append(P) = P, alt_ fork_, vPP_, dPP_

    _ix = 0  # initial coordinate of _P displaced from _P_ by last comp_P
    ix = x - len(e_)  # initial coordinate of P
    area = 0  # blob area

    while x >= _ix:  # P to _P connection eval, while horizontal overlap between P and _P:

        fork_oG = 0  # fork overlap gradient: oG += g, approx: oG = G * mw / len(e_)
        ex = x  # coordinate of current P element

        _P = _P_.popleft()  # _P = _P, _alt_, blob, blob_, _vPP_, _dPP_, root_:
        # line y-3 arrays consist of blobs, with one lower-line-exposed blob, but no exposed P

        _ix = P[0][1]  # sub- _P = _s, _ix, _x, _I, _D, _Dy, _M, _My, _G, _e_, _rdn, r
        I2, D2, Dy2, M2, My2, G2, e2_, alt2_, Py_, A = _P[2]  # summed sub-blob vars

        if P[0] == _P[0][0]:  # if s == _s: v or d sign match

            while ex > _ix:

                for e in e_:  # oG accumulation  # PM, PD from comp_P only
                    if type == 0: fork_oG += e[1]  # if vP: e = p, g, alt_g
                    else: fork_oG += e  # if dP: e = g
                    ex += 1

            fork = fork_oG, _P
            fork_.append(fork)  # _P inclusion in P
            _P[5].append(P)  # root_.append(P), to track continuing roots in form_PP

            I2 += I
            D2 += D; Dy2 += Dy  
            M2 += M; My2 += My
            G2 += G
            area += len(e_) # initialized with new fork?
            e2_.append(e_)  # or no separate e2_: Py_( P( e_?
            alt2_ += alt_   # or replaced by alt_blob_?
            Py_.append(_P)  # vertical array of patterns within a blob

            # also A: a * ave rolp * summed fork rolp?

            # sum of all root Ps per fork -> horizontally defined sub-blob,
            # if fork term: sub-blob is assigned to next continuing sub-blob in blob_
            # and summed in wider sub-blob till its term: no continuing connected sub-blob

        if _P[0][2] > ix:  # if _x > ix:

            buff_.append(_P)  # _P with updated root_ is buffered for next-P comp

        else: # no horizontal overlap between _P and next P, _fork_s are evaluated for term_blob

            if (len(root_) == 0 and y > r + 3) or y == Y - 1:  # _P or frame is terminated

                for blob in blob_:
                    blob, _vPP, _dPP = blob  # <= one _vPP and _dPP per higher-line blob:

                    term_P2(blob, A)  # possible 2D P re-orient and re-scan, but no direct recursion
                    if _vPP > 0: term_P2(_vPP, A)  # not for _vPP in _vPP_: only to eval for rdn?
                    if _dPP > 0: term_P2(_dPP, A)

            # if len(root_) > 1: split, sub-blob buff, root blobs inclusion after their term?

            buff_ += _P_  # for form_blob (next P)

    # no horizontal overlap between P and _P, evaluation for P comp to fork_, then buffered:

    if len(fork_) > 0:  # fork_ eval for comp_P -> _vPP_, _dPP_, vPP_, dPP_
        bA = A
        fork_, bA = fork_eval(0, fork_, bA)  # bA *= blob rdn

        if len(vPP_) > 0:
            vA = bA  # vPP_ eval for form_PP -> 2D value pattern, rdn alt_ blobs:
            vPP_, vA = fork_eval(1, vPP_, vA)  # simultaneous dPP form:

            dA = vA  # dPP_ eval for form_PP -> 2D difference pattern, rdn alt_ vPPs:
            dPP_, dA = fork_eval(2, dPP_, dA)

            # individual vPPs and dPPs are also modified in their forks

    P = P, alt_, fork_, vPP_, dPP_  # adding root_ (lower-line matches) at P_ -> _P_ conversion
    P_.append(P)  # P buffered in P_, terminated root Ps are stored in term_?

    # no term and no buff: _P is already included in its root Ps

    if type == 0: _vP_ = buff_
    else: _dP_ = buff_ # _P = _P, alt_, blob_, _vPP_, _dPP_

    return P_


def fork_eval(type, fork_, A):  # fork_ eval for comp_P and _vPP_, _dPP_ append, then eval for form_PP

    # fork eval per P slice or at blob term | split: variation accumulated in 1D or 2D?
    select_ = [] # for fork comp eval, or for _PP form eval

    for fork in fork_: # fork_ is preserved for assignment to last fork

        crit = fork[0]  # selection criterion: oG for fork | PM for vPP | PD for dPP
        rdn = fork[0][9]

        if crit > A:  # comp to _crit in select_,  inclusion in select_

            for select in select_:  # forks are re-evaluated at each rdn increment
                _crit = select[0]
                _rdn = select[0][9]

                if crit > _crit:  # criterion comp

                    _rdn += 1  # increment weaker-fork rdn, initialized per fork?
                    if _crit < A * _rdn: del (select)  # delete select from select_

                else: rdn += 1  # increment weaker-fork rdn

            if crit > A * rdn:  # inclusion after full select_ rdn assignment, also A *= rdn?
                select_.append(fork)

    while len(select_) > 0:
        fork = select_.pop  # no eval, already selected

        if type == 0: comp_P(fork)  # this is a type of fork, for either type of 1D P
        else: form_PP (type, fork)  # separate _vPP_, _dPP_ per _P were appended by comp_P

        # fork access by increasing rdn (max rdn for non-select) or coordinate, no separate max?

    return fork_, A


def comp_P(P, P_, _P, _P_, x):  # forms 2D derivatives of 1D P vars to define vPP and dPP:

    ddx = 0  # optional;

    s, I, D, Dy, M, My, G, e_, oG, rdn, alt_  = P  # select alt_ per fork, no olp: = mx?
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

    # np.array for direct accumulation, or simply iterator of initialization?

    P2_ = np.array([blob, vPP, dPP],
        dtype=[('crit', 'i4'), ('rdn', 'i4'), ('W', 'i4'), ('I2', 'i4'), ('D2', 'i4'), ('Dy2', 'i4'),
        ('M2', 'i4'), ('My2', 'i4'), ('G2', 'i4'), ('rdn2', 'i4'), ('alt2_', list), ('Py_', list)])

    P = s, I, D, Dy, M, My, G, r, e_, alt_, blob_  # _fork_ is empty, similar to tuple declaration?
    # returned Ps also include current derivatives per var?

    P_.append(P)  # _P_ = P_ for next-line comp, if no horizontal overlap between P and next _P

    return P_, _P_  # fork_ is accumulated within blob_eval


def form_PP(PP, fork, root_, blob_, _P_, _P2_, _x, A):  # forms vPPs, dPPs, and their var Ps

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
        blob_.append(fork)  # _P index and PP per fork, possibly multiple forks per P

        root_.append(P)  # connected Ps in future blob_ and _P2_

    return _P_, _P2_  # laced _vB_ and _dB_?


def term_P2(P2, A):  # sub-level 5: eval for rotation, re-scan, re-comp, recursion, accumulation, at PP term?

    term_ = []  # terminated root Ps per next fork P in Py_, last fork (CP) is stored in P2_?
                # accumulated per P2 within root?  incomplete inclusion: no-match Ps -> _P' cont_?

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


def level_1(f):  # last "_" denotes array vs. element, first "_" denotes higher-line variable or pattern

    global r; global a
    r = 1; a = 127  # filters, ultimately set by separate feedback, then a *= r

    global Y; global X
    Y, X = f.shape  # Y: frame height, X: frame width

    global _dP2_; global _vP2_  # 2D Ps terminated on line y-3
    _vP2_, _dP2_ = [],[]

    global y
    y, fd, fv, _vP_, _dP_, frame_ = 0,0,0,[],[],[]

    p_ = f[0, :]   # first line of pixels
    _t_= comp(p_)  # _t_ includes ycomp() results and Dy, My, dG, vG = 0

    for y in range(1, Y):  # vertical coordinate y is index of new line p_

        p_ = f[y, :]
        t_ = comp(p_)  # lateral pixel comp
        _vP_, _dP_ = ycomp(t_, _t_, fd, fv, _vP_, _dP_)  # vertical pixel comp

        # or _vP_, _dP_ are used in form_blob?
        _t_ = t_

        P2_ = _vP2_, _dP2_  # arrays of blobs terminated on current line, adjusted by term_P2
        frame_.append(P2_)  # line of patterns is added to frame of patterns
        _vP2_, _dP2_ = [],[]

    return frame_  # frame of 2D patterns is output to level 2

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
level_1(f)

