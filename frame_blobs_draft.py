import cv2
import argparse
from time import time
from collections import deque

''' frame() is my core algorithm of levels 1 + 2, modified for 2D: segmentation of image into blobs, then search within and between blobs.
    frame_blobs() is frame() restricted to initial definition of blobs per each of 4 derivatives (vs. per 2 gradients in current frame()).

    In my code, Le denotes level of encoding, 
    prefix '_' denotes higher-line variable or pattern, vs. same-type lower-line variable or pattern,
    postfix '_' denotes array name, vs. same-name elements of that array,
    y per line below is shown as relative to y of current line, which is incremented with top-down input within a frame
    
    frame_blobs() performs several steps of encoding, incremental per scan line defined by vertical coordinate y:

    1Le, line y:    x_comp(p_): lateral pixel comparison -> tuple t ) array t_
    2Le, line y- 1: y_comp(t_): vertical pixel comp -> 2D tuple t2 ) array t2_ 
    3Le, line y- 1+ rng: form_P(t2) -> 1D pattern P ) P_  
    4Le, line y- 2+ rng: scan_P_(P, _P) -> _P, fork_, root_: downward and upward connections between Ps of adjacent lines 
    5Le, line y- 3+ rng: form_blob(_P, blob) -> blob: merge connected Ps into non-forking blob segments
    6Le, line y- 4+ rng+ + blob depth: term_blob, form_net -> net: merge connected segments into network of terminated forks
    
    These functions are tested through form_P, I am currently debugging scan_P_. 
    All 2D functions (ycomp, scan_P_, etc.) input two lines: higher and lower, convert elements of lower line 
    into elements of new higher line, and displace elements of old higher line into some higher function.
    Higher-line elements include additional variables, derived while they were lower-line elements.
    
    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match.
    Each vertical and horizontal derivative forms separate blobs, suppressing overlapping orthogonal representations.
    They can also be summed to estimate diagonal or hypot derivatives, for blob orientation to maximize primary derivatives.
    Orientation increases primary dimension of blob to maximize match, and decreases secondary dimension to maximize difference.
'''

def lateral_comp(p_):  # comparison over x coordinate: between min_rng consecutive pixels within each line

    t_ = []  # complete tuples: summation range = rng
    rng_t_ = deque(maxlen=rng)  # array of tuples within rng of current pixel: summation range < rng
    i_rng = rng - 1  # max index of rng_t_
    pri_d, pri_m = 0, 0  # fuzzy derivatives in prior completed tuple

    for p in p_:  # pixel p is compared to rng of prior pixels within horizontal line, summing d and m per prior pixel:
        for index, (pri_p, d, m) in enumerate(rng_t_):

            d += p - pri_p  # fuzzy d: running sum of differences between pixel and all subsequent pixels within rng
            m += min(p, pri_p)  # fuzzy m: running sum of matches between pixel and all subsequent pixels within rng

            if index < i_rng:
                rng_t_[index] = (pri_p, d, m)
            else:
                t_.append((pri_p, d + pri_d, m + pri_m))  # completed bilateral tuple is transferred from rng_t_ to t_
                pri_d = d; pri_m = m

        rng_t_.appendleft((p, 0, 0))  # new tuple with initialized d and m, maxlen displaces completed tuple from rng_t_

    t_ += reversed(rng_t_)  # or tuples of last rng (incomplete, in reverse order) are discarded?
    return t_


def vertical_comp(t_, t2__, _dP_, _vP_, _dyP_, _vyP_, dframe, vframe, dyframe, vyframe):

    # comparison between rng vertically consecutive pixels, forming t2: 2D tuple of derivatives per pixel

    dP  = [0,0,0,0,0,0,0,0,0,[]]  # difference pattern = pri_s, I, D, Dy, M, My, Olps, t2_; initialized per line
    vP  = [0,0,0,0,0,0,0,0,0,[]]  # value pattern = pri_s, I, D, Dy, M, My, Olps, t2_
    dyP = [0,0,0,0,0,0,0,0,0,[]]  # vertical difference pattern = pri_s, I, D, Dy, M, My, Olps, t2_
    vyP = [0,0,0,0,0,0,0,0,0,[]]  # vertical value pattern = pri_s, I, D, Dy, M, My, Olps, t2_

    o_d_v  = (0,0,0)  # alt type overlap between dP and vP = (len, D, V): summed over overlap, within line?
    o_dy_vy= (0,0,0)  # alt type overlap between dyP and vyP = (len, Dy, Vy)
    o_d_dy = (0,0,0)  # alt direction overlap between dP and dyP = (len, D, Dy)
    o_v_vy = (0,0,0)  # alt direction overlap between vP and vyP = (len, V, Vy)
    o_d_vy = (0,0,0)  # alt type, alt direction overlap between dP and vyP = (len, D, Vy)
    o_v_dy = (0,0,0)  # alt type, alt direction overlap between vP and dyP = (len, V, Dy)

    dP_, vP_, dyP_, vyP_ = deque(),deque(),deque(),deque()  # line y - 1+rng
    dbuff_, vbuff_, dybuff_, vybuff_ = deque(),deque(),deque(),deque()  # line y- 2+rng: _Ps buffered by previous run of scan_P_
    dblob_, vblob_, dyblob_, vyblob_ = deque(),deque(),deque(),deque()  # line y- 3+rng: replaces _P_, exposed blobs include _Ps

    new_t2__ = deque()  # t2_s buffered for next line
    x = 0  # lateral coordinate of input pixel
    i_rng = rng - 1  # max t2_ index
    rng2 = rng * 2  # min bilateral lat | vert rng
    dy, my = 0, 0  # for initial rng of lines, to replace _dy, _vy = 0, 0: fuzzy derivatives in higher completed tuple?

    for (p, d, m), (t2_, _dy, _my) in zip(t_, t2__):  # pixel p is compared to rng of higher pixels in t2_, summing dy and my per higher pixel:
        x += 1
        index = 0

        for (_p, _d, _m, dy, my) in t2_:  # 2D tuples are vertically incomplete; prefix '_' denotes higher-line variable

            dy += p - _p   # fuzzy dy: running sum of differences between pixel and all lower pixels within rng
            my += min(p, _p)  # fuzzy my: running sum of matches between pixel and all lower pixels within rng

            if index < i_rng:
                t2_[index] = (_p, d, m, dy, my)

            elif x > rng2-1 and y > rng2:  # -1 for x|y = 0, min vert higher + lower rng is increased by x_comp on line y=0
                _v = _m - ave
                vy = my + _my - ave
                t2 = _p, _d, _v, dy + _dy, vy  # completed bilateral 2D tuples are inputted to form_P (pattern):

                # forms 1D patterns vP, vPy, dP, dPy: horizontal spans of same-sign derivative, each also recording 3 alt_P overlaps per P type:

                dP, vP, dyP, vyP, o_d_v,   o_d_dy, o_d_vy,  dP_,  dbuff_, _dP_,  dblob_, dframe =  form_P(0, t2, x, dP, vP, dyP, vyP, o_d_v,   o_d_dy, o_d_vy, dP_,  dbuff_, _dP_,  dblob_, dframe)
                vP, dP, vyP, dyP, o_d_v,   o_v_vy, o_v_dy,  vP_,  vbuff_, _vP_,  vblob_, vframe =  form_P(1, t2, x, vP, dP, vyP, dyP, o_d_v,   o_v_vy, o_v_dy, vP_,  vbuff_, _vP_,  vblob_, vframe)
                dyP, vyP, dP, vP, o_dy_vy, o_d_dy, o_v_dy, dyP_, dybuff_, _dyP_, dyblob_, dyframe= form_P(2, t2, x, dyP, vyP, dP, vP, o_dy_vy, o_d_dy, o_v_dy, dyP_, dybuff_, _dyP_, dyblob_, dyframe)
                vyP, dyP, vP, dP, o_vy_dy, o_v_vy, o_d_vy, vyP_, vybuff_, _vyP_, vyblob_, vyframe= form_P(3, t2, x, vyP, dyP, vP, dP, o_dy_vy, o_v_vy, o_d_vy, vyP_, vybuff_, _vyP_, vyblob_, vyframe)

            index += 1

        t2_.appendleft((p, d, m, 0, 0))  # initial fdy and fmy = 0, new t2 replaces completed t2 in vertical t2_ via maxlen
        new_t2__.append((t2_, dy, my))  # vertically-incomplete 2D array of tuples, converted to t2__, for next-line ycomp

    # line ends, current patterns are sent to scan_P_, t2s with incomplete lateral fd and fm are discarded?

    if y > rng2 + 2:  # starting with the first returned _P_: vertical interconnection of laterally incomplete patterns:

        dP_, dbuff_, _dP_, dblob_, dframe = scan_P_(0, x, dP, dP_, dbuff_, _dP_, dblob_, dframe)  # returns empty _dP_
        vP_, vbuff_, _vP_, vblob_, vframe = scan_P_(1, x, vP, vP_, vbuff_, _vP_, vblob_, vframe)  # returns empty _vP_
        dyP_, dybuff_, _dyP_, dyblob_, dyframe = scan_P_(3, x, dyP, dybuff_, dyP_, _dyP_, dyblob_, dyframe)  # returns empty _dyP_
        vyP_, vybuff_, _vyP_, vyblob_, vyframe = scan_P_(4, x, vyP, vybuff_, vyP_, _vyP_, vyblob_, vyframe)  # returns empty _vyP_

    # last vertical rng of lines (vertically incomplete t2__) is discarded,
    # but scan_P_ inputs vertically incomplete patterns, to be added to image_to_blobs() at y = Y-1

    return new_t2__, dP_, vP_, dyP_, vyP_, dframe, vframe, dyframe, vyframe  # extended in scan_P_


def form_P(typ, t2, x, P, alt_typ_P, alt_dir_P, alt_txd_P, typ_olp, dir_olp, txd_olp, P_, buff_, _P_, blob_, frame):

    # conditionally terminates and initializes, always accumulates, 1D pattern: dP | vP | dyP | vyP

    p, d, v, dy, vy = t2  # 2D tuple of quadrant derivatives per pixel
    pri_s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_ = P  # olp1: summed typ_olp, olp2: summed dir_olp, olp3: summed txd_olp

    len1, core1, core1a = typ_olp  # D, V | V, D | Dy,Vy | Vy,Dy: each core is summed within len of corresponding overlap
    len2, core2, core2a = dir_olp  # D,Dy | V,Vy | Dy, D | Vy, V; last "a" is for alternative
    len3, core3, core3a = txd_olp  # D,Vy | V,Dy | Dy, V | Vy, D; cores are re-ordered when form_P is called

    if   typ == 0: core = d  # core: derivative that defines corresponding type of pattern
    elif typ == 1: core = v
    elif typ == 2: core = dy  # last "y" is for vertical dimension
    else:          core = vy

    s = 1 if core > 0 else 0  # core = 0 is negative: no selection?
    if s != pri_s:  # x > rng*2 test before form_P, P is terminated, overlaps are evaluated for assignment to alt Ps:

        if typ == 0 or typ == 2:  # core = d | dy, alt cores v and vy are adjusted for reduced projected match of difference:
            core1 *= ave_rate  # core1 = core1.astype(int): 'float' object has no attribute 'astype'?
            core3 *= ave_rate  # core3 = core3.astype(int)

        else:  # core = v | vy, both adjusted for reduced projected match of difference:
            core1a *= ave_rate  # core1a = core1a.astype(int)
            core3a *= ave_rate  # core3a = core3a.astype(int)  # core2 and core2a are same-type: never adjusted

        if core1 < core1a: olp1 += len1
        else: alt_typ_P[6] += len1

        if core2 < core2a: olp2 += len2
        else: alt_dir_P[6] += len2

        if core3 < core3a: olp3 += len3
        else: alt_txd_P[6] += len3

        # immediate redundancy rate: P / (P + olp) length, accumulated in the weaker P | alt P at termination of either
        # projected redundancy rate: P core / (core + alt_cores): presumed alt Ps olp value
        # or added vars |alt_cores|: -> independent same-sign patterns?

        P = pri_s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_  # no ave * alt_rdn / e_: adj < cost?

        if y > rng*2 + 2:  # rng to form P, +1 to return it to _P_ by ycomp, +1 to input _P with empty _root_?
            P_, buff_, _P_, blob_, frame = scan_P_(typ, x, P, P_, buff_, _P_, blob_, frame)  # scans higher-line _Ps for contiguity
        else:
            P_.append((P, x, []))  # empty cont and root_ in the first line of Ps, returned by y_comp as _P_

        I, D, Dy, M, My, olp1, olp2, olp3, t2_ = 0,0,0,0,0,0,0,0,[]  # P initialization
        len1, core1, core1a, len2, core2, core2a, len3, core3, core3a = 0,0,0,0,0,0,0,0,0  # olp init, unless all-vars sum?

    # continued or initialized P and overlap vars are accumulated:

    I += p    # inputs and derivatives are summed as P parameters:
    D += d    # lateral D
    Dy += dy  # vertical D
    V += v    # lateral V
    Vy += vy  # vertical V
    t2_.append(t2)  # buffered for oriented rescan and incremental range | derivation comp

    P = [s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_]

    if typ == 0:   core1 += d;  core1a += v;  core2 += d; core2a += dy; core3 += d; core3a += vy
    elif typ == 1: core1 += v;  core1a += d;  core2 += v; core2a += vy; core3 += v; core3a += dy
    elif typ == 2: core1 += dy; core1a += vy; core2 += dy; core2a += d; core3 += dy; core3a += v
    else:          core1 += vy; core1a += dy; core2 += vy; core2a += v; core3 += vy; core3a += d
    len1 += 1; len2 += 1; len3 += 1

    typ_olp = len1, core1, core1a
    dir_olp = len2, core2, core2a
    txd_olp = len3, core3, core3a

    return P, alt_typ_P, alt_dir_P, alt_txd_P, typ_olp, dir_olp, txd_olp, P_, buff_, _P_, blob_, frame  # accumulated within line


def scan_P_(typ, x, P, P_, _buff_, _P_, blob_, frame):  # P scans shared-x-coordinate _Ps in _P_, forms overlaps

    buff_ = deque()  # displaced _Ps buffered for scan_P_(next P)
    fork_ = []  # _Ps connected to input P
    s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_ = P
    ix = x - len(t2_)  # initial x coordinate of P
    _ix = 0  # initial x coordinate of _P

    while x >= _ix:  # while horizontal overlap between P and _P_;  _Ps are displaced from _P_ if _x > ix, below:

        if len(_buff_) > 0:
            _P, _x, _fork_, cont, roots = _buff_.popleft()  # _Ps buffered in prior run of scan_P_
        else:
            _P, _x, _fork_ = _P_.popleft()  # _P: y-2, _root_: y-3, contains blobs that replace _Ps
            cont = [0,0,0,0,0,0]  # vertical contiguity between P and _P
            roots = 0  # number of Ps connected to current _P
        _ix = _x - len(_P[9])  # len(t2_) in (pri_s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_)

        if s == _P[0]:  # if s == _s (core sign match):

            t2_x = x  # horizontal coordinate of P quadrant loaded to compute contiguity for blob eval:
            olen, op, od, ov, ody, ovy = 0,0,0,0,0,0  # vars of vertical overlap between P and _P:

            while t2_x > _ix:
                for (p, d, v, dy, vy) in reversed(t2_):  # reversal for contents ! olen
                    op += p; od += d; ov += v; ody += dy; ody += vy; olen += 1  # contiguity rate: olp / P?
                    # vs. olen | olen + core, alt core cont is redundant?
                    t2_x -= 1

            cont[0] += olen; cont[1] += op; cont[2] += od; cont[3] += ov; cont[4] += ody; cont[5] += ovy
            # all-_P_ olps per P, but eval per blob

            fork_.append((_P, _fork_))  # _Ps connected to P, for terminated blob transfer to network
            roots += 1  # count of Ps connected to _P

        if _P[2] > ix:  # if _x > ix:
            buff_.append((_P, _x, _fork_, cont, roots))  # for next scan_P_
        else: # no x overlap between _P and next P

            if len(_fork_) == 1 and _fork_[0][1] == 1:  # _P'_fork_ == 1 and target blob'roots (in blob_) == 1
                blob = form_blob(_fork_[0], _P, _x)  # y-2 _P is packed in y-3 blob _fork_[0], binding is passed to y-1 P_?
            else:
                blob = (_P, (_x - len(t2_) / 2), 0, cont, [_P])  # blob init with _P, ave_x = _x - len(t2_)/2, Dx = 0
            _P = blob  # binds blob to fork _Ps in its root Ps?

            if roots == 0:
                net = blob, [(blob, fork_)]  # net is initialized with terminated blob, no fork_ per net?
                if len(_fork_) == 0:
                    frame = form_frame(typ, net, frame)  # all root-mediated forks terminated, net is packed into frame
                else:
                    _fork_, frame = term_blob(typ, net, _fork_, frame)  # recursive root network termination test
            else:
                blob_.append((blob, roots, _fork_))  # new or incremented blobs exposed to P_

    P = s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_  # P with no overlap to next _P is buffered for next-line scan_P_:
    P_.append((P, x, fork_))  # y_comp returns P_ as next-line _P_, packed in blob at y-3

    return P_, buff_, _P_, blob_, frame  # _P_ and buff_ exclude _Ps displaced into blob_


def term_blob(typ, net, fork_, frame):  # net starts as one terminated blob, then added to terminated forks in its fork_

    for index, (_net, _roots, _fork_) in enumerate(fork_):
        _net = form_network(_net, net)  # terminated network (blob) is included into its forks networks
        fork_[index][0] = _net  # return
        _roots -= 1

        if _roots == 0:
            if len(_fork_) == 0:  # no fork-mediated roots left, terminated net is packed in frame:
                frame = form_frame(typ, net, frame)
            else:
                _fork_, frame = term_blob(typ, _net, _fork_, frame)  # recursive root network termination test

    return fork_, frame  # fork_ contains incremented nets


def form_blob(blob, P, last_x):  # continued or initialized blob is incremented by attached _P, replace by zip?

    (s, L2, I2, D2, Dy2, V2, Vy2, Olp1, Olp2, Olp3, t2_), _x, Dx, Cont, Py_ = blob  # also ini fork_:
    (s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_), cont = P  # s is identical, t2_ is a replacement

    x = last_x - len(t2_)/2  # median x, becomes _x in blob, replaces lx
    dx = x - _x  # conditional full comp(x) and comp(S): internal vars are secondary?
    Dx += dx  # for blob normalization and orientation eval, | += |dx| for curved max_L norm, orient?

    L2 += len(t2_)  # t2_ in P buffered in Py_
    I2 += I
    D2 += D; Dy2 += Dy
    V2 += V; Vy2 += Vy
    Olp1 += olp1  # alt-typ overlap per blob
    Olp2 += olp2  # alt-dir overlap per blob
    Olp3 += olp3  # alt-txd overlap per blob
    Cont += cont  # vertical contiguity between Ps in Py, for blob eval?

    Py_.append((s, x, dx, I, D, Dy, V, Vy, olp1, olp2, olp3, cont, t2_))  # dx to normalize P before comp_P?
    blob = (s, L2, I2, D2, Dy2, V2, Vy2, Olp1, Olp2, Olp3, t2_), _x, Dx, Cont, Py_  # redundant s and t2_?

    return blob

def form_network(net, blob):  # continued or initialized network is incremented by attached blob and _root_

    (s, xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn, Olp1n, Olp2n, Olp3n, Contn), blob_ = net  # 2D blob_: fork_ per layer?
    ((s, L2, I2, D2, Dy2, V2, Vy2, Olp1, Olp2, Olp3, t2_), x, Dx, Cont, Py_), fork_ = blob  # s is redundant

    Dxn += Dx  # for net normalization, orient eval, += |Dx| for curved max_L?
    Ln += L2
    In += I2
    Dn += D2; Dyn += Dy2
    Vn += V2; Vyn += Vy2
    Olp1n += Olp1  # alt-typ overlap per net
    Olp2n += Olp2  # alt-dir overlap per net
    Olp3n += Olp3  # alt-txd overlap per net
    Contn += Cont  # vertical contiguity, for comp_P eval?

    blob_.append((x, Dx, L2, I2, D2, Dy2, V2, Vy2, Olp1, Olp2, Olp3, Cont, Py_, fork_))  # Dx to normalize blob before comp_P
    net = (s, xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn, Olp1n, Olp2n, Olp3n, Contn), blob_  # separate S_par tuple?

    return net

def form_frame(typ, net, frame):
    (s, xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn, Olp1n, Olp2n, Olp3n, Contn), blob_ = net

    if typ:
        Dxf, Lf, Contf, net_ = frame  # summed per typ regardless of sign, no typ olp: complete overlap?
    else:
        Dxf, Lf, If, Df, Dyf, Vf, Vyf, Contf, net_ = frame

    Dxf += Dxn  # for frame normalization, orient eval, += |Dxn| for curved max_L?
    Lf += Ln
    Contf += Contn  # for average vertical contiguity, relative to ave Ln?
    net_.append((xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn, Olp1n, Olp2n, Olp3n, Contn, blob_))  # Dxn to normalize net before comp_P

    if typ:
        frame = s, Dxf, Lf, Contf, net_
    else:
        If += In  # to compute averages, for dframe only: redundant for same-scope alt_frames?
        Df += Dn
        Dyf += Dyn
        Vf += Vn
        Vyf += Vyn
        frame = Dxf, Lf, Contf, net_, If, Df, Dyf, Vf, Vyf

    return frame


def image_to_blobs(f):  # postfix '_' distinguishes array vs. element, prefix '_' distinguishes higher-line vs. lower-line variable

    _dP_, _vP_, _dyP_, _vyP_ = deque(),deque(),deque(),deque()  # higher-line same- d-, v-, dy-, vy- sign 1D patterns

    dframe = 0,0,0,0,[], 0,0,0,0,0  # s, Dxf, Lf, yOlpf, net_, If, Df, Dyf, Vf, Vyf
    vframe = 0,0,0,0,[]  # s, Dxf, Lf, yOlpf, net_,  no If, Df, Dyf, Vf, Vyf: same as in dframe
    dyframe = 0,0,0,0,[]
    vyframe = 0,0,0,0,[]  # constituent net rep may select sum | [nets], same for blob?

    global y; y = 0  # vertical coordinate of current input line

    t2_ = deque(maxlen=rng)  # vertical buffer of incomplete quadrant tuples, for fuzzy ycomp
    t2__ = []  # vertical buffer + horizontal line: 2D array of 2D tuples, deque for speed?
    p_ = f[0, :]  # first line of pixels
    t_ = lateral_comp(p_)  # after part_comp (pop, no t_.append) while x < rng?

    for (p, d, m) in t_:
        t2 = p, d, m, 0, 0  # dy, my initialized at 0
        t2_.append(t2)  # only one tuple per first-line t2_
        t2__.append((t2_, 0, 0))  # _dy, _my initialized at 0

    for y in range(1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?

        p_ = f[y, :]  # vertical coordinate y is index of new line p_
        t_ = lateral_comp(p_)  # lateral pixel comparison
        # vertical pixel comparison:

        t2__, _dP_, _vP_, _dyP_, _vyP_, dframe, vframe, dyframe, vyframe = \
        vertical_comp(t_, t2__, _dP_, _vP_, _dyP_, _vyP_, dframe, vframe, dyframe, vyframe)

    # frame ends, last vertical rng of incomplete t2__ is discarded,
    # but vertically incomplete P_ patterns are still inputted in scan_P_?

    return dframe, vframe, dyframe, vyframe  # frame of 2D patterns is outputted to level 2

# pattern filters: eventually updated by higher-level feedback, initialized here as constants:

rng = 2  # number of leftward and upward pixels compared to each input pixel
ave = 127 * rng  # average match: value pattern filter
ave_rate = 0.25  # average match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/racoon.jpg')
arguments = vars(argument_parser.parse_args())

# read image as 2d-array of pixels (gray scale):
image = cv2.imread(arguments['image'], 0).astype(int)
Y, X = image.shape  # image height and width

start_time = time()
blobs = image_to_blobs(image)
end_time = time() - start_time
print(end_time)
