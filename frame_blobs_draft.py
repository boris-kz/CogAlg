import cv2
import argparse
from time import time
from collections import deque

''' frame() is my core algorithm of levels 1 + 2, modified for 2D: segmentation of image into blobs, then search within and between blobs.
    frame_blobs() is frame() restricted to initial definition of blobs per each of 4 derivatives (vs. per 2 gradients in current frame()).

    In my code, Le denotes level of encoding, 
    prefix '_' denotes higher-line variable or pattern, vs. same-type lower-line variable or pattern,
    postfix '_' denotes array name, vs. same-name elements of that array,
    y per line below is shown as relative to y of current line, where lines are inputted top-down within a frame
    
    frame_blobs performs several steps of encoding, incremental per scan line defined by vertical coordinate y:

    1Le, line y:    x_comp(p_): lateral pixel comparison -> tuple t ) array t_
    2Le, line y- 1: y_comp(t_): vertical pixel comp -> quadrant t2 ) array t2_ 
    3Le, line y- 1+ rng: form_P(t2) -> 1D pattern P ) P_  
    4Le, line y- 2+ rng: scan_P_(P, _P) -> _P, fork_, root_: downward and upward connections between Ps of adjacent lines 
    5Le, line y- 3+ rng+ blob depth: incr_blob(_P, blob) -> blob: merge connected Ps into non-forking blob segments
    6Le, line y- 4+ rng+ network depth: incr_net(blob, term_forks) -> net: merge connected segments into network of terminated forks
    
    These functions are tested through form_P, I am currently debugging scan_P_. 
    All 2D functions (ycomp, scan_P_, etc.) input two lines: higher and lower, convert elements of lower line 
    into elements of new higher line, and displace elements of old higher line into some higher function.
    Higher-line elements include additional variables, derived while they were lower-line elements.
    
    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match.
    Vertical and horizontal derivatives form separate blobs, each suppressing overlapping orthogonal representations.
    They can also be summed to estimate diagonal or hypot derivatives, for blob orientation to maximize primary derivatives.
    Orientation increases primary dimension of blob to maximize match, and decreases secondary dimension to maximize difference.

    Not implemented here: alternative segmentation by quadrant gradient (less predictive due to directional info loss): 
    Pixels are discrete samples of continuous image, so rightward and downward derivatives per pixel are 
    equally representative samples of 0-90 degree quadrant gradient: minimal unique unit of 2D gradient. 
    Thus, quadrant gradient is estimated as the average of these two orthogonally diverging derivatives.
    Resulting blob would be contiguous area of same-sign quadrant gradient, of difference for dblob or match deviation for vblob.
'''

def lateral_comp(p_):  # comparison over x coordinate: between min_rng consecutive pixels within each line

    t_ = []  # complete fuzzy tuples: summation range = rng
    rng_t_ = deque(maxlen=rng)  # array of fuzzy tuples within rng of current pixel: summation range < rng
    i_rng = rng - 1  # max index of rng_t_

    for p in p_:  # each pixel in horizontal line is compared to rng prior pixels, summing d and m per prior pixel:
        for index, (pri_p, d, m) in enumerate(rng_t_):

            d += p - pri_p  # fuzzy d: running sum of differences between pixel and all subsequent pixels in rng
            m += min(p, pri_p)  # fuzzy m: running sum of matches between pixel and all subsequent pixels in rng

            if index < i_rng:
                rng_t_[index] = (pri_p, d, m)
            else:
                t_.append((pri_p, d, m))  # completed tuple is transferred from it_ to t_

        rng_t_.appendleft((p, 0, 0))  # fd and fm of new tuple initialized at 0, maxlen displaces completed tuple from rng_t_

    t_ += reversed(rng_t_)  # or tuples of last rng (incomplete, in reverse order) are discarded?
    return t_


def vertical_comp(t_, t2__, _dP_, _vP_, _dyP_, _vyP_, dframe, vframe, dyframe, vyframe):

    # comparison between rng vertically consecutive pixels, forming t2: quadrant of derivatives per pixel

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

    dP_, vP_, dyP_, vyP_ = deque(),deque(),deque(),deque()
    dbuff_, vbuff_, dybuff_, vybuff_ = deque(),deque(),deque(),deque()  # _Ps buffered by previous run of scan_P_
    _dPi, _vPi, _dyPi, _vyPi = 0, 0, 0, 0  # indices of current _P in _P_, to replace _P with its blob in scan_P_

    new_t2__ = deque()  # t2_ buffer: 2D array of quadrant tuples
    x = 0  # horizontal coordinate of input pixel
    i_rng = rng - 1  # max t2_ index

    for (p, d, m), t2_ in zip(t_, t2__):  # compares same-x column pixels within vertical range = rng
        x += 1
        index = 0

        for (_p, _d, _m, dy, my) in t2_:  # quadrant tuples are vertically incomplete; prefix '_' denotes higher-line variable

            dy += p - _p   # fuzzy dy: running sum of differences between pixel and all lower pixels within rng
            my += min(p, _p)  # fuzzy my: running sum of matches between pixel and all lower pixels within rng

            if index < i_rng:
                t2_[index] = (_p, d, m, dy, my)
            else:
                _v = _m - ave
                vy = my - ave
                t2 = _p, _d, _v, dy, vy  # completed quadrants are inputted to form_P:

                # form 1D patterns vP, vPy, dP, dPy: horizontal spans of same-sign derivative, recording 3 overlapping alt Ps per P type:

                dP, vP, dyP, vyP, o_d_v,   o_d_dy, o_d_vy,  dP_,  dbuff_, _dP_,  _dPi, dframe =  form_P(0, t2, x, dP, vP, dyP, vyP, o_d_v,   o_d_dy, o_d_vy, dP_,  dbuff_, _dP_,  _dPi, dframe)
                vP, dP, vyP, dyP, o_d_v,   o_v_vy, o_v_dy,  vP_,  vbuff_, _vP_,  _vPi, vframe =  form_P(1, t2, x, vP, dP, vyP, dyP, o_d_v,   o_v_vy, o_v_dy, vP_,  vbuff_, _vP_,  _vPi, vframe)
                dyP, vyP, dP, vP, o_dy_vy, o_d_dy, o_v_dy, dyP_, dybuff_, _dyP_, _dyPi, dyframe= form_P(2, t2, x, dyP, vyP, dP, vP, o_dy_vy, o_d_dy, o_v_dy, dyP_, dybuff_, _dyP_, _dyPi, dyframe)
                vyP, dyP, vP, dP, o_vy_dy, o_v_vy, o_d_vy, vyP_, vybuff_, _vyP_, _vyPi, vyframe= form_P(3, t2, x, vyP, dyP, vP, dP, o_dy_vy, o_v_vy, o_d_vy, vyP_, vybuff_, _vyP_, _vyPi, vyframe)

            index += 1

        t2_.appendleft((p, d, m, 0, 0))  # initial fdy and fmy = 0, new t2 replaces completed t2 in vertical t2_ via maxlen
        new_t2__.append(t2_)     # vertically-incomplete tuple array is transferred to next t2__, for next-line ycomp

    # line ends, current patterns are sent to scan_P_, t2s with incomplete lateral fd and fm are discarded?

    if y > rng + 3:  # starting with the first line of complete t2s: vertical interconnection of laterally incomplete patterns:

        dP_, dbuff_, _dP_, _dPi, dframe = scan_P_(0, x, dP, dP_, dbuff_, _dP_, _dPi, dframe)  # returns empty _dP_
        vP_, vbuff_, _vP_, _vPi, vframe = scan_P_(1, x, vP, vP_, vbuff_, _vP_, _vPi, vframe)  # returns empty _vP_
        dyP_, dybuff_, _dyP_, _dyPi, dyframe = scan_P_(3, x, dyP, dybuff_, dyP_, _dyP_, _dyPi, dyframe)  # returns empty _dyP_
        vyP_, vybuff_, _vyP_, _vyPi, vyframe = scan_P_(4, x, vyP, vybuff_, vyP_, _vyP_, _vyPi, vyframe)  # returns empty _vyP_

    # last vertical rng of lines (vertically incomplete t2__) is discarded,
    # but scan_P_ inputs vertically incomplete patterns, to be added to image_to_blobs() at y = Y-1

    return new_t2__, dP_, vP_, dyP_, vyP_, dframe, vframe, dyframe, vyframe  # extended in scan_P_


def form_P(typ, t2, x, P, alt_typ_P, alt_dir_P, alt_txd_P, typ_olp, dir_olp, txd_olp, P_, buff_, _P_, _Pi, frame):

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
    if s != pri_s and x > rng + 2:  # P is terminated, overlaps are evaluated for assignment to alt Ps:

        if typ == 0 or typ ==2:  # core = d | dy, alt cores v and vy are adjusted for reduced projected match of difference:
            core1 *= ave_rate; core1 = core1.astype(int)
            core3 *= ave_rate; core3 = core3.astype(int)

        else:  # core = v | vy, both adjusted for reduced projected match of difference:
            core1a *= ave_rate; core1a = core1a.astype(int)
            core3a *= ave_rate; core3a = core3a.astype(int)  # core2 and core2a are same-type: never adjusted

        if core1 < core1a: olp1 += len1  # length of olp is accumulated in the weaker alt P until P or alt P terminates
        else: alt_typ_P[6] += len1   # or all alt cores sum: eval per blob only?

        if core2 < core2a: olp2 += len2
        else: alt_dir_P[6] += len2

        if core3 < core3a: olp3 += len3
        else: alt_txd_P[6] += len3

        P = pri_s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_  # no ave * alt_rdn / e_: adj < cost?

        if y > rng + 2:  # rng to form P, +1 to return it to _P_ by ycomp, +1 to input _P with empty _root_?
            P_, buff_, _P_, _Pi, frame = scan_P_(typ, x, P, P_, buff_, _P_, _Pi, frame)  # scans higher-line _Ps for contiguity
        else:
            P_.append((P, x, (0,0), []))  # empty cont and root_ in the first line of Ps, returned by y_comp as _P_

        I, D, Dy, M, My, olp1, olp2, olp3, t2_ = 0,0,0,0,0,0,0,0,[]  # P initialization
        len1, core1, core1a, len2, core2, core2a, len3, core3, core3a = 0,0,0,0,0,0,0,0,0  # olp init

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

    return P, alt_typ_P, alt_dir_P, alt_txd_P, typ_olp, dir_olp, txd_olp, P_, buff_, _P_, _Pi, frame  # accumulated within line


def scan_P_(typ, x, P, P_, _buff_, _P_, _P_index, frame):  # P scans shared-x-coordinate _Ps in _P_, forms overlaps

    buff_ = deque()  # displaced _Ps buffered for scan_P_(next P)
    root_ = []  # _Ps connected to current P, vs. fork_: Ps connected to current _P
    cont = (0,0)
    s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_ = P
    ix = x - len(t2_)  # initial x coordinate of P
    _ix = 0  # initial x coordinate of _P

    while x >= _ix:  # while horizontal overlap between P and _P_;  _Ps are displaced from _P_ if _x > ix, below:

        if len(_buff_) > 0:
            _P, _x, cont, _root_, fork_, _P_index = _buff_.popleft()  # _Ps buffered in prior run of scan_P_
        else:
            _P, _x, cont, _root_ = _P_.popleft()  # _P: y-2, _root_: y-3, contains blobs that replace _Ps at the index:
            _P_index += 1; fork_ = []  # Ps connected to current _P
        _ix = _x - len(_P[9])  # len(t2_) in (pri_s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_)

        if s == _P[0]:  # if s == _s (core sign match):

            t2_x = x  # horizontal coordinate of P quadrant loaded to compute contiguity for blob eval:
            olen, core = 0, 0  # len and core of vertical overlap between P and _P:

            while t2_x > _ix:
                for t2 in reversed(t2_):  # reversal for core
                    if   typ == 0: core += t2[1]  # t2 = p, d, v, dy, vy
                    elif typ == 1: core += t2[2]  # core is summed within overlap for full-blob contiguity eval?
                    elif typ == 2: core += t2[3]  # other t2 vars don't define contiguity? overflow?
                    else:          core += t2[4]
                    olen += 1  # overlap length
                    t2_x -= 1

            Olen, Core = cont  # vertical contiguity between P and _P
            Olen += olen; Core += core  # sum per fork ! root: down sum-> blob
            cont = Olen, Core

            root_.append((_P, _root_))  # _Ps connected to P, for terminated segment transfer to network
            fork_.append((P, x))  # Ps connected to _P, for terminated _P transfer to segment; cont for eval per blob, not fork | P?

        if _P[2] > ix:  # if _x > ix:
            buff_.append((_P, _x, cont, _root_, fork_, _P_index))  # for next scan_P_
        else: # no x overlap between _P and next P, _P is displaced from _P_:

            if len(_root_) == 1 and len(_root_[0][2]) == 1:  # _P'_root_ == 1 and target blob'_fork_ == 1
                blob = form_blob(_root_[0], _P, _x)  # y-2 _P is packed in continued y-3 blob
            else:
                blob = (_P, _x, 0, cont, [_P])  # blob init with _P( s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_), Dx = 0

            if len(fork_) == 0:  # blob is terminated and packed:
                net = (blob, [blob], [])  # net initialized with root_ = [], fork_ is lost
                _root_, frame = term_blob(_ix, typ, net, _root_, frame)  # net is recursively summed into higher _net

            _P_[_P_index] = blob, fork_, _root_  # blob replaces _P in y-3 _P_, addressed by P root_ and then by _P _root_

    P = s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_  # no overlap between P and next _P, P is buffered for next line:
    P_.append((P, x, cont, root_))  # y_comp returns P_ as _P_, for next-line scan_P_, cont is per root?

    return P_, buff_, _P_, _P_index, frame  # _P_ and buff_ exclude _Ps displaced into _fork_s per blob


def term_blob(_ix, typ, net, _root_, frame):  # terminated blob (init network) is included into higher forks networks

    for index, (_net, _fork_, __root_) in enumerate(_root_):

        for iindex, _P in enumerate(_fork_):
            if _P[1] == _ix:
                del _fork_[iindex]; break  # blob is moved from _fork_ to _net:

        _net = form_network(_net, net, _root_)  # net represents all terminated forks, including current blob
        _root_[index][0] = _net  # return

        if len(__root_) == 0 and len(_fork_) == 0:  # full term: no root-mediated forks left in _net
            frame = form_frame(typ, frame, net)  # terminated net is packed in frame, initialized in root function

        if len(_fork_) == 0:
            _root_, frame = term_blob(_ix, typ, _net, __root_, frame)  # recursive root net termination

    return _root_, frame  # _root_ refers to net


def form_blob(blob, P, lx):  # continued or initialized blob is incremented by attached _P, replace by zip?

    s, _x, L2, I2, D2, Dy2, V2, Vy2, Olp1, Olp2, Olp3, Cont, Py_, Dx = blob
    (s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_), cont = P  # s is identical

    x = lx - len(t2_)/2  # median x, becomes _x in blob, replaces lx
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
    blob = s, x, Dx, L2, I2, D2, Dy2, V2, Vy2, Olp1, Olp2, Olp3, Cont, Py_  # separate S_par tuple?

    return blob

def form_network(network, blob, _root_):  # continued or initialized network is incremented by attached blob and _root_

    s, xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn, Olp1n, Olp2n, Olp3n, Contn, blob_ = network  # or S_par tuple?
    s, x, Dx, L2, I2, D2, Dy2, V2, Vy2, Olp1, Olp2, Olp3, Cont, Py_ = blob  # s is redundant

    Dxn += Dx  # for net normalization, orient eval, += |Dx| for curved max_L?
    Ln += L2
    In += I2
    Dn += D2; Dyn += Dy2
    Vn += V2; Vyn += Vy2
    Olp1n += Olp1  # alt-typ overlap per net
    Olp2n += Olp2  # alt-dir overlap per net
    Olp3n += Olp3  # alt-txd overlap per net
    Contn += Cont  # vertical contiguity, for comp_P eval?

    blob_.append((x, Dx, L2, I2, D2, Dy2, V2, Vy2, Olp1, Olp2, Olp3, Cont, Py_, _root_))  # Dx to normalize blob before comp_P
    network = s, xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn, Olp1n, Olp2n, Olp3n, Contn, blob_  # separate S_par tuple?

    return network

def form_frame(typ, frame, net):
    s, xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn, Olp1n, Olp2n, Olp3n, Contn, blob_ = net

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

''' select by projected combined redundancy rate per term blob | network, for extended comp or select decoding:
    rdn coef: (Core + alt_typ_Core + alt_dir_Core + alt_txd_Core) / Core: proj cost / rep, alt rdn assign per olp

    olp rdn assign to min P_comb_vars: comb within P, same as comb match, 
    P value = P_comb_vars / rdn between Ps: total or to stronger olp?

    dir comb for projected oriented blob eval: (Core^2 + alt_dir_Core ^ 2): ver_P_ ( lat_P_?   
    fuzzy m: actual rep value per pixel, for more selective patterns of t, also within der_que if > min_len?
'''

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
        t2 = p, d, m, 0, 0  # fdy and fmy initialized at 0; alt named quad?
        t2_.append(t2)  # only one tuple per first-line t2_
        t2__.append(t2_)  # in same order as t_

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

rng = 1  # number of leftward and upward pixels compared to each input pixel
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
