import cv2
import argparse
from time import time
from collections import deque

''' Version of frame_draft() restricted to defining blobs, but extended to blob per each of 4 derivatives, vs. per 2 gradient types

    Core algorithm of levels 1 + 2, modified for 2D: segmentation of image into blobs. It performs several steps of encoding, 
    incremental per scan line defined by vertical coordinate y, relative to y of current input line.  nLe: level of encoding:

    1Le, line y:    comp(p_): lateral pixel comparison -> tuple t ) array t_
    2Le, line y- 1: ycomp(t_): vertical pixel comp -> quadrant t2 ) array t2_ 
    3Le, line y- 1+ rng: form_P(t2) -> 1D pattern P ) P_  
    4Le, line y- 2+ rng: scan_P_(P, _P) -> _P, fork_, root_: downward and upward connections between Ps of adjacent lines 
    5Le, line y- 3+ rng+ blob depth: incr_blob(_P, blob) -> blob: merge connected Ps into non-forking blob segments
    6Le, line y- 4+ rng+ network depth: incr_net(blob, term_forks) -> net: merge connected segments into network of terminated forks

    All 2D functions (ycomp, scan_P_, etc.) input two lines: higher and lower, convert elements of lower line 
    into elements of new higher line, and displace elements of old higher line into some higher function.
    Higher-line elements include additional variables, derived while they were lower-line elements.
    In my code, prefix '_' distinguishes higher-line variable or pattern from same-type lower-line variable or pattern 
    
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

    None of this is tested, except as analogue functions in line_POC() 
    '''
    # in the code below,
    # prefix '_' distinguishes higher-line variable or pattern from same-type lower-line variable or pattern
    # postfix '_' denotes array name, vs. same-name elements of that array


def horizontal_comp(p_):  # comparison between rng consecutive pixels within each line

    t_ = []  # complete fuzzy tuples: summation range = rng
    it_ = deque(maxlen=rng)  # incomplete fuzzy tuples: summation range < rng

    for p in p_:  # each pixel in horizontal line is compared to rng prior pixels, summing d and m per pixel:

        for index, it in enumerate(it_):
            pri_p, fd, fm = it

            d = p - pri_p  # difference between pixels
            m = min(p, pri_p)  # match between pixels

            fd += d  # fuzzy d: sum of ds between p and all prior ps within it_
            fm += m  # fuzzy m: sum of ms between p and all prior ps within it_

            it_[index] = (pri_p, fd, fm)

        if len(it_) == rng:  # or separate while x < rng: icomp(){ p = pop(p_).., no t_.append?
            t_.append((pri_p, fd, fm))  # completed tuple is transferred from it_ to t_

        it_.appendleft((p, 0, 0))  # new prior tuple, fd and fm are initialized at 0

    t_ += reversed(it_)  # or tuples of last rng (incomplete, in reverse order) are discarded?
    return t_


def vertical_comp(t_, t2__, _dP_, _vP_, _dyP_, _vyP_, _dPi, _vPi, _dyPi, _vyPi, dframe, vframe, dyframe, vyframe):

    # vertical comparison between rng consecutive pixels across lines, forming t2: quadrant of derivatives per pixel

    dP = [0,0,0,0,0,0,0,0,0,[]]  # difference pattern = pri_s, I, D, Dy, M, My, Olps, t2_; initialized per line
    vP = [0,0,0,0,0,0,0,0,0,[]]  # value pattern = pri_s, I, D, Dy, M, My, Olps, t2_
    dyP = [0,0,0,0,0,0,0,0,0,[]]  # vertical difference pattern = pri_s, I, D, Dy, M, My, Olps, t2_
    vyP = [0,0,0,0,0,0,0,0,0,[]]  # vertical value pattern = pri_s, I, D, Dy, M, My, Olps, t2_

    o_d_v  = (0,0,0)  # alt type overlap between dP and vP = (len, D, V): summed over overlap, within line?
    o_dy_vy= (0,0,0)  # alt type overlap between dyP and vyP = (len, Dy, Vy)
    o_d_dy = (0,0,0)  # alt direction overlap between dP and dyP = (len, D, Dy)
    o_v_vy = (0,0,0)  # alt direction overlap between vP and vyP = (len, V, Vy)
    o_d_vy = (0,0,0)  # alt type, alt direction overlap between dP and vyP = (len, D, Vy)
    o_v_dy = (0,0,0)  # alt type, alt direction overlap between vP and dyP = (len, V, Dy)

    dP_, vP_, dyP_, vyP_ = [],[],[],[]
    dbuff_, vbuff_, dybuff_, vybuff_ = deque(),deque(),deque(),deque()  # _Ps buffered by previous run of scan_P_
    new_t2__ = deque()  # t2_ buffer: 2D array of quadrant tuples
    x = 0  # horizontal coordinate

    for index, t, t2_ in enumerate(zip(t_, t2__)):  # compares same-x column pixels within vertical rng
        p, fd, fm = t
        x += 1

        for t2 in t2_:  # all t2s (quadrant tuples) are horizontally incomplete
            pri_p, _fd, _fm, fdy, fmy = t2  # prefix '_' denotes higher-line variable or pattern

            dy = p - pri_p  # vertical difference between pixels
            my = min(p, pri_p)  # vertical match between pixels

            fdy += dy  # fuzzy dy: sum of dy between p and all prior ps within t2_
            fmy += my  # fuzzy my: sum of my between p and all prior ps within t2_

            t2_[index] = pri_p, _fd, _fm, fdy, fmy

        if len(t2_) == rng:  # or while y < rng: i_ycomp(): t2_ = pop(t2__), t = pop(t_)., no form_P?

            fv = _fm - ave
            fvy = fmy - ave
            t2 = pri_p, _fd, fv, fdy, fvy  # completed quadrants are moved from t2_ to form_P:

            # form 1D patterns vP, vPy, dP, dPy: horizontal spans of same-sign derivative, with associated vars, 3 olp per P type:

            dP, vP, dyP, vyP, o_d_v,   o_d_dy, o_d_vy, dP_, _dP_, _dPi,  dbuff_, dframe =    form_P(0, t2, x, dP, vP, dyP, vyP, o_d_v,   o_d_dy, o_d_vy, dP_,  _dP_,  _dPi, dbuff_, dframe)
            vP, dP, vyP, dyP, o_d_v,   o_v_vy, o_v_dy, vP_, _vP_, _vPi,  vbuff_, vframe =    form_P(1, t2, x, vP, dP, vyP, dyP, o_d_v,   o_v_vy, o_v_dy, vP_,  _vP_,  _vPi, vbuff_, vframe)
            dyP, vyP, dP, vP, o_dy_vy, o_d_dy, o_v_dy, dyP_, _dyP_, _dyPi, dybuff_, dyframe= form_P(2, t2, x, dyP, vyP, dP, vP, o_dy_vy, o_d_dy, o_v_dy, dyP_, _dyP_, _dyPi, dybuff_, dyframe)
            vyP, dyP, vP, dP, o_vy_dy, o_v_vy, o_d_vy, vyP_, _vyP_, _vyPi, vybuff_, vyframe= form_P(3, t2, x, vyP, dyP, vP, dP, o_dy_vy, o_v_vy, o_d_vy, vyP_, _vyP_, _vyPi, vybuff_, vyframe)

        t2_.append((p, fd, fm, 0, 0))  # initial fdy and fmy = 0, new t2 replaces completed t2 in t2_
        new_t2__.appendleft(t2_)     # vertically-incomplete tuple array is transferred to next t2__, for next-line ycomp

    # line ends, current patterns are sent to scan_P_, t2s with incomplete lateral fd and fm are discarded?

    if y + 1 > rng:  # starting with the first line of complete t2s: vertical interconnection of laterally incomplete patterns:

        dP_, _dP_, _dPi, dbuff_, dframe = scan_P_(0, x, dP, dP_, _dP_, _dPi, dbuff_, dframe)  # returns empty _dP_
        vP_, _vP_, _vPi, vbuff_, vframe = scan_P_(1, x, vP, vP_, _vP_, _vPi, vbuff_, vframe)  # returns empty _vP_
        dyP_, _dyP_, _dyPi, dybuff_, dyframe = scan_P_(3, x, dyP, dyP_, _dyP_, _dyPi, dybuff_, dyframe)  # returns empty _dyP_
        vyP_, _vyP_, _vyPi, vybuff_, vyframe = scan_P_(4, x, vyP, vyP_, _vyP_, _vyPi, vybuff_, vyframe)  # returns empty _vyP_

    # last vertical rng of lines (vertically incomplete t2__) is discarded,
    # but scan_P_ inputs vertically incomplete patterns, to be added to image_to_blobs() at y = Y-1

    return new_t2__, _dP_, _vP_, _dyP_, _vyP_, _dPi, _vPi, _dyPi, _vyPi, dframe, vframe, dyframe, vyframe  # extended in scan_P_


def form_P(typ, t2, x, P, alt_typ_P, alt_dir_P, alt_txd_P, typ_olp, dir_olp, txd_olp, P_, _P_, _Pi, buff_, frame):

    # conditionally terminates and initializes, always accumulates, 1D pattern: dP | vP | dyP | vyP

    p, d, v, dy, vy = t2  # 2D tuple of quadrant derivatives per pixel, all fuzzy
    pri_s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_ = P  # olp1: summed typ_olp, olp2: summed dir_olp, olp3: summed txd_olp

    len1, core1, core1a = typ_olp  # D, V | V, D | Dy,Vy | Vy,Dy: each core is summed within len of corresponding overlap
    len2, core2, core2a = dir_olp  # D,Dy | V,Vy | Dy, D | Vy, V; last "a" is for alternative
    len3, core3, core3a = txd_olp  # D,Vy | V,Dy | Dy, V | Vy, D; cores are re-ordered when form_P is called

    if   typ == 0: core = d  # core: derivative that defines corresponding type of pattern
    elif typ == 1: core = v
    elif typ == 2: core = dy  # last "y" is for vertical
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
        else: alt_typ_P[6] += len1      # but not alt core for retroactive P eval: cost > gain?

        if core2 < core2a: olp2 += len2
        else: alt_dir_P[6] += len2

        if core3 < core3a: olp3 += len3
        else: alt_txd_P[6] += len3

        P = pri_s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_  # no ave * alt_rdn / e_: adj < cost?
        P_, _P_, _Pi, buff_, frame = scan_P_(typ, x, P, P_, _P_, _Pi, buff_, frame)  # scans higher-line _Ps

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

    return P, alt_typ_P, alt_dir_P, alt_txd_P, typ_olp, dir_olp, txd_olp, P_, _P_, _Pi, buff_, frame  # accumulated within line


def scan_P_(typ, x, P, P_, _P_, _P_index, _buff_, frame):  # P scans shared-x-coordinate _Ps in _P_, forms overlaps

    buff_ = deque()  # displaced _Ps buffered for scan_P_(next P)
    root_ = []  # for _Ps connected to current P
    fork_ = []  # for Ps connected to current _P

    s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_ = P
    ix = x - len(t2_)  # initial x coordinate of P
    _ix = 0  # initialized ix of last _P displaced from _P_

    while x >= _ix:  # while horizontal overlap between P and _P_:

        if _buff_:  # if len(_buff_) > 0
            _P, _P_index, _root_ = _buff_.popleft()  # _Ps buffered in prior run of scan_P_
        else:
            _P, _root_ = _P_.popleft()  # _P: y-2, _root_: y-3, contains blobs that replace _Ps at the index:
            _P_index += 1  # not for buff_: already contains index

        _ix = _P[1]
        if s == _P[0]:  # if s == _s (core sign match):

            t2_x = x  # horizontal coordinate of quadrant loaded to compute overlap, for blob contiguity eval?
            olen, core = 0, 0  # horizontal overlap between P and _P:
            while t2_x > _ix:
                for t2 in t2_:
                    if   typ == 0: core += t2[1]  # t2 = p, d, v, dy, vy
                    elif typ == 1: core += t2[2]  # core is summed within overlap for full-blob contiguity eval?
                    elif typ == 2: core += t2[3]  # other t2 vars don't define contiguity?
                    else:          core += t2[4]
                    olen += 1  # overlap length
                    t2_x += 1
            olp = olen, core  # vertical contiguity between P and _P, distinct from olp1, olp2, olp3 between alt Ps

            root_.append(((_P, _root_), olp))  # _Ps connected to P, for terminated segment transfer to network
            fork_.append((P, ix, olp))  # Ps connected to _P, for terminated _P transfer to segment

        if _P[2] > ix:  # if _x > ix:
            buff_.append((_P, _P_index, _root_))  # for next scan_P_
        else: # no horizontal overlap between _P and next P, _P is displaced from _P_:

            if len(_root_) == 1 and len(_root_[0][2]) == 1:  # _P'_root_ == 1 and target blob'_fork_ == 1
                blob = incr_blob(_root_[0], (_P, olp))  # y-2 _P is packed in continued y-3 blob
                init = 0
            else:
                blob = (_P, 0, olp, [_P])  # blob init with _P = s, ix, x, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_
                init = 1

            while len(fork_) == 0:  # blob inclusion into terminated forks network, conditionally recursive for higher layers:
                for index, (_net, _blob, _fork_, __root_) in enumerate(_root_):

                    for iindex, _P in enumerate(_fork_):
                        if _P[1] == _ix:
                            del _fork_[iindex]; break  # blob is moved from _fork_ to _net:

                    if init:  # net init at _P term: 1st run of while, to sum into higher _net, or future lower blobs at term
                        net = (blob, [blob], [])  # blob is packed into net with root_ = []: net up ! down: fork_ was deleted
                    else:
                        _net = incr_network(_net, net, _root_)  # net represents all terminated forks, including current blob
                        _root_[index][0] = _net  # return

                    if len(__root_) == 0 and len(_fork_) == 0:  # full term: no root-mediated forks left in _net
                        if typ:
                            frame = incr_frame(frame, net)  # terminated net is packed in frame, initialized in image_to_blobs
                        else:
                            frame = incr_dframe(frame, net)  # for dframe and dnet

                blob = _blob; fork_= _fork_; _root_= __root_, net = _net  # replace lower-layer vars in while len(fork_) == 0

            _P_[_P_index] = net, blob, fork_, _root_  # net, blob, fork_ -> _P_, addressed by P root_, then by _P _root_

    # no overlap between P and next _P, (P, root_) is packed into next _P_:

    P = s, ix, x, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_
    P_.append((P, root_))  # _P_ = P_ for next-line scan_P_()

    return P_, _P_, buff_, _P_index, frame  # _P_ and buff_ exclude _Ps displaced into _fork_s per blob

''' final blob select by Core div_comp to alt typ, dir, txd Cores -> projected combined redundancy rate?
    redun eval hierarchy: x alt typ ) dir ) txd?
    
    alt_dir_P: combined for orientation eval per blob: ver_P_ ( lat_P_?   
    redun eval per term blob | network, for extended comp or select decoding?
'''

def incr_blob(blob, _P):  # continued or initialized blob is incremented by attached _P, replace by zip?

    s, _ix, _x, L2, I2, D2, Dy2, V2, Vy2, Olp1, Olp2, Olp3, yOlp, Py_, Dx = blob
    (s, ix, lx, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_), yolp = _P  # s is re-assigned, ix and lx from scan_P_

    x = lx - len(t2_) / 2  # median x, becomes _x in blob, replaces lx?
    dx = x - _x  # conditional full comp(x) and comp(S): internal vars are secondary?
    Dx += dx  # for blob normalization and orientation eval, | += |dx| for curved max_L norm, orient?

    L2 += len(t2_)  # t2_ in P buffered in Py_
    I2 += I
    D2 += D; Dy2 += Dy
    V2 += V; Vy2 += Vy
    Olp1 += olp1  # alt-typ overlap per blob
    Olp2 += olp2  # alt-dir overlap per blob
    Olp3 += olp3  # alt-txd overlap per blob
    yOlp += yolp  # vertical contiguity between Ps in Py, for what?

    Py_.append((s, ix, lx, dx, I, D, Dy, V, Vy, olp1, olp2, olp3, yolp, t2_))  # dx to normalize P before comp_P?
    blob = s, ix, x, L2, I2, D2, Dy2, V2, Vy2, Olp1, Olp2, Olp3, yOlp, Py_, Dx  # separate S_par tuple?

    return blob

def incr_network(network, blob, _root_):  # continued or initialized network is incremented by attached blob and _root_

    s, ixn, xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn, Olp1n, Olp2n, Olp3n, yOlpn, blob_ = network  # or S_par tuple?
    s, ix, x, Dx, L2, I2, D2, Dy2, V2, Vy2, Olp1, Olp2, Olp3, yOlp, Py_ = blob  # s is re-assigned

    Dxn += Dx  # for net normalization, orient eval, += |Dx| for curved max_L?
    Ln += L2
    In += I2
    Dn += D2; Dyn += Dy2
    Vn += V2; Vyn += Vy2
    Olp1n += Olp1  # alt-typ overlap per net
    Olp2n += Olp2  # alt-dir overlap per net
    Olp3n += Olp3  # alt-txd overlap per net
    yOlpn += yOlp  # vertical contiguity, for comp_P eval?

    blob_.append((ix, x, Dx, L2, I2, D2, Dy2, V2, Vy2, Olp1, Olp2, Olp3, yOlp, Py_, _root_))  # Dx to normalize blob before comp_P
    network = s, ixn, xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn, Olp1n, Olp2n, Olp3n, yOlpn, blob_  # separate S_par tuple?

    return network

def incr_dframe(frame, net):

    sf, ixf, xf, Dxf, Lf, If, Df, Dyf, Vf, Vyf, yOlpf, net_ = frame  # s of core, summed regardless of sign, no typ olp: complete overlap?
    s, ixn, xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn, Olp1n, Olp2n, Olp3n, yOlpn, blob_ = net

    Dxf += Dxn  # for frame normalization, orient eval, += |Dxn| for curved max_L?
    Lf += Ln
    yOlpf += yOlpn  # for average vertical contiguity, relative to ave Ln?

    If += In  # to compute averages, for dframe only: redundant for same-scope alt_frames?
    Df += Dn
    Dyf += Dyn
    Vf += Vn
    Vyf += Vyn

    net_.append((ixn, xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn, Olp1n, Olp2n, Olp3n, yOlpn, blob_))  # Dxn to normalize net before comp_P
    frame = s, Dxf, Lf, If, Df, Dyf, Vf, Vyf, yOlpf, net_

    return frame

def incr_frame(frame, net):

    sf, Dxf, Lf, yOlpf, net_ = frame  # s of core, summed regardless of sign, no typ olp: complete?
    s, ixn, xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn, Olp1n, Olp2n, Olp3n, yOlpn, blob_ = net

    Dxf += Dxn  # for frame normalization, orient eval, += |Dxn| for curved max_L?
    Lf += Ln
    yOlpf += yOlpn  # for average vertical contiguity, relative to ave Ln?

    net_.append((ixn, xn, Dxn, Ln, In, Dn, Dyn, Vn, Vyn, Olp1n, Olp2n, Olp3n, yOlpn, blob_))  # Dxn to normalize net before comp_P
    frame = s, Dxf, Lf, yOlpf, net_

    return frame


def image_to_blobs(f):  # postfix '_' distinguishes array vs. element, prefix '_' distinguishes higher-line vs. lower-line variable

    _dP_, _vP_, _dyP_, _vyP_ = [],[],[],[]  # higher-line same- d-, v-, dy-, vy- sign 1D patterns, with refs to blob networks

    dframe = 0,0,0,0,0,0,0,0,0,[]  # s, Dxf, Lf, If, Df, Dyf, Vf, Vyf, yOlpf, net_
    vframe = 0,0,0,0,[]   # s, Dxf, Lf, yOlpf, net_:
    dyframe = 0,0,0,0,[]  # no If, Df, Dyf, Vf, Vyf: redundant to dframe
    vyframe = 0,0,0,0,[]  # constituent net rep may select sum | [nets], same for blob?

    global y; y = 0  # vertical coordinate of current input line

    t2_ = deque(maxlen=rng)  # vertical buffer of incomplete quadrant tuples, for fuzzy ycomp
    t2__ = []  # vertical buffer + horizontal line: 2D array of 2D tuples, deque for speed?
    p_ = f[0, :]  # first line of pixels
    t_ = horizontal_comp(p_)  # after part_comp (pop, no t_.append) while x < rng?

    for t in t_:
        p, d, m = t
        t2 = p, d, 0, m, 0  # fdy and fmy initialized at 0; alt named quad?
        t2_.append(t2)  # only one tuple per first-line t2_
        t2__.append(t2_)  # in same order as t_

    # part_ycomp (pop, no form_P) while y < rng?

    for y in range(1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?

        p_ = f[y, :]  # vertical coordinate y is index of new line p_
        t_ = horizontal_comp(p_)  # lateral pixel comparison

        _dPi, _vPi, _dyPi, _vyPi = 0, 0, 0, 0  # _P indices, initialized per line
        # vertical pixel comparison:

        t2__, _dP_, _vP_, _dyP_, _vyP_, _dPi, _vPi, _dyPi, _vyPi, dframe, vframe, dyframe, vyframe = \
        vertical_comp(t_, t2__, _dP_, _vP_, _dyP_, _vyP_, _dPi, _vPi, _dyPi, _vyPi, dframe, vframe, dyframe, vyframe)

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
