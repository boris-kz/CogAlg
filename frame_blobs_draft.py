import cv2
import argparse
from time import time
from collections import deque

''' core algorithm of levels 1 + 2, modified to segment image into blobs. It performs several steps of encoding, 
    incremental per scan line defined by vertical coordinate y, computed relative to y of current input line:

    y:    1st-level encoding by comp(p_): lateral pixel comparison -> tuple t ) array t_
    y- 1: 2nd-level encoding by ycomp(t_): vertical pixel comp -> quadrant t2 ) array t2_ 
    y- 1+ rng: 3Lev encoding by form_P(t2) -> 1D pattern P ) P_  
    y- 2+ rng: 4Lev encoding by scan_P_(P, _P) -> fork_, root_: vertical connections between Ps of adjacent lines 
    y- 3+ rng+ blob depth: 5Lev encoding by incr_blob(_P, blob) -> blob: merge connected Ps into blob segments
    y- 4+ rng+ netw depth: 6Lev encoding by incr_net(_blob, net) -> net: merge connected segments into network

    All 2D functions (ycomp, scan_P_, etc.) input two lines: higher and lower, convert elements of lower line 
    into elements of new higher line, and displace elements of old higher line into some higher function.
    Higher-line elements include additional variables, derived while they were lower-line elements.
    frame() is layered: partial lower functions can work without higher functions.
    None of this is tested, except as analogue functions in line_POC()  

    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match.
    Vertical and horizontal derivatives form separate blobs, suppressing overlapping orthogonal representations
    They are summed to estimate diagonal or hypot derivatives, for blob orientation to maximize corresponding derivatives
    Orientation increases primary dimension of blob to maximize match, and decreases secondary dimension to maximize difference.

    Less predictive segmentation by quadrant gradient: 
    Pixels are discrete samples of continuous image, so rightward and downward derivatives per pixel are 
    equally representative samples of 0-90 degree quadrant gradient: minimal unique unit of 2D gradient. 
    Thus, quadrant gradient is estimated as the average of these two orthogonally diverging derivatives.
    Blob is contiguous area of same-sign quadrant gradient, of difference for dblob or match deviation for vblob.

    postfix '_' denotes array name, vs. same-name elements of that array 
    prefix '_' denotes higher-line variable or pattern '''


def comp(p_):  # comparison of consecutive pixels within a line forms tuples: pixel, match, difference

    t_ = []  # complete fuzzy tuples: summation range = rng
    it_ = deque(maxlen=rng)  # incomplete fuzzy tuples: summation range < rng

    for p in p_:
        index = 0

        for it in it_:  # incomplete tuples, with summation range from 0 to rng
            pri_p, fd, fm = it

            d = p - pri_p  # difference between pixels
            m = min(p, pri_p)  # match between pixels

            fd += d  # fuzzy d: sum of ds between p and all prior ps within it_
            fm += m  # fuzzy m: sum of ms between p and all prior ps within it_

            it_[index] = pri_p, fd, fm
            index += 1

        if len(it_) == rng:  # or while x < rng: icomp(){ p = pop(p_).., no t_.append?
            t_.append((pri_p, fd, fm))  # completed tuple is transferred from it_ to t_

        it_.appendleft((p, 0, 0))  # new prior tuple, fd and fm are initialized at 0

    t_ += it_  # last rng of tuples remain incomplete
    return t_


def ycomp(t_, t2__, _vP_, _dP_, _vPy_, _dPy_, vblob_, dblob_, vyblob_, dyblob_, vnet_, dnet_, vynet_, dynet_):
    # vertical comparison between pixels of consecutive lines, forming quadrant t2 per pixel

    vP = [0,0,0,0,0,0,0,[]]  # value pattern = pri_s, I, D, Dy, M, My, Olp, t2_
    dP = [0,0,0,0,0,0,0,[]]  # difference pattern = pri_s, I, D, Dy, M, My, Olp, t2_
    vPy = [0,0,0,0,0,0,0,[]]  # vertical value pattern = pri_s, I, D, Dy, M, My, Olp, t2_
    dPy = [0,0,0,0,0,0,0,[]]  # vertical difference pattern = pri_s, I, D, Dy, M, My, Olp, t2_

    olp = 0  # len of overlap between vP and dP: lateral;  also t2 vars summed over olp? 3 olps per P type:
    yolp = 0  # len of overlap between vPy and dPy: lateral
    olpy = 0  # len of overlap between vP and vPy: vertical
    dolpy = 0  # len of overlap between dP and dPy: vertical
    xolp = 0  # len of overlap between vP and dPy: diagonal
    xdolp = 0  # len of overlap between dP and vPy: diagonal

    vP_, dP_, vPy_, dPy_ = [],[],[],[]
    x = 0; new_t2__ =[]  # t2_ buffer: 2D array

    for t, t2_ in zip(t_, t2__):  # compares vertically consecutive pixels, forms quadrant gradients
        p, d, m = t
        index = 0
        x += 1

        for t2 in t2_:
            pri_p, _d, fdy, _m, fmy = t2

            dy = p - pri_p  # vertical difference between pixels
            my = min(p, pri_p)  # vertical match between pixels

            fdy += dy  # fuzzy dy: sum of dys between p and all prior ps within t2_
            fmy += my  # fuzzy my: sum of mys between p and all prior ps within t2_

            t2_[index] = pri_p, _d, fdy, _m, fmy
            index += 1

        if len(t2_) == rng:  # or while y < rng: i_ycomp(): t2_ = pop(t2__), t = pop(t_)., no form_P?

            t2 = pri_p, _d, fdy, _m, fmy  # completed quadrants are moved from t2_ to form_P:
            # form 1D patterns vP, vPy, dP, dPy: horizontal spans of same-sign derivative, with associated vars:

            olp, vP, dP, vPy, dPy, vP_, _vP_, vblob_, vnet_ = form_P(1, t2, olp, vP, dP, vPy, dPy, vP_, _vP_, vblob_, vnet_, x)
            olp, dP, vP, dP_, _dP_, dblob_, dnet_ = form_P(0, t2, olp, dP, vP, dP_, _dP_, dblob_, dnet_, x)  # to be extended
            olp, vPy, dPy, vPy_, _vPy_, vyblob_, vynet_ = form_P(1, t2, olp, vPy, dPy, vPy_, _vPy_, vyblob_, vynet_, x)  # to be extended
            olp, dPy, vPy, dPy_, _dPy_, dyblob_, dynet_ = form_P(0, t2, olp, dPy, vPy, dPy_, _dPy_, dyblob_, dynet_, x)  # to be extended

        t2_.appendleft((p, d, 0, m, 0))  # initial fdy and fmy = 0, new t2 replaces completed t2 in t2_
        new_t2__.append(t2_)

    # line ends, patterns are terminated after inclusion of t2 with incomplete lateral fd and fm:

    ''' if olp:  # if vP x dP overlap len > 0, incomplete vg - ave / (rng / X-x)?
        olp assign to vP | vPy | dP | dPy: copied from form_P

        odG *= ave_k; odG = odG.astype(int)  # ave_k = V / I, to project V of odG

        if ovG > odG:  # comp of olp vG and olp dG, == goes to vP: secondary pattern?
            dP[7] += olp  # overlap of lesser-oG vP or dP, or P = P, Olp?
        else:
            vP[7] += olp  # to form rel_rdn = alt_rdn / len(e_)
    '''

    if y + 1 > rng:  # starting with the first line of complete t2s, not finished:

        vP_, _vP_, vblob_, vnet_ = scan_P_(0, vP, vP_, _vP_, vblob_, vnet_, x)  # returns empty _vP_
        dP_, _dP_, dblob_, dnet_ = scan_P_(1, dP, dP_, _dP_, dblob_, dnet_, x)  # returns empty _dP_
        vP_, _vP_, vblob_, vnet_ = scan_P_(0, vP, vP_, _vP_, vblob_, vnet_, x)  # returns empty _vPy_
        dP_, _dP_, dblob_, dnet_ = scan_P_(1, dP, dP_, _dP_, dblob_, dnet_, x)  # returns empty _dPy_

    return new_t2__, _vP_, _dP_, vblob_, dblob_, vnet_, dnet_  # extended in scan_P_


def form_P(typ, t2, olp, P, alt_P, P_, _P_, blob_, network, frame_rep, x):

    # forms 1D dP or vP, then scan_P_ adds forks in _P fork_s and accumulates blob_

    p, d, dy, m, my = t2  # 2D tuple of quadrant variables per pixel
    pri_s, I, D, Dy, M, My, Olp, t2_ = P  # initial pri_ vars = 0, or skip form?

    s = 1 if g > 0 else 0  # g = 0 is negative: no selection?
    if s != pri_s and x > rng + 2:  # P is terminated

        if typ: alt_oG *= ave_k; alt_oG = alt_oG.astype(int)  # ave V / I, to project V of odG
        else:   oG *= ave_k; oG = oG.astype(int)  # same for h_der and h_comp eval?

        if oG > alt_oG:  # comp between overlapping vG and dG
            Olp += olp  # olp is assigned to the weaker of P | alt_P, == -> P: local access
        else:
            alt_P[7] += olp

        P = pri_s, I, D, Dy, M, My, Olp, t2_  # no ave * alt_rdn / e_: adj < cost?
        P_, _P_, blob_, net_ = scan_P_(typ, P, P_, _P_, blob_, network, frame_rep, x)  # scans higher-line _Ps

        I, D, Dy, M, My, Olp, t2_ = 0, 0, 0, 0, 0, 0, []  # P initialization
        olp, oG, alt_oG = 0, 0, 0  # olp initialization

    # continued or initialized vars are accumulated (use zip S_vars?):

    olp += 1  # len of overlap to stronger alt-type P, accumulated until P or alt P terminates
    oG += g; alt_oG += alt_g  # for eval to assign olp to alt_rdn of vP or dP

    I += p    # inputs and derivatives are summed as P parameters:
    D += d    # lateral D
    Dy += dy  # vertical D
    M += m    # lateral M
    My += my  # vertical M

    t2_.append((p, d, dy, m, my))  # vs. p, g, alt_g in vP and g in dP:
    # full quadrants are buffered for oriented rescan, as well as incremental range | derivation comp

    P = [s, I, D, Dy, M, My, Olp, t2_]

    return olp, P, alt_P, P_, _P_, blob_, network, frame_rep  # accumulated in ycomp


def scan_P_(typ, P, P_, _P_, blob_, network, frame_rep, x):  # P scans shared-x_coord _Ps in _P_, forms overlaps

    buff_ = []  # for displaced _Ps buffered for scan_P_(next P)
    root_ = []  # for _Ps connected to current P
    s, I, D, Dy, M, My, G, Olp, t2_ = P
    ix = x - len(t2_)  # initial x of P
    _ix = 0  # ix of _P, displaced from _P_ by last scan_P_?

    while x >= _ix:  # while horizontal overlap between P and _P_:

        t2_x = x  # lateral coordinate of loaded quadrant
        olp = 0  # vertical overlap between P and _P: [] of summed vars
        _P, fork_, _root_ = _P_.popleft()  # forks in y-1, _P in y-2, root blobs in y-3

        if s == _P[0]:  # if s == _s: v or d sign match -> fork_.append

            while t2_x > _P[1]:  # t2_x > _ix:
                for t2 in t2_:  # all vars are summed within overlap between P and _P for blob eval:

                    if typ: olp += t2[0]  # quad t2 = p, d, dy, m, my
                    else:   olp += t2[1]
                    t2_x += 1

            root_.append((olp, _P))  # _Ps connected to P, for terminated segment transfer to network
            fork_.append((olp, P))  # Ps connected to _P, for terminated _P transfer to segment

        if _P[2] > ix:  # if _x > ix: _P is buffered for scan_P_(next P)
            buff_.append(_P)
        else: # no overlap between _P and next P, _P is packed into blob:

            if _root_ == 1:  # or start from while _root_ and fork_ == 0: load blob, _root_, fork_ in _root_?

                blob = incr_blob((olp, _P), _root_[0])
            else:  # blob is terminated, new blob is initialized:

                while _root_:  # recursion of incr_blob for higher layers of network?
                    for _blob in _root_: # root_' blobs termination:
                        if _blob[1] == 1: # if _blob' fork_ == 1
                            network = incr_net(_blob, network)
                        else:
                            frame_rep = incr_frame(network, frame_rep)
                            # if _blob' fork_ == 0 and _root_ == 0: net term and sum per frame

                blob = (_P, 0, olp, [_P]), fork_, _root_  # blob initialization,
                # _P = (_s, _x, _ix, len(_t2_), _I, _D, _Dy, _M, _My, _Olp), Dx = 0, Py_ = [_P]

    # no overlap between P and next _P, at next-line input: blob +=_P for root_ of P if fork_ != 0

    P = s, ix, x, I, D, Dy, M, My, G, Olp, t2_  # P becomes _P, oG is per new P in fork_?

    P_.append((P, [], root_))  # blob assign, forks init, _P_ = P_ for next-line scan_P_()
    buff_ += _P_  # excluding displaced _Ps

    return P_, buff_, blob_, network  # _P_ = buff_ for scan_P_(next P)

'''
    for _P, blob, fork_ in root_:  # final fork assignment and blob increment per _P

        blob = incr_blob(_P, blob)  # default per root, blob is modified in root _P?

        if fork_ != 1 or root_ != 1:  # blob split | merge, also if y == Y - 1 in frame()?
            blob_.append((blob, fork_))  # terminated blob_ is input line y - 3+ | record layer 5+

    if root_ == 1 and root_[0][3] == 1:  # blob assign if final P' root_==1 and root' fork_==1
        blob = root_[0][1]  # root' fork' blob
    else:
        blob = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [])  # init s, L2, I2, D2, Dy2, M2, My2, G2, OG, Olp2, Py_
'''

def incr_blob(_P, blob):  # continued or initialized blob is incremented by attached _P, replace by zip?

    s, _x, _ix, _lx, Dx, L2, I2, D2, Dy2, M2, My2, G2, OG, Olp, Py_ = blob  # or S_par tuple?
    oG, (s, ix, lx, I, D, Dy, M, My, G, olp, t2_) = _P  # s is re-assigned, ix and lx from scan_P_

    x = lx - len(t2_) / 2  # median x, becomes _x in blob, replaces ix and lx?
    dx = x - _x  # full comp(x) and comp(S) are conditional, internal vars are secondary
    Dx += dx  # for blob norm, orient eval, by OG vs. Mx += mx, += |dx| for curved max_L

    L2 += len(t2_)  # t2_ in P buffered in Py_
    I2 += I
    D2 += D
    Dy2 += Dy
    M2 += M
    My2 += My
    G2 += G  # blob value
    OG += oG  # vertical contiguity, for comp_P eval?
    Olp += olp  # adds to blob orient and comp_P cost?

    Py_.append((s, ix, lx, I, D, Dy, M, My, G, oG, olp, t2_, Dx))  # Dx to normalize P before comp_P
    blob = s, x, ix, lx, Dx, (L2, I2, D2, Dy2, M2, My2, G2, OG, Olp), Py_  # separate S_par tuple?

    return blob

def incr_net(blob, network):  # continued or initialized network is incremented by attached fork_
    # stub only, to be replaced:

    s, _x, _ix, _lx, Dx_net, L_net, I_net, D_net, Dy_net, M_net, My_net, yOlp_net, Olp_net, blob_ = network  # or S_par tuple?
    s, x, ix, lx, Dx, L2, I2, D2, Dy2, M2, My2, G2, yOlp, Olp, Py_ = blob  # s is re-assigned, ix and lx from scan_P_

    x = lx - L_net / 2  # median x, becomes _x in blob, replaces ix and lx?
    dx = x - _x  # full comp(x) and comp(S) are conditional, internal vars are secondary
    Dx += dx  # for blob norm, orient eval, by OG vs. Mx += mx, += |dx| for curved max_L

    L_net += L2   # t2_ in P buffered in Py_
    I_net += I2
    D_net += D2
    Dy_net += Dy2
    M_net += M2
    My_net += My2
    yOlp_net += yOlp  # vertical contiguity, for comp_P eval?
    Olp_net += Olp  # adds to blob orient and comp_P cost?

    blob_.append((ix, lx, Dx, L2, I2, D2, Dy2, M2, My2, G2, yOlp, Olp, Py_))  # Dx to normalize P before comp_P
    network = s, x, ix, lx, Dx_net, L_net, I_net, D_net, Dy_net, M_net, My_net, yOlp_net, Olp_net, blob_  # separate S_par tuple?

    return network

def incr_frame(network, frame_rep):
    return frame_rep


def frame(f):  # postfix '_' denotes array vs. element, prefix '_' denotes higher-line variable

    _vP_, _dP_ = [],[]  # same- v- and d- sign 1D patterns on a higher line
    vblob_, dblob_ = [],[]  # v- and d- sign blob segments, vertical concat -> blob network:
    vnet_, dnet_ = [],[]  # v- and d- sign blob segment networks, vertical concat -> frame in frame()
    frame_ = []

    global Y; global X; Y, X = f.shape  # Y: frame height, X: frame width
    global y; y = 0

    t2_ = deque(maxlen=rng)  # vertical buffer of incomplete quadrant tuples, for fuzzy ycomp
    t2__ = []  # vertical buffer + horizontal line: 2D array of 2D tuples, deque for speed?
    p_ = f[0, :]  # first line of pixels
    t_ = comp(p_)  # after part_comp (pop, no t_.append) while x < rng?

    for t in t_:
        p, d, m = t
        t2 = p, d, 0, m, 0  # fdy and fmy initialized at 0; alt named quad?
        t2_.append(t2)  # only one tuple per first-line t2_
        t2__.append(t2_)  # in same order as t_

    # part_ycomp (pop, no form_P) while y < rng?

    for y in range(1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?

        p_ = f[y, :]  # vertical coordinate y is index of new line p_
        t_ = comp(p_)  # lateral pixel comparison
        t2__, _vP_, _dP_, vblob_, dblob_, vnet_, dnet_ = ycomp(t_, t2__, _vP_, _dP_, vblob_, dblob_, vnet_, dnet_) # vertical pixel comp

        frame_.append((vnet_, dnet_))  # line of blobs is added to frame of blobs

    return frame_  # frame of 2D patterns is outputted to level 2

# pattern filters: eventually a higher-level feedback, initialized here as constants:

rng = 1  # number of leftward and upward pixels compared to input pixel
ave = 127 * rng  # filter, ultimately set by separate feedback, then ave *= rng?
div_a = 127  # not justified
ave_k = 0.25  # average V / I initialization

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/racoon.jpg')
arguments = vars(argument_parser.parse_args())
image = cv2.imread(arguments['image'], 0).astype(int)

start_time = time()
blobs = frame(image)
end_time = time() - start_time
print(end_time)
