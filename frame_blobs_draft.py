import cv2
import argparse
from time import time
from collections import deque

''' Core algorithm of levels 1 + 2, modified to segment image into blobs. It performs several steps of encoding, 
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
    Prefix '_' distinguishes higher-line variable or pattern from same-type lower-line variable or pattern 
    
    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match.
    Vertical and horizontal derivatives form separate blobs, suppressing overlapping orthogonal representations
    They can be summed to estimate diagonal or hypot derivatives, for blob orientation to maximize primary derivatives
    Orientation increases primary dimension of blob to maximize match, and decreases secondary dimension to maximize difference.

    Not implemented here, alternative segmentation by quadrant gradient (less predictive due to directional info loss): 
    Pixels are discrete samples of continuous image, so rightward and downward derivatives per pixel are 
    equally representative samples of 0-90 degree quadrant gradient: minimal unique unit of 2D gradient. 
    Thus, quadrant gradient is estimated as the average of these two orthogonally diverging derivatives.
    Resulting blob would be contiguous area of same-sign quadrant gradient, of difference for dblob or match deviation for vblob.

    None of this is tested, except as analogue functions in line_POC() 
    '''
    # postfix '_' denotes array name, vs. same-name elements of that array


def horizontal_comp(p_):  # comparison between rng consecutive pixels within horizontal line

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

        if len(it_) == rng:  # or while x < rng: icomp(){ p = pop(p_).., no t_.append?
            t_.append((pri_p, fd, fm))  # completed tuple is transferred from it_ to t_

        it_.append((p, 0, 0))  # new prior tuple, fd and fm are initialized at 0

    t_ += it_  # tuples of last rng remain incomplete
    return t_


def vertical_comp(t_, t2__, _vP_, _dP_, _vyP_, _dyP_, vnet, dnet, vynet, dynet, vframe, dframe, vyframe, dyframe):

    # vertical comparison between pixels of consecutive lines, forming quadrant t2 per pixel
    # prefix '_' denotes higher-line variable or pattern

    vP = [0,0,0,0,0,0,0,[]]  # value pattern = pri_s, I, D, Dy, M, My, Olp, t2_;  all initialized per line
    dP = [0,0,0,0,0,0,0,[]]  # difference pattern = pri_s, I, D, Dy, M, My, Olp, t2_
    vyP = [0,0,0,0,0,0,0,[]]  # vertical value pattern = pri_s, I, D, Dy, M, My, Olp, t2_
    dyP = [0,0,0,0,0,0,0,[]]  # vertical difference pattern = pri_s, I, D, Dy, M, My, Olp, t2_

    o_v_d  = (0,0,0)  # alt. type overlap between vP and dP = (len, V, D): summed over overlap, within line?
    o_vy_dy = (0,0,0)  # alt. type overlap between vyP and dyP = (len, Vy, Dy)
    o_v_vy = (0,0,0)  # alt. direction overlap between vP and vyP = (len, V, Vy)
    o_d_dy = (0,0,0)  # alt. direction overlap between dP and dyP = (len, D, Dy)
    o_v_dy = (0,0,0)  # alt. type and direction overlap between vP and dyP = (len, V, Dy)
    o_d_vy = (0,0,0)  # alt. type and direction overlap between dP and vyP = (len, D, Vy)

    vP_, dP_, vyP_, dyP_ = [],[],[],[]
    x = 0; new_t2__ = []  # t2_ buffer: 2D array

    for t, t2_ in zip(t_, t2__):  # compares vertically consecutive pixels within rng, right-to-left?
        p, d, m = t
        index = 0
        x += 1

        for t2 in t2_:
            pri_p, _d, fdy, _m, fmy = t2

            dy = p - pri_p  # vertical difference between pixels
            my = min(p, pri_p)  # vertical match between pixels

            fdy += dy  # fuzzy dy: sum of dy between p and all prior ps within t2_
            fmy += my  # fuzzy my: sum of my between p and all prior ps within t2_

            t2_[index] = pri_p, _d, fdy, _m, fmy
            index += 1

        if len(t2_) == rng:  # or while y < rng: i_ycomp(): t2_ = pop(t2__), t = pop(t_)., no form_P?

            v = _m - ave
            vy = fmy - ave
            t2 = pri_p, _d, fdy, v, vy  # completed quadrants are moved from t2_ to form_P:

            # form 1D patterns vP, vPy, dP, dPy: horizontal spans of same-sign derivative, with associated vars, 3 olp per P type:

            vP, dP, vyP, dyP, o_v_d, o_v_vy, o_v_dy, vP_, _vP_, vnet, vframe =       form_P(0, t2, x, vP, dP, vyP, dyP, o_v_d, o_v_vy, o_v_dy, vP_, _vP_, vnet, vframe)
            dP, vP, dyP, vyP, o_v_d, o_d_dy, o_d_vy, dP_, _dP_, dnet, dframe =       form_P(1, t2, x, dP, vP, dyP, vyP, o_v_d, o_d_dy, o_d_vy, dP_, _dP_, dnet, dframe)
            vyP, dyP, vP, dP, o_vy_dy, o_v_vy, o_d_vy, vyP_, _vyP_, vynet, vyframe = form_P(2, t2, x, vyP, dyP, vP, dP, o_vy_dy, o_v_vy, o_d_vy, vyP_, _vyP_, vynet, vyframe)
            dyP, vyP, dP, vP, o_vy_dy, o_d_dy, o_v_dy, dyP_, _dyP_, dynet, dyframe = form_P(3, t2, x, dyP, vyP, dP, vP, o_vy_dy, o_d_dy, o_v_dy, dyP_, _dyP_, dynet, dyframe)

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

        vP_, _vP_, vnet, vframe = scan_P_(0, x, vP, vP_, _vP_, vnet, vframe)  # returns empty _vP_
        dP_, _dP_, dnet, dframe = scan_P_(1, x, dP, dP_, _dP_, dnet, dframe)  # returns empty _dP_
        vyP_, _vyP_, vynet, vyframe = scan_P_(0, x, vyP, vyP_, _vyP_, vynet, vyframe)  # returns empty _vyP_
        dyP_, _dyP_, dynet, dyframe = scan_P_(1, x, dyP, dyP_, _dyP_, dynet, dyframe)  # returns empty _dyP_

    return new_t2__, _vP_, _dP_, _vyP_, _dyP_, vnet, dnet, vynet, dynet, vframe, dframe, vyframe, dyframe  # extended in scan_P_


def form_P(typ, t2, x, P, alt_typ_P, alt_dir_P, alt_txd_P, typ_olp, dir_olp, txd_olp, P_, _P_, network, frame):

    # forms 1D dP or vP, then scan_P_ adds forks in _P fork_s and accumulates blob_

    p, d, dy, v, vy = t2  # 2D tuple of quadrant variables per pixel
    pri_s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_ = P  # initial pri_ vars = 0, or skip form?

    # 1: len_typ_olp, core, alt_core; 2: len_dir_olp, core, alt_core; 3: len_txd_olp, core, alt_core
    # core: derivative the sign of which defines current type of pattern:

    if typ == 0: core = v
    elif typ == 1: core = d
    elif typ == 2: core = vy
    else: core = dy

    s = 1 if core > 0 else 0  # core = 0 is negative: no selection?
    if s != pri_s and x > rng + 2:  # P is terminated

        if typ == 0 or typ == 3:  # ave V / I, to project V of alt_Ps
            alt_c *= ave_k; alt_oG = alt_oG.astype(int)
        else:  oG *= ave_k; oG = oG.astype(int)  # same for h_der and h_comp eval? # relative value adjustment

        if oG > alt_oG:  # comp between overlapping vG and dG; olp: len, core, alt_core
            Olp += olp  # olp is assigned to the weaker of P | alt_P, == -> P: local access
        else:
            alt_P[7] += olp

        P = pri_s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_  # no ave * alt_rdn / e_: adj < cost?
        P_, _P_, blob_, net_ = scan_P_(typ, x, P, P_, _P_, network, frame)  # scans higher-line _Ps

        I, D, Dy, M, My, olp1, olp2, olp3, t2_ = 0, 0, 0, 0, 0, 0, 0, 0, []  # P initialization
        typ_olp = 0, 0, 0; dir_olp = 0, 0, 0; txd_olp = 0, 0, 0  # olp initialization

    # continued or initialized vars are accumulated (use zip S_vars?):

    olp += 1; oG += g; alt_oG += alt_g  # # len of overlap to stronger alt-type P, accumulated until P or alt P terminates for eval to assign olp to alt_rdn of vP or dP

    I += p    # inputs and derivatives are summed as P parameters:
    D += d    # lateral D
    Dy += dy  # vertical D
    V += v    # lateral V
    Vy += vy  # vertical V

    t2_.append((p, d, dy, v, vy))  # vs. p, g, alt_g in vP and g in dP:
    # full quadrants are buffered for oriented rescan, as well as incremental range | derivation comp

    P = [s, I, D, Dy, V, Vy, olp1, olp2, olp3, t2_]

    return P, alt_typ_P, alt_dir_P, alt_txd_P, typ_olp, dir_olp, txd_olp, P_, _P_, network, frame  # accumulated in ycomp


def scan_P_(typ, x, P, P_, _P_, network, frame):  # P scans shared-x_coord _Ps in _P_, forms overlaps

    buff_ = []  # for displaced _Ps buffered for scan_P_(next P)
    root_ = []  # for _Ps connected to current P
    s, I, D, Dy, M, My, G, Olp, t2_ = P
    ix = x - len(t2_)  # initial x of P
    _ix = 0  # ix of _P, displaced from _P_ by last scan_P_?

    while x >= _ix:  # while horizontal overlap between P and _P_:

        t2_x = x  # lateral coordinate of loaded quadrant
        olp = 0  # vertical overlap between P and _P: [] of summed vars?  different from olp between P types in comp_P
        _P, fork_, _root_ = _P_.popleft()  # forks in y-1, _P in y-2, root blobs in y-3

        if s == _P[0]:  # if s == _s (der sign match):
            while t2_x > _P[1]:  # t2_x > _ix:
                for t2 in t2_:  # all vars are summed within overlap between P and _P for blob eval:

                    if typ: olp += t2[0]  # t2 = p, d, dy, m, my
                    else:   olp += t2[1]
                    t2_x += 1

            root_.append((olp, _P))  # _Ps connected to P, for terminated segment transfer to network
            fork_.append((olp, P))  # Ps connected to _P, for terminated _P transfer to segment

        if _P[2] > ix:  # if _x > ix: _P is buffered for scan_P_(next P)
            buff_.append(_P)
        else: # no overlap between _P and next P, _P is packed into blob, higher blobs are tested for termination:

            while fork_ == 0 and _root_:  # recursion of incr_blob for higher layers of network?

                for blob, fork_, __root_ in _root_:  # root_' blobs termination test:
                    if fork_ == 1:
                        network = incr_net(blob, network)  # or net and frame are accessed from blob?
                    else:
                        frame = incr_frame(network, frame)
                        # if _blob' fork_ == 0 and _root_ == 0: net term and sum per frame

            if _root_ == 1:  # after while()
                blob = incr_blob((olp, _P), _root_[0])

            else:  # blob but not network is terminated:
                blob = (_P, 0, olp, [_P]), fork_, _root_  # blob initialization, attached to P as root?
                # _P = (_s, _x, _ix, len(_t2_), _I, _D, _Dy, _M, _My, _Olp), Dx = 0, Py_ = [_P]

            root_ = [(blob, [], _root_)]  # blob attached to P' root, empty fork_?

    # no overlap between P and next _P, at next-line input: blob +=_P for root_ of P if fork_ != 0

    P = s, ix, x, I, D, Dy, M, My, G, Olp, t2_  # P becomes _P, oG is per new P in fork_?

    P_.append((P, [], root_))  # blob assign, forks init, _P_ = P_ for next-line scan_P_()
    buff_ += _P_  # excluding displaced _Ps

    return P_, buff_, network, frame  # _P_ = buff_ for scan_P_(next P)

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

def incr_frame(network, frame):
    return frame


def image_to_blobs(f):  # postfix '_' distinguishes array vs. element, prefix '_' distinguishes higher-line vs. lower-line variable

    _vP_, _dP_, _vyP_, _dyP_ = [],[],[],[]  # same- v-, d-, vy-, dy- sign 1D patterns on a higher line
    # contain refs to v, d, vy, dy- blob segments, vertical concat -> blob network:

    vnet, dnet, vynet, dynet = [],[],[],[]  # same-sign blob segment networks, vertical concat -> corr frame in frame-to-blobs():
    vframe, dframe, vyframe, dyframe = [],[],[],[]  # actually tuples?

    global y; y = 0  # vertical coordinate of current input line

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
        t_ = horizontal_comp(p_)  # lateral pixel comparison, then vertical pixel comparison:
        t2__, _vP_, _dP_, _vyP_, _dyP_, vnet, dnet, vynet, dynet, vframe, dframe, vyframe, dyframe = \
        vertical_comp(t_, t2__, _vP_, _dP_, _vyP_, _dyP_, vnet, dnet, vynet, dynet, vframe, dframe, vyframe, dyframe)

    return vframe, dframe, vyframe, dyframe  # frame of 2D patterns is outputted to level 2

# pattern filters: eventually a higher-level feedback, initialized here as constants:

rng = 1  # number of leftward and upward pixels compared to input pixel
ave = 127 * rng  # filter, ultimately set by separate feedback, then ave *= rng?
ave_k = 0.25  # average V / I initialization

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
