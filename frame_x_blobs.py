import cv2
import argparse
import numpy as np
from scipy import misc
from time import time
from collections import deque

'''   
    frame() is core algorithm of levels 1 + 2, modified for 2D: segmentation of image into blobs, then recursive search within blobs,
    frame_blobs() defines initial blobs for each of 4 derivatives, frame_dblobs() is restricted to dblobs for debugging,
    frame_x_blobs defines dblobs only inside negative mblobs, and dyblobs only inside negative myblobs.
      
    Each performs several levels (Le) of encoding, incremental per scan line defined by vertical coordinate y, outlined below.
    value of y per Le line is shown relative to y of current input line, incremented by top-down scan of input image
    for y = rng *2: line y == P_, line y-1 == hP_, line y-2 == seg_, line y-4 == blob_:
    
    1Le, line y:    x_comp(p_): lateral pixel comparison -> tuple of derivatives (ders) array ders_
    2Le, from line y - 1 to line y - rng: y_comp(ders_): vertical pixel comp -> 2D tuple ders2 array ders2_ 
    3Le, line y - rng: form_P(ders2) -> 1D pattern P
    4Le, line y - rng - 1: scan_P_(P, hP) -> hP, roots: down-connections, fork_: up-connections between Ps 
    5Le, line y - rng - 2: form_segment(hP, seg) -> seg: merge vertically-connected _Ps in non-forking blob segments
    6Le, line y - rng - 1 + seg depth: form_blob(seg, blob): merge connected segments in fork_' incomplete blobs  
    
    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match.
    Each vertical and horizontal derivative forms separate blobs, suppressing overlapping orthogonal representations.
    
    They can also be summed to estimate diagonal or hypot derivatives, for blob orientation to maximize primary derivatives.
    Orientation increases primary dimension of blob to maximize match, and decreases secondary dimension to maximize difference.
    Subsequent union of lateral and vertical patterns is by strength only, orthogonal sign is not commeasurable?
    
    Initial pixel comparison is not novel, I design from the scratch to make it organic part of hierarchical algorithm.
    It would be much faster with matrix computation, but this is minor compared to higher-level processing.
    I implement it sequentially for consistency with accumulation into blobs: irregular and very difficult to map to matrices.
    
    All 2D functions (y_comp, scan_P_, form_segment, form_blob) input two lines: higher and lower, 
    convert elements of lower line into elements of new higher line, then displace elements of old higher line into higher function.
    Higher-line elements include additional variables, derived while they were lower-line elements.
    
    prefix '_' denotes higher-line variable or pattern, vs. same-type lower-line variable or pattern,
    postfix '_' denotes array name, vs. same-name elements of that array:
'''

# ************ MISCELLANEOUS FUNCTIONS **********************************************************************************
# Includes:
# -rebuild_blobs()
# -segment_sort_by_height()
# ***********************************************************************************************************************

def rebuild_blobs( frame, print_separate_blobs = 0 ):
    # Rebuilt data of blobs into an image
    blob_image = np.array([[[127] * 4] * X] * Y)

    for index, blob in enumerate(frame[2]):  # Iterate through blobs
        if print_separate_blobs: blob_img = np.array([[[127] * 4] * X] * Y)
        for seg in blob[2]:  # Iterate through segments
            y = seg[7]
            for (P, dx) in reversed(seg[5]):
                x = P[1]
                for i in range(P[2]):
                    blob_image[y, x, : 3] = [255, 255, 255] if P[0] else [0, 0, 0]
                    if print_separate_blobs: blob_img[y, x, : 3] = [255, 255, 255] if P[0] else [0, 0, 0]
                    x += 1
                y -= 1
        if print_separate_blobs:
            min_x, max_x, min_y, max_y = blob[1][:4]
            cv2.rectangle(blob_img, (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), (0, 255, 255), 1)
            cv2.imwrite('./images/raccoon_eye/blob%d.jpg' % (index), blob_img)

    return blob_image
    # ---------- rebuild_blobs() end ------------------------------------------------------------------------------------


def segment_sort_by_height(seg):
    # Used in sort() function at blob termination
    return seg[7]
    # ---------- segment_sort_by_height() end ---------------------------------------------------------------------------


# ************ MISCELLANEOUS FUNCTIONS END ******************************************************************************

# ************ MAIN FUNCTIONS *******************************************************************************************
# Includes:
# -lateral_comp()
# -vertical_comp()
# -form_P()
# -scan_P_()
# -form_segment()
# -form_blob()
# -image_to_blobs()
# ***********************************************************************************************************************
def lateral_comp(pixel_):
    # Comparison over x coordinate, within rng of consecutive pixels on each line

    ders1_ = []  # tuples of complete 1D derivatives: summation range = rng
    rng_ders1_ = deque(maxlen=rng)  # incomplete ders1s, within rng from input pixel: summation range < rng
    rng_ders1_.append((0, 0, 0))
    max_index = rng - 1  # max index of rng_ders1_

    for x, p in enumerate(pixel_):  # pixel p is compared to rng of prior pixels within horizontal line, summing d and m per prior pixel
        back_fd, back_fm = 0, 0  # fuzzy derivatives from rng of backward comps per pri_p
        for index, (pri_p, fd, fm) in enumerate(rng_ders1_):
            d = p - pri_p
            m = ave - abs(d)
            fd += d  # bilateral fuzzy d: running sum of differences between pri_p and all prior and subsequent pixels within rng
            fm += m  # bilateral fuzzy m: running sum of matches between pri_p and all prior and subsequent pixels within rng
            back_fd += d  # running sum of d between p and all prior pixels within rng
            back_fm += m  # running sum of m between p and all prior pixels within rng

            if index < max_index:
                rng_ders1_[index] = (pri_p, fd, fm)
            elif x > rng * 2 - 1:  # after pri_p comp over full bilateral rng
                ders1_.append((pri_p, fd, fm))  # completed bilateral tuple is transferred from rng_ders_ to ders_

        rng_ders1_.appendleft((p, back_fd, back_fm))  # new tuple with initialized d and m, maxlen displaces completed tuple
    # last incomplete rng_ders1_ in line are discarded, vs. ders1_ += reversed(rng_ders1_)
    return ders1_
    # ---------- lateral_comp() end -------------------------------------------------------------------------------------

def vertical_comp(ders1_, rng_ders2__, _xP_, _yP_, frame):
    # Comparison to bilateral rng of vertically consecutive pixels, forming ders2: pixel + lateral and vertical derivatives

    # Each of the following contains 2 types, per core variables m and d:
    xP = [[0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []],  # lateral pattern = pri_s, x0, L, I, D, Dy, M, My, Alt0:4 ders2_
          [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]]
    yP = [[0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []],
          [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]]
    xP_ = [deque(), deque()]
    yP_ = [deque(), deque()]  # line y - rng
    xbuff_ = [deque(), deque()]
    ybuff_ = [deque(), deque()]  # line y - rng - 1: _Ps buffered by previous run of scan_P_

    new_rng_ders2__ = deque()  # 2D array: line of rng_ders2_s buffered for next-line comp
    max_index = rng - 1  # max rng_ders2_ index
    x = rng  # lateral coordinate of pixel in input ders1

    for (p, d, m), rng_ders2_ in zip(ders1_, rng_ders2__):  # pixel comp to rng _pixels in rng_ders2_, summing dy and my
        index = 0
        back_dy, back_my = 0, 0
        for (_p, _d, fdy, _m, fmy) in rng_ders2_:  # vertical derivatives are incomplete; prefix '_' denotes higher-line variable

            dy = p - _p
            my = ave - abs(dy)
            fdy += dy  # running sum of differences between pixel _p and all higher and lower pixels within rng
            fmy += my  # running sum of matches between pixel _p and all higher and lower pixels within rng
            back_dy += dy   # running sum of d between pixel p and all higher pixels within rng
            back_my += my   # running sum of m between pixel p and all higher pixels within rng

            if index < max_index:
                rng_ders2_[index] = (_p, _d, fdy, _m, fmy)
            elif y > min_coord:
                ders = _p, _d, fdy, _m, fmy
                xP, xP_, xbuff_, _xP_, frame = form_P( ders, x, X - rng - 1, xP, xP_, xbuff_, _xP_, frame, 0 )  # lateral mP, typ = 0
                yP, yP_, ybuff_, _yP_, frame = form_P( ders, x, X - rng - 1, yP, yP_, ybuff_, _yP_, frame, 1 )  # vertical mP, typ = 1
            index += 1

        rng_ders2_.appendleft( ( p, d, back_dy, m, back_my ) )  # new ders2 displaces completed one in vertical rng_ders2_ via maxlen
        new_rng_ders2__.append(rng_ders2_)  # 2D array of vertically-incomplete 2D tuples, converted to rng_ders2__, for next-line vertical comp
        x += 1

    typ = dim     # terminate last higher line dP (typ = 2) within neg mPs
    while xbuff_[1]:
        hP = xbuff_[1].popleft()
        if hP[1] != 1:      # no roots
            frame = form_blob( hP, frame, typ )
    while _xP_[1]:
        hP, frame = form_segment(_xP_[1].popleft(), frame, typ)
        frame = form_blob( hP, frame, typ )

    typ += 1    # terminate last higher line dyP (typ = 3) within neg myPs
    while ybuff_[1]:
        hP = ybuff_[1].popleft()
        if hP[1] != 1:      # no roots
            frame = form_blob( hP, frame, typ )
    while _yP_[1]:
        hP, frame = form_segment(_yP_[1].popleft(), frame, typ)
        frame = form_blob( hP, frame, typ )

    return new_rng_ders2__, xP_, yP_, frame
    # ---------- vertical_comp() end ------------------------------------------------------------------------------------

def form_P(ders, x, max_x, P, P_, buff_, hP_, frame, typ, is_dP = False):
    # Initializes, and accumulates 1D pattern "
    # is_dP = bool(typ // dim), or computed directly for speed and clarity:

    p, d, dy, m, my = ders  # 2D tuple of derivatives per pixel, "y" denotes vertical vs. lateral derivatives
    if      typ == 0:   core = m;   alt0 = d;   alt1 = my;  alt2 = dy
    elif    typ == 1:   core = my;  alt0 = dy;  alt1 = m;   alt2 = d
    elif    typ == 2:   core = d;   alt0 = m;   alt1 = dy;  alt2 = my
    else:               core = dy;  alt0 = my;  alt1 = d;   alt2 = m

    s = 1 if core > 0 else 0
    pri_s, x0 = P[is_dP][:2]    # P[0] is mP, P[1] is dP
    if not ( s == pri_s or x == x0):  # P is terminated
        P, P_, buff_, hP_, frame = term_P(s, x, P, P_, buff_, hP_, frame, typ, is_dP)

    pri_s, x0, L, I, D, Dy, M, My, Alt0, Alt1, Alt2, ders_ = P[is_dP]   # Continued or initialized input and derivatives are accumulated:

    L += 1      # length of a pattern
    I += p      # summed input
    D += d      # lateral D
    Dy += dy    # vertical D
    M += m      # lateral M
    My += my    # vertical M
    Alt0 += abs(alt0)   # alternative derivative: m | d; indicate value, thus redundancy rate, of overlapping alt-core blobs
    Alt1 += abs(alt1)   # alternative direction:  x | y
    Alt2 += abs(alt2)   # alternative derivative and direction

    ders_.append(ders)  # ders2s are buffered for oriented rescan and incremental range | derivation comp
    P[is_dP] = s, x0, L, I, D, Dy, M, My, Alt0, Alt1, Alt2, ders_

    if x == max_x:  # P is terminated
        P, P_, buff_, hP_, frame = term_P(s, x + 1, P, P_, buff_, hP_, frame, typ, is_dP)

    return P, P_, buff_, hP_, frame  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_P() end -------------------------------------------------------------------------------------------

def term_P(s, x, P, P_, buff_, hP_, frame, typ, is_dP):
    # Terminates 1D pattern when sign-change is detected or at the end of P

    pri_s, x0, L, I, D, Dy, M, My, Alt0, Alt1, Alt2, ders_ = P[is_dP]
    if not is_dP and not pri_s:
        P[1] = [-1, x0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]  # dPs (P[1]) formed inside of negative mP (P[0])

        for i in range(L):
            P, P_, buff_, _P_, frame = form_P(ders_[i], x0 + i, x0 + L - 1, P, P_, buff_, hP_, frame, typ + dim, True) # is_dP = 1
        P[0] = pri_s, x0, L, I, D, Dy, M, My, Alt0, Alt1, Alt2, ders_, P_[1]

    if y == rng * 2:  # 1st line P_ is converted to init hP_;  scan_P_(), form_segment(), form_blob() use one type of Ps, hPs, buffs
        P_[is_dP].append([P[is_dP], 0, [], x - 1])  # P, roots, _fork_, x
    else:
        P_[is_dP], buff_[is_dP], hP_[is_dP], frame = scan_P_(x - 1, P[is_dP], P_[is_dP], buff_[is_dP], hP_[is_dP], frame, typ)  # P scans hP_
    P[is_dP] = s, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, []  # new P initialization

    return P, P_, buff_, hP_, frame

def scan_P_(x, P, P_, _buff_, hP_, frame, typ):
    # P scans shared-x-coordinate hPs in higher P_, combines overlapping Ps into blobs

    buff_ = deque()  # new buffer for displaced hPs (higher-line P tuples), for scan_P_(next P)
    fork_ = []  # refs to hPs connected to input P
    _x0 = 0  # to start while loop, next ini_x = _x + 1
    x0 = P[1]

    while _x0 <= x:  # while x values overlap between P and _P
        if _buff_:
            hP = _buff_.popleft()  # hP was extended to segment and buffered in prior scan_P_
        elif hP_:
            hP, frame = form_segment( hP_.popleft(), frame, typ )
        else:
            break  # higher line ends, all hPs are converted to segments
        roots = hP[1]
        _x0 = hP[5][-1][0][1]           # first_x
        _x = _x0 + hP[5][-1][0][2] - 1  # last_x = first_x + L - 1

        if P[0] == hP[6][0][0] and  not _x < x0 and not x < _x0: # P comb -> blob if s == _s, _last_x >= first_x and last_x >= _first_x
            roots += 1
            hP[1] = roots
            fork_.append(hP)  # P-connected hPs will be converted to segments at each _fork

        if _x > x:  # x overlap between hP and next P: hP is buffered for next scan_P_, else hP included in a blob segment
            buff_.append(hP)
        elif roots != 1:
            frame = form_blob(hP, frame, typ)  # segment is terminated and packed into its blob
        _x0 = _x + 1  # = first x of next _P

    buff_ += _buff_  # _buff_ is likely empty
    P_.append([P, 0, fork_, x])  # P with no overlap to next _P is extended to hP and buffered for next-line scan_P_

    return P_, buff_, hP_, frame  # hP_ and buff_ contain only remaining _Ps, with _x => next x
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_segment(hP, frame, typ):
    # Convert hP into new segment or add it to higher-line segment, merge blobs
    _P, roots, fork_, last_x = hP
    [s, first_x], params = _P[:2], list(_P[2:11])
    ave_x = (_P[2] - 1) // 2  # extra-x L = L-1 (1x in L)

    if not fork_:  # seg is initialized with initialized blob (params, coord_, remaining_roots, root_, xD)
        blob = [[s, 0, 0, 0, 0, 0, 0, 0, 0, 0], [_P[1], hP[3], y - rng - 1, 0, 0], 1, []]
        hP = [params, roots, fork_, ave_x, 0, [(_P, 0)], blob]
        blob[3].append(hP)
    else:
        if len(fork_) == 1 and fork_[0][1] == 1:  # hP has one fork: hP[2][0], and that fork has one root: hP
            # hP is merged into higher-line blob segment (Pars, roots, _fork_, ave_x, xD, Py_, blob) at hP[2][0]:
            fork = fork_[0]
            L, I, D, Dy, M, My, alt0, alt1, alt2 = params
            Ls, Is, Ds, Dys, Ms, Mys, alt0s, alt1s, alt2s = fork[0]  # seg params
            fork[0] = [Ls + L, Is + I, Ds + D, Dys + Dy, Ms + M, Mys + My, alt0s + alt0, alt1s + alt1, alt2s + alt2]
            fork[1] = roots
            dx = ave_x - fork[3]
            fork[3] = ave_x
            fork[4] += dx  # xD for seg normalization and orientation, or += |dx| for curved yL?
            fork[5].append((_P, dx))  # Py_: vertical buffer of Ps merged into seg
            hP = fork  # replace segment with including fork's segment
            blob = hP[6]

        else:  # if >1 forks, or 1 fork that has >1 roots:
            hP = [params, roots, fork_, ave_x, 0, [(_P, 0)], fork_[0][6]]  # seg is initialized with fork's blob
            blob = hP[6]
            blob[3].append(hP)  # segment is buffered into root_

            if len(fork_) > 1:  # merge blobs of all forks
                if fork_[0][1] == 1:  # if roots == 1
                    frame = form_blob(fork_[0], frame, typ, 1)  # merge seg of 1st fork into its blob

                for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                    if fork[1] == 1:
                        frame = form_blob(fork, frame, typ, 1)

                    if not fork[6] is blob:
                        [s, L, I, D, Dy, M, My, alt0, alt1, alt2], [min_x, max_x, min_y, xD, Ly], remaining_roots, root_ = fork[6]
                        blob[0][1] += L
                        blob[0][2] += I
                        blob[0][3] += D
                        blob[0][4] += Dy
                        blob[0][5] += M
                        blob[0][6] += My
                        blob[0][7] += alt0
                        blob[0][8] += alt1
                        blob[0][9] += alt2
                        blob[1][0] = min(min_x, blob[1][0])
                        blob[1][1] = max(max_x, blob[1][1])
                        blob[1][2] = min(min_y, blob[1][2])
                        blob[1][3] += xD
                        blob[1][4] += Ly
                        blob[2] += remaining_roots
                        for seg in root_:
                            if not seg is fork:
                                seg[6] = blob  # blobs in other forks are references to blob in the first fork
                                blob[3].append(seg)  # buffer of merged root segments
                        fork[6] = blob
                        blob[3].append(fork)
                    blob[2] -= 1

        blob[1][0] = min(first_x, blob[1][0])  # min_x
        blob[1][1] = max(last_x, blob[1][1])  # max_x
    return hP, frame
    # ---------- form_segment() end -----------------------------------------------------------------------------------------


def form_blob(term_seg, frame, typ, y_carry=0):
    # Terminated segment is merged into continued or initialized blob (all connected segments)

    [L, I, D, Dy, M, My, alt0, alt1, alt2], roots, fork_, x, xD, Py_, blob = term_seg  # unique blob in fork_[0][6] is ref'd by other forks
    blob[0][1] += L
    blob[0][2] += I
    blob[0][3] += D
    blob[0][4] += Dy
    blob[0][5] += M
    blob[0][6] += My
    blob[0][7] += alt0
    blob[0][8] += alt1
    blob[0][9] += alt2

    blob[1][3] += xD        # ave_x angle, to evaluate blob for re-orientation
    blob[1][4] += len(Py_)  # Ly = number of slices in segment

    blob[2] += roots - 1  # reference to term_seg is already in blob[9]
    term_seg.append(y - rng - 1 - y_carry)  # y_carry: min elevation of term_seg over current hP

    if not blob[2]:  # if remaining_roots == 0: blob is terminated and packed in frame
        [s, L, I, D, Dy, M, My, alt0, alt1, alt2], [min_x, max_x, min_y, xD, Ly], remaining_roots, root_ = blob
        if not typ:  # frame P are to compute averages, redundant for same-scope alt_frames
            frame[0][1] += I
            frame[0][2] += D
            frame[0][3] += Dy
            frame[0][4] += M
            frame[0][5] += My
        if not s and typ < dim:
            frame[0][0][typ] += L   # L of negative mblobs are summed

        frame[typ + 1][0] += xD  # ave_x angle, to evaluate frame for re-orientation
        frame[typ + 1][1] += Ly  # +L
        root_.sort(key=segment_sort_by_height)  # Sort segments by max_y
        frame[typ + 1][2].append(((s, L, I, D, Dy, M, My, alt0, alt1, alt2), (min_x, max_x, min_y, term_seg[7], xD, Ly), root_))

    return frame  # no term_seg return: no root segs refer to it
    # ---------- form_blob() end ----------------------------------------------------------------------------------------


def image_to_blobs(image):
    # Main body of the operation, postfix '_' denotes array vs. element, prefix '_' denotes higher-line vs. lower-line variable

    _xP_ = [deque(), deque()]
    _yP_ = [deque(), deque()]  # higher-line same- d-, m-, dy-, my- sign 1D patterns
    frame = [[[0, 0], 0, 0, 0, 0, 0], [0, 0, []], [0, 0, []], [0, 0, []], [0, 0, []]]  # [neg_mL, neg_myL, I, D, Dy, M, My], 4 x [xD, Ly, blob_]
    global y
    y = 0
    rng_ders2__ = []            # horizontal line of vertical buffers: 2D array of 2D tuples, deque for speed?
    pixel_ = image[0, :]    # first line of pixels
    ders1_ = lateral_comp(pixel_)   # after partial comp, while x < rng?

    for (p, d, m) in ders1_:
        ders2 = p, d, 0, m, 0       # dy, my initialized at 0
        rng_ders2_ = deque(maxlen=rng)  # vertical buffer of incomplete derivatives tuples, for fuzzy ycomp
        rng_ders2_.append(ders2)        # only one tuple in first-line rng_ders2_
        rng_ders2__.append(rng_ders2_)

    for y in range(1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?

        pixel_ = image[y, :]            # vertical coordinate y is index of new line p_
        ders1_ = lateral_comp(pixel_)   # lateral pixel comparison
        rng_ders2__, _xP_, _yP_, frame = vertical_comp( ders1_, rng_ders2__, _xP_, _yP_, frame )  # vertical pixel comparison

    # frame ends, last vertical rng of incomplete rng_ders2__ is discarded,
    # merge segs of last line into their blobs:
    y = Y
    for is_dP in range(2):
        typ = is_dP * dim
        hP_ = _xP_[is_dP]
        while hP_:
            hP, frame = form_segment( hP_.popleft(), frame, typ )
            frame = form_blob( hP, frame, typ )

        typ += 1
        hP_ = _yP_[is_dP]
        while hP_:
            hP, frame = form_segment( hP_.popleft(), frame, typ )
            frame = form_blob(hP, frame, typ)

    return frame  # frame of 2D patterns, to be outputted to level 2
    # ---------- image_to_blobs() end -----------------------------------------------------------------------------------


# ************ MAIN FUNCTIONS END ***************************************************************************************


# ************ PROGRAM BODY *********************************************************************************************

# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:

rng         = 2     # number of pixels compared to each pixel in four directions
min_coord   = rng * 2 - 1  # min x and y for form_P input: ders2 from comp over rng*2 (bidirectional: before and after pixel p)
ave         = 15    # |d| value that coincides with average match: mP filter
dim         = 2     # Number of dimesions

output = bool(0)

# Load inputs --------------------------------------------------------------------
# image = misc.face(gray=True)  # read image as 2d-array of pixels (gray scale):
# image = image.astype(int)
# or:
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/raccoon_eye.jpg')
arguments = vars(argument_parser.parse_args())
image = cv2.imread(arguments['image'], 0).astype(int)

Y, X = image.shape  # image height and width

# Main ---------------------------------------------------------------------------
start_time = time()
frame_of_blobs = image_to_blobs(image)
end_time = time() - start_time
print(end_time)

# Rebuild blob -------------------------------------------------------------------
if output:
    cv2.imwrite('./images/mblobs_horizontal.jpg', rebuild_blobs(frame_of_blobs[1]))
    cv2.imwrite('./images/mblobs_vertical.jpg', rebuild_blobs(frame_of_blobs[2]))
    cv2.imwrite('./images/dblobs_horizontal.jpg', rebuild_blobs(frame_of_blobs[3]))
    cv2.imwrite('./images/dblobs_vertical.jpg', rebuild_blobs(frame_of_blobs[4]))

# ************ PROGRAM BODY END ******************************************************************************************