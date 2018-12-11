import cv2
import argparse
import numpy
from scipy import misc
from time import time
from collections import deque

'''   
    frame() is my core algorithm of levels 1 + 2, modified for 2D: segmentation of image into blobs, then search within and between blobs.
    frame_blobs() is frame() limited to definition of initial blobs per each of 4 derivatives, vs. per 2 gradients in frame_draft.
    frame_dblobs() is updated version of frame_blobs with only one blob type: dblob, to ease debugging, currently in progress.

    Each performs several levels (Le) of encoding, incremental per scan line defined by vertical coordinate y, outlined below.
    value of y per Le line is shown relative to y of current input line, incremented by top-down scan of input image:

    1Le, line y:    x_comp(p_): lateral pixel comparison -> tuple of derivatives ders ) array ders_
    2Le, line y- 1: y_comp(ders_): vertical pixel comp -> 2D tuple ders2 ) array ders2_ 
    3Le, line y- 1+ rng*2: form_P(ders2) -> 1D pattern P
    4Le, line y- 2+ rng*2: scan_P_(P, hP) -> hP, roots: down-connections, fork_: up-connections between Ps 
    5Le, line y- 3+ rng*2: form_segment(hP, seg) -> seg: merge vertically-connected _Ps in non-forking blob segments
    6Le, line y- 4+ rng*2+ seg depth: form_blob(seg, blob): merge connected segments in fork_' incomplete blobs, recursively  

    for y = rng *2, the outputs are: y P ) y-1 hP ) y-2 seg ) y-4 blob ) y-5 frame

    Pixel comparison in 2D forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. 
    They are formed on the same level because average lateral match ~ average vertical match.
    Each vertical and horizontal derivative forms separate blobs, suppressing overlapping orthogonal representations.

    They can also be summed to estimate diagonal or hypot derivatives, for blob orientation to maximize primary derivatives.
    Orientation increases primary dimension of blob to maximize match, and decreases secondary dimension to maximize difference.
    Subsequent union of lateral and vertical patterns is by strength only, orthogonal sign is not commeasurable?

    Initial pixel comparison is not novel, I design from the scratch to make it organic part of hierarchical algorithm.
    It would be much faster with matrix computation, but this is minor compared to higher-level processing.
    I implement it sequentially for consistency with accumulation into blobs: irregular and very difficult to map to matrices.

    All 2D functions (y_comp, scan_P_, form_blob) input two lines: higher and lower, convert elements of lower line 
    into elements of new higher line, and displace elements of old higher line into higher function.
    Higher-line elements include additional variables, derived while they were lower-line elements.

    prefix '_' denotes higher-line variable or pattern, vs. same-type lower-line variable or pattern,
    postfix '_' denotes array name, vs. same-name elements of that array:
'''

# ************ UTILITY FUNCTIONS ****************************************************************************************
# Includes:
# -rebuild_blobs()
# ***********************************************************************************************************************

def rebuild_blobs( frame ):
    " Rebuilt data of blobs into an image "
    blob_image = numpy.array([[0] * X] * Y)

    for blob in frame[8]:  # Iterate through blobs
        if blob[0][0]:  # Choose positive dblobs
            for seg in blob[1]:  # Iterate through segments
                y = seg[7] - len(seg[5]) + 1
                for (P, dx) in seg[5]:
                    x = P[1]
                    for (p, d, dy, m, my) in P[8]:
                        blob_image[y, x] = 255
                        x += 1
                    y += 1

    return blob_image
    # ---------- rebuild_blobs() end ------------------------------------------------------------------------------------

# ************ UTILITY FUNCTIONS END ************************************************************************************

# ************ MAIN FUNCTIONS *******************************************************************************************
# Includes:
# -lateral_comp()
# -vertical_comp()
# -form_P()
# -P_to_segment()
# -scan_P_()
# -form_blob()
# -image_to_blobs
# ***********************************************************************************************************************

def lateral_comp(pixel_):
    " Comparison over x coordinate, within rng of consecutive pixels on each line "

    ders1_ = []  # tuples of complete 1D derivatives: summation range = rng
    rng_ders1_ = deque(maxlen=rng)  # incomplete ders1s, within rng from input pixel: summation range < rng
    rng_ders1_.append((0, 0, 0))
    max_index = rng - 1  # max index of rng_ders1_

    for x, p in enumerate(pixel_):  # pixel p is compared to rng of prior pixels within horizontal line, summing d and m per prior pixel
        back_fd, back_fm = 0, 0  # fuzzy derivatives from rng of backward comps per pri_p
        for index, (pri_p, fd, fm) in enumerate(rng_ders1_):

            d = p - pri_p
            m = ave - abs(d)
            fd += d  # bilateral fuzzy d: running sum of differences between pixel and all prior and subsequent pixels within rng
            fm += m  # bilateral fuzzy m: running sum of matches between pixel and all prior and subsequent pixels within rng
            back_fd += d
            back_fm += m  # running sum of d and m between pixel and all prior pixels within rng

            if index < max_index:
                rng_ders1_[index] = (pri_p, fd, fm)
            elif x > rng * 2 - 1:  # after pri_p comp over full bilateral rng
                ders1_.append((pri_p, fd, fm))  # completed bilateral tuple is transferred from rng_ders_ to ders_

        rng_ders1_.appendleft((p, back_fd, back_fm))  # new tuple with initialized d and m, maxlen displaces completed tuple
    ders1_ += reversed(rng_ders1_)  # tuples of last rng in line (incomplete, in reverse order), or discarded?
    return ders1_
    # ---------- lateral_comp() end -------------------------------------------------------------------------------------


def vertical_comp(ders1_, ders2__, _dP_, dframe):
    " Comparison between bilateral rng of vertically consecutive pixels, forming ders2: \
    tuple of pixel + its 2D derivatives "

    dP = 0, rng, 0, 0, 0, 0, 0, 0, []  # lateral difference pattern = pri_s, x0, L, I, D, Dy, V, Vy, ders2_
    dP_ = deque()  # line y - 1 + rng*2
    dbuff_ = deque()  # line y - 2 + rng*2: _Ps buffered by previous run of scan_P_
    new_ders2__ = deque()  # 2D array: line of ders2_s buffered for next-line comp
    max_index = rng - 1  # max ders2_ index
    min_coord = rng * 2 - 1  # min x and y for form_P input: ders2 from comp over rng*2 (bidirectional: before and after pixel p)
    x = rng  # lateral coordinate of pixel in input ders1

    for (p, d, m), ders2_ in zip(ders1_, ders2__):  # pixel comp to rng _pixels in ders2_, summing dy and my
        index = 0
        back_dy, back_my = 0, 0
        for (_p, _d, fdy, _m, fmy) in ders2_:  # vertical derivatives are incomplete; prefix '_' denotes higher-line variable

            dy = p - _p
            my = ave - abs(dy)
            fdy += dy  # running sum of differences between pixel _p and all higher and lower pixels within rng
            fmy += my  # running sum of matches between pixel _p and all higher and lower pixels within rng
            back_dy += dy
            back_my += my  # running sum of d and m between pixel _p and all higher pixels within rng

            if index < max_index:
                ders2_[index] = (_p, _d, fdy, _m, fmy)
            elif y > min_coord + ini_y:
                ders = _p, _d, fdy, _m, fmy
                dP, dP_, dbuff_, _dP_, dframe = form_P(ders, x, dP, dP_, dbuff_, _dP_, dframe)
            index += 1

        ders2_.appendleft((p, d, back_dy, m, back_my))  # new ders2 displaces completed one in vertical ders2_ via maxlen
        new_ders2__.append(ders2_)  # 2D array of vertically-incomplete 2D tuples, converted to ders2__, for next-line vertical comp
        x += 1

    if y > min_coord + ini_y:  # not-terminated P at the end of each line is buffered or scanned:

        if y == rng * 2 + ini_y or y == Y - 1:  # _P_ initialization by first line of Ps, empty until vertical_comp returns P_
            dP_.append([dP, 0, [], x - 1])  # empty _fork_ in the first line of hPs, x-1: delayed P displacement
        else:
            dP_, dbuff_, _dP_, dframe = scan_P_(x - 1, dP, dP_, dbuff_, _dP_, dframe)  # scans higher-line Ps for contiguity

    return new_ders2__, dP_, dframe
    # ---------- vertical_comp() end ------------------------------------------------------------------------------------


def form_P(ders, x, P, P_, buff_, hP_, frame):
    " Initializes, accumulates, and terminates 1D pattern: dP | vP | dyP | vyP "

    p, d, dy, v, vy = ders  # 2D tuple of derivatives per pixel, "y" denotes vertical vs. lateral derivatives
    s = 1 if d > 0 else 0  # core = 0 is negative: no selection?

    if s == P[0] or x == rng:  # s == pri_s or initialized: P is continued, else terminated:
        pri_s, x0, L, I, D, Dy, V, Vy, ders_ = P
    else:
        if y == rng * 2 + ini_y:  # 1st line: form_P converts P to initialized hP, forming initial P_ -> hP_
            P_.append([P, 0, [], x - 1])  # P, roots, _fork_, x
        else:
            P_, buff_, hP_, frame = scan_P_(x - 1, P, P_, buff_, hP_, frame)  # scans higher-line Ps for contiguity
            # x-1 for prior p
        x0, L, I, D, Dy, V, Vy, ders_ = x, 0, 0, 0, 0, 0, 0, []  # new P initialization

    L += 1  # length of a pattern, continued or initialized input and derivatives are accumulated:
    I += p  # summed input
    D += d  # lateral D
    Dy += dy  # vertical D
    V += v  # lateral V
    Vy += vy  # vertical V
    ders_.append(ders)  # ders2s are buffered for oriented rescan and incremental range | derivation comp

    P = s, x0, L, I, D, Dy, V, Vy, ders_
    return P, P_, buff_, hP_, frame  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_P() end -------------------------------------------------------------------------------------------


def P_to_segment( hP, frame ):
    " Turn hP into new segment or add to higher-line segment, also handle blob-merging "
    _P, roots = hP[:2]
    if y == rng * 2 + 1 + ini_y:  # 1st-line scan_P_ converts each hP to blob segment: Pars, roots, _fork_, ave_x, Dx, Py_, blob
        hP[0] = list(_P[2:8])
        hP += 0, [(_P, 0)], [_P[0], 0, 0, 0, 0, 0, 0, 0, y - rng - 1, [hP], 1] # form new blob, with min_y = current line
    else:
        if len(hP[2]) == 1 and hP[2][0][1] == 1:  # hP has one fork: hP[2][0], and that fork has one root: hP
            # hP is merged in blob segment (Pars, roots, _fork_, ave_x, Dx, Py_, blob) at hP[2][0]:
            s, x0, L, I, D, Dy, V, Vy, ders_ = _P
            Ls, Is, Ds, Dys, Vs, Vys = hP[2][0][0]
            hP[2][0][0] = [Ls + L, Is + I, Ds + D, Dys + Dy, Vs + V, Vys + Vy]  # seg parameters
            hP[2][0][1] = roots
            ave_x = (_P[2] - 1) // 2  # extra-x L = L-1 (1x in L)
            hP[2][0][3] = ave_x
            dx = ave_x - hP[2][0][3]
            hP[2][0][4] += dx  # Dx for seg normalization and orientation, or += |dx| for curved yL?
            hP[2][0][5].append((_P, dx))  # Py_: vertical buffer of Ps merged into seg
            hP = hP[2][0]  # hP id change?
            # hP[:] = hP[2][0]  # replace segment with including fork's segment

        elif not hP[2]:  # new seg with new blob
            hP[0] = list(_P[2:8])  # seg parameters
            hP += 0, [(_P, 0)], [_P[0], 0, 0, 0, 0, 0, 0, 0, y - rng - 1, [hP], 1]  # last blob var is roots

        else:  # if >1 forks, or 1 fork that has >1 roots:
            hP[0] = list(_P[2:8]);
            hP += 0, [(_P, 0)], hP[2][0][6]  # seg is initialized with fork's blob
            blob = hP[6]
            blob[9].append(hP)  # hP is buffered into root_

            if len(hP[2]) > 1:  # merge blobs of all forks
                if hP[2][0][1] == 1:
                    frame = form_blob(hP[2][0], frame, 1)  # merge seg of 1st fork into its blob

                for fork in hP[2][1:len(hP[2])]:  # merge blobs of other forks into blob of 1st fork
                    if fork[1] == 1:
                        frame = form_blob(fork, frame, 1)

                    if not fork[6] is blob:
                        blob[1] += fork[6][1]
                        blob[2] += fork[6][2]
                        blob[3] += fork[6][3]
                        blob[4] += fork[6][4]
                        blob[5] += fork[6][5]
                        blob[6] += fork[6][6]
                        blob[7] += fork[6][7]
                        blob[8] = min(fork[6][8], blob[8])
                        blob[10] += fork[6][10]
                        for seg in fork[6][9]:
                            seg[6] = blob  # blobs in other forks are references to blob in the first fork
                            blob[9].append(seg)  # buffer of merged root segments
                    blob[10] -= 1
    return hP, frame
    # ---------- handle_P() end -----------------------------------------------------------------------------------------


def scan_P_(x, P, P_, _buff_, hP_, frame):
    " P scans shared-x-coordinate hPs in higher P_, combines overlapping Ps into blobs "

    buff_ = deque()  # new buffer for displaced hPs (higher-line P tuples), for scan_P_(next P)
    fork_ = []  # refs to hPs connected to input P
    ini_x = 0  # to start while loop, next ini_x = _x + 1

    while ini_x <= x:  # while x values overlap between P and _P
        if _buff_:
            hP = _buff_.popleft()  # higher-line P tuple buffered in prior scan_P_, seg id == _fork_ id, referenced by root Ps
        elif hP_:
            hP, frame = P_to_segment( hP_.popleft(), frame )  # roots = 0: number of Ps connected to _P: pri_s, x0, L, I, D, Dy, V, Vy, ders_
        else:
            break  # higher line ends, all hPs are converted to segments

        _P, roots, _fork_, _x, dx, Py_, blob = hP

        if P[0] == hP[6][0]:  # if s == _s: core sign match, + selective inclusion if contiguity eval?
            roots += 1;
            hP[1] = roots
            fork_.append(hP)  # P-connected hPs will be converted to segments at each _fork

        if _x > x:  # x overlap between hP and next P: hP is buffered for next scan_P_, else hP included in a blob segment
            buff_.append(hP)
        elif roots != 1:
            frame = form_blob(hP, frame)  # bottom segment is terminated and added to internal blob

        ini_x = _x + 1  # = first x of next _P

    buff_ += _buff_  # _buff_ is likely empty
    P_.append([P, 0, fork_, x])  # P with no overlap to next _P is extended to hP and buffered for next-line scan_P_

    return P_, buff_, hP_, frame  # hP_ and buff_ contain only remaining _Ps, with _x => next x
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_blob(term_seg, frame, y_carry=0):
    " Terminated segment is merged into continued or initialized blob (all connected segments) "

    [L, I, D, Dy, V, Vy], roots, fork_, x, xD, Py_, blob = term_seg  # unique blob in fork_[0][6] is ref'd by other forks
    blob[1] += L
    blob[2] += I
    blob[3] += D
    blob[4] += Dy
    blob[5] += V
    blob[6] += Vy
    blob[7] += xD
    blob[10] += roots - 1  # reference to term_seg is already in blob[9]
    term_seg.append(y - rng - 1 - y_carry)  # y_carry: elevation of term_seg y over current hP' y

    if not blob[10]:
        s, L, I, D, Dy, V, Vy, xD, min_y, root_, remaining_roots = blob
        frame[0] += L  # frame P are to compute averages, redundant for same-scope alt_frames
        frame[1] += I
        frame[2] += D
        frame[3] += Dy
        frame[4] += V
        frame[5] += Vy
        frame[6] += xD  # for frame orient eval, += |xd| for curved max_L?
        frame[7] += min_y - term_seg[7] + 1 # Height of the whole blob
        frame[8].append(((s, L, I, D, Dy, V, Vy, x - xD // 2, xD, min_y, term_seg[7]), root_))  # blob_ buffer

    return frame  # no term_seg return: no root segs refer to it
    # ---------- form_blob() end ----------------------------------------------------------------------------------------


def image_to_blobs(image):
    " Main body of the operation, \
    postfix '_' denotes array vs. element, prefix '_' denotes higher-line vs. lower-line variable "

    _P_ = deque()  # higher-line same- d-, v-, dy-, vy- sign 1D patterns
    frame = [0, 0, 0, 0, 0, 0, 0, 0, []]  # L, I, D, Dy, V, Vy, xD, yD, blob_
    global y;
    y = ini_y  # initial line
    ders2__ = []  # horizontal line of vertical buffers: 2D array of 2D tuples, deque for speed?
    pixel_ = image[ini_y, :]  # first line of pixels at y == 0
    ders1_ = lateral_comp(pixel_)  # after part_comp (pop, no t_.append) while x < rng?

    for (p, d, m) in ders1_:
        ders2 = p, d, 0, m, 0  # dy, my initialized at 0
        ders2_ = deque(maxlen=rng)  # vertical buffer of incomplete derivatives tuples, for fuzzy ycomp
        ders2_.append(ders2)  # only one tuple in first-line ders2_
        ders2__.append(ders2_)

    for y in range(ini_y + 1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?

        pixel_ = image[y, :]  # vertical coordinate y is index of new line p_
        ders1_ = lateral_comp(pixel_)  # lateral pixel comparison
        ders2__, _P_, frame = vertical_comp(ders1_, ders2__, _P_, frame)  # vertical pixel comparison

    # frame ends, last vertical rng of incomplete ders2__ is discarded,
    # merge segs of last line into their blobs:
    y = Y
    hP_ = _P_
    while hP_:
        hP, frame = P_to_segment( hP_.popleft(), frame )  # roots = 0: number of Ps connected to _P: pri_s, L, I, D, Dy, V, Vy, ders_
        frame = form_blob(hP, frame)
    return frame  # frame of 2D patterns to be outputted to level 2
    # ---------- image_to_blobs() end -----------------------------------------------------------------------------------

# ************ MAIN FUNCTIONS END ***************************************************************************************



# ************ PROGRAM BODY *********************************************************************************************

# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:

rng = 2  # number of pixels compared to each pixel in four directions
ave = 31  # |d| value that coincides with average match: value pattern filter
ave_rate = 0.25  # average match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)
ini_y = 0  # that area in test image seems to be the most diverse

# Load inputs --------------------------------------------------------------------
image = misc.face(gray=True)  # read image as 2d-array of pixels (gray scale):
image = image.astype(int)
# or:
# argument_parser = argparse.ArgumentParser()
# argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/blobs_test.jpg')
# arguments = vars(argument_parser.parse_args())
# image = cv2.imread(arguments['image'], 0).astype(int)

Y, X = image.shape  # image height and width

# Main ---------------------------------------------------------------------------
start_time = time()
frame_of_blobs = image_to_blobs(image)
end_time = time() - start_time
print(end_time)

# Rebuild blob -------------------------------------------------------------------
cv2.imwrite('./images/blobs.jpg', rebuild_blobs( frame_of_blobs ))

# ************ PROGRAM BODY END ******************************************************************************************