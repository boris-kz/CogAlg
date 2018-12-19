import cv2
import argparse
import numpy as np
from scipy import misc
from time import time
from collections import deque

'''   
    frame() is core algorithm of levels 1 + 2, modified for 2D: segmentation of image into blobs, then recursive search within blobs.
    frame_blobs() is frame() limited to definition of initial blobs per each of 4 derivatives, vs. per 2 gradients in frame_draft.
    frame_dblobs() is updated version of frame_blobs with only one blob verte: dblob, to ease debugging, currently in progress.

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

    prefix '_' denotes higher-line variable or pattern, vs. same-verte lower-line variable or pattern,
    postfix '_' denotes array name, vs. same-name elements of that array:
'''


# ************ UTILITY FUNCTIONS ****************************************************************************************
# Includes:
# -rebuild_blobs()
# ***********************************************************************************************************************

def rebuild_blobs(frame):
    " Rebuilt data of blobs into an image "
    blob_image = np.array([[[127] * 4] * X] * Y)

    for index, blob in enumerate(frame[1]):  # Iterate through blobs
        # blob_img = np.array([[[127] * 4] * X] * Y)
        for seg in blob[2]:  # Iterate through segments
            y = seg[7]
            for (P, dx) in reversed(seg[5]):
                x = P[1]
                for i in range(P[2]):
                    blob_image[y, x, : 3] = [255, 255, 255] if P[0] else [0, 0, 0]
                    # blob_img[y, x, : 3] = [255, 255, 255] if P[0] else [0, 0, 0]
                    x += 1
                y -= 1
        # min_x, max_x, min_y, max_y = blob[1]
        # cv2.rectangle(blob_img, (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), (0, 255, 255), 1)
        # cv2.imwrite('./images/raccoon_eye/blob%d.jpg' % (index), blob_img)

    return blob_image
    # ---------- rebuild_blobs() end ------------------------------------------------------------------------------------


def segment_sort_by_height(seg):
    " Used in sort() function "
    return seg[7]
    # ---------- segment_sort_by_height() end ---------------------------------------------------------------------------


# ************ UTILITY FUNCTIONS END ************************************************************************************

# ************ MAIN FUNCTIONS *******************************************************************************************
# Includes:
# -lateral_comp()
# -vertical_comp()
# -form_P()
# -scan_P_()
# -form_segment()
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
    # last incomplete rng_ders1_ in line are discarded, vs. ders1_ += reversed(rng_ders1_)
    ders1_.append((0, 0, 0))  # For last ders in last P
    return ders1_
    # ---------- lateral_comp() end -------------------------------------------------------------------------------------


def vertical_comp(ders1_, ders2__, _P_, frame):
    " Comparison to bilateral rng of vertically consecutive pixels, forming ders2: pixel + lateral and vertical derivatives"

    # lateral difference pattern = pri_s, x0, L, I, D, Dy, V, Vy, ders2_
    P = [[0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []],
         [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]]
    P_ = [deque(), deque()]  # line y - 1 + rng*2
    buff_ = [deque(), deque()]  # line y - 2 + rng*2: _Ps buffered by previous run of scan_P_
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
                for vert in range(2):  # vertes of P: dP | dyP | mP | myP
                    P[vert], P_[vert], buff_[vert], _P_[vert], frame = form_P(ders, x, P[vert], P_[vert], buff_[vert], _P_[vert], frame, vert)
            index += 1

        ders2_.appendleft((p, d, back_dy, m, back_my))  # new ders2 displaces completed one in vertical ders2_ via maxlen
        new_ders2__.append(ders2_)  # 2D array of vertically-incomplete 2D tuples, converted to ders2__, for next-line vertical comp
        x += 1

    return new_ders2__, P_, frame
    # ---------- vertical_comp() end ------------------------------------------------------------------------------------


def form_P(ders, x, P, P_, buff_, hP_, frame, vert):
    " Initializes, accumulates, and terminates 1D pattern"

    p, d, dy, m, my = ders  # 2D tuple of derivatives per pixel, "y" denotes vertical vs. lateral derivatives
    if vert:    core = my; alt_der = dy; alt_dir = m; alt_both = d
    else:       core = m; alt_der = d; alt_dir = my; alt_both = dy

    s = 1 if core > 0 else 0
    pri_s, x0, L, I, D, Dy, M, My, alt_Der, alt_Dir, alt_Both, ders_ = P

    if not (s == pri_s or x == rng) or x == X - rng:  # P is terminated
        if y == rng * 2 + ini_y:  # 1st line: form_P converts P to initialized hP, forming initial P_ -> hP_
            P_.append([P, 0, [], x - 1])  # P, roots, _fork_, x
        else:
            if not pri_s:  # dPs formed inside of negative mP
                dP_ = [];
                dP = -1, x0, 0, 0, 0, 0, 0, 0, []  # pri_s, L, I, D, Dy, M, My, ders_
                ders_.append((0, 0, 0, 0, 0))
                for i in range(L + 1):
                    ip, id, idy, im, imy = ders_[i]
                    if vert:    sd = 1 if idy > 0 else 0
                    else:       sd = 1 if id > 0 else 0
                    pri_sd, x0d, Ld, Id, Dd, Dyd, Md, Myd, sders_ = dP
                    if (pri_sd != sd and not i == 0) or i == L:
                        dP_.append(dP)
                        x0d, Ld, Id, Dd, Dyd, Md, Myd, sders_ = x0 + i, 0, 0, 0, 0, 0, 0, []
                    Ld += 1
                    Id += ip
                    Dd += id
                    Dyd += idy
                    Md += im
                    Myd += imy
                    sders_.append((ip, id, idy, im, imy))
                    dP = sd, x0d, Ld, Id, Dd, Dyd, Md, Myd, sders_

                P = pri_s, x0, L, I, D, Dy, M, My, alt_Der, alt_Dir, alt_Both, dP_

            P_, buff_, hP_, frame = scan_P_(x - 1, P, P_, buff_, hP_, frame, vert)  # scans higher-line Ps for contiguity
            # x-1 for prior p
        x0, L, I, D, Dy, M, My, alt_Der, alt_Dir, alt_Both, ders_ = x, 0, 0, 0, 0, 0, 0, 0, 0, 0, []  # new P initialization

    L += 1  # length of a pattern, continued or initialized input and derivatives are accumulated:
    I += p  # summed input
    D += d  # lateral D
    Dy += dy  # vertical D
    M += m  # lateral M
    My += my  # vertical M
    alt_Der += abs(alt_der)  # abs alt cores indicate value of alt-core Ps, to compute P redundancy rate
    alt_Dir += abs(alt_dir)  # vs. specific overlaps: cost > gain in precision?
    alt_Both += abs(alt_both)

    # in frame_x_blobs, alt_Der and alt_Both are computed for comp_P eval, but add to rdn only within neg mPs

    ders_.append(ders)  # ders2s are buffered for oriented rescan and incremental range | derivation comp
    P = s, x0, L, I, D, Dy, M, My, alt_Der, alt_Dir, alt_Both, ders_

    return P, P_, buff_, hP_, frame  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_P() end -------------------------------------------------------------------------------------------


def scan_P_(x, P, P_, _buff_, hP_, frame, vert):
    " P scans shared-x-coordinate hPs in higher P_, combines overlapping Ps into blobs "

    buff_ = deque()  # new buffer for displaced hPs (higher-line P tuples), for scan_P_(next P)
    fork_ = []  # refs to hPs connected to input P
    ini_x = 0  # to start while loop, next ini_x = _x + 1

    while ini_x <= x:  # while x values overlap between P and _P
        if _buff_:
            hP = _buff_.popleft()  # hP was extended to segment and buffered in prior scan_P_
        elif hP_:
            hP, frame = form_segment(hP_.popleft(), frame, vert)
        else:
            break  # higher line ends, all hPs are converted to segments

        roots = hP[1]
        if P[0] == hP[6][0][0]:  # if s == _s: core sign match, + selective inclusion if contiguity eval?
            roots += 1;
            hP[1] = roots
            fork_.append(hP)  # P-connected hPs will be converted to segments at each _fork

        _x = hP[5][-1][0][1] + hP[5][-1][0][2] - 1  # last_x = first_x + L - 1

        if _x > x:  # x overlap between hP and next P: hP is buffered for next scan_P_, else hP included in a blob segment
            buff_.append(hP)
        elif roots != 1:
            frame = form_blob(hP, frame, vert)  # segment is terminated and packed into its blob

        ini_x = _x + 1  # = first x of next _P

    buff_ += _buff_  # _buff_ is likely empty
    P_.append([P, 0, fork_, x])  # P with no overlap to next _P is extended to hP and buffered for next-line scan_P_

    return P_, buff_, hP_, frame  # hP_ and buff_ contain only remaining _Ps, with _x => next x
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_segment(hP, frame, vert):
    " Convert hP into new segment or add it to higher-line segment, merge blobs "
    _P, roots, fork_, last_x = hP
    [s, first_x], params = _P[:2], list(_P[2:11])
    ave_x = (_P[2] - 1) // 2  # extra-x L = L-1 (1x in L)

    if not fork_:  # seg is initialized with initialized blob (params, coordinates, remaining_roots, root_, xD)
        blob = [[s, 0, 0, 0, 0, 0, 0, 0, 0, 0], [_P[1], hP[3], y - rng - 1], 1, [], 0]
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
                    frame = form_blob(fork_[0], frame, vert, 1)  # merge seg of 1st fork into its blob

                for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                    if fork[1] == 1:
                        frame = form_blob(fork, frame, vert, 1)

                    if not fork[6] is blob:
                        [s, L, I, D, Dy, M, My, alt0, alt1, alt2], [min_x, max_x, min_y], remaining_roots, root_, xD = fork[6]
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
                        blob[2] += remaining_roots
                        blob[4] += xD
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


def form_blob(term_seg, frame, vert, y_carry=0):
    " Terminated segment is merged into continued or initialized blob (all connected segments) "

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
    blob[2] += roots - 1  # reference to term_seg is already in blob[9]
    blob[4] += xD  # ave_x angle, to evaluate blob for re-orientation
    term_seg.append(y - rng - 1 - y_carry)  # y_carry: min elevation of term_seg over current hP

    if not blob[2]:  # if remaining_roots == 0: blob is terminated and packed in frame
        [s, L, I, D, Dy, M, My, alt0, alt1, alt2], [min_x, max_x, min_y], remaining_roots, root_, xD = blob
        if not vert:  # frame P are to compute averages, redundant for same-scope alt_frames
            frame[0][0] += L
            frame[0][1] += I
            frame[0][2] += D
            frame[0][3] += Dy
            frame[0][4] += M
            frame[0][5] += My

        frame[vert + 1][0] += xD  # ave_x angle, to evaluate frame for re-orientation
        root_.sort(key=segment_sort_by_height)  # Sort segments by max_y
        frame[vert + 1][1].append(((s, L, I, D, Dy, M, My, alt0, alt1, alt2), (min_x, max_x, min_y, term_seg[7]), root_, xD))

    return frame  # no term_seg return: no root segs refer to it
    # ---------- form_blob() end ----------------------------------------------------------------------------------------


def image_to_blobs(image):
    " Main body of the operation, postfix '_' denotes array vs. element, prefix '_' denotes higher-line vs. lower-line variable "

    _P_ = [deque(), deque(), deque(), deque()]  # higher-line same- d-, m-, dy-, my- sign 1D patterns
    frame = [[0, 0, 0, 0, 0, 0], [0, []], [0, []]]  # [L, I, D, Dy, M, My], 4 x [xD, blob_]

    global y
    y = ini_y  # initial line
    ders2__ = []  # horizontal line of vertical buffers: 2D array of 2D tuples, deque for speed?
    pixel_ = image[ini_y, :]  # first line of pixels at y == 0
    ders1_ = lateral_comp(pixel_)  # after partial comp, while x < rng?

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
    for vert in range(2):
        hP_ = _P_[vert]
        while hP_:
            hP, frame = form_segment(hP_.popleft(), frame, vert)
            frame = form_blob(hP, frame, vert)

    return frame  # frame of 2D patterns, to be outputted to level 2
    # ---------- image_to_blobs() end -----------------------------------------------------------------------------------


# ************ MAIN FUNCTIONS END ***************************************************************************************


# ************ PROGRAM BODY *********************************************************************************************

# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:

rng = 2  # number of pixels compared to each pixel in four directions
ave = 15  # |d| value that coincides with average match: mP filter
ave_rate = 0.25  # not used; match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)
ini_y = 0  # not used
max_frame_count = 1000

# Load inputs --------------------------------------------------------------------
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-v', '--video', help='path to video file', default= './videos/Test.avi')
arguments = vars(argument_parser.parse_args())
vid = cv2.VideoCapture(arguments['video'], 0)

ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
Y, X = frame.shape  # image height and width

# Main ---------------------------------------------------------------------------
start_time = time()

# hor_blob_ouput = cv2.VideoWriter("./videos/mxblobs.avi", -1, 29,(X,Y))
ver_blob_ouput = cv2.VideoWriter("./videos/myblobs.avi", -1, 29,(X,Y))

frame_count = 0

while ( vid.isOpened() and frame_count < max_frame_count):
    # Capture frame-by-frame ---------------------------------------------------------
    ret, frame = vid.read()
    if not ret:
        break

    print 'Current frame: %d\n' % ( frame_count )

    # Our operations on the frame come here ------------------------------------------
    frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )

    # Main operation -----------------------------------------------------------------
    frame_of_blobs = image_to_blobs( frame )

    # Rebuild blob -------------------------------------------------------------------
    # hor_blob_ouput.write( np.uint8( rebuild_blobs( frame_of_blobs[1] ) ) )
    ver_blob_ouput.write( np.uint8( rebuild_blobs( frame_of_blobs[2] ) ) )

    frame_count += 1

cv2.destroyAllWindows()
# hor_blob_ouput.release()
ver_blob_ouput.release()
end_time = time() - start_time
print(end_time)

# ************ PROGRAM BODY END ******************************************************************************************