import cv2
import argparse
import numpy
from scipy import misc
from time import time
from collections import deque

'''   
    frame() is my core algorithm of levels 1 + 2, modified for 2D: segmentation of image into blobs, then search within and between blobs.
    frame_blobs() is frame() limited to definition of initial blobs per each of 4 derivatives, vs. per 2 gradients in frame_draft.
    frame_dblobs() is updated version of frame_blobs with only one blob type: dblob, to ease debugging.
    frame_x_blob() forms dblobs only inside negative mblobs, to reduce redundancy

    Each performs several levels (Le) of encoding, incremental per scan line defined by vertical coordinate y, outlined below.
    value of y per Le line is shown relative to y of current input line, incremented by top-down scan of input image:

    1Le, line y:    x_comp(p_): lateral pixel comparison -> tuple of derivatives ders ) array ders_
    2Le, line y- 1: y_comp(ders_): vertical pixel comp -> 2D tuple ders2 ) array ders2_ 
    3Le, line y- 1+ rng*2: form_P(ders2) -> 1D pattern P
    4Le, line y- 2+ rng*2: scan_P_(P, hP) -> hP, roots: down-connections, fork_: up-connections between Ps 
    5Le, line y- 3+ rng*2: form_segment(hP, seg) -> seg: merge vertically-connected _Ps in non-forking blob segments
    6Le, line y- 4+ rng*2+ seg depth: form_blob(seg, blob): merge connected segments in fork_' incomplete blobs, recursively  

    for y = rng *2: line y == P_, line y-1 == hP_, line y-2 == seg_, line y-4 == blob_

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


# ************ UTILITY FUNCTIONS ****************************************************************************************
# Includes:
# -rebuild_blobs()
# ***********************************************************************************************************************

def rebuild_blobs(frame):
    " Rebuilt data of blobs into an image "
    blob_image = numpy.array([[[127] * 4] * X] * Y)

    for index, blob in enumerate(frame[2]):  # Iterate through blobs
        blob_img = numpy.array([[[127] * 4] * X] * Y)
        for seg in blob[2]:  # Iterate through segments
            y = seg[7]
            for (P, dx) in reversed(seg[5]):
                x = P[1]
                for i in range(P[2]):
                    blob_image[y, x, : 3] = [255, 255, 255] if P[0] else [0, 0, 0]
                    blob_img[y, x, : 3] = [255, 255, 255] if P[0] else [0, 0, 0]
                    x += 1
                y -= 1;

        min_x, max_x, min_y, max_y = blob[1]
        cv2.rectangle(blob_img, (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), (0, 255, 255), 1)
        cv2.imwrite('./images/raccoon_eye/blob%d.jpg' %(index), blob_img)

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
    ders1_.append( ( 0, 0, 0 ) )    # For last ders in last P
    return ders1_
    # ---------- lateral_comp() end -------------------------------------------------------------------------------------


def vertical_comp( ders1_, ders2__, _P_, frame_ ):
    " Comparison to bilateral rng of vertically consecutive pixels, forming ders2: pixel + lateral and vertical derivatives"

    # lateral difference pattern = pri_s, x0, L, I, D, Dy, V, Vy, ders2_
    P = [[0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []], \
         [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []], \
         [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []], \
         [0, rng, 0, 0, 0, 0, 0, 0, 0, 0, 0, []]]
    P_ = [deque(), deque(), deque(), deque()]  # line y - 1 + rng*2
    buff_ = [deque(), deque(), deque(), deque()]  # line y - 2 + rng*2: _Ps buffered by previous run of scan_P_
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
                for typ in range(4):
                    P[typ], P_[typ], buff_[typ], _P_[typ], frame_[typ] \
                        = form_P(ders, x, P[typ], P_[typ], buff_[typ], _P_[typ], frame_[typ], typ)
            index += 1

        ders2_.appendleft((p, d, back_dy, m, back_my))  # new ders2 displaces completed one in vertical ders2_ via maxlen
        new_ders2__.append(ders2_)  # 2D array of vertically-incomplete 2D tuples, converted to ders2__, for next-line vertical comp
        x += 1

    return new_ders2__, P_, frame_
    # ---------- vertical_comp() end ------------------------------------------------------------------------------------


def form_P( ders, x, P, P_, buff_, hP_, frame, typ ):
    " Initializes, accumulates, and terminates 1D pattern"

    p, d, dy, m, my = ders  # 2D tuple of derivatives per pixel, "y" denotes vertical vs. lateral derivatives
    if      typ == 0:   core = d; alt_der = m; alt_dir = dy; alt_both = my  # core: variable that defines current type of pattern,
    elif    typ == 1:   core = m; alt_der = d; alt_dir = my; alt_both = dy  # alt cores define overlapping alternative-type patterns:
    elif    typ == 2:   core = dy; alt_der = my; alt_dir = d; alt_both = m  # alt derivative, alt direction, alt derivative_and_direction
    else:               core = my; alt_der = dy; alt_dir = m; alt_both = d

    s = 1 if core > 0 else 0

    pri_s, x0, L, I, D, Dy, M, My, alt_Der, alt_Dir, alt_Both, ders_ = P

    if not (s == pri_s or x == rng) or x == X - rng:  # P is terminated
        if y == rng * 2 + ini_y:  # 1st line: form_P converts P to initialized hP, forming initial P_ -> hP_
            P_.append([P, 0, [], x - 1])  # P, roots, _fork_, x
        else:
            P_, buff_, hP_, frame = scan_P_(x - 1, P, P_, buff_, hP_, frame, typ)  # scans higher-line Ps for contiguity
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
    ders_.append(ders)  # ders2s are buffered for oriented rescan and incremental range | derivation comp

    P = s, x0, L, I, D, Dy, M, My, alt_Der, alt_Dir, alt_Both, ders_
    return P, P_, buff_, hP_, frame  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_P() end -------------------------------------------------------------------------------------------


def scan_P_(x, P, P_, _buff_, hP_, frame, typ ):
    " P scans shared-x-coordinate hPs in higher P_, combines overlapping Ps into blobs "

    buff_ = deque()  # new buffer for displaced hPs (higher-line P tuples), for scan_P_(next P)
    fork_ = []  # refs to hPs connected to input P
    ini_x = 0  # to start while loop, next ini_x = _x + 1

    while ini_x <= x:  # while x values overlap between P and _P
        if _buff_:
            hP = _buff_.popleft()  # hP was extended to segment and buffered in prior scan_P_
        elif hP_:
            hP, frame = form_segment(hP_.popleft(), frame, typ)
        else:
            break  # higher line ends, all hPs are converted to segments

        s, roots = hP[6][0][0], hP[1]
        if P[0] == s:  # if s == _s: core sign match, + selective inclusion if contiguity eval?
            roots += 1;
            hP[1] = roots
            fork_.append(hP)  # P-connected hPs will be converted to segments at each _fork

        _x = hP[5][-1][0][1] + hP[5][-1][0][2] - 1  # last_x = first_x + L - 1

        if _x > x:  # x overlap between hP and next P: hP is buffered for next scan_P_, else hP included in a blob segment
            buff_.append(hP)
        elif roots != 1:
            frame = form_blob(hP, frame, typ)  # segment is terminated and packed into its blob

        ini_x = _x + 1  # = first x of next _P

    buff_ += _buff_  # _buff_ is likely empty
    P_.append([P, 0, fork_, x])  # P with no overlap to next _P is extended to hP and buffered for next-line scan_P_

    return P_, buff_, hP_, frame  # hP_ and buff_ contain only remaining _Ps, with _x => next x
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_segment(hP, frame, typ):
    " Convert hP into new segment or add it to higher-line segment, merge blobs "
    _P, roots, fork_, last_x = hP
    [ s, first_x ], params = _P[:2], list(_P[2:11])
    ave_x = (_P[2] - 1) // 2  # extra-x L = L-1 (1x in L)

    if not fork_:  # seg is initialized with initialized blob
        # Init blob: pars, coordinates, remaining_roots, root_, xD
        blob = [[ s, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [ _P[1], hP[3], y - rng - 1 ], 1, [], 0]
        hP = [ params, roots, fork_, ave_x, 0, [(_P, 0)], blob ]
        blob[3].append(hP)
    else:
        if len(fork_) == 1 and fork_[0][1] == 1:  # hP has one fork: hP[2][0], and that fork has one root: hP
            # hP is merged into higher-line blob segment (Pars, roots, _fork_, ave_x, xD, Py_, blob) at hP[2][0]:
            fork = fork_[0]
            L, I, D, Dy, M, My, alt_Der, alt_Dir, alt_Both = params
            Ls, Is, Ds, Dys, Ms, Mys, alt_Ders, alt_Dirs, alt_Boths = fork[0]
            fork[0] = [Ls + L, Is + I, Ds + D, Dys + Dy, Ms + M, Mys + My, \
                           alt_Ders + alt_Der, alt_Dirs + alt_Dir, alt_Boths + alt_Both]  # seg parameters
            fork[1] = roots
            dx = ave_x - fork[3]
            fork[3] = ave_x
            fork[4] += dx  # xD for seg normalization and orientation, or += |dx| for curved yL?
            fork[5].append((_P, dx))  # Py_: vertical buffer of Ps merged into seg
            hP = fork  # replace segment with including fork's segment
            blob = hP[6]

        else:  # if >1 forks, or 1 fork that has >1 roots:
            hP = [params, roots, fork_, ave_x, 0, [(_P, 0)], fork_[0][6]] # seg is initialized with fork's blob
            blob = hP[6]
            blob[3].append(hP)  # segment is buffered into root_

            if len(fork_) > 1:  # merge blobs of all forks
                if fork_[0][1] == 1:    # if roots == 1
                    frame = form_blob(fork_[0], frame, typ, 1)  # merge seg of 1st fork into its blob

                for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                    if fork[1] == 1:
                        frame = form_blob(fork, frame, typ, 1)

                    if not fork[6] is blob:
                        [ s, L, I, D, Dy, M, My, alt_Der, alt_Dir, alt_Both ], [ min_x, max_x, min_y ], remaining_roots, root_, xD = fork[6]
                        blob[0][1] += L
                        blob[0][2] += I
                        blob[0][3] += D
                        blob[0][4] += Dy
                        blob[0][5] += M
                        blob[0][6] += My
                        blob[0][7] += alt_Der
                        blob[0][8] += alt_Dir
                        blob[0][9] += alt_Both
                        blob[1][0] = min(min_x, blob[1][0])
                        blob[1][1] = max(max_x, blob[1][1])
                        blob[1][2] = min(min_y, blob[1][2])
                        blob[2] += remaining_roots
                        blob[4] += xD
                        for seg in root_:
                            if not seg is fork:
                                seg[6] = blob       # blobs in other forks are references to blob in the first fork
                                blob[3].append(seg) # buffer of merged root segments
                        fork[6] = blob
                        blob[3].append(fork)
                    blob[2] -= 1

        blob[1][0] = min(first_x, blob[1][0])  # min_x
        blob[1][1] = max(last_x, blob[1][1])  # max_x
    return hP, frame
    # ---------- form_segment() end -----------------------------------------------------------------------------------------


def form_blob(term_seg, frame, typ, y_carry = 0):
    " Terminated segment is merged into continued or initialized blob (all connected segments) "

    [L, I, D, Dy, M, My, alt_Der, alt_Dir, alt_Both], roots, fork_, x, xD, Py_, blob = term_seg  # unique blob in fork_[0][6] is ref'd by other forks
    blob[0][1] += L
    blob[0][2] += I
    blob[0][3] += D
    blob[0][4] += Dy
    blob[0][5] += M
    blob[0][6] += My
    blob[0][7] += alt_Der
    blob[0][8] += alt_Dir
    blob[0][9] += alt_Both
    blob[2] += roots - 1  # reference to term_seg is already in blob[9]
    blob[4] += xD  # ave_x angle, to evaluate blob for re-orientation
    term_seg.append(y - rng - 1 - y_carry)  # y_carry: elevation of term_seg over current hP

    if not blob[2]:  # if remaining_roots == 0: blob is terminated and packed in frame
        [s, L, I, D, Dy, M, My, alt_Der, alt_Dir, alt_Both], [min_x, max_x, min_y], remaining_roots, root_, xD = blob
        if not typ: # frame P are to compute averages, redundant for same-scope alt_frames
            frame[0][0] += L
            frame[0][1] += I
            frame[0][2] += D
            frame[0][3] += Dy
            frame[0][4] += M
            frame[0][5] += My

        frame[1][0] += xD  # ave_x angle, to evaluate frame for re-orientation
        frame[1][1] += max_x - min_x + 1  # blob width
        frame[1][2] += term_seg[7] - min_y + 1  # blob height

        # Sort segments based on their bases' vertical coordinate
        root_.sort( key = segment_sort_by_height )

        # For Recursion within a blob
        if      typ == 0 or type == 3:  match = M; alt_match = My
        else:                           match = My; alt_match = M
        ori_val = alt_match / match             # orientation value?
        blob_val = L + I + D + Dy + M + My
        rdn = blob_val /( alt_Der + alt_Dir + alt_Both )



        frame[2].append( ( ( s, L, I, D, Dy, M, My, alt_Der, alt_Dir, alt_Both ), ( min_x, max_x, min_y, term_seg[7] ), root_, xD ) )

    return frame  # no term_seg return: no root segs refer to it
    # ---------- form_blob() end ----------------------------------------------------------------------------------------


def image_to_blobs(image):
    " Main body of the operation, postfix '_' denotes array vs. element, prefix '_' denotes higher-line vs. lower-line variable "

    _P_ = [deque(), deque(), deque(), deque()]  # higher-line same- d-, m-, dy-, my- sign 1D patterns

    frame_params = [ 0, 0, 0, 0, 0, 0 ] # L, I, D, Dy, M, My
    # Each frame: [ [ summed_xD, summed_width, summed_heigh ], blob_, summed_params ]
    frame_ = [ [ frame_params, [ 0, 0, 0 ], [] ], [ frame_params, [ 0, 0, 0 ], [] ], \
               [ frame_params, [ 0, 0, 0 ], [] ], [ frame_params, [ 0, 0, 0 ], [] ] ]

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
        ders2__, _P_, frame_ = vertical_comp( ders1_, ders2__, _P_, frame_ )  # vertical pixel comparison

    # frame ends, last vertical rng of incomplete ders2__ is discarded,
    # merge segs of last line into their blobs:
    y = Y
    for typ in range(4):
        hP_ = _P_[typ]
        while hP_:
            hP, frame_[typ] = form_segment(hP_.popleft(), frame_[typ], typ )
            frame_[typ] = form_blob(hP, frame_[typ], typ )

    return frame_  # frame of 2D patterns, to be outputted to level 2
    # ---------- image_to_blobs() end -----------------------------------------------------------------------------------


# ************ MAIN FUNCTIONS END ***************************************************************************************


# ************ PROGRAM BODY *********************************************************************************************

# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:

rng = 2  # number of pixels compared to each pixel in four directions
ave = 15  # |d| value that coincides with average match: mP filter
ave_rate = 0.25  # not used; match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)
ini_y = 0  # not used

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
cv2.imwrite('./images/dblobs_horizontal.jpg', rebuild_blobs(frame_of_blobs[0]))
# cv2.imwrite('./images/mblobs_horizontal.jpg', rebuild_blobs(frame_of_blobs[1]))
# cv2.imwrite('./images/dblobs_vertical.jpg', rebuild_blobs(frame_of_blobs[2]))
# cv2.imwrite('./images/mblobs_vertical.jpg', rebuild_blobs(frame_of_blobs[3]))

# Check for redundant segments  --------------------------------------------------
print 'Searching for redundant segments...\n'
for frame in frame_of_blobs:
    for blob in frame[2]:
        for i, seg in enumerate(blob):
            for j, seg2 in enumerate(blob):
                if i != j and seg is seg2: print 'Redundant segment detected!\n'

# ************ PROGRAM BODY END ******************************************************************************************