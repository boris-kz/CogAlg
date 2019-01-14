import cv2
import argparse
import numpy as np
from scipy import misc
from time import time
from collections import deque
import math as math

'''   
    frame_blobs_grad() defines blobs by gradient, vs. dx and dy. I did that in frame_old, trying it again due to suggestion by Stephan Verbeeck
    gradient is estimated as hypot(dx, dy) of a quadrant with +dx and +dy, in vertical_comp before form_P call.
    Complemented by intra_blob (recursive search within blobs), it will be 2D version of first-level core algorithm.
    Blob is a contiguous area of positive or negative derivatives from cross-comparison among adjacent pixels within an image. 

    frame_blobs() performs several levels (Le) of encoding, incremental per scan line defined by vertical coordinate y.
    value of y per Le line is shown relative to y of current input line, incremented by top-down scan of input image:
    
    1Le, line y:    x_comp(p_): lateral pixel comparison -> tuple of derivatives der ) array der_
    2Le, line y- 1: y_comp(der_): vertical pixel comp -> 2D tuple der2 ) array der2_ 
    3Le, line y- 1+ rng*2: form_P(der2) -> 1D pattern P
    4Le, line y- 2+ rng*2: scan_P_(P, hP) -> hP, roots: down-connections, fork_: up-connections between Ps 
    5Le, line y- 3+ rng*2: form_segment(hP, seg) -> seg: merge vertically-connected _Ps in non-forking blob segments
    6Le, line y- 4+ rng*2+ seg depth: form_blob(seg, blob): merge connected segments in fork_' incomplete blobs, recursively  
    if y = rng * 2: line y == P_, line y-1 == hP_, line y-2 == seg_, line y-4 == blob_

    Pixel comparison in 2D forms lateral and vertical differences per pixel, combined into gradient. 
    They are formed on the same level because average lateral match ~ average vertical match.
    This part is not novel, I design from the scratch to make it organic part of hierarchical algorithm.
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
# ***********************************************************************************************************************
def rebuild_blobs(blob_, print_separate_blobs=0):
    " Rebuilt data of blobs into an image "
    blob_image = np.array([[127] * X] * Y)

    for index, blob in enumerate(blob_):  # Iterate through blobs
        if print_separate_blobs: blob_img = np.array([[127] * X] * Y)
        for seg in blob[3]:  # Iterate through segments
            y = seg[7]
            for (P, dx) in reversed(seg[5]):
                x = P[1]
                for i in range(P[2]):
                    blob_image[y, x] = 255 if P[0] else 0
                    if print_separate_blobs: blob_img[y, x] = 255 if P[0] else 0
                    x += 1
                y -= 1
        if print_separate_blobs:
            min_x, max_x, min_y, max_y = blob[1][:4]
            cv2.rectangle(blob_img, (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), (0, 255, 255), 1)
            cv2.imwrite('./images/raccoon_eye/blob%d.jpg' % (index), blob_img)

    return blob_image
    # ---------- rebuild_blobs() end ------------------------------------------------------------------------------------

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
    " Comparison over x coordinate, within rng of consecutive pixels on each line "

    dert1_ = []  # tuples of complete 1D derivatives: summation range = rng
    rng_dert1_ = deque(maxlen=rng)  # incomplete dert1s, within rng from input pixel: summation range < rng
    rng_dert1_.append((0, 0))

    for x, p in enumerate(pixel_):  # pixel p is compared to rng of prior pixels within horizontal line, summing d per prior pixel
        back_d = 0
        for index, (pri_p, d) in enumerate(rng_dert1_):
            id = p - pri_p
            d += id
            back_d += id
            if index < max_index:
                rng_dert1_[index] = (pri_p, d)
            elif x > min_coord:  # after pri_p comp over rng
                dert1_.append((pri_p, d))  # completed unilateral tuple is transferred from rng_dert_ to dert_

        rng_dert1_.appendleft((p, back_d))
    # last incomplete rng_dert1_ in line are discarded, vs. dert1_ += reversed(rng_dert1_)
    return dert1_
    # ---------- lateral_comp() end -------------------------------------------------------------------------------------


def vertical_comp(dert1_, rng_dert2__, _P_, frame):
    " Comparison to bilateral rng of vertically consecutive pixels, forming der2: pixel + lateral and vertical derivatives"

    P = [0, rng, 0, 0, 0, 0, 0, 0, []]  # lateral pattern = pri_s, x0, L, I, G, M, D, Dy, der2_
    P_ = deque()
    buff_ = deque()  # line y - 2 + rng*2: _Ps buffered by previous run of scan_P_
    new_rng_dert2__ = deque()  # 2D array: line of rng_dert2_s buffered for next-line comp
    x = rng  # lateral coordinate of pixel in input dert1

    for (p, d), rng_dert2_ in zip(dert1_, rng_dert2__):  # pixel comp to rng _pixels in rng_dert2_, summing dy
        back_dy = 0
        for index, (_p, _d, _dy) in enumerate(rng_dert2_):  # vertical derivatives are incomplete; prefix '_' denotes higher-line variable
            idy = p - _p
            _dy += idy
            back_dy += idy
            if index < max_index:
                rng_dert2_[index] = (_p, _d, _dy)
            elif y > min_coord:
                g = int(math.hypot(_dy, _d))  # no explicit angle, quadrant is indicated by signs of d and dy
                m = ave * rng - g  # match is defined as below-average gradient
                dert = _p, g, m, _d, _dy
                P = form_P(dert, x, X - rng - 1, P, P_, buff_, _P_, frame)

        rng_dert2_.appendleft((p, d, back_dy))  # new der2 displaces completed one in vertical rng_der2_ via maxlen
        new_rng_dert2__.append(rng_dert2_)  # 2D array of vertically-incomplete 2D tuples, converted to rng_der2__, for next-line vertical comp
        x += 1

    return new_rng_dert2__, P_
    # ---------- vertical_comp() end ------------------------------------------------------------------------------------


def form_P(dert, x, max_x, P, P_, buff_, hP_, frame):
    " Initializes, and accumulates 1D pattern "
    p, g, m, d, dy = dert  # 2D tuple of derivatives per pixel, "y" denotes vertical vs. lateral derivatives

    s = 1 if m > 0 else 0
    pri_s, x0 = P[:2]
    if not (s == pri_s or x == x0):  # P is terminated:
        if y == min_coord + 1:  # 1st line P_ is converted to init hP_;  scan_P_(), form_segment(), form_blob() use one type of Ps, hPs, buffs
            P_.append([P, 0, [], x - 1])  # P, roots, _fork_, x
        else:
            scan_P_(x - 1, P, P_, buff_, hP_, frame)  # P scans hP_
        P = s, x, 0, 0, 0, 0, 0, 0, []  # new P initialization

    pri_s, x0, L, I, G, M, D, Dy, dert_ = P  # continued or initialized input and derivatives are accumulated:
    L += 1  # length of a pattern
    I += p  # summed input
    G += g  # summed gradient
    M += m  # summed match
    D += d  # lateral D
    Dy += dy  # vertical D
    dert_.append(dert)  # der2s are buffered for oriented rescan and incremental range | derivation comp
    P = s, x0, L, I, G, M, D, Dy, dert_

    if x == max_x:  # P is terminated:
        if y == min_coord + 1:  # 1st line P_ is converted to init hP_;  scan_P_(), form_segment(), form_blob() use one type of Ps, hPs, buffs
            P_.append([P, 0, [], x])  # P, roots, _fork_, x
        else:
            scan_P_(x, P, P_, buff_, hP_, frame)  # P scans hP_
        P = s, x, 0, 0, 0, 0, 0, 0, []  # new P initialization

    return P  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_P() end -------------------------------------------------------------------------------------------


def scan_P_(x, P, P_, _buff_, hP_, frame):
    " P scans shared-x-coordinate hPs in higher P_, combines overlapping Ps into blobs "

    fork_ = []  # refs to hPs connected to input P
    _x0 = 0  # to start while loop, next ini_x = _x + 1
    x0 = P[1]

    while _x0 <= x:  # while x values overlap between P and _P
        if _buff_:
            hP = _buff_.popleft()  # hP was extended to segment and buffered in prior scan_P_
        elif hP_:
            hP = form_segment(hP_.popleft(), frame)
        else:
            break  # higher line ends, all hPs are converted to segments
        roots = hP[1]
        _x0 = hP[5][-1][0][1]  # first_x
        _x = _x0 + hP[5][-1][0][2] - 1  # last_x = first_x + L - 1

        if P[0] == hP[6][0][0] and not _x < x0 and not x < _x0:  # P comb -> blob if s == _s, _last_x >= first_x and last_x >= _first_x
            roots += 1
            hP[1] = roots
            fork_.append(hP)  # P-connected hPs will be converted to segments at each _fork

        if _x > x:  # x overlap between hP and next P: hP is buffered for next scan_P_, else hP included in a blob segment
            _buff_.append(hP)
        elif roots != 1:
            form_blob(hP, frame)  # segment is terminated and packed into its blob
        _x0 = _x + 1  # = first x of next _P

    P_.append([P, 0, fork_, x])  # P with no overlap to next _P is extended to hP and buffered for next-line scan_P_
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_segment(hP, frame):
    " Convert hP into new segment or add it to higher-line segment, merge blobs "
    _P, roots, fork_, last_x = hP
    [s, first_x], params = _P[:2], list(_P[2:8])
    ave_x = (_P[2] - 1) // 2  # extra-x L = L-1 (1x in L)
    params += [0, 0]  # add abs_D and abs_Dy

    if not fork_:  # seg is initialized with initialized blob (params, coordinates, remaining_roots, root_, xD)
        blob = [[s, 0, 0, 0, 0, 0, 0, 0, 0], [first_x, last_x, y - rng - 1, 0, 0], 1, []]
        hP = [params, roots, fork_, ave_x, 0, [(_P, 0)], blob]
        blob[3].append(hP)
    else:
        if len(fork_) == 1 and fork_[0][1] == 1:  # hP has one fork: hP[2][0], and that fork has one root: hP
            # hP is merged into higher-line blob segment (Pars, roots, _fork_, ave_x, xD, Py_, blob) at hP[2][0]:
            fork = fork_[0]
            L, I, G, M, D, Dy, abs_D, abs_Dy = params
            Ls, Is, Gs, Ms, Ds, Dys, abs_Ds, abs_Dys = fork[0]  # seg params
            fork[0] = [Ls + L, Is + I, Gs + G, Ms + M, Ds + D, Dys + Dy, abs_Ds + abs(D), abs_Dys + abs(Dy)]
            fork[1] = roots
            dx = ave_x - fork[3]
            fork[3] = ave_x
            fork[4] += dx  # xD for seg normalization and orientation, or += |dx| for curved yL and Pm estimation?
            fork[5].append((_P, dx))  # Py_: vertical buffer of Ps merged into seg
            hP = fork  # replace segment with including fork's segment
            blob = hP[6]

        else:  # if >1 forks, or 1 fork that has >1 roots:
            hP = [params, roots, fork_, ave_x, 0, [(_P, 0)], fork_[0][6]]  # seg is initialized with fork's blob
            blob = hP[6]
            blob[3].append(hP)  # segment is buffered into root_

            if len(fork_) > 1:  # merge blobs of all forks
                if fork_[0][1] == 1:  # if roots == 1
                    form_blob(fork_[0], frame, 1)  # merge seg of 1st fork into its blob

                for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                    if fork[1] == 1:
                        form_blob(fork, frame, 1)

                    if not fork[6] is blob:
                        [s, L, I, G, M, D, Dy, abs_D, abs_Dy], [min_x, max_x, min_y, xD, Ly], remaining_roots, root_ = fork[6]
                        blob[0][1] += L
                        blob[0][2] += I
                        blob[0][3] += G
                        blob[0][4] += M
                        blob[0][5] += D
                        blob[0][6] += Dy
                        blob[0][7] += abs_D
                        blob[0][8] += abs_Dy
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
    return hP
    # ---------- form_segment() end -----------------------------------------------------------------------------------------


def form_blob(term_seg, frame, y_carry=0):
    " Terminated segment is merged into continued or initialized blob (all connected segments) "

    [L, I, G, M, D, Dy, abs_D, abs_Dy], roots, fork_, x, xD, Py_, blob = term_seg  # unique blob in fork_[0][6] is ref'd by other forks
    blob[0][1] += L
    blob[0][2] += I
    blob[0][3] += G
    blob[0][4] += M
    blob[0][5] += D
    blob[0][6] += Dy
    blob[0][7] += abs_D
    blob[0][8] += abs_Dy
    blob[1][3] += xD  # ave_x angle, to evaluate blob for re-orientation
    blob[1][4] += len(Py_)  # Ly = number of slices in segment
    blob[2] += roots - 1  # reference to term_seg is already in blob[9]
    term_seg.append(y - rng - 1 - y_carry)  # y_carry: min elevation of term_seg over current hP

    if not blob[2]:  # if remaining_roots == 0: blob is terminated and packed in frame
        [s, L, I, G, M, D, Dy, abs_D, abs_Dy], [min_x, max_x, min_y, xD, Ly], remaining_roots, root_ = blob
        # frame P are to compute averages, redundant for same-scope alt_frames
        frame[0] += I
        frame[1] += M
        frame[2] += M
        frame[3] += D
        frame[4] += Dy
        frame[5] += abs_D
        frame[6] += abs_Dy
        frame[7] += xD  # ave_x angle, to evaluate frame for re-orientation
        frame[8] += Ly  # +L
        frame[9].append(blob)
    # ---------- form_blob() end ----------------------------------------------------------------------------------------

def image_to_blobs(image):
    " Main body of the operation, postfix '_' denotes array vs. element, prefix '_' denotes higher-line vs. lower-line variable "

    _P_ = deque()  # higher-line same-m-sign 1D patterns
    frame = [0, 0, 0, 0, 0, 0, 0, 0, 0, []]
    global y
    y = 0
    rng_dert2__ = []  # horizontal line of vertical buffers: 2D array of 2D tuples, deque for speed?
    pixel_ = image[0, :]  # first line of pixels
    dert1_ = lateral_comp(pixel_)

    for (p, d) in dert1_:
        dert2 = p, d, 0  # only dy is initialized at 0, m is added after dert is complete
        rng_dert2_ = deque(maxlen=rng)  # vertical buffer of incomplete derivatives tuples, for fuzzy ycomp
        rng_dert2_.append(dert2)  # only one tuple in first-line rng_der2_
        rng_dert2__.append(rng_dert2_)

    for y in range(1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?

        pixel_ = image[y, :]  # vertical coordinate y is index of new line p_
        der1_ = lateral_comp(pixel_)  # lateral pixel comparison
        rng_der2__, _P_ = vertical_comp(der1_, rng_dert2__, _P_, frame)  # vertical pixel comparison

    # frame ends, last vertical rng of incomplete rng_der2__ is discarded,
    # merge segs of last line into their blobs:
    y = Y
    while _P_:  form_blob( form_segment(_P_.popleft(), frame), frame)
    return frame  # frame of 2D patterns, to be outputted to level 2
    # ---------- image_to_blobs() end -----------------------------------------------------------------------------------

# ************ MAIN FUNCTIONS END ***************************************************************************************


# ************ PROGRAM BODY *********************************************************************************************

# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:

rng = 1  # number of pixels compared to each pixel in four directions
max_index = rng - 1  # max index of rng_dert1_ and rng_dert2_
min_coord = rng * 2 - 1  # min x and y for form_P input: der2 from comp over rng
ave = 15  # |d| value that coincides with average match: mP filter

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
cv2.imwrite('./images/output.jpg', rebuild_blobs(frame_of_blobs[9]))
# Check for redundant segments  --------------------------------------------------
print: 'Searching for redundant segments...\n'
for blob in frame_of_blobs[9]:
    for i, seg in enumerate(blob[3]):
        for j, seg2 in enumerate(blob):
            if i != j and seg is seg2: print: 'Redundant segment detected!\n'

# ************ PROGRAM BODY END ******************************************************************************************import cv2
