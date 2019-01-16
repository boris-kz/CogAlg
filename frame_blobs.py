import cv2
import argparse
from time import time
from collections import deque
import math as math
from misc import draw_blobs
'''   
    frame_blobs() defines blobs: contiguous areas of positive or negative deviation of gradient. 
    Gradient is estimated as hypot(dx, dy) of a quadrant with +dx and +dy, from cross-comparison among adjacent pixels.
    Complemented by intra_blob (recursive search within blobs), it will be 2D version of first-level core algorithm.

    frame_blobs() performs several levels (Le) of encoding, incremental per scan line defined by vertical coordinate y.
    value of y per Le line is shown relative to y of current input line, incremented by top-down scan of input image:

    1Le, line y:                            x_comp(p_): lateral pixel comparison -> tuple of derivatives dert1 ) array dert1_
    2Le, line y - rng to y - 1:             y_comp(dert1_): vertical pixel comp -> 2D tuple dert2 ) array dert2_ 
    3Le, line y - rng:                      form_P(dert2) -> 1D pattern P
    4Le, line y - rng - 1:                  scan_P_(P, hP) -> hP, roots: down-connections, fork_: up-connections between Ps 
    5Le, line y - rng - 1:                  form_segment(hP, seg) -> seg: merge vertically-connected _Ps in non-forking blob segments
    6Le, line y - rng - 2 to y - rng - 1:   form_blob(seg, blob): merge connected segments in fork_' incomplete blobs, recursively  

    prefix '_' denotes higher-line variable or pattern, vs. same-type lower-line variable or pattern,
    postfix '_' denotes array name, vs. same-name elements of that array:
    if y = rng * 2: line y == P_, line y-1 == hP_, line y-2 == seg_, line y-4 == blob_

    Initial pixel comparison is not novel, I design from the scratch to make it organic part of hierarchical algorithm.
    It would be much faster with matrix computation, but this is minor compared to higher-level processing.
    I implement it sequentially for consistency with accumulation into blobs: irregular and very difficult to map to matrices.

    All 2D functions (y_comp, scan_P_, form_segment, form_blob) input two lines: higher and lower, 
    convert elements of lower line into elements of new higher line, then displace elements of old higher line into higher function.
    Higher-line elements include additional variables, derived while they were lower-line elements.
'''
# ************ MAIN FUNCTIONS *******************************************************************************************
# -pixel_comp()
# -form_P()
# -scan_P_()
# -form_segment()
# -form_blob()
# -image_to_blobs()
# ***********************************************************************************************************************
def pixel_comp(pixel_, _pixel_, _P_, frame):
    " Comparison of consecutive pixels to computes gradien "
    P = [-1, [1, -1], [0, 0, 0, 0, 0], []]    # s, [min_x, max_x], [L, I, G, Dx, Dy], dert_
    P_ = deque()
    buff_ = deque()
    _p = pixel_[0]; x = 1
    for p, __p in zip(pixel_[1:], _pixel_[1:]):  # pixel p is compared to prior pixels vertically and horizontally
        dx = p - _p
        dy = p - __p
        g = int(math.hypot(dy, dx)) - ave
        dert = p, g, dx, dy
        P = form_P(dert, x, X - 1, P, P_, buff_, _P_, frame)
        _p = p; x += 1
    return P_
    # ---------- pixel_comp() end ---------------------------------------------------------------------------------------
def form_P(dert, x, x_stop, P, P_, buff_, hP_, frame):
    " Initializes, and accumulates 1D pattern "
    p, g, dx, dy = dert  # 2D tuple of derivatives per pixel, "y" denotes vertical vs. lateral derivatives
    s = 1 if g > 0 else 0
    pri_s = P[0]
    if s != pri_s and pri_s != -1:  # P is terminated:
        P[1][1] = x - 1  # P's max_x
        scan_P_(P, P_, buff_, hP_, frame)  # P scans hP_
        P = s, [x, -1], [0, 0, 0, 0, 0], []  # new P initialization
    [min_x, max_x], [L, I, G, Dx, Dy], dert_ = P[1:]  # continued or initialized input and derivatives are accumulated:
    L += 1  # length of a pattern
    I += p  # summed input
    G += g  # summed gradient
    Dx += dx  # lateral D
    Dy += dy  # vertical D
    dert_.append(dert)  # der2s are buffered for oriented rescan and incremental range | derivation comp
    P = s, [min_x, max_x], [L, I, G, Dx, Dy], dert_

    if x == x_stop:  # P is terminated:
        P[1][1] = x  # P's max_x
        scan_P_(P, P_, buff_, hP_, frame)  # P scans hP_
        P = -1, [1, -1], [0, 0, 0, 0, 0], []  # new-line P initialization

    return P  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_P() end -------------------------------------------------------------------------------------------
def scan_P_(P, P_, _buff_, hP_, frame):
    " P scans shared-x-coordinate hPs in higher P_, combines overlapping Ps into blobs "
    fork_ = []  # refs to hPs connected to input P
    _min_x = 0  # to start while loop, next ini_x = _x + 1
    min_x, max_x = P[1]
    while _min_x <= max_x:  # while x values overlap between P and _P
        if _buff_:
            hP = _buff_.popleft()  # hP was extended to segment and buffered in prior scan_P_
        elif hP_:
            hP = form_segment(hP_.popleft(), frame)
        else:
            break  # higher line ends, all hPs are converted to segments
        roots = hP[4]
        _P = hP[3][-1][0]
        _min_x, _max_x = _P[1]  # first_x, last_x

        if P[0] == _P[0] and min_x <= _max_x and _min_x <= max_x:
            roots += 1
            hP[4] = roots
            fork_.append(hP)  # P-connected hPs will be converted to segments at each _fork

        if _max_x > max_x:  # x overlap between hP and next P: hP is buffered for next scan_P_, else hP included in a blob segment
            _buff_.append(hP)
        elif roots != 1:
            form_blob(hP, frame)  # segment is terminated and packed into its blob
        _min_x = _max_x + 1  # = first x of next _P

    P_.append((P, fork_))  # P with no overlap to next _P is extended to hP and buffered for next-line scan_P_
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_segment(hP, frame):
    " Convert hP into new segment or add it to higher-line segment, merge blobs "
    _P, fork_ = hP
    s, [min_x, max_x], params = _P[:-1]
    ave_x = (params[0] - 1) // 2  # extra-x L = L-1 (1x in L)

    if not fork_:  # seg is initialized with initialized blob (params, coordinates, remaining_roots, root_, xD)
        blob = [s, [min_x, max_x, y - 1, -1, 0, 0, 0], [0, 0, 0, 0, 0], [], 1]  # s, coords, params, root_, remaining_roots
        hP = [s, [min_x, max_x, y - 1, -1, 0, 0, ave_x], params, [(_P, 0)], 0, fork_, blob]
        blob[3].append(hP)
    else:
        if len(fork_) == 1 and fork_[0][4] == 1:  # hP has one fork: hP[2][0], and that fork has one root: hP
            # hP is merged into higher-line blob segment (Pars, roots, _fork_, ave_x, xD, Py_, blob) at hP[2][0]:
            fork = fork_[0]
            fork[1][0] = min(fork[1][0], min_x)
            fork[1][1] = max(fork[1][1], max_x)
            xd = ave_x - fork[1][5]
            fork[1][4] += xd
            fork[1][5] += abs(xd)
            fork[1][6] = ave_x
            L, I, G, Dx, Dy = params
            Ls, Is, Gs, Dxs, Dys = fork[2]  # seg params
            fork[2] = [Ls + L, Is + I, Gs + G, Dxs + Dx, Dys + Dy]
            fork[3].append((_P, xd))  # Py_: vertical buffer of Ps merged into seg
            fork[4] = 0  # reset roots
            hP = fork  # replace segment with including fork's segment
            blob = hP[6]

        else:  # if >1 forks, or 1 fork that has >1 roots:
            hP = [s, [min_x, max_x, y - 1, -1, 0, 0, ave_x], params, [(_P, 0)], 0, fork_, fork_[0][6]]  # seg is initialized with fork's blob
            blob = hP[6]
            blob[3].append(hP)  # segment is buffered into root_

            if len(fork_) > 1:  # merge blobs of all forks
                if fork_[0][4] == 1:  # if roots == 1
                    form_blob(fork_[0], frame, 1)  # merge seg of 1st fork into its blob

                for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                    if fork[4] == 1:
                        form_blob(fork, frame, 1)

                    if not fork[6] is blob:
                        [min_x, max_x, min_y, max_y, xD, abs_xD, Ly], [L, I, G, Dx, Dy], root_, remaining_roots = fork[6][1:]  # ommit sign
                        blob[1][0] = min(min_x, blob[1][0])
                        blob[1][1] = max(max_x, blob[1][1])
                        blob[1][2] = min(min_y, blob[1][2])
                        blob[1][4] += xD
                        blob[1][5] += abs_xD
                        blob[1][6] += Ly
                        blob[2][0] += L
                        blob[2][1] += I
                        blob[2][2] += G
                        blob[2][3] += Dx
                        blob[2][4] += Dy
                        blob[4] += remaining_roots
                        for seg in root_:
                            if not seg is fork:
                                seg[6] = blob  # blobs in other forks are references to blob in the first fork
                                blob[3].append(seg)  # buffer of merged root segments
                        fork[6] = blob
                        blob[3].append(fork)
                    blob[4] -= 1

        blob[1][0] = min(min_x, blob[1][0])
        blob[1][1] = max(max_x, blob[1][1])
    return hP
    # ---------- form_segment() end -----------------------------------------------------------------------------------------
def form_blob(term_seg, frame, y_carry=0):
    " Terminated segment is merged into continued or initialized blob (all connected segments) "
    [min_x, max_x, min_y, max_y, xD, abs_xD, ave_x], [L, I, G, Dx, Dy], Py_, roots, fork_, blob = term_seg[1:]  # ignore sign
    blob[1][4] += xD  # ave_x angle, to evaluate blob for re-orientation
    blob[1][5] += len(Py_)  # Ly = number of slices in segment
    blob[2][0] += L
    blob[2][1] += I
    blob[2][2] += G
    blob[2][3] += Dx
    blob[2][4] += Dy
    blob[4] += roots - 1  # reference to term_seg is already in blob[9]
    term_seg[1][3] = y - 1 - y_carry  # y_carry: min elevation of term_seg over current hP

    if not blob[4]:  # if remaining_roots == 0: blob is terminated and packed in frame
        blob[1][3] = term_seg[1][3]
        [min_x, max_x, min_y, max_y, xD, abs_xD, Ly], [L, I, G, Dx, Dy], root_, remaining_roots = blob[1:]  # ignore sign
        # frame P are to compute averages, redundant for same-scope alt_frames
        frame[0] += I
        frame[1] += G
        frame[2] += Dx
        frame[3] += Dy
        frame[4] += xD  # ave_x angle, to evaluate frame for re-orientation
        frame[5] += abs_xD
        frame[6] += Ly
        blob[3] = sorted(root_, key=lambda segment: segment[1][2])    # sorted by min_y
        frame[7].append(blob)
    # ---------- form_blob() end ----------------------------------------------------------------------------------------
def image_to_blobs(image):
    " Main body of the operation, postfix '_' denotes array vs. element, prefix '_' denotes higher-line vs. lower-line variable "
    _P_ = deque()  # higher-line same-m-sign 1D patterns
    frame = [0, 0, 0, 0, 0, 0, 0, []]
    global y
    y = 0
    _pixel_ = image[0, :]  # first line of pixels
    for y in range(1, Y):  # or Y-1: default term_blob in scan_P_ at y = Y?
        pixel_ = image[y, :]  # vertical coordinate y is index of new line p_
        _P_ = pixel_comp(pixel_, _pixel_, _P_, frame)  # vertical and lateral pixel comparison
        _pixel_ = pixel_
    # frame ends, merge segs of last line into their blobs:
    y = Y
    while _P_:  form_blob(form_segment(_P_.popleft(), frame), frame)
    return frame  # frame of 2D patterns, to be outputted to level 2
    # ---------- image_to_blobs() end -----------------------------------------------------------------------------------


# ************ MAIN FUNCTIONS END ***************************************************************************************

# ************ PROGRAM BODY *********************************************************************************************

# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:
ave = 15  # P-defining filter: value of gradient that coincides with average higher-level match
# Load inputs --------------------------------------------------------------------
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
draw_blobs('./debug', frame_of_blobs[7], (Y, X), debug=0)
# ************ PROGRAM BODY END ******************************************************************************************