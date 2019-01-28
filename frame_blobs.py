import cv2
import argparse
from time import time
from collections import deque
import math
import numpy as np

'''   
    frame_blobs() defines blobs: contiguous areas of positive or negative deviation of maximal gradient. 
    Gradient is estimated as hypot(dx, dy) of a quadrant with +dx and +dy, from cross-comparison among adjacent pixels.
    Complemented by intra_blob (recursive search within blobs), it will be 2D version of first-level core algorithm.
    
    frame_blobs() performs several levels (Le) of encoding, incremental per scan line defined by vertical coordinate y.
    value of y per Le line is shown relative to y of current input line, incremented by top-down scan of input image:
    
    1Le, line y:    x_comp(p_): lateral pixel comparison -> tuple of derivatives dert1 ) array dert1_
    2Le, line y- 1: y_comp(dert1_): vertical pixel comp -> 2D tuple dert2 ) array dert2_ 
    3Le, line y- 1+ rng: form_P(dert2) -> 1D pattern P
    4Le, line y- 2+ rng: scan_P_(P, hP) -> hP, roots: down-connections, fork_: up-connections between Ps 
    5Le, line y- 3+ rng: form_segment(hP, seg) -> seg: merge vertically-connected _Ps in non-forking blob segments
    6Le, line y- 4+ rng+ seg depth: form_blob(seg, blob): merge connected segments in fork_' incomplete blobs, recursively  
    
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
# -comp_pixel()
# -form_P()
# -scan_P_()
# -form_segment()
# -form_blob()
# -image_to_blobs()
# ***********************************************************************************************************************

def comp_pixel(_P_, frame):
    " Comparison of consecutive pixels to compute gradient "

    p__, [d__, [g__]] = frame[2]
    p_, lower_p_ = p__[y: y + 2]
    g_ = g__[y]
    d_ = d__[y]

    P_ = deque()
    buff_ = deque()
    p = p_[0]  # evaluated pixel
    x = 0
    P = [-1, [0, -1], [0, 0, 0, 0, 0]]  # s, [x_start, x_end], [L, I, G, Dx, Dy]
    
    for right_p, lower_p in zip(p_[1:], lower_p_[:-1]):  # pixel p is compared to vertically and horizontally subsequent pixels
        dy = lower_p - p    # compare with lower pixel
        dx = right_p - p    # compare with right-side pixel
        g = int(math.hypot(dy, dx)) - ave  # max gradient of right_and_down quadrant, unique for pixel p
        g_[x] = g           # g buffered in g__ per blob
        d_[x] = dy, dx      # d buffered in d__ per blob

        # Call form_P()
        dert = p, g, dx, dy
        P = form_P(dert, x, X - 2, P, P_, buff_, _P_, frame)

        p = right_p
        x += 1

    return P_
    # ---------- comp_pixel() end ---------------------------------------------------------------------------------------

def form_P(dert, x, x_stop, P, P_, buff_, hP_, frame):
    " Initializes, and accumulates 1D pattern "
    p, g, dx, dy = dert  # 2D tuple of derivatives per pixel
    s = 1 if g > 0 else 0
    pri_s = P[0]

    if s != pri_s and pri_s != -1:  # P is terminated:
        P[1][1] = x  # P's x_end
        scan_P_(P, P_, buff_, hP_, frame)  # P scans hP_
        P = [s, [x, -1], [0, 0, 0, 0, 0]]  # new P initialization

    boundaries, [L, I, G, Dx, Dy] = P[1:]  # continued or initialized input and derivatives are accumulated:
    L += 1  # length of a pattern
    I += p  # summed input
    G += g  # summed gradient
    Dx += dx  # lateral D
    Dy += dy  # vertical D
    # dert_ is avaiable through frame
    P = [s, boundaries, [L, I, G, Dx, Dy]]  # boundaries = [x_start, x_end]

    if x == x_stop:  # P is terminated:
        P[1][1] = x + 1  # P's x_end
        scan_P_(P, P_, buff_, hP_, frame)  # P scans hP_
    return P  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_P() end -------------------------------------------------------------------------------------------

def scan_P_(P, P_, _buff_, hP_, frame):
    " P scans shared-x-coordinate hPs in higher P_, combining overlapping Ps into blobs "
    fork_ = []  # refs to hPs connected to input P
    _x_start = 0  # to start while loop, next ini_x = _x + 1
    x_start, x_end = P[1]

    while _x_start < x_end:  # while x values overlap between P and _P
        if _buff_:
            hP = _buff_.popleft()  # hP was extended to segment and buffered in prior scan_P_
        elif hP_:
            hP = form_segment(hP_.popleft(), frame)
        else:
            break  # higher line ends, all hPs are converted to segments
        roots = hP[4]
        _P = hP[3][-1][0]
        _x_start, _x_end = _P[1]  # first_x, last_x

        if P[0] == _P[0] and x_start < _x_end and _x_start < x_end:
            roots += 1
            hP[4] = roots
            fork_.append(hP)  # P-connected hPs will be converted to segments at each _fork
        if _x_end > x_end:  # x overlap between hP and next P: hP is buffered for next scan_P_, else hP included in a blob segment
            _buff_.append(hP)
        elif roots != 1:
            form_blob(hP, frame)  # segment is terminated and packed into its blob
        _x_start = _x_end   # = first x of next _P

    P_.append((P, fork_))  # P with no overlap to next _P is extended to hP and buffered for next-line scan_P_
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------

def form_segment(hP, frame):
    " Convert hP into new segment or add it to higher-line segment, merge blobs "
    _P, fork_ = hP
    s, [x_start, x_end], params = _P
    ave_x = (params[0] - 1) // 2  # extra-x L = L-1 (1x in L)

    if not fork_:  # seg is initialized with initialized blob (params, coordinates, incomplete_segments, root_, xD)
        blob = [s, [x_start, x_end, y - 1, -1, 0, 0, 0], [0, 0, 0, 0, 0], [], 1]  # s, coords, params, root_, incomplete_segments
        hP = [s, [x_start, x_end, y - 1, -1, 0, 0, ave_x], params, [(_P, 0)], 0, fork_, blob]
        blob[3].append(hP)
    else:
        if len(fork_) == 1 and fork_[0][4] == 1:  # hP has one fork: hP[2][0], and that fork has one root: hP
            # hP is merged into higher-line blob segment (Pars, roots, _fork_, ave_x, xD, Py_, blob) at hP[2][0]:
            fork = fork_[0]
            fork[1][0] = min(fork[1][0], x_start)
            fork[1][1] = max(fork[1][1], x_end)
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
            hP = [s, [x_start, x_end, y - 1, -1, 0, 0, ave_x], params, [(_P, 0)], 0, fork_, fork_[0][6]]  # seg is initialized with fork's blob
            blob = hP[6]
            blob[3].append(hP)  # segment is buffered into root_
            if len(fork_) > 1:  # merge blobs of all forks
                if fork_[0][4] == 1:  # if roots == 1
                    form_blob(fork_[0], frame, 1)  # merge seg of 1st fork into its blob

                for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                    if fork[4] == 1:
                        form_blob(fork, frame, 1)
                    if not fork[6] is blob:
                        [x_start, x_end, y_start, y_end, xD, abs_xD, Ly], [L, I, G, Dx, Dy], root_, open_segments = fork[6][1:]  # ommit sign
                        blob[1][0] = min(x_start, blob[1][0])
                        blob[1][1] = max(x_end, blob[1][1])
                        blob[1][2] = min(y_start, blob[1][2])
                        blob[1][4] += xD
                        blob[1][5] += abs_xD
                        blob[1][6] += Ly
                        blob[2][0] += L
                        blob[2][1] += I
                        blob[2][2] += G
                        blob[2][3] += Dx
                        blob[2][4] += Dy
                        blob[4] += open_segments
                        for seg in root_:
                            if not seg is fork:
                                seg[6] = blob  # blobs in other forks are references to blob in the first fork
                                blob[3].append(seg)  # buffer of merged root segments
                        fork[6] = blob
                        blob[3].append(fork)
                    blob[4] -= 1
        blob[1][0] = min(x_start, blob[1][0])
        blob[1][1] = max(x_end, blob[1][1])
    return hP
    # ---------- form_segment() end -----------------------------------------------------------------------------------------

def form_blob(term_seg, frame, y_carry=0):
    " Terminated segment is merged into continued or initialized blob (all connected segments) "
    [x_start, x_end, y_start, y_end, xD, abs_xD, ave_x], [L, I, G, Dx, Dy], Py_, roots, fork_, blob = term_seg[1:]  # ignore sign

    blob[1][4] += xD  # ave_x angle, to evaluate blob for re-orientation
    blob[1][5] += len(Py_)  # Ly = number of slices in segment
    blob[2][0] += L
    blob[2][1] += I
    blob[2][2] += G
    blob[2][3] += Dx
    blob[2][4] += Dy
    blob[4] += roots - 1  # reference to term_seg is already in blob[9]
    term_seg[1][3] = y - y_carry  # y_carry: min elevation of term_seg over current hP. - 1 due to higher-line P

    if not blob[4]:  # if incomplete_segments == 0: blob is terminated and packed in frame
        blob[1][3] = term_seg[1][3]
        [x_start, x_end, y_start, y_end, xD, abs_xD, Ly], [L, I, G, Dx, Dy], root_, incomplete_segments = blob[1:]  # ignore sign
        # frame P are to compute averages, redundant for same-scope alt_frames
        frame[0][0] += I
        frame[0][1] += G
        frame[0][2] += Dx
        frame[0][3] += Dy
        frame[0][4] += xD  # ave_x angle, to evaluate frame for re-orientation
        frame[0][5] += abs_xD
        frame[0][6] += Ly
        frame[1].append(blob)
    # ---------- form_blob() end ----------------------------------------------------------------------------------------

def image_to_blobs(image):
    " root function, postfix '_' denotes array vs. element, prefix '_' denotes higher-line vs. lower-line variable "
    _P_ = deque()   # higher-line same-m-sign 1D patterns
    g__ = np.zeros((Y, X), dtype=int)
    d__ = np.zeros((Y, X, 2), dtype=int)
    frame = [[0, 0, 0, 0, 0, 0, 0], [], [image, [d__, [g__]]]]   # params, blob_, dert_tree
    global y
    for y in range(Y - 1):  # or Y-1: default term_blob in scan_P_ at y = Y?
        _P_ = comp_pixel(_P_, frame)  # vertical and lateral pixel comparison

    # frame ends, merge segs of last line into their blobs:
    y = Y - 1
    while _P_:  form_blob(form_segment(_P_.popleft(), frame), frame)
    return frame  # frame of 2D patterns, to be outputted to level 2
    # ---------- image_to_blobs() end -----------------------------------------------------------------------------------

# ************ MAIN FUNCTIONS END ***************************************************************************************

# ************ PROGRAM BODY *********************************************************************************************
# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:
from misc import get_filters
get_filters(globals())          # imports all filters at once

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
from misc import draw_blobs
# draw_blobs('./debug', frame_of_blobs[1], (Y, X), out_ablob=0, debug=0, show=1)
# ************ PROGRAM BODY END ******************************************************************************************
