import cv2
import argparse
from time import time
from collections import deque
import math
import numpy as np
import Classes

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

    p__, g__, d__ = frame.dert_map_
    p_, lower_p_ = p__[y: y + 2]
    g_ = g__[y]
    d_ = d__[y]

    P_ = deque()
    buff_ = deque()
    p = p_[0]  # evaluated pixel
    x = 0
    P = Classes.P(y)
    
    for right_p, lower_p in zip(p_[1:], lower_p_[:-1]):  # pixel p is compared to vertically and horizontally subsequent pixels
        dy = lower_p - p    # compare with lower pixel
        dx = right_p - p    # compare with right-side pixel
        g = int(math.hypot(dy, dx)) - ave  # max gradient of right_and_down quadrant, unique for pixel p
        g_[x] = g           # g buffered in g__
        d_[x] = dy, dx      # d buffered in d__

        # Call form_P()
        dert = p, g, dx, dy
        s = g > 0
        P = form_P(s, dert, x, P, P_, buff_, _P_, frame)

        p = right_p
        x += 1
    # terminate last P:
    P.terminate(x)  # P's x_end
    scan_P_(P, P_, buff_, _P_, frame)  # P scans hP_

    return P_
    # ---------- comp_pixel() end ---------------------------------------------------------------------------------------

def form_P(s, dert, x, P, P_, buff_, hP_, frame):
    " Initializes, and accumulates 1D pattern "
    pri_s = P.sign

    if s != pri_s and pri_s != -1:  # P is terminated:
        P.terminate(x)  # P's x_end
        scan_P_(P, P_, buff_, hP_, frame)  # P scans hP_
        P = Classes.P(y, x_start=x, sign=s)  # new P initialization

    if pri_s == -1: P.sign = s
    P.accum_params((1,) + dert)  # continued or initialized input and derivatives are accumulated
    # dert_ is avaiable through blob's dert__
    return P  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_P() end -------------------------------------------------------------------------------------------

def scan_P_(P, P_, _buff_, hP_, frame):
    " P scans shared-x-coordinate hPs in higher P_, combining overlapping Ps into blobs "
    fork_ = []  # refs to hPs connected to input P
    _x_start = 0  # to start while loop, next ini_x = _x + 1
    x_start, x_end = P.boundaries[:2]  # exclude y

    while _x_start < x_end:  # while x values overlap between P and _P
        if _buff_:
            seg = _buff_.popleft()  # hP was extended to segment and buffered in prior scan_P_
        elif hP_:
            seg = form_segment(hP_.popleft(), frame)
        else:
            break  # higher line ends, all hPs are converted to segments
        _P, xd = seg.Py_[-1]         # previous line P
        _x_start, _x_end = _P.boundaries[:2]    # first_x, last_x

        if P.sign == _P.sign and x_start < _x_end and _x_start < x_end:
            seg.roots += 1
            fork_.append(seg)  # P-connected hPs will be converted to segments at each _fork
        if _x_end > x_end:  # x overlap between hP and next P: hP is buffered for next scan_P_, else hP included in a blob segment
            _buff_.append(seg)
        elif seg.roots != 1:
            form_blob(seg, frame)  # segment is terminated and packed into its blob
        _x_start = _x_end   # = first x of next _P

    P_.append((P, fork_))  # P with no overlap to next _P is extended to hP and buffered for next-line scan_P_
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------

def form_segment(hP, frame):
    " Convert hP into new segment or add it to higher-line segment, merge blobs "
    _P, fork_ = hP
    if not fork_:
        # Case 1: no higher-line connection:
        # segment is initialized with initialized blob:
        seg = Classes.segment(_P, fork_)    # initialize segment with _P and fork_
        blob = Classes.blob(seg)        # initialize blob with segment
    else:
        if len(fork_) == 1 and fork_[0].roots == 1:
            # Case 2: single connection with higher-line segment, which has single lower-line connection (roots == 1):
            # hP is merged into higher-line connected segment:
            seg = fork_[0]  # the only fork
            seg.accum_P(_P) # merge _P into seg
        else:
            # Case 3: the remaining scenarios are considered here:
            # All of them should include initializing new segment with shared higher-line-connected-segment's blob:
            seg = Classes.segment(_P, fork_)        # seg is initialized
            blob = fork_[0].blob                    # choose first fork's blob
            blob.accum_segment(seg)                 # seg is added to fork's blob
            blob.extend_boundaries(_P.boundaries[:2])   # extend x-boundaries
            # If more than 1 fork: their blobs are joined through seg:
            # Try to merge them:
            if len(fork_) > 1:  # merge blobs of all forks
                # terminate all fork in fork_ if hasn't already done so: (roots == 1 is the only case where a segment could pass termination check)
                if fork_[0].roots == 1:  # if roots == 1
                    form_blob(fork_[0], frame, y_carry=1)   # last arguments is for vertical coordinate precision

                for fork in fork_[1:len(fork_)]:
                    if fork.roots == 1:
                        form_blob(fork, frame, y_carry=1)
                    # merge blobs of other forks into blob of 1st fork
                    blob.merge(fork.blob)
    return seg
    # ---------- form_segment() end -----------------------------------------------------------------------------------------

def form_blob(term_seg, frame, y_carry=0):
    " Terminate segments and blobs (if no segment left) "
    blob = term_seg.blob
    blob.term_segment(term_seg, y - y_carry) # y_carry: min elevation of term_seg over current hP

    # Check for blob termination:
    if not blob.open_segments:  # if open_segments == 0: blob is terminated and packed in frame
        blob.terminate(term_seg.y_end()).localize(frame)
        # frame P are to compute averages, redundant for same-scope alt_frames
        frame.accum_params(blob.params[1:] + blob.orientation_params)   # exclude L: blob.params = [L, I, G, Dx, Dy]; blob.orientation_params = [xD, abs_xD, Ly]
        frame.blob_.append(blob)
    # ---------- form_blob() end ----------------------------------------------------------------------------------------

def image_to_blobs(image):
    " root function, postfix '_' denotes array vs. element, prefix '_' denotes higher-line vs. lower-line variable "
    # initialize frame:
    g__ = np.zeros((Y, X), dtype=int)
    d__ = np.zeros((Y, X, 2), dtype=int)
    frame = Classes.frame(7)    # number of frame params = 7
    frame.dert_map_ = [image, g__, d__]
    frame.blob_ = []

    _P_ = deque()  # higher-line same-m-sign 1D patterns
    global y
    for y in range(Y - 1):
        _P_ = comp_pixel(_P_, frame)  # vertical and lateral pixel comparison

    # frame ends, merge segs of last line into their blobs:
    y = Y - 1
    while _P_:  form_blob(form_segment(_P_.popleft(), frame), frame)
    del frame.dert_map_
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
from DEBUG import DEBUG
DEBUG('./debug', frame_of_blobs.blob_, (Y, X), debug_ablob=0, debug_parts=0, debug_local=1, show=0)
# ************ PROGRAM BODY END ******************************************************************************************
