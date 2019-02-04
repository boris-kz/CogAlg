import cv2
import argparse
from time import time
from collections import deque
import math
import numpy as np
from frame_2D_alg.frame_main import Classes

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

def comp_pixel(p_, lower_p_, _P_, frame):
    " cross-compare adjacent pixels, compute gradient "
    dert_ = frame.dert__[y]
    P_ = deque()
    buff_ = deque()
    p = p_[0]  # input pixel
    x = 0
    P = Classes.P(y, x_1st=0)  # initialize P with: y, x_1st = 0, sign = -1, all params = 0, initially [L, I, G, Dx, Dy]

    for right_p, lower_p in zip(p_[1:], lower_p_[:-1]):  # pixel p is compared to vertically and horizontally subsequent pixels
        dy = lower_p - p    # compare to lower pixel
        dx = right_p - p    # compare to right-side pixel
        g = int(math.hypot(dy, dx)) - ave  # max gradient of right_and_down quadrant, unique for pixel p
        dert = [p, g, dx, dy]
        dert_[x] = dert     # dert ) dert_ ) dert__
        s = g > 0
        P = form_P(s, dert, x, P, P_, buff_, _P_, frame)    # sign is predefined
        p = right_p         # for next lateral comp
        x += 1

    P.terminate(x)  # terminate last P, P.boundaries = [x_1st, x_last, y]
    scan_P_(P, P_, buff_, _P_, frame)  # P scans hP_

    return lower_p_, P_
    # ---------- comp_pixel() end ---------------------------------------------------------------------------------------

def form_P(s, dert, x, P, P_, buff_, hP_, frame):
    " Initialize, accumulate, terminate 1D pattern "
    pri_s = P.sign

    if s != pri_s and pri_s != -1:
        P.terminate(x)  # P.boundaries = [x_1st, x_last, y]
        scan_P_(P, P_, buff_, hP_, frame)  # P scans hP_
        P = Classes.P(y, x_1st=x, sign=s)  # initialize P with y, x_1st = x, sign = s, all params ([L, I, G, Dx, Dy, optional A, sDa]) = 0

    if pri_s == -1: P.sign = s  # new-line P.sign is -1
    P.accum_params([1] + dert)  # P.params [L, I, G, Dx, Dy, optional A, sDa] accumulated with [1] + dert [1, p, g, dx, dy, optional a, sda]

    return P  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_P() end -------------------------------------------------------------------------------------------

def scan_P_(P, P_, _buff_, hP_, frame):
    " P scans shared-x-coordinate hPs in higher P_, combining overlapping Ps into blobs "
    fork_ = []    # refs to segments connected to input P
    _x_1st = 0    # to start while loop
    x_1st, x_last = P.boundaries[:2]

    while _x_1st < x_last:  # while x values overlap between P and _P
        if _buff_:
            seg = _buff_.popleft()  # seg that has been buffered in prior scan_P_
        elif hP_:
            seg = form_segment(hP_.popleft(), frame)    # merge _P into it's fork segments or form new segment
        else:
            break  # higher line ends, all hPs are converted to segments
        _P, xd = seg.Py_[-1]   # higher-line P
        _x_1st, _x_last = _P.boundaries[:2]

        if P.sign == _P.sign and x_1st < _x_last and _x_1st < x_last:
            seg.roots += 1
            fork_.append(seg)  # P-connected segments buffered into fork_
        if _x_last > x_last:   # x overlap between _P and next P: seg is buffered for next scan_P_
            _buff_.append(seg)
        elif seg.roots != 1:   # else seg is checked for termination
            form_blob(seg, frame)  # terminated segment is packed in its blob
        _x_1st = _x_last   # = first x of next _P

    P_.append((P, fork_))  # P with no overlap to next _P is extended to hP and buffered for next-line scan_P_
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------

def form_segment(hP, frame):
    " Convert hP into new segment or add it to higher-line segment, merge blobs "

    _P, fork_ = hP  # unpack _P and it's higher line connected segments (fork_)

    if not fork_: # if no higher-line connection:
        seg = Classes.segment(_P, fork_)  # init segment with _P and fork_: sign, boundaries, params, orient [xD, abs_xD], ave_x, Py_, roots, fork_
        blob = Classes.blob(seg)          # init blob with segment: sign, boundaries, params, orient [xD, abs_xD, Ly]), segment_, open_segments

    else:
        if len(fork_) == 1 and fork_[0].roots == 1: # single connection with higher-line segment, which has single lower-line connection: roots == 1
            # hP is merged into higher-line connected segment:
            seg = fork_[0]  # the only fork
            seg.accum_P(_P) # merge _P into seg, accumulating params, Py_ and orientation_params

        else:  # initialize new segment with shared higher-line-connected-segment's blob:
            seg = Classes.segment(_P, fork_)  # seg is initialized
            blob = fork_[0].blob              # load first fork' blob
            blob.accum_segment(seg)           # add seg to fork' blob
            blob.extend_boundaries(_P.boundaries[:2])   # extend x-boundaries

            if len(fork_) > 1:  # terminate all fork in fork_, merge blobs joined through current seg
                if fork_[0].roots == 1:  # segment could be terminated only if roots == 1
                    form_blob(fork_[0], frame, y_carry=1)  # y_carry for vertical alignment

                for fork in fork_[1:len(fork_)]:
                    if fork.roots == 1:  # segment could be terminated only if roots == 1
                        form_blob(fork, frame, y_carry=1)
                    blob.merge(fork.blob)   # merge blobs of other forks into blob of 1st fork
    return seg
    # ---------- form_segment() end -----------------------------------------------------------------------------------------

def form_blob(term_seg, frame, y_carry=0):
    " Terminate segments and completed blobs "
    blob = term_seg.blob  # blob of terminated segment
    blob.term_segment(segment=term_seg, y=y - y_carry)  # segments packed in blob, y_carry: min elevation of term_seg over current hP

    if not blob.open_segments:  # blob is terminated and packed into frame
        blob.terminate(term_seg.y_end()).localize(frame)
        frame.accum_params(blob.params[1:] + blob.orientation_params)  # frame.params: [I, G, Dx, Dy, xD, abs_xD, Ly], orient: [xD, abs_xD, Ly]
        frame.blob_.append(blob)    # blob is buffered into blob_
    # ---------- form_blob() end ----------------------------------------------------------------------------------------

def image_to_blobs(image):
    " root function, postfix '_' denotes array vs. element, prefix '_' denotes higher-line vs. lower-line variable "

    dert__ = np.empty((Y, X), dtype=object)         # init dert__ as empty object at each pixel
    frame = Classes.frame(dert__, num_params=7)     # init frame object: blob_, dert__, shape, params (= [I, G, Dx, Dy, xD, abs_xD, Ly])

    _P_ = deque()   # higher-line 1D patterns
    p_ = image[0]   # first horizontal line of pixels
    global y
    for y in range(Y - 1):
        lower_p_ = image[y + 1]                         # pixels at line y + 1
        p_, _P_ = comp_pixel(p_, lower_p_, _P_, frame)  # vertical and lateral pixel comparison, form Ps, segments and eventually blobs

    y = Y - 1   # frame ends, merge segs of last line into their blobs
    while _P_:  form_blob(form_segment(_P_.popleft(), frame), frame)

    frame.terminate()   # delete frame.dert__. derts are distributed to all blobs no need to keep their reference here
    return frame        # frame of 2D patterns, to be outputted to level 2
    # ---------- image_to_blobs() end -----------------------------------------------------------------------------------

# ************ MAIN FUNCTIONS END ***************************************************************************************

# ************ PROGRAM BODY *********************************************************************************************
from frame_2D_alg.misc import get_filters
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
from frame_2D_alg.frame_main.DEBUG import draw_blob
draw_blob('./debug', frame_of_blobs, debug_ablob=0, debug_parts=0, debug_local=1, show=0)
# ************ PROGRAM BODY END ******************************************************************************************