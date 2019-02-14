import cv2
import argparse
from time import time
from collections import deque
import math
from frame_2D_alg import Classes

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
# -image_to_blobs()
# ***********************************************************************************************************************

def comp_pixel(y, p_, lower_p_, dert_):
    " cross-compare adjacent pixels, compute gradient "
    P_ = deque()
    p = p_[0]  # input pixel
    x = 0
    P = Classes.cl_P(x0=0, num_params=dert_.shape[1]+1)  # initialize P with: y, x0 = 0, sign = -1, all params = 0, initially [L, I, G, Dx, Dy]

    for right_p, lower_p in zip(p_[1:], lower_p_[:-1]):  # pixel p is compared to vertically and horizontally subsequent pixels
        dy = lower_p - p  # compare to lower pixel
        dx = right_p - p  # compare to right-side pixel
        g = int(math.hypot(dy, dx)) - ave  # max gradient of right_and_down quadrant per pixel p
        dert = [p, g, dx, dy]
        dert_[x] = dert  # dert ) dert_ ) dert__ 
        s = g > 0
        P = Classes.form_P(x, y, s, dert, P, P_)  # sign is predefined
        p = right_p  # for next lateral comp
        x += 1

    P.terminate(x, y)  # terminate last P, P.box = [x0, xn, y]
    P_.append(P)
    return lower_p_, P_
    # ---------- comp_pixel() end ---------------------------------------------------------------------------------------

def image_to_blobs(image):
    " root function, postfix '_' denotes array vs. element, prefix '_' denotes higher-line vs. lower-line variable "

    dert__ = Classes.init_dert__(3, image.reshape((Y, X, 1)))       # init dert__ as a cube: depth is 1 + number of derivatives: p, g, dx, dy
    frame = Classes.cl_frame(dert__, copy_dert=True)  # init frame object: blob_, dert__, shape, params (= [I, G, Dx, Dy, xD, abs_xD, Ly])

    seg_ = deque()  # higher-line 1D patterns
    p_ = image[0]  # first horizontal line of pixels
    for y in range(Y - 1):
        lower_p_ = image[y + 1]  # pixels at line y + 1
        p_, P_ = comp_pixel(y, p_, lower_p_, dert_=frame.dert__[y])  # vertical and lateral pixel comparison, form Ps, segments and eventually blobs
        P_ = Classes.scan_P_(y, P_, seg_, frame)
        seg_ = Classes.form_segment(y, P_, frame)

    y = Y - 1  # frame ends, merge segs of last line into their blobs
    while seg_:  Classes.form_blob(y, seg_.popleft(), frame)

    frame.terminate()  # delete frame.dert__. derts are distributed to all blobs no need to keep their reference here
    return frame  # frame of 2D patterns, to be outputted to level 2
    # ---------- image_to_blobs() end -----------------------------------------------------------------------------------

# ************ MAIN FUNCTIONS END ***************************************************************************************
# ************ PROGRAM BODY *********************************************************************************************

from frame_2D_alg.misc import get_filters
get_filters(globals())  # imports all filters at once

# Load inputs --------------------------------------------------------------------
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./../images/raccoon_eye.jpg')
arguments = vars(argument_parser.parse_args())
image = cv2.imread(arguments['image'], 0).astype(int)
Y, X = image.shape  # image height and width

# Main ---------------------------------------------------------------------------
start_time = time()
frame_of_blobs = image_to_blobs(image)
end_time = time() - start_time
print(end_time)

# Rebuild blob -------------------------------------------------------------------
# from DEBUG import draw_blob
# draw_blob('./../debug', frame_of_blobs, debug_ablob=0, debug_parts=0, debug_local=0, show=0)
# ************ PROGRAM BODY END ******************************************************************************************
