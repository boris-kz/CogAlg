import cv2
import argparse
from time import time
from collections import deque
import math
import Classes

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
        g = abs(dy) + abs(dx) - ave  # max gradient of right_and_down quadrant per pixel p
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

    frame = Classes.cl_frame(image.reshape((Y, X, 1)), dert_levels=3, copy_dert=True)  # init frame object: blob_, dert__, shape, params (= [I, G, Dx, Dy, xD, abs_xD, Ly])

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
# argument_parser = argparse.ArgumentParser()
# argument_parser.add_argument('-i', '--image', help='path to image file', default='./../images/raccoon_eye.jpg')
# arguments = vars(argument_parser.parse_args())
# image = cv2.imread(arguments['image'], 0).astype(int)
image = cv2.imread(input_path, 0).astype(int)
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