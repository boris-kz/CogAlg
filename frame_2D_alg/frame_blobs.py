import cv2
from time import time
from collections import deque
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
# -image_to_blobs()
# -comp_pixel()
# -form_P_()
# -scan_P_()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************

def image_to_blobs(image):
    " root function, postfix '_' denotes array vs. element, prefix '_' denotes higher-line vs. lower-line variable "

    frame = [[0, 0, 0, 0], []]   # initialize frame: params, blob_

    comp_pixel(frame, image)            # bilateral comp of image's pixels, vertically and horizontally

    # clustering of inputs:

    seg_ = deque()   # buffer of running segments

    for y in range(1, height - 1):   # first and last row are discarded

        P_ = form_P_(y, frame)  # horizontal clustering
        P_ = scan_P_(P_, seg_, frame)
        seg_ = form_seg_(P_, frame)

    # frame ends, merge segs of last line into their blobs:
    while seg_:  form_blob(seg_.popleft(), frame)
    return frame  # frame of 2D patterns, to be outputted to level 2
    # ---------- image_to_blobs() end -----------------------------------------------------------------------------------

def comp_pixel(frame, p__):
    ''' Bilateral comparison of consecutive pixels, vertically and horizontally, over the whole image. '''

    Y, X = p__.shape    # height and width of frame
    dert__ = np.empty(shape=(Y, X, 4), dtype=int)   # initialize dert__ - place holders of parameters derived from pixels

    # vertical pixel comp. dy is twice the gradient magnitude at 90 degree
    dy__ = p__[2:, 1:-1] - p__[:-2, 1:-1]   # horizontal indices slicing (1:-1): first and last column are discarded

    # horizontal pixel comp. dx is twice the gradient magnitude at 0 degree:
    dx__ = p__[1:-1, 2:] - p__[1:-1, :-2]   # vertical indices slicing (1:-1): first and last row are discarded

    # sg is L1 norm of dy and dx, minus twice the value of ave
    sg__ = np.abs(dy__) + np.abs(dx__) - 2 * ave

    dert__[:, :, 0] = p__
    dert__[1:-1, 1:-1, 1] = dy__    # first row, last row, first column and last-column are discarded
    dert__[1:-1, 1:-1, 2] = dx__
    dert__[1:-1, 1:-1, 3] = sg__

    frame.append(dert__)    # pack dert__ into frame

    # ---------- comp_pixel() end ---------------------------------------------------------------------------------------

def form_P_(y, frame):
    ''' cluster horizontally consecutive inputs into Ps, buffered in P_ '''

    P_ = deque()  # initialize the output of this function

    dert_ = frame[-1][y, :, :]  # row y

    x_stop = width - 1
    x = 1                                   # first and last column are discarded

    while x < x_stop:

        s = dert_[x][-1] > 0  # s = (sg > 0)
        params = [0, 0, 0, 0, 0, 0, 0]  # L, Y, X, I, Dy, Dx, sG

        P = [s, params, []]

        while x < x_stop and s == P[0]:

            i, dy, dx, sg = dert_[x, :]

            # accumulate P's params:
            params[0] += 1      # L
            params[1] += y      # Y
            params[2] += x      # X
            params[3] += i      # I
            params[4] += dy     # dy
            params[5] += dx     # dx
            params[6] += sg     # sG

            P[2].append((y, x, i, dy, dx, sg))

            x += 1

            s = dert_[x][-1] > 0  # s = (sg > 0)

        if params[0]:       # if L > 0
            P_.append(P)    # P is packed into P_

    return P_

    # ---------- form_P_() end ------------------------------------------------------------------------------------------

def scan_P_(P_, seg_, frame):
    ''' each running segment in seg_ has 1 _P, or P of higher-line. P_ contain current line P.
        This function detects all connections between every P and _P in the form of fork_ '''

    new_P_ = deque()
    if P_ and seg_:             # if both are not empty

        P = P_.popleft()
        seg = seg_.popleft()
        _P = seg[2][-1]         # higher-line P is last element in seg's P_

        stop = False
        fork_ = []

        while not stop:

            x0 = P[2][0][1]     # P's first dert's x: y, x, i, dy, dx, sg = P[2][0]
            xn = P[2][-1][1]    # P's last dert's x: y, x, i, dy, dx, sg = P[2][-1]

            _x0 = _P[2][0][1]   # P's first dert's x: y, x, i, dy, dx, sg = P[2][0]
            _xn = _P[2][-1][1]  # P's last dert's x: y, x, i, dy, dx, sg = P[2][-1]

            if P[0] == _P[0] and _x0 <= xn and x0 <= _xn:  # check sign and olp
                seg[3] += 1
                fork_.append(seg)  # P-connected segments buffered into fork_

            if xn < _xn:    # P is on the left of _P: next P
                new_P_.append((P, fork_))
                fork_ = []
                if P_:      # switch to next P
                    P = P_.popleft()
                else:       # terminate loop
                    if seg[3] != 1: # if roots != 1
                        form_blob(seg, frame)
                    stop = True
            else:           # _P is on the left of P_: next _P
                if seg[3] != 1: # if roots != 1
                    form_blob(seg, frame)

                if seg_:    # switch to new _P
                    seg = seg_.popleft()
                    _P = seg[2][-1]
                else:       # terminate loop
                    new_P_.append((P, fork_))
                    stop = True

    # handle the remainders:
    while P_:
        new_P_.append((P_.popleft(), []))     # no fork
    while seg_:
        form_blob(seg_.popleft(), frame)  # roots always == 0

    return new_P_
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------

def form_seg_(P_, frame):
    " Convert or merge every P into segment. Merge blobs "

    new_seg_ = deque()

    while P_:

        P, fork_ = P_.popleft()
        s, params, dert_ = P

        if not fork_:  # seg is initialized with initialized blob (params, coordinates, incomplete_segments, root_, xD)
            blob = [s, [0] * (len(params) + 1), [], 1]      # s, params, seg_, open_segments
            seg = [s, [1] + params, [P], 0, fork_, blob]    # s, params. P_, roots, fork_, blob
            blob[2].append(seg)
        else:
            if len(fork_) == 1 and fork_[0][3] == 1:  # P has one fork and that fork has one root
                # P is merged into segment fork_[0] (seg):
                seg = fork_[0]

                L, Y, X, I, Dy, Dx, sG = params
                Ly, Ls, Ys, Xs, Is, Dys, Dxs, sGs = seg[1]     # fork's params

                seg[1] = [Ly + 1, Ls + L, Ys + Y, Xs + X, Is + I, Dys + Dy, Dxs + Dx, sGs + sG]

                seg[2].append(P)    # P_: vertical buffer of Ps merged into seg
                seg[3] = 0          # reset roots

            else:  # if > 1 forks, or 1 fork that has > 1 roots:
                blob = fork_[0][5]                       # fork's blob
                seg = [s, [1] + params, [P], 0, fork_, blob] # seg is initialized with fork's blob
                blob[2].append(seg) # segment is buffered into blob
                if len(fork_) > 1:  # merge blobs of all forks
                    if fork_[0][3] == 1:  # if roots == 1: fork hasn't been terminated
                        form_blob(fork_[0], frame)  # merge seg of 1st fork into its blob

                    for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                        if fork[3] == 1:
                            form_blob(fork, frame)

                        if not fork[5] is blob:
                            params, e_, open_segments = fork[5][1:]  # merged blob, ommit sign
                            blob[1] = [par1 + par2 for par1, par2 in zip(params, blob[1])]  # sum every params per type of 2 merging blobs
                            blob[3] += open_segments
                            for e in e_:
                                if not e is fork:
                                    e[5] = blob       # blobs in other forks are references to blob in the first fork
                                    blob[2].append(e) # buffer of merged root segments
                            fork[5] = blob
                            blob[2].append(fork)
                        blob[3] -= 1    # open_segments -= 1 due to merged blob shared seg

        new_seg_.append(seg)
    return new_seg_
    # ---------- form_seg_() end --------------------------------------------------------------------------------------------

def form_blob(term_seg, frame):
    " Terminated segment is merged into continued or initialized blob (all connected segments) "
    params, P_, roots, fork_, blob = term_seg[1:]

    blob[1] = [par1 + par2 for par1, par2 in zip(params, blob[1])]
    blob[3] += roots - 1    # number of open segments

    if not blob[3]:  # if open_segments == 0: blob is terminated and packed in frame
        blob.pop()
        [Ly, L, Y, X, I, Dy, Dx, sG] = blob[1]
        # frame P are to compute averages, redundant for same-scope alt_frames

        frame[0][0] += I
        frame[0][1] += Dy
        frame[0][2] += Dx
        frame[0][3] += sG
        frame[1].append(blob)
    # ---------- form_blob() end ----------------------------------------------------------------------------------------

# ************ MAIN FUNCTIONS END ***************************************************************************************

# ************ PROGRAM BODY *********************************************************************************************
# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:
from misc import get_filters
get_filters(globals())          # imports all filters at once

# Load inputs --------------------------------------------------------------------
image = cv2.imread(input_path, 0).astype(int)
height, width = image.shape

# Main ---------------------------------------------------------------------------
start_time = time()
frame_of_blobs = image_to_blobs(image)
end_time = time() - start_time
print(end_time)

# Rebuild blob -------------------------------------------------------------------
from frame_2D_alg.DEBUG import draw_blob
draw_blob('../debug/frame', frame_of_blobs)
# ************ PROGRAM BODY END ******************************************************************************************
