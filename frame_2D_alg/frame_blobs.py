import cv2
from time import time
from collections import deque, namedtuple
import numpy as np
import numpy.ma as ma
# Filters ------------------------------------------------------------------------
from frame_2D_alg.filters import get_filters
get_filters(globals())          # imports all filters at once
# --------------------------------------------------------------------------------
'''   
    frame_blobs() defines blobs: contiguous areas of positive or negative deviation of gradient. Gradient is estimated 
    as |dx| + |dy|, then selectively and more precisely as hypot(dx, dy), from cross-comparison among adjacent pixels.
    Complemented by intra_blob (recursive search within blobs), it will be a 2D version of first-level core algorithm.
    
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
    
    All 2D functions (y_comp, scan_P_, form_segment, form_blob) input two lines: higher and lower, convert elements of 
    lower line into elements of new higher line, then displace elements of old higher line into higher function.
    Higher-line elements include additional variables, derived while they were lower-line elements.
    Processing is mostly sequential because blobs are irregular and very difficult to map to matrices.
'''

# ************ MAIN FUNCTIONS *******************************************************************************************
# -image_to_blobs()
# -comp_pixel()
# -form_P_()
# -scan_P_()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************

def image_to_blobs(image):  # root function, postfix '_' denotes array vs element, prefix '_' denotes higher- vs lower- line variable

    frame = [[0, 0, 0, 0], []]  # params, blob_
    comp_pixel(frame, image)  # vertically and horizontally bilateral comparison of adjacent pixels
    seg_ = deque()  # buffer of running segments

    for y in range(1, height - 1):  # first and last row are discarded
        P_ = form_P_(y, frame)  # horizontal clustering
        P_ = scan_P_(P_, seg_, frame)
        seg_ = form_seg_(P_, frame)

    while seg_:  form_blob(seg_.popleft(), frame)  # frame ends, last-line segs are merged into their blobs
    return frame  # frame of 2D patterns

    # ---------- image_to_blobs() end -----------------------------------------------------------------------------------


def comp_pixel(frame, p__):  # bilateral comparison between vertically and horizontally consecutive pixels within image

    dert__ = ma.empty(shape=(width, height, 4), dtype=int)  # initialize dert__

    dy__ = p__[2:, 1:-1] - p__[:-2, 1:-1]  # vertical comp between rows -> dy, (1:-1): first and last column are discarded
    dx__ = p__[1:-1, 2:] - p__[1:-1, :-2]  # lateral comp between columns -> dx, (1:-1): first and last row are discarded
    g__ = np.abs(dy__) + np.abs(dx__) - ave  # deviation of gradient, initially approximated as |dy| + |dx|

    dert__[:, :, 0] = p__
    dert__[1:-1, 1:-1, 1] = dy__  # first row, last row, first column and last-column are discarded
    dert__[1:-1, 1:-1, 2] = dx__
    dert__[1:-1, 1:-1, 3] = g__

    frame.append(dert__)

    # ---------- comp_pixel() end ---------------------------------------------------------------------------------------

def form_P_(y, frame):  # cluster and sum horizontally consecutive pixels and their derivatives into Ps

    P_ = deque()  # P buffer
    dert_ = frame[-1][y, :, :]  # row of pixels + derivatives
    x_stop = width - 1
    x = 1  # first and last columns are discarded

    while x < x_stop:
        s = dert_[x][-1] > 0  # s = (g > 0)
        params = [0, 0, 0, 0, 0, 0, 0]  # L, Y, X, I, Dy, Dx, G
        P = [s, params, []]
        while x < x_stop and s == P[0]:
            i, dy, dx, g = dert_[x, :]  # accumulate P' params:
            params[0] += 1  # L
            params[1] += y  # Y
            params[2] += x  # X
            params[3] += i  # I
            params[4] += dy # dy
            params[5] += dx # dx
            params[6] += g  # G
            P[2].append((y, x, i, dy, dx, g))
            x += 1
            s = dert_[x][-1] > 0  # s = (g > 0)

        if params[0]:  # if L > 0
            P_.append(P)  # P is packed into P_
    return P_

    # ---------- form_P_() end ------------------------------------------------------------------------------------------

def scan_P_(P_, seg_, frame): # this function detects connections (forks) between Ps and _Ps, to form blob segments
    new_P_ = deque()

    if P_ and seg_:           # if both are not empty
        P = P_.popleft()      # input-line Ps
        seg = seg_.popleft()  # higher-line segments,
        _P = seg[2][-1]       # last element of each segment is higher-line P
        stop = False
        fork_ = []
        while not stop:
            x0 = P[2][0][1]  # first x in P
            xn = P[2][-1][1]  # last x in P
            _x0 = _P[2][0][1]  # first x in _P
            _xn = _P[2][-1][1]  # last x in _P

            if P[0] == _P[0] and _x0 <= xn and x0 <= _xn:  # test for sign match and x overlap
                seg[3] += 1
                fork_.append(seg)  # P-connected segments are buffered into fork_

            if xn < _xn:  # _P overlaps next P in P_
                new_P_.append((P, fork_))
                fork_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:  # if no P left: terminate loop
                    if seg[3] != 1:  # if roots != 1: terminate seg
                        form_blob(seg, frame)
                    stop = True
            else:  # no next-P overlap
                if seg[3] != 1:  # if roots != 1: terminate seg
                    form_blob(seg, frame)

                if seg_:  # load next _P
                    seg = seg_.popleft()
                    _P = seg[2][-1]
                else:  # if no seg left: terminate loop
                    new_P_.append((P, fork_))
                    stop = True

    while P_:  # handle Ps and segs that don't terminate at line's end
        new_P_.append((P_.popleft(), []))  # no fork
    while seg_:
        form_blob(seg_.popleft(), frame)  # roots always == 0
    return new_P_

    # ---------- scan_P_() end ------------------------------------------------------------------------------------------

def form_seg_(P_, frame):  # convert or merge every P into segment, merge blobs
    new_seg_ = deque()

    while P_:
        P, fork_ = P_.popleft()
        s, params, dert_ = P

        if not fork_:  # seg is initialized with initialized blob
            blob = [s, [0] * (len(params) + 1), [], 1]  # s, params, seg_, open_segments
            seg = [s, [1] + params, [P], 0, fork_, blob]  # s, params, P_, roots, fork_, blob
            blob[2].append(seg)

        else:
            if len(fork_) == 1 and fork_[0][3] == 1:  # P has one fork and that fork has one root
                seg = fork_[0]
                L, Y, X, I, Dy, Dx, G = params
                Ly, Ls, Ys, Xs, Is, Dys, Dxs, Gs = seg[1]  # fork segment params
                # P is merged into segment:
                seg[1] = [Ly + 1, Ls + L, Ys + Y, Xs + X, Is + I, Dys + Dy, Dxs + Dx, Gs + G]
                seg[2].append(P)  # P_: vertical buffer of Ps merged into seg
                seg[3] = 0  # reset roots

            else:  # if > 1 forks, or 1 fork that has > 1 roots:
                blob = fork_[0][5]
                seg = [s, [1] + params, [P], 0, fork_, blob]  # seg is initialized with fork blob
                blob[2].append(seg)  # segment is buffered into blob

                if len(fork_) > 1:  # merge blobs of all forks
                    if fork_[0][3] == 1:  # if roots == 1: fork hasn't been terminated
                        form_blob(fork_[0], frame)  # merge seg of 1st fork into its blob

                    for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                        if fork[3] == 1:
                            form_blob(fork, frame)

                        if not fork[5] is blob:
                            params, e_, open_segments = fork[5][1:]  # merged blob, omit sign
                            blob[1] = [par1 + par2 for par1, par2 in zip(blob[1], params)]  # sum same-type params of merging blobs
                            blob[3] += open_segments
                            for e in e_:
                                if not e is fork:
                                    e[5] = blob  # blobs in other forks are references to blob in the first fork
                                    blob[2].append(e)  # buffer of merged root segments
                            fork[5] = blob
                            blob[2].append(fork)
                        blob[3] -= 1  # open_segments -= 1 due to merged blob shared seg

        new_seg_.append(seg)
    return new_seg_

    # ---------- form_seg_() end --------------------------------------------------------------------------------------------

def form_blob(term_seg, frame):  # terminated segment is merged into continued or initialized blob (all connected segments)

    params, P_, roots, fork_, blob = term_seg[1:]
    blob[1] = [par1 + par2 for par1, par2 in zip(params, blob[1])]
    blob[3] += roots - 1  # number of open segments

    if not blob[3]:  # if open_segments == 0: blob is terminated and packed in frame
        s, [Ly, L, Y, X, I, Dy, Dx, G], e_ = blob[:3]
        y0 = 9999999
        x0 = 9999999
        yn = 0
        xn = 0
        for seg in e_:
            seg.pop()  # remove references to blob
            for P in seg[2]:
                y0 = min(y0, P[2][0][0])
                x0 = min(x0, P[2][0][1])
                yn = max(yn, P[2][0][0] + 1)
                xn = max(xn, P[2][-1][1] + 1)

        dert__ = frame[-1][y0:yn, x0:xn, :]
        map = np.zeros((height, width), dtype=bool)
        for seg in e_:
            for P in seg[2]:
                for y, x, i, dy, dx, g in P[2]:
                    map[y, x] = True
        map = map[y0:yn, x0:xn]

        frame[0][0] += I
        frame[0][1] += Dy
        frame[0][2] += Dx
        frame[0][3] += G

        frame[1].append(nt_blob(sign=s,
                                params=[Ly, L, Y, X, I, Dy, Dx, G],
                                e_=e_,
                                box=(y0, yn, x0, xn),
                                map=map,
                                dert__=dert__,
                                new_dert__=[None],
                                rng=1, ncomp=1,
                                sub_blob_=[]))
    # ---------- form_blob() end ----------------------------------------------------------------------------------------

# ************ PROGRAM BODY *********************************************************************************************

from frame_2D_alg.filters import get_filters
get_filters(globals())  # import all filters at once

# Load inputs --------------------------------------------------------------------
image = cv2.imread(input_path, 0).astype(int)
height, width = image.shape

# Main ---------------------------------------------------------------------------
start_time = time()

nt_blob = namedtuple('blob', 'sign params e_ box map dert__ new_dert__ rng ncomp sub_blob_')  # define named tuple
frame_of_blobs = image_to_blobs(image)

from frame_2D_alg.intra_blob import intra_blob_root
frame_of_blobs = intra_blob_root(frame_of_blobs)  # evaluate for deeper recursive clustering inside each blob

end_time = time() - start_time
print(end_time)

# Rebuild blob -------------------------------------------------------------------
from frame_2D_alg.DEBUG import draw_blobs
draw_blobs('../debug/out', frame_of_blobs, isb=3)
# ************ PROGRAM BODY END ******************************************************************************************
