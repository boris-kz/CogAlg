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

    frame = [[0, 0, 0, 0, 0], [], (height, width)]  # Derts, blob_
    derts__ = comp_pixel(image)  # vertically and horizontally bilateral comparison of adjacent pixels
    seg_ = deque()  # buffer of running segments

    for y in range(1, height - 1):  # first and last row are discarded
        P_ = form_P_(y, derts__)  # horizontal clustering
        P_ = scan_P_(P_, seg_, frame)
        seg_ = form_seg_(P_, frame)

    while seg_:  form_blob(seg_.popleft(), frame)  # frame ends, last-line segs are merged into their blobs
    return frame  # frame of 2D patterns

    # ---------- image_to_blobs() end -----------------------------------------------------------------------------------


def comp_pixel(p__):  # bilateral comparison between vertically and horizontally consecutive pixels within image

    ncomp__ = np.zeros(p__.shape, dtype=int)
    dy__    = np.zeros(p__.shape, dtype=int)
    dx__    = np.zeros(p__.shape, dtype=int)

    d__ = p__[1:, :] - p__[:-1, :]          # vertical comp between rows
    nc__ = np.full(shape=d__.shape, fill_value=1)

    ncomp__[1:, :] += nc__;     dy__[1:, :] += d__      # bilateral accumulation
    ncomp__[:-1, :] += nc__;    dy__[:-1, :] += d__     # bilateral accumulation

    d__ = p__[:, 1:] - p__[:, :-1]  # lateral comp between columns
    nc__ = np.full(shape=d__.shape, fill_value=1)

    ncomp__[:, 1:] += nc__;     dx__[:, 1:] += d__      # bilateral accumulation
    ncomp__[:, :-1] += nc__;    dx__[:, :-1] += d__     # bilateral accumulation

    g__ = np.abs(dy__) + np.abs(dx__) - ave             # deviation of gradient, initially approximated as |dy| + |dx|

    # break-down into derts:
    derts__ = [[(p__[y, x], (ncomp__[y, x], dy__[y, x], dx__[y, x], g__[y, x])) for x in range(width)] for y in range(height)]

    return derts__
    # ---------- comp_pixel() end ---------------------------------------------------------------------------------------

def form_P_(y, derts__):  # cluster and sum horizontally consecutive pixels and their derivatives into Ps

    P_ = deque()  # P buffer
    derts_ = derts__[y]  # row of pixels + derivatives
    x = 0

    while x < width:
        i, (n, dy, dx, g) = derts_[x]
        s = g > 0                   # pre-evaluate
        Derts = [0, 0, 0, 0, [0, 0, 0, 0]]  # L, Y, X, I, [N, Dy, Dx, G]
        P = [s, Derts, []]
        while x < width and s == P[0]:      # while s is still identical
            i, (n, dy, dx, g) = derts_[x]    # accumulate P' Derts:
            Derts[0] += 1  # L
            Derts[1] += y  # Y
            Derts[2] += x  # X
            Derts[3] += i  # I
            Derts[4][0] += n  # N
            Derts[4][1] += dy # Dy
            Derts[4][2] += dx # Dx
            Derts[4][3] += g  # G
            P[2].append((x, i, (n, dy, dx, g)))

            x += 1

            s = g > 0  # s = (g > 0)

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
            x0 = P[2][0][0]  # first x in P
            xn = P[2][-1][0]  # last x in P
            _x0 = _P[2][0][0]  # first x in _P
            _xn = _P[2][-1][0]  # last x in _P

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
        s, Derts, derts_ = P

        if not fork_:  # seg is initialized with initialized blob
            blob = [s, [0] * (len(Derts) + 1), [], 1]  # s, Derts, seg_, open_segments
            seg = [s, [1] + Derts, [P], 0, fork_, blob]  # s, Derts, P_, roots, fork_, blob
            blob[2].append(seg)

        else:
            if len(fork_) == 1 and fork_[0][3] == 1:  # P has one fork and that fork has one root
                seg = fork_[0]
                L, Y, X, I, [N, Dy, Dx, G] = Derts
                Ly, Ls, Ys, Xs, Is, [Ns, Dys, Dxs, Gs] = seg[1]  # fork segment Derts
                # P is merged into segment:
                seg[1] = [Ly + 1, Ls + L, Ys + Y, Xs + X, Is + I, [Ns + N, Dys + Dy, Dxs + Dx, Gs + G]]
                seg[2].append(P)  # P_: vertical buffer of Ps merged into seg
                seg[3] = 0  # reset roots

            else:  # if > 1 forks, or 1 fork that has > 1 roots:
                blob = fork_[0][5]
                seg = [s, [1] + Derts, [P], 0, fork_, blob]  # seg is initialized with fork blob
                blob[2].append(seg)  # segment is buffered into blob

                if len(fork_) > 1:  # merge blobs of all forks
                    if fork_[0][3] == 1:  # if roots == 1: fork hasn't been terminated
                        form_blob(fork_[0], frame)  # merge seg of 1st fork into its blob

                    for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                        if fork[3] == 1:
                            form_blob(fork, frame)

                        if not fork[5] is blob:
                            Derts, e_, open_segments = fork[5][1:]  # merged blob, omit sign
                            # sum same-type Derts of merging blobs
                            blob[1] = [[sub_par1 + sub_par2 for sub_par1, sub_par2 in zip(par1, par2)] if type(par1) != int else par1 + par2 for par1, par2 in zip(blob[1], Derts)]
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

    Derts, P_, roots, fork_, blob = term_seg[1:]
    # sum same-type Derts of blob and merging segment
    blob[1] = [[sub_par1 + sub_par2 for sub_par1, sub_par2 in zip(par1, par2)] if type(par1) != int else par1 + par2 for par1, par2 in zip(blob[1], Derts)]
    blob[3] += roots - 1  # number of open segments

    if not blob[3]:  # if open_segments == 0: blob is terminated and packed in frame
        s, [Ly, L, Y, X, I, [N, Dy, Dx, G]], e_ = blob[:3]
        y0 = 9999999
        x0 = 9999999
        yn = 0
        xn = 0

        map = np.zeros((height, width), dtype=bool)
        for seg in e_:
            seg.pop()  # remove references to blob
            for P in seg[2]:
                L, Y = P[1][:2]

                y = Y // L
                y0 = min(y0, y)
                yn = max(yn, y + 1)
                for derts in P[2]:
                    x = derts[0]
                    x0 = min(x0, x)
                    xn = max(xn, x + 1)
                    map[y, x] = True

        map = map[y0:yn, x0:xn]

        frame[0][0] += I
        frame[0][1] += N
        frame[0][2] += Dy
        frame[0][3] += Dx
        frame[0][4] += G

        frame[1].append(nt_blob(sign=s,
                                Derts=[Ly, L, Y, X, I, [N, Dy, Dx, G]],
                                seg_=e_,
                                box=(y0, yn, x0, xn),
                                map=map,
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

nt_blob = namedtuple('blob', 'sign Derts seg_ box map sub_blob_')  # define named tuple
frame_of_blobs = image_to_blobs(image)

# from frame_2D_alg.intra_blob import intra_blob_root
# frame_of_blobs = intra_blob_root(frame_of_blobs)  # evaluate for deeper recursive clustering inside each blob

end_time = time() - start_time
print(end_time)

# Rebuild blob -------------------------------------------------------------------
# from frame_2D_alg.DEBUG import draw_blobs
# draw_blobs('../debug/out', frame_of_blobs, isb=-1)
# ************ PROGRAM BODY END ******************************************************************************************
