import cv2
from time import time
from collections import deque, namedtuple
import numpy as np

'''   
    frame_blobs() defines blobs: contiguous areas of positive or negative deviation of gradient. Gradient is estimated 
    as |dx| + |dy|, then selectively and more precisely as hypot(dx, dy), from cross-comparison among adjacent pixels.

    Complemented by intra_blob (recursive search within blobs), it will be a 2D version of first-level core algorithm.
    frame_blobs() performs several levels (Le) of encoding, incremental per scan line defined by vertical coordinate y.
    value of y per Le line is shown relative to y of current input line, incremented by top-down scan of input image:

    1Le, line y:   comp_pixel (lateral and vertical comp) -> pixel + derivatives tuple: dert ) frame of derts: dert__ 
    2Le, line y-1: form_P(dert2) -> 1D pattern P
    3Le, line y-2: scan_P_(P, hP)-> hP, roots: down-connections, fork_: up-connections between Ps 
    4Le, line y-3: form_segment(hP, seg) -> seg: merge vertically-connected _Ps in non-forking blob segments
    5Le, line y-4+ seg depth: form_blob(seg, blob): merge connected segments in fork_ incomplete blobs, recursively  

    All 2D functions (y_comp, scan_P_, form_segment, form_blob) input two lines: higher and lower, convert elements of 
    lower line into elements of new higher line, then displace elements of old higher line into higher function.
    Higher-line elements include additional variables, derived while they were lower-line elements.
    Processing is mostly sequential because blobs are irregular and very difficult to map to matrices.

    prefix '_' denotes higher-line variable or pattern, vs. same-type lower-line variable or pattern,
    postfix '_' denotes array name, vs. same-name elements of that array
'''


# ************ MAIN FUNCTIONS *******************************************************************************************
# -image_to_blobs()
# -comp_pixel()
# -form_P_()
# -scan_P_()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************

def image_to_blobs(
        image):  # root function, postfix '_' denotes array vs element, prefix '_' denotes higher- vs lower- line variable

    frame = [[0, 0, 0, 0], [], image.shape]  # params, blob_, shape
    dert__ = comp_pixel(image)  # vertically and horizontally bilateral comparison of adjacent pixels
    seg_ = deque()  # buffer of running segments

    for y in range(1, height - 1):  # first and last row are discarded
        P_ = form_P_(dert__[y])  # horizontal clustering
        P_ = scan_P_(P_, seg_, frame)
        seg_ = form_seg_(y, P_, frame)

    while seg_:  form_blob(seg_.popleft(), frame)  # frame ends, last-line segs are merged into their blobs
    return frame  # frame of 2D patterns

    # ---------- image_to_blobs() end -----------------------------------------------------------------------------------


def comp_pixel(p__):  # bilateral comparison between vertically and horizontally consecutive pixels within image

    dert__ = np.empty(shape=(height, width, 4), dtype=int)  # initialize dert__

    dy__ = p__[2:, 1:-1] - p__[:-2,
                           1:-1]  # vertical comp between rows -> dy, (1:-1): first and last column are discarded
    dx__ = p__[1:-1, 2:] - p__[1:-1,
                           :-2]  # lateral comp between columns -> dx, (1:-1): first and last row are discarded
    g__ = np.abs(dy__) + np.abs(dx__) - ave  # deviation of gradient, initially approximated as |dy| + |dx|

    dert__[:, :, 0] = p__
    dert__[1:-1, 1:-1, 1] = dy__  # first row, last row, first column and last-column are discarded
    dert__[1:-1, 1:-1, 2] = dx__
    dert__[1:-1, 1:-1, 3] = g__

    return dert__

    # ---------- comp_pixel() end ---------------------------------------------------------------------------------------


def form_P_(dert_):  # horizontally cluster and sum consecutive pixels and their derivatives into Ps

    P_ = deque()  # row of Ps
    i, dy, dx, g = dert_[1]  # first dert
    x0, L, I, Dy, Dx, G = 1, 1, i, dy, dx, g  # P params
    P_dert_ = [(i, dy, dx, g)]  # dert buffer
    _s = g > 0  # sign

    for x, (i, dy, dx, g) in enumerate(dert_[2:-1], start=2):
        s = g > 0
        if s != _s:  # P is terminated and new P is initialized
            P_.append([_s, x0, L, I, Dy, Dx, G, P_dert_])
            x0, L, I, Dy, Dx, G = x, 0, 0, 0, 0, 0
            P_dert_ = []
        # accumulate P params:
        L += 1
        I += i
        Dy += dy
        Dx += dx
        G += g
        P_dert_.append((i, dy, dx, g))
        _s = s  # prior sign

    P_.append([_s, x0, L, I, Dy, Dx, G, P_dert_])  # last P in row
    return P_

    # ---------- form_P_() end ------------------------------------------------------------------------------------------


def scan_P_(P_, seg_, frame):  # integrate x overlaps (forks) between same-sign Ps and _Ps into blob segments

    new_P_ = deque()

    if P_ and seg_:  # if both are not empty
        P = P_.popleft()  # input-line Ps
        seg = seg_.popleft()  # higher-line segments,
        _P = seg[2][-1]  # last element of each segment is higher-line P
        stop = False
        fork_ = []
        while not stop:
            x0 = P[1]  # first x in P
            xn = x0 + P[2]  # first x in next P
            _x0 = _P[1]  # first x in _P
            _xn = _x0 + _P[2]  # first x in next _P

            if P[0] == _P[0] and _x0 < xn and x0 < _xn:  # test for sign match and x overlap
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

    while P_:  # terminate Ps and segs that continue at line's end
        new_P_.append((P_.popleft(), []))  # no fork
    while seg_:
        form_blob(seg_.popleft(), frame)  # roots always == 0

    return new_P_

    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_seg_(y, P_, frame):  # convert or merge every P into segment, merge blobs
    new_seg_ = deque()

    while P_:
        P, fork_ = P_.popleft()
        s, x0 = P[:2]  # s, x0
        params = P[2:-1]  # L, I, Dx, Dy, G
        xn = x0 + params[0]  # next-P x0 = x0 + L

        if not fork_:  # new_seg is initialized with initialized blob
            blob = [s, [0] * (len(params) + 1), [], 1, [y, x0, xn]]  # s, params, seg_, open_segments, box
            new_seg = [y, [1] + params, [P], 0, fork_, blob]  # y0, params, Py_, roots, fork_, blob
            blob[2].append(new_seg)
        else:
            if len(fork_) == 1 and fork_[0][3] == 1:  # P has one fork and that fork has one root
                new_seg = fork_[0]
                L, I, Dy, Dx, G = params
                Ly, Ls, Is, Dys, Dxs, Gs = new_seg[1]  # fork segment params, P is merged into segment:
                new_seg[1] = [Ly + 1, Ls + L, Is + I, Dys + Dy, Dxs + Dx, Gs + G]
                new_seg[2].append(P)  # Py_: vertical buffer of Ps
                new_seg[3] = 0  # reset roots
                blob = new_seg[-1]

            else:  # if > 1 forks, or 1 fork that has > 1 roots:
                blob = fork_[0][5]
                new_seg = [y, [1] + params, [P], 0, fork_, blob]  # new_seg is initialized with fork blob
                blob[2].append(new_seg)  # segment is buffered into blob

                if len(fork_) > 1:  # merge blobs of all forks
                    if fork_[0][3] == 1:  # if roots == 1: fork hasn't been terminated
                        form_blob(fork_[0], frame)  # merge seg of 1st fork into its blob

                    for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                        if fork[3] == 1:
                            form_blob(fork, frame)

                        if not fork[5] is blob:
                            params, seg_, open_segs, box = fork[5][1:]  # merged blob, omit sign
                            blob[1] = [par1 + par2 for par1, par2 in
                                       zip(blob[1], params)]  # sum params of merging blobs
                            blob[3] += open_segs
                            blob[4][0] = min(blob[4][0], box[0])  # extend box y0
                            blob[4][1] = min(blob[4][1], box[1])  # extend box x0
                            blob[4][2] = max(blob[4][2], box[2])  # extend box xn
                            for seg in seg_:
                                if not seg is fork:
                                    seg[5] = blob  # blobs in other forks are references to blob in the first fork
                                    blob[2].append(seg)  # buffer of merged root segments
                            fork[5] = blob
                            blob[2].append(fork)
                        blob[3] -= 1  # open_segments -= 1: shared with merged blob

        blob[4][1] = min(blob[4][1], x0)  # extend box x0
        blob[4][2] = max(blob[4][2], xn)  # extend box xn
        new_seg_.append(new_seg)

    return new_seg_

    # ---------- form_seg_() end --------------------------------------------------------------------------------------------


def form_blob(term_seg, frame):  # terminated segment is merged into continued or initialized blob (all connected segments)

    y0s, params, Py_, roots, fork_, blob = term_seg
    blob[1] = [par1 + par2 for par1, par2 in zip(params, blob[1])]
    blob[3] += roots - 1  # number of open segments

    if not blob[3]:  # if open_segments == 0: blob is terminated and packed in frame

        s, [Ly, L, I, Dy, Dx, G], seg_, open_segs, (y0, x0, xn) = blob
        yn = y0s + params[0]  # yn = y0 + Ly
        map = np.zeros((yn - y0, xn - x0), dtype=bool)  # local map of blob
        for seg in seg_:
            seg.pop()  # remove references to blob
            for y, P in zip(range(seg[0], seg[0] + seg[1][0]), seg[2]):
                x0P, LP = P[1:3]
                xnP = x0P + LP
                map[y - y0, x0P - x0:xnP - x0] = True

        frame[0][0] += I
        frame[0][1] += Dy
        frame[0][2] += Dx
        frame[0][3] += G

        frame[1].append(nt_blob(typ=0, sign=s, Ly=Ly, L=L,
                                Derts=[(I, Dy, Dx, G)],  # not selective to +sub_blobs as in sub_Derts
                                seg_=seg_,
                                root_blob = None,
                                sub_blob_=[],  # top layer, blob derts_ -> sub_blob derts_
                                sub_Derts=[],
                                # optional sub_blob_ Derts[:] = [(Ly, L, I, Dy, Dx, G)] if len(sub_blob_) > min
                                layer_f=0,
                                # if 1: sub_Derts = layer_Derts, sub_blob_= [(sub_Derts, derts_)], +=/ eval_layer
                                box=(y0, yn, x0, xn),  # boundary box
                                map=map,  # blob boolean map, to compute overlap
                                rng=1,  # for comp_range per blob,  # ncomp=1: for comp_range per dert, not here
                                ))
    # ---------- form_blob() end ----------------------------------------------------------------------------------------


# ************ PROGRAM BODY *********************************************************************************************

ave = 20


# Load inputs --------------------------------------------------------------------
image = cv2.imread('./../images/raccoon_eye.jpg', 0).astype(int)
height, width = image.shape

# Main ---------------------------------------------------------------------------
start_time = time()

nt_blob = namedtuple('blob', 'typ sign Ly L Derts seg_ root_blob sub_blob_ sub_Derts layer_f map box rng')
frame_of_blobs = image_to_blobs(image)

# from intra_blob_debug import intra_blob_hypot  # not yet functional, comment-out to run
# frame_of_blobs = intra_blob_hypot(frame_of_blobs)  # evaluate for deeper clustering inside each blob, recursively

end_time = time() - start_time
print(end_time)

# DEBUG --------------------------------------------------------------------------

# from DEBUG import draw_blobs
# draw_blobs('./../debug/out', frame_of_blobs, -1)

from intra_comp import unfold_blob, hypot_g
from DEBUG import map_dert___

for i, blob in enumerate(frame_of_blobs[1]):
    dert___ = unfold_blob(blob, hypot_g, rdn=0)
    map_dert___('./../debug/hypot_g' + str(i) + '.jpg', dert___)

'''
def alt_form_P_(y, dert__):  # horizontally cluster and sum consecutive pixels and their derivatives into Ps

    P_ = deque()  # P buffer
    L, I, Dy, Dx, G = 0, 0, 0, 0, 0
    Pdert_ = []
    dert_ = dert__[y]  # row of pixels + derivatives
    _i, _dy, _dx, _g = dert_[0]
    _s = _g > 0

    for x, (i, dy, dx, g) in enumerate(dert_[1:]):
        s = g > 0
        if s != _s:
            P_.append([_s, L, I, Dy, Dx, G, Pdert_])  # P is packed into P_
            L, I, Dy, Dx, G = 0, 0, 0, 0, 0   # new P
            Pdert_ = []
        L += 1
        I += _i  # accumulate P params
        Dy += _dy
        Dx += _dx
        G += _g
        Pdert_.append((y, x-1, i, dy, dx, g))
        _s = s; _i = i; _dy = dy; _dx = dx; _g = g  # convert dert to prior dert

    return P_

from filters import get_filters
get_filters(globals())  # imports all filters at once
from scipy import misc
image = misc.face(gray=True)  # input frame of pixels
image = image.astype(int)

op = np.array([[0] * width] * height)
for blob in frame_of_blobs[1]:
    y0, yn, x0, xn = blob.box
    map = blob.map
    slice = op[y0:yn, x0:xn]

    if blob.sign:
        slice[map] = 255
    else:
        slice[map] = 0

cv2.imwrite('./debug/frame.bmp', op)

'''
# ************ PROGRAM BODY END ******************************************************************************************