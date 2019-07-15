import numpy as np
from time import time
from collections import deque, namedtuple

'''   
    frame_blobs() defines blobs: contiguous areas of positive or negative deviation of gradient. Gradient is estimated 
    as |dx| + |dy|, then selectively and more precisely as hypot(dx, dy), from cross-comparison among adjacent pixels.
    Complemented by intra_blob (recursive search within blobs), it will be a 2D version of first-level core algorithm.

    frame_blobs() performs several levels (Le) of encoding, incremental per scan line defined by vertical coordinate y.
    value of y per Le line is shown relative to y of current input line, incremented by top-down scan of input image:
    canopy 

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

# Structures:

Dert = namedtuple('Dert', 'G, A, Dy, Dx, L, Ly')
Pattern = namedtuple('Pattern', 'sign, x0, I, G, Dy, Dx, L, dert_')
Segment = namedtuple('Segment', 'y, I, G, Dy, Dx, L, Ly, Py_')
Blob = namedtuple('Blob', 'Dert, sign, rng, box, map, seg_, dert__, sub_blob_, lLayers, root_blob, hLayers')
Frame = namedtuple('Frame', 'I, G, Dy, Dx, blob_, i__, dert__')

# Adjustable parameters:
kwidth = 3  # Declare initial kernel size. Tested values are 2 or 3.
ave = 20
DEBUG = True

if kwidth == 3:
    ave *= 4
    rng = 1
elif kwidth == 2:
    rng = 0
else:
    print("kwidth must be 2 or 3!")


# ************ MODULE FUNCTIONS *****************************************************************************************
# -image_to_blobs()
# -comp_pixel()
# -form_P_()
# -scan_P_()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************

def image_to_blobs(image):  # root function, postfix '_' denotes array vs element, prefix '_' denotes higher- vs lower- line variable

    i__, dert__ = comp_pixel(image)  # vertically and horizontally bilateral comparison of adjacent pixels
    frame = Frame(0, 0, 0, 0, [], i__, dert__)  # params, blob_, dert__
    seg_ = deque()  # buffer of running segments

    height, width = image.shape

    for y in range(height - kwidth + 1):  # first and last row are discarded
        P_ = form_P_(i__[y], dert__[:, y].T)  # horizontal clustering
        P_ = scan_P_(P_, seg_, frame)
        seg_ = form_seg_(y, P_, frame)

    while seg_:  form_blob(seg_.popleft(), frame)  # frame ends, last-line segs are merged into their blobs
    return frame  # frame of 2D patterns

    # ---------- image_to_blobs() end -----------------------------------------------------------------------------------


def comp_pixel(image):  # comparison between pixel and its neighbours within kernel, for the whole image

    # Initialize variables:
    if kwidth == 2:

        # Compare:
        dy__ = (image[1:, 1:] + image[:-1, 1:]) + (image[1:, :-1] - image[:-1, :-1]) * 0.5
        dx__ = (image[1:, 1:] - image[1:, :-1]) + (image[:-1, 1:] - image[:-1, :-1]) * 0.5

        # Sum pixel values:
        p__ = (image[:-1, :-1]
               + image[:-1, 1:]
               + image[1:, :-1]
               + image[1:, 1:]) * 0.25

    else:
        ycoef = np.array([-0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0])
        xcoef = np.array([-0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1])

        # Compare by subtracting centered image from translated image:
        d___ = np.array(list(map(lambda trans_slices:
                                 image[trans_slices] - image[1:-1, 1:-1],
                                 [
                                     (slice(None, -2), slice(None, -2)),
                                     (slice(None, -2), slice(1, -1)),
                                     (slice(None, -2), slice(2, None)),
                                     (slice(1, -1), slice(2, None)),
                                     (slice(2, None), slice(2, None)),
                                     (slice(2, None), slice(1, -1)),
                                     (slice(2, None), slice(None, -2)),
                                     (slice(1, -1), slice(None, -2)),
                                 ]))).swapaxes(0, 2).swapaxes(0, 1)

        # Decompose differences:
        dy__ = (d___ * ycoef).sum(axis=2)
        dx__ = (d___ * xcoef).sum(axis=2)

        # Sum pixel values:
        p__ = image[1:-1, 1:-1]

    # Compute gradient magnitudes per kernel:
    g__ = np.hypot(dy__, dx__)

    return p__[np.newaxis, ...], np.around(np.stack((g__, dy__, dx__), axis=0))


def form_P_(i_, dert_):  # horizontally cluster and sum consecutive pixels and their derivatives into Ps

    P_ = deque()  # row of Ps
    i = i_[0]
    g, dy, dx = dert_[0]  # first dert
    x0, I, G, Dy, Dx, L = 0, i, g, dy, dx, 1  # P params
    vg = g - ave
    _s = vg > 0  # sign

    for x, (i, (g, dy, dx)) in enumerate(zip(i_[1:], dert_[1:]), start=1):
        vg = g - ave
        s = vg > 0
        if s != _s:  # P is terminated and new P is initialized
            P = Pattern(_s, x0, I, G, Dy, Dx, L, dert_[x0:x0 + L])
            P_.append(P)
            x0, I, G, Dy, Dx, L = x, 0, 0, 0, 0, 0

        # accumulate P params:
        I += i
        G += vg
        Dy += dy
        Dx += dx
        L += 1
        _s = s  # prior sign

    P = Pattern(_s, x0, I, G, Dy, Dx, L, dert_[x0:x0 + L])
    P_.append(P)  # last P in row
    return P_

    # ---------- form_P_() end ------------------------------------------------------------------------------------------


def scan_P_(P_, seg_, frame):  # integrate x overlaps (forks) between same-sign Ps and _Ps into blob segments

    new_P_ = deque()

    if P_ and seg_:  # if both are not empty
        P = P_.popleft()  # input-line Ps
        seg = seg_.popleft()  # higher-line segments,
        _P = seg[-3][-1]  # last element of each segment is higher-line P
        fork_ = []

        while True:
            x0 = P.x0  # first x in P
            xn = x0 + P.L  # first x in next P
            _x0 = _P.x0  # first x in _P
            _xn = _x0 + _P.L  # first x in next _P

            if P.sign == _P.sign and _x0 < xn and x0 < _xn:  # test for sign match and x overlap
                seg[-1] += 1  # roots
                fork_.append(seg)  # P-connected segments are buffered into fork_

            if xn < _xn:  # _P overlaps next P in P_
                new_P_.append((P, fork_))
                fork_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:  # terminate loop
                    if seg[-1] != 1:  # if roots != 1: terminate seg
                        form_blob(seg, frame)
                    break
            else:  # no next-P overlap
                if seg[-1] != 1:  # if roots != 1: terminate seg
                    form_blob(seg, frame)

                if seg_:  # load next _P
                    seg = seg_.popleft()
                    _P = seg[-3][-1]
                else:  # if no seg left: terminate loop
                    new_P_.append((P, fork_))
                    break

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

        s, x0, I, G, Dy, Dx, L, dert_ = P
        xn = x0 + L  # next-P x0

        if not fork_:  # new_seg is initialized with initialized blob
            blob = [s, [0, 0, 0, 0, 0, 0], [], 1, [y, x0, xn]]  # s, [I, G, Dy, Dx, L], seg_, open_segments, box
            new_seg = [y, I, G, Dy, Dx, L, 1, [P], blob, 0]  # y0, I, G, Dy, Dx, N, L, Ly, Py_, blob, roots
            blob[2].append(new_seg)
        else:
            if len(fork_) == 1 and fork_[0][3] == 1:  # P has one fork and that fork has one root
                new_seg = fork_[0]

                Is, Gs, Dys, Dxs, Ls, Ly = new_seg[1:-3]  # fork segment params, P is merged into segment:
                new_seg[1:-3] = [Is + I, Gs + G, Dys + Dy, Dxs + Dx, Ls + L, Ly + 1]
                new_seg[-3].append(P)  # Py_: vertical buffer of Ps
                new_seg[-1] = 0  # reset roots
                blob = new_seg[-2]

            else:  # if > 1 forks, or 1 fork that has > 1 roots:
                blob = fork_[0][-2]
                new_seg = [y, I, G, Dy, Dx, L, 1, [P], blob, 0]  # new_seg is initialized with fork blob
                blob[2].append(new_seg)  # segment is buffered into blob

                if len(fork_) > 1:  # merge blobs of all forks
                    if fork_[0][-1] == 1:  # if roots == 1: fork hasn't been terminated
                        form_blob(fork_[0], frame)  # merge seg of 1st fork into its blob

                    for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                        if fork[-1] == 1:
                            form_blob(fork, frame)

                        if not fork[-2] is blob:
                            params, seg_, open_segs, box = fork[-2][1:]  # merged blob, omit sign
                            blob[1] = [par1 + par2 for par1, par2 in zip(blob[1], params)]  # sum merging blobs
                            blob[3] += open_segs
                            blob[4][0] = min(blob[4][0], box[0])  # extend box y0
                            blob[4][1] = min(blob[4][1], box[1])  # extend box x0
                            blob[4][2] = max(blob[4][2], box[2])  # extend box xn
                            for seg in seg_:
                                if not seg is fork:
                                    seg[-2] = blob  # blobs in other forks are references to blob in the first fork
                                    blob[2].append(seg)  # buffer of merged root segments
                            fork[-2] = blob
                            blob[2].append(fork)
                        blob[3] -= 1  # open_segments -= 1: shared with merged blob

        blob[4][1] = min(blob[4][1], x0)  # extend box x0
        blob[4][2] = max(blob[4][2], xn)  # extend box xn
        new_seg_.append(new_seg)

    return new_seg_

    # ---------- form_seg_() end --------------------------------------------------------------------------------------------


def form_blob(term_seg, frame):  # terminated segment is merged into continued or initialized blob (all connected segments)

    y0s, Is, Gs, Dys, Dxs, Ls, Lys, Py_, blob, roots = term_seg
    I, G, Dy, Dx, L, Ly = blob[1]
    blob[1] = [I + Is, G + Gs, Dy + Dys, Dx + Dxs, L + Ls, Ly + Lys]
    blob[-2] += roots - 1  # number of open segments

    if blob[-2] == 0:  # if open_segments == 0: blob is terminated and packed in frame
        s, [I, G, Dy, Dx, L, Ly], seg_, open_segs, [y0, x0, xn] = blob

        yn = y0s + Lys  # yn from last segment
        mask = np.zeros((yn - y0, xn - x0), dtype=bool)  # local map of blob
        new_seg_ = []
        for seg in seg_:
            y0s, Is, Gs, Dys, Dxs, Ls, Lys, Py_ = seg[:-2]  # blob and roots are ignored
            seg = Segment(y0s, Is, Gs, Dys, Dxs, Ls, Lys, Py_)  # convert to Segment namedtuple
            new_seg_.append(seg)  # add segment to blob as namedtuple
            for y, P in enumerate(seg.Py_, start=seg.y):
                Pxn = P.x0 + P.L
                mask[y - y0, P.x0 - x0:Pxn - x0] = True

        del seg_

        frame[0] += I
        frame[1] += G
        frame[2] += Dy
        frame[3] += Dx
        frame[4].append(Blob(Dert=[G, None, Dy, Dx, L, Ly],  # core Layer of current blob, A is None for g_Dert
                             sign=s,  # current g | ga sign
                             rng=rng,  # comp range
                             map=map,  # boolean map of blob to compute overlap
                             box=(y0, yn, x0, xn),  # boundary box
                             seg_=new_seg_,  # references down blob formation tree, in vertical (horizontal) order
                             dert__=[],
                             sub_blob_=[],  # ref to sub_blob derivation tree, sub_blob structure = blob structure
                             lLayers=[],  # summed reps of lower layers across sub_blob derivation tree
                             root_blob=blob,  # ref for feedback of all Derts params summed in sub_blobs
                             hLayers=[I]  # higher Dert params += higher-dert params, starting with I
                             ))
        del blob

    # ---------- form_blob() end ----------------------------------------------------------------------------------------


# ************ PROGRAM BODY *********************************************************************************************
if __name__ == '__main__':

    # Load inputs --------------------------------------------------------------------
    from utils import imread

    image = imread('./../images/raccoon_eye.jpg').astype(int)

    # Main ---------------------------------------------------------------------------

    start_time = time()
    frame_of_blobs = image_to_blobs(image)
    # frame_of_blobs = intra_blob(frame_of_blobs)  # evaluate for deeper clustering inside each blob, recursively

    # DEBUG --------------------------------------------------------------------------
    if DEBUG:
        from utils import *

        draw('./../debug/root_blobs', map_frame(frame_of_blobs))

        f_angle = 0b01
        f_derive = 0b10
        # from intra_blob_test import intra_blob
        # intra_blob(frame_of_blobs[1])

    # END DEBUG -----------------------------------------------------------------------

    end_time = time() - start_time
    print(end_time)
    # ************ PROGRAM BODY END ******************************************************************************************