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

from time import time
from collections import deque, defaultdict

import numpy as np
import numpy.ma as ma

from utils import imread

# -----------------------------------------------------------------------------
# Constants

MAX_G = 256 # 721.2489168102785 without normalization.

# -----------------------------------------------------------------------------
# Adjustable parameters
image_path = "./../images/raccoon_eye.jpg"
kwidth = 3 # Declare initial kernel size. Tested values are 2 or 3.
ave = 50
rng = int(kwidth == 3)
DEBUG = True

assert kwidth in (2, 3)

# -----------------------------------------------------------------------------
# Functions

def image_to_blobs(image):  # root function, postfix '_' denotes array vs element, prefix '_' denotes higher- vs lower- line variable

    dert__ = comp_pixel(image)  # vertically and horizontally bilateral comparison of adjacent pixels
    frame = dict(
        rng=1,
        dert__=dert__,
        Dert = dict(I=0, G=0, Dy=0, Dx=0),
        blob_=[],
    )

    seg_ = deque()  # buffer of running segments
    height, width = image.shape

    for y in range(height - kwidth + 1):  # first and last row are discarded
        P_ = form_P_(dert__[:, y].T)  # horizontal clustering
        P_ = scan_P_(P_, seg_, frame)
        seg_ = form_seg_(y, P_, frame)

    while seg_:  form_blob(seg_.popleft(), frame)  # frame ends, last-line segs are merged into their blobs
    return frame  # frame of 2D patterns


def comp_pixel(image):  # comparison between pixel and its neighbours within kernel, for the whole image

    # Initialize variables:
    if kwidth == 2:

        # Compare:
        dy__ = (image[1:, 1:] - image[:-1, 1:]) + (image[1:, :-1] - image[:-1, :-1]) * 0.5
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
        d___ = np.array(list(
            map(lambda trans_slices: image[trans_slices] - image[1:-1, 1:-1],
                [
                    (slice(None, -2), slice(None, -2)),
                    (slice(None, -2), slice(1, -1)),
                    (slice(None, -2), slice(2, None)),
                    (slice(1, -1), slice(2, None)),
                    (slice(2, None), slice(2, None)),
                    (slice(2, None), slice(1, -1)),
                    (slice(2, None), slice(None, -2)),
                    (slice(1, -1), slice(None, -2)),
                ]
            )
        )).swapaxes(0, 2).swapaxes(0, 1)

        # Decompose differences:
        dy__ = (d___ * ycoef).sum(axis=2)
        dx__ = (d___ * xcoef).sum(axis=2)

        # Sum pixel values:
        p__ = image[1:-1, 1:-1]

    # Compute gradient magnitudes per kernel:
    g__ = np.hypot(dy__, dx__) * 0.354801226089485

    return ma.around(ma.stack((p__, g__, dy__, dx__), axis=0))


def form_P_(dert_):  # horizontally cluster and sum consecutive pixels and their derivatives into Ps

    P_ = deque()  # row of Ps
    I, G, Dy, Dx, L, x0 = *dert_[0], 1, 0  # P params = first dert .
    G -= ave
    M = -G - ave
    _s = G > 0  # sign

    for x, (i, g, dy, dx) in enumerate(dert_[1:], start=1):
        vg = g - ave
        vm = -vg - ave
        s = vg > 0
        if s != _s:  # P is terminated and new P is initialized
            P = dict(I=I, G=G, M=M, Dy=Dy, Dx=Dx, L=L,
                     x0=x0, dert_=dert_[x0:x0+L], sign=_s)
            P_.append(P)
            I, G, M, Dy, Dx, L, x0 = 0, 0, 0, 0, 0, 0, x

        # Accumulate P params:
        I += i
        G += vg
        M += vm
        Dy += dy
        Dx += dx
        L += 1
        _s = s  # prior sign

    P = dict(I=I, G=G, M=M, Dy=Dy, Dx=Dx, L=L,
             x0=x0, dert_=dert_[x0:x0 + L], sign=_s)
    P_.append(P)    # last P in row
    return P_


def scan_P_(P_, seg_, frame):  # integrate x overlaps (forks) between same-sign Ps and _Ps into blob segments

    new_P_ = deque()

    if P_ and seg_:  # if both are not empty
        P = P_.popleft()  # input-line Ps
        seg = seg_.popleft()  # higher-line segments,
        _P = seg['Py_'][-1]  # last element of each segment is higher-line P
        fork_ = []

        while True:
            x0 = P['x0']  # first x in P
            xn = x0 + P['L']  # first x in next P
            _x0 = _P['x0']  # first x in _P
            _xn = _x0 + _P['L']  # first x in next _P

            if (P['sign'] == seg['sign']
                and _x0 < xn and x0 < _xn): # Test for sign match and x overlap.
                seg['roots'] += 1  # roots
                fork_.append(seg)  # P-connected segments are buffered into fork_

            if xn < _xn:  # _P overlaps next P in P_
                new_P_.append((P, fork_))
                fork_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:  # terminate loop
                    if seg['roots'] != 1:  # if roots != 1: terminate seg
                        form_blob(seg, frame)
                    break
            else:  # no next-P overlap
                if seg['roots'] != 1:  # if roots != 1: terminate seg
                    form_blob(seg, frame)

                if seg_:  # load next _P
                    seg = seg_.popleft()
                    _P = seg['Py_'][-1]
                else:  # if no seg left: terminate loop
                    new_P_.append((P, fork_))
                    break

    while P_:  # terminate Ps and segs that continue at line's end
        new_P_.append((P_.popleft(), []))  # no fork
    while seg_:
        form_blob(seg_.popleft(), frame)  # roots always == 0

    return new_P_


def form_seg_(y, P_, frame):
    """Convert or merge every P into segment, merge blobs."""
    new_seg_ = deque()

    while P_:
        P, fork_ = P_.popleft()

        s = P.pop('sign')
        I, G, M, Dy, Dx, L, x0, dert_ = P.values()
        xn = x0 + L     # next-P x0
        if not fork_:   # new_seg is initialized with initialized blob
            blob = dict(Dert=dict(I=0, G=0, M=0, Dy=0, Dx=0, S=0, Ly=0),
                        box=[y, x0, xn],
                        seg_=[],
                        sign=s,
                        open_segments=1)
            new_seg = dict(I=I, G=G, M=M, Dy=0, Dx=Dx, S=L, Ly=1,
                           y0=y, Py_=[P], blob=blob, roots=0, sign=s)
            blob['seg_'].append(new_seg)
        else:
            if len(fork_) == 1 and fork_[0]['roots'] == 1:  # P has one fork and that fork has one root
                new_seg = fork_[0]

                # Fork segment params, P is merged into segment:
                accum_Dert(new_seg,
                           # Params to update:
                           I=I, G=G, M=M, Dy=Dy, Dx=Dx, S=L, Ly=1)

                new_seg['Py_'].append(P)  # Py_: vertical buffer of Ps
                new_seg['roots'] = 0  # reset roots
                blob = new_seg['blob']

            else:  # if > 1 forks, or 1 fork that has > 1 roots:
                blob = fork_[0]['blob']
                new_seg = dict(I=I, G=G, M=M, Dy=0, Dx=Dx, S=L, Ly=1, # new_seg is initialized with fork blob
                               y0=y, Py_=[P], blob=blob, roots=0, sign=s)
                blob['seg_'].append(new_seg)  # segment is buffered into blob

                if len(fork_) > 1:  # merge blobs of all forks
                    if fork_[0]['roots'] == 1:  # if roots == 1: fork hasn't been terminated
                        form_blob(fork_[0], frame)  # merge seg of 1st fork into its blob

                    for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                        if fork['roots'] == 1:
                            form_blob(fork, frame)

                        if not fork['blob'] is blob:
                            Dert, box, seg_, s, open_segs = fork['blob'].values()  # merged blob
                            I, G, M, Dy, Dx, S, Ly = Dert.values()
                            accum_Dert(blob['Dert'],
                                       # Params to update:
                                       I=I, G=G, M=M, Dy=Dy, Dx=Dx, S=S, Ly=Ly)
                            blob['open_segments'] += open_segs
                            blob['box'][0] = min(blob['box'][0], box[0])  # extend box y0
                            blob['box'][1] = min(blob['box'][1], box[1])  # extend box x0
                            blob['box'][2] = max(blob['box'][2], box[2])  # extend box xn
                            for seg in seg_:
                                if not seg is fork:
                                    seg['blob'] = blob  # blobs in other forks are references to blob in the first fork.
                                    blob['seg_'].append(seg)  # buffer of merged root segments.
                            fork['blob'] = blob
                            blob['seg_'].append(fork)
                        blob['open_segments'] -= 1  # Shared with merged blob.

        blob['box'][1] = min(blob['box'][1], x0)  # extend box x0
        blob['box'][2] = max(blob['box'][2], xn)  # extend box xn
        new_seg_.append(new_seg)

    return new_seg_


def form_blob(seg, frame):  # terminated segment is merged into continued or initialized blob (all connected segments)

    blob = terminate_segment(seg)

    if blob['open_segments'] == 0:  # if open_segments == 0: blob is terminated and packed in frame
        terminate_blob(blob, seg, frame)


def terminate_segment(seg):
    I, G, M, Dy, Dx, S, Ly, y0, Py_, blob, roots, sign = seg.values()
    accum_Dert(blob['Dert'],
                # Params to update:
                I=I, G=G, M=M, Dy=Dy, Dx=Dx, S=S, Ly=Ly)
    blob['open_segments'] += roots - 1  # number of open segments
    return blob


def terminate_blob(blob, last_seg, frame):

    Dert, [y0, x0, xn], seg_, s, open_segs = blob.values()

    yn = last_seg['y0'] + last_seg['Ly'] # Compute yn.

    mask = np.ones((yn - y0, xn - x0), dtype=bool)  # local map of blob
    for seg in seg_:
        seg.pop('sign')
        seg.pop('roots')
        for y, P in enumerate(seg['Py_'], start=seg['y0']-y0):
            x_start = P['x0'] - x0
            x_stop = x_start + P['L']
            mask[y, x_start:x_stop] = False

    dert__ = frame['dert__'][:, y0:yn, x0:xn]
    dert__.mask[:] = mask
    blob.pop('open_segments')
    blob.update(box=(y0, yn, x0, xn),  # boundary box
                dert__=dert__,
                # Deprecated params: (dert__ and box are sufficient)
                # slices=(Ellipsis, slice(y0, yn), slice(x0, xn)),
                # mask=mask,
                root_fork=frame, # Equivalent of fork in lower layers.
                fork_=defaultdict(list), # Contain sub-blobs that belong to this blob.
                )

    # Update frame:
    frame['Dert'].update(
        I=frame['Dert']['I'] + blob['Dert']['I'],
        G=frame['Dert']['G'] + blob['Dert']['G'],
        Dy=frame['Dert']['Dy'] + blob['Dert']['Dy'],
        Dx=frame['Dert']['Dx'] + blob['Dert']['Dx'],
    )

    frame['blob_'].append(blob)

# -----------------------------------------------------------------------------
# Utilities

def accum_Dert(Dert : dict, **params) -> None:
    Dert.update({param:Dert[param]+value for param, value in params.items()})

# -----------------------------------------------------------------------------
# Main

if __name__ == '__main__':
    image = imread(image_path).astype(int)

    start_time = time()
    frame_of_blobs = image_to_blobs(image)
    # frame_of_blobs = intra_blob(frame_of_blobs)  # evaluate for deeper clustering inside each blob, recursively

    # DEBUG -------------------------------------------------------------------
    if DEBUG:
        from utils import imwrite, map_frame
        imwrite("./../visualization/images/blobs", map_frame(frame_of_blobs))

        # from intra_blob_test import intra_blob
        # intra_blob(frame_of_blobs[1])

    # END DEBUG ---------------------------------------------------------------

    end_time = time() - start_time
    print(end_time)