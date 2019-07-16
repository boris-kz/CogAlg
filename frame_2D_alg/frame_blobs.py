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
from collections import deque, namedtuple

import numpy as np
import intra_blob

# -----------------------------------------------------------------------------
# Structures

Dert = namedtuple('Dert', 'G, A, Dy, Dx, L, Ly')
Pattern = namedtuple('Pattern', 'sign, x0, I, G, Dy, Dx, L, dert_')
Segment = namedtuple('Segment', 'y, I, G, Dy, Dx, L, Ly, Py_')
Blob = namedtuple('Blob', 'Dert, sign, rng, box, mask, seg_, dert__, sub_blob_, lLayers, root_blob, hLayers')
Frame = namedtuple('Frame', 'I, G, Dy, Dx, blob_, i__, dert__')

# -----------------------------------------------------------------------------
# Adjustable parameters
image_path = "./../images/raccoon.jpg"
kwidth = 3 # Declare initial kernel size. Tested values are 2 or 3.
ave = 20 + 60 * (kwidth == 3)
rng = int(kwidth == 3)
DEBUG = True

assert kwidth in (2, 3)

# -----------------------------------------------------------------------------
# Functions

def image_to_blobs(image):  # root function, postfix '_' denotes array vs element, prefix '_' denotes higher- vs lower- line variable

    i__, dert__ = comp_pixel(image)  # vertically and horizontally bilateral comparison of adjacent pixels
    frame = dict(I=0, G=0, Dy=0, Dx=0, blob_=[], i__=i__, dert__=dert__) # params, blob_, dert__
    seg_ = deque()  # buffer of running segments

    height, width = image.shape
    # P__ = intra_blob.form_P__(0, i__[0], dert__, ave)
    # for y, P_ in enumerate(P__):
    for y in range(height - kwidth + 1):  # first and last row are discarded
        P_ = form_P_(i__[0, y], dert__[:, y].T)  # horizontal clustering
        P_ = scan_P_(P_, seg_, form_blob_func=form_blob, frame=frame)
        seg_ = form_seg_(y, P_, form_blob_func=form_blob, frame=frame)

    while seg_:  form_blob(seg_.popleft(), frame=frame)  # frame ends, last-line segs are merged into their blobs
    return frame  # frame of 2D patterns


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
            P = dict(sign=_s, x0=x0, I=I, G=G,
                     Dy=Dy, Dx=Dx, L=L, dert_=dert_[x0:x0+L])
            P_.append(P)
            x0, I, G, Dy, Dx, L = x, 0, 0, 0, 0, 0

        # accumulate P params:
        I += i
        G += vg
        Dy += dy
        Dx += dx
        L += 1
        _s = s  # prior sign

    P = dict(sign=_s, x0=x0, I=I, G=G,
             Dy=Dy, Dx=Dx, L=L, dert_=dert_[x0:x0 + L])
    P_.append(P)    # last P in row
    return P_


def scan_P_(P_, seg_, form_blob_func, **kwargs):  # integrate x overlaps (forks) between same-sign Ps and _Ps into blob segments

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

            if P['sign'] == _P['sign'] and _x0 < xn and x0 < _xn:  # test for sign match and x overlap
                seg['roots'] += 1  # roots
                fork_.append(seg)  # P-connected segments are buffered into fork_

            if xn < _xn:  # _P overlaps next P in P_
                new_P_.append((P, fork_))
                fork_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:  # terminate loop
                    if seg['roots'] != 1:  # if roots != 1: terminate seg
                        form_blob_func(seg, **kwargs)
                    break
            else:  # no next-P overlap
                if seg['roots'] != 1:  # if roots != 1: terminate seg
                    form_blob_func(seg, **kwargs)

                if seg_:  # load next _P
                    seg = seg_.popleft()
                    _P = seg['Py_'][-1]
                else:  # if no seg left: terminate loop
                    new_P_.append((P, fork_))
                    break

    while P_:  # terminate Ps and segs that continue at line's end
        new_P_.append((P_.popleft(), []))  # no fork
    while seg_:
        form_blob_func(seg_.popleft(), **kwargs)  # roots always == 0

    return new_P_


def form_seg_(y, P_, form_blob_func, **kwargs):
    """Convert or merge every P into segment, merge blobs."""
    new_seg_ = deque()

    while P_:
        P, fork_ = P_.popleft()

        s, x0, I, G, Dy, Dx, L, dert_ = P.values()
        xn = x0 + L     # next-P x0
        if not fork_:  # new_seg is initialized with initialized blob
            blob = dict(Dert=dict(I=0, G=0, Dy=0, Dx=0, L=0, Ly=0),
                        sign=s,
                        box=[y, x0, xn],
                        seg_=[],
                        open_segments=1)
            new_seg = dict(y0=y, I=I, G=G, Dy=0, Dx=Dx, L=L, Ly=1,
                           Py_=[P], blob=blob, roots=0)
            blob['seg_'].append(new_seg)
        else:
            if len(fork_) == 1 and fork_[0]['roots'] == 1:  # P has one fork and that fork has one root
                new_seg = fork_[0]
                # y0 = y, I = I, G = G, Dy = 0, Dx = Dx, L = L, Ly = 1,
                # Py_ = [P], blob = blob, roots = 0
                _, Is, Gs, Dys, Dxs, Ls, Ly, _, _, _ = new_seg.values()    # fork segment params, P is merged into segment:
                new_seg.update(I=Is + I,
                               G=Gs + G,
                               Dy=Dys + Dy,
                               Dx=Dxs + Dx,
                               L=Ls + L,
                               Ly=Ly + 1)
                new_seg['Py_'].append(P)  # Py_: vertical buffer of Ps
                new_seg['roots'] = 0  # reset roots
                blob = new_seg['blob']

            else:  # if > 1 forks, or 1 fork that has > 1 roots:
                blob = fork_[0]['blob']
                new_seg = dict(y0=y, I=I, G=G, Dy=0, Dx=Dx, L=L, Ly=1,
                               Py_=[P], blob=blob, roots=0) # new_seg is initialized with fork blob
                blob['seg_'].append(new_seg)  # segment is buffered into blob

                if len(fork_) > 1:  # merge blobs of all forks
                    if fork_[0]['roots'] == 1:  # if roots == 1: fork hasn't been terminated
                        form_blob_func(fork_[0], **kwargs)  # merge seg of 1st fork into its blob

                    for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                        if fork['roots'] == 1:
                            form_blob_func(fork, **kwargs)

                        if not fork['blob'] is blob:
                            Dert, s, box, seg_, open_segs = fork['blob'].values()  # merged blob
                            Is, Gs, Dys, Dxs, Ls, Lys = Dert.values()
                            I, G, Dy, Dx, L, Ly = blob['Dert'].values()
                            blob['Dert'].update(I=Is + I,
                                                G=Gs + G,
                                                Dy=Dys + Dy,
                                                Dx=Dxs + Dx,
                                                L=Ls + L,
                                                Ly=Lys + Ly)  # sum merging blobs
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


def form_blob(term_seg, frame):  # terminated segment is merged into continued or initialized blob (all connected segments)

    y0s, Is, Gs, Dys, Dxs, Ls, Lys, Py_, blob, roots = term_seg.values()
    I, G, Dy, Dx, L, Ly = blob['Dert'].values()
    blob['Dert'].update(I=Is + I,
                        G=Gs + G,
                        Dy=Dys + Dy,
                        Dx=Dxs + Dx,
                        L=Ls + L,
                        Ly=Lys + Ly)
    blob['open_segments'] += roots - 1  # number of open segments

    if blob['open_segments'] == 0:  # if open_segments == 0: blob is terminated and packed in frame
        Dert, s, [y0, x0, xn], seg_, open_segs = blob.values()

        yn = y0s + Lys  # yn from last segment
        mask = np.ones((yn - y0, xn - x0), dtype=bool) # local map of blob
        for seg in seg_:
            seg.pop('roots')
            for y, P in enumerate(seg['Py_'], start=seg['y0']):
                x_start = P['x0'] - x0
                x_stop = x_start + P['L']
                mask[y - y0, x_start:x_stop] = False

        I = Dert.pop('I')
        G, Dy, Dx, L, Ly = Dert.values()
        # Update frame:
        frame.update(I=frame['I']+I,
                     G=frame['G']+G,
                     Dy=frame['Dy']+Dy,
                     Dx=frame['Dx']+Dx)

        blob.pop('open_segments')
        blob.update(box=(y0, yn, x0, xn), # boundary box
                    slices=(Ellipsis, slice(y0, yn), slice(x0, xn)),
                    rng=rng,
                    mask=mask,
                    dert___=[frame['i__'], frame['dert__']],
                    hDerts=np.array([I, 0, 0, 0, 0]),
                    root_blob=None,
                    Layers={},
                    )
        frame['blob_'].append(blob)

        """
        frame['blob'].append(Blob(Dert = [G, None, Dy, Dx, L, Ly],  # core Layer of current blob, A is None for g_Dert
                             sign=s,  # current g | ga sign
                             rng =rng,  # comp range
                             mask=mask,   # boolean mask of blob to compute overlap
                             box=(y0, yn, x0, xn),  # boundary box
                             seg_=new_seg_,  # references down blob formation tree, in vertical (horizontal) order
                             dert__=frame.dert__,
                             sub_blob_=[],   # ref to sub_blob derivation tree, sub_blob structure = blob structure
                             lLayers=[],     # summed reps of lower layers across sub_blob derivation tree
                             root_blob=[blob],  # ref for feedback of all Derts params summed in sub_blobs
                             hLayers=[I]     # higher Dert params += higher-dert params, starting with I
                           ))
        """


# -----------------------------------------------------------------------------
# Main

if __name__ == '__main__':
    from utils import imread
    image = imread(image_path).astype(int)

    start_time = time()
    frame_of_blobs = image_to_blobs(image)
    # frame_of_blobs = intra_blob(frame_of_blobs)  # evaluate for deeper clustering inside each blob, recursively

    # DEBUG -------------------------------------------------------------------
    if DEBUG:
        from utils import *
        draw('./../debug/root_blobs', map_frame(frame_of_blobs))

        F_ANGLE = 0b01
        F_DERIV = 0b10
        # from intra_blob_test import intra_blob
        # intra_blob(frame_of_blobs[1])

    # END DEBUG ---------------------------------------------------------------

    end_time = time() - start_time
    print(end_time)