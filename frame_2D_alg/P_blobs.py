'''
    P_blobs is a terminal fork of intra_blob, will call comp_g and then comp_P (edge tracing) per terminated stack.

    Pixel-level parameters are accumulated in contiguous spans of same-sign gradient, first horizontally then vertically.
    Horizontal spans are Ps: 1D patterns, and vertical spans are first stacks of Ps, then blobs of stacks.

    This processing adds a level of encoding per row y, defined relative to y of current input row, with top-down scan:
    1Le, line y-1: form_P( dert_) -> 1D pattern P: contiguous row segment, a slice of a blob
    2Le, line y-2: scan_P_(P, hP) -> hP, up_connect_, down_connect_count: vertical connections per stack of Ps
    3Le, line y-3: form_stack(hP, stack) -> stack: merge vertically-connected _Ps into non-forking stacks of Ps
    4Le, line y-4+ stack depth: form_blob(stack, blob): merge connected stacks in blobs referred by up_connect_, recursively

    Higher-row elements include additional parameters, derived while they were lower-row elements.
    Resulting blob structure (fixed set of parameters per blob):
    - Dert: summed pixel-level dert params, Dx, surface area S, vertical depth Ly
    - sign = s: sign of gradient deviation
    - box  = [y0, yn, x0, xn],
    - dert__,  # 2D array of pixel-level derts: (p, dy, dx, g, m) tuples
    - stack_,  # contains intermediate blob composition structures: stacks and Ps, not meaningful on their own
    ( intra_blob structure extends Dert, adds fork params and sub_layers)

    Blob is 2D pattern: connectivity cluster defined by the sign of gradient deviation. Gradient represents 2D variation
    per pixel. It is used as inverse measure of partial match (predictive value) because direct match (min intensity)
    is not meaningful in vision. Intensity of reflected light doesn't correlate with predictive value of observed object
    (predictive value is physical density, hardness, inertia that represent resistance to change in positional parameters)

    Comparison range is fixed for each layer of search, to enable encoding of input pose parameters: coordinates, dimensions,
    orientation. These params are essential because value of prediction = precision of what * precision of where.
    Clustering is by nearest-neighbor connectivity only, to avoid overlap among the blobs.

    frame_blobs is a complex function with a simple purpose: to sum pixel-level params in blob-level params. These params
    were derived by pixel cross-comparison (cross-correlation) to represent predictive value per pixel, so they are also
    predictive on a blob level, and should be cross-compared between blobs on the next level of search and composition.
    Old diagrams are on https://kwcckw.github.io/CogAlg/
'''
"""
usage: frame_blobs_find_adj.py [-h] [-i IMAGE] [-v VERBOSE] [-n INTRA] [-r RENDER]
                      [-z ZOOM]
optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        path to image file
  -v VERBOSE, --verbose VERBOSE
                        print details, useful for debugging
  -n INTRA, --intra INTRA
                        run intra_blobs after frame_blobs
  -r RENDER, --render RENDER
                        render the process
  -z ZOOM, --zoom ZOOM  zooming ratio when rendering
"""

from time import time
from collections import deque
from pathlib import Path
import sys
import numpy as np

from class_cluster import ClusterStructure, NoneType
from class_bind import AdjBinder
# from comp_pixel import comp_pixel
from class_stream import BlobStreamer
from utils import (
    pairwise,
    imread, )

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback


class CP(ClusterStructure):
    I = int  # default type at initialization
    Dy = int
    Dx = int
    G = int
    M = int
    Dyy = int
    Dyx = int
    Dxy = int
    Dxx = int
    Ga = int
    Ma = int
    L = int
    x0 = int
    sign = NoneType
    dert_ = list


class Cstack(ClusterStructure):
    I = int
    Dy = int
    Dx = int
    G = int
    M = int
    Dyy = int
    Dyx = int
    Dxy = int
    Dxx = int
    Ga = int
    Ma = int
    S = int
    Ly = int
    y0 = int
    Py_ = list
    blob = NoneType
    down_connect_cnt = int
    sign = NoneType


class CBlob(ClusterStructure):
    Dert = dict
    box = list
    stack_ = list
    sign = NoneType
    open_stacks = int
    root_dert__ = object
    dert__ = object
    mask = object
    adj_blobs = list
    fopen = bool
    margin = list


# Functions:
# prefix '_' denotes higher-line variable or structure, vs. same-type lower-line variable or structure
# postfix '_' denotes array name, vs. same-name elements of that array


def P_blobs(dert__, mask, crit__, verbose=False, render=False):
    frame = dict(rng=1, dert__=dert__, mask=None, I=0, Dy=0, Dx=0, G=0, M=0, Dyy=0, Dyx=0, Dxy=0, Dxx=0, Ga=0, Ma=0, blob__=[])
    stack_ = deque()  # buffer of running vertical stacks of Ps
    height, width = dert__[0].shape

    if render:
        def output_path(input_path, suffix):
            return str(Path(input_path).with_suffix(suffix))

        streamer = BlobStreamer(CBlob, dert__[1],
                                record_path=output_path(arguments['image'],
                                                        suffix='.im2blobs.avi'))
    stack_binder = AdjBinder(Cstack)

    if verbose:
        start_time = time()
        print("Converting to image to blobs...")

    for y, dert_ in enumerate(zip(*dert__)):  # first and last row are discarded
        if verbose:
            print(f"\rProcessing line {y + 1}/{height}, ", end="")
            print(f"{len(frame['blob__'])} blobs converted", end="")
            sys.stdout.flush()

        P_binder = AdjBinder(CP)  # binder needs data about clusters of the same level
        P_ = form_P_(zip(*dert_), crit__[y], mask[y], P_binder)  # horizontal clustering

        if render:
            render = streamer.update_blob_conversion(y, P_)
        P_ = scan_P_(P_, stack_, frame, P_binder)  # vertical clustering, adds P up_connects and _P down_connect_cnt
        stack_ = form_stack_(P_, frame, y)
        stack_binder.bind_from_lower(P_binder)

    while stack_:  # frame ends, last-line stacks are merged into their blobs
        form_blob(stack_.popleft(), frame)

    blob_binder = AdjBinder(CBlob)
    blob_binder.bind_from_lower(stack_binder)
    assign_adjacents(blob_binder)  # add adj_blobs to each blob

    if verbose:  # print out at the end
        nblobs = len(frame['blob__'])
        print(f"\rImage has been successfully converted to "
              f"{nblobs} blob{'s' if nblobs != 1 else 0} in "
              f"{time() - start_time:.3} seconds", end="")
        blob_ids = [blob_id for blob_id in range(CBlob.instance_cnt)]
        merged_percentage = len([*filter(lambda bid: CBlob.get_instance(bid) is None, blob_ids)]) / len(blob_ids)
        print(f"\nPercentage of merged blobs: {merged_percentage}")

    if render:  # rendering mode after blob conversion
        path = output_path(arguments['image'],
                           suffix='.im2blobs.jpg')
        streamer.end_blob_conversion(y, img_out_path=path)

    return frame  # frame of blobs

''' 
Parameterized connectivity clustering functions below:
- form_P sums dert params within P and increments its L: horizontal length.
- scan_P_ searches for horizontal (x) overlap between Ps of consecutive (in y) rows.
- form_stack combines these overlapping Ps into vertical stacks of Ps, with one up_P to one down_P
- form_blob merges terminated or forking stacks into blob, removes redundant representations of the same blob 
  by multiple forked P stacks, then checks for blob termination and merger into whole-frame representation.
dert: tuple of derivatives per pixel, initially (p, dy, dx, g), will be extended in intra_blob
Dert: params of cluster structures (P, stack, blob): summed dert params + dimensions: vertical Ly and area S
'''

def form_P_(idert_, crit_, mask_, binder):  # segment dert__ into P__, in horizontal ) vertical order

    P_ = deque()  # row of Ps
    s_ = crit_ > 0
    x0 = 0
    try:
        while mask_[x0]:  # skip until not masked
            next(idert_)
            x0 += 1
    except IndexError:
        return P_  # the whole line is masked, return an empty P

    dert_ = [*next(idert_)]  # get first dert, dert_ is a generator/iterator
    (I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma), L = dert_, 1  # initialize P params

    _s = s_[x0]
    _mask = mask_[x0]  # mask bit per dert

    for x, (p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma) in enumerate(idert_, start=x0 + 1):  # loop left to right in each row of derts
        mask = mask_[x]
        if ~mask:  # current dert is not masked
            s = s_[x]
            if ~_mask and s != _s:  # prior dert is not masked and sign changed
                # pack P
                # terminate and pack P:
                P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, sign=_s, dert_=dert_)
                P_.append(P)
                # initialize P params:
                # initialize new P params:
                I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, L, x0, dert_ = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, []
            elif _mask:
                # initialize new P params:
                I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, L, x0, dert_ = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, []
        # current dert is masked
        elif ~_mask:  # prior dert is not masked
            # pack P
            P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, sign=_s, dert_=dert_)
            P_.append(P)
            # initialize P params: (redundant)
            # I, iDy, iDx, G, Dy, Dx, M, L, x0 = 0, 0, 0, 0, 0, 0, 0, 0, x + 1

        if ~mask:  # accumulate P params:
            # accumulate P params:
            I += p
            Dy += dy
            Dx += dx
            G += g
            M += m
            Dyy += dyy
            Dyx += dyx
            Dxy += dxy
            Dxx += dxx
            Ga += ga
            Ma += Ma
            L += 1
            dert_.append([p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma])  # accumulate dert into dert_ if no sign change
            _s = s  # prior sign
        _mask = mask

    if ~_mask:  # terminate and pack last P in a row if prior dert is unmasked
        P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, sign=_s, dert_=dert_)
        P_.append(P)

    for _P, P in pairwise(P_):
        if _P.x0 + _P.L == P.x0:  # check if Ps are adjacents
            binder.bind(_P, P)

    return P_


def scan_P_(P_, stack_, frame, binder):  # merge P into higher-row stack of Ps which have same sign and overlap by x_coordinate
    '''
    Each P in P_ scans higher-row _Ps (in stack_) left-to-right, testing for x-overlaps between Ps and same-sign _Ps.
    Overlap is represented as up_connect in P and is added to down_connect_cnt in _P. Scan continues until P.x0 >= _P.xn:
    no x-overlap between P and next _P. Then P is packed into its up_connect stacks or initializes a new stack.
    After such test, loaded _P is also tested for x-overlap to the next P.
    If negative, a stack with loaded _P is removed from stack_ (buffer of higher-row stacks) and tested for down_connect_cnt==0.
    If so: no lower-row connections, the stack is packed into connected blobs (referred by its up_connect_),
    else the stack is recycled into next_stack_, for next-row run of scan_P_.
    It's a form of breadth-first flood fill, with connects as vertices per stack of Ps: a node in connectivity graph.
    '''
    next_P_ = deque()  # to recycle P + up_connect_ that finished scanning _P, will be converted into next_stack_

    if P_ and stack_:  # if both input row and higher row have any Ps / _Ps left

        P = P_.popleft()  # load left-most (lowest-x) input-row P
        stack = stack_.popleft()  # higher-row stacks
        _P = stack.Py_[-1]  # last element of each stack is higher-row P
        up_connect_ = []  # list of same-sign x-overlapping _Ps per P

        while True:  # while both P_ and stack_ are not empty

            x0 = P.x0  # first x in P
            xn = x0 + P.L  # first x in next P
            _x0 = _P.x0  # first x in _P
            _xn = _x0 + _P.L  # first x in next _P

            if stack.G > 0:  # check for overlaps in 8 directions, else a blob may leak through its external blob
                if _x0 - 1 < xn and x0 < _xn + 1:  # x overlap between loaded P and _P
                    if P.sign == stack.sign:  # sign match
                        stack.down_connect_cnt += 1
                        up_connect_.append(stack)  # buffer P-connected higher-row stacks into P' up_connect_
                    else:
                        binder.bind(_P, P)

            else:  # -G, check for orthogonal overlaps only: 4 directions, edge blobs are more selective
                if _x0 < xn and x0 < _xn:  # x overlap between loaded P and _P
                    if P.sign == stack.sign:  # sign match
                        stack.down_connect_cnt += 1
                        up_connect_.append(stack)  # buffer P-connected higher-row stacks into P' up_connect_
                    else:
                        binder.bind(_P, P)

            if (xn < _xn or  # _P overlaps next P in P_
                    xn == _xn and stack.sign):  # check in 8 directions
                next_P_.append((P, up_connect_))  # recycle _P for the next run of scan_P_
                up_connect_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:  # terminate loop
                    if stack.down_connect_cnt != 1:  # terminate stack, merge it into up_connects' blobs
                        form_blob(stack, frame)
                    break
            else:  # no next-P overlap
                if stack.down_connect_cnt != 1:  # terminate stack, merge it into up_connects' blobs
                    form_blob(stack, frame)
                if stack_:  # load stack with next _P
                    stack = stack_.popleft()
                    _P = stack.Py_[-1]
                else:  # no stack left: terminate loop
                    next_P_.append((P, up_connect_))
                    break

    # terminate Ps and stacks that continue at row's end
    while P_:
        next_P_.append((P_.popleft(), []))  # no up_connect
    while stack_:
        form_blob(stack_.popleft(), frame)  # down_connect_cnt==0

    return next_P_  # each element is P + up_connect_ refs


def form_stack_(P_, frame, y):  # Convert or merge every P into its stack of Ps, merge blobs

    next_stack_ = deque()  # converted to stack_ in the next run of scan_P_

    while P_:
        P, up_connect_ = P_.popleft()
        I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, L, x0, s, dert_ = P.unpack()
        xn = x0 + L  # next-P x0
        if not up_connect_:
            # initialize new stack for each input-row P that has no connections in higher row, as in the whole top row:
            blob = CBlob(Dert=dict(I=0, Dy=0, Dx=0, G=0, M=0, Dyy=0, Dyx=0, Dxy=0, Dxx=0, Ga=0, Ma=0, S=0, Ly=0), box=[y, x0, xn], stack_=[], sign=s, open_stacks=1)
            new_stack = Cstack(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, S=L, Ly=1, y0=y, Py_=[P], blob=blob, down_connect_cnt=0, sign=s)
            new_stack.hid = blob.id
            # if stack.G - stack.Ga > ave * coeff * len(stack.Py):
            # comp_d_(stack)
            blob.stack_.append(new_stack)

        else:
            if len(up_connect_) == 1 and up_connect_[0].down_connect_cnt == 1:
                # P has one up_connect and that up_connect has one down_connect=P: merge P into up_connect stack:
                new_stack = up_connect_[0]
                new_stack.accumulate(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, S=L, Ly=1)
                new_stack.Py_.append(P)  # Py_: vertical buffer of Ps
                new_stack.down_connect_cnt = 0  # reset down_connect_cnt
                blob = new_stack.blob

            else:  # P has >1 up_connects, or 1 up_connect that has >1 down_connect_cnt:
                blob = up_connect_[0].blob
                # initialize new_stack with up_connect blob:
                new_stack = Cstack(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, S=L, Ly=1, y0=y, Py_=[P], blob=blob, down_connect_cnt=0, sign=s)
                new_stack.hid = blob.id
                blob.stack_.append(new_stack)  # stack is buffered into blob

                if len(up_connect_) > 1:  # merge blobs of all up_connects
                    if up_connect_[0].down_connect_cnt == 1:  # up_connect is not terminated
                        form_blob(up_connect_[0], frame)  # merge stack of 1st up_connect into its blob

                    for up_connect in up_connect_[1:len(up_connect_)]:  # merge blobs of other up_connects into blob of 1st up_connect
                        if up_connect.down_connect_cnt == 1:
                            form_blob(up_connect, frame)

                        if not up_connect.blob is blob:
                            Dert, box, stack_, s, open_stacks = up_connect.blob.unpack()[:5]  # merged blob
                            I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, S, Ly = Dert.values()
                            accum_Dert(blob.Dert, I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, S=S, Ly=Ly)
                            blob.open_stacks += open_stacks
                            blob.box[0] = min(blob.box[0], box[0])  # extend box y0
                            blob.box[1] = min(blob.box[1], box[1])  # extend box x0
                            blob.box[2] = max(blob.box[2], box[2])  # extend box xn
                            for stack in stack_:
                                if not stack is up_connect:
                                    stack.blob = blob  # blobs in other up_connects are refs to blob in first up_connect
                                    stack.hid = blob.id
                                    blob.stack_.append(stack)  # buffer of merged root stacks.

                            up_connect.blob = blob
                            up_connect.hid = blob.id
                            blob.stack_.append(up_connect)
                        blob.open_stacks -= 1  # overlap with merged blob.

        blob.box[1] = min(blob.box[1], x0)  # extend box x0
        blob.box[2] = max(blob.box[2], xn)  # extend box xn
        P.hid = new_stack.id
        next_stack_.append(new_stack)

    return next_stack_  # input for the next line of scan_P_


def form_blob(stack, frame):  # increment blob with terminated stack, check for blob termination and merger into frame

    I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, S, Ly, y0, Py_, blob, down_connect_cnt, sign = stack.unpack()
    # terminated stack is merged into continued or initialized blob (all connected stacks):
    accum_Dert(blob.Dert, I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, S=S, Ly=Ly)

    blob.open_stacks += down_connect_cnt - 1  # incomplete stack cnt + terminated stack down_connect_cnt - 1: stack itself
    # open stacks contain Ps of a current row and may be extended with new x-overlapping Ps in next run of scan_P_
    if blob.open_stacks == 0:  # number of incomplete stacks == 0: blob is terminated and packed in frame:
        last_stack = stack
        [y0, x0, xn], stack_, s, open_stacks = blob.unpack()[1:5]
        yn = last_stack.y0 + last_stack.Ly

        mask = np.ones((yn - y0, xn - x0), dtype=bool)  # mask box, then unmask Ps:
        for stack in stack_:
            for y, P in enumerate(stack.Py_, start=stack.y0 - y0):
                x_start = P.x0 - x0
                x_stop = x_start + P.L
                mask[y, x_start:x_stop] = False

        dert__ = tuple(derts[y0:yn, x0:xn] for derts in frame['dert__'])  # slice each dert array of the whole frame

        fopen = 0  # flag: blob on frame boundary
        if x0 == 0 or xn == frame['dert__'][0].shape[1] or y0 == 0 or yn == frame['dert__'][0].shape[0]:
            fopen = 1

        blob.root_dert__ = frame['dert__']
        blob.box = (y0, yn, x0, xn)
        blob.dert__ = dert__
        blob.mask = mask
        blob.adj_blobs = [[], 0, 0]
        blob.fopen = fopen

        frame.update(I=frame['I'] + blob.Dert['I'],
                     Dy=frame['Dy'] + blob.Dert['Dy'],
                     Dx=frame['Dx'] + blob.Dert['Dx'],
                     G=frame['G'] + blob.Dert['G'],
                     M=frame['M'] + blob.Dert['M'],
                     Dyy=frame['Dyy'] + blob.Dert['Dyy'],
                     Dyx=frame['Dyx'] + blob.Dert['Dyx'],
                     Dxy=frame['Dxy'] + blob.Dert['Dxy'],
                     Dxx=frame['Dxx'] + blob.Dert['Dxx'],
                     Ga=frame['Ga'] + blob.Dert['Ga'],
                     Ma=frame['Ma'] + blob.Dert['Ma'])

        frame['blob__'].append(blob)


def assign_adjacents(blob_binder):  # adjacents are connected opposite-sign blobs
    '''
    Assign adjacent blobs bilaterally according to adjacent pairs' ids in blob_binder.
    '''
    for blob_id1, blob_id2 in blob_binder.adj_pairs:
        assert blob_id1 < blob_id2
        blob1 = blob_binder.cluster_cls.get_instance(blob_id1)
        blob2 = blob_binder.cluster_cls.get_instance(blob_id2)

        y01, yn1, x01, xn1 = blob1.box
        y02, yn2, x02, xn2 = blob2.box

        if y01 < y02 and x01 < x02 and yn1 > yn2 and xn1 > xn2:
            pose1, pose2 = 0, 1  # 0: internal, 1: external
        elif y01 > y02 and x01 > x02 and yn1 < yn2 and xn1 < xn2:
            pose1, pose2 = 1, 0  # 1: external, 0: internal
        else:
            pose1 = pose2 = 2  # open, no need for fopen?

        # bilateral assignments
        blob1.adj_blobs[0].append((blob2, pose2))
        blob2.adj_blobs[0].append((blob1, pose1))
        blob1.adj_blobs[1] += blob2.Dert['S']
        blob2.adj_blobs[1] += blob1.Dert['S']
        blob1.adj_blobs[2] += blob2.Dert['G']
        blob2.adj_blobs[2] += blob1.Dert['G']


# -----------------------------------------------------------------------------
# Utilities

def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})

