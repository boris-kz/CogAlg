'''
    2D version of first-level core algorithm will have frame_blobs, intra_blob (recursive search within blobs), and comp_P.
    frame_blobs() forms parameterized blobs: contiguous areas of positive or negative deviation of gradient per pixel.    
    comp_pixel (lateral, vertical, diagonal) forms dert, queued in dert__: tuples of pixel + derivatives, over whole image.

    Then pixel-level and external parameters are accumulated in row segment Ps, vertical blob segment, and blobs,
    adding a level of encoding per row y, defined relative to y of current input row, with top-down scan:

    1Le, line y-1: form_P( dert_) -> 1D pattern P: contiguous row segment, a slice of a blob
    2Le, line y-2: scan_P_(P, hP) -> hP, up_connect_, down_connect_count: vertical connections per stack of Ps 
    3Le, line y-3: form_stack(hP, stack) -> stack: merge vertically-connected _Ps into non-forking stacks of Ps
    4Le, line y-4+ stack depth: form_blob(stack, blob): merge connected stacks in blobs referred by up_connect_, recursively

    Higher-row elements include additional parameters, derived while they were lower-row elements. Processing is bottom-up:
    from input-row to higher-row structures, sequential because blobs are irregular, not suited for matrix operations.
    Resulting blob structure (fixed set of parameters per blob): 

    - Dert = I, G, Dy, Dx, S, Ly: summed pixel-level dert params I, G, Dy, Dx, surface area S, vertical depth Ly
    - sign = s: sign of gradient deviation
    - box  = [y0, yn, x0, xn], 
    - dert__,  # 2D array of pixel-level derts: (p, g, dy, dx) tuples
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
    Please see diagrams of frame_blobs on https://kwcckw.github.io/CogAlg/
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
    imread, imwrite, map_frame_binary,
    WHITE, BLACK,
)

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback

class CP(ClusterStructure):
    I = int  # default type at initialization
    G = int
    Dy = int
    Dx = int
    L = int
    x0 = int
    sign = NoneType

class Cstack(ClusterStructure):
    I = int
    G = int
    Dy = int
    Dx = int
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

def comp_pixel(image):  # 2x2 pixel cross-correlation within image, as in edge detection operators
    # see comp_pixel_versions file for other versions and more explanation

    # input slices into sliding 2x2 kernel, each slice is a shifted 2D frame of grey-scale pixels:
    topleft__ = image[:-1, :-1]
    topright__ = image[:-1, 1:]
    bottomleft__ = image[1:, :-1]
    bottomright__ = image[1:, 1:]

    Gy__ = ((bottomleft__ + bottomright__) - (topleft__ + topright__))  # same as decomposition of two diagonal differences into Gy
    Gx__ = ((topright__ + bottomright__) - (topleft__ + bottomleft__))  # same as decomposition of two diagonal differences into Gx

    G__ = np.hypot(Gy__, Gx__)  # central gradient per kernel, between its four vertex pixels

    return (topleft__, G__, Gy__, Gx__)  # tuple of 2D arrays per param of dert (derivatives' tuple)
    # renamed dert__ = (p__, g__, dy__, dx__) for readability in functions below


def image_to_blobs(image, verbose=False, render=False):
    if verbose:
        start_time = time()
        print("Doing comparison...", end=" ")
    dert__ = comp_pixel(image)  # 2x2 cross-comparison / cross-correlation
    if verbose:
        print(f"Done in {(time() - start_time):f} seconds")

    frame = dict(rng=1, dert__=dert__, mask=None, I=0, G=0, Dy=0, Dx=0, blob__=[])
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
            print(f"\rProcessing line {y+1}/{height}, ", end="")
            print(f"{len(frame['blob__'])} blobs converted", end="")
            sys.stdout.flush()

        P_binder = AdjBinder(CP)  # binder needs data about clusters of the same level
        P_ = form_P_(zip(*dert_), P_binder)  # horizontal clustering

        if render:
            render = streamer.update_blob_conversion(y, P_)
        P_ = scan_P_(P_, stack_, frame, P_binder)  # vertical clustering, adds P up_connects and _P down_connect_cnt
        stack_ = form_stack_(P_, frame, y)
        stack_binder.bind_from_lower(P_binder)

    while stack_:  # frame ends, last-line stacks are merged into their blobs
        form_blob(stack_.popleft(), frame)

    blob_binder = AdjBinder(CBlob)
    blob_binder.bind_from_lower(stack_binder)
    assign_adjacent(blob_binder)  # add adj_blobs to each blob

    if verbose:  # print infos at the end
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

def form_P_(dert__, binder):  # horizontal clustering and summation of dert params into P params, per row of a frame
    # P is a segment of same-sign derts in horizontal slice of a blob

    P_ = deque()  # row of Ps
    I, G, Dy, Dx, L, x0 = *next(dert__), 1, 0  # initialize P params with 1st dert params
    G = int(G) - ave
    _s = G > 0  # sign
    for x, (p, g, dy, dx) in enumerate(dert__, start=1):    # dert__ is now a generator/iterator, no need for [1:]
        vg = int(g) - ave  # deviation of g
        s = vg > 0
        if s != _s:
            # terminate and pack P:
            P = CP(I=I, G=G, Dy=Dy, Dx=Dx, L=L, x0=x0, sign=_s)  # no need for dert_
            # initialize new P params:
            I, G, Dy, Dx, L, x0 = 0, 0, 0, 0, 0, x
            P_.append(P)
        # accumulate P params:
        I += p
        G += vg
        Dy += dy
        Dx += dx
        L += 1
        _s = s  # prior sign

    P = CP(I=I, G=G, Dy=Dy, Dx=Dx, L=L, x0=x0, sign=_s)  # last P in a row
    P_.append(P)

    for _P, P in pairwise(P_):
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
        I, G, Dy, Dx, L, x0, s = P.unpack()
        xn = x0 + L  # next-P x0
        if not up_connect_:
            # initialize new stack for each input-row P that has no connections in higher row, as in the whole top row:
            blob = CBlob(Dert=dict(I=0, G=0, Dy=0, Dx=0, S=0, Ly=0), box=[y, x0, xn], stack_=[], sign=s, open_stacks=1)
            new_stack = Cstack(I=I, G=G, Dy=0, Dx=Dx, S=L, Ly=1, y0=y, Py_=[P], blob=blob, down_connect_cnt=0, sign=s)
            new_stack.hid = blob.id
            blob.stack_.append(new_stack)

        else:
            if len(up_connect_) == 1 and up_connect_[0].down_connect_cnt == 1:
                # P has one up_connect and that up_connect has one down_connect=P: merge P into up_connect stack:
                new_stack = up_connect_[0]
                new_stack.accumulate(I=I, G=G, Dy=Dy, Dx=Dx, S=L, Ly=1)
                new_stack.Py_.append(P)  # Py_: vertical buffer of Ps
                new_stack.down_connect_cnt = 0  # reset down_connect_cnt
                blob = new_stack.blob

            else:  # P has >1 up_connects, or 1 up_connect that has >1 down_connect_cnt:
                blob = up_connect_[0].blob
                # initialize new_stack with up_connect blob:
                new_stack = Cstack(I=I, G=G, Dy=0, Dx=Dx, S=L, Ly=1, y0=y, Py_=[P], blob=blob, down_connect_cnt=0, sign=s)
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
                            I, G, Dy, Dx, S, Ly = Dert.values()
                            accum_Dert(blob.Dert, I=I, G=G, Dy=Dy, Dx=Dx, S=S, Ly=Ly)
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

    I, G, Dy, Dx, S, Ly, y0, Py_, blob, down_connect_cnt, sign = stack.unpack()
    # terminated stack is merged into continued or initialized blob (all connected stacks):
    accum_Dert(blob.Dert, I=I, G=G, Dy=Dy, Dx=Dx, S=S, Ly=Ly)

    blob.open_stacks += down_connect_cnt - 1  # incomplete stack cnt + terminated stack down_connect_cnt - 1: stack itself
    # open stacks contain Ps of a current row and may be extended with new x-overlapping Ps in next run of scan_P_
    if blob.open_stacks == 0:  # number of incomplete stacks == 0: blob is terminated and packed in frame:
        last_stack = stack
        Dert, [y0, x0, xn], stack_, s, open_stacks = blob.unpack()[:5]
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
        blob.box=(y0, yn, x0, xn)
        blob.dert__ = dert__
        blob.mask = mask
        blob.adj_blobs = [[], 0, 0]
        blob.fopen = fopen

        frame.update(I=frame['I'] + blob.Dert['I'],
                     G=frame['G'] + blob.Dert['G'],
                     Dy=frame['Dy'] + blob.Dert['Dy'],
                     Dx=frame['Dx'] + blob.Dert['Dx'])
        frame['blob__'].append(blob)


def assign_adjacent(blob_binder):  # adjacents are connected opposite-sign blobs
    '''
    Assign adjacent blobs bilaterally according to adjacent pairs' ids in blob_binder.
    '''
    for blob_id1, blob_id2 in blob_binder.adj_pairs:
        assert blob_id1 < blob_id2
        blob1 = blob_binder.cluster_cls.get_instance(blob_id1)
        blob2 = blob_binder.cluster_cls.get_instance(blob_id2)

        if blob1.box[1] < blob2.box[1]:  # yn1 < yn2: blob1 is potentially internal to blob2
            if blob1.fopen:
                pose1 = pose2 = 2  # 2 for open
            else:
                pose1, pose2 = 0, 1  # 0 for internal, 1 for external
        else:  # blob2 is potentially internal to blob1
            if blob2.fopen:
                pose1 = pose2 = 2
            else:
                pose1, pose2 = 1, 0

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


def update_dert(blob):  # add idy, idx, m to dert__
    i, g, dy, dx = blob.dert__
    blob.dert__ = (i,
                   np.zeros(i.shape),  # idy
                   np.zeros(i.shape),  # idx
                   g, dy, dx,
                   np.zeros(i.shape))  # m

    # no need to return, changes are applied to blob

# -----------------------------------------------------------------------------
# Main

if __name__ == '__main__':
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpeg')
    argument_parser.add_argument('-v', '--verbose', help='print details, useful for debugging', type=int, default=1)
    argument_parser.add_argument('-n', '--intra', help='run intra_blobs after frame_blobs', type=int, default=0)
    argument_parser.add_argument('-r', '--render', help='render the process', type=int, default=1)
    arguments = vars(argument_parser.parse_args())
    image = imread(arguments['image'])
    verbose = arguments['verbose']
    intra = arguments['intra']
    render = arguments['render']

    start_time = time()
    frame = image_to_blobs(image, verbose, render)

    if intra:  # Tentative call to intra_blob, omit for testing frame_blobs:

        if verbose:
            print("\rRunning intra_blob...")

        from intra_blob import (
            intra_blob, CDeepBlob, aveB,
        )

        deep_frame = frame, frame  # 1st frame initializes summed representation of hierarchy, 2nd is individual top layer
        deep_blob_i_ = []  # index of a blob with deep layers
        deep_layers = [[]]*len(frame['blob__'])  # for visibility only
        empty = np.zeros_like(frame['dert__'][0])
        deep_root_dert__ = (  # update root dert__
            frame['dert__'][0],    # i
            empty,                 # idy
            empty,                 # idx
            *frame['dert__'][1:],  # g, dy, dx
            empty,                 # m
        )

        for i, blob in enumerate(frame['blob__']):  # print('Processing blob number ' + str(bcount))
            '''
            Blob G: -|+ predictive value, positive value of -G blobs is lent to the value of their adjacent +G blobs. 
            +G "edge" blobs are low-match, valuable only as contrast: to the extent that their negative value cancels 
            positive value of adjacent -G "flat" blobs.
            '''
            G = blob.Dert['G']; adj_G = blob.adj_blobs[2]
            borrow_G = min(abs(G), abs(adj_G) / 2)
            '''
            int_G / 2 + ext_G / 2, because both borrow or lend bilaterally, 
            same as pri_M and next_M in line patterns but direction here is radial: inside-out
            borrow_G = min, ~ comp(G,_G): only value present in both parties can be borrowed from one to another
            Add borrow_G -= inductive leaking across external blob?
            '''
            blob = CDeepBlob(Dert=blob.Dert, box=blob.box, stack_=blob.stack_,
                             sign=blob.sign, root_dert__=deep_root_dert__,
                             dert__=blob.dert__, mask=blob.mask,
                             adj_blobs=blob.adj_blobs, fopen=blob.fopen) #, margin=blob.margin)
            if blob.sign:
                if G + borrow_G > aveB and blob.dert__[0].shape[0] > 3 and blob.dert__[0].shape[1] > 3:  # min blob dimensions
                    update_dert(blob)
                    deep_layers[i] = intra_blob(blob, rdn=1, rng=.0, fig=0, fcr=0, render=render)  # +G blob' dert__' comp_g

            elif -G - borrow_G > aveB and blob.dert__[0].shape[0] > 3 and blob.dert__[0].shape[1] > 3:  # min blob dimensions
                update_dert(blob)
                deep_layers[i] = intra_blob(blob, rdn=1, rng=1, fig=0, fcr=1, render=render)  # -G blob' dert__' comp_r in 3x3 kernels

            if deep_layers[i]:  # if there are deeper layers
                deep_blob_i_.append(i)  # indices of blobs with deep layers

        if verbose:
            print("\rFinished running intra_blob")

    end_time = time() - start_time
    if verbose:
        print(f"\nSession ended in {end_time:.2} seconds", end="")
    else:
        print(end_time)