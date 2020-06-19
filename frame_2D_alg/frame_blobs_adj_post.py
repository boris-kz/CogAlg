from time import time
from collections import deque, defaultdict
import numpy as np
from copy import copy
# from comp_pixel import comp_pixel
from utils import *

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

    This is clustering by connectivity because distance between clustered pixels should not exceed cross-comparison range.
    That range is fixed for each layer of search, to enable encoding of input pose parameters: coordinates, dimensions, 
    orientation. These params are essential because value of prediction = precision of what * precision of where. 

    frame_blobs is a complex function with a simple purpose: to sum pixel-level params in blob-level params. These params 
    were derived by pixel cross-comparison (cross-correlation) to represent predictive value per pixel, so they are also
    predictive on a blob level, and should be cross-compared between blobs on the next level of search and composition.
    Please see diagrams of frame_blobs on https://kwcckw.github.io/CogAlg/
'''

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback

# Functions:
# prefix '_' denotes higher-line variable or structure, vs. same-type lower-line variable or structure
# postfix '_' denotes array name, vs. same-name elements of that array

def comp_pixel(image):  # current version of 2x2 pixel cross-correlation within image

    # input slices to a sliding 2x2 kernel:
    topleft__ = image[:-1, :-1]
    topright__ = image[:-1, 1:]
    botleft__ = image[1:, :-1]
    botright__ = image[1:, 1:]

    dy__ = ((botleft__ + botright__) - (topleft__ + topright__))  # same as diagonal from left
    dx__ = ((topright__ + botright__) - (topleft__ + botleft__))  # same as diagonal from right
    g__ = np.hypot(dy__, dx__)  # gradient per kernel

    return ma.stack((topleft__, g__, dy__, dx__))  # 2D dert array


def image_to_blobs(image):
    dert__ = comp_pixel(image)  # 2x2 cross-comparison / cross-correlation

    frame = dict(rng=1, dert__=dert__, mask=None, I=0, G=0, Dy=0, Dx=0, blob__=[])
    stack_ = deque()  # buffer of running vertical stacks of Ps
    height, width = dert__.shape[1:]

    for y in range(height):  # first and last row are discarded
        print(f'Processing line {y}...')

        P_ = form_P_(dert__[:, y].T)  # horizontal clustering
        P_ = scan_P_(P_, stack_, frame)  # vertical clustering, adds P up_connects and _P down_connect_cnt
        stack_ = form_stack_(P_, frame, y)

    while stack_:  # frame ends, last-line stacks are merged into their blobs
        form_blob(stack_.popleft(), frame)

    return frame  # frame of blobs

''' 
Parameterized connectivity clustering functions below:
- form_P sums dert params within P and increments its L: horizontal length.
- scan_P_ searches for horizontal (x) overlap between Ps of consecutive (in y) rows.
- form_stack combines these overlapping Ps into vertical stacks of Ps, with 1 up_P to 1 down_P
- form_blob merges terminated or forking stacks into blob, removes redundant representations of the same blob 
  by multiple forked P stacks, then checks for blob termination and merger into whole-frame representation.

dert: tuple of derivatives per pixel, initially (p, dy, dx, g), will be extended in intra_blob
Dert: params of composite structures (P, stack, blob): summed dert params + dimensions: vertical Ly and area S
'''

def form_P_(dert__):  # horizontal clustering and summation of dert params into P params, per row of a frame
    # P is a segment of same-sign derts in horizontal slice of a blob

    P_ = deque()  # row of Ps
    I, G, Dy, Dx, L, x0 = *dert__[0], 1, 0  # initialize P params with 1st dert params
    G = int(G) - ave
    _s = G > 0  # sign
    for x, (p, g, dy, dx) in enumerate(dert__[1:], start=1):
        vg = int(g) - ave  # deviation of g
        s = vg > 0
        if s != _s:
            # terminate and pack P:
            P = dict(I=I, G=G, Dy=Dy, Dx=Dx, L=L, x0=x0, sign=_s)  # no need for dert_
            # initialize new P params:
            I, G, Dy, Dx, L, x0 = 0, 0, 0, 0, 0, x
            P_.append(P)

        I += p  # accumulate P params
        G += vg
        Dy += dy
        Dx += dx
        L += 1
        _s = s  # prior sign

    P = dict(I=I, G=G, Dy=Dy, Dx=Dx, L=L, x0=x0, sign=_s)  # last P in a row
    P_.append(P)

    return P_

def scan_P_(P_, stack_, frame):  # merge P into higher-row stack of Ps which have same sign and overlap by x_coordinate
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
        _P = stack['Py_'][-1]  # last element of each stack is higher-row P
        up_connect_ = []  # list of same-sign x-overlapping _Ps per P

        while True:  # while both P_ and stack_ are not empty

            x0 = P['x0']  # first x in P
            xn = x0 + P['L']  # first x in next P
            _x0 = _P['x0']  # first x in _P
            _xn = _x0 + _P['L']  # first x in next _P

            if stack['G'] > 0:  # check for overlaps in 8 directions, else a blob may leak through its external blob
                if _x0 - 1 < xn and x0 < _xn + 1:  # x overlap between loaded P and _P
                    if P['sign'] == stack['sign']:  # sign match
                        stack['down_connect_cnt'] += 1
                        up_connect_.append(stack)  # buffer P-connected higher-row stacks into P' up_connect_

            else:  # -G, check for orthogonal overlaps only: 4 directions, edge blobs are more selective
                if _x0 < xn and x0 < _xn:  # x overlap between loaded P and _P
                    if P['sign'] == stack['sign']:  # sign match
                        stack['down_connect_cnt'] += 1
                        up_connect_.append(stack)  # buffer P-connected higher-row stacks into P' up_connect_

            if xn < _xn:  # _P overlaps next P in P_
                next_P_.append((P, up_connect_))  # recycle _P for the next run of scan_P_
                up_connect_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:  # terminate loop
                    if stack['down_connect_cnt'] != 1:  # terminate stack, merge it into up_connects' blobs
                        form_blob(stack, frame)
                    break
            else:  # no next-P overlap
                if stack['down_connect_cnt'] != 1:  # terminate stack, merge it into up_connects' blobs
                    form_blob(stack, frame)
                if stack_:  # load stack with next _P
                    stack = stack_.popleft()
                    _P = stack['Py_'][-1]
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
        s = P.pop('sign')
        I, G, Dy, Dx, L, x0 = P.values()
        xn = x0 + L  # next-P x0
        if not up_connect_:
            # initialize new stack for each input-row P that has no connections in higher row, as in the whole top row:
            blob = dict(Dert=dict(I=0, G=0, Dy=0, Dx=0, S=0, Ly=0), box=[y, x0, xn], stack_=[], sign=s, open_stacks=1)
            new_stack = dict(I=I, G=G, Dy=0, Dx=Dx, S=L, Ly=1, y0=y, Py_=[P], blob=blob, down_connect_cnt=0, sign=s)
            blob['stack_'].append(new_stack)

        else:
            if len(up_connect_) == 1 and up_connect_[0]['down_connect_cnt'] == 1:
                # P has one up_connect and that up_connect has one down_connect=P: merge P into up_connect stack:
                new_stack = up_connect_[0]
                accum_Dert(new_stack, I=I, G=G, Dy=Dy, Dx=Dx, S=L, Ly=1)
                new_stack['Py_'].append(P)  # Py_: vertical buffer of Ps
                new_stack['down_connect_cnt'] = 0  # reset down_connect_cnt
                blob = new_stack['blob']

            else:  # P has >1 up_connects, or 1 up_connect that has >1 down_connect_cnt:
                blob = up_connect_[0]['blob']
                # initialize new_stack with up_connect blob:
                new_stack = dict(I=I, G=G, Dy=0, Dx=Dx, S=L, Ly=1, y0=y, Py_=[P], blob=blob, down_connect_cnt=0, sign=s)
                blob['stack_'].append(new_stack)  # stack is buffered into blob

                if len(up_connect_) > 1:  # merge blobs of all up_connects
                    if up_connect_[0]['down_connect_cnt'] == 1:  # up_connect is not terminated
                        form_blob(up_connect_[0], frame)  # merge stack of 1st up_connect into its blob

                    for up_connect in up_connect_[1:len(up_connect_)]:  # merge blobs of other up_connects into blob of 1st up_connect
                        if up_connect['down_connect_cnt'] == 1:
                            form_blob(up_connect, frame)

                        if not up_connect['blob'] is blob:
                            Dert, box, stack_, s, open_stacks = up_connect['blob'].values()  # merged blob
                            I, G, Dy, Dx, S, Ly = Dert.values()
                            accum_Dert(blob['Dert'], I=I, G=G, Dy=Dy, Dx=Dx, S=S, Ly=Ly)
                            blob['open_stacks'] += open_stacks
                            blob['box'][0] = min(blob['box'][0], box[0])  # extend box y0
                            blob['box'][1] = min(blob['box'][1], box[1])  # extend box x0
                            blob['box'][2] = max(blob['box'][2], box[2])  # extend box xn
                            for stack in stack_:
                                if not stack is up_connect:
                                    stack['blob'] = blob  # blobs in other up_connects are refs to blob in first up_connect
                                    blob['stack_'].append(stack)  # buffer of merged root stacks.

                            up_connect['blob'] = blob
                            blob['stack_'].append(up_connect)
                        blob['open_stacks'] -= 1  # overlap with merged blob.

        blob['box'][1] = min(blob['box'][1], x0)  # extend box x0
        blob['box'][2] = max(blob['box'][2], xn)  # extend box xn

        next_stack_.append(new_stack)

    return next_stack_  # input for the next line of scan_P_


def form_blob(stack, frame):  # increment blob with terminated stack, check for blob termination and merger into frame

    I, G, Dy, Dx, S, Ly, y0, Py_, blob, down_connect_cnt, sign = stack.values()
    # terminated stack is merged into continued or initialized blob (all connected stacks):
    accum_Dert(blob['Dert'], I=I, G=G, Dy=Dy, Dx=Dx, S=S, Ly=Ly)

    blob['open_stacks'] += down_connect_cnt - 1  # incomplete stack cnt + terminated stack down_connect_cnt - 1: stack itself
    # open stacks contain Ps of a current row and may be extended with new x-overlapping Ps in next run of scan_P_
    if blob['open_stacks'] == 0:  # number of incomplete stacks == 0: blob is terminated and packed in frame:
        last_stack = stack
        Dert, [y0, x0, xn], stack_, s, open_stacks= blob.values()
        yn = last_stack['y0'] + last_stack['Ly']

        mask = np.ones((yn - y0, xn - x0), dtype=bool)  # mask box, then unmask Ps:
        for stack in stack_:
            stack.pop('sign')
            stack.pop('down_connect_cnt')
            for y, P in enumerate(stack['Py_'], start=stack['y0'] - y0):
                x_start = P['x0'] - x0
                x_stop = x_start + P['L']
                mask[y, x_start:x_stop] = False

        dert__ = (frame['dert__'][:, y0:yn, x0:xn]).copy()  # copy mask as dert.mask
        dert__.mask[:] = True
        dert__.mask[:] = mask  # overwrite default mask 0s
        frame['dert__'][:, y0:yn, x0:xn] = dert__.copy()  # assign mask back to frame dert__

        blob_map = np.ones((frame['dert__'].shape[1], frame['dert__'].shape[2])).astype('bool')
        blob_map[y0:yn, x0:xn] = mask
        if blob['sign']:
            extended_map, map = add_margin(frame, blob_map, diag=1)  # orthogonal + diagonal margin
            # we already have dert__.mask, blob_map is redundant?
            # also, get the margin instead of extended_map?
        else:
            extended_map, map = add_margin(frame, blob_map, diag=0)  # orthogonal margin

        fopen = 0  # flag for blobs on frame boundary
        if x0 == 0 or xn == frame['dert__'].shape[2] or y0 == 0 or yn == frame['dert__'].shape[1]:
            fopen = 1

        blob.pop('open_stacks')
        blob.update(root_dert__=frame['dert__'],
                    box=(y0, yn, x0, xn),
                    dert__=dert__,
                    adj_blob_ = [[], []],
                    fopen=fopen,
                    extended_map = extended_map,  # for find_adjacent
                    map = blob_map
                    )
        frame.update(I=frame['I'] + blob['Dert']['I'],
                     G=frame['G'] + blob['Dert']['G'],
                     Dy=frame['Dy'] + blob['Dert']['Dy'],
                     Dx=frame['Dx'] + blob['Dert']['Dx'])

        frame['blob__'].append(blob)


def find_adjacent(frame):  # scan_blob__? draft, adjacents are blobs directly next to _blob
    '''
    2D version of scan_P_, but primarily vertical and checking for opposite-sign adjacency vs. same-sign overlap
    '''
    blob__ = []  # initialization
    while frame['blob__']:  # outer loop

        _blob = frame['blob__'].pop(0)  # pop left outer loop's blob
        _y0, _yn, _x0, _xn = _blob['box']
        if 'adj_blob_' in _blob:  # reuse the adj_blob_ from _blob, else we will wipe out the existing adj_blob_
            # _blob should be repacked in new_blob__, not in blob__, then no need to check?
            _adj_blob_ = _blob['adj_blob_']
        else:
            _adj_blob_ = [[], []]  # [adj_blobs], [positions]: 0 = internal to current blob, 1 = external, 2 = open

        i = 0  # inner loop counter
        while i <= len(frame['blob__']) - 1:  # vertical overlap between _blob and blob, including border

            blob = frame['blob__'][i]  # inner loop's blob
            if 'adj_blob_' in blob:
                adj_blob_ = blob['adj_blob_']
            else:
                adj_blob_ = [[], []]  # [adj_blobs], [positions]: 0 = internal to current blob, 1 = external, 2 = open
            y0, yn, x0, xn = blob['box']

            if y0 <= _yn and blob['sign'] != _blob['sign']: #  adjacent blobs have opposite sign and vertical overlap with _blob + margin

                margin_map = np.logical_xor(blob['extended_map'], blob['map'])  # get blob margin area only?
                # replace map with _blob['dert__'].mask?
                margin_AND = np.logical_and(margin_map, ~_blob['map'])  # AND blob margin and _blob area

                if margin_AND.any():  # at least one blob's margin element is in _blob: blob is adjacent
                    if np.count_nonzero(margin_AND) == np.count_nonzero(margin_map) and np.count_nonzero(margin_AND) != 0:
                        # all of blob margin is in _blob: _blob is external
                        if blob not in _adj_blob_[0]:
                            _adj_blob_[0].append(blob)
                            if blob['fopen'] == 1:
                            # this condition is not possible?
                                _adj_blob_[1].append(2)  # adj_blob is open, can't be internal
                            else:
                                _adj_blob_[1].append(0)  # adj_blob is internal
                        if _blob not in adj_blob_[0]:
                            adj_blob_[0].append(_blob)
                            adj_blob_[1].append(1)  # adj_blob is external

                    else:  # _blob is internal or open
                        if blob not in _adj_blob_[0]:
                            _adj_blob_[0].append(blob)
                            _adj_blob_[1].append(1)  # _adj_blob is external
                        if _blob not in adj_blob_[0]:
                            adj_blob_[0].append(_blob)
                            if _blob['fopen'] == 1:
                                adj_blob_[1].append(2)  # _adj_blob is open
                            else:
                                adj_blob_[1].append(0)  # _adj_blob is internal

            blob['adj_blob_'] =  adj_blob_  # pack adj_blob_ in blob
            frame['blob__'][i] = blob  # reassign blob in inner loop
            _blob['adj_blob_'] = _adj_blob_  # pack _adj_blob_ in _blob
            i += 1  # inner loop counter

        blob__.append(_blob)  # repack processed _blob into blob__
        # this should be new_blob__, else ?
    frame['blob__'] = blob__

    return frame


def add_margin(frame, blob_map, diag):  # get 1-pixel margin of blob, orthogonally or orthogonally and diagonally

    c_dert_loc = np.where(blob_map == False)  # unmasked area
    # rename "margin": get the margin only, we already have dert__.mask?
    # extend each dert by 1 for 4 different directions (excluding diagonal directions)
    c_dert_loc_top = (c_dert_loc[0] - 1, c_dert_loc[1])
    c_dert_loc_right = (c_dert_loc[0], c_dert_loc[1] + 1)
    c_dert_loc_bottom = (c_dert_loc[0] + 1, c_dert_loc[1])
    c_dert_loc_left = (c_dert_loc[0], c_dert_loc[1] - 1)

    # remove location of <0 or > image boundary (we cannot set the value to 0 or image boundary since those area is not the margin)
    ind_top = ~(c_dert_loc_top[0] < 0)  # if y < 0
    c_dert_loc_top = (c_dert_loc_top[0][ind_top], c_dert_loc_top[1][ind_top])  # (y,x)

    ind_right = ~(c_dert_loc_right[1] > frame['dert__'].shape[2] - 1)  # if x > X, -1 because index starts from 0, while shape starts from 1
    c_dert_loc_right = (c_dert_loc_right[0][ind_right], c_dert_loc_right[1][ind_right])

    ind_bottom = ~(c_dert_loc_bottom[0] > frame['dert__'].shape[1] - 1)  # if y > Y
    c_dert_loc_bottom = (c_dert_loc_bottom[0][ind_bottom], c_dert_loc_bottom[1][ind_bottom])

    ind_left = ~(c_dert_loc_left[1] < 0)  # if x < 0
    c_dert_loc_left = (c_dert_loc_left[0][ind_left], c_dert_loc_left[1][ind_left])

    extended_map = blob_map.copy()
    extended_map[c_dert_loc_top] = False
    extended_map[c_dert_loc_right] = False
    extended_map[c_dert_loc_bottom] = False
    extended_map[c_dert_loc_left] = False

    if diag:  # extend by one dert diagonally
        c_dert_loc_topleft = (c_dert_loc[0] - 1, c_dert_loc[1] - 1)
        c_dert_loc_topright = (c_dert_loc[0] - 1, c_dert_loc[1] + 1)
        c_dert_loc_bottomleft = (c_dert_loc[0] + 1, c_dert_loc[1] - 1)
        c_dert_loc_bottomright = (c_dert_loc[0] + 1, c_dert_loc[1] + 1)

        ind_topleft = ~np.logical_or(c_dert_loc_topleft[0] < 0, c_dert_loc_topleft[1] < 0)  # if y < 0 or x < 0
        c_dert_loc_topleft = (c_dert_loc_topleft[0][ind_topleft], c_dert_loc_topleft[1][ind_topleft])

        ind_topright = ~np.logical_or(c_dert_loc_topright[0] < 0, c_dert_loc_topright[1] > frame['dert__'].shape[2] - 1)  # if y < 0 or x > X
        c_dert_loc_topright = (c_dert_loc_topright[0][ind_topright], c_dert_loc_topright[1][ind_topright])

        ind_bottomleft = ~np.logical_or(c_dert_loc_bottomleft[0] > frame['dert__'].shape[1] - 1, c_dert_loc_bottomleft[1] < 0)  # if y > Y or x < 0
        c_dert_loc_bottomleft = (c_dert_loc_bottomleft[0][ind_bottomleft], c_dert_loc_bottomleft[1][ind_bottomleft])

        ind_bottomright = ~np.logical_or(c_dert_loc_bottomright[0] > frame['dert__'].shape[1] - 1, c_dert_loc_bottomright[1] > frame['dert__'].shape[2] - 1)
        # if y > Y or x > X
        c_dert_loc_bottomright = (c_dert_loc_bottomright[0][ind_bottomright], c_dert_loc_bottomright[1][ind_bottomright])

        extended_map[c_dert_loc_topleft] = False
        extended_map[c_dert_loc_topright] = False
        extended_map[c_dert_loc_bottomleft] = False
        extended_map[c_dert_loc_bottomright] = False

    return extended_map, blob_map


# -----------------------------------------------------------------------------
# Utilities

def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})


def update_dert(blob):  # Update blob dert with new params

    new_dert__ = np.zeros((7, blob['dert__'].shape[1], blob['dert__'].shape[2]))  # initialize with 0
    new_dert__ = ma.array(new_dert__, mask=True)  # create masked array
    new_dert__.mask = blob['dert__'][0].mask

    new_dert__[0] = blob['dert__'][0]  # i
    new_dert__[1] = 0  # idy
    new_dert__[2] = 0  # idx
    new_dert__[3] = blob['dert__'][1]  # g
    new_dert__[4] = blob['dert__'][2]  # dy
    new_dert__[5] = blob['dert__'][3]  # dx
    new_dert__[6] = 0  # m

    blob['dert__'] = new_dert__.copy()

    return blob

# -----------------------------------------------------------------------------
# Main

if __name__ == '__main__':
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpeg')
    arguments = vars(argument_parser.parse_args())
    image = imread(arguments['image'])

    start_time = time()
    frame = image_to_blobs(image)

    intra = 0
    if intra:  # Tentative call to intra_blob, omit for testing frame_blobs:

        from intra_blob_adj import *

        deep_frame = frame, frame
        bcount = 0
        deep_blob_i_ = []
        deep_layers = []
        layer_count = 0

        for blob in frame['blob__']:
            bcount += 1
            # print('Processing blob number ' + str(bcount))
            # blob.update({'fcr': 0, 'fig': 0, 'rdn': 0, 'rng': 1, 'ls': 0, 'sub_layers': []})

            if blob['sign']:
                if blob['Dert']['G'] > aveB and blob['Dert']['S'] > 20 and blob['dert__'].shape[1] > 3 and blob['dert__'].shape[2] > 3:
                    blob = update_dert(blob)

                    deep_layers.append(intra_blob(blob, rdn=1, rng=.0, fig=0, fcr=0))  # +G blob' dert__' comp_g
                    layer_count += 1

            elif -blob['Dert']['G'] > aveB and blob['Dert']['S'] > 6 and blob['dert__'].shape[1] > 3 and blob['dert__'].shape[2] > 3:

                blob = update_dert(blob)

                deep_layers.append(intra_blob(blob, rdn=1, rng=1, fig=0, fcr=1))  # -G blob' dert__' comp_r in 3x3 kernels
                layer_count += 1

            if len(deep_layers) > 0:
                if len(deep_layers[layer_count - 1]) > 2:
                    deep_blob_i_.append(bcount)  # indices of blobs with deep layers

    end_time = time() - start_time
    print(end_time)

# DEBUG -------------------------------------------------------------------

'''
    imwrite("images/gblobs.bmp",
        map_frame_binary(frame,
                         sign_map={
                             1: WHITE,  # 2x2 gblobs
                             0: BLACK
                         }))
'''
# END DEBUG ---------------------------------------------------------------