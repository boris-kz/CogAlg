from time import time
from collections import deque, defaultdict
import numpy as np
import cv2
from comp_pixel import comp_pixel

'''
    2D version of first-level core algorithm will have frame_blobs, intra_blob (recursive search within blobs), and comp_P.
    frame_blobs() forms parameterized blobs: contiguous areas of positive or negative deviation of gradient per pixel.    

    comp_pixel (lateral, vertical, diagonal) forms dert, queued in dert__: tuples of pixel + derivatives, over whole image. 
    Then pixel-level and external parameters are accumulated in row segment Ps, vertical blob segment, and blobs,
    adding a level of encoding per row y, defined relative to y of current input row, with top-down scan:

    1Le, line y-1: form_P( dert_) -> 1D pattern P: contiguous row segment, a slice of a blob
    2Le, line y-2: scan_P_(P, hP) -> hP, up_fork_, down_fork_count: vertical connections per stack of Ps 
    3Le, line y-3: form_stack(hP, stack) -> stack: merge vertically-connected _Ps into non-forking stacks of Ps
    4Le, line y-4+ stack depth: form_blob(stack, blob): merge connected stacks in blobs referred by up_fork_, recursively

    Higher-row elements include additional parameters, derived while they were lower-row elements. Processing is bottom-up:
    from input-row to higher-row structures, sequential because blobs are irregular, not suited for matrix operations.
    Resulting blob structure (fixed set of parameters per blob): 

    - root_fork = frame,  # replaced by blob-level fork in sub_blobs
    - Dert = dict(I, G, Dy, Dx, S, Ly), # summed pixel dert params (I, G, Dy, Dx), surface area S, vertical depth Ly
    - sign = s,  # sign of gradient deviation
    - box  = [y0, yn, x0, xn], 
    - map, # inverted mask
    - dert__,  # 2D array of pixel-level derts: (p, g, dy, dx) tuples
    - stack_,  # contains intermediate blob composition structures: stacks and Ps, not meaningful on their own
    ( intra_blob structure extends Dert, adds crit, rng, fork_)

    Blob is 2D pattern: connectivity cluster defined by the sign of gradient deviation. Gradient represents 2D variation
    per pixel. It is used as inverse measure of partial match (predictive value) because direct match (min intensity) 
    is not meaningful in vision. Intensity of reflected light doesn't correlate with predictive value of observed object: 
    some physical density, hardness, inertia that represent resistance to change in positional parameters.  

    This is clustering by connectivity because distance between clustered pixels should not exceed cross-comparison range.
    That range is fixed for each layer of search, to enable encoding of input pose parameters: coordinates, dimensions, 
    orientation. These params are essential because value of prediction = precision of what * precision of where. 

    frame_blobs is a complex function with a simple purpose: to sum pixel-level params in blob-level params. These params 
    were derived by pixel cross-comparison (cross-correlation) to represent predictive value per pixel, so they are also
    predictive on a blob level, and should be cross-compared between blobs on the next level of search and composition.

    Please see diagrams of frame_blobs on https://kwcckw.github.io/CogAlg/
'''
# Adjustable parameters:

kwidth = 3  # smallest input-centered kernel: frame | blob shrink by 2 pixels per row
ave = 50  # filter or hyper-parameter, set as a guess, latter adjusted by feedback

# ----------------------------------------------------------------------------------------------------------------------------------------
# Functions

# prefix '_' denotes higher-line variable or structure, vs. same-type lower-line variable or structure
# postfix '_' denotes array name, vs. same-name elements of that array

def image_to_blobs(image):  # root function, postfix '_' denotes array, prefix '_' denotes higher- vs lower-line variable

    dert__ = comp_pixel(image)  # comparison of central pixel to rim pixels in 3x3 kernel
    frame = dict(rng=1, dert__=dert__, mask=None, I=0, G=0, Dy=0, Dx=0, blob_=[])
    stack_ = deque()  # buffer of running stacks of Ps
    height, width = image.shape

    for y in range(height - kwidth + 1):  # first and last row are discarded
        P_ = form_P_(dert__[:, y].T)  # horizontal clustering
        P_ = scan_P_(P_, stack_, frame)  # vertical clustering, adds up_forks per P and down_fork_cnt per stack
        stack_ = form_stack_(y, P_, frame)

    while stack_:  # frame ends, last-line stacks are merged into their blobs:
        form_blob(stack_.popleft(), frame)

    return frame  # frame of blobs

''' 
Parameterized connectivity clustering functions below:

- form_P sums dert params within P and increments its L: horizontal length.
- scan_P_ searches for horizontal (x) overlap between Ps of consecutive (in y) rows.
- form_stack combines these overlapping Ps into vertical stacks of Ps, with 1 up_P to 1 down_P
- form_blob merges terminated or forking stacks into blob, removes redundant representations of the same blob 
  by multiple forked P stacks, then checks for blob termination and merger into whole-frame representation.

dert is a tuple of derivatives per pixel, initially (p, dy, dx, g), will be extended in intra_blob
Dert is params of a composite structure (P, stack, blob): summed dert params + dimensions: vertical Ly and area S
'''

def form_P_(dert_):  # horizontal clustering and summation of dert params into P params, per row of a frame
    # P is a segment of same-sign derts in horizontal slice of a blob

    P_ = deque()  # row of Ps
    I, G, Dy, Dx, L, x0 = *dert_[0], 1, 0  # initialize P params with 1st dert params
    G -= ave
    _s = G > 0  # sign

    for x, (p, g, dy, dx) in enumerate(dert_[1:], start=1):
        vg = g - ave
        s = vg > 0
        if s != _s:
            # terminate and pack P:
            P = dict(I=I, G=G, Dy=Dy, Dx=Dx, L=L, x0=x0, dert_=dert_[x0:x0 + L], sign=_s)
            P_.append(P)
            # initialize new P:
            I, G, Dy, Dx, L, x0 = 0, 0, 0, 0, 0, x
        # accumulate P params:
        I += p
        G += vg  # M += m only within negative vg blobs
        Dy += dy
        Dx += dx
        L += 1
        _s = s  # prior sign

    P = dict(I=I, G=G, Dy=Dy, Dx=Dx, L=L, x0=x0, dert_=dert_[x0:x0 + L], sign=_s)
    P_.append(P)  # terminate last P in a row
    return P_


def scan_P_(P_, stack_, frame):  # merge P into higher-row stack of Ps which have same sign and overlap by x_coordinate
    '''
    Each P in P_ scans higher-row _Ps (in stack_) left-to-right, testing for x-overlaps between Ps and same-sign _Ps.
    Overlap is represented as up_fork in P and is added to down_fork_cnt in _P. Scan continues until P.x0 >= _P.xn:
    no x-overlap between P and next _P. Then P is packed into its up_fork stacks or initializes a new stack.
    After such test, loaded _P is also tested for x-overlap to the next P.
    If negative, a stack with loaded _P is removed from stack_ (buffer of higher-row stacks) and tested for down_fork_cnt==0.
    If so: no lower-row connections, the stack is packed into connected blobs (referred by its up_fork_),
    else the stack is recycled into next_stack_, for next-row run of scan_P_.
    It's a form of breadth-first flood fill, with forks as vertices per stack of Ps: a node in connectivity graph.
    '''
    next_P_ = deque()  # to recycle P + up_fork_ that finished scanning _P, will be converted into next_stack_

    if P_ and stack_:  # if both input row and higher row have any Ps / _Ps left

        P = P_.popleft()  # load left-most (lowest-x) input-row P
        stack = stack_.popleft()  # higher-row stacks,
        _P = stack['Py_'][-1]  # last element of each stack is higher-row P
        up_fork_ = []  # list of same-sign x-overlapping _Ps per P

        while True:  # while both P_ and stack_ are not empty

            x0 = P['x0']  # first x in P
            xn = x0 + P['L']  # first x in next P
            _x0 = _P['x0']  # first x in _P
            _xn = _x0 + _P['L']  # first x in next _P

            if (P['sign'] == stack['sign']
                    and _x0 < xn and x0 < _xn):  # test for sign match and x overlap between loaded P and _P
                stack['down_fork_cnt'] += 1
                up_fork_.append(stack)  # P-connected higher-row stacks are buffered into up_fork_ per P

            if xn < _xn:  # _P overlaps next P in P_
                next_P_.append((P, up_fork_))  # recycle _P for the next run of scan_P_
                up_fork_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:  # terminate loop
                    if stack['down_fork_cnt'] != 1:  # terminate stack, merge it into up_forks' blobs
                        form_blob(stack, frame)
                    break
            else:  # no next-P overlap
                if stack['down_fork_cnt'] != 1:  # terminate stack, merge it into up_forks' blobs
                    form_blob(stack, frame)

                if stack_:  # load stack with next _P
                    stack = stack_.popleft()
                    _P = stack['Py_'][-1]
                else:  # no stack left: terminate loop
                    next_P_.append((P, up_fork_))
                    break

    while P_:  # terminate Ps and stacks that continue at row's end
        next_P_.append((P_.popleft(), []))  # no up_fork
    while stack_:
        form_blob(stack_.popleft(), frame)  # down_fork_cnt always == 0

    return next_P_  # each element is P + up_fork_ refs


def form_stack_(y, P_, frame):  # Convert or merge every P into its stack of Ps, merge blobs

    next_stack_ = deque()  # converted to stack_ in the next run of scan_P_

    while P_:
        P, up_fork_ = P_.popleft()
        s = P.pop('sign')
        I, G, Dy, Dx, L, x0, dert_ = P.values()
        xn = x0 + L  # next-P x0
        if not up_fork_:
            # initialize new stack for each input-row P that has no connections in higher row:
            blob = dict(Dert=dict(I=0, G=0, Dy=0, Dx=0, S=0, Ly=0), box=[y, x0, xn], stack_=[], sign=s, open_stacks=1)
            new_stack = dict(I=I, G=G, Dy=0, Dx=Dx, S=L, Ly=1, y0=y, Py_=[P], blob=blob, down_fork_cnt=0, sign=s)
            blob['stack_'].append(new_stack)
        else:
            if len(up_fork_) == 1 and up_fork_[0]['down_fork_cnt'] == 1:
                # P has one up_fork and that up_fork has one root: merge P into up_fork stack:
                new_stack = up_fork_[0]
                accum_Dert(new_stack, I=I, G=G, Dy=Dy, Dx=Dx, S=L, Ly=1)
                new_stack['Py_'].append(P)  # Py_: vertical buffer of Ps
                new_stack['down_fork_cnt'] = 0  # reset down_fork_cnt
                blob = new_stack['blob']

            else:  # if > 1 up_forks, or 1 up_fork that has > 1 down_fork_cnt:
                blob = up_fork_[0]['blob']
                # initialize new_stack with up_fork blob:
                new_stack = dict(I=I, G=G, Dy=0, Dx=Dx, S=L, Ly=1, y0=y, Py_=[P], blob=blob, down_fork_cnt=0, sign=s)
                blob['stack_'].append(new_stack)  # stack is buffered into blob

                if len(up_fork_) > 1:  # merge blobs of all up_forks
                    if up_fork_[0]['down_fork_cnt'] == 1:  # up_fork is not terminated
                        form_blob(up_fork_[0], frame)  # merge stack of 1st up_fork into its blob

                    for up_fork in up_fork_[1:len(up_fork_)]:  # merge blobs of other up_forks into blob of 1st up_fork
                        if up_fork['down_fork_cnt'] == 1:
                            form_blob(up_fork, frame)

                        if not up_fork['blob'] is blob:
                            Dert, box, stack_, s, open_stacks = up_fork['blob'].values()  # merged blob
                            I, G, Dy, Dx, S, Ly = Dert.values()
                            accum_Dert(blob['Dert'], I=I, G=G, Dy=Dy, Dx=Dx, S=S, Ly=Ly)
                            blob['open_stacks'] += open_stacks
                            blob['box'][0] = min(blob['box'][0], box[0])  # extend box y0
                            blob['box'][1] = min(blob['box'][1], box[1])  # extend box x0
                            blob['box'][2] = max(blob['box'][2], box[2])  # extend box xn
                            for stack in stack_:
                                if not stack is up_fork:
                                    stack[
                                        'blob'] = blob  # blobs in other up_forks are references to blob in the first up_fork.
                                    blob['stack_'].append(stack)  # buffer of merged root stacks.
                            up_fork['blob'] = blob
                            blob['stack_'].append(up_fork)
                        blob['open_stacks'] -= 1  # overlap with merged blob.

        blob['box'][1] = min(blob['box'][1], x0)  # extend box x0
        blob['box'][2] = max(blob['box'][2], xn)  # extend box xn
        next_stack_.append(new_stack)

    return next_stack_


def form_blob(stack, frame):  # increment blob with terminated stack, check for blob termination and merger into frame

    I, G, Dy, Dx, S, Ly, y0, Py_, blob, down_fork_cnt, sign = stack.values()
    accum_Dert(blob['Dert'], I=I, G=G, Dy=Dy, Dx=Dx, S=S, Ly=Ly)
    # terminated stack is merged into continued or initialized blob (all connected stacks):

    blob['open_stacks'] += down_fork_cnt - 1  # incomplete stack cnt + terminated stack down_fork_cnt - 1: stack itself
    # open stacks contain Ps of a current row and may be extended with new x-overlapping Ps in next run of scan_P_

    if blob['open_stacks'] == 0:  # if number of incomplete stacks == 0: blob is terminated and packed in frame
        last_stack = stack

        Dert, [y0, x0, xn], stack_, s, open_stacks = blob.values()
        yn = last_stack['y0'] + last_stack['Ly']

        mask = np.ones((yn - y0, xn - x0), dtype=bool)  # map of blob in coord box
        for stack in stack_:
            stack.pop('sign')
            stack.pop('down_fork_cnt')
            for y, P in enumerate(stack['Py_'], start=stack['y0'] - y0):
                x_start = P['x0'] - x0
                x_stop = x_start + P['L']
                mask[y, x_start:x_stop] = False
        dert__ = frame['dert__'][:, y0:yn, x0:xn]
        dert__.mask[:] = mask  # default mask is all 0s

        blob.pop('open_stacks')
        blob.update(box=(y0, yn, x0, xn),  # boundary box
                    map=~mask,  # to compute overlap in comp_blob
                    crit=1,  # clustering criterion is g
                    rng=1,  # if 3x3 kernel
                    dert__=dert__,  # dert__ + box replace slices=(Ellipsis, slice(y0, yn), slice(x0, xn))
                    root_fork=frame,
                    fork_=defaultdict(dict),  # or []? contains forks ( sub-blobs
                    )
        frame.update(I=frame['I'] + blob['Dert']['I'],
                     G=frame['G'] + blob['Dert']['G'],
                     Dy=frame['Dy'] + blob['Dert']['Dy'],
                     Dx=frame['Dx'] + blob['Dert']['Dx'])

        frame['blob_'].append(blob)


# -----------------------------------------------------------------------------
# Utilities

def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})


# -----------------------------------------------------------------------------
# Main

if __name__ == '__main__':
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon.jpg')
    arguments = vars(argument_parser.parse_args())
    image = cv2.imread(arguments['image'], 0).astype(int)

    start_time = time()
    frame_of_blobs = image_to_blobs(image)

    # from intra_blob import cluster_eval, intra_fork, cluster, aveF, aveC, aveB, etc.
    '''
    frame_of_deep_blobs = {  # initialize frame_of_deep_blobs
        'blob_': [],
        'params': {
            'I': frame_of_blobs['I'],
            'G': frame_of_blobs['G'],
            'Dy': frame_of_blobs['Dy'],
            'Dx': frame_of_blobs['Dx'],
            # deeper params are initialized when they are fetched
        }),
    }
    for blob in frame_of_blobs['blob_']:  # evaluate recursive sub-clustering in each blob, via cluster_eval -> intra_fork
    
        if blob['Dert']['G'] > aveB:  # +G blob directly calls intra_fork(comp_g), no immediate sub-clustering
            intra_fork(blob, aveF, aveC, aveB, ave, rng * 2 + 1, 1, fig=0, fa=0)  # nI = 1: g
        
        elif -blob['Dert']['G'] > aveB: # -G blob, sub-clustering by -vg for rng+ eval
            cluster_eval(blob, aveF, aveC, aveB, ave, rng + 1, 2, fig=0, fa=0)  # cluster by -g for rng+, idiomatic crit=2: not index 

        frame_of_deep_blobs['blob_'].append(blob)
        frame_of_deep_blobs['params'][1:] += blob['params'][1:]  # incorrect, for selected blob params only?
    '''

    # DEBUG -------------------------------------------------------------------
    from utils import map_frame

    cv2.imwrite("./images/blobs.bmp", map_frame(frame_of_blobs))
    # END DEBUG ---------------------------------------------------------------

    end_time = time() - start_time
    print(end_time)