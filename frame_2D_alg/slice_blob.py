'''
    This is a terminal fork of intra_blob.
    slice_blob converts selected smooth-edge blobs (high G, low Ga) into slice_blobs, adding internal P and stack structures.
    It then calls comp_g and comp_P per stack, to trace and vectorize these edge blobs.

    Pixel-level parameters are accumulated in contiguous spans of same-sign gradient, first horizontally then vertically.
    Horizontal spans are Ps: 1D patterns, and vertical spans are first stacks of Ps, then blob of connected stacks.

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

    Old diagrams: https://kwcckw.github.io/CogAlg/
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
from operator import attrgetter

from class_cluster import ClusterStructure, NoneType
from class_stream import BlobStreamer
from frame_blobs import CDeepBlob
from comp_slice_draft import comp_slice_blob
from frame_blobs import CDeepBlob

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback
aveG = 50  # filter for comp_g, assumed constant direction
flip_ave = 1000

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
    gdert_ = list
    Dg = int
    Mg = int

class CStack(ClusterStructure):
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
    A = int  # blob area
    Ly = int
    y0 = int
    Py_ = list  # Py_ or dPPy_
    sign = NoneType
    f_gstack = NoneType  # gPPy_ if 1, else Py_
    f_stack_PP = NoneType  # PPy_ if 1, else gPPy_ or Py_
    down_connect_cnt = int
    blob = NoneType
    stack_PP = object

class CBlob(ClusterStructure):
    Dert = dict
    box = list
    stack_ = list
    sign = NoneType
    open_stacks = int
    root_dert__ = object
    dert__ = object
    mask = object
    fopen = bool
    margin = list

# Functions:
# prefix '_' denotes higher-line variable or structure, vs. same-type lower-line variable or structure
# postfix '_' denotes array name, vs. same-name elements of that array. '__' is a 2D array


def slice_blob(blob, dert__, mask, crit__, AveB, verbose=False, render=False):
    frame = dict(rng=1, dert__=dert__, mask=None, I=0, Dy=0, Dx=0, G=0, M=0, Dyy=0, Dyx=0, Dxy=0, Dxx=0, Ga=0, Ma=0, blob__=[])
    stack_ = deque()  # buffer of running vertical stacks of Ps
    height, width = dert__[0].shape

    if render:
        def output_path(input_path, suffix):
            return str(Path(input_path).with_suffix(suffix))
        streamer = BlobStreamer(CBlob, dert__[1],
                                record_path=output_path(arguments['image'],
                                                        suffix='.im2blobs.avi'))
    if verbose:
        start_time = time()
        print("Converting to image to blobs...")


    for y, dert_ in enumerate(zip(*dert__)):  # first and last row are discarded
        if verbose:
            print(f"\rProcessing line {y + 1}/{height}, ", end="")
            print(f"{len(frame['blob__'])} blobs converted", end="")
            sys.stdout.flush()

        P_ = form_P_(zip(*dert_), crit__[y], mask[y])  # horizontal clustering
        if render:
            render = streamer.update_blob_conversion(y, P_)
        P_ = scan_P_(P_, stack_, frame)  # vertical clustering, adds P up_connects and _P down_connect_cnt
        stack_ = form_stack_(P_, frame, y)

    while stack_:  # frame ends, last-line stacks are merged into their blobs
        form_blob(stack_.popleft(), frame)

    # update blob to deep blob and add prior fork information
    for i, iblob in enumerate(frame['blob__']):
        frame['blob__'][i] = CDeepBlob(I=iblob.Dert['I'], Dy=iblob.Dert['Dy'], Dx=iblob.Dert['Dx'], G=iblob.Dert['G'], M=iblob.Dert['M'], A=iblob.Dert['A'],
                                       Ga = iblob.Dert['Ga'],Ma = iblob.Dert['Ma'],Dyy = iblob.Dert['Dyy'],Dyx = iblob.Dert['Dyx'],Dxy = iblob.Dert['Dxy'],Dxx = iblob.Dert['Dxx'],
                                       box=iblob.box, sign=iblob.sign,mask=iblob.mask, root_dert__=dert__, fopen=iblob.fopen, prior_fork=blob.prior_fork.copy(), stack_ = iblob.stack_)


    # tentative, flip_yx should operate on whole blob first

    for blob in frame['blob__']:
        for stack in blob.stack_:
            if stack.f_gstack:

                for istack in stack.Py_:
                    y0 = istack.y0
                    yn = istack.y0 + stack.Ly
                    x0 = min([P.x0 for P in istack.Py_])
                    xn = max([P.x0 + P.L for P in istack.Py_])

                    L_bias = (xn - x0 + 1) / (yn - y0 + 1)  # elongation: width / height, pref. comp over long dimension
                    G_bias = abs(istack.Dy) / abs(istack.Dx)  # ddirection: Gy / Gx, preferential comp over low G

                    if istack.G * L_bias * G_bias > flip_ave:  # y_bias = L_bias * G_bias: projected PM net gain:

                        flipped_Py_ = flip_yx(istack.Py_)  # rotate stack.Py_ by 90 degree, rescan blob vertically -> comp_slice_
    #                return stack, f_istack  # comp_slice if G + M + fflip * (flip_gain - flip_cost) > Ave_comp_slice?

            # evaluate for arbitrary-angle rotation here,
            # to replace flip if both vertical and horizontal dimensions are significantly different from the angle of blob axis.

            else:
                y0 = stack.y0
                yn = stack.y0 + stack.Ly
                x0 = min([P.x0 for P in stack.Py_])
                xn = max([P.x0 + P.L for P in stack.Py_])

                L_bias = (xn - x0 + 1) / (yn - y0 + 1)  # elongation: width / height, pref. comp over long dimension
                G_bias = abs(stack.Dy) / abs(stack.Dx)  # ddirection: Gy / Gx, preferential comp over low G

                if stack.G * L_bias * G_bias > flip_ave:  # y_bias = L_bias * G_bias: projected PM net gain:
                    flipped_Py_ = flip_yx(stack.Py_)  # rotate stack.Py_ by 90 degree, rescan blob vertically -> comp_slice_

    # draw low-ga blob' stacks, draw_stacks(frame)

    # evaluate P blobs
    comp_slice_blob(frame['blob__'], AveB)

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

def form_P_(idert_, crit_, mask_):  # segment dert__ into P__, in horizontal ) vertical order

    P_ = deque()  # row of Ps
    s_ = crit_ > 0
    x0 = 0
    try:
        while mask_[x0]:  # skip until not masked
            next(idert_)
            x0 += 1
    except IndexError:
        return P_  # the whole line is masked, return an empty P

    dert_ = [[*next(idert_)]]  # get first dert from idert_ (generator/iterator)
    (I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma), L = dert_[0], 1  # initialize P params
    _s = s_[x0]
    _mask = mask_[x0]  # mask bit per dert

    for x, (p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma) in enumerate(idert_, start=x0 + 1):  # loop left to right in each row of derts
        mask = mask_[x]
        if ~mask:  # current dert is not masked
            s = s_[x]
            if ~_mask and s != _s:  # prior dert is not masked and sign changed, terminate P:
                P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, sign=_s, dert_=dert_)
                P_.append(P)
                I= Dy= Dx= G= M= Dyy= Dyx= Dxy= Dxx= Ga= Ma= L= 0; x0 = x; dert_ = []  # initialize P params
            elif _mask:
                I= Dy= Dx= G= M= Dyy= Dyx= Dxy= Dxx= Ga= Ma= L= 0; x0 = x; dert_ = []  # initialize P params
        elif ~_mask:
            # dert is masked, prior dert is not masked, terminate P:
            P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, sign=_s, dert_=dert_)
            P_.append(P)

        if ~mask:  # accumulate P params:
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

    if ~_mask:  # terminate last P in a row if prior dert is unmasked
        P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, sign=_s, dert_=dert_)
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

            else:  # -G, check for orthogonal overlaps only: 4 directions, edge blobs are more selective
                if _x0 < xn and x0 < _xn:  # x overlap between loaded P and _P
                    if P.sign == stack.sign:  # sign match
                        stack.down_connect_cnt += 1
                        up_connect_.append(stack)  # buffer P-connected higher-row stacks into P' up_connect_

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
        I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, L, x0, s, dert_, _, _, _ = P.unpack()
        xn = x0 + L  # next-P x0
        if not up_connect_:
            # initialize new stack for each input-row P that has no connections in higher row, as in the whole top row:
            blob = CBlob(Dert=dict(I=0, Dy=0, Dx=0, G=0, M=0, Dyy=0, Dyx=0, Dxy=0, Dxx=0, Ga=0, Ma=0, A=0, Ly=0),
                         box=[y, x0, xn], stack_=[], sign=s, open_stacks=1)
            new_stack = CStack(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=L, Ly=1,
                               y0=y, Py_=[P], blob=blob, down_connect_cnt=0, sign=s, fPP=0)
            new_stack.hid = blob.id
            blob.stack_.append(new_stack)

        else:
            if len(up_connect_) == 1 and up_connect_[0].down_connect_cnt == 1:
                # P has one up_connect and that up_connect has one down_connect=P: merge P into up_connect stack:
                new_stack = up_connect_[0]
                new_stack.accumulate(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=L, Ly=1)
                new_stack.Py_.append(P)  # Py_: vertical buffer of Ps
                new_stack.down_connect_cnt = 0  # reset down_connect_cnt
                blob = new_stack.blob

            else:  # P has >1 up_connects, or 1 up_connect that has >1 down_connect_cnt:
                blob = up_connect_[0].blob
                # initialize new_stack with up_connect blob:
                new_stack = CStack(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=L, Ly=1,
                                   y0=y, Py_=[P], blob=blob, down_connect_cnt=0, sign=s, fPP=0)
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
                            I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, A, Ly = Dert.values()
                            accum_Dert(blob.Dert, I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=A, Ly=Ly)
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

    I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, A, Ly, y0, Py_, sign, f_gstack, f_stack_PP, down_connect_cnt, blob, stack_PP = stack.unpack()
    # terminated stack is merged into continued or initialized blob (all connected stacks):
    accum_Dert(blob.Dert, I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=A, Ly=Ly)

    blob.open_stacks += down_connect_cnt - 1  # incomplete stack cnt + terminated stack down_connect_cnt - 1: stack itself
    # open stacks contain Ps of a current row and may be extended with new x-overlapping Ps in next run of scan_P_

    if blob.open_stacks == 0:  # number of incomplete stacks == 0: blob is terminated and packed in frame:
        # if there is no re-evaluation: check for dert__ termination vs. open stacks
        last_stack = stack
        [y0, x0, xn], stack_, s, open_stacks = blob.unpack()[1:5]
        yn = last_stack.y0 + last_stack.Ly

        mask = np.ones((yn - y0, xn - x0), dtype=bool)  # mask box, then unmask Ps:

        for stack in stack_:
            for y, P in enumerate(stack.Py_, start=stack.y0 - y0):
                x_start = P.x0 - x0
                x_stop = x_start + P.L
                mask[y, x_start:x_stop] = False

#            form_gPPy_(stack)  # evaluate for comp_g, converting stack.Py_ to stack.PPy_

        dert__ = tuple(derts[y0:yn, x0:xn] for derts in frame['dert__'])  # slice each dert array of the whole frame

        fopen = 0  # flag: blob on frame boundary
        if x0 == 0 or xn == frame['dert__'][0].shape[1] or y0 == 0 or yn == frame['dert__'][0].shape[0]:
            fopen = 1

        blob.root_dert__ = frame['dert__']
        blob.box = (y0, yn, x0, xn)
        blob.dert__ = dert__
        blob.mask = mask
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


def form_gPPy_(stack):
    ave_PP = 100  # min summed value of gdert params

    if stack.G > aveG:
        stack_Dg = stack_Mg = 0
        gPPy_ = []  # may replace stack.Py_
        P = stack.Py_[0]

        # initialize PP params:
        Py_ = [P]; PP_I = P.I; PP_Dy = P.Dy; PP_Dx = P.Dx; PP_G = P.G; PP_M = P.M; PP_Dyy = P.Dyy; PP_Dyx = P.Dyx; PP_Dxy = P.Dxy; PP_Dxx = P.Dxx
        PP_Ga = P.Ga; PP_Ma = P.Ma; PP_A = P.L; PP_Ly = 1; PP_y0 = stack.y0

        _PP_sign = PP_G > aveG and P.L > 1

        for P in stack.Py_[1:]:
            PP_sign = P.G > aveG and P.L > 1  # PP sign
            if _PP_sign == PP_sign:  # accum PP:
                Py_.append(P)
                PP_I += P.I
                PP_Dy += P.Dy
                PP_Dx += P.Dx
                PP_G += P.G
                PP_M += P.M
                PP_Dyy += P.Dyy
                PP_Dyx += P.Dyx
                PP_Dxy += P.Dxy
                PP_Dxx += P.Dxx
                PP_Ga += P.Ga
                PP_Ma += P.Ma
                PP_A += P.L
                PP_Ly += 1

            else:  # sign change, terminate PP:
                if PP_G > aveG:
                    Py_, Dg, Mg = comp_g(Py_)  # adds gdert_, Dg, Mg per P in Py_
                    stack_Dg += abs(Dg)  # in all high-G Ps, regardless of direction
                    stack_Mg += Mg
                gPPy_.append(CStack(I=PP_I, Dy = PP_Dy, Dx = PP_Dx, G = PP_G, M = PP_M, Dyy = PP_Dyy, Dyx = PP_Dyx, Dxy = PP_Dxy, Dxx = PP_Dxx,
                                    Ga = PP_Ga, Ma = PP_Ma, A = PP_A, y0 = PP_y0, Ly = PP_Ly, Py_=Py_, sign =_PP_sign ))  # pack PP
                # initialize PP params:
                Py_ = [P]; PP_I = P.I; PP_Dy = P.Dy; PP_Dx = P.Dx; PP_G = P.G; PP_M = P.M; PP_Dyy = P.Dyy; PP_Dyx = P.Dyx; PP_Dxy = P.Dxy; PP_Dxx = P.Dxx
                PP_Ga = P.Ga; PP_Ma = P.Ma; PP_A = P.L; PP_Ly = 1; PP_y0 = stack.y0

            _PP_sign = PP_sign

        if PP_G > aveG:
            Py_, Dg, Mg = comp_g(Py_)  # adds gdert_, Dg, Mg per P
            stack_Dg += abs(Dg)  # stack params?
            stack_Mg += Mg
        if stack_Dg + stack_Mg < ave_PP:  # separate comp_P values, revert to Py_ if below-cost
             # terminate last PP
            gPPy_.append(CStack(I=PP_I, Dy = PP_Dy, Dx = PP_Dx, G = PP_G, M = PP_M, Dyy = PP_Dyy, Dyx = PP_Dyx, Dxy = PP_Dxy, Dxx = PP_Dxx,
                                Ga = PP_Ga, Ma = PP_Ma, A = PP_A, y0 = PP_y0, Ly = PP_Ly, Py_=Py_, sign =_PP_sign ))  # pack PP
            stack.Py_ = gPPy_
            stack.f_gstack = 1  # flag gPPy_ vs. Py_ in stack


def comp_g(Py_):  # cross-comp of gs in P.dert_, in gPP.Py_
    gP_ = []
    gP_Dg = gP_Mg = 0

    for P in Py_:
        Dg=Mg=0
        gdert_ = []
        _g = P.dert_[0][3]  # first g
        for dert in P.dert_[1:]:
            g = dert[3]
            dg = g - _g
            mg = min(g, _g)
            gdert_.append((dg, mg))  # no g: already in dert_
            Dg+=dg  # P-wide cross-sign, P.L is too short to form sub_Ps
            Mg+=mg
            _g = g
        P.gdert_ = gdert_
        P.Dg = Dg
        P.Mg = Mg
        gP_.append(P)
        gP_Dg += Dg
        gP_Mg += Mg  # positive, for stack evaluation to set fPP

    return gP_, gP_Dg, gP_Mg


def form_gP_(gdert_):
    # probably not needed.

    gP_ = []  # initialization
    _g, _Dg, _Mg = gdert_[0]  # first gdert
    _s = _Mg > 0  # initial sign, should we use ave here?

    for (g, Dg, Mg) in gdert_[1:]:
        s = Mg > 0  # current sign
        if _s != s:  # sign change
            gP_.append([_s, _Dg, _Mg])  # pack gP
            # update params
            _s = s
            _Dg = Dg
            _Mg = Mg
        else:  # accumulate params
            _Dg += Dg  # should we abs the value here?
            _Mg += Mg

    gP_.append([_s, _Dg, _Mg])  # pack last gP
    return gP_


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
        blob1.adj_blobs[1] += blob2.Dert['A']
        blob2.adj_blobs[1] += blob1.Dert['A']
        blob1.adj_blobs[2] += blob2.Dert['G']
        blob2.adj_blobs[2] += blob1.Dert['G']
        blob1.adj_blobs[3] += blob2.Dert['M']
        blob2.adj_blobs[3] += blob1.Dert['M']
        blob1.adj_blobs[4] += blob2.Dert['Ma']
        blob2.adj_blobs[4] += blob1.Dert['Ma']

# -----------------------------------------------------------------------------
# Utilities

def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})


def flip_yx(Py_):  # vertical-first run of form_P and deeper functions over blob's ders__

    y0 = 0
    yn = len(Py_)
    x0 = min([P.x0 for P in Py_])
    xn = max([P.x0 + P.L for P in Py_])

    # initialize list containing y and x size, number of sublist = number of params
    dert__ = [(np.zeros((yn - y0, xn - x0)) - 1) for _ in range(len(Py_[0].dert_[0]))]
    mask__ = np.zeros((yn - y0, xn - x0)) > 0

    # insert Py_ value into dert__
    for y, P in enumerate(Py_):
        for x, idert in enumerate(P.dert_):
            for i, (param, dert) in enumerate(zip(idert, dert__)):
                dert[y, x+(P.x0-x0)] = param

    # create mask and set masked area = True
    mask__[np.where(dert__[0] == -1)] = True

    # rotate 90 degree anti-clockwise (np.rot90 rotate 90 degree in anticlockwise direction)
    dert__flip = tuple([np.rot90(dert) for dert in dert__])
    mask__flip = np.rot90(mask__)

    flipped_Py_ = []
    # form vertical patterns after rotation
    from slice_blob import form_P_
    for y, dert_ in enumerate(zip(*dert__flip)):
        crit_ = dert_[3] > 0  # compute crit from G? dert_[3] is G
        P_ = list(form_P_(zip(*dert_), crit_, mask__flip[y])) # convert P_ to list , so that structure is same with Py_

    return flipped_Py_


def draw_stacks(frame):
    '''
    draw stacks per blob
    '''

    import cv2

    for blob_num, blob in enumerate(frame['blob__']):

        # initialization
        y0_ = []
        yn_ = []
        x0_ = []
        xn_ = []

        # loop eachstack
        if len(blob.stack_)>1:

            # retrieve region size of all stacks
            for stack in blob.stack_:
                y0_.append(stack.y0)
                yn_.append(stack.y0 + len(stack.Py_))
                x0_.append(min([P.x0 for P in stack.Py_]))
                xn_.append(max([P.x0 + P.L for P in stack.Py_]))
            y0 = min(y0_)
            yn = max(yn_)
            x0 = min(x0_)
            xn = max(xn_)

            # initialize image and insert value into each stack.
            # image value is start with 1 hence order of stack can be viewed from image value
            img = np.zeros((yn - y0, xn - x0))
            img_value = 1
            for stack in blob.stack_:
                for y, P in enumerate(stack.Py_):
                    for x, dert in enumerate(P.dert_):
                        img[y+(stack.y0-y0), x+(P.x0-x0)] = img_value
                img_value +=1 # increase image value at the end of current stack

            # list of colour for visualization purpose
            colour_list = [ ]
            colour_list.append([255,255,255]) # white
            colour_list.append([200,130,0]) # blue
            colour_list.append([75,25,230]) # red
            colour_list.append([25,255,255]) # yellow
            colour_list.append([75,180,60]) # green
            colour_list.append([212,190,250]) # pink
            colour_list.append([240,250,70]) # cyan
            colour_list.append([48,130,245]) # orange
            colour_list.append([180,30,145]) # purple
            colour_list.append([40,110,175]) # brown

            # initialization
            img_colour = np.zeros((yn - y0, xn - x0,3)).astype('uint8')
            img_index  = np.zeros((yn - y0, xn - x0,3)).astype('uint8')

            total_stacks = len(blob.stack_)
            for i in range(1,total_stacks+1):

                colour_index = i%10
                img_colour[np.where(img==i)] = colour_list[colour_index]
                i_float = float(i)
                img_index[np.where(img==i)] = (((i_float/total_stacks))*205) + 40


            cv2.imwrite('./images/stacks/stacks_blob_'+str(blob_num)+'_colour.bmp',img_colour)
            cv2.imwrite('./images/stacks/stacks_blob_'+str(blob_num)+'_index.bmp',img_index)