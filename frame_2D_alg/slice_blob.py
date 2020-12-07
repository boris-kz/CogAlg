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
from comp_slice_draft import comp_slice_blob
from frame_blobs import CDeepBlob

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback
aveG = 50  # filter for comp_g, assumed constant direction
flip_ave = 1000

# prefix '_' denotes higher-line variable or structure, vs. same-type lower-line variable or structure
# postfix '_' denotes array name, vs. same-name elements of that array. '__' is a 2D array
# prefix 'f' denotes flag

class CP(ClusterStructure):
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
    A = int  # stack area
    Ly = int
    y0 = int
    Py_ = list  # Py_ or dPPy_
    sign = NoneType
    f_gstack = NoneType  # gPPy_ if 1, else Py_
    f_stack_PP = NoneType  # PPy_ if 1, else gPPy_ or Py_
    up_connect_cnt = int
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
    mask__ = object
    fopen = bool
    margin = list
    f_sstack = NoneType  # true if stack_ is sstack_

# Functions:


def slice_blob(sliced_blob, dert__, sign__, mask__, prior_forks, verbose=False, render=False):

    stack_ = deque()  # buffer of running vertical stacks of Ps
    height, width = dert__[0].shape
    if render:
        def output_path(input_path, suffix): return str(Path(input_path).with_suffix(suffix))
        streamer = BlobStreamer(CBlob, dert__[1], record_path=output_path(arguments['image'], suffix='.im2blobs.avi'))
    if verbose: print("Converting to image...")

    for y, dert_ in enumerate(zip(*dert__)):  # first and last row are discarded
        if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()

        P_ = form_P_(list(zip(*dert_)), sign__[y], mask__[y])  # horizontal clustering
        if render: render = streamer.update_blob_conversion(y, P_)

        P_ = scan_P_(P_, stack_, sliced_blob)  # vertical clustering, adds P up_connects and _P down_connect_cnt
        stack_ = form_stack_(P_, sliced_blob, y)

    while stack_:  # dert__ ends, last-line stacks are merged into blob
        form_blob(stack_.popleft(), sliced_blob)

    form_sstack_(sliced_blob)  # cluster stacks into horizontally-oriented super-stacks

    flip_sstack_(sliced_blob)  # vertical-first re-scanning of selected sstacks
    # evaluation is always per sstack
    # need update comp_slice_blob for new sstack structure
    # comp_slice_blob(sliced_blob, AveB)  # cross-comp of vertically consecutive Ps in selected stacks

    if render: path = output_path(arguments['image'], suffix='.im2blobs.jpg'); streamer.end_blob_conversion(y, img_out_path=path)
    # diagnostic code should be as few lines as possible

    return sliced_blob  # sliced_blob instance of CDeepBlob

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

def form_P_(idert_, sign_, mask_):  # segment dert__ into P__, in horizontal ) vertical order

    P_ = deque()  # row of Ps
    x0 = 0
    try:
        while mask_[x0]:  # skip until not masked
            x0 += 1
    except IndexError:
        return P_  # the whole line is masked, return an empty P

    dert_ = [list(idert_[x0])]  # get first dert from idert_ (generator/iterator)
    (I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma), L = dert_[0], 1  # initialize P params
    _s = sign_[x0]
    _mask = mask_[x0]  # mask bit per dert

    for x, (p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma) in enumerate(idert_[x0 + 1:], start=x0 + 1):  # left to right in each row of derts
        mask = mask_[x]
        if ~mask:  # current dert is not masked
            s = sign_[x]
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


def scan_P_(P_, stack_, sliced_blob):  # merge P into higher-row stack of Ps which have same sign and overlap by x_coordinate
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
                        form_blob(stack, sliced_blob)
                    break
            else:  # no next-P overlap
                if stack.down_connect_cnt != 1:  # terminate stack, merge it into up_connects' blobs
                    form_blob(stack, sliced_blob)
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
        form_blob(stack_.popleft(), sliced_blob)  # down_connect_cnt==0

    return next_P_  # each element is P + up_connect_ refs


def form_stack_(P_, sliced_blob, y):  # Convert or merge every P into its stack of Ps, merge blobs

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
                                   y0=y, Py_=[P], blob=blob, down_connect_cnt=0, up_connect_cnt=1, sign=s, fPP=0)
                new_stack.hid = blob.id
                blob.stack_.append(new_stack)  # stack is buffered into blob

                if len(up_connect_) > 1:  # merge blobs of all up_connects
                    if up_connect_[0].down_connect_cnt == 1:  # up_connect is not terminated
                        form_blob(up_connect_[0], sliced_blob)  # merge stack of 1st up_connect into its blob

                    for up_connect in up_connect_[1:len(up_connect_)]:  # merge blobs of other up_connects into blob of 1st up_connect
                        blob.stack_[-1].up_connect_cnt +=1
                        if up_connect.down_connect_cnt == 1:
                            form_blob(up_connect, sliced_blob)

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


def form_blob(stack, sliced_blob):  # increment blob with terminated stack, check for blob termination and merger into frame

    I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, A, Ly, y0, Py_, sign, f_gstack, f_stack_PP, up_connect_cnt, down_connect_cnt, blob, stack_PP = stack.unpack()
    # terminated stack is merged into continued or initialized blob (all connected stacks):
    accum_Dert(blob.Dert, I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=A, Ly=Ly)

    blob.open_stacks += down_connect_cnt - 1  # incomplete stack cnt + terminated stack down_connect_cnt - 1: stack itself
    # open stacks contain Ps of a current row and may be extended with new x-overlapping Ps in next run of scan_P_

    if blob.open_stacks == 0:  # number of incomplete stacks == 0: blob is terminated and packed in frame:
        # if there is no re-evaluation: check for dert__ termination vs. open stacks
        last_stack = stack
        [y0, x0, xn], stack_, s, open_stacks = blob.unpack()[1:5]
        yn = last_stack.y0 + last_stack.Ly

        mask__ = np.ones((yn - y0, xn - x0), dtype=bool)  # mask box, then unmask Ps:

        for stack in stack_:
            for y, P in enumerate(stack.Py_, start=stack.y0 - y0):
                x_start = P.x0 - x0
                x_stop = x_start + P.L
                mask__[y, x_start:x_stop] = False

        dert__ = [derts[y0:yn, x0:xn] for derts in sliced_blob.root_dert__]  # slice all dert array
        sliced_blob.stack_ = blob.stack_
        sliced_blob.dert__ = dert__
        sliced_blob.mask__ = mask__


def form_gPPy_(stack_):  # convert selected stacks into gstacks, should be run over the whole stack_

    ave_PP = 100  # min summed value of gdert params

    for stack in stack_:
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

# -----------------------------------------------------------------------------
# Utilities

def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})


def draw_stacks(frame):
    '''
    draw stacks per blob
    '''
    import cv2
    for blob_num, blob in enumerate(frame['blob__']):
        y0_ = []
        yn_ = []
        x0_ = []
        xn_ = []

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

#           for debug purpose
#           from matplotlib import pyplot as plt
#           plt.imshow(img_colour)
#           plt.pause(1)
            cv2.imwrite('./images/stacks/stacks_blob_'+str(blob_num)+'_colour.bmp',img_colour)
            cv2.imwrite('./images/stacks/stacks_blob_'+str(blob_num)+'_index.bmp',img_index)


def form_sstack_(sliced_blob):
    '''
    form horizontal stacks of stacks
    not updated from Chee's version
    '''
    sstack_ = []
    _stack = sliced_blob.stack_[0]
    _f_up = _stack.up_connect_cnt > 0
    _f_ex = _f_up ^ _stack.down_connect_cnt > 0

    # initialize 1st sstack with _stack params,
    # stack_PP related params only relevant after form_gPPy

    sstack = CStack(I=_stack.I, Dy=_stack.Dy, Dx=_stack.Dx, G=_stack.G, M=_stack.M,
                    Dyy=_stack.Dyy, Dyx=_stack.Dyx, Dxy=_stack.Dxy, Dxx=_stack.Dxx,
                    Ga=_stack.Ga, Ma=_stack.Ma, A=_stack.A, Ly=_stack.Ly, y0=_stack.y0,
                    Py_=[_stack], sign=_stack.sign, blob=_stack.blob)

    for stack in sliced_blob.stack_[1:]:
        f_up = stack.up_connect_cnt > 0
        f_ex = _f_up ^ _stack.down_connect_cnt > 0

        if (f_up != _f_up) and (f_ex and _f_ex):
            # terminate sstack and append it to sstack_
            sstack_.append(sstack)
            # initialize new sstack, all values = 0 or []
            sstack = CStack()

        # append the horizontal stack_ and accumulate sstack params, regardless of termination
        sstack.accumulate(I=stack.I, Dy=stack.Dy, Dx=stack.Dx, G=stack.G, M=stack.M, Dyy=stack.Dyy, Dyx=stack.Dyx, Dxy=stack.Dxy, Dxx=stack.Dxx,
                          Ga=stack.Ga, Ma=stack.Ma, A=stack.A)
        sstack.Ly = max(sstack.y0 + sstack.Ly, stack.y0 + stack.Ly) - min(sstack.y0, stack.y0)
        # 1 line may contain multiple Ps, hence Ly need to be computed from max of y and min of y
        sstack.y0 = min(sstack.y0, stack.y0)  # y0 is min of stacks' y0
        sstack.Py_.append(stack)

        # update prior f_up and f_ex
        _f_up = f_up
        _f_ex = f_ex

    # stack_ = sstack_ if length of sstack_ >0
    if len(sstack_)>0:
        sliced_blob.stack_ = sstack_
        sliced_blob.f_sstack = 1


def flip_sstack_(sliced_blob):  # vertical-first run of form_P and deeper functions over blob's ders__
    '''
    flip selected sstacks
    '''
    for sstack in sliced_blob.stack_:
        f_flip = 0
        # get x0,xn,y0,yn from multiple stacks
        x0_ = xn_ = y0_ = yn_ = []

        for stack in sstack.Py_:
            x0_.append(min([Py.x0 for Py in stack.Py_]))
            xn_.append(max([Py.x0 + Py.L for Py in stack.Py_]))
            y0_.append(stack.y0)
            yn_.append(stack.y0 + stack.Ly)
        x0 = min(x0_)
        xn = max(xn_)
        y0 = min(y0_)
        yn = max(yn_)

        L_bias = (xn - x0 + 1) / (sstack.Ly)
        abs_Dx = abs(sstack.Dx);  if abs_Dx == 0: abs_Dx = 1  # prevent /0
        G_bias = abs(sstack.Dy) / abs_Dx  # ddirection: Gy / Gx, preferential comp over low G

        if sstack.G * sstack.Ma * L_bias * G_bias > flip_ave:  # vertical-first re-scanning of selected sstacks
            f_flip = 1
            dert__ = [(np.zeros((yn - y0, xn - x0)) - 1) for _ in range(len(sstack.Py_[0].dert_[0]))]
            # or we can replace 'len(stack.Py_[0].dert_[0])' with 11, since we would always having 11 params here
            mask__ = np.zeros((yn - y0, xn - x0)) > 0

            for stack in sstack.Py_:
                for y, P in enumerate(sstack.Py_):
                    for x, idert in enumerate(P.dert_):
                        for i, (param, dert) in enumerate(zip(idert, dert__)):
                                dert[y + (sstack.y0 - y0), x + (P.x0 - x0)] = param
            # update mask__
            mask__[np.where(dert__[0] == -1)] = True
            sign__ = dert__[3] * dert__[10] > 0

        if f_flip:  # flip dert, form P
            dert__flip = tuple([np.rot90(dert) for dert in dert__])
            mask__flip = np.rot90(mask__)
            sign__flip = np.rot90(sign__)

            # this section is still tentative
            stack_ = deque()  # buffer of running vertical stacks of Ps
            for y, dert_ in enumerate(zip(*dert__)):  # first and last row are discarded

                P_ = form_P_(list(zip(*dert_)), sign__[y], mask__[y])  # horizontal clustering
                P_ = scan_P_(P_, stack_, sliced_blob)  # vertical clustering, adds P up_connects and _P down_connect_cnt
                stack_ = form_stack_(P_, sliced_blob, y)

            while stack_:  # dert__ ends, last-line stacks are merged into blob
                form_blob(stack_.popleft(), sliced_blob)

        # we need to run form_gPPy per stack after flipping.
        # old comments:
        # form gPPy in sstack
        # form_gPPy_(sstack) # form gPPy after flipping?
        # draw low-ga blob' stacks
        # draw_stacks(frame)
        # evaluate P blobs


