'''
    This is a terminal fork of intra_blob.
    slice_blob converts selected smooth-edge blobs (high G, low Ga) into sliced blobs, adding internal P and stack structures.
    It then calls comp_g and comp_P per stack, to trace and vectorize these edges.
    No vectorization for flat blobs?

    Pixel-level parameters are accumulated in contiguous spans of same-sign gradient, first horizontally then vertically.
    Horizontal blob slices are Ps: 1D patterns, and vertical spans are first stacks of Ps, then blob of connected stacks.

    This processing adds a level of encoding per row y, defined relative to y of current input row, with top-down scan:
    1Le, line y-1: form_P( dert_) -> 1D pattern P: contiguous row segment, a slice of a blob
    2Le, line y-2: scan_P_(P, hP) -> hP, up_connect_, down_connect_count: vertical connections per stack of Ps
    3Le, line y-3: form_stack(hP, stack) -> stack: merge vertically-connected _Ps into non-forking stacks of Ps
    4Le, line y-4+ stack depth: term_stack(stack, blob): merge terminated stack into up_connect_, recursively
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
from frame_blobs import CBlob
from comp_slice_draft import comp_slice_blob

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback, not needed here
aveG = 50  # filter for comp_g, assumed constant direction
flip_ave = 1000
open_stacks = 0  # not needed for single blob?
term_stack_ = []

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
    # Dert:
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
    f_flip = NoneType  # horizontal if 1, else vertical
    up_connect_cnt = int
    down_connect_cnt = int
    stack_PP = object

# Functions:

def slice_blob(dert__, mask__, verbose=False):

    stack_ = deque()  # buffer of running vertical stacks of Ps
    height, width = dert__[0].shape
    if verbose: print("Converting to image...")

    for y, dert_ in enumerate(zip(*dert__)):  # first and last row are discarded?
        if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()

        P_ = form_P_(list(zip(*dert_)), mask__[y])  # horizontal clustering
        P_ = scan_P_(P_, stack_)  # vertical clustering, adds P up_connects and _P down_connect_cnt
        stack_ = form_stack_(P_, stack_, term_stack_, open_stacks, y)  # accumulate and terminate stacks

    # last line form_stack_()?
    # term_stack_ for direct access? or no explicit termination, sequential access through last-row stack_?

    form_sstack_(stack_)  # cluster stacks into horizontally-oriented super-stacks
    draw_stacks (stack_)  # visualization
#   flip_sstack_(stack_)  # vertical-first re-scanning of selected sstacks

    for sstack in stack_:  # convert selected stacks into gstacks
        form_gPPy_(sstack.Py_)  # sstack.Py_ = stack_

    return stack_  # added to blob, sequential access to higher stacks?

'''
Parameterized connectivity clustering functions below:
- form_P sums dert params within P and increments its L: horizontal length.
- scan_P_ searches for horizontal (x) overlap between Ps of consecutive (in y) rows.
- form_stack combines these overlapping Ps into vertical stacks of Ps, with one up_P to one down_P
- term_stack merges terminated stacks into blob

dert: tuple of derivatives per pixel, initially (p, dy, dx, g), extended in intra_blob
Dert: params of cluster structures (P, stack, blob): summed dert params + dimensions: vertical Ly and area A
'''

def form_P_(idert_, mask_):  # segment dert__ into P__, in horizontal ) vertical order

    P_ = deque()  # row of Ps
    dert_ = [list(idert_[0])]  # get first dert from idert_ (generator/iterator)
    _mask = mask_[0]  # mask bit per dert
    if ~_mask:
        I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = dert_[0]; L = 1; x0=0  # initialize P params with first dert

    for x, dert in enumerate(idert_[1:], start=1):  # left to right in each row of derts
        mask = mask_[x]  # masks = 1,_0: P termination, 0,_1: P initialization, 0,_0: P accumulation:
        if mask:
            if ~_mask:  # _dert is not masked, dert is masked, terminate P:
                P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, dert_=dert_)
                P_.append(P)
        else:  # dert is not masked
            if _mask:  # _dert is masked, initialize P params:
                I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = dert; L = 1; x0=x; dert_=[dert]
            else:
                I += dert[0]  # _dert is not masked, accumulate P params with (p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma) = dert
                Dy += dert[1]
                Dx += dert[2]
                G += dert[3]
                M += dert[4]
                Dyy += dert[5]
                Dyx += dert[6]
                Dxy += dert[7]
                Dxx += dert[8]
                Ga += dert[9]
                Ma += dert[10]
                L += 1
                dert_.append(dert)
        _mask = mask

    if ~_mask:  # terminate last P in a row
        P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, dert_=dert_)
        P_.append(P)

    return P_


def scan_P_(P_, stack_):  # merge P into higher-row stack of Ps which have same sign and overlap by x_coordinate
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

            x0 = P.x0         # first x in P
            xn = x0 + P.L     # first x in next P
            _x0 = _P.x0       # first x in _P
            _xn = _x0 + _P.L  # first x in next _P

            if (_x0 - 1 < xn and x0 < _xn + 1) and (P.sign == stack.sign):
                # x overlap between P and _P in 8 directions: +ma blob is less selective
                stack.down_connect_cnt += 1
                up_connect_.append(stack)  # P-connected higher-row stacks
            if (xn <= _xn):
                # _P overlaps next P in P_ in 8 directions, else if (xn < _xn)
                next_P_.append((P, up_connect_))  # recycle _P for the next run of scan_P_
                up_connect_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:  # terminate loop
                    break
            else:  # no next-P overlap
                if stack_:  # load stack with next _P
                    stack = stack_.popleft()
                    _P = stack.Py_[-1]
                else:  # no stack left: terminate loop
                    next_P_.append((P, up_connect_))
                    break

    while P_:  # terminate Ps that continue at row's end
        next_P_.append((P_.popleft(), []))  # no up_connect

    return next_P_  # each element is P + up_connect_ refs


def form_stack_(P_, stack_, term_stack_, open_stacks, y):

    # Convert or merge every P into its higher-row stack of Ps, possibly terminate stacks?
    next_stack_ = deque()  # converted to stack_ in the next run of scan_P_

    while P_:
        P, up_connect_ = P_.popleft()
        I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, L, x0, s, dert_, _, _, _ = P.unpack()
        xn = x0 + L  # next-P x0?
        if not up_connect_:
            # initialize new stack for each input-row P that has no connections in higher row, as in the whole top row:
            open_stacks += 1
            new_stack = CStack(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=L, Ly=1,
                               y0=y, Py_=[P], down_connect_cnt=0, up_connect_cnt=0, sign=s, fPP=0)
        else:
            if len(up_connect_) == 1 and up_connect_[0].down_connect_cnt == 1:
                # P has one up_connect and that up_connect has one down_connect=P: merge P into up_connect stack:
                new_stack = up_connect_[0]
                new_stack.accumulate(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=L, Ly=1)
                new_stack.Py_.append(P)   # Py_: vertical buffer of Ps
                new_stack.down_connect_cnt = 0  # reset down_connect_cnt

            else:  # P has >1 up_connects, or 1 up_connect that has >1 down_connect_cnt:
                # initialize stack with P:
                new_stack = CStack(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=L, Ly=1,
                                   y0=y, Py_=[P], down_connect_cnt=0, up_connect_cnt=1, sign=s, fPP=0)

                if len(up_connect_) > 1:
                    if up_connect_[0].down_connect_cnt == 1:  # up_connect is not terminated
                        stack_.append(new_stack)
                    for up_connect in up_connect_[1:len(up_connect_)]:
                        stack_[-1].up_connect_cnt +=1

                    # wrong, just pick 1st up_connect with 1 down_connect, the rest refer to that stack?
                    # same as before, but with stacks, not blobs?

        P.hid = new_stack.id
        next_stack_.append(new_stack)

    return next_stack_  # input for the next line of scan_P_


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
                    Py_ = [P]; PP_I = P.I; PP_Dy = P.Dy; PP_Dx = P.Dx; PP_G = P.G; PP_M = P.M; PP_Dyy = P.Dyy; PP_Dyx = P.Dyx; PP_Dxy = P.Dxy
                    PP_Dxx = P.Dxx; PP_Ga = P.Ga; PP_Ma = P.Ma; PP_A = P.L; PP_Ly = 1; PP_y0 = stack.y0

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


def draw_stacks(sliced_blob):
    '''
    tentative, still buggy
    '''
    import cv2

    y0_, yn_, x0_, xn_ = [],[],[],[]

    for sstack in sliced_blob.stack_:
        for stack in sstack.Py_:
            # retrieve region size of all stacks
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
    for sstack in sliced_blob.stack_:
        for stack in sstack.Py_:
            for y, P in enumerate(stack.Py_):
                for x, dert in enumerate(P.dert_):
                    img[y + (stack.y0 - y0), x + (P.x0 - x0)] = img_value
            img_value += 1  # increase image value at the end of current stack

    # list of colour for visualization purpose
    colour_list = []
    colour_list.append([255, 255, 255])  # white
    colour_list.append([200, 130, 0])  # blue
    colour_list.append([75, 25, 230])  # red
    colour_list.append([25, 255, 255])  # yellow
    colour_list.append([75, 180, 60])  # green
    colour_list.append([212, 190, 250])  # pink
    colour_list.append([240, 250, 70])  # cyan
    colour_list.append([48, 130, 245])  # orange
    colour_list.append([180, 30, 145])  # purple
    colour_list.append([40, 110, 175])  # brown

    # initialization
    img_colour = np.zeros((yn - y0, xn - x0, 3)).astype('uint8')
    total_stacks = img_value

    for i in range(1, total_stacks + 1):
        colour_index = i % 10
        img_colour[np.where(img == i)] = colour_list[colour_index]

    cv2.imwrite('./images/stacks/stacks_blob_' + str(sliced_blob.id) + '_colour.bmp', img_colour)


def form_sstack_(sliced_blob):
    '''
    form horizontal stacks of stacks
    '''
    sstack_ = []
    _stack = sliced_blob.stack_[0]
    _f_up = _stack.up_connect_cnt > 0
    _f_ex = _f_up ^ _stack.down_connect_cnt > 0
    # initialize 1st sstack with _stack params:

    sstack = CStack(I=_stack.I, Dy=_stack.Dy, Dx=_stack.Dx, G=_stack.G, M=_stack.M,
                    Dyy=_stack.Dyy, Dyx=_stack.Dyx, Dxy=_stack.Dxy, Dxx=_stack.Dxx,
                    Ga=_stack.Ga, Ma=_stack.Ma, A=_stack.A, Ly=_stack.Ly, y0=_stack.y0,
                    Py_=[_stack], sign=_stack.sign, blob=_stack.blob)

    for stack in sliced_blob.stack_[1:]:
        f_up = stack.up_connect_cnt > 0
        f_ex = _f_up ^ _stack.down_connect_cnt > 0

        if (f_up != _f_up) and (f_ex and _f_ex):
            sstack_.append(sstack)  # terminate sstack and append it to sstack_
            sstack = CStack()       # initialize sstack, all values = 0 or []

        # append the horizontal stack_ and accumulate sstack params, regardless of termination
        sstack.accumulate(I=stack.I, Dy=stack.Dy, Dx=stack.Dx, G=stack.G, M=stack.M, Dyy=stack.Dyy, Dyx=stack.Dyx, Dxy=stack.Dxy, Dxx=stack.Dxx,
                          Ga=stack.Ga, Ma=stack.Ma, A=stack.A)
        sstack.Ly = max(sstack.y0 + sstack.Ly, stack.y0 + stack.Ly) - min(sstack.y0, stack.y0)  # Ly = max y - min y: line may contain multiple Ps
        sstack.y0 = min(sstack.y0, stack.y0)  # y0 is min of stacks' y0
        sstack.Py_.append(stack)

        # update prior f_up and f_ex
        _f_up = f_up
        _f_ex = f_ex

    sstack_.append(sstack)  # terminate last sstack
    sliced_blob.stack_ = sstack_


def flip_sstack_(sliced_blob):
    '''
    flip and re-form Ps and stacks in selected sstacks
    '''
    for sstack in sliced_blob.stack_:
        x0_, xn_, y0_, yn_ = [],[],[],[]
        # find min and max x and y in sstack:
        for stack in sstack.Py_:
            x0_.append(min([Py.x0 for Py in stack.Py_]))
            xn_.append(max([Py.x0 + Py.L for Py in stack.Py_]))
            y0_.append(stack.y0)
            yn_.append(stack.y0 + stack.Ly)
        x0 = min(x0_)
        xn = max(xn_)
        y0 = min(y0_)
        yn = max(yn_)

        L_bias = (xn - x0 + 1) / (sstack.Ly)  # ratio of lx to Ly
        G_bias = abs(sstack.Dy) / (abs(sstack.Dx) + 1)  # ddirection: Gy / Gx, preferential comp over low G

        if sstack.G * sstack.Ma * L_bias * G_bias > flip_ave:  # vertical-first rescan of selected sstacks
            sstack.f_flip = 1

            sstack_mask__ = np.ones((yn - y0, xn - x0)).astype(bool)
            # unmask sstack:
            for stack in sstack.Py_:
                for y, P in enumerate(stack.Py_):
                    sstack_mask__[y, P.x0: (P.x0 + P.L)] = False  # unmask P

            # extract and flip sstack's dert__
            sstack_dert__ = tuple([np.rot90(param_dert__[y0:yn, x0:xn]) for param_dert__ in sliced_blob.dert__])
            sstack_mask__ = np.rot90(sstack_mask__)  # flip mask

            stack_ = deque()
            for y, dert_ in enumerate(zip(*sstack_dert__)):  # same operations as in root sliced_blob(), but on sstack

                P_ = form_P_(list(zip(*dert_)), sstack_mask__[y])  # horizontal clustering
                P_ = scan_P_(P_, stack_, sstack)  # vertical clustering, adds P up_connects and _P down_connect_cnt
                stack_ = form_stack_(P_, sstack, y)

            while stack_:  # dert__ ends, last-line stacks are merged into blob
                form_stack_(stack_.popleft().Py[0], sstack, y)  # probably wrong