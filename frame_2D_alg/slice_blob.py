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

from collections import deque
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
from class_cluster import ClusterStructure, NoneType

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback, not needed here
aveG = 50  # filter for comp_g, assumed constant direction
flip_ave = 2000

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
    # stack:
    A = int  # stack area
    Ly = int
    y0 = int
    Py_ = list  # Py_ or dPPy_
    sign = NoneType
    f_gstack = NoneType  # gPPy_ if 1, else Py_
    f_stack_PP = NoneType  # PPy_ if 1, else gPPy_ or Py_
    downconnect_cnt = int
    upconnect_ = list
    stack_PP = object
    stack_ = list  # ultimately all stacks, also replaces f_flip: vertical if empty, else horizontal


# Functions:

def slice_blob(blob, verbose=False):

    horizontal_bias = ( blob.box[3] - blob.box[2] + 1) / (blob.box[1] - blob.box[0] + 1) \
                      * (abs(blob.Dy) / abs(blob.Dx))
        # L_bias (Lx / Ly) * G_bias (Gy / Gx), blob.box = [y0,yn,x0,xn], ddirection: , preferential comp over low G

    if horizontal_bias > 1 and (blob.G * blob.Ma * horizontal_bias > flip_ave / 10):
        # rotate for scanning in vertical direction
        blob.fflip = 1   # flip dert__:
        blob.dert__ = tuple([np.rot90(dert) for dert in blob.dert__])
        blob.mask__ = np.rot90(blob.mask__)

    dert__ = blob.dert__
    mask__ = blob.mask__
    row_stack_ = []  # higher-row vertical stacks of Ps
    stack_ = []  # all terminated stacks
    height, width = dert__[0].shape
    if verbose: print("Converting to image...")

    for y, dert_ in enumerate(zip(*dert__)):  # first and last row are discarded?
        if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()

        P_ = form_P_(list(zip(*dert_)), mask__[y])  # horizontal clustering
        P_ = scan_P_(P_, row_stack_)  # vertical clustering, adds P upconnects and stack downconnect_cnt
        next_row_stack_ = form_stack_(P_, y)  # initialize and accumulate stacks with Ps, including the whole 1st row_stack_

        [stack_.append(stack) for stack in row_stack_ if not stack in next_row_stack_]  # buffer stacks that won't be used in next row
        row_stack_ = next_row_stack_  # stacks initialized or accumulated in form_stack_

    stack_ += row_stack_  # dert__ ends, all last-row stacks have no downconnects

    check_stacks_presence(stack_, mask__, f_plot=0)  # visualize stack_ and mask__

    sstack_ = form_sstack_(stack_)  # cluster stacks into horizontally-oriented super-stacks

    flip_sstack_(sstack_, dert__)  # vertical-first re-scanning of selected sstacks

    draw_sstack_(blob.fflip, stack_,sstack_) # draw stacks, sstacks and # draw stacks, sstacks and the rotated sstacks

    for sstack in sstack_:  # convert selected stacks into gstacks
        form_gPPy_(sstack.stack_)  # sstack.Py_ = stack_

    return sstack_  # partially rotated and gP-forming stack__

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


def scan_P_(P_, row_stack_):  # merge P into higher-row stack of Ps which have same sign and overlap by x_coordinate
    '''
    Each P in P_ scans higher-row _Ps (in stack_) left-to-right, testing for x-overlaps between Ps and same-sign _Ps.
    Overlap is represented as upconnect in P and is added to downconnect_cnt in _P. Scan continues until P.x0 >= _P.xn:
    no x-overlap between P and next _P. Then P is packed into its upconnect stack or initializes a new stack.
    After such test, loaded _P is also tested for x-overlap to the next P.
    If negative, a stack with loaded _P is removed from stack_ (buffer of higher-row stacks) and tested for downconnect_cnt==0.
    If so: no lower-row connections, the stack is packed into connected blobs (referred by its upconnect_),
    else the stack is recycled into next_stack_, for next-row run of scan_P_.
    It's a form of breadth-first flood fill, with connects as vertices per stack of Ps: a node in connectivity graph.
    '''
    next_P_ = deque()  # to recycle P + upconnect_ that finished scanning _P, will be converted into next_stack_

    if P_ and row_stack_:  # there are next P and next stack

        i = 0  # stack index
        P = P_.popleft()  # load left-most (lowest-x) input-row P
        stack = row_stack_[i]  # higher-row stacks
        _P = stack.Py_[-1]  # last element of each stack is higher-row P
        upconnect_ = []  # list of same-sign x-overlapping _Ps per P

        while True:           # loop until break
            x0 = P.x0         # first x in P
            xn = x0 + P.L     # first x in next P
            _x0 = _P.x0       # first x in _P
            _xn = _x0 + _P.L  # first x in next _P

            if _x0 - 1 < xn and _xn + 1 > x0:  # x overlap between P and _P in 8 directions: +ma blob is less selective
                stack.downconnect_cnt += 1
                upconnect_.append(stack)  # P-connected higher-row stacks

            if xn <= _xn:  # _P overlaps next P in P_ in 8 directions, if (xn < _xn) for 4 directions
                next_P_.append((P, upconnect_))  # recycle _P for the next run of scan_P_
                if P_:
                    upconnect_ = []  # reset for next P, if any
                    P = P_.popleft()  # load next P
                else:  # terminate loop
                    break
            else:  # no next-P overlap for current stack
                i += 1
                if i < len(row_stack_):  # load stack with next _P
                    stack = row_stack_[i]
                    _P = stack.Py_[-1]
                else:  # no stack left: terminate loop
                    next_P_.append((P, upconnect_))
                    break

    while P_:  # terminate Ps that continue at row's end
        next_P_.append((P_.popleft(), []))  # no upconnect

    return next_P_  # each element is P + upconnect_ refs


def form_stack_(P_, y):  # Convert or merge every P into higher-row stack of Ps

    # no termination test: P upconnects don't include stacks with 0 downconnects
    next_row_stack_ = []  # converted to row_stack_ in the next run of scan_P_

    while P_:
        P, upconnect_ = P_.popleft()
        I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, L, x0, s, dert_, _, _, _ = P.unpack()
        if not upconnect_:
            # initialize new stack for each input-row P that has no connections in higher row, as in the whole top row:
            stack = CStack(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=L, Ly=1,
                           y0=y, Py_=[P], downconnect_cnt=0, upconnect_=upconnect_, sign=s, fPP=0)
        else:
            if len(upconnect_) == 1 and upconnect_[0].downconnect_cnt == 1:
                # P has one upconnect and that upconnect has one downconnect=P: merge P into upconnect' stack:
                stack = upconnect_[0]
                stack.accumulate(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=L, Ly=1)
                stack.Py_.append(P)   # Py_: vertical buffer of Ps
                stack.downconnect_cnt = 0  # reset downconnect_cnt to that of last P

            else:  # P has >1 upconnects, or 1 upconnect that has >1 downconnects: initialize stack with P:
                stack = CStack(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=L, Ly=1,
                               y0=y, Py_=[P], downconnect_cnt=0, upconnect_=upconnect_, sign=s, fPP=0)

        next_row_stack_.append(stack)

    return next_row_stack_  # input for the next line of scan_P_


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


def form_sstack_(stack_):
    '''
    just a draft
    form horizontal stacks of stacks, read backwards, sub-access by upconnects?
    '''
    sstack_ = []
    '''
    _stack = stack_.copy.pop
    _f_up = len(_stack.upconnect_) > 0
    _f_ex = _f_up ^ _stack.downconnect_cnt > 0  # exclusive up- or down- connectivity

    # initialize 1st sstack with _stack params:
    sstack = CStack(I=_stack.I, Dy=_stack.Dy, Dx=_stack.Dx, G=_stack.G, M=_stack.M,
                    Dyy=_stack.Dyy, Dyx=_stack.Dyx, Dxy=_stack.Dxy, Dxx=_stack.Dxx,
                    Ga=_stack.Ga, Ma=_stack.Ma, A=_stack.A, Ly=_stack.Ly, y0=_stack.y0,
                    Py_=[_stack], sign=_stack.sign)
    '''
    for _stack in stack_.copy.pop:  # access in termination order

        if _stack.downconnect_cnt == 0:  # else skip, this stack is upconnect of prior _stack
            _f_up = len(_stack.upconnect_) > 0
            _f_ex = _f_up ^ _stack.downconnect_cnt > 0  # exclusive up- or down- connectivity

            sstack = CStack(I=_stack.I, Dy=_stack.Dy, Dx=_stack.Dx, G=_stack.G, M=_stack.M,
                            Dyy=_stack.Dyy, Dyx=_stack.Dyx, Dxy=_stack.Dxy, Dxx=_stack.Dxx,
                            Ga=_stack.Ga, Ma=_stack.Ma, A=_stack.A, Ly=_stack.Ly, y0=_stack.y0,
                            Py_=[_stack], sign=_stack.sign)

        for stack in _stack.upconnect_:  # access connected stacks only
            f_up = len(stack.upconnect_) > 0
            f_ex = f_up ^ stack.downconnect_cnt > 0

            if (f_up != _f_up) and (f_ex and _f_ex):
                # one of consecutive stacks is upconnected, the other is downconnected, both of are exclusively connected:
                # direction is reversed, which means the stacks are ordered horizontally

                # accumulate instead of termination:
                sstack.accumulate(I=stack.I, Dy=stack.Dy, Dx=stack.Dx, G=stack.G, M=stack.M, Dyy=stack.Dyy, Dyx=stack.Dyx, Dxy=stack.Dxy, Dxx=stack.Dxx,
                                  Ga=stack.Ga, Ma=stack.Ma, A=stack.A)
                sstack.Ly = max(sstack.y0 + sstack.Ly, stack.y0 + stack.Ly) - min(sstack.y0, stack.y0)  # Ly = max y - min y: line may contain multiple Ps
                sstack.y0 = min(sstack.y0, stack.y0)  # y0 is min of stacks' y0
                sstack.Py_.append(stack)

            else:  # needs a review:
                sstack_.append(sstack)  # terminate sstack and append it to sstack_
                sstack = CStack()       # initialize sstack, all values = 0 or []

        # update prior f_up and f_ex
        _f_up = f_up
        _f_ex = f_ex

    sstack_.append(sstack)  # terminate last sstack

    return sstack_


def flip_sstack_(sstack_, dert__):
    '''
    evaluate for flipping dert__ and re-forming Ps and stacks per sstack
    '''
    for sstack in sstack_:
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

        horizontal_bias = ((xn - x0 + 1) / sstack.Ly) * (abs(sstack.Dy) / (abs(sstack.Dx) + 1))
        # horizontal_bias = L_bias (lx / Ly) * G_bias (Gy / Gx, preferential comp over low G)

        if horizontal_bias > 1 and (sstack.G * sstack.Ma * horizontal_bias > flip_ave):
            # vertical-first rescan of selected sstacks:

            sstack_mask__ = np.ones((yn - y0, xn - x0)).astype(bool)
            # unmask sstack:
            for stack in sstack.Py_:
                for y, P in enumerate(stack.Py_):
                    # we need -x0 in x index so that P.x0 is relative to x0
                    sstack_mask__[y+(stack.y0-y0), P.x0-x0: (P.x0-x0 + P.L)] = False  # unmask P

            # extract and flip sstack's dert__
            sstack_dert__ = tuple([np.rot90(param_dert__[y0:yn, x0:xn]) for param_dert__ in dert__])
            sstack_mask__ = np.rot90(sstack_mask__)  # flip mask

            row_stack_ = []
            for y, dert_ in enumerate(zip(*sstack_dert__)):  # same operations as in root sliced_blob(), but on sstack

                P_ = form_P_(list(zip(*dert_)), sstack_mask__[y])  # horizontal clustering
                P_ = scan_P_(P_, row_stack_)  # vertical clustering, adds P upconnects and _P downconnect_cnt
                next_row_stack_ = form_stack_(P_, y)  # initialize and accumulate stacks with Ps

                [sstack.stack_.append(stack) for stack in row_stack_ if not stack in next_row_stack_]  # buffer terminated stacks
                row_stack_ = next_row_stack_

            sstack.stack_ += row_stack_   # dert__ ends, all last-row stacks have no downconnects

            check_stacks_presence(sstack.stack_, sstack_mask__, f_plot=0)
# -----------------------------------------------------------------------------
# Utilities

def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})


def draw_stacks(stack_):
    # retrieve region size of all stacks
    y0 = min([stack.y0 for stack in stack_])
    yn = max([stack.y0 + stack.Ly for stack in stack_])
    x0 = min([P.x0 for stack in stack_ for P in stack.Py_])
    xn = max([P.x0 + P.L for stack in stack_ for P in stack.Py_])

    img = np.zeros((yn - y0, xn - x0))
    stack_index = 1

    for stack in stack_:
        for y, P in enumerate(stack.Py_):
            for x, dert in enumerate(P.dert_):
                img[y + (stack.y0 - y0), x + (P.x0 - x0)] = stack_index
        stack_index += 1  # for next stack

    colour_list = []  # list of colours:
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
    total_stacks = stack_index

    for i in range(1, total_stacks + 1):
        colour_index = i % 10
        img_colour[np.where(img == i)] = colour_list[colour_index]
    #    cv2.imwrite('./images/stacks/stacks_blob_' + str(sliced_blob.id) + '_colour.bmp', img_colour)

    return img_colour

def check_stacks_presence(stack_, mask__, f_plot=0):
    '''
    visualize stack_ and mask__ to ensure that they cover the same area
    '''
    img_colour = draw_stacks(stack_)  # visualization to check if we miss out any stack from the mask
    img_sum = img_colour[:, :, 0] + img_colour[:, :, 1] + img_colour[:, :, 2]

    # check if all unmasked area is filled in plotted image
    check_ok_y = np.where(img_sum > 0)[0].all() == np.where(mask__ == False)[0].all()
    check_ok_x = np.where(img_sum > 0)[1].all() == np.where(mask__ == False)[1].all()

    # check if their shape is the same
    check_ok_shape_y = img_colour.shape[0] == mask__.shape[0]
    check_ok_shape_x = img_colour.shape[1] == mask__.shape[1]

    if not check_ok_y or not check_ok_x or not check_ok_shape_y or not check_ok_shape_x:
        print("----------------Missing stacks----------------")

    if f_plot:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img_colour)
        plt.subplot(1, 2, 2)
        plt.imshow(mask__ * 255)



def draw_sstack_(f_flip, stack_,sstack_):
    '''visualize stacks and sstacks and their scanning direction'''
    ''' runnable but not fully optimized yet '''
    colour_list = []  # list of colours:
    colour_list.append([200, 130, 0])  # blue
    colour_list.append([75, 25, 230])  # red
    colour_list.append([25, 255, 255])  # yellow
    colour_list.append([75, 180, 60])  # green
    colour_list.append([212, 190, 250])  # pink
    colour_list.append([240, 250, 70])  # cyan
    colour_list.append([48, 130, 245])  # orange
    colour_list.append([180, 30, 145])  # purple
    colour_list.append([40, 110, 175])  # brown
    colour_list.append([255, 255, 255])  # white

    mix_flip = 0
    if f_flip: # if stacks are flipped and scanned vertically in blob's derts

        # get boundary - x0,xn, y0, yn
        x0_rot, xn_rot, y0_rot, yn_rot = [], [], [], []
        for sstack in sstack_:
            for stack in sstack.Py_:
                x0_rot.append(min([Py.x0 for Py in stack.Py_]))
                xn_rot.append(max([Py.x0 + Py.L for Py in stack.Py_]))
                y0_rot.append(stack.y0)
                yn_rot.append(stack.y0 + stack.Ly)

        x0 = min(y0_rot)
        xn = max(yn_rot)
        y0 = min(x0_rot)
        yn = max(xn_rot)

        img_index_stacks = np.zeros((yn - y0, xn - x0))
        img_index_sstacks = np.zeros((yn - y0, xn - x0))
        img_index_sstacks_flipped = np.zeros((yn - y0, xn - x0))

        stack_index = 1
        sstack_index = 1
        sstack_flipped_index = 1

        for sstack in sstack_:
            for stack in sstack.Py_:
                for y, P in enumerate(stack.Py_):
                    for x, dert in enumerate(P.dert_):
                            img_index_stacks[x + (P.x0 - x0),xn-1-(y + (stack.y0 - y0))] = stack_index
                            img_index_sstacks[x + (P.x0 - x0),xn-1-(y + (stack.y0 - y0))] = sstack_index
                stack_index += 1 # for next stack
            sstack_index += 1  # for next sstack

        for sstack in sstack_:
            if sstack.stack_: # if sstack is flipped (blob's derts are flipped, and sstack is flipped too)
                mix_flip = 1
                sx0_, sxn_, sy0_, syn_ = [], [], [], []
                for stack in sstack.Py_:
                    sx0_.append(min([Py.x0 for Py in stack.Py_]))
                    sxn_.append(max([Py.x0 + Py.L for Py in stack.Py_]))
                    sy0_.append(stack.y0)
                    syn_.append(stack.y0 + stack.Ly)

                sx0 = min(sy0_)
                sxn = max(syn_)
                sy0 = min(sx0_)
                syn = max(sxn_)

                for stack in sstack.stack_:
                    for y, P in enumerate(stack.Py_):
                        for x, dert in enumerate(P.dert_):
                            img_index_sstacks_flipped[syn-1-(y + (stack.y0 - y0)), xn-sx0-1-(x + (P.x0 - x0))] = sstack_flipped_index
                    sstack_flipped_index += 1  # for next stack of flipped sstack


            else: # if sstack is not flipped (blob's derts are flipped, but sstack is not flipped)
                for stack in sstack.Py_:
                    for y, P in enumerate(stack.Py_):
                        for x, dert in enumerate(P.dert_):
                            img_index_sstacks_flipped[x + (P.x0 - x0),xn-1-(y + (stack.y0 - y0))] = sstack_flipped_index
                    sstack_flipped_index += 1  # for next stack of sstack


    else: # if stacks are not flipped and scanned horizontally in blob's derts

        # get boundary - x0,xn, y0, yn
        x0_, xn_, y0_, yn_ = [], [], [], []
        for sstack in sstack_:
            for stack in sstack.Py_:
                x0_.append(min([Py.x0 for Py in stack.Py_]))
                xn_.append(max([Py.x0 + Py.L for Py in stack.Py_]))
                y0_.append(stack.y0)
                yn_.append(stack.y0 + stack.Ly)

        x0 = min(x0_)
        xn = max(xn_)
        y0 = min(y0_)
        yn = max(yn_)

        img_index_stacks = np.zeros((yn - y0, xn - x0))
        img_index_sstacks = np.zeros((yn - y0, xn - x0))
        img_index_sstacks_flipped = np.zeros((yn - y0, xn - x0))

        stack_index = 1
        sstack_index = 1
        sstack_flipped_index = 1

        for sstack in sstack_:
            for stack in sstack.Py_:
                for y, P in enumerate(stack.Py_):
                    for x, dert in enumerate(P.dert_):
                        img_index_stacks[y + (stack.y0 - y0), x + (P.x0 - x0)] = stack_index
                        img_index_sstacks[y + (stack.y0 - y0), x + (P.x0 - x0)] = sstack_index
                stack_index += 1  # for next stack
            sstack_index += 1  # for next sstack


        for sstack in sstack_:
            if sstack.stack_: # if sstack is flipped (blob's derts are not flipped, but sstack is flipped)
                mix_flip = 1
                sx0_, sxn_, sy0_, syn_ = [], [], [], []
                for stack in sstack.Py_:
                    sx0_.append(min([Py.x0 for Py in stack.Py_]))
                    sxn_.append(max([Py.x0 + Py.L for Py in stack.Py_]))
                    sy0_.append(stack.y0)
                    syn_.append(stack.y0 + stack.Ly)

                sx0 = min(sx0_)
                sxn = max(sxn_)
                sy0 = min(sy0_)
                syn = max(syn_)

                for stack in sstack.stack_:
                    for y, P in enumerate(stack.Py_):
                        for x, dert in enumerate(P.dert_):
                            img_index_sstacks_flipped[sy0+(x + (P.x0 - x0)),sxn-1-(y + (stack.y0 - y0))] = sstack_flipped_index

                    sstack_flipped_index += 1  # for next stack of flipped sstack


            else: # if sstack is not flipped (blob's derts are not flipped,  sstack is also not flipped)
                for stack in sstack.Py_:
                    for y, P in enumerate(stack.Py_):
                        for x, dert in enumerate(P.dert_):
                            img_index_sstacks_flipped[y + (stack.y0 - y0),x + (P.x0 - x0)] = sstack_flipped_index
                    sstack_flipped_index += 1  # for next stack of sstack


    # initialize colour image
    img_colour_stacks = np.zeros((yn-y0, xn-x0, 3)).astype('uint8')
    img_colour_sstacks = np.zeros((yn-y0, xn-x0, 3)).astype('uint8')
    img_colour_sstacks_flipped = np.zeros((yn-y0, xn-x0, 3)).astype('uint8')

    # draw stacks and sstacks
    for i in range(1, stack_index):
        colour_index = i % 10
        img_colour_stacks[np.where(img_index_stacks == i)] = colour_list[colour_index]
    for i in range(1, sstack_index):
        colour_index = i % 10
        img_colour_sstacks[np.where(img_index_sstacks == i)] = colour_list[colour_index]
    for i in range(1, sstack_flipped_index):
        colour_index = i % 10
        img_colour_sstacks_flipped[np.where(img_index_sstacks_flipped == i)] = colour_list[colour_index]


    # draw image to figure and save it to disk
    plt.figure(1)
    plt.subplot(1,3,1)
    plt.imshow(np.uint8(img_colour_stacks))
    if f_flip: plt.title('Y stacks')
    else: plt.title('X stacks')
    plt.subplot(1,3,2)
    plt.imshow(img_colour_sstacks)
    if f_flip: plt.title('Y sstacks')
    else: plt.title('X sstacks')
    plt.subplot(1,3,3)
    plt.imshow(img_colour_sstacks_flipped)
    if mix_flip: plt.title('Mix XY sstacks')
    elif f_flip: plt.title('Y sstacks')
    else: plt.title('X sstacks')
    plt.savefig('./images/slice_blob/sstack_'+str(id(sstack_))+'.png')
    plt.close()
