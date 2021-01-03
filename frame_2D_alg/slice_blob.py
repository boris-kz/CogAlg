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

def slice_blob(dert__, mask__, verbose=False):

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

    draw_sstack_(sstack_) # draw stacks, sstacks and the rotated sstacks

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
    form horizontal stacks of stacks
    '''
    sstack_ = []
    _stack = stack_[0]
    _f_up = len(_stack.upconnect_) > 0
    _f_ex = _f_up ^ _stack.downconnect_cnt > 0
    # initialize 1st sstack with _stack params:

    sstack = CStack(I=_stack.I, Dy=_stack.Dy, Dx=_stack.Dx, G=_stack.G, M=_stack.M,
                    Dyy=_stack.Dyy, Dyx=_stack.Dyx, Dxy=_stack.Dxy, Dxx=_stack.Dxx,
                    Ga=_stack.Ga, Ma=_stack.Ma, A=_stack.A, Ly=_stack.Ly, y0=_stack.y0,
                    Py_=[_stack], sign=_stack.sign)

    for stack in stack_[1:]:
        f_up = len(stack.upconnect_) > 0
        f_ex = _f_up ^ _stack.downconnect_cnt > 0

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

    import cv2

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


def draw_sstack_(sstack_):

    ''' visualize the stacks, sstacks and flipped sstacks'''

    from matplotlib import pyplot as plt

    # initialization
    x0_all, xn_all, y0_all, yn_all = [],[],[],[]
    x_offset_all = []
    y_offset_all = []
    img_ori_all = []
    img_rot_all = []
    box_ori_all = []
    box_rot_all = []

    for sstack in sstack_:

        # get local x0,xn, y0, yn
        x0_, xn_, y0_, yn_ = [],[],[],[]
        for stack in sstack.Py_:
            x0_.append(min([Py.x0 for Py in stack.Py_]))
            xn_.append(max([Py.x0 + Py.L for Py in stack.Py_]))
            y0_.append(stack.y0)
            yn_.append(stack.y0 + stack.Ly)

        x0 = min(x0_)
        xn = max(xn_)
        y0 = min(y0_)
        yn = max(yn_)

        x_mid = round(x0+ ((xn-x0)/2))  # mid local x (common point for the rotated and non rotated derts)
        y_mid = round(y0+ ((yn-y0)/2))  # mid local y (common point for the rotated and non rotated derts)

        # get non-rotated derts from their stacks
        img = np.zeros((yn - y0, xn - x0))
        stack_index = 1
        for stack in sstack.Py_:
            for y, P in enumerate(stack.Py_):
                for x, dert in enumerate(P.dert_):
                    img[y + (stack.y0 - y0), x + (P.x0 - x0)] = stack_index
            stack_index += 1  # for next stack

        # accumulation of params
        x0_all.append(x0)
        xn_all.append(xn)
        y0_all.append(y0)
        yn_all.append(yn)
        box_ori_all.append([y0,yn,x0,xn])
        img_ori_all.append(img)

        if sstack.stack_: # for flipped sstack and their stacks

            # get rotated x0,xn, y0, yn
            x0_rot, xn_rot, y0_rot, yn_rot = [],[],[],[]
            for stack in sstack.stack_:
                x0_rot.append(min([Py.x0 for Py in stack.Py_]))
                xn_rot.append(max([Py.x0 + Py.L for Py in stack.Py_]))
                y0_rot.append(stack.y0)
                yn_rot.append(stack.y0 + stack.Ly)

            x0_rot = min(x0_rot)
            xn_rot = max(xn_rot)
            y0_rot = min(y0_rot)
            yn_rot = max(yn_rot)

            # get rotated derts from their stacks
            img_rot = np.zeros((yn_rot - y0_rot, xn_rot - x0_rot))
            stack_index = 1
            for stack in sstack.stack_:
                for y, P in enumerate(stack.Py_):
                    for x, dert in enumerate(P.dert_):
                        img_rot[y + (stack.y0 - y0_rot), x + (P.x0 - x0_rot)] = stack_index
                stack_index += 1  # for next stack
            # get rotated x0, xn, y0, yn
            # https://math.stackexchange.com/questions/270194/how-to-find-the-vertices-angle-after-rotation
            # (p,q) =  origin of rotation
            # θ = angle of rotation in radian (-90 degree in our case = -1.5708)
            # x′=(x−p)cos(θ)−(y−q)sin(θ)+p
            # y′=(x−p)sin(θ)+(y−q)cos(θ)+q
            x0_rot_common = int(round((x0-x_mid)* np.cos(-1.5708) - (y0-y_mid)*np.sin(-1.5708)+x_mid))
            xn_rot_common = int(round((xn-x_mid)* np.cos(-1.5708) - (yn-y_mid)*np.sin(-1.5708)+x_mid))
            y0_rot_common = int(round((x0-x_mid)* np.sin(-1.5708) + (y0-y_mid)*np.cos(-1.5708)+y_mid))
            yn_rot_common = int(round((xn-x_mid)* np.sin(-1.5708) + (yn-y_mid)*np.cos(-1.5708)+y_mid))


            # if the rotated location ending point is before the starting point (negative value), we need to invert their location
            if y0_rot_common > yn_rot_common:
                y0_rot_common, yn_rot_common = yn_rot_common+1, y0_rot_common+1  # + 1 because starting index is inclusive, while ending is not
            if x0_rot_common > xn_rot_common:
                x0_rot_common, xn_rot_common = xn_rot_common+1, x0_rot_common+1 # + 1 because starting index is inclusive, while ending is not

            # if the rotated location is beyond the current mask boundary (negative index), we extend the boundary by adding offset
            y_offset = 0
            if y0_rot_common <0:
                y_offset = -y0_rot_common
            x_offset = 0
            if x0_rot_common <0:
                x_offset = -x0_rot_common

            # bounding box of derts from original derts and rotated derts
            x0_common = min([x0,x0_rot_common])
            xn_common = max([xn,xn_rot_common])
            y0_common = min([y0,y0_rot_common])
            yn_common = max([yn,yn_rot_common])

            # accumulation of params
            x0_all.append(x0_common+x_offset)
            xn_all.append(xn_common+x_offset)
            y0_all.append(y0_common+y_offset)
            yn_all.append(yn_common+y_offset)
            x_offset_all.append(x_offset)
            y_offset_all.append(y_offset)
            img_rot_all.append(img_rot)
            box_rot_all.append([y0_rot_common,yn_rot_common,x0_rot_common,xn_rot_common])

        else: # no rotated stacks, set value as 0 or empty
            x_offset_all.append(0)
            y_offset_all.append(0)
            img_rot_all.append([])
            box_rot_all.append([])

    # initialize image from max yn and max xn
    img_stacks = np.zeros((np.max(yn_all), np.max(xn_all)))
    img_common = np.zeros((np.max(yn_all), np.max(xn_all)))
    img_common_rot = np.zeros((np.max(yn_all), np.max(xn_all)))

    for i,(img_ori,img_rot,box_ori,box_rot,x_offset,y_offset) in enumerate(zip(img_ori_all,img_rot_all,box_ori_all, box_rot_all, x_offset_all, y_offset_all)):

        y0,yn,x0,xn = box_ori
        # set index for stacks
        img_stacks[y0+y_offset:yn+y_offset,x0+x_offset:xn+x_offset] += img_ori

        # generate common index for sstack's derts and rotated sstack's derts
        img_ori[img_ori>0] = i+1
        img_common[y0+y_offset:yn+y_offset,x0+x_offset:xn+x_offset] += img_ori # assign common index for non-rotated sstacks

        if len(img_rot)>0:
            y0_rot,yn_rot,x0_rot,xn_rot = box_rot
            # generate common index for sstack's derts and rotated sstack's derts
            img_rot[img_rot>0] = i+1
            img_common_rot[y0_rot+y_offset:yn_rot+y_offset,x0_rot+x_offset:xn_rot+x_offset] += img_rot # assign common index for rotated sstacks
        else: # no rotated sstacks, # assign common index
            img_common_rot[y0+y_offset:yn+y_offset,x0+x_offset:xn+x_offset] += img_ori

        # set index of overlapping area = 255. There might be some overlapping area when some sstacks are rotated, while some are not.
        img_common_rot[img_common_rot>i+1] = 255

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
    colour_white = ([255, 255, 255])  # white

    # initialization
    img_colour_stacks = np.zeros((np.max(yn_all), np.max(xn_all), 3)).astype('uint8')
    img_colour_sstacks = np.zeros((np.max(yn_all), np.max(xn_all), 3)).astype('uint8')
    img_colour_sstacks_rot = np.zeros((np.max(yn_all), np.max(xn_all), 3)).astype('uint8')

    # draw stacks' image
    total_stacks = int(np.max(img_stacks))
    for i in range(1, total_stacks+1):
        colour_index = i % 9
        img_colour_stacks[np.where(img_stacks == i)] = colour_list[colour_index]

    # draw sstack and the rotated sstack/s image
    total_sstacks = int(np.max(img_common))
    for i in range(1, total_sstacks+1):
        colour_index = i % 9
        img_colour_sstacks[np.where(img_common == i)] = colour_list[colour_index]
        img_colour_sstacks_rot[np.where(img_common_rot == i)] = colour_list[colour_index]

    # set overlapping area = white
    img_colour_sstacks_rot[np.where(img_common_rot >= 255)] = colour_white

    # draw image to figure and save it to disk
    plt.figure(1)
    plt.subplot(1,3,1)
    plt.imshow(np.uint8(img_colour_stacks))
    plt.title('stacks')
    plt.subplot(1,3,2)
    plt.imshow(img_colour_sstacks)
    plt.title('sstacks')
    plt.subplot(1,3,3)
    plt.imshow(img_colour_sstacks_rot)
    plt.title('rotated sstacks')
    plt.savefig('./images/slice_blob/sstack_'+str(id(sstack_))+'.png')
    plt.close()