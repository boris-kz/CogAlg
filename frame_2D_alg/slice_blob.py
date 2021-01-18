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

from collections import deque
import sys
import numpy as np
from class_cluster import ClusterStructure, NoneType
from slice_utils import *

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
    A = int  # stack area
    Ly = int
    # box:
    x0 = int
    xn = int
    y0 = int
    # stack:
    Py_ = list  # Py_, dPPy_, or stack_
    sign = NoneType
    f_gstack = NoneType  # gPPy_ if 1, else Py_
    f_stackPP = NoneType  # for comp_slice: PPy_ if 1, else gPPy_ or Py_
    downconnect_cnt = int
    upconnect_ = list
    stack_PP = object  # replaces f_stackPP?
    stack_ = list  # ultimately all stacks, also replaces fflip: vertical if empty, else horizontal
    f_checked = int  # flag: stack has gone through form_sstack_recursive as upconnect

# Functions:

def slice_blob(blob, verbose=False):

    flip_eval(blob)
    dert__ = blob.dert__
    mask__ = blob.mask__
    row_stack_ = []  # higher-row vertical stacks of Ps
    stack_ = []  # all terminated stacks
    height, width = dert__[0].shape
    if verbose: print("Converting to image...")

    for y, dert_ in enumerate(zip(*dert__)):  # first and last row are discarded?
        if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()

        P_ = form_P_(list(zip(*dert_)), mask__[y])  # horizontal clustering
        P_ = scan_P_(P_, row_stack_)  # vertical clustering, adds P upconnect_ and stack downconnect_cnt
        next_row_stack_ = form_stack_(P_, y)  # initialize and accumulate stacks with Ps, including the whole 1st row_stack_

        [stack_.append(stack) for stack in row_stack_ if not stack in next_row_stack_]  # buffer stacks that won't be used in next row
        row_stack_ = next_row_stack_  # stacks initialized or accumulated in form_stack_

    stack_ += row_stack_  # dert__ ends, all last-row stacks have no downconnects
    if verbose: check_stacks_presence(stack_, mask__, f_plot=0)  # visualize stack_ and mask__ to see if they cover the same area

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
                           x0=x0, xn=x0+L, y0=y, Py_=[P], downconnect_cnt=0, upconnect_=upconnect_, sign=s, fPP=0)
        else:
            if len(upconnect_) == 1 and upconnect_[0].downconnect_cnt == 1:
                # P has one upconnect and that upconnect has one downconnect=P: merge P into upconnect' stack:
                stack = upconnect_[0]
                stack.accumulate(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=L, Ly=1)
                stack.x0 = min(stack.x0, x0)
                stack.xn = max(stack.xn, x0 + L)
                stack.Py_.append(P)   # Py_: vertical buffer of Ps
                stack.downconnect_cnt = 0  # reset downconnect_cnt to that of last P

            else:  # P has >1 upconnects, or 1 upconnect that has >1 downconnects: initialize stack with P:
                stack = CStack(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=L, Ly=1,
                               x0=x0, xn=x0+L, y0=y, Py_=[P], downconnect_cnt=0, upconnect_=upconnect_, sign=s, fPP=0)

        next_row_stack_.append(stack)

    return next_row_stack_  # input for the next line of scan_P_


def flip_eval(blob):

    horizontal_bias = ( blob.box[3] - blob.box[2]) / (blob.box[1] - blob.box[0]) \
                      * (abs(blob.Dy) / abs(blob.Dx))
        # L_bias (Lx / Ly) * G_bias (Gy / Gx), blob.box = [y0,yn,x0,xn], ddirection: , preferential comp over low G

    if horizontal_bias > 1 and (blob.G * blob.Ma * horizontal_bias > flip_ave / 10):

        blob.fflip = 1  # rotate 90 degrees for scanning in vertical direction
        blob.dert__ = tuple([np.rot90(dert) for dert in blob.dert__])
        blob.mask__ = np.rot90(blob.mask__)


def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})