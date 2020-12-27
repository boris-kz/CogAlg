from collections import deque
import sys
import numpy as np
from class_cluster import ClusterStructure, NoneType

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback, not needed here
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
    term_stack_ = list  # also replaces f_flip: vertical if empty, else horizontal
    downconnect_cnt = int
    upconnect_ = list
    stack_PP = object

# Functions:

def slice_blob(dert__, mask__, verbose=False):

    row_stack_ = deque()  # higher-row vertical stacks of Ps
    term_stack_ = []  # 2D array of terminated stacks
    height, width = dert__[0].shape
    if verbose: print("Converting to image...")

    for y, dert_ in enumerate(zip(*dert__)):  # first and last row are discarded?
        if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()

        P_ = form_P_(list(zip(*dert_)), mask__[y])  # horizontal clustering
        P_ = scan_P_(P_, term_stack_, row_stack_)  # vertical clustering, adds P upconnect_ and stack downconnect_cnt
        row_stack_ = form_stack_(P_, term_stack_, y)  # initialize and accumulate stacks with Ps, including the whole 1st row_stack_

    term_stack_.extend(row_stack_)  # dert__ ends, all last-row stacks are moved to term_stack_

    # below is for debugging, to check whether we miss out any stack from the mask
    img_colour = draw_stacks(term_stack_)  # visualization
    from matplotlib import pyplot as plt
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img_colour)
    plt.subplot(1,2,2)
    plt.imshow(mask__*255)

    sstack_ = form_sstack_(term_stack_)  # cluster stacks into horizontally-oriented super-stacks
#   flip_sstack_(row_stack_)  # vertical-first re-scanning of selected sstacks

    for sstack in sstack_:  # convert selected stacks into gstacks
        form_gPPy_(sstack.Py_)  # sstack.Py_ = stack_

    return sstack_  # partially rotated and gP-extended term_stack_

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


def scan_P_(P_, term_stack_, row_stack_):  # merge P into higher-row stack of Ps which have same sign and overlap by x_coordinate
    '''
    Each P in P_ scans higher-row _Ps (in stack_) left-to-right, testing for x-overlaps between Ps and same-sign _Ps.
    Overlap is represented as upconnect in P and is added to downconnect_cnt in _P. Scan continues until P.x0 >= _P.xn:
    no x-overlap between P and next _P. Then P is packed into sole upconnect stack or initializes a new stack.
    After such test, loaded _P is also tested for x-overlap to the next P.
    If negative, a stack with loaded _P is removed from stack_ (buffer of higher-row stacks) and tested for downconnect_cnt==0.
    If so: no lower-row connections, the stack is packed into connected blobs (referred by its upconnect_),
    else the stack is recycled into next_stack_, for next-row run of scan_P_.
    It's a form of breadth-first flood fill, with connects as vertices per stack of Ps: a node in connectivity graph.
    '''
    next_P_ = deque()  # to recycle (P, upconnect_)s that finished scanning _P, will be converted into next row_stack_

    if P_ and row_stack_:  # if both input row and higher row are not empty

        P = P_.popleft()  # load left-most (lowest-x) input-row P
        stack = row_stack_.popleft()  # higher-row stacks
        _P = stack.Py_[-1]  # last element of each stack is higher-row P
        upconnect_ = []  # list of same-sign x-overlapping _Ps per P

        while True:  # while both P_ and stack_ are not empty
            x0 = P.x0         # first x in P
            xn = x0 + P.L     # first x in next P
            _x0 = _P.x0       # first x in _P
            _xn = _x0 + _P.L  # first x in next _P

            if _x0 - 1 < xn and _xn + 1 > x0:  # x overlap between P and _P in 8 directions: +ma blob is less selective
                stack.downconnect_cnt += 1
                upconnect_.append(stack)  # P-connected higher-row stacks

            if xn <= _xn:  # _P overlaps next P in P_ in any of 8 directions, if (xn < _xn) for 4 directions
                next_P_.append((P, upconnect_))  # recycle _P for the next run of scan_P_
                if P_:
                    upconnect_ = []  # reset for next P, if any
                    P = P_.popleft()  # load next P
                else:  # terminate loop
                    break
            else:  # no next-P overlap
                if stack.downconnect_cnt != 1:
                    term_stack_.append(stack)  # terminated stack is buffered

                if row_stack_:  # load stack with next _P
                    stack = row_stack_.popleft()
                    _P = stack.Py_[-1]
                else:  # no stacks left: terminate loop
                    next_P_.append((P, upconnect_))
                    break

    while P_:  # terminate Ps that continue at row's end
        next_P_.append((P_.popleft(), []))  # no upconnect

    term_stack_ += row_stack_  # after last P in P_, all stacks should have downconnect_cnt=0

    return next_P_  # each element is P + upconnect_ refs


def form_stack_(P_, term_stack_, y):

    # Convert or merge every P into higher-row stack of Ps, terminate stack if downconnects != 1
    next_row_stack_ = deque()  # converted to row_stack_ in the next run of scan_P_

    while P_:
        P, upconnect_ = P_.popleft()
        I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, L, x0, s, dert_, _, _, _ = P.unpack()
        if not upconnect_:
            # initialize new stack for each input-row P that has no connections in higher row, as in the whole top row:
            stack = CStack(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=L, Ly=1,
                           y0=y, Py_=[P], downconnect_cnt=0, upconnect_=upconnect_, sign=s, fPP=0)

            # line 261 and 276 are temporary solution, still need to find a better way to add upconnects to term_stack_
            # add stack's upconnects into term_stack_
            [term_stack_.append(upconnect) for upconnect in upconnect_ if upconnect not in term_stack_ ]

        else:
            if len(upconnect_) == 1 and upconnect_[0].downconnect_cnt == 1:
                # P has one upconnect and that upconnect has one downconnect=P: merge P into upconnect' stack:
                stack = upconnect_[0]
                stack.accumulate(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=L, Ly=1)
                stack.Py_.append(P)   # Py_: vertical buffer of Ps
                stack.downconnect_cnt = 0  # reset downconnect_cnt

            else:  # P has >1 upconnects, or 1 upconnect that has >1 downconnects:  initialize stack with P:
                stack = CStack(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, A=L, Ly=1,
                               y0=y, Py_=[P], downconnect_cnt=0, upconnect_=upconnect_, sign=s, fPP=0)

                # add stack's upconnects into term_stack_
                [term_stack_.append(upconnect) for upconnect in upconnect_ if upconnect not in term_stack_ ]

        next_row_stack_.append(stack)

    return next_row_stack_  # input for the next line of scan_P_

'''
def terminate_stack_(row_stack_, term_stack_):
    # for each stack in row_stack_: if complete downconnect_cnt != 1 (or = 0?): include stack in term_stack__:
    for stack in row_stack_:
        if stack.downconnect_cnt != 1:  # separate trace-by-connect for stacks with downconnect_cnt > 1, out of order?
            term_stack_.append(stack)
#    if term_stack_:  # add a not-empty x-ordered row of terminated stacks to 2D term_stack__:
#        term_stack_.append(term_stack_)
'''

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


def draw_stacks(stack_):
    '''
    draw stacks, for debug purpose
    '''
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
    total_stacks = stack_index

    for i in range(1, total_stacks + 1):
        colour_index = i % 10
        img_colour[np.where(img == i)] = colour_list[colour_index]
    #    cv2.imwrite('./images/stacks/stacks_blob_' + str(sliced_blob.id) + '_colour.bmp', img_colour)

    return img_colour


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

            sstack_mask__ = np.ones((yn - y0, xn - x0)).astype(bool)
            # unmask sstack:
            for stack in sstack.Py_:
                for y, P in enumerate(stack.Py_):
                    sstack_mask__[y, P.x0: (P.x0 + P.L)] = False  # unmask P

            # extract and flip sstack's dert__
            sstack_dert__ = tuple([np.rot90(param_dert__[y0:yn, x0:xn]) for param_dert__ in sliced_blob.dert__])
            sstack_mask__ = np.rot90(sstack_mask__)  # flip mask

            row_stack_ = deque()
            for y, dert_ in enumerate(zip(*sstack_dert__)):  # same operations as in root sliced_blob(), but on sstack

                P_ = form_P_(list(zip(*dert_)), sstack_mask__[y])  # horizontal clustering
                P_ = scan_P_(P_, row_stack_)  # vertical clustering, adds P upconnects and _P downconnect_cnt
                next_row_stack_ = form_stack_(P_, y)  # initialize and accumulate stacks with Ps

                for stack in row_stack_:  # terminate stacks:
                    if stack.downconnect_cnt != 1:  # separate trace-by-connect for stacks with downconnect_cnt > 1, out of order?
                        sstack.term_stack_.append(stack)
                row_stack_ = next_row_stack_  # row_stack_ + initialized stacks - terminated stacks

            sstack.term_stack_ += row_stack_   # dert__ ends, last-row stacks are moved to sstack.term_stack__