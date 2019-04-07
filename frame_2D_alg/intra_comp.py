import numpy as np
from math import hypot
from math import atan2
from collections import deque, namedtuple

from frame_2D_alg.filters import get_filters
get_filters(globals())  # import all filters at once

nt_blob = namedtuple('blob', 'typ sign Y X Ly L Derts seg_ sub_blob_ layers_f sub_Derts map box add_dert rng ncomp')

# ************ FUNCTIONS ************************************************************************************************
# -form_sub_blob()
# -unfold()
# -find_P()
# -comp_angle()
# -comp_gradiente()
# -comp_range()
# -form_P_()
# -scan_P_()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************

''' this module is under revision '''


def form_sub_blob(dert_, root_blob):  # redefine blob as branch-specific master blob: local equivalent of frame

    seg_ = deque()

    while dert_:
        P_ = form_P_(dert_)                     # horizontal clustering
        P_ = scan_P_(P_, seg_, root_blob)       # vertical clustering
        seg_ = form_seg_(P_, blob, root_blob)   # vertical clustering
    while seg_: form_blob(seg_.popleft(), root_blob)  # terminate last running segments

    # ---------- add_sub_blob() end -----------------------------------------------------------------------------------------

def unfold_blob(blob, typ, shift_):     # perform compare while unfolding. shift_ is per direction
    # Process every segment bottom-up (spatially)
    # indicate comparands
    # if a comparand goes beyond the segment vertically, look into it's forks (recursive)
    # by convention, comparand's vertical coordinate is always smaller than that of the main comparand (yd > 0)
    # 4 typ of operations: hypot_g, comp_angle, comp_gradient, comp_range

    new_dert_ = []  # buffer for results

    if typ == 0:            # unfold for hypot_g
        for seg in blob.seg_:
            for y, P in enumerate(seg[2], start=seg[0]):            # y starts from y0
                for x, (i, dy, dx, g) in enumerate(P[-1], start=P[1]):      # x starts from x0

                    g = hypot(dy, dx) - ave
                    new_dert_.append((y, x, i, dy, dx, g))

    elif typ == 1:           # unfold for comp_angle

        a_ = []
        for seg in blob.seg_:
            for y, P in enumerate(seg[2], start=seg[0]):            # y starts from y0
                for x, (i, dy, dx, g) in enumerate(P[-1], start=P[1]):      # x starts from x0
                    a = int(atan2(dy, dx) * 128 / __math__.pi) + 128
                    a_.append((y, x, a))

        new_dert_ = comp_angle_(a_)

    else:                   # unfold for comp_gradient or comp_range
        if typ == 3:            # compute coefs_ for comp_range (if typ == 2)

            coefs_ = []

            for yd, xd in shift_:  # compute a pair of coefficient for each comp direction

                denominator = hypot(yd, xd)     # common denominator for the pair of coefficients

                coefs_.append((yd / denominator, xd / denominator))

        yd_, xd_ = zip(*shift_)    # here, we assume that each shift is in the form of (yd, xd) and yd > 0
        yd_ = [-yd for yd in dy_]   # comps are upward

        for seg in blob.seg_:       # iterate through blob's segments

            for iP, P in seg[2]:        # vertical search

                y = P[-1][0][0]         # y of first dert
                _P__ = []               # list of list of potential comparands' P (there might be more than 1 in forks)
                for yd in yd_:          # iterate through the list of comparands' yds
                    _P_ = []            # keep a list of potential comparands' P (there might be more than 1 in forks)
                    _iP = iP + yd       # index in Py_ based on vertical coordinate
                    find_P(seg, _P_, _iP)  # find all potential comparands' P

                for dert in P[2]:       # horizontal search

                    x = dert[0]
                    for xd, yd, _P_ in zip(xd_, yd_, _P__):  # iterate through potential comparands
                        _x = x + xd                 # horizontal coordinate

                        stop = False                # stop flag
                        _dert = None                # _dert initialization
                        for _P in _P_:              # iterate through potential comparands' Ps
                            if stop:
                                break

                            for _dert in _P[2]:     # iterate through potential comparands' Ps' derts
                                if _x == _dert[1]:  # if dert's coordinates are identical with target coordinates (vertical coordinates are already matched)
                                    _y = y + yd     # compute actual vertical coordinate
                                    stop = True     # stop
                                    break

                        if stop == True:            # if a comparand with the right coordinate is found:

                            if typ == 2:            # comp_gradient, compare g (dert[4])
                                vert = yd == -1     # indicate comp direction
                                dy, dx = comp_gradient(dert, _dert, vert)

                            elif typ == 3:          # comp_range, compare i (dert[2])
                                dy, dx = comp_range(dert, _dert, coefs_)

                            new_dert_.append((y, x, dy, dx))
                            new_dert_.append((_y, _x, dy, dx))

        # combine raw derts back into derts (with ncomp) and compute g:


        new_dert__.sort(key=lambda dert: dert[1])    # sorted by x coordinates
        new_dert__.sort(key=lambda dert: dert[0])    # sorted by y coordinates

        dert_buffer_ = new_dert_

        new_dert_ = []

        i = 0
        max_i = len(dert_buffer_) - 1
        while i < max_i:   # i goes through comp results with identical y, x (summing dy, dx along)

            y, x, dy, dx, g = dert_buffer_[i]    # initialize dert
            ncomp = 1                               # number of comps with current dert (at coordinates y, x)
            while i < max_i and y == dert_buffer_[i + 1][0] and x == dert_buffer_[i + 1][1]:    # y, x axes' coordinates are identical
                i += 1  # increment i

                # merge derts:
                ncomp += 1                  # +1 number of comps
                dy += new_dert_[i][3]       # sum dy
                dx += new_dert_[i][4]       # sum dx

            g = hypot(dy, dx) - ncomp * ave # compute g with complete dy, dx

            new_dert_.append((y, x, ncomp, dy, dx, g))     # buffer into new_new_dert_ for folding/clustering functions

            i += 1

    form_sub_blob(new_dert_, blob)

    # ---------- unfold_blob() end ------------------------------------------------------------------------------------------

def find_P(seg, _P_, _iP):     # used in unfold() to find all potential comparands' Ps (_P) with given vertical coordinate (P index in Py_)

    if _iP > 0:                 # if P's coordinate is within segment
        _P_.append(seg[2][_iP]) # buffer P with given index
    else:                       # if P's is beyond segment
        for fork in seg[4]:         # look for P in segment's forks. Stop if seg[4] == []  (no fork)
            find_P(fork, _P_, fork[1][0] - _iP)    # fork[1][0] - _iP: Ly - _iP (supposedly index of _P)

    # ---------- find_P() end -----------------------------------------------------------------------------------------------

def comp_angle_(a_):    # compare angles

    dert_ = [[0, 0, 0] for _ in range(len(a_))]   # ncomp, dy, dx

    dert_.sort(key=lambda dert: dert[1])        # sort by x
    dert_.sort(key=lambda dert: dert[0])        # sort by y, derts are now in row-major order

    # initialize indices:
    i = 0       # index of main comparand
    _iv = 0     # index of vertical comparand
    _ih = 0     # index of horizontal comparand

    while i < len(a_) and (_iv < len(a_) or _ih < len(a_)):  # this loop is per i

        y, x, a = a_[i]

        # horizontal comp:
        xh = x + 1

        while _ih < len(dert_) and dert_[_ih][0] < y:                               # search for right y coordinate
            _ih += 1

        while _ih < len(dert_) and dert_[_ih][0] == y and dert_[_ih][1] < xh:       # search for right x coordinate
            _ih += 1

        if _ih < len(dert_) and dert_[_ih][0] == yx and dert_[_ih][1] == xh:        # compare if coordinates matches

            _a = a_[_ih][2]

            dx = _a - a         # horizontal comp

            dert_[i][0] += 1    # bilateral accumulation
            dert_[i][2] += dx

            dert_[_ih][0] += 1  # bilateral accumulation
            dert_[_ih][2] += dx

        # vertical comp:

        yv = y + 1          # y coordinate of vertical comparand

        while _iv < len(dert_) and dert_[_iv][0] < y:                               # search for right y coordinate
            _iv += 1

        while _iv < len(dert_) and dert_[_iv][0] == yv and dert_[_iv][1] < x:       # search for right x coordinate
            _iv += 1

        if _iv < len(dert_) and dert_[_iv][0] == yx and dert_[_iv][1] == x:         # compare if coordinates matches

            _a = a_[_iv][2]

            dy = _a - a         # vertical comp

            dert_[i][0] += 1    # bilateral accumulation
            dert_[i][1] += dy

            dert_[_iv][0] += 1  # bilateral accumulation
            dert_[_iv][1] += dy

        i += 1

    full_dert_ = [(y, x, a, ncomp, dy, dx, hypot(dy, dx) // ncomp - ave) for (y, x, a), (ncomp, dy, dx) in zip(a_, dert_)]

    return full_dert_
    # ---------- comp_angle_() end ------------------------------------------------------------------------------------------

def comp_gradient(dert, _dert, vert=0):    # compare g of derts

    y, x, i, (ncomp, dy, dx), g = dert[:5]          # first 5 derts, exclude angle if there's one
    _y, _x, _i, (_ncomp, _dy, _dx), _g = _dert[:5]  # first 5 derts, exclude angle if there's one

    dg = _g - g         # compare g

    if vert:            # if vertical comp
        dgy = -dg       # vertical comp is upward so day is reversed in sign
        dgx = 0
    else:               # if horizontal comp
        dgy = 0
        dgx = dg

    return dgy, dgx

    # ---------- comp_gradient() end ----------------------------------------------------------------------------------------

def comp_range(dert, _dert, coefs):     # compare i of derts over indicated distance
    # comparison may include more than 2 directions: vertical, horizontal. So coefficients are needed to decompose di
    # into diy, dix

    y, x, i, (ncomp, dy, dx), g = dert[:5]          # first 5 derts, exclude angle if there's one
    _y, _x, _i, (_ncomp, _dy, _dx), _g = _dert[:5]  # first 5 derts, exclude angle if there's one

    di = _i - i                 # compare i

    diy = int(di * coefs[0])    # vertical coefficient
    dix = int(di * coefs[1])    # horizontal coefficient

    return diy, dix

    # ---------- comp_range() end -------------------------------------------------------------------------------------------

def form_P_(y, master_blob):  # cluster and sum horizontally consecutive pixels and their derivatives into Ps

    rng = master_blob.rng
    dert__ = master_blob.new_dert__[0]
    P_ = deque()  # initialize output
    dert_ = dert__[y, :, :]  # row of pixels + derivatives
    P_map_ = ~dert_.mask[:, 3]  # dert_.mask?
    x_stop = len(dert_) - rng
    x = rng  # first and last rng columns are discarded

    while x < x_stop:
        while x < x_stop and not P_map_[x]:
            x += 1
        if x < x_stop and P_map_[x]:
            s = dert_[x][-1] > 0  # s = g > 0
            P = [s, [0, 0, 0] + [0] * len(dert_[0]), []]
            while x < x_stop and P_map_[x] and s == P[0]:
                dert = dert_[x, :]
                # accumulate P params:
                P[1] = [par1 + par2 for par1, par2 in zip(P[1], [1, y, x] + list(dert))]
                P[2].append((y, x) + tuple(dert))
                x += 1
                s = dert_[x][-1] > 0  # s = g > 0

            if P[1][0]:  # if L > 0
                P_.append(P)  # P is packed into P_
    return P_

    # ---------- form_P_() end ------------------------------------------------------------------------------------------


def scan_P_(P_, seg_, master_blob,
            rng_inc=False):  # this function detects connections (forks) between Ps and _Ps, to form blob segments
    new_P_ = deque()

    if P_ and seg_:  # if both are not empty
        P = P_.popleft()  # input-line Ps
        seg = seg_.popleft()  # higher-line segments,
        _P = seg[2][-1]  # last element of each segment is higher-line P
        stop = False
        fork_ = []
        while not stop:
            x0 = P[2][0][1]  # first x in P
            xn = P[2][-1][1]  # last x in P
            _x0 = _P[2][0][1]  # first x in _P
            _xn = _P[2][-1][1]  # last x in _P

            if P[0] == _P[0] and _x0 <= xn and x0 <= _xn:  # test for sign match and x overlap
                seg[3] += 1
                fork_.append(seg)  # P-connected segments are buffered into fork_

            if xn < _xn:  # _P overlaps next P in P_
                new_P_.append((P, fork_))
                fork_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:
                    if seg[3] != 1:  # if roots != 1: terminate loop
                        form_blob(seg, master_blob, rng_inc)
                    stop = True
            else:
                if seg[3] != 1:  # if roots != 1
                    form_blob(seg, master_blob, rng_inc)
                if seg_:
                    seg = seg_.popleft()  # load next seg and _P
                    _P = seg[2][-1]
                else:
                    new_P_.append((P, fork_))
                    stop = True  # terminate loop

    while P_:  # handle Ps and segs that don't terminate at line end
        new_P_.append((P_.popleft(), []))  # no fork
    while seg_:
        form_blob(seg_.popleft(), master_blob, rng_inc)  # roots always == 0
    return new_P_

    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_seg_(P_, master_blob, rng_inc=False):  # Convert or merge every P into segment. Merge blobs
    new_seg_ = deque()
    while P_:
        P, fork_ = P_.popleft()
        s, params, dert_ = P

        if not fork_:  # seg is initialized with initialized blob
            blob = [s, [0] * (len(params) + 1), [], 1]  # s, params, seg_, open_segments
            seg = [s, [1] + params, [P], 0, fork_, blob]  # s, params. P_, roots, fork_, blob
            blob[2].append(seg)
        else:
            if len(fork_) == 1 and fork_[0][3] == 1:  # P has one fork and that fork has one root
                seg = fork_[0]
                # P is merged into segment:
                seg[1] = [par1 + par2 for par1, par2 in
                          zip([1] + params, seg[1])]  # sum all params of P into seg, in addition to +1 in Ly
                seg[2].append(P)  # Py_: vertical buffer of Ps merged into seg
                seg[3] = 0  # reset roots

            else:  # if > 1 forks, or 1 fork that has > 1 roots:
                blob = fork_[0][5]
                seg = [s, [1] + params, [P], 0, fork_, blob]  # seg is initialized with fork blob
                blob[2].append(seg)  # segment is buffered into blob

                if len(fork_) > 1:  # merge blobs of all forks
                    if fork_[0][3] == 1:  # if roots == 1: fork hasn't been terminated
                        form_blob(fork_[0], master_blob, rng_inc)  # merge seg of 1st fork into its blob

                    for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                        if fork[3] == 1:
                            form_blob(fork, master_blob, rng_inc)

                        if not fork[5] is blob:
                            params, e_, open_segments = fork[5][1:]  # merged blob, omit sign
                            blob[1] = [par1 + par2 for par1, par2 in
                                       zip(params, blob[1])]  # sum same-type params of merging blobs
                            blob[3] += open_segments
                            for e in e_:
                                if not e is fork:
                                    e[5] = blob  # blobs in other forks are references to blob in the first fork
                                    blob[2].append(e)  # buffer of merged root segments
                            fork[5] = blob
                            blob[2].append(fork)
                        blob[3] -= 1  # open_segments -= 1: shared seg is eliminated

        new_seg_.append(seg)
    return new_seg_

    # ---------- form_seg_() end --------------------------------------------------------------------------------------------


def form_blob(term_seg, master_blob,
              rng_inc=False):  # terminated segment is merged into continued or initialized blob (all connected segments)

    params, P_, roots, fork_, blob = term_seg[1:]

    blob[1] = [par1 + par2 for par1, par2 in zip(params, blob[1])]
    blob[3] += roots - 1  # number of open segments

    if not blob[3]:  # if open_segments == 0: blob is terminated and packed in master_blob
        blob.pop()  # remove open_segments
        s, blob_params, e_ = blob
        y0 = 9999999
        x0 = 9999999
        yn = 0
        xn = 0
        for seg in e_:
            seg.pop()  # remove references of blob
            for P in seg[2]:
                y0 = min(y0, P[2][0][0])
                x0 = min(x0, P[2][0][1])
                yn = max(yn, P[2][0][0] + 1)
                xn = max(xn, P[2][-1][1] + 1)

        map = np.zeros((yn - y0, xn - x0), dtype=bool)
        for seg in e_:
            for P in seg[2]:
                for y, x, i, dy, dx, g in P[2]:
                    map[y - y0, x - x0] = True
        map = map[y0:yn, x0:xn]

        master_blob.params[-4:] = [par1 + par2 for par1, par2 in zip(master_blob.params[-4:], blob_params[-4:])]

        master_blob.sub_blob_[-1].append(nt_blob(typ=0, sign=s, Ly=Ly, L=L,
                                                 Derts=[(I, Dy, Dx, G)],  # not selective to +sub blobs as sub_Derts
                                                 layerf=0,
                                                 # flag: derts_ = [(sub_Derts, derts_)], appended per intra_blob eval_layer
                                                 sub_Derts=[],
                                                 # sub_blob_ Derts += [(Ly, L, I, Dy, Dx, G)] if len(sub_blob_) > min
                                                 lay_Derts=[],
                                                 # layer_ Derts += [(Ly, L, I, Dy, Dx, G)] if len(layer_) > min
                                                 box=(y0, yn, x0, xn),
                                                 map=map,
                                                 add_dert=None,
                                                 rng=(master_blob.rng + 1) if rng_inc else 1,  # increase or reset rng
                                                 ncomp=(master_blob.ncomp + master_blob.rng + 1) if rng_inc else 1))
        '''
        replace:

        master_blob.sub_blobs[0][:] += Derts[:]  # accumulate each Derts param

        master_blob.sub_blobs[1][-1].append(nt_blob(
                                typ=typ,
                                sign=s,
                                Y=Y,
                                X=X,
                                Derts = [(Ly, L, I, Dy, Dx, G)],
                                derts_ = [dert__],  # extend all elements
                                sub_blobs = [],   # replace derts_, replaced by sub_layers
                                sub_layers = [],  # type ignored if == 1
                                box=(y0, yn, x0, xn),
                                map=map,
                                add_dert=None,
                                rng=1, ncomp=1
                                ))'''
    # ---------- form_blob() end -------------------------------------------------------------------------------------