import numpy as np
from math import hypot
from math import atan2
from collections import deque, namedtuple
from filters import get_filters
get_filters(globals())  # import all filters at once

nt_blob = namedtuple('blob', 'typ sign Ly L Derts seg_ sub_blob_ layers_f sub_Derts root_blob map box rng')

# ************ FUNCTIONS ************************************************************************************************
# -unfold_blob()
# -find_P()
# -form_sub_blobs()
# -comp_angle()
# -comp_gradient()
# -comp_range()
# -form_P_()
# -scan_P_()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************

''' this module is under revision '''

def unfold_blob(typ, blob, comp_typ, offset_, rdn):  # add typ, rdn, compare between Ps while unfolding, offset_ per direction?
    # unfold segments bottom-up (with decreasing y), select _Ps contiguous with P, compute yd > 0 (always _P_y < P_y)
    # at segment termination, unfold forks, then their segments (recursive)

    new_dert_ = []  # buffer for results
    if typ == 2:

        coefs_ = [] # compute coefs_ for comp_range (if typ == 2)
        for yd, xd in offset_:  # compute a pair of coefficient for each comp direction

            denominator = hypot(yd, xd)  # common denominator for the pair of coefficients
            coefs_.append((yd / denominator, xd / denominator))

    yd_, xd_ = zip(*offset_)  # here, we assume that each offset is in the form of (yd, xd) and yd > 0
    yd_ = [-yd for yd in dy_]  # comps are upward

    for seg in blob.seg_:  # iterate through blob's segments

        for iP, P in seg[2]:  # vertical search

            y = P[-1][0][0]  # y of first dert
            _P__ = []  # list of list of potential comparands' P (there might be more than 1 in forks)
            for yd in yd_:  # iterate through the list of comparands' yds
                _P_ = []  # keep a list of potential comparands' P (there might be more than 1 in forks)
                _iP = iP + yd  # index in Py_ based on vertical coordinate
                find_P(seg, _P_, _iP)  # find all potential comparands' P

            for dert in P[2]:  # horizontal search

                x = dert[0]
                for xd, yd, _P_ in zip(xd_, yd_, _P__):  # iterate through potential comparands
                    _x = x + xd  # horizontal coordinate

                    stop = False  # stop flag
                    _dert = None  # _dert initialization
                    for _P in _P_:  # iterate through potential comparands' Ps
                        if stop:
                            break

                        for _dert in _P[2]:  # iterate through potential comparands' Ps' derts
                            if _x == _dert[1]:  # if dert's coordinates are identical with target coordinates (vertical coordinates are already matched)
                                _y = y + yd  # compute actual vertical coordinate
                                stop = True  # stop
                                break

                    if stop == True:  # if a comparand with the right coordinate is found:

                        comp_typ()

                        if typ == 0:  # comp_angle, will compute angle (dert[5]) if it hasn't been computed

                            vert = yd == -1  # indicate comp direction
                            _i, i, dy, dx = comp_angle(dert, _dert, vert)

                        elif typ == 1:  # comp_deriv, compare g (dert[4])
                            vert = yd == -1  # indicate comp direction
                            _i, i, dy, dx = comp_deriv(dert, _dert, vert)

                        else:  # comp_range, compare i (dert[2])
                            _i, i, dy, dx = comp_range(dert, _dert, coefs_)

                        new_dert_.append((y, x, i, dy, dx))
                        new_dert_.append((_y, _x, _i, dy, dx))

    # combine raw derts back into derts (with ncomp) and compute g:

    new_dert_ = []

    new_dert_.sort(key=lambda dert: dert[1])  # sorted by x coordinates
    new_dert_.sort(key=lambda dert: dert[0])  # sorted by y coordinates

    i = 0
    max_i = len(new_dert_) - 1
    while i < max_i:  # i goes through comp results with identical y, x (summing dy, dx along)

        y, x, i, dy, dx, g = new_dert_[i]  # initialize dert
        ncomp = 1  # number of comps with current dert (at coordinates y, x)
        while i < max_i and y == new_dert_[i + 1][0] and x == new_dert_[i + 1][1]:  # y, x axes' coordinates are identical
            i += 1  # increment i

            # merge derts:
            ncomp += 1  # +1 number of comps
            dy += new_dert_[i][3]  # sum dy
            dx += new_dert_[i][4]  # sum dx

        g = hypot(dy, dx) - ncomp * ave  # compute g with complete dy, dx
        new_dert_.append((y, x, i, (ncomp, dy, dx), g))  # buffer into new_dert_ for folding/clustering functions

        i += 1

    return new_dert_

    # ---------- unfold() end -----------------------------------------------------------------------------------------------


def find_P(seg, _P_, _iP):  # used in unfold() to find all potential comparand Ps (_P) with given vertical coordinate (P index in Py_)

    if _iP > 0:  # if P's coordinate is within segment
        _P_.append(seg[2][_iP])  # buffer P with given index
    else:  # if P's is beyond segment
        for fork in seg[4]:  # look for P in segment's forks. Stop if seg[4] == []  (no fork)
            find_P(fork, _P_, fork[1][0] - _iP)  # fork[1][0] - _iP: Ly - _iP (supposedly index of _P)

    # ---------- find_P() end -----------------------------------------------------------------------------------------------


def form_sub_blobs(blob, comp_branch, add_dert=True):  # redefine blob as branch-specific master blob: local equivalent of frame

    height, width = blob.map.shape

    if add_dert:
        blob.Derts.append(0, 0, 0, 0)  # I, Dy, Dx, G
        # for i, derts in enumerate (blob.derts_):
        # blob.derts_[i] = derts.append((0, 0, 0, 0))  # i, dy, dx, g

    if height < 3 or width < 3:
        return False

    rng = comp_branch(blob)  # also adds a branch-specific dert_ to blob
    rng_inc = bool(rng - 1)  # flag for comp range increase

    if blob.new_dert__[0].mask.all():
        return False
    seg_ = deque()

    for y in range(rng, height - rng):
        P_ = form_P_(y, blob)  # horizontal clustering
        P_ = scan_P_(P_, seg_, blob, rng_inc)  # vertical clustering
        seg_ = form_seg_(P_, blob, rng_inc)  # vertical clustering
    while seg_: form_blob(seg_.popleft(), blob, rng_inc)  # terminate last running segments

    return True  # sub_blob_ != 0

    # ---------- branch_master_blob() end -------------------------------------------------------------------------------


def comp_angle(dert, _dert, vert=0):  # compute and compare angles

    if len(dert) < 6:  # check if angle exists in dert
        ncomp, dy, dx = dert[3]

        a = atan2(dy, dx)  # compute angle of dert

        dert.append(a)  # buffer for comps with other derts
    else:
        a = dert[5]  # assign to a if angle exists in dert

    if len(_dert) < 6:  # check if angle exists in _dert
        ncomp, dy, dx = _dert[3]

        _a = atan2(dy, dx)  # compute angle of _dert

        dert.append(_a)  # buffer for comps with other derts
    else:
        _a = dert[5]  # assign to _a if angle exists in dert

    da = _a - a  # compare angles

    if vert:  # if vertical comp
        day = -da  # vertical comp is upward so day is reversed in sign
        dax = 0
    else:  # if horizontal comp
        day = 0
        dax = da

    return _i, i, day, dax

    # ---------- comp_angle() end -------------------------------------------------------------------------------------------


def comp_deriv(dert, _dert, vert=0):  # compare g of derts

    y, x, i, (ncomp, dy, dx), g = dert[:5]  # first 5 derts, exclude angle if there's one
    _y, _x, _i, (_ncomp, _dy, _dx), _g = _dert[:5]  # first 5 derts, exclude angle if there's one

    dg = _g - g  # compare g

    if vert:  # if vertical comp
        dgy = -dg  # vertical comp is upward so day is reversed in sign
        dgx = 0
    else:  # if horizontal comp
        dgy = 0
        dgx = dg

    return _i, i, dgy, dgx

    # ---------- comp_deriv() end -------------------------------------------------------------------------------------------


def comp_range(dert, _dert, coefs):  # compare i of derts over indicated distance
    # comparison may include more than 2 directions: vertical, horizontal. So coefficients are needed to decompose di
    # into diy, dix

    y, x, i, (ncomp, dy, dx), g = dert[:5]  # first 5 derts, exclude angle if there's one
    _y, _x, _i, (_ncomp, _dy, _dx), _g = _dert[:5]  # first 5 derts, exclude angle if there's one

    di = _i - i  # compare i

    diy = int(di * coefs[0])  # vertical coefficient
    dix = int(di * coefs[1])  # horizontal coefficient

    return _i, i, diy, dix

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


def scan_P_(P_, seg_, master_blob, rng_inc=False):  # detects connections (forks) between Ps and _Ps, to form blob segments
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
        s, blob_params, seg_ = blob
        y0 = 9999999
        x0 = 9999999
        yn = 0
        xn = 0
        for seg in seg_:
            seg.pop()  # remove references of blob
            for P in seg[2]:
                y0 = min(y0, P[2][0][0])
                x0 = min(x0, P[2][0][1])
                yn = max(yn, P[2][0][0] + 1)
                xn = max(xn, P[2][-1][1] + 1)

        derts_ = master_blob.new_dert__[0][y0:yn, x0:xn, :]
        map = np.zeros((yn - y0, xn - x0), dtype=bool)
        for seg in seg_:
            for P in seg[2]:
                for y, x, i, dy, dx, g in P[2]:
                    map[y - y0, x - x0] = True
        map = map[y0:yn, x0:xn]

        master_blob.params[-4:] = [par1 + par2 for par1, par2 in zip(master_blob.params[-4:], blob_params[-4:])]

        master_blob.sub_blob_[-1].append(nt_blob(typ=0, sign=s, Y=Y, X=X, Ly=Ly, L=L,
                                Derts=[(I, Dy, Dx, G)],  # not selective to +sub_blobs as in sub_Derts
                                seg_=seg_,
                                sub_blob_=[],  # top layer, blob derts_ -> sub_blob derts_
                                sub_Derts=[],  # optional sub_blob_ Derts[:] = [(Ly, L, I, Dy, Dx, G)] if len(sub_blob_) > min
                                layer_f=0,     # if 1: sub_Derts = layer_Derts, sub_blob_= [(sub_Derts, derts_)], +=/ eval_layer
                                root_blob = blob,
                                box=(y0, yn, x0, xn),  # boundary box
                                map=map,       # blob boolean map, to compute overlap
                                rng=1,         # for comp_range per blob,  # ncomp=1: for comp_range per dert, not here
                                ))

    # ---------- form_blob() end -------------------------------------------------------------------------------------