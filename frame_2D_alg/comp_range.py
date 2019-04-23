from collections import deque
from math import hypot

# ************ FUNCTIONS ************************************************************************************************
# -comp_range()
# -lateral_comp()
# -vertical_comp()
# -find_and_comp_slice_()
# ***********************************************************************************************************************

def comp_range(P_, buff___):    # comp i at incremented range, dert_buff___ in blob ( dert__ in P_ line ( dert_ in P
    rng = buff___.maxlen

    derts__ = lateral_comp(P_, rng)                     # horizontal comparison (return current line)
    _derts__ = vertical_comp(derts__, buff___, rng)     # vertical and diagonal comparison (return last line in buff___)

    return _derts__  # return i indices and derts__

    # ---------- comp_range() end -------------------------------------------------------------------------------------------

def lateral_comp(P_, rng):  # horizontal comparison between pixels at distance == rng

    new_derts__ = []
    buff_ = deque(maxlen=rng)       # new dert's buffer
    max_index = rng - 1             # max_index in dert_buff_
    _x0 = 0                         # x0 of previous P

    for P in P_:
        x0 = P[1]
        derts_ = P[-1]
        for x in range(_x0, x0):    # coordinates in gaps between Ps
            buff_.append(None)      # buffer gap coords as None

        for derts in derts_:
            i = derts[-rng+1][0]        # +1 for future derts.append(new_dert)
            ncomp, dy, dx = derts[-1][-4:-1]
            if len(buff_) == rng and buff_[max_index] != None:  # xd == rng and coordinate is within P, not gaps

                _derts = buff_[max_index]
                _i = _derts[-rng][0]
                _ncomp, _dy, _dx = _derts[-1]

                d = i - _i      # lateral comparison
                ncomp += 1      # bilateral accumulation
                dx += d         # bilateral accumulation
                _ncomp += 1     # bilateral accumulation
                _dx += d        # bilateral accumulation

                _derts[-1] = _ncomp, _dy, _dx   # return
                new_derts_.append(_derts)       # next-line derts_

            derts.append((ncomp, dy, dx))       # new-layer dert
            buff_.appendleft(derts)             # for horizontal comp

        while buff_:            # terminate last derts in line
            derts = buff_.pop()
            if derts != None:                       # derts are within Ps, not gaps
                new_derts_.append(derts)

        new_derts__.append((x0, new_derts_))    # new line of P derts_ appended with new_derts_
        _x0 = x0

    return new_derts__

    # ---------- lateral_comp() end -----------------------------------------------------------------------------------------

def vertical_comp(derts__, buff___, rng):    # vertical and diagonal comparison

    # at len = maxlen(rng), first line of derts in last element of buff___ is returned to comp_range()

    for yd, _derts__ in enumerate(buff___, start=1):    # iterate through (rng - 1) higher lines

        if yd < rng:     # diagonal comp

            xd = rng - yd

            hyp = hypot(xd, yd)
            y_coef = yd / hyp       # to decompose d into dy. Whole computation could be replaced with a look-up table instead?
            x_coef = xd / hyp       # to decompose d into dx. Whole computation could be replaced with a look-up table instead?

            # upper-left comparisons:
            find_and_comp_slice_(_derts__, derts__, comparand_indices=(-rng, 0), shift = -xd, coefs = (y_coef, x_coef))

            # upper-right comparisons:
            find_and_comp_slice_(_derts__, derts__, comparand_indices=(-rng, 0), shift = xd, coefs = (y_coef, -x_coef))

        else:   # vertical_comp, as above but without shifting and with constant coef
            find_and_comp_slice_(_derts__, derts__, comparand_indices = (-rng, 0))

    if len(buff___) == rng:
        _derts__ = buff___[-1]
    else:
        _derts__ = []

    buff___.appendleft(derts__)  # buffer derts__ into buff___, after vertical_comp to preserve last derts__ in buff___

    return _derts__

    # ---------- vertical_comp() end ----------------------------------------------------------------------------------------

def find_and_comp_slice_(_derts__, derts__, comparand_indices, shift = 0, coefs = (1, 0)):        # utility function for comparing derts
    y_coef, x_coef = coefs      # to decompose d
    i1, i2 = comparand_indices

    index = 0   # index of _derts_

    _x0, _derts_ = _derts__[index]

    _x0 += shift    # optional, for diagonal comparisons only
    _xn = _x0 + len(_derts_)

    for x0, derts_ in derts__:  # iterate through derts__

        xn = x0 + len(derts_)

        while index < len(_derts__) and _xn < xn:  # while not overlap

            while index < len(_derts__) and not (x0 < _xn and _x0 < xn):  # while not overlap
                index += 1

                if index < len(_derts__):
                    _x0, _derts_ = _derts__[index]

                    _x0 += shift  # optional, for diagonal comparisons only
                    _xn = _x0 + len(_derts_)

            if index < len(_derts__):
                index += 1

                # compare slices:

                olp_x0 = max(x0, _x0)               # left overlap
                olp_xn = min(xn, _xn)               # right overlap
                start = max(0, olp_x0 - x0)                         # indices for dert_ slicing
                end = min(len(derts_), len(derts_) + olp_xn - xn)
                _start = max(0, olp_x0 - _x0)                       # indices for dert_ slicing
                _end = min(len(_derts_), len(_derts_) + olp_xn - _xn)

                for _derts, derts in zip(_derts_[_start:_end], derts_[start:end]):

                    i = derts[i1][i2]
                    ncomp, dy, dx = derts[-1][-3:]

                    _i = _derts[i1][i2]
                    _ncomp, _dy, _dx = _derts[-1][-3:]

                    d = i - _i  # difference

                    # decomposition into vertical and horizontal difference:
                    temp_dy = int(y_coef * d)
                    temp_dx = int(x_coef * d)

                    # bilateral accumulation:
                    ncomp += 1
                    dy += temp_dy
                    dx += temp_dx
                    _ncomp += 1
                    _dy += temp_dy
                    _dx += temp_dx

                    # return:
                    derts[-1] = ncomp, dy, dx
                    _derts[-1] = _ncomp, _dy, _dx

    # ---------- find_and_comp_slice_() end ---------------------------------------------------------------------------------