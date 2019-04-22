from collections import deque
from math import hypot

# ************ FUNCTIONS ************************************************************************************************
# -comp_range()
# -lateral_comp()
# -vertical_comp()
# -comp_slice_()
# ***********************************************************************************************************************

def comp_range(P_, buff___):    # comp i at incremented range, dert_buff___ in blob ( dert__ in P_ line ( dert_ in P
    rng = buff___.maxlen

    derts__ = lateral_comp(P_, rng)                     # horizontal comparison (return current line)
    _derts__ = vertical_comp(derts__, buff___, rng)     # vertical and diagonal comparison (return last line in buff___)

    buff___.appendleft(derts__)                         # buffer derts__ into buff___, after vertical_comp to preserve last derts__ in buff___

    return _derts__

    # ---------- comp_range() end -------------------------------------------------------------------------------------------

def lateral_comp(P_, rng):  # horizontal comparison between pixels at distance == rng

    new_derts__ = []
    buff_ = deque(maxlen=rng)       # new dert's buffer
    max_index = rng - 1             # max_index in dert_buff_
    _x0 = 0                         # x0 of previous P

    for P in P_:
        new_derts_ = []
        x0 = P[1]
        derts_ = P[-1]
        for x in range(_x0, x0):    # coordinates in gaps between Ps
            buff_.append(None)      # buffer gap coords as None

        for x, derts in enumerate(derts_, start=x0):
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

            i = 0   # index in upper-left _derts_, for comparison
            j = 0   # index in upper-right _derts_, for comparison

            _lx0, _lderts_ = _derts__[i]    # upper-left _derts_ for comparison
            _lx0 += xd                      # upper-left comparands shifted right to check for overlap
            _lxn = _lx0 + len(_lderts_)

            _rx0, _rderts_ = _derts__[j]    # upper-right _derts_ for comparison
            _rx0 -= xd                      # upper-right comparands shifted left to check for overlap
            _rxn = _rx0 + len(_rderts_)     # next _rx0

            for x0, derts_ in derts__:

                xn = x0 + len(derts_)

                while i < len(_derts__) and _lxn < xn:      # compare upper-left _derts_

                    while i < len(_derts__) and not (x0 < _lxn and _lx0 < xn):  # while not overlap
                        i += 1

                        if i < len(_derts__):
                            _lx0, _lderts_ = _derts__[i]    # upper-left _derts_ for comparison
                            _lx0 += xd                      # upper-left comparands shifted right to check for overlap
                            _lxn = _lx0 + len(_lderts_)

                    if i < len(_derts__):
                        i += 1

                        comp_slice_(_lderts_, derts_, (_lx0, _lxn, x0, xn), (y_coef, x_coef), comparand_index=-rng)

                while j < len(_derts__) and _rxn < xn:      # compare upper-right _derts_, as above

                    while j < len(_derts__) and not (x0 < _rxn and _rx0 < xn):  # while not overlap
                        j += 1

                        if j < len(_derts__):
                            _rx0, _rderts_ = _derts__[j]    # upper-right _derts_ for comparison
                            _rx0 += xd                      # upper-right comparands shifted left to check for overlap
                            _rxn = _rx0 + len(_rderts_)

                    if j < len(_derts__):
                        j += 1

                        comp_slice_(_rderts_, derts_, (_lx0, _lxn, x0, xn), (y_coef, -x_coef), comparand_index=-rng)
        else:   # vertical_comp, as above but without shifting and with constant coef

            i = 0   # index of _derts_

            _x0, _derts_ = _derts__[i]
            _xn = _x0 + len(_derts_)

            for x0, derts_ in derts__:  # iterate through derts__

                xn = x0 + len(derts_)

                while i < len(_derts__) and _xn < xn:   # while not overlap

                    while i < len(_derts__) and not (x0 < _xn and _x0 < xn):  # while not overlap
                        i += 1

                        if i < len(_derts__):
                            _x0, _derts_ = _derts__[i]
                            _xn = _x0 + len(_derts_)

                    if i < len(_derts__):
                        i += 1

                        comp_slice_(_derts_, derts_, (_x0, _xn, x0, xn), (1, 0), comparand_index=-rng)

    if len(buff___) == rng:
        return buff___[-1]
    else:
        return []

    # ---------- vertical_comp() end ----------------------------------------------------------------------------------------

def comp_slice_(_derts_, derts_, packed_coord, coefs, comparand_index=-1):        # utility function for comparing derts

    _x0, _xn, x0, xn = packed_coord     # bounds of _derts_ and derts_
    y_coef, x_coef = coefs              # to decompose d

    olp_x0 = max(x0, _x0)               # left overlap
    olp_xn = min(xn, _xn)               # right overlap

    start = max(0, olp_x0 - x0)                         # indices for dert_ slicing
    end = min(len(derts_), len(derts_) + olp_xn - xn)

    _start = max(0, olp_x0 - _x0)                       # indices for dert_ slicing
    _end = min(len(_derts_), len(_derts_) + olp_xn - _xn)

    for _derts, derts in zip(_derts_[_start:_end], derts_[start:end]):

        i = derts[comparand_index][0]
        ncomp, dy, dx = derts[-1]

        _i = _derts[comparand_index][0]
        _ncomp, _dy, _dx = _derts[-1]

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

    # ---------- comp_slice_() end ------------------------------------------------------------------------------------------