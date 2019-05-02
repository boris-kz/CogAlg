from collections import deque
from math import hypot

# ************ FUNCTIONS ************************************************************************************************
# -comp_range()
# -lateral_comp()
# -vertical_comp()
# -find_and_comp_slice_()
# ***********************************************************************************************************************

def comp_range(P_, buff___, alt):    # comp i at incremented range, dert_buff___ in blob ( dert__ in P_ line ( dert_ in P
    rng = buff___.maxlen

    derts__ = lateral_comp(P_, rng, alt)                     # horizontal comparison (return current line)
    _derts__ = vertical_comp(derts__, buff___, rng, alt)     # vertical and diagonal comparison (return last line in buff___)

    return _derts__

    # ---------- comp_range() end -------------------------------------------------------------------------------------------

def lateral_comp(P_, rng, alt):  # horizontal comparison between pixels at distance == rng
    i_dert, d_dert = comp_indices

    derts__ = []
    buff_ = deque(maxlen=rng)       # new dert's buffer
    max_index = rng - 1             # max_index in dert_buff_
    _x0 = 0                         # x0 of previous P

    for P in P_:
        x0 = P[1]
        derts_ = P[-1]
        new_derts_ = []

        for x in range(_x0, x0):    # coordinates in gaps between Ps
            buff_.append(None)      # buffer gap coords as None

        for derts in derts_:
            i = derts[alt-rng][0]    # draw i out from specified dert
            dy, dx, ncomp = derts[alt][-4:-1]   # draw accumulated dy, dx from specified dert

            if len(buff_) == rng and buff_[max_index] is not None:
                # xd == rng and coordinate is within P vs. gap

                _derts = buff_[max_index]       # rng-spaced dert, or dert at the end of deque with maxlen=rng
                _i = _derts[alt-rng][0]         # i from specified dert
                _dy, _dx, _ncomp = _derts[-1]   # accumulated dy, dx from freshly appended dert (in the code below)

                d = i - _i      # lateral comparison
                dx += d         # bilateral accumulation
                ncomp += 1      # bilateral accumulation
                _dx += d        # bilateral accumulation
                _ncomp += 1     # bilateral accumulation

                _derts[-1] = _dy, _dx, _ncomp   # return

                new_derts_.append(_derts)

            buff_.appendleft(derts + [(dy, dx, ncomp)])     # append new accumulated dy, dx for horizontal comp

        while buff_:
            _derts = buff_.pop()
            if _derts:              # if not gap
                new_derts_.append(_derts)   # return

        derts__.append((x0, derts_))        # new line of P derts_ appended with new_derts_
        _x0 = x0

    return derts__

    # ---------- lateral_comp() end -----------------------------------------------------------------------------------------

def vertical_comp(derts__, buff___, rng, alt):    # vertical and diagonal comparison

    # first line of derts in last element of buff___ is returned to comp_range() at len = maxlen(rng)

    for yd, _derts__ in enumerate(buff___, start=1):  # iterate through (rng - 1) higher lines

        if yd < rng:  # diagonal comp
            xd = rng - yd

            hyp = hypot(xd, yd)
            y_coef = yd / hyp       # to decompose d into dy, replace with look-up table?
            x_coef = xd / hyp       # to decompose d into dx, replace with look-up table?

            # upper-left comparisons:
            scan_slice_diag(_derts__, derts__, i_index=(alt-rng, 0), shift = -xd, coefs = (y_coef, x_coef))

            # upper-right comparisons:
            scan_slice_diag(_derts__, derts__, i_index=(alt-rng, 0), shift = xd, coefs = (y_coef, -x_coef))

        else:
            scan_slice_(_derts__, derts__, i_index=(alt-rng, 0))  # pure vertical: no shift, fixed coef

    if len(buff___) == rng:
        _derts__ = buff___[-1]
    else:
        _derts__ = []

    buff___.appendleft(derts__)  # buffer derts__ into buff___, after vertical_comp to preserve last derts__ in buff___

    return _derts__

    # ---------- vertical_comp() end ----------------------------------------------------------------------------------------

def scan_slice_(_derts__, derts__, i_index, fangle=False):  # unit of vertical comp
    i_dert, i_param = i_index   # two-level index of compared parameter in derts

    i_derts_ = 0     # index of _derts_
    _x0, _derts_ = _derts__[i_derts_]
    _xn = _x0 + len(_derts_)

    for x0, derts_ in derts__:  # iterate through derts__
        xn = x0 + len(derts_)

        while i_derts_ < len(_derts__):

            while i_derts_ < len(_derts__) and _xn <= x0:  # while no overlap
                i_derts_ += 1
                if i_derts_ < len(_derts__):
                    _x0, _derts_ = _derts__[i_derts_]
                    _xn = _x0 + len(_derts_)

            if i_derts_ < len(_derts__) and  _x0 < xn:   # if overlap, compare slice:

                olp_x0 = max(x0, _x0)   # left overlap
                olp_xn = min(xn, _xn)   # right overlap

                start = max(0, olp_x0 - x0)    # indices of slice derts_
                end = min(len(derts_), len(derts_) + olp_xn - xn)

                _start = max(0, olp_x0 - _x0)  # indices of slice _derts_
                _end = min(len(_derts_), len(_derts_) + olp_xn - _xn)

                for _derts, derts in zip(_derts_[_start:_end], derts_[start:end]):

                    i = derts[i_dert][i_param]          # input
                    dy, dx, ncomp = derts[-1]           # derivatives accumulated over current-rng comps

                    _i = _derts[i_dert][i_param]        # template
                    _dy, _dx, _ncomp = _derts[-1]       # derivatives accumulated over current-rng comps

                    d = i - _i
                    if fangle:          # correct angle diff:
                        if d > 127:
                            d -= 255
                        elif d < -127:
                            d += 255

                    # bilateral accumulation:

                    dy += d
                    ncomp += 1
                    _dy += d
                    _ncomp += 1

                    # return:
                    derts[-1] = dy, dx, ncomp       # pack back
                    _derts[-1] = _dy, _dx, _ncomp

            if _xn > xn:  # save _derts_ for next dert
                break

            i_derts_ += 1  # next _derts_
            if i_derts_ < len(_derts__):
                _x0, _derts_ = _derts__[i_derts_]
                _xn = _x0 + len(_derts_)

    # ---------- scan_slice_() end ------------------------------------------------------------------------------------------

def scan_slice_diag(_derts__, derts__, i_index, shift, coefs, fangle=False):  # unit of diagonal comp

    y_coef, x_coef = coefs      # to decompose d
    i_dert, i_param = i_index   # two-level index of compared parameter in derts

    i_derts_ = 0  # index of _derts_
    _x0, _derts_ = _derts__[i_derts_]

    _x0 += shift  # for diagonal comparisons only
    _xn = _x0 + len(_derts_)

    for x0, derts_ in derts__:  # iterate through derts__
        xn = x0 + len(derts_)

        while i_derts_ < len(_derts__):

            while i_derts_ < len(_derts__) and _xn <= x0:  # while no overlap
                i_derts_ += 1
                if i_derts_ < len(_derts__):
                    _x0, _derts_ = _derts__[i_derts_]
                    _x0 += shift  # for diagonal comparisons only
                    _xn = _x0 + len(_derts_)

            if i_derts_ < len(_derts__) and _x0 < xn:  # if overlap, compare slice:

                olp_x0 = max(x0, _x0)  # left overlap
                olp_xn = min(xn, _xn)  # right overlap

                start = max(0, olp_x0 - x0)  # indices of slice derts_
                end = min(len(derts_), len(derts_) + olp_xn - xn)

                _start = max(0, olp_x0 - _x0)  # indices of slice _derts_
                _end = min(len(_derts_), len(_derts_) + olp_xn - _xn)

                for _derts, derts in zip(_derts_[_start:_end], derts_[start:end]):

                    i = derts[i_dert][i_param]      # input
                    dy, dx, ncomp = derts[-1]       # derivatives accumulated over current-rng comps

                    _i = _derts[i_dert][i_param]    # template
                    _dy, _dx, _ncomp = _derts[-1]   # derivatives accumulated over current-rng comps

                    d = i - _i
                    # decomposition into vertical and horizontal differences:

                    partial_dy = int(y_coef * d)
                    partial_dx = int(x_coef * d)

                    # bilateral accumulation:

                    dy += partial_dy
                    dx += partial_dx
                    ncomp += 1
                    _dy += partial_dy
                    _dx += partial_dx
                    _ncomp += 1

                    # return:
                    derts[-1] = dy, dx, ncomp       # pack back
                    _derts[-1] = _dy, _dx, _ncomp

            if _xn > xn:  # save _derts_ for next dert
                break

            i_derts_ += 1  # next _derts
            if i_derts_ < len(_derts__):
                _x0, _derts_ = _derts__[i_derts_]
                _x0 += shift  # for diagonal comparisons only
                _xn = _x0 + len(_derts_)

    # ---------- scan_slice_diag() end -------------------------------------------------------------------------------------