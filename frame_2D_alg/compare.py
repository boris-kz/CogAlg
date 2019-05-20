from collections import deque
from math import hypot, pi
from cmath import rect

# constrains:
two_pi = 2 * pi
angle_coef = 256 / pi   # to scale angle into (-128, 128)

# ************ FUNCTIONS ************************************************************************************************
# -compare()
# -lateral_comp()
# -vertical_comp()
# -scan_slice()
# -scan_slice_diag()
# -compute_g()
# ***********************************************************************************************************************

def compare(P_, buff___, '''index''', Ave, fa=0):      # comp i at incremented range, dert_buff___ in blob ( dert__ in P_ line ( dert_ in P
    rng = buff___.maxlen

    derts__ = lateral_comp(P_, rng, '''index''', fa)                  # horizontal comparison (return current line)
    _derts__ = vertical_comp(derts__, buff___, rng, '''index''', fa)  # vertical and diagonal comparison (return last line in buff___)
    compute_g(_derts__, Ave, fa)

    return _derts__
    # ---------- compare() end ----------------------------------------------------------------------------------------------

def lateral_comp(P_, rng, '''index''', fa=0):     # horizontal comparison between pixels at distance == rng
    # [cyc][alt][typ]
    derts__ = []
    max_index = rng - 1             # max_index in dert_buff_

    for P in P_:
        x0 = P[1] + rng             # sub-P recedes
        derts_ = P[-1]
        new_derts_ = []


        buff_ = deque(maxlen=rng)  # new dert's buffer each slice

        for derts in derts_:
            i = derts['''index'''][fa]          # if fa, use angle in radian (index 1)
            dy, dx = derts['''index''']         # draw accumulated dy, dx from specified dert

            if len(buff_) == rng:
                # xd == rng and coordinate is within P vs. gap

                _derts = buff_[max_index]           # rng-spaced dert, or dert at the end of deque with maxlen=rng
                _i = _derts['''index'''][fa]        # i from specified dert
                _dy, _dx = _derts[-1]               # accumulated dy, dx from freshly appended dert (in the code below)

                d = i - _i      # lateral comparison
                dx += d         # bilateral accumulation
                _dx += d        # bilateral accumulation

                _derts[-1] = _dy, _dx       # return

                new_derts_.append(_derts)

            buff_.appendleft(derts + [(dy, dx)])     # append new accumulated dy, dx for horizontal comp

        if new_derts_:  # if not empty
            derts__.append((x0, new_derts_))        # new line of P derts_ appended with new_derts_

    return derts__

    # ---------- lateral_comp() end -----------------------------------------------------------------------------------------

def vertical_comp(derts__, buff___, '''index''', fa):    # vertical and diagonal comparison

    output = []     # initialize output

    # first line of derts in last element of buff___ is returned to compare() at len = maxlen(rng)
    yd = 1
    for index, _derts__ in enumerate(buff___):  # iterate through (rng - 1) higher lines

        if yd < rng:  # diagonal comp

            xd = rng - yd

            hyp = hypot(xd, yd)
            y_coef = yd / hyp       # to decompose d into dy, replace with look-up table?
            x_coef = xd / hyp       # to decompose d into dx, replace with look-up table?

            # upper-left comparisons:
            _derts__, derts__ = scan_slice_diag(_derts__, derts__, dert_index=('''index'''), shift = -xd, coefs = (y_coef, x_coef))

            # upper-right comparisons:
            _derts__, derts__ = scan_slice_diag(_derts__, derts__, dert_index=('''index'''), shift = xd, coefs = (y_coef, -x_coef))

            # return:
            buff___[index] = _derts__

        else:   # pure vertical comp, assign _derts__ to output
            output, derts__ = scan_slice_(_derts__, derts__, dert_index=('''index''')  # pure vertical: no shift, fixed coef

        yd += 1

    buff___.appendleft(derts__)  # buffer derts__ into buff___, after vertical_comp to preserve last derts__ in buff___

    return output   # = _derts__ if len(buff___) == rng else []

    # ---------- vertical_comp() end ----------------------------------------------------------------------------------------

def scan_slice_(_derts__, derts__, dert_index, fangle=False):     # unit of vertical comp
    '''index''' = dert_index   # two-level index of compared parameter in derts

    _new_derts__ = []
    new_derts__ = []

    i_derts_ = 0     # index of _derts_ for scanning
    _x0, _derts_ = _derts__[i_derts_]
    _xn = _x0 + len(_derts_)
    _flt = [True] * len(_derts_)    # used to filter incomplete _derts

    for x0, derts_ in derts__:      # iterate through derts__
        xn = x0 + len(derts_)
        flt = [True] * len(derts_)  # used for filtering of incomplete derts

        while i_derts_ < len(_derts__):

            while i_derts_ < len(_derts__) and _xn <= x0:  # while no overlap
                i_derts_ += 1
                if i_derts_ < len(_derts__):
                    _x0, _derts_ = _derts__[i_derts_]
                    _xn = _x0 + len(_derts_)
                    _flt = [True] * len(_derts_)    # used for filtering of incomplete _derts

            if i_derts_ < len(_derts__) and  _x0 < xn:   # if overlap, compare slice:

                olp_x0 = max(x0, _x0)   # left overlap
                olp_xn = min(xn, _xn)   # right overlap

                start = max(0, olp_x0 - x0)    # indices of slice derts_
                end = min(len(derts_), len(derts_) + olp_xn - xn)

                _start = max(0, olp_x0 - _x0)  # indices of slice _derts_
                _end = min(len(_derts_), len(_derts_) + olp_xn - _xn)

                flt[start:end] = [False for _ in flt[start:end]]        # update filter
                _flt[_start:_end] = [False for _ in _flt[_start:_end]]

                for _derts, derts in zip(_derts_[_start:_end], derts_[start:end]):

                    i = derts[i_dert][i_param]          # input
                    dy, dx, ncomp = derts[-1]           # derivatives accumulated over current-rng comps

                    _i = _derts[i_dert][i_param]        # template
                    _dy, _dx, _ncomp = _derts[-1]       # derivatives accumulated over current-rng comps

                    d = i - _i

                    if fangle:              # if i and _i are angular values:
                        d = rect(1, d)      # convert d into complex number: d = dx + dyj (with dx^2 + dy^2 == 1.0)

                    # bilateral accumulation:

                    dy += d
                    ncomp += 1
                    _dy += d
                    _ncomp += 1

                    # return:
                    derts[-1] = dy, dx, ncomp           # pack dy, dx, ncomp back into derts
                    _derts[-1] = _dy, _dx, _ncomp

            if _xn > xn:  # save _derts_ for next dert
                break

            # _derts_'s scanning ends, filter out incomplete _dertss (_derts_ might be broken into many slices):
            _new_derts__ += [(_x0 + start, _derts_[start:end]) for start, end in
                             zip(
                                 [i for i in range(len(_flt)) if not _flt[i] and (i == 0 or _flt[i - 1])],
                                 [i for i in range(len(_flt)) if not _flt[i] and (i == len(_flt) - 1 or _flt[i + 1])],
                             )
                             if start < end]

            i_derts_ += 1  # next _derts_

            if i_derts_ < len(_derts__):

                _x0, _derts_ = _derts__[i_derts_]
                _xn = _x0 + len(_derts_)
                _flt = [True] * len(_derts_)    # used for filtering of incomplete _derts

        # derts_'s scanning ends, filter out incomplete dertss (derts_ might be broken into many slices):
        new_derts__ += [(x0 + start, derts_[start:end]) for start, end in
                        zip(
                            [i for i in range(len(flt)) if not flt[i] and (i == 0 or flt[i - 1])],
                            [i for i in range(len(flt)) if not flt[i] and (i == len(flt) - 1 or flt[i + 1])],
                        )
                        if start < end]

    if i_derts_ < len(_derts__):    # last _derts_
        # _derts_'s scanning ends, filter out incomplete _dertss (_derts_ might be broken into many slices):
        _new_derts__ += [(_x0 + start, _derts_[start:end]) for start, end in
                         zip(
                             [i for i in range(len(_flt)) if not _flt[i] and (i == 0 or _flt[i - 1])],
                             [i for i in range(len(_flt)) if not _flt[i] and (i == len(_flt) - 1 or _flt[i + 1])],
                         )
                         if start < end]

    return _new_derts__, new_derts__
    # ---------- scan_slice_() end ------------------------------------------------------------------------------------------

def scan_slice_diag(_derts__, derts__, dert_index, shift, coefs, fangle=False):  # unit of diagonal comp
    '''index''' = dert_index  # two-level index of compared parameter in derts

    _new_derts__ = []
    new_derts__ = []

    y_coef, x_coef = coefs      # to decompose d
    i_dert, i_param = i_index   # two-level index of compared parameter in derts

    i_derts_ = 0                # index of _derts_
    _x0, _derts_ = _derts__[i_derts_]

    _x0 += shift  # for diagonal comparisons only
    _xn = _x0 + len(_derts_)
    _flt = [True] * len(_derts_)  # used for filtering of incomplete _derts

    for x0, derts_ in derts__:  # iterate through derts__
        xn = x0 + len(derts_)
        flt = [True] * len(derts_)      # used for filtering of incomplete derts

        while i_derts_ < len(_derts__):

            while i_derts_ < len(_derts__) and _xn <= x0:  # while no overlap
                i_derts_ += 1
                if i_derts_ < len(_derts__):
                    _x0, _derts_ = _derts__[i_derts_]
                    _x0 += shift  # for diagonal comparisons only
                    _xn = _x0 + len(_derts_)
                    _flt = [True] * len(_derts_)  # used for filtering of incomplete _derts

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
                    if fangle:              # if i and _i are angular values:
                        d = rect(1, d)      # convert d into complex number: d = dx + dyj (with dx^2 + dy^2 == 1.0)
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

            # _derts_'s scanning ends, filter out incomplete _dertss (_derts_ might be broken into many slices):
            _new_derts__ += [(_x0 + start - shift, _derts_[start:end]) for start, end in
                             zip(
                                 [i for i in range(len(_flt)) if not _flt[i] and (i == 0 or _flt[i - 1])],
                                 [i for i in range(len(_flt)) if
                                  not _flt[i] and (i == len(_flt) - 1 or _flt[i + 1])],
                             )
                             if start < end]

            i_derts_ += 1  # next _derts
            if i_derts_ < len(_derts__):
                _x0, _derts_ = _derts__[i_derts_]
                _x0 += shift  # for diagonal comparisons only
                _xn = _x0 + len(_derts_)
                _flt = [True] * len(_derts_)  # used for filtering of incomplete _derts

        # derts_'s scanning ends, filter out incomplete dertss (derts_ might be broken into many slices):
        new_derts__ += [(x0 + start - shift, derts_[start:end]) for start, end in
                        zip(
                            [i for i in range(len(flt)) if not flt[i] and (i == 0 or flt[i - 1])],
                            [i for i in range(len(flt)) if not flt[i] and (i == len(flt) - 1 or flt[i + 1])],
                        )
                        if start < end]

    if i_derts_ < len(_derts__):  # last _derts_
        # _derts_'s scanning ends, filter out incomplete _dertss (_derts_ might be broken into many slices):
        _new_derts__ += [(_x0 + start - shift, _derts_[start:end]) for start, end in
                         zip(
                             [i for i in range(len(_flt)) if not _flt[i] and (i == 0 or _flt[i - 1])],
                             [i for i in range(len(_flt)) if
                              not _flt[i] and (i == len(_flt) - 1 or _flt[i + 1])],
                         )
                         if start < end]

    return _new_derts__, new_derts__
    # ---------- scan_slice_diag() end -------------------------------------------------------------------------------------

def compute_g(derts__, fa=0):   # compute g from dx, dy
    if not fa:
        for x0, derts_ in derts__:
            for derts in derts_:
                dy, dx = derts[-1][:2]
                g = hypot(dy, dx)

                derts[-1] = (g,) + derts[-1]
    else:
        for x0, derts_ in derts__:
            for derts in derts_:
                dy, dx = derts[-1][:2]

                ga = hypot(phase(dy), phase(dx))

                if ga > pi: ga = two_pi - ga        # translate ga's scope into [0, pi) (g is unsigned)

                g = int(ga * angle_coef)    # transform to fit in scope [-128, 127)

                derts[-1] = (g,) + derts[-1]

    # ---------- compute_g() end -------------------------------------------------------------------------------------------