from collections import deque
from math import hypot, pi
from cmath import rect, phase

two_pi = 2 * pi  # angle constraint
angle_coef = 256 / pi   # to scale angle into (-128, 128)

# flags:
Flag_angle          = b001
Flag_inc_rng        = b010
Flag_hypot_g        = b100

# ************ FUNCTIONS ************************************************************************************************
# -compare_derts()
# -lateral_comp()
# -vertical_comp()
# -scan_slice()
# -scan_slice_diag()
# -compute_g()
# -hypot_g()
# -compute_a()
# ***********************************************************************************************************************

def comp_dert(P_, buff___, Ave, rng, flags):    # comparison of input param between derts at range = rng
    # dert_buff___ in blob ( dert__ in P_ line ( dert_ in P

    fa = flags & Flag_angle

    if flags & Flag_hypot_g:
        _derts__ = hypot_g(P_)
    else:
        if fa and rng == 1:
            compute_a(P_)  # compute angles within P

        derts__ = lateral_comp(P_, rng, fa)                  # horizontal comparison, returns current line
        _derts__ = vertical_comp(derts__, buff___, rng, fa)  # vertical and diagonal comp, returns last line in buff___
        compute_g(_derts__, Ave, fa)

    return _derts__

    # ---------- compare_derts() end ----------------------------------------------------------------------------------------

def lateral_comp(P_, rng, fa=0):  # horizontal comparison between pixels at distance == rng

    derts__ = []
    max_index = rng - 1           # max_index in dert_buff_

    if fa: cyc = -rng - 2  # cyc and rng are cross-convertible
    else:  cyc = -rng - 1  # [cyc][fa]: index of input dert and feedback Dert_forks

    for P in P_:
        x0 = P[1] + rng          # sub-P recedes by excluding incomplete _derts
        derts_ = P[-1]
        new_derts_ = []
        buff_ = deque(maxlen=rng)  # new dert's buffer each slice

        for derts in derts_:
            i = derts[cyc][-fa]          # if fa, use angle in radian (index 1)
            dy, dx = derts[cyc][-fa]     # draw accumulated dy, dx from specified dert

            if len(buff_) == rng:
                # xd == rng and coordinate is within P vs. gap

                _derts = buff_[max_index]  # rng-spaced dert, or dert at the end of deque with maxlen=rng
                _i = _derts[cyc][-fa]      # $ vs. separate derts_?
                _dy, _dx = _derts[-1]      # accumulated dy, dx from freshly appended dert (in the code below)

                d = i - _i      # lateral comparison
                dx += d         # bilateral accumulation
                _dx += d        # bilateral accumulation

                _derts[-1] = _dy, _dx       # return
                new_derts_.append(_derts)

            buff_.appendleft(derts + [(dy, dx)])     # append new accumulated dy, dx for horizontal comp

        if new_derts_:  # if not empty
            derts__.append((x0, new_derts_))        # new line of P derts_ appended with new_derts_

    return derts__

    # ---------- lateral_comp() end ---------------------------------------------------------------------------------------

def vertical_comp(derts__, buff___, rng, fa):    # vertical and diagonal comparison

    out_derts__ = []     # initialize output
    # first line of derts in last element of buff___ is returned to comp_dert() at len = maxlen(rng)
    yd = 1

    if fa: cyc = -rng - 2  # cyc and rng are cross-convertible
    else:  cyc = -rng - 1  # [cyc][fa]: index of input dert and feedback Dert_forks

    for index, _derts__ in enumerate(buff___):  # iterate through (rng - 1) higher lines
        if yd < rng:  # diagonal comp, if rng == 1, this returns False

            xd = rng - yd
            hyp = hypot(xd, yd)
            y_coef = yd / hyp       # to decompose d into dy, replace with look-up table?
            x_coef = xd / hyp       # to decompose d into dx, replace with look-up table?
            coefs = (y_coef, x_coef)
            shift = -xd

            # upper-left comps:
            _derts__, derts__ = scan_slice_diag(_derts__, derts__, shift, coefs, cyc, fa)

            # upper-right comps:
            # same ops on _derts__ shifted by upper-left comps, shift back for vertical comp?
            _derts__, derts__ = scan_slice_diag(_derts__, derts__, shift, coefs, cyc, fa)

            buff___[index] = _derts__ # return

        else:   # strictly vertical comp, no shift, fixed coef
            out_derts__, derts__ = scan_slice_(_derts__, derts__, cyc, fa)  # _derts__ are converted to out_derts__

        yd += 1
    buff___.appendleft(derts__)  # buffer derts__ into buff___, after vertical_comp to preserve last derts__ in buff___

    return out_derts__   # = _derts__ if len(buff___) == rng else []

    # ---------- vertical_comp() end ----------------------------------------------------------------------------------------

def scan_slice_(_derts__, derts__, cyc, fa=False):     # unit of vertical comp

    _new_derts__ = []
    new_derts__ = []

    i_derts_ = 0     # index of _derts_ for scanning
    _x0, _derts_ = _derts__[i_derts_]
    _xn = _x0 + len(_derts_)
    _exclude = [True] * len(_derts_)    # to exclude incomplete _derts from sub_Ps

    for x0, derts_ in derts__:      # iterate through derts__
        xn = x0 + len(derts_)
        exclude = [True] * len(derts_)  # to exclude incomplete derts from sub_Ps

        while i_derts_ < len(_derts__):

            while i_derts_ < len(_derts__) and _xn <= x0:  # while no overlap
                i_derts_ += 1
                if i_derts_ < len(_derts__):

                    _x0, _derts_ = _derts__[i_derts_]
                    _xn = _x0 + len(_derts_)
                    _exclude = [True] * len(_derts_)  # to exclude incomplete _derts from sub_Ps

            if i_derts_ < len(_derts__) and  _x0 < xn:   # if overlap, compare slice:

                olp_x0 = max(x0, _x0)  # left overlap
                olp_xn = min(xn, _xn)  # right overlap

                start = max(0, olp_x0 - x0)    # indices of slice derts_
                end = min(len(derts_), len(derts_) + olp_xn - xn)

                _start = max(0, olp_x0 - _x0)  # indices of slice _derts_
                _end = min(len(_derts_), len(_derts_) + olp_xn - _xn)

                exclude[start:end] = [False for _ in exclude[start:end]]  # update excluded derts
                _exclude[_start:_end] = [False for _ in _exclude[_start:_end]]

                for _derts, derts in zip(_derts_[_start:_end], derts_[start:end]):

                    i = derts[cyc][-fa]    # input
                    dy, dx = derts[-1]     # derivatives accumulated over current-rng comps

                    _i = _derts[cyc][-fa]  # template
                    _dy, _dx = _derts[-1]  # derivatives accumulated over current-rng comps

                    d = i - _i
                    if fa:              # if i and _i are angular values:
                        d = rect(1, d)  # convert d into complex number: d = dx + dyj (with dx^2 + dy^2 == 1.0)

                    # bilateral accumulation:

                    dy += d
                    _dy += d

                    # return:
                    derts[-1] = dy, dx      # pack dy, dx, ncomp back into derts
                    _derts[-1] = _dy, _dx

            if _xn > xn:  # save _derts_ for next dert
                break

            # derts_s scanning ends, filter out incomplete derts__ (multiple slices):

            _new_derts__ += [(_x0 + start, _derts_[start:end]) for start, end in
                             zip(
                                 [i for i in range(len(_exclude)) if not _exclude[i] and (i == 0 or _exclude[i - 1])],
                                 [i for i in range(len(_exclude)) if not _exclude[i] and (i == len(_exclude) - 1 or _exclude[i + 1])],
                             )
                             if start < end]

            i_derts_ += 1  # next _derts_
            if i_derts_ < len(_derts__):

                _x0, _derts_ = _derts__[i_derts_]
                _xn = _x0 + len(_derts_)
                _exclude = [True] * len(_derts_)  # to filter incomplete _derts

        # derts_s scanning ends, filter out incomplete derts__ (multiple slices):

        new_derts__ += [(x0 + start, derts_[start:end]) for start, end in
                        zip(
                            [i for i in range(len(exclude)) if not exclude[i] and (i == 0 or exclude[i - 1])],
                            [i for i in range(len(exclude)) if not exclude[i] and (i == len(exclude) - 1 or exclude[i + 1])],
                        )
                        if start < end]

    if i_derts_ < len(_derts__):  # derts_s scanning ends, filter out incomplete derts__ (multiple slices):

        _new_derts__ += [(_x0 + start, _derts_[start:end]) for start, end in
                         zip(
                             [i for i in range(len(_exclude)) if not _exclude[i] and (i == 0 or _exclude[i - 1])],
                             [i for i in range(len(_exclude)) if not _exclude[i] and (i == len(_exclude) - 1 or _exclude[i + 1])],
                         )
                         if start < end]

    return _new_derts__, new_derts__

    # ---------- scan_slice_() end ------------------------------------------------------------------------------------------

def scan_slice_diag(_derts__, derts__, shift, coefs, cyc, fa=False):  # unit of diagonal comp

    _new_derts__ = []
    new_derts__ = []

    y_coef, x_coef = coefs  # to decompose d
    i_derts_ = 0            # index of _derts_
    _x0, _derts_ = _derts__[i_derts_]

    _x0 += shift  # for diagonal comparisons only
    _xn = _x0 + len(_derts_)
    _exclude = [True] * len(_derts_)  # to filter incomplete _derts

    for x0, derts_ in derts__:  # iterate through derts__
        xn = x0 + len(derts_)
        exclude = [True] * len(derts_)  # to filter incomplete _derts

        while i_derts_ < len(_derts__):

            while i_derts_ < len(_derts__) and _xn <= x0:  # while no overlap
                i_derts_ += 1
                if i_derts_ < len(_derts__):

                    _x0, _derts_ = _derts__[i_derts_]
                    _x0 += shift    # for diagonal comparisons only
                    _xn = _x0 + len(_derts_)
                    _exclude = [True] * len(_derts_)  # to filter incomplete _derts

            if i_derts_ < len(_derts__) and _x0 < xn:  # if overlap, compare slice:

                olp_x0 = max(x0, _x0)  # left overlap
                olp_xn = min(xn, _xn)  # right overlap

                start = max(0, olp_x0 - x0)  # indices of slice derts_
                end = min(len(derts_), len(derts_) + olp_xn - xn)

                _start = max(0, olp_x0 - _x0)  # indices of slice _derts_
                _end = min(len(_derts_), len(_derts_) + olp_xn - _xn)

                for _derts, derts in zip(_derts_[_start:_end], derts_[start:end]):

                    i = derts[cyc][-fa]     # input
                    dy, dx = derts[-1]       # derivatives accumulated over current-rng comps

                    _i = _derts[cyc][-fa]   # template
                    _dy, _dx = _derts[-1]   # derivatives accumulated over current-rng comps

                    d = i - _i
                    if fa:              # if i and _i are angular values:
                        d = rect(1, d)  # convert d into complex number: d = dx + dyj (with dx^2 + dy^2 == 1.0)

                    # decomposition into vertical and horizontal differences:

                    partial_dy = int(y_coef * d)
                    partial_dx = int(x_coef * d)

                    # bilateral accumulation:

                    dy += partial_dy        # bilateral template accumulation:
                    dx += partial_dx
                    _dy += partial_dy
                    _dx += partial_dx

                    # return:
                    derts[-1] = dy, dx       # return to temporary vs. packed derts_?
                    _derts[-1] = _dy, _dx

            if _xn > xn:  # save _derts_ for next dert
                break
            # derts_s scanning ends, filter out incomplete derts__ (multiple slices):

            _new_derts__ += [(_x0 + start - shift, _derts_[start:end]) for start, end in
                             zip(
                                 [i for i in range(len(_exclude)) if not _exclude[i] and (i == 0 or _exclude[i - 1])],
                                 [i for i in range(len(_exclude)) if
                                  not _exclude[i] and (i == len(_exclude) - 1 or _exclude[i + 1])],
                             )
                             if start < end]

            i_derts_ += 1  # next _derts
            if i_derts_ < len(_derts__):

                _x0, _derts_ = _derts__[i_derts_]
                _x0 += shift  # for diagonal comparisons only
                _xn = _x0 + len(_derts_)
                _exclude = [True] * len(_derts_)  # used for filtering of incomplete _derts

        # derts_s scanning ends, filter out incomplete derts__ (multiple slices):

        new_derts__ += [(x0 + start - shift, derts_[start:end]) for start, end in
                        zip(
                            [i for i in range(len(exclude)) if not exclude[i] and (i == 0 or exclude[i - 1])],
                            [i for i in range(len(exclude)) if not exclude[i] and (i == len(exclude) - 1 or exclude[i + 1])],
                        )
                        if start < end]

    if i_derts_ < len(_derts__):  # derts_s scanning ends, filter out incomplete derts__ (multiple slices):

        _new_derts__ += [(_x0 + start - shift, _derts_[start:end]) for start, end in
                         zip(
                             [i for i in range(len(_exclude)) if not _exclude[i] and (i == 0 or _exclude[i - 1])],
                             [i for i in range(len(_exclude)) if
                              not _exclude[i] and (i == len(_exclude) - 1 or _exclude[i + 1])],
                         )
                         if start < end]

    return _new_derts__, new_derts__

    # ---------- scan_slice_diag() end -------------------------------------------------------------------------------------

def compute_g(derts__, Ave, fa=0):   # compute g from dx, dy

    for x0, derts_ in derts__:
        for derts in derts_:
            dy, dx = derts[-1][:2]

            if not fa:
                g = hypot(dy, dx)
            else:
                ga = hypot(phase(dy), phase(dx))
                if ga > pi: ga = two_pi - ga  # translate ga's scope into [0, pi) (g is unsigned)

                g = int(ga * angle_coef)  # transform to fit in scope [-128, 127)

            derts[-1] = (g,) + derts[-1]    # return

    # ---------- compute_g() end -------------------------------------------------------------------------------------------

def hypot_g(P_):  # compute g with math.hypot(), convert dert into derts
    derts__ = []

    for P in P_:
        x0 = P[1]
        dert_ = P[-1]
        derts_ = [[(p,), (hypot(dy, dx), dy, dx)] for p, _, dy, dx in dert_]
        derts__.append((x0, derts_))

    return derts__

    # ---------- hypot_g() end ----------------------------------------------------------------------------------------------

def compute_a(P_):  # compute angle from last dy, dx

    for P in P_:
        derts_ = P[-1]
        for derts in derts_:

            g, dy, dx, _ = derts[-1]
            a = complex(dx, dy)  # to complex number: a = dx + dyj
            a /= abs(a)  # normalize a so that abs(a) == 1 (hypot() from real and imaginary part of a == 1)
            a_radian = phase(_a)  # angular value of a in radians: a_radian in (-pi, pi)

            derts += [(a,), (a_radian,)]  # return

    # ---------- compute_a() end --------------------------------------------------------------------------------------------