from collections import deque
from math import hypot

def comp_range(P_, buff___):  # compare i param at incremented range

    # P_: current line Ps
    rng = buff___.maxlen  # dert_buff___ per blob ( dert__ per line ( dert_ per P

    derts__ = lateral_comp(P_, rng)                     # horizontal comparison (return current line)
    _derts__ = vertical_comp(derts__, buff___, rng)     # vertical and diagonal comparison (return last line in buff___)

    buff___.appendleft(derts__)                         # buffer derts__ into buff___, do this after vertical_comp to avoid overwriting last derts__ in buff___

    return _derts__


def lateral_comp(P_, rng):  # horizontal comparison between pixels at distance == rng

    new_derts__ = []                 # initialize output
    buff_ = deque(maxlen=rng)  # new dert's buffer
    max_index = rng - 1             # max_index of dert_buff_
    _x0 = 0                         # prior x0, or x0 of previous P

    for P in P_:
        new_derts_ = []
        x0 = P[1]
        derts_ = P[-1]
        for x in range(_x0, x0):    # through invalid coordinates
            buff_.append(None)      # buffer invalid derts as None

        for x, derts in enumerate(derts_, start=x0):
            i = derts[-rng+1][0]        # +1 due to this being done before new dert is appended
            ncomp, dy, dx = derts[-1][-4:-1]
            if len(buff_) == rng and buff_[max_index] != None:  # xd == rng and valid coordinate

                _derts = buff_[max_index]
                _i = _derts[-rng][0]
                _ncomp, _dy, _dx = _derts[-1]

                d = i - _i      # lateral comparison
                ncomp += 1      # bilateral accumulation
                dx += d         # bilateral accumulation
                _ncomp += 1     # bilateral accumulation
                _dx += d        # bilateral accumulation

                _derts[-1] = _ncomp, _dy, _dx   # assign back into _dert

                new_derts_.append(_derts)       # buffer into line derts_

            derts.append((ncomp, dy, dx))       # new dert

            buff_.appendleft(derts)             # buffer for horizontal comp

        while buff_:            # terminate last derts in line
            derts = buff_.pop()
            if derts != None:                       # valid derts
                new_derts_.append(derts)

        new_derts__.append((x0, new_derts_))    # each new_derts_ (span of horizontally continuous derts) is appended into new_derts__
        _x0 = x0

    return new_derts__

def vertical_comp(derts__, buff___, rng):    # vertical and diagonal comparison. Currently under revision

    # at len=maxlen=rng, first line of derts in last element of buff___ is returned to comp_range()

    for yd, _derts__ in enumerate(buff___, start=1):   # iterate through (rng - 1) upper lines

        if yd < rng:     # diagonal comp

            xd = rng - yd

            hyp = hypot(xd, yd)
            y_coef = yd / hyp       # to decompose d into dy. Whole computation could be replaced with a look-up table instead?
            x_coef = xd / hyp       # to decompose d into dx. Whole computation could be replaced with a look-up table instead?

            i = 0   # index of _derts_ containing upper-left derts for comparison
            j = 0   # index of _derts_ containing upper-right derts for comparison

            _lx0, _lderts_ = _derts__[i]    # _derts_ containing upper-left derts for comparison
            _lx0 += xd                      # upper-left comparands are shifted right horizontally for checking for overlap
            _lxn = _lx0 + len(_lderts_)

            _rx0, _rderts_ = _derts__[j]    # _derts_ containing upper-right derts for comparison
            _rx0 -= xd                      # upper-right comparands are shifted left horizontally for checking for overlap
            _rxn = _rx0 + len(_rderts_)

            for x0, derts_ in derts__:      # iterate through derts__

                xn = x0 + len(derts_)

                while i < len(_derts__) and _lxn < xn:      # upper-left comparisons

                    while i < len(_derts__) and not (x0 < _lxn and _lx0 < xn):  # while not overlap
                        i += 1

                        if i < len(_derts__):
                            _lx0, _lderts_ = _derts__[i]    # _derts_ containing upper-left derts for comparison
                            _lx0 += xd                      # upper-left comparands are shifted right horizontally for checking for overlap
                            _lxn = _lx0 + len(_lderts_)

                    if i < len(_derts__):
                        i += 1

                        compare_slices(_lderts_, derts_, (_lx0, _lxn, x0, xn), (y_coef, x_coef), comparand_index=-rng)

                while j < len(_derts__) and _rxn < xn:      # upper-right comparisons, same as above

                    while j < len(_derts__) and not (x0 < _rxn and _rx0 < xn):  # while not overlap
                        j += 1

                        if j < len(_derts__):
                            _rx0, _rderts_ = _derts__[j]    # _derts_ containing upper-right derts for comparison
                            _rx0 += xd                      # upper-right comparands are shifted left horizontally for checking for overlap
                            _rxn = _rx0 + len(_rderts_)

                    if j < len(_derts__):
                        j += 1

                        compare_slices(_rderts_, derts_, (_lx0, _lxn, x0, xn), (y_coef, -x_coef), comparand_index=-rng)
        else:   # pure vertical_comp, same as above except for non-shifting coordinate and constant coef

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

                        compare_slices(_derts_, derts_, (_x0, _xn, x0, xn), (1, 0), comparand_index=-rng)

    if len(buff___) == rng:
        return buff___[-1]
    else:
        return []

def compare_slices(_derts_, derts_, packed_coord, coefs, comparand_index=-1):        # utility function for comparing derts

    _x0, _xn, x0, xn = packed_coord     # bounds of _derts_ and derts_

    y_coef, x_coef = coefs              # for decomposition of d

    olp_x0 = max(x0, _x0)              # overlap
    olp_xn = min(xn, _xn)              # overlap

    start = max(0, olp_x0 - x0)                         # compute indices for dert_ slicing
    end = min(len(derts_), len(derts_) + olp_xn - xn)   # compute indices for dert_ slicing

    _start = max(0, olp_x0 - _x0)                              # compute indices for _dert_ slicing
    _end = min(len(_derts_), len(_derts_) + olp_xn - _xn)     # compute indices for _dert_ slicing

    for _derts, derts in zip(_derts_[_start:_end], derts_[start:end]):

        i = derts[comparand_index][0]
        ncomp, dy, dx = derts[-1]

        _i = _derts[comparand_index][0]
        _ncomp, _dy, _dx = _derts[-1]

        d = i - _i  # difference

        temp_dy = int(y_coef * d)  # decomposition into vertical difference
        temp_dx = int(x_coef * d)  # decomposition into horizontal difference

        ncomp += 1  # bilateral accumulation
        dy += temp_dy  # bilateral accumulation
        dx += temp_dx  # bilateral accumulation
        _ncomp += 1  # bilateral accumulation
        _dy += temp_dy  # bilateral accumulation
        _dx += temp_dx  # bilateral accumulation

        derts[-1] = ncomp, dy, dx  # assign back into dert
        _derts[-1] = _ncomp, _dy, _dx  # assign back into _dert